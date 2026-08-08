//! LP (Liquidity Provision) simulation loop.
//!
//! Processes historical pool swap events, accrues fees to active LP positions,
//! calls strategy for add/remove/rebalance decisions, and tracks IL + equity.

use chrono::{DateTime, Utc};
use dataloader::{PoolSwapEvent, PoolType};
use portfoliomanager::lp_position::{LpPosition, LpPoolType, LpPortfolioState};
use signal::SignalType;
use config::LpConfig;
use crate::types::LpBacktestResult;
use crate::amm_math::{constant_product, concentrated, stable_swap};

pub struct LpSimulationLoop {
    lp_config: LpConfig,
    portfolio: LpPortfolioState,
    equity_curve: Vec<(i64, f64)>,
    total_swaps: u64,
    fee_earning_swaps: u64,
    next_position_id: u32,
}

impl LpSimulationLoop {
    pub fn new(lp_config: LpConfig) -> Self {
        let portfolio = LpPortfolioState::new(lp_config.initial_deposit_usd);
        Self {
            lp_config,
            portfolio,
            equity_curve: Vec::new(),
            total_swaps: 0,
            fee_earning_swaps: 0,
            next_position_id: 0,
        }
    }

    pub fn run(&mut self, pool_swaps: &[PoolSwapEvent]) -> LpBacktestResult {
        if pool_swaps.is_empty() {
            return LpBacktestResult::default();
        }

        let initial_value = self.portfolio.balance_usd;
        let mut peak_value = initial_value;
        let mut max_drawdown = 0.0_f64;

        for swap in pool_swaps {
            self.total_swaps += 1;

            // 1. Accrue fees for all active positions
            self.accrue_fees(swap);

            // 2. Update position token amounts based on new price
            self.update_positions(swap);

            // 3. Track equity
            let current_value = self.portfolio.total_value_usd(swap.price());
            peak_value = peak_value.max(current_value);
            let dd = if peak_value > 0.0 { (peak_value - current_value) / peak_value } else { 0.0 };
            max_drawdown = max_drawdown.max(dd);

            self.equity_curve.push((swap.timestamp.timestamp_millis(), current_value));
        }

        let final_value = self.equity_curve.last().map(|(_, v)| *v).unwrap_or(initial_value);
        let hodl_value = initial_value; // HODL = just keep the deposit as-is (50/50 in tokens)

        // Calculate time span for APR
        let start_ts = pool_swaps.first().unwrap().timestamp;
        let end_ts = pool_swaps.last().unwrap().timestamp;
        let days = (end_ts - start_ts).num_seconds() as f64 / 86400.0;
        let annualized_apr = if days > 0.0 && initial_value > 0.0 {
            ((final_value - initial_value) / initial_value) * (365.0 / days)
        } else {
            0.0
        };

        // Time in range: ratio of fee-earning swaps to total
        let time_in_range = if self.total_swaps > 0 {
            self.fee_earning_swaps as f64 / self.total_swaps as f64
        } else {
            0.0
        };

        LpBacktestResult {
            total_fees_earned_usd: self.portfolio.total_fees_earned_usd,
            total_impermanent_loss_usd: self.portfolio.total_il_usd,
            total_gas_costs_usd: self.portfolio.total_gas_costs_usd,
            net_pnl_usd: final_value - initial_value,
            annualized_apr,
            time_in_range_pct: time_in_range * 100.0,
            total_swaps_processed: self.total_swaps,
            fee_earning_swaps: self.fee_earning_swaps,
            avg_position_size_usd: if !self.portfolio.positions.is_empty() {
                self.portfolio.positions.iter().map(|p| p.entry_amount_a + p.entry_amount_b).sum::<f64>()
                    / self.portfolio.positions.len() as f64
            } else { 0.0 },
            max_drawdown_pct: max_drawdown * 100.0,
            equity_curve: self.equity_curve.clone(),
            hodl_comparison: hodl_value,
            rebalance_count: self.portfolio.positions.iter().map(|p| p.rebalance_count).sum(),
            positions_opened: self.portfolio.positions.len() as u32,
            positions_closed: self.portfolio.closed_positions.len() as u32,
        }
    }

    /// Open an initial LP position using the deposit from balance.
    pub fn open_initial_position(&mut self, first_swap: &PoolSwapEvent) {
        let deposit = self.portfolio.balance_usd;
        if deposit <= 0.0 {
            return;
        }

        let price = first_swap.price();
        let amount_b = deposit * self.lp_config.deposit_ratio;
        let amount_a = if price > 0.0 { (deposit - amount_b) / price } else { 0.0 };

        let (tick_lower, tick_upper) = match self.lp_config.pool_type {
            config::LpPoolType::ConcentratedLiquidity => {
                let current_tick = first_swap.current_tick;
                let half_width = self.lp_config.range_width_ticks.unwrap_or(1000) as i32 / 2;
                (Some(current_tick - half_width), Some(current_tick + half_width))
            }
            _ => (None, None),
        };

        let sqrt_price = price.sqrt();
        let liquidity = match self.lp_config.pool_type {
            config::LpPoolType::ConcentratedLiquidity => {
                let sqrt_lower = concentrated::tick_to_sqrt_price(tick_lower.unwrap());
                let sqrt_upper = concentrated::tick_to_sqrt_price(tick_upper.unwrap());
                concentrated::liquidity_from_amounts(sqrt_price, sqrt_lower, sqrt_upper, amount_a, amount_b)
            }
            _ => {
                if first_swap.total_liquidity > 0.0 {
                    deposit / first_swap.total_liquidity * first_swap.total_liquidity
                } else {
                    (amount_a * amount_b).sqrt()
                }
            }
        };

        let pos = LpPosition {
            id: format!("lp_{}", self.next_position_id),
            pool_id: self.lp_config.pool_id.clone(),
            pool_type: to_lp_pool_type(self.lp_config.pool_type),
            exchange: self.lp_config.exchange.clone(),
            token_a: self.lp_config.token_a.clone(),
            token_b: self.lp_config.token_b.clone(),
            liquidity_units: liquidity,
            tick_lower,
            tick_upper,
            entry_sqrt_price: sqrt_price,
            entry_price: price,
            entry_amount_a: amount_a,
            entry_amount_b: amount_b,
            current_amount_a: amount_a,
            current_amount_b: amount_b,
            fees_earned_a: 0.0,
            fees_earned_b: 0.0,
            fees_earned_usd: 0.0,
            impermanent_loss_usd: 0.0,
            gas_costs_usd: self.lp_config.gas_cost_per_operation * self.lp_config.native_token_price_usd,
            opened_at: first_swap.timestamp,
            closed_at: None,
            rebalance_count: 0,
            is_active: true,
        };

        self.portfolio.total_gas_costs_usd += pos.gas_costs_usd;
        self.portfolio.balance_usd -= deposit;
        self.next_position_id += 1;
        self.portfolio.positions.push(pos);
    }

    fn accrue_fees(&mut self, swap: &PoolSwapEvent) {
        let mut total_fee_this_swap = 0.0_f64;

        for pos in self.portfolio.positions.iter_mut().filter(|p| p.is_active) {
            let fee = match pos.pool_type {
                LpPoolType::ConcentratedLiquidity => {
                    let tick_lower = pos.tick_lower.unwrap_or(i32::MIN);
                    let tick_upper = pos.tick_upper.unwrap_or(i32::MAX);
                    if !concentrated::is_in_range(swap.current_tick, tick_lower, tick_upper) {
                        continue;
                    }
                    concentrated::fee_share(pos.liquidity_units, swap.total_liquidity, swap.fee_amount)
                }
                LpPoolType::ConstantProduct => {
                    let pool_share = if swap.total_liquidity > 0.0 {
                        pos.liquidity_units / swap.total_liquidity
                    } else {
                        0.0
                    };
                    constant_product::fee_share(pool_share, swap.fee_amount)
                }
                LpPoolType::StableSwap => {
                    let pool_share = if swap.total_liquidity > 0.0 {
                        pos.liquidity_units / swap.total_liquidity
                    } else {
                        0.0
                    };
                    stable_swap::fee_share(pool_share, swap.fee_amount)
                }
            };

            if fee > 0.0 {
                pos.fees_earned_usd += fee;
                total_fee_this_swap += fee;
            }
        }

        if total_fee_this_swap > 0.0 {
            self.fee_earning_swaps += 1;
            self.portfolio.total_fees_earned_usd += total_fee_this_swap;
        }
    }

    fn update_positions(&mut self, swap: &PoolSwapEvent) {
        let current_price = swap.price();
        let sqrt_price = current_price.sqrt();

        for pos in self.portfolio.positions.iter_mut().filter(|p| p.is_active) {
            match pos.pool_type {
                LpPoolType::ConcentratedLiquidity => {
                    let sqrt_lower = concentrated::tick_to_sqrt_price(pos.tick_lower.unwrap_or(0));
                    let sqrt_upper = concentrated::tick_to_sqrt_price(pos.tick_upper.unwrap_or(0));
                    let (a, b) = concentrated::amounts_from_liquidity(
                        pos.liquidity_units, sqrt_price, sqrt_lower, sqrt_upper,
                    );
                    pos.current_amount_a = a;
                    pos.current_amount_b = b;

                    let il = concentrated::impermanent_loss(
                        pos.entry_sqrt_price, sqrt_price, sqrt_lower, sqrt_upper,
                    );
                    let entry_value = pos.entry_amount_a * pos.entry_price + pos.entry_amount_b;
                    pos.impermanent_loss_usd = il * entry_value;
                }
                LpPoolType::ConstantProduct => {
                    let price_ratio = if pos.entry_price > 0.0 { current_price / pos.entry_price } else { 1.0 };
                    let pool_share = if swap.total_liquidity > 0.0 {
                        pos.liquidity_units / swap.total_liquidity
                    } else { 0.0 };
                    let (a, b) = constant_product::token_amounts(swap.reserve_a, swap.reserve_b, pool_share);
                    pos.current_amount_a = a;
                    pos.current_amount_b = b;

                    let il = constant_product::impermanent_loss(price_ratio);
                    let entry_value = pos.entry_amount_a * pos.entry_price + pos.entry_amount_b;
                    pos.impermanent_loss_usd = il * entry_value;
                }
                LpPoolType::StableSwap => {
                    let price_ratio = if pos.entry_price > 0.0 { current_price / pos.entry_price } else { 1.0 };
                    let amp = 100.0; // Default amplification
                    let il = stable_swap::impermanent_loss(price_ratio, amp);
                    let entry_value = pos.entry_amount_a * pos.entry_price + pos.entry_amount_b;
                    pos.impermanent_loss_usd = il * entry_value;
                }
            }
        }

        self.portfolio.total_il_usd = self.portfolio.positions.iter()
            .filter(|p| p.is_active)
            .map(|p| p.impermanent_loss_usd)
            .sum();
    }
}

fn to_lp_pool_type(pt: config::LpPoolType) -> LpPoolType {
    match pt {
        config::LpPoolType::ConstantProduct => LpPoolType::ConstantProduct,
        config::LpPoolType::ConcentratedLiquidity => LpPoolType::ConcentratedLiquidity,
        config::LpPoolType::StableSwap => LpPoolType::StableSwap,
    }
}
