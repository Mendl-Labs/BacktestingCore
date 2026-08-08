//! N-leg basket statistical-arbitrage backtest execution -- the
//! generalization of `pair_simulation.rs`'s two-leg engine to 3+ legs,
//! backed by `get_basket_cointegration`'s Johansen procedure rather than
//! `get_pair_cointegration`'s Engle-Granger test.
//!
//! Deliberately a separate module/types from `pair_simulation.rs`, not a
//! generalized replacement of it: the pairs path (`PairSpec`,
//! `run_pair_backtest`) is already fully built end-to-end, including live
//! deployment on the SignalEngine side, and is left completely untouched.
//! This module mirrors its structure closely (same fill/risk/summary
//! conventions) but operates on `Vec<BasketLeg>`/weights instead of a fixed
//! two-symbol shape. See `pair_simulation.rs`'s own module doc for the
//! rest of the shared design rationale (why a dedicated execution path
//! instead of extending the general single-asset tick loop, why net
//! exposure is checked directly rather than via
//! `riskmanager::RiskManager::check_correlated_exposure`).

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::pair_simulation::PairLegFeeConfig;
use crate::types::{TradeLeg, TradeRecord};
use portfoliomanager::PositionSide;
use strategy::traits::{BasketSpec, BasketWeightMode, VenueSeries};

/// Configuration for `run_basket_backtest`. Mirrors `PairBacktestConfig`.
#[derive(Debug, Clone)]
pub struct BasketBacktestConfig {
    pub initial_capital: f64,
    /// Fraction of `initial_capital` used to size leg 0's notional at entry;
    /// leg `i`'s quantity is then `weights[i].abs() * qty_0` (side per the
    /// weight's sign relative to leg 0's).
    pub position_size_pct: f64,
    /// Rolling window (bars) for `Dynamic` weight re-estimation and the
    /// spread z-score recorded on each opened position.
    pub lookback_window: usize,
    /// Reject an entry whose net dollar exposure (signed notional summed
    /// across all legs) would exceed this fraction of `initial_capital`.
    pub max_net_exposure_pct: f64,
    /// Leverage applied to every leg's notional sizing. See
    /// `pair_simulation::PairBacktestConfig::leverage`'s doc for the
    /// no-op-at-1.0 guarantee -- identical here.
    pub leverage: f64,
    /// Flat maintenance-margin ratio for the liquidation-price calculation.
    /// Matches `config::TradingConfig`'s own default.
    pub maintenance_margin_ratio: f64,
    /// Adverse-slippage penalty applied to the liquidation exit price.
    /// Matches `config::TradingConfig`'s own default.
    pub liquidation_penalty_bps: f64,
}

impl Default for BasketBacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size_pct: 0.5,
            lookback_window: 60,
            max_net_exposure_pct: 0.25,
            leverage: 1.0,
            maintenance_margin_ratio: 0.005,
            liquidation_penalty_bps: 50.0,
        }
    }
}

#[derive(Debug, Clone)]
struct OpenLeg {
    symbol: String,
    exchange: String,
    side: &'static str,
    quantity: f64,
    entry_price: f64,
    entry_commission: f64,
    entry_slippage: f64,
}

#[derive(Debug, Clone)]
struct OpenBasketPosition {
    legs: Vec<OpenLeg>,
    weights: Vec<f64>,
    entry_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct BasketBacktestResult {
    pub trades: Vec<TradeRecord>,
    pub equity_curve: Vec<f64>,
    pub rejected_missing_data: usize,
    pub rejected_risk_limit: usize,
    /// Number of times an open position was force-closed by a margin call
    /// (isolated margin; any one leg breaching closes ALL legs atomically).
    /// Always `0` at `leverage=1.0`.
    pub liquidations: usize,
}

/// Summary statistics for a basket backtest -- identical shape/semantics to
/// `pair_simulation::PairSummaryMetrics`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasketSummaryMetrics {
    pub total_pnl: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
}

impl BasketBacktestResult {
    pub fn summarize(&self, bars_per_year: Option<f64>) -> BasketSummaryMetrics {
        let num_trades = self.trades.len();
        let total_pnl: f64 = self.trades.iter().filter_map(|t| t.pnl).sum();
        let wins = self.trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };
        let gross_profit: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p > 0.0).sum();
        let gross_loss: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p < 0.0).sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        let max_drawdown = crate::pair_simulation::max_drawdown_from_equity(&self.equity_curve);
        let sharpe_ratio = crate::pair_simulation::sharpe_from_equity(&self.equity_curve, bars_per_year);
        BasketSummaryMetrics { total_pnl, num_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio }
    }
}

fn timestamp_from_millis(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or_else(Utc::now)
}

/// Close an open basket position and produce its `TradeRecord`. Shared by
/// both the ordinary signal-driven close and the margin-call liquidation
/// path (see `run_basket_backtest`'s liquidation check) so the PnL/
/// `TradeLeg`/`TradeRecord` construction logic exists in exactly one place.
/// `exit_prices[i]` is the base price to run leg `i`'s exit `taker_fill`
/// against -- the tick's ordinary price for a surviving leg, or the
/// penalty-adjusted liquidation price for a leg that triggered the call.
fn close_basket_position(
    pos: OpenBasketPosition,
    exit_prices: &[f64],
    fees: &[PairLegFeeConfig],
    exit_time: DateTime<Utc>,
    exit_reason: &str,
    trade_id: usize,
) -> (TradeRecord, f64) {
    let n_legs = pos.legs.len();
    let mut net_pnl = 0.0;
    let mut total_commission = 0.0;
    let mut total_slippage = 0.0;
    let mut entry_notional = 0.0;
    let mut leg_records = Vec::with_capacity(n_legs);

    for i in 0..n_legs {
        let leg = &pos.legs[i];
        let (exit_fill, exit_comm, exit_slip) = crate::pair_simulation::taker_fill(exit_prices[i], leg.quantity, leg.side == "short", fees[i]);
        let raw = (exit_fill - leg.entry_price) * leg.quantity;
        let leg_pnl = (if leg.side == "long" { raw } else { -raw }) - leg.entry_commission - exit_comm;
        net_pnl += leg_pnl;
        total_commission += leg.entry_commission + exit_comm;
        total_slippage += leg.entry_slippage + exit_slip;
        entry_notional += leg.entry_price * leg.quantity;

        leg_records.push(TradeLeg {
            exchange: leg.exchange.clone(),
            symbol: leg.symbol.clone(),
            side: leg.side.to_string(),
            fill_price: leg.entry_price,
            quantity: leg.quantity,
            liquidity: "taker".to_string(),
            commission: leg.entry_commission,
            slippage_cost: leg.entry_slippage,
            timestamp: pos.entry_time,
        });
    }

    let pnl_pct = if entry_notional > 0.0 { net_pnl / entry_notional } else { 0.0 };
    let side_label = if pos.legs[0].side == "long" { "long_basket" } else { "short_basket" };

    let record = TradeRecord {
        trade_id,
        side: side_label.to_string(),
        entry_signal_price: pos.legs[0].entry_price,
        entry_fill_price: pos.legs[0].entry_price,
        exit_signal_price: Some(exit_prices[0]),
        exit_fill_price: Some(exit_prices[0]),
        quantity: pos.legs[0].quantity,
        pnl: Some(net_pnl),
        pnl_pct: Some(pnl_pct),
        commission: total_commission,
        slippage_cost: total_slippage,
        entry_time: pos.entry_time,
        exit_time: Some(exit_time),
        duration_secs: Some((exit_time - pos.entry_time).num_seconds()),
        entry_liquidity: "taker".to_string(),
        exit_liquidity: Some("taker".to_string()),
        entry_reason: format!("basket entry, weights={:?}", pos.weights),
        exit_reason: Some(exit_reason.to_string()),
        mae: None,
        mfe: None,
        legs: leg_records,
    };
    (record, net_pnl)
}

/// Run an N-leg basket backtest. `venues` must contain every `(symbol,
/// exchange)` key named by `basket_spec.legs`; `signals` is the joint
/// signal array from `compute_all_signals_multi_venue`: `1` = enter
/// long-combination (leg sides follow each weight's sign relative to
/// leg 0), `-1` = enter the opposite, `2` = close, `0` = hold.
pub fn run_basket_backtest(
    venues: &HashMap<(String, String), VenueSeries>,
    basket_spec: &BasketSpec,
    signals: &[i8],
    fees: &[PairLegFeeConfig],
    config: BasketBacktestConfig,
) -> Result<BasketBacktestResult, String> {
    let n_legs = basket_spec.legs.len();
    if n_legs < 3 {
        return Err("basket backtests require at least 3 legs -- use pair_simulation::run_pair_backtest for 2".to_string());
    }
    if fees.len() != n_legs {
        return Err(format!("fees array length ({}) must match the number of legs ({})", fees.len(), n_legs));
    }

    let leg_venues: Result<Vec<&VenueSeries>, String> = basket_spec.legs.iter()
        .map(|leg| {
            venues.get(&(leg.symbol.clone(), leg.exchange.clone()))
                .ok_or_else(|| format!("missing venue data for leg ({}, {})", leg.symbol, leg.exchange))
        })
        .collect();
    let leg_venues = leg_venues?;

    let n = signals.len();
    if n == 0 {
        return Err("empty signal array".to_string());
    }
    if leg_venues.iter().any(|v| v.prices.len() < n || v.timestamps.len() < n) {
        return Err("signals array longer than one or more legs' aligned price series".to_string());
    }

    let fixed_weights = match &basket_spec.weight_mode {
        BasketWeightMode::Fixed(w) if w.len() == n_legs => Some(w.clone()),
        BasketWeightMode::Fixed(_) => return Err("Fixed weight_mode's weights length must match legs length".to_string()),
        BasketWeightMode::Dynamic => None,
    };

    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut equity_curve: Vec<f64> = Vec::with_capacity(n);
    let mut rejected_missing_data = 0usize;
    let mut rejected_risk_limit = 0usize;
    let mut next_trade_id = 0usize;
    let mut liquidations = 0usize;
    let mut open: Option<OpenBasketPosition> = None;
    let mut realized_pnl = 0.0;

    for t in 0..n {
        let prices: Vec<f64> = leg_venues.iter().map(|v| v.prices[t]).collect();
        let prices_valid = prices.iter().all(|p| p.is_finite() && *p > 0.0);
        let signal = signals[t];

        // Margin-call liquidation check, evaluated before this tick's
        // ordinary signal handling -- mirrors `pair_simulation.rs`'s exact
        // placement/priority and atomic-close-of-every-leg reasoning (see
        // that module's doc note). No-op at `leverage<=1.0`.
        if config.leverage > 1.0 {
            if let Some(pos_ref) = open.as_ref() {
                let extremes: Vec<(f64, f64)> = leg_venues.iter().map(|v| (v.lows[t], v.highs[t])).collect();
                let extremes_valid = extremes.iter().all(|(l, h)| l.is_finite() && h.is_finite());
                if extremes_valid {
                    let sides: Vec<PositionSide> = pos_ref.legs.iter().map(|leg| crate::pair_simulation::leg_position_side(leg.side)).collect();
                    let triggered: Vec<bool> = (0..n_legs).map(|i| {
                        let extreme = if sides[i] == PositionSide::Long { extremes[i].0 } else { extremes[i].1 };
                        portfoliomanager::margin::is_liquidated(extreme, pos_ref.legs[i].entry_price, sides[i], config.leverage, config.maintenance_margin_ratio)
                    }).collect();

                    if triggered.iter().any(|&is_triggered| is_triggered) {
                        let pos = open.take().unwrap();
                        let exit_time = timestamp_from_millis(leg_venues[0].timestamps[t]);

                        // Triggered leg(s) exit at their penalty-adjusted
                        // liquidation price; surviving legs unwind at their
                        // ordinary current price (they didn't breach
                        // anything -- see pair_simulation.rs's rationale).
                        let exit_prices: Vec<f64> = (0..n_legs).map(|i| {
                            if triggered[i] {
                                let lp = portfoliomanager::margin::liquidation_price(pos.legs[i].entry_price, sides[i], config.leverage, config.maintenance_margin_ratio);
                                portfoliomanager::margin::apply_liquidation_penalty(lp, sides[i], config.liquidation_penalty_bps)
                            } else {
                                prices[i]
                            }
                        }).collect();

                        let triggered_symbols: Vec<&str> = (0..n_legs).filter(|&i| triggered[i]).map(|i| pos.legs[i].symbol.as_str()).collect();
                        let exit_reason = format!("margin_call_liquidation ({} leg)", triggered_symbols.join("+"));

                        let (record, net_pnl) = close_basket_position(pos, &exit_prices, fees, exit_time, &exit_reason, next_trade_id);
                        realized_pnl += net_pnl;
                        trades.push(record);
                        next_trade_id += 1;
                        liquidations += 1;

                        equity_curve.push(config.initial_capital + realized_pnl);
                        continue;
                    }
                }
            }
        }

        if open.is_none() && (signal == 1 || signal == -1) {
            if !prices_valid {
                rejected_missing_data += 1;
            } else {
                let weights = match &fixed_weights {
                    Some(w) => Some(w.clone()),
                    None => estimate_basket_weights(&leg_venues, t, config.lookback_window),
                };

                match weights {
                    None => rejected_missing_data += 1,
                    Some(w) => {
                        // Leverage scales notional/quantity only -- the
                        // capital actually at risk is unchanged. Exact
                        // no-op at `leverage=1.0`.
                        let (_, _, qty_0) = portfoliomanager::margin::size_leveraged_order(
                            config.initial_capital, config.position_size_pct, config.leverage, prices[0],
                        );
                        let quantities: Vec<f64> = (0..n_legs).map(|i| if i == 0 { qty_0 } else { w[i].abs() * qty_0 }).collect();

                        // Net exposure measured as deviation from the
                        // basket's own recent rolling baseline combination
                        // value, NOT from zero -- see
                        // pair_simulation.rs::rolling_spread_mean's doc
                        // comment for the full rationale (same fix, N-leg
                        // generalization): a weight vector estimated via
                        // cointegration leaves a real, nonzero intercept
                        // whenever the legs trade at different price scales,
                        // and that intercept is not itself hedge risk.
                        // Deliberately left as a plain notional comparison --
                        // see pair_simulation.rs's identical note: risk
                        // naturally (and correctly) scales up with leverage.
                        let current_combo = prices[0] + (1..n_legs).map(|i| w[i] * prices[i]).sum::<f64>();
                        let exceeds_risk_limit = match rolling_basket_combination_mean(&leg_venues, &w, t, config.lookback_window) {
                            Some(baseline) => {
                                let net_exposure = (qty_0 * (current_combo - baseline)).abs();
                                net_exposure > config.initial_capital * config.max_net_exposure_pct
                            }
                            None => false,
                        };
                        if exceeds_risk_limit {
                            rejected_risk_limit += 1;
                        } else {
                            let long_combination = signal == 1;
                            let mut open_legs = Vec::with_capacity(n_legs);
                            for i in 0..n_legs {
                                let weight_sign = if i == 0 { 1.0 } else { w[i].signum() };
                                let is_long = if long_combination { weight_sign > 0.0 } else { weight_sign < 0.0 };
                                let side: &'static str = if is_long { "long" } else { "short" };
                                let (fill_price, commission, slippage) = crate::pair_simulation::taker_fill(prices[i], quantities[i], is_long, fees[i]);
                                open_legs.push(OpenLeg {
                                    symbol: basket_spec.legs[i].symbol.clone(),
                                    exchange: basket_spec.legs[i].exchange.clone(),
                                    side,
                                    quantity: quantities[i],
                                    entry_price: fill_price,
                                    entry_commission: commission,
                                    entry_slippage: slippage,
                                });
                            }
                            open = Some(OpenBasketPosition {
                                legs: open_legs,
                                weights: w,
                                entry_time: timestamp_from_millis(leg_venues[0].timestamps[t]),
                            });
                        }
                    }
                }
            }
        } else if signal == 2 {
            if let Some(pos) = open.take() {
                if prices_valid {
                    let exit_time = timestamp_from_millis(leg_venues[0].timestamps[t]);
                    let (record, net_pnl) = close_basket_position(pos, &prices, fees, exit_time, "basket close signal", next_trade_id);
                    realized_pnl += net_pnl;
                    trades.push(record);
                    next_trade_id += 1;
                } else {
                    open = Some(pos);
                }
            }
        }

        let unrealized = open.as_ref().map(|pos| {
            if !prices_valid {
                0.0
            } else {
                pos.legs.iter().enumerate().map(|(i, leg)| {
                    let raw = (prices[i] - leg.entry_price) * leg.quantity;
                    (if leg.side == "long" { raw } else { -raw }) - leg.entry_commission
                }).sum()
            }
        }).unwrap_or(0.0);
        equity_curve.push(config.initial_capital + realized_pnl + unrealized);
    }

    Ok(BasketBacktestResult { trades, equity_curve, rejected_missing_data, rejected_risk_limit, liquidations })
}

/// Re-estimate basket weights via the Johansen procedure over the trailing
/// `window` bars up to (and including) tick `t`. Returns `None` when
/// there isn't enough trailing history, or the basket doesn't show any
/// cointegrating relation over that window (fail-closed: reject the entry
/// rather than trade on a degenerate/absent relationship).
/// Rolling MEAN of the basket's weighted price combination (`price[0] +
/// sum_i weights[i] * price[i]`) over the trailing `window` bars -- the
/// N-leg generalization of `pair_simulation::rolling_spread_mean`. See that
/// function's doc comment for why this is the baseline the net-exposure risk
/// gate measures deviation from, rather than zero. `None` when there isn't
/// enough trailing history for the configured window.
fn rolling_basket_combination_mean(leg_venues: &[&VenueSeries], weights: &[f64], t: usize, window: usize) -> Option<f64> {
    if t + 1 < window {
        return None;
    }
    let start = t + 1 - window;
    let n_legs = leg_venues.len();
    let sum: f64 = (start..=t)
        .map(|i| {
            leg_venues[0].prices[i]
                + (1..n_legs).map(|k| weights[k] * leg_venues[k].prices[i]).sum::<f64>()
        })
        .sum();
    Some(sum / window as f64)
}

/// `pub` so `program::worker`'s walk-forward window-count call site can run
/// a full-window weight estimate once (not the trailing rolling use above)
/// to build a representative basket spread series for regime detection.
pub fn estimate_basket_weights(leg_venues: &[&VenueSeries], t: usize, window: usize) -> Option<Vec<f64>> {
    if t + 1 < window {
        return None;
    }
    let start = t + 1 - window;
    let series: Vec<Vec<f64>> = leg_venues.iter().map(|v| v.prices[start..=t].to_vec()).collect();
    let result = quant_diagnostics::johansen_test(&series, 1)?;
    if result.cointegration_rank == 0 {
        return None;
    }
    let top_vector = result.cointegrating_vectors.first()?;
    quant_diagnostics::normalize_cointegrating_vector(top_vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use strategy::traits::BasketLeg;

    fn venue_series(prices: Vec<f64>) -> VenueSeries {
        let n = prices.len();
        VenueSeries {
            timestamps: (0..n).map(|i| i as i64 * 60_000).collect(),
            volumes: vec![1.0; n],
            opens: prices.clone(),
            highs: prices.clone(),
            lows: prices.clone(),
            prices,
        }
    }

    /// Like `venue_series`, but with independently-specified lows/highs so
    /// intrabar-extreme-only price action (e.g. a liquidating wick that
    /// never shows up in the close) can be constructed.
    fn venue_series_with_range(prices: Vec<f64>, lows: Vec<f64>, highs: Vec<f64>) -> VenueSeries {
        let n = prices.len();
        VenueSeries {
            timestamps: (0..n).map(|i| i as i64 * 60_000).collect(),
            volumes: vec![1.0; n],
            opens: prices.clone(),
            highs,
            lows,
            prices,
        }
    }

    fn zero_fee() -> PairLegFeeConfig {
        PairLegFeeConfig { taker_fee: 0.0, slippage_bps: 0.0 }
    }

    fn simple_basket_spec() -> BasketSpec {
        BasketSpec {
            legs: vec![
                BasketLeg { symbol: "AAA".to_string(), exchange: "test".to_string() },
                BasketLeg { symbol: "BBB".to_string(), exchange: "test".to_string() },
                BasketLeg { symbol: "CCC".to_string(), exchange: "test".to_string() },
            ],
            weight_mode: BasketWeightMode::Fixed(vec![1.0, -1.0, -0.5]),
        }
    }

    fn build_venues(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> HashMap<(String, String), VenueSeries> {
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(b));
        venues.insert(("CCC".to_string(), "test".to_string()), venue_series(c));
        venues
    }

    #[test]
    fn rejects_baskets_with_fewer_than_three_legs() {
        let spec = BasketSpec {
            legs: vec![
                BasketLeg { symbol: "AAA".to_string(), exchange: "test".to_string() },
                BasketLeg { symbol: "BBB".to_string(), exchange: "test".to_string() },
            ],
            weight_mode: BasketWeightMode::Fixed(vec![1.0, -1.0]),
        };
        let venues = build_venues(vec![1.0, 2.0], vec![1.0, 2.0], vec![]);
        let fees = vec![zero_fee(); 2];
        let result = run_basket_backtest(&venues, &spec, &[1, 2], &fees, BasketBacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn enter_and_close_produces_one_trade_with_three_legs() {
        let a = vec![100.0, 101.0, 102.0, 103.0];
        let b = vec![50.0, 50.2, 50.5, 50.8];
        let c = vec![30.0, 30.1, 30.3, 30.6];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, ..BasketBacktestConfig::default() };

        let signals = vec![1i8, 0, 0, 2];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();

        assert_eq!(result.trades.len(), 1);
        let trade = &result.trades[0];
        assert_eq!(trade.legs.len(), 3);
        assert_eq!(trade.legs[0].symbol, "AAA");
        assert_eq!(trade.legs[1].symbol, "BBB");
        assert_eq!(trade.legs[2].symbol, "CCC");
        // weight[1]=-1 (opposite of leg0), weight[2]=-0.5 (opposite of leg0)
        assert_eq!(trade.legs[0].side, "long");
        assert_eq!(trade.legs[1].side, "short");
        assert_eq!(trade.legs[2].side, "short");
        assert_eq!(result.equity_curve.len(), 4);
    }

    #[test]
    fn mismatched_fees_length_returns_error() {
        let a = vec![100.0, 101.0];
        let b = vec![50.0, 50.2];
        let c = vec![30.0, 30.1];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 2]; // wrong length (should be 3)
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &[1, 2], &fees, BasketBacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn missing_leg_venue_returns_error() {
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(vec![100.0, 101.0]));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(vec![50.0, 50.2]));
        // CCC missing entirely
        let fees = vec![zero_fee(); 3];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &[1, 2], &fees, BasketBacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn missing_price_rejects_the_whole_entry() {
        let a = vec![100.0, f64::NAN, 102.0];
        let b = vec![50.0, 50.2, 50.5];
        let c = vec![30.0, 30.1, 30.3];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, ..BasketBacktestConfig::default() };
        let signals = vec![0i8, 1, 2];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.rejected_missing_data, 1);
    }

    #[test]
    fn dynamic_weights_need_enough_lookback_history() {
        let a: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (0..10).map(|i| 50.0 + 0.5 * i as f64).collect();
        let c: Vec<f64> = (0..10).map(|i| 30.0 + 0.3 * i as f64).collect();
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let mut spec = simple_basket_spec();
        spec.weight_mode = BasketWeightMode::Dynamic;
        let config = BasketBacktestConfig { lookback_window: 60, ..BasketBacktestConfig::default() };
        let signals: Vec<i8> = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 2];
        let result = run_basket_backtest(&venues, &spec, &signals, &fees, config).unwrap();
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.rejected_missing_data, 1);
    }

    // -- net-exposure risk gate (N-leg generalization of pair_simulation's
    // identical fix -- see rolling_basket_combination_mean's doc comment) ---

    #[test]
    fn estimate_basket_weights_full_window_call_returns_none_on_insufficient_data() {
        // The exact call shape `program::worker`'s walk-forward window-count
        // signal now uses (t = n-1, window = n, i.e. the WHOLE series as one
        // "trailing" window) -- with too little data for even a minimal
        // cointegration test, this must degrade to None (spread-regime
        // signal simply unavailable) rather than panic.
        let a = vec![100.0, 101.0];
        let b = vec![50.0, 50.5];
        let venues: Vec<VenueSeries> = vec![venue_series(a), venue_series(b)];
        let refs: Vec<&VenueSeries> = venues.iter().collect();
        let n = refs[0].prices.len();
        assert_eq!(estimate_basket_weights(&refs, n.saturating_sub(1), n + 1), None);
    }

    #[test]
    fn well_hedged_basket_at_a_large_price_scale_gap_is_not_rejected() {
        // A weight vector estimated via cointegration leaves a real, nonzero
        // intercept whenever the legs trade at different price scales -- see
        // pair_simulation.rs's identical test for the full incident writeup.
        // The gate must not reject entries just because the absolute
        // combination is large, only when it deviates from its own recent
        // history.
        let a = vec![2000.0; 10];
        let b = vec![100.0; 10];
        let c = vec![50.0; 10];
        // combo = 2000 - 1.0*100 - 0.5*50 = 1875.0, constant throughout
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let spec = simple_basket_spec();
        let config = BasketBacktestConfig { lookback_window: 3, max_net_exposure_pct: 0.05, ..BasketBacktestConfig::default() };

        let mut signals = vec![0i8; 10];
        signals[5] = 1;
        signals[9] = 2;
        let result = run_basket_backtest(&venues, &spec, &signals, &fees, config).unwrap();
        assert_eq!(result.rejected_risk_limit, 0, "a stable (non-deviating) combination must not be rejected regardless of its absolute size");
        assert_eq!(result.trades.len(), 1);
    }

    #[test]
    fn basket_combination_diverging_from_its_recent_baseline_rejects_entry() {
        // The gate's actual purpose: catch a weight vector that's gone stale
        // -- legs have diverged sharply since it was estimated -- not merely
        // a large absolute combination value (see the test above).
        let a = vec![100.0, 100.0, 100.0, 100.0, 100.0, 500.0];
        let b = vec![100.0; 6];
        let c = vec![100.0; 6];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let spec = simple_basket_spec();
        let config = BasketBacktestConfig { lookback_window: 3, max_net_exposure_pct: 0.05, ..BasketBacktestConfig::default() };

        let signals = vec![0i8, 0, 0, 0, 0, 1];
        let result = run_basket_backtest(&venues, &spec, &signals, &fees, config).unwrap();
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.rejected_risk_limit, 1);
    }

    // -- leverage / margin-call liquidation --------------------------------

    #[test]
    fn leverage_one_is_explicitly_a_noop() {
        let a = vec![100.0, 101.0, 102.0, 103.0];
        let b = vec![50.0, 50.2, 50.5, 50.8];
        let c = vec![30.0, 30.1, 30.3, 30.6];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, leverage: 1.0, ..BasketBacktestConfig::default() };
        // Matches the pre-leverage formula exactly: notional = capital*pct, qty = notional/price.
        let expected_qty_0 = (config.initial_capital * config.position_size_pct) / 100.0;

        let signals = vec![1i8, 0, 0, 2];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();

        assert!((result.trades[0].quantity - expected_qty_0).abs() < 1e-6);
        assert_eq!(result.liquidations, 0);
    }

    #[test]
    fn leverage_scales_pnl_proportionally_to_notional() {
        let a = vec![100.0, 110.0];
        let b = vec![50.0, 50.0];
        let c = vec![30.0, 30.0];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];

        let config_1x = BasketBacktestConfig { max_net_exposure_pct: 1.0, leverage: 1.0, ..BasketBacktestConfig::default() };
        let config_4x = BasketBacktestConfig { max_net_exposure_pct: 1.0, leverage: 4.0, ..BasketBacktestConfig::default() };

        let signals = vec![1i8, 2];
        let result_1x = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config_1x).unwrap();
        let result_4x = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config_4x).unwrap();

        let pnl_1x = result_1x.trades[0].pnl.unwrap();
        let pnl_4x = result_4x.trades[0].pnl.unwrap();
        assert!((pnl_4x - 4.0 * pnl_1x).abs() < 1e-6, "pnl_1x={} pnl_4x={}", pnl_1x, pnl_4x);
        assert_eq!(result_4x.liquidations, 0);
    }

    #[test]
    fn single_leg_liquidation_force_closes_the_whole_basket() {
        // Leg C is short (weight -0.5 relative to leg 0's long). Its close
        // stays flat at 30, but the bar's HIGH spikes to 40 -- at 5x
        // leverage / 0.5% maintenance margin, a short's liquidation trigger
        // is entry*(1 + 1/5 - 0.005) = 35.85, so only the intrabar high
        // (not the close) crosses it. Legs A and B stay calm throughout.
        let a_prices = vec![100.0, 100.0];
        let b_prices = vec![50.0, 50.0];
        let c_prices = vec![30.0, 30.0];
        let c_highs = vec![30.0, 40.0];

        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series_with_range(a_prices.clone(), a_prices.clone(), a_prices));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series_with_range(b_prices.clone(), b_prices.clone(), b_prices));
        venues.insert(("CCC".to_string(), "test".to_string()), venue_series_with_range(c_prices.clone(), c_prices, c_highs));

        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, leverage: 5.0, ..BasketBacktestConfig::default() };
        let signals = vec![1i8, 0];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();

        assert_eq!(result.liquidations, 1, "leg C's intrabar high should have triggered a margin call");
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].legs.len(), 3, "all three legs should close atomically");
        let reason = result.trades[0].exit_reason.as_ref().unwrap();
        assert!(reason.contains("margin_call_liquidation"), "reason={reason}");
        assert!(reason.contains("CCC"), "reason={reason}");
    }

    #[test]
    fn liquidation_never_triggers_at_leverage_one_regardless_of_price() {
        let a_prices = vec![100.0, 100.0];
        let b_prices = vec![50.0, 50.0];
        let c_prices = vec![30.0, 30.0];
        let c_highs = vec![30.0, 1000.0]; // catastrophic intrabar spike

        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series_with_range(a_prices.clone(), a_prices.clone(), a_prices));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series_with_range(b_prices.clone(), b_prices.clone(), b_prices));
        venues.insert(("CCC".to_string(), "test".to_string()), venue_series_with_range(c_prices.clone(), c_prices, c_highs));

        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, leverage: 1.0, ..BasketBacktestConfig::default() };
        let signals = vec![1i8, 0];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();

        assert_eq!(result.liquidations, 0);
    }

    #[test]
    fn summarize_reports_trade_count() {
        let a = vec![100.0, 110.0, 100.0, 95.0];
        let b = vec![50.0, 50.0, 50.0, 50.0];
        let c = vec![30.0, 30.0, 30.0, 30.0];
        let venues = build_venues(a, b, c);
        let fees = vec![zero_fee(); 3];
        let config = BasketBacktestConfig { max_net_exposure_pct: 1.0, ..BasketBacktestConfig::default() };
        let signals = vec![1i8, 2, 1, 2];
        let result = run_basket_backtest(&venues, &simple_basket_spec(), &signals, &fees, config).unwrap();
        let summary = result.summarize(None);
        assert_eq!(summary.num_trades, 2);
    }
}
