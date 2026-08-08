//! Two-leg paired-position backtest execution for pairs trading /
//! statistical arbitrage.
//!
//! `Strategy::compute_all_signals_multi_venue` already delivers a jointly-
//! computed signal from two symbols' aligned price series (see
//! `strategy::traits::VenueSeries`) — what's been missing is turning that
//! signal into a real, coordinated two-leg execution instead of the ordinary
//! multi-venue backtest path's "compute jointly, execute against one primary
//! venue only" behavior. This module is a dedicated sibling to that path
//! (and to `python_simulation.rs`'s single-instrument tick loop), not a
//! retrofit of either — `OpenPosition`/`CandleSimulator`/`PortfolioState`'s
//! plain fill path all bake in single-symbol assumptions deeply enough
//! (including a hardcoded symbol string in `PortfolioState::apply_fill_executed`)
//! that extending them in place risks destabilizing every existing
//! single-asset strategy. Every entry/exit here is all-or-nothing: if either
//! leg lacks a valid price at a given tick, the whole pair trade is rejected
//! for that tick rather than opening a partial (unhedged) position.
//!
//! Net exposure is checked directly here (dollar value of `leg_a`'s notional
//! minus `leg_b`'s notional, since the two legs are opposite-signed by
//! construction) rather than via `riskmanager::RiskManager::check_correlated_exposure`
//! — that function sums *gross* exposure across positions correlation-linked
//! above a threshold, which is the right check for two positions expected to
//! move together, but the wrong one for a hedged pair where one leg is long
//! and the other short; using it here would flag a well-hedged pair as
//! riskier than an equivalent single directional bet, which is backwards.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::types::{TradeLeg, TradeRecord};
use portfoliomanager::PositionSide;
use strategy::traits::{HedgeRatioMode, PairSpec, VenueSeries};

/// Per-leg fee/slippage assumptions. Mirrors the shape `ExchangeFeeConfig`
/// already carries in `python_simulation.rs`, kept local and minimal here
/// since a pair only ever has two legs.
#[derive(Debug, Clone, Copy)]
pub struct PairLegFeeConfig {
    pub taker_fee: f64,
    pub slippage_bps: f64,
}

/// Configuration for `run_pair_backtest`.
#[derive(Debug, Clone, Copy)]
pub struct PairBacktestConfig {
    pub initial_capital: f64,
    /// Fraction of `initial_capital` used to size leg A's notional at entry;
    /// leg B's quantity is then `hedge_ratio.abs() * qty_a` (opposite side).
    pub position_size_pct: f64,
    /// Rolling window (in bars) used to (a) re-estimate the hedge ratio when
    /// `HedgeRatioMode::Dynamic` is declared, and (b) compute the spread's
    /// rolling z-score recorded on each opened position for diagnostics.
    pub lookback_window: usize,
    /// Reject an entry whose net dollar exposure (`|leg_a notional -
    /// hedge_ratio-implied leg_b notional|`) would exceed this fraction of
    /// `initial_capital`. A well-hedged pair's net exposure should be small;
    /// a large value here usually means the hedge ratio has drifted or the
    /// two legs' prices have diverged sharply since the ratio was estimated.
    pub max_net_exposure_pct: f64,
    /// Leverage applied to both legs' notional sizing (see
    /// `portfoliomanager::margin::size_leveraged_order` -- the capital
    /// actually at risk, `initial_capital * position_size_pct`, does not
    /// change with leverage, only the notional/quantity does). `1.0` (the
    /// default) is an exact no-op: notional sizing and the liquidation check
    /// below both become dead code at this value, byte-identical to this
    /// module's pre-leverage behavior.
    pub leverage: f64,
    /// Flat maintenance-margin ratio used for the liquidation-price
    /// calculation. Matches `config::TradingConfig`'s own default -- this
    /// isn't job-configurable anywhere else in the platform either.
    pub maintenance_margin_ratio: f64,
    /// Adverse-slippage penalty applied to the liquidation exit price.
    /// Matches `config::TradingConfig`'s own default.
    pub liquidation_penalty_bps: f64,
}

impl Default for PairBacktestConfig {
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

/// One leg's open-position state, tracked internally while a pair trade is open.
#[derive(Debug, Clone)]
struct OpenLeg {
    symbol: String,
    exchange: String,
    side: &'static str, // "long" or "short"
    quantity: f64,
    entry_price: f64,
    entry_commission: f64,
    entry_slippage: f64,
}

/// An open two-leg pair position.
#[derive(Debug, Clone)]
struct OpenPairPosition {
    leg_a: OpenLeg,
    leg_b: OpenLeg,
    hedge_ratio: f64,
    entry_time: DateTime<Utc>,
    entry_zscore: Option<f64>,
}

/// Result of a pair backtest run.
#[derive(Debug, Clone)]
pub struct PairBacktestResult {
    pub trades: Vec<TradeRecord>,
    /// Combined (net) equity curve, one entry per tick.
    pub equity_curve: Vec<f64>,
    /// Entries where a signal called for opening a pair but one leg lacked a
    /// valid price at that tick, so the whole entry was rejected.
    pub rejected_missing_data: usize,
    /// Entries rejected because net exposure would exceed
    /// `PairBacktestConfig::max_net_exposure_pct`.
    pub rejected_risk_limit: usize,
    /// Number of times an open position was force-closed by a margin call
    /// (isolated margin, both legs closed atomically -- see the module-level
    /// design note on this near `run_pair_backtest`). Always `0` at
    /// `leverage=1.0`.
    pub liquidations: usize,
}

/// Summary statistics derived from a `PairBacktestResult`. Deliberately a
/// small, honestly-scoped subset (no walk-forward/Monte Carlo escalation,
/// no deflated Sharpe) — mirrors the same scope cut the pre-existing
/// multi-venue path's own doc comment makes for GA support: real, useful
/// numbers now, with the deeper statistical-rigor escalation (the same
/// walk-forward/Monte Carlo machinery `python_validation.rs` already
/// provides for single-instrument runs) left as explicit future work rather
/// than rushed into this pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairSummaryMetrics {
    pub total_pnl: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    /// Simple (non-annualized-unless-`bars_per_year` given) Sharpe ratio
    /// computed from per-tick equity-curve returns.
    pub sharpe_ratio: f64,
}

impl PairBacktestResult {
    /// Compute summary metrics from `self.trades`/`self.equity_curve`.
    /// `bars_per_year` annualizes the Sharpe ratio (e.g. 252 for daily bars,
    /// 252*24 for hourly) — pass `None` to get the plain per-bar Sharpe.
    pub fn summarize(&self, bars_per_year: Option<f64>) -> PairSummaryMetrics {
        let num_trades = self.trades.len();
        let total_pnl: f64 = self.trades.iter().filter_map(|t| t.pnl).sum();

        let wins = self.trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };

        let gross_profit: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p > 0.0).sum();
        let gross_loss: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p < 0.0).sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };

        let max_drawdown = max_drawdown_from_equity(&self.equity_curve);
        let sharpe_ratio = sharpe_from_equity(&self.equity_curve, bars_per_year);

        PairSummaryMetrics { total_pnl, num_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio }
    }
}

/// Maximum peak-to-trough fractional drawdown over an equity curve. `0.0`
/// for an empty or non-positive curve.
pub(crate) fn max_drawdown_from_equity(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &e in equity_curve {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            let dd = (peak - e) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

/// Sharpe ratio (mean / stddev of per-tick simple returns), optionally
/// annualized by `bars_per_year`. `0.0` when there's fewer than 2 equity
/// points or the return series has zero variance.
pub(crate) fn sharpe_from_equity(equity_curve: &[f64], bars_per_year: Option<f64>) -> f64 {
    if equity_curve.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
        .collect();
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() as f64 - 1.0);
    let std = var.sqrt();
    if std <= 0.0 {
        return 0.0;
    }
    let raw = mean / std;
    match bars_per_year {
        Some(n) => raw * n.sqrt(),
        None => raw,
    }
}

/// Apply a simple taker fill: commission on notional, slippage widening the
/// price against the trader (buy pays up, sell/short receives less).
/// Mirrors `python_simulation.rs::execute_taker_fill`'s flat-slippage-bps
/// branch — deliberately not sharing that function directly (it's a private,
/// deeply single-instrument-coupled helper; see this module's doc comment).
pub(crate) fn taker_fill(mid_price: f64, quantity: f64, is_buy: bool, fee: PairLegFeeConfig) -> (f64, f64, f64) {
    let factor = fee.slippage_bps / 10_000.0;
    let fill_price = if is_buy { mid_price * (1.0 + factor) } else { mid_price * (1.0 - factor) };
    let notional = fill_price * quantity;
    let commission = notional * fee.taker_fee;
    let slippage_cost = (fill_price - mid_price).abs() * quantity;
    (fill_price, commission, slippage_cost)
}

fn timestamp_from_millis(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or_else(Utc::now)
}

pub(crate) fn leg_position_side(side: &str) -> PositionSide {
    if side == "long" { PositionSide::Long } else { PositionSide::Short }
}

/// Close an open pair position and produce its `TradeRecord`. Shared by both
/// the ordinary signal-driven close and the margin-call liquidation path
/// (see `run_pair_backtest`'s liquidation check) so the PnL/`TradeLeg`/
/// `TradeRecord` construction logic exists in exactly one place rather than
/// drifting between two hand-copies.
#[allow(clippy::too_many_arguments)]
fn close_pair_position(
    pos: OpenPairPosition,
    exit_fill_a: f64,
    exit_fill_b: f64,
    exit_comm_a: f64,
    exit_comm_b: f64,
    exit_slip_a: f64,
    exit_slip_b: f64,
    exit_time: DateTime<Utc>,
    exit_reason: &str,
    trade_id: usize,
) -> (TradeRecord, f64) {
    let leg_a_pnl = signed_pnl(&pos.leg_a, exit_fill_a) - pos.leg_a.entry_commission - exit_comm_a;
    let leg_b_pnl = signed_pnl(&pos.leg_b, exit_fill_b) - pos.leg_b.entry_commission - exit_comm_b;
    let net_pnl = leg_a_pnl + leg_b_pnl;

    let entry_notional = pos.leg_a.entry_price * pos.leg_a.quantity + pos.leg_b.entry_price * pos.leg_b.quantity;
    let pnl_pct = if entry_notional > 0.0 { net_pnl / entry_notional } else { 0.0 };
    let total_commission = pos.leg_a.entry_commission + pos.leg_b.entry_commission + exit_comm_a + exit_comm_b;
    let total_slippage = pos.leg_a.entry_slippage + pos.leg_b.entry_slippage + exit_slip_a + exit_slip_b;

    let leg_a_record = TradeLeg {
        exchange: pos.leg_a.exchange.clone(),
        symbol: pos.leg_a.symbol.clone(),
        side: pos.leg_a.side.to_string(),
        fill_price: pos.leg_a.entry_price,
        quantity: pos.leg_a.quantity,
        liquidity: "taker".to_string(),
        commission: pos.leg_a.entry_commission,
        slippage_cost: pos.leg_a.entry_slippage,
        timestamp: pos.entry_time,
    };
    let leg_b_record = TradeLeg {
        exchange: pos.leg_b.exchange.clone(),
        symbol: pos.leg_b.symbol.clone(),
        side: pos.leg_b.side.to_string(),
        fill_price: pos.leg_b.entry_price,
        quantity: pos.leg_b.quantity,
        liquidity: "taker".to_string(),
        commission: pos.leg_b.entry_commission,
        slippage_cost: pos.leg_b.entry_slippage,
        timestamp: pos.entry_time,
    };

    let side_label = if pos.leg_a.side == "long" { "long_spread" } else { "short_spread" };

    let record = TradeRecord {
        trade_id,
        side: side_label.to_string(),
        entry_signal_price: pos.leg_a.entry_price,
        entry_fill_price: pos.leg_a.entry_price,
        exit_signal_price: Some(exit_fill_a),
        exit_fill_price: Some(exit_fill_a),
        quantity: pos.leg_a.quantity,
        pnl: Some(net_pnl),
        pnl_pct: Some(pnl_pct),
        commission: total_commission,
        slippage_cost: total_slippage,
        entry_time: pos.entry_time,
        exit_time: Some(exit_time),
        duration_secs: Some((exit_time - pos.entry_time).num_seconds()),
        entry_liquidity: "taker".to_string(),
        exit_liquidity: Some("taker".to_string()),
        entry_reason: format!("pair entry, hedge_ratio={:.4}", pos.hedge_ratio),
        exit_reason: Some(exit_reason.to_string()),
        mae: None,
        mfe: None,
        legs: vec![leg_a_record, leg_b_record],
    };
    (record, net_pnl)
}

/// Run a two-leg paired-position backtest. `venues` must contain exactly the
/// two `(symbol, exchange)` keys named by `pair_spec` (already aligned to
/// common timestamps by the caller, the same contract
/// `compute_all_signals_multi_venue` itself relies on); `signals` is that
/// same joint signal array, one entry per aligned tick: `1` = enter
/// long-spread (long `symbol_a`, short `symbol_b`), `-1` = enter short-spread,
/// `2` = close, `0` = hold.
pub fn run_pair_backtest(
    venues: &HashMap<(String, String), VenueSeries>,
    pair_spec: &PairSpec,
    signals: &[i8],
    fee_a: PairLegFeeConfig,
    fee_b: PairLegFeeConfig,
    config: PairBacktestConfig,
) -> Result<PairBacktestResult, String> {
    let key_a = (pair_spec.symbol_a.clone(), pair_spec.exchange_a.clone());
    let key_b = (pair_spec.symbol_b.clone(), pair_spec.exchange_b.clone());

    let venue_a = venues.get(&key_a).ok_or_else(|| format!("missing venue data for leg A ({}, {})", pair_spec.symbol_a, pair_spec.exchange_a))?;
    let venue_b = venues.get(&key_b).ok_or_else(|| format!("missing venue data for leg B ({}, {})", pair_spec.symbol_b, pair_spec.exchange_b))?;

    let n = signals.len();
    if venue_a.prices.len() < n || venue_b.prices.len() < n || venue_a.timestamps.len() < n {
        return Err("signals array longer than one or both venues' aligned price series".to_string());
    }
    if n == 0 {
        return Err("empty signal array".to_string());
    }

    let fixed_ratio = match pair_spec.hedge_ratio_mode {
        HedgeRatioMode::Fixed(r) => Some(r),
        HedgeRatioMode::Dynamic => None,
    };

    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut equity_curve: Vec<f64> = Vec::with_capacity(n);
    let mut rejected_missing_data = 0usize;
    let mut rejected_risk_limit = 0usize;
    let mut next_trade_id = 0usize;
    let mut liquidations = 0usize;

    let mut open: Option<OpenPairPosition> = None;
    let mut realized_pnl = 0.0;

    for t in 0..n {
        let price_a = venue_a.prices[t];
        let price_b = venue_b.prices[t];
        let prices_valid = price_a.is_finite() && price_a > 0.0 && price_b.is_finite() && price_b > 0.0;

        let signal = signals[t];

        // Margin-call liquidation check, evaluated before this tick's
        // ordinary signal handling -- a margin call is an involuntary,
        // exchange-forced event that can happen regardless of what the
        // strategy's signal says this bar (mirrors `candle_sim.rs`'s exact
        // placement/priority). No-op at `leverage<=1.0` (the outer gate
        // below makes this whole block dead code, matching every other
        // leveraged engine's no-op contract). Any one leg breaching forces
        // an atomic close of the whole pair -- see this module's top-level
        // doc note on why partial/unhedged survivors are never simulated.
        if config.leverage > 1.0 {
            if let Some(pos_ref) = open.as_ref() {
                let (low_a, high_a) = (venue_a.lows[t], venue_a.highs[t]);
                let (low_b, high_b) = (venue_b.lows[t], venue_b.highs[t]);
                let extremes_valid = low_a.is_finite() && high_a.is_finite() && low_b.is_finite() && high_b.is_finite();
                if extremes_valid {
                    let side_a = leg_position_side(pos_ref.leg_a.side);
                    let side_b = leg_position_side(pos_ref.leg_b.side);
                    let extreme_a = if side_a == PositionSide::Long { low_a } else { high_a };
                    let extreme_b = if side_b == PositionSide::Long { low_b } else { high_b };

                    let liq_a = portfoliomanager::margin::is_liquidated(extreme_a, pos_ref.leg_a.entry_price, side_a, config.leverage, config.maintenance_margin_ratio);
                    let liq_b = portfoliomanager::margin::is_liquidated(extreme_b, pos_ref.leg_b.entry_price, side_b, config.leverage, config.maintenance_margin_ratio);

                    if liq_a || liq_b {
                        let pos = open.take().unwrap();
                        let exit_time = timestamp_from_millis(venue_a.timestamps[t]);

                        // Liquidated leg(s) exit at the penalty-adjusted
                        // liquidation price; a surviving leg is unwound at
                        // its ordinary current price (it didn't breach
                        // anything -- it's being closed only because its
                        // hedge partner was forced out).
                        let base_a = if liq_a {
                            let lp = portfoliomanager::margin::liquidation_price(pos.leg_a.entry_price, side_a, config.leverage, config.maintenance_margin_ratio);
                            portfoliomanager::margin::apply_liquidation_penalty(lp, side_a, config.liquidation_penalty_bps)
                        } else {
                            price_a
                        };
                        let base_b = if liq_b {
                            let lp = portfoliomanager::margin::liquidation_price(pos.leg_b.entry_price, side_b, config.leverage, config.maintenance_margin_ratio);
                            portfoliomanager::margin::apply_liquidation_penalty(lp, side_b, config.liquidation_penalty_bps)
                        } else {
                            price_b
                        };
                        // Normal commission still applies on top of the
                        // liquidation-adjusted price (real exchanges charge
                        // a liquidation fee in addition to executing worse
                        // than the trigger price) -- matches `candle_sim.rs`'s
                        // own liquidation handling, which still runs its fee
                        // model on the penalty-adjusted fill.
                        let (exit_fill_a, exit_comm_a, exit_slip_a) = taker_fill(base_a, pos.leg_a.quantity, pos.leg_a.side == "short", fee_a);
                        let (exit_fill_b, exit_comm_b, exit_slip_b) = taker_fill(base_b, pos.leg_b.quantity, pos.leg_b.side == "short", fee_b);

                        let triggered = match (liq_a, liq_b) {
                            (true, true) => format!("{}+{}", pos.leg_a.symbol, pos.leg_b.symbol),
                            (true, false) => pos.leg_a.symbol.clone(),
                            (false, true) => pos.leg_b.symbol.clone(),
                            (false, false) => unreachable!("liq_a || liq_b already checked"),
                        };
                        let exit_reason = format!("margin_call_liquidation ({} leg)", triggered);

                        let (record, net_pnl) = close_pair_position(
                            pos, exit_fill_a, exit_fill_b, exit_comm_a, exit_comm_b, exit_slip_a, exit_slip_b,
                            exit_time, &exit_reason, next_trade_id,
                        );
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
                let hedge_ratio = match fixed_ratio {
                    Some(r) => Some(r),
                    None => estimate_hedge_ratio(venue_a, venue_b, t, config.lookback_window),
                };

                match hedge_ratio {
                    None => rejected_missing_data += 1,
                    Some(hr) => {
                        // Leverage scales notional/quantity only -- the
                        // capital actually at risk (`initial_capital *
                        // position_size_pct`) is unchanged, matching
                        // `portfoliomanager::margin`'s documented identity.
                        // Exact no-op at `leverage=1.0`.
                        let (_, _, qty_a) = portfoliomanager::margin::size_leveraged_order(
                            config.initial_capital, config.position_size_pct, config.leverage, price_a,
                        );
                        let qty_b = hr.abs() * qty_a;

                        // Combined-leg exposure cap (2026-08): qty_a alone is bounded
                        // by size_leveraged_order/position_size_pct, but qty_b is a
                        // hedge-ratio MULTIPLE of qty_a with no independent cap of its
                        // own -- whenever |hedge_ratio| > 1 (routine for pairs whose
                        // legs trade at different price scales), the combined notional
                        // across both legs silently exceeds what position_size_pct
                        // alone would suggest. Confirmed live in production (job
                        // 50ac25dd, ADA-USD/XRP-USD, hedge_ratio=1.83): combined with
                        // GA-evolved pyramiding this produced a single trade losing
                        // 132% of account equity.
                        let (qty_a, qty_b) = portfoliomanager::margin::scale_paired_legs_to_cap(
                            qty_a, price_a, qty_b, price_b, config.initial_capital, config.leverage,
                        );

                        // Net exposure measured as deviation from the spread's
                        // own recent rolling baseline (mean over
                        // `lookback_window`), NOT from zero. A hedge ratio
                        // estimated via a levels regression (price_a ~
                        // hedge_ratio * price_b) leaves a real, nonzero
                        // intercept whenever the two legs trade at different
                        // price scales (e.g. ETH vs SOL, hedge_ratio=8.26 ->
                        // ETH - 8.26*SOL sits around $1,100, ~90% of ETH's own
                        // price) -- that intercept is NOT hedge risk, it's just
                        // where a well-hedged spread naturally sits. Comparing
                        // it to zero rejected essentially every entry for any
                        // pair whose legs differ in price scale, regardless of
                        // hedge quality (confirmed against real 2023 ETH/SOL
                        // daily closes -- a live incident, not a hypothetical).
                        // What actually signals a stale/wrong hedge ratio is
                        // the CURRENT spread moving away from where it's
                        // recently been sitting, which this measures directly.
                        // Deliberately left as a plain notional comparison, not
                        // adjusted back down for leverage: a mis-hedged pair
                        // really does carry proportionally more dollar risk at
                        // higher leverage, so this check naturally (and
                        // correctly) becomes stricter under leverage. Do not
                        // "fix" this to divide out leverage again.
                        let effective_hr = hr.abs();
                        // No baseline during the initial warmup (fewer than
                        // `lookback_window` bars of history) means there's
                        // nothing to measure deviation from -- the gate simply
                        // doesn't fire rather than falling back to the
                        // zero-baseline check this whole fix removes.
                        let exceeds_risk_limit = match rolling_spread_mean(venue_a, venue_b, effective_hr, t, config.lookback_window) {
                            Some(baseline) => {
                                let current_spread = price_a - effective_hr * price_b;
                                let net_exposure = (qty_a * (current_spread - baseline)).abs();
                                net_exposure > config.initial_capital * config.max_net_exposure_pct
                            }
                            None => false,
                        };
                        if exceeds_risk_limit {
                            rejected_risk_limit += 1;
                        } else {
                            let long_spread = signal == 1;
                            let (side_a, side_b) = if long_spread { ("long", "short") } else { ("short", "long") };

                            let (fill_a, comm_a, slip_a) = taker_fill(price_a, qty_a, side_a == "long", fee_a);
                            let (fill_b, comm_b, slip_b) = taker_fill(price_b, qty_b, side_b == "long", fee_b);

                            let entry_zscore = rolling_spread_zscore(venue_a, venue_b, hr, t, config.lookback_window);

                            open = Some(OpenPairPosition {
                                leg_a: OpenLeg {
                                    symbol: pair_spec.symbol_a.clone(),
                                    exchange: pair_spec.exchange_a.clone(),
                                    side: side_a,
                                    quantity: qty_a,
                                    entry_price: fill_a,
                                    entry_commission: comm_a,
                                    entry_slippage: slip_a,
                                },
                                leg_b: OpenLeg {
                                    symbol: pair_spec.symbol_b.clone(),
                                    exchange: pair_spec.exchange_b.clone(),
                                    side: side_b,
                                    quantity: qty_b,
                                    entry_price: fill_b,
                                    entry_commission: comm_b,
                                    entry_slippage: slip_b,
                                },
                                hedge_ratio: hr,
                                entry_time: timestamp_from_millis(venue_a.timestamps[t]),
                                entry_zscore,
                            });
                        }
                    }
                }
            }
        } else if signal == 2 {
            if let Some(pos) = open.take() {
                if prices_valid {
                    let exit_time = timestamp_from_millis(venue_a.timestamps[t]);

                    let (exit_fill_a, exit_comm_a, exit_slip_a) = taker_fill(price_a, pos.leg_a.quantity, pos.leg_a.side == "short", fee_a);
                    let (exit_fill_b, exit_comm_b, exit_slip_b) = taker_fill(price_b, pos.leg_b.quantity, pos.leg_b.side == "short", fee_b);

                    let (record, net_pnl) = close_pair_position(
                        pos, exit_fill_a, exit_fill_b, exit_comm_a, exit_comm_b, exit_slip_a, exit_slip_b,
                        exit_time, "pair close signal", next_trade_id,
                    );
                    realized_pnl += net_pnl;
                    trades.push(record);
                    next_trade_id += 1;
                } else {
                    // Can't close cleanly without valid prices -- keep the
                    // position open rather than closing on garbage data.
                    open = Some(pos);
                }
            }
        }

        // Mark-to-market equity for this tick.
        let unrealized = open.as_ref().map(|pos| {
            if !prices_valid {
                0.0
            } else {
                signed_pnl(&pos.leg_a, price_a) + signed_pnl(&pos.leg_b, price_b)
                    - pos.leg_a.entry_commission - pos.leg_b.entry_commission
            }
        }).unwrap_or(0.0);
        equity_curve.push(config.initial_capital + realized_pnl + unrealized);
    }

    Ok(PairBacktestResult {
        trades,
        equity_curve,
        rejected_missing_data,
        rejected_risk_limit,
        liquidations,
    })
}

/// Mark-to-market P&L for one leg at `current_price`, sign-adjusted for side.
fn signed_pnl(leg: &OpenLeg, current_price: f64) -> f64 {
    let raw = (current_price - leg.entry_price) * leg.quantity;
    if leg.side == "long" { raw } else { -raw }
}

/// Re-estimate the hedge ratio via OLS over the trailing `window` bars up to
/// (and including) tick `t`. Returns `None` when there isn't enough trailing
/// history yet (matching this module's fail-closed convention: reject the
/// entry rather than trade on a degenerate estimate).
fn estimate_hedge_ratio(venue_a: &VenueSeries, venue_b: &VenueSeries, t: usize, window: usize) -> Option<f64> {
    if t + 1 < window {
        return None;
    }
    let start = t + 1 - window;
    let y = &venue_a.prices[start..=t];
    let x = &venue_b.prices[start..=t];
    quant_diagnostics::ols_simple(y, x).map(|r| r.beta)
}

/// Rolling MEAN of the spread `price_a - hedge_ratio * price_b` at tick `t`,
/// over the trailing `window` bars -- the "typical" level the spread has
/// recently sat at, used as the baseline the net-exposure risk gate measures
/// deviation from (see that gate's own comment for why a hedge ratio
/// estimated from a levels regression leaves a real, nonzero intercept that
/// isn't itself risk). `None` when there isn't enough trailing history for
/// the configured window.
fn rolling_spread_mean(venue_a: &VenueSeries, venue_b: &VenueSeries, hedge_ratio: f64, t: usize, window: usize) -> Option<f64> {
    if t + 1 < window {
        return None;
    }
    let start = t + 1 - window;
    let sum: f64 = (start..=t)
        .map(|i| venue_a.prices[i] - hedge_ratio * venue_b.prices[i])
        .sum();
    Some(sum / window as f64)
}

/// Rolling z-score of the spread `price_a - hedge_ratio * price_b` at tick
/// `t`, recorded on an opened position for diagnostic/reporting purposes
/// only (not used to gate execution). `None` when there isn't enough
/// trailing history for the configured window.
fn rolling_spread_zscore(venue_a: &VenueSeries, venue_b: &VenueSeries, hedge_ratio: f64, t: usize, window: usize) -> Option<f64> {
    if t + 1 < window {
        return None;
    }
    let start = t + 1 - window;
    let spread: Vec<f64> = (start..=t)
        .map(|i| venue_a.prices[i] - hedge_ratio * venue_b.prices[i])
        .collect();
    let z = quant_diagnostics::spread_zscore(&spread, window);
    z.last().copied().filter(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use strategy::traits::VenueSeries;

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

    fn default_fee() -> PairLegFeeConfig {
        PairLegFeeConfig { taker_fee: 0.001, slippage_bps: 1.0 }
    }

    fn zero_fee() -> PairLegFeeConfig {
        PairLegFeeConfig { taker_fee: 0.0, slippage_bps: 0.0 }
    }

    fn simple_pair_spec() -> PairSpec {
        PairSpec {
            symbol_a: "AAA".to_string(),
            exchange_a: "test".to_string(),
            symbol_b: "BBB".to_string(),
            exchange_b: "test".to_string(),
            hedge_ratio_mode: HedgeRatioMode::Fixed(1.0),
        }
    }

    #[test]
    fn enter_and_close_produces_one_trade_with_two_legs() {
        let prices_a = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let prices_b = vec![50.0, 50.2, 50.5, 50.8, 51.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        // enter long-spread at t=0, close at t=4, hold in between
        let signals = vec![1i8, 0, 0, 0, 2];

        let result = run_pair_backtest(
            &venues,
            &simple_pair_spec(),
            &signals,
            default_fee(),
            default_fee(),
            PairBacktestConfig::default(),
        ).expect("should run");

        assert_eq!(result.trades.len(), 1);
        let trade = &result.trades[0];
        assert_eq!(trade.legs.len(), 2);
        assert_eq!(trade.side, "long_spread");
        assert_eq!(trade.legs[0].side, "long");
        assert_eq!(trade.legs[1].side, "short");
        assert_eq!(trade.legs[0].symbol, "AAA");
        assert_eq!(trade.legs[1].symbol, "BBB");
        assert!(trade.pnl.is_some());
        assert_eq!(result.rejected_missing_data, 0);
        assert_eq!(result.rejected_risk_limit, 0);
        assert_eq!(result.equity_curve.len(), 5);
    }

    #[test]
    fn short_spread_pnl_has_expected_sign() {
        // symbol A rises relative to B -- a long-spread position should
        // profit; a short-spread position (opposite legs) should lose.
        let prices_a = vec![100.0, 110.0];
        let prices_b = vec![100.0, 100.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let zero_fee = PairLegFeeConfig { taker_fee: 0.0, slippage_bps: 0.0 };
        let config = PairBacktestConfig { max_net_exposure_pct: 1.0, ..PairBacktestConfig::default() };

        let long_spread_signals = vec![1i8, 2];
        let long_result = run_pair_backtest(&venues, &simple_pair_spec(), &long_spread_signals, zero_fee, zero_fee, config).unwrap();
        assert!(long_result.trades[0].pnl.unwrap() > 0.0, "long-spread should profit when A rises relative to B");

        let short_spread_signals = vec![-1i8, 2];
        let short_result = run_pair_backtest(&venues, &simple_pair_spec(), &short_spread_signals, zero_fee, zero_fee, config).unwrap();
        assert!(short_result.trades[0].pnl.unwrap() < 0.0, "short-spread should lose when A rises relative to B");
    }

    #[test]
    fn missing_leg_price_rejects_the_whole_entry() {
        let prices_a = vec![100.0, f64::NAN, 102.0];
        let prices_b = vec![50.0, 50.2, 50.5];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        // signal fires exactly on the tick where leg A's price is NaN
        let signals = vec![0i8, 1, 2];
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, default_fee(), default_fee(), PairBacktestConfig::default()).unwrap();

        assert_eq!(result.trades.len(), 0, "no position should have opened");
        assert_eq!(result.rejected_missing_data, 1);
    }

    #[test]
    fn missing_venue_returns_error() {
        let prices_a = vec![100.0, 101.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        // leg B venue entirely absent
        let signals = vec![1i8, 2];
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, default_fee(), default_fee(), PairBacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn well_hedged_pair_at_a_large_price_scale_gap_is_not_rejected() {
        // Real incident: a hedge ratio estimated from a levels regression
        // between two assets trading at very different price scales (e.g.
        // ETH ~$2000 vs SOL ~$100, hedge_ratio~8) leaves a large but STABLE
        // intercept (price_a - hedge_ratio*price_b sitting around $1,200,
        // nowhere near zero) -- that's not hedge risk, just where a
        // well-hedged spread naturally sits given the two legs' price
        // scales. The old zero-baseline check rejected essentially every
        // such entry regardless of hedge quality; confirmed against real
        // 2023 ETH-USD/SOL-USD daily closes. The gate must not reject
        // entries just because the absolute combination is large, only when
        // it deviates from its own recent history.
        let prices_a = vec![2000.0; 10];
        let prices_b = vec![100.0; 10]; // spread = 2000 - 8.0*100 = 1200.0, constant throughout
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let mut spec = simple_pair_spec();
        spec.hedge_ratio_mode = HedgeRatioMode::Fixed(8.0);
        let config = PairBacktestConfig { lookback_window: 3, max_net_exposure_pct: 0.05, ..PairBacktestConfig::default() };

        let mut signals = vec![0i8; 10];
        signals[5] = 1;
        signals[9] = 2;
        let result = run_pair_backtest(&venues, &spec, &signals, default_fee(), default_fee(), config).unwrap();
        assert_eq!(result.rejected_risk_limit, 0, "a stable (non-deviating) spread must not be rejected regardless of its absolute size");
        assert_eq!(result.trades.len(), 1);
    }

    #[test]
    fn spread_diverging_from_its_recent_baseline_rejects_entry() {
        // The gate's actual purpose (per its own doc comment): catch a hedge
        // ratio that's gone stale -- prices have diverged sharply SINCE it
        // was estimated -- not merely a large absolute combination value
        // (see the test above, which is the case this whole fix corrects).
        let prices_a = vec![100.0, 100.0, 100.0, 100.0, 100.0, 500.0];
        let prices_b = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let mut spec = simple_pair_spec();
        spec.hedge_ratio_mode = HedgeRatioMode::Fixed(1.0);
        let config = PairBacktestConfig { lookback_window: 3, max_net_exposure_pct: 0.05, ..PairBacktestConfig::default() };

        let signals = vec![0i8, 0, 0, 0, 0, 1];
        let result = run_pair_backtest(&venues, &spec, &signals, default_fee(), default_fee(), config).unwrap();
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.rejected_risk_limit, 1);
    }

    #[test]
    fn dynamic_hedge_ratio_needs_enough_lookback_history() {
        let prices_a: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let prices_b: Vec<f64> = (0..10).map(|i| 50.0 + 0.5 * i as f64).collect();
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let mut spec = simple_pair_spec();
        spec.hedge_ratio_mode = HedgeRatioMode::Dynamic;
        // window (60) is far larger than the 10 bars of history available --
        // every entry signal should be rejected for lack of lookback data.
        let config = PairBacktestConfig { lookback_window: 60, ..PairBacktestConfig::default() };

        let signals: Vec<i8> = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 2];
        let result = run_pair_backtest(&venues, &spec, &signals, default_fee(), default_fee(), config).unwrap();
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.rejected_missing_data, 1);
    }

    // -- leverage / margin-call liquidation --------------------------------

    #[test]
    fn leverage_one_is_explicitly_a_noop() {
        let prices_a = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let prices_b = vec![50.0, 50.2, 50.5, 50.8, 51.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let signals = vec![1i8, 0, 0, 0, 2];
        let config = PairBacktestConfig { leverage: 1.0, ..PairBacktestConfig::default() };
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, default_fee(), default_fee(), config).unwrap();

        // Matches the pre-leverage formula exactly: notional = capital*pct, qty = notional/price.
        let expected_qty_a = (config.initial_capital * config.position_size_pct) / 100.0;
        assert!((result.trades[0].quantity - expected_qty_a).abs() < 1e-6);
        assert_eq!(result.liquidations, 0);
    }

    #[test]
    fn leverage_scales_pnl_proportionally_to_notional() {
        let prices_a = vec![100.0, 110.0];
        let prices_b = vec![100.0, 100.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let signals = vec![1i8, 2];
        let config_1x = PairBacktestConfig { max_net_exposure_pct: 1.0, leverage: 1.0, ..PairBacktestConfig::default() };
        let config_4x = PairBacktestConfig { max_net_exposure_pct: 1.0, leverage: 4.0, ..PairBacktestConfig::default() };

        let result_1x = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee(), zero_fee(), config_1x).unwrap();
        let result_4x = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee(), zero_fee(), config_4x).unwrap();

        let pnl_1x = result_1x.trades[0].pnl.unwrap();
        let pnl_4x = result_4x.trades[0].pnl.unwrap();
        assert!((pnl_4x - 4.0 * pnl_1x).abs() < 1e-6, "pnl_1x={} pnl_4x={}", pnl_1x, pnl_4x);
        assert_eq!(result_4x.liquidations, 0);
    }

    #[test]
    fn high_leverage_triggers_liquidation_via_intrabar_low_not_close() {
        // Leg A (long): close stays flat at 100, but the bar's LOW dips to
        // 75 -- at 5x leverage / 0.5% maintenance margin, the liquidation
        // trigger is entry*(1 - 1/5 + 0.005) = 80.5, so only the intrabar
        // low (not the close) crosses it.
        // Legs A and B start at the same price (equal hedge ratio => ~zero
        // net exposure at entry regardless of leverage) so only the
        // liquidation logic itself is under test here.
        let prices_a = vec![100.0, 100.0];
        let lows_a = vec![100.0, 75.0];
        let highs_a = vec![100.0, 100.0];
        let prices_b = vec![100.0, 100.0];
        let lows_b = vec![100.0, 100.0];
        let highs_b = vec![100.0, 100.0];

        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series_with_range(prices_a, lows_a, highs_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series_with_range(prices_b, lows_b, highs_b));

        let signals = vec![1i8, 0];
        let config = PairBacktestConfig { max_net_exposure_pct: 1.0, leverage: 5.0, ..PairBacktestConfig::default() };
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee(), zero_fee(), config).unwrap();

        assert_eq!(result.liquidations, 1, "leg A's intrabar low should have triggered a margin call");
        assert_eq!(result.trades.len(), 1);
        let reason = result.trades[0].exit_reason.as_ref().unwrap();
        assert!(reason.contains("margin_call_liquidation"), "reason={reason}");
        assert!(reason.contains("AAA"), "reason={reason}");
    }

    #[test]
    fn liquidation_never_triggers_at_leverage_one_regardless_of_price() {
        let prices_a = vec![100.0, 100.0];
        let lows_a = vec![100.0, 1.0]; // catastrophic intrabar crash
        let highs_a = vec![100.0, 100.0];
        let prices_b = vec![50.0, 50.0];
        let lows_b = vec![50.0, 50.0];
        let highs_b = vec![50.0, 50.0];

        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series_with_range(prices_a, lows_a, highs_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series_with_range(prices_b, lows_b, highs_b));

        let signals = vec![1i8, 0];
        let config = PairBacktestConfig { max_net_exposure_pct: 1.0, leverage: 1.0, ..PairBacktestConfig::default() };
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee(), zero_fee(), config).unwrap();

        assert_eq!(result.liquidations, 0);
    }

    #[test]
    fn liquidation_force_closes_both_legs_atomically() {
        let prices_a = vec![100.0, 100.0, 100.0];
        let lows_a = vec![100.0, 75.0, 100.0];
        let highs_a = vec![100.0, 100.0, 100.0];
        let prices_b = vec![100.0, 100.0, 100.0];
        let lows_b = vec![100.0, 100.0, 100.0];
        let highs_b = vec![100.0, 100.0, 100.0];

        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series_with_range(prices_a, lows_a, highs_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series_with_range(prices_b, lows_b, highs_b));

        // enter at t=0, liquidation fires at t=1, an explicit close signal at
        // t=2 should be a no-op (nothing left open).
        let signals = vec![1i8, 0, 2];
        let config = PairBacktestConfig { max_net_exposure_pct: 1.0, leverage: 5.0, ..PairBacktestConfig::default() };
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee(), zero_fee(), config).unwrap();

        assert_eq!(result.trades.len(), 1, "only the liquidation should produce a trade -- the later signal==2 has nothing open");
        assert_eq!(result.trades[0].legs.len(), 2);
        assert_eq!(result.liquidations, 1);
    }

    #[test]
    fn max_drawdown_from_equity_finds_the_worst_peak_to_trough_drop() {
        let curve = vec![100.0, 120.0, 90.0, 110.0, 60.0, 80.0];
        // Worst drawdown: peak 120 -> trough 60 = 50%.
        let dd = max_drawdown_from_equity(&curve);
        assert!((dd - 0.5).abs() < 1e-9, "expected 0.5, got {}", dd);
    }

    #[test]
    fn max_drawdown_from_equity_is_zero_for_a_monotonically_rising_curve() {
        let curve = vec![100.0, 110.0, 120.0, 130.0];
        assert_eq!(max_drawdown_from_equity(&curve), 0.0);
    }

    #[test]
    fn sharpe_from_equity_is_positive_for_a_steadily_rising_curve() {
        let curve: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let sharpe = sharpe_from_equity(&curve, None);
        assert!(sharpe > 0.0, "expected positive sharpe, got {}", sharpe);
    }

    #[test]
    fn sharpe_from_equity_annualizes_when_bars_per_year_given() {
        let curve: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let plain = sharpe_from_equity(&curve, None);
        let annualized = sharpe_from_equity(&curve, Some(252.0));
        assert!((annualized - plain * 252f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn sharpe_from_equity_is_zero_for_too_short_or_flat_curve() {
        assert_eq!(sharpe_from_equity(&[100.0], None), 0.0);
        assert_eq!(sharpe_from_equity(&[100.0, 100.0, 100.0], None), 0.0);
    }

    #[test]
    fn summarize_computes_win_rate_and_profit_factor_from_trades() {
        let prices_a = vec![100.0, 110.0, 100.0, 95.0];
        let prices_b = vec![100.0, 100.0, 100.0, 100.0];
        let mut venues = HashMap::new();
        venues.insert(("AAA".to_string(), "test".to_string()), venue_series(prices_a));
        venues.insert(("BBB".to_string(), "test".to_string()), venue_series(prices_b));

        let zero_fee = PairLegFeeConfig { taker_fee: 0.0, slippage_bps: 0.0 };
        let config = PairBacktestConfig { max_net_exposure_pct: 1.0, ..PairBacktestConfig::default() };
        // Winning long-spread trade (t0->t1: A rises), then a losing one (t2->t3: A falls).
        let signals = vec![1i8, 2, 1, 2];
        let result = run_pair_backtest(&venues, &simple_pair_spec(), &signals, zero_fee, zero_fee, config).unwrap();
        assert_eq!(result.trades.len(), 2);

        let summary = result.summarize(None);
        assert_eq!(summary.num_trades, 2);
        assert!((summary.win_rate - 0.5).abs() < 1e-9);
        assert!(summary.profit_factor > 0.0);
    }
}
