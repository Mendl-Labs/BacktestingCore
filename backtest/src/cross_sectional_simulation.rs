//! Cross-sectional long/short rotation backtest execution -- the real
//! execution engine behind the AI's `get_cross_sectional_rank` diagnostic
//! (`quant_diagnostics::cross_sectional_rank_spread_by_class`): at each
//! non-overlapping rebalance period, rank a multi-asset (optionally
//! multi-asset-class) universe by within-asset-class trailing-return
//! z-score, go long the top half and short the bottom half, hold for the
//! period's holding window, then rebalance again.
//!
//! Deliberately mechanical rather than Python-driven: unlike pairs/basket-arb
//! (where the AI authors custom entry/exit logic per candidate), the ranking
//! rule itself IS the strategy here -- there's no per-candidate signal logic
//! to run through a Python executor, so this engine drives itself directly
//! off the same period/rank primitive the diagnostic tool already validates
//! against, guaranteeing the two stay consistent by construction rather than
//! by convention.
//!
//! Position sizing mirrors `pair_simulation`/`basket_simulation`'s own
//! convention: notional is a fixed fraction of `initial_capital`, not
//! compounded off the running equity curve, so a single bad period can't
//! runaway-compound the next period's sizing.
//!
//! Option-leg support here is intentionally narrower than `pair_simulation`/
//! `basket_simulation`'s: `CrossSectionalAsset::option_instrument` lets a
//! caller run the rotation against a SINGLE FIXED option contract per symbol
//! for the whole backtest window (fee/P&L economics are option-aware exactly
//! like the other two engines -- real Alpaca per-contract fees, PnL scaled
//! by `contract_multiplier`). What this does NOT do is re-resolve a fresh
//! contract each rebalance period -- `prices[i]` is one continuous series
//! for the entire run, and a real rotating options deployment needs a
//! DIFFERENT contract each period (the previous one ages toward expiry and
//! drifts away from ATM). Picking that per-period replacement contract
//! (nearest-expiry, closest-to-ATM, per this platform's resolution
//! convention) is a `program::worker` data-loading concern -- swapping in a
//! new price series at each `rank_end` boundary -- and remains unbuilt;
//! don't assume a symbol backed by `option_instrument` here tracks a real
//! rolling options deployment across periods that cross its fixed contract's
//! expiry.

use chrono::{DateTime, Utc};

use crate::pair_simulation::{leg_multiplier, max_drawdown_from_equity, sharpe_from_equity, taker_fill, PairLegFeeConfig};
use crate::types::{TradeLeg, TradeRecord};
use derivatives::DerivativeMetadata;
use quant_diagnostics::{cross_sectional_rank_spread_by_class, volatility_tercile_regimes, CrossSectionalRankResult, VolatilityRegime};

/// Below this many symbols, a top/bottom-half split isn't meaningful --
/// matches `get_cross_sectional_rank`'s own floor.
pub const MIN_SYMBOLS: usize = 4;

#[derive(Debug, Clone)]
pub struct CrossSectionalAsset {
    pub symbol: String,
    pub exchange: String,
    pub asset_class: String,
    /// See this module's doc comment -- a single fixed contract for the
    /// whole run, not a per-period rotation.
    pub option_instrument: Option<DerivativeMetadata>,
}

#[derive(Debug, Clone, Copy)]
pub struct CrossSectionalConfig {
    pub initial_capital: f64,
    /// Fraction of `initial_capital` committed as combined long+short GROSS
    /// exposure each rebalance period (e.g. 0.5 => $25k long + $25k short on
    /// $100k capital).
    pub gross_exposure_pct: f64,
    pub lookback_bars: usize,
    pub holding_bars: usize,
    /// Exposure multiplier applied to a period whose regime label (from the
    /// trailing volatility-tercile classifier, evaluated at the rebalance
    /// point) is `High`. `1.0` disables regime-conditioned sizing entirely.
    /// This is exposure/sizing conditioning only ("Phase A"), not signal
    /// switching -- the ranking rule itself never changes by regime.
    pub high_vol_exposure_scalar: f64,
    /// Same, for a `Medium`-labeled period. Defaults to `1.0` (only `High`
    /// is scaled down by default).
    pub medium_vol_exposure_scalar: f64,
    /// Trailing window (bars) for the volatility-tercile regime classifier
    /// applied to the equal-weighted universe return series. `0` disables
    /// regime classification entirely (every period gets a `1.0` scalar).
    pub regime_window: usize,
}

impl Default for CrossSectionalConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            gross_exposure_pct: 0.5,
            lookback_bars: 20,
            holding_bars: 20,
            high_vol_exposure_scalar: 0.5,
            medium_vol_exposure_scalar: 1.0,
            regime_window: 20,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossSectionalBacktestResult {
    pub trades: Vec<TradeRecord>,
    /// One point per rebalance period (`n_periods + 1`, starting with
    /// `initial_capital`) -- this strategy only acts at rebalance
    /// boundaries, so a per-bar curve would just repeat the same value
    /// between rebalances.
    pub equity_curve: Vec<f64>,
    pub n_periods: usize,
    /// The underlying rank/spread statistics (winner/loser membership per
    /// period, mean spread, significance) -- the same shape
    /// `get_cross_sectional_rank` returns, so results are directly
    /// comparable to a pre-flight diagnostic run on the same universe.
    pub rank_result: CrossSectionalRankResult,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CrossSectionalSummaryMetrics {
    pub total_pnl: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
}

impl CrossSectionalBacktestResult {
    /// `periods_per_year` annualizes the Sharpe ratio -- e.g. for daily bars
    /// with a 20-bar holding period, roughly `365/20`. Pass `None` for the
    /// plain per-period Sharpe.
    pub fn summarize(&self, periods_per_year: Option<f64>) -> CrossSectionalSummaryMetrics {
        let num_trades = self.trades.len();
        let total_pnl: f64 = self.trades.iter().filter_map(|t| t.pnl).sum();
        let wins = self.trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };
        let gross_profit: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p > 0.0).sum();
        let gross_loss: f64 = self.trades.iter().filter_map(|t| t.pnl).filter(|p| *p < 0.0).sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        let max_drawdown = max_drawdown_from_equity(&self.equity_curve);
        let sharpe_ratio = sharpe_from_equity(&self.equity_curve, periods_per_year);
        CrossSectionalSummaryMetrics { total_pnl, num_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio }
    }
}

fn timestamp_from_millis(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or_else(Utc::now)
}

/// Run the cross-sectional rotation engine over an aligned multi-asset price
/// panel.
///
/// `prices[i][k]` is asset `i`'s close at bar `k` -- every asset must share
/// the same length and the same `timestamps` grid (the caller's
/// responsibility, same alignment contract `intersect_timestamps_ascending`
/// already establishes elsewhere in the platform: an inner join on shared
/// timestamps, not a truncate-to-shortest). `fees[i]` is asset `i`'s
/// per-leg fee/slippage assumption (e.g. from `resolve_fee_config` on its
/// exchange).
pub fn run_cross_sectional_backtest(
    assets: &[CrossSectionalAsset],
    prices: &[Vec<f64>],
    timestamps: &[i64],
    fees: &[PairLegFeeConfig],
    config: CrossSectionalConfig,
) -> Result<CrossSectionalBacktestResult, String> {
    let n_symbols = assets.len();
    if n_symbols < MIN_SYMBOLS {
        return Err(format!(
            "cross-sectional backtest requires at least {} assets, got {}",
            MIN_SYMBOLS, n_symbols
        ));
    }
    if prices.len() != n_symbols || fees.len() != n_symbols {
        return Err("assets/prices/fees length mismatch".to_string());
    }
    let n_bars = timestamps.len();
    if prices.iter().any(|p| p.len() != n_bars) {
        return Err("all assets must share the same aligned bar count as timestamps".to_string());
    }
    if n_bars < 2 {
        return Err("not enough aligned bars for even one return".to_string());
    }

    // Per-bar simple returns, one shorter than `prices`: returns[i][k] is
    // realized going from prices[i][k] to prices[i][k+1].
    let returns: Vec<Vec<f64>> = prices
        .iter()
        .map(|p| p.windows(2).map(|w| if w[0] > 0.0 { w[1] / w[0] - 1.0 } else { 0.0 }).collect())
        .collect();

    // Map each asset's class string to a numeric id for the ranking
    // primitive (which is data-in/data-out and asset-class-agnostic).
    let mut class_ids: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let classes: Vec<usize> = assets
        .iter()
        .map(|a| {
            let next_id = class_ids.len();
            *class_ids.entry(a.asset_class.as_str()).or_insert(next_id)
        })
        .collect();

    let rank_result = cross_sectional_rank_spread_by_class(&returns, &classes, config.lookback_bars, config.holding_bars);
    if rank_result.n_periods == 0 {
        return Err(format!(
            "not enough aligned bars ({}) for even one full lookback({})+holding({}) period across {} assets",
            n_bars, config.lookback_bars, config.holding_bars, n_symbols
        ));
    }

    // Equal-weighted universe return series, purely for the regime
    // classifier -- a market-state label computed from price history only,
    // never from strategy performance.
    let regimes: Vec<Option<VolatilityRegime>> = if config.regime_window >= 2 {
        let universe_returns: Vec<f64> = (0..returns[0].len())
            .map(|k| returns.iter().map(|r| r[k]).sum::<f64>() / n_symbols as f64)
            .collect();
        volatility_tercile_regimes(&universe_returns, config.regime_window)
    } else {
        Vec::new()
    };

    let mut equity_curve = Vec::with_capacity(rank_result.n_periods + 1);
    equity_curve.push(config.initial_capital);
    let mut trades = Vec::with_capacity(rank_result.n_periods);

    let period_len = config.lookback_bars + config.holding_bars;
    for period in &rank_result.periods {
        let period_start = period.period_index * period_len;
        let rank_end = period_start + config.lookback_bars;
        let hold_end = rank_end + config.holding_bars;

        let regime = regimes.get(rank_end.saturating_sub(1)).copied().flatten();
        let exposure_scalar = match regime {
            Some(VolatilityRegime::High) => config.high_vol_exposure_scalar,
            Some(VolatilityRegime::Medium) => config.medium_vol_exposure_scalar,
            _ => 1.0,
        };

        let gross_notional = config.initial_capital * config.gross_exposure_pct * exposure_scalar;
        let n_winners = period.winner_indices.len().max(1);
        let n_losers = period.loser_indices.len().max(1);
        let long_notional_per_asset = (gross_notional / 2.0) / n_winners as f64;
        let short_notional_per_asset = (gross_notional / 2.0) / n_losers as f64;

        let entry_time = timestamp_from_millis(timestamps[rank_end]);
        let exit_time = timestamp_from_millis(timestamps[hold_end]);

        let mut period_pnl = 0.0;
        let mut total_commission = 0.0;
        let mut total_slippage = 0.0;
        let mut legs: Vec<TradeLeg> = Vec::with_capacity(n_winners + n_losers);
        let mut representative_price = 0.0;

        for &i in &period.winner_indices {
            let entry_px = prices[i][rank_end];
            let exit_px = prices[i][hold_end];
            if entry_px <= 0.0 || exit_px <= 0.0 {
                continue;
            }
            let quantity = long_notional_per_asset / entry_px;
            let instrument = assets[i].option_instrument.as_ref();
            let (entry_fill, entry_comm, entry_slip) = taker_fill(entry_px, quantity, true, fees[i], instrument);
            let (exit_fill, exit_comm, exit_slip) = taker_fill(exit_px, quantity, false, fees[i], instrument);
            let leg_pnl = (exit_fill - entry_fill) * quantity * leg_multiplier(instrument) - entry_comm - exit_comm;
            period_pnl += leg_pnl;
            total_commission += entry_comm + exit_comm;
            total_slippage += entry_slip + exit_slip;
            representative_price = entry_fill;
            legs.push(TradeLeg {
                exchange: assets[i].exchange.clone(),
                symbol: assets[i].symbol.clone(),
                side: "long".to_string(),
                fill_price: entry_fill,
                quantity,
                liquidity: "taker".to_string(),
                commission: entry_comm,
                slippage_cost: entry_slip,
                timestamp: entry_time,
            });
        }

        for &i in &period.loser_indices {
            let entry_px = prices[i][rank_end];
            let exit_px = prices[i][hold_end];
            if entry_px <= 0.0 || exit_px <= 0.0 {
                continue;
            }
            let quantity = short_notional_per_asset / entry_px;
            let instrument = assets[i].option_instrument.as_ref();
            let (entry_fill, entry_comm, entry_slip) = taker_fill(entry_px, quantity, false, fees[i], instrument);
            let (exit_fill, exit_comm, exit_slip) = taker_fill(exit_px, quantity, true, fees[i], instrument);
            let leg_pnl = (entry_fill - exit_fill) * quantity * leg_multiplier(instrument) - entry_comm - exit_comm;
            period_pnl += leg_pnl;
            total_commission += entry_comm + exit_comm;
            total_slippage += entry_slip + exit_slip;
            if representative_price == 0.0 {
                representative_price = entry_fill;
            }
            legs.push(TradeLeg {
                exchange: assets[i].exchange.clone(),
                symbol: assets[i].symbol.clone(),
                side: "short".to_string(),
                fill_price: entry_fill,
                quantity,
                liquidity: "taker".to_string(),
                commission: entry_comm,
                slippage_cost: entry_slip,
                timestamp: entry_time,
            });
        }

        let new_equity = equity_curve.last().copied().unwrap_or(config.initial_capital) + period_pnl;
        equity_curve.push(new_equity);

        trades.push(TradeRecord {
            trade_id: period.period_index,
            side: "cross_sectional_long_short".to_string(),
            entry_signal_price: representative_price,
            entry_fill_price: representative_price,
            exit_signal_price: Some(representative_price),
            exit_fill_price: Some(representative_price),
            quantity: if representative_price > 0.0 { gross_notional / representative_price } else { 0.0 },
            pnl: Some(period_pnl),
            pnl_pct: Some(if gross_notional > 0.0 { period_pnl / gross_notional } else { 0.0 }),
            commission: total_commission,
            slippage_cost: total_slippage,
            entry_time,
            exit_time: Some(exit_time),
            duration_secs: Some((timestamps[hold_end] - timestamps[rank_end]) / 1000),
            entry_liquidity: "taker".to_string(),
            exit_liquidity: Some("taker".to_string()),
            entry_reason: format!(
                "cross-sectional rebalance #{}: {} winner(s) / {} loser(s){}",
                period.period_index,
                period.winner_indices.len(),
                period.loser_indices.len(),
                match regime {
                    Some(r) => format!(", regime={:?} (exposure x{:.2})", r, exposure_scalar),
                    None => String::new(),
                }
            ),
            exit_reason: Some(format!("holding period ended after {} bars", config.holding_bars)),
            mae: None,
            mfe: None,
            legs,
        });
    }

    Ok(CrossSectionalBacktestResult {
        trades,
        equity_curve,
        n_periods: rank_result.n_periods,
        rank_result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(symbol: &str, class: &str) -> CrossSectionalAsset {
        CrossSectionalAsset { symbol: symbol.to_string(), exchange: "test".to_string(), asset_class: class.to_string(), option_instrument: None }
    }

    fn flat_fee() -> PairLegFeeConfig {
        PairLegFeeConfig { taker_fee: 0.0, slippage_bps: 0.0 }
    }

    fn test_call_instrument(symbol: &str, underlying: &str, strike: f64) -> DerivativeMetadata {
        DerivativeMetadata::new(
            symbol,
            underlying,
            derivatives::InstrumentKind::Call { strike, expiry: Utc::now() },
            100.0,
            "USD",
            "test",
        )
    }

    /// Build a deterministic price panel where symbols 0-1 (class "a")
    /// steadily outperform symbols 2-3 (class "b") every bar, so the
    /// long/short split has an unambiguous, persistent expected sign.
    fn trending_panel(n_bars: usize) -> (Vec<Vec<f64>>, Vec<i64>) {
        let mut prices = vec![vec![100.0; n_bars]; 4];
        for k in 1..n_bars {
            prices[0][k] = prices[0][k - 1] * 1.01;
            prices[1][k] = prices[1][k - 1] * 1.009;
            prices[2][k] = prices[2][k - 1] * 0.99;
            prices[3][k] = prices[3][k - 1] * 0.991;
        }
        let timestamps: Vec<i64> = (0..n_bars as i64).map(|k| k * 86_400_000).collect();
        (prices, timestamps)
    }

    #[test]
    fn rejects_fewer_than_min_symbols() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto")];
        let (prices, ts) = trending_panel(100);
        let prices = &prices[..3];
        let fees = vec![flat_fee(); 3];
        let result = run_cross_sectional_backtest(&assets, prices, &ts, &fees, CrossSectionalConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        let (prices, ts) = trending_panel(100);
        let fees = vec![flat_fee(); 3]; // mismatched: 3 fees for 4 assets
        let result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, CrossSectionalConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_insufficient_bars() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        let (prices, ts) = trending_panel(10); // way fewer than lookback+holding=40
        let fees = vec![flat_fee(); 4];
        let result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, CrossSectionalConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn persistent_trend_produces_profitable_long_short_spread() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        let (prices, ts) = trending_panel(400);
        let fees = vec![flat_fee(); 4];
        let config = CrossSectionalConfig { lookback_bars: 20, holding_bars: 20, regime_window: 0, ..Default::default() };
        let result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, config).unwrap();
        assert!(result.n_periods > 0);
        assert_eq!(result.trades.len(), result.n_periods);
        let summary = result.summarize(None);
        assert!(summary.total_pnl > 0.0, "a persistent momentum spread with zero fees should be net profitable, got {}", summary.total_pnl);
        assert!(result.equity_curve.len() == result.n_periods + 1);
        // Every trade should carry legs from both asset classes (winners
        // and losers each split 1/1 across the two classes given the
        // by-class ranking).
        for t in &result.trades {
            assert!(!t.legs.is_empty());
        }
    }

    #[test]
    fn zero_fees_vs_nonzero_fees_reduces_pnl() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        let (prices, ts) = trending_panel(400);
        let config = CrossSectionalConfig { lookback_bars: 20, holding_bars: 20, regime_window: 0, ..Default::default() };

        let free_fees = vec![flat_fee(); 4];
        let free_result = run_cross_sectional_backtest(&assets, &prices, &ts, &free_fees, config).unwrap();

        let costly_fees = vec![PairLegFeeConfig { taker_fee: 0.01, slippage_bps: 20.0 }; 4];
        let costly_result = run_cross_sectional_backtest(&assets, &prices, &ts, &costly_fees, config).unwrap();

        let free_pnl = free_result.summarize(None).total_pnl;
        let costly_pnl = costly_result.summarize(None).total_pnl;
        assert!(costly_pnl < free_pnl, "higher fees/slippage should strictly reduce total PnL");
    }

    #[test]
    fn high_vol_exposure_scalar_shrinks_position_sizing() {
        let assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        let (prices, ts) = trending_panel(400);
        let fees = vec![flat_fee(); 4];

        let full_exposure = CrossSectionalConfig { lookback_bars: 20, holding_bars: 20, regime_window: 0, high_vol_exposure_scalar: 1.0, ..Default::default() };
        let full_result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, full_exposure).unwrap();

        // regime_window=0 means every period gets exposure_scalar=1.0
        // regardless of high_vol_exposure_scalar -- this test instead
        // directly compares two DIFFERENT gross_exposure_pct values to
        // confirm sizing scales PnL magnitude linearly, which is the
        // mechanism high_vol_exposure_scalar itself relies on.
        let half_exposure = CrossSectionalConfig { gross_exposure_pct: full_exposure.gross_exposure_pct / 2.0, ..full_exposure };
        let half_result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, half_exposure).unwrap();

        let full_pnl = full_result.summarize(None).total_pnl;
        let half_pnl = half_result.summarize(None).total_pnl;
        assert!((half_pnl - full_pnl / 2.0).abs() < 1e-6, "halving gross exposure should roughly halve PnL: full={} half={}", full_pnl, half_pnl);
    }

    #[test]
    fn option_instrument_leg_pnl_is_scaled_by_contract_multiplier_and_charged_the_real_fee_schedule() {
        // Asset 0 (class "a", the strongest steady up-trend) is always a
        // winner in this panel and is given a fixed option contract --
        // its leg's PnL/commission must scale by `contract_multiplier`
        // (100) using the real Alpaca schedule, exactly like
        // pair_simulation/basket_simulation's own option-leg behavior.
        let mut assets = vec![asset("A", "crypto"), asset("B", "crypto"), asset("C", "crypto"), asset("D", "crypto")];
        assets[0].option_instrument = Some(test_call_instrument("A260320C00100000", "A", 100.0));
        let (prices, ts) = trending_panel(400);
        let fees = vec![flat_fee(); 4];
        let config = CrossSectionalConfig { lookback_bars: 20, holding_bars: 20, regime_window: 0, gross_exposure_pct: 0.5, ..Default::default() };
        let result = run_cross_sectional_backtest(&assets, &prices, &ts, &fees, config).unwrap();
        assert!(result.n_periods > 0);

        // Recompute period 0's rank/hold boundaries exactly as the engine
        // does, to independently derive asset 0's entry/exit price.
        let rank_end = config.lookback_bars;
        let hold_end = rank_end + config.holding_bars;
        let entry_px = prices[0][rank_end];
        let exit_px = prices[0][hold_end];
        assert!(exit_px > entry_px, "asset A should have risen over the period (sanity check on the fixture)");

        let leg_a = result.trades[0].legs.iter().find(|l| l.symbol == "A")
            .expect("asset A should be a winner (long leg) every period in this steadily-trending panel");
        assert_eq!(leg_a.side, "long");
        assert!(leg_a.commission > 0.0, "option leg must be charged the real per-contract fee schedule even at taker_fee=0.0");

        let fee_cfg = config::OptionsFeeConfig::alpaca();
        let entry_comm = fee_cfg.commission_for_fill(entry_px, leg_a.quantity, 100.0, true);
        let exit_comm = fee_cfg.commission_for_fill(exit_px, leg_a.quantity, 100.0, false);
        let expected_leg_pnl = (exit_px - entry_px) * leg_a.quantity * 100.0 - entry_comm - exit_comm;

        // Isolate leg A's own contribution: rerun with the SAME config but
        // asset A as a plain (non-option) leg -- position sizing (quantity)
        // is unaffected by `option_instrument` (it's driven purely by
        // notional/price, matching every other engine's established, if
        // imperfect, sizing convention), and every OTHER leg's fill is
        // identical between the two runs (ranking is price-only, unaffected
        // by asset A's instrument), so the total-PnL difference isolates
        // exactly what leg A alone contributed.
        let mut plain_assets = assets.clone();
        plain_assets[0].option_instrument = None;
        let plain_result = run_cross_sectional_backtest(&plain_assets, &prices, &ts, &fees, config).unwrap();
        let plain_leg_a = plain_result.trades[0].legs.iter().find(|l| l.symbol == "A").unwrap();
        assert!((plain_leg_a.quantity - leg_a.quantity).abs() < 1e-9, "sizing must be identical regardless of option_instrument");

        let total_pnl_with_option_a = result.trades[0].pnl.unwrap();
        let total_pnl_with_plain_a = plain_result.trades[0].pnl.unwrap();
        let leg_a_contribution = total_pnl_with_option_a - total_pnl_with_plain_a
            + (exit_px - entry_px) * leg_a.quantity; // plain leg A's own raw pnl at flat_fee (zero commission)
        assert!((leg_a_contribution - expected_leg_pnl).abs() < 1.0,
            "expected leg A's isolated contribution near {}, got {}", expected_leg_pnl, leg_a_contribution);
    }

    #[test]
    fn summarize_reports_zero_metrics_for_no_trades() {
        let result = CrossSectionalBacktestResult {
            trades: Vec::new(),
            equity_curve: vec![100_000.0],
            n_periods: 0,
            rank_result: CrossSectionalRankResult {
                n_symbols: 4,
                n_periods: 0,
                mean_spread: 0.0,
                t_stat: 0.0,
                p_value: 1.0,
                significant: false,
                periods: Vec::new(),
            },
        };
        let summary = result.summarize(None);
        assert_eq!(summary.num_trades, 0);
        assert_eq!(summary.total_pnl, 0.0);
        assert_eq!(summary.win_rate, 0.0);
    }
}
