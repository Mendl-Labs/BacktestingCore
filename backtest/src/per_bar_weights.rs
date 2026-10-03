//! Opt-in per-bar weight export for the replication ladder's Tier III (gap-closure plan W3.5; IMPL capability audit
//! item 7). Pure bookkeeping over the simulator's own `PortfolioState`; no I/O.
//!
//! # What a cell means
//!
//! `rows[bar][i]` is the **signed held weight** of instrument `i` at the close of the sampled bar, AFTER that bar's
//! fills: `sum over open positions of the instrument of (sign x quantity x mark x contract_multiplier)` divided by the
//! equity the equity curve records at the same sample point (`equity_curve[bar]`). Long is positive, short negative,
//! flat is exactly `0.0`. The mark of a spot/linear position is the bar's close (what `update_unrealized_pnl` and
//! `equity_at` mark it at); an option leg is marked at its own `mark_price` (its Greeks-derived mark) and scaled by
//! its contract multiplier. If the equity at the sample point is not strictly positive the weights of that bar are
//! `NaN` (undefined, never silently `0.0`).
//!
//! Rows are taken exactly where the equity curve is sampled (every `equity_sample_interval` ticks and the last tick,
//! and the trading-halted branch), so `timestamps == equity_curve_timestamps` and `rows.len() == equity_curve.len()`.
//!
//! # Why the HELD weight, and how it maps onto the ladder's `SeriesRows`
//!
//! The general engine has no target-weight number: sizing is a per-entry quantity increment that the risk manager,
//! the aggregate cap, volume caps, limit fills and liquidations all change before (or instead of) a fill. The only
//! well-defined per-bar quantity is the realised position after the bar's fills. In `weightsim_rules::ladder::checks`,
//! `SeriesRows` row `t` carries `w_target[t]` = the standing target in force DURING bar `t` (decided at the close of
//! `t-1`) and `w_held[t]` = the start-of-bar held weights (post-trade weights at the close of `t-1`), and Tier III
//! compares `w_target`. For this engine a decision at the close of `t` is filled at that same close, so the position
//! in force during bar `t+1` IS the held weight at the close of `t`. The bridge (W3.2) therefore sets
//! `w_held[t] = rows[t-1]` and `w_target[t] = rows[t-1]`, i.e. this export is one bar EARLIER than the ladder's row
//! convention, and the bridge shifts it.

use portfoliomanager::{PortfolioState, Position, PositionSide};

use crate::types::PerBarWeights;

/// Signed notional of one position at `mark_fallback` (the bar's close): `sign x quantity x mark x multiplier`,
/// `0.0` for a closed or empty position. Option legs use their own `mark_price` when they have one.
pub fn signed_notional(p: &Position, mark_fallback: f64) -> f64 {
    if p.close_time.is_some() || p.quantity <= 0.0 {
        return 0.0;
    }
    let (is_option, multiplier) = match &p.instrument {
        Some(i) => (i.instrument_kind.is_option(), i.contract_multiplier),
        None => (false, 1.0),
    };
    let mark = if is_option {
        p.mark_price.unwrap_or(mark_fallback)
    } else {
        mark_fallback
    };
    let notional = p.quantity * mark * multiplier;
    match p.side {
        PositionSide::Long => notional,
        PositionSide::Short => -notional,
    }
}

/// Accumulates one row per equity-curve sample point.
#[derive(Debug, Clone)]
pub struct PerBarWeightTracker {
    symbols: Vec<String>,
    timestamps: Vec<i64>,
    rows: Vec<Vec<f64>>,
}

impl PerBarWeightTracker {
    /// `seed_symbol` is the run's data symbol: it is column 0 from the first row so a run that never trades still
    /// exports one all-zero column.
    pub fn new(seed_symbol: Option<String>, capacity: usize) -> Self {
        PerBarWeightTracker {
            symbols: seed_symbol.into_iter().collect(),
            timestamps: Vec::with_capacity(capacity),
            rows: Vec::with_capacity(capacity),
        }
    }

    fn column(&mut self, symbol: &str) -> usize {
        if let Some(i) = self.symbols.iter().position(|s| s == symbol) {
            return i;
        }
        self.symbols.push(symbol.to_string());
        for row in &mut self.rows {
            row.push(0.0);
        }
        self.symbols.len() - 1
    }

    /// Record the row for one sampled bar: `portfolio` after the bar's fills, `mark_fallback` the bar's close,
    /// `equity` the value pushed to the equity curve at this point, `timestamp_ms` its timestamp.
    pub fn sample(
        &mut self,
        portfolio: &PortfolioState,
        mark_fallback: f64,
        equity: f64,
        timestamp_ms: i64,
    ) {
        let mut notional = vec![0.0f64; self.symbols.len()];
        for p in &portfolio.positions {
            if p.close_time.is_some() || p.quantity <= 0.0 {
                continue;
            }
            let col = self.column(&p.symbol);
            if col >= notional.len() {
                notional.resize(self.symbols.len(), 0.0);
            }
            notional[col] += signed_notional(p, mark_fallback);
        }
        let row: Vec<f64> = if equity.is_finite() && equity > 0.0 {
            notional.iter().map(|n| n / equity).collect()
        } else {
            vec![f64::NAN; notional.len()]
        };
        self.timestamps.push(timestamp_ms);
        self.rows.push(row);
    }

    pub fn finish(self) -> PerBarWeights {
        PerBarWeights {
            symbols: self.symbols,
            timestamps: self.timestamps,
            rows: self.rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use derivatives::{DerivativeMetadata, InstrumentKind};

    fn pos(symbol: &str, side: PositionSide, qty: f64, entry: f64, margin: f64) -> Position {
        Position {
            symbol: symbol.into(),
            side,
            quantity: qty,
            entry_price: entry,
            mark_price: Some(entry),
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            open_time: Utc::now(),
            close_time: None,
            instrument: None,
            greeks: None,
            margin_posted: margin,
        }
    }

    /// The obviously-passing fixture: a long-only single-asset book is 1.0 while in position and 0.0 while flat.
    #[test]
    fn long_only_single_asset_is_one_in_position_and_zero_flat() {
        let mut t = PerBarWeightTracker::new(Some("BTC/USD".into()), 8);
        let mut pf = PortfolioState::default();
        pf.balance = 1000.0;
        // flat
        t.sample(&pf, 100.0, pf.get_total_value(), 1);
        // long 10 units at 100 with the whole book: equity = balance 0 + margin 1000 + unrealized 0
        pf.balance = 0.0;
        pf.positions
            .push(pos("BTC/USD", PositionSide::Long, 10.0, 100.0, 1000.0));
        pf.update_unrealized_pnl(100.0);
        t.sample(&pf, 100.0, pf.get_total_value(), 2);
        // the mark moves: notional and equity move together, the weight stays 1
        pf.update_unrealized_pnl(120.0);
        t.sample(&pf, 120.0, pf.get_total_value(), 3);
        // closed again
        pf.positions[0].close_time = Some(Utc::now());
        pf.balance = 1200.0;
        t.sample(&pf, 120.0, pf.get_total_value(), 4);
        let w = t.finish();
        assert_eq!(w.symbols, vec!["BTC/USD".to_string()]);
        assert_eq!(w.timestamps, vec![1, 2, 3, 4]);
        assert_eq!(w.rows, vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);
    }

    #[test]
    fn short_is_negative_and_partial_exposure_is_a_fraction() {
        let mut t = PerBarWeightTracker::new(Some("ETH/USD".into()), 2);
        let mut pf = PortfolioState::default();
        pf.balance = 500.0;
        pf.positions
            .push(pos("ETH/USD", PositionSide::Short, 5.0, 100.0, 500.0));
        pf.update_unrealized_pnl(100.0);
        // equity = 500 + 500 + 0 = 1000, short notional 500
        t.sample(&pf, 100.0, pf.get_total_value(), 1);
        assert_eq!(t.clone().finish().rows, vec![vec![-0.5]]);
    }

    #[test]
    fn a_new_instrument_appends_a_column_and_backfills_zero() {
        let mut t = PerBarWeightTracker::new(Some("BTC/USD".into()), 2);
        let mut pf = PortfolioState::default();
        pf.balance = 1000.0;
        t.sample(&pf, 100.0, 1000.0, 1);
        pf.positions
            .push(pos("BTC-OPT", PositionSide::Long, 2.0, 50.0, 100.0));
        t.sample(&pf, 100.0, 1000.0, 2);
        let w = t.finish();
        assert_eq!(
            w.symbols,
            vec!["BTC/USD".to_string(), "BTC-OPT".to_string()]
        );
        assert_eq!(w.rows, vec![vec![0.0, 0.0], vec![0.0, 0.2]]);
    }

    #[test]
    fn option_legs_use_their_own_mark_and_contract_multiplier() {
        let mut p = pos("BTC-25MAR26-100000-C", PositionSide::Long, 3.0, 0.05, 0.0);
        p.instrument = Some(DerivativeMetadata::new(
            "BTC-25MAR26-100000-C",
            "BTC",
            InstrumentKind::Call {
                strike: 100_000.0,
                expiry: Utc::now(),
            },
            10.0,
            "USD",
            "deribit",
        ));
        p.mark_price = Some(0.07);
        // 3 contracts x mark 0.07 x multiplier 10 = 2.1, regardless of the underlying's close
        assert!((signed_notional(&p, 100_000.0) - 2.1).abs() < 1e-12);
        p.side = PositionSide::Short;
        assert!((signed_notional(&p, 100_000.0) + 2.1).abs() < 1e-12);
        // a spot position is marked at the close, never at a stale mark_price
        let mut s = pos("BTC/USD", PositionSide::Long, 1.0, 100.0, 100.0);
        s.mark_price = Some(1.0);
        assert_eq!(signed_notional(&s, 250.0), 250.0);
    }

    #[test]
    fn closed_or_empty_positions_count_as_flat_and_dead_equity_is_nan() {
        let mut closed = pos("BTC/USD", PositionSide::Long, 1.0, 100.0, 100.0);
        closed.close_time = Some(Utc::now());
        assert_eq!(signed_notional(&closed, 100.0), 0.0);
        let empty = pos("BTC/USD", PositionSide::Long, 0.0, 100.0, 0.0);
        assert_eq!(signed_notional(&empty, 100.0), 0.0);
        let mut t = PerBarWeightTracker::new(None, 1);
        let mut pf = PortfolioState::default();
        pf.positions
            .push(pos("BTC/USD", PositionSide::Long, 1.0, 100.0, 100.0));
        t.sample(&pf, 100.0, 0.0, 1);
        let w = t.finish();
        assert_eq!(w.symbols, vec!["BTC/USD".to_string()]);
        assert!(w.rows[0][0].is_nan());
    }

    #[test]
    fn a_run_that_never_trades_exports_one_zero_column_per_sample() {
        let mut t = PerBarWeightTracker::new(Some("SPY".into()), 3);
        let pf = PortfolioState::default();
        for i in 0..3 {
            t.sample(&pf, 10.0, 100.0, i);
        }
        let w = t.finish();
        assert_eq!(w.symbols, vec!["SPY".to_string()]);
        assert_eq!(w.rows, vec![vec![0.0]; 3]);
        assert_eq!(w.timestamps, vec![0, 1, 2]);
    }
}
