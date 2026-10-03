//! `BacktestResult -> SeriesRows` bridge (gap-closure plan W3.2): turns a general-engine backtest result into the
//! replication ladder's per-bar row layout (`weightsim_rules::ladder::checks::SeriesRows`), so a result produced by
//! the general engine -- not the pure `weightsim` simulator the ladder certifies directly -- can be compared against
//! a pinned answer key through the SAME `checks::compare` Tier I/II/III machinery. No second comparison engine
//! (design C7: the comparator already exists, generic; this crate is only wiring a new producer into it).
//!
//! # Scope: what this bridge produces
//!
//! * `ret[t]` -- the bar's net return, derived from `equity_curve` (`equity[t] / equity[t-1] - 1`, with
//!   `equity[-1] := initial_capital`). `BacktestResult` has no separate per-bar net-return series, so this is
//!   derived rather than copied; it mirrors the ladder's own "flat at equity 1.0 the bar before the window"
//!   convention (`weightsim_rules::ladder::runner::entry_date`'s doc comment).
//! * `equity[t]` -- `equity_curve` verbatim (so Tier II can compare it when the key side has an equity column too).
//! * `w_target[t]`/`w_held[t]` -- W3.5's opt-in `per_bar_weights` export (`crate::per_bar_weights`), SHIFTED one bar
//!   forward exactly as that module's own doc comment specifies: *"the bridge (W3.2) therefore sets `w_held[t] =
//!   rows[t-1]` and `w_target[t] = rows[t-1]`"*. Row 0 of the output has no predecessor, so it is flat (all-zero),
//!   matching the ladder's pre-window convention (`runner::entry_date`: "the book is flat ... at the close of the
//!   bar before the window").
//! * `cost[t]`/`traded[t]` -- always `None`. The general engine does not export a per-bar cost or traded-notional
//!   SERIES (only aggregate figures in `TransactionCostAnalysis`), so these two Tier II columns are left absent
//!   rather than faked. `checks::compare` already treats an absent column on either side as "not checked"
//!   (`opt_pair`), so this narrows what Tier II checks; it does not weaken what it does check.
//!
//! # Refusals -- typed, never a panic, never a silently-empty result
//!
//! * [`BridgeError::MissingPerBarWeights`] -- `per_bar_weights` is `None` (the export is opt-in,
//!   `AnalysisConfig::export_per_bar_weights`, W3.5).
//! * [`BridgeError::NotBarsSourced`] -- `equity_curve_timestamps` is empty, or one of its values cannot be read as
//!   a calendar date. **Why this is the right refusal condition in THIS repo, read honestly:** the plan's own text
//!   points at ENG's `strategy_ensemble_service.rs:1225`-style synthetic-timestamp detection as the pattern to
//!   reuse, but that code lives in `BacktestingEngine`, a different crate graph this bridge cannot reach, and
//!   `BacktestResult` (`crate::types`) carries no `timestamps_source`/`is_synthetic` flag of its own. The one
//!   provenance signal the type DOES document is presence vs. absence: `equity_curve_timestamps`'s own doc comment
//!   (`crate::types`, "Bug #33") says the vector is "empty if not tracked." Every CORE simulation path that
//!   populates it writes a real bar timestamp (`timestamp.timestamp_millis()` in `simulation.rs`,
//!   `python_simulation.rs`, etc. -- verified by inspection; none synthesizes a sequential index). So "empty" is
//!   CORE's own honest equivalent of "not bars-sourced," and refusing on it is the correct, non-invented gate. A
//!   content-based heuristic (rejecting suspiciously small millisecond values, say) would be inventing a detector
//!   this codebase does not have and this PR was not asked to add.
//! * [`BridgeError::WeightsTimestampMismatch`] -- `per_bar_weights.timestamps` disagrees with
//!   `equity_curve_timestamps` (length or values), though `PerBarWeights::timestamps`'s own doc comment documents
//!   them as identical ("parallel to `rows` and identical to `equity_curve_timestamps`"). A mismatch means the
//!   result was hand-built or corrupted, not a real engine run; this refuses rather than silently misaligning rows.
//! * [`BridgeError::LengthMismatch`] -- `equity_curve`, `equity_curve_timestamps` and `per_bar_weights.rows` are not
//!   all the same length (defensive; W3.5's own tracker guarantees this in practice, see `per_bar_weights.rs`'s
//!   module doc, "so `timestamps == equity_curve_timestamps` and `rows.len() == equity_curve.len()`").
//!
//! # What this bridge deliberately does NOT attempt (the plan's own honesty, design W3.2)
//!
//! Trade counts "from `trade_log`" cannot be compared against the key's per-asset flip counter
//! (`weightsim_rules::ladder::runner::flips_by_key_convention`): `TradeRecord` (`crate::types`) carries no
//! instrument symbol field at all (verified by inspection), so a multi-asset run's trade log cannot be attributed
//! back to a specific column of `w_target`. [`trade_flips`] below derives the flip count from the per-bar weights
//! this bridge already has instead (sign changes of `w_target`, the SAME definition `flips_by_key_convention`
//! uses on the key side) -- a strictly better signal than the trade log for this purpose, and one this bridge can
//! give for free now that W3.5 exports per-bar weights. [`trade_log_count`] is still exposed separately as the one
//! honest, direct trade-count number `trade_log` itself can give; it is disclosure only, never fed to Tier I's
//! `trades_within_band`.
//!
//! One more honest limitation of the shift itself: because output row `t` reads raw row `t-1` and output row 0 is a
//! synthesized flat row, the raw export's OWN final row (the position state after the very last bar's fills) is
//! never read by anything -- the shift "loses" visibility of whatever happened on the last bar. This is inherent to
//! the one-bar-shift convention W3.5 specified, not a bug introduced here; a real-money consequence does not follow
//! from it (the ladder's own Tier II/III comparisons are keyed by DATE against the key's matching row, not by "the
//! last row of either series").
//!
//! This also means Tier III (weight agreement) is NOT the fallback-to-trade-log case the original W3 design
//! point anticipated ("Tier III falls back to `agent_faithfulness_check`'s trade-log tier until the engine exports
//! weights") -- W3.5 shipped the per-bar weight export first, so this bridge's `w_target`/`w_held` already let
//! `checks::compare` run Tier III directly on general-engine results. What is still missing for FULL Tier II parity
//! is only the cost/traded columns noted above.

use chrono::Datelike;
use weightsim::Date;
use weightsim_rules::ladder::checks::SeriesRows;

use crate::types::BacktestResult;

/// Why [`backtest_result_to_series_rows`] refused to bridge a result. Every variant is a typed refusal: the caller
/// gets no [`SeriesRows`] at all, never a partially-built or default one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    /// `per_bar_weights` is `None` (W3.5's export is opt-in and was not turned on for this run).
    MissingPerBarWeights,
    /// `equity_curve_timestamps` is empty, or a value in it is not a representable calendar date -- CORE's own
    /// equivalent of "not bars-sourced" (see module docs).
    NotBarsSourced,
    /// `per_bar_weights.timestamps` and `equity_curve_timestamps` disagree, though the type documents them as
    /// identical.
    WeightsTimestampMismatch,
    /// `equity_curve`, `equity_curve_timestamps` and `per_bar_weights.rows` are not all the same length.
    LengthMismatch { equity: usize, timestamps: usize, weight_rows: usize },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::MissingPerBarWeights => {
                write!(f, "BacktestResult.per_bar_weights is None (export_per_bar_weights was off for this run)")
            }
            BridgeError::NotBarsSourced => {
                write!(f, "BacktestResult.equity_curve_timestamps is empty or unparseable: timestamps are not bars-sourced")
            }
            BridgeError::WeightsTimestampMismatch => {
                write!(f, "per_bar_weights.timestamps does not match equity_curve_timestamps")
            }
            BridgeError::LengthMismatch { equity, timestamps, weight_rows } => write!(
                f,
                "equity_curve ({equity}), equity_curve_timestamps ({timestamps}) and per_bar_weights.rows \
                 ({weight_rows}) must all be the same length"
            ),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Unix-millisecond timestamp to a UTC calendar [`Date`]. The ladder's date type has no time-of-day: a bar is
/// identified by its calendar date (design S-1), matching how the key's own fixtures are dated.
fn date_from_millis(ms: i64) -> Option<Date> {
    let naive = chrono::DateTime::from_timestamp_millis(ms)?.date_naive();
    Date::new(naive.year(), naive.month() as u8, naive.day() as u8).ok()
}

/// Build the ladder's [`SeriesRows`] from a general-engine [`BacktestResult`], applying W3.5's shift convention
/// exactly (see module docs for the full mapping). Requires `per_bar_weights` populated and bars-sourced
/// `equity_curve_timestamps`; returns a typed [`BridgeError`] otherwise -- never a panic, never a silently-empty
/// result.
pub fn backtest_result_to_series_rows(result: &BacktestResult) -> Result<SeriesRows, BridgeError> {
    let weights = result.per_bar_weights.as_ref().ok_or(BridgeError::MissingPerBarWeights)?;
    if result.equity_curve_timestamps.is_empty() {
        return Err(BridgeError::NotBarsSourced);
    }
    if weights.timestamps != result.equity_curve_timestamps {
        return Err(BridgeError::WeightsTimestampMismatch);
    }
    let n = result.equity_curve.len();
    if result.equity_curve_timestamps.len() != n || weights.rows.len() != n {
        return Err(BridgeError::LengthMismatch {
            equity: n,
            timestamps: result.equity_curve_timestamps.len(),
            weight_rows: weights.rows.len(),
        });
    }

    let dates: Vec<Date> = result
        .equity_curve_timestamps
        .iter()
        .map(|&ms| date_from_millis(ms).ok_or(BridgeError::NotBarsSourced))
        .collect::<Result<_, _>>()?;

    let mut ret = Vec::with_capacity(n);
    for t in 0..n {
        let prev_equity = if t == 0 { result.initial_capital } else { result.equity_curve[t - 1] };
        ret.push(result.equity_curve[t] / prev_equity - 1.0);
    }

    let num_symbols = weights.symbols.len();
    let zero_row = vec![0.0; num_symbols];
    // W3.5's own doc comment (per_bar_weights.rs): "the bridge (W3.2) therefore sets w_held[t] = rows[t-1] and
    // w_target[t] = rows[t-1]". Row 0 has no predecessor, so it is flat, matching the ladder's own pre-window
    // convention.
    let shifted: Vec<Vec<f64>> =
        (0..n).map(|t| if t == 0 { zero_row.clone() } else { weights.rows[t - 1].clone() }).collect();

    Ok(SeriesRows {
        dates,
        ret,
        equity: Some(result.equity_curve.clone()),
        cost: None,
        traded: None,
        w_target: Some(shifted.clone()),
        w_held: Some(shifted),
    })
}

/// Asset-level sign flips of `rows.w_target`, using the SAME definition `weightsim_rules::ladder::runner::
/// flips_by_key_convention` uses on the key side (count of per-asset sign changes between successive rows). Use
/// this, not [`trade_log_count`], as the run-side counter for `checks::trades_within_band`: see the module docs for
/// why `trade_log` cannot be attributed per-asset at all.
pub fn trade_flips(rows: &SeriesRows) -> u64 {
    let Some(w) = &rows.w_target else { return 0 };
    let sign = |x: f64| -> i8 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    };
    let mut flips = 0u64;
    for i in 1..w.len() {
        for (a, b) in w[i - 1].iter().zip(&w[i]) {
            if sign(*a) != sign(*b) {
                flips += 1;
            }
        }
    }
    flips
}

/// The one trade-count number `trade_log` itself can give directly: how many trade records it holds (open or
/// closed). Disclosure only -- see the module docs for why this is not a substitute for [`trade_flips`].
pub fn trade_log_count(result: &BacktestResult) -> usize {
    result.trade_log.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_bar_weights::PerBarWeightTracker;
    use crate::types::PerBarWeights;
    use portfoliomanager::{PortfolioState, Position, PositionSide};
    use chrono::Utc;

    /// Milliseconds for `2024-01-01 + n` days (real bar timestamps, never synthetic indices).
    fn day_ms(n: i64) -> i64 {
        1_704_067_200_000 + n * 86_400_000
    }

    fn pos(side: PositionSide, qty: f64, entry: f64, margin: f64) -> Position {
        Position {
            symbol: "BTC/USD".into(),
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

    /// Long-only single-asset fixture: flat, then long, then flat again -- the same shape
    /// `per_bar_weights.rs`'s own obviously-passing fixture uses, run through this bridge.
    fn long_only_fixture() -> BacktestResult {
        let mut tracker = PerBarWeightTracker::new(Some("BTC/USD".into()), 4);
        let mut pf = PortfolioState::default();
        pf.balance = 1000.0;
        tracker.sample(&pf, 100.0, pf.get_total_value(), day_ms(0));
        pf.balance = 0.0;
        pf.positions.push(pos(PositionSide::Long, 10.0, 100.0, 1000.0));
        pf.update_unrealized_pnl(100.0);
        tracker.sample(&pf, 100.0, pf.get_total_value(), day_ms(1));
        pf.update_unrealized_pnl(120.0);
        tracker.sample(&pf, 120.0, pf.get_total_value(), day_ms(2));
        pf.positions[0].close_time = Some(Utc::now());
        pf.balance = 1200.0;
        tracker.sample(&pf, 120.0, pf.get_total_value(), day_ms(3));
        let weights = tracker.finish();

        let equity_curve = vec![1000.0, 1000.0, 1200.0, 1200.0];
        let equity_curve_timestamps = weights.timestamps.clone();
        BacktestResult {
            initial_capital: 1000.0,
            equity_curve,
            equity_curve_timestamps,
            per_bar_weights: Some(weights),
            ..BacktestResult::default()
        }
    }

    #[test]
    fn long_only_fixture_produces_the_expected_shift() {
        let result = long_only_fixture();
        let rows = backtest_result_to_series_rows(&result).expect("bridges cleanly");
        assert_eq!(rows.dates.len(), 4);
        assert_eq!(rows.dates[0], Date::new(2024, 1, 1).unwrap());
        assert_eq!(rows.dates[3], Date::new(2024, 1, 4).unwrap());

        // ret[t] = equity[t]/equity[t-1] - 1, equity[-1] := initial_capital
        assert_eq!(rows.ret[0], 0.0); // 1000/1000 - 1
        assert_eq!(rows.ret[1], 0.0); // 1000/1000 - 1
        assert!((rows.ret[2] - 0.2).abs() < 1e-12); // 1200/1000 - 1
        assert_eq!(rows.ret[3], 0.0); // 1200/1200 - 1

        // w_target[t] = w_held[t] = the raw export's rows[t-1]; row 0 is flat (no predecessor).
        let raw = [vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        let w_target = rows.w_target.as_ref().unwrap();
        let w_held = rows.w_held.as_ref().unwrap();
        assert_eq!(w_target[0], vec![0.0]); // flat pre-window row
        for t in 1..4 {
            assert_eq!(w_target[t], raw[t - 1], "w_target[{t}] should be the raw export's row {}", t - 1);
            assert_eq!(w_held[t], raw[t - 1]);
        }
        assert_eq!(rows.equity.as_ref().unwrap(), &result.equity_curve);
        assert!(rows.cost.is_none() && rows.traded.is_none());
    }

    #[test]
    fn missing_per_bar_weights_is_a_typed_refusal_not_a_panic() {
        let result = BacktestResult {
            equity_curve: vec![1.0, 1.0],
            equity_curve_timestamps: vec![day_ms(0), day_ms(1)],
            per_bar_weights: None,
            ..BacktestResult::default()
        };
        assert_eq!(backtest_result_to_series_rows(&result), Err(BridgeError::MissingPerBarWeights));
    }

    #[test]
    fn empty_timestamps_is_a_typed_refusal_not_a_panic() {
        let mut result = long_only_fixture();
        result.equity_curve_timestamps = Vec::new();
        // the per_bar_weights tracker's own timestamps are untouched, so this is purely a "not bars-sourced"
        // refusal, not a secondary mismatch.
        assert_eq!(backtest_result_to_series_rows(&result), Err(BridgeError::NotBarsSourced));
    }

    #[test]
    fn mismatched_weight_timestamps_is_a_typed_refusal() {
        let mut result = long_only_fixture();
        if let Some(w) = result.per_bar_weights.as_mut() {
            w.timestamps[1] += 1;
        }
        assert_eq!(backtest_result_to_series_rows(&result), Err(BridgeError::WeightsTimestampMismatch));
    }

    #[test]
    fn length_mismatch_is_a_typed_refusal() {
        let mut result = long_only_fixture();
        result.equity_curve.push(1.0);
        assert_eq!(
            backtest_result_to_series_rows(&result),
            Err(BridgeError::LengthMismatch { equity: 5, timestamps: 4, weight_rows: 4 })
        );
    }

    #[test]
    fn trade_flips_counts_sign_changes_of_w_target() {
        let result = long_only_fixture();
        let rows = backtest_result_to_series_rows(&result).unwrap();
        // Shifted output rows are [flat, flat, long, long] (the raw export's final flat row -- AFTER the position
        // closes on the last bar -- has no output row to land in under the one-bar shift: see the module docs'
        // "what this bridge deliberately does NOT attempt" note on this exact convention). One sign change.
        assert_eq!(trade_flips(&rows), 1);
    }

    #[test]
    fn trade_log_count_is_the_plain_length() {
        let mut result = BacktestResult::default();
        assert_eq!(trade_log_count(&result), 0);
        result.trade_log.push(crate::types::TradeRecord {
            trade_id: 0,
            side: "long".into(),
            entry_signal_price: 1.0,
            entry_fill_price: 1.0,
            exit_signal_price: None,
            exit_fill_price: None,
            quantity: 1.0,
            pnl: None,
            pnl_pct: None,
            commission: 0.0,
            slippage_cost: 0.0,
            entry_time: Utc::now(),
            exit_time: None,
            duration_secs: None,
            entry_liquidity: "taker".into(),
            exit_liquidity: None,
            entry_reason: "test".into(),
            exit_reason: None,
            mae: None,
            mfe: None,
            legs: Vec::new(),
        });
        assert_eq!(trade_log_count(&result), 1);
    }
}
