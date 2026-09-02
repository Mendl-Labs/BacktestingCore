//! Python Strategy Validation Pipeline — Walk-Forward + Monte Carlo + Auto-Escalation.
//!
//! Provides a staged validation pipeline for user-supplied Python strategies:
//!
//! - **Stage 1**: Quick single-pass backtest (~30 seconds). If strategy loses money → stop.
//! - **Stage 2**: Walk-forward analysis (5–10 windows). If OOS Sharpe < 0.5 → stop.
//! - **Stage 3**: Monte Carlo simulation (1000 runs). If P(ruin) > 20% → warn.
//!
//! Each stage streams progress via a callback, and early-stopping prevents wasting
//! compute on strategies that fail at earlier stages.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
#[cfg(feature = "python")]
use anyhow::Context;
// Unconditional: WalkForwardWindow's *_date fields (below) use DateTime<Utc>
// regardless of the `python` feature; Duration/NaiveDate stay gated since
// every other use of them in this file is inside python-gated code.
use chrono::{DateTime, Utc};
#[cfg(feature = "python")]
use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use crate::{log_error, log_info, log_warn};
#[cfg(feature = "python")]
use crate::logging_facade::BACKTEST_LOGGER;

use config::{BacktestConfig, ExchangeFeeConfig, GeneticConfig, ParameterValue};
use dataloader::MarketData;
#[cfg(feature = "python")]
use genetic::{
    Chromosome, DynamicChromosome, ParameterSchema, set_dynamic_schema,
    GeneticOptimizer, AsyncFitnessFn, AsyncContextFitnessFn, FitnessContext, FitnessResult,
    AdaptiveGeneticOptimizer, AdaptiveSamplingConfig,
    landscape::{self, GAReport},
    nsga2::{NsgaIIOptimizer, NsgaIIResult},
};
#[cfg(not(feature = "python"))]
use genetic::{ParameterSchema, landscape::GAReport};

#[cfg(feature = "python")]
use crate::monte_carlo::{self, MonteCarloResult};
#[cfg(not(feature = "python"))]
use crate::monte_carlo::MonteCarloResult;
use crate::types::BacktestResult;
#[cfg(feature = "python")]
use walkforward::{CPCVConfig, WalkForwardAnalyzer, WalkForwardConfig};

/// Fraction of each walk-forward window treated as training data (the
/// remainder is the OOS test slice). Shared by `run_walk_forward` and Stage
/// 1's subsampling boundary below — both must agree on where "training data"
/// ends, or Stage 1's quick sanity check silently peeks at data Stage 2 is
/// supposed to hold out.
/// Public so the multi-venue walk-forward path (`program/src/worker.rs`'s
/// `execute_multi_venue_python_validation`) can mirror the exact same
/// train/test split fraction instead of risking drift from a duplicated
/// constant (the same lesson `wf_windows`'s earlier cap drift taught).
pub const WF_TRAIN_FRAC: f64 = 0.7;

/// Hard disqualification threshold for GA fitness -- see
/// [`genetic::exceeds_param_complexity_gate`]'s doc comment for the full
/// rationale. Re-exported here (rather than duplicated) so this module and
/// `program::ga_eval_worker` (the process-pool path that's actually
/// authoritative whenever a `WorkerPool` is spawned) can never drift apart --
/// see quant council 2026-07-27 follow-up.
use genetic::exceeds_param_complexity_gate;

/// Index (exclusive) up to which Stage 1's quick sanity check may draw data,
/// given the full requested range's length. Pure function so the boundary
/// arithmetic is unit-testable without needing a live Python/PyO3 environment
/// (which the platform's own quick-backtest execution requires). Always
/// returns at least 1 (never an empty slice) and never more than `total_len`.
fn stage1_training_only_end(total_len: usize) -> usize {
    (((total_len as f64) * WF_TRAIN_FRAC) as usize).clamp(1, total_len.max(1)).min(total_len)
}

/// Auto-resolves the walk-forward window count when a job leaves
/// `wf_windows` at the platform-wide "auto" sentinel (`0`). Shared by
/// `run_validation_pipeline` (single-venue, below) and
/// `program::worker::execute_multi_venue_python_validation` (multi-venue) so
/// the two can never drift apart.
///
/// Quant council ruling (2026-07-29): window count purely from bar count
/// conflates two distinct things — statistical power per window (needs
/// enough bars for the train/test split to mean anything) and regime
/// diversity across windows (needs enough calendar time for each window to
/// plausibly span a distinct market regime). A 6-month backtest on 1-minute
/// bars and a 6-year backtest on daily bars can have the same bar count but
/// sample wildly different amounts of real market history — bar count alone
/// would size both identically. Take the calendar-time target (~1 window per
/// quarter) capped by the bar-count floor — whichever constraint is tighter
/// wins. Pure function so the arithmetic is unit-testable without a live
/// Python/PyO3 environment or real market data. Public so
/// `program::worker`'s multi-venue path can call the exact same logic.
///
/// 2026-08-07 follow-up: the calendar-time target above was always only a
/// *proxy* for regime diversity (more calendar time -> statistically more
/// likely to span multiple regimes), never actual regime detection. That
/// leaves a real gap: even a date range that genuinely contains multiple
/// volatility regimes can still have every OOS test window land inside a
/// single contiguous regime block (e.g. the most recent year happened to be
/// one long low-vol run), silently defeating the entire point of
/// walk-forward. `close_prices` (or, for a stat-arb leg pair/basket, the
/// SPREAD series -- not either leg's raw price, since that's what the
/// strategy actually trades and what can regime-shift under it) is optional
/// because not every call site has a clean single-series representation in
/// scope; when present it raises the target to at least the number of
/// distinct regime segments actually observed, still bounded by the same
/// `[3, 12]` clamp and bar-count ceiling as before -- this can only raise
/// the count within the existing safety rails, never bypass them.
pub fn resolve_auto_wf_windows(calendar_days: usize, data_len: usize, close_prices: Option<&[f64]>) -> usize {
    let mut calendar_target = (calendar_days / 90).clamp(3, 12);
    if let Some(prices) = close_prices {
        let returns = simple_returns(prices);
        let regimes = quant_diagnostics::volatility_tercile_regimes(&returns, REGIME_WINDOW);
        let segments = regime_segment_count(&regimes);
        calendar_target = calendar_target.max(segments.min(12));
    }
    let bar_ceiling = if data_len < 1000 { 3 } else { 8usize.min(data_len / 500) };
    calendar_target.min(bar_ceiling).max(3)
}

/// Trailing window (in bars) for the regime classification `resolve_auto_wf_windows`
/// feeds into `quant_diagnostics::volatility_tercile_regimes`. Mirrors
/// `program::volatility_regime_service::DEFAULT_WINDOW` (private to that
/// crate, so duplicated here by value) so the same "20-bar trailing realized
/// volatility" convention is used everywhere the platform classifies
/// regimes, not a second, silently-different lookback. `pub` so
/// `program::worker`'s walk-forward call sites can classify regimes with
/// the exact same window instead of a second, silently-driftable copy.
pub const REGIME_WINDOW: usize = 20;

/// Simple (non-log) percent-change return series from a price series. `pub`
/// for the same cross-crate reuse reason as `REGIME_WINDOW`.
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Contiguous regime **segments** -- maximal runs of the same trailing
/// volatility-tercile label -- across an already-classified regime series.
/// `None` entries (not enough trailing history yet to classify) are skipped
/// without resetting the running label or extending the current segment's
/// span, so a gap mid-series doesn't spuriously split or merge two segments
/// of the same regime around it. `end_idx` is inclusive (the last classified
/// index carrying that label), matching how callers use it to find a
/// representative bar inside the segment, not an exclusive Rust-range bound.
fn regime_segments(
    regimes: &[Option<quant_diagnostics::VolatilityRegime>],
) -> Vec<(usize, usize, quant_diagnostics::VolatilityRegime)> {
    let mut segments = Vec::new();
    let mut current: Option<(usize, usize, quant_diagnostics::VolatilityRegime)> = None;
    for (i, &r) in regimes.iter().enumerate() {
        if let Some(label) = r {
            match current {
                Some((start, _, prev_label)) if prev_label == label => {
                    current = Some((start, i, label));
                }
                Some(seg) => {
                    segments.push(seg);
                    current = Some((i, i, label));
                }
                None => {
                    current = Some((i, i, label));
                }
            }
        }
    }
    if let Some(seg) = current {
        segments.push(seg);
    }
    segments
}

/// Counts contiguous regime segments -- see `regime_segments`' doc comment
/// for the exact definition. Kept as a thin wrapper (rather than duplicating
/// the walk) for `resolve_auto_wf_windows`'s hot path, which only needs the
/// count, not the spans.
fn regime_segment_count(regimes: &[Option<quant_diagnostics::VolatilityRegime>]) -> usize {
    regime_segments(regimes).len()
}

/// Resolve the actual list of walk-forward window start offsets, ascending.
/// Each window spans `[offset, offset + window_size)` (train + purge gap +
/// test, contiguous); its TEST slice is the last `test_size` bars of that
/// span.
///
/// When `regimes` is `Some`, up to one window per DISTINCT regime label
/// actually present in the data (Low/Medium/High -- there are only ever
/// three) is reserved first, scarcest label first -- so a common regime's
/// placement can never accidentally consume the only viable slot for a
/// scarce one -- before the remaining window budget is filled with the
/// existing uniform, back-to-back spacing scheme. This is the placement half
/// of the regime-diversity work `resolve_auto_wf_windows` already does for
/// window *count*; see that function's 2026-08-07 doc comment for the gap
/// this closes (count alone doesn't guarantee any window's TEST slice
/// actually lands in an under-represented regime).
///
/// `regimes: None` reproduces the prior uniform-only placement exactly --
/// no behavior change for any caller that doesn't pass regime data.
///
/// Windows are kept mutually disjoint throughout (both the regime-reserved
/// ones and the uniform fill) -- overlapping OOS test windows correlate
/// their errors and overstate how much independent evidence walk-forward
/// actually gathered (see `resolve_wf_step`'s doc comment). This is a
/// reasonable, not perfectly optimal, interval packer: a regime segment near
/// the very end of the data can still get clipped by the `max_offset`
/// bound, and a real-but-too-short segment (fewer bars than `test_size`
/// could ever fit) is skipped rather than forced -- same "untested" outcome
/// `regime_coverage_narrative` already reports today, just now because
/// coverage is genuinely infeasible rather than an oversight.
pub fn resolve_wf_window_offsets(
    n: usize,
    window_size: usize,
    num_windows: usize,
    test_size: usize,
    regimes: Option<&[Option<quant_diagnostics::VolatilityRegime>]>,
) -> Vec<usize> {
    if num_windows == 0 || window_size == 0 || window_size > n {
        return Vec::new();
    }
    let max_offset = n - window_size;
    let uniform_step = resolve_wf_step(n, window_size, num_windows);

    let Some(regimes) = regimes else {
        return (0..num_windows).map(|i| (i * uniform_step).min(max_offset)).collect();
    };

    use quant_diagnostics::VolatilityRegime::{High, Low, Medium};
    let mut present: Vec<quant_diagnostics::VolatilityRegime> = [Low, Medium, High]
        .into_iter()
        .filter(|label| regimes.iter().any(|r| *r == Some(*label)))
        .collect();
    present.sort_by_key(|label| regimes.iter().filter(|r| **r == Some(*label)).count());

    let segments = regime_segments(regimes);
    let overlaps = |claimed: &[(usize, usize)], start: usize, end: usize| {
        claimed.iter().any(|(cs, ce)| start < *ce && end > *cs)
    };

    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut reserved_offsets: Vec<usize> = Vec::new();

    for label in present.into_iter().take(num_windows) {
        // Prefer the LARGEST segment of this label -- more robust than a
        // segment so short it's likely a one-bar classifier blip.
        let mut candidates: Vec<&(usize, usize, quant_diagnostics::VolatilityRegime)> =
            segments.iter().filter(|(_, _, l)| *l == label).collect();
        candidates.sort_by_key(|(s, e, _)| std::cmp::Reverse(e.saturating_sub(*s)));

        for (seg_start, seg_end, _) in candidates {
            // Place the window so its test slice's midpoint sits inside the
            // segment: test slice = [offset + window_size - test_size,
            // offset + window_size), midpoint ~= offset + window_size -
            // test_size/2. Solving for offset given a target midpoint:
            let target_mid = (seg_start + seg_end) / 2;
            let offset = (target_mid + test_size / 2).saturating_sub(window_size).min(max_offset);
            let end = offset + window_size;
            if overlaps(&claimed, offset, end) {
                continue;
            }
            claimed.push((offset, end));
            reserved_offsets.push(offset);
            break;
        }
        // No viable placement found for this (real but too-short, or fully
        // claimed-over) regime -- skip it, matching the pre-existing
        // "genuinely untestable" outcome.
    }

    let mut offsets = reserved_offsets;
    let mut candidate = 0usize;
    while offsets.len() < num_windows && candidate <= max_offset {
        let end = candidate + window_size;
        if !overlaps(&claimed, candidate, end) {
            claimed.push((candidate, end));
            offsets.push(candidate);
        }
        candidate += uniform_step.max(1);
    }

    offsets.sort_unstable();
    offsets
}

/// Per-bar close/reference price series from `MarketData`, mirroring the
/// exact per-variant match `backtest::engine::CandleSimulationEngine::get_reference_price`
/// already uses for a single price -- applied here across every bar instead
/// of just the first, to feed `resolve_auto_wf_windows`'s regime-diversity
/// signal. `pub` so `program::worker`'s call sites can build the same
/// signal from their own `Vec<MarketData>` without duplicating the match.
pub fn close_prices_from_market_data(data: &[MarketData]) -> Vec<f64> {
    data.iter()
        .filter_map(|d| match d {
            MarketData::Candle(c) => Some(c.close),
            MarketData::Trade(t) => Some(t.price),
            MarketData::PoolSwap(s) => Some(s.price()),
            MarketData::Generic(g) => Some(g.price),
            MarketData::OptionCandle(c) => Some(c.close),
        })
        .collect()
}

/// The predominant regime label spanning a window's test slice -- the label
/// at the slice's midpoint bar (not a full mode computation; windows are
/// disjoint and appropriately sized after the A1 full-range-coverage fix, so
/// a window's span rarely straddles more than one real regime segment,
/// making the midpoint a fair representative without extra bookkeeping).
/// `regimes` is indexed relative to the returns series (one shorter than
/// the original bar series `test_start_idx`/`test_end_idx` come from) --
/// close enough for a coarse coverage report that this doesn't warrant
/// exact realignment.
pub fn window_regime_label(
    regimes: &[Option<quant_diagnostics::VolatilityRegime>],
    test_start_idx: usize,
    test_end_idx: usize,
) -> Option<quant_diagnostics::VolatilityRegime> {
    let mid = test_start_idx + test_end_idx.saturating_sub(test_start_idx) / 2;
    regimes.get(mid).copied().flatten()
}

/// Narrative summarizing which volatility regimes the OOS test windows
/// actually covered, versus which regimes were genuinely present in the
/// full dataset. Even with A1's full-range-coverage fix, a genuinely
/// regime-poor dataset (e.g. one continuous low-vol run) will still only
/// ever produce Low-regime test windows -- worth saying plainly rather than
/// leaving the AI/user to assume "walk-forward passed" means "robust across
/// market conditions." Returns an empty string when no regime could be
/// classified at all (e.g. too little data), so callers can omit the field
/// entirely rather than surface a content-free line.
///
/// `tested_regimes` is one entry per completed window (its `test_regime`,
/// e.g. from `WalkForwardWindow` or an equivalent per-window accumulator for
/// call sites -- like the pairs/basket walk-forward -- that track window
/// results as scalars rather than a `WalkForwardWindow` collection).
pub fn regime_coverage_narrative(
    tested_regimes: &[Option<quant_diagnostics::VolatilityRegime>],
    all_regimes: &[Option<quant_diagnostics::VolatilityRegime>],
) -> String {
    use quant_diagnostics::VolatilityRegime;
    let label_name = |r: VolatilityRegime| match r {
        VolatilityRegime::Low => "Low",
        VolatilityRegime::Medium => "Medium",
        VolatilityRegime::High => "High",
    };
    let present: Vec<VolatilityRegime> = [VolatilityRegime::Low, VolatilityRegime::Medium, VolatilityRegime::High]
        .into_iter()
        .filter(|label| all_regimes.iter().any(|r| *r == Some(*label)))
        .collect();
    if present.is_empty() {
        return String::new();
    }

    let tested_counts: Vec<(VolatilityRegime, usize)> = present
        .iter()
        .map(|&label| (label, tested_regimes.iter().filter(|&&r| r == Some(label)).count()))
        .filter(|(_, count)| *count > 0)
        .collect();

    let tested_summary = tested_counts
        .iter()
        .map(|(label, count)| format!("{} ({})", label_name(*label), count))
        .collect::<Vec<_>>()
        .join(", ");

    if tested_counts.len() == present.len() {
        format!(
            "OOS windows covered: {} -- all {} detected regime(s) represented.",
            tested_summary,
            present.len()
        )
    } else {
        let untested: Vec<&str> = present
            .iter()
            .filter(|label| !tested_counts.iter().any(|(l, _)| l == *label))
            .map(|&label| label_name(label))
            .collect();
        format!(
            "OOS windows covered: {} only -- {} volatility regime(s) present in this data were never tested out-of-sample.",
            tested_summary,
            untested.join("/")
        )
    }
}

/// Structured counterpart to `regime_coverage_narrative`: `true` when at
/// least one volatility regime genuinely present in `all_regimes` was never
/// tested out-of-sample per `tested_regimes`. Exists so callers that need to
/// GATE on coverage (e.g. downgrading a Promising verdict) don't have to
/// parse the prose narrative -- see `regime_coverage_narrative`'s own doc
/// comment for the shared `present`/`tested_counts` logic this mirrors.
/// `false` when regime classification wasn't available at all (nothing
/// "present" to have missed), same as the narrative's empty-string case.
pub fn regime_coverage_incomplete(
    tested_regimes: &[Option<quant_diagnostics::VolatilityRegime>],
    all_regimes: &[Option<quant_diagnostics::VolatilityRegime>],
) -> bool {
    use quant_diagnostics::VolatilityRegime;
    let present: Vec<VolatilityRegime> = [VolatilityRegime::Low, VolatilityRegime::Medium, VolatilityRegime::High]
        .into_iter()
        .filter(|label| all_regimes.iter().any(|r| *r == Some(*label)))
        .collect();
    if present.is_empty() {
        return false;
    }
    present
        .iter()
        .any(|&label| !tested_regimes.iter().any(|&r| r == Some(label)))
}

// ---------------------------------------------------------------------------
// Cost hurdle pre-flight check (systematic elimination funnel, quant council)
// ---------------------------------------------------------------------------

/// Result of the cost-hurdle pre-flight check — the cheapest possible filter
/// in the elimination funnel. Runs before Stage 0, using only the requested
/// price series and the job's fee assumptions: no Python execution, no GA,
/// no walk-forward. An idea whose raw price movement can't plausibly clear
/// round-trip costs is dead regardless of what a full backtest would show.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostHurdleResult {
    pub passed: bool,
    /// Average absolute per-bar price move, in basis points.
    pub avg_move_bps: f64,
    /// Estimated round-trip cost (2 × taker fee + slippage), in basis points.
    pub cost_bps: f64,
    /// Multiple `avg_move_bps` must clear `cost_bps` by — scaled up for
    /// higher-frequency bars, since compounding costs and less reliable
    /// slippage estimates intraday make a flat multiple understate real risk.
    pub required_multiple: f64,
    /// `avg_move_bps / cost_bps` — how many multiples of cost the raw edge
    /// actually clears (compare against `required_multiple`).
    pub margin: f64,
}

/// Required cost-clearance multiple for a given average bar spacing. Higher
/// frequency (shorter bars) needs a bigger safety margin: at 15-minute bars
/// a strategy trading often pays round-trip costs many times a week, and a
/// flat 2-3x margin (reasonable for daily bars) understates how fast that
/// compounds. Pure function, unit-testable without live data.
///
/// `pub` (not feature-gated) so the research interview's asset-diagnostics
/// preflight (`program::asset_diagnostics_service`) can run this exact same
/// check before a backtest is even submitted, using the single source of
/// truth instead of a separately-drifting copy.
pub fn required_cost_multiple(avg_bar_minutes: f64) -> f64 {
    if avg_bar_minutes <= 15.0 {
        4.0
    } else if avg_bar_minutes <= 240.0 {
        3.0
    } else {
        2.5
    }
}

/// Extract a single closing/last price from one `MarketData` element. Mirrors
/// the match arms `python_simulation.rs`'s (private, unexported)
/// `extract_price_volume_ts` already uses — duplicated here rather than
/// exported across the module boundary for a one-field-deep read. Only
/// called from the feature-gated `check_cost_hurdle`.
#[cfg(feature = "python")]
fn market_data_close_price(md: &MarketData) -> Option<f64> {
    match md {
        MarketData::Trade(t) => Some(t.price),
        MarketData::Candle(c) => Some(c.close),
        MarketData::PoolSwap(s) => Some(s.price()),
        MarketData::Generic(g) => Some(g.close.unwrap_or(g.price)),
        MarketData::OptionCandle(c) => Some(c.close),
    }
}

/// Extract the timestamp from one `MarketData` element — `MarketData` has no
/// inherent `.timestamp()` method, so this mirrors the same per-variant match
/// `extract_price_volume_ts` uses. `DateTime`/`Utc` are only imported under
/// the `python` feature in this file (matching `run_validation_pipeline`'s
/// own gate), so this and its caller are gated the same way.
#[cfg(feature = "python")]
fn market_data_timestamp(md: &MarketData) -> DateTime<Utc> {
    match md {
        MarketData::Trade(t) => t.timestamp,
        MarketData::Candle(c) => c.timestamp,
        MarketData::PoolSwap(s) => s.timestamp,
        MarketData::Generic(g) => DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or_else(Utc::now),
        MarketData::OptionCandle(c) => c.timestamp,
    }
}

/// Core cost-hurdle arithmetic, separated from data extraction so it's
/// unit-testable with plain `Vec<f64>` fixtures instead of constructing
/// `MarketData`. `avg_bar_minutes` should reflect the actual spacing of the
/// requested series (derived from data timestamps, not a trusted config
/// field) so the required multiple matches the real granularity.
///
/// `pub` for the same reason as `required_cost_multiple` above.
pub fn compute_cost_hurdle(prices: &[f64], avg_bar_minutes: f64, taker_fee: f64, slippage_bps: f64) -> Option<CostHurdleResult> {
    if prices.len() < 2 {
        return None; // not enough data to estimate a move at all
    }
    let abs_returns: Vec<f64> = prices
        .windows(2)
        .filter_map(|w| if w[0] > 0.0 { Some(((w[1] - w[0]) / w[0]).abs()) } else { None })
        .collect();
    if abs_returns.is_empty() {
        return None;
    }
    let avg_move_bps = (abs_returns.iter().sum::<f64>() / abs_returns.len() as f64) * 10_000.0;
    let cost_bps = 2.0 * taker_fee * 10_000.0 + slippage_bps;
    let required_multiple = required_cost_multiple(avg_bar_minutes);
    let margin = if cost_bps > 0.0 { avg_move_bps / cost_bps } else { f64::INFINITY };
    Some(CostHurdleResult {
        passed: margin >= required_multiple,
        avg_move_bps,
        cost_bps,
        required_multiple,
        margin,
    })
}

/// Full cost-hurdle check against real `MarketData` — extracts prices and the
/// average bar spacing from the data itself, then delegates to
/// `compute_cost_hurdle` for the arithmetic.
#[cfg(feature = "python")]
fn check_cost_hurdle(market_data: &[MarketData], fee_config: Option<&ExchangeFeeConfig>, slippage_bps: f64) -> Option<CostHurdleResult> {
    if market_data.len() < 2 {
        return None;
    }
    let prices: Vec<f64> = market_data.iter().filter_map(market_data_close_price).collect();
    let first_ts = market_data_timestamp(market_data.first()?);
    let last_ts = market_data_timestamp(market_data.last()?);
    let span_minutes = (last_ts - first_ts).num_seconds() as f64 / 60.0;
    let avg_bar_minutes = if market_data.len() > 1 { (span_minutes / (market_data.len() - 1) as f64).max(0.0) } else { 0.0 };
    let taker_fee = fee_config.map(|f| f.taker_fee).unwrap_or(0.001);
    compute_cost_hurdle(&prices, avg_bar_minutes, taker_fee, slippage_bps)
}

// ---------------------------------------------------------------------------
// Cross-sectional consistency check (systematic elimination funnel, quant
// council, expanded scope) — after Stage 1 passes for a single-asset job,
// optionally re-run the same strategy against a small basket of related
// assets to see whether the edge is broadly consistent or narrow to the one
// submitted asset. Warning-only, never blocks the pipeline. Opt-in via
// ENABLE_CROSS_SECTIONAL_CHECK (default off) since it's a genuine extra cost
// (fetching other assets' data) for a nominally single-asset job.
// ---------------------------------------------------------------------------

fn cross_sectional_check_enabled() -> bool {
    std::env::var("ENABLE_CROSS_SECTIONAL_CHECK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(feature = "python")]
fn market_data_symbol_exchange(md: &MarketData) -> Option<(String, String)> {
    match md {
        MarketData::Candle(c) => Some((c.symbol.to_string(), c.exchange.to_string())),
        MarketData::Trade(t) => Some((t.symbol.to_string(), t.exchange.to_string())),
        // Uploaded custom data, DEX pool swaps, and option contracts (no
        // single "exchange" -- a listed option trades on a consolidated
        // tape across venues, unlike a per-exchange crypto pair) carry no
        // venue-symbol identity — there is no meaningful "related asset"
        // basket for them.
        MarketData::PoolSwap(_) | MarketData::Generic(_) | MarketData::OptionCandle(_) => None,
    }
}

/// No existing concept of "top-N assets by volume per asset class" exists in
/// this codebase yet (checked) — these are hardcoded starter baskets per
/// asset class, inferred from the submitted symbol's shape. A real ranked
/// universe is a follow-up once this advisory check proves useful in
/// practice.
const CROSS_SECTIONAL_CRYPTO_BASKET: &[&str] = &["BTC-USD", "ETH-USD", "SOL-USD"];
const CROSS_SECTIONAL_FOREX_BASKET: &[&str] = &["EUR-USD", "GBP-USD", "USD-JPY"];
const CROSS_SECTIONAL_EQUITY_BASKET: &[&str] = &["AAPL", "MSFT", "GOOGL"];
const CROSS_SECTIONAL_FOREX_CODES: &[&str] =
    &["EUR", "GBP", "USD", "JPY", "AUD", "CAD", "CHF", "NZD"];
const CROSS_SECTIONAL_CRYPTO_TICKERS: &[&str] = &[
    "BTC", "XBT", "ETH", "SOL", "ADA", "DOGE", "XRP", "AVAX", "MATIC", "DOT", "LINK", "LTC", "BCH",
];

fn cross_sectional_basket_for(symbol: &str) -> &'static [&'static str] {
    let norm = symbol.to_uppercase().replace('/', "-");
    let parts: Vec<&str> = norm.split('-').collect();
    if parts.len() == 2
        && CROSS_SECTIONAL_FOREX_CODES.contains(&parts[0])
        && CROSS_SECTIONAL_FOREX_CODES.contains(&parts[1])
    {
        return CROSS_SECTIONAL_FOREX_BASKET;
    }
    if let Some(base) = parts.first() {
        if CROSS_SECTIONAL_CRYPTO_TICKERS.contains(base) {
            return CROSS_SECTIONAL_CRYPTO_BASKET;
        }
    }
    CROSS_SECTIONAL_EQUITY_BASKET
}

/// Fetch a basket asset's candles and run the same cheap, subsampled quick
/// backtest Stage 1 uses. Returns `Ok(None)` when the asset has no data
/// (e.g. unsupported/delisted on this venue) so callers can distinguish
/// "nothing to compare" from "compared and found no edge."
#[cfg(feature = "python")]
async fn fetch_and_quick_backtest(
    exchange: &str,
    symbol: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
    candle_interval_minutes: i64,
    config: &ValidationConfig,
) -> Result<Option<BacktestResult>> {
    if std::env::var("MASSIVE_API_KEY").unwrap_or_default().is_empty() {
        return Ok(None);
    }

    let massive_exchange = dataloader::Exchange::from_str(exchange);
    let granularity = dataloader::CandleGranularity::from_minutes(candle_interval_minutes);
    let request = dataloader::DataRequest::new(massive_exchange, symbol, start_date, end_date)
        .with_granularity(granularity);

    let provider = dataloader::create_provider_for(dataloader::ProviderKind::for_exchange(exchange))
        .map_err(|e| anyhow::anyhow!("failed to create data provider: {}", e))?;

    let data = match provider.fetch(&request).await {
        Ok(d) => d,
        Err(e) => {
            log_warn!(BACKTEST_LOGGER, "[VALIDATION] Cross-sectional check: fetch failed for {} on {}: {}", symbol, exchange, e);
            return Err(anyhow::anyhow!("fetch error: {}", e));
        }
    };
    if data.is_empty() {
        return Ok(None);
    }

    // Smaller cap than Stage 1's 150K since this runs once per basket asset,
    // on top of the main pipeline's own compute.
    let step = (data.len() / 50_000).max(1);
    let quick: std::borrow::Cow<'_, [MarketData]> = if step > 1 {
        std::borrow::Cow::Owned(data.iter().step_by(step).cloned().collect())
    } else {
        std::borrow::Cow::Owned(data)
    };

    let result = run_single_backtest(&quick, config).await?;
    Ok(Some(result))
}

/// Run the cross-sectional consistency check for the asset backing
/// `market_data`, if enabled and if the data carries venue-symbol identity.
/// Returns `None` when disabled, when the asset/venue can't be identified,
/// or when no basket asset could be fetched for comparison — silence here
/// means "nothing to report," never a failure signal.
#[cfg(feature = "python")]
async fn run_cross_sectional_check(
    market_data: &[MarketData],
    config: &ValidationConfig,
) -> Option<StageVerdict> {
    if !cross_sectional_check_enabled() {
        return None;
    }

    let (symbol, exchange) = market_data.iter().find_map(market_data_symbol_exchange)?;
    let first_md = market_data.first()?;
    let last_md = market_data.last()?;
    let first_ts = market_data_timestamp(first_md);
    let last_ts = market_data_timestamp(last_md);
    let span_minutes = (last_ts - first_ts).num_seconds() as f64 / 60.0;
    let avg_bar_minutes = if market_data.len() > 1 {
        ((span_minutes / (market_data.len() - 1) as f64).max(1.0)).round() as i64
    } else {
        60
    };
    let start_date = first_ts.date_naive();
    let end_date = last_ts.date_naive();

    let norm_symbol = symbol.to_uppercase().replace('/', "-");
    let basket = cross_sectional_basket_for(&symbol);
    let candidates: Vec<&str> = basket
        .iter()
        .copied()
        .filter(|s| *s != norm_symbol)
        .take(3)
        .collect();
    if candidates.is_empty() {
        return None;
    }

    let mut attempted = 0usize;
    let mut consistent = 0usize;
    let mut fetch_failures = 0usize;

    for basket_symbol in &candidates {
        match fetch_and_quick_backtest(&exchange, basket_symbol, start_date, end_date, avg_bar_minutes, config).await {
            Ok(Some(result)) => {
                attempted += 1;
                if result.num_trades > 0 && result.total_pnl > 0.0 {
                    consistent += 1;
                }
            }
            Ok(None) => {
                // No data for this basket asset on this venue — skip silently.
            }
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] Cross-sectional check: {} failed: {}", basket_symbol, e);
                fetch_failures += 1;
            }
        }
    }

    if attempted == 0 {
        return None;
    }

    let ratio = consistent as f64 / attempted as f64;
    let message = if ratio >= 0.6 {
        format!(
            "Edge appears broadly consistent: {}/{} related assets ({}) also showed a profitable edge with this strategy — less likely to be overfit to {} specifically.",
            consistent, attempted, candidates.join(", "), norm_symbol,
        )
    } else if ratio == 0.0 {
        format!(
            "Edge appears narrow: 0/{} related assets ({}) showed a profitable edge with this same strategy. Consistent with either overfitting to {} specifically, or a real but asset-specific structural effect (a single-exchange quirk, roll/expiry pattern, etc.) — worth distinguishing before trusting this result.{}",
            attempted, candidates.join(", "), norm_symbol,
            if fetch_failures > 0 { format!(" ({} basket asset(s) could not be fetched for comparison.)", fetch_failures) } else { String::new() },
        )
    } else {
        format!(
            "Mixed cross-sectional result: {}/{} related assets ({}) showed a profitable edge with this same strategy — partial consistency, not universal. Worth checking whether the assets that failed are thin/illiquid (a capacity/liquidity issue) rather than a purely statistical one.",
            consistent, attempted, candidates.join(", "),
        )
    };

    Some(StageVerdict {
        stage: 1,
        name: "Cross-Sectional Consistency".into(),
        passed: true,
        message,
    })
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One leg of a pre-configured multi-leg option spread -- see
/// `ValidationConfig.option_spread` / `python_simulation::PythonSimConfig.
/// option_spread`. Every leg carries its OWN price series, aligned tick-
/// for-tick (same length and timestamp order) with the primary market_data
/// series and every other leg's series here -- the caller
/// (`options_backtest_data` in the `program` crate) is responsible for that
/// alignment before constructing this; `python_simulation::i8_to_signals`/
/// the `OptionSpread` fill path both index into `prices` by the SAME
/// `tick_idx` the primary series uses, with no independent timestamp
/// matching of their own.
///
/// Defined here (not in `python_simulation`, which is gated behind the
/// `python` feature) because `ValidationConfig` below is NOT feature-gated
/// and needs this type unconditionally.
///
/// Scope: this platform's `job_queue::OptionLeg` array (and therefore
/// `option_legs.len()`) already supports any leg count, but only 2-leg
/// (vertical) spreads are wired end-to-end today -- `options_backtest_data`
/// only resolves and aligns 2 legs. 3+-leg structures (iron condors,
/// butterflies) are a real, larger follow-up (N-way series alignment,
/// N-way margin/risk treatment) deliberately not attempted here.
#[derive(Debug, Clone)]
pub struct OptionSpreadLeg {
    pub instrument: derivatives::DerivativeMetadata,
    /// Signed ratio: positive = long, negative = short -- same convention
    /// as `signal::SpreadLeg.ratio` and `job_queue::OptionLeg.ratio`.
    pub ratio: i32,
    /// This leg's own price at each tick, aligned 1:1 with the primary
    /// `market_data` series by index (not by re-matching timestamps at
    /// fill time).
    pub prices: Vec<f64>,
}

/// Configuration for the validation pipeline.
pub struct ValidationConfig {
    /// User Python source code.
    pub python_source: String,
    /// Backtest config.
    pub backtest_config: BacktestConfig,
    /// Exchange fee config.
    pub fee_config: Option<ExchangeFeeConfig>,
    /// Supplementary data.
    pub supplementary_data: HashMap<String, f64>,
    /// Strategy parameters.
    pub parameters: HashMap<String, ParameterValue>,
    /// Number of walk-forward windows (0 = auto, default ~8).
    pub wf_windows: usize,
    /// Number of Monte Carlo simulations (default 1000).
    pub mc_runs: usize,
    /// Minimum OOS Sharpe to proceed from Stage 2 → Stage 3.
    pub min_oos_sharpe: f64,
    /// Maximum P(ruin) before issuing a warning.
    pub max_ruin_probability: f64,
    /// Progress callback: (stage, stage_name, current_step, total_steps).
    pub progress_callback: Option<Arc<dyn Fn(u8, &str, usize, usize) + Send + Sync>>,
    /// If set, enables GA optimization (Stage 2.5). Built from Python's `parameter_space()`.
    pub parameter_schema: Option<Arc<ParameterSchema>>,
    /// GA config overrides (population, generations). Uses sensible defaults if None.
    pub ga_config: Option<GeneticConfig>,
    /// Scores GA candidates. Required to run GA optimization (Stage 2.5) --
    /// `backtest` ships no built-in formula, since "what makes a strategy
    /// good" is a scoring-policy decision for the caller to own. `None`
    /// panics with a clear message the moment a GA stage is actually reached
    /// (fine to leave `None` for pipeline runs that skip GA entirely). Named
    /// `fitness_scorer` (not `fitness_fn`) to avoid confusion with the
    /// existing `AsyncContextFitnessFn` closures this file builds internally.
    pub fitness_scorer: Option<Arc<dyn genetic::fitness::FitnessFunction>>,
    /// Optional population-level fitness post-processing hook (e.g. z-score
    /// normalization), forwarded to `AdaptiveGeneticOptimizer::fitness_normalizer`.
    pub fitness_normalizer: Option<Arc<dyn Fn(&mut [genetic::FitnessResult]) + Send + Sync>>,
    /// Opaque config forwarded verbatim to `genetic::worker_pool::WorkerConfig`'s
    /// own opaque `fitness_weights: serde_json::Value` field for out-of-process
    /// GA workers -- `fitness_scorer` (a trait object) can't cross a process
    /// boundary, so callers that spawn a worker pool must serialize whatever
    /// their `FitnessFunction` implementation needs here themselves; `backtest`
    /// never interprets its contents. Ignored when the worker pool isn't used
    /// (`GA_WORKERS<=1` or a single logical worker).
    pub fitness_scorer_config_json: serde_json::Value,
    /// Minimum trades for MC to be considered non-degenerate.
    pub mc_min_trades: usize,
    /// MC confidence level for VaR/CVaR (e.g. 0.95).
    pub mc_confidence_level: f64,
    /// Hard GA disqualification cap on max drawdown, as a fraction of initial
    /// capital. Candidates exceeding it receive fitness 0.0 regardless of
    /// Sharpe, mirroring the built-in engine's hard gate. Sourced from the
    /// user's stated drawdown tolerance; None applies the platform-wide 40%.
    pub max_drawdown_hard_cap: Option<f64>,
    /// Pre-computed historical options-derived IV overlay, sorted by
    /// `IvSurface.timestamp`, forwarded verbatim into every
    /// `run_single_backtest` call this config drives -- computed by the
    /// caller (self-computed from historical option-contract trade prices
    /// via Black-Scholes inversion; no vendor supplies historical IV
    /// directly, confirmed 2026-08-29), never by `backtest` itself, which
    /// has no data-provider access. `None` (the default) is the existing,
    /// unaffected behavior for every strategy that doesn't need this --
    /// gated by the caller on `data_requirements().needs_iv_surface`, not
    /// unconditionally computed for every job.
    pub historical_iv_surfaces: Option<Vec<derivatives::IvSurface>>,
    /// Same shape and same forwarding pattern as `historical_iv_surfaces`
    /// above: forwarded verbatim into every `run_single_backtest` call this
    /// config drives, which is the pipeline's central, widely-reused single-
    /// evaluation primitive (GA candidate scoring, walk-forward IS/OOS
    /// splits, quick pre-checks, the final full-rigor evaluation -- all of
    /// them route through it), so setting this once here reaches every
    /// stage. See `PythonSimConfig.option_instrument`'s doc comment for what
    /// it actually does. `None` (the default) is the existing, unaffected
    /// behavior for every non-options job.
    pub option_instrument: Option<derivatives::DerivativeMetadata>,
    /// Same forwarding pattern as `option_instrument` above, for a multi-leg
    /// spread instead of a single contract. See `crate::python_simulation::
    /// OptionSpreadLeg`'s doc comment for the exact scope (2-leg verticals
    /// only today).
    pub option_spread: Option<Vec<OptionSpreadLeg>>,
    /// Same forwarding pattern as `option_instrument`/`option_spread` above.
    /// See `PythonSimConfig.underlying_series`'s doc comment -- required for
    /// `update_position_greeks` to have anything to price Greeks against.
    pub underlying_series: Option<Vec<f64>>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            python_source: String::new(),
            backtest_config: BacktestConfig::default(),
            fee_config: None,
            supplementary_data: HashMap::new(),
            parameters: HashMap::new(),
            wf_windows: 0,
            mc_runs: 1000,
            min_oos_sharpe: 0.5,
            max_ruin_probability: 0.20,
            progress_callback: None,
            parameter_schema: None,
            ga_config: None,
            fitness_scorer: None,
            fitness_normalizer: None,
            fitness_scorer_config_json: serde_json::Value::Null,
            mc_min_trades: 30,
            mc_confidence_level: 0.95,
            max_drawdown_hard_cap: None,
            historical_iv_surfaces: None,
            option_instrument: None,
            option_spread: None,
            underlying_series: None,
        }
    }
}

/// Result of the full validation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Which stage(s) completed.
    pub stages_completed: u8,
    /// Stage 1: Quick backtest result.
    pub quick_backtest: Option<BacktestResult>,
    /// Stage 2: Walk-forward result.
    pub walk_forward: Option<WalkForwardResult>,
    /// Stage 3: Monte Carlo result (standard shuffle).
    pub monte_carlo: Option<MonteCarloResult>,
    /// Stage 3b: Block bootstrap Monte Carlo result.
    pub block_bootstrap: Option<MonteCarloResult>,
    /// Stage 3c: Regime-aware Monte Carlo result.
    pub regime_mc: Option<MonteCarloResult>,
    /// Stage 2.5: GA optimization report (only if parameter_space was defined).
    pub ga_report: Option<GAReport>,
    /// CPCV (Combinatorial Purged Cross-Validation) results.
    pub cpcv_pbo: Option<f64>,
    pub cpcv_deflated_sharpe: Option<f64>,
    pub cpcv_parameter_stability: Option<f64>,
    /// GA-optimized full backtest result (when GA ran).
    pub optimized_backtest: Option<BacktestResult>,
    /// Overall verdict: "pass", "fail", "warn".
    pub verdict: String,
    /// Human-readable summary of findings.
    pub summary: String,
    /// Per-stage verdicts.
    pub stage_verdicts: Vec<StageVerdict>,
    /// Whether look-ahead bias was detected (Stage 0 perturbation test).
    #[serde(default)]
    pub look_ahead_detected: Option<bool>,
    /// Total elapsed time in seconds.
    pub elapsed_seconds: f64,
}

/// Verdict for a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageVerdict {
    pub stage: u8,
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// Walk-forward analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    /// Per-window results (train metrics vs test metrics).
    pub windows: Vec<WalkForwardWindow>,
    /// Average out-of-sample Sharpe ratio.
    pub avg_oos_sharpe: f64,
    /// Average out-of-sample PnL.
    pub avg_oos_pnl: f64,
    /// Average in-sample (per-window training-slice) Sharpe ratio -- the fair
    /// apples-to-apples comparator for `avg_oos_sharpe`, since both are computed
    /// by the SAME per-window process: `run_walk_forward` on either the
    /// strategy's declared default parameters (Stage 2), or the GA winner's
    /// own fixed parameters (Stage 2.5's post-GA report -- see
    /// `ValidationConfig.parameters` at that call site). Deliberately NOT the
    /// separate global full-dataset GA's Sharpe (`optimized_backtest`/`in_sample_result`
    /// in worker.rs) -- that number is chosen from many more candidates over more
    /// data and isn't a meaningful "in-sample" baseline for this OOS estimate.
    pub avg_train_sharpe: f64,
    /// Performance degradation: (avg train Sharpe - avg test Sharpe) / avg train Sharpe.
    pub overfitting_ratio: f64,
    /// Consistency: 1 / (1 + σ_sharpe) — measures uniformity of OOS Sharpe across windows.
    /// Higher is better (1.0 = identical Sharpe every window).
    pub consistency_score: f64,
    /// Fraction of windows where OOS Sharpe > 0 (profitability rate).
    pub oos_profitability_rate: f64,
    /// Number of windows.
    pub num_windows: usize,
    /// Combined out-of-sample result from all windows (equity curve, trade log, trade returns).
    /// This is the ground-truth OOS result and should be used as the primary result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oos_combined_result: Option<BacktestResult>,
    /// Total real (uncached) GA fitness evaluations this walk-forward pass
    /// itself spent searching for parameters, distinct from evaluating a
    /// fixed set. Always `None` today: both call sites of
    /// `run_walk_forward` -- Stage 2 (default parameters) and the post-GA
    /// report (the winner's own fixed parameters, replacing the old
    /// per-window mini-GA re-optimization this field used to count) --
    /// evaluate one already-known parameter set per window, with no
    /// chromosome search of their own. Feeds into
    /// `GAReport::total_fresh_evaluations`, which now only reflects the
    /// main GA/NSGA-II run's own real evaluations (plus sensitivity/
    /// landscape sweeps) -- correctly smaller than before, since the
    /// removed mini-GA reruns no longer happen.
    #[serde(default)]
    pub ga_trial_count: Option<usize>,
    /// Which volatility regimes the OOS test windows actually covered, vs.
    /// which were present in the full dataset -- see `regime_coverage_narrative`'s
    /// doc comment. Empty string when no regime could be classified at all.
    #[serde(default)]
    pub regime_coverage_narrative: String,
    /// Structured counterpart to `regime_coverage_narrative` -- see
    /// `regime_coverage_incomplete`'s doc comment.
    #[serde(default)]
    pub regime_coverage_incomplete: bool,
}

/// A single walk-forward window result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardWindow {
    /// Window bounds (tick indices into the data array).
    pub train_start_idx: usize,
    pub train_end_idx: usize,
    pub test_start_idx: usize,
    pub test_end_idx: usize,
    /// Real calendar bounds for the same indices above, so a UI can place
    /// this window on an actual timeline instead of a meaningless bar
    /// count -- added alongside the *_idx fields rather than replacing them
    /// since every existing consumer (parameter_stability, regime labeling)
    /// keys off the indices. `None` on old, already-persisted results
    /// (`#[serde(default)]`) rather than backfilling, since the source bar
    /// data needed to compute them isn't available after the fact.
    #[serde(default)]
    pub train_start_date: Option<DateTime<Utc>>,
    #[serde(default)]
    pub train_end_date: Option<DateTime<Utc>>,
    #[serde(default)]
    pub test_start_date: Option<DateTime<Utc>>,
    #[serde(default)]
    pub test_end_date: Option<DateTime<Utc>>,
    /// Training period metrics.
    pub train_pnl: f64,
    pub train_sharpe: f64,
    pub train_trades: usize,
    /// Test (out-of-sample) period metrics.
    pub test_pnl: f64,
    pub test_sharpe: f64,
    pub test_trades: usize,
    /// Additional OOS metrics for reporting.
    pub test_win_rate: f64,
    pub test_max_drawdown: f64,
    pub test_profit_factor: f64,
    /// This window's predominant volatility regime (see `window_regime_label`).
    /// `None` when regime classification wasn't available for this run.
    #[serde(default)]
    pub test_regime: Option<quant_diagnostics::VolatilityRegime>,
}

// ---------------------------------------------------------------------------
// OOS result combination
// ---------------------------------------------------------------------------

/// Combine multiple out-of-sample BacktestResults (one per WF window) into a
/// single aggregated OOS result. Equity curves are chained end-to-end (each
/// window starts where the previous one ended), and trade logs / returns are
/// concatenated. Aggregate metrics are recomputed from the combined data.
/// Public so the multi-venue walk-forward path can combine its own per-window
/// OOS results with the exact same aggregation logic the single-venue path
/// already uses, instead of a second, potentially-diverging implementation.
pub fn combine_oos_results(results: &[BacktestResult], initial_capital: f64) -> BacktestResult {
    if results.is_empty() {
        return BacktestResult { initial_capital, ..Default::default() };
    }

    // Concatenate trade returns and trade logs across all windows
    let mut all_trade_returns = Vec::new();
    let mut all_trade_pct_returns = Vec::new();
    let mut all_trade_maes = Vec::new();
    let mut all_trade_log = Vec::new();
    let mut total_pnl = 0.0;
    let mut total_trades = 0usize;
    let mut closed_trades = 0usize;
    let mut open_positions = 0usize;
    let mut total_gross_profit = 0.0;
    let mut total_gross_loss = 0.0;
    let mut total_commission = 0.0;

    for r in results {
        all_trade_returns.extend_from_slice(&r.trade_returns);
        all_trade_pct_returns.extend_from_slice(&r.trade_pct_returns);
        all_trade_maes.extend_from_slice(&r.trade_maes);
        all_trade_log.extend_from_slice(&r.trade_log);
        total_pnl += r.total_pnl;
        total_trades += r.num_trades;
        closed_trades += r.closed_trades;
        open_positions += r.open_positions;
        total_gross_profit += r.gross_profit.unwrap_or(0.0);
        total_gross_loss += r.gross_loss.unwrap_or(0.0);
        total_commission += r.transaction_costs.as_ref().map(|tc| tc.total_commission).unwrap_or(0.0);
    }

    // Chain equity curves: each window starts where the previous ended
    let mut chained_equity = Vec::new();
    let mut offset = initial_capital;
    for r in results {
        if r.equity_curve.is_empty() { continue; }
        let window_start = r.equity_curve[0];
        let scale = if window_start.abs() > 1e-12 { offset / window_start } else { 1.0 };
        for &v in &r.equity_curve {
            chained_equity.push(v * scale);
        }
        if let Some(&last) = chained_equity.last() {
            offset = last;
        }
    }

    // Bug #34: Reconcile the headline P&L with the chained (multiplicative) equity
    // curve so total_pnl, net_profit, the equity-curve endpoint, max_drawdown, and
    // Monte Carlo all share ONE basis. Summing each window's dollar P&L additively
    // (`total_pnl += r.total_pnl`) treats every window as a fresh, independent
    // $initial_capital account, which overstates the realized result vs the
    // compounded curve by the volatility drag across windows (∏(1+rᵢ) < 1+Σrᵢ).
    // Derive total_pnl from the chained equity endpoint; fall back to the additive
    // sum only when no equity curve is available to chain.
    let additive_total_pnl = total_pnl;
    let total_pnl = match chained_equity.last() {
        Some(&last) => last - initial_capital,
        None => additive_total_pnl,
    };

    // Concatenate per-window price series so callers can build a buy-and-hold
    // benchmark aligned to the same OOS time domain as the chained equity curve.
    // Prices are raw asset values, so we just concatenate (no scaling).
    let chained_price_series: Vec<f64> = results.iter()
        .flat_map(|r| r.price_series.iter().copied())
        .collect();

    // Compute aggregate metrics from pct returns (correct for Sharpe, win rate, etc.)
    // Falls back to dollar returns if pct returns are unavailable (non-Python strategies).
    let stat_returns: &[f64] = if !all_trade_pct_returns.is_empty() {
        &all_trade_pct_returns
    } else {
        &all_trade_returns
    };
    let n = stat_returns.len();
    let win_count = stat_returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = if n > 0 { win_count as f64 / n as f64 } else { 0.0 };

    let profit_factor = if total_gross_loss.abs() > 1e-12 {
        total_gross_profit / total_gross_loss.abs()
    } else if total_gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let avg_return = if n > 0 {
        stat_returns.iter().sum::<f64>() / n as f64
    } else { 0.0 };

    // Sharpe ratio from combined pct returns
    let sharpe = if n > 1 {
        let mean = avg_return;
        let variance = stat_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();
        if std > 1e-12 { mean / std * 252.0_f64.sqrt() } else { 0.0 }
    } else { 0.0 };

    // Sortino ratio from combined pct returns
    let sortino = if n > 1 {
        let mean = avg_return;
        let downside_var = stat_returns.iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / n as f64;
        let downside_std = downside_var.sqrt();
        if downside_std > 1e-12 { mean / downside_std * 252.0_f64.sqrt() } else { 0.0 }
    } else { 0.0 };

    // Max drawdown from chained equity curve
    let max_drawdown = if !chained_equity.is_empty() {
        let mut peak = f64::NEG_INFINITY;
        let mut max_dd = 0.0f64;
        for &v in &chained_equity {
            if v > peak { peak = v; }
            let dd = if peak > 0.0 { (peak - v) / peak } else { 0.0 };
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    } else { 0.0 };

    // First/last timestamps and prices from trade log
    let first_ts = results.first().and_then(|r| r.first_trade_timestamp);
    let last_ts = results.last().and_then(|r| r.last_trade_timestamp);
    let first_price = results.first().and_then(|r| r.first_trade_price);
    let last_price = results.last().and_then(|r| r.last_trade_price);

    // Calmar from annualized return / max drawdown.
    // Annualize total return by years spanned (first_ts .. last_ts). Without this,
    // multi-year OOS windows produce calmar that's N× too large in magnitude.
    let calmar = if max_drawdown > 1e-12 && !chained_equity.is_empty() && initial_capital > 0.0 {
        let final_eq = chained_equity.last().copied().unwrap_or(initial_capital);
        let total_return = final_eq / initial_capital - 1.0;
        let years = match (first_ts, last_ts) {
            (Some(a), Some(b)) => ((b - a).num_seconds() as f64 / (365.25 * 86400.0)).max(1e-6),
            _ => 1.0,
        };
        // Compound annualization is mathematically correct for total_return > -1.
        // Fall back to simple division when total_return <= -1 (catastrophic loss).
        let annualized = if total_return > -1.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            total_return / years
        };
        Some(annualized / max_drawdown)
    } else { None };

    // Volatility of returns (from pct returns — annualized)
    let volatility = if n > 1 {
        let mean = avg_return;
        let var = stat_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt() * 252.0_f64.sqrt())
    } else { None };

    // Net profit
    let net_profit = if initial_capital > 0.0 {
        Some(total_pnl / initial_capital * 100.0)
    } else { None };

    BacktestResult {
        total_pnl,
        num_trades: total_trades,
        closed_trades,
        open_positions,
        first_trade_timestamp: first_ts,
        last_trade_timestamp: last_ts,
        first_trade_price: first_price,
        last_trade_price: last_price,
        realized_pnl: Some(total_pnl),
        unrealized_pnl: Some(0.0),
        max_drawdown,
        sharpe_ratio: Some(sharpe),
        sortino_ratio: Some(sortino),
        calmar_ratio: calmar,
        profit_factor: Some(profit_factor),
        win_rate: Some(win_rate),
        avg_trade_return: Some(avg_return),
        median_trade_return: if n > 0 {
            let mut sorted = stat_returns.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(sorted[n / 2])
        } else { None },
        avg_trade_duration: {
            let durations: Vec<f64> = results.iter()
                .filter_map(|r| r.avg_trade_duration)
                .collect();
            if !durations.is_empty() {
                Some(durations.iter().sum::<f64>() / durations.len() as f64)
            } else { None }
        },
        volatility,
        gross_profit: Some(total_gross_profit),
        gross_loss: Some(total_gross_loss),
        net_profit,
        equity_curve: chained_equity,
        price_series: chained_price_series,
        trade_returns: all_trade_returns,
        trade_pct_returns: all_trade_pct_returns,
        trade_maes: all_trade_maes,
        initial_capital,
        trade_log: all_trade_log,
        transaction_costs: Some(crate::types::TransactionCostAnalysis {
            total_commission,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            total_transaction_costs: total_commission,
            costs_as_percentage_of_pnl: if total_pnl.abs() > 1e-12 { total_commission / total_pnl.abs() * 100.0 } else { 0.0 },
            avg_cost_per_trade: if total_trades > 0 { total_commission / total_trades as f64 } else { 0.0 },
            commission_by_exchange: Default::default(),
            total_gas_costs_usd: 0.0,
            avg_dex_slippage_bps: 0.0,
        }),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/// Run the full staged validation pipeline.
///
/// Returns early if an earlier stage fails, saving compute.
#[cfg(feature = "python")]
pub async fn run_validation_pipeline(
    market_data: &[MarketData],
    config: ValidationConfig,
) -> Result<ValidationResult> {
    let timing_start = std::time::Instant::now();
    let mut stages: Vec<StageVerdict> = Vec::new();
    let mut verdict = "pass".to_string();

    let report_progress = |stage: u8, name: &str, step: usize, total: usize| {
        if let Some(cb) = &config.progress_callback {
            cb(stage, name, step, total);
        }
    };

    // -----------------------------------------------------------------------
    // Cost hurdle pre-flight check (systematic elimination funnel, quant
    // council) — the cheapest possible filter. Runs before even Stage 0's
    // lookahead perturbation test, since it needs only the raw price series
    // and fee assumptions, no Python execution at all. A strategy whose raw
    // price movement can't plausibly clear round-trip costs is dead
    // regardless of what a full backtest would show — skip the entire
    // pipeline rather than spend compute discovering that the expensive way.
    // Reuses the existing verdict/summary NOT_VIABLE surfacing
    // (build_api_result_from_validation already turns a "fail" verdict into
    // a user-visible failure banner) rather than a new structured field.
    // ---------------------------------------------------------------------
    report_progress(0, "cost_hurdle", 0, 1);
    if let Some(hurdle) = check_cost_hurdle(market_data, config.fee_config.as_ref(), config.backtest_config.trading.slippage_bps) {
        log_info!(BACKTEST_LOGGER, "[VALIDATION] Cost hurdle: avg_move_bps={:.2} cost_bps={:.2} required={:.1}x margin={:.2}x passed={}",
            hurdle.avg_move_bps, hurdle.cost_bps, hurdle.required_multiple, hurdle.margin, hurdle.passed);
        stages.push(StageVerdict {
            stage: 0,
            name: "Cost Hurdle".into(),
            passed: hurdle.passed,
            message: format!(
                "Avg move {:.1}bps vs round-trip cost {:.1}bps ({:.2}x margin, {:.1}x required)",
                hurdle.avg_move_bps, hurdle.cost_bps, hurdle.margin, hurdle.required_multiple,
            ),
        });
        report_progress(0, "cost_hurdle", 1, 1);
        if !hurdle.passed {
            return Ok(ValidationResult {
                stages_completed: 0,
                quick_backtest: None,
                walk_forward: None,
                monte_carlo: None,
                block_bootstrap: None,
                regime_mc: None,
                ga_report: None,
                optimized_backtest: None,
                cpcv_pbo: None,
                cpcv_deflated_sharpe: None,
                cpcv_parameter_stability: None,
                verdict: "fail".into(),
                summary: format!(
                    "Cost hurdle failed: this asset's average move ({:.1}bps per bar) doesn't clear round-trip trading costs ({:.1}bps) by the required {:.1}x margin — the edge would need to be unusually strong to survive real execution costs at this frequency.",
                    hurdle.avg_move_bps, hurdle.cost_bps, hurdle.required_multiple,
                ),
                stage_verdicts: stages,
                look_ahead_detected: None,
                elapsed_seconds: timing_start.elapsed().as_secs_f64(),
            });
        }
    } else {
        report_progress(0, "cost_hurdle", 1, 1);
    }

    // -----------------------------------------------------------------------
    // Stage 0: Look-ahead bias detection (perturbation test)
    // -----------------------------------------------------------------------
    report_progress(0, "lookahead_detection", 0, 1);
    log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 0: Look-ahead bias detection");

    let look_ahead_detected = match detect_lookahead_bias(market_data, &config).await {
        Ok(detected) => {
            if detected {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] LOOK-AHEAD BIAS DETECTED: strategy signals change when future data is present");
            } else {
                log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 0 passed: no look-ahead bias detected");
            }
            Some(detected)
        }
        Err(e) => {
            log_warn!(BACKTEST_LOGGER, "[VALIDATION] Stage 0 skipped (error): {}", e);
            None
        }
    };
    report_progress(0, "lookahead_detection", 1, 1);

    // -----------------------------------------------------------------------
    // Stage 1: Quick single-pass backtest (subsampled for speed)
    // -----------------------------------------------------------------------
    report_progress(1, "quick_backtest", 0, 1);
    log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 1: Quick backtest");

    // Training-only boundary: Stage 2's walk-forward treats the final
    // (1 - WF_TRAIN_FRAC) of the requested range as the most recent OOS test
    // window (run_walk_forward's last window's test slice ends at `n`). Stage
    // 1's quick sanity check must never see past this boundary — otherwise
    // its pass/fail decision (and, downstream, whether Stage 2/2.5/3 even run)
    // is partly made on data walk-forward is supposed to hold out.
    let train_only_end = stage1_training_only_end(market_data.len());
    let train_only_data = &market_data[..train_only_end];

    // Subsample to ~150K ticks for the quick sanity check.
    // The vectorized compute_signals() has a 30s timeout; at 1M+ ticks Python
    // cannot complete in time even with numpy.  150K ticks is enough to verify
    // the strategy is functional (generates trades / doesn't crash) while
    // keeping Stage 1 well within the timeout for any dataset size.
    // Full-resolution analysis happens in later stages.
    let quick_subsample_step = (train_only_data.len() / 150_000).max(1);
    let quick_data: std::borrow::Cow<'_, [MarketData]> = if quick_subsample_step > 1 {
        log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 1 subsampling (training-only, {:.0}% of range): every {}th tick ({} → ~{} ticks)", WF_TRAIN_FRAC * 100.0, quick_subsample_step, train_only_data.len(), train_only_data.len() / quick_subsample_step);
        std::borrow::Cow::Owned(train_only_data.iter().step_by(quick_subsample_step).cloned().collect())
    } else {
        std::borrow::Cow::Borrowed(train_only_data)
    };

    let quick_result = match run_single_backtest(&quick_data, &config).await {
        Ok(r) => r,
        Err(e) => {
            log_error!(BACKTEST_LOGGER, "[VALIDATION] Stage 1 failed: {:?} (data_points={}, source_len={})", e, quick_data.len(), config.python_source.len());
            return Err(e).context("Stage 1: quick backtest failed");
        }
    };

    let quick_pnl = quick_result.total_pnl;
    let quick_trades = quick_result.num_trades;
    // Stage 1 only gates on "did the strategy generate trades?" — a non-degenerate
    // strategy is always worth understanding via walk-forward and Monte Carlo, even
    // when it loses money. WF and MC answer "is this strategy real and consistent?"
    // not "is it profitable?" — that is the validation-as-a-service value proposition.
    // Profitability with tuned parameters is assessed by the GA stage when available.
    let quick_passed = quick_trades > 0;

    log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 1: data_points={} subsample_step={} trades={} pnl=${:.2} passed={}", quick_data.len(), quick_subsample_step, quick_trades, quick_pnl, quick_passed);

    stages.push(StageVerdict {
        stage: 1,
        name: "Quick Backtest".into(),
        passed: quick_passed,
        message: if quick_trades == 0 {
            "Strategy generated no trades".into()
        } else {
            format!("PnL=${:.2}, {} trades", quick_pnl, quick_trades)
        },
    });

    report_progress(1, "quick_backtest", 1, 1);

    if !quick_passed {
        return Ok(ValidationResult {
            stages_completed: 1,
            quick_backtest: Some(quick_result),
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict: "fail".into(),
            summary: "Strategy generated no trades — check entry/exit logic and date range.".into(),
            stage_verdicts: stages,
            look_ahead_detected,
            elapsed_seconds: timing_start.elapsed().as_secs_f64(),
        });
    }

    // -----------------------------------------------------------------------
    // Cross-sectional consistency check (advisory, opt-in) — Stage 1 passed,
    // so it's worth checking whether the edge shows up on related assets too
    // before spending the much larger walk-forward/GA/Monte Carlo compute.
    // Never blocks the pipeline; appends an informational stage verdict.
    // -----------------------------------------------------------------------
    if let Some(verdict) = run_cross_sectional_check(market_data, &config).await {
        log_info!(BACKTEST_LOGGER, "[VALIDATION] Cross-sectional check: {}", verdict.message);
        stages.push(verdict);
    }

    // -----------------------------------------------------------------------
    // Stage 2: Walk-Forward Analysis
    // -----------------------------------------------------------------------
    let num_windows = if config.wf_windows > 0 {
        config.wf_windows
    } else {
        let calendar_days = market_data.first()
            .zip(market_data.last())
            .map(|(first, last)| (market_data_timestamp(last) - market_data_timestamp(first)).num_days().max(0) as usize)
            .unwrap_or(0);
        let close_prices = close_prices_from_market_data(market_data);
        resolve_auto_wf_windows(calendar_days, market_data.len(), Some(&close_prices))
    };

    log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 2: Walk-forward with {} windows", num_windows);
    report_progress(2, "walk_forward", 0, num_windows);

    let wf_result = run_walk_forward(market_data, &config, num_windows, |step, total| {
        report_progress(2, "walk_forward", step, total);
    }).await
    .context("Stage 2: walk-forward analysis failed")?;

    let wf_passed = wf_result.avg_oos_sharpe >= config.min_oos_sharpe
        && wf_result.consistency_score > 0.3;

    stages.push(StageVerdict {
        stage: 2,
        name: "Walk-Forward".into(),
        passed: wf_passed,
        message: format!(
            "OOS Sharpe={:.2}, consistency={:.0}%, degradation={:.0}%",
            wf_result.avg_oos_sharpe,
            wf_result.consistency_score * 100.0,
            wf_result.overfitting_ratio * 100.0,
        ),
    });

    report_progress(2, "walk_forward", num_windows, num_windows);

    let has_ga = config.parameter_schema.is_some();
    if !wf_passed && !has_ga {
        let fail_msg = format!(
            "Strategy failed Stage 2 (walk-forward): OOS Sharpe={:.2} < {:.2}",
            wf_result.avg_oos_sharpe, config.min_oos_sharpe,
        );
        verdict = "fail".into();
        return Ok(ValidationResult {
            stages_completed: 2,
            quick_backtest: Some(quick_result),
            walk_forward: Some(wf_result),
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict,
            summary: fail_msg,
            stage_verdicts: stages,
            look_ahead_detected,
            elapsed_seconds: timing_start.elapsed().as_secs_f64(),
        });
    }

    // -----------------------------------------------------------------------
    // Stage 2.5: GA Parameter Optimization (optional)
    // -----------------------------------------------------------------------
    // Only runs if a parameter_schema is provided (i.e. the Python strategy
    // declared a `parameter_space()` method).
    let mut ga_report: Option<GAReport> = None;
    let mut optimized_backtest_result: Option<BacktestResult> = None;
    let mut ga_wf: Option<WalkForwardResult> = None;

    if let Some(ref schema) = config.parameter_schema {
        log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 2.5: GA optimization ({} params)", schema.len());

        // Set thread-local schema for DynamicChromosome::random()
        set_dynamic_schema(schema.clone());

        let ga_config = config.ga_config.clone().unwrap_or_else(|| GeneticConfig {
            population_size: 50,
            generations: 30,
            crossover_rate: 0.8,
            mutation_rate: 0.15,
            use_monte_carlo_fitness: false, // We do MC in Stage 3
            enable_fitness_sharing: true,
            force_sequential: true, // Python GIL makes rayon parallelism counterproductive
            // NSGA-II multi-objective GA: rank the population by Sharpe (maximize)
            // and max drawdown (minimize) simultaneously instead of a single
            // weighted-sum fitness. `NsgaIIOptimizer` (genetic::nsga2) already
            // implements fast non-dominated sorting, crowding-distance-preserving
            // selection, and knee-point selection for a single best-compromise
            // chromosome -- it was previously fully wired into the branch below
            // but never selected by any caller, since every `GeneticConfig`
            // construction across the codebase used `OptimizationMode::SingleObjective`.
            // This is the default GA config for Python custom-strategy validation
            // (the primary strategy-discovery path), so it's the highest-leverage
            // place to actually turn NSGA-II on.
            optimization_mode: config::OptimizationMode::MultiObjective {
                objectives: vec![
                    config::ObjectiveDef {
                        field: "sharpe_ratio".to_string(),
                        direction: config::ObjectiveDirection::Maximize,
                    },
                    config::ObjectiveDef {
                        field: "max_drawdown".to_string(),
                        direction: config::ObjectiveDirection::Minimize,
                    },
                ],
            },
            ..GeneticConfig::default()
        });

        // Signal start with the correct total so DB shows e.g. 0/10 not 0/1
        report_progress(3, "ga_optimization", 0, ga_config.generations);

        // Subsample data for GA exploration: target ~1M ticks for fast evaluation.
        // Bounded to `train_only_data` (same boundary Stage 1 uses) so the GA's
        // own candidate search -- and everything that reuses this same
        // `fitness_fn` closure below (NSGA-II, seed-stability rerun, landscape
        // and sensitivity sweeps) -- never trains on the tail slice walk-forward
        // treats as OOS. Previously this sampled from the full `market_data`,
        // so the "best" parameters it selected had already seen the exact data
        // later reported as this strategy's out-of-sample performance -- a
        // direct train/test leak (quant council 2026-08-03 GA overfitting review).
        // The final best params are still re-evaluated on the full dataset via
        // run_single_backtest below for headline/reporting purposes only.
        let ga_subsample_step = (train_only_data.len() / 1_000_000).max(1);
        let ga_data: Vec<MarketData> = if ga_subsample_step > 1 {
            log_info!(BACKTEST_LOGGER, "[VALIDATION] GA subsampling: every {}th tick ({} → {} ticks, train-only)", ga_subsample_step, train_only_data.len(), train_only_data.len() / ga_subsample_step);
            train_only_data.iter().step_by(ga_subsample_step).cloned().collect()
        } else {
            train_only_data.to_vec()
        };
        let market_data_arc: Arc<Vec<MarketData>> = Arc::new(ga_data);
        let market_data_for_pool = Arc::clone(&market_data_arc);
        let python_source_arc = Arc::new(config.python_source.clone());
        let backtest_config_arc = Arc::new(config.backtest_config.clone());
        let fee_config_arc = Arc::new(config.fee_config.clone());
        let supplementary_data_arc = Arc::new(config.supplementary_data.clone());
        let fitness_scorer = config.fitness_scorer.clone()
            .expect("ValidationConfig.fitness_scorer must be set to run GA optimization");
        let fitness_normalizer_for_optimizer = config.fitness_normalizer.clone();
        let initial_capital = config.backtest_config.trading.initial_capital;
        // Hard drawdown disqualification for GA candidates. Defaults to the
        // platform-wide 40% cap; tightened to the user's stated tolerance when
        // provided. Mirrors the built-in engine's hard gate (engine.rs), which
        // this Python GA path previously lacked entirely.
        let dd_hard_cap = config.max_drawdown_hard_cap.unwrap_or(0.40).clamp(0.05, 0.60);
        // Captured as a plain f64 (not the whole `ga_config`) so this doesn't
        // fight the later `ga_config.clone()` at optimizer construction --
        // see the fitness_fn closure's embedded-OOS-folding branch.
        let embedded_oos_gap_penalty = ga_config.embedded_oos_gap_penalty;

        // --- B1: fitness memoization ---
        // The GA generations, NSGA-II, and the post-GA landscape/sensitivity sweeps all
        // route through this same closure. Identical (parameters, sample_rate) pairs —
        // late-generation elites carried unchanged, recombination duplicates, and
        // landscape/sensitivity probes that coincide with already-evaluated points — each
        // re-run a full Python simulation. Memoize the result keyed on (sample_rate, params).
        //
        // Stored entries have the (potentially multi-MB) equity_curve stripped: this GA
        // path runs with use_monte_carlo_fitness=false, so equity curves are unused
        // downstream (AdaptiveGeneticOptimizer names it `_best_equity_curve`) and the final
        // best is re-evaluated on full data below. That keeps each cache entry to a few
        // hundred bytes — no risk of the OOM seen previously on memory-limited pods.
        let fitness_cache: Arc<std::sync::Mutex<HashMap<String, FitnessResult>>> =
            Arc::new(std::sync::Mutex::new(HashMap::new()));
        let fitness_cache_hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fitness_cache_misses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Handles retained for post-run logging (the closure moves its own clones).
        let cache_hits_log = fitness_cache_hits.clone();
        let cache_misses_log = fitness_cache_misses.clone();

        let fitness_fn: AsyncContextFitnessFn<DynamicChromosome> = Arc::new(move |chromosome: &DynamicChromosome, ctx: FitnessContext| {
            let data = market_data_arc.clone();
            let source = python_source_arc.clone();
            let bc = backtest_config_arc.clone();
            let fc = fee_config_arc.clone();
            let supp = supplementary_data_arc.clone();
            let scorer = fitness_scorer.clone();
            let init_cap = initial_capital;
            // Captured as an owned value here (not inside the async block below)
            // so the params/trades gate doesn't hold a borrow of `chromosome`
            // across the returned future's lifetime.
            let param_count = chromosome.genes.len();
            let sample_rate = ctx.sample_rate;
            let oos_folds = ctx.oos_folds;
            let params = chromosome.to_param_map();

            // Convert to ParameterValue map, preserving int vs float distinction
            let param_values: HashMap<String, ParameterValue> = params
                .into_iter()
                .map(|(k, v)| {
                    let pv = match v {
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                ParameterValue::Int(i)
                            } else if let Some(f) = n.as_f64() {
                                ParameterValue::Float(f)
                            } else {
                                ParameterValue::Float(0.0)
                            }
                        }
                        serde_json::Value::Bool(b) => ParameterValue::Bool(b),
                        _ => ParameterValue::Float(0.0),
                    };
                    (k, pv)
                })
                .collect();

            let cache = fitness_cache.clone();
            let cache_hits = fitness_cache_hits.clone();
            let cache_misses = fitness_cache_misses.clone();
            // B1: deterministic memoization key. ParameterValue derives Debug and its repr
            // is stable for identical values, so identical chromosomes (at the same sample
            // rate AND fold count) map to identical keys -- oos_folds is part of the key
            // because it changes what gets computed for otherwise-identical params (a
            // single full-range evaluation vs an embedded IS/OOS-folded one), not just how
            // finely the same computation samples the data.
            let cache_key = {
                let mut parts: Vec<String> = param_values
                    .iter()
                    .map(|(k, v)| format!("{}={:?}", k, v))
                    .collect();
                parts.sort();
                format!("sr{}|f{}|{}", sample_rate, oos_folds, parts.join("|"))
            };

            Box::pin(async move {
                // B1: return a memoized result if this (params, sample_rate) was already
                // evaluated, skipping a full Python simulation.
                if let Some(cached) = cache.lock().ok().and_then(|g| g.get(&cache_key).cloned()) {
                    cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return cached;
                }
                cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Apply adaptive sampling: early generations use coarser data for speed.
                let sampled = genetic::adaptive_sampling::sample_market_data(&data, sample_rate);
                let eval_data = sampled.as_slice();

                // Scores one BacktestResult into a full FitnessResult --
                // extracted (unchanged from before) so the embedded-OOS-
                // folding branch below can reuse the exact same scoring
                // logic per fold instead of duplicating it.
                // For each candidate's own DSR gate (genetic::dsr_gate_fitness) --
                // same population*generations convention used elsewhere on
                // the platform for this quantity.
                let dsr_n_trials = ga_config.population_size * ga_config.generations;
                let score_backtest_result = |br: &BacktestResult| -> FitnessResult {
                    let sharpe = br.sharpe_ratio.unwrap_or(0.0);
                    let sortino = br.sortino_ratio.unwrap_or(sharpe);
                    let trades = br.num_trades;
                    let pf = br.profit_factor.unwrap_or(1.0);
                    let wr = br.win_rate.unwrap_or(0.0);
                    let _calmar = br.calmar_ratio.unwrap_or(0.0);
                    let fill = br.execution_metrics.as_ref().map(|m| m.fill_rate).unwrap_or(0.0);
                    let dd_frac = (br.max_drawdown / init_cap.max(1.0)).abs();
                    // Date-range-aware activity check: distinct from the
                    // n>=MIN_TRADES_FOR_SIGNIFICANCE floor (sample-size
                    // trust) -- this asks whether the strategy traded often
                    // enough, given the calendar span it was tested over, to
                    // have plausibly been exercised across multiple regimes.
                    // See quant council 2026-07-27.
                    let data_span_days = match (br.first_trade_timestamp, br.last_trade_timestamp) {
                        (Some(first), Some(last)) => ((last - first).num_seconds() as f64 / 86400.0).max(1.0),
                        _ => 1.0,
                    };
                    // Minimum Track Record Length (Bailey & López de
                    // Prado, 2014): self-referential to THIS candidate's
                    // own realized Sharpe/SE, distinct from the fixed
                    // MIN_TRADES_FOR_SIGNIFICANCE floor -- a candidate can
                    // clear 30 trades and still have an unproven Sharpe if
                    // its own statistics say it needs more. Annualization
                    // factor doesn't affect min_track_record_length itself
                    // (computed from per-period Sharpe/SE before
                    // annualizing).
                    let min_trl = if br.trade_pct_returns.len() >= 3 {
                        crate::statistical_significance::compute_sharpe_significance(
                            &br.trade_pct_returns, 0.0, 252.0,
                        ).min_track_record_length
                    } else {
                        None
                    };
                    let dsr = metrics::performance::deflated_sharpe_ratio(
                        dsr_n_trials as u32, &br.trade_pct_returns,
                    );

                    // Every GA candidate-evaluation call site on the
                    // platform routes through the same caller-supplied
                    // `FitnessFunction` so they all reward/penalize the
                    // same things -- see `ValidationConfig::fitness_scorer`.
                    let inputs = genetic::fitness::FitnessInputs {
                        equity_curve: br.equity_curve.clone(),
                        num_trades: trades,
                        total_liquidations: 0, // Python strategies don't evolve leverage today
                        net_pnl: br.total_pnl,
                        initial_capital: init_cap,
                        sharpe,
                        sortino,
                        profit_factor: pf,
                        win_rate: wr,
                        max_drawdown_frac: dd_frac,
                        max_drawdown_abs: br.max_drawdown,
                        fill_rate: fill,
                        total_fees: br.transaction_costs.as_ref().map(|tc| tc.total_commission).unwrap_or(0.0),
                        param_count,
                        min_trl,
                        dsr,
                        data_span_days,
                        max_drawdown_hard_cap: dd_hard_cap,
                        leverage_evolved: false,
                        ..Default::default()
                    };
                    scorer.compute(&inputs)
                };

                let fitness_result = if oos_folds > 0 {
                    // === EMBEDDED IN-SAMPLE/OUT-OF-SAMPLE FOLDING ===
                    // (GeneticConfig::embedded_oos_folds) -- candidate
                    // selection gets real OOS awareness instead of only
                    // one full-range in-sample evaluation. Always uses the
                    // full-resolution `data` (not the adaptively tick-
                    // sampled `eval_data`): oos_folds is only > 0 during the
                    // same late-generation window sample_rate is already 1
                    // in, so they'd be identical here anyway -- see
                    // `AdaptiveSamplingConfig::oos_folds_for_generation`'s
                    // doc comment.
                    let validation_config = ValidationConfig {
                        python_source: (*source).clone(),
                        backtest_config: (*bc).clone(),
                        fee_config: (*fc).clone(),
                        supplementary_data: (*supp).clone(),
                        parameters: param_values.clone(),
                        ..Default::default()
                    };
                    let folds = evaluate_is_oos_folds(&data, &validation_config, oos_folds).await;
                    if folds.is_empty() {
                        // Not enough data for even one real fold -- fail
                        // closed rather than silently falling back to an
                        // in-sample-only score, which would defeat the
                        // whole point of enabling this.
                        FitnessResult::failure()
                    } else {
                        let scored: Vec<(FitnessResult, FitnessResult)> = folds.iter()
                            .map(|(is_br, oos_br)| (score_backtest_result(is_br), score_backtest_result(oos_br)))
                            .collect();
                        let n = scored.len() as f64;
                        let mean_oos = scored.iter().map(|(_, oos)| oos.fitness).sum::<f64>() / n;
                        let mean_gap_penalty = scored.iter()
                            .map(|(is_f, oos_f)| (is_f.fitness - oos_f.fitness).max(0.0))
                            .sum::<f64>() / n;
                        let combined_fitness = mean_oos - embedded_oos_gap_penalty * mean_gap_penalty;
                        // Reuse the last fold's OOS FitnessResult for the
                        // rich display metrics (sharpe/trades/etc.) -- only
                        // `.fitness` (what actually drives GA selection)
                        // needs to reflect the real combined score.
                        let mut result = scored.into_iter().last()
                            .map(|(_, oos)| oos)
                            .unwrap_or_else(FitnessResult::failure);
                        result.fitness = combined_fitness;
                        result
                    }
                } else {
                    // === SINGLE FULL-RANGE EVALUATION (existing behavior) ===
                    let result = {
                        let sim_config = crate::PythonSimConfig {
                            python_source: (*source).clone(),
                            backtest_config: (*bc).clone(),
                            fee_config: (*fc).clone(),
                            supplementary_data: (*supp).clone(),
                            parameters: param_values,
                            progress_callback: None,
                            risk_manager: None,
                            max_trade_log_size: Some(500),
                            orderbook_snapshots: None,
                            historical_iv_surfaces: None,
                            multi_venue_data: None,
                            option_instrument: None,
                            option_spread: None,
                            underlying_series: None,
                        };
                        crate::python_simulation::run(eval_data, sim_config).await
                            .map(|r| r.backtest_result)
                    };
                    match result {
                        Ok(br) => score_backtest_result(&br),
                        Err(e) => {
                            log_warn!(BACKTEST_LOGGER, "[GA] Fitness evaluation failed: {}", e);
                            FitnessResult::failure()
                        }
                    }
                };
                // B1: cache a lightweight copy (equity_curve stripped — unused in this GA
                // path) so identical (params, sample_rate) pairs are not re-simulated.
                {
                    let mut entry = fitness_result.clone();
                    entry.equity_curve = Vec::new();
                    if let Ok(mut guard) = cache.lock() {
                        guard.insert(cache_key, entry);
                    }
                }
                fitness_result
            })
        });

        // Build progress callback for GA generations
        let ga_progress_cb = config.progress_callback.as_ref().map(|cb| {
            let cb = cb.clone();
            let total_gens = ga_config.generations;
            Arc::new(move |gen: usize, _total: usize| {
                cb(3, "ga_optimization", gen, total_gens);
                Box::pin(async {}) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
            }) as genetic::ProgressCallback
        });

        let mut optimizer = AdaptiveGeneticOptimizer::<DynamicChromosome>::new(
            ga_config.clone(),
            fitness_fn.clone(),
        );
        optimizer.progress_callback = ga_progress_cb;
        optimizer.fitness_normalizer = fitness_normalizer_for_optimizer;
        optimizer.sampling_config.oos_folds = ga_config.embedded_oos_folds;

        // --- Spawn process-pool workers for parallel Python evaluation (GIL bypass) ---
        optimizer.worker_pool = spawn_worker_pool_for_data(&market_data_for_pool, &config, "ga_initial");

        // --- Decide single-objective vs multi-objective ---
        let mut nsga2_result: Option<NsgaIIResult> = None;

        let (best_chromosome, best_fitness) = match &ga_config.optimization_mode {
            config::OptimizationMode::MultiObjective { objectives } => {
                // Convert config objectives → nsga2 objectives
                let nsga_objectives: Vec<genetic::nsga2::ObjectiveDef> = objectives.iter().map(|o| {
                    genetic::nsga2::ObjectiveDef {
                        field: o.field.clone(),
                        direction: match o.direction {
                            config::ObjectiveDirection::Maximize => genetic::nsga2::Direction::Maximize,
                            config::ObjectiveDirection::Minimize => genetic::nsga2::Direction::Minimize,
                        },
                    }
                }).collect();

                let nsga_optimizer = NsgaIIOptimizer::<DynamicChromosome> {
                    population_size: ga_config.population_size,
                    generations: ga_config.generations,
                    crossover_rate: ga_config.crossover_rate,
                    mutation_rate: ga_config.mutation_rate,
                    objectives: nsga_objectives,
                    fitness_fn: fitness_fn.clone(),
                    sampling_config: AdaptiveSamplingConfig::default(),
                    force_sequential: ga_config.force_sequential,
                };

                let (result, knee_chromo, knee_fitness) = nsga_optimizer.run().await;
                log_info!(BACKTEST_LOGGER, "[VALIDATION] NSGA-II complete: pareto_front={} solutions, hypervolume={:.4}",
                    result.pareto_front.len(), result.hypervolume);
                nsga2_result = Some(result);
                (knee_chromo, knee_fitness)
            }
            config::OptimizationMode::SingleObjective if ga_config.bayesian_optimization => {
                // Bayesian optimization replaces the whole GA search here --
                // see GeneticConfig::bayesian_optimization's doc comment for
                // why (guided GA not shown to reliably beat random search on
                // these small, expensive-to-evaluate parameter spaces).
                // Budget matches what a GA run at this population/generation
                // count would have spent (clamped -- see BO_MAX_TOTAL_EVALS).
                let total_evals = ga_config.population_size * ga_config.generations;
                log_info!(BACKTEST_LOGGER, "[VALIDATION] Running Bayesian optimization instead of GA (requested budget={}, clamped to <= {})", total_evals, landscape::BO_MAX_TOTAL_EVALS);
                // run_bayesian_optimization takes the plain (context-free)
                // genetic::AsyncFitnessFn, but fitness_fn here is the
                // generation-aware genetic::AsyncContextFitnessFn (needed by
                // optimizer.run()'s adaptive/progressive sampling below).
                // BO has no notion of "generation" -- every one of its
                // (comparatively few, <= BO_MAX_TOTAL_EVALS) evaluations
                // should be full-resolution, not progressively sampled, so
                // adapt with a fixed full_resolution context rather than
                // threading a second fitness-fn type through the genetic
                // crate's public API.
                let fitness_fn_for_bo: genetic::AsyncFitnessFn<DynamicChromosome> = {
                    let inner = fitness_fn.clone();
                    Arc::new(move |chromo: &DynamicChromosome| {
                        let inner = inner.clone();
                        let chromo = chromo.clone();
                        Box::pin(async move {
                            inner(&chromo, genetic::FitnessContext::full_resolution(0, 1)).await
                        })
                    })
                };
                landscape::run_bayesian_optimization(
                    schema,
                    &fitness_fn_for_bo,
                    total_evals,
                    1.0, // UCB kappa -- moderate exploration, no tuning evidence yet either way
                    &|| {},
                ).await
            }
            config::OptimizationMode::SingleObjective => {
                optimizer.run().await
            }
        };

        // Extract convergence history for the GA report
        let convergence_curve = optimizer.convergence_history.lock()
            .map(|h| h.clone())
            .unwrap_or_default();

        // Shut down worker pool if it was used
        if let Some(pool) = optimizer.worker_pool.take() {
            log_info!(BACKTEST_LOGGER, "[VALIDATION] Shutting down GA worker pool");
            pool.shutdown();
        }

        log_info!(BACKTEST_LOGGER, "[VALIDATION] GA complete: best_fitness={:.4}, params={:?}", best_fitness, best_chromosome.to_param_map());

        stages.push(StageVerdict {
            stage: 3, // Using 3 to represent 2.5 (u8 doesn't support half-stages)
            name: "GA Optimization".into(),
            passed: best_fitness > 0.0,
            message: format!(
                "Optimized {} params, best fitness={:.4}",
                schema.len(),
                best_fitness,
            ),
        });

        // --- Seed-stability check (optional, single-objective mode only) ---
        // Re-runs the GA with a second fixed seed and compares the two runs'
        // best parameters. Large parameter-space divergence despite similar
        // fitness is a concrete overfitting-fragility signal: many
        // equally-good-looking regions of parameter space rather than one
        // stable optimum. Disabled by default since it doubles GA cost.
        let seed_stability_report: Option<landscape::SeedStabilityReport> =
            if ga_config.check_seed_stability
                && matches!(ga_config.optimization_mode, config::OptimizationMode::SingleObjective)
            {
                if let Some(seed_a) = ga_config.random_seed {
                    // Arbitrary large odd offset (splitmix64 golden-ratio constant) so the
                    // second seed never collides with, or trivially neighbors, the first.
                    const SEED_STABILITY_OFFSET: u64 = 0x9E3779B97F4A7C15;
                    let seed_b = seed_a.wrapping_add(SEED_STABILITY_OFFSET);
                    log_info!(BACKTEST_LOGGER, "[VALIDATION] Seed-stability check: re-running GA with seed={} (vs seed={})", seed_b, seed_a);

                    let mut ga_config_b = ga_config.clone();
                    ga_config_b.random_seed = Some(seed_b);
                    // Stability run doesn't need its own re-run, so disable it here to
                    // avoid unbounded recursion if this config is ever reused.
                    ga_config_b.check_seed_stability = false;

                    let mut optimizer_b = AdaptiveGeneticOptimizer::<DynamicChromosome>::new(
                        ga_config_b,
                        fitness_fn.clone(),
                    );
                    optimizer_b.fitness_normalizer = config.fitness_normalizer.clone();
                    optimizer_b.sampling_config.oos_folds = optimizer_b.config.embedded_oos_folds;
                    optimizer_b.worker_pool = spawn_worker_pool_for_data(&market_data_for_pool, &config, "ga_seed_stability");

                    let (best_chromosome_b, best_fitness_b) = optimizer_b.run().await;

                    if let Some(pool) = optimizer_b.worker_pool.take() {
                        pool.shutdown();
                    }

                    let param_distance = best_chromosome.distance(&best_chromosome_b);
                    let report = landscape::SeedStabilityReport::new(
                        seed_a, seed_b, best_fitness, best_fitness_b, param_distance,
                    );
                    log_info!(
                        BACKTEST_LOGGER,
                        "[VALIDATION] Seed-stability: fitness_a={:.4} fitness_b={:.4} param_distance={:.4} fragile={}",
                        report.best_fitness_a, report.best_fitness_b, report.param_distance, report.fragile
                    );
                    Some(report)
                } else {
                    log_warn!(BACKTEST_LOGGER, "[VALIDATION] check_seed_stability enabled but no random_seed configured; skipping");
                    None
                }
            } else {
                None
            };

        // Run landscape heatmap (top 3 sensitivity-ranked pairs, 7×7 grid each)
        // Landscape/sensitivity use AsyncFitnessFn — wrap our context fn with full resolution.
        //
        // Each point gets its own short outer timeout (2026-08-16), well below the
        // strategy executor's own 30s compute_signals() timeout. This is a
        // diagnostic-only phase (sensitivity ranking + landscape visualization,
        // never feeds the actual GA decision), so a skipped point just means a
        // slightly sparser heatmap -- an acceptable tradeoff for capping the
        // worst case. Without this, a single pathological parameter combination
        // (e.g. an extreme lookback window) can cost up to the full 30s per
        // point, and with ~40 points evaluated mostly sequentially here, that
        // compounds into hours -- observed live on a random_control_arm job,
        // whose best_chromosome (never fitness-selected against, unlike guided
        // search) is disproportionately likely to sit in a slow-to-evaluate
        // region that guided search's own selection pressure would normally
        // steer away from. A timed-out point is skipped, not retried.
        const LANDSCAPE_EVAL_TIMEOUT_SECS: u64 = 8;

        // Skip the whole phase for random_control_arm comparison jobs: it's
        // diagnostic-only (sensitivity ranking + landscape visualization for the
        // UI), never feeds the guided-vs-random comparison metrics themselves,
        // and costs ~40 extra fitness evaluations per job -- on top of that, a
        // random_control job's best_chromosome is the one most likely to sit in
        // a slow-to-evaluate region (see comment above), so it pays the worst
        // case of the per-point timeout most often. Empty vecs are already a
        // schema-safe state (same as the `schema.len() < 2` path below feeds
        // into GAReport).
        let (sensitivity, landscape_reports): (Vec<landscape::SensitivityEntry>, Vec<landscape::LandscapeReport>) = if ga_config.random_control {
            report_progress(3, "ga_landscape", 0, 0);
            (vec![], vec![])
        } else {
        let landscape_fitness_fn: AsyncFitnessFn<DynamicChromosome> = {
            let ctx_fn = fitness_fn.clone();
            let total_gens = ga_config.generations;
            Arc::new(move |chromo: &DynamicChromosome| {
                let f = ctx_fn.clone();
                let c = chromo.clone();
                Box::pin(async move {
                    let ctx = FitnessContext::full_resolution(total_gens.saturating_sub(1), total_gens);
                    tokio::time::timeout(
                        std::time::Duration::from_secs(LANDSCAPE_EVAL_TIMEOUT_SECS),
                        f(&c, ctx),
                    )
                    .await
                    .unwrap_or_else(|_| FitnessResult::failure())
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = FitnessResult> + Send>>
            })
        };

        // Estimate total fitness evaluations for fine-grained progress:
        //   sensitivity: ~4 evals per param (2 for bools)
        //   landscape: 8 LHS points per pair; GP posterior drives the 20×20 vis grid (0 extra evals)
        const BO_N_INIT: usize = 8;
        const BO_VIS_GRID: usize = 20;
        let num_params = schema.params.len();
        let sensitivity_evals = num_params * 4; // upper-bound (bools are 2)
        let max_landscape_evals = 2 * BO_N_INIT;
        let total_landscape_evals = sensitivity_evals + max_landscape_evals;
        let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Sensitivity analysis first — we need scores to pick landscape pairs
        report_progress(3, "ga_landscape", 0, total_landscape_evals);
        let sensitivity = {
            let counter = eval_counter.clone();
            let cb = config.progress_callback.clone();
            let total = total_landscape_evals;
            landscape::sensitivity_analysis(
                schema, &landscape_fitness_fn, &best_chromosome, best_fitness,
                &move || {
                    let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if let Some(ref f) = cb { f(3, "ga_landscape", c, total); }
                },
            ).await
        };

        // Generate landscapes for the top 3 most-sensitive parameter pairs
        let landscape_reports: Vec<landscape::LandscapeReport> = if schema.len() >= 2 {
            #[allow(deprecated)]
            let pairs = if sensitivity.is_empty() {
                // Fallback: use range-width heuristic if sensitivity unavailable
                let (a, b) = schema.top_two_continuous_indices();
                vec![(a, b)]
            } else {
                schema.top_pairs_by_sensitivity(&sensitivity, 2)
            };
            let mut reports = Vec::with_capacity(pairs.len());
            for (a, b) in pairs {
                let counter = eval_counter.clone();
                let cb = config.progress_callback.clone();
                let total = total_landscape_evals;
                let lrep = landscape::bo_landscape_heatmap(
                    schema, &landscape_fitness_fn, &best_chromosome, a, b,
                    BO_N_INIT, BO_VIS_GRID,
                    &move || {
                        let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if let Some(ref f) = cb { f(3, "ga_landscape", c, total); }
                    },
                ).await;
                log_info!(BACKTEST_LOGGER, "[VALIDATION] BO landscape: flatness={:.2} ({} LHS evals → {}x{} GP vis grid)", lrep.flatness_score, BO_N_INIT, lrep.param_a_name, lrep.param_b_name);
                reports.push(lrep);
            }
            reports
        } else {
            vec![]
        };
        report_progress(3, "ga_landscape", total_landscape_evals, total_landscape_evals);
        (sensitivity, landscape_reports)
        };

        // B1: report memoization effectiveness across GA + landscape/sensitivity.
        {
            let hits = cache_hits_log.load(std::sync::atomic::Ordering::Relaxed);
            let misses = cache_misses_log.load(std::sync::atomic::Ordering::Relaxed);
            let total = hits + misses;
            let pct = if total > 0 { (hits as f64 / total as f64) * 100.0 } else { 0.0 };
            log_info!(BACKTEST_LOGGER, "[VALIDATION] GA fitness cache: {} hits / {} evals ({:.1}% simulations skipped)", hits, total, pct);
        }

        // Build GA report (convergence curve populated from optimizer tracking)
        // Evaluate the WINNER's own fixed params across real purged IS/OOS
        // windows for overfitting measurement -- reusing run_walk_forward
        // as-is (the same purge-gap-safe windowing Stage 2 already uses on
        // default params), just with config.parameters overridden to the
        // winner's own values instead of the strategy's declared defaults.
        // This replaces the old per-window mini-GA re-optimization
        // (run_walk_forward_with_ga's own independent search per window,
        // which measured "does re-optimizing from scratch each window still
        // work" rather than "does the winner this run actually found
        // generalize") -- cheaper, more directly answers the question this
        // report is for, and gets a real purge gap run_walk_forward_with_ga
        // never had (its own doc comment: "No purge gap in this path").
        report_progress(3, "ga_walk_forward", 0, num_windows);
        let winner_params: HashMap<String, ParameterValue> = best_chromosome.to_param_map()
            .into_iter()
            .map(|(k, v)| {
                let pv = match v {
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() { ParameterValue::Int(i) }
                        else { ParameterValue::Float(n.as_f64().unwrap_or(0.0)) }
                    }
                    serde_json::Value::Bool(b) => ParameterValue::Bool(b),
                    _ => ParameterValue::Float(0.0),
                };
                (k, pv)
            })
            .collect();
        let winner_config = ValidationConfig {
            python_source: config.python_source.clone(),
            backtest_config: config.backtest_config.clone(),
            fee_config: config.fee_config.clone(),
            supplementary_data: config.supplementary_data.clone(),
            parameters: winner_params,
            ..Default::default()
        };
        let ga_wf_result = run_walk_forward(
            market_data, &winner_config,
            num_windows,
            |step, total| { report_progress(3, "ga_walk_forward", step, total); },
        ).await;

        let (train_sharpe, test_sharpe) = match ga_wf_result {
            Ok(wf) => {
                let ts = wf.windows.iter().map(|w| w.train_sharpe).sum::<f64>() / wf.windows.len() as f64;
                let oos = wf.avg_oos_sharpe;
                ga_wf = Some(wf);
                (ts, oos)
            }
            Err(ref e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] Per-window GA WF failed, falling back: {}", e);
                (quick_result.sharpe_ratio.unwrap_or(0.0), wf_result.avg_oos_sharpe)
            }
        };
        let degradation_pct = if train_sharpe.abs() > 1e-9 {
            ((train_sharpe - test_sharpe) / train_sharpe.abs() * 100.0).clamp(-200.0, 200.0)
        } else {
            0.0
        };
        // Overfit / non-generalization classification.
        // The degradation ratio is only a meaningful "overfit" signal when the
        // in-sample (train) Sharpe is positive. When train Sharpe is <= 0 the
        // ratio flips sign, so a worthless strategy (negative Sharpe both in and
        // out of sample) can masquerade as "improvement" (negative degradation).
        // Classify those cases explicitly instead of trusting the bare ratio.
        let is_overfit = if train_sharpe <= 0.0 {
            // Not profitable in-sample -> never trustworthy. Flag whenever the
            // out-of-sample Sharpe is also non-positive or strictly worse.
            test_sharpe <= 0.0 || test_sharpe < train_sharpe
        } else {
            // Profitable in-sample: a large drop, or any sign flip to a negative
            // out-of-sample Sharpe, counts as overfitting.
            test_sharpe < 0.0 || degradation_pct > 50.0
        };
        let overfit_report = landscape::OverfitReport {
            train_sharpe,
            test_sharpe,
            degradation_pct,
            is_overfit,
        };

        // Bug #29 \u2014 Extract top-N diverse parameter sets from final GA
        // population so the UI can show alternative high-fitness candidates.
        let top_n_params = {
            let snap = optimizer.final_population.lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            if snap.is_empty() {
                vec![]
            } else {
                landscape::extract_top_n(&snap, &best_chromosome, 5)
            }
        };

        ga_report = Some(GAReport {
            best_params: best_chromosome.to_param_map(),
            best_fitness,
            top_n_params,
            landscapes: landscape_reports,
            sensitivity,
            convergence_curve,
            overfitting: Some(overfit_report),
            pareto_front: nsga2_result.as_ref().map(|r| r.pareto_front.clone()),
            pareto_hypervolume: nsga2_result.as_ref().map(|r| r.hypervolume),
            pareto_objectives: nsga2_result.as_ref().map(|r| r.objectives.clone()),
            seed_stability: seed_stability_report,
            // Real evaluation count: cache MISSES (genuine fresh Python
            // simulations) across the sensitivity + landscape sweeps --
            // those share `fitness_cache`/`fitness_cache_misses` above --
            // PLUS `optimizer.pool_evaluations`, the main GA/NSGA-II run's
            // own population evaluations whenever a worker pool is active.
            //
            // FIX (2026-08-27): this comment used to claim `cache_misses_log`
            // alone "already covers all three" (main run + sensitivity +
            // landscape). That was only ever true when the main run
            // evaluated fitness in-process. Once the worker-pool path was
            // added (`pool.evaluate_batch()`, the default whenever more
            // than 1 CPU is available -- see `spawn_worker_pool_for_data`),
            // every population evaluation in the main run bypasses
            // `fitness_fn`/`fitness_cache` entirely, and `cache_misses_log`
            // silently fell back to counting only the pre-loop baseline
            // chromosome (always 1) plus, for guided runs only, the
            // sensitivity/landscape sweep (~24-40) -- random_control runs
            // skip that sweep outright (see the `if ga_config.random_control`
            // branch above), so their `total_fresh_evaluations` read a flat
            // `1` regardless of how large the real search actually was.
            // Confirmed live: a real 30-generation x 50-chromosome
            // random_control run reported `1` despite ~1500 real
            // simulations. `pool_evaluations` (added alongside this fix,
            // see its doc comment in `genetic::AdaptiveGeneticOptimizer`)
            // closes that gap by counting every pool-dispatched batch
            // directly at the dispatch site, independent of which arm ran.
            //
            // `ga_wf.ga_trial_count` (populated above) is always `None` now
            // -- the post-GA walk-forward evaluates the winner's own
            // already-known parameters, not a fresh search, so there's
            // nothing extra to add for it.
            total_fresh_evaluations: Some(
                cache_misses_log.load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_add(optimizer.pool_evaluations.load(std::sync::atomic::Ordering::Relaxed))
                    .saturating_add(ga_wf.as_ref().and_then(|wf| wf.ga_trial_count).unwrap_or(0))
            ),
        });

        // Re-run the best chromosome on full data to get an optimized BacktestResult
        // (used for MC in Stage 3).
        report_progress(3, "ga_final_backtest", 0, 1);
        {
            let param_map = best_chromosome.to_param_map();
            let opt_params: HashMap<String, ParameterValue> = param_map
                .into_iter()
                .map(|(k, v)| {
                    let pv = match v {
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                ParameterValue::Int(i)
                            } else {
                                ParameterValue::Float(n.as_f64().unwrap_or(0.0))
                            }
                        }
                        serde_json::Value::Bool(b) => ParameterValue::Bool(b),
                        _ => ParameterValue::Float(0.0),
                    };
                    (k, pv)
                })
                .collect();

            let opt_config = ValidationConfig {
                parameters: opt_params,
                ..ValidationConfig {
                    python_source: config.python_source.clone(),
                    backtest_config: config.backtest_config.clone(),
                    fee_config: config.fee_config.clone(),
                    supplementary_data: config.supplementary_data.clone(),
                    ..Default::default()
                }
            };

            if let Ok(opt_br) = run_single_backtest(market_data, &opt_config).await {
                // Defensive guard: a GA "winner" that produces zero trades on
                // the full dataset is a degenerate parameter set (e.g.
                // min_position_size > max_position_size). Surface it loudly
                // and downgrade the verdict instead of silently persisting
                // all-zero metrics (job 64e39acf).
                if opt_br.num_trades == 0 {
                    log_warn!(BACKTEST_LOGGER,
                        "[VALIDATION] GA best chromosome produced 0 trades on full data — \
                         degenerate parameter set selected (params={:?}); downgrading verdict",
                        best_chromosome.to_param_map());
                    if verdict == "pass" {
                        verdict = "warn".into();
                    }
                }
                optimized_backtest_result = Some(opt_br);
            }
        }
        report_progress(3, "ga_final_backtest", 1, 1);

        // Signal GA complete using the actual generation count so the DB doesn't
        // get overwritten with a hardcoded 1/1 after correct per-gen updates.
        let final_gens = ga_config.generations;
        report_progress(3, "ga_optimization", final_gens, final_gens);
    }

    // -----------------------------------------------------------------------
    // Stage 3.5: CPCV (Combinatorial Purged Cross-Validation) — only if GA ran
    // -----------------------------------------------------------------------
    let mut cpcv_pbo: Option<f64> = None;
    let mut cpcv_deflated_sharpe: Option<f64> = None;
    let mut cpcv_parameter_stability: Option<f64> = None;

    if config.parameter_schema.is_some() && market_data.len() >= 250 {
        log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage 3.5: CPCV analysis");

        let cpcv_config = CPCVConfig {
            n_folds: 5,
            n_test_folds: 2,
            purge_gap_days: 5,
            min_samples_per_fold: 50,
        };

        let wf_config = WalkForwardConfig::default();
        let analyzer = WalkForwardAnalyzer::new(wf_config, config.backtest_config.clone());

        let cpcv_validation_config = Arc::new(ValidationConfig {
            python_source: config.python_source.clone(),
            backtest_config: config.backtest_config.clone(),
            fee_config: config.fee_config.clone(),
            supplementary_data: config.supplementary_data.clone(),
            parameters: config.parameters.clone(),
            progress_callback: None,
            parameter_schema: None,
            ga_config: None,
            fitness_scorer: None,
            ..Default::default()
        });
        let cpcv_market_data: Arc<Vec<MarketData>> = Arc::new(market_data.to_vec());

        let cpcv_result = analyzer.run_cpcv_analysis(
            market_data.len(),
            cpcv_config,
            |train_indices, test_indices| {
                let cfg = cpcv_validation_config.clone();
                let data = cpcv_market_data.clone();
                async move {
                    // Extract train/test slices
                    let train_data: Vec<MarketData> = train_indices.iter()
                        .filter_map(|&i| data.get(i).cloned())
                        .collect();
                    let test_data: Vec<MarketData> = test_indices.iter()
                        .filter_map(|&i| data.get(i).cloned())
                        .collect();

                    // Run backtest on training data to get IS sharpe
                    let train_result = run_single_backtest(&train_data, &cfg).await
                        .map_err(|e| anyhow::anyhow!("CPCV fold train backtest failed: {}", e))?;
                    let is_sharpe = train_result.sharpe_ratio.unwrap_or(0.0);

                    // Run backtest on test data to get OOS metrics
                    let test_result = run_single_backtest(&test_data, &cfg).await
                        .map_err(|e| anyhow::anyhow!("CPCV fold test backtest failed: {}", e))?;

                    Ok(walkforward::CpcvFoldEval {
                        optimal_parameters: std::collections::HashMap::new(),
                        in_sample_sharpe: is_sharpe,
                        test_metrics: walkforward::TradingMetrics {
                            total_return: test_result.total_pnl,
                            sharpe_ratio: test_result.sharpe_ratio.unwrap_or(0.0),
                            max_drawdown: test_result.max_drawdown,
                            win_rate: test_result.win_rate.unwrap_or(0.0),
                            profit_factor: test_result.profit_factor.unwrap_or(1.0),
                            total_trades: test_result.num_trades,
                            avg_trade_duration_hours: test_result.avg_trade_duration.unwrap_or(0.0),
                            volatility: test_result.volatility.unwrap_or(0.0),
                        },
                    })
                }
            },
        ).await;

        match cpcv_result {
            Ok(result) => {
                cpcv_pbo = Some(result.probability_of_backtest_overfit);
                // Only trust CPCV's deflated Sharpe when every test fold had
                // enough trades for its own Sharpe to mean anything -- a low-
                // trade-count strategy (e.g. 18 trades over 5 folds) can leave
                // individual folds with 2-3 trades each, whose unstable Sharpe
                // blows up sharpe_variance_across_folds and produces a DSR
                // many orders of magnitude from 0 or 1 (observed in production:
                // a strategy with an actual Sharpe of 0.97 got a CPCV DSR of
                // ~1e-118). worker.rs's `if let Some(cpcv_dsr) = ...` overrides
                // the properly lineage-aware DSR unconditionally whenever this
                // is `Some`, so an unreliable value here silently wins over a
                // trustworthy one -- gating it at the source is simpler than
                // threading a reliability flag through every caller.
                cpcv_deflated_sharpe = if result.min_trades_per_fold >= walkforward::MIN_RELIABLE_TRADES_PER_FOLD {
                    Some(result.deflated_sharpe_ratio)
                } else {
                    log_warn!(BACKTEST_LOGGER, "[VALIDATION] CPCV DSR not trusted: min_trades_per_fold={} < {}",
                        result.min_trades_per_fold, walkforward::MIN_RELIABLE_TRADES_PER_FOLD);
                    None
                };
                cpcv_parameter_stability = Some(result.parameter_stability_score);

                let cpcv_passed = result.probability_of_backtest_overfit < 0.5;
                stages.push(StageVerdict {
                    stage: 4,
                    name: "CPCV Analysis".into(),
                    passed: cpcv_passed,
                    message: format!(
                        "PBO={:.2}, DSR={:.3}, stability={:.2}",
                        result.probability_of_backtest_overfit,
                        result.deflated_sharpe_ratio,
                        result.parameter_stability_score,
                    ),
                });
                if !cpcv_passed {
                    verdict = "warn".into();
                }
                log_info!(BACKTEST_LOGGER, "[VALIDATION] CPCV: PBO={:.2}, DSR={:.3}",
                    result.probability_of_backtest_overfit, result.deflated_sharpe_ratio);
            }
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] CPCV analysis failed: {}", e);
            }
        }
    }

    // Use optimized result for MC if GA ran AND it has enough trades for meaningful
    // Monte Carlo resampling. Otherwise fall back to quick_result which typically
    // has far more trades (e.g. 24K vs 30).
    //
    // PREFERRED: when walk-forward produced an OOS-combined result, use THAT for
    // MC so the bands reflect the same out-of-sample realized P&L shown to the
    // user. The IS quick/optimized backtests can be wildly more profitable than
    // OOS due to overfit, which previously made MC P50 land above $0 while the
    // realized OOS PnL was negative (Bug #15).
    let min_mc_trades = config.mc_min_trades;
    let oos_combined_for_mc: Option<&BacktestResult> = ga_wf
        .as_ref()
        .and_then(|w| w.oos_combined_result.as_ref())
        .or_else(|| wf_result.oos_combined_result.as_ref())
        .filter(|r| r.trade_returns.len() >= min_mc_trades || r.num_trades >= min_mc_trades);
    let opt_has_enough = optimized_backtest_result
        .as_ref()
        .map_or(false, |r| r.trade_returns.len() >= min_mc_trades || r.num_trades >= min_mc_trades);
    let (mc_base_result, mc_base_label) = if let Some(oos) = oos_combined_for_mc {
        (oos, "OOS-combined")
    } else if opt_has_enough {
        (optimized_backtest_result.as_ref().unwrap(), "GA-optimized")
    } else {
        (&quick_result, "quick backtest fallback")
    };
    log_info!(BACKTEST_LOGGER, "[VALIDATION] MC base: {} trades ({})",
        mc_base_result.num_trades, mc_base_label);

    // -----------------------------------------------------------------------
    // Stage 3: Monte Carlo Simulation
    // -----------------------------------------------------------------------
    let mc_stage_num: u8 = if config.parameter_schema.is_some() { 4 } else { 3 };
    log_info!(BACKTEST_LOGGER, "[VALIDATION] Stage {}: Monte Carlo ({} runs)", mc_stage_num, config.mc_runs);
    report_progress(mc_stage_num, "monte_carlo", 0, config.mc_runs);

    let mc_progress: Option<Arc<dyn Fn(&str, usize, usize) + Send + Sync>> =
        config.progress_callback.as_ref().map(|cb| {
            let cb = cb.clone();
            let stage = mc_stage_num;
            Arc::new(move |_phase: &str, current: usize, total: usize| {
                cb(stage, "monte_carlo", current, total);
            }) as Arc<dyn Fn(&str, usize, usize) + Send + Sync>
        });

    let mc_result = monte_carlo::run_monte_carlo_simulation_with_progress(
        mc_base_result,
        config.mc_runs,
        mc_progress,
    ).context("Monte Carlo simulation failed")?;
    let mut mc_result = mc_result;
    // Record which result the MC was computed from so the UI can label the bands
    // (they may describe a DIFFERENT, out-of-sample trade set than the in-sample
    // trade log / equity curve shown alongside them — see Bug #15 / Bug #33).
    mc_result.base_source = format!("{} ({} trades)", mc_base_label, mc_base_result.num_trades);

    // Estimate ruin probability: fraction of percentile 5 equity that's negative
    let ruin_prob = estimate_ruin_probability(&mc_result);
    let mc_passed = ruin_prob <= config.max_ruin_probability;

    if !mc_passed {
        verdict = "warn".into();
    }

    stages.push(StageVerdict {
        stage: mc_stage_num,
        name: "Monte Carlo".into(),
        passed: mc_passed,
        message: format!(
            "VaR95=${:.2}, CVaR95=${:.2}, ruin_prob={:.1}%, {}",
            mc_result.var_95,
            mc_result.cvar_95,
            ruin_prob * 100.0,
            if mc_result.var_unreliable { "(unreliable: <30 trades)" } else { "" },
        ),
    });

    report_progress(mc_stage_num, "monte_carlo", config.mc_runs, config.mc_runs);

    // Block bootstrap MC — auto-computed block size: n^(1/3)
    let block_bootstrap_result = {
        let n_trades = mc_base_result.trade_returns.len();
        let block_size = (n_trades as f64).powf(1.0 / 3.0).round().max(3.0) as usize;
        let bb_config = monte_carlo::BlockBootstrapConfig {
            block_size,
            overlapping: true,
            min_block_size: 3,
        };
        monte_carlo::run_block_bootstrap_simulation(mc_base_result, config.mc_runs, bb_config).ok()
    };

    // Regime-aware MC — detect current regime from equity curve volatility
    let regime_mc_result = {
        let eq = &mc_base_result.equity_curve;
        if eq.len() > 20 {
            // Compute per-step returns as volatility proxy
            let returns: Vec<f64> = eq.windows(2)
                .map(|w| if w[0].abs() > 1e-12 { (w[1] - w[0]) / w[0] } else { 0.0 })
                .collect();
            // Rolling volatility (20-period)
            let lookback = 20usize;
            let vol_per_trade: Vec<f64> = returns.iter().enumerate().map(|(i, _)| {
                let start = i.saturating_sub(lookback);
                let window = &returns[start..=i];
                let mean_w = window.iter().sum::<f64>() / window.len() as f64;
                let var = window.iter().map(|r| (r - mean_w).powi(2)).sum::<f64>() / window.len() as f64;
                var.sqrt()
            }).collect();
            let current_vol = vol_per_trade.last().copied().unwrap_or(0.02);
            let regime_cfg = monte_carlo::RegimeAwareConfig {
                volatility_lookback: lookback,
                current_volatility: current_vol,
                regime_weight: 0.7,
                volatility_thresholds: [0.01, 0.025, 0.05],
            };
            monte_carlo::run_regime_aware_simulation(mc_base_result, config.mc_runs, regime_cfg, &vol_per_trade).ok()
        } else {
            None
        }
    };

    // -----------------------------------------------------------------------
    // Final summary
    // -----------------------------------------------------------------------
    let total_stages = if config.parameter_schema.is_some() { 4 } else { 3 };
    let ga_blurb = if ga_report.is_some() {
        format!(" GA: fitness={:.4}.", ga_report.as_ref().unwrap().best_fitness)
    } else {
        String::new()
    };

    // Prefer GA walk-forward result (Stage 3) over default-param WF (Stage 2)
    let final_wf = ga_wf.unwrap_or(wf_result);

    let summary = format!(
        "Stages completed: {}/{}. Quick: PnL=${:.2}. WF: OOS Sharpe={:.2}, consistency={:.0}%.{} MC: VaR95=${:.2}, ruin={:.1}%. Verdict: {}",
        total_stages,
        total_stages,
        quick_pnl,
        final_wf.avg_oos_sharpe,
        final_wf.consistency_score * 100.0,
        ga_blurb,
        mc_result.var_95,
        ruin_prob * 100.0,
        verdict,
    );

    Ok(ValidationResult {
        stages_completed: total_stages as u8,
        quick_backtest: Some(quick_result),
        walk_forward: Some(final_wf),
        monte_carlo: Some(mc_result),
        block_bootstrap: block_bootstrap_result,
        regime_mc: regime_mc_result,
        ga_report,
        optimized_backtest: optimized_backtest_result,
        cpcv_pbo,
        cpcv_deflated_sharpe,
        cpcv_parameter_stability,
        verdict,
        summary,
        stage_verdicts: stages,
        look_ahead_detected,
        elapsed_seconds: timing_start.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Stage implementations
// ---------------------------------------------------------------------------

/// Run a single-pass backtest over all market data.
#[cfg(feature = "python")]
async fn run_single_backtest(
    data: &[MarketData],
    config: &ValidationConfig,
) -> Result<BacktestResult> {
        let sim_config = crate::PythonSimConfig {
            python_source: config.python_source.clone(),
            backtest_config: config.backtest_config.clone(),
            fee_config: config.fee_config.clone(),
            supplementary_data: config.supplementary_data.clone(),
            parameters: config.parameters.clone(),
            progress_callback: None,
            risk_manager: None,
            max_trade_log_size: None,
            orderbook_snapshots: None,
            multi_venue_data: None,
            historical_iv_surfaces: config.historical_iv_surfaces.clone(),
            option_instrument: config.option_instrument.clone(),
            option_spread: config.option_spread.clone(),
            underlying_series: config.underlying_series.clone(),
        };
        let result = crate::python_simulation::run(data, sim_config).await?;
        Ok(result.backtest_result)
}

/// Evaluate one FIXED parameter set (`config.parameters`) across `num_folds`
/// purged in-sample/out-of-sample window pairs, reusing the same window-
/// placement (`resolve_wf_window_offsets`, uniform placement -- no regime
/// classification needed for this per-candidate use) and purge-gap
/// (`resolve_wf_purge_gap_bars`) machinery `run_walk_forward` uses on
/// default parameters. Unlike `run_walk_forward` and
/// `run_walk_forward_with_ga`, this does NOT re-optimize anything per
/// window -- a GA candidate's parameters are already fixed by construction,
/// so each fold only needs `run_single_backtest` called twice (train slice,
/// test slice), not a fresh search. This is the primitive the GA's own
/// per-candidate fitness function uses to embed real OOS awareness into
/// selection itself -- see `GeneticConfig::embedded_oos_folds`'s doc
/// comment for why that matters.
///
/// Returns one `(is_result, oos_result)` pair per fold that could actually
/// be produced -- fewer than `num_folds` if the data is too short for all
/// of them, empty if the data is too short for even one (mirrors
/// `run_walk_forward`'s own tolerance for a window whose backtest fails:
/// skip it, don't abort the whole evaluation).
#[cfg(feature = "python")]
async fn evaluate_is_oos_folds(
    data: &[MarketData],
    config: &ValidationConfig,
    num_folds: usize,
) -> Vec<(BacktestResult, BacktestResult)> {
    let n = data.len();
    if num_folds == 0 || n < 100 {
        return Vec::new();
    }

    let window_size = n / num_folds;
    if window_size == 0 {
        return Vec::new();
    }
    let train_size = (window_size as f64 * WF_TRAIN_FRAC) as usize;
    let test_size = window_size - train_size;
    if train_size == 0 || test_size == 0 {
        return Vec::new();
    }

    let span_ms = match (data.first().map(market_data_timestamp), data.last().map(market_data_timestamp)) {
        (Some(first), Some(last)) => (last - first).num_milliseconds(),
        _ => 0,
    };
    let purge_gap_bars = resolve_wf_purge_gap_bars(n, span_ms, test_size);

    let offsets: Vec<usize> = resolve_wf_window_offsets(n, window_size, num_folds, test_size, None)
        .into_iter()
        .filter(|&offset| offset + train_size + purge_gap_bars + test_size <= n)
        .collect();

    let mut results = Vec::with_capacity(num_folds);
    for offset in offsets {
        if results.len() >= num_folds {
            break;
        }
        let train_start = offset;
        let train_end = offset + train_size;
        let test_start = train_end + purge_gap_bars;
        let test_end = (test_start + test_size).min(n);

        let is_result = match run_single_backtest(&data[train_start..train_end], config).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        let oos_result = match run_single_backtest(&data[test_start..test_end], config).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        results.push((is_result, oos_result));
    }
    results
}

/// Result of a cheap, standalone signal-density dry run (see
/// `quick_signal_dry_run`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickSignalDryRunResult {
    /// Number of bars the dry run actually simulated (after subsampling).
    pub bars_tested: usize,
    /// Trades generated on this sample.
    pub num_trades: usize,
    /// PnL on this sample -- informational only, NOT a verdict on
    /// profitability. The only thing this dry run is meant to answer is
    /// "does this strategy's entry logic ever fire on real data at all?".
    pub total_pnl: f64,
}

/// Cheap, standalone signal-density dry run: runs the strategy against a
/// (subsampled) slice of real market data and reports how many trades it
/// generates, WITHOUT walk-forward, GA, or Monte Carlo. This is exactly
/// `run_validation_pipeline`'s Stage 1 factored out as its own entrypoint, so
/// `tool_validate_strategy` (agentic workflow, `program/src/workflow/
/// executor.rs`) can catch a strategy whose entry conditions never fire on
/// real data BEFORE the model commits to `submit_full_backtest`'s
/// full-rigor, real-job pipeline -- instead of only discovering "0 trades"
/// after that job has already run to completion.
#[cfg(feature = "python")]
pub async fn quick_signal_dry_run(
    python_source: &str,
    market_data: &[MarketData],
    backtest_config: &BacktestConfig,
    fee_config: Option<&ExchangeFeeConfig>,
) -> Result<QuickSignalDryRunResult> {
    if market_data.is_empty() {
        anyhow::bail!("No market data available for signal dry run");
    }

    let config = ValidationConfig {
        python_source: python_source.to_string(),
        backtest_config: backtest_config.clone(),
        fee_config: fee_config.cloned(),
        ..Default::default()
    };

    // Same subsampling as Stage 1: keep the dry run within Python's 30s
    // compute_signals() timeout regardless of how much data was fetched.
    let quick_subsample_step = (market_data.len() / 150_000).max(1);
    let quick_data: std::borrow::Cow<'_, [MarketData]> = if quick_subsample_step > 1 {
        std::borrow::Cow::Owned(market_data.iter().step_by(quick_subsample_step).cloned().collect())
    } else {
        std::borrow::Cow::Borrowed(market_data)
    };

    let result = run_single_backtest(&quick_data, &config)
        .await
        .context("Signal dry run failed")?;

    Ok(QuickSignalDryRunResult {
        bars_tested: quick_data.len(),
        num_trades: result.num_trades,
        total_pnl: result.total_pnl,
    })
}

/// Calendar-day purge gap between a walk-forward window's train and test
/// slices, matching `CPCVConfig::purge_gap_days` (walkforward crate) exactly
/// -- CPCV already embargoes this way; plain walk-forward (used by pairs,
/// basket_arb, and any non-GA-schema job -- most jobs on this platform) did
/// not, letting a rolling-window indicator (a 20-day MA, a rolling z-score)
/// leak information across the train/test boundary: the last few
/// observations before the split and the first few after it are not
/// actually independent. See `run_walk_forward`'s doc comment for how this
/// gets converted from calendar days to a bar count for arbitrary candle
/// granularity.
const WF_PURGE_GAP_DAYS: f64 = 5.0;

/// Pure conversion of `WF_PURGE_GAP_DAYS` into a bar count for data spanning
/// `span_ms` over `n` bars, capped at half of `test_size` -- split out from
/// `run_walk_forward` so the arithmetic (the part most worth locking down
/// with tests) doesn't require a full async Python backtest harness to
/// exercise. `span_ms` is the timestamp delta between the first and last bar
/// in the dataset (0 or `n <= 1` falls back to a 1ms-per-bar assumption,
/// same as treating the gap as negligible rather than dividing by zero).
fn resolve_wf_purge_gap_bars(n: usize, span_ms: i64, test_size: usize) -> usize {
    let avg_bar_ms = if n > 1 { (span_ms as f64 / (n - 1) as f64).max(1.0) } else { 1.0 };
    let gap_ms = WF_PURGE_GAP_DAYS * 86_400_000.0;
    ((gap_ms / avg_bar_ms).round() as usize).min(test_size / 2)
}

/// Bar-offset step between successive walk-forward windows, chosen so `num_windows`
/// windows of `window_size` bars each span the FULL available range `n` -- the LAST
/// window's end lands at `n`, not partway through the data.
///
/// 2026-08 fix: the previous flat `window_size * 0.5` step, combined with the
/// `windows.len() < num_windows` loop-termination check every call site used, meant
/// the mechanical sweep stopped once it collected `num_windows` windows -- but at a
/// fixed 50% step, spanning the FULL range would need roughly `2 * num_windows`
/// windows, so the loop always terminated partway through (54-67% of the range,
/// depending on num_windows, verified numerically). The most recent 33-46% of every
/// requested backtest window was silently never walk-forward-tested. This directly
/// contradicted `stage1_training_only_end`'s own doc comment and test name
/// (`stage1_training_only_end_excludes_the_tail_walk_forward_treats_as_oos`), which
/// explicitly assumed the last window's test slice reaches `n`.
///
/// With `window_size = n / num_windows` (as every call site already computes it),
/// solving for the last window's offset to equal `n - window_size` generally works
/// out to `step == window_size` -- i.e. disjoint, back-to-back windows, not the old
/// 50%-overlap ones. That's a deliberate improvement, not just a side effect:
/// overlapping OOS test windows share bars, which correlates their errors and
/// overstates how much independent evidence the walk-forward pass actually gathered.
pub fn resolve_wf_step(n: usize, window_size: usize, num_windows: usize) -> usize {
    if num_windows <= 1 || window_size >= n {
        return window_size.max(1);
    }
    ((n - window_size) / (num_windows - 1)).max(1)
}

/// Run walk-forward analysis over market data, splitting into rolling windows.
///
/// Each window uses a train slice then (optionally) skips a `WF_PURGE_GAP_DAYS`-wide embargo
/// between train and test, then evaluates on the test slice. The gap consumes bars from the
/// dataset (i.e. windows span train + gap + test), but those gap bars are excluded from both
/// slices to reduce information leakage for rolling-window indicators. The strategy runs on
/// training data, then evaluated on test data, separated by the gap.
/// happens between windows (that's Phase 1.10 with GA).
#[cfg(feature = "python")]
async fn run_walk_forward(
    data: &[MarketData],
    config: &ValidationConfig,
    num_windows: usize,
    on_window_complete: impl Fn(usize, usize),
) -> Result<WalkForwardResult> {
    let n = data.len();
    if n < 100 {
        anyhow::bail!("Insufficient data for walk-forward analysis (need >= 100 ticks, got {})", n);
    }

    // Calculate window sizes
    // Each window covers (window_size = n / num_windows) ticks, spanning the FULL
    // range n across num_windows windows (see resolve_wf_step's doc comment).
    let window_size = n / num_windows.max(1);
    let train_size = (window_size as f64 * WF_TRAIN_FRAC) as usize;
    let test_size = window_size - train_size;

    // Convert WF_PURGE_GAP_DAYS to a bar count from the data's own average
    // spacing (capped at half the test window so a short/coarse-granularity
    // window can never purge the ENTIRE test slice away -- see
    // resolve_wf_purge_gap_bars's doc comment).
    let span_ms = match (data.first().map(market_data_timestamp), data.last().map(market_data_timestamp)) {
        (Some(first), Some(last)) => (last - first).num_milliseconds(),
        _ => 0,
    };
    let purge_gap_bars = resolve_wf_purge_gap_bars(n, span_ms, test_size);

    // Bar-based annualization factor derived from this dataset's OWN average
    // bar spacing (2026-08 annualization fix) -- every per-window OOS Sharpe
    // below previously annualized with a flat 365 regardless of actual
    // candle granularity, systematically understating the true annualized
    // Sharpe magnitude for anything trading faster than daily bars.
    let bars_per_year = metrics::significance::observations_per_year_from_span(n, Some(span_ms as f64 / 1000.0));

    // Regime-coverage signal (2026-08): classify the full series once so
    // each window can report its predominant regime -- see
    // window_regime_label/regime_coverage_narrative's doc comments.
    let regimes = quant_diagnostics::volatility_tercile_regimes(&simple_returns(&close_prices_from_market_data(data)), REGIME_WINDOW);

    // Window placement (2026-08 stratified-by-regime upgrade, Phase 2):
    // reserves a window per detected regime before falling back to uniform
    // spacing -- see resolve_wf_window_offsets' doc comment. Filtered against
    // the same bound the original while-loop enforced (offset + train + gap
    // + test <= n), since the placement function itself doesn't know about
    // the purge gap.
    let offsets: Vec<usize> = resolve_wf_window_offsets(n, window_size, num_windows, test_size, Some(&regimes))
        .into_iter()
        .filter(|&offset| offset + train_size + purge_gap_bars + test_size <= n)
        .collect();

    let mut windows: Vec<WalkForwardWindow> = Vec::new();
    let mut oos_results: Vec<BacktestResult> = Vec::new();
    let mut window_idx = 0usize;

    for offset in offsets {
        if windows.len() >= num_windows {
            break;
        }
        let train_start = offset;
        let train_end = offset + train_size;
        let test_start = train_end + purge_gap_bars;
        let test_end = (test_start + test_size).min(n);

        // Subsample each window slice to cap at ~500K ticks for speed.
        // Walk-forward is a relative comparison (train vs test), so subsampling
        // preserves the signal while being ~4-8x faster per window.
        // Returns Cow::Borrowed when no subsampling needed (avoids .to_vec()).
        fn subsample_slice(slice: &[MarketData]) -> std::borrow::Cow<'_, [MarketData]> {
            let step = (slice.len() / 500_000).max(1);
            if step > 1 {
                std::borrow::Cow::Owned(slice.iter().step_by(step).cloned().collect())
            } else {
                std::borrow::Cow::Borrowed(slice)
            }
        }

        // Run on training period (subsampled)
        let train_data = subsample_slice(&data[train_start..train_end]);
        let train_result = match run_single_backtest(&train_data, config).await {
            Ok(r) => r,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[WF] Window {} train backtest failed, skipping: {}", windows.len() + 1, e);
                continue;
            }
        };

        // Run on test period (subsampled)
        let test_data = subsample_slice(&data[test_start..test_end]);
        let test_result = match run_single_backtest(&test_data, config).await {
            Ok(r) => r,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[WF] Window {} test backtest failed, skipping: {}", windows.len() + 1, e);
                continue;
            }
        };

        let train_pnl = train_result.total_pnl;
        let test_pnl = test_result.total_pnl;
        let train_sharpe = train_result.sharpe_ratio.unwrap_or(0.0);
        // Bug 4: skip flat warmup prefix in equity curve when computing OOS Sharpe.
        // Strategies with long indicator periods (e.g. 200-bar MA) produce ~N flat ticks
        // at the start of each OOS window, diluting the tick-return-based Sharpe ratio.
        let test_sharpe = {
            let ec = &test_result.equity_curve;
            let base = ec.first().copied().unwrap_or(0.0);
            let warmup_end = ec.iter().position(|&v| (v - base).abs() > 1e-9).unwrap_or(0);
            let slice = if warmup_end > 0 { &ec[warmup_end..] } else { ec.as_slice() };
            if slice.len() >= 3 {
                let oos_ret: Vec<f64> = slice.windows(2)
                    .filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
                    .collect();
                if oos_ret.len() >= 2 {
                    let mean = oos_ret.iter().sum::<f64>() / oos_ret.len() as f64;
                    let var = oos_ret.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / (oos_ret.len() - 1) as f64;
                    let std = var.sqrt();
                    if std > 0.0 { ((mean / std) * bars_per_year.sqrt()).clamp(-20.0, 20.0) } else { 0.0 }
                } else { test_result.sharpe_ratio.unwrap_or(0.0).clamp(-20.0, 20.0) }
            } else { test_result.sharpe_ratio.unwrap_or(0.0).clamp(-20.0, 20.0) }
        };

        windows.push(WalkForwardWindow {
            train_start_idx: train_start,
            train_end_idx: train_end,
            test_start_idx: test_start,
            test_end_idx: test_end,
            train_start_date: data.get(train_start).map(market_data_timestamp),
            train_end_date: data.get(train_end.saturating_sub(1)).map(market_data_timestamp),
            test_start_date: data.get(test_start).map(market_data_timestamp),
            test_end_date: data.get(test_end.saturating_sub(1)).map(market_data_timestamp),
            train_pnl,
            train_sharpe,
            train_trades: train_result.num_trades,
            test_pnl,
            test_sharpe,
            test_trades: test_result.num_trades,
            test_win_rate: test_result.win_rate.unwrap_or(0.0),
            test_max_drawdown: test_result.max_drawdown,
            test_profit_factor: test_result.profit_factor.unwrap_or(0.0),
            test_regime: window_regime_label(&regimes, test_start, test_end),
        });

        oos_results.push(test_result);
        window_idx += 1;
        on_window_complete(window_idx, num_windows);
    }

    if windows.is_empty() {
        anyhow::bail!("Walk-forward produced no windows (data too short for {} windows)", num_windows);
    }

    // Aggregate metrics
    let avg_oos_sharpe = windows.iter().map(|w| w.test_sharpe).sum::<f64>()
        / windows.len() as f64;
    let avg_oos_pnl = windows.iter().map(|w| w.test_pnl).sum::<f64>()
        / windows.len() as f64;
    let avg_train_sharpe = windows.iter().map(|w| w.train_sharpe).sum::<f64>()
        / windows.len() as f64;

    // Bug #21 — emit the true signed degradation ratio. Clamping to >=0 hides
    // the case where OOS outperforms IS (which is a meaningful diagnostic).
    let overfitting_ratio = if avg_train_sharpe.abs() > 1e-9 {
        (avg_train_sharpe - avg_oos_sharpe) / avg_train_sharpe.abs()
    } else {
        // IS Sharpe is undefined (strategy barely traded in-sample).
        // Treat as worst-case: maximum degradation indicator.
        1.0
    };

    let oos_profitability_rate = windows.iter()
        .filter(|w| w.test_sharpe > 0.0)
        .count() as f64
        / windows.len() as f64;

    // Consistency: 1 / (1 + σ_sharpe) — measures uniformity of OOS Sharpe across windows.
    // Higher = more consistent (lower Sharpe spread). NOT % of profitable windows
    // (see oos_profitability_rate for that).
    let consistency_score = if windows.len() > 1 {
        let sharpes: Vec<f64> = windows.iter().map(|w| w.test_sharpe).collect();
        let variance = sharpes.iter()
            .map(|s| (s - avg_oos_sharpe).powi(2))
            .sum::<f64>() / (sharpes.len() - 1) as f64;
        let std_dev = variance.sqrt();
        (1.0 / (1.0 + std_dev)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Combine all OOS test results into a single aggregated result
    let initial_capital = config.backtest_config.trading.initial_capital;
    let oos_combined = if !oos_results.is_empty() {
        Some(combine_oos_results(&oos_results, initial_capital))
    } else {
        None
    };

    let tested_regimes: Vec<Option<quant_diagnostics::VolatilityRegime>> = windows.iter().map(|w| w.test_regime).collect();
    let regime_coverage_narrative = regime_coverage_narrative(&tested_regimes, &regimes);
    let regime_coverage_incomplete = regime_coverage_incomplete(&tested_regimes, &regimes);

    Ok(WalkForwardResult {
        windows,
        avg_oos_sharpe,
        avg_oos_pnl,
        avg_train_sharpe,
        overfitting_ratio,
        consistency_score,
        oos_profitability_rate,
        num_windows: window_idx,
        oos_combined_result: oos_combined,
        ga_trial_count: None,
        regime_coverage_narrative,
        regime_coverage_incomplete,
    })
}

/// Spawn a `WorkerPool` for parallel Python GA evaluation, writing market data
/// to a temporary file (SimBin preferred, bincode fallback).  Returns `None` if
/// the pool cannot be created or only one worker is requested.
#[cfg(feature = "python")]
fn spawn_worker_pool_for_data(
    data: &[MarketData],
    config: &ValidationConfig,
    data_label: &str,
) -> Option<genetic::worker_pool::WorkerPool> {
    let num_workers = std::env::var("GA_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| num_cpus::get().min(4))
        .max(1);

    if num_workers <= 1 {
        return None;
    }

    // Try SimBin (mmap-friendly, zero-copy)
    let data_path = std::env::temp_dir().join(format!("{data_label}.simbin"));
    let simbin_ok = (|| -> anyhow::Result<()> {
        let (sym, exch) = data.iter().find_map(|md| {
            if let MarketData::Trade(td) = md {
                Some((td.symbol.as_ref().to_string(), td.exchange.as_ref().to_string()))
            } else {
                None
            }
        }).unwrap_or_else(|| ("UNKNOWN".into(), "UNKNOWN".into()));

        let ticks: Vec<data_prep::SimulationTick> = data.iter().map(|md| {
            match md {
                MarketData::Trade(td) => data_prep::SimulationTick::new(
                    td.timestamp.timestamp_millis(),
                    td.price,
                    td.quantity,
                    td.side == dataloader::tick::Side::Sell,
                ),
                MarketData::Candle(c) => data_prep::SimulationTick::new(
                    c.timestamp.timestamp_millis(),
                    c.close,
                    c.volume,
                    false,
                ),
                MarketData::PoolSwap(s) => data_prep::SimulationTick::new(
                    s.timestamp.timestamp_millis(),
                    s.price(),
                    s.amount_in,
                    false,
                ),
                MarketData::Generic(g) => data_prep::SimulationTick::new(
                    g.timestamp_ms,
                    g.price,
                    g.volume.unwrap_or(0.0),
                    false,
                ),
                MarketData::OptionCandle(c) => data_prep::SimulationTick::new(
                    c.timestamp.timestamp_millis(),
                    c.close,
                    c.volume,
                    false,
                ),
            }
        }).collect();

        let mut writer = data_prep::binary_format::BinaryFileWriter::create(&data_path)?;
        writer.set_metadata(&sym, &exch);
        writer.write_ticks(&ticks)?;
        writer.finalize()?;
        Ok(())
    })();

    let (effective_path, is_simbin) = if simbin_ok.is_ok() {
        (data_path.to_string_lossy().to_string(), true)
    } else {
        let bincode_path = std::env::temp_dir().join(format!("{data_label}.bin"));
        match bincode::serialize(data) {
            Ok(encoded) => {
                if std::fs::write(&bincode_path, &encoded).is_ok() {
                    (bincode_path.to_string_lossy().to_string(), false)
                } else {
                    log_warn!(BACKTEST_LOGGER, "[VALIDATION] Failed to write GA data file for {}, falling back to sequential", data_label);
                    return None;
                }
            }
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] Failed to serialise GA data for {} ({}), falling back to sequential", data_label, e);
                return None;
            }
        }
    };

    // Total candidates this run will evaluate, for the DSR gate (see
    // genetic::dsr_gate_fitness's doc comment) -- same population*generations
    // convention already used at final-verdict time in engine.rs.
    let dsr_n_trials = config.ga_config.as_ref()
        .map(|g| g.population_size * g.generations);

    let worker_cfg = genetic::worker_pool::WorkerConfig {
        python_source: config.python_source.clone(),
        backtest_config: serde_json::to_value(&config.backtest_config).unwrap_or_default(),
        fee_config: config.fee_config.as_ref().and_then(|fc| serde_json::to_value(fc).ok()),
        supplementary_data: config.supplementary_data.clone(),
        fitness_weights: config.fitness_scorer_config_json.clone(),
        initial_capital: config.backtest_config.trading.initial_capital,
        market_data_path: effective_path,
        orderbook_data_path: None,
        simbin_format: is_simbin,
        strategy_type: None,
        precomputed_path: None,
        max_drawdown_hard_cap: config.max_drawdown_hard_cap.unwrap_or(0.40).clamp(0.05, 0.60),
        dsr_n_trials,
    };

    match genetic::worker_pool::WorkerPool::spawn(num_workers, worker_cfg) {
        Ok(pool) => {
            log_info!(BACKTEST_LOGGER,
                "[VALIDATION] Spawned {} GA worker processes for '{}' (simbin={})",
                pool.num_workers(), data_label, is_simbin);
            Some(pool)
        }
        Err(e) => {
            log_warn!(BACKTEST_LOGGER,
                "[VALIDATION] Failed to spawn worker pool for '{}' ({}), falling back to sequential", data_label, e);
            None
        }
    }
}

/// Estimate ruin probability from Monte Carlo fan chart.
///
/// Uses the fan chart's 5th percentile: if final equity at p5 is below
/// 50% of initial capital, that path constitutes "ruin".
#[allow(dead_code)]
fn estimate_ruin_probability(mc: &MonteCarloResult) -> f64 {
    if mc.fan_chart.is_empty() {
        return 0.0;
    }

    // Fan chart: each entry is [p5, p25, p50, p75, p95] at that trade step.
    // Initial capital is at step 0's p50.
    let initial_capital = mc.fan_chart.first()
        .map(|step| step[2]) // p50 at step 0
        .unwrap_or(10000.0);

    let ruin_threshold = initial_capital * 0.5; // 50% drawdown = ruin

    // Count how many steps the p5 (worst-case 5th percentile) path is below ruin
    let ruin_steps = mc.fan_chart.iter()
        .filter(|step| step[0] < ruin_threshold) // p5 < threshold
        .count();

    // Ruin probability approximation: fraction of path that's below threshold
    // This is conservative — if 5th percentile hits ruin, ~5% or more of paths do.
    if ruin_steps > 0 {
        // If the FINAL step's p5 is below ruin, estimate P(ruin) ≈ (p5_ruin/total) * 0.2
        let final_p5 = mc.fan_chart.last().map(|s| s[0]).unwrap_or(initial_capital);
        if final_p5 < ruin_threshold {
            // p5 is 5th percentile, so at least 5% of paths went below
            0.05_f64.max(ruin_steps as f64 / mc.fan_chart.len() as f64 * 0.5)
        } else {
            // Some intermediate steps touched ruin but recovered
            (ruin_steps as f64 / mc.fan_chart.len() as f64) * 0.05
        }
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Parameter schema extraction from Python source
// ---------------------------------------------------------------------------

/// Extract the `parameter_space()` dict from a Python strategy by running it via PyO3.
///
/// Returns `None` if the strategy doesn't define `parameter_space()`.
/// Returns `Some(Arc<ParameterSchema>)` if it does.
#[cfg(feature = "python")]
pub fn extract_parameter_schema(
    python_source: &str,
) -> Option<Arc<ParameterSchema>> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyModule};
    use std::collections::HashMap;

    Python::with_gil(|py| {
        // Inject SDK so `from trading_platform import ...` works
        if let Err(e) = strategy::strategies::inject_sdk(py) {
            log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: SDK injection failed: {:?}", e);
            return None;
        }

        // Pre-import numpy/pandas so user code can `import numpy as np`
        let _ = py.run_bound("import numpy; import pandas", None, None);

        // Load the user module. Unique name per call -- a fixed literal here
        // let a previous job's stale module (and its Strategy class) leak
        // into this compilation via a shared sys.modules entry; see
        // strategy crate's python_strategy.rs unique_module_name doc comment
        // for the full incident writeup.
        let module = match PyModule::from_code_bound(
            py, python_source, "strategy.py", &strategy::strategies::python_strategy::unique_module_name(),
        ) {
            Ok(m) => m,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: module compile error: {}", e);
                return None;
            }
        };

        // Find the Strategy class
        let strategy_class = match module.getattr("Strategy") {
            Ok(c) => c,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: no Strategy class: {}", e);
                return None;
            }
        };
        let instance = match strategy_class.call0() {
            Ok(i) => i,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: instantiation failed: {}", e);
                return None;
            }
        };

        // Check if parameter_space exists
        if !instance.hasattr("parameter_space").unwrap_or(false) {
            return None;
        }

        // Call parameter_space()
        let ps_result = match instance.call_method0("parameter_space") {
            Ok(r) => r,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: parameter_space() call failed: {}", e);
                return None;
            }
        };
        let ps_dict = match ps_result.downcast::<PyDict>() {
            Ok(d) => d,
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: parameter_space() didn't return dict: {}", e);
                return None;
            }
        };

        // Convert Python dict → HashMap<String, HashMap<String, serde_json::Value>>
        //
        // FIX (2026-08-27): every conversion step below used to end in a bare
        // `?` inside this `Python::with_gil` closure -- since the closure
        // itself returns `Option<Arc<ParameterSchema>>`, that silently
        // short-circuited the WHOLE function to `None` on ANY parse failure,
        // with zero logging anywhere. Confirmed live: a real strategy whose
        // `parameter_space()` returned `{"param": [5, 8, 12]}` (a list of
        // discrete choices -- a real, once-used convention, not a typo) hit
        // the `value.downcast::<PyDict>()` failure on every single parameter
        // and silently produced `has_ga = false` for the whole job. The job
        // still ran and "completed" normally (default-parameter backtest,
        // no error surfaced anywhere), so from the outside this was
        // indistinguishable from "this strategy just doesn't define
        // parameter_space()" -- diagnosing it took directly instrumenting
        // this function, because every one of the several log_warn! calls
        // elsewhere in this same function never fired. Explicit match arms
        // now log exactly which parameter and which step failed.
        let mut raw_dict: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();

        for (key, value) in ps_dict.iter() {
            let param_name: String = match key.extract() {
                Ok(s) => s,
                Err(e) => {
                    log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: a parameter_space() key wasn't a string: {}", e);
                    return None;
                }
            };
            let param_dict = match value.downcast::<PyDict>() {
                Ok(d) => d,
                Err(_) => {
                    log_warn!(
                        BACKTEST_LOGGER,
                        "[VALIDATION] extract_parameter_schema: parameter_space()['{}'] isn't a dict (got {}) -- the expected shape is \
                         {{\"type\": \"int\"|\"float\"|\"bool\", \"min\": ..., \"max\": ..., \"default\": ...}}, not a list of discrete choices",
                        param_name, value.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "<unknown type>".to_string())
                    );
                    return None;
                }
            };

            let mut entry: HashMap<String, serde_json::Value> = HashMap::new();
            for (k, v) in param_dict.iter() {
                let k_str: String = match k.extract() {
                    Ok(s) => s,
                    Err(e) => {
                        log_warn!(BACKTEST_LOGGER, "[VALIDATION] extract_parameter_schema: a key inside parameter_space()['{}'] wasn't a string: {}", param_name, e);
                        return None;
                    }
                };
                let val = python_value_to_json(&v);
                entry.insert(k_str, val);
            }

            raw_dict.insert(param_name, entry);
        }

        match ParameterSchema::from_raw_dict(&raw_dict) {
            Ok(schema) => {
                log_info!(BACKTEST_LOGGER, "[VALIDATION] Extracted parameter_space: {} params: {:?}", schema.len(), schema.params.iter().map(|p| &p.name).collect::<Vec<_>>());
                Some(schema)
            }
            Err(e) => {
                log_warn!(BACKTEST_LOGGER, "[VALIDATION] Failed to parse parameter_space: {}", e);
                None
            }
        }
    })
}

/// Convert a Python value to serde_json::Value.
/// IMPORTANT: bool must be checked before i64 (Python bool is a subclass of int),
/// and i64 must be checked before f64 so that Python integers are stored as JSON
/// integers. If stored as floats, serde_json's as_i64() returns None for 1000.0,
/// which would silently corrupt parameter_space() bounds to their fallback defaults.
#[cfg(feature = "python")]
fn python_value_to_json(value: &pyo3::Bound<'_, pyo3::PyAny>) -> serde_json::Value {
    use pyo3::types::PyAnyMethods;
    if let Ok(v) = value.extract::<bool>() {
        serde_json::Value::from(v)
    } else if let Ok(v) = value.extract::<i64>() {
        serde_json::Value::from(v)
    } else if let Ok(v) = value.extract::<f64>() {
        serde_json::Value::from(v)
    } else if let Ok(v) = value.extract::<String>() {
        serde_json::Value::from(v)
    } else {
        serde_json::Value::Null
    }
}

/// Stub for non-python builds.
#[cfg(not(feature = "python"))]
pub fn extract_parameter_schema(_python_source: &str) -> Option<Arc<ParameterSchema>> {
    None
}

// ---------------------------------------------------------------------------
// Look-Ahead Bias Detection
// ---------------------------------------------------------------------------

/// Cheap textual heuristic: could this strategy source produce different
/// results across identical runs? Used to gate the (costly) determinism
/// pre-check to strategies that actually use randomness. Erring toward
/// false positives is fine — it just runs the extra check needlessly; the
/// cost only lands on strategies mentioning these APIs (regime models, MC
/// sampling), never on plain indicator strategies.
#[cfg(feature = "python")]
fn source_may_be_stochastic(source: &str) -> bool {
    const MARKERS: &[&str] = &[
        "random", "default_rng", "RandomState", "np.random", "numpy.random",
        "shuffle", "permutation", "KMeans", "GaussianMixture", ".sample(",
        "choice(", "rand(", "randn(", "randint(",
    ];
    MARKERS.iter().any(|m| source.contains(m))
}

/// Detect look-ahead bias by comparing signals generated from truncated vs full data.
///
/// Runs `compute_signals()` on the first 60% of data in isolation, then on the full
/// dataset. If the signals for the first 60% differ when future data is present,
/// the strategy is using information it wouldn't have in live trading.
///
/// Returns `Ok(true)` if look-ahead bias is detected, `Ok(false)` if clean.
/// Returns `Err` if the strategy fails to produce signals (not a look-ahead issue).
#[cfg(feature = "python")]
pub async fn detect_lookahead_bias(
    market_data: &[MarketData],
    config: &ValidationConfig,
) -> Result<bool> {
    use crate::python_simulation;

    let n = market_data.len();
    if n < 200 {
        return Ok(false);
    }

    let split = (n as f64 * 0.6) as usize;
    let truncated_data = &market_data[..split];

    let sim_config = || crate::PythonSimConfig {
        python_source: config.python_source.clone(),
        backtest_config: config.backtest_config.clone(),
        fee_config: config.fee_config.clone(),
        supplementary_data: config.supplementary_data.clone(),
        parameters: config.parameters.clone(),
        progress_callback: None,
        risk_manager: None,
        max_trade_log_size: None,
        orderbook_snapshots: None,
        historical_iv_surfaces: None,
        multi_venue_data: None,
        option_instrument: None,
        option_spread: None,
        underlying_series: None,
    };

    let result_truncated = python_simulation::run(truncated_data, sim_config()).await?;

    // Determinism pre-check — ONLY for strategies whose source uses randomness
    // (HMM/EM/k-means regime models are the motivating case). The look-ahead
    // test compares a truncated vs a full run and blames any divergence on
    // future-data leakage; a NON-deterministic strategy (unseeded RNG) diverges
    // for reasons unrelated to look-ahead, producing a FALSE positive. When the
    // source can't be stochastic we skip this entirely so the common
    // deterministic case pays nothing (Stage 0 already runs on full,
    // non-subsampled data — a blanket extra run would be wasteful). When it can,
    // one extra run on the 60% slice catches it; we then report non-determinism
    // as its own actionable condition (seed your RNG) rather than mislabeling it.
    if source_may_be_stochastic(&config.python_source) {
        let result_truncated_2 = python_simulation::run(truncated_data, sim_config()).await?;
        let t1 = result_truncated.backtest_result.num_trades;
        let t2 = result_truncated_2.backtest_result.num_trades;
        if t1 != t2 {
            anyhow::bail!(
                "strategy is non-deterministic: two runs on identical data produced \
                 different results ({t1} vs {t2} trades). Seed every stochastic step \
                 (e.g. `rng = np.random.default_rng(0)` and initialize EM/k-means from \
                 it) so results are reproducible — the look-ahead test cannot run on a \
                 non-deterministic strategy."
            );
        }
    }

    let result_full = python_simulation::run(market_data, sim_config()).await?;

    let trades_truncated = result_truncated.backtest_result.num_trades;
    let cutoff_ts = truncated_data.last().map(|d| match d {
        MarketData::Candle(c) => c.timestamp,
        MarketData::Trade(t) => t.timestamp,
        MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or_default(),
        MarketData::PoolSwap(s) => s.timestamp,
        MarketData::OptionCandle(c) => c.timestamp,
    });
    let trades_full_in_window = match cutoff_ts {
        Some(ts) => result_full
            .backtest_result
            .trade_log
            .iter()
            .filter(|t| t.entry_time <= ts)
            .count(),
        None => result_full.backtest_result.trade_log.len(),
    };

    if trades_truncated == 0 && trades_full_in_window == 0 {
        return Ok(false);
    }

    let trade_diff = (trades_truncated as i64 - trades_full_in_window as i64).unsigned_abs();
    let max_trades = trades_truncated.max(trades_full_in_window).max(1);
    let divergence_ratio = trade_diff as f64 / max_trades as f64;

    Ok(divergence_ratio > 0.05)
}

#[cfg(not(feature = "python"))]
pub async fn detect_lookahead_bias(
    _market_data: &[MarketData],
    _config: &ValidationConfig,
) -> Result<bool> {
    Ok(false)
}

// ---------------------------------------------------------------------------
// Stub for non-python builds
// ---------------------------------------------------------------------------

#[cfg(not(feature = "python"))]
pub async fn run_validation_pipeline(
    _market_data: &[MarketData],
    _config: ValidationConfig,
) -> Result<ValidationResult> {
    anyhow::bail!("Python feature not enabled — cannot run Python validation pipeline")
}

/// Result of tick-data validation comparing candle-sim results with tick-level replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickValidationResult {
    pub candle_sharpe: f64,
    pub tick_sharpe: f64,
    pub sharpe_divergence_pct: f64,
    pub candle_pnl: f64,
    pub tick_pnl: f64,
    pub pnl_divergence_pct: f64,
    pub flagged: bool,
}

/// Validate a candle-sim result against tick-level data if available (Item 5).
///
/// Attempts to load tick data for the symbol/exchange. If found, runs the candle simulator
/// on tick-level candles (1-minute or finer) and compares with the original candle result.
/// Returns None if tick data unavailable.
pub async fn validate_on_tick_data(
    candle_result: &crate::candle_sim::CandleSimResult,
    _symbol: &str,
    _exchange: &str,
    _config: &BacktestConfig,
) -> Option<TickValidationResult> {
    // Tick data loading depends on database connectivity — this is a placeholder
    // that returns None when tick data is unavailable (common in backtesting-only env).
    // When tick data IS available, it would:
    // 1. Load tick data from DB for the symbol/exchange/time range
    // 2. Aggregate into 1-minute candles
    // 3. Run CandleSimulator with same params on 1-min data
    // 4. Compare Sharpe/PnL with the candle_result

    let _ = candle_result;

    // For now, provide a stub that demonstrates the interface.
    // Real implementation requires a database connection and tick data provider.
    None
}

/// Compare candle-level and tick-level results and produce divergence metrics.
pub fn compute_tick_divergence(
    candle_result: &crate::candle_sim::CandleSimResult,
    tick_result: &crate::candle_sim::CandleSimResult,
) -> TickValidationResult {
    let sharpe_div = if candle_result.sharpe_ratio.abs() > 1e-10 {
        ((candle_result.sharpe_ratio - tick_result.sharpe_ratio) / candle_result.sharpe_ratio * 100.0).abs()
    } else {
        0.0
    };

    let pnl_div = if candle_result.total_pnl.abs() > 1e-10 {
        ((candle_result.total_pnl - tick_result.total_pnl) / candle_result.total_pnl * 100.0).abs()
    } else {
        0.0
    };

    TickValidationResult {
        candle_sharpe: candle_result.sharpe_ratio,
        tick_sharpe: tick_result.sharpe_ratio,
        sharpe_divergence_pct: sharpe_div,
        candle_pnl: candle_result.total_pnl,
        tick_pnl: tick_result.total_pnl,
        pnl_divergence_pct: pnl_div,
        flagged: sharpe_div > 30.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceeds_param_complexity_gate_passes_a_well_sampled_candidate() {
        // 3 tunable parameters, 100 trades -> 0.03 params/trade, well under
        // the 0.1 (10 trades/parameter) threshold.
        assert!(!exceeds_param_complexity_gate(3, 100));
    }

    #[test]
    fn exceeds_param_complexity_gate_flags_a_thin_sample() {
        // 6 tunable parameters, only 20 trades -> 0.3 params/trade, well over
        // the threshold -- exactly the "many parameters, few trades" shape
        // the gate exists to catch.
        assert!(exceeds_param_complexity_gate(6, 20));
    }

    #[test]
    fn exceeds_param_complexity_gate_is_exclusive_at_the_exact_threshold() {
        // 1 parameter, 10 trades -> exactly 0.1, the threshold itself --
        // strictly greater-than, so sitting exactly on the line still passes.
        assert!(!exceeds_param_complexity_gate(1, 10));
        // Just over the line (11 params/trade ratio 0.1000...1) should flag.
        assert!(exceeds_param_complexity_gate(101, 1000));
    }

    #[test]
    fn exceeds_param_complexity_gate_treats_zero_trades_as_one_trade() {
        // Guards the division: with 0 trades, the denominator floors to 1
        // rather than dividing by zero, so any positive parameter count with
        // zero trades is flagged (extremely thin/no sample).
        assert!(exceeds_param_complexity_gate(1, 0));
        assert!(!exceeds_param_complexity_gate(0, 0));
    }

    #[cfg(feature = "python")]
    #[test]
    fn stochastic_source_detection() {
        // Regime/MC strategies that use randomness are flagged for the
        // determinism pre-check...
        assert!(source_may_be_stochastic("rng = np.random.default_rng(0)"));
        assert!(source_may_be_stochastic("from numpy.random import default_rng"));
        assert!(source_may_be_stochastic("labels = KMeans(2).fit_predict(x)"));
        assert!(source_may_be_stochastic("s = returns.sample(100)"));
        assert!(source_may_be_stochastic("idx = np.random.choice(n)"));
        // ...but a plain indicator strategy is not (no extra run for it).
        let indicator = "signals = np.zeros(n)\nema = pd.Series(prices).ewm(span=20).mean()";
        assert!(!source_may_be_stochastic(indicator));
    }

    // ── evaluate_is_oos_folds (embedded IS/OOS folding, guard clauses) ──
    // Only the early-return guard clauses are exercised here (no real
    // PyO3 backtest fixture) -- num_folds=0 and too-little-data both
    // return before ever touching Python, same degenerate-input caution
    // style as resolve_wf_purge_gap_bars's own tests.

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn evaluate_is_oos_folds_returns_empty_for_zero_folds() {
        let config = ValidationConfig::default();
        let folds = evaluate_is_oos_folds(&[], &config, 0).await;
        assert!(folds.is_empty());
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn evaluate_is_oos_folds_returns_empty_for_insufficient_data() {
        // n < 100 -- the same floor run_walk_forward enforces.
        let config = ValidationConfig::default();
        let folds = evaluate_is_oos_folds(&[], &config, 3).await;
        assert!(folds.is_empty());
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.python_source.is_empty());
        assert!(config.fee_config.is_none());
        assert!(config.supplementary_data.is_empty());
        assert!(config.parameters.is_empty());
        assert_eq!(config.wf_windows, 0);
        assert_eq!(config.mc_runs, 1000);
        assert!((config.min_oos_sharpe - 0.5).abs() < f64::EPSILON);
        assert!((config.max_ruin_probability - 0.20).abs() < f64::EPSILON);
        assert!(config.progress_callback.is_none());
        assert!(config.parameter_schema.is_none());
        assert!(config.ga_config.is_none());
        assert!(config.fitness_scorer.is_none());
        assert_eq!(config.mc_min_trades, 30);
        assert!((config.mc_confidence_level - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn stage1_training_only_end_never_exceeds_wf_train_frac_of_total() {
        for total in [0usize, 1, 2, 10, 1000, 1_000_000, 12_345] {
            let boundary = stage1_training_only_end(total);
            assert!(boundary <= total, "boundary {} must not exceed total {}", boundary, total);
            if total > 0 {
                assert!(boundary >= 1, "boundary must be at least 1 for non-empty data (total={})", total);
            } else {
                assert_eq!(boundary, 0, "empty data must yield an empty training slice");
            }
        }
    }

    #[test]
    fn stage1_training_only_end_matches_wf_train_frac_ratio() {
        // For a large total, the boundary should land close to WF_TRAIN_FRAC
        // of the range — i.e. Stage 1 sees roughly the same "training" portion
        // run_walk_forward's first window treats as train, not the tail that
        // becomes the most recent OOS test window.
        let total = 100_000;
        let boundary = stage1_training_only_end(total);
        let ratio = boundary as f64 / total as f64;
        assert!((ratio - WF_TRAIN_FRAC).abs() < 0.01, "ratio {} should be close to WF_TRAIN_FRAC {}", ratio, WF_TRAIN_FRAC);
    }

    #[test]
    fn stage1_training_only_end_excludes_the_tail_walk_forward_treats_as_oos() {
        // The core regression this fix targets: Stage 1 must never see the
        // final slice of data that run_walk_forward's last window uses as its
        // test (OOS) period. Concretely, the boundary must be strictly less
        // than the total length whenever there's enough data for a real
        // train/test split to matter.
        let total = 500_000;
        let boundary = stage1_training_only_end(total);
        assert!(boundary < total, "Stage 1 must not see the full range — some tail must be excluded");
        // The excluded tail should be roughly (1 - WF_TRAIN_FRAC) of the range.
        let excluded_frac = (total - boundary) as f64 / total as f64;
        assert!((excluded_frac - (1.0 - WF_TRAIN_FRAC)).abs() < 0.01);
    }

    // ---------------------------------------------------------------------
    // resolve_auto_wf_windows — calendar-time-aware walk-forward window count
    // ---------------------------------------------------------------------

    #[test]
    fn resolve_auto_wf_windows_short_calendar_span_stays_low_even_with_huge_bar_count() {
        // 3 months of 1-second bars: enormous data_len, but only one quarter
        // of real calendar history -- the core regression this heuristic
        // fixes. Bar-count-only sizing would have picked 8 here.
        let calendar_days = 90;
        let data_len = 5_000_000;
        assert_eq!(resolve_auto_wf_windows(calendar_days, data_len, None), 3);
    }

    #[test]
    fn resolve_auto_wf_windows_long_calendar_span_is_capped_by_sparse_bar_count() {
        // 10 years of daily bars (~2500 bars): calendar time alone would want
        // 12 windows, but there isn't enough data per window at that count --
        // the bar-count floor must win.
        let calendar_days = 3650;
        let data_len = 2500;
        let windows = resolve_auto_wf_windows(calendar_days, data_len, None);
        assert!(windows <= 8usize.min(data_len / 500), "must respect the bar-count ceiling");
        assert!(windows >= 3, "must never drop below the stated minimum of 3");
    }

    #[test]
    fn resolve_auto_wf_windows_matching_calendar_span_and_bar_count_lands_at_the_intended_target() {
        // 2 years, plenty of bars: both constraints agree around 8.
        let calendar_days = 730;
        let data_len = 500_000;
        assert_eq!(resolve_auto_wf_windows(calendar_days, data_len, None), 8);
    }

    #[test]
    fn resolve_auto_wf_windows_never_returns_below_three() {
        for calendar_days in [0usize, 1, 30, 89] {
            for data_len in [0usize, 1, 100, 999, 1000, 1500] {
                let windows = resolve_auto_wf_windows(calendar_days, data_len, None);
                assert!(windows >= 3, "calendar_days={} data_len={} produced {} windows, expected >= 3", calendar_days, data_len, windows);
            }
        }
    }

    #[test]
    fn resolve_auto_wf_windows_never_exceeds_twelve() {
        assert!(resolve_auto_wf_windows(100_000, 10_000_000, None) <= 12);
    }

    // ---------------------------------------------------------------------
    // resolve_auto_wf_windows — regime-diversity signal (`close_prices`)
    // ---------------------------------------------------------------------

    #[test]
    fn regime_segment_count_is_zero_for_empty_or_all_none() {
        assert_eq!(regime_segment_count(&[]), 0);
        assert_eq!(regime_segment_count(&[None, None, None]), 0);
    }

    #[test]
    fn regime_segment_count_counts_a_single_segment_once() {
        use quant_diagnostics::VolatilityRegime;
        let regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Low), Some(VolatilityRegime::Low)];
        assert_eq!(regime_segment_count(&regimes), 1);
    }

    #[test]
    fn regime_segment_count_counts_each_transition() {
        use quant_diagnostics::VolatilityRegime;
        let regimes = vec![
            Some(VolatilityRegime::Low), Some(VolatilityRegime::Low),
            Some(VolatilityRegime::Medium),
            Some(VolatilityRegime::High), Some(VolatilityRegime::High),
            Some(VolatilityRegime::Low),
        ];
        assert_eq!(regime_segment_count(&regimes), 4);
    }

    #[test]
    fn regime_segment_count_skips_none_gaps_without_fragmenting_a_segment() {
        use quant_diagnostics::VolatilityRegime;
        // A None gap (e.g. missing bars) mid-run of the same label must not
        // be counted as a second segment.
        let regimes = vec![Some(VolatilityRegime::Low), None, None, Some(VolatilityRegime::Low)];
        assert_eq!(regime_segment_count(&regimes), 1);
    }

    #[test]
    fn resolve_auto_wf_windows_none_close_prices_matches_prior_pure_calendar_behavior() {
        let calendar_days = 730;
        let data_len = 500_000;
        assert_eq!(
            resolve_auto_wf_windows(calendar_days, data_len, None),
            resolve_auto_wf_windows(calendar_days, data_len, Some(&[]))
        );
    }

    #[test]
    fn resolve_auto_wf_windows_raises_target_for_a_synthetic_multi_regime_series() {
        // Short calendar span (90 days -> pure-calendar target would be 3),
        // but a price series that visibly cycles through several distinct
        // volatility episodes (long flat stretches, then a violent stretch,
        // repeated) -- the regime-segment floor should raise the target
        // above the calendar-only baseline, still within the bar ceiling.
        let mut prices = Vec::new();
        let mut price = 100.0_f64;
        for cycle in 0..6 {
            let step = if cycle % 2 == 0 { 0.01 } else { 5.0 };
            for i in 0..40 {
                price += if i % 2 == 0 { step } else { -step };
                prices.push(price);
            }
        }
        let calendar_days = 90;
        let data_len = prices.len();
        let baseline = resolve_auto_wf_windows(calendar_days, data_len, None);
        let regime_aware = resolve_auto_wf_windows(calendar_days, data_len, Some(&prices));
        assert!(
            regime_aware >= baseline,
            "regime-aware target ({regime_aware}) should never fall below the pure-calendar baseline ({baseline})"
        );
    }

    #[test]
    fn resolve_auto_wf_windows_bar_ceiling_still_wins_when_segments_are_large() {
        // Sparse data (data_len < 1000 -> bar_ceiling is 3) with a price
        // series engineered to produce many regime segments -- the ceiling
        // must still cap the result at 3, proving segments can raise the
        // target but never bypass the existing statistical-power floor.
        let mut prices = Vec::new();
        let mut price = 100.0_f64;
        for i in 0..200 {
            let step = if (i / 5) % 2 == 0 { 0.01 } else { 5.0 };
            price += if i % 2 == 0 { step } else { -step };
            prices.push(price);
        }
        let windows = resolve_auto_wf_windows(9999, prices.len(), Some(&prices));
        assert_eq!(windows, 3, "bar-count ceiling for sparse data must still win over any regime-segment floor");
    }

    // ---------------------------------------------------------------------
    // Walk-forward train/test purge gap (embargo)
    // ---------------------------------------------------------------------

    #[test]
    fn resolve_wf_purge_gap_bars_is_nonzero_for_daily_bars() {
        // 400 daily bars spanning ~400 days -> avg_bar_ms ~= 1 day, so a
        // 5-day gap should land at (roughly) 5 bars.
        let n = 400;
        let span_ms = 400 * 86_400_000i64;
        let test_size = 60; // plenty of headroom vs. the 5-bar gap
        let gap = resolve_wf_purge_gap_bars(n, span_ms, test_size);
        assert_eq!(gap, 5, "daily bars should convert WF_PURGE_GAP_DAYS to ~5 bars, got {gap}");
    }

    #[test]
    fn resolve_wf_purge_gap_bars_scales_down_for_finer_granularity() {
        // Hourly bars: 5 days = 120 hours = 120 bars at this granularity --
        // the SAME calendar embargo should mean more bars, not the same
        // literal bar count, when the data is more fine-grained.
        let n = 24 * 400; // 400 days of hourly bars
        let span_ms = 400 * 86_400_000i64;
        let test_size = 10_000; // large enough not to trigger the cap
        let gap = resolve_wf_purge_gap_bars(n, span_ms, test_size);
        assert_eq!(gap, 120, "hourly bars should convert 5 calendar days to 120 bars, got {gap}");
    }

    #[test]
    fn resolve_wf_purge_gap_bars_never_exceeds_half_the_test_window() {
        // A short/coarse-granularity window (data_len small enough that the
        // computed gap would exceed the test slice) must be capped, not
        // allowed to swallow the entire (or more than the entire) test
        // window -- a smaller-than-intended embargo is a much better failure
        // mode than a walk-forward window with zero or negative test bars.
        let n = 20;
        let span_ms = 20 * 86_400_000i64; // daily bars, gap would want to be ~5
        let test_size = 4; // smaller than the natural 5-bar gap
        let gap = resolve_wf_purge_gap_bars(n, span_ms, test_size);
        assert!(gap <= test_size / 2, "gap {gap} must not exceed half of test_size {test_size}");
    }

    #[test]
    fn resolve_wf_purge_gap_bars_handles_degenerate_span_without_panicking() {
        // n <= 1 or span_ms == 0 must fall back gracefully (1ms-per-bar
        // assumption, so the nominal gap comes out huge and immediately
        // hits the half-of-test_size cap) -- never divide by zero or panic,
        // and always stay within the same bound every other case respects.
        for (n, span_ms, test_size) in [(0usize, 0i64, 100usize), (1, 0, 100), (100, 0, 100)] {
            let gap = resolve_wf_purge_gap_bars(n, span_ms, test_size);
            assert!(gap <= test_size / 2, "n={n} span_ms={span_ms} test_size={test_size} produced ungapped gap={gap}");
        }
    }

    // ---------------------------------------------------------------------
    // resolve_wf_step — full-range walk-forward coverage fix
    // ---------------------------------------------------------------------

    #[test]
    fn resolve_wf_step_last_window_reaches_the_end_of_the_data() {
        // Bounded by `num_windows` bars of integer-division truncation loss
        // (window_size = n / num_windows drops the remainder) -- negligible
        // in practice, not a real coverage gap the way the old formula's
        // 33-46%-of-the-range shortfall was.
        for (n, num_windows) in [(10_000usize, 3usize), (10_000, 8), (10_000, 12), (1_000_000, 8), (997, 5)] {
            let window_size = n / num_windows;
            let step = resolve_wf_step(n, window_size, num_windows);
            let last_offset = (num_windows - 1) * step;
            let last_end = last_offset + window_size;
            assert!(
                last_end <= n && last_end >= n.saturating_sub(num_windows),
                "n={n} num_windows={num_windows} window_size={window_size} step={step}: \
                 last window ends at {last_end}, expected within {num_windows} bars of {n} (full-range coverage)"
            );
        }
    }

    #[test]
    fn resolve_wf_step_produces_disjoint_back_to_back_windows() {
        // With window_size = n / num_windows, spanning the full range means
        // step == window_size (no overlap) -- a deliberate improvement over
        // the old 50%-overlap behavior (see the function's own doc comment).
        let n = 10_000;
        let num_windows = 8;
        let window_size = n / num_windows;
        let step = resolve_wf_step(n, window_size, num_windows);
        assert_eq!(step, window_size);
    }

    #[test]
    fn resolve_wf_step_single_window_returns_window_size() {
        assert_eq!(resolve_wf_step(1000, 500, 1), 500);
        assert_eq!(resolve_wf_step(1000, 500, 0), 500);
    }

    #[test]
    fn resolve_wf_step_never_returns_zero() {
        // A zero step would infinite-loop every call site's `offset += step`
        // sweep -- this is the hard safety floor, not just a nicety.
        for (n, window_size, num_windows) in [(100usize, 100usize, 2usize), (10, 10, 5), (0, 0, 3)] {
            assert!(resolve_wf_step(n, window_size, num_windows) >= 1);
        }
    }

    // ---------------------------------------------------------------------
    // regime_segments / resolve_wf_window_offsets
    // ---------------------------------------------------------------------

    fn make_regimes(spec: &[(usize, quant_diagnostics::VolatilityRegime)]) -> Vec<Option<quant_diagnostics::VolatilityRegime>> {
        spec.iter().flat_map(|(count, label)| std::iter::repeat(Some(*label)).take(*count)).collect()
    }

    #[test]
    fn regime_segments_splits_on_label_change_and_skips_none_gaps() {
        use quant_diagnostics::VolatilityRegime::{High, Low, Medium};
        let mut regimes = make_regimes(&[(5, Low), (3, High), (4, Medium)]);
        // Insert a None gap inside the Low run -- must not split it.
        regimes[2] = None;
        let segments = regime_segments(&regimes);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].2, Low);
        assert_eq!(segments[0].0, 0);
        assert_eq!(segments[0].1, 4); // still spans the whole Low run despite the gap at index 2
        assert_eq!(segments[1].2, High);
        assert_eq!(segments[2].2, Medium);
    }

    #[test]
    fn regime_segment_count_matches_regime_segments_len() {
        use quant_diagnostics::VolatilityRegime::{High, Low, Medium};
        let regimes = make_regimes(&[(5, Low), (3, High), (4, Medium), (2, Low)]);
        assert_eq!(regime_segment_count(&regimes), regime_segments(&regimes).len());
        assert_eq!(regime_segment_count(&regimes), 4);
    }

    #[test]
    fn resolve_wf_window_offsets_none_regimes_matches_prior_uniform_behavior() {
        let n = 10_000;
        let num_windows = 8;
        let window_size = n / num_windows;
        let step = resolve_wf_step(n, window_size, num_windows);
        let expected: Vec<usize> = (0..num_windows).map(|i| (i * step).min(n - window_size)).collect();
        assert_eq!(resolve_wf_window_offsets(n, window_size, num_windows, window_size / 3, None), expected);
    }

    #[test]
    fn resolve_wf_window_offsets_places_a_window_in_every_present_regime() {
        use quant_diagnostics::VolatilityRegime::{High, Low, Medium};
        // 900 bars: a long Low run, then a short High spike, then Medium --
        // uniform spacing alone (3 evenly-spaced windows) would very plausibly
        // miss the short High segment entirely.
        let regimes = make_regimes(&[(600, Low), (60, High), (240, Medium)]);
        let n = regimes.len();
        let window_size = 150;
        let test_size = 45;
        let num_windows = 3;
        let offsets = resolve_wf_window_offsets(n, window_size, num_windows, test_size, Some(&regimes));

        let segments = regime_segments(&regimes);
        let mut covered: Vec<quant_diagnostics::VolatilityRegime> = Vec::new();
        for &offset in &offsets {
            let test_start = offset + window_size - test_size;
            let test_end = offset + window_size;
            let mid = (test_start + test_end) / 2;
            if let Some(label) = regimes.get(mid.min(n - 1)).copied().flatten() {
                if !covered.contains(&label) {
                    covered.push(label);
                }
            }
        }
        for (_, _, label) in &segments {
            assert!(
                covered.contains(label),
                "regime {:?} present in the data but no window's test slice landed in it (offsets={:?})",
                label, offsets
            );
        }
        assert!(covered.contains(&Low) && covered.contains(&High) && covered.contains(&Medium));
    }

    #[test]
    fn resolve_wf_window_offsets_windows_stay_disjoint_with_regimes() {
        use quant_diagnostics::VolatilityRegime::{High, Low, Medium};
        let regimes = make_regimes(&[(300, Low), (50, High), (300, Medium), (50, Low), (300, High)]);
        let n = regimes.len();
        let window_size = 120;
        let offsets = resolve_wf_window_offsets(n, window_size, 6, 36, Some(&regimes));
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                let (a, b) = (offsets[i], offsets[j]);
                assert!(
                    a + window_size <= b || b + window_size <= a,
                    "windows at {} and {} overlap (window_size={})", a, b, window_size
                );
            }
        }
    }

    #[test]
    fn resolve_wf_window_offsets_skips_a_regime_too_short_to_fit_a_test_slice() {
        use quant_diagnostics::VolatilityRegime::{High, Low};
        // A single-bar High blip can never host a 40-bar test slice -- must
        // not panic or produce a malformed offset, just quietly omit it.
        let mut regimes = make_regimes(&[(500, Low)]);
        regimes[250] = Some(High);
        let n = regimes.len();
        let offsets = resolve_wf_window_offsets(n, 120, 4, 40, Some(&regimes));
        assert!(!offsets.is_empty());
        for &offset in &offsets {
            assert!(offset + 120 <= n);
        }
    }

    #[test]
    fn resolve_wf_window_offsets_empty_regimes_falls_back_to_uniform() {
        let n = 1000;
        let window_size = 200;
        let num_windows = 4;
        let empty_regimes: Vec<Option<quant_diagnostics::VolatilityRegime>> = vec![None; n];
        let with_none_regimes = resolve_wf_window_offsets(n, window_size, num_windows, 60, Some(&empty_regimes));
        let with_no_regimes = resolve_wf_window_offsets(n, window_size, num_windows, 60, None);
        assert_eq!(with_none_regimes, with_no_regimes);
    }

    #[test]
    fn resolve_wf_window_offsets_zero_window_size_or_num_windows_returns_empty() {
        assert!(resolve_wf_window_offsets(1000, 0, 4, 10, None).is_empty());
        assert!(resolve_wf_window_offsets(1000, 100, 0, 10, None).is_empty());
        assert!(resolve_wf_window_offsets(100, 500, 4, 10, None).is_empty());
    }

    // ---------------------------------------------------------------------
    // window_regime_label / regime_coverage_narrative
    // ---------------------------------------------------------------------

    #[test]
    fn window_regime_label_reads_the_midpoint_bar() {
        use quant_diagnostics::VolatilityRegime;
        let regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        assert_eq!(window_regime_label(&regimes, 0, 2), Some(VolatilityRegime::Medium));
    }

    #[test]
    fn window_regime_label_out_of_range_returns_none() {
        use quant_diagnostics::VolatilityRegime;
        let regimes = vec![Some(VolatilityRegime::Low)];
        assert_eq!(window_regime_label(&regimes, 100, 200), None);
    }

    #[test]
    fn regime_coverage_narrative_is_empty_when_no_regime_classified() {
        assert_eq!(regime_coverage_narrative(&[], &[]), "");
        assert_eq!(regime_coverage_narrative(&[], &[None, None]), "");
    }

    #[test]
    fn regime_coverage_narrative_reports_full_coverage() {
        use quant_diagnostics::VolatilityRegime;
        let tested_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        let all_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        let narrative = regime_coverage_narrative(&tested_regimes, &all_regimes);
        assert!(narrative.contains("all 3 detected regime(s) represented"), "{narrative}");
        assert!(narrative.contains("Low (1)"));
        assert!(narrative.contains("Medium (1)"));
        assert!(narrative.contains("High (1)"));
    }

    #[test]
    fn regime_coverage_narrative_flags_partial_coverage() {
        use quant_diagnostics::VolatilityRegime;
        // All 6 test windows landed in Low, but the full dataset also had
        // genuine Medium/High segments -- this is exactly the silent-gap
        // failure mode this narrative exists to surface.
        let tested_regimes: Vec<Option<VolatilityRegime>> = vec![Some(VolatilityRegime::Low); 6];
        let all_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        let narrative = regime_coverage_narrative(&tested_regimes, &all_regimes);
        assert!(narrative.contains("Low (6) only"), "{narrative}");
        assert!(narrative.contains("Medium"));
        assert!(narrative.contains("High"));
        assert!(narrative.contains("never tested out-of-sample"));
    }

    #[test]
    fn regime_coverage_incomplete_is_false_when_no_regime_classified() {
        assert!(!regime_coverage_incomplete(&[], &[]));
        assert!(!regime_coverage_incomplete(&[], &[None, None]));
    }

    #[test]
    fn regime_coverage_incomplete_is_false_for_full_coverage() {
        use quant_diagnostics::VolatilityRegime;
        let tested_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        let all_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        assert!(!regime_coverage_incomplete(&tested_regimes, &all_regimes));
    }

    #[test]
    fn regime_coverage_incomplete_is_true_for_partial_coverage() {
        use quant_diagnostics::VolatilityRegime;
        let tested_regimes: Vec<Option<VolatilityRegime>> = vec![Some(VolatilityRegime::Low); 6];
        let all_regimes = vec![Some(VolatilityRegime::Low), Some(VolatilityRegime::Medium), Some(VolatilityRegime::High)];
        assert!(regime_coverage_incomplete(&tested_regimes, &all_regimes));
    }

    // ---------------------------------------------------------------------
    // Cost hurdle pre-flight check
    // ---------------------------------------------------------------------

    #[test]
    fn required_cost_multiple_scales_with_bar_frequency() {
        assert_eq!(required_cost_multiple(1.0), 4.0);
        assert_eq!(required_cost_multiple(15.0), 4.0);
        assert_eq!(required_cost_multiple(16.0), 3.0);
        assert_eq!(required_cost_multiple(240.0), 3.0);
        assert_eq!(required_cost_multiple(241.0), 2.5);
        assert_eq!(required_cost_multiple(1440.0), 2.5);
    }

    #[test]
    fn compute_cost_hurdle_fails_when_move_is_smaller_than_cost() {
        // Prices barely move at all (1bp average absolute return) against a
        // realistic retail cost structure — should fail clearly.
        let prices: Vec<f64> = (0..100).map(|i| 100.0 * (1.0 + 0.0001 * (i % 2) as f64)).collect();
        let result = compute_cost_hurdle(&prices, 1440.0, 0.001, 5.0).expect("enough data for a result");
        assert!(!result.passed, "tiny moves should never clear a realistic cost hurdle");
        assert!(result.margin < result.required_multiple);
    }

    #[test]
    fn compute_cost_hurdle_passes_when_move_comfortably_clears_cost() {
        // Large, clean 5% alternating moves against low crypto-tier costs.
        let prices: Vec<f64> = (0..100).map(|i| 100.0 * (1.0 + 0.05 * if i % 2 == 0 { 1.0 } else { -1.0 })).collect();
        let result = compute_cost_hurdle(&prices, 1440.0, 0.001, 5.0).expect("enough data for a result");
        assert!(result.passed, "large moves should comfortably clear cost");
        assert!(result.margin > result.required_multiple);
    }

    #[test]
    fn compute_cost_hurdle_requires_a_higher_margin_at_higher_frequency() {
        // Same price series, same fees — only the bar frequency differs.
        // The same margin that passes at daily bars may fail at 5-minute bars
        // because the required multiple scales up.
        let prices: Vec<f64> = (0..100).map(|i| 100.0 * (1.0 + 0.001 * if i % 2 == 0 { 1.0 } else { -1.0 })).collect();
        let daily = compute_cost_hurdle(&prices, 1440.0, 0.001, 5.0).unwrap();
        let five_min = compute_cost_hurdle(&prices, 5.0, 0.001, 5.0).unwrap();
        // Same margin either way (same prices/fees) but a higher bar for 5-minute data.
        assert!((daily.margin - five_min.margin).abs() < 1e-9);
        assert!(five_min.required_multiple > daily.required_multiple);
    }

    #[test]
    fn compute_cost_hurdle_returns_none_for_insufficient_data() {
        assert!(compute_cost_hurdle(&[], 1440.0, 0.001, 5.0).is_none());
        assert!(compute_cost_hurdle(&[100.0], 1440.0, 0.001, 5.0).is_none());
    }

    #[test]
    fn compute_cost_hurdle_zero_cost_always_passes() {
        // Degenerate but must not panic (division by zero guarded) — zero
        // fees/slippage means any nonzero move clears an infinite margin.
        let prices = vec![100.0, 100.001, 100.0, 100.001];
        let result = compute_cost_hurdle(&prices, 1440.0, 0.0, 0.0).expect("enough data");
        assert!(result.passed);
        assert!(result.margin.is_infinite());
    }

    #[test]
    fn cross_sectional_check_disabled_by_default() {
        std::env::remove_var("ENABLE_CROSS_SECTIONAL_CHECK");
        assert!(!cross_sectional_check_enabled());
    }

    #[test]
    fn cross_sectional_check_enabled_via_env_var() {
        std::env::set_var("ENABLE_CROSS_SECTIONAL_CHECK", "true");
        assert!(cross_sectional_check_enabled());
        std::env::set_var("ENABLE_CROSS_SECTIONAL_CHECK", "1");
        assert!(cross_sectional_check_enabled());
        std::env::remove_var("ENABLE_CROSS_SECTIONAL_CHECK");
    }

    #[test]
    fn cross_sectional_basket_classifies_crypto() {
        let basket = cross_sectional_basket_for("BTC-USD");
        assert_eq!(basket, CROSS_SECTIONAL_CRYPTO_BASKET);
        let basket = cross_sectional_basket_for("eth/usd");
        assert_eq!(basket, CROSS_SECTIONAL_CRYPTO_BASKET);
    }

    #[test]
    fn cross_sectional_basket_classifies_forex() {
        let basket = cross_sectional_basket_for("EUR-USD");
        assert_eq!(basket, CROSS_SECTIONAL_FOREX_BASKET);
        let basket = cross_sectional_basket_for("usd/jpy");
        assert_eq!(basket, CROSS_SECTIONAL_FOREX_BASKET);
    }

    #[test]
    fn cross_sectional_basket_defaults_to_equity() {
        let basket = cross_sectional_basket_for("AAPL");
        assert_eq!(basket, CROSS_SECTIONAL_EQUITY_BASKET);
    }

    #[test]
    fn test_validation_result_serde() {
        let result = ValidationResult {
            stages_completed: 2,
            quick_backtest: None,
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: Some(0.15),
            cpcv_deflated_sharpe: Some(1.2),
            cpcv_parameter_stability: Some(0.85),
            verdict: "pass".into(),
            summary: "Strategy passed all stages".into(),
            stage_verdicts: vec![],
            look_ahead_detected: None,
            elapsed_seconds: 42.5,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.stages_completed, 2);
        assert_eq!(deserialized.verdict, "pass");
        assert!((deserialized.elapsed_seconds - 42.5).abs() < f64::EPSILON);
        assert_eq!(deserialized.cpcv_pbo, Some(0.15));
    }

    #[test]
    fn test_stage_verdict_serde() {
        let verdict = StageVerdict {
            stage: 1,
            name: "Quick Backtest".into(),
            passed: true,
            message: "PnL=$1000.00, 50 trades".into(),
        };
        let json = serde_json::to_string(&verdict).unwrap();
        let deserialized: StageVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.stage, 1);
        assert!(deserialized.passed);
        assert_eq!(deserialized.name, "Quick Backtest");
    }

    #[test]
    fn test_walk_forward_result_serde() {
        let result = WalkForwardResult {
            windows: vec![],
            avg_oos_sharpe: 1.5,
            avg_oos_pnl: 5000.0,
            avg_train_sharpe: 2.0,
            overfitting_ratio: 0.2,
            consistency_score: 0.85,
            oos_profitability_rate: 0.75,
            num_windows: 8,
            oos_combined_result: None,
            ga_trial_count: Some(400),
            regime_coverage_narrative: String::new(),
            regime_coverage_incomplete: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: WalkForwardResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.num_windows, 8);
        assert!((deserialized.avg_oos_sharpe - 1.5).abs() < f64::EPSILON);
        assert!((deserialized.consistency_score - 0.85).abs() < f64::EPSILON);
        assert_eq!(deserialized.ga_trial_count, Some(400));
    }

    #[test]
    fn test_walk_forward_window_serde() {
        let window = WalkForwardWindow {
            train_start_idx: 0,
            train_end_idx: 1000,
            test_start_idx: 1000,
            test_end_idx: 1200,
            train_start_date: None,
            train_end_date: None,
            test_start_date: None,
            test_end_date: None,
            train_pnl: 500.0,
            train_sharpe: 1.8,
            train_trades: 50,
            test_pnl: 200.0,
            test_sharpe: 1.2,
            test_trades: 20,
            test_win_rate: 0.6,
            test_max_drawdown: 0.05,
            test_profit_factor: 1.5,
            test_regime: None,
        };
        let json = serde_json::to_string(&window).unwrap();
        let deserialized: WalkForwardWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.train_start_idx, 0);
        assert_eq!(deserialized.test_end_idx, 1200);
        assert_eq!(deserialized.test_trades, 20);
    }

    #[test]
    fn test_validation_result_all_none() {
        let result = ValidationResult {
            stages_completed: 0,
            quick_backtest: None,
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict: "fail".into(),
            summary: String::new(),
            stage_verdicts: vec![],
            look_ahead_detected: None,
            elapsed_seconds: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"stages_completed\":0"));
    }

    #[test]
    fn test_validation_result_clone() {
        let result = ValidationResult {
            stages_completed: 3,
            quick_backtest: None,
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict: "warn".into(),
            summary: "Monte Carlo shows high risk".into(),
            stage_verdicts: vec![StageVerdict {
                stage: 3,
                name: "Monte Carlo".into(),
                passed: false,
                message: "P(ruin) > 20%".into(),
            }],
            look_ahead_detected: None,
            elapsed_seconds: 100.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.verdict, "warn");
        assert_eq!(cloned.stage_verdicts.len(), 1);
        assert!(!cloned.stage_verdicts[0].passed);
    }

    #[test]
    fn test_validation_result_debug() {
        let result = ValidationResult {
            stages_completed: 1,
            quick_backtest: None,
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict: "pass".into(),
            summary: "OK".into(),
            stage_verdicts: vec![],
            look_ahead_detected: None,
            elapsed_seconds: 1.0,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("stages_completed"));
    }

    #[test]
    fn test_walk_forward_result_empty_windows() {
        let result = WalkForwardResult {
            windows: vec![],
            avg_oos_sharpe: 0.0,
            avg_oos_pnl: 0.0,
            avg_train_sharpe: 0.0,
            overfitting_ratio: 0.0,
            consistency_score: 0.0,
            oos_profitability_rate: 0.0,
            num_windows: 0,
            oos_combined_result: None,
            ga_trial_count: None,
            regime_coverage_narrative: String::new(),
            regime_coverage_incomplete: false,
        };
        assert_eq!(result.windows.len(), 0);
        assert_eq!(result.num_windows, 0);
    }

    #[test]
    fn test_stage_verdict_clone_and_debug() {
        let verdict = StageVerdict {
            stage: 2,
            name: "Walk-Forward".into(),
            passed: true,
            message: "OOS Sharpe=1.50".into(),
        };
        let cloned = verdict.clone();
        assert_eq!(cloned.stage, 2);
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("Walk-Forward"));
    }

    #[test]
    fn test_extract_parameter_schema_no_python() {
        // Without Python feature, should return None
        #[cfg(not(feature = "python"))]
        {
            let schema = extract_parameter_schema("class MyStrategy: pass");
            assert!(schema.is_none());
        }
    }

    #[test]
    fn test_estimate_ruin_probability_empty_fan_chart() {
        use metrics::PerformanceMetrics;
        let mc = MonteCarloResult {
            num_runs: 100,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: HashMap::new(),
            var_95: 0.0,
            cvar_95: 0.0,
            data_points: 0,
            var_unreliable: true,
            fan_chart: vec![],
            base_source: String::new(),
            mc_rng_seed: 0,
            bb_p5_total_pnl: None,
        };
        let prob = estimate_ruin_probability(&mc);
        assert_eq!(prob, 0.0);
    }

    #[test]
    fn test_estimate_ruin_probability_no_ruin() {
        use metrics::PerformanceMetrics;
        // Fan chart where p5 never drops below 50% of initial capital
        let mc = MonteCarloResult {
            num_runs: 100,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: HashMap::new(),
            var_95: 0.0,
            cvar_95: 0.0,
            data_points: 100,
            var_unreliable: false,
            fan_chart: vec![
                [9000.0, 9500.0, 10000.0, 10500.0, 11000.0], // step 0
                [9200.0, 9700.0, 10200.0, 10700.0, 11200.0], // step 1
                [9500.0, 10000.0, 10500.0, 11000.0, 11500.0], // step 2
            ],
            base_source: String::new(),
            mc_rng_seed: 0,
            bb_p5_total_pnl: None,
        };
        let prob = estimate_ruin_probability(&mc);
        assert_eq!(prob, 0.0);
    }

    #[test]
    fn test_estimate_ruin_probability_with_ruin() {
        use metrics::PerformanceMetrics;
        // Fan chart starting at 10000, but p5 drops below 5000 (ruin threshold)
        let mc = MonteCarloResult {
            num_runs: 100,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: HashMap::new(),
            var_95: 0.0,
            cvar_95: 0.0,
            data_points: 100,
            var_unreliable: false,
            fan_chart: vec![
                [10000.0, 10000.0, 10000.0, 10000.0, 10000.0], // step 0
                [4000.0, 7000.0, 10000.0, 12000.0, 14000.0],   // step 1: p5 < 5000
                [3000.0, 6000.0, 10000.0, 13000.0, 16000.0],   // step 2: p5 < 5000
            ],
            base_source: String::new(),
            mc_rng_seed: 0,
            bb_p5_total_pnl: None,
        };
        let prob = estimate_ruin_probability(&mc);
        assert!(prob > 0.0, "Should detect ruin probability when p5 drops below threshold");
    }

    #[test]
    fn test_validation_config_custom_fields() {
        let config = ValidationConfig {
            python_source: "class MyStrategy: pass".into(),
            backtest_config: config::BacktestConfig::default(),
            fee_config: None,
            supplementary_data: HashMap::new(),
            parameters: HashMap::new(),
            wf_windows: 5,
            mc_runs: 500,
            min_oos_sharpe: 1.0,
            max_ruin_probability: 0.10,
            progress_callback: None,
            parameter_schema: None,
            ga_config: None,
            fitness_scorer: None,
            fitness_normalizer: None,
            fitness_scorer_config_json: serde_json::Value::Null,
            mc_min_trades: 50,
            mc_confidence_level: 0.99,
            max_drawdown_hard_cap: None,
            historical_iv_surfaces: None,
            option_instrument: None,
            option_spread: None,
            underlying_series: None,
        };
        assert_eq!(config.wf_windows, 5);
        assert_eq!(config.mc_runs, 500);
        assert!((config.min_oos_sharpe - 1.0).abs() < f64::EPSILON);
        assert!((config.max_ruin_probability - 0.10).abs() < f64::EPSILON);
        assert_eq!(config.mc_min_trades, 50);
        assert!((config.mc_confidence_level - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation_result_with_stage_verdicts() {
        let verdicts = vec![
            StageVerdict { stage: 1, name: "Quick Backtest".into(), passed: true, message: "PnL > 0".into() },
            StageVerdict { stage: 2, name: "Walk-Forward".into(), passed: true, message: "OOS Sharpe=1.5".into() },
            StageVerdict { stage: 3, name: "Monte Carlo".into(), passed: false, message: "P(ruin)=25%".into() },
        ];
        let result = ValidationResult {
            stages_completed: 3,
            quick_backtest: None,
            walk_forward: None,
            monte_carlo: None,
            block_bootstrap: None,
            regime_mc: None,
            ga_report: None,
            optimized_backtest: None,
            cpcv_pbo: None,
            cpcv_deflated_sharpe: None,
            cpcv_parameter_stability: None,
            verdict: "warn".into(),
            summary: "High ruin probability".into(),
            stage_verdicts: verdicts,
            look_ahead_detected: None,
            elapsed_seconds: 120.0,
        };
        assert_eq!(result.stage_verdicts.len(), 3);
        assert!(result.stage_verdicts[0].passed);
        assert!(!result.stage_verdicts[2].passed);
        let json = serde_json::to_string(&result).unwrap();
        let deser: ValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.stage_verdicts.len(), 3);
    }

    // ================================================================
    // Bug #34 — combine_oos_results must reconcile total_pnl with the
    // chained (multiplicative) equity curve, not the additive per-window
    // dollar sum, so the headline OOS PnL agrees with the OOS equity curve
    // endpoint, max drawdown, and Monte Carlo.
    // ================================================================

    /// Build a one-window OOS BacktestResult from an equity curve and matching
    /// per-trade fractional/dollar returns. Each window starts from its OWN
    /// `initial_capital` base (as walk-forward windows do).
    fn oos_window(equity_curve: Vec<f64>, pct_returns: Vec<f64>) -> BacktestResult {
        let initial_capital = equity_curve.first().copied().unwrap_or(10_000.0);
        let final_eq = equity_curve.last().copied().unwrap_or(initial_capital);
        // Per-window total_pnl is the window's own additive dollar result.
        let total_pnl = final_eq - initial_capital;
        let dollar_returns: Vec<f64> =
            pct_returns.iter().map(|p| p * initial_capital).collect();
        BacktestResult {
            total_pnl,
            num_trades: pct_returns.len(),
            closed_trades: pct_returns.len(),
            equity_curve,
            trade_returns: dollar_returns,
            trade_pct_returns: pct_returns,
            initial_capital,
            ..Default::default()
        }
    }

    #[test]
    fn test_combine_oos_total_pnl_matches_chained_equity_endpoint() {
        let initial_capital = 10_000.0;
        // Window 1: 10000 -> 12000 (+20%). Window 2: 10000 -> 9000 (-10%).
        // Additive per-window dollar sum = +2000 + (-1000) = +1000.
        // Chained (compounded): 10000 * 1.2 * 0.9 = 10800 -> +800.
        // These DISAGREE — the fix must report the chained +800.
        let w1 = oos_window(vec![10_000.0, 11_000.0, 12_000.0], vec![0.20]);
        let w2 = oos_window(vec![10_000.0, 9_500.0, 9_000.0], vec![-0.10]);

        let combined = combine_oos_results(&[w1, w2], initial_capital);

        let chained_end = combined.equity_curve.last().copied().unwrap();
        // Headline PnL must equal the chained equity endpoint minus capital.
        assert!(
            (combined.total_pnl - (chained_end - initial_capital)).abs() < 1e-6,
            "total_pnl {} should match chained endpoint pnl {}",
            combined.total_pnl,
            chained_end - initial_capital
        );
        // Specifically the compounded result, NOT the additive +1000 sum.
        assert!((chained_end - 10_800.0).abs() < 1e-6, "chained end was {}", chained_end);
        assert!((combined.total_pnl - 800.0).abs() < 1e-6, "total_pnl was {}", combined.total_pnl);
        // net_profit (percent of capital) must agree with total_pnl.
        let net_pct = combined.net_profit.unwrap();
        assert!((net_pct - (combined.total_pnl / initial_capital * 100.0)).abs() < 1e-6);
        // realized_pnl must agree with total_pnl too.
        assert!((combined.realized_pnl.unwrap() - combined.total_pnl).abs() < 1e-6);
    }

    #[test]
    fn test_combine_oos_falls_back_to_additive_without_equity_curve() {
        let initial_capital = 10_000.0;
        // Windows with NO equity curve → cannot chain → additive fallback.
        let mut w1 = oos_window(vec![], vec![0.05]);
        w1.total_pnl = 500.0;
        let mut w2 = oos_window(vec![], vec![-0.02]);
        w2.total_pnl = -200.0;

        let combined = combine_oos_results(&[w1, w2], initial_capital);
        assert!(combined.equity_curve.is_empty());
        // Falls back to additive sum 500 + (-200) = 300.
        assert!((combined.total_pnl - 300.0).abs() < 1e-6, "total_pnl was {}", combined.total_pnl);
    }
}
