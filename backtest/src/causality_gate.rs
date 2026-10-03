//! W2.1 / gate G1: causality-by-truncation on agent-authored Python strategies.
//!
//! The dominant defect of the 2026-09-21 documented-strategy benchmark (7 of the 9 failures) was "end-of-array":
//! the strategy's `compute_signals()` treats the LAST AVAILABLE bar as a decision point, or anchors a window to the
//! end of the array, so the signal it emits at bar `t` depends on whether bars after `t` exist. A single full-series
//! run can never expose that; only re-running the strategy on a series that genuinely ENDS at `t` can.
//!
//! The test is the one `weightsim::harness::check_rule_truncation` applies to compiled `WeightRule`s, ported to the
//! Python bulk-signal path: for each sampled bar `t`, `compute_signals(series[..=t])[t]` must equal
//! `compute_signals(series)[t]`. Signals are discrete (`i8`: 1 / -1 / 0 / 2), so the comparison is exact; the float
//! comparison helper ([`signals_differ`]) exists for float-valued outputs (tolerance [`FLOAT_TOLERANCE`]).
//!
//! Sampling ([`sample_truncation_bars`]): `K` bars per asset (default [`DEFAULT_SAMPLE_BARS_PER_ASSET`]), drawn
//! deterministically from a fixed seed, [`TAIL_SHARE`] of them from the last [`TAIL_FRACTION`] of the series (where
//! end-of-array defects bite hardest), and ALWAYS including the final bar `n-1` and the bar before it `n-2`. Note that
//! truncating at the final bar is the full series itself, so `n-1` can only ever expose non-determinism; `n-2` is the
//! last bar at which "one later bar exists" and is the sharpest single probe for the defect class.
//!
//! The pure parts of this module (sampling, comparison, the typed report) compile and are unit-tested without the
//! `python` feature; [`causality_truncation_check`] needs it.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Bars sampled per asset in `Sampled` mode (SCORECARD Amendment 9's number).
pub const DEFAULT_SAMPLE_BARS_PER_ASSET: usize = 60;
/// Fixed seed of the deterministic sampler. Changing it changes which bars every strategy is checked on.
pub const DEFAULT_SEED: u64 = 0x5EED_2026_1002_0001;
/// The last `TAIL_FRACTION` of the series is the "tail" the sampler is biased toward.
pub const TAIL_FRACTION: f64 = 0.20;
/// Share of the sample drawn from the tail (the rest is spread over the earlier bars).
pub const TAIL_SHARE: f64 = 0.70;
/// Tolerance for float-valued outputs. Discrete `i8` signals are compared exactly (tolerance 0).
pub const FLOAT_TOLERANCE: f64 = 1e-12;
/// Default wall-clock budget of one gate run (per asset). The final bars are checked first, so when the budget runs
/// out the sharpest probes have already been made; the report says how many bars were actually checked.
pub const DEFAULT_MAX_SECONDS: f64 = 600.0;

/// Which scan Stage 0 of the validation pipeline runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookaheadScanMode {
    /// The G1 gate on `K` sampled bars per asset (default). Bounded work; this is what makes the stage usable on
    /// multi-asset submissions (the full scan measured 40+ minutes there).
    Sampled,
    /// The G1 gate on EVERY bar from the strategy's declared history window to the end, plus the legacy
    /// 60 %-vs-full trade-count perturbation test. Slow by design; available for forensics.
    Full,
}

impl LookaheadScanMode {
    /// Environment variable that selects the mode (`sampled` | `full`, case-insensitive). Unset or unknown = `Sampled`.
    /// An environment knob rather than a `ValidationConfig` field because the Engine constructs `ValidationConfig`
    /// exhaustively at four sites; a new required field would break its build mid-slice.
    pub const ENV: &'static str = "PYTHON_LOOKAHEAD_SCAN_MODE";

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sampled" | "sample" => Some(LookaheadScanMode::Sampled),
            "full" | "exhaustive" => Some(LookaheadScanMode::Full),
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        std::env::var(Self::ENV)
            .ok()
            .and_then(|v| Self::parse(&v))
            .unwrap_or(LookaheadScanMode::Sampled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LookaheadScanMode::Sampled => "sampled",
            LookaheadScanMode::Full => "full",
        }
    }
}

/// Configuration of one gate run.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalityGateConfig {
    pub mode: LookaheadScanMode,
    /// `K` in `Sampled` mode; ignored in `Full` mode (every bar is checked).
    pub sample_bars_per_asset: usize,
    pub seed: u64,
    /// Wall-clock budget per asset in seconds (see [`DEFAULT_MAX_SECONDS`]).
    pub max_seconds: f64,
}

impl Default for CausalityGateConfig {
    fn default() -> Self {
        CausalityGateConfig {
            mode: LookaheadScanMode::Sampled,
            sample_bars_per_asset: DEFAULT_SAMPLE_BARS_PER_ASSET,
            seed: DEFAULT_SEED,
            max_seconds: DEFAULT_MAX_SECONDS,
        }
    }
}

impl CausalityGateConfig {
    /// Environment variable overriding `sample_bars_per_asset` (a positive integer).
    pub const ENV_SAMPLE_BARS: &'static str = "PYTHON_LOOKAHEAD_SAMPLE_BARS";
    /// Environment variable overriding `max_seconds` (a positive number of seconds).
    pub const ENV_MAX_SECONDS: &'static str = "PYTHON_LOOKAHEAD_MAX_SECONDS";

    /// The defaults, with [`LookaheadScanMode::ENV`], [`Self::ENV_SAMPLE_BARS`] and [`Self::ENV_MAX_SECONDS`]
    /// applied when set and valid. The seed is never read from the environment: it is part of the gate's definition.
    pub fn from_env() -> Self {
        let mut c = CausalityGateConfig {
            mode: LookaheadScanMode::from_env(),
            ..CausalityGateConfig::default()
        };
        if let Some(k) = std::env::var(Self::ENV_SAMPLE_BARS)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            if k > 0 {
                c.sample_bars_per_asset = k;
            }
        }
        if let Some(s) = std::env::var(Self::ENV_MAX_SECONDS)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
        {
            if s.is_finite() && s > 0.0 {
                c.max_seconds = s;
            }
        }
        c
    }
}

/// One bar at which the strategy's output depends on bars after it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalityViolation {
    /// Symbol of the series (the primary venue's symbol for a multi-venue strategy).
    pub asset: String,
    pub bar_index: usize,
    /// Signal at `bar_index` when the whole series was visible.
    pub full_signal: f64,
    /// Signal at `bar_index` when the series ended at `bar_index`. `NaN` when the truncated run failed (a Python
    /// exception or a wrong-length result): the strategy could not have been run live at that bar with the history
    /// it would have had, which is the same defect class.
    pub truncated_signal: f64,
    /// Why the truncated run produced no comparable signal, when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The typed result of one gate run on one series. `passed()` is the blocking verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalityTruncationReport {
    pub asset: String,
    pub mode: LookaheadScanMode,
    pub n_bars: usize,
    /// The bars the gate planned to check (ascending). Always contains `n_bars - 1` and, when it exists, `n_bars - 2`.
    pub sampled_bars: Vec<usize>,
    /// How many of `sampled_bars` were actually checked (fewer than planned only when the budget ran out).
    pub bars_checked: usize,
    pub violations: Vec<CausalityViolation>,
    /// `false` when the strategy has no bulk `compute_signals()` path (it defines `generate_signals()` instead, which
    /// the tick loop feeds one bar at a time and which therefore cannot see later bars). The gate then neither
    /// passes nor fails on evidence; `passed()` is `true` and `message` says why.
    pub applicable: bool,
    /// Two runs on the identical full series disagreed. Such a strategy cannot be certified causal (or anything
    /// else): `passed()` is `false`.
    pub nondeterministic: bool,
    pub budget_exhausted: bool,
    pub elapsed_seconds: f64,
    pub seed: u64,
    pub message: String,
}

impl CausalityTruncationReport {
    /// The blocking verdict: no violation at any checked bar and a deterministic strategy.
    pub fn passed(&self) -> bool {
        !self.nondeterministic && self.violations.is_empty()
    }

    fn not_applicable(
        asset: &str,
        n_bars: usize,
        cfg: &CausalityGateConfig,
        why: &str,
        elapsed: f64,
    ) -> Self {
        CausalityTruncationReport {
            asset: asset.to_string(),
            mode: cfg.mode,
            n_bars,
            sampled_bars: Vec::new(),
            bars_checked: 0,
            violations: Vec::new(),
            applicable: false,
            nondeterministic: false,
            budget_exhausted: false,
            elapsed_seconds: elapsed,
            seed: cfg.seed,
            message: format!("not applicable: {why}"),
        }
    }
}

/// splitmix64 (the same generator `weightsim::harness` uses): deterministic, dependency-free.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw `want` distinct bars from `lo..hi` (half-open) into `out`, deterministically from `stream`, skipping bars
/// already in `out`. Draws fewer when the range is too small.
fn draw_distinct(out: &mut Vec<usize>, lo: usize, hi: usize, want: usize, stream: &mut u64) {
    if hi <= lo {
        return;
    }
    let span = hi - lo;
    let available = span.saturating_sub(out.iter().filter(|&&b| b >= lo && b < hi).count());
    let want = want.min(available);
    let mut drawn = 0usize;
    let mut guard = 0usize;
    while drawn < want {
        *stream = splitmix64(*stream);
        let b = lo + (*stream % span as u64) as usize;
        if !out.contains(&b) {
            out.push(b);
            drawn += 1;
        }
        guard += 1;
        if guard > 64 * span + 64 {
            // Degenerate (should be unreachable since `want <= available`); fall back to a sweep so the function
            // always terminates with the requested count.
            for b in lo..hi {
                if drawn >= want {
                    break;
                }
                if !out.contains(&b) {
                    out.push(b);
                    drawn += 1;
                }
            }
            break;
        }
    }
}

/// The bars one gate run checks on a series of `n_bars` bars: `k` distinct bars in `[first_bar, n_bars)`, ascending,
/// always including `n_bars - 1` and (when `>= first_bar`) `n_bars - 2`, with [`TAIL_SHARE`] of the remainder drawn
/// from the last [`TAIL_FRACTION`] of the series and the rest from the bars before it. Deterministic in `seed`.
/// `k >= n_bars - first_bar` returns every bar. Empty when `n_bars == 0` or `first_bar >= n_bars`.
pub fn sample_truncation_bars(n_bars: usize, k: usize, first_bar: usize, seed: u64) -> Vec<usize> {
    if n_bars == 0 || first_bar >= n_bars || k == 0 {
        return Vec::new();
    }
    let candidates = n_bars - first_bar;
    if k >= candidates {
        return (first_bar..n_bars).collect();
    }
    let mut out: Vec<usize> = vec![n_bars - 1];
    if n_bars >= 2 && n_bars - 2 >= first_bar && out.len() < k {
        out.push(n_bars - 2);
    }
    let remaining = k - out.len();
    // Tail = last ceil(TAIL_FRACTION * n) bars, never starting before first_bar.
    let tail_len = ((n_bars as f64) * TAIL_FRACTION).ceil() as usize;
    let tail_start = n_bars.saturating_sub(tail_len).max(first_bar);
    let want_tail = ((remaining as f64) * TAIL_SHARE).round() as usize;
    let want_head = remaining - want_tail;
    let mut stream = seed ^ splitmix64((n_bars as u64) << 20 ^ k as u64 ^ (first_bar as u64) << 40);
    draw_distinct(&mut out, tail_start, n_bars, want_tail, &mut stream);
    draw_distinct(&mut out, first_bar, tail_start, want_head, &mut stream);
    // If one side had too few bars, top up from the other so the count is exactly k when possible.
    if out.len() < k {
        let short = k - out.len();
        draw_distinct(&mut out, first_bar, n_bars, short, &mut stream);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// `true` when two outputs differ beyond `tol`. NaN-aware: `NaN` vs `NaN` is "same", `NaN` vs a number differs.
/// Use `tol = 0.0` for discrete signals and [`FLOAT_TOLERANCE`] for float-valued outputs.
pub fn signals_differ(a: f64, b: f64, tol: f64) -> bool {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => false,
        (true, false) | (false, true) => true,
        (false, false) => (a - b).abs() > tol,
    }
}

/// Compare the signal at `t` between a full run and a run on the series truncated at `t`. `None` when they agree.
/// A truncated run that produced no signal, or one of the wrong length, is a violation (see
/// [`CausalityViolation::truncated_signal`]).
pub fn compare_at(
    asset: &str,
    t: usize,
    full: &[i8],
    truncated: Option<&[i8]>,
    truncated_error: Option<&str>,
) -> Option<CausalityViolation> {
    let full_signal = full.get(t).map_or(f64::NAN, |&s| f64::from(s));
    match truncated {
        Some(tr) if tr.len() == t + 1 => {
            let ts = f64::from(tr[t]);
            if signals_differ(full_signal, ts, 0.0) {
                Some(CausalityViolation {
                    asset: asset.to_string(),
                    bar_index: t,
                    full_signal,
                    truncated_signal: ts,
                    detail: None,
                })
            } else {
                None
            }
        }
        Some(tr) => Some(CausalityViolation {
            asset: asset.to_string(),
            bar_index: t,
            full_signal,
            truncated_signal: f64::NAN,
            detail: Some(format!(
                "truncated run returned {} signals for {} bars",
                tr.len(),
                t + 1
            )),
        }),
        None => Some(CausalityViolation {
            asset: asset.to_string(),
            bar_index: t,
            full_signal,
            truncated_signal: f64::NAN,
            detail: Some(match truncated_error {
                Some(e) => format!("truncated run failed: {e}"),
                None => "truncated run returned no signals".to_string(),
            }),
        }),
    }
}

/// Human-readable one-liner for a stage verdict / summary.
pub fn describe(report: &CausalityTruncationReport) -> String {
    if !report.applicable {
        return report.message.clone();
    }
    if report.nondeterministic {
        return format!(
            "non-deterministic: two runs on the identical {}-bar series disagreed; seed every stochastic step",
            report.n_bars
        );
    }
    if let Some(v) = report.violations.first() {
        let kind = match &v.detail {
            Some(d) => d.clone(),
            None => format!(
                "signal {} with later bars present vs {} without",
                v.full_signal, v.truncated_signal
            ),
        };
        return format!(
            "look-ahead (end-of-array) at {} of {} checked bars; first at bar {} of {}: {} -- the signal at a bar must not depend on whether later bars exist",
            report.violations.len(),
            report.bars_checked,
            v.bar_index,
            report.n_bars,
            kind
        );
    }
    format!(
        "signal unchanged when later bars are removed at every one of {} checked bars ({} mode, {:.1}s{})",
        report.bars_checked,
        report.mode.as_str(),
        report.elapsed_seconds,
        if report.budget_exhausted { ", budget exhausted" } else { "" }
    )
}

/// Truncate a multi-venue bundle so every venue ends at or before `cutoff_ts` (venues are timestamp-aligned and
/// ascending by contract). Pure; shared by the multi-venue gate.
pub fn truncate_venues_at(
    venues: &HashMap<(String, String), strategy::traits::VenueSeries>,
    cutoff_ts: i64,
) -> HashMap<(String, String), strategy::traits::VenueSeries> {
    venues
        .iter()
        .map(|(k, v)| {
            let len = v.timestamps.partition_point(|&ts| ts <= cutoff_ts);
            let cut = |x: &Vec<f64>| {
                if x.len() >= len {
                    x[..len].to_vec()
                } else {
                    x.clone()
                }
            };
            (
                k.clone(),
                strategy::traits::VenueSeries {
                    prices: cut(&v.prices),
                    volumes: cut(&v.volumes),
                    timestamps: v.timestamps[..len].to_vec(),
                    opens: cut(&v.opens),
                    highs: cut(&v.highs),
                    lows: cut(&v.lows),
                },
            )
        })
        .collect()
}

#[cfg(feature = "python")]
mod run {
    use super::*;
    use crate::logging_facade::BACKTEST_LOGGER;
    use crate::{log_info, log_warn};
    use anyhow::{Context, Result};
    use config::ParameterValue;
    use dataloader::MarketData;
    use strategy::executor::{self, StrategyExecutor};
    use strategy::traits::VenueSeries;

    /// The series a gate run truncates: the flat single-venue arrays, or the multi-venue bundle plus its primary.
    enum Series<'a> {
        Flat {
            prices: &'a [f64],
            volumes: &'a [f64],
            timestamps: &'a [i64],
            opens: &'a [f64],
            highs: &'a [f64],
            lows: &'a [f64],
        },
        MultiVenue {
            venues: &'a HashMap<(String, String), VenueSeries>,
            primary_timestamps: &'a [i64],
        },
    }

    impl<'a> Series<'a> {
        fn n_bars(&self) -> usize {
            match self {
                Series::Flat { prices, .. } => prices.len(),
                Series::MultiVenue {
                    primary_timestamps, ..
                } => primary_timestamps.len(),
            }
        }

        /// `compute_signals` on the first `len` bars.
        async fn signals_for_prefix(
            &self,
            ex: &mut dyn StrategyExecutor,
            len: usize,
        ) -> (Option<Vec<i8>>, Option<String>) {
            match self {
                Series::Flat {
                    prices,
                    volumes,
                    timestamps,
                    opens,
                    highs,
                    lows,
                } => {
                    let cut = |x: &'a [f64]| if x.len() >= len { &x[..len] } else { x };
                    ex.compute_all_signals(
                        &prices[..len],
                        &volumes[..len],
                        &timestamps[..len],
                        cut(opens),
                        cut(highs),
                        cut(lows),
                    )
                    .await
                }
                Series::MultiVenue {
                    venues,
                    primary_timestamps,
                } => {
                    let cutoff = primary_timestamps[len - 1];
                    let truncated = truncate_venues_at(venues, cutoff);
                    ex.compute_all_signals_multi_venue(&truncated).await
                }
            }
        }
    }

    fn symbol_of(md: &MarketData) -> String {
        match md {
            MarketData::Candle(c) => c.symbol.to_string(),
            MarketData::Trade(t) => t.symbol.to_string(),
            MarketData::OptionCandle(c) => c.contract_ticker.to_string(),
            MarketData::PoolSwap(_) | MarketData::Generic(_) => "primary".to_string(),
        }
    }

    /// The G1 gate on one flat series (the validation pipeline's path; the Engine calls the pipeline once per asset,
    /// so "K per asset" is K here). `Err` only when the strategy cannot be built or its FULL run fails -- those errors
    /// belong to Stage 1, which reports them to the user; the caller treats `Err` as "stage skipped", never as a pass
    /// on evidence.
    pub async fn causality_truncation_check(
        market_data: &[MarketData],
        python_source: &str,
        parameters: HashMap<String, ParameterValue>,
        cfg: &CausalityGateConfig,
    ) -> Result<CausalityTruncationReport> {
        if market_data.is_empty() {
            anyhow::bail!("no market data for the causality gate");
        }
        let asset = symbol_of(&market_data[0]);
        let (mut ex, _tier) = executor::build_executor(python_source, parameters)
            .await
            .context("causality gate: failed to build strategy executor")?;
        let wants_ohlc = ex.compute_signals_accepts_ohlc();
        let n = market_data.len();
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        let mut timestamps = Vec::with_capacity(n);
        let (mut opens, mut highs, mut lows) = (Vec::new(), Vec::new(), Vec::new());
        for md in market_data {
            let (o, h, l, c, v, ts) = crate::python_simulation::extract_ohlcv_ts(md);
            prices.push(c);
            volumes.push(v);
            timestamps.push(ts.timestamp_millis());
            if wants_ohlc {
                opens.push(o);
                highs.push(h);
                lows.push(l);
            }
        }
        let series = Series::Flat {
            prices: &prices,
            volumes: &volumes,
            timestamps: &timestamps,
            opens: &opens,
            highs: &highs,
            lows: &lows,
        };
        run_gate(&asset, &mut *ex, &series, python_source, cfg).await
    }

    /// The G1 gate on a multi-venue bundle (the Engine's `execute_multi_venue_python_validation` path). The signal
    /// vector is aligned to `primary`'s series; every venue is truncated by timestamp at each sampled primary bar.
    pub async fn causality_truncation_check_multi_venue(
        venues: &HashMap<(String, String), VenueSeries>,
        primary: &(String, String),
        python_source: &str,
        parameters: HashMap<String, ParameterValue>,
        cfg: &CausalityGateConfig,
    ) -> Result<CausalityTruncationReport> {
        let p = venues
            .get(primary)
            .ok_or_else(|| anyhow::anyhow!("primary venue {primary:?} is not in the bundle"))?;
        if p.timestamps.is_empty() {
            anyhow::bail!("primary venue has no bars");
        }
        let (mut ex, _tier) = executor::build_executor(python_source, parameters)
            .await
            .context("causality gate: failed to build strategy executor")?;
        if !ex.accepts_multi_venue() {
            return Ok(CausalityTruncationReport::not_applicable(
                &primary.0,
                p.timestamps.len(),
                cfg,
                "the strategy does not implement compute_signals_multi_venue()",
                0.0,
            ));
        }
        let series = Series::MultiVenue {
            venues,
            primary_timestamps: &p.timestamps,
        };
        run_gate(&primary.0, &mut *ex, &series, python_source, cfg).await
    }

    async fn run_gate(
        asset: &str,
        ex: &mut dyn StrategyExecutor,
        series: &Series<'_>,
        python_source: &str,
        cfg: &CausalityGateConfig,
    ) -> Result<CausalityTruncationReport> {
        let started = std::time::Instant::now();
        let n = series.n_bars();
        if ex.defines_generate_signals() {
            return Ok(CausalityTruncationReport::not_applicable(
                asset,
                n,
                cfg,
                "the strategy defines generate_signals() (per-tick path, fed one bar at a time)",
                started.elapsed().as_secs_f64(),
            ));
        }
        // The full run: the reference every truncated run is compared with.
        let (full, full_err) = series.signals_for_prefix(ex, n).await;
        let full = match (full, full_err) {
            (Some(s), _) if s.len() == n => s,
            (Some(s), _) => anyhow::bail!(
                "compute_signals() returned {} signals for {} bars",
                s.len(),
                n
            ),
            (None, Some(e)) => anyhow::bail!("compute_signals() failed on the full series: {e}"),
            (None, None) => {
                return Ok(CausalityTruncationReport::not_applicable(
                    asset,
                    n,
                    cfg,
                    "the strategy has no vectorized compute_signals() path",
                    started.elapsed().as_secs_f64(),
                ))
            }
        };
        // Determinism pre-check, only when the source can be stochastic (same rule as the legacy stage): an unseeded
        // RNG would otherwise be blamed on look-ahead.
        let mut nondeterministic = false;
        if crate::python_validation::source_may_be_stochastic(python_source) {
            let (again, _) = series.signals_for_prefix(ex, n).await;
            nondeterministic = again.as_deref() != Some(full.as_slice());
        }
        let first_bar = ex
            .required_data_window()
            .max(1)
            .min(n.saturating_sub(1).max(1));
        let planned = match cfg.mode {
            LookaheadScanMode::Sampled => {
                sample_truncation_bars(n, cfg.sample_bars_per_asset, first_bar, cfg.seed)
            }
            LookaheadScanMode::Full => sample_truncation_bars(n, n, first_bar, cfg.seed),
        };
        let mut violations = Vec::new();
        let mut checked = 0usize;
        let mut budget_exhausted = false;
        if !nondeterministic {
            // Final bars first (the sharpest probes), then the rest ascending.
            let head: Vec<usize> = planned.iter().rev().take(2).copied().collect();
            let mut order = head.clone();
            order.extend(planned.iter().copied().filter(|b| !head.contains(b)));
            for t in order {
                if started.elapsed().as_secs_f64() > cfg.max_seconds {
                    budget_exhausted = true;
                    break;
                }
                let (tr, err) = series.signals_for_prefix(ex, t + 1).await;
                checked += 1;
                if let Some(v) = compare_at(asset, t, &full, tr.as_deref(), err.as_deref()) {
                    violations.push(v);
                }
            }
            violations.sort_by_key(|v| v.bar_index);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let mut report = CausalityTruncationReport {
            asset: asset.to_string(),
            mode: cfg.mode,
            n_bars: n,
            sampled_bars: planned,
            bars_checked: checked,
            violations,
            applicable: true,
            nondeterministic,
            budget_exhausted,
            elapsed_seconds: elapsed,
            seed: cfg.seed,
            message: String::new(),
        };
        report.message = describe(&report);
        if report.passed() {
            log_info!(
                BACKTEST_LOGGER,
                "[CAUSALITY G1] {} PASS: {} bars checked of {} ({} mode) in {:.2}s{}",
                asset,
                report.bars_checked,
                n,
                cfg.mode.as_str(),
                elapsed,
                if budget_exhausted {
                    " (budget exhausted)"
                } else {
                    ""
                }
            );
        } else {
            log_warn!(
                BACKTEST_LOGGER,
                "[CAUSALITY G1] {} FAIL in {:.2}s: {}",
                asset,
                elapsed,
                report.message
            );
        }
        Ok(report)
    }
}

#[cfg(feature = "python")]
pub use run::{causality_truncation_check, causality_truncation_check_multi_venue};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_always_includes_the_final_two_bars_and_is_within_bounds() {
        for (n, k, first) in [
            (1000usize, 60usize, 1usize),
            (300, 60, 100),
            (5000, 60, 1),
            (200, 10, 50),
            (61, 60, 0),
        ] {
            let s = sample_truncation_bars(n, k, first, DEFAULT_SEED);
            assert!(s.contains(&(n - 1)), "n={n} k={k}: {s:?}");
            assert!(s.contains(&(n - 2)), "n={n} k={k}: {s:?}");
            assert!(
                s.iter().all(|&b| b >= first && b < n),
                "n={n} first={first}: {s:?}"
            );
            assert!(s.windows(2).all(|w| w[0] < w[1]), "sorted, distinct: {s:?}");
            assert_eq!(s.len(), k.min(n - first), "n={n} k={k} first={first}");
        }
    }

    #[test]
    fn sample_is_deterministic_in_the_seed_and_changes_with_it() {
        let a = sample_truncation_bars(2500, 60, 1, DEFAULT_SEED);
        let b = sample_truncation_bars(2500, 60, 1, DEFAULT_SEED);
        assert_eq!(a, b);
        let c = sample_truncation_bars(2500, 60, 1, DEFAULT_SEED ^ 1);
        assert_ne!(a, c);
        assert!(c.contains(&2499) && c.contains(&2498));
    }

    #[test]
    fn sample_is_biased_toward_the_last_twenty_percent() {
        let n = 10_000;
        let s = sample_truncation_bars(n, 60, 1, DEFAULT_SEED);
        let tail_start = n - (n as f64 * TAIL_FRACTION).ceil() as usize;
        let in_tail = s.iter().filter(|&&b| b >= tail_start).count();
        // 2 pinned + round(0.7 * 58) = 41 drawn from the tail = 43 of 60.
        assert_eq!(in_tail, 43, "{s:?}");
        assert_eq!(s.len() - in_tail, 17);
    }

    #[test]
    fn sample_returns_every_bar_when_k_covers_the_range_and_nothing_on_degenerate_input() {
        assert_eq!(
            sample_truncation_bars(10, 60, 3, 1),
            (3..10).collect::<Vec<_>>()
        );
        assert_eq!(
            sample_truncation_bars(10, 7, 3, 1),
            (3..10).collect::<Vec<_>>()
        );
        assert!(sample_truncation_bars(0, 60, 0, 1).is_empty());
        assert!(sample_truncation_bars(10, 60, 10, 1).is_empty());
        assert!(sample_truncation_bars(10, 0, 0, 1).is_empty());
        assert_eq!(sample_truncation_bars(1, 60, 0, 1), vec![0]);
    }

    #[test]
    fn full_mode_sample_is_every_bar_from_first_bar() {
        assert_eq!(
            sample_truncation_bars(500, 500, 100, 7),
            (100..500).collect::<Vec<_>>()
        );
    }

    #[test]
    fn signals_differ_is_exact_for_discrete_and_tolerant_for_floats() {
        assert!(!signals_differ(1.0, 1.0, 0.0));
        assert!(signals_differ(1.0, 0.0, 0.0));
        assert!(!signals_differ(0.3, 0.3 + 1e-13, FLOAT_TOLERANCE));
        assert!(signals_differ(0.3, 0.3 + 1e-11, FLOAT_TOLERANCE));
        assert!(!signals_differ(f64::NAN, f64::NAN, 0.0));
        assert!(signals_differ(f64::NAN, 0.0, 0.0));
    }

    #[test]
    fn compare_at_reports_the_typed_violation_shapes() {
        let full = [0i8, 0, 1, -1, 2];
        assert_eq!(compare_at("X", 2, &full, Some(&[0, 0, 1]), None), None);
        let v = compare_at("X", 3, &full, Some(&[0, 0, 1, 1]), None).unwrap();
        assert_eq!(
            (
                v.asset.as_str(),
                v.bar_index,
                v.full_signal,
                v.truncated_signal,
                v.detail
            ),
            ("X", 3, -1.0, 1.0, None)
        );
        let wrong = compare_at("X", 3, &full, Some(&[0, 0]), None).unwrap();
        assert!(
            wrong.truncated_signal.is_nan()
                && wrong.detail.unwrap().contains("2 signals for 4 bars")
        );
        let failed = compare_at("X", 4, &full, None, Some("IndexError")).unwrap();
        assert!(failed.truncated_signal.is_nan() && failed.detail.unwrap().contains("IndexError"));
        let none = compare_at("X", 4, &full, None, None).unwrap();
        assert!(none.detail.unwrap().contains("no signals"));
    }

    #[test]
    fn report_passed_and_describe() {
        let mut r = CausalityTruncationReport {
            asset: "SPY".into(),
            mode: LookaheadScanMode::Sampled,
            n_bars: 1000,
            sampled_bars: vec![999],
            bars_checked: 1,
            violations: vec![],
            applicable: true,
            nondeterministic: false,
            budget_exhausted: false,
            elapsed_seconds: 0.5,
            seed: DEFAULT_SEED,
            message: String::new(),
        };
        assert!(r.passed());
        assert!(describe(&r).contains("unchanged"));
        r.violations.push(CausalityViolation {
            asset: "SPY".into(),
            bar_index: 998,
            full_signal: 0.0,
            truncated_signal: 1.0,
            detail: None,
        });
        assert!(!r.passed());
        assert!(describe(&r).contains("bar 998"));
        r.violations.clear();
        r.nondeterministic = true;
        assert!(!r.passed());
        assert!(describe(&r).contains("non-deterministic"));
        let json = serde_json::to_string(&r).unwrap();
        let back: CausalityTruncationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn scan_mode_parses_and_defaults_to_sampled() {
        assert_eq!(
            LookaheadScanMode::parse("FULL"),
            Some(LookaheadScanMode::Full)
        );
        assert_eq!(
            LookaheadScanMode::parse(" sampled "),
            Some(LookaheadScanMode::Sampled)
        );
        assert_eq!(LookaheadScanMode::parse("nope"), None);
        assert_eq!(
            CausalityGateConfig::default().mode,
            LookaheadScanMode::Sampled
        );
        assert_eq!(CausalityGateConfig::default().sample_bars_per_asset, 60);
    }

    #[test]
    fn truncate_venues_cuts_every_venue_at_the_cutoff_timestamp() {
        use strategy::traits::VenueSeries;
        let mk = |ts: &[i64]| VenueSeries {
            prices: ts.iter().map(|&t| t as f64).collect(),
            volumes: vec![1.0; ts.len()],
            timestamps: ts.to_vec(),
            opens: vec![],
            highs: vec![],
            lows: vec![],
        };
        let mut v = HashMap::new();
        v.insert(("BTC".to_string(), "a".to_string()), mk(&[10, 20, 30, 40]));
        v.insert(
            ("BTC".to_string(), "b".to_string()),
            mk(&[10, 20, 30, 40, 50]),
        );
        let cut = truncate_venues_at(&v, 30);
        assert_eq!(
            cut[&("BTC".to_string(), "a".to_string())].timestamps,
            vec![10, 20, 30]
        );
        assert_eq!(
            cut[&("BTC".to_string(), "b".to_string())].prices,
            vec![10.0, 20.0, 30.0]
        );
        assert!(cut[&("BTC".to_string(), "b".to_string())].opens.is_empty());
    }
}
