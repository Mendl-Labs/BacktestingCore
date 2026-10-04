//! Single-asset SMA crossover trend (`weightsim::WeightRule`). Each day, hold the WHOLE sleeve (weight 1.0) in the
//! sleeve's one instrument when its close is strictly above the simple average of its last `N` daily closes,
//! INCLUDING today's; otherwise the whole sleeve is in cash (weight 0.0). Long only, single instrument.
//!
//! Unlike `crypto.rs`/`etf.rs` (which predate `weightsim::WeightRule` and are plain `decide_*` functions over this
//! crate's own `Panel`), this is the first rule in `reference-rules` that implements the trait directly, so it adds
//! `weightsim` as a path dependency (see `Cargo.toml`; `weightsim` itself has zero dependencies, so the crate stays
//! dependency-light).
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Strictly above*: `close == SMA` is NOT above (weight 0.0 on a tie), compared via [`crate::exact::compare_to_mean`]
//!    -- the same exact (non-floating-point-summation-order-dependent) tie-break already used by `crypto.rs`/`etf.rs`,
//!    so binary rounding in the SMA's running sum cannot turn a real tie into a signal, or vice versa.
//! 2. *SMA includes the current bar* (the same "R1" reading used by the crypto and ETF trend rules in this crate).
//! 3. *Universe placeholder.* Unlike `crypto_trend_100d`/`etf_trend_faber`, which name real instruments, this
//!    primitive is meant to run on whatever single asset a library entry's `RuleSpec` names, so [`universe`]
//!    returns the generic placeholder `["ASSET"]`, never a real ticker.
//! 4. *Insufficient history is a SILENT SKIP, not a refusal* (the OTHER convention `weightsim::WeightRule::min_history_bars`
//!    documents, as opposed to `crypto_trend_100d`'s adapter, whose `min_history_bars() == 1` makes its own
//!    `InsufficientHistory` a `Warmup` refusal instead). `min_history_bars()` here returns `N`; per `weightsim::sim`
//!    (`decision_bar(t) and t+1 >= min_history_bars`), the simulator never calls `target_weights` before `N` closes
//!    are visible, so [`SmaCrossoverTrend::target_weights`] assumes `h.len() >= N` and never itself checks or
//!    refuses for insufficient history.
//! 5. *Rebalance policy*: [`weightsim::RebalancePolicy::EveryBar`], following the one other DAILY rule already wired
//!    as a `WeightRule` in this codebase (`weightsim_rules::adapters::CryptoTrendRule`, also daily, also
//!    trend-following): the sleeve is traded back to the standing 1.0/0.0 target every bar. A `Daily` schedule has
//!    no natural "next decision" for units to drift until the way a month-end schedule does, which is why the OTHER
//!    daily rule in this codebase does not use `OnDecision` either.
//! 6. [`declared_parameters`] reports the one compile-time-configurable parameter under the key `"N"` (the spec's
//!    own name for it), rendered as its bare canonical JSON integer (e.g. `"200"`, no surrounding quotes, the same
//!    `.to_string()` convention `weightsim_rules::adapters::CryptoTrendRule::declared_parameters` uses for its own
//!    numeric constants).
//! 7. *Scale guard.* [`crate::exact::compare_to_mean`] refuses (rather than panicking) when the close/window span too
//!    wide a binary-exponent range for its exact integer comparison (`price_scale_too_wide`, the same code
//!    `RuleError::PriceScaleTooWide` uses in `crypto.rs`/`etf.rs`) or when the window holds more than 4096 bars; the
//!    latter only matters for an `N` far larger than any documented sleeve uses. `Panel::new` already guarantees
//!    every close is finite and positive, so the non-finite/non-positive branch of that guard is unreachable here.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

use crate::exact::compare_to_mean;

/// Default lookback `N`: number of daily closes in the average, including today's.
pub const SMA_CROSSOVER_DEFAULT_N: usize = 200;

/// Generic single-asset universe placeholder (interpretation choice 3): the real instrument is named by the
/// library entry's `RuleSpec`, not by this rule.
pub const SMA_CROSSOVER_UNIVERSE: [&str; 1] = ["ASSET"];

/// Version string recorded in every run (it is part of the series digest).
pub const SMA_CROSSOVER_VERSION: &str = concat!("reference-rules ", env!("CARGO_PKG_VERSION"), " sma_crossover_trend");

/// `sma_crossover_trend`: weight 1.0 in the sleeve's one asset when its close is strictly above the simple average
/// of its last `n` daily closes (today's included), else weight 0.0. See the module doc for the interpretation
/// choices (universe placeholder, rebalance policy, silent-skip warm-up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmaCrossoverTrend {
    n: usize,
}

impl SmaCrossoverTrend {
    /// A rule with lookback `n` (must be >= 1; the simulator never calls `target_weights` before `n` closes are
    /// visible, see `min_history_bars`).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "sma_crossover_trend: N must be >= 1, got {n}");
        SmaCrossoverTrend { n }
    }

    /// The configured lookback `N`.
    pub fn n(&self) -> usize {
        self.n
    }
}

impl Default for SmaCrossoverTrend {
    /// `N` = [`SMA_CROSSOVER_DEFAULT_N`] (200).
    fn default() -> Self {
        SmaCrossoverTrend::new(SMA_CROSSOVER_DEFAULT_N)
    }
}

/// Pure computation, exposed for direct unit testing with synthetic slices (no `HistoryView` needed): weight 1.0 if
/// `closes` (ending at the decision bar, `closes.len() >= n`) is strictly above the simple average of its last `n`
/// values (today's included), else weight 0.0. `target_weights` is a one-line wrapper around this over
/// `h.closes(0)`.
///
/// Panics if `closes.len() < n` (the caller's contract, enforced by `min_history_bars`/the simulator -- see
/// interpretation choice 4); never called by `target_weights` until that holds.
pub(crate) fn sma_crossover_weight(closes: &[f64], n: usize) -> Result<f64, RuleRefusal> {
    let window = &closes[closes.len() - n..];
    let close = *closes.last().expect("closes is non-empty: n >= 1 and closes.len() >= n");
    match compare_to_mean(close, window) {
        Some(cmp) if cmp.ordering.is_gt() => Ok(1.0),
        Some(_) => Ok(0.0),
        None => Err(RuleRefusal::data(
            "price_scale_too_wide",
            format!("sma_crossover_trend: close/window span too wide a scale for exact comparison (n={n})"),
        )),
    }
}

impl WeightRule for SmaCrossoverTrend {
    fn id(&self) -> &'static str {
        "sma_crossover_trend"
    }
    fn impl_version(&self) -> String {
        SMA_CROSSOVER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &SMA_CROSSOVER_UNIVERSE
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([("N", self.n.to_string())])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        self.n
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        sma_crossover_weight(h.closes(0), self.n).map(|w| vec![w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 5;

    #[test]
    fn min_history_bars_is_n() {
        assert_eq!(SmaCrossoverTrend::new(N).min_history_bars(), N);
        assert_eq!(SmaCrossoverTrend::new(37).min_history_bars(), 37);
        assert_eq!(SmaCrossoverTrend::default().min_history_bars(), SMA_CROSSOVER_DEFAULT_N);
        assert_eq!(SMA_CROSSOVER_DEFAULT_N, 200);
    }

    #[test]
    fn declared_parameters_render_n_as_a_bare_json_integer() {
        let params = SmaCrossoverTrend::new(200).declared_parameters();
        assert_eq!(params["N"], "200");
        let params = SmaCrossoverTrend::new(37).declared_parameters();
        assert_eq!(params["N"], "37");
    }

    #[test]
    fn schedule_policy_universe_and_id_are_the_documented_ones() {
        let r = SmaCrossoverTrend::default();
        assert_eq!(r.id(), "sma_crossover_trend");
        assert_eq!(r.universe(), &["ASSET"]);
        assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(r.rebalance_policy(), RebalancePolicy::EveryBar);
    }

    // (a) Clean crossover: close strictly ABOVE the SMA of the last N closes -> weight 1.0. Two bogus leading
    // closes (9999.0) are included in `closes` but must be EXCLUDED from the N=5 window; if a buggy implementation
    // used an (N+1)-wide window it would pull one of them in and flip this answer (see mutant M-SMA1 below).
    #[test]
    fn above_sma_gives_full_weight() {
        let closes = [9999.0, 9999.0, 100.0, 100.0, 100.0, 100.0, 140.0];
        // Hand-computed: window = last 5 closes = [100, 100, 100, 100, 140]; mean = 540 / 5 = 108.0; 140 > 108.0.
        let w = sma_crossover_weight(&closes, N).unwrap();
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (b) Close strictly BELOW the SMA of the last N closes -> weight 0.0.
    #[test]
    fn below_sma_gives_zero_weight() {
        let closes = [1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 60.0];
        // Hand-computed: window = last 5 closes = [100, 100, 100, 100, 60]; mean = 460 / 5 = 92.0; 60 < 92.0.
        let w = sma_crossover_weight(&closes, N).unwrap();
        assert!(w.abs() < 1e-9, "expected weight 0.0, got {w}");
    }

    // (c) Exact tie: close_t == SMA(window) -> weight 0.0 (the spec's `>`, not `>=`). The window is five equal
    // closes (50.0 each), so the mean is EXACTLY 50.0 in binary floating point (5 * 50.0 = 250.0 and 250.0 / 5.0 =
    // 50.0 are both exact; no rounding from summation order can occur), and the decision close is also exactly
    // 50.0 -- a hand-verifiable tie, not a "close enough" one. The two bogus leading closes (1.0, 1.0) are, again,
    // outside the N=5 window; including them would raise the mean above the decision close and turn the tie into
    // a false "above" signal (see mutant M-SMA1).
    #[test]
    fn exact_tie_is_not_above() {
        let closes = [1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0];
        let window = &closes[closes.len() - N..];
        let mean: f64 = window.iter().sum::<f64>() / N as f64;
        assert_eq!(mean, 50.0, "fixture must be an EXACT tie by construction");
        assert_eq!(*closes.last().unwrap(), mean);
        let w = sma_crossover_weight(&closes, N).unwrap();
        assert!(w.abs() < 1e-9, "a tie must be weight 0.0 (>, not >=), got {w}");
    }

    // (d) Minimum valid history: exactly N closes visible (h.len() == min_history_bars()), no extra bars at all.
    // This is the earliest bar the simulator would ever call `target_weights` on, and the boundary where an
    // (N+1)-wide window bug would underflow `closes.len() - n` (panic) rather than merely compute the wrong answer.
    #[test]
    fn exactly_n_bars_of_history_gives_a_defined_answer() {
        let closes = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(closes.len(), N);
        // Hand-computed: mean = (10+20+30+40+50) / 5 = 150 / 5 = 30.0; 50.0 > 30.0.
        let w = sma_crossover_weight(&closes, N).unwrap();
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    #[test]
    fn default_constructor_uses_the_documented_default_n() {
        assert_eq!(SmaCrossoverTrend::default(), SmaCrossoverTrend::new(SMA_CROSSOVER_DEFAULT_N));
    }
}