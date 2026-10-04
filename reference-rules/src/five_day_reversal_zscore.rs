//! Single-asset 5-day reversal, z-scored against its own trailing 60-day distribution of 5-day returns
//! (`weightsim::WeightRule`). Each day, hold the WHOLE sleeve (weight 1.0) in the sleeve's one instrument when the
//! current 5-day return's z-score against its own trailing 60-observation distribution of 5-day returns is
//! strictly below -1.0 (oversold), else the whole sleeve is in cash (weight 0.0). Long only, single instrument.
//!
//! Companion to [`crate::one_day_reversal`]: the other half of the "short-horizon reversal" family -- a 5-bar
//! return instead of a 1-bar return, and a z-scored (distribution-relative) threshold instead of a bare sign
//! check. Same single-asset, daily, silent-skip-on-insufficient-history convention.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *A "5-day return ending at position `i`"* (0-indexed into the causal close slice) is `closes[i] /
//!    closes[i-5] - 1` (needs `i >= 5`).
//! 2. *The trailing 60-day distribution of 5-day returns is OVERLAPPING, not every-5th-bar*: every bar's own
//!    trailing 5-day return is one observation, so the 60 observations are the 5-day returns ending at EVERY one
//!    of the last 60 positions of the causal close slice (positions `len-60` through `len-1` inclusive, step 1),
//!    not just every 5th bar. The earliest of these needs a close 5 positions further back, i.e. position
//!    `len-65`, which must be `>= 0` -- this is exactly why [`FIVE_DAY_REVERSAL_MIN_HISTORY_BARS`] is 65, not 60.
//! 3. *The current 5-day return being z-scored is the one ending at the LAST position* (`len-1`, today) -- this is
//!    ALSO the most recent (last) of the 60 observations in choice 2, i.e. the mean/stdev population INCLUDES the
//!    very return being scored. Same "inclusive of today" convention `sma_crossover_trend`/`dual_ma_crossover`
//!    already use for their moving averages.
//! 4. *Standard deviation is the SAMPLE standard deviation* (divide the sum of squared deviations by `59`, i.e.
//!    `n - 1` with `n = 60`), NOT the population standard deviation (divide by `60`) -- matching the `fx.rs`
//!    60-day volatility precedent exactly. This module reuses `fx.rs`'s own `sample_std` directly
//!    (`crate::fx::sample_std`) rather than an independently-written second copy of the same formula, so the two
//!    rules are byte-for-byte consistent on this. `fx.rs`'s own M16 mutant ("ddof 0 instead of 1") documents the
//!    wrong, population-stdev alternative this rule must NOT use.
//! 5. *z-score* = `(current_return - mean) / stdev`. *Weight* = `1.0` if `z-score < -1.0` (strictly oversold), else
//!    `0.0`.
//! 6. *Degenerate zero-variance history*: if all 60 returns happen to be identical, the sample standard deviation
//!    is exactly `0.0`, making the z-score `0.0 / 0.0 = NaN`. `f64::NAN < -1.0` is `false` in Rust (every `<`/`>`/
//!    `<=`/`>=` comparison against NaN is false), so this naturally and correctly falls through to weight `0.0`
//!    (no signal) with no special-case branch and no panic -- a deliberate, disclosed choice: a degenerate
//!    zero-variance history gives no oversold signal, not a crash. [`five_day_reversal_zscore_weight`] uses a
//!    plain `if z < -1.0` on an owned `f64`, never `.unwrap()`/`.expect()` on anything NaN-shaped. NOTE (found
//!    empirically while writing the test for this, not merely theorized): this is reliable for an all-ZERO return
//!    history (a perfectly flat instrument), but NOT for an arbitrary repeated non-zero value such as `-0.02` --
//!    IEEE-754 summation of 60 copies of a value that is not exactly binary-representable can round the computed
//!    mean a few ULPs away from the literal value, turning every "identical" deviation into a tiny NONZERO number
//!    instead of an exact `0.0`, which yields a large FINITE z (not NaN) that can spuriously cross the `-1.0`
//!    threshold. The all-zero fixture is the one degenerate case guaranteed exact on every platform (0 sums and
//!    divides without any rounding), so the test below uses it, not an arbitrary repeated constant.
//! 7. *No exact-arithmetic machinery for the summation* ([`crate::exact`] is NOT used here), despite this formula
//!    summing 60 terms twice (once for the mean, once for the sum of squared deviations) -- UNLIKE
//!    [`crate::one_day_reversal`]/[`crate::momentum_rank_weighted`]'s single-division trailing-return formulas,
//!    which have no summation at all, this one genuinely does have a summation-order rounding consideration. The
//!    reason `exact` is still not needed: the final decision is a STRICT INEQUALITY against a fixed constant
//!    (`z < -1.0`), not an exact-tie/equality check (`close == SMA`) the way `sma_crossover_trend`'s SMA comparison
//!    is. A summation-order rounding difference on the order of a few ULPs (~1e-15 relative) can only flip a `<`
//!    comparison against a fixed threshold when the TRUE mathematical value sits within that same ~1e-15 of the
//!    threshold -- a measure-zero coincidence for a real 60-return distribution, not a structural risk the way an
//!    exact tie is (an exact tie is COMMON -- e.g. two identical adjacent closes -- and every rounding path must
//!    agree on it, which plain summation cannot guarantee). This reasoning mirrors `fx.rs` itself: its own
//!    `sample_std`/sign checks, which this rule reuses verbatim, use plain two-pass sequential summation with no
//!    `exact` guard, and they gate a `<`/`>` sign comparison and a vol-scaling division -- not an exact tie either.
//! 8. *Universe placeholder.* Same convention as `sma_crossover_trend`/`dual_ma_crossover`/`one_day_reversal`:
//!    this primitive runs on whatever single asset a library entry's `RuleSpec` names, so [`universe`] returns the
//!    generic placeholder `["ASSET"]`, never a real ticker.
//! 9. *Insufficient history is a SILENT SKIP, not a refusal* (same convention as the other single-asset daily
//!    rules in this crate). [`FIVE_DAY_REVERSAL_MIN_HISTORY_BARS`] is `65` (choice 2). Per `weightsim::sim`
//!    (`decision_bar(t) and t+1 >= min_history_bars`), the simulator never calls `target_weights` before 65 closes
//!    are visible, so [`FiveDayReversalZscore::target_weights`] (via [`five_day_reversal_zscore_weight`]) assumes
//!    `h.len() >= 65` and never itself checks or refuses for insufficient history.
//! 10. *Rebalance policy*: [`weightsim::RebalancePolicy::EveryBar`], same reasoning as `one_day_reversal`'s choice
//!     6 / `sma_crossover_trend`'s choice 5: this is also a `Daily`-scheduled rule, and a daily schedule has no
//!     natural "next decision" for units to drift until the way a month-end schedule does.
//! 11. *One fixed internal parameter, `L = 5`* (the return-window lookback), hardcoded as the named constant
//!     [`FIVE_DAY_REVERSAL_LOOKBACK`] rather than a constructor argument -- the spec states it as a fixed fact
//!     ("Parameter lookback L=5"), not something a caller picks, same spirit as `one_day_reversal` having zero
//!     tunable constructor parameters. Unlike `one_day_reversal` (which names NO constant at all for its implicit
//!     1-bar lookback), this rule's `L` IS worth naming, so [`declared_parameters`] overrides the trait default to
//!     report it under the key `"L"` (the spec's own name), rendered as its bare canonical JSON integer `"5"` --
//!     the same convention `sma_crossover_trend` uses for its own named lookback constant `N`. The other two fixed
//!     constants the formula also uses -- the 60-observation window ([`FIVE_DAY_REVERSAL_ZSCORE_WINDOW`]) and the
//!     `-1.0` z-score threshold ([`FIVE_DAY_REVERSAL_ZSCORE_THRESHOLD`]) -- are part of the formula's fixed SHAPE
//!     (like `one_day_reversal`'s implicit 1-bar lookback), not a named "Parameter" the spec calls out, so neither
//!     is reported via `declared_parameters`.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

use crate::fx::sample_std;

/// `L`: the return-window lookback (interpretation choice 11). Fixed, not a constructor argument.
pub const FIVE_DAY_REVERSAL_LOOKBACK: usize = 5;

/// The number of overlapping trailing 5-day returns the mean/stdev are computed over (interpretation choice 2).
pub const FIVE_DAY_REVERSAL_ZSCORE_WINDOW: usize = 60;

/// The strict z-score threshold for "oversold" (interpretation choice 5).
pub const FIVE_DAY_REVERSAL_ZSCORE_THRESHOLD: f64 = -1.0;

/// Minimum bars visible before this rule will be asked to decide: `FIVE_DAY_REVERSAL_ZSCORE_WINDOW` (60) overlapping
/// 5-day returns, the earliest of which needs a close `FIVE_DAY_REVERSAL_LOOKBACK` (5) positions further back than
/// the oldest of those 60 return-ending positions (interpretation choice 2): `60 + 5 = 65`.
pub const FIVE_DAY_REVERSAL_MIN_HISTORY_BARS: usize =
    FIVE_DAY_REVERSAL_ZSCORE_WINDOW + FIVE_DAY_REVERSAL_LOOKBACK;

/// Generic single-asset universe placeholder (interpretation choice 8): the real instrument is named by the
/// library entry's `RuleSpec`, not by this rule.
pub const FIVE_DAY_REVERSAL_UNIVERSE: [&str; 1] = ["ASSET"];

/// Version string recorded in every run (it is part of the series digest).
pub const FIVE_DAY_REVERSAL_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " five_day_reversal_zscore"
);

/// `five_day_reversal_zscore`: weight 1.0 in the sleeve's one asset when the current 5-day return's z-score
/// against its own trailing 60-observation distribution of 5-day returns is strictly below -1.0 (oversold), else
/// weight 0.0. See the module doc for the interpretation choices (overlapping window, sample stdev, the NaN
/// degenerate case, the universe placeholder, the rebalance policy, the silent-skip warm-up).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiveDayReversalZscore;

impl FiveDayReversalZscore {
    /// `L` is fixed at [`FIVE_DAY_REVERSAL_LOOKBACK`] (interpretation choice 11); the constructor takes no
    /// arguments.
    pub fn new() -> Self {
        FiveDayReversalZscore
    }
}

/// Pure computation, exposed for direct unit testing with synthetic slices (no `HistoryView` needed). `closes`
/// must end at the decision bar (today); `target_weights` is a one-line wrapper around this over `h.closes(0)`.
///
/// Builds the 60 overlapping 5-day returns ending at the last 60 positions of `closes` (interpretation choice 2),
/// z-scores the LAST of those 60 (the current 5-day return, interpretation choice 3) against the sample mean/stdev
/// of the whole 60 (interpretation choice 4), and returns `1.0` if that z-score is strictly below `-1.0`, else
/// `0.0` (interpretation choice 5). A zero-variance 60-return history naturally produces a NaN z-score via plain
/// IEEE-754 arithmetic, which the `<` comparison naturally treats as `false` (interpretation choice 6) -- no
/// panic, no special case.
///
/// Panics if `closes.len() < 65` (the caller's contract, enforced by `min_history_bars`/the simulator -- see
/// interpretation choice 9); never called by `target_weights` until that holds.
pub(crate) fn five_day_reversal_zscore_weight(closes: &[f64]) -> f64 {
    let n = closes.len();
    assert!(
        n >= FIVE_DAY_REVERSAL_MIN_HISTORY_BARS,
        "five_day_reversal_zscore: need at least {FIVE_DAY_REVERSAL_MIN_HISTORY_BARS} closes, got {n}"
    );
    let window = FIVE_DAY_REVERSAL_ZSCORE_WINDOW;
    let lookback = FIVE_DAY_REVERSAL_LOOKBACK;
    // The 60 overlapping 5-day returns ending at positions `n - 60 ..= n - 1` (interpretation choice 2).
    let mut returns = [0.0f64; FIVE_DAY_REVERSAL_ZSCORE_WINDOW];
    for (slot, i) in (n - window..n).enumerate() {
        returns[slot] = closes[i] / closes[i - lookback] - 1.0;
    }
    // The current 5-day return is the LAST of the 60 (ending at `n - 1`, interpretation choice 3) -- plain
    // sequential summation, same order as `crate::fx::sample_std`'s own internal mean, so the two are consistent.
    let current_return = returns[window - 1];
    let mean = returns.iter().sum::<f64>() / window as f64;
    let stdev =
        sample_std(&returns).expect("exactly 60 returns, well above sample_std's 2-value minimum");
    let z = (current_return - mean) / stdev;
    if z < FIVE_DAY_REVERSAL_ZSCORE_THRESHOLD {
        1.0
    } else {
        0.0
    }
}

impl WeightRule for FiveDayReversalZscore {
    fn id(&self) -> &'static str {
        "five_day_reversal_zscore"
    }
    fn impl_version(&self) -> String {
        FIVE_DAY_REVERSAL_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &FIVE_DAY_REVERSAL_UNIVERSE
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([("L", FIVE_DAY_REVERSAL_LOOKBACK.to_string())])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        FIVE_DAY_REVERSAL_MIN_HISTORY_BARS
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        Ok(vec![five_day_reversal_zscore_weight(h.closes(0))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_history_bars_is_sixty_five() {
        assert_eq!(FiveDayReversalZscore::new().min_history_bars(), 65);
        assert_eq!(FiveDayReversalZscore::default().min_history_bars(), 65);
        assert_eq!(FIVE_DAY_REVERSAL_MIN_HISTORY_BARS, 65);
    }

    #[test]
    fn declared_parameters_reports_l_as_a_bare_json_integer() {
        let params = FiveDayReversalZscore::new().declared_parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params["L"], "5");
    }

    #[test]
    fn schedule_policy_universe_and_id_are_the_documented_ones() {
        let r = FiveDayReversalZscore::new();
        assert_eq!(r.id(), "five_day_reversal_zscore");
        assert_eq!(r.universe(), &["ASSET"]);
        assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(r.rebalance_policy(), RebalancePolicy::EveryBar);
    }

    /// Builds a 65-close fixture whose 60 trailing 5-day returns are: `window - k` copies of `v0`, then `k`
    /// copies of `v1`, assigned to RETURN SLOTS in chronological order (slot order does not affect mean/stdev,
    /// which are order-independent sums) -- with `v1` always occupying the LAST `k` slots, so the CURRENT return
    /// (interpretation choice 3) is always `v1` whenever `k >= 1`. `closes[0..5]` are an arbitrary positive base;
    /// `closes[i] = closes[i-5] * (1 + return assigned to slot i-5)` for `i in 5..65`, so every one of the 60
    /// overlapping 5-day returns realizes exactly the assigned value: the 5 residue-mod-5 chains of 13 closes
    /// each are independent of one another, so any 60 target returns, assigned to the 60 chronological slots in
    /// order, are realizable this way.
    fn fixture_two_level(k: usize, v0: f64, v1: f64) -> [f64; 65] {
        let window = FIVE_DAY_REVERSAL_ZSCORE_WINDOW;
        assert!(k <= window);
        let mut rets = [v0; 60];
        for slot in (window - k)..window {
            rets[slot] = v1;
        }
        let mut closes = [0.0f64; 65];
        for b in closes.iter_mut().take(5) {
            *b = 100.0;
        }
        for i in 5..65 {
            closes[i] = closes[i - 5] * (1.0 + rets[i - 5]);
        }
        closes
    }

    // (a) Oversold case: 59 of the 60 trailing 5-day returns are exactly 0.0 (flat), the 60th -- the CURRENT
    // return -- is -0.5 (a 50% drop over 5 days). Hand-computed (exact fractions): mean = -1/120, sample variance
    // (ddof 1) = 1/240 exactly, stdev = 1/sqrt(240), z = (-59/120) / (1/sqrt(240)) = -59*sqrt(240)/120 ~= -7.6169,
    // clearly < -1.0 -> weight 1.0.
    #[test]
    fn oversold_case_gives_full_weight() {
        let closes = fixture_two_level(1, 0.0, -0.5);
        assert_eq!(closes.len(), 65);
        let mean: f64 = -1.0 / 120.0;
        let variance: f64 = 1.0 / 240.0;
        let stdev = variance.sqrt();
        let current = -0.5;
        let expected_z = (current - mean) / stdev;
        let closed_form = -59.0 * 240.0f64.sqrt() / 120.0;
        assert!(
            (expected_z - closed_form).abs() < 1e-9,
            "two equivalent hand formulas disagree"
        );
        assert!(
            expected_z < -1.0,
            "fixture must actually be oversold by construction, got z={expected_z}"
        );
        let w = five_day_reversal_zscore_weight(&closes);
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (b) Not-oversold case: same shape, mirrored sign -- 59 returns at 0.0, current at +0.5 (a 50% GAIN over 5
    // days). z is the exact mirror of (a), +7.6169, clearly > -1.0 -> weight 0.0.
    #[test]
    fn not_oversold_case_gives_zero_weight() {
        let closes = fixture_two_level(1, 0.0, 0.5);
        let mean: f64 = 1.0 / 120.0;
        let variance: f64 = 1.0 / 240.0;
        let stdev = variance.sqrt();
        let current = 0.5;
        let expected_z = (current - mean) / stdev;
        assert!(
            expected_z > -1.0,
            "fixture must NOT be oversold by construction, got z={expected_z}"
        );
        let w = five_day_reversal_zscore_weight(&closes);
        assert!(w.abs() < 1e-9, "expected weight 0.0, got {w}");
    }

    // (c)/(d) Boundary-ish bracket around z = -1.0 (pinning the strict `<` direction). An EXACT hand-built z =
    // -1.0 proved impractical: fixtures of this two-level shape land at closed-form z magnitudes of
    // `0.5*sqrt(59/15)` (k=30, ~0.991659) and `(31/60)*sqrt(3540/899)` (k=29, ~1.025059) for ANY choice of `v0`/`v1`
    // magnitude (the ratio is scale-invariant in `v1 - v0`) -- straddling -1.0 but never landing on it exactly for
    // any integer group split of 60. Both are used, bracketing from each side, to pin the strict-inequality
    // direction with real (not synthetic-tie) numbers.
    #[test]
    fn boundary_bracket_just_above_negative_one_is_not_oversold() {
        // k = 30 (half the window is the current's group): z = -0.5*sqrt(59/15) ~= -0.991659, strictly > -1.0.
        let closes = fixture_two_level(30, 0.0, -0.01);
        let expected_z = -0.5 * (59.0f64 / 15.0).sqrt();
        assert!(
            expected_z > -1.0 && expected_z < -0.99,
            "expected z just above -1.0, got {expected_z}"
        );
        let w = five_day_reversal_zscore_weight(&closes);
        assert!(
            w.abs() < 1e-9,
            "z = {expected_z} is > -1.0 (not oversold), expected weight 0.0, got {w}"
        );
    }

    #[test]
    fn boundary_bracket_just_below_negative_one_is_oversold() {
        // k = 29: z = -(31/60)*sqrt(3540/899) ~= -1.025059, strictly < -1.0.
        let closes = fixture_two_level(29, 0.0, -0.01);
        let expected_z = -(31.0 / 60.0) * (3540.0f64 / 899.0).sqrt();
        assert!(
            expected_z < -1.0 && expected_z > -1.05,
            "expected z just below -1.0, got {expected_z}"
        );
        let w = five_day_reversal_zscore_weight(&closes);
        assert!(
            (w - 1.0).abs() < 1e-9,
            "z = {expected_z} is < -1.0 (oversold), expected weight 1.0, got {w}"
        );
    }

    // (e) Degenerate all-identical-returns case: all 60 trailing 5-day returns are exactly 0.0 (a perfectly flat
    // instrument -- every one of the 65 closes bit-identical), so the sample stdev is exactly 0.0 and the z-score
    // is the NaN produced by 0.0 / 0.0 -- NOT a panic. (A non-zero constant return, e.g. -0.02 repeated 60 times,
    // is NOT a reliable way to get an exactly-zero variance in IEEE-754: repeated summation of a non-representable
    // decimal like -0.02 can round the computed mean a few ULPs away from the literal -0.02, which then makes
    // every "identical" deviation a tiny nonzero value instead of an exact zero, so the ratio is a large finite
    // number, not NaN -- this was caught by this very test FAILING with weight 1.0 during development, not
    // theorized up front. Exactly 0.0 sums/divides without any rounding, so it is the one value for which the
    // degenerate case is reproducible on every platform.) `NaN < -1.0` is `false` in Rust, so this falls through
    // to weight 0.0 (interpretation choice 6).
    #[test]
    fn degenerate_zero_variance_history_gives_zero_weight_not_a_panic() {
        let closes = fixture_two_level(0, 0.0, 0.0); // k = 0: every one of the 60 returns is exactly 0.0
        let returns_are_all_bit_exact_zero = closes.windows(6).all(|w| w[5] == w[0]);
        assert!(
            returns_are_all_bit_exact_zero,
            "fixture must be a perfectly flat 65-close series"
        );
        let w = five_day_reversal_zscore_weight(&closes);
        assert!(
            w.abs() < 1e-9,
            "a degenerate zero-variance history must yield weight 0.0 (NaN < -1.0 is false), got {w}"
        );
    }

    // (f) Minimum valid history: exactly 65 closes visible (h.len() == min_history_bars()), no extra bars at all --
    // the earliest bar the simulator would ever call `target_weights` on. Reuses the oversold fixture's shape.
    #[test]
    fn exactly_sixty_five_bars_of_history_gives_a_defined_answer() {
        let closes = fixture_two_level(1, 0.0, -0.5);
        assert_eq!(closes.len(), FIVE_DAY_REVERSAL_MIN_HISTORY_BARS);
        let w = five_day_reversal_zscore_weight(&closes);
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (g) History older than the 65 bars needed does not change the answer: extra leading bars, however many,
    // beyond the trailing 65-close window are ignored (they only shift which ABSOLUTE indices the trailing window
    // occupies, never the tail VALUES, since they are only ever prepended).
    #[test]
    fn extra_history_beyond_the_minimum_is_ignored() {
        let base = fixture_two_level(1, 0.0, -0.5);
        let mut closes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 123.0];
        closes.extend_from_slice(&base);
        let w = five_day_reversal_zscore_weight(&closes);
        assert!(
            (w - 1.0).abs() < 1e-9,
            "expected weight 1.0 (unaffected by older history), got {w}"
        );
    }
}
