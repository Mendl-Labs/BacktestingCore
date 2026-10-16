//! Single-asset one-day reversal (`weightsim::WeightRule`). Each day, hold the WHOLE sleeve (weight 1.0) in the
//! sleeve's one instrument when YESTERDAY's bar return was strictly negative (buy after a down day), else the
//! whole sleeve is in cash (weight 0.0, i.e. flat after an up day or an exactly-flat day). Long only, single
//! instrument, simplest possible short-horizon reversal: no lookback parameter, no sizing, just yesterday's sign.
//!
//! Companion to [`crate::sma_crossover_trend`]/[`crate::dual_ma_crossover`]: same single-asset, daily,
//! silent-skip-on-insufficient-history convention (NOT the per-asset-typed-exclusion pattern
//! `top_n_winners`/`momentum_rank_weighted` use for their fixed baskets -- this rule has exactly one instrument, so
//! there is nothing to exclude, only to skip warm-up on).
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Yesterday's return, not today's*: the decision at the close of bar `t` (today) looks at the return realized
//!    OVER bar `t-1` (yesterday), i.e. `close_(t-1) / close_(t-2) - 1`. Today's own close (`close_t`) is the
//!    decision bar's presence requirement only -- its VALUE never enters the formula at all. (Tested explicitly:
//!    two fixtures sharing the same `close_(t-2)`/`close_(t-1)` pair but wildly different `close_t` must produce
//!    the identical weight.)
//! 2. *Strictly negative*: `yesterday_return < 0.0`, not `<= 0.0` -- a yesterday return of EXACTLY zero (two
//!    bit-identical consecutive closes, e.g. a halted/unchanged session) gives weight 0.0 (flat), not 1.0 (long).
//!    Tested at that exact boundary with a hand-verifiable bit-exact tie (two identical closes, so the ratio is
//!    exactly `1.0` and the return is exactly `0.0`, no floating-point rounding involved).
//! 3. *No exact-arithmetic machinery needed* (unlike [`crate::sma_crossover_trend`]/[`crate::dual_ma_crossover`]'s
//!    SMA comparisons, which route through [`crate::exact`] to guard against summation-order rounding). This
//!    formula is a single division and a single subtraction over two scalars -- no summation at all, so there is
//!    no summation-order rounding artifact for an exact cross-multiply to guard against (the same reasoning
//!    [`crate::momentum_rank_weighted`]'s module doc gives, choice 4, for skipping `exact` on its own one-division/
//!    one-subtraction trailing-return formula). Plain IEEE-754 `f64` comparison is exact enough here: the only
//!    boundary that matters (return exactly `0.0`) is produced bit-exactly whenever the two closes are bit-equal,
//!    with no rounding path that could manufacture or hide that boundary.
//! 4. *Universe placeholder.* Same convention as `sma_crossover_trend`/`dual_ma_crossover` (their own choice 3):
//!    this primitive runs on whatever single asset a library entry's `RuleSpec` names, so [`universe`] returns the
//!    generic placeholder `["ASSET"]`, never a real ticker.
//! 5. *Insufficient history is a SILENT SKIP, not a refusal* (same convention as `sma_crossover_trend`/
//!    `dual_ma_crossover`, their own choice 4). [`OneDayReversal::min_history_bars`] returns `3`: today's close (the
//!    decision bar's presence requirement, bar `t`), plus the two closes the formula actually reads,
//!    `close_(t-1)` and `close_(t-2)`. Per `weightsim::sim` (`decision_bar(t) and t+1 >= min_history_bars`), the
//!    simulator never calls `target_weights` before 3 closes are visible, so [`OneDayReversal::target_weights`]
//!    (via [`one_day_reversal_weight`]) assumes `h.len() >= 3` and never itself checks or refuses for insufficient
//!    history.
//! 6. *Rebalance policy*: [`weightsim::RebalancePolicy::EveryBar`], same reasoning as `sma_crossover_trend`'s
//!    choice 5 and `dual_ma_crossover`'s choice 5: this is also a `Daily`-scheduled rule, and a daily schedule has
//!    no natural "next decision" for units to drift until the way a month-end schedule does.
//! 7. *No tunable parameters at all*, unlike `sma_crossover_trend` (`N`) or `dual_ma_crossover` (`N1`, `N2`): the
//!    formula is fixed (yesterday's sign, full-stop), so [`OneDayReversal`] is a zero-sized unit-like struct with a
//!    trivial [`OneDayReversal::new`] constructor (kept, rather than a bare `OneDayReversal` literal everywhere,
//!    for symmetry with the other single-asset rules' `::new`/`::default` call sites) and [`declared_parameters`]
//!    is NOT overridden -- it falls through to `WeightRule`'s own default (`BTreeMap::new()`), since there is
//!    genuinely nothing compile-time-configurable to report, unlike `sma_crossover_trend`/`dual_ma_crossover` which
//!    both override it to report their numeric constants.

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Minimum bars visible before this rule will be asked to decide: today's close (the decision bar) plus the two
/// closes yesterday's return itself reads (interpretation choice 5).
pub const ONE_DAY_REVERSAL_MIN_HISTORY_BARS: usize = 3;

/// Generic single-asset universe placeholder (interpretation choice 4): the real instrument is named by the
/// library entry's `RuleSpec`, not by this rule.
pub const ONE_DAY_REVERSAL_UNIVERSE: [&str; 1] = ["ASSET"];

/// Version string recorded in every run (it is part of the series digest).
pub const ONE_DAY_REVERSAL_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " one_day_reversal"
);

/// `one_day_reversal`: weight 1.0 in the sleeve's one asset when YESTERDAY's bar return was strictly negative,
/// else weight 0.0. See the module doc for the interpretation choices (yesterday vs. today, the strict-negative
/// boundary, the universe placeholder, the rebalance policy, the silent-skip warm-up).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OneDayReversal;

impl OneDayReversal {
    /// This rule has no tunable parameters (interpretation choice 7); the constructor takes no arguments.
    pub fn new() -> Self {
        OneDayReversal
    }
}

/// Pure computation, exposed for direct unit testing with synthetic slices (no `HistoryView` needed): weight 1.0
/// if yesterday's bar return, `closes[closes.len() - 2] / closes[closes.len() - 3] - 1.0`, is strictly negative,
/// else weight 0.0. `closes` must end at the decision bar (today); `target_weights` is a one-line wrapper around
/// this over `h.closes(0)`. Today's own close, `closes[closes.len() - 1]`, is never read.
///
/// Panics if `closes.len() < 3` (the caller's contract, enforced by `min_history_bars`/the simulator -- see
/// interpretation choice 5); never called by `target_weights` until that holds.
pub(crate) fn one_day_reversal_weight(closes: &[f64]) -> f64 {
    let n = closes.len();
    assert!(n >= 3, "one_day_reversal: need at least 3 closes, got {n}");
    let yesterday = closes[n - 2];
    let day_before_yesterday = closes[n - 3];
    let yesterday_return = yesterday / day_before_yesterday - 1.0;
    if yesterday_return < 0.0 {
        1.0
    } else {
        0.0
    }
}

impl WeightRule for OneDayReversal {
    fn id(&self) -> &'static str {
        "one_day_reversal"
    }
    fn impl_version(&self) -> String {
        ONE_DAY_REVERSAL_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &ONE_DAY_REVERSAL_UNIVERSE
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        ONE_DAY_REVERSAL_MIN_HISTORY_BARS
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        Ok(vec![one_day_reversal_weight(h.closes(0))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_history_bars_is_three() {
        assert_eq!(OneDayReversal::new().min_history_bars(), 3);
        assert_eq!(OneDayReversal::default().min_history_bars(), 3);
        assert_eq!(ONE_DAY_REVERSAL_MIN_HISTORY_BARS, 3);
    }

    #[test]
    fn declared_parameters_is_empty_the_trait_default() {
        // No tunable parameters at all (interpretation choice 7): falls through to `WeightRule`'s own default.
        assert!(OneDayReversal::new().declared_parameters().is_empty());
    }

    #[test]
    fn schedule_policy_universe_and_id_are_the_documented_ones() {
        let r = OneDayReversal::new();
        assert_eq!(r.id(), "one_day_reversal");
        assert_eq!(r.universe(), &["ASSET"]);
        assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(r.rebalance_policy(), RebalancePolicy::EveryBar);
    }

    // (a) Down day: yesterday's return (day_before=100.0 -> yesterday=90.0, a -10% move) is strictly negative ->
    // weight 1.0 (buy after a down day). Today's close (the trailing element) is deliberately a large, unrelated
    // value (12345.0) to make it obvious it plays no role in the computation.
    #[test]
    fn down_day_gives_full_weight() {
        let closes = [100.0, 90.0, 12345.0];
        // Hand-computed: yesterday_return = 90.0 / 100.0 - 1.0 = -0.10, strictly negative.
        let w = one_day_reversal_weight(&closes);
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (b) Up day: yesterday's return (day_before=100.0 -> yesterday=110.0, a +10% move) is strictly positive ->
    // weight 0.0 (flat after an up day).
    #[test]
    fn up_day_gives_zero_weight() {
        let closes = [100.0, 110.0, 1.0];
        // Hand-computed: yesterday_return = 110.0 / 100.0 - 1.0 = 0.10, strictly positive.
        let w = one_day_reversal_weight(&closes);
        assert!(w.abs() < 1e-9, "expected weight 0.0, got {w}");
    }

    // (c) Exact-zero boundary: yesterday's return is EXACTLY 0.0 (day_before and yesterday are bit-identical
    // closes, 50.0 each, so 50.0 / 50.0 == 1.0 exactly and 1.0 - 1.0 == 0.0 exactly -- a hand-verifiable tie, not
    // a "close enough" one) -> weight 0.0, NOT 1.0 (the spec's `<`, not `<=`). This is the test the `<=` mutant
    // below is designed to flip.
    #[test]
    fn exact_zero_yesterday_return_is_not_negative() {
        let closes = [50.0, 50.0, 999.0];
        let yesterday_return = closes[1] / closes[0] - 1.0;
        assert_eq!(
            yesterday_return, 0.0,
            "fixture must be an EXACT zero return by construction"
        );
        let w = one_day_reversal_weight(&closes);
        assert!(
            w.abs() < 1e-9,
            "an exact-zero return must be weight 0.0 (<, not <=), got {w}"
        );
    }

    // (d) Today's close value never affects the result: same (day_before, yesterday) pair, wildly different
    // today's closes.
    #[test]
    fn todays_close_value_does_not_affect_the_result() {
        let a = one_day_reversal_weight(&[100.0, 90.0, 1.0]);
        let b = one_day_reversal_weight(&[100.0, 90.0, 1_000_000.0]);
        let c = one_day_reversal_weight(&[100.0, 90.0, 0.0001]);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert!((a - 1.0).abs() < 1e-9);
    }

    // (e) History older than the 3 bars needed does not change the answer: extra leading bars, however many,
    // beyond the trailing (day_before, yesterday, today) triple are ignored.
    #[test]
    fn extra_history_beyond_the_minimum_is_ignored() {
        let mut closes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12345.0];
        closes.extend_from_slice(&[100.0, 90.0, 1.0]); // same trailing triple as the down-day test above
        let w = one_day_reversal_weight(&closes);
        assert!(
            (w - 1.0).abs() < 1e-9,
            "expected weight 1.0 (unaffected by older history), got {w}"
        );
    }

    // (f) Minimum valid history: exactly 3 closes visible (h.len() == min_history_bars()), no extra bars at all.
    // This is the earliest bar the simulator would ever call `target_weights` on.
    #[test]
    fn exactly_three_bars_of_history_gives_a_defined_answer() {
        let closes = [100.0, 90.0, 1.0];
        assert_eq!(closes.len(), ONE_DAY_REVERSAL_MIN_HISTORY_BARS);
        let w = one_day_reversal_weight(&closes);
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }
}
