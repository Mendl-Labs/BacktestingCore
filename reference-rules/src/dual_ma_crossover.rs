//! Dual moving-average crossover (`weightsim::WeightRule`). Single asset, daily: hold the WHOLE sleeve (weight
//! 1.0) in the sleeve's one instrument when the simple average of its last `N1` daily closes (the "fast" SMA,
//! INCLUDING today's) is strictly above the simple average of its last `N2` daily closes (the "slow" SMA, also
//! including today's), `N1 < N2`; otherwise the whole sleeve is in cash (weight 0.0). Long only, single instrument.
//!
//! Companion to [`crate::sma_crossover_trend`] (close-vs-one-SMA), the first rule in `reference-rules` to implement
//! `weightsim::WeightRule` directly; its module doc documents conventions shared by both rules (the generic
//! single-asset universe placeholder, the exact-arithmetic tie-break, the silent-skip warm-up convention, the
//! `EveryBar` rebalance policy for a daily schedule). This doc covers only what differs for a two-SMA comparison.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Strictly above*: `SMA(N1) == SMA(N2)` is NOT above (weight 0.0 on a tie), compared via
//!    [`crate::exact::compare_means`] -- an exact (non-floating-point-summation-order-dependent) comparison of the
//!    two means directly (the cross product of each exact sum with the OTHER window's length, in `i128` integer
//!    arithmetic), the same discipline [`crate::exact::compare_to_mean`] already applies to a close vs one mean.
//!    Comparing the two SMAs via this cross-multiply (rather than rounding each mean to f64 first and comparing
//!    those two f64s) avoids a false tie/non-tie purely from where each mean happens to round.
//! 2. *Both SMAs include the current bar* (the same "R1" reading used by `sma_crossover_trend` and the crypto/ETF
//!    trend rules in this crate).
//! 3. *Universe placeholder.* Same convention as `sma_crossover_trend` (its interpretation choice 3): this
//!    primitive runs on whatever single asset a library entry's `RuleSpec` names, so [`universe`] returns the
//!    generic placeholder `["ASSET"]`, never a real ticker.
//! 4. *Insufficient history for the SLOW MA is a SILENT SKIP, not a refusal* (same convention as
//!    `sma_crossover_trend`, its interpretation choice 4). `min_history_bars()` returns `n2`: since `n1 < n2` is
//!    enforced by [`DualMaCrossover::new`], enough history for the slow MA always means enough for the fast one
//!    too, so the slow window is the binding constraint. Per `weightsim::sim` (`decision_bar(t) and
//!    t+1 >= min_history_bars`), the simulator never calls `target_weights` before `n2` closes are visible, so
//!    [`DualMaCrossover::target_weights`] (via [`dual_ma_weight`]) assumes `h.len() >= n2` and never itself checks
//!    or refuses for insufficient history.
//! 5. *Rebalance policy*: [`weightsim::RebalancePolicy::EveryBar`], same reasoning as `sma_crossover_trend`'s
//!    interpretation choice 5: this is also a daily trend rule, and a `Daily` schedule has no natural "next
//!    decision" for units to drift until the way a month-end schedule does.
//! 6. [`declared_parameters`] reports BOTH compile-time-configurable parameters, under the keys `"N1"` (fast) and
//!    `"N2"` (slow), each rendered as its bare canonical JSON integer (e.g. `"50"`, `"200"`, no surrounding
//!    quotes), the same convention `sma_crossover_trend::declared_parameters` uses for its own `"N"`.
//! 7. *Scale guard.* [`crate::exact::compare_means`] refuses (rather than panicking) under the same conditions
//!    [`crate::exact::compare_to_mean`] does (`price_scale_too_wide`: a non-finite/non-positive value, or values
//!    spanning too wide a binary-exponent range, or a window longer than 4096 bars). `Panel::new` already
//!    guarantees every close is finite and positive, so the non-finite/non-positive branch is unreachable here.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

use crate::exact::compare_means;

/// Default fast lookback `N1`: number of daily closes in the fast average, including today's.
pub const DUAL_MA_DEFAULT_N1: usize = 50;
/// Default slow lookback `N2`: number of daily closes in the slow average, including today's.
pub const DUAL_MA_DEFAULT_N2: usize = 200;

/// Generic single-asset universe placeholder (interpretation choice 3): the real instrument is named by the
/// library entry's `RuleSpec`, not by this rule.
pub const DUAL_MA_UNIVERSE: [&str; 1] = ["ASSET"];

/// Version string recorded in every run (it is part of the series digest).
pub const DUAL_MA_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " dual_ma_crossover"
);

/// `dual_ma_crossover`: weight 1.0 in the sleeve's one asset when the fast SMA (last `n1` daily closes) is
/// strictly above the slow SMA (last `n2` daily closes), both including today's close, else weight 0.0. See the
/// module doc for the interpretation choices (universe placeholder, rebalance policy, silent-skip warm-up, the
/// exact two-mean tie-break).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualMaCrossover {
    n1: usize,
    n2: usize,
}

impl DualMaCrossover {
    /// A rule with fast lookback `n1` and slow lookback `n2`. Panics unless `n1 < n2` (the simulator never calls
    /// `target_weights` before `n2` closes are visible, see `min_history_bars`; `n1 < n2` is also what makes `n2`
    /// alone the binding history requirement -- see interpretation choice 4).
    pub fn new(n1: usize, n2: usize) -> Self {
        assert!(
            n1 < n2,
            "dual_ma_crossover: N1 must be < N2, got N1={n1}, N2={n2}"
        );
        DualMaCrossover { n1, n2 }
    }

    /// The configured fast lookback `N1`.
    pub fn n1(&self) -> usize {
        self.n1
    }

    /// The configured slow lookback `N2`.
    pub fn n2(&self) -> usize {
        self.n2
    }
}

impl Default for DualMaCrossover {
    /// `N1` = [`DUAL_MA_DEFAULT_N1`] (50), `N2` = [`DUAL_MA_DEFAULT_N2`] (200).
    fn default() -> Self {
        DualMaCrossover::new(DUAL_MA_DEFAULT_N1, DUAL_MA_DEFAULT_N2)
    }
}

/// Pure computation, exposed for direct unit testing with synthetic slices (no `HistoryView` needed): weight 1.0
/// if the simple average of the last `n1` values of `closes` (ending at the decision bar, today's included,
/// `closes.len() >= n2`) is strictly above the simple average of the last `n2` values, else weight 0.0.
/// `target_weights` is a one-line wrapper around this over `h.closes(0)`.
///
/// Panics if `closes.len() < n2` (the caller's contract, enforced by `min_history_bars`/the simulator -- see
/// interpretation choice 4); never called by `target_weights` until that holds. Does NOT itself check `n1 < n2`
/// (that is `DualMaCrossover::new`'s precondition; this free function trusts its caller the same way
/// `sma_crossover_trend::sma_crossover_weight` trusts `closes.len() >= n`).
pub(crate) fn dual_ma_weight(closes: &[f64], n1: usize, n2: usize) -> Result<f64, RuleRefusal> {
    let fast = &closes[closes.len() - n1..];
    let slow = &closes[closes.len() - n2..];
    match compare_means(fast, slow) {
        Some(cmp) if cmp.ordering.is_gt() => Ok(1.0),
        Some(_) => Ok(0.0),
        None => Err(RuleRefusal::data(
            "price_scale_too_wide",
            format!("dual_ma_crossover: close/window span too wide a scale for exact comparison (n1={n1}, n2={n2})"),
        )),
    }
}

impl WeightRule for DualMaCrossover {
    fn id(&self) -> &'static str {
        "dual_ma_crossover"
    }
    fn impl_version(&self) -> String {
        DUAL_MA_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &DUAL_MA_UNIVERSE
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([("N1", self.n1.to_string()), ("N2", self.n2.to_string())])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        self.n2
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        dual_ma_weight(h.closes(0), self.n1, self.n2).map(|w| vec![w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N1: usize = 3;
    const N2: usize = 5;

    #[test]
    fn min_history_bars_is_n2() {
        assert_eq!(DualMaCrossover::new(N1, N2).min_history_bars(), N2);
        assert_eq!(DualMaCrossover::new(10, 37).min_history_bars(), 37);
        assert_eq!(
            DualMaCrossover::default().min_history_bars(),
            DUAL_MA_DEFAULT_N2
        );
        assert_eq!(DUAL_MA_DEFAULT_N1, 50);
        assert_eq!(DUAL_MA_DEFAULT_N2, 200);
    }

    #[test]
    fn declared_parameters_render_n1_and_n2_as_bare_json_integers() {
        let params = DualMaCrossover::new(50, 200).declared_parameters();
        assert_eq!(params["N1"], "50");
        assert_eq!(params["N2"], "200");
        let params = DualMaCrossover::new(10, 37).declared_parameters();
        assert_eq!(params["N1"], "10");
        assert_eq!(params["N2"], "37");
    }

    #[test]
    fn schedule_policy_universe_and_id_are_the_documented_ones() {
        let r = DualMaCrossover::default();
        assert_eq!(r.id(), "dual_ma_crossover");
        assert_eq!(r.universe(), &["ASSET"]);
        assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(r.rebalance_policy(), RebalancePolicy::EveryBar);
    }

    // (a) Clean crossover: fast SMA (last N1=3 closes) strictly ABOVE the slow SMA (last N2=5 closes) -> weight
    // 1.0. Deliberately NOT a fixture where the fast triplet is all-high and the rest of the slow window is all
    // low (that shape survives a fast-window off-by-one: dragging in one more low bar still leaves the widened
    // fast mean above the slow mean). Instead the 4th-from-end close (10.0) is low enough, and the 5th-from-end
    // (90.0) high enough, that widening the fast window by one bar (an (N1+1)-wide window bug) pulls the 4th-from-
    // end bar in and drags the fast mean (77.5) BELOW the correctly-computed slow mean (80.0), flipping this
    // answer to 0.0 -- see mutant M23 below, which this test is what catches. The two bogus leading closes
    // (9999.0) sit outside even the slow window and must be excluded from both windows regardless.
    #[test]
    fn fast_above_slow_gives_full_weight() {
        let closes = [9999.0, 9999.0, 90.0, 10.0, 100.0, 100.0, 100.0];
        // Hand-computed: slow window = last 5 = [90,10,100,100,100]; mean = 400/5 = 80.0.
        // fast window = last 3 = [100,100,100]; mean = 300/3 = 100.0. 100.0 > 80.0.
        let w = dual_ma_weight(&closes, N1, N2).unwrap();
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (b) Fast SMA strictly BELOW the slow SMA -> weight 0.0.
    #[test]
    fn fast_below_slow_gives_zero_weight() {
        let closes = [9999.0, 9999.0, 100.0, 100.0, 100.0, 10.0, 10.0];
        // Hand-computed: slow window = last 5 = [100,100,100,10,10]; mean = 320/5 = 64.0.
        // fast window = last 3 = [100,10,10]; mean = 120/3 = 40.0. 40.0 < 64.0.
        let w = dual_ma_weight(&closes, N1, N2).unwrap();
        assert!(w.abs() < 1e-9, "expected weight 0.0, got {w}");
    }

    // (c) Exact tie: SMA(N1) == SMA(N2) -> weight 0.0 (the spec's `>`, not `>=`). Constructed so EVERY close in the
    // slow window (and so, trivially, every close in the fast window, a trailing subset of it) is exactly 50.0: 5
    // * 50.0 = 250.0 and 250.0 / 5.0 = 50.0 are both exact, as are 3 * 50.0 = 150.0 and 150.0 / 3.0 = 50.0, so both
    // means are the SAME binary double, 50.0 -- a hand-verifiable tie, not a "close enough" one. The two bogus
    // leading closes (1.0, 1.0) sit outside the slow window; including either in the slow window would pull its
    // mean below 50.0 and turn the tie into a false "above" signal.
    #[test]
    fn exact_tie_is_not_above() {
        let closes = [1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0];
        let slow = &closes[closes.len() - N2..];
        let fast = &closes[closes.len() - N1..];
        let slow_mean: f64 = slow.iter().sum::<f64>() / N2 as f64;
        let fast_mean: f64 = fast.iter().sum::<f64>() / N1 as f64;
        assert_eq!(
            slow_mean, 50.0,
            "fixture must be an EXACT tie by construction"
        );
        assert_eq!(
            fast_mean, slow_mean,
            "fixture must be an EXACT tie by construction"
        );
        let w = dual_ma_weight(&closes, N1, N2).unwrap();
        assert!(
            w.abs() < 1e-9,
            "a tie must be weight 0.0 (>, not >=), got {w}"
        );
    }

    // (d) Minimum valid history: exactly N2 closes visible (h.len() == min_history_bars()), no extra bars at all.
    // This is the earliest bar the simulator would ever call `target_weights` on, and the boundary where an
    // over-wide window bug would underflow `closes.len() - n` (panic) rather than merely compute the wrong answer.
    #[test]
    fn exactly_n2_bars_of_history_gives_a_defined_answer() {
        let closes = [10.0, 10.0, 100.0, 100.0, 100.0];
        assert_eq!(closes.len(), N2);
        // Hand-computed: slow mean = (10+10+100+100+100)/5 = 320/5 = 64.0; fast mean = (100+100+100)/3 = 100.0.
        let w = dual_ma_weight(&closes, N1, N2).unwrap();
        assert!((w - 1.0).abs() < 1e-9, "expected weight 1.0, got {w}");
    }

    // (e) More history than N2 does not change the answer: older bars beyond the slow window are correctly
    // ignored, however many of them there are.
    #[test]
    fn extra_history_beyond_n2_is_ignored() {
        let mut closes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12345.0];
        closes.extend_from_slice(&[10.0, 10.0, 100.0, 100.0, 100.0]); // same tail as the N2-boundary test above
        let w = dual_ma_weight(&closes, N1, N2).unwrap();
        assert!(
            (w - 1.0).abs() < 1e-9,
            "expected weight 1.0 (unaffected by older history), got {w}"
        );
    }

    #[test]
    fn default_constructor_uses_the_documented_defaults() {
        assert_eq!(
            DualMaCrossover::default(),
            DualMaCrossover::new(DUAL_MA_DEFAULT_N1, DUAL_MA_DEFAULT_N2)
        );
    }
}
