//! Cross-sectional momentum: `momentum_rank_weighted` (`weightsim::WeightRule`). Same universe/lookback convention
//! as the companion [`crate::top_n_winners`], but instead of top-N equal-weight, EVERY eligible asset gets a
//! positive weight proportional to its momentum RANK (linear rank weighting): the highest-momentum eligible asset
//! gets the largest weight, the lowest-momentum eligible asset gets the smallest (but still positive) weight, and
//! there is no momentum-based cutoff. Long only.
//!
//! Companion to [`crate::top_n_winners`]: same fixed-basket universe, same trailing-`L`-bar-return ranking, same
//! tie-break and insufficient-history-exclusion conventions, same typed-exclusion problem (`WeightRule::target_weights`
//! can only return a bare `Vec<f64>`, no room for a per-asset typed exclusion) solved the same way, via an additional
//! public method ([`MomentumRankWeighted::rank`]) rather than folding the exclusion into the weight vector.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Weighting direction* (the task spec own plain-English description of this rule contains an internal
//!    contradiction -- "1 = highest momentum, lowest weight" vs., two sentences later, "negative-momentum ones" get
//!    "a smaller weight" -- which only makes sense if HIGH momentum gets a LARGE weight. The second, unambiguous
//!    clause governs: highest momentum = highest weight, progressively lower (and negative) momentum =
//!    progressively smaller (but still positive) weight. This is also the only reading consistent with "linear rank
//!    weighting" as an actual technique. Implemented exactly as the formula below states; tested explicitly for
//!    DIRECTION (not just magnitude/sum), since this is the one place a sign mistake could slip through unnoticed.
//! 2. *Exact formula*: among the `K` eligible (non-excluded) assets, rank them `1..=K` by trailing `L`-bar return
//!    (`close_t / close_(t-L) - 1`) descending (rank 1 = highest return), ties broken by ascending alphabetical
//!    ticker order (choice 4). `weight(rank r) = (K + 1 - r) / (K * (K + 1) / 2)`. Rank 1 gets the largest weight
//!    (`K / (K*(K+1)/2)`), rank `K` gets the smallest positive weight (`1 / (K*(K+1)/2)`); the `K` weights sum to
//!    exactly 1.0 (triangular-number normalization). `K == 0` (every asset excluded) yields an all-zero weight
//!    vector -- no panic, no division by zero (mirrors [`crate::top_n_winners`] degenerate-case shape, which
//!    likewise emits an all-zero vector rather than a refusal when nothing is eligible).
//! 3. *"No exclusion" means no MOMENTUM-based cutoff only*: unlike `top_n_winners`, nothing is zeroed out purely for
//!    ranking low -- every eligible asset gets some positive weight, however small. It does NOT mean the
//!    insufficient-history exclusion goes away: an asset lacking `lookback + 1` closes is still a TYPED exclusion
//!    (same convention as `top_n_winners`, choice 7 there), entirely unrelated to the "no momentum cutoff" property.
//!    An excluded asset is removed from BOTH the ranking and the denominator `K` -- the remaining eligible assets
//!    weights are computed exactly as if the excluded one were never part of the universe.
//! 4. *Tie-break and exact-tie arithmetic*: identical to `top_n_winners` choices 1-2, reused verbatim because this
//!    rule computes the SAME trailing-return formula (one IEEE-754 division, one subtraction -- no summation, so no
//!    summation-order rounding artifact for a cross-multiply to guard against). Exact ties use plain `f64 ==`,
//!    broken by ascending alphabetical ticker order, producing a strict total order (no averaged/shared ranks).
//! 5. *Universe is a FIXED basket named at construction* (`Vec<&'static str>`), same as `top_n_winners`, not a
//!    generic single-asset placeholder.
//! 6. *`min_history_bars()` returns `lookback + 1`* -- the minimum a SINGLE asset needs to be ranked at all -- not a
//!    basket-wide gate (same reasoning as `top_n_winners` choice 4).
//! 7. *`decision_schedule()` is a constructor parameter*, not hardcoded, carried through from `top_n_winners`
//!    (the spec says "same universe/lookback" as that primitive, which made cadence configurable).
//! 8. *`rebalance_policy()` is [`weightsim::RebalancePolicy::OnDecision`]*, same reasoning as `top_n_winners` choice
//!    6: a periodically-rebalanced cross-sectional basket has a natural next-decision point to drift until.
//! 9. *No `n_winners`/N parameter*: every eligible asset participates, so unlike `top_n_winners` there is no winner
//!    count to configure. [`MomentumRankWeighted::new`] takes only a universe, a lookback and a schedule.
//! 10. *`excluded` is reported in universe order*, not sorted alphabetically -- same as `top_n_winners` choice 10;
//!     alphabetical order only governs the tie-break among assets that ARE ranked.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Default trailing lookback `L` (trading days): `close_t / close_(t-L) - 1`. Same default as the companion
/// `top_n_winners` primitive.
pub const MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK: usize = 126;

/// Version string recorded in every run (it is part of the series digest).
pub const MOMENTUM_RANK_WEIGHTED_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " momentum_rank_weighted"
);

/// Typed per-bar ranking result (interpretation choice 3). `weights` is aligned to `universe()` order and is
/// exactly what [`MomentumRankWeighted::target_weights`] returns; `excluded` additionally names which universe
/// slots (in universe order) lacked enough history to be ranked at all this bar -- a distinction `target_weights`
/// bare `Vec<f64>` cannot express.
#[derive(Clone, Debug, PartialEq)]
pub struct RankWeightedRanking {
    /// Target weight per universe slot: `(K + 1 - r) / (K * (K + 1) / 2)` for each eligible asset at rank `r` of
    /// `K`, `0.0` for every excluded asset. Every eligible asset weight is strictly positive.
    pub weights: Vec<f64>,
    /// Tickers excluded from ranking this bar for insufficient history (fewer than `lookback + 1` closes visible),
    /// in universe order.
    pub excluded: Vec<&'static str>,
}

/// `momentum_rank_weighted`: a fixed basket, ranked each decision by trailing `lookback`-bar return, every eligible
/// asset weighted proportional to its rank (highest momentum = highest weight), no momentum-based cutoff. See the
/// module doc for the interpretation choices (direction, formula, tie-break, exact-tie arithmetic, per-asset
/// exclusion, rebalance policy).
#[derive(Clone, Debug, PartialEq)]
pub struct MomentumRankWeighted {
    universe: Vec<&'static str>,
    lookback: usize,
    schedule: DecisionSchedule,
}

impl MomentumRankWeighted {
    /// A rule over `universe` (must be non-empty), ranking by trailing `lookback`-bar return (`lookback >= 1`),
    /// rebalanced on `schedule`. There is no winner count: every eligible asset participates at a rank-proportional
    /// weight.
    pub fn new(universe: Vec<&'static str>, lookback: usize, schedule: DecisionSchedule) -> Self {
        assert!(
            !universe.is_empty(),
            "momentum_rank_weighted: universe must be non-empty"
        );
        assert!(
            lookback >= 1,
            "momentum_rank_weighted: lookback L must be >= 1, got {lookback}"
        );
        MomentumRankWeighted {
            universe,
            lookback,
            schedule,
        }
    }

    /// Convenience constructor: `lookback` = [`MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK`] (126),
    /// `schedule` = [`DecisionSchedule::Daily`].
    pub fn with_defaults(universe: Vec<&'static str>) -> Self {
        MomentumRankWeighted::new(
            universe,
            MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK,
            DecisionSchedule::Daily,
        )
    }

    /// The configured trailing lookback `L`.
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// The typed ranking for this bar (interpretation choice 3): which universe slots are excluded for
    /// insufficient history, and the resulting rank-proportional weights. [`Self::target_weights`] is a thin
    /// wrapper returning only `.weights`.
    pub fn rank(&self, h: &HistoryView<'_>) -> RankWeightedRanking {
        let closes_by_asset: Vec<&[f64]> = (0..self.universe.len()).map(|i| h.closes(i)).collect();
        rank_weighted(&closes_by_asset, &self.universe, self.lookback)
    }
}

/// Pure computation, exposed for direct unit testing with synthetic per-asset close slices (no `HistoryView`/
/// `Panel` needed): `closes_by_asset[i]` is asset `i` causal close slice ending at the decision bar (the same
/// shape `HistoryView::closes(i)` returns), `universe[i]` its ticker. [`MomentumRankWeighted::rank`] is a one-line
/// wrapper around this over `h.closes(0..universe.len())`.
///
/// An asset with fewer than `lookback + 1` closes is EXCLUDED from ranking (interpretation choices 3/6), not ranked
/// last and not given return `0.0`; it is also removed from the denominator `K`. Eligible assets are ranked
/// descending by trailing `lookback`-bar return (`close_t / close_(t-lookback) - 1`), ties broken by ascending
/// alphabetical ticker (interpretation choices 1/4). With `K` eligible assets, the asset at rank `r` (1 = highest
/// return) gets weight `(K + 1 - r) / (K * (K + 1) / 2)`: rank 1 gets the largest weight, rank `K` the smallest
/// positive weight, and the `K` weights sum to exactly 1.0. Every other slot (excluded asset) is weight `0.0`.
/// `K == 0` yields an all-zero weight vector.
///
/// Panics if `closes_by_asset.len() != universe.len()` (the caller contract).
pub(crate) fn rank_weighted(
    closes_by_asset: &[&[f64]],
    universe: &[&'static str],
    lookback: usize,
) -> RankWeightedRanking {
    assert_eq!(
        closes_by_asset.len(),
        universe.len(),
        "momentum_rank_weighted: one close slice per universe asset"
    );
    let mut weights = vec![0.0; universe.len()];
    let mut excluded = Vec::new();
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for (i, &closes) in closes_by_asset.iter().enumerate() {
        if closes.len() < lookback + 1 {
            excluded.push(universe[i]);
            continue;
        }
        let last = *closes.last().expect("closes.len() >= lookback + 1 >= 2");
        let base = closes[closes.len() - 1 - lookback];
        let trailing_return = last / base - 1.0;
        candidates.push((i, trailing_return));
    }
    // Descending by return; exact ties broken by ascending alphabetical ticker (choices 1/4).
    candidates.sort_by(|&(ia, ra), &(ib, rb)| {
        rb.partial_cmp(&ra)
            .expect("trailing returns are finite: computed from finite positive closes")
            .then_with(|| universe[ia].cmp(universe[ib]))
    });
    let k = candidates.len();
    if k > 0 {
        let denom = (k * (k + 1) / 2) as f64;
        for (rank_minus_one, &(i, _)) in candidates.iter().enumerate() {
            let r = rank_minus_one + 1; // rank 1 = highest return
            weights[i] = (k + 1 - r) as f64 / denom;
        }
    }
    RankWeightedRanking { weights, excluded }
}

impl WeightRule for MomentumRankWeighted {
    fn id(&self) -> &'static str {
        "momentum_rank_weighted"
    }
    fn impl_version(&self) -> String {
        MOMENTUM_RANK_WEIGHTED_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &self.universe
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([("L", self.lookback.to_string())])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.schedule
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        self.lookback + 1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        Ok(self.rank(h).weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 3;

    fn slices<'a>(v: &'a [Vec<f64>]) -> Vec<&'a [f64]> {
        v.iter().map(|c| c.as_slice()).collect()
    }

    #[test]
    fn min_history_bars_is_lookback_plus_one() {
        assert_eq!(
            MomentumRankWeighted::new(vec!["A", "B"], L, DecisionSchedule::Daily)
                .min_history_bars(),
            L + 1
        );
        assert_eq!(
            MomentumRankWeighted::new(vec!["A", "B"], 37, DecisionSchedule::Daily)
                .min_history_bars(),
            38
        );
    }

    #[test]
    fn declared_parameters_render_l_as_a_bare_json_integer() {
        let r = MomentumRankWeighted::new(vec!["A", "B"], 126, DecisionSchedule::Daily);
        let params = r.declared_parameters();
        assert_eq!(params["L"], "126");
        assert_eq!(
            params.len(),
            1,
            "no N parameter: every eligible asset participates"
        );
    }

    #[test]
    fn schedule_is_a_constructor_parameter_and_policy_is_on_decision() {
        let daily = MomentumRankWeighted::new(vec!["A", "B"], L, DecisionSchedule::Daily);
        assert_eq!(daily.decision_schedule(), DecisionSchedule::Daily);
        let monthly =
            MomentumRankWeighted::new(vec!["A", "B"], L, DecisionSchedule::LastBarOfMonth);
        assert_eq!(
            monthly.decision_schedule(),
            DecisionSchedule::LastBarOfMonth
        );
        assert_eq!(monthly.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(monthly.id(), "momentum_rank_weighted");
    }

    // (a) Basic rank-weighting, no ties or exclusions: 5 assets, 4 bars each (exactly lookback + 1), clear trailing
    // returns, K = 5. Denominator = 5*6/2 = 15. DDD has the highest return (rank 1) -> weight 5/15; AAA rank 2 ->
    // 4/15; BBB rank 3 -> 3/15; CCC rank 4 -> 2/15; EEE (most negative, rank 5 = lowest) -> 1/15. This is the one
    // test that pins DIRECTION explicitly: the highest-momentum asset (DDD) must get the LARGEST weight, and the
    // lowest-momentum eligible asset (EEE, which is negative) must get the SMALLEST (but still positive) weight --
    // the exact opposite of a literal "1 = highest momentum, lowest weight" misreading.
    #[test]
    fn basic_rank_weighting_highest_momentum_gets_highest_weight() {
        let universe = vec!["AAA", "BBB", "CCC", "DDD", "EEE"];
        let closes = vec![
            vec![100.0, 110.0, 130.0, 150.0], // AAA: 150/100 - 1 = 0.5  -> rank 2
            vec![100.0, 105.0, 110.0, 120.0], // BBB: 120/100 - 1 = 0.2  -> rank 3
            vec![100.0, 95.0, 92.0, 90.0],    // CCC: 90/100 - 1 = -0.1  -> rank 4
            vec![100.0, 140.0, 170.0, 200.0], // DDD: 200/100 - 1 = 1.0  -> rank 1 (highest)
            vec![100.0, 90.0, 85.0, 80.0],    // EEE: 80/100 - 1 = -0.2  -> rank 5 (lowest)
        ];
        let r = rank_weighted(&slices(&closes), &universe, L);
        assert!(r.excluded.is_empty());
        assert_eq!(r.weights.len(), 5);
        let denom = 15.0_f64; // K=5 -> 5*6/2
        assert!(
            (r.weights[3] - 5.0 / denom).abs() < 1e-9,
            "DDD (highest momentum) should get the LARGEST weight, got {}",
            r.weights[3]
        );
        assert!(
            (r.weights[0] - 4.0 / denom).abs() < 1e-9,
            "AAA rank 2, got {}",
            r.weights[0]
        );
        assert!(
            (r.weights[1] - 3.0 / denom).abs() < 1e-9,
            "BBB rank 3, got {}",
            r.weights[1]
        );
        assert!(
            (r.weights[2] - 2.0 / denom).abs() < 1e-9,
            "CCC rank 4, got {}",
            r.weights[2]
        );
        assert!((r.weights[4] - 1.0 / denom).abs() < 1e-9, "EEE (lowest momentum, negative) should get the SMALLEST but still positive weight, got {}", r.weights[4]);
        assert!(
            r.weights[3] > r.weights[0],
            "DDD (rank 1) must outweigh AAA (rank 2)"
        );
        assert!(
            r.weights[0] > r.weights[1],
            "AAA (rank 2) must outweigh BBB (rank 3)"
        );
        assert!(
            r.weights[1] > r.weights[2],
            "BBB (rank 3) must outweigh CCC (rank 4)"
        );
        assert!(
            r.weights[2] > r.weights[4],
            "CCC (rank 4) must outweigh EEE (rank 5)"
        );
        for &w in &r.weights {
            assert!(w > 0.0, "every eligible asset, even negative-momentum ones, must get a strictly positive weight, got {w}");
        }
        assert!(
            (r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "weights must sum to exactly 1.0"
        );
    }

    // (b) One asset excluded for insufficient history: CCC has only L=3 bars (needs L+1=4), so it must be a TYPED
    // exclusion in `excluded`, NOT silently folded into the weights at 0.0, AND the remaining eligible assets (K=2)
    // weights must be computed as if CCC never existed (denominator 2*3/2 = 3, not 3*4/2 = 6).
    #[test]
    fn insufficient_history_is_a_typed_exclusion_and_shrinks_the_denominator() {
        let universe = vec!["AAA", "BBB", "CCC"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0], // AAA: 130/100 - 1 = 0.3 (4 bars, enough) -> rank 1 of K=2
            vec![100.0, 102.0, 106.0, 110.0], // BBB: 110/100 - 1 = 0.1 (4 bars, enough) -> rank 2 of K=2
            vec![100.0, 110.0, 120.0],        // CCC: only 3 bars, needs 4 -> excluded
        ];
        let r = rank_weighted(&slices(&closes), &universe, L);
        assert_eq!(r.excluded, vec!["CCC"], "CCC must be a typed exclusion");
        assert!(
            (r.weights[0] - 2.0 / 3.0).abs() < 1e-9,
            "AAA should get 2/3 (rank 1 of K=2), got {}",
            r.weights[0]
        );
        assert!(
            (r.weights[1] - 1.0 / 3.0).abs() < 1e-9,
            "BBB should get 1/3 (rank 2 of K=2), got {}",
            r.weights[1]
        );
        assert!(
            r.weights[2].abs() < 1e-9,
            "CCC excluded -> weight 0.0 in the plain vector too, got {}",
            r.weights[2]
        );
        assert!(
            (r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "fully invested among the eligible K=2, denominator must NOT count CCC"
        );
        assert!(!r.excluded.contains(&"AAA"));
        assert!(!r.excluded.contains(&"BBB"));
    }

    // (c) Exact tie broken by ascending alphabetical order. BBB and AAA have IDENTICAL (base, last) pairs, so their
    // computed returns are bit-identical by construction (same inputs, same formula), a hand-verifiable tie, not a
    // "close enough" one. With K=3 total, the alphabetically-earlier of the tied pair (AAA) must win rank 1 (the
    // larger weight), even though BBB appears first in the universe order.
    #[test]
    fn exact_tie_is_broken_by_ascending_alphabetical_ticker() {
        let universe = vec!["BBB", "AAA", "CCC"];
        let closes = vec![
            vec![100.0, 120.0, 140.0, 150.0], // BBB: 150/100 - 1 = 0.5
            vec![100.0, 120.0, 140.0, 150.0], // AAA: 150/100 - 1 = 0.5 (bit-identical to BBB's)
            vec![100.0, 95.0, 92.0, 90.0], // CCC: 90/100 - 1 = -0.1 (clearly lower, no interference)
        ];
        let ret = |c: &[f64]| c[3] / c[0] - 1.0;
        assert_eq!(
            ret(&closes[0]).to_bits(),
            ret(&closes[1]).to_bits(),
            "fixture must be an EXACT tie by construction"
        );
        let r = rank_weighted(&slices(&closes), &universe, L);
        assert!(r.excluded.is_empty());
        assert!(
            (r.weights[1] - 3.0 / 6.0).abs() < 1e-9,
            "AAA (ascending alphabetical) wins the tie -> rank 1, got {}",
            r.weights[1]
        );
        assert!(
            (r.weights[0] - 2.0 / 6.0).abs() < 1e-9,
            "BBB loses the tie-break -> rank 2, got {}",
            r.weights[0]
        );
        assert!(
            (r.weights[2] - 1.0 / 6.0).abs() < 1e-9,
            "CCC is clearly lowest -> rank 3, got {}",
            r.weights[2]
        );
        assert!(
            r.weights[1] > r.weights[0],
            "tie-break winner AAA must strictly outweigh BBB"
        );
    }

    // (d) Degenerate K=1 case: a single eligible asset must get weight exactly 1.0 (denominator 1*2/2 = 1).
    #[test]
    fn single_eligible_asset_gets_weight_exactly_one() {
        let universe = vec!["AAA", "BBB"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0], // AAA: 4 bars, eligible
            vec![100.0, 110.0],               // BBB: 2 bars, excluded (needs 4)
        ];
        let r = rank_weighted(&slices(&closes), &universe, L);
        assert_eq!(r.excluded, vec!["BBB"]);
        assert!(
            (r.weights[0] - 1.0).abs() < 1e-9,
            "sole eligible asset must get weight exactly 1.0, got {}",
            r.weights[0]
        );
        assert!(r.weights[1].abs() < 1e-9);
    }

    // (e) Degenerate K=0 case: every asset excluded -> all-zero weight vector, no panic, no division by zero.
    #[test]
    fn all_excluded_yields_all_zero_weights_no_panic() {
        let universe = vec!["AAA", "BBB", "CCC"];
        let closes = vec![
            vec![100.0, 110.0],      // AAA: 2 bars, excluded (needs 4)
            vec![100.0],             // BBB: 1 bar, excluded
            vec![100.0, 90.0, 80.0], // CCC: 3 bars, excluded (needs 4)
        ];
        let r = rank_weighted(&slices(&closes), &universe, L);
        let mut excluded_sorted = r.excluded.clone();
        excluded_sorted.sort();
        assert_eq!(excluded_sorted, vec!["AAA", "BBB", "CCC"]);
        assert_eq!(r.weights, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn target_weights_matches_rank_dot_weights() {
        // target_weights cannot be exercised directly without a HistoryView (pub(crate) constructor in weightsim),
        // but rank_weighted is exactly what both `rank` and `target_weights` delegate to; this test documents that
        // `target_weights` is a thin wrapper returning only `.weights`, nothing more, nothing less. K=2 eligible
        // assets -> denominator 2*3/2 = 3: AAA (rank 1) gets 2/3, BBB (rank 2) gets 1/3.
        let universe = vec!["AAA", "BBB"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0],
            vec![100.0, 102.0, 106.0, 110.0],
        ];
        let r = rank_weighted(&slices(&closes), &universe, L);
        assert!((r.weights[0] - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.weights[1] - 1.0 / 3.0).abs() < 1e-9);
    }
}
