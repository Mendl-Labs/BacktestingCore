//! Cross-sectional momentum: `top_n_winners` (`weightsim::WeightRule`). A FIXED basket of assets, rebalanced daily
//! or monthly (a constructor parameter). Each decision, rank every asset in the universe by its trailing `L`-bar
//! return (`close_t / close_(t-L) - 1`), hold the top `N` equal-weighted, zero weight on the rest. Long only.
//!
//! Companion to [`crate::sma_crossover_trend`]/[`crate::dual_ma_crossover`] (the two single-asset rules already in
//! this crate), but the first MULTI-ASSET (basket) rule in `reference-rules`: `universe()` names a fixed basket
//! given at construction, not a generic single-asset placeholder, and ranking requires comparing assets against
//! EACH OTHER rather than an asset against its own history.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Tie-break* (the one genuinely arbitrary convention choice in this primitive, per the spec): when two assets'
//!    trailing returns are EXACTLY equal, the tie is broken by ASCENDING alphabetical ticker order (the
//!    alphabetically earlier ticker ranks higher). Both sides of a review must pick the SAME convention here;
//!    there is no principled reason to prefer it over descending order, it is simply the one this implementation
//!    uses.
//! 2. *Exact-tie arithmetic*: the tie check is a plain `f64 ==` on the two computed returns, NOT a cross-multiply
//!    through [`crate::exact::compare_means`]/[`crate::exact::compare_to_mean`]. This is deliberate, not a
//!    shortcut: those helpers exist because a MEAN is a running SUM of many terms divided once, and the order in
//!    which binary floating point accumulates that sum can round differently from the true mean, turning a real
//!    tie into a false signal (see their doc comments and tests). A trailing return here is produced by exactly
//!    ONE IEEE-754 division (`close_t / close_(t-L)`) followed by exactly ONE subtraction (`- 1.0`), each a single
//!    correctly-rounded primitive operation, not an accumulation. IEEE-754 division returns the correctly-rounded
//!    value of the TRUE mathematical quotient of its two (exact, since both are already finite doubles) operands;
//!    if two asset's true quotients are mathematically equal, rounding each to the nearest double under the same
//!    rounding mode necessarily produces the SAME double, and subtracting `1.0` from two identical doubles again
//!    produces identical doubles. So a plain `==` on the computed returns is exactly as precise as a cross-multiply
//!    would be here -- there is no summation-order artifact for a cross-multiply to guard against, because there is
//!    no summation at all.
//! 3. *Universe is a FIXED basket named at construction* (`Vec<&'static str>`), unlike `sma_crossover_trend`'s and
//!    `dual_ma_crossover`'s generic single-asset `["ASSET"]` placeholder: a basket rule genuinely needs its
//!    universe's tickers and count at construction time, since the default `N` ("top 20%") depends on universe
//!    size.
//! 4. *`min_history_bars()` returns `lookback + 1`* -- the minimum closes a SINGLE asset needs to be ranked at all
//!    (today's close plus the close `lookback` bars back) -- NOT a basket-wide gate that blocks every decision
//!    until EVERY asset in the universe has that much history. An asset short of `lookback + 1` closes on a given
//!    bar is EXCLUDED from that bar's ranking (see choice 7); blocking the whole decision on the slowest-to-warm-up
//!    asset would defeat the point of a per-asset exclusion.
//! 5. *`decision_schedule()` is a constructor parameter*, not hardcoded: the spec explicitly calls out rebalance
//!    cadence ("daily or monthly") as configurable, unlike the two single-asset primitives, which are both daily
//!    only.
//! 6. *`rebalance_policy()` is [`weightsim::RebalancePolicy::OnDecision`]*, not `EveryBar` (the single-asset
//!    primitives' choice): this rule is a periodically-rebalanced cross-sectional BASKET -- the top-N target is set
//!    at a scheduled decision and units drift between decisions, rather than trading the whole book back to target
//!    on every bar between rebalances. `EveryBar` made sense for a DAILY trend rule with no natural "next decision"
//!    to drift until; a cross-sectional rebalance with an explicit schedule (daily or monthly) has exactly such a
//!    natural next decision, so `OnDecision` is the better fit. This is genuinely ambiguous and is this rule's own
//!    disclosed choice, not dictated by the spec.
//! 7. *Per-asset typed exclusion (design choice (a) from the task)*: `WeightRule::target_weights` can only return a
//!    bare `Result<Vec<f64>, RuleRefusal>` -- one `f64` per universe slot, or a single refusal for the WHOLE
//!    decision -- with no room to mark an individual slot as "excluded" rather than "ranked and given weight
//!    0.0". [`TopNWinners::rank`] is an ADDITIONAL public method (not part of `WeightRule`) that exposes the typed
//!    distinction: it returns a [`TopNRanking`] with both `weights` (aligned to `universe()` order, the same vector
//!    `target_weights` would return) and `excluded` (the tickers, in universe order, that lacked `lookback + 1`
//!    closes this bar and so were excluded from ranking entirely). `target_weights` is a one-line wrapper that
//!    calls `rank` and returns only `.weights`, so the simulator still sees a plain weight vector (0.0 for an
//!    excluded asset, indistinguishable from "ranked and lost" at that call site), while any caller who wants the
//!    typed distinction can get it from `rank` directly.
//! 8. *Default `N`* = `(universe.len() as f64 * 0.2).round() as usize`, floor 1 (see [`default_n_winners`]).
//!    [`TopNWinners`] has no blanket `Default` impl (it needs a universe, which has no sensible default); instead
//!    [`TopNWinners::with_defaults`] is a convenience constructor taking only the universe, using
//!    [`TOP_N_WINNERS_DEFAULT_LOOKBACK`] (126) and the computed default `N`, with [`DecisionSchedule::Daily`].
//! 9. *Fewer eligible (non-excluded) assets than `N`* (not addressed by the verbatim spec -- this is this rule's
//!    own disclosed choice): the actual winner count is `min(N, eligible.len())`, and each ACTUAL winner is weighted
//!    `1 / winner_count` -- i.e. the sleeve is renormalized to be fully invested among whichever eligible assets
//!    there are, rather than splitting a notional `1/N` per winner slot and leaving `(N - winner_count) / N` idle
//!    in cash. Nothing in this crate's conventions documents a cash-drag behavior for a basket rule, and leaving
//!    weight unaccounted-for with no stated cash semantics seemed worse than fully investing the eligible set.
//! 10. *`excluded` is reported in universe order*, not sorted alphabetically -- alphabetical order only governs the
//!     tie-break among assets that ARE ranked (choice 1); which slots are excluded is reported in the basket's own
//!     fixed order, the same order `weights` uses.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Default trailing lookback `L` (trading days): `close_t / close_(t-L) - 1`.
pub const TOP_N_WINNERS_DEFAULT_LOOKBACK: usize = 126;

/// Version string recorded in every run (it is part of the series digest).
pub const TOP_N_WINNERS_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " top_n_winners"
);

/// Default `N` (interpretation choice 8): top 20% of the universe, rounded, floor 1.
pub(crate) fn default_n_winners(universe_len: usize) -> usize {
    ((universe_len as f64 * 0.2).round() as usize).max(1)
}

/// Typed per-bar ranking result (interpretation choice 7). `weights` is aligned to `universe()` order and is
/// exactly what [`TopNWinners::target_weights`] returns; `excluded` additionally names which universe slots (in
/// universe order) lacked enough history to be ranked at all this bar -- a distinction `target_weights`'s bare
/// `Vec<f64>` cannot express.
#[derive(Clone, Debug, PartialEq)]
pub struct TopNRanking {
    /// Target weight per universe slot: `1 / winner_count` for each winner, `0.0` for every other ranked asset AND
    /// for every excluded asset.
    pub weights: Vec<f64>,
    /// Tickers excluded from ranking this bar for insufficient history (fewer than `lookback + 1` closes visible),
    /// in universe order.
    pub excluded: Vec<&'static str>,
}

/// `top_n_winners`: a fixed basket, ranked each decision by trailing `lookback`-bar return, top `n_winners`
/// equal-weighted, the rest at zero. See the module doc for the interpretation choices (tie-break, exact-tie
/// arithmetic, per-asset exclusion, rebalance policy, fewer-survivors-than-N).
#[derive(Clone, Debug, PartialEq)]
pub struct TopNWinners {
    universe: Vec<&'static str>,
    lookback: usize,
    n_winners: usize,
    schedule: DecisionSchedule,
}

impl TopNWinners {
    /// A rule over `universe` (must be non-empty), ranking by trailing `lookback`-bar return (`lookback >= 1`),
    /// holding the top `n_winners` (`n_winners >= 1`) equal-weighted, rebalanced on `schedule`.
    pub fn new(
        universe: Vec<&'static str>,
        lookback: usize,
        n_winners: usize,
        schedule: DecisionSchedule,
    ) -> Self {
        assert!(
            !universe.is_empty(),
            "top_n_winners: universe must be non-empty"
        );
        assert!(
            lookback >= 1,
            "top_n_winners: lookback L must be >= 1, got {lookback}"
        );
        assert!(
            n_winners >= 1,
            "top_n_winners: n_winners N must be >= 1, got {n_winners}"
        );
        TopNWinners {
            universe,
            lookback,
            n_winners,
            schedule,
        }
    }

    /// Convenience constructor (interpretation choice 8): `lookback` = [`TOP_N_WINNERS_DEFAULT_LOOKBACK`] (126),
    /// `n_winners` = top 20% of `universe` (rounded, floor 1), `schedule` = [`DecisionSchedule::Daily`].
    pub fn with_defaults(universe: Vec<&'static str>) -> Self {
        let n_winners = default_n_winners(universe.len());
        TopNWinners::new(
            universe,
            TOP_N_WINNERS_DEFAULT_LOOKBACK,
            n_winners,
            DecisionSchedule::Daily,
        )
    }

    /// The configured trailing lookback `L`.
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// The configured winner count `N`.
    pub fn n_winners(&self) -> usize {
        self.n_winners
    }

    /// The typed ranking for this bar (interpretation choice 7): which universe slots are excluded for
    /// insufficient history, and the resulting weights. [`Self::target_weights`] is a thin wrapper returning only
    /// `.weights`.
    pub fn rank(&self, h: &HistoryView<'_>) -> TopNRanking {
        let closes_by_asset: Vec<&[f64]> = (0..self.universe.len()).map(|i| h.closes(i)).collect();
        rank_top_n(
            &closes_by_asset,
            &self.universe,
            self.lookback,
            self.n_winners,
        )
    }
}

/// Pure computation, exposed for direct unit testing with synthetic per-asset close slices (no `HistoryView`/
/// `Panel` needed): `closes_by_asset[i]` is asset `i`'s causal close slice ending at the decision bar (the same
/// shape `HistoryView::closes(i)` returns), `universe[i]` its ticker. [`TopNWinners::rank`] is a one-line wrapper
/// around this over `h.closes(0..universe.len())`.
///
/// An asset with fewer than `lookback + 1` closes is EXCLUDED from ranking (interpretation choice 4/7), not ranked
/// last and not given return `0.0`. Eligible assets are ranked descending by trailing `lookback`-bar return
/// (`close_t / close_(t-lookback) - 1`), ties broken by ascending alphabetical ticker (interpretation choices 1-2).
/// The actual winner count is `min(n_winners, eligible.len())`, each winner weighted `1 / winner_count`
/// (interpretation choice 9); every other slot (non-winner ranked asset or excluded asset) is weight `0.0`.
///
/// Panics if `closes_by_asset.len() != universe.len()` (the caller's contract).
pub(crate) fn rank_top_n(
    closes_by_asset: &[&[f64]],
    universe: &[&'static str],
    lookback: usize,
    n_winners: usize,
) -> TopNRanking {
    assert_eq!(
        closes_by_asset.len(),
        universe.len(),
        "top_n_winners: one close slice per universe asset"
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
    // Descending by return; exact ties broken by ascending alphabetical ticker (choices 1-2).
    candidates.sort_by(|&(ia, ra), &(ib, rb)| {
        rb.partial_cmp(&ra)
            .expect("trailing returns are finite: computed from finite positive closes")
            .then_with(|| universe[ia].cmp(universe[ib]))
    });
    let winner_count = candidates.len().min(n_winners);
    if winner_count > 0 {
        let w = 1.0 / winner_count as f64;
        for &(i, _) in &candidates[..winner_count] {
            weights[i] = w;
        }
    }
    TopNRanking { weights, excluded }
}

impl WeightRule for TopNWinners {
    fn id(&self) -> &'static str {
        "top_n_winners"
    }
    fn impl_version(&self) -> String {
        TOP_N_WINNERS_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &self.universe
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("L", self.lookback.to_string()),
            ("N", self.n_winners.to_string()),
        ])
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
            TopNWinners::new(vec!["A", "B"], L, 1, DecisionSchedule::Daily).min_history_bars(),
            L + 1
        );
        assert_eq!(
            TopNWinners::new(vec!["A", "B"], 37, 1, DecisionSchedule::Daily).min_history_bars(),
            38
        );
    }

    #[test]
    fn declared_parameters_render_l_and_n_as_bare_json_integers() {
        let r = TopNWinners::new(vec!["A", "B"], 126, 5, DecisionSchedule::Daily);
        let params = r.declared_parameters();
        assert_eq!(params["L"], "126");
        assert_eq!(params["N"], "5");
    }

    #[test]
    fn schedule_is_a_constructor_parameter_and_policy_is_on_decision() {
        let daily = TopNWinners::new(vec!["A", "B"], L, 1, DecisionSchedule::Daily);
        assert_eq!(daily.decision_schedule(), DecisionSchedule::Daily);
        let monthly = TopNWinners::new(vec!["A", "B"], L, 1, DecisionSchedule::LastBarOfMonth);
        assert_eq!(
            monthly.decision_schedule(),
            DecisionSchedule::LastBarOfMonth
        );
        assert_eq!(monthly.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(monthly.id(), "top_n_winners");
    }

    // (a) Basic ranking, no ties or exclusions: 5 assets, 4 bars each (exactly lookback + 1), clear trailing
    // returns. DDD (+100%) and AAA (+50%) are the clear top 2; BBB (+20%), CCC (-10%) and EEE (-20%) are not.
    #[test]
    fn basic_ranking_picks_the_clear_top_n_equal_weighted() {
        let universe = vec!["AAA", "BBB", "CCC", "DDD", "EEE"];
        let closes = vec![
            vec![100.0, 110.0, 130.0, 150.0], // AAA: 150/100 - 1 = 0.5
            vec![100.0, 105.0, 110.0, 120.0], // BBB: 120/100 - 1 = 0.2
            vec![100.0, 95.0, 92.0, 90.0],    // CCC: 90/100 - 1 = -0.1
            vec![100.0, 140.0, 170.0, 200.0], // DDD: 200/100 - 1 = 1.0
            vec![100.0, 90.0, 85.0, 80.0],    // EEE: 80/100 - 1 = -0.2
        ];
        let r = rank_top_n(&slices(&closes), &universe, L, 2);
        assert!(r.excluded.is_empty());
        assert_eq!(r.weights.len(), 5);
        assert!(
            (r.weights[0] - 0.5).abs() < 1e-9,
            "AAA should win, got {}",
            r.weights[0]
        ); // AAA
        assert!(
            r.weights[1].abs() < 1e-9,
            "BBB should lose, got {}",
            r.weights[1]
        ); // BBB
        assert!(
            r.weights[2].abs() < 1e-9,
            "CCC should lose, got {}",
            r.weights[2]
        ); // CCC
        assert!(
            (r.weights[3] - 0.5).abs() < 1e-9,
            "DDD should win, got {}",
            r.weights[3]
        ); // DDD
        assert!(
            r.weights[4].abs() < 1e-9,
            "EEE should lose, got {}",
            r.weights[4]
        ); // EEE
        assert!((r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    // (b) One asset excluded for insufficient history: CCC has only L=3 bars (needs L+1=4), so it must be a TYPED
    // exclusion in `excluded`, NOT silently weight-0.0'd alongside a ranked-and-lost asset.
    #[test]
    fn insufficient_history_is_a_typed_exclusion_not_a_silent_zero() {
        let universe = vec!["AAA", "BBB", "CCC"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0], // AAA: 130/100 - 1 = 0.3 (4 bars, enough)
            vec![100.0, 102.0, 106.0, 110.0], // BBB: 110/100 - 1 = 0.1 (4 bars, enough)
            vec![100.0, 110.0, 120.0],        // CCC: only 3 bars, needs 4 -> excluded
        ];
        let r = rank_top_n(&slices(&closes), &universe, L, 1);
        assert_eq!(r.excluded, vec!["CCC"], "CCC must be a typed exclusion");
        assert!(
            (r.weights[0] - 1.0).abs() < 1e-9,
            "AAA should be the sole winner, got {}",
            r.weights[0]
        );
        assert!(
            r.weights[1].abs() < 1e-9,
            "BBB ranked and lost -> weight 0.0, got {}",
            r.weights[1]
        );
        assert!(
            r.weights[2].abs() < 1e-9,
            "CCC excluded -> weight 0.0 in the plain vector too, got {}",
            r.weights[2]
        );
        // The typed distinction lives ONLY in `excluded`: both BBB (ranked, lost) and CCC (excluded) show 0.0 in
        // `weights`, but only CCC appears in `excluded`.
        assert!(!r.excluded.contains(&"BBB"));
    }

    // (c) Exact tie broken by ascending alphabetical order. BBB and AAA have IDENTICAL (base, last) pairs, so their
    // computed returns are bit-identical by construction (same inputs, same formula), a hand-verifiable tie, not a
    // "close enough" one. With n_winners = 1, only the alphabetically-earlier of the tied pair (AAA) should win,
    // even though BBB appears first in the universe order.
    #[test]
    fn exact_tie_is_broken_by_ascending_alphabetical_ticker() {
        let universe = vec!["BBB", "AAA", "CCC"];
        let closes = vec![
            vec![100.0, 120.0, 140.0, 150.0], // BBB: 150/100 - 1 = 0.5
            vec![100.0, 120.0, 140.0, 150.0], // AAA: 150/100 - 1 = 0.5 (bit-identical to BBB's)
            vec![100.0, 95.0, 92.0, 90.0], // CCC: 90/100 - 1 = -0.1 (clearly lower, no interference)
        ];
        // Confirm the fixture really is an exact (bit-identical) tie before trusting the ranking outcome.
        let ret = |c: &[f64]| c[3] / c[0] - 1.0;
        assert_eq!(
            ret(&closes[0]).to_bits(),
            ret(&closes[1]).to_bits(),
            "fixture must be an EXACT tie by construction"
        );
        let r = rank_top_n(&slices(&closes), &universe, L, 1);
        assert!(r.excluded.is_empty());
        assert!(
            r.weights[0].abs() < 1e-9,
            "BBB loses the tie-break, got {}",
            r.weights[0]
        );
        assert!(
            (r.weights[1] - 1.0).abs() < 1e-9,
            "AAA (ascending alphabetical) wins the tie, got {}",
            r.weights[1]
        );
        assert!(
            r.weights[2].abs() < 1e-9,
            "CCC is clearly behind, got {}",
            r.weights[2]
        );
    }

    // (d) Default N is 20% of the universe, rounded, floor 1.
    #[test]
    fn default_n_is_twenty_percent_of_universe_floor_one() {
        assert_eq!(default_n_winners(10), 2);
        assert_eq!(default_n_winners(20), 4);
        assert_eq!(default_n_winners(7), 1); // round(1.4) = 1
        assert_eq!(default_n_winners(1), 1); // round(0.2) = 0, floored up to 1
        let r = TopNWinners::with_defaults(vec!["ONLY"]);
        assert_eq!(r.n_winners(), 1);
        assert_eq!(r.lookback(), TOP_N_WINNERS_DEFAULT_LOOKBACK);
        assert_eq!(TOP_N_WINNERS_DEFAULT_LOOKBACK, 126);
        assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
        let r10 =
            TopNWinners::with_defaults(vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]);
        assert_eq!(r10.n_winners(), 2);
    }

    // (e) Fewer eligible (non-excluded) assets than N: only AAA and BBB have enough history; CCC, DDD and EEE are
    // all excluded. With n_winners = 3 there are only 2 eligible assets, so BOTH win and are renormalized to
    // 1/2 each (interpretation choice 9), not left partially in cash at a notional 1/3 each.
    #[test]
    fn fewer_survivors_than_n_renormalizes_among_the_actual_winners() {
        let universe = vec!["AAA", "BBB", "CCC", "DDD", "EEE"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0], // AAA: 4 bars, eligible, 0.3
            vec![100.0, 102.0, 106.0, 110.0], // BBB: 4 bars, eligible, 0.1
            vec![100.0, 110.0],               // CCC: 2 bars, excluded
            vec![100.0],                      // DDD: 1 bar, excluded
            vec![100.0, 90.0, 80.0],          // EEE: 3 bars, excluded (needs 4)
        ];
        let r = rank_top_n(&slices(&closes), &universe, L, 3);
        let mut excluded_sorted = r.excluded.clone();
        excluded_sorted.sort();
        assert_eq!(excluded_sorted, vec!["CCC", "DDD", "EEE"]);
        assert!(
            (r.weights[0] - 0.5).abs() < 1e-9,
            "AAA should win with 1/2, got {}",
            r.weights[0]
        );
        assert!(
            (r.weights[1] - 0.5).abs() < 1e-9,
            "BBB should win with 1/2, got {}",
            r.weights[1]
        );
        assert!(
            (r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "fully invested among survivors"
        );
    }

    #[test]
    fn target_weights_matches_rank_dot_weights() {
        // target_weights cannot be exercised directly without a HistoryView (pub(crate) constructor in weightsim),
        // but rank_top_n is exactly what both `rank` and `target_weights` delegate to; this test documents that
        // `target_weights` is a thin wrapper returning only `.weights`, nothing more, nothing less.
        let universe = vec!["AAA", "BBB"];
        let closes = vec![
            vec![100.0, 110.0, 120.0, 130.0],
            vec![100.0, 102.0, 106.0, 110.0],
        ];
        let r = rank_top_n(&slices(&closes), &universe, L, 1);
        assert_eq!(r.weights, vec![1.0, 0.0]);
    }
}
