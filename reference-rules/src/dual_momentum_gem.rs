//! Dual momentum GEM (Antonacci's "Global Equities Momentum", *Dual Momentum Investing*, 2014): a monthly,
//! two-stage, long-only rule over a fixed 3-asset universe (SPY, EFA, AGG). At each month-end:
//!   1. *Absolute momentum gate*: the trailing 12-month total return of SPY (the US equities proxy). If it is
//!      positive, go to stage 2; otherwise (zero or negative) hold AGG (the defensive bond proxy) at 100%.
//!   2. *Relative momentum choice*: compare SPY's trailing 12-month return against EFA's (the international
//!      equities proxy, MSCI EAFE). Hold whichever is higher, 100%, long only.
//! At any decision the sleeve holds exactly one of {SPY, EFA, AGG} at 100% weight -- never a blend, never bare
//! cash (AGG stands in for the "cash" leg of the published rule).
//!
//! Implements `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView`, like `momentum_rank_weighted` and
//! `low_vol_quintile_tilt` (unlike `etf`/`crypto`/`fx` in this crate, which are `Panel`/`PriceSeries` primitives
//! adapted by the downstream `weightsim-rules` crate). Per this crate's convention each primitive module is
//! self-contained: the month-end detection below is its own copy, not a reuse of `crate::months` (whose
//! `month_end_indices` operates on `chrono::NaiveDate`, not `weightsim::Date`).
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *"Trailing 12-month total return" is measured month-end to month-end*, not calendar-day to calendar-day: the
//!    current month-end close divided by the close exactly 12 month-ends earlier, minus 1. This needs 13 month-end
//!    closes in view (the current one plus 12 trailing) -- the same "N month-end marks, including the current
//!    one" counting convention `etf.rs`'s `ETF_SMA_MONTH_ENDS` already uses in this crate.
//! 2. *Month-end detection is self-contained*, reusing the exact predicate
//!    `weightsim::DecisionSchedule::LastBarOfMonth::is_decision_bar` already encodes (a bar is a month-end if it is
//!    the last bar in view, or the next bar is in a different calendar month), reimplemented locally rather than
//!    imported, since `is_decision_bar` takes the FULL panel calendar and this rule only ever sees the causal,
//!    truncated `HistoryView` slice (`0..=t`).
//! 3. *Absolute momentum gate uses "positive", not "above a T-bill rate"* -- the simpler, more commonly cited
//!    formulation of the rule (and the one named in this platform's `ROBUST_PREMIA` note); this rule does not
//!    model a separate risk-free/T-bill series. `return <= 0.0` (zero or negative) fails the gate.
//! 4. *AGG's own price history never enters the decision.* The published rule holds the bond proxy as the
//!    defensive leg purely because the equities gate failed, not because AGG itself showed any momentum; this
//!    implementation never reads AGG's closes to decide anything (only to confirm it is present in the universe).
//! 5. *Price-only, no total-return adjustment*: SPY/EFA/AGG closes are used as delivered (this crate's universal
//!    convention; see `etf.rs` choice 8) -- dividends are not added back, the same caveat as every other ETF
//!    primitive here.
//! 6. *Tie-break, EFA vs SPY in stage 2*: an EXACT tie (bit-identical trailing 12-month returns) is broken toward
//!    the alphabetically-earlier ticker ("EFA" < "SPY"), the same "ascending alphabetical ticker order" tie-break
//!    convention `momentum_rank_weighted` and `low_vol_quintile_tilt` already use in this crate. A genuine
//!    bit-identical tie is vanishingly unlikely with real market data; this only matters for a hand-constructed
//!    fixture.
//! 7. *Non-finite trailing return is a typed `RefusalKind::Data` refusal*, not a panic or a silent default --
//!    unreachable given `Panel`'s own finite-and-positive close invariant, but defended anyway (same discipline as
//!    `low_vol_quintile_tilt` choice 6).
//! 8. *`min_history_bars()` returns 1*, like `EtfTrendRule`: the real gate is this rule's own typed
//!    `InsufficientHistory`-shaped `Warmup` refusal (fewer than 13 month-ends visible), not the generic simulator
//!    pre-check -- the exact same reasoning `weightsim_rules::adapters::EtfTrendRule::min_history_bars`'s own doc
//!    comment gives.
//! 9. *Schedule and rebalance policy*: `DecisionSchedule::LastBarOfMonth` / `RebalancePolicy::OnDecision`, matching
//!    the compiled `etf_trend_faber` rule: the sleeve is re-decided at each month-end and units drift between
//!    decisions.
//! 10. *Universe order is fixed* `[SPY, EFA, AGG]` (`DUAL_MOMENTUM_GEM_SYMBOLS`); `target_weights` always returns a
//!     length-3 vector in that order, with exactly one entry equal to `1.0` and the rest `0.0`.

use std::collections::BTreeMap;

use weightsim::{Date, DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Fixed universe, in the exact order `target_weights` returns: SPY (US equities proxy), EFA (international
/// equities proxy, MSCI EAFE), AGG (US aggregate bond proxy, the defensive leg).
pub const DUAL_MOMENTUM_GEM_SYMBOLS: [&str; 3] = ["SPY", "EFA", "AGG"];

/// Trailing lookback, in months, of the absolute- and relative-momentum return (interpretation choice 1).
pub const DUAL_MOMENTUM_GEM_TRAILING_MONTHS: usize = 12;

/// Month-end closes needed in view: the trailing window plus the current month-end (interpretation choice 1).
pub const DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED: usize = DUAL_MOMENTUM_GEM_TRAILING_MONTHS + 1;

/// Version string recorded in every run (it is part of the series digest).
pub const DUAL_MOMENTUM_GEM_VERSION: &str = concat!(
    "reference-rules ",
    env!("CARGO_PKG_VERSION"),
    " dual_momentum_gem"
);

/// Indices of month-end bars within `dates` (interpretation choice 2): `t` is a month-end if it is the last index,
/// or the next bar falls in a different calendar month. Ascending order, same length convention as
/// `crate::months::month_end_indices` but reimplemented over `weightsim::Date` (self-contained per this crate's
/// per-module convention).
fn month_end_indices(dates: &[Date]) -> Vec<usize> {
    let n = dates.len();
    let mut idx = Vec::new();
    for t in 0..n {
        if t + 1 == n || !dates[t].same_month(dates[t + 1]) {
            idx.push(t);
        }
    }
    idx
}

/// Pure decision over raw close slices, exposed for direct unit testing without a `HistoryView`/`Panel`.
/// `spy_closes`/`efa_closes` and `dates` share the same causal length (bars `0..=t`, the same shape
/// `HistoryView::closes`/`HistoryView::dates` return). Returns the target weight of `[SPY, EFA, AGG]`, in that
/// order -- exactly one entry `1.0`, the rest `0.0` -- or a typed refusal.
///
/// AGG's own closes are never read (interpretation choice 4): the caller does not even need to pass them.
pub(crate) fn decide_gem(
    spy_closes: &[f64],
    efa_closes: &[f64],
    dates: &[Date],
) -> Result<[f64; 3], RuleRefusal> {
    let month_ends = month_end_indices(dates);
    if month_ends.len() < DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED {
        return Err(RuleRefusal::warmup(format!(
            "need {DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED} month-end closes (a {DUAL_MOMENTUM_GEM_TRAILING_MONTHS}-month \
             trailing window plus the current month-end), have {}",
            month_ends.len()
        )));
    }
    let window = &month_ends[month_ends.len() - DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED..];
    let now = window[DUAL_MOMENTUM_GEM_TRAILING_MONTHS];
    let then = window[0];

    let spy_return = spy_closes[now] / spy_closes[then] - 1.0;
    let efa_return = efa_closes[now] / efa_closes[then] - 1.0;
    if !spy_return.is_finite() || !efa_return.is_finite() {
        return Err(RuleRefusal::data(
            "non_finite_trailing_return",
            format!(
                "trailing {DUAL_MOMENTUM_GEM_TRAILING_MONTHS}-month return is non-finite (SPY {spy_return}, EFA \
                 {efa_return})"
            ),
        ));
    }

    // Stage 1: absolute momentum gate (interpretation choice 3).
    if spy_return <= 0.0 {
        return Ok([0.0, 0.0, 1.0]); // AGG
    }

    // Stage 2: relative momentum choice. An exact tie favours the alphabetically-earlier ticker, EFA
    // (interpretation choice 6).
    if efa_return >= spy_return {
        Ok([0.0, 1.0, 0.0]) // EFA
    } else {
        Ok([1.0, 0.0, 0.0]) // SPY
    }
}

/// `dual_momentum_gem`: Antonacci's GEM over SPY/EFA/AGG. See the module doc for every interpretation choice.
#[derive(Clone, Copy, Debug, Default)]
pub struct DualMomentumGemRule;

impl WeightRule for DualMomentumGemRule {
    fn id(&self) -> &'static str {
        "dual_momentum_gem"
    }
    fn impl_version(&self) -> String {
        DUAL_MOMENTUM_GEM_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &DUAL_MOMENTUM_GEM_SYMBOLS
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            (
                "trailing_months",
                DUAL_MOMENTUM_GEM_TRAILING_MONTHS.to_string(),
            ),
            (
                "month_ends_needed",
                DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED.to_string(),
            ),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::LastBarOfMonth
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        // Not enough month-ends is this rule's own `InsufficientHistory` (a Warmup refusal), not a silent skip --
        // same convention and reasoning as `weightsim_rules::adapters::EtfTrendRule` (interpretation choice 8).
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let spy = h.closes(0);
        let efa = h.closes(1);
        let w = decide_gem(spy, efa, h.dates())?;
        Ok(w.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weightsim::{simulate, Panel, SimConfig};

    fn d(y: i32, m: u8) -> Date {
        Date::new(y, m, 1).unwrap()
    }

    /// One bar per calendar month, ascending, starting January of `start_year`. Trivially satisfies
    /// `LastBarOfMonth` (every bar is, by construction, the only -- hence last -- bar of its month), so a fixture
    /// does not need a dense daily calendar to exercise month-end logic.
    fn monthly_dates(n: usize, start_year: i32) -> Vec<Date> {
        (0..n)
            .map(|i| {
                let year = start_year + (i as i32) / 12;
                let month = (i % 12) as u8 + 1;
                d(year, month)
            })
            .collect()
    }

    // ---- decide_gem: pure function tests --------------------------------------------------------------------

    #[test]
    fn negative_spy_absolute_momentum_holds_agg() {
        // SPY: 100 -> 90 over the trailing window (-10%, fails the absolute momentum gate). EFA and its own
        // trailing return are irrelevant once the gate fails; AGG's closes are never even read.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![95.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 100.0;
        *spy.last_mut().unwrap() = 90.0;
        let efa = vec![50.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        let w = decide_gem(&spy, &efa, &dates).expect("enough history");
        assert_eq!(
            w,
            [0.0, 0.0, 1.0],
            "negative SPY trailing momentum must hold AGG: {w:?}"
        );
    }

    #[test]
    fn zero_spy_absolute_momentum_holds_agg() {
        // Exactly 0% trailing return is "non-positive" (interpretation choice 3): the gate must still fail.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let spy = vec![100.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        let efa = vec![50.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        let w = decide_gem(&spy, &efa, &dates).expect("enough history");
        assert_eq!(
            w,
            [0.0, 0.0, 1.0],
            "exactly zero SPY trailing momentum must hold AGG (non-positive): {w:?}"
        );
    }

    #[test]
    fn positive_spy_higher_than_efa_holds_spy() {
        // SPY: 100 -> 130 (+30%); EFA: 50 -> 55 (+10%). SPY clears the absolute gate and beats EFA in stage 2.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![115.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 100.0;
        *spy.last_mut().unwrap() = 130.0;
        let mut efa = vec![52.5; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        efa[0] = 50.0;
        *efa.last_mut().unwrap() = 55.0;
        let w = decide_gem(&spy, &efa, &dates).expect("enough history");
        assert_eq!(
            w,
            [1.0, 0.0, 0.0],
            "SPY (+30%) must beat EFA (+10%) and be held: {w:?}"
        );
    }

    #[test]
    fn positive_spy_lower_than_efa_holds_efa() {
        // SPY: 100 -> 110 (+10%); EFA: 50 -> 65 (+30%). SPY clears the absolute gate but loses to EFA in stage 2.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![105.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 100.0;
        *spy.last_mut().unwrap() = 110.0;
        let mut efa = vec![57.5; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        efa[0] = 50.0;
        *efa.last_mut().unwrap() = 65.0;
        let w = decide_gem(&spy, &efa, &dates).expect("enough history");
        assert_eq!(
            w,
            [0.0, 1.0, 0.0],
            "EFA (+30%) must beat SPY (+10%) and be held: {w:?}"
        );
    }

    #[test]
    fn exact_tie_is_broken_toward_efa_the_alphabetically_earlier_ticker() {
        // SPY: 100 -> 130 (+30%); EFA: 50 -> 65 (+30%), a bit-identical tie by construction. Interpretation
        // choice 6: the tie must break to EFA ("EFA" < "SPY" ascending), not SPY.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![115.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 100.0;
        *spy.last_mut().unwrap() = 130.0;
        let mut efa = vec![57.5; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        efa[0] = 50.0;
        *efa.last_mut().unwrap() = 65.0;
        let spy_ret = spy.last().unwrap() / spy[0] - 1.0;
        let efa_ret = efa.last().unwrap() / efa[0] - 1.0;
        assert_eq!(
            spy_ret.to_bits(),
            efa_ret.to_bits(),
            "fixture must be an EXACT tie by construction"
        );
        let w = decide_gem(&spy, &efa, &dates).expect("enough history");
        assert_eq!(
            w,
            [0.0, 1.0, 0.0],
            "an exact tie must break to EFA (alphabetically earlier), not SPY: {w:?}"
        );
    }

    #[test]
    fn insufficient_month_ends_is_a_typed_warmup_refusal() {
        // One month-end short of the 13 needed (12 trailing + current).
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED - 1, 2020);
        let spy = vec![100.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED - 1];
        let efa = vec![50.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED - 1];
        let err =
            decide_gem(&spy, &efa, &dates).expect_err("one short of 13 month-ends must refuse");
        assert_eq!(err.kind, weightsim::RefusalKind::Warmup);
        assert_eq!(err.code, "insufficient_history");
    }

    #[test]
    fn non_finite_trailing_return_is_a_typed_data_refusal() {
        // A zero `then` price (never produced by a real `Panel`, which requires every close finite and > 0) drives
        // the computed return to +-infinity; interpretation choice 7 requires this to be a typed `Data` refusal,
        // not a panic or a silently wrong decision.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![100.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 0.0;
        let efa = vec![50.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        let err = decide_gem(&spy, &efa, &dates).expect_err("a non-finite return must refuse");
        assert_eq!(err.kind, weightsim::RefusalKind::Data);
        assert_eq!(err.code, "non_finite_trailing_return");
    }

    // ---- WeightRule: static facts and end-to-end wiring -----------------------------------------------------

    #[test]
    fn declared_facts_match_the_module_doc() {
        let rule = DualMomentumGemRule;
        assert_eq!(rule.id(), "dual_momentum_gem");
        assert_eq!(rule.universe(), &["SPY", "EFA", "AGG"]);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::LastBarOfMonth);
        assert_eq!(rule.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(rule.min_history_bars(), 1);
        let params = rule.declared_parameters();
        assert_eq!(params["trailing_months"], "12");
        assert_eq!(params["month_ends_needed"], "13");
    }

    #[test]
    fn target_weights_matches_decide_gem_through_a_real_panel_and_simulate() {
        // Same scenario as `positive_spy_higher_than_efa_holds_spy`, but driven through the public `WeightRule`/
        // `simulate` API over a real `Panel`, confirming the trait wiring (universe order, HistoryView slicing)
        // end to end, not just the pure helper.
        let dates = monthly_dates(DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED, 2020);
        let mut spy = vec![115.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        spy[0] = 100.0;
        *spy.last_mut().unwrap() = 130.0;
        let mut efa = vec![52.5; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];
        efa[0] = 50.0;
        *efa.last_mut().unwrap() = 55.0;
        let agg = vec![20.0; DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED];

        let symbols: Vec<String> = DUAL_MOMENTUM_GEM_SYMBOLS
            .iter()
            .map(|s| s.to_string())
            .collect();
        let panel = Panel::new(symbols, dates, vec![spy, efa, agg]).unwrap();
        let r = simulate(&panel, &DualMomentumGemRule, &SimConfig::default()).unwrap();
        let w = r.row(&r.target_weights, panel.n_bars() - 1).to_vec();
        assert_eq!(
            w,
            vec![1.0, 0.0, 0.0],
            "SPY must be held on the final decision bar: {w:?}"
        );
    }

    #[test]
    fn insufficient_history_through_simulate_is_a_tolerated_warmup_not_an_abort() {
        // One month-end short of 13: `min_history_bars() == 1` means the simulator calls the rule on every
        // (monthly) decision bar, and the rule's own Warmup refusal must be tolerated under `OnRefusal::Abort`
        // until the first successful decision (never happens here), not escalate to a hard simulation error.
        let n = DUAL_MOMENTUM_GEM_MONTH_ENDS_NEEDED - 1;
        let dates = monthly_dates(n, 2020);
        let spy = vec![100.0; n];
        let efa = vec![50.0; n];
        let agg = vec![20.0; n];
        let symbols: Vec<String> = DUAL_MOMENTUM_GEM_SYMBOLS
            .iter()
            .map(|s| s.to_string())
            .collect();
        let panel = Panel::new(symbols, dates, vec![spy, efa, agg]).unwrap();
        let r = simulate(&panel, &DualMomentumGemRule, &SimConfig::default()).unwrap();
        assert!(
            r.decision.iter().all(|&dec| !dec),
            "no bar ever has the required 13 month-ends"
        );
        assert!(
            r.refused.iter().any(|&ref_| ref_),
            "each monthly bar must be attempted and tolerated as Warmup"
        );
    }
}
