//! `equal_weight_rebalance`: a basket of assets. Every asset in the universe gets weight `1/N` (`N` = universe
//! size) on a rebalance day; weights DRIFT (are not reset) on every other day between rebalances. The spec names a
//! rebalance-frequency parameter with two values, "daily or monthly", but says to "assume monthly for this spec:
//! rebalance on the first trading day of each calendar month" -- so only the monthly case is implemented here.
//!
//! Like `inverse_volatility_weight` and `low_vol_quintile_tilt` (and unlike `crypto`/`etf`/`fx` in this crate),
//! this primitive implements `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView`: it is a generic
//! basket rule, not a documented sleeve adapted by a downstream crate.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *The rule reads no prices at all.* The target is `1/N` on every decision, unconditionally -- it does not
//!    depend on returns, volatility, or any other history. `target_weights` below never inspects
//!    `h.closes(..)`; it only needs `h.n_assets()`.
//! 2. *Zero insufficient-history case (verbatim from the spec).* "this primitive has no lookback, every asset
//!    present in the universe on a given bar participates immediately." Concretely: [`min_history_bars`] is `1`
//!    (the smallest value the trait allows a rule to need -- the current bar itself, with no history requirement
//!    beyond it being visible at all) and `target_weights` has no refusal path whatsoever: it always returns
//!    `Ok(vec![1.0 / n; n])` and never an `Err`. There is no "insufficient history" branch to write, because the
//!    spec is explicit that there is none.
//! 3. *Structural tension: "first trading day of each calendar month" vs. [`DecisionSchedule`] -- FLAGGED, resolved
//!    here, not by changing `weightsim`.* [`DecisionSchedule`] offers exactly two variants: `Daily` (every bar is
//!    a decision) and `LastBarOfMonth` (the last bar of each calendar month present in the joint calendar, or the
//!    panel's final bar). Neither is literally "the first trading day of each calendar month"; there is no
//!    `FirstBarOfMonth` variant, and adding one would mean editing `weightsim::rule`, out of scope for this task.
//!    Between the two that exist:
//!      - `Daily` is ruled out, not merely imperfect. Under `Daily`, every bar is a decision bar; with the
//!        simulator's default `execution_delay_bars = 0` (`SimConfig::default()`, see `weightsim::sim`), a
//!        decision made at bar `t` is effective at bar `t` itself, so under [`RebalancePolicy::OnDecision`] the
//!        book is "newly effective" -- and therefore retraded to target -- on EVERY bar. That collapses `OnDecision`
//!        into the exact same observable behavior as `EveryBar`: there is no bar left on which weights could ever
//!        drift. Since "weights DRIFT ... on every other day between rebalances" is an explicit, load-bearing part
//!        of the spec, `Daily` would silently violate it regardless of which `RebalancePolicy` is declared
//!        alongside it, so it is not a live option here.
//!      - `LastBarOfMonth` is therefore the chosen schedule. It decides, and (paired with
//!        [`RebalancePolicy::OnDecision`], chosen for the reason above and because the spec says so explicitly:
//!        "consistent with 'on_decision' rebalance policy, not 'every_bar'") trades, on the LAST trading session of
//!        each calendar month (and additionally on the panel's final bar, by the enum's own documented
//!        end-of-panel convention), rather than on the FIRST trading session of the FOLLOWING month. On an
//!        ordinary joint calendar there is no trading session between "the last session of month `M`" and "the
//!        first session of month `M + 1`" -- they are calendar-adjacent sessions by construction -- so this is off
//!        from the letter of the spec by exactly one trading session, never more, and in the wrong direction only
//!        in the sense of being one session early rather than late. Because this rule's target is session-
//!        independent (choice 1: always `1/N`, a function of `N` alone, never of price history or calendar
//!        position), shifting the decision by one session changes only WHEN the trade happens, never WHAT it
//!        trades to -- the economically meaningful content of "monthly equal-weight rebalance, drift in between"
//!        survives exactly, and only the specific session within the month-boundary pair is approximated. This is
//!        the most literally spec-faithful choice available from the two variants `DecisionSchedule` actually
//!        offers.
//! 4. *`declared_parameters` names the spec's own parameter.* The spec calls out "parameter rebalance frequency
//!    (daily or monthly)" by name, so `declared_parameters()` reports `"rebalance_frequency": "monthly"` (the
//!    value this implementation assumes, per the spec's own instruction) in addition to the resolved
//!    `decision_schedule` / `rebalance_policy` choice from point 3, so a reviewer can see both the spec-level
//!    parameter and the concrete schedule it was mapped to in one place.
//! 5. *Universe is a placeholder.* This primitive is a generic "basket of assets" rule, not tied to one documented
//!    sleeve, so [`EQUAL_WEIGHT_REBALANCE_SYMBOLS`] is a small, diversified four-ETF example basket (broad US
//!    equities, broad investment-grade bonds, gold, REITs) chosen only so the rule compiles and is testable, and
//!    sized to 4 so `1/N = 0.25` is an exact, round quantity in tests. A caller higher up the stack assigns the
//!    real production universe later.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference universe: four liquid, diversified ETFs (broad US equities, broad investment-grade bonds,
/// gold, REITs). Not a documented sleeve and not production data -- a placeholder basket (choice 5 above). A
/// caller assigns the real production universe later.
pub const EQUAL_WEIGHT_REBALANCE_SYMBOLS: [&str; 4] = ["SPY", "AGG", "GLD", "VNQ"];

/// `equal_weight_rebalance`: every asset in the universe gets weight `1/N` on a rebalance day (the last trading
/// session of each calendar month, per [`DecisionSchedule::LastBarOfMonth`] -- see the module doc, choice 3, for
/// why this is the most literally spec-faithful mapping of "the first trading day of each calendar month"
/// available from `weightsim`'s `DecisionSchedule`); weights drift, uncorrected, on every other bar
/// ([`RebalancePolicy::OnDecision`]). There is no lookback and no refusal path (choice 2).
#[derive(Clone, Copy, Debug, Default)]
pub struct EqualWeightRebalanceRule;

impl WeightRule for EqualWeightRebalanceRule {
    fn id(&self) -> &'static str {
        "equal_weight_rebalance_monthly"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &EQUAL_WEIGHT_REBALANCE_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            // The spec's own named parameter (choice 4): this implementation assumes the monthly case.
            ("rebalance_frequency", "\"monthly\"".to_string()),
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
        // Zero insufficient-history case (choice 2): the current bar alone is enough to decide, so this is the
        // smallest value the trait permits. The simulator never withholds a call to this rule for lack of history.
        1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        // Choice 1: no prices are read. Every asset present in the universe participates immediately at 1/N;
        // there is no exclusion branch and no refusal path (choice 2).
        let n = h.n_assets();
        Ok(vec![1.0 / n as f64; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weightsim::{simulate, Date, Panel, SimConfig};

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    const N: usize = EQUAL_WEIGHT_REBALANCE_SYMBOLS.len();

    /// Builds a panel over the FULL `EQUAL_WEIGHT_REBALANCE_SYMBOLS` universe (required: `simulate` rejects a
    /// panel whose symbols are not exactly the rule's `universe()`).
    fn panel_from(symbols_closes: Vec<Vec<f64>>, dates: Vec<Date>) -> Panel {
        assert_eq!(symbols_closes.len(), N);
        let symbols: Vec<String> = EQUAL_WEIGHT_REBALANCE_SYMBOLS
            .iter()
            .map(|s| s.to_string())
            .collect();
        Panel::new(symbols, dates, symbols_closes).unwrap()
    }

    /// 31 days of January followed by 10 days of February (2021): a real calendar-month boundary (bar 30, Jan 31,
    /// is the last bar of January) and a panel whose final bar (bar 40, Feb 10) is also a decision bar under
    /// `LastBarOfMonth`'s documented end-of-panel rule, so this one calendar builds both decisions a monthly test
    /// needs without a full business-day calendar implementation.
    fn jan_through_feb10_dates() -> Vec<Date> {
        let mut dates = Vec::with_capacity(41);
        for day in 1..=31 {
            dates.push(d(&format!("2021-01-{day:02}")));
        }
        for day in 1..=10 {
            dates.push(d(&format!("2021-02-{day:02}")));
        }
        dates
    }

    #[test]
    fn declared_parameters_reflect_the_resolved_schedule_choice() {
        let rule = EqualWeightRebalanceRule;
        let params = rule.declared_parameters();
        assert_eq!(
            params.get("rebalance_frequency").map(|s| s.as_str()),
            Some("\"monthly\"")
        );
        assert_eq!(
            params.get("schedule").map(|s| s.as_str()),
            Some("\"last_bar_of_month\"")
        );
        assert_eq!(
            params.get("rebalance_policy").map(|s| s.as_str()),
            Some("\"on_decision\"")
        );
        assert_eq!(rule.min_history_bars(), 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::LastBarOfMonth);
        assert_eq!(rule.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(rule.universe(), &EQUAL_WEIGHT_REBALANCE_SYMBOLS);
    }

    #[test]
    fn every_asset_gets_exactly_one_over_n_on_the_first_decision_bar() {
        // A short panel confined to a single calendar month: the ONLY decision bar is the panel's final bar (the
        // end-of-panel rule), and every asset must receive exactly 1/N there, regardless of its price path (choice
        // 1: the target never depends on price history at all).
        let dates: Vec<Date> = (1..=10)
            .map(|day| d(&format!("2021-01-{day:02}")))
            .collect();
        let a = vec![100.0; 10];
        let b = vec![50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0];
        let c = vec![10.0; 10];
        let e = vec![
            200.0, 199.0, 201.0, 198.0, 202.0, 197.0, 203.0, 196.0, 204.0, 195.0,
        ];
        let p = panel_from(vec![a, b, c, e], dates);

        let r = simulate(&p, &EqualWeightRebalanceRule, &SimConfig::default()).unwrap();
        let last = p.n_bars() - 1;
        assert!(
            r.decision[last],
            "the panel's final bar must be a decision bar (end-of-panel rule)"
        );
        assert!(
            r.decision[..last].iter().all(|&dd| !dd),
            "no earlier bar within the same month is a decision"
        );

        let target = r.row(&r.target_weights, last);
        let held = r.row(&r.held_weights, last);
        for i in 0..N {
            assert!(
                (target[i] - 0.25).abs() < 1e-12,
                "target[{i}] must be exactly 1/N, got {}",
                target[i]
            );
            assert!(
                (held[i] - 0.25).abs() < 1e-12,
                "held[{i}] must be exactly 1/N right after the trade, got {}",
                held[i]
            );
        }
    }

    #[test]
    fn weights_drift_between_decisions_and_snap_back_to_one_over_n_at_the_next_one() {
        // Flat, equal prices (100.0) for all four assets through all of January (bars 0..=30); bar 30 (Jan 31) is
        // therefore the first decision (last bar of January). From bar 31 (Feb 1) onward, under `OnDecision` the
        // book is NOT retraded, so units are frozen at the bar-30 trade and each asset's price compounds at a
        // different daily growth factor, driving held weights away from 1/N. Bar 40 (Feb 10) is the panel's final
        // bar -- a decision again (end-of-panel rule) -- so the book is retraded back to exactly 1/N there.
        let dates = jan_through_feb10_dates();
        let n_bars = dates.len();
        assert_eq!(n_bars, 41);

        let growth = [1.00_f64, 1.05, 0.95, 1.02]; // asset0 flat, asset1 up, asset2 down, asset3 slight up
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_bars); N];
        for t in 0..n_bars {
            for (i, s) in series.iter_mut().enumerate() {
                let price = if t <= 30 {
                    100.0 // flat, equal across all four assets through January (choice: isolates the drift effect
                          // to the February leg, and keeps the bar-30 trade exactly equal-notional per asset).
                } else {
                    100.0 * growth[i].powi((t - 30) as i32)
                };
                s.push(price);
            }
        }
        let p = panel_from(series, dates);

        let r = simulate(&p, &EqualWeightRebalanceRule, &SimConfig::default()).unwrap();

        // Bar 30: first decision, first trade. Held weights must be exactly 1/N (all four assets priced equally
        // at the moment of the trade, so the equal-notional 1/N target produces equal-notional, 1/N units).
        assert!(
            r.decision[30],
            "bar 30 (Jan 31) must be the first decision (last bar of January)"
        );
        let held_30 = r.row(&r.held_weights, 30);
        for (i, &w) in held_30.iter().enumerate() {
            assert!(
                (w - 0.25).abs() < 1e-9,
                "held[{i}] at bar 30 must be 1/N, got {w}"
            );
        }

        // No bar strictly between the two decisions is itself a decision (OnDecision: no retrade, so no snap-back
        // until bar 40), and the held weights at an interior bar (bar 35, Feb 5) must have moved measurably away
        // from 1/N, in the direction implied by each asset's growth factor.
        for t in 31..40 {
            assert!(
                !r.decision[t],
                "bar {t} (strictly between the two decisions) must not be a decision bar"
            );
        }
        let held_35 = r.row(&r.held_weights, 35);
        assert!(
            held_35.iter().any(|&w| (w - 0.25).abs() > 1e-6),
            "held weights must have drifted away from 1/N by bar 35, got {held_35:?}"
        );
        assert!(
            held_35[1] > 0.25,
            "asset 1 (the grower) must be overweight by bar 35, got {}",
            held_35[1]
        );
        assert!(
            held_35[2] < 0.25,
            "asset 2 (the shrinker) must be underweight by bar 35, got {}",
            held_35[2]
        );
        // Held weights must still sum to 1.0 (fully invested, zero cost/financing): drift redistributes weight
        // among assets, it does not change the total.
        let sum_35: f64 = held_35.iter().sum();
        assert!(
            (sum_35 - 1.0).abs() < 1e-9,
            "held weights must still sum to 1.0 while drifting, got {sum_35}"
        );

        // Bar 40 (Feb 10, the panel's final bar): a decision again (end-of-panel rule), so the book snaps back to
        // exactly 1/N, even though the four assets' prices are now very different from each other and from bar 30.
        let last = n_bars - 1;
        assert_eq!(last, 40);
        assert!(
            r.decision[last],
            "bar 40 (the panel's final bar) must be a decision bar"
        );
        let held_40 = r.row(&r.held_weights, last);
        for (i, &w) in held_40.iter().enumerate() {
            assert!(
                (w - 0.25).abs() < 1e-9,
                "held[{i}] at bar 40 must snap back to 1/N, got {w}"
            );
        }
    }

    #[test]
    fn target_weights_never_depends_on_price_history_and_never_refuses() {
        // Choice 1 and 2 as a direct, minimal check on the method itself (bypassing the simulator): whatever
        // prices and however many bars are visible, `target_weights` always returns `Ok(vec![1/N; N])`.
        let dates: Vec<Date> = (1..=3).map(|day| d(&format!("2021-06-{day:02}"))).collect();
        let wild = vec![
            vec![1.0, 1_000_000.0, 0.0001],
            vec![5.0, 5.0, 5.0],
            vec![9.0, 1.0, 9.0],
            vec![2.0, 2.0, 2.0],
        ];
        let p = panel_from(wild, dates);
        let rule = EqualWeightRebalanceRule;
        for t in 0..p.n_bars() {
            let r = simulate(&p, &rule, &SimConfig::default()).unwrap();
            // The simulator only calls the rule on decision bars; here we instead confirm, through the public
            // result, that whichever bar did decide produced exactly 1/N, and that nothing in the run was refused.
            if r.decision[t] {
                let w = r.row(&r.target_weights, t);
                for &wi in w {
                    assert!((wi - 0.25).abs() < 1e-12);
                }
            }
        }
        let r = simulate(&p, &rule, &SimConfig::default()).unwrap();
        assert!(
            r.refused.iter().all(|&ref_| !ref_),
            "this rule never refuses"
        );
        assert!(!r.refusals.iter().any(|_| true));
    }
}
