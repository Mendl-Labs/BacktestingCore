//! (b) Causality: future-poisoning test, truncation test, rule-level truncation, a deliberately leaky rule that the
//! harness MUST catch (proving the harness fails when it should), and a direct check of what a rule can see.

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use std::sync::Mutex;
use weightsim::harness::*;
use weightsim::*;

fn net_cfg() -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() }
}

#[test]
fn poisoning_all_bars_after_t_leaves_every_output_up_to_t_bit_identical_s3() {
    let panel = s3_panel();
    let make = |_: &Panel| S3TestRule;
    let mut checked = 0;
    for &t in &[100usize, 101, 150, 200, 250, 271] {
        for seed in [1u64, 2, 3] {
            let rep = check_poisoning(&make, &panel, &net_cfg(), t, seed).unwrap();
            assert!(
                rep.is_clean(),
                "S3 leaked from the future at T={t} seed={seed}: {:?}",
                &rep.mismatches[..rep.mismatches.len().min(5)]
            );
            assert_eq!(rep.compared_through_bar, t);
            checked += 1;
        }
    }
    assert_eq!(checked, 18);
}

#[test]
fn poisoning_all_bars_after_t_leaves_every_output_up_to_t_bit_identical_s1_on_decision() {
    let panel = s1_panel();
    let make = |_: &Panel| S1TestRule;
    for &t in &[220usize, 230, 260, 300, 350, 420] {
        let rep = check_poisoning(&make, &panel, &net_cfg(), t, 7).unwrap();
        assert!(rep.is_clean(), "S1 leaked from the future at T={t}: {:?}", rep.mismatches);
    }
}

#[test]
fn poisoning_actually_changes_the_future_so_the_test_is_not_vacuous() {
    let panel = s3_panel();
    let t = 150;
    let poisoned = poison_panel(&panel, t + 1, 1);
    assert_eq!(&poisoned.closes(0)[..=t], &panel.closes(0)[..=t]);
    assert!(poisoned.closes(0)[t + 1..].iter().zip(&panel.closes(0)[t + 1..]).all(|(a, b)| a != b));
    assert_eq!(poisoned.dates(), panel.dates(), "dates are an input and stay intact");
    // and the outputs AFTER t really do differ, so equality up to t is informative
    let a = simulate(&panel, &S3TestRule, &SimConfig::default()).unwrap();
    let b = simulate(&poisoned, &S3TestRule, &SimConfig::default()).unwrap();
    assert!(compare_prefix(&a, &b, panel.n_bars() - 1).iter().any(|m| m.bar > t));
    assert!(compare_prefix(&a, &b, t).is_empty());
}

#[test]
fn truncating_the_panel_leaves_all_earlier_outputs_bit_identical() {
    let make_s3 = |_: &Panel| S3TestRule;
    let make_s1 = |_: &Panel| S1TestRule;
    let p3 = s3_panel();
    for cut in [100usize, 130, 199, 250] {
        let rep = check_truncation(&make_s3, &p3, &net_cfg(), cut).unwrap();
        assert!(rep.is_clean(), "S3 truncation at {cut}: {:?}", rep.mismatches);
        assert_eq!(rep.compared_through_bar, cut, "Daily schedule: the cut bar is comparable");
    }
    let p1 = s1_panel();
    let dates = p1.dates();
    // a cut exactly on a month-end (comparable through the cut) and cuts in mid-month (compared strictly before it)
    let month_end = (250..dates.len() - 1).find(|&t| !dates[t].same_month(dates[t + 1])).unwrap();
    let rep = check_truncation(&make_s1, &p1, &net_cfg(), month_end).unwrap();
    assert!(rep.is_clean(), "{:?}", rep.mismatches);
    assert_eq!(rep.compared_through_bar, month_end);
    for cut in [month_end + 3, month_end + 9, 400] {
        let rep = check_truncation(&make_s1, &p1, &net_cfg(), cut).unwrap();
        assert!(rep.is_clean(), "S1 truncation at {cut}: {:?}", rep.mismatches);
        let full_month_end = !dates[cut].same_month(dates[cut + 1]);
        let expect = if full_month_end { cut } else { cut - 1 };
        assert_eq!(
            rep.compared_through_bar, expect,
            "mid-month cut: the truncated last bar is (by definition) a month-end"
        );
    }
}

#[test]
fn rule_level_truncation_agrees_at_every_sampled_decision_bar() {
    let make_s3 = |_: &Panel| S3TestRule;
    let p3 = s3_panel();
    let bars: Vec<usize> = (99..p3.n_bars()).step_by(7).collect();
    assert!(check_rule_truncation(&make_s3, &p3, &bars).is_empty());
    let make_s1 = |_: &Panel| S1TestRule;
    let p1 = s1_panel();
    let dates = p1.dates();
    let me: Vec<usize> = (0..dates.len() - 1).filter(|&t| !dates[t].same_month(dates[t + 1])).collect();
    assert!(check_rule_truncation(&make_s1, &p1, &me).is_empty());
}

/// A rule that illegitimately holds its own copy of the panel and reads the NEXT bar's close.
struct LeakyRule {
    future: Panel,
}

impl WeightRule for LeakyRule {
    fn id(&self) -> &'static str {
        "leaky"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &CRY
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let t = h.len() - 1;
        // peek at tomorrow through the captured copy: long if tomorrow closes higher
        let w = (0..h.n_assets())
            .map(|i| {
                let c = self.future.closes(i);
                if t + 1 < c.len() && c[t + 1] > c[t] {
                    0.5
                } else {
                    0.0
                }
            })
            .collect();
        Ok(w)
    }
}

#[test]
fn a_leaky_rule_is_caught_by_the_poisoning_harness() {
    let panel = s3_panel();
    let make = |p: &Panel| LeakyRule { future: p.clone() };
    let mut caught = 0;
    for &t in &[120usize, 150, 180, 210] {
        let rep = check_poisoning(&make, &panel, &SimConfig::default(), t, 5).unwrap();
        if !rep.is_clean() {
            caught += 1;
            assert!(
                rep.mismatches.iter().any(|m| m.bar == t),
                "the leak is at the last un-poisoned bar T={t}: {:?}",
                rep.mismatches
            );
        }
    }
    assert!(caught >= 3, "poisoning must expose a rule that reads bar t+1 (caught {caught}/4)");
}

#[test]
fn a_leaky_rule_is_caught_by_the_truncation_harness_and_the_rule_level_check() {
    let panel = s3_panel();
    let make = |p: &Panel| LeakyRule { future: p.clone() };
    // The leaky weight at the last bar of a cut panel is 0 (no tomorrow), but 0.5 in the full panel whenever
    // tomorrow is an up-move; so a cut is exposed with probability ~ 3/4. Many cuts, most must be caught.
    let cuts: Vec<usize> = (120..240).step_by(10).collect();
    let caught = cuts
        .iter()
        .filter(|&&c| !check_truncation(&make, &panel, &SimConfig::default(), c).unwrap().is_clean())
        .count();
    assert!(
        caught * 2 >= cuts.len(),
        "truncation must expose a rule that reads bar t+1 (caught {caught}/{})",
        cuts.len()
    );
    let bars: Vec<usize> = (100..200).collect();
    assert!(!check_rule_truncation(&make, &panel, &bars).is_empty());
}

#[test]
fn the_leaky_rule_also_beats_the_honest_key_which_proves_the_leak_is_real() {
    // Sanity: the leak is worth something (else the tests above could pass for the wrong reason).
    let panel = s3_panel();
    let leaky = simulate(&panel, &LeakyRule { future: panel.clone() }, &s3_config()).unwrap();
    let honest = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    assert!(leaky.metrics().unwrap().sharpe > honest.metrics().unwrap().sharpe + 2.0);
}

/// Records exactly what the simulator shows it.
struct SpyRule {
    seen: Mutex<Vec<(usize, Date, usize, usize)>>,
}

impl WeightRule for SpyRule {
    fn id(&self) -> &'static str {
        "spy"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &CRY
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        self.seen.lock().unwrap().push((h.len(), h.date(), h.closes(0).len(), h.dates().len()));
        Ok(vec![0.1, 0.1])
    }
}

#[test]
fn a_rule_sees_exactly_bars_0_to_t_and_nothing_more() {
    let panel = s3_panel();
    let spy = SpyRule { seen: Mutex::new(Vec::new()) };
    simulate(&panel, &spy, &SimConfig::default()).unwrap();
    let seen = spy.seen.lock().unwrap();
    assert_eq!(seen.len(), panel.n_bars());
    for (t, &(len, date, ncloses, ndates)) in seen.iter().enumerate() {
        assert_eq!(len, t + 1, "view length at decision bar {t}");
        assert_eq!(ncloses, t + 1);
        assert_eq!(ndates, t + 1);
        assert_eq!(date, panel.dates()[t], "the last visible date is the decision date");
    }
}

#[test]
fn date_only_lookahead_is_an_input_the_schedule_ignores_prices() {
    // Poisoning changes prices, not dates: the month-end schedule (and so every decision flag) must be identical for
    // the WHOLE run, including bars after T. This is the documented, intended date-only input (design 2.4(4)).
    let panel = s1_panel();
    let poisoned = poison_panel(&panel, 200, 9);
    let rule =
        FnRule::new(&ETF, DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision, 1, |_| Ok(vec![0.2; 5]));
    let a = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    let b = simulate(&poisoned, &rule, &SimConfig::default()).unwrap();
    assert_eq!(a.decision, b.decision);
    assert!(a.decision.iter().filter(|x| **x).count() > 15);
}
