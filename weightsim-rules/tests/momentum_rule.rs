//! Always-on tests of the `momentum_rank_weighted` sleeve (Amendment 14): the adapter is the published point, and each
//! of the five pre-registered Tier IV mutants (section 6, T4) is a wrong implementation that the ladder's own
//! comparison catches. The "key" here is the honest rule's own gross run on the crate's synthetic ETF prices (the real
//! key is the Python export on the evidence window, checked in the Engine), so what is proven is detection power: the
//! same `compare` the ladder uses fails every mutant and passes the honest rule.

mod common;

use common::s1_panel;
use weightsim::{simulate, CostModel, DecisionSchedule, Panel, RebalancePolicy, SimConfig, WeightRule};
use weightsim_rules::adapters::MomentumRankWeightedRule;
use weightsim_rules::ladder::checks::{compare, Comparison};
use weightsim_rules::ladder::fixtures::{KeyBar, SleeveKey};
use weightsim_rules::ladder::mutants::{run_mutant_on, Mutant, MutantSleeve, PeekMomentum};
use weightsim_rules::ladder::runner::{key_rows_for_mutants, rows_from_sim, run_gross};
use weightsim_rules::ladder::{FixtureSetId, Registry};

const ETF: [&str; 5] = ["SPY", "EFA", "IEF", "DBC", "VNQ"];

/// The honest rule's gross run over the synthetic panel as a key (net columns are copies: mutants read returns and
/// weights only).
fn synthetic_key(panel: &Panel) -> SleeveKey {
    assert!(panel.n_bars() > 400, "the synthetic panel is long enough for L = 126 plus a window");
    let start = panel.dates()[200];
    let end = panel.dates()[panel.n_bars() - 1];
    let cfg = SimConfig { start: Some(start), end: Some(end), cost: CostModel::ZERO, ..SimConfig::default() };
    let sim = simulate(panel, &MomentumRankWeightedRule::default(), &cfg).unwrap();
    let rows = rows_from_sim(&sim, start, end, false).unwrap();
    let w_target = rows.w_target.clone().unwrap();
    let w_held = rows.w_held.clone().unwrap();
    let bars = (0..rows.dates.len())
        .map(|i| KeyBar {
            date: rows.dates[i],
            ret_gross: rows.ret[i],
            ret_net: rows.ret[i],
            equity_gross: 1.0,
            equity_net: 1.0,
            cost: 0.0,
            turnover: 0.0,
            decision: true,
            w_target: w_target[i].clone(),
            w_held: w_held[i].clone(),
        })
        .collect();
    SleeveKey {
        code: "S4",
        rule_id: "momentum_rank_weighted",
        symbols: ETF.iter().map(|s| (*s).to_string()).collect(),
        bars,
        flips: 0,
    }
}

fn against_key(panel: &Panel, key: &SleeveKey, m: Mutant) -> Comparison {
    let run = run_mutant_on(panel, key, m).unwrap();
    compare(&key_rows_for_mutants(key), &run.rows).unwrap()
}

#[test]
fn the_adapter_is_the_published_point() {
    let r = MomentumRankWeightedRule::default();
    assert_eq!(r.id(), "momentum_rank_weighted");
    assert_eq!(r.universe(), &ETF[..]);
    assert_eq!(r.lookback(), 126);
    assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
    assert_eq!(r.rebalance_policy(), RebalancePolicy::OnDecision);
    assert_eq!(r.min_history_bars(), 127);
    assert_eq!(r.execution_lag_bars(), 0);
    assert_eq!(r.declared_parameters().get("L").map(String::as_str), Some("126"));
    assert!(r.impl_version().starts_with("weightsim-rules "));
}

#[test]
fn the_rule_ranks_best_to_largest_weight_and_the_weights_sum_to_one() {
    let panel = s1_panel();
    let cfg = SimConfig { start: Some(panel.dates()[200]), ..SimConfig::default() };
    let sim = simulate(&panel, &MomentumRankWeightedRule::default(), &cfg).unwrap();
    for t in 130..sim.n_bars() {
        let w = sim.row(&sim.target_weights, t);
        if w.iter().all(|x| *x == 0.0) {
            continue;
        }
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "bar {t}: weights sum to {}", w.iter().sum::<f64>());
        assert!(w.iter().all(|x| *x > 0.0), "bar {t}: every ranked instrument is held");
        // The decision at the close of bar t ranks on closes through t.
        let ret: Vec<f64> = (0..5).map(|a| panel.closes(a)[t] / panel.closes(a)[t - 126] - 1.0).collect();
        let best = (0..5).max_by(|a, b| ret[*a].partial_cmp(&ret[*b]).unwrap()).unwrap();
        let worst = (0..5).min_by(|a, b| ret[*a].partial_cmp(&ret[*b]).unwrap()).unwrap();
        assert!(w[best] > w[worst], "bar {t}: the highest-momentum instrument must carry the largest weight");
        assert!((w[best] - 5.0 / 15.0).abs() < 1e-12 && (w[worst] - 1.0 / 15.0).abs() < 1e-12, "bar {t}: K = 5");
    }
}

#[test]
fn the_honest_rule_passes_the_comparison_it_is_the_key_of() {
    let panel = s1_panel();
    let key = synthetic_key(&panel);
    let sim = run_gross(&panel, &MomentumRankWeightedRule::default(), &key, 0).unwrap();
    let rows = rows_from_sim(&sim, key.bars[0].date, key.bars[key.bars.len() - 1].date, false).unwrap();
    let cmp = compare(&key_rows_for_mutants(&key), &rows).unwrap();
    assert!(cmp.failed_tiers().is_empty(), "the control must pass: {:?}", cmp.failed_tiers());
}

/// Amendment 14 T4: every one of the five mutants must FAIL at least one of Tiers I-III on the window.
#[test]
fn every_momentum_mutant_is_killed_by_the_ladders_own_comparison() {
    let panel = s1_panel();
    let key = synthetic_key(&panel);
    assert_eq!(Mutant::MOMENTUM.len(), 5);
    for m in Mutant::MOMENTUM {
        assert_eq!(m.sleeve(), MutantSleeve::S4);
        let cmp = against_key(&panel, &key, m);
        assert!(!cmp.failed_tiers().is_empty(), "{} was NOT caught: {:?}", m.name(), cmp);
        // Every mutant here changes the returns, so Tier II (per-bar identity) catches each, whatever the data.
        assert!(!cmp.tier2_pass, "{}: Tier II must catch it (max|ret diff| {:e})", m.name(), cmp.max_abs_ret_diff);
    }
}

#[test]
fn the_weight_mutants_are_caught_by_tier_iii_and_the_timing_mutants_by_the_weights_too() {
    let panel = s1_panel();
    let key = synthetic_key(&panel);
    // Half sizing and equal weights change EVERY weight cell (rank weights are never all 0.5x or 1/K), so the
    // fraction of agreeing cells is far below the 0.98 floor.
    for m in [Mutant::S4HalfSizing, Mutant::S4EqualWeight] {
        let cmp = against_key(&panel, &key, m);
        let t3 = cmp.tier3.expect("a mutant with weights is compared on weights");
        assert!(t3.agreement < 0.98, "{}: Tier III agreement {}", m.name(), t3.agreement);
    }
    // A one-day-late book holds yesterday's weights: the weight cells differ on every bar where the ranking moved.
    let late = against_key(&panel, &key, Mutant::S4ExtraDelay);
    assert!(late.max_abs_w_target_diff.unwrap() > 1e-9);
    // The off-by-one lookback changes some bar's ranking, hence some weight.
    let off = against_key(&panel, &key, Mutant::S4LookbackOffByOne);
    assert!(off.max_abs_w_target_diff.unwrap() > 1e-9, "L = 125 must rank differently on some bar");
}

#[test]
fn the_lookahead_mutant_leaks_and_the_causality_harnesses_see_it() {
    let panel = s1_panel();
    let cfg = SimConfig::default();
    let honest =
        weightsim::harness::check_poisoning(&|_p: &Panel| MomentumRankWeightedRule::default(), &panel, &cfg, 300, 7)
            .unwrap();
    assert!(honest.is_clean(), "the honest rule reads no future bar");
    let leaky = weightsim::harness::check_poisoning(&|p: &Panel| PeekMomentum::new(p), &panel, &cfg, 300, 7).unwrap();
    assert!(!leaky.is_clean(), "poisoning the future must change the peeking rule's past");
    let samples: Vec<usize> = (200..260).collect();
    assert!(!weightsim::harness::check_rule_truncation(&|p: &Panel| PeekMomentum::new(p), &panel, &samples).is_empty());
    assert!(weightsim::harness::check_rule_truncation(
        &|_p: &Panel| MomentumRankWeightedRule::default(),
        &panel,
        &samples
    )
    .is_empty());
}

#[test]
fn the_momentum_rule_is_registered_in_its_own_set_and_not_in_the_legacy_one() {
    let reg = Registry::library();
    let e = reg.get("momentum_rank_weighted").unwrap();
    assert_eq!(e.set, FixtureSetId::MomentumEvidence);
    assert_eq!((e.key.sleeve_code, e.panel), ("S4", weightsim_rules::ladder::PanelSpec::Etf));
    assert_eq!(e.mutants.to_vec(), Mutant::MOMENTUM.to_vec());
    assert_eq!(Mutant::of_set(FixtureSetId::MomentumEvidence), &Mutant::MOMENTUM[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::Legacy), &Mutant::ALL[..]);
    assert!(reg.iter_set(FixtureSetId::Legacy).all(|r| r.id != "momentum_rank_weighted"));
    assert_eq!(reg.by_sleeve(MutantSleeve::S4).unwrap().id, "momentum_rank_weighted");
}
