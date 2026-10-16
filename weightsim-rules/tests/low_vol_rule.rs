//! Always-on tests of the `low_vol_quintile_tilt_20d` sleeve (Amendment 15): the adapter is the published point, and
//! each pre-registered Tier IV mutant (section 6, T4) that some tier CAN fail is a wrong implementation that the
//! ladder's own comparison catches. The "key" here is the honest rule's own gross run on a synthetic seven-asset panel
//! (the real key is the Python export on the evidence window, checked in the Engine), so what is proven is detection
//! power: the same `compare` the ladder uses fails every catchable mutant and passes the honest rule.
//!
//! The fifth pre-registered mutant, `lowvol_ddof1_not_ddof0`, is PROVABLY EQUIVALENT to the rule (sample stdev is the
//! population stdev times the same constant for every instrument, so the ranking is unchanged): a test below shows no
//! tier can fail it. That is a finding about the amendment, not a tier to loosen.

use weightsim::{simulate, CostModel, DecisionSchedule, Panel, RebalancePolicy, SimConfig, WeightRule};
use weightsim_rules::adapters::LowVolQuintileTiltAdapter;
use weightsim_rules::ladder::checks::{compare, Comparison};
use weightsim_rules::ladder::fixtures::{KeyBar, SleeveKey};
use weightsim_rules::ladder::mutants::{run_mutant_on, Mutant, MutantSleeve, PeekLowVol};
use weightsim_rules::ladder::runner::{key_rows_for_mutants, rows_from_sim, run_gross};
use weightsim_rules::ladder::{FixtureSetId, PanelSpec, Registry};

const LOWVOL: [&str; 7] = ["DIA", "EEM", "EFA", "GLD", "IWM", "SPY", "TLT"];
const N_BARS: usize = 700;

/// A deterministic seven-asset panel whose volatilities rotate, so the lowest-volatility instrument changes over time.
fn lowvol_panel() -> Panel {
    use chrono::{Datelike, Duration, NaiveDate, Weekday};
    let mut dates = Vec::with_capacity(N_BARS);
    let mut day = NaiveDate::from_ymd_opt(2018, 1, 1).unwrap();
    while dates.len() < N_BARS {
        if !matches!(day.weekday(), Weekday::Sat | Weekday::Sun) {
            dates.push(day);
        }
        day += Duration::days(1);
    }
    let mut csv = String::from("symbol,date_utc,close\n");
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for (a, sym) in LOWVOL.iter().enumerate() {
        let mut close = 50.0 + 10.0 * a as f64;
        for (t, d) in dates.iter().enumerate() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = ((state >> 33) as f64) / 2147483648.0 * 2.0 - 1.0;
            let vol = 0.004 + 0.012 * (0.5 + 0.5 * (t as f64 / 37.0 + a as f64 * 0.9).sin());
            close *= 1.0 + vol * z;
            csv.push_str(&format!("{sym},{d},{close:.6}\n"));
        }
    }
    Panel::from_long_csv(&csv, &LOWVOL).unwrap()
}

/// The honest rule's gross run over the synthetic panel as a key (net columns are copies: mutants read returns and
/// weights only).
fn synthetic_key(panel: &Panel) -> SleeveKey {
    let start = panel.dates()[100];
    let end = panel.dates()[panel.n_bars() - 1];
    let cfg = SimConfig { start: Some(start), end: Some(end), cost: CostModel::ZERO, ..SimConfig::default() };
    let sim = simulate(panel, &LowVolQuintileTiltAdapter::default(), &cfg).unwrap();
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
        code: "S5",
        rule_id: "low_vol_quintile_tilt_20d",
        symbols: LOWVOL.iter().map(|s| (*s).to_string()).collect(),
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
    let r = LowVolQuintileTiltAdapter::default();
    assert_eq!(r.id(), "low_vol_quintile_tilt_20d");
    assert_eq!(r.universe(), &LOWVOL[..]);
    assert_eq!(r.decision_schedule(), DecisionSchedule::Daily);
    assert_eq!(r.rebalance_policy(), RebalancePolicy::EveryBar);
    assert_eq!(r.min_history_bars(), 21);
    assert_eq!(r.execution_lag_bars(), 0);
    // The five parameters the library entry pins, as JSON-shaped literals (what `binding` parses back and compares).
    let p = r.declared_parameters();
    assert_eq!(p.len(), 5, "{p:?}");
    assert_eq!(p["lookback_days"], "20");
    assert_eq!(p["quintile_fraction"].parse::<f64>().unwrap(), 0.2);
    assert_eq!(p["schedule"], "\"daily\"");
    assert_eq!(p["rebalance_policy"], "\"every_bar\"");
    assert_eq!(p["max_weight_per_instrument"], "1.0", "a float literal, as the entry declares 1.0");
    assert!(r.impl_version().starts_with("weightsim-rules "));
}

/// The population stdev (ddof = 0) of the trailing 20 simple returns, straight from the definition.
fn pop_std(c: &[f64], t: usize) -> f64 {
    let r: Vec<f64> = (t - 19..=t).map(|k| c[k] / c[k - 1] - 1.0).collect();
    let m = r.iter().sum::<f64>() / 20.0;
    (r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 20.0).sqrt()
}

#[test]
fn the_rule_holds_exactly_the_lowest_volatility_instrument_at_full_weight() {
    let panel = lowvol_panel();
    let cfg = SimConfig { start: Some(panel.dates()[100]), ..SimConfig::default() };
    let sim = simulate(&panel, &LowVolQuintileTiltAdapter::default(), &cfg).unwrap();
    let mut held_set = std::collections::BTreeSet::new();
    for t in 100..sim.n_bars() {
        let w = sim.row(&sim.target_weights, t);
        let vols: Vec<f64> = (0..7).map(|a| pop_std(panel.closes(a), t)).collect();
        let lowest = (0..7).min_by(|a, b| vols[*a].partial_cmp(&vols[*b]).unwrap()).unwrap();
        assert_eq!(w.iter().filter(|x| **x > 0.0).count(), 1, "bar {t}: round(0.2 * 7) = 1 holding");
        assert!((w[lowest] - 1.0).abs() < 1e-15, "bar {t}: the lowest-volatility instrument carries 100%");
        held_set.insert(lowest);
    }
    assert!(held_set.len() >= 3, "the synthetic panel must rotate the lowest-volatility instrument: {held_set:?}");
}

#[test]
fn the_honest_rule_passes_the_comparison_it_is_the_key_of() {
    let panel = lowvol_panel();
    let key = synthetic_key(&panel);
    let sim = run_gross(&panel, &LowVolQuintileTiltAdapter::default(), &key, 0).unwrap();
    let rows = rows_from_sim(&sim, key.bars[0].date, key.bars[key.bars.len() - 1].date, false).unwrap();
    let cmp = compare(&key_rows_for_mutants(&key), &rows).unwrap();
    assert!(cmp.failed_tiers().is_empty(), "the control must pass: {:?}", cmp.failed_tiers());
}

/// Amendment 15 T4: every mutant that some tier can fail must FAIL at least one of Tiers I-III on the window.
#[test]
fn every_catchable_lowvol_mutant_is_killed_by_the_ladders_own_comparison() {
    let panel = lowvol_panel();
    let key = synthetic_key(&panel);
    assert_eq!(Mutant::LOWVOL.len(), 4);
    for m in Mutant::LOWVOL {
        assert_eq!(m.sleeve(), MutantSleeve::S5);
        let cmp = against_key(&panel, &key, m);
        assert!(!cmp.failed_tiers().is_empty(), "{} was NOT caught: {:?}", m.name(), cmp);
        // Every one of these changes the returns on some bar, so Tier II (per-bar identity) catches each.
        assert!(!cmp.tier2_pass, "{}: Tier II must catch it (max|ret diff| {:e})", m.name(), cmp.max_abs_ret_diff);
    }
}

#[test]
fn the_direction_and_timing_mutants_move_the_held_instrument() {
    let panel = lowvol_panel();
    let key = synthetic_key(&panel);
    // Holding the HIGHEST-volatility quintile instead of the lowest changes the selection on (almost) every bar.
    let top = against_key(&panel, &key, Mutant::S5TopQuintileNotBottom);
    let t3 = top.tier3.expect("a mutant with weights is compared on weights");
    assert!(t3.agreement < 0.5, "top-quintile mutant: Tier III agreement {}", t3.agreement);
    // A one-day-late book holds yesterday's selection: the weight cells differ wherever the selection moved.
    let late = against_key(&panel, &key, Mutant::S5ExtraDelay);
    assert!(late.max_abs_w_target_diff.unwrap() > 1e-9);
    // The 19-day window changes the ranking on some bar, hence the selection.
    let off = against_key(&panel, &key, Mutant::S5LookbackOffByOne);
    assert!(off.max_abs_w_target_diff.unwrap() > 1e-9, "a 19-day window must rank differently on some bar");
}

/// The fifth pre-registered mutant cannot be caught by any tier: it selects the same instrument on every bar.
#[test]
fn the_ddof1_mutant_is_equivalent_to_the_rule_and_no_tier_can_fail_it() {
    let panel = lowvol_panel();
    let key = synthetic_key(&panel);
    assert_eq!(Mutant::LOWVOL_EQUIVALENT, [Mutant::S5Ddof1]);
    assert_eq!(Mutant::S5Ddof1.sleeve(), MutantSleeve::S5);
    assert!(!Mutant::LOWVOL.contains(&Mutant::S5Ddof1));
    let cmp = against_key(&panel, &key, Mutant::S5Ddof1);
    assert!(
        cmp.failed_tiers().is_empty(),
        "if a tier fails it, the equivalence argument is wrong: {:?}",
        cmp.failed_tiers()
    );
    assert_eq!(cmp.max_abs_w_target_diff, Some(0.0), "the weights are bit-identical");
    assert_eq!(cmp.max_abs_ret_diff, 0.0, "so are the returns");
}

#[test]
fn the_lookahead_mutant_leaks_and_the_causality_harnesses_see_it() {
    let panel = lowvol_panel();
    let cfg = SimConfig::default();
    let honest =
        weightsim::harness::check_poisoning(&|_p: &Panel| LowVolQuintileTiltAdapter::default(), &panel, &cfg, 300, 7)
            .unwrap();
    assert!(honest.is_clean(), "the honest rule reads no future bar");
    let leaky = weightsim::harness::check_poisoning(&|p: &Panel| PeekLowVol::new(p), &panel, &cfg, 300, 7).unwrap();
    assert!(!leaky.is_clean(), "poisoning the future must change the peeking rule's past");
    let samples: Vec<usize> = (200..260).collect();
    assert!(!weightsim::harness::check_rule_truncation(&|p: &Panel| PeekLowVol::new(p), &panel, &samples).is_empty());
    assert!(weightsim::harness::check_rule_truncation(
        &|_p: &Panel| LowVolQuintileTiltAdapter::default(),
        &panel,
        &samples
    )
    .is_empty());
}

#[test]
fn the_low_vol_rule_is_registered_in_its_own_set_and_not_in_the_others() {
    let reg = Registry::library();
    let e = reg.get("low_vol_quintile_tilt_20d").unwrap();
    assert_eq!(e.set, FixtureSetId::LowVolEvidence);
    assert_eq!((e.key.sleeve_code, e.panel), ("S5", PanelSpec::LowVol));
    assert_eq!(e.mutants.to_vec(), Mutant::LOWVOL.to_vec());
    assert_eq!(Mutant::of_set(FixtureSetId::LowVolEvidence), &Mutant::LOWVOL[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::Legacy), &Mutant::ALL[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::MomentumEvidence), &Mutant::MOMENTUM[..]);
    for other in [FixtureSetId::Legacy, FixtureSetId::MomentumEvidence] {
        assert!(reg.iter_set(other).all(|r| r.id != "low_vol_quintile_tilt_20d"));
    }
    assert_eq!(reg.iter_set(FixtureSetId::LowVolEvidence).count(), 1);
    assert_eq!(reg.by_sleeve(MutantSleeve::S5).unwrap().id, "low_vol_quintile_tilt_20d");
}
