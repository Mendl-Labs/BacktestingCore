//! Always-on tests of the `dual_momentum_gem` sleeve (Amendment 16): the adapter is the published point, the rule walks
//! all three of its holdings on a synthetic panel whose month-end returns are known exactly, and each of the five
//! pre-registered Tier IV mutants (section 6, T4) is a wrong implementation that the ladder's own comparison catches.
//! The "key" here is the honest rule's own gross run on that synthetic panel (the real key is the Python export on the
//! evidence window, which does not exist until the owner-only data step has run), so what is proven is detection power:
//! the same `compare` the ladder uses fails every mutant and passes the honest rule.
//!
//! Every mutant is shown to differ from the rule on this panel BY ARITHMETIC (see `gem_panel`), not by luck: the
//! Amendment 15 `ddof` mutant was equivalent to its rule and could not be caught by any tier, and Amendment 16 requires
//! each mutant to come with the argument why it differs on the pre-registered data.
#![allow(clippy::needless_range_loop)]

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use weightsim::{simulate, CostModel, DecisionSchedule, Panel, RebalancePolicy, SimConfig, WeightRule};
use weightsim_rules::adapters::DualMomentumGemAdapter;
use weightsim_rules::ladder::checks::{compare, Comparison};
use weightsim_rules::ladder::fixtures::{KeyBar, SleeveKey};
use weightsim_rules::ladder::mutants::{run_mutant_on, GemVariant, Mutant, MutantSleeve, PeekGem};
use weightsim_rules::ladder::runner::{key_rows_for_mutants, rows_from_sim, run_gross};
use weightsim_rules::ladder::{FixtureSetId, PanelSpec, Registry};

const GEM: [&str; 3] = ["SPY", "EFA", "AGG"];
const N_BARS: usize = 1100;

/// Monthly growth of each instrument by month index `m` (January 2016 = 0). Every close in month `m` moves by the
/// same daily factor, so the month-end close of month `m` is the month-end close of month `m - 1` times `1 + g(m)`
/// EXACTLY, and a trailing 12-month-end return is the product of twelve `1 + g`. Hand-derived decisions (ln sums):
///   * `m = 12..=20`: SPY gate open (+11.9% falling to +2.9%), EFA ahead of SPY  -> EFA
///   * `m = 21..=35`: SPY 12-month return <= 0 (-0.12% at m = 21, deeper after)  -> AGG
///   * `m = 36`:      SPY +0.32%, EFA +0.55% (EFA still ahead)                   -> EFA
///   * `m >= 37`:     SPY ahead of EFA                                           -> SPY
fn g_spy(m: usize) -> f64 {
    if m < 18 {
        0.01
    } else if m < 30 {
        -0.02
    } else {
        0.015
    }
}

fn g_efa(m: usize) -> f64 {
    if m < 8 {
        0.005
    } else if m < 18 {
        0.02
    } else if m < 30 {
        -0.01
    } else {
        0.008
    }
}

fn g_agg(_m: usize) -> f64 {
    0.002
}

/// The synthetic SPY/EFA/AGG panel and the bar index of every calendar month's last bar.
fn gem_panel() -> (Panel, Vec<usize>) {
    let mut dates = Vec::with_capacity(N_BARS);
    let mut day = NaiveDate::from_ymd_opt(2016, 1, 1).unwrap();
    while dates.len() < N_BARS {
        if !matches!(day.weekday(), Weekday::Sat | Weekday::Sun) {
            dates.push(day);
        }
        day += Duration::days(1);
    }
    let months: Vec<usize> = dates.iter().map(|d| (d.year() - 2016) as usize * 12 + d.month0() as usize).collect();
    let mut count = vec![0usize; months[N_BARS - 1] + 1];
    let mut month_ends = vec![0usize; count.len()];
    for (t, m) in months.iter().enumerate() {
        count[*m] += 1;
        month_ends[*m] = t;
    }
    let growth: [fn(usize) -> f64; 3] = [g_spy, g_efa, g_agg];
    let mut csv = String::from("symbol,date_utc,close\n");
    for (a, sym) in GEM.iter().enumerate() {
        let mut close = 100.0_f64;
        for (t, d) in dates.iter().enumerate() {
            if t > 0 {
                let m = months[t];
                close *= (1.0 + growth[a](m)).powf(1.0 / count[m] as f64);
            }
            csv.push_str(&format!("{sym},{d},{close:.6}\n"));
        }
    }
    (Panel::from_long_csv(&csv, &GEM).unwrap(), month_ends)
}

/// The held instrument (index into `GEM`) of a one-hot weight row.
fn held(w: &[f64]) -> usize {
    let held: Vec<usize> = (0..w.len()).filter(|i| w[*i] > 0.0).collect();
    assert_eq!(held.len(), 1, "exactly one holding: {w:?}");
    assert!((w[held[0]] - 1.0).abs() < 1e-15, "the holding carries 100%: {w:?}");
    held[0]
}

/// The honest rule's gross run over the synthetic panel as a key (net columns are copies: mutants read returns and
/// weights only). The window starts after the month-end of month 14, so every branch change below is inside it.
fn synthetic_key(panel: &Panel, month_ends: &[usize]) -> SleeveKey {
    let start = panel.dates()[month_ends[14] + 1];
    let end = panel.dates()[panel.n_bars() - 1];
    let cfg = SimConfig { start: Some(start), end: Some(end), cost: CostModel::ZERO, ..SimConfig::default() };
    let sim = simulate(panel, &DualMomentumGemAdapter::default(), &cfg).unwrap();
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
        code: "S6",
        rule_id: "dual_momentum_gem",
        symbols: GEM.iter().map(|s| (*s).to_string()).collect(),
        bars,
        flips: 0,
    }
}

/// A GEM rule with one thing changed, as the mutants build it.
fn variant(trailing_months: usize, invert_relative: bool, gate_open: bool) -> GemVariant {
    GemVariant { name: "test_variant", trailing_months, invert_relative, gate_open }
}

fn against_key(panel: &Panel, key: &SleeveKey, m: Mutant) -> Comparison {
    let run = run_mutant_on(panel, key, m).unwrap();
    compare(&key_rows_for_mutants(key), &run.rows).unwrap()
}

#[test]
fn the_adapter_is_the_published_point() {
    let r = DualMomentumGemAdapter::default();
    assert_eq!(r.id(), "dual_momentum_gem");
    assert_eq!(r.universe(), &GEM[..]);
    assert_eq!(r.decision_schedule(), DecisionSchedule::LastBarOfMonth);
    assert_eq!(r.rebalance_policy(), RebalancePolicy::OnDecision);
    assert_eq!(r.min_history_bars(), 1);
    assert_eq!(r.execution_lag_bars(), 0);
    // The five parameters the library entry pins, as JSON-shaped literals (what `binding` parses back and compares).
    let p = r.declared_parameters();
    assert_eq!(p.len(), 5, "{p:?}");
    assert_eq!(p["trailing_months"], "12");
    assert_eq!(p["month_ends_needed"], "13");
    assert_eq!(p["schedule"], "\"last_bar_of_month\"");
    assert_eq!(p["rebalance_policy"], "\"on_decision\"");
    assert_eq!(p["max_weight_per_instrument"], "1.0", "a float literal, as the entry declares 1.0");
    assert!(r.impl_version().starts_with("weightsim-rules "));
}

#[test]
fn the_rule_walks_all_three_branches_exactly_as_derived_by_hand() {
    let (panel, me) = gem_panel();
    let sim = simulate(&panel, &DualMomentumGemAdapter::default(), &SimConfig::default()).unwrap();
    assert!(me.len() > 45, "the panel must reach the SPY branch for several months: {} months", me.len());
    // 13 month-end closes are needed: months 0..=11 are warm-up, month 12 is the first decision.
    for m in 0..12 {
        assert!(!sim.decision[me[m]], "month {m}: fewer than 13 month-ends, no decision");
    }
    let (mut efa, mut agg, mut spy) = (0, 0, 0);
    for m in 12..me.len() {
        assert!(sim.decision[me[m]], "month {m}: a decision bar");
        let h = held(sim.row(&sim.target_weights, me[m]));
        // Indices into GEM: EFA (1) at months 12..=20 and 36, AGG (2) at 21..=35, SPY (0) from month 37 on.
        let want = match m {
            12..=20 | 36 => 1,
            21..=35 => 2,
            _ => 0,
        };
        assert_eq!(GEM[h], GEM[want], "month {m}");
        match h {
            0 => spy += 1,
            1 => efa += 1,
            _ => agg += 1,
        }
    }
    assert_eq!((efa, agg), (10, 15), "EFA at months 12..=20 and 36, AGG at 21..=35");
    assert!(spy >= 10, "SPY from month 37 on: {spy} decisions");
    // Only month-end bars are decision bars.
    assert_eq!((0..sim.n_bars()).filter(|t| sim.decision[*t]).count(), me.len() - 12);
}

#[test]
fn the_honest_rule_passes_the_comparison_it_is_the_key_of() {
    let (panel, me) = gem_panel();
    let key = synthetic_key(&panel, &me);
    let sim = run_gross(&panel, &DualMomentumGemAdapter::default(), &key, 0).unwrap();
    let rows = rows_from_sim(&sim, key.bars[0].date, key.bars[key.bars.len() - 1].date, false).unwrap();
    let cmp = compare(&key_rows_for_mutants(&key), &rows).unwrap();
    assert!(cmp.failed_tiers().is_empty(), "the control must pass: {:?}", cmp.failed_tiers());
}

/// Amendment 16 T4: every one of the five mutants must FAIL at least one of Tiers I-III on the window.
#[test]
fn every_gem_mutant_is_killed_by_the_ladders_own_comparison() {
    let (panel, me) = gem_panel();
    let key = synthetic_key(&panel, &me);
    assert_eq!(Mutant::GEM.len(), 5);
    for m in Mutant::GEM {
        assert_eq!(m.sleeve(), MutantSleeve::S6);
        let cmp = against_key(&panel, &key, m);
        assert!(!cmp.failed_tiers().is_empty(), "{} was NOT caught: {:?}", m.name(), cmp);
        // Every mutant here changes the held instrument for at least a month, so Tier II (per-bar identity) catches
        // each, whatever the data, and the standing target differs somewhere.
        assert!(!cmp.tier2_pass, "{}: Tier II must catch it (max|ret diff| {:e})", m.name(), cmp.max_abs_ret_diff);
        assert!(cmp.max_abs_w_target_diff.unwrap() > 1e-9, "{}: the weights must differ somewhere", m.name());
    }
}

/// The arithmetic behind each mutant's one-line argument, on the synthetic panel: where the mutant's decision differs
/// from the rule's (the months below come from the table on `g_spy`/`g_efa`).
#[test]
fn each_mutant_differs_from_the_rule_where_the_amendment_argument_says() {
    let (panel, me) = gem_panel();
    let sim = |rule: &dyn WeightRule| simulate(&panel, rule, &SimConfig::default()).unwrap();
    let honest = sim(&DualMomentumGemAdapter::default());

    // relative_pick_inverted: differs at every gate-open decision (months 12..=20, 36 and 37..), never at a gate-closed one.
    let inverted = sim(&variant(12, true, false));
    for m in 12..me.len() {
        let a = held(honest.row(&honest.target_weights, me[m]));
        let b = held(inverted.row(&inverted.target_weights, me[m]));
        if (21..=35).contains(&m) {
            assert_eq!((a, b), (2, 2), "month {m}: gate closed, both hold AGG");
        } else {
            assert_ne!(a, b, "month {m}: gate open, the inverted pick holds the other one");
        }
    }

    // gate_always_open: differs exactly at the AGG decisions.
    let open = sim(&variant(12, false, true));
    for m in 12..me.len() {
        let a = held(honest.row(&honest.target_weights, me[m]));
        let b = held(open.row(&open.target_weights, me[m]));
        assert_eq!(a != b, (21..=35).contains(&m), "month {m}");
        assert_ne!(b, 2, "month {m}: the open gate never holds AGG");
    }

    // lookback 11 months: differs at month 36 (SPY +2.37% vs EFA +1.57% over 11 months; the rule's 12 months pick EFA).
    let short = sim(&variant(11, false, false));
    assert_eq!(held(honest.row(&honest.target_weights, me[36])), 1);
    assert_eq!(held(short.row(&short.target_weights, me[36])), 0);
}

#[test]
fn the_lookahead_mutant_leaks_and_the_truncation_harness_sees_it() {
    let (panel, me) = gem_panel();
    let cfg = SimConfig::default();
    let honest =
        weightsim::harness::check_poisoning(&|_p: &Panel| DualMomentumGemAdapter::default(), &panel, &cfg, 300, 7)
            .unwrap();
    assert!(honest.is_clean(), "the honest rule reads no future bar");
    // Months 20, 35 and 36 are the decisions whose NEXT month-end flips the holding (EFA -> AGG, AGG -> EFA, EFA ->
    // SPY): on the full panel the peeking rule sees the next month-end, on the truncated one it cannot.
    let samples = vec![me[20], me[35], me[36]];
    let leaky = weightsim::harness::check_rule_truncation(&|p: &Panel| PeekGem::new(p), &panel, &samples);
    assert_eq!(leaky, samples, "all three samples must expose the peek");
    let clean =
        weightsim::harness::check_rule_truncation(&|_p: &Panel| DualMomentumGemAdapter::default(), &panel, &samples);
    assert!(clean.is_empty(), "the honest rule decides the same on a truncated panel: {clean:?}");
}

#[test]
fn the_gem_rule_is_registered_in_its_own_set_and_not_in_the_others() {
    let reg = Registry::library();
    let e = reg.get("dual_momentum_gem").unwrap();
    assert_eq!(e.set, FixtureSetId::GemEvidence);
    assert_eq!((e.key.sleeve_code, e.panel), ("S6", PanelSpec::Gem));
    assert_eq!(e.mutants.to_vec(), Mutant::GEM.to_vec());
    assert_eq!(Mutant::of_set(FixtureSetId::GemEvidence), &Mutant::GEM[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::Legacy), &Mutant::ALL[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::MomentumEvidence), &Mutant::MOMENTUM[..]);
    for other in [FixtureSetId::Legacy, FixtureSetId::MomentumEvidence] {
        assert!(reg.iter_set(other).all(|r| r.id != "dual_momentum_gem"));
    }
    assert_eq!(reg.iter_set(FixtureSetId::GemEvidence).count(), 1);
    assert_eq!(reg.by_sleeve(MutantSleeve::S6).unwrap().id, "dual_momentum_gem");
}
