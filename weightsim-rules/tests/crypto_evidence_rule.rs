//! Always-on tests of the crypto evidence sleeve (Amendment 13): the adapter is the published `crypto_trend_100d`
//! point under its evidence-set id, and each of the five pre-registered Tier IV mutants (section 5 and T4) is a wrong
//! implementation that the ladder's own comparison catches. The "key" here is the honest rule's own gross run on the
//! crate's synthetic BTC/ETH prices (the real key is the Python export on the evidence window, checked in the Engine),
//! so what is proven is detection power: the same `compare` the ladder uses fails every mutant and passes the honest
//! rule, and every mutant differs from the honest rule's returns.

mod common;

use common::s3_panel;
use weightsim::{simulate, CostModel, DecisionSchedule, Panel, RebalancePolicy, SimConfig, WeightRule};
use weightsim_rules::adapters::{CryptoTrendEvidenceRule, CryptoTrendRule};
use weightsim_rules::ladder::checks::{compare, Comparison};
use weightsim_rules::ladder::fixtures::{required_files_for, KeyBar, SleeveKey};
use weightsim_rules::ladder::mutants::{run_mutant_on, Mutant, MutantSleeve};
use weightsim_rules::ladder::runner::{key_rows_for_mutants, rows_from_sim, run_gross};
use weightsim_rules::ladder::{FixtureSetId, PanelSpec, Registry};

const CRY: [&str; 2] = ["BTC", "ETH"];

/// The honest rule's gross run over the synthetic panel as a key (net columns are copies: mutants read returns and
/// weights only).
fn synthetic_key(panel: &Panel) -> SleeveKey {
    assert!(panel.n_bars() > 400, "the synthetic panel is long enough for a 100-day warm-up plus a window");
    let start = panel.dates()[150];
    let end = panel.dates()[panel.n_bars() - 1];
    let cfg = SimConfig { start: Some(start), end: Some(end), cost: CostModel::ZERO, ..SimConfig::default() };
    let sim = simulate(panel, &CryptoTrendEvidenceRule::default(), &cfg).unwrap();
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
        code: "S3e",
        rule_id: "crypto_trend_100d_evidence",
        symbols: CRY.iter().map(|s| (*s).to_string()).collect(),
        bars,
        flips: 0,
    }
}

fn against_key(panel: &Panel, key: &SleeveKey, m: Mutant) -> Comparison {
    let run = run_mutant_on(panel, key, m).unwrap();
    compare(&key_rows_for_mutants(key), &run.rows).unwrap()
}

#[test]
fn the_adapter_is_the_published_crypto_trend_point_under_the_evidence_id() {
    let e = CryptoTrendEvidenceRule::default();
    let s3 = CryptoTrendRule::default();
    assert_eq!(e.id(), "crypto_trend_100d_evidence");
    assert_eq!(s3.id(), "crypto_trend_100d", "the legacy rule keeps its id");
    assert_eq!(e.universe(), &CRY[..]);
    assert_eq!(e.decision_schedule(), DecisionSchedule::Daily);
    assert_eq!(e.rebalance_policy(), RebalancePolicy::EveryBar);
    assert_eq!(e.min_history_bars(), 100);
    assert_eq!(e.execution_lag_bars(), 0);
    assert!(e.inner().is_published());
    // Everything but the id is the legacy rule's, byte for byte (the parameters the library entry pins included).
    assert_eq!(e.declared_parameters(), s3.declared_parameters());
    assert_eq!(e.impl_version(), s3.impl_version());
    let p = e.declared_parameters();
    assert_eq!(p.len(), 4);
    assert_eq!(p["sma_days"], "100");
    assert_eq!(p["weight_per_instrument"], "0.5");
    assert_eq!(p["schedule"], "\"daily\"");
    assert_eq!(p["rebalance_policy"], "\"every_bar\"");
}

#[test]
fn the_evidence_rule_makes_the_same_decisions_as_the_legacy_rule() {
    let panel = s3_panel();
    let cfg = SimConfig { start: Some(panel.dates()[150]), ..SimConfig::default() };
    let a = simulate(&panel, &CryptoTrendEvidenceRule::default(), &cfg).unwrap();
    let b = simulate(&panel, &CryptoTrendRule::default(), &cfg).unwrap();
    assert_eq!(a.n_bars(), b.n_bars());
    for t in 0..a.n_bars() {
        assert_eq!(a.row(&a.target_weights, t), b.row(&b.target_weights, t), "bar {t}");
    }
}

#[test]
fn the_honest_rule_passes_the_comparison_it_is_the_key_of() {
    let panel = s3_panel();
    let key = synthetic_key(&panel);
    let sim = run_gross(&panel, &CryptoTrendEvidenceRule::default(), &key, 0).unwrap();
    let rows = rows_from_sim(&sim, key.bars[0].date, key.bars[key.bars.len() - 1].date, false).unwrap();
    let cmp = compare(&key_rows_for_mutants(&key), &rows).unwrap();
    assert!(cmp.failed_tiers().is_empty(), "the control must pass: {:?}", cmp.failed_tiers());
}

/// Amendment 13 T4: every one of the five mutants must FAIL at least one of Tiers I-III on the window, and none of
/// them may reproduce the honest rule's returns (a mutant equal to the rule would be a pre-registration blocker).
#[test]
fn every_crypto_evidence_mutant_is_killed_by_the_ladders_own_comparison() {
    let panel = s3_panel();
    let key = synthetic_key(&panel);
    assert_eq!(Mutant::CRYPTO_EVIDENCE.len(), 5);
    for m in Mutant::CRYPTO_EVIDENCE {
        assert_eq!(m.sleeve(), MutantSleeve::S3e);
        let cmp = against_key(&panel, &key, m);
        assert!(!cmp.failed_tiers().is_empty(), "{} was NOT caught: {:?}", m.name(), cmp);
        assert!(!cmp.tier2_pass, "{}: Tier II must catch it (max|ret diff| {:e})", m.name(), cmp.max_abs_ret_diff);
        assert!(cmp.max_abs_ret_diff > 1e-9, "{} reproduces the rule's returns (equivalent mutant)", m.name());
    }
}

#[test]
fn each_s3e_mutant_is_the_same_wrong_implementation_as_its_s3_twin() {
    let panel = s3_panel();
    let key = synthetic_key(&panel);
    let pairs = [
        (Mutant::S3eSameDayPeek, Mutant::S3SameDayPeek),
        (Mutant::S3eExtraDelay, Mutant::S3ExtraDelay),
        (Mutant::S3eSmaExcludesToday, Mutant::S3SmaExcludesToday),
        (Mutant::S3eHalfSizing, Mutant::S3HalfSizing),
        (Mutant::S3eDriftingSubaccounts, Mutant::S3DriftingSubaccounts),
    ];
    for (e, s3) in pairs {
        let (a, b) = (run_mutant_on(&panel, &key, e).unwrap(), run_mutant_on(&panel, &key, s3).unwrap());
        assert_eq!(a.rows.dates, b.rows.dates, "{}", e.name());
        assert_eq!(a.rows.ret, b.rows.ret, "{}", e.name());
        assert_eq!(a.flips, b.flips, "{}", e.name());
        assert_eq!(e.name(), s3.name(), "the pre-registered names are the S3 names");
        assert_eq!(e.description(), s3.description());
    }
    // The five are pairwise different wrong implementations of the rule.
    let rets: Vec<Vec<f64>> =
        Mutant::CRYPTO_EVIDENCE.iter().map(|m| run_mutant_on(&panel, &key, *m).unwrap().rows.ret).collect();
    for i in 0..rets.len() {
        for j in (i + 1)..rets.len() {
            assert_ne!(rets[i], rets[j], "mutants {i} and {j} are the same series");
        }
    }
}

#[test]
fn the_weight_mutants_are_caught_by_tier_iii_and_the_timing_mutants_by_the_weights_too() {
    let panel = s3_panel();
    let key = synthetic_key(&panel);
    // Half sizing changes EVERY nonzero weight cell (25% instead of 50%).
    let half = against_key(&panel, &key, Mutant::S3eHalfSizing);
    let t3 = half.tier3.expect("a mutant with weights is compared on weights");
    assert!(t3.disagreeing > 0 && t3.agreement < 1.0, "half sizing: Tier III agreement {}", t3.agreement);
    // A one-day-late book holds yesterday's weights.
    let late = against_key(&panel, &key, Mutant::S3eExtraDelay);
    assert!(late.max_abs_w_target_diff.unwrap() > 1e-9);
    // Excluding today from the average flips the decision on some bar, hence some weight.
    let sma = against_key(&panel, &key, Mutant::S3eSmaExcludesToday);
    assert!(sma.max_abs_w_target_diff.unwrap() > 1e-9, "the 100 previous closes must decide differently on some bar");
}

#[test]
fn the_crypto_evidence_rule_is_registered_in_its_own_set_and_the_legacy_s3_is_untouched() {
    let reg = Registry::library();
    let e = reg.get("crypto_trend_100d_evidence").unwrap();
    assert_eq!(e.set, FixtureSetId::CryptoEvidence);
    assert_eq!((e.key.sleeve_code, e.panel), ("S3e", PanelSpec::Crypto));
    assert_eq!(e.key.perbar_file, "key/S3e_crypto_trend_100d_perbar.csv");
    assert_eq!(e.key.shadow_file, "shadow_saved/shadow_S3e_daily_returns.csv");
    assert_eq!(e.mutants.to_vec(), Mutant::CRYPTO_EVIDENCE.to_vec());
    assert_eq!(Mutant::of_set(FixtureSetId::CryptoEvidence), &Mutant::CRYPTO_EVIDENCE[..]);
    assert_eq!(Mutant::of_set(FixtureSetId::Legacy), &Mutant::ALL[..]);
    assert!(e.canaries.is_empty(), "the 2016-2020 canary Sharpes do not transfer to this window (Amendment 13 T4)");
    assert_eq!(reg.by_sleeve(MutantSleeve::S3e).unwrap().id, "crypto_trend_100d_evidence");
    // The legacy S3 is exactly what it was: same id, sleeve, set, files, mutants and canaries.
    let s3 = reg.get("crypto_trend_100d").unwrap();
    assert_eq!((s3.set, s3.key.sleeve_code), (FixtureSetId::Legacy, "S3"));
    assert_eq!(s3.key.perbar_file, "key/S3_crypto_trend_100d_perbar.csv");
    assert_eq!(s3.mutants.to_vec(), Mutant::ALL[..5].to_vec());
    assert_eq!(s3.canaries.len(), 2);
    assert_eq!(reg.by_sleeve(MutantSleeve::S3).unwrap().id, "crypto_trend_100d");
    assert!(reg.iter_set(FixtureSetId::Legacy).all(|r| r.id != "crypto_trend_100d_evidence"));
    // One spec, two fixture sets: each set resolves its own entry.
    assert_eq!(s3.spec, e.spec);
    assert_eq!(reg.get_spec_in_set(FixtureSetId::Legacy, &e.spec).unwrap().id, "crypto_trend_100d");
    assert_eq!(reg.get_spec_in_set(FixtureSetId::CryptoEvidence, &e.spec).unwrap().id, "crypto_trend_100d_evidence");
    assert_eq!(reg.get_spec(&e.spec).unwrap().id, "crypto_trend_100d", "the earlier set's entry first");
}

#[test]
fn the_evidence_set_requires_exactly_its_own_files() {
    assert_eq!(
        required_files_for(FixtureSetId::CryptoEvidence),
        vec![
            "ladder_candles.csv",
            "key/key_metrics.json",
            "key/mutants.json",
            "key/S3e_crypto_trend_100d_perbar.csv",
            "shadow_saved/shadow_S3e_daily_returns.csv",
        ]
    );
}
