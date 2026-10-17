//! W3.1: the rule registry replacing the two-arm `if/else`. Always on (synthetic fixtures).
//!
//! The one property that matters: the registry is a REFACTOR of the ladder's wiring, so every digest the ladder
//! produces over the crate's own synthetic fixtures is byte-identical to the value measured on the tree BEFORE the
//! registry existed (captured by running this crate's tests at `local-dev` commit cd0f462, 2026-10-02). The real-data
//! digest (`7da209aa...`) is pinned the same way in `tests/ladder_real.rs` and in the Engine's
//! `CERTIFIED_LADDER_DIGESTS`.

mod common;

use weightsim::{DecisionSchedule, RebalancePolicy, WeightRule};
use weightsim_rules::ladder::fixtures::{
    required_files, required_files_for, Fixtures, LadderError, Pins, REAL_CANDLES_SHA256, REAL_MANIFEST_SHA256,
};
use weightsim_rules::ladder::mutants::{Mutant, MutantSleeve};
use weightsim_rules::ladder::verify::LIBRARY_RULE_IDS;
use weightsim_rules::ladder::{
    replicate, rule_facts, run_ladder_with, CadenceSpec, FixtureSetId, LadderOptions, OverlaySpec, PanelSpec,
    RegisteredRule, Registry, RegistryError, ReplicateError, RuleSpec, SignalSpec, UniverseSpec, WeightingSpec,
    LIBRARY_RULES,
};

const NO_CANARIES: LadderOptions = LadderOptions { check_canaries: false };

/// Measured BEFORE the registry (tree at `local-dev` cd0f462): the ladder digest and the four base-run series
/// digests over the synthetic fixtures of `tests/common`.
const PRE_REGISTRY_LADDER_DIGEST: &str = "e21e94b1f9b4a3c40313514e231c0ea0753b066af2c85a59a8d6c6052265c3db";
const PRE_REGISTRY_SERIES_DIGESTS: [(&str, &str); 4] = [
    ("S1 gross", "640d62c6e3d87acb53cb0643a9bdce458ca2b41bb0dadcb4cf12ed2a7881068e"),
    ("S1 net", "6e64c47b434ebbe25f73755810a3e61dfcafe190b1bba8f21934d02bdf3e5a55"),
    ("S3 gross", "0a2ce28aec227a3084a2786dbec6cb9793e3271cbf72744f25145908d474c59d"),
    ("S3 net", "d4cd9a87c3fc4ac5051d3fb457033b4caa7600669db5b30f507202977b37c5da"),
];

#[test]
fn the_registry_changes_no_digest_over_the_synthetic_fixtures() {
    let fx = common::fixtures();
    let report = run_ladder_with(&fx, &NO_CANARIES).unwrap();
    assert!(report.passed(), "{report}");
    assert_eq!(report.digest, PRE_REGISTRY_LADDER_DIGEST, "the ladder digest changed");
    let got = report.series_digests();
    assert_eq!(got.len(), PRE_REGISTRY_SERIES_DIGESTS.len());
    for ((name, want), (gname, gd)) in PRE_REGISTRY_SERIES_DIGESTS.iter().zip(&got) {
        assert_eq!(name, gname);
        assert_eq!(want, gd, "series digest of {name} changed");
    }
    // The replication path (what the Engine stores and verifies) produces the same series digests.
    for (entry, basis_digests) in LIBRARY_RULES.iter().zip(PRE_REGISTRY_SERIES_DIGESTS.chunks(2)) {
        let run = replicate(&fx, entry.id).unwrap();
        assert_eq!(run.gross_sha256, basis_digests[0].1, "{} gross", entry.id);
        assert_eq!(run.net_sha256, basis_digests[1].1, "{} net", entry.id);
    }
}

#[test]
fn the_library_table_is_the_certified_rules_in_sleeve_order() {
    let reg = Registry::library();
    assert_eq!(reg.ids(), vec!["etf_trend_faber", "crypto_trend_100d", "momentum_rank_weighted", "dual_momentum_gem"]);
    let legacy: Vec<&str> = reg.iter_set(FixtureSetId::Legacy).map(|r| r.id).collect();
    assert_eq!(legacy, LIBRARY_RULE_IDS.to_vec(), "LIBRARY_RULE_IDS is a view of the legacy set of the registry");
    assert_eq!(reg.len(), 4);
    assert!(!reg.is_empty());
    let s1 = reg.get("etf_trend_faber").unwrap();
    let s3 = reg.get("crypto_trend_100d").unwrap();
    assert_eq!((s1.key.sleeve_code, s3.key.sleeve_code), ("S1", "S3"));
    assert_eq!((s1.panel, s3.panel), (PanelSpec::Etf, PanelSpec::Crypto));
    assert_eq!(s1.key.perbar_file, "key/S1_etf_trend_faber_perbar.csv");
    assert_eq!(s3.key.perbar_file, "key/S3_crypto_trend_100d_perbar.csv");
    assert_eq!(reg.by_sleeve(MutantSleeve::S1).unwrap().id, "etf_trend_faber");
    assert_eq!(reg.by_sleeve(MutantSleeve::S3).unwrap().id, "crypto_trend_100d");
    // The factory builds the rule the id names, and the stamped version is the adapter's.
    for e in reg.iter() {
        let rule = (e.factory)();
        assert_eq!(rule.id(), e.id);
        assert_eq!(e.impl_version(), rule.impl_version());
        assert!(e.impl_version().starts_with("weightsim-rules "));
        assert_eq!(e.sleeve().unwrap().code(), e.key.sleeve_code);
    }
    // Only the crypto sleeve carries the Layer C canaries, in the amendment's order.
    assert!(s1.canaries.is_empty());
    assert_eq!(
        s3.canaries.iter().map(|c| (c.check, c.mutant)).collect::<Vec<_>>(),
        vec![
            ("canary.s3_same_day_peek_sharpe", Mutant::S3SameDayPeek),
            ("canary.s3_extra_delay_sharpe", Mutant::S3ExtraDelay)
        ]
    );
}

#[test]
fn every_entrys_mutants_are_exactly_its_sleeves_mutants_in_table_order() {
    let reg = Registry::library();
    let mut seen = 0usize;
    for e in reg.iter() {
        let sleeve = e.sleeve().unwrap();
        let want: Vec<Mutant> = Mutant::of_set(e.set).iter().copied().filter(|m| m.sleeve() == sleeve).collect();
        assert_eq!(e.mutants.to_vec(), want, "{}", e.id);
        for c in e.canaries {
            assert!(e.mutants.contains(&c.mutant), "a canary names one of its own rule's mutants");
        }
        seen += e.mutants.len();
    }
    assert_eq!(
        seen,
        Mutant::ALL.len() + Mutant::MOMENTUM.len() + Mutant::GEM.len(),
        "every named mutant belongs to exactly one registered rule"
    );
}

#[test]
fn every_entry_binds_to_the_fixture_set_it_claims() {
    let fx = common::fixtures();
    for e in Registry::library().iter_set(FixtureSetId::Legacy) {
        let sf = e.fixture(&fx).unwrap();
        assert_eq!(sf.key.rule_id, e.id);
        assert_eq!(sf.key.code, e.key.sleeve_code);
        let universe: Vec<&str> = (e.factory)().universe().to_vec();
        assert_eq!(sf.panel.symbols().iter().map(String::as_str).collect::<Vec<_>>(), universe);
        assert_eq!(sf.key.symbols.iter().map(String::as_str).collect::<Vec<_>>(), universe);
    }
    assert!(fx.sleeve("S2").is_none());
}

#[test]
fn registering_a_duplicate_id_is_rejected() {
    let dup = vec![LIBRARY_RULES[0], LIBRARY_RULES[1], LIBRARY_RULES[0]];
    assert_eq!(
        Registry::from_rules(dup).err(),
        Some(RegistryError::DuplicateRule { rule_id: "etf_trend_faber".into() })
    );
    // Two different ids on one sleeve are rejected too.
    let other_id = RegisteredRule { id: "etf_trend_faber_v2", ..LIBRARY_RULES[0] };
    assert_eq!(
        Registry::from_rules(vec![LIBRARY_RULES[0], other_id]).err(),
        Some(RegistryError::DuplicateSleeve { sleeve_code: "S1".into() })
    );
    // The reviewed table itself is valid, and an empty registry is a registry.
    assert!(Registry::from_rules(LIBRARY_RULES.to_vec()).is_ok());
    assert!(Registry::from_rules(vec![]).unwrap().is_empty());
}

#[test]
fn an_unknown_id_is_a_typed_error_never_a_panic() {
    let reg = Registry::library();
    for id in ["momentum_12_1", "", "ETF_TREND_FABER", "etf_trend_faber "] {
        assert_eq!(reg.get(id).err(), Some(RegistryError::UnknownRule { rule_id: id.into() }));
        assert!(rule_facts(id).is_none());
        let e = replicate(&common::fixtures(), id).expect_err("replicate refuses");
        assert_eq!(e, ReplicateError::UnknownRule { rule_id: id.into() });
        assert!(e.to_string().contains("unknown library rule"));
    }
    let e = RegistryError::UnknownRule { rule_id: "x".into() };
    assert_eq!(e.to_string(), "`x` is not a registered library rule");
}

#[test]
fn rule_facts_are_the_registry_entrys_facts() {
    for e in Registry::library().iter() {
        let f = rule_facts(e.id).unwrap();
        let rule = (e.factory)();
        assert_eq!(f.id, e.id);
        assert_eq!(f.sleeve_code, e.key.sleeve_code);
        assert_eq!(f.universe, rule.universe().to_vec());
        assert_eq!(f.declared_parameters, rule.declared_parameters());
        assert_eq!(f.decision_schedule, rule.decision_schedule());
        assert_eq!(f.rebalance_policy, rule.rebalance_policy());
        assert_eq!(f.min_history_bars, rule.min_history_bars());
    }
}

#[test]
fn the_registry_is_keyed_by_rule_spec_and_each_spec_describes_its_compiled_rule() {
    let reg = Registry::library();
    for e in reg.iter() {
        // The spec is the registry's key: looking it up returns the entry that carries it.
        assert_eq!(reg.get_spec(&e.spec).unwrap().id, e.id);
        // The cadence and universe the spec declares are the compiled rule's own.
        let rule = (e.factory)();
        assert_eq!(e.spec.cadence.decision, rule.decision_schedule(), "{}", e.id);
        assert_eq!(e.spec.cadence.rebalance, rule.rebalance_policy(), "{}", e.id);
        let UniverseSpec::Fixed(symbols) = e.spec.universe;
        assert_eq!(symbols.to_vec(), rule.universe().to_vec(), "{}", e.id);
        assert_eq!(e.spec.overlay, OverlaySpec::None);
    }
    // The two certified specs, exactly as the rules are described in their adapter docs.
    let s1 = reg.get("etf_trend_faber").unwrap().spec;
    let s3 = reg.get("crypto_trend_100d").unwrap().spec;
    assert_eq!(s1.signal, SignalSpec::CloseAboveSma { window: 10 });
    assert_eq!(s3.signal, SignalSpec::CloseAboveSma { window: 100 });
    assert_eq!(s1.weighting, WeightingSpec::FixedPerInstrument { weight: 0.20 });
    assert_eq!(s3.weighting, WeightingSpec::FixedPerInstrument { weight: 0.50 });
    assert_eq!(
        s1.cadence,
        CadenceSpec { decision: DecisionSchedule::LastBarOfMonth, rebalance: RebalancePolicy::OnDecision }
    );
    assert_eq!(s3.cadence, CadenceSpec { decision: DecisionSchedule::Daily, rebalance: RebalancePolicy::EveryBar });
    // A spec that no entry carries is a typed error, and a second entry with the same spec is refused.
    let unknown = RuleSpec { weighting: WeightingSpec::FixedPerInstrument { weight: 0.10 }, ..s1 };
    assert_eq!(reg.get_spec(&unknown).err(), Some(RegistryError::UnknownSpec { spec: unknown }));
    let twin = RegisteredRule {
        id: "etf_trend_faber_twin",
        key: weightsim_rules::ladder::KeyRef { sleeve_code: "S9", ..LIBRARY_RULES[0].key },
        ..LIBRARY_RULES[0]
    };
    assert_eq!(
        Registry::from_rules(vec![LIBRARY_RULES[0], twin]).err(),
        Some(RegistryError::DuplicateSpec { spec: s1 })
    );
}

/// Amendment 14 / test T9: giving the registry fixture sets must leave the legacy set byte-identical. The pins are the
/// literals the Engine's `certified.rs` carries (`CERTIFIED_MANIFEST_SHA256`, `CERTIFIED_CANDLES_SHA256`); the ladder
/// digest over the synthetic legacy set is pinned above (`PRE_REGISTRY_LADDER_DIGEST`) and the real one in
/// `tests/ladder_real.rs` / the Engine's `EXPECTED_LADDER_DIGEST`.
#[test]
fn the_legacy_fixture_set_is_untouched_by_fixture_sets() {
    assert_eq!(REAL_MANIFEST_SHA256, "d51444c33fc2630e12315e936f79580e13dca834ca0fdbc40a242ef496bd6d90");
    assert_eq!(REAL_CANDLES_SHA256, "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365");
    assert_eq!(Pins::REAL.manifest_sha256, REAL_MANIFEST_SHA256);
    assert_eq!(Pins::REAL.candles_sha256, REAL_CANDLES_SHA256);
    // Both certified rules are in the legacy set, in the same order as before; the legacy file list is unchanged.
    for e in Registry::library().iter_set(FixtureSetId::Legacy) {
        assert_eq!(e.set, FixtureSetId::Legacy);
    }
    assert_eq!(Registry::library().iter_set(FixtureSetId::Legacy).count(), 2);
    assert_eq!(required_files(), required_files_for(FixtureSetId::Legacy));
    assert_eq!(
        required_files(),
        vec![
            "ladder_candles.csv",
            "key/key_metrics.json",
            "key/mutants.json",
            "key/S1_etf_trend_faber_perbar.csv",
            "shadow_saved/shadow_S1_daily_returns.csv",
            "key/S3_crypto_trend_100d_perbar.csv",
            "shadow_saved/shadow_S3_daily_returns.csv",
        ]
    );
    let fx = common::fixtures();
    assert_eq!(fx.set, FixtureSetId::Legacy);
    assert!(fx.try_panel(PanelSpec::Etf).is_some() && fx.try_panel(PanelSpec::Crypto).is_some());
}

#[test]
fn a_rule_of_another_fixture_set_is_refused_by_a_loaded_set() {
    let fx = common::fixtures();
    let foreign = RegisteredRule { set: FixtureSetId::MomentumEvidence, ..LIBRARY_RULES[0] };
    match foreign.fixture(&fx) {
        Err(LadderError::Inconsistent(m)) => assert!(m.contains("MomentumEvidence"), "{m}"),
        other => panic!("expected an Inconsistent error, got {:?}", other.map(|_| ())),
    }
    // The registry can filter by set: a foreign entry is not in the legacy listing.
    let reg = Registry::from_rules(vec![
        LIBRARY_RULES[0],
        RegisteredRule { set: FixtureSetId::MomentumEvidence, ..LIBRARY_RULES[1] },
    ])
    .unwrap();
    assert_eq!(reg.iter_set(FixtureSetId::Legacy).map(|r| r.id).collect::<Vec<_>>(), vec!["etf_trend_faber"]);
    assert_eq!(
        reg.iter_set(FixtureSetId::MomentumEvidence).map(|r| r.id).collect::<Vec<_>>(),
        vec!["crypto_trend_100d"]
    );
}

#[test]
fn a_fixture_set_never_loads_another_sets_files() {
    // The synthetic LEGACY files carry S1/S3 only: loading them as the momentum set must fail on the missing S4
    // key (the manifest does not list it), never silently pass or fall back to the legacy sleeves.
    let (files, ms, cs) = common::ready();
    let reader = |name: &str| -> Result<Vec<u8>, String> {
        files.files.get(name).cloned().ok_or_else(|| format!("{name}: not in the synthetic table"))
    };
    let pins = Pins { manifest_sha256: ms, candles_sha256: cs };
    assert!(Fixtures::load_set(&reader, pins, FixtureSetId::Legacy).is_ok());
    match Fixtures::load_set(&reader, pins, FixtureSetId::MomentumEvidence) {
        Err(LadderError::Pin(m)) => assert!(m.contains("S4_momentum_rank_weighted_perbar.csv"), "{m}"),
        other => panic!("expected a Pin error for the missing S4 key, got {:?}", other.map(|_| ())),
    }
    assert_eq!(
        required_files_for(FixtureSetId::MomentumEvidence),
        vec![
            "ladder_candles.csv",
            "key/key_metrics.json",
            "key/mutants.json",
            "key/S4_momentum_rank_weighted_perbar.csv",
            "shadow_saved/shadow_S4_daily_returns.csv",
        ]
    );
}

#[test]
fn every_panel_spec_names_the_symbols_of_its_universe() {
    assert_eq!(PanelSpec::Etf.symbols(), &reference_rules::ETF_SYMBOLS[..]);
    assert_eq!(PanelSpec::Crypto.symbols(), &reference_rules::CRYPTO_SYMBOLS[..]);
    assert_eq!(PanelSpec::Gem.symbols(), &reference_rules::DUAL_MOMENTUM_GEM_SYMBOLS[..]);
}

#[test]
fn the_gem_files_are_exactly_its_own_sets_and_loading_another_set_never_finds_them() {
    assert_eq!(
        required_files_for(FixtureSetId::GemEvidence),
        vec![
            "ladder_candles.csv",
            "key/key_metrics.json",
            "key/mutants.json",
            "key/S6_dual_momentum_gem_perbar.csv",
            "shadow_saved/shadow_S6_daily_returns.csv",
        ]
    );
    // The legacy files carry S1/S3 only: loading them as the GEM set fails on the missing S6 key.
    let (files, ms, cs) = common::ready();
    let reader = |name: &str| -> Result<Vec<u8>, String> {
        files.files.get(name).cloned().ok_or_else(|| format!("{name}: not in the synthetic table"))
    };
    let pins = Pins { manifest_sha256: ms, candles_sha256: cs };
    match Fixtures::load_set(&reader, pins, FixtureSetId::GemEvidence) {
        Err(LadderError::Pin(m)) => assert!(m.contains("S6_dual_momentum_gem_perbar.csv"), "{m}"),
        other => panic!("expected a Pin error for the missing S6 key, got {:?}", other.map(|_| ())),
    }
    // The legacy set still loads, and loads no GEM panel.
    let fx = Fixtures::load_set(&reader, pins, FixtureSetId::Legacy).unwrap();
    assert!(fx.try_panel(PanelSpec::Gem).is_none());
}
