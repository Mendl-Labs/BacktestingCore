//! Env-gated: the replication ladder on the REAL pinned data (vendor-derived, so not in this public repository).
//!
//! Set `WEIGHTSIM_RULES_LADDER_DIR` to the Engine's `program/tests/fixtures/replication_ladder` directory; otherwise
//! every test here prints SKIPPED and passes. The Engine's own `program/tests/ladder_certification.rs` runs the same
//! `ladder::self_test` against the committed fixtures on every CI run.

use std::path::PathBuf;

use weightsim_rules::ladder::fixtures::Fixtures;
use weightsim_rules::ladder::{self_test, LadderReport};

const ENV: &str = "WEIGHTSIM_RULES_LADDER_DIR";

fn dir(test: &str) -> Option<PathBuf> {
    match std::env::var(ENV) {
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            println!("SKIPPED {test}: set {ENV} to the replication_ladder fixture directory (vendor data is not in this repository)");
            None
        }
    }
}

fn certified() -> Option<(Fixtures, LadderReport)> {
    let dir = dir("real ladder")?;
    let fx = Fixtures::from_dir(&dir).unwrap_or_else(|e| panic!("fixtures: {e}"));
    match self_test(&fx) {
        Ok(r) => Some((fx, r)),
        Err(e) => panic!("{e}"),
    }
}

/// Amendment 11 section 2, transcribed by hand (NOT read from mutants.json): the tiers that must catch each mutant.
const AMENDMENT_11_CAUGHT_BY: [(&str, &[&str]); 8] = [
    ("s3_same_day_peek", &["tier1_bands", "tier2_identity", "tier3_weights"]),
    ("s3_extra_1day_delay", &["tier1_bands", "tier2_identity", "tier3_weights"]),
    ("s3_sma_excludes_today", &["tier1_bands", "tier2_identity"]),
    ("s3_half_sizing", &["tier1_bands", "tier2_identity", "tier3_weights"]),
    ("s3_drifting_subaccounts", &["tier1_bands", "tier2_identity"]),
    ("s1_sma_excludes_current", &["tier1_bands", "tier2_identity", "tier3_weights"]),
    ("s1_one_bar_late", &["tier2_identity"]),
    ("s1_daily_rebalanced_20", &["tier2_identity"]),
];

#[test]
fn real_ladder_passes_tiers_i_to_iv_on_the_pinned_data() {
    let Some((_fx, r)) = certified() else { return };
    println!("{r}");
    assert!(r.passed());
    assert!(r.failures().is_empty());
    // Tiers I-III, S1 and S3, gross and net.
    assert_eq!(r.sleeves.len(), 2);
    for s in &r.sleeves {
        for b in [&s.gross, &s.net] {
            assert!(b.tier1_pass && b.tier2_pass && b.tier3_pass, "{} {}", s.code, b.basis);
            assert!(b.window_matches_key && b.cmp.covers_key_exactly());
        }
    }
    let (s1, s3) = (&r.sleeves[0], &r.sleeves[1]);
    assert_eq!((s1.code, s3.code), ("S1", "S3"));
    assert_eq!((s1.gross.cmp.common_days, s3.gross.cmp.common_days), (2303, 1827));
    assert_eq!((s1.gross.flips_run, s3.gross.flips_run), (99, 138));
    // Amendment 11 key numbers (gross | net).
    let near = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol;
    assert!(near(s1.gross.cmp.key_sharpe, 0.6449, 5e-5) && near(s1.net.cmp.key_sharpe, 0.6105, 5e-5));
    assert!(near(s3.gross.cmp.key_sharpe, 2.1001, 5e-5) && near(s3.net.cmp.key_sharpe, 2.0723, 5e-5));
    assert!(near(s1.gross.cmp.key_cagr, 0.0414, 5e-5) && near(s1.net.cmp.key_cagr, 0.0391, 5e-5));
    assert!(near(s3.gross.cmp.key_cagr, 2.1824, 5e-5) && near(s3.net.cmp.key_cagr, 2.1265, 5e-5));
    // The run reproduces the key, not merely comes close: per-bar identity far inside 1e-9.
    for s in &r.sleeves {
        assert!(s.gross.cmp.max_abs_ret_diff <= 1e-9 && s.net.cmp.max_abs_ret_diff <= 1e-9);
        assert!(s.vs_shadow_saved_max_abs <= 1e-9);
    }
}

#[test]
fn all_eight_mutants_fail_certification_in_the_tiers_the_amendment_names() {
    let Some((_fx, r)) = certified() else { return };
    assert_eq!(r.mutants.len(), 8);
    for (name, tiers) in AMENDMENT_11_CAUGHT_BY {
        let m = r.mutants.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("mutant {name} did not run"));
        let mut got = m.caught_by.clone();
        got.sort_unstable();
        let mut want = tiers.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "{name}");
        assert!(m.mismatches.is_empty(), "{name}: {:?}", m.mismatches);
    }
    // The two mutants that pass every Tier I band are caught by Tier II alone.
    for name in ["s1_one_bar_late", "s1_daily_rebalanced_20"] {
        let m = r.mutants.iter().find(|m| m.name == name).unwrap();
        assert!(m.cmp.bands_pass && !m.cmp.tier2_pass && m.cmp.tier3.unwrap().pass, "{name}");
    }
    // Layer C canaries.
    let sharpe = |n: &str| r.mutants.iter().find(|m| m.name == n).unwrap().cmp.run_sharpe;
    assert!((sharpe("s3_same_day_peek") - 3.39).abs() <= 0.05);
    assert!((sharpe("s3_extra_1day_delay") - 2.16).abs() <= 0.05);
}

/// Digests measured on aarch64 (WSL2 Ubuntu). They depend only on IEEE +, -, *, /, sqrt in the simulation path, so
/// they are expected to be identical on x86; the Engine's CI (x86) checks exactly that. A change here means the
/// certified behaviour changed and must be reviewed together with a passing ladder.
const PINNED_LADDER_DIGEST: &str = "7da209aac8c254b3beaabd23086e3610e1521886c1372c43d455c04fff339681";
const PINNED_SERIES_DIGESTS: [(&str, &str); 4] = [
    ("S1 gross", "0a857c83d2558b8e9b4d5ca2175ec9f8bd06b640867575185635c1322bbe0987"),
    ("S1 net", "2bf54e4607077ca9f0037e7898a0e95de1394781045f597c236742faa73996d7"),
    ("S3 gross", "35e20b47ddd7b28cda5f8d5bf1b9cf6abca689e162e69ce5696e42ff67a35156"),
    ("S3 net", "20e4b939a33d5c75b5a6b4b6317f5a218ab4416dec41d3dce7ef8a900ff81e2c"),
];

#[test]
fn ladder_output_is_reproducible_and_matches_the_pinned_digests() {
    let Some((fx, a)) = certified() else { return };
    let b = self_test(&fx).unwrap();
    assert_eq!(a.digest, b.digest);
    assert_eq!(a.series_digests(), b.series_digests());
    println!("LADDER DIGEST {}", a.digest);
    for (name, d) in a.series_digests() {
        println!("SERIES DIGEST {name}: {d}");
    }
    let got = a.series_digests();
    for ((name, want), (gname, gd)) in PINNED_SERIES_DIGESTS.iter().zip(&got) {
        assert_eq!(name, gname);
        assert_eq!(want, gd, "series digest of {name} changed");
    }
    assert_eq!(a.digest, PINNED_LADDER_DIGEST, "the ladder digest changed");
}

#[test]
fn without_the_flat_start_the_net_series_cannot_reproduce_the_key() {
    // Negative control on the real data: crypto has traded since its 100th bar in 2015, so a rule that is not started
    // flat at the entry bar carries that history's equity into the window and misses the key's entry cost.
    let Some(dir) = dir("flat-start negative control") else { return };
    let fx = Fixtures::from_dir(&dir).unwrap();
    let (_, checks) = weightsim_rules::ladder::certify_sleeve(
        &fx,
        weightsim_rules::ladder::mutants::MutantSleeve::S3,
        &|_p: &weightsim::Panel| weightsim_rules::CryptoTrendRule,
    )
    .unwrap();
    let failed: Vec<&str> = checks.iter().filter(|c| !c.passed).map(|c| c.name.as_str()).collect();
    assert!(failed.contains(&"S3.net.tier2.identity"), "{failed:?}");
    assert!(failed.contains(&"S3.gross.tier2.identity"), "{failed:?}");
    // the returns themselves are the same rule's returns: the bands still pass
    assert!(!failed.contains(&"S3.gross.tier1.bands"), "{failed:?}");
    // and the certified (flat-started) configuration passes the same checks
    let entry = weightsim_rules::ladder::runner::entry_date(&fx.crypto_panel, &fx.s3).unwrap();
    let (_, ok) = weightsim_rules::ladder::certify_sleeve(
        &fx,
        weightsim_rules::ladder::mutants::MutantSleeve::S3,
        &|_p: &weightsim::Panel| weightsim_rules::FlatUntil::new(weightsim_rules::CryptoTrendRule, entry),
    )
    .unwrap();
    assert!(ok.iter().all(|c| c.passed));
}
