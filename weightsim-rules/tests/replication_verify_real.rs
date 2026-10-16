//! Env-gated: replicate and re-verify on the REAL pinned data (vendor-derived, so not in this public repository).
//!
//! Set `WEIGHTSIM_RULES_LADDER_DIR` to the Engine's `program/tests/fixtures/replication_ladder` directory; otherwise
//! every test here prints SKIPPED and passes. Only digests and headline numbers (already public in this repository's
//! `ladder_real.rs` and in the pre-registration) appear below; no vendor data is copied here.

use std::path::PathBuf;

use weightsim::Date;
use weightsim_rules::ladder::fixtures::{Fixtures, Pins, F_MANIFEST};
use weightsim_rules::ladder::verify::*;

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

/// The four series digests pinned in `ladder_real.rs` (S1 gross, S1 net, S3 gross, S3 net).
const PINNED: [&str; 4] = [
    "0a857c83d2558b8e9b4d5ca2175ec9f8bd06b640867575185635c1322bbe0987",
    "2bf54e4607077ca9f0037e7898a0e95de1394781045f597c236742faa73996d7",
    "35e20b47ddd7b28cda5f8d5bf1b9cf6abca689e162e69ce5696e42ff67a35156",
    "20e4b939a33d5c75b5a6b4b6317f5a218ab4416dec41d3dce7ef8a900ff81e2c",
];

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn real_replication_reproduces_the_pinned_digests_and_verifies_from_the_columns_alone() {
    let Some(dir) = dir("real replication") else { return };
    let fx = Fixtures::from_dir(&dir).unwrap_or_else(|e| panic!("fixtures: {e}"));
    let s1 = replicate(&fx, "etf_trend_faber").unwrap();
    let s3 = replicate(&fx, "crypto_trend_100d").unwrap();
    assert_eq!([&s1.gross_sha256, &s1.net_sha256, &s3.gross_sha256, &s3.net_sha256].map(String::as_str), PINNED);
    assert_eq!(s1.gross.digest().unwrap(), s1.gross_sha256);
    assert_eq!(s3.net.digest().unwrap(), s3.net_sha256);
    assert!(s1.rule_impl_version.starts_with("weightsim-rules 0.1.0 over reference-rules, flat until "));

    for (run, key_gross, key_net, flips, bars) in
        [(&s1, (0.6449, 0.0414), (0.6105, 0.0391), 99, 2303), (&s3, (2.1001, 2.1824), (2.0723, 2.1265), 138, 1827)]
    {
        // Every tier, Tier IV included, on the columns alone, with every claim of the run checked.
        let req = VerifyRequest { claims: run.claims(), ..VerifyRequest::new(&run.rule_id, &run.gross, &run.net) };
        let v = verify_with(&fx, &req, &VerifyOptions::default()).unwrap_or_else(|e| panic!("{}: {e}", run.rule_id));
        assert!(v.tier_failures().is_empty() && v.tier4.as_ref().unwrap().passed());
        for (b, (sharpe, cagr)) in [(&v.gross, key_gross), (&v.net, key_net)] {
            assert_eq!((b.summary.n, b.summary.flips), (bars, flips));
            assert!(near(b.summary.sharpe, sharpe, 5e-5), "{} Sharpe {}", run.rule_id, b.summary.sharpe);
            assert!(near(b.summary.cagr, cagr, 5e-5), "{} CAGR {}", run.rule_id, b.summary.cagr);
            assert!(b.report.cmp.max_abs_ret_diff <= 1e-9);
        }
        assert_eq!(v.gross.report.series_sha256, run.gross_sha256);
        assert_eq!(v.net.report.series_sha256, run.net_sha256);

        // One ulp in one return, a NaN, a dropped row and a swapped pair are each refused with their own error.
        let mut n = run.net.clone();
        let t = n.n_bars() / 2;
        n.ret[t] = f64::from_bits(n.ret[t].to_bits() + 1);
        match verify_with(&fx, &VerifyRequest::new(&run.rule_id, &run.gross, &n), &VerifyOptions { tier4: false }) {
            Err(VerifyError::DigestMismatch { basis: "net", first_difference: Some(d), .. }) => {
                assert_eq!((d.column, d.bar), ("ret", t));
            }
            other => panic!("one ulp: {:?}", other.map(|_| "accepted")),
        }
        let mut n = run.net.clone();
        n.ret[t] = f64::NAN;
        assert!(matches!(verify(&fx, &run.rule_id, &run.gross, &n), Err(VerifyError::NonFinite { .. })));
        let e = verify(&fx, &run.rule_id, &run.net, &run.gross).unwrap_err();
        assert!(matches!(e, VerifyError::CostModelMismatch { basis: "gross", .. }));
        let mut g = run.gross.clone();
        let (a, b): (Date, Date) = (g.dates[0], g.dates[1]);
        g.dates[0] = b;
        g.dates[1] = a;
        assert!(matches!(verify(&fx, &run.rule_id, &g, &run.net), Err(VerifyError::Shape { .. })));
    }
}

#[test]
fn real_fixtures_from_bytes_equal_fixtures_from_the_directory() {
    let Some(dir) = dir("real bytes loader") else { return };
    let from_dir = Fixtures::from_dir(&dir).unwrap();
    // Read every file of the manifest (and the manifest) into memory, as an Engine binary would embed them.
    let mut names: Vec<String> = from_dir.verified_files.iter().map(|(n, _)| n.clone()).collect();
    names.push(F_MANIFEST.to_string());
    let owned: Vec<(String, Vec<u8>)> =
        names.iter().map(|n| (n.clone(), std::fs::read(dir.join(n)).expect("fixture file"))).collect();
    let table: Vec<(&str, &[u8])> = owned.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let from_bytes = Fixtures::from_files(&table, Pins::REAL).unwrap();
    assert_eq!(from_bytes.manifest_sha256, from_dir.manifest_sha256);
    assert_eq!(from_bytes.verified_files, from_dir.verified_files);
    assert_eq!(format!("{from_bytes:?}"), format!("{from_dir:?}"));
    for rule in LIBRARY_RULE_IDS {
        let (a, b) = (replicate(&from_bytes, rule).unwrap(), replicate(&from_dir, rule).unwrap());
        assert_eq!((a.gross_sha256, a.net_sha256), (b.gross_sha256, b.net_sha256));
    }
    // One byte of one embedded file is a pin error under the real pins.
    let mut changed = owned[0].1.clone();
    let mid = changed.len() / 2;
    changed[mid] ^= 1;
    let mut t2 = table.clone();
    t2[0] = (owned[0].0.as_str(), changed.as_slice());
    assert!(matches!(Fixtures::from_files(&t2, Pins::REAL), Err(weightsim_rules::ladder::LadderError::Pin(_))));
}
