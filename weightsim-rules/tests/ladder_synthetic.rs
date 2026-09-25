//! Always-on: the ladder LOGIC, exercised end to end on an in-memory synthetic answer key in the exact layout of the
//! real (private) fixtures. No vendor data. The real-data run is `tests/ladder_real.rs` (env-gated).
//!
//! The synthetic key is built from independent oracle rules (`tests/common`), so the adapters, the simulator
//! alignment, the tier checks and the mutants are all checked against something they did not produce. Every
//! tolerance and every tier is then broken on purpose, one at a time, and the ladder must notice exactly that.

mod common;

use common::*;
use weightsim_rules::ladder::fixtures::{Fixtures, LadderError, Pins, REAL_CANDLES_SHA256, REAL_MANIFEST_SHA256};
use weightsim_rules::ladder::mutants::Mutant;
use weightsim_rules::ladder::{
    run_ladder, run_ladder_with, self_test, self_test_dir, LadderOptions, LadderReport, SelfTestError,
};

const NO_CANARIES: LadderOptions = LadderOptions { check_canaries: false };

fn run(fx: &Fixtures) -> LadderReport {
    run_ladder_with(fx, &NO_CANARIES).expect("ladder runs")
}

fn failed(r: &LadderReport) -> Vec<String> {
    r.failures().iter().map(|c| c.name.clone()).collect()
}

fn check<'a>(r: &'a LadderReport, name: &str) -> &'a weightsim_rules::ladder::Check {
    r.checks.iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no check named {name}"))
}

// ---------------------------------------------------------------------------------------------------- the good run

#[test]
fn the_synthetic_ladder_certifies_end_to_end() {
    let fx = fixtures();
    let r = run(&fx);
    println!("{r}");
    assert!(r.passed(), "failed: {:?}", failed(&r));
    assert!(r.checks.len() >= 40, "only {} checks ran", r.checks.len());

    for s in &r.sleeves {
        for b in [&s.gross, &s.net] {
            assert!(b.tier1_pass && b.tier2_pass && b.tier3_pass && b.window_matches_key, "{} {}", s.code, b.basis);
            assert!(b.cmp.covers_key_exactly());
            assert!(b.cmp.max_abs_ret_diff <= 1e-15, "{} {} ret diff {:e}", s.code, b.basis, b.cmp.max_abs_ret_diff);
            assert!(b.cmp.tier3.unwrap().agreement == 1.0);
            assert_eq!(b.flips_run, b.flips_key);
        }
        assert!(s.vs_shadow_saved_max_abs <= 1e-9, "{} vs the Python key: {:e}", s.code, s.vs_shadow_saved_max_abs);
        assert!(s.key_self_consistency_max_abs <= 1e-9);
        assert_eq!(s.cost_max_bar_error, 0.0);
        assert_eq!(s.poisoning_mismatches, 0);
        assert_eq!(s.truncation_disagreements, 0);
        assert_eq!(s.determinism_mismatches, 0);
        assert!(s.cost_relative_gap <= 0.10);
    }
    // Net differs from gross: costs were really charged.
    assert_ne!(r.sleeves[0].gross.series_sha256, r.sleeves[0].net.series_sha256);
    assert_ne!(r.sleeves[1].gross.series_sha256, r.sleeves[1].net.series_sha256);
    assert_eq!(r.series_digests().len(), 4);

    // Tier IV: all eight, by name, each caught.
    assert_eq!(r.mutants.len(), 8);
    for (m, rep) in Mutant::ALL.iter().zip(&r.mutants) {
        assert_eq!(rep.name, m.name());
        assert!(rep.caught(), "{} was not caught: {:?}", rep.name, rep.cmp);
        assert!(rep.mismatches.is_empty(), "{}: {:?}", rep.name, rep.mismatches);
        assert!(rep.cmp.common_days >= 100);
    }
    // The two mutants a monthly rule's bands cannot see are caught by Tier II (they always are, whatever the data).
    for name in ["s1_one_bar_late", "s1_daily_rebalanced_20"] {
        let m = r.mutants.iter().find(|m| m.name == name).unwrap();
        assert!(m.caught_by.contains(&"tier2_identity"), "{name}: {:?}", m.caught_by);
    }
    assert!(check(&r, "tier4.all_mutants_caught").passed);

    let text = r.to_string();
    assert!(text.contains("VERDICT: PASS") && text.contains(&r.digest) && text.contains("s1_one_bar_late"));
    assert_eq!(r.digest.len(), 64);
}

#[test]
fn the_ladder_is_deterministic() {
    let fx = fixtures();
    let a = run(&fx);
    let b = run(&fx);
    assert_eq!(a.digest, b.digest);
    assert_eq!(a.series_digests(), b.series_digests());
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn canaries_are_real_numbers_of_the_real_data_and_fail_on_other_data() {
    let fx = fixtures();
    // With canaries on (the certification default) the synthetic data cannot pass: their Sharpes are not 3.39/2.16.
    let r = run_ladder(&fx).unwrap();
    let f = failed(&r);
    assert_eq!(f, vec!["canary.s3_same_day_peek_sharpe".to_string(), "canary.s3_extra_delay_sharpe".to_string()]);
    match self_test(&fx) {
        Err(SelfTestError::Failed(rep)) => {
            assert_eq!(rep.failures().len(), 2);
            let msg = SelfTestError::Failed(rep).to_string();
            assert!(msg.contains("canary.s3_same_day_peek_sharpe") && msg.contains("FAILED"));
        }
        other => panic!("expected a failed self-test, got {other:?}"),
    }
    // With them off, the same fixtures certify.
    assert!(run_ladder_with(&fx, &NO_CANARIES).unwrap().passed());
}

// ------------------------------------------------------------------------------------ breaking each tier, one at a time

#[test]
fn tier2_tolerance_is_exactly_one_e_minus_nine_on_every_compared_column() {
    type Tamper = fn(&mut Fixtures, f64);
    let cases: Vec<(&str, Tamper, &str)> = vec![
        ("S3 net return", |fx, e| fx.s3.bars[50].ret_net += e, "S3.net.tier2.identity"),
        ("S3 gross return", |fx, e| fx.s3.bars[50].ret_gross += e, "S3.gross.tier2.identity"),
        ("S1 net return", |fx, e| fx.s1.bars[50].ret_net += e, "S1.net.tier2.identity"),
        ("S1 gross return", |fx, e| fx.s1.bars[50].ret_gross += e, "S1.gross.tier2.identity"),
        ("S3 net equity", |fx, e| fx.s3.bars[10].equity_net += e, "S3.net.tier2.identity"),
        ("S3 gross equity", |fx, e| fx.s3.bars[10].equity_gross += e, "S3.gross.tier2.identity"),
        ("S3 net cost", |fx, e| fx.s3.bars[10].cost += e, "S3.net.tier2.identity"),
        ("S3 net turnover", |fx, e| fx.s3.bars[10].turnover += e, "S3.net.tier2.identity"),
        ("S3 held weight", |fx, e| fx.s3.bars[30].w_held[0] += e, "S3.gross.tier2.identity"),
        ("S1 held weight", |fx, e| fx.s1.bars[30].w_held[1] += e, "S1.gross.tier2.identity"),
    ];
    // A changed return also moves the metrics recomputed from the key, which the (separate, looser) key
    // self-consistency check may notice; it is not a tier, so it is filtered out here.
    let tiers_only = |r: &LadderReport| -> Vec<String> {
        failed(r).into_iter().filter(|n| !n.ends_with("key_self_consistency")).collect()
    };
    for (label, tamper, expect) in cases {
        // 5e-10 is inside the tolerance: nothing may fail.
        let mut fx = fixtures();
        tamper(&mut fx, 5e-10);
        let r = run(&fx);
        assert!(tiers_only(&r).is_empty(), "{label}: a 5e-10 change must be tolerated, failed: {:?}", failed(&r));
        // 2e-9 is outside: exactly the named check fails, nothing else.
        let mut fx = fixtures();
        tamper(&mut fx, 2e-9);
        let r = run(&fx);
        assert_eq!(tiers_only(&r), vec![expect.to_string()], "{label}");
    }
}

#[test]
fn target_weights_are_tier2_and_tier3_and_tier3_tolerates_small_disagreement() {
    // Both bases share the same standing target, so both fail; the net key has no held weights.
    let mut fx = fixtures();
    fx.s3.bars[40].w_target[0] += 2e-9;
    let f = failed(&run(&fx));
    assert_eq!(f, vec!["S3.gross.tier2.identity".to_string(), "S3.net.tier2.identity".to_string()]);
    // 1e-7 is still inside Tier III's 1e-6 cell tolerance.
    let mut fx = fixtures();
    fx.s3.bars[40].w_target[0] += 1e-7;
    let r = run(&fx);
    assert_eq!(failed(&r), vec!["S3.gross.tier2.identity".to_string(), "S3.net.tier2.identity".to_string()]);
    assert_eq!(r.sleeves[1].gross.cmp.tier3.unwrap().disagreeing, 0);
    // A block of grossly wrong targets breaks Tier III as well.
    let mut fx = fixtures();
    for b in fx.s3.bars.iter_mut().take(40) {
        b.w_target[0] += 0.5;
        b.w_target[1] += 0.5;
    }
    let r = run(&fx);
    let f = failed(&r);
    for name in ["S3.gross.tier3.weights", "S3.net.tier3.weights", "S3.gross.tier2.identity", "S3.net.tier2.identity"] {
        assert!(f.contains(&name.to_string()), "{name} should fail, got {f:?}");
    }
    let t3 = r.sleeves[1].gross.cmp.tier3.unwrap();
    assert!(!t3.pass && t3.agreement < 0.98);
}

#[test]
fn tier1_bands_fail_on_a_wrong_key_and_the_verdict_names_them() {
    // Scale the whole S3 gross key by 1.3: correlation stays 1, Sharpe and CAGR move far outside the bands.
    let mut fx = fixtures();
    for b in fx.s3.bars.iter_mut() {
        b.ret_gross *= 1.3;
    }
    let r = run(&fx);
    let f = failed(&r);
    assert!(f.contains(&"S3.gross.tier1.bands".to_string()), "{f:?}");
    assert!(f.contains(&"S3.gross.tier2.identity".to_string()));
    assert!(!f.contains(&"S3.net.tier1.bands".to_string()), "the net key was not touched");
    // Decorrelate S1 net: the correlation band fails.
    let mut fx = fixtures();
    let n = fx.s1.bars.len();
    let rev: Vec<f64> = fx.s1.bars.iter().rev().map(|b| b.ret_net).collect();
    for (i, r) in rev.into_iter().enumerate().take(n) {
        fx.s1.bars[i].ret_net = r;
    }
    let f = failed(&run(&fx));
    assert!(f.contains(&"S1.net.tier1.bands".to_string()) && !f.contains(&"S1.gross.tier1.bands".to_string()), "{f:?}");
}

#[test]
fn trades_band_is_five_percent_and_the_exact_counter_is_reported_separately() {
    // key 21 vs run 22: 4.76% is inside the band, but the exact counter differs.
    let mut fx = fixtures();
    let real = fx.s3.flips;
    fx.s3.flips = real - 1;
    let r = run(&fx);
    assert_eq!(failed(&r), vec!["S3.trades.exact".to_string()], "flips {real} -> {}", real - 1);
    assert!(r.sleeves[1].gross.tier1_pass && r.sleeves[1].net.tier1_pass, "4.76% is inside the Tier I trades band");
    // A 20% gap breaks the band on both bases.
    let mut fx = fixtures();
    fx.s3.flips = real * 12 / 10 + 2;
    let r = run(&fx);
    let f = failed(&r);
    for name in ["S3.gross.tier1.trades", "S3.net.tier1.trades", "S3.trades.exact"] {
        assert!(f.contains(&name.to_string()), "{f:?}");
    }
    // the sleeve's Tier I verdict includes the trades band
    assert!(!r.sleeves[1].gross.tier1_pass && !r.sleeves[1].net.tier1_pass);
    assert!(r.sleeves[1].gross.cmp.bands_pass, "only the trades band is broken here");
}

#[test]
fn the_original_shadow_key_and_the_key_metrics_are_checked() {
    // Shadow saved returns (the ORIGINAL key): 2e-9 fails, 5e-10 is tolerated.
    let mut fx = fixtures();
    fx.shadow_s3[20].1 += 5e-10;
    assert!(run(&fx).passed());
    let mut fx = fixtures();
    fx.shadow_s3[20].1 += 2e-9;
    assert_eq!(failed(&run(&fx)), vec!["S3.gross.vs_shadow_saved".to_string()]);
    let mut fx = fixtures();
    fx.shadow_s1[20].1 += 2e-9;
    assert_eq!(failed(&run(&fx)), vec!["S1.gross.vs_shadow_saved".to_string()]);
    // A shadow file that covers other bars than the key is useless and must fail.
    let mut fx = fixtures();
    fx.shadow_s3.pop();
    assert_eq!(failed(&run(&fx)), vec!["S3.gross.vs_shadow_saved".to_string()]);
    let mut fx = fixtures();
    fx.shadow_s1.remove(0);
    assert_eq!(failed(&run(&fx)), vec!["S1.gross.vs_shadow_saved".to_string()]);
    // The key must equal its own recorded metrics.
    let mut fx = fixtures();
    fx.recorded_s3[1].sharpe += 1e-6;
    assert_eq!(failed(&run(&fx)), vec!["S3.key_self_consistency".to_string()]);
    let mut fx = fixtures();
    fx.recorded_s1[0].max_drawdown += 1e-6;
    assert_eq!(failed(&run(&fx)), vec!["S1.key_self_consistency".to_string()]);
    let mut fx = fixtures();
    fx.recorded_s1[0].obs += 1.0;
    assert_eq!(failed(&run(&fx)), vec!["S1.key_self_consistency".to_string()]);
}

#[test]
fn a_key_longer_than_the_simulated_window_fails_the_window_check() {
    let mut fx = fixtures();
    let mut extra = fx.s3.bars.last().unwrap().clone();
    extra.date = extra.date.add_days(30);
    fx.s3.bars.push(extra);
    let f = failed(&run(&fx));
    assert!(f.contains(&"S3.gross.window".to_string()) && f.contains(&"S3.net.window".to_string()), "{f:?}");
}

// -------------------------------------------------------------------------------------------------- Tier IV machinery

#[test]
fn every_field_of_mutants_json_is_compared() {
    type Tamper = fn(&mut weightsim_rules::ladder::fixtures::ExpectedMutant);
    let cases: Vec<(&str, Tamper)> = vec![
        ("caught_by drops a tier", |e| {
            e.caught_by.pop();
        }),
        ("caught_by gains a tier", |e| e.caught_by.push("tier3_weights".to_string())),
        ("escapes_tier1 flipped", |e| e.escapes_tier1 = !e.escapes_tier1),
        ("common_days", |e| e.common_days += 1),
        ("corr", |e| e.corr += 1e-5),
        ("d_sharpe", |e| e.d_sharpe += 1e-5),
        ("d_cagr_pp", |e| e.d_cagr_pp += 1e-4),
        ("mutant_sharpe", |e| e.mutant_sharpe += 1e-5),
        ("max_abs_return_diff", |e| e.max_abs_return_diff *= 1.01),
        ("w_target diff removed", |e| e.max_abs_w_target_diff = None),
        ("w_held diff off", |e| e.max_abs_w_held_diff = e.max_abs_w_held_diff.map(|v| v + 1e-3)),
        ("tier3 cells", |e| e.tier3 = e.tier3.map(|(a, d, c)| (a, d, c + 1))),
        ("tier3 disagreeing", |e| e.tier3 = e.tier3.map(|(a, d, c)| (a, d + 1, c))),
        ("tier3 agreement", |e| e.tier3 = e.tier3.map(|(a, d, c)| (a + 1e-4, d, c))),
    ];
    let names = ["s1_one_bar_late", "s3_same_day_peek"];
    for name in names {
        for (label, tamper) in &cases {
            let mut fx = fixtures();
            let e = fx.expected_mutants.iter_mut().find(|e| e.name == name).unwrap();
            let before = e.clone();
            tamper(e);
            if *e == before {
                continue; // e.g. tier3 tamper on a mutant that has none
            }
            let r = run(&fx);
            assert_eq!(failed(&r), vec![format!("mutant.{name}.matches_mutants_json")], "{name}: {label}");
        }
    }
    // Rounding-level noise (the json keeps 6 decimals) is tolerated.
    let mut fx = fixtures();
    for e in fx.expected_mutants.iter_mut() {
        e.corr += 1e-7;
        e.d_sharpe -= 1e-7;
        e.d_cagr_pp += 1e-6;
        e.mutant_sharpe += 1e-7;
    }
    assert!(run(&fx).passed());
}

#[test]
fn a_mutant_with_no_entry_in_mutants_json_is_a_failure() {
    let mut fx = fixtures();
    fx.expected_mutants.retain(|e| e.name != "s3_half_sizing");
    assert_eq!(failed(&run(&fx)), vec!["mutant.s3_half_sizing.matches_mutants_json".to_string()]);
}

#[test]
fn an_uncaught_mutant_fails_certification() {
    // Make the S1 key BE the wrong-rebalance-mode mutant's own output: that mutant then passes every tier, which is
    // exactly the situation Tier IV exists to flag (a certification that cannot tell the mutant from the truth).
    let fx0 = fixtures();
    let mrun = weightsim_rules::ladder::mutants::run_mutant(&fx0, Mutant::S1WrongRebalanceMode).unwrap();
    assert_eq!(mrun.rows.dates.len(), fx0.s1.bars.len());
    let mut fx = fx0.clone();
    for (i, b) in fx.s1.bars.iter_mut().enumerate() {
        assert_eq!(b.date, mrun.rows.dates[i]);
        b.ret_gross = mrun.rows.ret[i];
        b.w_target = mrun.rows.w_target.as_ref().unwrap()[i].clone();
        b.w_held = mrun.rows.w_held.as_ref().unwrap()[i].clone();
    }
    let r = run_ladder_with(&fx, &NO_CANARIES).unwrap();
    let f = failed(&r);
    for name in [
        "mutant.s1_daily_rebalanced_20.caught",
        "mutant.s1_daily_rebalanced_20.tier2_catches_band_escaper",
        "tier4.all_mutants_caught",
    ] {
        assert!(f.contains(&name.to_string()), "{name} should fail, got {f:?}");
    }
    assert!(!r.mutants.iter().find(|m| m.name == "s1_daily_rebalanced_20").unwrap().caught());
    assert!(!r.passed());
    // The honest key still catches it.
    assert!(check(&run(&fx0), "mutant.s1_daily_rebalanced_20.caught").passed);
}

// ----------------------------------------------------------------------------------------- fixtures: pins and files

#[test]
fn the_real_pins_are_the_amendment_11_values() {
    assert_eq!(REAL_MANIFEST_SHA256, "d51444c33fc2630e12315e936f79580e13dca834ca0fdbc40a242ef496bd6d90");
    assert_eq!(REAL_CANDLES_SHA256, "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365");
    assert_eq!(Pins::REAL.manifest_sha256, REAL_MANIFEST_SHA256);
    assert_eq!(Pins::REAL.candles_sha256, REAL_CANDLES_SHA256);
}

fn expect_pin_error(r: Result<Fixtures, LadderError>, what: &str) {
    match r {
        Err(LadderError::Pin(m)) => println!("{what}: {m}"),
        other => panic!("{what}: expected a pin error, got {:?}", other.map(|_| "Ok")),
    }
}

#[test]
fn the_manifest_trust_chain_rejects_every_kind_of_tampering() {
    let (files, ms, cs) = ready();
    // 1. wrong manifest pin
    expect_pin_error(load(files, &"0".repeat(64), cs), "wrong manifest pin");
    // 2. wrong candles pin
    expect_pin_error(load(files, ms, &"0".repeat(64)), "wrong candles pin");
    // 3. a changed byte in a listed file (the manifest itself is untouched, so its pin still matches)
    for name in [
        "ladder_candles.csv",
        "key/S1_etf_trend_faber_perbar.csv",
        "key/S3_crypto_trend_100d_perbar.csv",
        "key/key_metrics.json",
        "key/mutants.json",
        "shadow_saved/shadow_S1_daily_returns.csv",
        "shadow_saved/shadow_S3_daily_returns.csv",
    ] {
        let mut f = SyntheticFiles { files: files.files.clone() };
        let bytes = f.files.get_mut(name).unwrap();
        let last = bytes.len() - 2;
        bytes[last] = if bytes[last] == b'1' { b'2' } else { b'1' };
        expect_pin_error(load(&f, ms, cs), name);
    }
    // 4. a changed manifest (pin mismatch)
    let mut f = SyntheticFiles { files: files.files.clone() };
    let m = f.files.get_mut("MANIFEST.json").unwrap();
    m.extend_from_slice(b"\n");
    expect_pin_error(load(&f, ms, cs), "manifest changed");
    // 5. a required file that the manifest does not list (manifest re-pinned, so only the required-file rule can object)
    let mut f = SyntheticFiles { files: files.files.clone() };
    f.files.remove("key/mutants.json");
    let (f, ms2, cs2) = with_manifest(f);
    expect_pin_error(load(&f, &ms2, &cs2), "required file not listed");
    // 6. wrong size in the manifest
    let mut f = SyntheticFiles { files: files.files.clone() };
    let text = String::from_utf8(f.files["MANIFEST.json"].clone()).unwrap();
    let name = "key/key_metrics.json";
    let len = f.files[name].len();
    let patched = text.replacen(&format!("\"bytes\": {len}"), &format!("\"bytes\": {}", len + 1), 1);
    assert_ne!(patched, text);
    f.files.insert("MANIFEST.json".into(), patched.clone().into_bytes());
    let ms3 = weightsim::sha256_hex(patched.as_bytes());
    expect_pin_error(load(&f, &ms3, cs), "wrong size");
    // 7. a listed file that is missing
    let mut f = SyntheticFiles { files: files.files.clone() };
    f.files.remove("key/S1_etf_trend_faber_perbar.csv");
    match load(&f, ms, cs) {
        Err(LadderError::Io(_)) => {}
        other => panic!("expected an i/o error, got {:?}", other.map(|_| "Ok")),
    }
}

#[test]
fn an_unlisted_extra_file_is_ignored_and_a_listed_unrequired_file_is_still_verified() {
    let (files, _, _) = ready();
    // A listed file the ladder does not need (like the real manifest's python files) is still hash-checked.
    let mut f = SyntheticFiles { files: files.files.clone() };
    f.files.insert("export_key.py".into(), b"print('key')\n".to_vec());
    let (mut f, ms, cs) = with_manifest(f);
    assert!(load(&f, &ms, &cs).is_ok());
    f.files.insert("export_key.py".into(), b"print('other')\n".to_vec());
    expect_pin_error(load(&f, &ms, &cs), "tampered unrequired file");
}

#[test]
fn from_dir_reads_the_same_bytes_and_the_real_pins_are_enforced() {
    let (files, ms, cs) = ready();
    let dir = std::env::temp_dir().join(format!("wsr-ladder-{}-{}", std::process::id(), ms.get(..8).unwrap()));
    let _ = std::fs::remove_dir_all(&dir);
    for (name, bytes) in &files.files {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }
    let fx = Fixtures::from_dir_with_pins(&dir, Pins { manifest_sha256: ms, candles_sha256: cs }).unwrap();
    assert_eq!(fx.manifest_sha256, *ms);
    assert_eq!(fx.candles_sha256, *cs);
    assert_eq!(
        fx.verified_files.len(),
        files.files.len() - 1,
        "every manifest entry (all files but the manifest itself)"
    );
    assert!(run(&fx).passed());
    // The synthetic set is not the real one: the REAL pins must refuse it.
    expect_pin_error(Fixtures::from_dir(&dir), "real pins on synthetic files");
    match self_test_dir(&dir) {
        Err(SelfTestError::Infrastructure(LadderError::Pin(_))) => {}
        other => panic!("expected an infrastructure pin error, got {:?}", other.map(|_| "Ok")),
    }
    // A missing directory is an i/o error, not a panic.
    assert!(matches!(Fixtures::from_dir(&dir.join("nope")), Err(LadderError::Io(_))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn key_layout_errors_are_parse_errors_not_panics() {
    let (files, _, _) = ready();
    let base = String::from_utf8(files.files["key/S3_crypto_trend_100d_perbar.csv"].clone()).unwrap();
    let edit_field = |line: usize, field: usize, value: &str| -> String {
        let mut lines: Vec<String> = base.lines().map(str::to_string).collect();
        let mut f: Vec<String> = lines[line].split(',').map(str::to_string).collect();
        f[field] = value.to_string();
        lines[line] = f.join(",");
        lines.join("\n") + "\n"
    };
    let cases: Vec<(&str, String)> = vec![
        ("missing column", base.replacen("w_held_ETH", "w_held_XXX", 1)),
        ("bad number", edit_field(1, 1, "zero")),
        ("bad date", edit_field(2, 0, "2016-13-45")),
        ("dates not ascending", edit_field(2, 0, "2015-01-01")),
        ("short row", format!("{}\n2020-12-31,1,2\n", base.trim_end())),
        ("empty", String::new()),
        ("header only", format!("{}\n", base.lines().next().unwrap())),
        ("excluded bar", edit_field(1, 10, "1")),
    ];
    for (label, text) in cases {
        let mut f = SyntheticFiles { files: files.files.clone() };
        f.files.insert("key/S3_crypto_trend_100d_perbar.csv".into(), text.into_bytes());
        let (f, ms, cs) = with_manifest(f);
        match load(&f, &ms, &cs) {
            Err(LadderError::Parse(_)) | Err(LadderError::Inconsistent(_)) => {}
            other => panic!("{label}: expected a parse error, got {:?}", other.map(|_| "Ok")),
        }
    }
}
