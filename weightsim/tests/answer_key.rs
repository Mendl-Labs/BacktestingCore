//! (a) Per-bar identity with the pinned ANSWER KEY (`shadow.py`), using hand-written test rules.
//!
//! Two tiers of evidence:
//!  * always run: the synthetic-panel key in `tests/fixtures/` (generated ONCE by `gen_answer_key.py` from the pinned
//!    `shadow.py` logic; hashes pinned in `MANIFEST.sha256`);
//!  * env-gated: the REAL pinned ladder fixture and the recorded `shadow_S1/S3_daily_returns.csv` (vendor-derived, so
//!    not copied into this repo). Set `WEIGHTSIM_LADDER_DIR` to the `replication_ladder` directory; otherwise the test
//!    prints SKIPPED.

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::*;

const TOL: f64 = 1e-9;

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/").to_string() + name)
        .unwrap_or_else(|e| panic!("cannot read fixture {name}: {e}"))
}

#[test]
fn manifest_pins_every_committed_fixture_by_sha256() {
    let manifest = String::from_utf8(fixture_bytes("MANIFEST.sha256")).unwrap();
    let mut listed = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let (hash, name) = line.split_once("  ").expect("`<sha256>  <file>` format");
        assert_eq!(
            sha256_hex(&fixture_bytes(name)),
            hash,
            "fixture {name} changed: regenerate deliberately, not by accident"
        );
        listed.push(name.to_string());
    }
    // Every fixture file in the directory (other than the manifest itself and .gitattributes) must be listed.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    for e in std::fs::read_dir(dir).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().to_string();
        if name == "MANIFEST.sha256" || name.starts_with('.') {
            continue;
        }
        assert!(listed.contains(&name), "fixture {name} is not pinned in MANIFEST.sha256");
    }
    assert!(listed.len() >= 7);
}

#[test]
fn s1_test_rule_matches_key_per_bar_returns_and_equity() {
    let panel = s1_panel();
    let sim = simulate(&panel, &S1TestRule, &s1_config()).unwrap();
    let (kd, kr, ke) = parse_returns(KEY_S1_RETURNS);

    assert_eq!(sim.window_dates(), kd.as_slice(), "counted bars must be exactly the key's bars");
    let dr = max_abs_diff(sim.window_returns(), &kr);
    let de = max_abs_diff(&sim.window_equity(), &ke);
    println!("S1 synthetic: {} bars, max|ret diff| = {dr:e}, max|equity diff| = {de:e}", kd.len());
    assert!(dr <= TOL, "per-bar return identity broken: {dr:e}");
    assert!(de <= TOL, "per-bar equity identity broken: {de:e}");

    // The simulator's own (un-rebased) equity path IS the key's cumprod: S1 starts at 1.0 on the first fill bar.
    let w = sim.window.unwrap();
    assert_eq!(sim.equity[w.first_bar - 1], 1.0);
    assert!(max_abs_diff(&sim.equity[w.first_bar..=w.last_bar], &ke) <= TOL);
    // Zero cost: this run's return equals its pre-cost return, and nothing was charged.
    assert!(sim.cost.iter().all(|&c| c == 0.0));
    assert_eq!(sim.ret, sim.ret_pre_cost);
}

#[test]
fn s1_test_rule_matches_key_signals_weights_and_month_end_dates() {
    let panel = s1_panel();
    let sim = simulate(&panel, &S1TestRule, &s1_config()).unwrap();
    let (sd, srows) = parse_signals(KEY_S1_SIGNALS);
    let decision_dates: Vec<Date> = (0..sim.n_bars()).filter(|&t| sim.decision[t]).map(|t| sim.dates[t]).collect();
    // Month-ends come from the calendar; the key derived them independently with pandas groupby-last.
    assert_eq!(decision_dates, sd, "decision dates must equal the key's month-end signal dates");
    let mut cells = 0;
    let mut agree = 0;
    let mut max_diff = 0.0f64;
    for (date, row) in sd.iter().zip(&srows) {
        let t = sim.dates.iter().position(|x| x == date).unwrap();
        for (i, s) in row.iter().enumerate() {
            let want = 0.2 * s;
            let got = sim.row(&sim.target_weights, t)[i];
            max_diff = max_diff.max((got - want).abs());
            cells += 1;
            if (got - want).abs() <= 1e-6 {
                agree += 1;
            }
        }
    }
    println!("S1 synthetic weights: {agree}/{cells} cells agree, max diff {max_diff:e}");
    assert!(max_diff <= 1e-12);
    assert_eq!(agree, cells);
    // Tier-III style agreement is 1.0 here (it is only a floor; the identity above is the discriminator).
    assert!(agree as f64 / cells as f64 >= 0.98);
}

#[test]
fn s1_test_rule_metrics_and_flips_match_key() {
    let panel = s1_panel();
    let sim = simulate(&panel, &S1TestRule, &s1_config()).unwrap();
    let m = sim.metrics().unwrap();
    assert_eq!(m.n as f64, key_metric("S1", "obs"));
    assert!((m.years - key_metric("S1", "years")).abs() < 1e-12);
    assert!((m.ppy - key_metric("S1", "ppy")).abs() < 1e-9);
    for (name, got) in [
        ("cagr", m.cagr),
        ("vol", m.vol),
        ("sharpe", m.sharpe),
        ("max_drawdown", m.max_drawdown),
        ("final_equity", m.final_equity),
    ] {
        let want = key_metric("S1", name);
        assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "S1 {name}: got {got}, key {want}");
    }
    // The shadow's own rounded output (`shadow.metrics()`), rounded the same way.
    assert_eq!((m.cagr * 1e4).round() / 1e4, key_metric("S1", "cagr_shadow_rounded"));
    assert_eq!((m.sharpe * 1e3).round() / 1e3, key_metric("S1", "sharpe_shadow_rounded"));
    assert_eq!((m.max_drawdown * 1e4).round() / 1e4, key_metric("S1", "max_drawdown_shadow_rounded"));
    let flips: u64 = sim.signal_flips.iter().sum();
    assert_eq!(flips as f64, key_metric("S1", "flips"));
}

#[test]
fn s3_test_rule_matches_key_per_bar_returns_and_equity() {
    let panel = s3_panel();
    let sim = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    let (kd, kr, ke) = parse_returns(KEY_S3_RETURNS);
    assert_eq!(sim.window_dates(), kd.as_slice(), "counted bars must be exactly the key's bars (incl. the ETH gap)");
    let dr = max_abs_diff(sim.window_returns(), &kr);
    let de = max_abs_diff(&sim.window_equity(), &ke);
    println!("S3 synthetic: {} bars, max|ret diff| = {dr:e}, max|equity diff| = {de:e}", kd.len());
    assert!(dr <= TOL, "per-bar return identity broken: {dr:e}");
    assert!(de <= TOL, "per-bar equity identity broken: {de:e}");
    assert_eq!(sim.ret, sim.ret_pre_cost);
}

#[test]
fn s3_test_rule_matches_key_signals_weights_metrics_and_flips() {
    let panel = s3_panel();
    let sim = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    let (sd, srows) = parse_signals(KEY_S3_SIGNALS);
    let mut max_diff = 0.0f64;
    for (date, row) in sd.iter().zip(&srows) {
        let t = sim.dates.iter().position(|x| x == date).unwrap();
        assert!(sim.decision[t], "Daily schedule decides on every bar with enough history");
        for (i, s) in row.iter().enumerate() {
            max_diff = max_diff.max((sim.row(&sim.target_weights, t)[i] - 0.5 * s).abs());
        }
    }
    assert!(max_diff <= 1e-12, "weights differ from the key by {max_diff:e}");
    let m = sim.metrics().unwrap();
    assert_eq!(m.n as f64, key_metric("S3", "obs"));
    for (name, got) in
        [("cagr", m.cagr), ("vol", m.vol), ("sharpe", m.sharpe), ("max_drawdown", m.max_drawdown), ("ppy", m.ppy)]
    {
        let want = key_metric("S3", name);
        assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "S3 {name}: got {got}, key {want}");
    }
    assert_eq!((m.sharpe * 1e3).round() / 1e3, key_metric("S3", "sharpe_shadow_rounded"));
    let flips: u64 = sim.signal_flips.iter().sum();
    assert_eq!(flips as f64, key_metric("S3", "flips"));
}

#[test]
fn key_fixtures_are_non_trivial_so_the_identity_is_not_vacuous() {
    // Guard against a degenerate key (e.g. always flat) that any simulator would match.
    let (_, r1, _) = parse_returns(KEY_S1_RETURNS);
    let (_, r3, _) = parse_returns(KEY_S3_RETURNS);
    let (n1, n3) = (r1.iter().filter(|x| **x != 0.0).count(), r3.iter().filter(|x| **x != 0.0).count());
    println!("key non-zero return days: S1 {n1}/{}, S3 {n3}/{}", r1.len(), r3.len());
    assert!(n1 > 200, "S1 key is invested on too few days: {n1}");
    assert!(n3 > 100, "S3 key is invested on too few days: {n3}");
    assert!(key_metric("S1", "flips") >= 10.0 && key_metric("S3", "flips") >= 10.0);
    assert!(r3.contains(&0.0), "the S3 key must include flat (out-of-market) days");
}

// ------------------------------------------------------------------------------ real pinned key (env-gated)

const LADDER_SHA: &str = "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365";
const S1_RET_SHA: &str = "bbccd21e6a37ad53bb5b747ec79bb3980cf161509893a8dda03de043ad84881d";
const S3_RET_SHA: &str = "5390d6c8c02cd17b629c5ae0645fe3a6fc385af833f8aa08de71e92448c37d87";
const S1_SIG_SHA: &str = "81717083558de340c200e9fd55b10ce4b6b14e214ca5269530f24317f98476ab";
const S3_SIG_SHA: &str = "ced88d667dfe71347a41dc30f9a8a52400c34965c37fe7b353fa036525397035";

fn read_pinned(dir: &str, name: &str, sha: &str) -> Vec<u8> {
    let bytes = std::fs::read(format!("{dir}/{name}")).unwrap_or_else(|e| panic!("cannot read {dir}/{name}: {e}"));
    assert_eq!(sha256_hex(&bytes), sha, "{name} does not match its pinned sha256");
    bytes
}

fn round_to(x: f64, dp: i32) -> f64 {
    let f = 10f64.powi(dp);
    (x * f).round() / f
}

#[test]
fn real_ladder_matches_recorded_shadow_files_per_bar() {
    let dir = match std::env::var("WEIGHTSIM_LADDER_DIR") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            println!("SKIPPED real_ladder_matches_recorded_shadow_files_per_bar: set WEIGHTSIM_LADDER_DIR to the replication_ladder directory (vendor data is not copied into this repo)");
            return;
        }
    };
    let csv = read_pinned(&dir, "ladder_candles.csv", LADDER_SHA);
    let s1_ret = String::from_utf8(read_pinned(&dir, "shadow_S1_daily_returns.csv", S1_RET_SHA)).unwrap();
    let s3_ret = String::from_utf8(read_pinned(&dir, "shadow_S3_daily_returns.csv", S3_RET_SHA)).unwrap();
    let s1_sig = String::from_utf8(read_pinned(&dir, "shadow_S1_monthend_signals.csv", S1_SIG_SHA)).unwrap();
    let s3_sig = String::from_utf8(read_pinned(&dir, "shadow_S3_daily_signals.csv", S3_SIG_SHA)).unwrap();
    let summary = std::fs::read_to_string(format!("{dir}/shadow_summary.json")).unwrap();

    // Load through the fixture source, which verifies the pin before parsing.
    let etf = PriceSource::Fixture { csv: &csv, expected_sha256: LADDER_SHA }.load(&ETF).unwrap();
    let cry = PriceSource::Fixture { csv: &csv, expected_sha256: LADDER_SHA }.load(&CRY).unwrap();

    // ---- S1
    let sim1 = simulate(&etf, &S1TestRule, &s1_config()).unwrap();
    let (kd, kr, _) = parse_returns(&s1_ret);
    assert_eq!(sim1.window_dates(), kd.as_slice());
    let dr = max_abs_diff(sim1.window_returns(), &kr);
    let ke = weightsim::metrics::cumprod_one_plus(&kr);
    let de = max_abs_diff(&sim1.window_equity(), &ke);
    println!(
        "REAL S1: {} bars {} .. {}, max|ret diff| = {dr:e}, max|equity diff| = {de:e}",
        kd.len(),
        kd[0],
        kd[kd.len() - 1]
    );
    assert!(dr <= TOL && de <= TOL);
    let flips1: u64 = sim1.signal_flips.iter().sum();
    assert_eq!(flips1, 99, "S1 signal_flips per shadow_summary.json");
    let m1 = sim1.metrics().unwrap();
    assert_eq!(m1.n, 2303);
    assert_eq!(round_to(m1.ppy, 1), 251.4);
    assert_eq!(round_to(m1.cagr, 4), 0.0414);
    assert_eq!(round_to(m1.vol, 4), 0.0663);
    assert_eq!(round_to(m1.sharpe, 3), 0.645);
    assert_eq!(round_to(m1.max_drawdown, 4), -0.0897);
    assert!(summary.contains("\"asset_trades\": 99"));
    // month-end decision dates and weights vs the recorded signal file
    let (sd, srows) = parse_signals(&s1_sig);
    let dec: Vec<Date> = (0..sim1.n_bars()).filter(|&t| sim1.decision[t]).map(|t| sim1.dates[t]).collect();
    assert_eq!(dec, sd, "S1 decision dates == recorded month-end signal dates");
    let (mut cells, mut agree) = (0usize, 0usize);
    for (date, row) in sd.iter().zip(&srows) {
        let t = sim1.dates.iter().position(|x| x == date).unwrap();
        for (i, s) in row.iter().enumerate() {
            cells += 1;
            if (sim1.row(&sim1.target_weights, t)[i] - 0.2 * s).abs() <= 1e-6 {
                agree += 1;
            }
        }
    }
    println!("REAL S1 weights: {agree}/{cells} cells agree");
    assert_eq!(agree, cells);

    // ---- S3
    let sim3 = simulate(&cry, &S3TestRule, &s3_config()).unwrap();
    let (kd, kr, _) = parse_returns(&s3_ret);
    assert_eq!(sim3.window_dates(), kd.as_slice());
    let dr = max_abs_diff(sim3.window_returns(), &kr);
    let ke = weightsim::metrics::cumprod_one_plus(&kr);
    let de = max_abs_diff(&sim3.window_equity(), &ke);
    let rel_de =
        sim3.window_equity().iter().zip(&ke).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0, f64::max);
    println!(
        "REAL S3: {} bars {} .. {}, max|ret diff| = {dr:e}, max|equity diff| = {de:e} (relative {rel_de:e})",
        kd.len(),
        kd[0],
        kd[kd.len() - 1]
    );
    assert!(dr <= TOL, "S3 return identity");
    assert!(de <= TOL, "S3 equity identity (absolute; equity reaches ~{})", ke[ke.len() - 1]);
    let flips3: u64 = sim3.signal_flips.iter().sum();
    assert_eq!(flips3, 138);
    let m3 = sim3.metrics().unwrap();
    assert_eq!(m3.n, 1827);
    assert_eq!(round_to(m3.ppy, 1), 365.5);
    assert_eq!(round_to(m3.cagr, 4), 2.1824);
    assert_eq!(round_to(m3.vol, 4), 0.6532);
    assert_eq!(round_to(m3.sharpe, 3), 2.1);
    assert_eq!(round_to(m3.max_drawdown, 4), -0.5082);
    let (sd, srows) = parse_signals(&s3_sig);
    let (mut cells, mut agree) = (0usize, 0usize);
    for (date, row) in sd.iter().zip(&srows) {
        let t = sim3.dates.iter().position(|x| x == date).unwrap();
        for (i, s) in row.iter().enumerate() {
            cells += 1;
            if (sim3.row(&sim3.target_weights, t)[i] - 0.5 * s).abs() <= 1e-6 {
                agree += 1;
            }
        }
    }
    println!("REAL S3 weights: {agree}/{cells} cells agree");
    assert_eq!(agree, cells);
    println!("REAL LADDER: S1 series sha256 {}", sim1.series_sha256);
    println!("REAL LADDER: S3 series sha256 {}", sim3.series_sha256);
}
