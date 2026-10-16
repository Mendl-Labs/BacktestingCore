//! PF1 acceptance (b) on the ALWAYS-ON synthetic fixtures: the Rust book simulator reproduces, PER BAR and to the
//! pre-registered tolerance (AMENDMENT 12: 1e-9), the output of an independent Python engine that mirrors the PF0 book
//! key (`tests/book_fixtures/gen_book_key.py`; the mirror itself is proven bit-equal to the real key on the real data by
//! its `--verify-real` mode, so these fixtures are the key's semantics on synthetic prices, not a second opinion).
//!
//! Every column of every per-bar file is compared, gross and net: returns, equity, cost, turnover, cost over previous
//! equity, exposures, cash fraction, every `w_target_*` and `w_held_*`, `share_*`, `contrib_*`, `traded_gross_*`,
//! `shadow_ret_gross_*`, `shadow_ret_net_*`, and the exact flags (`run`, `refused`, `decision_*`, `planned_*`).
//! The identity is also non-vacuous by construction: the assertions at the end of each test check that the fixture
//! really contains the situations the configuration is about (refusals, allocator changes, binding capital cap, ...).
//! The real-data version of the same comparison is `tests/book_key_real.rs` (env-gated, run by the owner).

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::book::*;
use weightsim::*;

const TOL: f64 = 1e-9;

fn key_text(name: &str) -> &'static str {
    match name {
        "book_cert_60_40" => include_str!("book_fixtures/book_cert_60_40_perbar.csv"),
        "book_live_60_40" => include_str!("book_fixtures/book_live_60_40_perbar.csv"),
        "book_due_filter_60_40" => include_str!("book_fixtures/book_due_filter_60_40_perbar.csv"),
        "book_scaled_50_30" => include_str!("book_fixtures/book_scaled_50_30_perbar.csv"),
        "book_grosscap_60_40" => include_str!("book_fixtures/book_grosscap_60_40_perbar.csv"),
        "book_invvol" => include_str!("book_fixtures/book_invvol_perbar.csv"),
        "single_etf_everybar_filter" => include_str!("book_fixtures/single_etf_everybar_filter_perbar.csv"),
        "synthetic_netting" => include_str!("book_fixtures/synthetic_netting_perbar.csv"),
        other => panic!("no fixture {other}"),
    }
}

fn run_case(c: &Case) -> (BookResult, BookResult) {
    let (panel, book, cfg) = build_case(c);
    simulate_book_gross_and_net(&panel, &book, &cfg).unwrap_or_else(|e| panic!("{}: {e}", c.name))
}

fn report(name: &str, k: &KeyCompare) {
    println!(
        "KEYDIFF {name:28} rows {:4} cells {:6} not-bit-identical {:5} max|diff| {:.3e} (worst column {}) flags exact: {}",
        k.rows,
        k.cells,
        k.cells_not_bit_identical,
        k.worst,
        k.worst_col,
        k.worst_flag == 0.0
    );
}

fn check_against_key(c: &Case) -> (BookResult, BookResult) {
    let (g, n) = run_case(c);
    let key = parse_csv(key_text(c.name));
    let k = compare_book_to_key(&key, &g, &n);
    report(c.name, &k);
    assert!(k.worst <= TOL, "{}: max |diff| {:.3e} in column {} exceeds {TOL:e}", c.name, k.worst, k.worst_col);
    assert_eq!(k.worst_flag, 0.0, "{}: flags (run/refused/decision/planned) must match exactly", c.name);
    for (name, w) in &k.by_col {
        assert!(*w <= TOL, "{}: column {name} differs by {w:e}", c.name);
    }
    (g, n)
}

#[test]
fn certification_book_matches_the_key_per_bar_gross_and_net() {
    let (g, n) = check_against_key(&case("book_cert_60_40"));
    // non-vacuity: both sleeves trade, the ETF sleeve has carry rows, gross and net differ
    assert!(g.sleeve_open.chunks(2).filter(|s| !s[0]).count() > 20, "ETF weekend/holiday carry rows exist");
    assert!(n.total_cost() > 0.0 && g.total_cost() == 0.0);
    assert!(g.refusals_with_code("gross_above_cap").is_empty());
}

#[test]
fn live_faithful_book_matches_the_key_per_bar_gross_and_net() {
    let (g, _n) = check_against_key(&case("book_live_60_40"));
    let etf_planned = g.planned.chunks(2).filter(|p| p[0]).count();
    assert!(etf_planned > 60, "AllSleevesOnAnyDue plans the ETF sleeve whenever it is open");
}

#[test]
fn only_due_with_filter_and_budget_matches_the_key() {
    let (g, _) = check_against_key(&case("book_due_filter_60_40"));
    let planned = (1..g.n_bars()).filter(|&k| g.planned[k * 2]).count();
    let decided = (1..g.n_bars()).filter(|&k| g.decision[k * 2]).count();
    assert!(decided >= 3, "the ETF sleeve decided inside the window");
    assert_eq!(planned, decided, "PerSleeve cadence plans the monthly sleeve only on its decision bars");
}

#[test]
fn scaled_book_with_binding_allocated_capital_matches_the_key() {
    let (g, _) = check_against_key(&case("book_scaled_50_30"));
    let alloc = meta_f("allocated_capital_currency") / CAPITAL0;
    let binding = (1..g.n_bars()).filter(|&k| g.equity_pre[k] > alloc).count();
    let free = (1..g.n_bars()).filter(|&k| g.equity_pre[k] <= alloc).count();
    assert!(
        binding > 5 && free > 5,
        "capital base = min(equity, allocated) is exercised on both sides: {binding}/{free}"
    );
    assert!(g.risk_scale.iter().all(|&r| r == 0.8));
}

#[test]
fn gross_cap_book_refuses_the_whole_book_and_matches_the_key() {
    let (g, n) = check_against_key(&case("book_grosscap_60_40"));
    let refused = g.refusals_with_code("gross_above_cap");
    assert!(refused.len() > 5 && refused.len() < g.n_bars() - 5, "{} refusals", refused.len());
    assert_eq!(refused.len(), g.book_refused.iter().filter(|b| **b).count());
    for k in 0..g.n_bars() {
        if g.book_refused[k] {
            // hold-previous, whole book: nothing traded, no cost, units unchanged
            assert_eq!(g.traded_notional[k], 0.0);
            assert_eq!(n.cost[k], 0.0);
            if k > 0 {
                assert_eq!(g.inst_row(&g.units, k), g.inst_row(&g.units, k - 1));
            }
        }
    }
}

#[test]
fn frozen_inverse_vol_allocator_book_matches_the_key() {
    let (g, _) = check_against_key(&case("book_invvol"));
    let etf_share: Vec<f64> = (0..g.n_bars()).map(|k| g.share[k * 2]).collect();
    let mut distinct: Vec<u64> = etf_share.iter().map(|v| v.to_bits()).collect();
    distinct.sort();
    distinct.dedup();
    assert!(distinct.len() >= 3, "the allocator changed the shares: {distinct:?}");
    assert_eq!(etf_share[0], 0.5, "initial shares hold until a review has enough data");
    for k in 0..g.n_bars() {
        assert!((g.share[k * 2] + g.share[k * 2 + 1] - 1.0).abs() < 1e-12, "shares sum to the total");
    }
}

#[test]
fn single_sleeve_with_a_policy_override_and_the_trade_filter_matches_the_key() {
    let (g, _) = check_against_key(&case("single_etf_everybar_filter"));
    assert!(
        g.planned.iter().filter(|p| **p).count() > 60,
        "EveryBar policy override plans the ETF sleeve on every open bar"
    );
}

#[test]
fn netting_book_with_shared_instruments_matches_the_key() {
    let panel = netting_panel();
    let book = netting_book(&panel);
    let (g, n) = simulate_book_gross_and_net(&panel, &book, &netting_config()).unwrap();
    let key = parse_csv(key_text("synthetic_netting"));
    let k = compare_book_to_key(&key, &g, &n);
    report("synthetic_netting", &k);
    assert!(k.worst <= TOL && k.worst_flag == 0.0);
    // the netted target of the shared instrument X is share-weighted signed (0.5*0.6 + 0.5*(-0.8) = -0.1), not |0.3|+|0.4|
    let x = g.instruments.iter().position(|s| s == "X").unwrap();
    assert!((g.target_weights[x] - (0.5 * 0.6 + 0.5 * -0.8)).abs() < 1e-15, "{}", g.target_weights[x]);
    // no per-sleeve contribution columns exist for shared instruments in the key; ours still sums to the total
    assert!(g.attribution_report().max_abs_err_contrib_sum <= 1e-12);
}

#[test]
fn the_test_rules_reproduce_the_python_decisions_bit_for_bit() {
    // The generator's Python rules and the Rust test rules must agree on every decision (else the identity above would
    // be comparing two different strategies).
    let panel = key_book_panel();
    let (g, _) = run_case(&case("book_cert_60_40"));
    let mut want: Vec<(String, String, Vec<f64>)> = Vec::new();
    for line in BOOK_DECISIONS.lines().skip(1) {
        let p: Vec<&str> = line.split(',').collect();
        want.push((p[0].to_string(), p[1].to_string(), p[2..].iter().map(|x| x.parse().unwrap()).collect()));
    }
    let start = meta("start_bar");
    let end = meta("end");
    // decisions dated inside the account window appear as decision flags with the same dates
    for (sid, s) in [("etf", 0usize), ("cry", 1usize)] {
        let want_dates: Vec<String> =
            want.iter().filter(|w| w.0 == sid && w.1 > start && w.1 <= end).map(|w| w.1.clone()).collect();
        let got_dates: Vec<String> =
            (1..g.n_bars()).filter(|&k| g.decision[k * 2 + s]).map(|k| g.times[k].date().to_string()).collect();
        assert_eq!(want_dates, got_dates, "{sid} decision dates");
    }
    // and the standing targets equal the python weights: check the ETF sleeve targets on its decision bars
    let e0 = panel.instrument_index("E1").unwrap();
    for w in want.iter().filter(|w| w.0 == "etf" && w.1 > start && w.1 <= end) {
        let k = (1..g.n_bars()).find(|&k| g.times[k].date().to_string() == w.1).unwrap();
        for i in 0..3 {
            let got = g.target_weights[k * g.n_instruments() + e0 + i];
            let expect = 0.6 * w.2[i];
            assert_eq!(got.to_bits(), (expect * 1.0).to_bits(), "ETF target on {} instrument {i}", w.1);
        }
    }
}

#[test]
fn cadence_modes_reproduce_the_f1_numbers_on_the_synthetic_fixture() {
    let (gx, nx) = run_case(&case("book_due_filter_60_40"));
    let (gy, ny) = run_case(&case("book_live_60_40"));
    let got = f1_stats((&gx, &nx), (&gy, &ny), &E);
    let mut want: Vec<(String, f64)> = Vec::new();
    for line in BOOK_F1.lines().skip(1) {
        let p: Vec<&str> = line.split(',').collect();
        want.push((p[0].to_string(), p[1].parse().unwrap()));
    }
    assert_eq!(got.len(), want.len());
    for ((name, mine), (wname, theirs)) in got.iter().zip(&want) {
        assert_eq!(*name, wname.as_str());
        let is_count = name.contains("bars") || *name == "bars";
        let tol = if is_count { 0.0 } else { 1e-9 * theirs.abs().max(1.0) };
        println!("F1 {name:32} rust {mine:<24} python {theirs}");
        assert!((mine - theirs).abs() <= tol, "F1 {name}: {mine} vs {theirs}");
    }
    // the substance of finding F1: same decisions, different cadence => the ETF sleeve is planned far more often
    let get = |k: &str| got.iter().find(|(n, _)| *n == k).unwrap().1;
    assert!(get("etf_planned_bars_Y") > 15.0 * get("etf_planned_bars_X"));
    assert!(get("etf_weight_gap_bars_gt_1e-9") > 20.0, "the two modes are different accounts");
    assert!(
        get("ret_net_corr") > 0.99,
        "and invisible to the return-correlation band: only the per-bar identity sees them"
    );
}

#[test]
fn book_fixtures_are_pinned_by_their_own_manifest() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/book_fixtures");
    let manifest = std::fs::read_to_string(format!("{dir}/MANIFEST.sha256")).unwrap();
    let mut listed = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let (hash, name) = line.split_once("  ").expect("`<sha256>  <file>` format");
        let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
        assert_eq!(sha256_hex(&bytes), hash, "fixture {name} changed: regenerate with gen_book_key.py deliberately");
        listed.push(name.to_string());
    }
    for e in std::fs::read_dir(dir).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().to_string();
        if name == "MANIFEST.sha256" || name.starts_with('.') || name == "__pycache__" {
            continue;
        }
        assert!(listed.contains(&name), "fixture {name} is not pinned in MANIFEST.sha256");
    }
    assert!(listed.len() >= 14 && listed.contains(&"gen_book_key.py".to_string()));
}
