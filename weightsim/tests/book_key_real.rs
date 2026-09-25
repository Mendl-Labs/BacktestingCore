//! PF1 acceptance (b) on the REAL pinned data (env-gated; vendor-derived files are not copied into this public repo):
//! the Rust book simulator against the PF0 BOOK KEY (`replication_ladder_book/`, pre-registered in Amendment 12).
//!
//!   WEIGHTSIM_LADDER_DIR   = .../replication_ladder        (`ladder_candles.csv`, `key/*_decisions.csv`, T0 per-bar files)
//!   WEIGHTSIM_BOOK_KEY_DIR = .../replication_ladder_book   (`key/*_perbar.csv`, `MANIFEST.json`)
//!
//! Without both variables every test here prints `SKIPPED ...`. With them, every input is verified by sha256 against the
//! pins before use, the sleeve target weights are the T0 key's decisions (the book key does not re-derive rule logic: it
//! tests the account layer), and the following are compared:
//!  * the five per-bar book key files `book_cert_60_40`, `book_live_60_40`, `book_scaled_50_30`, `book_grosscap_60_40`,
//!    `book_invvol` plus `synthetic_netting`, every column, gross and net, tolerance 1e-9 (flags exact);
//!  * the one-sleeve book against the T0 per-bar files `S1_..._perbar.csv` and `S3_..._perbar.csv` (Amendment 12 section 2:
//!    the PF0 golden, measured bit-for-bit for the Python engine);
//!  * the finding-F1 cadence numbers of Amendment 12 section 3.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::book::*;
use common::*;
use weightsim::*;

const CANDLES_SHA: &str = "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365";
const TOL: f64 = 1e-9;

struct Real {
    ladder: String,
    book: String,
    candles: String,
}

fn real() -> Option<Real> {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let (ladder, book) = match (get("WEIGHTSIM_LADDER_DIR"), get("WEIGHTSIM_BOOK_KEY_DIR")) {
        (Some(a), Some(b)) => (a, b),
        _ => return None,
    };
    let candles = std::fs::read_to_string(format!("{ladder}/ladder_candles.csv")).expect("ladder_candles.csv");
    assert_eq!(sha256_hex(candles.as_bytes()), CANDLES_SHA, "ladder_candles.csv is not the pinned T0 fixture");
    Some(Real { ladder, book, candles })
}

macro_rules! skip_unless_real {
    ($name:literal) => {
        match real() {
            Some(r) => r,
            None => {
                println!(
                    "SKIPPED {}: set WEIGHTSIM_LADDER_DIR and WEIGHTSIM_BOOK_KEY_DIR (vendor data is not copied into this repo)",
                    $name
                );
                return;
            }
        }
    };
}

/// `"<name>": {"bytes": .., "role": "output", "sha256": "<64 hex>"}` -> the hash of `name` in the book MANIFEST.json.
fn manifest_sha(manifest: &str, name: &str) -> String {
    let at = manifest.find(&format!("\"{name}\"")).unwrap_or_else(|| panic!("{name} not in MANIFEST.json"));
    let rest = &manifest[at..];
    let s = rest.find("\"sha256\": \"").expect("sha256 field") + "\"sha256\": \"".len();
    rest[s..s + 64].to_string()
}

impl Real {
    fn book_file(&self, name: &str) -> String {
        let text =
            std::fs::read_to_string(format!("{}/key/{name}", self.book)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let manifest = std::fs::read_to_string(format!("{}/MANIFEST.json", self.book)).unwrap();
        assert_eq!(
            sha256_hex(text.as_bytes()),
            manifest_sha(&manifest, &format!("key/{name}")),
            "{name}: not the pinned key file"
        );
        text
    }
    fn t0_file(&self, name: &str) -> String {
        std::fs::read_to_string(format!("{}/key/{name}", self.ladder)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }
    /// ETF sessions declare closed every clock date on which that ETF has no bar (this test has no holiday table); the
    /// gap detector is therefore not exercised here (it is, on the synthetic fixtures).
    fn panel(&self) -> BookPanel {
        let all: Vec<(&str, SessionKind)> =
            ETF.iter().chain(CRY.iter()).map(|s| (*s, SessionKind::Continuous)).collect();
        let probe = BookPanel::from_long_csv(&self.candles, &all).unwrap();
        let mut inst: Vec<(&str, SessionKind)> = Vec::new();
        for s in ETF {
            let i = probe.instrument_index(s).unwrap();
            let closed: Vec<Date> =
                (0..probe.n_bars()).filter(|&u| probe.close(i)[u].is_none()).map(|u| probe.times()[u].date()).collect();
            inst.push((s, SessionKind::exchange("observed_closures", closed)));
        }
        for s in CRY {
            inst.push((s, SessionKind::Continuous));
        }
        let panel = BookPanel::from_long_csv(&self.candles, &inst).unwrap();
        let cry_end = Panel::from_long_csv(&self.candles, &CRY).unwrap().dates().last().copied().unwrap();
        truncate_after(&panel, &cry_end.to_string())
    }
    fn decisions(&self) -> (std::collections::HashMap<Date, Vec<f64>>, std::collections::HashMap<Date, Vec<f64>>) {
        (
            read_decisions(&self.t0_file("S1_etf_trend_faber_decisions.csv"), 5),
            read_decisions(&self.t0_file("S3_crypto_trend_100d_decisions.csv"), 2),
        )
    }
}

fn scripted(
    id: &'static str,
    uni: &[&'static str],
    schedule: DecisionSchedule,
    policy: RebalancePolicy,
    dec: &std::collections::HashMap<Date, Vec<f64>>,
) -> ScriptedRule {
    ScriptedRule { id, universe: uni.to_vec(), schedule, policy, decisions: dec.clone() }
}

fn real_book(panel: &BookPanel, c: &Case, r: &Real) -> (Book, BookConfig) {
    let (d1, d3) = r.decisions();
    let alloc = c.invvol_lookback.map(|l| AllocatorSpec::InverseVol { lookback_bars: l, total: 1.0 });
    let share = |v: f64| if alloc.is_some() { ShareSpec::Allocated { initial: v } } else { ShareSpec::Fixed(v) };
    let etf = scripted("etf_trend_faber", &ETF, DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision, &d1);
    let cry = scripted("crypto_trend_100d", &CRY, DecisionSchedule::Daily, RebalancePolicy::EveryBar, &d3);
    let mut book = Book::new(vec![
        SleeveSpec::from_rule("etf", etf, instrument_index(panel, &ETF), share(c.shares.0)),
        SleeveSpec::from_rule("crypto", cry, instrument_index(panel, &CRY), share(c.shares.1)),
    ]);
    if let Some(a) = alloc {
        book = book.with_allocator(a);
    }
    let mut cfg = config_for_case(c);
    let start = d1.keys().min().unwrap().max(d3.keys().min().unwrap());
    cfg.account_start = Some(BarTime::from_date(*start));
    (book, cfg)
}

fn real_case(name: &'static str) -> Case {
    let live = |n| Case {
        name: n,
        shares: (0.6, 0.4),
        cadence: BookCadence::AllSleevesOnAnyDue,
        filter: true,
        budget: true,
        risk_scale: 1.0,
        allocated_currency: None,
        max_gross: None,
        invvol_lookback: None,
        etf_only: false,
        etf_every_bar: false,
    };
    let base = |n| Case { cadence: BookCadence::PerSleeve, filter: false, budget: false, ..live(n) };
    match name {
        "book_cert_60_40" => base(name),
        "book_live_60_40" => live(name),
        "book_due_filter_60_40" => Case { cadence: BookCadence::PerSleeve, ..live(name) },
        "book_all_nofilter_60_40" => Case { cadence: BookCadence::AllSleevesOnAnyDue, ..base(name) },
        "book_scaled_50_30" => {
            Case { shares: (0.5, 0.3), risk_scale: 0.8, allocated_currency: Some(1.2 * CAPITAL0), ..live(name) }
        }
        "book_grosscap_60_40" => Case { max_gross: Some(0.85), ..base(name) },
        "book_invvol" => Case { shares: (0.5, 0.5), invvol_lookback: Some(60), ..base(name) },
        other => panic!("no real case {other}"),
    }
}

fn run_real(panel: &BookPanel, r: &Real, name: &'static str) -> (BookResult, BookResult) {
    let c = real_case(name);
    let (book, cfg) = real_book(panel, &c, r);
    simulate_book_gross_and_net(panel, &book, &cfg).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn real_book_matches_the_five_per_bar_key_files_gross_and_net() {
    let r = skip_unless_real!("real_book_matches_the_five_per_bar_key_files_gross_and_net");
    let panel = r.panel();
    let mut worst_all = 0.0f64;
    let mut cells_all = 0usize;
    let mut nbi_all = 0usize;
    for name in ["book_cert_60_40", "book_live_60_40", "book_scaled_50_30", "book_grosscap_60_40", "book_invvol"] {
        let (g, n) = run_real(&panel, &r, name);
        let key = parse_csv(&r.book_file(&format!("{name}_perbar.csv")));
        let k = compare_book_to_key(&key, &g, &n);
        println!(
            "REAL KEYDIFF {name:24} rows {:5} cells {:6} not-bit-identical {:5} max|diff| {:.3e} (worst column {:?}) flags exact {}",
            k.rows,
            k.cells,
            k.cells_not_bit_identical,
            k.worst,
            k.worst_col,
            k.worst_flag == 0.0
        );
        // per-column maxima, for the report
        let mut cols: Vec<&(String, f64)> = k.by_col.iter().collect();
        cols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!(
            "    largest per-column max|diff|: {:?}",
            cols.iter().take(4).map(|c| (c.0.as_str(), c.1)).collect::<Vec<_>>()
        );
        assert!(
            g.refusals.iter().all(|x| x.code == "gross_above_cap"),
            "{name}: only gross-cap refusals expected, got {:?}",
            g.refusals.iter().find(|x| x.code != "gross_above_cap")
        );
        assert!(k.worst <= TOL, "{name}: max |diff| {:.3e} in {}", k.worst, k.worst_col);
        assert_eq!(k.worst_flag, 0.0, "{name}: flags must match exactly");
        worst_all = worst_all.max(k.worst);
        cells_all += k.cells;
        nbi_all += k.cells_not_bit_identical;
        if name == "book_grosscap_60_40" {
            assert_eq!(g.refusals_with_code("gross_above_cap").len(), 319, "Amendment 12: 319 whole-book refusals");
            let first = g.refusals_with_code("gross_above_cap")[0].time.date().to_string();
            assert_eq!(first, "2017-08-01", "Amendment 12: first refusal on 2017-08-01");
        }
    }
    println!("REAL BOOK KEY: {cells_all} cells over 5 files, {nbi_all} not bit-identical, worst max|diff| {worst_all:.3e} (tolerance {TOL:e})");
}

#[test]
fn real_netting_fixture_matches_the_key_file() {
    let r = skip_unless_real!("real_netting_fixture_matches_the_key_file");
    let panel = netting_panel();
    let book = netting_book(&panel);
    let (g, n) = simulate_book_gross_and_net(&panel, &book, &netting_config()).unwrap();
    let key = parse_csv(&r.book_file("synthetic_netting_perbar.csv"));
    let k = compare_book_to_key(&key, &g, &n);
    println!(
        "REAL KEYDIFF synthetic_netting rows {} cells {} not-bit-identical {} max|diff| {:.3e}",
        k.rows, k.cells, k.cells_not_bit_identical, k.worst
    );
    assert!(k.worst <= TOL && k.worst_flag == 0.0);
}

#[test]
fn real_one_sleeve_books_reproduce_the_t0_per_bar_files() {
    let r = skip_unless_real!("real_one_sleeve_books_reproduce_the_t0_per_bar_files");
    let (d1, d3) = r.decisions();
    for (sid, uni, sched, pol, dec, file, name) in [
        (
            "etf",
            ETF.to_vec(),
            DecisionSchedule::LastBarOfMonth,
            RebalancePolicy::OnDecision,
            &d1,
            "S1_etf_trend_faber_perbar.csv",
            "S1",
        ),
        (
            "crypto",
            CRY.to_vec(),
            DecisionSchedule::Daily,
            RebalancePolicy::EveryBar,
            &d3,
            "S3_crypto_trend_100d_perbar.csv",
            "S3",
        ),
    ] {
        let panel = Panel::from_long_csv(&r.candles, &uni).unwrap();
        let bp = BookPanel::from_panel(&panel);
        let rule = scripted(if name == "S1" { "etf_trend_faber" } else { "crypto_trend_100d" }, &uni, sched, pol, dec);
        let book = Book::new(vec![SleeveSpec::from_rule(sid, rule, (0..uni.len()).collect(), ShareSpec::Fixed(1.0))]);
        let mut cfg = BookConfig {
            sim: SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() },
            ..BookConfig::default()
        };
        cfg.account_start = Some(BarTime::from_date(*dec.keys().min().unwrap()));
        let (g, n) = simulate_book_gross_and_net(&bp, &book, &cfg).unwrap();
        let key = parse_csv(&r.t0_file(file));
        let k = compare_book_to_key(&key, &g, &n);
        println!(
            "REAL T0 {name}: rows {} cells {} not-bit-identical {} max|diff| {:.3e} (worst column {:?})",
            k.rows, k.cells, k.cells_not_bit_identical, k.worst, k.worst_col
        );
        assert!(k.worst <= TOL && k.worst_flag == 0.0, "{name}");
        // metrics of the book equal the T0 key's (account clock = own calendar for one sleeve)
        let m = g.metrics().unwrap();
        println!("REAL T0 {name}: gross sharpe {} cagr {} n {}", m.sharpe, m.cagr, m.n);
    }
}

#[test]
fn real_cadence_modes_reproduce_the_amendment_12_f1_numbers() {
    let r = skip_unless_real!("real_cadence_modes_reproduce_the_amendment_12_f1_numbers");
    let panel = r.panel();
    let etf = ["SPY", "EFA", "IEF", "DBC", "VNQ"];
    let stats = |x: &'static str, y: &'static str| {
        let rx = run_real(&panel, &r, x);
        let ry = run_real(&panel, &r, y);
        f1_stats((&rx.0, &rx.1), (&ry.0, &ry.1), &etf)
    };
    let check = |label: &str, got: &[(&'static str, f64)], want: &[(&str, f64, f64)]| {
        for (name, expect, tol) in want {
            let v = got.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no stat {name}")).1;
            println!("F1 {label} {name:30} rust {v:<22} amendment {expect} (tol {tol})");
            assert!((v - expect).abs() <= *tol, "F1 {label} {name}: {v} vs pre-registered {expect}");
        }
    };
    // Amendment 12 section 3, pair 1: filter 10/2% + budget cash (book_due_filter vs book_live)
    let a = stats("book_due_filter_60_40", "book_live_60_40");
    check(
        "filter+budget",
        &a,
        &[
            ("etf_planned_bars_X", 42.0, 0.0),
            ("etf_planned_bars_Y", 882.0, 0.0),
            ("etf_trade_bars_gross_X", 39.0, 0.0),
            ("etf_trade_bars_gross_Y", 316.0, 0.0),
            ("etf_weight_gap_L1_mean", 0.01395, 5e-6),
            ("etf_weight_gap_L1_max", 0.1120, 5e-5),
            ("etf_weight_gap_bars_gt_1e-9", 1185.0, 0.0),
            ("etf_weight_gap_bars_gt_1e-2", 466.0, 0.0),
            ("ret_gross_diff_max_abs", 1.77e-3, 5e-6),
            ("ret_gross_diff_mean_abs", 4.2e-5, 5e-7),
            ("ret_gross_corr", 0.999963, 5e-7),
            ("net_d_sharpe_Y_minus_X", -0.0052, 5e-5),
            ("net_d_cagr_pp_Y_minus_X", -0.124, 5e-4),
            ("turnover_per_year_X", 7.85, 5e-3),
            ("turnover_per_year_Y", 8.30, 5e-3),
            ("cost_bps_per_year_X", 78.47, 5e-3),
            ("cost_bps_per_year_Y", 83.05, 5e-3),
        ],
    );
    // pair 2: no filter + certification cash (book_cert vs book_all_nofilter)
    let b = stats("book_cert_60_40", "book_all_nofilter_60_40");
    check(
        "nofilter+cert",
        &b,
        &[
            ("etf_planned_bars_X", 42.0, 0.0),
            ("etf_planned_bars_Y", 882.0, 0.0),
            ("etf_trade_bars_gross_X", 42.0, 0.0),
            ("etf_trade_bars_gross_Y", 862.0, 0.0),
            ("etf_weight_gap_L1_mean", 0.01400, 5e-6),
            ("etf_weight_gap_L1_max", 0.1118, 5e-5),
            ("etf_weight_gap_bars_gt_1e-9", 1167.0, 0.0),
            ("etf_weight_gap_bars_gt_1e-2", 450.0, 0.0),
            ("ret_gross_diff_max_abs", 1.11e-3, 5e-6),
            ("ret_gross_diff_mean_abs", 4.1e-5, 5e-7),
            ("ret_gross_corr", 0.999969, 5e-7),
            ("net_d_sharpe_Y_minus_X", -0.0031, 5e-5),
            ("net_d_cagr_pp_Y_minus_X", -0.080, 5e-4),
            ("turnover_per_year_X", 8.165, 5e-4),
            ("turnover_per_year_Y", 8.979, 5e-4),
            ("cost_bps_per_year_X", 81.65, 5e-3),
            ("cost_bps_per_year_Y", 89.79, 5e-3),
        ],
    );
}
