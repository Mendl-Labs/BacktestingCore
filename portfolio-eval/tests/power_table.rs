//! The PF5 power table: pinned so that it cannot silently change.
//!
//! * always on: the committed `POWER_TABLE.md` block hashes to a constant in this file and to the digest printed in
//!   the file; its parsed numbers satisfy the statistical sanity properties; every number quoted in the prose is
//!   checked against the table; a reduced Monte Carlo is pinned bit for bit and sanity-checked;
//! * `--ignored` (CI runs it in release mode): regenerate the whole block and compare byte for byte.
//!
//! To change the table on purpose: `cargo run --release --bin power_table -- --write`, then update
//! `PINNED_BLOCK_SHA256` below and, if a quoted number moved, the prose and `QUOTED` below.

use portfolio_eval::power::*;
use portfolio_eval::sha256::sha256_hex;

const PINNED_BLOCK_SHA256: &str = "2b900978e702cd38bc6bade7d18df372f82adfed503669b39e8734bdedfe1831";
const TABLE: &str = include_str!("../POWER_TABLE.md");

fn block() -> (&'static str, &'static str) {
    let b = TABLE.find(BEGIN_MARKER).expect("BEGIN marker");
    let e = TABLE.find(END_MARKER).expect("END marker");
    let inner = &TABLE[b + BEGIN_MARKER.len() + 1..e];
    let split = inner.find("\nsha256 of the generated section above: `").expect("digest line");
    (&inner[..split], &inner[split..])
}

/// One parsed row of a main table.
#[derive(Debug, Clone)]
struct Row {
    rho: f64,
    years: f64,
    bars: usize,
    power: Vec<f64>,
    mde_analytic: f64,
    mde_mc: Option<f64>,
    boot_se: f64,
}

fn parse_main(body: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut rho = f64::NAN;
    let mut in_main = false;
    for line in body.lines() {
        if line.starts_with("## Size and power") {
            in_main = true;
        }
        if line.starts_with("## Robustness") {
            in_main = false;
        }
        if let Some(rest) = line.strip_prefix("### Correlation rho = ") {
            rho = rest.split_whitespace().next().unwrap().parse().unwrap();
            continue;
        }
        if !in_main || !line.starts_with("| ") {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(|c| c.trim()).collect();
        let Ok(years) = cells[0].parse::<f64>() else { continue };
        let n_eff = 8;
        rows.push(Row {
            rho,
            years,
            bars: cells[1].parse().unwrap(),
            power: cells[2..2 + n_eff].iter().map(|c| c.parse().unwrap()).collect(),
            mde_analytic: cells[2 + n_eff].parse().unwrap(),
            mde_mc: cells[3 + n_eff].parse().ok(),
            boot_se: cells[4 + n_eff].parse().unwrap(),
        });
    }
    rows
}

fn parse_scenarios(body: &str) -> Vec<(String, Vec<f64>, f64, Option<f64>)> {
    let mut out = Vec::new();
    let mut on = false;
    for line in body.lines() {
        if line.starts_with("## Robustness") {
            on = true;
            continue;
        }
        if !on || !line.starts_with("| ") || line.starts_with("| scenario") {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(|c| c.trim()).collect();
        if cells.len() < 12 || cells[0].starts_with("---") {
            continue;
        }
        out.push((
            cells[0].to_string(),
            cells[2..10].iter().map(|c| c.parse().unwrap()).collect(),
            cells[10].parse().unwrap(),
            cells[11].parse().ok(),
        ));
    }
    out
}

fn row(rows: &[Row], rho: f64, years: f64) -> &Row {
    rows.iter().find(|r| r.rho == rho && r.years == years).expect("row")
}

#[test]
fn the_committed_block_is_pinned_by_digest() {
    let (body, tail) = block();
    let actual = sha256_hex(body.as_bytes());
    assert_eq!(actual, PINNED_BLOCK_SHA256, "the generated block changed: regenerate on purpose and update the pin");
    assert!(tail.contains(&format!("`{actual}`")), "the digest printed in POWER_TABLE.md must match the block");
    assert!(!TABLE.contains("placeholder"));
}

#[test]
fn the_block_describes_the_committed_experiment() {
    let (body, _) = block();
    let s = committed_main_spec();
    let header = format!(
        "Experiment: {} repetitions per cell, {} bootstrap replicates, {} bars/year, base Sharpe {}, candidate share {}, one-sided alpha {}, target power {}, seed 0x{:016x}.",
        s.reps, s.n_boot, s.periods_per_year, s.base_sharpe, s.share, s.alpha, s.power, s.seed
    );
    assert!(body.contains(&header), "header line differs from committed_main_spec()");
    assert_eq!((s.horizons_years.clone(), s.corrs.clone()), (vec![2.0, 3.5, 5.0, 10.0], vec![0.0, 0.3, 0.6]));
    let rows = parse_main(body);
    assert_eq!(rows.len(), 12);
    for r in &rows {
        assert_eq!(r.bars, s.n_obs(r.years));
        assert_eq!(r.power.len(), s.effects.len());
    }
    let scen = parse_scenarios(body);
    assert_eq!(scen.len(), committed_scenarios().len());
    for ((name, ..), sc) in scen.iter().zip(committed_scenarios()) {
        assert_eq!(name, sc.name);
    }
}

#[test]
fn parsed_numbers_have_the_statistical_properties_of_a_correct_test() {
    let (body, _) = block();
    let rows = parse_main(body);
    let z = 2.4865; // z_(0.95) + z_(0.8)
    let mut sizes = Vec::new();
    for r in &rows {
        // size: false-positive rate near 5% (400 reps: standard error up to 2.5 points; allow ~3.5 SE across 12 cells)
        assert!(r.power[0] >= 1.5 && r.power[0] <= 9.0, "size {} at rho {} years {}", r.power[0], r.rho, r.years);
        sizes.push(r.power[0]);
        // power grows with the true effect (common random numbers): monotone up to 3 points of Monte Carlo noise
        for w in r.power.windows(2) {
            assert!(w[1] >= w[0] - 3.0, "power not monotone in effect: {:?}", r.power);
        }
        assert!(r.power[7] >= r.power[0] + 80.0, "a 1.5 Sharpe improvement must be detected: {:?}", r.power);
        // analytic MDE and bootstrap SE agree (MDE = z x SE)
        assert!(
            (r.boot_se * z / r.mde_analytic - 1.0).abs() < 0.05,
            "SE {} vs MDE/z {}",
            r.boot_se,
            r.mde_analytic / z
        );
        // Monte Carlo MDE is close to, and not below, the analytic one
        let mc = r.mde_mc.expect("power reaches 80% inside the grid for every main cell");
        assert!(mc / r.mde_analytic > 0.95 && mc / r.mde_analytic < 1.15, "MC {mc} analytic {}", r.mde_analytic);
    }
    let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
    assert!((mean - 5.0).abs() < 1.0, "mean size {mean}");
    // longer horizons are more powerful and have a smaller MDE; higher correlation is more precise
    for rho in [0.0, 0.3, 0.6] {
        let ys = [2.0, 3.5, 5.0, 10.0];
        for w in ys.windows(2) {
            let (a, b) = (row(&rows, rho, w[0]), row(&rows, rho, w[1]));
            assert!(b.mde_analytic < a.mde_analytic);
            assert!(b.boot_se < a.boot_se);
            for e in 3..7 {
                assert!(b.power[e] >= a.power[e] - 4.0, "rho {rho}: power fell with more data at effect column {e}");
            }
        }
    }
    for years in [2.0, 3.5, 5.0, 10.0] {
        assert!(row(&rows, 0.3, years).mde_analytic < row(&rows, 0.0, years).mde_analytic);
        assert!(row(&rows, 0.6, years).mde_analytic < row(&rows, 0.3, years).mde_analytic);
    }
    // and the analytic column IS the crate's formula
    let spec = committed_main_spec();
    for r in &rows {
        assert!((spec.analytic_mde(r.years, r.rho).unwrap() - r.mde_analytic).abs() < 0.0051);
    }
}

/// Numbers quoted in the prose of POWER_TABLE.md: (rho, years, column index into effects, quoted percent).
const QUOTED: [(f64, f64, usize, f64); 6] = [
    (0.3, 5.0, 3, 26.5),
    (0.3, 5.0, 4, 52.2),
    (0.3, 3.5, 1, 10.8),
    (0.3, 3.5, 2, 18.0),
    (0.0, 3.5, 2, 10.0),
    (0.6, 3.5, 1, 11.0),
];

#[test]
fn numbers_quoted_in_the_prose_match_the_table() {
    let (body, _) = block();
    let rows = parse_main(body);
    for (rho, years, col, want) in QUOTED {
        assert_eq!(row(&rows, rho, years).power[col], want, "rho {rho} years {years} col {col}");
    }
    // MDE quotes: 3.5y rho 0.3: analytic 0.83, MC 0.90; 2y: 1.10/1.19; 10y: 0.49/0.52; 3.5y rho 0.6 analytic 0.61
    let r = row(&rows, 0.3, 3.5);
    assert_eq!((r.mde_analytic, r.mde_mc), (0.83, Some(0.90)));
    let r = row(&rows, 0.3, 2.0);
    assert_eq!((r.mde_analytic, r.mde_mc), (1.10, Some(1.19)));
    let r = row(&rows, 0.3, 10.0);
    assert_eq!((r.mde_analytic, r.mde_mc), (0.49, Some(0.52)));
    assert_eq!(row(&rows, 0.6, 3.5).mde_analytic, 0.61);
    assert_eq!(row(&rows, 0.3, 5.0).mde_mc, Some(0.73));
    // size range and mean quoted as "3.5% to 6.2% (mean about 5.1%)"
    let sizes: Vec<f64> = rows.iter().map(|r| r.power[0]).collect();
    assert_eq!(sizes.iter().cloned().fold(f64::INFINITY, f64::min), 3.5);
    assert_eq!(sizes.iter().cloned().fold(0.0, f64::max), 6.2);
    let mean = sizes.iter().sum::<f64>() / 12.0;
    assert!((mean - 5.1).abs() < 0.05, "{mean}");
    // scenarios quoted: fat tails size 7.0, crypto calendar MDE, base-SR-1.5 MDE, AR(1) MDE, small candidate MDE
    let scen = parse_scenarios(body);
    let get = |needle: &str| scen.iter().find(|s| s.0.contains(needle)).unwrap().clone();
    assert_eq!(get("fat tails").1[0], 7.0);
    assert_eq!(get("AR(1)").3, Some(0.95));
    assert_eq!(get("AR(1)").2, 0.83);
    assert_eq!(get("base SR 1.5").3, Some(0.91));
    assert_eq!(get("small candidate").2, 0.29);
    assert_eq!(get("small candidate").3, Some(0.32));
    assert!(get("crypto calendar").0.contains("1278 bars"));
    // the crypto-calendar row has the same power as the 252-bar baseline within Monte Carlo noise (SE up to 2.9 points)
    let base = get("baseline");
    let crypto = get("crypto calendar");
    for e in 0..8 {
        assert!((base.1[e] - crypto.1[e]).abs() < 8.0, "column {e}: {} vs {}", base.1[e], crypto.1[e]);
    }
    // derived figures in the prose
    let spec = committed_main_spec();
    let sr_x = |share: f64| {
        let mut s = spec.clone();
        s.share = share;
        s.candidate_sharpe(0.3, 0.3)
    };
    assert!(sr_x(0.2) > 1.4 && sr_x(0.2) < 1.7, "20% candidate needs a standalone Sharpe of about 1.5: {}", sr_x(0.2));
    let same_sharpe_lift = |rho: f64| 0.5 / ((1.0 + rho) / 2.0_f64).sqrt() - 0.5;
    assert!((same_sharpe_lift(0.0) - 0.207).abs() < 0.001);
    assert!((same_sharpe_lift(0.3) - 0.120).abs() < 0.001);
    assert!((same_sharpe_lift(0.6) - 0.059).abs() < 0.001);
}

fn reduced_spec() -> PowerSpec {
    PowerSpec {
        horizons_years: vec![2.0],
        corrs: vec![0.3],
        effects: vec![0.0, 0.5, 1.5],
        reps: 120,
        n_boot: 99,
        periods_per_year: 252.0,
        base_sharpe: 0.5,
        share: 0.5,
        alpha: 0.05,
        power: 0.8,
        seed: 0x5045_5254_4553_5431,
        innovations: Innovations::GAUSSIAN,
    }
}

#[test]
fn a_reduced_monte_carlo_is_pinned_bit_for_bit_and_statistically_sane() {
    let res = run_power(&reduced_spec(), 2).unwrap();
    assert_eq!(res.digest(), REDUCED_DIGEST, "the reduced Monte Carlo changed");
    let rates: Vec<f64> = res.cells.iter().map(|c| c.rate()).collect();
    // size (true effect 0) near 5%: 120 reps have a standard error of 2 points
    assert!(rates[0] <= 0.13, "size {}", rates[0]);
    // power rises with the effect and a 1.5 Sharpe improvement is found almost always at 2 years, rho 0.3
    assert!(rates[1] > rates[0] && rates[1] < 0.6, "{rates:?}");
    assert!(rates[2] > 0.85, "{rates:?}");
    let same = run_power(&reduced_spec(), 1).unwrap();
    assert_eq!(same, res, "thread count must not matter");
}

const REDUCED_DIGEST: &str = "d96cfcfe9766d89246a4959a7a8d0b9183f13e35bb213be27ef74b021c4c2c94";

#[test]
#[ignore = "runs the full experiment (about 40 s in release on two cores); CI runs it with --release --ignored"]
fn the_committed_block_regenerates_byte_for_byte() {
    let generated = generated_block(4).unwrap();
    let b = TABLE.find(BEGIN_MARKER).unwrap();
    let e = TABLE.find(END_MARKER).unwrap() + END_MARKER.len() + 1;
    assert_eq!(
        &TABLE[b..e],
        generated,
        "POWER_TABLE.md is stale: run `cargo run --release --bin power_table -- --write`"
    );
}
