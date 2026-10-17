//! R41-A(3)(b), the glue half of the per-rule N-generality attestation: the rules are driven through the simulator
//! (the path a cell takes) on seeded synthetic panels of 9 and 11 instruments, and what the simulator records is
//! checked against the R41-A invariants. The decision arithmetic itself is compared with an independent reference in
//! the unit tests next to the code (`adapters::tests` for the trend rule, `momentum_rank_weighted::tests` for the
//! rank weights); this file proves the policy and the universe actually reach the simulator. Synthetic data only.

mod common;

use chrono::{Datelike, NaiveDate};
use weightsim::{simulate, Panel, SimConfig, WeightRule};
use weightsim_rules::adapters::MomentumRankWeightedRule;
use weightsim_rules::{from_naive, EtfTrendRule, WeightPolicy};

const NAMES: [&str; 11] = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11"];

fn synthetic_panel(n: usize, bars: usize, seed: u64) -> Panel {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut dates = Vec::with_capacity(bars);
    let mut d = NaiveDate::from_ymd_opt(2015, 1, 1).unwrap();
    while dates.len() < bars {
        if d.weekday().number_from_monday() <= 5 {
            dates.push(from_naive(d));
        }
        d = d.succ_opt().unwrap();
    }
    let mut closes = Vec::with_capacity(n);
    for j in 0..n {
        let mut p: i64 = 400 + 12 * j as i64;
        let mut col = Vec::with_capacity(bars);
        for _ in 0..bars {
            let drift = if j % 2 == 0 { 1 } else { -1 };
            p = (p + (next() % 9) as i64 - 4 + drift).max(8);
            col.push(p as f64 * 0.25);
        }
        closes.push(col);
    }
    let symbols = NAMES[..n].iter().map(|s| (*s).to_string()).collect();
    Panel::new(symbols, dates, closes).unwrap()
}

#[test]
fn equal_over_n_reaches_the_simulator_as_one_over_n_and_never_exceeds_full_investment() {
    for n in [9usize, 11] {
        let panel = synthetic_panel(n, 1500, 40 + n as u64);
        let rule = EtfTrendRule::with_weight_policy(NAMES[..n].to_vec(), 10, 0, WeightPolicy::EqualOverN);
        assert_eq!(rule.universe().len(), n);
        let w = 1.0_f64 / n as f64;
        // `max_gross: Some(1.0)` makes the simulator itself refuse any decision above 100% (refuse, not clip).
        let cfg = SimConfig { max_gross: Some(1.0), ..SimConfig::default() };
        let sim = simulate(&panel, &rule, &cfg).unwrap();
        let (mut held_cells, mut cash_cells) = (0usize, 0usize);
        for t in 0..sim.n_bars() {
            let row = sim.row(&sim.target_weights, t);
            assert_eq!(row.len(), n);
            for x in row {
                assert!(*x == 0.0 || x.to_bits() == w.to_bits(), "N = {n}, bar {t}: {x} is neither 0 nor 1/N");
                if *x == 0.0 {
                    cash_cells += 1;
                } else {
                    held_cells += 1;
                }
            }
            assert!(sim.gross_exposure[t] <= 1.0 + 1e-9, "N = {n}, bar {t}: gross {}", sim.gross_exposure[t]);
        }
        assert!(held_cells > 100 && cash_cells > 100, "N = {n}: held {held_cells}, cash {cash_cells}");
    }
}

#[test]
fn momentum_rank_weights_reach_the_simulator_at_n_9_and_11() {
    for n in [9usize, 11] {
        let panel = synthetic_panel(n, 450, 90 + n as u64);
        let rule = MomentumRankWeightedRule::new(NAMES[..n].to_vec(), 126);
        // The declared concentration is the K = N top weight, 2 / (N + 1).
        let declared: f64 = rule.declared_parameters()["max_weight_per_instrument"].parse().unwrap();
        assert!((declared - 2.0 / (n as f64 + 1.0)).abs() < 1e-12, "N = {n}: declared {declared}");
        let sim = simulate(&panel, &rule, &SimConfig::default()).unwrap();
        let denom = (n * (n + 1) / 2) as f64;
        let mut checked = 0;
        for t in 130..sim.n_bars() {
            let w = sim.row(&sim.target_weights, t);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "N = {n}, bar {t}: gross {}", w.iter().sum::<f64>());
            let mut sorted: Vec<f64> = w.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, x) in sorted.iter().enumerate() {
                let want = (i + 1) as f64 / denom;
                assert!((x - want).abs() < 1e-12, "N = {n}, bar {t}: sorted weight {i} is {x}, want {want}");
            }
            checked += 1;
        }
        assert!(checked > 300);
    }
}

/// K5: the default instance is byte-identical to the published rule. Its `declared_parameters` and series digest are
/// the pre-R41 ones (the pinned ladder digests themselves are asserted in `tests/registry.rs`,
/// `PRE_REGISTRY_LADDER_DIGEST`, and in `tests/ladder_real.rs`; this change touches no fixture and no pin). Here: an
/// explicit `FixedConstant(0.20)` is the same run as `default()`, and `EqualOverN` at N = 5 (same weight, 1.0/5.0
/// being bit-identical to 0.20) produces the same weights, returns and equity, bit for bit.
#[test]
fn the_default_policy_run_is_unchanged_and_equal_over_n_at_five_reproduces_it() {
    let panel = common::s1_panel();
    let cfg = SimConfig::default();
    let published = simulate(&panel, &EtfTrendRule::default(), &cfg).unwrap();
    let explicit_policy = WeightPolicy::FixedConstant(0.20);
    let explicit = EtfTrendRule::with_weight_policy(common::ETF.to_vec(), 10, 0, explicit_policy);
    assert_eq!(explicit, EtfTrendRule::default());
    let explicit = simulate(&panel, &explicit, &cfg).unwrap();
    assert_eq!(explicit.series_sha256, published.series_sha256);

    let equal = EtfTrendRule::with_weight_policy(common::ETF.to_vec(), 10, 0, WeightPolicy::EqualOverN);
    let equal = simulate(&panel, &equal, &cfg).unwrap();
    assert_eq!(equal.target_weights, published.target_weights);
    assert_eq!(equal.held_weights, published.held_weights);
    assert_eq!(equal.ret, published.ret);
    assert_eq!(equal.equity, published.equity);
    assert!(published.target_weights.contains(&0.2), "the published run holds something");
}
