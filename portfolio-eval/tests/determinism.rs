//! Bit-identical repeats, including from four concurrent threads, and thread-count independence of the Monte Carlo.

use portfolio_eval::bootstrap::bootstrap_indices;
use portfolio_eval::marginal::{marginal_contribution, MarginalConfig};
use portfolio_eval::pbo::{pbo_cscv, PboMetric};
use portfolio_eval::power::*;
use portfolio_eval::rng::Rng;

fn data(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng::seed_from_u64(seed);
    let base: Vec<f64> = (0..n).map(|_| 0.01 * (0.03 + r.normal())).collect();
    let comb: Vec<f64> = base.iter().map(|b| 0.5 * b + 0.005 * (0.05 + r.normal())).collect();
    (base, comb)
}

#[test]
fn four_threads_computing_the_same_marginal_test_agree_bit_for_bit() {
    let (b, c) = data(7, 400);
    let mut cfg = MarginalConfig::new(252.0);
    cfg.n_boot = 299;
    let reference = marginal_contribution(&b, &c, &cfg).unwrap();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4).map(|_| s.spawn(|| marginal_contribution(&b, &c, &cfg).unwrap())).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in &results {
        assert_eq!(r, &reference);
        assert_eq!(r.delta_sharpe.to_bits(), reference.delta_sharpe.to_bits());
        assert_eq!(r.boot_se.to_bits(), reference.boot_se.to_bits());
        assert_eq!(r.ci_low.to_bits(), reference.ci_low.to_bits());
        assert_eq!(r.ci_high.to_bits(), reference.ci_high.to_bits());
    }
}

#[test]
fn bootstrap_indices_and_pbo_are_repeatable_across_threads() {
    let want = bootstrap_indices(700, 7.5, 31).unwrap();
    let cfgs: Vec<Vec<f64>> = (0..5).map(|i| data(i, 120).0).collect();
    let refs: Vec<&[f64]> = cfgs.iter().map(|c| c.as_slice()).collect();
    let pbo_want = pbo_cscv(&refs, 6, PboMetric::Sharpe).unwrap();
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    assert_eq!(bootstrap_indices(700, 7.5, 31).unwrap(), want);
                    assert_eq!(pbo_cscv(&refs, 6, PboMetric::Sharpe).unwrap(), pbo_want);
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    });
}

fn tiny_spec() -> PowerSpec {
    PowerSpec {
        horizons_years: vec![1.0, 2.0],
        corrs: vec![0.0, 0.5],
        effects: vec![0.0, 0.5, 1.0],
        reps: 9,
        n_boot: 39,
        periods_per_year: 252.0,
        base_sharpe: 0.5,
        share: 0.5,
        alpha: 0.05,
        power: 0.8,
        seed: 4242,
        innovations: Innovations::GAUSSIAN,
    }
}

#[test]
fn monte_carlo_is_independent_of_the_thread_count_and_bit_identical_on_repeat() {
    let one = run_power(&tiny_spec(), 1).unwrap();
    let four = run_power(&tiny_spec(), 4).unwrap();
    let seven = run_power(&tiny_spec(), 7).unwrap();
    let again = run_power(&tiny_spec(), 4).unwrap();
    assert_eq!(one, four);
    assert_eq!(one, seven);
    assert_eq!(four, again);
    assert_eq!(one.digest(), four.digest());
    for (a, b) in one.cells.iter().zip(&four.cells) {
        assert_eq!(a.mean_boot_se.to_bits(), b.mean_boot_se.to_bits());
        assert_eq!(a.mean_delta.to_bits(), b.mean_delta.to_bits());
        assert_eq!(a.mean_block.to_bits(), b.mean_block.to_bits());
    }
    // a different seed changes the experiment
    let mut other = tiny_spec();
    other.seed += 1;
    assert_ne!(run_power(&other, 2).unwrap().digest(), one.digest());
}
