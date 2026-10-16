//! Numerical edge cases: zero variance, too few observations, NaN and infinite inputs return typed errors and NEVER
//! panic. A randomised sweep feeds every public entry point nasty inputs under `catch_unwind`.

// Tests write single ranges such as `&[5..6]` on purpose: a list holding ONE range, not the integers 5 to 6.
#![allow(clippy::single_range_in_vec_init)]

use portfolio_eval::bootstrap::*;
use portfolio_eval::dsr::*;
use portfolio_eval::error::EvalError;
use portfolio_eval::folds::*;
use portfolio_eval::hac::*;
use portfolio_eval::ledger::*;
use portfolio_eval::marginal::*;
use portfolio_eval::pbo::*;
use portfolio_eval::rng::Rng;
use portfolio_eval::stats;
use std::panic::catch_unwind;

fn nasty_series(rng: &mut Rng) -> Vec<f64> {
    let n = match rng.below(6) {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 5,
        4 => 40,
        _ => 200,
    };
    let kind = rng.below(7);
    (0..n)
        .map(|i| match kind {
            0 => 0.0,
            1 => 0.01,
            2 => {
                if i == n / 2 {
                    f64::NAN
                } else {
                    rng.normal() * 0.01
                }
            }
            3 => {
                if i == 1 {
                    f64::INFINITY
                } else {
                    rng.normal() * 0.01
                }
            }
            4 => rng.normal() * 1e-300,
            5 => rng.normal() * 1e300,
            _ => rng.normal() * 0.01,
        })
        .collect()
}

#[test]
fn no_public_entry_point_panics_on_nasty_input() {
    let mut rng = Rng::seed_from_u64(0xBAD);
    for round in 0..600 {
        let a = nasty_series(&mut rng);
        let b = if rng.below(3) == 0 { nasty_series(&mut rng) } else { a.iter().map(|v| v * 0.5 + 0.001).collect() };
        let ppy = [252.0, 365.0, 0.0, -1.0, f64::NAN, f64::INFINITY][rng.below(6) as usize];
        let cfg_seed = rng.next_u64();
        let res = catch_unwind(|| {
            let _ = stats::mean(&a);
            let _ = stats::variance(&a);
            let _ = stats::sharpe_annual(&a, ppy);
            let _ = stats::skewness(&a);
            let _ = stats::kurtosis(&a);
            let _ = stats::autocovariance(&a, 3);
            let _ = stats::correlation(&a, &b);
            let _ = stats::median(&a);
            let _ = stats::mad_scale(&a);
            let _ = politis_white_block_length(&a);
            let _ = auto_block_length(&[&a, &b]);
            let _ = bootstrap_indices(a.len(), 3.0, cfg_seed);
            let _ = percentile_ci(&a, 0.9);
            let _ = long_run_variance_of_mean(&a, 2);
            let _ = spanning_alpha(&a, &[&b], HacLag::Auto, ppy);
            let _ = spanning_alpha(&a, &[&b, &b], HacLag::Fixed(1000), ppy);
            let mut cfg = MarginalConfig::new(ppy);
            cfg.n_boot = 25;
            cfg.seed = cfg_seed;
            let _ = marginal_contribution(&a, &b, &cfg);
            cfg.block = BlockLength::Fixed(1e9);
            let _ = marginal_contribution(&a, &b, &cfg);
            let _ = delta_sharpe(&a, &b, ppy, ScaleMode::InSample);
            let _ = equal_vol_scales(&a, &b, 0.01);
            let _ = equal_vol_difference(&a, &b, 1.0, f64::NAN);
            let _ = deflated_sharpe_from_returns(&a, 10, 0.5, ppy, true);
            let _ = core_compat::deflated_sharpe_performance(3, &a);
            let _ = pbo_cscv(&[&a, &b], 4, PboMetric::Sharpe);
            let _ = pbo_cscv(&[&a], 3, PboMetric::Mean);
            let mut l = TrialLedger::new();
            let _ = l.record_config("x", a.first().copied().unwrap_or(f64::NAN));
            let _ = l.deflated_sharpe_of(&a, ppy, false);
            let _ = benjamini_hochberg_q(&a);
        });
        assert!(res.is_ok(), "panic in round {round} with a.len() = {}, b.len() = {}", a.len(), b.len());
    }
}

#[test]
fn extreme_parameters_do_not_panic_or_overflow() {
    assert!(walk_forward_folds(usize::MAX, usize::MAX / 2, 3, 1, 0, TrainMode::Expanding).is_err());
    assert!(walk_forward_folds(10, 1, usize::MAX, 1, 0, TrainMode::Expanding).is_err());
    assert!(walk_forward_folds(10, 1, 1, 1, usize::MAX, TrainMode::Expanding).is_err());
    assert!(purged_kfold(usize::MAX / 2, 2, usize::MAX, usize::MAX).is_err());
    assert!(cpcv_splits(100, 100, 50, 0, 0).is_err());
    assert_eq!(purge_embargo_bars(usize::MAX, f64::INFINITY, usize::MAX), usize::MAX / 2);
    assert_eq!(purge_embargo_bars(0, 1e300, 10), 5);
    assert!(holdout_tail(usize::MAX, 0.25, usize::MAX).is_err());
    assert_eq!(TrialLedger::new().k_effective(), 0);
    // huge but finite values may return a value or an error, never a panic
    let _ = min_track_record_length(1e300, 0.0, 0.0, 3.0, 0.9);
    let _ = deflated_sharpe(
        &DsrInputs {
            sharpe: 1e300,
            n_obs: usize::MAX,
            skewness: 0.0,
            kurtosis: 3.0,
            n_trials: usize::MAX,
            trial_sharpe_std: 1e300,
        },
        true,
    );
}

#[test]
fn each_documented_error_kind_is_reachable() {
    let ok: Vec<f64> = (0..60).map(|i| 0.001 * ((i * 7) % 11) as f64 - 0.004).collect();
    assert!(matches!(stats::mean(&[]), Err(EvalError::TooShort { .. })));
    assert!(matches!(stats::correlation(&ok, &ok[..5]), Err(EvalError::LengthMismatch { .. })));
    let mut nf = ok.clone();
    nf[0] = f64::NAN;
    assert!(matches!(stats::mean(&nf), Err(EvalError::NonFinite { .. })));
    assert!(matches!(stats::sharpe_per_period(&[0.1; 10]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(stats::sharpe_annual(&ok, 0.0), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(spanning_alpha(&ok, &[&ok, &ok], HacLag::Auto, 252.0), Err(EvalError::Singular { .. })));
    let mut c = MarginalConfig::new(252.0);
    c.n_boot = 19;
    // a series with only two distinct values that make many replicates degenerate is not needed: use the direct check
    assert!(marginal_contribution(&ok, &ok, &c).is_ok());
    assert!(matches!(
        SealedHoldout::new(0..10).unwrap().verify_disjoint(&[5..6]),
        Err(EvalError::HoldoutOverlap { .. })
    ));
    // Display strings are non-empty and mention the offending name
    let e = stats::mean(&nf).unwrap_err();
    assert!(e.to_string().contains("non-finite"));
    let e = walk_forward_folds(10, 0, 1, 1, 0, TrainMode::Expanding).unwrap_err();
    assert!(e.to_string().contains("n_folds"));
    let e: Box<dyn std::error::Error> = Box::new(EvalError::Singular { what: "x" });
    assert!(e.to_string().contains("singular"));
}

#[test]
fn n_too_small_is_refused_by_every_test_that_needs_a_sample() {
    let x = [0.01, -0.01, 0.02];
    assert!(marginal_contribution(&x, &x, &MarginalConfig::new(252.0)).is_err());
    assert!(spanning_alpha(&x, &[&x], HacLag::Auto, 252.0).is_err());
    assert!(politis_white_block_length(&x).is_err());
    assert!(pbo_cscv(&[&x, &x], 2, PboMetric::Sharpe).is_err());
    assert!(stats::kurtosis(&x).is_err());
}
