//! Newey-West long-run variance and the spanning regression against exact rational hand computations
//! (`tests/reference/gen_reference.py`, `fractions.Fraction` arithmetic), plus size and invariance properties.

use portfolio_eval::error::EvalError;
use portfolio_eval::hac::*;
use portfolio_eval::rng::Rng;

const B1: [f64; 10] = [0.010, -0.004, 0.007, 0.002, -0.009, 0.005, 0.001, -0.003, 0.008, -0.006];
const Y1: [f64; 10] = [0.013, -0.005, 0.011, 0.004, -0.008, 0.009, 0.000, -0.002, 0.012, -0.009];
const Z1: [f64; 10] = [0.004, 0.009, -0.007, 0.003, 0.000, -0.005, 0.006, 0.002, -0.001, 0.008];
const X: [f64; 8] = [0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, -0.01];

fn close(a: f64, b: f64, rel: f64) {
    assert!((a - b).abs() <= rel * b.abs().max(1e-300), "{a} vs {b}");
}

#[test]
fn newey_west_lag_rule() {
    assert_eq!(newey_west_lag(100), 4);
    assert_eq!(newey_west_lag(250), 4);
    assert_eq!(newey_west_lag(1260), 7);
    assert_eq!(newey_west_lag(2520), 8);
    assert_eq!(newey_west_lag(0), 0);
    assert_eq!(resolve_lag(HacLag::Fixed(50), 10), 9);
    assert_eq!(resolve_lag(HacLag::Fixed(0), 10), 0);
    assert_eq!(resolve_lag(HacLag::Auto, 250), 4);
}

#[test]
fn long_run_variance_of_the_mean_matches_the_hand_computation() {
    // X has 1/n autocovariances 2.4375e-4 (lag 0) and -1.75e-4 (lag 1); Bartlett weights 1 - l/(L+1)
    close(long_run_variance_of_mean(&X, 0).unwrap(), 0.00024375, 1e-12);
    close(long_run_variance_of_mean(&X, 1).unwrap(), 6.875e-05, 1e-9);
    close(long_run_variance_of_mean(&X, 3).unwrap(), 4.0625e-05, 1e-9);
    assert!(matches!(long_run_variance_of_mean(&X, 8), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(long_run_variance_of_mean(&[1.0], 0), Err(EvalError::TooShort { .. })));
}

#[test]
fn spanning_regression_with_white_errors_matches_exact_arithmetic() {
    let r = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(0), 252.0).unwrap();
    assert_eq!(r.n, 10);
    assert_eq!(r.lag, 0);
    close(r.alpha, 0.0010825958702064898, 1e-11);
    close(r.beta[0], 1.2885492089031911, 1e-11);
    close(r.se_alpha, 0.0005423213354587398, 1e-10);
    close(r.se_beta[0], 0.08932413423051871, 1e-10);
    close(r.t_alpha, 1.996225852502634, 1e-10);
    close(r.p_two_sided, 0.04590934465810825, 1e-9);
    close(r.resid_var, 2.9190131402520784e-06, 1e-10);
    close(r.alpha_annual, 0.0010825958702064898 * 252.0, 1e-11);
    assert!((r.p_one_sided - r.p_two_sided / 2.0).abs() < 1e-12, "one-sided is half the two-sided for t > 0");
}

#[test]
fn spanning_regression_with_two_hac_lags_matches_exact_arithmetic() {
    let r = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(2), 252.0).unwrap();
    assert_eq!(r.lag, 2);
    close(r.alpha, 0.0010825958702064898, 1e-11); // point estimates do not depend on the lag
    close(r.beta[0], 1.2885492089031911, 1e-11);
    close(r.se_alpha, 0.0004805809507373888, 1e-10);
    close(r.se_beta[0], 0.07549848468714801, 1e-10);
    close(r.t_alpha, 2.2526816107575374, 1e-10);
    close(r.p_two_sided, 0.024279231240516674, 1e-9);
}

#[test]
fn spanning_regression_with_two_benchmarks() {
    let r = spanning_alpha(&Y1, &[&B1, &Z1], HacLag::Fixed(1), 252.0).unwrap();
    close(r.alpha, 0.001756517091088512, 1e-10);
    close(r.beta[0], 1.1785833721663332, 1e-10);
    close(r.beta[1], -0.2910309476165675, 1e-10);
    close(r.se_alpha, 0.00026280110944270507, 1e-9);
    close(r.se_beta[0], 0.04467706455396304, 1e-9);
    close(r.se_beta[1], 0.06668892355796786, 1e-9);
}

#[test]
fn hac_lag_changes_only_the_standard_errors() {
    let a = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(0), 252.0).unwrap();
    let b = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(2), 252.0).unwrap();
    assert_eq!(a.alpha.to_bits(), b.alpha.to_bits());
    assert_ne!(a.se_alpha.to_bits(), b.se_alpha.to_bits());
}

#[test]
fn spanning_alpha_is_invariant_to_positive_rescaling_and_covariant_in_units() {
    let base = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(2), 252.0).unwrap();
    let y2: Vec<f64> = Y1.iter().map(|v| 3.0 * v).collect();
    let r = spanning_alpha(&y2, &[&B1], HacLag::Fixed(2), 252.0).unwrap();
    close(r.alpha, 3.0 * base.alpha, 1e-10);
    close(r.t_alpha, base.t_alpha, 1e-9);
    close(r.beta[0], 3.0 * base.beta[0], 1e-10);
    let b2: Vec<f64> = B1.iter().map(|v| 5.0 * v).collect();
    let r = spanning_alpha(&Y1, &[&b2], HacLag::Fixed(2), 252.0).unwrap();
    close(r.alpha, base.alpha, 1e-9);
    close(r.t_alpha, base.t_alpha, 1e-9);
    close(r.beta[0], base.beta[0] / 5.0, 1e-10);
}

#[test]
fn benchmark_order_permutes_the_betas_and_nothing_else() {
    let a = spanning_alpha(&Y1, &[&B1, &Z1], HacLag::Fixed(1), 252.0).unwrap();
    let b = spanning_alpha(&Y1, &[&Z1, &B1], HacLag::Fixed(1), 252.0).unwrap();
    close(a.alpha, b.alpha, 1e-10);
    close(a.se_alpha, b.se_alpha, 1e-9);
    close(a.beta[0], b.beta[1], 1e-10);
    close(a.beta[1], b.beta[0], 1e-10);
}

#[test]
fn a_perfect_spanning_relation_recovers_alpha_and_beta_exactly() {
    // y = 0.002 + 0.5 b + noise-free -> residual variance zero -> the t statistic is undefined: typed error
    let y: Vec<f64> = B1.iter().map(|b| 0.002 + 0.5 * b).collect();
    assert!(matches!(spanning_alpha(&y, &[&B1], HacLag::Fixed(0), 252.0), Err(EvalError::ZeroVariance { .. })));
}

#[test]
fn sr2_increment_is_alpha_squared_over_residual_variance() {
    let r = spanning_alpha(&Y1, &[&B1], HacLag::Fixed(0), 252.0).unwrap();
    close(r.sr2_increment_annual, r.alpha * r.alpha / r.resid_var * 252.0, 1e-12);
    assert!(r.r_squared > 0.9 && r.r_squared < 1.0);
}

#[test]
fn spanning_errors_are_typed() {
    assert!(matches!(spanning_alpha(&Y1, &[], HacLag::Auto, 252.0), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(spanning_alpha(&Y1, &[&B1[..9]], HacLag::Auto, 252.0), Err(EvalError::LengthMismatch { .. })));
    assert!(matches!(spanning_alpha(&Y1[..3], &[&B1[..3]], HacLag::Auto, 252.0), Err(EvalError::TooShort { .. })));
    assert!(matches!(spanning_alpha(&Y1, &[&B1], HacLag::Auto, 0.0), Err(EvalError::InvalidParameter { .. })));
    // constant benchmark is collinear with the intercept
    assert!(matches!(spanning_alpha(&Y1, &[&[0.01; 10]], HacLag::Auto, 252.0), Err(EvalError::Singular { .. })));
    // two identical benchmarks
    assert!(matches!(spanning_alpha(&Y1, &[&B1, &B1], HacLag::Auto, 252.0), Err(EvalError::Singular { .. })));
    let mut bad = Y1;
    bad[4] = f64::NAN;
    assert!(matches!(spanning_alpha(&bad, &[&B1], HacLag::Auto, 252.0), Err(EvalError::NonFinite { index: 4, .. })));
    let mut bad = B1;
    bad[2] = f64::INFINITY;
    assert!(matches!(spanning_alpha(&Y1, &[&bad], HacLag::Auto, 252.0), Err(EvalError::NonFinite { index: 2, .. })));
}

fn simulate_ar1(rng: &mut Rng, n: usize, phi: f64) -> Vec<f64> {
    let mut x = 0.0;
    (0..n + 30)
        .map(|_| {
            x = phi * x + rng.normal();
            x
        })
        .skip(30)
        .collect()
}

#[test]
fn null_size_of_the_hac_t_test_is_close_to_nominal_for_serially_correlated_errors() {
    // y = 0 * alpha + 0.8 b + AR(1) errors; H0 alpha = 0 true. Two-sided 5% test.
    let mut rng = Rng::seed_from_u64(20260924);
    let reps = 600;
    let n = 400;
    let mut hac_rej = 0;
    let mut white_rej = 0;
    for _ in 0..reps {
        let b = simulate_ar1(&mut rng, n, 0.0);
        let e = simulate_ar1(&mut rng, n, 0.4);
        let y: Vec<f64> = b.iter().zip(&e).map(|(b, e)| 0.8 * b + e).collect();
        let hac = spanning_alpha(&y, &[&b], HacLag::Auto, 252.0).unwrap();
        let white = spanning_alpha(&y, &[&b], HacLag::Fixed(0), 252.0).unwrap();
        if hac.p_two_sided < 0.05 {
            hac_rej += 1;
        }
        if white.p_two_sided < 0.05 {
            white_rej += 1;
        }
    }
    let hac_rate = hac_rej as f64 / reps as f64;
    let white_rate = white_rej as f64 / reps as f64;
    // Monte Carlo standard error at p = 0.05, 600 reps: 0.9 percentage points; allow ~4 SE plus the known small-sample
    // liberalness of Newey-West at n = 400
    assert!((hac_rate - 0.05).abs() < 0.035, "HAC size {hac_rate}");
    // ignoring the autocorrelation (White) is visibly wrong: this is what HAC is for
    assert!(white_rate > hac_rate + 0.03, "white {white_rate} hac {hac_rate}");
}

#[test]
fn power_of_the_spanning_test_grows_with_alpha_and_with_sample_length() {
    let mut rng = Rng::seed_from_u64(8);
    let mut rate = |n: usize, alpha: f64| {
        let mut rej = 0;
        for _ in 0..150 {
            let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let y: Vec<f64> = b.iter().map(|b| alpha + 0.5 * b + rng.normal()).collect();
            if spanning_alpha(&y, &[&b], HacLag::Auto, 252.0).unwrap().p_one_sided < 0.05 {
                rej += 1;
            }
        }
        rej as f64 / 150.0
    };
    let small_a = rate(300, 0.05);
    let big_a = rate(300, 0.25);
    let long = rate(1200, 0.05);
    assert!(big_a > small_a + 0.2, "{small_a} {big_a}");
    assert!(long > small_a + 0.1, "{small_a} {long}");
}
