//! Basic statistics against exact rational hand computations (`tests/reference/gen_reference.py`) and typed errors.

use portfolio_eval::error::EvalError;
use portfolio_eval::stats::*;

const X: [f64; 8] = [0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, -0.01];
const Y: [f64; 8] = [0.005, -0.003, 0.008, 0.001, -0.002, 0.006, 0.0, 0.004];

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1e-300).max(1.0e-12), "{a} vs {b}");
}

#[test]
fn mean_variance_std_sharpe_match_exact_arithmetic() {
    // mean = 0.04/8 = 0.005 exactly; sum of squared deviations = 0.001950 (hand: 25+625+625+25+100+100+225+225 = 1950 e-6)
    close(mean(&X).unwrap(), 0.005, 1e-15);
    close(variance(&X).unwrap(), 0.0002785714285714286, 1e-14);
    close(std_dev(&X).unwrap(), 0.0002785714285714286_f64.sqrt(), 1e-14);
    close(sharpe_per_period(&X).unwrap(), 0.299572344757639, 1e-13);
    close(sharpe_annual(&X, 252.0).unwrap(), 0.299572344757639 * 252f64.sqrt(), 1e-13);
}

#[test]
fn ddof_is_one_not_zero() {
    // two points 0 and 2: mean 1, squared deviations 1+1: variance 2 with ddof 1 (1 with ddof 0)
    assert_eq!(variance(&[0.0, 2.0]).unwrap(), 2.0);
    assert_eq!(std_dev(&[0.0, 2.0]).unwrap(), 2.0_f64.sqrt());
}

#[test]
fn skewness_and_kurtosis_use_population_moments() {
    // [0,0,0,4]: mean 1, deviations -1,-1,-1,3 -> m2 = 3, m3 = 6, m4 = 21: skew 6/3^1.5, kurt 21/9
    let d = [0.0, 0.0, 0.0, 4.0];
    close(skewness(&d).unwrap(), 6.0 / 3.0_f64.powf(1.5), 1e-14);
    close(kurtosis(&d).unwrap(), 21.0 / 9.0, 1e-14);
    // a symmetric sample has zero skew (exact hand result for X)
    assert!(skewness(&X).unwrap().abs() < 1e-12);
    close(kurtosis(&X).unwrap(), 1.9013806706114398, 1e-13);
}

#[test]
fn autocovariance_and_correlation() {
    close(autocovariance(&X, 0).unwrap(), 0.00024375, 1e-13);
    close(autocovariance(&X, 1).unwrap(), -0.000175, 1e-12);
    close(correlation(&X, &Y).unwrap(), 0.33486129192298497, 1e-13);
    // correlation is invariant to positive scale and shift, and flips sign with a negative scale
    let y2: Vec<f64> = Y.iter().map(|v| 5.0 * v + 3.0).collect();
    close(correlation(&X, &y2).unwrap(), correlation(&X, &Y).unwrap(), 1e-12);
    let y3: Vec<f64> = Y.iter().map(|v| -2.0 * v).collect();
    close(correlation(&X, &y3).unwrap(), -correlation(&X, &Y).unwrap(), 1e-12);
}

#[test]
fn quantile_median_mad() {
    let v = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(quantile_sorted(&v, 0.0), 1.0);
    assert_eq!(quantile_sorted(&v, 1.0), 4.0);
    assert_eq!(quantile_sorted(&v, 0.5), 2.5);
    assert_eq!(quantile_sorted(&v, 1.0 / 3.0), 2.0);
    assert_eq!(quantile_sorted(&v, 2.0), 4.0); // clamped
    assert!(quantile_sorted(&[], 0.5).is_nan());
    assert_eq!(median(&[5.0, 1.0, 3.0]).unwrap(), 3.0);
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    // MAD of [1,2,3,4,100]: median 3, deviations 2,1,0,1,97 -> median 1 -> 1.4826
    assert!((mad_scale(&[1.0, 2.0, 3.0, 4.0, 100.0]).unwrap() - 1.4826).abs() < 1e-12);
    // a wild outlier barely moves the robust scale but explodes the plain one
    let mut sharpes: Vec<f64> = (0..50).map(|i| 0.5 + 0.02 * (i as f64 - 25.0) / 25.0).collect();
    let before = mad_scale(&sharpes).unwrap();
    sharpes.push(400.0);
    let after = mad_scale(&sharpes).unwrap();
    assert!((after - before).abs() < 0.02, "{before} {after}");
    assert!(std_dev(&sharpes).unwrap() > 50.0);
}

#[test]
fn typed_errors_never_panics() {
    assert!(matches!(mean(&[]), Err(EvalError::TooShort { need: 1, got: 0, .. })));
    assert!(matches!(mean(&[1.0, f64::NAN]), Err(EvalError::NonFinite { index: 1, .. })));
    assert!(matches!(mean(&[f64::INFINITY]), Err(EvalError::NonFinite { index: 0, .. })));
    assert!(matches!(variance(&[1.0]), Err(EvalError::TooShort { need: 2, got: 1, .. })));
    assert!(matches!(sharpe_per_period(&[0.01, 0.01, 0.01]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(sharpe_per_period(&[0.0; 10]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(sharpe_per_period(&[0.1; 50]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(sharpe_annual(&X, 0.0), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(sharpe_annual(&X, f64::NAN), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(sharpe_annual(&X, -1.0), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(skewness(&[1.0, 2.0]), Err(EvalError::TooShort { .. })));
    assert!(matches!(kurtosis(&[1.0, 2.0, 3.0]), Err(EvalError::TooShort { .. })));
    assert!(matches!(skewness(&[2.0; 6]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(autocovariance(&X, 8), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(correlation(&X, &Y[..5]), Err(EvalError::LengthMismatch { .. })));
    assert!(matches!(correlation(&X, &[1.0; 8]), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(median(&[f64::NAN]), Err(EvalError::NonFinite { .. })));
}

#[test]
fn sharpe_is_invariant_to_positive_scale_and_flips_sign_when_negated() {
    let base = sharpe_per_period(&X).unwrap();
    for k in [1e-6, 0.5, 3.0, 1e6] {
        let s: Vec<f64> = X.iter().map(|v| k * v).collect();
        close(sharpe_per_period(&s).unwrap(), base, 1e-12);
    }
    let neg: Vec<f64> = X.iter().map(|v| -v).collect();
    close(sharpe_per_period(&neg).unwrap(), -base, 1e-12);
    // permutation invariance (the Sharpe of a set of returns does not depend on their order)
    let mut p = X.to_vec();
    p.reverse();
    close(sharpe_per_period(&p).unwrap(), base, 1e-12);
}
