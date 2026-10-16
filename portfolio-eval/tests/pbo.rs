//! Probability of backtest overfitting (CSCV) on constructed examples with known answers, a brute-force cross-check
//! (`tests/reference/gen_reference.py`), and properties.

use portfolio_eval::error::EvalError;
use portfolio_eval::pbo::*;
use portfolio_eval::rng::Rng;

const A: [f64; 16] =
    [0.02, 0.03, 0.01, 0.02, -0.02, -0.01, -0.03, -0.02, 0.02, 0.03, 0.01, 0.02, -0.02, -0.01, -0.03, -0.02];

fn b_anti() -> Vec<f64> {
    (0..16).map(|i| -A[i] + 0.001 * (((i * 7) % 5) as f64 - 2.0)).collect()
}
fn c_noise() -> Vec<f64> {
    (0..16).map(|i| 0.001 * ((((i * 5) % 7) as f64) - 3.0)).collect()
}
fn d_alt() -> Vec<f64> {
    (0..16).map(|i| 0.004 * if i % 2 == 0 { 1.0 } else { -1.0 } + 0.0005 * (i % 3) as f64).collect()
}

#[test]
fn two_configurations_that_alternate_regimes_are_fully_overfit() {
    // S=2 blocks: config A wins block 0 and loses block 1, B the opposite. Choosing the in-sample winner ALWAYS picks
    // the out-of-sample loser: omega = 1/3, lambda = ln(1/2) < 0 in both splits -> PBO = 1.
    let a: Vec<f64> = (0..8).map(|i| if i < 4 { 0.02 + 0.001 * i as f64 } else { -0.02 + 0.001 * i as f64 }).collect();
    let b: Vec<f64> = (0..8).map(|i| if i < 4 { -0.02 - 0.001 * i as f64 } else { 0.02 - 0.001 * i as f64 }).collect();
    let r = pbo_cscv(&[&a, &b], 2, PboMetric::Mean).unwrap();
    assert_eq!(r.n_splits, 2);
    assert_eq!(r.pbo, 1.0);
    for l in &r.logits {
        assert!((l - (1.0f64 / 2.0).ln()).abs() < 1e-12, "{l}");
    }
    assert_eq!(r.selected, vec![0, 1]);
    assert_eq!(r.prob_loss, 1.0);
    assert!(r.degradation_slope.is_finite());
}

#[test]
fn a_dominant_configuration_has_zero_pbo() {
    let dom: Vec<f64> = (0..16).map(|i| 0.01 + 0.002 * (((i * 3) % 4) as f64)).collect();
    let worse: Vec<f64> = (0..16).map(|i| dom[i] - 0.02 + 0.001 * (i % 2) as f64).collect();
    let r = pbo_cscv(&[&dom, &worse], 4, PboMetric::Sharpe).unwrap();
    assert_eq!(r.pbo, 0.0);
    assert_eq!(r.n_splits, 6);
    assert!(r.selected.iter().all(|s| *s == 0));
    // omega = 2/3 every time -> lambda = ln 2
    for l in &r.logits {
        assert!((l - 2.0f64.ln()).abs() < 1e-12);
    }
    assert_eq!(r.prob_loss, 0.0);
}

#[test]
fn matches_the_brute_force_reference_with_three_and_four_configurations() {
    let (b, c, d) = (b_anti(), c_noise(), d_alt());
    let r = pbo_cscv(&[&A, &b, &c], 4, PboMetric::Sharpe).unwrap();
    assert_eq!(r.pbo, 1.0);
    let want = [0.0, -1.0986122886681098, 0.0, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098];
    assert_eq!(r.n_splits, 6);
    for (l, w) in r.logits.iter().zip(want) {
        assert!((l - w).abs() < 1e-12, "{l} vs {w}");
    }
    let r = pbo_cscv(&[&A, &b, &c, &d], 8, PboMetric::Sharpe).unwrap();
    assert_eq!(r.n_splits, 70);
    assert!((r.pbo - 0.7714285714285715).abs() < 1e-15, "{}", r.pbo);
    let want = [
        0.4054651081081642,
        -1.3862943611198906,
        -1.3862943611198906,
        -0.4054651081081643,
        1.3862943611198908,
        -1.3862943611198906,
    ];
    for (l, w) in r.logits.iter().zip(want) {
        assert!((l - w).abs() < 1e-10, "{l} vs {w}");
    }
}

#[test]
fn pure_noise_configurations_have_pbo_near_one_half() {
    // N iid noise strategies: the IS winner is a coin flip out of sample: PBO ~ 0.5 (Bailey et al. 2015)
    let mut rng = Rng::seed_from_u64(555);
    let cfgs: Vec<Vec<f64>> = (0..20).map(|_| (0..400).map(|_| 0.01 * rng.normal()).collect()).collect();
    let refs: Vec<&[f64]> = cfgs.iter().map(|c| c.as_slice()).collect();
    let r = pbo_cscv(&refs, 8, PboMetric::Sharpe).unwrap();
    assert_eq!(r.n_splits, 70);
    assert!(r.pbo > 0.2 && r.pbo < 0.8, "pbo {}", r.pbo);
    // and the IS-OOS relation of the winners is negative or flat (selection on noise degrades)
    assert!(r.degradation_slope < 0.3, "{}", r.degradation_slope);
}

#[test]
fn a_planted_edge_gives_low_pbo() {
    let mut rng = Rng::seed_from_u64(556);
    let mut cfgs: Vec<Vec<f64>> = (0..20).map(|_| (0..400).map(|_| 0.01 * rng.normal()).collect()).collect();
    cfgs[7] = (0..400).map(|_| 0.004 + 0.01 * rng.normal()).collect(); // SR 0.4 per period: a strong persistent edge
    let refs: Vec<&[f64]> = cfgs.iter().map(|c| c.as_slice()).collect();
    let r = pbo_cscv(&refs, 8, PboMetric::Sharpe).unwrap();
    assert!(r.pbo < 0.15, "pbo {}", r.pbo);
    assert!(r.selected.iter().filter(|s| **s == 7).count() as f64 > 0.8 * r.n_splits as f64);
}

#[test]
fn invariant_to_configuration_order_and_to_positive_rescaling() {
    let mut rng = Rng::seed_from_u64(557);
    let cfgs: Vec<Vec<f64>> = (0..6).map(|_| (0..96).map(|_| 0.01 * rng.normal()).collect()).collect();
    let refs: Vec<&[f64]> = cfgs.iter().map(|c| c.as_slice()).collect();
    let base = pbo_cscv(&refs, 6, PboMetric::Sharpe).unwrap();
    let mut rev: Vec<&[f64]> = refs.clone();
    rev.reverse();
    let r = pbo_cscv(&rev, 6, PboMetric::Sharpe).unwrap();
    assert_eq!(r.pbo, base.pbo);
    for (a, b) in r.logits.iter().zip(&base.logits) {
        assert!((a - b).abs() < 1e-12);
    }
    let scaled: Vec<Vec<f64>> =
        cfgs.iter().enumerate().map(|(i, c)| c.iter().map(|v| v * (i as f64 + 1.5)).collect()).collect();
    let srefs: Vec<&[f64]> = scaled.iter().map(|c| c.as_slice()).collect();
    let s = pbo_cscv(&srefs, 6, PboMetric::Sharpe).unwrap();
    assert_eq!(s.pbo, base.pbo, "the Sharpe-based PBO is invariant to per-configuration positive scaling");
    assert_eq!(s.selected, base.selected);
}

#[test]
fn leading_remainder_rows_are_dropped_not_the_recent_ones() {
    // 18 rows, 4 blocks -> block length 4, the first 2 rows are dropped. Poisoning the first two rows changes nothing.
    let mut rng = Rng::seed_from_u64(558);
    let cfgs: Vec<Vec<f64>> = (0..3).map(|_| (0..18).map(|_| 0.01 * rng.normal()).collect()).collect();
    let refs: Vec<&[f64]> = cfgs.iter().map(|c| c.as_slice()).collect();
    let a = pbo_cscv(&refs, 4, PboMetric::Mean).unwrap();
    let mut poisoned = cfgs.clone();
    for c in poisoned.iter_mut() {
        c[0] = 5.0;
        c[1] = -5.0;
    }
    let prefs: Vec<&[f64]> = poisoned.iter().map(|c| c.as_slice()).collect();
    let b = pbo_cscv(&prefs, 4, PboMetric::Mean).unwrap();
    assert_eq!(a, b);
    // whereas changing the LAST rows does change the answer's inputs
    let mut tail = cfgs.clone();
    for (i, c) in tail.iter_mut().enumerate() {
        c[17] = 0.5 * i as f64;
    }
    let trefs: Vec<&[f64]> = tail.iter().map(|c| c.as_slice()).collect();
    assert_ne!(pbo_cscv(&trefs, 4, PboMetric::Mean).unwrap().logits, a.logits);
}

#[test]
fn ties_get_the_mid_rank_and_flat_configurations_are_neutral() {
    // two identical configurations tie everywhere: the selected one (lowest index) has one equal partner
    let x: Vec<f64> = (0..16).map(|i| 0.01 * ((i % 4) as f64 - 1.5)).collect();
    let r = pbo_cscv(&[&x, &x], 4, PboMetric::Mean).unwrap();
    // rank = 1 + 0.5 = 1.5 of 2 -> omega = 0.5 -> lambda = 0 -> counted as overfit (lambda <= 0)
    assert!(r.logits.iter().all(|l| l.abs() < 1e-12));
    assert_eq!(r.pbo, 1.0);
    assert!(r.selected.iter().all(|s| *s == 0), "in-sample ties go to the lowest index");
    // a configuration that never trades (all zeros) has Sharpe 0, not an error
    let flat = vec![0.0; 16];
    let good: Vec<f64> = (0..16).map(|i| 0.01 + 0.002 * ((i * 3 % 4) as f64)).collect();
    let r = pbo_cscv(&[&flat, &good], 4, PboMetric::Sharpe).unwrap();
    assert_eq!(r.pbo, 0.0);
}

#[test]
fn typed_errors() {
    let x = vec![0.01; 16];
    let y: Vec<f64> = (0..16).map(|i| i as f64 * 0.001).collect();
    assert!(matches!(pbo_cscv(&[&x], 4, PboMetric::Sharpe), Err(EvalError::TooShort { need: 2, got: 1, .. })));
    assert!(matches!(pbo_cscv(&[], 4, PboMetric::Sharpe), Err(EvalError::TooShort { .. })));
    assert!(matches!(pbo_cscv(&[&x, &y], 3, PboMetric::Sharpe), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(pbo_cscv(&[&x, &y], 0, PboMetric::Sharpe), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(pbo_cscv(&[&x, &y[..15]], 4, PboMetric::Sharpe), Err(EvalError::LengthMismatch { .. })));
    assert!(matches!(pbo_cscv(&[&x, &y], 10, PboMetric::Sharpe), Err(EvalError::TooShort { .. })));
    let mut bad = y.clone();
    bad[5] = f64::NAN;
    assert!(matches!(pbo_cscv(&[&x, &bad], 4, PboMetric::Sharpe), Err(EvalError::NonFinite { index: 5, .. })));
    let long: Vec<f64> = (0..400).map(|i| i as f64).collect();
    assert!(pbo_cscv(&[&long, &long], 40, PboMetric::Mean).is_err(), "too many combinations");
}
