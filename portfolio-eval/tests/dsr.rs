//! Deflated Sharpe, PSR, MinTRL and BH against hand computations with `statistics.NormalDist` quantiles
//! (`tests/reference/gen_reference.py`), the published expected-maximum table, and properties.

use portfolio_eval::dsr::*;
use portfolio_eval::error::EvalError;
use portfolio_eval::rng::Rng;
use portfolio_eval::stats;

fn close(a: f64, b: f64, rel: f64) {
    assert!((a - b).abs() <= rel * b.abs().max(1e-12), "{a} vs {b}");
}

fn inputs(sr: f64, n: usize, skew: f64, kurt: f64, k: usize, sd: f64) -> DsrInputs {
    DsrInputs { sharpe: sr, n_obs: n, skewness: skew, kurtosis: kurt, n_trials: k, trial_sharpe_std: sd }
}

#[test]
fn expected_max_of_normals_matches_the_reference_formula_and_the_published_table() {
    // Bailey-Lopez de Prado approximation evaluated with an independent Phi^{-1}
    let table = [
        (1usize, 0.0),
        (2, 0.5197553442805939),
        (10, 1.57459830134575),
        (50, 2.276303093420348),
        (100, 2.5306028932016846),
        (1000, 3.255121513652723),
    ];
    for (n, want) in table {
        close(expected_max_normal(n), want, 1e-11);
    }
    assert_eq!(expected_max_normal(0), 0.0);
    // The approximation tracks the exact expected maximum of N standard normals (order-statistics tables: 0.5642,
    // 1.5388, 2.5076, 3.2414 for N = 2, 10, 100, 1000) to within 0.05 (worst at N = 2).
    for (n, exact) in [(2usize, 0.5642), (10, 1.5388), (100, 2.5076), (1000, 3.2414)] {
        assert!((expected_max_normal(n) - exact).abs() < 0.05, "N={n}");
    }
    // strictly increasing in the number of trials
    let mut last = -1.0;
    for n in 1..300 {
        let v = expected_max_normal(n);
        assert!(v >= last, "n={n}");
        last = v;
    }
}

#[test]
fn deflated_sharpe_matches_hand_computed_values() {
    let r = deflated_sharpe(&inputs(0.1, 1250, -0.5, 5.0, 50, 0.03), true).unwrap();
    close(r.dsr, 0.8618174925894071, 1e-11);
    close(r.sr0, 0.06828909280261043, 1e-11);
    close(r.se, 0.02913209472651295, 1e-11);
    close(r.z, 1.0885213540284715, 1e-11);
    // one trial: no deflation, SR0 = 0, DSR = PSR against zero
    let r1 = deflated_sharpe(&inputs(0.1, 1250, -0.5, 5.0, 1, 0.03), true).unwrap();
    assert_eq!(r1.sr0, 0.0);
    close(r1.dsr, 0.9997011326446887, 1e-11);
    close(probabilistic_sharpe(0.1, 0.0, 1250, -0.5, 5.0, true).unwrap(), 0.9997011326446887, 1e-11);
}

#[test]
fn variance_floor_only_widens_the_interval() {
    // positive skew makes the raw variance term SMALLER than the normal baseline; the floor undoes that
    let floored = deflated_sharpe(&inputs(0.05, 500, 0.8, 4.0, 20, 0.02), true).unwrap();
    let raw = deflated_sharpe(&inputs(0.05, 500, 0.8, 4.0, 20, 0.02), false).unwrap();
    close(floored.dsr, 0.6055515820745286, 1e-11);
    close(raw.dsr, 0.6075726235579431, 1e-11);
    close(floored.se, 0.04476614810358452, 1e-11);
    close(raw.se, 0.043904501026897476, 1e-11);
    assert!(floored.se >= raw.se);
    // an absurdly skewed sample would make the unfloored variance non-positive: refuse, do not return NaN
    assert!(matches!(
        deflated_sharpe(&inputs(2.0, 100, 20.0, 3.0, 5, 0.1), false),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(deflated_sharpe(&inputs(2.0, 100, 20.0, 3.0, 5, 0.1), true).is_ok());
}

#[test]
fn min_track_record_length_by_hand() {
    close(min_track_record_length(0.1, 0.0, -0.5, 5.0, 0.95).unwrap(), 287.7876061341135, 1e-11);
    // normal returns (skew 0, kurt 3), SR 0.05 vs benchmark 0.02, 97.5%: 1 + (1 + 0.5 x 0.0025) (1.96/0.03)^2
    close(min_track_record_length(0.05, 0.02, 0.0, 3.0, 0.975).unwrap(), 4274.622938022211, 1e-11);
    assert!(min_track_record_length(0.02, 0.05, 0.0, 3.0, 0.95).is_err());
    assert!(min_track_record_length(0.05, 0.05, 0.0, 3.0, 0.95).is_err());
    assert!(min_track_record_length(0.05, 0.0, 0.0, 3.0, 0.5).is_err());
    assert!(min_track_record_length(0.05, 0.0, 0.0, 3.0, 1.0).is_err());
    assert!(min_track_record_length(f64::NAN, 0.0, 0.0, 3.0, 0.95).is_err());
}

#[test]
fn dsr_properties_monotone_in_trials_sharpe_and_length() {
    // more trials -> lower DSR
    let mut last = 2.0;
    for k in [1usize, 2, 5, 20, 100, 1000, 10_000] {
        let d = deflated_sharpe(&inputs(0.08, 1000, 0.0, 3.0, k, 0.03), true).unwrap().dsr;
        assert!(d <= last + 1e-15, "k={k}");
        last = d;
    }
    // higher Sharpe -> higher DSR
    let mut last = -1.0;
    for i in 0..20 {
        let sr = -0.05 + 0.01 * i as f64;
        let d = deflated_sharpe(&inputs(sr, 1000, 0.0, 3.0, 50, 0.03), true).unwrap().dsr;
        assert!(d >= last, "sr={sr}");
        last = d;
    }
    // for a fixed positive per-period Sharpe well above SR0, more observations -> higher DSR
    let mut last = 0.0;
    for n in [100usize, 200, 400, 800, 1600, 3200] {
        let d = deflated_sharpe(&inputs(0.08, n, 0.0, 3.0, 20, 0.02), true).unwrap().dsr;
        assert!(d > last, "n={n}");
        last = d;
    }
    // larger trial dispersion -> higher hurdle -> lower DSR; zero dispersion means no deflation
    let lo = deflated_sharpe(&inputs(0.08, 1000, 0.0, 3.0, 50, 0.01), true).unwrap().dsr;
    let hi = deflated_sharpe(&inputs(0.08, 1000, 0.0, 3.0, 50, 0.05), true).unwrap().dsr;
    let none = deflated_sharpe(&inputs(0.08, 1000, 0.0, 3.0, 50, 0.0), true).unwrap();
    assert!(hi < lo && lo < none.dsr);
    assert_eq!(none.sr0, 0.0);
}

#[test]
fn dsr_of_a_series_is_invariant_to_positive_scaling_and_uses_the_series_moments() {
    let mut rng = Rng::seed_from_u64(4);
    let r: Vec<f64> = (0..600).map(|_| 0.0006 + 0.01 * rng.normal()).collect();
    let a = deflated_sharpe_from_returns(&r, 30, 0.6, 252.0, true).unwrap();
    let r2: Vec<f64> = r.iter().map(|v| v * 5.0).collect();
    let b = deflated_sharpe_from_returns(&r2, 30, 0.6, 252.0, true).unwrap();
    close(a.dsr, b.dsr, 1e-10);
    // equals the explicit-moment call
    let sr = stats::sharpe_per_period(&r).unwrap();
    let direct = deflated_sharpe(
        &inputs(sr, 600, stats::skewness(&r).unwrap(), stats::kurtosis(&r).unwrap(), 30, 0.6 / 252f64.sqrt()),
        true,
    )
    .unwrap();
    close(a.dsr, direct.dsr, 1e-12);
    // annual dispersion is converted with sqrt(ppy): the same annual number at 365 bars is a different per-period hurdle
    let c = deflated_sharpe_from_returns(&r, 30, 0.6, 365.0, true).unwrap();
    assert!(c.sr0 < a.sr0);
}

#[test]
fn dsr_typed_errors() {
    assert!(matches!(
        deflated_sharpe(&inputs(0.1, 1250, 0.0, 3.0, 0, 0.03), true),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(deflated_sharpe(&inputs(0.1, 2, 0.0, 3.0, 5, 0.03), true), Err(EvalError::TooShort { .. })));
    assert!(matches!(
        deflated_sharpe(&inputs(f64::NAN, 100, 0.0, 3.0, 5, 0.03), true),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        deflated_sharpe(&inputs(0.1, 100, f64::INFINITY, 3.0, 5, 0.03), true),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        deflated_sharpe(&inputs(0.1, 100, 0.0, 3.0, 5, -0.01), true),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        deflated_sharpe(&inputs(0.1, 100, 0.0, 3.0, 5, f64::NAN), true),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        deflated_sharpe_from_returns(&[0.01; 50], 5, 0.5, 252.0, true),
        Err(EvalError::ZeroVariance { .. })
    ));
    assert!(matches!(
        deflated_sharpe_from_returns(&[0.01, -0.01, 0.02], 5, 0.5, 252.0, true),
        Err(EvalError::TooShort { .. })
    ));
    assert!(deflated_sharpe_from_returns(&[0.01, -0.01, 0.02, 0.0, 0.03], 5, 0.5, 0.0, true).is_err());
    assert!(probabilistic_sharpe(0.1, 0.0, 1, 0.0, 3.0, true).is_err());
}

#[test]
fn benjamini_hochberg_hand_computed_and_properties() {
    let q = benjamini_hochberg_q(&[0.01, 0.04, 0.03, 0.005]).unwrap();
    assert_eq!(q, vec![0.02, 0.04, 0.04, 0.02]);
    let q = benjamini_hochberg_q(&[0.001, 0.2, 0.05, 0.3, 0.02, 0.02]).unwrap();
    for (a, b) in q.iter().zip([0.006, 0.24, 0.075, 0.3, 0.04, 0.04]) {
        assert!((a - b).abs() < 1e-12, "{a} {b}");
    }
    assert!(benjamini_hochberg_q(&[]).unwrap().is_empty());
    assert_eq!(benjamini_hochberg_q(&[0.03]).unwrap(), vec![0.03]);
    // q >= p, capped at 1, monotone in p, and permutation-equivariant
    let mut rng = Rng::seed_from_u64(6);
    let p: Vec<f64> = (0..40).map(|_| rng.unit()).collect();
    let q = benjamini_hochberg_q(&p).unwrap();
    for (pi, qi) in p.iter().zip(&q) {
        assert!(*qi >= *pi - 1e-15 && *qi <= 1.0);
    }
    let mut pairs: Vec<(f64, f64)> = p.iter().copied().zip(q.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for w in pairs.windows(2) {
        assert!(w[1].1 >= w[0].1 - 1e-15);
    }
    let mut perm: Vec<usize> = (0..40).collect();
    for i in (1..40).rev() {
        perm.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let pp: Vec<f64> = perm.iter().map(|i| p[*i]).collect();
    let qp = benjamini_hochberg_q(&pp).unwrap();
    for (k, i) in perm.iter().enumerate() {
        assert_eq!(qp[k].to_bits(), q[*i].to_bits());
    }
    // errors
    assert!(matches!(benjamini_hochberg_q(&[0.1, f64::NAN]), Err(EvalError::NonFinite { index: 1, .. })));
    assert!(benjamini_hochberg_q(&[1.5]).is_err());
    assert!(benjamini_hochberg_q(&[-0.1]).is_err());
}

#[test]
fn bh_with_more_candidates_is_more_conservative() {
    // the same raw p-values, embedded in a larger family, get larger q-values (the price of screening more candidates)
    let small = benjamini_hochberg_q(&[0.004, 0.02]).unwrap();
    let big = benjamini_hochberg_q(&[0.004, 0.02, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    assert!(big[0] > small[0] && big[1] > small[1]);
}
