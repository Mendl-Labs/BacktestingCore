//! Equivalence with Core on shared inputs. The golden numbers were produced by Core's OWN unmodified source
//! (`bash tests/reference/core_goldens.sh` copies `metrics/src/significance.rs`, `metrics/src/performance.rs` and
//! `quant-diagnostics/src/multiple_testing.rs` into a scratch crate and prints them); `dsr::core_compat` mirrors those
//! formulas and must agree. This also documents, in numbers, how the crate's own `deflated_sharpe` differs from Core's.

use portfolio_eval::detmath;
use portfolio_eval::dsr::{benjamini_hochberg_q, core_compat, deflated_sharpe, DsrInputs};
use portfolio_eval::stats;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    assert!((a - b).abs() <= rel * b.abs() + abs, "{a} vs {b}");
}

// Same deterministic series as the generator in tests/reference/core_goldens.sh
fn series(n: usize, mu: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let a = ((t * 37 + 11) % 101) as f64 / 101.0 - 0.5;
            let b = ((t * 53 + 7) % 89) as f64 / 89.0 - 0.5;
            mu + amp * (a + 0.6 * b * b * if t % 5 == 0 { 3.0 } else { 1.0 } - 0.1)
        })
        .collect()
}

#[test]
fn mirror_of_metrics_significance_deflated_sharpe_matches_core() {
    #[allow(clippy::type_complexity)]
    let goldens: [((f64, usize, usize, f64, f64), f64); 6] = [
        ((1.5, 1, 500, 1.0, 365.0), 0.96042314717601),
        ((1.5, 20, 500, 1.0, 365.0), 0.36793231094464063),
        ((2.0, 100, 1260, 0.5, 252.0), 0.9624451034084989),
        ((0.8, 1000, 900, 0.7, 252.0), 0.00411348055780403),
        ((-0.4, 10, 300, 1.0, 365.0), 0.04319372214452232),
        ((3.0, 5000, 2500, 1.27, 252.0), 5.59641632746835e-7),
    ];
    for ((obs, k, n, sd, ppy), want) in goldens {
        let got = core_compat::deflated_sharpe_significance(obs, k, n, sd, ppy);
        // exp/ln come from detmath instead of libm: agreement to 1e-12 relative, not bitwise
        close(got, want, 1e-11, 1e-15);
    }
    // Core's documented degenerate conventions
    assert_eq!(core_compat::deflated_sharpe_significance(1.0, 5, 0, 1.0, 252.0), 0.5);
    assert_eq!(core_compat::deflated_sharpe_significance(1.0, 5, 100, 0.0, 252.0), 0.5);
}

#[test]
fn mirror_of_metrics_performance_deflated_sharpe_matches_core() {
    let r1 = series(500, 0.0004, 0.01);
    let r2 = series(1260, 0.0002, 0.012);
    let r3 = series(120, -0.0003, 0.02);
    let goldens = [
        (&r1, 1u32, 0.6139508814110521),
        (&r1, 10, 0.10841618717571133),
        (&r1, 200, 0.0016892432582777395),
        (&r2, 1, 0.0037545682118280133),
        (&r2, 10, 3.512471449718113e-5),
        (&r2, 200, 6.716097900039131e-9),
        (&r3, 1, 0.02653754963867183),
        (&r3, 10, 0.000635171689586489),
        (&r3, 200, 4.5556427308302005e-7),
    ];
    for (r, k, want) in goldens {
        let got = core_compat::deflated_sharpe_performance(k, r).unwrap();
        close(got, want, 1e-10, 1e-18);
    }
    assert_eq!(core_compat::deflated_sharpe_performance(0, &r1), None);
    assert_eq!(core_compat::deflated_sharpe_performance(5, &r1[..3]), None);
    assert_eq!(core_compat::deflated_sharpe_performance(5, &[0.01; 50]), None);
}

#[test]
fn benjamini_hochberg_equals_core() {
    assert_eq!(benjamini_hochberg_q(&[0.01, 0.04, 0.03, 0.005]).unwrap(), vec![0.02, 0.04, 0.04, 0.02]);
    let q = benjamini_hochberg_q(&[0.001, 0.2, 0.05, 0.3, 0.02, 0.02]).unwrap();
    let core: [f64; 6] = [0.006, 0.24000000000000005, 0.07500000000000001, 0.3, 0.04, 0.04];
    for (a, b) in q.iter().zip(core) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn crate_dsr_equals_core_performance_dsr_when_the_same_inputs_are_supplied() {
    // Core's `performance::deflated_sharpe_ratio` is the Bailey moment-corrected variance with the floor and the crude
    // expected maximum sr_star = (1 - g + g ln N)/sqrt(n-1). Feeding the crate's function that same SR0 (through the
    // trial dispersion input) reproduces Core exactly: the two differ ONLY in the expected-maximum approximation.
    // a series whose per-period Sharpe is exactly 0.05, so the DSR values below are well inside (0, 1) and a wrong
    // expected maximum or variance would show
    let r0 = series(1260, 0.0, 0.012);
    let shift = 0.05 * stats::std_dev(&r0).unwrap() - stats::mean(&r0).unwrap();
    let r: Vec<f64> = r0.iter().map(|v| v + shift).collect();
    let n = r.len();
    // Core's `performance::deflated_sharpe_ratio` divides by the POPULATION standard deviation (ddof = 0)
    let sr = stats::sharpe_per_period(&r).unwrap() * (r.len() as f64 / (r.len() - 1) as f64).sqrt();
    let (skew, kurt) = (stats::skewness(&r).unwrap(), stats::kurtosis(&r).unwrap());
    for k in [10usize, 200] {
        let g = 0.5772156649015329;
        let sr_star = (1.0 - g + g * (k as f64).ln()) / ((n - 1) as f64).sqrt();
        // choose the dispersion so that dispersion x E[max normal] == sr_star
        let disp = sr_star / portfolio_eval::dsr::expected_max_normal(k);
        let ours = deflated_sharpe(
            &DsrInputs { sharpe: sr, n_obs: n, skewness: skew, kurtosis: kurt, n_trials: k, trial_sharpe_std: disp },
            true,
        )
        .unwrap();
        let core = core_compat::deflated_sharpe_performance(k as u32, &r).unwrap();
        let core_z = core_compat::deflated_sharpe_performance_z(k as u32, &r).unwrap();
        // with the SAME z, the crate's exact normal cdf and Core's inaccurate one differ (see the next test); the z
        // statistics themselves are identical
        assert!((ours.z - core_z).abs() < 1e-12, "k={k}: z {} vs {}", ours.z, core_z);
        assert!((ours.dsr - detmath::norm_cdf(core_z)).abs() < 1e-12);
        // the DSR values then differ by exactly Core's cdf error at that z (a few tenths of a percentage point here)
        assert!(core > 0.02 && core < 0.98, "k={k}: core {core} should be informative");
        assert!((ours.dsr - core).abs() < 0.04, "k={k}: ours {} core {}", ours.dsr, core);
    }
}

#[test]
fn where_the_crate_deliberately_differs_from_core() {
    // Core's significance DSR uses the iid variance 1/n and a Gumbel-corrected extreme-value expected maximum; the
    // crate uses the Bailey-Lopez de Prado moments-corrected variance and the (1-g)Phi^-1(1-1/N) + g Phi^-1(1-1/(Ne))
    // expected maximum. Same order of magnitude, not identical: recorded here so nobody expects bit equality.
    let (obs, k, n, sd, ppy) = (2.0, 100usize, 1260usize, 0.5, 252.0);
    let core = core_compat::deflated_sharpe_significance(obs, k, n, sd, ppy);
    let ours = deflated_sharpe(
        &DsrInputs {
            sharpe: obs / f64::sqrt(ppy),
            n_obs: n,
            skewness: 0.0,
            kurtosis: 3.0,
            n_trials: k,
            trial_sharpe_std: sd / f64::sqrt(ppy),
        },
        true,
    )
    .unwrap();
    assert!((core - ours.dsr).abs() < 0.05, "core {core} ours {}", ours.dsr);
    assert!((core - ours.dsr).abs() > 1e-9, "they are not the same formula");
}

#[test]
fn core_performance_normal_cdf_is_not_the_7_5e_8_approximation_its_comment_claims() {
    // FINDING (not fixed here: Core's metrics crate is not modified by this change). `metrics::performance::normal_cdf`
    // is documented as Abramowitz-Stegun 26.2.17 with max error 7.5e-8, but applies the 7.1.26 erf polynomial to x
    // instead of x/sqrt(2). Its true maximum absolute error is 0.037, which feeds `performance::deflated_sharpe_ratio`.
    let mut worst: f64 = 0.0;
    let mut at = 0.0;
    let mut x = -6.0;
    while x <= 6.0 {
        let d = (core_compat::normal_cdf_approx(x) - detmath::norm_cdf(x)).abs();
        if d > worst {
            worst = d;
            at = x;
        }
        x += 0.001;
    }
    assert!(worst > 0.036 && worst < 0.038, "max error {worst} at x = {at}");
    assert!((at - 0.567).abs() < 0.05, "at {at}");
    // the values Core would return
    assert!((core_compat::normal_cdf_approx(1.0) - 0.8703286406601964).abs() < 1e-12);
    assert!((detmath::norm_cdf(1.0) - 0.8413447460685429).abs() < 1e-12);
    assert!((core_compat::normal_cdf_approx(0.0) - 0.5).abs() < 1e-8);
}
