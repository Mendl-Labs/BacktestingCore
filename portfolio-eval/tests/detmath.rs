//! The deterministic elementary functions against published constants and the platform libm, plus edge cases.

use portfolio_eval::detmath::*;

fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        ((a - b) / b).abs()
    }
}

#[test]
fn exp_matches_libm_over_the_whole_range() {
    let mut x = -700.0;
    while x < 700.0 {
        assert!(rel(exp(x), x.exp()) < 2e-15, "exp({x})");
        x += 3.7;
    }
    assert_eq!(exp(0.0), 1.0);
    assert!(rel(exp(1.0), std::f64::consts::E) < 2e-16);
    assert!(rel(exp(-1.0), 0.367_879_441_171_442_33) < 2e-16);
}

#[test]
fn exp_edge_cases() {
    assert!(exp(f64::NAN).is_nan());
    assert_eq!(exp(f64::INFINITY), f64::INFINITY);
    assert_eq!(exp(f64::NEG_INFINITY), 0.0);
    assert_eq!(exp(800.0), f64::INFINITY);
    assert_eq!(exp(-800.0), 0.0);
    assert!(exp(-740.0) > 0.0 && exp(-740.0) < 1e-320);
}

#[test]
fn ln_matches_libm_and_known_values() {
    for k in -300..300 {
        let x = 1.37_f64.powi(k % 40) * 10f64.powi(k / 40);
        assert!(rel(ln(x), x.ln()) < 5e-15 || (ln(x) - x.ln()).abs() < 1e-15, "ln({x})");
    }
    assert_eq!(ln(1.0), 0.0);
    assert!(rel(ln(2.0), std::f64::consts::LN_2) < 1e-16);
    assert!(rel(ln(10.0), std::f64::consts::LN_10) < 2e-16);
    // close to 1 the relative accuracy must survive
    assert!(rel(ln(1.0 + 1e-9), (1.0_f64 + 1e-9).ln()) < 1e-12);
    // a subnormal
    assert!(rel(ln(1e-310), (1e-310_f64).ln()) < 1e-14);
}

#[test]
fn ln_edge_cases() {
    assert!(ln(-1.0).is_nan());
    assert!(ln(f64::NAN).is_nan());
    assert_eq!(ln(0.0), f64::NEG_INFINITY);
    assert_eq!(ln(f64::INFINITY), f64::INFINITY);
}

#[test]
fn exp_and_ln_are_inverse() {
    for i in 1..200 {
        let x = i as f64 * 0.137 - 7.0;
        assert!((ln(exp(x)) - x).abs() < 1e-13, "{x}");
    }
}

#[test]
fn pow_matches_libm() {
    assert!(rel(pow(2.5, 2.0 / 9.0), 2.5_f64.powf(2.0 / 9.0)) < 1e-14);
    assert!(rel(pow(12.6, 2.0 / 9.0), 12.6_f64.powf(2.0 / 9.0)) < 1e-14);
    assert!(pow(-1.0, 0.5).is_nan());
    assert!(pow(0.0, 0.5).is_nan());
}

#[test]
fn sin_cos_2pi_matches_libm_and_identities() {
    for i in 0..1000 {
        let u = i as f64 / 1000.0;
        let (s, c) = sin_cos_2pi(u);
        let a = 2.0 * std::f64::consts::PI * u;
        assert!((s - a.sin()).abs() < 2e-15, "sin u={u}");
        assert!((c - a.cos()).abs() < 2e-15, "cos u={u}");
        assert!((s * s + c * c - 1.0).abs() < 1e-15, "pythagoras u={u}");
    }
    assert_eq!(sin_cos_2pi(0.0), (0.0, 1.0));
    let (s, c) = sin_cos_2pi(0.25);
    assert!((s - 1.0).abs() < 1e-15 && c.abs() < 1e-15);
    let (s, c) = sin_cos_2pi(0.5);
    assert!(s.abs() < 2e-15 && (c + 1.0).abs() < 1e-15);
}

#[test]
fn erfc_matches_published_values() {
    // Values from standard tables of the complementary error function.
    let table = [
        (0.0, 1.0),
        (0.5, 0.4795001221869534),
        (1.0, 0.15729920705028513),
        (2.0, 0.004677734981047265),
        (3.0, 2.209_049_699_858_544e-5),
        (5.0, 1.5374597944280351e-12),
        (10.0, 2.0884875837625446e-45),
    ];
    for (x, want) in table {
        assert!(rel(erfc(x), want) < 1e-13, "erfc({x}) = {} want {want}", erfc(x));
    }
    assert!(rel(erfc(-1.0), 2.0 - 0.157_299_207_050_285_13) < 1e-15);
    assert_eq!(erfc(30.0), 0.0);
    assert!(erfc(f64::NAN).is_nan());
}

#[test]
fn norm_cdf_symmetry_and_known_values() {
    assert_eq!(norm_cdf(0.0), 0.5);
    assert!(rel(norm_cdf(1.96), 0.9750021048517796) < 1e-13);
    assert!(rel(norm_cdf(-3.0), 0.0013498980316301035) < 1e-13);
    for i in -60..60 {
        let x = i as f64 * 0.1;
        assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-15, "{x}");
        assert!((norm_cdf(x) + norm_sf(x) - 1.0).abs() < 1e-15, "{x}");
    }
}

#[test]
fn norm_ppf_matches_published_quantiles() {
    let table = [
        (0.5, 0.0),
        (0.8, 0.8416212335729144),
        (0.9, 1.2815515655446008),
        (0.95, 1.6448536269514715),
        (0.975, 1.9599639845400536),
        (0.99, 2.3263478740408408),
        (0.999, 3.090_232_306_167_813),
        (0.999_999, 4.753424308817089),
    ];
    for (p, want) in table {
        assert!((norm_ppf(p) - want).abs() < 2e-15 * want.abs().max(1.0), "ppf({p}) = {} want {want}", norm_ppf(p));
        assert!((norm_ppf(1.0 - p) + want).abs() < 1e-14 * want.abs().max(1.0), "lower tail {p}");
    }
}

#[test]
fn norm_ppf_inverts_cdf_and_isf_inverts_sf() {
    for i in 1..200 {
        let p = i as f64 / 200.0;
        assert!((norm_cdf(norm_ppf(p)) - p).abs() < 1e-14, "{p}");
    }
    for k in 1..12 {
        let q = 10f64.powi(-k);
        let z = norm_isf(q);
        assert!(rel(norm_sf(z), q) < 1e-12, "isf({q}) = {z}");
    }
}

#[test]
fn norm_ppf_edge_cases() {
    assert!(norm_ppf(f64::NAN).is_nan());
    assert!(norm_ppf(-0.1).is_nan());
    assert!(norm_ppf(1.1).is_nan());
    assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
    assert_eq!(norm_ppf(1.0), f64::INFINITY);
    assert_eq!(norm_isf(0.0), f64::INFINITY);
    assert_eq!(norm_isf(1.0), f64::NEG_INFINITY);
    assert!(norm_isf(f64::NAN).is_nan());
}

#[test]
fn bit_patterns_are_pinned_across_platforms() {
    // These are the exact bits produced on aarch64 and required on x86_64 CI: the point of `detmath` is that they do not
    // depend on the platform libm.
    assert_eq!(exp(1.0).to_bits(), 0x4005_bf0a_8b14_5768); // one ulp below e: the series is accurate to ~1 ulp, and fixed
    assert_eq!(ln(2.0).to_bits(), 0x3fe6_2e42_fefa_39ef);
    assert_eq!(norm_ppf(0.975).to_bits(), 0x3fff_5c03_31ee_ff84); // within 2 ulp of the correctly rounded value
    assert_eq!(norm_cdf(1.0).to_bits(), 0x3fea_ec4b_d120_d37e);
    assert_eq!(erfc(2.0).to_bits(), 0x3f73_28f5_ec35_0e67);
    assert_eq!(norm_ppf(0.999).to_bits(), 0x4008_b8cb_b720_4470);
}
