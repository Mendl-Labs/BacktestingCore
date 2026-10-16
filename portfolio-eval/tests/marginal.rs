//! The paired equal-volatility marginal test: agreement with an independent implementation (same RNG and index
//! scheme, plain float arithmetic, `tests/reference/gen_reference.py`), the algebra of equal-vol scaling, one- versus
//! two-sided p-values, the analytic minimum detectable effect, invariance/permutation/monotonicity properties and a
//! null-size check. The full size/power grid is `tests/power_table.rs`.

use portfolio_eval::error::EvalError;
use portfolio_eval::hac::HacLag;
use portfolio_eval::marginal::*;
use portfolio_eval::rng::Rng;
use portfolio_eval::stats;

fn close(a: f64, b: f64, rel: f64) {
    assert!((a - b).abs() <= rel * b.abs().max(1e-12), "{a} vs {b}");
}

/// Same construction as `marginal_series` in gen_reference.py (integer arithmetic, then the same float expressions).
fn fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let base: Vec<f64> = (0..n).map(|t| (((t * 37 + 11) % 23) as f64 - 11.0) / 1000.0).collect();
    let comb: Vec<f64> = (0..n).map(|t| 0.8 * base[t] + (((t * 29 + 5) % 19) as f64 - 9.0) / 1500.0 + 0.0004).collect();
    (base, comb)
}

fn cfg(n_boot: usize, block: f64, seed: u64) -> MarginalConfig {
    let mut c = MarginalConfig::new(252.0);
    c.n_boot = n_boot;
    c.block = BlockLength::Fixed(block);
    c.seed = seed;
    c
}

#[test]
fn matches_the_independent_implementation_in_sample_scaling() {
    let (b, c) = fixture(60);
    let r = marginal_contribution(&b, &c, &cfg(199, 3.0, 99)).unwrap();
    close(r.delta_sharpe, 0.930356313267067, 1e-11);
    close(r.boot_se, 1.0307463218743023, 1e-9);
    assert_eq!(r.p_one_sided, 0.21);
    assert_eq!(r.p_two_sided, 0.38);
    close(r.ci_low, -1.0662136422624215, 1e-9);
    close(r.ci_high, 2.9667267552229033, 1e-9);
    close(r.mde, 2.5629248169184295, 1e-9);
    close(r.sharpe_base, 0.35997210977896693, 1e-11);
    close(r.sharpe_combined, 1.290328423046034, 1e-11);
    assert_eq!(r.n, 60);
    assert_eq!(r.n_boot, 199);
    assert_eq!(r.n_valid_boot, 199);
    assert_eq!(r.block_length, 3.0);
    assert!(!r.significant() && r.inconclusive());
}

#[test]
fn matches_the_independent_implementation_iid_block_and_ex_ante_scaling() {
    let (b, c) = fixture(60);
    let r = marginal_contribution(&b, &c, &cfg(99, 1.0, 5)).unwrap();
    close(r.delta_sharpe, 0.930356313267067, 1e-11);
    close(r.boot_se, 1.3952454508575265, 1e-9);
    assert_eq!(r.p_one_sided, 0.29);
    assert_eq!(r.p_two_sided, 0.52);
    close(r.ci_low, -1.3143134132870014, 1e-9);
    close(r.ci_high, 4.56261662430911, 1e-9);

    // ex-ante constants: 1.1 x and 0.9 x the in-sample volatilities
    let sd0 = stats::std_dev(&b).unwrap();
    let sd1 = stats::std_dev(&c).unwrap();
    let mut k = cfg(99, 6.5, 77);
    k.scale = ScaleMode::ExAnte { base_vol: sd0 * 1.1, combined_vol: sd1 * 0.9 };
    let r = marginal_contribution(&b, &c, &k).unwrap();
    close(r.delta_sharpe, 1.106450875302593, 1e-11);
    close(r.boot_se, 0.9732638118918873, 1e-9);
    assert_eq!(r.p_one_sided, 0.12);
    assert_eq!(r.p_two_sided, 0.22);
    close(r.mde, 2.4199960009273123, 1e-9);
    // the reported in-sample Sharpe ratios do not depend on the ex-ante constants
    close(r.sharpe_base, 0.35997210977896693, 1e-11);
}

#[test]
fn matches_the_independent_implementation_on_a_planted_shift() {
    let b: Vec<f64> = (0..80).map(|t| (((t * 41 + 3) % 27) as f64 - 13.0) / 900.0).collect();
    let c: Vec<f64> = (0..80).map(|t| b[t] + 0.0006 + (((t * 13 + 1) % 17) as f64 - 8.0) / 4000.0).collect();
    let r = marginal_contribution(&b, &c, &cfg(199, 2.0, 3)).unwrap();
    close(r.delta_sharpe, 1.135518162716911, 1e-11);
    close(r.boot_se, 0.20703461984399577, 1e-9);
    assert_eq!(r.p_one_sided, 0.005);
    assert_eq!(r.p_two_sided, 0.005);
    assert!(r.significant());
    close(r.ci_low, 0.77945108144078, 1e-9);
    close(r.ci_high, 1.5683733256536614, 1e-9);
    assert_eq!(r.p_one_sided, 1.0 / 200.0, "the smallest attainable p is 1/(B+1), never 0");
}

#[test]
fn delta_sharpe_point_estimate_by_hand() {
    // 40 alternating points. base: 0.03, 0.01 -> mean 0.02, deviations +-0.01; combined: 0.10, 0.02 -> mean 0.06,
    // deviations +-0.04. With ddof 1 both standard deviations carry the factor k = sqrt(40/39):
    // SR_base = 0.02 / (0.01 k) = 2/k, SR_comb = 0.06 / (0.04 k) = 1.5/k, so Delta = -0.5/k x sqrt(252): the combined
    // book has the HIGHER mean but the LOWER Sharpe, and equal-volatility scaling says it is worse.
    let base: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 0.03 } else { 0.01 }).collect();
    let comb: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 0.10 } else { 0.02 }).collect();
    let k = (40.0f64 / 39.0).sqrt();
    let d = delta_sharpe(&base, &comb, 252.0, ScaleMode::InSample).unwrap();
    close(d, -0.5 / k * 252f64.sqrt(), 1e-12);
    assert!(d < 0.0);
    assert!(stats::mean(&comb).unwrap() > stats::mean(&base).unwrap());
    // ex-ante constants equal to the true per-period volatilities give the same number
    let e =
        delta_sharpe(&base, &comb, 252.0, ScaleMode::ExAnte { base_vol: 0.01 * k, combined_vol: 0.04 * k }).unwrap();
    close(e, d, 1e-12);
    // and doubling the ex-ante volatility of the combined book halves its contribution
    let e2 =
        delta_sharpe(&base, &comb, 252.0, ScaleMode::ExAnte { base_vol: 0.01 * k, combined_vol: 0.08 * k }).unwrap();
    close(e2, (0.75 / k - 2.0 / k) * 252f64.sqrt(), 1e-12);
}

#[test]
fn equal_vol_scaling_a_levered_copy_adds_nothing() {
    let (b0, _) = fixture(200);
    let b: Vec<f64> = b0.iter().map(|v| v + 0.002).collect(); // a clearly positive mean
                                                              // combined = 3 x base: same Sharpe, three times the mean. Equal-vol delta is 0; the raw mean difference is huge.
    let c: Vec<f64> = b.iter().map(|v| 3.0 * v).collect();
    let d = delta_sharpe(&b, &c, 252.0, ScaleMode::InSample).unwrap();
    assert!(d.abs() < 1e-12, "levered copy must have delta 0, got {d}");
    let raw = (stats::mean(&c).unwrap() - stats::mean(&b).unwrap()) * 252.0;
    assert!(raw > 0.5, "the unscaled mean difference {raw} would call the levered copy a big improvement");
    let r = marginal_contribution(&b, &c, &cfg(199, 2.0, 1)).unwrap();
    assert!(r.delta_sharpe.abs() < 1e-12);
    assert!(!r.significant(), "a levered copy must not pass the test (p = {})", r.p_one_sided);
    assert!(r.p_one_sided > 0.3, "p = {}", r.p_one_sided);
    // the bootstrap replicates are all ~0 too (the volatilities are re-estimated inside every replicate)
    assert!(r.boot_se < 1e-9, "{}", r.boot_se);
    // a genuinely better combined book at higher leverage does pass
    let better: Vec<f64> =
        b.iter().enumerate().map(|(t, v)| 3.0 * v + 0.001 * (((t * 7) % 5) as f64 - 2.0) + 0.0008).collect();
    let r2 = marginal_contribution(&b, &better, &cfg(199, 2.0, 1)).unwrap();
    assert!(r2.delta_sharpe > 0.0);
}

#[test]
fn equal_vol_scales_and_difference_series() {
    let (b, c) = fixture(120);
    let (c0, c1) = equal_vol_scales(&b, &c, 0.01).unwrap();
    close(c0 * stats::std_dev(&b).unwrap(), 0.01, 1e-12);
    close(c1 * stats::std_dev(&c).unwrap(), 0.01, 1e-12);
    let d = equal_vol_difference(&b, &c, c0, c1).unwrap();
    assert_eq!(d.len(), 120);
    close(d[7], c1 * c[7] - c0 * b[7], 1e-14);
    // mean of the scaled difference over the target vol IS the Sharpe difference (per period)
    let per_period = stats::mean(&d).unwrap() / 0.01;
    let want = stats::sharpe_per_period(&c).unwrap() - stats::sharpe_per_period(&b).unwrap();
    close(per_period, want, 1e-11);
    assert!(equal_vol_scales(&b, &c, 0.0).is_err());
    assert!(equal_vol_scales(&b, &c, f64::NAN).is_err());
    assert!(equal_vol_scales(&[0.1; 10], &c, 0.01).is_err());
    assert!(equal_vol_scales(&b, &[0.1; 10], 0.01).is_err());
    assert!(equal_vol_difference(&b, &c[..5], 1.0, 1.0).is_err());
    assert!(equal_vol_difference(&b, &c, f64::NAN, 1.0).is_err());
}

#[test]
fn one_sided_versus_two_sided_p_values() {
    let b: Vec<f64> = (0..80).map(|t| (((t * 41 + 3) % 27) as f64 - 13.0) / 900.0).collect();
    let better: Vec<f64> = (0..80).map(|t| b[t] + 0.0006 + (((t * 13 + 1) % 17) as f64 - 8.0) / 4000.0).collect();
    let worse: Vec<f64> = (0..80).map(|t| b[t] - 0.0006 + (((t * 13 + 1) % 17) as f64 - 8.0) / 4000.0).collect();
    let up = marginal_contribution(&b, &better, &cfg(499, 2.0, 4)).unwrap();
    let down = marginal_contribution(&b, &worse, &cfg(499, 2.0, 4)).unwrap();
    // a clear improvement: one-sided p is (about) half the two-sided p and both are small
    assert!(up.p_one_sided < 0.01 && up.p_two_sided < 0.02);
    assert!(up.p_one_sided <= up.p_two_sided);
    assert!((up.p_two_sided - 2.0 * up.p_one_sided).abs() < 0.01 + 1e-12, "{} {}", up.p_one_sided, up.p_two_sided);
    // a clear deterioration: the one-sided test for `improves` must NOT reject (p near 1), the two-sided one does
    assert!(down.delta_sharpe < 0.0);
    assert!(down.p_one_sided > 0.95, "{}", down.p_one_sided);
    assert!(down.p_two_sided < 0.02, "{}", down.p_two_sided);
    assert!(!down.significant());
}

#[test]
fn minimum_detectable_effect_formula_and_analytic_values() {
    // (z_0.95 + z_0.8) x se = (1.6448536269514715 + 0.8416212335729144) x se
    close(mde_from_se(1.0, 0.05, 0.8).unwrap(), 2.486_474_860_524_386, 1e-13);
    close(mde_from_se(0.34, 0.05, 0.8).unwrap(), 0.34 * 2.486_474_860_524_386, 1e-13);
    assert_eq!(mde_from_se(0.0, 0.05, 0.8).unwrap(), 0.0);
    // one-sided 2.5% and 90% power: 1.959963984540054 + 1.2815515655446004
    close(mde_from_se(1.0, 0.025, 0.9).unwrap(), 1.959963984540054 + 1.2815515655446004, 1e-13);
    assert!(mde_from_se(1.0, 0.0, 0.8).is_err());
    assert!(mde_from_se(1.0, 0.5, 0.8).is_err());
    assert!(mde_from_se(1.0, 0.05, 1.0).is_err());
    assert!(mde_from_se(-1.0, 0.05, 0.8).is_err());
    assert!(mde_from_se(f64::NAN, 0.05, 0.8).is_err());

    // Memmel/Jobson-Korkie SE by hand: n=1260, ppy=252, SR1=SR2=0 (per-period), rho=0.6:
    // var = (2 - 1.2)/1260 per period, annual = sqrt(252 x 0.8 / 1260) = 0.4
    close(analytic_se_delta_sharpe(1260, 252.0, 0.0, 0.0, 0.6).unwrap(), 0.4, 1e-13);
    // rho = 0 -> sqrt(252 x 2 / 1260) = sqrt(0.4)
    close(analytic_se_delta_sharpe(1260, 252.0, 0.0, 0.0, 0.0).unwrap(), 0.4_f64.sqrt(), 1e-13);
    // the SR^2 terms add: SR1 = SR2 = 1 annual -> per period 1/sqrt(252); add 0.5(1/252 + 1/252 - 2 rho^2/252)
    let se = analytic_se_delta_sharpe(1260, 252.0, 1.0, 1.0, 0.6).unwrap();
    let want = (252.0_f64 * ((2.0 - 1.2) + 0.5 * (1.0 / 252.0 + 1.0 / 252.0 - 2.0 * 0.36 / 252.0)) / 1260.0).sqrt();
    close(se, want, 1e-13);
    assert!(analytic_se_delta_sharpe(1, 252.0, 0.0, 0.0, 0.0).is_err());
    assert!(analytic_se_delta_sharpe(100, 252.0, 0.0, 0.0, 1.5).is_err());
    assert!(analytic_se_delta_sharpe(100, 0.0, 0.0, 0.0, 0.0).is_err());
    assert!(analytic_se_delta_sharpe(100, 252.0, f64::NAN, 0.0, 0.0).is_err());
}

#[test]
fn analytic_mde_is_a_fixed_point_and_monotone_in_length_and_correlation() {
    let m = mde_analytic_iid(1260, 252.0, 0.5, 0.894, 0.05, 0.8).unwrap();
    let se = analytic_se_delta_sharpe(1260, 252.0, 0.5, 0.5 + m, 0.894).unwrap();
    close(m, mde_from_se(se, 0.05, 0.8).unwrap(), 1e-9);
    // monotone: more data -> smaller MDE
    let mut last = f64::INFINITY;
    for n in [126usize, 252, 504, 882, 1260, 2520, 5040] {
        let m = mde_analytic_iid(n, 252.0, 0.5, 0.7, 0.05, 0.8).unwrap();
        assert!(m < last, "n={n}: {m} !< {last}");
        last = m;
    }
    // monotone: higher correlation between the books -> more precise paired difference -> smaller MDE
    let mut last = f64::INFINITY;
    for rho in [0.0, 0.3, 0.6, 0.9, 0.99] {
        let m = mde_analytic_iid(1260, 252.0, 0.5, rho, 0.05, 0.8).unwrap();
        assert!(m < last, "rho={rho}");
        last = m;
    }
    // roughly 1/sqrt(years): 4x the data halves the MDE (SR^2 terms make it slightly different)
    let a = mde_analytic_iid(504, 252.0, 0.5, 0.7, 0.05, 0.8).unwrap();
    let b = mde_analytic_iid(2016, 252.0, 0.5, 0.7, 0.05, 0.8).unwrap();
    assert!((a / b - 2.0).abs() < 0.05, "{a} {b}");
}

#[test]
fn invariant_to_positive_scaling_of_either_book_and_of_time_units() {
    let (b, c) = fixture(150);
    let k = cfg(199, 3.0, 12);
    let base = marginal_contribution(&b, &c, &k).unwrap();
    for (sb, sc) in [(2.0, 1.0), (1.0, 5.0), (1e-3, 7.0)] {
        let b2: Vec<f64> = b.iter().map(|v| v * sb).collect();
        let c2: Vec<f64> = c.iter().map(|v| v * sc).collect();
        let r = marginal_contribution(&b2, &c2, &k).unwrap();
        close(r.delta_sharpe, base.delta_sharpe, 1e-9);
        close(r.boot_se, base.boot_se, 1e-8);
        assert_eq!(r.p_one_sided, base.p_one_sided);
        assert_eq!(r.p_two_sided, base.p_two_sided);
        close(r.ci_low, base.ci_low, 1e-8);
    }
}

#[test]
fn swapping_the_books_flips_the_sign_of_delta() {
    let (b, c) = fixture(150);
    let k = cfg(99, 1.0, 12);
    let ab = marginal_contribution(&b, &c, &k).unwrap();
    let ba = marginal_contribution(&c, &b, &k).unwrap();
    close(ab.delta_sharpe, -ba.delta_sharpe, 1e-12);
    assert!(ab.delta_sharpe > 0.0);
    assert!(ab.p_one_sided < 0.5 && ba.p_one_sided > 0.5);
}

#[test]
fn point_estimate_is_invariant_to_row_permutation_and_annualisation_scales_it() {
    let (b, c) = fixture(150);
    let mut idx: Vec<usize> = (0..150).collect();
    let mut rng = Rng::seed_from_u64(5);
    for i in (1..idx.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        idx.swap(i, j);
    }
    let bp: Vec<f64> = idx.iter().map(|i| b[*i]).collect();
    let cp: Vec<f64> = idx.iter().map(|i| c[*i]).collect();
    let d1 = delta_sharpe(&b, &c, 252.0, ScaleMode::InSample).unwrap();
    let d2 = delta_sharpe(&bp, &cp, 252.0, ScaleMode::InSample).unwrap();
    close(d1, d2, 1e-12);
    let d365 = delta_sharpe(&b, &c, 365.0, ScaleMode::InSample).unwrap();
    close(d365, d1 * (365.0f64 / 252.0).sqrt(), 1e-12);
}

#[test]
fn deterministic_bit_identical_repeats() {
    let (b, c) = fixture(300);
    let mut k = MarginalConfig::new(252.0);
    k.n_boot = 199;
    let r1 = marginal_contribution(&b, &c, &k).unwrap();
    let r2 = marginal_contribution(&b, &c, &k).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r1.boot_se.to_bits(), r2.boot_se.to_bits());
    let mut k2 = k;
    k2.seed ^= 1;
    let r3 = marginal_contribution(&b, &c, &k2).unwrap();
    assert_ne!(r1.boot_se.to_bits(), r3.boot_se.to_bits(), "a different seed must change the replicates");
    // the point estimate does not depend on the seed
    assert_eq!(r1.delta_sharpe.to_bits(), r3.delta_sharpe.to_bits());
}

#[test]
fn bootstrap_standard_error_tracks_the_analytic_one_for_iid_gaussian_data() {
    let mut rng = Rng::seed_from_u64(99);
    let n = 1260;
    let z1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let z2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let rho = 0.6;
    let base: Vec<f64> = z1.iter().map(|z| 0.01 * (0.03 + z)).collect();
    let comb: Vec<f64> = z1
        .iter()
        .zip(&z2)
        .map(|(a, b)| 0.01 * (0.05 + 0.5 * a + 0.5 * (rho * a + (1.0f64 - rho * rho).sqrt() * b)))
        .collect();
    let mut k = MarginalConfig::new(252.0);
    k.n_boot = 999;
    k.block = BlockLength::Fixed(1.0);
    let r = marginal_contribution(&base, &comb, &k).unwrap();
    let analytic = analytic_se_delta_sharpe(n, 252.0, r.sharpe_base, r.sharpe_combined, r.corr_books).unwrap();
    assert!((r.boot_se / analytic - 1.0).abs() < 0.12, "bootstrap {} analytic {}", r.boot_se, analytic);
}

#[test]
fn bootstrap_se_shrinks_like_one_over_root_n() {
    let mut rng = Rng::seed_from_u64(3);
    let mut se_for = |n: usize| {
        let mut acc = 0.0;
        for _ in 0..6 {
            let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let e: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let base: Vec<f64> = z.iter().map(|v| 0.01 * (0.03 + v)).collect();
            let comb: Vec<f64> = z.iter().zip(&e).map(|(a, b)| 0.01 * (0.03 + 0.7 * a + 0.7 * b)).collect();
            let mut k = MarginalConfig::new(252.0);
            k.n_boot = 199;
            k.block = BlockLength::Fixed(1.0);
            acc += marginal_contribution(&base, &comb, &k).unwrap().boot_se / 6.0;
        }
        acc
    };
    let s1 = se_for(250);
    let s4 = se_for(1000);
    assert!(s4 < s1, "{s1} {s4}");
    assert!((s1 / s4 - 2.0).abs() < 0.3, "ratio {}", s1 / s4);
}

#[test]
fn null_size_is_close_to_nominal_for_a_diluted_candidate_with_no_edge() {
    // Two Monte Carlo experiments of the shipped test with its default (automatic) block length: candidate adds
    // exactly nothing (equal Sharpe at equal volatility). Size at alpha = 0.05 must be within Monte Carlo error.
    let mut rng = Rng::seed_from_u64(0xD00D);
    let n = 504;
    let reps = 200;
    let (w, rho, sr_p): (f64, f64, f64) = (0.5, 0.3, 0.5 / 252f64.sqrt());
    let var_b = (1.0 - w) * (1.0 - w) + w * w + 2.0 * w * (1.0 - w) * rho;
    let sr_x = sr_p * (var_b.sqrt() - 1.0 + w) / w; // zero incremental Sharpe
    let mut rej = 0;
    for rep in 0..reps {
        let z1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let z2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let base: Vec<f64> = z1.iter().map(|z| 0.01 * (sr_p + z)).collect();
        let comb: Vec<f64> = (0..n)
            .map(|t| {
                let x = 0.01 * (sr_x + rho * z1[t] + (1.0f64 - rho * rho).sqrt() * z2[t]);
                (1.0 - w) * base[t] + w * x
            })
            .collect();
        let mut k = MarginalConfig::new(252.0);
        k.n_boot = 99;
        k.seed = rep as u64;
        if marginal_contribution(&base, &comb, &k).unwrap().significant() {
            rej += 1;
        }
    }
    let rate = rej as f64 / reps as f64;
    // MC standard error at 5% with 200 reps is 1.5 pp; allow four
    assert!((rate - 0.05).abs() < 0.06, "size {rate}");
}

#[test]
fn a_planted_incremental_sharpe_is_detected_and_its_confidence_interval_covers_it() {
    let mut rng = Rng::seed_from_u64(1717);
    let n = 2520; // 10 years: a Sharpe difference of about 0.9 has SE about 0.17 here
    let z1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let z2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    // base: SR 0.5; candidate X independent with SR 2.0; combined = 0.5 base + 0.5 X
    let base: Vec<f64> = z1.iter().map(|z| 0.01 * (0.5 / 252f64.sqrt() + z)).collect();
    let comb: Vec<f64> = (0..n).map(|t| 0.5 * base[t] + 0.5 * 0.01 * (2.0 / 252f64.sqrt() + z2[t])).collect();
    let true_delta = 1.25 / 0.5f64.sqrt() - 0.5; // SR(0.5 P + 0.5 X) - SR(P)
    let mut k = MarginalConfig::new(252.0);
    k.n_boot = 999;
    let r = marginal_contribution(&base, &comb, &k).unwrap();
    assert!(r.significant(), "p = {}", r.p_one_sided);
    assert!(r.ci_low < true_delta && true_delta < r.ci_high, "{} not in [{}, {}]", true_delta, r.ci_low, r.ci_high);
    assert!(r.mde > 0.0 && r.mde < 1.5, "mde {}", r.mde);
}

#[test]
fn the_report_carries_a_spanning_diagnostic_of_the_candidate_on_the_base_book() {
    let (b, c) = fixture(200);
    let cand: Vec<f64> = (0..200).map(|t| 0.4 * b[t] + (((t * 29 + 5) % 19) as f64 - 9.0) / 1500.0 + 0.0004).collect();
    let rep = marginal_report(&b, &c, &cand, &cfg(99, 2.0, 3), HacLag::Auto).unwrap();
    assert_eq!(rep.spanning.n, 200);
    assert!(rep.spanning.beta[0] > 0.3 && rep.spanning.beta[0] < 0.5, "{}", rep.spanning.beta[0]);
    assert!(rep.spanning.alpha > 0.0);
    assert!(rep.primary.delta_sharpe.is_finite());
}

#[test]
fn typed_errors_for_degenerate_and_malformed_inputs() {
    let (b, c) = fixture(60);
    let k = cfg(99, 2.0, 1);
    assert!(matches!(marginal_contribution(&b, &c[..59], &k), Err(EvalError::LengthMismatch { .. })));
    assert!(matches!(
        marginal_contribution(&b[..29], &c[..29], &k),
        Err(EvalError::TooShort { need: 30, got: 29, .. })
    ));
    assert!(matches!(marginal_contribution(&[], &[], &k), Err(EvalError::TooShort { .. })));
    let mut nan = b.clone();
    nan[10] = f64::NAN;
    assert!(matches!(marginal_contribution(&nan, &c, &k), Err(EvalError::NonFinite { index: 10, .. })));
    let mut inf = c.clone();
    inf[3] = f64::NEG_INFINITY;
    assert!(matches!(marginal_contribution(&b, &inf, &k), Err(EvalError::NonFinite { index: 3, .. })));
    assert!(matches!(marginal_contribution(&[0.0; 60], &c, &k), Err(EvalError::ZeroVariance { .. })));
    assert!(matches!(marginal_contribution(&b, &[0.01; 60], &k), Err(EvalError::ZeroVariance { .. })));
    let mut bad = k;
    bad.n_boot = 10;
    assert!(matches!(marginal_contribution(&b, &c, &bad), Err(EvalError::InvalidParameter { .. })));
    bad = k;
    bad.alpha = 0.6;
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.power = 1.0;
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.ci_level = 0.0;
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.periods_per_year = f64::NAN;
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.block = BlockLength::Fixed(0.5);
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.scale = ScaleMode::ExAnte { base_vol: 0.0, combined_vol: 1.0 };
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    bad = k;
    bad.scale = ScaleMode::ExAnte { base_vol: 1.0, combined_vol: f64::NAN };
    assert!(marginal_contribution(&b, &c, &bad).is_err());
    assert!(delta_sharpe(&b, &c, 0.0, ScaleMode::InSample).is_err());
    assert!(matches!(delta_sharpe(&b, &c[..10], 252.0, ScaleMode::InSample), Err(EvalError::LengthMismatch { .. })));
}

#[test]
fn automatic_block_length_is_used_and_reported() {
    let (b, c) = fixture(200);
    let mut k = MarginalConfig::new(252.0);
    k.n_boot = 99;
    let r = marginal_contribution(&b, &c, &k).unwrap();
    assert!(r.block_length >= 1.0 && r.block_length <= 200.0);
    let mut k1 = k;
    k1.block = BlockLength::Fixed(r.block_length);
    let r1 = marginal_contribution(&b, &c, &k1).unwrap();
    assert_eq!(r1.boot_se.to_bits(), r.boot_se.to_bits(), "Auto equals Fixed(auto value)");
}

#[test]
fn identical_books_have_delta_zero_and_p_value_one_because_ties_count() {
    // combined == base: every replicate statistic is exactly 0 = the estimate, so every replicate satisfies
    // `Delta* - Delta_hat >= Delta_hat` and p = (1 + B) / (B + 1) = 1 exactly. Not counting ties would give 1/(B+1).
    let (b, _) = fixture(100);
    let r = marginal_contribution(&b, &b, &cfg(99, 2.0, 8)).unwrap();
    assert_eq!(r.delta_sharpe, 0.0);
    assert_eq!(r.p_one_sided, 1.0);
    assert_eq!(r.p_two_sided, 1.0);
    assert_eq!(r.boot_se, 0.0);
    assert!(!r.significant());
}

#[test]
fn the_automatic_block_length_is_the_politis_white_maximum_over_both_books_and_their_difference() {
    use portfolio_eval::bootstrap::{auto_block_length, politis_white_block_length};
    // base: serially independent; combined: strongly autocorrelated. The longest block must win.
    let mut rng = Rng::seed_from_u64(123);
    let n = 400;
    let base: Vec<f64> = (0..n).map(|_| 0.01 * rng.normal()).collect();
    let mut x = 0.0;
    let comb: Vec<f64> = (0..n + 30)
        .map(|_| {
            x = 0.8 * x + rng.normal();
            0.01 * x
        })
        .skip(30)
        .collect();
    let sd0 = stats::std_dev(&base).unwrap();
    let sd1 = stats::std_dev(&comb).unwrap();
    let diff: Vec<f64> = base.iter().zip(&comb).map(|(b, c)| c / sd1 - b / sd0).collect();
    let want = auto_block_length(&[&base, &comb, &diff]).unwrap();
    let (pb, pc) = (politis_white_block_length(&base).unwrap(), politis_white_block_length(&comb).unwrap());
    assert!(pc > 4.0 * pb.max(1.0), "the autocorrelated book should need a much longer block: {pb} vs {pc}");
    let mut k = MarginalConfig::new(252.0);
    k.n_boot = 99;
    let r = marginal_contribution(&base, &comb, &k).unwrap();
    assert_eq!(r.block_length, want);
    assert!(r.block_length >= pc);
}
