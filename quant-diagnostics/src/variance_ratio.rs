//! Lo-MacKinlay variance ratio test with a heteroskedasticity-robust
//! z-statistic (Lo, A. W., & MacKinlay, A. C. (1988), "Stock Market Prices
//! Do Not Follow Random Walks").
//!
//! A deviation of VR(q) from 1.0 only means something once it's paired with
//! its significance test — a bare magnitude (e.g. "VR = 0.85") says nothing
//! about whether that's noise or a real departure from a random walk.

use serde::{Deserialize, Serialize};

/// Result of the variance ratio test at one aggregation horizon `q`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VrResult {
    /// Aggregation horizon (number of base periods per "long" return).
    pub horizon: usize,
    /// VR(q) = Var(q-period return) / (q * Var(1-period return)).
    /// ~1.0 under a random walk; >1.0 momentum/trending; <1.0 mean-reverting.
    pub vr: f64,
    /// Heteroskedasticity-robust z-statistic for VR(q) - 1 == 0.
    pub z_stat: f64,
    /// Two-sided p-value from the standard normal approximation.
    pub p_value: f64,
    /// True when |z_stat| implies rejection of the random-walk null at 5%.
    pub significant: bool,
}

/// Standard normal CDF via the Abramowitz-Stegun erf approximation
/// (max absolute error ~1.5e-7 — plenty for a plain-language significance
/// flag, not used for anything requiring tighter precision).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = pdf * poly;
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

fn two_sided_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Run the Lo-MacKinlay variance ratio test at each requested horizon.
///
/// `returns` should be simple or log per-period returns (not prices) in
/// chronological order. Horizons with insufficient data (`q >= returns.len()`,
/// or fewer than `2*q` observations for a stable estimate) are skipped.
pub fn variance_ratio(returns: &[f64], horizons: &[usize]) -> Vec<VrResult> {
    let t = returns.len();
    if t < 4 {
        return Vec::new();
    }

    let mu = returns.iter().sum::<f64>() / t as f64;

    // Unbiased 1-period sample variance.
    let sigma_a_sq = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / (t as f64 - 1.0);
    if sigma_a_sq <= 0.0 {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(horizons.len());

    for &q in horizons {
        if q < 2 || q >= t || t < 2 * q {
            continue;
        }
        let tf = t as f64;
        let qf = q as f64;

        // Overlapping q-period return variance estimator.
        let m = qf * (tf - qf + 1.0) * (1.0 - qf / tf);
        if m <= 0.0 {
            continue;
        }
        let mut sigma_c_sq_sum = 0.0;
        for start in q..=t {
            let window_sum: f64 = returns[(start - q)..start].iter().sum();
            sigma_c_sq_sum += (window_sum - qf * mu).powi(2);
        }
        let sigma_c_sq = sigma_c_sq_sum / m;
        let vr = sigma_c_sq / sigma_a_sq;

        // Heteroskedasticity-robust asymptotic variance theta*(q).
        let denom: f64 = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>().powi(2);
        if denom <= 0.0 {
            continue;
        }
        let mut theta_star = 0.0;
        for j in 1..q {
            let mut delta_j_sum = 0.0;
            for tt in (j + 1)..=t {
                delta_j_sum += (returns[tt - 1] - mu).powi(2) * (returns[tt - 1 - j] - mu).powi(2);
            }
            let delta_j = tf * delta_j_sum / denom;
            let weight = 2.0 * (qf - j as f64) / qf;
            theta_star += weight * weight * delta_j;
        }

        if theta_star <= 0.0 {
            continue;
        }
        // theta_star is the asymptotic variance of sqrt(T)*(VR(q)-1), so the
        // z-statistic needs the sqrt(T) factor reinstated here.
        let z_stat = tf.sqrt() * (vr - 1.0) / theta_star.sqrt();
        let p_value = two_sided_p_value(z_stat);

        results.push(VrResult {
            horizon: q,
            vr,
            z_stat,
            p_value,
            significant: p_value < 0.05,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn random_walk_returns(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, 1.0).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn empty_returns_yield_no_results() {
        assert!(variance_ratio(&[], &[2, 4]).is_empty());
    }

    #[test]
    fn too_short_series_yields_no_results() {
        let returns = vec![0.1, -0.2, 0.05];
        assert!(variance_ratio(&returns, &[10]).is_empty());
    }

    #[test]
    fn horizon_of_one_or_covering_full_series_is_skipped() {
        let returns = random_walk_returns(50, 1);
        let results = variance_ratio(&returns, &[1, 50]);
        assert!(results.is_empty());
    }

    #[test]
    fn random_walk_vr_is_close_to_one_across_horizons() {
        let returns = random_walk_returns(5000, 42);
        let results = variance_ratio(&returns, &[2, 4, 8, 16]);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(
                (r.vr - 1.0).abs() < 0.15,
                "horizon {} VR {} too far from 1.0 for a random walk",
                r.horizon,
                r.vr
            );
        }
    }

    #[test]
    fn trending_series_has_vr_above_one_and_significant() {
        // Momentum: each period's return correlates positively with the last.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut returns = Vec::with_capacity(5000);
        let mut prev = 0.0;
        for _ in 0..5000 {
            let shock: f64 = dist.sample(&mut rng);
            let r = 0.3 * prev + shock;
            returns.push(r);
            prev = r;
        }
        let results = variance_ratio(&returns, &[2, 4, 8]);
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.vr > 1.0, "horizon {} expected VR > 1.0, got {}", r.horizon, r.vr);
            assert!(r.significant, "horizon {} expected significance", r.horizon);
        }
    }

    #[test]
    fn mean_reverting_series_has_vr_below_one_and_significant() {
        // Mean reversion: each period's return anti-correlates with the last.
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut returns = Vec::with_capacity(5000);
        let mut prev = 0.0;
        for _ in 0..5000 {
            let shock: f64 = dist.sample(&mut rng);
            let r = -0.3 * prev + shock;
            returns.push(r);
            prev = r;
        }
        let results = variance_ratio(&returns, &[2, 4, 8]);
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.vr < 1.0, "horizon {} expected VR < 1.0, got {}", r.horizon, r.vr);
            assert!(r.significant, "horizon {} expected significance", r.horizon);
        }
    }

    #[test]
    fn zero_variance_series_yields_no_results() {
        let returns = vec![0.0; 100];
        assert!(variance_ratio(&returns, &[2, 4]).is_empty());
    }

    #[test]
    fn p_value_is_within_unit_interval() {
        let returns = random_walk_returns(500, 3);
        for r in variance_ratio(&returns, &[2, 4, 8]) {
            assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
        }
    }
}
