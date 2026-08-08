//! Johansen procedure for N-way (basket) cointegration -- generalizes
//! `cointegration.rs`'s pairwise Engle-Granger test to 3+ series. Engle-
//! Granger doesn't generalize past 2 series (there's no single "regress Y on
//! X" once there are 3+ candidate legs); Johansen instead estimates how many
//! independent stationary linear combinations ("cointegrating relations")
//! exist among N series at once, via the eigenvalues of a generalized
//! eigenvalue problem built from two auxiliary VAR regressions.
//!
//! Johansen, S. (1991), "Estimation and Hypothesis Testing of Cointegration
//! Vectors in Gaussian Vector Autoregressive Models," *Econometrica*.
//!
//! Critical values below are approximate asymptotic figures for the
//! "unrestricted intercept, no deterministic trend" specification (the
//! conventional choice for testing cointegration among price/log-price
//! levels), compiled from commonly-reproduced tables (Johansen & Juselius
//! 1990; the specific numbers used here follow Enders, *Applied Econometric
//! Time Series*). Treat them the same way as this crate's ADF/Engle-Granger
//! critical values: a plain-language significance flag, not a forensically
//! exact reproduction of the canonical response-surface tables -- if a
//! basket cointegration verdict is gating real capital, cross-check against
//! a maintained statistical package (e.g. R's `urca::ca.jo`) rather than
//! trusting these constants to the last decimal.

use serde::{Deserialize, Serialize};

use crate::linalg;

/// Result of the Johansen procedure on an N-series basket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohansenResult {
    /// Eigenvalues of the generalized eigenproblem, descending. Each is a
    /// squared canonical correlation between the levels and differences,
    /// in `[0, 1)`.
    pub eigenvalues: Vec<f64>,
    /// Trace statistic for testing "at most r cointegrating relations",
    /// indexed by r (`trace_statistics[r]` tests H0: rank <= r).
    pub trace_statistics: Vec<f64>,
    /// Approximate 95% critical value for each trace test, aligned with
    /// `trace_statistics`. `None` when the basket is larger than this
    /// module's hardcoded table covers (see module doc).
    pub trace_critical_values_95: Vec<Option<f64>>,
    /// Max-eigenvalue statistic for testing "exactly r" vs "r+1".
    pub max_eigen_statistics: Vec<f64>,
    /// Approximate 95% critical value for each max-eigenvalue test.
    pub max_eigen_critical_values_95: Vec<Option<f64>>,
    /// Estimated cointegration rank from the sequential trace-test
    /// procedure: the largest r for which "rank <= r" was NOT rejected,
    /// i.e. the number of independent stationary combinations the data
    /// supports. `0` means no cointegration was detected.
    pub cointegration_rank: usize,
    /// Cointegrating vectors (eigenvectors), one per series, ordered by
    /// eigenvalue descending. `cointegrating_vectors[0]` is the strongest
    /// (most stationary) combination -- see `normalize_cointegrating_vector`
    /// to turn it into per-leg hedge ratios relative to the first symbol.
    pub cointegrating_vectors: Vec<Vec<f64>>,
    pub n_obs: usize,
    pub lag_order: usize,
}

/// Approximate 95% trace-statistic critical value for `n_minus_r` (number
/// of series minus the rank under the null). `None` beyond this table's
/// range (baskets larger than ~6 legs) -- see module doc's caveat.
fn trace_critical_value_95(n_minus_r: usize) -> Option<f64> {
    match n_minus_r {
        1 => Some(9.24),
        2 => Some(19.96),
        3 => Some(34.91),
        4 => Some(53.12),
        5 => Some(76.07),
        6 => Some(102.14),
        _ => None,
    }
}

/// Approximate 95% max-eigenvalue-statistic critical value for `n_minus_r`.
fn max_eigen_critical_value_95(n_minus_r: usize) -> Option<f64> {
    match n_minus_r {
        1 => Some(9.24),
        2 => Some(15.67),
        3 => Some(22.00),
        4 => Some(28.14),
        5 => Some(34.40),
        6 => Some(40.30),
        _ => None,
    }
}

/// OLS residuals of `y` regressed on `z_columns` (each column the same
/// length as `y`). Local to this module -- deliberately not shared with
/// `cointegration.rs`'s own `ols_multi` (that one also returns standard
/// errors for the ADF t-stat; this one only ever needs the residuals).
fn ols_residuals(y: &[f64], z_columns: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = y.len();
    let k = z_columns.len();
    if k == 0 || n <= k {
        return None;
    }
    for c in z_columns {
        if c.len() != n {
            return None;
        }
    }

    let mut ztz = vec![vec![0.0; k]; k];
    let mut zty = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            ztz[i][j] = (0..n).map(|t| z_columns[i][t] * z_columns[j][t]).sum();
        }
        zty[i] = (0..n).map(|t| z_columns[i][t] * y[t]).sum();
    }
    let ztz_inv = linalg::invert_matrix(&ztz)?;
    let coeffs: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ztz_inv[i][j] * zty[j]).sum()).collect();
    let residuals: Vec<f64> = (0..n)
        .map(|t| y[t] - (0..k).map(|i| coeffs[i] * z_columns[i][t]).sum::<f64>())
        .collect();
    Some(residuals)
}

/// Run the Johansen procedure on `series` (N series, each the same length,
/// in levels -- e.g. log prices). `lag_order` is the number of lagged
/// difference terms in the underlying VAR (mirrors the augmentation lag in
/// `cointegration::adf_test`) -- `0` is a reasonable default for daily-bar
/// baskets; raise it if the legs have meaningfully autocorrelated short-run
/// dynamics beyond one lag.
pub fn johansen_test(series: &[Vec<f64>], lag_order: usize) -> Option<JohansenResult> {
    let n = series.len();
    if n < 2 {
        return None;
    }
    let t_len = series[0].len();
    if series.iter().any(|s| s.len() != t_len) {
        return None;
    }

    let dy: Vec<Vec<f64>> = series.iter().map(|s| (0..t_len - 1).map(|t| s[t + 1] - s[t]).collect()).collect();
    let m_total = t_len.saturating_sub(1);
    if m_total <= lag_order {
        return None;
    }
    let start = lag_order;
    let m = m_total - start;
    // Need enough observations relative to the regressor count (intercept +
    // lag_order*n) and the later n x n matrix inversions.
    if m <= (1 + lag_order * n) || m <= n {
        return None;
    }

    let mut z_columns: Vec<Vec<f64>> = Vec::with_capacity(1 + lag_order * n);
    z_columns.push(vec![1.0; m]);
    for lag in 1..=lag_order {
        for series_dy in dy.iter() {
            let col: Vec<f64> = (start..m_total).map(|t| series_dy[t - lag]).collect();
            z_columns.push(col);
        }
    }

    // R0: residuals of current-period differences; R1: residuals of the
    // pre-change level (same index convention as cointegration::adf_test --
    // level[t] pairs with dy[t] = series[t+1] - series[t]).
    let mut r0 = vec![vec![0.0; n]; m];
    let mut r1 = vec![vec![0.0; n]; m];
    for i in 0..n {
        let y0: Vec<f64> = (start..m_total).map(|t| dy[i][t]).collect();
        let y1: Vec<f64> = (start..m_total).map(|t| series[i][t]).collect();
        let res0 = ols_residuals(&y0, &z_columns)?;
        let res1 = ols_residuals(&y1, &z_columns)?;
        for k in 0..m {
            r0[k][i] = res0[k];
            r1[k][i] = res1[k];
        }
    }

    let mf = m as f64;
    let mut s00 = vec![vec![0.0; n]; n];
    let mut s01 = vec![vec![0.0; n]; n];
    let mut s11 = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            s00[i][j] = (0..m).map(|k| r0[k][i] * r0[k][j]).sum::<f64>() / mf;
            s01[i][j] = (0..m).map(|k| r0[k][i] * r1[k][j]).sum::<f64>() / mf;
            s11[i][j] = (0..m).map(|k| r1[k][i] * r1[k][j]).sum::<f64>() / mf;
        }
    }

    let s00_inv = linalg::invert_matrix(&s00)?;
    let s01_t = linalg::transpose(&s01);
    let target = linalg::matmul(&linalg::matmul(&s01_t, &s00_inv), &s01);

    // Symmetric-definite generalized eigenproblem `target * v = lambda * s11
    // * v`, solved via Cholesky (s11 = L L^T) reducing to the standard
    // symmetric eigenproblem `(L^-1 target L^-T) w = lambda w`, with
    // `v = L^-T w`.
    let l = linalg::cholesky(&s11)?;
    let l_inv = linalg::invert_lower_triangular(&l)?;
    let l_inv_t = linalg::transpose(&l_inv);
    let sym_matrix = linalg::matmul(&linalg::matmul(&l_inv, &target), &l_inv_t);

    let (eigenvalues_raw, eigenvectors_w) = linalg::jacobi_eigen(&sym_matrix, 200);
    let eigenvalues: Vec<f64> = eigenvalues_raw.iter().map(|&lam| lam.clamp(0.0, 0.999_999)).collect();
    let cointegrating_vectors: Vec<Vec<f64>> = eigenvectors_w
        .iter()
        .map(|w| (0..n).map(|i| (0..n).map(|j| l_inv_t[i][j] * w[j]).sum()).collect())
        .collect();

    let mut trace_statistics = Vec::with_capacity(n);
    let mut max_eigen_statistics = Vec::with_capacity(n);
    for r in 0..n {
        let trace: f64 = -mf * eigenvalues[r..].iter().map(|&lam| (1.0 - lam).ln()).sum::<f64>();
        trace_statistics.push(trace);
        max_eigen_statistics.push(-mf * (1.0 - eigenvalues[r]).ln());
    }

    let trace_critical_values_95: Vec<Option<f64>> = (0..n).map(|r| trace_critical_value_95(n - r)).collect();
    let max_eigen_critical_values_95: Vec<Option<f64>> = (0..n).map(|r| max_eigen_critical_value_95(n - r)).collect();

    let mut cointegration_rank = 0;
    for r in 0..n {
        match trace_critical_values_95[r] {
            Some(cv) if trace_statistics[r] > cv => cointegration_rank = r + 1,
            _ => break,
        }
    }
    cointegration_rank = cointegration_rank.min(n.saturating_sub(1));

    Some(JohansenResult {
        eigenvalues,
        trace_statistics,
        trace_critical_values_95,
        max_eigen_statistics,
        max_eigen_critical_values_95,
        cointegration_rank,
        cointegrating_vectors,
        n_obs: m,
        lag_order,
    })
}

/// Normalize a cointegrating vector so its first component is `1.0` --
/// turns raw eigenvector weights into "1 unit of leg 0 per `weights[i]`
/// units of leg i" hedge ratios, the natural N-way generalization of
/// pairwise `cointegration::OlsResult::beta`. Returns `None` when the first
/// component is too close to zero to normalize against.
pub fn normalize_cointegrating_vector(v: &[f64]) -> Option<Vec<f64>> {
    let pivot = v.first().copied()?;
    if pivot.abs() < 1e-9 {
        return None;
    }
    Some(v.iter().map(|w| w / pivot).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut v = Vec::with_capacity(n);
        let mut cum = 0.0;
        for _ in 0..n {
            cum += dist.sample(&mut rng);
            v.push(cum);
        }
        v
    }

    fn ar1_series(n: usize, phi: f64, noise_sd: f64, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, noise_sd).unwrap();
        let mut v = Vec::with_capacity(n);
        let mut prev = 0.0;
        for _ in 0..n {
            let shock: f64 = dist.sample(&mut rng);
            prev = phi * prev + shock;
            v.push(prev);
        }
        v
    }

    #[test]
    fn rejects_fewer_than_two_series() {
        let x = random_walk(200, 1);
        assert!(johansen_test(&[x], 0).is_none());
    }

    #[test]
    fn rejects_mismatched_series_lengths() {
        let a = random_walk(200, 1);
        let b = random_walk(150, 2);
        assert!(johansen_test(&[a, b], 0).is_none());
    }

    #[test]
    fn rejects_too_short_series_for_requested_lag() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        assert!(johansen_test(&[a, b], 5).is_none());
    }

    #[test]
    fn three_independent_random_walks_show_no_cointegration() {
        let a = random_walk(2000, 10);
        let b = random_walk(2000, 20);
        let c = random_walk(2000, 30);
        let result = johansen_test(&[a, b, c], 1).expect("should fit");
        assert_eq!(
            result.cointegration_rank, 0,
            "three independent random walks should show rank 0, got trace stats {:?} vs cv {:?}",
            result.trace_statistics, result.trace_critical_values_95
        );
    }

    #[test]
    fn one_common_trend_basket_shows_rank_at_least_one() {
        // A common stochastic trend `x`, plus two other series each a linear
        // combination of x with stationary AR(1) noise -- a textbook rank-1
        // (well, rank could be higher, but at least 1) 3-series basket.
        let trend = random_walk(2000, 40);
        let noise_b = ar1_series(2000, 0.3, 1.0, 41);
        let noise_c = ar1_series(2000, 0.3, 1.0, 42);
        let b: Vec<f64> = trend.iter().zip(noise_b.iter()).map(|(t, n)| 1.5 * t + n).collect();
        let c: Vec<f64> = trend.iter().zip(noise_c.iter()).map(|(t, n)| 0.8 * t + n).collect();

        let result = johansen_test(&[trend, b, c], 1).expect("should fit");
        assert!(
            result.cointegration_rank >= 1,
            "expected at least 1 cointegrating relation, got rank {} (trace={:?}, cv={:?})",
            result.cointegration_rank, result.trace_statistics, result.trace_critical_values_95
        );
    }

    #[test]
    fn eigenvalues_are_sorted_descending_and_in_valid_range() {
        let a = random_walk(1000, 50);
        let b = random_walk(1000, 51);
        let result = johansen_test(&[a, b], 1).expect("should fit");
        for w in result.eigenvalues.windows(2) {
            assert!(w[0] >= w[1], "eigenvalues should be descending");
        }
        for &lam in &result.eigenvalues {
            assert!((0.0..1.0).contains(&lam), "eigenvalue {} out of [0,1)", lam);
        }
    }

    #[test]
    fn trace_statistics_are_non_increasing_in_r() {
        // trace(r) sums fewer terms as r increases, so it must be
        // monotonically non-increasing.
        let a = random_walk(1000, 60);
        let b = random_walk(1000, 61);
        let c = random_walk(1000, 62);
        let result = johansen_test(&[a, b, c], 1).expect("should fit");
        for w in result.trace_statistics.windows(2) {
            assert!(w[0] >= w[1] - 1e-6, "trace statistics should be non-increasing in r");
        }
    }

    #[test]
    fn normalize_cointegrating_vector_scales_first_component_to_one() {
        let v = vec![2.0, 4.0, -6.0];
        let normalized = normalize_cointegrating_vector(&v).unwrap();
        assert!((normalized[0] - 1.0).abs() < 1e-9);
        assert!((normalized[1] - 2.0).abs() < 1e-9);
        assert!((normalized[2] - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn normalize_cointegrating_vector_rejects_zero_pivot() {
        let v = vec![0.0, 4.0, -6.0];
        assert!(normalize_cointegrating_vector(&v).is_none());
    }
}
