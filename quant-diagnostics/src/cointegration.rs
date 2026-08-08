//! Engle-Granger two-step cointegration test for pairs trading / statistical
//! arbitrage: OLS hedge ratio, Augmented Dickey-Fuller (ADF) unit-root test,
//! rolling spread z-score, and Ornstein-Uhlenbeck mean-reversion half-life.
//!
//! Engle, R. F., & Granger, C. W. J. (1987), "Co-integration and Error
//! Correction: Representation, Estimation, and Testing." Critical values for
//! both the plain ADF test and the Engle-Granger residual test are
//! hand-rolled approximations from MacKinnon's response-surface tables
//! (MacKinnon, J. G. (1996), "Numerical Distribution Functions for Unit Root
//! and Cointegration Tests"), linearly interpolated by sample size — plenty
//! for a plain-language significance flag, not an exact reproduction of the
//! response-surface polynomial. A bare hedge ratio or ADF statistic says
//! nothing on its own; both only mean something paired with the
//! stationarity/cointegration significance test.

use serde::{Deserialize, Serialize};

/// Result of an OLS regression `y = alpha + beta * x + residual`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsResult {
    /// Hedge ratio: units of `x` per unit of `y` in the cointegrating relationship.
    pub beta: f64,
    pub alpha: f64,
    pub r_squared: f64,
    /// Residual series `y - (alpha + beta * x)` — the spread to test for stationarity.
    #[serde(skip)]
    pub residuals: Vec<f64>,
}

/// Approximate 1%/5%/10% left-tail critical values for a Dickey-Fuller-type statistic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CriticalValues {
    pub pct_1: f64,
    pub pct_5: f64,
    pub pct_10: f64,
}

/// Result of an Augmented Dickey-Fuller unit-root test on one series (either
/// a raw price/return series, or the residuals of an estimated regression —
/// see `engle_granger_test`, which uses different critical values for the
/// latter case since ADF's own critical values are invalid for testing
/// residuals of an estimated relationship).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AdfResult {
    /// t-statistic on the lagged-level coefficient. More negative = stronger
    /// evidence against a unit root (i.e. the series is stationary).
    pub statistic: f64,
    /// Approximate one-sided p-value (probability of a statistic this low or
    /// lower under the unit-root null).
    pub p_value: f64,
    pub critical_values: CriticalValues,
    /// Number of lagged-difference terms included, chosen by AIC over `0..=max_lag`.
    pub lags_used: usize,
    pub n_obs: usize,
    /// True when `statistic < critical_values.pct_5` (rejects the unit-root
    /// null at the 5% level).
    pub is_stationary: bool,
}

/// Result of the full Engle-Granger two-step cointegration test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationResult {
    pub hedge_ratio: OlsResult,
    pub adf_on_residuals: AdfResult,
    /// True when the OLS residuals reject the unit-root null at 5%, using
    /// Engle-Granger (not plain ADF) critical values.
    pub is_cointegrated: bool,
}

/// Simple OLS: `y = alpha + beta * x + residual`. Returns `None` for
/// mismatched/too-short input or a degenerate (zero-variance) `x`.
pub fn ols_simple(y: &[f64], x: &[f64]) -> Option<OlsResult> {
    let n = y.len();
    if n != x.len() || n < 3 {
        return None;
    }
    let nf = n as f64;
    let mean_x = x.iter().sum::<f64>() / nf;
    let mean_y = y.iter().sum::<f64>() / nf;

    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..n {
        sxy += (x[i] - mean_x) * (y[i] - mean_y);
        sxx += (x[i] - mean_x).powi(2);
    }
    if sxx <= 0.0 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;

    let residuals: Vec<f64> = (0..n).map(|i| y[i] - (alpha + beta * x[i])).collect();
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    Some(OlsResult {
        beta,
        alpha,
        r_squared,
        residuals,
    })
}

/// Minimal multiple-OLS via normal equations (X'X)^-1 X'y, for the small
/// number of regressors (intercept + level + up to ~12 lag terms) the ADF
/// regression needs. Not optimized/numerically hardened for ill-conditioned
/// or large designs — fine for this crate's scale, matching its existing
/// "hand-rolled approximation, no external dependency" convention.
struct MultiOlsResult {
    coeffs: Vec<f64>,
    se: Vec<f64>,
    residuals: Vec<f64>,
    n: usize,
    k: usize,
}

fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > max_val {
                max_val = a[r][col].abs();
                pivot_row = r;
            }
        }
        if max_val < 1e-10 {
            return None; // singular / near-singular design
        }
        a.swap(col, pivot_row);
        inv.swap(col, pivot_row);

        let pivot = a[col][col];
        for j in 0..n {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[r][j] -= factor * a[col][j];
                inv[r][j] -= factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

fn ols_multi(y: &[f64], columns: &[Vec<f64>]) -> Option<MultiOlsResult> {
    let n = y.len();
    let k = columns.len();
    if k == 0 || n <= k {
        return None;
    }
    for c in columns {
        if c.len() != n {
            return None;
        }
    }

    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            xtx[i][j] = (0..n).map(|t| columns[i][t] * columns[j][t]).sum();
        }
        xty[i] = (0..n).map(|t| columns[i][t] * y[t]).sum();
    }

    let xtx_inv = invert_matrix(&xtx)?;
    let coeffs: Vec<f64> = (0..k)
        .map(|i| (0..k).map(|j| xtx_inv[i][j] * xty[j]).sum())
        .collect();

    let residuals: Vec<f64> = (0..n)
        .map(|t| y[t] - (0..k).map(|i| coeffs[i] * columns[i][t]).sum::<f64>())
        .collect();

    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let dof = n as f64 - k as f64;
    if dof <= 0.0 {
        return None;
    }
    let sigma2 = ss_res / dof;
    let se: Vec<f64> = (0..k).map(|i| (sigma2 * xtx_inv[i][i]).sqrt()).collect();

    Some(MultiOlsResult {
        coeffs,
        se,
        residuals,
        n,
        k,
    })
}

/// Fit the (augmented) Dickey-Fuller regression
/// `Δy_t = c + gamma * y_{t-1} + Σ δ_i * Δy_{t-i} + ε_t` at a fixed lag
/// order, returning the level coefficient `gamma`, its standard error, the
/// number of usable observations, and the residuals (for AIC-based lag
/// selection in `fit_adf_regression`).
fn adf_regression_at_lag(y: &[f64], lags: usize) -> Option<MultiOlsResult> {
    let n = y.len();
    if n < lags + 5 {
        return None;
    }
    let dy: Vec<f64> = (0..n - 1).map(|i| y[i + 1] - y[i]).collect();
    if dy.len() <= lags {
        return None;
    }

    let start = lags;
    let m = dy.len() - start;
    if m < lags + 3 {
        return None;
    }

    let mut dep = Vec::with_capacity(m);
    let mut intercept = Vec::with_capacity(m);
    let mut level = Vec::with_capacity(m);
    let mut lag_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(m); lags];

    for t in start..dy.len() {
        dep.push(dy[t]);
        intercept.push(1.0);
        level.push(y[t]); // pre-change level corresponding to Δy_t = dy[t]
        for l in 1..=lags {
            lag_cols[l - 1].push(dy[t - l]);
        }
    }

    let mut columns = vec![intercept, level];
    columns.extend(lag_cols);
    ols_multi(&dep, &columns)
}

/// Choose the augmentation lag order by AIC over `0..=max_lag`, then return
/// `(gamma, se_gamma, n_obs, lags_used)` from the winning regression.
fn fit_adf_regression(series: &[f64], max_lag: usize) -> Option<(f64, f64, usize, usize)> {
    let mut best: Option<(usize, MultiOlsResult, f64)> = None;
    for lags in 0..=max_lag {
        if let Some(res) = adf_regression_at_lag(series, lags) {
            let ss_res: f64 = res.residuals.iter().map(|r| r * r).sum();
            if ss_res <= 0.0 {
                continue;
            }
            let n = res.n as f64;
            let k = res.k as f64;
            let aic = n * (ss_res / n).ln() + 2.0 * k;
            let better = best.as_ref().map(|(_, _, b)| aic < *b).unwrap_or(true);
            if better {
                best = Some((lags, res, aic));
            }
        }
    }
    let (lags_used, res, _aic) = best?;
    let se_gamma = res.se[1];
    if se_gamma <= 0.0 || !se_gamma.is_finite() {
        return None;
    }
    Some((res.coeffs[1], se_gamma, res.n, lags_used))
}

/// MacKinnon-style approximate critical-value breakpoints: `(n, 1%, 5%, 10%)`,
/// linearly interpolated by sample size. Values below the smallest listed
/// `n` clamp to that row; values above the largest clamp to the asymptotic row.
fn interpolate_critical(table: &[(f64, f64, f64, f64)], n: f64) -> CriticalValues {
    if n <= table[0].0 {
        let (_, a, b, c) = table[0];
        return CriticalValues {
            pct_1: a,
            pct_5: b,
            pct_10: c,
        };
    }
    for w in table.windows(2) {
        let (n0, a0, b0, c0) = w[0];
        let (n1, a1, b1, c1) = w[1];
        if n <= n1 {
            let frac = (n - n0) / (n1 - n0);
            return CriticalValues {
                pct_1: a0 + frac * (a1 - a0),
                pct_5: b0 + frac * (b1 - b0),
                pct_10: c0 + frac * (c1 - c0),
            };
        }
    }
    let (_, a, b, c) = table[table.len() - 1];
    CriticalValues {
        pct_1: a,
        pct_5: b,
        pct_10: c,
    }
}

/// Approximate critical values for a plain ADF test (constant, no trend) —
/// MacKinnon (1996) asymptotic response surface, coarsely interpolated.
fn adf_critical_values(n: usize) -> CriticalValues {
    const TABLE: [(f64, f64, f64, f64); 6] = [
        (25.0, -3.75, -3.00, -2.63),
        (50.0, -3.58, -2.93, -2.60),
        (100.0, -3.51, -2.89, -2.58),
        (250.0, -3.46, -2.88, -2.57),
        (500.0, -3.44, -2.87, -2.57),
        (100_000.0, -3.43, -2.86, -2.57),
    ];
    interpolate_critical(&TABLE, n as f64)
}

/// Approximate critical values for the Engle-Granger residual test with 2
/// variables (constant, no trend) — more negative than plain ADF's, since
/// testing residuals of an *estimated* regression requires more extreme
/// values than testing a raw series (MacKinnon 1991/2010).
fn engle_granger_critical_values(n: usize) -> CriticalValues {
    const TABLE: [(f64, f64, f64, f64); 6] = [
        (25.0, -4.30, -3.60, -3.24),
        (50.0, -4.05, -3.42, -3.10),
        (100.0, -3.97, -3.37, -3.06),
        (250.0, -3.93, -3.35, -3.04),
        (500.0, -3.91, -3.34, -3.04),
        (100_000.0, -3.90, -3.34, -3.04),
    ];
    interpolate_critical(&TABLE, n as f64)
}

/// Approximate one-sided p-value via piecewise-linear interpolation/
/// extrapolation through the three (critical value, p) anchor points. This
/// is a coarse approximation (not a reproduction of the true Dickey-Fuller
/// distribution's tail) — adequate for a plain-language significance flag,
/// same spirit as this crate's hand-rolled normal-CDF approximation in
/// `variance_ratio.rs`, but explicitly less precise since the true
/// distribution here is not normal.
fn approximate_df_p_value(statistic: f64, cv: &CriticalValues) -> f64 {
    let points = [(cv.pct_1, 0.01), (cv.pct_5, 0.05), (cv.pct_10, 0.10)];
    if statistic <= points[0].0 {
        let slope = (points[1].1 - points[0].1) / (points[1].0 - points[0].0);
        let p = points[0].1 + slope * (statistic - points[0].0);
        return p.clamp(0.0001, points[0].1);
    }
    if statistic >= points[2].0 {
        let slope = (points[2].1 - points[1].1) / (points[2].0 - points[1].0);
        let p = points[2].1 + slope * (statistic - points[2].0);
        return p.clamp(points[2].1, 0.999);
    }
    for w in points.windows(2) {
        let (cv_a, p_a) = w[0];
        let (cv_b, p_b) = w[1];
        if statistic >= cv_a && statistic <= cv_b {
            let frac = (statistic - cv_a) / (cv_b - cv_a);
            return p_a + frac * (p_b - p_a);
        }
    }
    0.5
}

/// Run an Augmented Dickey-Fuller unit-root test directly on `series`
/// (e.g. to sanity-check a spread's stationarity on its own, outside a full
/// two-step cointegration test). Lag order is chosen by AIC over `0..=max_lag`.
pub fn adf_test(series: &[f64], max_lag: usize) -> Option<AdfResult> {
    let (gamma, se_gamma, n_obs, lags_used) = fit_adf_regression(series, max_lag)?;
    let statistic = gamma / se_gamma;
    let critical_values = adf_critical_values(n_obs);
    let p_value = approximate_df_p_value(statistic, &critical_values);
    Some(AdfResult {
        statistic,
        p_value,
        critical_values,
        lags_used,
        n_obs,
        is_stationary: statistic < critical_values.pct_5,
    })
}

/// Full Engle-Granger two-step cointegration test: OLS hedge ratio of `y` on
/// `x`, then ADF (with Engle-Granger critical values) on the residuals.
/// `max_lag` bounds the AIC lag search for the residual ADF regression.
pub fn engle_granger_test(y: &[f64], x: &[f64], max_lag: usize) -> Option<CointegrationResult> {
    let hedge_ratio = ols_simple(y, x)?;
    let (gamma, se_gamma, n_obs, lags_used) = fit_adf_regression(&hedge_ratio.residuals, max_lag)?;
    let statistic = gamma / se_gamma;
    let critical_values = engle_granger_critical_values(n_obs);
    let p_value = approximate_df_p_value(statistic, &critical_values);
    let is_stationary = statistic < critical_values.pct_5;
    let adf_on_residuals = AdfResult {
        statistic,
        p_value,
        critical_values,
        lags_used,
        n_obs,
        is_stationary,
    };
    Some(CointegrationResult {
        hedge_ratio,
        adf_on_residuals,
        is_cointegrated: is_stationary,
    })
}

/// Rolling z-score of a spread series over a trailing `window`. The first
/// `window - 1` entries are `NaN` (insufficient history), matching the
/// pandas `.rolling().mean()` convention this codebase's Python strategy
/// side already expects.
pub fn spread_zscore(spread: &[f64], window: usize) -> Vec<f64> {
    let n = spread.len();
    let mut out = vec![f64::NAN; n];
    if window < 2 {
        return out;
    }
    for i in (window - 1)..n {
        let win = &spread[(i + 1 - window)..=i];
        let mean = win.iter().sum::<f64>() / window as f64;
        let var = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
        let std = var.sqrt();
        out[i] = if std > 0.0 { (spread[i] - mean) / std } else { 0.0 };
    }
    out
}

/// Ornstein-Uhlenbeck mean-reversion half-life of `spread`, in bars: fits
/// `Δspread_t = c + gamma * spread_{t-1} + ε_t` (the lags=0 Dickey-Fuller
/// regression), converts to the implied AR(1) coefficient `phi = 1 + gamma`,
/// and returns `-ln(2) / ln(phi)`. Returns `None` when `phi` is outside
/// `(0, 1)` — the spread isn't mean-reverting in the simple AR(1) sense (a
/// non-cointegrated pair, or one whose spread wanders/oscillates rather than
/// decaying back to its mean).
pub fn half_life(spread: &[f64]) -> Option<f64> {
    let (gamma, _se_gamma, _n_obs, _lags_used) = fit_adf_regression(spread, 0)?;
    let phi = 1.0 + gamma;
    if phi <= 0.0 || phi >= 1.0 {
        return None;
    }
    Some(-(2f64.ln()) / phi.ln())
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

    /// AR(1) series with a given mean-reversion coefficient `phi` (0 < phi < 1).
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

    // ---- ols_simple ----

    #[test]
    fn ols_recovers_exact_linear_relationship() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 3.0).collect();
        let ols = ols_simple(&y, &x).unwrap();
        assert!((ols.beta - 2.0).abs() < 1e-8);
        assert!((ols.alpha - 3.0).abs() < 1e-6);
        assert!((ols.r_squared - 1.0).abs() < 1e-8);
    }

    #[test]
    fn ols_rejects_too_short_or_mismatched_input() {
        assert!(ols_simple(&[1.0, 2.0], &[1.0, 2.0]).is_none());
        assert!(ols_simple(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn ols_rejects_zero_variance_x() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 5.0, 5.0, 5.0];
        assert!(ols_simple(&y, &x).is_none());
    }

    // ---- adf_test ----

    #[test]
    fn stationary_ar1_series_rejects_unit_root() {
        let series = ar1_series(2000, 0.5, 1.0, 1);
        let result = adf_test(&series, 4).expect("regression should fit");
        assert!(
            result.is_stationary,
            "phi=0.5 AR(1) should be flagged stationary, got statistic {}",
            result.statistic
        );
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn random_walk_does_not_reject_unit_root() {
        let series = random_walk(2000, 2);
        let result = adf_test(&series, 4).expect("regression should fit");
        assert!(
            !result.is_stationary,
            "pure random walk should not reject the unit-root null, got statistic {}",
            result.statistic
        );
    }

    #[test]
    fn adf_p_value_is_within_unit_interval() {
        for seed in [1, 2, 3, 4] {
            let series = ar1_series(500, 0.7, 1.0, seed);
            if let Some(r) = adf_test(&series, 4) {
                assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
            }
        }
    }

    #[test]
    fn adf_too_short_series_yields_none() {
        assert!(adf_test(&[1.0, 2.0, 3.0], 4).is_none());
    }

    // ---- engle_granger_test ----

    #[test]
    fn genuinely_cointegrated_pair_is_detected() {
        // x: a random walk (the common stochastic trend). y = 1.5*x + stationary noise.
        let x = random_walk(2000, 10);
        let noise = ar1_series(2000, 0.3, 1.0, 11);
        let y: Vec<f64> = x.iter().zip(noise.iter()).map(|(xi, ni)| 1.5 * xi + ni).collect();

        let result = engle_granger_test(&y, &x, 4).expect("regression should fit");
        assert!(
            (result.hedge_ratio.beta - 1.5).abs() < 0.1,
            "expected hedge ratio near 1.5, got {}",
            result.hedge_ratio.beta
        );
        assert!(
            result.is_cointegrated,
            "constructed cointegrated pair should be detected, got adf statistic {}",
            result.adf_on_residuals.statistic
        );
    }

    #[test]
    fn two_independent_random_walks_are_not_cointegrated() {
        let x = random_walk(2000, 20);
        let y = random_walk(2000, 21);
        let result = engle_granger_test(&y, &x, 4).expect("regression should fit");
        assert!(
            !result.is_cointegrated,
            "two independent random walks should not cointegrate, got adf statistic {}",
            result.adf_on_residuals.statistic
        );
    }

    #[test]
    fn engle_granger_too_short_series_yields_none() {
        assert!(engle_granger_test(&[1.0, 2.0], &[1.0, 2.0], 2).is_none());
    }

    // ---- spread_zscore ----

    #[test]
    fn spread_zscore_leading_window_is_nan() {
        let spread = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = spread_zscore(&spread, 3);
        assert!(z[0].is_nan());
        assert!(z[1].is_nan());
        assert!(!z[2].is_nan());
    }

    #[test]
    fn spread_zscore_flags_an_outlier() {
        let mut spread = vec![0.0; 30];
        spread[29] = 10.0; // sharp deviation from a flat history
        let z = spread_zscore(&spread, 20);
        assert!(z[29] > 3.0, "expected a large positive z-score, got {}", z[29]);
    }

    #[test]
    fn spread_zscore_window_below_two_is_all_nan() {
        let spread = vec![1.0, 2.0, 3.0];
        let z = spread_zscore(&spread, 1);
        assert!(z.iter().all(|v| v.is_nan()));
    }

    // ---- half_life ----

    #[test]
    fn half_life_matches_theoretical_ar1_value() {
        let phi = 0.9;
        let spread = ar1_series(5000, phi, 1.0, 99);
        let hl = half_life(&spread).expect("should fit");
        let theoretical = -(2f64.ln()) / phi.ln();
        assert!(
            (hl - theoretical).abs() / theoretical < 0.25,
            "half-life {} too far from theoretical {}",
            hl,
            theoretical
        );
    }

    #[test]
    fn half_life_is_none_or_implausibly_long_for_a_random_walk() {
        // A true random walk has phi == 1 exactly (no mean reversion at all),
        // but OLS's well-known finite-sample downward bias for near-unit-root
        // series (the same bias ADF/Dickey-Fuller tests exist to correct for)
        // means the *estimated* phi often comes out just under 1 even here —
        // so `half_life` won't always return `None`. What it must never do is
        // report a half-life that looks like genuine, tradeable mean
        // reversion: for a 2000-bar series, anything at or beyond a sizable
        // fraction of the series length reflects noise in the phi estimate,
        // not real reversion.
        let series = random_walk(2000, 30);
        if let Some(hl) = half_life(&series) {
            assert!(
                hl > 200.0,
                "random walk half-life {} looks like real mean reversion, not estimation noise",
                hl
            );
        }
    }
}
