//! Volatility risk premium (VRP) significance test — one-sample z-test of
//! whether a current implied-volatility reading differs significantly from
//! the distribution of the underlying's own trailing realized volatility.
//!
//! A bare "IV looks elevated" read (comparing one IV snapshot against one
//! point estimate of RV) says nothing about whether that gap is real
//! structure or noise around a wide RV distribution. This test instead
//! builds the full trailing RV distribution (rolling-window realized vol,
//! reusing `volatility_regime::rolling_volatility`) and asks whether IV sits
//! far enough outside it to reject H0: IV == mean(RV) — the same
//! normal-CDF-approximation shape as `variance_ratio::two_sided_p_value` /
//! `metrics::significance::sharpe_significance`, mirrored deliberately
//! rather than invented fresh (`crate::ic::two_sided_p_value` is reused
//! directly — that helper is already shared crate-wide with `seasonality`).
//!
//! No historical IV series is available from this platform's options vendor
//! (deliberate limitation, not a bug — see `options_signals_service.rs`'s
//! own doc comment), so IV-rank-vs-its-own-history is not a test this crate
//! can perform. IV-vs-RV-distribution is: RV is derived purely from OHLCV,
//! which the platform already has full history for.

use serde::{Deserialize, Serialize};

use crate::ic::two_sided_p_value;
use crate::volatility_regime::rolling_volatility;

/// Minimum trailing RV observations required to run the test — too few and
/// the "distribution" is really just a couple of point estimates, the same
/// reasoning `compute_ic`'s own `n < 5` guard and `variance_ratio`'s `t < 4`
/// guard apply elsewhere in this crate, just set higher here since a
/// meaningful RV *distribution* (not just a single VR/IC estimate) needs
/// more than a handful of rolling windows to not be dominated by overlap.
const MIN_RV_OBSERVATIONS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VrpResult {
    /// z-statistic for H0: near_atm_iv == mean(trailing RV distribution).
    pub z_stat: f64,
    /// Two-sided p-value from the standard normal approximation.
    pub p_value: f64,
    /// Mean of the trailing rolling-RV distribution, annualized to the same
    /// units as `near_atm_iv` (via `periods_per_year`).
    pub rv_mean: f64,
    /// Sample std-dev of the trailing rolling-RV distribution (annualized).
    pub rv_std: f64,
    /// `near_atm_iv - rv_mean`. Positive: IV priced above what trailing RV
    /// would justify (the classic "sell vol" VRP direction). Negative: IV
    /// priced below trailing RV.
    pub iv_rv_spread: f64,
    /// Number of rolling-RV windows the distribution was built from.
    pub n_obs: usize,
    /// True when |z_stat| implies rejection of H0 at 5%.
    pub significant: bool,
}

/// Test whether `near_atm_iv` differs significantly from the distribution of
/// the underlying's own trailing realized volatility.
///
/// `returns` should be the underlying's per-bar simple or log returns,
/// chronological, spanning the lookback window the caller wants the RV
/// distribution built over (e.g. 6-12 months of daily bars). `rv_window` is
/// the rolling-RV window in bars (e.g. 20 for a ~1-month daily-bar RV
/// estimate). `periods_per_year` must match the bar granularity of
/// `returns` (see `metrics::significance::periods_per_year_for_interval` on
/// the BacktestingEngine side, which is this crate's only consumer for this
/// function today) — it's what turns the per-bar rolling std-dev into an
/// annualized figure comparable to `near_atm_iv`, which options vendors
/// always quote annualized.
///
/// Returns `None` when there isn't enough trailing RV data to form a
/// meaningful distribution, or the distribution is degenerate (zero
/// variance — e.g. a constant return series).
pub fn vrp_significance(
    near_atm_iv: f64,
    returns: &[f64],
    rv_window: usize,
    periods_per_year: f64,
) -> Option<VrpResult> {
    if !near_atm_iv.is_finite() || !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return None;
    }

    let annualization = periods_per_year.sqrt();
    let rv_series: Vec<f64> = rolling_volatility(returns, rv_window)
        .into_iter()
        .flatten()
        .map(|daily_vol| daily_vol * annualization)
        .collect();

    let n = rv_series.len();
    if n < MIN_RV_OBSERVATIONS {
        return None;
    }

    let rv_mean = rv_series.iter().sum::<f64>() / n as f64;
    let rv_var =
        rv_series.iter().map(|v| (v - rv_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let rv_std = rv_var.sqrt();
    if rv_std <= 0.0 {
        return None;
    }

    let se = rv_std / (n as f64).sqrt();
    let z_stat = (near_atm_iv - rv_mean) / se;
    let p_value = two_sided_p_value(z_stat);

    Some(VrpResult {
        z_stat,
        p_value,
        rv_mean,
        rv_std,
        iv_rv_spread: near_atm_iv - rv_mean,
        n_obs: n,
        significant: p_value < 0.05,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn stable_vol_returns(n: usize, daily_sigma: f64, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, daily_sigma).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn too_short_series_yields_none() {
        let returns = vec![0.01, -0.02, 0.03];
        assert!(vrp_significance(0.25, &returns, 20, 365.0).is_none());
    }

    #[test]
    fn empty_returns_yield_none() {
        assert!(vrp_significance(0.25, &[], 20, 365.0).is_none());
    }

    #[test]
    fn zero_variance_returns_yield_none() {
        let returns = vec![0.0; 200];
        assert!(vrp_significance(0.25, &returns, 20, 365.0).is_none());
    }

    #[test]
    fn non_finite_or_non_positive_inputs_yield_none() {
        let returns = stable_vol_returns(200, 0.01, 1);
        assert!(vrp_significance(f64::NAN, &returns, 20, 365.0).is_none());
        assert!(vrp_significance(0.25, &returns, 20, 0.0).is_none());
        assert!(vrp_significance(0.25, &returns, 20, -365.0).is_none());
    }

    #[test]
    fn iv_matching_the_rv_distribution_mean_is_not_significant() {
        // ~1% daily sigma annualizes to ~1% * sqrt(365) ≈ 19.1%.
        let returns = stable_vol_returns(1000, 0.01, 42);
        let result = vrp_significance(0.191, &returns, 20, 365.0).unwrap();
        assert!(result.p_value > 0.10, "expected a non-significant p-value, got {}", result.p_value);
        assert!(!result.significant);
    }

    #[test]
    fn iv_far_above_the_rv_distribution_is_significant_and_positive_spread() {
        let returns = stable_vol_returns(1000, 0.01, 42);
        // Deliberately far above the ~19.1% RV mean -- a real VRP read.
        let result = vrp_significance(0.60, &returns, 20, 365.0).unwrap();
        assert!(result.significant, "expected significance, p={}", result.p_value);
        assert!(result.iv_rv_spread > 0.0, "expected positive spread, got {}", result.iv_rv_spread);
        assert!(result.z_stat > 0.0);
    }

    #[test]
    fn iv_far_below_the_rv_distribution_is_significant_and_negative_spread() {
        let returns = stable_vol_returns(1000, 0.01, 42);
        let result = vrp_significance(0.02, &returns, 20, 365.0).unwrap();
        assert!(result.significant, "expected significance, p={}", result.p_value);
        assert!(result.iv_rv_spread < 0.0, "expected negative spread, got {}", result.iv_rv_spread);
        assert!(result.z_stat < 0.0);
    }

    #[test]
    fn p_value_is_within_unit_interval() {
        let returns = stable_vol_returns(500, 0.015, 7);
        for iv in [0.05, 0.15, 0.25, 0.40, 0.80] {
            let result = vrp_significance(iv, &returns, 20, 365.0).unwrap();
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0, "iv={iv} p={}", result.p_value);
        }
    }

    #[test]
    fn rv_mean_and_std_are_annualized_not_raw_daily_values() {
        let returns = stable_vol_returns(1000, 0.01, 42);
        let result = vrp_significance(0.191, &returns, 20, 365.0).unwrap();
        // A raw daily std-dev of ~0.01 would never be this large -- confirms
        // the sqrt(periods_per_year) annualization actually applied.
        assert!(result.rv_mean > 0.10, "rv_mean looks unannualized: {}", result.rv_mean);
    }

    #[test]
    fn n_obs_matches_the_number_of_valid_rolling_windows() {
        let returns = stable_vol_returns(200, 0.01, 3);
        let result = vrp_significance(0.20, &returns, 20, 365.0).unwrap();
        // 200 bars, window 20 -> 200 - 20 + 1 = 181 valid windows.
        assert_eq!(result.n_obs, 181);
    }
}
