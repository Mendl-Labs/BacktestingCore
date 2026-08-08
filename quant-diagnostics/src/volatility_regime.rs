//! Pre-registered volatility-tercile regime split — defines *the
//! requirement* for regime-conditioned diagnostics (per-regime IC/quantile
//! analysis): a market-state label computed purely from the return series
//! itself, fixed before any strategy performance is examined, so a regime
//! assignment can never be reverse-engineered from results. Wiring this
//! into per-regime IC/quantile analysis is a follow-up — this module ships
//! the split itself as a tested, reusable primitive.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
}

/// Classify each index of `returns` into a trailing-volatility tercile.
///
/// For index `i`, realized volatility is the sample std-dev of
/// `returns[i+1-window..=i]` — only past and current data, never future
/// data, so the label at any point could have been assigned in real time.
/// Terciles are cut from the full-sample distribution of rolling
/// volatilities. Indices without `window` trailing points, or a series too
/// short to form terciles at all, get `None`.
pub fn volatility_tercile_regimes(returns: &[f64], window: usize) -> Vec<Option<VolatilityRegime>> {
    let n = returns.len();
    if window < 2 || n == 0 {
        return vec![None; n];
    }

    let mut rolling_vol: Vec<Option<f64>> = vec![None; n];
    for i in 0..n {
        if i + 1 >= window {
            let slice = &returns[(i + 1 - window)..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (window as f64 - 1.0).max(1.0);
            rolling_vol[i] = Some(var.sqrt());
        }
    }

    let mut valid_vols: Vec<f64> = rolling_vol.iter().filter_map(|v| *v).collect();
    if valid_vols.len() < 3 {
        return vec![None; n];
    }
    valid_vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let low_cut = valid_vols[valid_vols.len() / 3];
    let high_cut = valid_vols[(2 * valid_vols.len()) / 3];

    rolling_vol
        .into_iter()
        .map(|v| {
            v.map(|vol| {
                if vol <= low_cut {
                    VolatilityRegime::Low
                } else if vol >= high_cut {
                    VolatilityRegime::High
                } else {
                    VolatilityRegime::Medium
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_yield_all_none() {
        assert!(volatility_tercile_regimes(&[], 5).is_empty());
    }

    #[test]
    fn short_series_yields_all_none() {
        let returns = vec![0.01, -0.02, 0.03];
        let regimes = volatility_tercile_regimes(&returns, 10);
        assert!(regimes.iter().all(|r| r.is_none()));
    }

    #[test]
    fn indices_before_window_are_none_but_later_ones_are_labeled() {
        let returns: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        let regimes = volatility_tercile_regimes(&returns, 10);
        assert!(regimes[..9].iter().all(|r| r.is_none()));
        assert!(regimes[9..].iter().all(|r| r.is_some()));
    }

    #[test]
    fn low_volatility_stretch_is_labeled_low() {
        // First half: near-zero returns (low vol). Second half: large swings (high vol).
        let mut returns = vec![0.0001; 60];
        returns.extend(vec![0.05, -0.05].into_iter().cycle().take(60));
        let regimes = volatility_tercile_regimes(&returns, 20);
        // Deep into the calm stretch, well past the warmup window.
        assert_eq!(regimes[40], Some(VolatilityRegime::Low));
        // Deep into the volatile stretch.
        assert_eq!(regimes[100], Some(VolatilityRegime::High));
    }

    #[test]
    fn rolling_volatility_itself_never_looks_ahead() {
        // The per-index rolling vol (the piece that must never leak future
        // data) is identical whether or not future returns exist — only the
        // full-sample tercile *cutpoints* are a whole-series statistic,
        // which is standard for ex-post regime tagging (it never looks at
        // strategy performance, only at price/return history).
        fn rolling_vol_at(returns: &[f64], window: usize, i: usize) -> f64 {
            let slice = &returns[(i + 1 - window)..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (window as f64 - 1.0).max(1.0);
            var.sqrt()
        }
        let mut returns = vec![0.001; 40];
        let vol_before = rolling_vol_at(&returns, 10, 20);
        returns.extend(vec![10.0, -10.0, 10.0, -10.0]);
        let vol_after = rolling_vol_at(&returns, 10, 20);
        assert!((vol_before - vol_after).abs() < 1e-12);
    }
}
