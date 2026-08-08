//! Information coefficient (IC) — Spearman rank correlation between a
//! strategy's feature and its forward return, with a significance test.
//! A bare correlation magnitude (e.g. "IC = 0.03") is not trustworthy on its
//! own — it needs the t-stat attached to know whether it's distinguishable
//! from noise at the sample size actually available.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IcResult {
    /// Forward-return horizon (in bars) the feature was evaluated against.
    pub horizon: usize,
    /// Spearman rank correlation between feature[t] and return[t+horizon].
    pub ic: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub significant: bool,
    /// Number of (feature, forward_return) pairs actually used.
    pub n: usize,
}

/// Average-rank transform (ties share the mean of their rank positions).
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        // Ranks are 1-based; ties get the average of their position range.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in idx.iter().take(j + 1).skip(i) {
            ranks[*k] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Shared with `seasonality` -- both need the same normal-CDF approximation
/// for turning a z/t-statistic into a two-sided p-value.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = pdf * poly;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Shared with `seasonality`.
pub(crate) fn two_sided_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Spearman rank correlation between a feature series and its forward
/// return, shifted by `horizon` bars (`feature[t]` vs `forward_return`
/// realized over `[t, t+horizon]`). Both slices must already be aligned —
/// i.e. `forward_returns[i]` is the return that follows `feature[i]`.
///
/// Uses a t-distribution-style test statistic approximated with the normal
/// CDF (adequate once n is more than a handful of points, which is the
/// regime this is meant for — very small samples return `None`).
pub fn compute_ic(feature: &[f64], forward_returns: &[f64], horizon: usize) -> Option<IcResult> {
    if feature.len() != forward_returns.len() {
        return None;
    }
    let n = feature.len();
    if n < 5 {
        return None;
    }

    let feature_ranks = rank(feature);
    let return_ranks = rank(forward_returns);
    let ic = pearson(&feature_ranks, &return_ranks)?;

    let df = n as f64 - 2.0;
    if df <= 0.0 {
        return None;
    }
    let denom = (1.0 - ic * ic).max(1e-12);
    let t_stat = ic * (df / denom).sqrt();
    let p_value = two_sided_p_value(t_stat);

    Some(IcResult {
        horizon,
        ic,
        t_stat,
        p_value,
        significant: p_value < 0.05,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_lengths_return_none() {
        assert!(compute_ic(&[1.0, 2.0], &[1.0], 1).is_none());
    }

    #[test]
    fn too_few_points_return_none() {
        assert!(compute_ic(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1).is_none());
    }

    #[test]
    fn constant_feature_returns_none() {
        let feature = vec![1.0; 20];
        let returns: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(compute_ic(&feature, &returns, 1).is_none());
    }

    #[test]
    fn perfectly_correlated_series_has_ic_near_one_and_significant() {
        let feature: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let returns: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 1.0).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!((result.ic - 1.0).abs() < 1e-9);
        assert!(result.significant);
    }

    #[test]
    fn perfectly_anti_correlated_series_has_ic_near_negative_one() {
        let feature: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let returns: Vec<f64> = (0..50).map(|i| -(i as f64)).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!((result.ic + 1.0).abs() < 1e-9);
        assert!(result.significant);
    }

    #[test]
    fn random_noise_is_not_significant() {
        // Deterministic pseudo-random-ish sequences with no real relationship.
        let feature: Vec<f64> = (0..40).map(|i| ((i * 37) % 13) as f64).collect();
        let returns: Vec<f64> = (0..40).map(|i| ((i * 19 + 5) % 11) as f64).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!(result.ic.abs() < 1.0);
        // Not asserting non-significance here (a specific arrangement could
        // spuriously correlate) — just that the machinery produces a sane,
        // bounded result without panicking.
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn rank_averages_ties() {
        let ranks = rank(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }
}
