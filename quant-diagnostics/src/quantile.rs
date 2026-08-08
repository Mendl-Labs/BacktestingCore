//! Quantile analysis — bucket a feature's values into ranked groups and
//! compare each group's mean forward return. Bucket count degrades
//! gracefully with sample size instead of producing unstable buckets (or
//! silently disabling the check) when a test period is short.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantileBucket {
    /// 0-indexed bucket, ordered from lowest to highest feature value.
    pub bucket: usize,
    pub mean_feature: f64,
    pub mean_forward_return: f64,
    pub count: usize,
}

/// Below this many points, no bucketing is meaningful at all.
const MIN_POINTS_FOR_ANY_BUCKETING: usize = 9;

/// Choose a bucket count that degrades gracefully with sample size:
/// terciles for small samples, quintiles once there's enough data, and the
/// caller's requested count only once there's comfortably enough per bucket.
fn resolve_bucket_count(n: usize, requested: usize) -> usize {
    let requested = requested.max(2);
    if n < MIN_POINTS_FOR_ANY_BUCKETING {
        0
    } else if n < 30 {
        3
    } else if n < 100 {
        requested.min(5)
    } else {
        requested.min(10)
    }
}

/// Bucket `feature` into ranked groups and report each group's mean forward
/// return. `requested_buckets` is a target, not a guarantee — the actual
/// count degrades per [`resolve_bucket_count`] based on `n`. Returns an
/// empty vec when there isn't enough data for any meaningful bucketing.
pub fn quantile_analysis(
    feature: &[f64],
    forward_returns: &[f64],
    requested_buckets: usize,
) -> Vec<QuantileBucket> {
    if feature.len() != forward_returns.len() {
        return Vec::new();
    }
    let n = feature.len();
    let bucket_count = resolve_bucket_count(n, requested_buckets);
    if bucket_count == 0 {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| feature[a].partial_cmp(&feature[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut buckets = Vec::with_capacity(bucket_count);
    let base_size = n / bucket_count;
    let remainder = n % bucket_count;
    let mut cursor = 0;

    for b in 0..bucket_count {
        // Distribute the remainder across the first buckets so every point
        // is used and no bucket is starved.
        let size = base_size + if b < remainder { 1 } else { 0 };
        if size == 0 {
            continue;
        }
        let slice = &idx[cursor..cursor + size];
        cursor += size;

        let mean_feature = slice.iter().map(|&i| feature[i]).sum::<f64>() / size as f64;
        let mean_forward_return = slice.iter().map(|&i| forward_returns[i]).sum::<f64>() / size as f64;

        buckets.push(QuantileBucket {
            bucket: b,
            mean_feature,
            mean_forward_return,
            count: size,
        });
    }

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_lengths_return_empty() {
        assert!(quantile_analysis(&[1.0, 2.0], &[1.0], 5).is_empty());
    }

    #[test]
    fn too_few_points_returns_empty() {
        let feature: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let returns = feature.clone();
        assert!(quantile_analysis(&feature, &returns, 5).is_empty());
    }

    #[test]
    fn small_sample_degrades_to_terciles() {
        let feature: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let returns = feature.clone();
        let buckets = quantile_analysis(&feature, &returns, 5);
        assert_eq!(buckets.len(), 3);
    }

    #[test]
    fn large_sample_uses_requested_buckets() {
        let feature: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let returns = feature.clone();
        let buckets = quantile_analysis(&feature, &returns, 5);
        assert_eq!(buckets.len(), 5);
    }

    #[test]
    fn buckets_cover_every_point() {
        let feature: Vec<f64> = (0..37).map(|i| i as f64).collect();
        let returns = feature.clone();
        let buckets = quantile_analysis(&feature, &returns, 5);
        let total: usize = buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 37);
    }

    #[test]
    fn monotonic_relationship_produces_monotonic_bucket_means() {
        let feature: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let returns: Vec<f64> = feature.iter().map(|f| f * 2.0).collect();
        let buckets = quantile_analysis(&feature, &returns, 5);
        for w in buckets.windows(2) {
            assert!(w[1].mean_forward_return > w[0].mean_forward_return);
        }
    }
}
