//! Calendar-bucket seasonality test — groups a return series into caller-
//! supplied buckets (e.g. day-of-week, month-of-year) and reports each
//! bucket's mean return with a significance test against zero. Generic over
//! bucket meaning: this crate has no calendar dependency, so callers convert
//! their own timestamps into small integer bucket ids (e.g. 0=Monday..6=Sunday)
//! before calling in.

use serde::{Deserialize, Serialize};

use crate::ic::two_sided_p_value;

/// Minimum observations in a bucket before a significance test is meaningful.
/// Mirrors `compute_ic`'s own threshold.
const MIN_BUCKET_N: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeasonalityBucket {
    /// Caller-defined bucket id (e.g. 0..6 for day-of-week, 0..11 for month).
    pub bucket: u8,
    pub mean_return: f64,
    pub n: usize,
    pub t_stat: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Group `returns[i]` by `bucket_ids[i]` and test each bucket's mean return
/// against zero (one-sample t-test, approximated with the normal CDF —
/// adequate once a bucket has more than a handful of points). Buckets with
/// fewer than `MIN_BUCKET_N` observations are omitted rather than reported
/// with an unreliable statistic. Output is sorted by bucket id ascending.
pub fn seasonality_by_bucket(returns: &[f64], bucket_ids: &[u8]) -> Vec<SeasonalityBucket> {
    if returns.len() != bucket_ids.len() || returns.is_empty() {
        return Vec::new();
    }

    let mut max_bucket = 0u8;
    for &b in bucket_ids {
        max_bucket = max_bucket.max(b);
    }

    let mut out = Vec::new();
    for bucket in 0..=max_bucket {
        let values: Vec<f64> = returns
            .iter()
            .zip(bucket_ids.iter())
            .filter(|(_, &b)| b == bucket)
            .map(|(&r, _)| r)
            .collect();
        let n = values.len();
        if n < MIN_BUCKET_N {
            continue;
        }

        let mean = values.iter().sum::<f64>() / n as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
        let std_dev = var.sqrt();
        let std_err = std_dev / (n as f64).sqrt();
        // A near-zero std_err (an effectively constant bucket) would blow up
        // t_stat via floating-point noise even when the "true" variance is
        // zero -- an exact `<= 0.0` check misses that, since summing equal
        // floats can leave a tiny nonzero rounding residue. Use a small
        // absolute floor instead of exact-zero comparison.
        if std_err < 1e-9 {
            continue;
        }

        let t_stat = mean / std_err;
        let p_value = two_sided_p_value(t_stat);
        out.push(SeasonalityBucket {
            bucket,
            mean_return: mean,
            n,
            t_stat,
            p_value,
            significant: p_value < 0.05,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_lengths_return_empty() {
        assert!(seasonality_by_bucket(&[1.0, 2.0], &[0]).is_empty());
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(seasonality_by_bucket(&[], &[]).is_empty());
    }

    #[test]
    fn buckets_with_too_few_observations_are_omitted() {
        // Bucket 0 has 3 points (< MIN_BUCKET_N=5), bucket 1 has 6 with real variance.
        let returns = vec![0.01, 0.02, 0.03, 0.01, 0.03, -0.01, 0.02, 0.015, 0.005];
        let buckets = vec![0, 0, 0, 1, 1, 1, 1, 1, 1];
        let result = seasonality_by_bucket(&returns, &buckets);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bucket, 1);
        assert_eq!(result[0].n, 6);
    }

    #[test]
    fn a_bucket_with_a_consistent_positive_mean_is_significant() {
        let returns: Vec<f64> = (0..20).map(|i| 0.02 + (i as f64 % 3.0) * 0.0001).collect();
        let buckets = vec![0u8; 20];
        let result = seasonality_by_bucket(&returns, &buckets);
        assert_eq!(result.len(), 1);
        assert!(result[0].mean_return > 0.0);
        assert!(result[0].significant);
    }

    #[test]
    fn a_bucket_with_zero_variance_is_skipped_not_panicked() {
        let returns = vec![0.01; 10];
        let buckets = vec![0u8; 10];
        let result = seasonality_by_bucket(&returns, &buckets);
        // std_err is 0 for a constant series -- must be omitted, not divide-by-zero.
        assert!(result.is_empty());
    }

    #[test]
    fn output_is_sorted_by_bucket_ascending() {
        let mut returns = vec![0.01; 6];
        returns.extend(vec![-0.02, 0.03, -0.01, 0.02, 0.01, 0.04]);
        let mut buckets = vec![1u8; 6];
        buckets.extend(vec![0u8; 6]);
        let result = seasonality_by_bucket(&returns, &buckets);
        let ids: Vec<u8> = result.iter().map(|b| b.bucket).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}
