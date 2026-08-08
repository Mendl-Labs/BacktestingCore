//! Benjamini-Hochberg false-discovery-rate correction — used whenever
//! several statistical tests are run on the same underlying data and
//! reported together (e.g. a variance-ratio term structure at several
//! horizons, a day-of-week seasonality bucket per weekday, a regime-
//! conditioned IC per volatility regime). Reporting each test's raw p-value
//! independently understates the true false-positive rate; this correction
//! is the standard, cheap fix (Benjamini & Hochberg 1995).

/// Compute BH-adjusted p-values ("q-values"), same order and length as the
/// input. A test at original index `i` is significant at level `alpha` iff
/// `result[i] < alpha`. Returns an empty vec for empty input.
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Standard BH construction: q_(i) = min_{k=i}^{m} min(1, m/k * p_(k)),
    // computed from the largest rank down so each adjusted value is never
    // smaller than a higher-ranked (larger p-value) test's adjustment.
    let mut adjusted_sorted = vec![0.0; m];
    let mut min_so_far = 1.0_f64;
    for rank in (0..m).rev() {
        let (_, p) = indexed[rank];
        let bh_value = (p * m as f64 / (rank as f64 + 1.0)).min(1.0);
        min_so_far = min_so_far.min(bh_value);
        adjusted_sorted[rank] = min_so_far;
    }

    let mut result = vec![0.0; m];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = adjusted_sorted[rank];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty() {
        assert!(benjamini_hochberg(&[]).is_empty());
    }

    #[test]
    fn single_p_value_is_unchanged() {
        let adjusted = benjamini_hochberg(&[0.03]);
        assert_eq!(adjusted, vec![0.03]);
    }

    #[test]
    fn known_example_matches_hand_computed_values() {
        // p = [0.01, 0.04, 0.03, 0.005], m=4.
        // Sorted: 0.005(rank1), 0.01(rank2), 0.03(rank3), 0.04(rank4).
        // q_(4)=0.04, q_(3)=min(0.04, 4/3*0.03=0.04)=0.04,
        // q_(2)=min(0.04, 4/2*0.01=0.02)=0.02, q_(1)=min(0.02, 4/1*0.005=0.02)=0.02.
        let adjusted = benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005]);
        assert!((adjusted[0] - 0.02).abs() < 1e-9);
        assert!((adjusted[1] - 0.04).abs() < 1e-9);
        assert!((adjusted[2] - 0.04).abs() < 1e-9);
        assert!((adjusted[3] - 0.02).abs() < 1e-9);
    }

    #[test]
    fn adjusted_values_are_never_smaller_than_raw_p_values_scaled_by_rank() {
        // Sanity: BH adjustment never makes a result LESS conservative than
        // its own Bonferroni-at-its-rank value would allow -- i.e. every
        // adjusted value should be >= the raw p-value (adjustment inflates,
        // never deflates, individual significance).
        let raw = vec![0.001, 0.2, 0.05, 0.3, 0.02];
        let adjusted = benjamini_hochberg(&raw);
        for (r, a) in raw.iter().zip(adjusted.iter()) {
            assert!(a >= r, "adjusted {} should be >= raw {}", a, r);
        }
    }

    #[test]
    fn all_large_p_values_stay_non_significant() {
        let adjusted = benjamini_hochberg(&[0.5, 0.6, 0.7]);
        assert!(adjusted.iter().all(|&q| q > 0.05));
    }

    #[test]
    fn adjusted_values_are_monotonic_when_sorted_by_rank() {
        let raw = vec![0.2, 0.001, 0.15, 0.04, 0.09];
        let adjusted = benjamini_hochberg(&raw);
        let mut pairs: Vec<(f64, f64)> = raw.iter().copied().zip(adjusted.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in pairs.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-12, "BH q-values must be non-decreasing in p-value rank");
        }
    }
}
