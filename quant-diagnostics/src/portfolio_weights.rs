//! Shrinkage-covariance minimum-variance portfolio weighting (breadth
//! Phase 2, 2026-09-04 quant council).
//!
//! Replaces naive univariate weighting (equal / inverse-volatility /
//! Sharpe-weighted -- each a function of one constituent's OWN stats only)
//! with a weight-solve that actually consumes the covariance structure
//! between constituents, the way `effective_breadth` already measures it
//! but nothing previously *used* it to set weights.
//!
//! Two deliberate, documented simplifications versus the full academic
//! machinery, matching this crate's existing "hand-rolled approximation,
//! dependency-light" convention (see `linalg.rs`'s own doc comment) and the
//! council's own Round 2 ruling that a working, honestly-scoped v1 beats
//! trying to ship a "final" covariance model before real usage data exists
//! to calibrate against:
//!
//! 1. **Shrinkage intensity** is a simple, sample-size-scaled heuristic
//!    (`n / (n + t)`, clamped), NOT the full Ledoit-Wolf (2003/2004)
//!    asymptotic-MSE-minimizing estimator -- that requires estimating the
//!    covariance of the covariance-matrix entries themselves, more
//!    machinery than a v1 needs. The shrinkage TARGET (constant
//!    correlation, same variances as the sample) is the well-established
//!    Ledoit-Wolf target and is implemented exactly.
//! 2. **Long-only projection** clips negative unconstrained min-variance
//!    weights to zero and renormalizes, rather than solving the exact
//!    non-negativity-constrained quadratic program. A reasonable
//!    approximation for the small constituent counts this is sized for
//!    (see `MAX_CANDIDATE_POOL` in `program`'s `strategy_ensemble_service`),
//!    not exact QP.
//!
//! Turnover capping (limiting how much a weight may move per rebalance) is
//! deliberately a SEPARATE, later post-hoc step (`cap_turnover`) rather than
//! a penalty term inside the optimization objective itself -- the council's
//! Mathematical Purist argued for an in-objective penalty, but that turns
//! this from a closed-form solve into a full QP; the post-hoc cap is an
//! honest, documented simplification for v1, not a silent substitution.

use crate::linalg::invert_matrix;

/// Sample covariance matrix (N x N) of N return series, each of length T.
/// `returns[i]` is series `i`'s T observations. Uses the unbiased (T-1)
/// divisor. Returns `None` if fewer than 2 observations or fewer than 2
/// series are given.
pub fn sample_covariance(returns: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = returns.len();
    if n < 2 {
        return None;
    }
    let t = returns[0].len();
    if t < 2 || returns.iter().any(|r| r.len() != t) {
        return None;
    }
    let means: Vec<f64> = returns.iter().map(|r| r.iter().sum::<f64>() / t as f64).collect();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let c: f64 = (0..t).map(|k| (returns[i][k] - means[i]) * (returns[j][k] - means[j])).sum::<f64>()
                / (t - 1) as f64;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    Some(cov)
}

/// Shrink `sample_cov` toward a constant-correlation target: same
/// diagonal (variances) as the sample, off-diagonal `rho_bar * sigma_i *
/// sigma_j` where `rho_bar` is the average of every sample pairwise
/// correlation. Shrinkage intensity `delta = clamp(n / (n + t_eff), 0.1,
/// 0.9)` -- see module doc for why this is a documented simplification of
/// the full Ledoit-Wolf estimator rather than the estimator itself.
/// `t_eff` is the number of return observations the sample covariance was
/// built from (needed for the intensity formula but not recoverable from
/// the matrix alone).
pub fn shrink_toward_constant_correlation(sample_cov: &[Vec<f64>], t_eff: usize) -> Vec<Vec<f64>> {
    let n = sample_cov.len();
    if n == 0 {
        return Vec::new();
    }
    let sigmas: Vec<f64> = (0..n).map(|i| sample_cov[i][i].max(0.0).sqrt()).collect();

    let mut rho_sum = 0.0;
    let mut rho_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if sigmas[i] > 1e-12 && sigmas[j] > 1e-12 {
                rho_sum += sample_cov[i][j] / (sigmas[i] * sigmas[j]);
                rho_count += 1;
            }
        }
    }
    let rho_bar = if rho_count > 0 { rho_sum / rho_count as f64 } else { 0.0 };

    let delta = (n as f64 / (n as f64 + t_eff as f64)).clamp(0.1, 0.9);

    let mut shrunk = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let target = if i == j {
                sample_cov[i][i]
            } else {
                rho_bar * sigmas[i] * sigmas[j]
            };
            shrunk[i][j] = delta * target + (1.0 - delta) * sample_cov[i][j];
        }
    }
    shrunk
}

/// Unconstrained minimum-variance weights from a covariance matrix:
/// `w = Sigma^-1 * 1 / (1' * Sigma^-1 * 1)`. Returns `None` if the matrix
/// is singular/near-singular (see `linalg::invert_matrix`).
fn unconstrained_min_variance_weights(cov: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = cov.len();
    if n == 0 {
        return None;
    }
    let inv = invert_matrix(cov)?;
    let row_sums: Vec<f64> = inv.iter().map(|row| row.iter().sum::<f64>()).collect();
    let total: f64 = row_sums.iter().sum();
    if total.abs() < 1e-12 {
        return None;
    }
    Some(row_sums.iter().map(|s| s / total).collect())
}

/// Long-only minimum-variance weights: solve the unconstrained closed form,
/// then clip any negative weight to zero and renormalize the remaining
/// positive weights to sum to 1.0. See module doc for why this is a
/// documented approximation of the exact non-negativity-constrained QP
/// rather than that QP itself. Falls back to equal weighting if the
/// unconstrained solve is singular, or if every weight comes out
/// non-positive (degenerate covariance input).
pub fn min_variance_weights(shrunk_cov: &[Vec<f64>]) -> Vec<f64> {
    let n = shrunk_cov.len();
    if n == 0 {
        return Vec::new();
    }
    let equal = vec![1.0 / n as f64; n];
    let raw = match unconstrained_min_variance_weights(shrunk_cov) {
        Some(w) => w,
        None => return equal,
    };
    let clipped: Vec<f64> = raw.iter().map(|w| w.max(0.0)).collect();
    let total: f64 = clipped.iter().sum();
    if total < 1e-12 {
        return equal;
    }
    clipped.iter().map(|w| w / total).collect()
}

/// Cap how far each weight may move from `prior_weights` in one rebalance
/// (post-hoc turnover control -- see module doc), then renormalize so the
/// capped weights still sum to 1.0. `max_change` is the maximum absolute
/// per-constituent weight delta allowed (e.g. 0.10 = at most a 10
/// percentage-point move per rebalance). `prior_weights` and `new_weights`
/// must be the same length; mismatched lengths (a constituent entered or
/// left the pool) return `new_weights` unchanged -- turnover capping only
/// applies to weights being RE-solved for the same fixed set of
/// constituents, not to pool composition changes.
pub fn cap_turnover(new_weights: &[f64], prior_weights: &[f64], max_change: f64) -> Vec<f64> {
    if new_weights.len() != prior_weights.len() || new_weights.is_empty() {
        return new_weights.to_vec();
    }
    let capped: Vec<f64> = new_weights.iter().zip(prior_weights.iter())
        .map(|(&w, &p)| (w - p).clamp(-max_change, max_change) + p)
        .map(|w| w.max(0.0))
        .collect();
    let total: f64 = capped.iter().sum();
    if total < 1e-12 {
        return prior_weights.to_vec();
    }
    capped.iter().map(|w| w / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_covariance_rejects_too_few_series_or_points() {
        assert!(sample_covariance(&[vec![1.0, 2.0]]).is_none());
        assert!(sample_covariance(&[vec![1.0], vec![2.0]]).is_none());
    }

    #[test]
    fn sample_covariance_of_identical_series_has_equal_diagonal_and_full_correlation() {
        let a = vec![0.01, -0.02, 0.03, 0.01, -0.01];
        let cov = sample_covariance(&[a.clone(), a]).unwrap();
        assert!((cov[0][0] - cov[1][1]).abs() < 1e-12);
        assert!((cov[0][1] - cov[0][0]).abs() < 1e-12, "identical series should have covariance == variance");
    }

    #[test]
    fn shrinkage_target_uses_average_correlation_and_sample_variances() {
        let a = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02];
        let b = vec![-0.01, 0.02, -0.02, 0.00, 0.01, -0.015];
        let sample = sample_covariance(&[a, b]).unwrap();
        let shrunk = shrink_toward_constant_correlation(&sample, 6);
        // Diagonal is a blend of (identical) sample and target diagonals,
        // so it should still equal the sample's own diagonal exactly.
        assert!((shrunk[0][0] - sample[0][0]).abs() < 1e-9);
        assert!((shrunk[1][1] - sample[1][1]).abs() < 1e-9);
    }

    #[test]
    fn shrinkage_intensity_increases_with_more_series_relative_to_data() {
        // With only 2 series, the constant-correlation TARGET trivially
        // equals the sample itself (a single pair's "average" correlation
        // is just that pair's own correlation), making shrinkage a no-op
        // regardless of intensity -- needs >= 3 series with HETEROGENEOUS
        // pairwise correlations for the target to actually differ from the
        // sample, which is what this test needs to be meaningful.
        let a = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02];
        let b = vec![0.02, -0.01, 0.025, 0.015, -0.02, 0.018]; // strongly correlated with a
        let c = vec![-0.01, 0.015, -0.005, 0.02, 0.01, -0.015]; // weakly/negatively correlated with a
        let sample = sample_covariance(&[a, b, c]).unwrap();
        let shrunk_short_history = shrink_toward_constant_correlation(&sample, 6);
        let shrunk_long_history = shrink_toward_constant_correlation(&sample, 5000);
        let dist = |m: &[Vec<f64>]| (m[0][1] - sample[0][1]).abs();
        assert!(dist(&shrunk_short_history) > dist(&shrunk_long_history),
            "shorter history should shrink further from the raw sample off-diagonal: short={}, long={}",
            dist(&shrunk_short_history), dist(&shrunk_long_history));
    }

    #[test]
    fn min_variance_favors_the_lower_variance_uncorrelated_constituent() {
        // Two uncorrelated series, "a" much calmer than "b" -- min-variance
        // should put substantially more weight on "a".
        let cov = vec![
            vec![0.0001, 0.0],
            vec![0.0, 0.01],
        ];
        let w = min_variance_weights(&cov);
        assert!(w[0] > w[1], "expected the calmer constituent to get more weight, got {:?}", w);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn min_variance_weights_are_never_negative() {
        // Strongly negatively-correlated pair can push the unconstrained
        // solve negative on one leg -- must clip to zero, not short it.
        let cov = vec![
            vec![0.0004, -0.00038],
            vec![-0.00038, 0.0004],
        ];
        let w = min_variance_weights(&cov);
        assert!(w.iter().all(|&x| x >= 0.0), "got {:?}", w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn min_variance_falls_back_to_equal_on_singular_matrix() {
        let cov = vec![
            vec![0.0001, 0.0001],
            vec![0.0001, 0.0001],
        ];
        let w = min_variance_weights(&cov);
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.5).abs() < 1e-9);
        assert!((w[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn turnover_cap_limits_the_move_and_still_sums_to_one() {
        let prior = vec![0.5, 0.5];
        let target = vec![0.9, 0.1];
        let capped = cap_turnover(&target, &prior, 0.10);
        assert!((capped[0] - 0.6).abs() < 1e-9, "got {:?}", capped);
        assert!((capped.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn turnover_cap_is_a_noop_within_the_allowed_band() {
        let prior = vec![0.4, 0.6];
        let target = vec![0.43, 0.57];
        let capped = cap_turnover(&target, &prior, 0.10);
        assert!((capped[0] - 0.43).abs() < 1e-6);
    }

    #[test]
    fn turnover_cap_passes_through_on_mismatched_lengths() {
        let prior = vec![1.0];
        let target = vec![0.5, 0.5];
        let capped = cap_turnover(&target, &prior, 0.10);
        assert_eq!(capped, target);
    }
}
