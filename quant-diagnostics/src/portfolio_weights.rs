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

/// Standard textbook default risk-aversion coefficient (Black & Litterman's
/// own 1992 worked example uses ~2.5). A platform-wide constant, never
/// agent-chosen: letting an agent pick among optimization techniques, or
/// tune either technique's own internal knobs, would reintroduce exactly
/// the data-snooping-one-layer-up risk the "fixed, stated decision rule"
/// design principle (2026-09 quant council) exists to prevent.
pub const BLACK_LITTERMAN_RISK_AVERSION: f64 = 2.5;
/// Standard practitioner default for `tau`, the scalar controlling how much
/// weight the PRIOR (equilibrium) return estimate itself carries relative
/// to the views (He & Litterman 1999 and most implementations use a small
/// value in the 0.01-0.05 range). Also a fixed constant, not agent-chosen.
pub const BLACK_LITTERMAN_TAU: f64 = 0.025;

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()).collect()
}

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

/// Black-Litterman posterior weights (breadth Phase 3, 2026-09 quant
/// council): blends an equilibrium prior implied by the portfolio's own
/// current/reference allocation with per-constituent "views" -- for this
/// platform's meta-portfolio (a portfolio of already-validated STRATEGIES,
/// not raw assets), the view for constituent `i` is that strategy's own
/// backtest-derived expected return and the view's uncertainty is that same
/// strategy's own statistical uncertainty (e.g. Sharpe standard error
/// scaled to a return), NEVER an agent-supplied number -- consistent with
/// the "no agent discretion among optimization techniques or their inputs"
/// design principle. There is no true "market portfolio" for a set of
/// strategies the way there is for public equities, so `prior_weights` is
/// the portfolio's own last-known (or equal, if none yet) allocation --
/// reverse-optimizing THAT into an implied prior return is the standard,
/// principled Black-Litterman substitute for a market-cap prior when one
/// doesn't exist.
///
/// - `cov`: N x N covariance matrix of constituent returns.
/// - `prior_weights`: the reference allocation to reverse-optimize into an
///   implied equilibrium return (`pi = risk_aversion * cov * prior_weights`).
/// - `view_returns[i]`: constituent `i`'s own stated absolute expected
///   return (an absolute view on every constituent, i.e. the view "pick"
///   matrix P is implicitly the identity -- the simplest, most common BL
///   application, and the only one that needs no additional agent-supplied
///   structure).
/// - `view_uncertainty[i]`: constituent `i`'s own view variance (larger =
///   less confident in that view, pulling the posterior further toward the
///   prior for that constituent). Clamped to a small positive floor so a
///   caller passing 0.0 (perfect confidence) can never divide by zero or
///   produce an infinitely-confident view.
/// - `risk_aversion` / `tau`: pass `BLACK_LITTERMAN_RISK_AVERSION` /
///   `BLACK_LITTERMAN_TAU` unless a caller has a specific, non-agent-chosen
///   reason to override (parameterized for testability).
///
/// Returns `None` on mismatched input lengths or a singular covariance
/// matrix -- callers decide their own fallback (e.g. `min_variance_weights`
/// or equal weight), mirroring `unconstrained_min_variance_weights`'s own
/// contract rather than silently substituting one here.
pub fn black_litterman_weights(
    cov: &[Vec<f64>],
    prior_weights: &[f64],
    view_returns: &[f64],
    view_uncertainty: &[f64],
    risk_aversion: f64,
    tau: f64,
) -> Option<Vec<f64>> {
    let n = cov.len();
    if n == 0
        || prior_weights.len() != n
        || view_returns.len() != n
        || view_uncertainty.len() != n
        || cov.iter().any(|row| row.len() != n)
    {
        return None;
    }

    // Implied equilibrium excess returns from reverse-optimizing the prior
    // allocation: pi = delta * Sigma * w_prior.
    let pi: Vec<f64> = mat_vec_mul(cov, prior_weights).iter().map(|x| x * risk_aversion).collect();

    let tau_cov: Vec<Vec<f64>> = cov.iter().map(|row| row.iter().map(|&v| v * tau).collect()).collect();
    let tau_cov_inv = invert_matrix(&tau_cov)?;

    // P = identity (one absolute view per constituent), so P' * Omega^-1 * P
    // reduces to a diagonal matrix of 1/omega_i.
    let omega_inv_diag: Vec<f64> = view_uncertainty.iter().map(|&u| 1.0 / u.max(1e-10)).collect();

    let mut precision = tau_cov_inv.clone();
    for i in 0..n {
        precision[i][i] += omega_inv_diag[i];
    }
    let posterior_cov = invert_matrix(&precision)?;

    let tau_cov_inv_pi = mat_vec_mul(&tau_cov_inv, &pi);
    let rhs: Vec<f64> = (0..n).map(|i| tau_cov_inv_pi[i] + omega_inv_diag[i] * view_returns[i]).collect();
    let mu_bl = mat_vec_mul(&posterior_cov, &rhs);

    // Unconstrained mean-variance weights from the posterior returns:
    // w = (delta * Sigma)^-1 * mu_BL.
    let cov_inv = invert_matrix(cov)?;
    let raw: Vec<f64> = mat_vec_mul(&cov_inv, &mu_bl).iter().map(|x| x / risk_aversion).collect();

    // Long-only projection + renormalize, same convention as
    // min_variance_weights (see module doc for why this is a documented
    // approximation of the exact non-negativity-constrained QP).
    let clipped: Vec<f64> = raw.iter().map(|w| w.max(0.0)).collect();
    let total: f64 = clipped.iter().sum();
    if total < 1e-12 {
        return Some(vec![1.0 / n as f64; n]);
    }
    Some(clipped.iter().map(|w| w / total).collect())
}

/// Hierarchical Risk Parity (López de Prado 2016) -- breadth Phase 3, 2026-09
/// quant council, alongside `black_litterman_weights`. Unlike
/// `min_variance_weights` (a single closed-form solve over the FULL
/// covariance matrix, sensitive to inversion error on a near-singular or
/// heavily correlated matrix) HRP never inverts the covariance matrix at
/// all: it clusters constituents by correlation structure, orders them so
/// similar constituents sit adjacent (quasi-diagonalization), then
/// recursively bisects that ordering top-down, splitting capital between
/// each half in inverse proportion to the half's own cluster variance. This
/// is the standard argument for HRP as a THIRD option (not a replacement)
/// alongside min-variance and Black-Litterman: it stays well-behaved when
/// the correlation matrix is ill-conditioned (many highly-correlated
/// constituents, or more constituents than return observations), at the
/// cost of being a heuristic rather than a provably-optimal solve the way
/// min-variance is under its own assumptions. Naturally long-only (no
/// projection/clipping step needed, unlike the other two techniques here).
///
/// Two documented simplifications, matching this module's own established
/// convention (see module doc):
/// 1. **Single-linkage clustering** via a straightforward O(n^3) nearest-
///    cluster scan -- the textbook choice for HRP and adequate for the
///    small constituent counts this is sized for (see
///    `MAX_CANDIDATE_POOL`); a large-n optimized implementation (e.g.
///    SciPy's `scipy.cluster.hierarchy.linkage`) is unnecessary machinery
///    for a v1.
/// 2. **Correlation-based distance** `d(i,j) = sqrt(0.5*(1 - corr(i,j)))`,
///    the standard HRP distance metric (bounded in [0, 1], zero for
///    perfectly correlated pairs) -- not a distance-of-distances matrix
///    (López de Prado's own optional refinement), which adds clustering
///    stability at the cost of real complexity for a marginal v1 benefit.
pub fn hrp_weights(cov: &[Vec<f64>]) -> Vec<f64> {
    let n = cov.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let sigmas: Vec<f64> = (0..n).map(|i| cov[i][i].max(0.0).sqrt()).collect();
    let mut distance = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let corr = if sigmas[i] > 1e-12 && sigmas[j] > 1e-12 {
                (cov[i][j] / (sigmas[i] * sigmas[j])).clamp(-1.0, 1.0)
            } else if i == j {
                1.0
            } else {
                0.0
            };
            distance[i][j] = (0.5 * (1.0 - corr)).max(0.0).sqrt();
        }
    }

    let order = quasi_diagonal_order(&distance);

    let mut weights = vec![1.0; n];
    let mut clusters: Vec<Vec<usize>> = vec![order];
    while clusters.iter().any(|c| c.len() > 1) {
        let mut next_clusters = Vec::with_capacity(clusters.len() * 2);
        for cluster in clusters {
            if cluster.len() <= 1 {
                next_clusters.push(cluster);
                continue;
            }
            let mid = cluster.len() / 2;
            let left = cluster[..mid].to_vec();
            let right = cluster[mid..].to_vec();
            let var_left = cluster_variance(cov, &left);
            var_split_weights(&mut weights, &left, &right, var_left, cluster_variance(cov, &right));
            next_clusters.push(left);
            next_clusters.push(right);
        }
        clusters = next_clusters;
    }
    weights
}

/// Applies one HRP bisection split: capital is allocated between `left` and
/// `right` in INVERSE proportion to each side's own cluster variance (the
/// lower-variance side gets the larger share), multiplied into whatever
/// share each side's constituents already carry from earlier (coarser)
/// splits -- this is what makes the recursion produce a genuine top-down
/// allocation rather than resetting at each level.
fn var_split_weights(weights: &mut [f64], left: &[usize], right: &[usize], var_left: f64, var_right: f64) {
    let total_var = var_left + var_right;
    let alloc_left = if total_var > 1e-18 { 1.0 - var_left / total_var } else { 0.5 };
    let alloc_right = 1.0 - alloc_left;
    for &i in left {
        weights[i] *= alloc_left;
    }
    for &i in right {
        weights[i] *= alloc_right;
    }
}

/// Inverse-variance-weighted variance of a cluster (López de Prado's own
/// `cluster_var`): the cluster's members are weighted by their own inverse
/// variance (`1/cov[i][i]`, normalized to sum to 1) -- a cheap, diagonal-
/// only proxy for "this cluster's own internal minimum-variance portfolio,"
/// avoiding a second matrix inversion inside the recursion -- then that
/// weight vector's quadratic form `w' Cov w` against the FULL covariance
/// (including cross-terms within the cluster) is the cluster's variance.
fn cluster_variance(cov: &[Vec<f64>], members: &[usize]) -> f64 {
    let inv_var: Vec<f64> = members.iter().map(|&i| 1.0 / cov[i][i].max(1e-12)).collect();
    let total: f64 = inv_var.iter().sum();
    let w: Vec<f64> = if total > 1e-18 {
        inv_var.iter().map(|v| v / total).collect()
    } else {
        vec![1.0 / members.len() as f64; members.len()]
    };
    let mut var = 0.0;
    for (a, &i) in members.iter().enumerate() {
        for (b, &j) in members.iter().enumerate() {
            var += w[a] * w[b] * cov[i][j];
        }
    }
    var.max(0.0)
}

/// Single-linkage agglomerative clustering, returning the leaf order from
/// an in-order traversal of the resulting dendrogram (the "quasi-diagonal"
/// order -- adjacent indices in the returned `Vec` are the most similar
/// pairs/clusters, exactly what the recursive bisection needs to split
/// along real cluster boundaries rather than an arbitrary index order).
fn quasi_diagonal_order(distance: &[Vec<f64>]) -> Vec<usize> {
    let n = distance.len();
    // Each active cluster: its own leaf-order-preserving member list (for
    // the final traversal) alongside the same members again (for computing
    // single-linkage distance to every other active cluster).
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > 1 {
        let mut best = (0usize, 1usize, f64::INFINITY);
        for a in 0..clusters.len() {
            for b in (a + 1)..clusters.len() {
                let d = clusters[a]
                    .iter()
                    .flat_map(|&i| clusters[b].iter().map(move |&j| distance[i][j]))
                    .fold(f64::INFINITY, f64::min);
                if d < best.2 {
                    best = (a, b, d);
                }
            }
        }
        let (a, b, _) = best;
        // Remove the higher index first so the lower index's position
        // (and thus `a`'s slot) is never invalidated by the first removal.
        let members_b = clusters.remove(b);
        let mut members_a = clusters.remove(a);
        members_a.extend(members_b);
        clusters.push(members_a);
    }

    clusters.into_iter().next().unwrap_or_default()
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

    // ── black_litterman_weights ────────────────────────────────────

    #[test]
    fn black_litterman_returns_none_on_mismatched_lengths_or_singular_cov() {
        let cov = vec![vec![0.0001, 0.0], vec![0.0, 0.0001]];
        assert!(black_litterman_weights(&cov, &[0.5], &[0.1, 0.1], &[0.01, 0.01], 2.5, 0.025).is_none());
        assert!(black_litterman_weights(&cov, &[0.5, 0.5], &[0.1], &[0.01, 0.01], 2.5, 0.025).is_none());
        assert!(black_litterman_weights(&cov, &[0.5, 0.5], &[0.1, 0.1], &[0.01], 2.5, 0.025).is_none());
        let singular = vec![vec![0.0001, 0.0001], vec![0.0001, 0.0001]];
        assert!(black_litterman_weights(&singular, &[0.5, 0.5], &[0.1, 0.1], &[0.01, 0.01], 2.5, 0.025).is_none());
    }

    #[test]
    fn black_litterman_with_infinite_view_uncertainty_reduces_to_the_prior_weights() {
        // When Omega -> infinity (no confidence in any view), mu_BL -> pi
        // exactly, and the mean-variance solve on pi alone reproduces
        // prior_weights exactly: w = (delta*Sigma)^-1 * (delta*Sigma*w_prior) / delta = w_prior.
        let cov = vec![
            vec![0.0004, 0.00005],
            vec![0.00005, 0.0009],
        ];
        let prior = vec![0.3, 0.7];
        // View returns are deliberately wild/wrong -- they must not move the
        // result at all given near-infinite uncertainty.
        let wild_views = vec![5.0, -5.0];
        let huge_uncertainty = vec![1e12, 1e12];
        let w = black_litterman_weights(&cov, &prior, &wild_views, &huge_uncertainty, 2.5, 0.025).unwrap();
        assert!((w[0] - 0.3).abs() < 1e-6, "got {:?}", w);
        assert!((w[1] - 0.7).abs() < 1e-6, "got {:?}", w);
    }

    #[test]
    fn black_litterman_confident_view_shifts_weight_toward_the_favored_constituent() {
        // Two uncorrelated, equal-variance constituents starting from an
        // equal prior. A strong, confident (low-uncertainty) view that
        // constituent 0 has a much higher expected return than constituent 1
        // must shift the posterior weight toward constituent 0.
        let cov = vec![
            vec![0.0004, 0.0],
            vec![0.0, 0.0004],
        ];
        let prior = vec![0.5, 0.5];
        let views = vec![0.20, 0.02]; // constituent 0 stated far more promising
        let confident = vec![1e-6, 1e-6]; // both views held with high confidence
        let w = black_litterman_weights(&cov, &prior, &views, &confident, 2.5, 0.025).unwrap();
        assert!(w[0] > w[1], "expected the favored constituent to get more weight, got {:?}", w);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn black_litterman_symmetric_inputs_produce_equal_weights() {
        let cov = vec![
            vec![0.0004, 0.0],
            vec![0.0, 0.0004],
        ];
        let w = black_litterman_weights(&cov, &[0.5, 0.5], &[0.1, 0.1], &[0.02, 0.02], 2.5, 0.025).unwrap();
        assert!((w[0] - 0.5).abs() < 1e-9, "got {:?}", w);
        assert!((w[1] - 0.5).abs() < 1e-9, "got {:?}", w);
    }

    #[test]
    fn black_litterman_weights_are_never_negative_and_sum_to_one() {
        // A strongly negative view on one leg combined with high correlation
        // can push the unconstrained solve negative -- must clip, not short.
        let cov = vec![
            vec![0.0004, 0.00038],
            vec![0.00038, 0.0004],
        ];
        let w = black_litterman_weights(&cov, &[0.5, 0.5], &[-0.3, 0.3], &[1e-6, 1e-6], 2.5, 0.025).unwrap();
        assert!(w.iter().all(|&x| x >= 0.0), "got {:?}", w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    // ── hrp_weights ─────────────────────────────────────────────────

    #[test]
    fn hrp_weights_handles_degenerate_sizes() {
        assert_eq!(hrp_weights(&[]), Vec::<f64>::new());
        assert_eq!(hrp_weights(&[vec![0.0004]]), vec![1.0]);
    }

    #[test]
    fn hrp_two_assets_reduces_to_the_classic_inverse_variance_formula() {
        // Analytically-derived invariant: with exactly 2 (singleton)
        // clusters, cluster_variance of a single-member cluster is just
        // that member's own variance (inverse-variance-normalized weight
        // on ONE member is trivially 1.0) -- so HRP's single bisection is
        // exactly the textbook 2-asset inverse-variance-parity formula:
        // w_i = (1/var_i) / sum(1/var_j).
        let var_a = 0.0001;
        let var_b = 0.0009;
        let cov = vec![
            vec![var_a, 0.0],
            vec![0.0, var_b],
        ];
        let w = hrp_weights(&cov);
        let expected_a = (1.0 / var_a) / (1.0 / var_a + 1.0 / var_b);
        assert!((w[0] - expected_a).abs() < 1e-9, "got {:?}, expected w[0]={}", w, expected_a);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hrp_favors_the_lower_variance_uncorrelated_constituent() {
        let cov = vec![
            vec![0.0001, 0.0],
            vec![0.0, 0.01],
        ];
        let w = hrp_weights(&cov);
        assert!(w[0] > w[1], "expected the calmer constituent to get more weight, got {:?}", w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hrp_weights_are_always_non_negative_and_sum_to_one_for_a_realistic_basket() {
        // 4 assets, mixed correlation structure and variances -- a case
        // real enough to exercise more than one level of the recursion.
        let cov = vec![
            vec![0.0004, 0.00030, 0.00005, -0.00002],
            vec![0.00030, 0.0004, 0.00003, -0.00001],
            vec![0.00005, 0.00003, 0.0009, 0.0002],
            vec![-0.00002, -0.00001, 0.0002, 0.0009],
        ];
        let w = hrp_weights(&cov);
        assert_eq!(w.len(), 4);
        assert!(w.iter().all(|&x| x >= 0.0), "HRP is naturally long-only, got {:?}", w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9, "got {:?}", w);
    }

    #[test]
    fn quasi_diagonal_order_places_the_most_correlated_pair_adjacent() {
        // Assets 0 and 1 are near-identical (distance ~0); asset 2 is
        // uncorrelated with both (distance ~1 = sqrt(0.5)). The returned
        // order must place 0 and 1 next to each other, wherever 2 falls.
        let distance = vec![
            vec![0.0, 0.02, 0.70],
            vec![0.02, 0.0, 0.70],
            vec![0.70, 0.70, 0.0],
        ];
        let order = quasi_diagonal_order(&distance);
        assert_eq!(order.len(), 3);
        let pos0 = order.iter().position(|&x| x == 0).unwrap();
        let pos1 = order.iter().position(|&x| x == 1).unwrap();
        assert_eq!((pos0 as i64 - pos1 as i64).abs(), 1, "0 and 1 must be adjacent in {:?}", order);
    }

    #[test]
    fn cluster_variance_of_a_singleton_is_its_own_variance() {
        let cov = vec![
            vec![0.0004, 0.0001],
            vec![0.0001, 0.0009],
        ];
        assert!((cluster_variance(&cov, &[0]) - 0.0004).abs() < 1e-12);
        assert!((cluster_variance(&cov, &[1]) - 0.0009).abs() < 1e-12);
    }

    #[test]
    fn var_split_weights_gives_more_capital_to_the_lower_variance_side() {
        let mut weights = vec![1.0, 1.0];
        var_split_weights(&mut weights, &[0], &[1], 0.0001, 0.0009);
        assert!(weights[0] > weights[1], "got {:?}", weights);
        assert!((weights[0] + weights[1] - 1.0).abs() < 1e-9);
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
