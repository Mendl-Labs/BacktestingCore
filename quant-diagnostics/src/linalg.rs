//! Small, hand-rolled linear algebra helpers shared by `cointegration.rs`
//! (pairwise Engle-Granger, via `invert_matrix`) and `johansen.rs` (N-way
//! cointegration, via Cholesky + the Jacobi eigenvalue algorithm). Matrices
//! are plain `Vec<Vec<f64>>`, row-major, square unless noted -- no external
//! numerical dependency, matching this crate's existing "hand-rolled
//! approximation, dependency-light" convention. Sized for the handful of
//! symbols a stat-arb basket realistically has (think low single digits to
//! maybe two dozen), not for general-purpose numerical computing -- none of
//! this is optimized for large N.

/// Invert a square matrix via Gauss-Jordan elimination with partial
/// pivoting. Returns `None` for a singular/near-singular matrix. General
/// (not assumed symmetric).
pub fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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
            return None; // singular / near-singular
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

/// Matrix multiply `a (m x k) * b (k x n) -> (m x n)`.
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if m > 0 { a[0].len() } else { 0 };
    let n = if !b.is_empty() { b[0].len() } else { 0 };
    let mut out = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i][t] * b[t][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Transpose a matrix.
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Cholesky decomposition of a symmetric positive-definite matrix `a`:
/// returns lower-triangular `l` such that `a = l * l^T`. Returns `None` if
/// `a` isn't positive definite (a non-positive value would need to be
/// square-rooted on the diagonal).
pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Invert a lower-triangular matrix via forward substitution. Returns
/// `None` if any diagonal entry is (near) zero.
pub fn invert_lower_triangular(l: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = l.len();
    let mut inv = vec![vec![0.0; n]; n];
    for col in 0..n {
        if l[col][col].abs() < 1e-12 {
            return None;
        }
        inv[col][col] = 1.0 / l[col][col];
        for row in (col + 1)..n {
            let mut sum = 0.0;
            for k in col..row {
                sum += l[row][k] * inv[k][col];
            }
            if l[row][row].abs() < 1e-12 {
                return None;
            }
            inv[row][col] = -sum / l[row][row];
        }
    }
    Some(inv)
}

/// Classical cyclic Jacobi eigenvalue algorithm for a symmetric matrix.
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is the i-th
/// eigenvector (as a `Vec<f64>`), both sorted by eigenvalue descending.
/// Converges for any real symmetric matrix; `max_sweeps` bounds the
/// iteration count (a basket-sized N x N matrix converges in a handful of
/// sweeps in practice -- this is a safety cap, not a tuning knob callers
/// need to think about).
pub fn jacobi_eigen(a: &[Vec<f64>], max_sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    if n <= 1 {
        let eigenvalues = if n == 1 { vec![m[0][0]] } else { vec![] };
        let eigenvectors = if n == 1 { vec![vec![1.0]] } else { vec![] };
        return (eigenvalues, eigenvectors);
    }

    for _ in 0..max_sweeps {
        let mut off_diag_norm = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_norm += m[p][q] * m[p][q];
            }
        }
        if off_diag_norm.sqrt() < 1e-12 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = if (m[p][p] - m[q][q]).abs() < 1e-15 {
                    std::f64::consts::FRAC_PI_4 * m[p][q].signum()
                } else {
                    0.5 * (2.0 * m[p][q] / (m[q][q] - m[p][p])).atan()
                };
                let c = theta.cos();
                let s = theta.sin();

                // Apply rotation to M: M' = J^T M J, only rows/cols p,q change.
                for i in 0..n {
                    let m_ip = m[i][p];
                    let m_iq = m[i][q];
                    m[i][p] = c * m_ip - s * m_iq;
                    m[i][q] = s * m_ip + c * m_iq;
                }
                for i in 0..n {
                    let m_pi = m[p][i];
                    let m_qi = m[q][i];
                    m[p][i] = c * m_pi - s * m_qi;
                    m[q][i] = s * m_pi + c * m_qi;
                }

                // Accumulate eigenvectors: V' = V J.
                for i in 0..n {
                    let v_ip = v[i][p];
                    let v_iq = v[i][q];
                    v[i][p] = c * v_ip - s * v_iq;
                    v[i][q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
    let mut eigenvectors: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| v[j][i]).collect()).collect();

    // Sort descending by eigenvalue.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_values: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_vectors: Vec<Vec<f64>> = order.iter().map(|&i| eigenvectors[i].clone()).collect();
    eigenvalues = sorted_values;
    eigenvectors = sorted_vectors;

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
    }

    #[test]
    fn invert_matrix_recovers_identity_for_itself() {
        let inv = invert_matrix(&identity(3)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn invert_matrix_returns_none_for_singular() {
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert_matrix(&singular).is_none());
    }

    #[test]
    fn invert_matrix_matches_hand_computed_2x2() {
        // [[4,7],[2,6]]^-1 = 1/10 * [[6,-7],[-2,4]]
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_matrix(&a).unwrap();
        assert!((inv[0][0] - 0.6).abs() < 1e-9);
        assert!((inv[0][1] - (-0.7)).abs() < 1e-9);
        assert!((inv[1][0] - (-0.2)).abs() < 1e-9);
        assert!((inv[1][1] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn matmul_identity_is_a_noop() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matmul(&a, &identity(2));
        assert_eq!(result, a);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = transpose(&a);
        assert_eq!(t, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }

    #[test]
    fn cholesky_reconstructs_the_original_matrix() {
        // A well-known SPD matrix.
        let a = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        let l = cholesky(&a).unwrap();
        let reconstructed = matmul(&l, &transpose(&l));
        for i in 0..3 {
            for j in 0..3 {
                assert!((reconstructed[i][j] - a[i][j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn cholesky_returns_none_for_non_positive_definite() {
        let not_pd = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // eigenvalues -1, 3
        assert!(cholesky(&not_pd).is_none());
    }

    #[test]
    fn invert_lower_triangular_recovers_identity() {
        let l = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        let inv = invert_lower_triangular(&l).unwrap();
        let product = matmul(&l, &inv);
        assert!((product[0][0] - 1.0).abs() < 1e-9);
        assert!((product[0][1] - 0.0).abs() < 1e-9);
        assert!((product[1][0] - 0.0).abs() < 1e-9);
        assert!((product[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jacobi_eigen_recovers_diagonal_matrix_eigenvalues() {
        let a = vec![
            vec![5.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (values, _vectors) = jacobi_eigen(&a, 100);
        assert!((values[0] - 5.0).abs() < 1e-9);
        assert!((values[1] - 3.0).abs() < 1e-9);
        assert!((values[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jacobi_eigen_matches_known_2x2_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1.
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (values, _vectors) = jacobi_eigen(&a, 100);
        assert!((values[0] - 3.0).abs() < 1e-9);
        assert!((values[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jacobi_eigen_eigenvectors_satisfy_av_eq_lambda_v() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (values, vectors) = jacobi_eigen(&a, 100);
        for i in 0..2 {
            let v = &vectors[i];
            // A*v
            let av: Vec<f64> = (0..2).map(|r| a[r][0] * v[0] + a[r][1] * v[1]).collect();
            let lv: Vec<f64> = v.iter().map(|x| x * values[i]).collect();
            for k in 0..2 {
                assert!((av[k] - lv[k]).abs() < 1e-6, "A*v != lambda*v for eigenpair {}", i);
            }
        }
    }

    #[test]
    fn jacobi_eigen_handles_1x1_matrix() {
        let a = vec![vec![7.0]];
        let (values, vectors) = jacobi_eigen(&a, 10);
        assert_eq!(values, vec![7.0]);
        assert_eq!(vectors, vec![vec![1.0]]);
    }
}
