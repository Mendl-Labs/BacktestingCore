//! Exact comparison of a close with the arithmetic mean of a window of closes.
//!
//! The signal is `close > mean(window)`. Evaluating that in binary floating point can turn a real tie into
//! "above" (ten closes of 0.1 sum to 0.9999999999999999, so the naive mean is below 0.1). Here every f64 is
//! decomposed into `mantissa * 2^exponent` and the comparison `close * n  vs  sum` is done in integers on a
//! common binary scale, so a tie is exactly a tie. The reported mean is the exact mean rounded once to f64.

use std::cmp::Ordering;

/// Largest binary-exponent spread accepted between the smallest and largest value of a window.
const MAX_SHIFT: i32 = 64;

/// Decompose a positive finite f64 into (mantissa, exponent) with value = mantissa * 2^exponent.
fn decompose(x: f64) -> Option<(u64, i32)> {
    if !(x.is_finite() && x > 0.0) {
        return None;
    }
    let bits = x.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if exp_bits == 0 {
        Some((frac, -1074))
    } else {
        Some((frac | (1u64 << 52), exp_bits - 1075))
    }
}

/// `x * 2^e` without intermediate overflow/underflow for the exponents that occur here.
fn ldexp(x: f64, e: i32) -> f64 {
    let h = e / 2;
    x * 2f64.powi(h) * 2f64.powi(e - h)
}

/// Result of comparing a close with the mean of a window.
pub(crate) struct MeanComparison {
    /// `close` compared with the exact mean of `window`.
    pub ordering: Ordering,
    /// The exact mean rounded once to f64 (informational).
    pub mean: f64,
}

/// Compare `close` with the exact mean of `window`. `None` when a value is not a positive finite number, the
/// window is empty, or the values span too wide a scale for the integer arithmetic (caller refuses).
pub(crate) fn compare_to_mean(close: f64, window: &[f64]) -> Option<MeanComparison> {
    if window.is_empty() || window.len() > 4096 {
        return None;
    }
    let (cm, ce) = decompose(close)?;
    let mut parts = Vec::with_capacity(window.len());
    let mut emin = ce;
    for &x in window {
        let (m, e) = decompose(x)?;
        emin = emin.min(e);
        parts.push((m, e));
    }
    let mut sum: i128 = 0;
    for (m, e) in parts {
        let shift = e - emin;
        if shift > MAX_SHIFT {
            return None;
        }
        sum = sum.checked_add((m as i128) << shift)?;
    }
    let cshift = ce - emin;
    if cshift > MAX_SHIFT {
        return None;
    }
    let n = window.len() as i128;
    let close_n = ((cm as i128) << cshift).checked_mul(n)?;
    let mean = ldexp(sum as f64, emin) / window.len() as f64;
    Some(MeanComparison {
        ordering: close_n.cmp(&sum),
        mean,
    })
}

/// Result of comparing the means of two windows directly against each other (cross-multiplied, so neither mean is
/// rounded to f64 before the comparison -- see [`compare_means`]).
/// `mean_a`/`mean_b` are informational only (no production caller currently reads them, only this module's
/// own tests do), kept for parity with `MeanComparison::mean` and so a future caller can report both SMAs
/// without recomputing them.
#[allow(dead_code)]
pub(crate) struct TwoMeanComparison {
    /// `mean(window_a)` compared with `mean(window_b)`.
    pub ordering: Ordering,
    /// The exact mean of `window_a`, rounded once to f64 (informational).
    pub mean_a: f64,
    /// The exact mean of `window_b`, rounded once to f64 (informational).
    pub mean_b: f64,
}

/// Sum decomposed `(mantissa, exponent)` parts exactly as an `i128`, every part shifted onto the common scale
/// `emin` (so the result is an exact integer multiple of `2^emin`). `None` if any part's exponent spread from
/// `emin` exceeds [`MAX_SHIFT`], or on integer overflow.
fn exact_sum(parts: &[(u64, i32)], emin: i32) -> Option<i128> {
    let mut sum: i128 = 0;
    for &(m, e) in parts {
        let shift = e - emin;
        if shift > MAX_SHIFT {
            return None;
        }
        sum = sum.checked_add((m as i128) << shift)?;
    }
    Some(sum)
}

/// Compare `mean(window_a)` with `mean(window_b)` exactly, for two windows that may differ in length (the case
/// `dual_ma_crossover` needs: a fast SMA window vs a slower, longer one). Equivalent to comparing `sum_a / n_a`
/// with `sum_b / n_b`, done as the cross product `sum_a * n_b` vs `sum_b * n_a` in `i128` integer arithmetic on a
/// common binary scale shared by every value in EITHER window, so a tie between the two means is exactly a tie
/// (the same discipline [`compare_to_mean`] applies to a close vs one mean -- see its doc comment for why naive
/// floating-point summation can turn a real tie into a false signal). `None` when a value is not a positive finite
/// number, either window is empty or longer than 4096 bars, or the values span too wide a scale for the integer
/// arithmetic (caller refuses).
pub(crate) fn compare_means(window_a: &[f64], window_b: &[f64]) -> Option<TwoMeanComparison> {
    if window_a.is_empty() || window_a.len() > 4096 || window_b.is_empty() || window_b.len() > 4096
    {
        return None;
    }
    let mut parts_a = Vec::with_capacity(window_a.len());
    let mut parts_b = Vec::with_capacity(window_b.len());
    let mut emin = i32::MAX;
    for &x in window_a {
        let (m, e) = decompose(x)?;
        emin = emin.min(e);
        parts_a.push((m, e));
    }
    for &x in window_b {
        let (m, e) = decompose(x)?;
        emin = emin.min(e);
        parts_b.push((m, e));
    }
    let sum_a = exact_sum(&parts_a, emin)?;
    let sum_b = exact_sum(&parts_b, emin)?;
    let na = window_a.len() as i128;
    let nb = window_b.len() as i128;
    let lhs = sum_a.checked_mul(nb)?;
    let rhs = sum_b.checked_mul(na)?;
    let mean_a = ldexp(sum_a as f64, emin) / window_a.len() as f64;
    let mean_b = ldexp(sum_b as f64, emin) / window_b.len() as f64;
    Some(TwoMeanComparison {
        ordering: lhs.cmp(&rhs),
        mean_a,
        mean_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_mean_tie_is_a_tie_even_where_naive_float_division_is_not() {
        // window_a = three copies of the IEEE double 0.1; its naive f64 sum/3 rounds to 0.10000000000000002 (NOT
        // bit-identical to the double 0.1), yet the TRUE mean of three copies of that exact double is that double
        // again. window_b = a single copy of the same double. A naive f64-mean-then-compare would see
        // 0.10000000000000002 vs 0.1 and report "above"; the exact cross-multiply must report a tie.
        let a = [0.1f64; 3];
        let b = [0.1f64];
        let naive_mean_a: f64 = a.iter().sum::<f64>() / 3.0;
        assert_ne!(
            naive_mean_a.to_bits(),
            b[0].to_bits(),
            "fixture must expose the naive-sum rounding artifact"
        );
        let c = compare_means(&a, &b).unwrap();
        assert_eq!(c.ordering, Ordering::Equal);
    }

    #[test]
    fn two_mean_strictly_above_and_below() {
        let fast = [100.0, 100.0, 100.0]; // mean 100
        let slow = [50.0, 50.0, 50.0, 50.0]; // mean 50
        assert_eq!(
            compare_means(&fast, &slow).unwrap().ordering,
            Ordering::Greater
        );
        assert_eq!(
            compare_means(&slow, &fast).unwrap().ordering,
            Ordering::Less
        );
    }

    #[test]
    fn two_mean_rejects_non_positive_and_wide_scale() {
        assert!(compare_means(&[0.0], &[1.0]).is_none());
        assert!(compare_means(&[1.0], &[f64::NAN]).is_none());
        assert!(compare_means(&[1.0], &[]).is_none());
        assert!(compare_means(&[1e300], &[1e-300]).is_none());
    }

    #[test]
    fn two_mean_is_the_true_mean_of_each_window() {
        let a = [1.0, 2.0, 3.0, 4.0]; // mean 2.5
        let b = [10.0, 20.0]; // mean 15.0
        let c = compare_means(&a, &b).unwrap();
        assert_eq!(c.mean_a, 2.5);
        assert_eq!(c.mean_b, 15.0);
        assert_eq!(c.ordering, Ordering::Less);
    }

    #[test]
    fn tie_is_a_tie_even_where_naive_float_mean_is_not() {
        let w = [0.1f64; 10];
        let naive: f64 = w.iter().sum::<f64>() / 10.0;
        assert!(
            0.1 > naive,
            "the naive mean is below 0.1, so the naive rule would say 'above'"
        );
        let c = compare_to_mean(0.1, &w).unwrap();
        assert_eq!(c.ordering, Ordering::Equal);
    }

    #[test]
    fn strictly_above_and_below() {
        let w = [100.0, 100.0, 100.0, 100.0];
        assert_eq!(
            compare_to_mean(100.000001, &w).unwrap().ordering,
            Ordering::Greater
        );
        assert_eq!(
            compare_to_mean(99.999999, &w).unwrap().ordering,
            Ordering::Less
        );
    }

    #[test]
    fn rejects_non_positive_and_wide_scale() {
        assert!(compare_to_mean(0.0, &[1.0]).is_none());
        assert!(compare_to_mean(1.0, &[f64::NAN]).is_none());
        assert!(compare_to_mean(1.0, &[]).is_none());
        assert!(compare_to_mean(1e300, &[1e-300]).is_none());
    }

    #[test]
    fn mean_is_the_true_mean() {
        let w = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(compare_to_mean(2.5, &w).unwrap().mean, 2.5);
    }
}
