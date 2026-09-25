//! Small numeric helpers that make f64 arithmetic behave like the planner's exact decimals at the places where the two
//! meet: 8-decimal rounding, tie-preserving comparisons and order-independent sums. Everything here is IEEE-exact
//! (`+ - * /`, `floor`, `ceil`, comparisons, `total_cmp`), so the results are bit-identical on every platform.

/// Relative tolerance used wherever a limit or a threshold is compared: a limit is BREACHED only when the measured value
/// exceeds `cap * (1 + EDGE_TOL)`, a trade threshold DROPS a trade only when the change is below
/// `threshold * (1 - EDGE_TOL)`. Rationale: the live planner compares exact decimals, so a value that is exactly on the
/// boundary in decimal (`0.1 + 0.2` against `0.3`, a delta of exactly 10 units) is on the permitted side. The same value
/// computed in f64 can land one ulp on the other side. The tolerance is 1e-12 relative, i.e. thousands of ulps and
/// ten thousand times smaller than the 1e-9 boundary band the parity tests allow (design 5.4 test 1). Values inside the
/// band may still be classified differently from the Decimal planner; that is the documented boundary ambiguity.
pub const EDGE_TOL: f64 = 1e-12;

/// `10^dp` for the decimal places this crate uses (0..=18), from an exact table (no `powi`, whose result is not
/// guaranteed to be identical across platforms).
pub(crate) fn pow10(dp: u32) -> f64 {
    const T: [f64; 19] =
        [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18];
    T[dp.min(18) as usize]
}

/// Slack, in ulps, by which a scaled value may sit below an integer quantum and still count as that quantum.
const SNAP_ULPS: f64 = 4.0;

/// `floor(x)` at `dp` decimals for `x >= 0`. A value at most 4 ulps below a quantum boundary (the f64 spelling of
/// `0.3` or of `20000 * 0.1234567891234`) counts as being on it, as the exact decimal would.
pub fn floor_dp(x: f64, dp: u32) -> f64 {
    let s = pow10(dp);
    let y = x * s;
    (y + y * SNAP_ULPS * f64::EPSILON).floor() / s
}

/// `ceil(x)` at `dp` decimals for `x >= 0`, with the same snap in the other direction.
pub fn ceil_dp(x: f64, dp: u32) -> f64 {
    let s = pow10(dp);
    let y = x * s;
    (y - y * SNAP_ULPS * f64::EPSILON).ceil() / s
}

/// Round toward ZERO at `dp` decimals: for a long it is the usual floor, for a short it keeps the short from being
/// made bigger by rounding (a plain floor would round a negative value away from zero). This is the planner's
/// `round_toward_zero` (8 decimals) on f64.
pub fn round_toward_zero_dp(x: f64, dp: u32) -> f64 {
    if x < 0.0 {
        -floor_dp(-x, dp)
    } else {
        floor_dp(x, dp)
    }
}

/// Sum that does not depend on the order of the terms: sorted by `total_cmp`, then added from the smallest. Used for
/// every cross-instrument aggregate (gross, net, class gross, margin) so that permuting the instruments cannot move a
/// bit, and so that a refusal never depends on how the caller happened to list them.
pub fn stable_sum(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mut s = 0.0;
    for x in v {
        s += x;
    }
    s
}

/// `value > cap`, tolerant: true only when `value` exceeds `cap` by more than [`EDGE_TOL`] relative. An infinite cap
/// is never exceeded.
pub fn exceeds(value: f64, cap: f64) -> bool {
    if cap.is_infinite() {
        return false;
    }
    value > cap + cap.abs() * EDGE_TOL
}

/// `value < threshold`, tolerant: true only when `value` is below `threshold` by more than [`EDGE_TOL`] relative.
pub fn below(value: f64, threshold: f64) -> bool {
    value < threshold - threshold.abs() * EDGE_TOL
}
