//! Basic sample statistics with typed errors. Conventions (fixed, and tested):
//!
//! * variance and standard deviation use `ddof = 1` (the same as `weightsim::metrics` and the Sharpe ratio of the key);
//! * a Sharpe ratio is `mean / std` per period and `x sqrt(periods_per_year)` annualised, no risk-free rate (callers pass
//!   excess returns);
//! * skewness and kurtosis are the population (`1/n`) standardised moments used by Bailey and Lopez de Prado (2014);
//!   kurtosis is NOT excess (a normal has 3);
//! * a series is "degenerate" when its standard deviation is at most `1e-14` of its largest absolute value, which
//!   catches constant series whose two-pass variance is rounding noise.

use crate::error::{invalid, EvalError, Result};

/// Refuse NaN / infinite values and too-short series.
pub fn check_series(what: &'static str, x: &[f64], min_len: usize) -> Result<()> {
    if x.len() < min_len {
        return Err(EvalError::TooShort { what, need: min_len, got: x.len() });
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(EvalError::NonFinite { what, index });
    }
    Ok(())
}

pub(crate) fn check_ppy(ppy: f64) -> Result<()> {
    if !(ppy.is_finite() && ppy > 0.0) {
        return Err(invalid("periods_per_year", format!("must be finite and positive, got {ppy}")));
    }
    Ok(())
}

/// Arithmetic mean.
pub fn mean(x: &[f64]) -> Result<f64> {
    check_series("mean input", x, 1)?;
    Ok(mean_unchecked(x))
}

pub(crate) fn mean_unchecked(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in x {
        s += *v;
    }
    s / x.len() as f64
}

/// Sample variance, `ddof = 1`.
pub fn variance(x: &[f64]) -> Result<f64> {
    check_series("variance input", x, 2)?;
    Ok(variance_unchecked(x, mean_unchecked(x)))
}

pub(crate) fn variance_unchecked(x: &[f64], m: f64) -> f64 {
    let mut ss = 0.0;
    for v in x {
        let d = *v - m;
        ss += d * d;
    }
    ss / (x.len() - 1) as f64
}

/// Sample standard deviation, `ddof = 1`.
pub fn std_dev(x: &[f64]) -> Result<f64> {
    Ok(variance(x)?.sqrt())
}

pub(crate) fn is_degenerate(x: &[f64], std: f64) -> bool {
    let mut max_abs = 0.0_f64;
    for v in x {
        max_abs = max_abs.max(v.abs());
    }
    std <= 1e-14 * max_abs
}

/// Per-period Sharpe ratio `mean / std`. `ZeroVariance` for a constant series.
pub fn sharpe_per_period(x: &[f64]) -> Result<f64> {
    check_series("sharpe input", x, 2)?;
    let m = mean_unchecked(x);
    let sd = variance_unchecked(x, m).sqrt();
    if is_degenerate(x, sd) {
        return Err(EvalError::ZeroVariance { what: "sharpe input" });
    }
    Ok(m / sd)
}

/// Annualised Sharpe ratio `mean / std x sqrt(ppy)`.
pub fn sharpe_annual(x: &[f64], periods_per_year: f64) -> Result<f64> {
    check_ppy(periods_per_year)?;
    Ok(sharpe_per_period(x)? * periods_per_year.sqrt())
}

/// Population skewness `m3 / m2^{3/2}` (Bailey-Lopez de Prado convention). Needs `n >= 3` and non-zero variance.
pub fn skewness(x: &[f64]) -> Result<f64> {
    check_series("skewness input", x, 3)?;
    let (m2, m3, _) = central_moments(x);
    if is_degenerate(x, m2.sqrt()) {
        return Err(EvalError::ZeroVariance { what: "skewness input" });
    }
    Ok(m3 / (m2 * m2.sqrt()))
}

/// Population kurtosis `m4 / m2^2` (NOT excess; a normal has 3). Needs `n >= 4` and non-zero variance.
pub fn kurtosis(x: &[f64]) -> Result<f64> {
    check_series("kurtosis input", x, 4)?;
    let (m2, _, m4) = central_moments(x);
    if is_degenerate(x, m2.sqrt()) {
        return Err(EvalError::ZeroVariance { what: "kurtosis input" });
    }
    Ok(m4 / (m2 * m2))
}

fn central_moments(x: &[f64]) -> (f64, f64, f64) {
    let m = mean_unchecked(x);
    let (mut a2, mut a3, mut a4) = (0.0, 0.0, 0.0);
    for v in x {
        let d = *v - m;
        let d2 = d * d;
        a2 += d2;
        a3 += d2 * d;
        a4 += d2 * d2;
    }
    let n = x.len() as f64;
    (a2 / n, a3 / n, a4 / n)
}

/// Sample autocovariance at `lag` with the `1/n` normalisation (biased, positive semi-definite), about the sample mean.
pub fn autocovariance(x: &[f64], lag: usize) -> Result<f64> {
    check_series("autocovariance input", x, 2)?;
    if lag >= x.len() {
        return Err(invalid("lag", format!("lag {lag} must be below the series length {}", x.len())));
    }
    let m = mean_unchecked(x);
    let mut s = 0.0;
    for t in lag..x.len() {
        s += (x[t] - m) * (x[t - lag] - m);
    }
    Ok(s / x.len() as f64)
}

/// Pearson correlation.
pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(EvalError::LengthMismatch { what: "correlation", left: x.len(), right: y.len() });
    }
    check_series("correlation x", x, 2)?;
    check_series("correlation y", y, 2)?;
    let (mx, my) = (mean_unchecked(x), mean_unchecked(y));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let sdx = (sxx / (x.len() - 1) as f64).sqrt();
    let sdy = (syy / (y.len() - 1) as f64).sqrt();
    if is_degenerate(x, sdx) {
        return Err(EvalError::ZeroVariance { what: "correlation x" });
    }
    if is_degenerate(y, sdy) {
        return Err(EvalError::ZeroVariance { what: "correlation y" });
    }
    Ok(sxy / (sxx.sqrt() * syy.sqrt()))
}

/// Quantile of an ascending-sorted slice by linear interpolation (Hyndman-Fan type 7, the R and numpy default).
/// `q` is clamped to `[0, 1]`; an empty slice gives `NaN`.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let h = q * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Median (type-7 quantile at 0.5).
pub fn median(x: &[f64]) -> Result<f64> {
    check_series("median input", x, 1)?;
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(quantile_sorted(&v, 0.5))
}

/// Normal-consistent robust scale: `1.4826 x median(|x - median(x)|)` (MAD). Used for trial dispersion because plain
/// standard deviations were inflated by blown-up configurations (7.47 versus a 1.27 null bar, memory 2026-09-18).
pub fn mad_scale(x: &[f64]) -> Result<f64> {
    check_series("mad input", x, 1)?;
    let med = median(x)?;
    let dev: Vec<f64> = x.iter().map(|v| (v - med).abs()).collect();
    Ok(1.4826 * median(&dev)?)
}
