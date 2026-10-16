//! Heteroskedasticity-and-autocorrelation-consistent (Newey-West) inference, and the spanning regression of a
//! candidate sleeve on the existing book (Huberman-Kandel 1987; Gibbons-Ross-Shanken 1989):
//!
//! ```text
//! x_t = alpha + beta' p_t + eps_t        SR^2(P + X) - SR^2(P) = alpha^2 / sigma_eps^2     (P the benchmark set)
//! ```
//!
//! `alpha` is the diagnostic reported beside the primary paired test (design 3.4). Standard errors use the Bartlett
//! kernel (Newey and West 1987) sandwich `(Z'Z)^-1 Omega (Z'Z)^-1` with `Omega = sum_t s_t s_t' + sum_{l=1}^{L} w_l
//! sum_t (s_t s_{t-l}' + s_{t-l} s_t')`, `s_t = z_t e_t`, `w_l = 1 - l/(L+1)`. There is NO small-sample degrees-of-
//! freedom correction (the statsmodels `use_correction=False` default), and p-values use the normal distribution, so
//! for `n` below about 100 they are optimistic; callers get `n` in the result.

use crate::detmath;
use crate::error::{invalid, EvalError, Result};
use crate::stats;

/// How many autocovariance lags the kernel uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HacLag {
    /// Newey-West (1994) rule `floor(4 (n/100)^(2/9))`.
    Auto,
    /// A fixed number of lags (0 gives the White heteroskedasticity-robust estimator).
    Fixed(usize),
}

/// The Newey-West (1994) automatic lag `floor(4 (n/100)^(2/9))`.
pub fn newey_west_lag(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (4.0 * detmath::pow(n as f64 / 100.0, 2.0 / 9.0)).floor() as usize
}

/// Resolve a [`HacLag`] for a sample of `n`, never above `n - 1`.
pub fn resolve_lag(lag: HacLag, n: usize) -> usize {
    let l = match lag {
        HacLag::Auto => newey_west_lag(n),
        HacLag::Fixed(l) => l,
    };
    l.min(n.saturating_sub(1))
}

/// Bartlett-kernel long-run variance of the MEAN of `x` (about its sample mean): `G0 + 2 sum_{l=1}^{L} (1 - l/(L+1)) G_l`
/// with `G_l` the `1/n` autocovariances.
pub fn long_run_variance_of_mean(x: &[f64], lag: usize) -> Result<f64> {
    stats::check_series("long-run variance input", x, 2)?;
    if lag >= x.len() {
        return Err(invalid("lag", format!("lag {lag} must be below the series length {}", x.len())));
    }
    let mut lrv = stats::autocovariance(x, 0)?;
    for l in 1..=lag {
        let w = 1.0 - l as f64 / (lag as f64 + 1.0);
        lrv += 2.0 * w * stats::autocovariance(x, l)?;
    }
    Ok(lrv)
}

/// Result of a spanning regression of `y` on an intercept and the benchmark returns.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanningResult {
    pub n: usize,
    pub lag: usize,
    /// Intercept per period.
    pub alpha: f64,
    /// Intercept x periods_per_year.
    pub alpha_annual: f64,
    /// Slope on each benchmark, in the order given.
    pub beta: Vec<f64>,
    /// Newey-West standard error of `alpha` (per period).
    pub se_alpha: f64,
    /// Newey-West standard errors of the slopes.
    pub se_beta: Vec<f64>,
    pub t_alpha: f64,
    /// Two-sided normal p-value of `t_alpha`.
    pub p_two_sided: f64,
    /// One-sided normal p-value for `H0: alpha <= 0`.
    pub p_one_sided: f64,
    /// Residual variance `sum e^2 / (n - K - 1)`.
    pub resid_var: f64,
    pub r_squared: f64,
    /// `alpha^2 / resid_var x periods_per_year`: the annualised gain in squared Sharpe from adding `y` to the benchmark.
    pub sr2_increment_annual: f64,
}

fn invert(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let p = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut inv: Vec<Vec<f64>> = (0..p).map(|i| (0..p).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let scale = a.iter().flat_map(|r| r.iter()).fold(0.0_f64, |s, v| s.max(v.abs()));
    for col in 0..p {
        let mut piv = col;
        for r in col + 1..p {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() <= 1e-13 * scale || !m[piv][col].is_finite() {
            return Err(EvalError::Singular { what: "spanning regression normal equations" });
        }
        m.swap(col, piv);
        inv.swap(col, piv);
        let d = m[col][col];
        for j in 0..p {
            m[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..p {
            if r != col {
                let f = m[r][col];
                if f != 0.0 {
                    for j in 0..p {
                        m[r][j] -= f * m[col][j];
                        inv[r][j] -= f * inv[col][j];
                    }
                }
            }
        }
    }
    Ok(inv)
}

/// Regress `y` on an intercept and `benchmarks` (each the same length as `y`) with Newey-West standard errors.
/// `periods_per_year` only annualises the reported `alpha_annual` and `sr2_increment_annual`.
pub fn spanning_alpha(y: &[f64], benchmarks: &[&[f64]], lag: HacLag, periods_per_year: f64) -> Result<SpanningResult> {
    stats::check_ppy(periods_per_year)?;
    if benchmarks.is_empty() {
        return Err(invalid("benchmarks", "at least one benchmark series is required"));
    }
    let n = y.len();
    let p = benchmarks.len() + 1;
    stats::check_series("spanning y", y, p + 2)?;
    for b in benchmarks {
        if b.len() != n {
            return Err(EvalError::LengthMismatch { what: "spanning benchmark", left: n, right: b.len() });
        }
        stats::check_series("spanning benchmark", b, p + 2)?;
    }
    let l = resolve_lag(lag, n);
    let z = |t: usize, j: usize| if j == 0 { 1.0 } else { benchmarks[j - 1][t] };
    let mut a = vec![vec![0.0; p]; p];
    let mut c = vec![0.0; p];
    for t in 0..n {
        for i in 0..p {
            let zi = z(t, i);
            c[i] += zi * y[t];
            for j in 0..p {
                a[i][j] += zi * z(t, j);
            }
        }
    }
    let inv = invert(&a)?;
    let theta: Vec<f64> = (0..p).map(|i| (0..p).map(|j| inv[i][j] * c[j]).sum()).collect();
    let mut resid = vec![0.0; n];
    let mut sse = 0.0;
    for t in 0..n {
        let fit: f64 = (0..p).map(|j| theta[j] * z(t, j)).sum();
        resid[t] = y[t] - fit;
        sse += resid[t] * resid[t];
    }
    let ybar = stats::mean_unchecked(y);
    let sst: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    // scores s_t = z_t e_t
    let score = |t: usize, j: usize| z(t, j) * resid[t];
    let mut omega = vec![vec![0.0; p]; p];
    for t in 0..n {
        for i in 0..p {
            for j in 0..p {
                omega[i][j] += score(t, i) * score(t, j);
            }
        }
    }
    for k in 1..=l {
        let w = 1.0 - k as f64 / (l as f64 + 1.0);
        for t in k..n {
            for i in 0..p {
                for j in 0..p {
                    omega[i][j] += w * (score(t, i) * score(t - k, j) + score(t - k, i) * score(t, j));
                }
            }
        }
    }
    // V = inv * omega * inv
    let mut tmp = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            tmp[i][j] = (0..p).map(|k| inv[i][k] * omega[k][j]).sum();
        }
    }
    let mut v = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            v[i][j] = (0..p).map(|k| tmp[i][k] * inv[k][j]).sum();
        }
    }
    let se_alpha = v[0][0].sqrt();
    if !(se_alpha.is_finite() && se_alpha > 0.0) {
        return Err(EvalError::ZeroVariance { what: "spanning residuals" });
    }
    let se_beta: Vec<f64> = (1..p).map(|i| v[i][i].sqrt()).collect();
    let alpha = theta[0];
    let t_alpha = alpha / se_alpha;
    let resid_var = sse / (n - p) as f64;
    Ok(SpanningResult {
        n,
        lag: l,
        alpha,
        alpha_annual: alpha * periods_per_year,
        beta: theta[1..].to_vec(),
        se_alpha,
        se_beta,
        t_alpha,
        p_two_sided: 2.0 * detmath::norm_sf(t_alpha.abs()),
        p_one_sided: detmath::norm_sf(t_alpha),
        resid_var,
        r_squared: if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN },
        sr2_increment_annual: if resid_var > 0.0 {
            alpha * alpha / resid_var * periods_per_year
        } else {
            f64::INFINITY
        },
    })
}
