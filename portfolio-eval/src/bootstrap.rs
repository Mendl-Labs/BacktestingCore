//! Stationary block bootstrap (Politis and Romano 1994) on row indices, and the Politis-White (2004) automatic mean
//! block length with the Patton-Politis-White (2009) correction.
//!
//! The bootstrap resamples ROW INDICES, not values, so several series can be resampled with the SAME indices
//! (paired resampling): cross-series correlation and the serial dependence inside blocks are both preserved (design
//! 4.4, "Sampling uncertainty"). Index generation uses integer arithmetic only, so the drawn indices (and the pinned
//! digests in the tests) are identical on every platform.
//!
//! Algorithm: `I_1 ~ U{0..n-1}`; for `t > 1`, with probability `p = 1 / mean_block` start a new block
//! (`I_t ~ U{0..n-1}`), otherwise continue the current block circularly (`I_t = (I_{t-1} + 1) mod n`). Block lengths
//! are geometric with mean `mean_block`.

use crate::detmath;
use crate::error::{invalid, EvalError, Result};
use crate::rng::Rng;
use crate::sha256::sha256_hex;
use crate::stats;

/// Reusable index generator for a series of length `n` and a given mean block length.
#[derive(Clone, Debug)]
pub struct StationaryBootstrap {
    n: usize,
    mean_block: f64,
    /// `P(new block) x 2^64`, or `None` when every draw starts a new block (`mean_block == 1`, the iid bootstrap).
    new_block_threshold: Option<u64>,
}

impl StationaryBootstrap {
    /// `n >= 2`; `mean_block` finite and in `[1, n]`.
    pub fn new(n: usize, mean_block: f64) -> Result<Self> {
        if n < 2 {
            return Err(EvalError::TooShort { what: "bootstrap series", need: 2, got: n });
        }
        if !(mean_block.is_finite() && mean_block >= 1.0 && mean_block <= n as f64) {
            return Err(invalid("mean_block", format!("must be finite and in [1, n={n}], got {mean_block}")));
        }
        let new_block_threshold =
            if mean_block <= 1.0 { None } else { Some(((1.0 / mean_block) * 18_446_744_073_709_551_616.0) as u64) };
        Ok(StationaryBootstrap { n, mean_block, new_block_threshold })
    }

    /// The mean block length this generator was built with.
    pub fn mean_block(&self) -> f64 {
        self.mean_block
    }

    /// Fill `out` with one bootstrap sample of indices in `0..n`.
    pub fn fill(&self, rng: &mut Rng, out: &mut [usize]) {
        let n = self.n as u64;
        let mut prev = 0usize;
        for (t, slot) in out.iter_mut().enumerate() {
            let idx = if t == 0 {
                rng.below(n) as usize
            } else {
                match self.new_block_threshold {
                    None => rng.below(n) as usize,
                    Some(th) => {
                        if rng.next_u64() < th {
                            rng.below(n) as usize
                        } else {
                            let nx = prev + 1;
                            if nx == self.n {
                                0
                            } else {
                                nx
                            }
                        }
                    }
                }
            };
            *slot = idx;
            prev = idx;
        }
    }
}

/// One bootstrap sample of `n` indices from `seed`.
pub fn bootstrap_indices(n: usize, mean_block: f64, seed: u64) -> Result<Vec<usize>> {
    let bs = StationaryBootstrap::new(n, mean_block)?;
    let mut rng = Rng::seed_from_u64(seed);
    let mut out = vec![0usize; n];
    bs.fill(&mut rng, &mut out);
    Ok(out)
}

/// SHA-256 (hex) of the indices as little-endian `u64`s; pins a bootstrap draw exactly.
pub fn indices_digest(indices: &[usize]) -> String {
    let mut bytes = Vec::with_capacity(indices.len() * 8);
    for i in indices {
        bytes.extend_from_slice(&(*i as u64).to_le_bytes());
    }
    sha256_hex(&bytes)
}

/// Evaluate `stat` on `n_boot` bootstrap samples (one RNG stream seeded with `seed`, replicates drawn in order).
pub fn bootstrap_statistic<F: FnMut(&[usize]) -> f64>(
    n: usize,
    mean_block: f64,
    n_boot: usize,
    seed: u64,
    mut stat: F,
) -> Result<Vec<f64>> {
    let bs = StationaryBootstrap::new(n, mean_block)?;
    let mut rng = Rng::seed_from_u64(seed);
    let mut idx = vec![0usize; n];
    let mut out = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        bs.fill(&mut rng, &mut idx);
        out.push(stat(&idx));
    }
    Ok(out)
}

/// Percentile confidence interval `[q_{(1-level)/2}, q_{1-(1-level)/2}]` (type-7 quantiles) of the finite values.
pub fn percentile_ci(values: &[f64], level: f64) -> Result<(f64, f64)> {
    if !(level > 0.0 && level < 1.0) {
        return Err(invalid("ci_level", format!("must be in (0, 1), got {level}")));
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return Err(EvalError::TooShort { what: "bootstrap replicates", need: 2, got: v.len() });
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let tail = (1.0 - level) / 2.0;
    Ok((stats::quantile_sorted(&v, tail), stats::quantile_sorted(&v, 1.0 - tail)))
}

const LN_10: f64 = std::f64::consts::LN_10;

/// Politis-White (2004) optimal mean block length for the STATIONARY bootstrap of the sample mean of `x`, with the
/// Patton-Politis-White (2009) correction (`D_SB = 2 g(0)^2`), flat-top lag window, and the usual upper bound
/// `ceil(min(3 sqrt(n), n/3))`. Never below 1 (which is the iid bootstrap). Needs `n >= 20`.
pub fn politis_white_block_length(x: &[f64]) -> Result<f64> {
    stats::check_series("block-length input", x, 20)?;
    let n = x.len();
    let nf = n as f64;
    let sd = stats::variance_unchecked(x, stats::mean_unchecked(x)).sqrt();
    if stats::is_degenerate(x, sd) {
        return Err(EvalError::ZeroVariance { what: "block-length input" });
    }
    let log10n = detmath::ln(nf) / LN_10;
    let kn = 5usize.max(log10n.sqrt().ceil() as usize);
    let m_max = ((nf.sqrt().ceil() as usize) + kn).min(n - 1);
    let mut r = Vec::with_capacity(m_max + 1);
    for k in 0..=m_max {
        r.push(stats::autocovariance(x, k)?);
    }
    let crit = 2.0 * (log10n / nf).sqrt();
    // smallest m such that the following kn autocorrelations are all insignificant
    let mut mhat = m_max.saturating_sub(kn);
    if m_max >= kn {
        for m in 0..=(m_max - kn) {
            if (1..=kn).all(|j| (r[m + j] / r[0]).abs() < crit) {
                mhat = m;
                break;
            }
        }
    }
    let big_m = (2 * mhat.max(1)).min(m_max);
    let mf = big_m as f64;
    let mut g = r[0];
    let mut big_g = 0.0;
    for k in 1..=big_m {
        let s = k as f64 / mf;
        let lam = if s <= 0.5 { 1.0 } else { 2.0 * (1.0 - s) };
        g += 2.0 * lam * r[k];
        big_g += 2.0 * lam * k as f64 * r[k];
    }
    let b_max = (3.0 * nf.sqrt()).min(nf / 3.0).ceil();
    if g.is_nan() || g <= 0.0 || big_g == 0.0 {
        return Ok(1.0);
    }
    let b = detmath::pow(nf * big_g * big_g / (g * g), 1.0 / 3.0);
    if !b.is_finite() {
        return Ok(1.0);
    }
    Ok(b.clamp(1.0, b_max.max(1.0)))
}

/// The largest Politis-White block length over several series that will be resampled together.
pub fn auto_block_length(series: &[&[f64]]) -> Result<f64> {
    if series.is_empty() {
        return Err(invalid("series", "at least one series is required"));
    }
    let mut best = 1.0_f64;
    for s in series {
        best = best.max(politis_white_block_length(s)?);
    }
    Ok(best)
}
