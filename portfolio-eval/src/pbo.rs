//! Probability of backtest overfitting by combinatorially symmetric cross-validation (CSCV; Bailey, Borwein, Lopez de
//! Prado and Zhu 2015), over the matrix of portfolio returns of the `N` configurations tried on a book (design 4.4).
//!
//! Procedure: cut the `T` rows into `S` equal contiguous blocks (`S` even; the remainder `T mod S` rows are dropped
//! from the START so the most recent data is kept). For every choice of `S/2` blocks as in-sample (`C(S, S/2)` splits,
//! lexicographic), pick the configuration `n*` with the best in-sample performance (ties: lowest index), find its
//! out-of-sample performance rank among all `N` configurations, `omega = rank / (N + 1)` (1-based rank, ties get the
//! mid-rank), and the logit `lambda = ln(omega / (1 - omega))`. `PBO = #{lambda <= 0} / #splits`.
//!
//! A configuration whose in-sample or out-of-sample Sharpe is undefined (zero variance, e.g. it never traded in that
//! half) gets performance 0. Path stitching and position resets at block boundaries are a stated approximation of the
//! design (the returns matrix is taken as given).

use crate::detmath;
use crate::error::{invalid, EvalError, Result};
use crate::folds::{combinations, combinations_count};
use crate::stats;

/// Performance measure used to rank configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PboMetric {
    /// Per-period Sharpe ratio (`mean / std`, `ddof = 1`).
    Sharpe,
    /// Mean return.
    Mean,
}

/// Result of [`pbo_cscv`].
#[derive(Clone, Debug, PartialEq)]
pub struct PboResult {
    /// Probability of backtest overfitting `P(lambda <= 0)`.
    pub pbo: f64,
    pub n_splits: usize,
    pub n_configs: usize,
    pub n_blocks: usize,
    /// `lambda` of every split, in split order.
    pub logits: Vec<f64>,
    /// Index of the in-sample best configuration in every split.
    pub selected: Vec<usize>,
    /// Fraction of splits where the selected configuration's OUT-OF-SAMPLE performance is negative.
    pub prob_loss: f64,
    /// OLS slope of OOS on IS performance of the selected configurations (`NaN` when IS performance is constant).
    pub degradation_slope: f64,
}

#[derive(Clone, Copy)]
struct BlockSums {
    n: f64,
    s: f64,
    q: f64,
}

fn perf(metric: PboMetric, n: f64, s: f64, q: f64, centre: f64) -> f64 {
    // sums are of values centred by `centre` (the config's overall mean): mean = centre + s/n
    let mean = centre + s / n;
    match metric {
        PboMetric::Mean => mean,
        PboMetric::Sharpe => {
            let var = (q - s * s / n) / (n - 1.0);
            // treat rounding-level variance as zero (a flat configuration)
            if var > 1e-24 * (1.0 + mean * mean) {
                mean / var.sqrt()
            } else {
                0.0
            }
        }
    }
}

/// CSCV probability of backtest overfitting. `returns[i]` is the return series of configuration `i`; all series must
/// have the same length `T >= 2 x n_blocks`.
pub fn pbo_cscv(returns: &[&[f64]], n_blocks: usize, metric: PboMetric) -> Result<PboResult> {
    let n_cfg = returns.len();
    if n_cfg < 2 {
        return Err(EvalError::TooShort { what: "PBO configurations", need: 2, got: n_cfg });
    }
    if n_blocks < 2 || !n_blocks.is_multiple_of(2) {
        return Err(invalid("n_blocks", format!("must be an even number >= 2, got {n_blocks}")));
    }
    match combinations_count(n_blocks, n_blocks / 2) {
        Some(c) if c <= 200_000 => {}
        _ => return Err(invalid("n_blocks", "too many combinations (limit 200000)")),
    }
    let t = returns[0].len();
    for r in returns {
        if r.len() != t {
            return Err(EvalError::LengthMismatch { what: "PBO configuration returns", left: t, right: r.len() });
        }
        stats::check_series("PBO configuration returns", r, 2 * n_blocks)?;
    }
    let block_len = t / n_blocks;
    let start = t - block_len * n_blocks;
    // per-config centre and per-block centred sums
    let mut centres = vec![0.0; n_cfg];
    let mut sums: Vec<Vec<BlockSums>> = Vec::with_capacity(n_cfg);
    for (i, r) in returns.iter().enumerate() {
        let used = &r[start..];
        let c = stats::mean_unchecked(used);
        centres[i] = c;
        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let (mut s, mut q) = (0.0, 0.0);
            for v in &used[b * block_len..(b + 1) * block_len] {
                let d = v - c;
                s += d;
                q += d * d;
            }
            blocks.push(BlockSums { n: block_len as f64, s, q });
        }
        sums.push(blocks);
    }
    let splits = combinations(n_blocks, n_blocks / 2);
    let mut logits = Vec::with_capacity(splits.len());
    let mut selected = Vec::with_capacity(splits.len());
    let mut is_perf = Vec::with_capacity(splits.len());
    let mut oos_perf = Vec::with_capacity(splits.len());
    let mut in_is = vec![false; n_blocks];
    for combo in &splits {
        in_is.iter_mut().for_each(|v| *v = false);
        for &b in combo {
            in_is[b] = true;
        }
        let mut is_p = vec![0.0; n_cfg];
        let mut oos_p = vec![0.0; n_cfg];
        for i in 0..n_cfg {
            let (mut ni, mut si, mut qi) = (0.0, 0.0, 0.0);
            let (mut no, mut so, mut qo) = (0.0, 0.0, 0.0);
            for b in 0..n_blocks {
                let bs = sums[i][b];
                if in_is[b] {
                    ni += bs.n;
                    si += bs.s;
                    qi += bs.q;
                } else {
                    no += bs.n;
                    so += bs.s;
                    qo += bs.q;
                }
            }
            is_p[i] = perf(metric, ni, si, qi, centres[i]);
            oos_p[i] = perf(metric, no, so, qo, centres[i]);
        }
        let mut best = 0usize;
        for i in 1..n_cfg {
            if is_p[i] > is_p[best] {
                best = i;
            }
        }
        let target = oos_p[best];
        let below = oos_p.iter().filter(|v| **v < target).count() as f64;
        let equal_others = oos_p.iter().enumerate().filter(|(i, v)| *i != best && **v == target).count() as f64;
        let rank = below + 1.0 + equal_others / 2.0; // mid-rank, 1-based
        let omega = rank / (n_cfg as f64 + 1.0);
        logits.push(detmath::ln(omega / (1.0 - omega)));
        selected.push(best);
        is_perf.push(is_p[best]);
        oos_perf.push(target);
    }
    let ns = splits.len() as f64;
    let pbo = logits.iter().filter(|l| **l <= 0.0).count() as f64 / ns;
    let prob_loss = oos_perf.iter().filter(|v| **v < 0.0).count() as f64 / ns;
    let mi = stats::mean_unchecked(&is_perf);
    let mo = stats::mean_unchecked(&oos_perf);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for k in 0..is_perf.len() {
        sxx += (is_perf[k] - mi) * (is_perf[k] - mi);
        sxy += (is_perf[k] - mi) * (oos_perf[k] - mo);
    }
    let degradation_slope = if sxx > 0.0 { sxy / sxx } else { f64::NAN };
    Ok(PboResult {
        pbo,
        n_splits: splits.len(),
        n_configs: n_cfg,
        n_blocks,
        logits,
        selected,
        prob_loss,
        degradation_slope,
    })
}
