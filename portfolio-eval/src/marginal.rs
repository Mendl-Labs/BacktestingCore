//! Marginal contribution of a candidate to an existing book (design 3.4 and the Appendix A chairman's ruling).
//!
//! The object is the pair of NET return series of two simulations on the SAME joint calendar and protocol windows:
//! book `P` (`base`) and book `P + X` (`combined`). The primary statistic is the difference of annualised Sharpe ratios
//! at equal volatility,
//!
//! ```text
//! Delta = sqrt(ppy) ( mean(r_PX) / sd(r_PX) - mean(r_P) / sd(r_P) )
//! ```
//!
//! Dividing each mean by its own volatility IS the equal-ex-volatility scaling: `c_i = target_vol / sd_i` makes
//! `mean(c_PX r_PX - c_P r_P) / target_vol` equal to `Delta`. A book that is just a levered copy of `P` therefore has
//! `Delta = 0`; comparing raw means (the "missing scaling" error) would call it a large improvement. With
//! [`ScaleMode::ExAnte`] the volatilities are constants fixed BEFORE the evaluation window (from an estimation window,
//! see [`equal_vol_scales`]) instead of the in-sample ones.
//!
//! Inference is a paired stationary block bootstrap ([`crate::bootstrap`]): the same resampled rows are used for both
//! books, so their correlation and the serial dependence inside blocks are preserved, and the volatilities are
//! re-estimated inside every replicate. The one-sided p-value for `H0: Delta <= 0` uses the null-centred bootstrap
//! distribution `Delta* - Delta_hat`:
//!
//! ```text
//! p = (1 + #{ Delta*_b - Delta_hat >= Delta_hat }) / (B + 1)
//! ```
//!
//! The "+1" makes the p-value valid (never exactly 0). The report ALWAYS carries the minimum detectable effect,
//! `(z_{1-alpha} + z_{power}) x bootstrap_se`, because on a few years of daily data most true effects are below it
//! (`POWER_TABLE.md`): "cannot distinguish from zero below x" must be shown, not a silently unpowered gate.

use crate::bootstrap::{auto_block_length, percentile_ci, StationaryBootstrap};
use crate::detmath;
use crate::error::{invalid, EvalError, Result};
use crate::hac::{spanning_alpha, HacLag, SpanningResult};
use crate::rng::Rng;
use crate::stats;

/// Fewest paired observations the test accepts (below this a bootstrap of a Sharpe difference is not meaningful).
pub const MIN_MARGINAL_OBS: usize = 30;
/// Fewest bootstrap replicates (so that `alpha = 0.05` is attainable: `(1 + 0) / (19 + 1) = 0.05`).
pub const MIN_BOOT: usize = 19;

/// Which volatilities scale each book's mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleMode {
    /// Each book's own sample standard deviation, re-estimated inside every bootstrap replicate (Sharpe difference).
    InSample,
    /// Constants fixed before the evaluation window (per-period standard deviations, both positive).
    ExAnte { base_vol: f64, combined_vol: f64 },
}

/// Mean bootstrap block length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockLength {
    Fixed(f64),
    /// Politis-White, the largest over the two books and their equal-vol difference.
    Auto,
}

/// Settings of [`marginal_contribution`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginalConfig {
    pub periods_per_year: f64,
    pub n_boot: usize,
    pub block: BlockLength,
    pub seed: u64,
    /// One-sided test level and the level used for the MDE.
    pub alpha: f64,
    /// Target power of the MDE.
    pub power: f64,
    /// Two-sided coverage of the reported percentile interval.
    pub ci_level: f64,
    pub scale: ScaleMode,
}

impl MarginalConfig {
    /// Defaults: 999 replicates, automatic block length, `alpha = 0.05`, power 0.8, 95% interval, in-sample scaling.
    pub fn new(periods_per_year: f64) -> Self {
        MarginalConfig {
            periods_per_year,
            n_boot: 999,
            block: BlockLength::Auto,
            seed: 0x5046_3520_4d41_5247,
            alpha: 0.05,
            power: 0.8,
            ci_level: 0.95,
            scale: ScaleMode::InSample,
        }
    }
}

/// Outcome of the paired marginal test.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginalResult {
    pub n: usize,
    pub periods_per_year: f64,
    pub block_length: f64,
    pub n_boot: usize,
    pub n_valid_boot: usize,
    /// Annualised in-sample Sharpe of each book.
    pub sharpe_base: f64,
    pub sharpe_combined: f64,
    /// `Delta`, the annualised Sharpe difference at equal volatility.
    pub delta_sharpe: f64,
    /// Standard deviation of the bootstrap replicates of `Delta`.
    pub boot_se: f64,
    pub ci_level: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// One-sided p-value for `H0: Delta <= 0`.
    pub p_one_sided: f64,
    /// Two-sided p-value for `H0: Delta = 0`.
    pub p_two_sided: f64,
    pub alpha: f64,
    pub power: f64,
    /// Minimum detectable `Delta` at (`alpha`, `power`) for THIS sample: `(z_{1-alpha} + z_power) x boot_se`.
    pub mde: f64,
    /// Correlation of the two books' returns.
    pub corr_books: f64,
}

impl MarginalResult {
    /// `true` when the one-sided test rejects at `alpha`.
    pub fn significant(&self) -> bool {
        self.p_one_sided <= self.alpha
    }
    /// The candidate is neither established as a contributor nor ruled out below the MDE.
    pub fn inconclusive(&self) -> bool {
        !self.significant()
    }
}

/// The primary result plus the spanning-regression diagnostic of the candidate sleeve on the base book.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginalReport {
    pub primary: MarginalResult,
    pub spanning: SpanningResult,
}

fn check_pair(base: &[f64], combined: &[f64]) -> Result<()> {
    if base.len() != combined.len() {
        return Err(EvalError::LengthMismatch { what: "marginal test books", left: base.len(), right: combined.len() });
    }
    stats::check_series("base book returns", base, MIN_MARGINAL_OBS)?;
    stats::check_series("combined book returns", combined, MIN_MARGINAL_OBS)?;
    Ok(())
}

fn resolve_scale(scale: ScaleMode, sd0: f64, sd1: f64) -> Result<(f64, f64)> {
    match scale {
        ScaleMode::InSample => Ok((sd0, sd1)),
        ScaleMode::ExAnte { base_vol, combined_vol } => {
            if !(base_vol.is_finite() && base_vol > 0.0 && combined_vol.is_finite() && combined_vol > 0.0) {
                return Err(invalid("scale", "ex-ante volatilities must be finite and positive"));
            }
            Ok((base_vol, combined_vol))
        }
    }
}

/// Point estimate `Delta` (annualised).
pub fn delta_sharpe(base: &[f64], combined: &[f64], periods_per_year: f64, scale: ScaleMode) -> Result<f64> {
    stats::check_ppy(periods_per_year)?;
    check_pair(base, combined)?;
    let (m0, m1) = (stats::mean_unchecked(base), stats::mean_unchecked(combined));
    let (sd0, sd1) = (stats::variance_unchecked(base, m0).sqrt(), stats::variance_unchecked(combined, m1).sqrt());
    if stats::is_degenerate(base, sd0) {
        return Err(EvalError::ZeroVariance { what: "base book returns" });
    }
    if stats::is_degenerate(combined, sd1) {
        return Err(EvalError::ZeroVariance { what: "combined book returns" });
    }
    let (s0, s1) = resolve_scale(scale, sd0, sd1)?;
    Ok((m1 / s1 - m0 / s0) * periods_per_year.sqrt())
}

/// Static constants `(c_base, c_combined)` that give both books the same per-period volatility `target_vol` on an
/// ESTIMATION window (data strictly before the evaluation window, so the scale is ex-ante).
pub fn equal_vol_scales(estimation_base: &[f64], estimation_combined: &[f64], target_vol: f64) -> Result<(f64, f64)> {
    if !(target_vol.is_finite() && target_vol > 0.0) {
        return Err(invalid("target_vol", "must be finite and positive"));
    }
    let sd0 = stats::std_dev(estimation_base)?;
    let sd1 = stats::std_dev(estimation_combined)?;
    if stats::is_degenerate(estimation_base, sd0) {
        return Err(EvalError::ZeroVariance { what: "estimation window of the base book" });
    }
    if stats::is_degenerate(estimation_combined, sd1) {
        return Err(EvalError::ZeroVariance { what: "estimation window of the combined book" });
    }
    Ok((target_vol / sd0, target_vol / sd1))
}

/// The paired difference series `d_t = c_combined r_combined,t - c_base r_base,t` of the equal-volatility scaled books.
pub fn equal_vol_difference(base: &[f64], combined: &[f64], c_base: f64, c_combined: f64) -> Result<Vec<f64>> {
    if base.len() != combined.len() {
        return Err(EvalError::LengthMismatch {
            what: "equal-vol difference",
            left: base.len(),
            right: combined.len(),
        });
    }
    if !(c_base.is_finite() && c_combined.is_finite()) {
        return Err(invalid("scale", "scaling constants must be finite"));
    }
    stats::check_series("base book returns", base, 1)?;
    stats::check_series("combined book returns", combined, 1)?;
    Ok(base.iter().zip(combined).map(|(b, c)| c_combined * c - c_base * b).collect())
}

/// `(z_{1-alpha} + z_{power}) x se`: the smallest true effect detected with probability `power` by a one-sided level
/// `alpha` test whose statistic has standard error `se`.
pub fn mde_from_se(se: f64, alpha: f64, power: f64) -> Result<f64> {
    if !(alpha > 0.0 && alpha < 0.5) {
        return Err(invalid("alpha", format!("must be in (0, 0.5), got {alpha}")));
    }
    if !(power > 0.0 && power < 1.0) {
        return Err(invalid("power", format!("must be in (0, 1), got {power}")));
    }
    if !(se.is_finite() && se >= 0.0) {
        return Err(invalid("se", "must be finite and non-negative"));
    }
    Ok((detmath::norm_isf(alpha) + detmath::norm_ppf(power)) * se)
}

/// Asymptotic standard error of the ANNUALISED difference of two Sharpe ratios for iid normal returns (Jobson-Korkie
/// 1981, corrected by Memmel 2003):
/// `Var(SR1 - SR2) = (1/n) [ 2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2 ]` in per-period units, times `ppy`.
/// `sr_*` are ANNUALISED Sharpe ratios, `rho` the correlation of the two return series.
pub fn analytic_se_delta_sharpe(
    n: usize,
    periods_per_year: f64,
    sr_base: f64,
    sr_combined: f64,
    rho: f64,
) -> Result<f64> {
    stats::check_ppy(periods_per_year)?;
    if n < 2 {
        return Err(EvalError::TooShort { what: "analytic Sharpe-difference SE", need: 2, got: n });
    }
    if !(rho.is_finite() && (-1.0..=1.0).contains(&rho) && sr_base.is_finite() && sr_combined.is_finite()) {
        return Err(invalid("rho", "correlation must be in [-1, 1] and Sharpe ratios finite"));
    }
    let s1 = sr_combined / periods_per_year.sqrt();
    let s2 = sr_base / periods_per_year.sqrt();
    let v = (2.0 - 2.0 * rho + 0.5 * (s1 * s1 + s2 * s2 - 2.0 * s1 * s2 * rho * rho)) / n as f64;
    Ok((v.max(0.0) * periods_per_year).sqrt())
}

/// Analytic MDE (iid normal returns) of the annualised Sharpe difference: solves `Delta = (z_{1-a} + z_p) SE(Delta)`
/// where the SE depends on `sr_combined = sr_base + Delta`, by fixed-point iteration.
pub fn mde_analytic_iid(
    n: usize,
    periods_per_year: f64,
    sr_base: f64,
    rho: f64,
    alpha: f64,
    power: f64,
) -> Result<f64> {
    let mut delta = 0.0;
    for _ in 0..60 {
        let se = analytic_se_delta_sharpe(n, periods_per_year, sr_base, sr_base + delta, rho)?;
        let next = mde_from_se(se, alpha, power)?;
        if (next - delta).abs() < 1e-13 {
            return Ok(next);
        }
        delta = next;
    }
    Ok(delta)
}

/// The paired equal-volatility marginal test. See the module docs.
pub fn marginal_contribution(base: &[f64], combined: &[f64], cfg: &MarginalConfig) -> Result<MarginalResult> {
    stats::check_ppy(cfg.periods_per_year)?;
    if cfg.n_boot < MIN_BOOT || cfg.n_boot > 1_000_000 {
        return Err(invalid("n_boot", format!("must be in [{MIN_BOOT}, 1000000], got {}", cfg.n_boot)));
    }
    if !(cfg.alpha > 0.0 && cfg.alpha < 0.5) {
        return Err(invalid("alpha", format!("must be in (0, 0.5), got {}", cfg.alpha)));
    }
    if !(cfg.power > 0.0 && cfg.power < 1.0) {
        return Err(invalid("power", format!("must be in (0, 1), got {}", cfg.power)));
    }
    if !(cfg.ci_level > 0.0 && cfg.ci_level < 1.0) {
        return Err(invalid("ci_level", format!("must be in (0, 1), got {}", cfg.ci_level)));
    }
    check_pair(base, combined)?;
    let n = base.len();
    let nf = n as f64;
    let (m0, m1) = (stats::mean_unchecked(base), stats::mean_unchecked(combined));
    let (sd0, sd1) = (stats::variance_unchecked(base, m0).sqrt(), stats::variance_unchecked(combined, m1).sqrt());
    if stats::is_degenerate(base, sd0) {
        return Err(EvalError::ZeroVariance { what: "base book returns" });
    }
    if stats::is_degenerate(combined, sd1) {
        return Err(EvalError::ZeroVariance { what: "combined book returns" });
    }
    let (s0, s1) = resolve_scale(cfg.scale, sd0, sd1)?;
    let root_ppy = cfg.periods_per_year.sqrt();
    let delta_hat = (m1 / s1 - m0 / s0) * root_ppy;

    let block_length = match cfg.block {
        BlockLength::Fixed(b) => b,
        BlockLength::Auto => {
            let diff: Vec<f64> = base.iter().zip(combined).map(|(b, c)| c / sd1 - b / sd0).collect();
            let diff_degenerate =
                stats::is_degenerate(&diff, stats::variance_unchecked(&diff, stats::mean_unchecked(&diff)).sqrt());
            if diff_degenerate {
                auto_block_length(&[base, combined])?
            } else {
                auto_block_length(&[base, combined, &diff])?
            }
        }
    };
    let bs = StationaryBootstrap::new(n, block_length)?;

    // Centre each series by its full-sample mean so that the replicate variance is computed about a well-conditioned
    // origin: var* = (sum c^2 - (sum c)^2 / n) / (n - 1).
    let c0: Vec<f64> = base.iter().map(|v| v - m0).collect();
    let c1: Vec<f64> = combined.iter().map(|v| v - m1).collect();
    let q0: Vec<f64> = c0.iter().map(|v| v * v).collect();
    let q1: Vec<f64> = c1.iter().map(|v| v * v).collect();

    let mut rng = Rng::seed_from_u64(cfg.seed);
    let mut idx = vec![0usize; n];
    let mut reps: Vec<f64> = Vec::with_capacity(cfg.n_boot);
    for _ in 0..cfg.n_boot {
        bs.fill(&mut rng, &mut idx);
        let (mut a0, mut a1, mut b0, mut b1) = (0.0, 0.0, 0.0, 0.0);
        for &i in &idx {
            a0 += c0[i];
            a1 += c1[i];
            b0 += q0[i];
            b1 += q1[i];
        }
        let mean0 = m0 + a0 / nf;
        let mean1 = m1 + a1 / nf;
        let (rs0, rs1) = match cfg.scale {
            ScaleMode::InSample => {
                let v0 = (b0 - a0 * a0 / nf) / (nf - 1.0);
                let v1 = (b1 - a1 * a1 / nf) / (nf - 1.0);
                if v0 > 0.0 && v1 > 0.0 {
                    (v0.sqrt(), v1.sqrt())
                } else {
                    (f64::NAN, f64::NAN)
                }
            }
            ScaleMode::ExAnte { .. } => (s0, s1),
        };
        reps.push((mean1 / rs1 - mean0 / rs0) * root_ppy);
    }
    let valid: Vec<f64> = reps.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.len() * 2 < cfg.n_boot {
        return Err(EvalError::DegenerateBootstrap { valid: valid.len(), requested: cfg.n_boot });
    }
    let nv = valid.len();
    let mut ge = 0usize;
    let mut ge_abs = 0usize;
    for v in &valid {
        let centred = v - delta_hat;
        if centred >= delta_hat {
            ge += 1;
        }
        if centred.abs() >= delta_hat.abs() {
            ge_abs += 1;
        }
    }
    let p_one_sided = (1.0 + ge as f64) / (nv as f64 + 1.0);
    let p_two_sided = ((1.0 + ge_abs as f64) / (nv as f64 + 1.0)).min(1.0);
    let boot_se = stats::variance_unchecked(&valid, stats::mean_unchecked(&valid)).sqrt();
    let (ci_low, ci_high) = percentile_ci(&valid, cfg.ci_level)?;
    let mde = mde_from_se(boot_se, cfg.alpha, cfg.power)?;
    Ok(MarginalResult {
        n,
        periods_per_year: cfg.periods_per_year,
        block_length,
        n_boot: cfg.n_boot,
        n_valid_boot: nv,
        sharpe_base: m0 / sd0 * root_ppy,
        sharpe_combined: m1 / sd1 * root_ppy,
        delta_sharpe: delta_hat,
        boot_se,
        ci_level: cfg.ci_level,
        ci_low,
        ci_high,
        p_one_sided,
        p_two_sided,
        alpha: cfg.alpha,
        power: cfg.power,
        mde,
        corr_books: stats::correlation(base, combined)?,
    })
}

/// The primary marginal test plus the spanning regression of the candidate sleeve's returns on the base book (the
/// diagnostic of design 3.4). `candidate` is the sleeve's own return series on the same calendar.
pub fn marginal_report(
    base: &[f64],
    combined: &[f64],
    candidate: &[f64],
    cfg: &MarginalConfig,
    lag: HacLag,
) -> Result<MarginalReport> {
    let primary = marginal_contribution(base, combined, cfg)?;
    let spanning = spanning_alpha(candidate, &[base], lag, cfg.periods_per_year)?;
    Ok(MarginalReport { primary, spanning })
}
