//! Deflated Sharpe ratio, probabilistic Sharpe ratio, minimum track record length and Benjamini-Hochberg q-values.
//!
//! What Core already has (and this module deliberately does NOT modify; it MIRRORS it in [`core_compat`] and tests
//! equivalence on shared inputs, `tests/core_equivalence.rs`):
//!
//! * `metrics::significance::deflated_sharpe_ratio(observed_sharpe, n_trials, n_days, sharpe_std, ppy)`: annualised
//!   Sharpe in, iid variance `1/n`, expected maximum by an extreme-value approximation with a Gumbel correction, and
//!   the Abramowitz-Stegun 7.1.26 `erfc` (1.5e-7 absolute error).
//! * `metrics::performance::deflated_sharpe_ratio(num_trials, returns)`: Bailey-Lopez de Prado variance with skewness
//!   and kurtosis, floored at the normal baseline `1/(n-1)`, expected maximum `(1 - g + g ln N)/sqrt(n-1)`.
//! * `quant_diagnostics::multiple_testing::benjamini_hochberg`.
//!
//! What [`deflated_sharpe`] adds, as the design specifies (4.4): the trial dispersion is an INPUT (the robust
//! dispersion of the book's trial ledger, [`crate::ledger`]), the expected maximum is the Bailey-Lopez de Prado (2014)
//! formula `(1 - g) Phi^{-1}(1 - 1/N) + g Phi^{-1}(1 - 1/(N e))`, and `erfc`/`Phi^{-1}` are the accurate ones of
//! [`crate::detmath`]. The variance floor of Core's second function is kept by default (moment corrections may widen
//! the interval but sampling noise in skewness must not shrink it). None of these formulas repairs autocorrelation;
//! block-bootstrap intervals ([`crate::marginal`]) are the primary uncertainty statement and the DSR is reported beside
//! them (design 4.4).

use crate::detmath;
use crate::error::{invalid, EvalError, Result};
use crate::stats;

/// Euler-Mascheroni constant.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Expected maximum of `n_trials` independent standard normal draws by the Bailey-Lopez de Prado (2014, eq. 26)
/// approximation `(1 - g) Phi^{-1}(1 - 1/N) + g Phi^{-1}(1 - 1/(N e))`. Zero for `n_trials <= 1`.
pub fn expected_max_normal(n_trials: usize) -> f64 {
    if n_trials <= 1 {
        return 0.0;
    }
    let n = n_trials as f64;
    (1.0 - EULER_GAMMA) * detmath::norm_isf(1.0 / n) + EULER_GAMMA * detmath::norm_isf(1.0 / (n * std::f64::consts::E))
}

/// Inputs of [`deflated_sharpe`]; every Sharpe ratio here is PER PERIOD (not annualised).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsrInputs {
    /// Observed per-period Sharpe of the selected configuration.
    pub sharpe: f64,
    /// Number of return observations `T`.
    pub n_obs: usize,
    /// Skewness of the returns (population moment).
    pub skewness: f64,
    /// Kurtosis of the returns, NOT excess (normal = 3).
    pub kurtosis: f64,
    /// Effective number of independent trials `K_effective` from the ledger.
    pub n_trials: usize,
    /// Dispersion (standard deviation, robust in the ledger) of the per-period Sharpe ratios across the trials.
    pub trial_sharpe_std: f64,
}

/// Result of the deflated Sharpe ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsrResult {
    /// `Phi(z)`: probability the observed Sharpe exceeds the best of the trials under the null.
    pub dsr: f64,
    /// The null benchmark `SR0` (per period).
    pub sr0: f64,
    /// Standard error of the Sharpe estimator used (per period).
    pub se: f64,
    pub z: f64,
}

fn sharpe_se_sq(sr: f64, n_obs: usize, skew: f64, kurt: f64, floor_at_normal: bool) -> Result<f64> {
    if n_obs < 3 {
        return Err(EvalError::TooShort { what: "Sharpe standard error", need: 3, got: n_obs });
    }
    if !(sr.is_finite() && skew.is_finite() && kurt.is_finite()) {
        return Err(invalid("moments", "Sharpe, skewness and kurtosis must be finite"));
    }
    let nm1 = (n_obs - 1) as f64;
    let raw = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / nm1;
    let baseline = 1.0 / nm1;
    let v = if floor_at_normal { raw.max(baseline) } else { raw };
    if v.is_nan() || v <= 0.0 {
        return Err(invalid(
            "moments",
            "the Sharpe variance term is not positive (skewness too large for this Sharpe)",
        ));
    }
    Ok(v)
}

/// Deflated Sharpe ratio (Bailey and Lopez de Prado 2014): `Phi( (SR - SR0) / se )` with
/// `SR0 = trial_sharpe_std x E[max of N standard normals]` and
/// `se^2 = (1 - skew SR + (kurt - 1)/4 SR^2) / (T - 1)`. `floor_at_normal` (recommended, Core's convention) floors
/// `se^2` at `1/(T-1)`.
pub fn deflated_sharpe(inp: &DsrInputs, floor_at_normal: bool) -> Result<DsrResult> {
    if inp.n_trials == 0 {
        return Err(invalid("n_trials", "at least one trial (the selected configuration itself) is required"));
    }
    if !(inp.trial_sharpe_std.is_finite() && inp.trial_sharpe_std >= 0.0) {
        return Err(invalid("trial_sharpe_std", "must be finite and non-negative"));
    }
    let se = sharpe_se_sq(inp.sharpe, inp.n_obs, inp.skewness, inp.kurtosis, floor_at_normal)?.sqrt();
    let sr0 = inp.trial_sharpe_std * expected_max_normal(inp.n_trials);
    let z = (inp.sharpe - sr0) / se;
    Ok(DsrResult { dsr: detmath::norm_cdf(z), sr0, se, z })
}

/// Probabilistic Sharpe ratio: the probability that the true per-period Sharpe exceeds `benchmark`.
pub fn probabilistic_sharpe(
    sharpe: f64,
    benchmark: f64,
    n_obs: usize,
    skewness: f64,
    kurtosis: f64,
    floor_at_normal: bool,
) -> Result<f64> {
    let se = sharpe_se_sq(sharpe, n_obs, skewness, kurtosis, floor_at_normal)?.sqrt();
    Ok(detmath::norm_cdf((sharpe - benchmark) / se))
}

/// Minimum track record length (Bailey and Lopez de Prado 2012): the number of observations needed for the
/// observed per-period `sharpe` to exceed `benchmark` with probability `confidence`,
/// `1 + (1 - skew SR + (kurt - 1)/4 SR^2) (Phi^{-1}(confidence) / (SR - benchmark))^2`.
pub fn min_track_record_length(
    sharpe: f64,
    benchmark: f64,
    skewness: f64,
    kurtosis: f64,
    confidence: f64,
) -> Result<f64> {
    if !(confidence > 0.5 && confidence < 1.0) {
        return Err(invalid("confidence", format!("must be in (0.5, 1), got {confidence}")));
    }
    if !(sharpe.is_finite() && benchmark.is_finite() && skewness.is_finite() && kurtosis.is_finite()) {
        return Err(invalid("moments", "inputs must be finite"));
    }
    if sharpe <= benchmark {
        return Err(invalid("sharpe", "the observed Sharpe must exceed the benchmark for a finite track record"));
    }
    let term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe;
    if term.is_nan() || term <= 0.0 {
        return Err(invalid("moments", "the Sharpe variance term is not positive"));
    }
    let z = detmath::norm_ppf(confidence) / (sharpe - benchmark);
    Ok(1.0 + term * z * z)
}

/// Deflated Sharpe ratio of a return series: computes the per-period Sharpe, skewness and kurtosis from `returns`.
/// `trial_sharpe_std_annual` is the ANNUALISED dispersion of the trial Sharpe ratios (the ledger reports annual units).
pub fn deflated_sharpe_from_returns(
    returns: &[f64],
    n_trials: usize,
    trial_sharpe_std_annual: f64,
    periods_per_year: f64,
    floor_at_normal: bool,
) -> Result<DsrResult> {
    stats::check_ppy(periods_per_year)?;
    let sr = stats::sharpe_per_period(returns)?;
    let skew = stats::skewness(returns)?;
    let kurt = stats::kurtosis(returns)?;
    deflated_sharpe(
        &DsrInputs {
            sharpe: sr,
            n_obs: returns.len(),
            skewness: skew,
            kurtosis: kurt,
            n_trials,
            trial_sharpe_std: trial_sharpe_std_annual / periods_per_year.sqrt(),
        },
        floor_at_normal,
    )
}

/// Benjamini-Hochberg adjusted p-values ("q-values"), same order as the input (mirror of
/// `quant_diagnostics::multiple_testing::benjamini_hochberg`, with typed errors instead of silent NaN handling):
/// `q_(i) = min_{k >= i} min(1, m/k p_(k))`. Used across the candidates screened for marginal contribution.
pub fn benjamini_hochberg_q(p_values: &[f64]) -> Result<Vec<f64>> {
    for (i, p) in p_values.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(p) {
            return Err(EvalError::NonFinite { what: "p-value (must be finite and in [0, 1])", index: i });
        }
    }
    let m = p_values.len();
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut adjusted = vec![0.0; m];
    let mut min_so_far = 1.0_f64;
    for rank in (0..m).rev() {
        let bh = (indexed[rank].1 * m as f64 / (rank as f64 + 1.0)).min(1.0);
        min_so_far = min_so_far.min(bh);
        adjusted[rank] = min_so_far;
    }
    let mut out = vec![0.0; m];
    for (rank, &(orig, _)) in indexed.iter().enumerate() {
        out[orig] = adjusted[rank];
    }
    Ok(out)
}

/// Bit-for-bit mirrors of Core's formulas (see the module docs), kept so equivalence with Core can be tested on
/// shared inputs and so a reader can see exactly how the crate's own DSR differs. They use the same approximations
/// as Core (A&S 7.1.26 `erfc`, Acklam probit without refinement, Gumbel-corrected expected maximum); only the
/// elementary `exp`/`ln` come from [`crate::detmath`].
#[allow(clippy::excessive_precision)] // constants and formulas are copied verbatim from Core
pub mod core_compat {
    use crate::detmath;

    fn erfc_as(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let a = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let result = poly * detmath::exp(-a * a);
        if sign < 0.0 {
            2.0 - result
        } else {
            result
        }
    }

    fn probit_acklam(p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < 1e-15 {
            return 0.0;
        }
        const A: [f64; 6] = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        const D: [f64; 4] =
            [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
        let p_low = 0.02425;
        let p_high = 1.0 - p_low;
        if p < p_low {
            let q = (-2.0 * detmath::ln(p)).sqrt();
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
        } else {
            let q = (-2.0 * detmath::ln(1.0 - p)).sqrt();
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        }
    }

    fn expected_max_normal_core(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let n_f = n as f64;
        let z = probit_acklam(1.0 - 1.0 / n_f);
        let euler_mascheroni = 0.5772156649;
        let ln_n = detmath::ln(n_f);
        if ln_n > 0.0 && z.abs() > 1e-12 {
            z + euler_mascheroni / (z * (2.0 * ln_n).sqrt())
        } else {
            z
        }
    }

    /// Mirror of `metrics::significance::deflated_sharpe_ratio(observed_sharpe, n_trials, n_days, sharpe_std, ppy)`.
    pub fn deflated_sharpe_significance(
        observed_sharpe: f64,
        n_trials: usize,
        n_days: usize,
        sharpe_std: f64,
        periods_per_year: f64,
    ) -> f64 {
        let periods_per_year =
            if periods_per_year.is_finite() && periods_per_year > 0.0 { periods_per_year } else { 365.0 };
        if n_days == 0 || sharpe_std <= 0.0 {
            return 0.5;
        }
        let e_max = expected_max_normal_core(n_trials);
        let expected_max_sharpe = e_max * sharpe_std;
        let daily_obs = observed_sharpe / periods_per_year.sqrt();
        let daily_emax = expected_max_sharpe / periods_per_year.sqrt();
        let se = (1.0 / n_days as f64).sqrt();
        let t_dsr = (daily_obs - daily_emax) / se.max(1e-12);
        let dsr = 0.5 * erfc_as(-t_dsr / std::f64::consts::SQRT_2);
        const ERFC_SATURATION_ARG: f64 = 27.0;
        if (-t_dsr / std::f64::consts::SQRT_2).abs() >= ERFC_SATURATION_ARG {
            return dsr.clamp(1e-15, 1.0 - 1e-15);
        }
        dsr
    }

    /// Mirror of the private `normal_cdf` in `metrics::performance`. Its comment claims Abramowitz-Stegun 26.2.17 with
    /// 7.5e-8 maximum error, but the code applies the 7.1.26 `erf` polynomial to `x` instead of `x/sqrt(2)`: the true
    /// maximum absolute error is 0.037 (near `x = 0.57`), measured in `tests/core_equivalence.rs`. Reproduced here so
    /// the discrepancy is testable; the crate's own functions use [`crate::detmath::norm_cdf`].
    pub fn normal_cdf_approx(x: f64) -> f64 {
        const P: f64 = 0.3275911;
        const A: [f64; 5] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
        let t = 1.0 / (1.0 + P * x.abs());
        let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
        let erf_approx = 1.0 - poly * detmath::exp(-x * x / 2.0);
        0.5 * (1.0 + if x >= 0.0 { 1.0 } else { -1.0 } * erf_approx)
    }

    /// Mirror of `metrics::performance::deflated_sharpe_ratio(num_trials, returns)` (per-period Sharpe with the population
    /// standard deviation, moment-corrected variance floored at `1/(n-1)`, expected maximum `(1 - g + g ln N)/sqrt(n-1)`).
    pub fn deflated_sharpe_performance(num_trials: u32, returns: &[f64]) -> Option<f64> {
        deflated_sharpe_performance_z(num_trials, returns).map(normal_cdf_approx)
    }

    /// The standardised statistic `z` of the same function BEFORE Core applies its (inaccurate) normal cdf.
    pub fn deflated_sharpe_performance_z(num_trials: u32, returns: &[f64]) -> Option<f64> {
        let n = returns.len();
        if n < 4 || num_trials == 0 {
            return None;
        }
        let n_f = n as f64;
        let n_trials_f = num_trials as f64;
        let mean = returns.iter().sum::<f64>() / n_f;
        // Core's `simd_std_dev` is the POPULATION standard deviation (divides by n, not n-1); the Sharpe ratio and the
        // standardised moments of this function inherit that convention.
        let var = returns.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n_f;
        let std = var.sqrt();
        if std < 1e-10 {
            return None;
        }
        let sr = mean / std;
        let skewness = returns.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n_f;
        let excess_kurtosis = returns.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n_f - 3.0;
        const EULER_GAMMA: f64 = 0.5772156649015329;
        let sr_star = (1.0 - EULER_GAMMA + EULER_GAMMA * detmath::ln(n_trials_f)) / (n_f - 1.0).sqrt();
        let se_sq_normal_baseline = 1.0 / (n_f - 1.0);
        let se_sq = ((1.0 - skewness * sr + ((excess_kurtosis + 2.0) / 4.0) * sr * sr) / (n_f - 1.0))
            .max(se_sq_normal_baseline);
        Some((sr - sr_star) / se_sq.sqrt())
    }
}
