//! Strategy significance testing.
//!
//! Provides Sharpe ratio t-statistics, p-values, the Deflated Sharpe Ratio
//! (Bailey & López de Prado, 2014), and a permutation-based p-value test.
//! These tools answer: *"Is this strategy genuinely profitable, or did we
//! just get lucky from multiple testing?"*

use crate::logging_facade::METRICS_LOGGER;
use crate::log_warn;

// ---------------------------------------------------------------------------
// Periods-per-year derivation
// ---------------------------------------------------------------------------

/// Periods-per-year implied by a candle granularity, e.g. `1440.0` (daily)
/// -> `365.0`, `60.0` (hourly) -> `8760.0`. This is the one place every
/// annualization factor in this codebase should ultimately derive from,
/// instead of a bare `252.0`/`365.0` literal chosen independent of what the
/// backtest actually traded.
///
/// Deliberately calendar-day-based (`365`, not the `252` trading-day
/// convention some equity literature uses) -- this platform trades crypto
/// and FX alongside equities, and a single shared constant that's
/// approximately right everywhere beats a precisely-right-for-equities
/// constant applied uniformly to markets that trade on different calendars.
///
/// This is a BAR-based derivation (periods = candles), not a trade-based one
/// -- the more precise fix would derive periods-per-year from actual trade
/// entry/exit cadence (a strategy holding multiple bars per trade trades less
/// often than one bar implies), but that requires trade-level timestamps
/// that aren't available at every call site this feeds. Confirmed live in
/// production (2026-08 win-rate investigation): every annualization call
/// site previously used a flat `252.0`/`365.0` regardless of
/// `candle_interval_minutes`, systematically understating the true
/// annualized Sharpe magnitude (positive or negative) for any strategy
/// trading faster than daily bars -- the already-catastrophic Sharpes found
/// that session on 60m/240m candles were likely even more extreme than
/// reported once correctly annualized.
pub fn periods_per_year_for_interval(candle_interval_minutes: f64) -> f64 {
    let interval = if candle_interval_minutes.is_finite() && candle_interval_minutes > 0.0 {
        candle_interval_minutes
    } else {
        1440.0 // daily fallback for a degenerate/missing interval
    };
    (365.0 * 24.0 * 60.0) / interval
}

/// Preferred periods-per-year derivation when a return series' actual first/
/// last observation timestamps are known: `num_observations / span_years`,
/// e.g. a strategy holding for ~2.8 days per trade over a year produces
/// ~130 trades/year, not `365`. Extracted from a pattern already proven in
/// `backtest::python_simulation::build_backtest_result` -- this is the
/// SAME formula, made reusable so every other annualization call site in
/// the codebase can derive from real trade cadence instead of guessing at a
/// flat 252/365 literal independent of how often the strategy actually
/// traded. Falls back to `365.0` when the span is zero/unknown or there are
/// no observations -- a strategy that hasn't traded enough to measure its
/// own cadence gets the same conservative default this codebase already
/// used everywhere before this fix.
pub fn observations_per_year_from_span(num_observations: usize, span_secs: Option<f64>) -> f64 {
    if num_observations == 0 {
        return 365.0;
    }
    match span_secs {
        Some(secs) if secs > 0.0 => {
            let years = secs / (365.25 * 86_400.0);
            (num_observations as f64 / years).max(1.0)
        }
        _ => 365.0,
    }
}

// ---------------------------------------------------------------------------
// Sharpe t-statistic & p-value
// ---------------------------------------------------------------------------

/// Compute the t-statistic for an annualised Sharpe ratio.
///
/// `t = sharpe * sqrt(n)` where `n` is the number of return observations at
/// whatever granularity `periods_per_year` describes. The Sharpe passed in
/// must have been annualised with `sqrt(periods_per_year)` -- the caller and
/// this function must agree on the same factor, or the de-annualisation step
/// below silently uses the wrong divisor.
pub fn sharpe_t_stat(annualised_sharpe: f64, n_days: usize, periods_per_year: f64) -> f64 {
    let periods_per_year = if periods_per_year.is_finite() && periods_per_year > 0.0 {
        periods_per_year
    } else {
        365.0
    };
    let period_sharpe = annualised_sharpe / periods_per_year.sqrt();
    period_sharpe * (n_days as f64).sqrt()
}

/// One-sided p-value for a Sharpe t-statistic under H₀: true Sharpe ≤ 0.
///
/// Uses the normal approximation (valid for n > 30).
pub fn sharpe_p_value(t: f64) -> f64 {
    // P(Z > t) = 0.5 * erfc(t / sqrt(2))
    0.5 * erfc(t / std::f64::consts::SQRT_2)
}

/// Compute both the t-statistic and p-value in one call.
pub fn sharpe_significance(annualised_sharpe: f64, n_days: usize, periods_per_year: f64) -> (f64, f64) {
    let t = sharpe_t_stat(annualised_sharpe, n_days, periods_per_year);
    let p = sharpe_p_value(t);
    (t, p)
}

// ---------------------------------------------------------------------------
// Deflated Sharpe Ratio (DSR)
// ---------------------------------------------------------------------------

/// Deflated Sharpe Ratio — adjusts an observed Sharpe for multiple testing.
///
/// Bailey et al. (2014): accounts for the number of strategy trials so that
/// the probability of observing `observed_sharpe` by chance is correctly
/// estimated.
///
/// # Arguments
/// * `observed_sharpe` — annualised Sharpe of the selected strategy.
/// * `n_trials` — total number of strategy/parameter variants tested.
/// * `n_days` — number of return observations, at whatever granularity
///   `periods_per_year` describes (not necessarily literal calendar days).
/// * `sharpe_std` — standard deviation of Sharpe ratios across all trials
///   (if unknown, a reasonable default is 1.0).
/// * `periods_per_year` — must match whatever factor `observed_sharpe` was
///   annualised with (see `periods_per_year_for_interval`) -- a mismatch
///   here silently de-annualises with the wrong divisor.
///
/// # Returns
/// The probability that the observed Sharpe exceeds what we'd expect from
/// the best of `n_trials` draws under the null.  Values close to 1.0 mean
/// the strategy is likely genuine; values close to 0.0 mean it's likely
/// due to overfitting / chance.
pub fn deflated_sharpe_ratio(
    observed_sharpe: f64,
    n_trials: usize,
    n_days: usize,
    sharpe_std: f64,
    periods_per_year: f64,
) -> f64 {
    let periods_per_year = if periods_per_year.is_finite() && periods_per_year > 0.0 {
        periods_per_year
    } else {
        365.0
    };
    // n_days == 0 / sharpe_std <= 0.0 are genuine "cannot compute at all"
    // cases (division by zero) -- 0.5 (neutral: no basis to judge either way)
    // is the honest answer, not 1.0 ("definitely not overfit"). This
    // previously also short-circuited whenever n_trials <= 1, on the
    // reasoning that a single trial needs no multiple-testing correction --
    // true, but the old code skipped straight to the MAXIMUM possible
    // confidence score regardless of what observed_sharpe actually was,
    // so a strategy with a negative Sharpe (or zero trades and sharpe=0.0,
    // called from python_simulation.rs's `deflated_sharpe_ratio(s, 1, n, 1.0)`
    // with n_trials hardcoded to 1) still got reported as DSR=1.0, "certainly
    // genuine". `expected_max_normal(n<=1)` already correctly returns 0.0 (no
    // deflation needed), so simply not special-casing n_trials <= 1 lets the
    // formula below fall through to an honest, undeflated significance test
    // of the real observed_sharpe instead.
    if n_days == 0 || sharpe_std <= 0.0 {
        return 0.5;
    }

    // Expected maximum Sharpe under null (Expected max of n_trials iid N(0,1))
    let e_max = expected_max_normal(n_trials);
    // Scale to Sharpe space
    let expected_max_sharpe = e_max * sharpe_std;

    // De-annualise observed Sharpe for the t-test
    let daily_obs = observed_sharpe / periods_per_year.sqrt();
    let daily_emax = expected_max_sharpe / periods_per_year.sqrt();

    let se = (1.0 / n_days as f64).sqrt(); // standard error of daily sharpe
    let t_dsr = (daily_obs - daily_emax) / se.max(1e-12);

    // Prob(observed > expected_max): Φ(t_dsr)
    let dsr = 0.5 * erfc(-t_dsr / std::f64::consts::SQRT_2);

    // `erfc`'s `(-a*a).exp()` term underflows to an exact 0.0 once `a`
    // exceeds ~27.3 -- not an artifact of this particular polynomial
    // approximation, but a hard f64 representational limit (no linear-scale
    // f64 can hold a value below ~4.9e-324, and exp(-27.3^2) is already past
    // that floor). Past this point `dsr` is an exact 0.0 or 1.0 regardless of
    // how much further `|t_dsr|` grows, silently discarding any real
    // magnitude information. This is expected/harmless for a genuinely
    // overwhelming, real edge (any threshold check treats 1.0 and
    // 1 - 1e-300 identically) -- but it is also exactly the failure mode
    // that let CPCV's fold-refit instability (see worker.rs's CPCV override
    // sanity gate) produce an indistinguishable-from-perfect DSR out of a
    // wildly unstable input. Clamp strictly inside (0, 1) and log so a
    // saturated value is at least visible/traceable rather than silently
    // read as literal certainty, and doesn't collide in exact-equality
    // comparisons with the legitimate `n_trials<=1`/no-deflation DSR=1.0 case.
    const ERFC_SATURATION_ARG: f64 = 27.0;
    if (-t_dsr / std::f64::consts::SQRT_2).abs() >= ERFC_SATURATION_ARG {
        log_warn!(
            METRICS_LOGGER,
            "[DSR] erfc saturated at |t_dsr|={:.2} (observed_sharpe={:.4}, n_trials={}, n_days={}); \
             clamping to avoid a deceptive exact 0.0/1.0",
            t_dsr, observed_sharpe, n_trials, n_days
        );
        return dsr.clamp(1e-15, 1.0 - 1e-15);
    }
    dsr
}

// ---------------------------------------------------------------------------
// Permutation test
// ---------------------------------------------------------------------------

/// Empirical permutation p-value.
///
/// `shuffled_sharpes` should be the Sharpe ratios obtained from many
/// random-shuffle (or circular-block bootstrap) permutations of the original
/// return series.  The p-value is the fraction of shuffled Sharpes that
/// meet or exceed the observed one.
pub fn permutation_test_pvalue(observed_sharpe: f64, shuffled_sharpes: &[f64]) -> f64 {
    if shuffled_sharpes.is_empty() {
        return 1.0;
    }
    let count = shuffled_sharpes
        .iter()
        .filter(|&&s| s >= observed_sharpe)
        .count();
    (count as f64 + 1.0) / (shuffled_sharpes.len() as f64 + 1.0) // +1 correction
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expected maximum of `n` independent standard-normal draws.
///
/// Approximation: E[max] ≈ Φ⁻¹(1 - 1/n) for large n, with a small-n
/// correction via the Euler–Mascheroni constant.
fn expected_max_normal(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    // Approximation from extreme-value theory
    let z = probit(1.0 - 1.0 / n_f);
    // Gumbel correction: E[max] ≈ z_n + γ / (z_n * sqrt(2 * ln(n)))
    let euler_mascheroni = 0.5772156649;
    let ln_n = n_f.ln();
    if ln_n > 0.0 && z.abs() > 1e-12 {
        z + euler_mascheroni / (z * (2.0 * ln_n).sqrt())
    } else {
        z
    }
}

/// Complementary error function (Abramowitz & Stegun 7.1.26 approximation).
fn erfc(x: f64) -> f64 {
    // For negative x: erfc(-x) = 2 - erfc(x)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();

    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741
                + t * (-1.453152027
                    + t * 1.061405429))));
    let result = poly * (-a * a).exp();

    if sign < 0.0 { 2.0 - result } else { result }
}

/// Probit function (inverse of the standard normal CDF).
///
/// Rational approximation accurate to ~4.5 × 10⁻⁴ (Peter Acklam).
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Coefficients (Acklam)
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
    const D: [f64; 4] = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
        / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0]*r + A[1])*r + A[2])*r + A[3])*r + A[4])*r + A[5]) * q
        / (((((B[0]*r + B[1])*r + B[2])*r + B[3])*r + B[4])*r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
        / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_t_stat() {
        // Sharpe 1.5, 365 days → daily_sharpe ≈ 0.0785, t ≈ 1.5
        let t = sharpe_t_stat(1.5, 365, 365.0);
        assert!((t - 1.5).abs() < 0.01, "t = {}", t);
    }

    #[test]
    fn test_p_value_positive_sharpe() {
        let (t, p) = sharpe_significance(2.0, 500, 365.0);
        assert!(t > 1.0);
        assert!(p < 0.10, "p = {}", p);
    }

    #[test]
    fn test_p_value_zero_sharpe() {
        let (_, p) = sharpe_significance(0.0, 365, 365.0);
        assert!((p - 0.5).abs() < 0.05, "p = {}", p);
    }

    // Regression guard: a single trial (or the hardcoded n_trials=1 call site
    // in python_simulation.rs) must not report DSR=1.0 ("certainly genuine")
    // regardless of how bad observed_sharpe actually is -- that was a real,
    // production-observed bug (job d603dab0: 0 trades, sharpe=0.0, reported
    // DSR=1.0; job 0aea4aad: sharpe=-5.6, also reported DSR=1.0).
    #[test]
    fn test_dsr_with_one_trial_reflects_the_observed_sharpe_not_a_fixed_1_0() {
        let dsr_good = deflated_sharpe_ratio(2.0, 1, 365, 1.0, 365.0);
        let dsr_bad = deflated_sharpe_ratio(-5.6, 1, 365, 1.0, 365.0);
        let dsr_zero = deflated_sharpe_ratio(0.0, 1, 365, 1.0, 365.0);
        assert!(dsr_good > 0.9, "a strong single-trial Sharpe should score high: {dsr_good}");
        assert!(dsr_bad < 0.1, "a bad single-trial Sharpe must not score 1.0: {dsr_bad}");
        assert!((dsr_zero - 0.5).abs() < 0.05, "a zero Sharpe is a coin flip, not certainty: {dsr_zero}");
    }

    #[test]
    fn test_dsr_genuinely_uncomputable_inputs_return_a_neutral_0_5_not_1_0() {
        assert_eq!(deflated_sharpe_ratio(3.0, 10, 0, 1.0, 365.0), 0.5);
        assert_eq!(deflated_sharpe_ratio(3.0, 10, 365, 0.0, 365.0), 0.5);
    }

    #[test]
    fn test_dsr_reduces_with_more_trials() {
        let dsr_few = deflated_sharpe_ratio(1.5, 5, 365, 1.0, 365.0);
        let dsr_many = deflated_sharpe_ratio(1.5, 100, 365, 1.0, 365.0);
        assert!(
            dsr_many < dsr_few,
            "more trials should lower DSR: few={:.3}, many={:.3}",
            dsr_few,
            dsr_many
        );
    }

    // -- periods_per_year_for_interval --------------------------------------

    #[test]
    fn periods_per_year_for_interval_daily_is_365() {
        assert!((periods_per_year_for_interval(1440.0) - 365.0).abs() < 1e-6);
    }

    #[test]
    fn periods_per_year_for_interval_hourly_is_8760() {
        assert!((periods_per_year_for_interval(60.0) - 8760.0).abs() < 1e-6);
    }

    #[test]
    fn periods_per_year_for_interval_degenerate_input_falls_back_to_daily() {
        assert!((periods_per_year_for_interval(0.0) - 365.0).abs() < 1e-6);
        assert!((periods_per_year_for_interval(-5.0) - 365.0).abs() < 1e-6);
        assert!((periods_per_year_for_interval(f64::NAN) - 365.0).abs() < 1e-6);
    }

    // -- observations_per_year_from_span --------------------------------------

    #[test]
    fn observations_per_year_from_span_derives_real_cadence() {
        // 130 trades over exactly one year -> ~130 observations/year, not 365.
        let one_year_secs = 365.25 * 86_400.0;
        let opy = observations_per_year_from_span(130, Some(one_year_secs));
        assert!((opy - 130.0).abs() < 1.0, "opy = {opy}");
    }

    #[test]
    fn observations_per_year_from_span_handles_sub_year_span() {
        // 40 trades over ~90 days -> roughly 40 * (365/90) ~= 162/year.
        let ninety_days_secs = 90.0 * 86_400.0;
        let opy = observations_per_year_from_span(40, Some(ninety_days_secs));
        assert!((opy - 162.0).abs() < 5.0, "opy = {opy}");
    }

    #[test]
    fn observations_per_year_from_span_falls_back_to_365_when_span_unknown_or_zero() {
        assert_eq!(observations_per_year_from_span(50, None), 365.0);
        assert_eq!(observations_per_year_from_span(50, Some(0.0)), 365.0);
        assert_eq!(observations_per_year_from_span(50, Some(-100.0)), 365.0);
    }

    #[test]
    fn observations_per_year_from_span_falls_back_to_365_with_zero_observations() {
        assert_eq!(observations_per_year_from_span(0, Some(1000.0)), 365.0);
    }

    #[test]
    fn observations_per_year_from_span_never_returns_below_one() {
        // A huge span with very few observations shouldn't collapse toward
        // zero -- floored at 1 observation/year.
        let ten_years_secs = 10.0 * 365.25 * 86_400.0;
        let opy = observations_per_year_from_span(1, Some(ten_years_secs));
        assert!(opy >= 1.0, "opy = {opy}");
    }

    // Regression guard: using the WRONG periods_per_year (e.g. a flat 252/365
    // applied regardless of actual candle granularity) silently mis-scales
    // the t-statistic/DSR. This confirms the same annualised Sharpe produces
    // a materially different t-stat depending on which periods_per_year is
    // supplied -- callers must pass the value matching how the Sharpe was
    // actually annualised, not a hardcoded literal independent of the data.
    #[test]
    fn sharpe_t_stat_differs_materially_between_daily_and_hourly_periods_per_year() {
        let t_daily = sharpe_t_stat(1.5, 200, periods_per_year_for_interval(1440.0));
        let t_hourly = sharpe_t_stat(1.5, 200, periods_per_year_for_interval(60.0));
        assert!(
            t_daily > t_hourly * 3.0,
            "t-stats should diverge materially: daily={t_daily}, hourly={t_hourly}"
        );
    }

    #[test]
    fn test_permutation_pvalue() {
        let shuffled = vec![0.5, 0.8, 1.0, 1.2, 0.3, 0.7];
        let p = permutation_test_pvalue(1.5, &shuffled);
        // None of the shuffled values >= 1.5, so p ≈ 1/7 ≈ 0.143
        assert!(p < 0.25, "p = {}", p);
    }

    #[test]
    fn test_erfc_values() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-6);
        assert!(erfc(3.0) < 0.001);
        assert!((erfc(-3.0) - 2.0 + erfc(3.0)).abs() < 0.01);
    }

    #[test]
    fn test_probit_symmetry() {
        let z = probit(0.975);
        assert!((z - 1.96).abs() < 0.01, "z = {}", z);
        let z2 = probit(0.025);
        assert!((z2 + 1.96).abs() < 0.01, "z2 = {}", z2);
    }
}
