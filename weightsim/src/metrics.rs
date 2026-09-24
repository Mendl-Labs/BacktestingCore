//! Replication metrics with the ANSWER KEY's definitions (design 2.8), stamped `answer_key_v1`.
//!
//! These deliberately differ from the platform's general metrics (365 periods per year, population variance):
//!   * `years = (last_date - first_date).days / 365.25` over the counted returns
//!   * `ppy   = n_returns / years` (NOT 365 and NOT 252)
//!   * `std`  uses ddof = 1 (sample standard deviation)
//!   * `sharpe = mean / std * sqrt(ppy)`, NaN when `std == 0`
//!   * `vol    = std * sqrt(ppy)`
//!   * `cagr   = cum_final^(1/years) - 1`
//!   * max drawdown from `cum = cumprod(1 + r)` and `cum / cummax(cum) - 1`, with NO initial 1.0 point: a first-day
//!     loss is not a drawdown from 1.0. That quirk of the key (`shadow.py::metrics`) is reproduced on purpose.
//!
//! Only +, -, *, /, sqrt in sequential order are used, apart from `powf` in CAGR, which callers compare with tolerance.

use crate::date::Date;

/// Name recorded with every result that uses these definitions.
pub const METRIC_DEFINITIONS: &str = "answer_key_v1";

/// Metrics over a run of counted returns.
#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    pub n: usize,
    pub first_date: Date,
    pub last_date: Date,
    pub years: f64,
    pub ppy: f64,
    pub mean: f64,
    pub std_ddof1: f64,
    pub cagr: f64,
    pub vol: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub final_equity: f64,
}

/// Arithmetic mean, sequential left-to-right summation.
pub fn mean(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for &v in x {
        s += v;
    }
    s / x.len() as f64
}

/// Sample standard deviation (ddof = 1), two-pass, sequential. NaN for fewer than two observations.
pub fn std_ddof1(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let mut ss = 0.0;
    for &v in x {
        let d = v - m;
        ss += d * d;
    }
    (ss / (x.len() - 1) as f64).sqrt()
}

/// Nearest-rank percentile (`p` in (0, 1]): `sorted[ceil(p*n) - 1]`. NaN for empty input.
pub fn percentile_nearest_rank(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in percentile input"));
    let rank = (p * v.len() as f64).ceil() as usize;
    v[rank.clamp(1, v.len()) - 1]
}

/// Median (mean of the two middle values for even n). NaN for empty input.
pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in median input"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Cumulative product `cumprod(1 + r)` (no initial 1.0 point), sequential.
pub fn cumprod_one_plus(r: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(r.len());
    let mut c = 1.0;
    for &v in r {
        c *= 1.0 + v;
        out.push(c);
    }
    out
}

/// `answer_key_v1` metrics over `returns` dated by `dates` (same length, ascending). `None` for fewer than two
/// returns or a zero-length date span.
pub fn answer_key_metrics(dates: &[Date], returns: &[f64]) -> Option<Metrics> {
    assert_eq!(dates.len(), returns.len(), "dates and returns must align");
    if returns.len() < 2 {
        return None;
    }
    let first = dates[0];
    let last = dates[dates.len() - 1];
    let years = first.days_until(last) as f64 / 365.25;
    if !(years > 0.0) {
        return None;
    }
    let n = returns.len();
    let ppy = n as f64 / years;
    let m = mean(returns);
    let sd = std_ddof1(returns);
    let cum = cumprod_one_plus(returns);
    let final_equity = cum[cum.len() - 1];
    let cagr = final_equity.powf(1.0 / years) - 1.0;
    let vol = sd * ppy.sqrt();
    let sharpe = if sd > 0.0 { m / sd * ppy.sqrt() } else { f64::NAN };
    let mut peak = f64::NEG_INFINITY;
    let mut mdd = f64::INFINITY;
    for &c in &cum {
        if c > peak {
            peak = c;
        }
        let dd = c / peak - 1.0;
        if dd < mdd {
            mdd = dd;
        }
    }
    Some(Metrics {
        n,
        first_date: first,
        last_date: last,
        years,
        ppy,
        mean: m,
        std_ddof1: sd,
        cagr,
        vol,
        sharpe,
        max_drawdown: mdd,
        final_equity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    /// Hand-computed. r = [0.1, -0.1, 0.2, 0.0]: mean 0.05; deviations 0.05, -0.15, 0.15, -0.05 -> squares sum 0.05;
    /// var(ddof 1) = 0.05/3; cum = 1.1, 0.99, 1.188, 1.188; max drawdown = 0.99/1.1 - 1 = -0.1.
    /// Dates 2020-01-01 .. 2021-01-01 span 366 days -> years = 366/365.25, ppy = 4/years.
    #[test]
    fn definitions_match_hand_computation() {
        let dates = [d("2020-01-01"), d("2020-01-02"), d("2020-01-03"), d("2021-01-01")];
        let r = [0.1, -0.1, 0.2, 0.0];
        let m = answer_key_metrics(&dates, &r).unwrap();
        let years = 366.0 / 365.25;
        let ppy = 4.0 / years;
        let sd = (0.05f64 / 3.0).sqrt();
        assert!((m.years - years).abs() < 1e-15);
        assert!((m.ppy - ppy).abs() < 1e-12);
        assert!((m.mean - 0.05).abs() < 1e-15);
        assert!((m.std_ddof1 - sd).abs() < 1e-15);
        assert!((m.vol - sd * ppy.sqrt()).abs() < 1e-12);
        assert!((m.sharpe - 0.05 / sd * ppy.sqrt()).abs() < 1e-12);
        assert!((m.final_equity - 1.188).abs() < 1e-15);
        assert!((m.cagr - (1.188f64.powf(1.0 / years) - 1.0)).abs() < 1e-12);
        assert!((m.max_drawdown - (0.99 / 1.1 - 1.0)).abs() < 1e-15);
        assert_eq!(m.n, 4);
    }

    #[test]
    fn ppy_is_n_over_years_not_a_calendar_constant() {
        // 3 returns over exactly 730.5 days would be years = 2.0; use 731 days: ppy = 3 / (731/365.25).
        let dates = [d("2020-01-01"), d("2020-06-01"), d("2022-01-01")];
        let m = answer_key_metrics(&dates, &[0.01, 0.02, -0.01]).unwrap();
        assert!((m.ppy - 3.0 / (731.0 / 365.25)).abs() < 1e-12);
        assert!((m.ppy - 365.0).abs() > 300.0);
    }

    #[test]
    fn std_uses_ddof_one_not_population() {
        let x = [1.0, 2.0, 3.0, 4.0];
        // sum sq dev = 5.0 -> ddof1: sqrt(5/3); population would be sqrt(5/4).
        assert!((std_ddof1(&x) - (5.0f64 / 3.0).sqrt()).abs() < 1e-15);
        assert!((std_ddof1(&x) - (5.0f64 / 4.0).sqrt()).abs() > 0.1);
        assert!(std_ddof1(&[1.0]).is_nan());
    }

    #[test]
    fn max_drawdown_has_no_initial_one_point() {
        // A first-day loss of 5% followed by gains is NOT a drawdown in the key's definition.
        let dates = [d("2020-01-01"), d("2020-01-02"), d("2020-01-03")];
        let m = answer_key_metrics(&dates, &[-0.05, 0.10, 0.10]).unwrap();
        assert_eq!(m.max_drawdown, 0.0);
        // With an initial 1.0 point it would have been -0.05.
        let m2 = answer_key_metrics(&dates, &[0.10, -0.20, 0.05]).unwrap();
        assert!((m2.max_drawdown - (0.88 / 1.1 - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn zero_volatility_gives_nan_sharpe_and_degenerate_inputs_give_none() {
        let dates = [d("2020-01-01"), d("2020-01-02"), d("2020-01-03")];
        let m = answer_key_metrics(&dates, &[0.01, 0.01, 0.01]).unwrap();
        assert!(m.sharpe.is_nan());
        assert!(answer_key_metrics(&dates[..1], &[0.01]).is_none());
        assert!(answer_key_metrics(&[], &[]).is_none());
    }

    #[test]
    fn percentile_and_median_definitions() {
        let x = [5.0, 1.0, 3.0, 2.0, 4.0, 10.0, 9.0, 8.0, 7.0, 6.0];
        assert_eq!(median(&x), 5.5);
        assert_eq!(median(&x[..5]), 3.0);
        // nearest rank p90 of 10 values = 9th smallest = 9
        assert_eq!(percentile_nearest_rank(&x, 0.9), 9.0);
        assert_eq!(percentile_nearest_rank(&x, 1.0), 10.0);
        // 7 values: ceil(0.9 * 7) = ceil(6.3) = 7 -> the maximum (a floor would give the 6th value)
        assert_eq!(percentile_nearest_rank(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0.9), 7.0);
        assert!(median(&[]).is_nan());
    }
}
