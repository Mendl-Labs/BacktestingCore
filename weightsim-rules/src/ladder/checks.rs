//! The certification checks, exactly as pre-registration Amendment 11 defines them (design 1.4). Pure functions on
//! plain series; no I/O.
//!
//! * **Tier I** (the pre-registered bands, kept as the floor): return correlation >= 0.99, |dSharpe| <= 0.05,
//!   |dCAGR| <= 0.5 percentage points, all on the days common to the run and the key with the KEY's metric definitions
//!   (`weightsim::answer_key_metrics`); plus asset-level trades (signal flips) within 5% of the key's counter.
//! * **Tier II** (per-bar identity, the binding test): max |ret_run - ret_key| <= 1e-9, and, where the series exist,
//!   equity, cost, traded notional, standing target weights and start-of-bar held weights all within 1e-9.
//! * **Tier III** (weight agreement): the fraction of (bar, asset) cells with |w_run - w_key| <= 1e-6 is >= 0.98.
//!   A floor, not a discriminator (a one-bar-late monthly rule agrees on 99.1% of cells).
//!
//! A run is compared with the key on the days both have; a base certification additionally demands that the run
//! covers exactly the key's bars (see `crate::ladder`).

use weightsim::{answer_key_metrics, Date};

use super::fixtures::LadderError;

pub const TIER1_CORR_MIN: f64 = 0.99;
pub const TIER1_D_SHARPE_MAX: f64 = 0.05;
pub const TIER1_D_CAGR_PP_MAX: f64 = 0.5;
/// Trades (signal flips) must be within this fraction of the key's counter.
pub const TIER1_TRADES_REL_MAX: f64 = 0.05;
pub const TIER2_TOL: f64 = 1e-9;
pub const TIER3_CELL_TOL: f64 = 1e-6;
pub const TIER3_MIN_AGREEMENT: f64 = 0.98;

pub const CAUGHT_TIER1: &str = "tier1_bands";
pub const CAUGHT_TIER2: &str = "tier2_identity";
pub const CAUGHT_TIER3: &str = "tier3_weights";

/// Column-oriented per-bar series, dated by the bar the return is EARNED on. Optional columns are `None` where a
/// side has no such series (mutants compared on returns and weights only, the drifting-sub-accounts mutant on returns).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeriesRows {
    pub dates: Vec<Date>,
    pub ret: Vec<f64>,
    pub equity: Option<Vec<f64>>,
    pub cost: Option<Vec<f64>>,
    pub traded: Option<Vec<f64>>,
    /// `[bar][asset]`
    pub w_target: Option<Vec<Vec<f64>>>,
    /// `[bar][asset]`
    pub w_held: Option<Vec<Vec<f64>>>,
}

/// Tier III numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tier3 {
    pub cells: usize,
    pub disagreeing: usize,
    pub agreement: f64,
    pub pass: bool,
}

/// Every number of a run-versus-key comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub key_days: usize,
    pub run_days: usize,
    pub common_days: usize,
    // Tier I
    pub corr: f64,
    pub key_sharpe: f64,
    pub run_sharpe: f64,
    pub d_sharpe: f64,
    pub key_cagr: f64,
    pub run_cagr: f64,
    pub d_cagr_pp: f64,
    pub corr_ok: bool,
    pub d_sharpe_ok: bool,
    pub d_cagr_ok: bool,
    /// All three return bands.
    pub bands_pass: bool,
    // Tier II
    pub max_abs_ret_diff: f64,
    pub max_abs_equity_diff: Option<f64>,
    pub max_abs_cost_diff: Option<f64>,
    pub max_abs_traded_diff: Option<f64>,
    pub max_abs_w_target_diff: Option<f64>,
    pub max_abs_w_held_diff: Option<f64>,
    pub tier2_pass: bool,
    // Tier III
    pub tier3: Option<Tier3>,
}

impl Comparison {
    /// Names of the tiers this comparison FAILS (`tier1_bands`, `tier2_identity`, `tier3_weights`), the vocabulary of
    /// `mutants.json`'s `caught_by`.
    pub fn failed_tiers(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.bands_pass {
            out.push(CAUGHT_TIER1);
        }
        if !self.tier2_pass {
            out.push(CAUGHT_TIER2);
        }
        if let Some(t3) = &self.tier3 {
            if !t3.pass {
                out.push(CAUGHT_TIER3);
            }
        }
        out
    }
    /// The run reproduces the key over exactly the key's bars.
    pub fn covers_key_exactly(&self) -> bool {
        self.common_days == self.key_days && self.common_days == self.run_days
    }
    pub fn tier3_pass(&self) -> bool {
        self.tier3.as_ref().is_none_or(|t| t.pass)
    }
}

/// Pearson correlation (sequential sums). NaN when either series is constant.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..x.len() {
        sx += x[i];
        sy += y[i];
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (a, b) = (x[i] - mx, y[i] - my);
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    }
}

/// Indices `(key_i, run_i)` of the dates both series have (both ascending).
fn common_indices(a: &[Date], b: &[Date]) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn max_abs_diff_by(pairs: &[(usize, usize)], a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &(i, j) in pairs {
        let d = (a[i] - b[j]).abs();
        // NaN must poison the maximum (a NaN difference is a failure, never "small").
        if d.is_nan() {
            return f64::NAN;
        }
        if d > m {
            m = d;
        }
    }
    m
}

fn max_abs_diff_rows(pairs: &[(usize, usize)], a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0f64;
    for &(i, j) in pairs {
        for (x, y) in a[i].iter().zip(&b[j]) {
            let d = (x - y).abs();
            if d.is_nan() {
                return f64::NAN;
            }
            if d > m {
                m = d;
            }
        }
    }
    m
}

fn opt_pair<'a, T>(k: &'a Option<T>, r: &'a Option<T>) -> Option<(&'a T, &'a T)> {
    match (k, r) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

/// `true` when `d <= tol` and `d` is a number.
fn within(d: f64, tol: f64) -> bool {
    d <= tol
}

/// Compare a run with a key on the days they have in common.
pub fn compare(key: &SeriesRows, run: &SeriesRows) -> Result<Comparison, LadderError> {
    let pairs = common_indices(&key.dates, &run.dates);
    if pairs.len() < 3 {
        return Err(LadderError::Inconsistent(format!("only {} common days between run and key", pairs.len())));
    }
    let dates: Vec<Date> = pairs.iter().map(|&(i, _)| key.dates[i]).collect();
    let kr: Vec<f64> = pairs.iter().map(|&(i, _)| key.ret[i]).collect();
    let rr: Vec<f64> = pairs.iter().map(|&(_, j)| run.ret[j]).collect();
    let km = answer_key_metrics(&dates, &kr)
        .ok_or_else(|| LadderError::Inconsistent("key metrics undefined on the common days".into()))?;
    let rm = answer_key_metrics(&dates, &rr)
        .ok_or_else(|| LadderError::Inconsistent("run metrics undefined on the common days".into()))?;
    let corr = pearson(&kr, &rr);
    let d_sharpe = rm.sharpe - km.sharpe;
    let d_cagr_pp = 100.0 * (rm.cagr - km.cagr);
    let corr_ok = corr >= TIER1_CORR_MIN;
    let d_sharpe_ok = d_sharpe.abs() <= TIER1_D_SHARPE_MAX;
    let d_cagr_ok = d_cagr_pp.abs() <= TIER1_D_CAGR_PP_MAX;

    let max_abs_ret_diff = max_abs_diff_by(&pairs, &key.ret, &run.ret);
    let max_abs_equity_diff = opt_pair(&key.equity, &run.equity).map(|(k, r)| max_abs_diff_by(&pairs, k, r));
    let max_abs_cost_diff = opt_pair(&key.cost, &run.cost).map(|(k, r)| max_abs_diff_by(&pairs, k, r));
    let max_abs_traded_diff = opt_pair(&key.traded, &run.traded).map(|(k, r)| max_abs_diff_by(&pairs, k, r));
    let max_abs_w_target_diff = opt_pair(&key.w_target, &run.w_target).map(|(k, r)| max_abs_diff_rows(&pairs, k, r));
    let max_abs_w_held_diff = opt_pair(&key.w_held, &run.w_held).map(|(k, r)| max_abs_diff_rows(&pairs, k, r));

    let mut tier2_pass = within(max_abs_ret_diff, TIER2_TOL);
    for d in [max_abs_equity_diff, max_abs_cost_diff, max_abs_traded_diff, max_abs_w_target_diff, max_abs_w_held_diff]
        .into_iter()
        .flatten()
    {
        tier2_pass = tier2_pass && within(d, TIER2_TOL);
    }

    let tier3 = opt_pair(&key.w_target, &run.w_target).map(|(k, r)| {
        let mut cells = 0usize;
        let mut ok = 0usize;
        for &(i, j) in &pairs {
            for (x, y) in k[i].iter().zip(&r[j]) {
                cells += 1;
                if (x - y).abs() <= TIER3_CELL_TOL {
                    ok += 1;
                }
            }
        }
        let agreement = ok as f64 / cells as f64;
        Tier3 { cells, disagreeing: cells - ok, agreement, pass: agreement >= TIER3_MIN_AGREEMENT }
    });

    Ok(Comparison {
        key_days: key.dates.len(),
        run_days: run.dates.len(),
        common_days: pairs.len(),
        corr,
        key_sharpe: km.sharpe,
        run_sharpe: rm.sharpe,
        d_sharpe,
        key_cagr: km.cagr,
        run_cagr: rm.cagr,
        d_cagr_pp,
        corr_ok,
        d_sharpe_ok,
        d_cagr_ok,
        bands_pass: corr_ok && d_sharpe_ok && d_cagr_ok,
        max_abs_ret_diff,
        max_abs_equity_diff,
        max_abs_cost_diff,
        max_abs_traded_diff,
        max_abs_w_target_diff,
        max_abs_w_held_diff,
        tier2_pass,
        tier3,
    })
}

/// Trades check: the run's signal flips are within 5% of the key's counter. Returns `(relative gap, ok)`.
pub fn trades_within_band(run_flips: u64, key_flips: u64) -> (f64, bool) {
    if key_flips == 0 {
        return (if run_flips == 0 { 0.0 } else { f64::INFINITY }, run_flips == 0);
    }
    let rel = (run_flips as f64 - key_flips as f64).abs() / key_flips as f64;
    (rel, rel <= TIER1_TRADES_REL_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    fn dates(n: usize) -> Vec<Date> {
        (0..n).map(|i| Date::from_days_since_epoch(18_000 + i as i64)).collect()
    }

    /// A deterministic, non-constant return series.
    fn rets(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.001 * (((i * 7919) % 23) as f64 - 11.0) / 11.0).collect()
    }

    fn rows(ret: Vec<f64>) -> SeriesRows {
        SeriesRows { dates: dates(ret.len()), ret, ..SeriesRows::default() }
    }

    #[test]
    fn identical_series_pass_everything() {
        let r = rets(300);
        let c = compare(&rows(r.clone()), &rows(r)).unwrap();
        assert!(c.bands_pass && c.tier2_pass && c.tier3.is_none());
        assert_eq!((c.key_days, c.run_days, c.common_days), (300, 300, 300));
        assert!(c.covers_key_exactly());
        assert_eq!(c.max_abs_ret_diff, 0.0);
        assert!((c.corr - 1.0).abs() < 1e-12);
        assert!(c.failed_tiers().is_empty());
    }

    #[test]
    fn tier2_tolerance_is_one_e_minus_nine_inclusive() {
        let r = rets(300);
        let mut ok = r.clone();
        ok[100] += 5e-10;
        let mut bad = r.clone();
        bad[100] += 2e-9;
        let c_ok = compare(&rows(r.clone()), &rows(ok)).unwrap();
        let c_bad = compare(&rows(r), &rows(bad)).unwrap();
        assert!(c_ok.tier2_pass);
        assert!(!c_bad.tier2_pass);
        // A 2e-9 error is invisible to the Tier I bands: only Tier II sees it.
        assert!(c_bad.bands_pass);
        assert_eq!(c_bad.failed_tiers(), vec![CAUGHT_TIER2]);
    }

    #[test]
    fn nan_difference_fails_tier2() {
        let r = rets(50);
        let mut nan = r.clone();
        nan[10] = f64::NAN;
        // NaN in the run poisons the metrics as well, so the comparison must fail somewhere, never pass.
        let c = compare(&rows(r), &rows(nan)).unwrap();
        assert!(!c.tier2_pass && !c.bands_pass && c.max_abs_ret_diff.is_nan());
    }

    #[test]
    fn tier1_bands_boundaries() {
        let r = rets(400);
        // Scale the run's returns: correlation stays 1, CAGR and Sharpe move.
        let scaled: Vec<f64> = r.iter().map(|x| x * 1.5).collect();
        let c = compare(&rows(r.clone()), &rows(scaled)).unwrap();
        assert!(c.corr_ok);
        assert!(!c.d_cagr_ok || !c.d_sharpe_ok || c.bands_pass);
        // A decorrelated run fails the correlation band.
        let other = rets(400).iter().rev().copied().collect::<Vec<_>>();
        let c2 = compare(&rows(r), &rows(other)).unwrap();
        assert!(!c2.corr_ok && !c2.bands_pass);
        assert!(c2.failed_tiers().contains(&CAUGHT_TIER1));
    }

    #[test]
    fn common_days_only_and_partial_windows() {
        let r = rets(300);
        let key = rows(r.clone());
        // Run starts 20 bars later.
        let run = SeriesRows { dates: dates(300)[20..].to_vec(), ret: r[20..].to_vec(), ..SeriesRows::default() };
        let c = compare(&key, &run).unwrap();
        assert_eq!((c.key_days, c.run_days, c.common_days), (300, 280, 280));
        assert!(!c.covers_key_exactly());
        assert!(c.tier2_pass, "identical on the common days");
    }

    #[test]
    fn too_few_common_days_is_an_error() {
        let r = rets(10);
        let key = rows(r.clone());
        let run = SeriesRows { dates: dates(10)[8..].to_vec(), ret: r[8..].to_vec(), ..SeriesRows::default() };
        assert!(compare(&key, &run).is_err());
    }

    #[test]
    fn weights_tier2_and_tier3() {
        let r = rets(200);
        let wk: Vec<Vec<f64>> = (0..200).map(|i| vec![0.2 * ((i / 10) % 2) as f64, 0.2]).collect();
        let mut key = rows(r.clone());
        key.w_target = Some(wk.clone());
        key.w_held = Some(wk.clone());
        let mut run = key.clone();
        assert!(compare(&key, &run).unwrap().tier2_pass);
        // One cell off by 1e-7: Tier III still agrees (<= 1e-6) but Tier II fails (> 1e-9).
        run.w_target.as_mut().unwrap()[50][0] += 1e-7;
        let c = compare(&key, &run).unwrap();
        assert!(!c.tier2_pass);
        assert!(c.tier3.unwrap().pass);
        assert_eq!(c.tier3.unwrap().disagreeing, 0);
        assert_eq!(c.failed_tiers(), vec![CAUGHT_TIER2]);
        // Held weights count in Tier II too.
        let mut run2 = key.clone();
        run2.w_held.as_mut().unwrap()[7][1] += 1e-8;
        let c2 = compare(&key, &run2).unwrap();
        assert!(!c2.tier2_pass);
        assert!(c2.max_abs_w_target_diff.unwrap() == 0.0 && c2.max_abs_w_held_diff.unwrap() > 9e-9);
        // 5% of the cells wrong by 0.2: Tier III fails (agreement 0.95 < 0.98).
        let mut run3 = key.clone();
        for row in run3.w_target.as_mut().unwrap().iter_mut().take(20) {
            row[0] += 0.2;
            row[1] += 0.2;
        }
        let c3 = compare(&key, &run3).unwrap();
        let t3 = c3.tier3.unwrap();
        assert_eq!((t3.cells, t3.disagreeing), (400, 40));
        assert!((t3.agreement - 0.9).abs() < 1e-12 && !t3.pass);
        assert!(c3.failed_tiers().contains(&CAUGHT_TIER3));
    }

    #[test]
    fn tier3_agreement_boundary_is_inclusive_at_98_percent() {
        let r = rets(50);
        let mut key = rows(r);
        key.w_target = Some(vec![vec![0.5, 0.5]; 50]);
        let mut run = key.clone();
        // 100 cells; break exactly 2 -> 0.98 (pass), 3 -> 0.97 (fail).
        run.w_target.as_mut().unwrap()[0][0] = 0.0;
        run.w_target.as_mut().unwrap()[1][0] = 0.0;
        assert!(compare(&key, &run).unwrap().tier3.unwrap().pass);
        run.w_target.as_mut().unwrap()[2][0] = 0.0;
        assert!(!compare(&key, &run).unwrap().tier3.unwrap().pass);
    }

    #[test]
    fn equity_cost_and_traded_columns_are_compared_when_present() {
        let r = rets(100);
        let mut key = rows(r);
        key.equity = Some((0..100).map(|i| 1.0 + i as f64 * 0.01).collect());
        key.cost = Some(vec![1e-4; 100]);
        key.traded = Some(vec![0.1; 100]);
        let mut run = key.clone();
        assert!(compare(&key, &run).unwrap().tier2_pass);
        run.equity.as_mut().unwrap()[5] += 1e-8;
        assert!(!compare(&key, &run).unwrap().tier2_pass);
        let mut run = key.clone();
        run.cost.as_mut().unwrap()[5] += 1e-8;
        assert!(!compare(&key, &run).unwrap().tier2_pass);
        let mut run = key.clone();
        run.traded.as_mut().unwrap()[5] += 1e-8;
        assert!(!compare(&key, &run).unwrap().tier2_pass);
    }

    #[test]
    fn trades_band_is_five_percent_inclusive() {
        assert_eq!(trades_within_band(99, 99), (0.0, true));
        assert!(trades_within_band(103, 99).1); // 4.04%
        assert!(!trades_within_band(104, 99).1); // 5.05%
        assert!(trades_within_band(105, 100).1); // exactly 5%: inclusive
        assert!(!trades_within_band(106, 100).1);
        assert!(trades_within_band(95, 100).1);
        assert!(!trades_within_band(94, 100).1);
        assert!(trades_within_band(0, 0).1);
        assert!(!trades_within_band(1, 0).1);
    }

    /// Bisection on a monotone-increasing `f` for `f(x) = target`.
    fn solve(mut lo: f64, mut hi: f64, target: f64, f: impl Fn(f64) -> f64) -> f64 {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Noisy, positive-drift daily returns with a sample std of about `sd`.
    fn noisy(n: usize, sd: f64) -> Vec<f64> {
        (0..n).map(|i| 0.0003 + sd * (((i * 7919 + 13) % 101) as f64 - 50.0) / 29.0).collect()
    }

    #[test]
    fn tier1_sharpe_band_edges_are_exact() {
        let key = noisy(600, 0.003);
        let ds = dates(600);
        let m = answer_key_metrics(&ds, &key).unwrap();
        // a constant shift moves the mean and nothing else: dSharpe = shift / std * sqrt(ppy)
        for (target, ok) in [(0.0499, true), (0.0501, false), (-0.0499, true), (-0.0501, false)] {
            let shift = target * m.std_ddof1 / m.ppy.sqrt();
            let run: Vec<f64> = key.iter().map(|r| r + shift).collect();
            let c = compare(&rows(key.clone()), &rows(run)).unwrap();
            assert!((c.d_sharpe - target).abs() < 1e-9, "dSharpe {} for target {target}", c.d_sharpe);
            assert_eq!(c.d_sharpe_ok, ok, "dSharpe {target}");
            assert!(c.corr_ok && c.d_cagr_ok, "only the Sharpe band should be in play here (dCAGR {} pp)", c.d_cagr_pp);
            assert_eq!(c.bands_pass, ok);
        }
    }

    #[test]
    fn tier1_cagr_band_edges_are_exact_and_in_percentage_points() {
        let key = noisy(600, 0.003);
        let ds = dates(600);
        let base = answer_key_metrics(&ds, &key).unwrap().cagr;
        // scaling the returns leaves correlation and Sharpe untouched and moves CAGR
        for (target_pp, ok) in [(0.499, true), (0.501, false), (-0.499, true), (-0.501, false)] {
            let k = solve(0.0, 3.0, 0.0, |k| {
                let run: Vec<f64> = key.iter().map(|r| r * k).collect();
                100.0 * (answer_key_metrics(&ds, &run).unwrap().cagr - base) - target_pp
            });
            let run: Vec<f64> = key.iter().map(|r| r * k).collect();
            let c = compare(&rows(key.clone()), &rows(run)).unwrap();
            assert!((c.d_cagr_pp - target_pp).abs() < 1e-6, "dCAGR {} pp for target {target_pp}", c.d_cagr_pp);
            assert_eq!(c.d_cagr_ok, ok, "dCAGR {target_pp}");
            assert!(c.corr_ok && c.d_sharpe_ok);
            assert_eq!(c.bands_pass, ok);
        }
    }

    #[test]
    fn tier1_correlation_band_edge() {
        let key = noisy(600, 0.003);
        let noise: Vec<f64> = (0..600).map(|i| (((i * 104_729 + 7) % 211) as f64 - 105.0) / 61.0 * 0.003).collect();
        for (target, ok) in [(0.9905, true), (0.9895, false)] {
            // corr decreases as the noise weight grows
            let w = solve(0.0, 5.0, -target, |w| {
                let run: Vec<f64> = key.iter().zip(&noise).map(|(a, n)| a + w * n).collect();
                -pearson(&key, &run)
            });
            let run: Vec<f64> = key.iter().zip(&noise).map(|(a, n)| a + w * n).collect();
            let c = compare(&rows(key.clone()), &rows(run)).unwrap();
            assert!((c.corr - target).abs() < 1e-6, "corr {} for target {target}", c.corr);
            assert_eq!(c.corr_ok, ok, "corr {target}");
        }
    }

    #[test]
    fn tier2_boundary_is_inclusive_at_exactly_the_tolerance() {
        let mut key = rets(60);
        key[5] = 0.0;
        let mut run = key.clone();
        run[5] = 1e-9; // |0 - 1e-9| is exactly the pre-registered tolerance
        assert!(compare(&rows(key.clone()), &rows(run.clone())).unwrap().tier2_pass);
        run[5] = 1.0000001e-9;
        assert!(!compare(&rows(key), &rows(run)).unwrap().tier2_pass);
    }

    #[test]
    fn tier3_cell_tolerance_is_one_e_minus_six_inclusive() {
        let mut key = rows(rets(40));
        key.w_target = Some(vec![vec![0.5, 0.0]; 40]);
        let mut run = key.clone();
        run.w_target.as_mut().unwrap()[3][1] = 1e-6; // exactly at the pre-registered tolerance: agrees
        assert_eq!(compare(&key, &run).unwrap().tier3.unwrap().disagreeing, 0);
        run.w_target.as_mut().unwrap()[3][1] = 1.01e-6;
        assert_eq!(compare(&key, &run).unwrap().tier3.unwrap().disagreeing, 1);
    }

    #[test]
    fn the_constants_are_the_preregistered_values() {
        assert_eq!(TIER1_CORR_MIN, 0.99);
        assert_eq!(TIER1_D_SHARPE_MAX, 0.05);
        assert_eq!(TIER1_D_CAGR_PP_MAX, 0.5);
        assert_eq!(TIER1_TRADES_REL_MAX, 0.05);
        assert_eq!(TIER2_TOL, 1e-9);
        assert_eq!(TIER3_CELL_TOL, 1e-6);
        assert_eq!(TIER3_MIN_AGREEMENT, 0.98);
    }

    #[test]
    fn pearson_known_values() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-15);
        assert!((pearson(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]) + 1.0).abs() < 1e-15);
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_nan());
        let c = pearson(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0]);
        assert!((c - 0.8).abs() < 1e-12);
    }

    #[test]
    fn date_helper_is_used() {
        assert!(d("2020-01-02") > d("2020-01-01"));
    }
}
