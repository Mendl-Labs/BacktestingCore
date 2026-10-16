//! Information coefficient (IC) — Spearman rank correlation between a
//! strategy's feature and its forward return, with a significance test.
//! A bare correlation magnitude (e.g. "IC = 0.03") is not trustworthy on its
//! own — it needs the t-stat attached to know whether it's distinguishable
//! from noise at the sample size actually available.
//!
//! The sample size that matters is the number of INDEPENDENT observations, not
//! the number of rows. A forward return over `horizon > 1` bars overlaps its
//! neighbours, and a persistent feature (a trend, a volatility level, a funding
//! rate) barely changes from one bar to the next, so treating every row as an
//! independent sample overstates significance -- measured on pure noise at
//! alpha = 0.05 the unadjusted test rejected 17-64% of the time at horizons of
//! 5 and 20 bars for persistent features. `compute_ic` therefore tests with an
//! effective sample size (see [`compute_ic`]).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IcResult {
    /// Forward-return horizon (in bars) the feature was evaluated against.
    pub horizon: usize,
    /// Spearman rank correlation between feature[t] and return[t+horizon].
    pub ic: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub significant: bool,
    /// Number of (feature, forward_return) pairs actually used.
    pub n: usize,
    /// Effective number of independent observations behind `t_stat` and
    /// `p_value` (`n / kappa`, see [`compute_ic`]). Equal to `n` when nothing needs
    /// adjusting (horizon 1, or a non-persistent feature); smaller when overlapping
    /// forward returns and a persistent feature share information. `None` only for
    /// a result built by hand or deserialised from data written before this field
    /// existed.
    #[serde(default)]
    pub n_effective: Option<f64>,
}

/// Average-rank transform (ties share the mean of their rank positions).
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        // Ranks are 1-based; ties get the average of their position range.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in idx.iter().take(j + 1).skip(i) {
            ranks[*k] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Sample autocorrelation of `x` at `lag` (the standard biased estimator).
/// 0.0 for a degenerate input (constant series, or a lag at or beyond the length).
fn autocorrelation(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if lag == 0 || lag >= n {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let denom: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let num: f64 = (0..n - lag).map(|t| (x[t] - mean) * (x[t + lag] - mean)).sum();
    num / denom
}

/// Variance-inflation factor for a correlation between two autocorrelated
/// series (Bartlett 1935; the same correction as Hansen-Hodrick for overlapping
/// forecast horizons): `kappa = 1 + 2 * sum_{k=1}^{L} rho_x(k) * rho_y(k)`, where
/// `rho` is the sample autocorrelation of each RANKED series and `L = horizon - 1`
/// (overlapping `horizon`-bar returns share information for `horizon - 1` lags).
///
/// Estimated from the data, so it is harmless when `horizon` is not really an
/// overlap window (e.g. a lead/lag scan passing a lag): non-overlapping returns
/// have near-zero autocorrelation and `kappa` stays ~1. Clamped at 1.0 so the
/// adjustment can only make a result LESS significant than the unadjusted test,
/// never more. Exactly 1.0 at `horizon <= 1`.
fn overlap_inflation(feature_ranks: &[f64], return_ranks: &[f64], horizon: usize) -> f64 {
    let n = feature_ranks.len();
    let max_lag = horizon.saturating_sub(1).min(n / 4);
    if max_lag == 0 {
        return 1.0;
    }
    let s: f64 = (1..=max_lag)
        .map(|k| autocorrelation(feature_ranks, k) * autocorrelation(return_ranks, k))
        .sum();
    (1.0 + 2.0 * s).max(1.0)
}

/// Shared with `seasonality` -- both need the same normal-CDF approximation
/// for turning a z/t-statistic into a two-sided p-value.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = pdf * poly;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Shared with `seasonality`.
pub(crate) fn two_sided_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Spearman rank correlation between a feature series and its forward
/// return, shifted by `horizon` bars (`feature[t]` vs `forward_return`
/// realized over `[t, t+horizon]`). Both slices must already be aligned —
/// i.e. `forward_returns[i]` is the return that follows `feature[i]`.
///
/// Uses a t-distribution-style test statistic approximated with the normal
/// CDF (adequate once n is more than a handful of points, which is the
/// regime this is meant for — very small samples return `None`).
pub fn compute_ic(feature: &[f64], forward_returns: &[f64], horizon: usize) -> Option<IcResult> {
    if feature.len() != forward_returns.len() {
        return None;
    }
    let n = feature.len();
    if n < 5 {
        return None;
    }

    let feature_ranks = rank(feature);
    let return_ranks = rank(forward_returns);
    let ic = pearson(&feature_ranks, &return_ranks)?;

    let kappa = overlap_inflation(&feature_ranks, &return_ranks, horizon);
    let n_effective = (n as f64 / kappa).max(3.0);
    let df = n_effective - 2.0;
    if df <= 0.0 {
        return None;
    }
    let denom = (1.0 - ic * ic).max(1e-12);
    let t_stat = ic * (df / denom).sqrt();
    let p_value = two_sided_p_value(t_stat);

    Some(IcResult {
        horizon,
        ic,
        t_stat,
        p_value,
        significant: p_value < 0.05,
        n,
        n_effective: Some(n_effective),
    })
}

/// Pool per-asset `IcResult`s into one significance test, correcting for
/// cross-asset correlation via `effective_breadth` -- summing raw t-stats
/// (or naively averaging Fisher-z scores) across N assets overstates power
/// by treating them as N independent samples when correlated assets share
/// much of the same information. Uses a Fisher z-transform (the standard
/// way to average correlations, since raw ICs don't add linearly), weighted
/// by each asset's own sample size, then inflates the combined variance by
/// `N / effective_breadth` so the pooled t-stat reflects the REAL number of
/// independent bets, not the nominal asset count.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PooledIcResult {
    /// Weighted-average IC across assets (Fisher-z averaged, then inverted).
    pub pooled_ic: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub significant: bool,
    /// Total (feature, forward_return) pairs summed across every asset.
    pub n_total: usize,
    /// Number of assets pooled.
    pub n_assets: usize,
    pub effective_breadth: f64,
}

/// `per_asset` must be non-empty; an IC of exactly +-1.0 (zero-variance
/// Fisher z) is clamped away from the singularity rather than rejected, so
/// one perfectly-correlated asset can't blow up the whole pool. Each asset is
/// weighted by its EFFECTIVE sample size (`n_effective`, falling back to `n`), so
/// an asset whose overlapping returns carry little independent information does
/// not dominate the pool. Returns
/// `None` for an empty slice or when every asset's IC is unusable (each
/// `1 - ic^2 <= 0`, which only happens for a length-0 input in practice
/// since `compute_ic` itself already guards near-zero variance).
pub fn pooled_ic(per_asset: &[IcResult], effective_breadth: f64) -> Option<PooledIcResult> {
    if per_asset.is_empty() {
        return None;
    }
    let n_assets = per_asset.len();
    let effective_breadth = effective_breadth.clamp(1.0, n_assets as f64);

    let mut sum_weight = 0.0;
    let mut sum_weighted_z = 0.0;
    let mut n_total = 0usize;
    for r in per_asset {
        let weight = (r.n_effective.unwrap_or(r.n as f64) - 3.0).max(1.0);
        let ic_clamped = r.ic.clamp(-0.999999, 0.999999);
        let z = ic_clamped.atanh();
        sum_weighted_z += weight * z;
        sum_weight += weight;
        n_total += r.n;
    }
    if sum_weight <= 0.0 {
        return None;
    }
    let z_bar = sum_weighted_z / sum_weight;
    // Naive pooled variance is 1/sum_weight (independent-samples Fisher-z
    // variance); inflate by N/N_eff so correlated assets don't manufacture
    // significance out of redundant observations.
    let pooled_variance = (1.0 / sum_weight) * (n_assets as f64 / effective_breadth);
    let t_stat = z_bar / pooled_variance.sqrt();
    let p_value = two_sided_p_value(t_stat);

    Some(PooledIcResult {
        pooled_ic: z_bar.tanh(),
        t_stat,
        p_value,
        significant: p_value < 0.05,
        n_total,
        n_assets,
        effective_breadth,
    })
}

#[cfg(test)]
mod pooled_ic_tests {
    use super::*;

    fn ic(ic: f64, n: usize) -> IcResult {
        // t_stat/p_value/significant are recomputed by compute_ic in real
        // use; pooled_ic only reads .ic and .n, so hand-building a fixture
        // with plausible-but-unused values for the rest is fine here.
        IcResult { horizon: 1, ic, t_stat: 0.0, p_value: 1.0, significant: false, n, n_effective: None }
    }

    /// A moderate, well-conditioned IC (~0.3, not a perfect monotonic
    /// relationship) -- deliberately NOT near +-1.0, since both the raw
    /// t-stat formula (1/(1-ic^2) blowup) and the Fisher-z transform
    /// (atanh singularity) become numerically extreme near a perfect
    /// correlation, in DIFFERENT ways, which breaks any comparison between
    /// them. A realistic small-IC fixture is also the actually-relevant
    /// case for this platform's own signal tests.
    fn moderate_ic_fixture() -> IcResult {
        let feature: Vec<f64> = (0..60).map(|i| ((i * 37) % 23) as f64).collect();
        let forward_return: Vec<f64> = feature.iter().enumerate()
            .map(|(i, f)| 0.3 * f + ((i * 17) % 11) as f64 * 2.0)
            .collect();
        compute_ic(&feature, &forward_return, 1).expect("enough data for a moderate fixture")
    }

    #[test]
    fn identical_series_collapse_effective_breadth_toward_one_asset() {
        // N copies of the SAME asset's IC: effective_breadth=1 should give a
        // t-stat close to what a single asset with the same n would produce,
        // not one inflated by pretending there are N independent samples.
        let single = moderate_ic_fixture();
        let per_asset = vec![single; 5];
        let pooled = pooled_ic(&per_asset, 1.0).unwrap();
        assert_eq!(pooled.n_assets, 5);
        assert!((pooled.pooled_ic - single.ic).abs() < 1e-6, "pooled {} vs single {}", pooled.pooled_ic, single.ic);
        // Same z, same weight-per-asset -> pooled t roughly equals the
        // single-asset t once N/N_eff cancels the N-fold weight sum. Compared
        // against the Fisher-z t-stat compute_ic itself would report for the
        // same (ic, n), not compute_ic's own raw-Pearson t-stat, which uses a
        // different formula and does not agree with the atanh-based one even
        // at moderate IC.
        let z = single.ic.atanh();
        let single_fisher_t = z / (1.0 / (single.n as f64 - 3.0)).sqrt();
        assert!((pooled.t_stat - single_fisher_t).abs() / single_fisher_t.abs() < 0.05,
            "pooled t {} should be close to single-asset Fisher t {}", pooled.t_stat, single_fisher_t);
    }

    #[test]
    fn full_effective_breadth_scales_t_stat_with_sqrt_n() {
        // Same per-asset IC values, but effective_breadth == n_assets (fully
        // independent): pooled t should grow roughly like sqrt(N) relative
        // to the single-asset Fisher-z t-stat, the classic power gain from
        // real breadth.
        let single = moderate_ic_fixture();
        let z = single.ic.atanh();
        let single_fisher_t = z / (1.0 / (single.n as f64 - 3.0)).sqrt();
        let n = 4;
        let per_asset = vec![single; n];
        let pooled = pooled_ic(&per_asset, n as f64).unwrap();
        let ratio = pooled.t_stat / single_fisher_t;
        assert!((ratio - (n as f64).sqrt()).abs() < 0.05, "expected ~sqrt({n}) scaling, got ratio {ratio}");
    }

    #[test]
    fn effective_breadth_is_clamped_into_one_to_n_assets() {
        let per_asset = vec![ic(0.1, 40), ic(0.1, 40)];
        // Caller-supplied breadth outside [1, n_assets] must not produce a
        // negative or explosive variance.
        let too_high = pooled_ic(&per_asset, 100.0).unwrap();
        let too_low = pooled_ic(&per_asset, 0.0).unwrap();
        assert_eq!(too_high.effective_breadth, 2.0);
        assert_eq!(too_low.effective_breadth, 1.0);
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(pooled_ic(&[], 1.0).is_none());
    }

    #[test]
    fn n_total_sums_every_assets_observation_count() {
        let per_asset = vec![ic(0.1, 40), ic(-0.2, 60)];
        let pooled = pooled_ic(&per_asset, 2.0).unwrap();
        assert_eq!(pooled.n_total, 100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_lengths_return_none() {
        assert!(compute_ic(&[1.0, 2.0], &[1.0], 1).is_none());
    }

    #[test]
    fn too_few_points_return_none() {
        assert!(compute_ic(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1).is_none());
    }

    #[test]
    fn constant_feature_returns_none() {
        let feature = vec![1.0; 20];
        let returns: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(compute_ic(&feature, &returns, 1).is_none());
    }

    #[test]
    fn perfectly_correlated_series_has_ic_near_one_and_significant() {
        let feature: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let returns: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 1.0).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!((result.ic - 1.0).abs() < 1e-9);
        assert!(result.significant);
    }

    #[test]
    fn perfectly_anti_correlated_series_has_ic_near_negative_one() {
        let feature: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let returns: Vec<f64> = (0..50).map(|i| -(i as f64)).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!((result.ic + 1.0).abs() < 1e-9);
        assert!(result.significant);
    }

    #[test]
    fn random_noise_is_not_significant() {
        // Deterministic pseudo-random-ish sequences with no real relationship.
        let feature: Vec<f64> = (0..40).map(|i| ((i * 37) % 13) as f64).collect();
        let returns: Vec<f64> = (0..40).map(|i| ((i * 19 + 5) % 11) as f64).collect();
        let result = compute_ic(&feature, &returns, 1).expect("enough data");
        assert!(result.ic.abs() < 1.0);
        // Not asserting non-significance here (a specific arrangement could
        // spuriously correlate) — just that the machinery produces a sane,
        // bounded result without panicking.
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn rank_averages_ties() {
        let ranks = rank(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }
}

/// Placebo and power tests for the overlap / persistence adjustment. The placebo
/// is the point: on pure noise the unadjusted test must be shown to reject far
/// above its nominal level, and the adjusted one must not.
#[cfg(test)]
mod overlap_adjustment_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, StudentT};

    /// The pre-fix test, reproduced: normal-approximation t-test on the raw row count.
    fn naive_p(ic: f64, n: usize) -> f64 {
        let t = ic * ((n as f64 - 2.0) / (1.0 - ic * ic).max(1e-12)).sqrt();
        two_sided_p_value(t)
    }

    /// An AR(1) feature (persistence `phi`) and iid fat-tailed daily returns; the
    /// forward returns are the OVERLAPPING `h`-bar returns, exactly as the platform's
    /// IC services build them. `beta > 0` plants a real edge: the feature at t-1
    /// predicts the 1-bar return at t (in units of daily volatility per feature s.d.).
    fn case(rng: &mut StdRng, n: usize, phi: f64, h: usize, beta: f64) -> (Vec<f64>, Vec<f64>) {
        let t5 = StudentT::new(5.0).unwrap();
        let z = Normal::new(0.0, 1.0).unwrap();
        let total = n + h + 1;
        let mut f = vec![0.0; total];
        for i in 1..total {
            f[i] = phi * f[i - 1] + z.sample(rng);
        }
        let mean = f.iter().sum::<f64>() / total as f64;
        let sd = (f.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total as f64).sqrt();
        let mut price = vec![1.0; total];
        for i in 1..total {
            let r = 0.03 * t5.sample(rng) + beta * 0.03 * f[i - 1] / sd;
            price[i] = price[i - 1] * (1.0 + r);
        }
        let feature = f[1..1 + n].to_vec();
        let forward = (0..n).map(|i| price[i + 1 + h] / price[i + 1] - 1.0).collect();
        (feature, forward)
    }

    #[test]
    fn on_pure_noise_the_naive_test_overrejects_and_the_adjusted_test_does_not() {
        let mut rng = StdRng::seed_from_u64(20260920);
        let trials = 400;
        let (mut naive_rej, mut adj_rej) = (0, 0);
        for _ in 0..trials {
            let (f, r) = case(&mut rng, 1000, 0.9, 5, 0.0);
            let res = compute_ic(&f, &r, 5).unwrap();
            naive_rej += (naive_p(res.ic, res.n) < 0.05) as usize;
            adj_rej += (res.p_value < 0.05) as usize;
        }
        let (naive_rate, adj_rate) = (naive_rej as f64 / trials as f64, adj_rej as f64 / trials as f64);
        // The defect this fixes: a persistent feature against overlapping 5-bar returns.
        assert!(naive_rate > 0.20, "the placebo no longer exposes the naive test's over-rejection: {naive_rate:.3}");
        // The fix: back near the nominal 5% (loose bound for 400 trials).
        assert!(adj_rate < 0.10, "adjusted false-positive rate {adj_rate:.3} is not near the nominal 5%");
    }

    #[test]
    fn a_real_predictive_edge_is_still_detected_after_the_adjustment() {
        let mut rng = StdRng::seed_from_u64(7);
        let trials = 100;
        let detected = (0..trials)
            .filter(|_| {
                let (f, r) = case(&mut rng, 2000, 0.9, 5, 0.10);
                compute_ic(&f, &r, 5).unwrap().p_value < 0.05
            })
            .count();
        // Prototype measured ~91% at this size; the bound leaves wide margin for 100 trials.
        assert!(detected as f64 / trials as f64 > 0.70, "only {detected}/{trials} planted edges detected");
    }

    #[test]
    fn horizon_one_is_exactly_the_unadjusted_test() {
        let mut rng = StdRng::seed_from_u64(3);
        let (f, r) = case(&mut rng, 500, 0.9, 1, 0.05);
        let res = compute_ic(&f, &r, 1).unwrap();
        assert_eq!(res.n, 500);
        assert_eq!(res.n_effective, Some(500.0));
        let expected_t = res.ic * ((500.0 - 2.0) / (1.0 - res.ic * res.ic)).sqrt();
        assert!((res.t_stat - expected_t).abs() < 1e-12, "t {} vs unadjusted {}", res.t_stat, expected_t);
    }

    #[test]
    fn the_adjustment_can_only_reduce_significance_never_increase_it() {
        let mut rng = StdRng::seed_from_u64(11);
        for &(phi, h) in &[(0.0, 5), (0.5, 5), (0.9, 5), (0.9, 20), (0.97, 20), (0.3, 3)] {
            for _ in 0..25 {
                let (f, r) = case(&mut rng, 600, phi, h, 0.03);
                let res = compute_ic(&f, &r, h).unwrap();
                assert!(res.n_effective.unwrap() <= res.n as f64 + 1e-9);
                assert!(res.p_value >= naive_p(res.ic, res.n) - 1e-12, "phi {phi} h {h}: adjusted p {} < naive p {}", res.p_value, naive_p(res.ic, res.n));
            }
        }
    }

    #[test]
    fn a_non_persistent_feature_barely_loses_effective_sample_size() {
        // A lead/lag scan passes a lag as `horizon`; an independent feature must not be penalised much.
        let mut rng = StdRng::seed_from_u64(19);
        let mean_ratio = (0..50)
            .map(|_| {
                let (f, r) = case(&mut rng, 1000, 0.0, 20, 0.0);
                let res = compute_ic(&f, &r, 20).unwrap();
                res.n_effective.unwrap() / res.n as f64
            })
            .sum::<f64>()
            / 50.0;
        assert!(mean_ratio > 0.90, "independent feature lost too much effective n: ratio {mean_ratio:.3}");
    }

    #[test]
    fn a_persistent_feature_at_a_long_horizon_shrinks_the_effective_sample_a_lot() {
        let mut rng = StdRng::seed_from_u64(23);
        let (f, r) = case(&mut rng, 2000, 0.95, 20, 0.0);
        let res = compute_ic(&f, &r, 20).unwrap();
        assert!(res.n_effective.unwrap() < 0.35 * res.n as f64, "n_effective {:?} of n {}", res.n_effective, res.n);
    }

    #[test]
    fn pooled_ic_weights_each_asset_by_its_effective_sample() {
        let full = IcResult { horizon: 5, ic: 0.2, t_stat: 0.0, p_value: 1.0, significant: false, n: 1000, n_effective: Some(1000.0) };
        let shrunk = IcResult { n_effective: Some(100.0), ..full };
        let strong = pooled_ic(&[full, full], 2.0).unwrap();
        let weak = pooled_ic(&[shrunk, shrunk], 2.0).unwrap();
        assert!(weak.t_stat < strong.t_stat, "effective-n weighting not applied: {} vs {}", weak.t_stat, strong.t_stat);
        assert_eq!(weak.n_total, 2000, "n_total keeps the raw pair count");
    }

    #[test]
    fn results_serialised_before_the_field_existed_still_deserialise() {
        let old = r#"{"horizon":5,"ic":0.1,"t_stat":2.0,"p_value":0.045,"significant":true,"n":100}"#;
        let parsed: IcResult = serde_json::from_str(old).expect("old rows must still load");
        assert_eq!(parsed.n_effective, None);
        assert_eq!(parsed.n, 100);
    }
}
