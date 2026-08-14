//! Cross-sectional rank spread — the pure-math half of a cross-sectional
//! momentum/reversal test (Jegadeesh & Titman 1993-style): rank a panel of
//! assets by a trailing return signal at each of several rebalance points,
//! split into top/bottom halves, and test whether the top-minus-bottom
//! forward-return spread is distinguishable from zero.
//!
//! Adapted to a small (platform tool caps at 10) universe as a median split
//! rather than deciles -- there aren't enough symbols per call for a true
//! decile spread to mean anything.
//!
//! Periods are laid out back-to-back and non-overlapping by construction
//! (each period's forward-return window ends exactly where the next
//! period's trailing window begins) -- this sidesteps the overlapping-window
//! autocorrelation problem that would otherwise require a Newey-West-style
//! correction before a plain t-test on the period spreads is valid (2026-07-29
//! quant council ruling on cross-sectional tooling).

use serde::{Deserialize, Serialize};

use crate::ic::two_sided_p_value;

/// Below this many symbols, a top/bottom-half split isn't meaningful (need
/// at least 2 winners and 2 losers).
const MIN_SYMBOLS: usize = 4;
/// Below this many non-overlapping rebalance periods, a significance test on
/// the period spreads is too thin to trust -- the result is still returned
/// (mean_spread, periods) but `significant` is forced false.
const MIN_PERIODS_FOR_TEST: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancePeriod {
    pub period_index: usize,
    /// Indices into the caller's symbol list, ranked into the top half by
    /// the ranking signal over this period's lookback window (trailing
    /// return for `cross_sectional_rank_spread`, negative trailing
    /// volatility -- i.e. the LOWEST-volatility half -- for
    /// `cross_sectional_volatility_rank_spread`).
    pub winner_indices: Vec<usize>,
    /// Indices into the caller's symbol list, ranked into the bottom half.
    /// An odd-sized universe drops the single middle-ranked symbol from both
    /// groups rather than assigning it arbitrarily.
    pub loser_indices: Vec<usize>,
    pub winner_mean_forward_return: f64,
    pub loser_mean_forward_return: f64,
    /// winner_mean_forward_return - loser_mean_forward_return for this period.
    pub spread: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSectionalRankResult {
    pub n_symbols: usize,
    pub n_periods: usize,
    pub mean_spread: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub significant: bool,
    pub periods: Vec<RebalancePeriod>,
}

impl CrossSectionalRankResult {
    fn empty(n_symbols: usize) -> Self {
        Self {
            n_symbols,
            n_periods: 0,
            mean_spread: 0.0,
            t_stat: 0.0,
            p_value: 1.0,
            significant: false,
            periods: Vec::new(),
        }
    }
}

/// `returns` is an aligned panel: `returns[i]` is symbol i's per-bar simple
/// return series. All symbols must be the same length and cover the same
/// bars -- aligning by timestamp (not just truncating to the shortest
/// series) is the caller's responsibility, since index-position alignment
/// across symbols with different missing-bar patterns would silently
/// scramble the panel.
///
/// Ranks by cumulative (summed) return over a trailing `lookback_bars`
/// window, splits into top/bottom halves, and measures each half's mean
/// cumulative return over the following `holding_bars` window. Requires at
/// least 4 symbols and enough bars for at least one full lookback+holding
/// period; returns an empty (n_periods=0) result rather than panicking when
/// there isn't enough data.
pub fn cross_sectional_rank_spread(
    returns: &[Vec<f64>],
    lookback_bars: usize,
    holding_bars: usize,
) -> CrossSectionalRankResult {
    cross_sectional_rank_spread_by(returns, lookback_bars, holding_bars, |window| {
        window.iter().sum::<f64>()
    })
}

/// Same top/bottom-half forward-spread test as `cross_sectional_rank_spread`,
/// but ranks each period by NEGATIVE trailing realized volatility (population
/// stdev of the lookback window's per-bar returns) instead of trailing
/// cumulative return -- so `winner_indices` is the LOWEST-volatility half and
/// `loser_indices` the highest, and a positive `mean_spread` means the
/// low-volatility half subsequently outperformed the high-volatility half.
/// This is the small-universe, no-market-index-required version of the
/// low-volatility anomaly (Ang/Hodges/Xing/Zhang 2006; Frazzini-Pedersen
/// 2014's "betting against beta" restated cross-sectionally rather than via
/// CAPM beta, since crypto/FX have no uncontested market-portfolio proxy to
/// compute beta against -- 2026-07-30 quant council ruling). Same shape,
/// same data/period requirements, same significance test as the momentum
/// variant above.
pub fn cross_sectional_volatility_rank_spread(
    returns: &[Vec<f64>],
    lookback_bars: usize,
    holding_bars: usize,
) -> CrossSectionalRankResult {
    cross_sectional_rank_spread_by(returns, lookback_bars, holding_bars, |window| {
        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let variance = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        -variance.sqrt()
    })
}

fn cross_sectional_rank_spread_by(
    returns: &[Vec<f64>],
    lookback_bars: usize,
    holding_bars: usize,
    signal_fn: impl Fn(&[f64]) -> f64,
) -> CrossSectionalRankResult {
    let n_symbols = returns.len();
    if n_symbols < MIN_SYMBOLS || lookback_bars == 0 || holding_bars == 0 {
        return CrossSectionalRankResult::empty(n_symbols);
    }
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    let period_len = lookback_bars + holding_bars;
    if min_len < period_len {
        return CrossSectionalRankResult::empty(n_symbols);
    }

    let half = n_symbols / 2;
    let n_periods = min_len / period_len;

    let mut periods = Vec::with_capacity(n_periods);
    let mut spreads = Vec::with_capacity(n_periods);

    for k in 0..n_periods {
        let period_start = k * period_len;
        let rank_end = period_start + lookback_bars;
        let hold_end = rank_end + holding_bars;

        let mut signal: Vec<(usize, f64)> = (0..n_symbols)
            .map(|i| (i, signal_fn(&returns[i][period_start..rank_end])))
            .collect();
        signal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let winner_indices: Vec<usize> = signal[..half].iter().map(|(i, _)| *i).collect();
        let loser_indices: Vec<usize> = signal[n_symbols - half..].iter().map(|(i, _)| *i).collect();

        let fwd_return = |idx: usize| returns[idx][rank_end..hold_end].iter().sum::<f64>();
        let winner_mean = winner_indices.iter().map(|&i| fwd_return(i)).sum::<f64>() / half as f64;
        let loser_mean = loser_indices.iter().map(|&i| fwd_return(i)).sum::<f64>() / half as f64;
        let spread = winner_mean - loser_mean;

        spreads.push(spread);
        periods.push(RebalancePeriod {
            period_index: k,
            winner_indices,
            loser_indices,
            winner_mean_forward_return: winner_mean,
            loser_mean_forward_return: loser_mean,
            spread,
        });
    }

    finish_result(n_symbols, periods, spreads)
}

/// Shared tail for both `cross_sectional_rank_spread_by` and
/// `cross_sectional_rank_spread_by_class`: turns a completed period/spread
/// list into the significance-tested result. Pulled out so the class-aware
/// ranking variant doesn't have to re-derive the same t-test/guard logic.
fn finish_result(
    n_symbols: usize,
    periods: Vec<RebalancePeriod>,
    spreads: Vec<f64>,
) -> CrossSectionalRankResult {
    let n = spreads.len();
    let mean_spread = spreads.iter().sum::<f64>() / n as f64;

    if n < MIN_PERIODS_FOR_TEST {
        return CrossSectionalRankResult {
            n_symbols,
            n_periods: n,
            mean_spread,
            t_stat: 0.0,
            p_value: 1.0,
            significant: false,
            periods,
        };
    }

    let var = spreads.iter().map(|s| (s - mean_spread).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let std_err = (var / n as f64).sqrt();
    // A near-zero std_err (an effectively constant spread across periods)
    // would blow up t_stat via floating-point noise -- guard with a small
    // absolute floor rather than an exact-zero comparison (mirrors
    // `seasonality_by_bucket`'s identical guard).
    let (t_stat, p_value, significant) = if std_err < 1e-9 {
        (0.0, 1.0, false)
    } else {
        let t = mean_spread / std_err;
        let p = two_sided_p_value(t);
        (t, p, p < 0.05)
    };

    CrossSectionalRankResult {
        n_symbols,
        n_periods: n,
        mean_spread,
        t_stat,
        p_value,
        significant,
        periods,
    }
}

/// Same rebalance-period/forward-spread test as `cross_sectional_rank_spread`,
/// but ranks each period by within-asset-class z-scored trailing return
/// instead of a flat cross-universe trailing return -- so a crypto asset is
/// only ever compared against other crypto assets (and forex against forex,
/// equities against equities) when deciding who's a "winner" this period,
/// never against a different asset class's return distribution directly.
///
/// This matters because momentum's magnitude/base-rate differs materially by
/// asset class (2026-07-30 quant council "Mathematical Purist" ruling) -- a
/// flat rank across a mixed-asset-class universe would let one class's wider
/// return dispersion (crypto routinely swings far more than forex) silently
/// dominate the ranking, so "winners" would just mean "was crypto" rather
/// than "actually outperformed its own peers." `classes[i]` is symbol `i`'s
/// asset-class id (any `usize`, e.g. an index into the caller's own class
/// list) -- symbols sharing the same id are compared against each other.
/// A class with fewer than 2 members can't be z-scored against peers, so
/// every member of a singleton class gets a neutral (zero) signal that
/// period -- it still participates in the overall rank (so it CAN end up a
/// winner/loser if the other classes' spread is thin enough), it just never
/// gets a same-class comparison boost.
///
/// Forward returns (what `winner_mean_forward_return`/`spread` measure) are
/// always the raw, un-z-scored return -- only the ranking SIGNAL is
/// class-relative; the payoff being measured is real dollars, not a
/// standardized score.
pub fn cross_sectional_rank_spread_by_class(
    returns: &[Vec<f64>],
    classes: &[usize],
    lookback_bars: usize,
    holding_bars: usize,
) -> CrossSectionalRankResult {
    let n_symbols = returns.len();
    if n_symbols < MIN_SYMBOLS || lookback_bars == 0 || holding_bars == 0 || classes.len() != n_symbols {
        return CrossSectionalRankResult::empty(n_symbols);
    }
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    let period_len = lookback_bars + holding_bars;
    if min_len < period_len {
        return CrossSectionalRankResult::empty(n_symbols);
    }

    let half = n_symbols / 2;
    let n_periods = min_len / period_len;

    let mut periods = Vec::with_capacity(n_periods);
    let mut spreads = Vec::with_capacity(n_periods);

    for k in 0..n_periods {
        let period_start = k * period_len;
        let rank_end = period_start + lookback_bars;
        let hold_end = rank_end + holding_bars;

        let raw_signal: Vec<f64> = (0..n_symbols)
            .map(|i| returns[i][period_start..rank_end].iter().sum::<f64>())
            .collect();

        // Per-class mean/stdev of this period's raw trailing-return signal.
        let mut class_sum: std::collections::HashMap<usize, (f64, usize)> = std::collections::HashMap::new();
        for (i, &v) in raw_signal.iter().enumerate() {
            let e = class_sum.entry(classes[i]).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
        let class_mean: std::collections::HashMap<usize, f64> =
            class_sum.iter().map(|(&c, &(sum, n))| (c, sum / n as f64)).collect();
        let mut class_sq_dev: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for (i, &v) in raw_signal.iter().enumerate() {
            let mean = class_mean[&classes[i]];
            *class_sq_dev.entry(classes[i]).or_insert(0.0) += (v - mean).powi(2);
        }
        let class_std: std::collections::HashMap<usize, f64> = class_sq_dev
            .iter()
            .map(|(&c, &sq)| {
                let n = class_sum[&c].1 as f64;
                (c, if n > 1.0 { (sq / (n - 1.0)).sqrt() } else { 0.0 })
            })
            .collect();

        let z_signal: Vec<(usize, f64)> = (0..n_symbols)
            .map(|i| {
                let std = class_std[&classes[i]];
                let z = if std > 1e-12 { (raw_signal[i] - class_mean[&classes[i]]) / std } else { 0.0 };
                (i, z)
            })
            .collect();

        let mut signal = z_signal;
        signal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let winner_indices: Vec<usize> = signal[..half].iter().map(|(i, _)| *i).collect();
        let loser_indices: Vec<usize> = signal[n_symbols - half..].iter().map(|(i, _)| *i).collect();

        let fwd_return = |idx: usize| returns[idx][rank_end..hold_end].iter().sum::<f64>();
        let winner_mean = winner_indices.iter().map(|&i| fwd_return(i)).sum::<f64>() / half as f64;
        let loser_mean = loser_indices.iter().map(|&i| fwd_return(i)).sum::<f64>() / half as f64;
        let spread = winner_mean - loser_mean;

        spreads.push(spread);
        periods.push(RebalancePeriod {
            period_index: k,
            winner_indices,
            loser_indices,
            winner_mean_forward_return: winner_mean,
            loser_mean_forward_return: loser_mean,
            spread,
        });
    }

    finish_result(n_symbols, periods, spreads)
}

/// Grinold's Fundamental Law of Active Management, restated as an effective
/// (correlation-adjusted) breadth: `N / (1 + (N-1) * rho_avg)`, where
/// `rho_avg` is the average pairwise correlation across `returns`' full
/// series. A universe of 10 assets that are all 90%+ correlated has an
/// effective breadth close to 1 real independent bet, not 10 -- this is what
/// should feed a multiple-comparisons/DSR trial count for a cross-sectional
/// strategy built on that universe, not the raw symbol count (2026-08 quant
/// council Phase 0 measurement: cross-asset-class correlation runs 5-15x
/// lower than within-crypto correlation, which is exactly the gap this
/// formula is meant to capture -- a low-rho universe keeps close to its full
/// nominal breadth, a high-rho one collapses toward 1).
///
/// Returns `n_symbols` unchanged (no correlation adjustment) if fewer than 2
/// series are supplied or all series are too short to correlate. Clamped to
/// `[1.0, n_symbols]` -- a negative average correlation could otherwise push
/// the formula above the nominal count, which isn't a meaningful "more
/// independent bets than assets" result.
pub fn effective_breadth(returns: &[Vec<f64>]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return n as f64;
    }

    let mut sum_rho = 0.0;
    let mut n_pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if let Some(rho) = pearson_correlation(&returns[i], &returns[j]) {
                sum_rho += rho;
                n_pairs += 1;
            }
        }
    }
    if n_pairs == 0 {
        return n as f64;
    }
    let rho_avg = sum_rho / n_pairs as f64;
    let n_f = n as f64;
    let effective = n_f / (1.0 + (n_f - 1.0) * rho_avg);
    effective.clamp(1.0, n_f)
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    let len = a.len().min(b.len());
    if len < 2 {
        return None;
    }
    let a = &a[..len];
    let b = &b[..len];
    let mean_a = a.iter().sum::<f64>() / len as f64;
    let mean_b = b.iter().sum::<f64>() / len as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..len {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 1e-18 || var_b <= 1e-18 {
        return None;
    }
    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_symbols_returns_empty() {
        let returns = vec![vec![0.01; 20]; 3];
        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, 0);
        assert_eq!(result.n_symbols, 3);
    }

    #[test]
    fn too_few_bars_returns_empty() {
        let returns = vec![vec![0.01; 5]; 4];
        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, 0);
    }

    #[test]
    fn zero_lookback_or_holding_bars_returns_empty() {
        let returns = vec![vec![0.01; 40]; 6];
        assert_eq!(cross_sectional_rank_spread(&returns, 0, 5).n_periods, 0);
        assert_eq!(cross_sectional_rank_spread(&returns, 5, 0).n_periods, 0);
    }

    #[test]
    fn odd_symbol_count_drops_middle_from_both_groups() {
        // 5 symbols, half = 2 -- winners and losers should each have 2
        // members, and no index should appear in both.
        let returns = vec![
            vec![0.05; 20], // rank 1
            vec![0.03; 20], // rank 2
            vec![0.0; 20],  // rank 3 (middle -- dropped)
            vec![-0.03; 20], // rank 4
            vec![-0.05; 20], // rank 5
        ];
        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert!(result.n_periods > 0);
        let p = &result.periods[0];
        assert_eq!(p.winner_indices.len(), 2);
        assert_eq!(p.loser_indices.len(), 2);
        assert!(!p.winner_indices.contains(&2));
        assert!(!p.loser_indices.contains(&2));
        let overlap: Vec<&usize> = p.winner_indices.iter().filter(|i| p.loser_indices.contains(i)).collect();
        assert!(overlap.is_empty());
    }

    #[test]
    fn persistent_momentum_signal_is_detected_and_significant() {
        // Symbols 0-1 always outperform symbols 2-3 in both the ranking
        // window and the forward window, consistently across many periods --
        // a clean, strong momentum-style cross-sectional relationship. A
        // small per-period oscillation is mixed in so the period spreads
        // have nonzero variance (a perfectly constant spread would hit the
        // near-zero-std_err guard and report p=1.0, same as the
        // `identical_symbols` test below -- that's a separate, deliberate
        // edge case, not what this test is checking).
        let period_len = 10;
        let n_periods = 20;
        let total_bars = period_len * n_periods;
        let high: Vec<f64> = (0..total_bars).map(|i| 0.01 + 0.001 * (((i / period_len) % 3) as f64 - 1.0)).collect();
        let low: Vec<f64> = (0..total_bars).map(|i| -0.01 - 0.001 * (((i / period_len) % 3) as f64 - 1.0)).collect();
        let returns = vec![high.clone(), high, low.clone(), low];

        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, n_periods);
        assert!(result.mean_spread > 0.0);
        assert!(result.significant, "expected a persistent spread to be significant, got p={}", result.p_value);
        // Winners should always be symbols 0/1, losers always 2/3.
        for p in &result.periods {
            assert!(p.winner_indices.contains(&0) && p.winner_indices.contains(&1));
            assert!(p.loser_indices.contains(&2) && p.loser_indices.contains(&3));
        }
    }

    #[test]
    fn identical_symbols_have_zero_spread_and_are_not_significant() {
        // Every symbol has an identical return series -- there is no real
        // cross-sectional relationship, and the winner/loser split is
        // arbitrary tie-breaking. Spread must be exactly zero every period,
        // which should hit the near-zero-std_err guard rather than reporting
        // a spurious significant result.
        let returns = vec![vec![0.01; 60]; 6];
        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert!(result.n_periods >= MIN_PERIODS_FOR_TEST);
        assert_eq!(result.mean_spread, 0.0);
        assert!(!result.significant);
        assert_eq!(result.p_value, 1.0);
    }

    // --- cross_sectional_volatility_rank_spread ---

    #[test]
    fn volatility_rank_puts_low_vol_symbols_in_winner_half() {
        // Symbols 0-1 have near-zero-variance returns (low vol); symbols 2-3
        // oscillate wildly (high vol). Both groups average to ~0 return, so
        // this isolates the ranking-by-volatility behavior specifically.
        let period_len = 10;
        let n_periods = 20;
        let total_bars = period_len * n_periods;
        let low_vol: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.0001 } else { -0.0001 }).collect();
        let high_vol: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.05 } else { -0.05 }).collect();
        let returns = vec![low_vol.clone(), low_vol, high_vol.clone(), high_vol];

        let result = cross_sectional_volatility_rank_spread(&returns, 5, 5);
        assert!(result.n_periods > 0);
        for p in &result.periods {
            assert!(p.winner_indices.contains(&0) && p.winner_indices.contains(&1), "low-vol symbols should be winners");
            assert!(p.loser_indices.contains(&2) && p.loser_indices.contains(&3), "high-vol symbols should be losers");
        }
    }

    #[test]
    fn volatility_rank_detects_a_persistent_low_vol_outperformance_spread() {
        // Low-vol symbols (0-1) consistently earn a small positive drift;
        // high-vol symbols (2-3) consistently earn a small negative drift,
        // on top of oscillating noise sized so ranking-by-volatility (not
        // ranking-by-return) is what separates the groups. A per-period-
        // varying component (cycling every 3 periods, mirrors
        // `persistent_momentum_signal_is_detected_and_significant` above)
        // keeps the period spreads from being exactly constant -- a
        // perfectly constant spread would hit the near-zero-std_err guard
        // and report p=1.0 regardless of how large the spread itself is,
        // same edge case `identical_symbols_have_zero_spread...` checks
        // separately. The per-period component is constant WITHIN each
        // 5-bar lookback/holding window (period_len=10 is a multiple of the
        // window size), so it shifts each period's mean without adding
        // spurious in-window volatility that could scramble the ranking.
        let period_len = 10;
        let n_periods = 20;
        let total_bars = period_len * n_periods;
        let low_vol: Vec<f64> = (0..total_bars)
            .map(|i| {
                let period_component = 0.001 * (((i / period_len) % 3) as f64 - 1.0);
                let osc = if i % 2 == 0 { 0.0001 } else { -0.0001 };
                0.003 + period_component + osc
            })
            .collect();
        let high_vol: Vec<f64> = (0..total_bars)
            .map(|i| {
                let period_component = -0.001 * (((i / period_len) % 3) as f64 - 1.0);
                let osc = if i % 2 == 0 { 0.05 } else { -0.05 };
                -0.003 + period_component + osc
            })
            .collect();
        let returns = vec![low_vol.clone(), low_vol, high_vol.clone(), high_vol];

        let result = cross_sectional_volatility_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, n_periods);
        assert!(result.mean_spread > 0.0, "low-vol half should show a positive forward-return spread over high-vol half");
        assert!(result.significant, "expected a persistent low-vol spread to be significant, got p={}", result.p_value);
        for p in &result.periods {
            assert!(p.winner_indices.contains(&0) && p.winner_indices.contains(&1));
            assert!(p.loser_indices.contains(&2) && p.loser_indices.contains(&3));
        }
    }

    #[test]
    fn volatility_rank_too_few_symbols_returns_empty() {
        let returns = vec![vec![0.01; 20]; 3];
        let result = cross_sectional_volatility_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, 0);
        assert_eq!(result.n_symbols, 3);
    }

    #[test]
    fn volatility_rank_identical_symbols_have_zero_spread_and_are_not_significant() {
        let returns = vec![vec![0.01; 60]; 6];
        let result = cross_sectional_volatility_rank_spread(&returns, 5, 5);
        assert!(result.n_periods >= MIN_PERIODS_FOR_TEST);
        assert_eq!(result.mean_spread, 0.0);
        assert!(!result.significant);
    }

    #[test]
    fn below_minimum_periods_reports_result_without_significance_test() {
        // Exactly 2 periods -- below MIN_PERIODS_FOR_TEST=3, so the result
        // should still carry mean_spread/periods but force significant=false.
        let period_len = 10;
        let returns = vec![
            vec![0.01; period_len * 2],
            vec![0.01; period_len * 2],
            vec![-0.01; period_len * 2],
            vec![-0.01; period_len * 2],
        ];
        let result = cross_sectional_rank_spread(&returns, 5, 5);
        assert_eq!(result.n_periods, 2);
        assert!(!result.significant);
        assert_eq!(result.t_stat, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    // --- cross_sectional_rank_spread_by_class ---

    #[test]
    fn by_class_too_few_symbols_returns_empty() {
        let returns = vec![vec![0.01; 20]; 3];
        let classes = vec![0, 0, 1];
        let result = cross_sectional_rank_spread_by_class(&returns, &classes, 5, 5);
        assert_eq!(result.n_periods, 0);
    }

    #[test]
    fn by_class_mismatched_classes_length_returns_empty() {
        let returns = vec![vec![0.01; 40]; 4];
        let classes = vec![0, 0, 1]; // one short
        let result = cross_sectional_rank_spread_by_class(&returns, &classes, 5, 5);
        assert_eq!(result.n_periods, 0);
    }

    #[test]
    fn by_class_ranks_within_class_not_across() {
        // Two classes of 2. Class 0's symbols both have modest positive
        // returns; class 1's symbols both have huge positive returns. A flat
        // (non-class-aware) rank would put both class-1 symbols in the
        // winner half every period since their raw return dwarfs class 0's.
        // A class-relative rank instead compares each symbol only to its own
        // class peer, so within EACH class one symbol should win and one
        // should lose, regardless of the cross-class magnitude gap.
        let period_len = 10;
        let n_periods = 20;
        let total_bars = period_len * n_periods;
        // Class 0: symbol 0 slightly beats symbol 1 every period.
        let c0_hi: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.011 } else { 0.009 }).collect();
        let c0_lo: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.009 } else { 0.011 }).collect();
        // Class 1: symbol 2 slightly beats symbol 3 every period, but both
        // are an order of magnitude larger than class 0's returns.
        let c1_hi: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.31 } else { 0.29 }).collect();
        let c1_lo: Vec<f64> = (0..total_bars).map(|i| if i % 2 == 0 { 0.29 } else { 0.31 }).collect();
        let returns = vec![c0_hi, c0_lo, c1_hi, c1_lo];
        let classes = vec![0, 0, 1, 1];

        let result = cross_sectional_rank_spread_by_class(&returns, &classes, 5, 5);
        assert!(result.n_periods > 0);
        for p in &result.periods {
            // Exactly one winner and one loser from each class, not both
            // class-1 symbols sweeping the winner half.
            let winner_classes: Vec<usize> = p.winner_indices.iter().map(|&i| classes[i]).collect();
            let loser_classes: Vec<usize> = p.loser_indices.iter().map(|&i| classes[i]).collect();
            assert!(winner_classes.contains(&0), "class 0 should have a representative among winners");
            assert!(winner_classes.contains(&1), "class 1 should have a representative among winners");
            assert!(loser_classes.contains(&0), "class 0 should have a representative among losers");
            assert!(loser_classes.contains(&1), "class 1 should have a representative among losers");
        }
    }

    #[test]
    fn by_class_singleton_class_gets_neutral_signal() {
        // Class 2 has only one member (symbol 3) -- it can't be z-scored
        // against a peer, so it should get signal 0 every period rather than
        // panicking or dividing by zero.
        let returns = vec![vec![0.01; 40]; 4];
        let classes = vec![0, 0, 1, 2];
        let result = cross_sectional_rank_spread_by_class(&returns, &classes, 5, 5);
        assert!(result.n_periods > 0);
    }

    // --- effective_breadth ---

    #[test]
    fn effective_breadth_of_fewer_than_two_series_is_the_raw_count() {
        assert_eq!(effective_breadth(&[]), 0.0);
        assert_eq!(effective_breadth(&[vec![0.01, 0.02, 0.03]]), 1.0);
    }

    #[test]
    fn effective_breadth_of_perfectly_correlated_assets_collapses_toward_one() {
        let series = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, -0.02];
        let returns = vec![series.clone(); 8];
        let breadth = effective_breadth(&returns);
        assert!(breadth < 1.5, "8 perfectly correlated assets should collapse close to 1 effective bet, got {}", breadth);
    }

    #[test]
    fn effective_breadth_of_uncorrelated_assets_stays_close_to_raw_count() {
        // Deterministic pseudo-independent series (distinct phase-shifted
        // sinusoids) rather than a random seed, so the test is reproducible.
        let n = 6;
        let bars = 200;
        let returns: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                (0..bars)
                    .map(|i| ((i as f64 * 0.7 + k as f64 * 2.3).sin()) * 0.01)
                    .collect()
            })
            .collect();
        let breadth = effective_breadth(&returns);
        assert!(breadth > n as f64 * 0.5, "near-independent assets should keep most of their raw breadth, got {} of {}", breadth, n);
    }

    #[test]
    fn effective_breadth_is_clamped_to_at_most_the_raw_count() {
        // Strongly negatively correlated pair -- rho_avg near -1 would push
        // the raw formula above n without the clamp.
        let a: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = a.iter().map(|v| -v).collect();
        let returns = vec![a, b];
        let breadth = effective_breadth(&returns);
        assert!(breadth <= 2.0);
    }
}
