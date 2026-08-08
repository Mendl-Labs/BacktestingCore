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
}
