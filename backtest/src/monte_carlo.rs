//! Monte Carlo simulation module for backtesting
//! 
//! Performs Monte Carlo analysis by randomizing trade sequences to assess
//! strategy robustness and generate confidence intervals for performance metrics.
//! 
//! Methods:
//! - Shuffle: Randomize trade order to test path dependency
//! - Bootstrap: Sample with replacement for variance estimation
//! - Block Bootstrap: Preserve autocorrelation in trade sequences
//! - Regime-Aware: Sample from similar market conditions

use crate::BacktestResult;
use crate::logging_facade::BACKTEST_LOGGER;
use crate::log_info;
use metrics::PerformanceMetrics;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Minimum realized (closed) trades before Monte Carlo output is treated as
/// reliable rather than a degenerate default. Below this, shuffled/bootstrapped
/// paths are nearly identical (too few trades to meaningfully reorder or
/// resample), and any reported percentile spread is an artifact of sample
/// size, not a real Monte Carlo estimate. Single source of truth -- every
/// call site that used to redeclare `const MC_MIN_TRADES: usize = 30;` locally
/// should reference this instead.
pub const MC_MIN_TRADES: usize = 30;

/// Monte Carlo run count once `MC_MIN_TRADES` is cleared. Deliberately NOT a
/// function of trade count -- a bootstrap percentile's own estimation noise
/// depends on how many times you resample a fixed trade set, not on how big
/// that set is (those are the two different things `MC_MIN_TRADES` and this
/// constant each answer). Empirically verified (quant council 2026-08-02) on
/// a real 285-trade production job (basket_arb, job e0f174cf): the p5
/// total-PnL estimate varied by ~4% of its own value across different random
/// seeds at 200 runs, ~1% at 2,000, ~0.4% at 10,000 -- consistent with the
/// theoretical sqrt(n) convergence rate for order-statistic estimation.
/// 2,000 is the point past which further runs buy little extra precision
/// against the Promising-verdict gate's `p5 total PnL > 0` check; raising it
/// further is cheap (each run is O(num_trades), microseconds) but not
/// obviously worth the extra compute over 2,000 given that measurement.
/// Previously every call site applied its own ad hoc floor (100 in three
/// validation-mode paths, 200 in others) with no shared justification for
/// either number -- this was organic drift across call sites written at
/// different times, not a deliberate per-mode design choice.
pub const DEFAULT_MC_RUNS: usize = 2000;

/// Resolve the Monte Carlo run count to actually use for a job: whatever was
/// requested (0 when the caller has no opinion), floored at `DEFAULT_MC_RUNS`.
/// There is no platform default below this -- see `DEFAULT_MC_RUNS`'s doc
/// comment for the empirical basis. A caller may still request MORE than the
/// default; this only ever raises a too-low or absent request, never lowers
/// an explicit one.
pub fn resolve_mc_runs(requested: usize) -> usize {
    requested.max(DEFAULT_MC_RUNS)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    pub num_runs: usize,
    pub mean_metrics: PerformanceMetrics,
    pub stddev_metrics: PerformanceMetrics,
    pub percentile_metrics: HashMap<u8, PerformanceMetrics>, // percentile -> metrics
    pub var_95: f64, // Value at Risk (95th percentile) in USD
    pub cvar_95: f64, // Conditional VaR (expected loss beyond 95th percentile) in USD
    /// Number of data points used for Monte Carlo (trade count)
    /// VaR/CVaR are unreliable if this is < 30
    #[serde(default)]
    pub data_points: usize,
    /// True if VaR/CVaR should be considered unreliable due to insufficient data
    #[serde(default)]
    pub var_unreliable: bool,
    /// Fan chart data: per-trade-step percentile equity bands (p5, p25, p50, p75, p95)
    /// Each entry is [p5, p25, p50, p75, p95] equity at that trade step.
    /// Length = trade count + 1 (includes initial capital at step 0).
    #[serde(default)]
    pub fan_chart: Vec<[f64; 5]>,
    /// Human-readable description of WHICH backtest result the Monte Carlo was
    /// computed from (e.g. "out-of-sample walk-forward (30 trades)" vs the
    /// in-sample backtest). The validation pipeline intentionally runs MC on the
    /// out-of-sample walk-forward result so the bands aren't inflated by
    /// in-sample overfit — which means the MC can describe a DIFFERENT, smaller
    /// trade set than the in-sample trade log / equity curve shown alongside it.
    /// Surfacing this lets the UI label the bands correctly instead of implying
    /// they reconcile trade-for-trade with the displayed backtest.
    #[serde(default)]
    pub base_source: String,
    /// Base RNG seed used for this MC run. Derived from the job's trade
    /// fingerprint (count + capital + first/last trade returns) so that the same
    /// job always produces the same MC paths. Stored for audit / reproducibility.
    #[serde(default, skip_serializing_if = "crate::monte_carlo::is_zero_u64")]
    pub mc_rng_seed: u64,
    /// Block-bootstrap 5th-percentile total PnL. Preserves trade-return
    /// autocorrelation — a more conservative tail bound than shuffle MC.
    /// None when block bootstrap was not run alongside the shuffle simulation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bb_p5_total_pnl: Option<f64>,
}

#[doc(hidden)]
pub fn is_zero_u64(v: &u64) -> bool { *v == 0 }

// ============================================
// Block Bootstrap Configuration
// ============================================

/// Configuration for block bootstrap Monte Carlo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBootstrapConfig {
    /// Block size for preserving autocorrelation (typically 5-20 trades)
    pub block_size: usize,
    /// Whether to use overlapping blocks (circular block bootstrap)
    pub overlapping: bool,
    /// Minimum block size (auto-adjusts if data is too short)
    pub min_block_size: usize,
}

impl Default for BlockBootstrapConfig {
    fn default() -> Self {
        Self {
            block_size: 10,
            overlapping: true,
            min_block_size: 3,
        }
    }
}

impl BlockBootstrapConfig {
    /// Optimal block size based on data length (Politis & White rule of thumb)
    pub fn optimal_block_size(data_length: usize) -> usize {
        // Rule of thumb: block_size ≈ n^(1/3)
        let optimal = (data_length as f64).powf(1.0 / 3.0).ceil() as usize;
        optimal.max(3).min(data_length / 4).max(1)
    }
}

// ============================================
// Regime-Aware Sampling Configuration
// ============================================

/// Market regime classification for regime-aware sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegimeClass {
    LowVolatility,
    NormalVolatility,
    HighVolatility,
    Crisis,
}

/// Configuration for regime-aware Monte Carlo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAwareConfig {
    /// Lookback window for volatility calculation
    pub volatility_lookback: usize,
    /// Current volatility level (for regime detection)
    pub current_volatility: f64,
    /// Regime similarity weight (0.0 = random, 1.0 = only same regime)
    pub regime_weight: f64,
    /// Volatility thresholds: [low/normal, normal/high, high/crisis]
    pub volatility_thresholds: [f64; 3],
}

impl Default for RegimeAwareConfig {
    fn default() -> Self {
        Self {
            volatility_lookback: 20,
            current_volatility: 0.02,
            regime_weight: 0.7,
            volatility_thresholds: [0.01, 0.025, 0.05], // 1%, 2.5%, 5% daily vol
        }
    }
}

impl RegimeAwareConfig {
    /// Classify volatility into regime
    pub fn classify_regime(&self, volatility: f64) -> MarketRegimeClass {
        if volatility < self.volatility_thresholds[0] {
            MarketRegimeClass::LowVolatility
        } else if volatility < self.volatility_thresholds[1] {
            MarketRegimeClass::NormalVolatility
        } else if volatility < self.volatility_thresholds[2] {
            MarketRegimeClass::HighVolatility
        } else {
            MarketRegimeClass::Crisis
        }
    }
    
    /// Get regime for current conditions
    pub fn current_regime(&self) -> MarketRegimeClass {
        self.classify_regime(self.current_volatility)
    }
}

/// Helper functions for aggregation
#[inline]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[inline]
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        let mean_val = mean(values);
        let variance = values.iter()
            .map(|x| (x - mean_val).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }
}

#[inline]
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = (p / 100.0 * (values.len() - 1) as f64).round() as usize;
    values[index.min(values.len() - 1)]
}

/// Extract percentile from pre-sorted slice (no re-sorting needed)
#[inline]
fn percentile_from_sorted(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index = (p / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

/// Pre-computed constant for annualization (365 days for crypto markets)
const SQRT_365: f64 = 19.1049731745428; // sqrt(365)

/// Run Monte Carlo simulation with progress callback
/// 
/// This simulates alternative "what if" scenarios by shuffling the order of trades
/// while keeping the same trades, helping assess if results are due to luck or skill.
pub fn run_monte_carlo_simulation_with_progress(
    base_result: &BacktestResult,
    num_runs: usize,
    progress_callback: Option<std::sync::Arc<dyn Fn(&str, usize, usize) + Send + Sync>>,
) -> anyhow::Result<MonteCarloResult> {
    run_monte_carlo_simulation_internal(base_result, num_runs, progress_callback)
}

/// Run Monte Carlo simulation by randomizing trade sequences
/// 
/// This simulates alternative "what if" scenarios by shuffling the order of trades
/// while keeping the same trades, helping assess if results are due to luck or skill.
pub fn run_monte_carlo_simulation(
    base_result: &BacktestResult,
    num_runs: usize,
) -> anyhow::Result<MonteCarloResult> {
    run_monte_carlo_simulation_internal(base_result, num_runs, None)
}

/// Run lightweight Monte Carlo for genetic fitness evaluation
/// 
/// Returns only the mean and standard deviation of Sharpe ratios for fast consistency checking.
/// Used in lazy evaluation to assess parameter stability without full analysis.
pub fn run_lightweight_monte_carlo(
    base_result: &BacktestResult,
    num_runs: usize,
) -> anyhow::Result<(f64, f64)> {
    // Use actual trade returns (not candle-level equity deltas)
    let trade_returns: Vec<f64> = if !base_result.trade_returns.is_empty() {
        base_result.trade_returns.clone()
    } else if base_result.equity_curve.len() > 1 {
        // Fallback: derive from equity curve, filter out zero deltas
        base_result.equity_curve.windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| d.abs() > 1e-10)
            .collect()
    } else {
        Vec::new()
    };
    
    // If no trades, return zero mean/stddev
    if trade_returns.is_empty() {
        return Ok((0.0, 0.0));
    }
    
    let n = trade_returns.len();
    let base_seed = trade_fingerprint_seed(base_result);

    // NOTE: shuffling a return series does NOT change its mean or standard
    // deviation (both are order-invariant), so a shuffle-based Sharpe would be
    // identical on every run. Bootstrap resampling (with replacement) is used
    // instead so the Sharpe distribution actually varies across runs.
    let sharpe_ratios: Vec<f64> = (0..num_runs)
        .into_par_iter()
        .map(|run_idx| {
            let mut rng = StdRng::seed_from_u64(base_seed ^ run_idx as u64);

            // Welford's online algorithm for single-pass mean and variance
            let mut mean_acc = 0.0_f64;
            let mut m2 = 0.0_f64;

            for i in 0..n {
                let ret = trade_returns[rng.gen_range(0..n)];
                let delta = ret - mean_acc;
                mean_acc += delta / (i + 1) as f64;
                let delta2 = ret - mean_acc;
                m2 += delta * delta2;
            }
            
            // Calculate Sharpe ratio
            if n > 1 && m2 > 0.0 {
                let variance = m2 / (n - 1) as f64;
                let std_ret = variance.sqrt();
                if std_ret > 0.0 {
                    mean_acc / std_ret * SQRT_365 // Annualized
                } else {
                    0.0
                }
            } else {
                0.0
            }
        })
        .collect();
    
    // Calculate mean and standard deviation of Sharpe ratios using Welford's
    let mut mean_sharpe = 0.0_f64;
    let mut m2_sharpe = 0.0_f64;
    
    for (i, &sharpe) in sharpe_ratios.iter().enumerate() {
        let delta = sharpe - mean_sharpe;
        mean_sharpe += delta / (i + 1) as f64;
        let delta2 = sharpe - mean_sharpe;
        m2_sharpe += delta * delta2;
    }
    
    let std_sharpe = if num_runs > 1 {
        (m2_sharpe / (num_runs - 1) as f64).sqrt()
    } else {
        0.0
    };
    
    Ok((mean_sharpe, std_sharpe))
}

/// Per-trade **account-level** fractional returns used for Monte Carlo equity
/// reconstruction. Each value is `net_pnl / initial_capital` — the trade's P&L
/// expressed as a fraction of the account base.
///
/// Compounding these (`equity *= 1 + frac`) is:
///   * bounded — avoids the negative-equity / >100% drawdown problem that
///     summing dollar P&Ls causes when stitching walk-forward windows
///     (each window referencing its own fresh capital base), and
///   * on the correct **account** scale, so the simulated bands line up with
///     the realized equity curve (`equity += net_pnl`).
///
/// Bug #33 — do NOT use `trade_pct_returns` here. That field is return-on-
/// POSITION-notional (`net_pnl / cost_basis`); a ~$600 position on a $10k
/// account makes it ~16x larger than the true account return. Compounding it
/// as an account return inflated the MC median far above the realized result
/// (e.g. +96% bands on a strategy that actually lost money).
fn account_fractional_returns(base_result: &BacktestResult) -> Vec<f64> {
    let initial_capital = base_result.initial_capital;
    if initial_capital <= 0.0 {
        return Vec::new();
    }
    if !base_result.trade_returns.is_empty() {
        // Dollar P&L per closed trade ÷ account base.
        base_result.trade_returns.iter().map(|&d| d / initial_capital).collect()
    } else if base_result.equity_curve.len() > 1 {
        // Fallback for built-in strategies that don't populate per-trade P&L:
        // derive dollar deltas from the equity curve, then express as account
        // fractions. (Less accurate — candle-level, not trade-level.)
        base_result.equity_curve.windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| d.abs() > 1e-10) // Remove zero deltas (no-trade candles)
            .map(|d| d / initial_capital)
            .collect()
    } else {
        Vec::new()
    }
}

/// Derive a deterministic u64 seed from a job's trade fingerprint. Mixes trade
/// count, initial capital, and the first + last 5 trade returns so that any two
/// jobs with different trade histories get different seeds, while a single job
/// always produces the same MC paths regardless of which machine runs it.
pub fn trade_fingerprint_seed(base_result: &BacktestResult) -> u64 {
    let mut hasher = DefaultHasher::new();
    base_result.num_trades.hash(&mut hasher);
    base_result.initial_capital.to_bits().hash(&mut hasher);
    let returns = &base_result.trade_returns;
    for r in returns.iter().take(5).chain(returns.iter().rev().take(5)) {
        r.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn run_monte_carlo_simulation_internal(
    base_result: &BacktestResult,
    num_runs: usize,
    progress_callback: Option<std::sync::Arc<dyn Fn(&str, usize, usize) + Send + Sync>>,
) -> anyhow::Result<MonteCarloResult> {
    let base_seed = trade_fingerprint_seed(base_result);
    // ========================================================================
    // TRADE-LEVEL Monte Carlo: shuffle/bootstrap TRADE P&Ls, not candle deltas.
    //
    // The equity curve has one point per candle (~155K for 6 months @ 1min),
    // but most candle-to-candle deltas are 0 (no trade happening). Shuffling
    // candle deltas produces nearly identical paths every run. Instead, we
    // operate on per-trade returns (one value per closed trade), which
    // correctly tests path sensitivity to trade ordering.
    //
    // We work in ACCOUNT-LEVEL fractional returns (net_pnl / initial_capital)
    // and compound them. See account_fractional_returns() for why this is both
    // bounded (walk-forward stitching) and on the correct account scale.
    // ========================================================================

    let account_fracs: Vec<f64> = account_fractional_returns(base_result);

    let num_trades = account_fracs.len();

    // MAE (max adverse excursion) per trade — used to inject intra-trade
    // drawdowns into the reconstructed equity paths.
    let trade_maes: Vec<f64> = if base_result.trade_maes.len() == num_trades {
        base_result.trade_maes.clone()
    } else {
        vec![0.0; num_trades]
    };

    let initial_capital = base_result.initial_capital;
    // With <30 trades, shuffled/bootstrapped paths are nearly identical.
    if num_trades < 30 {
        log_info!(BACKTEST_LOGGER, "Monte Carlo: only {} trades (<30 minimum), returning unreliable result", num_trades);
        let mut percentiles = HashMap::new();
        percentiles.insert(5, PerformanceMetrics::default());
        percentiles.insert(25, PerformanceMetrics::default());
        percentiles.insert(50, PerformanceMetrics::default());
        percentiles.insert(75, PerformanceMetrics::default());
        percentiles.insert(95, PerformanceMetrics::default());

        return Ok(MonteCarloResult {
            num_runs,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: percentiles,
            var_95: 0.0,
            cvar_95: 0.0,
            data_points: num_trades,
            var_unreliable: true,
            fan_chart: vec![[initial_capital; 5]],
            base_source: String::new(),
            mc_rng_seed: base_seed,
            bb_p5_total_pnl: None,
        });
    }

    // Legacy guard for exactly zero trades (shouldn't reach here after the above)
    if num_trades == 0 {
        let mut percentiles = HashMap::new();
        percentiles.insert(5, PerformanceMetrics::default());
        percentiles.insert(25, PerformanceMetrics::default());
        percentiles.insert(50, PerformanceMetrics::default());
        percentiles.insert(75, PerformanceMetrics::default());
        percentiles.insert(95, PerformanceMetrics::default());

        return Ok(MonteCarloResult {
            num_runs,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: percentiles,
            var_95: 0.0,
            cvar_95: 0.0,
            data_points: 0,
            var_unreliable: true,
            fan_chart: vec![[initial_capital; 5]],
            base_source: String::new(),
            mc_rng_seed: base_seed,
            bb_p5_total_pnl: None,
        });
    }

    // At least one bootstrap run is required: shuffle runs produce identical
    // net_profit / Sharpe / win_rate on every run (those statistics are
    // order-invariant), so all distribution statistics come from the
    // bootstrap half. Shuffle runs contribute only path-dependent statistics
    // (max drawdown and the fan chart).
    let bootstrap_runs = (num_runs / 2).max(1).min(num_runs);
    let shuffle_runs = num_runs - bootstrap_runs;

    let completed_runs = AtomicUsize::new(0);
    let log_interval = (num_runs / 10).max(100);

    log_info!(BACKTEST_LOGGER, "Monte Carlo starting | total_runs={} trades={} shuffle={} bootstrap={}",
              num_runs, num_trades, shuffle_runs, bootstrap_runs);

    // ========================================================================
    // Method 1: Trade order shuffling (tests path dependency)
    // Same trades, different order → different drawdown and Sharpe paths
    // ========================================================================
    let shuffle_results: Vec<(f64, f64, f64, f64, Vec<f64>)> = (0..shuffle_runs)
        .into_par_iter()
        .map(|run_idx| {
            let completed = completed_runs.fetch_add(1, Ordering::Relaxed);
            if completed % log_interval == 0 {
                log_info!(BACKTEST_LOGGER, "Monte Carlo progress | run={}/{} ({:.0}%)",
                         completed, num_runs, (completed as f64 / num_runs as f64) * 100.0);
            }

            let mut rng = StdRng::seed_from_u64(base_seed ^ run_idx as u64);
            let mut indices: Vec<usize> = (0..num_trades).collect();
            indices.shuffle(&mut rng);

            // Build equity path from shuffled trade returns
            let mut equity_path = Vec::with_capacity(num_trades + 1);
            equity_path.push(initial_capital);
            let mut equity = initial_capital;
            let mut peak = initial_capital;
            let mut max_dd = 0.0_f64;
            let mut mean_acc = 0.0_f64;
            let mut m2 = 0.0_f64;
            let mut wins = 0_usize;

            for (i, &idx) in indices.iter().enumerate() {
                let frac = account_fracs[idx];
                let mae_val = trade_maes[idx];

                // Compound the account-level fractional return on CURRENT equity.
                let ret = equity * frac;
                // MAE is stored in dollars vs the window's initial-capital base;
                // scale it proportionally to current equity when compounding.
                let mae_dollar = mae_val * (equity / initial_capital).max(0.0);

                // Check intra-trade trough before applying the return
                if mae_dollar > 0.0 {
                    let trough = equity - mae_dollar;
                    let dd = if peak > 0.0 { (peak - trough) / peak } else { 0.0 };
                    if dd > max_dd { max_dd = dd; }
                }

                equity += ret;
                equity_path.push(equity);
                if equity > peak { peak = equity; }
                let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
                if dd > max_dd { max_dd = dd; }
                if frac > 0.0 { wins += 1; }

                // Welford on the per-trade account return so the MC Sharpe is
                // consistent with the reconstructed equity paths.
                let delta = frac - mean_acc;
                mean_acc += delta / (i + 1) as f64;
                let delta2 = frac - mean_acc;
                m2 += delta * delta2;
            }

            let net_profit = equity - initial_capital;
            let win_rate = wins as f64 / num_trades as f64;
            let sharpe = if num_trades > 1 && m2 > 0.0 {
                let std_ret = (m2 / (num_trades - 1) as f64).sqrt();
                if std_ret > 0.0 { mean_acc / std_ret * SQRT_365 } else { 0.0 }
            } else { 0.0 };

            (net_profit, sharpe, max_dd, win_rate, equity_path)
        }).collect();

    // Collect results.
    // Shuffle runs have order-invariant net_profit/sharpe/win_rate (identical
    // every run), so only their path-dependent outputs (max drawdown, equity
    // path) enter the aggregates. Distribution statistics for net_profit,
    // Sharpe, and win rate come exclusively from the bootstrap runs below.
    let mut all_net_profits = Vec::with_capacity(bootstrap_runs);
    let mut all_sharpe_ratios = Vec::with_capacity(bootstrap_runs);
    let mut all_max_drawdowns = Vec::with_capacity(num_runs);
    let mut all_win_rates = Vec::with_capacity(bootstrap_runs);
    let mut all_equity_paths: Vec<Vec<f64>> = Vec::with_capacity(num_runs);

    for (_net_profit, _sharpe, max_dd, _win_rate, eq_path) in shuffle_results {
        all_max_drawdowns.push(max_dd);
        all_equity_paths.push(eq_path);
    }

    if let Some(ref callback) = progress_callback {
        callback("monte_carlo", shuffle_runs, num_runs);
    }

    // ========================================================================
    // Method 2: Bootstrap resampling (tests return distribution sensitivity)
    // Sample trades WITH replacement → can change win rate and trade mix
    // ========================================================================
    let bootstrap_results: Vec<(f64, f64, f64, f64, Vec<f64>)> = (0..bootstrap_runs)
        .into_par_iter()
        .map(|run_idx| {
            let completed = completed_runs.fetch_add(1, Ordering::Relaxed);
            if completed % log_interval == 0 {
                log_info!(BACKTEST_LOGGER, "Monte Carlo progress | run={}/{} ({:.0}%)",
                         completed, num_runs, (completed as f64 / num_runs as f64) * 100.0);
            }

            // Offset seed by shuffle_runs to avoid overlap with shuffle seeds
            let mut rng = StdRng::seed_from_u64(base_seed ^ (shuffle_runs + run_idx) as u64);

            // Bootstrap: sample num_trades returns with replacement
            let mut equity_path = Vec::with_capacity(num_trades + 1);
            equity_path.push(initial_capital);
            let mut equity = initial_capital;
            let mut peak = initial_capital;
            let mut max_dd = 0.0_f64;
            let mut mean_acc = 0.0_f64;
            let mut m2 = 0.0_f64;
            let mut wins = 0_usize;

            for i in 0..num_trades {
                let idx = rng.gen_range(0..num_trades);
                let frac = account_fracs[idx];
                let mae_val = trade_maes[idx];

                // Compound the account-level fractional return on CURRENT equity.
                let ret = equity * frac;
                // MAE is stored in dollars vs the window's initial-capital base;
                // scale it proportionally to current equity when compounding.
                let mae_dollar = mae_val * (equity / initial_capital).max(0.0);

                // Check intra-trade trough before applying the return
                if mae_dollar > 0.0 {
                    let trough = equity - mae_dollar;
                    let dd = if peak > 0.0 { (peak - trough) / peak } else { 0.0 };
                    if dd > max_dd { max_dd = dd; }
                }

                equity += ret;
                equity_path.push(equity);
                if equity > peak { peak = equity; }
                let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
                if dd > max_dd { max_dd = dd; }
                if frac > 0.0 { wins += 1; }

                // Welford on the per-trade account return so the MC Sharpe is
                // consistent with the reconstructed equity paths.
                let delta = frac - mean_acc;
                mean_acc += delta / (i + 1) as f64;
                let delta2 = frac - mean_acc;
                m2 += delta * delta2;
            }

            let net_profit = equity - initial_capital;
            let win_rate = wins as f64 / num_trades as f64;
            let sharpe = if num_trades > 1 && m2 > 0.0 {
                let std_ret = (m2 / (num_trades - 1) as f64).sqrt();
                if std_ret > 0.0 { mean_acc / std_ret * SQRT_365 } else { 0.0 }
            } else { 0.0 };

            (net_profit, sharpe, max_dd, win_rate, equity_path)
        }).collect();

    for (net_profit, sharpe, max_dd, win_rate, eq_path) in bootstrap_results {
        all_net_profits.push(net_profit);
        all_sharpe_ratios.push(sharpe);
        all_max_drawdowns.push(max_dd);
        all_win_rates.push(win_rate);
        all_equity_paths.push(eq_path);
    }

    if let Some(ref callback) = progress_callback {
        callback("monte_carlo", num_runs, num_runs);
    }

    // ========================================================================
    // Aggregate statistics
    // ========================================================================
    let mean_net_profit = mean(&all_net_profits);
    let std_net_profit = std_dev(&all_net_profits);
    let mean_sharpe = mean(&all_sharpe_ratios);
    let std_sharpe = std_dev(&all_sharpe_ratios);
    let mean_dd = mean(&all_max_drawdowns);
    let std_dd = std_dev(&all_max_drawdowns);
    let mean_win_rate = mean(&all_win_rates);
    let std_win_rate = std_dev(&all_win_rates);

    // Sort for percentiles
    all_net_profits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_sharpe_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_max_drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_win_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut percentiles = HashMap::new();
    for &p in &[5, 25, 50, 75, 95] {
        percentiles.insert(p, PerformanceMetrics {
            net_profit: percentile_from_sorted(&all_net_profits, p as f64),
            sharpe_ratio: percentile_from_sorted(&all_sharpe_ratios, p as f64),
            max_drawdown: percentile_from_sorted(&all_max_drawdowns, p as f64),
            win_rate: percentile_from_sorted(&all_win_rates, p as f64),
            gross_profit: 0.0,
            gross_loss: 0.0,
            profit_factor: 0.0,
            average_trade_return: 0.0,
            median_trade_return: 0.0,
            number_of_trades: base_result.num_trades,
            volatility: base_result.volatility.unwrap_or(0.0),
            average_trade_duration: base_result.avg_trade_duration.unwrap_or(0.0),
            total_commission: 0.0,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            transaction_costs_percentage: 0.0,
        });
    }

    // ========================================================================
    // Fan chart: per-trade-step percentile bands [p5, p25, p50, p75, p95]
    // ========================================================================
    let path_len = num_trades + 1;
    let mut fan_chart = Vec::with_capacity(path_len);
    for step in 0..path_len {
        let mut step_equities: Vec<f64> = all_equity_paths.iter()
            .filter_map(|path| path.get(step).copied())
            .collect();
        step_equities.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if step_equities.is_empty() {
            fan_chart.push([initial_capital; 5]);
        } else {
            fan_chart.push([
                percentile_from_sorted(&step_equities, 5.0),
                percentile_from_sorted(&step_equities, 25.0),
                percentile_from_sorted(&step_equities, 50.0),
                percentile_from_sorted(&step_equities, 75.0),
                percentile_from_sorted(&step_equities, 95.0),
            ]);
        }
    }

    // ========================================================================
    // VaR / CVaR
    // ========================================================================
    // Bug #32 \u2014 ensure VaR/CVaR sign convention is consistent (both positive
    // loss magnitudes) and that VaR \u2264 CVaR holds. The previous implementation
    // could emit `var_95 = 0` with `cvar_95 > 0` whenever the 5th percentile
    // was non-negative but the average of the worst 5% was a loss, which is
    // confusing to consumers. When the 5th-percentile sample is non-negative
    // we fall back to the worst single sample inside the tail bucket so VaR
    // still carries a meaningful loss magnitude.
    let var_95_index = (0.05 * all_net_profits.len() as f64) as usize;
    let safe_idx = var_95_index.min(all_net_profits.len().saturating_sub(1));
    let percentile_5_profit = all_net_profits[safe_idx];
    let losses_beyond_var: Vec<f64> = all_net_profits.iter()
        .take(var_95_index)
        .copied()
        .collect();
    let cvar_95 = if !losses_beyond_var.is_empty() {
        let avg_worst = mean(&losses_beyond_var);
        if avg_worst < 0.0 { -avg_worst } else { 0.0 }
    } else {
        0.0
    };
    let var_95 = {
        let raw = if percentile_5_profit < 0.0 { -percentile_5_profit } else { 0.0 };
        if raw <= 0.0 && cvar_95 > 0.0 {
            // Tail losses exist below the 5% cutoff even though the cutoff sample
            // is non-negative. Use the worst single sample in the tail bucket
            // (which is by definition the most negative entry) so VaR reflects
            // a real loss magnitude rather than masking the risk as zero.
            losses_beyond_var.iter().cloned().fold(0.0_f64, f64::min).abs()
        } else {
            raw
        }
    };
    // Enforce the standard inequality VaR \u2264 CVaR (both as magnitudes).
    let var_95 = var_95.min(cvar_95).max(0.0);

    log_info!(BACKTEST_LOGGER, "Monte Carlo complete | runs={} trades={} mean_sharpe={:.3} VaR95=${:.2} CVaR95=${:.2}",
              num_runs, num_trades, mean_sharpe, var_95, cvar_95);

    Ok(MonteCarloResult {
        num_runs,
        mean_metrics: PerformanceMetrics {
            net_profit: mean_net_profit,
            gross_profit: base_result.gross_profit.unwrap_or(0.0),
            gross_loss: base_result.gross_loss.unwrap_or(0.0),
            sharpe_ratio: mean_sharpe,
            max_drawdown: mean_dd,
            win_rate: mean_win_rate,
            profit_factor: base_result.profit_factor.unwrap_or(0.0),
            average_trade_return: base_result.avg_trade_return.unwrap_or(0.0),
            median_trade_return: base_result.median_trade_return.unwrap_or(0.0),
            number_of_trades: base_result.num_trades,
            volatility: base_result.volatility.unwrap_or(0.0),
            average_trade_duration: base_result.avg_trade_duration.unwrap_or(0.0),
            total_commission: base_result.transaction_costs.as_ref().map(|tc| tc.total_commission).unwrap_or(0.0),
            total_slippage: base_result.transaction_costs.as_ref().map(|tc| tc.total_slippage).unwrap_or(0.0),
            total_market_impact: base_result.transaction_costs.as_ref().map(|tc| tc.total_market_impact).unwrap_or(0.0),
            transaction_costs_percentage: base_result.transaction_costs.as_ref().map(|tc| tc.costs_as_percentage_of_pnl).unwrap_or(0.0),
        },
        stddev_metrics: PerformanceMetrics {
            net_profit: std_net_profit,
            gross_profit: 0.0,
            gross_loss: 0.0,
            sharpe_ratio: std_sharpe,
            max_drawdown: std_dd,
            win_rate: std_win_rate,
            profit_factor: 0.0,
            average_trade_return: 0.0,
            median_trade_return: 0.0,
            number_of_trades: 0,
            volatility: 0.0,
            average_trade_duration: 0.0,
            total_commission: 0.0,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            transaction_costs_percentage: 0.0,
        },
        percentile_metrics: percentiles,
        var_95,
        cvar_95,
        data_points: num_trades,
        var_unreliable: num_trades < 30,
        fan_chart,
        base_source: String::new(),
        mc_rng_seed: base_seed,
        bb_p5_total_pnl: None,
    })
}

// ============================================
// Block Bootstrap Monte Carlo
// ============================================

/// Run block bootstrap Monte Carlo simulation
/// 
/// Preserves autocorrelation in trade sequences by sampling blocks of consecutive trades
/// rather than individual trades. This is important for strategies where trade outcomes
/// are correlated (e.g., trend following).
pub fn run_block_bootstrap_simulation(
    base_result: &BacktestResult,
    num_runs: usize,
    config: BlockBootstrapConfig,
) -> anyhow::Result<MonteCarloResult> {
    // Use actual per-trade returns (not candle-level equity deltas), expressed
    // as ACCOUNT-level fractions (net_pnl / initial_capital) and compounded
    // below. See account_fractional_returns() for the rationale (bounded across
    // stitched walk-forward windows AND on the correct account scale).
    let equity_deltas: Vec<f64> = account_fractional_returns(base_result);
    
    // MAE per trade for intra-trade drawdown injection
    let bb_trade_maes: Vec<f64> = if base_result.trade_maes.len() == equity_deltas.len() {
        base_result.trade_maes.clone()
    } else {
        vec![0.0; equity_deltas.len()]
    };

    if equity_deltas.is_empty() {
        return run_monte_carlo_simulation(base_result, num_runs);
    }
    
    // Determine effective block size
    let block_size = config.block_size
        .max(config.min_block_size)
        .min(equity_deltas.len() / 2)
        .max(1);
    
    let n = equity_deltas.len();
    let base_seed = trade_fingerprint_seed(base_result);

    // Run block bootstrap in parallel
    let results: Vec<(f64, f64, f64)> = (0..num_runs)
        .into_par_iter()
        .map(|run_idx| {
            let mut rng = StdRng::seed_from_u64(base_seed ^ run_idx as u64);

            // Sample blocks to reconstruct sequence
            let mut bootstrap_sample = Vec::with_capacity(n);
            let mut bootstrap_mae_sample = Vec::with_capacity(n);
            
            while bootstrap_sample.len() < n {
                // Random starting point for block
                let start_idx = if config.overlapping {
                    rng.gen_range(0..n)
                } else {
                    rng.gen_range(0..n.saturating_sub(block_size).max(1))
                };
                
                // Extract block (with wrap-around for circular bootstrap)
                for offset in 0..block_size {
                    if bootstrap_sample.len() >= n {
                        break;
                    }
                    let idx = if config.overlapping {
                        (start_idx + offset) % n // Circular
                    } else {
                        (start_idx + offset).min(n - 1)
                    };
                    bootstrap_sample.push(equity_deltas[idx]);
                    bootstrap_mae_sample.push(bb_trade_maes[idx]);
                }
            }
            
            // Truncate to exact length
            bootstrap_sample.truncate(n);
            bootstrap_mae_sample.truncate(n);
            
            // Calculate metrics
            let mean_ret = if !bootstrap_sample.is_empty() {
                bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64
            } else {
                0.0
            };
            
            let std_ret = if bootstrap_sample.len() > 1 {
                let variance = bootstrap_sample.iter()
                    .map(|x| (x - mean_ret).powi(2))
                    .sum::<f64>() / (bootstrap_sample.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };
            
            let sharpe = if std_ret > 0.0 {
                mean_ret / std_ret * SQRT_365
            } else {
                0.0
            };
            
            // Reconstruct equity path by compounding account-level fractional
            // returns, with MAE-augmented intra-trade trough.
            let initial_capital = base_result.initial_capital;
            let mut equity = initial_capital;
            let mut peak = equity;
            let mut max_dd = 0.0;
            for (i, &frac) in bootstrap_sample.iter().enumerate() {
                let mae_val = bootstrap_mae_sample[i];
                let ret = equity * frac;
                let mae_dollar = mae_val * (equity / initial_capital).max(0.0);
                if mae_dollar > 0.0 {
                    let trough = equity - mae_dollar;
                    let dd = if peak > 0.0 { (peak - trough) / peak } else { 0.0 };
                    if dd > max_dd { max_dd = dd; }
                }
                equity += ret;
                if equity > peak {
                    peak = equity;
                }
                let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            let net_profit = equity - initial_capital;
            
            (net_profit, sharpe, max_dd)
        })
        .collect();
    
    // Aggregate results
    let all_net_profits: Vec<f64> = results.iter().map(|r| r.0).collect();
    let all_sharpes: Vec<f64> = results.iter().map(|r| r.1).collect();
    let all_drawdowns: Vec<f64> = results.iter().map(|r| r.2).collect();
    
    // Build result using helper functions
    build_monte_carlo_result(
        base_result,
        num_runs,
        all_net_profits,
        all_sharpes,
        all_drawdowns,
    )
}

// ============================================
// Regime-Aware Monte Carlo
// ============================================

/// Run regime-aware Monte Carlo simulation
/// 
/// Samples trades from similar market regimes to provide more realistic
/// expectations for current conditions. Uses volatility-based regime classification.
pub fn run_regime_aware_simulation(
    base_result: &BacktestResult,
    num_runs: usize,
    config: RegimeAwareConfig,
    volatility_per_trade: &[f64], // Volatility at each trade
) -> anyhow::Result<MonteCarloResult> {
    // Use actual per-trade returns (not candle-level equity deltas), expressed
    // as ACCOUNT-level fractions (net_pnl / initial_capital) and compounded
    // below. See account_fractional_returns() for the rationale.
    let equity_deltas: Vec<f64> = account_fractional_returns(base_result);
    
    if equity_deltas.is_empty() {
        return run_monte_carlo_simulation(base_result, num_runs);
    }

    // MAE per trade for intra-trade drawdown injection
    let ra_trade_maes: Vec<f64> = if base_result.trade_maes.len() == equity_deltas.len() {
        base_result.trade_maes.clone()
    } else {
        vec![0.0; equity_deltas.len()]
    };

    // Classify each trade into a regime
    let trade_regimes: Vec<MarketRegimeClass> = volatility_per_trade.iter()
        .map(|&vol| config.classify_regime(vol))
        .collect();
    
    // If volatility data doesn't match, use uniform regime
    let trade_regimes = if trade_regimes.len() != equity_deltas.len() {
        vec![MarketRegimeClass::NormalVolatility; equity_deltas.len()]
    } else {
        trade_regimes
    };
    
    let current_regime = config.current_regime();
    let regime_weight = config.regime_weight.clamp(0.0, 1.0);
    
    // Group trades by regime
    let mut regime_indices: HashMap<MarketRegimeClass, Vec<usize>> = HashMap::new();
    for (idx, &regime) in trade_regimes.iter().enumerate() {
        regime_indices.entry(regime).or_default().push(idx);
    }
    
    let n = equity_deltas.len();
    let base_seed = trade_fingerprint_seed(base_result);

    // Run regime-aware sampling in parallel
    let results: Vec<(f64, f64, f64)> = (0..num_runs)
        .into_par_iter()
        .map(|run_idx| {
            let mut rng = StdRng::seed_from_u64(base_seed ^ run_idx as u64);
            let mut bootstrap_sample = Vec::with_capacity(n);
            let mut mae_sample = Vec::with_capacity(n);
            
            for _ in 0..n {
                // Weighted selection: prefer same regime with probability regime_weight
                let use_same_regime = rng.gen::<f64>() < regime_weight;
                
                let selected_idx = if use_same_regime {
                    // Sample from same regime
                    if let Some(indices) = regime_indices.get(&current_regime) {
                        if !indices.is_empty() {
                            indices[rng.gen_range(0..indices.len())]
                        } else {
                            rng.gen_range(0..n)
                        }
                    } else {
                        rng.gen_range(0..n)
                    }
                } else {
                    // Random sample from any regime
                    rng.gen_range(0..n)
                };
                
                bootstrap_sample.push(equity_deltas[selected_idx]);
                mae_sample.push(ra_trade_maes[selected_idx]);
            }
            
            // Calculate metrics
            let mean_ret = if !bootstrap_sample.is_empty() {
                bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64
            } else {
                0.0
            };
            
            let std_ret = if bootstrap_sample.len() > 1 {
                let variance = bootstrap_sample.iter()
                    .map(|x| (x - mean_ret).powi(2))
                    .sum::<f64>() / (bootstrap_sample.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };
            
            let sharpe = if std_ret > 0.0 {
                mean_ret / std_ret * SQRT_365
            } else {
                0.0
            };
            
            // Reconstruct equity path by compounding account-level fractional
            // returns, with MAE-augmented intra-trade trough.
            let initial_capital = base_result.initial_capital;
            let mut equity = initial_capital;
            let mut peak = equity;
            let mut max_dd = 0.0;
            for (i, &frac) in bootstrap_sample.iter().enumerate() {
                let mae_val = mae_sample[i];
                let ret = equity * frac;
                let mae_dollar = mae_val * (equity / initial_capital).max(0.0);
                if mae_dollar > 0.0 {
                    let trough = equity - mae_dollar;
                    let dd = if peak > 0.0 { (peak - trough) / peak } else { 0.0 };
                    if dd > max_dd { max_dd = dd; }
                }
                equity += ret;
                if equity > peak {
                    peak = equity;
                }
                let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            let net_profit = equity - initial_capital;
            
            (net_profit, sharpe, max_dd)
        })
        .collect();
    
    // Aggregate results
    let all_net_profits: Vec<f64> = results.iter().map(|r| r.0).collect();
    let all_sharpes: Vec<f64> = results.iter().map(|r| r.1).collect();
    let all_drawdowns: Vec<f64> = results.iter().map(|r| r.2).collect();
    
    build_monte_carlo_result(
        base_result,
        num_runs,
        all_net_profits,
        all_sharpes,
        all_drawdowns,
    )
}

/// Helper to build MonteCarloResult from simulation outputs
fn build_monte_carlo_result(
    base_result: &BacktestResult,
    num_runs: usize,
    all_net_profits: Vec<f64>,
    all_sharpes: Vec<f64>,
    all_drawdowns: Vec<f64>,
) -> anyhow::Result<MonteCarloResult> {
    let mean_net_profit = mean(&all_net_profits);
    let std_net_profit = std_dev(&all_net_profits);
    let mean_sharpe = mean(&all_sharpes);
    let std_sharpe = std_dev(&all_sharpes);
    let mean_dd = mean(&all_drawdowns);
    let std_dd = std_dev(&all_drawdowns);
    
    // Win rate from base result
    let base_win_rate = base_result.win_rate.unwrap_or(0.0);
    
    // Calculate percentiles
    let mut percentiles = HashMap::new();
    for &p in &[5, 25, 50, 75, 95] {
        let mut profits_copy = all_net_profits.clone();
        let mut sharpe_copy = all_sharpes.clone();
        let mut dd_copy = all_drawdowns.clone();
        
        percentiles.insert(p, PerformanceMetrics {
            net_profit: percentile(&mut profits_copy, p as f64),
            sharpe_ratio: percentile(&mut sharpe_copy, p as f64),
            max_drawdown: percentile(&mut dd_copy, p as f64),
            win_rate: base_win_rate,
            gross_profit: 0.0,
            gross_loss: 0.0,
            profit_factor: 0.0,
            average_trade_return: 0.0,
            median_trade_return: 0.0,
            number_of_trades: base_result.num_trades,
            volatility: base_result.volatility.unwrap_or(0.0),
            average_trade_duration: base_result.avg_trade_duration.unwrap_or(0.0),
            total_commission: 0.0,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            transaction_costs_percentage: 0.0,
        });
    }
    
    // VaR and CVaR
    let mut sorted_profits = all_net_profits.clone();
    sorted_profits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let var_95_index = (0.05 * sorted_profits.len() as f64) as usize;
    let percentile_5_profit = sorted_profits[var_95_index];
    
    let var_95 = if percentile_5_profit < 0.0 {
        -percentile_5_profit
    } else {
        0.0
    };
    
    let losses_beyond_var: Vec<f64> = sorted_profits.iter()
        .take(var_95_index)
        .copied()
        .collect();
    let cvar_95 = if !losses_beyond_var.is_empty() {
        let avg_worst = mean(&losses_beyond_var);
        if avg_worst < 0.0 { -avg_worst } else { 0.0 }
    } else {
        var_95
    };
    
    let data_points = base_result.trade_returns.len();
    
    Ok(MonteCarloResult {
        num_runs,
        mean_metrics: PerformanceMetrics {
            net_profit: mean_net_profit,
            gross_profit: base_result.gross_profit.unwrap_or(0.0),
            gross_loss: base_result.gross_loss.unwrap_or(0.0),
            sharpe_ratio: mean_sharpe,
            max_drawdown: mean_dd,
            win_rate: base_win_rate,
            profit_factor: base_result.profit_factor.unwrap_or(0.0),
            average_trade_return: base_result.avg_trade_return.unwrap_or(0.0),
            median_trade_return: base_result.median_trade_return.unwrap_or(0.0),
            number_of_trades: base_result.num_trades,
            volatility: base_result.volatility.unwrap_or(0.0),
            average_trade_duration: base_result.avg_trade_duration.unwrap_or(0.0),
            total_commission: base_result.transaction_costs.as_ref().map(|tc| tc.total_commission).unwrap_or(0.0),
            total_slippage: base_result.transaction_costs.as_ref().map(|tc| tc.total_slippage).unwrap_or(0.0),
            total_market_impact: base_result.transaction_costs.as_ref().map(|tc| tc.total_market_impact).unwrap_or(0.0),
            transaction_costs_percentage: base_result.transaction_costs.as_ref().map(|tc| tc.costs_as_percentage_of_pnl).unwrap_or(0.0),
        },
        stddev_metrics: PerformanceMetrics {
            net_profit: std_net_profit,
            gross_profit: 0.0,
            gross_loss: 0.0,
            sharpe_ratio: std_sharpe,
            max_drawdown: std_dd,
            win_rate: 0.0,
            profit_factor: 0.0,
            average_trade_return: 0.0,
            median_trade_return: 0.0,
            number_of_trades: 0,
            volatility: 0.0,
            average_trade_duration: 0.0,
            total_commission: 0.0,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            transaction_costs_percentage: 0.0,
        },
        percentile_metrics: percentiles,
        var_95,
        cvar_95,
        data_points,
        var_unreliable: data_points < 30,
        fan_chart: Vec::new(), // Block bootstrap doesn't generate fan chart (uses main MC for that)
        base_source: String::new(),
        mc_rng_seed: 0,
        bb_p5_total_pnl: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // resolve_mc_runs
    // ================================================================

    #[test]
    fn resolve_mc_runs_floors_a_zero_or_unset_request_at_the_default() {
        assert_eq!(resolve_mc_runs(0), DEFAULT_MC_RUNS);
    }

    #[test]
    fn resolve_mc_runs_floors_a_too_low_explicit_request() {
        assert_eq!(resolve_mc_runs(50), DEFAULT_MC_RUNS);
    }

    #[test]
    fn resolve_mc_runs_never_lowers_an_explicit_request_above_the_default() {
        assert_eq!(resolve_mc_runs(5000), 5000);
    }

    #[test]
    fn default_mc_runs_is_at_least_an_order_of_magnitude_above_the_old_ad_hoc_floors() {
        // Regression guard for the specific regression this fixes: the old
        // 100/200 floors scattered across worker.rs call sites must not creep
        // back in as the "real" value once they're gone from that file.
        assert!(DEFAULT_MC_RUNS >= 2000);
    }

    // ================================================================
    // mean
    // ================================================================

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean(&[5.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_positive() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_mixed() {
        assert!((mean(&[-1.0, 1.0]) - 0.0).abs() < 1e-10);
    }

    // ================================================================
    // std_dev
    // ================================================================

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_single() {
        assert_eq!(std_dev(&[5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_identical() {
        assert!((std_dev(&[3.0, 3.0, 3.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev_known() {
        // [1, 2, 3] -> mean=2, variance=((1-2)^2+(2-2)^2+(3-2)^2)/2 = 2/2 = 1, std=1
        assert!((std_dev(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-10);
    }

    // ================================================================
    // percentile
    // ================================================================

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&mut [], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&mut [42.0], 50.0), 42.0);
    }

    #[test]
    fn test_percentile_median() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&mut data, 50.0), 3.0);
    }

    #[test]
    fn test_percentile_extremes() {
        let mut data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(percentile(&mut data, 0.0), 10.0);
        assert_eq!(percentile(&mut data, 100.0), 50.0);
    }

    // ================================================================
    // percentile_from_sorted
    // ================================================================

    #[test]
    fn test_percentile_from_sorted_empty() {
        assert_eq!(percentile_from_sorted(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_from_sorted_known() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_from_sorted(&sorted, 50.0), 3.0);
        assert_eq!(percentile_from_sorted(&sorted, 0.0), 1.0);
    }

    // ================================================================
    // BlockBootstrapConfig
    // ================================================================

    #[test]
    fn test_block_bootstrap_default() {
        let config = BlockBootstrapConfig::default();
        assert_eq!(config.block_size, 10);
        assert!(config.overlapping);
        assert_eq!(config.min_block_size, 3);
    }

    #[test]
    fn test_optimal_block_size_small() {
        // n=8 -> 8^(1/3) = 2.0, ceil=2, max(2,3)=3, min(3,2)=2, max(2,1)=2
        let size = BlockBootstrapConfig::optimal_block_size(8);
        assert!(size >= 1 && size <= 8);
    }

    #[test]
    fn test_optimal_block_size_medium() {
        // n=1000 -> 1000^(1/3) = 10.0, ceil=10, max(10,3)=10, min(10,250)=10
        let size = BlockBootstrapConfig::optimal_block_size(1000);
        assert_eq!(size, 10);
    }

    #[test]
    fn test_optimal_block_size_large() {
        // n=10000 -> 10000^(1/3) ≈ 21.5, ceil=22, max(22,3)=22, min(22,2500)=22
        let size = BlockBootstrapConfig::optimal_block_size(10000);
        assert_eq!(size, 22);
    }

    // ================================================================
    // RegimeAwareConfig
    // ================================================================

    #[test]
    fn test_regime_aware_default() {
        let config = RegimeAwareConfig::default();
        assert_eq!(config.volatility_lookback, 20);
        assert!((config.current_volatility - 0.02).abs() < 1e-10);
        assert!((config.regime_weight - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_regime_low() {
        let config = RegimeAwareConfig::default();
        assert_eq!(config.classify_regime(0.005), MarketRegimeClass::LowVolatility);
    }

    #[test]
    fn test_classify_regime_normal() {
        let config = RegimeAwareConfig::default();
        assert_eq!(config.classify_regime(0.015), MarketRegimeClass::NormalVolatility);
    }

    #[test]
    fn test_classify_regime_high() {
        let config = RegimeAwareConfig::default();
        assert_eq!(config.classify_regime(0.035), MarketRegimeClass::HighVolatility);
    }

    #[test]
    fn test_classify_regime_crisis() {
        let config = RegimeAwareConfig::default();
        assert_eq!(config.classify_regime(0.10), MarketRegimeClass::Crisis);
    }

    #[test]
    fn test_current_regime() {
        let config = RegimeAwareConfig::default(); // current_volatility = 0.02
        assert_eq!(config.current_regime(), MarketRegimeClass::NormalVolatility);
    }

    // ================================================================
    // MonteCarloResult struct
    // ================================================================

    #[test]
    fn test_monte_carlo_result_var_unreliable_flag() {
        let result = MonteCarloResult {
            num_runs: 100,
            mean_metrics: PerformanceMetrics::default(),
            stddev_metrics: PerformanceMetrics::default(),
            percentile_metrics: HashMap::new(),
            var_95: -500.0,
            cvar_95: -600.0,
            data_points: 10,
            var_unreliable: true,
            fan_chart: Vec::new(),
            base_source: String::new(),
            mc_rng_seed: 0,
            bb_p5_total_pnl: None,
        };
        assert!(result.var_unreliable);
        assert_eq!(result.data_points, 10);
    }

    // ================================================================
    // Bug #32 — fractional returns are compounded (not summed) so stitched
    // walk-forward windows can't drive simulated equity negative.
    // ================================================================

    /// Build a base result mimicking a stitched walk-forward OOS aggregate:
    /// each window independently started from $10K, so the per-trade DOLLAR
    /// P&Ls summed additively would blow the account past zero, while the
    /// per-trade FRACTIONAL returns compounded stay bounded above zero.
    fn stitched_oos_base() -> BacktestResult {
        let initial_capital = 10_000.0;
        let mut trade_pct_returns = Vec::new();
        // 30 trades of -4% and 10 of -1% (net losing, but each > -100%).
        for _ in 0..30 { trade_pct_returns.push(-0.04); }
        for _ in 0..10 { trade_pct_returns.push(-0.01); }
        // Dollar P&Ls as if each trade earned its pct on a fresh $10K base.
        // Sum = 30*(-400) + 10*(-100) = -$13,000 → additive equity = -$3,000.
        let trade_returns: Vec<f64> =
            trade_pct_returns.iter().map(|p| p * initial_capital).collect();
        BacktestResult {
            initial_capital,
            num_trades: trade_returns.len(),
            trade_returns,
            trade_pct_returns,
            ..Default::default()
        }
    }

    #[test]
    fn test_mc_compounds_fractional_returns_stays_bounded() {
        let base = stitched_oos_base();
        let result = run_monte_carlo_simulation(&base, 400).expect("MC should run");

        assert_eq!(result.data_points, 40);
        assert!(!result.var_unreliable, "40 trades should be reliable");

        // Every percentile equity must stay above zero (net_profit > -capital)
        // and drawdown must never exceed 100% — both impossible if dollar P&Ls
        // were summed additively across the 8 stacked windows.
        for &p in &[5u8, 25, 50, 75, 95] {
            let m = &result.percentile_metrics[&p];
            assert!(
                m.net_profit > -base.initial_capital,
                "p{} net_profit {} drove equity below zero",
                p, m.net_profit
            );
            assert!(
                m.max_drawdown <= 1.0,
                "p{} max_drawdown {} exceeded 100%",
                p, m.max_drawdown
            );
        }

        // No point on any fan-chart band may be negative equity.
        for band in &result.fan_chart {
            for &equity in band {
                assert!(equity > 0.0, "fan chart produced negative equity {}", equity);
            }
        }

        // Fan chart starts at initial capital and ends bounded above zero.
        assert!((result.fan_chart[0][2] - base.initial_capital).abs() < 1e-6);
        let last = result.fan_chart.last().unwrap();
        assert!(last[2] > 0.0 && last[2] < base.initial_capital,
            "median ending equity {} should be a bounded loss", last[2]);
    }

    #[test]
    fn test_mc_falls_back_to_dollar_when_no_pct() {
        // Built-in strategies don't populate trade_pct_returns; the additive
        // dollar path must still work (no panic, sensible data_points).
        let mut base = stitched_oos_base();
        base.trade_pct_returns.clear(); // force dollar fallback
        let result = run_monte_carlo_simulation(&base, 200).expect("MC should run");
        assert_eq!(result.data_points, 40);
        assert!(!result.var_unreliable);
    }

    #[test]
    fn test_mc_uses_account_scale_not_position_scale() {
        // Bug #33 regression. 40 identical trades, each a +$6 net P&L on a ~$600
        // position: that is +1% on the POSITION but only +0.06% on the $10k
        // ACCOUNT. The realized account outcome is +$240 (equity += net_pnl).
        // The MC must reconstruct on the ACCOUNT scale — it must NOT compound the
        // +1% position return (`trade_pct_returns`) against the whole account,
        // which would balloon to ~+$4,900 (the old inflated-bands bug).
        let initial_capital = 10_000.0;
        let n = 40usize;
        let trade_returns = vec![6.0_f64; n]; // dollar P&L per closed trade
        let trade_pct_returns = vec![0.01_f64; n]; // position-level return (must be IGNORED)
        let base = BacktestResult {
            initial_capital,
            num_trades: n,
            trade_returns,
            trade_pct_returns,
            ..Default::default()
        };
        let result = run_monte_carlo_simulation(&base, 200).expect("MC should run");
        assert_eq!(result.data_points, n);

        // Account-scale compounding: (1 + 6/10000)^40 - 1 ≈ +$243.
        // Position-scale (buggy) would be (1.01)^40 - 1 ≈ +$4,900.
        let median_net = result.percentile_metrics[&50].net_profit;
        assert!(
            median_net > 150.0 && median_net < 400.0,
            "median net_profit {} is not on the account scale (~$243) — \
             position-level returns leaked into compounding",
            median_net
        );

        // Fan-chart median ending equity must agree with the median net profit.
        let last_median = result.fan_chart.last().unwrap()[2];
        assert!(
            (last_median - (initial_capital + median_net)).abs() < 1.0,
            "fan-chart median {} disagrees with net profit {}",
            last_median, median_net
        );
    }
}