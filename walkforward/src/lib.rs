//! Walk-forward analysis module

use std::collections::HashMap;
use std::future::Future;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

pub mod window;
pub mod runner;
pub mod result;
pub mod auto_export;
pub mod logging_facade;

// Re-export key functions for external use
pub use runner::run_walkforward_for_params;
// Note: run_walkforward_realistic is implemented directly in ga_worker to avoid circular dependency
pub use result::{WalkForwardSummary, WalkForwardWindowResult};

// Trading performance metrics for walkforward analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_trade_duration_hours: f64,
    pub volatility: f64,
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 1.0,
            total_trades: 0,
            avg_trade_duration_hours: 0.0,
            volatility: 0.0,
        }
    }
}

impl TradingMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct Window {
    pub id: u32,
    pub optimization_window: OptimizationWindow,
    pub test_window: TestWindow,
}

// Main types needed by integration tests
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub optimization_window_size: std::time::Duration,
    pub test_window_size: std::time::Duration,
    pub max_parameters: usize,
    pub optimization_period_days: i64,
    pub test_period_days: i64,
    pub step_size_days: i64,
    pub min_trades_per_period: usize,
    pub parameters_to_optimize: Vec<String>,
    pub optimization_metric: String,
    /// Embargo gap between training end and test start (days, default 0)
    pub embargo_days: i64,
    /// Anchored mode: training window always starts from the beginning of data
    pub anchored: bool,
}

impl WalkForwardConfig {
    pub fn new(
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        optimization_window_size: std::time::Duration,
        test_window_size: std::time::Duration,
        max_parameters: usize,
    ) -> Self {
        let optimization_period_days = optimization_window_size.as_secs() / (24 * 3600);
        let test_period_days = test_window_size.as_secs() / (24 * 3600);
        
        Self {
            start_date,
            end_date,
            optimization_window_size,
            test_window_size,
            max_parameters,
            optimization_period_days: optimization_period_days as i64,
            test_period_days: test_period_days as i64,
            step_size_days: 21, // Default to 1 month steps
            min_trades_per_period: 10,
            parameters_to_optimize: vec!["ma_period".to_string()],
            optimization_metric: "sharpe_ratio".to_string(),
            embargo_days: 0,
            anchored: false,
        }
    }
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        use chrono::TimeZone;
        let start_date = chrono::Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end_date = chrono::Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap();
        
        Self {
            start_date,
            end_date,
            optimization_window_size: std::time::Duration::from_secs(252 * 24 * 3600),
            test_window_size: std::time::Duration::from_secs(63 * 24 * 3600),
            max_parameters: 5,
            optimization_period_days: 252, // 1 year
            test_period_days: 63, // 3 months
            step_size_days: 21, // 1 month
            min_trades_per_period: 10,
            parameters_to_optimize: vec!["ma_period".to_string()],
            optimization_metric: "sharpe_ratio".to_string(),
            embargo_days: 0,
            anchored: false,
        }
    }
}

impl WalkForwardConfig {
    pub fn is_valid(&self) -> bool {
        self.optimization_period_days > 30 &&
        self.test_period_days > 0 &&
        self.step_size_days > 0 &&
        self.min_trades_per_period > 0 &&
        !self.parameters_to_optimize.is_empty() &&
        !self.optimization_metric.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationWindow {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestWindow {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub window_id: u32,
    pub optimization_window: OptimizationWindow,
    pub test_window: TestWindow,
    pub optimal_parameters: HashMap<String, f64>,
    pub optimization_score: f64,
    pub out_of_sample_performance: TradingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    pub window_results: Vec<WindowResult>,
    pub overall_metrics: TradingMetrics,
    pub parameter_stability_score: f64,
    pub total_windows: usize,
    pub successful_windows: usize,
    pub avg_optimization_time_ms: u64,
}

// ============================================
// CPCV (Combinatorial Purged Cross-Validation)
// ============================================

/// Configuration for CPCV analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVConfig {
    /// Number of folds to create
    pub n_folds: usize,
    /// Number of test folds per combination (typically 2)
    pub n_test_folds: usize,
    /// Purge gap in days (embargo period to prevent lookahead)
    pub purge_gap_days: i64,
    /// Minimum samples per fold
    pub min_samples_per_fold: usize,
}

impl Default for CPCVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_test_folds: 2,
            purge_gap_days: 5,
            min_samples_per_fold: 50,
        }
    }
}

impl CPCVConfig {
    /// Calculate total number of combinations: C(n_folds, n_test_folds)
    pub fn total_combinations(&self) -> usize {
        let n = self.n_folds;
        let k = self.n_test_folds;
        if k > n { return 0; }
        
        // C(n, k) = n! / (k! * (n-k)!)
        let mut result = 1usize;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

/// A single CPCV fold with purging applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVFold {
    pub fold_id: usize,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_indices: Vec<usize>,
}

/// Result from a single CPCV combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVCombinationResult {
    pub combination_id: usize,
    pub test_fold_ids: Vec<usize>,
    pub train_fold_ids: Vec<usize>,
    pub optimal_parameters: HashMap<String, f64>,
    pub test_metrics: TradingMetrics,
    pub overfit_score: f64,
}

/// Comprehensive CPCV analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVResult {
    pub combination_results: Vec<CPCVCombinationResult>,
    pub overall_metrics: TradingMetrics,
    pub probability_of_backtest_overfit: f64, // PBO
    pub deflated_sharpe_ratio: f64,
    pub parameter_stability_score: f64,
    pub total_combinations: usize,
    pub avg_test_sharpe: f64,
    pub sharpe_variance_across_folds: f64,
    /// Fewest trades seen in any single test fold. Each fold's Sharpe (and
    /// therefore `sharpe_variance_across_folds`/`deflated_sharpe_ratio`) is
    /// only a meaningful estimate when there are enough trades in that fold
    /// to say anything -- a strategy with, say, 18 total trades split across
    /// 5 CPCV folds can leave individual test folds with 2-3 trades, whose
    /// wildly unstable per-fold Sharpe blows up the cross-fold variance and
    /// produces a `deflated_sharpe_ratio` many orders of magnitude away from
    /// zero or one (observed in production: a strategy with a genuinely
    /// decent overall Sharpe of 0.97 got a CPCV-derived DSR of ~1e-118).
    /// Callers should treat `deflated_sharpe_ratio` as unreliable and prefer
    /// a different DSR source when this is below a reasonable floor (see
    /// `MIN_RELIABLE_TRADES_PER_FOLD`).
    pub min_trades_per_fold: usize,
}

/// Minimum trades a CPCV test fold needs before its Sharpe estimate (and
/// therefore the cross-fold `deflated_sharpe_ratio`) is trusted. Smaller than
/// `MC_MIN_TRADES` (30, used for whole-backtest Monte Carlo) since a fold is
/// inherently a slice of the full trade set -- but still enough that one or
/// two outlier trades can't single-handedly swing a fold's Sharpe to an
/// extreme.
pub const MIN_RELIABLE_TRADES_PER_FOLD: usize = 10;

/// Fewest trades in any single fold's test window. `0` for an empty slice
/// (no folds ran at all -- definitely not reliable).
fn min_trades_across_folds(combination_results: &[CPCVCombinationResult]) -> usize {
    combination_results.iter()
        .map(|c| c.test_metrics.total_trades)
        .min()
        .unwrap_or(0)
}

/// Result of evaluating a single CPCV fold (returned by the caller's callback).
#[derive(Debug, Clone)]
pub struct CpcvFoldEval {
    /// Best parameters found on the training set.
    pub optimal_parameters: HashMap<String, f64>,
    /// Sharpe ratio measured on the in-sample (training) set.
    pub in_sample_sharpe: f64,
    /// Full trading metrics measured on the out-of-sample (test) set.
    pub test_metrics: TradingMetrics,
}

pub struct WalkForwardAnalyzer {
    pub config: WalkForwardConfig,
    backtest_config: config::BacktestConfig,
}

impl WalkForwardAnalyzer {
    pub fn new(config: WalkForwardConfig, backtest_config: config::BacktestConfig) -> Self {
        Self {
            config,
            backtest_config,
        }
    }
    
    pub fn get_optimization_period_days(&self) -> i64 {
        self.config.optimization_period_days
    }
    
    pub fn get_config(&self) -> &WalkForwardConfig {
        &self.config
    }
    
    pub async fn run_analysis(&self) -> anyhow::Result<WalkForwardResult> {
        // Placeholder implementation
        Ok(WalkForwardResult {
            window_results: Vec::new(),
            overall_metrics: TradingMetrics::default(),
            parameter_stability_score: 0.0,
            total_windows: 0,
            successful_windows: 0,
            avg_optimization_time_ms: 0,
        })
    }
    
    pub fn validate_config(&self) -> anyhow::Result<()> {
        if !self.config.is_valid() {
            return Err(anyhow::anyhow!("Invalid walk forward configuration"));
        }
        Ok(())
    }
    
    pub fn create_windows(&self, start_date: DateTime<Utc>, end_date: DateTime<Utc>) -> Vec<Window> {
        // Create a temporary config with the provided dates
        let temp_analyzer = WalkForwardAnalyzer {
            config: WalkForwardConfig {
                start_date,
                end_date,
                ..self.config.clone()
            },
            backtest_config: self.backtest_config.clone(),
        };
        temp_analyzer.generate_windows()
    }
    
    pub fn generate_windows(&self) -> Vec<Window> {
        let start_date = self.config.start_date;
        let end_date = self.config.end_date;
        let mut windows = Vec::new();
        let mut window_id = 1;
        let mut current_opt_start = start_date;
        
        loop {
            let opt_end = current_opt_start + chrono::Duration::days(self.config.optimization_period_days);
            let test_start = opt_end + chrono::Duration::days(self.config.embargo_days);
            let test_end = test_start + chrono::Duration::days(self.config.test_period_days);
            
            if test_end > end_date {
                break;
            }
            
            windows.push(Window {
                id: window_id,
                optimization_window: OptimizationWindow {
                    start_date: current_opt_start,
                    end_date: opt_end,
                },
                test_window: TestWindow {
                    start_date: test_start,
                    end_date: test_end,
                },
            });
            
            window_id += 1;
            current_opt_start = current_opt_start + chrono::Duration::days(self.config.step_size_days);
        }
        
        windows
    }

    pub async fn optimize_parameters(
        &self,
        parameter_ranges: &HashMap<String, (f64, f64)>,
        _optimization_window: &OptimizationWindow
    ) -> anyhow::Result<HashMap<String, f64>> {
        // Mock optimization - return middle values from ranges
        let mut optimal_params = HashMap::new();
        
        for (param_name, (min_val, max_val)) in parameter_ranges {
            let optimal_value = (min_val + max_val) / 2.0;
            optimal_params.insert(param_name.clone(), optimal_value);
        }
        
        // Add some constraint logic for MA parameters
        if let (Some(short_ma), Some(long_ma)) = (optimal_params.get("short_ma"), optimal_params.get("long_ma")) {
            if short_ma >= long_ma {
                optimal_params.insert("short_ma".to_string(), long_ma - 5.0);
            }
        }
        
        Ok(optimal_params)
    }
    
    pub async fn run_backtest_with_parameters(
        &self,
        _parameters: &HashMap<String, f64>,
        _test_window: &TestWindow
    ) -> anyhow::Result<TradingMetrics> {
        // Mock backtest results
        Ok(TradingMetrics {
            total_return: 8.5,
            sharpe_ratio: 1.4,
            max_drawdown: 6.2,
            win_rate: 0.62,
            profit_factor: 1.7,
            total_trades: 18,
            avg_trade_duration_hours: 28.0,
            volatility: 0.16,
        })
    }
    
    pub async fn run_walk_forward_analysis(
        &self,
        parameter_ranges: HashMap<String, (f64, f64)>,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>
    ) -> anyhow::Result<WalkForwardResult> {
        let windows = self.create_windows(start_date, end_date);
        let mut window_results = Vec::new();
        let start_time = std::time::Instant::now();
        
        for window in windows {
            let optimal_params = self.optimize_parameters(&parameter_ranges, &window.optimization_window).await?;
            let out_of_sample_performance = self.run_backtest_with_parameters(&optimal_params, &window.test_window).await?;
            
            let window_result = WindowResult {
                window_id: window.id,
                optimization_window: window.optimization_window,
                test_window: window.test_window,
                optimal_parameters: optimal_params,
                optimization_score: out_of_sample_performance.sharpe_ratio,
                out_of_sample_performance,
            };
            
            window_results.push(window_result);
        }
        
        let overall_metrics = Self::aggregate_metrics(&window_results);
        let parameter_stability_score = Self::calculate_parameter_stability(&window_results);
        let total_time = start_time.elapsed().as_millis();
        
        Ok(WalkForwardResult {
            window_results: window_results.clone(),
            overall_metrics,
            parameter_stability_score,
            total_windows: window_results.len(),
            successful_windows: window_results.len(), // All successful in mock
            avg_optimization_time_ms: (total_time / window_results.len() as u128) as u64,
        })
    }
    
    pub fn aggregate_metrics(window_results: &[WindowResult]) -> TradingMetrics {
        if window_results.is_empty() {
            return TradingMetrics::default();
        }
        
        let total_return = window_results.iter().map(|w| w.out_of_sample_performance.total_return).sum::<f64>() / window_results.len() as f64;
        let sharpe_ratio = window_results.iter().map(|w| w.out_of_sample_performance.sharpe_ratio).sum::<f64>() / window_results.len() as f64;
        let max_drawdown = window_results.iter().map(|w| w.out_of_sample_performance.max_drawdown).fold(0.0, f64::max);
        let win_rate = window_results.iter().map(|w| w.out_of_sample_performance.win_rate).sum::<f64>() / window_results.len() as f64;
        let profit_factor = window_results.iter().map(|w| w.out_of_sample_performance.profit_factor).sum::<f64>() / window_results.len() as f64;
        let total_trades = window_results.iter().map(|w| w.out_of_sample_performance.total_trades).sum::<usize>();
        let avg_trade_duration = window_results.iter().map(|w| w.out_of_sample_performance.avg_trade_duration_hours).sum::<f64>() / window_results.len() as f64;
        let volatility = window_results.iter().map(|w| w.out_of_sample_performance.volatility).sum::<f64>() / window_results.len() as f64;
        
        TradingMetrics {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_duration_hours: avg_trade_duration,
            volatility,
        }
    }
    
    pub fn calculate_parameter_stability(window_results: &[WindowResult]) -> f64 {
        if window_results.len() < 2 {
            return 1.0;
        }
        
        let mut total_stability = 0.0;
        let mut param_count = 0;
        
        // Get all parameter names
        let all_param_names: std::collections::HashSet<String> = window_results.iter()
            .flat_map(|w| w.optimal_parameters.keys().cloned())
            .collect();
        
        for param_name in all_param_names {
            let values: Vec<f64> = window_results.iter()
                .filter_map(|w| w.optimal_parameters.get(&param_name))
                .copied()
                .collect();
            
            if values.len() >= 2 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let coefficient_of_variation = if mean != 0.0 { variance.sqrt() / mean.abs() } else { 0.0 };
                
                // Higher stability for lower coefficient of variation
                let param_stability = 1.0 / (1.0 + coefficient_of_variation);
                total_stability += param_stability;
                param_count += 1;
            }
        }
        
        if param_count > 0 {
            total_stability / param_count as f64
        } else {
            1.0
        }
    }
    
    // ============================================
    // CPCV (Combinatorial Purged Cross-Validation)
    // ============================================
    
    /// Generate all k-combinations of fold indices for test sets
    fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut combination = vec![0; k];
        
        Self::generate_combinations_recursive(&mut result, &mut combination, 0, 0, n, k);
        result
    }
    
    fn generate_combinations_recursive(
        result: &mut Vec<Vec<usize>>,
        combination: &mut Vec<usize>,
        start: usize,
        index: usize,
        n: usize,
        k: usize,
    ) {
        if index == k {
            result.push(combination.clone());
            return;
        }
        
        for i in start..=(n - k + index) {
            combination[index] = i;
            Self::generate_combinations_recursive(result, combination, i + 1, index + 1, n, k);
        }
    }
    
    /// Create CPCV folds with purging applied
    pub fn create_cpcv_folds(
        &self,
        data_length: usize,
        config: &CPCVConfig,
    ) -> Vec<CPCVFold> {
        let fold_size = data_length / config.n_folds;
        let purge_samples = (config.purge_gap_days as usize).min(fold_size / 4);
        
        let mut folds = Vec::new();
        
        for fold_id in 0..config.n_folds {
            let fold_start = fold_id * fold_size;
            let fold_end = if fold_id == config.n_folds - 1 {
                data_length
            } else {
                (fold_id + 1) * fold_size
            };
            
            // Calculate purge zone (overlap between train and test)
            let purge_start = if fold_start >= purge_samples {
                fold_start - purge_samples
            } else {
                0
            };
            let purge_end = (fold_end + purge_samples).min(data_length);
            
            let test_indices: Vec<usize> = (fold_start..fold_end).collect();
            let purged_indices: Vec<usize> = (purge_start..fold_start)
                .chain(fold_end..purge_end)
                .collect();
            
            // Train indices are everything outside test and purge
            let train_indices: Vec<usize> = (0..data_length)
                .filter(|i| !test_indices.contains(i) && !purged_indices.contains(i))
                .collect();
            
            folds.push(CPCVFold {
                fold_id,
                train_indices,
                test_indices,
                purged_indices,
            });
        }
        
        folds
    }
    
    /// Run full CPCV analysis.
    ///
    /// `eval_fn` is called once per combination with `(train_indices, test_indices)`.
    /// It must optimise on the training set, evaluate on the test set, and return a
    /// [`CpcvFoldEval`] containing IS sharpe, OOS metrics, and the chosen parameters.
    pub async fn run_cpcv_analysis<F, Fut>(
        &self,
        data_length: usize,
        cpcv_config: CPCVConfig,
        eval_fn: F,
    ) -> anyhow::Result<CPCVResult>
    where
        F: Fn(Vec<usize>, Vec<usize>) -> Fut + Send + Sync,
        Fut: Future<Output = anyhow::Result<CpcvFoldEval>> + Send,
    {
        let folds = self.create_cpcv_folds(data_length, &cpcv_config);
        let combinations = Self::generate_combinations(cpcv_config.n_folds, cpcv_config.n_test_folds);
        
        let mut combination_results = Vec::new();
        let mut in_sample_sharpes = Vec::new();
        let mut out_of_sample_sharpes = Vec::new();
        
        for (combo_id, test_fold_ids) in combinations.iter().enumerate() {
            // Train folds are all folds not in test
            let train_fold_ids: Vec<usize> = (0..cpcv_config.n_folds)
                .filter(|f| !test_fold_ids.contains(f))
                .collect();
            
            // Aggregate train indices (excluding purged zones around test folds)
            let test_indices: std::collections::HashSet<usize> = test_fold_ids.iter()
                .flat_map(|&f| folds[f].test_indices.iter().copied())
                .collect();
            let purged_indices: std::collections::HashSet<usize> = test_fold_ids.iter()
                .flat_map(|&f| folds[f].purged_indices.iter().copied())
                .collect();
            
            let train_indices: Vec<usize> = (0..data_length)
                .filter(|i| !test_indices.contains(i) && !purged_indices.contains(i))
                .collect();
            let test_indices_vec: Vec<usize> = test_indices.into_iter().collect();
            
            // Evaluate via callback
            let fold_eval = eval_fn(train_indices, test_indices_vec).await?;
            
            in_sample_sharpes.push(fold_eval.in_sample_sharpe);
            out_of_sample_sharpes.push(fold_eval.test_metrics.sharpe_ratio);
            
            // Overfit score: how much IS beats OOS
            let overfit_score = if fold_eval.in_sample_sharpe.abs() > 1e-10 {
                (fold_eval.in_sample_sharpe - fold_eval.test_metrics.sharpe_ratio) / fold_eval.in_sample_sharpe
            } else {
                0.0
            };
            
            combination_results.push(CPCVCombinationResult {
                combination_id: combo_id,
                test_fold_ids: test_fold_ids.clone(),
                train_fold_ids,
                optimal_parameters: fold_eval.optimal_parameters,
                test_metrics: fold_eval.test_metrics,
                overfit_score,
            });
        }
        
        // Calculate Probability of Backtest Overfit (PBO)
        let pbo = self.calculate_pbo(&in_sample_sharpes, &out_of_sample_sharpes);
        
        // Calculate deflated Sharpe ratio using significance module
        let avg_oos_sharpe = out_of_sample_sharpes.iter().sum::<f64>() / out_of_sample_sharpes.len().max(1) as f64;
        let sharpe_variance = Self::calculate_variance(&out_of_sample_sharpes);
        let n_trials = combination_results.len();
        let n_days = data_length;
        let sharpe_std = sharpe_variance.sqrt().max(1e-10);
        // CPCV operates generically over bar indices with no candle-interval
        // or trade-timestamp data available at this layer (unlike the
        // single-asset/pairs backtest paths, which now derive this from real
        // trade span -- see `metrics::significance::observations_per_year_from_span`).
        // 365.0 preserves this path's existing behavior exactly rather than
        // guessing at a derivation this generic algorithm can't support today.
        let deflated_sharpe = metrics::significance::deflated_sharpe_ratio(
            avg_oos_sharpe,
            n_trials,
            n_days,
            sharpe_std,
            365.0,
        );
        
        // Aggregate overall metrics
        let overall_metrics = Self::aggregate_cpcv_metrics(&combination_results);
        
        // Parameter stability across combinations
        let param_stability = Self::calculate_cpcv_parameter_stability(&combination_results);
        
        let min_trades_per_fold = min_trades_across_folds(&combination_results);

        Ok(CPCVResult {
            combination_results: combination_results.clone(),
            overall_metrics,
            probability_of_backtest_overfit: pbo,
            deflated_sharpe_ratio: deflated_sharpe,
            parameter_stability_score: param_stability,
            total_combinations: combination_results.len(),
            avg_test_sharpe: avg_oos_sharpe,
            sharpe_variance_across_folds: sharpe_variance,
            min_trades_per_fold,
        })
    }
    
    /// Calculate Probability of Backtest Overfit (PBO)
    /// Based on Bailey, Borwein, López de Prado (2016)
    fn calculate_pbo(&self, in_sample: &[f64], out_of_sample: &[f64]) -> f64 {
        if in_sample.len() != out_of_sample.len() || in_sample.is_empty() {
            return 0.5; // No information
        }
        
        let n = in_sample.len();
        
        // Rank both IS and OOS Sharpe ratios
        let mut is_ranked: Vec<(usize, f64)> = in_sample.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        is_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let median_rank = n / 2;
        
        // Count combinations where best IS performer has below-median OOS
        let mut overfit_count = 0;
        for (rank, (combo_idx, _is_sharpe)) in is_ranked.iter().enumerate() {
            if rank < median_rank {
                // This combo was top-half by IS
                let oos_sharpe = out_of_sample[*combo_idx];
                let oos_median = Self::calculate_median(out_of_sample);
                
                if oos_sharpe < oos_median {
                    overfit_count += 1;
                }
            }
        }
        
        // PBO = proportion of top IS that underperformed OOS median
        overfit_count as f64 / median_rank.max(1) as f64
    }
    
    fn calculate_median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
    
    fn calculate_variance(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }
    
    fn aggregate_cpcv_metrics(results: &[CPCVCombinationResult]) -> TradingMetrics {
        if results.is_empty() {
            return TradingMetrics::default();
        }
        
        let n = results.len() as f64;
        TradingMetrics {
            total_return: results.iter().map(|r| r.test_metrics.total_return).sum::<f64>() / n,
            sharpe_ratio: results.iter().map(|r| r.test_metrics.sharpe_ratio).sum::<f64>() / n,
            max_drawdown: results.iter().map(|r| r.test_metrics.max_drawdown).fold(0.0, f64::max),
            win_rate: results.iter().map(|r| r.test_metrics.win_rate).sum::<f64>() / n,
            profit_factor: results.iter().map(|r| r.test_metrics.profit_factor).sum::<f64>() / n,
            total_trades: results.iter().map(|r| r.test_metrics.total_trades).sum::<usize>() / results.len(),
            avg_trade_duration_hours: results.iter().map(|r| r.test_metrics.avg_trade_duration_hours).sum::<f64>() / n,
            volatility: results.iter().map(|r| r.test_metrics.volatility).sum::<f64>() / n,
        }
    }
    
    fn calculate_cpcv_parameter_stability(results: &[CPCVCombinationResult]) -> f64 {
        if results.len() < 2 {
            return 1.0;
        }
        
        let all_param_names: std::collections::HashSet<String> = results.iter()
            .flat_map(|r| r.optimal_parameters.keys().cloned())
            .collect();
        
        let mut total_stability = 0.0;
        let mut param_count = 0;
        
        for param_name in all_param_names {
            let values: Vec<f64> = results.iter()
                .filter_map(|r| r.optimal_parameters.get(&param_name))
                .copied()
                .collect();
            
            if values.len() >= 2 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let cv = if mean.abs() > 1e-10 {
                    Self::calculate_variance(&values).sqrt() / mean.abs()
                } else {
                    0.0
                };
                total_stability += 1.0 / (1.0 + cv);
                param_count += 1;
            }
        }
        
        if param_count > 0 {
            total_stability / param_count as f64
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    // ── min_trades_across_folds ──

    fn mk_combo(total_trades: usize) -> CPCVCombinationResult {
        CPCVCombinationResult {
            combination_id: 0,
            test_fold_ids: vec![0],
            train_fold_ids: vec![1, 2, 3, 4],
            optimal_parameters: HashMap::new(),
            test_metrics: TradingMetrics { total_trades, ..TradingMetrics::default() },
            overfit_score: 0.0,
        }
    }

    #[test]
    fn min_trades_across_folds_empty_is_zero() {
        assert_eq!(min_trades_across_folds(&[]), 0);
    }

    #[test]
    fn min_trades_across_folds_finds_the_thinnest_fold() {
        let folds = vec![mk_combo(40), mk_combo(3), mk_combo(25)];
        assert_eq!(min_trades_across_folds(&folds), 3);
    }

    #[test]
    fn min_trades_across_folds_below_reliability_floor_is_detectable() {
        // Regression guard for the production incident this backs: a
        // strategy with 18 total trades spread across 5 folds left at least
        // one fold with only 2 trades, whose Sharpe is not a meaningful
        // estimate -- callers must be able to tell this from a healthy case.
        let thin_folds = vec![mk_combo(4), mk_combo(2), mk_combo(5)];
        assert!(min_trades_across_folds(&thin_folds) < MIN_RELIABLE_TRADES_PER_FOLD);

        let healthy_folds = vec![mk_combo(40), mk_combo(35), mk_combo(50)];
        assert!(min_trades_across_folds(&healthy_folds) >= MIN_RELIABLE_TRADES_PER_FOLD);
    }

    // ── TradingMetrics ──

    #[test]
    fn trading_metrics_default() {
        let m = TradingMetrics::default();
        assert_eq!(m.total_return, 0.0);
        assert_eq!(m.total_trades, 0);
        assert_eq!(m.profit_factor, 1.0);
    }

    #[test]
    fn trading_metrics_new_equals_default() {
        let a = TradingMetrics::new();
        let b = TradingMetrics::default();
        assert_eq!(a.total_return, b.total_return);
    }

    #[test]
    fn trading_metrics_serde_roundtrip() {
        let m = TradingMetrics {
            total_return: 12.5,
            sharpe_ratio: 1.8,
            max_drawdown: 5.0,
            win_rate: 0.65,
            profit_factor: 2.1,
            total_trades: 42,
            avg_trade_duration_hours: 4.5,
            volatility: 0.18,
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: TradingMetrics = serde_json::from_str(&json).unwrap();
        assert!((m2.sharpe_ratio - 1.8).abs() < 1e-10);
    }

    // ── WalkForwardConfig ──

    #[test]
    fn walkforward_config_default_is_valid() {
        let cfg = WalkForwardConfig::default();
        assert!(cfg.is_valid());
        assert_eq!(cfg.optimization_period_days, 252);
        assert_eq!(cfg.test_period_days, 63);
        assert!(!cfg.anchored);
    }

    #[test]
    fn walkforward_config_invalid_short_optimization() {
        let mut cfg = WalkForwardConfig::default();
        cfg.optimization_period_days = 10; // < 30
        assert!(!cfg.is_valid());
    }

    #[test]
    fn walkforward_config_invalid_empty_params() {
        let mut cfg = WalkForwardConfig::default();
        cfg.parameters_to_optimize.clear();
        assert!(!cfg.is_valid());
    }

    // ── CPCVConfig ──

    #[test]
    fn cpcv_config_default() {
        let cfg = CPCVConfig::default();
        assert_eq!(cfg.n_folds, 5);
        assert_eq!(cfg.n_test_folds, 2);
    }

    #[test]
    fn cpcv_total_combinations() {
        let cfg = CPCVConfig { n_folds: 5, n_test_folds: 2, purge_gap_days: 5, min_samples_per_fold: 50 };
        assert_eq!(cfg.total_combinations(), 10); // C(5,2) = 10
    }

    #[test]
    fn cpcv_total_combinations_k_greater_than_n() {
        let cfg = CPCVConfig { n_folds: 2, n_test_folds: 5, purge_gap_days: 5, min_samples_per_fold: 50 };
        assert_eq!(cfg.total_combinations(), 0);
    }

    // ── WalkForwardAnalyzer static methods ──

    #[test]
    fn aggregate_metrics_empty() {
        let agg = WalkForwardAnalyzer::aggregate_metrics(&[]);
        assert_eq!(agg.total_trades, 0);
        assert_eq!(agg.total_return, 0.0);
    }

    #[test]
    fn aggregate_metrics_single_window() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
        let results = vec![WindowResult {
            window_id: 1,
            optimization_window: OptimizationWindow { start_date: start, end_date: end },
            test_window: TestWindow { start_date: end, end_date: end + chrono::Duration::days(30) },
            optimal_parameters: HashMap::new(),
            optimization_score: 1.5,
            out_of_sample_performance: TradingMetrics {
                total_return: 10.0, sharpe_ratio: 1.5, max_drawdown: 4.0,
                win_rate: 0.6, profit_factor: 2.0, total_trades: 20,
                avg_trade_duration_hours: 5.0, volatility: 0.15,
            },
        }];
        let agg = WalkForwardAnalyzer::aggregate_metrics(&results);
        assert!((agg.total_return - 10.0).abs() < 1e-10);
        assert_eq!(agg.total_trades, 20);
    }

    #[test]
    fn parameter_stability_single_window_is_1() {
        assert_eq!(WalkForwardAnalyzer::calculate_parameter_stability(&[]), 1.0);
    }

    #[test]
    fn parameter_stability_stable_params() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
        let make_result = |id, ma_val: f64| WindowResult {
            window_id: id,
            optimization_window: OptimizationWindow { start_date: start, end_date: end },
            test_window: TestWindow { start_date: end, end_date: end + chrono::Duration::days(30) },
            optimal_parameters: {
                let mut m = HashMap::new();
                m.insert("ma_period".to_string(), ma_val);
                m
            },
            optimization_score: 1.0,
            out_of_sample_performance: TradingMetrics::default(),
        };
        let results = vec![make_result(1, 20.0), make_result(2, 21.0), make_result(3, 20.5)];
        let stability = WalkForwardAnalyzer::calculate_parameter_stability(&results);
        assert!(stability > 0.8, "stable params should have high stability, got {stability}");
    }

    // ── generate_combinations ──

    #[test]
    fn generate_combinations_c5_2() {
        let combos = WalkForwardAnalyzer::generate_combinations(5, 2);
        assert_eq!(combos.len(), 10);
    }

    #[test]
    fn generate_combinations_c4_1() {
        let combos = WalkForwardAnalyzer::generate_combinations(4, 1);
        assert_eq!(combos.len(), 4);
    }

    // ── calculate_median / calculate_variance ──

    #[test]
    fn calculate_median_odd() {
        assert!((WalkForwardAnalyzer::calculate_median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn calculate_median_even() {
        assert!((WalkForwardAnalyzer::calculate_median(&[4.0, 1.0, 2.0, 3.0]) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn calculate_median_empty() {
        assert_eq!(WalkForwardAnalyzer::calculate_median(&[]), 0.0);
    }

    #[test]
    fn calculate_variance_basic() {
        // Values: [2, 4, 6] mean=4, sample variance = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 2 = 4.0
        let v = WalkForwardAnalyzer::calculate_variance(&[2.0, 4.0, 6.0]);
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn calculate_variance_single() {
        assert_eq!(WalkForwardAnalyzer::calculate_variance(&[5.0]), 0.0);
    }

    // ── generate_windows via WalkForwardAnalyzer ──

    #[test]
    fn generate_windows_produces_sequential_windows() {
        let start = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let cfg = WalkForwardConfig {
            start_date: start,
            end_date: end,
            optimization_window_size: std::time::Duration::from_secs(180 * 86400),
            test_window_size: std::time::Duration::from_secs(60 * 86400),
            max_parameters: 5,
            optimization_period_days: 180,
            test_period_days: 60,
            step_size_days: 30,
            min_trades_per_period: 10,
            parameters_to_optimize: vec!["p".to_string()],
            optimization_metric: "sharpe".to_string(),
            embargo_days: 0,
            anchored: false,
        };
        let analyzer = WalkForwardAnalyzer::new(cfg, config::BacktestConfig::default());
        let windows = analyzer.generate_windows();
        assert!(!windows.is_empty());
        for w in &windows {
            assert!(w.optimization_window.start_date < w.optimization_window.end_date);
            assert!(w.test_window.start_date < w.test_window.end_date);
        }
    }

    // ── create_cpcv_folds ──

    #[test]
    fn create_cpcv_folds_correct_count() {
        let cfg = WalkForwardConfig::default();
        let analyzer = WalkForwardAnalyzer::new(cfg, config::BacktestConfig::default());
        let cpcv = CPCVConfig { n_folds: 5, n_test_folds: 2, purge_gap_days: 2, min_samples_per_fold: 10 };
        let folds = analyzer.create_cpcv_folds(500, &cpcv);
        assert_eq!(folds.len(), 5);
        for f in &folds {
            assert!(!f.test_indices.is_empty());
            assert!(!f.train_indices.is_empty());
        }
    }
}
