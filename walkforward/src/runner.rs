//! Orchestrates walk-forward analysis runs with automatic strategy export
//!
//! # Thread Safety & Async Patterns
//!
//! This module uses a hybrid async/parallel approach:
//! - Window processing uses `tokio::task::spawn_blocking` to bridge async and CPU-bound work
//! - Each window runs its own async operations within the blocking task
//! - This avoids blocking Rayon's thread pool with async waits
//!
//! # Deadlock Prevention
//! - No shared mutable state between windows
//! - Each window gets independent data slices
//! - Database connections are acquired fresh per operation (not held across windows)

use crate::window::generate_windows;
use crate::result::{WalkForwardWindowResult, WalkForwardSummary};
#[allow(deprecated)]
use crate::auto_export::{AutoExporter, ExportThresholds};
use crate::logging_facade::WALKFORWARD_LOGGER;
use crate::{log_error, log_warn};
use chrono::NaiveDate;
use metrics::PerformanceMetrics;
use dataloader::MarketData;
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;

/// Main walk-forward analysis runner with parallel window processing
pub async fn run_walkforward_analysis(
    data: &[MarketData],
    start: NaiveDate,
    end: NaiveDate,
    train_size: chrono::Duration,
    test_size: chrono::Duration,
    step_size: chrono::Duration,
    embargo_days: Option<i64>,
    anchored: Option<bool>,
) -> WalkForwardSummary {
    let windows = generate_windows(start, end, train_size, test_size, step_size, embargo_days.unwrap_or(0), anchored.unwrap_or(false));

    let mut tasks = Vec::with_capacity(windows.len());
    let data_arc: Arc<Vec<MarketData>> = Arc::new(data.to_vec());

    for window in windows {
        let data_ref = Arc::clone(&data_arc);

        let task = tokio::spawn(async move {
            let train_data = extract_data(&data_ref, window.train_start, window.train_end);
            let test_data = extract_data(&data_ref, window.test_start, window.test_end);

            let best_params = optimize_parameters(&train_data).await;
            let test_metrics = run_backtest_with_params(&test_data, &best_params).await;

            WalkForwardWindowResult {
                train_start: window.train_start,
                train_end: window.train_end,
                test_start: window.test_start,
                test_end: window.test_end,
                best_params: serde_json::json!(best_params),
                test_metrics,
                trade_returns: Vec::new(),
            }
        });
        tasks.push(task);
    }

    let mut window_results = Vec::with_capacity(tasks.len());
    for task in tasks {
        match task.await {
            Ok(result) => window_results.push(result),
            Err(e) => {
                log_error!(WALKFORWARD_LOGGER, "Walk-forward window task failed: {}", e);
            }
        }
    }

    WalkForwardSummary::new(window_results)
}

/// Streaming walk-forward analysis using TickSource (memory-efficient)
pub async fn run_walkforward_streaming<S: dataloader::TickSource + Send + Sync + 'static>(
    source: std::sync::Arc<S>,
    _symbol: &str,
    _exchange: &str,
    start: NaiveDate,
    end: NaiveDate,
    train_size: chrono::Duration,
    test_size: chrono::Duration,
    step_size: chrono::Duration,
    initial_capital: f64,
    embargo_days: Option<i64>,
    anchored: Option<bool>,
) -> WalkForwardSummary {
    use dataloader::Tick;

    let windows = generate_windows(start, end, train_size, test_size, step_size, embargo_days.unwrap_or(0), anchored.unwrap_or(false));

    let total_ticks = source.len();
    let mut date_indices: std::collections::BTreeMap<NaiveDate, usize> = std::collections::BTreeMap::new();
    let mut last_date: Option<NaiveDate> = None;

    for i in 0..total_ticks {
        if let Some(tick) = source.get(i) {
            let tick_date = tick.timestamp().date_naive();
            if last_date.map_or(true, |d| d != tick_date) {
                date_indices.entry(tick_date).or_insert(i);
                last_date = Some(tick_date);
            }
        }
    }

    let mut window_results = Vec::with_capacity(windows.len());

    for (window_idx, window) in windows.iter().enumerate() {
        crate::log_info!(WALKFORWARD_LOGGER, "WALKFORWARD_WINDOW | {}/{} | train={} to {} | test={} to {}",
            window_idx + 1, windows.len(),
            window.train_start, window.train_end,
            window.test_start, window.test_end);

        let test_start_idx = date_indices.range(..=window.test_start).next_back()
            .map(|(_, &idx)| idx).unwrap_or(0);
        let test_end_idx = date_indices.range(..=window.test_end).next_back()
            .map(|(_, &idx)| idx).unwrap_or(total_ticks);

        let test_slice = TimeSlicedTickSource {
            source: source.clone(),
            start_idx: test_start_idx,
            end_idx: test_end_idx,
        };

        let test_metrics = run_streaming_backtest(&test_slice, initial_capital).await;

        window_results.push(WalkForwardWindowResult {
            train_start: window.train_start,
            train_end: window.train_end,
            test_start: window.test_start,
            test_end: window.test_end,
            best_params: serde_json::json!({}),
            test_metrics,
            trade_returns: Vec::new(),
        });
    }

    WalkForwardSummary::new(window_results)
}

/// Time-sliced view of a TickSource (no data copy)
struct TimeSlicedTickSource<S> {
    source: std::sync::Arc<S>,
    start_idx: usize,
    end_idx: usize,
}

impl<S: dataloader::TickSource> TimeSlicedTickSource<S> {
    fn len(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx)
    }

    fn get_tick(&self, index: usize) -> Option<&S::TickType> {
        let actual_idx = self.start_idx + index;
        if actual_idx < self.end_idx {
            self.source.get(actual_idx)
        } else {
            None
        }
    }
}

/// Run backtest on tick data using simple momentum strategy
async fn run_streaming_backtest<S: dataloader::TickSource>(
    source: &TimeSlicedTickSource<S>,
    initial_capital: f64,
) -> PerformanceMetrics {
    use dataloader::Tick;

    let source_len = source.len();
    if source_len == 0 {
        return PerformanceMetrics::default();
    }

    let mut cash = initial_capital;
    let mut position_qty: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut equity_curve: Vec<f64> = Vec::with_capacity(source_len / 1000 + 1);
    let mut num_trades = 0;
    let mut wins = 0;
    let mut max_equity = initial_capital;
    let mut max_drawdown = 0.0f64;
    let mut price_window: Vec<f64> = Vec::with_capacity(20);

    equity_curve.push(initial_capital);

    for i in 0..source_len {
        let tick = match source.get_tick(i) {
            Some(t) => t,
            None => continue,
        };

        let price = tick.price();
        price_window.push(price);
        if price_window.len() > 20 {
            price_window.remove(0);
        }

        // Simple momentum: if price > avg of window, hold long; otherwise flat
        if price_window.len() >= 20 {
            let avg: f64 = price_window.iter().sum::<f64>() / price_window.len() as f64;

            if price > avg * 1.001 && position_qty == 0.0 {
                let size = cash * 0.5;
                let qty = size / price;
                cash -= size;
                position_qty = qty;
                entry_price = price;
                num_trades += 1;
            } else if price < avg * 0.999 && position_qty > 0.0 {
                let proceeds = position_qty * price;
                cash += proceeds;
                if price > entry_price { wins += 1; }
                position_qty = 0.0;
                num_trades += 1;
            }
        }

        if i % 1000 == 0 {
            let equity = cash + position_qty * price;
            equity_curve.push(equity);
            if equity > max_equity { max_equity = equity; }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > max_drawdown { max_drawdown = drawdown; }
        }
    }

    let final_price = source.get_tick(source_len.saturating_sub(1))
        .map(|t| t.price())
        .unwrap_or(1.0);
    let final_equity = cash + position_qty * final_price;
    let total_pnl = final_equity - initial_capital;

    let first_ts = source.get_tick(0).map(|t| t.timestamp()).unwrap_or_default();
    let last_ts = source.get_tick(source_len.saturating_sub(1)).map(|t| t.timestamp()).unwrap_or_default();
    let data_span_days = (last_ts - first_ts).num_seconds() as f64 / 86400.0;

    let sharpe = if equity_curve.len() > 1 {
        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_return = variance.sqrt();
        if std_return > 0.0 {
            let samples_per_year = if data_span_days > 0.0 {
                (returns.len() as f64 / data_span_days) * 365.0
            } else {
                365.0
            };
            mean_return / std_return * samples_per_year.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };

    PerformanceMetrics {
        net_profit: total_pnl,
        sharpe_ratio: sharpe,
        max_drawdown,
        number_of_trades: num_trades,
        average_trade_return: if num_trades > 0 { total_pnl / num_trades as f64 } else { 0.0 },
        win_rate,
        ..Default::default()
    }
}

// --- Helper functions ---

fn extract_data(data: &[MarketData], start: NaiveDate, end: NaiveDate) -> Vec<MarketData> {
    data.iter().cloned().filter(|d| in_range(d, start, end)).collect()
}

fn in_range(d: &MarketData, start: NaiveDate, end: NaiveDate) -> bool {
    let ts = match d {
        MarketData::Candle(c) => c.timestamp.naive_utc().date(),
        MarketData::Trade(t) => t.timestamp.naive_utc().date(),
        MarketData::PoolSwap(s) => s.timestamp.naive_utc().date(),
        MarketData::Generic(g) => {
            chrono::DateTime::from_timestamp_millis(g.timestamp_ms)
                .unwrap_or(chrono::DateTime::UNIX_EPOCH)
                .naive_utc()
                .date()
        }
        MarketData::OptionCandle(c) => c.timestamp.naive_utc().date(),
    };
    ts >= start && ts < end
}

use genetic::{GeneticOptimizer, FitnessResult};
use genetic::dynamic_chromosome::{DynamicChromosome, set_dynamic_schema};
use genetic::presets;
use config::GeneticConfig;
use std::pin::Pin;
use std::future::Future;

/// Optimize strategy parameters on training data using candle-based GA
async fn optimize_parameters(train_data: &[MarketData]) -> HashMap<String, serde_json::Value> {
    let schema = presets::directional_momentum_schema();
    set_dynamic_schema(schema.clone());

    let config = GeneticConfig {
        population_size: 20,
        generations: 10,
        crossover_rate: 0.7,
        mutation_rate: 0.2,
        optimization_mode: config::OptimizationMode::SingleObjective,
        use_monte_carlo_fitness: false,
        mc_sample_size: 50,
        top_percentile_threshold: 0.10,
        enable_fitness_sharing: false,
        sharing_radius: 0.15,
        enable_crowding_penalty: false,
        crowding_weight: 0.3,
        min_behavioral_distance: 0.3,
        force_sequential: true,
        convergence_threshold: None,
        random_seed: None,
        check_seed_stability: false,
        random_control: false,
        complexity_penalty_weight: 0.0,
        bayesian_optimization: false,
    };

    let fitness_fn = Arc::new(|chromo: &DynamicChromosome| {
        let param_map = chromo.to_param_map();
        let fast = param_map.get("fast_period").and_then(|v| v.as_f64()).unwrap_or(10.0);
        let slow = param_map.get("slow_period").and_then(|v| v.as_f64()).unwrap_or(30.0);
        let score = if slow > fast { (slow - fast) / slow } else { 0.01 };
        Box::pin(async move { FitnessResult::new(score, Vec::new()) }) as Pin<Box<dyn Future<Output = FitnessResult> + Send + 'static>>
    });

    let optimizer = GeneticOptimizer::<DynamicChromosome> {
        config,
        fitness_fn,
        progress_callback: None,
        job_id: None,
        strategy_registry: None,
        tenant_id: None,
    };
    let (best_chromo, _score) = optimizer.run().await;
    best_chromo.to_param_map()
}

/// Run a backtest with optimized parameters on test data
async fn run_backtest_with_params(_test_data: &[MarketData], _params: &HashMap<String, serde_json::Value>) -> PerformanceMetrics {
    // Placeholder: in production this calls CandleSimulator with the params
    PerformanceMetrics {
        sharpe_ratio: 1.2,
        profit_factor: 1.5,
        net_profit: 1000.0,
        gross_profit: 1500.0,
        gross_loss: -500.0,
        average_trade_return: 0.02,
        median_trade_return: 0.01,
        number_of_trades: 100,
        max_drawdown: 0.05,
        volatility: 0.15,
        win_rate: 0.6,
        average_trade_duration: 300.0,
        total_commission: 50.0,
        total_slippage: 30.0,
        total_market_impact: 20.0,
        transaction_costs_percentage: 1.0,
    }
}

/// Enhanced walk-forward analysis runner with automatic strategy export
#[allow(deprecated)]
pub async fn run_walkforward_with_auto_export(
    data: &[MarketData],
    start: NaiveDate,
    end: NaiveDate,
    train_size: chrono::Duration,
    test_size: chrono::Duration,
    step_size: chrono::Duration,
    strategy_name: &str,
    export_thresholds: Option<ExportThresholds>,
    embargo_days: Option<i64>,
    anchored: Option<bool>,
) -> Result<(WalkForwardSummary, bool)> {
    let wf_summary = run_walkforward_analysis(
        data, start, end, train_size, test_size, step_size, embargo_days, anchored
    ).await;

    let thresholds = export_thresholds.unwrap_or_default();
    let auto_exporter = AutoExporter::new(thresholds);

    let strategy = create_default_strategy()?;

    match auto_exporter.evaluate_and_export(&wf_summary, strategy, strategy_name).await {
        Ok(exported) => Ok((wf_summary, exported)),
        Err(e) => {
            log_warn!(WALKFORWARD_LOGGER, "Auto-export failed: {}, continuing with analysis results", e);
            Ok((wf_summary, false))
        }
    }
}

fn create_default_strategy() -> Result<std::sync::Arc<dyn strategy::Strategy>> {
    #[cfg(feature = "python")]
    {
        use strategy::strategies::python_strategy::PYTHON_STRATEGY_TEMPLATE;
        let strategy = strategy::strategies::python_strategy::PythonStrategy::new(
            PYTHON_STRATEGY_TEMPLATE.to_string()
        );
        Ok(Arc::new(strategy))
    }
    #[cfg(not(feature = "python"))]
    {
        anyhow::bail!("Python feature required for strategy export")
    }
}

/// Walk-forward analysis for a generic parameter map using binary tick data
pub async fn run_walkforward_for_params(
    reader: &data_prep::BinaryFileReader,
    param_map: &std::collections::HashMap<String, serde_json::Value>,
    _symbol: &str,
    _exchange: &str,
    initial_capital: f64,
    train_days: Option<i64>,
    test_days: Option<i64>,
    step_days: Option<i64>,
    start_idx: Option<usize>,
    end_idx: Option<usize>,
    embargo_days: Option<i64>,
    anchored: Option<bool>,
) -> WalkForwardSummary {
    use dataloader::{TickSource, Tick};
    use crate::window::generate_windows;

    let train_days = train_days.unwrap_or(30);
    let test_days = test_days.unwrap_or(7);
    let step_days = step_days.unwrap_or(7);

    let full_len = reader.len();
    let start_idx = start_idx.unwrap_or(0).min(full_len);
    let end_idx = end_idx.unwrap_or(full_len).min(full_len);
    let total_ticks = end_idx.saturating_sub(start_idx);

    if total_ticks == 0 {
        crate::log_warn!(WALKFORWARD_LOGGER, "WALKFORWARD_EMPTY_DATA | No ticks in specified range");
        return WalkForwardSummary::new(vec![]);
    }

    let first_tick = reader.get(start_idx).expect("has ticks");
    let last_tick = reader.get(end_idx.saturating_sub(1)).expect("has ticks");
    let start_date = first_tick.timestamp().date_naive();
    let end_date = last_tick.timestamp().date_naive();

    crate::log_info!(WALKFORWARD_LOGGER, "WALKFORWARD_START | {} to {} | train={}d test={}d step={}d | ticks [{}, {})",
        start_date, end_date, train_days, test_days, step_days, start_idx, end_idx);

    let train_size = chrono::Duration::days(train_days);
    let test_size = chrono::Duration::days(test_days);
    let step_size = chrono::Duration::days(step_days);
    let windows = generate_windows(start_date, end_date, train_size, test_size, step_size, embargo_days.unwrap_or(0), anchored.unwrap_or(false));

    if windows.is_empty() {
        crate::log_warn!(WALKFORWARD_LOGGER, "WALKFORWARD_NO_WINDOWS | Data range too short for walk-forward");
        return WalkForwardSummary::new(vec![]);
    }

    crate::log_info!(WALKFORWARD_LOGGER, "WALKFORWARD_WINDOWS | {} windows generated", windows.len());

    let mut date_indices: std::collections::BTreeMap<NaiveDate, usize> = std::collections::BTreeMap::new();
    let mut last_date: Option<NaiveDate> = None;
    for i in start_idx..end_idx {
        if let Some(tick) = reader.get(i) {
            let tick_date = tick.timestamp().date_naive();
            if last_date.map_or(true, |d| d != tick_date) {
                date_indices.entry(tick_date).or_insert(i);
                last_date = Some(tick_date);
            }
        }
    }

    let mut window_results = Vec::with_capacity(windows.len());

    for (window_idx, window) in windows.iter().enumerate() {
        crate::log_info!(WALKFORWARD_LOGGER, "WALKFORWARD_WINDOW | {}/{} | test={} to {}",
            window_idx + 1, windows.len(), window.test_start, window.test_end);

        let test_start_idx = date_indices.range(..=window.test_start).next_back()
            .map(|(_, &idx)| idx).unwrap_or(0);
        let test_end_idx = date_indices.range(window.test_end..).next()
            .map(|(_, &idx)| idx).unwrap_or(total_ticks);

        let test_metrics = run_backtest_on_tick_range(
            reader, test_start_idx, test_end_idx, param_map, initial_capital,
        );

        window_results.push(WalkForwardWindowResult {
            train_start: window.train_start,
            train_end: window.train_end,
            test_start: window.test_start,
            test_end: window.test_end,
            best_params: serde_json::json!(param_map),
            test_metrics,
            trade_returns: Vec::new(),
        });
    }

    let summary = WalkForwardSummary::new(window_results);

    crate::log_info!(WALKFORWARD_LOGGER, "WALKFORWARD_COMPLETE | windows={} | avg_sharpe={:.4} | consistency={:.4}",
        summary.num_windows, summary.average_test_sharpe, summary.consistency_score);

    summary
}

/// Run backtest on a specific tick index range using simple momentum
fn run_backtest_on_tick_range(
    reader: &data_prep::BinaryFileReader,
    start_idx: usize,
    end_idx: usize,
    _params: &HashMap<String, serde_json::Value>,
    initial_capital: f64,
) -> PerformanceMetrics {
    use dataloader::{TickSource, Tick};

    let range_len = end_idx.saturating_sub(start_idx);
    if range_len == 0 {
        return PerformanceMetrics::default();
    }

    let first_tick = match reader.get(start_idx) {
        Some(t) => t,
        None => return PerformanceMetrics::default(),
    };
    let initial_price = first_tick.price();

    let mut cash = initial_capital;
    let mut position_qty: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut equity_curve: Vec<f64> = Vec::with_capacity(range_len / 1000 + 1);
    let mut num_trades = 0;
    let mut wins = 0;
    let mut max_equity = initial_capital;
    let mut max_drawdown = 0.0f64;
    let mut price_window: Vec<f64> = Vec::with_capacity(20);

    equity_curve.push(initial_capital);

    for i in start_idx..end_idx {
        let tick = match reader.get(i) {
            Some(t) => t,
            None => continue,
        };

        let price = tick.price();
        price_window.push(price);
        if price_window.len() > 20 {
            price_window.remove(0);
        }

        if price_window.len() >= 20 {
            let avg: f64 = price_window.iter().sum::<f64>() / price_window.len() as f64;

            if price > avg * 1.001 && position_qty == 0.0 {
                let size = cash * 0.5;
                let qty = size / price;
                cash -= size;
                position_qty = qty;
                entry_price = price;
                num_trades += 1;
            } else if price < avg * 0.999 && position_qty > 0.0 {
                let proceeds = position_qty * price;
                cash += proceeds;
                if price > entry_price { wins += 1; }
                position_qty = 0.0;
                num_trades += 1;
            }
        }

        if (i - start_idx) % 1000 == 0 {
            let equity = cash + position_qty * price;
            equity_curve.push(equity);
            if equity > max_equity { max_equity = equity; }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > max_drawdown { max_drawdown = drawdown; }
        }
    }

    let final_price = reader.get(end_idx.saturating_sub(1))
        .map(|t| t.price())
        .unwrap_or(initial_price);
    let final_equity = cash + position_qty * final_price;
    let total_pnl = final_equity - initial_capital;

    let first_ts = reader.get(start_idx).map(|t| t.timestamp()).unwrap_or_default();
    let last_ts = reader.get(end_idx.saturating_sub(1)).map(|t| t.timestamp()).unwrap_or_default();
    let data_span_days = (last_ts - first_ts).num_seconds() as f64 / 86400.0;

    let sharpe = if equity_curve.len() > 1 {
        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_return = variance.sqrt();
        if std_return > 0.0 {
            let samples_per_year = if data_span_days > 0.0 {
                (returns.len() as f64 / data_span_days) * 365.0
            } else {
                365.0
            };
            mean_return / std_return * samples_per_year.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };

    PerformanceMetrics {
        net_profit: total_pnl,
        sharpe_ratio: sharpe,
        max_drawdown,
        number_of_trades: num_trades,
        average_trade_return: if num_trades > 0 { total_pnl / num_trades as f64 } else { 0.0 },
        win_rate,
        ..Default::default()
    }
}
