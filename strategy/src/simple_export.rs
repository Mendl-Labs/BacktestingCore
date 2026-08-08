//! Simple Strategy Export System
//!
//! Exports validated strategies directly to live trading after walk-forward analysis.
//!
//! ## Deployment Pipeline
//!
//! The pipeline relies on walk-forward analysis for strategy validation:
//!
//! ```text
//! GA Optimization → Walk-Forward Analysis → Monte Carlo → Export to SignalEngine
//!                         ↓
//!              (Out-of-sample validation)
//! ```
//!
//! Walk-forward analysis provides superior validation compared to paper trading:
//! - Tests on historical out-of-sample data (not just in-sample)
//! - Validates parameter stability across multiple time windows
//! - Combined with Monte Carlo for robustness testing
//!
//! Paper trading was removed as it only validates infrastructure (covered by
//! integration tests) and cannot catch strategy issues that walk-forward misses.

use crate::logging_facade::STRATEGY_LOGGER;
use crate::log_info;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Simple strategy export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Minimum Sharpe ratio to qualify for export
    pub min_sharpe_ratio: f64,
    /// Minimum profit factor to qualify for export  
    pub min_profit_factor: f64,
    /// Minimum win rate (%) to qualify for export
    pub min_win_rate: f64,
    /// Maximum acceptable drawdown percentage
    pub max_drawdown_pct: f64,
    /// Minimum number of trades required
    pub min_trades: u64,
    /// Export directory path
    pub export_path: String,
}

impl ExportConfig {
    /// Conservative export criteria - only best strategies
    pub fn conservative() -> Self {
        Self {
            min_sharpe_ratio: 2.0,
            min_profit_factor: 1.8,
            min_win_rate: 0.7,
            max_drawdown_pct: 0.1,
            min_trades: 100,
            export_path: "C:/Users/ikenn/Projects/TradingPlatform/SignalEngine/exported_strategies".to_string(),
        }
    }

    /// Aggressive export criteria - more experimental strategies
    pub fn aggressive() -> Self {
        Self {
            min_sharpe_ratio: 1.0,
            min_profit_factor: 1.1,
            min_win_rate: 0.55,
            max_drawdown_pct: 0.25,
            min_trades: 50,
            export_path: "C:/Users/ikenn/Projects/TradingPlatform/SignalEngine/exported_strategies".to_string(),
        }
    }

    /// Research export criteria - for analysis purposes
    pub fn research() -> Self {
        Self {
            min_sharpe_ratio: 0.5,
            min_profit_factor: 1.0,
            min_win_rate: 0.5,
            max_drawdown_pct: 0.4,
            min_trades: 20,
            export_path: "C:/Users/ikenn/Projects/TradingPlatform/SignalEngine/research_strategies".to_string(),
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            min_sharpe_ratio: 1.5,
            min_profit_factor: 1.3,
            min_win_rate: 0.6,
            max_drawdown_pct: 0.15,
            min_trades: 100,
            export_path: "C:/Users/ikenn/Projects/TradingPlatform/SignalEngine/exported_strategies".to_string(),
        }
    }
}

/// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    /// Total number of signals generated
    pub total_signals: u64,
    /// Number of buy signals
    pub buy_signals: u64,
    /// Number of sell signals
    pub sell_signals: u64,
    /// Average signal strength
    pub avg_signal_strength: f64,
    /// Strategy uptime
    pub uptime_percentage: f64,
    /// Last signal timestamp
    pub last_signal_time: Option<DateTime<Utc>>,
    /// Strategy-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            total_signals: 0,
            buy_signals: 0,
            sell_signals: 0,
            avg_signal_strength: 0.0,
            uptime_percentage: 0.0,
            last_signal_time: None,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Backtest results for export evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub total_pnl: f64,
    pub num_trades: u64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub avg_trade_return: f64,
    pub median_trade_return: f64,
    pub avg_trade_duration_hours: f64,
    pub volatility: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub net_profit: f64,
    pub signals_generated: u64,
    pub strategy_metrics: StrategyMetrics,
    pub backtest_start: DateTime<Utc>,
    pub backtest_end: DateTime<Utc>,
}

/// Exportable strategy for live trading (SignalEngine format)
#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct ExportableStrategy {
    /// Strategy configuration section
    pub config: StrategyConfig,
    /// Backtest performance results
    pub performance: PerformanceData,
    /// Optimized parameters from backtesting
    pub optimized_parameters: HashMap<String, ParameterValue>,
    /// Live trading configuration
    pub live_trading_config: LiveTradingConfig,
    /// Export metadata
    pub export_metadata: ExportMetadata,
}

/// Strategy configuration for SignalEngine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub id: String,
    pub name: String,
    pub strategy_type: String,
    pub symbol: String,
    pub exchange: String,
    pub enabled: bool,
    pub parameters: HashMap<String, ParameterValue>,
}

/// Performance data in SignalEngine format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub total_pnl: String,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub max_drawdown: String,
    pub num_trades: u64,
    pub avg_trade_return: f64,
    pub volatility: f64,
    pub avg_trade_duration_hours: f64,
    pub backtest_start: DateTime<Utc>,
    pub backtest_end: DateTime<Utc>,
}

/// Parameter value enum for SignalEngine compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
}

/// Live trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTradingConfig {
    pub max_position_pct: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: Option<f64>,
    pub daily_loss_limit: String,
    pub cooldown_minutes: u32,
    pub risk_scaling: f64,
}

impl Default for LiveTradingConfig {
    fn default() -> Self {
        Self {
            max_position_pct: 0.05,
            stop_loss_pct: 0.02,
            take_profit_pct: None,
            daily_loss_limit: "1000.00".to_string(),
            cooldown_minutes: 30,
            risk_scaling: 1.0,
        }
    }
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub export_time: DateTime<Utc>,
    pub strategy_version: String,
    pub engine_version: String,
    pub authorized_by: String,
    pub approval_level: String,
    pub notes: String,
}

/// Simple strategy exporter
pub struct StrategyExporter {
    config: ExportConfig,
}

impl StrategyExporter {
    /// Create new exporter with default config
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
        }
    }

    /// Create new exporter with custom config
    pub fn with_config(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Create exporter with preset configuration
    pub fn with_preset(preset: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = match preset.to_lowercase().as_str() {
            "conservative" => ExportConfig::conservative(),
            "aggressive" => ExportConfig::aggressive(), 
            "research" => ExportConfig::research(),
            "default" => ExportConfig::default(),
            _ => return Err(format!("Unknown preset: {}. Available: conservative, aggressive, research, default", preset).into()),
        };
        Ok(Self { config })
    }

    /// Check if strategy meets export criteria
    pub fn should_export(&self, results: &BacktestResults) -> bool {
        self.should_export_with_reason(results).0
    }

    /// Check if strategy meets export criteria and return reason if not
    pub fn should_export_with_reason(&self, results: &BacktestResults) -> (bool, Option<String>) {
        let mut reasons = Vec::new();
        
        if results.sharpe_ratio < self.config.min_sharpe_ratio {
            reasons.push(format!("Sharpe ratio {:.2} < {:.2}", 
                results.sharpe_ratio, self.config.min_sharpe_ratio));
        }
        
        if results.profit_factor < self.config.min_profit_factor {
            reasons.push(format!("Profit factor {:.2} < {:.2}", 
                results.profit_factor, self.config.min_profit_factor));
        }
        
        if results.win_rate < self.config.min_win_rate {
            reasons.push(format!("Win rate {:.1}% < {:.1}%", 
                results.win_rate * 100.0, self.config.min_win_rate * 100.0));
        }
        
        let max_drawdown_pct = results.max_drawdown.abs();
        if max_drawdown_pct > self.config.max_drawdown_pct {
            reasons.push(format!("Max drawdown {:.1}% > {:.1}%", 
                max_drawdown_pct * 100.0, self.config.max_drawdown_pct * 100.0));
        }
        
        if results.num_trades < self.config.min_trades {
            reasons.push(format!("Number of trades {} < {}", 
                results.num_trades, self.config.min_trades));
        }
        
        if reasons.is_empty() {
            (true, None)
        } else {
            (false, Some(reasons.join(", ")))
        }
    }

    /// Export strategy to JSON file if it meets criteria
    pub async fn export_if_qualified(
        &self,
        strategy_name: &str,
        symbol: &str,
        exchange: &str,
        parameters: HashMap<String, f64>,
        results: BacktestResults,
    ) -> Result<Option<String>> {
        if !self.should_export(&results) {
            return Ok(None);
        }

        // Convert f64 parameters to ParameterValue enum
        let param_values: HashMap<String, ParameterValue> = parameters.iter()
            .map(|(k, v)| (k.clone(), ParameterValue::Float(*v)))
            .collect();

        let exportable = ExportableStrategy {
            config: StrategyConfig {
                id: format!("{}-{}-001", strategy_name.to_lowercase(), symbol.to_lowercase()),
                name: strategy_name.to_string(),
                strategy_type: strategy_name.to_string(),
                symbol: symbol.to_string(),
                exchange: exchange.to_string(),
                enabled: true,
                parameters: param_values.clone(),
            },
            performance: PerformanceData {
                total_pnl: results.total_pnl.to_string(),
                sharpe_ratio: results.sharpe_ratio,
                profit_factor: results.profit_factor,
                win_rate: results.win_rate / 100.0, // Convert percentage to decimal
                max_drawdown: format!("-{}", results.max_drawdown),
                num_trades: results.num_trades,
                avg_trade_return: results.avg_trade_return,
                volatility: results.volatility,
                avg_trade_duration_hours: results.avg_trade_duration_hours,
                backtest_start: results.backtest_start,
                backtest_end: results.backtest_end,
            },
            optimized_parameters: param_values,
            live_trading_config: LiveTradingConfig::default(),
            export_metadata: ExportMetadata {
                export_time: Utc::now(),
                strategy_version: "1.0.0".to_string(),
                engine_version: "0.1.0".to_string(),
                authorized_by: "backtesting-engine".to_string(),
                approval_level: "Development".to_string(),
                notes: format!("Auto-exported strategy with Sharpe: {:.2}, PF: {:.2}, Win Rate: {:.1}%", 
                    results.sharpe_ratio, results.profit_factor, results.win_rate),
            },
        };

        let filename = format!(
            "{}_{}.json",
            strategy_name.to_lowercase().replace(" ", "-"),
            exportable.export_metadata.export_time.format("%Y%m%d_%H%M%S"),
        );
        let filepath = Path::new(&self.config.export_path).join(&filename);

        // Create directory if it doesn't exist
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write strategy to file
        let json = serde_json::to_string_pretty(&exportable)?;
        fs::write(&filepath, json).await?;

        log_info!(STRATEGY_LOGGER, "Strategy exported to SignalEngine: {} (Sharpe: {:.2}, PF: {:.2}, Win Rate: {:.1}%)",
            filename, exportable.performance.sharpe_ratio,
            exportable.performance.profit_factor, exportable.performance.win_rate * 100.0);

        Ok(Some(filepath.to_string_lossy().to_string()))
    }

    /// Export strategy unconditionally (for testing)
    pub async fn force_export(
        &self,
        strategy_name: &str,
        symbol: &str,
        exchange: &str,
        parameters: HashMap<String, f64>,
        results: BacktestResults,
    ) -> Result<String> {
        // Convert f64 parameters to ParameterValue enum
        let param_values: HashMap<String, ParameterValue> = parameters.iter()
            .map(|(k, v)| (k.clone(), ParameterValue::Float(*v)))
            .collect();

        let exportable = ExportableStrategy {
            config: StrategyConfig {
                id: format!("{}-{}-forced", strategy_name.to_lowercase(), symbol.to_lowercase()),
                name: strategy_name.to_string(),
                strategy_type: strategy_name.to_string(),
                symbol: symbol.to_string(),
                exchange: exchange.to_string(),
                enabled: true,
                parameters: param_values.clone(),
            },
            performance: PerformanceData {
                total_pnl: results.total_pnl.to_string(),
                sharpe_ratio: results.sharpe_ratio,
                profit_factor: results.profit_factor,
                win_rate: results.win_rate / 100.0, // Convert percentage to decimal
                max_drawdown: format!("-{}", results.max_drawdown),
                num_trades: results.num_trades,
                avg_trade_return: results.avg_trade_return,
                volatility: results.volatility,
                avg_trade_duration_hours: results.avg_trade_duration_hours,
                backtest_start: results.backtest_start,
                backtest_end: results.backtest_end,
            },
            optimized_parameters: param_values,
            live_trading_config: LiveTradingConfig::default(),
            export_metadata: ExportMetadata {
                export_time: Utc::now(),
                strategy_version: "1.0.0".to_string(),
                engine_version: "0.1.0".to_string(),
                authorized_by: "backtesting-engine".to_string(),
                approval_level: "Testing".to_string(),
                notes: "Force exported strategy for testing purposes".to_string(),
            },
        };

        let filename = format!(
            "{}_force_export_{}.json",
            strategy_name.to_lowercase().replace(" ", "-"),
            exportable.export_metadata.export_time.format("%Y%m%d_%H%M%S"),
        );
        let filepath = Path::new(&self.config.export_path).join(&filename);

        // Create directory if it doesn't exist
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write strategy to file
        let json = serde_json::to_string_pretty(&exportable)?;
        fs::write(&filepath, json).await?;

        log_info!(STRATEGY_LOGGER, "Strategy force exported to SignalEngine: {}", filename);

        Ok(filepath.to_string_lossy().to_string())
    }

    /// List all exported strategies
    pub async fn list_exported(&self) -> Result<Vec<String>> {
        let export_dir = Path::new(&self.config.export_path);
        if !export_dir.exists() {
            return Ok(Vec::new());
        }

        let mut entries = fs::read_dir(export_dir).await?;
        let mut exported_files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".json") && (filename.contains("force_export") || !filename.contains("force_export")) {
                    exported_files.push(filename.to_string());
                }
            }
        }

        Ok(exported_files)
    }

    /// Export multiple strategies in batch
    pub async fn export_batch(
        &self,
        strategies: Vec<(ExportableStrategy, BacktestResults)>
    ) -> Result<BatchExportResult> {
        let mut successful = Vec::new();
        let mut failed = Vec::new();
        
        log_info!(STRATEGY_LOGGER, "Starting batch export of {} strategies", strategies.len());
        
        for (strategy, backtest_results) in strategies {
            let strategy_name = strategy.config.name.clone();
            
            // Convert ParameterValue HashMap to f64 HashMap
            let mut f64_params = HashMap::new();
            for (key, value) in &strategy.config.parameters {
                if let ParameterValue::Float(f) = value {
                    f64_params.insert(key.clone(), *f);
                }
            }
            
            match self.export_if_qualified(
                &strategy.config.name,
                &strategy.config.symbol,
                &strategy.config.exchange,
                f64_params,
                backtest_results.clone()
            ).await {
                Ok(Some(path)) => {
                    successful.push((strategy_name, path));
                },
                Ok(None) => {
                    let (_, reason) = self.should_export_with_reason(&backtest_results);
                    failed.push((strategy_name, format!("Did not meet criteria: {}", 
                        reason.unwrap_or("Unknown reason".to_string()))));
                },
                Err(e) => {
                    failed.push((strategy_name, format!("Export error: {}", e)));
                }
            }
        }
        
        let result = BatchExportResult { successful, failed };
        result.print_summary();
        Ok(result)
    }

    /// Get export statistics for a set of strategies
    pub fn get_export_stats(&self, strategies: &[(ExportableStrategy, BacktestResults)]) -> ExportStatistics {
        let total = strategies.len();
        let mut qualified = 0;
        let mut rejection_reasons = HashMap::new();
        
        for (_, results) in strategies {
            let (should_export, reason) = self.should_export_with_reason(results);
            if should_export {
                qualified += 1;
            } else if let Some(reason) = reason {
                *rejection_reasons.entry(reason).or_insert(0) += 1;
            }
        }
        
        ExportStatistics {
            total_evaluated: total,
            qualified_for_export: qualified,
            rejection_rate: if total > 0 { (total - qualified) as f64 / total as f64 } else { 0.0 },
            rejection_reasons,
        }
    }

    /// Load exported strategy from file
    pub async fn load_exported(&self, filename: &str) -> Result<ExportableStrategy> {
        let filepath = Path::new(&self.config.export_path).join(filename);
        let content = fs::read_to_string(&filepath).await?;
        let strategy: ExportableStrategy = serde_json::from_str(&content)?;
        Ok(strategy)
    }

    /// List exported strategies (alias for list_exported)
    pub async fn list_exported_strategies(&self) -> Result<Vec<String>> {
        self.list_exported().await
    }
}

/// Result of batch export operation
#[derive(Debug, Clone)]
pub struct BatchExportResult {
    pub successful: Vec<(String, String)>, // (strategy_name, file_path)
    pub failed: Vec<(String, String)>,     // (strategy_name, reason)
}

impl BatchExportResult {
    pub fn print_summary(&self) {
        use crate::logging_facade::STRATEGY_LOGGER;
        
        STRATEGY_LOGGER.info_sync(format!("Batch Export Summary: Successful: {}, Failed: {}", 
            self.successful.len(), self.failed.len()));
        
        if !self.successful.is_empty() {
            let names: Vec<_> = self.successful.iter().map(|(name, _)| name.as_str()).collect();
            STRATEGY_LOGGER.info_sync(format!("Exported Strategies: {}", names.join(", ")));
        }
        
        if !self.failed.is_empty() {
            for (name, reason) in self.failed.iter().take(5) {
                STRATEGY_LOGGER.warn_sync(format!("Failed Export - {}: {}", name, reason));
            }
            if self.failed.len() > 5 {
                STRATEGY_LOGGER.warn_sync(format!("... and {} more failures", self.failed.len() - 5));
            }
        }
    }
}

/// Export statistics for batch operations
#[derive(Debug, Clone)]
pub struct ExportStatistics {
    pub total_evaluated: usize,
    pub qualified_for_export: usize,
    pub rejection_rate: f64,
    pub rejection_reasons: HashMap<String, usize>,
}

impl ExportStatistics {
    pub fn print_summary(&self) {
        use crate::logging_facade::STRATEGY_LOGGER;
        
        STRATEGY_LOGGER.info_sync(format!("Export Statistics: Total Evaluated: {}, Qualified: {}, Rejection Rate: {:.1}%",
            self.total_evaluated, self.qualified_for_export, self.rejection_rate * 100.0));
        
        if !self.rejection_reasons.is_empty() {
            let mut sorted_reasons: Vec<_> = self.rejection_reasons.iter().collect();
            sorted_reasons.sort_by(|a, b| b.1.cmp(a.1));
            
            for (reason, count) in sorted_reasons.iter().take(3) {
                STRATEGY_LOGGER.info_sync(format!("Top Rejection Reason: {} ({})", reason, count));
            }
        }
    }
}

impl Default for StrategyExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_good_results() -> BacktestResults {
        BacktestResults {
            total_pnl: 5000.0,
            num_trades: 200,
            max_drawdown: 0.08,
            sharpe_ratio: 2.5,
            profit_factor: 2.0,
            win_rate: 0.65,
            avg_trade_return: 25.0,
            median_trade_return: 20.0,
            avg_trade_duration_hours: 4.0,
            volatility: 0.15,
            gross_profit: 8000.0,
            gross_loss: 3000.0,
            net_profit: 5000.0,
            signals_generated: 400,
            strategy_metrics: StrategyMetrics::default(),
            backtest_start: Utc::now() - chrono::Duration::days(365),
            backtest_end: Utc::now(),
        }
    }

    fn make_bad_results() -> BacktestResults {
        BacktestResults {
            sharpe_ratio: 0.5,
            profit_factor: 0.8,
            win_rate: 0.40,
            max_drawdown: 0.30,
            num_trades: 10,
            ..make_good_results()
        }
    }

    // ── ExportConfig presets ──

    #[test]
    fn export_config_default() {
        let c = ExportConfig::default();
        assert!((c.min_sharpe_ratio - 1.5).abs() < 1e-10);
        assert_eq!(c.min_trades, 100);
    }

    #[test]
    fn export_config_conservative() {
        let c = ExportConfig::conservative();
        assert!(c.min_sharpe_ratio > ExportConfig::default().min_sharpe_ratio);
    }

    #[test]
    fn export_config_aggressive() {
        let c = ExportConfig::aggressive();
        assert!(c.min_sharpe_ratio < ExportConfig::default().min_sharpe_ratio);
    }

    #[test]
    fn export_config_research() {
        let c = ExportConfig::research();
        assert!(c.min_sharpe_ratio < ExportConfig::aggressive().min_sharpe_ratio);
    }

    #[test]
    fn export_config_serde_roundtrip() {
        let c = ExportConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: ExportConfig = serde_json::from_str(&json).unwrap();
        assert!((c2.min_sharpe_ratio - c.min_sharpe_ratio).abs() < 1e-10);
    }

    // ── StrategyExporter ──

    #[test]
    fn exporter_with_preset_valid() {
        assert!(StrategyExporter::with_preset("conservative").is_ok());
        assert!(StrategyExporter::with_preset("aggressive").is_ok());
        assert!(StrategyExporter::with_preset("research").is_ok());
        assert!(StrategyExporter::with_preset("default").is_ok());
    }

    #[test]
    fn exporter_with_preset_invalid() {
        assert!(StrategyExporter::with_preset("nonexistent").is_err());
    }

    // ── should_export ──

    #[test]
    fn should_export_good_strategy() {
        let exporter = StrategyExporter::new();
        assert!(exporter.should_export(&make_good_results()));
    }

    #[test]
    fn should_export_bad_strategy_rejected() {
        let exporter = StrategyExporter::new();
        assert!(!exporter.should_export(&make_bad_results()));
    }

    #[test]
    fn should_export_with_reason_gives_details() {
        let exporter = StrategyExporter::new();
        let (pass, reason) = exporter.should_export_with_reason(&make_bad_results());
        assert!(!pass);
        let reason = reason.unwrap();
        assert!(reason.contains("Sharpe") || reason.contains("Profit factor") || reason.contains("Win rate"));
    }

    #[test]
    fn should_export_with_reason_pass_no_reason() {
        let exporter = StrategyExporter::new();
        let (pass, reason) = exporter.should_export_with_reason(&make_good_results());
        assert!(pass);
        assert!(reason.is_none());
    }

    #[test]
    fn should_export_low_trades_rejected() {
        let exporter = StrategyExporter::new(); // min_trades=100
        let mut r = make_good_results();
        r.num_trades = 5;
        assert!(!exporter.should_export(&r));
    }

    #[test]
    fn should_export_high_drawdown_rejected() {
        let exporter = StrategyExporter::new(); // max_drawdown_pct=0.15
        let mut r = make_good_results();
        r.max_drawdown = 0.50;
        assert!(!exporter.should_export(&r));
    }

    // ── get_export_stats ──

    #[test]
    fn get_export_stats_all_qualify() {
        let exporter = StrategyExporter::new();
        let good = make_good_results();
        let strategies: Vec<(ExportableStrategy, BacktestResults)> = vec![
            (make_exportable(), good.clone()),
            (make_exportable(), good.clone()),
        ];
        let stats = exporter.get_export_stats(&strategies);
        assert_eq!(stats.total_evaluated, 2);
        assert_eq!(stats.qualified_for_export, 2);
        assert!((stats.rejection_rate).abs() < 1e-10);
    }

    #[test]
    fn get_export_stats_mixed() {
        let exporter = StrategyExporter::new();
        let strategies: Vec<(ExportableStrategy, BacktestResults)> = vec![
            (make_exportable(), make_good_results()),
            (make_exportable(), make_bad_results()),
        ];
        let stats = exporter.get_export_stats(&strategies);
        assert_eq!(stats.total_evaluated, 2);
        assert_eq!(stats.qualified_for_export, 1);
        assert!((stats.rejection_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn get_export_stats_empty() {
        let exporter = StrategyExporter::new();
        let strategies: Vec<(ExportableStrategy, BacktestResults)> = vec![];
        let stats = exporter.get_export_stats(&strategies);
        assert_eq!(stats.total_evaluated, 0);
        assert!((stats.rejection_rate).abs() < 1e-10);
    }

    // ── Default impls ──

    #[test]
    fn strategy_metrics_default() {
        let m = StrategyMetrics::default();
        assert_eq!(m.total_signals, 0);
        assert!(m.last_signal_time.is_none());
    }

    #[test]
    fn live_trading_config_default() {
        let c = LiveTradingConfig::default();
        assert!((c.max_position_pct - 0.05).abs() < 1e-10);
        assert_eq!(c.cooldown_minutes, 30);
    }

    #[test]
    fn parameter_value_serde_roundtrip() {
        let values = vec![
            ParameterValue::Float(3.14),
            ParameterValue::Integer(42),
            ParameterValue::Boolean(true),
            ParameterValue::String("test".to_string()),
        ];
        for v in &values {
            let json = serde_json::to_string(v).unwrap();
            let v2: ParameterValue = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&v2).unwrap();
            assert_eq!(json, json2);
        }
    }

    fn make_exportable() -> ExportableStrategy {
        ExportableStrategy {
            config: StrategyConfig {
                id: "test-1".to_string(),
                name: "Test".to_string(),
                strategy_type: "momentum".to_string(),
                symbol: "BTC-USD".to_string(),
                exchange: "kraken".to_string(),
                enabled: true,
                parameters: HashMap::new(),
            },
            performance: PerformanceData {
                total_pnl: "5000.0".to_string(),
                sharpe_ratio: 2.5,
                profit_factor: 2.0,
                win_rate: 0.65,
                max_drawdown: "-0.08".to_string(),
                num_trades: 200,
                avg_trade_return: 25.0,
                volatility: 0.15,
                avg_trade_duration_hours: 4.0,
                backtest_start: Utc::now() - chrono::Duration::days(365),
                backtest_end: Utc::now(),
            },
            optimized_parameters: HashMap::new(),
            live_trading_config: LiveTradingConfig::default(),
            export_metadata: ExportMetadata {
                export_time: Utc::now(),
                strategy_version: "1.0".to_string(),
                engine_version: "0.1".to_string(),
                authorized_by: "test".to_string(),
                approval_level: "dev".to_string(),
                notes: "test".to_string(),
            },
        }
    }
}
