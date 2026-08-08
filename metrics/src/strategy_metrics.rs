//! Strategy-Specific Metrics Profiles
//!
//! This module defines which metrics are relevant for directional strategies
//! and provides filtered metric views.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy category for metrics filtering
/// 
/// Only Python-based directional strategies are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyType {
    /// Custom Python directional strategies
    Custom,
}

impl Default for StrategyType {
    fn default() -> Self {
        Self::Custom
    }
}

impl StrategyType {
    /// Parse strategy type from string
    pub fn from_str(_s: &str) -> Self {
        Self::Custom
    }
}

/// Metric category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricCategory {
    /// Core performance metrics (Sharpe, returns, etc.)
    Performance,
    /// Risk metrics (drawdown, volatility, VaR)
    Risk,
    /// Execution metrics (fill rate, slippage, latency)
    Execution,
    /// Inventory/position metrics (skew, turnover)
    Inventory,
    /// Cost metrics (commissions, market impact)
    Costs,
    /// Trade statistics (win rate, avg duration)
    TradeStats,
    /// Advanced/statistical metrics (Omega, Tail, Kelly)
    Advanced,
}

/// Individual metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Unique metric identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Description/tooltip
    pub description: String,
    /// Category for grouping
    pub category: MetricCategory,
    /// Unit of measurement (%, $, ratio, etc.)
    pub unit: String,
    /// Number of decimal places for display
    pub decimals: u8,
    /// Whether higher is better (for color coding)
    pub higher_is_better: bool,
    /// Warning threshold (yellow)
    pub warning_threshold: Option<f64>,
    /// Critical threshold (red)
    pub critical_threshold: Option<f64>,
}

/// Metrics profile for a strategy type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsProfile {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Primary metrics (always shown prominently)
    pub primary_metrics: Vec<String>,
    /// Secondary metrics (shown in detail view)
    pub secondary_metrics: Vec<String>,
    /// Hidden metrics (not relevant for this strategy)
    pub hidden_metrics: Vec<String>,
    /// Custom metric definitions
    pub metric_definitions: HashMap<String, MetricDefinition>,
}

impl MetricsProfile {
    /// Get profile for directional/custom strategies
    pub fn directional() -> Self {
        let mut definitions = HashMap::new();
        
        definitions.insert("net_profit".to_string(), MetricDefinition {
            id: "net_profit".to_string(),
            name: "Net P&L".to_string(),
            description: "Total profit/loss after all costs".to_string(),
            category: MetricCategory::Performance,
            unit: "$".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(0.0),
            critical_threshold: Some(-1000.0),
        });
        
        definitions.insert("sharpe_ratio".to_string(), MetricDefinition {
            id: "sharpe_ratio".to_string(),
            name: "Sharpe Ratio".to_string(),
            description: "Risk-adjusted return (annualized)".to_string(),
            category: MetricCategory::Performance,
            unit: "".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.0),
            critical_threshold: Some(0.0),
        });
        
        definitions.insert("max_drawdown".to_string(), MetricDefinition {
            id: "max_drawdown".to_string(),
            name: "Max Drawdown".to_string(),
            description: "Largest peak-to-trough decline".to_string(),
            category: MetricCategory::Risk,
            unit: "%".to_string(),
            decimals: 2,
            higher_is_better: false,
            warning_threshold: Some(15.0),
            critical_threshold: Some(30.0),
        });
        
        definitions.insert("trend_accuracy".to_string(), MetricDefinition {
            id: "trend_accuracy".to_string(),
            name: "Trend Accuracy".to_string(),
            description: "% of trades in correct trend direction".to_string(),
            category: MetricCategory::Performance,
            unit: "%".to_string(),
            decimals: 1,
            higher_is_better: true,
            warning_threshold: Some(50.0),
            critical_threshold: Some(40.0),
        });
        
        definitions.insert("avg_holding_period".to_string(), MetricDefinition {
            id: "avg_holding_period".to_string(),
            name: "Avg Holding Period".to_string(),
            description: "Average time in position".to_string(),
            category: MetricCategory::TradeStats,
            unit: "hours".to_string(),
            decimals: 1,
            higher_is_better: false, // Neutral really
            warning_threshold: None,
            critical_threshold: None,
        });
        
        definitions.insert("win_rate".to_string(), MetricDefinition {
            id: "win_rate".to_string(),
            name: "Win Rate".to_string(),
            description: "Percentage of winning trades".to_string(),
            category: MetricCategory::TradeStats,
            unit: "%".to_string(),
            decimals: 1,
            higher_is_better: true,
            warning_threshold: Some(40.0),
            critical_threshold: Some(30.0),
        });
        
        definitions.insert("avg_win_loss_ratio".to_string(), MetricDefinition {
            id: "avg_win_loss_ratio".to_string(),
            name: "Win/Loss Ratio".to_string(),
            description: "Average win size / average loss size".to_string(),
            category: MetricCategory::TradeStats,
            unit: "x".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.5),
            critical_threshold: Some(1.0),
        });
        
        definitions.insert("profit_factor".to_string(), MetricDefinition {
            id: "profit_factor".to_string(),
            name: "Profit Factor".to_string(),
            description: "Gross profit / gross loss".to_string(),
            category: MetricCategory::Performance,
            unit: "x".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.3),
            critical_threshold: Some(1.0),
        });
        
        definitions.insert("omega_ratio".to_string(), MetricDefinition {
            id: "omega_ratio".to_string(),
            name: "Omega Ratio".to_string(),
            description: "Probability-weighted gains vs losses".to_string(),
            category: MetricCategory::Advanced,
            unit: "".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.5),
            critical_threshold: Some(1.0),
        });
        
        definitions.insert("sortino_ratio".to_string(), MetricDefinition {
            id: "sortino_ratio".to_string(),
            name: "Sortino Ratio".to_string(),
            description: "Return / downside deviation".to_string(),
            category: MetricCategory::Performance,
            unit: "".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.5),
            critical_threshold: Some(0.0),
        });
        
        definitions.insert("calmar_ratio".to_string(), MetricDefinition {
            id: "calmar_ratio".to_string(),
            name: "Calmar Ratio".to_string(),
            description: "Annualized return / max drawdown".to_string(),
            category: MetricCategory::Performance,
            unit: "".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.0),
            critical_threshold: Some(0.0),
        });
        
        definitions.insert("num_trades".to_string(), MetricDefinition {
            id: "num_trades".to_string(),
            name: "Total Trades".to_string(),
            description: "Number of completed trades".to_string(),
            category: MetricCategory::TradeStats,
            unit: "".to_string(),
            decimals: 0,
            higher_is_better: false, // Directional strategies want fewer, higher quality trades
            warning_threshold: None,
            critical_threshold: Some(5.0),
        });
        
        Self {
            strategy_type: StrategyType::Custom,
            primary_metrics: vec![
                "net_profit".to_string(),
                "sharpe_ratio".to_string(),
                "max_drawdown".to_string(),
                "win_rate".to_string(),
                "avg_win_loss_ratio".to_string(),
                "trend_accuracy".to_string(),
            ],
            secondary_metrics: vec![
                "sortino_ratio".to_string(),
                "calmar_ratio".to_string(),
                "omega_ratio".to_string(),
                "profit_factor".to_string(),
                "avg_holding_period".to_string(),
                "num_trades".to_string(),
            ],
            hidden_metrics: vec![
                // Not relevant for directional strategies
                "spread_capture".to_string(),
                "inventory_turnover".to_string(),
                "fill_rate".to_string(),
                "adverse_selection".to_string(),
                "max_inventory".to_string(),
                "avg_inventory_skew".to_string(),
            ],
            metric_definitions: definitions,
        }
    }
    
    
    /// Get profile for a strategy type
    pub fn for_strategy(strategy_type: StrategyType) -> Self {
        match strategy_type {
            StrategyType::Custom => Self::directional(),
        }
    }
    
    /// Check if a metric should be shown for this strategy
    pub fn should_show_metric(&self, metric_id: &str) -> bool {
        !self.hidden_metrics.contains(&metric_id.to_string())
    }
    
    /// Check if a metric is primary (shown prominently)
    pub fn is_primary_metric(&self, metric_id: &str) -> bool {
        self.primary_metrics.contains(&metric_id.to_string())
    }
    
    /// Get metric definition if it exists
    pub fn get_metric_definition(&self, metric_id: &str) -> Option<&MetricDefinition> {
        self.metric_definitions.get(metric_id)
    }
    
    /// Get all visible metrics in order (primary first, then secondary)
    pub fn get_visible_metrics(&self) -> Vec<&String> {
        let mut metrics: Vec<&String> = self.primary_metrics.iter().collect();
        metrics.extend(self.secondary_metrics.iter());
        metrics
    }
}

/// Filtered metrics result for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredMetrics {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Primary metrics with values
    pub primary: Vec<MetricValue>,
    /// Secondary metrics with values
    pub secondary: Vec<MetricValue>,
    /// Metric definitions for rendering
    pub definitions: HashMap<String, MetricDefinition>,
}

/// Single metric with value and status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    /// Metric ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Numeric value
    pub value: f64,
    /// Formatted value string
    pub formatted: String,
    /// Status based on thresholds
    pub status: MetricStatus,
    /// Category for grouping
    pub category: MetricCategory,
}

/// Metric status for color coding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricStatus {
    /// Good value (green)
    Good,
    /// Warning level (yellow)
    Warning,
    /// Critical level (red)
    Critical,
    /// Neutral/informational (gray)
    Neutral,
}

impl FilteredMetrics {
    /// Create filtered metrics from raw values
    pub fn from_raw(
        strategy_type: StrategyType,
        raw_metrics: &HashMap<String, f64>,
    ) -> Self {
        let profile = MetricsProfile::for_strategy(strategy_type);
        
        let mut primary = Vec::new();
        let mut secondary = Vec::new();
        
        for metric_id in &profile.primary_metrics {
            if let Some(&value) = raw_metrics.get(metric_id) {
                if let Some(def) = profile.get_metric_definition(metric_id) {
                    primary.push(MetricValue::from_definition(def, value));
                }
            }
        }
        
        for metric_id in &profile.secondary_metrics {
            if let Some(&value) = raw_metrics.get(metric_id) {
                if let Some(def) = profile.get_metric_definition(metric_id) {
                    secondary.push(MetricValue::from_definition(def, value));
                }
            }
        }
        
        Self {
            strategy_type,
            primary,
            secondary,
            definitions: profile.metric_definitions,
        }
    }
}

impl MetricValue {
    /// Create from definition and value
    pub fn from_definition(def: &MetricDefinition, value: f64) -> Self {
        let status = Self::calculate_status(def, value);
        let formatted = Self::format_value(def, value);
        
        Self {
            id: def.id.clone(),
            name: def.name.clone(),
            value,
            formatted,
            status,
            category: def.category,
        }
    }
    
    /// Calculate status based on thresholds
    fn calculate_status(def: &MetricDefinition, value: f64) -> MetricStatus {
        match (def.warning_threshold, def.critical_threshold, def.higher_is_better) {
            (Some(warn), Some(crit), true) => {
                // Higher is better: below critical is bad, below warning is warning
                if value < crit { MetricStatus::Critical }
                else if value < warn { MetricStatus::Warning }
                else { MetricStatus::Good }
            }
            (Some(warn), Some(crit), false) => {
                // Lower is better: above critical is bad, above warning is warning
                if value > crit { MetricStatus::Critical }
                else if value > warn { MetricStatus::Warning }
                else { MetricStatus::Good }
            }
            _ => MetricStatus::Neutral,
        }
    }
    
    /// Format value for display
    fn format_value(def: &MetricDefinition, value: f64) -> String {
        let formatted_num = format!("{:.prec$}", value, prec = def.decimals as usize);
        
        if def.unit.is_empty() {
            formatted_num
        } else if def.unit == "%" {
            format!("{}%", formatted_num)
        } else if def.unit == "$" {
            format!("${}", formatted_num)
        } else {
            format!("{} {}", formatted_num, def.unit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_type_parsing() {
        assert_eq!(StrategyType::from_str("momentum"), StrategyType::Custom);
        assert_eq!(StrategyType::from_str("arbitrage"), StrategyType::Custom);
        assert_eq!(StrategyType::from_str("unknown"), StrategyType::Custom);
    }
    
    #[test]
    fn test_directional_profile() {
        let profile = MetricsProfile::directional();
        
        assert!(profile.is_primary_metric("trend_accuracy"));
        assert!(profile.is_primary_metric("avg_win_loss_ratio"));
        assert!(!profile.should_show_metric("inventory_turnover"));
        assert!(!profile.should_show_metric("spread_capture"));
    }
    
    #[test]
    fn test_filtered_metrics() {
        let mut raw = HashMap::new();
        raw.insert("net_profit".to_string(), 5000.0);
        raw.insert("sharpe_ratio".to_string(), 1.5);
        raw.insert("max_drawdown".to_string(), 8.0);
        
        let filtered = FilteredMetrics::from_raw(StrategyType::Custom, &raw);
        
        assert_eq!(filtered.strategy_type, StrategyType::Custom);
        assert!(!filtered.primary.is_empty());
    }
    
    #[test]
    fn test_metric_status() {
        let def = MetricDefinition {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "".to_string(),
            category: MetricCategory::Performance,
            unit: "".to_string(),
            decimals: 2,
            higher_is_better: true,
            warning_threshold: Some(1.0),
            critical_threshold: Some(0.0),
        };
        
        assert_eq!(MetricValue::calculate_status(&def, 2.0), MetricStatus::Good);
        assert_eq!(MetricValue::calculate_status(&def, 0.5), MetricStatus::Warning);
        assert_eq!(MetricValue::calculate_status(&def, -0.5), MetricStatus::Critical);
    }
}
