//! Integration tests for strategy_metrics module
//! Covers FilteredMetrics::from_raw edge cases, MetricValue formatting, and MetricsProfile methods

use std::collections::HashMap;
use metrics::strategy_metrics::{
    StrategyType, MetricsProfile, FilteredMetrics, MetricValue, MetricStatus,
    MetricCategory, MetricDefinition,
};

// ============================================================================
// STRATEGY TYPE
// ============================================================================

#[test]
fn test_strategy_type_from_str_custom_fallback() {
    let custom_variants = ["momentum", "trend_following", "arbitrage", "unknown", "", "market_making", "mm"];
    for s in &custom_variants {
        assert_eq!(StrategyType::from_str(s), StrategyType::Custom, "Failed for: {}", s);
    }
}

#[test]
fn test_strategy_type_default() {
    assert_eq!(StrategyType::default(), StrategyType::Custom);
}

// ============================================================================
// METRICS PROFILE
// ============================================================================

#[test]
fn test_for_strategy_routes_correctly() {
    let dir = MetricsProfile::for_strategy(StrategyType::Custom);
    assert_eq!(dir.strategy_type, StrategyType::Custom);
}

#[test]
fn test_directional_hides_mm_metrics() {
    let profile = MetricsProfile::directional();
    assert!(!profile.should_show_metric("spread_capture"));
    assert!(!profile.should_show_metric("inventory_turnover"));
    assert!(!profile.should_show_metric("fill_rate"));
    assert!(!profile.should_show_metric("adverse_selection"));
    assert!(!profile.should_show_metric("max_inventory"));
    assert!(!profile.should_show_metric("avg_inventory_skew"));
}

#[test]
fn test_get_visible_metrics_order() {
    let profile = MetricsProfile::directional();
    let visible = profile.get_visible_metrics();
    
    // Primary metrics should come first
    assert!(visible.len() >= profile.primary_metrics.len());
    for (i, id) in profile.primary_metrics.iter().enumerate() {
        assert_eq!(visible[i], id);
    }
}

#[test]
fn test_get_metric_definition_exists() {
    let profile = MetricsProfile::directional();
    let def = profile.get_metric_definition("net_profit").unwrap();
    assert_eq!(def.name, "Net P&L");
    assert_eq!(def.unit, "$");
    assert!(def.higher_is_better);
}

#[test]
fn test_get_metric_definition_nonexistent() {
    let profile = MetricsProfile::directional();
    assert!(profile.get_metric_definition("nonexistent_metric").is_none());
}

#[test]
fn test_is_primary_vs_secondary() {
    let profile = MetricsProfile::directional();
    assert!(profile.is_primary_metric("net_profit"));
    assert!(!profile.is_primary_metric("sortino_ratio")); // Secondary
}

// ============================================================================
// FILTERED METRICS
// ============================================================================

#[test]
fn test_filtered_metrics_from_raw_full() {
    let mut raw = HashMap::new();
    raw.insert("net_profit".to_string(), 5000.0);
    raw.insert("sharpe_ratio".to_string(), 1.5);
    raw.insert("max_drawdown".to_string(), 8.0);
    raw.insert("profit_factor".to_string(), 2.3);
    raw.insert("commission_ratio".to_string(), 15.0);
    raw.insert("inventory_turnover".to_string(), 5.0);
    raw.insert("sortino_ratio".to_string(), 2.1);
    raw.insert("win_rate".to_string(), 55.0);
    
    let filtered = FilteredMetrics::from_raw(StrategyType::Custom, &raw);
    assert_eq!(filtered.strategy_type, StrategyType::Custom);
    assert!(!filtered.primary.is_empty());
    assert!(!filtered.secondary.is_empty());
    assert!(!filtered.definitions.is_empty());
}

#[test]
fn test_filtered_metrics_empty_raw() {
    let raw = HashMap::new();
    let filtered = FilteredMetrics::from_raw(StrategyType::Custom, &raw);
    assert!(filtered.primary.is_empty());
    assert!(filtered.secondary.is_empty());
}

#[test]
fn test_filtered_metrics_partial_raw() {
    let mut raw = HashMap::new();
    raw.insert("net_profit".to_string(), 1000.0);
    // Only one metric provided
    
    let filtered = FilteredMetrics::from_raw(StrategyType::Custom, &raw);
    assert_eq!(filtered.primary.len(), 1);
    assert_eq!(filtered.primary[0].id, "net_profit");
}

#[test]
fn test_filtered_metrics_ignores_unknown_keys() {
    let mut raw = HashMap::new();
    raw.insert("unknown_metric_xyz".to_string(), 42.0);
    
    let filtered = FilteredMetrics::from_raw(StrategyType::Custom, &raw);
    // Unknown metrics should not appear in primary or secondary
    assert!(filtered.primary.is_empty());
    assert!(filtered.secondary.is_empty());
}

// ============================================================================
// METRIC VALUE
// ============================================================================

#[test]
fn test_metric_value_format_dollar() {
    let def = MetricDefinition {
        id: "net_profit".to_string(),
        name: "Net P&L".to_string(),
        description: "".to_string(),
        category: MetricCategory::Performance,
        unit: "$".to_string(),
        decimals: 2,
        higher_is_better: true,
        warning_threshold: None,
        critical_threshold: None,
    };
    let mv = MetricValue::from_definition(&def, 1234.56);
    assert_eq!(mv.formatted, "$1234.56");
}

#[test]
fn test_metric_value_format_percent() {
    let def = MetricDefinition {
        id: "win_rate".to_string(),
        name: "Win Rate".to_string(),
        description: "".to_string(),
        category: MetricCategory::TradeStats,
        unit: "%".to_string(),
        decimals: 1,
        higher_is_better: true,
        warning_threshold: None,
        critical_threshold: None,
    };
    let mv = MetricValue::from_definition(&def, 55.5);
    assert_eq!(mv.formatted, "55.5%");
}

#[test]
fn test_metric_value_format_no_unit() {
    let def = MetricDefinition {
        id: "sharpe_ratio".to_string(),
        name: "Sharpe".to_string(),
        description: "".to_string(),
        category: MetricCategory::Performance,
        unit: "".to_string(),
        decimals: 2,
        higher_is_better: true,
        warning_threshold: None,
        critical_threshold: None,
    };
    let mv = MetricValue::from_definition(&def, 1.25);
    assert_eq!(mv.formatted, "1.25");
}

#[test]
fn test_metric_value_format_custom_unit() {
    let def = MetricDefinition {
        id: "turnover".to_string(),
        name: "Turnover".to_string(),
        description: "".to_string(),
        category: MetricCategory::Inventory,
        unit: "x/day".to_string(),
        decimals: 1,
        higher_is_better: true,
        warning_threshold: None,
        critical_threshold: None,
    };
    let mv = MetricValue::from_definition(&def, 3.5);
    assert_eq!(mv.formatted, "3.5 x/day");
}

#[test]
fn test_metric_status_higher_is_better() {
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
    
    let good = MetricValue::from_definition(&def, 2.0);
    assert_eq!(good.status, MetricStatus::Good);
    
    let warning = MetricValue::from_definition(&def, 0.5);
    assert_eq!(warning.status, MetricStatus::Warning);
    
    let critical = MetricValue::from_definition(&def, -0.5);
    assert_eq!(critical.status, MetricStatus::Critical);
}

#[test]
fn test_metric_status_lower_is_better() {
    let def = MetricDefinition {
        id: "test".to_string(),
        name: "Test".to_string(),
        description: "".to_string(),
        category: MetricCategory::Risk,
        unit: "%".to_string(),
        decimals: 2,
        higher_is_better: false,
        warning_threshold: Some(10.0),
        critical_threshold: Some(20.0),
    };
    
    let good = MetricValue::from_definition(&def, 5.0);
    assert_eq!(good.status, MetricStatus::Good);
    
    let warning = MetricValue::from_definition(&def, 15.0);
    assert_eq!(warning.status, MetricStatus::Warning);
    
    let critical = MetricValue::from_definition(&def, 25.0);
    assert_eq!(critical.status, MetricStatus::Critical);
}

#[test]
fn test_metric_status_no_thresholds() {
    let def = MetricDefinition {
        id: "test".to_string(),
        name: "Test".to_string(),
        description: "".to_string(),
        category: MetricCategory::TradeStats,
        unit: "".to_string(),
        decimals: 0,
        higher_is_better: true,
        warning_threshold: None,
        critical_threshold: None,
    };
    
    let mv = MetricValue::from_definition(&def, 100.0);
    assert_eq!(mv.status, MetricStatus::Neutral);
}
