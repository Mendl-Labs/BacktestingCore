//! Tests for performance metrics: sharpe_ratio_from_equity_curve and PerformanceMetrics::from_trade_data

use metrics::{
    sharpe_ratio_from_equity_curve, PerformanceMetrics,
};

// ============================================================================
// SHARPE RATIO FROM EQUITY CURVE
// ============================================================================

#[test]
fn test_sharpe_from_equity_curve_basic() {
    // Steadily increasing equity curve
    let equity = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 
                      106.0, 107.0, 108.0, 109.0, 110.0];
    let sharpe = sharpe_ratio_from_equity_curve(&equity, 0.02, 1).unwrap();
    assert!(sharpe > 0.0, "Increasing equity => positive Sharpe");
}

#[test]
fn test_sharpe_from_equity_curve_declining() {
    let equity = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0];
    let sharpe = sharpe_ratio_from_equity_curve(&equity, 0.02, 1).unwrap();
    assert!(sharpe < 0.0, "Declining equity => negative Sharpe");
}

#[test]
fn test_sharpe_from_equity_curve_single_point() {
    let equity = vec![100.0];
    assert!(sharpe_ratio_from_equity_curve(&equity, 0.02, 1).is_none());
}

#[test]
fn test_sharpe_from_equity_curve_two_points() {
    let equity = vec![100.0, 110.0];
    // Only 1 return, std_dev undefined for single return
    let result = sharpe_ratio_from_equity_curve(&equity, 0.0, 1);
    // Should return None since we can't compute std_dev with 1 data point
    assert!(result.is_none());
}

#[test]
fn test_sharpe_from_equity_curve_zero_start() {
    // Zero starting value should filter out zero-division
    let equity = vec![0.0, 100.0, 110.0];
    let result = sharpe_ratio_from_equity_curve(&equity, 0.0, 1);
    // First return is skipped (dividing by 0), so we have one return from 100->110
    // One return can't compute std_dev, should be None
    assert!(result.is_none());
}

#[test]
fn test_sharpe_from_equity_curve_intraday_aggregation() {
    // 10 "intraday" points representing 2 days of 5 points each
    let equity = vec![100.0, 100.2, 100.4, 100.6, 100.8, 
                      101.0, 101.2, 101.4, 101.6, 101.8, 102.0];
    let sharpe = sharpe_ratio_from_equity_curve(&equity, 0.0, 5).unwrap();
    assert!(sharpe > 0.0, "Profitable strategy with intraday aggregation => positive Sharpe");
}

#[test]
fn test_sharpe_from_equity_curve_flat() {
    // Flat equity curve => zero returns => zero std_dev => None
    let equity = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let result = sharpe_ratio_from_equity_curve(&equity, 0.0, 1);
    assert!(result.is_none(), "Flat curve => zero volatility => None");
}

// ============================================================================
// PERFORMANCE METRICS STRUCT
// ============================================================================

#[test]
fn test_performance_metrics_from_trade_data() {
    let trade_returns = vec![10.0, -5.0, 15.0, -3.0, 8.0, -2.0, 12.0, -4.0];
    let equity_curve = vec![100.0, 110.0, 105.0, 120.0, 117.0, 125.0, 123.0, 135.0, 131.0];
    let trade_durations = vec![3600.0, 7200.0, 1800.0, 5400.0, 2700.0, 4500.0, 3000.0, 6000.0];
    
    let metrics = PerformanceMetrics::from_trade_data(
        &trade_returns,
        &equity_curve,
        &trade_durations,
        0.02,
    );
    
    assert_eq!(metrics.number_of_trades, 8);
    assert!((metrics.net_profit - 31.0).abs() < 0.001);
    assert!((metrics.gross_profit - 45.0).abs() < 0.001);
    assert!((metrics.gross_loss - (-14.0)).abs() < 0.001);
    assert!(metrics.win_rate > 0.0 && metrics.win_rate < 1.0);
    assert!(metrics.max_drawdown >= 0.0);
    assert!(metrics.volatility >= 0.0);
    assert!(metrics.average_trade_duration > 0.0);
}

#[test]
fn test_performance_metrics_empty_data() {
    let metrics = PerformanceMetrics::from_trade_data(&[], &[], &[], 0.02);
    
    assert_eq!(metrics.number_of_trades, 0);
    assert!((metrics.net_profit - 0.0).abs() < 0.001);
    assert!((metrics.sharpe_ratio - 0.0).abs() < 0.001); // unwrap_or(0.0)
    assert!((metrics.win_rate - 0.0).abs() < 0.001);
}

#[test]
fn test_performance_metrics_default_costs() {
    let metrics = PerformanceMetrics::from_trade_data(
        &[10.0, -5.0],
        &[100.0, 110.0, 105.0],
        &[3600.0, 7200.0],
        0.0,
    );
    
    assert!((metrics.total_commission - 0.0).abs() < 0.001);
    assert!((metrics.total_slippage - 0.0).abs() < 0.001);
    assert!((metrics.total_market_impact - 0.0).abs() < 0.001);
    assert!((metrics.transaction_costs_percentage - 0.0).abs() < 0.001);
}

#[test]
fn test_performance_metrics_serialization() {
    let metrics = PerformanceMetrics::from_trade_data(
        &[10.0, -5.0, 15.0],
        &[100.0, 110.0, 105.0, 120.0],
        &[3600.0, 7200.0, 1800.0],
        0.02,
    );
    
    // Should be serializable
    let json = serde_json::to_string(&metrics).unwrap();
    assert!(json.contains("sharpe_ratio"));
    assert!(json.contains("net_profit"));
    
    // Should be deserializable
    let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();
    assert!((deserialized.net_profit - metrics.net_profit).abs() < 0.001);
}
