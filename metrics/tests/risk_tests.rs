//! Comprehensive risk metrics tests
//! Covers sortino, calmar, drawdown variants, ulcer/pain indices

use metrics::risk::{
    sortino_ratio, calmar_ratio, current_drawdown, average_drawdown,
    ulcer_index, ulcer_performance_index, pain_index, downside_deviation,
    max_drawdown, volatility,
};

// ============================================================================
// SORTINO RATIO
// ============================================================================

#[test]
fn test_sortino_positive_returns_no_downside() {
    // All positive returns => no downside deviation => capped at 99.99
    let returns = vec![0.01, 0.02, 0.015, 0.018, 0.012];
    let result = sortino_ratio(&returns, 0.0).unwrap();
    assert!((result - 99.99).abs() < 0.01, "No downside should cap at 99.99");
}

#[test]
fn test_sortino_mixed_returns() {
    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008];
    let result = sortino_ratio(&returns, 0.02).unwrap();
    assert!(result.is_finite(), "Sortino should be finite for mixed returns");
}

#[test]
fn test_sortino_negative_returns() {
    let returns = vec![-0.01, -0.02, -0.015, -0.018, -0.012];
    let result = sortino_ratio(&returns, 0.0).unwrap();
    assert!(result < 0.0, "Sortino should be negative for losing strategies");
}

#[test]
fn test_sortino_empty() {
    let returns: Vec<f64> = vec![];
    assert!(sortino_ratio(&returns, 0.0).is_none());
}

#[test]
fn test_sortino_annualization() {
    // returns with known values to verify annualization factor of sqrt(365)
    let returns = vec![0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.006, -0.002,
                       0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.006, -0.002];
    let result = sortino_ratio(&returns, 0.0).unwrap();
    // Should be annualized (multiplied by sqrt(365))
    assert!(result.abs() > 1.0, "Annualized sortino should be meaningfully different from daily");
}

// ============================================================================
// CALMAR RATIO
// ============================================================================

#[test]
fn test_calmar_positive() {
    let result = calmar_ratio(0.25, 0.10).unwrap();
    assert!((result - 2.5).abs() < 0.001);
}

#[test]
fn test_calmar_negative_return() {
    let result = calmar_ratio(-0.10, 0.20).unwrap();
    assert!((result - (-0.5)).abs() < 0.001);
}

#[test]
fn test_calmar_zero_drawdown_positive_return() {
    let result = calmar_ratio(0.15, 0.0).unwrap();
    assert!((result - 99.99).abs() < 0.01, "Zero drawdown with positive return caps at 99.99");
}

#[test]
fn test_calmar_zero_drawdown_zero_return() {
    assert!(calmar_ratio(0.0, 0.0).is_none());
}

#[test]
fn test_calmar_zero_drawdown_negative_return() {
    assert!(calmar_ratio(-0.05, 0.0).is_none());
}

// ============================================================================
// CURRENT DRAWDOWN
// ============================================================================

#[test]
fn test_current_drawdown_at_peak() {
    let curve = vec![100.0, 110.0, 120.0, 130.0];
    let dd = current_drawdown(&curve).unwrap();
    assert!((dd - 0.0).abs() < 0.001, "At peak => 0 drawdown");
}

#[test]
fn test_current_drawdown_in_drawdown() {
    let curve = vec![100.0, 120.0, 90.0];
    let dd = current_drawdown(&curve).unwrap();
    assert!((dd - 0.25).abs() < 0.001, "90 vs peak 120 => 25% drawdown");
}

#[test]
fn test_current_drawdown_empty() {
    let curve: Vec<f64> = vec![];
    assert!(current_drawdown(&curve).is_none());
}

#[test]
fn test_current_drawdown_single() {
    let curve = vec![100.0];
    let dd = current_drawdown(&curve).unwrap();
    assert!((dd - 0.0).abs() < 0.001);
}

// ============================================================================
// AVERAGE DRAWDOWN
// ============================================================================

#[test]
fn test_average_drawdown_monotonic_increase() {
    let curve = vec![100.0, 110.0, 120.0, 130.0];
    let dd = average_drawdown(&curve).unwrap();
    assert!((dd - 0.0).abs() < 0.001, "No drawdown on monotonic increase");
}

#[test]
fn test_average_drawdown_with_dip() {
    let curve = vec![100.0, 90.0, 100.0];
    let dd = average_drawdown(&curve).unwrap();
    assert!(dd > 0.0, "Should have positive average drawdown");
    assert!(dd < 0.10, "Average drawdown should be moderate");
}

#[test]
fn test_average_drawdown_empty() {
    let curve: Vec<f64> = vec![];
    assert!(average_drawdown(&curve).is_none());
}

// ============================================================================
// DOWNSIDE DEVIATION
// ============================================================================

#[test]
fn test_downside_deviation_all_above_target() {
    let returns = vec![0.05, 0.10, 0.03, 0.08];
    let dd = downside_deviation(&returns, 0.0).unwrap();
    assert!((dd - 0.0).abs() < 0.001, "No returns below target => 0 downside deviation");
}

#[test]
fn test_downside_deviation_mixed() {
    let returns = vec![0.05, -0.03, 0.02, -0.06, 0.01];
    let dd = downside_deviation(&returns, 0.0).unwrap();
    assert!(dd > 0.0, "Should have positive downside deviation");
}

#[test]
fn test_downside_deviation_empty() {
    let returns: Vec<f64> = vec![];
    assert!(downside_deviation(&returns, 0.0).is_none());
}

#[test]
fn test_downside_deviation_with_nonzero_target() {
    let returns = vec![0.01, 0.02, 0.005, 0.015, 0.008];
    // With target 0.03, all returns are below target
    let dd = downside_deviation(&returns, 0.03).unwrap();
    assert!(dd > 0.0, "All returns below target should give positive downside deviation");
}

// ============================================================================
// ULCER INDEX
// ============================================================================

#[test]
fn test_ulcer_index_no_drawdown() {
    let curve = vec![100.0, 110.0, 120.0, 130.0, 140.0];
    let ui = ulcer_index(&curve).unwrap();
    assert!((ui - 0.0).abs() < 0.001, "No drawdowns => 0 ulcer index");
}

#[test]
fn test_ulcer_index_steady_decline() {
    let curve = vec![100.0, 90.0, 80.0, 70.0, 60.0];
    let ui = ulcer_index(&curve).unwrap();
    assert!(ui > 0.0, "Declining curve should have positive ulcer index");
}

#[test]
fn test_ulcer_index_v_shaped() {
    let curve = vec![100.0, 90.0, 80.0, 90.0, 100.0];
    let ui = ulcer_index(&curve).unwrap();
    assert!(ui > 0.0, "V-shape recovery still has ulcer index > 0");
}

#[test]
fn test_ulcer_index_empty() {
    let curve: Vec<f64> = vec![];
    assert!(ulcer_index(&curve).is_none());
}

// ============================================================================
// ULCER PERFORMANCE INDEX
// ============================================================================

#[test]
fn test_upi_profitable_strategy() {
    let curve = vec![100.0, 105.0, 103.0, 110.0, 108.0, 115.0];
    let upi = ulcer_performance_index(&curve, 0.02).unwrap();
    assert!(upi > 0.0, "Profitable strategy should have positive UPI");
}

#[test]
fn test_upi_no_drawdown_profitable() {
    let curve = vec![100.0, 110.0, 120.0, 130.0];
    let upi = ulcer_performance_index(&curve, 0.0).unwrap();
    assert!((upi - 99.99).abs() < 0.01, "No drawdown + profit caps at 99.99");
}

#[test]
fn test_upi_single_point() {
    let curve = vec![100.0];
    assert!(ulcer_performance_index(&curve, 0.0).is_none());
}

#[test]
fn test_upi_losing_strategy() {
    let curve = vec![100.0, 95.0, 90.0, 85.0];
    let upi = ulcer_performance_index(&curve, 0.0).unwrap();
    assert!(upi < 0.0, "Losing strategy should have negative UPI");
}

// ============================================================================
// PAIN INDEX
// ============================================================================

#[test]
fn test_pain_index_no_drawdown() {
    let curve = vec![100.0, 110.0, 120.0, 130.0];
    let pi = pain_index(&curve).unwrap();
    assert!((pi - 0.0).abs() < 0.001);
}

#[test]
fn test_pain_index_with_drawdown() {
    let curve = vec![100.0, 90.0, 80.0, 90.0, 100.0];
    let pi = pain_index(&curve).unwrap();
    assert!(pi > 0.0, "Pain index should be positive with drawdowns");
}

#[test]
fn test_pain_index_empty() {
    let curve: Vec<f64> = vec![];
    assert!(pain_index(&curve).is_none());
}

#[test]
fn test_pain_index_less_than_ulcer_index() {
    // Pain index (arithmetic mean) should generally be <= Ulcer index (RMS) 
    // because RMS >= mean for any distribution
    let curve = vec![100.0, 95.0, 90.0, 92.0, 88.0, 94.0, 96.0, 91.0];
    let pi = pain_index(&curve).unwrap();
    let ui = ulcer_index(&curve).unwrap();
    assert!(pi <= ui + 0.001, "Pain index should be <= ulcer index (RMS >= mean)");
}

// ============================================================================
// CROSS-METRIC CONSISTENCY
// ============================================================================

#[test]
fn test_max_drawdown_current_drawdown_consistency() {
    let curve = vec![100.0, 120.0, 90.0, 110.0, 80.0];
    let max_dd = max_drawdown(&curve).unwrap();
    let cur_dd = current_drawdown(&curve).unwrap();
    assert!(cur_dd <= max_dd + 0.001, "Current drawdown should be <= max drawdown");
}

#[test]
fn test_average_drawdown_less_than_max() {
    let curve = vec![100.0, 110.0, 95.0, 105.0, 90.0, 100.0];
    let avg_dd = average_drawdown(&curve).unwrap();
    let max_dd = max_drawdown(&curve).unwrap();
    assert!(avg_dd <= max_dd + 0.001, "Average drawdown should be <= max drawdown");
}
