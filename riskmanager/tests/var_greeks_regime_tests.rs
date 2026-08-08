//! Tests for VaR, Greeks, Correlation, and Volatility Regime subsystems of RiskManager.
//!
//! Covers methods with zero existing test coverage:
//!   - update_returns / calculate_var / calculate_cvar / check_var_limits
//!   - check_greeks / check_expiry_concentration
//!   - update_correlation / update_position / get_correlation
//!   - calculate_correlated_exposure / check_correlated_exposure
//!   - update_volatility / get_regime_position_multiplier / get_current_regime
//!   - check_order_comprehensive

use riskmanager::{RiskManager, RiskLimits, RiskMetrics, RiskAction, VolatilityRegime};
use derivatives::Greeks;
use chrono::{Duration, Utc};
use std::collections::HashMap;

fn default_metrics() -> RiskMetrics {
    RiskMetrics::default()
}

// ============================================================================
// VaR SYSTEM
// ============================================================================

#[test]
fn update_returns_adds_to_history() {
    let mut rm = RiskManager::new();
    for i in 0..50 {
        rm.update_returns(0.01 * (i as f64 - 25.0));
    }
    // After 50 returns, VaR calculation should work
    let var = rm.calculate_var(100_000.0);
    assert!(var >= 0.0, "VaR should be non-negative");
}

#[test]
fn calculate_var_not_enough_data() {
    let mut rm = RiskManager::new();
    for _ in 0..20 {
        rm.update_returns(-0.01);
    }
    // < 30 data points → returns 0
    assert_eq!(rm.calculate_var(100_000.0), 0.0);
}

#[test]
fn calculate_var_with_sufficient_data() {
    let mut rm = RiskManager::new();
    // 50 returns: mix of gains and losses
    for i in 0..50 {
        let r = -0.05 + 0.002 * i as f64; // range: -0.05 to +0.048
        rm.update_returns(r);
    }
    let var = rm.calculate_var(100_000.0);
    assert!(var > 0.0, "VaR should be positive with negative tail");
}

#[test]
fn calculate_var_all_positive_returns() {
    let mut rm = RiskManager::new();
    for _ in 0..50 {
        rm.update_returns(0.01); // All positive
    }
    // 5th percentile is still positive → VaR = 0
    assert_eq!(rm.calculate_var(100_000.0), 0.0);
}

#[test]
fn calculate_cvar_not_enough_data() {
    let mut rm = RiskManager::new();
    for _ in 0..20 {
        rm.update_returns(-0.01);
    }
    assert_eq!(rm.calculate_cvar(100_000.0), 0.0);
}

#[test]
fn calculate_cvar_with_sufficient_data() {
    let mut rm = RiskManager::new();
    for i in 0..50 {
        let r = -0.05 + 0.002 * i as f64;
        rm.update_returns(r);
    }
    let cvar = rm.calculate_cvar(100_000.0);
    let var = rm.calculate_var(100_000.0);
    // CVaR >= VaR (expected shortfall is worse than the threshold)
    assert!(cvar >= var, "CVaR {cvar} should be >= VaR {var}");
}

#[test]
fn calculate_cvar_all_positive_returns() {
    let mut rm = RiskManager::new();
    for _ in 0..50 {
        rm.update_returns(0.01);
    }
    assert_eq!(rm.calculate_cvar(100_000.0), 0.0);
}

#[test]
fn update_returns_window_capped() {
    let mut rm = RiskManager::new();
    // Default window is 252; push 300 returns
    for i in 0..300 {
        rm.update_returns(-0.001 * (i % 10) as f64);
    }
    // VaR should still work — no panic from oversized vector
    let _var = rm.calculate_var(100_000.0);
}

// ============================================================================
// check_var_limits
// ============================================================================

#[test]
fn check_var_limits_proceed_when_within() {
    let rm = RiskManager::new(); // default var_limit = 5000
    let action = rm.check_var_limits(100_000.0, 3000.0, 4000.0);
    assert_eq!(action, RiskAction::Proceed);
}

#[test]
fn check_var_limits_scale_on_var_breach() {
    let rm = RiskManager::new(); // var_limit = 5000
    let action = rm.check_var_limits(100_000.0, 10_000.0, 4000.0);
    match action {
        RiskAction::ScalePosition(allowed, msg) => {
            // Scale factor = 5000/10000 = 0.5 → allowed = 50000
            assert!((allowed - 50_000.0).abs() < 1.0);
            assert!(msg.contains("VaR"));
        }
        _ => panic!("Expected ScalePosition, got {:?}", action),
    }
}

#[test]
fn check_var_limits_scale_on_cvar_breach() {
    let rm = RiskManager::new(); // cvar_limit = 7500
    // VaR within limit, CVaR exceeds
    let action = rm.check_var_limits(100_000.0, 3000.0, 15_000.0);
    match action {
        RiskAction::ScalePosition(allowed, msg) => {
            // Scale factor = 7500/15000 = 0.5
            assert!((allowed - 50_000.0).abs() < 1.0);
            assert!(msg.contains("CVaR"));
        }
        _ => panic!("Expected ScalePosition, got {:?}", action),
    }
}

#[test]
fn check_var_limits_var_checked_before_cvar() {
    let rm = RiskManager::new();
    // Both breach — VaR checked first
    let action = rm.check_var_limits(100_000.0, 10_000.0, 15_000.0);
    match action {
        RiskAction::ScalePosition(_, msg) => assert!(msg.contains("VaR")),
        _ => panic!("Expected ScalePosition"),
    }
}

#[test]
fn check_var_limits_no_limits_set() {
    let limits = RiskLimits {
        var_limit: None,
        cvar_limit: None,
        ..RiskLimits::default()
    };
    let rm = RiskManager::with_limits(limits);
    let action = rm.check_var_limits(100_000.0, 99999.0, 99999.0);
    assert_eq!(action, RiskAction::Proceed);
}

// ============================================================================
// GREEKS RISK CHECKS
// ============================================================================

#[test]
fn check_greeks_all_within_limits() {
    let rm = RiskManager::new(); // max_net_delta=10, max_gamma=5, max_vega=10000
    let greeks = Greeks { delta: 5.0, gamma: 2.0, theta: -1.0, vega: 500.0, rho: 0.0 };
    assert_eq!(rm.check_greeks(&greeks), RiskAction::Proceed);
}

#[test]
fn check_greeks_delta_breach() {
    let rm = RiskManager::new();
    let greeks = Greeks { delta: 15.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 };
    match rm.check_greeks(&greeks) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("delta")),
        other => panic!("Expected RejectOrder for delta, got {:?}", other),
    }
}

#[test]
fn check_greeks_negative_delta_breach() {
    let rm = RiskManager::new();
    let greeks = Greeks { delta: -15.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 };
    match rm.check_greeks(&greeks) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("delta")),
        other => panic!("Expected RejectOrder, got {:?}", other),
    }
}

#[test]
fn check_greeks_gamma_breach() {
    let rm = RiskManager::new();
    let greeks = Greeks { delta: 1.0, gamma: 10.0, theta: 0.0, vega: 0.0, rho: 0.0 };
    match rm.check_greeks(&greeks) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("gamma")),
        other => panic!("Expected RejectOrder for gamma, got {:?}", other),
    }
}

#[test]
fn check_greeks_vega_breach() {
    let rm = RiskManager::new();
    let greeks = Greeks { delta: 1.0, gamma: 1.0, theta: 0.0, vega: 50000.0, rho: 0.0 };
    match rm.check_greeks(&greeks) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("vega")),
        other => panic!("Expected RejectOrder for vega, got {:?}", other),
    }
}

#[test]
fn check_greeks_theta_no_limit_by_default() {
    let rm = RiskManager::new(); // max_theta = None by default
    let greeks = Greeks { delta: 1.0, gamma: 1.0, theta: -999.0, vega: 1.0, rho: 0.0 };
    assert_eq!(rm.check_greeks(&greeks), RiskAction::Proceed);
}

#[test]
fn check_greeks_theta_with_custom_limit() {
    let limits = RiskLimits {
        max_theta: Some(50.0),
        ..RiskLimits::default()
    };
    let rm = RiskManager::with_limits(limits);
    let greeks = Greeks { delta: 1.0, gamma: 1.0, theta: -100.0, vega: 1.0, rho: 0.0 };
    match rm.check_greeks(&greeks) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("theta")),
        other => panic!("Expected RejectOrder for theta, got {:?}", other),
    }
}

// ============================================================================
// CHECK EXPIRY CONCENTRATION
// ============================================================================

#[test]
fn check_expiry_concentration_within_limit() {
    let rm = RiskManager::new(); // max_notional_per_expiry = 100000
    let mut expiry_map = HashMap::new();
    expiry_map.insert(Utc::now() + Duration::days(30), 50_000.0);
    assert_eq!(rm.check_expiry_concentration(&expiry_map), RiskAction::Proceed);
}

#[test]
fn check_expiry_concentration_exceeds_limit() {
    let rm = RiskManager::new();
    let mut expiry_map = HashMap::new();
    expiry_map.insert(Utc::now() + Duration::days(30), 200_000.0);
    match rm.check_expiry_concentration(&expiry_map) {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("Notional")),
        other => panic!("Expected RejectOrder, got {:?}", other),
    }
}

#[test]
fn check_expiry_concentration_no_limit_set() {
    let limits = RiskLimits {
        max_notional_per_expiry: None,
        ..RiskLimits::default()
    };
    let rm = RiskManager::with_limits(limits);
    let mut expiry_map = HashMap::new();
    expiry_map.insert(Utc::now(), 999_999.0);
    assert_eq!(rm.check_expiry_concentration(&expiry_map), RiskAction::Proceed);
}

#[test]
fn check_expiry_concentration_multiple_buckets() {
    let rm = RiskManager::new();
    let mut expiry_map = HashMap::new();
    expiry_map.insert(Utc::now() + Duration::days(30), 90_000.0); // OK
    expiry_map.insert(Utc::now() + Duration::days(60), 150_000.0); // Exceeds
    match rm.check_expiry_concentration(&expiry_map) {
        RiskAction::RejectOrder(_) => {}
        other => panic!("Expected RejectOrder, got {:?}", other),
    }
}

// ============================================================================
// CORRELATION RISK MANAGEMENT
// ============================================================================

#[test]
fn update_and_get_correlation() {
    let mut rm = RiskManager::new();
    rm.update_correlation("BTC", "ETH", 0.85);
    assert_eq!(rm.get_correlation("BTC", "ETH"), Some(0.85));
    assert_eq!(rm.get_correlation("ETH", "BTC"), Some(0.85)); // Symmetric
}

#[test]
fn get_correlation_missing() {
    let rm = RiskManager::new();
    assert_eq!(rm.get_correlation("BTC", "ETH"), None);
}

#[test]
fn update_correlation_replaces_existing() {
    let mut rm = RiskManager::new();
    rm.update_correlation("BTC", "ETH", 0.5);
    rm.update_correlation("BTC", "ETH", 0.9);
    assert_eq!(rm.get_correlation("BTC", "ETH"), Some(0.9));
}

#[test]
fn update_position_and_calculate_correlated_exposure() {
    let mut rm = RiskManager::new(); // correlation_threshold = 0.7
    rm.update_correlation("BTC", "ETH", 0.85);
    rm.update_position("BTC", 10_000.0);
    rm.update_position("ETH", 5_000.0);
    
    let exposure = rm.calculate_correlated_exposure("BTC");
    // BTC own position (10000) + ETH (5000, corr > 0.7) = 15000
    assert!((exposure - 15_000.0).abs() < 1.0);
}

#[test]
fn calculate_correlated_exposure_below_threshold() {
    let mut rm = RiskManager::new();
    rm.update_correlation("BTC", "SOL", 0.3); // Below 0.7 threshold
    rm.update_position("BTC", 10_000.0);
    rm.update_position("SOL", 5_000.0);
    
    let exposure = rm.calculate_correlated_exposure("BTC");
    // Only own position — SOL is not "correlated" (0.3 < 0.7)
    assert!((exposure - 10_000.0).abs() < 1.0);
}

#[test]
fn calculate_correlated_exposure_no_positions() {
    let rm = RiskManager::new();
    assert_eq!(rm.calculate_correlated_exposure("BTC"), 0.0);
}

#[test]
fn check_correlated_exposure_within_limit() {
    let mut rm = RiskManager::new(); // max_correlated_exposure = 50000
    rm.update_correlation("BTC", "ETH", 0.9);
    rm.update_position("BTC", 10_000.0);
    rm.update_position("ETH", 5_000.0);
    
    let action = rm.check_correlated_exposure("BTC", 10_000.0);
    assert_eq!(action, RiskAction::Proceed);
}

#[test]
fn check_correlated_exposure_exceeds_limit() {
    let mut rm = RiskManager::new();
    rm.update_correlation("BTC", "ETH", 0.9);
    rm.update_position("BTC", 30_000.0);
    rm.update_position("ETH", 20_000.0);
    
    // Current exposure = 30k + 20k = 50k (at limit)
    // Adding 10k → 60k > 50k → ScalePosition
    let action = rm.check_correlated_exposure("BTC", 10_000.0);
    match action {
        RiskAction::RejectOrder(_) | RiskAction::ScalePosition(_, _) => {}
        other => panic!("Expected rejection or scaling, got {:?}", other),
    }
}

#[test]
fn check_correlated_exposure_no_limit() {
    let limits = RiskLimits {
        max_correlated_exposure: None,
        ..RiskLimits::default()
    };
    let rm = RiskManager::with_limits(limits);
    assert_eq!(rm.check_correlated_exposure("BTC", 999_999.0), RiskAction::Proceed);
}

// ============================================================================
// VOLATILITY REGIME
// ============================================================================

#[test]
fn default_regime_is_normal() {
    let rm = RiskManager::new();
    assert_eq!(rm.get_current_regime(), VolatilityRegime::Normal);
}

#[test]
fn regime_multiplier_normal() {
    let rm = RiskManager::new();
    assert!((rm.get_regime_position_multiplier() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn update_volatility_builds_baseline() {
    let mut rm = RiskManager::new();
    // Need 20+ observations for baseline
    for _ in 0..25 {
        rm.update_volatility(0.02); // Normal vol
    }
    assert_eq!(rm.get_current_regime(), VolatilityRegime::Normal);
}

#[test]
fn update_volatility_detects_high_regime() {
    let mut rm = RiskManager::new();
    // Build baseline at 0.02
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    // Spike to 0.04 → ratio 0.04/0.02 = 2.0 → above high_threshold 1.5 but below extreme 2.5
    rm.update_volatility(0.04);
    assert_eq!(rm.get_current_regime(), VolatilityRegime::High);
}

#[test]
fn update_volatility_detects_extreme_regime() {
    let mut rm = RiskManager::new();
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    // Spike to 0.06 → ratio 0.06/0.02 = 3.0 → above extreme 2.5
    rm.update_volatility(0.06);
    assert_eq!(rm.get_current_regime(), VolatilityRegime::Extreme);
}

#[test]
fn update_volatility_detects_low_regime() {
    let mut rm = RiskManager::new();
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    // Drop to 0.01 → ratio = 0.01/0.02 = 0.5 → below 0.7 threshold → Low
    rm.update_volatility(0.01);
    assert_eq!(rm.get_current_regime(), VolatilityRegime::Low);
}

#[test]
fn regime_multiplier_high() {
    let mut rm = RiskManager::new();
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    rm.update_volatility(0.04);
    // High regime → multiplier = 0.5 (default high_vol_position_mult)
    assert!((rm.get_regime_position_multiplier() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn regime_multiplier_extreme() {
    let mut rm = RiskManager::new();
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    rm.update_volatility(0.06);
    // Extreme regime → multiplier = 0.2 (default)
    assert!((rm.get_regime_position_multiplier() - 0.2).abs() < f64::EPSILON);
}

#[test]
fn regime_multiplier_low() {
    let mut rm = RiskManager::new();
    for _ in 0..25 {
        rm.update_volatility(0.02);
    }
    rm.update_volatility(0.01);
    assert!((rm.get_regime_position_multiplier() - 1.2).abs() < f64::EPSILON);
}

#[test]
fn volatility_history_capped_at_100() {
    let mut rm = RiskManager::new();
    for i in 0..150 {
        rm.update_volatility(0.01 + 0.001 * (i as f64 % 5.0));
    }
    // Should not panic; regime should be valid
    let regime = rm.get_current_regime();
    assert!(matches!(regime, VolatilityRegime::Low | VolatilityRegime::Normal | VolatilityRegime::High | VolatilityRegime::Extreme));
}

// ============================================================================
// check_order_comprehensive
// ============================================================================

#[test]
fn comprehensive_check_all_pass() {
    let rm = RiskManager::new();
    let metrics = default_metrics();
    let action = rm.check_order_comprehensive(0.5, 0.0, &metrics, "BTC", 10_000.0);
    assert_eq!(action, RiskAction::Proceed);
}

#[test]
fn comprehensive_check_basic_failure_first() {
    let rm = RiskManager::new(); // max_order_size = 1.0
    let metrics = default_metrics();
    // Order size 5.0 > max 1.0 → ReducePosition from basic check
    let action = rm.check_order_comprehensive(5.0, 0.0, &metrics, "BTC", 10_000.0);
    match action {
        RiskAction::ReducePosition(_) => {}
        other => panic!("Expected ReducePosition from basic check, got {:?}", other),
    }
}

#[test]
fn comprehensive_check_var_failure() {
    let rm = RiskManager::new(); // var_limit = 5000
    let mut metrics = default_metrics();
    metrics.current_var = 10_000.0; // Exceeds var_limit
    let action = rm.check_order_comprehensive(0.5, 0.0, &metrics, "BTC", 100_000.0);
    match action {
        RiskAction::ScalePosition(_, msg) => assert!(msg.contains("VaR")),
        other => panic!("Expected ScalePosition for VaR, got {:?}", other),
    }
}

#[test]
fn comprehensive_check_greeks_failure() {
    let rm = RiskManager::new(); // max_net_delta = 10
    let mut metrics = default_metrics();
    metrics.portfolio_greeks = Some(Greeks { 
        delta: 20.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 
    });
    let action = rm.check_order_comprehensive(0.5, 0.0, &metrics, "BTC", 10_000.0);
    match action {
        RiskAction::RejectOrder(msg) => assert!(msg.contains("delta")),
        other => panic!("Expected RejectOrder for delta, got {:?}", other),
    }
}

// ============================================================================
// VOLATILITY REGIME ENUM
// ============================================================================

#[test]
fn volatility_regime_default() {
    assert_eq!(VolatilityRegime::default(), VolatilityRegime::Normal);
}

#[test]
fn volatility_regime_serialization() {
    let json = serde_json::to_string(&VolatilityRegime::Extreme).unwrap();
    let deserialized: VolatilityRegime = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, VolatilityRegime::Extreme);
}

// ============================================================================
// set_var_from_monte_carlo
// ============================================================================

#[test]
fn set_var_from_monte_carlo_clears_returns() {
    let mut rm = RiskManager::new();
    for _ in 0..50 {
        rm.update_returns(-0.01);
    }
    rm.set_var_from_monte_carlo(5000.0, 7500.0);
    // After MC set, historical returns are cleared, so VaR = 0
    assert_eq!(rm.calculate_var(100_000.0), 0.0);
}
