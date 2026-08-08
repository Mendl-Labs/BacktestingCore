//! Advanced metrics tests: Omega ratio suite, tail analysis, Kelly criterion, CrossAssetMonitor

use metrics::advanced::{
    omega_ratio, omega_ratio_suite, tail_ratio, tail_analysis,
    kelly_criterion, KellyAssessment, CrossAssetMonitor,
};

// ============================================================================
// OMEGA RATIO
// ============================================================================

#[test]
fn test_omega_ratio_all_positive() {
    let returns = vec![0.01, 0.02, 0.03, 0.015, 0.025];
    let omega = omega_ratio(&returns, 0.0).unwrap();
    assert!(omega > 10.0, "All positive returns => very high omega: got {}", omega);
}

#[test]
fn test_omega_ratio_all_negative() {
    let returns = vec![-0.01, -0.02, -0.03, -0.015];
    let omega = omega_ratio(&returns, 0.0).unwrap();
    assert!(omega < 0.001, "All negative returns => near-zero omega: got {}", omega);
}

#[test]
fn test_omega_ratio_balanced() {
    let returns = vec![0.02, -0.02, 0.02, -0.02, 0.02, -0.02];
    let omega = omega_ratio(&returns, 0.0).unwrap();
    assert!((omega - 1.0).abs() < 0.01, "Balanced returns => omega ~1.0: got {}", omega);
}

#[test]
fn test_omega_ratio_empty() {
    assert!(omega_ratio(&[], 0.0).is_none());
}

#[test]
fn test_omega_ratio_with_threshold() {
    let returns = vec![0.01, 0.02, 0.03, 0.04, 0.05];
    // Threshold at 0.03 — only 0.04 and 0.05 are gains; 0.01,0.02 are losses
    let omega = omega_ratio(&returns, 0.03).unwrap();
    assert!(omega > 0.0 && omega.is_finite(), "Omega with threshold should be finite");
}

// ============================================================================
// OMEGA RATIO SUITE
// ============================================================================

#[test]
fn test_omega_suite_returns_all_three() {
    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008,
                       0.012, -0.003, 0.008, -0.007, 0.018, -0.004, 0.009];
    let suite = omega_ratio_suite(&returns, 0.02);
    assert!(suite.omega_at_zero.is_some());
    assert!(suite.omega_at_rfr.is_some());
    assert!(suite.omega_at_median.is_some());
    assert!((suite.threshold_zero - 0.0).abs() < f64::EPSILON);
    assert!((suite.threshold_rfr - 0.02).abs() < f64::EPSILON);
}

#[test]
fn test_omega_suite_empty() {
    let suite = omega_ratio_suite(&[], 0.02);
    assert!(suite.omega_at_zero.is_none());
    assert!(suite.omega_at_rfr.is_none());
}

#[test]
fn test_omega_at_zero_greater_than_at_rfr() {
    // For positive mean returns, omega at 0 should be > omega at rfr (higher threshold = lower omega)
    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008,
                       0.012, -0.003, 0.008, -0.007, 0.018, -0.004, 0.009];
    let suite = omega_ratio_suite(&returns, 0.05);
    if let (Some(at_zero), Some(at_rfr)) = (suite.omega_at_zero, suite.omega_at_rfr) {
        assert!(at_zero >= at_rfr, "Omega at 0 should be >= omega at higher threshold");
    }
}

// ============================================================================
// TAIL RATIO
// ============================================================================

#[test]
fn test_tail_ratio_symmetric() {
    // Generate roughly symmetric returns
    let mut returns: Vec<f64> = Vec::new();
    for i in 0..100 {
        let val = (i as f64 - 50.0) / 100.0; // -0.50 to 0.49
        returns.push(val);
    }
    let ratio = tail_ratio(&returns, 5.0).unwrap();
    // Should be approximately 1.0 for symmetric distribution
    assert!((ratio - 1.0).abs() < 0.3, "Symmetric returns should have tail ratio ~1.0: got {}", ratio);
}

#[test]
fn test_tail_ratio_insufficient_data() {
    let returns = vec![0.01, -0.01, 0.02]; // Only 3 data points
    assert!(tail_ratio(&returns, 5.0).is_none(), "Need >= 20 data points");
}

#[test]
fn test_tail_ratio_positive_skew() {
    // Right-skewed: small losses, occasional large gains
    let mut returns = vec![-0.01; 80]; // 80 small losses
    returns.extend(vec![0.10; 20]); // 20 big gains
    let ratio = tail_ratio(&returns, 5.0).unwrap();
    assert!(ratio > 1.0, "Right-skewed should have tail ratio > 1: got {}", ratio);
}

// ============================================================================
// TAIL ANALYSIS
// ============================================================================

#[test]
fn test_tail_analysis_sufficient_data() {
    let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 500.0).collect();
    let analysis = tail_analysis(&returns);
    assert!(analysis.tail_ratio_5.is_some());
    assert!(analysis.tail_ratio_1.is_some());
    assert!(analysis.left_tail_5 < 0.0, "5th percentile should be negative");
    assert!(analysis.right_tail_95 > 0.0, "95th percentile should be positive");
}

#[test]
fn test_tail_analysis_insufficient_data() {
    let returns = vec![0.01, -0.01, 0.02];
    let analysis = tail_analysis(&returns);
    assert!(analysis.tail_ratio_5.is_none());
}

#[test]
fn test_tail_analysis_skewness_sign() {
    // Right-skewed distribution
    let mut returns: Vec<f64> = vec![0.001; 80];
    returns.extend(vec![0.10; 20]); // Large positive outliers
    let analysis = tail_analysis(&returns);
    assert!(analysis.skewness > 0.0, "Right skew should give positive skewness: got {}", analysis.skewness);
}

#[test]
fn test_tail_analysis_kurtosis_fat_tails() {
    // Fat tails: mostly near zero with some extreme values
    let mut returns: Vec<f64> = vec![0.0; 80];
    returns.extend(vec![0.50, -0.50, 0.40, -0.40, 0.30, -0.30,
                        0.20, -0.20, 0.60, -0.60,
                        0.45, -0.45, 0.55, -0.55,
                        0.35, -0.35, 0.25, -0.25,
                        0.15, -0.15]); // 20 extreme values
    let analysis = tail_analysis(&returns);
    assert!(analysis.kurtosis > 0.0, "Fat tails should give positive excess kurtosis: got {}", analysis.kurtosis);
}

// ============================================================================
// KELLY CRITERION
// ============================================================================

#[test]
fn test_kelly_basic() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    assert!(result.kelly_full > 0.0, "Profitable strategy should have positive Kelly");
    assert!(result.kelly_half < result.kelly_full, "Half Kelly < full Kelly");
    assert!(result.kelly_quarter < result.kelly_half, "Quarter Kelly < half Kelly");
    assert!(result.win_probability > 0.0 && result.win_probability < 1.0);
    assert!(result.win_loss_ratio > 0.0);
    assert!(result.edge > 0.0, "Profitable strategy should have positive edge");
}

#[test]
fn test_kelly_insufficient_data() {
    let returns = vec![0.01, -0.01, 0.02]; // Only 3 trades
    assert!(kelly_criterion(&returns).is_none());
}

#[test]
fn test_kelly_all_wins() {
    let returns = vec![0.01, 0.02, 0.03, 0.015, 0.025, 0.012, 0.018, 0.022, 0.009, 0.011];
    assert!(kelly_criterion(&returns).is_none(), "All wins => no loss data => None");
}

#[test]
fn test_kelly_all_losses() {
    let returns = vec![-0.01, -0.02, -0.03, -0.015, -0.025, -0.012, -0.018, -0.022, -0.009, -0.011];
    assert!(kelly_criterion(&returns).is_none(), "All losses => no win data => None");
}

#[test]
fn test_kelly_validate_position_conservative() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    
    let validation = result.validate_position_size(result.kelly_quarter * 0.5);
    assert_eq!(validation.assessment, KellyAssessment::Conservative);
}

#[test]
fn test_kelly_validate_position_overleveraged() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    
    let validation = result.validate_position_size(result.kelly_full * 2.0);
    assert_eq!(validation.assessment, KellyAssessment::OverLeveraged);
}

#[test]
fn test_kelly_validate_position_no_position() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    
    let validation = result.validate_position_size(0.0);
    assert_eq!(validation.assessment, KellyAssessment::NoPosition);
}

#[test]
fn test_kelly_validate_position_optimal() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    
    // Position between quarter and half Kelly should be "Optimal"
    let mid = (result.kelly_quarter + result.kelly_half) / 2.0;
    let validation = result.validate_position_size(mid);
    assert_eq!(validation.assessment, KellyAssessment::Optimal);
}

#[test]
fn test_kelly_growth_rate() {
    let returns = vec![0.05, -0.03, 0.04, -0.02, 0.06, -0.03, 0.05, -0.02, 0.04, -0.01];
    let result = kelly_criterion(&returns).unwrap();
    
    // Growth rate at optimal fraction should be positive for profitable strategy
    let validation = result.validate_position_size(result.kelly_half);
    assert!(validation.expected_growth_rate > 0.0, "Expected growth at half Kelly should be positive");
    
    // Growth rate at zero should be zero
    let validation_zero = result.validate_position_size(0.0);
    assert!((validation_zero.expected_growth_rate - 0.0).abs() < 0.001);
}

// ============================================================================
// CROSS-ASSET MONITOR
// ============================================================================

#[test]
fn test_cross_asset_monitor_add_and_correlate() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    // Add correlated returns for two assets
    for i in 0..20 {
        let base = (i as f64 * 0.1).sin() * 0.05;
        monitor.add_return("BTC", base + 0.001);
        monitor.add_return("ETH", base * 0.8 + 0.002); // Highly correlated
    }
    
    let corr = monitor.calculate_correlation("BTC", "ETH").unwrap();
    assert!(corr > 0.5, "Correlated assets should have correlation > 0.5: got {}", corr);
}

#[test]
fn test_cross_asset_monitor_insufficient_data() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    for i in 0..5 {
        monitor.add_return("BTC", i as f64 * 0.01);
        monitor.add_return("ETH", i as f64 * 0.02);
    }
    
    assert!(monitor.calculate_correlation("BTC", "ETH").is_none(), "< 10 data points => None");
}

#[test]
fn test_cross_asset_monitor_unknown_asset() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    for i in 0..20 {
        monitor.add_return("BTC", i as f64 * 0.01);
    }
    
    assert!(monitor.calculate_correlation("BTC", "UNKNOWN").is_none());
}

#[test]
fn test_cross_asset_monitor_correlation_matrix() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    for i in 0..20 {
        let base = (i as f64 * 0.3).sin() * 0.05;
        monitor.add_return("BTC", base);
        monitor.add_return("ETH", base * 0.9);
        monitor.add_return("SOL", -base * 0.7); // Negatively correlated
    }
    
    let matrix = monitor.get_correlation_matrix();
    assert_eq!(matrix.assets.len(), 3);
    // Diagonal should be 1.0
    for i in 0..3 {
        assert!((matrix.matrix[i][i] - 1.0).abs() < 0.001);
    }
}

#[test]
fn test_cross_asset_monitor_average_correlation() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    for i in 0..20 {
        let base = (i as f64 * 0.2).sin() * 0.05;
        monitor.add_return("A", base);
        monitor.add_return("B", base * 0.8);
    }
    
    let avg = monitor.average_correlation().unwrap();
    assert!(avg > 0.0 && avg <= 1.0, "Average correlation should be in (0, 1]: got {}", avg);
}

#[test]
fn test_cross_asset_monitor_single_asset() {
    let mut monitor = CrossAssetMonitor::new(20);
    
    for i in 0..20 {
        monitor.add_return("BTC", i as f64 * 0.01);
    }
    
    assert!(monitor.average_correlation().is_none(), "Single asset => no correlation");
}

#[test]
fn test_cross_asset_window_size_minimum() {
    // Window size less than 10 should be clamped to 10
    let mut monitor = CrossAssetMonitor::new(5);
    
    for i in 0..15 {
        let val = (i as f64 * 0.2).sin() * 0.05;
        monitor.add_return("A", val);
        monitor.add_return("B", val * 0.9);
    }
    
    // Should still work because window is clamped to minimum 10
    let corr = monitor.calculate_correlation("A", "B");
    assert!(corr.is_some(), "Clamped window should allow correlation calculation");
}
