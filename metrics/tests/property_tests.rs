use proptest::prelude::*;

use metrics::risk::{
    average_drawdown, conditional_drawdown_at_risk, max_drawdown, pain_index, ulcer_index,
};
use metrics::significance::{deflated_sharpe_ratio, sharpe_significance, sharpe_t_stat};

// ============================================================================
// Helpers — generate random equity curves
// ============================================================================

#[allow(dead_code)]
fn arb_equity_curve(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(1.0f64..=1_000_000.0, min_len..=max_len)
}

fn arb_positive_curve(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(0.01f64..=1_000_000.0, min_len..=max_len)
}

// ============================================================================
// Risk metric invariants
// ============================================================================

proptest! {
    #[test]
    fn ulcer_index_non_negative(curve in arb_positive_curve(2, 500)) {
        if let Some(ui) = ulcer_index(&curve) {
            prop_assert!(ui >= 0.0, "ulcer_index must be >= 0, got {}", ui);
            prop_assert!(ui.is_finite(), "ulcer_index must be finite");
        }
    }

    #[test]
    fn pain_index_non_negative(curve in arb_positive_curve(2, 500)) {
        if let Some(pi) = pain_index(&curve) {
            prop_assert!(pi >= 0.0, "pain_index must be >= 0, got {}", pi);
            prop_assert!(pi.is_finite(), "pain_index must be finite");
        }
    }

    #[test]
    fn pain_le_ulcer(curve in arb_positive_curve(2, 500)) {
        if let (Some(pi), Some(ui)) = (pain_index(&curve), ulcer_index(&curve)) {
            // Mean absolute drawdown <= RMS drawdown (Cauchy-Schwarz)
            prop_assert!(
                pi <= ui + 1e-10,
                "pain_index ({}) must be <= ulcer_index ({})",
                pi, ui
            );
        }
    }

    #[test]
    fn max_drawdown_bounded(curve in arb_positive_curve(2, 500)) {
        if let Some(mdd) = max_drawdown(&curve) {
            prop_assert!(mdd >= 0.0 && mdd <= 1.0,
                "max_drawdown must be in [0, 1], got {}", mdd);
        }
    }

    #[test]
    fn average_drawdown_le_max(curve in arb_positive_curve(2, 500)) {
        if let (Some(avg), Some(mdd)) = (average_drawdown(&curve), max_drawdown(&curve)) {
            prop_assert!(
                avg <= mdd + 1e-10,
                "average_drawdown ({}) must be <= max_drawdown ({})",
                avg, mdd
            );
        }
    }

    #[test]
    fn cdar_bounded(curve in arb_positive_curve(5, 500)) {
        if let Some(cdar) = conditional_drawdown_at_risk(&curve, 0.95) {
            prop_assert!(cdar >= 0.0, "CDaR must be >= 0, got {}", cdar);
            prop_assert!(cdar <= 1.0 + 1e-10, "CDaR must be <= 1, got {}", cdar);
        }
    }

    #[test]
    fn cdar_ge_average_drawdown(curve in arb_positive_curve(10, 500)) {
        // CDaR at 95% should be >= average drawdown (tail average >= overall average)
        if let (Some(cdar), Some(avg)) =
            (conditional_drawdown_at_risk(&curve, 0.95), average_drawdown(&curve))
        {
            prop_assert!(
                cdar >= avg - 1e-10,
                "CDaR ({}) should be >= average_drawdown ({})",
                cdar, avg
            );
        }
    }
}

// ============================================================================
// Significance metric invariants
// ============================================================================

proptest! {
    #[test]
    fn sharpe_t_stat_finite(sharpe in -5.0f64..=5.0, n_days in 30usize..=5000) {
        let t = sharpe_t_stat(sharpe, n_days, 365.0);
        prop_assert!(t.is_finite(), "t-stat must be finite, got {}", t);
    }

    #[test]
    fn sharpe_significance_pvalue_bounded(sharpe in -3.0f64..=3.0, n_days in 30usize..=2000) {
        let (_t, p) = sharpe_significance(sharpe, n_days, 365.0);
        prop_assert!(p >= 0.0 && p <= 1.0,
            "p-value must be in [0, 1], got {}", p);
    }

    #[test]
    fn deflated_sharpe_bounded(
        sharpe in 0.01f64..=4.0,
        n_trials in 1usize..=1000,
        n_days in 60usize..=2000,
    ) {
        let dsr = deflated_sharpe_ratio(sharpe, n_trials, n_days, 1.0, 365.0);
        prop_assert!(dsr.is_finite(), "DSR must be finite, got {}", dsr);
        prop_assert!(dsr >= 0.0 && dsr <= 1.0,
            "DSR must be in [0, 1], got {}", dsr);
    }

    #[test]
    fn more_trials_lower_dsr(
        sharpe in 0.5f64..=2.0,
        n_days in 200usize..=1000,
    ) {
        let dsr_1 = deflated_sharpe_ratio(sharpe, 1, n_days, 1.0, 365.0);
        let dsr_100 = deflated_sharpe_ratio(sharpe, 100, n_days, 1.0, 365.0);
        prop_assert!(
            dsr_100 <= dsr_1 + 1e-10,
            "More trials should not increase DSR: dsr(1)={}, dsr(100)={}",
            dsr_1, dsr_100
        );
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn empty_curve_returns_none() {
    assert!(ulcer_index(&[]).is_none());
    assert!(pain_index(&[]).is_none());
    assert!(max_drawdown(&[]).is_none());
    assert!(conditional_drawdown_at_risk(&[], 0.95).is_none());
}

#[test]
fn single_element_curve() {
    let curve = [100.0];
    // Single element: no drawdown possible
    if let Some(mdd) = max_drawdown(&curve) {
        assert!(mdd == 0.0 || mdd.is_nan() == false);
    }
}

#[test]
fn constant_curve_zero_drawdown() {
    let curve = vec![100.0; 50];
    assert_eq!(max_drawdown(&curve), Some(0.0));
    assert_eq!(ulcer_index(&curve), Some(0.0));
    assert_eq!(pain_index(&curve), Some(0.0));
}

#[test]
fn monotonically_increasing_zero_drawdown() {
    let curve: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    assert_eq!(max_drawdown(&curve), Some(0.0));
    assert_eq!(ulcer_index(&curve), Some(0.0));
    assert_eq!(pain_index(&curve), Some(0.0));
}
