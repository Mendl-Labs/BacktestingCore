//! Per-feature diagnostics (information coefficient + quantile analysis)
//! for the optional `compute_features()` SDK hook. Pure logic, no PyO3
//! dependency — kept out of the `python`-feature-gated `python_simulation`
//! module so it stays unit-testable under default features, mirroring the
//! cost-hurdle split in `python_validation.rs`.

use std::collections::HashMap;

use crate::types::FeatureDiagnostic;

/// Compute per-feature IC + quantile analysis against forward returns, at a
/// horizon derived from the strategy's own average holding period. Returns
/// `None` when no features were computed (hook absent, disabled, or every
/// feature failed its numeric/length validation).
pub fn compute_feature_diagnostics(
    features: &Option<HashMap<String, Vec<f64>>>,
    prices: &[f64],
    timestamps_ms: &[i64],
    avg_trade_duration_secs: Option<f64>,
) -> Option<Vec<FeatureDiagnostic>> {
    let features = features.as_ref()?;
    if features.is_empty() || prices.len() < 2 {
        return None;
    }

    // Derive the forward-return horizon in bars from the strategy's actual
    // average holding period — never a fixed default. Falls back to 1 bar
    // when no trade has closed yet or bar spacing can't be determined.
    let avg_bar_ms = if timestamps_ms.len() > 1 {
        let span = (timestamps_ms[timestamps_ms.len() - 1] - timestamps_ms[0]) as f64;
        (span / (timestamps_ms.len() - 1) as f64).max(1.0)
    } else {
        0.0
    };
    let horizon = match (avg_trade_duration_secs, avg_bar_ms) {
        (Some(secs), bar_ms) if bar_ms > 0.0 => {
            (((secs * 1000.0) / bar_ms).round() as usize).max(1)
        }
        _ => 1,
    };

    if prices.len() <= horizon {
        return None;
    }
    let forward_returns: Vec<f64> = (0..prices.len() - horizon)
        .map(|i| {
            if prices[i] > 0.0 {
                (prices[i + horizon] - prices[i]) / prices[i]
            } else {
                0.0
            }
        })
        .collect();

    let mut diagnostics = Vec::new();
    for (name, values) in features.iter() {
        if values.len() != prices.len() {
            continue;
        }
        let aligned_feature = &values[..forward_returns.len()];
        let ic = match quant_diagnostics::compute_ic(aligned_feature, &forward_returns, horizon) {
            Some(ic) => ic,
            None => continue,
        };
        let quantile_buckets = quant_diagnostics::quantile_analysis(aligned_feature, &forward_returns, 5);
        diagnostics.push(FeatureDiagnostic {
            feature_name: name.clone(),
            ic,
            quantile_buckets,
        });
    }

    if diagnostics.is_empty() {
        None
    } else {
        Some(diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_diagnostics_none_when_no_features_computed() {
        let result = compute_feature_diagnostics(&None, &[1.0, 2.0, 3.0], &[0, 60_000, 120_000], None);
        assert!(result.is_none());
    }

    #[test]
    fn feature_diagnostics_none_when_features_empty() {
        let features = Some(HashMap::new());
        let result = compute_feature_diagnostics(&features, &[1.0, 2.0, 3.0], &[0, 60_000, 120_000], None);
        assert!(result.is_none());
    }

    #[test]
    fn feature_diagnostics_skips_misaligned_feature_length() {
        let mut features = HashMap::new();
        features.insert("bad_len".to_string(), vec![1.0, 2.0]); // shorter than prices
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let timestamps: Vec<i64> = (0..50).map(|i| i * 60_000).collect();
        let result = compute_feature_diagnostics(&Some(features), &prices, &timestamps, None);
        assert!(result.is_none());
    }

    #[test]
    fn feature_diagnostics_defaults_to_one_bar_horizon_with_no_closed_trades() {
        let mut features = HashMap::new();
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        features.insert("trend".to_string(), prices.clone());
        let timestamps: Vec<i64> = (0..50).map(|i| i * 60_000).collect();
        let result = compute_feature_diagnostics(&Some(features), &prices, &timestamps, None)
            .expect("enough data for a 1-bar horizon");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].feature_name, "trend");
        assert_eq!(result[0].ic.horizon, 1);
    }

    #[test]
    fn feature_diagnostics_derives_horizon_from_avg_trade_duration() {
        let mut features = HashMap::new();
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64).collect();
        features.insert("trend".to_string(), prices.clone());
        // 1-minute bars; average trade held for 5 minutes -> horizon should be 5 bars.
        let timestamps: Vec<i64> = (0..200).map(|i| i * 60_000).collect();
        let result = compute_feature_diagnostics(&Some(features), &prices, &timestamps, Some(300.0))
            .expect("enough data");
        assert_eq!(result[0].ic.horizon, 5);
    }

    #[test]
    fn feature_diagnostics_finds_informative_feature() {
        // A feature identical to price should be strongly informative of the
        // forward return (both trend upward together) — sanity check that
        // the IC pipeline actually detects a planted signal.
        let mut features = HashMap::new();
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 2.0).collect();
        features.insert("price_level".to_string(), prices.clone());
        let timestamps: Vec<i64> = (0..100).map(|i| i * 60_000).collect();
        let result = compute_feature_diagnostics(&Some(features), &prices, &timestamps, None)
            .expect("enough data");
        assert_eq!(result[0].ic.horizon, 1);
        assert!(result[0].ic.significant);
        assert!(!result[0].quantile_buckets.is_empty());
    }
}
