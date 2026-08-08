//! Fast signal generation from pre-computed indicators and declarative rules.
//!
//! Evaluates crossover/threshold rules in pure Rust using pre-computed indicator
//! arrays. No Python, no GIL — enables parallel GA evaluation with rayon.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::candle_sim::{SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD};
use crate::indicator_precompute::PrecomputedIndicators;

/// Reference to an indicator value, resolved at evaluation time from the chromosome's params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorRef {
    /// Name of the indicator (matches key in PrecomputedIndicators)
    pub indicator_name: String,
    /// Which chromosome parameter determines the period (e.g., "fast_period")
    pub param_key: String,
    /// For multi-value indicators, which suffix (e.g., "macd", "signal", "upper")
    pub suffix: Option<String>,
}

/// A signal generation rule evaluated per-bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalRule {
    /// BUY when indicator_a crosses above indicator_b
    CrossoverAbove { a: IndicatorRef, b: IndicatorRef },
    /// SELL when indicator_a crosses below indicator_b
    CrossoverBelow { a: IndicatorRef, b: IndicatorRef },
    /// BUY when indicator drops below a threshold (e.g., RSI < 30)
    BelowThreshold { indicator: IndicatorRef, threshold_param: String },
    /// SELL when indicator rises above a threshold (e.g., RSI > 70)
    AboveThreshold { indicator: IndicatorRef, threshold_param: String },
}

/// A complete signal rule set for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRuleSet {
    pub buy_rules: Vec<SignalRule>,
    pub sell_rules: Vec<SignalRule>,
    /// If true, ALL buy rules must fire to generate BUY (AND logic).
    /// If false, ANY buy rule firing generates BUY (OR logic).
    pub require_all_buy: bool,
    /// Same for sell rules.
    pub require_all_sell: bool,
}

/// Generate a signal array from rules + pre-computed indicators + chromosome params.
///
/// Returns i8 array: 1=BUY, -1=SELL, 0=HOLD
pub fn evaluate_rules(
    rules: &SignalRuleSet,
    precomputed: &PrecomputedIndicators,
    params: &HashMap<String, serde_json::Value>,
    num_bars: usize,
) -> Vec<i8> {
    let mut signals = vec![SIGNAL_HOLD; num_bars];

    for i in 1..num_bars {
        let buy_fired = evaluate_rule_set(&rules.buy_rules, precomputed, params, i, rules.require_all_buy);
        let sell_fired = evaluate_rule_set(&rules.sell_rules, precomputed, params, i, rules.require_all_sell);

        if buy_fired && !sell_fired {
            signals[i] = SIGNAL_BUY;
        } else if sell_fired && !buy_fired {
            signals[i] = SIGNAL_SELL;
        }
    }

    signals
}

fn evaluate_rule_set(
    rules: &[SignalRule],
    precomputed: &PrecomputedIndicators,
    params: &HashMap<String, serde_json::Value>,
    bar_index: usize,
    require_all: bool,
) -> bool {
    if rules.is_empty() {
        return false;
    }

    let mut any_fired = false;
    let mut all_fired = true;

    for rule in rules {
        let fired = evaluate_single_rule(rule, precomputed, params, bar_index);
        if fired {
            any_fired = true;
        } else {
            all_fired = false;
        }
    }

    if require_all { all_fired } else { any_fired }
}

fn evaluate_single_rule(
    rule: &SignalRule,
    precomputed: &PrecomputedIndicators,
    params: &HashMap<String, serde_json::Value>,
    i: usize,
) -> bool {
    match rule {
        SignalRule::CrossoverAbove { a, b } => {
            let (a_curr, a_prev) = resolve_indicator_pair(a, precomputed, params, i);
            let (b_curr, b_prev) = resolve_indicator_pair(b, precomputed, params, i);
            match (a_curr, a_prev, b_curr, b_prev) {
                (Some(ac), Some(ap), Some(bc), Some(bp)) => {
                    ap <= bp && ac > bc
                }
                _ => false,
            }
        }
        SignalRule::CrossoverBelow { a, b } => {
            let (a_curr, a_prev) = resolve_indicator_pair(a, precomputed, params, i);
            let (b_curr, b_prev) = resolve_indicator_pair(b, precomputed, params, i);
            match (a_curr, a_prev, b_curr, b_prev) {
                (Some(ac), Some(ap), Some(bc), Some(bp)) => {
                    ap >= bp && ac < bc
                }
                _ => false,
            }
        }
        SignalRule::BelowThreshold { indicator, threshold_param } => {
            let threshold = params.get(threshold_param).and_then(|v| v.as_f64()).unwrap_or(30.0);
            let (curr, _) = resolve_indicator_pair(indicator, precomputed, params, i);
            curr.map(|v| v < threshold).unwrap_or(false)
        }
        SignalRule::AboveThreshold { indicator, threshold_param } => {
            let threshold = params.get(threshold_param).and_then(|v| v.as_f64()).unwrap_or(70.0);
            let (curr, _) = resolve_indicator_pair(indicator, precomputed, params, i);
            curr.map(|v| v > threshold).unwrap_or(false)
        }
    }
}

/// Resolve an indicator reference to (current_value, previous_value) at bar index i.
fn resolve_indicator_pair(
    ind_ref: &IndicatorRef,
    precomputed: &PrecomputedIndicators,
    params: &HashMap<String, serde_json::Value>,
    i: usize,
) -> (Option<f64>, Option<f64>) {
    let period = params.get(&ind_ref.param_key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .unwrap_or(0) as u32;

    let arr = if let Some(ref suffix) = ind_ref.suffix {
        precomputed.get_multi(&ind_ref.indicator_name, period, suffix)
    } else {
        precomputed.get_single(&ind_ref.indicator_name, period)
    };

    match arr {
        Some(data) if i > 0 && i < data.len() => {
            let curr = if data[i].is_nan() { None } else { Some(data[i]) };
            let prev = if data[i - 1].is_nan() { None } else { Some(data[i - 1]) };
            (curr, prev)
        }
        _ => (None, None),
    }
}

use crate::indicator_precompute::IndicatorParamBinding;
use strategy::indicator_registry::IndicatorSpec;

/// Returns a SignalRuleSet for known rule-based strategy types, or None for Python-only strategies.
pub fn rules_for_strategy(strategy_type: &str) -> Option<SignalRuleSet> {
    match strategy_type {
        "directional_momentum" | "sma_crossover" | "ema_crossover" => {
            let ind_type = if strategy_type == "ema_crossover" { "ema" } else { "sma" };
            let _ = ind_type; // naming is the same in indicator_name keys
            Some(SignalRuleSet {
                buy_rules: vec![SignalRule::CrossoverAbove {
                    a: IndicatorRef { indicator_name: "fast_ma".into(), param_key: "fast_period".into(), suffix: None },
                    b: IndicatorRef { indicator_name: "slow_ma".into(), param_key: "slow_period".into(), suffix: None },
                }],
                sell_rules: vec![SignalRule::CrossoverBelow {
                    a: IndicatorRef { indicator_name: "fast_ma".into(), param_key: "fast_period".into(), suffix: None },
                    b: IndicatorRef { indicator_name: "slow_ma".into(), param_key: "slow_period".into(), suffix: None },
                }],
                require_all_buy: true,
                require_all_sell: true,
            })
        }
        "rsi_mean_reversion" => Some(SignalRuleSet {
            buy_rules: vec![SignalRule::BelowThreshold {
                indicator: IndicatorRef { indicator_name: "rsi".into(), param_key: "rsi_period".into(), suffix: None },
                threshold_param: "oversold_threshold".into(),
            }],
            sell_rules: vec![SignalRule::AboveThreshold {
                indicator: IndicatorRef { indicator_name: "rsi".into(), param_key: "rsi_period".into(), suffix: None },
                threshold_param: "overbought_threshold".into(),
            }],
            require_all_buy: true,
            require_all_sell: true,
        }),
        _ => None,
    }
}

/// Returns indicator bindings for known rule-based strategy types.
pub fn indicator_bindings_for_strategy(strategy_type: &str) -> Option<Vec<IndicatorParamBinding>> {
    match strategy_type {
        "directional_momentum" | "sma_crossover" => Some(vec![
            IndicatorParamBinding {
                indicator_name: "fast_ma".into(),
                spec_template: IndicatorSpec::Sma { period: 0 },
                param_key: "fast_period".into(),
            },
            IndicatorParamBinding {
                indicator_name: "slow_ma".into(),
                spec_template: IndicatorSpec::Sma { period: 0 },
                param_key: "slow_period".into(),
            },
        ]),
        "ema_crossover" => Some(vec![
            IndicatorParamBinding {
                indicator_name: "fast_ma".into(),
                spec_template: IndicatorSpec::Ema { period: 0 },
                param_key: "fast_period".into(),
            },
            IndicatorParamBinding {
                indicator_name: "slow_ma".into(),
                spec_template: IndicatorSpec::Ema { period: 0 },
                param_key: "slow_period".into(),
            },
        ]),
        "rsi_mean_reversion" => Some(vec![
            IndicatorParamBinding {
                indicator_name: "rsi".into(),
                spec_template: IndicatorSpec::Rsi { period: 0 },
                param_key: "rsi_period".into(),
            },
        ]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_sim::CandleArrays;
    use crate::indicator_precompute::{PrecomputedIndicators, IndicatorParamBinding};
    use std::sync::Arc;
    use strategy::indicator_registry::IndicatorSpec;
    use genetic::dynamic_chromosome::{ParameterSchema, ParameterDef, ParamType, Gene};

    fn setup_sma_crossover() -> (PrecomputedIndicators, SignalRuleSet, HashMap<String, serde_json::Value>) {
        // Price that trends up: 100, 101, 102, ... 120
        let closes: Vec<f64> = (100..=120).map(|i| i as f64).collect();
        let n = closes.len();
        let candles = Arc::new(CandleArrays {
            opens: closes.clone(),
            highs: closes.iter().map(|c| c + 0.5).collect(),
            lows: closes.iter().map(|c| c - 0.5).collect(),
            closes,
            volumes: vec![1000.0; n],
            timestamps_ms: (0..n as i64).map(|i| i * 86_400_000).collect(),
        });

        let schema = ParameterSchema {
            params: vec![
                ParameterDef { name: "fast_period".into(), param_type: ParamType::Int { min: 3, max: 5 }, default: Gene::Int(3) },
                ParameterDef { name: "slow_period".into(), param_type: ParamType::Int { min: 7, max: 10 }, default: Gene::Int(7) },
            ],
        };

        let bindings = vec![
            IndicatorParamBinding {
                indicator_name: "fast_sma".into(),
                spec_template: IndicatorSpec::Sma { period: 0 },
                param_key: "fast_period".into(),
            },
            IndicatorParamBinding {
                indicator_name: "slow_sma".into(),
                spec_template: IndicatorSpec::Sma { period: 0 },
                param_key: "slow_period".into(),
            },
        ];

        let precomputed = PrecomputedIndicators::from_candles_and_bindings(candles, &schema, &bindings);

        let rules = SignalRuleSet {
            buy_rules: vec![SignalRule::CrossoverAbove {
                a: IndicatorRef { indicator_name: "fast_sma".into(), param_key: "fast_period".into(), suffix: None },
                b: IndicatorRef { indicator_name: "slow_sma".into(), param_key: "slow_period".into(), suffix: None },
            }],
            sell_rules: vec![SignalRule::CrossoverBelow {
                a: IndicatorRef { indicator_name: "fast_sma".into(), param_key: "fast_period".into(), suffix: None },
                b: IndicatorRef { indicator_name: "slow_sma".into(), param_key: "slow_period".into(), suffix: None },
            }],
            require_all_buy: true,
            require_all_sell: true,
        };

        let mut params = HashMap::new();
        params.insert("fast_period".into(), serde_json::json!(3));
        params.insert("slow_period".into(), serde_json::json!(7));

        (precomputed, rules, params)
    }

    #[test]
    fn uptrend_generates_no_sell() {
        let (precomputed, rules, params) = setup_sma_crossover();
        let n = precomputed.candles.len();
        let signals = evaluate_rules(&rules, &precomputed, &params, n);
        // In a pure uptrend, fast SMA is always above slow SMA after warmup
        // so no sell signals should fire
        assert!(!signals.iter().any(|&s| s == SIGNAL_SELL));
    }
}
