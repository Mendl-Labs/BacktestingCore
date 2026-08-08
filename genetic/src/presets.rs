//! Strategy parameter schema presets.
//!
//! Each preset defines the optimizable parameter space for a built-in strategy.
//! These are **data-driven** definitions — the genetic optimizer works with
//! [`DynamicChromosome`](super::dynamic_chromosome::DynamicChromosome) and
//! does not need to know which strategy the schema came from.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dynamic_chromosome::{Gene, ParamType, ParameterDef, ParameterSchema};

/// Look up a parameter schema by strategy name.
///
/// Returns `None` for Python/custom strategies — those provide their own schema
/// via `parameter_space()`.
pub fn schema_for_strategy(name: &str) -> Option<Arc<ParameterSchema>> {
    match name {
        "directional" | "momentum" | "trend" | "mean_reversion"
        | "BuiltinMarketMaker" | "BuiltinMarketMaking" | "avellaneda_stoikov" | "market_making" => {
            Some(directional_momentum_schema())
        }
        _ => None,
    }
}

/// Look up non-optimized default parameters by strategy name.
///
/// Returns an empty map for unknown strategies.
pub fn defaults_for_strategy(name: &str) -> HashMap<String, serde_json::Value> {
    match name {
        "directional" | "momentum" | "trend" | "mean_reversion"
        | "BuiltinMarketMaker" | "BuiltinMarketMaking" | "avellaneda_stoikov" | "market_making" => {
            directional_momentum_defaults()
        }
        _ => HashMap::new(),
    }
}

#[deprecated(note = "Market making removed; use directional_momentum_schema() instead")]
pub fn avellaneda_stoikov_schema() -> Arc<ParameterSchema> {
    directional_momentum_schema()
}

#[deprecated(note = "Market making removed; use directional_momentum_defaults() instead")]
pub fn avellaneda_stoikov_defaults() -> HashMap<String, serde_json::Value> {
    directional_momentum_defaults()
}

/// Directional / momentum strategy parameter schema.
///
/// Defines 8 optimizable parameters covering period selection,
/// entry/exit thresholds, stop-loss/take-profit, and position sizing.
/// Works for momentum crossover, trend-following, and mean-reversion
/// strategy families.
pub fn directional_momentum_schema() -> Arc<ParameterSchema> {
    let params = vec![
        ParameterDef {
            name: "fast_period".into(),
            param_type: ParamType::Int { min: 5, max: 50 },
            default: Gene::Int(10),
        },
        ParameterDef {
            name: "slow_period".into(),
            param_type: ParamType::Int { min: 20, max: 200 },
            default: Gene::Int(30),
        },
        ParameterDef {
            name: "entry_threshold".into(),
            param_type: ParamType::Float { min: 0.001, max: 0.05 },
            default: Gene::Float(0.01),
        },
        ParameterDef {
            name: "exit_threshold".into(),
            param_type: ParamType::Float { min: 0.001, max: 0.05 },
            default: Gene::Float(0.005),
        },
        ParameterDef {
            name: "stop_loss_pct".into(),
            param_type: ParamType::Float { min: 0.01, max: 0.10 },
            default: Gene::Float(0.03),
        },
        ParameterDef {
            name: "take_profit_pct".into(),
            param_type: ParamType::Float { min: 0.01, max: 0.20 },
            default: Gene::Float(0.06),
        },
        ParameterDef {
            name: "risk_per_trade".into(),
            param_type: ParamType::Float { min: 0.005, max: 0.05 },
            default: Gene::Float(0.01),
        },
        ParameterDef {
            name: "max_position_pct".into(),
            param_type: ParamType::Float { min: 0.1, max: 1.0 },
            default: Gene::Float(0.5),
        },
        ParameterDef {
            name: "leverage".into(),
            // A generous exploration range -- real per-exchange caps (e.g.
            // Alpaca 4x, Coinbase 1x, Oanda 50x) still clamp this down per
            // job at fitness-evaluation time (see
            // `config::ExchangeFeeConfig::resolve_leverage`), so this bound
            // is the GA's search range, not the enforced safety ceiling.
            // Default 1.0 matches `config::default_leverage()` exactly, so
            // a chromosome that doesn't benefit from leverage can settle at
            // today's exact pre-leverage behavior.
            param_type: ParamType::Float { min: 1.0, max: 20.0 },
            default: Gene::Float(1.0),
        },
    ];

    Arc::new(ParameterSchema { params })
}

/// Default non-optimized parameters for directional / momentum strategies.
pub fn directional_momentum_defaults() -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();

    m.insert("trend_filter_enabled".into(), serde_json::json!(true));
    m.insert("trend_ma_period".into(), serde_json::json!(100));
    m.insert("min_bars_between_trades".into(), serde_json::json!(5));
    m.insert("atr_period".into(), serde_json::json!(14));
    m.insert("volatility_filter_mult".into(), serde_json::json!(1.5));

    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_for_strategy_lookup() {
        assert!(schema_for_strategy("directional").is_some());
        assert!(schema_for_strategy("momentum").is_some());
        assert!(schema_for_strategy("trend").is_some());
        assert!(schema_for_strategy("mean_reversion").is_some());
        assert!(schema_for_strategy("SomePythonStrategy").is_none());
    }

    #[test]
    fn test_directional_momentum_schema_has_correct_count() {
        let schema = directional_momentum_schema();
        assert_eq!(schema.params.len(), 9, "Directional schema should have 9 optimizable params");
    }

    #[test]
    fn test_directional_momentum_schema_leverage_default_is_noop() {
        let schema = directional_momentum_schema();
        let leverage = schema.params.iter().find(|p| p.name == "leverage").unwrap();
        match leverage.param_type {
            ParamType::Float { min, max } => {
                assert_eq!(min, 1.0);
                assert_eq!(max, 20.0);
            }
            _ => panic!("leverage should be a Float param"),
        }
        assert!(matches!(leverage.default, Gene::Float(v) if v == 1.0));
    }

    #[test]
    fn test_directional_momentum_schema_has_int_periods() {
        let schema = directional_momentum_schema();
        let fast = schema.params.iter().find(|p| p.name == "fast_period").unwrap();
        let slow = schema.params.iter().find(|p| p.name == "slow_period").unwrap();
        assert!(matches!(fast.param_type, ParamType::Int { .. }));
        assert!(matches!(slow.param_type, ParamType::Int { .. }));
        assert!(matches!(fast.default, Gene::Int(10)));
        assert!(matches!(slow.default, Gene::Int(30)));
    }

    #[test]
    fn test_directional_momentum_defaults() {
        let defaults = directional_momentum_defaults();
        assert!(defaults.len() >= 5);
        assert!(defaults.contains_key("trend_filter_enabled"));
        assert!(defaults.contains_key("atr_period"));
    }
}
