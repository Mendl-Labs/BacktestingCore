//! Strategy configuration types and parameter management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a trading strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Unique identifier for the strategy
    pub id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Type of strategy (e.g., "Custom", "Momentum")
    pub strategy_type: String,
    
    /// Trading symbol (e.g., "BTCUSD")
    pub symbol: String,
    
    /// Exchange name (e.g., "kraken", "coinbase")
    pub exchange: String,
    
    /// Strategy parameters
    pub parameters: HashMap<String, config::ParameterValue>,
    
    /// Risk management limits
    pub risk_limits: RiskLimits,

    /// Whether the strategy is enabled
    pub enabled: bool,

    /// Cross-exchange strategy support (2026-07-22): when set (2+ entries),
    /// this strategy declares multiple `(symbol, exchange)` venues instead of
    /// just the single pair above. `symbol`/`exchange` above stay the primary
    /// venue for backward compatibility with every existing single-venue
    /// consumer -- `venues`, when present, should include that same primary
    /// pair as its first element. See `Strategy::compute_all_signals_multi_venue`
    /// (traits.rs) for how a multi-venue strategy actually receives per-venue
    /// data.
    #[serde(default)]
    pub venues: Option<Vec<VenueScope>>,
}

/// One `(symbol, exchange)` pair a multi-venue strategy declares interest in.
/// See `StrategyConfig::venues`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VenueScope {
    pub symbol: String,
    pub exchange: String,
}

// Re-export ParameterValue from config crate to avoid duplication
pub use config::ParameterValue;

// Old local ParameterValue commented out to avoid conflicts
/*
/// Parameter value that can be of different types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    Array(Vec<ParameterValue>),
}

// Note: Methods like as_int(), as_float(), etc. are defined in config::ParameterValue
*/

/// Risk management limits for a strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskLimits {
    /// Maximum position size
    pub max_position_size: Option<f64>,
    
    /// Maximum daily loss allowed
    pub max_daily_loss: Option<f64>,
    
    /// Maximum number of open orders
    pub max_open_orders: Option<u32>,
    
    /// Maximum drawdown percentage
    pub max_drawdown_percent: Option<f64>,
    
    /// Stop loss percentage
    pub stop_loss_percent: Option<f64>,
    
    /// Take profit percentage
    pub take_profit_percent: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_limits_default_all_none() {
        let r = RiskLimits::default();
        assert!(r.max_position_size.is_none());
        assert!(r.max_daily_loss.is_none());
        assert!(r.max_open_orders.is_none());
        assert!(r.max_drawdown_percent.is_none());
        assert!(r.stop_loss_percent.is_none());
        assert!(r.take_profit_percent.is_none());
    }

    #[test]
    fn risk_limits_serde_roundtrip() {
        let r = RiskLimits {
            max_position_size: Some(1000.0),
            max_daily_loss: Some(500.0),
            max_open_orders: Some(10),
            max_drawdown_percent: Some(5.0),
            stop_loss_percent: Some(2.0),
            take_profit_percent: Some(3.0),
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: RiskLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_open_orders, Some(10));
        assert!((decoded.max_position_size.unwrap() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn strategy_config_serde_roundtrip() {
        let c = StrategyConfig {
            id: "strat-1".into(),
            name: "Test Strategy".into(),
            strategy_type: "Custom".into(),
            symbol: "BTCUSD".into(),
            exchange: "kraken".into(),
            parameters: HashMap::new(),
            risk_limits: RiskLimits::default(),
            enabled: true,
            venues: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        let decoded: StrategyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "strat-1");
        assert_eq!(decoded.strategy_type, "Custom");
        assert!(decoded.enabled);
    }

    #[test]
    fn strategy_config_with_parameters() {
        let mut params = HashMap::new();
        params.insert("spread".into(), config::ParameterValue::Float(0.01));
        params.insert("depth".into(), config::ParameterValue::Int(5));
        let c = StrategyConfig {
            id: "s1".into(), name: "Test".into(), strategy_type: "Custom".into(),
            symbol: "ETHUSD".into(), exchange: "binance".into(),
            parameters: params, risk_limits: RiskLimits::default(), enabled: false,
            venues: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        let decoded: StrategyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.parameters.len(), 2);
        assert!(!decoded.enabled);
    }

    #[test]
    fn strategy_config_venues_defaults_to_none_when_absent_from_json() {
        // Older persisted configs (or any producer that predates this field)
        // have no "venues" key at all -- #[serde(default)] must not error.
        let c = StrategyConfig {
            id: "s1".into(), name: "Test".into(), strategy_type: "Custom".into(),
            symbol: "BTCUSD".into(), exchange: "kraken".into(),
            parameters: HashMap::new(), risk_limits: RiskLimits::default(), enabled: true,
            venues: None,
        };
        let mut value = serde_json::to_value(&c).unwrap();
        value.as_object_mut().unwrap().remove("venues");
        let decoded: StrategyConfig = serde_json::from_value(value).unwrap();
        assert!(decoded.venues.is_none());
    }

    #[test]
    fn strategy_config_venues_roundtrips_multiple_scopes() {
        let c = StrategyConfig {
            id: "s2".into(), name: "Multi-venue Test".into(), strategy_type: "Custom".into(),
            symbol: "BTC-USD".into(), exchange: "kraken".into(),
            parameters: HashMap::new(), risk_limits: RiskLimits::default(), enabled: true,
            venues: Some(vec![
                VenueScope { symbol: "BTC-USD".into(), exchange: "kraken".into() },
                VenueScope { symbol: "BTC-USD".into(), exchange: "coinbase".into() },
                VenueScope { symbol: "BTC-USD".into(), exchange: "binance".into() },
            ]),
        };
        let json = serde_json::to_string(&c).unwrap();
        let decoded: StrategyConfig = serde_json::from_str(&json).unwrap();
        let venues = decoded.venues.expect("venues should round-trip");
        assert_eq!(venues.len(), 3);
        assert_eq!(venues[0], VenueScope { symbol: "BTC-USD".into(), exchange: "kraken".into() });
        assert_eq!(venues[2].exchange, "binance");
    }
}
