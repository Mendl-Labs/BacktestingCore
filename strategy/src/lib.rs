//! Trading Strategy Module
//!
//! This module provides a comprehensive framework for implementing and managing
//! trading strategies with technical analysis, market data processing,
//! signal generation, and user-controlled deployment workflow.

pub mod traits;
pub mod signal;
pub mod indicators;
pub mod strategies;
pub mod simple_export;
pub mod deployment;
pub mod integration;
pub mod config;
pub mod logging_facade;
pub mod state_store;
pub mod feature_compiler;
pub mod indicator_registry;
pub mod executor;
pub mod mm_pricing;
pub mod strategy_source_validator;

// Re-export main types for convenience
pub use traits::{
    Strategy, StrategyContext, StrategyError, StrategyResult, StrategyMetrics,
    DataRequirements, StrategyCategory, PortfolioSnapshot,
};
pub use signal::{Signal, SignalType, SignalStrength, SignalReason};
pub use indicators::{IndicatorValue};
pub use strategy_source_validator::validate_strategy_source;
pub use simple_export::{
    StrategyExporter, BacktestResults, ExportableStrategy, ExportConfig,
};
pub use deployment::{
    DeploymentRequest, DeploymentStatus, BacktestMetrics, UserRiskConfig,
    AdvisoryWarning, RiskAcknowledgment, PlatformLimits, PlatformValidation,
    ApprovalRequirement, generate_advisory_warnings, generate_required_acknowledgments,
};
pub use integration::{BacktestIntegration};
pub use config::{ParameterValue, StrategyConfig, RiskLimits as StrategyRiskLimits, VenueScope};
pub use riskmanager::{RiskManager, RiskMetrics};
#[cfg(feature = "python")]
pub use strategies::inject_sdk;
#[cfg(feature = "python")]
pub use strategies::unique_module_name;

// Phase 1.11: Native Rust Executor
pub use state_store::{
    StateStore, StateSchema, StateType, FieldDef,
    BoundedDeque, BoundedHashSet,
    ModelRef, FeatureExpr, OutputMapping,
    extract_state_schema, extract_model_ref, extract_features,
};
pub use feature_compiler::{
    CompiledExpr, CompiledFeature, CompileError, CompilationResult,
    compile_features, evaluate_features,
};
pub use executor::{
    ExecutionTier, StrategyExecutor, TierAnalysis,
    NativeExecutor, HybridExecutor, PythonExecutor,
    analyse_strategy, build_executor,
};
pub use indicator_registry::{IndicatorSpec, IndicatorBank, extract_indicators};
pub use mm_pricing::{PricingFormula, MMPricingExecutor, extract_pricing};

use anyhow::Result;
use std::collections::HashMap;
use chrono::Duration;
use crate::logging_facade::STRATEGY_LOGGER;

/// Strategy manager that coordinates multiple strategies
#[derive(Debug)]
pub struct StrategyManager {
    /// Active strategies
    strategies: HashMap<String, Box<dyn Strategy>>,
    /// Strategy configurations
    configs: HashMap<String, StrategyConfig>,
    /// Global risk manager
    risk_manager: RiskManager,
}

impl StrategyManager {
    fn underlying_from_symbol(symbol: &str) -> String {
        if let Some((base, _)) = symbol.split_once('/') {
            return base.to_string();
        }
        if let Some((base, _)) = symbol.split_once('-') {
            return base.to_string();
        }
        symbol.to_string()
    }

    fn build_expiry_metadata(
        config: &StrategyConfig,
        instrument_kind: traits::InstrumentKind,
    ) -> traits::DerivativeMetadata {
        traits::DerivativeMetadata {
            symbol: config.symbol.clone(),
            underlying: Self::underlying_from_symbol(&config.symbol),
            instrument_kind,
            contract_multiplier: 1.0,
            settlement_currency: "USD".to_string(),
            exchange: config.exchange.clone(),
        }
    }

    fn to_directional_signal_type(signal_type: &SignalType) -> Option<SignalType> {
        match signal_type {
            SignalType::Buy
            | SignalType::ScaleIn
            | SignalType::BuyOption { .. }
            | SignalType::BuyFuture { .. } => Some(SignalType::Buy),
            SignalType::Sell
            | SignalType::ScaleOut
            | SignalType::SellOption { .. }
            | SignalType::SellFuture { .. }
            | SignalType::ExerciseOption { .. } => Some(SignalType::Sell),
            SignalType::HedgeDelta {
                target_delta,
                current_delta,
                ..
            } => {
                if current_delta > target_delta {
                    Some(SignalType::Sell)
                } else {
                    Some(SignalType::Buy)
                }
            }
            _ => None,
        }
    }

    fn normalize_signal_for_execution(signal: &Signal) -> Vec<Signal> {
        match &signal.signal_type {
            SignalType::OptionSpread { legs } => {
                if legs.is_empty() {
                    return Vec::new();
                }

                let base_qty = signal.quantity.unwrap_or(0.0);
                if base_qty <= 0.0 {
                    return Vec::new();
                }

                let mut expanded = Vec::with_capacity(legs.len());
                for (idx, leg) in legs.iter().enumerate() {
                    if leg.ratio == 0 {
                        return Vec::new();
                    }
                    let Some(dir) = Self::to_directional_signal_type(&leg.signal_type) else {
                        return Vec::new();
                    };

                    let mut leg_signal = signal.clone();
                    leg_signal.id = format!("{}::leg{}", signal.id, idx).into();
                    leg_signal.signal_type = dir;
                    leg_signal.quantity = Some(base_qty * (leg.ratio.abs() as f64));
                    leg_signal.price = leg.limit_price.or(signal.price);
                    leg_signal.is_limit = signal.is_limit || leg.limit_price.is_some();
                    expanded.push(leg_signal);
                }
                expanded
            }
            _ => {
                let mut normalized = signal.clone();
                if let Some(dir) = Self::to_directional_signal_type(&signal.signal_type) {
                    normalized.signal_type = dir;
                }
                vec![normalized]
            }
        }
    }

    /// Create a new strategy manager
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            configs: HashMap::new(),
            risk_manager: RiskManager::new(),
        }
    }

    /// Register a new strategy
    pub fn register_strategy<S>(&mut self, config: StrategyConfig, strategy: S) -> Result<()>
    where
        S: Strategy + 'static,
    {
        let id = config.id.clone();
        self.configs.insert(id.clone(), config);
        self.strategies.insert(id, Box::new(strategy));
        Ok(())
    }

    /// Remove a strategy
    pub fn unregister_strategy(&mut self, strategy_id: &str) -> Result<()> {
        self.strategies.remove(strategy_id);
        self.configs.remove(strategy_id);
        Ok(())
    }

    /// Get a reference to a strategy
    pub fn get_strategy(&self, strategy_id: &str) -> Option<&dyn Strategy> {
        self.strategies.get(strategy_id).map(|s| s.as_ref())
    }

    /// Get a mutable reference to a strategy


    /// Apply a closure to a mutable reference to a strategy, avoiding trait object lifetime issues
    pub fn with_strategy_mut<F, R>(&mut self, strategy_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut dyn Strategy) -> R,
    {
        self.strategies.get_mut(strategy_id).map(|s| f(s.as_mut()))
    }

    /// Get strategy configuration
    pub fn get_config(&self, strategy_id: &str) -> Option<&StrategyConfig> {
        self.configs.get(strategy_id)
    }

    /// List all registered strategy IDs
    pub fn list_strategies(&self) -> Vec<&str> {
        self.strategies.keys().map(|s| s.as_str()).collect()
    }

    /// Process market data for all enabled strategies
    pub async fn process_market_data(
        &mut self,
        market_data: &dataloader::MarketData,
        context: &StrategyContext,
    ) -> Result<Vec<(String, Vec<Signal>)>> {
        let mut all_signals = Vec::new();
        let expiry_cutoff = context.timestamp + Duration::hours(1);

        for (strategy_id, strategy) in &mut self.strategies {
            if let Some(config) = self.configs.get(strategy_id) {
                if !config.enabled {
                    continue;
                }

                // Check if this strategy applies to this market data
                match market_data {
                    dataloader::MarketData::Trade(trade) => {
                        if &*trade.symbol != config.symbol ||
                           &*trade.exchange != config.exchange {
                            continue;
                        }
                    }
                    dataloader::MarketData::Candle(candle) => {
                        if &*candle.symbol != config.symbol ||
                           &*candle.exchange != config.exchange {
                            continue;
                        }
                    }
                    dataloader::MarketData::PoolSwap(s) => {
                        if &*s.exchange != config.exchange {
                            continue;
                        }
                    }
                    dataloader::MarketData::Generic(_) => {
                        // Generic data has no symbol/exchange; allow through
                    }
                    dataloader::MarketData::OptionCandle(c) => {
                        // Options data has no exchange field (Polygon's
                        // consolidated-tape options data isn't segmented by
                        // exchange the way per-exchange crypto candles are)
                        // -- filter on underlying vs the strategy's symbol only.
                        if c.underlying != config.symbol {
                            continue;
                        }
                    }
                }

                let mut merged_signals = Vec::new();

                // Dispatch expiry lifecycle callbacks for configured instruments
                // that are approaching expiry within the next hour.
                for instrument_kind in strategy.required_instruments() {
                    let should_fire = instrument_kind
                        .expiry()
                        .map(|exp| exp >= context.timestamp && exp <= expiry_cutoff)
                        .unwrap_or(false);
                    if !should_fire {
                        continue;
                    }

                    let meta = Self::build_expiry_metadata(config, instrument_kind.clone());
                    match strategy.on_expiry(&meta, context).await {
                        Ok(mut expiry_signals) => merged_signals.append(&mut expiry_signals),
                        Err(e) => {
                            log_warn!(STRATEGY_LOGGER, "Strategy {} expiry hook failed: {}", strategy_id, e);
                        }
                    }
                }

                // Generate signals
                match strategy.generate_signals(market_data, context).await {
                    Ok(signals) => {
                        for signal in signals {
                            let normalized = Self::normalize_signal_for_execution(&signal);
                            if normalized.is_empty() {
                                if matches!(signal.signal_type, SignalType::OptionSpread { .. }) {
                                    log_warn!(
                                        STRATEGY_LOGGER,
                                        "Strategy {} emitted invalid OptionSpread; rejecting spread atomically",
                                        strategy_id
                                    );
                                }
                                continue;
                            }
                            merged_signals.extend(normalized);
                        }
                        if !merged_signals.is_empty() {
                            all_signals.push((strategy_id.clone(), merged_signals));
                        }
                    }
                    Err(e) => {
                        log_warn!(STRATEGY_LOGGER, "Strategy {} failed to generate signals: {}", strategy_id, e);
                    }
                }
            }
        }

        Ok(all_signals)
    }

    /// Get the global risk manager
    pub fn risk_manager(&self) -> &RiskManager {
        &self.risk_manager
    }

    /// Get mutable reference to risk manager
    pub fn risk_manager_mut(&mut self) -> &mut RiskManager {
        &mut self.risk_manager
    }

    /// Update parameters for all registered strategies
    pub async fn update_all_parameters(&mut self, parameters: HashMap<String, ParameterValue>) -> Result<()> {
        for strategy in self.strategies.values_mut() {
            strategy.update_parameters(parameters.clone()).await?;
        }
        Ok(())
    }
}

impl Default for StrategyManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for strategy development
pub mod utils {

    /// Calculate simple moving average
    pub fn sma(values: &[f64], period: usize) -> Option<f64> {
        if values.len() < period {
            return None;
        }
        let sum: f64 = values.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }

    /// Calculate exponential moving average
    pub fn ema(values: &[f64], period: usize, previous_ema: Option<f64>) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        let current_value = *values.last().unwrap();
        let alpha = 2.0 / (period as f64 + 1.0);

        match previous_ema {
            Some(prev) => Some(alpha * current_value + (1.0 - alpha) * prev),
            None => {
                // Bootstrap with SMA
                if values.len() >= period {
                    sma(values, period)
                } else {
                    Some(current_value)
                }
            }
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> Option<f64> {
        if values.len() < 2 {
            return None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;

        Some(variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_value_conversions() {
        let int_param = ParameterValue::Int(42);
        assert_eq!(int_param.as_int(), Some(42));
        assert_eq!(int_param.as_float(), Some(42.0));

        let float_param = ParameterValue::Float(3.14);
        assert_eq!(float_param.as_float(), Some(3.14));

        let string_param = ParameterValue::String("test".to_string());
        assert_eq!(string_param.as_string(), Some("test"));

        let bool_param = ParameterValue::Bool(true);
        assert_eq!(bool_param.as_bool(), Some(true));
    }

    #[test]
    fn test_strategy_manager_creation() {
        let manager = StrategyManager::new();
        assert_eq!(manager.list_strategies().len(), 0);
    }

    #[test]
    fn test_sma_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(utils::sma(&values, 3), Some(4.0)); // (3+4+5)/3
        assert_eq!(utils::sma(&values, 10), None); // Not enough data
    }

    #[test]
    fn test_ema_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = utils::ema(&values, 3, None);
        assert!(ema.is_some());
        
        let ema2 = utils::ema(&[6.0], 3, ema);
        assert!(ema2.is_some());
        assert!(ema2.unwrap() > ema.unwrap()); // Should trend upward
    }
}
