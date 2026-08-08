//! Portfolio/Multi-Symbol Simulation Engine
//!
//! This module provides portfolio-level backtesting across multiple symbols
//! using the CandleSimulator for position simulation:
//! - Synchronized candle-based processing across assets
//! - Portfolio-level capital allocation
//! - Cross-asset risk management
//! - Aggregated performance metrics

use crate::candle_sim::{CandleSimulator, CandleArrays, CandleSimResult};
use crate::logging_facade::BACKTEST_LOGGER;
use crate::portfolio_risk::PortfolioRiskMetrics;
use crate::log_info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy-specific parameters for an asset in portfolio simulation
#[derive(Debug, Clone)]
pub struct AssetStrategyParams {
    pub stop_loss_pct: Option<f64>,
    pub take_profit_pct: Option<f64>,
    pub position_size_pct: f64,
    pub commission_rate: f64,
    pub slippage_bps: f64,
}

impl Default for AssetStrategyParams {
    fn default() -> Self {
        Self {
            stop_loss_pct: Some(0.05),
            take_profit_pct: Some(0.10),
            position_size_pct: 0.5,
            commission_rate: 0.001,
            slippage_bps: 1.0,
        }
    }
}

/// Per-symbol simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolResult {
    pub symbol: String,
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_volume: f64,
    pub total_fees: f64,
    pub allocation_weight: f64,
    pub allocated_capital: f64,
    pub final_equity: f64,
}

/// Portfolio-level aggregated result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSimResult {
    pub symbol_results: Vec<SymbolResult>,
    pub total_pnl: f64,
    pub portfolio_sharpe: f64,
    pub portfolio_max_drawdown: f64,
    pub total_trades: usize,
    pub portfolio_win_rate: f64,
    pub portfolio_profit_factor: f64,
    pub total_volume: f64,
    pub total_fees: f64,
    pub equity_curve: Vec<f64>,
    pub trade_returns: Vec<f64>,
    pub risk_metrics: Option<PortfolioRiskMetrics>,
    pub initial_capital: f64,
    pub final_equity: f64,
    pub correlation_matrix: Option<Vec<CorrelationPair>>,
    pub first_event_timestamp_ms: Option<i64>,
    pub last_event_timestamp_ms: Option<i64>,
}

/// Correlation between two symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    pub symbol1: String,
    pub symbol2: String,
    pub correlation: f64,
}

/// Configuration for cross-asset correlation impact penalty
#[derive(Debug, Clone)]
pub struct CrossAssetImpactConfig {
    pub enabled: bool,
    pub correlation_impact_bps: f64,
    pub threshold: f64,
}

impl Default for CrossAssetImpactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            correlation_impact_bps: 5.0,
            threshold: 0.6,
        }
    }
}

/// Configuration for cross-exchange capital transfer latency
#[derive(Debug, Clone)]
pub struct CrossExchangeConfig {
    pub enabled: bool,
    pub transfer_delay_ms: i64,
}

impl Default for CrossExchangeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            transfer_delay_ms: 30 * 60 * 1000, // 30 minutes
        }
    }
}

/// Portfolio simulator that runs candle-based simulations across multiple symbols
pub struct PortfolioSimulator {
    initial_capital: f64,
    pub cross_asset_config: CrossAssetImpactConfig,
    pub cross_exchange_config: CrossExchangeConfig,
}

impl PortfolioSimulator {
    pub fn new(initial_capital: f64, _fee_configs: HashMap<String, config::ExchangeFeeConfig>) -> Self {
        Self {
            initial_capital,
            cross_asset_config: CrossAssetImpactConfig::default(),
            cross_exchange_config: CrossExchangeConfig::default(),
        }
    }

    /// Run portfolio simulation across multiple symbols using candle data and signal arrays.
    pub fn run_candle_based(
        &self,
        symbol_candles: &HashMap<String, CandleArrays>,
        symbol_signals: &HashMap<String, Vec<i8>>,
        asset_params: &HashMap<String, AssetStrategyParams>,
        allocation_weights: Option<&[f64]>,
    ) -> PortfolioSimResult {
        let start_time = std::time::Instant::now();
        let symbols: Vec<String> = symbol_candles.keys().cloned().collect();
        let num_symbols = symbols.len();

        log_info!(
            BACKTEST_LOGGER,
            "PORTFOLIO_SIM_STARTED: num_assets={}, initial_capital={:.2}",
            num_symbols,
            self.initial_capital
        );

        if num_symbols == 0 {
            return self.empty_result();
        }

        let weights: Vec<f64> = allocation_weights
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0 / num_symbols as f64; num_symbols]);

        let mut symbol_results: Vec<SymbolResult> = Vec::with_capacity(num_symbols);
        let mut all_equity_curves: Vec<Vec<f64>> = Vec::new();
        let mut all_trade_returns: Vec<f64> = Vec::new();

        for (i, symbol) in symbols.iter().enumerate() {
            let weight = weights.get(i).copied().unwrap_or(1.0 / num_symbols as f64);
            let allocated_capital = self.initial_capital * weight;

            let default_params = AssetStrategyParams::default();
            let params = asset_params.get(symbol).unwrap_or(&default_params);

            let candles = match symbol_candles.get(symbol) {
                Some(c) => c,
                None => continue,
            };
            let signals = match symbol_signals.get(symbol) {
                Some(s) => s,
                None => continue,
            };

            let sim = CandleSimulator::new(
                allocated_capital,
                params.commission_rate,
                config::SlippageModel::Flat { bps: params.slippage_bps },
            ).with_realism(crate::candle_sim::RealismConfig {
                intrabar_path: true,
                market_impact_k: 0.3,
                spread_from_range: true,
                adverse_selection: true,
                ..Default::default()
            });

            let result = sim.run(
                signals,
                candles,
                params.stop_loss_pct,
                params.take_profit_pct,
                params.position_size_pct,
                None,
            );

            all_trade_returns.extend(result.trade_returns.iter().cloned());
            all_equity_curves.push(result.equity_curve.clone());

            let final_eq = result.equity_curve.last().copied().unwrap_or(allocated_capital);

            symbol_results.push(SymbolResult {
                symbol: symbol.clone(),
                total_pnl: result.total_pnl,
                sharpe_ratio: result.sharpe_ratio,
                max_drawdown: result.max_drawdown,
                num_trades: result.num_trades,
                win_rate: result.win_rate,
                profit_factor: result.profit_factor,
                total_volume: 0.0,
                total_fees: result.total_commission,
                allocation_weight: weight,
                allocated_capital,
                final_equity: final_eq,
            });
        }

        // Item 7: Cross-asset correlation impact penalty
        if self.cross_asset_config.enabled && symbol_results.len() >= 2 {
            let correlation_matrix = self.calculate_correlation_matrix(&symbol_results, &all_equity_curves);
            for pair in &correlation_matrix {
                if pair.correlation > self.cross_asset_config.threshold {
                    let idx1 = symbol_results.iter().position(|r| r.symbol == pair.symbol1);
                    let idx2 = symbol_results.iter().position(|r| r.symbol == pair.symbol2);
                    if let (Some(i1), Some(i2)) = (idx1, idx2) {
                        let overlap_value = symbol_results[i1].allocated_capital
                            .min(symbol_results[i2].allocated_capital);
                        let penalty = pair.correlation
                            * self.cross_asset_config.correlation_impact_bps / 10_000.0
                            * overlap_value;
                        symbol_results[i1].total_pnl -= penalty * 0.5;
                        symbol_results[i2].total_pnl -= penalty * 0.5;
                        symbol_results[i1].final_equity -= penalty * 0.5;
                        symbol_results[i2].final_equity -= penalty * 0.5;
                    }
                }
            }
        }

        let result = self.aggregate_results(symbol_results, all_equity_curves, all_trade_returns);

        let elapsed_ms = start_time.elapsed().as_millis();
        log_info!(
            BACKTEST_LOGGER,
            "PORTFOLIO_SIM_COMPLETED: final_equity={:.2}, total_trades={}, elapsed_ms={}",
            result.final_equity,
            result.total_trades,
            elapsed_ms
        );

        result
    }

    fn aggregate_results(
        &self,
        symbol_results: Vec<SymbolResult>,
        all_equity_curves: Vec<Vec<f64>>,
        all_trade_returns: Vec<f64>,
    ) -> PortfolioSimResult {
        let portfolio_equity_curve = self.aggregate_equity_curves(&all_equity_curves);

        let total_pnl: f64 = symbol_results.iter().map(|r| r.total_pnl).sum();
        let total_trades: usize = symbol_results.iter().map(|r| r.num_trades).sum();
        let total_volume: f64 = symbol_results.iter().map(|r| r.total_volume).sum();
        let total_fees: f64 = symbol_results.iter().map(|r| r.total_fees).sum();
        let final_equity: f64 = symbol_results.iter().map(|r| r.final_equity).sum();

        let portfolio_sharpe = self.calculate_portfolio_sharpe(&portfolio_equity_curve);
        let portfolio_max_drawdown = self.calculate_max_drawdown(&portfolio_equity_curve);

        let portfolio_win_rate = if total_trades > 0 {
            symbol_results.iter()
                .map(|r| r.win_rate * r.num_trades as f64)
                .sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = all_trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = all_trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let portfolio_profit_factor = if gross_loss > 0.0 {
            (gross_profit / gross_loss).min(100.0)
        } else if gross_profit > 0.0 {
            10.0
        } else {
            1.0
        };

        let correlation_matrix = self.calculate_correlation_matrix(&symbol_results, &all_equity_curves);

        PortfolioSimResult {
            symbol_results,
            total_pnl,
            portfolio_sharpe,
            portfolio_max_drawdown,
            total_trades,
            portfolio_win_rate,
            portfolio_profit_factor,
            total_volume,
            total_fees,
            equity_curve: portfolio_equity_curve,
            trade_returns: all_trade_returns,
            risk_metrics: None,
            initial_capital: self.initial_capital,
            final_equity,
            correlation_matrix: Some(correlation_matrix),
            first_event_timestamp_ms: None,
            last_event_timestamp_ms: None,
        }
    }

    fn aggregate_equity_curves(&self, curves: &[Vec<f64>]) -> Vec<f64> {
        if curves.is_empty() {
            return vec![self.initial_capital];
        }

        let max_len = curves.iter().map(|c| c.len()).max().unwrap_or(0);
        if max_len == 0 {
            return vec![self.initial_capital];
        }

        let mut portfolio_curve = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let total_equity: f64 = curves.iter()
                .map(|curve| {
                    if i < curve.len() { curve[i] } else { curve.last().copied().unwrap_or(0.0) }
                })
                .sum();
            portfolio_curve.push(total_equity);
        }
        portfolio_curve
    }

    fn calculate_portfolio_sharpe(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.len() < 10 {
            return 0.0;
        }

        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .filter(|r| r.is_finite())
            .collect();

        if returns.len() < 10 {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            (mean_return / std_dev) * (365.0_f64).sqrt()
        } else {
            0.0
        }
    }

    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut peak = equity_curve[0];
        let mut max_dd = 0.0;

        for &equity in equity_curve.iter() {
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }

    fn calculate_correlation_matrix(
        &self,
        symbol_results: &[SymbolResult],
        equity_curves: &[Vec<f64>],
    ) -> Vec<CorrelationPair> {
        let mut correlations = Vec::new();
        if equity_curves.len() < 2 {
            return correlations;
        }

        let returns: Vec<Vec<f64>> = equity_curves.iter()
            .map(|curve| {
                curve.windows(2)
                    .map(|w| if w[0] > 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
                    .collect()
            })
            .collect();

        for i in 0..symbol_results.len() {
            for j in (i + 1)..symbol_results.len() {
                let corr = Self::calculate_correlation(&returns[i], &returns[j]);
                correlations.push(CorrelationPair {
                    symbol1: symbol_results[i].symbol.clone(),
                    symbol2: symbol_results[j].symbol.clone(),
                    correlation: corr,
                });
            }
        }
        correlations
    }

    fn calculate_correlation(returns1: &[f64], returns2: &[f64]) -> f64 {
        let n = returns1.len().min(returns2.len());
        if n < 5 {
            return 0.0;
        }

        let mean1 = returns1[..n].iter().sum::<f64>() / n as f64;
        let mean2 = returns2[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..n {
            let d1 = returns1[i] - mean1;
            let d2 = returns2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom > 1e-10 { cov / denom } else { 0.0 }
    }

    fn empty_result(&self) -> PortfolioSimResult {
        PortfolioSimResult {
            symbol_results: Vec::new(),
            total_pnl: 0.0,
            portfolio_sharpe: 0.0,
            portfolio_max_drawdown: 0.0,
            total_trades: 0,
            portfolio_win_rate: 0.0,
            portfolio_profit_factor: 1.0,
            total_volume: 0.0,
            total_fees: 0.0,
            equity_curve: vec![self.initial_capital],
            trade_returns: Vec::new(),
            risk_metrics: None,
            initial_capital: self.initial_capital,
            final_equity: self.initial_capital,
            correlation_matrix: None,
            first_event_timestamp_ms: None,
            last_event_timestamp_ms: None,
        }
    }
}

/// Convert PortfolioSimResult to BacktestResult for API compatibility
impl From<PortfolioSimResult> for crate::types::BacktestResult {
    fn from(result: PortfolioSimResult) -> Self {
        use chrono::{TimeZone, Utc};

        crate::types::BacktestResult {
            total_pnl: result.total_pnl,
            max_drawdown: result.portfolio_max_drawdown,
            sharpe_ratio: Some(result.portfolio_sharpe),
            num_trades: result.total_trades,
            profit_factor: Some(result.portfolio_profit_factor),
            win_rate: Some(result.portfolio_win_rate),
            equity_curve: result.equity_curve,
            closed_trades: result.total_trades,
            trade_returns: result.trade_returns,
            initial_capital: result.initial_capital,
            first_trade_timestamp: result.first_event_timestamp_ms.map(|ms| Utc.timestamp_millis_opt(ms).unwrap()),
            last_trade_timestamp: result.last_event_timestamp_ms.map(|ms| Utc.timestamp_millis_opt(ms).unwrap()),
            transaction_costs: Some(crate::types::TransactionCostAnalysis {
                total_commission: result.total_fees,
                total_slippage: 0.0,
                total_market_impact: 0.0,
                total_transaction_costs: result.total_fees,
                costs_as_percentage_of_pnl: if result.total_pnl.abs() > 0.01 {
                    (result.total_fees / result.total_pnl.abs()) * 100.0
                } else {
                    0.0
                },
                avg_cost_per_trade: if result.total_trades > 0 {
                    result.total_fees / result.total_trades as f64
                } else {
                    0.0
                },
                commission_by_exchange: std::collections::HashMap::new(),
                total_gas_costs_usd: 0.0,
                avg_dex_slippage_bps: 0.0,
            }),
            inventory_metrics: Some(crate::types::InventoryMetrics {
                total_volume_traded: result.total_volume,
                avg_inventory_value: 0.0,
                turnover_per_day: 0.0,
                buy_fills: 0,
                sell_fills: 0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                imbalance_ratio: 0.0,
                starting_inventory: 0.0,
                starting_inventory_value: 0.0,
                starting_mark_price: 0.0,
                ending_inventory: 0.0,
                ending_inventory_value: 0.0,
                ending_mark_price: 0.0,
                starting_capital: result.initial_capital,
                ending_capital: result.final_equity,
                final_equity: result.final_equity,
            }),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_portfolio() {
        let fee_configs = HashMap::new();
        let sim = PortfolioSimulator::new(100_000.0, fee_configs);
        let candles: HashMap<String, CandleArrays> = HashMap::new();
        let signals: HashMap<String, Vec<i8>> = HashMap::new();
        let params: HashMap<String, AssetStrategyParams> = HashMap::new();

        let result = sim.run_candle_based(&candles, &signals, &params, None);

        assert_eq!(result.symbol_results.len(), 0);
        assert_eq!(result.initial_capital, 100_000.0);
        assert_eq!(result.final_equity, 100_000.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let r1 = vec![0.01, 0.02, -0.01, 0.03, -0.02];
        let r2 = vec![0.01, 0.02, -0.01, 0.03, -0.02];
        let corr = PortfolioSimulator::calculate_correlation(&r1, &r2);
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_asset_impact() {
        use crate::candle_sim::{CandleArrays, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_CLOSE};

        let fee_configs = HashMap::new();
        let mut sim = PortfolioSimulator::new(100_000.0, fee_configs);
        sim.cross_asset_config = CrossAssetImpactConfig {
            enabled: true,
            correlation_impact_bps: 10.0,
            threshold: 0.5,
        };

        // Two identical assets (correlation=1.0) with same signals → penalty applied
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let candles = CandleArrays {
            opens: closes.clone(),
            highs: closes.iter().map(|c| c * 1.01).collect(),
            lows: closes.iter().map(|c| c * 0.99).collect(),
            closes: closes.clone(),
            volumes: vec![1000.0; 20],
            timestamps_ms: (0..20i64).map(|i| i * 86_400_000).collect(),
        };

        let mut signals = vec![SIGNAL_HOLD; 20];
        signals[0] = SIGNAL_BUY;
        signals[19] = SIGNAL_CLOSE;

        let mut symbol_candles = HashMap::new();
        symbol_candles.insert("BTC".to_string(), candles.clone());
        symbol_candles.insert("ETH".to_string(), candles);

        let mut symbol_signals = HashMap::new();
        symbol_signals.insert("BTC".to_string(), signals.clone());
        symbol_signals.insert("ETH".to_string(), signals);

        let params = AssetStrategyParams { commission_rate: 0.0, slippage_bps: 0.0, ..Default::default() };
        let mut asset_params = HashMap::new();
        asset_params.insert("BTC".to_string(), params.clone());
        asset_params.insert("ETH".to_string(), params);

        let result_with_penalty = sim.run_candle_based(&symbol_candles, &symbol_signals, &asset_params, None);

        sim.cross_asset_config.enabled = false;
        let result_without_penalty = sim.run_candle_based(&symbol_candles, &symbol_signals, &asset_params, None);

        assert!(result_with_penalty.total_pnl < result_without_penalty.total_pnl,
            "cross-asset penalty should reduce PnL: {} vs {}", result_with_penalty.total_pnl, result_without_penalty.total_pnl);
    }
}
