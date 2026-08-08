//! Compatibility stubs for removed market-making simulation.
//!
//! These types exist solely to allow the program crate's distributed workers
//! to compile. The actual market-making simulation has been removed because
//! the data provider only supplies candle data, not tick-level orderbook data.
//!
//! TODO: Rewrite ga_worker.rs and distributed worker to use CandleSimulator.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEvent {
    pub timestamp_ms: i64,
    pub price: f64,
    pub quantity: f64,
    pub is_bid: bool,
    pub event_type: EventType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    Trade,
    Quote,
    Cancel,
    New,
    Modify,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FillModelType {
    Aggressive,
    Passive,
    Realistic,
}

impl FillModelType {
    pub fn for_exchange(_exchange: &str) -> Self {
        FillModelType::Realistic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealisticASParams {
    pub risk_aversion: f64,
    pub order_size: f64,
    pub order_size_pct: f64,
    pub initial_capital: f64,
    pub window_ms: f64,
    pub min_quote_lifetime_ms: f64,
    pub inventory_target: f64,
    pub volatility_cap: f64,
    pub time_horizon_minutes: f64,
    pub market_depth_k: f64,
    pub volatility_adjustment_factor: f64,
    pub inventory_penalty_factor: f64,
    pub trend_sensitivity: f64,
    pub flow_sensitivity: f64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub min_spread_bps: f64,
    pub max_trades_per_minute: f64,
    pub fee_rate: f64,
    pub initial_inventory_pct: f64,
    pub fill_model: FillModelType,
    pub initial_30d_volume: f64,
}

impl Default for RealisticASParams {
    fn default() -> Self {
        Self {
            risk_aversion: 0.3,
            order_size: 0.01,
            order_size_pct: 0.01,
            initial_capital: 100_000.0,
            window_ms: 60_000.0,
            min_quote_lifetime_ms: 500.0,
            inventory_target: 0.0,
            volatility_cap: 2.0,
            time_horizon_minutes: 5.0,
            market_depth_k: 10.0,
            volatility_adjustment_factor: 0.5,
            inventory_penalty_factor: 0.5,
            trend_sensitivity: 0.0,
            flow_sensitivity: 0.0,
            take_profit_pct: 0.0,
            stop_loss_pct: 0.0,
            min_spread_bps: 50.0,
            max_trades_per_minute: 10.0,
            fee_rate: 0.0004,
            initial_inventory_pct: 0.5,
            fill_model: FillModelType::Realistic,
            initial_30d_volume: 0.0,
        }
    }
}

impl RealisticASParams {
    pub fn from_param_map(params: &HashMap<String, serde_json::Value>) -> Self {
        let mut p = Self::default();
        if let Some(v) = params.get("risk_aversion").and_then(|v| v.as_f64()) { p.risk_aversion = v; }
        if let Some(v) = params.get("order_size").and_then(|v| v.as_f64()) { p.order_size = v; }
        if let Some(v) = params.get("order_size_pct").and_then(|v| v.as_f64()) { p.order_size_pct = v; }
        if let Some(v) = params.get("initial_capital").and_then(|v| v.as_f64()) { p.initial_capital = v; }
        if let Some(v) = params.get("window_ms").and_then(|v| v.as_f64()) { p.window_ms = v; }
        if let Some(v) = params.get("min_quote_lifetime_ms").and_then(|v| v.as_f64()) { p.min_quote_lifetime_ms = v; }
        if let Some(v) = params.get("inventory_target").and_then(|v| v.as_f64()) { p.inventory_target = v; }
        if let Some(v) = params.get("volatility_cap").and_then(|v| v.as_f64()) { p.volatility_cap = v; }
        if let Some(v) = params.get("time_horizon_minutes").and_then(|v| v.as_f64()) { p.time_horizon_minutes = v; }
        if let Some(v) = params.get("market_depth_k").and_then(|v| v.as_f64()) { p.market_depth_k = v; }
        if let Some(v) = params.get("fee_rate").and_then(|v| v.as_f64()) { p.fee_rate = v; }
        if let Some(v) = params.get("initial_inventory_pct").and_then(|v| v.as_f64()) { p.initial_inventory_pct = v; }
        if let Some(v) = params.get("initial_30d_volume").and_then(|v| v.as_f64()) { p.initial_30d_volume = v; }
        p
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub partial_fill_pct: f64,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct RealisticSimResult {
    pub total_pnl: f64,
    pub total_fees: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub equity_curve: Vec<f64>,
    pub trade_returns: Vec<f64>,
    pub total_volume: f64,
    pub avg_trade_pnl: f64,
    pub avg_fill_rate: f64,
    pub avg_time_in_trade_ms: f64,
    pub first_event_timestamp_ms: Option<i64>,
    pub last_event_timestamp_ms: Option<i64>,
    pub inventory_metrics: InventoryMetrics,
    pub best_trade_pnl: f64,
    pub worst_trade_pnl: f64,
    pub avg_queue_position: f64,
    pub execution_metrics: ExecutionMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct InventoryMetrics {
    pub starting_inventory: f64,
    pub starting_inventory_value: f64,
    pub starting_capital: f64,
    pub starting_mark_price: f64,
    pub ending_inventory: f64,
    pub ending_inventory_value: f64,
    pub ending_capital: f64,
    pub ending_mark_price: f64,
    pub final_equity: f64,
    pub avg_inventory: f64,
    pub max_inventory: f64,
    pub zero_crossings: usize,
}

pub struct RealisticSimulator {
    initial_capital: f64,
}

impl RealisticSimulator {
    pub fn with_exchange_config(initial_capital: f64, _fee_config: config::ExchangeFeeConfig) -> Self {
        Self { initial_capital }
    }

    pub fn run(&self, _events: &[OrderBookEvent], _params: &RealisticASParams) -> RealisticSimResult {
        RealisticSimResult {
            total_pnl: 0.0,
            total_fees: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            sharpe_ratio: 0.0,
            profit_factor: 0.0,
            max_drawdown: 0.0,
            equity_curve: vec![self.initial_capital],
            trade_returns: Vec::new(),
            total_volume: 0.0,
            avg_trade_pnl: 0.0,
            avg_fill_rate: 0.0,
            avg_time_in_trade_ms: 0.0,
            first_event_timestamp_ms: None,
            last_event_timestamp_ms: None,
            inventory_metrics: InventoryMetrics {
                starting_capital: self.initial_capital,
                ending_capital: self.initial_capital,
                final_equity: self.initial_capital,
                ..Default::default()
            },
            best_trade_pnl: 0.0,
            worst_trade_pnl: 0.0,
            avg_queue_position: 0.0,
            execution_metrics: ExecutionMetrics::default(),
        }
    }
}

pub struct VolumeTracker;
impl VolumeTracker {
    pub fn new() -> Self { Self }
}

pub fn ticks_to_trade_events(_data: &[data_prep::SimulationTick]) -> Vec<OrderBookEvent> {
    Vec::new()
}

pub fn market_data_to_order_events(_data: &[dataloader::MarketData]) -> Vec<OrderBookEvent> {
    Vec::new()
}

impl From<&data_prep::orderbook_binary::OrderBookEventTick> for OrderBookEvent {
    fn from(tick: &data_prep::orderbook_binary::OrderBookEventTick) -> Self {
        use data_prep::orderbook_binary::{EventTypeCode, SideCode};
        let event_type = if tick.event_type == EventTypeCode::TRADE {
            EventType::Trade
        } else if tick.event_type == EventTypeCode::NEW {
            EventType::New
        } else if tick.event_type == EventTypeCode::MODIFY {
            EventType::Modify
        } else if tick.event_type == EventTypeCode::CANCEL {
            EventType::Cancel
        } else {
            EventType::Quote
        };
        OrderBookEvent {
            timestamp_ms: tick.timestamp_ms,
            price: tick.price,
            quantity: tick.quantity,
            is_bid: tick.side == SideCode::BUY,
            event_type,
        }
    }
}

impl From<RealisticSimResult> for crate::types::BacktestResult {
    fn from(r: RealisticSimResult) -> Self {
        crate::types::BacktestResult {
            total_pnl: r.total_pnl,
            num_trades: r.num_trades,
            closed_trades: r.num_trades,
            open_positions: 0,
            volume_constrained_pct: None,
            first_trade_timestamp: None,
            last_trade_timestamp: None,
            first_trade_price: None,
            last_trade_price: None,
            realized_pnl: Some(r.total_pnl),
            unrealized_pnl: Some(0.0),
            max_drawdown: r.max_drawdown,
            sharpe_ratio: Some(r.sharpe_ratio),
            sortino_ratio: None,
            calmar_ratio: None,
            profit_factor: Some(r.profit_factor),
            win_rate: Some(r.win_rate),
            avg_trade_return: Some(r.avg_trade_pnl),
            median_trade_return: None,
            avg_trade_duration: None,
            volatility: None,
            gross_profit: None,
            gross_loss: None,
            net_profit: Some(r.total_pnl),
            equity_curve: r.equity_curve,
            equity_curve_timestamps: Vec::new(),
            trade_returns: r.trade_returns,
            trade_pct_returns: Vec::new(),
            trade_maes: Vec::new(),
            initial_capital: r.inventory_metrics.starting_capital,
            data_quality: None,
            transaction_costs: None,
            market_impact: None,
            inventory_metrics: None,
            execution_metrics: None,
            risk_halt_reason: None,
            sharpe_t_stat: None,
            significance_pvalue: None,
            deflated_sharpe: None,
            ulcer_index: None,
            pain_index: None,
            cdar_95: None,
            trade_log: Vec::new(),
            price_series: Vec::new(),
            signal_counts: None,
            compute_signals_error: None,
            candle_interval_minutes: None,
            python_source_code: None,
            strategy_display_name: None,
            lp_metrics: None,
            analysis_mode: None,
            ga_trial_count: None,
            ga_is_sharpe_mean: None,
            ga_is_sharpe_std: None,
            ga_is_sharpe_pct_above_threshold: None,
            analysis_stages_run: Vec::new(),
            feature_diagnostics: None,
        }
    }
}
