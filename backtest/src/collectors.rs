//! MetricCollector trait and concrete collector implementations.
//!
//! Decomposes the 44+ metric fields from `SimulationLoop` into pluggable,
//! independently-testable collectors. Use `new_lean()` for GA inner-loop
//! (CorePnl only) or `new_full()` for final evaluation (all collectors).

use chrono::{DateTime, Utc};
use orderbook::BookSide;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use crate::types::{
    TransactionCostAnalysis, MarketImpactAnalysis, LiquidityWarning,
    InventoryMetrics, ExecutionMetrics,
};

// ─── Event payloads ─────────────────────────────────────────────────────────

/// A fill event that collectors can observe.
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub order_id: Arc<str>,
    pub side: BookSide,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub slippage: f64,
    pub exchange: Arc<str>,
    pub fee_tier_bps: u32,
    pub timestamp: DateTime<Utc>,
}

/// An order-placement event.
#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub order_id: Arc<str>,
    pub side: BookSide,
    pub price: f64,
    pub spread_bps: f64,
    pub timestamp: DateTime<Utc>,
}

/// A market-data trade event.
#[derive(Debug, Clone)]
pub struct MarketTradeEvent {
    pub price: f64,
    pub quantity: f64,
    pub side: BookSide,
    pub timestamp: DateTime<Utc>,
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Pluggable metric collector. Implement this trait to add new metric
/// categories without touching the core simulation loop.
pub trait MetricCollector: Send {
    /// Called for each fill executed by the simulation.
    fn on_fill(&mut self, _event: &FillEvent) {}

    /// Called when an order is placed on the book.
    fn on_order_placed(&mut self, _event: &OrderEvent) {}

    /// Called when an order is cancelled.
    fn on_order_cancelled(&mut self, _order_id: &str, _timestamp: DateTime<Utc>) {}

    /// Called for each market trade tick (for impact/opportunity tracking).
    fn on_market_trade(&mut self, _event: &MarketTradeEvent) {}

    /// Called periodically (e.g. every N ticks) with the current inventory position.
    fn on_inventory_sample(&mut self, _net_position: f64, _timestamp: DateTime<Utc>) {}

    /// Called with the dollar value of the current inventory (position value, not cash).
    fn on_inventory_value(&mut self, _value: f64) {}

    /// Called when a latency rejection occurs.
    fn on_latency_rejection(&mut self, _side: BookSide) {}

    /// Called when a potential fill is missed (price gap).
    fn on_missed_fill(&mut self, _side: BookSide, _price_gap: f64) {}

    /// Populate the relevant fields of `BacktestResult` with collected metrics.
    fn finalize(&self, result: &mut crate::types::BacktestResult);

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;

    /// Downcast support for reading accumulated values from concrete collectors.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Mutable downcast support for writing to concrete collectors.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Downcast a collector to a concrete type for reading accumulated values.
pub fn get_collector<T: 'static>(collectors: &[Box<dyn MetricCollector>]) -> Option<&T> {
    collectors.iter().find_map(|c| c.as_any().downcast_ref::<T>())
}

/// Downcast a collector to a concrete type for mutable access.
pub fn get_collector_mut<T: 'static>(collectors: &mut [Box<dyn MetricCollector>]) -> Option<&mut T> {
    collectors.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<T>())
}

// ─── 1. TransactionCostCollector ────────────────────────────────────────────

/// Tracks commission, slippage, and market impact.
pub struct TransactionCostCollector {
    pub total_commission: f64,
    pub total_slippage: f64,
    pub total_market_impact: f64,
    pub commission_by_exchange: FxHashMap<String, f64>,
    pub commission_by_tier_bps: FxHashMap<u32, f64>,
    pub fills_by_tier_bps: FxHashMap<u32, u32>,
    pub trades_with_impact: usize,
    pub impact_costs_bps: Vec<f64>,
    pub liquidity_warnings: Vec<LiquidityWarning>,
}

impl TransactionCostCollector {
    pub fn new() -> Self {
        Self {
            total_commission: 0.0,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            commission_by_exchange: FxHashMap::default(),
            commission_by_tier_bps: FxHashMap::default(),
            fills_by_tier_bps: FxHashMap::default(),
            trades_with_impact: 0,
            impact_costs_bps: Vec::with_capacity(1000),
            liquidity_warnings: Vec::with_capacity(100),
        }
    }

    /// Record market impact for a given order size and market conditions.
    pub fn record_market_impact(&mut self, impact_bps: f64, price: f64, order_size: f64,
                                 rolling_avg_volume: f64, timestamp: DateTime<Utc>) {
        self.trades_with_impact += 1;
        self.impact_costs_bps.push(impact_bps);

        let size_ratio = if rolling_avg_volume > 0.0 { order_size / rolling_avg_volume } else { 0.0 };
        if size_ratio > 0.1 {
            self.liquidity_warnings.push(LiquidityWarning {
                timestamp,
                order_size,
                available_liquidity: rolling_avg_volume,
                liquidity_ratio: size_ratio,
                message: format!("Order size is {:.1}% of available volume - high market impact expected",
                                 size_ratio * 100.0),
            });
        }

        let impact_cost = (impact_bps / 10000.0) * price * order_size;
        self.total_market_impact += impact_cost;
    }
}

impl MetricCollector for TransactionCostCollector {
    fn on_fill(&mut self, event: &FillEvent) {
        self.total_commission += event.commission;
        self.total_slippage += event.slippage;
        *self.commission_by_exchange.entry(event.exchange.to_string()).or_default() += event.commission;
        *self.commission_by_tier_bps.entry(event.fee_tier_bps).or_default() += event.commission;
        *self.fills_by_tier_bps.entry(event.fee_tier_bps).or_default() += 1;
    }

    fn finalize(&self, result: &mut crate::types::BacktestResult) {
        let total = self.total_commission + self.total_slippage + self.total_market_impact;
        let pct = if result.total_pnl.abs() > 0.0 { (total / result.total_pnl.abs()) * 100.0 } else { 0.0 };

        result.transaction_costs = Some(TransactionCostAnalysis {
            total_commission: self.total_commission,
            total_slippage: self.total_slippage,
            total_market_impact: self.total_market_impact,
            total_transaction_costs: total,
            costs_as_percentage_of_pnl: pct,
            avg_cost_per_trade: if result.num_trades > 0 { total / result.num_trades as f64 } else { 0.0 },
            commission_by_exchange: self.commission_by_exchange.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            total_gas_costs_usd: 0.0,
            avg_dex_slippage_bps: 0.0,
        });

        result.market_impact = Some(MarketImpactAnalysis {
            trades_with_impact: self.trades_with_impact,
            avg_impact_bps: if !self.impact_costs_bps.is_empty() {
                self.impact_costs_bps.iter().sum::<f64>() / self.impact_costs_bps.len() as f64
            } else { 0.0 },
            max_impact_bps: self.impact_costs_bps.iter().cloned().fold(0.0, f64::max),
            total_impact_cost: self.total_market_impact,
            liquidity_warnings: self.liquidity_warnings.clone(),
        });
    }

    fn name(&self) -> &'static str { "TransactionCost" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── 2. InventoryCollector ──────────────────────────────────────────────────

/// Tracks buy/sell volumes, fills, and inventory value samples.
pub struct InventoryCollector {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub buy_fills: usize,
    pub sell_fills: usize,
    pub buy_qty: f64,
    pub sell_qty: f64,
    pub inventory_value_samples: Vec<f64>,
    pub inventory_snapshots: Vec<(DateTime<Utc>, f64)>,
    pub max_inventory: f64,
    pub min_inventory: f64,
    pub market_buy_trades: usize,
    pub market_sell_trades: usize,
}

impl InventoryCollector {
    pub fn new() -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            buy_fills: 0,
            sell_fills: 0,
            buy_qty: 0.0,
            sell_qty: 0.0,
            inventory_value_samples: Vec::with_capacity(1000),
            inventory_snapshots: Vec::with_capacity(1000),
            max_inventory: 0.0,
            min_inventory: 0.0,
            market_buy_trades: 0,
            market_sell_trades: 0,
        }
    }
}

impl MetricCollector for InventoryCollector {
    fn on_fill(&mut self, event: &FillEvent) {
        let trade_value = event.price * event.quantity;
        match event.side {
            BookSide::Bid => {
                self.buy_volume += trade_value;
                self.buy_fills += 1;
                self.buy_qty += event.quantity;
            }
            BookSide::Ask => {
                self.sell_volume += trade_value;
                self.sell_fills += 1;
                self.sell_qty += event.quantity;
            }
        }
    }

    fn on_market_trade(&mut self, event: &MarketTradeEvent) {
        match event.side {
            BookSide::Bid => self.market_buy_trades += 1,
            BookSide::Ask => self.market_sell_trades += 1,
        }
    }

    fn on_inventory_sample(&mut self, net_position: f64, timestamp: DateTime<Utc>) {
        self.inventory_snapshots.push((timestamp, net_position));
        if net_position > self.max_inventory { self.max_inventory = net_position; }
        if net_position < self.min_inventory { self.min_inventory = net_position; }
    }

    fn on_inventory_value(&mut self, value: f64) {
        if value > 0.0 {
            self.inventory_value_samples.push(value);
        }
    }

    fn finalize(&self, result: &mut crate::types::BacktestResult) {
        let total_volume = self.buy_volume + self.sell_volume;
        let avg_inventory = if !self.inventory_value_samples.is_empty() {
            self.inventory_value_samples.iter().sum::<f64>() / self.inventory_value_samples.len() as f64
        } else {
            result.initial_capital * 0.5
        };

        // Derive elapsed days from actual inventory snapshot timestamps when
        // available; fall back to a rough data-point heuristic otherwise.
        let estimated_days = match (self.inventory_snapshots.first(), self.inventory_snapshots.last()) {
            (Some((first_ts, _)), Some((last_ts, _))) if last_ts > first_ts => {
                ((*last_ts - *first_ts).num_seconds() as f64 / 86_400.0).max(1.0 / 24.0)
            }
            _ => (result.equity_curve.len() as f64 / 5000.0).max(1.0),
        };
        let turnover = if avg_inventory > 0.0 {
            total_volume / avg_inventory / estimated_days
        } else { 0.0 };

        let imbalance = if total_volume > 0.0 {
            (self.buy_volume - self.sell_volume) / total_volume
        } else { 0.0 };

        result.inventory_metrics = Some(InventoryMetrics {
            total_volume_traded: total_volume,
            avg_inventory_value: avg_inventory,
            turnover_per_day: turnover,
            buy_fills: self.buy_fills,
            sell_fills: self.sell_fills,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
            imbalance_ratio: imbalance,
            starting_inventory: 0.0,
            starting_inventory_value: 0.0,
            starting_mark_price: 0.0,
            ending_inventory: 0.0,
            ending_inventory_value: 0.0,
            ending_mark_price: 0.0,
            starting_capital: 0.0,
            ending_capital: 0.0,
            final_equity: 0.0,
        });
    }

    fn name(&self) -> &'static str { "Inventory" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── 3. ExecutionQualityCollector ───────────────────────────────────────────

/// Tracks order lifecycle: placement, fills, cancellation, latency.
pub struct ExecutionQualityCollector {
    pub order_trackers: FxHashMap<String, OrderTracker>,
    pub orders_cancelled: usize,
    pub time_to_fill_samples: Vec<f64>,
    pub bid_orders_placed: usize,
    pub ask_orders_placed: usize,
    pub quote_spreads_bps: Vec<f64>,
    pub latency_rejections: usize,
    pub latency_rejections_bid: usize,
    pub latency_rejections_ask: usize,
    pub fill_count_histogram: FxHashMap<u32, u32>,
    pub total_fill_count: u32,
    pub total_order_count: u32,
}

/// Order lifecycle tracker (moved from simulation.rs)
pub struct OrderTracker {
    pub order_id: String,
    pub side: BookSide,
    pub placed_at: DateTime<Utc>,
    pub price: f64,
    pub filled: bool,
    pub fill_count: usize,
    pub first_fill_at: Option<DateTime<Utc>>,
}

impl ExecutionQualityCollector {
    pub fn new() -> Self {
        Self {
            order_trackers: FxHashMap::default(),
            orders_cancelled: 0,
            time_to_fill_samples: Vec::with_capacity(1000),
            bid_orders_placed: 0,
            ask_orders_placed: 0,
            quote_spreads_bps: Vec::with_capacity(1000),
            latency_rejections: 0,
            latency_rejections_bid: 0,
            latency_rejections_ask: 0,
            fill_count_histogram: FxHashMap::default(),
            total_fill_count: 0,
            total_order_count: 0,
        }
    }

    /// Track a fill on an existing order.
    pub fn track_fill(&mut self, order_id: &str, timestamp: DateTime<Utc>) {
        if let Some(tracker) = self.order_trackers.get_mut(order_id) {
            tracker.fill_count += 1;
            if !tracker.filled {
                tracker.filled = true;
                tracker.first_fill_at = Some(timestamp);
                let ttf = (timestamp - tracker.placed_at).num_milliseconds() as f64 / 1000.0;
                self.time_to_fill_samples.push(ttf);
            }
        }
        self.total_fill_count += 1;
    }
}

impl MetricCollector for ExecutionQualityCollector {
    fn on_order_placed(&mut self, event: &OrderEvent) {
        match event.side {
            BookSide::Bid => self.bid_orders_placed += 1,
            BookSide::Ask => self.ask_orders_placed += 1,
        }
        self.total_order_count += 1;
        self.order_trackers.insert(event.order_id.to_string(), OrderTracker {
            order_id: event.order_id.to_string(),
            side: event.side.clone(),
            placed_at: event.timestamp,
            price: event.price,
            filled: false,
            fill_count: 0,
            first_fill_at: None,
        });
        self.quote_spreads_bps.push(event.spread_bps);
    }

    fn on_order_cancelled(&mut self, order_id: &str, _timestamp: DateTime<Utc>) {
        self.orders_cancelled += 1;
        // Record fill-count histogram when order ends
        if let Some(tracker) = self.order_trackers.get(order_id) {
            *self.fill_count_histogram.entry(tracker.fill_count as u32).or_default() += 1;
        }
    }

    fn on_latency_rejection(&mut self, side: BookSide) {
        self.latency_rejections += 1;
        match side {
            BookSide::Bid => self.latency_rejections_bid += 1,
            BookSide::Ask => self.latency_rejections_ask += 1,
        }
    }

    fn finalize(&self, result: &mut crate::types::BacktestResult) {
        let orders_placed = self.bid_orders_placed + self.ask_orders_placed;
        let orders_filled = self.order_trackers.values().filter(|t| t.filled).count();
        let orders_with_fills = self.order_trackers.values().filter(|t| t.fill_count > 0).count();
        let total_fills: usize = self.order_trackers.values().map(|t| t.fill_count).sum();

        let fill_rate = if orders_placed > 0 { orders_filled as f64 / orders_placed as f64 } else { 0.0 };
        let cancel_rate = if orders_placed > 0 { self.orders_cancelled as f64 / orders_placed as f64 } else { 0.0 };
        let avg_fills = if orders_with_fills > 0 { total_fills as f64 / orders_with_fills as f64 } else { 0.0 };

        let avg_ttf = if !self.time_to_fill_samples.is_empty() {
            self.time_to_fill_samples.iter().sum::<f64>() / self.time_to_fill_samples.len() as f64
        } else { 0.0 };

        let median_ttf = if !self.time_to_fill_samples.is_empty() {
            let mut sorted = self.time_to_fill_samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 && sorted.len() > 1 { (sorted[mid-1] + sorted[mid]) / 2.0 } else { sorted[mid] }
        } else { 0.0 };

        let bid_fills = self.order_trackers.values().filter(|t| t.filled && matches!(t.side, BookSide::Bid)).count();
        let ask_fills = self.order_trackers.values().filter(|t| t.filled && matches!(t.side, BookSide::Ask)).count();
        let bid_fill_rate = if self.bid_orders_placed > 0 { bid_fills as f64 / self.bid_orders_placed as f64 } else { 0.0 };
        let ask_fill_rate = if self.ask_orders_placed > 0 { ask_fills as f64 / self.ask_orders_placed as f64 } else { 0.0 };

        let avg_spread = if !self.quote_spreads_bps.is_empty() {
            self.quote_spreads_bps.iter().sum::<f64>() / self.quote_spreads_bps.len() as f64
        } else { 0.0 };

        result.execution_metrics = Some(ExecutionMetrics {
            orders_placed,
            orders_filled,
            orders_cancelled: self.orders_cancelled,
            total_fills,
            fill_rate,
            cancellation_rate: cancel_rate,
            avg_fills_per_order: avg_fills,
            avg_time_to_fill_seconds: avg_ttf,
            median_time_to_fill_seconds: median_ttf,
            bid_orders: self.bid_orders_placed,
            ask_orders: self.ask_orders_placed,
            bid_fills,
            ask_fills,
            bid_fill_rate,
            ask_fill_rate,
            avg_quote_spread_bps: avg_spread,
            // Round-trip metrics filled by RoundTripCollector
            round_trip_win_rate: None,
            total_round_trips: 0,
            profitable_round_trips: 0,
            spread_capture_rate: None,
            adverse_selection_rate: None,
            avg_pnl_per_round_trip: None,
        });
    }

    fn name(&self) -> &'static str { "ExecutionQuality" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── 4. MissedOpportunityCollector ──────────────────────────────────────────

/// Tracks missed fills and price gaps for fill-rate diagnostics.
pub struct MissedOpportunityCollector {
    pub missed_bid_opportunities: usize,
    pub missed_ask_opportunities: usize,
    pub bid_price_gap_sum: f64,
    pub ask_price_gap_sum: f64,
}

impl MissedOpportunityCollector {
    pub fn new() -> Self {
        Self {
            missed_bid_opportunities: 0,
            missed_ask_opportunities: 0,
            bid_price_gap_sum: 0.0,
            ask_price_gap_sum: 0.0,
        }
    }
}

impl MetricCollector for MissedOpportunityCollector {
    fn on_missed_fill(&mut self, side: BookSide, price_gap: f64) {
        match side {
            BookSide::Bid => {
                self.missed_bid_opportunities += 1;
                self.bid_price_gap_sum += price_gap;
            }
            BookSide::Ask => {
                self.missed_ask_opportunities += 1;
                self.ask_price_gap_sum += price_gap;
            }
        }
    }

    fn finalize(&self, _result: &mut crate::types::BacktestResult) {
        // Missed opportunity data is logged but not stored in BacktestResult
        // (diagnostic only). Could be extended with a new field if needed.
    }

    fn name(&self) -> &'static str { "MissedOpportunity" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── 5. RoundTripCollector ──────────────────────────────────────────────────

/// Tracks buy/sell fill prices for round-trip profitability analysis.
pub struct RoundTripCollector {
    pub buy_fill_prices: Vec<f64>,
    pub sell_fill_prices: Vec<f64>,
    pub fee_rate: f64,
}

impl RoundTripCollector {
    pub fn new(fee_rate: f64) -> Self {
        Self {
            buy_fill_prices: Vec::with_capacity(1000),
            sell_fill_prices: Vec::with_capacity(1000),
            fee_rate,
        }
    }
}

impl MetricCollector for RoundTripCollector {
    fn on_fill(&mut self, event: &FillEvent) {
        match event.side {
            BookSide::Bid => self.buy_fill_prices.push(event.price),
            BookSide::Ask => self.sell_fill_prices.push(event.price),
        }
    }

    fn finalize(&self, result: &mut crate::types::BacktestResult) {
        let total_rt = self.buy_fill_prices.len().min(self.sell_fill_prices.len());
        if total_rt == 0 { return; }

        let mut profitable = 0_usize;
        let mut total_pnl = 0.0_f64;
        for i in 0..total_rt {
            let buy = self.buy_fill_prices[i];
            let sell = self.sell_fill_prices[i];
            let net = (sell - buy) - self.fee_rate * (buy + sell);
            total_pnl += net;
            if net > 0.0 { profitable += 1; }
        }

        let avg_price = (self.buy_fill_prices.iter().sum::<f64>()
            + self.sell_fill_prices.iter().sum::<f64>())
            / (self.buy_fill_prices.len() + self.sell_fill_prices.len()) as f64;

        // Merge into execution_metrics (or create if missing)
        let em = result.execution_metrics.get_or_insert_with(ExecutionMetrics::default);
        em.round_trip_win_rate = Some(profitable as f64 / total_rt as f64);
        em.total_round_trips = total_rt;
        em.profitable_round_trips = profitable;
        em.avg_pnl_per_round_trip = Some(total_pnl / total_rt as f64);

        let avg_spread = em.avg_quote_spread_bps;
        em.spread_capture_rate = if avg_spread > 0.0 && avg_price > 0.0 {
            let captured = (total_pnl / total_rt as f64) / avg_price * 10000.0;
            Some((captured / avg_spread).clamp(0.0, 2.0))
        } else { None };

        em.adverse_selection_rate = Some((total_rt - profitable) as f64 / total_rt as f64);
    }

    fn name(&self) -> &'static str { "RoundTrip" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── 6. TimestampCollector ──────────────────────────────────────────────────

/// Tracks first/last trade timestamps and prices.
pub struct TimestampCollector {
    pub first_trade_timestamp: Option<DateTime<Utc>>,
    pub last_trade_timestamp: Option<DateTime<Utc>>,
    pub first_trade_price: Option<f64>,
    pub last_trade_price: Option<f64>,
}

impl TimestampCollector {
    pub fn new() -> Self {
        Self {
            first_trade_timestamp: None,
            last_trade_timestamp: None,
            first_trade_price: None,
            last_trade_price: None,
        }
    }
}

impl MetricCollector for TimestampCollector {
    fn on_market_trade(&mut self, event: &MarketTradeEvent) {
        if self.first_trade_timestamp.is_none() {
            self.first_trade_timestamp = Some(event.timestamp);
            self.first_trade_price = Some(event.price);
        }
        self.last_trade_timestamp = Some(event.timestamp);
        self.last_trade_price = Some(event.price);
    }

    fn finalize(&self, result: &mut crate::types::BacktestResult) {
        result.first_trade_timestamp = self.first_trade_timestamp;
        result.last_trade_timestamp = self.last_trade_timestamp;
        result.first_trade_price = self.first_trade_price;
        result.last_trade_price = self.last_trade_price;
    }

    fn name(&self) -> &'static str { "Timestamp" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── Collector Set helpers ──────────────────────────────────────────────────

/// Build the lean set of collectors (GA inner loop — minimal overhead).
pub fn lean_collectors() -> Vec<Box<dyn MetricCollector>> {
    vec![
        Box::new(TimestampCollector::new()),
    ]
}

/// Build the full set of collectors (final evaluation — all diagnostics).
pub fn full_collectors(fee_rate: f64) -> Vec<Box<dyn MetricCollector>> {
    vec![
        Box::new(TransactionCostCollector::new()),
        Box::new(InventoryCollector::new()),
        Box::new(ExecutionQualityCollector::new()),
        Box::new(MissedOpportunityCollector::new()),
        Box::new(RoundTripCollector::new(fee_rate)),
        Box::new(TimestampCollector::new()),
    ]
}

/// Dispatch a fill event to all collectors.
#[inline]
pub fn dispatch_fill(collectors: &mut [Box<dyn MetricCollector>], event: &FillEvent) {
    for c in collectors.iter_mut() { c.on_fill(event); }
}

/// Dispatch an order-placed event to all collectors.
#[inline]
pub fn dispatch_order_placed(collectors: &mut [Box<dyn MetricCollector>], event: &OrderEvent) {
    for c in collectors.iter_mut() { c.on_order_placed(event); }
}

/// Dispatch an order-cancelled event to all collectors.
#[inline]
pub fn dispatch_order_cancelled(collectors: &mut [Box<dyn MetricCollector>], order_id: &str, ts: DateTime<Utc>) {
    for c in collectors.iter_mut() { c.on_order_cancelled(order_id, ts); }
}

/// Dispatch a market-trade event to all collectors.
#[inline]
pub fn dispatch_market_trade(collectors: &mut [Box<dyn MetricCollector>], event: &MarketTradeEvent) {
    for c in collectors.iter_mut() { c.on_market_trade(event); }
}

/// Dispatch a latency-rejection event to all collectors.
#[inline]
pub fn dispatch_latency_rejection(collectors: &mut [Box<dyn MetricCollector>], side: BookSide) {
    for c in collectors.iter_mut() { c.on_latency_rejection(side.clone()); }
}

/// Dispatch a missed-fill event to all collectors.
#[inline]
pub fn dispatch_missed_fill(collectors: &mut [Box<dyn MetricCollector>], side: BookSide, gap: f64) {
    for c in collectors.iter_mut() { c.on_missed_fill(side.clone(), gap); }
}

/// Finalize all collectors, populating the BacktestResult.
pub fn finalize_all(collectors: &[Box<dyn MetricCollector>], result: &mut crate::types::BacktestResult) {
    for c in collectors { c.finalize(result); }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_fill(side: BookSide, price: f64, qty: f64) -> FillEvent {
        FillEvent {
            order_id: "test-1".into(),
            side,
            price,
            quantity: qty,
            commission: price * qty * 0.001,
            slippage: 0.0,
            exchange: "massive".into(),
            fee_tier_bps: 10,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_transaction_cost_collector() {
        let mut tc = TransactionCostCollector::new();
        tc.on_fill(&sample_fill(BookSide::Bid, 50000.0, 0.1));
        tc.on_fill(&sample_fill(BookSide::Ask, 50100.0, 0.1));
        assert!(tc.total_commission > 0.0);
        assert_eq!(tc.commission_by_exchange.len(), 1);
    }

    #[test]
    fn test_inventory_collector() {
        let mut ic = InventoryCollector::new();
        ic.on_fill(&sample_fill(BookSide::Bid, 50000.0, 0.5));
        ic.on_fill(&sample_fill(BookSide::Ask, 50100.0, 0.3));
        assert_eq!(ic.buy_fills, 1);
        assert_eq!(ic.sell_fills, 1);
        assert!((ic.buy_qty - 0.5).abs() < 1e-10);
        assert!((ic.sell_qty - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip_collector() {
        let mut rt = RoundTripCollector::new(0.0001); // 1 bps fee
        // Buy at 50000, sell at 50200 → spread=200, fees≈10 → profitable
        rt.on_fill(&sample_fill(BookSide::Bid, 50000.0, 0.1));
        rt.on_fill(&sample_fill(BookSide::Ask, 50200.0, 0.1));

        let mut result = crate::types::BacktestResult::default();
        result.execution_metrics = Some(ExecutionMetrics::default());
        rt.finalize(&mut result);

        let em = result.execution_metrics.unwrap();
        assert_eq!(em.total_round_trips, 1);
        assert_eq!(em.profitable_round_trips, 1);
    }

    #[test]
    fn test_timestamp_collector() {
        let mut ts = TimestampCollector::new();
        let now = Utc::now();
        ts.on_market_trade(&MarketTradeEvent {
            price: 50000.0, quantity: 1.0, side: BookSide::Bid, timestamp: now,
        });
        assert_eq!(ts.first_trade_price, Some(50000.0));
        assert_eq!(ts.last_trade_price, Some(50000.0));
    }

    #[test]
    fn test_missed_opportunity_collector() {
        let mut mc = MissedOpportunityCollector::new();
        mc.on_missed_fill(BookSide::Bid, 5.0);
        mc.on_missed_fill(BookSide::Bid, 3.0);
        mc.on_missed_fill(BookSide::Ask, 2.0);
        assert_eq!(mc.missed_bid_opportunities, 2);
        assert_eq!(mc.missed_ask_opportunities, 1);
        assert!((mc.bid_price_gap_sum - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_execution_quality_collector() {
        let mut eq = ExecutionQualityCollector::new();
        let now = Utc::now();
        eq.on_order_placed(&OrderEvent {
            order_id: "o1".into(),
            side: BookSide::Bid,
            price: 50000.0,
            spread_bps: 10.0,
            timestamp: now,
        });
        assert_eq!(eq.bid_orders_placed, 1);
        assert_eq!(eq.total_order_count, 1);

        // Simulate a fill
        eq.track_fill("o1", now + chrono::Duration::milliseconds(500));
        assert_eq!(eq.total_fill_count, 1);
        assert_eq!(eq.time_to_fill_samples.len(), 1);
    }

    #[test]
    fn test_lean_vs_full_collectors() {
        let lean = lean_collectors();
        let full = full_collectors(0.001);
        assert_eq!(lean.len(), 1, "lean should have 1 collector (Timestamp)");
        assert_eq!(full.len(), 6, "full should have 6 collectors");
    }

    #[test]
    fn test_dispatch_fill() {
        let mut collectors = full_collectors(0.001);
        let fill = sample_fill(BookSide::Bid, 50000.0, 0.1);
        dispatch_fill(&mut collectors, &fill);
        // Should not panic — verifies dispatch works for all collectors.
    }

    #[test]
    fn test_finalize_all() {
        let mut collectors = full_collectors(0.001);
        let fill = sample_fill(BookSide::Bid, 50000.0, 0.1);
        dispatch_fill(&mut collectors, &fill);

        let mut result = crate::types::BacktestResult::default();
        finalize_all(&collectors, &mut result);

        assert!(result.transaction_costs.is_some());
        assert!(result.inventory_metrics.is_some());
        assert!(result.execution_metrics.is_some());
    }
}
