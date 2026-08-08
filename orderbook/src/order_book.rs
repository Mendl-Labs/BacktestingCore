use crate::order_utils;
// Utility for converting Signal to OrderBookEvent moved to order_utils.rs

/// Core order book implementation
///
/// This module provides the main OrderBook structure that manages bid and ask sides
/// and handles all order book operations.

use crate::level::Level;
use crate::types::{
    BookSide, OrderBookEvent, OrderInfo, SnapshotLevel, LiquidityType, Fill,
    VolumeEntry, SlippageConfig, ExecutionLevel, MarketOrderExecution,
};
use crate::{DepthLevel, MarketDepth, OrderBookConfig};
use config::ExchangeFeeConfig;
use crate::logging_facade::ORDERBOOK_LOGGER;
use crate::{log_debug, log_warn};
use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Main order book structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol being traded
    pub symbol: String,
    /// Exchange name
    pub exchange: String,
    /// Bid side (buy orders) - ordered by price descending (highest first)
    pub bids: BTreeMap<OrderedFloat<f64>, Level>,
    /// Ask side (sell orders) - ordered by price ascending (lowest first)
    pub asks: BTreeMap<OrderedFloat<f64>, Level>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Configuration
    config: OrderBookConfig,
    /// Event sequence number
    pub sequence_number: u64,
    /// Fee configuration for this exchange
    pub fee_config: Option<ExchangeFeeConfig>,
    /// Rolling 30-day volume tracking
    pub volume_history: VecDeque<VolumeEntry>,
    /// Current 30-day volume in USD
    pub volume_30d: f64,
    /// Slippage configuration
    pub slippage_config: SlippageConfig,
}

impl OrderBook {

    /// Submit an order from a Signal (convenience for simulation)
    pub fn submit_order_from_signal(&mut self, signal: &signal::Signal, timestamp: DateTime<Utc>) -> anyhow::Result<()> {
        // Use the utility function from order_utils
        let event = order_utils::signal_to_order_event(signal, timestamp);
        self.apply_event(event)
    }

    /// Match all possible orders (comprehensive matching engine)
    /// Returns fills with proper maker/taker fee calculations
    pub fn match_all_orders(&mut self, timestamp: DateTime<Utc>) -> Vec<Fill> {
        let mut fills = Vec::new();
        
        while self.is_crossed() {
            // Get the best bid and ask price keys
            let best_bid_key = self.bids.keys().next_back().copied();
            let best_ask_key = self.asks.keys().next().copied();
            
            if let (Some(bid_key), Some(ask_key)) = (best_bid_key, best_ask_key) {
                // Get the matching price (use the ask price as the execution price)
                let execution_price = self.asks.get(&ask_key).unwrap().price;
                
                // Calculate the maximum quantity that can be matched
                let bid_quantity = self.bids.get(&bid_key).unwrap().total_quantity;
                let ask_quantity = self.asks.get(&ask_key).unwrap().total_quantity;
                let match_quantity = bid_quantity.min(ask_quantity);
                
                if match_quantity <= 0.0 {
                    break;
                }
                
                // Execute trade on ask side (they are MAKERS - their limit orders were resting)
                let ask_executed_orders = {
                    let ask_level = self.asks.get_mut(&ask_key).unwrap();
                    ask_level.execute_trade(match_quantity, timestamp).unwrap_or_default()
                };
                
                // Execute trade on bid side (they are MAKERS - their limit orders were resting)  
                let bid_executed_orders = {
                    let bid_level = self.bids.get_mut(&bid_key).unwrap();
                    bid_level.execute_trade(match_quantity, timestamp).unwrap_or_default()
                };
                
                // Create fill records for bid orders (MAKERS - they provided liquidity)
                for bid_order in bid_executed_orders {
                    let fill_quantity = bid_order.original_quantity - bid_order.remaining_quantity;
                    let mut order_with_price = bid_order;
                    order_with_price.price = execution_price;
                    
                    let fill = self.create_fill(&order_with_price, fill_quantity, LiquidityType::Maker, timestamp);
                    
                    // Track volume for fee tier calculation
                    let _volume_usd = fill.price * fill.quantity;
                    fills.push(fill);
                }
                
                // Create fill records for ask orders (MAKERS - they provided liquidity)
                for ask_order in ask_executed_orders {
                    let fill_quantity = ask_order.original_quantity - ask_order.remaining_quantity;
                    let mut order_with_price = ask_order;
                    order_with_price.price = execution_price;
                    
                    let fill = self.create_fill(&order_with_price, fill_quantity, LiquidityType::Maker, timestamp);
                    
                    // Track volume for fee tier calculation
                    let _volume_usd = fill.price * fill.quantity;
                    fills.push(fill);
                }
                
                // Remove empty levels
                if self.bids.get(&bid_key).map_or(true, |level| level.is_empty()) {
                    self.bids.remove(&bid_key);
                }
                if self.asks.get(&ask_key).map_or(true, |level| level.is_empty()) {
                    self.asks.remove(&ask_key);
                }
                
                self.last_updated = timestamp;
            } else {
                break;
            }
        }
        
        // Track total volume for fee tier calculation
        let total_volume_usd: f64 = fills
            .iter()
            .map(|fill| fill.price * fill.quantity)
            .sum();
            
        if total_volume_usd > 0.0 {
            self.add_volume(total_volume_usd, timestamp);
        }
        
        fills
    }
    /// Create a new empty order book
    pub fn new(symbol: String, exchange: String, config: OrderBookConfig) -> Self {
        Self {
            symbol,
            exchange,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_updated: Utc::now(),
            config,
            sequence_number: 0,
            fee_config: None,
            volume_history: VecDeque::new(),
            volume_30d: 0.0,
            slippage_config: SlippageConfig::default(),
        }
    }

    /// Create a new order book with fee configuration
    pub fn new_with_fees(
        symbol: String, 
        exchange: String, 
        config: OrderBookConfig,
        fee_config: ExchangeFeeConfig
    ) -> Self {
        Self {
            symbol,
            exchange,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_updated: Utc::now(),
            config,
            sequence_number: 0,
            fee_config: Some(fee_config),
            volume_history: VecDeque::new(),
            volume_30d: 0.0,
            slippage_config: SlippageConfig::default(),
        }
    }

    /// Create a new order book with full configuration
    pub fn new_with_config(
        symbol: String, 
        exchange: String, 
        config: OrderBookConfig,
        fee_config: Option<ExchangeFeeConfig>,
        slippage_config: SlippageConfig
    ) -> Self {
        Self {
            symbol,
            exchange,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_updated: Utc::now(),
            config,
            sequence_number: 0,
            fee_config,
            volume_history: VecDeque::new(),
            volume_30d: 0.0,
            slippage_config,
        }
    }

    /// Set fee configuration
    pub fn set_fee_config(&mut self, fee_config: ExchangeFeeConfig) {
        self.fee_config = Some(fee_config);
    }

    /// Set slippage configuration
    pub fn set_slippage_config(&mut self, slippage_config: SlippageConfig) {
        self.slippage_config = slippage_config;
    }

    /// Simulate a market order execution without consuming the order book
    /// This is useful for backtesting where we want to see the slippage impact without modifying the book
    pub fn simulate_market_order(
        &self, 
        side: BookSide, 
        quantity: f64
    ) -> MarketOrderExecution {
        let mut execution_levels = Vec::new();
        let mut remaining_quantity = quantity;
        let mut total_cost = 0.0;
        let mut cumulative_quantity = 0.0;
        
        // Determine which side of the book to analyze
        let levels = match side {
            BookSide::Bid => &self.asks, // Buying would consume asks
            BookSide::Ask => &self.bids, // Selling would consume bids
        };
        
        // Get sorted keys
        let level_keys: Vec<_> = match side {
            BookSide::Bid => levels.keys().cloned().collect(), // Best ask first (lowest price)
            BookSide::Ask => levels.keys().rev().cloned().collect(), // Best bid first (highest price)
        };
        
        for level_key in level_keys {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            if let Some(level) = levels.get(&level_key) {
                let available_quantity = level.total_quantity;
                let execute_quantity = remaining_quantity.min(available_quantity);
                let level_cost = level.price * execute_quantity;
                
                cumulative_quantity += execute_quantity;
                execution_levels.push(ExecutionLevel {
                    price: level.price,
                    quantity: execute_quantity,
                    cumulative_quantity,
                });
                
                total_cost += level_cost;
                remaining_quantity -= execute_quantity;
            }
        }
        
        let executed_quantity = quantity - remaining_quantity;
        
        // Calculate VWAP
        let vwap = if executed_quantity > 0.0 {
            total_cost / executed_quantity
        } else {
            0.0
        };
        
        // Calculate slippage
        let slippage_bps = if let Some(first_level) = execution_levels.first() {
            let first_price = first_level.price;
            let price_diff = match side {
                BookSide::Bid => (vwap - first_price) / first_price,
                BookSide::Ask => (first_price - vwap) / first_price,
            };
            price_diff * 10000.0
        } else {
            0.0
        };
        
        MarketOrderExecution {
            requested_quantity: quantity,
            executed_quantity,
            vwap,
            execution_levels,
            total_cost,
            slippage_bps,
        }
    }

    /// Execute a market order against the order book, consuming actual liquidity
    /// This provides realistic slippage by walking through actual order book levels
    pub fn execute_market_order(
        &mut self, 
        side: BookSide, 
        quantity: f64, 
        timestamp: DateTime<Utc>
    ) -> Result<MarketOrderExecution> {
        let mut execution_levels = Vec::new();
        let mut remaining_quantity = quantity;
        let mut total_cost = 0.0;
        let mut cumulative_quantity = 0.0;
        
        // Determine which side of the book to consume
        let levels = match side {
            BookSide::Bid => &mut self.asks, // Buying consumes asks
            BookSide::Ask => &mut self.bids, // Selling consumes bids
        };
        
        // Store keys to iterate over (to avoid borrowing issues)
        let level_keys: Vec<_> = match side {
            BookSide::Bid => levels.keys().cloned().collect(), // Best ask first (lowest price)
            BookSide::Ask => levels.keys().rev().cloned().collect(), // Best bid first (highest price)
        };
        
        for level_key in level_keys {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            let level_price = {
                if let Some(level) = levels.get(&level_key) {
                    level.price
                } else {
                    continue;
                }
            };
            
            let available_quantity = {
                if let Some(level) = levels.get(&level_key) {
                    level.total_quantity
                } else {
                    continue;
                }
            };
            
            // Calculate how much we can execute at this level
            let execute_quantity = remaining_quantity.min(available_quantity);
            let level_cost = level_price * execute_quantity;
            
            // Execute the trade at this level
            if let Some(level) = levels.get_mut(&level_key) {
                let _executed_orders = level.execute_trade(execute_quantity, timestamp)?;
                
                // Record this execution level
                cumulative_quantity += execute_quantity;
                execution_levels.push(ExecutionLevel {
                    price: level_price,
                    quantity: execute_quantity,
                    cumulative_quantity,
                });
                
                total_cost += level_cost;
                remaining_quantity -= execute_quantity;
                
                // Remove empty levels
                if level.is_empty() {
                    levels.remove(&level_key);
                }
            }
        }
        
        let executed_quantity = quantity - remaining_quantity;
        
        // Calculate VWAP
        let vwap = if executed_quantity > 0.0 {
            total_cost / executed_quantity
        } else {
            0.0
        };
        
        // Calculate slippage (if we have execution levels)
        let slippage_bps = if let Some(first_level) = execution_levels.first() {
            let first_price = first_level.price;
            let price_diff = match side {
                BookSide::Bid => (vwap - first_price) / first_price, // Buying: higher VWAP = positive slippage
                BookSide::Ask => (first_price - vwap) / first_price, // Selling: lower VWAP = positive slippage
            };
            price_diff * 10000.0 // Convert to basis points
        } else {
            0.0
        };
        
        Ok(MarketOrderExecution {
            requested_quantity: quantity,
            executed_quantity,
            vwap,
            execution_levels,
            total_cost,
            slippage_bps,
        })
    }

    /// Add volume to tracking history and update 30-day volume
    fn add_volume(&mut self, volume_usd: f64, timestamp: DateTime<Utc>) {
        // Add new volume entry
        self.volume_history.push_back(VolumeEntry {
            timestamp,
            volume_usd,
        });

        // Remove entries older than 30 days
        let cutoff = timestamp - Duration::days(30);
        while let Some(entry) = self.volume_history.front() {
            if entry.timestamp >= cutoff {
                break;
            }
            self.volume_history.pop_front();
        }

        // Recalculate 30-day volume
        self.volume_30d = self.volume_history
            .iter()
            .map(|entry| entry.volume_usd)
            .sum();
    }

    /// Calculate slippage based on order size and market depth
    fn calculate_slippage(&self, side: BookSide, order_size_usd: f64) -> f64 {
        let levels = match side {
            BookSide::Bid => &self.asks, // Buying hits asks
            BookSide::Ask => &self.bids, // Selling hits bids
        };

        // Calculate available liquidity in the top N levels
        let mut total_liquidity = 0.0;
        let mut level_count = 0;

        for (price, level) in levels.iter() {
            if level_count >= self.slippage_config.depth_levels {
                break;
            }
            let price_f64 = price.into_inner();
            let quantity = level.total_quantity;
            total_liquidity += price_f64 * quantity;
            level_count += 1;
        }

        // Base slippage
        let mut slippage_bps = self.slippage_config.base_slippage_bps;

        // Add impact based on order size relative to available liquidity
        if total_liquidity > 0.0 {
            let impact_ratio = order_size_usd / total_liquidity;
            let impact_slippage = impact_ratio * self.slippage_config.impact_coefficient * 10000.0; // Convert to bps
            slippage_bps += impact_slippage;
        } else {
            // No liquidity - maximum slippage
            slippage_bps = self.slippage_config.max_slippage_bps;
        }

        // Apply bounds
        slippage_bps = slippage_bps.max(self.slippage_config.min_slippage_bps);
        slippage_bps = slippage_bps.min(self.slippage_config.max_slippage_bps);

        slippage_bps / 10000.0 // Convert back to decimal
    }

    /// Calculate execution price including slippage
    fn apply_slippage(&self, base_price: f64, side: BookSide, slippage_rate: f64) -> f64 {
        match side {
            BookSide::Bid => base_price * (1.0 + slippage_rate), // Buying - price goes up
            BookSide::Ask => base_price * (1.0 - slippage_rate), // Selling - price goes down
        }
    }

    /// Create a fill with fee calculation and slippage
    fn create_fill(&self, order: &OrderInfo, quantity: f64, liquidity_type: LiquidityType, timestamp: DateTime<Utc>) -> Fill {
        let base_price = order.price;
        
        // Calculate slippage - only applies to TAKERS (market orders)
        // MAKERS (limit orders being filled) get their exact limit price
        let (execution_price, slippage_bps) = if liquidity_type == LiquidityType::Maker {
            // Maker: no slippage, we get our limit price
            (base_price, None)
        } else {
            // Taker: apply slippage based on order size and market impact
            let trade_value_usd = base_price * quantity;
            let slippage_rate = self.calculate_slippage(order.side, trade_value_usd);
            let slipped_price = self.apply_slippage(base_price, order.side, slippage_rate);
            let bps = if slippage_rate > 0.0 { Some(slippage_rate * 10000.0) } else { None };
            (slipped_price, bps)
        };
        
        // Calculate final trade value with execution price
        let final_trade_value = execution_price * quantity;
        
        // Calculate fees
        let (fee_rate, fee_amount) = if let Some(ref fee_config) = self.fee_config {
            let is_maker = liquidity_type == LiquidityType::Maker;
            let rate = fee_config.get_fee_rate(self.volume_30d, is_maker);
            let fee = if rate >= 0.0 {
                final_trade_value * rate
            } else {
                // Negative fee (rebate)
                final_trade_value * (-rate) * -1.0
            };
            (rate, fee)
        } else {
            (0.0, 0.0)
        };

        Fill {
            order_id: order.order_id.clone(),
            side: order.side,
            price: execution_price,  // Use slippage-adjusted price (or exact limit price for makers)
            base_price: Some(base_price),  // Original price before slippage
            quantity,
            liquidity_type,
            fee_rate,
            fee_amount,
            slippage_bps,  // Already computed above (None for makers)
            timestamp,
        }
    }

    /// Apply an order book event
    pub fn apply_event(&mut self, event: OrderBookEvent) -> Result<()> {
        match event {
            OrderBookEvent::NewOrder {
                order_id,
                side,
                price,
                quantity,
                timestamp,
            } => {
                self.add_order(order_id, side, price, quantity, timestamp)?;
            }
            OrderBookEvent::ModifyOrder {
                order_id,
                new_quantity,
                timestamp,
            } => {
                self.modify_order(&order_id, new_quantity, timestamp)?;
            }
            OrderBookEvent::CancelOrder { order_id, timestamp } => {
                self.cancel_order(&order_id, timestamp)?;
            }
            OrderBookEvent::Trade {
                side,
                price,
                quantity,
                timestamp,
                ..
            } => {
                self.execute_trade(side, price, quantity, timestamp)?;
            }
            OrderBookEvent::Snapshot {
                bids,
                asks,
                timestamp,
            } => {
                self.apply_snapshot(bids, asks, timestamp)?;
            }
        }
        
        self.sequence_number += 1;
        self.cleanup_empty_levels();
        Ok(())
    }

    /// Add a new order to the book
    pub fn add_order(
        &mut self,
        order_id: Arc<str>,
        side: BookSide,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        let price_key = OrderedFloat(price);

        let order = OrderInfo {
            order_id: order_id.clone(),
            side,
            original_quantity: quantity,
            remaining_quantity: quantity,
            price,
            timestamp,
            order_type: crate::types::OrderType::Limit,
            liquidity_type: None, // Will be set when the order is filled
        };

        let levels = match side {
            BookSide::Bid => &mut self.bids,
            BookSide::Ask => &mut self.asks,
        };

        // Get or create level
        let level = levels
            .entry(price_key)
            .or_insert_with(|| Level::new(price, side, self.config.track_individual_orders));

        level.add_order(order)?;
        self.last_updated = timestamp;

        log_debug!(ORDERBOOK_LOGGER, 
            "Added order {} to {} side at price {} with quantity {}",
            order_id,
            match side {
                BookSide::Bid => "bid",
                BookSide::Ask => "ask",
            },
            price,
            quantity
        );

        Ok(())
    }

    /// Modify an existing order
    pub fn modify_order(
        &mut self,
        order_id: &str,
        new_quantity: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        // Search both sides for the order
        for (_, level) in self.bids.iter_mut().chain(self.asks.iter_mut()) {
            if level.orders.as_ref().map_or(false, |orders| orders.contains_key(order_id)) {
                level.modify_order(order_id, new_quantity, timestamp)?;
                self.last_updated = timestamp;
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Order {} not found for modification", order_id))
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: &str, timestamp: DateTime<Utc>) -> Result<()> {
        // Search both sides for the order
        for (_, level) in self.bids.iter_mut().chain(self.asks.iter_mut()) {
            if level.orders.as_ref().map_or(false, |orders| orders.contains_key(order_id)) {
                level.remove_order(order_id, timestamp)?;
                self.last_updated = timestamp;
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Order {} not found for cancellation", order_id))
    }

    /// Execute a trade against the book
    pub fn execute_trade(
        &mut self,
        side: BookSide,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<OrderInfo>> {
        let price_key = OrderedFloat(price);

        let levels = match side {
            BookSide::Bid => &mut self.bids,
            BookSide::Ask => &mut self.asks,
        };

        if let Some(level) = levels.get_mut(&price_key) {
            let executed_orders = level.execute_trade(quantity, timestamp)?;
            self.last_updated = timestamp;
            Ok(executed_orders)
        } else {
            log_warn!(ORDERBOOK_LOGGER, "Trade execution attempted at non-existent price level: {}", price);
            Ok(Vec::new())
        }
    }

    /// Execute a trade against the book and return fills
    /// This is used when processing incoming market trades that may fill our resting orders
    /// Execute an incoming market trade against our resting orders
    /// 
    /// For market making simulation:
    /// - Incoming buy trade (side=Ask): fills our ask orders at or below the trade price
    /// - Incoming sell trade (side=Bid): fills our bid orders at or above the trade price
    /// 
    /// Orders are filled at THEIR order price (not the trade price) - this is how
    /// market makers capture the spread.
    pub fn execute_trade_with_fills(
        &mut self,
        side: BookSide,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    ) -> Vec<Fill> {
        let mut fills = Vec::new();
        let mut remaining_quantity = quantity;
        
        let trade_price = OrderedFloat(price);

        // Collect price levels to process (to avoid borrow issues)
        // The trade 'side' is the AGGRESSOR's side (taker):
        // - side = Bid means a buyer aggressed -> fills our ASK orders (sellers)
        // - side = Ask means a seller aggressed -> fills our BID orders (buyers)
        let levels_to_process: Vec<OrderedFloat<f64>> = match side {
            BookSide::Bid => {
                // Buyer aggressed - fills our asks at or below trade price
                // Best asks (lowest) get filled first
                self.asks.keys()
                    .filter(|&k| *k <= trade_price)
                    .copied()
                    .collect()
            }
            BookSide::Ask => {
                // Seller aggressed - fills our bids at or above trade price
                // Best bids (highest) get filled first - reverse iterator
                self.bids.keys()
                    .rev()
                    .filter(|&k| *k >= trade_price)
                    .copied()
                    .collect()
            }
        };

        // Collect all executed orders first, then create fills
        let mut executed_orders_all: Vec<OrderInfo> = Vec::new();
        let mut levels_to_remove = Vec::new();
        
        for level_price in levels_to_process {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            // Access the OPPOSITE side: aggressor side = Bid -> fills asks, Ask -> fills bids
            let levels = match side {
                BookSide::Bid => &mut self.asks,  // Buyer fills asks
                BookSide::Ask => &mut self.bids,  // Seller fills bids
            };
            
            if let Some(level) = levels.get_mut(&level_price) {
                // Execute as much as possible from this level
                let level_available = level.total_quantity;
                let fill_qty = remaining_quantity.min(level_available);
                
                if fill_qty > 0.0 {
                    let executed_orders = level.execute_trade(fill_qty, timestamp).unwrap_or_default();
                    executed_orders_all.extend(executed_orders);
                    
                    remaining_quantity -= fill_qty;
                    
                    // Mark level for removal if empty
                    if level.is_empty() {
                        levels_to_remove.push(level_price);
                    }
                }
            }
        }
        
        // Remove empty levels from the OPPOSITE side
        for level_price in levels_to_remove {
            match side {
                BookSide::Bid => { self.asks.remove(&level_price); }  // Buyer filled asks
                BookSide::Ask => { self.bids.remove(&level_price); }  // Seller filled bids
            }
        }
        
        // Now create fills from all executed orders (no mutable borrow conflict)
        for order in executed_orders_all {
            let order_fill_qty = order.original_quantity - order.remaining_quantity;
            if order_fill_qty > 0.0 {
                let fill = self.create_fill(&order, order_fill_qty, LiquidityType::Maker, timestamp);
                fills.push(fill);
            }
        }
        
        if !fills.is_empty() {
            self.last_updated = timestamp;
        }

        // Track volume for fee tier calculation
        let total_volume_usd: f64 = fills
            .iter()
            .map(|fill| fill.price * fill.quantity)
            .sum();
            
        if total_volume_usd > 0.0 {
            self.add_volume(total_volume_usd, timestamp);
        }

        fills
    }

    /// Apply a complete snapshot to rebuild the book
    pub fn apply_snapshot(
        &mut self,
        bids: Vec<SnapshotLevel>,
        asks: Vec<SnapshotLevel>,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        // Clear existing levels
        self.bids.clear();
        self.asks.clear();

        // Rebuild from snapshot
        for bid in bids {
            let price_key = OrderedFloat(bid.price);
            
            let mut level = Level::new(bid.price, BookSide::Bid, false);
            level.total_quantity = bid.quantity;
            level.order_count = bid.order_count;
            level.last_updated = timestamp;
            
            self.bids.insert(price_key, level);
        }

        for ask in asks {
            let price_key = OrderedFloat(ask.price);
            
            let mut level = Level::new(ask.price, BookSide::Ask, false);
            level.total_quantity = ask.quantity;
            level.order_count = ask.order_count;
            level.last_updated = timestamp;
            
            self.asks.insert(price_key, level);
        }

        self.last_updated = timestamp;
        Ok(())
    }

    /// Get market depth information
    pub fn get_market_depth(&self, max_levels: Option<usize>) -> MarketDepth {
        let limit = max_levels.unwrap_or(self.config.max_levels.unwrap_or(100));

        let bids: Vec<DepthLevel> = self
            .bids
            .iter()
            .rev() // Highest bids first
            .take(limit)
            .map(|(price, level)| DepthLevel {
                price: price.0,
                quantity: level.total_quantity,
                order_count: level.order_count,
            })
            .collect();

        let asks: Vec<DepthLevel> = self
            .asks
            .iter() // Lowest asks first
            .take(limit)
            .map(|(price, level)| DepthLevel {
                price: price.0,
                quantity: level.total_quantity,
                order_count: level.order_count,
            })
            .collect();

        let best_bid = bids.first().cloned();
        let best_ask = asks.first().cloned();
        let spread = match (&best_bid, &best_ask) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        };

        MarketDepth {
            timestamp: self.last_updated,
            symbol: self.symbol.clone(),
            exchange: self.exchange.clone(),
            bids,
            asks,
            best_bid,
            best_ask,
            spread,
        }
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.values().next_back().map(|level| level.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.values().next().map(|level| level.price)
    }

    /// Get the current spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Remove empty levels from the book
    fn cleanup_empty_levels(&mut self) {
        self.bids.retain(|_, level| !level.is_empty());
        self.asks.retain(|_, level| !level.is_empty());
    }

    /// Check if the order book is crossed (best bid >= best ask)
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => bid >= ask,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BookSide;

    #[test]
    fn test_order_book_creation() {
        let config = OrderBookConfig::default();
        let book = OrderBook::new("BTCUSD".to_string(), "TEST".to_string(), config);
        
        assert_eq!(book.symbol, "BTCUSD");
        assert_eq!(book.exchange, "TEST");
        assert!(book.bids.is_empty());
        assert!(book.asks.is_empty());
    }

    #[tokio::test]
    async fn test_add_orders() {
        let config = OrderBookConfig::default();
        let mut book = OrderBook::new("BTCUSD".to_string(), "TEST".to_string(), config);
        
        // Add bid
        book.add_order(
            Arc::from("bid1"),
            BookSide::Bid,
            100.0,
            10.0,
            Utc::now(),
        ).unwrap();
        
        // Add ask
        book.add_order(
            Arc::from("ask1"),
            BookSide::Ask,
            105.0,
            15.0,
            Utc::now(),
        ).unwrap();
        
        assert_eq!(book.bids.len(), 1);
        assert_eq!(book.asks.len(), 1);
        assert_eq!(book.best_bid(), Some(100.0));
        assert_eq!(book.best_ask(), Some(105.0));
    }
    
    #[tokio::test]
    async fn test_realistic_slippage_modeling() {
        let mut order_book = OrderBook::new(
            "BTC-USD".to_string(),
            "test_exchange".to_string(),
            config::OrderBookConfig::default(),
        );
        
        let now = Utc::now();
        
        // Add some ask levels to the book (people selling)
        order_book.add_order(Arc::from("ask1"), BookSide::Ask, 50000.0, 1.0, now).unwrap();
        order_book.add_order(Arc::from("ask2"), BookSide::Ask, 50010.0, 2.0, now).unwrap();
        order_book.add_order(Arc::from("ask3"), BookSide::Ask, 50020.0, 3.0, now).unwrap();
        
        // Add some bid levels (people buying)
        order_book.add_order(Arc::from("bid1"), BookSide::Bid, 49990.0, 1.5, now).unwrap();
        order_book.add_order(Arc::from("bid2"), BookSide::Bid, 49980.0, 2.5, now).unwrap();
        
        // Simulate a large market buy order (should consume asks)
        let market_buy_result = order_book.simulate_market_order(
            BookSide::Bid,  // Buying
            4.0  // Buy 4 BTC
        );
        
        println!("Market Buy Order Results:");
        println!("Requested: {} BTC", market_buy_result.requested_quantity);
        println!("Executed: {} BTC", market_buy_result.executed_quantity);
        println!("VWAP: ${}", market_buy_result.vwap);
        println!("Total Cost: ${}", market_buy_result.total_cost);
        println!("Slippage: {} bps", market_buy_result.slippage_bps);
        
        for (i, level) in market_buy_result.execution_levels.iter().enumerate() {
            println!("Level {}: {} BTC @ ${} (cumulative: {})", 
                i + 1, level.quantity, level.price, level.cumulative_quantity);
        }
        
        // The order should have consumed:
        // - 1.0 BTC @ $50,000 = $50,000
        // - 2.0 BTC @ $50,010 = $100,020
        // - 1.0 BTC @ $50,020 = $50,020  (partial fill of level 3)
        // Total: 4.0 BTC for $200,040
        // VWAP: $50,010 (vs first level $50,000 = 0.02% = 2 bps slippage)
        
        assert_eq!(market_buy_result.executed_quantity, 4.0);
        assert!(market_buy_result.slippage_bps > 0.0); // Should have positive slippage
        assert_eq!(market_buy_result.execution_levels.len(), 3); // Should have hit 3 levels
        
        // Now test a market sell order
        let market_sell_result = order_book.simulate_market_order(
            BookSide::Ask,  // Selling
            3.0  // Sell 3 BTC
        );
        
        println!("\nMarket Sell Order Results:");
        println!("Requested: {} BTC", market_sell_result.requested_quantity);
        println!("Executed: {} BTC", market_sell_result.executed_quantity);  
        println!("VWAP: ${}", market_sell_result.vwap);
        println!("Total Proceeds: ${}", market_sell_result.total_cost);
        println!("Slippage: {} bps", market_sell_result.slippage_bps);
        
        // Should consume bids starting from highest price
        assert!(market_sell_result.slippage_bps > 0.0);
    }
}
