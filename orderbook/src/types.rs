//! Core types and enums for order book operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type of liquidity provided/removed
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LiquidityType {
    Maker,  // Added liquidity to the book (limit order that rests)
    Taker,  // Removed liquidity from the book (market order or aggressive limit)
}

/// Side of the order book (bid or ask)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BookSide {
    Bid,
    Ask,
}

impl BookSide {
    /// Get the opposite side
    pub fn opposite(&self) -> Self {
        match self {
            BookSide::Bid => BookSide::Ask,
            BookSide::Ask => BookSide::Bid,
        }
    }

    /// Convert from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BID" | "BUY" | "B" => Some(BookSide::Bid),
            "ASK" | "SELL" | "S" | "OFFER" => Some(BookSide::Ask),
            _ => None,
        }
    }
}

/// Type of order book event
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderBookEvent {
    /// New order added to book
    NewOrder {
        order_id: Arc<str>,
        side: BookSide,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    },
    /// Order quantity modified
    ModifyOrder {
        order_id: Arc<str>,
        new_quantity: f64,
        timestamp: DateTime<Utc>,
    },
    /// Order cancelled/removed
    CancelOrder {
        order_id: Arc<str>,
        timestamp: DateTime<Utc>,
    },
    /// Trade executed (partial or full fill)
    Trade {
        aggressor_order_id: Option<Arc<str>>,
        passive_order_id: Option<Arc<str>>,
        side: BookSide,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    },
    /// Complete order book snapshot
    Snapshot {
        bids: Vec<SnapshotLevel>,
        asks: Vec<SnapshotLevel>,
        timestamp: DateTime<Utc>,
    },
}

/// Level in an order book snapshot
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

/// Complete order book snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: String,
    pub exchange: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<SnapshotLevel>,
    pub asks: Vec<SnapshotLevel>,
    pub sequence_number: Option<u64>,
}

/// Individual order information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderInfo {
    pub order_id: Arc<str>,
    pub side: BookSide,
    pub original_quantity: f64,
    pub remaining_quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub order_type: OrderType,
    pub liquidity_type: Option<LiquidityType>, // Set when order is filled
}

/// Trade fill information with fee calculation and slippage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: Arc<str>,
    pub side: BookSide,
    pub price: f64,                 // Execution price (includes slippage)
    pub base_price: Option<f64>,    // Original price before slippage
    pub quantity: f64,
    pub liquidity_type: LiquidityType,
    pub fee_rate: f64,              // Applied fee rate
    pub fee_amount: f64,            // Calculated fee amount
    pub slippage_bps: Option<f64>,  // Applied slippage in basis points
    pub timestamp: DateTime<Utc>,
}

/// Type of order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
    Stop,
    StopLimit,
}

impl Default for OrderType {
    fn default() -> Self {
        OrderType::Limit
    }
}

/// Volume tracking entry for fee tier calculation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeEntry {
    pub timestamp: DateTime<Utc>,
    pub volume_usd: f64,
}

impl VolumeEntry {
    pub fn new(volume_usd: f64, timestamp: DateTime<Utc>) -> Self {
        Self { timestamp, volume_usd }
    }
}

/// Details of how an order was executed across multiple price levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLevel {
    pub price: f64,
    pub quantity: f64,
    pub cumulative_quantity: f64,
}

/// Result of executing a market order against the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketOrderExecution {
    pub requested_quantity: f64,
    pub executed_quantity: f64,
    pub vwap: f64,
    pub execution_levels: Vec<ExecutionLevel>,
    pub total_cost: f64,
    pub slippage_bps: f64,
}

/// Slippage configuration for realistic execution modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    pub enabled: bool,
    pub base_slippage_bps: f64,    // Base slippage in basis points
    pub impact_coefficient: f64,   // How much order size affects slippage
    pub depth_levels: usize,       // Number of book levels to analyze
    pub min_slippage_bps: f64,     // Minimum slippage floor
    pub max_slippage_bps: f64,     // Maximum slippage cap
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_slippage_bps: 0.5,    // 0.5 basis point base
            impact_coefficient: 0.01,   // 1% impact factor (was 10% - too aggressive)
            depth_levels: 5,            // Analyze 5 levels deep
            min_slippage_bps: 0.1,     // 0.1 bps minimum
            max_slippage_bps: 10.0,    // 10 bps maximum (was 50 - too high)
        }
    }
}

/// Market impact configuration for permanent price impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactConfig {
    pub enabled: bool,
    pub impact_coefficient: f64,     // Base impact per unit volume (0.01 = 1% per BTC)
    pub decay_half_life_ms: u64,     // Half-life for impact decay (60000 = 1 minute)
    pub max_impact_bps: f64,         // Maximum cumulative impact (100 = 1%)
}

impl Default for MarketImpactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            impact_coefficient: 0.01,   // 1% impact per BTC executed
            decay_half_life_ms: 60_000, // 1 minute decay half-life
            max_impact_bps: 100.0,      // Max 1% cumulative impact
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_side_opposite() {
        assert_eq!(BookSide::Bid.opposite(), BookSide::Ask);
        assert_eq!(BookSide::Ask.opposite(), BookSide::Bid);
    }

    #[test]
    fn test_book_side_from_str() {
        assert_eq!(BookSide::from_str("BID"), Some(BookSide::Bid));
        assert_eq!(BookSide::from_str("ask"), Some(BookSide::Ask));
        assert_eq!(BookSide::from_str("BUY"), Some(BookSide::Bid));
        assert_eq!(BookSide::from_str("SELL"), Some(BookSide::Ask));
        assert_eq!(BookSide::from_str("invalid"), None);
    }
}
