//! Tick trait for unified market data abstraction
//!
//! This trait allows the simulation loop to work with different data sources:
//! - `MarketData` (in-memory from database)
//! - `SimulationTick` (memory-mapped binary files)
//!
//! This abstraction enables zero-copy backtesting on large datasets.

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub fn is_buy(&self) -> bool {
        matches!(self, Side::Buy)
    }
    
    #[inline]
    pub fn is_sell(&self) -> bool {
        matches!(self, Side::Sell)
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }
}

impl From<&str> for Side {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sell" | "s" | "ask" | "a" => Side::Sell,
            _ => Side::Buy,
        }
    }
}

/// Core tick data trait for simulation
/// 
/// This trait extracts the essential fields needed by `SimulationLoop`:
/// - timestamp: When the tick occurred
/// - price: The price level
/// - quantity: Trade/order size
/// - side: Buy or sell
///
/// All implementations must provide fast access to these fields.
pub trait Tick: Send + Sync {
    /// Get the timestamp
    fn timestamp(&self) -> DateTime<Utc>;
    
    /// Get the timestamp as milliseconds since Unix epoch
    fn timestamp_ms(&self) -> i64 {
        self.timestamp().timestamp_millis()
    }
    
    /// Get the price as f64
    fn price(&self) -> f64;
    
    /// Get the quantity as f64
    fn quantity(&self) -> f64;
    
    /// Get the trade side
    fn side(&self) -> Side;
    
    /// Get the event type (0=New, 1=Modify, 2=Cancel, 3=Trade)
    /// Default returns 3 (Trade) for backward compatibility
    fn event_type(&self) -> u8 {
        3 // Trade by default
    }
    
    /// Check if this is a sell
    #[inline]
    fn is_sell(&self) -> bool {
        self.side().is_sell()
    }
    
    /// Check if this is a buy
    #[inline]
    fn is_buy(&self) -> bool {
        self.side().is_buy()
    }
    
    /// Get the notional value (price * quantity)
    #[inline]
    fn value(&self) -> f64 {
        self.price() * self.quantity()
    }
}

// Implement Tick for TradeData
impl Tick for super::TradeData {
    #[inline]
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    
    #[inline]
    fn price(&self) -> f64 {
        self.price
    }
    
    #[inline]
    fn quantity(&self) -> f64 {
        self.quantity
    }
    
    #[inline]
    fn side(&self) -> Side {
        self.side
    }
}

// Implement Tick for MarketData (delegates to inner type)
impl Tick for super::MarketData {
    fn timestamp(&self) -> DateTime<Utc> {
        match self {
            super::MarketData::Trade(t) => t.timestamp,
            super::MarketData::Candle(c) => c.timestamp,
            super::MarketData::Generic(g) => DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(DateTime::UNIX_EPOCH),
            super::MarketData::PoolSwap(s) => s.timestamp,
            super::MarketData::OptionCandle(c) => c.timestamp,
        }
    }

    fn price(&self) -> f64 {
        match self {
            super::MarketData::Trade(t) => t.price,
            super::MarketData::Candle(c) => c.close,
            super::MarketData::Generic(g) => g.price,
            super::MarketData::PoolSwap(s) => s.price(),
            super::MarketData::OptionCandle(c) => c.close,
        }
    }

    fn quantity(&self) -> f64 {
        match self {
            super::MarketData::Trade(t) => t.quantity,
            super::MarketData::Candle(c) => c.volume,
            super::MarketData::Generic(g) => g.quantity,
            super::MarketData::PoolSwap(s) => s.amount_in,
            super::MarketData::OptionCandle(c) => c.volume,
        }
    }

    fn side(&self) -> Side {
        match self {
            super::MarketData::Trade(t) => t.side,
            super::MarketData::Candle(_) => Side::Buy, // Candles don't have side
            super::MarketData::Generic(g) => g.side,
            super::MarketData::PoolSwap(_) => Side::Buy,
            super::MarketData::OptionCandle(_) => Side::Buy, // Option candles don't have side either
        }
    }
}

/// A tick source that can provide ticks for simulation
/// 
/// This trait abstracts over different data sources:
/// - Memory-mapped binary files (zero-copy, constant memory)
/// - In-memory vectors (legacy database loads)
pub trait TickSource: Send + Sync {
    /// The tick type this source provides
    type TickType: Tick;
    
    /// Get the number of ticks
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a tick by index
    fn get(&self, index: usize) -> Option<&Self::TickType>;
    
    /// Iterate over all ticks
    fn iter(&self) -> impl Iterator<Item = &Self::TickType>;
}

/// Wrapper for a slice of MarketData
pub struct MarketDataSlice<'a> {
    data: &'a [super::MarketData],
}

impl<'a> MarketDataSlice<'a> {
    pub fn new(data: &'a [super::MarketData]) -> Self {
        Self { data }
    }
}

impl<'a> TickSource for MarketDataSlice<'a> {
    type TickType = super::MarketData;
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, index: usize) -> Option<&Self::TickType> {
        self.data.get(index)
    }
    
    fn iter(&self) -> impl Iterator<Item = &Self::TickType> {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Side ──

    #[test]
    fn side_buy_is_buy() {
        assert!(Side::Buy.is_buy());
        assert!(!Side::Buy.is_sell());
    }

    #[test]
    fn side_sell_is_sell() {
        assert!(Side::Sell.is_sell());
        assert!(!Side::Sell.is_buy());
    }

    #[test]
    fn side_as_str() {
        assert_eq!(Side::Buy.as_str(), "buy");
        assert_eq!(Side::Sell.as_str(), "sell");
    }

    #[test]
    fn side_from_str_buy_variants() {
        assert_eq!(Side::from("buy"), Side::Buy);
        assert_eq!(Side::from("BUY"), Side::Buy);
        assert_eq!(Side::from("b"), Side::Buy);
        assert_eq!(Side::from("anything"), Side::Buy); // default
    }

    #[test]
    fn side_from_str_sell_variants() {
        assert_eq!(Side::from("sell"), Side::Sell);
        assert_eq!(Side::from("SELL"), Side::Sell);
        assert_eq!(Side::from("s"), Side::Sell);
        assert_eq!(Side::from("ask"), Side::Sell);
        assert_eq!(Side::from("ASK"), Side::Sell);
        assert_eq!(Side::from("a"), Side::Sell);
        assert_eq!(Side::from("A"), Side::Sell);
    }

    #[test]
    fn side_serde_roundtrip() {
        for side in [Side::Buy, Side::Sell] {
            let json = serde_json::to_string(&side).unwrap();
            let decoded: Side = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, side);
        }
    }

    // ── Tick trait via TradeData ──

    use std::sync::Arc;

    fn make_trade(price: f64, quantity: f64, side: Side) -> super::super::TradeData {
        super::super::TradeData {
            timestamp: Utc::now(),
            symbol: Arc::from("BTC-USD"),
            exchange: Arc::from("massive"),
            price,
            quantity,
            side,
            trade_id: 1,
            value: price * quantity,
        }
    }

    #[test]
    fn trade_data_tick_basics() {
        let td = make_trade(50000.0, 1.5, Side::Buy);
        assert!((td.price() - 50000.0).abs() < 1e-10);
        assert!((td.quantity() - 1.5).abs() < 1e-10);
        assert!(td.is_buy());
        assert!(!td.is_sell());
        assert!((td.value() - 75000.0).abs() < 1e-10);
        assert_eq!(td.event_type(), 3);
    }

    #[test]
    fn trade_data_tick_timestamp_ms() {
        use chrono::TimeZone;
        let ts = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let td = super::super::TradeData {
            timestamp: ts,
            symbol: Arc::from("BTC-USD"),
            exchange: Arc::from("massive"),
            price: 100.0,
            quantity: 1.0,
            side: Side::Sell,
            trade_id: 1,
            value: 100.0,
        };
        assert_eq!(td.timestamp_ms(), ts.timestamp_millis());
        assert!(td.is_sell());
    }

    // ── MarketDataSlice ──

    #[test]
    fn market_data_slice_empty() {
        let data: Vec<super::super::MarketData> = vec![];
        let slice = MarketDataSlice::new(&data);
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
        assert!(slice.get(0).is_none());
    }

    #[test]
    fn market_data_slice_with_trades() {
        let trade = make_trade(100.0, 2.0, Side::Buy);
        let data = vec![super::super::MarketData::Trade(trade)];
        let slice = MarketDataSlice::new(&data);
        assert_eq!(slice.len(), 1);
        assert!(!slice.is_empty());
        assert!(slice.get(0).is_some());
        assert!(slice.get(1).is_none());
        assert_eq!(slice.iter().count(), 1);
    }
}
