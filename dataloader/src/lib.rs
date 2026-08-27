pub mod logging_facade;
pub mod cache;
pub mod aggregation;
pub mod generic;
pub mod models;
pub mod pool_state;
pub mod pool_provider;
pub mod provider;
pub mod supplementary;
pub mod orderbook_csv;
pub mod massive_provider;
pub mod sui_dex_provider;
pub mod tick;

pub use orderbook_csv::{CsvOrderbookLoader, OrderbookSnapshot, PriceLevel, CsvOrderbookError};
pub use massive_provider::{MassiveDataProvider, MassiveProviderError};
pub use sui_dex_provider::SuiDexDataProvider;
pub use pool_state::{PoolSwapEvent, SwapDirection, PoolType};
pub use pool_provider::{PoolDataProvider, PoolDataRequest};
pub use models::{
    DataRequest, DataType, Exchange, ProviderConfig, CacheConfig, CacheStats, CandleGranularity,
    TickerInfo, TickerQuery, TickerSnapshot,
    OptionCandle, OptionChainSnapshot, OptionContractRef, OptionType,
};
pub use aggregation::aggregate_to_candles;
pub use provider::{
    MarketDataProvider, ProviderError, ProviderKind,
    create_provider, create_provider_for, create_provider_with_config,
};
pub use tick::{Tick, TickSource, Side, MarketDataSlice};
pub use generic::{GenericLoader, GenericTick, ColumnMapping, ColumnRef, TimestampFormat, GenericLoaderError};
pub use supplementary::{SupplementaryLoader, SupplementaryData, SupplementaryFileConfig, SupplementaryError};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "postgres")]
use diesel_async::{AsyncPgConnection, pooled_connection::deadpool};

/// Core data loader for market data processing
/// NOTE: Historical market data is loaded from Polygon/Massive API via MassiveDataProvider.
/// Database is only used for candles (aggregated data) and user-generated data.
pub struct DataLoader {
    #[cfg(feature = "postgres")]
    pool: Arc<deadpool::Pool<AsyncPgConnection>>,
    cache: Arc<cache::DataCache>,
}

#[cfg(not(feature = "postgres"))]
impl DataLoader {
    /// Create a new DataLoader instance without database
    pub fn new() -> Self {
        Self {
            cache: Arc::new(cache::DataCache::new()),
        }
    }

    /// Create a mock DataLoader instance for testing
    pub fn mock() -> Self {
        Self::new()
    }

    /// Clear the cache
    pub async fn invalidate_cache(&self) {
        self.cache.clear().await;
    }
}

/// Market data types supported by the loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketData {
    Candle(Candle),
    Trade(TradeData),
    /// Generic user-uploaded data with custom fields
    Generic(generic::GenericTick),
    /// DEX pool swap event with full pool state (for LP backtesting)
    PoolSwap(pool_state::PoolSwapEvent),
    /// A single options-contract OHLCV bar (strike/expiration/contract_type
    /// carried alongside the bar itself, unlike a plain `Candle`). Added for
    /// stock-options support -- existing `match`es on this enum already use
    /// a wildcard arm where they don't care about every variant (see
    /// `extract_candle_ohlcv` etc. in `backtest::python_simulation`), so this
    /// is additive and does not require touching those sites.
    OptionCandle(crate::models::OptionCandle),
}

/// OHLCV candle data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub symbol: Arc<str>,
    pub exchange: Arc<str>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trade_count: i64,
}

impl Candle {
    /// Construct from BigDecimal values (database / legacy callers)
    pub fn from_bigdecimal(
        timestamp: DateTime<Utc>,
        symbol: impl Into<Arc<str>>,
        exchange: impl Into<Arc<str>>,
        open: bigdecimal::BigDecimal,
        high: bigdecimal::BigDecimal,
        low: bigdecimal::BigDecimal,
        close: bigdecimal::BigDecimal,
        volume: bigdecimal::BigDecimal,
        trade_count: i64,
    ) -> Self {
        use bigdecimal::ToPrimitive;
        Self {
            timestamp,
            symbol: symbol.into(),
            exchange: exchange.into(),
            open: open.to_f64().unwrap_or(0.0),
            high: high.to_f64().unwrap_or(0.0),
            low: low.to_f64().unwrap_or(0.0),
            close: close.to_f64().unwrap_or(0.0),
            volume: volume.to_f64().unwrap_or(0.0),
            trade_count,
        }
    }
}

/// Trade data extracted from historical orders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub timestamp: DateTime<Utc>,
    pub symbol: Arc<str>,
    pub exchange: Arc<str>,
    pub price: f64,
    pub quantity: f64,
    pub side: tick::Side,
    pub trade_id: i64,
    /// Pre-computed value (price * quantity)
    #[serde(default)]
    pub value: f64,
}

impl TradeData {
    /// Create a new TradeData from f64 values
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: impl Into<Arc<str>>,
        exchange: impl Into<Arc<str>>,
        price: f64,
        quantity: f64,
        side: tick::Side,
        trade_id: i64,
    ) -> Self {
        Self {
            timestamp,
            symbol: symbol.into(),
            exchange: exchange.into(),
            price,
            quantity,
            side,
            trade_id,
            value: price * quantity,
        }
    }

    /// Construct from BigDecimal values (database / legacy callers)
    pub fn from_bigdecimal(
        timestamp: DateTime<Utc>,
        symbol: impl Into<Arc<str>>,
        exchange: impl Into<Arc<str>>,
        price: bigdecimal::BigDecimal,
        quantity: bigdecimal::BigDecimal,
        side: impl AsRef<str>,
        trade_id: impl AsRef<str>,
    ) -> Self {
        use bigdecimal::ToPrimitive;
        let p = price.to_f64().unwrap_or(0.0);
        let q = quantity.to_f64().unwrap_or(0.0);
        Self {
            timestamp,
            symbol: symbol.into(),
            exchange: exchange.into(),
            price: p,
            quantity: q,
            side: tick::Side::from(side.as_ref()),
            trade_id: trade_id.as_ref().parse::<i64>().unwrap_or(0),
            value: p * q,
        }
    }
}

/// Lightweight tick data for memory-efficient backtesting simulation
/// Only 40 bytes vs ~400 bytes for full HistoricalOrder
/// Use this when running distributed backtests with large datasets
///
/// **DEPRECATED**: Use `data_prep::SimulationTick` (48 bytes, `#[repr(C)]` Pod/Zeroable,
/// mmap-compatible) for all new code. This version lacks `event_type` and precomputed
/// features (`volatility_1m`, `ema_5m`, `imbalance`).
#[deprecated(note = "Use data_prep::SimulationTick (48B, Pod, mmap-compatible) instead")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct SimulationTick {
    /// Timestamp as Unix milliseconds (8 bytes)
    pub timestamp_ms: i64,
    /// Price as f64 (8 bytes)
    pub price: f64,
    /// Quantity as f64 (8 bytes)
    pub quantity: f64,
    /// Side: 0 = Buy, 1 = Sell (1 byte + 7 padding)
    pub side: u8,
}

#[allow(deprecated)]
impl SimulationTick {
    /// Create from price and quantity
    pub fn new(timestamp_ms: i64, price: f64, quantity: f64, is_sell: bool) -> Self {
        Self {
            timestamp_ms,
            price,
            quantity,
            side: if is_sell { 1 } else { 0 },
        }
    }

    /// Check if this is a sell order
    #[inline]
    pub fn is_sell(&self) -> bool {
        self.side == 1
    }

    /// Check if this is a buy order
    #[inline]
    pub fn is_buy(&self) -> bool {
        self.side == 0
    }

    /// Get timestamp as DateTime<Utc>
    pub fn timestamp(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp_ms)
            .unwrap_or_else(|| DateTime::UNIX_EPOCH)
    }
}

#[allow(deprecated)]
impl From<&TradeData> for SimulationTick {
    fn from(trade: &TradeData) -> Self {
        Self {
            timestamp_ms: trade.timestamp.timestamp_millis(),
            price: trade.price,
            quantity: trade.quantity,
            side: if trade.side.is_sell() { 1 } else { 0 },
        }
    }
}

#[allow(deprecated)]
impl From<&Candle> for SimulationTick {
    fn from(candle: &Candle) -> Self {
        Self {
            timestamp_ms: candle.timestamp.timestamp_millis(),
            price: candle.close,
            quantity: candle.volume,
            side: 0, // Candles don't have a side, default to buy
        }
    }
}

#[allow(deprecated)]
impl From<&MarketData> for SimulationTick {
    fn from(data: &MarketData) -> Self {
        match data {
            MarketData::Candle(c) => SimulationTick::from(c),
            MarketData::Trade(t) => SimulationTick::from(t),
            MarketData::Generic(g) => SimulationTick::from(g),
            MarketData::PoolSwap(s) => SimulationTick {
                timestamp_ms: s.timestamp.timestamp_millis(),
                price: s.price(),
                quantity: s.amount_in,
                side: 0,
            },
            MarketData::OptionCandle(c) => SimulationTick {
                timestamp_ms: c.timestamp.timestamp_millis(),
                price: c.close,
                quantity: c.volume,
                side: 0, // Option candles don't have a side, same as plain candles
            },
        }
    }
}

/// Convert a slice of MarketData to Vec<SimulationTick>
/// This reduces memory by ~10x for large datasets
///
/// **DEPRECATED**: Use `data_prep::SimulationTick` conversions instead.
#[deprecated(note = "Use data_prep::SimulationTick conversions instead")]
#[allow(deprecated)]
pub fn to_simulation_ticks(data: &[MarketData]) -> Vec<SimulationTick> {
    data.iter().map(SimulationTick::from).collect()
}

/// Query parameters for data retrieval
#[derive(Debug, Clone)]
pub struct QueryParams {
    pub symbol: String,
    pub exchange: String,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
}

/// Monte Carlo simulation parameters
#[derive(Debug, Clone)]
pub struct MonteCarloParams {
    pub iterations: usize,
    pub window_minutes: i32,
    pub seed: Option<u64>,
    pub use_bootstrap: bool,
    pub block_size: Option<usize>,
}

#[cfg(feature = "postgres")]
impl DataLoader {
    /// Create a new DataLoader instance
    pub fn new(pool: Arc<deadpool::Pool<AsyncPgConnection>>) -> Self {
        Self {
            pool,
            cache: Arc::new(cache::DataCache::new()),
        }
    }

    // NOTE: load_candles() removed - candle data is now loaded from Massive/Polygon API via MassiveDataProvider.
    // See massive_provider.rs for the current data loading implementation.
    // The following deprecated code was removed:
    // - load_candles() - queried databaseschema::ops::candles_ops::get_candles_*
    // - filter_candles() - helper for filtering candle results
    //
    // Historical trades/orderbook data comes from Massive/Polygon API, not stored in DB.
    // Dashboard candles are still stored in DB for UI purposes but not used in backtesting.

    /// Clear the cache
    pub async fn invalidate_cache(&self) {
        self.cache.clear().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_params_creation() {
        let params = QueryParams {
            symbol: "BTCUSD".to_string(),
            exchange: "BINANCE".to_string(),
            start_time: None,
            end_time: None,
            limit: Some(100),
        };
        assert_eq!(params.symbol, "BTCUSD");
        assert_eq!(params.limit, Some(100));
    }
}
