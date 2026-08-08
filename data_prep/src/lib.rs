//! Data Preparation Library for Backtesting
//!
//! This crate provides the "librarian" functionality:
//! 1. Stream historical data from PostgreSQL in chunks
//! 2. Convert heavy HistoricalOrder structs to lightweight SimulationTick
//! 3. Optionally precompute features (volatility, EMA, etc.)
//! 4. Write to memory-mappable binary files
//!
//! # Architecture
//!
//! ```text
//!                    ┌─────────────────────────────────────────┐
//!                    │          Data Prep Job                  │
//!                    │                                         │
//!                    │  ┌─────────────┐    ┌───────────────┐  │
//!   PostgreSQL ─────►│  │ DB Streamer │───►│ Binary Writer │──┼──► .bin file
//!   (7M rows)        │  │ (10K chunks)│    │               │  │    (280 MB)
//!                    │  └─────────────┘    └───────────────┘  │
//!                    │         │                              │
//!                    │         ▼                              │
//!                    │  ┌─────────────┐                       │
//!                    │  │  Feature    │ (optional)            │
//!                    │  │  Computer   │                       │
//!                    │  └─────────────┘                       │
//!                    └─────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use data_prep::{DataPrepConfig, prepare_data};
//!
//! let config = DataPrepConfig {
//!     symbol: "BTC/USD".to_string(),
//!     exchange: "Kraken".to_string(),
//!     output_dir: PathBuf::from("/data/historical"),
//!     chunk_size: 10_000,
//!     compute_features: true,
//! };
//!
//! prepare_data(&db_pool, &config).await?;
//! ```

pub mod binary_format;
pub mod db_streamer;
pub mod feature_compute;
pub mod logging_facade;
pub mod tick;
pub mod orderbook_binary;

// Re-export main types
pub use binary_format::{BinaryFileHeader, BinaryFileWriter, BinaryFileReader, SlicedTickSource, HEADER_SIZE};
pub use orderbook_binary::{
    OrderBookEventTick, OrderBookBinaryWriter, OrderBookBinaryReader, 
    OrderBookBinaryHeader, EventTypeCode, SideCode
};
pub use db_streamer::DatabaseStreamer;
pub use feature_compute::FeatureComputer;
pub use tick::SimulationTick;

use std::path::PathBuf;
#[cfg(feature = "postgres")]
use anyhow::Result;

/// Configuration for data preparation job
#[derive(Debug, Clone)]
pub struct DataPrepConfig {
    /// Trading symbol (e.g., "BTC/USD")
    pub symbol: String,
    /// Exchange name (e.g., "Kraken")
    pub exchange: String,
    /// Output directory for binary files
    pub output_dir: PathBuf,
    /// Number of rows to fetch per database query
    pub chunk_size: usize,
    /// Whether to precompute features (volatility, EMA, etc.)
    pub compute_features: bool,
    /// Start time filter (optional)
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    /// End time filter (optional)
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for DataPrepConfig {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            exchange: String::new(),
            output_dir: PathBuf::from("/data/historical"),
            chunk_size: 10_000,
            compute_features: false,
            start_time: None,
            end_time: None,
        }
    }
}

/// Main entry point: prepare data for backtesting
/// 
/// NOTE: Historical order data preparation from database has been removed.
/// Historical data is now loaded directly from Polygon/Massive API (see dataloader::massive_provider).
/// This function is kept for API compatibility but returns an error.
#[cfg(feature = "postgres")]
pub async fn prepare_data(
    _pool: &diesel_async::pooled_connection::deadpool::Pool<diesel_async::AsyncPgConnection>,
    config: &DataPrepConfig,
) -> Result<PrepareResult> {
    crate::log_warn!(crate::logging_facade::DATA_PREP_LOGGER, "Database-based data preparation has been deprecated (symbol={}, exchange={}). Use Massive/Polygon API for historical data.", config.symbol, config.exchange);
    
    // Return error explaining the change
    Err(anyhow::anyhow!(
        "Database-based historical data preparation has been removed. \
        Historical market data should now be loaded directly from Tardis API. \
        See dataloader::massive_provider for streaming historical data."
    ))
}

/// Result of data preparation
#[derive(Debug)]
pub struct PrepareResult {
    pub output_path: PathBuf,
    pub total_ticks: u64,
    pub file_size_bytes: u64,
}

/// Prepare full L3 order book data (new, modify, cancel, trade events)
/// 
/// NOTE: Historical orderbook data preparation from database has been removed.
/// Historical data is now loaded directly from Polygon/Massive API (see dataloader::massive_provider).
/// This function is kept for API compatibility but returns an error.
#[cfg(feature = "postgres")]
pub async fn prepare_orderbook_data(
    _pool: &diesel_async::pooled_connection::deadpool::Pool<diesel_async::AsyncPgConnection>,
    config: &DataPrepConfig,
) -> Result<PrepareResult> {
    crate::log_warn!(crate::logging_facade::DATA_PREP_LOGGER, "Database-based orderbook data preparation has been deprecated (symbol={}, exchange={}). Use Massive/Polygon API for historical data.", config.symbol, config.exchange);
    
    // Return error explaining the change
    Err(anyhow::anyhow!(
        "Database-based historical orderbook data preparation has been removed. \
        Historical market data should now be loaded directly from Tardis API. \
        See dataloader::massive_provider for streaming historical data."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_prep_config_default() {
        let config = DataPrepConfig::default();
        assert_eq!(config.symbol, "");
        assert_eq!(config.exchange, "");
        assert_eq!(config.output_dir, PathBuf::from("/data/historical"));
        assert_eq!(config.chunk_size, 10_000);
        assert!(!config.compute_features);
        assert!(config.start_time.is_none());
        assert!(config.end_time.is_none());
    }

    #[test]
    fn test_data_prep_config_custom() {
        let config = DataPrepConfig {
            symbol: "BTC/USD".to_string(),
            exchange: "Kraken".to_string(),
            output_dir: PathBuf::from("/tmp/test"),
            chunk_size: 5_000,
            compute_features: true,
            start_time: None,
            end_time: None,
        };
        assert_eq!(config.symbol, "BTC/USD");
        assert_eq!(config.exchange, "Kraken");
        assert_eq!(config.chunk_size, 5_000);
        assert!(config.compute_features);
    }

    #[test]
    fn test_data_prep_config_clone() {
        let config = DataPrepConfig {
            symbol: "ETH/USD".to_string(),
            exchange: "Binance".to_string(),
            ..DataPrepConfig::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.symbol, "ETH/USD");
        assert_eq!(cloned.exchange, "Binance");
    }

    #[test]
    fn test_prepare_result_debug() {
        let result = PrepareResult {
            output_path: PathBuf::from("/tmp/output.bin"),
            total_ticks: 1_000_000,
            file_size_bytes: 50_000_000,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("1000000"));
        assert!(debug_str.contains("50000000"));
    }
}
