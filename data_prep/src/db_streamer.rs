//! Database streaming stub
//!
//! NOTE: Historical market data streaming from database has been removed.
//! Historical data is now loaded from Polygon/Massive API (see dataloader::massive_provider).
//!
//! This module provides stub implementations for backward compatibility.

#[allow(unused_imports)]
use anyhow::Result;
#[allow(unused_imports)]
use chrono::{DateTime, Utc};

#[cfg(feature = "postgres")]
use diesel_async::AsyncPgConnection;
#[cfg(feature = "postgres")]
use diesel_async::pooled_connection::deadpool::Pool;

/// Database streamer stub
/// 
/// NOTE: Historical order streaming has been removed.
/// Use Massive/Polygon API for historical market data (see dataloader::massive_provider).
#[cfg(feature = "postgres")]
pub struct DatabaseStreamer {
    #[allow(dead_code)]
    pool: Pool<AsyncPgConnection>,
    #[allow(dead_code)]
    chunk_size: usize,
}

#[cfg(feature = "postgres")]
impl DatabaseStreamer {
    /// Create a new database streamer
    pub fn new(pool: Pool<AsyncPgConnection>, chunk_size: usize) -> Self {
        Self { pool, chunk_size }
    }
    
    /// Count total orders (deprecated - always returns 0)
    /// 
    /// NOTE: Historical order table has been removed.
    /// Use Tardis API for historical data.
    pub async fn count_orders(
        &self,
        _symbol: &str,
        _exchange: &str,
        _start_time: Option<DateTime<Utc>>,
        _end_time: Option<DateTime<Utc>>,
    ) -> Result<i64> {
        // Historical orders table no longer exists
        // Use Tardis API for historical data
        Ok(0)
    }
    
    /// Count all events (deprecated - always returns 0)
    /// 
    /// NOTE: Historical order table has been removed.
    /// Use Tardis API for historical data.
    pub async fn count_all_events(
        &self,
        _symbol: &str,
        _exchange: &str,
        _start_time: Option<DateTime<Utc>>,
        _end_time: Option<DateTime<Utc>>,
    ) -> Result<i64> {
        // Historical orders table no longer exists
        // Use Tardis API for historical data
        Ok(0)
    }
}

#[cfg(not(feature = "postgres"))]
pub struct DatabaseStreamer;

#[cfg(not(feature = "postgres"))]
impl DatabaseStreamer {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_streamer_non_postgres_creation() {
        #[cfg(not(feature = "postgres"))]
        {
            let _streamer = DatabaseStreamer::new();
        }
    }

    #[test]
    fn test_database_streamer_stub_exists() {
        // Verify the DatabaseStreamer type exists and can be referenced
        let _type_name = std::any::type_name::<DatabaseStreamer>();
        assert!(!_type_name.is_empty());
    }
}
