//! Pool data provider trait for LP backtesting.
//!
//! Abstracts over different pool data sources (The Graph, custom CSV, etc.)

use async_trait::async_trait;
use chrono::NaiveDate;

use crate::pool_state::PoolSwapEvent;
use crate::provider::ProviderError;

/// Request for historical pool swap data.
#[derive(Debug, Clone)]
pub struct PoolDataRequest {
    pub pool_id: String,
    pub exchange: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
}

impl PoolDataRequest {
    pub fn new(pool_id: impl Into<String>, exchange: impl Into<String>, start: NaiveDate, end: NaiveDate) -> Self {
        Self {
            pool_id: pool_id.into(),
            exchange: exchange.into(),
            start_date: start,
            end_date: end,
        }
    }
}

/// Trait for fetching historical pool swap events for LP backtesting.
#[async_trait]
pub trait PoolDataProvider: Send + Sync {
    async fn fetch_pool_swaps(&self, request: &PoolDataRequest) -> Result<Vec<PoolSwapEvent>, ProviderError>;
}
