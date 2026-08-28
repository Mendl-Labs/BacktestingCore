//! Source-agnostic market data provider trait and factory.
//!
//! The [`MarketDataProvider`] trait abstracts over different historical data
//! sources.  Consumers depend on the trait, and the concrete implementation is
//! selected at runtime via [`create_provider`] or [`create_provider_with_config`].
//!
//! ## Adding a new provider
//!
//! 1. Create a new module (e.g. `my_provider.rs`).
//! 2. Implement [`MarketDataProvider`] for your struct.
//! 3. Register it in [`create_provider`] below.

use async_trait::async_trait;
use std::fmt;

use chrono::NaiveDate;

use crate::models::{
    CacheStats, DataRequest, OptionCandle, OptionChainSnapshot, OptionContractRef, TickerInfo,
    TickerQuery, TickerSnapshot,
};
use crate::massive_provider::MassiveDataProvider;
use crate::sui_dex_provider::SuiDexDataProvider;
use crate::MarketData;

// ============================================================================
// Provider Error
// ============================================================================

/// Unified error type returned by any [`MarketDataProvider`].
#[derive(Debug)]
pub enum ProviderError {
    /// Environment not configured (e.g. missing API key)
    NotConfigured(String),
    /// Network / API error
    Fetch(String),
    /// No data available for the requested range
    NoData {
        exchange: String,
        symbol: String,
        from: String,
        to: String,
    },
    /// Rate-limited by upstream API
    RateLimited { retry_after_secs: u32 },
    /// Catch-all
    Other(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConfigured(msg) => write!(f, "Provider not configured: {}", msg),
            Self::Fetch(msg) => write!(f, "Fetch error: {}", msg),
            Self::NoData { exchange, symbol, from, to } => {
                write!(f, "No data for {} on {} between {} and {}", symbol, exchange, from, to)
            }
            Self::RateLimited { retry_after_secs } => {
                write!(f, "Rate limited — retry after {}s", retry_after_secs)
            }
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}


// ============================================================================
// MarketDataProvider trait
// ============================================================================

/// Trait that all market data providers must implement.
///
/// Providers fetch historical OHLCV data and return a sorted
/// `Vec<MarketData>` ready for the simulation engine.
#[async_trait]
pub trait MarketDataProvider: Send + Sync + std::fmt::Debug {
    /// Human-readable name of this provider (e.g. "Massive (Polygon.io)").
    fn name(&self) -> &str;

    /// Fetch market data for the given request.
    ///
    /// Implementations should handle caching, retries, and pagination
    /// internally.  The returned vector **must** be sorted by timestamp.
    async fn fetch(&self, request: &DataRequest) -> Result<Vec<MarketData>, ProviderError>;

    /// Return cache statistics, if the provider has a cache.
    async fn cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// Clear internal caches.
    async fn clear_cache(&self) {
        // Default: no-op
    }

    /// List/search tickers this provider actually supports. Not every
    /// provider implements ticker discovery (e.g. the Sui DEX provider has
    /// no such catalog) — the default returns an empty list rather than an
    /// error, so callers can treat "unsupported" the same as "no matches"
    /// instead of having to special-case it.
    async fn list_tickers(&self, _query: &TickerQuery) -> Result<Vec<TickerInfo>, ProviderError> {
        Ok(Vec::new())
    }

    /// List option contracts for `underlying` as of `as_of` (the reference
    /// date used to determine which contracts existed/were listed). Default
    /// returns an empty list, same rationale as `list_tickers`: most
    /// providers have no options catalog at all, and "unsupported" should
    /// read the same as "no matches" rather than requiring callers to
    /// special-case it.
    async fn list_option_contracts(
        &self,
        _underlying: &str,
        _as_of: NaiveDate,
    ) -> Result<Vec<OptionContractRef>, ProviderError> {
        Ok(Vec::new())
    }

    /// Fetch OHLCV bars for a single option contract across a date range.
    /// Default returns an empty list (no options support).
    async fn fetch_option_aggregates(
        &self,
        _contract_ticker: &str,
        _from: NaiveDate,
        _to: NaiveDate,
    ) -> Result<Vec<OptionCandle>, ProviderError> {
        Ok(Vec::new())
    }

    /// Fetch a point-in-time greeks/IV/open-interest snapshot for every
    /// listed contract on `underlying` expiring on or before
    /// `expiration_lte` (when set -- `None` means no server-side expiration
    /// bound). Default returns an empty list -- snapshot/greeks data
    /// commonly requires a higher account tier than base
    /// contracts/aggregates, so an empty result here means either "not
    /// supported by this provider" or "not included in this account's plan,"
    /// and callers must treat both the same way (advisory, never blocking).
    async fn fetch_option_snapshot(
        &self,
        _underlying: &str,
        _expiration_lte: Option<NaiveDate>,
    ) -> Result<Vec<OptionChainSnapshot>, ProviderError> {
        Ok(Vec::new())
    }

    /// Current (or, outside trading hours, most recent completed day's)
    /// volume/price snapshot for every actively-traded ticker in `market`
    /// ("stocks", "crypto", or "forex") -- lets a caller rank ticker-discovery
    /// candidates by real liquidity instead of a discovery endpoint's own
    /// arbitrary default ordering. Default returns an empty list, same
    /// rationale as `list_tickers`/`fetch_option_snapshot`: most providers
    /// have no such market-wide snapshot at all, or it requires a higher
    /// account tier, and an empty result here should read as "no ranking
    /// signal available" (caller falls back to unranked order) rather than
    /// a hard error every unsupported/ungated caller has to special-case.
    async fn snapshot_market(&self, _market: &str) -> Result<Vec<TickerSnapshot>, ProviderError> {
        Ok(Vec::new())
    }
}

// ============================================================================
// Factory functions
// ============================================================================

/// Which provider backend to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderKind {
    /// Massive / Polygon.io REST API — returns OHLCV candles.
    Massive,
    /// Sui DEX (Cetus + DeepBook) via The Graph subgraphs.
    SuiDex,
}

impl ProviderKind {
    /// Detect the right provider from environment variables.
    ///
    /// Returns `Massive` when `MASSIVE_API_KEY` is set.
    /// Panics are avoided: missing key is surfaced as a runtime error at
    /// [`create_provider`] time, not here.
    pub fn from_env() -> Self {
        Self::Massive
    }

    /// Select the correct provider based on exchange name.
    ///
    /// DEX exchanges ("cetus", "deepbook") route to the Sui subgraph provider;
    /// everything else uses the Massive/Polygon provider.
    pub fn for_exchange(exchange: &str) -> Self {
        match exchange.to_lowercase().as_str() {
            "cetus" | "deepbook" => Self::SuiDex,
            _ => Self::Massive,
        }
    }
}

/// Create a market data provider from environment variables.
///
/// Returns `Err(ProviderError::NotConfigured)` when `MASSIVE_API_KEY` is not set.
pub fn create_provider() -> Result<Box<dyn MarketDataProvider>, ProviderError> {
    create_provider_for(ProviderKind::from_env())
}

/// Create a specific provider kind from environment variables.
pub fn create_provider_for(kind: ProviderKind) -> Result<Box<dyn MarketDataProvider>, ProviderError> {
    match kind {
        ProviderKind::Massive => {
            let api_key = std::env::var("MASSIVE_API_KEY").unwrap_or_default();
            if api_key.is_empty() {
                return Err(ProviderError::NotConfigured(
                    "MASSIVE_API_KEY is not set — set it to your Polygon.io API key".into(),
                ));
            }
            let provider = MassiveDataProvider::from_env()
                .map_err(ProviderError::from)?;
            Ok(Box::new(provider))
        }
        ProviderKind::SuiDex => {
            Ok(Box::new(SuiDexDataProvider::from_env()))
        }
    }
}

/// Create a provider with an explicit API key.
pub fn create_provider_with_config(
    _kind: ProviderKind,
    _api_key: String,
    _config: crate::models::ProviderConfig,
) -> Result<Box<dyn MarketDataProvider>, ProviderError> {
    create_provider_for(ProviderKind::Massive)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ProviderError Display ----

    #[test]
    fn test_provider_error_not_configured_display() {
        let e = ProviderError::NotConfigured("missing key".into());
        assert!(format!("{}", e).contains("Provider not configured"));
        assert!(format!("{}", e).contains("missing key"));
    }

    #[test]
    fn test_provider_error_fetch_display() {
        let e = ProviderError::Fetch("timeout".into());
        assert!(format!("{}", e).contains("Fetch error"));
    }

    #[test]
    fn test_provider_error_no_data_display() {
        let e = ProviderError::NoData {
            exchange: "massive".into(),
            symbol: "BTC/USDT".into(),
            from: "2024-01-01".into(),
            to: "2024-01-31".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("BTC/USDT"));
        assert!(msg.contains("massive"));
    }

    #[test]
    fn test_provider_error_rate_limited_display() {
        let e = ProviderError::RateLimited { retry_after_secs: 60 };
        assert!(format!("{}", e).contains("60s"));
    }

    #[test]
    fn test_provider_error_other_display() {
        let e = ProviderError::Other("unknown".into());
        assert_eq!(format!("{}", e), "unknown");
    }

    #[test]
    fn test_provider_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ProviderError::Other("test".into()));
        let _ = format!("{}", e);
    }

    // ---- ProviderKind ----

    #[test]
    fn test_provider_kind_from_env() {
        let kind = ProviderKind::from_env();
        assert_eq!(kind, ProviderKind::Massive);
    }

    #[test]
    fn test_provider_kind_debug() {
        let kind = ProviderKind::Massive;
        assert_eq!(format!("{:?}", kind), "Massive");
    }

    #[test]
    fn test_provider_kind_clone_eq() {
        let a = ProviderKind::Massive;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- Factory functions ----

    #[test]
    fn test_create_provider_for_massive_no_key() {
        // Without MASSIVE_API_KEY env var, should return NotConfigured error
        std::env::remove_var("MASSIVE_API_KEY");
        let result = create_provider_for(ProviderKind::Massive);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProviderError::NotConfigured(_)));
    }
}
