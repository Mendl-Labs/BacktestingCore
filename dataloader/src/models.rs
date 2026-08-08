//! Vendor-agnostic data models for the dataloader crate
//!
//! These types are used by all data providers (Tardis, database, file uploads, etc.)
//! and should not contain any vendor-specific logic.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

// ============================================================================
// Supported Exchanges
// ============================================================================

/// Exchanges supported by the platform
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Exchange {
    Kraken,
    Coinbase,
    #[serde(rename = "binance-us")]
    BinanceUs,
    Gemini,
    Binance,
    Bybit,
    Okx,
    Deribit,
    /// Breakout prop firm — uses Kraken data feed
    Breakout,
    /// Any other exchange (stores the raw exchange ID)
    #[serde(untagged)]
    Other(String),
}

impl Exchange {
    /// Parse from string (flexible matching). Always succeeds — unknown exchanges
    /// become `Other(id)` which stores the raw ID for data fetching.
    pub fn from_str(s: &str) -> Self {
        let normalized = s.to_lowercase().replace(['_', '-'], "");
        match normalized.as_str() {
            "kraken" => Self::Kraken,
            "coinbase" | "coinbasepro" => Self::Coinbase,
            "binanceus" => Self::BinanceUs,
            "gemini" => Self::Gemini,
            "binance" => Self::Binance,
            "bybit" => Self::Bybit,
            "okx" | "okex" => Self::Okx,
            "deribit" => Self::Deribit,
            "breakout" => Self::Breakout,
            _ => Self::Other(s.to_lowercase()),
        }
    }

    /// Get display name
    pub fn display_name(&self) -> String {
        match self {
            Self::Kraken => "Kraken".to_string(),
            Self::Coinbase => "Coinbase".to_string(),
            Self::BinanceUs => "Binance US".to_string(),
            Self::Gemini => "Gemini".to_string(),
            Self::Binance => "Binance".to_string(),
            Self::Bybit => "Bybit".to_string(),
            Self::Okx => "OKX".to_string(),
            Self::Deribit => "Deribit".to_string(),
            Self::Breakout => "Breakout".to_string(),
            Self::Other(id) => {
                // Capitalize first letter of each word
                id.split('-')
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().chain(c).collect(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    /// Get well-known exchanges (does not include Other)
    pub fn well_known() -> &'static [Exchange] {
        &[
            Self::Kraken,
            Self::Coinbase,
            Self::BinanceUs,
            Self::Gemini,
            Self::Binance,
            Self::Bybit,
            Self::Okx,
            Self::Deribit,
            Self::Breakout,
        ]
    }

    /// Returns true if this is a well-known exchange with optimized symbol handling
    pub fn is_well_known(&self) -> bool {
        !matches!(self, Self::Other(_))
    }

    /// Get the lowercase identifier string for this exchange.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Kraken => "kraken",
            Self::Coinbase => "coinbase",
            Self::BinanceUs => "binanceus",
            Self::Gemini => "gemini",
            Self::Binance => "binance",
            Self::Bybit => "bybit",
            Self::Okx => "okx",
            Self::Deribit => "deribit",
            Self::Breakout => "breakout",
            Self::Other(id) => id.as_str(),
        }
    }
}

impl std::fmt::Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// Data Request Configuration
// ============================================================================

/// What type of data to fetch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Trade ticks only (smaller, faster)
    TradesOnly,
    /// Trades + L2 orderbook snapshots (larger, for market making)
    TradesAndOrderbook,
}

/// OHLCV candle granularity for Massive (Polygon.io) requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleGranularity {
    Minute1,
    Minute5,
    Minute15,
    Hour1,
    Hour4,
    Day1,
}

impl CandleGranularity {
    /// Returns `(multiplier, timespan)` for the Polygon `/v2/aggs` endpoint.
    pub fn to_polygon_params(self) -> (u32, &'static str) {
        match self {
            Self::Minute1  => (1,  "minute"),
            Self::Minute5  => (5,  "minute"),
            Self::Minute15 => (15, "minute"),
            Self::Hour1    => (1,  "hour"),
            Self::Hour4    => (4,  "hour"),
            Self::Day1     => (1,  "day"),
        }
    }

    /// Construct from candle interval minutes (e.g. from `candle_interval_minutes` form field).
    /// Unrecognised values fall back to `Hour1`.
    pub fn from_minutes(minutes: i64) -> Self {
        match minutes {
            1    => Self::Minute1,
            5    => Self::Minute5,
            15   => Self::Minute15,
            60   => Self::Hour1,
            240  => Self::Hour4,
            1440 => Self::Day1,
            _    => Self::Hour1,
        }
    }

    /// Interval in minutes (useful for logging / passing back to the aggregation layer).
    pub fn as_minutes(self) -> i64 {
        match self {
            Self::Minute1  => 1,
            Self::Minute5  => 5,
            Self::Minute15 => 15,
            Self::Hour1    => 60,
            Self::Hour4    => 240,
            Self::Day1     => 1440,
        }
    }
}

/// Request for historical market data
#[derive(Debug, Clone)]
pub struct DataRequest {
    /// Exchange to fetch from
    pub exchange: Exchange,
    /// Trading symbol (e.g., "BTC-USD", "ETH-USDT", "AAPL")
    pub symbol: String,
    /// Asset class: "crypto" (default) or "stocks".
    pub asset_class: Option<String>,
    /// Start date (inclusive)
    pub from: NaiveDate,
    /// End date (inclusive)
    pub to: NaiveDate,
    /// What data to fetch
    pub data_type: DataType,
    /// Candle granularity — used by providers that return OHLCV directly (e.g. Massive).
    /// `None` means the provider should choose its default (typically `Hour1`).
    pub granularity: Option<CandleGranularity>,
}

impl DataRequest {
    pub fn new(exchange: Exchange, symbol: impl Into<String>, from: NaiveDate, to: NaiveDate) -> Self {
        Self {
            exchange,
            symbol: symbol.into(),
            asset_class: None,
            from,
            to,
            data_type: DataType::TradesOnly,
            granularity: None,
        }
    }

    pub fn with_asset_class(mut self, asset_class: impl Into<String>) -> Self {
        self.asset_class = Some(asset_class.into());
        self
    }

    pub fn with_orderbook(mut self) -> Self {
        self.data_type = DataType::TradesAndOrderbook;
        self
    }

    pub fn with_granularity(mut self, granularity: CandleGranularity) -> Self {
        self.granularity = Some(granularity);
        self
    }

    /// Get number of days in this request
    pub fn num_days(&self) -> i64 {
        (self.to - self.from).num_days() + 1
    }
}

// ============================================================================
// Cache Types (vendor-agnostic)
// ============================================================================

/// Configuration for data provider caching
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Maximum total memory usage (bytes)
    pub max_memory_bytes: usize,
    /// Time-to-live for cache entries
    pub ttl: std::time::Duration,
    /// Whether caching is enabled
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            ttl: std::time::Duration::from_secs(3600), // 1 hour
            enabled: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub memory_bytes: usize,
    pub max_entries: usize,
    pub max_memory_bytes: usize,
}

// ============================================================================
// Ticker discovery (list/search available symbols)
// ============================================================================

/// Query parameters for [`crate::MarketDataProvider::list_tickers`]. All
/// fields optional — an empty query means "give me some active tickers,
/// provider's default ordering."
#[derive(Debug, Clone, Default)]
pub struct TickerQuery {
    /// Provider-level market category, e.g. "stocks", "crypto", "fx".
    /// Left to the caller to map from the platform's own asset_class
    /// ("forex" -> "fx") since that mapping is provider-specific.
    pub market: Option<String>,
    /// Free-text search over ticker symbol and company/asset name.
    pub search: Option<String>,
    /// Filter by active trading status. `None` = provider default (usually
    /// active-only).
    pub active: Option<bool>,
    /// Max results to return. Providers may cap this lower than requested.
    pub limit: u32,
}

/// One ticker/symbol as returned by a provider's ticker-discovery endpoint.
/// Deliberately narrow — only the fields a caller deciding "is this a real,
/// tradeable symbol" actually needs, not the full reference-data record
/// (CIK, FIGI, currency metadata, etc.) a provider might expose.
#[derive(Debug, Clone)]
pub struct TickerInfo {
    pub ticker: String,
    pub name: Option<String>,
    pub market: Option<String>,
    pub active: Option<bool>,
    pub primary_exchange: Option<String>,
    /// ISO date string when this ticker was delisted, if ever. `None` means
    /// currently listed (or the provider simply didn't report one). Exists so
    /// callers can flag survivorship-bias risk: a symbol delisted at or before
    /// the requested backtest end date was not actually tradeable for that
    /// whole window the way a naive "it has data" check would suggest.
    pub delisted_utc: Option<String>,
}

// ============================================================================
// Options-chain data (Phase 0 of stock-options support)
// ============================================================================

/// Call or put. Deliberately separate from `strategy`/`derivatives` crates'
/// own option-kind representations (`InstrumentKind::Call{strike,expiry}` /
/// `Put{..}`) — this is the vendor-facing, data-layer shape; the simulation
/// layer's `InstrumentKind` is what a strategy/fill actually operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    Call,
    Put,
}

/// One contract row as returned by a provider's options-contracts reference
/// endpoint (e.g. Polygon's `/v3/reference/options/contracts`) — narrow, like
/// `TickerInfo`, since this is a discovery/catalog shape, not a full
/// reference-data record.
#[derive(Debug, Clone)]
pub struct OptionContractRef {
    /// Full contract ticker in the provider's own format (e.g. Polygon's OCC
    /// style `O:SPY250620C00450000`).
    pub ticker: String,
    /// The underlying's plain symbol (e.g. `"SPY"`).
    pub underlying: String,
    pub strike: f64,
    pub expiration: NaiveDate,
    pub contract_type: OptionType,
}

/// One OHLCV bar for a single options contract — same shape as `Candle`
/// (see `dataloader::Candle`) plus the contract-identifying fields a plain
/// equity/crypto candle has no need for. Kept as a distinct type (rather
/// than bolting strike/expiration onto `Candle` itself) so every existing
/// `Candle` construction/consumption site is untouched by this addition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionCandle {
    pub contract_ticker: String,
    pub underlying: String,
    pub strike: f64,
    pub expiration: NaiveDate,
    pub contract_type: OptionType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Point-in-time greeks/IV/open-interest snapshot for one contract, as
/// returned by a provider's options-snapshot endpoint (e.g. Polygon's
/// `/v3/snapshot/options/{underlying}`). All fields optional — snapshot
/// endpoints commonly gate greeks/IV behind a higher account tier than the
/// base contracts/aggregates endpoints, so a provider may only be able to
/// populate a subset (or none) depending on what the caller's plan includes.
#[derive(Debug, Clone, Default)]
pub struct OptionChainSnapshot {
    pub contract_ticker: String,
    pub implied_volatility: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub open_interest: Option<i64>,
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for a market data provider
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Maximum concurrent HTTP requests
    pub max_concurrency: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Chunk size for parallel fetching (days per chunk)
    pub chunk_days: i64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 60,
            request_timeout_secs: 60,
            max_retries: 2,
            initial_backoff_ms: 200,
            cache: CacheConfig::default(),
            chunk_days: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Exchange ──

    #[test]
    fn exchange_from_str_known_exchanges() {
        assert_eq!(Exchange::from_str("kraken"), Exchange::Kraken);
        assert_eq!(Exchange::from_str("KRAKEN"), Exchange::Kraken);
        assert_eq!(Exchange::from_str("coinbase"), Exchange::Coinbase);
        assert_eq!(Exchange::from_str("coinbase-pro"), Exchange::Coinbase);
        assert_eq!(Exchange::from_str("COINBASE_PRO"), Exchange::Coinbase);
        assert_eq!(Exchange::from_str("binance-us"), Exchange::BinanceUs);
        assert_eq!(Exchange::from_str("binanceus"), Exchange::BinanceUs);
        assert_eq!(Exchange::from_str("okx"), Exchange::Okx);
        assert_eq!(Exchange::from_str("okex"), Exchange::Okx);
        assert_eq!(Exchange::from_str("bybit"), Exchange::Bybit);
        assert_eq!(Exchange::from_str("deribit"), Exchange::Deribit);
        assert_eq!(Exchange::from_str("breakout"), Exchange::Breakout);
    }

    #[test]
    fn exchange_from_str_unknown() {
        match Exchange::from_str("totally-unknown") {
            Exchange::Other(s) => assert_eq!(s, "totally-unknown"),
            _ => panic!("expected Other"),
        }
    }

    #[test]
    fn exchange_display_name_known() {
        assert_eq!(Exchange::Kraken.display_name(), "Kraken");
        assert_eq!(Exchange::BinanceUs.display_name(), "Binance US");
        assert_eq!(Exchange::Okx.display_name(), "OKX");
    }

    #[test]
    fn exchange_display_name_other() {
        let e = Exchange::Other("my-exchange".to_string());
        assert_eq!(e.display_name(), "My Exchange");
    }

    #[test]
    fn exchange_well_known() {
        let known = Exchange::well_known();
        assert_eq!(known.len(), 9);
        assert!(known.contains(&Exchange::Kraken));
    }

    #[test]
    fn exchange_is_well_known() {
        assert!(Exchange::Kraken.is_well_known());
        assert!(!Exchange::Other("x".to_string()).is_well_known());
    }

    #[test]
    fn exchange_display_trait() {
        assert_eq!(format!("{}", Exchange::Kraken), "Kraken");
    }

    #[test]
    fn exchange_serde_roundtrip() {
        let e = Exchange::Kraken;
        let json = serde_json::to_string(&e).unwrap();
        let e2: Exchange = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }

    // ── DataRequest ──

    #[test]
    fn data_request_new() {
        let from = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let to = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
        let req = DataRequest::new(Exchange::Kraken, "BTC-USD", from, to);
        assert_eq!(req.symbol, "BTC-USD");
        assert_eq!(req.data_type, DataType::TradesOnly);
    }

    #[test]
    fn data_request_with_orderbook() {
        let from = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let to = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
        let req = DataRequest::new(Exchange::Kraken, "BTC-USD", from, to).with_orderbook();
        assert_eq!(req.data_type, DataType::TradesAndOrderbook);
    }

    #[test]
    fn data_request_num_days() {
        let from = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let to = NaiveDate::from_ymd_opt(2023, 1, 10).unwrap();
        let req = DataRequest::new(Exchange::Kraken, "BTC-USD", from, to);
        assert_eq!(req.num_days(), 10); // inclusive
    }

    #[test]
    fn data_request_num_days_single() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        let req = DataRequest::new(Exchange::Kraken, "BTC-USD", d, d);
        assert_eq!(req.num_days(), 1);
    }

    // ── CacheConfig ──

    #[test]
    fn cache_config_default() {
        let c = CacheConfig::default();
        assert_eq!(c.max_entries, 100);
        assert!(c.enabled);
        assert_eq!(c.ttl.as_secs(), 3600);
    }

    // ── ProviderConfig ──

    #[test]
    fn provider_config_default() {
        let p = ProviderConfig::default();
        assert_eq!(p.max_concurrency, 60);
        assert_eq!(p.max_retries, 2);
        assert_eq!(p.chunk_days, 1);
    }
}
