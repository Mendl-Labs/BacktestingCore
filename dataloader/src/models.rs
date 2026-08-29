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
    /// Reconstruct the universe as it existed on this historical date,
    /// instead of the provider's live/current catalog. `None` = today
    /// (the provider's default). Confirmed live 2026-08-28 against
    /// Polygon/Massive's own docs: `/v3/reference/tickers` accepts a `date`
    /// param that returns tickers active as of that date, distinct from
    /// today's live listing -- e.g. a symbol delisted in 2020 will show up
    /// for `as_of = 2018-01-01` but not for `as_of = None`, and a symbol
    /// that IPO'd in 2022 will NOT show up for `as_of = 2018-01-01`. Exists
    /// to close the exact survivorship-bias gap `get_available_assets`'s
    /// own doc comment used to call out: discovery previously only ever
    /// surfaced currently-active tickers, so any basket built from it could
    /// only ever contain names that survive to today, regardless of how
    /// carefully downstream candle-fetch code checked `delisted_utc` for
    /// mid-window drops.
    pub as_of: Option<NaiveDate>,
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

/// One ticker's current-day trading activity, as returned by a provider's
/// market-wide snapshot endpoint. Exists specifically to let a caller RANK
/// candidates by real liquidity instead of accepting a ticker-discovery
/// endpoint's own arbitrary default ordering (Polygon's `/v3/reference/
/// tickers`, what `TickerInfo`/`list_tickers` are sourced from, returns
/// results alphabetically by ticker symbol with no relevance signal at all
/// -- confirmed live 2026-08-27 that this let an AI research agent's asset
/// selection collapse onto only the handful of famous names it already knew
/// from training, since an alphabetical page of obscure tickers gave it
/// nothing useful to browse).
#[derive(Debug, Clone)]
pub struct TickerSnapshot {
    pub ticker: String,
    /// Last/close price for the current trading day, in the asset's own
    /// quote currency. `0.0` if the provider had no trade yet today (e.g.
    /// stocks queried outside market hours) -- callers should fall back to
    /// `prev_day_notional_volume` in that case rather than treat a $0
    /// notional volume as real.
    pub price: f64,
    /// Today's cumulative traded volume, in the asset's OWN base units --
    /// NOT directly comparable across tickers with wildly different unit
    /// prices (2,000,000 units of a $0.0003 token is a few hundred dollars
    /// of real notional, not a sign of deep liquidity). Callers ranking by
    /// liquidity should use `notional_volume()`/`prev_day_notional_volume()`
    /// below, not this field directly.
    pub day_volume: f64,
    /// Volume-weighted average price for the current trading day -- paired
    /// with `day_volume` to compute real notional (USD-equivalent) volume,
    /// a fairer per-trade price estimate than the single last-trade `price`
    /// for a volume-weighted calculation.
    pub day_vwap: f64,
    /// Same three fields for the PREVIOUS completed trading day -- the
    /// fallback for stocks queried outside market hours, when `day_*` are
    /// all zero because no trade has happened yet today.
    pub prev_day_close: f64,
    pub prev_day_volume: f64,
    pub prev_day_vwap: f64,
}

impl TickerSnapshot {
    /// Real (USD-equivalent) notional volume traded so far today --
    /// `day_volume * day_vwap`, not raw `day_volume` alone. `0.0` when
    /// today has no trade data yet (see `day_volume`'s doc comment); check
    /// `prev_day_notional_volume()` in that case.
    pub fn notional_volume(&self) -> f64 {
        self.day_volume * self.day_vwap
    }

    /// Same computation for the previous completed trading day -- the
    /// fallback ranking signal when today's own snapshot is empty.
    pub fn prev_day_notional_volume(&self) -> f64 {
        self.prev_day_volume * self.prev_day_vwap
    }

    /// Best available notional-volume estimate: today's if it's real
    /// (non-zero), otherwise the previous day's. This is the value callers
    /// ranking candidates by liquidity should actually sort on.
    pub fn best_notional_volume(&self) -> f64 {
        let today = self.notional_volume();
        if today > 0.0 {
            today
        } else {
            self.prev_day_notional_volume()
        }
    }
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

/// Optional range filters for `list_option_contracts`. All `None` by
/// default (unfiltered -- the provider's own default page, which for a
/// liquid underlying can be entirely consumed by the nearest few
/// expirations' full strike ladder, confirmed live 2026-08-29 against
/// Polygon: 1000 unfiltered rows for SPY covered only 4 near-term
/// expirations, none anywhere near a 30-45 DTE target). Setting
/// `expiration_gte`/`expiration_lte`/`strike_gte`/`strike_lte` narrows the
/// server-side result to what's actually needed -- confirmed live the same
/// day that Polygon's `/v3/reference/options/contracts` accepts all four as
/// `field.gte`/`field.lte` query params, returning an exact, unpaginated
/// match (496 rows, no `next_url`) for a combined expiration+strike window.
#[derive(Debug, Clone, Default)]
pub struct OptionContractQuery {
    pub expiration_gte: Option<NaiveDate>,
    pub expiration_lte: Option<NaiveDate>,
    pub strike_gte: Option<f64>,
    pub strike_lte: Option<f64>,
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
    /// From the snapshot row's own `details` sub-object when present,
    /// falling back to parsing the OCC contract ticker otherwise -- a
    /// consumer should never need a second call to `list_option_contracts`
    /// just to learn a row's own strike/expiration/type.
    pub strike: Option<f64>,
    pub expiration: Option<NaiveDate>,
    pub contract_type: Option<OptionType>,
    /// The underlying's own current price, as carried in this same row's
    /// `underlying_asset.price` -- same reading for every row of one
    /// `fetch_option_snapshot` call, exposed per-row since Polygon nests it
    /// there rather than once at the top level.
    pub underlying_price: Option<f64>,
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

    // ── TickerSnapshot ──

    fn snapshot(price: f64, day_volume: f64, day_vwap: f64, prev_close: f64, prev_volume: f64, prev_vwap: f64) -> TickerSnapshot {
        TickerSnapshot {
            ticker: "TEST".to_string(),
            price,
            day_volume,
            day_vwap,
            prev_day_close: prev_close,
            prev_day_volume: prev_volume,
            prev_day_vwap: prev_vwap,
        }
    }

    #[test]
    fn notional_volume_multiplies_volume_by_vwap_not_raw_volume() {
        // The exact confusion this type exists to prevent: a huge raw
        // volume of a cheap token isn't real liquidity on its own.
        let cheap_high_volume = snapshot(0.00034, 2_176_983.0, 0.0003413, 0.0, 0.0, 0.0);
        assert!((cheap_high_volume.notional_volume() - 743.03).abs() < 1.0);
    }

    #[test]
    fn best_notional_volume_falls_back_to_prev_day_when_today_is_zero() {
        // Stocks queried outside market hours: day.* is all zero, not
        // omitted -- must fall back to prevDay rather than report $0.
        let outside_market_hours = snapshot(0.0, 0.0, 0.0, 212.86, 1_278_324.0, 213.9281);
        assert_eq!(outside_market_hours.notional_volume(), 0.0);
        assert!(outside_market_hours.best_notional_volume() > 0.0);
        assert_eq!(outside_market_hours.best_notional_volume(), outside_market_hours.prev_day_notional_volume());
    }

    #[test]
    fn best_notional_volume_prefers_today_when_both_are_real() {
        let active = snapshot(100.0, 1000.0, 99.5, 95.0, 2000.0, 94.0);
        assert_eq!(active.best_notional_volume(), active.notional_volume());
        assert_ne!(active.best_notional_volume(), active.prev_day_notional_volume());
    }

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
