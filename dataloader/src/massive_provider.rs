//! Massive (Polygon.io) Market Data Provider
//!
//! Fetches historical OHLCV candles via the Polygon REST API and returns them
//! as `MarketData::Candle` events.  Consumers that need tick-level data should
//! use a different provider; candle-based strategies work directly with the
//! output of this provider (no further aggregation needed).
//!
//! ## Environment variables
//!
//! | Variable             | Description                              | Default                     |
//! |----------------------|------------------------------------------|-----------------------------|
//! | `MASSIVE_API_KEY`    | API key (required)                       | —                           |
//! | `MASSIVE_API_BASE_URL` | Base URL override (for testing / staging) | `https://api.polygon.io` |
//!
//! ## Pagination
//!
//! Polygon paginates large date ranges via a `next_url` cursor.  This provider
//! follows `next_url` automatically until all candles are retrieved.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{NaiveDate, TimeZone, Utc};
use reqwest::{header, Client};
use serde::Deserialize;
use thiserror::Error;

use crate::models::{
    CandleGranularity, DataRequest, OptionCandle, OptionChainSnapshot, OptionContractRef,
    OptionType, TickerInfo, TickerQuery,
};
use crate::provider::{MarketDataProvider, ProviderError};
use crate::{Candle, MarketData};
use crate::logging_facade::DATALOADER_LOGGER;
use crate::{log_info_structured, log_warn_structured, log_error_structured};

// ============================================================================
// Error Type
// ============================================================================

#[derive(Error, Debug)]
pub enum MassiveProviderError {
    #[error("MASSIVE_API_KEY environment variable not set")]
    MissingApiKey,

    #[error("HTTP error: {0}")]
    HttpError(String),

    #[error("API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limited — retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u32 },

    #[error("No data for {symbol} between {from} and {to}")]
    NoData { symbol: String, from: String, to: String },
}

impl From<MassiveProviderError> for ProviderError {
    fn from(e: MassiveProviderError) -> Self {
        match e {
            MassiveProviderError::MissingApiKey => {
                ProviderError::NotConfigured("MASSIVE_API_KEY not set".into())
            }
            MassiveProviderError::RateLimited { retry_after_secs } => {
                ProviderError::RateLimited { retry_after_secs }
            }
            MassiveProviderError::NoData { symbol, from, to } => {
                ProviderError::NoData {
                    exchange: "massive".into(),
                    symbol,
                    from,
                    to,
                }
            }
            other => ProviderError::Fetch(other.to_string()),
        }
    }
}

// ============================================================================
// Polygon API Response Types
// ============================================================================

/// Aggregate bar returned by `/v2/aggs/ticker/{ticker}/range/…`
#[derive(Debug, Deserialize)]
struct AggBar {
    /// Open price
    o: f64,
    /// High price
    h: f64,
    /// Low price
    l: f64,
    /// Close price
    c: f64,
    /// Volume
    v: f64,
    /// Number of trades
    n: Option<i64>,
    /// Unix timestamp in milliseconds (start of the bar)
    t: i64,
}

/// Top-level response from `/v2/aggs/…`
#[derive(Debug, Deserialize)]
struct AggResponse {
    ticker: Option<String>,
    results: Option<Vec<AggBar>>,
    results_count: Option<i64>,
    /// Cursor for the next page of results
    next_url: Option<String>,
    status: Option<String>,
}

/// One result row from `/v3/reference/tickers`. Only the fields
/// `TickerInfo` actually needs -- the real response also carries
/// cik/composite_figi/share_class_figi/currency metadata/last_updated_utc/
/// delisted_utc, deliberately not modeled here since nothing downstream
/// uses them (see `TickerInfo`'s own doc comment).
#[derive(Debug, Deserialize)]
struct TickerResult {
    ticker: Option<String>,
    name: Option<String>,
    market: Option<String>,
    active: Option<bool>,
    primary_exchange: Option<String>,
    delisted_utc: Option<String>,
}

/// Top-level response from `/v3/reference/tickers`.
#[derive(Debug, Deserialize)]
struct TickerListResponse {
    results: Option<Vec<TickerResult>>,
    next_url: Option<String>,
    status: Option<String>,
}

/// One contract row from `/v3/reference/options/contracts`.
#[derive(Debug, Deserialize)]
struct OptionContractResult {
    ticker: Option<String>,
    underlying_ticker: Option<String>,
    strike_price: Option<f64>,
    expiration_date: Option<String>,
    contract_type: Option<String>,
}

/// Top-level response from `/v3/reference/options/contracts`.
#[derive(Debug, Deserialize)]
struct OptionContractListResponse {
    results: Option<Vec<OptionContractResult>>,
    next_url: Option<String>,
    status: Option<String>,
}

/// The `details` sub-object of one row from `/v3/snapshot/options/{underlying}`.
#[derive(Debug, Deserialize)]
struct OptionSnapshotDetails {
    ticker: Option<String>,
}

/// The `greeks` sub-object of one row from `/v3/snapshot/options/{underlying}`.
/// Optional at the row level too -- Polygon omits this object entirely when
/// the account tier doesn't include greeks, rather than sending nulls.
#[derive(Debug, Deserialize, Default)]
struct OptionSnapshotGreeks {
    delta: Option<f64>,
    gamma: Option<f64>,
    theta: Option<f64>,
    vega: Option<f64>,
}

/// One contract row from `/v3/snapshot/options/{underlying}`.
#[derive(Debug, Deserialize)]
struct OptionSnapshotResult {
    details: Option<OptionSnapshotDetails>,
    greeks: Option<OptionSnapshotGreeks>,
    implied_volatility: Option<f64>,
    open_interest: Option<i64>,
}

/// Top-level response from `/v3/snapshot/options/{underlying}`.
#[derive(Debug, Deserialize)]
struct OptionSnapshotResponse {
    results: Option<Vec<OptionSnapshotResult>>,
    status: Option<String>,
}

// ============================================================================
// Provider Struct
// ============================================================================

#[derive(Debug)]
pub struct MassiveDataProvider {
    api_key: String,
    base_url: String,
    client: Client,
}

impl MassiveDataProvider {
    /// Build from environment variables.
    ///
    /// Returns `Err(MassiveProviderError::MissingApiKey)` when `MASSIVE_API_KEY`
    /// is absent or empty.
    pub fn from_env() -> Result<Self, MassiveProviderError> {
        let api_key = std::env::var("MASSIVE_API_KEY")
            .unwrap_or_default();
        if api_key.is_empty() {
            return Err(MassiveProviderError::MissingApiKey);
        }

        let base_url = std::env::var("MASSIVE_API_BASE_URL")
            .unwrap_or_else(|_| "https://api.polygon.io".into());

        let mut headers = header::HeaderMap::new();
        let auth_value = format!("Bearer {}", api_key);
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&auth_value)
                .map_err(|e| MassiveProviderError::HttpError(e.to_string()))?,
        );

        let client = Client::builder()
            .default_headers(headers)
            // Per-request timeout. Large paginated pages (up to 50k bars) can be slow,
            // so this is generous; transient timeouts are additionally retried per page.
            .timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| MassiveProviderError::HttpError(e.to_string()))?;

        Ok(Self { api_key, base_url, client })
    }

    // ------------------------------------------------------------------
    // Symbol conversion
    // ------------------------------------------------------------------

    /// Strips Massive/Polygon's own ticker prefix (`C:` forex, `X:` crypto,
    /// `I:` indices, `O:` options) if the symbol already arrived in that raw
    /// provider format instead of the platform's own dash convention -- e.g.
    /// copied verbatim from a ticker-search result's `ticker` field instead
    /// of being converted first. Without this, `symbol_to_ticker` would
    /// double-prefix an already-prefixed symbol (e.g. `"C:C:EURUSD"`),
    /// which Polygon's API rejects with a 400 "Ticker was incorrectly
    /// formatted" error -- the same root cause `program`'s
    /// `api::metadata::strip_polygon_prefix` fixes for the date-range
    /// lookup path, duplicated here since this crate doesn't depend on
    /// `program`.
    fn strip_polygon_prefix(symbol: &str) -> &str {
        symbol
            .strip_prefix("C:")
            .or_else(|| symbol.strip_prefix("X:"))
            .or_else(|| symbol.strip_prefix("I:"))
            .or_else(|| symbol.strip_prefix("O:"))
            .unwrap_or(symbol)
    }

    /// Convert a platform symbol to a Polygon ticker.
    ///
    /// Crypto examples:
    /// - `"BTC-USD"` → `"X:BTCUSD"`
    /// - `"ETH-USDT"` → `"X:ETHUSDT"`
    ///
    /// Stock examples:
    /// - `"AAPL"` → `"AAPL"` (no prefix)
    ///
    /// Forex examples:
    /// - `"EUR-USD"` → `"C:EURUSD"`
    fn symbol_to_ticker(symbol: &str, asset_class: Option<&str>) -> String {
        let symbol = Self::strip_polygon_prefix(symbol.trim());
        match asset_class {
            Some("stocks") => symbol.to_uppercase(),
            Some("forex") => {
                let stripped = symbol.replace(['-', '/'], "").to_uppercase();
                format!("C:{}", stripped)
            }
            _ => {
                let stripped = symbol.replace('-', "").to_uppercase();
                format!("X:{}", stripped)
            }
        }
    }

    // ------------------------------------------------------------------
    // Fetch helpers
    // ------------------------------------------------------------------

    /// Fetch all aggregate bars for `ticker` across the given date range,
    /// following `next_url` pagination automatically.
    async fn fetch_bars(
        &self,
        ticker: &str,
        from: NaiveDate,
        to: NaiveDate,
        granularity: CandleGranularity,
    ) -> Result<Vec<AggBar>, MassiveProviderError> {
        let (mult, timespan) = granularity.to_polygon_params();
        let from_str = from.format("%Y-%m-%d").to_string();
        let to_str   = to.format("%Y-%m-%d").to_string();

        // Initial URL
        let initial_url = format!(
            "{}/v2/aggs/ticker/{}/range/{}/{}/{}/{}?adjusted=true&sort=asc&limit=50000",
            self.base_url, ticker, mult, timespan, from_str, to_str
        );

        let mut url = initial_url;
        let mut all_bars: Vec<AggBar> = Vec::new();

        loop {
            let agg = self.fetch_page_with_retry(&url).await?;

            if let Some(bars) = agg.results {
                all_bars.extend(bars);
            }

            match agg.next_url {
                Some(next) if !next.is_empty() => {
                    // next_url already contains the auth params when using query-param auth,
                    // but we set the Authorization header globally, so just follow the URL.
                    url = next;
                }
                _ => break,
            }
        }

        Ok(all_bars)
    }

    /// Fetch a single page, retrying transient failures (request timeouts,
    /// connection errors, and 5xx responses) with exponential backoff.
    ///
    /// A large multi-year, fine-granularity range is split by Polygon into many
    /// paginated pages; a single slow page must not fail the entire job. Rate
    /// limits (429), client errors (4xx), and parse errors are NOT retried here —
    /// they are returned immediately so the caller can surface a precise error.
    async fn fetch_page_with_retry(
        &self,
        url: &str,
    ) -> Result<AggResponse, MassiveProviderError> {
        const MAX_ATTEMPTS: u32 = 5;
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;

            let send_result = self.client.get(url).send().await;

            let resp = match send_result {
                Ok(resp) => resp,
                Err(e) => {
                    // Retry transient network failures (timeouts / connection issues).
                    let transient = e.is_timeout() || e.is_connect() || e.is_request();
                    if transient && attempt < MAX_ATTEMPTS {
                        let backoff = Self::retry_backoff(attempt);
                        log::warn!(
                            "Massive page request failed transiently (attempt {}/{}, retrying in {}s): {}",
                            attempt, MAX_ATTEMPTS, backoff.as_secs(), e
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Err(MassiveProviderError::HttpError(e.to_string()));
                }
            };

            let status = resp.status();
            if status == 429 {
                let retry = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(60);
                log_warn_structured!(DATALOADER_LOGGER, "RATE_LIMITED",
                    "symbol" => url,
                    "retry_after" => retry,
                );
                return Err(MassiveProviderError::RateLimited { retry_after_secs: retry });
            }
            if status.is_server_error() && attempt < MAX_ATTEMPTS {
                let backoff = Self::retry_backoff(attempt);
                log::warn!(
                    "Massive page returned server error {} (attempt {}/{}, retrying in {}s)",
                    status.as_u16(), attempt, MAX_ATTEMPTS, backoff.as_secs()
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                log_error_structured!(DATALOADER_LOGGER, "DATA_FETCH_ERROR",
                    "symbol" => url,
                    "error" => format!("API error (status {}): {}", status.as_u16(), &body),
                );
                return Err(MassiveProviderError::ApiError {
                    status: status.as_u16(),
                    message: body,
                });
            }

            return resp
                .json::<AggResponse>()
                .await
                .map_err(|e| MassiveProviderError::ParseError(e.to_string()));
        }
    }

    /// Exponential backoff with a cap: 1s, 2s, 4s, 8s, capped at 15s.
    fn retry_backoff(attempt: u32) -> Duration {
        let secs = 1u64 << attempt.saturating_sub(1).min(4); // 1,2,4,8,16
        Duration::from_secs(secs.min(15))
    }

    /// Query `/v3/reference/tickers` (Polygon.io's ticker-discovery/search
    /// endpoint) for tickers matching `query`. A single page only -- no
    /// `next_url` following, unlike `fetch_bars` -- since this is a discovery
    /// aid (the caller wants "some real options," not an exhaustive catalog
    /// walk of possibly tens of thousands of tickers).
    async fn list_tickers_impl(
        &self,
        query: &TickerQuery,
    ) -> Result<Vec<TickerInfo>, MassiveProviderError> {
        let limit = query.limit.clamp(1, 1000).to_string();
        let mut params: Vec<(&str, String)> = vec![("limit", limit)];
        if let Some(market) = &query.market {
            params.push(("market", market.clone()));
        }
        if let Some(search) = &query.search {
            params.push(("search", search.clone()));
        }
        if let Some(active) = query.active {
            params.push(("active", active.to_string()));
        }

        // reqwest's .query() builds a correctly percent-encoded query string
        // (via the `url` crate internally) -- no reason to hand-roll that.
        let url = format!("{}/v3/reference/tickers", self.base_url);
        let resp = self.client.get(&url).query(&params).send().await
            .map_err(|e| MassiveProviderError::HttpError(e.to_string()))?;

        let status = resp.status();
        if status == 429 {
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(60);
            return Err(MassiveProviderError::RateLimited { retry_after_secs: retry });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(MassiveProviderError::ApiError { status: status.as_u16(), message: body });
        }

        let parsed: TickerListResponse = resp.json().await
            .map_err(|e| MassiveProviderError::ParseError(e.to_string()))?;

        Ok(parsed.results.unwrap_or_default()
            .into_iter()
            // Skip any row genuinely missing a ticker symbol -- there's
            // nothing usable to return for it.
            .filter_map(|r| r.ticker.map(|ticker| TickerInfo {
                ticker,
                name: r.name,
                market: r.market,
                active: r.active,
                primary_exchange: r.primary_exchange,
                delisted_utc: r.delisted_utc,
            }))
            .collect())
    }

    /// Parse Polygon's `contract_type` string ("call"/"put") into `OptionType`.
    /// Case-insensitive; unrecognised values return `None` so the caller can
    /// skip the row rather than guess.
    fn parse_contract_type(s: &str) -> Option<OptionType> {
        match s.to_lowercase().as_str() {
            "call" => Some(OptionType::Call),
            "put" => Some(OptionType::Put),
            _ => None,
        }
    }

    /// Parse a Polygon OCC-format option ticker (`O:{underlying}{YYMMDD}{C|P}{strike*1000, 8-digit zero-padded}`,
    /// e.g. `O:SPY250620C00450000` -> underlying `SPY`, expiration 2025-06-20,
    /// call, strike 450.0) into its components. Returns `None` for anything
    /// that doesn't match this exact shape rather than guessing.
    fn parse_occ_ticker(ticker: &str) -> Option<(String, f64, NaiveDate, OptionType)> {
        let rest = ticker.strip_prefix("O:")?;
        // Fixed-length suffix: 6-digit date + 1-char type + 8-digit strike.
        const SUFFIX_LEN: usize = 15;
        if rest.len() <= SUFFIX_LEN {
            return None;
        }
        let split_at = rest.len() - SUFFIX_LEN;
        let underlying = rest[..split_at].to_string();
        let suffix = &rest[split_at..];

        let date_str = &suffix[0..6];
        let type_char = suffix.as_bytes().get(6).copied()? as char;
        let strike_str = &suffix[7..15];

        let expiration = NaiveDate::parse_from_str(date_str, "%y%m%d").ok()?;
        let contract_type = match type_char {
            'C' => OptionType::Call,
            'P' => OptionType::Put,
            _ => return None,
        };
        let strike_thousandths: f64 = strike_str.parse().ok()?;
        let strike = strike_thousandths / 1000.0;

        Some((underlying, strike, expiration, contract_type))
    }

    /// Query `/v3/reference/options/contracts` for contracts on `underlying`
    /// listed as of `as_of`. Single page only, same discovery-aid rationale
    /// as `list_tickers_impl` -- a full options chain across every
    /// expiration for a liquid underlying can run to thousands of contracts,
    /// and this is meant to answer "what strikes/expirations exist," not
    /// serve as an exhaustive paginated catalog walk.
    async fn list_option_contracts_impl(
        &self,
        underlying: &str,
        as_of: NaiveDate,
    ) -> Result<Vec<OptionContractRef>, MassiveProviderError> {
        let url = format!("{}/v3/reference/options/contracts", self.base_url);
        let params: Vec<(&str, String)> = vec![
            ("underlying_ticker", underlying.to_uppercase()),
            ("as_of", as_of.format("%Y-%m-%d").to_string()),
            ("limit", "1000".to_string()),
        ];

        let resp = self.client.get(&url).query(&params).send().await
            .map_err(|e| MassiveProviderError::HttpError(e.to_string()))?;

        let status = resp.status();
        if status == 429 {
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(60);
            return Err(MassiveProviderError::RateLimited { retry_after_secs: retry });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(MassiveProviderError::ApiError { status: status.as_u16(), message: body });
        }

        let parsed: OptionContractListResponse = resp.json().await
            .map_err(|e| MassiveProviderError::ParseError(e.to_string()))?;

        Ok(parsed.results.unwrap_or_default()
            .into_iter()
            // Skip any row missing a field this type requires to be useful --
            // there's no sensible default for a missing strike or expiration.
            .filter_map(|r| {
                let ticker = r.ticker?;
                let strike = r.strike_price?;
                let expiration = NaiveDate::parse_from_str(&r.expiration_date?, "%Y-%m-%d").ok()?;
                let contract_type = Self::parse_contract_type(&r.contract_type?)?;
                Some(OptionContractRef {
                    ticker,
                    underlying: r.underlying_ticker.unwrap_or_else(|| underlying.to_uppercase()),
                    strike,
                    expiration,
                    contract_type,
                })
            })
            .collect())
    }

    /// Query `/v3/snapshot/options/{underlying}` for a point-in-time
    /// greeks/IV/open-interest snapshot of every listed contract. Best-effort:
    /// this endpoint commonly requires a higher Polygon account tier than
    /// the base contracts/aggregates endpoints, so a 403 here means "not on
    /// this plan," not a bug -- surfaced as a normal `ApiError` like any
    /// other non-success status, for the caller to treat as advisory data
    /// that just isn't available rather than a hard failure.
    async fn fetch_option_snapshot_impl(
        &self,
        underlying: &str,
    ) -> Result<Vec<OptionChainSnapshot>, MassiveProviderError> {
        let url = format!("{}/v3/snapshot/options/{}", self.base_url, underlying.to_uppercase());

        let resp = self.client.get(&url).send().await
            .map_err(|e| MassiveProviderError::HttpError(e.to_string()))?;

        let status = resp.status();
        if status == 429 {
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(60);
            return Err(MassiveProviderError::RateLimited { retry_after_secs: retry });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(MassiveProviderError::ApiError { status: status.as_u16(), message: body });
        }

        let parsed: OptionSnapshotResponse = resp.json().await
            .map_err(|e| MassiveProviderError::ParseError(e.to_string()))?;

        Ok(parsed.results.unwrap_or_default()
            .into_iter()
            .filter_map(|r| {
                let contract_ticker = r.details?.ticker?;
                let greeks = r.greeks.unwrap_or_default();
                Some(OptionChainSnapshot {
                    contract_ticker,
                    implied_volatility: r.implied_volatility,
                    delta: greeks.delta,
                    gamma: greeks.gamma,
                    theta: greeks.theta,
                    vega: greeks.vega,
                    open_interest: r.open_interest,
                })
            })
            .collect())
    }
}

// ============================================================================
// MarketDataProvider impl
// ============================================================================

#[async_trait]
impl MarketDataProvider for MassiveDataProvider {
    fn name(&self) -> &str {
        "Massive (Polygon.io)"
    }

    async fn fetch(&self, request: &DataRequest) -> Result<Vec<MarketData>, ProviderError> {
        let ticker = Self::symbol_to_ticker(&request.symbol, request.asset_class.as_deref());
        let granularity = request.granularity.unwrap_or(CandleGranularity::Hour1);

        log_info_structured!(DATALOADER_LOGGER, "DATA_FETCH_STARTED",
            "symbol" => &request.symbol,
            "exchange" => "massive",
            "from" => request.from,
            "to" => request.to,
            "granularity" => format!("{:?}", granularity),
        );

        let fetch_start = std::time::Instant::now();

        let bars = self
            .fetch_bars(&ticker, request.from, request.to, granularity)
            .await
            .map_err(ProviderError::from)?;

        if bars.is_empty() {
            log_error_structured!(DATALOADER_LOGGER, "DATA_FETCH_ERROR",
                "symbol" => &request.symbol,
                "error" => "No data returned for date range",
            );
            return Err(ProviderError::NoData {
                exchange: "massive".into(),
                symbol: request.symbol.clone(),
                from: request.from.to_string(),
                to: request.to.to_string(),
            });
        }

        let symbol: Arc<str> = Arc::from(request.symbol.as_str());
        let exchange: Arc<str> = Arc::from("massive");

        let candles: Vec<MarketData> = bars
            .into_iter()
            .map(|b| {
                // Polygon timestamps are milliseconds since Unix epoch
                let ts = Utc.timestamp_millis_opt(b.t).single().unwrap_or_else(Utc::now);
                MarketData::Candle(Candle {
                    timestamp: ts,
                    symbol: Arc::clone(&symbol),
                    exchange: Arc::clone(&exchange),
                    open: b.o,
                    high: b.h,
                    low: b.l,
                    close: b.c,
                    volume: b.v,
                    trade_count: b.n.unwrap_or(0),
                })
            })
            .collect();

        let elapsed_ms = fetch_start.elapsed().as_millis();
        log_info_structured!(DATALOADER_LOGGER, "DATA_FETCH_COMPLETED",
            "symbol" => &request.symbol,
            "rows" => candles.len(),
            "elapsed_ms" => elapsed_ms,
        );
        Ok(candles)
    }

    async fn list_tickers(&self, query: &TickerQuery) -> Result<Vec<TickerInfo>, ProviderError> {
        self.list_tickers_impl(query).await.map_err(ProviderError::from)
    }

    async fn list_option_contracts(
        &self,
        underlying: &str,
        as_of: NaiveDate,
    ) -> Result<Vec<OptionContractRef>, ProviderError> {
        self.list_option_contracts_impl(underlying, as_of).await.map_err(ProviderError::from)
    }

    async fn fetch_option_aggregates(
        &self,
        contract_ticker: &str,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<Vec<OptionCandle>, ProviderError> {
        // An options contract ticker (e.g. `O:SPY250620C00450000`) is a valid
        // Polygon aggs-endpoint ticker just like an equity/crypto one, so
        // this reuses `fetch_bars`/`fetch_page_with_retry` unchanged rather
        // than duplicating the HTTP/pagination/retry path. Daily bars are the
        // sensible default for options backtesting (matches how historical
        // options data is conventionally distributed as EOD snapshots).
        // Parse the OCC-derived fields back out of the ticker itself rather
        // than requiring the caller to pass strike/expiration/underlying
        // separately -- `list_option_contracts` is the source of truth for
        // those fields when available; this keeps `fetch_option_aggregates`
        // usable standalone too. A ticker that isn't valid OCC format is a
        // genuine caller error for an options-specific method, not something
        // to paper over with placeholder values.
        let (underlying, strike, expiration, contract_type) =
            Self::parse_occ_ticker(contract_ticker).ok_or_else(|| {
                ProviderError::Fetch(format!(
                    "not a recognized OCC option ticker: {}",
                    contract_ticker
                ))
            })?;

        let bars = self
            .fetch_bars(contract_ticker, from, to, CandleGranularity::Day1)
            .await
            .map_err(ProviderError::from)?;

        Ok(bars
            .into_iter()
            .map(|b| {
                let ts = Utc.timestamp_millis_opt(b.t).single().unwrap_or_else(Utc::now);
                OptionCandle {
                    contract_ticker: contract_ticker.to_string(),
                    underlying: underlying.clone(),
                    strike,
                    expiration,
                    contract_type,
                    timestamp: ts,
                    open: b.o,
                    high: b.h,
                    low: b.l,
                    close: b.c,
                    volume: b.v,
                }
            })
            .collect())
    }

    async fn fetch_option_snapshot(
        &self,
        underlying: &str,
    ) -> Result<Vec<OptionChainSnapshot>, ProviderError> {
        self.fetch_option_snapshot_impl(underlying).await.map_err(ProviderError::from)
    }
}

#[cfg(test)]
mod symbol_to_ticker_tests {
    use super::MassiveDataProvider;

    #[test]
    fn converts_platform_symbol_to_polygon_ticker() {
        assert_eq!(MassiveDataProvider::symbol_to_ticker("EUR-USD", Some("forex")), "C:EURUSD");
        assert_eq!(MassiveDataProvider::symbol_to_ticker("BTC-USD", Some("crypto")), "X:BTCUSD");
        assert_eq!(MassiveDataProvider::symbol_to_ticker("aapl", Some("stocks")), "AAPL");
    }

    #[test]
    fn does_not_double_prefix_a_leaked_raw_ticker() {
        // The exact bug this fixes: a symbol already in Massive/Polygon's
        // own raw ticker format (e.g. copied verbatim from a ticker-search
        // result's `ticker` field instead of being converted to the
        // platform's "EUR-USD" convention) must not get double-prefixed
        // into "C:C:EURUSD" -- Polygon's API returns a 400 "Ticker was
        // incorrectly formatted" error for that.
        assert_eq!(MassiveDataProvider::symbol_to_ticker("C:EURUSD", Some("forex")), "C:EURUSD");
        assert_eq!(MassiveDataProvider::symbol_to_ticker("X:BTCUSD", Some("crypto")), "X:BTCUSD");
    }
}
