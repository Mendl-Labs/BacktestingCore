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
    OptionType, TickerInfo, TickerQuery, TickerSnapshot,
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

/// The `day`/`prevDay` sub-object shape shared by both
/// `/v2/snapshot/locale/us/markets/stocks/tickers` and
/// `/v2/snapshot/locale/global/markets/crypto/tickers` -- confirmed live
/// 2026-08-27 against both endpoints, same field names in each. `c`/`v`/`vw`
/// are all `0.0` (not omitted) when there's no trade data for that period
/// yet, e.g. `day` for a stock queried outside market hours -- never missing
/// from the JSON, just zeroed, so plain (non-`Option`) `f64` with serde's
/// default is the right shape here.
#[derive(Debug, Deserialize, Default)]
struct MarketSnapshotDay {
    #[serde(default)]
    c: f64,
    #[serde(default)]
    v: f64,
    #[serde(default)]
    vw: f64,
}

/// One ticker row from either market-wide snapshot endpoint.
#[derive(Debug, Deserialize)]
struct MarketSnapshotResult {
    ticker: Option<String>,
    #[serde(default)]
    day: MarketSnapshotDay,
    #[serde(rename = "prevDay", default)]
    prev_day: MarketSnapshotDay,
}

/// Top-level response from either market-wide snapshot endpoint. `tickers`
/// (not `results`) is the real field name Polygon uses here, unlike every
/// other endpoint in this file -- confirmed live, not a guess.
#[derive(Debug, Deserialize)]
struct MarketSnapshotResponse {
    tickers: Option<Vec<MarketSnapshotResult>>,
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
    ///
    /// `query.as_of`, when set, is forwarded as Polygon's `date` param --
    /// confirmed live 2026-08-28 against Polygon/Massive's own docs, this
    /// reconstructs the universe as it existed on that historical date
    /// rather than today's live catalog (see `TickerQuery::as_of`'s doc
    /// comment for the survivorship-bias rationale).
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
        if let Some(as_of) = query.as_of {
            params.push(("date", as_of.format("%Y-%m-%d").to_string()));
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

    /// Query `/v2/snapshot/locale/{us|global}/markets/{stocks|crypto}/tickers`
    /// for a market-wide current volume/price snapshot. Same "advisory,
    /// never a hard failure" contract as `fetch_option_snapshot_impl`: a
    /// non-success status just means no ranking signal is available this
    /// call, not that the caller's whole request should fail.
    async fn snapshot_market_impl(
        &self,
        market: &str,
    ) -> Result<Vec<TickerSnapshot>, MassiveProviderError> {
        let url = match market {
            "stocks" => format!("{}/v2/snapshot/locale/us/markets/stocks/tickers", self.base_url),
            "crypto" => format!("{}/v2/snapshot/locale/global/markets/crypto/tickers", self.base_url),
            // Confirmed live 2026-08-27: real coverage, same day/prevDay
            // shape as stocks/crypto, 1198 tickers in one call. Takes the
            // caller's "forex" (not Polygon's own "fx" market-query
            // shorthand used by list_tickers/TickerQuery -- a different,
            // unrelated vocabulary for a different endpoint) since that's
            // this endpoint's real URL path segment.
            "forex" => format!("{}/v2/snapshot/locale/global/markets/forex/tickers", self.base_url),
            _ => return Ok(Vec::new()), // no snapshot coverage for options/etc.
        };

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

        let parsed: MarketSnapshotResponse = resp.json().await
            .map_err(|e| MassiveProviderError::ParseError(e.to_string()))?;

        Ok(parsed.tickers.unwrap_or_default()
            .into_iter()
            .filter_map(|r| {
                let ticker = r.ticker?;
                Some(TickerSnapshot {
                    ticker,
                    price: r.day.c,
                    day_volume: r.day.v,
                    day_vwap: r.day.vw,
                    prev_day_close: r.prev_day.c,
                    prev_day_volume: r.prev_day.v,
                    prev_day_vwap: r.prev_day.vw,
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
                // Polygon's daily equity bars are timestamped at midnight US
                // Eastern Time (e.g. "2024-01-02" -> 1704171600000, 05:00
                // UTC in winter), while forex/crypto daily bars are
                // timestamped at midnight UTC (same date -> 1704153600000,
                // 00:00 UTC) -- confirmed directly against the API for
                // MSFT/AUD-USD/BTC-USD on the same dates. That ~4-5 hour
                // (DST-dependent) offset means an exact-timestamp intersect
                // join (`intersect_timestamps_ascending`) finds ZERO shared
                // bars between a stocks leg and any forex/crypto leg even on
                // days both markets are open, breaking every cross-asset-
                // class backtest (pairs, basket, cross-sectional) that
                // includes an equity. Re-anchor stocks' daily bars to UTC
                // midnight of the same calendar date so they align with
                // every other asset class's convention. Safe because ET is
                // always behind UTC, so the ET-midnight timestamp always
                // falls within the SAME UTC calendar date, never rolls over.
                let ts = if granularity == CandleGranularity::Day1
                    && request.asset_class.as_deref() == Some("stocks")
                {
                    ts.date_naive().and_hms_opt(0, 0, 0).map(|dt| dt.and_utc()).unwrap_or(ts)
                } else {
                    ts
                };
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

    async fn snapshot_market(&self, market: &str) -> Result<Vec<TickerSnapshot>, ProviderError> {
        self.snapshot_market_impl(market).await.map_err(ProviderError::from)
    }
}

#[cfg(test)]
mod market_snapshot_parsing_tests {
    use super::MarketSnapshotResponse;

    // Real response shapes confirmed live against the actual API 2026-08-27
    // (both endpoints), not guessed from documentation.

    const REAL_STOCKS_SAMPLE: &str = r#"{
        "status": "OK",
        "tickers": [{
            "ticker": "ATI",
            "day": {"o": 0, "h": 0, "l": 0, "c": 0, "v": 0, "vw": 0},
            "prevDay": {"o": 217, "h": 217.84, "l": 211.66, "c": 212.86, "v": 1278324.008908, "vw": 213.9281}
        }]
    }"#;

    const REAL_CRYPTO_SAMPLE: &str = r#"{
        "status": "OK",
        "tickers": [{
            "ticker": "X:SWEATUSD",
            "day": {"o": 0.000344, "h": 0.000345, "l": 0.00034, "c": 0.00034, "v": 2176983.5771599994, "vw": 0.0003413},
            "prevDay": {"o": 0.000343, "h": 0.000348, "l": 0.000341, "c": 0.000345, "v": 5360790.16347, "vw": 0.0003442}
        }]
    }"#;

    #[test]
    fn parses_real_stocks_snapshot_shape_including_zeroed_day_outside_market_hours() {
        let parsed: MarketSnapshotResponse = serde_json::from_str(REAL_STOCKS_SAMPLE).unwrap();
        let tickers = parsed.tickers.unwrap();
        assert_eq!(tickers.len(), 1);
        assert_eq!(tickers[0].ticker.as_deref(), Some("ATI"));
        assert_eq!(tickers[0].day.c, 0.0);
        assert!((tickers[0].prev_day.v - 1278324.008908).abs() < 0.01);
        assert!((tickers[0].prev_day.vw - 213.9281).abs() < 0.0001);
    }

    #[test]
    fn parses_real_crypto_snapshot_shape_with_live_day_data() {
        let parsed: MarketSnapshotResponse = serde_json::from_str(REAL_CRYPTO_SAMPLE).unwrap();
        let tickers = parsed.tickers.unwrap();
        assert_eq!(tickers.len(), 1);
        assert_eq!(tickers[0].ticker.as_deref(), Some("X:SWEATUSD"));
        assert!((tickers[0].day.v - 2176983.5771599994).abs() < 0.01);
        assert!((tickers[0].day.vw - 0.0003413).abs() < 0.0000001);
    }

    const REAL_FOREX_SAMPLE: &str = r#"{
        "status": "OK",
        "tickers": [{
            "ticker": "C:GBPKWD",
            "day": {"o": 0.41856152280534, "h": 0.41856152280534, "l": 0.41797553666585, "c": 0.418162631109453, "v": 29, "vw": 0.4183},
            "prevDay": {"o": 0.419929740882974, "h": 0.419957909505437, "l": 0.418365025001313, "c": 0.418581593374548, "v": 62, "vw": 0.4191}
        }]
    }"#;

    #[test]
    fn parses_real_forex_snapshot_shape() {
        let parsed: MarketSnapshotResponse = serde_json::from_str(REAL_FOREX_SAMPLE).unwrap();
        let tickers = parsed.tickers.unwrap();
        assert_eq!(tickers.len(), 1);
        assert_eq!(tickers[0].ticker.as_deref(), Some("C:GBPKWD"));
        assert!((tickers[0].day.v - 29.0).abs() < 0.01);
        assert!((tickers[0].day.vw - 0.4183).abs() < 0.0001);
    }

    #[test]
    fn tickers_field_not_results_is_the_real_key_name() {
        // This endpoint uses "tickers" as its top-level key, unlike every
        // other endpoint in this file ("results") -- a real, confirmed
        // difference, not an inconsistency to "fix" by renaming.
        let malformed = r#"{"status": "OK", "results": []}"#;
        let parsed: MarketSnapshotResponse = serde_json::from_str(malformed).unwrap();
        assert!(parsed.tickers.is_none(), "a response keyed \"results\" instead of \"tickers\" must parse to None, proving the field name matters");
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
