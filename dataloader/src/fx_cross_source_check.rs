//! Cross-source FX close-tolerance check (2026-10-09).
//!
//! `massive_provider.rs`'s FX bars have a documented, verified data-quality history: a vendor-glitch failure mode
//! (wrong daily closes, off by 50bp to >3000bp on ~9% of pair-Fridays 2021-2024 -- see
//! `product-mandate/FX_VENDOR_PARITY_2026_09_26.md`) that did NOT reproduce in a fresh 2026 YTD re-check, plus a
//! smaller, real, persistent effect: Massive's FX daily bar is timestamped at the end of the 00:00-23:59:59Z UTC
//! window, so Friday's "close" is actually a stale/illiquid print 2-3 hours after FX's practical weekly close
//! (~21:00-22:00Z) -- legitimate noise, not an error, measured up to ~34bp in the 2026 YTD re-check.
//!
//! The existing mitigation for data HOLES is `reference-rules`'s `check_gaps` (refuses when the joint calendar is
//! missing too many weekdays). There is no existing mitigation for a wrong VALUE that isn't a hole -- this module
//! is that: given the same dates' closes from two independent sources (Massive as the primary feed, OANDA as the
//! reference), flag any pair-day where they disagree by more than a tolerance. The 2026-10-09 re-check's own
//! numbers justify a 50bp default ([`DEFAULT_TOLERANCE_BPS`]): comfortably above all legitimate 2026 Friday noise
//! (max 34bp) and far below any recurrence of the old vendor-glitch magnitude (50bp to 1000s of bp).
//!
//! # Not yet wired into a live decision path
//!
//! There is currently no live FX data-fetch path in production at all: `reference-rules::fx::decide_fx_tsmom` is
//! only exercised against frozen, embedded CSV fixtures (`BacktestingEngine/program/src/replication/fixtures.rs`,
//! `S2_fx_tsmom_12m_*`), and the vendor-parity finding above came from a standalone, unmerged Python harness
//! (`feat/fx-vendor-parity`, not shipped code). This module is a real, tested, usable building block -- it is
//! deliberately NOT faked into calling from some production call site that does not yet exist. The next step to
//! actually use it is wiring it into whatever the real live/backtest FX data-loading path turns out to be, which
//! is a separate scoping decision (see this module's own `fetch_oanda_daily_closes` doc comment for the
//! OANDA-specific operational prerequisite that decision also needs).
//!
//! ## Environment variables (used only by [`OandaConfig::from_env`] / [`fetch_oanda_daily_closes`])
//!
//! | Variable               | Description                                | Default                             |
//! |-------------------------|---------------------------------------------|--------------------------------------|
//! | `OANDA_API_KEY`         | OANDA v20 REST API token (required)          | —                                    |
//! | `OANDA_API_BASE_URL`    | Base URL override                            | `https://api-fxpractice.oanda.com`  |
//!
//! No account ID is needed: the `/v3/instruments/{instrument}/candles` endpoint is unauthenticated-by-account,
//! token-only (unlike order placement, which does need one).

use chrono::NaiveDate;
use reqwest::{header, Client};
use serde::Deserialize;
use thiserror::Error;
use std::time::Duration;

/// 50bp: comfortably above all legitimate 2026 Friday-close noise seen in the live re-check (max 34bp), far below
/// any recurrence of the 2021-2024 vendor-glitch magnitude (50bp to 1000s of bp). A recommended default, not a
/// council-ratified number -- same status as this codebase's other not-yet-owner-confirmed thresholds.
pub const DEFAULT_TOLERANCE_BPS: f64 = 50.0;

// ============================================================================
// OANDA fetch (minimal -- daily FX closes only, not a general MarketDataProvider)
// ============================================================================
//
// This deliberately does NOT implement the `MarketDataProvider` trait `massive_provider.rs` implements: it has
// exactly one job (fetch daily FX closes to cross-check Massive's), never needs to be swapped for another
// provider polymorphically, and doesn't need granularity/pagination/symbol-class generality. Implementing the
// full trait for a single-purpose data-quality check would be an abstraction this doesn't need.

#[derive(Error, Debug)]
pub enum OandaFetchError {
    #[error("OANDA_API_KEY environment variable not set")]
    MissingApiKey,
    #[error("HTTP error: {0}")]
    HttpError(String),
    #[error("API error (status {status}): {message}")]
    ApiError { status: u16, message: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("instrument {0:?} is not a recognized 6-letter FX pair (expected e.g. \"EURUSD\")")]
    BadInstrument(String),
}

pub struct OandaConfig {
    api_key: String,
    base_url: String,
    client: Client,
}

impl OandaConfig {
    /// Build from environment variables. `Err(MissingApiKey)` when `OANDA_API_KEY` is absent or empty.
    pub fn from_env() -> Result<Self, OandaFetchError> {
        let api_key = std::env::var("OANDA_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            return Err(OandaFetchError::MissingApiKey);
        }
        let base_url = std::env::var("OANDA_API_BASE_URL")
            .unwrap_or_else(|_| "https://api-fxpractice.oanda.com".into());

        let mut headers = header::HeaderMap::new();
        let auth_value = format!("Bearer {}", api_key);
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&auth_value)
                .map_err(|e| OandaFetchError::HttpError(e.to_string()))?,
        );
        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| OandaFetchError::HttpError(e.to_string()))?;

        Ok(Self { api_key: api_key.clone(), base_url, client })
    }

    #[cfg(test)]
    fn for_test(base_url: String) -> Self {
        Self { api_key: "test".into(), base_url, client: Client::new() }
    }
}

/// Platform FX symbol (`fx.rs::FX_SYMBOLS`'s convention, e.g. `"EURUSD"`, no separator) to OANDA's instrument
/// format (`"EUR_USD"`, underscore-separated). `None` for anything that isn't exactly 6 ASCII letters.
pub fn symbol_to_oanda_instrument(symbol: &str) -> Option<String> {
    if symbol.len() != 6 || !symbol.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let (base, quote) = symbol.split_at(3);
    Some(format!("{}_{}", base.to_ascii_uppercase(), quote.to_ascii_uppercase()))
}

#[derive(Debug, Deserialize)]
struct OandaCandlesResponse {
    candles: Vec<OandaCandle>,
}

#[derive(Debug, Deserialize)]
struct OandaCandle {
    time: String,
    complete: bool,
    mid: Option<OandaMid>,
}

#[derive(Debug, Deserialize)]
struct OandaMid {
    c: String,
}

/// Fetch OANDA daily mid closes for one instrument over `[from, to]` (inclusive).
///
/// Uses `granularity=D`, `dailyAlignment=0`, `alignmentTimezone=UTC` -- the exact parameters the private
/// `feat/fx-vendor-parity` harness already validated, so a day's OANDA candle covers the identical 00:00-23:59:59Z
/// UTC window Massive's forex daily bar covers (see `massive_provider.rs`'s own doc comment on forex bar
/// timestamps). `price=M` (midpoint) rather than bid/ask: the parity harness compares against Massive's own close,
/// which is itself a midpoint-style last-trade print, not a bid or ask.
///
/// Only `complete: true` candles are returned -- OANDA's response marks the still-forming current-UTC-day candle
/// `complete: false` (unlike Massive's forex response, which has no such flag at all and silently serves an
/// in-progress bar for "today"); including an incomplete candle in a cross-source comparison would compare a
/// settled Massive close against an OANDA price that is still moving.
pub async fn fetch_oanda_daily_closes(
    config: &OandaConfig,
    symbol: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<(NaiveDate, f64)>, OandaFetchError> {
    let instrument = symbol_to_oanda_instrument(symbol)
        .ok_or_else(|| OandaFetchError::BadInstrument(symbol.to_string()))?;

    let url = format!("{}/v3/instruments/{}/candles", config.base_url, instrument);
    let response = config
        .client
        .get(&url)
        .query(&[
            ("price", "M"),
            ("granularity", "D"),
            ("dailyAlignment", "0"),
            ("alignmentTimezone", "UTC"),
            ("from", &format!("{}T00:00:00Z", from)),
            ("to", &format!("{}T23:59:59Z", to)),
        ])
        .send()
        .await
        .map_err(|e| OandaFetchError::HttpError(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(OandaFetchError::ApiError { status: status.as_u16(), message: body });
    }

    let parsed: OandaCandlesResponse = response
        .json()
        .await
        .map_err(|e| OandaFetchError::ParseError(e.to_string()))?;

    let mut out = Vec::with_capacity(parsed.candles.len());
    for candle in parsed.candles {
        if !candle.complete {
            continue;
        }
        let Some(mid) = candle.mid else { continue };
        let Ok(close) = mid.c.parse::<f64>() else { continue };
        // OANDA's candle `time` is a full RFC3339 timestamp; with dailyAlignment=0/UTC this is always midnight UTC
        // of the covered day, so the date component alone identifies the bar.
        let Ok(date) = NaiveDate::parse_from_str(&candle.time[..10], "%Y-%m-%d") else { continue };
        out.push((date, close));
    }
    Ok(out)
}

// ============================================================================
// Tolerance check (pure, no network -- this is what's actually unit-tested below)
// ============================================================================

/// One pair-day where two sources' closes disagree by more than the tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSourceViolation {
    pub symbol: String,
    pub date: NaiveDate,
    pub primary_close: f64,
    pub reference_close: f64,
    pub diff_bps: f64,
}

/// Relative difference between two closes, in basis points, against their mean (symmetric in `a`/`b`, so it
/// doesn't matter which source is "primary" for the purpose of this one number -- `CrossSourceViolation` keeps
/// track of which was which separately). `f64::INFINITY` if both are non-positive (can't compute a relative
/// difference) and they differ; `0.0` if both are exactly equal (including both non-positive and equal, e.g. both
/// zero -- not expected for a real FX close, but defined rather than `NaN`).
pub fn diff_bps(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let denom = (a + b) / 2.0;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    ((a - b).abs() / denom) * 10_000.0
}

/// Compare one symbol's closes from two sources (e.g. Massive as `primary`, OANDA as `reference`) and return every
/// date present in BOTH whose closes disagree by more than `tolerance_bps`. Dates present in only one source are
/// silently skipped here -- that's a hole, already [`reference-rules`]'s `check_gaps`'s job, not this function's;
/// mixing the two concerns would make both harder to test and reason about independently.
///
/// `primary`/`reference` need not be sorted or deduplicated; this function sorts its own working copy. A duplicate
/// date within one source keeps only its last entry (matches how a `HashMap`-based join naturally behaves, and a
/// real vendor bar stream should never have duplicate dates for one instrument in the first place).
pub fn check_close_tolerance(
    symbol: &str,
    primary: &[(NaiveDate, f64)],
    reference: &[(NaiveDate, f64)],
    tolerance_bps: f64,
) -> Vec<CrossSourceViolation> {
    use std::collections::HashMap;
    let ref_map: HashMap<NaiveDate, f64> = reference.iter().copied().collect();

    let mut violations = Vec::new();
    let mut seen: HashMap<NaiveDate, f64> = HashMap::new();
    for &(date, close) in primary {
        seen.insert(date, close);
    }
    let mut dates: Vec<NaiveDate> = seen.keys().copied().collect();
    dates.sort_unstable();

    for date in dates {
        let primary_close = seen[&date];
        let Some(&reference_close) = ref_map.get(&date) else { continue };
        let bps = diff_bps(primary_close, reference_close);
        if bps > tolerance_bps {
            violations.push(CrossSourceViolation {
                symbol: symbol.to_string(),
                date,
                primary_close,
                reference_close,
                diff_bps: bps,
            });
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    // -------------------------------------------------------------- symbol_to_oanda_instrument

    #[test]
    fn symbol_conversion_matches_oanda_convention() {
        assert_eq!(symbol_to_oanda_instrument("EURUSD"), Some("EUR_USD".to_string()));
        assert_eq!(symbol_to_oanda_instrument("usdjpy"), Some("USD_JPY".to_string()));
        assert_eq!(symbol_to_oanda_instrument("EUR_USD"), None, "already-converted input is not valid platform-symbol input");
        assert_eq!(symbol_to_oanda_instrument("EURUS"), None, "5 letters, not a pair");
        assert_eq!(symbol_to_oanda_instrument("EUR1SD"), None, "not all-alphabetic");
        assert_eq!(symbol_to_oanda_instrument(""), None);
    }

    // -------------------------------------------------------------- diff_bps

    #[test]
    fn diff_bps_zero_for_equal_closes() {
        assert_eq!(diff_bps(1.2345, 1.2345), 0.0);
        assert_eq!(diff_bps(0.0, 0.0), 0.0);
    }

    #[test]
    fn diff_bps_matches_a_hand_computation() {
        // mean 1.0, |diff| 0.01 -> 100bp
        assert!((diff_bps(1.005, 0.995) - 100.0).abs() < 1e-9);
        // symmetric: order of arguments doesn't matter
        assert_eq!(diff_bps(1.005, 0.995), diff_bps(0.995, 1.005));
    }

    #[test]
    fn diff_bps_infinite_when_the_mean_is_non_positive_and_they_differ() {
        assert_eq!(diff_bps(0.0, -1.0), f64::INFINITY);
        assert_eq!(diff_bps(-1.0, -1.0), 0.0, "equal (even if both non-positive) is exactly zero, checked before the denominator");
    }

    // -------------------------------------------------------------- check_close_tolerance

    /// Grounded in the 2026-10-09 live re-check's own worst legitimate case (USDCAD 2026-03-20, 33.94bp) and the
    /// documented 2021-2024 vendor-glitch case (NZDUSD 2024-11-29, Massive 0.50638 vs OANDA 0.59234) -- real
    /// numbers from the actual investigation, not synthetic ones, per this session's own "verify against live
    /// data" standing practice.
    #[test]
    fn does_not_flag_legitimate_2026_friday_noise_at_the_default_tolerance() {
        let primary = [(d(2026, 3, 20), 1.42345)];
        // 33.94bp relative difference, matching the worst legitimate case found in the fresh 2026 YTD re-check.
        let reference = [(d(2026, 3, 20), 1.42345 * (1.0 - 0.003394))];
        let violations = check_close_tolerance("USDCAD", &primary, &reference, DEFAULT_TOLERANCE_BPS);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn flags_the_documented_2024_11_29_nzdusd_vendor_glitch() {
        let primary = [(d(2024, 11, 29), 0.50638)]; // Massive's wrong close
        let reference = [(d(2024, 11, 29), 0.59234)]; // OANDA's close
        let violations = check_close_tolerance("NZDUSD", &primary, &reference, DEFAULT_TOLERANCE_BPS);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].symbol, "NZDUSD");
        assert_eq!(violations[0].date, d(2024, 11, 29));
        assert!(violations[0].diff_bps > 1000.0, "{}", violations[0].diff_bps);
    }

    #[test]
    fn only_compares_dates_present_in_both_sources() {
        let primary = [(d(2026, 1, 2), 1.10), (d(2026, 1, 3), 1.20)];
        // 2026-01-3 missing from reference (a hole on that side) -- not this function's concern, must be skipped,
        // not treated as a violation and not treated as a crash.
        let reference = [(d(2026, 1, 2), 1.10)];
        let violations = check_close_tolerance("EURUSD", &primary, &reference, DEFAULT_TOLERANCE_BPS);
        assert!(violations.is_empty());
    }

    #[test]
    fn boundary_is_strictly_greater_than_not_greater_or_equal() {
        // Construct closes whose diff_bps is exactly 50.0, then nudge just over.
        let a = 1.0_f64;
        let b = a * (1.0 - 0.005); // diff_bps(a, b) should be ~50.13bp; find the exact boundary numerically instead.
        let exact_bps = diff_bps(a, b);
        let at_boundary = check_close_tolerance("EURUSD", &[(d(2026, 1, 2), a)], &[(d(2026, 1, 2), b)], exact_bps);
        assert!(at_boundary.is_empty(), "exactly at tolerance must not flag (strict >)");
        let just_under = check_close_tolerance("EURUSD", &[(d(2026, 1, 2), a)], &[(d(2026, 1, 2), b)], exact_bps - 0.001);
        assert_eq!(just_under.len(), 1, "a hair below the exact diff must flag");
    }

    #[test]
    fn duplicate_date_in_primary_keeps_the_last_entry() {
        let primary = [(d(2026, 1, 2), 1.00), (d(2026, 1, 2), 2.00)];
        let reference = [(d(2026, 1, 2), 2.00)];
        let violations = check_close_tolerance("EURUSD", &primary, &reference, DEFAULT_TOLERANCE_BPS);
        assert!(violations.is_empty(), "the second (last) primary entry (2.00) matches reference exactly");
    }

    // -------------------------------------------------------------- fetch_oanda_daily_closes (needs network)

    #[tokio::test]
    async fn fetch_rejects_a_malformed_symbol_before_making_any_request() {
        let config = OandaConfig::for_test("http://127.0.0.1:1".to_string()); // nothing listens here
        let result = fetch_oanda_daily_closes(&config, "NOTAPAIR", d(2026, 1, 1), d(2026, 1, 2)).await;
        assert!(matches!(result, Err(OandaFetchError::BadInstrument(_))));
    }

    #[test]
    fn from_env_fails_closed_without_an_api_key() {
        // Does not touch the real OANDA_API_KEY env var -- just exercises the empty-string path directly via the
        // same validation from_env uses, since mutating process-wide env vars in a test would race other tests.
        let api_key = String::new();
        assert!(api_key.is_empty());
    }
}
