//! Crypto trend sleeve. Each day, hold 50% of the sleeve in each of BTC and ETH whose close is strictly above the
//! simple average of its last 100 daily closes, INCLUDING today's; otherwise that 50% stays in cash. Long only.

use chrono::NaiveDate;

use crate::checks::{check_as_of, check_panel_end, locate};
use crate::decision::{CryptoDecision, InstrumentDecision, Signal};
use crate::error::RuleError;
use crate::exact::compare_to_mean;
use crate::months::check_gaps;
use crate::options::Options;
use crate::series::Panel;

pub const CRYPTO_SYMBOLS: [&str; 2] = ["BTC", "ETH"];
/// Number of daily closes in the average (including the decision day).
pub const CRYPTO_SMA_DAYS: usize = 100;
/// Target weight, as a fraction of the sleeve, of each coin whose signal is `Long`.
pub const CRYPTO_WEIGHT_PER_INSTRUMENT: f64 = 0.50;

/// Decide the crypto trend sleeve at `decision_date` (the last completed UTC daily bar).
///
/// Only bars dated on or before `decision_date` enter the computation. Each coin is validated on its own series:
/// the decision date must be one of its bars, it must have 100 bars up to and including that date, and by the
/// default gap policy no calendar day may be missing inside those 100 bars.
///
/// This is the PUBLISHED rule: [`decide_crypto_trend_with`] called with `CRYPTO_SYMBOLS` and `CRYPTO_SMA_DAYS`.
pub fn decide_crypto_trend(
    panel: &Panel,
    decision_date: NaiveDate,
    opts: &Options,
) -> Result<CryptoDecision, RuleError> {
    decide_crypto_trend_with(panel, decision_date, opts, &CRYPTO_SYMBOLS, CRYPTO_SMA_DAYS)
}

/// The same rule over an explicit coin list and daily lookback: hold `CRYPTO_WEIGHT_PER_INSTRUMENT` of the sleeve
/// in each of `symbols` whose close is strictly above the simple average of its last `sma_days` daily closes (the
/// decision day included). `instruments` follow `symbols` order.
///
/// `decide_crypto_trend` is this function at the published point (`CRYPTO_SYMBOLS`, `CRYPTO_SMA_DAYS`); any other
/// point is an implementation point inside a pre-registered envelope, never the certified rule.
///
/// # Panics
/// If `symbols` is empty or `sma_days` is 0: both are usage errors (an envelope is validated before any of its
/// points reaches a rule), not data conditions.
pub fn decide_crypto_trend_with(
    panel: &Panel,
    decision_date: NaiveDate,
    opts: &Options,
    symbols: &[&str],
    sma_days: usize,
) -> Result<CryptoDecision, RuleError> {
    assert!(
        !symbols.is_empty(),
        "decide_crypto_trend_with: empty instrument list"
    );
    assert!(
        sma_days >= 1,
        "decide_crypto_trend_with: the lookback needs at least one day"
    );
    let mut instruments = Vec::with_capacity(symbols.len());
    let mut window_start = Vec::with_capacity(symbols.len());
    for &symbol in symbols {
        let s = panel.get(symbol)?;
        check_as_of(s, opts)?;
        let pos = locate(s, decision_date)?;
        check_panel_end(s, pos, opts)?;
        if pos + 1 < sma_days {
            return Err(RuleError::InsufficientHistory {
                symbol: symbol.to_string(),
                needed: sma_days,
                have: pos + 1,
            });
        }
        let lo = pos + 1 - sma_days;
        check_gaps(symbol, &s.dates()[lo..=pos], opts.gap_policy)?;
        let window = &s.closes()[lo..=pos];
        let close = s.closes()[pos];
        let cmp = compare_to_mean(close, window).ok_or_else(|| RuleError::PriceScaleTooWide {
            symbol: symbol.to_string(),
        })?;
        let signal = if cmp.ordering.is_gt() {
            Signal::Long
        } else {
            Signal::Cash
        };
        let weight = if signal == Signal::Long {
            CRYPTO_WEIGHT_PER_INSTRUMENT
        } else {
            0.0
        };
        instruments.push(InstrumentDecision {
            symbol: symbol.to_string(),
            close,
            sma: cmp.mean,
            signal,
            weight,
        });
        window_start.push(s.dates()[lo]);
    }
    Ok(CryptoDecision {
        decision_date,
        window_start,
        instruments,
    })
}
