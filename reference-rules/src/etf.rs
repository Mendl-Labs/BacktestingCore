//! ETF trend sleeve (Faber-style). At a month-end, hold 20% of the sleeve in each of SPY, EFA, IEF, DBC, VNQ
//! whose month-end close is strictly above the simple average of its last 10 month-end closes, INCLUDING the
//! current one; otherwise that 20% stays in cash.

use chrono::NaiveDate;

use crate::checks::{check_as_of, check_month_end, locate};
use crate::decision::{EtfDecision, InstrumentDecision, Signal};
use crate::error::RuleError;
use crate::exact::compare_to_mean;
use crate::months::{check_gaps, month_end_indices};
use crate::options::Options;
use crate::series::Panel;

pub const ETF_SYMBOLS: [&str; 5] = ["SPY", "EFA", "IEF", "DBC", "VNQ"];
/// Number of month-end closes in the average (including the current month-end).
pub const ETF_SMA_MONTH_ENDS: usize = 10;
/// Target weight, as a fraction of the sleeve, of each ETF whose signal is `Long`.
pub const ETF_WEIGHT_PER_INSTRUMENT: f64 = 0.20;

struct Gathered<'a> {
    symbol: &'a str,
    month_end_dates: Vec<NaiveDate>,
    month_end_closes: Vec<f64>,
}

/// Decide the ETF trend sleeve at `decision_date` (a month-end; see `Options::month_end_mode`).
///
/// Only bars dated on or before `decision_date` enter the computation. Each instrument is validated on its own
/// series (no silent joint-dropna as in the reference tool); the ten month-end dates must then agree across
/// the five ETFs, since they trade the same sessions.
///
/// This is the PUBLISHED rule: [`decide_etf_trend_with`] called with `ETF_SYMBOLS` and `ETF_SMA_MONTH_ENDS`.
pub fn decide_etf_trend(
    panel: &Panel,
    decision_date: NaiveDate,
    opts: &Options,
) -> Result<EtfDecision, RuleError> {
    decide_etf_trend_with(panel, decision_date, opts, &ETF_SYMBOLS, ETF_SMA_MONTH_ENDS)
}

/// The same rule over an explicit instrument list and month-end lookback: hold `ETF_WEIGHT_PER_INSTRUMENT` of the
/// sleeve in each of `symbols` whose month-end close is strictly above the simple average of its last
/// `sma_month_ends` month-end closes (the current one included). `instruments` follow `symbols` order.
///
/// `decide_etf_trend` is this function at the published point (`ETF_SYMBOLS`, `ETF_SMA_MONTH_ENDS`); any other
/// point is an implementation point inside a pre-registered envelope, never the certified rule.
///
/// # Panics
/// If `symbols` is empty or `sma_month_ends` is 0: both are usage errors (an envelope is validated before any of
/// its points reaches a rule), not data conditions.
pub fn decide_etf_trend_with(
    panel: &Panel,
    decision_date: NaiveDate,
    opts: &Options,
    symbols: &[&str],
    sma_month_ends: usize,
) -> Result<EtfDecision, RuleError> {
    assert!(
        !symbols.is_empty(),
        "decide_etf_trend_with: empty instrument list"
    );
    assert!(
        sma_month_ends >= 1,
        "decide_etf_trend_with: the lookback needs at least one month-end"
    );
    let mut gathered: Vec<Gathered<'_>> = Vec::with_capacity(symbols.len());
    for &symbol in symbols {
        let s = panel.get(symbol)?;
        check_as_of(s, opts)?;
        let pos = locate(s, decision_date)?;
        check_month_end(s, pos, opts)?;
        let idx = month_end_indices(&s.dates()[..=pos]);
        if idx.len() < sma_month_ends {
            return Err(RuleError::InsufficientHistory {
                symbol: symbol.to_string(),
                needed: sma_month_ends,
                have: idx.len(),
            });
        }
        let idx = &idx[idx.len() - sma_month_ends..];
        check_gaps(symbol, &s.dates()[idx[0]..=pos], opts.gap_policy)?;
        gathered.push(Gathered {
            symbol,
            month_end_dates: idx.iter().map(|&i| s.dates()[i]).collect(),
            month_end_closes: idx.iter().map(|&i| s.closes()[i]).collect(),
        });
    }
    for g in &gathered[1..] {
        for (a, b) in gathered[0].month_end_dates.iter().zip(&g.month_end_dates) {
            if a != b {
                return Err(RuleError::MonthEndMismatch {
                    symbol_a: gathered[0].symbol.to_string(),
                    date_a: *a,
                    symbol_b: g.symbol.to_string(),
                    date_b: *b,
                });
            }
        }
    }
    let mut instruments = Vec::with_capacity(gathered.len());
    for g in &gathered {
        let close = g.month_end_closes[sma_month_ends - 1];
        let cmp = compare_to_mean(close, &g.month_end_closes[..]).ok_or_else(|| {
            RuleError::PriceScaleTooWide {
                symbol: g.symbol.to_string(),
            }
        })?;
        let signal = if cmp.ordering.is_gt() {
            Signal::Long
        } else {
            Signal::Cash
        };
        let weight = if signal == Signal::Long {
            ETF_WEIGHT_PER_INSTRUMENT
        } else {
            0.0
        };
        instruments.push(InstrumentDecision {
            symbol: g.symbol.to_string(),
            close,
            sma: cmp.mean,
            signal,
            weight,
        });
    }
    Ok(EtfDecision {
        decision_date,
        month_end_dates: gathered[0].month_end_dates.clone(),
        instruments,
    })
}
