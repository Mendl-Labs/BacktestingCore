//! Turn-of-month calendar sleeve: single asset, daily. Full weight on the last trading day of a calendar month
//! and the first three trading days of the next, zero otherwise -- a pure function of which dates are present in
//! the series, never of an exchange calendar (same "derive the calendar from the data" stance as `months.rs`).
//!
//! # Window definition
//! For every bar that is a CONFIRMED last trading day of its calendar month (a later bar exists, dated in a
//! different month -- see [`is_confirmed_month_end`]), the window is that bar plus the next
//! [`TURN_OF_MONTH_FOLLOWING_DAYS`] (3) trading days that follow it in the data, whatever calendar dates those
//! happen to fall on. "Trading day" means nothing more than "a date present in the input series"; no missing-day
//! detection or exchange calendar is consulted anywhere in this file.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Confirmed month end only*: a bar is "the last trading day of the month" only when a LATER bar in a
//!    different calendar month exists in the series. This mirrors `months::month_end_indices`'s general rule but
//!    deliberately does NOT take that function's special case for the final bar of a series (which treats the
//!    last bar as an automatic month end "even if the month is incomplete", design choice C7, meant for live
//!    decision scheduling). Doing that here would wrongly mark an arbitrary, possibly mid-month, final bar as a
//!    month end. Consequence: the LAST bar of a series can get weight 1.0 only through the "day after a month
//!    end" branch (case 2 below), never through the "is itself a month end" branch (case 1) -- there is no later
//!    bar to confirm it.
//! 2. *Positional, not calendar, lookahead*: once a month end at index `me` is confirmed, the window is the 4
//!    bars at indices `me, me+1, me+2, me+3` (clipped to the series). No check is made that `me+2`/`me+3` are
//!    still within the immediate next calendar month rather than spilling into a month after that; for any data
//!    with more than 3 trading days per month (every real case) this never matters, and the spec's own wording
//!    ("3rd trading day of the next calendar month") is itself positional once "next calendar month" is read as
//!    "whatever bars immediately follow the month end".
//! 3. *First bar of the series*: can only ever be judged via case 1 above (is it a confirmed month end, because a
//!    later bar in a new month exists?). It can NEVER be judged "inside the window because it follows a month
//!    end" (case 2), because case 2 needs a bar strictly before it to BE that confirmed month end, and nothing
//!    precedes the first bar. A first bar that LOOKS like it should be the 2nd or 3rd day of a turn-of-month
//!    window (purely by its own calendar date) gets weight 0.0 here, because the series contains no evidence that
//!    a month actually just ended -- the data could just as well start mid-month. This is a deliberate refusal to
//!    guess, not a bug (see `tests::first_bar_cannot_confirm_a_preceding_month_end`).
//! 4. *Last bars of the series*: if the series ends before the window closes (fewer than 3 bars follow a
//!    confirmed month end), the bars that DO exist still get weight 1.0 under case 2; there is nothing to refuse
//!    and no lookahead/causality problem, because the function is handed the WHOLE series, not a causally
//!    truncated view. Whether and how a live/causal caller should treat a window that is still "open" at the
//!    truncation point (today might yet turn out to be inside it once tomorrow's bar arrives) is a decision for
//!    that later integration layer, not for this pure calendar function.
//! 5. *No history requirement*: unlike every other rule in this crate, there is no `min_history_bars` concept and
//!    no refusal for insufficient history -- bar 0 is always decidable (see point 3: it is simply never "day
//!    after a month end", not undecidable).

use chrono::{Datelike, NaiveDate};

use crate::error::RuleError;

/// Number of trading days after a confirmed month-end bar that stay inside the turn-of-month window (so the full
/// window, including the month-end bar itself, spans `TURN_OF_MONTH_FOLLOWING_DAYS + 1` = 4 trading days).
pub const TURN_OF_MONTH_FOLLOWING_DAYS: usize = 3;

/// Weight assigned to every bar inside the turn-of-month window; every other bar gets `0.0`.
pub const TURN_OF_MONTH_WEIGHT: f64 = 1.0;

/// The date-only series this primitive needs. `PriceSeries` (close-only) doesn't fit: turn-of-month depends on
/// nothing but which dates are present, never on price. Invariants mirror `PriceSeries`: non-empty, strictly
/// ascending dates, a valid symbol.
#[derive(Debug, Clone, PartialEq)]
pub struct TradingDays {
    symbol: String,
    dates: Vec<NaiveDate>,
}

impl TradingDays {
    pub fn new(symbol: impl Into<String>, dates: Vec<NaiveDate>) -> Result<Self, RuleError> {
        let symbol = symbol.into();
        if symbol.is_empty() || symbol.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(RuleError::InvalidSymbol { symbol });
        }
        if dates.is_empty() {
            return Err(RuleError::EmptySeries { symbol });
        }
        for i in 1..dates.len() {
            if dates[i] <= dates[i - 1] {
                return Err(RuleError::NonMonotonic {
                    symbol,
                    index: i,
                    previous: dates[i - 1],
                    current: dates[i],
                });
            }
        }
        Ok(Self { symbol, dates })
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    pub fn dates(&self) -> &[NaiveDate] {
        &self.dates
    }
    pub fn len(&self) -> usize {
        self.dates.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }

    /// Index of the bar dated exactly `date`.
    pub fn position_of(&self, date: NaiveDate) -> Option<usize> {
        self.dates.binary_search(&date).ok()
    }
}

fn same_month(a: NaiveDate, b: NaiveDate) -> bool {
    a.year() == b.year() && a.month() == b.month()
}

/// Is `dates[i]` a CONFIRMED last trading day of its calendar month: does a later bar exist, dated in a
/// different month? `false` when `i` is the last index (nothing later to confirm it -- see interpretation
/// choices 1 and 3 in the module docs); never a guess.
fn is_confirmed_month_end(dates: &[NaiveDate], i: usize) -> bool {
    match dates.get(i + 1) {
        Some(&next) => !same_month(dates[i], next),
        None => false,
    }
}

/// Decide the turn-of-month weight of bar `t` of `days`: `TURN_OF_MONTH_WEIGHT` (1.0) inside the window, `0.0`
/// otherwise.
///
/// # Window inspected
/// Looks only at `days.dates()[t.saturating_sub(TURN_OF_MONTH_FOLLOWING_DAYS) ..= min(t + 1, days.len() - 1)]`:
/// up to `TURN_OF_MONTH_FOLLOWING_DAYS` (3) bars before `t` and exactly one bar after `t`. Nothing further away
/// is ever consulted -- in particular there is no history requirement, and no reach into the next month beyond
/// that single following bar.
///
/// # Panics
/// Panics if `t >= days.len()` (same contract as plain slice indexing; `days` is never empty by construction).
pub fn decide_turn_of_month(days: &TradingDays, t: usize) -> f64 {
    let dates = days.dates();
    assert!(
        t < dates.len(),
        "bar index {t} out of range (series has {} bars)",
        dates.len()
    );
    // Case 1: t is itself a confirmed month end.
    if is_confirmed_month_end(dates, t) {
        return TURN_OF_MONTH_WEIGHT;
    }
    // Case 2: t is one of the TURN_OF_MONTH_FOLLOWING_DAYS trading days right after a confirmed month end.
    for k in 1..=TURN_OF_MONTH_FOLLOWING_DAYS {
        if t >= k {
            let me = t - k;
            if is_confirmed_month_end(dates, me) {
                return TURN_OF_MONTH_WEIGHT;
            }
        }
    }
    0.0
}

/// Convenience wrapper over [`decide_turn_of_month`] for a caller who has a date rather than a bar index.
/// `RuleError::DateNotInPanel` when `date` is not one of `days`' bars.
pub fn decide_turn_of_month_on(days: &TradingDays, date: NaiveDate) -> Result<f64, RuleError> {
    let t = days
        .position_of(date)
        .ok_or_else(|| RuleError::DateNotInPanel {
            symbol: days.symbol().to_string(),
            date,
        })?;
    Ok(decide_turn_of_month(days, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn days(dates: Vec<NaiveDate>) -> TradingDays {
        TradingDays::new("TEST", dates).unwrap()
    }

    #[test]
    fn mid_month_day_is_zero() {
        let s = days(vec![
            d(2024, 1, 10),
            d(2024, 1, 15),
            d(2024, 1, 16),
            d(2024, 1, 31),
            d(2024, 2, 1),
        ]);
        assert_eq!(decide_turn_of_month(&s, 1), 0.0); // Jan 15
    }

    #[test]
    fn last_trading_day_of_month_is_one() {
        let s = days(vec![d(2024, 1, 30), d(2024, 1, 31), d(2024, 2, 1)]);
        assert_eq!(decide_turn_of_month(&s, 1), 1.0); // Jan 31, confirmed by Feb 1
    }

    #[test]
    fn first_second_third_trading_days_of_next_month_are_one() {
        let s = days(vec![
            d(2024, 1, 31), // 0: month end
            d(2024, 2, 1),  // 1: +1
            d(2024, 2, 2),  // 2: +2
            d(2024, 2, 5),  // 3: +3
            d(2024, 2, 6),  // 4: +4 -> should be 0 (window closed)
        ]);
        assert_eq!(decide_turn_of_month(&s, 1), 1.0);
        assert_eq!(decide_turn_of_month(&s, 2), 1.0);
        assert_eq!(decide_turn_of_month(&s, 3), 1.0);
    }

    #[test]
    fn fourth_trading_day_of_next_month_is_zero_window_closed() {
        let s = days(vec![
            d(2024, 1, 31),
            d(2024, 2, 1),
            d(2024, 2, 2),
            d(2024, 2, 5),
            d(2024, 2, 6),
        ]);
        assert_eq!(decide_turn_of_month(&s, 4), 0.0);
    }

    #[test]
    fn gap_near_boundary_uses_data_not_calendar() {
        // Jan 31 is month end (confirmed by Feb 1). The "2nd trading day" after it is Feb 5 even though Feb 2-4
        // don't appear in the data at all -- a multi-day calendar gap that must not break the 3-trading-day
        // count, because "trading day" means "present in the data", not "a day on some exchange calendar".
        let s = days(vec![
            d(2024, 1, 31), // 0: month end
            d(2024, 2, 1),  // 1: +1 trading day
            d(2024, 2, 5),  // 2: +2 trading day (gap of 3 calendar days, still the 2nd trading day)
            d(2024, 2, 6),  // 3: +3 trading day
            d(2024, 2, 7),  // 4: +4 -> window closed
            d(2024, 2, 20), // 5: unambiguously mid-month, far later
        ]);
        assert_eq!(decide_turn_of_month(&s, 0), 1.0);
        assert_eq!(decide_turn_of_month(&s, 1), 1.0);
        assert_eq!(decide_turn_of_month(&s, 2), 1.0); // Feb 5, 2nd trading day despite the gap
        assert_eq!(decide_turn_of_month(&s, 3), 1.0); // Feb 6, 3rd trading day
        assert_eq!(decide_turn_of_month(&s, 4), 0.0); // Feb 7, window closed
        assert_eq!(decide_turn_of_month(&s, 5), 0.0);
    }

    #[test]
    fn first_bar_that_is_the_only_bar_of_its_month_is_a_confirmed_month_end() {
        // Bar 0 is the ONLY January bar; Feb 1 confirms it as the (admittedly degenerate) last trading day of
        // January. This is the "decidable from bar 1" case the spec calls out explicitly.
        let s = days(vec![d(2024, 1, 31), d(2024, 2, 1)]);
        assert_eq!(decide_turn_of_month(&s, 0), 1.0);
    }

    #[test]
    fn first_bar_cannot_confirm_a_preceding_month_end() {
        // Bar 0's date LOOKS like it could be the 2nd trading day of a turn-of-month window, but nothing
        // precedes it in the series, so there is no confirmed month end to anchor a window to. Must be 0.0, not
        // a guess.
        let s = days(vec![d(2024, 2, 2), d(2024, 2, 3), d(2024, 2, 20)]);
        assert_eq!(decide_turn_of_month(&s, 0), 0.0);
    }

    #[test]
    fn last_bar_of_series_mid_window_is_still_one() {
        // The series ends one trading day after a confirmed month end; the 3rd day of the window simply isn't in
        // the data yet. The bar that DOES exist is still decided correctly via case 2 (backward-looking only).
        let s = days(vec![d(2024, 1, 31), d(2024, 2, 1)]);
        assert_eq!(decide_turn_of_month(&s, 1), 1.0); // Feb 1, last bar of the series
    }

    #[test]
    fn last_bar_of_series_not_confirmable_as_month_end_is_zero() {
        // The series simply stops on Jan 15 (mid-month); with no later bar to confirm it, it must NOT be treated
        // as an automatic month end (unlike `months::month_end_indices`'s scheduling convention).
        let s = days(vec![d(2024, 1, 10), d(2024, 1, 15)]);
        assert_eq!(decide_turn_of_month(&s, 1), 0.0);
    }

    #[test]
    fn decide_turn_of_month_on_looks_up_by_date() {
        let s = days(vec![d(2024, 1, 31), d(2024, 2, 1)]);
        assert_eq!(decide_turn_of_month_on(&s, d(2024, 1, 31)).unwrap(), 1.0);
        assert!(matches!(
            decide_turn_of_month_on(&s, d(2024, 3, 1)),
            Err(RuleError::DateNotInPanel { .. })
        ));
    }

    #[test]
    fn trading_days_rejects_empty_and_non_monotonic() {
        assert!(matches!(
            TradingDays::new("X", vec![]),
            Err(RuleError::EmptySeries { .. })
        ));
        assert!(matches!(
            TradingDays::new("X", vec![d(2024, 1, 2), d(2024, 1, 1)]),
            Err(RuleError::NonMonotonic { .. })
        ));
    }
}
