//! Day-of-week calendar sleeve: single asset, daily. Full weight on every bar whose OWN date falls on a target
//! weekday (default Monday), zero otherwise.
//!
//! # Contrast with `turn_of_month`
//! `turn_of_month` cannot be decided from a single bar in isolation: confirming that a bar is the last trading
//! day of its month requires inspecting the NEXT bar in the series (see `is_confirmed_month_end` there), and the
//! "day after a month end" window requires inspecting bars BEFORE it. `day_of_week` needs none of that. Which
//! weekday a calendar date falls on is a fixed, context-free fact about that one date (`chrono::Datelike::
//! weekday()`), so this primitive is decidable from a single `NaiveDate` in total isolation: no lookback, no
//! lookahead, no neighbouring-bar inspection, no history requirement, no "confirmed by a later bar" subtlety, and
//! therefore no `RuleError` variant is ever reachable from the core decision function -- there is nothing to
//! refuse.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Target weekday, default Monday*: the spec says "Parameter: target weekday (default Monday)". This crate
//!    already depends on `chrono`, and `chrono::Weekday` is the natural, already-validated type for that
//!    parameter (no need to invent a 0-6 integer encoding or a bespoke enum) -- see [`decide_day_of_week`]'s
//!    `target: Weekday` argument and [`DAY_OF_WEEK_DEFAULT_TARGET`] for the default value.
//! 2. *Signature: bare `NaiveDate`, not `TradingDays` + index.* `turn_of_month::decide_turn_of_month` takes a
//!    `TradingDays` series plus a bar index `t` because it genuinely needs to see neighbouring bars. This
//!    primitive never looks at any bar but its own, so threading a whole series and index through it would be
//!    pure ceremony: it can never fail (no `EmptySeries`, `NonMonotonic`, or `DateNotInPanel` is possible for a
//!    function of one date), and every caller would have to construct a `TradingDays` just to hand this function
//!    a date it already had. The bare-`NaiveDate` signature is therefore the honest one: it documents, in the
//!    type signature itself, that no other bar is ever consulted. For a caller who nonetheless wants positional
//!    parity with `decide_turn_of_month(&TradingDays, usize)` (for example a future dispatcher that treats every
//!    calendar primitive uniformly), [`decide_day_of_week_on`] below reuses `turn_of_month::TradingDays` (already
//!    `pub` from this crate's root) rather than inventing a second date-only series type for the same purpose --
//!    it simply looks up the date and delegates to the pure, bare-date function; it is convenience only, never
//!    required for correctness.
//! 3. *No history requirement*: like `turn_of_month` (and unlike the SMA-based rules), there is no
//!    `min_history_bars` concept here at all -- every single date, including the very first bar of any series, is
//!    always fully decidable on its own.
//! 4. *Weekday source of truth*: `NaiveDate::weekday()` is proleptic-Gregorian and matches the ISO weekday used
//!    everywhere else `chrono` appears in this crate; no timezone or exchange-calendar concept is involved, same
//!    "derive from the data's own dates" stance as `turn_of_month` and `months.rs`.

use chrono::{Datelike, NaiveDate, Weekday};

use crate::error::RuleError;
use crate::turn_of_month::TradingDays;

/// Weight assigned to a bar whose date falls on the target weekday; every other bar gets `0.0`.
pub const DAY_OF_WEEK_WEIGHT: f64 = 1.0;

/// Default target weekday when the caller does not pick one explicitly.
pub const DAY_OF_WEEK_DEFAULT_TARGET: Weekday = Weekday::Mon;

/// Decide the day-of-week weight of `date`: [`DAY_OF_WEEK_WEIGHT`] (1.0) if `date` falls on `target`, `0.0`
/// otherwise. A pure function of `date` and `target` alone -- no other bar, no history, no clock.
pub fn decide_day_of_week(date: NaiveDate, target: Weekday) -> f64 {
    if date.weekday() == target {
        DAY_OF_WEEK_WEIGHT
    } else {
        0.0
    }
}

/// Convenience wrapper over [`decide_day_of_week`] using [`DAY_OF_WEEK_DEFAULT_TARGET`] (Monday).
pub fn decide_day_of_week_default(date: NaiveDate) -> f64 {
    decide_day_of_week(date, DAY_OF_WEEK_DEFAULT_TARGET)
}

/// Convenience wrapper over [`decide_day_of_week`] for a caller who wants the same `(&TradingDays, usize-looked-
/// up-by-date)` shape as [`crate::turn_of_month::decide_turn_of_month_on`], for interface parity with other
/// calendar primitives in this crate. `days` is consulted ONLY to validate that `date` is actually one of its
/// bars (via `RuleError::DateNotInPanel`); the returned weight never depends on any bar but `date` itself --
/// reusing `TradingDays` here is purely for a uniform dispatcher signature, not because the computation needs a
/// series (see interpretation choice 2 above).
pub fn decide_day_of_week_on(
    days: &TradingDays,
    date: NaiveDate,
    target: Weekday,
) -> Result<f64, RuleError> {
    days.position_of(date)
        .ok_or_else(|| RuleError::DateNotInPanel {
            symbol: days.symbol().to_string(),
            date,
        })?;
    Ok(decide_day_of_week(date, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn default_target_monday_is_one_on_a_monday() {
        // 2024-01-01 is a Monday.
        assert_eq!(decide_day_of_week_default(d(2024, 1, 1)), 1.0);
        assert_eq!(decide_day_of_week(d(2024, 1, 1), Weekday::Mon), 1.0);
    }

    #[test]
    fn default_target_monday_is_zero_on_tuesday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 2)), 0.0); // Tue
    }

    #[test]
    fn default_target_monday_is_zero_on_wednesday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 3)), 0.0); // Wed
    }

    #[test]
    fn default_target_monday_is_zero_on_thursday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 4)), 0.0); // Thu
    }

    #[test]
    fn default_target_monday_is_zero_on_friday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 5)), 0.0); // Fri
    }

    #[test]
    fn default_target_monday_is_zero_on_saturday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 6)), 0.0); // Sat
    }

    #[test]
    fn default_target_monday_is_zero_on_sunday() {
        assert_eq!(decide_day_of_week_default(d(2024, 1, 7)), 0.0); // Sun
    }

    #[test]
    fn non_default_target_friday_changes_behavior() {
        // Same week as above: with target = Friday, only 2024-01-05 (Fri) should be 1.0, and the previously-1.0
        // Monday must now be 0.0 -- proves the parameter actually changes which day is selected.
        assert_eq!(decide_day_of_week(d(2024, 1, 1), Weekday::Fri), 0.0); // Mon
        assert_eq!(decide_day_of_week(d(2024, 1, 5), Weekday::Fri), 1.0); // Fri
    }

    #[test]
    fn every_weekday_of_a_full_week_gets_exactly_one_match() {
        // Sanity on enum coverage: for every possible target weekday, exactly one of the 7 consecutive calendar
        // dates 2024-01-01..=2024-01-07 (a full Mon..Sun span) is marked 1.0, the rest 0.0.
        let week = [
            d(2024, 1, 1),
            d(2024, 1, 2),
            d(2024, 1, 3),
            d(2024, 1, 4),
            d(2024, 1, 5),
            d(2024, 1, 6),
            d(2024, 1, 7),
        ];
        let targets = [
            Weekday::Mon,
            Weekday::Tue,
            Weekday::Wed,
            Weekday::Thu,
            Weekday::Fri,
            Weekday::Sat,
            Weekday::Sun,
        ];
        for target in targets {
            let matches: usize = week
                .iter()
                .map(|&date| decide_day_of_week(date, target))
                .filter(|&w| w == DAY_OF_WEEK_WEIGHT)
                .count();
            assert_eq!(matches, 1, "target {target:?} should match exactly one day");
        }
    }

    #[test]
    fn decide_day_of_week_on_looks_up_by_date_and_validates_membership() {
        let days = TradingDays::new("TEST", vec![d(2024, 1, 1), d(2024, 1, 2), d(2024, 1, 5)]).unwrap();
        assert_eq!(
            decide_day_of_week_on(&days, d(2024, 1, 1), Weekday::Mon).unwrap(),
            1.0
        );
        assert_eq!(
            decide_day_of_week_on(&days, d(2024, 1, 2), Weekday::Mon).unwrap(),
            0.0
        );
        assert!(matches!(
            decide_day_of_week_on(&days, d(2024, 1, 3), Weekday::Mon),
            Err(RuleError::DateNotInPanel { .. })
        ));
    }
}
