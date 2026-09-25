//! `schedule::due`, calendar month-ends and the cadence modes.

use portfolio_construct::schedule::*;

fn d(y: i32, m: u8, day: u8) -> CivilDate {
    CivilDate::new(y, m, day).unwrap()
}

/// Days since 1970-01-01 (Howard Hinnant's algorithm), used as an independent oracle for weekdays and month lengths.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(if m <= 2 { y - 1 } else { y });
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn weekday(date: CivilDate) -> i64 {
    // 1970-01-01 was a Thursday; 0 = Monday.
    (days_from_civil(date.year, u32::from(date.month), u32::from(date.day)) + 3).rem_euclid(7)
}

#[test]
fn civil_dates_reject_impossible_days() {
    assert!(CivilDate::new(2024, 2, 30).is_err());
    assert!(CivilDate::new(2023, 2, 29).is_err());
    assert!(CivilDate::new(2024, 2, 29).is_ok(), "2024 is a leap year");
    assert!(CivilDate::new(2100, 2, 29).is_err(), "2100 is not (century rule)");
    assert!(CivilDate::new(2000, 2, 29).is_ok(), "2000 is (400 rule)");
    assert!(CivilDate::new(2024, 13, 1).is_err());
    assert!(CivilDate::new(2024, 0, 1).is_err());
    assert!(CivilDate::new(2024, 4, 31).is_err());
    assert!(CivilDate::new(2024, 1, 0).is_err());
}

#[test]
fn succ_rolls_days_months_and_years() {
    assert_eq!(d(2024, 1, 31).succ(), d(2024, 2, 1));
    assert_eq!(d(2024, 2, 28).succ(), d(2024, 2, 29));
    assert_eq!(d(2024, 2, 29).succ(), d(2024, 3, 1));
    assert_eq!(d(2023, 2, 28).succ(), d(2023, 3, 1));
    assert_eq!(d(2100, 2, 28).succ(), d(2100, 3, 1));
    assert_eq!(d(2023, 12, 31).succ(), d(2024, 1, 1));
    assert_eq!(d(2023, 4, 30).succ(), d(2023, 5, 1));
    assert_eq!(d(2023, 4, 29).succ(), d(2023, 4, 30));
}

#[test]
fn calendar_month_end_hand_cases() {
    assert!(d(2024, 2, 29).is_calendar_month_end());
    assert!(!d(2024, 2, 28).is_calendar_month_end());
    assert!(d(2023, 2, 28).is_calendar_month_end());
    assert!(d(2100, 2, 28).is_calendar_month_end(), "2100 has no Feb 29");
    assert!(d(2023, 12, 31).is_calendar_month_end());
    assert!(d(2023, 4, 30).is_calendar_month_end());
    assert!(!d(2023, 4, 29).is_calendar_month_end());
    assert!(!d(2023, 1, 30).is_calendar_month_end());
    assert!(d(2023, 1, 31).is_calendar_month_end());
}

#[test]
fn due_daily_is_always_true() {
    for date in [d(2024, 1, 1), d(2024, 2, 29), d(2024, 12, 31)] {
        assert!(due(Cadence::Daily, date, None));
        assert!(due(Cadence::Daily, date, Some(date.succ())));
    }
}

#[test]
fn due_calendar_month_end_ignores_the_data_calendar() {
    assert!(due(Cadence::CalendarMonthEnd, d(2019, 6, 30), None), "Sunday, still the last calendar day");
    assert!(due(Cadence::CalendarMonthEnd, d(2019, 6, 30), Some(d(2019, 7, 1))));
    assert!(!due(Cadence::CalendarMonthEnd, d(2019, 6, 28), Some(d(2019, 7, 1))), "Friday: not the last CALENDAR day");
}

#[test]
fn due_last_bar_of_month_reads_the_next_bar() {
    // Friday 2019-06-28 is the last bar of June (the next bar is Monday 2019-07-01), although June 30 is a Sunday.
    assert!(due(Cadence::LastBarOfMonth, d(2019, 6, 28), Some(d(2019, 7, 1))));
    assert!(!due(Cadence::LastBarOfMonth, d(2019, 6, 27), Some(d(2019, 6, 28))));
    // Same month number in different years is not the same month.
    assert!(due(Cadence::LastBarOfMonth, d(2019, 6, 28), Some(d(2020, 6, 1))));
    // The last bar of the calendar counts as a month-end even when its month is incomplete (weightsim's definition).
    assert!(due(Cadence::LastBarOfMonth, d(2019, 6, 14), None));
    // The first bar of a month is not due.
    assert!(!due(Cadence::LastBarOfMonth, d(2019, 7, 1), Some(d(2019, 7, 2))));
}

#[test]
fn wall_clock_and_data_calendar_month_ends_agree_except_when_the_month_ends_on_a_weekend() {
    // Thirty years of weekday bars. LastBarOfMonth fires once per month on the last weekday; CalendarMonthEnd fires on the
    // last calendar day. They coincide exactly when the last calendar day is a weekday.
    let mut bars: Vec<CivilDate> = Vec::new();
    let mut cur = d(1995, 1, 2);
    while cur < d(2025, 1, 1) {
        if weekday(cur) < 5 {
            bars.push(cur);
        }
        cur = cur.succ();
    }
    let mut data_due = 0;
    let mut wall_due_on_bars = 0;
    let mut disagreements = 0;
    for (i, &bar) in bars.iter().enumerate() {
        let next = bars.get(i + 1).copied();
        let a = due(Cadence::LastBarOfMonth, bar, next);
        let b = due(Cadence::CalendarMonthEnd, bar, next);
        data_due += usize::from(a);
        wall_due_on_bars += usize::from(b);
        if a != b {
            disagreements += 1;
            // Every disagreement is a month whose last calendar day is a weekend: the last bar is a Friday (or earlier).
            assert!(a && !b, "{bar:?}: the data calendar fires, the wall clock does not");
            let last_day = {
                let mut x = bar;
                while x.same_month(x.succ()) {
                    x = x.succ();
                }
                x
            };
            assert!(weekday(last_day) >= 5, "{bar:?}: month ends on {last_day:?}");
        }
    }
    assert_eq!(data_due, 360, "30 years x 12 months, one last bar each");
    // Months whose last calendar day is Sat or Sun are exactly the disagreements; the rest agree.
    assert_eq!(wall_due_on_bars + disagreements, data_due);
    // 103 of the 360 months (1995-2024) end on a Saturday or Sunday (counted independently with Python's `calendar`).
    assert_eq!(disagreements, 103);
}

#[test]
fn plan_flags_per_sleeve_plans_only_the_due_sleeves() {
    // ETF (monthly) and crypto (daily). ETF due only on month-ends.
    assert_eq!(plan_flags(BookCadence::PerSleeve, &[false, true], &[true, true]), vec![false, true]);
    assert_eq!(plan_flags(BookCadence::PerSleeve, &[true, true], &[true, true]), vec![true, true]);
    assert_eq!(plan_flags(BookCadence::PerSleeve, &[false, false], &[true, true]), vec![false, false]);
    // A due sleeve whose market is closed cannot be planned.
    assert_eq!(plan_flags(BookCadence::PerSleeve, &[true, true], &[false, true]), vec![false, true]);
}

#[test]
fn plan_flags_all_on_any_due_replans_every_tradable_sleeve() {
    // Finding F1: when any sleeve is due (crypto daily), every sleeve is re-targeted -- the ETF one too.
    assert_eq!(plan_flags(BookCadence::AllSleevesOnAnyDue, &[false, true], &[true, true]), vec![true, true]);
    // On a weekend the ETF market is closed: it cannot trade even though the run happens.
    assert_eq!(plan_flags(BookCadence::AllSleevesOnAnyDue, &[false, true], &[false, true]), vec![false, true]);
    // Nothing due: no run.
    assert_eq!(plan_flags(BookCadence::AllSleevesOnAnyDue, &[false, false], &[true, true]), vec![false, false]);
    // The two modes coincide when every sleeve is due, and differ when only some are.
    assert_eq!(
        plan_flags(BookCadence::AllSleevesOnAnyDue, &[true, true], &[true, true]),
        plan_flags(BookCadence::PerSleeve, &[true, true], &[true, true])
    );
    assert_ne!(
        plan_flags(BookCadence::AllSleevesOnAnyDue, &[false, true], &[true, true]),
        plan_flags(BookCadence::PerSleeve, &[false, true], &[true, true])
    );
}
