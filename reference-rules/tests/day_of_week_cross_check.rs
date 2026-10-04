//! Two-provider cross-check for `day_of_week` (W3.6, Calendar family).
//!
//! Unlike `turn_of_month`, this primitive is a pure, context-free function of a single date (Monday=0..Sunday=6,
//! `chrono::Weekday`), so there was no real spec ambiguity to resolve: the independent Python implementation
//! (Kimi/Moonshot `kimi-k3`, zero shared context with the Rust author) used the identical
//! `date.weekday() == target` rule on the first attempt and agreed with the Rust side immediately, no correction
//! round needed. Every expected value is an exact 1.0 or 0.0, so "agreement to 1e-9" is exact equality.

use chrono::{NaiveDate, Weekday};
use reference_rules::decide_day_of_week;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

/// A full Monday..Sunday calendar week, 2024-01-01 (Mon) through 2024-01-07 (Sun).
fn week() -> [NaiveDate; 7] {
    [
        d(2024, 1, 1),
        d(2024, 1, 2),
        d(2024, 1, 3),
        d(2024, 1, 4),
        d(2024, 1, 5),
        d(2024, 1, 6),
        d(2024, 1, 7),
    ]
}

/// Independently cross-checked against the Python key (see module doc): default target (Monday) flags only the
/// 1st day of the week.
#[test]
fn default_monday_target_matches_python_key() {
    let want = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (date, &w) in week().iter().zip(want.iter()) {
        let got = decide_day_of_week(*date, Weekday::Mon);
        assert!((got - w).abs() < 1e-9, "{date:?}: got {got}, want {w}");
    }
}

/// Independently cross-checked against the Python key: a non-default target (Friday) flags only the 5th day.
#[test]
fn non_default_friday_target_matches_python_key() {
    let want = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for (date, &w) in week().iter().zip(want.iter()) {
        let got = decide_day_of_week(*date, Weekday::Fri);
        assert!((got - w).abs() < 1e-9, "{date:?}: got {got}, want {w}");
    }
}

/// Independently cross-checked: every one of the 7 possible targets flags exactly one day of a full week.
#[test]
fn every_target_matches_exactly_one_day_matches_python_key() {
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
        let total: f64 = week()
            .iter()
            .map(|&date| decide_day_of_week(date, target))
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "target {target:?}: total {total}"
        );
    }
}
