//! Two-provider cross-check for `turn_of_month` (W3.6, Calendar family).
//!
//! The expected values below were produced by an INDEPENDENT Python implementation (Kimi/Moonshot `kimi-k3`, zero
//! shared context with the Rust author) of the same verbatim spec, then hand-verified against the written spec
//! text bar-by-bar (see the comment blocks). The two implementations initially disagreed at the series edges: the
//! first Python draft treated "last date present in a (year, month) group" as sufficient on its own to be "the
//! last trading day of the month," which marks the literal final bar of ANY finite series as a month end (and
//! marks the first 1-3 bars of a series as inside a window even when the data has zero bars for the preceding
//! month). The Rust implementation requires POSITIVE evidence from the data -- a later bar dated in a strictly
//! later calendar month -- before treating a bar as a confirmed month end (mirroring this crate's existing
//! `months.rs` convention, and avoiding the perverse result that an arbitrary, mid-month data pull would always
//! mark its own last day as a false "turn of month"). This is judged the more faithful reading of the spec text
//! ("the last trading day of the calendar month" is not merely "the data stopped here"), so the Python side was
//! corrected to the same confirmed-anchor rule and the two implementations now agree exactly (this file). The
//! series-edge interpretation itself is disclosed as a genuine, only-partially-resolved spec ambiguity in the PR
//! description, not asserted as uniquely correct.
//!
//! Every expected value is an exact 1.0 or 0.0 (not a continuous quantity), so "agreement to 1e-9" here means
//! exact equality -- consistent with the 1e-9 tolerance convention used by this crate's other identity checks,
//! trivially satisfied because no floating-point arithmetic occurs in this rule at all.

use chrono::NaiveDate;
use reference_rules::{decide_turn_of_month, TradingDays};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

/// Fixture A: three confirmed month-end crossings (Jan->Feb, Feb->Mar, Mar->Apr), each with the window's 2nd
/// trading day separated from the 1st by a weekend-style gap (so "trading day" is visibly derived from the data,
/// not assumed from a calendar), plus a final bar deep in April with no further data (exercises the "last bar of
/// the whole series, far past any window" case).
fn fixture_a() -> TradingDays {
    let dates = vec![
        d(2024, 1, 2),
        d(2024, 1, 15),
        d(2024, 1, 16),
        d(2024, 1, 30),
        d(2024, 1, 31), // confirmed month end (Jan)
        d(2024, 2, 1),  // +1
        d(2024, 2, 2),  // +2
        d(2024, 2, 5),  // +3 (gap: Feb 3-4 absent from the data)
        d(2024, 2, 6),  // +4, window closed
        d(2024, 2, 20),
        d(2024, 2, 29), // confirmed month end (Feb, leap year)
        d(2024, 3, 1),  // +1
        d(2024, 3, 4),  // +2 (gap: Mar 2-3 absent)
        d(2024, 3, 5),  // +3
        d(2024, 3, 6),  // +4, window closed
        d(2024, 3, 20),
        d(2024, 3, 29), // confirmed month end (Mar)
        d(2024, 4, 1),  // +1
        d(2024, 4, 2),  // +2
        d(2024, 4, 3),  // +3
        d(2024, 4, 4),  // +4, window closed
        d(2024, 4, 15), // last bar of the series, mid-month, outside every window
    ];
    TradingDays::new("FIXTURE_A", dates).unwrap()
}

/// Independently cross-checked against the Python key (see module doc). Same length as `fixture_a()`.
const FIXTURE_A_EXPECTED: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 1.0, //
    1.0, 1.0, 1.0, 0.0, 0.0, //
    1.0, 1.0, 1.0, 1.0, 0.0, //
    0.0, 1.0, 1.0, 1.0, 1.0, //
    0.0, 0.0,
];

#[test]
fn fixture_a_matches_cross_checked_python_key() {
    let days = fixture_a();
    assert_eq!(days.len(), FIXTURE_A_EXPECTED.len());
    for t in 0..days.len() {
        let got = decide_turn_of_month(&days, t);
        let want = FIXTURE_A_EXPECTED[t];
        assert!(
            (got - want).abs() < 1e-9,
            "bar {t} ({:?}): got {got}, want {want}",
            days.dates()[t]
        );
    }
}

/// Fixture B: the series ends one trading day after a confirmed month end -- the window is still "open" (its 2nd
/// and 3rd days are not in the data yet), but the bar that DOES exist is still correctly inside the window.
#[test]
fn fixture_b_series_ends_mid_window() {
    let days = TradingDays::new("FIXTURE_B", vec![d(2024, 1, 31), d(2024, 2, 1)]).unwrap();
    assert_eq!(decide_turn_of_month(&days, 0), 1.0); // confirmed month end
    assert_eq!(decide_turn_of_month(&days, 1), 1.0); // +1, last bar of the series
}

/// Fixture C: the series starts mid-stream in February with ZERO January bars present at all. Per the
/// confirmed-anchor rule, there is no evidence in the data that any month actually ended just before Feb 2, so
/// these early bars are NOT treated as inside a turn-of-month window merely because of their own calendar
/// position -- both implementations agree they are 0.0 here.
#[test]
fn fixture_c_series_starts_mid_month_with_no_prior_month_data() {
    let days = TradingDays::new(
        "FIXTURE_C",
        vec![d(2024, 2, 2), d(2024, 2, 3), d(2024, 2, 20)],
    )
    .unwrap();
    for t in 0..days.len() {
        assert_eq!(decide_turn_of_month(&days, t), 0.0, "bar {t}");
    }
}
