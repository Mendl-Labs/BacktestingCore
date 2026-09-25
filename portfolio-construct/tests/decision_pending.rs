//! `Cadence::DecisionPending`, the LIVE monthly cadence (council Ruling 1 of `COUNCIL_ETF_TIMING_AND_CADENCE.md`, work item W8):
//! evaluate on every run, plan iff `D_computable > D_acted` (or nothing acted yet).
//!
//! Everything is synthetic and public-repo safe: a weekday calendar with US-style holidays (the same rules as SignalEngine's
//! `etf_pending_decision.rs`, whose acceptance test 10 this mirrors), no vendor data, no clock. The driver's `ClosedBarsOnly`
//! provider is modelled by [`closed_bars`]: a run on `as_of` sees the bars dated strictly BEFORE `as_of`.

use portfolio_construct::schedule::*;
use std::collections::BTreeMap;

fn d(y: i32, m: u8, day: u8) -> CivilDate {
    CivilDate::new(y, m, day).unwrap()
}

// ------------------------------------------------------------------------------------------------ independent oracles

/// Days since 1970-01-01 (Howard Hinnant), an oracle independent of the crate's `succ`.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(if m <= 2 { y - 1 } else { y });
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> CivilDate {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    CivilDate::new(y as i32, m as u8, day as u8).unwrap()
}

fn ord(date: CivilDate) -> i64 {
    days_from_civil(date.year, u32::from(date.month), u32::from(date.day))
}

fn add_days(date: CivilDate, n: i64) -> CivilDate {
    civil_from_days(ord(date) + n)
}

/// 0 = Monday .. 6 = Sunday.
fn weekday(date: CivilDate) -> i64 {
    (ord(date) + 3).rem_euclid(7)
}

fn nth_weekday(y: i32, m: u8, wd: i64, n: i64) -> CivilDate {
    let first = d(y, m, 1);
    let shift = (wd - weekday(first)).rem_euclid(7);
    add_days(first, shift + 7 * (n - 1))
}

fn last_weekday(y: i32, m: u8, wd: i64) -> CivilDate {
    let mut x = if m == 12 { d(y + 1, 1, 1) } else { d(y, m + 1, 1) };
    x = add_days(x, -1);
    while weekday(x) != wd {
        x = add_days(x, -1);
    }
    x
}

/// Easter Sunday (anonymous Gregorian algorithm).
fn easter(y: i32) -> CivilDate {
    let a = y % 19;
    let b = y / 100;
    let c = y % 100;
    let dd = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - dd - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = (h + l - 7 * m + 114) % 31 + 1;
    d(y, month as u8, day as u8)
}

/// A synthetic exchange calendar: weekdays except US-style holidays (New Year, MLK, Presidents, Good Friday, Memorial, July 4,
/// Labor, Thanksgiving, Christmas); a holiday on a weekend is simply not shifted.
struct Calendar {
    holidays: std::collections::BTreeSet<CivilDate>,
}

impl Calendar {
    fn us(from_year: i32, to_year: i32) -> Calendar {
        let mut holidays = std::collections::BTreeSet::new();
        for y in from_year..=to_year {
            holidays.insert(d(y, 1, 1));
            holidays.insert(nth_weekday(y, 1, 0, 3));
            holidays.insert(nth_weekday(y, 2, 0, 3));
            holidays.insert(add_days(easter(y), -2));
            holidays.insert(last_weekday(y, 5, 0));
            holidays.insert(d(y, 7, 4));
            holidays.insert(nth_weekday(y, 9, 0, 1));
            holidays.insert(nth_weekday(y, 11, 3, 4));
            holidays.insert(d(y, 12, 25));
        }
        Calendar { holidays }
    }
    fn is_session(&self, date: CivilDate) -> bool {
        weekday(date) < 5 && !self.holidays.contains(&date)
    }
    fn sessions(&self, from: CivilDate, to: CivilDate) -> Vec<CivilDate> {
        (ord(from)..=ord(to)).map(civil_from_days).filter(|x| self.is_session(*x)).collect()
    }
    /// The last session of month `(y, m)`, found by walking back from the last calendar day.
    fn last_session(&self, y: i32, m: u8) -> CivilDate {
        let mut x = if m == 12 { d(y + 1, 1, 1) } else { d(y, m + 1, 1) };
        x = add_days(x, -1);
        while !self.is_session(x) {
            x = add_days(x, -1);
        }
        x
    }
    fn first_session(&self, y: i32, m: u8) -> CivilDate {
        let mut x = d(y, m, 1);
        while !self.is_session(x) {
            x = add_days(x, 1);
        }
        x
    }
}

fn prev_month(y: i32, m: u8) -> (i32, u8) {
    if m == 1 {
        (y - 1, 12)
    } else {
        (y, m - 1)
    }
}

// ------------------------------------------------------------------------------------------------ the driver, in miniature

/// One run of the live ETF sleeve on `as_of`: the visible bars are those dated strictly before `as_of - lag_days`; if the sleeve
/// is pending it acts on `D_computable` and the ledger advances. Returns the decision acted on, if any.
fn run(bars: &[CivilDate], as_of: CivilDate, lag_days: i64, acted: &mut Option<CivilDate>) -> Option<CivilDate> {
    let visible = closed_bars(bars, add_days(as_of, -lag_days));
    let window = &visible[visible.len().saturating_sub(60)..];
    let ev = evaluate_decision(window, *acted).unwrap();
    if ev.pending {
        let decision = ev.computable.expect("pending implies computable");
        *acted = Some(advance_acted(*acted, decision));
        Some(decision)
    } else {
        None
    }
}

/// Every calendar day in `[from, to]`: `(run date, decision acted on)` for the runs that acted.
fn drive(
    bars: &[CivilDate],
    from: CivilDate,
    to: CivilDate,
    lag_days: i64,
    mut acted: Option<CivilDate>,
) -> Vec<(CivilDate, CivilDate)> {
    (ord(from)..=ord(to))
        .map(civil_from_days)
        .filter_map(|day| run(bars, day, lag_days, &mut acted).map(|dec| (day, dec)))
        .collect()
}

fn us_bars(from_year: i32, to_year: i32) -> Vec<CivilDate> {
    Calendar::us(from_year - 1, to_year + 1).sessions(d(from_year, 1, 1), d(to_year, 12, 31))
}

// ------------------------------------------------------------------------------------------------ unit cases

#[test]
fn the_oracle_calendar_is_right_on_known_days() {
    assert_eq!(weekday(d(2019, 9, 30)), 0, "Monday");
    assert_eq!(weekday(d(2020, 2, 29)), 5, "Saturday");
    assert_eq!(weekday(d(2021, 5, 31)), 0, "Monday");
    assert_eq!(nth_weekday(2019, 9, 0, 1), d(2019, 9, 2), "Labor Day 2019");
    assert_eq!(last_weekday(2021, 5, 0), d(2021, 5, 31), "Memorial Day 2021");
    assert_eq!(easter(2019), d(2019, 4, 21));
    assert_eq!(easter(2020), d(2020, 4, 12));
    assert_eq!(civil_from_days(ord(d(2024, 2, 29)) + 1), d(2024, 3, 1));
    let cal = Calendar::us(2019, 2021);
    assert!(!cal.is_session(d(2019, 9, 2)) && !cal.is_session(d(2021, 5, 31)) && !cal.is_session(d(2020, 2, 29)));
    assert_eq!((cal.last_session(2019, 8), cal.first_session(2019, 9)), (d(2019, 8, 30), d(2019, 9, 3)));
    assert_eq!((cal.last_session(2021, 5), cal.first_session(2021, 6)), (d(2021, 5, 28), d(2021, 6, 1)));
}

#[test]
fn computable_decision_is_the_last_bar_of_the_newest_month_the_data_proves_over() {
    let bars = [d(2019, 8, 28), d(2019, 8, 29), d(2019, 8, 30), d(2019, 9, 3), d(2019, 9, 4)];
    // the newest bar is in September: August is proven over; its last bar is the decision
    assert_eq!(computable_decision_date(&bars).unwrap(), Some(d(2019, 8, 30)));
    // one bar of September fewer: still proven (the first September bar is enough: NextMonthBar)
    assert_eq!(computable_decision_date(&bars[..4]).unwrap(), Some(d(2019, 8, 30)));
    // without a September bar August is NOT proven over: the decision is July's last bar
    let july = [d(2019, 7, 30), d(2019, 7, 31), d(2019, 8, 28), d(2019, 8, 30)];
    assert_eq!(
        computable_decision_date(&july).unwrap(),
        Some(d(2019, 7, 31)),
        "the month of the newest bar is never complete"
    );
    // only one month of data, or no data: nothing computable
    assert_eq!(computable_decision_date(&bars[..3]).unwrap(), None);
    assert_eq!(computable_decision_date(&bars[..1]).unwrap(), None);
    assert_eq!(computable_decision_date(&[]).unwrap(), None);
    // the same month number in another year is another month
    assert_eq!(computable_decision_date(&[d(2019, 6, 28), d(2020, 6, 1)]).unwrap(), Some(d(2019, 6, 28)));
    // year boundary
    assert_eq!(
        computable_decision_date(&[d(2019, 12, 30), d(2019, 12, 31), d(2020, 1, 2)]).unwrap(),
        Some(d(2019, 12, 31))
    );
}

#[test]
fn bars_must_be_strictly_ascending_and_the_error_names_the_bar() {
    assert_eq!(
        computable_decision_date(&[d(2019, 8, 30), d(2019, 9, 3), d(2019, 9, 3)]),
        Err(BarsError::NotStrictlyAscending { index: 2 }),
        "a duplicate bar"
    );
    assert_eq!(
        computable_decision_date(&[d(2019, 9, 3), d(2019, 8, 30)]),
        Err(BarsError::NotStrictlyAscending { index: 1 }),
        "unsorted"
    );
    let msg = BarsError::NotStrictlyAscending { index: 2 }.to_string();
    assert!(msg.contains("strictly ascending") && msg.contains("bar 2"), "{msg}");
    assert!(evaluate_decision(&[d(2019, 9, 3), d(2019, 8, 30)], None).is_err());
}

#[test]
fn closed_bars_are_those_dated_strictly_before_the_run_date() {
    let bars = [d(2019, 9, 26), d(2019, 9, 27), d(2019, 9, 30), d(2019, 10, 1)];
    assert_eq!(closed_bars(&bars, d(2019, 9, 30)), &bars[..2], "a bar dated the run date is not closed yet");
    assert_eq!(closed_bars(&bars, d(2019, 10, 1)), &bars[..3]);
    assert_eq!(closed_bars(&bars, d(2019, 10, 2)), &bars[..4]);
    assert_eq!(closed_bars(&bars, d(2019, 9, 1)), &bars[..0]);
    assert_eq!(closed_bars(&bars, d(2030, 1, 1)), &bars[..]);
}

#[test]
fn pending_truth_table() {
    let (a, b) = (d(2019, 8, 30), d(2019, 9, 30));
    assert!(decision_pending(Some(b), Some(a)), "a newer decision is computable");
    assert!(!decision_pending(Some(a), Some(a)), "already acted on");
    assert!(!decision_pending(Some(a), Some(b)), "an older decision than the acted one is never pending");
    assert!(decision_pending(Some(a), None), "entry: nothing acted yet");
    assert!(!decision_pending(None, Some(a)), "nothing computable");
    assert!(!decision_pending(None, None), "nothing computable and nothing acted");
}

#[test]
fn d_acted_is_monotone_and_idempotent() {
    let (a, b) = (d(2019, 8, 30), d(2019, 9, 30));
    assert_eq!(advance_acted(None, a), a);
    assert_eq!(advance_acted(Some(a), b), b);
    assert_eq!(advance_acted(Some(b), a), b, "a replayed older decision never moves D_acted back");
    assert_eq!(advance_acted(Some(b), b), b);
    for (last, x) in [(None, a), (Some(a), b), (Some(b), a), (Some(b), b)] {
        let once = advance_acted(last, x);
        assert_eq!(advance_acted(Some(once), x), once, "idempotent");
        assert!(last.is_none_or(|l| once >= l), "never decreases");
        assert!(once >= x);
    }
}

#[test]
fn due_on_answers_every_cadence_and_the_stateless_due_never_says_pending() {
    let monday = d(2019, 9, 30);
    let inputs = DueInputs {
        date: d(2019, 10, 2),
        next_bar: Some(d(2019, 10, 3)),
        computable_decision: Some(monday),
        last_acted: Some(d(2019, 8, 30)),
    };
    assert!(due_on(Cadence::DecisionPending, &inputs));
    assert!(due_on(Cadence::Daily, &inputs));
    assert!(!due_on(Cadence::CalendarMonthEnd, &inputs), "Oct 2 is not the last calendar day");
    assert!(!due_on(Cadence::LastBarOfMonth, &inputs));
    let acted = DueInputs { last_acted: Some(monday), ..inputs };
    assert!(!due_on(Cadence::DecisionPending, &acted));
    let entry = DueInputs { last_acted: None, ..inputs };
    assert!(due_on(Cadence::DecisionPending, &entry));
    let nothing = DueInputs { computable_decision: None, ..inputs };
    assert!(!due_on(Cadence::DecisionPending, &nothing));
    // the stateless entry point has neither a panel nor a ledger: nothing computable, so never pending
    assert!(!due(Cadence::DecisionPending, d(2019, 10, 2), Some(d(2019, 10, 3))));
    assert!(!due(Cadence::DecisionPending, d(2019, 9, 30), None));
    // and the three calendar cadences answer exactly as before through both entry points
    for date in [d(2019, 6, 28), d(2019, 6, 30), d(2019, 7, 1)] {
        for next in [None, Some(d(2019, 7, 1)), Some(d(2019, 7, 2))] {
            for c in [Cadence::Daily, Cadence::CalendarMonthEnd, Cadence::LastBarOfMonth] {
                assert_eq!(due(c, date, next), due_on(c, &DueInputs::stateless(date, next)));
            }
        }
    }
}

#[test]
fn plan_flags_for_supports_only_due_and_all_on_any_due_with_a_pending_etf_sleeve() {
    let cadences = [Cadence::DecisionPending, Cadence::Daily];
    let base = DueInputs {
        date: d(2019, 10, 1),
        next_bar: None,
        computable_decision: Some(d(2019, 8, 30)),
        last_acted: Some(d(2019, 8, 30)),
    };
    let inputs = [base, base];
    // not pending: crypto (daily) alone is due
    assert_eq!(plan_flags_for(BookCadence::PerSleeve, &cadences, &inputs, &[true, true]), vec![false, true]);
    assert_eq!(
        plan_flags_for(BookCadence::AllSleevesOnAnyDue, &cadences, &inputs, &[true, true]),
        vec![true, true],
        "F1 legacy"
    );
    // ETF market closed (weekend): not tradable, so never planned in either mode
    assert_eq!(plan_flags_for(BookCadence::AllSleevesOnAnyDue, &cadences, &inputs, &[false, true]), vec![false, true]);
    // pending: both are planned in both modes
    let pending = DueInputs { computable_decision: Some(d(2019, 9, 30)), ..base };
    let inputs = [pending, pending];
    assert_eq!(plan_flags_for(BookCadence::PerSleeve, &cadences, &inputs, &[true, true]), vec![true, true]);
    assert_eq!(plan_flags_for(BookCadence::AllSleevesOnAnyDue, &cadences, &inputs, &[true, true]), vec![true, true]);
    // pending but the market is closed: a due sleeve that cannot trade is not planned (it stays pending for the next run)
    assert_eq!(plan_flags_for(BookCadence::PerSleeve, &cadences, &inputs, &[false, true]), vec![false, true]);
    // ETF-only account, nothing pending: no run at all in either mode
    assert_eq!(plan_flags_for(BookCadence::PerSleeve, &[Cadence::DecisionPending], &[base], &[true]), vec![false]);
    assert_eq!(
        plan_flags_for(BookCadence::AllSleevesOnAnyDue, &[Cadence::DecisionPending], &[base], &[true]),
        vec![false]
    );
}

// ------------------------------------------------------------------------------------------------ hand-computed cases (Ruling 1 acceptance 1-5)

fn assert_single_action(
    label: &str,
    bars: &[CivilDate],
    from: CivilDate,
    to: CivilDate,
    lag: i64,
    acted: Option<CivilDate>,
    want: (CivilDate, CivilDate),
) {
    let actions = drive(bars, from, to, lag, acted);
    assert_eq!(actions, vec![want], "{label}: exactly one action, (run date, decision)");
}

#[test]
fn monday_month_end_2019_09_30_acts_on_2019_10_02_on_the_september_decision() {
    // Ruling 1 (1): ETF-only, runs every day 2019-09-25 .. 2019-10-08, the August decision already acted on.
    let bars = us_bars(2019, 2019);
    assert_single_action(
        "2019-09-30",
        &bars,
        d(2019, 9, 25),
        d(2019, 10, 8),
        0,
        Some(d(2019, 8, 30)),
        (d(2019, 10, 2), d(2019, 9, 30)),
    );
    // nothing on the calendar month-end itself and nothing on the 1st: the run of 09-30 sees bars through 09-27 only, the run
    // of 10-01 sees 09-30 as its newest bar (September again), so August is still the newest completed month
    let mut acted = Some(d(2019, 8, 30));
    assert_eq!(run(&bars, d(2019, 9, 30), 0, &mut acted), None);
    assert_eq!(run(&bars, d(2019, 10, 1), 0, &mut acted), None);
    assert_eq!(
        computable_decision_date(closed_bars(&bars, d(2019, 9, 30))).unwrap(),
        Some(d(2019, 8, 30)),
        "one month behind: the U3 defect"
    );
    assert_eq!(computable_decision_date(closed_bars(&bars, d(2019, 10, 1))).unwrap(), Some(d(2019, 8, 30)));
    assert_eq!(computable_decision_date(closed_bars(&bars, d(2019, 10, 2))).unwrap(), Some(d(2019, 9, 30)));
    assert_eq!(run(&bars, d(2019, 10, 2), 0, &mut acted), Some(d(2019, 9, 30)));
    assert_eq!(acted, Some(d(2019, 9, 30)));
}

#[test]
fn saturday_month_end_2020_02_29_acts_on_2020_03_03_on_the_february_decision() {
    // Ruling 1 (2): last February session is Friday 02-28; the first March bar is Monday 03-02; nothing on 03-01 or 03-02.
    let bars = us_bars(2020, 2020);
    assert_single_action(
        "2020-02-29",
        &bars,
        d(2020, 2, 24),
        d(2020, 3, 6),
        0,
        Some(d(2020, 1, 31)),
        (d(2020, 3, 3), d(2020, 2, 28)),
    );
    let mut acted = Some(d(2020, 1, 31));
    for day in [d(2020, 2, 28), d(2020, 2, 29), d(2020, 3, 1), d(2020, 3, 2)] {
        assert_eq!(run(&bars, day, 0, &mut acted), None, "{day:?}");
    }
    assert_eq!(run(&bars, d(2020, 3, 3), 0, &mut acted), Some(d(2020, 2, 28)));
}

#[test]
fn memorial_day_2021_05_31_shifts_the_month_end_and_acts_on_2021_06_02() {
    // Ruling 1 (3): a Monday holiday on the last calendar day; last May session Friday 05-28, first June session Tuesday 06-01.
    let bars = us_bars(2021, 2021);
    assert!(!bars.contains(&d(2021, 5, 31)) && bars.contains(&d(2021, 5, 28)) && bars.contains(&d(2021, 6, 1)));
    assert_single_action(
        "2021-05-31",
        &bars,
        d(2021, 5, 25),
        d(2021, 6, 6),
        0,
        Some(d(2021, 4, 30)),
        (d(2021, 6, 2), d(2021, 5, 28)),
    );
}

#[test]
fn labor_day_2019_09_02_shifts_the_start_of_the_month_and_acts_on_2019_09_04() {
    // last August session Friday 08-30 (08-31 Sat), Sep 1 Sun, Sep 2 Labor Day, first September session Tuesday 09-03.
    let bars = us_bars(2019, 2019);
    assert!(!bars.contains(&d(2019, 9, 2)) && bars.contains(&d(2019, 9, 3)));
    assert_single_action(
        "2019-09-02",
        &bars,
        d(2019, 8, 28),
        d(2019, 9, 6),
        0,
        Some(d(2019, 7, 31)),
        (d(2019, 9, 4), d(2019, 8, 30)),
    );
    // and the run of Labor Day itself and of 09-03 (the first September bar is not yet closed) does nothing
    let mut acted = Some(d(2019, 7, 31));
    for day in [d(2019, 8, 30), d(2019, 8, 31), d(2019, 9, 1), d(2019, 9, 2), d(2019, 9, 3)] {
        assert_eq!(run(&bars, day, 0, &mut acted), None, "{day:?}");
    }
    assert_eq!(run(&bars, d(2019, 9, 4), 0, &mut acted), Some(d(2019, 8, 30)));
}

#[test]
fn a_month_end_on_a_thursday_before_a_friday_acts_on_the_saturday() {
    // 2019-10-31 is a Thursday, the first November bar is Friday 11-01: the run of Saturday 11-02 acts (a driver that runs
    // every calendar day, as the council's daily 00:10Z run does).
    let bars = us_bars(2019, 2019);
    assert_single_action(
        "2019-10-31",
        &bars,
        d(2019, 10, 25),
        d(2019, 11, 6),
        0,
        Some(d(2019, 9, 30)),
        (d(2019, 11, 2), d(2019, 10, 31)),
    );
    assert_eq!(weekday(d(2019, 11, 2)), 5);
}

#[test]
fn a_missed_stretch_of_runs_acts_once_on_the_latest_decision_when_the_driver_returns() {
    // Ruling 1 (5): the driver is down 2019-10-02 .. 2019-10-06 and back on 10-07: one action, on the September decision.
    let bars = us_bars(2019, 2019);
    let mut acted = Some(d(2019, 8, 30));
    for day in (ord(d(2019, 9, 25))..=ord(d(2019, 10, 1))).map(civil_from_days) {
        assert_eq!(run(&bars, day, 0, &mut acted), None);
    }
    assert_eq!(run(&bars, d(2019, 10, 7), 0, &mut acted), Some(d(2019, 9, 30)), "acts once, on the September decision");
    for day in (ord(d(2019, 10, 8))..=ord(d(2019, 10, 20))).map(civil_from_days) {
        assert_eq!(run(&bars, day, 0, &mut acted), None, "and never again for it");
    }
    // a longer outage skipping a whole boundary acts ONCE on the latest decision, never on a backlog
    let mut acted = Some(d(2019, 7, 31));
    assert_eq!(run(&bars, d(2019, 10, 15), 0, &mut acted), Some(d(2019, 9, 30)));
    assert_eq!(run(&bars, d(2019, 10, 16), 0, &mut acted), None, "the skipped August decision is not replayed");
}

#[test]
fn vendor_lag_delays_the_action_to_the_first_day_the_new_month_bar_is_visible_and_it_acts_once() {
    // Ruling 1 (4): the vendor shows a bar 3 days late. The first October bar is 10-01, visible to a run whose
    // (as_of - 3 days) is after it: 10-05.
    let bars = us_bars(2019, 2019);
    assert_single_action(
        "lag 3",
        &bars,
        d(2019, 9, 25),
        d(2019, 10, 12),
        3,
        Some(d(2019, 8, 30)),
        (d(2019, 10, 5), d(2019, 9, 30)),
    );
    assert_single_action(
        "lag 0",
        &bars,
        d(2019, 9, 25),
        d(2019, 10, 12),
        0,
        Some(d(2019, 8, 30)),
        (d(2019, 10, 2), d(2019, 9, 30)),
    );
    // at no lag does the decision come out earlier than without lag, and a bigger lag is never earlier than a smaller one
    let mut last = d(1900, 1, 1);
    for lag in 0..8 {
        let actions = drive(&bars, d(2019, 9, 25), d(2019, 10, 20), lag, Some(d(2019, 8, 30)));
        assert_eq!(actions.len(), 1, "lag {lag}");
        assert!(actions[0].0 >= last, "lag {lag}");
        assert_eq!(actions[0].1, d(2019, 9, 30));
        last = actions[0].0;
    }
}

#[test]
fn a_new_account_enters_once_on_the_decision_in_force_and_then_waits_for_the_boundary() {
    // Ruling 9 (a): mid-month entry acts once on the in-force decision.
    let bars = us_bars(2019, 2019);
    let actions = drive(&bars, d(2019, 9, 17), d(2019, 10, 8), 0, None);
    assert_eq!(actions, vec![(d(2019, 9, 17), d(2019, 8, 30)), (d(2019, 10, 2), d(2019, 9, 30))]);
    // an account entering on a day when nothing is computable (the very first month of data) does not act
    let bars = vec![d(2019, 1, 2), d(2019, 1, 3), d(2019, 1, 4)];
    assert!(drive(&bars, d(2019, 1, 3), d(2019, 1, 10), 0, None).is_empty());
}

#[test]
fn the_wall_clock_month_end_would_act_a_month_behind_every_time_which_is_why_it_must_not_be_the_live_cadence() {
    // CalendarMonthEnd fires on the last calendar day; on that run date the data holds no bar of the new month, so the newest
    // decision the panel supports is the PREVIOUS month's (finding U3), on weekday and weekend month-ends alike.
    let bars = us_bars(1995, 2024);
    let mut checked = 0;
    for y in 1996..=2023 {
        for m in 1..=12u8 {
            let last_day = {
                let first_next = if m == 12 { d(y + 1, 1, 1) } else { d(y, m + 1, 1) };
                add_days(first_next, -1)
            };
            assert!(due(Cadence::CalendarMonthEnd, last_day, None), "{last_day:?}");
            let computable = computable_decision_date(closed_bars(&bars, last_day)).unwrap().unwrap();
            let (py, pm) = prev_month(y, m);
            assert_eq!(
                (computable.year, computable.month),
                (py, pm),
                "on {last_day:?} the panel supports the PREVIOUS month"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 28 * 12);
    // ... and a weekend month-end: the cadence still fires (2020-02-29 Saturday) and the decision is still January's
    assert!(due(Cadence::CalendarMonthEnd, d(2020, 2, 29), None));
    let bars = us_bars(2020, 2020);
    assert_eq!(computable_decision_date(closed_bars(&bars, d(2020, 2, 29))).unwrap(), Some(d(2020, 1, 31)));
}

// ------------------------------------------------------------------------------------------------ properties over 1990-2030

fn property_over_years(first_year: i32, last_year: i32) {
    let cal = Calendar::us(1987, 2032);
    let bars = cal.sessions(d(first_year - 2, 1, 1), d(last_year + 1, 1, 31));
    // Start in December of the previous year so the entry action happens before the window under test.
    let (from, to) = (d(first_year - 1, 12, 1), d(last_year, 12, 31));
    let mut acted: Option<CivilDate> = None;
    let mut plans: BTreeMap<(i32, u8), Vec<(CivilDate, CivilDate)>> = BTreeMap::new();
    let mut days = 0usize;
    let mut prev_acted: Option<CivilDate> = None;
    let mut prev_computable: Option<CivilDate> = None;
    for day in (ord(from)..=ord(to)).map(civil_from_days) {
        days += 1;
        let visible = closed_bars(&bars, day);
        let window = &visible[visible.len().saturating_sub(60)..];
        let ev = evaluate_decision(window, acted).unwrap();
        // the trailing window and the whole history give the same answer (the function only needs the newest two months)
        assert_eq!(ev.computable, computable_decision_date(visible).unwrap(), "{day:?}");
        // D_computable never decreases as time passes
        assert!(prev_computable.is_none_or(|p| ev.computable.is_some_and(|c| c >= p)), "{day:?}");
        prev_computable = ev.computable.or(prev_computable);
        if ev.pending {
            let decision = ev.computable.unwrap();
            plans.entry((day.year, day.month)).or_default().push((day, decision));
            acted = Some(advance_acted(acted, decision));
            // idempotent: the same run again (a retry, a duplicate tick) is not pending any more
            assert!(!evaluate_decision(window, acted).unwrap().pending, "{day:?}");
        }
        // D_acted is monotone
        assert!(prev_acted.is_none_or(|p| acted.is_some_and(|a| a >= p)), "{day:?}");
        prev_acted = acted.or(prev_acted);
    }
    assert!(days as i64 >= i64::from(last_year - first_year + 1) * 365, "every calendar day was run ({days})");
    for y in first_year..=last_year {
        for m in 1..=12u8 {
            let (py, pm) = prev_month(y, m);
            // The oracle walks the calendar: act the day after the first session of the month; decide the last session of the
            // previous month.
            let expected = (add_days(cal.first_session(y, m), 1), cal.last_session(py, pm));
            assert_eq!(
                plans.get(&(y, m)),
                Some(&vec![expected]),
                "{y}-{m:02}: exactly one ETF action, on the day after the first session, on the previous month's last session"
            );
        }
    }
    let extra: Vec<_> = plans.keys().filter(|(y, _)| !(first_year..=last_year).contains(y)).collect();
    assert!(
        extra.iter().all(|k| **k == (first_year - 1, 12)),
        "no action outside the window except the entry a month earlier: {extra:?}"
    );
}

// One test per span so the four run in parallel.
#[test]
fn accept_10_property_exactly_one_etf_action_per_calendar_month_1990_to_1999() {
    property_over_years(1990, 1999);
}

#[test]
fn accept_10_property_exactly_one_etf_action_per_calendar_month_2000_to_2009() {
    property_over_years(2000, 2009);
}

#[test]
fn accept_10_property_exactly_one_etf_action_per_calendar_month_2010_to_2019() {
    property_over_years(2010, 2019);
}

#[test]
fn accept_10_property_exactly_one_etf_action_per_calendar_month_2020_to_2030() {
    property_over_years(2020, 2030);
}

#[test]
fn computable_decision_matches_an_independent_oracle_on_every_prefix_of_forty_years() {
    // For every prefix of the bar list: the last bar strictly before the first day of the newest bar's month.
    let bars = us_bars(1990, 2030);
    let mut prev: Option<CivilDate> = None;
    for end in 1..=bars.len() {
        let window = &bars[end.saturating_sub(60)..end];
        let got = computable_decision_date(window).unwrap();
        let newest = bars[end - 1];
        let month_start = d(newest.year, newest.month, 1);
        let want = bars[..end].iter().rev().find(|b| **b < month_start).copied();
        assert_eq!(got, want, "prefix ending {newest:?}");
        assert!(prev.is_none_or(|p| got.is_some_and(|g| g >= p)), "monotone in the data at {newest:?}");
        prev = got.or(prev);
    }
}

#[test]
fn an_etf_plus_crypto_account_plans_the_etf_once_per_boundary_under_only_due_and_every_session_under_the_legacy_mode() {
    // F1 (Ruling 3) on the driver's own decision: 2019-09-25 .. 2019-10-08, ETF pending / crypto daily.
    let cal = Calendar::us(2019, 2019);
    let bars = cal.sessions(d(2019, 1, 1), d(2019, 12, 31));
    let cadences = [Cadence::DecisionPending, Cadence::Daily];
    let (mut only_due, mut legacy) = (Vec::new(), Vec::new());
    let mut acted = Some(d(2019, 8, 30));
    for day in (ord(d(2019, 9, 25))..=ord(d(2019, 10, 8))).map(civil_from_days) {
        let ev = evaluate_decision(closed_bars(&bars, day), acted).unwrap();
        let mine = DueInputs { date: day, next_bar: None, computable_decision: ev.computable, last_acted: acted };
        let inputs = [mine, mine];
        let tradable = [cal.is_session(day), true];
        let a = plan_flags_for(BookCadence::PerSleeve, &cadences, &inputs, &tradable);
        let b = plan_flags_for(BookCadence::AllSleevesOnAnyDue, &cadences, &inputs, &tradable);
        assert!(a[1] && b[1], "crypto is planned every day in both modes");
        if a[0] {
            only_due.push(day);
            acted = Some(advance_acted(acted, ev.computable.unwrap()));
        }
        if b[0] {
            legacy.push(day);
        }
    }
    assert_eq!(only_due, vec![d(2019, 10, 2)], "ETF planned once");
    assert_eq!(legacy.len(), 10, "the legacy mode re-plans the ETF sleeve on all 10 sessions of the range");
}
