//! (h) `LastBarOfMonth` vs `Daily` on real month boundaries (weekends, year end, leap February, missing bars).

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::*;

/// A 2019-09 .. 2020-06 US-style trading calendar: weekdays minus a few holidays.
fn trading_calendar() -> Vec<Date> {
    let holidays: Vec<Date> =
        ["2019-09-02", "2019-11-28", "2019-12-25", "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10"]
            .iter()
            .map(|s| d(s))
            .collect();
    let mut out = Vec::new();
    let mut cur = d("2019-09-02");
    let end = d("2020-06-30");
    while cur <= end {
        if cur.weekday() < 5 && !holidays.contains(&cur) {
            out.push(cur);
        }
        cur = cur.add_days(1);
    }
    out
}

fn calendar_panel(dates: Vec<Date>) -> Panel {
    let n = dates.len();
    let closes = vec![(0..n).map(|t| 100.0 + t as f64 * 0.01).collect::<Vec<_>>()];
    Panel::new(vec!["A".into()], dates, closes).unwrap()
}

fn decision_dates(panel: &Panel, schedule: DecisionSchedule) -> Vec<Date> {
    let rule = FnRule::new(&["A"], schedule, RebalancePolicy::OnDecision, 1, |_| Ok(vec![0.5]));
    let sim = simulate(panel, &rule, &SimConfig::default()).unwrap();
    (0..sim.n_bars()).filter(|&t| sim.decision[t]).map(|t| sim.dates[t]).collect()
}

#[test]
fn last_bar_of_month_hits_the_real_month_boundaries() {
    let panel = calendar_panel(trading_calendar());
    let got = decision_dates(&panel, DecisionSchedule::LastBarOfMonth);
    // Hand-listed from the calendar: Nov 30 2019 is a Saturday -> Nov 29; Feb 29 2020 is a Saturday -> Feb 28;
    // May 31 2020 is a Sunday -> May 29 (Fri); Dec 31 2019 is a Tuesday; the final bar (Jun 30) is a month-end.
    let want: Vec<Date> = [
        "2019-09-30",
        "2019-10-31",
        "2019-11-29",
        "2019-12-31",
        "2020-01-31",
        "2020-02-28",
        "2020-03-31",
        "2020-04-30",
        "2020-05-29",
        "2020-06-30",
    ]
    .iter()
    .map(|s| d(s))
    .collect();
    assert_eq!(got, want);
}

#[test]
fn month_end_follows_the_calendar_present_not_the_weekday_rule() {
    // If the vendor is missing 2019-10-31, the last bar of October is the 30th; a rule of "last weekday" would be wrong.
    let dates: Vec<Date> = trading_calendar().into_iter().filter(|x| *x != d("2019-10-31")).collect();
    let got = decision_dates(&calendar_panel(dates), DecisionSchedule::LastBarOfMonth);
    assert!(got.contains(&d("2019-10-30")) && !got.contains(&d("2019-10-31")));
    assert_eq!(got.len(), 10);
}

#[test]
fn final_bar_of_the_panel_counts_as_a_month_end_even_mid_month() {
    let dates: Vec<Date> = trading_calendar().into_iter().filter(|x| *x <= d("2020-03-13")).collect();
    let got = decision_dates(&calendar_panel(dates), DecisionSchedule::LastBarOfMonth);
    assert_eq!(
        *got.last().unwrap(),
        d("2020-03-13"),
        "the key's month_end_dates() takes the last bar present (choice C7)"
    );
    assert_eq!(got.len(), 7);
}

#[test]
fn daily_schedule_decides_on_every_bar_once_history_is_long_enough() {
    let panel = calendar_panel(trading_calendar());
    let n = panel.n_bars();
    let rule = FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::OnDecision, 25, |_| Ok(vec![0.5]));
    let sim = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    let decisions = sim.decision.iter().filter(|x| **x).count();
    assert_eq!(decisions, n - 24, "min_history_bars = 25 skips the first 24 bars silently");
    assert!(sim.refusals.is_empty(), "skipping for min history is not a refusal");
    assert!(!sim.decision[23] && sim.decision[24]);
}

#[test]
fn monthly_and_daily_schedules_produce_different_books_for_the_same_rule() {
    // Same rule and policy (EveryBar), different schedule: a rule whose answer changes daily is sampled monthly.
    let dates = trading_calendar();
    let n = dates.len();
    let closes: Vec<f64> = (0..n).map(|t| 100.0 + ((t * 37) % 11) as f64).collect();
    let panel = Panel::new(vec!["A".into()], dates, vec![closes]).unwrap();
    let f = |h: &HistoryView<'_>| {
        let c = h.closes(0);
        Ok(vec![if c.len() > 1 && c[c.len() - 1] > c[c.len() - 2] { 1.0 } else { -1.0 }])
    };
    let daily = simulate(
        &panel,
        &FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, f),
        &SimConfig::default(),
    )
    .unwrap();
    let monthly = simulate(
        &panel,
        &FnRule::new(&["A"], DecisionSchedule::LastBarOfMonth, RebalancePolicy::EveryBar, 1, f),
        &SimConfig::default(),
    )
    .unwrap();
    assert_eq!(daily.decision.iter().filter(|x| **x).count(), n);
    assert_eq!(monthly.decision.iter().filter(|x| **x).count(), 10);
    assert_ne!(daily.series_sha256, monthly.series_sha256);
    // monthly: the standing target only changes on month-end bars
    for t in 1..n {
        if !monthly.decision[t] {
            assert_eq!(monthly.target_weights[t], monthly.target_weights[t - 1], "bar {t}");
        }
    }
}

#[test]
fn year_boundary_and_leap_february_are_handled() {
    let dates: Vec<Date> = ["2019-12-30", "2019-12-31", "2020-01-02", "2020-02-27", "2020-02-28", "2020-03-02"]
        .iter()
        .map(|s| d(s))
        .collect();
    let got = decision_dates(&calendar_panel(dates), DecisionSchedule::LastBarOfMonth);
    assert_eq!(got, vec![d("2019-12-31"), d("2020-01-02"), d("2020-02-28"), d("2020-03-02")]);
    // 2019-12 and 2020-12 are different months even though the month number matches
    let d2: Vec<Date> = ["2019-12-31", "2020-12-01"].iter().map(|s| d(s)).collect();
    assert_eq!(
        decision_dates(&calendar_panel(d2), DecisionSchedule::LastBarOfMonth),
        vec![d("2019-12-31"), d("2020-12-01")]
    );
}

#[test]
fn synthetic_ladder_calendar_month_ends_equal_the_key_signal_dates() {
    // The key (pandas groupby-last) derived its month-ends independently of `Date::same_month`.
    let panel = s1_panel();
    let sim = simulate(&panel, &S1TestRule, &s1_config()).unwrap();
    let (sd, _) = parse_signals(KEY_S1_SIGNALS);
    let got: Vec<Date> = (0..sim.n_bars()).filter(|&t| sim.decision[t]).map(|t| sim.dates[t]).collect();
    assert_eq!(got, sd);
    // The joint calendar dropped SPY's 2018-07-04 gap for every ETF.
    assert!(!panel.dates().contains(&d("2018-07-04")));
}
