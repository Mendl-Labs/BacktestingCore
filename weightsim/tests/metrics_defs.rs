//! (i) Metric definitions (`answer_key_v1`): ddof 1, ppy = n / years, years = days / 365.25, drawdown without the
//! initial point. Values are hand-computed; the simulator is used only to produce the return path.

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::*;

/// Single asset, weight 1.0 every bar, zero cost: ret_t = P_t / P_{t-1} - 1 exactly. Prices are chosen so the counted
/// returns are [0.1, -0.1, 0.2, 0.0] (the first bar is the entry and earns nothing).
fn hand_panel() -> Panel {
    // bar0 100 (entry), bar1 110 (+10%), bar2 99 (-10%), bar3 118.8 (+20%), bar4 118.8 (0%)
    let dates: Vec<Date> =
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2021-01-01"].iter().map(|s| d(s)).collect();
    Panel::new(vec!["A".into()], dates, vec![vec![100.0, 110.0, 99.0, 118.8, 118.8]]).unwrap()
}

#[test]
fn simulator_metrics_equal_hand_computed_answer_key_v1_values() {
    let rule = const_rule(&["A"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let sim = simulate(&hand_panel(), &rule, &SimConfig::default()).unwrap();
    assert_eq!(sim.metric_definitions, "answer_key_v1");
    let m = sim.metrics().unwrap();
    // counted returns are dated 2020-01-02, -03, -04 and 2021-01-01: r = [0.1, -0.1, 0.2, 0.0]
    assert_eq!(m.n, 4);
    assert_eq!(m.first_date, d("2020-01-02"));
    assert_eq!(m.last_date, d("2021-01-01"));
    let years = 365.0 / 365.25; // 2020-01-02 -> 2021-01-01 is 365 days
    let ppy = 4.0 / years;
    let sd = (0.05f64 / 3.0).sqrt(); // ddof 1: sum of squared deviations 0.05, n-1 = 3
    assert!((m.years - years).abs() < 1e-14);
    assert!((m.ppy - ppy).abs() < 1e-10);
    assert!((m.std_ddof1 - sd).abs() < 1e-12);
    assert!((m.sharpe - 0.05 / sd * ppy.sqrt()).abs() < 1e-9);
    assert!((m.vol - sd * ppy.sqrt()).abs() < 1e-9);
    assert!((m.final_equity - 1.188).abs() < 1e-12);
    assert!((m.cagr - (1.188f64.powf(1.0 / years) - 1.0)).abs() < 1e-9);
    assert!((m.max_drawdown - (0.99 / 1.1 - 1.0)).abs() < 1e-12);
    // definitions that would betray a wrong convention
    assert!((m.ppy - 365.0).abs() > 1.0, "ppy must be n/years, not 365");
    let pop_sd = (0.05f64 / 4.0).sqrt();
    assert!((m.std_ddof1 - pop_sd).abs() > 1e-3, "ddof must be 1");
}

#[test]
fn window_equity_is_the_cumulative_product_without_an_initial_point() {
    let rule = const_rule(&["A"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let sim = simulate(&hand_panel(), &rule, &SimConfig::default()).unwrap();
    let eq = sim.window_equity();
    let want = [1.1, 0.99, 1.188, 1.188];
    assert_eq!(eq.len(), 4);
    for (a, b) in eq.iter().zip(want) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn a_first_bar_loss_is_not_a_drawdown_in_the_key_definition() {
    let dates: Vec<Date> = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"].iter().map(|s| d(s)).collect();
    let panel = Panel::new(vec!["A".into()], dates, vec![vec![100.0, 95.0, 104.5, 114.95]]).unwrap();
    let rule = const_rule(&["A"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let m = simulate(&panel, &rule, &SimConfig::default()).unwrap().metrics().unwrap();
    assert_eq!(m.max_drawdown, 0.0, "cum = 0.95, 1.045, 1.1495: never below its running max");
}

#[test]
fn metrics_use_the_counted_window_only() {
    // Same run, two windows: metrics must depend only on returns dated inside [start, end].
    let panel = s3_panel();
    let full = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    let narrow = simulate(
        &panel,
        &S3TestRule,
        &SimConfig { start: Some(d("2016-04-01")), end: Some(d("2016-08-31")), ..SimConfig::default() },
    )
    .unwrap();
    let w = narrow.window.unwrap();
    assert_eq!(narrow.dates[w.first_bar], d("2016-04-01"));
    assert_eq!(narrow.dates[w.last_bar], d("2016-08-31"));
    let m = narrow.metrics().unwrap();
    assert_eq!(m.first_date, d("2016-04-01"));
    assert_eq!(m.last_date, d("2016-08-31"));
    assert_ne!(m.n, full.metrics().unwrap().n);
    // returns inside the narrow window are the same numbers as in the wide window
    let (fd, fr) = (full.window_dates(), full.window_returns());
    let i0 = fd.iter().position(|x| *x == d("2016-04-01")).unwrap();
    assert!(max_abs_diff(narrow.window_returns(), &fr[i0..i0 + narrow.window_returns().len()]) < 1e-14);
}

#[test]
fn signal_flip_counter_matches_a_hand_count() {
    // Decision signs per bar: +,+,-,-,0,+  => flips of the single asset at bars 2, 4, 5 = 3 (first excluded).
    let dates = weekdays(d("2020-03-02"), 6);
    let panel = Panel::new(vec!["A".into()], dates, vec![vec![100.0, 101.0, 102.0, 101.0, 103.0, 104.0]]).unwrap();
    let signs = [0.5, 0.5, -0.5, -0.5, 0.0, 0.5];
    let rule = FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, move |h| {
        Ok(vec![signs[h.len() - 1]])
    });
    let sim = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    assert_eq!(sim.signal_flips, vec![3]);
    // with a start date the counter begins at max(start, first decision): decisions from bar 2 on are +... -,-,0,+ = 2 flips
    let sim2 = simulate(&panel, &rule, &SimConfig { start: Some(panel.dates()[3]), ..SimConfig::default() }).unwrap();
    assert_eq!(sim2.signal_flips, vec![2]);
}

#[test]
fn metric_definition_stamp_is_recorded_on_every_result() {
    assert_eq!(METRIC_DEFINITIONS, "answer_key_v1");
    let sim = simulate(&s3_panel(), &S3TestRule, &s3_config()).unwrap();
    assert_eq!(sim.metric_definitions, METRIC_DEFINITIONS);
    assert!(SIMULATOR_VERSION.starts_with("weightsim "));
}
