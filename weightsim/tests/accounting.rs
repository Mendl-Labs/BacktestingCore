//! Accounting tests: (c) zero-cost identity against closed-form oracles, (d) cost identity, (f) refusal holds the
//! whole book, (g) drift vs fixed-weight rebalance policies, plus the property tests of design 6.1 (scale linearity,
//! permutation invariance, cash never created, leverage and shorts conserve accounting), financing, delay, limits.

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::harness::check_cost_identity;
use weightsim::*;

const A: [&str; 1] = ["A"];
const AB: [&str; 2] = ["A", "B"];
const ABC: [&str; 3] = ["A", "B", "C"];

fn five_bar_dates() -> Vec<Date> {
    ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"].iter().map(|s| d(s)).collect()
}

fn panel_of(symbols: &[&str], dates: Vec<Date>, cols: Vec<Vec<f64>>) -> Panel {
    Panel::new(symbols.iter().map(|s| s.to_string()).collect(), dates, cols).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ------------------------------------------------------------------------------------------------ (c) zero cost

/// A time-varying, signed, levered weight vector (gross up to 1.3 x ... with shorts), a pure function of the bar index.
fn wfn(t: usize) -> Vec<f64> {
    let s = if (t / 7) % 2 == 0 { 1.0 } else { -1.0 };
    vec![0.5 * s, -0.3, 0.2 + 0.1 * s]
}

fn month_end_indep(dates: &[Date], t: usize) -> bool {
    // Deliberately a different formulation from `Date::same_month`.
    t + 1 == dates.len() || dates[t].month() != dates[t + 1].month() || dates[t].year() != dates[t + 1].year()
}

#[test]
fn zero_cost_every_bar_equals_sum_w_r_closed_form() {
    let panel = synth_panel(&ABC, 300, 11, "2019-01-01");
    let rule = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| Ok(wfn(h.len() - 1)));
    let sim = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    // Oracle: ret_t = sum_i w_i(t-1) * (P_i(t) / P_i(t-1) - 1). Decision at t-1 earns bar t.
    let mut worst = 0.0f64;
    for t in 1..panel.n_bars() {
        let w = wfn(t - 1);
        let mut r = 0.0;
        for i in 0..3 {
            r += w[i] * (panel.closes(i)[t] / panel.closes(i)[t - 1] - 1.0);
        }
        worst = worst.max((sim.ret[t] - r).abs());
    }
    println!("EveryBar zero-cost closed-form identity: max|diff| = {worst:e}");
    assert!(worst <= 1e-12);
    assert_eq!(sim.window.unwrap().first_bar, 1, "first fill at bar 0, first counted return at bar 1");
}

#[test]
fn zero_cost_on_decision_equals_unit_bookkeeping_closed_form() {
    let panel = synth_panel(&ABC, 300, 12, "2019-01-01");
    let rule =
        FnRule::new(&ABC, DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision, 1, |h| Ok(wfn(h.len() - 1)));
    let sim = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    // Oracle: plain unit bookkeeping (independent loop, independent month-end test).
    let n = panel.n_bars();
    let (mut units, mut cash) = (vec![0.0; 3], 1.0);
    let mut eq = Vec::new();
    for t in 0..n {
        let mut v = cash;
        for i in 0..3 {
            v += units[i] * panel.closes(i)[t];
        }
        eq.push(v);
        if month_end_indep(panel.dates(), t) {
            let w = wfn(t);
            let mut inv = 0.0;
            for i in 0..3 {
                units[i] = w[i] * v / panel.closes(i)[t];
                inv += units[i] * panel.closes(i)[t];
            }
            cash = v - inv;
        }
    }
    let first_fill = (0..n).find(|&t| month_end_indep(panel.dates(), t)).unwrap();
    let mut worst = 0.0f64;
    for t in first_fill + 1..n {
        worst = worst.max((sim.ret[t] - (eq[t] / eq[t - 1] - 1.0)).abs());
        worst = worst.max((sim.equity[t] - eq[t]).abs());
    }
    println!("OnDecision zero-cost closed-form identity: max|diff| = {worst:e}");
    assert!(worst <= 1e-12);
    assert_eq!(sim.window.unwrap().first_bar, first_fill + 1);
}

#[test]
fn net_run_with_the_zero_preset_is_bit_identical_to_the_gross_run() {
    let panel = s3_panel();
    let cfg = SimConfig { cost: CostModel::ZERO, ..s3_config() };
    let (gross, net) = simulate_gross_and_net(&panel, &S3TestRule, &cfg).unwrap();
    assert_eq!(gross.series_sha256, net.series_sha256);
    assert!(net.cost.iter().all(|&c| c == 0.0));
    assert_eq!(gross.cost_model_id, "zero");
}

#[test]
fn gross_run_ignores_the_requested_cost_and_financing() {
    let panel = s3_panel();
    let cfg = SimConfig {
        cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE,
        financing: Financing::FlatAnnual { long_bps: 500.0, short_bps: 500.0, cash_bps: 100.0 },
        ..s3_config()
    };
    let (gross, net) = simulate_gross_and_net(&panel, &S3TestRule, &cfg).unwrap();
    assert_eq!(gross.cost_model_id, "zero");
    assert!(gross.cost.iter().all(|&c| c == 0.0));
    assert!(gross.financing.iter().all(|&c| c == 0.0));
    assert!(net.total_cost() > 0.0);
    assert!(net.financing.iter().any(|&c| c != 0.0));
    // and the gross run equals a plain zero-cost run of the same rule
    let plain = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    assert_eq!(gross.series_sha256, plain.series_sha256);
}

// ------------------------------------------------------------------------------------------------- (d) costs

/// Hand computation. Asset A: 100, 100, 110, 121, 121 on the five-bar calendar; month-ends are bar 1 (Jan 31) and
/// bar 4 (Feb 5, last bar). w = 0.5 at each decision, 10 bps per side.
///   bar1: E_pre 1, buy 0.5 -> cost 0.0005; E = 0.9995, cash 0.4995, units 0.005
///   bar2: E = 0.4995 + 0.005*110 = 1.0495;  bar3: E = 0.4995 + 0.005*121 = 1.1045
///   bar4: E_pre 1.1045; target 0.55225; holding 0.605 -> traded 0.05275, cost 0.00005275, E = 1.10444725
#[test]
fn cost_identity_hand_computed_multi_trade() {
    let panel = panel_of(&A, five_bar_dates(), vec![vec![100.0, 100.0, 110.0, 121.0, 121.0]]);
    let rule = const_rule(&A, vec![0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision);
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let (gross, net) = simulate_gross_and_net(&panel, &rule, &cfg).unwrap();
    let want_eq = [1.0, 0.9995, 1.0495, 1.1045, 1.10444725];
    let want_cost = [0.0, 0.0005, 0.0, 0.0, 0.00005275];
    let want_traded = [0.0, 0.5, 0.0, 0.0, 0.05275];
    for t in 0..5 {
        assert!(close(net.equity[t], want_eq[t], 1e-15), "equity[{t}] = {}", net.equity[t]);
        assert!(close(net.cost[t], want_cost[t], 1e-15), "cost[{t}] = {}", net.cost[t]);
        assert!(close(net.traded_notional[t], want_traded[t], 1e-15), "traded[{t}] = {}", net.traded_notional[t]);
    }
    // gross: no cost anywhere
    let want_gross = [1.0, 1.0, 1.05, 1.105, 1.105];
    for t in 0..5 {
        assert!(close(gross.equity[t], want_gross[t], 1e-15));
    }
    // per-bar identity: pre-cost equity minus post-cost equity is exactly this bar's cost
    for t in 1..5 {
        let pre_minus_post = (net.ret_pre_cost[t] - net.ret[t]) * net.equity[t - 1];
        assert!(close(pre_minus_post, net.cost[t], 1e-15), "bar {t}");
        assert_eq!(net.cost[t], net.traded_notional[t] * CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE.rate());
    }
}

#[test]
fn cost_is_charged_on_turnover_not_on_notional_held() {
    // Flat prices, constant weight 1.0, EveryBar: the only trade is the entry. Holding notional 1.0 for 50 bars must
    // cost 10 bps ONCE (turnover), not 50 times (notional).
    let dates = weekdays(d("2020-03-02"), 50);
    let panel = panel_of(&A, dates, vec![vec![100.0; 50]]);
    let rule = const_rule(&A, vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let net = simulate(&panel, &rule, &cfg).unwrap();
    // Entry: buy 1.0 of notional on equity 1.0 -> cost 1e-3. Targets are sized on PRE-cost equity (S-9), so the book is
    // 1e-3 over-sized after the fill; the next bar trades that excess back (turnover 1e-3 -> cost 1e-6), and so on:
    // cost_k = (1e-3)^(k+1), a geometric series. This second-order term is the documented consequence of S-9.
    for k in 0..3 {
        let want = 0.001f64.powi(k as i32 + 1);
        assert!(close(net.cost[k], want, 1e-15), "cost[{k}] = {} want {want}", net.cost[k]);
    }
    let limit = 0.001 / (1.0 - 0.001);
    assert!(close(net.total_cost(), limit, 1e-15), "total cost {} vs geometric limit {limit}", net.total_cost());
    // A charge on NOTIONAL would be 50 x 1e-3 = 0.05; turnover-only cost is ~1e-3 in total.
    assert!(net.total_cost() < 0.0011);
    assert!(close(net.equity[49], 1.0 - limit, 1e-15));
    assert!(close(net.total_traded_notional(), 1.0 + 0.001 / (1.0 - 0.001), 1e-12));
}

/// Independent closed-form recursion for a single asset with constant weight w rebalanced every bar under a
/// proportional cost c: with E the PRE-cost equity, E_t = E_{t-1} (1 + w r_t) - cost_{t-1}, traded_t = w |E_t - E_{t-1}(1 + r_t)|,
/// cost_t = c * traded_t, post-cost equity = E_t - cost_t. (Different algebra from the simulator's unit bookkeeping.)
#[test]
fn every_bar_cost_matches_independent_recursion() {
    let panel = synth_panel(&A, 250, 21, "2019-06-03");
    let w = 0.8;
    let rate = 0.001;
    let rule = const_rule(&A, vec![w], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let sim = simulate(&panel, &rule, &cfg).unwrap();

    let p = panel.closes(0);
    let mut worst = 0.0f64;
    // bar 0: entry from flat, pre-cost equity 1.0
    let mut e_pre_prev = 1.0;
    let mut cost_prev = rate * (w * 1.0);
    worst = worst.max((sim.cost[0] - cost_prev).abs());
    worst = worst.max((sim.equity[0] - (1.0 - cost_prev)).abs());
    for t in 1..p.len() {
        let r = p[t] / p[t - 1] - 1.0;
        let e_pre = e_pre_prev * (1.0 + w * r) - cost_prev;
        let traded = w * (e_pre - e_pre_prev * (1.0 + r)).abs();
        let cost = rate * traded;
        worst = worst.max((sim.cost[t] - cost).abs());
        worst = worst.max((sim.equity[t] - (e_pre - cost)).abs());
        e_pre_prev = e_pre;
        cost_prev = cost;
    }
    println!("EveryBar cost recursion: max|diff| = {worst:e}");
    assert!(worst <= 1e-12, "cost/equity differ from the independent recursion by {worst:e}");
}

#[test]
fn cost_scales_linearly_with_the_rate_on_the_first_trade() {
    let panel = five_bar_panel_single();
    let rule = const_rule(&A, vec![0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision);
    let mk =
        |bps: f64| CostModel { id: "t", commission_bps: bps, half_spread_bps: 0.0, slippage_bps: 0.0, source_note: "" };
    let c10 = simulate(&panel, &rule, &SimConfig { cost: mk(10.0), ..SimConfig::default() }).unwrap().cost[1];
    let c20 = simulate(&panel, &rule, &SimConfig { cost: mk(20.0), ..SimConfig::default() }).unwrap().cost[1];
    assert!(close(c20, 2.0 * c10, 1e-18));
    // components add: commission + half-spread + slippage
    let split = CostModel { id: "t", commission_bps: 5.0, half_spread_bps: 3.0, slippage_bps: 2.0, source_note: "" };
    let cs = simulate(&panel, &rule, &SimConfig { cost: split, ..SimConfig::default() }).unwrap().cost[1];
    assert!(close(cs, c10, 1e-18));
}

fn five_bar_panel_single() -> Panel {
    panel_of(&A, five_bar_dates(), vec![vec![100.0, 100.0, 110.0, 121.0, 121.0]])
}

#[test]
fn net_vs_gross_is_within_ten_percent_of_turnover_times_cost_on_real_rules() {
    for (name, panel, rule_is_s1) in [("S1", s1_panel(), true), ("S3", s3_panel(), false)] {
        let cfg = SimConfig {
            cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE,
            ..if rule_is_s1 { s1_config() } else { s3_config() }
        };
        let (gross, net) = if rule_is_s1 {
            simulate_gross_and_net(&panel, &S1TestRule, &cfg).unwrap()
        } else {
            simulate_gross_and_net(&panel, &S3TestRule, &cfg).unwrap()
        };
        let rep = check_cost_identity(&gross, &net, &CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE);
        println!("{name}: {rep:?}");
        assert_eq!(rep.max_bar_cost_error, 0.0, "cost_t must equal rate x traded_t exactly on every bar");
        assert!((rep.total_cost - rep.rate_times_total_traded).abs() <= 1e-12 * rep.rate_times_total_traded.max(1.0));
        assert!(rep.predicted_drag > 0.0 && rep.actual_drag > 0.0);
        assert!(
            rep.relative_gap <= 0.10,
            "{name}: net/gross drag departs from turnover x cost by {:.3}",
            rep.relative_gap
        );
    }
}

#[test]
fn decisions_do_not_depend_on_cost() {
    // The rule sees only prices, so gross and net runs make the same decisions and hold the same targets.
    let panel = s3_panel();
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..s3_config() };
    let (gross, net) = simulate_gross_and_net(&panel, &S3TestRule, &cfg).unwrap();
    assert_eq!(gross.decision, net.decision);
    assert!(gross.target_weights.iter().zip(&net.target_weights).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(net.equity[net.n_bars() - 1] < gross.equity[gross.n_bars() - 1]);
    assert_ne!(gross.series_sha256, net.series_sha256);
}

// ------------------------------------------------------------------------------------------- (f) refusal semantics

fn refusal_rule(policy: RebalancePolicy) -> FnRule {
    // Weights depend on the month of the decision: Jan -> W1, Feb -> refusal, Mar -> W2. Three assets, whole book.
    FnRule::new(&ABC, DecisionSchedule::LastBarOfMonth, policy, 1, |h| match h.date().month() {
        1 => Ok(vec![0.4, 0.3, -0.2]),
        2 => Err(RuleRefusal::data("stale_bar", "test refusal")),
        _ => Ok(vec![-0.1, 0.5, 0.3]),
    })
}

fn quarter_panel() -> Panel {
    // Jan 2020 .. Mar 2020 weekdays, three assets.
    let dates: Vec<Date> = weekdays(d("2020-01-02"), 63).into_iter().filter(|x| x.month() <= 3).collect();
    let n = dates.len();
    let mut cols = Vec::new();
    for i in 0..3 {
        let mut p = 50.0 * (i + 1) as f64;
        let mut c = Vec::new();
        for t in 0..n {
            p *= 1.0 + 0.004 * (uniform(5, i as u64, t as u64) - 0.5) + 0.0005 * (i as f64 + 1.0);
            c.push(p);
        }
        cols.push(c);
    }
    panel_of(&ABC, dates, cols)
}

#[test]
fn refusal_holds_the_whole_book_under_on_decision() {
    let panel = quarter_panel();
    let cfg = SimConfig {
        on_refusal: OnRefusal::HoldPrevious,
        cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE,
        ..SimConfig::default()
    };
    let sim = simulate(&panel, &refusal_rule(RebalancePolicy::OnDecision), &cfg).unwrap();
    let dates = panel.dates();
    let feb_end = (0..dates.len()).rev().find(|&t| dates[t].month() == 2).unwrap();
    let jan_end = (0..dates.len()).rev().find(|&t| dates[t].month() == 1).unwrap();
    let mar_end = dates.len() - 1;
    assert!(sim.refused[feb_end] && !sim.decision[feb_end]);
    assert_eq!(sim.refusals.len(), 1);
    assert_eq!(sim.refusals[0].date, dates[feb_end]);
    assert_eq!(sim.refusals[0].code, "stale_bar");
    // every leg of the book is untouched from the January fill until the March fill (bit-identical units)
    for t in jan_end + 1..mar_end {
        for i in 0..3 {
            assert_eq!(
                sim.row(&sim.units, t)[i].to_bits(),
                sim.row(&sim.units, jan_end)[i].to_bits(),
                "bar {t} asset {i}"
            );
        }
        assert_eq!(sim.traded_notional[t], 0.0);
    }
    assert_eq!(sim.cost[feb_end], 0.0);
    // the standing target across the refusal is still January's whole vector
    for i in 0..3 {
        assert_eq!(sim.row(&sim.target_weights, feb_end)[i], [0.4, 0.3, -0.2][i]);
    }
    // March then trades the whole book to the new target
    assert!(sim.decision[mar_end] && sim.traded_notional[mar_end] > 0.0);
    assert!((0..3).any(|i| sim.row(&sim.units, mar_end)[i] != sim.row(&sim.units, feb_end)[i]));
}

#[test]
fn refusal_keeps_the_previous_standing_target_under_every_bar() {
    let panel = quarter_panel();
    let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() };
    let sim = simulate(&panel, &refusal_rule(RebalancePolicy::EveryBar), &cfg).unwrap();
    let dates = panel.dates();
    let feb_end = (0..dates.len()).rev().find(|&t| dates[t].month() == 2).unwrap();
    let jan_end = (0..dates.len()).rev().find(|&t| dates[t].month() == 1).unwrap();
    for t in jan_end..dates.len() - 1 {
        for i in 0..3 {
            assert_eq!(sim.row(&sim.target_weights, t)[i], [0.4, 0.3, -0.2][i], "bar {t}");
            // EveryBar: held weights equal the standing target (zero cost), across the refusal
            assert!((sim.row(&sim.held_weights, t)[i] - [0.4, 0.3, -0.2][i]).abs() < 1e-12, "bar {t} asset {i}");
        }
    }
    assert!(sim.refused[feb_end]);
}

#[test]
fn abort_policy_fails_the_run_and_names_date_and_code() {
    let panel = quarter_panel();
    let err = simulate(&panel, &refusal_rule(RebalancePolicy::OnDecision), &SimConfig::default()).unwrap_err();
    match err {
        SimError::RuleRefused { date, refusal } => {
            assert_eq!(date.month(), 2);
            assert_eq!(refusal.code, "stale_bar");
            assert_eq!(refusal.kind, RefusalKind::Data);
        }
        other => panic!("expected RuleRefused, got {other:?}"),
    }
}

#[test]
fn warmup_refusal_is_tolerated_only_before_the_first_decision() {
    let panel = quarter_panel();
    // refuses (warmup) in January, decides in February, warmup-refuses again in March
    let rule = FnRule::new(&ABC, DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision, 1, |h| {
        match h.date().month() {
            2 => Ok(vec![0.3, 0.3, 0.3]),
            _ => Err(RuleRefusal::warmup("test")),
        }
    });
    // Abort: the January warmup is tolerated, the March one is not.
    let err = simulate(&panel, &rule, &SimConfig::default()).unwrap_err();
    assert!(matches!(err, SimError::RuleRefused { date, .. } if date.month() == 3));
    // HoldPrevious: both recorded, run completes.
    let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() };
    let sim = simulate(&panel, &rule, &cfg).unwrap();
    assert_eq!(sim.refusals.len(), 2);
    assert!(sim.refusals.iter().all(|r| r.kind == RefusalKind::Warmup));
}

#[test]
fn refusing_from_the_start_leaves_the_book_flat_with_no_counted_window() {
    let panel = quarter_panel();
    let rule = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| {
        Err(RuleRefusal::data("no_data", "x"))
    });
    let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() };
    let sim = simulate(&panel, &rule, &cfg).unwrap();
    assert!(sim.equity.iter().all(|&e| e == 1.0));
    assert!(sim.units.iter().all(|&u| u == 0.0));
    assert!(sim.window.is_none() && sim.metrics().is_none());
    assert_eq!(sim.refusals.len(), panel.n_bars());
}

#[test]
fn malformed_weight_vectors_are_errors_never_partially_applied() {
    let panel = quarter_panel();
    let short = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.5, 0.5]));
    assert!(matches!(simulate(&panel, &short, &SimConfig::default()), Err(SimError::InvalidWeights { .. })));
    let nan =
        FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.5, f64::NAN, 0.1]));
    assert!(matches!(simulate(&panel, &nan, &SimConfig::default()), Err(SimError::InvalidWeights { .. })));
}

// ------------------------------------------------------------------------------------- (g) drift vs fixed weights

/// A: 100,100,110,121,121; B flat 100; w = (0.5, 0.5); decisions at bar 1 and bar 4.
///  OnDecision (drift):  E = 1, 1, 1.05, 1.105, 1.105 ; held A weight at bar 3 = 0.605/1.105
///  EveryBar (fixed):    E = 1, 1, 1.05, 1.1025, 1.1025; held weights exactly (0.5, 0.5) after every bar
fn two_asset_five_bar() -> Panel {
    panel_of(&AB, five_bar_dates(), vec![vec![100.0, 100.0, 110.0, 121.0, 121.0], vec![100.0; 5]])
}

#[test]
fn on_decision_lets_units_drift_hand_computed() {
    let rule = const_rule(&AB, vec![0.5, 0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision);
    let sim = simulate(&two_asset_five_bar(), &rule, &SimConfig::default()).unwrap();
    let want = [1.0, 1.0, 1.05, 1.105, 1.105];
    for t in 0..5 {
        assert!(close(sim.equity[t], want[t], 1e-15), "equity[{t}]");
    }
    // units are constant between decisions (bar 1 .. bar 3)
    for t in 2..=3 {
        assert_eq!(sim.row(&sim.units, t), sim.row(&sim.units, 1));
    }
    // held weights drift
    assert!(close(sim.row(&sim.held_weights, 3)[0], 0.605 / 1.105, 1e-15));
    assert!(close(sim.row(&sim.held_weights, 3)[1], 0.5 / 1.105, 1e-15));
    // and snap back to target at the decision bar
    assert!(close(sim.row(&sim.held_weights, 4)[0], 0.5, 1e-15));
    // trades happen only on decision bars (fills: bar1 both legs, bar4 both legs)
    assert_eq!(sim.rebalance_bars, 2);
    assert!(sim.traded_notional[2] == 0.0 && sim.traded_notional[3] == 0.0);
}

#[test]
fn every_bar_holds_fixed_weights_hand_computed() {
    let rule = const_rule(&AB, vec![0.5, 0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::EveryBar);
    let sim = simulate(&two_asset_five_bar(), &rule, &SimConfig::default()).unwrap();
    let want = [1.0, 1.0, 1.05, 1.1025, 1.1025];
    for t in 0..5 {
        assert!(close(sim.equity[t], want[t], 1e-15), "equity[{t}]");
    }
    for t in 1..5 {
        assert!(
            close(sim.row(&sim.held_weights, t)[0], 0.5, 1e-15) && close(sim.row(&sim.held_weights, t)[1], 0.5, 1e-15)
        );
    }
    // it trades on days with no decision (bar 2, bar 3)
    assert!(sim.traded_notional[2] > 0.0 && sim.traded_notional[3] > 0.0);
    // and the two policies really are different accounting on the SAME targets
    let on = simulate(
        &two_asset_five_bar(),
        &const_rule(&AB, vec![0.5, 0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision),
        &SimConfig::default(),
    )
    .unwrap();
    assert!(on.equity[4] != sim.equity[4]);
    assert_eq!(on.target_weights, sim.target_weights);
}

struct Rebased<R: WeightRule>(R, RebalancePolicy);
impl<R: WeightRule> WeightRule for Rebased<R> {
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn impl_version(&self) -> String {
        self.0.impl_version()
    }
    fn universe(&self) -> &[&'static str] {
        self.0.universe()
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.0.decision_schedule()
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.1
    }
    fn min_history_bars(&self) -> usize {
        self.0.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        self.0.target_weights(h)
    }
}

#[test]
fn wrong_rebalance_mode_is_invisible_to_correlation_but_not_to_per_bar_identity() {
    // S1 on the synthetic panel, correct mode (OnDecision) vs the wrong mode (EveryBar on the same targets).
    let panel = s1_panel();
    let right = simulate(&panel, &S1TestRule, &s1_config()).unwrap();
    let wrong = simulate(&panel, &Rebased(S1TestRule, RebalancePolicy::EveryBar), &s1_config()).unwrap();
    assert_eq!(right.window, wrong.window);
    let (a, b) = (right.window_returns(), wrong.window_returns());
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let (va, vb): (f64, f64) = (a.iter().map(|x| (x - ma).powi(2)).sum(), b.iter().map(|y| (y - mb).powi(2)).sum());
    let corr = cov / (va * vb).sqrt();
    let diff = max_abs_diff(a, b);
    println!("wrong-mode S1: corr {corr:.6}, max per-bar diff {diff:e}");
    assert!(corr > 0.99, "the bands would not see it");
    assert!(diff > 1e-4, "but the per-bar identity must");
    // and it does not match the key
    let (_, kr, _) = parse_returns(KEY_S1_RETURNS);
    assert!(max_abs_diff(b, &kr) > 1e-4);
}

// --------------------------------------------------------------------------------------------- property tests

fn levered_panel() -> Panel {
    synth_panel(&ABC, 260, 33, "2018-01-01")
}

#[test]
fn per_bar_returns_scale_linearly_with_target_weights_every_bar() {
    let panel = levered_panel();
    let rule = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| Ok(wfn(h.len() - 1)));
    let base = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    for k in [0.25, 0.5, 2.0, 3.0] {
        let scaled = simulate(&panel, &rule, &SimConfig { risk_scale: k, ..SimConfig::default() }).unwrap();
        let mut worst = 0.0f64;
        for t in 1..panel.n_bars() {
            worst = worst.max((scaled.ret[t] - k * base.ret[t]).abs());
        }
        assert!(worst <= 1e-12, "risk_scale {k}: linearity broken by {worst:e}");
    }
}

#[test]
fn asset_permutation_does_not_change_returns() {
    let panel = levered_panel();
    // A rule defined per SYMBOL: weight +0.4 / -0.3 / +0.2 for A / B / C, sign by 10-bar momentum of that symbol.
    let w_of = |sym: &str| match sym {
        "A" => 0.4,
        "B" => -0.3,
        _ => 0.2,
    };
    let make = |univ: &[&'static str]| {
        let u = univ.to_vec();
        FnRule::new(univ, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 11, move |h| {
            Ok((0..h.n_assets())
                .map(|i| {
                    let c = h.closes(i);
                    let mom = if c[c.len() - 1] > c[c.len() - 11] { 1.0 } else { -1.0 };
                    w_of(u[i]) * mom
                })
                .collect())
        })
    };
    let base = simulate(&panel, &make(&ABC), &SimConfig::default()).unwrap();
    for order in [[2usize, 0, 1], [1, 2, 0], [2, 1, 0]] {
        let permuted = panel.with_asset_order(&order);
        let univ: Vec<&'static str> = order.iter().map(|&i| ABC[i]).collect();
        let sim = simulate(&permuted, &make(&univ), &SimConfig::default()).unwrap();
        assert!(max_abs_diff(&sim.ret, &base.ret) <= 1e-12, "order {order:?}");
        assert!(max_abs_diff(&sim.equity, &base.equity) <= 1e-12);
    }
}

#[test]
fn self_financing_and_no_cash_creation_with_leverage_shorts_and_costs() {
    // Levered and signed: net +1.4 (borrowing) in one regime, net -0.6 with a short leg in the other; gross 2.2 / 2.2.
    fn wlev(t: usize) -> Vec<f64> {
        let s = if (t / 7) % 2 == 0 { 1.0 } else { -1.0 };
        vec![1.2 * s, 0.6 * s, -0.4]
    }
    let panel = levered_panel();
    let rule = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| Ok(wlev(h.len() - 1)));
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let sim = simulate(&panel, &rule, &cfg).unwrap();
    let k = 3;
    let mut worst_flow = 0.0f64;
    let mut worst_eq = 0.0f64;
    let mut saw_negative_cash = false;
    for t in 0..panel.n_bars() {
        // equity recomputed from the recorded cash and units
        let mut inv = 0.0;
        for i in 0..k {
            inv += sim.row(&sim.units, t)[i] * panel.closes(i)[t];
        }
        worst_eq = worst_eq.max((sim.cash[t] + inv - sim.equity[t]).abs() / sim.equity[t].abs().max(1.0));
        if sim.cash[t] < 0.0 {
            saw_negative_cash = true;
        }
        if t > 0 {
            // cash flow of the bar = -(bought units x price) - cost   (no financing configured)
            let mut spent = 0.0;
            for i in 0..k {
                spent += (sim.row(&sim.units, t)[i] - sim.row(&sim.units, t - 1)[i]) * panel.closes(i)[t];
            }
            let flow = sim.cash[t] - sim.cash[t - 1];
            worst_flow = worst_flow.max((flow - (-spent - sim.cost[t])).abs() / sim.equity[t].abs().max(1.0));
        }
        // exposures reported are those of the recorded weights
        let g: f64 = sim.row(&sim.held_weights, t).iter().map(|x| x.abs()).sum();
        let nn: f64 = sim.row(&sim.held_weights, t).iter().sum();
        assert!((g - sim.gross_exposure[t]).abs() < 1e-12 && (nn - sim.net_exposure[t]).abs() < 1e-12);
    }
    assert!(saw_negative_cash, "the test weights are levered, so cash must go negative (borrowing)");
    assert!(
        worst_eq <= 1e-12 && worst_flow <= 1e-10,
        "equity identity {worst_eq:e}, cash-flow identity {worst_flow:e}"
    );
    // Zero cost, EveryBar: post-trade exposures equal the target's gross (2.2) and net (+1.4 / -0.6) exactly.
    let z = simulate(&panel, &rule, &SimConfig::default()).unwrap();
    for t in 0..panel.n_bars() {
        let w = wlev(t);
        let g: f64 = w.iter().map(|x| x.abs()).sum();
        let nn: f64 = w.iter().sum();
        assert!((z.gross_exposure[t] - g).abs() < 1e-12 && (z.net_exposure[t] - nn).abs() < 1e-12, "bar {t}");
    }
}

#[test]
fn a_flat_rule_creates_no_money() {
    let panel = levered_panel();
    let rule = const_rule(&ABC, vec![0.0, 0.0, 0.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let sim = simulate(&panel, &rule, &cfg).unwrap();
    assert!(sim.equity.iter().all(|&e| e == 1.0));
    assert!(sim.ret.iter().all(|&r| r == 0.0));
    assert_eq!(sim.total_cost(), 0.0);
}

#[test]
fn nonpositive_equity_is_an_error_not_a_negative_number() {
    // 10x long into a 20% fall wipes the account out.
    let dates = weekdays(d("2020-03-02"), 4);
    let panel = panel_of(&A, dates, vec![vec![100.0, 100.0, 80.0, 80.0]]);
    let rule = const_rule(&A, vec![10.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    assert!(matches!(simulate(&panel, &rule, &SimConfig::default()), Err(SimError::NonPositiveEquity { .. })));
}

// -------------------------------------------------------------------------------------- financing, delay, limits

#[test]
fn financing_accrues_actual_365_over_calendar_days_hand_computed() {
    // Fri 2020-01-31 (bar 1) -> Mon 2020-02-03 (bar 2): 3 calendar days. Flat price, long 0.5 EveryBar, cash 0.5.
    let panel = panel_of(&A, five_bar_dates(), vec![vec![100.0; 5]]);
    let rule = const_rule(&A, vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let cfg = SimConfig {
        financing: Financing::FlatAnnual { long_bps: 200.0, short_bps: 0.0, cash_bps: 100.0 },
        ..SimConfig::default()
    };
    let sim = simulate(&panel, &rule, &cfg).unwrap();
    // bar 0 -> bar 1: 1 day, entered at bar 0 (units 0.005, cash 0.5)
    let f1 = 1.0 / 365.0 * (0.01 * 0.5 - 0.02 * 0.5);
    assert!(close(sim.financing[1], f1, 1e-18));
    // bar 1 -> bar 2: 3 days on the same notionals (equity moved slightly, so recompute from the recorded state)
    let f2 = 3.0 / 365.0 * (0.01 * sim.cash[1] - 0.02 * (sim.row(&sim.units, 1)[0] * 100.0));
    assert!(close(sim.financing[2], f2, 1e-18), "{} vs {}", sim.financing[2], f2);
    assert!(sim.financing[2] < 0.0 && sim.financing[0] == 0.0);
    // shorts pay short_bps; borrowing (negative cash) pays the cash rate as a debit
    let short_rule = const_rule(&A, vec![-0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let cfg2 = SimConfig {
        financing: Financing::FlatAnnual { long_bps: 0.0, short_bps: 300.0, cash_bps: 0.0 },
        ..SimConfig::default()
    };
    let s2 = simulate(&panel, &short_rule, &cfg2).unwrap();
    assert!(close(s2.financing[1], 1.0 / 365.0 * (-0.03 * 0.5), 1e-18));
}

#[test]
fn execution_delay_shifts_the_fill_by_whole_bars() {
    let panel = levered_panel();
    let rule = FnRule::new(&ABC, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| Ok(wfn(h.len() - 1)));
    let d1 = simulate(&panel, &rule, &SimConfig { execution_delay_bars: 1, ..SimConfig::default() }).unwrap();
    // decision at bar t-2 earns bar t
    let mut worst = 0.0f64;
    for t in 2..panel.n_bars() {
        let w = wfn(t - 2);
        let mut r = 0.0;
        for i in 0..3 {
            r += w[i] * (panel.closes(i)[t] / panel.closes(i)[t - 1] - 1.0);
        }
        worst = worst.max((d1.ret[t] - r).abs());
    }
    assert!(worst <= 1e-12, "delay-1 closed form differs by {worst:e}");
    assert_eq!(d1.window.unwrap().first_bar, 2, "first fill at bar 1, first counted return at bar 2");
}

#[test]
fn max_gross_refuses_instead_of_clipping() {
    let panel = levered_panel();
    let rule = const_rule(&ABC, vec![1.5, -1.0, 0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar); // gross 3.0
    let refused = simulate(&panel, &rule, &SimConfig { max_gross: Some(2.0), ..SimConfig::default() });
    assert!(
        matches!(refused, Err(SimError::MaxGrossBreached { gross, limit, .. }) if close(gross, 3.0, 1e-12) && limit == 2.0)
    );
    let ok = simulate(&panel, &rule, &SimConfig { max_gross: Some(3.0), ..SimConfig::default() }).unwrap();
    assert!(close(ok.gross_exposure[10], 3.0, 1e-12));
}

#[test]
fn universe_mismatch_and_bad_config_are_errors() {
    let panel = levered_panel();
    let wrong = const_rule(&["A", "B"], vec![0.5, 0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    assert!(matches!(simulate(&panel, &wrong, &SimConfig::default()), Err(SimError::UniverseMismatch { .. })));
    let reordered =
        const_rule(&["B", "A", "C"], vec![0.1, 0.1, 0.1], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    assert!(matches!(simulate(&panel, &reordered, &SimConfig::default()), Err(SimError::UniverseMismatch { .. })));
    let ok = const_rule(&ABC, vec![0.1, 0.1, 0.1], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    assert!(matches!(
        simulate(&panel, &ok, &SimConfig { initial_equity: 0.0, ..SimConfig::default() }),
        Err(SimError::BadConfig(_))
    ));
    assert!(matches!(
        simulate(&panel, &ok, &SimConfig { max_gross: Some(-1.0), ..SimConfig::default() }),
        Err(SimError::BadConfig(_))
    ));
    let mut bad_cost = CostModel::ZERO;
    bad_cost.slippage_bps = -1.0;
    assert!(matches!(
        simulate(&panel, &ok, &SimConfig { cost: bad_cost, ..SimConfig::default() }),
        Err(SimError::BadConfig(_))
    ));
}

#[test]
fn returns_are_independent_of_initial_equity() {
    let panel = s3_panel();
    let a = simulate(&panel, &S3TestRule, &s3_config()).unwrap();
    let b = simulate(&panel, &S3TestRule, &SimConfig { initial_equity: 1_000_000.0, ..s3_config() }).unwrap();
    assert!(max_abs_diff(a.window_returns(), b.window_returns()) <= 1e-12);
}

#[test]
fn exposure_statistics_are_reported_over_the_counted_window() {
    let rule = const_rule(&AB, vec![0.5, 0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::EveryBar);
    let sim = simulate(&two_asset_five_bar(), &rule, &SimConfig::default()).unwrap();
    let g = sim.gross_exposure_stats.unwrap();
    assert!(
        close(g.mean, 1.0, 1e-15)
            && close(g.median, 1.0, 1e-15)
            && close(g.p90, 1.0, 1e-15)
            && close(g.max, 1.0, 1e-15)
    );
    // drifting book: held gross is 1.0 always (long-only, fully invested), net likewise; use the 0.5-cash variant
    let half = const_rule(&AB, vec![0.25, 0.25], DecisionSchedule::LastBarOfMonth, RebalancePolicy::EveryBar);
    let s2 = simulate(&two_asset_five_bar(), &half, &SimConfig::default()).unwrap();
    assert!(close(s2.gross_exposure_stats.unwrap().mean, 0.5, 1e-15));
    assert!(close(s2.net_exposure_stats.unwrap().max, 0.5, 1e-15));
}
