//! PF1 book properties (design 6.5, PF1 row): attribution identity, share linearity, sleeve and instrument permutation
//! invariance, cash conservation, hold-previous whole-book refusal, union-clock carry that preserves the multi-day
//! return, cost identity, zero-cost identity, the gross-cap refusal boundary, `IndependentSubAccounts` reproducing the
//! legacy sum of curves, the overlay hook, stateful rules, data gaps, cadence details and configuration errors.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::book::*;
use common::*;
use std::sync::{Arc, Mutex};
use weightsim::*;

const NET: CostModel = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;

fn run_case(c: &Case) -> (BookResult, BookResult) {
    let (panel, book, cfg) = build_case(c);
    simulate_book_gross_and_net(&panel, &book, &cfg).unwrap()
}

/// Daily panel from named price series starting at `start`; `closed_weekends` gives the series an exchange session
/// (closed Saturday and Sunday) and skips those dates.
fn mini_panel(start: &str, days: usize, series: &[(&str, bool, Vec<f64>)]) -> BookPanel {
    let all: Vec<Date> = (0..days).map(|i| d(start).add_days(i as i64)).collect();
    let mut rows = Vec::new();
    for (name, exchange, prices) in series {
        let mut r = Vec::new();
        let mut p = 0;
        for date in &all {
            if *exchange && date.weekday() >= 5 {
                continue;
            }
            r.push((*date, prices[p]));
            p += 1;
        }
        assert_eq!(p, prices.len(), "{name}: price count must match the number of bars");
        let session = if *exchange { SessionKind::exchange("test", vec![]) } else { SessionKind::Continuous };
        rows.push((name.to_string(), session, r));
    }
    BookPanel::from_dated_series(rows).unwrap()
}

/// A sleeve that decides on its first own bar only (a buy-and-hold entry) and refuses afterwards.
fn once_rule(universe: &[&'static str], w: Vec<f64>, policy: RebalancePolicy) -> FnRule {
    FnRule::new(universe, DecisionSchedule::Daily, policy, 1, move |h| {
        if h.len() == 1 {
            Ok(w.clone())
        } else {
            Err(RuleRefusal::data("no_change", "the rule decides once"))
        }
    })
}

fn hold_cfg() -> BookConfig {
    let mut cfg = BookConfig::default();
    cfg.sim.on_refusal = OnRefusal::HoldPrevious;
    cfg
}

// ------------------------------------------------------------------------------------------ attribution identity
#[test]
fn attribution_identity_holds_to_1e_12_on_every_configuration_gross_and_net() {
    let mut checked = 0;
    for c in cases() {
        let (g, n) = run_case(&c);
        for (what, r) in [("gross", &g), ("net", &n)] {
            let a = r.attribution_report();
            assert!(
                a.max_abs_err_total <= 1e-12,
                "{} {what}: ret vs SUM w r - cost: {:e}",
                c.name,
                a.max_abs_err_total
            );
            assert!(
                a.max_abs_err_contrib_sum <= 1e-12,
                "{} {what}: contributions vs total: {:e}",
                c.name,
                a.max_abs_err_contrib_sum
            );
            assert_eq!(a.bars_checked, r.n_bars() - 1);
            checked += 1;
        }
    }
    // shared instruments, signed weights, netting
    let panel = netting_panel();
    let (g, n) = simulate_book_gross_and_net(&panel, &netting_book(&panel), &netting_config()).unwrap();
    for r in [&g, &n] {
        let a = r.attribution_report();
        assert!(a.max_abs_err_total <= 1e-12 && a.max_abs_err_contrib_sum <= 1e-12, "{a:?}");
        checked += 1;
    }
    assert_eq!(checked, 2 * cases().len() + 2);
}

#[test]
fn attribution_identity_includes_financing() {
    let (panel, book, mut cfg) = build_case(&case("book_live_60_40"));
    cfg.sim.financing = Financing::FlatAnnual { long_bps: 300.0, short_bps: 100.0, cash_bps: 150.0 };
    let r = simulate_book(&panel, &book, &cfg).unwrap();
    assert!(r.financing.iter().any(|f| *f != 0.0), "financing accrued");
    // weekends: carry rows accrue over calendar days between clock bars, one day per bar here
    let a = r.attribution_report();
    assert!(a.max_abs_err_total <= 1e-12, "{a:?}");
    // financing is credit-positive in the column and earned on cash: the book holds cash, so the net effect is positive at 150bps/yr on cash
    // versus 300 bps on long value; just check the sign convention on a pure-cash book below
    let flat = FnRule::new(&["C1"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.0]));
    let p = mini_panel("2020-01-01", 6, &[("C1", false, vec![10.0; 6])]);
    let b = Book::new(vec![SleeveSpec::from_rule("s", flat, vec![0], ShareSpec::Fixed(1.0))]);
    let mut c2 = BookConfig::default();
    c2.sim.financing = Financing::FlatAnnual { long_bps: 0.0, short_bps: 0.0, cash_bps: 365.0 };
    let r = simulate_book(&p, &b, &c2).unwrap();
    // 365 bps per annum on cash = 1e-4 per calendar day, actual/365
    for k in 1..r.n_bars() {
        assert!((r.financing[k] - 1e-4 * r.cash[k - 1]).abs() < 1e-15, "bar {k}: {}", r.financing[k]);
    }
    assert!(r.equity[5] > 1.0);
}

// ------------------------------------------------------------------------------------------ linearity, permutation
#[test]
fn gross_return_is_linear_in_the_shares_of_every_bar_sleeves() {
    // Two EveryBar sleeves sharing an instrument: r = SUM_s share_s * (SUM_j w_sj r_j), exactly linear.
    let panel = netting_panel();
    let cfg0 = BookConfig {
        sim: SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() },
        ..BookConfig::default()
    };
    let sl = |id: &str, a: bool, share: f64, p: &BookPanel| {
        if a {
            SleeveSpec::from_rule(id, NetRuleA, instrument_index(p, &["X", "Y"]), ShareSpec::Fixed(share))
        } else {
            SleeveSpec::from_rule(id, NetRuleB, instrument_index(p, &["X", "Z"]), ShareSpec::Fixed(share))
        }
    };
    let solo_a = simulate_book(&panel, &Book::new(vec![sl("a", true, 1.0, &panel)]), &cfg0).unwrap();
    let solo_b = simulate_book(&panel, &Book::new(vec![sl("b", false, 1.0, &panel)]), &cfg0).unwrap();
    for (sa, sb) in [(0.5, 0.5), (0.3, 0.9), (1.2, 0.4)] {
        let both =
            simulate_book(&panel, &Book::new(vec![sl("a", true, sa, &panel), sl("b", false, sb, &panel)]), &cfg0)
                .unwrap();
        for k in 1..both.n_bars() {
            let want = sa * solo_a.ret[k] + sb * solo_b.ret[k];
            assert!((both.ret[k] - want).abs() <= 1e-12, "shares {sa}/{sb} bar {k}: {} vs {want}", both.ret[k]);
        }
    }
}

#[test]
fn sleeve_and_instrument_permutation_do_not_change_the_answer() {
    let c = case("book_live_60_40");
    let (panel, book, cfg) = build_case(&c);
    let base = simulate_book_gross_and_net(&panel, &book, &cfg).unwrap();
    // sleeves in the opposite order
    let mut rev = book.clone();
    rev.sleeves.reverse();
    let swapped = simulate_book_gross_and_net(&panel, &rev, &cfg).unwrap();
    // instruments in another order, the sleeves' universes re-pointed at the new indices
    let order = [4usize, 2, 0, 3, 1];
    let perm = panel.with_instrument_order(&order);
    let mut pbook = book_for_case(&perm, &c);
    pbook.sleeves.reverse();
    pbook.sleeves.reverse();
    let permuted = simulate_book_gross_and_net(&perm, &pbook, &cfg).unwrap();
    for (name, other) in [("sleeve order", &swapped), ("instrument order", &permuted)] {
        for (a, b) in [(&base.0, &other.0), (&base.1, &other.1)] {
            for k in 0..a.n_bars() {
                assert!((a.ret[k] - b.ret[k]).abs() <= 1e-12, "{name}: ret bar {k}");
                assert!((a.equity[k] - b.equity[k]).abs() <= 1e-12, "{name}: equity bar {k}");
                assert!((a.cost[k] - b.cost[k]).abs() <= 1e-12, "{name}: cost bar {k}");
            }
        }
    }
    // per-instrument weights follow the instruments through the permutation
    for (j_new, &j_old) in order.iter().enumerate() {
        let name_old = &panel.instruments()[j_old];
        assert_eq!(&perm.instruments()[j_new], name_old);
        for k in 0..base.0.n_bars() {
            let w_old = base.0.held_weights[k * 5 + j_old];
            let w_new = permuted.0.held_weights[k * 5 + j_new];
            assert!((w_old - w_new).abs() <= 1e-12, "held weight of {name_old} bar {k}");
        }
    }
}

// ------------------------------------------------------------------------------------------ conservation
#[test]
fn cash_is_conserved_and_equity_is_cash_plus_marked_units() {
    for name in ["book_cert_60_40", "book_live_60_40", "book_scaled_50_30", "book_invvol"] {
        let (g, n) = run_case(&case(name));
        let ni = g.n_instruments();
        for r in [&g, &n] {
            for k in 1..r.n_bars() {
                let mut mv = 0.0;
                let mut dmv = 0.0;
                for j in 0..ni {
                    mv += r.units[k * ni + j] * r.marks[k * ni + j];
                    dmv += (r.units[k * ni + j] - r.units[(k - 1) * ni + j]) * r.marks[k * ni + j];
                }
                assert!((r.equity[k] - (r.cash[k] + mv)).abs() <= 1e-12, "{name}: equity = cash + units*mark at {k}");
                // cash moves only by trades and their cost: cash_k - cash_(k-1) = -SUM d_units * mark - cost (+ financing)
                let dcash = r.cash[k] - r.cash[k - 1];
                assert!((dcash - (-dmv - r.cost[k] + r.financing[k])).abs() <= 1e-12, "{name}: cash flow at {k}");
            }
        }
    }
}

#[test]
fn nothing_is_traded_or_charged_on_a_refused_bar_and_units_are_held() {
    let (g, n) = run_case(&case("book_grosscap_60_40"));
    let ni = g.n_instruments();
    let mut refused = 0;
    for k in 1..g.n_bars() {
        if g.book_refused[k] {
            refused += 1;
            for r in [&g, &n] {
                assert_eq!(r.traded_notional[k], 0.0);
                assert_eq!(r.cost[k], 0.0);
                assert_eq!(r.inst_row(&r.units, k), r.inst_row(&r.units, k - 1), "units held on refused bar {k}");
                assert_eq!(r.traded_by_instrument[k * ni..(k + 1) * ni].iter().sum::<f64>(), 0.0);
            }
        }
    }
    assert!(refused > 5);
    // and the refusal is reported with its numbers, once per refused bar
    let rs = g.refusals_with_code("gross_above_cap");
    assert_eq!(rs.len(), (0..g.n_bars()).filter(|&k| g.book_refused[k]).count());
    assert!(rs.iter().all(|r| r.sleeve.is_none() && r.message.contains("exceeds cap")));
}

#[test]
fn a_refusing_sleeve_keeps_its_standing_target_and_the_other_sleeves_carry_on() {
    let panel = mini_panel(
        "2020-01-01",
        12,
        &[
            ("A", false, (0..12).map(|i| 10.0 + i as f64).collect()),
            ("B", false, (0..12).map(|i| 20.0 - i as f64 * 0.5).collect()),
        ],
    );
    let a = FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| {
        if h.len() >= 5 && h.len() <= 7 {
            Err(RuleRefusal::data("stale", "stale"))
        } else {
            Ok(vec![if h.len() < 5 { 0.2 } else { 0.4 }])
        }
    });
    let b = FnRule::new(&["B"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.3]));
    let book = Book::new(vec![
        SleeveSpec::from_rule("a", a, vec![0], ShareSpec::Fixed(0.5)),
        SleeveSpec::from_rule("b", b, vec![1], ShareSpec::Fixed(0.5)),
    ]);
    let r = simulate_book(&panel, &book, &hold_cfg()).unwrap();
    // bars 4..=6 (h.len() 5..=7): sleeve a refused, holds the 0.2 target; sleeve b unaffected
    for k in 4..=6 {
        assert!(r.rule_refused[k * 2] && !r.rule_refused[k * 2 + 1], "bar {k}");
        assert!((r.target_weights[k * 2] - 0.5 * 0.2).abs() < 1e-15, "a keeps its previous target on bar {k}");
        assert!((r.target_weights[k * 2 + 1] - 0.5 * 0.3).abs() < 1e-15);
    }
    assert!(
        (r.target_weights[7 * 2] - 0.5 * 0.4).abs() < 1e-15,
        "a's new decision takes effect once the rule answers again"
    );
    assert_eq!(r.refusals.iter().filter(|x| x.sleeve == Some(0)).count(), 3);
    assert!(r.refusals.iter().all(|x| x.code == "stale"));
    // under Abort the same refusal is an error naming the date and code
    let e = simulate_book(&panel, &book, &BookConfig::default()).unwrap_err();
    assert!(matches!(e, BookError::Sim(SimError::RuleRefused { .. })), "{e:?}");
}

// ------------------------------------------------------------------------------------------ union clock
#[test]
fn carry_rows_earn_exactly_zero_and_the_next_real_bar_earns_the_full_change() {
    // Fri 2020-01-03 .. Tue 2020-01-07. E trades Mon-Fri only (Fri 100, Mon 110, Tue 121); C is flat and out of the market.
    let panel = mini_panel("2020-01-03", 5, &[("E", true, vec![100.0, 110.0, 121.0]), ("C", false, vec![10.0; 5])]);
    let book = Book::new(vec![
        SleeveSpec::from_rule(
            "e",
            once_rule(&["E"], vec![0.6], RebalancePolicy::OnDecision),
            vec![0],
            ShareSpec::Fixed(0.5),
        ),
        SleeveSpec::from_rule(
            "c",
            once_rule(&["C"], vec![0.0], RebalancePolicy::EveryBar),
            vec![1],
            ShareSpec::Fixed(0.5),
        ),
    ]);
    let r = simulate_book(&panel, &book, &hold_cfg()).unwrap();
    assert_eq!(r.n_bars(), 5);
    let dates: Vec<String> = r.times.iter().map(|t| t.date().to_string()).collect();
    assert_eq!(dates, ["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]);
    assert_eq!(r.sleeve_open.chunks(2).map(|c| c[0]).collect::<Vec<_>>(), [true, false, false, true, true]);
    // marks of E on the weekend are the carried Friday close; the return of E over the weekend is exactly 0.0
    let e = 0;
    assert_eq!(r.marks[1 * 2 + e], 100.0);
    assert_eq!(r.marks[2 * 2 + e], 100.0);
    assert_eq!(r.contrib[2], 0.0, "Saturday contribution of the ETF sleeve is exactly zero");
    assert_eq!(r.contrib[2 * 2], 0.0);
    // Monday earns the FULL change since Friday: units = 0.3/100, +10 => +0.03 of the initial equity
    assert!((r.ret[3] - 0.03).abs() < 1e-15, "{}", r.ret[3]);
    assert!((r.contrib[3 * 2] - 0.03).abs() < 1e-15);
    // no return is lost: the product of the rows equals the price change of the held position
    let prod = (1.0 + r.ret[1]) * (1.0 + r.ret[2]) * (1.0 + r.ret[3]);
    assert!((prod - 1.03).abs() < 1e-14, "{prod}");
    assert!(r.attribution_report().max_abs_err_total <= 1e-12);
}

#[test]
fn a_closed_market_cannot_trade_unless_the_diagnostic_flag_says_so() {
    let panel = mini_panel(
        "2020-01-03",
        5,
        &[("E", true, vec![100.0, 110.0, 121.0]), ("C", false, vec![10.0, 11.0, 12.0, 11.0, 10.0])],
    );
    let mk = || {
        Book::new(vec![
            SleeveSpec::from_rule(
                "e",
                once_rule(&["E"], vec![0.6], RebalancePolicy::OnDecision),
                vec![0],
                ShareSpec::Fixed(0.5),
            ),
            SleeveSpec::from_rule(
                "c",
                FnRule::new(&["C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.5])),
                vec![1],
                ShareSpec::Fixed(0.5),
            ),
        ])
    };
    let mut cfg = hold_cfg();
    cfg.cadence = BookCadence::AllSleevesOnAnyDue;
    let r = simulate_book(&panel, &mk(), &cfg).unwrap();
    // Saturday (bar 1): the crypto sleeve is due, the driver runs, the ETF sleeve is CLOSED and is not planned
    assert!(r.run[1] && r.planned[2 + 1] && !r.planned[2], "closed ETF market is not planned");
    assert!(r.planned[3 * 2], "open again on Monday: planned (re-targeted by the driver)");
    cfg.trade_on_closed_market = true;
    let r2 = simulate_book(&panel, &mk(), &cfg).unwrap();
    assert!(r2.planned[2], "diagnostic: closed market planned at the stale close");
    // PerSleeve: the ETF sleeve is due only on its decision bar
    let r3 = simulate_book(&panel, &mk(), &hold_cfg()).unwrap();
    assert_eq!(r3.due.chunks(2).filter(|c| c[0]).count(), 1);
}

#[test]
fn month_end_decisions_use_the_sleeves_own_calendar_not_the_union_clock() {
    let (g, _) = run_case(&case("book_cert_60_40"));
    // union clock has Sat 2019-03-30 and Sun 03-31; the ETF's last March bar is Fri 03-29 (the first decision, k = 0)
    let dec_dates: Vec<String> =
        (0..g.n_bars()).filter(|&k| g.decision[k * 2]).map(|k| g.times[k].date().to_string()).collect();
    assert_eq!(
        dec_dates,
        ["2019-03-29", "2019-04-30", "2019-05-31", "2019-06-28"],
        "last ETF bars of Mar, Apr, May, Jun"
    );
    // June 28 is a Friday and the last ETF bar of the month; the crypto sleeve decides every day
    let cry_days = (0..g.n_bars()).filter(|&k| g.decision[k * 2 + 1]).count();
    assert_eq!(cry_days, g.n_bars(), "crypto decides on every clock bar");
}

#[test]
fn execution_delay_counts_the_decision_sleeves_own_bars() {
    // ETF sleeve decides on Friday 2020-01-03 (own bar 0) and, with delay 1, fills on its NEXT OWN bar (Monday), not on Saturday.
    let panel = mini_panel("2020-01-03", 5, &[("E", true, vec![100.0, 110.0, 121.0]), ("C", false, vec![10.0; 5])]);
    let book = Book::new(vec![
        SleeveSpec::from_rule(
            "e",
            once_rule(&["E"], vec![0.6], RebalancePolicy::OnDecision),
            vec![0],
            ShareSpec::Fixed(1.0),
        ),
        SleeveSpec::from_rule(
            "c",
            once_rule(&["C"], vec![0.0], RebalancePolicy::EveryBar),
            vec![1],
            ShareSpec::Fixed(0.5),
        ),
    ]);
    let mut cfg = hold_cfg();
    cfg.sim.execution_delay_bars = 1;
    let r = simulate_book(&panel, &book, &cfg).unwrap();
    assert_eq!(r.units[0], 0.0, "not filled on the decision bar");
    assert_eq!(r.units[2], 0.0, "not filled on the closed Saturday");
    assert!(r.units[3 * 2] > 0.0, "filled on the ETF's next own bar (Monday)");
    // the standing target became effective on Monday: PerSleeve due flag on Monday only
    assert_eq!(r.due.chunks(2).map(|c| c[0]).collect::<Vec<_>>(), [false, false, false, true, false]);
}

#[test]
fn account_start_gives_warm_up_history_to_rules_but_trades_and_decides_only_from_the_start() {
    let panel = mini_panel("2020-01-01", 30, &[("C", false, (0..30).map(|i| 10.0 + i as f64).collect())]);
    let seen = Arc::new(Mutex::new(Vec::<usize>::new()));
    let s2 = seen.clone();
    let rule = FnRule::new(&["C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, move |h| {
        s2.lock().unwrap().push(h.len());
        Ok(vec![0.5])
    });
    let mut cfg = BookConfig::default();
    cfg.account_start = Some(BarTime::from_date(d("2020-01-11")));
    let r =
        simulate_book(&panel, &Book::new(vec![SleeveSpec::from_rule("s", rule, vec![0], ShareSpec::Fixed(1.0))]), &cfg)
            .unwrap();
    assert_eq!(r.n_bars(), 20, "the account clock starts at 2020-01-11");
    assert_eq!(r.times[0].date(), d("2020-01-11"));
    let calls = seen.lock().unwrap().clone();
    assert_eq!(calls[0], 11, "the first decision sees the 10 warm-up bars plus the start bar");
    assert_eq!(calls.len(), 20, "no decision is taken before the account starts");
    assert_eq!(r.equity[0], 1.0);
    assert_eq!(r.clock_index[0], 10);
    // after the last bar
    let bad = BookConfig { account_start: Some(BarTime::from_date(d("2021-01-01"))), ..BookConfig::default() };
    let book = Book::new(vec![SleeveSpec::from_rule(
        "s",
        const_rule(&["C"], vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
        vec![0],
        ShareSpec::Fixed(1.0),
    )]);
    assert!(matches!(simulate_book(&panel, &book, &bad), Err(BookError::BadBook(_))));
}

// ------------------------------------------------------------------------------------------ gaps
#[test]
fn a_missing_bar_on_an_open_market_is_a_named_gap_never_a_silent_fill() {
    let panel = key_book_panel();
    let c1 = panel.instrument_index("C1").unwrap();
    // remove C1's bar on an interior weekend day (crypto is Continuous, so this is a GAP, not a closure)
    let u = panel.times().iter().position(|t| t.date() == d("2019-05-12")).unwrap();
    let holed = panel.with_bars_removed(&[(c1, u)]);
    let book = book_for_case(&holed, &case("book_cert_60_40"));
    let cfg = config_for_case(&case("book_cert_60_40"));
    // Abort (the certification default): an error naming the sleeve, the instrument and the bar
    let mut abort = cfg.clone();
    abort.sim.on_refusal = OnRefusal::Abort;
    match simulate_book(&holed, &book, &abort).unwrap_err() {
        BookError::DataGap { sleeve, instrument, time } => {
            assert_eq!((sleeve.as_str(), instrument.as_str()), ("cry", "C1"));
            assert_eq!(time.date(), d("2019-05-12"));
        }
        other => panic!("expected DataGap, got {other:?}"),
    }
    // HoldPrevious: recorded, the sleeve is not open that bar, valuations carry, and the next bar earns the full change
    let r = simulate_book(&holed, &book, &cfg).unwrap();
    let k = (0..r.n_bars()).find(|&k| r.times[k].date() == d("2019-05-12")).unwrap();
    let gaps = r.refusals_with_code("data_gap");
    assert_eq!(gaps.len(), 1);
    assert_eq!((gaps[0].bar, gaps[0].sleeve), (k, Some(1)));
    assert!(gaps[0].message.contains("C1"));
    assert!(!r.sleeve_open[k * 2 + 1], "the crypto sleeve has no own bar on the gap day");
    let ni = r.n_instruments();
    for j in [c1, panel.instrument_index("C2").unwrap()] {
        assert_eq!(
            r.marks[k * ni + j],
            r.marks[(k - 1) * ni + j],
            "both crypto instruments are carried at the last close"
        );
    }
    let real_next = panel.close(c1)[u + 1].unwrap();
    assert_eq!(r.marks[(k + 1) * ni + c1], real_next);
    assert!(r.attribution_report().max_abs_err_total <= 1e-12, "no return lost over the gap");
    // a declared closure (holiday) is not a gap: the ETF sleeve is closed on Good Friday and nothing is refused
    let good_friday = panel.times().iter().position(|t| t.date() == d("2019-04-19")).unwrap();
    let e1 = panel.instrument_index("E1").unwrap();
    assert_eq!(panel.availability(e1, good_friday), Availability::Closed);
    // ... while an UNDECLARED missing ETF weekday is a gap on the ETF sleeve
    let wed = panel.times().iter().position(|t| t.date() == d("2019-05-15")).unwrap();
    let holed2 = panel.with_bars_removed(&[(e1, wed)]);
    assert_eq!(holed2.availability(e1, wed), Availability::Gap);
    let r2 = simulate_book(&holed2, &book_for_case(&holed2, &case("book_cert_60_40")), &cfg).unwrap();
    assert_eq!(r2.refusals_with_code("data_gap").len(), 1);
    assert_eq!(r2.refusals_with_code("data_gap")[0].sleeve, Some(0));
}

// ------------------------------------------------------------------------------------------ costs
#[test]
fn zero_cost_preset_is_bit_identical_to_the_gross_run_and_the_cost_identity_holds() {
    for name in ["book_cert_60_40", "book_live_60_40", "book_scaled_50_30", "book_invvol"] {
        let c = case(name);
        let (panel, book, mut cfg) = build_case(&c);
        let (g, n) = simulate_book_gross_and_net(&panel, &book, &cfg).unwrap();
        cfg.sim.cost = CostModel::ZERO;
        let z = simulate_book(&panel, &book, &cfg).unwrap();
        assert_eq!(z.series_sha256, g.series_sha256, "{name}: the explicit zero preset is the gross run");
        assert_eq!(z.total_cost(), 0.0);
        // cost identity: per bar cost = rate * traded notional, exactly (the net run's own turnover)
        let rate = NET.rate();
        for k in 0..n.n_bars() {
            assert_eq!(n.cost[k], n.traded_notional[k] * rate, "{name}: bar {k}");
        }
        assert!(n.total_cost() > 0.0);
        // total cost = rate x total traded to summation rounding
        assert!(
            (n.total_cost() - rate * n.total_traded_notional()).abs() <= 1e-15 * n.total_traded_notional().max(1.0)
        );
        // net vs gross final equity within 10% of turnover x cost (first-order identity, as in T1)
        let mut pred = 0.0;
        for k in 0..n.n_bars() {
            if n.traded_notional[k] > 0.0 {
                pred += rate * (n.traded_notional[k] / n.equity_pre[k]);
            }
        }
        let last = n.n_bars() - 1;
        let actual = 1.0 - n.equity[last] / g.equity[last];
        assert!(pred > 0.0 && (actual - pred).abs() / pred < 0.10, "{name}: actual {actual} vs predicted {pred}");
    }
}

#[test]
fn shadow_curves_are_charged_their_own_costs_never_the_joint_accounts() {
    let (g, n) = run_case(&case("book_live_60_40"));
    // gross run: shadow gross == shadow cost (no cost anywhere)
    assert_eq!(g.shadow_ret_gross, g.shadow_ret_cost);
    // net run: the gross shadow is unchanged, the cost shadow is lower on the bars where the sleeve alone rebalanced
    assert_eq!(n.shadow_ret_gross, g.shadow_ret_gross);
    assert_ne!(n.shadow_ret_cost, n.shadow_ret_gross);
    // the crypto shadow (EveryBar) pays its own turnover cost: net shadow return below gross on every bar it traded
    let below = (1..n.n_bars()).filter(|&k| n.shadow_ret_cost[k * 2 + 1] < n.shadow_ret_gross[k * 2 + 1]).count();
    assert!(below > 40);
    // the joint account's cost on a bar where ONLY the ETF sleeve traded must not appear in the crypto shadow
    // (book_cert: PerSleeve; on ETF decision bars the joint cost includes the ETF trade, the crypto shadow does not)
    let (g2, n2) = run_case(&case("book_cert_60_40"));
    for k in 1..n2.n_bars() {
        if g2.decision[k * 2] {
            let joint_cost_frac = n2.cost[k] / n2.equity[k - 1];
            let cry_shadow_gap = n2.shadow_ret_gross[k * 2 + 1] - n2.shadow_ret_cost[k * 2 + 1];
            assert!(joint_cost_frac > 0.0);
            assert!((cry_shadow_gap - joint_cost_frac).abs() > 1e-9 || joint_cost_frac == 0.0);
        }
    }
}

// ------------------------------------------------------------------------------------------ gross cap boundary at 1e-9
#[test]
fn book_level_gross_cap_boundary_refuses_beyond_1e_9_and_permits_up_to_it() {
    let panel = mini_panel(
        "2020-01-01",
        8,
        &[
            ("A", false, (0..8).map(|i| 10.0 + i as f64).collect()),
            ("B", false, (0..8).map(|i| 20.0 - i as f64).collect()),
        ],
    );
    let mk = |cap: f64| {
        let book = Book::new(vec![SleeveSpec::from_rule(
            "s",
            FnRule::new(&["A", "B"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.6, -0.4])),
            vec![0, 1],
            ShareSpec::Fixed(1.0),
        )]);
        let mut cfg = hold_cfg();
        cfg.sim.max_gross = Some(cap);
        simulate_book(&panel, &book, &cfg).unwrap()
    };
    // gross = 0.6 + 0.4 = 1.0
    let at = mk(1.0);
    assert!(at.refusals.is_empty(), "gross exactly at the cap is permitted");
    assert!(at.units.iter().any(|u| *u != 0.0));
    let inside = mk(1.0 / (1.0 - 1e-9));
    assert!(inside.refusals.is_empty());
    let outside = mk(1.0 / (1.0 + 1e-9));
    assert_eq!(outside.refusals_with_code("gross_above_cap").len(), 8, "refused on every bar");
    assert!(outside.units.iter().all(|u| *u == 0.0), "hold-previous from flat: never partially clipped");
    assert!(outside.equity.iter().all(|e| *e == 1.0));
    // under Abort the same breach is an error with the numbers
    let book = Book::new(vec![SleeveSpec::from_rule(
        "s",
        FnRule::new(&["A", "B"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.6, -0.4])),
        vec![0, 1],
        ShareSpec::Fixed(1.0),
    )]);
    let mut cfg = BookConfig::default();
    cfg.sim.max_gross = Some(0.5);
    match simulate_book(&panel, &book, &cfg).unwrap_err() {
        BookError::Sim(SimError::MaxGrossBreached { gross, limit, .. }) => {
            assert!((gross - 1.0).abs() < 1e-15);
            assert_eq!(limit, 0.5);
        }
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------------------------------------ IndependentSubAccounts
#[test]
fn independent_sub_accounts_reproduce_the_legacy_sum_of_curves() {
    // Two sleeves on separate instruments, each enters once and holds; the legacy engine's portfolio is the SUM of the
    // two sub-account curves, each with capital share * total.
    let pa: Vec<f64> = (0..10).map(|i| 100.0 * (1.0 + 0.02 * i as f64 + 0.01 * ((i * i) % 5) as f64)).collect();
    let pb: Vec<f64> = (0..10).map(|i| 50.0 * (1.0 - 0.01 * i as f64 + 0.015 * ((i * 3) % 4) as f64)).collect();
    let panel = mini_panel("2020-01-01", 10, &[("A", false, pa.clone()), ("B", false, pb.clone())]);
    let book = Book::new(vec![
        SleeveSpec::from_rule(
            "a",
            once_rule(&["A"], vec![0.8], RebalancePolicy::OnDecision),
            vec![0],
            ShareSpec::Fixed(0.6),
        ),
        SleeveSpec::from_rule(
            "b",
            once_rule(&["B"], vec![1.0], RebalancePolicy::OnDecision),
            vec![1],
            ShareSpec::Fixed(0.4),
        ),
    ]);
    let mut cfg = hold_cfg();
    cfg.mode = AccountMode::IndependentSubAccounts;
    cfg.sim.initial_equity = 1000.0;
    let r = simulate_book(&panel, &book, &cfg).unwrap();
    for k in 0..10 {
        // legacy: sub-account a = 600 capital, 80% invested at bar 0 and held; b = 400, 100% invested
        let a = 600.0 * (0.2 + 0.8 * pa[k] / pa[0]);
        let b = 400.0 * (pb[k] / pb[0]);
        assert!((r.equity[k] - (a + b)).abs() <= 1e-9, "bar {k}: {} vs {}", r.equity[k], a + b);
    }
    assert!(r.attribution_report().max_abs_err_total <= 1e-12);
    // with a cost each sub-account pays its own: entry cost = rate * invested
    cfg.sim.cost = NET;
    let rc = simulate_book(&panel, &book, &cfg).unwrap();
    let rate = NET.rate();
    for k in 0..10 {
        let a = 600.0 * (1.0 - rate * 0.8) * (0.2 / (1.0 - rate * 0.8) + 0.8 * pa[k] / pa[0]) / 1.0;
        let a_closed = 600.0 * (0.2 + 0.8 * pa[k] / pa[0]) - 600.0 * rate * 0.8;
        let b_closed = 400.0 * (pb[k] / pb[0]) - 400.0 * rate * 1.0;
        let _ = a;
        assert!((rc.equity[k] - (a_closed + b_closed)).abs() <= 1e-9, "bar {k}");
    }
    // the JOINT account with EveryBar sleeves re-balances between sleeves, so it differs from the sum of drifting sub-accounts
    let every = |mode: AccountMode| {
        let book = Book::new(vec![
            SleeveSpec::from_rule(
                "a",
                const_rule(&["A"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
                vec![0],
                ShareSpec::Fixed(0.5),
            ),
            SleeveSpec::from_rule(
                "b",
                const_rule(&["B"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
                vec![1],
                ShareSpec::Fixed(0.5),
            ),
        ]);
        let mut cfg = hold_cfg();
        cfg.mode = mode;
        simulate_book(&panel, &book, &cfg).unwrap()
    };
    let joint = every(AccountMode::Joint);
    let indep = every(AccountMode::IndependentSubAccounts);
    // independent EveryBar-per-sleeve at 100% each = buy and hold of each sleeve's own capital
    for k in 0..10 {
        let want = 0.5 * pa[k] / pa[0] + 0.5 * pb[k] / pb[0];
        assert!((indep.equity[k] - want).abs() <= 1e-12, "bar {k}");
    }
    let gap = (0..10).map(|k| (joint.equity[k] - indep.equity[k]).abs()).fold(0.0, f64::max);
    assert!(gap > 1e-4, "the rebalancing return between sleeves is real and must not be called an error: {gap}");
}

#[test]
fn independent_sub_accounts_refuse_joint_only_features() {
    let panel = mini_panel("2020-01-01", 4, &[("A", false, vec![1.0, 2.0, 3.0, 4.0])]);
    let book = |a: AllocatorSpec| {
        let share =
            if a == AllocatorSpec::Fixed { ShareSpec::Fixed(1.0) } else { ShareSpec::Allocated { initial: 1.0 } };
        Book::new(vec![SleeveSpec::from_rule(
            "a",
            const_rule(&["A"], vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
            vec![0],
            share,
        )])
        .with_allocator(a)
    };
    for tweak in [
        Box::new(|c: &mut BookConfig| c.allocated_capital = Some(1.0)) as Box<dyn Fn(&mut BookConfig)>,
        Box::new(|c: &mut BookConfig| c.trade_filter = Some(TradeFilter { min_abs: 0.0, min_pct: 0.0 })),
        Box::new(|c: &mut BookConfig| c.sim.max_gross = Some(2.0)),
        Box::new(|c: &mut BookConfig| c.cadence = BookCadence::AllSleevesOnAnyDue),
        Box::new(|c: &mut BookConfig| c.cash_policy = CashPolicy::Budget),
    ] {
        let mut cfg = BookConfig::default();
        cfg.mode = AccountMode::IndependentSubAccounts;
        tweak(&mut cfg);
        assert!(matches!(simulate_book(&panel, &book(AllocatorSpec::Fixed), &cfg), Err(BookError::Unsupported(_))));
    }
    let mut cfg = BookConfig::default();
    cfg.mode = AccountMode::IndependentSubAccounts;
    assert!(matches!(
        simulate_book(&panel, &book(AllocatorSpec::InverseVol { lookback_bars: 3, total: 1.0 }), &cfg),
        Err(BookError::Unsupported(_))
    ));
}

// ------------------------------------------------------------------------------------------ overlay
struct Ladder {
    shrink_at: f64,
    recover_at: f64,
    halt_at: f64,
    log: Arc<Mutex<Vec<OverlayInput>>>,
}

struct LadderRun<'a> {
    cfg: &'a Ladder,
    peak: f64,
    shrunk: bool,
}

impl Overlay for Ladder {
    fn start(&self) -> Box<dyn OverlayRun + '_> {
        Box::new(LadderRun { cfg: self, peak: 0.0, shrunk: false })
    }
}

impl OverlayRun for LadderRun<'_> {
    fn step(&mut self, i: &OverlayInput) -> OverlayDecision {
        self.cfg.log.lock().unwrap().push(*i);
        self.peak = self.peak.max(i.equity).max(i.initial_equity);
        let dd = i.equity / self.peak - 1.0;
        let halt = dd <= -self.cfg.halt_at;
        if !self.shrunk && dd <= -self.cfg.shrink_at {
            self.shrunk = true;
        } else if self.shrunk && dd >= -self.cfg.recover_at {
            self.shrunk = false; // recovery hysteresis: back only above the (shallower) recovery level
        }
        OverlayDecision { scale: if self.shrunk { 0.5 } else { 1.0 }, halt }
    }
}

fn crash_panel(prices: &[f64]) -> BookPanel {
    mini_panel("2020-01-01", prices.len(), &[("A", false, prices.to_vec())])
}

fn full_invest_book(overlay: Option<Arc<dyn Overlay>>) -> Book {
    let mut b = Book::new(vec![SleeveSpec::from_rule(
        "a",
        const_rule(&["A"], vec![1.0], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
        vec![0],
        ShareSpec::Fixed(1.0),
    )]);
    b.overlay = overlay;
    b
}

#[test]
fn overlay_scales_risk_from_post_cost_equity_with_recovery_hysteresis() {
    let prices =
        [100.0, 100.0, 95.0, 90.0, 88.0, 91.0, 94.0, 97.0, 100.0, 104.0, 108.0, 112.0, 118.0, 124.0, 130.0, 136.0];
    let panel = crash_panel(&prices);
    let log = Arc::new(Mutex::new(Vec::new()));
    let ladder = Arc::new(Ladder { shrink_at: 0.08, recover_at: 0.03, halt_at: 0.5, log: log.clone() });
    let mut cfg = hold_cfg();
    cfg.sim.cost = NET;
    let r = simulate_book(&panel, &full_invest_book(Some(ladder)), &cfg).unwrap();
    let plain = simulate_book(&panel, &full_invest_book(None), &cfg).unwrap();
    // the overlay saw the simulated post-cost equity marked at each bar's close, before that bar's trading
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen.len(), prices.len());
    for k in 0..prices.len() {
        assert_eq!(seen[k].bar, k);
        assert_eq!(seen[k].equity, r.equity_pre[k], "overlay input equity at bar {k}");
        assert_eq!(seen[k].time, r.times[k]);
    }
    // Re-derive the ladder from the recorded equity only (an independent recomputation of the state machine).
    let mut peak = 1.0f64;
    let mut shrunk = false;
    let mut want_scale = Vec::new();
    let mut dds = Vec::new();
    for k in 0..prices.len() {
        peak = peak.max(r.equity_pre[k]);
        let dd = r.equity_pre[k] / peak - 1.0;
        if !shrunk && dd <= -0.08 {
            shrunk = true;
        } else if shrunk && dd >= -0.03 {
            shrunk = false;
        }
        want_scale.push(if shrunk { 0.5 } else { 1.0 });
        dds.push(dd);
    }
    assert_eq!(r.risk_scale, want_scale, "risk scale in force follows the ladder on post-cost equity");
    // the ladder really did shrink, sat in the hysteresis band, and released
    let first_shrunk = r.risk_scale.iter().position(|s| *s == 0.5).expect("the 12% drop shrinks the book");
    assert!(dds[first_shrunk] <= -0.08 && first_shrunk > 0 && r.risk_scale[first_shrunk - 1] == 1.0);
    assert!(
        (first_shrunk..prices.len()).any(|k| r.risk_scale[k] == 0.5 && dds[k] > -0.08 && dds[k] < -0.03),
        "hysteresis: still shrunk while the drawdown is between the shrink and the recovery levels: {dds:?} {:?}",
        r.risk_scale
    );
    let released = (first_shrunk..prices.len()).find(|&k| r.risk_scale[k] == 1.0).expect("released after the recovery");
    assert!(dds[released] >= -0.03);
    // the scale multiplies the constant risk scale: the target weight is scale * 1.0 on every bar
    for k in 0..prices.len() {
        assert_eq!(r.target_weights[k], r.risk_scale[k], "bar {k}");
    }
    assert!(r.equity.last().unwrap() != plain.equity.last().unwrap(), "the overlay changed the path");
    assert!(r.halted_at.is_none());
}

#[test]
fn an_overlay_halt_flattens_the_book_and_stays_flat() {
    let prices = [100.0, 100.0, 90.0, 80.0, 70.0, 100.0, 120.0, 130.0];
    let panel = crash_panel(&prices);
    let log = Arc::new(Mutex::new(Vec::new()));
    let ladder = Arc::new(Ladder { shrink_at: 0.99, recover_at: 0.0, halt_at: 0.25, log });
    let r = simulate_book(&panel, &full_invest_book(Some(ladder)), &hold_cfg()).unwrap();
    let h = r.halted_at.expect("the 25% drawdown halts the book");
    assert_eq!(h, 4, "halted at the bar whose marked equity is 30% below the peak");
    assert!(r.halted[h] && !r.halted[h - 1]);
    // flattened through the same trade path on the halt bar, and stays flat: the later rally is not earned
    assert_eq!(r.units[h], 0.0);
    for k in h..prices.len() {
        assert_eq!(r.units[k], 0.0);
        assert!(r.halted[k]);
    }
    for k in h + 1..prices.len() {
        assert_eq!(r.ret[k], 0.0, "no risk is counted after the halt");
    }
    assert!((r.equity[h] - 0.7).abs() < 1e-12);
    assert!(r.traded_notional[h] > 0.0, "the flattening trade");
}

// ------------------------------------------------------------------------------------------ stateful rules
/// A rule with memory: a hysteresis band. It goes long when the close rises above `hi` and flat when it falls below
/// `lo`; between the two it keeps its state. `poison_state_on_refusal` makes it mutate the state and then refuse.
#[derive(Clone)]
struct Hysteresis {
    hi: f64,
    lo: f64,
    refuse_at: Option<usize>,
}

impl StatefulRule for Hysteresis {
    type State = bool;
    fn id(&self) -> &'static str {
        "hysteresis"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["A"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn data_need(&self) -> DataNeed {
        DataNeed::CompleteJointCalendar
    }
    fn vol_scaling(&self) -> VolScaling {
        VolScaling::None
    }
    fn init(&self) -> bool {
        false
    }
    fn step(&self, st: &mut bool, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let c = h.last_close(0);
        if c > self.hi {
            *st = true;
        } else if c < self.lo {
            *st = false;
        }
        if self.refuse_at == Some(h.len()) {
            *st = !*st; // mutate, then refuse: the simulator must roll the state back
            return Err(RuleRefusal::data("bad_bar", "refused after touching the state"));
        }
        Ok(vec![if *st { 1.0 } else { 0.0 }])
    }
}

#[test]
fn a_stateful_rule_keeps_its_state_between_decisions_and_a_refusal_rolls_it_back() {
    let prices = [10.0, 12.0, 11.0, 9.0, 10.0, 11.0, 13.0, 12.0, 8.0, 10.0];
    let panel = crash_panel(&prices);
    let book = |refuse_at: Option<usize>| {
        Book::new(vec![SleeveSpec::from_stateful(
            "h",
            Hysteresis { hi: 11.5, lo: 9.5, refuse_at },
            vec![0],
            ShareSpec::Fixed(1.0),
        )])
    };
    let r = simulate_book(&panel, &book(None), &hold_cfg()).unwrap();
    // long at 12 (bar 1), stays long at 11 (between the bands), flat at 9, flat at 10, 11 (between), long at 13, long at 12, flat at 8, flat at 10
    let want = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let got: Vec<f64> = (0..prices.len()).map(|k| r.target_weights[k]).collect();
    assert_eq!(got, want, "hysteresis: the state persists inside the band");
    // Refused on bar 5 (h.len() == 6): the rule flips its state before refusing; the simulator restores it, so the
    // standing target is held AND the state is what it was: bar 6 (13 > hi) is long either way, but bar 5's flip must not leak.
    let refused = simulate_book(&panel, &book(Some(6)), &hold_cfg()).unwrap();
    assert!(refused.rule_refused[5]);
    assert_eq!(refused.target_weights[5], 0.0, "previous standing target held");
    // A refusal on bar 2 (in the band, state long): a leaked flip would make the rule flat at bar 3? bar 3 is 9 < lo: flat anyway.
    // Use bar 4 (10.0, state flat, between the bands): leak would turn the state long and bar 5 (11.0, in band) would emit long.
    let leak_probe = simulate_book(&panel, &book(Some(5)), &hold_cfg()).unwrap();
    assert!(leak_probe.rule_refused[4]);
    assert_eq!(
        leak_probe.target_weights[5], 0.0,
        "rolled back: bar 5 sees the state from bar 3, flat, not the flipped state"
    );
    // two runs of the same book do not share state (the simulator owns it)
    let b = book(None);
    let r1 = simulate_book(&panel, &b, &hold_cfg()).unwrap();
    let r2 = simulate_book(&panel, &b, &hold_cfg()).unwrap();
    assert_eq!(r1.series_sha256, r2.series_sha256);
    assert_eq!(r1.series_sha256, r.series_sha256);
}

#[test]
fn availability_masked_rules_and_double_vol_scaling_are_refused_not_approximated() {
    #[derive(Clone)]
    struct Masked(DataNeed, VolScaling);
    impl StatefulRule for Masked {
        type State = ();
        fn id(&self) -> &'static str {
            "masked"
        }
        fn impl_version(&self) -> String {
            "t".into()
        }
        fn universe(&self) -> &[&'static str] {
            &["A"]
        }
        fn decision_schedule(&self) -> DecisionSchedule {
            DecisionSchedule::Daily
        }
        fn rebalance_policy(&self) -> RebalancePolicy {
            RebalancePolicy::EveryBar
        }
        fn min_history_bars(&self) -> usize {
            1
        }
        fn data_need(&self) -> DataNeed {
            self.0
        }
        fn vol_scaling(&self) -> VolScaling {
            self.1
        }
        fn init(&self) {}
        fn step(&self, _: &mut (), _: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
            Ok(vec![0.5])
        }
    }
    let panel = crash_panel(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let masked = Book::new(vec![SleeveSpec::from_stateful(
        "m",
        Masked(DataNeed::AvailabilityMasked, VolScaling::None),
        vec![0],
        ShareSpec::Fixed(1.0),
    )]);
    assert!(matches!(simulate_book(&panel, &masked, &hold_cfg()), Err(BookError::Unsupported(_))));
    // an Internal-vol-scaling sleeve under the volatility-based allocator would be scaled twice
    let twice = Book::new(vec![SleeveSpec::from_stateful(
        "m",
        Masked(DataNeed::CompleteJointCalendar, VolScaling::Internal),
        vec![0],
        ShareSpec::Allocated { initial: 1.0 },
    )])
    .with_allocator(AllocatorSpec::InverseVol { lookback_bars: 3, total: 1.0 });
    match simulate_book(&panel, &twice, &hold_cfg()) {
        Err(BookError::BadBook(m)) => assert!(m.contains("twice")),
        other => panic!("{other:?}"),
    }
    // the same sleeve with fixed shares is fine
    let ok = Book::new(vec![SleeveSpec::from_stateful(
        "m",
        Masked(DataNeed::CompleteJointCalendar, VolScaling::Internal),
        vec![0],
        ShareSpec::Fixed(1.0),
    )]);
    assert!(simulate_book(&panel, &ok, &hold_cfg()).is_ok());
}

// ------------------------------------------------------------------------------------------ validation
#[test]
fn malformed_books_are_errors() {
    let panel = crash_panel(&[1.0, 2.0, 3.0, 4.0]);
    let rule = || const_rule(&["A"], vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar);
    let sl = |id: &str, share: ShareSpec| SleeveSpec::from_rule(id, rule(), vec![0], share);
    let run = |b: Book| simulate_book(&panel, &b, &hold_cfg());
    assert!(matches!(run(Book::new(vec![])), Err(BookError::BadBook(_))));
    assert!(matches!(
        run(Book::new(vec![sl("a", ShareSpec::Fixed(0.5)), sl("a", ShareSpec::Fixed(0.5))])),
        Err(BookError::BadBook(_))
    ));
    assert!(matches!(run(Book::new(vec![sl("", ShareSpec::Fixed(0.5))])), Err(BookError::BadBook(_))));
    assert!(matches!(run(Book::new(vec![sl("a", ShareSpec::Fixed(0.0))])), Err(BookError::BadBook(_))));
    assert!(matches!(run(Book::new(vec![sl("a", ShareSpec::Fixed(f64::NAN))])), Err(BookError::BadBook(_))));
    assert!(matches!(run(Book::new(vec![sl("a", ShareSpec::Allocated { initial: 1.0 })])), Err(BookError::BadBook(_))));
    assert!(matches!(
        run(Book::new(vec![sl("a", ShareSpec::Fixed(1.0))])
            .with_allocator(AllocatorSpec::InverseVol { lookback_bars: 3, total: 1.0 })),
        Err(BookError::BadBook(_))
    ));
    assert!(matches!(
        run(Book::new(vec![sl("a", ShareSpec::Allocated { initial: 1.0 })])
            .with_allocator(AllocatorSpec::InverseVol { lookback_bars: 1, total: 1.0 })),
        Err(BookError::BadBook(_))
    ));
    let bad_universe = SleeveSpec::from_rule("a", rule(), vec![0, 0], ShareSpec::Fixed(1.0));
    assert!(matches!(
        run(Book::new(vec![bad_universe])),
        Err(BookError::Sim(SimError::UniverseMismatch { .. })) | Err(BookError::BadBook(_))
    ));
    let oob = SleeveSpec::from_rule("a", rule(), vec![3], ShareSpec::Fixed(1.0));
    assert!(matches!(run(Book::new(vec![oob])), Err(BookError::BadBook(_))));
    let mismatch = SleeveSpec::from_rule(
        "a",
        const_rule(&["B"], vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
        vec![0],
        ShareSpec::Fixed(1.0),
    );
    assert!(matches!(run(Book::new(vec![mismatch])), Err(BookError::Sim(SimError::UniverseMismatch { .. }))));
    let mut cfg = hold_cfg();
    cfg.allocated_capital = Some(-1.0);
    assert!(simulate_book(&panel, &Book::new(vec![sl("a", ShareSpec::Fixed(1.0))]), &cfg).is_err());
    let mut cfg = hold_cfg();
    cfg.trade_filter = Some(TradeFilter { min_abs: -1.0, min_pct: 0.0 });
    assert!(simulate_book(&panel, &Book::new(vec![sl("a", ShareSpec::Fixed(1.0))]), &cfg).is_err());
    let mut cfg = hold_cfg();
    cfg.sim.risk_scale = f64::NAN;
    assert!(simulate_book(&panel, &Book::new(vec![sl("a", ShareSpec::Fixed(1.0))]), &cfg).is_err());
    // invalid weights from a rule are errors, never partially applied
    let nan_rule = FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![f64::NAN]));
    let e = simulate_book(
        &panel,
        &Book::new(vec![SleeveSpec::from_rule("a", nan_rule, vec![0], ShareSpec::Fixed(1.0))]),
        &hold_cfg(),
    )
    .unwrap_err();
    assert!(matches!(e, BookError::Sim(SimError::InvalidWeights { .. })));
    // budget cash policy is defined for long-only books
    let short = FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![-0.5]));
    let mut cfg = hold_cfg();
    cfg.cash_policy = CashPolicy::Budget;
    let e = simulate_book(
        &panel,
        &Book::new(vec![SleeveSpec::from_rule("a", short, vec![0], ShareSpec::Fixed(1.0))]),
        &cfg,
    )
    .unwrap_err();
    assert!(matches!(e, BookError::Construct(ConstructRefusal::BudgetNeedsLongOnly { .. })));
}

#[test]
fn book_metrics_use_the_account_clock_and_each_sleeve_its_own() {
    let (g, _) = run_case(&case("book_cert_60_40"));
    let m = g.metrics().unwrap();
    assert_eq!(g.metric_definitions, BOOK_METRIC_DEFINITIONS);
    // account clock: every calendar day is a bar => ~365 per year; the ETF sleeve alone on its own calendar ~252
    assert!(m.ppy > 360.0 && m.ppy < 370.0, "{}", m.ppy);
    let etf = g.sleeve_metrics(0, true).unwrap();
    let cry = g.sleeve_metrics(1, true).unwrap();
    assert!(etf.ppy > 240.0 && etf.ppy < 265.0, "own-calendar ETF ppy {}", etf.ppy);
    assert!(cry.ppy > 360.0 && cry.ppy < 370.0);
    // the ETF shadow has no carry rows: one row per ETF own bar after the first
    let own_bars = (0..g.n_bars()).filter(|&k| g.sleeve_open[k * 2]).count();
    assert_eq!(g.shadows[0].gross.len(), own_bars - 1);
    assert_eq!(g.shadows[1].gross.len(), g.n_bars() - 1);
    assert!(g.shadows[0].dates.windows(2).all(|w| w[0] < w[1]));
    // window: counted from the bar after the first fill
    assert_eq!(g.window.unwrap().first_bar, 1);
    assert_eq!(g.window_returns().len(), g.n_bars() - 1);
    assert_eq!(m.n, g.n_bars() - 1);
}

// ------------------------------------------------------------------------------------------ valuation and the PF2 boundary
#[test]
fn an_instrument_with_a_bar_is_still_carried_when_its_sleeve_is_not_open() {
    // E2 is closed (declared) on 2019-05-15 while E1 and E3 trade: the ETF sleeve has no own bar (its joint calendar needs
    // all three), nothing is refused (a declared closure is not a gap), and ALL three ETF instruments are valued at their
    // carried closes that bar (one valuation convention per sleeve bar, the key's); the next bar earns the full change.
    let panel = key_book_panel();
    let u = panel.times().iter().position(|t| t.date() == d("2019-05-15")).unwrap();
    let e2 = panel.instrument_index("E2").unwrap();
    let mut sessions = panel.sessions().to_vec();
    sessions[e2] =
        SessionKind::exchange("test_us", ETF_HOLIDAYS.iter().map(|s| d(s)).chain([d("2019-05-15")]).collect());
    let close: Vec<Vec<Option<f64>>> = (0..panel.n_instruments())
        .map(|i| {
            let mut c = panel.close(i).to_vec();
            if i == e2 {
                c[u] = None;
            }
            c
        })
        .collect();
    let holed = BookPanel::new(panel.instruments().to_vec(), sessions, panel.times().to_vec(), close).unwrap();
    assert_eq!(holed.availability(e2, u), Availability::Closed);
    let c = case("book_cert_60_40");
    let r = simulate_book(&holed, &book_for_case(&holed, &c), &config_for_case(&c)).unwrap();
    let k = (0..r.n_bars()).find(|&k| r.times[k].date() == d("2019-05-15")).unwrap();
    assert!(r.refusals.is_empty(), "a declared closure is not a gap");
    assert!(!r.sleeve_open[k * 2] && r.sleeve_open[k * 2 + 1]);
    let ni = r.n_instruments();
    let e1 = panel.instrument_index("E1").unwrap();
    assert!(panel.close(e1)[u].is_some(), "E1 does have a real bar that day");
    assert_eq!(r.marks[k * ni + e1], r.marks[(k - 1) * ni + e1], "E1 is carried because its sleeve is not open");
    assert_eq!(r.marks[(k + 1) * ni + e1], panel.close(e1)[u + 1].unwrap(), "the next bar is valued at the real close");
    assert!(r.attribution_report().max_abs_err_total <= 1e-12);
}

/// A construction that delegates to the minimal one with half the risk scale and counts its calls: the simulator
/// delegates ALL sizing to the `Construct` it is given (this is the PF2 boundary).
struct Halved {
    inner: MinimalConstruct,
    calls: Mutex<usize>,
    seen_planned: Mutex<usize>,
}

impl Construct for Halved {
    fn construct(&self, i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal> {
        *self.calls.lock().unwrap() += 1;
        *self.seen_planned.lock().unwrap() += i.planned.iter().filter(|p| **p).count();
        let p = ConstructPolicy { risk_scale: i.policy.risk_scale * 0.5, ..*i.policy };
        self.inner.construct(&ConstructInputs { policy: &p, ..*i })
    }
    fn inverse_vol_shares(&self, r: &[&[f64]], lookback: usize, total: f64) -> Option<Vec<f64>> {
        self.inner.inverse_vol_shares(r, lookback, total)
    }
}

#[test]
fn the_simulator_delegates_all_sizing_to_the_construct_it_is_given() {
    let c = case("book_cert_60_40");
    let (panel, book, cfg) = build_case(&c);
    // the minimal construction through the generic entry point IS `simulate_book`
    let a = simulate_book(&panel, &book, &cfg).unwrap();
    let b = simulate_book_with(&panel, &book, &cfg, &MinimalConstruct).unwrap();
    assert_eq!(a.series_sha256, b.series_sha256);
    // a construction with half the risk scale gives the run of a half risk scale, and it is called once per driven bar
    let halved = Halved { inner: MinimalConstruct, calls: Mutex::new(0), seen_planned: Mutex::new(0) };
    let h = simulate_book_with(&panel, &book, &cfg, &halved).unwrap();
    let mut cfg_half = cfg.clone();
    cfg_half.sim.risk_scale = 0.5;
    let h2 = simulate_book(&panel, &book, &cfg_half).unwrap();
    assert_eq!(
        h.units.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        h2.units.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        h.ret.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        h2.ret.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        *halved.calls.lock().unwrap(),
        a.run.iter().filter(|r| **r).count(),
        "one construct call per driven bar"
    );
    assert!(*halved.seen_planned.lock().unwrap() > 0);
    assert_ne!(a.ret, h.ret);
}

#[test]
fn contributions_of_a_shared_instrument_are_split_by_the_sleeves_standing_components() {
    // X is held by both sleeves with opposite signs: sleeve a's component is |0.5*wa_x|, sleeve b's |0.5*wb_x|; the position's
    // contribution is split in that proportion (never all to the first owner), and sleeve-specific instruments go to their sleeve.
    let panel = netting_panel();
    let (g, _) = simulate_book_gross_and_net(&panel, &netting_book(&panel), &netting_config()).unwrap();
    let ni = g.n_instruments();
    let (x, y, z) = (0usize, 1usize, 2usize);
    for k in 1..g.n_bars() {
        // the weights the sleeves held ENTERING bar k are the rules' answers at own bar k-1
        let i = k - 1;
        let wa: [f64; 2] = if (i / 7) % 2 == 0 { [0.6, 0.4] } else { [0.3, 0.7] };
        let wb: [f64; 2] = if (i / 5) % 3 != 1 { [-0.8, 0.2] } else { [0.5, -0.3] };
        let (ca, cb) = ((0.5 * wa[0]).abs(), (0.5 * wb[0]).abs());
        let r = |j: usize| g.marks[k * ni + j] / g.marks[(k - 1) * ni + j] - 1.0;
        let w_open = |j: usize| g.held_weights[(k - 1) * ni + j];
        let piece_x = w_open(x) * r(x);
        let want_a = w_open(y) * r(y) + piece_x * (ca / (ca + cb));
        let want_b = w_open(z) * r(z) + piece_x * (cb / (ca + cb));
        assert!(
            (g.contrib[k * 2] - want_a).abs() <= 1e-14,
            "bar {k}: contribution of a {} vs {want_a}",
            g.contrib[k * 2]
        );
        assert!(
            (g.contrib[k * 2 + 1] - want_b).abs() <= 1e-14,
            "bar {k}: contribution of b {} vs {want_b}",
            g.contrib[k * 2 + 1]
        );
    }
}
