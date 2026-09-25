//! Always-on unit tests of the ladder's run plumbing (`ladder::runner`) and the `FlatUntil` wrapper, on tiny scripted
//! rules over a hand-made price panel, so every alignment convention is pinned by an exact expectation.

mod common;

use common::*;
use weightsim::*;
use weightsim_rules::ladder::fixtures::{KeyBar, SleeveKey};
use weightsim_rules::ladder::runner::{entry_date, flips_by_key_convention, key_rows, rows_from_sim, Basis};
use weightsim_rules::{CryptoTrendRule, FlatUntil};

/// Weights by bar index, Daily schedule.
struct Script {
    weights: Vec<Vec<f64>>,
    policy: RebalancePolicy,
}

impl WeightRule for Script {
    fn id(&self) -> &'static str {
        "script"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["A", "B"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.policy
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        Ok(self.weights[h.len() - 1].clone())
    }
}

fn panel(n: usize) -> Panel {
    let dates: Vec<Date> = (0..n).map(|i| Date::from_days_since_epoch(18_000 + i as i64)).collect();
    let a: Vec<f64> = (0..n).map(|i| 100.0 * (1.0 + 0.01 * ((i * 5) % 7) as f64)).collect();
    let b: Vec<f64> = (0..n).map(|i| 50.0 * (1.0 + 0.02 * ((i * 3) % 5) as f64)).collect();
    Panel::new(vec!["A".into(), "B".into()], dates, vec![a, b]).unwrap()
}

fn script(n: usize, f: impl Fn(usize) -> [f64; 2]) -> Script {
    Script { weights: (0..n).map(|i| f(i).to_vec()).collect(), policy: RebalancePolicy::EveryBar }
}

#[test]
fn rows_are_aligned_to_the_key_convention_ret_at_t_weights_at_t_minus_one() {
    let n = 12;
    let p = panel(n);
    // decisions from bar 2 on: weights (0.5,0) on bars 2..5, (0.25,0.25) after
    let rule = script(n, |i| {
        if i < 2 {
            [0.0, 0.0]
        } else if i < 6 {
            [0.5, 0.0]
        } else {
            [0.25, 0.25]
        }
    });
    let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
    let sim = simulate(&p, &rule, &cfg).unwrap();
    let rows = rows_from_sim(&sim, p.dates()[0], p.dates()[n - 1], true).unwrap();
    // the first decision (bar 0 succeeds with zeros: min_history 1) => window opens at bar 1
    assert_eq!(rows.dates[0], p.dates()[1]);
    assert_eq!(rows.dates.len(), n - 1);
    for (k, i) in (1..n).enumerate() {
        assert_eq!(rows.dates[k], p.dates()[i]);
        assert_eq!(rows.ret[k], sim.ret[i]);
        assert_eq!(rows.equity.as_ref().unwrap()[k], sim.equity[i]);
        // weights are those in force DURING bar i: standing target and held weights after bar i-1
        assert_eq!(rows.w_target.as_ref().unwrap()[k], sim.row(&sim.target_weights, i - 1).to_vec());
        assert_eq!(rows.w_held.as_ref().unwrap()[k], sim.row(&sim.held_weights, i - 1).to_vec());
        // cost and turnover are fractions of the PRE-cost equity
        let pre = sim.equity[i] + sim.cost[i];
        assert_eq!(rows.cost.as_ref().unwrap()[k], sim.cost[i] / pre);
        assert_eq!(rows.traded.as_ref().unwrap()[k], sim.traded_notional[i] / pre);
    }
    // a concrete number: at bar 3 the standing target is the bar-2 decision (0.5, 0)
    assert_eq!(rows.w_target.as_ref().unwrap()[2], vec![0.5, 0.0]);
    // costs really occurred on the rebalance bars
    assert!(rows.cost.as_ref().unwrap().iter().any(|&c| c > 0.0));
    // without `full` the equity/cost/traded columns are absent
    let lean = rows_from_sim(&sim, p.dates()[0], p.dates()[n - 1], false).unwrap();
    assert!(lean.equity.is_none() && lean.cost.is_none() && lean.traded.is_none());
    assert_eq!(lean.ret, rows.ret);
}

#[test]
fn the_window_opens_after_the_first_decision_not_after_the_first_fill() {
    let n = 12;
    let p = panel(n);
    let rule = script(n, |i| if i < 3 { [0.0, 0.0] } else { [0.5, 0.5] });
    // one bar of execution delay: first decision at bar 0 (zeros), first non-zero fill at bar 4
    let cfg = SimConfig { execution_delay_bars: 1, ..SimConfig::default() };
    let sim = simulate(&p, &rule, &cfg).unwrap();
    assert_eq!(sim.window.unwrap().first_bar, 2, "the simulator's own window opens after the first FILL (bar 1)");
    let rows = rows_from_sim(&sim, p.dates()[0], p.dates()[n - 1], false).unwrap();
    assert_eq!(rows.dates[0], p.dates()[1], "the ladder's rows open after the first DECISION (bar 0)");
    assert_eq!(rows.ret[0], 0.0, "flat bar");
    // an explicit start later than that is honoured, an earlier end truncates
    let rows = rows_from_sim(&sim, p.dates()[5], p.dates()[8], false).unwrap();
    assert_eq!((rows.dates[0], *rows.dates.last().unwrap()), (p.dates()[5], p.dates()[8]));
    assert_eq!(rows.dates.len(), 4);
    // a start after the end, or before any data but with no decision, are errors
    assert!(rows_from_sim(&sim, p.dates()[9], p.dates()[3], false).is_err());
    let never = Script { weights: vec![vec![0.0, 0.0]; n], policy: RebalancePolicy::EveryBar };
    let cfg = SimConfig { execution_delay_bars: 20, ..SimConfig::default() };
    let s = simulate(&p, &never, &cfg).unwrap();
    // decisions happened (zeros) but no fill ever did: rows still exist, opening after the first decision
    assert!(rows_from_sim(&s, p.dates()[0], p.dates()[n - 1], false).is_ok());
    assert!(rows_from_sim(&s, Date::from_days_since_epoch(30_000), Date::from_days_since_epoch(30_010), false).is_err());
}

#[test]
fn flips_follow_the_keys_convention_baseline_is_the_last_decision_before_the_window() {
    let n = 10;
    let p = panel(n);
    // per-asset signs by decision bar: A: 0,0,1,1,0,1,1,1,0,0 ; B: 0,1,1,0,0,0,1,1,1,1
    let a = [0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0];
    let b = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5];
    let rule = script(n, |i| [a[i], b[i]]);
    let sim = simulate(&p, &rule, &SimConfig::default()).unwrap();
    let d = |i: usize| p.dates()[i];
    // sign changes between successive decisions: A at 2,4,5,8 (4); B at 1,3,6 (3)
    assert_eq!(flips_by_key_convention(&sim, d(0), d(9)), 7);
    // window opens at bar 4: baseline = the decision at bar 3 (last one dated before it); changes after: A 4,5,8; B 6
    assert_eq!(flips_by_key_convention(&sim, d(4), d(9)), 4);
    // the baseline decision is NOT itself counted as a flip: window opens at bar 2: baseline bar 1 (A 0, B 1);
    // changes: A 2,4,5,8 (4) and B 3,6 (2)
    assert_eq!(flips_by_key_convention(&sim, d(2), d(9)), 6);
    // the end date is inclusive and cuts later decisions: up to bar 5 from baseline bar 3: A 4,5 ; B none
    assert_eq!(flips_by_key_convention(&sim, d(4), d(5)), 2);
    assert_eq!(flips_by_key_convention(&sim, d(4), d(4)), 1);
    // a window opening before the first decision baselines on the first decision
    assert_eq!(flips_by_key_convention(&sim, d(0), d(0)), 0);
}

fn fake_key(dates: &[Date]) -> SleeveKey {
    SleeveKey {
        code: "S3",
        rule_id: "crypto_trend_100d",
        symbols: vec!["BTC".into(), "ETH".into()],
        bars: dates
            .iter()
            .map(|&date| KeyBar {
                date,
                ret_gross: 0.0,
                ret_net: 0.0,
                equity_gross: 1.0,
                equity_net: 1.0,
                cost: 0.0,
                turnover: 0.0,
                decision: true,
                w_target: vec![0.0, 0.0],
                w_held: vec![0.0, 0.0],
            })
            .collect(),
        flips: 0,
    }
}

#[test]
fn the_entry_bar_is_the_panel_bar_before_the_keys_first_bar() {
    let p = panel(10);
    let key = fake_key(&p.dates()[4..8]);
    assert_eq!(entry_date(&p, &key).unwrap(), p.dates()[3]);
    // gaps in the panel calendar do not matter: it is the previous PANEL bar, not the previous calendar day
    let dates = vec![d("2020-01-02"), d("2020-01-06"), d("2020-01-07")];
    let gappy = Panel::new(vec!["A".into()], dates.clone(), vec![vec![1.0, 2.0, 3.0]]).unwrap();
    assert_eq!(entry_date(&gappy, &fake_key(&dates[1..])).unwrap(), d("2020-01-02"));
    // a key that starts on the first bar of the panel has no entry bar
    assert!(entry_date(&p, &fake_key(&p.dates()[0..4])).is_err());
}

#[test]
fn key_rows_offer_the_columns_the_key_defines_for_each_basis() {
    let p = panel(6);
    let mut key = fake_key(&p.dates()[1..5]);
    key.bars[2].ret_net = 0.5;
    key.bars[2].cost = 0.25;
    key.bars[2].turnover = 0.75;
    let g = key_rows(&key, Basis::Gross);
    let n = key_rows(&key, Basis::Net);
    assert_eq!(g.ret[2], 0.0);
    assert_eq!(n.ret[2], 0.5);
    // gross basis: no cost, no turnover comparison (the key's turnover is the NET run's), held weights present
    assert_eq!(g.cost.as_ref().unwrap(), &vec![0.0; 4]);
    assert!(g.traded.is_none() && g.w_held.is_some());
    // net basis: cost and turnover from the key, no held weights (the key's are the gross run's)
    assert_eq!(n.cost.as_ref().unwrap()[2], 0.25);
    assert_eq!(n.traded.as_ref().unwrap()[2], 0.75);
    assert!(n.w_held.is_none() && n.w_target.is_some());
    assert_eq!(g.dates, n.dates);
    assert_eq!(Basis::Gross.name(), "gross");
    assert_eq!(Basis::Net.name(), "net");
}

#[test]
fn flat_until_holds_cash_before_the_entry_bar_and_delegates_after() {
    let cry = s3_panel();
    let entry = cry.dates()[250];
    let gated = FlatUntil::new(CryptoTrendRule, entry);
    assert_eq!(gated.first_trade(), entry);
    assert_eq!(gated.id(), "crypto_trend_100d");
    assert_eq!(gated.universe(), CryptoTrendRule.universe());
    assert_eq!(gated.decision_schedule(), CryptoTrendRule.decision_schedule());
    assert_eq!(gated.rebalance_policy(), CryptoTrendRule.rebalance_policy());
    assert_eq!(gated.min_history_bars(), CryptoTrendRule.min_history_bars());
    assert_eq!(gated.declared_parameters(), CryptoTrendRule.declared_parameters());
    assert!(gated.impl_version().contains(&entry.to_string()));
    let plain = simulate(&cry, &CryptoTrendRule, &SimConfig::default()).unwrap();
    let flat = simulate(&cry, &gated, &SimConfig::default()).unwrap();
    // before the entry: all-zero targets, no positions, no cost; equity stays 1
    let k = 2;
    for t in 0..250 {
        assert!(flat.row(&flat.target_weights, t).iter().all(|&w| w == 0.0), "bar {t}");
        assert!(flat.row(&flat.units, t).iter().all(|&u| u == 0.0));
        assert_eq!(flat.equity[t], 1.0);
    }
    // from the entry: identical decisions to the plain rule
    for t in 250..cry.n_bars() {
        assert_eq!(flat.row(&flat.target_weights, t), plain.row(&plain.target_weights, t), "bar {t}");
    }
    assert!(plain.row(&plain.target_weights, 200).iter().any(|&w| w > 0.0) || k == 2, "fixture sanity");
    // Entering costs: the first fill pays 10 bps of the traded notional, which is exactly the invested weight
    let net = simulate(
        &cry,
        &gated,
        &SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() },
    )
    .unwrap();
    let invested: f64 = net.row(&net.target_weights, 250).iter().sum();
    assert!(invested > 0.0, "the fixture should be invested at the entry bar");
    assert!((net.cost[250] - 0.001 * invested).abs() <= 1e-15, "entry cost {} for weight {invested}", net.cost[250]);
    assert!((net.equity[250] - (1.0 - 0.001 * invested)).abs() <= 1e-15);
    // The gated rule is causal (the entry date is a date, not a price).
    let r = weightsim::harness::check_poisoning(
        &|_p: &Panel| FlatUntil::new(CryptoTrendRule, entry),
        &cry,
        &SimConfig::default(),
        300,
        3,
    )
    .unwrap();
    assert!(r.is_clean());
}
