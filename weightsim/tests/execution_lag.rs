//! Rule-level execution lag (`WeightRule::execution_lag_bars`, the tranche axis of a certified envelope): lag 0 is the
//! certified path bit for bit, a rule lag of `k` moves every fill `k` bars later exactly as
//! `SimConfig::execution_delay_bars = k` does, the two add, and a decision whose effective bar is past the end of the
//! panel is never effective.

mod common;

use common::*;
use weightsim::*;

/// Any rule plus a rule-level lag; every other fact forwards to the inner rule.
struct Lagged<R: WeightRule> {
    inner: R,
    lag: usize,
}

impl<R: WeightRule> WeightRule for Lagged<R> {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn impl_version(&self) -> String {
        self.inner.impl_version()
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.inner.decision_schedule()
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.inner.rebalance_policy()
    }
    fn execution_lag_bars(&self) -> usize {
        self.lag
    }
    fn min_history_bars(&self) -> usize {
        self.inner.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        self.inner.target_weights(h)
    }
}

fn net_cfg(base: SimConfig) -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..base }
}

/// The default lag (0) and an explicit lag of 0 are the same run as a rule without the method: same digest, same
/// units, with and without a configured delay, gross and net.
#[test]
fn lag_zero_is_the_certified_path_bit_for_bit() {
    let s1 = s1_panel();
    let a = simulate(&s1, &S1TestRule, &s1_config()).unwrap();
    let b = simulate(&s1, &Lagged { inner: S1TestRule, lag: 0 }, &s1_config()).unwrap();
    assert_eq!(a.series_sha256, b.series_sha256);
    assert_eq!(a.units, b.units);
    assert_eq!(a.window, b.window);
    let s3 = s3_panel();
    for cfg in [s3_config(), net_cfg(s3_config()), SimConfig { execution_delay_bars: 1, ..net_cfg(s3_config()) }] {
        let a = simulate(&s3, &S3TestRule, &cfg).unwrap();
        let b = simulate(&s3, &Lagged { inner: S3TestRule, lag: 0 }, &cfg).unwrap();
        assert_eq!(a.series_sha256, b.series_sha256);
        assert_eq!(a.units, b.units);
    }
}

/// A rule lag of `k` is the configured delay of `k`, bit for bit, and a rule lag plus a configured delay is their sum.
#[test]
fn rule_lag_k_equals_configured_delay_k_and_the_two_add() {
    let s3 = s3_panel();
    for k in 1..=3 {
        let via_rule = simulate(&s3, &Lagged { inner: S3TestRule, lag: k }, &net_cfg(s3_config())).unwrap();
        let via_cfg =
            simulate(&s3, &S3TestRule, &SimConfig { execution_delay_bars: k, ..net_cfg(s3_config()) }).unwrap();
        assert_eq!(via_rule.series_sha256, via_cfg.series_sha256, "lag {k}");
        assert_eq!(via_rule.units, via_cfg.units, "lag {k}");
        let cfg = SimConfig { execution_delay_bars: k - 1, ..net_cfg(s3_config()) };
        let mixed = simulate(&s3, &Lagged { inner: S3TestRule, lag: 1 }, &cfg).unwrap();
        assert_eq!(mixed.series_sha256, via_cfg.series_sha256, "delay {} + lag 1", k - 1);
    }
    let s1 = s1_panel();
    let via_rule = simulate(&s1, &Lagged { inner: S1TestRule, lag: 1 }, &s1_config()).unwrap();
    let via_cfg = simulate(&s1, &S1TestRule, &SimConfig { execution_delay_bars: 1, ..s1_config() }).unwrap();
    assert_eq!(via_rule.series_sha256, via_cfg.series_sha256);
    assert_ne!(via_rule.series_sha256, simulate(&s1, &S1TestRule, &s1_config()).unwrap().series_sha256);
}

/// On a tiny panel with a constant daily rule: the decision is taken at bar 0 whatever the lag, the first non-zero
/// units appear at bar `k` exactly, nothing is held and no target stands before that bar, and the counted window opens
/// the bar after.
#[test]
fn lag_k_shifts_the_first_fill_by_exactly_k_bars() {
    let panel = synth_panel(&["A", "B"], 40, 7, "2020-01-01");
    let n = panel.n_bars();
    let first_fill = |sim: &SimResult| (0..n).find(|&t| sim.row(&sim.units, t).iter().any(|u| *u != 0.0));
    for k in 0..=5 {
        let inner = const_rule(&["A", "B"], vec![0.5, 0.5], DecisionSchedule::Daily, RebalancePolicy::OnDecision);
        let sim = simulate(&panel, &Lagged { inner, lag: k }, &SimConfig::default()).unwrap();
        assert!(sim.decision[0], "lag {k}: the decision is still taken at bar 0");
        assert_eq!(first_fill(&sim), Some(k), "lag {k}");
        for t in 0..k {
            assert!(sim.row(&sim.units, t).iter().all(|u| *u == 0.0), "lag {k}: units held at bar {t}");
            assert_eq!(sim.row(&sim.target_weights, t), &[0.0, 0.0][..], "lag {k}: a target stands at bar {t}");
            assert_eq!(sim.row(&sim.held_weights, t), &[0.0, 0.0][..]);
        }
        assert_eq!(sim.row(&sim.target_weights, k), &[0.5, 0.5][..]);
        assert_eq!(sim.window.unwrap().first_bar, k + 1);
    }
}

/// A decision whose effective bar is past the end of the panel is simply never effective: no fill, no standing target,
/// no counted window, equity untouched.
#[test]
fn a_decision_past_the_end_of_the_panel_is_never_effective() {
    // Twelve weekdays from 2020-01-01 are all in January: one month-end decision, at the last bar (t = 11).
    let panel = synth_panel(&["A", "B"], 12, 3, "2020-01-01");
    let make = |lag| Lagged {
        inner: const_rule(&["A", "B"], vec![0.5, 0.5], DecisionSchedule::LastBarOfMonth, RebalancePolicy::OnDecision),
        lag,
    };
    let lagged = simulate(&panel, &make(1), &SimConfig::default()).unwrap();
    assert_eq!(lagged.decision.iter().filter(|d| **d).count(), 1);
    assert!(lagged.decision[11]);
    assert!(lagged.units.iter().all(|u| *u == 0.0));
    assert!(lagged.target_weights.iter().all(|w| *w == 0.0));
    assert_eq!(lagged.rebalance_bars, 0);
    assert!(lagged.window.is_none());
    assert!(lagged.equity.iter().all(|e| *e == 1.0));
    // With no lag the same decision fills at that last bar.
    let plain = simulate(&panel, &make(0), &SimConfig::default()).unwrap();
    assert!(plain.row(&plain.units, 11).iter().all(|u| *u != 0.0));
    assert_eq!(plain.rebalance_bars, 1);
}
