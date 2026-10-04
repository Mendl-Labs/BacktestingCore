//! W3.6 two-provider identity check for `equal_weight_rebalance` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_equal_weight_rebalance.csv`) was built independently of both
//! implementations: 4 synthetic assets matching `EqualWeightRebalanceRule::universe()` exactly
//! (`SPY, AGG, GLD, VNQ`), 70 daily bars from 2024-01-01 to 2024-03-10, each asset compounding at
//! its own fixed daily growth factor (SPY flat, AGG +0.1%/day, GLD -0.1%/day, VNQ +0.2%/day) so
//! drift between decisions is deterministic and easy to recompute independently.
//!
//! The rule's own target is the content-free constant `1/N` (choice 1 in
//! `equal_weight_rebalance.rs`); the actual content under test here is the SCHEDULE/DRIFT
//! behavior: `DecisionSchedule::LastBarOfMonth` + `RebalancePolicy::OnDecision` resolve the spec's
//! "first trading day of month" gap (there is no such `DecisionSchedule` variant) to "last trading
//! day of month" instead (documented, off by exactly one session, never affecting WHAT is traded
//! since the target is session-independent). The expected `held_weights` below come from an
//! independent Kimi (Moonshot `kimi-k3`) Python implementation of the verbatim spec
//! (`work/fixtures/kimi_p3_content.txt`), mechanically re-pointed from "first trading day of
//! month" to "last trading day of month" to match the Rust side's resolution of that same
//! documented gap (the drift arithmetic itself -- `held[t] = held[t-1]*(1+r) / sum(...)` -- is
//! untouched Kimi code); both agree with an independent by-hand Python ground truth
//! (`work/fixtures/gen_p3.py`) to machine precision. Tolerance 1e-9, matching this codebase's
//! other identity checks.

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{EqualWeightRebalanceRule, EQUAL_WEIGHT_REBALANCE_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_equal_weight_rebalance.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &EQUAL_WEIGHT_REBALANCE_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 70);
    let r = simulate(&panel, &EqualWeightRebalanceRule, &SimConfig::default()).unwrap();

    // Before the first decision (bar 30, 2024-01-31, last bar of January): the book has no position yet, so
    // held weights are all 0.0 (no position invented ahead of the rule's first successful decision).
    let held_20 = r.row(&r.held_weights, 20);
    for &w in held_20 {
        assert!(w.abs() < TOL, "pre-decision held weight must be 0.0, got {w}");
    }

    // Bar 30: first decision/trade. Held weights snap to exactly 1/N (all four assets priced equally at the
    // moment of the trade).
    let held_30 = r.row(&r.held_weights, 30);
    for &w in held_30 {
        assert!((w - 0.25).abs() < TOL, "held weight at the first decision must be 0.25, got {w}");
    }

    // Bar 45 (2024-02-15), strictly between the Jan-31 and Feb-29 decisions: weights have drifted away from
    // 1/N by each asset's own compounding since the bar-30 trade. Values below are the independently-computed
    // (Kimi + by-hand) ground truth, symbol order [SPY, AGG, GLD, VNQ].
    let held_45 = r.row(&r.held_weights, 45);
    let want_45 = [0.2480999473522632_f64, 0.2518476102818976, 0.24440438608888776, 0.25564805627695153];
    for (i, (&g, &w)) in held_45.iter().zip(want_45.iter()).enumerate() {
        assert!((g - w).abs() < TOL, "asset {i} ({}) at bar 45: got {g}, want {w}", EQUAL_WEIGHT_REBALANCE_SYMBOLS[i]);
    }
    let sum_45: f64 = held_45.iter().sum();
    assert!((sum_45 - 1.0).abs() < TOL, "held weights must still sum to 1.0 while drifting, got {sum_45}");

    // Bar 59 (2024-02-29, last bar of February) and bar 69 (2024-03-10, panel's final bar) are both decisions:
    // the book snaps back to exactly 1/N at each.
    for &t in &[59usize, 69usize] {
        let held = r.row(&r.held_weights, t);
        for &w in held {
            assert!((w - 0.25).abs() < TOL, "held weight at decision bar {t} must be 0.25, got {w}");
        }
    }
}
