//! W3.6 two-provider identity check for `inverse_vol_risk_parity` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_inverse_vol_risk_parity.csv`) was built independently of both
//! implementations: 4 synthetic assets matching `InverseVolRiskParityRule::universe()` exactly
//! (`SPY, EFA, AGG, DBC`), 70 daily bars, each asset's simple daily return alternating a fixed
//! +amp/-amp (no asset has zero stdev, deliberately -- see below) so the trailing-60 population
//! stdev is a clean, reproducible number. The expected weights below are the values an
//! independent Kimi (Moonshot `kimi-k3`) Python implementation of the SAME verbatim spec produced
//! on this exact fixture, which agreed with a by-hand Python ground truth to machine precision;
//! both the Rust rule under test here and that external implementation must agree to the 1e-9
//! tolerance used by this codebase's other identity checks.
//!
//! Deliberately excludes a zero-stdev asset: the spec is silent on how to treat exactly-zero
//! trailing volatility (as opposed to insufficient history), and the two providers in fact
//! resolved it differently (Rust refuses the whole decision as degenerate; the independent Kimi
//! implementation of this same primitive silently excludes the zero-vol asset instead, which
//! also disagrees with Kimi's OWN primitive-1 implementation, which gave zero-vol assets ALL the
//! weight). This three-way disagreement is a real, flagged spec gap, not a bug in any one
//! implementation -- the fixture below sidesteps it so the identity test is not contaminated by
//! an orthogonal ambiguity.

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{InverseVolRiskParityRule, INVERSE_VOL_RISK_PARITY_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_inverse_vol_risk_parity.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &INVERSE_VOL_RISK_PARITY_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 70);
    let r = simulate(&panel, &InverseVolRiskParityRule, &SimConfig::default()).unwrap();
    let got = r.row(&r.target_weights, panel.n_bars() - 1);

    // symbol order is INVERSE_VOL_RISK_PARITY_SYMBOLS = [SPY, EFA, AGG, DBC]
    let want = [0.26086956521738974_f64, 0.08695652173912988, 0.5217391304347856, 0.13043478260869482];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < TOL,
            "asset {i} ({}): got {g}, want {w}",
            INVERSE_VOL_RISK_PARITY_SYMBOLS[i]
        );
    }
    let sum: f64 = got.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "weights must sum to 1.0, got {sum}");
}
