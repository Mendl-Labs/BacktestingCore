//! W3.6 two-provider identity check for `inverse_volatility_weight` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_inverse_volatility_weight.csv`) was built independently of both
//! implementations: 5 synthetic assets matching `InverseVolatilityWeightRule::universe()` exactly
//! (`SPY, IWM, EFA, TLT, GLD`), 30 daily bars, each asset's simple daily return alternating a fixed
//! +amp/-amp so the trailing-20 population stdev is a clean, reproducible number. The expected
//! weights below are the values an independent Kimi (Moonshot `kimi-k3`) Python implementation of
//! the SAME verbatim spec produced on this exact fixture, which in turn agreed with a by-hand
//! Python ground truth to machine precision (~1e-15); both the Rust rule under test here and that
//! external implementation must agree to the 1e-9 tolerance used by this codebase's other identity
//! checks (e.g. `weightsim/tests/answer_key.rs`, `book_key.rs`).

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{InverseVolatilityWeightRule, INVERSE_VOLATILITY_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_inverse_volatility_weight.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &INVERSE_VOLATILITY_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 30);
    let r = simulate(&panel, &InverseVolatilityWeightRule, &SimConfig::default()).unwrap();
    let got = r.row(&r.target_weights, panel.n_bars() - 1);

    // symbol order is INVERSE_VOLATILITY_SYMBOLS = [SPY, IWM, EFA, TLT, GLD]
    let want = [
        0.3448275862068962_f64,
        0.17241379310344807,
        0.11494252873563197,
        0.13793103448275867,
        0.22988505747126514,
    ];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < TOL, "asset {i} ({}): got {g}, want {w}", INVERSE_VOLATILITY_SYMBOLS[i]);
    }
    let sum: f64 = got.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "weights must sum to 1.0, got {sum}");
}
