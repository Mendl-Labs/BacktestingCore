//! W3.6 two-provider identity check for `low_vol_quintile_tilt` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_low_vol_quintile_tilt.csv`) was built independently of both
//! implementations: 7 synthetic assets matching `LowVolQuintileTiltRule::universe()` exactly
//! (`DIA, EEM, EFA, GLD, IWM, SPY, TLT`), 30 daily bars. DIA and EEM are deliberately given the
//! EXACT same trailing stdev (tied for lowest), so this fixture also exercises the spec's
//! alphabetical tie-break: DIA must win the sole bottom-quintile slot (`quintile_count(7) == 1`),
//! never EEM. The expected weights below are what an independent Kimi (Moonshot `kimi-k3`) Python
//! implementation of the SAME verbatim spec produced on this exact fixture, which agreed with a
//! by-hand Python ground truth to machine precision; both the Rust rule under test here and that
//! external implementation must agree to the 1e-9 tolerance used by this codebase's other
//! identity checks (e.g. `weightsim/tests/answer_key.rs`, `book_key.rs`).

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{LowVolQuintileTiltRule, LOW_VOL_QUINTILE_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_low_vol_quintile_tilt.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &LOW_VOL_QUINTILE_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 30);
    let r = simulate(&panel, &LowVolQuintileTiltRule, &SimConfig::default()).unwrap();
    let got = r.row(&r.target_weights, panel.n_bars() - 1);

    // symbol order is LOW_VOL_QUINTILE_SYMBOLS = [DIA, EEM, EFA, GLD, IWM, SPY, TLT]
    // DIA wins the DIA/EEM tie alphabetically; it is the sole bottom-quintile holding (weight 1.0).
    let want = [1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < TOL, "asset {i} ({}): got {g}, want {w}", LOW_VOL_QUINTILE_SYMBOLS[i]);
    }
}
