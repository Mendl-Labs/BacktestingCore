//! W3.6 two-provider identity check for `cointegration_spread_threshold` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_cointegration_spread_threshold.csv`) was built independently of
//! both implementations: 2 synthetic assets matching `CointegrationSpreadThresholdRule::universe()`
//! exactly (`EWA, EWC`), 105 daily bars, a smooth low-noise log-price relationship through bar 101
//! (hedge ratio and spread well-defined, no signal, all within the FIRST refresh cycle `[90, 179]`
//! so the cross-cycle "hedge ratio held fixed" behavior -- already proven by the Rust side's own
//! `hedge_ratio_is_fixed_across_a_full_cycle_not_recomputed_every_bar` test -- is not what this
//! fixture is exercising), then a calibrated jump on bar 102 driving z far below -2.0 (enter
//! long-EWA/short-EWC), bar 103 calibrated inside the hysteresis band (HOLD), and bar 104
//! calibrated back inside the exit band (|z| < 0.5, flattening).
//!
//! REAL CROSS-CHECK DIVERGENCE FOUND (same shape as primitive #5's), not silently resolved: the
//! spec explicitly pins down the HEDGE-RATIO window (`[t-L, t-1]`, strictly excluding the refresh
//! bar) -- and both an independent Kimi (Moonshot `kimi-k3`) response and this Rust implementation
//! agree on that part exactly, down to the refresh-cadence formula. But the spec does NOT separately
//! pin down the Z-SCORE reference window once the ratio is fixed, and the two sides again diverged:
//! this Rust implementation uses an INCLUSIVE window `[s-L+1, s]` (matching #1/#2/#4's "trailing L
//! bars ending at the current bar" convention and #5's own inclusive choice), while Kimi's fully
//! independent response (`work/fixtures/kimi_p6_content.txt`) used an EXCLUSIVE window `[t-L, t-1]`
//! with bar t scored as an out-of-sample point -- the same reading it picked independently for #5.
//! For the actual 1e-9 identity check below, Kimi's own code was mechanically adapted (ONLY the
//! window-slicing changed; the refresh-schedule and state-machine logic is untouched Kimi code) to
//! the inclusive convention, exactly as primitive #3's DecisionSchedule gap and primitive #5's
//! window gap were handled (`work/fixtures/kimi_p6_adapted_impl.py`). Neither window reading is
//! wrong against the spec text; this is a second instance of the same genuine, flagged ambiguity.

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{CointegrationSpreadThresholdRule, COINT_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_cointegration_spread_threshold.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &COINT_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 105);
    let r = simulate(&panel, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();

    // Bar 89: one short of L+1=91 bars -- silent skip, flat.
    let w89 = r.row(&r.target_weights, 89);
    assert!(w89[0].abs() < TOL && w89[1].abs() < TOL, "bar 89 must be flat (insufficient history): {w89:?}");

    // Bar 90: first eligible bar (91 bars visible), smooth relationship, no signal -- flat.
    let w90 = r.row(&r.target_weights, 90);
    assert!(w90[0].abs() < TOL && w90[1].abs() < TOL, "bar 90 must be flat (no signal yet): {w90:?}");

    // Bar 102: calibrated entry -- z far below -entry_threshold -> long-EWA/short-EWC.
    let w102 = r.row(&r.target_weights, 102);
    assert!((w102[0] - 0.5).abs() < TOL && (w102[1] + 0.5).abs() < TOL, "bar 102 must enter long-EWA/short-EWC: {w102:?}");

    // Bar 103: calibrated hysteresis band -- HOLD unchanged.
    let w103 = r.row(&r.target_weights, 103);
    assert!((w103[0] - 0.5).abs() < TOL && (w103[1] + 0.5).abs() < TOL, "bar 103 must hold the position: {w103:?}");

    // Bar 104: calibrated exit -- |z| < exit_threshold -> flat.
    let w104 = r.row(&r.target_weights, 104);
    assert!(w104[0].abs() < TOL && w104[1].abs() < TOL, "bar 104 must exit to flat: {w104:?}");
}
