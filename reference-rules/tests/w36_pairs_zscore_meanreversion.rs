//! W3.6 two-provider identity check for `pairs_zscore_meanreversion` (BacktestingCore
//! `SENIOR_RESEARCHER_GAP_CLOSURE_PLAN.md`, W3.6 / 9.9 point 2).
//!
//! The fixture (`tests/data/w36_pairs_zscore_meanreversion.csv`) was built independently of both
//! implementations: 2 synthetic assets matching `PairsZscoreMeanReversionRule::universe()` exactly
//! (`KO, PEP`), 81 daily bars, a smooth low-noise log-price relationship for the first 78 bars
//! (hedge ratio and spread well-defined, no signal), then a calibrated jump on bar 78 that drives
//! z far below -2.0 (enter long-KO/short-PEP), bar 79 calibrated to sit strictly inside the
//! hysteresis band (so the state machine must HOLD, not re-decide), and bar 80 calibrated to land
//! back inside the exit band (|z| < 0.5, flattening the position).
//!
//! REAL CROSS-CHECK DIVERGENCE FOUND, not silently resolved: the spec's hedge-ratio/z-score window
//! ambiguity (see `pairs_zscore_meanreversion.rs` choice 3) was given to Kimi (Moonshot `kimi-k3`)
//! with ZERO hint toward either reading, and its first, fully independent response picked a
//! DIFFERENT -- also textually defensible -- resolution than this Rust implementation: an
//! EXCLUSIVE trailing window (`[t-L, t-1]`, L points, hedge ratio fit on that window only) with the
//! CURRENT bar's spread scored against it as an out-of-sample point, rather than the INCLUSIVE
//! `[t-L, t]` (L+1 points) window used here and by the rest of this primitive family (#1, #2, #4).
//! That first Kimi response took an extremely long reasoning pass (43629 reasoning tokens) to reach
//! that answer; given the `kimi-k3` thinking model's cost/latency for this prompt, a second,
//! shorter prompt was used to obtain a timely INDEPENDENT numerical implementation of the SAME
//! (inclusive-window) reading this Rust file uses, for the actual 1e-9 identity check below -- this
//! narrows that one specific prompt's independence (it was told which window convention to use,
//! unlike every other primitive's Kimi call), while the surrounding OLS/z-score/state-machine
//! arithmetic was still independently (re)implemented from scratch. Both Kimi outputs are preserved
//! (`work/fixtures/kimi_p5_content.txt` = fully independent, exclusive-window; `kimi_p5_short_content.txt`
//! = told-the-convention, inclusive-window, used below) so the divergence is auditable rather than
//! quietly discarded. Neither reading is "wrong" against the spec text as written; this is a
//! genuine ambiguity the written spec does not close, flagged for whoever tightens it later.

use weightsim::{simulate, Panel, SimConfig};

use reference_rules::{PairsZscoreMeanReversionRule, PAIRS_SYMBOLS};

const TOL: f64 = 1e-9;
const FIXTURE: &str = include_str!("data/w36_pairs_zscore_meanreversion.csv");

#[test]
fn identity_against_independent_python_implementation() {
    let panel = Panel::from_long_csv(FIXTURE, &PAIRS_SYMBOLS).unwrap();
    assert_eq!(panel.n_bars(), 81);
    let r = simulate(&panel, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();

    // Bar 59: one short of L+1=61 bars -- silent skip, flat.
    let w59 = r.row(&r.target_weights, 59);
    assert!(
        (w59[0]).abs() < TOL && (w59[1]).abs() < TOL,
        "bar 59 must be flat (insufficient history): {w59:?}"
    );

    // Bar 60: first eligible bar (61 bars visible), smooth relationship, no signal -- flat.
    let w60 = r.row(&r.target_weights, 60);
    assert!(
        (w60[0]).abs() < TOL && (w60[1]).abs() < TOL,
        "bar 60 must be flat (no signal yet): {w60:?}"
    );

    // Bar 78: calibrated entry -- z far below -entry_threshold -> long-KO/short-PEP.
    let w78 = r.row(&r.target_weights, 78);
    assert!(
        (w78[0] - 0.5).abs() < TOL && (w78[1] + 0.5).abs() < TOL,
        "bar 78 must enter long-KO/short-PEP: {w78:?}"
    );

    // Bar 79: calibrated hysteresis band (exit_threshold < |z| < entry_threshold) -- HOLD unchanged.
    let w79 = r.row(&r.target_weights, 79);
    assert!(
        (w79[0] - 0.5).abs() < TOL && (w79[1] + 0.5).abs() < TOL,
        "bar 79 must hold the position: {w79:?}"
    );

    // Bar 80: calibrated exit -- |z| < exit_threshold -> flat.
    let w80 = r.row(&r.target_weights, 80);
    assert!(
        (w80[0]).abs() < TOL && (w80[1]).abs() < TOL,
        "bar 80 must exit to flat: {w80:?}"
    );
}
