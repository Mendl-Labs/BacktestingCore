//! Two-provider cross-check for `donchian_breakout` (W3.6, Range/volume family).
//!
//! The independent Python implementation (Kimi/Moonshot `kimi-k3`, zero shared context with the Rust author) was
//! given the same verbatim spec plus an explicit resolution of the one genuine ambiguity it raises (whether the
//! "silent skip" region's 0.0 is a real, inheritable decided weight that seeds the stateful hold-chain, or a
//! separate "undecided" marker) -- both sides independently landed on treating it as a real, inheritable 0.0 (the
//! Rust author's module docs, choice 4, reach the identical conclusion for the identical reason: there is no
//! operational difference between "no opinion yet" and "decided to be flat"). On six fixtures covering a clean
//! breakout up, a breakout down after holding at 1, a hold at 1, a hold at 0, an off-by-one window-boundary check
//! (an outlier exactly N+1 bars back must not leak into the channel), and the insufficient-history skip region,
//! the two implementations agree exactly -- no correction round was needed. Every expected value is an exact 1.0
//! or 0.0, so "agreement to 1e-9" is exact equality.

use chrono::NaiveDate;
use reference_rules::{decide_donchian_breakout, HlcSeries};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn series(bars: &[(NaiveDate, f64, f64, f64)]) -> HlcSeries {
    let dates = bars.iter().map(|b| b.0).collect();
    let highs = bars.iter().map(|b| b.1).collect();
    let lows = bars.iter().map(|b| b.2).collect();
    let closes = bars.iter().map(|b| b.3).collect();
    HlcSeries::new("XCHK", dates, highs, lows, closes).unwrap()
}

/// Cross-checked against the Python key (see module doc): a clean breakout up, n=2.
#[test]
fn breakout_up_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 12.0, 4.0, 11.0),
    ]);
    let got = decide_donchian_breakout(&s, 2);
    let want = [0.0, 0.0, 1.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "bar {i}: got {g}, want {w}");
    }
}

/// Cross-checked: breakout down after holding at 1, n=2.
#[test]
fn breakout_down_after_hold_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 12.0, 4.0, 11.0),
        (d(2024, 1, 4), 5.0, 2.0, 3.0),
    ]);
    let got = decide_donchian_breakout(&s, 2);
    let want = [0.0, 0.0, 1.0, 0.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "bar {i}: got {g}, want {w}");
    }
}

/// Cross-checked: hold at 1 between channel bounds, n=2.
#[test]
fn hold_at_one_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 12.0, 4.0, 11.0),
        (d(2024, 1, 4), 10.0, 6.0, 8.0),
    ]);
    let got = decide_donchian_breakout(&s, 2);
    let want = [0.0, 0.0, 1.0, 1.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "bar {i}: got {g}, want {w}");
    }
}

/// Cross-checked: hold at 0 (never broken out), n=2.
#[test]
fn hold_at_zero_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 9.0, 6.0, 8.0),
    ]);
    let got = decide_donchian_breakout(&s, 2);
    let want = [0.0, 0.0, 0.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "bar {i}: got {g}, want {w}");
    }
}

/// Cross-checked: off-by-one window boundary (n=2) -- an outlier exactly n+1=3 bars back from the decision bar
/// must be excluded from the channel.
#[test]
fn off_by_one_boundary_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 1000.0, 999.0, 999.5),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 12.0, 6.0, 7.0),
        (d(2024, 1, 4), 16.0, 14.0, 15.0),
    ]);
    let got = decide_donchian_breakout(&s, 2);
    assert!((got[2] - 0.0).abs() < 1e-9);
    assert!((got[3] - 1.0).abs() < 1e-9);
}

/// Cross-checked: insufficient history (fewer than n+1 bars) is a silent skip, flat.
#[test]
fn insufficient_history_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0),
        (d(2024, 1, 3), 10.0, 5.0, 7.0),
    ]);
    let got = decide_donchian_breakout(&s, 3);
    for (i, &g) in got.iter().enumerate() {
        assert!((g - 0.0).abs() < 1e-9, "bar {i}: got {g}, want 0.0");
    }
}
