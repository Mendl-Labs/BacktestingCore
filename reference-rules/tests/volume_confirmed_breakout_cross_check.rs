//! Two-provider cross-check for `volume_confirmed_breakout` (W3.6, Range/volume family).
//!
//! The independent Python implementation (Kimi/Moonshot `kimi-k3`, zero shared context with the Rust author) was
//! given the same verbatim spec, with the plain Donchian primitive's resolved conventions (flat=0.0 hold-chain
//! seed, max-not-min insufficient-history threshold) restated explicitly. Both sides agree exactly on five
//! fixtures covering: a price breakout WITH volume confirmation (flips), a price breakout WITHOUT volume
//! confirmation (holds -- the primitive's whole point), a volume spike with NO price breakout (holds -- volume
//! alone never flips anything), the insufficient-history boundary using `max`, not `min`, of two DIFFERENT lookback
//! windows, and an off-by-one check on the volume window itself. No correction round was needed. Every expected
//! value is an exact 1.0 or 0.0, so "agreement to 1e-9" is exact equality.

use chrono::NaiveDate;
use reference_rules::{decide_volume_confirmed_breakout, OhlcvSeries};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn series(bars: &[(NaiveDate, f64, f64, f64, f64)]) -> OhlcvSeries {
    let dates = bars.iter().map(|b| b.0).collect();
    let highs = bars.iter().map(|b| b.1).collect();
    let lows = bars.iter().map(|b| b.2).collect();
    let closes = bars.iter().map(|b| b.3).collect();
    let volumes = bars.iter().map(|b| b.4).collect();
    OhlcvSeries::new("XCHK", dates, highs, lows, closes, volumes).unwrap()
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "bar {i}: got {g}, want {w}");
    }
}

/// Cross-checked against the Python key: price breakout WITH volume confirmation flips the weight.
#[test]
fn breakout_with_volume_confirmation_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 3), 12.0, 4.0, 11.0, 300.0),
    ]);
    assert_close(
        &decide_volume_confirmed_breakout(&s, 2, 2),
        &[0.0, 0.0, 1.0],
    );
}

/// Cross-checked: price breakout WITHOUT volume confirmation holds unchanged.
#[test]
fn breakout_without_volume_confirmation_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 3), 12.0, 4.0, 11.0, 120.0),
    ]);
    assert_close(
        &decide_volume_confirmed_breakout(&s, 2, 2),
        &[0.0, 0.0, 0.0],
    );
}

/// Cross-checked: a volume spike with no price breakout never flips anything.
#[test]
fn volume_spike_without_price_breakout_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 3), 9.0, 6.0, 8.0, 1000.0),
    ]);
    assert_close(
        &decide_volume_confirmed_breakout(&s, 2, 2),
        &[0.0, 0.0, 0.0],
    );
}

/// Cross-checked: insufficient-history threshold is `max`, not `min`, of two DIFFERENT lookback windows
/// (price_n=2, volume_n=4).
#[test]
fn max_not_min_history_threshold_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 3), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 4), 20.0, 3.0, 15.0, 10000.0),
        (d(2024, 1, 5), 10.0, 5.0, 7.0, 100.0),
    ]);
    assert_close(
        &decide_volume_confirmed_breakout(&s, 2, 4),
        &[0.0, 0.0, 0.0, 0.0, 0.0],
    );
}

/// Cross-checked: off-by-one boundary on the volume window (price_n=volume_n=2) -- an outlier volume exactly
/// n+1=3 bars back must be excluded from the trailing average.
#[test]
fn volume_window_off_by_one_boundary_matches_python_key() {
    let s = series(&[
        (d(2024, 1, 1), 10.0, 5.0, 7.0, 100000.0),
        (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 3), 10.0, 5.0, 7.0, 100.0),
        (d(2024, 1, 4), 20.0, 3.0, 15.0, 300.0),
    ]);
    let got = decide_volume_confirmed_breakout(&s, 2, 2);
    assert!((got[2] - 0.0).abs() < 1e-9);
    assert!((got[3] - 1.0).abs() < 1e-9);
}
