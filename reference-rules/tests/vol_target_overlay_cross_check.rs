//! Two-provider cross-check for `vol_target_overlay` (W3.6, Overlays family).
//!
//! The independent Python implementation (Kimi/Moonshot `kimi-k3`, zero shared context with the Rust author) was
//! given the same verbatim spec plus the same two explicitly-flagged edge-case questions (the spec only pins down
//! the EXACT `0.0` realized-vol case; negative/NaN/infinite realized vol and the one-sidedness of the leverage
//! cap are genuine spec gaps). Both sides independently landed on identical behavior for every case this crate
//! exercises: non-positive or non-finite `trailing_realized_vol` (negative, NaN, +-inf, exact zero) all pass the
//! base weight through unchanged; the cap applies to the scaling FACTOR (not the final output), one-sided
//! (upper bound only); a negative `base_weight` (short) is scaled by the same factor as the equivalent long,
//! preserving sign.
//!
//! One disclosed, non-blocking divergence: for an INVALID `target_vol` (negative, NaN or infinite -- itself not a
//! case this crate's spec addresses, since `target_vol` is a caller-chosen parameter, normally the compile-time
//! default, not a measured quantity), the Rust implementation lets the resulting NaN/inf propagate into the
//! output, while the first Kimi draft raises `ValueError`. This is a difference in defensive-programming
//! philosophy on an input the spec never defines, not a disagreement about the primitive's actual rule, so it is
//! disclosed here rather than forced to agree; this cross-check (and the fixtures below) only exercises valid
//! `target_vol` values, where both implementations agree exactly.
//!
//! Agreement is to 1e-9 per the task's convention (most values here are 1e-12-exact since no primitive-specific
//! rounding is introduced beyond the one division and one multiplication the spec itself requires).

use reference_rules::{vol_target_overlay, VOL_TARGET_DEFAULT_TARGET_VOL, VOL_TARGET_MAX_LEVERAGE};

fn assert_close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
}

/// Cross-checked: realized vol at the target -> scale factor is exactly 1.0, output unchanged.
#[test]
fn ratio_near_one_matches_python_key() {
    assert_close(vol_target_overlay(0.2, 0.15, 0.15), 0.2);
}

/// Cross-checked: realized vol far below target (ratio 5.0) -> the 2.0x cap bites; output is base_weight * 2.0,
/// NOT base_weight * the uncapped 5.0.
#[test]
fn leverage_cap_bites_matches_python_key() {
    let out = vol_target_overlay(0.1, 0.03, 0.15);
    assert_close(out, 0.1 * VOL_TARGET_MAX_LEVERAGE);
    assert_close(out, 0.2);
    assert!(
        (out - 0.5).abs() > 1e-9,
        "must not equal the uncapped ratio's output"
    );
}

/// Cross-checked: realized vol moderately below target (ratio 1.5) -> cap does not bite, uncapped ratio applies.
#[test]
fn leverage_cap_does_not_bite_matches_python_key() {
    assert_close(vol_target_overlay(0.2, 0.10, 0.15), 0.3);
}

/// Cross-checked: the spec's explicit exact-zero edge case -- output = base_weight unchanged, not a crash, not
/// +inf, not 0.0.
#[test]
fn exact_zero_realized_vol_matches_python_key() {
    assert_close(vol_target_overlay(0.4, 0.0, 0.15), 0.4);
}

/// Cross-checked: a negative base_weight (short) is scaled by the same capped factor as an equivalent long,
/// preserving sign.
#[test]
fn negative_base_weight_matches_python_key() {
    let out = vol_target_overlay(-0.1, 0.03, 0.15);
    assert_close(out, -0.1 * VOL_TARGET_MAX_LEVERAGE);
    assert!(out < 0.0);
}

/// Cross-checked: negative, NaN and +-infinite realized vol are all treated identically to the exact-zero case --
/// a genuine spec gap both implementations independently closed the same way.
#[test]
fn non_positive_and_non_finite_realized_vol_matches_python_key() {
    assert_close(vol_target_overlay(0.3, -0.05, 0.15), 0.3);
    assert_close(vol_target_overlay(0.3, f64::NAN, 0.15), 0.3);
    assert_close(vol_target_overlay(0.3, f64::INFINITY, 0.15), 0.3);
    assert_close(vol_target_overlay(0.3, f64::NEG_INFINITY, 0.15), 0.3);
}

/// Cross-checked: a zero base_weight stays zero regardless of the scale factor.
#[test]
fn zero_base_weight_matches_python_key() {
    assert_close(vol_target_overlay(0.0, 0.03, 0.15), 0.0);
    assert_close(vol_target_overlay(0.0, 0.0, 0.15), 0.0);
}

/// The default target_vol constant matches the spec (0.15 = 15% annualized) and the leverage cap is 2.0x.
#[test]
fn default_constants_match_spec() {
    assert_close(VOL_TARGET_DEFAULT_TARGET_VOL, 0.15);
    assert_close(VOL_TARGET_MAX_LEVERAGE, 2.0);
}
