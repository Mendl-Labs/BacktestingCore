//! Two-provider cross-check for `stop_loss_overlay` (W3.6, Overlays family, final primitive).
//!
//! The independent Python implementation (Kimi/Moonshot `kimi-k3`, zero shared context with the Rust author) was
//! given the same verbatim spec plus the same two explicitly-flagged spec gaps (invalid prices; an out-of-range
//! `stop_loss_pct`), with no hint of either side's answer. On every case the spec actually defines -- long/short
//! stop triggers, the INCLUSIVE `<=`/`>=` boundaries (preserved exactly, not normalized to the strict `>`/`<`
//! breakout convention used elsewhere in this crate), the flat (`base_weight == 0.0`) no-op that holds regardless
//! of prices, and the degenerate `stop_loss_pct == 0.0` breakeven-touch case -- the two implementations agree
//! exactly.
//!
//! One disclosed, non-blocking divergence, by design: for the two spec gaps, Kimi chose fail-fast (`ValueError`
//! on invalid prices or an out-of-range `stop_loss_pct`), while the Rust implementation chose fail-CLOSED
//! (flatten the position, `output = 0.0`) -- the opposite default from the sibling `vol_target_overlay`, which
//! fails OPEN (leaves the weight unchanged) on its own analogous gap. The asymmetry is deliberate and documented
//! in `stop_loss_overlay.rs`'s module doc: a stop-loss exists specifically to bound downside risk, so silently
//! leaving a position exposed when its stop condition cannot be verified is the dangerous failure mode here,
//! whereas for a vol-target rescale the safer default is to leave the position untouched. This is a genuine,
//! context-dependent design choice on inputs the spec never defines, not a disagreement about the primitive's
//! actual rule -- not forced to agree, exactly as `vol_target_overlay`'s analogous divergence was handled. This
//! cross-check accordingly exercises ONLY the core, spec-defined behavior (valid inputs), where both
//! implementations agree exactly.

use reference_rules::{stop_loss_overlay, STOP_LOSS_OVERLAY_DEFAULT_PCT};

fn assert_close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
}

/// Cross-checked: a long position whose price has fallen past the stop level is flattened.
#[test]
fn long_stopped_out_matches_python_key() {
    assert_close(stop_loss_overlay(0.2, 100.0, 85.0, 0.10), 0.0);
}

/// Cross-checked: a long position above the stop level is left unchanged.
#[test]
fn long_not_stopped_out_matches_python_key() {
    assert_close(stop_loss_overlay(0.2, 100.0, 95.0, 0.10), 0.2);
}

/// Cross-checked: the long-stop boundary is INCLUSIVE -- touching the stop level exactly triggers.
#[test]
fn long_exact_boundary_inclusive_matches_python_key() {
    assert_close(stop_loss_overlay(0.2, 100.0, 90.0, 0.10), 0.0);
}

/// Cross-checked: a long position just above the (inclusive) boundary does not trigger.
#[test]
fn long_just_above_boundary_matches_python_key() {
    assert_close(stop_loss_overlay(0.2, 100.0, 90.000001, 0.10), 0.2);
}

/// Cross-checked: a short position whose price has risen past the stop level is flattened.
#[test]
fn short_stopped_out_matches_python_key() {
    assert_close(stop_loss_overlay(-0.2, 100.0, 115.0, 0.10), 0.0);
}

/// Cross-checked: a short position below the stop level is left unchanged.
#[test]
fn short_not_stopped_out_matches_python_key() {
    assert_close(stop_loss_overlay(-0.2, 100.0, 105.0, 0.10), -0.2);
}

/// Cross-checked: the short-stop boundary is INCLUSIVE -- touching the stop level exactly triggers. Uses
/// stop_loss_pct = 0.25 (an exact binary fraction) so entry_price * (1 + stop_loss_pct) rounds to exactly 125.0,
/// avoiding an f64-rounding false negative that 0.10 would introduce at this exact-equality boundary.
#[test]
fn short_exact_boundary_inclusive_matches_python_key() {
    assert_close(stop_loss_overlay(-0.2, 100.0, 125.0, 0.25), 0.0);
}

/// Cross-checked: a short position just below the (inclusive) boundary does not trigger.
#[test]
fn short_just_below_boundary_matches_python_key() {
    assert_close(stop_loss_overlay(-0.2, 100.0, 109.999999, 0.10), -0.2);
}

/// Cross-checked: a flat position (base_weight == 0.0) is always a no-op, regardless of adverse prices.
#[test]
fn flat_position_no_op_matches_python_key() {
    assert_close(stop_loss_overlay(0.0, 100.0, 1.0, 0.10), 0.0);
    assert_close(stop_loss_overlay(0.0, 100.0, 1_000_000.0, 0.10), 0.0);
}

/// Cross-checked: stop_loss_pct == 0.0 is a valid, degenerate "stop at breakeven touch" configuration.
#[test]
fn zero_stop_loss_pct_breakeven_matches_python_key() {
    assert_close(stop_loss_overlay(0.2, 100.0, 100.0, 0.0), 0.0);
    assert_close(stop_loss_overlay(0.2, 100.0, 100.01, 0.0), 0.2);
}

/// The default stop_loss_pct constant matches the spec (0.10 = 10%).
#[test]
fn default_constant_matches_spec() {
    assert_close(STOP_LOSS_OVERLAY_DEFAULT_PCT, 0.10);
}
