//! Volatility-target overlay: a pure rescaling function applied to an EXISTING weight, not a rule that produces
//! one from raw prices. Given a base weight already decided by some other rule, scale it up or down so the
//! position's expected contribution to portfolio volatility sits near a target annualized volatility, using the
//! asset's own trailing realized volatility as the estimate of what "no scaling" would currently deliver.
//!
//! `scale = target_vol / trailing_realized_vol` (how much more or less volatile the asset currently is than the
//! target), and `output = base_weight * scale`, with `scale` capped. `trailing_realized_vol` is expected to be
//! the caller's annualized trailing-20-bar realized volatility (population stdev of daily returns * sqrt(252));
//! this module does not compute it -- it is a pure function of three numbers, with no series, no dates, and no
//! history of its own.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Zero realized volatility is the spec's explicit degenerate case*: when `trailing_realized_vol == 0.0`
//!    exactly (a flat-price fixture, or an asset that simply hasn't moved over the window), `scale` would be
//!    `target_vol / 0.0` -- division by zero. The spec calls this out explicitly: output = `base_weight`
//!    unchanged, i.e. `scale` is treated as `1.0`, not as `+inf` and not as `0.0`. This is NOT "no position" and
//!    NOT "maximum leverage"; it is "we have no information to scale by, so don't touch the weight the base rule
//!    already decided."
//! 2. *Negative, NaN or infinite `trailing_realized_vol`* is not addressed by the spec (a real annualized
//!    population stdev can never be negative, NaN or infinite, but nothing stops a caller from passing a bad
//!    value in). This module treats every non-finite value AND every value `<= 0.0` identically to the exact-zero
//!    case: output = `base_weight` unchanged. Reasoning: the zero-case's own justification -- "cannot scale by
//!    something that isn't a real positive volatility" -- applies with exactly the same force to a negative
//!    number (which would flip the sign of `scale` for no economically meaningful reason), to NaN (which would
//!    poison `output` into NaN silently), and to `+inf`/`-inf` (which would force `output` to 0.0 or NaN). None of
//!    those are better defaults than "leave the base weight alone"; a value this module cannot interpret as a
//!    real volatility is handled the same way as "no information", not surfaced as a crash or a silent NaN. A
//!    non-finite `target_vol` is deliberately NOT special-cased the same way: it is a caller-chosen parameter
//!    (normally the compile-time default below), not a measured quantity, so if a caller passes a broken
//!    `target_vol` the resulting NaN/inf in `output` is left to propagate rather than silently papered over.
//! 3. *The leverage cap applies to the SCALING FACTOR, not to the final output weight's magnitude.* The spec's
//!    wording ("cap the scaling factor at 2.0") is taken literally: `scale` itself is clamped to at most
//!    [`VOL_TARGET_MAX_LEVERAGE`] (2.0) before being multiplied into `base_weight`, rather than clamping
//!    `base_weight * scale` to `[-2.0, 2.0]` or to `base_weight.abs() * 2.0`. Concretely, a `base_weight` of 0.1
//!    scaled by an uncapped ratio of 5.0 produces 0.2 (`0.1 * 2.0`, the capped factor), not `2.0` (the output
//!    clamped to the cap's raw value) and not `0.5` (`0.1 * 5.0`, uncapped).
//! 4. *The cap is one-sided*, i.e. `scale.min(VOL_TARGET_MAX_LEVERAGE)` with no corresponding lower clamp. Once
//!    past the zero/non-finite guard above, both `trailing_realized_vol` and `target_vol` are `> 0.0`, so the raw
//!    ratio `target_vol / trailing_realized_vol` is always strictly positive -- there is no symmetric negative
//!    case coming from the ratio itself that would need a `-2.0` floor. The cap only ever bites when realized
//!    volatility is LOW relative to target (pushing the scale-UP factor high, e.g. a quiet asset the overlay
//!    wants to lever up to reach the target); when realized volatility is HIGH relative to target the ratio is
//!    already comfortably under 1.0 and needs no cap at all (there is no corresponding "minimum leverage" floor
//!    in the spec, so a very volatile asset can be scaled down arbitrarily close to zero, just never negative
//!    relative to its own sign). `base_weight` itself may be negative (a short position fed in by the base
//!    rule); the cap is applied to the always-nonnegative `scale` BEFORE multiplying by `base_weight`, so a short
//!    is scaled by exactly the same factor as a long of the same magnitude would be, and its sign is preserved
//!    (see `negative_base_weight_scales_as_a_short` below).

/// Default annualized target volatility (15%) used when a caller has no more specific parameter to pass.
pub const VOL_TARGET_DEFAULT_TARGET_VOL: f64 = 0.15;

/// Maximum leverage multiplier applied to the raw `target_vol / trailing_realized_vol` scaling factor. The ratio
/// is clamped to at most this value before being multiplied into `base_weight`; see interpretation choices 3-4
/// in the module docs for why this is a cap on the factor, one-sided, and applied before considering the sign of
/// `base_weight`.
pub const VOL_TARGET_MAX_LEVERAGE: f64 = 2.0;

/// Rescale `base_weight` toward a target annualized volatility, given the asset's own trailing realized
/// volatility.
///
/// `output = base_weight * min(target_vol / trailing_realized_vol, VOL_TARGET_MAX_LEVERAGE)`, except that when
/// `trailing_realized_vol` is not a usable positive volatility -- exactly `0.0`, negative, `NaN` or infinite --
/// the scaling factor is treated as `1.0` and `output == base_weight` (see interpretation choices 1-2 in the
/// module docs). `target_vol` is not validated: a non-finite `target_vol` propagates into a non-finite `output`
/// rather than being silently substituted.
///
/// This is a pure function: no I/O, no series, no history. Callers are expected to pass an already-computed
/// annualized trailing-20-bar realized volatility (population stdev of daily returns * sqrt(252)) as
/// `trailing_realized_vol`, and typically [`VOL_TARGET_DEFAULT_TARGET_VOL`] (0.15) as `target_vol`.
pub fn vol_target_overlay(base_weight: f64, trailing_realized_vol: f64, target_vol: f64) -> f64 {
    if !trailing_realized_vol.is_finite() || trailing_realized_vol <= 0.0 {
        return base_weight;
    }
    let scale = (target_vol / trailing_realized_vol).min(VOL_TARGET_MAX_LEVERAGE);
    base_weight * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realized_vol_near_target_leaves_weight_near_unchanged() {
        // trailing_realized_vol == target_vol -> scale == 1.0 exactly.
        let out = vol_target_overlay(0.2, 0.15, 0.15);
        assert!((out - 0.2).abs() < 1e-12);
    }

    #[test]
    fn low_realized_vol_triggers_the_leverage_cap() {
        // target/realized = 0.15 / 0.03 = 5.0, far above the 2.0 cap -> capped factor applies, not the raw ratio.
        let out = vol_target_overlay(0.1, 0.03, 0.15);
        assert!((out - 0.1 * VOL_TARGET_MAX_LEVERAGE).abs() < 1e-12);
        assert!((out - 0.2).abs() < 1e-12);
        // Confirm it is NOT the uncapped scale (which would be 0.1 * 5.0 = 0.5).
        assert!((out - 0.5).abs() > 1e-9);
    }

    #[test]
    fn moderate_realized_vol_does_not_trigger_the_cap() {
        // target/realized = 0.15 / 0.10 = 1.5, comfortably under 2.0 -> uncapped ratio applies.
        let out = vol_target_overlay(0.2, 0.10, 0.15);
        assert!((out - 0.2 * 1.5).abs() < 1e-12);
        assert!((out - 0.3).abs() < 1e-12);
    }

    #[test]
    fn zero_realized_vol_leaves_base_weight_unchanged() {
        let out = vol_target_overlay(0.4, 0.0, 0.15);
        assert_eq!(out, 0.4);
    }

    #[test]
    fn negative_base_weight_scales_as_a_short() {
        // Same scale factor as the equivalent long (target/realized = 0.15/0.03 = 5.0, capped to 2.0); the sign
        // of base_weight is preserved exactly, the magnitude scaling is identical.
        let out = vol_target_overlay(-0.1, 0.03, 0.15);
        assert!((out - (-0.1 * VOL_TARGET_MAX_LEVERAGE)).abs() < 1e-12);
        assert!((out - (-0.2)).abs() < 1e-12);
        assert!(out < 0.0);
    }

    #[test]
    fn negative_realized_vol_treated_like_zero() {
        let out = vol_target_overlay(0.3, -0.05, 0.15);
        assert_eq!(out, 0.3);
    }

    #[test]
    fn nan_realized_vol_treated_like_zero() {
        let out = vol_target_overlay(0.3, f64::NAN, 0.15);
        assert_eq!(out, 0.3);
    }

    #[test]
    fn infinite_realized_vol_treated_like_zero() {
        let out_pos_inf = vol_target_overlay(0.3, f64::INFINITY, 0.15);
        assert_eq!(out_pos_inf, 0.3);
        let out_neg_inf = vol_target_overlay(0.3, f64::NEG_INFINITY, 0.15);
        assert_eq!(out_neg_inf, 0.3);
    }

    #[test]
    fn default_target_vol_constant_matches_spec() {
        assert_eq!(VOL_TARGET_DEFAULT_TARGET_VOL, 0.15);
        assert_eq!(VOL_TARGET_MAX_LEVERAGE, 2.0);
    }

    #[test]
    fn zero_base_weight_stays_zero_regardless_of_scale() {
        assert_eq!(vol_target_overlay(0.0, 0.03, 0.15), 0.0);
        assert_eq!(vol_target_overlay(0.0, 0.0, 0.15), 0.0);
    }
}
