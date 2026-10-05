//! Stop-loss overlay: a pure rescaling function applied to an EXISTING weight, not a rule that produces one
//! from raw prices. Given a base weight already decided by some other rule, plus the price at which that
//! position was entered and the current price, flatten the position (output 0.0) if the adverse move since
//! entry has reached or passed a configured percentage threshold; otherwise leave the weight untouched. This
//! is a BINARY stop only -- there is no partial de-risking as the stop level is approached, and no re-entry
//! logic; once flattened by this function on a given call, a later call with the same `entry_price` and a
//! price that has since recovered simply evaluates its own trigger condition again (the "position" has no
//! memory of having been stopped out before, because this module has no state at all).
//!
//! `output = base_weight`, unless:
//! * `base_weight > 0.0` (long) and `current_price <= entry_price * (1 - stop_loss_pct)`, in which case
//!   `output = 0.0`; or
//! * `base_weight < 0.0` (short) and `current_price >= entry_price * (1 + stop_loss_pct)`, in which case
//!   `output = 0.0`.
//!
//! `stop_loss_pct` is a parameter (default [`STOP_LOSS_OVERLAY_DEFAULT_PCT`], 0.10 = 10%), not a measured
//! quantity. This module does not know the asset, the venue, or the clock -- it is a pure function of four
//! numbers, with no series, no dates and no history of its own.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *`base_weight == 0.0` is an explicit no-op, not an error.* A stop-loss on a flat position has nothing to
//!    protect: there is no entry to have moved away from, so the function returns `0.0` immediately and does
//!    not even look at `entry_price`, `current_price` or `stop_loss_pct`. This holds for ANY price inputs,
//!    including non-finite or negative ones -- the spec says "regardless of prices", and this is taken
//!    literally: the flat-position check runs before any price or parameter validity check below, so a flat
//!    position can never be rejected for having bad price data attached to it.
//! 2. *The two trigger comparisons are mutually exclusive by construction, not by an extra guard.* Because a
//!    single `f64` cannot be simultaneously `> 0.0` and `< 0.0`, the implementation below uses a single
//!    `if base_weight > 0.0 { long check } else { short check }` (reached only once the `== 0.0` case has
//!    already returned), so the short branch is structurally unreachable for a positive `base_weight` and vice
//!    versa -- there is no path where both the long-stop and short-stop formulas are evaluated against the
//!    same call.
//! 3. *Inclusive boundaries are intentional and preserved exactly as specified*: `current_price` touching the
//!    stop level exactly DOES trigger the stop (`<=` for the long case, `>=` for the short case), unlike the
//!    strict `>`/`<` breakouts elsewhere in this crate (`donchian_breakout`, `volume_confirmed_breakout`). A
//!    stop-loss is a protective floor/ceiling, not a breakout signal: the economically sensible reading of "the
//!    position has lost (at least) `stop_loss_pct`" is satisfied AT the threshold, not only strictly past it,
//!    so the inclusive comparisons are kept exactly as given rather than normalized to match the strict-breakout
//!    convention used elsewhere in this crate.
//! 4. *`entry_price` or `current_price` non-positive, NaN or infinite* is not addressed by the spec (a real
//!    traded price can never be `<= 0.0` or non-finite, but nothing stops a caller from passing a bad value
//!    in). Unlike [`crate::vol_target_overlay`]'s analogous question about `trailing_realized_vol` -- where the
//!    chosen default was "leave `base_weight` unchanged, we have no information to scale by" -- this module
//!    defaults to **flattening** (`output = 0.0`) whenever `base_weight != 0.0` and either price fails
//!    `is_finite() && > 0.0`. The reasoning is deliberately different because the two overlays carry opposite
//!    risk asymmetries: `vol_target_overlay` is a performance-tuning rescale, where "leave the weight alone"
//!    when the rescale can't be computed is the low-risk default (worst case, the position is not rescaled
//!    toward a vol target this bar). A stop-loss overlay exists specifically to bound downside risk; if the
//!    prices needed to verify "has the stop been breached?" are themselves untrustworthy, silently leaving the
//!    position at `base_weight` risks holding an exposure whose real stop condition cannot be checked at all --
//!    an unbounded, silent failure of the exact protection this function exists to provide. Flattening is the
//!    conservative choice: the downside of wrongly flattening a healthy position is a bounded opportunity cost,
//!    while the downside of wrongly staying exposed on bad data is unbounded. This asymmetry, not a desire for
//!    consistency with the sibling overlay, is why the default is the opposite of `vol_target_overlay`'s.
//! 5. *`stop_loss_pct` negative, NaN or infinite* is likewise not addressed by the spec and is ALSO guarded to
//!    flatten (`output = 0.0` when `base_weight != 0.0`), for a related but distinct reason. `stop_loss_pct` is
//!    a caller-chosen parameter, not a measured quantity -- by the same reasoning `vol_target_overlay` applies
//!    to its own caller-chosen `target_vol`, one might expect it to be left unvalidated and allowed to
//!    propagate. It is NOT left unvalidated here, because of a correctness trap specific to this primitive's
//!    comparisons: IEEE-754 float comparisons involving `NaN` are always `false`. If `stop_loss_pct` were `NaN`
//!    and left unguarded, `entry_price * (1 - f64::NAN)` is `NaN`, and `current_price <= NaN` evaluates to
//!    `false` in Rust -- silently taking the "not triggered, output unchanged" branch. That is the opposite of
//!    `vol_target_overlay`'s analogous case, where a non-finite `target_vol` visibly poisons `output` into a
//!    loud `NaN` the caller cannot miss. Here it would instead vanish into an ordinary-looking unchanged weight,
//!    masking exactly the failure a stop-loss is supposed to catch. A negative (but finite) `stop_loss_pct` is
//!    guarded for a simpler reason: it inverts which side of `entry_price` the stop level sits on (for a long,
//!    `entry_price * (1 - stop_loss_pct)` moves ABOVE `entry_price` instead of below it), which can cause a
//!    healthy, barely-moved position to be spuriously flattened the moment price ticks at all -- a silent,
//!    direction-flipped bug rather than a usable configuration. Because the resulting error's direction is
//!    unpredictable (it can either over- or under-protect depending on magnitude), this module declines to
//!    guess and takes the same protective default as every other unverifiable-input case: flatten.
//! 6. *`stop_loss_pct == 0.0` is valid and NOT guarded* -- it is a legitimate (if aggressive) "stop at the
//!    first sign of loss, even a single tick at or through breakeven" configuration, not a sign of a bad input.
//!    It flows through the ordinary comparison unchanged.
//! 7. *`stop_loss_pct >= 1.0` (and finite) is valid and NOT guarded either*, despite being an unusual
//!    configuration (a 100%-or-more stop). No special case is needed: for a long, the stop level
//!    `entry_price * (1 - stop_loss_pct)` becomes zero or negative, and since a valid `current_price` is
//!    already required to be `> 0.0` by the price-validity guard above, the trigger condition
//!    `current_price <= stop_level` can never be true -- the arithmetic degrades gracefully to "this stop can
//!    never fire," which is a safe (merely useless) outcome, not a dangerous one, so it needs no explicit
//!    guard. The symmetric short case (`entry_price * (1 + stop_loss_pct)` growing large) degrades the same
//!    way.

/// Default stop-loss percentage (10%) used when a caller has no more specific parameter to pass.
pub const STOP_LOSS_OVERLAY_DEFAULT_PCT: f64 = 0.10;

/// Flatten `base_weight` to `0.0` if the adverse move from `entry_price` to `current_price` has reached or
/// passed `stop_loss_pct`; otherwise return `base_weight` unchanged.
///
/// * `base_weight == 0.0` always returns `0.0`, regardless of any other argument (see interpretation choice 1
///   in the module docs) -- a stop-loss on a flat position is a no-op, not an error.
/// * For `base_weight > 0.0` (long), the stop level is `entry_price * (1.0 - stop_loss_pct)`; triggered when
///   `current_price <= stop_level` (inclusive).
/// * For `base_weight < 0.0` (short), the stop level is `entry_price * (1.0 + stop_loss_pct)`; triggered when
///   `current_price >= stop_level` (inclusive).
/// * For `base_weight != 0.0`, if `entry_price` or `current_price` is not finite and strictly positive, or if
///   `stop_loss_pct` is not finite or is negative, this function returns `0.0` (flattens) rather than
///   evaluating an unverifiable trigger condition -- see interpretation choices 4-5 in the module docs for why
///   this overlay fails closed (flatten) where the sibling [`crate::vol_target_overlay`] fails open (leave
///   unchanged).
///
/// This is a pure function: no I/O, no series, no history, no memory of prior calls.
pub fn stop_loss_overlay(
    base_weight: f64,
    entry_price: f64,
    current_price: f64,
    stop_loss_pct: f64,
) -> f64 {
    if base_weight == 0.0 {
        return 0.0;
    }

    let prices_valid = entry_price.is_finite()
        && entry_price > 0.0
        && current_price.is_finite()
        && current_price > 0.0;
    let pct_valid = stop_loss_pct.is_finite() && stop_loss_pct >= 0.0;
    if !prices_valid || !pct_valid {
        return 0.0;
    }

    if base_weight > 0.0 {
        let stop_level = entry_price * (1.0 - stop_loss_pct);
        if current_price <= stop_level {
            return 0.0;
        }
    } else {
        let stop_level = entry_price * (1.0 + stop_loss_pct);
        if current_price >= stop_level {
            return 0.0;
        }
    }

    base_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_position_stopped_out_below_stop_level() {
        // entry 100, stop 10% -> stop level 90. current 85 < 90 -> stopped out.
        let out = stop_loss_overlay(0.2, 100.0, 85.0, 0.10);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn long_position_not_stopped_out_above_stop_level() {
        let out = stop_loss_overlay(0.2, 100.0, 95.0, 0.10);
        assert_eq!(out, 0.2);
    }

    #[test]
    fn long_position_exact_boundary_is_inclusive_and_triggers() {
        // current_price exactly equal to the stop level (100 * 0.9 = 90.0 exactly) must trigger: <=, not <.
        let out = stop_loss_overlay(0.2, 100.0, 90.0, 0.10);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn long_position_just_above_boundary_does_not_trigger() {
        let out = stop_loss_overlay(0.2, 100.0, 90.000001, 0.10);
        assert_eq!(out, 0.2);
    }

    #[test]
    fn short_position_stopped_out_above_stop_level() {
        // entry 100, stop 10% -> stop level 110. current 115 > 110 -> stopped out.
        let out = stop_loss_overlay(-0.2, 100.0, 115.0, 0.10);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn short_position_not_stopped_out_below_stop_level() {
        let out = stop_loss_overlay(-0.2, 100.0, 105.0, 0.10);
        assert_eq!(out, -0.2);
    }

    #[test]
    fn short_position_exact_boundary_is_inclusive_and_triggers() {
        // Uses stop_loss_pct = 0.25 (an exact binary fraction) so entry_price * (1 + stop_loss_pct) rounds to
        // exactly 125.0 in f64 -- 100.0 * 1.10 would round to 110.00000000000001, which would make an
        // "exact boundary" test with 0.10 spuriously fail on a correct implementation. current_price exactly
        // equal to the stop level (100 * 1.25 = 125.0 exactly) must trigger: >=, not >.
        let out = stop_loss_overlay(-0.2, 100.0, 125.0, 0.25);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn short_position_just_below_boundary_does_not_trigger() {
        let out = stop_loss_overlay(-0.2, 100.0, 109.999999, 0.10);
        assert_eq!(out, -0.2);
    }

    #[test]
    fn flat_position_is_a_no_op_regardless_of_adverse_prices() {
        // Wildly adverse / nonsensical price combinations must still return 0.0, never error or diverge.
        assert_eq!(stop_loss_overlay(0.0, 100.0, 1.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.0, 100.0, 1_000_000.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.0, 100.0, 0.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.0, -5.0, f64::NAN, 2.0), 0.0);
        assert_eq!(
            stop_loss_overlay(0.0, f64::INFINITY, f64::NEG_INFINITY, -1.0),
            0.0
        );
    }

    #[test]
    fn invalid_entry_price_flattens_a_nonzero_long() {
        assert_eq!(stop_loss_overlay(0.2, 0.0, 95.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.2, -100.0, 95.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.2, f64::NAN, 95.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(0.2, f64::INFINITY, 95.0, 0.10), 0.0);
    }

    #[test]
    fn invalid_current_price_flattens_a_nonzero_short() {
        assert_eq!(stop_loss_overlay(-0.2, 100.0, 0.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(-0.2, 100.0, -50.0, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(-0.2, 100.0, f64::NAN, 0.10), 0.0);
        assert_eq!(stop_loss_overlay(-0.2, 100.0, f64::INFINITY, 0.10), 0.0);
    }

    #[test]
    fn negative_stop_loss_pct_flattens_rather_than_inverting_the_trigger() {
        // Without the guard, a negative pct would move the long stop level ABOVE entry_price and spuriously
        // trigger on an essentially unmoved position; this module refuses to evaluate it instead.
        let out = stop_loss_overlay(0.2, 100.0, 100.0, -0.05);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn nan_stop_loss_pct_flattens_rather_than_silently_passing_through() {
        // Without the guard, NaN comparisons are always false in Rust, which would silently take the
        // "not triggered" branch and leave base_weight unchanged -- masking the bad input. This module
        // guards explicitly so the failure is the protective default (flatten), not a silent pass-through.
        let out = stop_loss_overlay(0.3, 100.0, 85.0, f64::NAN);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn infinite_stop_loss_pct_flattens() {
        assert_eq!(stop_loss_overlay(0.3, 100.0, 85.0, f64::INFINITY), 0.0);
        assert_eq!(stop_loss_overlay(0.3, 100.0, 85.0, f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn zero_stop_loss_pct_is_valid_and_triggers_at_breakeven_touch() {
        // stop level == entry_price exactly; current_price touching entry_price triggers (inclusive <=).
        assert_eq!(stop_loss_overlay(0.2, 100.0, 100.0, 0.0), 0.0);
        // Above entry_price, not triggered.
        assert_eq!(stop_loss_overlay(0.2, 100.0, 100.01, 0.0), 0.2);
    }

    #[test]
    fn stop_loss_pct_at_or_above_one_degrades_to_never_triggering_for_a_long() {
        // stop level <= 0.0; a valid current_price (> 0.0) can never satisfy current_price <= stop_level.
        let out = stop_loss_overlay(0.2, 100.0, 1.0, 1.5);
        assert_eq!(out, 0.2);
    }

    #[test]
    fn default_stop_loss_pct_constant_matches_spec() {
        assert_eq!(STOP_LOSS_OVERLAY_DEFAULT_PCT, 0.10);
    }
}
