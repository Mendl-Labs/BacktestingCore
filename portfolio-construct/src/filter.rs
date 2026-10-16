//! The trade filter (design 3.2 `TradeFilter`): the planner's two numbers.
//!
//! For one instrument with target notional `T` and current notional `C` (`delta = T - C`), the trade is DROPPED when
//! `|delta| < min_abs` (account-currency units) or `|delta| < min_pct * ref`, where `ref = |T|` and, when the target is
//! zero (a full exit), `ref = |C|` (so a full exit is never exempt from the band). Both comparisons are strict: a delta
//! exactly on the threshold trades. The band is a fraction of the TARGET, not of the current holding; the mutant
//! "filter on current instead of target" is one of the design's 6.5 list.

use crate::num::below;
use crate::rounding::SizeRefusal;

/// Why a candidate trade was not sized or not placed.
#[derive(Clone, Debug, PartialEq)]
pub enum SkipReason {
    /// The instrument has no usable (positive, finite) price.
    NoPrice,
    /// A short is held in an instrument no signed sleeve manages; it is never touched.
    ShortPositionHeld,
    /// `|delta| < min_abs`.
    BelowMinAbs { delta: f64, min: f64 },
    /// `|delta| < min_pct * reference`.
    BelowMinPct { delta: f64, min: f64 },
    /// The quantity rounder refused (minimums, unknown instrument, ...).
    VenueRefused(SizeRefusal),
    /// The buy did not fit the cash left after the reserve and fees; scaled down, it fell below the venue minimum.
    CutBelowVenueMinimum,
    /// No cash (or buying power) above the reserve.
    NoCashAvailable,
}

/// The planner's `min_trade_abs` and `min_trade_pct`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeFilter {
    /// Account-currency units. The planner default is 10.
    pub min_abs: f64,
    /// Fraction of the target, in `[0, 1)`. The planner default is 0.02.
    pub min_pct: f64,
}

impl TradeFilter {
    /// No filter: every non-zero delta trades.
    pub const NONE: TradeFilter = TradeFilter { min_abs: 0.0, min_pct: 0.0 };
    /// The planner's defaults (`RunConfig`): 10 units and 2%.
    pub const PLANNER_DEFAULT: TradeFilter =
        TradeFilter { min_abs: crate::DEFAULT_MIN_TRADE_ABS, min_pct: crate::DEFAULT_MIN_TRADE_PCT };

    pub const fn new(min_abs: f64, min_pct: f64) -> Self {
        TradeFilter { min_abs, min_pct }
    }

    /// `Err` text when the parameters are outside their ranges (`min_abs >= 0`, `0 <= min_pct < 1`, finite).
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.min_abs.is_finite() || self.min_abs < 0.0 {
            return Err("min_trade_abs must be finite and not negative");
        }
        if !self.min_pct.is_finite() || self.min_pct < 0.0 || self.min_pct >= 1.0 {
            return Err("min_trade_pct must be in [0, 1)");
        }
        Ok(())
    }

    /// `None` when the trade from `current` to `target` passes; otherwise the reason it is dropped. A zero delta is
    /// not a trade and is the caller's business (this function is only asked about non-zero deltas).
    pub fn check(&self, target: f64, current: f64) -> Option<SkipReason> {
        let abs_delta = (target - current).abs();
        if below(abs_delta, self.min_abs) {
            return Some(SkipReason::BelowMinAbs { delta: abs_delta, min: self.min_abs });
        }
        let reference = if target == 0.0 { current.abs() } else { target.abs() };
        let pct_min = self.min_pct * reference;
        if below(abs_delta, pct_min) {
            return Some(SkipReason::BelowMinPct { delta: abs_delta, min: pct_min });
        }
        None
    }

    /// The holding after the filter has decided: `target` when the trade passes (or there is nothing to trade), the
    /// unchanged `current` when it is dropped. Idempotent: `apply(t, apply(t, c)) == apply(t, c)`.
    pub fn apply(&self, target: f64, current: f64) -> f64 {
        if target == current || self.check(target, current).is_some() {
            current
        } else {
            target
        }
    }
}
