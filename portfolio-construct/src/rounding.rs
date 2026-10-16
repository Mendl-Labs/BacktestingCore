//! Quantity rounding by venue rules (design 3.2 `QuantityRounder`).
//!
//! The live planner asks a `VenueRules` to turn a wanted quantity into one the venue accepts: ALWAYS rounded down, or a
//! typed refusal, never bumped up to a minimum. This module is the f64 restatement of that contract plus a generic
//! table-driven implementation ([`LotRounder`]) whose per-instrument facts (unit precision, minimum quantity, minimum
//! order value, maximum order units) are the ones the venue adapters read from the broker (Kraken `lot_decimals` /
//! `ordermin` / `costmin`, Alpaca fractionable vs whole shares and a minimum notional, OANDA `tradeUnitsPrecision` /
//! `minimumTradeSize` / `maximumOrderUnits`). No venue constant lives here: the caller supplies the table, and the
//! backtester passes `None` (no rounding) in research runs.

use crate::num::floor_dp;
use std::collections::BTreeMap;

/// Order side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side {
    Buy,
    Sell,
}

/// Why a quantity could not be turned into a sendable order size (mirrors the planner's `SizeRefusal`).
#[derive(Clone, Debug, PartialEq)]
pub enum SizeRefusal {
    UnknownInstrument(String),
    NotTradable(String),
    RoundsToZero,
    BelowMinQuantity { min: f64 },
    BelowMinCost { min: f64 },
    Other(String),
}

/// Turns a wished quantity (a magnitude; the side gives the sign) into a sendable one.
pub trait QuantityRounder {
    /// The quantity to send for a wish of `quantity` at reference `price`: rounded DOWN to the venue's precision, or a
    /// refusal. Never larger than `quantity` (the caller treats a larger answer as a hard error).
    fn round_quantity(&self, symbol: &str, side: Side, quantity: f64, price: f64) -> Result<f64, SizeRefusal>;
}

/// The identity: fractional units, no minimums. What a research run uses; also `rounding: None`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactUnits;

impl QuantityRounder for ExactUnits {
    fn round_quantity(&self, _symbol: &str, _side: Side, quantity: f64, _price: f64) -> Result<f64, SizeRefusal> {
        Ok(quantity)
    }
}

/// One instrument's lot rules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LotRule {
    /// Decimal places of the unit quantity (0 = whole units/shares).
    pub units_dp: u32,
    /// Smallest quantity accepted (after rounding). 0 = none.
    pub min_quantity: f64,
    /// Smallest order value (quantity * price) accepted. 0 = none.
    pub min_notional: f64,
    /// Largest single order in units. `None` = no stated limit.
    pub max_order_units: Option<f64>,
}

impl LotRule {
    /// Fractional to `units_dp` decimals, no minimums.
    pub const fn new(units_dp: u32) -> Self {
        LotRule { units_dp, min_quantity: 0.0, min_notional: 0.0, max_order_units: None }
    }
    pub const fn with_min_quantity(mut self, min: f64) -> Self {
        self.min_quantity = min;
        self
    }
    pub const fn with_min_notional(mut self, min: f64) -> Self {
        self.min_notional = min;
        self
    }
    pub const fn with_max_order_units(mut self, max: f64) -> Self {
        self.max_order_units = Some(max);
        self
    }
}

/// A table of lot rules keyed by symbol (trimmed, upper case). A symbol with no row is `UnknownInstrument`: nothing is
/// guessed, exactly as the planner's venue rules.
#[derive(Clone, Debug, Default)]
pub struct LotRounder {
    rules: BTreeMap<String, LotRule>,
}

impl LotRounder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, symbol: &str, rule: LotRule) -> Self {
        self.rules.insert(symbol.trim().to_uppercase(), rule);
        self
    }
}

impl QuantityRounder for LotRounder {
    fn round_quantity(&self, symbol: &str, _side: Side, quantity: f64, price: f64) -> Result<f64, SizeRefusal> {
        let rule = self
            .rules
            .get(&symbol.trim().to_uppercase())
            .ok_or_else(|| SizeRefusal::UnknownInstrument(symbol.to_string()))?;
        let q = floor_dp(quantity, rule.units_dp);
        if q <= 0.0 {
            return Err(SizeRefusal::RoundsToZero);
        }
        if q < rule.min_quantity {
            return Err(SizeRefusal::BelowMinQuantity { min: rule.min_quantity });
        }
        if q * price < rule.min_notional {
            return Err(SizeRefusal::BelowMinCost { min: rule.min_notional });
        }
        if let Some(max) = rule.max_order_units {
            if q > max {
                return Err(SizeRefusal::Other(format!("{q} units exceeds maximumOrderUnits {max}")));
            }
        }
        Ok(q)
    }
}
