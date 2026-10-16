//! Margin models (design 3.2 `MarginModel`; council rulings R3 and R4, `VENUE_FACTS.md`).
//!
//! A margin model answers three questions about a TARGET book of absolute notionals, all in account currency:
//! how much margin does it use (`margin_used`), what is the most it may use (`ceiling`), and how much new exposure in
//! one instrument can still be opened (`buying_power`, with the factor that bound it recorded, R3).
//!
//! Facts used, each from `VENUE_FACTS.md` (public broker documentation and account-neutral practice measurements):
//! * **OANDA**: margin is charged PER POSITION at the instrument's `marginRate` (measured 2026-09-23: EUR/USD 0.02,
//!   USD/JPY 0.05), with NO offset between instruments (a long EUR/USD and a short USD/JPY both consume margin), so the
//!   book's margin is the plain sum of `rate_i * |notional_i|`. `marginCallPercent = marginUsed / NAV` and
//!   `marginCloseoutPercent = marginUsed / (2 * NAV)`. R3: default ceiling `margin used <= 50% of NAV` for signed
//!   sleeves, configurable; the conservative NOTIONAL buying-power derivation is kept and the binding factor recorded.
//! * **Alpaca (Reg T)**: R4: opening-capacity margin rate is `max(asset requirement, 0.50)` for marginable securities
//!   and 1.00 for non-marginable ones; the broker's own buying power stays authoritative, and the smaller of
//!   `buying_power` and `regt_buying_power` (overnight Reg T 2x) is the figure to use. The model's ceiling is
//!   `initial margin used <= equity`.
//!
//! What this is NOT: maintenance margin, Alpaca's concentration rule or short-borrow costs. Those are live-only facts
//! monitored, not asserted equal (design 5.2 level 3).

use crate::construct::InstrumentFacts;
use crate::num::stable_sum;

/// The factor that limited a buying-power figure (R3: "record which factor bound the plan").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuyingPowerFactor {
    /// Margin available (`NAV - margin used`, or `equity - margin used`) was the smaller.
    MarginAvailable,
    /// The configured margin-used ceiling was the smaller.
    Ceiling,
    /// A broker-reported figure was the smaller.
    BrokerFigure,
}

/// New exposure, in account-currency NOTIONAL, that can still be opened in one instrument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuyingPower {
    pub notional: f64,
    pub bound_by: BuyingPowerFactor,
}

pub trait MarginModel {
    /// Stable name for provenance and reports.
    fn name(&self) -> &'static str;

    /// Initial-margin rate for one instrument as a fraction of `|notional|`, or `None` when the model needs a fact the
    /// instrument does not carry (fail closed: the caller refuses the book rather than guessing).
    fn rate(&self, inst: &InstrumentFacts) -> Option<f64>;

    /// The most margin the book may use, in currency, given the account's equity (NAV). `None` = no ceiling.
    fn ceiling(&self, equity: f64) -> Option<f64>;

    /// Margin used by a book of `(instrument, |target notional|)`: the sum of `rate_i * |notional_i|`, no offsets.
    /// `None` when any instrument's rate is unknown.
    fn margin_used(&self, book: &[(&InstrumentFacts, f64)]) -> Option<f64> {
        let mut terms = Vec::with_capacity(book.len());
        for (inst, notional) in book {
            terms.push(self.rate(inst)? * notional.abs());
        }
        Some(stable_sum(&terms))
    }

    /// New exposure in `inst` that may still be opened, given the equity and the margin already used.
    fn buying_power(&self, equity: f64, margin_used: f64, inst: &InstrumentFacts) -> Option<BuyingPower>;
}

/// No margin model: margin is 0, there is no ceiling, and buying power is unbounded. The research default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoMargin;

impl MarginModel for NoMargin {
    fn name(&self) -> &'static str {
        "none"
    }
    fn rate(&self, _inst: &InstrumentFacts) -> Option<f64> {
        Some(0.0)
    }
    fn ceiling(&self, _equity: f64) -> Option<f64> {
        None
    }
    fn buying_power(&self, _equity: f64, _margin_used: f64, _inst: &InstrumentFacts) -> Option<BuyingPower> {
        Some(BuyingPower { notional: f64::INFINITY, bound_by: BuyingPowerFactor::MarginAvailable })
    }
}

/// OANDA (R3). The instrument's `margin_rate` field carries the broker's `marginRate` (0.02 = 50:1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OandaMargin {
    /// Ceiling on margin used as a fraction of NAV. R3 default 0.50 ([`crate::OANDA_DEFAULT_MARGIN_CEILING`]).
    pub ceiling_fraction_of_nav: f64,
}

impl OandaMargin {
    /// R3 default: margin used <= 50% of NAV.
    pub const fn r3_default() -> Self {
        OandaMargin { ceiling_fraction_of_nav: crate::OANDA_DEFAULT_MARGIN_CEILING }
    }
    pub const fn with_ceiling(ceiling_fraction_of_nav: f64) -> Self {
        OandaMargin { ceiling_fraction_of_nav }
    }
    /// OANDA's `marginCallPercent`: `marginUsed / NAV` (>= 1.0 is a margin call).
    pub fn margin_call_percent(margin_used: f64, nav: f64) -> f64 {
        margin_used / nav
    }
    /// OANDA's `marginCloseoutPercent`: `marginUsed / (2 * NAV)` (>= 1.0 is a closeout).
    pub fn margin_closeout_percent(margin_used: f64, nav: f64) -> f64 {
        margin_used / (2.0 * nav)
    }
}

impl MarginModel for OandaMargin {
    fn name(&self) -> &'static str {
        "oanda"
    }
    fn rate(&self, inst: &InstrumentFacts) -> Option<f64> {
        inst.margin_rate.filter(|r| r.is_finite() && *r > 0.0)
    }
    fn ceiling(&self, equity: f64) -> Option<f64> {
        Some(self.ceiling_fraction_of_nav * equity)
    }
    fn buying_power(&self, equity: f64, margin_used: f64, inst: &InstrumentFacts) -> Option<BuyingPower> {
        let rate = self.rate(inst)?;
        let available = (equity - margin_used).max(0.0);
        let room = (self.ceiling_fraction_of_nav * equity - margin_used).max(0.0);
        // Conservative: the smaller of the two margin amounts, converted to NOTIONAL at this instrument's own rate.
        if room <= available {
            Some(BuyingPower { notional: room / rate, bound_by: BuyingPowerFactor::Ceiling })
        } else {
            Some(BuyingPower { notional: available / rate, bound_by: BuyingPowerFactor::MarginAvailable })
        }
    }
}

/// Alpaca Reg T (R4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlpacaRegT {
    /// Ceiling on initial margin used as a fraction of equity (1.0 = the Reg T rule "initial margin <= equity").
    pub ceiling_fraction_of_equity: f64,
    /// The broker's own opening buying power, if known: R4 says use the smaller of `buying_power` and
    /// `regt_buying_power`; pass that minimum here (see [`AlpacaRegT::with_broker_figures`]).
    pub broker_buying_power: Option<f64>,
}

impl AlpacaRegT {
    pub const fn r4_default() -> Self {
        AlpacaRegT { ceiling_fraction_of_equity: 1.0, broker_buying_power: None }
    }
    /// R4: the smaller of the broker's `buying_power` and `regt_buying_power`.
    pub fn with_broker_figures(buying_power: f64, regt_buying_power: f64) -> Self {
        AlpacaRegT { ceiling_fraction_of_equity: 1.0, broker_buying_power: Some(buying_power.min(regt_buying_power)) }
    }
}

impl MarginModel for AlpacaRegT {
    fn name(&self) -> &'static str {
        "alpaca_regt"
    }
    fn rate(&self, inst: &InstrumentFacts) -> Option<f64> {
        if !inst.marginable {
            return Some(1.0);
        }
        // R4: max(asset requirement, 0.50); an instrument without a stated requirement gets the 0.50 floor.
        let asset = inst.margin_rate.unwrap_or(0.0);
        if !asset.is_finite() || asset < 0.0 {
            return None;
        }
        Some(asset.max(crate::ALPACA_MIN_OPENING_MARGIN_RATE))
    }
    fn ceiling(&self, equity: f64) -> Option<f64> {
        Some(self.ceiling_fraction_of_equity * equity)
    }
    fn buying_power(&self, equity: f64, margin_used: f64, inst: &InstrumentFacts) -> Option<BuyingPower> {
        let rate = self.rate(inst)?;
        let room = (self.ceiling_fraction_of_equity * equity - margin_used).max(0.0) / rate;
        match self.broker_buying_power {
            Some(bp) if bp < room => {
                Some(BuyingPower { notional: bp.max(0.0), bound_by: BuyingPowerFactor::BrokerFigure })
            }
            _ => Some(BuyingPower { notional: room, bound_by: BuyingPowerFactor::MarginAvailable }),
        }
    }
}
