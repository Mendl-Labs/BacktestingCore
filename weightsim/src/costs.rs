//! Transaction costs and the financing hook (design 2.5, 2.6).
//!
//! Costs are charged on TURNOVER only (S-9): `cost = rate x sum_i |traded notional_i|` at each rebalance, deducted
//! from equity at the rebalance instant. Nothing is charged for holding. Financing is a separate, explicit hook and is
//! `None` in every certification run.

/// A named, declared cost preset. `ZERO` is an explicit preset, not the absence of one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub id: &'static str,
    pub commission_bps: f64,
    pub half_spread_bps: f64,
    pub slippage_bps: f64,
    pub source_note: &'static str,
}

impl CostModel {
    /// All zero: gross runs and the identity tier.
    pub const ZERO: CostModel = CostModel {
        id: "zero",
        commission_bps: 0.0,
        half_spread_bps: 0.0,
        slippage_bps: 0.0,
        source_note: "explicit zero-cost preset; gross runs and identity tests",
    };

    /// 10 bps per side on every asset. A cost-ACCOUNTING device for the certification ladder, explicitly not a
    /// realism claim (design 2.5). Venue presets (`venue_*_v1`) are NOT defined here: they must come from measured
    /// venue facts, not be invented in the simulator.
    pub const CERTIFICATION_FLAT_10BPS_PER_SIDE: CostModel = CostModel {
        id: "certification_flat_10bps_per_side",
        commission_bps: 10.0,
        half_spread_bps: 0.0,
        slippage_bps: 0.0,
        source_note: "10 bps per side, all assets; cost-accounting device, NOT a realism claim",
    };

    /// Look a preset up by its id (requests select presets by id; there are no free numeric fields).
    pub fn by_id(id: &str) -> Option<CostModel> {
        match id {
            "zero" => Some(CostModel::ZERO),
            "certification_flat_10bps_per_side" => Some(CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE),
            _ => None,
        }
    }

    /// Total bps charged per unit of traded notional.
    pub fn rate_bps(&self) -> f64 {
        self.commission_bps + self.half_spread_bps + self.slippage_bps
    }

    /// Fractional rate (bps / 10_000).
    pub fn rate(&self) -> f64 {
        self.rate_bps() / 10_000.0
    }

    /// `true` when every component is finite and non-negative.
    pub fn is_valid(&self) -> bool {
        [self.commission_bps, self.half_spread_bps, self.slippage_bps].iter().all(|v| v.is_finite() && *v >= 0.0)
    }
}

/// Financing hook (design 2.6, S-8). Accrued on positions and cash held over the calendar days between two
/// consecutive bars, actual/365, using the start-of-interval notionals (previous bar's close).
///
/// `PolicyRateCarry` (Core's policy-rate table) is Stage T5 and is intentionally not part of T1; adding it is one more
/// variant plus one match arm in [`Financing::accrual`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Financing {
    /// No financing (certification; the answer key ignores it).
    None,
    /// Flat annual rates in bps. `long_bps` is a cost per annum on long market value, `short_bps` a cost per annum on
    /// short market value (both >= 0 mean cost); `cash_bps` is earned on the signed cash balance (positive cash earns,
    /// negative cash - borrowing - pays at the same rate; choice C6).
    FlatAnnual { long_bps: f64, short_bps: f64, cash_bps: f64 },
}

impl Financing {
    /// Cash accrued over `days` calendar days: `days/365 * 1e-4 * (cash_bps*cash - long_bps*long - short_bps*short)`.
    /// `long_value` and `short_value` are non-negative market values.
    pub fn accrual(&self, days: i64, cash: f64, long_value: f64, short_value: f64) -> f64 {
        match self {
            Financing::None => 0.0,
            Financing::FlatAnnual { long_bps, short_bps, cash_bps } => {
                let frac = days as f64 / 365.0;
                frac * 1e-4 * (cash_bps * cash - long_bps * long_value - short_bps * short_value)
            }
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Financing::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_preset_charges_nothing_and_is_addressable_by_id() {
        assert_eq!(CostModel::ZERO.rate(), 0.0);
        assert_eq!(CostModel::by_id("zero"), Some(CostModel::ZERO));
        assert_eq!(CostModel::by_id("nope"), None);
    }

    #[test]
    fn certification_preset_is_ten_bps() {
        let c = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;
        assert_eq!(c.rate_bps(), 10.0);
        assert!((c.rate() - 0.001).abs() < 1e-18);
        assert!(c.is_valid());
    }

    #[test]
    fn rate_sums_all_components() {
        let c = CostModel { id: "t", commission_bps: 1.0, half_spread_bps: 2.0, slippage_bps: 3.0, source_note: "" };
        assert_eq!(c.rate_bps(), 6.0);
    }

    #[test]
    fn negative_or_nan_components_are_invalid() {
        let mut c = CostModel::ZERO;
        c.slippage_bps = -1.0;
        assert!(!c.is_valid());
        c.slippage_bps = f64::NAN;
        assert!(!c.is_valid());
    }

    #[test]
    fn financing_none_is_exactly_zero() {
        assert_eq!(Financing::None.accrual(30, 5.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn flat_annual_matches_hand_computation() {
        // 10 days, cash 0.4 earning 200 bps, long 0.5 costing 300 bps, short 0.1 costing 100 bps.
        let f = Financing::FlatAnnual { long_bps: 300.0, short_bps: 100.0, cash_bps: 200.0 };
        let want = 10.0 / 365.0 * (0.02 * 0.4 - 0.03 * 0.5 - 0.01 * 0.1);
        assert!((f.accrual(10, 0.4, 0.5, 0.1) - want).abs() < 1e-15);
    }
}
