//! W7.4 (council Ruling R10): the execution model of a run, and the two LAYERS a result can belong to.
//!
//! The certified replication stays same-close and digest-pinned: a decision at the close of `t` is filled at that
//! close, at the declared cost preset, nothing else. That is [`ExecutionModel::certification`] (`delay 0`, `slippage 0`),
//! and applying it to a configuration is the identity, so every pinned digest is untouched by construction.
//!
//! A SEPARATE, labelled LIVE-REALISTIC layer applies a per-sleeve execution delay plus slippage: a decision made at the
//! close of `t` is executed at the close of `t + delay_bars`, at that close plus slippage. Slippage is charged on
//! turnover in the ledger (`cost = traded_notional x (preset rate + slippage)`), which is exactly "that close +/-
//! slippage" on the traded notional and keeps the Layer C cost identity exact (`cost_t == rate x traded_t`). The
//! per-sleeve defaults are pre-registered in [`crate::preregistered`] and changeable only by amendment.
//!
//! A live-realistic result is never mixed with a certified one: it carries [`Layer::LiveRealistic`], its cost-model id
//! carries the `+live_realistic` suffix ([`crate::CostModel::with_live_slippage`]), and `CostModel::by_id` does not
//! resolve such an id, so a verifier can never mistake the one for the other.

use crate::costs::CostModel;
use crate::sim::{SimConfig, SimError};
use std::fmt;

/// Which layer a result belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Same-close fills, declared cost preset, digest-pinned.
    Certified,
    /// Per-sleeve execution delay plus pre-registered slippage; the customer-facing expected range.
    LiveRealistic,
}

impl Layer {
    /// The label a result is tagged with (`"certified"` / `"live_realistic"`).
    pub fn label(self) -> &'static str {
        match self {
            Layer::Certified => "certified",
            Layer::LiveRealistic => "live_realistic",
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// How decisions are executed: a decision at the close of `t` is executed at the close of `t + delay_bars` (the
/// sleeve's own bars, [`SimConfig::execution_delay_bars`]), at that close plus `slippage_bps` on the traded notional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutionModel {
    pub delay_bars: u32,
    pub slippage_bps: f64,
}

impl ExecutionModel {
    /// The certified path: no delay, no slippage. [`Self::apply_to`] is the identity for this model.
    pub const fn certification() -> Self {
        ExecutionModel { delay_bars: 0, slippage_bps: 0.0 }
    }

    pub const fn new(delay_bars: u32, slippage_bps: f64) -> Self {
        ExecutionModel { delay_bars, slippage_bps }
    }

    /// `true` for `{ 0, 0 }` exactly.
    pub fn is_certification(&self) -> bool {
        self.delay_bars == 0 && self.slippage_bps == 0.0
    }

    pub fn layer(&self) -> Layer {
        if self.is_certification() {
            Layer::Certified
        } else {
            Layer::LiveRealistic
        }
    }

    pub fn is_valid(&self) -> bool {
        self.slippage_bps.is_finite() && self.slippage_bps >= 0.0
    }

    /// `cfg` under this execution model. The certification model returns `cfg` unchanged (a clone, every field
    /// bit-identical, so the digest of the run is the certified digest). Any other model SETS
    /// `execution_delay_bars = delay_bars` and replaces the cost preset by the same preset plus `slippage_bps`
    /// ([`CostModel::with_live_slippage`]). Refused: a non-finite or negative slippage, and a configuration whose
    /// cost preset already carries the live-realistic overlay (the layer is applied once).
    pub fn apply_to(&self, cfg: &SimConfig) -> Result<SimConfig, SimError> {
        if !self.is_valid() {
            return Err(SimError::BadConfig(format!(
                "execution model slippage {} is not finite and >= 0",
                self.slippage_bps
            )));
        }
        if self.is_certification() {
            return Ok(cfg.clone());
        }
        if cfg.cost.is_live_realistic() {
            return Err(SimError::BadConfig(format!(
                "cost preset `{}` already carries the live-realistic overlay; the layer is applied once",
                cfg.cost.id
            )));
        }
        Ok(SimConfig {
            execution_delay_bars: self.delay_bars as usize,
            cost: cfg.cost.with_live_slippage(self.slippage_bps),
            ..cfg.clone()
        })
    }
}

impl fmt::Display for ExecutionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (delay {} bar(s), slippage {} bps)", self.layer(), self.delay_bars, self.slippage_bps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::costs::Financing;
    use crate::rule::OnRefusal;

    #[test]
    fn certification_model_is_the_identity_on_any_config() {
        let cfg = SimConfig {
            start: None,
            end: None,
            initial_equity: 3.0,
            cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE,
            financing: Financing::FlatAnnual { long_bps: 1.0, short_bps: 2.0, cash_bps: 3.0 },
            on_refusal: OnRefusal::HoldPrevious,
            execution_delay_bars: 0,
            risk_scale: 0.5,
            max_gross: Some(2.0),
        };
        let m = ExecutionModel::certification();
        assert!(m.is_certification() && m.layer() == Layer::Certified);
        let out = m.apply_to(&cfg).unwrap();
        assert_eq!(out.cost, cfg.cost);
        assert_eq!(out.cost.id, "certification_flat_10bps_per_side");
        assert_eq!(out.execution_delay_bars, 0);
        assert_eq!(out.initial_equity, 3.0);
        assert_eq!(out.financing, cfg.financing);
        assert_eq!(out.on_refusal, cfg.on_refusal);
        assert_eq!(out.risk_scale, 0.5);
        assert_eq!(out.max_gross, Some(2.0));
    }

    #[test]
    fn a_live_model_sets_the_delay_and_overlays_the_slippage_once() {
        let m = ExecutionModel::new(1, 5.0);
        assert_eq!(m.layer(), Layer::LiveRealistic);
        assert_eq!(Layer::LiveRealistic.label(), "live_realistic");
        let cfg = SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() };
        let out = m.apply_to(&cfg).unwrap();
        assert_eq!(out.execution_delay_bars, 1);
        assert_eq!(out.cost.id, "certification_flat_10bps_per_side+live_realistic");
        assert_eq!(out.cost.rate_bps(), 15.0);
        assert!(out.cost.is_live_realistic() && !cfg.cost.is_live_realistic());
        assert!(matches!(m.apply_to(&out), Err(SimError::BadConfig(_))));
        assert!(matches!(ExecutionModel::new(0, -1.0).apply_to(&cfg), Err(SimError::BadConfig(_))));
        assert!(matches!(ExecutionModel::new(0, f64::NAN).apply_to(&cfg), Err(SimError::BadConfig(_))));
        assert_eq!(m.to_string(), "live_realistic (delay 1 bar(s), slippage 5 bps)");
    }
}
