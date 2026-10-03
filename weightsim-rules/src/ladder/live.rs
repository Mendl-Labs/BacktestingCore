//! W7.4 (council Ruling R10): the LIVE-REALISTIC layer of a library rule, beside (never inside) its certified
//! replication.
//!
//! [`replicate`](super::verify::replicate) stays the certified, same-close, digest-pinned run. [`replicate_live_realistic`]
//! runs the SAME rule on the SAME fixtures under the ladder's base configuration plus a pre-registered
//! [`ExecutionModel`] (per sleeve: ETF trend delay 1 bar, crypto trend delay 0, each plus slippage bps; see
//! `weightsim::preregistered`) and returns a [`LiveRealisticRun`] labelled [`Layer::LiveRealistic`]. It is the
//! customer-facing expected figure; the certified figure is a different value with a different label, and the two
//! are never combined: the live cost-model id does not resolve through `CostModel::by_id`, so a stored live run can
//! never be verified as a certified one, and [`replicate_live_realistic`] refuses the certification model.

use std::fmt;

use weightsim::{
    live_realistic_default_for, simulate, ExecutionModel, Layer, Refusal, SeriesColumns, SimResult, SleeveClass,
    LIVE_REALISTIC_CONFIG_VERSION,
};

use super::checks::SeriesRows;
use super::fixtures::{Fixtures, LadderError};
use super::runner::{rows_from_sim, sleeve_config};
use super::verify::{key_span, library_rule, summarize, ReplicateError, ReplicationConfig, RunSummary};
use crate::adapters::{CryptoTrendRule, EtfTrendRule};

/// Label of the certified layer (what [`super::verify::ReplicationRun`] is).
pub const CERTIFIED_LAYER: &str = "certified";
/// Label of the live-realistic layer.
pub const LIVE_REALISTIC_LAYER: &str = "live_realistic";

/// The pre-registered sleeve class of a library rule, or `None` for an id this crate does not implement.
pub fn sleeve_class_of(rule_id: &str) -> Option<SleeveClass> {
    use weightsim::WeightRule;
    if rule_id == EtfTrendRule.id() {
        Some(SleeveClass::EtfTrend)
    } else if rule_id == CryptoTrendRule.id() {
        Some(SleeveClass::CryptoTrend)
    } else {
        None
    }
}

/// The pre-registered live-realistic execution model of a library rule.
pub fn live_realistic_model_for(rule_id: &str) -> Option<ExecutionModel> {
    sleeve_class_of(rule_id).map(live_realistic_default_for)
}

/// Why a live-realistic run could not be produced.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveRealisticError {
    /// The model is `{ delay 0, slippage 0 }`: that is the certified layer, produced by `replicate`, not here.
    NotLiveRealistic,
    Replicate(ReplicateError),
}

impl From<ReplicateError> for LiveRealisticError {
    fn from(e: ReplicateError) -> Self {
        LiveRealisticError::Replicate(e)
    }
}

impl From<LadderError> for LiveRealisticError {
    fn from(e: LadderError) -> Self {
        LiveRealisticError::Replicate(ReplicateError::Ladder(e))
    }
}

impl fmt::Display for LiveRealisticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiveRealisticError::NotLiveRealistic => {
                write!(f, "the certification execution model is the certified layer (use `replicate`), not a live-realistic one")
            }
            LiveRealisticError::Replicate(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LiveRealisticError {}

/// A live-realistic run, packaged like a [`super::verify::ReplicationRun`] but carrying its layer, its execution
/// model and the pre-registration version, and only a NET series (the customer figure has no gross basis).
#[derive(Clone, Debug)]
pub struct LiveRealisticRun {
    /// Always [`Layer::LiveRealistic`].
    pub layer: Layer,
    pub rule_id: String,
    pub sleeve_class: SleeveClass,
    pub execution_model: ExecutionModel,
    /// [`LIVE_REALISTIC_CONFIG_VERSION`] at the time of the run.
    pub config_version: &'static str,
    /// The certified preset the slippage was overlaid on.
    pub base_cost_model_id: String,
    /// The overlaid preset's id (`<base>+live_realistic`).
    pub cost_model_id: String,
    pub rule_impl_version: String,
    /// Full-resolution net series under the execution model.
    pub net: SeriesColumns,
    pub net_sha256: String,
    pub net_summary: RunSummary,
    /// The counted window in the key's per-bar layout (returns, equity, cost, turnover, weights).
    pub rows: SeriesRows,
    pub refusals: Vec<Refusal>,
    pub manifest_sha256: String,
    pub candles_sha256: String,
}

impl LiveRealisticRun {
    pub fn layer_label(&self) -> &'static str {
        self.layer.label()
    }
}

/// [`replicate_live_realistic_with`] under the default base preset and the rule's pre-registered model.
pub fn replicate_live_realistic(fx: &Fixtures, rule_id: &str) -> Result<LiveRealisticRun, LiveRealisticError> {
    let model = live_realistic_model_for(rule_id)
        .ok_or_else(|| ReplicateError::UnknownRule { rule_id: rule_id.to_string() })?;
    replicate_live_realistic_with(fx, rule_id, &ReplicationConfig::default(), model)
}

/// Run a library rule on the fixtures under the ladder's base configuration (`base`'s cost preset, the key's window,
/// flat start) plus `model`, and package the live-realistic result. Refuses the certification model.
pub fn replicate_live_realistic_with(
    fx: &Fixtures,
    rule_id: &str,
    base: &ReplicationConfig,
    model: ExecutionModel,
) -> Result<LiveRealisticRun, LiveRealisticError> {
    if model.is_certification() {
        return Err(LiveRealisticError::NotLiveRealistic);
    }
    let sleeve_class =
        sleeve_class_of(rule_id).ok_or_else(|| ReplicateError::UnknownRule { rule_id: rule_id.to_string() })?;
    let cost = weightsim::CostModel::by_id(&base.cost_model_id)
        .ok_or_else(|| ReplicateError::UnknownCostPreset { cost_model_id: base.cost_model_id.clone() })?;
    let p = library_rule(fx, rule_id)?;
    let cfg = model.apply_to(&sleeve_config(p.key, cost)).map_err(|e| LadderError::Sim(e.to_string()))?;
    let net: SimResult = simulate(p.panel, &*p.rule, &cfg).map_err(|e| LadderError::Sim(e.to_string()))?;
    let (first, last) = key_span(p.key);
    let m = net.metrics().ok_or_else(|| LadderError::Sim("the run has no counted window".into()))?;
    // The simulator's own flip counter (sign changes of the target between successive decisions); the key's
    // convention (`flips_by_key_convention`) is defined for undelayed runs only.
    let flips: u64 = net.signal_flips.iter().sum();
    let rows = rows_from_sim(&net, first, last, true)?;
    Ok(LiveRealisticRun {
        layer: Layer::LiveRealistic,
        rule_id: net.rule_id.clone(),
        sleeve_class,
        execution_model: model,
        config_version: LIVE_REALISTIC_CONFIG_VERSION,
        base_cost_model_id: base.cost_model_id.clone(),
        cost_model_id: cfg.cost.id.to_string(),
        rule_impl_version: net.rule_impl_version.clone(),
        net_summary: summarize(&m, flips),
        net_sha256: net.series_sha256.clone(),
        net: net.series_columns(),
        rows,
        refusals: net.refusals.clone(),
        manifest_sha256: fx.manifest_sha256.clone(),
        candles_sha256: fx.candles_sha256.clone(),
    })
}
