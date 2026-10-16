//! Rules with memory between decisions (design 3.1): [`StatefulRule`], its object-safe wrapper [`DynRule`] and the
//! per-run handle [`RuleRun`].
//!
//! The SIMULATOR owns the state. A rule never stores anything between decisions itself; it receives `&mut State` on
//! every call together with a [`HistoryView`] cut at the decision bar, so the poisoning and truncation harnesses still
//! bound everything a stateful rule can see. The state is `Clone`, and the run handle uses that: if a step REFUSES,
//! the state is rolled back to what it was before the step (a refused decision must leave no trace, exactly like a
//! refused stateless decision keeps the previous standing target, S-5).
//!
//! Every [`WeightRule`] runs as a stateless `StatefulRule` through the [`Stateless`] adapter, so the whole T1 rule
//! library plugs into books unchanged.

use crate::panel::HistoryView;
use crate::rule::{DecisionSchedule, RebalancePolicy, RuleRefusal, WeightRule};
use std::collections::BTreeMap;

/// What a rule needs from the data of its sleeve (design 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataNeed {
    /// Every instrument of the universe has a bar on every bar the rule is shown (its OWN joint calendar). Implemented.
    CompleteJointCalendar,
    /// The rule tolerates instruments that are not listed yet (cross-sectional universes with late listings) and is
    /// shown an availability mask. NOT implemented in weightsim 0.2: `simulate_book` refuses such a sleeve with
    /// `BookError::Unsupported` instead of approximating (the masked view arrives with the cross-sectional rule, PF4).
    AvailabilityMasked,
}

/// Whether the rule already scales its own risk (design 3.1: "forbids double scaling").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolScaling {
    /// The rule emits raw fractions of sleeve capital.
    None,
    /// The rule applies its own (joint) volatility scaling, e.g. the FX sleeve. A book must not apply a second
    /// volatility-based scaling on top: a `VolScaling::Internal` sleeve together with a volatility-based allocator
    /// (`AllocatorSpec::InverseVol`) is a configuration error.
    Internal,
}

/// A frozen rule with memory. The trait is `Send + Sync`; the state is owned by the simulator, one per run.
pub trait StatefulRule: Send + Sync {
    /// The memory carried between decisions. `Clone` so a refused step can be rolled back and runs can be forked.
    type State: Clone + Send;

    /// Library id, e.g. `crypto_trend_100d`.
    fn id(&self) -> &'static str;
    /// Recorded in every run.
    fn impl_version(&self) -> String;
    /// Canonical symbols in a fixed order; the sleeve's universe must equal this list exactly.
    fn universe(&self) -> &[&'static str];
    /// Compile-time constants (canonical rendering of each), as for [`WeightRule::declared_parameters`].
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::new()
    }
    fn decision_schedule(&self) -> DecisionSchedule;
    fn rebalance_policy(&self) -> RebalancePolicy;
    /// The simulator does not call the rule before this many OWN bars are visible (silent skip, not a refusal).
    fn min_history_bars(&self) -> usize;
    fn data_need(&self) -> DataNeed;
    fn vol_scaling(&self) -> VolScaling;
    /// Fresh state at the start of a run.
    fn init(&self) -> Self::State;
    /// Target weights (signed fractions of SLEEVE capital, universe order) for a decision at the close of the last
    /// visible bar, or a typed refusal. May update `st`; if it refuses the simulator restores the previous state.
    fn step(&self, st: &mut Self::State, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal>;
}

/// Adapter that runs any T1 [`WeightRule`] as a stateless [`StatefulRule`] (`State = ()`), so the whole T1 rule library
/// plugs into books unchanged.
///
/// It is a wrapper, not a blanket `impl<R: WeightRule> StatefulRule for R`, on purpose: with a blanket impl every
/// `WeightRule` type would also carry `StatefulRule::id`/`universe`/... and code that imports both traits (for example
/// `use weightsim::*`) would fail with "multiple applicable items in scope" on every `rule.id()` call. The wrapper keeps
/// the T1 API source-compatible; [`crate::SleeveSpec::from_rule`] applies it for you.
#[derive(Clone, Debug)]
pub struct Stateless<R>(pub R);

impl<R: WeightRule> StatefulRule for Stateless<R> {
    type State = ();
    fn id(&self) -> &'static str {
        WeightRule::id(&self.0)
    }
    fn impl_version(&self) -> String {
        WeightRule::impl_version(&self.0)
    }
    fn universe(&self) -> &[&'static str] {
        WeightRule::universe(&self.0)
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        WeightRule::declared_parameters(&self.0)
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        WeightRule::decision_schedule(&self.0)
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        WeightRule::rebalance_policy(&self.0)
    }
    fn min_history_bars(&self) -> usize {
        WeightRule::min_history_bars(&self.0)
    }
    fn data_need(&self) -> DataNeed {
        DataNeed::CompleteJointCalendar
    }
    fn vol_scaling(&self) -> VolScaling {
        VolScaling::None
    }
    fn init(&self) {}
    fn step(&self, _st: &mut (), h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        self.0.target_weights(h)
    }
}

/// The per-run handle of a rule: owns the state, hides its type.
pub trait RuleRun: Send {
    /// One decision. A refusal leaves the state exactly as it was before the call.
    fn step(&mut self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal>;
}

/// Object-safe view of a [`StatefulRule`], so a book can hold sleeves whose rules have different state types.
pub trait DynRule: Send + Sync {
    fn id(&self) -> &'static str;
    fn impl_version(&self) -> String;
    fn universe(&self) -> &[&'static str];
    fn decision_schedule(&self) -> DecisionSchedule;
    fn rebalance_policy(&self) -> RebalancePolicy;
    fn min_history_bars(&self) -> usize;
    fn data_need(&self) -> DataNeed;
    fn vol_scaling(&self) -> VolScaling;
    /// Start a run: fresh state.
    fn start(&self) -> Box<dyn RuleRun + '_>;
}

struct Run<'a, R: StatefulRule> {
    rule: &'a R,
    state: R::State,
}

impl<R: StatefulRule> RuleRun for Run<'_, R> {
    fn step(&mut self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let backup = self.state.clone();
        match self.rule.step(&mut self.state, h) {
            Ok(w) => Ok(w),
            Err(e) => {
                self.state = backup;
                Err(e)
            }
        }
    }
}

/// The only implementor of [`DynRule`]: wraps a [`StatefulRule`]. (No blanket impl: see [`Stateless`].)
pub struct DynAdapter<R: StatefulRule>(pub R);

impl<R: StatefulRule> DynRule for DynAdapter<R> {
    fn id(&self) -> &'static str {
        StatefulRule::id(&self.0)
    }
    fn impl_version(&self) -> String {
        StatefulRule::impl_version(&self.0)
    }
    fn universe(&self) -> &[&'static str] {
        StatefulRule::universe(&self.0)
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        StatefulRule::decision_schedule(&self.0)
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        StatefulRule::rebalance_policy(&self.0)
    }
    fn min_history_bars(&self) -> usize {
        StatefulRule::min_history_bars(&self.0)
    }
    fn data_need(&self) -> DataNeed {
        StatefulRule::data_need(&self.0)
    }
    fn vol_scaling(&self) -> VolScaling {
        StatefulRule::vol_scaling(&self.0)
    }
    fn start(&self) -> Box<dyn RuleRun + '_> {
        Box::new(Run { rule: &self.0, state: StatefulRule::init(&self.0) })
    }
}
