//! The rule interface: a pure function from the history of ALL assets in the universe up to and including bar `t`
//! to a signed target weight vector (design 2.1, 2.3). The simulator, not the rule, owns the decision schedule and
//! the rebalance policy's accounting; a rule only declares which it needs.

use crate::date::Date;
pub use crate::panel::HistoryView;
use std::collections::BTreeMap;
use std::fmt;

/// When the simulator asks the rule for a decision (design S-3). The schedule is a calendar input, never a rule input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionSchedule {
    /// Every bar.
    Daily,
    /// The last bar of each calendar month present in the joint calendar (the key's `month_end_dates`). The final bar
    /// of the panel counts as a month-end even if the month is incomplete (that is the key's definition; see the
    /// crate docs, choice C7).
    LastBarOfMonth,
}

impl DecisionSchedule {
    /// Is bar `t` a decision bar? Uses only the calendar (dates), never prices.
    pub fn is_decision_bar(self, dates: &[Date], t: usize) -> bool {
        match self {
            DecisionSchedule::Daily => true,
            DecisionSchedule::LastBarOfMonth => t + 1 == dates.len() || !dates[t].same_month(dates[t + 1]),
        }
    }
}

/// How the book is traded between decisions (design S-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebalancePolicy {
    /// Trade to target only on the bar a new target becomes effective; units drift in between (S1).
    OnDecision,
    /// After every bar's mark-to-market, trade back to the standing target weights (S2, S3).
    EveryBar,
}

/// What the simulator does when a rule refuses (design S-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnRefusal {
    /// Keep the previous standing target for the WHOLE book (never one leg); record the refusal.
    HoldPrevious,
    /// Fail the run with the refusal (certification default). The only tolerated refusal is `Warmup` before any
    /// decision has ever succeeded.
    Abort,
}

/// Why a rule refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalKind {
    /// Not enough history yet. Tolerated under `Abort` only until the first successful decision.
    Warmup,
    /// Data-quality refusal (gap, stale bar, incomplete series).
    Data,
    /// Anything else.
    Other,
}

/// A typed refusal. `code` is the rule's own stable error code (so a live/backtest divergence can be traced).
#[derive(Clone, Debug, PartialEq)]
pub struct RuleRefusal {
    pub kind: RefusalKind,
    pub code: &'static str,
    pub message: String,
}

impl RuleRefusal {
    pub fn new(kind: RefusalKind, code: &'static str, message: impl Into<String>) -> Self {
        RuleRefusal { kind, code, message: message.into() }
    }
    pub fn warmup(message: impl Into<String>) -> Self {
        RuleRefusal::new(RefusalKind::Warmup, "insufficient_history", message)
    }
    pub fn data(code: &'static str, message: impl Into<String>) -> Self {
        RuleRefusal::new(RefusalKind::Data, code, message)
    }
}

impl fmt::Display for RuleRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} refusal [{}]: {}", self.kind, self.code, self.message)
    }
}

impl std::error::Error for RuleRefusal {}

/// A frozen weight-target rule. Fixed in reviewed code, never caller-supplied Python (design 2.1).
pub trait WeightRule: Send + Sync {
    /// Library id, e.g. `crypto_trend_100d`.
    fn id(&self) -> &'static str;
    /// Recorded in every run.
    fn impl_version(&self) -> String;
    /// Canonical symbols in a fixed order; the panel's columns must equal this list exactly.
    fn universe(&self) -> &[&'static str];
    /// Compile-time constants (the Engine asserts them equal to the library entry's `parameters`). Values are the
    /// canonical JSON rendering of the constant. (Design says `serde_json::Value`; T1 keeps the crate dependency-free,
    /// choice C10.)
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::new()
    }
    fn decision_schedule(&self) -> DecisionSchedule;
    fn rebalance_policy(&self) -> RebalancePolicy;
    /// The simulator does not call the rule before this many bars are visible (silent skip, not a refusal).
    fn min_history_bars(&self) -> usize;
    /// Target weights (signed fractions of sleeve equity, universe order) for a decision at the close of the last
    /// visible bar, or a typed refusal.
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal>;
}

/// The name used by the task text; identical to [`WeightRule`].
pub use self::WeightRule as Rule;
