//! Mandate limits as the construction layer sees them (design 3.2 `Limits`).
//!
//! All caps are fractions of the CAPITAL BASE (`min(equity, allocated)`), exactly as the guard measures them, so a plan
//! and the guard that checks it can never disagree about what "25% of equity" means. A breach fails the WHOLE book
//! (council R1/R2: refuse, never clip); the only scaling permitted is the constant risk scale fixed at plan approval.

use std::collections::BTreeMap;

/// How a breach of a limit is treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitPolicy {
    /// Council R1/R2: every limit is checked on the whole target book and any breach refuses the whole book. This is the
    /// backtester's default and the behaviour the design specifies.
    RefuseWholeBook,
    /// The live planner as it is today (deviation ledger item 1): only a plan with a signed sleeve is checked, and only
    /// against the gross cap; position, class and net limits are enforced per order by the guard, which drops the
    /// offending order and keeps the rest. Selecting this mode reproduces the planner's plan-level behaviour so the
    /// parity tests can separate "planner and spec disagree" from "planner has not implemented R1 yet".
    PlannerFaithful,
}

/// Mandate limits. Caps are fractions of the capital base; `f64::INFINITY` means "no cap".
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Largest `|target|` in any single instrument (`exposure.max_position`).
    pub max_position: f64,
    /// Largest sum of `|target|` in an asset class, keyed by lower-case class name (`exposure.max_asset_class`).
    pub max_asset_class: BTreeMap<String, f64>,
    /// Largest sum of `|target|` over the book (`exposure.max_gross`).
    pub max_gross: f64,
    /// Largest `|sum of targets|` (`exposure.max_net`).
    pub max_net: f64,
    /// May any target be negative (`universe.shorting`)?
    pub shorting: bool,
    /// `universe.leverage_max_gross`. The effective gross cap is `min(max_gross, leverage_max_gross)`, which is what the
    /// planner's `Policy::compile` stores as its `max_gross`.
    pub leverage_max_gross: f64,
    /// Breach handling, see [`LimitPolicy`].
    pub policy: LimitPolicy,
}

impl Limits {
    /// No cap of any kind, shorting allowed: what a research run that only wants the sizing uses.
    pub fn unlimited() -> Self {
        Limits {
            max_position: f64::INFINITY,
            max_asset_class: BTreeMap::new(),
            max_gross: f64::INFINITY,
            max_net: f64::INFINITY,
            shorting: true,
            leverage_max_gross: f64::INFINITY,
            policy: LimitPolicy::RefuseWholeBook,
        }
    }

    /// The mandate baseline the planner tests use, generalised: long-only, one times gross and net, no per-position or
    /// class cap.
    pub fn long_only_unit() -> Self {
        Limits { max_gross: 1.0, max_net: 1.0, shorting: false, leverage_max_gross: 1.0, ..Limits::unlimited() }
    }

    /// `min(max_gross, leverage_max_gross)`: the cap the plan-level gross check uses.
    pub fn effective_max_gross(&self) -> f64 {
        self.max_gross.min(self.leverage_max_gross)
    }

    pub fn with_max_gross(mut self, cap: f64) -> Self {
        self.max_gross = cap;
        self
    }
    pub fn with_leverage_max_gross(mut self, cap: f64) -> Self {
        self.leverage_max_gross = cap;
        self
    }
    pub fn with_max_net(mut self, cap: f64) -> Self {
        self.max_net = cap;
        self
    }
    pub fn with_max_position(mut self, cap: f64) -> Self {
        self.max_position = cap;
        self
    }
    pub fn with_class_cap(mut self, class: &str, cap: f64) -> Self {
        self.max_asset_class.insert(class.trim().to_lowercase(), cap);
        self
    }
    pub fn with_shorting(mut self, shorting: bool) -> Self {
        self.shorting = shorting;
        self
    }
    pub fn with_policy(mut self, policy: LimitPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// `Err` text when a cap is negative or NaN.
    pub fn validate(&self) -> Result<(), &'static str> {
        let ok = |v: f64| !v.is_nan() && v >= 0.0;
        if !ok(self.max_position) || !ok(self.max_gross) || !ok(self.max_net) || !ok(self.leverage_max_gross) {
            return Err("limits must be non-negative numbers");
        }
        if self.max_asset_class.values().any(|v| !ok(*v)) {
            return Err("asset-class caps must be non-negative numbers");
        }
        Ok(())
    }
}
