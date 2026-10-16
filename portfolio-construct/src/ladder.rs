//! The drawdown ladder and the daily-loss limit on f64 (design 3.2 `Ladder`; SignalEngine `rebalancer-risk::overlay`).
//!
//! `Ladder::step(&self, &mut LadderState, equity, day_start) -> LadderDecision` is the f64 restatement of the Decimal
//! state machine that produces the planner's `risk_scale`. The rules, each mirrored from the live overlay:
//!
//! * **Drawdown** is measured from the high-water mark: `loss = hwm - equity`; rung `at` (a fraction) is hit when
//!   `loss >= at * hwm`, INCLUSIVE. The high-water mark only ratchets up while the account may add risk (Active or
//!   Shrunk) and is frozen once halted.
//! * **Ladder.** Rungs ascend strictly. The highest rung hit decides: a shrink rung scales every target by its scale, the
//!   last rung halts and flattens.
//! * **Daily loss.** `day_start - equity >= limit * day_start` (inclusive) halts, whatever the drawdown. `day_start` is
//!   supplied by the caller (the simulator's first observation of the trading day), as in the design's signature.
//! * **Recovery (hysteresis).** A shrink rung is released only when `loss <= at * recovery_fraction * hwm` (inclusive),
//!   one rung at a time; the default recovery fraction is 0.5.
//! * **Halted is sticky**: no equity path leaves it (a backtest reports "halted at t" and counts no new risk; the human
//!   resume rule is out of scope).
//! * **Monotone**: for a fixed prior state, lower equity never yields a larger scale.
//! * **Fail closed**: non-positive or non-finite equity halts.
//!
//! Boundary convention. The live overlay compares exact decimals, so an equity exactly ON a trigger triggers and an
//! equity exactly ON a release level releases. In f64 the same value can land one ulp off (`0.1 * 10000`), so a
//! trigger fires when `loss >= threshold * (1 - EDGE_TOL)` and a release happens when `loss <= threshold * (1 +
//! EDGE_TOL)`, with `EDGE_TOL = 1e-12`: ties behave as in the decimal implementation. Decisions can differ from the
//! Decimal ladder only for an equity within 1e-12 relative of a rung boundary (design 5.2).

use crate::num::EDGE_TOL;

/// What a rung does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RungAction {
    Shrink { scale: f64 },
    HaltFlatten,
}

/// One rung: a drawdown fraction and what it does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rung {
    /// Fraction below the high-water mark, in `(0, 1]`.
    pub at: f64,
    pub action: RungAction,
}

/// The default recovery fraction: a shrink is released when the drawdown is back within half of the rung's trigger.
pub const DEFAULT_RECOVERY_FRACTION: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct Ladder {
    daily_loss_limit: f64,
    rungs: Vec<Rung>,
    recovery_fraction: f64,
}

/// Machine codes of the decisions (mirrors the live `RiskCode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderCode {
    NoAction,
    DrawdownShrink,
    ShrinkHeld,
    PartialRecovery,
    Recovered,
    DrawdownHalt,
    DailyLossHalt,
    AlreadyHalted,
    EquityInvalid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LadderAction {
    None,
    Shrink { scale: f64 },
    HaltFlatten,
}

/// Why a halt happened (the first cause wins; a daily-loss halt is preferred when both hit, as live).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderHalt {
    DailyLoss,
    DrawdownLadder,
    EquityInvalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderStatus {
    Active,
    /// The shrink rung (index into the ladder) in force.
    Shrunk(usize),
    Halted(LadderHalt),
}

/// The state the ladder carries between bars.
#[derive(Clone, Debug, PartialEq)]
pub struct LadderState {
    pub hwm: Option<f64>,
    pub status: LadderStatus,
}

impl LadderState {
    pub fn new() -> Self {
        LadderState { hwm: None, status: LadderStatus::Active }
    }
    /// True once a halt rung or the daily-loss limit has fired. Sticky: only a human resume leaves a halt live, and a
    /// backtest has no resume.
    pub fn is_halted(&self) -> bool {
        matches!(self.status, LadderStatus::Halted(_))
    }
}

impl Default for LadderState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderDecision {
    /// The multiplier for every target: 1, a rung's scale, or 0 (halted).
    pub scale: f64,
    pub action: LadderAction,
    /// The primary reason (the live decision can carry several; the halts also list `DrawdownHalt` when both hit, see
    /// `both_halts`).
    pub code: LadderCode,
    /// True when a daily-loss halt and the halt rung both hit on this step.
    pub both_halts: bool,
    /// The shrink rung in force after the step.
    pub rung: Option<usize>,
    /// Drawdown from the high-water mark as a fraction (0 when there is no mark).
    pub drawdown: f64,
    /// Loss since day-start as a fraction (negative = a gain; 0 without a usable day-start).
    pub daily_loss: f64,
}

/// Why a ladder could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderError(pub &'static str);

impl Ladder {
    /// Checks the invariants of the live `RiskPolicy::new`: fractions in `(0, 1]`; rungs strictly ascending, ending in a
    /// halt, only the last rung halts, shrink scales in `(0, 1)` and never scaling back up.
    pub fn new(daily_loss_limit: f64, rungs: Vec<Rung>, recovery_fraction: f64) -> Result<Self, LadderError> {
        let unit = |v: f64| v.is_finite() && v > 0.0 && v <= 1.0;
        if !unit(daily_loss_limit) {
            return Err(LadderError("daily_loss_limit must be in (0, 1]"));
        }
        if !unit(recovery_fraction) {
            return Err(LadderError("recovery_fraction must be in (0, 1]"));
        }
        if rungs.is_empty() {
            return Err(LadderError("the ladder needs at least one rung"));
        }
        let last = rungs.len() - 1;
        let mut prev_at = 0.0;
        let mut prev_scale = 1.0;
        for (i, r) in rungs.iter().enumerate() {
            if !(r.at.is_finite() && r.at > prev_at && r.at <= 1.0) {
                return Err(LadderError("ladder rungs must be strictly ascending fractions in (0, 1]"));
            }
            prev_at = r.at;
            match r.action {
                RungAction::Shrink { scale } => {
                    if i == last {
                        return Err(LadderError("the last rung must be halt_flatten"));
                    }
                    if !(scale > 0.0 && scale < 1.0) {
                        return Err(LadderError("a shrink scale must be in (0, 1)"));
                    }
                    if scale > prev_scale {
                        return Err(LadderError("a later shrink rung cannot scale back up"));
                    }
                    prev_scale = scale;
                }
                RungAction::HaltFlatten => {
                    if i != last {
                        return Err(LadderError("only the last rung may be halt_flatten"));
                    }
                }
            }
        }
        Ok(Ladder { daily_loss_limit, rungs, recovery_fraction })
    }

    pub fn daily_loss_limit(&self) -> f64 {
        self.daily_loss_limit
    }
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }
    pub fn recovery_fraction(&self) -> f64 {
        self.recovery_fraction
    }

    /// One evaluation: fold in the equity (ratchet the high-water mark), decide, and update `st`.
    /// `day_start` is the equity at the first observation of the account-local trading day (`<= 0` disables the daily
    /// check for this step).
    pub fn step(&self, st: &mut LadderState, equity: f64, day_start: f64) -> LadderDecision {
        // Halted is sticky, whatever the equity.
        if st.is_halted() {
            return LadderDecision {
                scale: 0.0,
                action: LadderAction::None,
                code: LadderCode::AlreadyHalted,
                both_halts: false,
                rung: None,
                drawdown: 0.0,
                daily_loss: 0.0,
            };
        }
        if !(equity.is_finite() && equity > 0.0) {
            st.status = LadderStatus::Halted(LadderHalt::EquityInvalid);
            return LadderDecision {
                scale: 0.0,
                action: LadderAction::HaltFlatten,
                code: LadderCode::EquityInvalid,
                both_halts: false,
                rung: None,
                drawdown: 0.0,
                daily_loss: 0.0,
            };
        }
        // observe: the mark ratchets up while the account may add risk.
        let hwm = match st.hwm {
            Some(h) if h >= equity => h,
            _ => equity,
        };
        st.hwm = Some(hwm);
        let drawdown = fraction_of(hwm, equity);
        let mut daily_loss = 0.0;
        let mut daily_halt = false;
        if day_start > 0.0 && day_start.is_finite() {
            daily_loss = fraction_of(day_start, equity);
            daily_halt = breached(day_start, equity, self.daily_loss_limit);
        }
        // The highest rung hit.
        let mut raw: Option<usize> = None;
        for (i, r) in self.rungs.iter().enumerate() {
            if breached(hwm, equity, r.at) {
                raw = Some(i);
            }
        }
        let drawdown_halt = raw.is_some_and(|i| self.rungs[i].action == RungAction::HaltFlatten);
        if daily_halt || drawdown_halt {
            let why = if daily_halt { LadderHalt::DailyLoss } else { LadderHalt::DrawdownLadder };
            st.status = LadderStatus::Halted(why);
            return LadderDecision {
                scale: 0.0,
                action: LadderAction::HaltFlatten,
                code: if daily_halt { LadderCode::DailyLossHalt } else { LadderCode::DrawdownHalt },
                both_halts: daily_halt && drawdown_halt,
                rung: None,
                drawdown,
                daily_loss,
            };
        }
        // Shrink posture, with hysteresis on the way back.
        let last_shrink = self.rungs.len().checked_sub(2);
        let entered = match st.status {
            LadderStatus::Shrunk(r) => Some(r),
            _ => None,
        };
        let mut held = match (entered, last_shrink) {
            (Some(c), Some(l)) => Some(c.min(l)),
            _ => None,
        };
        while let Some(c) = held {
            if recovered(hwm, equity, self.rungs[c].at, self.recovery_fraction) {
                held = c.checked_sub(1);
            } else {
                break;
            }
        }
        let target = match (raw, held) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        match target {
            None => {
                let code = if entered.is_some() { LadderCode::Recovered } else { LadderCode::NoAction };
                st.status = LadderStatus::Active;
                LadderDecision {
                    scale: 1.0,
                    action: LadderAction::None,
                    code,
                    both_halts: false,
                    rung: None,
                    drawdown,
                    daily_loss,
                }
            }
            Some(i) => {
                let RungAction::Shrink { scale } = self.rungs[i].action else {
                    // Unreachable (a halt rung returned above); fail closed anyway.
                    st.status = LadderStatus::Halted(LadderHalt::DrawdownLadder);
                    return LadderDecision {
                        scale: 0.0,
                        action: LadderAction::HaltFlatten,
                        code: LadderCode::DrawdownHalt,
                        both_halts: false,
                        rung: None,
                        drawdown,
                        daily_loss,
                    };
                };
                let newly = raw == Some(i) && entered != Some(i);
                let stepped_down = entered.is_some_and(|c| i < c);
                let code = if newly {
                    LadderCode::DrawdownShrink
                } else if stepped_down {
                    LadderCode::PartialRecovery
                } else {
                    LadderCode::ShrinkHeld
                };
                st.status = LadderStatus::Shrunk(i);
                LadderDecision {
                    scale,
                    action: LadderAction::Shrink { scale },
                    code,
                    both_halts: false,
                    rung: Some(i),
                    drawdown,
                    daily_loss,
                }
            }
        }
    }
}

/// `reference - equity >= fraction * reference`, inclusive, tie-preserving (see the module docs).
fn breached(reference: f64, equity: f64, fraction: f64) -> bool {
    let loss = reference - equity;
    let threshold = fraction * reference;
    loss >= threshold - threshold.abs() * EDGE_TOL
}

/// `reference - equity <= at * recovery * reference`, inclusive, tie-preserving.
fn recovered(reference: f64, equity: f64, at: f64, recovery: f64) -> bool {
    let loss = reference - equity;
    let threshold = (at * recovery) * reference;
    loss <= threshold + threshold.abs() * EDGE_TOL
}

/// Loss as a fraction of the reference (negative for a gain).
fn fraction_of(reference: f64, equity: f64) -> f64 {
    (reference - equity) / reference
}
