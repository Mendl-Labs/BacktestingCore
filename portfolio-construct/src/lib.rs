//! `portfolio-construct`: the shared portfolio-construction SPECIFICATION, on f64 (phase PF2 of
//! `product-mandate/PORTFOLIO_FIRST_BACKTESTER_DESIGN.md`, sections 3.2 and 5).
//!
//! Zero dependencies, no I/O, no clock, no price source: the same inputs give bit-identical outputs. It is the one place
//! in the platform where "what the account should hold, and what to trade to get there" is written as code that both
//! the backtester (`weightsim` v0.2, through `simulate_book`) and the live rebalancer's parity tests (SignalEngine, PF3)
//! can call. It is NOT the live planner: the planner's exact-Decimal arithmetic, guard, buying-power source, tags and
//! idempotency stay in SignalEngine. The claim this crate supports is "same rules and same sizing specification, equal
//! to a stated tolerance", never "backtest equals live by construction" (design 1.4).
//!
//! # Public API
//! * [`construct`]: `ConstructInputs -> Result<ConstructOutput, ConstructRefusal>` (see the module docs of
//!   [`construct`](mod@construct) for the ten steps).
//! * [`Limits`] and [`LimitPolicy`]; [`MarginModel`] with [`NoMargin`], [`OandaMargin`] (R3), [`AlpacaRegT`] (R4);
//!   [`TradeFilter`]; [`QuantityRounder`] with [`LotRounder`] and [`ExactUnits`].
//! * [`Ladder`] (drawdown ladder and daily-loss limit, f64), [`AllocatorSpec`] / [`AllocatorState`]
//!   (`Fixed`, `Equal`, `InverseVol`), [`schedule::due`] and [`BookCadence`].
//!
//! # Design choices (numbered; the README repeats them)
//! 1. **Signature.** `construct` and the input/output types follow design 3.2. Three fields are added to
//!    `ConstructInputs` because the design leaves them to the planner: `funding` (cash or buying-power budget),
//!    `target_dp` (round each target toward zero to N decimals) and `unmanaged_gross`. Sleeve weights are sparse
//!    `(instrument index, weight)` pairs so that "named with weight 0" (sell the held position) differs from "not
//!    named" (leave it alone), as in the planner.
//! 2. **Decimal comparability.** Targets are `(capital_base * raw) * risk_scale`, optionally rounded toward zero at 8
//!    decimals (`target_dp = Some(8)`, the planner's quantum). Compared with the planner's `target_notional` the f64
//!    value is within ONE quantum (1e-8 currency units): a value computed as 0.30000000000000004 vs exact 0.3 lands on
//!    the same quantum thanks to a 4-ulp snap ([`floor_dp`]); a residual disagreement is at most one quantum.
//!    Research runs use `None` (unrounded), because the answer key is unrounded f64 on equity 1.0.
//! 3. **Tie-preserving boundaries.** Every cap and threshold is compared with a 1e-12 relative tolerance ([`EDGE_TOL`])
//!    in the direction that lets an exact decimal tie behave as the planner does (a delta of exactly 10 trades; gross
//!    exactly at the cap is permitted; a value 1e-9 over the cap refuses). Inside that 1e-12 band the answer can differ
//!    from the Decimal planner; the parity tests allow a 1e-9 band (design 5.4 test 1).
//! 4. **Whole-book refusal.** Any breached limit refuses the whole book ([`ConstructRefusal`]), never clips (R1/R2).
//!    [`LimitPolicy::PlannerFaithful`] reproduces today's planner (only signed plans, only the gross cap; the guard
//!    denies per order); the difference is item 1 of the known-deviation ledger.
//! 5. **Order independence.** Sleeves are combined in id order, cross-instrument sums use a sorted-order sum, so
//!    permuting sleeves or instruments does not change a bit of the targets or of gross/net/margin.
//! 6. **Ladder.** `step(&mut state, equity, day_start)`; day-start is supplied by the caller as in the design. A halt is
//!    sticky and the state records why.
//! 7. **Allocators** are static between reviews; `InverseVol` is the key's definition (60-bar sample deviation of each
//!    sleeve's own-calendar returns, shares proportional to `1/sd`) with an optional deviation floor.
//! 8. **Schedule** carries its own civil-date type (no `chrono`); `weightsim` converts its `Date` when it integrates. Since 0.2
//!    it also carries the LIVE monthly cadence, [`Cadence::DecisionPending`] (council Ruling 1: evaluate every run, plan iff
//!    `D_computable > D_acted`), as pure functions; [`Cadence::CalendarMonthEnd`] stays as the documented wall-clock predicate
//!    of the U3 defect and must not be used for the live ETF cadence.
//!
//! # Known deviations from the live planner and from the PF0 key (the ledger)
//! Planner (SignalEngine `main` at 4937a43): (1) limits: the planner checks only the gross cap and only for signed plans
//! and lets the guard deny single orders; this crate refuses the whole book on any breach (R1/R2 not yet in the planner).
//! (2) The guard's per-order checks (turnover, orders per day, price staleness, universe, venue leverage, position
//! units, halted/expired mandate) are not modelled. (3) Lot rounding is a caller-supplied table, not the adapters'
//! `prepare_order`. (4) Quantities are f64: at a lot boundary the floor can differ by one venue quantum. (5) A zero
//! risk scale is allowed (flatten); the planner refuses 0. (6) Fees are `ceil8(notional * rate)` on f64. (7) Shorting is
//! a whole-book refusal here; the planner's guard denies the short order and places the rest.
//! Book key (Amendment 12 D1-D8): D1 the key is unrounded f64 (`target_dp = None` reproduces it); D2 lots (the key holds
//! fractional units: `rounding = None`); D3 cash: the key's `certification` policy is `Funding::Unconstrained`, its
//! `budget` policy is `Funding::Cash` with reserve 0 and the fee equal to the cost rate; D4 gross cap: whole-book refusal
//! (`LimitPolicy::RefuseWholeBook`) equals the key's hold-previous refusal; D5 the trade filter is expressed in
//! currency units, the caller converts (`CAPITAL0`); D6 fills and prices are the simulator's; D7 cadence timing
//! (`schedule`); D8 ladder, margin, financing, shorting availability and data gates are outside the key.

pub mod alloc;
pub mod construct;
pub mod filter;
pub mod ladder;
pub mod limits;
pub mod margin;
mod num;
pub mod rounding;
pub mod schedule;

pub use alloc::{sample_std, AllocError, AllocatorSpec, AllocatorState, FreezeRule, HoldReason, ReviewOutcome};
pub use construct::{
    approval_risk_scale, capital_base, construct, Binding, BindingKind, ConstructInputs, ConstructOutput,
    ConstructRefusal, Funding, InputError, InstrumentFacts, LegKind, Line, RiskScale, Skip, SleeveTargets, TradeIntent,
    WeightBounds,
};
pub use filter::{SkipReason, TradeFilter};
pub use ladder::{
    Ladder, LadderAction, LadderCode, LadderDecision, LadderError, LadderHalt, LadderState, LadderStatus, Rung,
    RungAction, DEFAULT_RECOVERY_FRACTION,
};
pub use limits::{LimitPolicy, Limits};
pub use margin::{AlpacaRegT, BuyingPower, BuyingPowerFactor, MarginModel, NoMargin, OandaMargin};
pub use num::{ceil_dp, floor_dp, round_toward_zero_dp, stable_sum, EDGE_TOL};
pub use rounding::{ExactUnits, LotRounder, LotRule, QuantityRounder, Side, SizeRefusal};
pub use schedule::{
    advance_acted, closed_bars, computable_decision_date, decision_pending, due, due_on, evaluate_decision, plan_flags,
    plan_flags_for, BarsError, BookCadence, Cadence, CivilDate, DecisionEvaluation, DueInputs,
};

/// The reference FX rule clips every weight to +-3 AFTER its volatility scaling (observed maximum 1.87), so a per-instrument
/// sleeve weight above 3 is not a reproduction of any documented rule; a sanity ceiling on the INPUT, never a permission
/// (planner `MAX_ABS_WEIGHT_CAP`).
pub const MAX_ABS_WEIGHT_CAP: f64 = 3.0;
/// The planner's default absolute minimum trade, in account-currency units (`RunConfig`).
pub const DEFAULT_MIN_TRADE_ABS: f64 = 10.0;
/// The planner's default minimum trade as a fraction of the target (`RunConfig`).
pub const DEFAULT_MIN_TRADE_PCT: f64 = 0.02;
/// The Decimal quantum of the planner's target notionals (8 decimals).
pub const TARGET_DP: u32 = 8;
/// Council R2: the reference FX sleeve's p90 gross exposure needed by the rule (multiples of equity).
pub const R2_REFERENCE_P90_GROSS: f64 = 4.41;
/// Council R2: headroom applied to the cap when fixing the approval-time scale.
pub const R2_HEADROOM: f64 = 0.9;
/// Council R3: default ceiling on OANDA margin used, as a fraction of NAV.
pub const OANDA_DEFAULT_MARGIN_CEILING: f64 = 0.50;
/// Council R4: the floor of Alpaca's opening margin rate for marginable securities.
pub const ALPACA_MIN_OPENING_MARGIN_RATE: f64 = 0.50;
/// The planner's per-buy fee rounding slack (each fee is rounded UP by at most 1e-8).
pub const FEE_SLACK_PER_BUY: f64 = 1e-8;
