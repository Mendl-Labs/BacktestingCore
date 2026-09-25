//! `construct`: sleeve targets, shares, capital base and risk scale in; a sized target book, the trades that reach it,
//! the constraints that bind and, when a limit is breached, a WHOLE-BOOK refusal out (design 3.2).
//!
//! Order of operations (identical to the live planner's, plus the explicit steps the planner leaves to the guard):
//!
//! 1. validate the inputs (shares, weights, parameters);
//! 2. `capital_base = min(equity, allocated_capital)` (equity when there is no allocation cap);
//! 3. combine `raw_i = SUM_s share_s * w_(s,i)`, SIGNED, so opposite sleeves net; the terms are added in sleeve-id
//!    order, so the result never depends on the order the caller lists the sleeves;
//! 4. `target_notional_i = (capital_base * raw_i) * (approval_constant * ladder)`, optionally rounded toward zero to
//!    `target_dp` decimals (8 in live-faithful runs, `None` in research runs);
//! 5. limits (position, asset class, gross, net, shorting): any breach refuses the WHOLE book (council R1/R2), nothing
//!    is ever clipped;
//! 6. margin model: margin used and its ceiling; a breach refuses the whole book;
//! 7. trade filter on the per-instrument delta (`TradeFilter`), drop reasons recorded;
//! 8. optional quantity rounding by venue rules (always DOWN);
//! 9. reductions before increases, a trade that crosses zero is two legs (close, then open);
//! 10. funding: increases are limited by cash (after a reserve and fees) or by the broker's buying power, all of them
//!     scaled by ONE common factor when they do not fit.
//!
//! Instruments the sleeves do not name are left alone (their value counts in `unmanaged_gross`). The crate never reads a
//! clock, a price source or a random number; the same inputs give bit-identical outputs.

use crate::filter::{SkipReason, TradeFilter};
use crate::limits::{LimitPolicy, Limits};
use crate::margin::MarginModel;
use crate::num::{ceil_dp, exceeds, round_toward_zero_dp, stable_sum};
use crate::rounding::{QuantityRounder, Side};
use std::collections::BTreeMap;

// -------------------------------------------------------------------------------------------------------------------
// Inputs
// -------------------------------------------------------------------------------------------------------------------

/// How a sleeve's weights are bounded (mirrors the planner's `WeightBounds`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightBounds {
    /// Each weight in `[0, 1]`, weights summing to at most 1. Shorts and leverage are impossible for the sleeve.
    LongOnlyUnit,
    /// Signed: each `|weight| <= max_abs_weight`, with `max_abs_weight` in `(0, MAX_ABS_WEIGHT_CAP]`; no sum rule.
    Signed { max_abs_weight: f64 },
}

/// One sleeve's target weights for one construction step.
#[derive(Clone, Debug, PartialEq)]
pub struct SleeveTargets {
    /// Unique sleeve id (trimmed).
    pub id: String,
    /// Fraction of capital allocated to the sleeve, in `(0, 1]`; all shares sum to at most 1.
    pub share: f64,
    /// `(instrument index, weight)` pairs: signed fractions of the SLEEVE's capital. An instrument that appears here
    /// with weight 0 is MANAGED (its held position is sold); an instrument that appears in no sleeve is left alone.
    pub weights: Vec<(usize, f64)>,
    pub bounds: WeightBounds,
}

impl SleeveTargets {
    /// A long-only sleeve (the default).
    pub fn long_only(id: &str, share: f64, weights: Vec<(usize, f64)>) -> Self {
        SleeveTargets { id: id.trim().to_string(), share, weights, bounds: WeightBounds::LongOnlyUnit }
    }
    /// A signed sleeve with `|weight| <= max_abs_weight`.
    pub fn signed(id: &str, share: f64, max_abs_weight: f64, weights: Vec<(usize, f64)>) -> Self {
        SleeveTargets { id: id.trim().to_string(), share, weights, bounds: WeightBounds::Signed { max_abs_weight } }
    }
    fn is_signed(&self) -> bool {
        matches!(self.bounds, WeightBounds::Signed { .. })
    }
}

/// The multiplier applied to every target: the constant fixed at plan approval (R2) times the drawdown ladder's scale.
/// Both are recorded so a report can say which one shrank the book.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskScale {
    /// Council R2: fixed at plan approval, in `(0, 1]`. See [`approval_risk_scale`].
    pub approval_constant: f64,
    /// The ladder's current scale, in `[0, 1]` (0 = halted: every target is zero, i.e. flatten).
    pub ladder: f64,
}

impl RiskScale {
    pub const ONE: RiskScale = RiskScale { approval_constant: 1.0, ladder: 1.0 };
    pub const fn new(approval_constant: f64, ladder: f64) -> Self {
        RiskScale { approval_constant, ladder }
    }
    /// The single multiplier, `approval_constant * ladder`.
    pub fn product(&self) -> f64 {
        self.approval_constant * self.ladder
    }
}

/// Council R2: when the mandate's gross cap is below what the rule needs, the plan runs at a CONSTANT scale fixed at
/// approval, `c = min(1, headroom * cap / reference_p90_gross)` with headroom 0.9 and the reference FX sleeve's p90
/// gross exposure 4.41. `cap` is the effective gross cap as a multiple of the capital base.
pub fn approval_risk_scale(cap: f64) -> f64 {
    (crate::R2_HEADROOM * cap / crate::R2_REFERENCE_P90_GROSS).min(1.0)
}

/// Per-instrument facts for one construction step. Prices and holdings come from the caller (the simulator's mark, or
/// the broker's positions); nothing here is looked up.
#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentFacts {
    pub symbol: String,
    pub venue: String,
    /// Asset class label for class caps (compared trimmed and lower case).
    pub class: String,
    /// Usable price; `None`, non-finite or non-positive means "no price": the instrument is skipped (`NoPrice`), as the
    /// planner does, and contributes nothing to the target book.
    pub price: Option<f64>,
    /// Signed units currently held.
    pub held_units: f64,
    /// Margin rule input: OANDA `marginRate`, Alpaca per-asset requirement. `None` = not stated.
    pub margin_rate: Option<f64>,
    /// Alpaca `marginable` (false = 100% requirement). Ignored by the other models.
    pub marginable: bool,
    /// `false` = the instrument's target still counts in the gross/limit/margin checks but no trade is emitted (its sleeve
    /// is not due this step, backtester cadence `PerSleeve`). Live plans use `true` everywhere.
    pub in_scope: bool,
}

impl InstrumentFacts {
    pub fn new(symbol: &str, venue: &str, class: &str, price: f64) -> Self {
        InstrumentFacts {
            symbol: symbol.to_string(),
            venue: venue.to_string(),
            class: class.to_string(),
            price: Some(price),
            held_units: 0.0,
            margin_rate: None,
            marginable: true,
            in_scope: true,
        }
    }
    pub fn with_held(mut self, units: f64) -> Self {
        self.held_units = units;
        self
    }
    pub fn with_margin_rate(mut self, rate: f64) -> Self {
        self.margin_rate = Some(rate);
        self
    }
    pub fn with_marginable(mut self, marginable: bool) -> Self {
        self.marginable = marginable;
        self
    }
    pub fn with_in_scope(mut self, in_scope: bool) -> Self {
        self.in_scope = in_scope;
        self
    }
    pub fn without_price(mut self) -> Self {
        self.price = None;
        self
    }
    /// Current signed notional, `held_units * price` (0 without a price).
    pub fn held_notional(&self) -> f64 {
        self.price.map_or(0.0, |p| self.held_units * p)
    }
}

/// What limits the increasing orders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Funding {
    /// Research and certification runs (the T0 convention): increases execute at the full target, no cash test, no fees
    /// (the simulator charges costs itself).
    Unconstrained,
    /// A cash book (the planner's default): increases are limited by `cash - reserve`, fees included.
    Cash {
        /// The broker's actual cash.
        cash: f64,
        /// `min_cash_reserve`, a fraction of the CAPITAL BASE, rounded up to 8 decimals.
        reserve_fraction: f64,
        /// Fee as a fraction of notional, charged per order (rounded up to 8 decimals). A venue fact the caller owns.
        fee_rate: f64,
        /// Count the net proceeds of the reductions as cash available for the increases of the same plan.
        credit_sell_proceeds: bool,
    },
    /// A margin book: the broker's buying power (notional) REPLACES cash as the budget of the increases; reductions are
    /// not credited back.
    BuyingPower { buying_power: f64, reserve_fraction: f64, fee_rate: f64 },
}

/// Everything one construction step needs. The first nine fields are the design's `ConstructInputs`; `funding`,
/// `target_dp` and `unmanaged_gross` are the extensions this crate adds (design 3.2 leaves cash and rounding of the
/// target to the planner).
#[derive(Clone, Copy)]
pub struct ConstructInputs<'a> {
    /// The account's equity (NAV) in account currency, positive.
    pub equity: f64,
    /// `capital_base = min(equity, allocated)`; `None` = no allocation cap.
    pub allocated_capital: Option<f64>,
    pub sleeves: &'a [SleeveTargets],
    pub risk_scale: RiskScale,
    pub limits: &'a Limits,
    pub instruments: &'a [InstrumentFacts],
    pub margin: &'a dyn MarginModel,
    pub trade_filter: TradeFilter,
    /// Live-faithful lot rounding; `None` in research runs (fractional units).
    pub rounding: Option<&'a dyn QuantityRounder>,
    /// Cash / buying-power budget of the increasing orders.
    pub funding: Funding,
    /// Round each target notional toward zero to this many decimals (`Some(8)` = the planner's Decimal quantum);
    /// `None` = unrounded f64 (the answer key's convention).
    pub target_dp: Option<u32>,
    /// Absolute value of positions the sleeves do not manage; only used for `projected_gross` / `needs_margin`.
    pub unmanaged_gross: f64,
}

// -------------------------------------------------------------------------------------------------------------------
// Outputs
// -------------------------------------------------------------------------------------------------------------------

/// Which part of a trade this order is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegKind {
    /// The whole trade, in one order.
    Whole,
    /// First order of a trade that crosses zero: closes the held position exactly.
    Close,
    /// Second order of a trade that crosses zero: opens the position on the other side.
    Open,
}

/// One order the plan wants placed.
#[derive(Clone, Debug, PartialEq)]
pub struct TradeIntent {
    pub instrument: usize,
    pub symbol: String,
    pub venue: String,
    pub side: Side,
    /// A magnitude, already rounded DOWN by the rounder when there is one.
    pub quantity: f64,
    pub price: f64,
    /// `quantity * price`.
    pub notional: f64,
    /// `ceil8(notional * fee_rate)` for a funded plan, 0 otherwise.
    pub est_fee: f64,
    /// Moves the position toward zero (a sell of a long, a buy that covers a short); everything else adds exposure.
    pub reducing: bool,
    pub leg: LegKind,
}

/// A trade that was not sized or not placed, with the reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Skip {
    pub instrument: usize,
    pub symbol: String,
    pub reason: SkipReason,
}

/// What the construction saw for one managed instrument (the planner's `InstrumentLine`).
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub instrument: usize,
    pub symbol: String,
    /// Ids of the sleeves that name the instrument, in id order.
    pub sleeves: Vec<String>,
    pub held_units: f64,
    pub current_notional: f64,
    pub target_notional: f64,
}

/// Which constraint a [`Binding`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    CapitalBase,
    Gross,
    Net,
    Position,
    AssetClass,
    Margin,
}

/// One constraint's use of its cap, for reporting: `bound` is true when it is within 1e-9 of the cap.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub kind: BindingKind,
    /// The instrument symbol (Position), the class (AssetClass) or the model name (Margin); empty otherwise.
    pub label: String,
    pub used: f64,
    pub cap: f64,
    pub bound: bool,
}

impl Binding {
    /// `used / cap` (0 when both are 0, infinity when only the cap is 0).
    pub fn ratio(&self) -> f64 {
        if self.cap == 0.0 {
            if self.used == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.used / self.cap
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstructOutput {
    pub capital_base: f64,
    /// The multiplier that was applied, `approval_constant * ladder`.
    pub risk_scale_applied: f64,
    /// Target notional per instrument index (0 for an instrument no sleeve names, and for a skipped `NoPrice` one).
    pub target_notional: Vec<f64>,
    /// True for the instruments some sleeve names.
    pub named: Vec<bool>,
    pub lines: Vec<Line>,
    /// Reductions first, then increases; each group ordered by (venue, symbol); a crossing is close then open.
    pub trades: Vec<TradeIntent>,
    pub skipped: Vec<Skip>,
    /// Sum of `|target|` over the lines.
    pub gross: f64,
    /// Sum of the signed targets.
    pub net: f64,
    /// Margin the target book uses under the margin model (0 for `NoMargin`).
    pub margin_used: f64,
    /// The targets need margin: a short target, or projected gross above equity.
    pub needs_margin: bool,
    /// `gross` plus the value of the unmanaged positions.
    pub projected_gross: f64,
    /// Cash (or buying power) left after the increases; `None` for `Funding::Unconstrained`.
    pub funding_left: Option<f64>,
    /// Every finite constraint, closest first.
    pub binding: Vec<Binding>,
}

/// Why the inputs were rejected before any arithmetic (mirrors the planner's `PlanError` validation).
#[derive(Clone, Debug, PartialEq)]
pub enum InputError {
    BadRiskScale { approval_constant: f64, ladder: f64 },
    BadParam(&'static str),
    BadShare(String),
    SharesExceedOne(f64),
    DuplicateSleeve(String),
    BadWeight { sleeve: String, symbol: String },
    WeightsExceedOne(String),
    DuplicateWeight { sleeve: String, symbol: String },
    BadSignedWeight { sleeve: String, symbol: String, max: f64 },
    BadMaxAbsWeight { sleeve: String, max: f64 },
    EquityInvalid(f64),
    UnknownInstrument { sleeve: String, index: usize },
    DuplicateInstrument(String),
    MarginRateUnknown(String),
    NonFinite(&'static str),
}

/// Why the WHOLE book is refused (council R1/R2: never a partial book, never clipped).
#[derive(Clone, Debug, PartialEq)]
pub enum ConstructRefusal {
    GrossAboveCap {
        gross: f64,
        cap: f64,
    },
    NetAboveCap {
        net: f64,
        cap: f64,
    },
    PositionAboveCap {
        instrument: usize,
        symbol: String,
        notional: f64,
        cap: f64,
    },
    ClassAboveCap {
        class: String,
        gross: f64,
        cap: f64,
    },
    MarginExceeded {
        model: &'static str,
        used: f64,
        ceiling: f64,
    },
    ShortingForbidden {
        instrument: usize,
        symbol: String,
        target: f64,
    },
    /// The targets need margin and the caller gave a cash budget instead of buying power (planner:
    /// `BuyingPowerRequired`).
    BuyingPowerRequired {
        gross: f64,
    },
    /// The rounder returned more than was asked for: rounders may only round down (planner: `VenueRoundedUp`).
    VenueRoundedUp {
        symbol: String,
        wished: f64,
        returned: f64,
    },
    Invalid(InputError),
}

impl std::fmt::Display for ConstructRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstructRefusal::GrossAboveCap { gross, cap } => {
                write!(f, "target gross {gross} exceeds the gross cap {cap}")
            }
            ConstructRefusal::NetAboveCap { net, cap } => write!(f, "target net {net} exceeds the net cap {cap}"),
            ConstructRefusal::PositionAboveCap { symbol, notional, cap, .. } => {
                write!(f, "target {notional} in {symbol} exceeds the per-position cap {cap}")
            }
            ConstructRefusal::ClassAboveCap { class, gross, cap } => {
                write!(f, "asset class {class} would be {gross}; the cap is {cap}")
            }
            ConstructRefusal::MarginExceeded { model, used, ceiling } => {
                write!(f, "margin used {used} exceeds the {model} ceiling {ceiling}")
            }
            ConstructRefusal::ShortingForbidden { symbol, target, .. } => {
                write!(f, "target {target} in {symbol} is short and shorting is not permitted")
            }
            ConstructRefusal::BuyingPowerRequired { gross } => {
                write!(f, "the targets need margin (gross {gross}) and no buying power was supplied")
            }
            ConstructRefusal::VenueRoundedUp { symbol, wished, returned } => {
                write!(f, "venue rules rounded {symbol} UP ({wished} -> {returned}); rules must only round down")
            }
            ConstructRefusal::Invalid(e) => write!(f, "invalid input: {e:?}"),
        }
    }
}

impl std::error::Error for ConstructRefusal {}

// -------------------------------------------------------------------------------------------------------------------
// Validation
// -------------------------------------------------------------------------------------------------------------------

fn validate(i: &ConstructInputs<'_>) -> Result<(), InputError> {
    use InputError::*;
    let rs = i.risk_scale;
    let ac = rs.approval_constant;
    if !(ac > 0.0 && ac <= 1.0 && rs.ladder >= 0.0 && rs.ladder <= 1.0) {
        return Err(BadRiskScale { approval_constant: ac, ladder: rs.ladder });
    }
    i.trade_filter.validate().map_err(BadParam)?;
    i.limits.validate().map_err(BadParam)?;
    if !i.equity.is_finite() || i.equity <= 0.0 {
        return Err(EquityInvalid(i.equity));
    }
    if let Some(a) = i.allocated_capital {
        if !a.is_finite() || a <= 0.0 {
            return Err(BadParam("allocated_capital must be positive and finite"));
        }
    }
    if !i.unmanaged_gross.is_finite() || i.unmanaged_gross < 0.0 {
        return Err(BadParam("unmanaged_gross must be finite and not negative"));
    }
    if let Some(dp) = i.target_dp {
        if dp > 18 {
            return Err(BadParam("target_dp must be at most 18"));
        }
    }
    match i.funding {
        Funding::Unconstrained => {}
        Funding::Cash { cash, reserve_fraction, fee_rate, .. } => {
            if !cash.is_finite() || cash < 0.0 {
                return Err(BadParam("cash must be finite and not negative"));
            }
            check_funding_params(reserve_fraction, fee_rate)?;
        }
        Funding::BuyingPower { buying_power, reserve_fraction, fee_rate } => {
            if !buying_power.is_finite() || buying_power < 0.0 {
                return Err(BadParam("buying_power must be finite and not negative"));
            }
            check_funding_params(reserve_fraction, fee_rate)?;
        }
    }
    // instruments: unique symbols
    let mut seen_symbols = std::collections::BTreeSet::new();
    for inst in i.instruments {
        if !seen_symbols.insert(inst.symbol.trim().to_uppercase()) {
            return Err(DuplicateInstrument(inst.symbol.clone()));
        }
        if !inst.held_units.is_finite() {
            return Err(NonFinite("held_units"));
        }
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut total_share = 0.0;
    for s in i.sleeves {
        let id = s.id.trim().to_string();
        if !ids.insert(id.clone()) {
            return Err(DuplicateSleeve(id));
        }
        if !s.share.is_finite() || s.share <= 0.0 || s.share > 1.0 {
            return Err(BadShare(id));
        }
        total_share += s.share;
        if let WeightBounds::Signed { max_abs_weight } = s.bounds {
            if !(max_abs_weight > 0.0 && max_abs_weight <= crate::MAX_ABS_WEIGHT_CAP) {
                return Err(BadMaxAbsWeight { sleeve: id, max: max_abs_weight });
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut total_weight = 0.0;
        for &(j, w) in &s.weights {
            let Some(inst) = i.instruments.get(j) else {
                return Err(UnknownInstrument { sleeve: id, index: j });
            };
            if !w.is_finite() {
                return Err(NonFinite("weight"));
            }
            match s.bounds {
                WeightBounds::LongOnlyUnit => {
                    if w < 0.0 || exceeds(w, 1.0) {
                        return Err(BadWeight { sleeve: id, symbol: inst.symbol.clone() });
                    }
                }
                WeightBounds::Signed { max_abs_weight } => {
                    if w.abs() > max_abs_weight {
                        return Err(BadSignedWeight { sleeve: id, symbol: inst.symbol.clone(), max: max_abs_weight });
                    }
                }
            }
            if !seen.insert(j) {
                return Err(DuplicateWeight { sleeve: id, symbol: inst.symbol.clone() });
            }
            total_weight += w;
        }
        if s.bounds == WeightBounds::LongOnlyUnit && exceeds(total_weight, 1.0) {
            return Err(WeightsExceedOne(id));
        }
    }
    if exceeds(total_share, 1.0) {
        return Err(SharesExceedOne(total_share));
    }
    Ok(())
}

fn check_funding_params(reserve_fraction: f64, fee_rate: f64) -> Result<(), InputError> {
    if !reserve_fraction.is_finite() || !(0.0..=1.0).contains(&reserve_fraction) {
        return Err(InputError::BadParam("reserve_fraction must be in [0, 1]"));
    }
    if !fee_rate.is_finite() || !(0.0..1.0).contains(&fee_rate) {
        return Err(InputError::BadParam("fee_rate must be in [0, 1)"));
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------------------------
// construct
// -------------------------------------------------------------------------------------------------------------------

/// `min(equity, allocated)`, or the equity when there is no allocation cap.
pub fn capital_base(equity: f64, allocated: Option<f64>) -> f64 {
    match allocated {
        Some(a) => equity.min(a),
        None => equity,
    }
}

struct Candidate {
    j: usize,
    side: Side,
    qty: f64,
    price: f64,
    reducing: bool,
    leg: LegKind,
}

fn fee_for(notional: f64, rate: f64) -> f64 {
    if rate == 0.0 {
        0.0
    } else {
        ceil_dp(notional * rate, 8)
    }
}

fn sort_key(inst: &InstrumentFacts) -> (String, String) {
    (inst.venue.trim().to_lowercase(), inst.symbol.trim().to_uppercase())
}

/// The design's `construct`. See the module docs for the order of operations.
pub fn construct(i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal> {
    validate(i).map_err(ConstructRefusal::Invalid)?;
    let n = i.instruments.len();
    let cb = capital_base(i.equity, i.allocated_capital);
    let rs = i.risk_scale.product();

    // 3. combine the sleeves, in id order.
    let mut order: Vec<usize> = (0..i.sleeves.len()).collect();
    order.sort_by(|a, b| i.sleeves[*a].id.trim().cmp(i.sleeves[*b].id.trim()));
    let mut raw: Vec<Option<f64>> = vec![None; n];
    let mut signed_inst = vec![false; n];
    let mut labels: Vec<Vec<String>> = vec![Vec::new(); n];
    for &si in &order {
        let s = &i.sleeves[si];
        for &(j, w) in &s.weights {
            let term = s.share * w;
            raw[j] = Some(match raw[j] {
                Some(acc) => acc + term,
                None => term,
            });
            signed_inst[j] |= s.is_signed();
            labels[j].push(s.id.trim().to_string());
        }
    }
    let signed_plan = i.sleeves.iter().any(SleeveTargets::is_signed);

    // 4. targets.
    let mut skipped: Vec<Skip> = Vec::new();
    let mut lines: Vec<Line> = Vec::new();
    let mut target_notional = vec![0.0; n];
    let mut named = vec![false; n];
    for j in 0..n {
        let Some(w) = raw[j] else { continue };
        named[j] = true;
        let inst = &i.instruments[j];
        let skip = |reason: SkipReason| Skip { instrument: j, symbol: inst.symbol.clone(), reason };
        let price = match inst.price {
            Some(p) if p.is_finite() && p > 0.0 => p,
            _ => {
                skipped.push(skip(SkipReason::NoPrice));
                continue;
            }
        };
        if inst.held_units < 0.0 && !signed_inst[j] {
            skipped.push(skip(SkipReason::ShortPositionHeld));
            continue;
        }
        let mut target = (cb * w) * rs;
        if let Some(dp) = i.target_dp {
            target = round_toward_zero_dp(target, dp);
        }
        target_notional[j] = target;
        lines.push(Line {
            instrument: j,
            symbol: inst.symbol.clone(),
            sleeves: labels[j].clone(),
            held_units: inst.held_units,
            current_notional: inst.held_units * price,
            target_notional: target,
        });
    }

    // 5-6. whole-book checks.
    let abs_targets: Vec<f64> = lines.iter().map(|l| l.target_notional.abs()).collect();
    let signed_targets: Vec<f64> = lines.iter().map(|l| l.target_notional).collect();
    let gross = stable_sum(&abs_targets);
    let net = stable_sum(&signed_targets);
    let projected_gross = gross + i.unmanaged_gross;
    let needs_margin = lines.iter().any(|l| l.target_notional < 0.0) || exceeds(projected_gross, i.equity);

    let lim = i.limits;
    let refuse_all = lim.policy == LimitPolicy::RefuseWholeBook;
    let gross_cap = lim.effective_max_gross() * cb;
    if (refuse_all || signed_plan) && exceeds(gross, gross_cap) {
        return Err(ConstructRefusal::GrossAboveCap { gross, cap: gross_cap });
    }
    if refuse_all {
        let net_cap = lim.max_net * cb;
        if exceeds(net.abs(), net_cap) {
            return Err(ConstructRefusal::NetAboveCap { net, cap: net_cap });
        }
        let pos_cap = lim.max_position * cb;
        for l in &lines {
            if exceeds(l.target_notional.abs(), pos_cap) {
                return Err(ConstructRefusal::PositionAboveCap {
                    instrument: l.instrument,
                    symbol: l.symbol.clone(),
                    notional: l.target_notional,
                    cap: pos_cap,
                });
            }
        }
        let mut class_terms: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for l in &lines {
            let class = i.instruments[l.instrument].class.trim().to_lowercase();
            class_terms.entry(class).or_default().push(l.target_notional.abs());
        }
        for (class, terms) in &class_terms {
            if let Some(frac) = lim.max_asset_class.get(class) {
                let class_gross = stable_sum(terms);
                let cap = frac * cb;
                if exceeds(class_gross, cap) {
                    return Err(ConstructRefusal::ClassAboveCap { class: class.clone(), gross: class_gross, cap });
                }
            }
        }
        if !lim.shorting {
            if let Some(l) = lines.iter().find(|l| l.target_notional < 0.0) {
                return Err(ConstructRefusal::ShortingForbidden {
                    instrument: l.instrument,
                    symbol: l.symbol.clone(),
                    target: l.target_notional,
                });
            }
        }
    }
    // margin model (reported under both policies; refused only under R1/R2 semantics)
    let mut margin_terms = Vec::with_capacity(lines.len());
    for l in &lines {
        let inst = &i.instruments[l.instrument];
        let Some(rate) = i.margin.rate(inst) else {
            return Err(ConstructRefusal::Invalid(InputError::MarginRateUnknown(inst.symbol.clone())));
        };
        margin_terms.push(rate * l.target_notional.abs());
    }
    let margin_used = stable_sum(&margin_terms);
    let margin_ceiling = i.margin.ceiling(i.equity);
    if refuse_all {
        if let Some(ceiling) = margin_ceiling {
            if exceeds(margin_used, ceiling) {
                return Err(ConstructRefusal::MarginExceeded { model: i.margin.name(), used: margin_used, ceiling });
            }
        }
    }
    if signed_plan && needs_margin && matches!(i.funding, Funding::Cash { .. }) {
        return Err(ConstructRefusal::BuyingPowerRequired { gross: projected_gross });
    }

    // 7-9. trades.
    let mut reducers: Vec<Candidate> = Vec::new();
    let mut increasers: Vec<Candidate> = Vec::new();
    for l in &lines {
        let j = l.instrument;
        let inst = &i.instruments[j];
        if !inst.in_scope {
            continue;
        }
        let price = inst.price.unwrap_or(0.0);
        let skip = |reason: SkipReason| Skip { instrument: j, symbol: inst.symbol.clone(), reason };
        let (target, current) = (l.target_notional, l.current_notional);
        let delta = target - current;
        // Zero, or float noise around zero: not a trade (an exact-decimal planner sees exactly 0 here).
        if delta == 0.0 || delta.abs() <= crate::EDGE_TOL * target.abs().max(current.abs()) {
            continue;
        }
        if let Some(reason) = i.trade_filter.check(target, current) {
            skipped.push(skip(reason));
            continue;
        }
        let abs_delta = delta.abs();
        let held = inst.held_units;
        let held_abs = held.abs();
        let crossing = (current > 0.0 && target < 0.0) || (current < 0.0 && target > 0.0);
        let legs: Vec<(LegKind, Side, f64, bool)> = if crossing {
            let close_side = if held > 0.0 { Side::Sell } else { Side::Buy };
            let open_side = if target > 0.0 { Side::Buy } else { Side::Sell };
            vec![(LegKind::Close, close_side, held_abs, true), (LegKind::Open, open_side, target.abs() / price, false)]
        } else {
            let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
            let reduction = (side == Side::Sell && held > 0.0) || (side == Side::Buy && held < 0.0);
            let wished = if !reduction {
                abs_delta / price
            } else if target == 0.0 {
                held_abs
            } else {
                (abs_delta / price).min(held_abs)
            };
            vec![(LegKind::Whole, side, wished, reduction)]
        };
        let mut pending: Vec<Candidate> = Vec::new();
        let mut residual = 0.0;
        for (leg, side, wished, reducing) in legs {
            // The open leg also covers whatever the venue's rounding left of the closed position, so the final
            // position can never exceed the target.
            let wished = if leg == LegKind::Open { wished + residual } else { wished };
            let qty = match i.rounding {
                None => wished,
                Some(r) => match r.round_quantity(&inst.symbol, side, wished, price) {
                    Ok(q) => {
                        if exceeds(q, wished) {
                            return Err(ConstructRefusal::VenueRoundedUp {
                                symbol: inst.symbol.clone(),
                                wished,
                                returned: q,
                            });
                        }
                        q
                    }
                    Err(refusal) => {
                        skipped.push(skip(SkipReason::VenueRefused(refusal)));
                        if leg == LegKind::Close {
                            pending.clear();
                            break;
                        }
                        continue;
                    }
                },
            };
            if leg == LegKind::Close {
                residual = wished - qty;
            }
            pending.push(Candidate { j, side, qty, price, reducing, leg });
        }
        for c in pending {
            if c.reducing {
                reducers.push(c);
            } else {
                increasers.push(c);
            }
        }
    }
    let by_venue_symbol =
        |a: &Candidate, b: &Candidate| sort_key(&i.instruments[a.j]).cmp(&sort_key(&i.instruments[b.j]));
    reducers.sort_by(by_venue_symbol);
    increasers.sort_by(by_venue_symbol);

    // 10. funding of the increases.
    let (fee_rate, reserve_fraction) = match i.funding {
        Funding::Unconstrained => (0.0, 0.0),
        Funding::Cash { fee_rate, reserve_fraction, .. } | Funding::BuyingPower { fee_rate, reserve_fraction, .. } => {
            (fee_rate, reserve_fraction)
        }
    };
    let mut sim_cash = if let Funding::Cash { cash, .. } = i.funding { cash } else { 0.0 };
    let mut trades: Vec<TradeIntent> = Vec::new();
    for c in &reducers {
        let notional = c.qty * c.price;
        let fee = fee_for(notional, fee_rate);
        sim_cash = match c.side {
            Side::Sell => sim_cash + notional - fee,
            Side::Buy => sim_cash - notional - fee,
        };
        trades.push(intent(i, c, c.qty, fee));
    }
    let mut buy_qtys: Vec<f64> = increasers.iter().map(|c| c.qty).collect();
    let mut funding_left: Option<f64> = None;
    let mut available_positive = true;
    if !matches!(i.funding, Funding::Unconstrained) {
        let budget = match i.funding {
            Funding::BuyingPower { buying_power, .. } => buying_power,
            Funding::Cash { cash, credit_sell_proceeds, .. } => {
                if credit_sell_proceeds {
                    sim_cash
                } else {
                    sim_cash.min(cash)
                }
            }
            Funding::Unconstrained => 0.0,
        };
        let reserve = ceil_dp(reserve_fraction * cb, 8);
        let available = budget - reserve;
        available_positive = available > 0.0;
        if !increasers.is_empty() {
            let total = stable_sum(
                &increasers
                    .iter()
                    .zip(&buy_qtys)
                    .map(|(c, q)| {
                        let nt = q * c.price;
                        nt + fee_for(nt, fee_rate)
                    })
                    .collect::<Vec<f64>>(),
            );
            if total > available {
                // One common factor for every increasing order, against a budget shrunk by the per-order fee rounding
                // slack (each fee is rounded UP by at most 1e-8), so the re-rounded total fits.
                let usable = available - crate::FEE_SLACK_PER_BUY * increasers.len() as f64;
                let factor = if usable > 0.0 { usable / total } else { 0.0 };
                for (c, q) in increasers.iter().zip(buy_qtys.iter_mut()) {
                    let scaled = *q * factor;
                    if scaled <= 0.0 {
                        *q = 0.0;
                        continue;
                    }
                    *q = match i.rounding {
                        None => scaled,
                        Some(r) => match r.round_quantity(&i.instruments[c.j].symbol, c.side, scaled, c.price) {
                            Ok(rounded) if !exceeds(rounded, scaled) => rounded,
                            Ok(rounded) => {
                                return Err(ConstructRefusal::VenueRoundedUp {
                                    symbol: i.instruments[c.j].symbol.clone(),
                                    wished: scaled,
                                    returned: rounded,
                                })
                            }
                            Err(_) => 0.0,
                        },
                    };
                }
            }
        }
        let mut spent = Vec::new();
        for (c, q) in increasers.iter().zip(&buy_qtys) {
            let nt = q * c.price;
            spent.push(nt + fee_for(nt, fee_rate));
        }
        funding_left = Some(match i.funding {
            Funding::BuyingPower { buying_power, .. } => buying_power - stable_sum(&spent),
            _ => sim_cash - stable_sum(&spent),
        });
    }
    for (c, q) in increasers.iter().zip(&buy_qtys) {
        if *q == 0.0 {
            let reason =
                if available_positive { SkipReason::CutBelowVenueMinimum } else { SkipReason::NoCashAvailable };
            skipped.push(Skip { instrument: c.j, symbol: i.instruments[c.j].symbol.clone(), reason });
            continue;
        }
        let fee = fee_for(*q * c.price, fee_rate);
        trades.push(intent(i, c, *q, fee));
    }

    // Binding constraints, for reporting.
    let mut binding: Vec<Binding> = Vec::new();
    let mut push = |kind: BindingKind, label: String, used: f64, cap: f64| {
        if cap.is_finite() {
            let bound = used >= cap - cap.abs() * 1e-9;
            binding.push(Binding { kind, label, used, cap, bound });
        }
    };
    if let Some(a) = i.allocated_capital {
        push(BindingKind::CapitalBase, String::new(), i.equity, a);
    }
    push(BindingKind::Gross, String::new(), gross, gross_cap);
    push(BindingKind::Net, String::new(), net.abs(), lim.max_net * cb);
    if let Some(l) = lines.iter().max_by(|a, b| a.target_notional.abs().total_cmp(&b.target_notional.abs())) {
        push(BindingKind::Position, l.symbol.clone(), l.target_notional.abs(), lim.max_position * cb);
    }
    let mut class_gross: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for l in &lines {
        class_gross
            .entry(i.instruments[l.instrument].class.trim().to_lowercase())
            .or_default()
            .push(l.target_notional.abs());
    }
    for (class, cap) in &lim.max_asset_class {
        let used = class_gross.get(class).map_or(0.0, |t| stable_sum(t));
        push(BindingKind::AssetClass, class.clone(), used, cap * cb);
    }
    if let Some(ceiling) = margin_ceiling {
        push(BindingKind::Margin, i.margin.name().to_string(), margin_used, ceiling);
    }
    binding.sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));

    Ok(ConstructOutput {
        capital_base: cb,
        risk_scale_applied: rs,
        target_notional,
        named,
        lines,
        trades,
        skipped,
        gross,
        net,
        margin_used,
        needs_margin,
        projected_gross,
        funding_left,
        binding,
    })
}

fn intent(i: &ConstructInputs<'_>, c: &Candidate, qty: f64, fee: f64) -> TradeIntent {
    let inst = &i.instruments[c.j];
    TradeIntent {
        instrument: c.j,
        symbol: inst.symbol.clone(),
        venue: inst.venue.clone(),
        side: c.side,
        quantity: qty,
        price: c.price,
        notional: qty * c.price,
        est_fee: fee,
        reducing: c.reducing,
        leg: c.leg,
    }
}
