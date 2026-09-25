//! The specification of a BOOK (design 3.3): sleeves, shares, allocator, cadence, account mode, overlay hook, and the
//! errors of [`crate::simulate_book`]. The simulation itself is in [`crate::book_sim`], its output in
//! [`crate::book_result`].

use crate::bartime::BarTime;
use crate::book_panel::BookPanelError;
use crate::construct::{CashPolicy, ConstructRefusal, TradeFilter};
use crate::rule::RebalancePolicy;
use crate::rule::WeightRule;
use crate::sim::{SimConfig, SimError};
use crate::stateful::{DynAdapter, DynRule, StatefulRule, Stateless};
use std::fmt;
use std::sync::Arc;

/// How much of the book a sleeve gets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShareSpec {
    /// A fixed share (design: `Fixed(f64)`). Must be finite and > 0.
    Fixed(f64),
    /// The share is set by the book's allocator; `initial` applies until the first successful review.
    Allocated { initial: f64 },
}

impl ShareSpec {
    pub fn initial(&self) -> f64 {
        match self {
            ShareSpec::Fixed(v) => *v,
            ShareSpec::Allocated { initial } => *initial,
        }
    }
}

/// One strategy sleeve of a book.
#[derive(Clone)]
pub struct SleeveSpec {
    pub id: String,
    /// Frozen rule (any `WeightRule` via the blanket impl, or a `StatefulRule`).
    pub rule: Arc<dyn DynRule>,
    /// Indices into `BookPanel::instruments`, in the rule's universe order.
    pub universe: Vec<usize>,
    pub share: ShareSpec,
    /// Overrides the rule's own rebalance policy for this sleeve (`None` = the rule's).
    pub policy: Option<RebalancePolicy>,
    /// PER-SLEEVE execution delay (weightsim 0.3, council Ruling 4): a decision taken at the close of the sleeve's OWN bar
    /// `t` becomes effective, and is traded, at the close of the sleeve's own bar `t + d`. `None` (the default) means "the
    /// book-level value", `BookConfig::sim.execution_delay_bars`, so every book that never calls
    /// [`SleeveSpec::with_execution_delay`] behaves, bit for bit, as it did before 0.3. `Some(d)` OVERRIDES the book-level
    /// value for this sleeve only (including `Some(0)` under a non-zero book-level delay).
    pub execution_delay: Option<usize>,
}

impl SleeveSpec {
    pub fn new(id: impl Into<String>, rule: Arc<dyn DynRule>, universe: Vec<usize>, share: ShareSpec) -> SleeveSpec {
        SleeveSpec { id: id.into(), rule, universe, share, policy: None, execution_delay: None }
    }
    /// Give this sleeve its own execution delay, in the sleeve's OWN bars (see [`SleeveSpec::execution_delay`]). A live
    /// ETF sleeve is `with_execution_delay(1)` (decided at the month-end close, acted one session later) while a crypto
    /// sleeve in the same account stays at 0.
    pub fn with_execution_delay(mut self, bars: usize) -> SleeveSpec {
        self.execution_delay = Some(bars);
        self
    }
    /// The delay in force for this sleeve: its own if set, else the book-level `book_default`.
    pub fn effective_delay(&self, book_default: usize) -> usize {
        self.execution_delay.unwrap_or(book_default)
    }
    /// A sleeve running a T1 [`WeightRule`] (wrapped in [`Stateless`]).
    pub fn from_rule<R: WeightRule + 'static>(
        id: impl Into<String>,
        rule: R,
        universe: Vec<usize>,
        share: ShareSpec,
    ) -> SleeveSpec {
        SleeveSpec::new(id, Arc::new(DynAdapter(Stateless(rule))), universe, share)
    }
    /// A sleeve running a rule with memory.
    pub fn from_stateful<R: StatefulRule + 'static>(
        id: impl Into<String>,
        rule: R,
        universe: Vec<usize>,
        share: ShareSpec,
    ) -> SleeveSpec {
        SleeveSpec::new(id, Arc::new(DynAdapter(rule)), universe, share)
    }
    pub fn with_policy(mut self, policy: RebalancePolicy) -> SleeveSpec {
        self.policy = Some(policy);
        self
    }
    pub(crate) fn effective_policy(&self) -> RebalancePolicy {
        self.policy.unwrap_or_else(|| self.rule.rebalance_policy())
    }
}

/// How shares are set. Only STATIC allocators exist (design R-P6): shares change only at declared review dates and
/// only from information available at the review close.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AllocatorSpec {
    /// Shares are the `ShareSpec::Fixed` values, forever.
    Fixed,
    /// Frozen inverse-volatility shares: at every calendar month-end of the account clock (and on its last bar) each
    /// sleeve's share becomes `total * (1/sd_s) / SUM(1/sd)`, with `sd_s` the sample standard deviation of the last
    /// `lookback_bars` GROSS returns of that sleeve's own unit-capital shadow account on its own calendar, through the
    /// review close. Until `lookback_bars` returns exist (or when a volatility is zero) the initial shares stay.
    /// Every sleeve must be `ShareSpec::Allocated`.
    InverseVol { lookback_bars: usize, total: f64 },
}

/// Which sleeves are re-targeted when the account is driven (design 3.3, finding F1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookCadence {
    /// T1 `PerSleeve` semantics: a sleeve is planned only on the bars where it is DUE (a new target became effective,
    /// or its policy is `EveryBar`). The certification mode.
    PerSleeve,
    /// The live driver's behaviour: when ANY sleeve is due, EVERY sleeve that has a bar is re-targeted to its standing
    /// target (so a monthly sleeve in an account with a daily one is re-targeted every day).
    AllSleevesOnAnyDue,
}

/// How the account is modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountMode {
    /// One joint account: one cash pool, sleeves combined and netted, one cost line.
    Joint,
    /// LEGACY EMULATION ONLY (old-versus-new comparison, never evidence): every sleeve is its own sub-account with
    /// capital `share * initial_equity`, never rebalanced against the others, and the book equity is the SUM of the
    /// sub-account equities, which is what the old engine's summed curves are. No joint construction applies
    /// (`allocated_capital`, `trade_filter`, `max_gross`, overlay and a non-`Fixed` allocator are refused).
    IndependentSubAccounts,
}

/// Book-level account configuration. The single-rule fields are exactly [`SimConfig`] (so a one-sleeve book run with
/// the same `SimConfig` is bit-identical to `simulate`).
#[derive(Clone, Debug)]
pub struct BookConfig {
    /// `start`/`end` (counting window, as in T1), `initial_equity`, `cost`, `financing`, `on_refusal`,
    /// `execution_delay_bars` (in the decision sleeve's OWN bars; the DEFAULT for every sleeve, which a sleeve overrides
    /// with [`SleeveSpec::with_execution_delay`]), `risk_scale` (constant approval scale) and
    /// `max_gross` (a breach REFUSES the whole book on that bar; under `OnRefusal::Abort` it fails the run instead).
    pub sim: SimConfig,
    /// The account is flat before the first clock instant at or after this time; earlier bars are history only (rules
    /// see them, nothing trades and no decision is taken). `None` = the account starts on the first clock instant.
    pub account_start: Option<BarTime>,
    pub cadence: BookCadence,
    pub mode: AccountMode,
    /// The mandate's allocated capital in account units: `capital_base = min(equity, allocated)`.
    pub allocated_capital: Option<f64>,
    pub trade_filter: Option<TradeFilter>,
    pub cash_policy: CashPolicy,
    /// Diagnostic: also re-target sleeves on bars where their market is closed, at the stale carried close (what an
    /// unattended daily driver would attempt on a weekend). `false` in every certification.
    pub trade_on_closed_market: bool,
}

impl Default for BookConfig {
    /// The certification configuration: `SimConfig::default()`, joint account, per-sleeve cadence, no filter, no cap,
    /// certification cash.
    fn default() -> Self {
        BookConfig {
            sim: SimConfig::default(),
            account_start: None,
            cadence: BookCadence::PerSleeve,
            mode: AccountMode::Joint,
            allocated_capital: None,
            trade_filter: None,
            cash_policy: CashPolicy::Certification,
            trade_on_closed_market: false,
        }
    }
}

/// A book: sleeves, allocator, optional overlay. The account configuration is separate ([`BookConfig`]).
#[derive(Clone)]
pub struct Book {
    pub sleeves: Vec<SleeveSpec>,
    pub allocator: AllocatorSpec,
    pub overlay: Option<Arc<dyn Overlay>>,
}

impl Book {
    pub fn new(sleeves: Vec<SleeveSpec>) -> Book {
        Book { sleeves, allocator: AllocatorSpec::Fixed, overlay: None }
    }
    pub fn with_allocator(mut self, a: AllocatorSpec) -> Book {
        self.allocator = a;
        self
    }
    pub fn with_overlay(mut self, o: Arc<dyn Overlay>) -> Book {
        self.overlay = Some(o);
        self
    }
}

// ------------------------------------------------------------------------------------------------ overlay hook

/// What the overlay sees at each account bar (design 3.3 "overlay in the loop").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayInput {
    /// Account bar index (0 = the first bar of the account clock).
    pub bar: usize,
    pub time: BarTime,
    /// Simulated POST-COST equity marked at this bar's closes, before this bar's trading.
    pub equity: f64,
    pub initial_equity: f64,
}

/// The overlay's answer for this bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayDecision {
    /// Multiplies the construction risk scale for this bar (1.0 = no change).
    pub scale: f64,
    /// Flatten the whole book on this bar and stay flat (sticky). A halt is reported (`halted_at`); the human resume
    /// rule is out of scope for a backtest.
    pub halt: bool,
}

/// A drawdown/daily-loss ladder or any other equity-driven overlay. The ladder itself is PF2's (`portfolio-construct`);
/// this is only the hook the book calls, so a shared ladder plugs in without touching the simulator.
pub trait Overlay: Send + Sync {
    fn start(&self) -> Box<dyn OverlayRun + '_>;
}

/// The per-run state of an overlay.
pub trait OverlayRun: Send {
    fn step(&mut self, input: &OverlayInput) -> OverlayDecision;
}

// ------------------------------------------------------------------------------------------------ refusals, errors

/// A refusal recorded during a book run (never silent).
#[derive(Clone, Debug, PartialEq)]
pub struct BookRefusal {
    /// Account bar index.
    pub bar: usize,
    pub time: BarTime,
    /// The sleeve concerned; `None` for a whole-book refusal.
    pub sleeve: Option<usize>,
    pub kind: crate::rule::RefusalKind,
    /// The rule's own code, `gross_above_cap` (whole book) or `data_gap` (a named instrument had no bar).
    pub code: &'static str,
    pub message: String,
}

/// Reasons a book run fails.
#[derive(Clone, Debug, PartialEq)]
pub enum BookError {
    /// A single-rule error surfaced by the book (bad config, universe mismatch, rule refusal under `Abort`, invalid
    /// weights, gross cap under `Abort`, non-positive equity).
    Sim(SimError),
    Panel(BookPanelError),
    /// Under `OnRefusal::Abort`: an instrument of a sleeve had no bar although its market was open.
    DataGap {
        sleeve: String,
        instrument: String,
        time: BarTime,
    },
    /// A feature the simulator does not implement yet is refused, never approximated.
    Unsupported(String),
    /// The book itself is inconsistent.
    BadBook(String),
    /// A whole-book construction refusal that is a usage error (e.g. `CashPolicy::Budget` with a short).
    Construct(ConstructRefusal),
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookError::Sim(e) => write!(f, "{e}"),
            BookError::Panel(e) => write!(f, "{e}"),
            BookError::DataGap { sleeve, instrument, time } => {
                write!(f, "data gap: sleeve {sleeve} instrument {instrument} has no bar at {time} although its market is open")
            }
            BookError::Unsupported(m) => write!(f, "unsupported: {m}"),
            BookError::BadBook(m) => write!(f, "bad book: {m}"),
            BookError::Construct(c) => write!(f, "construction refused: {c:?}"),
        }
    }
}

impl std::error::Error for BookError {}

impl From<SimError> for BookError {
    fn from(e: SimError) -> Self {
        BookError::Sim(e)
    }
}

impl From<BookPanelError> for BookError {
    fn from(e: BookPanelError) -> Self {
        BookError::Panel(e)
    }
}
