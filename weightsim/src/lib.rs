//! `weightsim`: a pure, deterministic weight-target sleeve simulator.
//!
//! Implements Stage T1 of `product-mandate/BACKTESTER_TRUTH_DESIGN.md` (Section 2, plus the T1 tests of Section 6.1).
//! A *rule* maps the history of every asset in its universe, up to and including bar `t`, to a signed target weight
//! vector; the simulator owns the decision schedule, the fills, the accounting, the costs and the metrics.
//!
//! What this crate is NOT: it performs no I/O, no vendor calls, no database access, no Python and no network; it
//! has no dependencies at all; it does not depend on, and cannot affect, the `backtest` crate, the worker paths or
//! the job queue. No library rules live here (those are T3); tests use hand-written test rules.
//!
//! # Public API map
//! * [`Panel`], [`HistoryView`], [`PriceSource`]: data and the causal view handed to rules ([`panel`])
//! * [`WeightRule`] (alias [`Rule`]), [`DecisionSchedule`], [`RebalancePolicy`], [`OnRefusal`], [`RuleRefusal`] ([`rule`])
//! * [`simulate`], [`simulate_gross_and_net`], [`SimConfig`], [`SimResult`], [`SimError`] ([`sim`])
//! * [`CostModel`] (with the explicit [`CostModel::ZERO`] preset), [`Financing`] ([`costs`])
//! * [`answer_key_metrics`], [`Metrics`], [`METRIC_DEFINITIONS`] ([`metrics`])
//! * [`harness`]: poisoning / truncation / determinism / cost-identity checks, reusable by the Engine self-test
//! * [`Date`], [`sha256_hex`]: dependency-free helpers
//!
//! # Semantics (numbered as in the design so tests can cite them)
//! S-1 joint calendar (inner join, never filled); S-2 decision at the close of `t` on `0..=t`; S-3 schedule is
//! `Daily` or `LastBarOfMonth`; S-4 policy is `OnDecision` (units drift) or `EveryBar` (fixed weights); S-5 a refusal
//! holds the WHOLE book; S-6 fill at the close of the decision bar (`delay = 0`), the position earns the next
//! close-to-close return; S-7 returns are dated by the bar they are earned on; S-8 cash and borrowing (financing hook);
//! S-9 costs on turnover only; S-10 no constraints in replication (an optional refuse-not-clip `max_gross`); S-11 a
//! gross run and a net run are two executions of the same code; S-12 columnar outputs plus a SHA-256 series digest.
//!
//! # Design choices made where the design is ambiguous (each one favours the simplest, most verifiable simulator)
//!
//! * **C1 Crate has zero dependencies.** The design allows `chrono/serde/serde_json/sha2/hex`; a date type, a SHA-256
//!   (checked against NIST vectors and `sha256sum`) and hand-written errors are ~300 lines and remove the whole
//!   dependency surface from the numerical path and from the ARM64 build.
//! * **C2 Trait name.** The design calls the trait `WeightRule`; the task text calls it `Rule`. `WeightRule` is the
//!   definition and `Rule` a re-export alias.
//! * **C3 Equity is post-cost, defined as `equity_pre - cost`.** Targets are sized on PRE-cost equity (S-9), the cost is
//!   taken out of cash, so post-trade weights are slightly above target by roughly `cost / equity` (second order,
//!   stated by the Purist in the design). `ret` (this run) is post-cost, `ret_pre_cost` is before this bar's cost.
//! * **C4 Cost timing.** The cost of a rebalance at the close of `t` lowers the return dated `t` (not `t+1`).
//! * **C5 First-fill convention.** The counted window starts at `max(first fill bar + 1, first bar dated >= start)`.
//!   The rule sees warm-up history before `start`; the sleeve is already invested when the window opens, exactly as in
//!   the key (S3: the decision of 2015-12-31 earns the 2016-01-01 return). Returns are scale-free, so results do not
//!   depend on `initial_equity`.
//! * **C6 Financing.** `Financing::FlatAnnual{long_bps, short_bps, cash_bps}`: long/short market value is charged
//!   `long_bps`/`short_bps` per annum, the signed cash balance earns `cash_bps` (one rate for both signs), accrued
//!   actual/365 per calendar day between bars on the previous close's notionals and credited to cash at the new
//!   bar. `PolicyRateCarry` is T5 and is not in this crate; the enum is `#[non_exhaustive]` and adding it touches
//!   `Financing::accrual` only.
//! * **C7 Month-end.** `LastBarOfMonth` = the last bar of every calendar month present in the joint calendar,
//!   INCLUDING the final bar of the panel even when its month is incomplete (that is `shadow.py::month_end_dates`).
//!   Consequence for the truncation harness: cutting a panel makes its last bar a decision bar, so that one row is
//!   compared only when both panels agree (see [`harness::check_truncation`]).
//! * **C8 Warm-up vs refusal.** The simulator does not call a rule before `min_history_bars()` bars are visible. A rule
//!   that needs another kind of warm-up (S1 needs 10 month-end closes) returns `RuleRefusal::warmup`, which even
//!   `Abort` tolerates until the first successful decision; after that, every refusal is a refusal.
//! * **C9 Refusal under `EveryBar`.** A refusal keeps the previous standing target; under `EveryBar` the book keeps
//!   being rebalanced to that (unchanged) target each bar, under `OnDecision` no trade happens at all.
//! * **C10 `declared_parameters`** returns `BTreeMap<&'static str, String>` (canonical JSON rendering of each
//!   constant) instead of `serde_json::Value`, to keep C1.
//! * **C11 Delay.** `execution_delay_bars = d` fills the decision of bar `t` at the close of bar `t + d`, sized on the
//!   equity of that bar. Decisions whose fill bar is past the end are never executed.
//! * **C12 Signal flips** (the key's trade counter) count sign changes of the target per asset between successive
//!   successful decisions dated in `[max(start, first decision), end]`, excluding the first (S1 key: 99, S3 key: 138).
//! * **C13 Series digest** covers every per-bar column bit-for-bit (dates, returns, equity, cash, cost, turnover,
//!   financing, exposures, flags, target/held weights, units) plus rule id/version, symbols, cost id and metric
//!   definition. Windowing parameters are not part of it (they select rows of the digested series).
//! * **C14 Percentiles** for exposure statistics are nearest-rank (`sorted[ceil(p*n)-1]`); the median is the mean
//!   of the two middle values for even `n`.

// Deliberate, crate-wide clippy exceptions:
//  * `needless_range_loop`: the simulation path indexes several parallel per-asset vectors in lock step with plain
//    `for i in 0..k` loops on purpose, so the summation order (left to right, asset 0 first) is obvious and identical
//    to the key's; iterator chains would obscure the very thing the identity tests pin down.
//  * `neg_cmp_op_on_partial_ord`: `!(x > 0.0)` is written on purpose so that NaN takes the error branch.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod costs;
pub mod date;
pub mod harness;
pub mod metrics;
pub mod panel;
pub mod rule;
pub mod sha256;
pub mod sim;

pub use costs::{CostModel, Financing};
pub use date::{Date, DateError};
pub use metrics::{answer_key_metrics, Metrics, METRIC_DEFINITIONS};
pub use panel::{HistoryView, Panel, PanelError, PriceSource};
pub use rule::{DecisionSchedule, OnRefusal, RebalancePolicy, RefusalKind, Rule, RuleRefusal, WeightRule};
pub use sha256::{sha256, sha256_hex};
pub use sim::{simulate, simulate_gross_and_net, ExposureStats, Refusal, SimConfig, SimError, SimResult, Window};

/// Crate version recorded in run provenance (`weightsim <version>`).
pub const SIMULATOR_VERSION: &str = concat!("weightsim ", env!("CARGO_PKG_VERSION"));
