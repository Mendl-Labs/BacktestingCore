//! `weightsim-rules`: the library decision rules inside the weight-target simulator, and the replication ladder that
//! certifies the pair against the pinned answer key.
//!
//! Stage T3 (milestone A: ETF and crypto certified in CI) of `product-mandate/BACKTESTER_TRUTH_DESIGN.md`.
//!
//! * [`adapters`]: [`EtfTrendRule`] and [`CryptoTrendRule`], thin `weightsim::WeightRule` adapters around
//!   `reference_rules::{decide_etf_trend, decide_crypto_trend}` (replay options, date conversion, refusal mapping).
//!   The FX adapter is Stage T5 and is deliberately absent.
//! * [`ladder`]: pure certification functions. Load and verify the answer-key fixtures (`ladder::Fixtures`), run the
//!   rules gross and net through the simulator, and apply pre-registration Amendment 11 exactly: Tier I bands,
//!   Tier II per-bar identity, Tier III weight agreement, Tier IV eight named mutants that must each fail. Result:
//!   a [`ladder::LadderReport`] (plain structs, `Display`) and [`ladder::self_test`], which is `Err` unless
//!   everything passes.
//!
//! Dependencies: `weightsim` and `reference-rules` (by path; neither depends on the other or on this crate) and
//! `chrono` (for the rule crate's date type). No serde, no I/O other than reading fixture files, no network.
//!
//! Vendor-derived data never lives in this (public) repository: the real fixtures stay in the private Engine repo and
//! the real-data tests are env-gated (`WEIGHTSIM_RULES_LADDER_DIR`). The always-on tests use the synthetic fixtures
//! of `../weightsim/tests/fixtures/` and in-memory synthetic answer keys.

// Deliberate, crate-wide clippy exceptions (the same two as `weightsim`, whose loops and NaN-aware comparisons this
// crate mirrors): `needless_range_loop` (parallel per-asset vectors are indexed in lock step so the summation order is
// the key's), `neg_cmp_op_on_partial_ord` (`!(x <= tol)` is written on purpose so that NaN counts as a failure).
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod adapters;
pub mod ladder;

pub use adapters::{from_naive, map_rule_error, to_naive, CryptoTrendRule, EtfTrendRule, FlatUntil, ADAPTER_VERSION};
