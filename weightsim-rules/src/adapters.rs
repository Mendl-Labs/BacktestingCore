//! Thin adapters that express the library decision rules (`reference-rules`) as `weightsim::WeightRule`s.
//!
//! An adapter does four things and nothing else:
//!  1. declares the schedule, rebalance policy and universe (the simulator, not the rule, owns them);
//!  2. converts weightsim's dependency-free [`Date`] to and from chrono's `NaiveDate` ([`to_naive`], [`from_naive`]);
//!  3. builds a `reference_rules::Panel` from the causal `HistoryView` (bars `0..=t` only) and calls the rule with the
//!     REPLAY options (`Options::etf_replay(MonthEndMode::Explicit)`, `Options::crypto_replay()` with
//!     `GapPolicy::Unchecked`), no `as_of`;
//!  4. maps `RuleError` to `RuleRefusal` ([`map_rule_error`]): `InsufficientHistory` is `Warmup`, gaps and stale data are
//!     `Data`, everything else is `Other`, each with a stable code.
//!
//! Replay options, not live options: the ladder replays a 10-year history in which the decision date is the last bar of
//! the (already truncated) view, so `MonthEndMode::Explicit` is exact (the view ends on the decision date, so no later
//! bar of the month can exist). Crypto uses `GapPolicy::Unchecked` because the joint BTC/ETH calendar of the pinned
//! fixture has holes in 2015 (ETH is missing six days, BTC one) that the 100-bar windows of the first 2016 decisions
//! still contain; the live gap policy stays strict in the rebalancer.
//!
//! The FX rule is NOT adapted here (Stage T5).

use std::collections::BTreeMap;

use chrono::{Datelike, NaiveDate};
use reference_rules::{
    decide_crypto_trend, decide_etf_trend, GapPolicy, MonthEndMode, Options, Panel as RrPanel, PriceSeries, RuleError,
    CRYPTO_SMA_DAYS, CRYPTO_SYMBOLS, CRYPTO_WEIGHT_PER_INSTRUMENT, ETF_SMA_MONTH_ENDS, ETF_SYMBOLS,
    ETF_WEIGHT_PER_INSTRUMENT,
};
use weightsim::{Date, DecisionSchedule, HistoryView, RebalancePolicy, RefusalKind, RuleRefusal, WeightRule};

/// Version string recorded in every run (it is part of the series digest).
pub const ADAPTER_VERSION: &str = concat!("weightsim-rules ", env!("CARGO_PKG_VERSION"), " over reference-rules");

/// weightsim `Date` to chrono `NaiveDate`. Infallible: a `Date` is always a valid calendar date.
pub fn to_naive(d: Date) -> NaiveDate {
    NaiveDate::from_ymd_opt(d.year(), u32::from(d.month()), u32::from(d.day()))
        .expect("a weightsim::Date is always a valid calendar date")
}

/// chrono `NaiveDate` to weightsim `Date`.
pub fn from_naive(n: NaiveDate) -> Date {
    Date::new(n.year(), n.month() as u8, n.day() as u8).expect("a chrono NaiveDate is always a valid calendar date")
}

/// Stable machine code for a `RuleError` variant (recorded in the run's refusals).
fn error_code(e: &RuleError) -> &'static str {
    match e {
        RuleError::LengthMismatch { .. } => "length_mismatch",
        RuleError::EmptySeries { .. } => "empty_series",
        RuleError::InvalidSymbol { .. } => "invalid_symbol",
        RuleError::DuplicateSymbol { .. } => "duplicate_symbol",
        RuleError::NonMonotonic { .. } => "non_monotonic",
        RuleError::InvalidPrice { .. } => "invalid_price",
        RuleError::PriceScaleTooWide { .. } => "price_scale_too_wide",
        RuleError::MissingInstrument { .. } => "missing_instrument",
        RuleError::DateNotInPanel { .. } => "date_not_in_panel",
        RuleError::NotMonthEnd { .. } => "not_month_end",
        RuleError::MonthNotComplete { .. } => "month_not_complete",
        RuleError::PanelDoesNotEndOnDecisionDate { .. } => "panel_does_not_end_on_decision_date",
        RuleError::InsufficientHistory { .. } => "insufficient_history",
        RuleError::StaleData { .. } => "stale_data",
        RuleError::FormingBar { .. } => "forming_bar",
        RuleError::DataGap { .. } => "data_gap",
        RuleError::MonthEndMismatch { .. } => "month_end_mismatch",
        RuleError::NoCompletedMonth { .. } => "no_completed_month",
        RuleError::ZeroVolatility { .. } => "zero_volatility",
        RuleError::DegenerateSleeveVolatility { .. } => "degenerate_sleeve_volatility",
        RuleError::NonFiniteValue { .. } => "non_finite_value",
    }
}

/// `RuleError` to `RuleRefusal`: not enough history is a `Warmup` (tolerated by `Abort` only until the first
/// successful decision), a gap or stale data is a `Data` refusal, everything else is `Other`.
pub fn map_rule_error(e: RuleError) -> RuleRefusal {
    let message = e.to_string();
    let code = error_code(&e);
    let kind = match e {
        RuleError::InsufficientHistory { .. } => RefusalKind::Warmup,
        RuleError::DataGap { .. } | RuleError::StaleData { .. } => RefusalKind::Data,
        _ => RefusalKind::Other,
    };
    RuleRefusal::new(kind, code, message)
}

/// Build the rule crate's panel from raw slices (bars `0..=t` of every asset, universe order).
fn build_panel(symbols: &[&str], dates: &[Date], closes: &[&[f64]]) -> Result<RrPanel, RuleRefusal> {
    if symbols.len() != closes.len() {
        return Err(RuleRefusal::new(RefusalKind::Other, "universe_mismatch", "symbols and price columns differ"));
    }
    let naive: Vec<NaiveDate> = dates.iter().map(|&d| to_naive(d)).collect();
    let mut series = Vec::with_capacity(symbols.len());
    for (s, c) in symbols.iter().zip(closes) {
        series.push(PriceSeries::new(*s, naive.clone(), c.to_vec()).map_err(map_rule_error)?);
    }
    RrPanel::new(series).map_err(map_rule_error)
}

/// Weights of the instruments of a decision, checked to follow `symbols` order.
fn ordered_weights<'a>(
    symbols: &[&str],
    instruments: impl Iterator<Item = (&'a str, f64)>,
) -> Result<Vec<f64>, RuleRefusal> {
    let mut out = Vec::with_capacity(symbols.len());
    for (want, (got, w)) in symbols.iter().zip(instruments) {
        if *want != got {
            return Err(RuleRefusal::new(
                RefusalKind::Other,
                "instrument_order",
                format!("expected {want}, got {got}"),
            ));
        }
        out.push(w);
    }
    if out.len() != symbols.len() {
        return Err(RuleRefusal::new(RefusalKind::Other, "instrument_count", "decision has too few instruments"));
    }
    Ok(out)
}

/// ETF decision on raw slices; the decision date is the last date.
pub(crate) fn etf_weights(dates: &[Date], closes: &[&[f64]]) -> Result<Vec<f64>, RuleRefusal> {
    let panel = build_panel(&ETF_SYMBOLS, dates, closes)?;
    let date = to_naive(*dates.last().expect("a decision needs at least one bar"));
    let d = decide_etf_trend(&panel, date, &Options::etf_replay(MonthEndMode::Explicit)).map_err(map_rule_error)?;
    ordered_weights(&ETF_SYMBOLS, d.instruments.iter().map(|i| (i.symbol.as_str(), i.weight)))
}

/// Crypto decision on raw slices; the decision date is the last date.
pub(crate) fn crypto_weights(dates: &[Date], closes: &[&[f64]]) -> Result<Vec<f64>, RuleRefusal> {
    let panel = build_panel(&CRYPTO_SYMBOLS, dates, closes)?;
    let date = to_naive(*dates.last().expect("a decision needs at least one bar"));
    let opts = Options { gap_policy: GapPolicy::Unchecked, ..Options::crypto_replay() };
    let d = decide_crypto_trend(&panel, date, &opts).map_err(map_rule_error)?;
    ordered_weights(&CRYPTO_SYMBOLS, d.instruments.iter().map(|i| (i.symbol.as_str(), i.weight)))
}

fn view_columns<'a>(h: &HistoryView<'a>) -> Vec<&'a [f64]> {
    (0..h.n_assets()).map(|i| h.closes(i)).collect()
}

/// `etf_trend_faber`: at each month-end, 20% of the sleeve in every ETF (SPY, EFA, IEF, DBC, VNQ) whose month-end
/// close is strictly above the average of its last 10 month-end closes (current one included); units drift between
/// month-ends (`RebalancePolicy::OnDecision`).
#[derive(Clone, Copy, Debug, Default)]
pub struct EtfTrendRule;

impl WeightRule for EtfTrendRule {
    fn id(&self) -> &'static str {
        "etf_trend_faber"
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &ETF_SYMBOLS
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("sma_month_ends", ETF_SMA_MONTH_ENDS.to_string()),
            ("weight_per_instrument", ETF_WEIGHT_PER_INSTRUMENT.to_string()),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::LastBarOfMonth
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        // Not enough month-ends is the rule's own `InsufficientHistory` (a Warmup refusal), not a silent skip.
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        etf_weights(h.dates(), &view_columns(h))
    }
}

/// `crypto_trend_100d`: every day, 50% of the sleeve in each of BTC and ETH whose close is strictly above the average
/// of its last 100 daily closes (today's included); restored to the standing weights every bar
/// (`RebalancePolicy::EveryBar`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CryptoTrendRule;

impl WeightRule for CryptoTrendRule {
    fn id(&self) -> &'static str {
        "crypto_trend_100d"
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &CRYPTO_SYMBOLS
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("sma_days", CRYPTO_SMA_DAYS.to_string()),
            ("weight_per_instrument", CRYPTO_WEIGHT_PER_INSTRUMENT.to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"every_bar\"".to_string()),
        ])
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        CRYPTO_SMA_DAYS
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        crypto_weights(h.dates(), &view_columns(h))
    }
}

/// A rule that starts the sleeve flat: every decision dated before `first_trade` is the all-zero target (cash), every
/// later decision is the inner rule's. The rule still sees the whole history (warm-up), it just is not traded until
/// `first_trade`, where the book enters its first position and pays the entry cost.
///
/// This is how a run reproduces the answer key's ledger, which is "flat with equity 1.0 at the close of the bar before
/// the window": without it a rule that has traded for months before the window (crypto, from its 100th bar) carries
/// that history's equity and open position into the window, and the NET series differs from the key by the entry cost
/// on the first bars (gross returns are unaffected). Schedule, policy, universe and minimum history are the inner
/// rule's. Proposed for promotion to a `weightsim::SimConfig` field (`trade_from`) in a later stage.
#[derive(Clone, Debug)]
pub struct FlatUntil<R: WeightRule> {
    inner: R,
    first_trade: Date,
}

impl<R: WeightRule> FlatUntil<R> {
    pub fn new(inner: R, first_trade: Date) -> Self {
        FlatUntil { inner, first_trade }
    }
    pub fn first_trade(&self) -> Date {
        self.first_trade
    }
}

impl<R: WeightRule> WeightRule for FlatUntil<R> {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn impl_version(&self) -> String {
        format!("{}, flat until {}", self.inner.impl_version(), self.first_trade)
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        self.inner.declared_parameters()
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.inner.decision_schedule()
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.inner.rebalance_policy()
    }
    fn min_history_bars(&self) -> usize {
        self.inner.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        if h.date() < self.first_trade {
            Ok(vec![0.0; h.n_assets()])
        } else {
            self.inner.target_weights(h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn date_conversion_is_exact_both_ways_including_leap_days_and_year_ends() {
        for (y, m, d) in [(2020, 2, 29), (2019, 12, 31), (2016, 1, 1), (2026, 8, 31), (1970, 1, 1), (2000, 3, 1)] {
            let w = Date::new(y, m as u8, d as u8).unwrap();
            assert_eq!(to_naive(w), nd(y, m, d));
            assert_eq!(from_naive(nd(y, m, d)), w);
            assert_eq!(w.days_since_epoch(), i64::from(nd(y, m, d).num_days_from_ce()) - 719_163);
        }
    }

    #[test]
    fn error_mapping_kinds_and_codes() {
        let d = nd(2020, 1, 31);
        let cases: Vec<(RuleError, RefusalKind, &str)> = vec![
            (
                RuleError::InsufficientHistory { symbol: "SPY".into(), needed: 10, have: 3 },
                RefusalKind::Warmup,
                "insufficient_history",
            ),
            (
                RuleError::DataGap { symbol: "SPY".into(), from: d, to: d, missing: 9, tolerance: 3, weekdays: true },
                RefusalKind::Data,
                "data_gap",
            ),
            (
                RuleError::StaleData { symbol: "SPY".into(), last_bar: d, as_of: d, max_stale_days: 5 },
                RefusalKind::Data,
                "stale_data",
            ),
            (RuleError::MissingInstrument { symbol: "SPY".into() }, RefusalKind::Other, "missing_instrument"),
            (
                RuleError::NotMonthEnd { symbol: "SPY".into(), decision_date: d, later_bar_in_month: d },
                RefusalKind::Other,
                "not_month_end",
            ),
            (RuleError::FormingBar { symbol: "SPY".into(), bar_date: d, as_of: d }, RefusalKind::Other, "forming_bar"),
            (
                RuleError::MonthEndMismatch { symbol_a: "A".into(), date_a: d, symbol_b: "B".into(), date_b: d },
                RefusalKind::Other,
                "month_end_mismatch",
            ),
            (RuleError::PriceScaleTooWide { symbol: "SPY".into() }, RefusalKind::Other, "price_scale_too_wide"),
        ];
        for (e, kind, code) in cases {
            let r = map_rule_error(e.clone());
            assert_eq!(r.kind, kind, "{e:?}");
            assert_eq!(r.code, code, "{e:?}");
            assert!(!r.message.is_empty());
        }
    }

    #[test]
    fn declared_parameters_are_the_rule_crates_constants() {
        let e = EtfTrendRule.declared_parameters();
        assert_eq!(e["sma_month_ends"], "10");
        assert_eq!(e["weight_per_instrument"], "0.2");
        let c = CryptoTrendRule.declared_parameters();
        assert_eq!(c["sma_days"], "100");
        assert_eq!(c["weight_per_instrument"], "0.5");
    }

    #[test]
    fn schedules_policies_universes_are_the_documented_ones() {
        assert_eq!(EtfTrendRule.decision_schedule(), DecisionSchedule::LastBarOfMonth);
        assert_eq!(EtfTrendRule.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(EtfTrendRule.universe(), &["SPY", "EFA", "IEF", "DBC", "VNQ"]);
        assert_eq!(CryptoTrendRule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(CryptoTrendRule.rebalance_policy(), RebalancePolicy::EveryBar);
        assert_eq!(CryptoTrendRule.universe(), &["BTC", "ETH"]);
        assert_eq!(CryptoTrendRule.min_history_bars(), 100);
        assert_eq!(EtfTrendRule.id(), "etf_trend_faber");
        assert_eq!(CryptoTrendRule.id(), "crypto_trend_100d");
    }
}
