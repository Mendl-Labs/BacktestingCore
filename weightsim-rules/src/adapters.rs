//! Thin adapters that express the library decision rules (`reference-rules`) as `weightsim::WeightRule`s.
//!
//! An adapter does four things and nothing else:
//!  1. declares the schedule, rebalance policy and universe (the simulator, not the rule, owns them);
//!  2. converts weightsim's dependency-free [`Date`] to and from chrono's `NaiveDate` ([`to_naive`], [`from_naive`]);
//!  3. builds a `reference_rules::Panel` from the causal `HistoryView` (bars `0..=t` only) and calls the rule with the
//!     REPLAY options (`Options::etf_replay(MonthEndMode::Explicit)`, `Options::crypto_replay()` with
//!     `GapPolicy::Unchecked`), no `as_of`, and with the INSTANCE's instruments and lookback
//!     (`decide_etf_trend_with`, `decide_crypto_trend_with`): the `Default` instance of each adapter is the published
//!     point (the rule crate's constants, no execution lag) and is the only instance the ladder certifies;
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
    decide_crypto_trend_with, decide_etf_trend_with, low_vol_quintile_tilt::quintile_count, GapPolicy,
    LowVolQuintileTiltRule as RrLowVolQuintileTilt, MomentumRankWeighted, MonthEndMode, Options, Panel as RrPanel,
    PriceSeries, RankWeightedRanking, RuleError, CRYPTO_SMA_DAYS, CRYPTO_SYMBOLS, CRYPTO_WEIGHT_PER_INSTRUMENT,
    ETF_SMA_MONTH_ENDS, ETF_SYMBOLS, ETF_WEIGHT_PER_INSTRUMENT, LOW_VOL_QUINTILE_SYMBOLS,
    MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK,
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

/// ETF decision on raw slices (columns in `symbols` order, `sma_month_ends` month-end closes in the average); the
/// decision date is the last date.
pub(crate) fn etf_weights(
    symbols: &[&str],
    sma_month_ends: usize,
    dates: &[Date],
    closes: &[&[f64]],
) -> Result<Vec<f64>, RuleRefusal> {
    let panel = build_panel(symbols, dates, closes)?;
    let date = to_naive(*dates.last().expect("a decision needs at least one bar"));
    let d = decide_etf_trend_with(&panel, date, &Options::etf_replay(MonthEndMode::Explicit), symbols, sma_month_ends)
        .map_err(map_rule_error)?;
    ordered_weights(symbols, d.instruments.iter().map(|i| (i.symbol.as_str(), i.weight)))
}

/// Crypto decision on raw slices (columns in `symbols` order, `sma_days` daily closes in the average); the decision
/// date is the last date.
pub(crate) fn crypto_weights(
    symbols: &[&str],
    sma_days: usize,
    dates: &[Date],
    closes: &[&[f64]],
) -> Result<Vec<f64>, RuleRefusal> {
    let panel = build_panel(symbols, dates, closes)?;
    let date = to_naive(*dates.last().expect("a decision needs at least one bar"));
    let opts = Options { gap_policy: GapPolicy::Unchecked, ..Options::crypto_replay() };
    let d = decide_crypto_trend_with(&panel, date, &opts, symbols, sma_days).map_err(map_rule_error)?;
    ordered_weights(symbols, d.instruments.iter().map(|i| (i.symbol.as_str(), i.weight)))
}

/// Canonical JSON rendering of an instrument list (`["SPY","EFA"]`), the value format of `declared_parameters`.
fn json_string_array(symbols: &[&str]) -> String {
    let mut out = String::from("[");
    for (i, s) in symbols.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push(']');
    out
}

fn view_columns<'a>(h: &HistoryView<'a>) -> Vec<&'a [f64]> {
    (0..h.n_assets()).map(|i| h.closes(i)).collect()
}

/// `etf_trend_faber`: at each month-end, 20% of the sleeve in every ETF (SPY, EFA, IEF, DBC, VNQ) whose month-end
/// close is strictly above the average of its last 10 month-end closes (current one included); units drift between
/// month-ends (`RebalancePolicy::OnDecision`).
///
/// Parametric instance. [`EtfTrendRule::default`] is the PUBLISHED point (`ETF_SYMBOLS`, `ETF_SMA_MONTH_ENDS`, no
/// execution lag): the only instance the ladder certifies, with the `id`, `universe` and `declared_parameters` the
/// Engine's library entry binds to. Any other instance is an implementation point inside a pre-registered envelope of
/// the same documented premium: it keeps the `id` (it is not another strategy), runs the same rule over its own
/// instruments and lookback, is executed `lag` bars after each decision, and additionally declares
/// `execution_lag_bars` and `instruments`, so no non-default instance can bind as the certified entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EtfTrendRule {
    universe: Vec<&'static str>,
    sma_month_ends: usize,
    lag: usize,
}

impl Default for EtfTrendRule {
    /// The published point: `ETF_SYMBOLS`, `ETF_SMA_MONTH_ENDS`, no execution lag.
    fn default() -> Self {
        EtfTrendRule::new(ETF_SYMBOLS.to_vec(), ETF_SMA_MONTH_ENDS, 0)
    }
}

impl EtfTrendRule {
    /// An implementation point: `universe` in column order, `sma_month_ends` month-end closes in the average (the
    /// current one included), decisions effective `lag` bars after the month-end they are taken on.
    ///
    /// # Panics
    /// If `universe` is empty or `sma_month_ends` is 0 (usage errors: an envelope is validated before any of its
    /// points becomes a rule).
    pub fn new(universe: Vec<&'static str>, sma_month_ends: usize, lag: usize) -> Self {
        assert!(!universe.is_empty(), "EtfTrendRule needs at least one instrument");
        assert!(sma_month_ends >= 1, "EtfTrendRule needs a lookback of at least one month-end");
        EtfTrendRule { universe, sma_month_ends, lag }
    }
    pub fn sma_month_ends(&self) -> usize {
        self.sma_month_ends
    }
    pub fn lag(&self) -> usize {
        self.lag
    }
    /// Is this the published (certified) point?
    pub fn is_published(&self) -> bool {
        self.universe == ETF_SYMBOLS && self.sma_month_ends == ETF_SMA_MONTH_ENDS && self.lag == 0
    }
}

impl WeightRule for EtfTrendRule {
    fn id(&self) -> &'static str {
        "etf_trend_faber"
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &self.universe
    }
    /// The published point declares exactly the four keys the library entry pins (byte-identical to the pre-parametric
    /// adapter); every other point adds `execution_lag_bars` and `instruments`.
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        let mut m = BTreeMap::from([
            ("sma_month_ends", self.sma_month_ends.to_string()),
            ("weight_per_instrument", ETF_WEIGHT_PER_INSTRUMENT.to_string()),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ]);
        if !self.is_published() {
            m.insert("execution_lag_bars", self.lag.to_string());
            m.insert("instruments", json_string_array(&self.universe));
        }
        m
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::LastBarOfMonth
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn execution_lag_bars(&self) -> usize {
        self.lag
    }
    fn min_history_bars(&self) -> usize {
        // Not enough month-ends is the rule's own `InsufficientHistory` (a Warmup refusal), not a silent skip.
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        etf_weights(&self.universe, self.sma_month_ends, h.dates(), &view_columns(h))
    }
}

/// `crypto_trend_100d`: every day, 50% of the sleeve in each of BTC and ETH whose close is strictly above the average
/// of its last 100 daily closes (today's included); restored to the standing weights every bar
/// (`RebalancePolicy::EveryBar`).
///
/// Parametric instance, exactly as [`EtfTrendRule`]: [`CryptoTrendRule::default`] is the PUBLISHED point
/// (`CRYPTO_SYMBOLS`, `CRYPTO_SMA_DAYS`, no execution lag) and the only certified one; any other instance keeps the
/// `id`, runs over its own coins and lookback, is executed `lag` bars after each decision and additionally declares
/// `execution_lag_bars` and `instruments`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CryptoTrendRule {
    universe: Vec<&'static str>,
    sma_days: usize,
    lag: usize,
}

impl Default for CryptoTrendRule {
    /// The published point: `CRYPTO_SYMBOLS`, `CRYPTO_SMA_DAYS`, no execution lag.
    fn default() -> Self {
        CryptoTrendRule::new(CRYPTO_SYMBOLS.to_vec(), CRYPTO_SMA_DAYS, 0)
    }
}

impl CryptoTrendRule {
    /// An implementation point: `universe` in column order, `sma_days` daily closes in the average (the decision day
    /// included), decisions effective `lag` bars after the day they are taken on.
    ///
    /// # Panics
    /// If `universe` is empty or `sma_days` is 0 (usage errors: an envelope is validated before any of its points
    /// becomes a rule).
    pub fn new(universe: Vec<&'static str>, sma_days: usize, lag: usize) -> Self {
        assert!(!universe.is_empty(), "CryptoTrendRule needs at least one instrument");
        assert!(sma_days >= 1, "CryptoTrendRule needs a lookback of at least one day");
        CryptoTrendRule { universe, sma_days, lag }
    }
    pub fn sma_days(&self) -> usize {
        self.sma_days
    }
    pub fn lag(&self) -> usize {
        self.lag
    }
    /// Is this the published (certified) point?
    pub fn is_published(&self) -> bool {
        self.universe == CRYPTO_SYMBOLS && self.sma_days == CRYPTO_SMA_DAYS && self.lag == 0
    }
}

impl WeightRule for CryptoTrendRule {
    fn id(&self) -> &'static str {
        "crypto_trend_100d"
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        &self.universe
    }
    /// The published point declares exactly the four keys the library entry pins (byte-identical to the pre-parametric
    /// adapter); every other point adds `execution_lag_bars` and `instruments`.
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        let mut m = BTreeMap::from([
            ("sma_days", self.sma_days.to_string()),
            ("weight_per_instrument", CRYPTO_WEIGHT_PER_INSTRUMENT.to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"every_bar\"".to_string()),
        ]);
        if !self.is_published() {
            m.insert("execution_lag_bars", self.lag.to_string());
            m.insert("instruments", json_string_array(&self.universe));
        }
        m
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn execution_lag_bars(&self) -> usize {
        self.lag
    }
    fn min_history_bars(&self) -> usize {
        self.sma_days
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        crypto_weights(&self.universe, self.sma_days, h.dates(), &view_columns(h))
    }
}

/// `momentum_rank_weighted` (Amendment 14, library entry `etf_cross_sectional_momentum`): the compiled
/// `reference_rules::MomentumRankWeighted` over the five ETFs, ranked daily by trailing 126-bar return, every ranked
/// instrument weighted `(K + 1 - r) / (K (K + 1) / 2)`.
///
/// Unlike the trend rules, the reference rule already IS a `WeightRule`, so this adapter only fixes the published
/// point (`Default`: `ETF_SYMBOLS`, `L = 126`, `DecisionSchedule::Daily`, no execution lag) and stamps the adapter
/// version on the run like every other certified rule. Everything else is the reference rule's, unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct MomentumRankWeightedRule {
    inner: MomentumRankWeighted,
}

impl Default for MomentumRankWeightedRule {
    /// The published point: `ETF_SYMBOLS`, `L = 126`, daily.
    fn default() -> Self {
        MomentumRankWeightedRule::new(ETF_SYMBOLS.to_vec(), MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK)
    }
}

impl MomentumRankWeightedRule {
    /// An implementation point: `universe` in column order, trailing `lookback`-bar return, daily decisions.
    ///
    /// # Panics
    /// If `universe` is empty or `lookback` is 0 (the reference rule's own usage errors).
    pub fn new(universe: Vec<&'static str>, lookback: usize) -> Self {
        MomentumRankWeightedRule { inner: MomentumRankWeighted::new(universe, lookback, DecisionSchedule::Daily) }
    }
    pub fn lookback(&self) -> usize {
        self.inner.lookback()
    }
    /// The typed ranking of the reference rule for this bar (the weights, and which slots lacked history).
    pub fn ranking(&self, h: &HistoryView<'_>) -> RankWeightedRanking {
        self.inner.rank(h)
    }
}

impl WeightRule for MomentumRankWeightedRule {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    /// The four parameters the library entry pins (Amendment 14 section 1, verbatim from the entry's template): the
    /// reference rule's own `L`, plus the schedule, the rebalance policy and the largest weight any one instrument can
    /// get (rank 1 of `K`: `K / (K (K + 1) / 2)`), so the entry's parameters bind with no drift in either direction.
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        let k = self.inner.universe().len() as f64;
        BTreeMap::from([
            ("L", self.inner.lookback().to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
            ("max_weight_per_instrument", (k / (k * (k + 1.0) / 2.0)).to_string()),
        ])
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
        self.inner.target_weights(h)
    }
}

/// `low_vol_quintile_tilt_20d` (Amendment 15, library entry `etf_low_vol_quintile_tilt`): the compiled
/// `reference_rules::LowVolQuintileTiltRule` over the seven placeholder ETFs (DIA, EEM, EFA, GLD, IWM, SPY, TLT). Each
/// day the bottom volatility quintile (`round(0.2 K)`, floored at 1: ONE instrument with K = 7) of the trailing
/// 20-day population stdev of simple returns is held equal-weighted; every bar is re-traded to the target.
///
/// The reference rule already IS a `WeightRule`, so this adapter only stamps the adapter version on the run and adds
/// the one parameter the library entry pins beyond the rule's own four (`max_weight_per_instrument`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LowVolQuintileTiltAdapter {
    inner: RrLowVolQuintileTilt,
}

impl LowVolQuintileTiltAdapter {
    pub fn new() -> Self {
        LowVolQuintileTiltAdapter::default()
    }
}

impl WeightRule for LowVolQuintileTiltAdapter {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn impl_version(&self) -> String {
        ADAPTER_VERSION.to_string()
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    /// The five parameters the library entry pins (Amendment 15 section 1, verbatim from the entry's template): the
    /// reference rule's own `lookback_days`, `quintile_fraction`, `schedule` and `rebalance_policy`, plus the largest
    /// weight any one instrument can get (`1 / quintile_count(K)`, which is 1.0 with K = 7). Written with `{:?}` so
    /// the JSON literal is `1.0`, a float, exactly as the entry declares it.
    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        let mut p = self.inner.declared_parameters();
        let held = quintile_count(LOW_VOL_QUINTILE_SYMBOLS.len()) as f64;
        p.insert("max_weight_per_instrument", format!("{:?}", 1.0 / held));
        p
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
        self.inner.target_weights(h)
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
    fn execution_lag_bars(&self) -> usize {
        self.inner.execution_lag_bars()
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
        let e = EtfTrendRule::default().declared_parameters();
        assert_eq!(e["sma_month_ends"], "10");
        assert_eq!(e["weight_per_instrument"], "0.2");
        let c = CryptoTrendRule::default().declared_parameters();
        assert_eq!(c["sma_days"], "100");
        assert_eq!(c["weight_per_instrument"], "0.5");
    }

    /// The published point's map is a LITERAL copy of what the pre-parametric unit structs returned: same four keys,
    /// same values, nothing more. The Engine's binding (`diff_against`) compares the library entry against exactly
    /// this map, so any extra key or changed value would force an `entry_hash` bump and make the rebalancer's
    /// `check_entry` refuse every existing plan.
    #[test]
    fn default_declared_parameters_are_exactly_the_pre_parametric_maps() {
        let etf_before: BTreeMap<&'static str, String> = BTreeMap::from([
            ("sma_month_ends", "10".to_string()),
            ("weight_per_instrument", "0.2".to_string()),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ]);
        assert_eq!(EtfTrendRule::default().declared_parameters(), etf_before);
        let crypto_before: BTreeMap<&'static str, String> = BTreeMap::from([
            ("sma_days", "100".to_string()),
            ("weight_per_instrument", "0.5".to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"every_bar\"".to_string()),
        ]);
        assert_eq!(CryptoTrendRule::default().declared_parameters(), crypto_before);
        // An explicitly constructed published point is the default, not a "non-default" instance.
        assert_eq!(EtfTrendRule::new(ETF_SYMBOLS.to_vec(), ETF_SMA_MONTH_ENDS, 0), EtfTrendRule::default());
        assert_eq!(EtfTrendRule::new(ETF_SYMBOLS.to_vec(), ETF_SMA_MONTH_ENDS, 0).declared_parameters(), etf_before);
        assert_eq!(CryptoTrendRule::new(CRYPTO_SYMBOLS.to_vec(), CRYPTO_SMA_DAYS, 0), CryptoTrendRule::default());
        assert!(EtfTrendRule::default().is_published());
        assert!(CryptoTrendRule::default().is_published());
        assert_eq!(EtfTrendRule::default().execution_lag_bars(), 0);
        assert_eq!(CryptoTrendRule::default().execution_lag_bars(), 0);
    }

    /// A non-default point keeps the id and declares what makes it a different point: lag and instruments, on top of
    /// the (possibly changed) lookback. Each axis alone is enough to leave the published point.
    #[test]
    fn non_default_instances_keep_the_id_and_declare_lag_and_instruments() {
        let lagged = EtfTrendRule::new(ETF_SYMBOLS.to_vec(), ETF_SMA_MONTH_ENDS, 3);
        assert!(!lagged.is_published());
        assert_eq!(lagged.id(), "etf_trend_faber");
        assert_eq!(lagged.universe(), EtfTrendRule::default().universe());
        assert_eq!(lagged.execution_lag_bars(), 3);
        let p = lagged.declared_parameters();
        assert_eq!(p.len(), 6);
        assert_eq!(p["execution_lag_bars"], "3");
        assert_eq!(p["instruments"], "[\"SPY\",\"EFA\",\"IEF\",\"DBC\",\"VNQ\"]");
        assert_eq!(p["sma_month_ends"], "10");
        assert_eq!(p["weight_per_instrument"], "0.2");

        let lookback = EtfTrendRule::new(ETF_SYMBOLS.to_vec(), 8, 0);
        assert!(!lookback.is_published());
        let p = lookback.declared_parameters();
        assert_eq!(p["sma_month_ends"], "8");
        assert_eq!(p["execution_lag_bars"], "0");
        assert!(p.contains_key("instruments"));

        let proxies = EtfTrendRule::new(vec!["IVV", "EFA", "IEF", "DBC", "VNQ"], ETF_SMA_MONTH_ENDS, 0);
        assert!(!proxies.is_published());
        assert_eq!(proxies.universe(), &["IVV", "EFA", "IEF", "DBC", "VNQ"]);
        assert_eq!(proxies.declared_parameters()["instruments"], "[\"IVV\",\"EFA\",\"IEF\",\"DBC\",\"VNQ\"]");

        let crypto = CryptoTrendRule::new(vec!["BTC"], 80, 1);
        assert!(!crypto.is_published());
        assert_eq!(crypto.id(), "crypto_trend_100d");
        assert_eq!(crypto.universe(), &["BTC"]);
        assert_eq!(crypto.min_history_bars(), 80);
        assert_eq!(crypto.execution_lag_bars(), 1);
        let p = crypto.declared_parameters();
        assert_eq!(p.len(), 6);
        assert_eq!(p["sma_days"], "80");
        assert_eq!(p["execution_lag_bars"], "1");
        assert_eq!(p["instruments"], "[\"BTC\"]");
        assert_eq!(p["schedule"], "\"daily\"");

        // FlatUntil forwards the lag like every other rule fact.
        assert_eq!(FlatUntil::new(lagged.clone(), Date::new(2016, 1, 1).unwrap()).execution_lag_bars(), 3);
        assert_eq!(FlatUntil::new(EtfTrendRule::default(), Date::new(2016, 1, 1).unwrap()).execution_lag_bars(), 0);
    }

    #[test]
    fn json_string_array_is_compact_and_escapes() {
        assert_eq!(json_string_array(&["SPY", "EFA"]), "[\"SPY\",\"EFA\"]");
        assert_eq!(json_string_array(&["BTC"]), "[\"BTC\"]");
        assert_eq!(json_string_array(&["a\"b", "c\\d"]), "[\"a\\\"b\",\"c\\\\d\"]");
    }

    #[test]
    fn schedules_policies_universes_are_the_documented_ones() {
        assert_eq!(EtfTrendRule::default().decision_schedule(), DecisionSchedule::LastBarOfMonth);
        assert_eq!(EtfTrendRule::default().rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(EtfTrendRule::default().universe(), &["SPY", "EFA", "IEF", "DBC", "VNQ"]);
        assert_eq!(CryptoTrendRule::default().decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(CryptoTrendRule::default().rebalance_policy(), RebalancePolicy::EveryBar);
        assert_eq!(CryptoTrendRule::default().universe(), &["BTC", "ETH"]);
        assert_eq!(CryptoTrendRule::default().min_history_bars(), 100);
        assert_eq!(EtfTrendRule::default().id(), "etf_trend_faber");
        assert_eq!(CryptoTrendRule::default().id(), "crypto_trend_100d");
    }
}
