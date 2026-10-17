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
    decide_crypto_trend_with, decide_etf_trend_with_weight, GapPolicy, MomentumRankWeighted, MonthEndMode, Options,
    Panel as RrPanel, PriceSeries, RankWeightedRanking, RuleError, CRYPTO_SMA_DAYS, CRYPTO_SYMBOLS,
    CRYPTO_WEIGHT_PER_INSTRUMENT, ETF_SMA_MONTH_ENDS, ETF_SYMBOLS, ETF_WEIGHT_PER_INSTRUMENT,
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
    weight_per_instrument: f64,
    dates: &[Date],
    closes: &[&[f64]],
) -> Result<Vec<f64>, RuleRefusal> {
    let panel = build_panel(symbols, dates, closes)?;
    let date = to_naive(*dates.last().expect("a decision needs at least one bar"));
    let opts = Options::etf_replay(MonthEndMode::Explicit);
    let d = decide_etf_trend_with_weight(&panel, date, &opts, symbols, sma_month_ends, weight_per_instrument)
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

/// How the ETF trend rule sizes an instrument whose signal is `Long` (R41-A(1)): a policy, not a tunable number.
///
/// * [`WeightPolicy::FixedConstant`]: the same fraction of the sleeve for every held instrument, whatever the
///   universe size. The default is `FixedConstant(ETF_WEIGHT_PER_INSTRUMENT)` (0.20, Faber's 1/5): the published rule.
/// * [`WeightPolicy::EqualOverN`]: `1.0 / N` per held instrument for an N-instrument universe, so the maximum gross is
///   100% (the rest stays cash) at any N. `1.0 / 5.0` is bit-identical to the literal `0.20` (IEEE division is
///   correctly rounded), so the canonical five-ETF universe is unchanged under either policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightPolicy {
    FixedConstant(f64),
    EqualOverN,
}

impl Default for WeightPolicy {
    /// The published sizing: `FixedConstant(ETF_WEIGHT_PER_INSTRUMENT)`.
    fn default() -> Self {
        WeightPolicy::FixedConstant(ETF_WEIGHT_PER_INSTRUMENT)
    }
}

impl WeightPolicy {
    /// Target weight of one held instrument on an `n`-instrument universe.
    pub fn weight_per_instrument(self, n: usize) -> f64 {
        match self {
            WeightPolicy::FixedConstant(w) => w,
            WeightPolicy::EqualOverN => 1.0 / n as f64,
        }
    }
    /// Stable name recorded in `declared_parameters` of a non-default instance.
    fn name(self) -> &'static str {
        match self {
            WeightPolicy::FixedConstant(_) => "\"fixed_constant\"",
            WeightPolicy::EqualOverN => "\"equal_1_over_N\"",
        }
    }
}

/// Smallest universe an [`WeightPolicy::EqualOverN`] ETF trend rule accepts (R41-A(1)(i)): at N >= 5, `1/N <= 0.20`,
/// so no cell is more concentrated per name than the canonical five-ETF point.
pub const ETF_EQUAL_OVER_N_MIN_UNIVERSE: usize = 5;

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
///
/// R41-A(1): the sizing is a [`WeightPolicy`], default `FixedConstant(0.20)` (the published rule, byte-identical
/// `declared_parameters`, series and digests). [`EtfTrendRule::with_weight_policy`] selects `EqualOverN` (1/N per
/// held instrument, N >= 5); an instance with a non-default policy additionally declares `weight_policy`, and its
/// `weight_per_instrument` is the derived weight.
#[derive(Clone, Debug, PartialEq)]
pub struct EtfTrendRule {
    universe: Vec<&'static str>,
    sma_month_ends: usize,
    lag: usize,
    weight_policy: WeightPolicy,
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
        EtfTrendRule::with_weight_policy(universe, sma_month_ends, lag, WeightPolicy::default())
    }
    /// [`EtfTrendRule::new`] with an explicit [`WeightPolicy`] (the constructor path used for R41 cells).
    ///
    /// # Panics
    /// As `new`, and if the policy is `EqualOverN` on a universe of fewer than [`ETF_EQUAL_OVER_N_MIN_UNIVERSE`]
    /// (5) instruments, or `FixedConstant` with a weight outside `(0, 1]`.
    pub fn with_weight_policy(
        universe: Vec<&'static str>,
        sma_month_ends: usize,
        lag: usize,
        weight_policy: WeightPolicy,
    ) -> Self {
        assert!(!universe.is_empty(), "EtfTrendRule needs at least one instrument");
        assert!(sma_month_ends >= 1, "EtfTrendRule needs a lookback of at least one month-end");
        if let WeightPolicy::FixedConstant(w) = weight_policy {
            assert!(w > 0.0 && w <= 1.0, "EtfTrendRule fixed weight must be in (0, 1], got {w}");
        }
        assert!(
            weight_policy != WeightPolicy::EqualOverN || universe.len() >= ETF_EQUAL_OVER_N_MIN_UNIVERSE,
            "EtfTrendRule with EqualOverN needs a universe of at least {ETF_EQUAL_OVER_N_MIN_UNIVERSE}, got {}",
            universe.len()
        );
        EtfTrendRule { universe, sma_month_ends, lag, weight_policy }
    }
    pub fn weight_policy(&self) -> WeightPolicy {
        self.weight_policy
    }
    /// Target weight of each held instrument under this instance's policy and universe size.
    pub fn weight_per_instrument(&self) -> f64 {
        self.weight_policy.weight_per_instrument(self.universe.len())
    }
    pub fn sma_month_ends(&self) -> usize {
        self.sma_month_ends
    }
    pub fn lag(&self) -> usize {
        self.lag
    }
    /// Is this the published (certified) point?
    pub fn is_published(&self) -> bool {
        self.universe == ETF_SYMBOLS
            && self.sma_month_ends == ETF_SMA_MONTH_ENDS
            && self.lag == 0
            && self.weight_policy == WeightPolicy::default()
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
            ("weight_per_instrument", self.weight_per_instrument().to_string()),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ]);
        if !self.is_published() {
            m.insert("execution_lag_bars", self.lag.to_string());
            m.insert("instruments", json_string_array(&self.universe));
            // Only a non-default policy is declared, so every pre-R41 non-default instance keeps its exact map.
            if self.weight_policy != WeightPolicy::default() {
                m.insert("weight_policy", self.weight_policy.name().to_string());
            }
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
        etf_weights(&self.universe, self.sma_month_ends, self.weight_per_instrument(), h.dates(), &view_columns(h))
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

    // ------------------------------------------------------------------------------------------------------------
    // R41-A(1): WeightPolicy, and R41-A(3)(b): the per-rule N-generality attestation of the trend rule.
    // ------------------------------------------------------------------------------------------------------------

    const NAMES: [&str; 11] = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11"];

    /// Seeded linear congruential generator (no dependency, fully reproducible).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    /// `n` consecutive weekdays from 2015-01-01 (no holidays: every gap is the weekend).
    fn weekday_calendar(n: usize) -> Vec<NaiveDate> {
        let mut out = Vec::with_capacity(n);
        let mut d = nd(2015, 1, 1);
        while out.len() < n {
            if d.weekday().number_from_monday() <= 5 {
                out.push(d);
            }
            d = d.succ_opt().unwrap();
        }
        out
    }

    /// Quarter-dollar tick prices, one column per asset: every sum and comparison the rule makes on `ticks * 0.25` is
    /// exact in f64, so the integer reference below is the exact answer, not an approximation. Column roles repeat
    /// every four assets: a random walk, a never-falling walk (always above its average), a never-rising walk (floors
    /// at 8 ticks and stays there: always at or below its average), and a random walk that is dead flat for 600 bars
    /// (exact `close == average` ties).
    fn synthetic_ticks(n_assets: usize, n_bars: usize, seed: u64) -> Vec<Vec<i64>> {
        let mut rng = Lcg(seed);
        (0..n_assets)
            .map(|j| {
                let mut p: i64 = 400 + 12 * j as i64;
                let mut v = Vec::with_capacity(n_bars);
                for t in 0..n_bars {
                    let step = (rng.next() % 9) as i64 - 4;
                    let step = match j % 4 {
                        1 => step.abs(),
                        2 => -step.abs(),
                        3 if (300..900).contains(&t) => 0,
                        _ => step,
                    };
                    p = (p + step).max(8);
                    v.push(p);
                }
                v
            })
            .collect()
    }

    fn ticks_to_closes(ticks: &[i64]) -> Vec<f64> {
        ticks.iter().map(|&k| k as f64 * 0.25).collect()
    }

    /// Independent reference for the trend rule's decision at bar `t` (integer arithmetic, no shared code): the last
    /// `months` month-end closes of the view `0..=t` (its last bar counts as a month-end), compared with their sum.
    /// `None` when the view holds fewer than `months` month-ends (the rule must refuse with a warm-up).
    fn reference_trend_signs(
        dates: &[NaiveDate],
        ticks: &[Vec<i64>],
        t: usize,
        months: usize,
    ) -> Option<Vec<std::cmp::Ordering>> {
        let mut month_ends = Vec::new();
        for i in 0..=t {
            if i == t || dates[i].month() != dates[i + 1].month() {
                month_ends.push(i);
            }
        }
        if month_ends.len() < months {
            return None;
        }
        let window = &month_ends[month_ends.len() - months..];
        Some(
            ticks
                .iter()
                .map(|col| {
                    let sum: i64 = window.iter().map(|&i| col[i]).sum();
                    (col[t] * months as i64).cmp(&sum)
                })
                .collect(),
        )
    }

    #[test]
    fn weight_policy_default_is_the_published_constant_and_equal_over_n_is_one_over_n() {
        assert_eq!(WeightPolicy::default(), WeightPolicy::FixedConstant(ETF_WEIGHT_PER_INSTRUMENT));
        assert_eq!(WeightPolicy::default().weight_per_instrument(5).to_bits(), 0.20_f64.to_bits());
        // A fixed constant ignores N; EqualOverN is exactly 1/N.
        assert_eq!(WeightPolicy::FixedConstant(0.25).weight_per_instrument(11), 0.25);
        for n in 1..=20usize {
            assert_eq!(WeightPolicy::EqualOverN.weight_per_instrument(n).to_bits(), (1.0_f64 / n as f64).to_bits());
        }
        // N = 5: bit-identical to the literal 0.20 (IEEE division is correctly rounded), so the canonical point holds.
        assert_eq!(WeightPolicy::EqualOverN.weight_per_instrument(5).to_bits(), ETF_WEIGHT_PER_INSTRUMENT.to_bits());
        assert_eq!(WeightPolicy::EqualOverN.weight_per_instrument(5).to_bits(), 0.20_f64.to_bits());
    }

    /// K5: the default instance, however it is built, is the published point with exactly today's four-key map.
    #[test]
    fn the_default_policy_leaves_the_published_rule_byte_identical() {
        let four_keys: BTreeMap<&'static str, String> = BTreeMap::from([
            ("sma_month_ends", "10".to_string()),
            ("weight_per_instrument", "0.2".to_string()),
            ("schedule", "\"last_bar_of_month\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ]);
        let explicit = EtfTrendRule::with_weight_policy(
            ETF_SYMBOLS.to_vec(),
            ETF_SMA_MONTH_ENDS,
            0,
            WeightPolicy::FixedConstant(ETF_WEIGHT_PER_INSTRUMENT),
        );
        assert_eq!(explicit, EtfTrendRule::default());
        assert!(explicit.is_published());
        assert_eq!(explicit.weight_policy(), WeightPolicy::default());
        assert_eq!(explicit.weight_per_instrument().to_bits(), ETF_WEIGHT_PER_INSTRUMENT.to_bits());
        assert_eq!(explicit.declared_parameters(), four_keys);
        assert_eq!(EtfTrendRule::default().declared_parameters(), four_keys);
        assert_eq!(EtfTrendRule::default().id(), "etf_trend_faber");
        assert_eq!(EtfTrendRule::default().impl_version(), ADAPTER_VERSION);
        // A pre-R41 non-default instance keeps its six-key map: no `weight_policy` key unless the policy is non-default.
        let lagged = EtfTrendRule::new(ETF_SYMBOLS.to_vec(), ETF_SMA_MONTH_ENDS, 3);
        assert_eq!(lagged.weight_policy(), WeightPolicy::default());
        assert_eq!(lagged.declared_parameters().len(), 6);
        assert!(!lagged.declared_parameters().contains_key("weight_policy"));
    }

    #[test]
    fn a_non_default_policy_is_not_the_published_point_and_declares_itself() {
        // EqualOverN on the canonical five is the same weight, but it is not the certified instance: it declares the
        // policy, so it can never bind as the library entry.
        let five = EtfTrendRule::with_weight_policy(ETF_SYMBOLS.to_vec(), 10, 0, WeightPolicy::EqualOverN);
        assert!(!five.is_published());
        assert_ne!(five, EtfTrendRule::default());
        assert_eq!(five.weight_per_instrument().to_bits(), 0.20_f64.to_bits());
        let p = five.declared_parameters();
        assert_eq!(p.len(), 7);
        assert_eq!(p["weight_per_instrument"], "0.2");
        assert_eq!(p["weight_policy"], "\"equal_1_over_N\"");
        assert_eq!(p["execution_lag_bars"], "0");
        assert!(p.contains_key("instruments"));

        let nine = EtfTrendRule::with_weight_policy(NAMES[..9].to_vec(), 10, 0, WeightPolicy::EqualOverN);
        assert!(!nine.is_published());
        assert_eq!(nine.id(), "etf_trend_faber");
        assert_eq!(nine.universe().len(), 9);
        assert_eq!(nine.weight_per_instrument(), 1.0 / 9.0);
        let p = nine.declared_parameters();
        assert_eq!(p["weight_per_instrument"], (1.0_f64 / 9.0).to_string());
        assert_eq!(p["weight_policy"], "\"equal_1_over_N\"");

        // A fixed constant other than the published one is a non-default policy too, and says so.
        let heavy = EtfTrendRule::with_weight_policy(ETF_SYMBOLS.to_vec(), 10, 0, WeightPolicy::FixedConstant(0.25));
        assert!(!heavy.is_published());
        assert_eq!(heavy.declared_parameters()["weight_per_instrument"], "0.25");
        assert_eq!(heavy.declared_parameters()["weight_policy"], "\"fixed_constant\"");
    }

    #[test]
    #[should_panic(expected = "needs a universe of at least 5")]
    fn equal_over_n_rejects_a_universe_smaller_than_five() {
        let _ = EtfTrendRule::with_weight_policy(NAMES[..4].to_vec(), 10, 0, WeightPolicy::EqualOverN);
    }

    #[test]
    fn equal_over_n_accepts_exactly_five_and_up() {
        assert_eq!(ETF_EQUAL_OVER_N_MIN_UNIVERSE, 5);
        for n in 5..=11 {
            let r = EtfTrendRule::with_weight_policy(NAMES[..n].to_vec(), 10, 0, WeightPolicy::EqualOverN);
            assert!(r.weight_per_instrument() <= ETF_WEIGHT_PER_INSTRUMENT, "N = {n}");
        }
    }

    #[test]
    #[should_panic(expected = "fixed weight must be in (0, 1]")]
    fn a_fixed_constant_outside_zero_one_is_refused() {
        let _ = EtfTrendRule::with_weight_policy(ETF_SYMBOLS.to_vec(), 10, 0, WeightPolicy::FixedConstant(1.5));
    }

    /// The N-generality attestation (R41-A(3)(b)) for the trend rule: on seeded synthetic panels of 2000 weekdays
    /// (about 92 month-ends) with ties, never-long and always-long assets and warm-up months, the compiled rule
    /// under `EqualOverN` returns, at every month-end, bit-for-bit the weights of an independent integer reference
    /// that holds `1/N` per asset whose close is strictly above the average of its last 10 month-end closes.
    #[test]
    fn equal_over_n_matches_an_independent_reference_at_n_5_9_and_11() {
        let cal = weekday_calendar(2000);
        let dates: Vec<Date> = cal.iter().map(|&d| from_naive(d)).collect();
        let (mut longs, mut cashes, mut ties) = (0usize, 0usize, 0usize);
        let (mut warmups, mut decisions) = (0usize, 0usize);
        for (n, seed) in [(5usize, 11u64), (9, 22), (11, 33)] {
            let ticks = synthetic_ticks(n, cal.len(), seed);
            let rule = EtfTrendRule::with_weight_policy(NAMES[..n].to_vec(), 10, 0, WeightPolicy::EqualOverN);
            let independent_weight = 1.0_f64 / n as f64;
            assert_eq!(rule.weight_per_instrument().to_bits(), independent_weight.to_bits());
            let closes: Vec<Vec<f64>> = ticks.iter().map(|c| ticks_to_closes(c)).collect();
            for t in 0..cal.len() - 1 {
                if cal[t].month() == cal[t + 1].month() {
                    continue;
                }
                decisions += 1;
                let cols: Vec<&[f64]> = closes.iter().map(|c| &c[..=t]).collect();
                let got = etf_weights(&NAMES[..n], 10, rule.weight_per_instrument(), &dates[..=t], &cols);
                match reference_trend_signs(&cal, &ticks, t, 10) {
                    None => {
                        warmups += 1;
                        let refusal = got.expect_err("fewer than 10 month-ends must refuse");
                        assert_eq!(refusal.kind, RefusalKind::Warmup, "N = {n}, bar {t}");
                    }
                    Some(signs) => {
                        let got = got.unwrap_or_else(|r| panic!("N = {n}, bar {t}: unexpected refusal {r:?}"));
                        assert_eq!(got.len(), n);
                        for (a, (w, sign)) in got.iter().zip(&signs).enumerate() {
                            let want = if sign.is_gt() { independent_weight } else { 0.0 };
                            assert_eq!(w.to_bits(), want.to_bits(), "N = {n}, bar {t}, asset {a}: {w} vs {want}");
                            match sign {
                                std::cmp::Ordering::Greater => longs += 1,
                                std::cmp::Ordering::Equal => ties += 1,
                                std::cmp::Ordering::Less => cashes += 1,
                            }
                        }
                        let gross: f64 = got.iter().sum();
                        assert!(gross <= 1.0 + 1e-12, "N = {n}, bar {t}: gross {gross} exceeds 100%");
                    }
                }
            }
        }
        // The panels exercise what the attestation claims to: both signals, exact ties and the warm-up refusals.
        assert!(decisions > 200, "several hundred decisions, got {decisions}");
        assert!(longs > 100 && cashes > 100, "longs {longs}, cashes {cashes}");
        assert!(ties > 20, "exact close == average ties, got {ties}");
        assert!(warmups >= 3 * 9, "warm-up refusals, got {warmups}");
    }

    /// N = 5 identity: on the canonical five names, `EqualOverN` and the published `FixedConstant(0.20)` return the
    /// same weights bit-for-bit on every month-end of a long synthetic panel.
    #[test]
    fn equal_over_n_at_five_is_bit_identical_to_the_published_constant() {
        let cal = weekday_calendar(2000);
        let dates: Vec<Date> = cal.iter().map(|&d| from_naive(d)).collect();
        let ticks = synthetic_ticks(5, cal.len(), 77);
        let closes: Vec<Vec<f64>> = ticks.iter().map(|c| ticks_to_closes(c)).collect();
        let published = EtfTrendRule::new(NAMES[..5].to_vec(), 10, 0);
        let equal = EtfTrendRule::with_weight_policy(NAMES[..5].to_vec(), 10, 0, WeightPolicy::EqualOverN);
        let mut compared = 0;
        for t in 0..cal.len() - 1 {
            if cal[t].month() == cal[t + 1].month() {
                continue;
            }
            let cols: Vec<&[f64]> = closes.iter().map(|c| &c[..=t]).collect();
            let a = etf_weights(&NAMES[..5], 10, published.weight_per_instrument(), &dates[..=t], &cols);
            let b = etf_weights(&NAMES[..5], 10, equal.weight_per_instrument(), &dates[..=t], &cols);
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
                    assert_eq!(bits(&a), bits(&b), "bar {t}");
                    compared += 1;
                }
                (Err(a), Err(b)) => assert_eq!((a.kind, a.code), (b.kind, b.code)),
                _ => panic!("bar {t}: one policy decided and the other refused"),
            }
        }
        assert!(compared > 80);
    }

    #[test]
    fn a_non_finite_or_non_positive_close_is_refused_not_decided() {
        let cal = weekday_calendar(260);
        let dates: Vec<Date> = cal.iter().map(|&d| from_naive(d)).collect();
        let ticks = synthetic_ticks(9, cal.len(), 5);
        let t = (0..cal.len() - 1).rev().find(|&t| cal[t].month() != cal[t + 1].month()).unwrap();
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let mut closes: Vec<Vec<f64>> = ticks.iter().map(|c| ticks_to_closes(c)).collect();
            closes[3][t - 5] = bad;
            let cols: Vec<&[f64]> = closes.iter().map(|c| &c[..=t]).collect();
            let got = etf_weights(&NAMES[..9], 10, 1.0 / 9.0, &dates[..=t], &cols);
            let err = got.expect_err("a bad close is refused");
            assert_eq!(err.code, "invalid_price", "{bad}");
        }
    }
}
