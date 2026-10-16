//! Tier IV: the named mutant list of Amendment 11 section 2, re-implemented as wrapper rules and configuration
//! changes around the real adapters. Each one is a deliberately WRONG implementation of a library rule; the
//! certification must fail every one of them (the acceptance test has to be able to fail).
//!
//! | mutant | how it is built here |
//! |---|---|
//! | `s3_same_day_peek` | wrapper rule that returns the crypto decision of the NEXT bar (peeks at its own copy of the panel), so the position earns the bar whose close it was decided on |
//! | `s3_extra_1day_delay` | configuration: `execution_delay_bars = 1` |
//! | `s3_sma_excludes_today` | independent rule: close > mean of the 100 PREVIOUS closes |
//! | `s3_half_sizing` | wrapper rule: the crypto weights times 0.5 (25% per coin) |
//! | `s3_drifting_subaccounts` | pure function: two independent 50% sub-accounts that are never rebalanced back to 50/50, fed by the base run's own signals (what the old portfolio path does) |
//! | `s1_sma_excludes_current` | independent rule: month-end close > mean of the 10 month-end closes BEFORE the current one |
//! | `s1_one_bar_late` | configuration: `execution_delay_bars = 1` |
//! | `s1_daily_rebalanced_20` | wrapper rule whose `rebalance_policy()` is `EveryBar` (weights restored every bar instead of drifting) |
//!
//! The five mutants of the momentum evidence set (Amendment 14 section 6, T4; sleeve `S4`, rule
//! `momentum_rank_weighted`) are listed in [`Mutant::MOMENTUM`]:
//!
//! | mutant | how it is built here |
//! |---|---|
//! | `momentum_lookahead_peek` | wrapper rule that ranks on the NEXT bar's closes (its own copy of the panel), so the weights earn the bar whose close they were ranked on |
//! | `momentum_extra_1day_delay` | configuration: `execution_delay_bars = 1` |
//! | `momentum_lookback_off_by_one` | the real rule with `L = 125` (`close_(t-125)` instead of `close_(t-126)`) |
//! | `momentum_half_sizing` | wrapper rule: the rank weights times 0.5 (50% cash) |
//! | `momentum_equal_weight_not_rank_weight` | the real ranking's eligibility with `1/K` for every eligible instrument instead of `(K+1-r)/(K(K+1)/2)` |

use weightsim::{
    DecisionSchedule, HistoryView, Panel, RebalancePolicy, RefusalKind, RuleRefusal, SimResult, WeightRule,
};

use reference_rules::{CRYPTO_SMA_DAYS, CRYPTO_SYMBOLS, ETF_SYMBOLS, MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK};

use super::checks::SeriesRows;
use super::fixtures::{Fixtures, LadderError, SleeveKey};
use super::registry::FixtureSetId;
use super::runner::{rows_from_sim, run_gross};
use crate::adapters::{crypto_weights, CryptoTrendRule, EtfTrendRule, MomentumRankWeightedRule};

/// The eight mutants, in the order of the amendment's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    S3SameDayPeek,
    S3ExtraDelay,
    S3SmaExcludesToday,
    S3HalfSizing,
    S3DriftingSubaccounts,
    S1SmaExcludesCurrent,
    S1OneBarLate,
    S1WrongRebalanceMode,
    // The momentum evidence set (Amendment 14 section 6, T4), in the amendment's order.
    S4LookaheadPeek,
    S4ExtraDelay,
    S4LookbackOffByOne,
    S4HalfSizing,
    S4EqualWeight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantSleeve {
    S1,
    S3,
    /// `momentum_rank_weighted` in the `MomentumEvidence` fixture set.
    S4,
}

impl MutantSleeve {
    /// The sleeve code the registry and the fixtures address this sleeve by.
    pub fn code(self) -> &'static str {
        match self {
            MutantSleeve::S1 => "S1",
            MutantSleeve::S3 => "S3",
            MutantSleeve::S4 => "S4",
        }
    }

    pub fn from_code(code: &str) -> Option<MutantSleeve> {
        match code {
            "S1" => Some(MutantSleeve::S1),
            "S3" => Some(MutantSleeve::S3),
            "S4" => Some(MutantSleeve::S4),
            _ => None,
        }
    }
}

impl Mutant {
    pub const ALL: [Mutant; 8] = [
        Mutant::S3SameDayPeek,
        Mutant::S3ExtraDelay,
        Mutant::S3SmaExcludesToday,
        Mutant::S3HalfSizing,
        Mutant::S3DriftingSubaccounts,
        Mutant::S1SmaExcludesCurrent,
        Mutant::S1OneBarLate,
        Mutant::S1WrongRebalanceMode,
    ];

    /// The five mutants of the momentum evidence set (sleeve `S4`), in the order of Amendment 14 section 6 (T4).
    pub const MOMENTUM: [Mutant; 5] = [
        Mutant::S4LookaheadPeek,
        Mutant::S4ExtraDelay,
        Mutant::S4LookbackOffByOne,
        Mutant::S4HalfSizing,
        Mutant::S4EqualWeight,
    ];

    /// The mutants of one fixture set, in table order: what its ladder runs in Tier IV. [`Mutant::ALL`] is the
    /// legacy set's eight, unchanged.
    pub fn of_set(set: FixtureSetId) -> &'static [Mutant] {
        match set {
            FixtureSetId::Legacy => &Mutant::ALL,
            FixtureSetId::MomentumEvidence => &Mutant::MOMENTUM,
        }
    }

    /// The name used in `mutants.json`.
    pub fn name(self) -> &'static str {
        match self {
            Mutant::S3SameDayPeek => "s3_same_day_peek",
            Mutant::S3ExtraDelay => "s3_extra_1day_delay",
            Mutant::S3SmaExcludesToday => "s3_sma_excludes_today",
            Mutant::S3HalfSizing => "s3_half_sizing",
            Mutant::S3DriftingSubaccounts => "s3_drifting_subaccounts",
            Mutant::S1SmaExcludesCurrent => "s1_sma_excludes_current",
            Mutant::S1OneBarLate => "s1_one_bar_late",
            Mutant::S1WrongRebalanceMode => "s1_daily_rebalanced_20",
            Mutant::S4LookaheadPeek => "momentum_lookahead_peek",
            Mutant::S4ExtraDelay => "momentum_extra_1day_delay",
            Mutant::S4LookbackOffByOne => "momentum_lookback_off_by_one",
            Mutant::S4HalfSizing => "momentum_half_sizing",
            Mutant::S4EqualWeight => "momentum_equal_weight_not_rank_weight",
        }
    }

    pub fn sleeve(self) -> MutantSleeve {
        match self {
            Mutant::S1SmaExcludesCurrent | Mutant::S1OneBarLate | Mutant::S1WrongRebalanceMode => MutantSleeve::S1,
            Mutant::S4LookaheadPeek
            | Mutant::S4ExtraDelay
            | Mutant::S4LookbackOffByOne
            | Mutant::S4HalfSizing
            | Mutant::S4EqualWeight => MutantSleeve::S4,
            _ => MutantSleeve::S3,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Mutant::S3SameDayPeek => "position earns the bar whose close it was decided on",
            Mutant::S3ExtraDelay => "one extra day between decision and fill",
            Mutant::S3SmaExcludesToday => "SMA over the 100 previous closes, excluding today",
            Mutant::S3HalfSizing => "25% per coin instead of 50%",
            Mutant::S3DriftingSubaccounts => "independent 50% sub-accounts that drift (old portfolio path)",
            Mutant::S1SmaExcludesCurrent => "SMA over the 10 month-ends before the current one",
            Mutant::S1OneBarLate => "month-end decision executed one bar late",
            Mutant::S1WrongRebalanceMode => "weights restored to 20% every bar instead of drifting",
            Mutant::S4LookaheadPeek => {
                "ranks on the next bar's closes, so the weights earn the bar they were ranked on"
            }
            Mutant::S4ExtraDelay => "one extra day between decision and fill",
            Mutant::S4LookbackOffByOne => "trailing return over 125 bars instead of 126",
            Mutant::S4HalfSizing => "rank weights halved, 50% cash",
            Mutant::S4EqualWeight => "1/K for every ranked instrument instead of the linear rank weights",
        }
    }
}

// --------------------------------------------------------------------------------------------- mutant rules

/// Wraps a real rule; optionally scales its weights and/or overrides the rebalance policy.
struct Wrapped<R: WeightRule> {
    inner: R,
    name: &'static str,
    scale: f64,
    policy: Option<RebalancePolicy>,
}

impl<R: WeightRule> WeightRule for Wrapped<R> {
    fn id(&self) -> &'static str {
        self.name
    }
    fn impl_version(&self) -> String {
        format!("mutant of {}", self.inner.impl_version())
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.inner.decision_schedule()
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.policy.unwrap_or_else(|| self.inner.rebalance_policy())
    }
    fn execution_lag_bars(&self) -> usize {
        self.inner.execution_lag_bars()
    }
    fn min_history_bars(&self) -> usize {
        self.inner.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        Ok(self.inner.target_weights(h)?.into_iter().map(|w| w * self.scale).collect())
    }
}

/// Crypto rule that peeks: at the close of bar `t` it returns the decision the real rule makes at the close of `t+1`.
/// It holds its own copy of the panel, which is exactly what a leaky rule looks like (the poisoning harness must
/// catch it: see `tests/ladder_synthetic.rs`).
pub struct PeekCrypto {
    full: Panel,
}

impl PeekCrypto {
    pub fn new(panel: &Panel) -> PeekCrypto {
        PeekCrypto { full: panel.clone() }
    }
}

impl WeightRule for PeekCrypto {
    fn id(&self) -> &'static str {
        "mutant_s3_same_day_peek"
    }
    fn impl_version(&self) -> String {
        "mutant".into()
    }
    fn universe(&self) -> &[&'static str] {
        &CRYPTO_SYMBOLS
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        CryptoTrendRule::default().min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let end = (h.len() + 1).min(self.full.n_bars());
        let cols: Vec<&[f64]> = (0..self.full.n_assets()).map(|i| &self.full.closes(i)[..end]).collect();
        crypto_weights(&CRYPTO_SYMBOLS, CRYPTO_SMA_DAYS, &self.full.dates()[..end], &cols)
    }
}

/// Crypto: close > mean of the 100 previous closes (today excluded), 50% per coin.
struct CryptoSmaExcludesToday;

impl WeightRule for CryptoSmaExcludesToday {
    fn id(&self) -> &'static str {
        "mutant_s3_sma_excludes_today"
    }
    fn impl_version(&self) -> String {
        "mutant".into()
    }
    fn universe(&self) -> &[&'static str] {
        &CRYPTO_SYMBOLS
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        101
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let mut w = Vec::with_capacity(h.n_assets());
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let n = c.len();
            let mut s = 0.0;
            for &v in &c[n - 101..n - 1] {
                s += v;
            }
            w.push(if c[n - 1] > s / 100.0 { 0.5 } else { 0.0 });
        }
        Ok(w)
    }
}

/// ETF: month-end close > mean of the 10 month-end closes BEFORE the current one, 20% per ETF.
struct EtfSmaExcludesCurrent;

impl WeightRule for EtfSmaExcludesCurrent {
    fn id(&self) -> &'static str {
        "mutant_s1_sma_excludes_current"
    }
    fn impl_version(&self) -> String {
        "mutant".into()
    }
    fn universe(&self) -> &[&'static str] {
        &ETF_SYMBOLS
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::LastBarOfMonth
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let dates = h.dates();
        let mut me: Vec<usize> = Vec::new();
        for i in 0..dates.len() {
            if i + 1 == dates.len() || !dates[i].same_month(dates[i + 1]) {
                me.push(i);
            }
        }
        if me.len() < 11 {
            return Err(RuleRefusal::new(RefusalKind::Warmup, "insufficient_history", "need 11 month-end closes"));
        }
        let prev = &me[me.len() - 11..me.len() - 1];
        let mut w = Vec::with_capacity(h.n_assets());
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &i in prev {
                s += c[i];
            }
            w.push(if c[c.len() - 1] > s / 10.0 { 0.2 } else { 0.0 });
        }
        Ok(w)
    }
}

/// Momentum weights from raw close slices (each ending at the decision bar): rank the instruments that have
/// `lookback + 1` closes by trailing return, descending, ties by ascending ticker; with `K` ranked, rank `r` gets
/// `(K + 1 - r) / (K (K + 1) / 2)`, or `1 / K` when `equal` is set. A deliberately independent re-statement of the
/// reference rule's formula, used by the mutants that need the ranking on data the rule is not given.
fn momentum_weights(universe: &[&'static str], closes: &[&[f64]], lookback: usize, equal: bool) -> Vec<f64> {
    let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(universe.len());
    for (i, c) in closes.iter().enumerate() {
        if c.len() > lookback {
            ranked.push((i, c[c.len() - 1] / c[c.len() - 1 - lookback] - 1.0));
        }
    }
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(universe[a.0].cmp(universe[b.0]))
    });
    let k = ranked.len();
    let mut w = vec![0.0; universe.len()];
    for (pos, (i, _)) in ranked.iter().enumerate() {
        w[*i] = if equal { 1.0 / k as f64 } else { (k + 1 - (pos + 1)) as f64 / ((k * (k + 1)) as f64 / 2.0) };
    }
    w
}

/// Momentum rule that peeks: at the close of bar `t` it ranks on the closes through bar `t+1`. It holds its own copy of
/// the panel, which is exactly what a leaky rule looks like (the poisoning harness must catch it).
pub struct PeekMomentum {
    full: Panel,
}

impl PeekMomentum {
    pub fn new(panel: &Panel) -> PeekMomentum {
        PeekMomentum { full: panel.clone() }
    }
}

impl WeightRule for PeekMomentum {
    fn id(&self) -> &'static str {
        "mutant_momentum_lookahead_peek"
    }
    fn impl_version(&self) -> String {
        "mutant".into()
    }
    fn universe(&self) -> &[&'static str] {
        &ETF_SYMBOLS
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK + 1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let end = self.full.n_bars().min(h.len() + 1);
        let cols: Vec<&[f64]> = (0..self.full.n_assets()).map(|i| &self.full.closes(i)[..end]).collect();
        Ok(momentum_weights(&ETF_SYMBOLS, &cols, MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK, false))
    }
}

/// Momentum with the right ranking and eligibility but `1/K` for every ranked instrument.
struct EqualWeightMomentum {
    inner: MomentumRankWeightedRule,
}

impl WeightRule for EqualWeightMomentum {
    fn id(&self) -> &'static str {
        "mutant_momentum_equal_weight_not_rank_weight"
    }
    fn impl_version(&self) -> String {
        "mutant".into()
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
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
        let ranking = self.inner.ranking(h);
        let k = ranking.weights.iter().filter(|w| **w > 0.0).count();
        Ok(ranking.weights.iter().map(|w| if *w > 0.0 { 1.0 / k as f64 } else { 0.0 }).collect())
    }
}

/// Two independent sub-accounts of 0.5 each, invested in a coin (whole account) when its base-run target is on and
/// never rebalanced back to 50/50. Returns are those of the summed equity, over the sleeve's counted window.
fn drifting_subaccounts(panel: &Panel, base: &SimResult, key: &SleeveKey) -> Result<SeriesRows, LadderError> {
    let n = panel.n_bars();
    let k = panel.n_assets();
    let (start, end) = (key.bars[0].date, key.bars[key.bars.len() - 1].date);
    let mut e = vec![0.5f64; k];
    let mut total = vec![0.0f64; n];
    let mut sum0 = 0.0;
    for v in &e {
        sum0 += v;
    }
    total[0] = sum0;
    for t in 1..n {
        for a in 0..k {
            let pos = base.row(&base.target_weights, t - 1)[a] / 0.5;
            let c = panel.closes(a);
            let r = c[t] / c[t - 1] - 1.0;
            e[a] *= 1.0 + pos * r;
        }
        let mut s = 0.0;
        for v in &e {
            s += v;
        }
        total[t] = s;
    }
    let mut rows = SeriesRows::default();
    for t in 1..n {
        let d = panel.dates()[t];
        if d >= start && d <= end {
            rows.dates.push(d);
            rows.ret.push(total[t] / total[t - 1] - 1.0);
        }
    }
    if rows.dates.len() < 3 {
        return Err(LadderError::Sim("drifting sub-accounts: window too short".into()));
    }
    Ok(rows)
}

/// A mutant run: its rows in the key layout and its signal-flip count (when a simulation produced it).
pub struct MutantRun {
    pub rows: SeriesRows,
    pub flips: Option<u64>,
    pub series_sha256: Option<String>,
}

fn from_sim(sim: &SimResult, key: &SleeveKey) -> Result<MutantRun, LadderError> {
    Ok(MutantRun {
        rows: rows_from_sim(sim, key.bars[0].date, key.bars[key.bars.len() - 1].date, false)?,
        flips: Some(sim.signal_flips.iter().sum()),
        series_sha256: Some(sim.series_sha256.clone()),
    })
}

/// Run one mutant on the fixtures (gross, zero cost, the sleeve's counted window).
///
/// The mutants are NOT started flat at the entry bar (`FlatUntil` is for the certified base runs, whose net series
/// depends on the entry): they are gross replays of the pandas mutants, which trade from their own first decision.
pub fn run_mutant(fx: &Fixtures, m: Mutant) -> Result<MutantRun, LadderError> {
    // The mutant's sleeve names the panel and key it is replayed against.
    let sleeve = m.sleeve();
    let sf = fx.sleeve(sleeve.code()).ok_or_else(|| {
        LadderError::Inconsistent(format!("the fixture set has no sleeve `{}` for mutant {}", sleeve.code(), m.name()))
    })?;
    run_mutant_on(sf.panel, sf.key, m)
}

/// [`run_mutant`] on an explicit panel and key (the key supplies the counted window): the mutant replays do not need
/// a loaded fixture set, which is how the always-on tests exercise them on synthetic prices.
pub fn run_mutant_on(panel: &Panel, key: &SleeveKey, m: Mutant) -> Result<MutantRun, LadderError> {
    match m {
        Mutant::S3SameDayPeek => from_sim(&run_gross(panel, &PeekCrypto::new(panel), key, 0)?, key),
        Mutant::S3ExtraDelay => from_sim(&run_gross(panel, &CryptoTrendRule::default(), key, 1)?, key),
        Mutant::S3SmaExcludesToday => from_sim(&run_gross(panel, &CryptoSmaExcludesToday, key, 0)?, key),
        Mutant::S3HalfSizing => {
            let rule =
                Wrapped { inner: CryptoTrendRule::default(), name: "mutant_s3_half_sizing", scale: 0.5, policy: None };
            from_sim(&run_gross(panel, &rule, key, 0)?, key)
        }
        Mutant::S3DriftingSubaccounts => Ok(MutantRun {
            // The sub-accounts are fed by the unmutated rule's own signals over the WHOLE history (they hold
            // positions, and so drift, from the rule's first signal in 2015, not from the entry bar).
            rows: drifting_subaccounts(panel, &run_gross(panel, &CryptoTrendRule::default(), key, 0)?, key)?,
            flips: None,
            series_sha256: None,
        }),
        Mutant::S1SmaExcludesCurrent => from_sim(&run_gross(panel, &EtfSmaExcludesCurrent, key, 0)?, key),
        Mutant::S1OneBarLate => from_sim(&run_gross(panel, &EtfTrendRule::default(), key, 1)?, key),
        Mutant::S1WrongRebalanceMode => {
            let rule = Wrapped {
                inner: EtfTrendRule::default(),
                name: "mutant_s1_daily_rebalanced_20",
                scale: 1.0,
                policy: Some(RebalancePolicy::EveryBar),
            };
            from_sim(&run_gross(panel, &rule, key, 0)?, key)
        }
        Mutant::S4LookaheadPeek => from_sim(&run_gross(panel, &PeekMomentum::new(panel), key, 0)?, key),
        Mutant::S4ExtraDelay => from_sim(&run_gross(panel, &MomentumRankWeightedRule::default(), key, 1)?, key),
        Mutant::S4LookbackOffByOne => {
            let rule = MomentumRankWeightedRule::new(ETF_SYMBOLS.to_vec(), MOMENTUM_RANK_WEIGHTED_DEFAULT_LOOKBACK - 1);
            from_sim(&run_gross(panel, &rule, key, 0)?, key)
        }
        Mutant::S4HalfSizing => {
            let rule = Wrapped {
                inner: MomentumRankWeightedRule::default(),
                name: "mutant_momentum_half_sizing",
                scale: 0.5,
                policy: None,
            };
            from_sim(&run_gross(panel, &rule, key, 0)?, key)
        }
        Mutant::S4EqualWeight => {
            let rule = EqualWeightMomentum { inner: MomentumRankWeightedRule::default() };
            from_sim(&run_gross(panel, &rule, key, 0)?, key)
        }
    }
}
