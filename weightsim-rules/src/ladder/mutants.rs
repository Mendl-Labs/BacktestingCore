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

use weightsim::{
    DecisionSchedule, HistoryView, Panel, RebalancePolicy, RefusalKind, RuleRefusal, SimResult, WeightRule,
};

use reference_rules::{CRYPTO_SYMBOLS, ETF_SYMBOLS};

use super::checks::SeriesRows;
use super::fixtures::{Fixtures, LadderError, SleeveKey};
use super::runner::{rows_from_sim, run_gross};
use crate::adapters::{crypto_weights, CryptoTrendRule, EtfTrendRule};

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantSleeve {
    S1,
    S3,
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
        }
    }

    pub fn sleeve(self) -> MutantSleeve {
        match self {
            Mutant::S1SmaExcludesCurrent | Mutant::S1OneBarLate | Mutant::S1WrongRebalanceMode => MutantSleeve::S1,
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
        CryptoTrendRule.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let end = (h.len() + 1).min(self.full.n_bars());
        let cols: Vec<&[f64]> = (0..self.full.n_assets()).map(|i| &self.full.closes(i)[..end]).collect();
        crypto_weights(&self.full.dates()[..end], &cols)
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
    let (etf, cry) = (&fx.etf_panel, &fx.crypto_panel);
    match m {
        Mutant::S3SameDayPeek => from_sim(&run_gross(cry, &PeekCrypto::new(cry), &fx.s3, 0)?, &fx.s3),
        Mutant::S3ExtraDelay => from_sim(&run_gross(cry, &CryptoTrendRule, &fx.s3, 1)?, &fx.s3),
        Mutant::S3SmaExcludesToday => from_sim(&run_gross(cry, &CryptoSmaExcludesToday, &fx.s3, 0)?, &fx.s3),
        Mutant::S3HalfSizing => {
            let rule = Wrapped { inner: CryptoTrendRule, name: "mutant_s3_half_sizing", scale: 0.5, policy: None };
            from_sim(&run_gross(cry, &rule, &fx.s3, 0)?, &fx.s3)
        }
        Mutant::S3DriftingSubaccounts => Ok(MutantRun {
            // The sub-accounts are fed by the unmutated rule's own signals over the WHOLE history (they hold
            // positions, and so drift, from the rule's first signal in 2015, not from the entry bar).
            rows: drifting_subaccounts(cry, &run_gross(cry, &CryptoTrendRule, &fx.s3, 0)?, &fx.s3)?,
            flips: None,
            series_sha256: None,
        }),
        Mutant::S1SmaExcludesCurrent => from_sim(&run_gross(etf, &EtfSmaExcludesCurrent, &fx.s1, 0)?, &fx.s1),
        Mutant::S1OneBarLate => from_sim(&run_gross(etf, &EtfTrendRule, &fx.s1, 1)?, &fx.s1),
        Mutant::S1WrongRebalanceMode => {
            let rule = Wrapped {
                inner: EtfTrendRule,
                name: "mutant_s1_daily_rebalanced_20",
                scale: 1.0,
                policy: Some(RebalancePolicy::EveryBar),
            };
            from_sim(&run_gross(etf, &rule, &fx.s1, 0)?, &fx.s1)
        }
    }
}
