//! Running a rule through `weightsim` under the ladder's declared configuration, and turning a `SimResult` into the
//! key's per-bar layout.
//!
//! Alignment (the key's convention, Amendment 11): the row dated `t` is the return over `(t-1, t]`. `ret`, `equity`,
//! `cost` and `traded` are the simulator's values AT bar `t`; `w_target` and `w_held` are the standing target and the
//! post-trade held weights AT bar `t-1` (the position in force during bar `t`). Equity is the simulator's post-cost
//! equity path from `initial_equity = 1`, so the net run's equity includes the cost of the first fill (which the key's
//! `equity_net` includes too).

use weightsim::{simulate, simulate_gross_and_net, CostModel, Date, OnRefusal, SimConfig, SimResult, WeightRule};

use super::checks::SeriesRows;
use super::fixtures::{KeyBar, LadderError, SleeveKey};

/// The certification configuration of a sleeve: the key's own first and last bar are the counted window, `Abort` on
/// any refusal after the first successful decision, no delay, no scaling, no cap, no financing, the declared cost.
pub fn sleeve_config(key: &SleeveKey, cost: CostModel) -> SimConfig {
    SimConfig {
        start: Some(key.bars[0].date),
        end: Some(key.bars[key.bars.len() - 1].date),
        cost,
        on_refusal: OnRefusal::Abort,
        ..SimConfig::default()
    }
}

/// A gross run and a net run of the same decisions.
pub struct BaseRuns {
    pub gross: SimResult,
    pub net: SimResult,
}

/// Run `rule` on `panel` gross (zero cost) and net (`certification_flat_10bps_per_side`).
pub fn run_gross_and_net<R: WeightRule + ?Sized>(
    panel: &weightsim::Panel,
    rule: &R,
    key: &SleeveKey,
) -> Result<BaseRuns, LadderError> {
    let cfg = sleeve_config(key, CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE);
    let (gross, net) = simulate_gross_and_net(panel, rule, &cfg).map_err(|e| LadderError::Sim(e.to_string()))?;
    Ok(BaseRuns { gross, net })
}

/// One gross run (zero cost) of `rule` under the sleeve's window (used by the mutants).
pub fn run_gross<R: WeightRule + ?Sized>(
    panel: &weightsim::Panel,
    rule: &R,
    key: &SleeveKey,
    delay_bars: usize,
) -> Result<SimResult, LadderError> {
    let cfg = SimConfig { execution_delay_bars: delay_bars, ..sleeve_config(key, CostModel::ZERO) };
    simulate(panel, rule, &cfg).map_err(|e| LadderError::Sim(e.to_string()))
}

/// The bars of a run dated in `[start, end]` and after its first successful decision, as key-layout rows.
///
/// The window opens the bar AFTER the first successful decision (the key ledger's convention), not after the first
/// fill: for a run with an execution delay the bars between the two are flat rows (return 0, weights 0), exactly as
/// the pandas mutant replays count them. For an undelayed run the first fill IS that decision, so this equals the
/// simulator's own counted window (the ladder checks that).
///
/// `full` adds equity, cost and traded notional (net or gross runs compared against the key's matching columns);
/// mutants are compared on returns and weights only.
pub fn rows_from_sim(sim: &SimResult, start: Date, end: Date, full: bool) -> Result<SeriesRows, LadderError> {
    let first_decision = sim
        .decision
        .iter()
        .position(|&d| d)
        .ok_or_else(|| LadderError::Sim("the run never made a successful decision".into()))?;
    let first_bar = sim
        .dates
        .iter()
        .position(|&d| d >= start)
        .ok_or_else(|| LadderError::Sim("the run has no bar on or after the window start".into()))?
        .max(first_decision + 1);
    let last_bar = sim
        .dates
        .iter()
        .rposition(|&d| d <= end)
        .ok_or_else(|| LadderError::Sim("the run has no bar on or before the window end".into()))?;
    if first_bar > last_bar {
        return Err(LadderError::Sim("the counted window is empty".into()));
    }
    let w = weightsim::Window { first_bar, last_bar };
    let range = w.first_bar..=w.last_bar;
    let mut rows = SeriesRows {
        dates: range.clone().map(|i| sim.dates[i]).collect(),
        ret: range.clone().map(|i| sim.ret[i]).collect(),
        w_target: Some(range.clone().map(|i| sim.row(&sim.target_weights, i - 1).to_vec()).collect()),
        w_held: Some(range.clone().map(|i| sim.row(&sim.held_weights, i - 1).to_vec()).collect()),
        ..SeriesRows::default()
    };
    if full {
        // The key's `cost` and `turnover` are fractions of the PRE-cost equity at the rebalance (equity + cost).
        let pre = |i: usize| sim.equity[i] + sim.cost[i];
        rows.equity = Some(range.clone().map(|i| sim.equity[i]).collect());
        rows.cost = Some(range.clone().map(|i| sim.cost[i] / pre(i)).collect());
        rows.traded = Some(range.map(|i| sim.traded_notional[i] / pre(i)).collect());
    }
    Ok(rows)
}

/// The bar the sleeve starts trading at: the last bar of `panel` strictly before the key's first return bar.
///
/// The key's ledger is flat with equity 1.0 at the close of that bar and enters its first position there, paying the
/// entry cost (Amendment 11: "the book is flat and equity 1.0 at the close of the bar before the window"). A rule that
/// was already invested when the window opens (S3 trades from its 100th bar in 2015) would differ from the key by the
/// entry cost and by the equity accumulated before the window.
pub fn entry_date(panel: &weightsim::Panel, key: &SleeveKey) -> Result<Date, LadderError> {
    let first = key.bars[0].date;
    panel
        .dates()
        .iter()
        .rev()
        .copied()
        .find(|&d| d < first)
        .ok_or_else(|| LadderError::Inconsistent("the key's first bar is the first bar of the panel".into()))
}

/// The key's trade counter, by the key's own convention: per asset, the number of decision-to-decision changes of the
/// sign of the target between successive successful decisions, the first decision being the baseline. The baseline is
/// the last decision dated before the first counted return (the entry bar), so a decision that only establishes the
/// starting position is not a "flip" but the decision before the window opens is still the reference for the first
/// one inside it. Valid for runs without an execution delay (the standing target after bar `t` is the decision of `t`).
pub fn flips_by_key_convention(sim: &SimResult, first_counted: Date, last_counted: Date) -> u64 {
    let bars: Vec<usize> = (0..sim.n_bars()).filter(|&t| sim.decision[t]).collect();
    let base = bars.iter().rposition(|&t| sim.dates[t] < first_counted).unwrap_or(0);
    let mut flips = 0u64;
    let mut prev: Option<Vec<i8>> = None;
    for &t in &bars[base..] {
        if sim.dates[t] > last_counted {
            break;
        }
        let signs: Vec<i8> = sim
            .row(&sim.target_weights, t)
            .iter()
            .map(|&w| {
                if w > 0.0 {
                    1
                } else if w < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        if let Some(p) = &prev {
            flips += p.iter().zip(&signs).filter(|(a, b)| a != b).count() as u64;
        }
        prev = Some(signs);
    }
    flips
}

/// Which basis of the key a run is compared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    Gross,
    Net,
}

impl Basis {
    pub fn name(self) -> &'static str {
        match self {
            Basis::Gross => "gross",
            Basis::Net => "net",
        }
    }
}

/// The key in the same layout as [`rows_from_sim`]. `w_held` is the key's gross-run column, so it is offered only for
/// the gross basis (a net run's held weights are the same target plus the cost, which the key does not export).
pub fn key_rows(key: &SleeveKey, basis: Basis) -> SeriesRows {
    let pick = |f: fn(&KeyBar) -> f64| -> Vec<f64> { key.bars.iter().map(f).collect() };
    SeriesRows {
        dates: key.bars.iter().map(|b| b.date).collect(),
        ret: match basis {
            Basis::Gross => pick(|b| b.ret_gross),
            Basis::Net => pick(|b| b.ret_net),
        },
        equity: Some(match basis {
            Basis::Gross => pick(|b| b.equity_gross),
            Basis::Net => pick(|b| b.equity_net),
        }),
        cost: Some(match basis {
            Basis::Gross => vec![0.0; key.bars.len()],
            Basis::Net => pick(|b| b.cost),
        }),
        // The key's turnover is the NET run's traded notional (the gross run trades slightly different amounts).
        traded: match basis {
            Basis::Gross => None,
            Basis::Net => Some(pick(|b| b.turnover)),
        },
        w_target: Some(key.bars.iter().map(|b| b.w_target.clone()).collect()),
        w_held: match basis {
            Basis::Gross => Some(key.bars.iter().map(|b| b.w_held.clone()).collect()),
            Basis::Net => None,
        },
    }
}

/// Key rows restricted to the columns a mutant run has (returns, weights), for the mutant comparison.
pub fn key_rows_for_mutants(key: &SleeveKey) -> SeriesRows {
    let mut r = key_rows(key, Basis::Gross);
    r.equity = None;
    r.cost = None;
    r.traded = None;
    r
}

/// Dates of a series as a plain vector (helper for tests and reports).
pub fn first_last(rows: &SeriesRows) -> Option<(Date, Date)> {
    Some((*rows.dates.first()?, *rows.dates.last()?))
}
