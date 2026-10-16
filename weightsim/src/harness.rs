//! Reusable verification harnesses (design 2.4, 6.1): causality (poisoning, truncation), determinism and the cost
//! identity. They are library code, not test code, so the Engine self-test (T3) can call the very same functions.
//!
//! Every harness takes a rule FACTORY `Fn(&Panel) -> R` rather than a rule. Ordinary rules ignore the panel they are
//! built from; a rule that (illegitimately) captures the panel it was built from and reads the future through it is
//! caught, because the harness builds it from the poisoned or truncated panel and compares the outputs.

use crate::costs::CostModel;
use crate::panel::{HistoryView, Panel};
use crate::rule::WeightRule;
use crate::sim::{simulate, SimConfig, SimError, SimResult};

/// One field of one bar that differs bit-for-bit between two runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub field: &'static str,
    pub bar: usize,
}

/// Outcome of a causality check. Clean means nothing dated `<= compared_through_bar` differed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalityReport {
    pub compared_through_bar: usize,
    pub mismatches: Vec<Mismatch>,
}

impl CausalityReport {
    pub fn is_clean(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// splitmix64: deterministic, dependency-free pseudo-randomness for garbage generation.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Replace every close at bar index `>= first_poisoned_bar` by deterministic, finite, positive garbage (between
/// 0.01x and 100x of the original, plus 1). Dates are left intact: date-only look-ahead is an input, not a leak.
pub fn poison_panel(panel: &Panel, first_poisoned_bar: usize, seed: u64) -> Panel {
    panel.with_prices_replaced_from(first_poisoned_bar, |i, t, old| {
        let h = splitmix64(seed ^ splitmix64((i as u64) << 32 | t as u64));
        let u = (h >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        old * (0.01 + u * 99.99) + 1.0
    })
}

fn bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Bit-for-bit comparison of every per-bar output dated `<= through_bar`.
pub fn compare_prefix(a: &SimResult, b: &SimResult, through_bar: usize) -> Vec<Mismatch> {
    let mut out = Vec::new();
    let k = a.n_assets();
    let last = through_bar.min(a.n_bars() - 1).min(b.n_bars() - 1);
    for t in 0..=last {
        let scalar_cols: [(&'static str, &Vec<f64>, &Vec<f64>); 9] = [
            ("ret", &a.ret, &b.ret),
            ("ret_pre_cost", &a.ret_pre_cost, &b.ret_pre_cost),
            ("equity", &a.equity, &b.equity),
            ("cash", &a.cash, &b.cash),
            ("cost", &a.cost, &b.cost),
            ("traded_notional", &a.traded_notional, &b.traded_notional),
            ("financing", &a.financing, &b.financing),
            ("gross_exposure", &a.gross_exposure, &b.gross_exposure),
            ("net_exposure", &a.net_exposure, &b.net_exposure),
        ];
        for (name, x, y) in scalar_cols {
            if !bits_eq(x[t], y[t]) {
                out.push(Mismatch { field: name, bar: t });
            }
        }
        for (name, x, y) in [
            ("target_weights", &a.target_weights, &b.target_weights),
            ("held_weights", &a.held_weights, &b.held_weights),
            ("units", &a.units, &b.units),
        ] {
            if (0..k).any(|i| !bits_eq(x[t * k + i], y[t * k + i])) {
                out.push(Mismatch { field: name, bar: t });
            }
        }
        if a.decision[t] != b.decision[t] {
            out.push(Mismatch { field: "decision", bar: t });
        }
        if a.refused[t] != b.refused[t] {
            out.push(Mismatch { field: "refused", bar: t });
        }
    }
    let ra: Vec<_> = a.refusals.iter().filter(|r| r.bar <= last).map(|r| (r.bar, r.code)).collect();
    let rb: Vec<_> = b.refusals.iter().filter(|r| r.bar <= last).map(|r| (r.bar, r.code)).collect();
    if ra != rb {
        out.push(Mismatch { field: "refusals", bar: last });
    }
    out
}

/// Simulator poisoning test (design 2.4(2)): run on `panel` and on a copy whose prices after `last_kept_bar` are
/// garbage (dates unchanged). Every output dated `<= last_kept_bar` must be bit-identical. This catches leakage
/// anywhere in the simulator (fill prices, mark-to-market, cost, schedule), not only in a rule.
pub fn check_poisoning<R, F>(
    make_rule: &F,
    panel: &Panel,
    cfg: &SimConfig,
    last_kept_bar: usize,
    seed: u64,
) -> Result<CausalityReport, SimError>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    assert!(last_kept_bar + 1 < panel.n_bars(), "nothing left to poison");
    let poisoned = poison_panel(panel, last_kept_bar + 1, seed);
    let clean = simulate(panel, &make_rule(panel), cfg)?;
    let dirty = simulate(&poisoned, &make_rule(&poisoned), cfg)?;
    Ok(CausalityReport {
        compared_through_bar: last_kept_bar,
        mismatches: compare_prefix(&clean, &dirty, last_kept_bar),
    })
}

/// Run-level truncation test: simulate the full panel and the panel cut after `cut_bar`; everything dated
/// `<= cut_bar` must be bit-identical.
///
/// One row is legitimately different: under `LastBarOfMonth` the last bar of the truncated panel counts as a
/// month-end (the key's definition), which the full panel need not agree with. So bar `cut_bar` itself is compared
/// only when both panels agree on whether it is a decision bar (design 2.4(4)).
pub fn check_truncation<R, F>(
    make_rule: &F,
    panel: &Panel,
    cfg: &SimConfig,
    cut_bar: usize,
) -> Result<CausalityReport, SimError>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    let cut = panel.truncated(cut_bar + 1);
    let full_rule = make_rule(panel);
    let full = simulate(panel, &full_rule, cfg)?;
    let short = simulate(&cut, &make_rule(&cut), cfg)?;
    let sched = full_rule.decision_schedule();
    let agree = sched.is_decision_bar(panel.dates(), cut_bar) == sched.is_decision_bar(cut.dates(), cut_bar);
    let through = if agree || cut_bar == 0 { cut_bar } else { cut_bar - 1 };
    Ok(CausalityReport { compared_through_bar: through, mismatches: compare_prefix(&full, &short, through) })
}

/// Per-rule truncation test (design 2.4(3)): for each sampled bar `t`, the rule's answer on the full panel's view
/// `[..=t]` must equal its answer on the view of a panel that ends at `t`. Returns the bars that disagree.
pub fn check_rule_truncation<R, F>(make_rule: &F, panel: &Panel, sample_bars: &[usize]) -> Vec<usize>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    let full_rule = make_rule(panel);
    let mut bad = Vec::new();
    for &t in sample_bars {
        let cut = panel.truncated(t + 1);
        let cut_rule = make_rule(&cut);
        let a = full_rule.target_weights(&HistoryView::new(panel, t));
        let b = cut_rule.target_weights(&HistoryView::new(&cut, t));
        let same = match (&a, &b) {
            (Ok(x), Ok(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| bits_eq(*p, *q)),
            (Err(x), Err(y)) => x == y,
            _ => false,
        };
        if !same {
            bad.push(t);
        }
    }
    bad
}

/// Determinism: `runs` executions must produce the same series digest and bit-identical per-bar outputs.
pub fn check_determinism<R, F>(
    make_rule: &F,
    panel: &Panel,
    cfg: &SimConfig,
    runs: usize,
) -> Result<Vec<Mismatch>, SimError>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    assert!(runs >= 2);
    let first = simulate(panel, &make_rule(panel), cfg)?;
    let mut all = Vec::new();
    for _ in 1..runs {
        let again = simulate(panel, &make_rule(panel), cfg)?;
        all.extend(compare_prefix(&first, &again, first.n_bars() - 1));
        if again.series_sha256 != first.series_sha256 {
            all.push(Mismatch { field: "series_sha256", bar: 0 });
        }
    }
    Ok(all)
}

/// Numbers behind the cost identity (design 2.5, Layer C).
#[derive(Clone, Debug, PartialEq)]
pub struct CostIdentityReport {
    pub rate: f64,
    /// Largest `|cost_t - rate * traded_t|` over the net run (exactly 0 by construction; the test asserts it).
    pub max_bar_cost_error: f64,
    pub total_cost: f64,
    pub rate_times_total_traded: f64,
    /// `1 - net_final_equity / gross_final_equity`.
    pub actual_drag: f64,
    /// First-order prediction `rate * sum_t (traded_t / equity_pre_t)` (turnover x cost).
    pub predicted_drag: f64,
    /// `|actual - predicted| / predicted` (NaN when nothing was traded).
    pub relative_gap: f64,
}

/// Compare a gross run (zero cost) with a net run of the same decisions.
pub fn check_cost_identity(gross: &SimResult, net: &SimResult, cost: &CostModel) -> CostIdentityReport {
    let rate = cost.rate();
    let mut max_err = 0.0f64;
    let mut pred = 0.0;
    for t in 0..net.n_bars() {
        let e = (net.cost[t] - rate * net.traded_notional[t]).abs();
        if e > max_err {
            max_err = e;
        }
        if net.traded_notional[t] > 0.0 {
            pred += rate * (net.traded_notional[t] / (net.equity[t] + net.cost[t]));
        }
    }
    let actual = 1.0 - net.equity[net.n_bars() - 1] / gross.equity[gross.n_bars() - 1];
    CostIdentityReport {
        rate,
        max_bar_cost_error: max_err,
        total_cost: net.total_cost(),
        rate_times_total_traded: rate * net.total_traded_notional(),
        actual_drag: actual,
        predicted_drag: pred,
        relative_gap: if pred > 0.0 { (actual - pred).abs() / pred } else { f64::NAN },
    }
}
