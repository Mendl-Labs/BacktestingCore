//! Verification harnesses for BOOKS (design 6.5, PF1 "causality" column): poisoning of ALL instruments after a bar,
//! truncation, determinism. Library code, so the Engine self-test (PF3b) can call the very same functions.
//!
//! Like the single-rule harnesses in [`crate::harness`], every check takes a book FACTORY `Fn(&BookPanel) -> Book`
//! (not a book): a rule that (illegitimately) captures the panel it was built from and reads the future through it is
//! caught, because the factory is called on the poisoned or truncated panel and the outputs are compared.
//!
//! What is compared is EVERYTHING the run reports through the compared bar, bit for bit: returns, equity, cash, costs,
//! weights, units, marks, decisions, cadence flags, shares (the allocator's output, hence its inputs), the shadow
//! curves, contributions, the overlay's risk scale and halt flag, and the refusals.

use crate::bartime::BarTime;
use crate::book::{AllocatorSpec, Book, BookConfig, BookError};
use crate::book_panel::BookPanel;
use crate::book_result::BookResult;
use crate::book_sim::simulate_book;

/// One field of one account bar that differs bit-for-bit between two runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookMismatch {
    pub field: &'static str,
    pub bar: usize,
}

/// Outcome of a book causality check: clean means nothing at or before `compared_through_clock_bar` differed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookCausalityReport {
    /// Union-clock index of the last compared bar.
    pub compared_through_clock_bar: usize,
    pub mismatches: Vec<BookMismatch>,
}

impl BookCausalityReport {
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

/// Garbage close for instrument `i` at clock bar `t`: the original times a per-instrument level shift in `[0.5, 2)`
/// (a persistent jump at the cut) times bar noise in `[0.9, 1.1)`, plus 1e-3.
///
/// Milder than the single-rule harness's iid 0.01x-100x on purpose: a book may be levered, or drift into leverage between
/// rebalances (an `OnDecision` sleeve next to an `EveryBar` one), and a poison so violent that the POISONED run is ruined
/// (`NonPositiveEquity`, which `check_book_poisoning` reports as an error, not a pass) would make the check
/// inconclusive. A leak that reads the future still sees different numbers and the comparison is bit for bit.
fn poison_price(seed: u64, i: usize, t: usize, old: f64) -> f64 {
    let unit = |h: u64| (h >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
    let level = 0.5 + 1.5 * unit(splitmix64(seed ^ splitmix64(i as u64)));
    let noise = 0.9 + 0.2 * unit(splitmix64(seed ^ splitmix64((i as u64) << 32 | t as u64)));
    old * level * noise + 1e-3
}

fn beq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Bit-for-bit comparison of every column of the account bars whose union-clock index is `<= through_clock_bar`.
pub fn compare_book_prefix(a: &BookResult, b: &BookResult, through_clock_bar: usize) -> Vec<BookMismatch> {
    let mut out = Vec::new();
    let n = a.n_instruments();
    let s_n = a.n_sleeves();
    let nb = a.n_bars().min(b.n_bars());
    for k in 0..nb {
        if a.clock_index[k] > through_clock_bar || b.clock_index[k] > through_clock_bar {
            break;
        }
        if a.times[k] != b.times[k] {
            out.push(BookMismatch { field: "times", bar: k });
        }
        let scalars: [(&'static str, &Vec<f64>, &Vec<f64>); 11] = [
            ("ret", &a.ret, &b.ret),
            ("ret_pre_cost", &a.ret_pre_cost, &b.ret_pre_cost),
            ("equity", &a.equity, &b.equity),
            ("equity_pre", &a.equity_pre, &b.equity_pre),
            ("cash", &a.cash, &b.cash),
            ("cost", &a.cost, &b.cost),
            ("traded_notional", &a.traded_notional, &b.traded_notional),
            ("financing", &a.financing, &b.financing),
            ("gross_exposure", &a.gross_exposure, &b.gross_exposure),
            ("net_exposure", &a.net_exposure, &b.net_exposure),
            ("risk_scale", &a.risk_scale, &b.risk_scale),
        ];
        for (name, x, y) in scalars {
            if !beq(x[k], y[k]) {
                out.push(BookMismatch { field: name, bar: k });
            }
        }
        let bools: [(&'static str, &Vec<bool>, &Vec<bool>); 3] = [
            ("run", &a.run, &b.run),
            ("book_refused", &a.book_refused, &b.book_refused),
            ("halted", &a.halted, &b.halted),
        ];
        for (name, x, y) in bools {
            if x[k] != y[k] {
                out.push(BookMismatch { field: name, bar: k });
            }
        }
        let sleeve_f: [(&'static str, &Vec<f64>, &Vec<f64>); 4] = [
            ("share", &a.share, &b.share),
            ("contrib", &a.contrib, &b.contrib),
            ("shadow_ret_gross", &a.shadow_ret_gross, &b.shadow_ret_gross),
            ("shadow_ret_cost", &a.shadow_ret_cost, &b.shadow_ret_cost),
        ];
        for (name, x, y) in sleeve_f {
            if (0..s_n).any(|s| !beq(x[k * s_n + s], y[k * s_n + s])) {
                out.push(BookMismatch { field: name, bar: k });
            }
        }
        let sleeve_b: [(&'static str, &Vec<bool>, &Vec<bool>); 5] = [
            ("decision", &a.decision, &b.decision),
            ("rule_refused", &a.rule_refused, &b.rule_refused),
            ("sleeve_open", &a.sleeve_open, &b.sleeve_open),
            ("due", &a.due, &b.due),
            ("planned", &a.planned, &b.planned),
        ];
        for (name, x, y) in sleeve_b {
            if (0..s_n).any(|s| x[k * s_n + s] != y[k * s_n + s]) {
                out.push(BookMismatch { field: name, bar: k });
            }
        }
        let inst: [(&'static str, &Vec<f64>, &Vec<f64>); 5] = [
            ("marks", &a.marks, &b.marks),
            ("units", &a.units, &b.units),
            ("target_weights", &a.target_weights, &b.target_weights),
            ("held_weights", &a.held_weights, &b.held_weights),
            ("traded_by_instrument", &a.traded_by_instrument, &b.traded_by_instrument),
        ];
        for (name, x, y) in inst {
            if (0..n).any(|j| !beq(x[k * n + j], y[k * n + j])) {
                out.push(BookMismatch { field: name, bar: k });
            }
        }
    }
    let last_k = a.clock_index.iter().rposition(|&c| c <= through_clock_bar).unwrap_or(0);
    let ra: Vec<_> = a.refusals.iter().filter(|r| r.bar <= last_k).map(|r| (r.bar, r.sleeve, r.code)).collect();
    let rb: Vec<_> = b.refusals.iter().filter(|r| r.bar <= last_k).map(|r| (r.bar, r.sleeve, r.code)).collect();
    if ra != rb {
        out.push(BookMismatch { field: "refusals", bar: last_k });
    }
    out
}

/// Poisoning test for books (design 6.5): run on `panel` and on a copy whose prices after the union-clock bar
/// `last_kept_bar` are garbage for EVERY instrument (times and the availability pattern unchanged). Everything
/// through `last_kept_bar` must be bit-identical, shadow curves, allocator shares, overlay state included.
pub fn check_book_poisoning<F>(
    make_book: &F,
    panel: &BookPanel,
    cfg: &BookConfig,
    last_kept_bar: usize,
    seed: u64,
) -> Result<BookCausalityReport, BookError>
where
    F: Fn(&BookPanel) -> Book,
{
    assert!(last_kept_bar + 1 < panel.n_bars(), "nothing left to poison");
    let poisoned = poison_book_panel(panel, last_kept_bar + 1, seed);
    let clean = simulate_book(panel, &make_book(panel), cfg)?;
    let dirty = simulate_book(&poisoned, &make_book(&poisoned), cfg)?;
    Ok(BookCausalityReport {
        compared_through_clock_bar: last_kept_bar,
        mismatches: compare_book_prefix(&clean, &dirty, last_kept_bar),
    })
}

/// Garbage prices (deterministic, finite, positive; see `poison_price`) for every present close at union-clock index `>= first_poisoned_bar`.
pub fn poison_book_panel(panel: &BookPanel, first_poisoned_bar: usize, seed: u64) -> BookPanel {
    panel.with_prices_replaced_from(first_poisoned_bar, |i, t, old| poison_price(seed, i, t, old))
}

/// Truncation test for books: simulate the full panel and the panel cut after union-clock bar `cut_bar`; everything
/// through `cut_bar` must be bit-identical, except the one row that is legitimately different: the last own bar of a
/// truncated sleeve calendar (and the last clock bar for the allocator review) counts as a month-end (C7), which the
/// full panel need not agree with. Those rows are excluded, exactly as [`crate::harness::check_truncation`] does.
pub fn check_book_truncation<F>(
    make_book: &F,
    panel: &BookPanel,
    cfg: &BookConfig,
    cut_bar: usize,
) -> Result<BookCausalityReport, BookError>
where
    F: Fn(&BookPanel) -> Book,
{
    let cut = panel.truncated(cut_bar + 1);
    let full_book = make_book(panel);
    let full = simulate_book(panel, &full_book, cfg)?;
    let short = simulate_book(&cut, &make_book(&cut), cfg)?;
    let mut through = cut_bar;
    for sp in &full_book.sleeves {
        let cal_full = panel.sleeve_calendar(&sp.universe)?;
        let cal_cut = match cut.sleeve_calendar(&sp.universe) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let t = cal_cut.panel.n_bars() - 1;
        let sched = sp.rule.decision_schedule();
        let full_dec = sched.is_decision_bar(cal_full.panel.dates(), t);
        let cut_dec = sched.is_decision_bar(cal_cut.panel.dates(), t);
        if full_dec != cut_dec {
            through = through.min(cal_cut.union_index[t].saturating_sub(1));
        }
    }
    if matches!(full_book.allocator, AllocatorSpec::InverseVol { .. }) {
        let times: &[BarTime] = panel.times();
        let full_review = cut_bar + 1 == times.len() || !times[cut_bar + 1].date().same_month(times[cut_bar].date());
        if !full_review {
            through = through.min(cut_bar.saturating_sub(1));
        }
    }
    Ok(BookCausalityReport {
        compared_through_clock_bar: through,
        mismatches: compare_book_prefix(&full, &short, through),
    })
}

/// Determinism: `runs` executions must produce the same series digest and bit-identical columns.
pub fn check_book_determinism<F>(
    make_book: &F,
    panel: &BookPanel,
    cfg: &BookConfig,
    runs: usize,
) -> Result<Vec<BookMismatch>, BookError>
where
    F: Fn(&BookPanel) -> Book,
{
    assert!(runs >= 2);
    let first = simulate_book(panel, &make_book(panel), cfg)?;
    let last = panel.n_bars() - 1;
    let mut all = Vec::new();
    for _ in 1..runs {
        let again = simulate_book(panel, &make_book(panel), cfg)?;
        all.extend(compare_book_prefix(&first, &again, last));
        if again.series_sha256 != first.series_sha256 {
            all.push(BookMismatch { field: "series_sha256", bar: 0 });
        }
    }
    Ok(all)
}
