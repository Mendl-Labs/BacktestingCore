//! Always-on: the eight mutants really ARE the mutants of Amendment 11 section 2.
//!
//! The synthetic `mutants.json` is produced by this crate itself, so it cannot vouch for the mutant implementations.
//! Here each mutant's per-bar returns and weights are recomputed in closed form (S3) or with a plain ledger (S1)
//! written independently in this file, in the style of the pandas replays of `mutant_replays.py`, and compared to what
//! `ladder::mutants::run_mutant` produced. (On the real data the same is proven against the recorded `mutants.json`
//! by `tests/ladder_real.rs`.)

// Index loops mirror the pandas replays term by term (parallel per-asset vectors, summation order).
#![allow(clippy::needless_range_loop)]

mod common;

use common::*;
use weightsim::*;
use weightsim_rules::ladder::checks::SeriesRows;
use weightsim_rules::ladder::mutants::{run_mutant, Mutant};

const TOL: f64 = 1e-12;

fn index_of(p: &Panel, d: Date) -> usize {
    p.dates().iter().position(|&x| x == d).expect("date on the panel calendar")
}

// ------------------------------------------------------------------------------------------------------ S3 closed form

/// `sig[a][t]`: close > mean of the `n` closes ending at `t` (`include_today`) or of the `n` closes before `t`.
fn sma_signal(c: &[f64], t: usize, n: usize, include_today: bool) -> f64 {
    let (lo, hi) = if include_today {
        if t + 1 < n {
            return 0.0;
        }
        (t + 1 - n, t + 1)
    } else {
        if t < n {
            return 0.0;
        }
        (t - n, t)
    };
    let mut s = 0.0;
    for &v in &c[lo..hi] {
        s += v;
    }
    if c[t] > s / n as f64 {
        1.0
    } else {
        0.0
    }
}

/// pandas-style replay: position for bar t is `sig` from `shift` bars ago; return = sum size*pos*pct_change.
fn s3_closed_form(p: &Panel, t: usize, shift: usize, size: f64, include_today: bool) -> (f64, Vec<f64>) {
    let mut r = 0.0;
    let mut w = Vec::new();
    for a in 0..p.n_assets() {
        let c = p.closes(a);
        let pos = sma_signal(c, t - shift, 100, include_today);
        w.push(size * pos);
        r += size * pos * (c[t] / c[t - 1] - 1.0);
    }
    (r, w)
}

fn check_s3(m: Mutant, shift: usize, size: f64, include_today: bool) {
    let fx = fixtures();
    let run = run_mutant(&fx, m).unwrap();
    let rows: &SeriesRows = &run.rows;
    assert!(rows.dates.len() > 150, "{}: only {} bars", m.name(), rows.dates.len());
    for (i, &d) in rows.dates.iter().enumerate() {
        let t = index_of(&fx.crypto_panel, d);
        let (r, w) = s3_closed_form(&fx.crypto_panel, t, shift, size, include_today);
        assert!((rows.ret[i] - r).abs() <= TOL, "{} {d}: ret {} vs closed form {r}", m.name(), rows.ret[i]);
        let wt = &rows.w_target.as_ref().unwrap()[i];
        let wh = &rows.w_held.as_ref().unwrap()[i];
        for a in 0..w.len() {
            assert!((wt[a] - w[a]).abs() <= TOL, "{} {d} target {a}: {} vs {}", m.name(), wt[a], w[a]);
            assert!((wh[a] - w[a]).abs() <= 1e-9, "{} {d} held {a}: {} vs {}", m.name(), wh[a], w[a]);
        }
    }
}

#[test]
fn s3_same_day_peek_earns_the_bar_it_was_decided_on() {
    // position for bar t = signal computed at the close of t itself (shift 0)
    check_s3(Mutant::S3SameDayPeek, 0, 0.5, true);
}

#[test]
fn s3_extra_delay_uses_the_signal_of_two_bars_ago() {
    check_s3(Mutant::S3ExtraDelay, 2, 0.5, true);
}

#[test]
fn s3_sma_excludes_today_averages_the_previous_hundred_closes() {
    check_s3(Mutant::S3SmaExcludesToday, 1, 0.5, false);
}

#[test]
fn s3_half_sizing_is_twenty_five_percent_per_coin() {
    check_s3(Mutant::S3HalfSizing, 1, 0.25, true);
}

#[test]
fn s3_base_replay_shift_one_is_the_certified_rule() {
    // The unmutated pandas replay (shift 1, 50%) is what the certified rule does: the closed form must equal the
    // Python key, so the closed-form helper itself is not lying.
    let fx = fixtures();
    let (kd, kr) = parse_returns(PY_KEY_S3_RETURNS);
    for (d, r) in kd.iter().zip(&kr) {
        let t = index_of(&fx.crypto_panel, *d);
        let (got, _) = s3_closed_form(&fx.crypto_panel, t, 1, 0.5, true);
        assert!((got - r).abs() <= 1e-12, "{d}: {got} vs Python key {r}");
    }
}

#[test]
fn s3_drifting_subaccounts_never_rebalance_back_to_fifty_fifty() {
    let fx = fixtures();
    let run = run_mutant(&fx, Mutant::S3DriftingSubaccounts).unwrap();
    let p = &fx.crypto_panel;
    // two accounts of 0.5; a coin's whole account is invested when its shift-1 signal is on
    let mut e = [0.5f64, 0.5];
    let mut tot = vec![e[0] + e[1]; p.n_bars()];
    for t in 1..p.n_bars() {
        for a in 0..2 {
            let c = p.closes(a);
            let pos = sma_signal(c, t - 1, 100, true);
            e[a] *= 1.0 + pos * (c[t] / c[t - 1] - 1.0);
        }
        tot[t] = e[0] + e[1];
    }
    assert!(run.rows.dates.len() > 150);
    assert!(run.rows.w_target.is_none() && run.rows.w_held.is_none(), "this mutant has no weights (Tier III is n/a)");
    for (i, &d) in run.rows.dates.iter().enumerate() {
        let t = index_of(p, d);
        let want = tot[t] / tot[t - 1] - 1.0;
        assert!((run.rows.ret[i] - want).abs() <= TOL, "{d}: {} vs {want}", run.rows.ret[i]);
    }
    // ... and it really does differ from the fixed-weight replay (otherwise it would not be a mutant).
    let mut differs = 0;
    for (i, &d) in run.rows.dates.iter().enumerate() {
        let t = index_of(p, d);
        if (run.rows.ret[i] - s3_closed_form(p, t, 1, 0.5, true).0).abs() > 1e-9 {
            differs += 1;
        }
    }
    assert!(differs > 20, "the drifting account should differ from the rebalanced one on many bars, got {differs}");
}

// -------------------------------------------------------------------------------------------------- S1 plain ledger

struct LedgerRow {
    bar: usize,
    ret: f64,
    w_target: Vec<f64>,
    w_open: Vec<f64>,
}

/// A literal Rust transcription of `export_key.py::run_ledger` (gross): flat with equity 1.0 at the close of
/// `start_i`, one row per later bar.
fn ledger(
    p: &Panel,
    decisions: &std::collections::BTreeMap<usize, Vec<f64>>,
    every_bar: bool,
    start_i: usize,
) -> Vec<LedgerRow> {
    let n = p.n_assets();
    let mut units = vec![0.0f64; n];
    let mut cash = 1.0f64;
    let mut e_prev = 1.0f64;
    let mut target: Option<Vec<f64>> = None;
    let mut out = Vec::new();
    for i in start_i..p.n_bars() {
        let (e_pre, w_open);
        if i == start_i {
            e_pre = 1.0;
            w_open = vec![0.0; n];
        } else {
            let mut mv = 0.0;
            for j in 0..n {
                mv += units[j] * p.closes(j)[i];
            }
            e_pre = cash + mv;
            w_open = (0..n).map(|j| units[j] * p.closes(j)[i - 1] / e_prev).collect();
        }
        let w_target_force = target.clone().unwrap_or_else(|| vec![0.0; n]);
        let dec = decisions.get(&i);
        if let Some(d) = dec {
            target = Some(d.clone());
        }
        let reb = target.is_some() && (if every_bar { true } else { dec.is_some() });
        if reb {
            let t = target.as_ref().unwrap();
            let new_units: Vec<f64> = (0..n).map(|j| t[j] * e_pre / p.closes(j)[i]).collect();
            units = new_units;
            let mut mv2 = 0.0;
            for j in 0..n {
                mv2 += units[j] * p.closes(j)[i];
            }
            cash = e_pre - mv2;
        }
        let e = e_pre;
        if i > start_i {
            out.push(LedgerRow { bar: i, ret: e / e_prev - 1.0, w_target: w_target_force, w_open });
        }
        e_prev = e;
    }
    out
}

fn month_ends(p: &Panel) -> Vec<usize> {
    let d = p.dates();
    (0..d.len()).filter(|&i| i + 1 == d.len() || !d[i].same_month(d[i + 1])).collect()
}

/// Month-end decisions: `k` month-end closes averaged, `include_current` or the `k` before it.
fn s1_decisions(p: &Panel, include_current: bool) -> std::collections::BTreeMap<usize, Vec<f64>> {
    let me = month_ends(p);
    let mut out = std::collections::BTreeMap::new();
    let need = if include_current { 10 } else { 11 };
    for (m, &i) in me.iter().enumerate() {
        if m + 1 < need {
            continue;
        }
        let window: Vec<usize> = if include_current { me[m + 1 - 10..=m].to_vec() } else { me[m - 10..m].to_vec() };
        let w: Vec<f64> = (0..p.n_assets())
            .map(|a| {
                let c = p.closes(a);
                let mut s = 0.0;
                for &k in &window {
                    s += c[k];
                }
                if c[i] > s / 10.0 {
                    0.2
                } else {
                    0.0
                }
            })
            .collect();
        out.insert(i, w);
    }
    out
}

fn check_s1(m: Mutant, decisions: std::collections::BTreeMap<usize, Vec<f64>>, every_bar: bool) {
    let fx = fixtures();
    let p = &fx.etf_panel;
    let start_i = *decisions.keys().next().unwrap();
    let rows = ledger(p, &decisions, every_bar, start_i);
    let run = run_mutant(&fx, m).unwrap();
    assert_eq!(run.rows.dates.len(), rows.len(), "{}: bar count", m.name());
    let (wt, wh) = (run.rows.w_target.as_ref().unwrap(), run.rows.w_held.as_ref().unwrap());
    for (i, lr) in rows.iter().enumerate() {
        assert_eq!(run.rows.dates[i], p.dates()[lr.bar], "{}: date at row {i}", m.name());
        assert!(
            (run.rows.ret[i] - lr.ret).abs() <= TOL,
            "{} row {i}: ret {} vs ledger {}",
            m.name(),
            run.rows.ret[i],
            lr.ret
        );
        for a in 0..p.n_assets() {
            assert!((wt[i][a] - lr.w_target[a]).abs() <= TOL, "{} row {i} target {a}", m.name());
            assert!(
                (wh[i][a] - lr.w_open[a]).abs() <= 1e-9,
                "{} row {i} held {a}: {} vs {}",
                m.name(),
                wh[i][a],
                lr.w_open[a]
            );
        }
    }
}

#[test]
fn the_plain_ledger_reproduces_the_certified_rule() {
    // Base check of the test-side ledger against the committed PYTHON key (S1 returns), so the ledger is trustworthy.
    let p = s1_panel();
    let dec = s1_decisions(&p, true);
    let start_i = *dec.keys().next().unwrap();
    let rows = ledger(&p, &dec, false, start_i);
    let (kd, kr) = parse_returns(PY_KEY_S1_RETURNS);
    assert_eq!(rows.len(), kd.len());
    for (lr, (d, r)) in rows.iter().zip(kd.iter().zip(&kr)) {
        assert_eq!(p.dates()[lr.bar], *d);
        assert!((lr.ret - r).abs() <= 1e-12, "{d}: ledger {} vs Python key {r}", lr.ret);
    }
}

#[test]
fn s1_one_bar_late_executes_each_month_end_decision_on_the_next_bar() {
    let fx = fixtures();
    let dec = s1_decisions(&fx.etf_panel, true);
    // ledger start = the first (undelayed) decision bar; every decision moves one bar later
    let start_i = *dec.keys().next().unwrap();
    let late: std::collections::BTreeMap<usize, Vec<f64>> = dec.iter().map(|(i, w)| (i + 1, w.clone())).collect();
    let rows = ledger(&fx.etf_panel, &late, false, start_i);
    let run = run_mutant(&fx, Mutant::S1OneBarLate).unwrap();
    assert_eq!(run.rows.dates.len(), rows.len());
    let (wt, wh) = (run.rows.w_target.as_ref().unwrap(), run.rows.w_held.as_ref().unwrap());
    // the first row is flat (the fill happens on the NEXT bar): return 0, no target, no holding
    assert_eq!(run.rows.ret[0], 0.0);
    assert!(wt[0].iter().all(|&x| x == 0.0) && wh[0].iter().all(|&x| x == 0.0));
    for (i, lr) in rows.iter().enumerate() {
        assert_eq!(run.rows.dates[i], fx.etf_panel.dates()[lr.bar]);
        assert!((run.rows.ret[i] - lr.ret).abs() <= TOL, "row {i}");
        for a in 0..5 {
            assert!(
                (wt[i][a] - lr.w_target[a]).abs() <= TOL && (wh[i][a] - lr.w_open[a]).abs() <= 1e-9,
                "row {i} asset {a}"
            );
        }
    }
}

#[test]
fn s1_daily_rebalanced_restores_twenty_percent_every_bar() {
    let fx = fixtures();
    let dec = s1_decisions(&fx.etf_panel, true);
    check_s1(Mutant::S1WrongRebalanceMode, dec, true);
}

#[test]
fn s1_sma_excludes_current_averages_the_ten_month_ends_before_it() {
    let fx = fixtures();
    let dec = s1_decisions(&fx.etf_panel, false);
    check_s1(Mutant::S1SmaExcludesCurrent, dec, false);
}
