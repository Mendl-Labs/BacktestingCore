//! ALWAYS-ON golden tests on the small SYNTHETIC ladder that ships with the crate.
//!
//! The real-history goldens (`golden.rs`, `golden_fx.rs`) need vendor-derived data that a public repository must not
//! carry, so they are env-gated (`REFRULES_LADDER_DIR`). These tests are their always-on counterparts: the same
//! questions asked of `tests/data/synthetic_ladder_candles.csv` (SPY EFA IEF DBC VNQ BTC ETH, deterministic
//! synthetic prices with a missing SPY day and holes in the ETH series) and of the answer key that the pinned
//! `shadow.py` produced on it (`key_S1_signals.csv`, `key_S3_signals.csv`). The three fixtures are byte-identical
//! copies of `weightsim/tests/fixtures/` (generated once by `gen_answer_key.py` there; never regenerated here).
//! The key is used ONLY as expected output.
//!
//! What the synthetic data must show to be worth anything (asserted below, not assumed): both 0 and 1 signals for
//! every instrument, several flips, refusals at the same kind of places (early history, a hole in the window).

mod common;

use common::*;
use reference_rules::*;

const SYNTH_LADDER_SHA: &str = "1c42f9b3f9be74093b08ec99043c202c8f4885d9609d38ffa2b16b4929176efc";
const KEY_S1_SHA: &str = "002f6db99f7043e90196d2dcd57bd822320dc25e3b7a288daca5ae9f86cbec49";
const KEY_S3_SHA: &str = "9b6bf8900cbd5babb04cfc3bdc793f6048540d6520d4ba17896b235ae92f7d56";

fn etf_panel() -> Panel {
    panel_of(&load_synthetic_ladder(), &ETF_SYMBOLS)
}

fn crypto_panel() -> Panel {
    panel_of(&load_synthetic_ladder(), &CRYPTO_SYMBOLS)
}

/// Both values occur for every instrument and the signals change many times, so agreement is not vacuous.
fn assert_key_has_power(
    rows: &[(chrono::NaiveDate, Vec<u8>)],
    n_instruments: usize,
    min_flips: usize,
) {
    let mut total = 0;
    for j in 0..n_instruments {
        let col: Vec<u8> = rows.iter().map(|r| r.1[j]).collect();
        assert!(col.contains(&0) && col.contains(&1), "instrument {j}");
        let flips = col.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(flips >= 1, "instrument {j}: no flip");
        total += flips;
    }
    assert!(total >= min_flips, "only {total} flips");
}

#[test]
fn synthetic_inputs_are_the_recorded_files() {
    assert_eq!(sha256_of("synthetic_ladder_candles.csv"), SYNTH_LADDER_SHA);
    assert_eq!(sha256_of("key_S1_signals.csv"), KEY_S1_SHA);
    assert_eq!(sha256_of("key_S3_signals.csv"), KEY_S3_SHA);
}

/// When the weightsim crate is next to this one (the Core checkout), the copies are byte-identical to its fixtures
/// (they are pinned there by MANIFEST.sha256; this catches a drift between the two copies). Skipped, loudly, when
/// the sibling directory is absent (for example in a packaged copy of the crate).
#[test]
fn synthetic_inputs_are_byte_identical_to_the_weightsim_fixtures() {
    let sibling = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("weightsim")
        .join("tests")
        .join("fixtures");
    if !sibling.is_dir() {
        println!(
            "SKIPPED synthetic_inputs_are_byte_identical_to_the_weightsim_fixtures: {} not found",
            sibling.display()
        );
        return;
    }
    for name in [
        "synthetic_ladder_candles.csv",
        "key_S1_signals.csv",
        "key_S3_signals.csv",
    ] {
        assert_eq!(
            std::fs::read(data_path(name)).unwrap(),
            std::fs::read(sibling.join(name)).unwrap(),
            "{name} differs from weightsim/tests/fixtures/{name}"
        );
    }
}

/// ETF trend, explicit month-end mode: every month-end of the key (21 dates, 105 signals), none skipped, weights
/// exactly 0.2 * signal.
#[test]
fn synthetic_etf_every_month_end_explicit_mode() {
    let panel = etf_panel();
    let (header, rows) = load_key("key_S1_signals.csv");
    assert_eq!(header, ETF_SYMBOLS.to_vec());
    assert_key_has_power(&rows, 5, 10);
    let opts = Options::etf_replay(MonthEndMode::Explicit);
    let (mut hit, mut tot) = (0, 0);
    for (date, expected) in &rows {
        let dec = decide_etf_trend(&panel, *date, &opts).unwrap_or_else(|e| panic!("{date}: {e}"));
        for (sym, exp) in ETF_SYMBOLS.iter().zip(expected) {
            let got = dec.get(sym).unwrap();
            hit += usize::from(got.signal.as_int() == *exp);
            tot += 1;
            assert_eq!(got.weight, 0.2 * f64::from(*exp), "{date} {sym}");
        }
    }
    println!(
        "synthetic S1 ETF trend (explicit mode): {hit}/{tot} signals agree over {} month-ends",
        rows.len()
    );
    assert_eq!((hit, tot, rows.len()), (105, 105, 21));
}

/// Calendar-free mode: every month-end that has a later-month bar agrees; the final month-end (no July bar in the
/// data) is refused with MonthNotComplete rather than decided.
#[test]
fn synthetic_etf_next_month_bar_mode() {
    let panel = etf_panel();
    let (_, rows) = load_key("key_S1_signals.csv");
    let opts = Options::etf_replay(MonthEndMode::NextMonthBar);
    let (mut hit, mut tot, mut refused) = (0, 0, Vec::new());
    for (date, expected) in &rows {
        match decide_etf_trend(&panel, *date, &opts) {
            Ok(dec) => {
                for (sym, exp) in ETF_SYMBOLS.iter().zip(expected) {
                    hit += usize::from(dec.get(sym).unwrap().signal.as_int() == *exp);
                    tot += 1;
                }
            }
            Err(e) => refused.push((*date, e)),
        }
    }
    println!(
        "synthetic S1 ETF trend (next-month-bar mode): {hit}/{tot} agree; refused: {refused:?}"
    );
    assert_eq!((hit, tot), (100, 100));
    assert_eq!(refused.len(), 1);
    assert_eq!(refused[0].0, d(2019, 6, 28));
    assert!(matches!(refused[0].1, RuleError::MonthNotComplete { .. }));
}

/// The month-end helper reproduces the key's list of month-end dates, and history before the first key date is
/// refused (fewer than 10 month-ends).
#[test]
fn synthetic_month_end_helper_and_warmup() {
    let panel = etf_panel();
    let (_, rows) = load_key("key_S1_signals.csv");
    for sym in ETF_SYMBOLS {
        let all = month_end_dates(panel.get(sym).unwrap());
        assert_eq!(all.len(), 30, "{sym}");
        let key_dates: Vec<_> = rows.iter().map(|r| r.0).collect();
        assert_eq!(&all[9..], &key_dates[..], "{sym}");
        assert_eq!(
            completed_month_end_dates(panel.get(sym).unwrap()),
            all[..29].to_vec()
        );
    }
    assert_eq!(
        latest_decision_date(&panel, &ETF_SYMBOLS).unwrap(),
        d(2019, 5, 31)
    );
    let opts = Options::etf_replay(MonthEndMode::Explicit);
    let all = month_end_dates(panel.get("SPY").unwrap());
    for (k, date) in all[..9].iter().enumerate() {
        match decide_etf_trend(&panel, *date, &opts) {
            Err(RuleError::InsufficientHistory {
                needed: 10, have, ..
            }) => assert_eq!(have, k + 1),
            other => panic!("{date}: {other:?}"),
        }
    }
}

/// Crypto trend with the production gap policy (no missing calendar day in the 100-bar window). The synthetic ETH
/// series misses 2016-03-15 (a day the key does not list either: the shadow drops it with `dropna`), so exactly the
/// key days whose window contains it, 2016-03-16 ..= 2016-06-22 (99 days), are refused with DataGap (a block of
/// consecutive days that is NOT at the start of the history, unlike the real data); every other day agrees.
#[test]
fn synthetic_crypto_every_day_strict_gap_policy() {
    let panel = crypto_panel();
    let (header, rows) = load_key("key_S3_signals.csv");
    assert_eq!(header, CRYPTO_SYMBOLS.to_vec());
    assert_key_has_power(&rows, 2, 10);
    let opts = Options::crypto_replay();
    let (mut hit, mut tot) = (0usize, 0usize);
    let mut skipped = Vec::new();
    for (date, expected) in &rows {
        match decide_crypto_trend(&panel, *date, &opts) {
            Ok(dec) => {
                for (sym, exp) in CRYPTO_SYMBOLS.iter().zip(expected) {
                    let got = dec.get(sym).unwrap();
                    hit += usize::from(got.signal.as_int() == *exp);
                    tot += 1;
                    assert_eq!(got.weight, 0.5 * f64::from(*exp));
                }
            }
            Err(e) => skipped.push((*date, e)),
        }
    }
    let skipped_dates: Vec<_> = skipped.iter().map(|s| s.0).collect();
    println!(
        "synthetic S3 crypto trend (strict gaps): {hit}/{tot} agree over {} days; skipped {} days ({} .. {})",
        rows.len() - skipped.len(),
        skipped.len(),
        skipped_dates.first().map(|x| x.to_string()).unwrap_or_default(),
        skipped_dates.last().map(|x| x.to_string()).unwrap_or_default(),
    );
    assert_eq!(hit, tot);
    assert!(tot > 0);
    assert!(
        skipped
            .iter()
            .all(|s| matches!(s.1, RuleError::DataGap { .. })),
        "{skipped:?}"
    );
    // The hole (2016-03-15) is inside the 100-bar window of exactly the days 2016-03-15 .. 2016-06-22 (100 calendar
    // days; the key has no row for 2016-03-15 itself), and of no other day of the key.
    assert!(skipped_dates
        .windows(2)
        .all(|w| (w[1] - w[0]).num_days() == 1));
    assert!(skipped_dates
        .iter()
        .all(|x| *x >= d(2016, 3, 16) && *x <= d(2016, 6, 22)));
    assert_eq!(skipped_dates.len(), 99);
    assert_eq!((hit, tot), (348, 348));
    assert_eq!(hit + 2 * skipped.len(), 2 * rows.len());
}

/// Pure-rule agreement as the reference evaluates it: joint (dropna) panel, no gap check. Every one of the key's days
/// is evaluated and all agree.
#[test]
fn synthetic_crypto_every_day_reference_semantics() {
    let panel = inner_join(&crypto_panel());
    let (_, rows) = load_key("key_S3_signals.csv");
    let mut opts = Options::crypto_replay();
    opts.gap_policy = GapPolicy::Unchecked;
    let (mut hit, mut tot) = (0, 0);
    for (date, expected) in &rows {
        let dec =
            decide_crypto_trend(&panel, *date, &opts).unwrap_or_else(|e| panic!("{date}: {e}"));
        for (sym, exp) in CRYPTO_SYMBOLS.iter().zip(expected) {
            hit += usize::from(dec.get(sym).unwrap().signal.as_int() == *exp);
            tot += 1;
        }
    }
    println!("synthetic S3 crypto trend (reference semantics: joint panel, gaps unchecked): {hit}/{tot} agree over {} days, none skipped", rows.len());
    assert_eq!((hit, tot), (2 * rows.len(), 2 * rows.len()));
    assert_eq!(rows.len(), 273);
}
