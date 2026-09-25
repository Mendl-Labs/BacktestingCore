//! Always-on: the library-rule adapters reproduce the committed PYTHON answer key of `weightsim`'s synthetic fixture
//! (`../weightsim/tests/fixtures/`, generated once by `gen_answer_key.py` from the pinned `shadow.py` logic), and the
//! adapter plumbing (schedule, refusal mapping, causality) behaves as documented.

mod common;

use common::*;
use weightsim::*;
use weightsim_rules::ladder::mutants::PeekCrypto;
use weightsim_rules::{CryptoTrendRule, EtfTrendRule};

fn s3_config() -> SimConfig {
    SimConfig { start: Some(d("2016-01-01")), end: Some(d("2020-12-31")), ..SimConfig::default() }
}

#[test]
fn etf_adapter_reproduces_the_python_key_per_bar() {
    let panel = s1_panel();
    let sim = simulate(&panel, &EtfTrendRule, &SimConfig::default()).unwrap();
    let (kd, kr) = parse_returns(PY_KEY_S1_RETURNS);
    assert_eq!(sim.window_dates(), kd.as_slice(), "counted bars must be exactly the key's bars");
    let diff = max_abs_diff(sim.window_returns(), &kr);
    println!("ETF adapter vs Python key (synthetic): {} bars, max|ret - key| = {diff:e}", kd.len());
    assert!(diff <= 1e-15, "S1 adapter differs from the Python key by {diff:e}");
    // Same weights as the independent oracle rule, bit for bit, on every bar.
    let oracle = simulate(&panel, &OracleS1, &SimConfig::default()).unwrap();
    assert_eq!(sim.ret, oracle.ret);
    assert_eq!(sim.target_weights, oracle.target_weights);
    assert_eq!(sim.held_weights, oracle.held_weights);
    let flips: u64 = sim.signal_flips.iter().sum();
    assert_eq!(flips as f64, py_key_metric("S1", "flips"));
}

#[test]
fn crypto_adapter_reproduces_the_python_key_per_bar() {
    let panel = s3_panel();
    let sim = simulate(&panel, &CryptoTrendRule, &s3_config()).unwrap();
    let (kd, kr) = parse_returns(PY_KEY_S3_RETURNS);
    assert_eq!(sim.window_dates(), kd.as_slice(), "counted bars must be exactly the key's bars");
    let diff = max_abs_diff(sim.window_returns(), &kr);
    println!("Crypto adapter vs Python key (synthetic): {} bars, max|ret - key| = {diff:e}", kd.len());
    assert!(diff <= 1e-15, "S3 adapter differs from the Python key by {diff:e}");
    let oracle = simulate(&panel, &OracleS3, &s3_config()).unwrap();
    assert_eq!(sim.ret, oracle.ret);
    assert_eq!(sim.target_weights, oracle.target_weights);
    let flips: u64 = sim.signal_flips.iter().sum();
    assert_eq!(flips as f64, py_key_metric("S3", "flips"));
    let m = sim.metrics().unwrap();
    assert!((m.sharpe - py_key_metric("S3", "sharpe")).abs() < 1e-9);
    assert!((m.cagr - py_key_metric("S3", "cagr")).abs() < 1e-9);
}

#[test]
fn adapters_match_the_python_key_signals_on_every_decision_date() {
    // ETF: month-end decision dates and weights (0.2 x signal).
    let panel = s1_panel();
    let sim = simulate(&panel, &EtfTrendRule, &SimConfig::default()).unwrap();
    let (sd, rows) = parse_signals(PY_KEY_S1_SIGNALS);
    let dec: Vec<Date> = (0..sim.n_bars()).filter(|&t| sim.decision[t]).map(|t| sim.dates[t]).collect();
    assert_eq!(dec, sd, "decision dates must equal the key's month-end signal dates");
    let mut cells = 0;
    for (date, row) in sd.iter().zip(&rows) {
        let t = sim.dates.iter().position(|x| x == date).unwrap();
        for (i, s) in row.iter().enumerate() {
            assert!((sim.row(&sim.target_weights, t)[i] - 0.2 * s).abs() <= 1e-12, "{date} asset {i}");
            cells += 1;
        }
    }
    assert!(cells > 100);
    // Crypto: every listed day.
    let panel = s3_panel();
    let sim = simulate(&panel, &CryptoTrendRule, &s3_config()).unwrap();
    let (sd, rows) = parse_signals(PY_KEY_S3_SIGNALS);
    let mut cells = 0;
    for (date, row) in sd.iter().zip(&rows) {
        let t = sim.dates.iter().position(|x| x == date).unwrap();
        assert!(sim.decision[t]);
        for (i, s) in row.iter().enumerate() {
            assert!((sim.row(&sim.target_weights, t)[i] - 0.5 * s).abs() <= 1e-12, "{date} asset {i}");
            cells += 1;
        }
    }
    assert!(cells > 400);
}

#[test]
fn adapter_series_digests_are_stable_across_runs() {
    let panel = s3_panel();
    let a = simulate(&panel, &CryptoTrendRule, &s3_config()).unwrap();
    let b = simulate(&panel, &CryptoTrendRule, &s3_config()).unwrap();
    assert_eq!(a.series_sha256, b.series_sha256);
    let e = simulate(&s1_panel(), &EtfTrendRule, &SimConfig::default()).unwrap();
    assert_ne!(e.series_sha256, a.series_sha256);
}

#[test]
fn etf_warmup_is_a_warmup_refusal_and_abort_tolerates_it_until_the_first_decision() {
    // Under HoldPrevious the early month-ends are recorded as Warmup refusals with the rule's own code.
    let panel = s1_panel();
    let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() };
    let sim = simulate(&panel, &EtfTrendRule, &cfg).unwrap();
    assert!(!sim.refusals.is_empty());
    assert!(sim.refusals.iter().all(|r| r.kind == RefusalKind::Warmup && r.code == "insufficient_history"));
    // The first decision comes at the 10th month-end and no refusal follows it.
    let first_dec = (0..sim.n_bars()).find(|&t| sim.decision[t]).unwrap();
    assert!(sim.refusals.iter().all(|r| r.bar < first_dec));
    // Under Abort the very same run succeeds (warm-up is tolerated before the first success)...
    assert!(simulate(&panel, &EtfTrendRule, &SimConfig::default()).is_ok());
    // ...and the two runs make the same decisions.
    let abort = simulate(&panel, &EtfTrendRule, &SimConfig::default()).unwrap();
    assert_eq!(abort.decision, sim.decision);
}

/// Weekday dates from `start`, skipping `hole` (inclusive range of day offsets counted in weekdays).
fn weekday_calendar(start: &str, n: usize, hole: Option<(usize, usize)>) -> Vec<Date> {
    let mut out = Vec::new();
    let mut cur = d(start);
    let mut k = 0;
    while out.len() < n {
        if cur.weekday() < 5 {
            let in_hole = hole.is_some_and(|(a, b)| k >= a && k <= b);
            if !in_hole {
                out.push(cur);
            }
            k += 1;
        }
        cur = cur.add_days(1);
    }
    out
}

fn synthetic_etf_panel_with_hole(hole: Option<(usize, usize)>) -> Panel {
    let dates = weekday_calendar("2017-01-02", 400, hole);
    let closes: Vec<Vec<f64>> = (0..5)
        .map(|a| {
            let mut p = 100.0 + 10.0 * a as f64;
            (0..dates.len())
                .map(|t| {
                    p *= 1.0 + 0.0004 * ((t * (3 + a)) % 7) as f64 - 0.0011;
                    p
                })
                .collect()
        })
        .collect();
    Panel::new(ETF.iter().map(|s| s.to_string()).collect(), dates, closes).unwrap()
}

#[test]
fn a_data_gap_is_a_data_refusal_with_its_own_code() {
    // A ten-weekday hole inside the ten-month-end window: the rule's strict replay gap policy refuses (`DataGap`).
    let panel = synthetic_etf_panel_with_hole(Some((200, 210)));
    let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..SimConfig::default() };
    let sim = simulate(&panel, &EtfTrendRule, &cfg).unwrap();
    let data: Vec<_> = sim.refusals.iter().filter(|r| r.kind == RefusalKind::Data).collect();
    assert!(!data.is_empty(), "expected data-gap refusals, got {:?}", sim.refusals);
    assert!(data.iter().all(|r| r.code == "data_gap"));
    // Under Abort a Data refusal after the first success is fatal, not tolerated.
    let err = simulate(&panel, &EtfTrendRule, &SimConfig::default()).unwrap_err();
    assert!(matches!(err, SimError::RuleRefused { ref refusal, .. } if refusal.kind == RefusalKind::Data), "{err}");
    // Without the hole there is no Data refusal at all.
    let clean = synthetic_etf_panel_with_hole(None);
    let sim = simulate(&clean, &EtfTrendRule, &cfg).unwrap();
    assert!(sim.refusals.iter().all(|r| r.kind == RefusalKind::Warmup));
}

#[test]
fn the_crypto_adapter_ignores_calendar_gaps_by_design() {
    // GapPolicy::Unchecked: the synthetic ETH series has holes and the adapter still decides on them.
    let sim = simulate(&s3_panel(), &CryptoTrendRule, &s3_config()).unwrap();
    assert!(sim.refusals.is_empty());
    let days: Vec<i64> = sim.dates.windows(2).map(|w| w[0].days_until(w[1])).collect();
    assert!(days.iter().any(|&x| x > 1), "the fixture is expected to contain calendar holes");
}

#[test]
fn simulator_poisoning_is_clean_for_the_adapters_and_catches_a_peeking_rule() {
    let cfg = SimConfig::default();
    let etf = s1_panel();
    let r = weightsim::harness::check_poisoning(&|_p: &Panel| EtfTrendRule, &etf, &cfg, 300, 7).unwrap();
    assert!(r.is_clean(), "{:?}", r.mismatches);
    let cry = s3_panel();
    let cfg3 = s3_config();
    let r = weightsim::harness::check_poisoning(&|_p: &Panel| CryptoTrendRule, &cry, &cfg3, 200, 7).unwrap();
    assert!(r.is_clean(), "{:?}", r.mismatches);
    // The same-day-peek mutant reads the future through its own copy of the panel: the harness must flag it. Only
    // the decision made on the last kept bar can differ, and a signal rarely flips on one bar, so try many cuts.
    let flagged = (150..420usize)
        .step_by(7)
        .filter(|&cut| {
            !weightsim::harness::check_poisoning(&|p: &Panel| PeekCrypto::new(p), &cry, &cfg3, cut, 7)
                .unwrap()
                .is_clean()
        })
        .count();
    assert!(flagged >= 3, "the poisoning harness caught the peeking rule at only {flagged} cuts");
    // ...while the honest adapter is clean at every one of those cuts.
    for cut in (150..420usize).step_by(7) {
        let r = weightsim::harness::check_poisoning(&|_p: &Panel| CryptoTrendRule, &cry, &cfg3, cut, 7).unwrap();
        assert!(r.is_clean(), "cut {cut}: {:?}", r.mismatches);
    }
}

#[test]
fn rule_truncation_is_clean_for_the_adapters() {
    let etf = s1_panel();
    let samples: Vec<usize> = (150..600).step_by(23).collect();
    assert!(weightsim::harness::check_rule_truncation(&|_p: &Panel| EtfTrendRule, &etf, &samples).is_empty());
    let cry = s3_panel();
    let samples: Vec<usize> = (120..420).step_by(17).collect();
    assert!(weightsim::harness::check_rule_truncation(&|_p: &Panel| CryptoTrendRule, &cry, &samples).is_empty());
    // ... and it flags the peeking rule.
    assert!(!weightsim::harness::check_rule_truncation(&|p: &Panel| PeekCrypto::new(p), &cry, &samples).is_empty());
}

#[test]
fn adapters_match_a_direct_call_of_the_rule_crate() {
    use reference_rules::{decide_crypto_trend, GapPolicy, Options, PriceSeries};
    let cry = s3_panel();
    let t = 250;
    let dates: Vec<chrono::NaiveDate> = cry.dates()[..=t].iter().map(|&x| weightsim_rules::to_naive(x)).collect();
    let series: Vec<PriceSeries> =
        (0..2).map(|i| PriceSeries::new(CRY[i], dates.clone(), cry.closes(i)[..=t].to_vec()).unwrap()).collect();
    let panel = reference_rules::Panel::new(series).unwrap();
    let opts = Options { gap_policy: GapPolicy::Unchecked, ..Options::crypto_replay() };
    let direct = decide_crypto_trend(&panel, *dates.last().unwrap(), &opts).unwrap();
    let want: Vec<f64> = direct.instruments.iter().map(|i| i.weight).collect();
    // Run the adapter through the simulator with a probe that records what the rule returned at bar t.
    let sim = simulate(&cry, &CryptoTrendRule, &SimConfig::default()).unwrap();
    assert_eq!(sim.row(&sim.target_weights, t), want.as_slice());
    assert!(sim.decision[t]);
}
