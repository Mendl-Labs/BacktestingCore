//! weightsim 0.3, council Ruling 4 / work item W8: PER-SLEEVE execution delay and the delay-sensitivity table (W1).
//!
//! Semantics under test (also in the crate docs, P15): a sleeve's decision at the close of ITS OWN bar `t` is executed at the
//! close of ITS OWN bar `t + d_sleeve`; `SleeveSpec::execution_delay = None` means the book-level
//! `BookConfig::sim.execution_delay_bars`, so a book that never sets a per-sleeve delay is bit-identical to 0.2 (that is the
//! whole of `book_identity.rs`, `book_key.rs`, `book_props.rs` and `book_causality.rs`, unmodified, with their pinned digests).
//!
//! What is proved here, all with always-on synthetic data:
//!  * hand-computed timing on a weekend calendar (own bars, not account bars; Saturday is not a fill day for an ETF at d = 1);
//!  * the book-level value is the default and a per-sleeve `Some(d)` overrides it, bit for bit (digest equality);
//!  * a one-sleeve book with a per-sleeve delay is bit-identical to `simulate` with that delay (the T1 identity survives);
//!  * a mixed-delay book equals, sleeve by sleeve, the single-sleeve books run with the same delays (shadows and cadence flags
//!    bit for bit) and the attribution identity holds to 1e-12;
//!  * INDEPENDENT ORACLE: a mixed-delay book equals, on every column except the rule's own decision bookkeeping, the same book
//!    whose rules were wrapped so that each emits its decision `d` own bars late, run with delay 0 everywhere: this exercises
//!    the gross-cap refusal, the trade filter, the cash policy, shares, the allocator and the shadows under delay;
//!  * poisoning, truncation and determinism harnesses with per-sleeve delays; a leaky rule is still caught;
//!  * a delay that is not smaller than the sleeve's own history is refused, one less executes only the first decision;
//!  * the delay-sensitivity table on the S1 and S3 synthetic fixtures, equal row by row to `simulate` at that delay.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::book::*;
use common::*;
use std::collections::HashSet;
use std::sync::Mutex;
use weightsim::*;

const NET: CostModel = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Daily panel from named price series starting at `start`; exchange series skip Saturday and Sunday.
fn mini_panel(start: &str, days: usize, series: &[(&str, bool, Vec<f64>)]) -> BookPanel {
    let all: Vec<Date> = (0..days).map(|i| d(start).add_days(i as i64)).collect();
    let mut rows = Vec::new();
    for (name, exchange, prices) in series {
        let mut r = Vec::new();
        let mut p = 0;
        for date in &all {
            if *exchange && date.weekday() >= 5 {
                continue;
            }
            r.push((*date, prices[p]));
            p += 1;
        }
        assert_eq!(p, prices.len(), "{name}: price count must match the number of bars");
        let session = if *exchange { SessionKind::exchange("test", vec![]) } else { SessionKind::Continuous };
        rows.push((name.to_string(), session, r));
    }
    BookPanel::from_dated_series(rows).unwrap()
}

/// A sleeve that decides on its first own bar only (a buy-and-hold entry) and refuses afterwards.
fn once_rule(universe: &[&'static str], w: Vec<f64>, policy: RebalancePolicy) -> FnRule {
    FnRule::new(universe, DecisionSchedule::Daily, policy, 1, move |h| {
        if h.len() == 1 {
            Ok(w.clone())
        } else {
            Err(RuleRefusal::data("no_change", "the rule decides once"))
        }
    })
}

fn hold_cfg() -> BookConfig {
    let mut cfg = BookConfig::default();
    cfg.sim.on_refusal = OnRefusal::HoldPrevious;
    cfg
}

/// `book` with the per-sleeve delays `delays[s]` (in sleeve order).
fn with_delays(mut book: Book, delays: &[usize]) -> Book {
    assert_eq!(book.sleeves.len(), delays.len());
    for (sp, &x) in book.sleeves.iter_mut().zip(delays) {
        sp.execution_delay = Some(x);
    }
    book
}

fn digest(panel: &BookPanel, book: &Book, cfg: &BookConfig) -> String {
    simulate_book(panel, book, cfg).unwrap().series_sha256
}

// ------------------------------------------------------------------------------------------- hand-computed timing

/// Friday 2020-01-03 .. Tuesday 2020-01-07. The ETF-like sleeve `e` (exchange: Fri, Mon, Tue) enters once with weight 0.6;
/// the crypto-like sleeve `c` (every day) holds 0.5 of its share every bar.
fn weekend_panel() -> BookPanel {
    mini_panel(
        "2020-01-03",
        5,
        &[("E", true, vec![100.0, 110.0, 121.0]), ("C", false, vec![10.0, 11.0, 12.0, 11.0, 10.0])],
    )
}

fn weekend_book() -> Book {
    Book::new(vec![
        SleeveSpec::from_rule(
            "e",
            once_rule(&["E"], vec![0.6], RebalancePolicy::OnDecision),
            vec![0],
            ShareSpec::Fixed(1.0),
        ),
        SleeveSpec::from_rule(
            "c",
            const_rule(&["C"], vec![0.5], DecisionSchedule::Daily, RebalancePolicy::EveryBar),
            vec![1],
            ShareSpec::Fixed(0.5),
        ),
    ])
}

#[test]
fn etf_at_d1_and_crypto_at_d0_trade_on_their_own_bars_hand_computed() {
    let panel = weekend_panel();
    let book = with_delays(weekend_book(), &[1, 0]);
    let r = simulate_book(&panel, &book, &hold_cfg()).unwrap();
    // account bars: 0 Fri, 1 Sat, 2 Sun, 3 Mon, 4 Tue. Instrument 0 = E, 1 = C.
    let unit = |k: usize, j: usize| r.units[k * 2 + j];
    // crypto, d = 0: first target effective on its bar 0: units = share * w * equity / price = 0.5 * 0.5 * 1.0 / 10
    assert!((unit(0, 1) - 0.025).abs() < 1e-15, "{}", unit(0, 1));
    // ETF, d = 1: decided on Friday (own bar 0), NOT filled on Friday, NOT on the closed Saturday or Sunday (no own bar),
    // filled on its next OWN bar, Monday (account bar 3), at Monday's close, sized on Monday's pre-trade equity
    assert_eq!(unit(0, 0), 0.0, "not on the decision bar");
    assert_eq!(unit(1, 0), 0.0, "not on the closed Saturday");
    assert_eq!(unit(2, 0), 0.0, "not on the closed Sunday");
    assert!(
        (unit(3, 0) - 0.6 * r.equity_pre[3] / 110.0).abs() < 1e-15,
        "{} vs {}",
        unit(3, 0),
        0.6 * r.equity_pre[3] / 110.0
    );
    assert_eq!(unit(4, 0), unit(3, 0), "OnDecision: it drifts afterwards");
    // the cadence flags: the ETF is due on Monday only, the crypto sleeve every bar
    assert_eq!(r.due.chunks(2).map(|c| c[0]).collect::<Vec<_>>(), [false, false, false, true, false]);
    assert!(r.due.chunks(2).all(|c| c[1]), "EveryBar");
    // and the decision itself was taken on Friday
    assert_eq!(r.decision.chunks(2).map(|c| c[0]).collect::<Vec<_>>(), [true, false, false, false, false]);
    assert!(r.attribution_report().max_abs_err_total <= 1e-12);
}

#[test]
fn the_delays_are_independent_per_sleeve_in_both_directions() {
    let panel = weekend_panel();
    // ETF immediate, crypto delayed by two own bars (its own bars are every day: Friday, Saturday, Sunday ...)
    let r = simulate_book(&panel, &with_delays(weekend_book(), &[0, 2]), &hold_cfg()).unwrap();
    let unit = |k: usize, j: usize| r.units[k * 2 + j];
    assert!((unit(0, 0) - 0.6 * 1.0 / 100.0).abs() < 1e-15, "ETF fills at once on Friday: {}", unit(0, 0));
    assert_eq!((unit(0, 1), unit(1, 1)), (0.0, 0.0), "crypto has not filled on its bars 0 and 1");
    assert!(
        (unit(2, 1) - 0.25 * r.equity_pre[2] / 12.0).abs() < 1e-15,
        "crypto fills on its own bar 2 (Sunday): {}",
        unit(2, 1)
    );
    assert_eq!(r.due.chunks(2).map(|c| c[0]).collect::<Vec<_>>(), [true, false, false, false, false]);
    // crypto EveryBar: due on every bar even before its first standing target exists (nothing to trade yet)
    assert!(r.due.chunks(2).all(|c| c[1]));
}

#[test]
fn the_book_level_delay_is_the_default_and_a_sleeve_value_overrides_it_bit_for_bit() {
    let panel = weekend_panel();
    let book_default =
        |x: usize| BookConfig { sim: SimConfig { execution_delay_bars: x, ..hold_cfg().sim }, ..hold_cfg() };
    // book-level 1, ETF inherits it, crypto overrides to 0  ==  explicit (1, 0) under book-level 0
    let a = digest(&panel, &with_delays(weekend_book(), &[1, 0]), &hold_cfg());
    let mut overridden = weekend_book();
    overridden.sleeves[1] = overridden.sleeves[1].clone().with_execution_delay(0);
    let b = digest(&panel, &overridden, &book_default(1));
    assert_eq!(a, b, "inherit for e (book-level 1), Some(0) for c");
    // no override at all under book-level 1  ==  explicit (1, 1) under book-level 0
    assert_eq!(
        digest(&panel, &weekend_book(), &book_default(1)),
        digest(&panel, &with_delays(weekend_book(), &[1, 1]), &hold_cfg())
    );
    // the override really is an override: Some(0) under a non-zero default differs from inheriting
    assert_ne!(a, digest(&panel, &weekend_book(), &book_default(1)));
    // and delay 0 everywhere is the same as never having set anything
    assert_eq!(
        digest(&panel, &weekend_book(), &hold_cfg()),
        digest(&panel, &with_delays(weekend_book(), &[0, 0]), &hold_cfg())
    );
    // effective_delay is the accessor of that rule
    let sp = weekend_book().sleeves[0].clone();
    assert_eq!((sp.effective_delay(2), sp.clone().with_execution_delay(0).effective_delay(2)), (2, 0));
}

/// The decision of own bar `t` is the weight `(t + 1) / 100`; with delay `d` the standing target after own bar `t` is the
/// decision of own bar `t - d`, i.e. `(t - d + 1) / 100`: a value that identifies WHICH decision is in force.
fn ramp_sleeve(delay: usize) -> SleeveSpec {
    SleeveSpec::from_rule(
        "e",
        FnRule::new(&["E"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| {
            Ok(vec![h.len() as f64 / 100.0])
        }),
        vec![0],
        ShareSpec::Fixed(1.0),
    )
    .with_execution_delay(delay)
}

#[test]
fn a_decision_at_own_bar_t_is_in_force_from_own_bar_t_plus_d_across_weekends() {
    // Fri 01-03 .. Tue 01-14 (12 days). E (exchange) has 8 own bars: u0 Fri, u3 Mon, u4 Tue, u5 Wed, u6 Thu, u7 Fri, u10 Mon, u11 Tue.
    // C decides once on bar 0 and is never due again, so on the weekend bars nothing is constructed and the recorded target is
    // the standing one.
    let panel = mini_panel(
        "2020-01-03",
        12,
        &[("E", true, (0..8).map(|i| 100.0 + i as f64).collect()), ("C", false, vec![10.0; 12])],
    );
    let own_bar_of_union =
        [Some(0), None, None, Some(1), Some(2), Some(3), Some(4), Some(5), None, None, Some(6), Some(7)];
    let mk = |delay: usize| {
        Book::new(vec![
            ramp_sleeve(delay),
            SleeveSpec::from_rule(
                "c",
                once_rule(&["C"], vec![0.0], RebalancePolicy::OnDecision),
                vec![1],
                ShareSpec::Fixed(0.5),
            ),
        ])
    };
    for delay in [0usize, 1, 2, 3] {
        let r = simulate_book(&panel, &mk(delay), &hold_cfg()).unwrap();
        // the standing decision index after each union bar: the last own bar t' <= current with t' - delay >= 0
        let mut last_own: Option<usize> = None;
        for u in 0..12 {
            if let Some(t) = own_bar_of_union[u] {
                last_own = Some(t);
            }
            let want = match last_own {
                Some(t) if t >= delay => (t - delay + 1) as f64 / 100.0,
                _ => 0.0,
            };
            let got = r.target_weights[u * 2];
            assert!((got - want).abs() < 1e-15, "delay {delay} union bar {u}: target {got} want {want}");
        }
        // on the closed weekend bars the ETF is not open, so it can neither decide nor be re-targeted
        for u in [1usize, 2, 8, 9] {
            assert!(!r.sleeve_open[u * 2] && !r.due[u * 2] && !r.decision[u * 2], "weekend bar {u}");
        }
        assert!(r.attribution_report().max_abs_err_total <= 1e-12);
    }
}

#[test]
fn a_delay_not_smaller_than_the_sleeves_own_history_is_refused_and_one_less_executes_the_first_decision_only() {
    let panel = mini_panel(
        "2020-01-03",
        12,
        &[("E", true, (0..8).map(|i| 100.0 + i as f64).collect()), ("C", false, vec![10.0; 12])],
    );
    let mk = |delay: usize| {
        Book::new(vec![
            ramp_sleeve(delay),
            SleeveSpec::from_rule(
                "c",
                once_rule(&["C"], vec![0.0], RebalancePolicy::OnDecision),
                vec![1],
                ShareSpec::Fixed(0.5),
            ),
        ])
    };
    // E has 8 own bars: delay 8 could never execute anything
    match simulate_book(&panel, &mk(8), &hold_cfg()).unwrap_err() {
        BookError::BadBook(m) => assert!(m.contains("sleeve e") && m.contains("execution delay of 8"), "{m}"),
        other => panic!("expected BadBook, got {other:?}"),
    }
    // usize::MAX: refused, no overflow
    assert!(matches!(simulate_book(&panel, &mk(usize::MAX), &hold_cfg()), Err(BookError::BadBook(_))));
    // the same refusal through the book-level default (no per-sleeve value)
    let mut cfg = hold_cfg();
    cfg.sim.execution_delay_bars = 8;
    let plain = Book::new(vec![
        SleeveSpec::from_rule(
            "e",
            FnRule::new(&["E"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| {
                Ok(vec![h.len() as f64 / 100.0])
            }),
            vec![0],
            ShareSpec::Fixed(1.0),
        ),
        SleeveSpec::from_rule(
            "c",
            once_rule(&["C"], vec![0.0], RebalancePolicy::OnDecision),
            vec![1],
            ShareSpec::Fixed(0.5),
        ),
    ]);
    assert!(matches!(simulate_book(&panel, &plain, &cfg), Err(BookError::BadBook(_))));
    // delay 7 = the last own bar: only the decision of own bar 0 is ever executed, on the last bar (union bar 11)
    let r = simulate_book(&panel, &mk(7), &hold_cfg()).unwrap();
    for u in 0..11 {
        assert_eq!(r.target_weights[u * 2], 0.0, "union bar {u}");
    }
    assert!((r.target_weights[11 * 2] - 0.01).abs() < 1e-15, "{}", r.target_weights[11 * 2]);
    // a sleeve that never executes anything before the end of the window is not an error (C11): every other sleeve is fine
    assert!(r.decision.chunks(2).filter(|c| c[0]).count() == 8);
}

// ------------------------------------------------------------------------------------------- identities with simulate

fn one_sleeve_book<R: WeightRule + 'static>(panel: &Panel, rule: R, delay: Option<usize>) -> (BookPanel, Book) {
    let n = panel.n_assets();
    let mut sp = SleeveSpec::from_rule("only", rule, (0..n).collect(), ShareSpec::Fixed(1.0));
    if let Some(x) = delay {
        sp = sp.with_execution_delay(x);
    }
    (BookPanel::from_panel(panel), Book::new(vec![sp]))
}

#[test]
fn a_one_sleeve_book_with_a_sleeve_delay_is_bit_identical_to_simulate_with_that_delay() {
    let s1 = s1_panel();
    let s3 = s3_panel();
    for dl in [0usize, 1, 2, 3, 5] {
        for (name, panel, cfg) in [("S1", &s1, net(s1_config())), ("S3", &s3, net(s3_config()))] {
            let want = if name == "S1" {
                simulate(panel, &S1TestRule, &SimConfig { execution_delay_bars: dl, ..cfg.clone() }).unwrap()
            } else {
                simulate(panel, &S3TestRule, &SimConfig { execution_delay_bars: dl, ..cfg.clone() }).unwrap()
            };
            // the delay carried by the SLEEVE, book-level delay 0
            let got = if name == "S1" {
                let (bp, book) = one_sleeve_book(panel, S1TestRule, Some(dl));
                simulate_book(&bp, &book, &BookConfig { sim: cfg.clone(), ..BookConfig::default() }).unwrap()
            } else {
                let (bp, book) = one_sleeve_book(panel, S3TestRule, Some(dl));
                simulate_book(&bp, &book, &BookConfig { sim: cfg.clone(), ..BookConfig::default() }).unwrap()
            };
            let t1 = got.one_sleeve_sim_result().unwrap();
            assert_eq!(t1.series_sha256, want.series_sha256, "{name} d={dl}: SERIES DIGEST");
            assert_eq!(t1.metrics(), want.metrics(), "{name} d={dl}: metrics");
            assert_eq!(bits(&t1.ret), bits(&want.ret), "{name} d={dl}: returns");
            assert_eq!(t1.window, want.window);
        }
    }
}

fn net(cfg: SimConfig) -> SimConfig {
    SimConfig { cost: NET, ..cfg }
}

#[test]
fn per_sleeve_delays_reproduce_the_book_level_delay_on_every_named_configuration() {
    // Every sleeve carrying `Some(k)` under book-level 0 is the same book as no sleeve value under book-level `k`, for the
    // certification, live, filter+budget, scaled+allocated, gross-cap and inverse-vol configurations.
    for c in cases().into_iter().filter(|c| !c.etf_only) {
        for k in [1usize, 2] {
            let (panel, book, cfg) = build_case(&c);
            let mut cfg_k = cfg.clone();
            cfg_k.sim.execution_delay_bars = k;
            let want = simulate_book_gross_and_net(&panel, &book, &cfg_k).unwrap();
            let got = simulate_book_gross_and_net(&panel, &with_delays(book.clone(), &[k, k]), &cfg).unwrap();
            assert_eq!(want.0.series_sha256, got.0.series_sha256, "{} k={k} gross", c.name);
            assert_eq!(want.1.series_sha256, got.1.series_sha256, "{} k={k} net", c.name);
            // and the delay is not vacuous: it changes the answer
            let base = simulate_book(&panel, &book, &cfg).unwrap().series_sha256;
            assert_ne!(base, want.1.series_sha256, "{} k={k}", c.name);
        }
    }
}

// ------------------------------------------------------------------------------------------- per-sleeve equality with solo books

/// Mixed delays used below: (ETF, crypto).
const DELAY_PAIRS: [(usize, usize); 5] = [(1, 0), (0, 1), (2, 1), (3, 0), (1, 2)];

fn solo_of(spec: &SleeveSpec) -> Book {
    Book::new(vec![SleeveSpec { share: ShareSpec::Fixed(spec.share.initial()), ..spec.clone() }])
}

#[test]
fn every_sleeve_of_a_mixed_delay_book_has_the_shadow_and_cadence_of_its_own_single_sleeve_book() {
    let mut checks = 0;
    for c in cases().into_iter().filter(|c| !c.etf_only) {
        let (panel, book0, cfg) = build_case(&c);
        for (de, dc) in DELAY_PAIRS {
            let book = with_delays(book0.clone(), &[de, dc]);
            let (g, n) = simulate_book_gross_and_net(&panel, &book, &cfg).unwrap();
            for (s, spec) in book.sleeves.iter().enumerate() {
                let (sg, sn) = simulate_book_gross_and_net(&panel, &solo_of(spec), &cfg).unwrap();
                for (what, mixed, alone) in [("gross", &g, &sg), ("net", &n, &sn)] {
                    let ctx = format!("{} ({de},{dc}) sleeve {} {what}", c.name, spec.id);
                    // shadows: the sleeve-alone unit-capital accounts, bit for bit
                    assert_eq!(mixed.shadows[s].dates, alone.shadows[0].dates, "{ctx}: shadow dates");
                    assert_eq!(bits(&mixed.shadows[s].gross), bits(&alone.shadows[0].gross), "{ctx}: shadow gross");
                    assert_eq!(bits(&mixed.shadows[s].cost), bits(&alone.shadows[0].cost), "{ctx}: shadow cost");
                    assert_eq!(mixed.n_bars(), alone.n_bars());
                    let s_n = mixed.n_sleeves();
                    for k in 0..mixed.n_bars() {
                        let i = k * s_n + s;
                        assert_eq!(mixed.decision[i], alone.decision[k], "{ctx}: decision bar {k}");
                        assert_eq!(mixed.rule_refused[i], alone.rule_refused[k], "{ctx}: rule_refused bar {k}");
                        assert_eq!(mixed.sleeve_open[i], alone.sleeve_open[k], "{ctx}: sleeve_open bar {k}");
                        assert_eq!(mixed.due[i], alone.due[k], "{ctx}: due bar {k}");
                        if c.cadence == BookCadence::PerSleeve {
                            assert_eq!(mixed.planned[i], alone.planned[k], "{ctx}: planned bar {k}");
                        }
                    }
                    checks += 1;
                }
                // the sleeve's own-calendar metrics are the solo book's, and the delay moved them
                assert_eq!(n.sleeve_metrics(s, false), sn.sleeve_metrics(0, false));
            }
            // attribution identity survives mixed delays, gross and net
            for r in [&g, &n] {
                let a = r.attribution_report();
                assert!(
                    a.max_abs_err_total <= 1e-12 && a.max_abs_err_contrib_sum <= 1e-12,
                    "{} ({de},{dc}): {a:?}",
                    c.name
                );
            }
        }
    }
    assert_eq!(checks, 6 * DELAY_PAIRS.len() * 2 * 2);
}

#[test]
fn one_sleeves_delay_never_moves_the_other_sleeves_shadow_or_cadence() {
    let c = case("book_live_60_40");
    let (panel, book0, cfg) = build_case(&c);
    let run = |de: usize, dc: usize| simulate_book(&panel, &with_delays(book0.clone(), &[de, dc]), &cfg).unwrap();
    let base = run(0, 0);
    for de in [1usize, 2, 3, 5] {
        let r = run(de, 0);
        // crypto (sleeve 1) untouched in its own series
        assert_eq!(bits(&r.shadows[1].gross), bits(&base.shadows[1].gross), "etf d={de}: crypto shadow gross");
        assert_eq!(bits(&r.shadows[1].cost), bits(&base.shadows[1].cost), "etf d={de}: crypto shadow cost");
        for k in 0..r.n_bars() {
            assert_eq!(r.due[k * 2 + 1], base.due[k * 2 + 1]);
            assert_eq!(r.decision[k * 2 + 1], base.decision[k * 2 + 1]);
        }
        // ... while the ETF sleeve's own series did move
        assert_ne!(bits(&r.shadows[0].gross), bits(&base.shadows[0].gross), "etf d={de}");
    }
    for dc in [1usize, 2, 5] {
        let r = run(0, dc);
        assert_eq!(bits(&r.shadows[0].gross), bits(&base.shadows[0].gross), "crypto d={dc}: ETF shadow gross");
        for k in 0..r.n_bars() {
            assert_eq!(r.due[k * 2], base.due[k * 2], "crypto d={dc}: ETF due bar {k}");
        }
        assert_ne!(bits(&r.shadows[1].gross), bits(&base.shadows[1].gross), "crypto d={dc}");
    }
}

// ------------------------------------------------------------------------------------------- independent oracle

/// ORACLE: wraps a rule so that a decision taken at own bar `t` is EMITTED at own bar `t + delay`, as a decision of a daily
/// sleeve that the simulator then executes with delay 0. The wrapper learns which own bars are decision bars from a set of
/// dates computed by the test from the panel (a causal `HistoryView` cannot know whether the current bar is a month-end).
/// The queue lives in a mutex, not in the rule state, because a refusal rolls the state back and "nothing due" is a refusal.
struct Delayed<R: WeightRule> {
    inner: R,
    universe: &'static [&'static str],
    delay: usize,
    decision_dates: HashSet<Date>,
    queue: Mutex<Vec<(usize, Vec<f64>)>>,
}

impl<R: WeightRule> StatefulRule for Delayed<R> {
    type State = ();
    fn id(&self) -> &'static str {
        "delayed_oracle"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        self.universe
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.inner.rebalance_policy()
    }
    fn min_history_bars(&self) -> usize {
        self.inner.min_history_bars()
    }
    fn data_need(&self) -> DataNeed {
        DataNeed::CompleteJointCalendar
    }
    fn vol_scaling(&self) -> VolScaling {
        VolScaling::None
    }
    fn init(&self) {
        // one queue per RUN: the same book value is used for the gross and the net execution
        self.queue.lock().unwrap().clear();
    }
    fn step(&self, _st: &mut (), h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let t = h.len() - 1;
        let mut q = self.queue.lock().unwrap();
        if self.decision_dates.contains(&h.date()) {
            if let Ok(w) = self.inner.target_weights(h) {
                q.push((t + self.delay, w));
            }
        }
        match q.first() {
            Some((due, _)) if *due == t => Ok(q.remove(0).1),
            _ => Err(RuleRefusal::data("nothing_due", "no decision reaches its execution bar on this bar")),
        }
    }
}

fn own_dates(panel: &BookPanel, universe: &[usize]) -> Vec<Date> {
    panel.sleeve_calendar(universe).unwrap().panel.dates().to_vec()
}

fn last_bars_of_months(dates: &[Date]) -> HashSet<Date> {
    (0..dates.len()).filter(|&i| i + 1 == dates.len() || !dates[i].same_month(dates[i + 1])).map(|i| dates[i]).collect()
}

/// The book of a named case with each sleeve wrapped in [`Delayed`]: ETF delayed `de` own bars, crypto `dc`, run at delay 0.
fn oracle_book(panel: &BookPanel, c: &Case, de: usize, dc: usize) -> Book {
    let etf_idx = instrument_index(panel, &E);
    let cry_idx = instrument_index(panel, &C);
    let alloc = c.invvol_lookback.map(|l| AllocatorSpec::InverseVol { lookback_bars: l, total: 1.0 });
    let share = |v: f64| if alloc.is_some() { ShareSpec::Allocated { initial: v } } else { ShareSpec::Fixed(v) };
    let etf_dates = own_dates(panel, &etf_idx);
    let cry_dates = own_dates(panel, &cry_idx);
    let etf = Delayed {
        inner: BookEtfRule,
        universe: &E,
        delay: de,
        decision_dates: last_bars_of_months(&etf_dates),
        queue: Mutex::new(Vec::new()),
    };
    let cry = Delayed {
        inner: BookCryRule,
        universe: &C,
        delay: dc,
        decision_dates: cry_dates.iter().copied().collect(),
        queue: Mutex::new(Vec::new()),
    };
    let mut b = Book::new(vec![
        SleeveSpec::from_stateful("etf", etf, etf_idx, share(c.shares.0)),
        SleeveSpec::from_stateful("cry", cry, cry_idx, share(c.shares.1)),
    ]);
    if let Some(a) = alloc {
        b = b.with_allocator(a);
    }
    b
}

/// Every column except the rule's own decision bookkeeping (`decision`, `rule_refused`, the rule refusals): those are dated
/// on the decision bar in one run and on the execution bar in the other by construction.
fn assert_same_but_rule_bookkeeping(a: &BookResult, b: &BookResult, what: &str) {
    assert_eq!(a.n_bars(), b.n_bars(), "{what}");
    let last = *a.clock_index.last().unwrap();
    let bad: Vec<_> = compare_book_prefix(a, b, last)
        .into_iter()
        .filter(|m| !matches!(m.field, "decision" | "rule_refused" | "refusals"))
        .collect();
    assert!(bad.is_empty(), "{what}: {:?}", &bad[..bad.len().min(6)]);
    let cap =
        |r: &BookResult| r.refusals_with_code("gross_above_cap").iter().map(|x| (x.bar, x.sleeve)).collect::<Vec<_>>();
    assert_eq!(cap(a), cap(b), "{what}: gross-cap refusals");
    assert_eq!(bits(&a.ret), bits(&b.ret), "{what}");
    assert_eq!(a.window, b.window, "{what}: counted window");
}

#[test]
fn a_mixed_delay_book_equals_the_same_book_with_delayed_decision_streams_at_delay_zero() {
    let mut compared = 0;
    let mut cap_refusals = 0;
    let mut skipped_trades = 0;
    for c in cases().into_iter().filter(|c| !c.etf_only) {
        let (panel, book0, cfg) = build_case(&c);
        for (de, dc) in DELAY_PAIRS {
            let real = simulate_book_gross_and_net(&panel, &with_delays(book0.clone(), &[de, dc]), &cfg).unwrap();
            let oracle = simulate_book_gross_and_net(&panel, &oracle_book(&panel, &c, de, dc), &cfg).unwrap();
            for (what, x, y) in [("gross", &real.0, &oracle.0), ("net", &real.1, &oracle.1)] {
                assert_same_but_rule_bookkeeping(x, y, &format!("{} ({de},{dc}) {what}", c.name));
                compared += 1;
            }
            cap_refusals += real.1.refusals_with_code("gross_above_cap").len();
            skipped_trades +=
                (0..real.1.n_bars()).filter(|&k| real.1.run[k] && real.1.traded_notional[k] == 0.0).count();
            // not vacuous: the delays changed the run relative to no delay
            let plain = simulate_book(&panel, &book0, &cfg).unwrap();
            assert_ne!(plain.series_sha256, real.1.series_sha256, "{} ({de},{dc})", c.name);
        }
    }
    assert_eq!(compared, 6 * DELAY_PAIRS.len() * 2);
    assert!(cap_refusals > 20, "the gross-cap case must refuse under delay ({cap_refusals})");
    assert!(skipped_trades > 50, "the trade filter must be exercised under delay ({skipped_trades})");
}

#[test]
fn the_oracle_itself_is_sensitive_to_the_delay_it_models() {
    // If the wrapper ignored its delay the equality test above would still pass against a broken simulator that also ignores
    // it; this pins the wrapper: (1, 0) and (0, 0) oracles differ, and the (0, 0) oracle equals the undelayed real book.
    let c = case("book_live_60_40");
    let (panel, book0, cfg) = build_case(&c);
    let plain = simulate_book(&panel, &book0, &cfg).unwrap();
    let o00 = simulate_book(&panel, &oracle_book(&panel, &c, 0, 0), &cfg).unwrap();
    assert_same_but_rule_bookkeeping(&plain, &o00, "oracle (0,0) vs undelayed");
    let o10 = simulate_book(&panel, &oracle_book(&panel, &c, 1, 0), &cfg).unwrap();
    assert_ne!(bits(&o10.ret), bits(&o00.ret));
}

// ------------------------------------------------------------------------------------------- harnesses with delays

fn account_start_index(panel: &BookPanel) -> usize {
    let start = BarTime::from_date(d(&meta("start_bar")));
    panel.times().iter().position(|t| *t >= start).unwrap()
}

fn cuts(panel: &BookPanel) -> Vec<usize> {
    let a0 = account_start_index(panel);
    let find = |s: &str| panel.times().iter().position(|t| t.date() == d(s)).unwrap();
    let mut v = vec![
        a0,
        a0 + 1,
        a0 + 2,
        a0 + 4,
        find("2019-03-31"),
        find("2019-04-29"),
        find("2019-04-30"),
        find("2019-05-01"),
        find("2019-05-31"),
        find("2019-06-14"),
        panel.n_bars() - 2,
    ];
    v.sort();
    v.dedup();
    v
}

#[test]
fn poisoning_and_truncation_are_clean_with_per_sleeve_delays() {
    let mut checks = 0;
    for name in ["book_cert_60_40", "book_live_60_40", "book_scaled_50_30", "book_grosscap_60_40", "book_invvol"] {
        let c = case(name);
        let (panel, _, cfg) = build_case(&c);
        for (de, dc) in [(1usize, 0usize), (2, 1), (0, 3)] {
            let make = |p: &BookPanel| with_delays(book_for_case(p, &c), &[de, dc]);
            for cut in cuts(&panel) {
                let rep = check_book_poisoning(&make, &panel, &cfg, cut, 7).unwrap();
                assert!(
                    rep.is_clean(),
                    "poison {name} ({de},{dc}) cut {cut}: {:?}",
                    &rep.mismatches[..rep.mismatches.len().min(5)]
                );
                let rep = check_book_truncation(&make, &panel, &cfg, cut).unwrap();
                assert!(
                    rep.is_clean(),
                    "truncate {name} ({de},{dc}) cut {cut}: {:?}",
                    &rep.mismatches[..rep.mismatches.len().min(5)]
                );
                assert!(
                    rep.compared_through_clock_bar + 1 >= cut,
                    "at most the one C7 row is excluded: {name} cut {cut}"
                );
                checks += 2;
            }
        }
    }
    assert!(checks >= 5 * 3 * 8 * 2);
}

/// A crypto-like sleeve rule that (illegitimately) reads the close of `C1` `ahead` union bars in the FUTURE through the panel it
/// was built from (the crypto calendar is the union clock, so `ahead` bars are `ahead` own bars).
struct LeakyCry {
    panel: BookPanel,
    ahead: usize,
}

impl WeightRule for LeakyCry {
    fn id(&self) -> &'static str {
        "leaky_cry"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &C
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let u = self.panel.times().iter().position(|t| t.date() == h.date()).unwrap();
        let i = self.panel.instrument_index("C1").unwrap();
        let up = match (self.panel.close(i).get(u + self.ahead).copied().flatten(), self.panel.close(i)[u]) {
            (Some(future), Some(now)) => future > now,
            _ => false,
        };
        Ok(vec![if up { 0.5 } else { 0.0 }, 0.0])
    }
}

fn leaky_book(p: &BookPanel, ahead: usize, delay: usize) -> Book {
    Book::new(vec![
        SleeveSpec::from_rule("etf", BookEtfRule, instrument_index(p, &E), ShareSpec::Fixed(0.6)),
        SleeveSpec::from_rule(
            "cry",
            LeakyCry { panel: p.clone(), ahead },
            instrument_index(p, &C),
            ShareSpec::Fixed(0.4),
        )
        .with_execution_delay(delay),
    ])
}

#[test]
fn a_leak_beyond_the_delay_is_caught_and_a_peek_no_further_than_the_delay_is_invisible_by_construction() {
    let panel = key_book_panel();
    let mut cfg = config_for_case(&case("book_cert_60_40"));
    cfg.cadence = BookCadence::PerSleeve;
    let c1 = panel.instrument_index("C1").unwrap();
    let a0 = account_start_index(&panel);
    // a decision bar t whose close two bars later is higher: with the cut at t + 1 the leaky decision is executed on the cut
    let t = (a0 + 12..panel.n_bars() - 5)
        .find(|&t| matches!((panel.close(c1)[t], panel.close(c1)[t + 2]), (Some(a), Some(b)) if b > a))
        .expect("a rising crypto stretch exists in the window");
    let cut = t + 1;

    // ahead = 2 with delay 1: the decision of bar t read bar t + 2 and is executed at bar t + 1: inside the compared prefix
    let leak2 = |p: &BookPanel| leaky_book(p, 2, 1);
    let mut caught = 0;
    for seed in 1..=8u64 {
        if !check_book_poisoning(&leak2, &panel, &cfg, cut, seed).unwrap().is_clean() {
            caught += 1;
        }
    }
    assert!(caught >= 2, "poisoning must catch a delayed sleeve that reads two bars ahead ({caught} of 8 seeds)");
    // truncation compares through the cut, except the one C7 row of a truncated ETF calendar: so cut at ETF month-ends, where
    // nothing is excluded, and require that the leak (a decision reading a close beyond the cut) shows on at least one of them
    let month_ends: Vec<usize> = ["2019-04-30", "2019-05-31"]
        .iter()
        .map(|s| panel.times().iter().position(|t| t.date() == d(s)).unwrap())
        .collect();
    let truncation_caught =
        month_ends.iter().filter(|&&m| !check_book_truncation(&leak2, &panel, &cfg, m).unwrap().is_clean()).count();
    assert!(truncation_caught >= 1, "truncation catches a delayed two-bars-ahead peek at an ETF month-end cut");

    // ahead = 1 with delay 1: the peeked close is the one on which the decision is executed anyway, so nothing that lands
    // inside the compared prefix depends on the future: NOT a leak of the executed strategy, and the harness (correctly, and
    // by design) reports clean. Documented so nobody reads "clean" as "the rule never peeks".
    let leak1 = |p: &BookPanel| leaky_book(p, 1, 1);
    for seed in 1..=8u64 {
        assert!(check_book_poisoning(&leak1, &panel, &cfg, cut, seed).unwrap().is_clean(), "seed {seed}");
    }
    // ... while the same one-bar peek WITHOUT a delay is a leak and is caught
    let leak1_d0 = |p: &BookPanel| leaky_book(p, 1, 0);
    let rising = (a0 + 12..panel.n_bars() - 4)
        .find(|&u| matches!((panel.close(c1)[u], panel.close(c1)[u + 1]), (Some(a), Some(b)) if b > a))
        .unwrap();
    let caught0 =
        (1..=8u64).filter(|&s| !check_book_poisoning(&leak1_d0, &panel, &cfg, rising, s).unwrap().is_clean()).count();
    assert!(caught0 >= 2, "a one-bar peek with no delay is a leak ({caught0} of 8 seeds)");
    // and the honest delayed book on the same harness is clean
    let honest = |p: &BookPanel| with_delays(book_for_case(p, &case("book_cert_60_40")), &[1, 0]);
    assert!(check_book_poisoning(&honest, &panel, &cfg, cut, 3).unwrap().is_clean());
}

#[test]
fn delayed_books_are_deterministic_repeated_and_concurrent_and_the_mixed_book_digest_is_pinned() {
    let c = case("book_live_60_40");
    let (panel, _, cfg) = build_case(&c);
    let make = |p: &BookPanel| with_delays(book_for_case(p, &c), &[1, 0]);
    assert!(check_book_determinism(&make, &panel, &cfg, 5).unwrap().is_empty());
    let want = simulate_book(&panel, &make(&panel), &cfg).unwrap().series_sha256;
    let digests: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> =
            (0..4).map(|_| s.spawn(|| simulate_book(&panel, &make(&panel), &cfg).unwrap().series_sha256)).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(digests.iter().all(|x| *x == want));
    println!("DIGEST book_live_60_40 net ETF d=1 crypto d=0: {want}");
    assert_eq!(want, GOLDEN_LIVE_ETF1_CRY0_NET);
    // and it is not the undelayed digest of the same book (which is pinned in book_causality.rs)
    assert_ne!(want, "b4419673fd83d693dd6f5f2760598e87352d7a2cb6460ceb502e2ebbf4282924");
}

/// GOLDEN digest of the live-realistic mixed book on the committed synthetic fixture: `book_live_60_40` with the ETF sleeve at
/// delay 1 and the crypto sleeve at delay 0 (net run). Produced on aarch64 Linux; IEEE-exact arithmetic only.
const GOLDEN_LIVE_ETF1_CRY0_NET: &str = "7f903aee1e7b5ebae76636317020cf18ea3d271c73f99f45c66a44bc974d7953";

// ------------------------------------------------------------------------------------------- delay sensitivity (W1)

#[test]
fn the_standard_delays_are_the_councils_table() {
    assert_eq!(STANDARD_DELAYS, [0, 1, 2, 3, 5]);
}

#[test]
fn aligned_correlation_is_pearson_over_the_common_dates() {
    let da = [d("2020-01-01"), d("2020-01-02"), d("2020-01-03"), d("2020-01-06"), d("2020-01-07")];
    let ra = [1.0, 2.0, 3.0, 4.0, 5.0];
    // perfectly correlated on the three common dates, other dates ignored
    let db = [d("2020-01-02"), d("2020-01-03"), d("2020-01-04"), d("2020-01-07")];
    let rb = [20.0, 30.0, 999.0, 50.0];
    assert!((aligned_correlation(&da, &ra, &db, &rb) - 1.0).abs() < 1e-15);
    // hand-computed: x = [1, 2, 3, 4], y = [2, 1, 4, 3]: mean 2.5 both; sxx = 5, syy = 5, sxy = (-1.5*-0.5)+(-0.5*-1.5)+(0.5*1.5)+(1.5*0.5) = 3.0; r = 0.6
    let dx = [d("2020-01-01"), d("2020-01-02"), d("2020-01-03"), d("2020-01-06")];
    assert!((aligned_correlation(&dx, &[1.0, 2.0, 3.0, 4.0], &dx, &[2.0, 1.0, 4.0, 3.0]) - 0.6).abs() < 1e-15);
    assert!((aligned_correlation(&dx, &[1.0, 2.0, 3.0, 4.0], &dx, &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-15);
    // degenerate inputs are NaN, never a number
    assert!(aligned_correlation(&dx, &[1.0, 1.0, 1.0, 1.0], &dx, &[1.0, 2.0, 3.0, 4.0]).is_nan(), "constant series");
    assert!(aligned_correlation(&dx[..1], &[1.0], &dx[..1], &[1.0]).is_nan(), "one common date");
    let far = [d("2021-01-01"), d("2021-01-02")];
    assert!(aligned_correlation(&dx, &[1.0, 2.0, 3.0, 4.0], &far, &[1.0, 2.0]).is_nan(), "no common date");
}

/// The table of a one-sleeve book equals `simulate` at every delay, row by row (gross and net metrics), and the baseline
/// correlation is 1.
fn check_single_sleeve_table<R: WeightRule + 'static>(
    name: &str,
    panel: &Panel,
    rule: impl Fn() -> R,
    cfg: &SimConfig,
) -> Vec<DelayRow> {
    let (bp, book) = one_sleeve_book(panel, rule(), None);
    let bcfg = BookConfig { sim: cfg.clone(), ..BookConfig::default() };
    let rows = delay_sensitivity(&bp, &book, &bcfg, DelayScope::Book, &STANDARD_DELAYS).unwrap();
    assert_eq!(rows.iter().map(|r| r.delay).collect::<Vec<_>>(), STANDARD_DELAYS);
    for r in &rows {
        let net = simulate(panel, &rule(), &SimConfig { execution_delay_bars: r.delay, ..cfg.clone() }).unwrap();
        let gross = simulate(
            panel,
            &rule(),
            &SimConfig {
                execution_delay_bars: r.delay,
                cost: CostModel::ZERO,
                financing: Financing::None,
                ..cfg.clone()
            },
        )
        .unwrap();
        assert_eq!(r.net, net.metrics(), "{name} d={} net metrics vs simulate", r.delay);
        assert_eq!(r.gross, gross.metrics(), "{name} d={} gross metrics vs simulate", r.delay);
        assert_eq!(r.total_cost, net.total_cost(), "{name} d={}", r.delay);
    }
    assert!((rows[0].corr_net_to_baseline - 1.0).abs() < 1e-12, "the baseline correlates with itself");
    for r in &rows[1..] {
        assert!(
            r.corr_net_to_baseline < 1.0 && r.corr_net_to_baseline > -1.0,
            "{name} d={}: {}",
            r.delay,
            r.corr_net_to_baseline
        );
    }
    // the rows differ from each other (a delay is not a no-op): distinct digests
    let mut digs: Vec<&str> = rows.iter().map(|r| r.series_sha256.as_str()).collect();
    digs.sort();
    digs.dedup();
    assert_eq!(digs.len(), rows.len(), "{name}: every delay is a different run");
    println!("DELAY SENSITIVITY {name} (synthetic fixture)\n{}", format_delay_table(&rows));
    rows
}

#[test]
fn the_delay_sensitivity_table_of_the_s1_and_s3_fixtures_equals_simulate_at_each_delay() {
    let s1 = check_single_sleeve_table("S1 ETF", &s1_panel(), || S1TestRule, &net(s1_config()));
    let s3 = check_single_sleeve_table("S3 crypto", &s3_panel(), || S3TestRule, &net(s3_config()));
    // the daily crypto sleeve is more exposed to a one-day delay than the monthly ETF one is to a one-session delay in
    // terms of return correlation with the undelayed run (a structural property of the fixtures, not a claim about data)
    assert!(s1[1].corr_net_to_baseline.is_finite() && s3[1].corr_net_to_baseline.is_finite());
    let table = format_delay_table(&s3);
    assert_eq!(table.lines().count(), 1 + STANDARD_DELAYS.len());
    assert!(
        table.lines().next().unwrap().contains("net_sharpe") && table.lines().next().unwrap().contains("corr_to_base")
    );
}

#[test]
fn book_scope_overrides_every_sleeve_and_sleeve_scope_touches_one() {
    let c = case("book_live_60_40");
    let (panel, book0, cfg) = build_case(&c);
    // sleeve 0 = ETF already carries Some(3); Book scope replaces it
    let preset = with_delays(book0.clone(), &[3, 2]);
    let rows = delay_sensitivity(&panel, &preset, &cfg, DelayScope::Book, &[0, 1]).unwrap();
    assert_eq!(
        rows[0].series_sha256,
        digest(&panel, &book0, &cfg),
        "Book d=0 is the undelayed book, whatever was preset"
    );
    assert_eq!(rows[1].series_sha256, digest(&panel, &with_delays(book0.clone(), &[1, 1]), &cfg));
    // Sleeve(0) at d: the ETF moves, the crypto sleeve keeps what it had (its own Some(2) here)
    let rows = delay_sensitivity(&panel, &preset, &cfg, DelayScope::Sleeve(0), &[0, 1, 5]).unwrap();
    for (row, dl) in rows.iter().zip([0usize, 1, 5]) {
        assert_eq!(row.series_sha256, digest(&panel, &with_delays(book0.clone(), &[dl, 2]), &cfg), "Sleeve(0) d={dl}");
    }
    // Sleeve(1) on an unset book: the ETF keeps the book-level default
    let mut cfg1 = cfg.clone();
    cfg1.sim.execution_delay_bars = 1;
    let rows = delay_sensitivity(&panel, &book0, &cfg1, DelayScope::Sleeve(1), &[0, 2]).unwrap();
    assert_eq!(rows[0].series_sha256, digest(&panel, &with_delays(book0.clone(), &[1, 0]), &cfg));
    assert_eq!(rows[1].series_sha256, digest(&panel, &with_delays(book0.clone(), &[1, 2]), &cfg));
    // the row for the undelayed sleeve carries a correlation of exactly 1 to itself, the delayed ones do not
    assert!((rows[0].corr_net_to_baseline - 1.0).abs() < 1e-12 && rows[1].corr_net_to_baseline < 1.0);
}

#[test]
fn delay_sensitivity_refuses_bad_requests_cleanly() {
    let c = case("book_cert_60_40");
    let (panel, book0, cfg) = build_case(&c);
    assert!(matches!(delay_sensitivity(&panel, &book0, &cfg, DelayScope::Book, &[]), Err(BookError::BadBook(_))));
    match delay_sensitivity(&panel, &book0, &cfg, DelayScope::Sleeve(7), &[0]).unwrap_err() {
        BookError::BadBook(m) => assert!(m.contains("sleeve 7") && m.contains("has 2"), "{m}"),
        other => panic!("{other:?}"),
    }
    // a delay larger than the data is refused, not silently flat
    assert!(matches!(
        delay_sensitivity(&panel, &book0, &cfg, DelayScope::Book, &[0, 1_000_000]),
        Err(BookError::BadBook(_))
    ));
    // an error in the middle of the table fails the whole table (no partial rows are returned)
    assert!(delay_sensitivity(&panel, &book0, &cfg, DelayScope::Book, &[0, 1, 1_000_000, 2]).is_err());
}
