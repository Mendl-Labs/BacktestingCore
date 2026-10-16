//! PF1 acceptance (a): `simulate_book` with ONE sleeve is bit-identical to `simulate` (the certified T1 API, which is
//! unchanged): every per-bar column, the refusals, the window, the flip counter, the exposure statistics, the metrics
//! and the SERIES DIGEST. On the synthetic S1/S3 fixtures always; on the real pinned ladder candles when
//! `WEIGHTSIM_LADDER_DIR` points at the `replication_ladder` directory (vendor data is not copied into this repo;
//! without the variable that test prints SKIPPED).

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::*;
use weightsim::*;

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Every field of two T1 results must match bit for bit.
pub fn assert_sim_identical(a: &SimResult, b: &SimResult, what: &str) {
    assert_eq!(a.rule_id, b.rule_id, "{what}: rule_id");
    assert_eq!(a.symbols, b.symbols, "{what}: symbols");
    assert_eq!(a.dates, b.dates, "{what}: dates");
    for (name, x, y) in [
        ("ret", &a.ret, &b.ret),
        ("ret_pre_cost", &a.ret_pre_cost, &b.ret_pre_cost),
        ("equity", &a.equity, &b.equity),
        ("cash", &a.cash, &b.cash),
        ("cost", &a.cost, &b.cost),
        ("traded_notional", &a.traded_notional, &b.traded_notional),
        ("financing", &a.financing, &b.financing),
        ("gross_exposure", &a.gross_exposure, &b.gross_exposure),
        ("net_exposure", &a.net_exposure, &b.net_exposure),
        ("target_weights", &a.target_weights, &b.target_weights),
        ("held_weights", &a.held_weights, &b.held_weights),
        ("units", &a.units, &b.units),
    ] {
        assert_eq!(bits(x), bits(y), "{what}: column {name}");
    }
    assert_eq!(a.decision, b.decision, "{what}: decision");
    assert_eq!(a.refused, b.refused, "{what}: refused");
    assert_eq!(a.refusals, b.refusals, "{what}: refusals");
    assert_eq!(a.window, b.window, "{what}: window");
    assert_eq!(a.signal_flips, b.signal_flips, "{what}: signal_flips");
    assert_eq!(a.fills_per_asset, b.fills_per_asset, "{what}: fills_per_asset");
    assert_eq!(a.rebalance_bars, b.rebalance_bars, "{what}: rebalance_bars");
    assert_eq!(a.gross_exposure_stats, b.gross_exposure_stats, "{what}: gross exposure stats");
    assert_eq!(a.net_exposure_stats, b.net_exposure_stats, "{what}: net exposure stats");
    assert_eq!(a.series_sha256, b.series_sha256, "{what}: SERIES DIGEST");
    assert_eq!(a.metrics(), b.metrics(), "{what}: metrics");
}

pub fn one_sleeve<R: WeightRule + 'static>(panel: &Panel, rule: R) -> (BookPanel, Book) {
    let bp = BookPanel::from_panel(panel);
    let n = panel.n_assets();
    let book = Book::new(vec![SleeveSpec::from_rule("only", rule, (0..n).collect(), ShareSpec::Fixed(1.0))]);
    (bp, book)
}

/// `simulate` and the one-sleeve `simulate_book` under the same `SimConfig`.
fn check<R: WeightRule + Clone + 'static>(panel: &Panel, rule: R, cfg: &SimConfig, what: &str) -> SimResult {
    let want = simulate(panel, &rule, cfg).unwrap_or_else(|e| panic!("{what}: simulate failed: {e}"));
    let (bp, book) = one_sleeve(panel, rule);
    let bcfg = BookConfig { sim: cfg.clone(), ..BookConfig::default() };
    let br = simulate_book(&bp, &book, &bcfg).unwrap_or_else(|e| panic!("{what}: simulate_book failed: {e}"));
    let got = br.one_sleeve_sim_result().expect("one-sleeve view");
    assert_sim_identical(&want, &got, what);
    want
}

fn net(cfg: SimConfig) -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..cfg }
}

/// `S1TestRule` is a unit struct without `Clone`; a local newtype keeps the generic `check` simple.
#[derive(Clone)]
pub struct S1TestRuleC;
impl WeightRule for S1TestRuleC {
    fn id(&self) -> &'static str {
        S1TestRule.id()
    }
    fn impl_version(&self) -> String {
        S1TestRule.impl_version()
    }
    fn universe(&self) -> &[&'static str] {
        &ETF
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
        S1TestRule.target_weights(h)
    }
}

#[derive(Clone)]
struct S3TestRuleC;
impl WeightRule for S3TestRuleC {
    fn id(&self) -> &'static str {
        S3TestRule.id()
    }
    fn impl_version(&self) -> String {
        S3TestRule.impl_version()
    }
    fn universe(&self) -> &[&'static str] {
        &CRY
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        100
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        S3TestRule.target_weights(h)
    }
}

#[test]
fn one_sleeve_s1_is_bit_identical_to_simulate_gross_and_net() {
    let p = s1_panel();
    let g = check(&p, S1TestRuleC, &s1_config(), "S1 gross");
    let n = check(&p, S1TestRuleC, &net(s1_config()), "S1 net 10bps");
    assert!(g.metrics().is_some() && n.metrics().is_some());
    assert!(g.rebalance_bars > 5, "the S1 fixture must actually rebalance (non-vacuous identity)");
    assert_ne!(g.series_sha256, n.series_sha256);
}

#[test]
fn one_sleeve_s3_is_bit_identical_to_simulate_gross_and_net() {
    let p = s3_panel();
    let g = check(&p, S3TestRuleC, &s3_config(), "S3 gross");
    let n = check(&p, S3TestRuleC, &net(s3_config()), "S3 net 10bps");
    assert!(g.rebalance_bars > 100 && n.total_cost() > 0.0);
}

#[test]
fn identity_holds_under_delay_risk_scale_financing_and_windows() {
    let p3 = s3_panel();
    for (what, cfg) in [
        ("delay 1", SimConfig { execution_delay_bars: 1, ..net(s3_config()) }),
        ("delay 2 + risk scale 0.5", SimConfig { execution_delay_bars: 2, risk_scale: 0.5, ..net(s3_config()) }),
        (
            "financing",
            SimConfig {
                financing: Financing::FlatAnnual { long_bps: 250.0, short_bps: 120.0, cash_bps: 80.0 },
                ..net(s3_config())
            },
        ),
        (
            "late start, early end",
            SimConfig { start: Some(d("2016-03-01")), end: Some(d("2016-08-31")), ..net(s3_config()) },
        ),
        ("initial equity 1e6", SimConfig { initial_equity: 1.0e6, ..net(s3_config()) }),
    ] {
        check(&p3, S3TestRuleC, &cfg, what);
    }
    let p1 = s1_panel();
    check(
        &p1,
        S1TestRuleC,
        &SimConfig { risk_scale: 0.7, execution_delay_bars: 1, ..net(s1_config()) },
        "S1 delay+scale",
    );
}

#[derive(Clone)]
struct Levered;
impl WeightRule for Levered {
    fn id(&self) -> &'static str {
        "levered"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["A", "B", "C"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        5
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let s = if (h.len() / 9) % 2 == 0 { 1.0 } else { -1.0 };
        Ok(vec![0.6 * s, -0.4, 0.3])
    }
}

#[test]
fn identity_holds_for_signed_levered_books_with_costs() {
    let panel = synth_panel(&["A", "B", "C"], 300, 42, "2019-01-01");
    let r = check(&panel, Levered, &net(SimConfig::default()), "levered net");
    assert!(r.gross_exposure_stats.unwrap().max > 1.0 || r.net_exposure_stats.is_some());
    // and with the max_gross cap NOT binding
    check(&panel, Levered, &SimConfig { max_gross: Some(1.5), ..net(SimConfig::default()) }, "levered cap not binding");
}

/// Refusing rule: warm-up refusals until bar 12, then a data refusal every 7th bar.
#[derive(Clone)]
struct Refuser(RebalancePolicy);
impl WeightRule for Refuser {
    fn id(&self) -> &'static str {
        "refuser"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["A", "B"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.0
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        if h.len() < 12 {
            return Err(RuleRefusal::warmup("warming up"));
        }
        if h.len() % 7 == 0 {
            return Err(RuleRefusal::data("stale", "stale bar"));
        }
        Ok(vec![0.5, if (h.len() / 5) % 2 == 0 { 0.3 } else { -0.3 }])
    }
}

#[test]
fn refusals_hold_the_whole_book_identically_under_both_policies() {
    let panel = synth_panel(&["A", "B"], 120, 7, "2019-03-01");
    for pol in [RebalancePolicy::OnDecision, RebalancePolicy::EveryBar] {
        let cfg = SimConfig { on_refusal: OnRefusal::HoldPrevious, ..net(SimConfig::default()) };
        let r = check(&panel, Refuser(pol), &cfg, "refuser hold");
        assert!(r.refusals.len() > 10, "refusals were recorded");
    }
    // Abort: the same error, same date and code, from both
    let panel = synth_panel(&["A", "B"], 120, 7, "2019-03-01");
    let want = simulate(&panel, &Refuser(RebalancePolicy::EveryBar), &SimConfig::default()).unwrap_err();
    let (bp, book) = one_sleeve(&panel, Refuser(RebalancePolicy::EveryBar));
    let got = simulate_book(&bp, &book, &BookConfig::default()).unwrap_err();
    assert_eq!(got, BookError::Sim(want));
}

#[test]
fn errors_are_the_same_errors() {
    let panel = synth_panel(&["A", "B", "C"], 60, 3, "2019-01-01");
    // gross cap breach: same date, same gross, same limit (delay 0)
    let cfg = SimConfig { max_gross: Some(0.5), ..SimConfig::default() };
    let want = simulate(&panel, &Levered, &cfg).unwrap_err();
    let (bp, book) = one_sleeve(&panel, Levered);
    let got = simulate_book(&bp, &book, &BookConfig { sim: cfg, ..BookConfig::default() }).unwrap_err();
    assert!(matches!(want, SimError::MaxGrossBreached { .. }));
    assert_eq!(got, BookError::Sim(want));
    // universe mismatch
    let bad =
        FnRule::new(&["A", "C", "B"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.1; 3]));
    let want = simulate(&panel, &bad, &SimConfig::default()).unwrap_err();
    let (bp, book) = one_sleeve(
        &panel,
        FnRule::new(&["A", "C", "B"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.1; 3])),
    );
    assert_eq!(simulate_book(&bp, &book, &BookConfig::default()).unwrap_err(), BookError::Sim(want));
    // invalid weights
    let short =
        FnRule::new(&["A", "B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.1; 2]));
    let want = simulate(&panel, &short, &SimConfig::default()).unwrap_err();
    let (bp, book) = one_sleeve(
        &panel,
        FnRule::new(&["A", "B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.1; 2])),
    );
    assert_eq!(simulate_book(&bp, &book, &BookConfig::default()).unwrap_err(), BookError::Sim(want));
    // bad config
    let bad_cfg = SimConfig { initial_equity: -1.0, ..SimConfig::default() };
    let want = simulate(&panel, &Levered, &bad_cfg).unwrap_err();
    let (bp, book) = one_sleeve(&panel, Levered);
    assert_eq!(
        simulate_book(&bp, &book, &BookConfig { sim: bad_cfg, ..BookConfig::default() }).unwrap_err(),
        BookError::Sim(want)
    );
}

#[test]
fn the_one_sleeve_view_exists_only_for_a_one_sleeve_full_universe_book() {
    let panel = synth_panel(&["A", "B", "C"], 60, 3, "2019-01-01");
    let bp = BookPanel::from_panel(&panel);
    // two sleeves
    let book2 = Book::new(vec![
        SleeveSpec::from_rule(
            "s0",
            FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.3])),
            vec![0],
            ShareSpec::Fixed(0.5),
        ),
        SleeveSpec::from_rule(
            "s1",
            FnRule::new(&["B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.3, 0.2])),
            vec![1, 2],
            ShareSpec::Fixed(0.5),
        ),
    ]);
    let r = simulate_book(&bp, &book2, &BookConfig::default()).unwrap();
    assert!(r.one_sleeve_sim_result().is_none());
    // one sleeve over a strict subset of the instruments
    let book1 = Book::new(vec![SleeveSpec::from_rule(
        "s0",
        FnRule::new(&["A"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |_| Ok(vec![0.3])),
        vec![0],
        ShareSpec::Fixed(1.0),
    )]);
    assert!(simulate_book(&bp, &book1, &BookConfig::default()).unwrap().one_sleeve_sim_result().is_none());
}

// ------------------------------------------------------------------------------------------- real pinned data
fn real_candles() -> Option<String> {
    let dir = match std::env::var("WEIGHTSIM_LADDER_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return None,
    };
    let text = std::fs::read_to_string(format!("{dir}/ladder_candles.csv"))
        .expect("ladder_candles.csv in WEIGHTSIM_LADDER_DIR");
    // the pinned T0 fixture: refuse any other file
    assert_eq!(
        sha256_hex(text.as_bytes()),
        "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365",
        "ladder_candles.csv is not the pinned T0 fixture"
    );
    Some(text)
}

#[test]
fn real_pinned_ladder_one_sleeve_books_are_bit_identical_to_simulate() {
    let text = match real_candles() {
        Some(t) => t,
        None => {
            println!("SKIPPED real_pinned_ladder_one_sleeve_books_are_bit_identical_to_simulate: set WEIGHTSIM_LADDER_DIR to the replication_ladder directory (vendor data is not copied into this repo)");
            return;
        }
    };
    let s1 = Panel::from_long_csv(&text, &ETF).unwrap();
    let s3 = Panel::from_long_csv(&text, &CRY).unwrap();
    let mut n_checked = 0;
    for (what, cfg) in [("S1 gross", s1_config()), ("S1 net", net(s1_config()))] {
        let r = check(&s1, S1TestRuleC, &cfg, what);
        println!("real {what}: {} bars, digest {}", r.dates.len(), r.series_sha256);
        n_checked += 1;
    }
    for (what, cfg) in [("S3 gross", s3_config()), ("S3 net", net(s3_config()))] {
        let r = check(&s3, S3TestRuleC, &cfg, what);
        println!("real {what}: {} bars, digest {}", r.dates.len(), r.series_sha256);
        n_checked += 1;
    }
    println!("REAL ONE-SLEEVE IDENTITY: {n_checked} runs bit-identical (all columns, refusals, window, flips, digest)");
}
