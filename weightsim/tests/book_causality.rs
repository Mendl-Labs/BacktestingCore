//! PF1 causality and determinism for BOOKS (design 6.5 "causality" column): poisoning of ALL instruments after `T`
//! (outputs through `T` bit-identical, including the shadow curves, the allocator's shares, the contributions and the
//! overlay's state), truncation, a leaky rule that MUST be caught, repeated and multi-threaded determinism, and pinned
//! series digests.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]

mod common;

use common::book::*;
use common::*;
use std::sync::{Arc, Mutex};
use weightsim::*;

fn account_start_index(panel: &BookPanel) -> usize {
    let start = BarTime::from_date(d(&meta("start_bar")));
    panel.times().iter().position(|t| *t >= start).unwrap()
}

/// Cut points through the window: the start bar, the first bars, month-end review and decision bars, the middle, the end.
fn cuts(panel: &BookPanel) -> Vec<usize> {
    let a0 = account_start_index(panel);
    let find = |s: &str| panel.times().iter().position(|t| t.date() == d(s)).unwrap();
    let mut v = vec![
        a0,
        a0 + 1,
        a0 + 2,
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
fn poisoning_all_instruments_after_t_leaves_everything_through_t_bit_identical() {
    let mut checks = 0;
    for name in [
        "book_cert_60_40",
        "book_live_60_40",
        "book_scaled_50_30",
        "book_grosscap_60_40",
        "book_invvol",
        "book_due_filter_60_40",
    ] {
        let c = case(name);
        let (panel, _, cfg) = build_case(&c);
        let make = |p: &BookPanel| book_for_case(p, &c);
        for cut in cuts(&panel) {
            for seed in [1u64, 99] {
                let rep = check_book_poisoning(&make, &panel, &cfg, cut, seed).unwrap();
                assert!(
                    rep.is_clean(),
                    "{name} cut {cut} seed {seed}: {:?}",
                    &rep.mismatches[..rep.mismatches.len().min(5)]
                );
                checks += 1;
            }
        }
    }
    assert!(checks >= 6 * 8 * 2);
}

#[test]
fn poisoning_reaches_the_net_run_the_shadows_and_the_allocator() {
    // The comparison covers the columns that a look-ahead in an allocator or a shadow account would corrupt.
    let c = case("book_invvol");
    let (panel, book, mut cfg) = build_case(&c);
    cfg.sim.cost = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;
    let cut = panel.times().iter().position(|t| t.date() == d("2019-05-31")).unwrap();
    let clean = simulate_book(&panel, &book, &cfg).unwrap();
    let dirty_panel = poison_book_panel(&panel, cut + 1, 5);
    let dirty = simulate_book(&dirty_panel, &book_for_case(&dirty_panel, &c), &cfg).unwrap();
    assert!(compare_book_prefix(&clean, &dirty, cut).is_empty());
    // ... and the poisoning is real: after the cut the runs differ (so the comparison window is what makes the test pass)
    assert!(!compare_book_prefix(&clean, &dirty, panel.n_bars() - 1).is_empty());
    let k = clean.clock_index.iter().position(|&u| u == cut).unwrap();
    // shares at the cut bar are the ones the allocator computed from returns through the cut
    assert!(clean.share[k * 2] != 0.5, "the allocator had already re-estimated by 2019-05-31");
    assert_eq!(clean.share[k * 2].to_bits(), dirty.share[k * 2].to_bits());
}

#[test]
fn truncation_leaves_everything_through_the_cut_bit_identical() {
    for name in ["book_cert_60_40", "book_live_60_40", "book_invvol", "book_scaled_50_30"] {
        let c = case(name);
        let (panel, _, cfg) = build_case(&c);
        let make = |p: &BookPanel| book_for_case(p, &c);
        for cut in cuts(&panel) {
            let rep = check_book_truncation(&make, &panel, &cfg, cut).unwrap();
            assert!(rep.is_clean(), "{name} cut {cut}: {:?}", &rep.mismatches[..rep.mismatches.len().min(5)]);
            assert!(rep.compared_through_clock_bar + 1 >= cut, "at most the one C7 row is excluded: {name} cut {cut}");
        }
    }
}

/// A sleeve rule that (illegitimately) reads the FUTURE through the panel it was built from.
struct LeakyEtf {
    panel: BookPanel,
}

impl WeightRule for LeakyEtf {
    fn id(&self) -> &'static str {
        "leaky_etf"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &E
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
        let i = self.panel.instrument_index("E1").unwrap();
        let up = match (self.panel.close(i).get(u + 1).copied().flatten(), self.panel.close(i)[u]) {
            (Some(next), Some(now)) => next > now,
            _ => false,
        };
        Ok(vec![if up { 0.3 } else { 0.0 }, 0.0, 0.0])
    }
}

fn leaky_book(panel: &BookPanel) -> Book {
    Book::new(vec![
        SleeveSpec::from_rule(
            "etf",
            LeakyEtf { panel: panel.clone() },
            instrument_index(panel, &E),
            ShareSpec::Fixed(0.6),
        ),
        SleeveSpec::from_rule("cry", BookCryRule, instrument_index(panel, &C), ShareSpec::Fixed(0.4)),
    ])
}

#[test]
fn a_leaky_rule_is_caught_by_poisoning_and_by_truncation() {
    let panel = key_book_panel();
    let mut cfg = config_for_case(&case("book_cert_60_40"));
    cfg.cadence = BookCadence::PerSleeve;
    // a cut whose NEXT day the leaky rule can see going up (so the truncated run, which cannot see it, must differ)
    let e1 = panel.instrument_index("E1").unwrap();
    let a0 = account_start_index(&panel);
    let cut = (a0 + 5..panel.n_bars() - 2)
        .find(|&u| matches!((panel.close(e1)[u], panel.close(e1)[u + 1]), (Some(a), Some(b)) if b > a))
        .expect("a rising ETF day exists in the window");
    // (the poison shifts each instrument's level by a seed-dependent factor, so whether tomorrow's poisoned close is above
    // today's real one depends on the seed: the leak is caught for at least one of these seeds at this cut)
    let mut caught = Vec::new();
    for seed in 1..=8u64 {
        let rep = check_book_poisoning(&leaky_book, &panel, &cfg, cut, seed).unwrap();
        if !rep.is_clean() {
            assert!(
                rep.mismatches.iter().any(|m| m.field == "target_weights" || m.field == "ret"),
                "{:?}",
                &rep.mismatches[..rep.mismatches.len().min(5)]
            );
            caught.push(seed);
        }
    }
    assert!(caught.len() >= 2, "poisoning must catch a rule that reads tomorrow's close (caught for seeds {caught:?})");
    let rep = check_book_truncation(&leaky_book, &panel, &cfg, cut).unwrap();
    assert!(!rep.is_clean(), "truncation must catch it too");
    // and the honest rule on the same harness is clean (the harness itself is not trigger-happy)
    let honest = |p: &BookPanel| book_for_case(p, &case("book_cert_60_40"));
    assert!(check_book_poisoning(&honest, &panel, &cfg, cut, 3).unwrap().is_clean());
}

#[test]
fn a_look_ahead_in_the_overlay_input_would_be_caught_and_the_real_overlay_state_is_causal() {
    // an overlay is part of the run: its risk scale and halt flag are compared bit for bit through T
    struct Dd(Arc<Mutex<Vec<f64>>>);
    struct DdRun<'a>(&'a Dd, f64);
    impl Overlay for Dd {
        fn start(&self) -> Box<dyn OverlayRun + '_> {
            Box::new(DdRun(self, 1.0))
        }
    }
    impl OverlayRun for DdRun<'_> {
        fn step(&mut self, i: &OverlayInput) -> OverlayDecision {
            self.1 = self.1.max(i.equity);
            self.0 .0.lock().unwrap().push(i.equity);
            OverlayDecision { scale: if i.equity / self.1 < 0.995 { 0.5 } else { 1.0 }, halt: false }
        }
    }
    let c = case("book_live_60_40");
    let (panel, _, cfg) = build_case(&c);
    let log = Arc::new(Mutex::new(Vec::new()));
    let make = |p: &BookPanel| book_for_case(p, &c).with_overlay(Arc::new(Dd(log.clone())));
    for cut in cuts(&panel) {
        let rep = check_book_poisoning(&make, &panel, &cfg, cut, 8).unwrap();
        assert!(rep.is_clean(), "cut {cut}: {:?}", &rep.mismatches[..rep.mismatches.len().min(5)]);
    }
    let r = simulate_book(&panel, &make(&panel), &cfg).unwrap();
    assert!(r.risk_scale.contains(&0.5) && r.risk_scale.contains(&1.0), "the overlay was active and inactive");
}

// ------------------------------------------------------------------------------------------- determinism
fn net_run(name: &str) -> BookResult {
    let (panel, book, cfg) = build_case(&case(name));
    simulate_book(&panel, &book, &cfg).unwrap()
}

#[test]
fn repeated_runs_are_bit_identical() {
    for name in ["book_cert_60_40", "book_live_60_40", "book_invvol", "book_grosscap_60_40"] {
        let c = case(name);
        let (panel, _, cfg) = build_case(&c);
        let make = |p: &BookPanel| book_for_case(p, &c);
        assert!(check_book_determinism(&make, &panel, &cfg, 5).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn concurrent_runs_on_shared_data_give_the_same_digest() {
    let c = case("book_live_60_40");
    let (panel, _, cfg) = build_case(&c);
    let want = simulate_book(&panel, &book_for_case(&panel, &c), &cfg).unwrap().series_sha256;
    let digests: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..4)
            .map(|_| s.spawn(|| simulate_book(&panel, &book_for_case(&panel, &c), &cfg).unwrap().series_sha256))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(digests.iter().all(|x| *x == want), "{digests:?} vs {want}");
    // and several books over ONE shared panel and ONE shared rule set, concurrently (the Engine runs many books at once)
    let names = ["book_cert_60_40", "book_invvol", "book_grosscap_60_40", "book_scaled_50_30"];
    let serial: Vec<String> = names.iter().map(|n| net_run(n).series_sha256).collect();
    let parallel: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> = names.iter().map(|n| s.spawn(move || net_run(n).series_sha256)).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(serial, parallel);
}

/// GOLDEN digests for the committed synthetic book fixture (net runs). Produced on aarch64 Linux; the simulation path
/// uses only IEEE-exact operations (+ - * / sqrt), so an x86-64 build must reproduce them exactly. If this test fails on
/// one architecture only, cross-platform reproducibility is broken.
const GOLDEN_CERT_NET: &str = "148fe122fdef81eb445fa8602d8545f4945ffa33753ae24449ff46ec23c36c09";
const GOLDEN_LIVE_NET: &str = "b4419673fd83d693dd6f5f2760598e87352d7a2cb6460ceb502e2ebbf4282924";
const GOLDEN_INVVOL_NET: &str = "59df2a352821400287949361b903fb919f0e7acbc3270038ddcb2a5c78484dae";
const GOLDEN_SCALED_NET: &str = "faf2730726e9b7138e1d215cb18024e2e525b2f3132c5c851234d8ae04d84871";
const GOLDEN_NETTING_NET: &str = "65bd5f79fe5de044bf264630d13aa3d711d2e60ec51b7ad7fea6d0224357e67e";
const GOLDEN_S3_ONE_SLEEVE_T1_VIEW: &str = "2932738e542adbf8ca5097e3079249d03dd676b4e57e1f1a789502875fc97430";

#[test]
fn series_digests_match_the_pinned_goldens() {
    let cert = net_run("book_cert_60_40").series_sha256;
    let live = net_run("book_live_60_40").series_sha256;
    let inv = net_run("book_invvol").series_sha256;
    let scaled = net_run("book_scaled_50_30").series_sha256;
    let panel = netting_panel();
    let netting = simulate_book(&panel, &netting_book(&panel), &netting_config()).unwrap().series_sha256;
    // the one-sleeve T1 view of an S3 book is the T1 digest of `simulate` (bit-identity is asserted in book_identity.rs)
    let s3 = s3_panel();
    let bp = BookPanel::from_panel(&s3);
    let book = Book::new(vec![SleeveSpec::from_rule("s3", S3TestRule, vec![0, 1], ShareSpec::Fixed(1.0))]);
    let cfg = BookConfig {
        sim: SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..s3_config() },
        ..BookConfig::default()
    };
    let s3_t1 = simulate_book(&bp, &book, &cfg).unwrap().one_sleeve_sim_result().unwrap().series_sha256;
    println!("DIGEST book_cert_60_40 net   {cert}");
    println!("DIGEST book_live_60_40 net   {live}");
    println!("DIGEST book_invvol net       {inv}");
    println!("DIGEST book_scaled_50_30 net {scaled}");
    println!("DIGEST netting net           {netting}");
    println!("DIGEST S3 one-sleeve T1 view {s3_t1}");
    assert_eq!(cert, GOLDEN_CERT_NET);
    assert_eq!(live, GOLDEN_LIVE_NET);
    assert_eq!(inv, GOLDEN_INVVOL_NET);
    assert_eq!(scaled, GOLDEN_SCALED_NET);
    assert_eq!(netting, GOLDEN_NETTING_NET);
    assert_eq!(s3_t1, GOLDEN_S3_ONE_SLEEVE_T1_VIEW);
}

#[test]
fn digest_is_sensitive_to_prices_costs_shares_cadence_and_configuration() {
    let c = case("book_live_60_40");
    let (panel, book, cfg) = build_case(&c);
    let base = simulate_book(&panel, &book, &cfg).unwrap().series_sha256;
    assert_eq!(base.len(), 64);
    let cut = panel.times().iter().position(|t| t.date() == d("2019-06-10")).unwrap();
    // 1. one close moved by one ulp
    let poked =
        panel.with_prices_replaced_from(
            cut,
            |i, t, old| if i == 3 && t == cut { f64::from_bits(old.to_bits() + 1) } else { old },
        );
    assert_ne!(simulate_book(&poked, &book_for_case(&poked, &c), &cfg).unwrap().series_sha256, base);
    // 2. cost preset
    let mut c2 = cfg.clone();
    c2.sim.cost = CostModel::ZERO;
    assert_ne!(simulate_book(&panel, &book, &c2).unwrap().series_sha256, base);
    // 3. cadence
    let mut c3 = cfg.clone();
    c3.cadence = BookCadence::PerSleeve;
    assert_ne!(simulate_book(&panel, &book, &c3).unwrap().series_sha256, base);
    // 4. shares
    let other = Case { shares: (0.61, 0.39), ..c.clone() };
    assert_ne!(simulate_book(&panel, &book_for_case(&panel, &other), &cfg).unwrap().series_sha256, base);
    // 5. risk scale, filter, account start, sleeve id (the digest covers what the run PRODUCED, so a configuration change
    // that alters no output, such as a cash policy that never binds, correctly leaves it unchanged)
    let mut c5 = cfg.clone();
    c5.sim.risk_scale = 0.9;
    assert_ne!(simulate_book(&panel, &book, &c5).unwrap().series_sha256, base);
    let mut c6 = cfg.clone();
    c6.trade_filter = None;
    assert_ne!(simulate_book(&panel, &book, &c6).unwrap().series_sha256, base);
    let mut c8 = cfg.clone();
    c8.account_start = Some(BarTime::from_date(d("2019-04-01")));
    assert_ne!(simulate_book(&panel, &book, &c8).unwrap().series_sha256, base);
    let mut renamed = book.clone();
    renamed.sleeves[0].id = "etf2".into();
    assert_ne!(simulate_book(&panel, &renamed, &cfg).unwrap().series_sha256, base);
}
