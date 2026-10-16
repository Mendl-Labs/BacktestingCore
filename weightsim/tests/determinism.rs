//! (e) Determinism: bit-identical repeated runs, identical results across threads, a pinned series digest (so the
//! ARM64 build here and the x86 CI runner can be compared: they must print and assert the same hex), and sensitivity
//! of the digest to every input that matters.

// Index loops mirror the closed-form oracles term by term; `% 2 == 0` regime toggles are clearer than is_multiple_of.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::harness::check_determinism;
use weightsim::*;

fn net_cfg(base: SimConfig) -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..base }
}

#[test]
fn repeated_runs_are_bit_identical() {
    let make_s3 = |_: &Panel| S3TestRule;
    let make_s1 = |_: &Panel| S1TestRule;
    assert!(check_determinism(&make_s3, &s3_panel(), &net_cfg(s3_config()), 5).unwrap().is_empty());
    assert!(check_determinism(&make_s1, &s1_panel(), &net_cfg(s1_config()), 5).unwrap().is_empty());
    // gross runs too
    assert!(check_determinism(&make_s3, &s3_panel(), &s3_config(), 3).unwrap().is_empty());
}

#[test]
fn concurrent_runs_on_shared_data_give_the_same_digest() {
    let panel = s3_panel();
    let cfg = net_cfg(s3_config());
    let want = simulate(&panel, &S3TestRule, &cfg).unwrap().series_sha256;
    let digests: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> =
            (0..4).map(|_| s.spawn(|| simulate(&panel, &S3TestRule, &cfg).unwrap().series_sha256)).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(digests.iter().all(|d| *d == want));
}

/// GOLDEN digests for the committed synthetic panel. They were produced on aarch64 Linux; the simulation path uses
/// only IEEE-exact operations (+ - * / sqrt), so an x86-64 build must reproduce them exactly. If this test fails on
/// one architecture only, cross-platform reproducibility is broken.
const GOLDEN_S1_GROSS: &str = "a6bae736c2f06218a77a473863f666abcb72d0ffa9d0bc540801f4bc2da4bd88";
const GOLDEN_S3_NET: &str = "2932738e542adbf8ca5097e3079249d03dd676b4e57e1f1a789502875fc97430";
const GOLDEN_SYNTH_LEVERED: &str = "c2b13ce18f43746602565f954f07a1dfa92000c16d1a9b9ea3102f5266c01ec6";

fn levered_run() -> SimResult {
    let panel = synth_panel(&["A", "B", "C"], 300, 42, "2019-01-01");
    let rule = FnRule::new(&["A", "B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 5, |h| {
        let s = if (h.len() / 9) % 2 == 0 { 1.0 } else { -1.0 };
        Ok(vec![0.6 * s, -0.4, 0.3])
    });
    simulate(&panel, &rule, &net_cfg(SimConfig::default())).unwrap()
}

#[test]
fn series_digests_match_the_pinned_goldens() {
    let s1 = simulate(&s1_panel(), &S1TestRule, &s1_config()).unwrap().series_sha256;
    let s3 = simulate(&s3_panel(), &S3TestRule, &net_cfg(s3_config())).unwrap().series_sha256;
    let lv = levered_run().series_sha256;
    println!("DIGEST S1 gross          {s1}");
    println!("DIGEST S3 net 10bps      {s3}");
    println!("DIGEST synthetic levered {lv}");
    assert_eq!(s1, GOLDEN_S1_GROSS);
    assert_eq!(s3, GOLDEN_S3_NET);
    assert_eq!(lv, GOLDEN_SYNTH_LEVERED);
}

#[test]
fn digest_is_sensitive_to_prices_costs_rules_and_config() {
    let panel = s3_panel();
    let base = simulate(&panel, &S3TestRule, &s3_config()).unwrap().series_sha256;
    // 1. one price moved by one ulp
    let poked =
        panel.with_prices_replaced_from(
            260,
            |i, t, old| if i == 0 && t == 260 { f64::from_bits(old.to_bits() + 1) } else { old },
        );
    assert_ne!(simulate(&poked, &S3TestRule, &s3_config()).unwrap().series_sha256, base);
    // 2. a different cost preset
    assert_ne!(simulate(&panel, &S3TestRule, &net_cfg(s3_config())).unwrap().series_sha256, base);
    // 3. a different rule id/version
    let other = FnRule::new(&CRY, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 100, |_| Ok(vec![0.5, 0.5]));
    assert_ne!(simulate(&panel, &other, &s3_config()).unwrap().series_sha256, base);
    // 4. a different execution delay
    assert_ne!(
        simulate(&panel, &S3TestRule, &SimConfig { execution_delay_bars: 1, ..s3_config() }).unwrap().series_sha256,
        base
    );
    // 5. a different risk scale
    assert_ne!(
        simulate(&panel, &S3TestRule, &SimConfig { risk_scale: 0.5, ..s3_config() }).unwrap().series_sha256,
        base
    );
    // and the digest is a 64-char lowercase hex string
    assert_eq!(base.len(), 64);
    assert!(base.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
}
