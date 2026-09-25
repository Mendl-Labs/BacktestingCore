//! Always-on negative controls: hand the per-sleeve certification a rule that is WRONG in a specific way and check
//! that exactly the checks that should notice do notice, while the honest rule passes every one. A certification is only
//! worth something if it can fail (Tier IV), and each of its causality checks has to be shown live.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::*;
use weightsim::*;
use weightsim_rules::ladder::mutants::MutantSleeve;
use weightsim_rules::ladder::{certify_sleeve, Check};
use weightsim_rules::{CryptoTrendRule, FlatUntil};

fn failed(checks: &[Check]) -> Vec<String> {
    checks.iter().filter(|c| !c.passed).map(|c| c.name.clone()).collect()
}

fn entry(fx: &weightsim_rules::ladder::Fixtures) -> Date {
    weightsim_rules::ladder::runner::entry_date(&fx.crypto_panel, &fx.s3).unwrap()
}

/// Weight 0.5 on a coin when the NEXT bar closes higher, read from its own copy of the panel. A blatant leak.
struct LeakS3 {
    full: Panel,
}

impl WeightRule for LeakS3 {
    fn id(&self) -> &'static str {
        "leak_s3"
    }
    fn impl_version(&self) -> String {
        "leak".into()
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
        let t = h.len() - 1;
        Ok((0..2)
            .map(|a| {
                let c = self.full.closes(a);
                if t + 1 < c.len() && c[t + 1] > c[t] {
                    0.5
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// Correct decisions, but returns cash on the LAST bar of the panel it was built from (the 'end-of-array' defect of
/// Amendment 10: behaviour that depends on whether later bars exist).
struct EndOfArray {
    total: usize,
}

impl WeightRule for EndOfArray {
    fn id(&self) -> &'static str {
        "end_of_array"
    }
    fn impl_version(&self) -> String {
        "eoa".into()
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
        if h.len() == self.total {
            return Ok(vec![0.0, 0.0]);
        }
        OracleS3.target_weights(h)
    }
}

/// Correct up to a drift that depends on how many times it has been called (shared state): not deterministic.
struct Flaky {
    calls: Arc<AtomicUsize>,
}

impl WeightRule for Flaky {
    fn id(&self) -> &'static str {
        "flaky"
    }
    fn impl_version(&self) -> String {
        "flaky".into()
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
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let w = OracleS3.target_weights(h)?;
        Ok(w.into_iter().map(|x| x * (1.0 + n as f64 * 1e-9)).collect())
    }
}

/// Causal and deterministic but wrong: 25% per coin instead of 50%.
struct Half;

impl WeightRule for Half {
    fn id(&self) -> &'static str {
        "half"
    }
    fn impl_version(&self) -> String {
        "half".into()
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
        Ok(OracleS3.target_weights(h)?.into_iter().map(|w| w * 0.5).collect())
    }
}

#[test]
fn the_honest_rule_passes_every_sleeve_check() {
    let fx = fixtures();
    let e = entry(&fx);
    let (rep, checks) =
        certify_sleeve(&fx, MutantSleeve::S3, &|_p: &Panel| FlatUntil::new(CryptoTrendRule, e)).unwrap();
    assert!(failed(&checks).is_empty(), "{:?}", failed(&checks));
    assert!(checks.len() >= 15);
    assert_eq!(rep.code, "S3");
    // The independent oracle rule certifies too (it is what the key was made from).
    let (_, checks) = certify_sleeve(&fx, MutantSleeve::S3, &|_p: &Panel| FlatUntil::new(OracleS3, e)).unwrap();
    assert!(failed(&checks).is_empty(), "{:?}", failed(&checks));
}

#[test]
fn a_leaky_rule_is_caught_by_poisoning_truncation_and_the_identity_tiers() {
    let fx = fixtures();
    let e = entry(&fx);
    let (rep, checks) =
        certify_sleeve(&fx, MutantSleeve::S3, &|p: &Panel| FlatUntil::new(LeakS3 { full: p.clone() }, e)).unwrap();
    let f = failed(&checks);
    for name in
        ["S3.causality.poisoning", "S3.causality.truncation", "S3.gross.tier2.identity", "S3.net.tier2.identity"]
    {
        assert!(f.contains(&name.to_string()), "{name} should fail, got {f:?}");
    }
    assert!(rep.poisoning_mismatches > 0 && rep.truncation_disagreements > 0);
    // a leak this blatant also blows the Tier I bands (it earns the future)
    assert!(f.contains(&"S3.gross.tier1.bands".to_string()), "{f:?}");
}

#[test]
fn an_end_of_array_rule_is_caught_by_truncation_but_not_by_poisoning() {
    let fx = fixtures();
    let e = entry(&fx);
    let n = fx.crypto_panel.n_bars();
    let (_, checks) = certify_sleeve(&fx, MutantSleeve::S3, &|p: &Panel| {
        // built from the panel it will see: a truncated panel gives a different `total`
        assert!(p.n_bars() <= n);
        FlatUntil::new(EndOfArray { total: p.n_bars() }, e)
    })
    .unwrap();
    let f = failed(&checks);
    assert!(f.contains(&"S3.causality.truncation".to_string()), "{f:?}");
    assert!(!f.contains(&"S3.causality.poisoning".to_string()), "poisoning changes prices, not lengths: {f:?}");
}

#[test]
fn a_nondeterministic_rule_is_caught_by_the_determinism_check() {
    let fx = fixtures();
    let e = entry(&fx);
    // the call counter is shared by every rule the factory builds, so the second execution sees different answers
    let calls = Arc::new(AtomicUsize::new(0));
    let (rep, checks) =
        certify_sleeve(&fx, MutantSleeve::S3, &|_p: &Panel| FlatUntil::new(Flaky { calls: calls.clone() }, e)).unwrap();
    let f = failed(&checks);
    assert!(f.contains(&"S3.determinism".to_string()), "{f:?}");
    assert!(rep.determinism_mismatches > 0);
}

#[test]
fn a_causal_but_wrong_rule_fails_the_tiers_and_not_the_causality_checks() {
    let fx = fixtures();
    let e = entry(&fx);
    let (_, checks) = certify_sleeve(&fx, MutantSleeve::S3, &|_p: &Panel| FlatUntil::new(Half, e)).unwrap();
    let f = failed(&checks);
    for name in ["S3.gross.tier2.identity", "S3.net.tier2.identity", "S3.gross.tier3.weights", "S3.net.tier3.weights"] {
        assert!(f.contains(&name.to_string()), "{name} should fail, got {f:?}");
    }
    for name in ["S3.causality.poisoning", "S3.causality.truncation", "S3.determinism", "S3.gross.window"] {
        assert!(!f.contains(&name.to_string()), "{name} should still pass, got {f:?}");
    }
}
