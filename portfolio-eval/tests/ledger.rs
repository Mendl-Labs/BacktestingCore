//! Trial ledger, sealed holdout and effective breadth.

// Tests write single ranges such as `&[5..6]` on purpose: a list holding ONE range, not the integers 5 to 6.
#![allow(clippy::single_range_in_vec_init)]

use portfolio_eval::error::EvalError;
use portfolio_eval::ledger::*;
use portfolio_eval::rng::Rng;
use portfolio_eval::stats;

#[test]
fn every_new_configuration_increments_the_trial_count_and_repeats_do_not() {
    let mut l = TrialLedger::new();
    assert_eq!(l.k_effective(), 0);
    assert_eq!(l.record_config("a", 0.5).unwrap(), RecordOutcome::New);
    assert_eq!(l.k_effective(), 1);
    assert_eq!(l.record_config("b", 0.7).unwrap(), RecordOutcome::New);
    assert_eq!(l.k_effective(), 2);
    assert_eq!(l.record_config("a", 9.9).unwrap(), RecordOutcome::Duplicate { upgraded: false });
    assert_eq!(l.k_effective(), 2);
    assert_eq!(l.n_configs(), 2);
    // the FIRST recorded Sharpe is kept (no silent overwrite by a later, luckier evaluation)
    assert_eq!(l.sharpes(), vec![0.5, 0.7]);
    for i in 0..218 {
        l.record_config(&format!("cfg{i}"), 0.1 * (i % 7) as f64).unwrap();
    }
    assert_eq!(l.k_effective(), 220);
}

#[test]
fn k_effective_formula_matches_the_design_example() {
    // design 3.5: configs 220, universe 1, allocator 1, prior 0, screens 0 -> 220
    // K = configs x universe x allocator + screens + candidates + lineage_prior + concurrent
    let mut l = TrialLedger::new();
    for i in 0..10 {
        l.record_config(&format!("c{i}"), 0.1 * i as f64).unwrap();
    }
    l.set_universe_variants(3).unwrap();
    l.set_allocator_variants(2).unwrap();
    assert_eq!(l.k_effective(), 60);
    l.add_screens(7);
    assert_eq!(l.k_effective(), 67);
    l.add_candidates_screened(5);
    assert_eq!(l.k_effective(), 72);
    l.set_lineage_prior(100);
    assert_eq!(l.k_effective(), 172);
    l.set_concurrent_tenant(8);
    assert_eq!(l.k_effective(), 180);
    l.add_screens(3);
    assert_eq!(l.summary().screens, 10);
    assert_eq!(l.k_effective(), 183);
    let s = l.summary();
    assert_eq!(
        (s.configs, s.universe_variants, s.allocator_variants, s.lineage_prior, s.concurrent_tenant),
        (10, 3, 2, 100, 8)
    );
    assert_eq!(s.k_effective, 183);
    assert!(l.set_universe_variants(0).is_err());
    assert!(l.set_allocator_variants(0).is_err());
    // saturating, never overflowing or panicking
    l.set_lineage_prior(usize::MAX);
    assert_eq!(l.k_effective(), usize::MAX);
}

#[test]
fn failed_configurations_still_count_but_do_not_enter_the_dispersion() {
    let mut l = TrialLedger::new();
    l.record_config("ok1", 0.4).unwrap();
    l.record_config("ok2", 0.6).unwrap();
    assert_eq!(l.record_failed_config("boom").unwrap(), RecordOutcome::NewFailed);
    assert_eq!(l.record_config("nan", f64::NAN).unwrap(), RecordOutcome::NewFailed);
    assert_eq!(l.record_config("inf", f64::INFINITY).unwrap(), RecordOutcome::NewFailed);
    assert_eq!(l.k_effective(), 5);
    assert_eq!(l.sharpes().len(), 2);
    assert_eq!(l.summary().failed_configs, 3);
    // a retry of a failed trial upgrades it in place, still one trial
    assert_eq!(l.record_config("boom", 0.5).unwrap(), RecordOutcome::Duplicate { upgraded: true });
    assert_eq!(l.k_effective(), 5);
    assert_eq!(l.sharpes().len(), 3);
    assert!(l.record_config("", 0.1).is_err());
}

#[test]
fn robust_dispersion_survives_blown_up_configurations_where_the_plain_one_does_not() {
    // memory 2026-09-18: blown-up configurations inflated the plain std to 7.47 against a 1.27 null bar
    let mut l = TrialLedger::new();
    for i in 0..60 {
        l.record_config(&format!("n{i}"), 1.2 * ((i as f64) / 59.0 - 0.5) * 2.0).unwrap();
    }
    let robust_before = l.dispersion_robust().unwrap();
    for i in 0..3 {
        l.record_config(&format!("blown{i}"), 40.0 + i as f64).unwrap();
    }
    let robust = l.dispersion_robust().unwrap();
    let plain = l.dispersion_plain().unwrap();
    assert!((robust - robust_before).abs() < 0.08, "{robust_before} -> {robust}");
    assert!(plain > 4.0 * robust, "plain {plain} robust {robust}");
    let s = l.summary();
    assert_eq!(s.dispersion_robust, Some(robust));
    assert_eq!(s.dispersion_plain, Some(plain));
}

#[test]
fn ledger_is_invariant_to_insertion_order() {
    let mut rng = Rng::seed_from_u64(12);
    let entries: Vec<(String, f64)> = (0..50).map(|i| (format!("cfg{i}"), rng.normal())).collect();
    let mut a = TrialLedger::new();
    for (id, s) in &entries {
        a.record_config(id, *s).unwrap();
    }
    let mut b = TrialLedger::new();
    for (id, s) in entries.iter().rev() {
        b.record_config(id, *s).unwrap();
    }
    assert_eq!(a.k_effective(), b.k_effective());
    assert_eq!(a.dispersion_robust().unwrap().to_bits(), b.dispersion_robust().unwrap().to_bits());
    assert_eq!(a.dispersion_plain().unwrap().to_bits(), b.dispersion_plain().unwrap().to_bits());
}

#[test]
fn dispersion_needs_two_finite_sharpes() {
    let mut l = TrialLedger::new();
    assert!(matches!(l.dispersion_robust(), Err(EvalError::TooShort { need: 2, got: 0, .. })));
    l.record_config("a", 1.0).unwrap();
    assert!(matches!(l.dispersion_robust(), Err(EvalError::TooShort { need: 2, got: 1, .. })));
    assert!(l.dispersion_plain().is_err());
    assert_eq!(l.summary().dispersion_robust, None);
}

#[test]
fn ledger_deflated_sharpe_uses_k_effective_and_robust_dispersion() {
    let mut rng = Rng::seed_from_u64(31);
    let oos: Vec<f64> = (0..800).map(|_| 0.0005 + 0.01 * rng.normal()).collect();
    let mut l = TrialLedger::new();
    assert!(l.deflated_sharpe_of(&oos, 252.0, true).is_err(), "an empty ledger must refuse");
    for i in 0..40 {
        l.record_config(&format!("c{i}"), 0.3 * rng.normal()).unwrap();
    }
    let few = l.deflated_sharpe_of(&oos, 252.0, true).unwrap();
    // every extra trial (here via screens) lowers the DSR
    l.add_screens(500);
    let many = l.deflated_sharpe_of(&oos, 252.0, true).unwrap();
    assert!(many.dsr < few.dsr, "{} {}", many.dsr, few.dsr);
    assert!(many.sr0 > few.sr0);
    // and it equals the direct call
    let direct = portfolio_eval::dsr::deflated_sharpe_from_returns(
        &oos,
        l.k_effective(),
        l.dispersion_robust().unwrap(),
        252.0,
        true,
    )
    .unwrap();
    assert_eq!(many, direct);
}

#[test]
fn holdout_tail_is_max_of_fraction_and_minimum() {
    // 1000 bars: 25% = 250 bars; a 24-month minimum of 504 bars wins
    assert_eq!(holdout_tail(1000, 0.25, 0).unwrap(), 750..1000);
    assert_eq!(holdout_tail(1000, 0.25, 504).unwrap(), 496..1000);
    assert_eq!(holdout_tail(1001, 0.25, 0).unwrap(), 750..1001); // ceil(250.25) = 251 bars
    assert!(matches!(holdout_tail(100, 0.25, 100), Err(EvalError::TooShort { .. })));
    assert!(holdout_tail(100, 0.0, 10).is_err());
    assert!(holdout_tail(100, 1.0, 10).is_err());
    assert!(holdout_tail(100, f64::NAN, 10).is_err());
    assert!(holdout_tail(0, 0.25, 0).is_err());
}

#[test]
fn first_look_spends_the_holdout_and_every_later_look_is_post_hoc() {
    let mut h = SealedHoldout::new(750..1000).unwrap();
    assert!(!h.is_spent());
    assert_eq!(h.looks(), 0);
    let l1 = h.look(1_000);
    assert!(l1.first_look && !l1.post_hoc && l1.looks == 1 && l1.spent_at_ms == 1_000);
    assert!(h.is_spent());
    assert_eq!(h.spent_at_ms(), Some(1_000));
    let l2 = h.look(2_000);
    assert!(!l2.first_look && l2.post_hoc && l2.looks == 2);
    assert_eq!(l2.spent_at_ms, 1_000, "the spend time is the FIRST look, not the latest");
    let l3 = h.look(3_000);
    assert!(l3.post_hoc && l3.looks == 3);
    assert_eq!(h.spent_at_ms(), Some(1_000));
    assert_eq!(h.range(), 750..1000);
    assert!(SealedHoldout::new(5..5).is_err());
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 9..5;
    assert!(SealedHoldout::new(reversed).is_err());
}

#[test]
fn holdout_must_not_overlap_tuning_ranges() {
    let h = SealedHoldout::new(750..1000).unwrap();
    assert!(h.verify_disjoint(&[0..750]).is_ok(), "touching at the boundary is disjoint");
    assert!(h.verify_disjoint(&[0..300, 400..750, 1000..1100]).is_ok());
    assert!(matches!(h.verify_disjoint(&[0..751]), Err(EvalError::HoldoutOverlap { other_end: 751, .. })));
    assert!(matches!(h.verify_disjoint(&[999..1200]), Err(EvalError::HoldoutOverlap { other_start: 999, .. })));
    assert!(h.verify_disjoint(&[800..900]).is_err());
    assert!(h.verify_disjoint(&[700..1100]).is_err());
    assert!(h.verify_disjoint(&[5..5]).is_ok(), "an empty range overlaps nothing");
    assert!(h.verify_disjoint(&[]).is_ok());
}

#[test]
fn effective_breadth_is_the_participation_ratio() {
    // identity: N independent bets
    let id: Vec<Vec<f64>> = (0..5).map(|i| (0..5).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    assert!((effective_breadth_from_correlation(&id).unwrap() - 5.0).abs() < 1e-12);
    // all ones: one bet
    let ones = vec![vec![1.0; 4]; 4];
    assert!((effective_breadth_from_correlation(&ones).unwrap() - 1.0).abs() < 1e-12);
    // two perfectly correlated pairs: 2 bets. matrix N=4: sum of squares = 4 (diag) + 2 x 2 x 1 = 8 -> 16/8 = 2
    let m =
        vec![vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]];
    assert!((effective_breadth_from_correlation(&m).unwrap() - 2.0).abs() < 1e-12);
    // equicorrelation rho: N^2 / (N + N(N-1) rho^2)
    let (n, rho) = (6usize, 0.3);
    let eq: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect()).collect();
    let want = (n * n) as f64 / (n as f64 + (n * (n - 1)) as f64 * rho * rho);
    assert!((effective_breadth_from_correlation(&eq).unwrap() - want).abs() < 1e-12);
    // errors
    assert!(effective_breadth_from_correlation(&[]).is_err());
    assert!(matches!(
        effective_breadth_from_correlation(&[vec![1.0, 0.0], vec![0.0]]),
        Err(EvalError::LengthMismatch { .. })
    ));
    assert!(effective_breadth_from_correlation(&[vec![1.0, 0.5], vec![0.4, 1.0]]).is_err()); // asymmetric
    assert!(effective_breadth_from_correlation(&[vec![1.0, 1.5], vec![1.5, 1.0]]).is_err()); // out of range
    assert!(effective_breadth_from_correlation(&[vec![0.9, 0.0], vec![0.0, 1.0]]).is_err()); // diagonal
    assert!(matches!(effective_breadth_from_correlation(&[vec![f64::NAN]]), Err(EvalError::NonFinite { .. })));
}

#[test]
fn correlation_matrices_of_simulated_series_give_breadth_between_one_and_n() {
    let mut rng = Rng::seed_from_u64(9);
    let n = 200;
    let common: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let series: Vec<Vec<f64>> = (0..8).map(|_| common.iter().map(|c| 0.7 * c + rng.normal()).collect()).collect();
    let corr: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            (0..8).map(|j| if i == j { 1.0 } else { stats::correlation(&series[i], &series[j]).unwrap() }).collect()
        })
        .collect();
    let b = effective_breadth_from_correlation(&corr).unwrap();
    assert!(b > 1.0 && b < 8.0, "{b}");
}

#[test]
fn a_hub_instrument_shared_by_many_legs_is_one_bet() {
    // memory 2026-09-19: EUR-SEK (instrument 1) in 8 of 9 legs
    let mut legs: Vec<Vec<u32>> = (0..8).map(|i| vec![1, 100 + i as u32]).collect();
    legs.push(vec![200, 201]);
    assert_eq!(independent_leg_groups(&legs), 2);
    // fully disjoint legs are all independent
    assert_eq!(independent_leg_groups(&[vec![1, 2], vec![3, 4], vec![5, 6]]), 3);
    // a chain connects transitively
    assert_eq!(independent_leg_groups(&[vec![1, 2], vec![2, 3], vec![3, 4], vec![9, 10]]), 2);
    assert_eq!(independent_leg_groups(&[]), 0);
    assert_eq!(independent_leg_groups(&[vec![]]), 1);
    // order does not matter
    let mut rev = legs.clone();
    rev.reverse();
    assert_eq!(independent_leg_groups(&rev), 2);
}
