//! Allocators: `Fixed`, `Equal`, `InverseVol` (static shares, reviews only at declared dates, past-only).

mod common;

use common::*;
use portfolio_construct::*;

const INVVOL_FIXTURE: &str = include_str!("fixtures/invvol_synthetic.txt");

fn inv_vol(lookback: usize, floor: f64) -> AllocatorSpec {
    AllocatorSpec::InverseVol { lookback_bars: lookback, floor, freeze: FreezeRule::AtReviewDates }
}

#[test]
fn equal_and_fixed_are_static_and_reviews_change_nothing() {
    let mut eq = AllocatorState::new(AllocatorSpec::Equal, 3, 0.9).unwrap();
    assert_eq!(eq.shares(), &[0.3, 0.3, 0.3][..], "0.9 / 3");
    let r = [0.01, -0.01, 0.02, -0.02];
    assert_eq!(eq.review(&[&r, &r, &r], &[4, 4, 4]), ReviewOutcome::Held(HoldReason::Static));
    assert_eq!(eq.shares(), &[0.3, 0.3, 0.3][..]);
    let mut fx = AllocatorState::new(AllocatorSpec::Fixed(vec![0.5, 0.25]), 2, 1.0).unwrap();
    assert_eq!(fx.shares(), &[0.5, 0.25][..]);
    assert_eq!(fx.review(&[&r, &r], &[4, 4]), ReviewOutcome::Held(HoldReason::Static));
    assert_eq!(fx.shares(), &[0.5, 0.25][..]);
    let one = AllocatorState::new(AllocatorSpec::Equal, 1, 1.0).unwrap();
    assert_eq!(one.shares(), &[1.0][..]);
}

#[test]
fn fixed_shares_are_validated() {
    let new = |v: Vec<f64>, n: usize, total: f64| AllocatorState::new(AllocatorSpec::Fixed(v), n, total);
    assert_eq!(new(vec![0.5], 2, 1.0).unwrap_err(), AllocError::BadLength);
    assert_eq!(new(vec![], 0, 1.0).unwrap_err(), AllocError::BadLength);
    assert_eq!(new(vec![0.5, 0.0], 2, 1.0).unwrap_err(), AllocError::BadShare);
    assert_eq!(new(vec![0.5, 1.1], 2, 1.0).unwrap_err(), AllocError::BadShare);
    assert_eq!(new(vec![0.6, 0.5], 2, 1.0).unwrap_err(), AllocError::SharesExceedTotal);
    assert!(new(vec![0.6, 0.4], 2, 1.0).is_ok(), "exactly the total is fine");
    assert!(new(vec![0.5, 0.5], 2, 0.9).is_err(), "shares above the total");
    assert!(AllocatorState::new(AllocatorSpec::Equal, 2, 0.0).is_err());
    assert!(AllocatorState::new(AllocatorSpec::Equal, 2, 1.5).is_err());
    assert!(AllocatorState::new(inv_vol(1, 0.0), 2, 1.0).is_err(), "lookback below 2");
    assert!(AllocatorState::new(inv_vol(5, -1.0), 2, 1.0).is_err(), "negative floor");
}

#[test]
fn inverse_vol_hand_computed_two_to_one() {
    // A alternates +-0.01, B alternates +-0.02, 4 bars each. ddof 1: sd_A = sqrt(4 * 0.0001 / 3) = 0.0115470054,
    // sd_B = 2 * sd_A. inverse-vol shares 2/3 and 1/3 of the total.
    let a = [0.01, -0.01, 0.01, -0.01];
    let b = [0.02, -0.02, 0.02, -0.02];
    let mut al = AllocatorState::new(inv_vol(4, 0.0), 2, 1.0).unwrap();
    assert_eq!(al.shares(), &[0.5, 0.5][..], "initial equal split");
    assert_eq!(al.review(&[&a, &b], &[4, 4]), ReviewOutcome::Updated);
    assert_close(al.shares()[0], 2.0 / 3.0, 1e-15, "A");
    assert_close(al.shares()[1], 1.0 / 3.0, 1e-15, "B");
    assert_close(sample_std(&a), (4.0f64 * 0.0001 / 3.0).sqrt(), 1e-18, "sd A");
    // total 0.9 scales the shares: 0.6 / 0.3.
    let mut al = AllocatorState::new(inv_vol(4, 0.0), 2, 0.9).unwrap();
    al.review(&[&a, &b], &[4, 4]);
    assert_close(al.shares()[0], 0.6, 1e-15, "A of 0.9");
    assert_close(al.shares()[1], 0.3, 1e-15, "B of 0.9");
}

#[test]
fn inverse_vol_uses_the_last_lookback_bars_of_the_visible_history() {
    // A has huge early returns that must fall out of a 4-bar window at visible = 8.
    let a = [0.5, -0.5, 0.5, -0.5, 0.01, -0.01, 0.01, -0.01];
    let b = [0.0, 0.0, 0.0, 0.0, 0.02, -0.02, 0.02, -0.02];
    let mut al = AllocatorState::new(inv_vol(4, 0.0), 2, 1.0).unwrap();
    al.review(&[&a, &b], &[8, 8]);
    assert_close(al.shares()[0], 2.0 / 3.0, 1e-15, "window = the last 4 returns");
    // Visible 4 sees the early window: A is far more volatile than B's zeros -> B has zero deviation -> held.
    let mut al = AllocatorState::new(inv_vol(4, 0.0), 2, 1.0).unwrap();
    assert_eq!(al.review(&[&a, &b], &[4, 4]), ReviewOutcome::Held(HoldReason::DegenerateVolatility));
}

#[test]
fn inverse_vol_holds_without_history_and_on_degenerate_volatility() {
    let a = [0.01, -0.01, 0.01];
    let z = [0.0, 0.0, 0.0];
    let mut al = AllocatorState::new(inv_vol(4, 0.0), 2, 1.0).unwrap();
    assert_eq!(al.review(&[&a, &a], &[3, 3]), ReviewOutcome::Held(HoldReason::NotEnoughHistory));
    let a4 = [0.01, -0.01, 0.01, -0.01];
    let z4 = [0.0; 4];
    assert_eq!(
        al.review(&[&a4, &z4], &[4, 4]),
        ReviewOutcome::Held(HoldReason::DegenerateVolatility),
        "zero deviation"
    );
    assert_eq!(al.shares(), &[0.5, 0.5][..], "held: shares are unchanged");
    let nan4 = [0.01, f64::NAN, 0.01, -0.01];
    assert_eq!(
        al.review(&[&a4, &nan4], &[4, 4]),
        ReviewOutcome::Held(HoldReason::DegenerateVolatility),
        "a non-finite return"
    );
    // One sleeve short of history holds the whole review, not just that sleeve.
    assert_eq!(al.review(&[&a4, &z], &[4, 3]), ReviewOutcome::Held(HoldReason::NotEnoughHistory));
    // Wrong arity is a hold, never a panic.
    assert_eq!(al.review(&[&a4], &[4]), ReviewOutcome::Held(HoldReason::NotEnoughHistory));
    // A floor lets a zero-variance sleeve receive a finite weight: sd_B = max(0, 0.01) = 0.01, sd_A = 0.011547...
    let mut floored = AllocatorState::new(inv_vol(4, 0.01), 2, 1.0).unwrap();
    assert_eq!(floored.review(&[&a4, &z4], &[4, 4]), ReviewOutcome::Updated);
    let sd_a = sample_std(&a4);
    let expect_a = (1.0 / sd_a) / (1.0 / sd_a + 1.0 / 0.01);
    assert_close(floored.shares()[0], expect_a, 1e-12, "floored");
    assert!(floored.shares()[1] > floored.shares()[0]);
}

#[test]
fn inverse_vol_shares_sum_to_the_total_and_favour_the_calmer_sleeve() {
    let mut rng = SplitMix64(7);
    for case in 0..200 {
        let n = 2 + (case % 3);
        let rets: Vec<Vec<f64>> =
            (0..n).map(|k| (0..80).map(|_| (rng.unit() - 0.5) * 0.01 * (1.0 + 3.0 * k as f64)).collect()).collect();
        let refs: Vec<&[f64]> = rets.iter().map(|v| v.as_slice()).collect();
        let vis = vec![80; n];
        let mut al = AllocatorState::new(inv_vol(60, 0.0), n, 1.0).unwrap();
        assert_eq!(al.review(&refs, &vis), ReviewOutcome::Updated);
        let total: f64 = al.shares().iter().sum();
        assert_close(total, 1.0, 1e-12, "sum of shares");
        // Sleeve 0 has the smallest scale by construction -> the largest share.
        assert!(al.shares().iter().skip(1).all(|s| *s < al.shares()[0]), "{case}: {:?}", al.shares());
        assert!(al.shares().iter().all(|s| *s > 0.0 && *s <= 1.0));
    }
}

#[test]
fn allocators_see_only_the_past_poisoned_future_changes_nothing() {
    // Values at or after the visible index are replaced by NaN, +-1e9 and garbage: every bit of the result is unchanged.
    let mut rng = SplitMix64(99);
    for case in 0..100 {
        let n = 3;
        let full: Vec<Vec<f64>> =
            (0..n).map(|k| (0..300).map(|_| (rng.unit() - 0.5) * 0.02 * (1.0 + k as f64)).collect()).collect();
        let vis: Vec<usize> = (0..n).map(|_| rng.range(60, 250) as usize).collect();
        let clean: Vec<&[f64]> = full.iter().map(|v| v.as_slice()).collect();
        let mut a = AllocatorState::new(inv_vol(60, 0.0), n, 1.0).unwrap();
        assert_eq!(a.review(&clean, &vis), ReviewOutcome::Updated);
        let poisoned: Vec<Vec<f64>> = full
            .iter()
            .zip(&vis)
            .map(|(v, &m)| {
                let mut p = v.clone();
                for (i, x) in p.iter_mut().enumerate().skip(m) {
                    *x = match (i + case) % 4 {
                        0 => f64::NAN,
                        1 => 1e9,
                        2 => -1e9,
                        _ => f64::INFINITY,
                    };
                }
                p
            })
            .collect();
        let prefs: Vec<&[f64]> = poisoned.iter().map(|v| v.as_slice()).collect();
        let mut b = AllocatorState::new(inv_vol(60, 0.0), n, 1.0).unwrap();
        assert_eq!(b.review(&prefs, &vis), ReviewOutcome::Updated, "case {case}");
        let bits = |s: &AllocatorState| s.shares().iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&a), bits(&b), "case {case}: the allocator read past the visible index");
    }
}

#[test]
fn shares_are_static_between_reviews() {
    // The shares returned by `shares()` do not move unless review() is called; a Held review does not move them either.
    let a: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
    let b: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 0.03 } else { -0.03 }).collect();
    let mut al = AllocatorState::new(inv_vol(60, 0.0), 2, 1.0).unwrap();
    let initial = al.shares().to_vec();
    for _ in 0..5 {
        assert_eq!(al.shares(), initial.as_slice());
    }
    assert_eq!(al.review(&[&a, &b], &[80, 80]), ReviewOutcome::Updated);
    let after = al.shares().to_vec();
    assert_ne!(after, initial);
    assert_eq!(al.review(&[&a, &b], &[10, 10]), ReviewOutcome::Held(HoldReason::NotEnoughHistory));
    assert_eq!(al.shares(), after.as_slice(), "a held review keeps the previous static shares");
}

/// The synthetic fixture: returns and expected shares generated by `tests/fixtures/gen_invvol.py`, which re-states the
/// book key's InverseVol arithmetic in plain Python. The Rust shares must equal the Python ones to 1e-12.
#[test]
fn inverse_vol_matches_the_python_reference_on_synthetic_series() {
    let mut rets: Vec<Vec<f64>> = Vec::new();
    let mut reviews: Vec<(Vec<usize>, Option<Vec<f64>>)> = Vec::new();
    for line in INVVOL_FIXTURE.lines() {
        let mut parts = line.split(' ');
        match parts.next() {
            Some("SLEEVE") => {
                let (_i, _n) = (parts.next().unwrap(), parts.next().unwrap());
                rets.push(parts.next().unwrap().split(',').map(|x| x.parse::<f64>().unwrap()).collect());
            }
            Some("REVIEW") => {
                let vis: Vec<usize> = parts.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
                let shares = match parts.next().unwrap() {
                    "UPDATED" => Some(parts.next().unwrap().split(',').map(|x| x.parse::<f64>().unwrap()).collect()),
                    "HELD" => None,
                    other => panic!("{other}"),
                };
                reviews.push((vis, shares));
            }
            _ => panic!("bad fixture line"),
        }
    }
    assert_eq!(rets.len(), 3);
    assert_eq!(reviews.len(), 7);
    let refs: Vec<&[f64]> = rets.iter().map(|v| v.as_slice()).collect();
    let mut al = AllocatorState::new(inv_vol(60, 0.0), 3, 1.0).unwrap();
    let mut updates = 0;
    for (vis, expected) in &reviews {
        let before = al.shares().to_vec();
        match (al.review(&refs, vis), expected) {
            (ReviewOutcome::Updated, Some(exp)) => {
                updates += 1;
                for (got, want) in al.shares().iter().zip(exp) {
                    assert!((got - want).abs() <= 1e-12, "visible {vis:?}: {got} vs python {want}");
                }
            }
            (ReviewOutcome::Held(_), None) => assert_eq!(al.shares(), before.as_slice()),
            (got, want) => panic!("visible {vis:?}: outcome {got:?} but python says {want:?}"),
        }
    }
    assert_eq!(updates, 6);
}
