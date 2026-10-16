//! Fold construction: hand-computed boundaries (checked against a brute-force Python reference), the edges of purge and
//! embargo, and property tests (disjointness, gaps, coverage, counts) on random parameters.

// Tests write single ranges such as `&[5..6]` on purpose: a list holding ONE range, not the integers 5 to 6.
#![allow(clippy::single_range_in_vec_init)]

use portfolio_eval::error::EvalError;
use portfolio_eval::folds::*;
use portfolio_eval::rng::Rng;

fn r(a: usize, b: usize) -> std::ops::Range<usize> {
    a..b
}

#[test]
fn walk_forward_hand_computed_expanding() {
    // n=100, 4 folds of 10 test bars = last 40 bars; purge 5: train ends 5 bars before each test start.
    let f = walk_forward_folds(100, 4, 10, 20, 5, TrainMode::Expanding).unwrap();
    let want = [((0, 55), (60, 70)), ((0, 65), (70, 80)), ((0, 75), (80, 90)), ((0, 85), (90, 100))];
    assert_eq!(f.len(), 4);
    for (fold, ((a, b), (c, d))) in f.iter().zip(want) {
        assert_eq!(fold.train, vec![r(a, b)]);
        assert_eq!(fold.test, vec![r(c, d)]);
    }
}

#[test]
fn walk_forward_hand_computed_rolling() {
    let f = walk_forward_folds(100, 4, 10, 20, 5, TrainMode::Rolling { train_len: 30 }).unwrap();
    let want = [((25, 55), (60, 70)), ((35, 65), (70, 80)), ((45, 75), (80, 90)), ((55, 85), (90, 100))];
    for (fold, ((a, b), (c, d))) in f.iter().zip(want) {
        assert_eq!(fold.train, vec![r(a, b)]);
        assert_eq!(fold.test, vec![r(c, d)]);
    }
}

#[test]
fn walk_forward_purge_gap_is_exactly_purge_bars() {
    for purge in [0usize, 1, 5, 17] {
        let f = walk_forward_folds(200, 3, 20, 10, purge, TrainMode::Expanding).unwrap();
        for fold in &f {
            let train_end = fold.train[0].end;
            let test_start = fold.test[0].start;
            assert_eq!(test_start - train_end, purge, "purge {purge}");
            // the last `purge` bars before the test are in neither set
            for i in train_end..test_start {
                assert!(!fold.in_train(i) && !fold.in_test(i));
            }
        }
    }
}

#[test]
fn walk_forward_edges() {
    // exactly enough data: total test 3*10 + purge 5 + min_train 20 = 55
    let f = walk_forward_folds(55, 3, 10, 20, 5, TrainMode::Expanding).unwrap();
    assert_eq!(f[0].train, vec![r(0, 20)]);
    assert_eq!(f[0].test, vec![r(25, 35)]);
    assert_eq!(f[2].test, vec![r(45, 55)]);
    // one bar short
    assert!(matches!(
        walk_forward_folds(54, 3, 10, 20, 5, TrainMode::Expanding),
        Err(EvalError::TooShort { need: 55, got: 54, .. })
    ));
    // zero purge: train ends where the test starts
    let f = walk_forward_folds(40, 2, 10, 20, 0, TrainMode::Expanding).unwrap();
    assert_eq!(f[0].train, vec![r(0, 20)]);
    assert_eq!(f[0].test, vec![r(20, 30)]);
    // bad parameters
    assert!(walk_forward_folds(100, 0, 10, 20, 5, TrainMode::Expanding).is_err());
    assert!(walk_forward_folds(100, 4, 0, 20, 5, TrainMode::Expanding).is_err());
    assert!(walk_forward_folds(100, 4, 10, 0, 5, TrainMode::Expanding).is_err());
    assert!(walk_forward_folds(100, 4, 10, 20, 5, TrainMode::Rolling { train_len: 10 }).is_err());
    assert!(walk_forward_folds(usize::MAX, usize::MAX, 2, 1, 0, TrainMode::Expanding).is_err());
}

#[test]
fn purge_embargo_rule_and_cap() {
    // max(5, rebalance, ceil(holding)), capped at test_len / 2
    assert_eq!(purge_embargo_bars(1, 1.0, 100), 5);
    assert_eq!(purge_embargo_bars(21, 3.0, 100), 21);
    assert_eq!(purge_embargo_bars(1, 12.2, 100), 13);
    assert_eq!(purge_embargo_bars(60, 40.0, 100), 50); // capped at half a test window
    assert_eq!(purge_embargo_bars(1, 1.0, 8), 4); // the cap wins over the 5-bar floor
    assert_eq!(purge_embargo_bars(1, f64::NAN, 100), 5);
    assert_eq!(purge_embargo_bars(1, -3.0, 100), 5);
    assert_eq!(purge_embargo_bars(1, 1.0, 0), 0);
    assert_eq!(MIN_PURGE_BARS, 5);
}

#[test]
fn kfold_hand_computed_with_purge_and_embargo_edges() {
    // n=23, k=4: blocks 6,6,6,5. purge 2, embargo 3.
    let f = purged_kfold(23, 4, 2, 3).unwrap();
    assert_eq!(f[0].test, vec![r(0, 6)]);
    assert_eq!(f[0].train, vec![r(9, 23)]); // embargo removes 6,7,8
    assert_eq!(f[1].test, vec![r(6, 12)]);
    assert_eq!(f[1].train, vec![r(0, 4), r(15, 23)]); // purge removes 4,5; embargo 12,13,14
    assert_eq!(f[2].test, vec![r(12, 18)]);
    assert_eq!(f[2].train, vec![r(0, 10), r(21, 23)]);
    assert_eq!(f[3].test, vec![r(18, 23)]);
    assert_eq!(f[3].train, vec![r(0, 16)]); // purge removes 16,17; embargo runs past the end
}

#[test]
fn kfold_purge_of_exactly_the_preceding_block_and_more() {
    // purge larger than the data before the first block clamps at 0 instead of underflowing
    // with huge purge every fold has an empty training set -> a typed error, not a panic
    assert!(matches!(purged_kfold(20, 4, 100, 0), Err(EvalError::TooShort { .. })));
    assert!(purged_kfold(20, 1, 0, 0).is_err());
    assert!(purged_kfold(3, 4, 0, 0).is_err());
}

#[test]
fn group_partition_sizes_differ_by_at_most_one() {
    let g = group_ranges(23, 4).unwrap();
    assert_eq!(g, vec![r(0, 6), r(6, 12), r(12, 18), r(18, 23)]);
    assert!(group_ranges(3, 0).is_err());
    assert!(group_ranges(3, 4).is_err());
}

#[test]
fn combinations_are_lexicographic_and_counted() {
    assert_eq!(combinations(4, 2), vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3]]);
    assert_eq!(combinations(3, 0), vec![Vec::<usize>::new()]);
    assert_eq!(combinations(3, 4), Vec::<Vec<usize>>::new());
    assert_eq!(combinations_count(6, 2), Some(15));
    assert_eq!(combinations_count(10, 5), Some(252));
    assert_eq!(combinations_count(3, 5), Some(0));
    assert_eq!(combinations(8, 4).len() as u64, combinations_count(8, 4).unwrap());
    assert_eq!(combinations_count(200, 100), None);
}

#[test]
fn cpcv_counts_and_merged_test_blocks() {
    // 6 groups, 2 test groups: C(6,2) = 15 splits; each group is tested in C(5,1) = 5 of them
    let f = cpcv_splits(60, 6, 2, 3, 3).unwrap();
    assert_eq!(f.len(), 15);
    let groups = group_ranges(60, 6).unwrap();
    for g in &groups {
        let n = f.iter().filter(|x| x.test.iter().any(|t| t.start <= g.start && g.end <= t.end)).count();
        assert_eq!(n, 5);
    }
    // adjacent groups (0,1) form ONE test range, non-adjacent (0,2) two ranges
    assert_eq!(f[0].test, vec![r(0, 20)]);
    assert_eq!(f[1].test, vec![r(0, 10), r(20, 30)]);
    // the first split's training set: purge/embargo only cut around the merged block
    assert_eq!(f[0].train, vec![r(23, 60)]);
    assert!(cpcv_splits(60, 6, 0, 0, 0).is_err());
    assert!(cpcv_splits(60, 6, 6, 0, 0).is_err());
    assert!(cpcv_splits(60, 1, 1, 0, 0).is_err());
    assert!(cpcv_splits(60, 60, 30, 0, 0).is_err()); // too many combinations
}

fn assert_valid(fold: &Fold, n: usize, purge: usize, embargo: usize, maximal: bool) {
    // ranges are sorted, in bounds and non-empty
    for rg in fold.train.iter().chain(fold.test.iter()) {
        assert!(rg.start < rg.end && rg.end <= n, "range {rg:?}");
    }
    for w in fold.train.windows(2) {
        assert!(w[0].end < w[1].start, "train ranges must be separated: {:?}", fold.train);
    }
    // disjoint
    for i in 0..n {
        assert!(!(fold.in_train(i) && fold.in_test(i)), "bar {i} in both");
    }
    // purge and embargo gaps around every test range
    for t in &fold.test {
        for i in t.start.saturating_sub(purge)..t.start {
            assert!(!fold.in_train(i), "bar {i} within {purge} before the test range {t:?} is trained on");
        }
        for i in t.end..(t.end + embargo).min(n) {
            assert!(!fold.in_train(i), "bar {i} within {embargo} after the test range {t:?} is trained on");
        }
    }
    // maximality (K-fold and CPCV): every bar that is not test and not within a gap IS trained on
    for i in 0..n {
        if !maximal {
            break;
        }
        let near_test = fold.test.iter().any(|t| i + purge >= t.start && i < t.end + embargo);
        assert_eq!(fold.in_train(i), !near_test && !fold.in_test(i), "bar {i}");
    }
}

#[test]
fn property_kfold_and_cpcv_are_valid_for_random_parameters() {
    let mut rng = Rng::seed_from_u64(2024);
    let mut checked = 0;
    for _ in 0..300 {
        let n = 30 + rng.below(200) as usize;
        let k = 2 + rng.below(6) as usize;
        let purge = rng.below(6) as usize;
        let embargo = rng.below(6) as usize;
        if let Ok(folds) = purged_kfold(n, k, purge, embargo) {
            checked += 1;
            assert_eq!(folds.len(), k);
            // test blocks partition 0..n exactly once
            let mut cover = vec![0u8; n];
            for f in &folds {
                assert_valid(f, n, purge, embargo, true);
                for t in &f.test {
                    for i in t.clone() {
                        cover[i] += 1;
                    }
                }
            }
            assert!(cover.iter().all(|c| *c == 1));
        }
        let g = 4 + rng.below(4) as usize;
        let kt = 1 + rng.below((g - 1) as u64) as usize;
        if let Ok(folds) = cpcv_splits(n, g, kt, purge, embargo) {
            assert_eq!(folds.len() as u64, combinations_count(g, kt).unwrap());
            for f in &folds {
                assert_valid(f, n, purge, embargo, true);
            }
        }
    }
    assert!(checked > 100);
}

#[test]
fn property_walk_forward_never_overlaps_and_respects_purge() {
    let mut rng = Rng::seed_from_u64(77);
    for _ in 0..300 {
        let n = 50 + rng.below(400) as usize;
        let k = 1 + rng.below(6) as usize;
        let tl = 1 + rng.below(30) as usize;
        let mt = 1 + rng.below(20) as usize;
        let purge = rng.below(10) as usize;
        let mode = if rng.below(2) == 0 {
            TrainMode::Expanding
        } else {
            TrainMode::Rolling { train_len: mt + rng.below(40) as usize }
        };
        let Ok(folds) = walk_forward_folds(n, k, tl, mt, purge, mode) else { continue };
        assert_eq!(folds.len(), k);
        let mut prev_end = None;
        for f in &folds {
            assert_valid(f, n, purge, 0, false);
            let (tr, te) = (&f.train[0], &f.test[0]);
            // training strictly precedes the test window, with a gap of exactly `purge`
            assert_eq!(te.start - tr.end, purge);
            assert!(tr.end - tr.start >= mt);
            assert_eq!(te.end - te.start, tl);
            // test windows are contiguous and non-overlapping, and end at the last bar
            if let Some(pe) = prev_end {
                assert_eq!(te.start, pe);
            }
            prev_end = Some(te.end);
            if let TrainMode::Rolling { train_len } = mode {
                assert!(tr.end - tr.start <= train_len);
            } else {
                assert_eq!(tr.start, 0);
            }
        }
        assert_eq!(prev_end, Some(n));
    }
}

#[test]
fn more_data_never_removes_walk_forward_folds() {
    // monotone in sample length: the number of feasible folds cannot decrease as n grows
    let mut last = 0;
    for n in 20..400 {
        let mut feasible = 0;
        for k in 1..30 {
            if walk_forward_folds(n, k, 10, 15, 5, TrainMode::Expanding).is_ok() {
                feasible = k;
            }
        }
        assert!(feasible >= last, "n={n}: {feasible} < {last}");
        last = feasible;
    }
}
