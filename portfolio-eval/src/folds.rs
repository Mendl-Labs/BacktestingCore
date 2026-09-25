//! Folds with purge and embargo on a joint calendar, expressed as bar-index ranges (design 4.3, "Walk-forward").
//!
//! * `purge` removes the `purge` bars immediately BEFORE each test range from training: a training observation whose
//!   label or holding period reaches into the test range must not be trained on.
//! * `embargo` removes the `embargo` bars immediately AFTER each test range from training (serial dependence leaking
//!   backwards from the test period into later training data). It only matters when training data can lie after the
//!   test range (purged K-fold, CPCV); walk-forward trains strictly before the test range.
//! * Design value for both: `max(5 bars, rebalance interval, mean in-sample holding period)`, capped at half a test
//!   window ([`purge_embargo_bars`]).
//!
//! Ranges are half-open `[start, end)` over `0..n_bars`. The functions here do not look at any returns, so they cannot
//! leak information; the property tests in `tests/folds.rs` check disjointness, the purge and embargo gaps and
//! coverage on random inputs.

use crate::error::{invalid, EvalError, Result};
use std::ops::Range;

/// Lower bound of the design's purge/embargo rule.
pub const MIN_PURGE_BARS: usize = 5;

/// One train/test split. `test` is one range for walk-forward and K-fold, several for CPCV.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    pub train: Vec<Range<usize>>,
    pub test: Vec<Range<usize>>,
}

impl Fold {
    /// Number of training bars.
    pub fn train_len(&self) -> usize {
        self.train.iter().map(|r| r.end - r.start).sum()
    }
    /// Number of test bars.
    pub fn test_len(&self) -> usize {
        self.test.iter().map(|r| r.end - r.start).sum()
    }
    /// Is bar `i` in a training range?
    pub fn in_train(&self, i: usize) -> bool {
        self.train.iter().any(|r| r.contains(&i))
    }
    /// Is bar `i` in a test range?
    pub fn in_test(&self, i: usize) -> bool {
        self.test.iter().any(|r| r.contains(&i))
    }
}

/// How the training window moves in walk-forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainMode {
    /// Train on everything from bar 0 up to the purge gap.
    Expanding,
    /// Train on at most `train_len` bars ending at the purge gap.
    Rolling { train_len: usize },
}

/// The design's purge/embargo length: `max(5, rebalance interval, ceil(mean holding period))`, capped at half a test
/// window (`test_len / 2`, integer division).
pub fn purge_embargo_bars(rebalance_interval_bars: usize, mean_holding_bars: f64, test_len: usize) -> usize {
    let holding =
        if mean_holding_bars.is_finite() && mean_holding_bars > 0.0 { mean_holding_bars.ceil() as usize } else { 0 };
    let raw = MIN_PURGE_BARS.max(rebalance_interval_bars).max(holding);
    raw.min(test_len / 2)
}

/// Walk-forward folds. The LAST `n_folds x test_len` bars are the contiguous, equal-length test windows (so the most
/// recent data is always evaluated and no remainder is silently dropped); fold `f` trains on bars ending `purge` bars
/// before its test window (from bar 0 when expanding, or the latest `train_len` bars when rolling). Every fold must
/// have at least `min_train` training bars.
pub fn walk_forward_folds(
    n_bars: usize,
    n_folds: usize,
    test_len: usize,
    min_train: usize,
    purge: usize,
    mode: TrainMode,
) -> Result<Vec<Fold>> {
    if n_folds == 0 {
        return Err(invalid("n_folds", "must be at least 1"));
    }
    if test_len == 0 {
        return Err(invalid("test_len", "must be at least 1"));
    }
    if min_train == 0 {
        return Err(invalid("min_train", "must be at least 1"));
    }
    if let TrainMode::Rolling { train_len } = mode {
        if train_len < min_train {
            return Err(invalid("train_len", format!("rolling train_len {train_len} is below min_train {min_train}")));
        }
    }
    let total_test =
        n_folds.checked_mul(test_len).ok_or_else(|| invalid("test_len", "n_folds x test_len overflows usize"))?;
    let need = total_test
        .checked_add(purge)
        .and_then(|v| v.checked_add(min_train))
        .ok_or_else(|| invalid("purge", "sizes overflow usize"))?;
    if need > n_bars {
        return Err(EvalError::TooShort { what: "walk-forward folds", need, got: n_bars });
    }
    let first_test = n_bars - total_test;
    let mut folds = Vec::with_capacity(n_folds);
    for f in 0..n_folds {
        let test_start = first_test + f * test_len;
        let test_end = test_start + test_len;
        let train_end = test_start - purge;
        let train_start = match mode {
            TrainMode::Expanding => 0,
            TrainMode::Rolling { train_len } => train_end.saturating_sub(train_len),
        };
        let (train, test) = (train_start..train_end, test_start..test_end);
        folds.push(Fold { train: vec![train], test: vec![test] });
    }
    Ok(folds)
}

/// Partition `0..n_bars` into `n_groups` contiguous groups whose sizes differ by at most one (the first
/// `n_bars % n_groups` groups are one bar longer).
pub fn group_ranges(n_bars: usize, n_groups: usize) -> Result<Vec<Range<usize>>> {
    if n_groups == 0 {
        return Err(invalid("n_groups", "must be at least 1"));
    }
    if n_bars < n_groups {
        return Err(EvalError::TooShort { what: "group partition", need: n_groups, got: n_bars });
    }
    let base = n_bars / n_groups;
    let rem = n_bars % n_groups;
    let mut out = Vec::with_capacity(n_groups);
    let mut start = 0;
    for g in 0..n_groups {
        let len = base + usize::from(g < rem);
        out.push(start..start + len);
        start += len;
    }
    Ok(out)
}

/// Training ranges for a set of test ranges: everything in `0..n_bars` except the test ranges, the `purge` bars before
/// each of them and the `embargo` bars after each of them.
pub fn train_ranges_excluding(
    n_bars: usize,
    tests: &[Range<usize>],
    purge: usize,
    embargo: usize,
) -> Vec<Range<usize>> {
    let mut forbidden: Vec<Range<usize>> =
        tests.iter().map(|r| r.start.saturating_sub(purge)..r.end.saturating_add(embargo).min(n_bars)).collect();
    forbidden.sort_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::new();
    for r in forbidden {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    let mut train = Vec::new();
    let mut cursor = 0;
    for r in merged {
        if r.start > cursor {
            train.push(cursor..r.start);
        }
        cursor = cursor.max(r.end);
    }
    if cursor < n_bars {
        train.push(cursor..n_bars);
    }
    train
}

/// Purged K-fold (Lopez de Prado): `k` contiguous test blocks, each trained on the rest minus the purge and embargo
/// gaps around it.
pub fn purged_kfold(n_bars: usize, k: usize, purge: usize, embargo: usize) -> Result<Vec<Fold>> {
    if k < 2 {
        return Err(invalid("k", "purged K-fold needs at least 2 folds"));
    }
    let groups = group_ranges(n_bars, k)?;
    let mut folds = Vec::with_capacity(k);
    for g in groups {
        let train = train_ranges_excluding(n_bars, std::slice::from_ref(&g), purge, embargo);
        if train.is_empty() {
            return Err(EvalError::TooShort { what: "purged K-fold training set", need: 1, got: 0 });
        }
        folds.push(Fold { train, test: vec![g] });
    }
    Ok(folds)
}

/// `C(n, k)` with overflow detection.
pub fn combinations_count(n: usize, k: usize) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r.checked_mul((n - i) as u128)? / (i as u128 + 1);
        if r > u64::MAX as u128 {
            return None;
        }
    }
    Some(r as u64)
}

/// The `k`-subsets of `0..n` in lexicographic order.
pub fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // advance
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Combinatorial purged cross-validation splits (Lopez de Prado 2018, ch. 12): the bars are cut into `n_groups`
/// contiguous groups and every choice of `k_test` groups is a test set (`C(n_groups, k_test)` splits, lexicographic),
/// with purge and embargo around every contiguous test block. Adjacent chosen groups form ONE test block, so no gap is
/// cut between them.
pub fn cpcv_splits(n_bars: usize, n_groups: usize, k_test: usize, purge: usize, embargo: usize) -> Result<Vec<Fold>> {
    if n_groups < 2 || k_test == 0 || k_test >= n_groups {
        return Err(invalid(
            "k_test",
            format!("need 1 <= k_test < n_groups, got k_test={k_test}, n_groups={n_groups}"),
        ));
    }
    match combinations_count(n_groups, k_test) {
        Some(c) if c <= 200_000 => {}
        _ => return Err(invalid("n_groups", "too many combinations (limit 200000)")),
    }
    let groups = group_ranges(n_bars, n_groups)?;
    let mut folds = Vec::new();
    for combo in combinations(n_groups, k_test) {
        let mut test: Vec<Range<usize>> = Vec::new();
        for g in combo {
            let r = groups[g].clone();
            match test.last_mut() {
                Some(last) if last.end == r.start => last.end = r.end,
                _ => test.push(r),
            }
        }
        let train = train_ranges_excluding(n_bars, &test, purge, embargo);
        if train.is_empty() {
            return Err(EvalError::TooShort { what: "CPCV training set", need: 1, got: 0 });
        }
        folds.push(Fold { train, test });
    }
    Ok(folds)
}
