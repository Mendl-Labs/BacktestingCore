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

// ---------------------------------------------------------------------------------------------------------------
// Regime-stratified window placement (slice A1-1 of
// `product-mandate/SHARED_VERIFICATION_INFRASTRUCTURE_PLAN.md`, section 2.1 item A1).
//
// This block is a deliberate PORT of `backtest::python_validation::resolve_wf_window_offsets` (and its
// `regime_segments`/`resolve_wf_step` helpers) from the discovery engine's own hand-rolled walk-forward, not a new
// design. The plan doc's own words: "reserves one window per distinct volatility-tercile regime present in the data
// (scarcest label first) before falling back to uniform spacing... A direct swap would silently drop regime-diversity
// window placement." `portfolio_eval::folds::walk_forward_folds` is meant to become the single source of truth for
// fold-slicing across the codebase's five duplicate walk-forward implementations (plan section 0 item 1), and cannot
// do that honestly until it can reproduce this behavior. The algorithm below (segment detection, scarcest-first
// reservation, midpoint-targeted placement, disjointness bookkeeping, uniform fallback) is copied faithfully from
// `python_validation.rs`, not re-derived from this crate's own understanding of "regime stratification".
//
// `portfolio-eval` is a zero-dependency crate (see the crate's own module doc, P1) and is deliberately excluded from
// the workspace so it never has to resolve the workspace's private git dependencies -- so it cannot depend on
// `quant-diagnostics` (which itself pulls in `serde`) to reuse `quant_diagnostics::VolatilityRegime` directly.
// `RegimeLabel` below is the smallest possible mirror of that enum's tercile semantics (three labels, `Copy`,
// structural equality, no serialization) -- not a new taxonomy.
// ---------------------------------------------------------------------------------------------------------------

/// A volatility-tercile regime label. Mirrors `quant_diagnostics::VolatilityRegime`'s three-way
/// Low/Medium/High split (`quant-diagnostics/src/volatility_regime.rs`) structurally -- same three
/// labels, same meaning (a market-state classification computed purely from the trailing return
/// series) -- but is redefined here, rather than imported, because `portfolio-eval` is a
/// zero-dependency crate excluded from the workspace (see this crate's own module doc, P1) and
/// `VolatilityRegime` itself depends on `serde`. Callers translate their own `VolatilityRegime`
/// values into this type before calling [`walk_forward_folds_stratified`] or
/// [`resolve_wf_window_offsets`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeLabel {
    Low,
    Medium,
    High,
}

/// Contiguous regime **segments** -- maximal runs of the same regime label -- across an
/// already-classified regime series. `None` entries (not enough trailing history yet to classify)
/// are skipped without resetting the running label or extending the current segment's span, so a
/// gap mid-series doesn't spuriously split or merge two segments of the same regime around it.
/// `end_idx` is inclusive. Ported verbatim (module-level doc above) from
/// `backtest::python_validation::regime_segments`.
fn regime_segments(regimes: &[Option<RegimeLabel>]) -> Vec<(usize, usize, RegimeLabel)> {
    let mut segments = Vec::new();
    let mut current: Option<(usize, usize, RegimeLabel)> = None;
    for (i, &r) in regimes.iter().enumerate() {
        if let Some(label) = r {
            match current {
                Some((start, _, prev_label)) if prev_label == label => {
                    current = Some((start, i, label));
                }
                Some(seg) => {
                    segments.push(seg);
                    current = Some((i, i, label));
                }
                None => {
                    current = Some((i, i, label));
                }
            }
        }
    }
    if let Some(seg) = current {
        segments.push(seg);
    }
    segments
}

/// Bar-offset step between successive uniformly-spaced windows, chosen so `num_windows` windows of
/// `window_size` bars each span the FULL available range `n` (the last window's end lands at `n`).
/// Ported verbatim from `backtest::python_validation::resolve_wf_step` (see that function's doc
/// comment for the 2026-08 overlap-vs-coverage history this formula fixes); kept private here since
/// only [`resolve_wf_window_offsets`]'s uniform-fallback path needs it.
fn uniform_wf_step(n: usize, window_size: usize, num_windows: usize) -> usize {
    if num_windows <= 1 || window_size >= n {
        return window_size.max(1);
    }
    ((n - window_size) / (num_windows - 1)).max(1)
}

/// Resolve the actual list of walk-forward window start offsets, ascending. Each window spans
/// `[offset, offset + window_size)`; its TEST slice is the last `test_size` bars of that span.
///
/// When `regimes` is `Some`, up to one window per DISTINCT regime label actually present in the
/// data (Low/Medium/High -- there are only ever three) is reserved first, scarcest label first --
/// so a common regime's placement can never accidentally consume the only viable slot for a scarce
/// one -- before the remaining window budget is filled with the existing uniform, back-to-back
/// spacing scheme.
///
/// `regimes: None` reproduces the prior uniform-only placement exactly -- no behavior change for
/// any caller that doesn't pass regime data. This is the load-bearing contract this port must
/// preserve; see `tests/regime_stratified_folds.rs`.
///
/// Windows are kept mutually disjoint throughout (both the regime-reserved ones and the uniform
/// fill) -- overlapping OOS test windows correlate their errors and overstate how much independent
/// evidence walk-forward actually gathered. This is a reasonable, not perfectly optimal, interval
/// packer: a regime segment near the very end of the data can still get clipped by the `max_offset`
/// bound, and a real-but-too-short segment (fewer bars than `test_size` could ever fit) is skipped
/// rather than forced.
///
/// Faithful port (see the module-level note above) of
/// `backtest::python_validation::resolve_wf_window_offsets`; do not "improve" the placement logic
/// here without also updating that function, or the two diverge silently.
pub fn resolve_wf_window_offsets(
    n: usize,
    window_size: usize,
    num_windows: usize,
    test_size: usize,
    regimes: Option<&[Option<RegimeLabel>]>,
) -> Vec<usize> {
    if num_windows == 0 || window_size == 0 || window_size > n {
        return Vec::new();
    }
    let max_offset = n - window_size;
    let uniform_step = uniform_wf_step(n, window_size, num_windows);

    let Some(regimes) = regimes else {
        return (0..num_windows).map(|i| (i * uniform_step).min(max_offset)).collect();
    };

    use RegimeLabel::{High, Low, Medium};
    let mut present: Vec<RegimeLabel> =
        [Low, Medium, High].into_iter().filter(|label| regimes.iter().any(|r| *r == Some(*label))).collect();
    present.sort_by_key(|label| regimes.iter().filter(|r| **r == Some(*label)).count());

    let segments = regime_segments(regimes);
    let overlaps =
        |claimed: &[(usize, usize)], start: usize, end: usize| claimed.iter().any(|(cs, ce)| start < *ce && end > *cs);

    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut reserved_offsets: Vec<usize> = Vec::new();

    for label in present.into_iter().take(num_windows) {
        // Prefer the LARGEST segment of this label -- more robust than a segment so short it's
        // likely a one-bar classifier blip.
        let mut candidates: Vec<&(usize, usize, RegimeLabel)> =
            segments.iter().filter(|(_, _, l)| *l == label).collect();
        candidates.sort_by_key(|(s, e, _)| std::cmp::Reverse(e.saturating_sub(*s)));

        for (seg_start, seg_end, _) in candidates {
            // Place the window so its test slice's midpoint sits inside the segment: test slice =
            // [offset + window_size - test_size, offset + window_size), midpoint ~= offset +
            // window_size - test_size/2. Solving for offset given a target midpoint:
            let target_mid = (seg_start + seg_end) / 2;
            let offset = (target_mid + test_size / 2).saturating_sub(window_size).min(max_offset);
            let end = offset + window_size;
            if overlaps(&claimed, offset, end) {
                continue;
            }
            claimed.push((offset, end));
            reserved_offsets.push(offset);
            break;
        }
        // No viable placement found for this (real but too-short, or fully claimed-over) regime --
        // skip it, matching the pre-existing "genuinely untestable" outcome.
    }

    let mut offsets = reserved_offsets;
    let mut candidate = 0usize;
    while offsets.len() < num_windows && candidate <= max_offset {
        let end = candidate + window_size;
        if !overlaps(&claimed, candidate, end) {
            claimed.push((candidate, end));
            offsets.push(candidate);
        }
        candidate += uniform_step.max(1);
    }

    offsets.sort_unstable();
    offsets
}

/// Regime-stratified variant of [`walk_forward_folds`]: reserves one fold's test window per
/// distinct regime label present in `regime_labels` (scarcest first) before falling back to the
/// same uniform spacing, porting `backtest::python_validation::resolve_wf_window_offsets`'s
/// placement algorithm (see the module-level note above and that function's own doc comment,
/// ported as [`resolve_wf_window_offsets`] in this module).
///
/// This is an ADDITIVE, SIBLING function, not a change to `walk_forward_folds`'s existing public
/// signature -- Rust has no optional/default parameters, so adding a parameter to
/// `walk_forward_folds` directly would force every existing caller (including the Engine's
/// `replication_robustness` module, which pins this crate by git rev) to change its call sites for
/// a capability most callers don't need yet. `walk_forward_folds` itself is untouched by this slice
/// -- not one line of its body changed.
///
/// **Contract:** `regime_labels: None` delegates directly to `walk_forward_folds` with the exact
/// same arguments -- byte-for-byte, by construction, since it is a literal call to that function,
/// not a re-implementation of it. There is zero behavior change for any caller that doesn't pass
/// regime data (and no existing caller does, since this function is new).
///
/// When `regime_labels` is `Some`, each fold still has exactly `test_len` test bars and a purge gap
/// of `purge` bars immediately before it, and training follows `mode` exactly as
/// `walk_forward_folds` does. The per-fold train span used to size each reserved placement window
/// is `min_train` under `TrainMode::Expanding` (the floor every expanding fold is guaranteed to
/// have) and `train_len` under `TrainMode::Rolling`. The reserved `[offset, offset + window_size)`
/// spans `resolve_wf_window_offsets` places are always kept mutually disjoint; under
/// `TrainMode::Rolling` the realized train range matches that reserved span exactly, so a fold's
/// full train+purge+test window stays disjoint from every other fold's. Under
/// `TrainMode::Expanding`, training always reaches back to bar 0 -- exactly as it already does in
/// `walk_forward_folds` -- so per-fold TRAIN ranges can and do overlap across folds (only the TEST
/// placement, and the purge gap immediately before it, are new/regime-aware; expanding folds have
/// always shared their `[0, ..)` prefix).
///
/// Returns a typed error (never a panic, never a shorter-than-requested list of folds) if there
/// isn't room for `n_folds` mutually disjoint windows of the required size, mirroring this crate's
/// P2 "refuse, never default" discipline.
pub fn walk_forward_folds_stratified(
    n_bars: usize,
    n_folds: usize,
    test_len: usize,
    min_train: usize,
    purge: usize,
    mode: TrainMode,
    regime_labels: Option<&[Option<RegimeLabel>]>,
) -> Result<Vec<Fold>> {
    let Some(regimes) = regime_labels else {
        return walk_forward_folds(n_bars, n_folds, test_len, min_train, purge, mode);
    };

    if n_folds == 0 {
        return Err(invalid("n_folds", "must be at least 1"));
    }
    if test_len == 0 {
        return Err(invalid("test_len", "must be at least 1"));
    }
    if min_train == 0 {
        return Err(invalid("min_train", "must be at least 1"));
    }
    let train_span = match mode {
        TrainMode::Expanding => min_train,
        TrainMode::Rolling { train_len } => {
            if train_len < min_train {
                return Err(invalid(
                    "train_len",
                    format!("rolling train_len {train_len} is below min_train {min_train}"),
                ));
            }
            train_len
        }
    };
    let window_size = train_span
        .checked_add(purge)
        .and_then(|v| v.checked_add(test_len))
        .ok_or_else(|| invalid("purge", "sizes overflow usize"))?;
    let need =
        n_folds.checked_mul(window_size).ok_or_else(|| invalid("n_folds", "n_folds x window_size overflows usize"))?;
    if window_size > n_bars || need > n_bars {
        return Err(EvalError::TooShort {
            what: "regime-stratified walk-forward folds",
            need: need.max(window_size),
            got: n_bars,
        });
    }

    let offsets = resolve_wf_window_offsets(n_bars, window_size, n_folds, test_len, Some(regimes));
    if offsets.len() < n_folds {
        return Err(EvalError::TooShort {
            what: "regime-stratified walk-forward folds (placement)",
            need: n_folds,
            got: offsets.len(),
        });
    }

    let mut folds = Vec::with_capacity(n_folds);
    for offset in offsets {
        let test_end = offset + window_size;
        let test_start = test_end - test_len;
        let train_end = test_start - purge;
        let train_start = match mode {
            TrainMode::Expanding => 0,
            TrainMode::Rolling { train_len } => train_end.saturating_sub(train_len),
        };
        folds.push(Fold { train: vec![train_start..train_end], test: vec![test_start..test_end] });
    }
    Ok(folds)
}

#[cfg(test)]
mod regime_placement_tests {
    use super::*;

    fn make_regimes(spec: &[(usize, RegimeLabel)]) -> Vec<Option<RegimeLabel>> {
        spec.iter().flat_map(|(count, label)| std::iter::repeat(Some(*label)).take(*count)).collect()
    }

    // Mirrors `backtest::python_validation::regime_segments_splits_on_label_change_and_skips_none_gaps`
    // (that helper is private there too, hence the inline test here rather than in `tests/`).
    #[test]
    fn regime_segments_splits_on_label_change_and_skips_none_gaps() {
        use RegimeLabel::{High, Low, Medium};
        let mut regimes = make_regimes(&[(5, Low), (3, High), (4, Medium)]);
        // Insert a None gap inside the Low run -- must not split it.
        regimes[2] = None;
        let segments = regime_segments(&regimes);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].2, Low);
        assert_eq!(segments[0].0, 0);
        assert_eq!(segments[0].1, 4); // still spans the whole Low run despite the gap at index 2
        assert_eq!(segments[1].2, High);
        assert_eq!(segments[2].2, Medium);
    }
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
