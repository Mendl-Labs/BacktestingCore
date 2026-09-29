//! Regime-stratified window placement (slice A1-1 of
//! `product-mandate/SHARED_VERIFICATION_INFRASTRUCTURE_PLAN.md`, section 2.1 item A1).
//!
//! This file mirrors `backtest::python_validation.rs`'s own `resolve_wf_window_offsets` test suite
//! (same scenarios, same fixture shapes, ported to `RegimeLabel`) to confirm the ported
//! `portfolio_eval::folds::resolve_wf_window_offsets` produces equivalent placement decisions, plus
//! new tests for the crate's own `walk_forward_folds_stratified` entry point -- in particular the
//! load-bearing "no behavior change when `regime_labels` is `None`" contract.

use portfolio_eval::error::EvalError;
use portfolio_eval::folds::{resolve_wf_window_offsets, walk_forward_folds, walk_forward_folds_stratified, RegimeLabel, TrainMode};

fn make_regimes(spec: &[(usize, RegimeLabel)]) -> Vec<Option<RegimeLabel>> {
    spec.iter().flat_map(|(count, label)| std::iter::repeat(Some(*label)).take(*count)).collect()
}

// ---------------------------------------------------------------------------------------------
// Golden vectors mirroring `backtest::python_validation`'s `resolve_wf_window_offsets` test suite.
// ---------------------------------------------------------------------------------------------

// Mirrors `resolve_wf_window_offsets_none_regimes_matches_prior_uniform_behavior`.
#[test]
fn resolve_wf_window_offsets_none_regimes_matches_prior_uniform_behavior() {
    let n = 10_000;
    let num_windows = 8;
    let window_size = n / num_windows;
    // Hand-computed uniform step, independent of `uniform_wf_step` (private): windows spread so the
    // last one's end reaches `n`.
    let step = (n - window_size) / (num_windows - 1);
    let expected: Vec<usize> = (0..num_windows).map(|i| (i * step).min(n - window_size)).collect();
    assert_eq!(resolve_wf_window_offsets(n, window_size, num_windows, window_size / 3, None), expected);
}

// Mirrors `resolve_wf_window_offsets_places_a_window_in_every_present_regime`.
#[test]
fn resolve_wf_window_offsets_places_a_window_in_every_present_regime() {
    use RegimeLabel::{High, Low, Medium};
    // 900 bars: a long Low run, then a short High spike, then Medium -- uniform spacing alone (3
    // evenly-spaced windows) would very plausibly miss the short High segment entirely.
    let regimes = make_regimes(&[(600, Low), (60, High), (240, Medium)]);
    let n = regimes.len();
    let window_size = 150;
    let test_size = 45;
    let num_windows = 3;
    let offsets = resolve_wf_window_offsets(n, window_size, num_windows, test_size, Some(&regimes));

    let mut covered: Vec<RegimeLabel> = Vec::new();
    for &offset in &offsets {
        let test_start = offset + window_size - test_size;
        let test_end = offset + window_size;
        let mid = (test_start + test_end) / 2;
        if let Some(label) = regimes.get(mid.min(n - 1)).copied().flatten() {
            if !covered.contains(&label) {
                covered.push(label);
            }
        }
    }
    assert!(covered.contains(&Low) && covered.contains(&High) && covered.contains(&Medium), "covered={covered:?}");
}

// Mirrors `resolve_wf_window_offsets_windows_stay_disjoint_with_regimes`.
#[test]
fn resolve_wf_window_offsets_windows_stay_disjoint_with_regimes() {
    use RegimeLabel::{High, Low, Medium};
    let regimes = make_regimes(&[(300, Low), (50, High), (300, Medium), (50, Low), (300, High)]);
    let n = regimes.len();
    let window_size = 120;
    let offsets = resolve_wf_window_offsets(n, window_size, 6, 36, Some(&regimes));
    for i in 0..offsets.len() {
        for j in (i + 1)..offsets.len() {
            let (a, b) = (offsets[i], offsets[j]);
            assert!(a + window_size <= b || b + window_size <= a, "windows at {a} and {b} overlap (window_size={window_size})");
        }
    }
}

// Mirrors `resolve_wf_window_offsets_skips_a_regime_too_short_to_fit_a_test_slice`.
#[test]
fn resolve_wf_window_offsets_skips_a_regime_too_short_to_fit_a_test_slice() {
    use RegimeLabel::{High, Low};
    // A single-bar High blip can never host a 40-bar test slice -- must not panic or produce a
    // malformed offset, just quietly omit it.
    let mut regimes = make_regimes(&[(500, Low)]);
    regimes[250] = Some(High);
    let n = regimes.len();
    let offsets = resolve_wf_window_offsets(n, 120, 4, 40, Some(&regimes));
    assert!(!offsets.is_empty());
    for &offset in &offsets {
        assert!(offset + 120 <= n);
    }
}

// Mirrors `resolve_wf_window_offsets_empty_regimes_falls_back_to_uniform`.
#[test]
fn resolve_wf_window_offsets_empty_regimes_falls_back_to_uniform() {
    let n = 1000;
    let window_size = 200;
    let num_windows = 4;
    let empty_regimes: Vec<Option<RegimeLabel>> = vec![None; n];
    let with_none_regimes = resolve_wf_window_offsets(n, window_size, num_windows, 60, Some(&empty_regimes));
    let with_no_regimes = resolve_wf_window_offsets(n, window_size, num_windows, 60, None);
    assert_eq!(with_none_regimes, with_no_regimes);
}

// Mirrors `resolve_wf_window_offsets_zero_window_size_or_num_windows_returns_empty`.
#[test]
fn resolve_wf_window_offsets_zero_window_size_or_num_windows_returns_empty() {
    assert!(resolve_wf_window_offsets(1000, 0, 4, 10, None).is_empty());
    assert!(resolve_wf_window_offsets(1000, 100, 0, 10, None).is_empty());
    assert!(resolve_wf_window_offsets(100, 500, 4, 10, None).is_empty());
}

// ---------------------------------------------------------------------------------------------
// `walk_forward_folds_stratified`'s own contract: the load-bearing "None is a no-op" guarantee,
// plus Fold-level regime coverage and disjointness.
// ---------------------------------------------------------------------------------------------

/// THE load-bearing contract: `walk_forward_folds_stratified(.., None)` must reproduce
/// `walk_forward_folds`'s exact output -- byte-for-byte -- across a range of fixed inputs,
/// including inputs that make `walk_forward_folds` itself error. This is checked directly (not
/// just "trust the delegation"), covering both the Ok and Err paths.
#[test]
fn walk_forward_folds_stratified_none_is_byte_for_byte_identical_to_walk_forward_folds() {
    let cases: Vec<(usize, usize, usize, usize, usize, TrainMode)> = vec![
        (100, 4, 10, 20, 5, TrainMode::Expanding),
        (100, 4, 10, 20, 5, TrainMode::Rolling { train_len: 30 }),
        (200, 3, 20, 10, 0, TrainMode::Expanding),
        (200, 3, 20, 10, 17, TrainMode::Expanding),
        (55, 3, 10, 20, 5, TrainMode::Expanding),
        (54, 3, 10, 20, 5, TrainMode::Expanding), // deliberately one bar short -> Err
        (40, 2, 10, 20, 0, TrainMode::Expanding),
        (100, 0, 10, 20, 5, TrainMode::Expanding),                    // bad n_folds -> Err
        (100, 4, 0, 20, 5, TrainMode::Expanding),                     // bad test_len -> Err
        (100, 4, 10, 0, 5, TrainMode::Expanding),                     // bad min_train -> Err
        (100, 4, 10, 20, 5, TrainMode::Rolling { train_len: 10 }),    // train_len < min_train -> Err
        (usize::MAX, usize::MAX, 2, 1, 0, TrainMode::Expanding),      // overflow -> Err
    ];
    for (n_bars, n_folds, test_len, min_train, purge, mode) in cases {
        let before = walk_forward_folds(n_bars, n_folds, test_len, min_train, purge, mode);
        let after = walk_forward_folds_stratified(n_bars, n_folds, test_len, min_train, purge, mode, None);
        assert_eq!(
            before, after,
            "n_bars={n_bars} n_folds={n_folds} test_len={test_len} min_train={min_train} purge={purge} mode={mode:?}"
        );
    }
}

/// Same contract, property-style over many random-ish parameter combinations (same generator shape
/// as `tests/folds.rs`'s own `property_walk_forward_never_overlaps_and_respects_purge`).
#[test]
fn walk_forward_folds_stratified_none_matches_walk_forward_folds_over_many_inputs() {
    let mut checked = 0;
    for n in [50usize, 77, 120, 200, 401] {
        for k in 1..=6usize {
            for tl in [1usize, 5, 10, 29] {
                for mt in [1usize, 7, 19] {
                    for purge in [0usize, 3, 9] {
                        for mode in [TrainMode::Expanding, TrainMode::Rolling { train_len: mt + 12 }] {
                            let before = walk_forward_folds(n, k, tl, mt, purge, mode);
                            let after = walk_forward_folds_stratified(n, k, tl, mt, purge, mode, None);
                            assert_eq!(before, after, "n={n} k={k} tl={tl} mt={mt} purge={purge} mode={mode:?}");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 500);
}

/// With real, distinct regimes present, `walk_forward_folds_stratified` places at least one fold's
/// TEST window in each present regime -- the Fold-level analogue of
/// `resolve_wf_window_offsets_places_a_window_in_every_present_regime`.
#[test]
fn walk_forward_folds_stratified_reserves_a_fold_per_present_regime() {
    use RegimeLabel::{High, Low, Medium};
    let regimes = make_regimes(&[(600, Low), (60, High), (240, Medium)]);
    let n_bars = regimes.len();
    let folds = walk_forward_folds_stratified(n_bars, 3, 45, 60, 5, TrainMode::Expanding, Some(&regimes)).unwrap();
    assert_eq!(folds.len(), 3);

    let mut covered: Vec<RegimeLabel> = Vec::new();
    for f in &folds {
        let t = &f.test[0];
        let mid = (t.start + t.end) / 2;
        if let Some(label) = regimes.get(mid.min(n_bars - 1)).copied().flatten() {
            if !covered.contains(&label) {
                covered.push(label);
            }
        }
    }
    assert!(covered.contains(&Low) && covered.contains(&High) && covered.contains(&Medium), "covered={covered:?}");
}

/// Folds built from regime-stratified placement still respect `test_len`, the `purge` gap, and
/// `mode`'s train-window shape exactly like `walk_forward_folds` -- only the PLACEMENT differs.
/// TEST windows are always mutually disjoint; full train+test windows are only guaranteed disjoint
/// under `TrainMode::Rolling` (under `Expanding`, training always reaches back to bar 0, so train
/// ranges legitimately overlap across folds -- exactly as they already do in `walk_forward_folds`,
/// see `walk_forward_hand_computed_expanding`'s own golden vector).
#[test]
fn walk_forward_folds_stratified_folds_respect_test_len_purge_and_mode() {
    use RegimeLabel::{High, Low, Medium};
    let regimes = make_regimes(&[(300, Low), (50, High), (300, Medium), (50, Low), (300, High)]);
    let n_bars = regimes.len();
    let test_len = 36;
    let purge = 7;
    let min_train = 40;
    for mode in [TrainMode::Expanding, TrainMode::Rolling { train_len: 60 }] {
        let folds = walk_forward_folds_stratified(n_bars, 4, test_len, min_train, purge, mode, Some(&regimes)).unwrap();
        for f in &folds {
            let (tr, te) = (&f.train[0], &f.test[0]);
            assert_eq!(te.end - te.start, test_len);
            assert_eq!(te.start - tr.end, purge);
            assert!(tr.end - tr.start >= min_train);
            match mode {
                TrainMode::Expanding => assert_eq!(tr.start, 0),
                TrainMode::Rolling { train_len } => assert!(tr.end - tr.start <= train_len),
            }
        }
        // TEST windows are always mutually disjoint.
        for i in 0..folds.len() {
            for j in (i + 1)..folds.len() {
                let (a, b) = (&folds[i].test[0], &folds[j].test[0]);
                assert!(a.end <= b.start || b.end <= a.start, "test windows {i} and {j} overlap under mode {mode:?}");
            }
        }
        // Full train+test windows are additionally disjoint under Rolling, where the realized train
        // span matches the reserved placement window exactly.
        if matches!(mode, TrainMode::Rolling { .. }) {
            for i in 0..folds.len() {
                for j in (i + 1)..folds.len() {
                    let (a, b) = (&folds[i], &folds[j]);
                    let (a_start, a_end) = (a.train[0].start, a.test[0].end);
                    let (b_start, b_end) = (b.train[0].start, b.test[0].end);
                    assert!(a_end <= b_start || b_end <= a_start, "folds {i} and {j} overlap under mode {mode:?}");
                }
            }
        }
    }
}

/// A regime segment too short to ever host a test slice is quietly skipped, not forced -- and the
/// call still succeeds with a uniform-filled remainder, mirroring
/// `resolve_wf_window_offsets_skips_a_regime_too_short_to_fit_a_test_slice`.
#[test]
fn walk_forward_folds_stratified_skips_a_regime_too_short_to_fit_a_test_slice() {
    use RegimeLabel::{High, Low};
    let mut regimes = make_regimes(&[(500, Low)]);
    regimes[250] = Some(High);
    let n_bars = regimes.len();
    let folds = walk_forward_folds_stratified(n_bars, 4, 40, 20, 5, TrainMode::Expanding, Some(&regimes)).unwrap();
    assert_eq!(folds.len(), 4);
    for f in &folds {
        assert!(f.test[0].end <= n_bars);
    }
}

/// `walk_forward_folds_stratified`'s Fold-building is a faithful wrapper over
/// `resolve_wf_window_offsets`: every fold's `[offset, offset + window_size)` reserved span (with
/// `window_size = min_train + purge + test_len` under `TrainMode::Expanding`) must come directly
/// from that function's own offsets, for regimes ranging from "none present" (mirroring
/// `resolve_wf_window_offsets_empty_regimes_falls_back_to_uniform`) to several distinct labels.
/// This is deliberately NOT a comparison against `regime_labels: None` (which takes the entirely
/// separate, unchanged `walk_forward_folds` code path -- see the load-bearing contract test above);
/// it is a check that the `Some(..)` path is wired to the ported placement function correctly.
#[test]
fn walk_forward_folds_stratified_offsets_match_resolve_wf_window_offsets_directly() {
    use RegimeLabel::{High, Low, Medium};
    let n_bars = 1000;
    let (n_folds, test_len, min_train, purge) = (4, 60, 40, 5);
    let window_size = min_train + purge + test_len;
    let scenarios: Vec<Vec<Option<RegimeLabel>>> = vec![
        vec![None; n_bars],
        make_regimes(&[(1000, Low)]),
        make_regimes(&[(400, Low), (300, Medium), (300, High)]),
    ];
    for regimes in scenarios {
        let expected_offsets = resolve_wf_window_offsets(n_bars, window_size, n_folds, test_len, Some(&regimes));
        let folds =
            walk_forward_folds_stratified(n_bars, n_folds, test_len, min_train, purge, TrainMode::Expanding, Some(&regimes))
                .unwrap();
        let actual_offsets: Vec<usize> = folds.iter().map(|f| f.test[0].end - window_size).collect();
        assert_eq!(actual_offsets, expected_offsets);
    }
}

/// Bad parameters refuse with a typed error under the regime path too -- never a panic, never a
/// shorter-than-requested fold list (this crate's P2 discipline).
#[test]
fn walk_forward_folds_stratified_bad_parameters_refuse_with_typed_errors() {
    let regimes = make_regimes(&[(100, RegimeLabel::Low)]);
    assert!(matches!(
        walk_forward_folds_stratified(100, 0, 10, 20, 5, TrainMode::Expanding, Some(&regimes)),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        walk_forward_folds_stratified(100, 4, 0, 20, 5, TrainMode::Expanding, Some(&regimes)),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        walk_forward_folds_stratified(100, 4, 10, 0, 5, TrainMode::Expanding, Some(&regimes)),
        Err(EvalError::InvalidParameter { .. })
    ));
    assert!(matches!(
        walk_forward_folds_stratified(100, 4, 10, 20, 5, TrainMode::Rolling { train_len: 10 }, Some(&regimes)),
        Err(EvalError::InvalidParameter { .. })
    ));
    // Not enough room for 4 disjoint windows of the required size.
    assert!(matches!(
        walk_forward_folds_stratified(50, 20, 10, 20, 5, TrainMode::Expanding, Some(&regimes)),
        Err(EvalError::TooShort { .. })
    ));
}
