//! `cointegration_spread_threshold`: EXACTLY two assets, daily. Trades the same log-price spread, z-score,
//! enter/exit/hold mechanics as the immediately preceding sibling primitive, `pairs_zscore_meanreversion`
//! (hereafter "#5") -- see that module's doc for the full derivation of the state machine and the statefulness
//! resolution (replay, not mutable state). The ONE structural difference, per this primitive's own spec: the
//! Johansen-style hedge ratio is NOT recomputed fresh on every bar. Instead it is computed ONCE from the `L`
//! bars strictly BEFORE a refresh point, then held FIXED across the next `L` bars of decisions, and refreshed
//! only every `L` bars thereafter. Default lookback `L` = 90 trading days ([`COINT_LOOKBACK_DAYS`]), default
//! entry threshold 2.0 ([`COINT_ENTRY_THRESHOLD`]), exit threshold 0.5 ([`COINT_EXIT_THRESHOLD`], choice 1 below).
//!
//! Like #5 (and `inverse_volatility_weight` / `low_vol_quintile_tilt`), this primitive implements
//! `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView`, as a self-contained copy -- `#5`'s file is
//! not imported or modified.
//!
//! # The statefulness problem
//! Identical to #5: `target_weights(&self, h)` carries no previous-position argument, so "otherwise HOLD the
//! current position unchanged" is resolved by replaying the deterministic state machine over the full causal
//! history in `h` on every call, starting from an assumed-flat position, and returning only the final state. See
//! #5's module doc for the full argument; it applies here unchanged.
//!
//! # The refresh-schedule problem (the one genuinely new structural piece)
//! There is no mutable state to remember "the last bar the hedge ratio was refreshed on" across calls -- exactly
//! the same constraint that makes the position itself stateless-by-replay. The refresh schedule must therefore be
//! a deterministic, pure function of the bar index alone, so that re-deriving it from scratch on every call (as
//! part of the replay) always reproduces the same answer a persisted "last refresh" counter would have given.
//!
//! The schedule used here: anchor refresh point `r0 = L` (0-indexed bar; the earliest bar with a full `L`-bar
//! window strictly BEFORE it, bars `[0, L-1]`). Refresh points thereafter are every `L` bars: `r0, r0+L, r0+2L,
//! ...`. For a decision at bar `s` (`s >= L`), the APPLICABLE refresh point is
//! `r(s) = L * (s / L)` (integer division), i.e. the largest multiple of `L` that is `<= s`. The hedge ratio
//! used for bar `s`'s decision is computed once from bars `[r(s) - L, r(s) - 1]` and is identical for every bar
//! in the half-open cycle `[r(s), r(s) + L)` -- it changes only when `s` crosses into the next multiple of `L`.
//! This is a closed-form function of `s` and `L` alone (no recursion, no persisted counter), so it reproduces,
//! bar for bar, exactly what an external "refresh every L bars" counter would have produced, while staying a pure
//! function of `h` as the trait requires. [`CointegrationSpreadThresholdRule::refresh_point`] is this function;
//! [`CointegrationSpreadThresholdRule::hedge_ratio_for_refresh`] computes the ratio for a given refresh point.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Single named threshold, resolved as entry-only; exit threshold reused from #5 by analogy (the genuine
//!    ambiguity the task calls out).* The spec names exactly one number, "entry absolute-spread-deviation
//!    threshold (default: 2.0 standard deviations)," immediately followed by "Same entry/exit/hold logic as #5."
//!    Two readings were considered: (a) no separate exit threshold -- exit exactly at `z == 0`, or symmetrically
//!    at the SAME 2.0 value; or (b) reuse #5's `exit_threshold = 0.5` by analogy. Reading (a) with the exit value
//!    equal to the entry value was rejected because it collapses the HOLD branch to an empty set: #5's transition
//!    is `if z < -entry: enter-long; elif z > entry: enter-short; elif |z| < exit: exit-flat; else: hold`. If
//!    `exit == entry`, those first three branches are already exhaustive for every `z` except a measure-zero
//!    boundary, so "hold" -- a state the spec explicitly names as part of "the same entry/exit/hold logic as #5"
//!    -- could (almost) never be reached. Exit-at-exactly-zero was also rejected: it is an even narrower
//!    measure-zero condition, so in practice every bar once in a position would just hold forever (a position,
//!    once entered, can only ever flatten on the literally-impossible floating-point coincidence `z == 0.0`),
//!    which is not a genuine "exit" branch either. Reading (b) is the only one of the three that preserves an
//!    actual, non-degenerate hold region bounded strictly between two distinct thresholds -- i.e. it is the only
//!    reading that actually matches "the SAME entry/exit/hold logic as #5," not just the entry number -- so this
//!    implementation sets [`COINT_EXIT_THRESHOLD`] = 0.5, identical to `PAIRS_EXIT_THRESHOLD`.
//! 2. *Log-price spread uses natural log* (`f64::ln`), same as #5, per the spec's shared notation.
//! 3. *Hedge ratio is the OLS slope of `ln(close_A)` regressed ON `ln(close_B)`* (A dependent, B regressor,
//!    `beta = cov(lnA, lnB) / var(lnB)`), same regression direction as #5 -- the spec does not ask for a
//!    different one, only a different REFRESH CADENCE, so the regression itself is carried over unchanged.
//! 4. *Window-construction for the z-score, carried over from #5's choice 3, but now spec-mandated rather than
//!    merely the simplest reading.* #5 had to choose, as a judgment call, whether the `L` historical spread
//!    values inside the trailing mean/stdev window should each be reconstructed with their OWN bar's
//!    freshly-recomputed hedge ratio, or uniformly with the ONE ratio computed at the window's own evaluation
//!    bar. This primitive's spec removes that judgment call entirely: it explicitly says the hedge ratio is
//!    "held fixed" across `L` bars of decisions, so there is only one ratio available for the whole z-score
//!    window in any case. Every spread value inside a given decision's trailing `L`-bar z-score window
//!    (`[s - L + 1, s]`) is reconstructed with that one cycle-fixed ratio.
//! 5. *Population stdev (ddof = 0)*, same as #5 and every other lookback-based primitive in this crate.
//! 6. *Minimum history is exactly `L + 1` bars, and here that is the TIGHT bound, not a one-bar buffer like #5's.*
//!    #5's `L + 1` came from the spec's own literal text, one bar more than its `L`-bar arithmetic strictly
//!    needed. Here the arithmetic itself needs `L + 1`: the first decision bar is `s = L` (0-indexed), which
//!    needs BOTH the hedge-ratio window `[0, L - 1]` (`L` bars) AND the z-score window `[1, L]` (`L` bars); the
//!    union of those two windows is bars `[0, L]`, i.e. `L + 1` bars, with no bar unused. Per the task's
//!    instruction to use "the same insufficient-history convention as #5," [`CointegrationSpreadThresholdRule::
//!    min_history_bars`] returns `L + 1` -- the same FORMULA as #5, even though the justification underneath it
//!    differs (tight here, buffered there).
//! 7. *Insufficient history = silent skip, not a refusal -- same dual mechanism as #5.* Primary: `min_history_bars
//!    () == L + 1` means `weightsim::simulate()` never calls this rule before that many bars are visible (silent
//!    skip, not a refusal). Secondary, defensive: `target_weights` also checks `h.closes(i).len()` directly and
//!    returns `Ok(vec![0.0, 0.0])` (flat) rather than `Err` if either leg is short, for a direct caller that
//!    bypasses `min_history_bars`. As in #5, `HistoryView`'s rectangular panel makes this per-asset branch
//!    structurally unreachable through the public `simulate()` API with exactly two assets; it is kept anyway as
//!    a free, literal safety net.
//! 8. *A degenerate hedge-ratio regression OR a degenerate z-score window, at any bar during replay, is "no
//!    signal, hold" -- never a refusal; same rationale as #5's choice 7, with one extra consequence worth stating
//!    explicitly.* The `L`-bar regression window can have exactly zero variance in `ln(close_B)` (hedge ratio
//!    undefined), and the z-score window's reconstructed spread can likewise have exactly zero stdev. Both are
//!    treated as "no signal" for the affected bar(s), holding whatever position preceded them -- never a typed
//!    refusal, for the same "would refuse forever once baked into replay history" argument #5 gives. The NEW
//!    consequence of "held fixed": if the regression AT A REFRESH POINT is degenerate (hedge ratio `None`), then
//!    EVERY bar in that entire upcoming `L`-bar cycle has no usable ratio and therefore no z-score, so the state
//!    machine holds for the WHOLE cycle, not just one bar -- a direct, intended structural consequence of
//!    computing the ratio once per cycle rather than once per bar.
//! 9. *Entering vs. holding at the exact threshold values*: identical strict-inequality convention to #5's
//!    choice 8 (`z < -entry`, `z > +entry`, `|z| < exit`; exact equality at any boundary falls through to hold).
//! 10. *`decision_schedule()` is `Daily`*, per the spec's explicit "daily" (same as #5).
//! 11. *`rebalance_policy()` is `OnDecision`, re-derived independently (not merely copied from #5's conclusion).*
//!     This rule's replay produces a genuinely new target-weight value ONLY on a bar where the state machine
//!     transitions (enter, exit or flip); every bar in between -- including every bar where the hedge ratio is
//!     simply being held fixed rather than refreshed -- returns the prior call's value bit-for-bit, because a
//!     held-fixed ratio that still produces a z-score inside the hysteresis band takes the explicit "hold"
//!     branch. Under `RebalancePolicy::EveryBar`, the simulator would re-trade back to the standing `+-0.5`
//!     weights on every one of those non-transition bars purely because the (`Daily`) schedule asks for a
//!     decision every bar -- a real, unnecessary round-trip trade on a bar where nothing changed, which
//!     contradicts "hold the current position unchanged" as directly as it did for #5. `RebalancePolicy::
//!     OnDecision` ("trade to target only on the bar a new target becomes effective; units drift in between")
//!     re-trades exactly on transition bars and lets the position drift through every hold, matching the spec.
//! 12. *Universe is a placeholder pair: [`COINT_SYMBOLS`] = `["EWA", "EWC"]`* (iShares MSCI Australia / iShares
//!     MSCI Canada), the canonical Johansen-cointegration teaching pair (Ernest Chan, *Algorithmic Trading*, uses
//!     exactly this pair to illustrate a Johansen-estimated hedge ratio), chosen deliberately DIFFERENT from #5's
//!     KO/PEP placeholder so the two sibling primitives are visibly distinguishable in any registry or log, and
//!     chosen BECAUSE of the "Johansen-style" wording in this primitive's own spec. As with #5's KO/PEP, this is a
//!     placeholder for a generic two-asset primitive, not a verified cointegration claim and not tied to one
//!     documented sleeve; a caller higher up the stack assigns the real production pair later.
//! 13. *Weights are fixed-magnitude, not vol-scaled*: `+-0.5` on each leg when in a position, `0.0`/`0.0` when
//!     flat -- identical to #5, a fixed dollar-neutral (gross 1.0) pairs book.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference pair: iShares MSCI Australia / iShares MSCI Canada (choice 12 above) -- the standard
/// Johansen-cointegration teaching pair. A placeholder two-asset universe so this generic primitive compiles and
/// is testable; not a documented sleeve and not a verified cointegration claim.
pub const COINT_SYMBOLS: [&str; 2] = ["EWA", "EWC"];

/// Lookback `L`, in trading days (default 90). Governs BOTH (a) the width of the hedge-ratio regression window
/// and (b) the refresh cadence -- the hedge ratio is recomputed only once every `L` bars, not every bar (the key
/// structural difference from #5; see the module doc's "refresh-schedule problem" section). Also the width of the
/// trailing z-score window, same as #5.
pub const COINT_LOOKBACK_DAYS: usize = 90;

/// Entry z-score threshold (default 2.0, the spec's one named number): enter a position when `|z| > COINT_ENTRY_THRESHOLD`.
pub const COINT_ENTRY_THRESHOLD: f64 = 2.0;

/// Exit z-score threshold (default 0.5): exit to flat when `|z| < COINT_EXIT_THRESHOLD` while in a position.
/// Reused from #5 by analogy -- the spec names only the entry value; see module doc choice 1 for why this, not a
/// symmetric 2.0 or an exact-zero exit, is the reading that actually preserves "the same ... hold logic as #5."
pub const COINT_EXIT_THRESHOLD: f64 = 0.5;

/// The rule's internal notion of "the current position," re-derived from scratch by replay on every
/// `target_weights` call (see the module doc's "statefulness problem" section). Not persisted anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Position {
    Flat,
    /// weight_A = +0.5, weight_B = -0.5.
    LongAShortB,
    /// weight_A = -0.5, weight_B = +0.5.
    ShortALongB,
}

impl Position {
    fn weights(self) -> [f64; 2] {
        match self {
            Position::Flat => [0.0, 0.0],
            Position::LongAShortB => [0.5, -0.5],
            Position::ShortALongB => [-0.5, 0.5],
        }
    }

    /// One step of the state machine (spec's enter/exit/hold rule, choice 9 above). `z = None` means no usable
    /// z-score at this bar -- either a degenerate regression/window (choice 8), or an entire cycle whose refresh
    /// point was degenerate: treated as no signal, hold.
    fn transition(self, z: Option<f64>) -> Position {
        match z {
            None => self,
            Some(z) => {
                if z < -COINT_ENTRY_THRESHOLD {
                    Position::LongAShortB
                } else if z > COINT_ENTRY_THRESHOLD {
                    Position::ShortALongB
                } else if z.abs() < COINT_EXIT_THRESHOLD {
                    Position::Flat
                } else {
                    self
                }
            }
        }
    }
}

/// `cointegration_spread_threshold`: trade a two-asset log-price spread's z-score back to its trailing mean, with
/// hysteresis (enter at `+-COINT_ENTRY_THRESHOLD`, exit at `+-COINT_EXIT_THRESHOLD`, otherwise hold) -- the same
/// mechanics as #5 (`pairs_zscore_meanreversion`), but driven by a Johansen-style hedge ratio that is computed
/// once every `COINT_LOOKBACK_DAYS` bars and held fixed across the following cycle, rather than recomputed fresh
/// on every bar. See the module doc for the refresh-schedule derivation and every interpretation choice, in
/// particular choice 1 (the single-vs-two-threshold ambiguity) and choice 11 (`OnDecision`, not `EveryBar`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CointegrationSpreadThresholdRule;

impl CointegrationSpreadThresholdRule {
    /// The refresh point applicable to a decision at bar `s` (0-indexed, requires `s >= COINT_LOOKBACK_DAYS`):
    /// the largest multiple of `L` that is `<= s`. Pure function of `s` and `L` alone -- see the module doc's
    /// "refresh-schedule problem" section for why this reproduces, bar for bar, what a persisted "last refresh"
    /// counter would have given, without needing one.
    fn refresh_point(s: usize) -> usize {
        let l = COINT_LOOKBACK_DAYS;
        l * (s / l)
    }

    /// The hedge ratio applicable to refresh point `r` (requires `r >= COINT_LOOKBACK_DAYS`): the OLS slope of
    /// `ln(close_A)` on `ln(close_B)` over the `L` bars STRICTLY BEFORE `r`, i.e. `[r - L, r - 1]` -- never
    /// including bar `r` itself, so every bar this ratio is later applied to (`[r, r + L)`) is fully causal with
    /// respect to the bars that estimated it. Returns `None` if the regression is degenerate (choice 8 above:
    /// zero/non-finite variance in `ln(close_B)` over that window).
    fn hedge_ratio_for_refresh(closes_a: &[f64], closes_b: &[f64], r: usize) -> Option<f64> {
        let l = COINT_LOOKBACK_DAYS;
        debug_assert!(r >= l, "a refresh point must have a full L-bar window strictly before it");
        let start = r - l;
        let end = r - 1; // inclusive
        let window_a = &closes_a[start..=end];
        let window_b = &closes_b[start..=end];

        let ln_a: Vec<f64> = window_a.iter().map(|x| x.ln()).collect();
        let ln_b: Vec<f64> = window_b.iter().map(|x| x.ln()).collect();

        let l_f = l as f64;
        let mean_a = ln_a.iter().sum::<f64>() / l_f;
        let mean_b = ln_b.iter().sum::<f64>() / l_f;

        let cov_ab: f64 = ln_a.iter().zip(&ln_b).map(|(a, b)| (a - mean_a) * (b - mean_b)).sum();
        let var_b: f64 = ln_b.iter().map(|b| (b - mean_b) * (b - mean_b)).sum();
        if !(var_b.is_finite() && var_b > 0.0) {
            return None; // degenerate hedge ratio (choice 8): zero/non-finite variance in ln(close_B).
        }
        let beta = cov_ab / var_b;
        beta.is_finite().then_some(beta)
    }

    /// The z-score of bar `s`'s spread against its own trailing `COINT_LOOKBACK_DAYS`-bar window `[s - L + 1,
    /// s]`, reconstructing every spread value in that window with the single GIVEN `hedge_ratio` (choice 4 above
    /// -- the ratio is an input, never recomputed here). `closes_a`/`closes_b` must each have at least `s + 1`
    /// elements. Returns `None` if the z-score cannot be finitely computed (choice 8: zero/non-finite spread
    /// stdev in the window).
    fn zscore_at(closes_a: &[f64], closes_b: &[f64], s: usize, hedge_ratio: f64) -> Option<f64> {
        let l = COINT_LOOKBACK_DAYS;
        debug_assert!(s + 1 >= l, "caller must only evaluate bars with a full L-bar window");
        let start = s + 1 - l;
        let window_a = &closes_a[start..=s];
        let window_b = &closes_b[start..=s];

        let ln_a: Vec<f64> = window_a.iter().map(|x| x.ln()).collect();
        let ln_b: Vec<f64> = window_b.iter().map(|x| x.ln()).collect();

        let l_f = l as f64;
        let spreads: Vec<f64> = ln_a.iter().zip(&ln_b).map(|(a, b)| a - hedge_ratio * b).collect();
        let mean_spread = spreads.iter().sum::<f64>() / l_f;
        let var_spread: f64 =
            spreads.iter().map(|sp| (sp - mean_spread) * (sp - mean_spread)).sum::<f64>() / l_f; // ddof = 0
        let stdev_spread = var_spread.sqrt();
        if !(stdev_spread.is_finite() && stdev_spread > 0.0) {
            return None; // degenerate z-score (choice 8): zero/non-finite spread stdev.
        }

        let current_spread = spreads[l - 1]; // window's last element = spread at bar s.
        let z = (current_spread - mean_spread) / stdev_spread;
        z.is_finite().then_some(z)
    }
}

impl WeightRule for CointegrationSpreadThresholdRule {
    fn id(&self) -> &'static str {
        "cointegration_spread_threshold_90d"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &COINT_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("lookback_days", COINT_LOOKBACK_DAYS.to_string()),
            ("hedge_ratio_refresh_days", COINT_LOOKBACK_DAYS.to_string()),
            ("entry_threshold", COINT_ENTRY_THRESHOLD.to_string()),
            ("exit_threshold", COINT_EXIT_THRESHOLD.to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ])
    }

    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }

    fn rebalance_policy(&self) -> RebalancePolicy {
        // OnDecision, not EveryBar -- see module doc choice 11, independently re-derived for this primitive. A
        // hold (including every bar where the hedge ratio is simply being held fixed, not refreshed) must not be
        // re-traded just because the Daily schedule asks for a decision every bar.
        RebalancePolicy::OnDecision
    }

    fn min_history_bars(&self) -> usize {
        // L + 1 bars: the tight bound here (choice 6 above), not a one-bar buffer like #5's.
        COINT_LOOKBACK_DAYS + 1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let l = COINT_LOOKBACK_DAYS;
        let needed = l + 1;
        let closes_a = h.closes(0);
        let closes_b = h.closes(1);

        // Defensive, structurally-unreachable-via-simulate() per-asset check (choice 7 above): silent skip, not a
        // refusal.
        if closes_a.len() < needed || closes_b.len() < needed {
            return Ok(vec![0.0, 0.0]);
        }

        // Replay the state machine from the first bar with a full L-bar window (s = L, 0-indexed) through the
        // current (last) bar, starting from an assumed-flat position. The hedge ratio is recomputed only when the
        // applicable refresh point changes (every L bars), not on every iteration -- the direct implementation of
        // "held fixed."
        let last = h.len() - 1;
        let mut state = Position::Flat;
        let mut cached_refresh: Option<usize> = None;
        let mut cached_hedge_ratio: Option<f64> = None;

        for s in l..=last {
            let r = Self::refresh_point(s);
            if cached_refresh != Some(r) {
                cached_hedge_ratio = Self::hedge_ratio_for_refresh(closes_a, closes_b, r);
                cached_refresh = Some(r);
            }
            let z = cached_hedge_ratio.and_then(|hr| Self::zscore_at(closes_a, closes_b, s, hr));
            state = state.transition(z);
        }

        Ok(state.weights().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weightsim::{simulate, Date, Panel, SimConfig};

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    const L: usize = COINT_LOOKBACK_DAYS;

    /// Builds a two-asset panel over exactly `COINT_SYMBOLS`, on a simple synthetic ascending daily calendar
    /// starting 2020-01-01 (rolling across months as needed), mirroring #5's test helper exactly.
    fn panel_from(closes_a: Vec<f64>, closes_b: Vec<f64>) -> Panel {
        assert_eq!(closes_a.len(), closes_b.len());
        let n_bars = closes_a.len();
        let dates: Vec<Date> = (0..n_bars).map(date_for_bar).collect();
        let symbols: Vec<String> = COINT_SYMBOLS.iter().map(|s| s.to_string()).collect();
        Panel::new(symbols, dates, vec![closes_a, closes_b]).unwrap()
    }

    /// Bar `i`'s synthetic calendar date: one calendar day per bar starting 2020-01-01, rolling month to month in
    /// 28-day blocks (test-only convenience; `Panel`/`HistoryView` only require strictly ascending dates).
    fn date_for_bar(i: usize) -> Date {
        let total_day = i as u32;
        let month_offset = total_day / 28;
        let day_in_month = total_day % 28 + 1;
        let month = 1 + month_offset;
        let (y, m) = (2020 + (month - 1) / 12, (month - 1) % 12 + 1);
        d(&format!("{y:04}-{m:02}-{day_in_month:02}"))
    }

    /// Runs the full `closes_a`/`closes_b` series through `simulate()` and returns the LAST bar's target weights.
    fn last_weights(closes_a: Vec<f64>, closes_b: Vec<f64>) -> Vec<f64> {
        let p = panel_from(closes_a, closes_b);
        let r = simulate(&p, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();
        r.row(&r.target_weights, p.n_bars() - 1).to_vec()
    }

    /// Mirrors exactly what `target_weights` does for a single bar `s`, for use in calibration (test-only;
    /// production replay in `target_weights` caches this across bars in the same cycle, but recomputing it fresh
    /// per call here is simpler for a one-off calibration probe and produces an identical answer).
    fn rule_zscore_at(closes_a: &[f64], closes_b: &[f64], s: usize) -> Option<f64> {
        let r = CointegrationSpreadThresholdRule::refresh_point(s);
        let hedge_ratio = CointegrationSpreadThresholdRule::hedge_ratio_for_refresh(closes_a, closes_b, r)?;
        CointegrationSpreadThresholdRule::zscore_at(closes_a, closes_b, s, hedge_ratio)
    }

    /// A trailing-`L`-bar warmup (smooth, slightly different drifts on each leg, small deterministic wiggle) of
    /// length `n_bars`, followed by a final-bar jump added directly to the log-prices of the FINAL bar only.
    /// Mirrors #5's `warmup_then_jump` helper. Because the hedge ratio for THIS primitive is fixed from the
    /// window `[0, L - 1]` (never including the jumped bar as long as `n_bars - 1 >= L`), the jump here only ever
    /// perturbs the z-score's spread reconstruction, never the hedge-ratio regression -- which is the whole point
    /// of the structural difference under test.
    fn warmup_then_jump(n_bars: usize, jump_a: f64, jump_b: f64) -> (Vec<f64>, Vec<f64>) {
        let mut a = Vec::with_capacity(n_bars);
        let mut b = Vec::with_capacity(n_bars);
        for i in 0..n_bars {
            let wiggle = 0.001 * if i % 2 == 0 { 1.0 } else { -1.0 };
            let ln_a = 4.0 + 0.0005 * i as f64 + wiggle;
            let ln_b = 4.0 + 0.0003 * i as f64 + 0.6 * wiggle;
            a.push(ln_a.exp());
            b.push(ln_b.exp());
        }
        if n_bars > 0 {
            let last = n_bars - 1;
            let ln_a_last = a[last].ln() + jump_a;
            let ln_b_last = b[last].ln() + jump_b;
            a[last] = ln_a_last.exp();
            b[last] = ln_b_last.exp();
        }
        (a, b)
    }

    #[test]
    fn declared_parameters_reflect_the_constants() {
        let rule = CointegrationSpreadThresholdRule;
        let params = rule.declared_parameters();
        assert_eq!(params.get("lookback_days").map(|s| s.as_str()), Some("90"));
        assert_eq!(params.get("hedge_ratio_refresh_days").map(|s| s.as_str()), Some("90"));
        assert_eq!(params.get("entry_threshold").map(|s| s.as_str()), Some("2"));
        assert_eq!(params.get("exit_threshold").map(|s| s.as_str()), Some("0.5"));
        assert_eq!(rule.min_history_bars(), L + 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(rule.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(rule.universe(), &COINT_SYMBOLS);
    }

    #[test]
    fn insufficient_history_is_a_silent_skip_not_a_refusal() {
        // Exactly L bars (one short of L + 1): min_history_bars() keeps the simulator from ever calling the rule.
        let (a, b) = warmup_then_jump(L, -10.0, 0.0);
        let p = panel_from(a, b);
        let r = simulate(&p, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();
        assert!(r.decision.iter().all(|&dec| !dec), "no bar has L + 1 bars visible");
        assert!(r.refused.iter().all(|&ref_| !ref_), "a silent skip must not be recorded as a refusal");
        assert!(r.target_weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn large_negative_zscore_enters_long_a_short_b() {
        let (a, b) = warmup_then_jump(L + 5, -5.0, 0.0);
        let w = last_weights(a, b);
        assert_eq!(w, vec![0.5, -0.5], "z far below -entry_threshold must enter long-A/short-B: {w:?}");
    }

    #[test]
    fn large_positive_zscore_enters_short_a_long_b() {
        let (a, b) = warmup_then_jump(L + 5, 5.0, 0.0);
        let w = last_weights(a, b);
        assert_eq!(w, vec![-0.5, 0.5], "z far above +entry_threshold must enter short-A/long-B: {w:?}");
    }

    #[test]
    fn exits_to_flat_once_z_falls_back_inside_the_exit_band() {
        // Same calibration-by-search approach as #5's equivalent test, using `rule_zscore_at` (which mirrors the
        // rule's own fixed-hedge-ratio mechanism) instead of a per-bar-fresh z-score.
        let n_bars = L + 6;
        let enter_at = n_bars - 2;
        let exit_at = n_bars - 1;
        let (baseline, b) = warmup_then_jump(n_bars, 0.0, 0.0);
        let base_enter = baseline[enter_at];
        let base_exit = baseline[exit_at];
        let enter_jump = -5.0_f64;

        let build = |exit_jump: f64| -> Vec<f64> {
            let mut a = baseline.clone();
            a[enter_at] = (base_enter.ln() + enter_jump).exp();
            a[exit_at] = (base_exit.ln() + exit_jump).exp();
            a
        };

        let exit_jump = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0]
            .into_iter()
            .find(|&candidate| {
                let a = build(candidate);
                match rule_zscore_at(&a[..=exit_at], &b[..=exit_at], exit_at) {
                    Some(z) => z.abs() < COINT_EXIT_THRESHOLD,
                    None => false,
                }
            })
            .expect("calibration must find a jump landing inside the exit band");

        let a = build(exit_jump);

        let p_enter = panel_from(a[..=enter_at].to_vec(), b[..=enter_at].to_vec());
        let r_enter = simulate(&p_enter, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();
        let w_enter = r_enter.row(&r_enter.target_weights, enter_at).to_vec();
        assert_eq!(w_enter, vec![0.5, -0.5], "setup must enter long-A/short-B at bar {enter_at}: {w_enter:?}");

        let p_full = panel_from(a, b);
        let r_full = simulate(&p_full, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();
        let w_exit = r_full.row(&r_full.target_weights, exit_at).to_vec();
        assert_eq!(w_exit, vec![0.0, 0.0], "z back inside the exit band must flatten the position: {w_exit:?}");
    }

    #[test]
    fn holds_the_position_unchanged_in_the_hysteresis_band() {
        let n_bars = L + 6;
        let enter_at = n_bars - 2;
        let hold_at = n_bars - 1;
        let (baseline, b) = warmup_then_jump(n_bars, 0.0, 0.0);
        let base_enter = baseline[enter_at];
        let base_hold = baseline[hold_at];
        let enter_jump = -5.0_f64;

        let build = |hold_jump: f64| -> Vec<f64> {
            let mut a = baseline.clone();
            a[enter_at] = (base_enter.ln() + enter_jump).exp();
            a[hold_at] = (base_hold.ln() + hold_jump).exp();
            a
        };

        let hold_jump = [-1.9, -1.7, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3]
            .into_iter()
            .find(|&candidate| {
                let a = build(candidate);
                match rule_zscore_at(&a[..=hold_at], &b[..=hold_at], hold_at) {
                    Some(z) => z.abs() > COINT_EXIT_THRESHOLD && z.abs() < COINT_ENTRY_THRESHOLD,
                    None => false,
                }
            })
            .expect("calibration must find a jump landing inside the hysteresis band");

        let a = build(hold_jump);
        let p = panel_from(a, b);
        let r = simulate(&p, &CointegrationSpreadThresholdRule, &SimConfig::default()).unwrap();
        let w_enter = r.row(&r.target_weights, enter_at).to_vec();
        let w_hold = r.row(&r.target_weights, hold_at).to_vec();
        assert_eq!(w_enter, vec![0.5, -0.5], "setup must enter long-A/short-B at bar {enter_at}: {w_enter:?}");
        assert_eq!(w_hold, vec![0.5, -0.5], "position must hold unchanged inside the hysteresis band: {w_hold:?}");
    }

    /// THE key structural test: proves the hedge ratio is NOT recomputed every bar, but held fixed across a full
    /// `L`-bar cycle. Builds two exactly-affine regimes -- `ln_a = (5/3) * ln_b` over bars `[0, L-1]` (regime A)
    /// and `ln_a = (3/5) * ln_b` over bars `[L, 2L-1]` (regime B) -- so the TRUE OLS beta of each window is exact
    /// (an affine relationship has zero residual, so `cov/var` lands on the affine coefficient to float
    /// precision), giving an unambiguous, exactly-known "what a fresh per-bar recompute would have produced" to
    /// compare against.
    #[test]
    fn hedge_ratio_is_fixed_across_a_full_cycle_not_recomputed_every_bar() {
        let l = L;
        let mut a = Vec::with_capacity(2 * l);
        let mut b = Vec::with_capacity(2 * l);
        // Regime A: bars [0, l-1]. ln_a is EXACTLY (5/3) * ln_b, so the true OLS beta over any sub-window fully
        // inside this regime is exactly 5/3.
        for i in 0..l {
            let ln_b = 4.0 + 0.0003 * i as f64;
            let ln_a = (5.0 / 3.0) * ln_b;
            a.push(ln_a.exp());
            b.push(ln_b.exp());
        }
        // Regime B: bars [l, 2l-1]. ln_a is EXACTLY (3/5) * ln_b -- a different, exactly-known slope.
        for j in 0..l {
            let ln_b = 5.0 + 0.0005 * j as f64;
            let ln_a = (3.0 / 5.0) * ln_b;
            a.push(ln_a.exp());
            b.push(ln_b.exp());
        }

        // s1 (first bar of the cycle) and s2 (last bar of the cycle) must resolve to the SAME refresh point.
        let s1 = l;
        let s2 = 2 * l - 1;
        assert_eq!(CointegrationSpreadThresholdRule::refresh_point(s1), l);
        assert_eq!(CointegrationSpreadThresholdRule::refresh_point(s2), l);

        // The ratio anchoring this whole cycle comes from regime A alone (window [0, l-1]).
        let fixed_ratio = CointegrationSpreadThresholdRule::hedge_ratio_for_refresh(&a, &b, l)
            .expect("regime A's window is non-degenerate");
        assert!(
            (fixed_ratio - 5.0 / 3.0).abs() < 1e-9,
            "the refresh-point ratio must equal regime A's exact beta: {fixed_ratio}"
        );

        // What a #5-style FRESH per-bar recompute would have produced at s2, using the trailing L-bar window
        // ending at s2 -- entirely inside regime B. Computed independently here (not via the rule's own
        // `hedge_ratio_for_refresh`), to avoid the comparison being circular.
        let naive_fresh_at_s2 = {
            let start = s2 + 1 - l;
            let window_a = &a[start..=s2];
            let window_b = &b[start..=s2];
            let ln_a: Vec<f64> = window_a.iter().map(|x| x.ln()).collect();
            let ln_b: Vec<f64> = window_b.iter().map(|x| x.ln()).collect();
            let l_f = l as f64;
            let mean_a = ln_a.iter().sum::<f64>() / l_f;
            let mean_b = ln_b.iter().sum::<f64>() / l_f;
            let cov_ab: f64 = ln_a.iter().zip(&ln_b).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum();
            let var_b: f64 = ln_b.iter().map(|y| (y - mean_b) * (y - mean_b)).sum();
            cov_ab / var_b
        };
        assert!(
            (naive_fresh_at_s2 - 3.0 / 5.0).abs() < 1e-9,
            "sanity: the trailing window ending at s2 really is pure regime B: {naive_fresh_at_s2}"
        );
        assert!(
            (naive_fresh_at_s2 - fixed_ratio).abs() > 0.5,
            "a fresh per-bar recompute at s2 would differ sharply from the cycle's fixed ratio, proving the \
             underlying data genuinely shifted between the two bars: fixed={fixed_ratio}, naive={naive_fresh_at_s2}"
        );

        // The rule's OWN mechanism, asked for the ratio applicable to s2's decision (via s2's own refresh
        // point), must still return the ORIGINAL regime-A ratio -- not the regime-B naive value.
        let ratio_used_for_s2 = CointegrationSpreadThresholdRule::hedge_ratio_for_refresh(
            &a,
            &b,
            CointegrationSpreadThresholdRule::refresh_point(s2),
        )
        .unwrap();
        assert_eq!(
            ratio_used_for_s2, fixed_ratio,
            "s1 and s2 share one refresh point, so they must be decided using one shared hedge ratio"
        );

        // And the cycle genuinely does end: the NEXT refresh point is different and does pick up regime B.
        let next_refresh = CointegrationSpreadThresholdRule::refresh_point(2 * l);
        assert_eq!(next_refresh, 2 * l);
        let next_ratio = CointegrationSpreadThresholdRule::hedge_ratio_for_refresh(&a, &b, next_refresh).unwrap();
        assert!(
            (next_ratio - 3.0 / 5.0).abs() < 1e-9,
            "the cycle immediately after this one must pick up regime B's ratio: {next_ratio}"
        );
    }
}
