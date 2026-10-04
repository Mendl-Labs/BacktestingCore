//! `pairs_zscore_meanreversion`: EXACTLY two assets (a defined pair), daily. Trade the log-price spread
//! `spread_t = ln(close_A,t) - hedge_ratio_t * ln(close_B,t)` back toward its own trailing mean: enter long-A/
//! short-B when the spread's z-score drops below `-entry_threshold`, enter short-A/long-B when it rises above
//! `+entry_threshold`, exit to flat when `|z| < exit_threshold` while currently in a position, and otherwise HOLD
//! the current position unchanged. Default lookback `L` = 60 trading days ([`PAIRS_LOOKBACK_DAYS`]), default entry
//! threshold 2.0 ([`PAIRS_ENTRY_THRESHOLD`]), default exit threshold 0.5 ([`PAIRS_EXIT_THRESHOLD`]).
//!
//! Like `inverse_volatility_weight` / `low_vol_quintile_tilt` (and unlike `crypto`/`etf`/`fx` in this crate), this
//! primitive implements `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView`.
//!
//! # The statefulness problem (read this first)
//! `WeightRule::target_weights` takes `&self` (not `&mut self`) and a single argument, `h: &HistoryView<'_>`,
//! which exposes only the FULL causal history of every asset from bar 0 up to and including the current decision
//! bar -- there is no "previous target weight" or "current position" argument anywhere in the trait, and a rule
//! may not carry mutable state across calls (`&self`, and `WeightRule: Send + Sync` is implemented on a unit
//! struct below with no fields). But this primitive's "otherwise HOLD the current position unchanged" clause is
//! explicitly stateful: today's decision depends on what the position already was, not purely on today's z-score.
//!
//! The resolution: since `h` already hands over every earlier bar's full history, each call re-derives the
//! position at every earlier decision bar by re-running the SAME deterministic transition rule over that earlier
//! history, rather than needing any externally persisted state. Concretely, `target_weights` replays the state
//! machine from the first bar that has enough history (`PAIRS_LOOKBACK_DAYS + 1` closes) through the current bar,
//! starting from an assumed-flat position before any decision was possible, and returns only the FINAL state from
//! that replay. This is correct (the replay is a pure, deterministic function of `h`, and `h` is a strict prefix
//! of the full series the simulator ultimately walks, so bar `s`'s replayed state here is byte-for-byte the same
//! state bar `s`'s own replay would have computed) but it is `O(bars_visible * L)` per call, so the whole backtest
//! is `O(T^2 * L)`. That is fine for a reference primitive and a unit-test-sized panel; a production-grade rule
//! with genuinely external state would instead be wrapped by a stateful adapter outside `weightsim`'s pure-function
//! `WeightRule` contract, but that is out of scope for this primitive.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Log-price spread uses natural log* (`f64::ln`), per the spec's own notation (`ln(close_A,t)`).
//! 2. *Hedge ratio is the OLS slope of `ln(close_A)` regressed ON `ln(close_B)`*, i.e. `ln(close_A) = alpha + beta
//!    * ln(close_B)`, so `beta = cov(lnA, lnB) / var(lnB)` (A is the dependent variable, B the regressor), computed
//!    fresh over the trailing `L` bars ending at the bar being evaluated -- never a single fixed hedge ratio for
//!    the whole series, per the spec's explicit instruction.
//! 3. *Window-construction ambiguity (flagged by the task, resolved here).* The spec requires recomputing the
//!    hedge ratio "fresh every bar using only the trailing L-bar window" AND z-scoring the current spread "against
//!    its own trailing L-bar mean/stdev", but does not say whether the `L` historical spread values inside that
//!    mean/stdev window should each have been computed with THEIR OWN bar's freshly-recomputed hedge ratio (which
//!    would require, at bar `t`, up to `2L - 1` bars of underlying closes: `L` bars to re-derive the hedge ratio at
//!    bar `t - L + 1`, plus `L - 1` more to do the same at each later bar in the window), or whether the simpler
//!    reading applies: build the entire trailing window's `L` spread values using the ONE hedge ratio just computed
//!    from the current bar's own `L`-bar regression, then take that window's mean/stdev. This implementation takes
//!    the SECOND, simpler and more literal reading: one hedge ratio per decision bar, applied uniformly to
//!    reconstruct all `L` spread values in that bar's own mean/stdev window. It is the most literally defensible
//!    reading of the spec text as written (the spec names exactly one hedge ratio per bar, singular, and separately
//!    names one mean/stdev window, without ever saying the window's own members should be recomputed under their
//!    own distinct hedge ratios), it is the only reading buildable from exactly `L + 1` trailing closes (matching
//!    the spec's own insufficient-history threshold below), and it avoids a second, much deeper recursive
//!    dependency on history that the spec never asks for.
//! 4. *Population stdev (ddof = 0)* for the z-score's denominator, consistent with every other lookback-based
//!    primitive in this crate: `sqrt(mean((spread - mean(spread))^2))` over exactly the trailing `L` spread values,
//!    i.e. divide by `L`, not `L - 1`. The spec says "population stdev" explicitly.
//! 5. *"Trailing L bars" needs `L` closes to form the regression and the spread window*, but the spec's own
//!    insufficient-history rule is stated as "fewer than L + 1 bars for either asset" (one bar more than the `L`
//!    the arithmetic above strictly requires). This implementation honors the spec's literal threshold (`L + 1`)
//!    rather than the tighter `L` the math would allow, both in [`PairsZscoreMeanReversionRule::min_history_bars`]
//!    and in the defensive per-asset check inside `target_weights`; the one extra buffered bar is simply unused.
//! 6. *Insufficient history = silent skip, not a refusal -- implemented via BOTH mechanisms, per the task's own
//!    menu.* Primary mechanism: [`PairsZscoreMeanReversionRule::min_history_bars`] returns `L + 1`, so under the
//!    public `weightsim::simulate()` path the simulator never calls this rule at all before that many bars are
//!    visible (the trait's own "silent skip, not a refusal" contract) -- the decision flag stays false and the
//!    target-weight row stays at its zero-initialized default, which already IS "flat, no position." Secondary,
//!    defensive mechanism: `target_weights` ALSO checks each asset's `h.closes(i).len()` independently and returns
//!    `Ok(vec![0.0, 0.0])` (flat, not an `Err`) if either is short, for a direct caller that bypasses
//!    `min_history_bars` (e.g. calls the trait method directly on a short `HistoryView`). As in
//!    `inverse_volatility_weight.rs` / `low_vol_quintile_tilt.rs` (choice 5 in both), `HistoryView` is one shared,
//!    rectangular panel -- `Panel::inner_join` keeps only dates common to every requested symbol, so with exactly
//!    two assets in the universe this per-asset check is all-or-nothing and structurally unreachable through the
//!    public `simulate()` API once `min_history_bars` is respected. It is kept anyway, for the same reason the
//!    sibling primitives keep theirs: it is a free, literal, zero-cost safety net.
//! 7. *Degenerate regression/z-score inputs mid-replay are treated as "no signal, hold," never a refusal.* The
//!    trailing-`L`-bar window used for the hedge ratio can have exactly zero variance in `ln(close_B)` (a frozen
//!    quote), making the OLS slope undefined (`0/0`); the resulting spread window can likewise have exactly zero
//!    stdev. The spec only names "insufficient history" as a reason to do anything other than the normal
//!    enter/exit/hold logic, so rather than inventing a new typed refusal for this failure mode, a bar whose hedge
//!    ratio or z-score cannot be finitely computed is treated, for THAT bar only, as providing no signal: the state
//!    machine holds whatever position preceded it, exactly as it would for a z-score sitting inside the
//!    exit/entry hysteresis band. This was chosen over a typed `RefusalKind::Data` refusal (the pattern
//!    `inverse_volatility_weight.rs` uses for zero/non-finite stdev) because of the replay design above: a
//!    refusal at a PAST bar inside the lookback window would, on every later call, still land on that same
//!    degenerate historical bar during replay and so would refuse the decision FOREVER, which is far too fragile
//!    for what is, going forward in real data, a rare transient data artifact rather than a structural reason to
//!    stop trading the pair. This keeps `target_weights` total for any input that clears `min_history_bars`.
//! 8. *Entering vs. holding at the exact threshold values.* The spec's inequalities are strict (`z < -entry`,
//!    `z > +entry`, `|z| < exit`), so `z` exactly equal to `-entry_threshold`, `+entry_threshold`, `exit_threshold`
//!    or `-exit_threshold` falls through to "otherwise HOLD the current position unchanged" -- implemented as a
//!    single `else` arm after the three strict checks, so equality at any boundary is never misclassified as a
//!    crossing.
//! 9. *`decision_schedule()` is `Daily`*, per the spec's explicit "daily."
//! 10. *`rebalance_policy()` is `OnDecision`, not `EveryBar` -- the genuine design choice the task calls out.*
//!     `RebalancePolicy::EveryBar` means "after every bar's mark-to-market, trade back to the standing target
//!     weights" -- under a `Daily` schedule (every bar is a decision bar) that would force a real trade back to
//!     the exact `+-0.5` weights on EVERY bar of a multi-day hold, even though the target value has not changed,
//!     which directly contradicts the spec's explicit instruction to "HOLD the current position unchanged" (a
//!     hold is a non-event: no trade, units drift with price until the next actual state change).
//!     `RebalancePolicy::OnDecision` ("trade to target only on the bar a new target becomes effective; units
//!     drift in between") is the one that matches: this rule's replay only produces a NEW target value on the bar
//!     a transition (enter/exit/flip) actually happens, and holds the prior call's return value bit-for-bit on
//!     every bar in between, so `OnDecision` re-trades exactly on transition bars and lets the position drift
//!     through the hold -- precisely "hold the current position unchanged."
//! 11. *Universe is a placeholder pair.* [`PAIRS_SYMBOLS`] = `["KO", "PEP"]` (Coca-Cola / PepsiCo), a commonly
//!     cited textbook example of a potentially cointegrated consumer-staples pair. This is a placeholder for a
//!     generic two-asset primitive, exactly as `INVERSE_VOLATILITY_SYMBOLS` / `LOW_VOL_QUINTILE_SYMBOLS` are
//!     placeholders in the sibling primitives -- not a claim that this pair is actually cointegrated in
//!     production data, and not tied to one documented sleeve. A caller higher up the stack assigns the real
//!     production pair later; that is out of scope here.
//! 12. *Weights are fixed-magnitude, not vol-scaled.* Per the spec: `+-0.5` on each leg when in a position, `0.0`/
//!     `0.0` when flat -- a fixed dollar-neutral (gross 1.0) pairs book, not sized by any volatility estimate.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference pair: Coca-Cola / PepsiCo (choice 11 above). A placeholder two-asset universe so this
/// generic primitive compiles and is testable; not a documented sleeve and not a verified cointegration claim.
pub const PAIRS_SYMBOLS: [&str; 2] = ["KO", "PEP"];

/// Lookback `L`, in trading days, of the trailing window used to (a) recompute the OLS hedge ratio fresh every
/// bar and (b) z-score the resulting spread (default 60).
pub const PAIRS_LOOKBACK_DAYS: usize = 60;

/// Entry z-score threshold (default 2.0): enter a position when `|z| > PAIRS_ENTRY_THRESHOLD`.
pub const PAIRS_ENTRY_THRESHOLD: f64 = 2.0;

/// Exit z-score threshold (default 0.5): exit to flat when `|z| < PAIRS_EXIT_THRESHOLD` while in a position.
pub const PAIRS_EXIT_THRESHOLD: f64 = 0.5;

/// The rule's internal notion of "the current position" -- the very state the task's statefulness problem is
/// about. Not persisted anywhere; re-derived from scratch by replay on every `target_weights` call (see the module
/// doc's "statefulness problem" section).
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

    /// One step of the state machine (spec's enter/exit/hold rule, choice 8 above). `z = None` means the z-score
    /// could not be finitely computed at this bar (choice 7 above): treated as no signal, hold.
    fn transition(self, z: Option<f64>) -> Position {
        match z {
            None => self,
            Some(z) => {
                if z < -PAIRS_ENTRY_THRESHOLD {
                    Position::LongAShortB
                } else if z > PAIRS_ENTRY_THRESHOLD {
                    Position::ShortALongB
                } else if z.abs() < PAIRS_EXIT_THRESHOLD {
                    Position::Flat
                } else {
                    self
                }
            }
        }
    }
}

/// `pairs_zscore_meanreversion`: trade a two-asset log-price spread's z-score back to its trailing mean, with
/// hysteresis (enter at `+-PAIRS_ENTRY_THRESHOLD`, exit at `+-PAIRS_EXIT_THRESHOLD`, otherwise hold). See the
/// module doc for the statefulness resolution (replay, not mutable state) and every interpretation choice, in
/// particular choice 3 (the hedge-ratio/z-score window ambiguity) and choice 10 (`OnDecision`, not `EveryBar`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PairsZscoreMeanReversionRule;

impl PairsZscoreMeanReversionRule {
    /// The z-score of the current (last) bar's spread against the trailing `PAIRS_LOOKBACK_DAYS`-bar window ending
    /// at `s` (0-indexed bar), using ONE hedge ratio (OLS slope of `ln(close_A)` on `ln(close_B)`) computed fresh
    /// over that same window and applied uniformly to reconstruct every spread value in the window (choice 3
    /// above). `closes_a`/`closes_b` must each have at least `s + 1` elements. Returns `None` if the hedge ratio or
    /// the z-score cannot be finitely computed (choice 7 above: zero/non-finite variance somewhere in the window).
    fn zscore_at(closes_a: &[f64], closes_b: &[f64], s: usize) -> Option<f64> {
        let l = PAIRS_LOOKBACK_DAYS;
        debug_assert!(s + 1 >= l, "caller must only evaluate bars with a full L-bar window");
        let start = s + 1 - l;
        let window_a = &closes_a[start..=s];
        let window_b = &closes_b[start..=s];

        let ln_a: Vec<f64> = window_a.iter().map(|x| x.ln()).collect();
        let ln_b: Vec<f64> = window_b.iter().map(|x| x.ln()).collect();

        let l_f = l as f64;
        let mean_a = ln_a.iter().sum::<f64>() / l_f;
        let mean_b = ln_b.iter().sum::<f64>() / l_f;

        let cov_ab: f64 = ln_a.iter().zip(&ln_b).map(|(a, b)| (a - mean_a) * (b - mean_b)).sum();
        let var_b: f64 = ln_b.iter().map(|b| (b - mean_b) * (b - mean_b)).sum();
        if !(var_b.is_finite() && var_b > 0.0) {
            return None; // degenerate hedge ratio (choice 7): zero/non-finite variance in ln(close_B).
        }
        let hedge_ratio = cov_ab / var_b;
        if !hedge_ratio.is_finite() {
            return None;
        }

        // Reconstruct every spread value in the window with this ONE hedge ratio (choice 3).
        let spreads: Vec<f64> = ln_a.iter().zip(&ln_b).map(|(a, b)| a - hedge_ratio * b).collect();
        let mean_spread = spreads.iter().sum::<f64>() / l_f;
        let var_spread: f64 =
            spreads.iter().map(|sp| (sp - mean_spread) * (sp - mean_spread)).sum::<f64>() / l_f; // ddof = 0
        let stdev_spread = var_spread.sqrt();
        if !(stdev_spread.is_finite() && stdev_spread > 0.0) {
            return None; // degenerate z-score (choice 7): zero/non-finite spread stdev.
        }

        let current_spread = spreads[l - 1]; // window's last element = spread at bar s.
        let z = (current_spread - mean_spread) / stdev_spread;
        z.is_finite().then_some(z)
    }
}

impl WeightRule for PairsZscoreMeanReversionRule {
    fn id(&self) -> &'static str {
        "pairs_zscore_meanreversion_60d"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &PAIRS_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("lookback_days", PAIRS_LOOKBACK_DAYS.to_string()),
            ("entry_threshold", PAIRS_ENTRY_THRESHOLD.to_string()),
            ("exit_threshold", PAIRS_EXIT_THRESHOLD.to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"on_decision\"".to_string()),
        ])
    }

    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }

    fn rebalance_policy(&self) -> RebalancePolicy {
        // OnDecision, not EveryBar -- see module doc choice 10. A hold must not be re-traded every bar just
        // because the schedule asks for a decision every bar.
        RebalancePolicy::OnDecision
    }

    fn min_history_bars(&self) -> usize {
        // L + 1 bars, per the spec's own literal insufficient-history threshold (choice 5 above), one more than
        // the L the regression/z-score arithmetic strictly needs.
        PAIRS_LOOKBACK_DAYS + 1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let needed = PAIRS_LOOKBACK_DAYS + 1;
        let closes_a = h.closes(0);
        let closes_b = h.closes(1);

        // Defensive, structurally-unreachable-via-simulate() per-asset check (choice 6 above): silent skip, not a
        // refusal.
        if closes_a.len() < needed || closes_b.len() < needed {
            return Ok(vec![0.0, 0.0]);
        }

        // Replay the state machine (the statefulness resolution described in the module doc) from the first bar
        // with a full L-bar window through the current (last) bar, starting from an assumed-flat position before
        // any decision was ever possible.
        let l = PAIRS_LOOKBACK_DAYS;
        let first_eligible = l - 1; // 0-indexed bar s where s + 1 == L (a full L-bar window first exists).
        let last = h.len() - 1;
        let mut state = Position::Flat;
        for s in first_eligible..=last {
            let z = Self::zscore_at(closes_a, closes_b, s);
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

    const L: usize = PAIRS_LOOKBACK_DAYS;

    /// Builds a two-asset panel over exactly `PAIRS_SYMBOLS`, on a simple synthetic ascending daily calendar
    /// starting 2020-01-01 (rolling across months as needed so every date stays valid without a full calendar
    /// implementation in the test). `closes_a`/`closes_b` must have equal length.
    fn panel_from(closes_a: Vec<f64>, closes_b: Vec<f64>) -> Panel {
        assert_eq!(closes_a.len(), closes_b.len());
        let n_bars = closes_a.len();
        let dates: Vec<Date> = (0..n_bars).map(|i| date_for_bar(i)).collect();
        let symbols: Vec<String> = PAIRS_SYMBOLS.iter().map(|s| s.to_string()).collect();
        Panel::new(symbols, dates, vec![closes_a, closes_b]).unwrap()
    }

    /// Bar `i`'s synthetic calendar date: one calendar day per bar starting 2020-01-01, rolling month to month in
    /// 28-day blocks (test-only convenience; `Panel`/`HistoryView` only require strictly ascending dates).
    fn date_for_bar(i: usize) -> Date {
        let total_day = i as u32; // 0-indexed offset from 2020-01-01
        let month_offset = total_day / 28;
        let day_in_month = total_day % 28 + 1;
        let month = 1 + month_offset;
        let (y, m) = (2020 + (month - 1) / 12, (month - 1) % 12 + 1);
        d(&format!("{y:04}-{m:02}-{day_in_month:02}"))
    }

    /// Runs the full `closes_a`/`closes_b` series through `simulate()` and returns the LAST bar's target weights
    /// (row `n_bars - 1`). `HistoryView::new` is `pub(crate)`, so driving the rule through the public
    /// `weightsim::simulate()` entry point (per this crate's test convention) is the only way to reach it.
    fn last_weights(closes_a: Vec<f64>, closes_b: Vec<f64>) -> Vec<f64> {
        let p = panel_from(closes_a, closes_b);
        let r = simulate(&p, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();
        r.row(&r.target_weights, p.n_bars() - 1).to_vec()
    }

    /// A trailing-`L`-bar window of constant log-returns on both legs (so the hedge ratio and the spread's mean
    /// are stable and non-degenerate) followed by a final-bar jump that pushes `ln(close_A)` sharply away from
    /// `hedge_ratio * ln(close_B)`, i.e. a clean spread blow-out on the last bar. `jump_a`/`jump_b` are added
    /// directly to the log-prices of the FINAL bar only (everything before the final bar is the smooth warmup
    /// series), so the sign/magnitude of the last bar's z-score is controlled directly by the caller rather than
    /// reverse-engineered from price moves.
    fn warmup_then_jump(n_bars: usize, jump_a: f64, jump_b: f64) -> (Vec<f64>, Vec<f64>) {
        let mut a = Vec::with_capacity(n_bars);
        let mut b = Vec::with_capacity(n_bars);
        // Smooth, slightly different drifts so the hedge ratio is well-defined and not exactly 1.0, but otherwise
        // a tight, low-noise relationship (small deterministic wiggle) so the spread's trailing stdev is small and
        // a last-bar jump reliably produces a large |z|.
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
        let rule = PairsZscoreMeanReversionRule;
        let params = rule.declared_parameters();
        assert_eq!(params.get("lookback_days").map(|s| s.as_str()), Some("60"));
        assert_eq!(params.get("entry_threshold").map(|s| s.as_str()), Some("2"));
        assert_eq!(params.get("exit_threshold").map(|s| s.as_str()), Some("0.5"));
        assert_eq!(rule.min_history_bars(), L + 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(rule.rebalance_policy(), RebalancePolicy::OnDecision);
        assert_eq!(rule.universe(), &PAIRS_SYMBOLS);
    }

    #[test]
    fn insufficient_history_is_a_silent_skip_not_a_refusal() {
        // Exactly L bars (one short of L + 1): min_history_bars() keeps the simulator from ever calling the rule.
        let (a, b) = warmup_then_jump(L, -10.0, 0.0);
        let p = panel_from(a, b);
        let r = simulate(&p, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();
        assert!(r.decision.iter().all(|&dec| !dec), "no bar has L + 1 bars visible");
        assert!(r.refused.iter().all(|&ref_| !ref_), "a silent skip must not be recorded as a refusal");
        assert!(r.target_weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn large_negative_zscore_enters_long_a_short_b() {
        // A sharp drop in ln(close_A) on the final bar (holding B's path fixed) blows the spread deeply negative
        // relative to its own trailing mean -> z << -entry_threshold -> enter long-A / short-B.
        let (a, b) = warmup_then_jump(L + 5, -5.0, 0.0);
        let w = last_weights(a, b);
        assert_eq!(w, vec![0.5, -0.5], "z far below -entry_threshold must enter long-A/short-B: {w:?}");
    }

    #[test]
    fn large_positive_zscore_enters_short_a_long_b() {
        // Mirror image: a sharp jump UP in ln(close_A) blows the spread deeply positive -> z >> +entry_threshold
        // -> enter short-A / long-B.
        let (a, b) = warmup_then_jump(L + 5, 5.0, 0.0);
        let w = last_weights(a, b);
        assert_eq!(w, vec![-0.5, 0.5], "z far above +entry_threshold must enter short-A/long-B: {w:?}");
    }

    #[test]
    fn exits_to_flat_once_z_falls_back_inside_the_exit_band() {
        // Build a panel where the rule enters long-A/short-B at bar `enter_at` (a sharp drop), then find -- by
        // calibrating against the SAME private `zscore_at` the rule itself uses -- a bar-`exit_at` adjustment to
        // close_A that lands |z| strictly inside the exit band (< PAIRS_EXIT_THRESHOLD) while that bar's trailing
        // window still contains the `enter_at` outlier. A full, un-calibrated "snap back to the pre-jump smooth
        // price" does NOT reliably land inside the exit band: the single large outlier still inside the L-bar
        // window shifts the window's own mean/stdev enough that the recovered bar's z-score is not always small.
        let n_bars = L + 6;
        let enter_at = n_bars - 2;
        let exit_at = n_bars - 1;
        let (baseline, b) = warmup_then_jump(n_bars, 0.0, 0.0); // smooth, no jumps anywhere.
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
                match PairsZscoreMeanReversionRule::zscore_at(&a[..=exit_at], &b[..=exit_at], exit_at) {
                    Some(z) => z.abs() < PAIRS_EXIT_THRESHOLD,
                    None => false,
                }
            })
            .expect("calibration must find a jump landing inside the exit band");

        let a = build(exit_jump);

        // Sanity check the setup actually enters at `enter_at` before testing the exit at `exit_at`.
        let p_enter = panel_from(a[..=enter_at].to_vec(), b[..=enter_at].to_vec());
        let r_enter = simulate(&p_enter, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();
        let w_enter = r_enter.row(&r_enter.target_weights, enter_at).to_vec();
        assert_eq!(w_enter, vec![0.5, -0.5], "setup must enter long-A/short-B at bar {enter_at}: {w_enter:?}");

        let p_full = panel_from(a, b);
        let r_full = simulate(&p_full, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();
        let w_exit = r_full.row(&r_full.target_weights, exit_at).to_vec();
        assert_eq!(w_exit, vec![0.0, 0.0], "z back inside the exit band must flatten the position: {w_exit:?}");
    }

    #[test]
    fn holds_the_position_unchanged_in_the_hysteresis_band() {
        // After entering long-A/short-B on a sharp drop at `enter_at`, bar `hold_at` must land with |z| strictly
        // between the exit and entry thresholds (the hysteresis band) so the state machine takes the explicit
        // "otherwise HOLD" branch, not the exit branch or an accidental re-entry. Rather than guessing a jump size
        // blind, this calibrates it using the SAME private `zscore_at` the rule itself uses, over a small grid of
        // candidate jump magnitudes, so the test provably exercises the hold branch rather than merely producing
        // matching weights for the wrong reason.
        let n_bars = L + 6;
        let enter_at = n_bars - 2;
        let hold_at = n_bars - 1;
        let (baseline, b) = warmup_then_jump(n_bars, 0.0, 0.0); // smooth, no jumps anywhere.
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
                match PairsZscoreMeanReversionRule::zscore_at(&a[..=hold_at], &b[..=hold_at], hold_at) {
                    Some(z) => z.abs() > PAIRS_EXIT_THRESHOLD && z.abs() < PAIRS_ENTRY_THRESHOLD,
                    None => false,
                }
            })
            .expect("calibration must find a jump landing inside the hysteresis band");

        let a = build(hold_jump);
        let p = panel_from(a, b);
        let r = simulate(&p, &PairsZscoreMeanReversionRule, &SimConfig::default()).unwrap();
        let w_enter = r.row(&r.target_weights, enter_at).to_vec();
        let w_hold = r.row(&r.target_weights, hold_at).to_vec();
        assert_eq!(w_enter, vec![0.5, -0.5], "setup must enter long-A/short-B at bar {enter_at}: {w_enter:?}");
        assert_eq!(w_hold, vec![0.5, -0.5], "position must hold unchanged inside the hysteresis band: {w_hold:?}");
    }
}
