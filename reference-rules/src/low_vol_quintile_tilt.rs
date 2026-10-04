//! `low_vol_quintile_tilt`: a basket of assets, daily. Rank all eligible assets by trailing volatility ascending
//! (same vol-estimation convention as [`crate::inverse_volatility_weight`]: `L` = 20 trading days, population
//! stdev (ddof = 0) of simple daily returns), hold the bottom quintile (lowest-vol 20%) equal-weighted, zero
//! weight on the rest. Default `L` = 20 trading days ([`LOW_VOL_QUINTILE_LOOKBACK_DAYS`]).
//!
//! Like `inverse_volatility_weight` (and unlike `crypto`/`etf`/`fx` in this crate), this primitive implements
//! `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView`: it is a generic basket rule, not a
//! documented sleeve adapted by a downstream crate. Per this crate's convention each primitive module is
//! self-contained, so the volatility helper here is its own copy rather than a shared/reused private helper from
//! `inverse_volatility_weight.rs`.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Returns are SIMPLE daily returns, not log returns*: `r_k = close_k / close_{k-1} - 1`, per the spec's
//!    explicit convention (same as #1 / `inverse_volatility_weight`).
//! 2. *Population stdev (ddof = 0)*: `sqrt(mean((r - mean(r))^2))` over exactly the trailing `L` returns (divide
//!    by `L`, not `L - 1`).
//! 3. *"Trailing L bars" of returns needs `L + 1` closes* (`L` returns are computed from `L + 1` consecutive
//!    closes), same reading as #1, confirmed by the spec's own insufficient-history threshold.
//! 4. *Insufficient history = typed exclusion, not a zero return.* An asset with fewer than `L + 1` closes visible
//!    is excluded from ranking entirely; its slot in the returned weight vector is forced to `0.0`. As in
//!    `inverse_volatility_weight`, `WeightRule::target_weights` must return a same-length, same-order `Vec<f64>`,
//!    so "excluded" can only be expressed as a `0.0` in that slot.
//! 5. *Structural tension with `HistoryView` (documented per the task, not resolved by changing `weightsim`).*
//!    `HistoryView` is one shared, rectangular panel: every asset in the universe has exactly the same number of
//!    visible bars at a given decision index (`Panel::inner_join` keeps only dates common to every requested
//!    symbol). So per-asset insufficient-history exclusion is, in THIS simulator, all-or-nothing at a given
//!    decision: either every asset clears `L + 1` bars, or none do (and `min_history_bars()` below stops the
//!    simulator from calling the rule at all before that point, so the exclusion branch below is unreachable in
//!    practice through the public API — kept anyway as a defensive, typed, literally spec-faithful check, exactly
//!    as `inverse_volatility_weight` does, for the same reason: it becomes meaningful the day `weightsim` grows a
//!    panel representation where assets can differ in visible length).
//! 6. *Zero or non-finite stdev of an eligible asset.* Unlike `inverse_volatility_weight` (which divides by
//!    stdev and so must refuse on exactly zero), this rule only RANKS by stdev — it never divides by it — so a
//!    perfectly flat price path (stdev exactly 0.0) is not a degenerate case here: it is simply the lowest
//!    possible volatility and a legitimate (indeed maximally eligible) candidate for the bottom quintile. A
//!    NON-FINITE stdev (NaN or infinite — unreachable for closes that satisfy `Panel`'s finite-and-positive
//!    invariant, but defended anyway since nothing in this rule re-derives that invariant) WOULD corrupt the
//!    total order used for ranking (NaN is incomparable), so that case alone is raised as a typed
//!    `RefusalKind::Data` refusal for the whole decision, mirroring choice 6 of `inverse_volatility_weight` but
//!    narrowed to the failure mode that actually threatens this rule's logic.
//! 7. *Quintile sizing when the (eligible) universe size `n` is not a multiple of 5 — FLAGGED AMBIGUITY, resolved
//!    here.* The spec says "hold the bottom quintile (lowest-vol 20%)" but does not say how to round `0.2 * n` to
//!    an integer count of holdings when it is not already a whole number. This implementation uses
//!    nearest-integer rounding (`(n as f64 / 5.0).round()`, i.e. round-half-away-from-zero), floored at a minimum
//!    of 1 so the quintile is never empty (`quintile_count` below). Nearest-integer rounding was chosen over
//!    floor (which would round `20% of n` down, e.g. 1 holding for `n` up to 7, systematically UNDER-weighting
//!    the "20%" target as `n` grows within each band of 5) or ceiling (which would systematically OVER-weight
//!    it, e.g. 2 holdings at `n = 6`, a 33% quintile) because it tracks `0.2 * n` most closely on average without
//!    a directional bias. This is a defensible choice, not the only one; floor and ceiling are both arguably
//!    equally literal readings of "20%", and a different, equally reasonable implementation could pick either.
//! 8. *Tie-breaking at the quintile boundary.* The spec requires ascending alphabetical ticker order whenever two
//!    assets tie exactly on trailing stdev. Implemented as a stable sort on `(stdev, symbol)` ascending, so ties
//!    are broken purely by symbol and the boundary (the `k`-th vs `k+1`-th ranked asset, `k` = `quintile_count`)
//!    is well-defined even when several assets share the exact same stdev value.
//! 9. *Universe is a placeholder.* This primitive is a generic "basket of assets" rule, not tied to one
//!    documented sleeve, so [`LOW_VOL_QUINTILE_SYMBOLS`] is a small, liquid, diversified seven-ETF example
//!    universe (broad equities, small caps, developed/emerging international, gold, long bonds), deliberately
//!    sized to 7 (not a multiple of 5) so the quintile-sizing choice above (choice 7) is exercised rather than
//!    sidestepped by this module's own default universe; `quintile_count(7) == 1`. It deliberately does not
//!    reuse `ETF_SYMBOLS` (a different, documented sleeve) or `INVERSE_VOLATILITY_SYMBOLS` (a different generic
//!    placeholder basket). A caller higher up the stack assigns the real production universe later.
//! 10. *Schedule and rebalance policy.* `decision_schedule()` is `Daily` per the spec ("basket of assets,
//!     daily"). `rebalance_policy()` is not specified; `EveryBar` is chosen to match this crate's other `Daily`
//!     basket primitive (`inverse_volatility_weight`). Under `Daily` every bar is a decision bar, so
//!     `OnDecision` and `EveryBar` are operationally identical here (no "between decisions" gap to drift in).

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference universe: seven liquid, diversified ETFs (broad US equities, small caps, developed
/// international, emerging markets, gold, long Treasuries). Not a documented sleeve and not production data — a
/// placeholder basket, deliberately sized to 7 (not a multiple of 5) so the bottom-quintile rounding rule (module
/// doc choice 7) is actually exercised. Already in ascending alphabetical order.
pub const LOW_VOL_QUINTILE_SYMBOLS: [&str; 7] = ["DIA", "EEM", "EFA", "GLD", "IWM", "SPY", "TLT"];

/// Lookback `L`, in trading days, of trailing daily returns the stdev is computed over (default 20). `L + 1`
/// trailing closes are required per asset (choice 3 above). Same convention as
/// [`crate::inverse_volatility_weight::INVERSE_VOLATILITY_LOOKBACK_DAYS`].
pub const LOW_VOL_QUINTILE_LOOKBACK_DAYS: usize = 20;

/// The quintile fraction, for `declared_parameters()` and documentation; not itself used in the rounding formula
/// (see `quintile_count` and module doc choice 7).
pub const LOW_VOL_QUINTILE_FRACTION: f64 = 0.2;

/// How many of `n_eligible` ranked assets make up "the bottom quintile" (module doc choice 7): nearest-integer
/// rounding of `0.2 * n_eligible`, floored at a minimum of 1 so the quintile is never empty.
pub fn quintile_count(n_eligible: usize) -> usize {
    if n_eligible == 0 {
        return 0;
    }
    ((n_eligible as f64) / 5.0).round().max(1.0) as usize
}

/// `low_vol_quintile_tilt`: rank all eligible assets by trailing `L`-day simple-daily-return population stdev
/// (ddof = 0) ascending, hold the bottom quintile equal-weighted, zero weight on the rest. Ties at the quintile
/// boundary break by ascending alphabetical ticker order. See the module doc for every interpretation choice, in
/// particular choice 5 (the `HistoryView` rectangular-panel constraint), choice 6 (zero stdev is not degenerate
/// here, only non-finite stdev is), and choice 7 (quintile-size rounding for non-multiple-of-5 universes).
#[derive(Clone, Copy, Debug, Default)]
pub struct LowVolQuintileTiltRule;

impl WeightRule for LowVolQuintileTiltRule {
    fn id(&self) -> &'static str {
        "low_vol_quintile_tilt_20d"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &LOW_VOL_QUINTILE_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("lookback_days", LOW_VOL_QUINTILE_LOOKBACK_DAYS.to_string()),
            ("quintile_fraction", LOW_VOL_QUINTILE_FRACTION.to_string()),
            ("schedule", "\"daily\"".to_string()),
            ("rebalance_policy", "\"every_bar\"".to_string()),
        ])
    }

    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }

    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }

    fn min_history_bars(&self) -> usize {
        // L + 1 bars (indices 0..=L) are needed before L trailing returns exist at all (choice 3 above). Before
        // that the simulator silently skips the rule (never calls it), per the trait's own contract.
        LOW_VOL_QUINTILE_LOOKBACK_DAYS + 1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let n = h.n_assets();
        let needed = LOW_VOL_QUINTILE_LOOKBACK_DAYS + 1;
        let mut weights = vec![0.0_f64; n];

        // (stdev, symbol, index) for every eligible asset, i.e. one with >= L + 1 closes visible (choice 4).
        let mut eligible: Vec<(f64, &str, usize)> = Vec::with_capacity(n);

        for i in 0..n {
            let closes = h.closes(i);
            if closes.len() < needed {
                // Insufficient history: typed exclusion (choice 4). Leave weights[i] at 0.0 and do not add it to
                // the ranking pool.
                continue;
            }
            let window = &closes[closes.len() - needed..];
            let mut returns = Vec::with_capacity(LOW_VOL_QUINTILE_LOOKBACK_DAYS);
            for k in 1..window.len() {
                returns.push(window[k] / window[k - 1] - 1.0);
            }
            let l = returns.len() as f64;
            let mean = returns.iter().sum::<f64>() / l;
            let variance = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / l; // ddof = 0
            let stdev = variance.sqrt();
            if !stdev.is_finite() {
                // Non-finite stdev would corrupt the ranking's total order (choice 6); zero itself is fine and
                // simply ranks lowest.
                return Err(RuleRefusal::data(
                    "non_finite_volatility",
                    format!(
                        "{}: population stdev of the trailing {} daily returns is non-finite ({}); cannot rank",
                        h.symbols()[i],
                        LOW_VOL_QUINTILE_LOOKBACK_DAYS,
                        stdev
                    ),
                ));
            }
            eligible.push((stdev, h.symbols()[i].as_str(), i));
        }

        if eligible.is_empty() {
            // Every asset lacks L + 1 closes. Unreachable once the simulator respects `min_history_bars` (choice
            // 5), but kept as a typed refusal rather than an empty ranking.
            return Err(RuleRefusal::warmup(format!(
                "no asset in the universe has {needed} trailing closes (lookback {LOW_VOL_QUINTILE_LOOKBACK_DAYS} days)"
            )));
        }

        // Ascending by (stdev, symbol): stdev primary, symbol breaks exact ties alphabetically (choice 8).
        eligible.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-finite stdev already rejected above").then(a.1.cmp(b.1)));

        let k = quintile_count(eligible.len());
        let held = &eligible[..k.min(eligible.len())];
        let w = 1.0 / held.len() as f64;
        for &(_, _, i) in held {
            weights[i] = w;
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weightsim::{simulate, Date, Panel, SimConfig};

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    /// Builds a panel over the FULL `LOW_VOL_QUINTILE_SYMBOLS` universe (required: `simulate` rejects a panel
    /// whose symbols are not exactly the rule's `universe()`). `closes` supplies the first `closes.len()` assets;
    /// any remaining slots up to all 7 symbols are padded with an arbitrary, non-degenerate filler series so tests
    /// can focus on the assets they actually care about. All assets share the same synthetic ascending daily
    /// calendar starting 2020-01-01.
    fn panel_from(mut closes: Vec<Vec<f64>>) -> Panel {
        let n_bars = closes[0].len();
        while closes.len() < LOW_VOL_QUINTILE_SYMBOLS.len() {
            let start = 10.0 * (closes.len() as f64 + 1.0);
            closes.push(noisy_series(start, n_bars, 0.1));
        }
        let dates: Vec<Date> = (0..n_bars)
            .map(|i| {
                // Simple synthetic ascending calendar: one calendar day per bar, starting 2020-01-01. Fine for a
                // unit test; HistoryView/Panel only require strictly ascending dates, not a real trading calendar.
                let day = 1 + i as u32;
                let (y, m, dd) = (2020, 1, day);
                if dd <= 28 {
                    d(&format!("{y:04}-{m:02}-{dd:02}"))
                } else {
                    // Roll into February for runs longer than 28 bars (keeps every date valid without a full
                    // calendar implementation in the test).
                    d(&format!("{y:04}-02-{:02}", dd - 28))
                }
            })
            .collect();
        let symbols: Vec<String> =
            LOW_VOL_QUINTILE_SYMBOLS[..closes.len()].iter().map(|s| s.to_string()).collect();
        Panel::new(symbols, dates, closes).unwrap()
    }

    /// Constant daily growth factor `g` for `n_bars` bars starting at `start`, giving a perfectly constant simple
    /// daily return of `g - 1` and therefore a population stdev of exactly 0.
    fn flat_growth_series(start: f64, g: f64, n_bars: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n_bars);
        let mut p = start;
        for _ in 0..n_bars {
            v.push(p);
            p *= g;
        }
        v
    }

    /// Alternating two-step growth pattern, giving a non-zero, finite stdev of daily returns whose magnitude
    /// scales with `amplitude`.
    fn noisy_series(start: f64, n_bars: usize, amplitude: f64) -> Vec<f64> {
        let mut v = Vec::with_capacity(n_bars);
        let mut p = start;
        for i in 0..n_bars {
            v.push(p);
            let g = if i % 2 == 0 { 1.0 + amplitude } else { 1.0 / (1.0 + amplitude) };
            p *= g;
        }
        v
    }

    const L: usize = LOW_VOL_QUINTILE_LOOKBACK_DAYS;

    // `HistoryView`'s constructor is `pub(crate)` to `weightsim` (by design: only the simulator may build one), so
    // these tests drive the rule the same way any real caller must: through the public `weightsim::simulate` entry
    // point over a `Panel`, then inspect the resulting `target_weights` column. `SimConfig::default()` is zero
    // cost / zero financing / no delay / `OnRefusal::Abort`, which is exactly what each test below wants.

    /// The last bar's target weights (row `n_bars - 1` of the flat `target_weights` matrix).
    fn last_weights(p: &Panel) -> Vec<f64> {
        let r = simulate(p, &LowVolQuintileTiltRule, &SimConfig::default()).unwrap();
        r.row(&r.target_weights, p.n_bars() - 1).to_vec()
    }

    #[test]
    fn declared_parameters_reflect_the_lookback_constant() {
        let rule = LowVolQuintileTiltRule;
        let params = rule.declared_parameters();
        assert_eq!(params.get("lookback_days").map(|s| s.as_str()), Some("20"));
        assert_eq!(rule.min_history_bars(), L + 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(rule.universe(), &LOW_VOL_QUINTILE_SYMBOLS);
    }

    #[test]
    fn quintile_count_rounds_to_nearest_with_a_floor_of_one() {
        // Module doc choice 7: nearest-integer rounding of n / 5, minimum 1.
        assert_eq!(quintile_count(0), 0);
        assert_eq!(quintile_count(1), 1);
        assert_eq!(quintile_count(2), 1);
        assert_eq!(quintile_count(3), 1);
        assert_eq!(quintile_count(5), 1);
        assert_eq!(quintile_count(7), 1); // this module's own universe size
        assert_eq!(quintile_count(8), 2);
        assert_eq!(quintile_count(10), 2);
    }

    #[test]
    fn bottom_quintile_is_equal_weighted_and_the_rest_are_zero() {
        // 7 assets, all with enough history. quintile_count(7) == 1, so exactly the single lowest-vol asset
        // should get weight 1.0 and every other asset must get exactly 0.0.
        let calmest = noisy_series(100.0, L + 1, 0.001);
        let a = noisy_series(50.0, L + 1, 0.02);
        let b = noisy_series(60.0, L + 1, 0.03);
        let c = noisy_series(70.0, L + 1, 0.04);
        let p = panel_from(vec![calmest, a, b, c]);
        let w = last_weights(&p);
        assert_eq!(w.len(), LOW_VOL_QUINTILE_SYMBOLS.len());
        let held: Vec<usize> = (0..w.len()).filter(|&i| w[i] != 0.0).collect();
        assert_eq!(held.len(), 1, "exactly quintile_count(7) == 1 asset should be held: {w:?}");
        assert_eq!(held[0], 0, "the calmest asset (index 0, DIA) must be the one held: {w:?}");
        assert!((w[0] - 1.0).abs() < 1e-12, "the sole held asset must get weight 1.0, got {}", w[0]);
        for (i, &wi) in w.iter().enumerate().skip(1) {
            assert_eq!(wi, 0.0, "asset {i} outside the bottom quintile must get weight 0.0, got {wi}");
        }
    }

    #[test]
    fn ties_at_the_quintile_boundary_break_alphabetically() {
        // DIA and EEM (universe indices 0 and 1) are given the EXACT same flat growth factor, so they tie exactly
        // on stdev (both 0.0) for the lowest-vol spot. Every other asset is strictly noisier. quintile_count(7)
        // == 1, so only one asset is held; the tie must break to the alphabetically-first ticker, DIA (index 0),
        // never EEM (index 1).
        let dia = flat_growth_series(100.0, 1.01, L + 1);
        let eem = flat_growth_series(50.0, 1.01, L + 1);
        let efa = noisy_series(70.0, L + 1, 0.05);
        let p = panel_from(vec![dia, eem, efa]);
        let w = last_weights(&p);
        assert!((w[0] - 1.0).abs() < 1e-12, "DIA (alphabetically first of the tied pair) must be held: {w:?}");
        assert_eq!(w[1], 0.0, "EEM must lose the alphabetical tie-break and get zero weight: {w:?}");
    }

    #[test]
    fn insufficient_history_means_no_decision_ever_happens_not_a_refusal() {
        // With only L bars (one short of the required L + 1), `min_history_bars()` keeps the simulator from ever
        // calling the rule at all (trait contract: "silent skip, not a refusal") -- the same structural reason the
        // per-asset exclusion branch in `target_weights` can never actually trigger through this public API (every
        // asset in a `HistoryView` always shares the same bar count; see module doc choice 5). So the sanity check
        // for "insufficient history" here is: no decision happens and nothing is refused.
        let a = noisy_series(100.0, L, 0.01);
        let b = noisy_series(50.0, L, 0.02);
        let p = panel_from(vec![a, b]);
        let r = simulate(&p, &LowVolQuintileTiltRule, &SimConfig::default()).unwrap();
        assert!(r.decision.iter().all(|&d| !d), "no bar has enough history to decide");
        assert!(r.refused.iter().all(|&ref_| !ref_), "a silent skip must not be recorded as a refusal");
        assert!(r.target_weights.iter().all(|&w| w == 0.0));

        // One bar later (L + 1 bars) both assets clear the threshold and the final bar decides successfully.
        let a2 = noisy_series(100.0, L + 1, 0.01);
        let b2 = noisy_series(50.0, L + 1, 0.02);
        let p2 = panel_from(vec![a2, b2]);
        let r2 = simulate(&p2, &LowVolQuintileTiltRule, &SimConfig::default()).unwrap();
        assert!(r2.decision[p2.n_bars() - 1], "L + 1 bars must be enough to decide");
    }

    #[test]
    fn zero_volatility_is_not_a_refusal_here_it_simply_ranks_lowest() {
        // Unlike inverse_volatility_weight (which divides by stdev), this rule only ranks by it, so a perfectly
        // flat series (stdev exactly 0) is a legitimate, maximally-eligible low-vol candidate, not a degenerate
        // case (choice 6). It must win the bottom quintile outright, with no refusal raised.
        let flat = flat_growth_series(100.0, 1.0, L + 1);
        let a = noisy_series(50.0, L + 1, 0.02);
        let b = noisy_series(60.0, L + 1, 0.03);
        let p = panel_from(vec![flat, a, b]);
        let w = last_weights(&p);
        assert!((w[0] - 1.0).abs() < 1e-12, "the flat (zero-vol) asset must be held: {w:?}");
    }

    #[test]
    fn weights_sum_to_the_number_held_times_their_equal_share() {
        // General sanity check independent of which assets are held: the held assets' weights must sum to 1.0
        // (equal-weighted quintile) and every other weight must be exactly 0.0.
        let a = noisy_series(100.0, L + 1, 0.01);
        let b = noisy_series(50.0, L + 1, 0.02);
        let p = panel_from(vec![a, b]);
        let w = last_weights(&p);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "held weights must sum to 1.0, got {sum}");
        assert!(w.iter().all(|&wi| wi == 0.0 || (wi - 1.0 / quintile_count(7) as f64).abs() < 1e-12));
    }
}
