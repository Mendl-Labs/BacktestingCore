//! `inverse_volatility_weight`: a basket of assets, daily. Each asset's weight is proportional to
//! `1 / stdev(daily returns over the trailing L bars)` (lower realized vol -> higher weight), normalized so the
//! weights sum to 1.0 over the assets that have enough history. Default `L` = 20 trading days
//! ([`INVERSE_VOLATILITY_LOOKBACK_DAYS`]).
//!
//! Unlike `crypto`/`etf`/`fx` in this crate, this primitive implements `weightsim::WeightRule` DIRECTLY against
//! `weightsim::HistoryView` rather than this crate's own `Panel`/`PriceSeries` types: it is a generic basket rule,
//! not a documented sleeve adapted by a downstream crate. `reference-rules/Cargo.toml` depends on `weightsim` as a
//! plain path dependency for this reason.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Returns are SIMPLE daily returns, not log returns*: `r_k = close_k / close_{k-1} - 1`. Stated explicitly
//!    per the spec, since both conventions are common and they are not numerically interchangeable even for a
//!    second moment like stdev.
//! 2. *Population stdev (ddof = 0)*: `sqrt(mean((r - mean(r))^2))` over exactly the trailing `L` returns, i.e.
//!    divide by `L`, not `L - 1`.
//! 3. *"Trailing L bars" of returns needs `L + 1` closes* (`L` returns are computed from `L + 1` consecutive
//!    closes). The spec's own insufficient-history threshold ("fewer than `L + 1` closes") confirms this reading,
//!    so it is not left ambiguous.
//! 4. *Insufficient history = typed exclusion, not a zero return.* An asset with fewer than `L + 1` closes visible
//!    is dropped from the normalization entirely (it is not assigned a return of 0, which would bias `stdev`
//!    downward and inflate its own weight). Concretely, its entry in the returned weight vector is forced to
//!    `0.0` and it contributes nothing to the sum used to normalize the others — `WeightRule::target_weights`
//!    must return a `Vec<f64>` the same length and order as `universe()`, so "excluded" can only be expressed as
//!    a 0.0 in that slot, never as a shorter vector; the remaining assets' weights still sum to 1.0.
//! 5. *Structural tension with `HistoryView` (documented per the task, not resolved by changing `weightsim`).*
//!    `HistoryView` is one shared, rectangular panel: at a given decision index every asset in the universe has
//!    exactly the same number of visible bars (`HistoryView::len()`), by construction (`Panel::inner_join` keeps
//!    only dates common to every requested symbol; there is no per-asset padding or NaN). So in THIS simulator,
//!    per-asset insufficient-history exclusion can only ever be all-or-nothing at a given decision: either every
//!    asset in the universe has `>= L + 1` bars, or none of them do. The per-asset check below is still
//!    implemented exactly as specified (checking `h.closes(i).len()` independently for each `i`) rather than
//!    collapsed into a single universe-wide check, so the rule is already correct and literally spec-faithful if
//!    `weightsim` ever grows a panel representation where assets can differ in visible length (e.g. a later
//!    listing date). Today it is unreachable in practice because `min_history_bars()` (below) stops the simulator
//!    from calling the rule at all before bar `L` (index `L`, i.e. `L + 1` bars visible) — at that point every
//!    asset already clears the threshold together, so the "no asset survives" refusal path also never fires
//!    once the rule starts being called, but is kept as a defensive, typed refusal rather than a silent `NaN`.
//! 6. *Zero or non-finite stdev is a refusal, NOT a silent exclusion.* The spec only names "insufficient history"
//!    (fewer than `L + 1` closes) as a reason to exclude an asset. A flat price path over the lookback window
//!    (stdev exactly 0, e.g. a frozen/stale feed) is a different failure mode — `1 / 0` is undefined, not merely
//!    "not enough data" — so it is raised as a typed `RefusalKind::Data` refusal for the whole decision (mirrors
//!    `fx.rs`'s `DegenerateSleeveVolatility` treatment of zero sleeve volatility) rather than being folded into
//!    the exclusion rule or silently producing an infinite/NaN weight.
//! 7. *Universe is a placeholder.* This primitive is a generic "basket of assets" rule, not tied to one documented
//!    sleeve, so [`INVERSE_VOLATILITY_SYMBOLS`] is a small, liquid, diversified five-ETF example basket (equities,
//!    small caps, long bonds, gold) chosen only so the rule compiles and is testable; it deliberately does not
//!    reuse `ETF_SYMBOLS` (a different, documented sleeve). A caller higher up the stack assigns the real
//!    production universe later; that is out of scope here.
//! 8. *Schedule and rebalance policy.* `decision_schedule()` is `Daily` per the spec ("basket of assets, daily").
//!    `rebalance_policy()` is not specified; `EveryBar` is chosen to match this crate's other `Daily` primitive
//!    (`crypto_trend_100d`). Under a `Daily` schedule every bar is a decision bar, so `OnDecision` ("drift between
//!    decisions") and `EveryBar` ("re-trade to target every bar") are operationally identical here — there is no
//!    "between decisions" gap for units to drift in — so this choice has no observable effect on the simulated
//!    weights.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference universe: five liquid, diversified ETFs (broad equities, small caps, long Treasuries, gold).
/// Not a documented sleeve and not production data — a placeholder basket so this generic primitive compiles and
/// is testable (choice 7 above). A caller assigns the real universe later.
pub const INVERSE_VOLATILITY_SYMBOLS: [&str; 5] = ["SPY", "IWM", "EFA", "TLT", "GLD"];

/// Lookback `L`, in trading days, of trailing daily returns the stdev is computed over (default 20). `L + 1`
/// trailing closes are required per asset (choice 3 above).
pub const INVERSE_VOLATILITY_LOOKBACK_DAYS: usize = 20;

/// `inverse_volatility_weight`: weight each asset proportional to `1 / stdev` of its trailing `L`-day simple daily
/// returns (population stdev, ddof = 0), normalized to sum to 1.0 over assets with enough history. See the module
/// doc for every interpretation choice, in particular choice 5 (the `HistoryView` rectangular-panel constraint) and
/// choice 6 (zero/non-finite stdev is a refusal, not an exclusion).
#[derive(Clone, Copy, Debug, Default)]
pub struct InverseVolatilityWeightRule;

impl WeightRule for InverseVolatilityWeightRule {
    fn id(&self) -> &'static str {
        "inverse_volatility_weight_20d"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &INVERSE_VOLATILITY_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("lookback_days", INVERSE_VOLATILITY_LOOKBACK_DAYS.to_string()),
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
        INVERSE_VOLATILITY_LOOKBACK_DAYS + 1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let n = h.n_assets();
        let needed = INVERSE_VOLATILITY_LOOKBACK_DAYS + 1;
        let mut inv_vol = vec![0.0_f64; n];
        let mut any_included = false;

        for i in 0..n {
            let closes = h.closes(i);
            if closes.len() < needed {
                // Insufficient history: typed exclusion (choice 4). Leave inv_vol[i] at 0.0, which both marks it
                // excluded in the returned vector and contributes nothing to the normalization sum below.
                continue;
            }
            let window = &closes[closes.len() - needed..];
            let mut returns = Vec::with_capacity(INVERSE_VOLATILITY_LOOKBACK_DAYS);
            for k in 1..window.len() {
                returns.push(window[k] / window[k - 1] - 1.0);
            }
            let l = returns.len() as f64;
            let mean = returns.iter().sum::<f64>() / l;
            let variance = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / l; // ddof = 0
            let stdev = variance.sqrt();
            if !(stdev.is_finite() && stdev > 0.0) {
                // Zero/non-finite stdev is a refusal, not an exclusion (choice 6): 1/stdev is undefined, which is
                // a different failure mode than "not enough data".
                return Err(RuleRefusal::data(
                    "degenerate_volatility",
                    format!(
                        "{}: population stdev of the trailing {} daily returns is {} (zero or non-finite); \
                         inverse-volatility weight is undefined",
                        h.symbols()[i],
                        INVERSE_VOLATILITY_LOOKBACK_DAYS,
                        stdev
                    ),
                ));
            }
            inv_vol[i] = 1.0 / stdev;
            any_included = true;
        }

        if !any_included {
            // Every asset lacks L + 1 closes. Unreachable once the simulator respects `min_history_bars` (choice
            // 5), but kept as a typed refusal rather than a 0/0 normalization.
            return Err(RuleRefusal::warmup(format!(
                "no asset in the universe has {needed} trailing closes (lookback {INVERSE_VOLATILITY_LOOKBACK_DAYS} days)"
            )));
        }

        let total: f64 = inv_vol.iter().sum();
        Ok(inv_vol.iter().map(|w| w / total).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weightsim::{simulate, Date, Panel, RefusalKind, SimConfig, SimError};

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    /// Builds a panel over the FULL `INVERSE_VOLATILITY_SYMBOLS` universe (required: `simulate` rejects a panel
    /// whose symbols are not exactly the rule's `universe()`). `closes` supplies the first `closes.len()` assets;
    /// any remaining slots up to all 5 symbols are padded with an arbitrary, non-degenerate filler series so tests
    /// can focus on the one or two assets they actually care about. All assets share the same synthetic ascending
    /// daily calendar starting 2020-01-01.
    fn panel_from(mut closes: Vec<Vec<f64>>) -> Panel {
        let n_bars = closes[0].len();
        while closes.len() < INVERSE_VOLATILITY_SYMBOLS.len() {
            let start = 10.0 * (closes.len() as f64 + 1.0);
            closes.push(noisy_series(start, n_bars, 0.02));
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
            INVERSE_VOLATILITY_SYMBOLS[..closes.len()].iter().map(|s| s.to_string()).collect();
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

    /// Alternating two-step growth pattern, giving a non-zero, finite stdev of daily returns.
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

    const L: usize = INVERSE_VOLATILITY_LOOKBACK_DAYS;

    #[test]
    fn declared_parameters_reflect_the_lookback_constant() {
        let rule = InverseVolatilityWeightRule;
        let params = rule.declared_parameters();
        assert_eq!(params.get("lookback_days").map(|s| s.as_str()), Some("20"));
        assert_eq!(rule.min_history_bars(), L + 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(rule.universe(), &INVERSE_VOLATILITY_SYMBOLS);
    }

    // `HistoryView`'s constructor is `pub(crate)` to `weightsim` (by design: only the simulator may build one), so
    // these tests drive the rule the same way any real caller must: through the public `weightsim::simulate` entry
    // point over a `Panel`, then inspect the resulting `target_weights` column. `SimConfig::default()` is zero
    // cost / zero financing / no delay / `OnRefusal::Abort`, which is exactly what each test below wants.

    /// The last bar's target weights (row `n_bars - 1` of the flat `target_weights` matrix).
    fn last_weights(p: &Panel) -> Vec<f64> {
        let r = simulate(p, &InverseVolatilityWeightRule, &SimConfig::default()).unwrap();
        r.row(&r.target_weights, p.n_bars() - 1).to_vec()
    }

    #[test]
    fn weights_sum_to_one_when_every_asset_has_enough_history() {
        let a = noisy_series(100.0, L + 1, 0.01);
        let b = noisy_series(50.0, L + 1, 0.03);
        let p = panel_from(vec![a, b]);
        let w = last_weights(&p);
        assert_eq!(w.len(), INVERSE_VOLATILITY_SYMBOLS.len());
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1.0, got {sum}");
    }

    #[test]
    fn lower_realized_vol_gets_a_higher_weight() {
        // Asset 0 is calmer (smaller amplitude) than asset 1, so it must end up with the larger weight.
        let calm = noisy_series(100.0, L + 1, 0.005);
        let volatile = noisy_series(100.0, L + 1, 0.05);
        let p = panel_from(vec![calm, volatile]);
        let w = last_weights(&p);
        assert!(w[0] > w[1], "calmer asset should get the higher weight: {w:?}");
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
        let r = simulate(&p, &InverseVolatilityWeightRule, &SimConfig::default()).unwrap();
        assert!(r.decision.iter().all(|&d| !d), "no bar has enough history to decide");
        assert!(r.refused.iter().all(|&ref_| !ref_), "a silent skip must not be recorded as a refusal");
        assert!(r.target_weights.iter().all(|&w| w == 0.0));

        // One bar later (L + 1 bars) both assets clear the threshold and the final bar decides successfully.
        let a2 = noisy_series(100.0, L + 1, 0.01);
        let b2 = noisy_series(50.0, L + 1, 0.02);
        let p2 = panel_from(vec![a2, b2]);
        let r2 = simulate(&p2, &InverseVolatilityWeightRule, &SimConfig::default()).unwrap();
        assert!(r2.decision[p2.n_bars() - 1], "L + 1 bars must be enough to decide");
    }

    #[test]
    fn zero_volatility_is_a_refusal_not_a_silent_exclusion() {
        // A perfectly flat (constant growth factor 1.0, i.e. constant price) series has population stdev exactly
        // 0; 1/0 is undefined, so this must be a typed Data refusal (surfaced by `simulate` as `RuleRefused`, since
        // only `Warmup` refusals before the first successful decision are tolerated under `OnRefusal::Abort`), not
        // a 0-weight exclusion or a NaN/inf weight.
        let flat = flat_growth_series(100.0, 1.0, L + 1);
        let other = noisy_series(50.0, L + 1, 0.02);
        let p = panel_from(vec![flat, other]);
        let err = simulate(&p, &InverseVolatilityWeightRule, &SimConfig::default()).unwrap_err();
        match err {
            SimError::RuleRefused { refusal, .. } => {
                assert_eq!(refusal.kind, RefusalKind::Data);
                assert_eq!(refusal.code, "degenerate_volatility");
            }
            other => panic!("expected RuleRefused, got {other:?}"),
        }
    }
}
