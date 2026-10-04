//! `inverse_vol_risk_parity`: a basket of assets, daily. Each asset's weight is proportional to
//! `1 / stdev(daily returns over the trailing L bars)` (lower realized vol -> higher weight), normalized so the
//! weights sum to 1.0 over the assets that have enough history. Default `L` = 60 trading days
//! ([`INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS`]), rebalanced to target on EVERY bar (`RebalancePolicy::EveryBar`).
//!
//! Per the crate's per-primitive-module convention, this is a self-contained copy of the vol-estimation logic in
//! `inverse_volatility_weight.rs` (primitive #1), not a shared helper -- the two files intentionally duplicate the
//! same math rather than factor it out, so each primitive module stands alone. Like #1, this primitive implements
//! `weightsim::WeightRule` DIRECTLY against `weightsim::HistoryView` rather than this crate's own `Panel`/
//! `PriceSeries` types: it is a generic basket rule, not a documented sleeve adapted by a downstream crate.
//!
//! # Relationship to `inverse_volatility_weight` (#1) and the spec's claimed "key structural difference"
//! The spec for this primitive states it differs from #1 "only in rebalance_policy (every_bar vs the plan's
//! default for that family) and lookback window (60 vs 20)", and asks that `every_bar` be stated explicitly as
//! "the key structural difference from #3" (`equal_weight_rebalance`, which uses `LastBarOfMonth` + `OnDecision`,
//! a monthly schedule where drift between decisions is genuinely observable).
//!
//! That comparison to #3 holds. **The comparison to #1 does not**: reading `inverse_volocity_weight.rs`'s own
//! choice 8 shows it ALSO declares `decision_schedule() = Daily` and `rebalance_policy() = EveryBar` -- and its own
//! doc comment already explains why: under a `Daily` schedule, every bar is a decision bar, so there is no "between
//! decisions" gap for units to drift in, and `OnDecision` vs `EveryBar` are operationally identical (bit-for-bit
//! the same simulated weights). So the spec's premise that there IS an observable rebalance-policy difference
//! between this primitive and #1 does not hold today: both are `Daily` + `EveryBar`, and the only thing that
//! actually distinguishes this primitive from #1 in a running simulation is the lookback window (60 vs 20 days)
//! and the (placeholder) universe. This is documented here plainly rather than silently "fixed" by picking
//! `OnDecision` for this primitive to manufacture a difference -- the spec's literal instruction (`every_bar`) is
//! implemented exactly as written, per the task's own direction to resolve a spec/structure tension by documenting
//! it, not by rewriting the spec.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Returns are SIMPLE daily returns, not log returns*: `r_k = close_k / close_{k-1} - 1`. Same convention as
//!    #1, stated explicitly since both conventions are common and are not numerically interchangeable even for a
//!    second moment like stdev.
//! 2. *Population stdev (ddof = 0)*: `sqrt(mean((r - mean(r))^2))` over exactly the trailing `L` returns, i.e.
//!    divide by `L`, not `L - 1`. Same convention as #1.
//! 3. *"Trailing L bars" of returns needs `L + 1` closes* (`L` returns are computed from `L + 1` consecutive
//!    closes), same reading as #1.
//! 4. *Insufficient history = typed exclusion, not a zero return.* An asset with fewer than `L + 1` closes visible
//!    is dropped from the normalization entirely (it is not assigned a return of 0, which would bias `stdev`
//!    downward and inflate its own weight). Concretely, its entry in the returned weight vector is forced to
//!    `0.0` and it contributes nothing to the sum used to normalize the others -- same convention as #1.
//! 5. *Structural tension with `HistoryView` (documented per the task, not resolved by changing `weightsim`).*
//!    `HistoryView` is one shared, rectangular panel: at a given decision index every asset in the universe has
//!    exactly the same number of visible bars (`HistoryView::len()`), by construction (`Panel::inner_join` keeps
//!    only dates common to every requested symbol; there is no per-asset padding or NaN). So in THIS simulator,
//!    per-asset insufficient-history exclusion can only ever be all-or-nothing at a given decision: either every
//!    asset in the universe has `>= L + 1` bars, or none of them do. The per-asset check below is still
//!    implemented exactly as specified (checking `h.closes(i).len()` independently for each `i`) rather than
//!    collapsed into a single universe-wide check, so the rule is already correct and literally spec-faithful if
//!    `weightsim` ever grows a panel representation where assets can differ in visible length. Today it is
//!    unreachable in practice because `min_history_bars()` (below) stops the simulator from calling the rule at
//!    all before bar `L` (index `L`, i.e. `L + 1` bars visible) -- at that point every asset already clears the
//!    threshold together, so the "no asset survives" refusal path also never fires once the rule starts being
//!    called, but is kept as a defensive, typed refusal rather than a silent `NaN`. Same convention as #1.
//! 6. *Zero or non-finite stdev is a refusal, NOT a silent exclusion.* The spec only names "insufficient history"
//!    (fewer than `L + 1` closes) as a reason to exclude an asset. A flat price path over the lookback window
//!    (stdev exactly 0, e.g. a frozen/stale feed) is a different failure mode -- `1 / 0` is undefined, not merely
//!    "not enough data" -- so it is raised as a typed `RefusalKind::Data` refusal for the whole decision, mirroring
//!    #1's choice 6, rather than being folded into the exclusion rule or silently producing an infinite/NaN weight.
//! 7. *Universe is a placeholder, deliberately different from #1's.* This primitive is a generic "basket of
//!    assets" rule, not tied to one documented sleeve, so [`INVERSE_VOL_RISK_PARITY_SYMBOLS`] is a small, liquid,
//!    diversified four-ETF example basket (broad equities, international developed equities, aggregate bonds,
//!    commodities) chosen only so the rule compiles and is testable. It deliberately does not reuse
//!    `INVERSE_VOLATILITY_SYMBOLS` (#1's basket) or `ETF_SYMBOLS` (a different, documented sleeve) -- the task
//!    says this primitive's universe "does not need to match #1's", so a distinct basket is used to avoid implying
//!    the two primitives are meant to run over the same universe. A caller higher up the stack assigns the real
//!    production universe later; that is out of scope here.
//! 8. *Schedule and rebalance policy.* `decision_schedule()` is `Daily` and `rebalance_policy()` is `EveryBar`,
//!    both exactly per this primitive's spec text ("rebalanced to target on EVERY bar (every_bar policy ... this
//!    is the key structural difference from #3, state it explicitly")). See the module-level section above for why
//!    this does NOT distinguish this primitive from #1 in practice, even though the spec frames it as doing so.

use std::collections::BTreeMap;

use weightsim::{DecisionSchedule, HistoryView, RebalancePolicy, RuleRefusal, WeightRule};

/// Example reference universe: four liquid, diversified ETFs (broad US equities, developed-market international
/// equities, aggregate bonds, broad commodities). Not a documented sleeve and not production data -- a placeholder
/// basket, deliberately distinct from [`crate::INVERSE_VOLATILITY_SYMBOLS`] (#1's basket), so this generic
/// primitive compiles and is testable (choice 7 above). A caller assigns the real universe later.
pub const INVERSE_VOL_RISK_PARITY_SYMBOLS: [&str; 4] = ["SPY", "EFA", "AGG", "DBC"];

/// Lookback `L`, in trading days, of trailing daily returns the stdev is computed over (default 60, per spec).
/// `L + 1` trailing closes are required per asset (choice 3 above).
pub const INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS: usize = 60;

/// `inverse_vol_risk_parity`: weight each asset proportional to `1 / stdev` of its trailing `L`-day simple daily
/// returns (population stdev, ddof = 0), normalized to sum to 1.0 over assets with enough history, rebalanced to
/// target every bar. See the module doc for every interpretation choice, in particular the section discussing
/// whether `EveryBar` actually distinguishes this primitive from `InverseVolatilityWeightRule` (#1) in practice
/// (it does not, under a `Daily` schedule -- both are `Daily` + `EveryBar`).
#[derive(Clone, Copy, Debug, Default)]
pub struct InverseVolRiskParityRule;

impl WeightRule for InverseVolRiskParityRule {
    fn id(&self) -> &'static str {
        "inverse_vol_risk_parity_60d"
    }

    fn impl_version(&self) -> String {
        concat!("reference-rules ", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn universe(&self) -> &[&'static str] {
        &INVERSE_VOL_RISK_PARITY_SYMBOLS
    }

    fn declared_parameters(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            (
                "lookback_days",
                INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS.to_string(),
            ),
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
        INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS + 1
    }

    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let n = h.n_assets();
        let needed = INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS + 1;
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
            let mut returns = Vec::with_capacity(INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS);
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
                        INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS,
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
                "no asset in the universe has {needed} trailing closes (lookback {INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS} days)"
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

    /// Builds a panel over the FULL `INVERSE_VOL_RISK_PARITY_SYMBOLS` universe (required: `simulate` rejects a
    /// panel whose symbols are not exactly the rule's `universe()`). `closes` supplies the first `closes.len()`
    /// assets; any remaining slots up to all 4 symbols are padded with an arbitrary, non-degenerate filler series
    /// so tests can focus on the one or two assets they actually care about. All assets share the same synthetic
    /// ascending daily calendar starting 2020-01-01.
    fn panel_from(mut closes: Vec<Vec<f64>>) -> Panel {
        let n_bars = closes[0].len();
        while closes.len() < INVERSE_VOL_RISK_PARITY_SYMBOLS.len() {
            let start = 10.0 * (closes.len() as f64 + 1.0);
            closes.push(noisy_series(start, n_bars, 0.02));
        }
        let dates: Vec<Date> = (0..n_bars)
            .map(|i| {
                // Simple synthetic ascending calendar: one calendar day per bar, starting 2020-01-01, rolling
                // through months of 28 days each. Fine for a unit test; HistoryView/Panel only require strictly
                // ascending dates, not a real trading calendar.
                let total_day = i as u32; // 0-based offset from 2020-01-01
                let month_offset = total_day / 28;
                let day_in_month = total_day % 28 + 1;
                let month = 1 + month_offset;
                let (y, m) = (2020 + (month - 1) / 12, (month - 1) % 12 + 1);
                d(&format!("{y:04}-{m:02}-{day_in_month:02}"))
            })
            .collect();
        let symbols: Vec<String> = INVERSE_VOL_RISK_PARITY_SYMBOLS[..closes.len()]
            .iter()
            .map(|s| s.to_string())
            .collect();
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
            let g = if i % 2 == 0 {
                1.0 + amplitude
            } else {
                1.0 / (1.0 + amplitude)
            };
            p *= g;
        }
        v
    }

    const L: usize = INVERSE_VOL_RISK_PARITY_LOOKBACK_DAYS;

    #[test]
    fn declared_parameters_reflect_the_lookback_constant() {
        let rule = InverseVolRiskParityRule;
        let params = rule.declared_parameters();
        assert_eq!(params.get("lookback_days").map(|s| s.as_str()), Some("60"));
        assert_eq!(rule.min_history_bars(), L + 1);
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        assert_eq!(rule.rebalance_policy(), RebalancePolicy::EveryBar);
        assert_eq!(rule.universe(), &INVERSE_VOL_RISK_PARITY_SYMBOLS);
    }

    // `HistoryView`'s constructor is `pub(crate)` to `weightsim` (by design: only the simulator may build one), so
    // these tests drive the rule the same way any real caller must: through the public `weightsim::simulate` entry
    // point over a `Panel`, then inspect the resulting `target_weights` column. `SimConfig::default()` is zero
    // cost / zero financing / no delay / `OnRefusal::Abort`, which is exactly what each test below wants.

    /// The last bar's target weights (row `n_bars - 1` of the flat `target_weights` matrix).
    fn last_weights(p: &Panel) -> Vec<f64> {
        let r = simulate(p, &InverseVolRiskParityRule, &SimConfig::default()).unwrap();
        r.row(&r.target_weights, p.n_bars() - 1).to_vec()
    }

    #[test]
    fn weights_sum_to_one_when_every_asset_has_enough_history() {
        let a = noisy_series(100.0, L + 1, 0.01);
        let b = noisy_series(50.0, L + 1, 0.03);
        let p = panel_from(vec![a, b]);
        let w = last_weights(&p);
        assert_eq!(w.len(), INVERSE_VOL_RISK_PARITY_SYMBOLS.len());
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "weights must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn lower_realized_vol_gets_a_higher_weight() {
        // Asset 0 is calmer (smaller amplitude) than asset 1, so it must end up with the larger weight.
        let calm = noisy_series(100.0, L + 1, 0.005);
        let volatile = noisy_series(100.0, L + 1, 0.05);
        let p = panel_from(vec![calm, volatile]);
        let w = last_weights(&p);
        assert!(
            w[0] > w[1],
            "calmer asset should get the higher weight: {w:?}"
        );
    }

    #[test]
    fn weight_is_proportional_to_inverse_stdev() {
        // All four universe slots filled explicitly (no padding via `panel_from`) with distinct, hand-computed
        // stdevs: verify the normalized weight ratios match 1/stdev ratios directly, not just an ordering.
        let a = noisy_series(100.0, L + 1, 0.01);
        let b = noisy_series(80.0, L + 1, 0.02);
        let c = noisy_series(60.0, L + 1, 0.03);
        let e = noisy_series(40.0, L + 1, 0.04);
        let p = panel_from(vec![a.clone(), b.clone(), c.clone(), e.clone()]);
        let w = last_weights(&p);

        fn population_stdev_of_returns(closes: &[f64]) -> f64 {
            let returns: Vec<f64> = (1..closes.len())
                .map(|k| closes[k] / closes[k - 1] - 1.0)
                .collect();
            let n = returns.len() as f64;
            let mean = returns.iter().sum::<f64>() / n;
            (returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n).sqrt()
        }

        let inv_vols: Vec<f64> = [&a, &b, &c, &e]
            .iter()
            .map(|s| 1.0 / population_stdev_of_returns(s))
            .collect();
        let total: f64 = inv_vols.iter().sum();
        let expected: Vec<f64> = inv_vols.iter().map(|v| v / total).collect();

        for i in 0..4 {
            assert!(
                (w[i] - expected[i]).abs() < 1e-9,
                "weight[{i}] = {} does not match expected inverse-vol proportion {}",
                w[i],
                expected[i]
            );
        }
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
        let r = simulate(&p, &InverseVolRiskParityRule, &SimConfig::default()).unwrap();
        assert!(
            r.decision.iter().all(|&d| !d),
            "no bar has enough history to decide"
        );
        assert!(
            r.refused.iter().all(|&ref_| !ref_),
            "a silent skip must not be recorded as a refusal"
        );
        assert!(r.target_weights.iter().all(|&w| w == 0.0));

        // One bar later (L + 1 bars) both assets clear the threshold and the final bar decides successfully.
        let a2 = noisy_series(100.0, L + 1, 0.01);
        let b2 = noisy_series(50.0, L + 1, 0.02);
        let p2 = panel_from(vec![a2, b2]);
        let r2 = simulate(&p2, &InverseVolRiskParityRule, &SimConfig::default()).unwrap();
        assert!(
            r2.decision[p2.n_bars() - 1],
            "L + 1 bars must be enough to decide"
        );
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
        let err = simulate(&p, &InverseVolRiskParityRule, &SimConfig::default()).unwrap_err();
        match err {
            SimError::RuleRefused { refusal, .. } => {
                assert_eq!(refusal.kind, RefusalKind::Data);
                assert_eq!(refusal.code, "degenerate_volatility");
            }
            other => panic!("expected RuleRefused, got {other:?}"),
        }
    }

    #[test]
    fn every_bar_and_on_decision_are_observationally_identical_under_daily_schedule() {
        // Direct check of the module doc's central claim: under `DecisionSchedule::Daily`, every bar is a decision
        // bar, so `rebalance_policy() = EveryBar` has no "between decisions" gap to differ from `OnDecision` in.
        // This is not a simulate() comparison (the trait only exposes one policy per rule instance); it instead
        // pins down the two preconditions that make the two policies equivalent here, so a future change to either
        // this rule's schedule or weightsim's Daily semantics would break this test rather than silently drift.
        let rule = InverseVolRiskParityRule;
        assert_eq!(rule.decision_schedule(), DecisionSchedule::Daily);
        let dates: Vec<Date> = (0..5)
            .map(|i| d(&format!("2020-01-{:02}", i + 1)))
            .collect();
        for t in 0..dates.len() {
            assert!(
                DecisionSchedule::Daily.is_decision_bar(&dates, t),
                "every bar must be a decision bar under Daily, so OnDecision vs EveryBar cannot differ"
            );
        }
    }
}
