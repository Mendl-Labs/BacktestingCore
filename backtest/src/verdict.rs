//! Rust port of `gui/src/lib/verdict.ts`'s three-tier backtest verdict.
//!
//! `verdict.ts` is the source of truth for the UI (Results page header, Job
//! List badge, Dashboard recents, Compare overlay) and is browser-only. This
//! module exists for server-side callers that need the same Promising /
//! Underperformed / Inconclusive decision without a browser in the loop --
//! initially the autonomous workflow executor (`program/src/workflow/`),
//! which must decide on its own whether to stop iterating.
//!
//! There is no shared source of truth between the two implementations --
//! this file is kept in sync with `verdict.ts` BY HAND. Any threshold or
//! gate change on one side must be mirrored on the other. The unit tests
//! below mirror `gui/src/lib/__tests__/verdict.test.ts`'s input/expected-
//! output table case-for-case so drift is caught immediately by either
//! test suite.
//!
//! Only the decision-making surface is ported (`compute_verdict`,
//! `promotion_block_reason`, `explain_low_dsr`, `VerdictThresholds`) --
//! `verdict.ts`'s UI-string helpers (`verdictLabel`, `verdictTooltip`,
//! `verdictColorClasses`, `verdictJustification`, `recommendedAction`) are
//! rendering concerns with no server-side caller and are not ported.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Promising,
    Underperformed,
    Inconclusive,
}

/// Inputs the verdict logic needs. All fields optional so callers can pass
/// partial results without pre-filtering -- mirrors `VerdictInputs` in
/// `verdict.ts`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerdictInputs {
    pub oos_sharpe_ratio: Option<f64>,
    pub oos_max_drawdown: Option<f64>,
    pub mc_p5_total_pnl: Option<f64>,
    pub is_sharpe_ratio: Option<f64>,
    pub is_max_drawdown: Option<f64>,
    pub walk_forward_enabled: Option<bool>,
    pub monte_carlo_enabled: Option<bool>,

    // --- Robustness gates (block a Promising verdict when explicitly bad) ---
    pub total_trades: Option<i64>,
    pub result_quality: Option<String>,
    pub is_statistically_significant: Option<bool>,
    pub is_overfit: Option<bool>,

    // --- Quant-hardening gates ---
    pub analysis_mode: Option<String>,
    pub last_oos_window_sharpe: Option<f64>,
    pub deflated_sharpe_ratio: Option<f64>,
    pub num_strategies_tested: Option<i64>,
    pub sharpe_std_error: Option<f64>,
    pub expected_max_sharpe_under_null: Option<f64>,
    pub information_ratio: Option<f64>,
    pub bb_p5_total_pnl: Option<f64>,
    pub wf_consistency_score: Option<f64>,
    pub oos_num_trades: Option<i64>,
    pub oos_profitability_rate: Option<f64>,
    /// Bailey & López de Prado minimum OOS track record length -- how many
    /// OOS observations are needed to reject Sharpe <= 0 at 95% confidence,
    /// given this run's own OOS Sharpe standard error. Distinct from the DSR
    /// gate: DSR corrects for multiple-comparisons search bias, this
    /// corrects for the OOS sample simply being too short to trust the
    /// Sharpe point estimate at all. Compare only against
    /// `oos_n_observations` from the same `oos_statistical_significance`
    /// result, never against `oos_num_trades`.
    pub oos_min_track_record_length: Option<i64>,
    /// Actual OOS observation count backing `oos_min_track_record_length`.
    pub oos_n_observations: Option<i64>,
    pub strategy_type: Option<String>,
    pub has_fee_config: Option<bool>,
    /// True when a cross-venue arbitrage backtest's opportunity collapses to
    /// zero/negative PnL at a moderate (1.5x) widening of the synthetic-spread
    /// assumption. `None` for every non-arbitrage run (never gates there).
    pub arbitrage_spread_fragile: Option<bool>,

    // --- Diversified portfolio gates ---
    pub is_portfolio: Option<bool>,
    pub portfolio_sharpe: Option<f64>,
    pub portfolio_max_drawdown: Option<f64>,
    pub worst_asset_max_drawdown: Option<f64>,
    pub num_assets_failing_gate: Option<i64>,
    pub pooled_deflated_sharpe_ratio: Option<f64>,

    // --- User-personalized gates (LaunchForm risk-comfort selection) ---
    /// User-stated max-drawdown tolerance for THIS run, as a fraction (0.10 =
    /// 10%), from the LaunchForm's risk-comfort dropdown. Safety-preserving:
    /// can only TIGHTEN the platform's own `UNDERPERFORMED_MAX_DRAWDOWN`
    /// ceiling, never loosen it -- see `effective_max_drawdown_cap`. `None`
    /// when the user picked "Not sure" or the run wasn't launched through
    /// LaunchForm at all.
    pub user_max_drawdown_tolerance: Option<f64>,

    /// Tenant-level DSR promotion floor (`tenants.dsr_floor_override`,
    /// DB-enforced `>= 0.95`). Only ever RAISES the effective bar above
    /// `PROMISING_MIN_DSR` -- see `effective_min_dsr`. `None` for tenants
    /// that haven't set one (the platform default applies).
    pub dsr_floor_override: Option<f64>,
}

/// The drawdown ceiling actually enforced for this run: the platform's own
/// fixed ceiling, tightened by a user-stated tolerance when one was given.
/// Never loosened -- a `user_max_drawdown_tolerance` above the platform
/// ceiling is clamped, not honored. Mirrors `effectiveMaxDrawdownCap` in
/// `verdict.ts`.
fn effective_max_drawdown_cap(inputs: &VerdictInputs) -> f64 {
    inputs
        .user_max_drawdown_tolerance
        .map(|t| t.min(VerdictThresholds::UNDERPERFORMED_MAX_DRAWDOWN))
        .unwrap_or(VerdictThresholds::UNDERPERFORMED_MAX_DRAWDOWN)
}

/// Threshold constants. Mirrors `VERDICT_THRESHOLDS` in `verdict.ts`.
pub struct VerdictThresholds;
impl VerdictThresholds {
    /// 2026-08-03: lowered from 1.0 -- the platform's strategy universe is
    /// OHLCV-only (no order flow, alt-data, or fundamentals), so 0.3-0.6 is
    /// the realistic range for a genuine, non-overfit OOS edge (in line with
    /// live CTA/trend-follower Sharpes on price-only signals).
    pub const PROMISING_MIN_SHARPE: f64 = 0.5;
    pub const PROMISING_MIN_MC_P5_PNL: f64 = 0.0;
    pub const UNDERPERFORMED_MAX_SHARPE: f64 = 0.0;
    pub const UNDERPERFORMED_MAX_DRAWDOWN: f64 = 0.25;
    pub const PROMISING_MIN_TRADES: i64 = 100;
    pub const INSUFFICIENT_MIN_TRADES: i64 = 30;
    /// DSR threshold below which the GA likely over-fitted (Bailey et al. 2014).
    pub const PROMISING_MIN_DSR: f64 = 0.95;
    /// Information Ratio threshold: strategy must beat the risk-free rate on OOS data.
    pub const PROMISING_MIN_IR: f64 = 0.0;
}

/// The DSR bar actually enforced for this run: the platform's own fixed
/// floor, raised by a tenant-stated override when one is set. Never
/// lowered -- a `dsr_floor_override` below the platform floor is clamped,
/// not honored (the DB CHECK constraint should already prevent this, but
/// the gate stays safe even if that invariant is ever violated some other
/// way). Mirrors `effectiveMinDsr` in `verdict.ts`.
fn effective_min_dsr(inputs: &VerdictInputs) -> f64 {
    inputs
        .dsr_floor_override
        .map(|floor| floor.max(VerdictThresholds::PROMISING_MIN_DSR))
        .unwrap_or(VerdictThresholds::PROMISING_MIN_DSR)
}

/// Single-leg catastrophic-failure floor for a diversified portfolio.
/// Stricter than the portfolio-level Underperformed drawdown threshold (25%)
/// since one bad leg can still be masked by blending with healthy ones.
const PORTFOLIO_WORST_ASSET_MAX_DRAWDOWN: f64 = 0.40;

/// Robustness gate. Returns a short human-readable reason when a run that
/// would otherwise be Promising must be held back to Inconclusive, or `None`
/// when no gate fires. Only explicitly-bad signals trigger a gate --
/// missing/`None` fields never do. Mirrors `promotionBlockReason`.
pub fn promotion_block_reason(inputs: &VerdictInputs) -> Option<String> {
    if inputs.analysis_mode.as_deref() == Some("development") {
        return Some(
            "development mode — run in Production mode for out-of-sample validation".to_string(),
        );
    }

    if inputs.is_overfit == Some(true) {
        return Some("overfitting detected (large in-sample vs out-of-sample gap)".to_string());
    }
    if inputs.is_statistically_significant == Some(false) {
        return Some("results are not statistically significant".to_string());
    }

    // Minimum track record length gate (Bailey & López de Prado): checked
    // BEFORE the crude trade-count floor below on purpose (design council,
    // 2026-08-31) -- when it has data, it gives a mechanism-adaptive answer
    // derived from THIS result's own Sharpe standard error ("needs ~N more
    // OOS observations to trust this Sharpe") instead of a flat "too few
    // trades" that's identical regardless of what was actually tested or how
    // close the sample is to sufficient. Even a run that clears DSR can
    // still have too short an OOS sample to trust the Sharpe estimate at
    // all -- DSR corrects for how much was searched, this corrects for how
    // little data backs the number being reported. Falls through to the
    // crude floor below when this data isn't available (no walk-forward
    // ran, or a portfolio path without per-asset OOS stats).
    if let (Some(min_trl), Some(n_obs)) =
        (inputs.oos_min_track_record_length, inputs.oos_n_observations)
    {
        if n_obs < min_trl {
            return Some(format!(
                "only {} OOS observations — minimum track record length for this Sharpe estimate is {}",
                n_obs, min_trl
            ));
        }
    }

    let quality = inputs.result_quality.as_deref();
    if matches!(quality, Some("insufficient") | Some("limited")) {
        return Some("too few trades for statistical confidence".to_string());
    }
    if quality != Some("sufficient") {
        if let Some(trades) = inputs.total_trades {
            if trades < VerdictThresholds::PROMISING_MIN_TRADES {
                return Some("too few trades for statistical confidence".to_string());
            }
        }
    }

    // Recency gate: most recent OOS window must not be negative Sharpe.
    if let Some(w) = inputs.last_oos_window_sharpe {
        if w < 0.0 {
            return Some(format!(
                "most recent OOS window had Sharpe {:.2} — strategy may have recently degraded",
                w
            ));
        }
    }

    // Deflated Sharpe gate: corrects for GA multiple-comparisons selection bias.
    if let Some(dsr) = inputs.deflated_sharpe_ratio {
        let min_dsr = effective_min_dsr(inputs);
        if dsr < min_dsr {
            return Some(format!(
                "Deflated Sharpe Ratio {:.2} below {} — likely overfitting from parameter search",
                dsr,
                min_dsr
            ));
        }
    }

    // Information Ratio gate.
    if let Some(ir) = inputs.information_ratio {
        if ir < VerdictThresholds::PROMISING_MIN_IR {
            return Some(
                "information ratio negative — strategy did not generate excess return above the risk-free rate on OOS data"
                    .to_string(),
            );
        }
    }

    // Block-bootstrap gate: preserves trade-return autocorrelation.
    if let Some(bb) = inputs.bb_p5_total_pnl {
        if bb < 0.0 {
            return Some(
                "block-bootstrap 5th-percentile PnL negative — strategy edge may not survive trade-sequence autocorrelation"
                    .to_string(),
            );
        }
    }

    // WFO consistency gate.
    if let Some(wc) = inputs.wf_consistency_score {
        if wc < 0.6 {
            return Some(format!(
                "walk-forward consistency {:.0}% — strategy did not perform consistently across OOS windows",
                wc * 100.0
            ));
        }
    }

    // WFO profitability-rate gate.
    if let Some(pr) = inputs.oos_profitability_rate {
        if pr < 0.5 {
            return Some(format!(
                "only {:.0}% of walk-forward windows were profitable — strategy did not beat zero return in most periods",
                pr * 100.0
            ));
        }
    }

    // Arbitrage spread-sensitivity gate (cross-exchange strategy rigor gap-close).
    if inputs.arbitrage_spread_fragile == Some(true) {
        return Some(
            "arbitrage opportunity does not survive a moderate spread-assumption stress test"
                .to_string(),
        );
    }

    // Portfolio gate.
    if inputs.is_portfolio == Some(true) {
        if let Some(n) = inputs.num_assets_failing_gate {
            if n > 0 {
                let plural = if n != 1 { "s" } else { "" };
                return Some(format!(
                    "{} portfolio asset{} failed their own promotion gate individually",
                    n, plural
                ));
            }
        }
    }

    None
}

/// Compute the verdict tier. Mirrors `computeVerdict`.
pub fn compute_verdict(inputs: &VerdictInputs) -> Verdict {
    if inputs.is_portfolio == Some(true) {
        return compute_portfolio_verdict(inputs);
    }

    let sharpe = inputs.oos_sharpe_ratio.or(inputs.is_sharpe_ratio);
    let drawdown = inputs.oos_max_drawdown.or(inputs.is_max_drawdown);
    let mc_p5 = inputs.mc_p5_total_pnl;

    if let Some(s) = sharpe {
        if s < VerdictThresholds::UNDERPERFORMED_MAX_SHARPE {
            return Verdict::Underperformed;
        }
    }
    if let Some(d) = drawdown {
        if d > effective_max_drawdown_cap(inputs) {
            return Verdict::Underperformed;
        }
    }

    let sharpe_strong = sharpe
        .map(|s| s > VerdictThresholds::PROMISING_MIN_SHARPE)
        .unwrap_or(false);
    let mc_robust = match mc_p5 {
        None => !inputs.monte_carlo_enabled.unwrap_or(false),
        Some(p) => p > VerdictThresholds::PROMISING_MIN_MC_P5_PNL,
    };

    if sharpe_strong && mc_robust {
        return if promotion_block_reason(inputs).is_some() {
            Verdict::Inconclusive
        } else {
            Verdict::Promising
        };
    }

    Verdict::Inconclusive
}

/// Verdict tier for a diversified portfolio job. Mirrors `computePortfolioVerdict`.
fn compute_portfolio_verdict(inputs: &VerdictInputs) -> Verdict {
    let sharpe = inputs.portfolio_sharpe;
    let drawdown = inputs.portfolio_max_drawdown;
    let mc_p5 = inputs.mc_p5_total_pnl;
    let worst_asset_drawdown = inputs.worst_asset_max_drawdown;

    if let Some(s) = sharpe {
        if s < VerdictThresholds::UNDERPERFORMED_MAX_SHARPE {
            return Verdict::Underperformed;
        }
    }
    if let Some(d) = drawdown {
        if d > effective_max_drawdown_cap(inputs) {
            return Verdict::Underperformed;
        }
    }
    if let Some(w) = worst_asset_drawdown {
        if w > PORTFOLIO_WORST_ASSET_MAX_DRAWDOWN {
            return Verdict::Underperformed;
        }
    }

    let sharpe_strong = sharpe
        .map(|s| s > VerdictThresholds::PROMISING_MIN_SHARPE)
        .unwrap_or(false);
    let mc_robust = match mc_p5 {
        None => !inputs.monte_carlo_enabled.unwrap_or(false),
        Some(p) => p > VerdictThresholds::PROMISING_MIN_MC_P5_PNL,
    };

    if sharpe_strong && mc_robust {
        return if promotion_block_reason(inputs).is_some() {
            Verdict::Inconclusive
        } else {
            Verdict::Promising
        };
    }

    Verdict::Inconclusive
}

/// Trial count above which a large search looks like a meaningful contributor
/// to a low DSR. Heuristic, not derived from the DSR formula itself.
const DSR_EXPLAIN_LARGE_TRIAL_COUNT: i64 = 1000;
/// Sharpe standard-error above which the estimate itself looks too uncertain
/// to trust yet. Same heuristic caveat as the trial-count threshold above.
const DSR_EXPLAIN_HIGH_SHARPE_STD_ERROR: f64 = 0.3;

/// When DSR is below the Promising threshold, produce a plain-language
/// explanation of which factor looks like a driver. Returns `None` when DSR
/// is missing, not actually below threshold, or neither factor looks large.
/// Purely explanatory -- never gates anything itself (`promotion_block_reason`
/// already does that). Mirrors `explainLowDsr`.
pub fn explain_low_dsr(inputs: &VerdictInputs) -> Option<String> {
    let dsr = inputs.deflated_sharpe_ratio?;
    if dsr >= effective_min_dsr(inputs) {
        return None;
    }

    let mut reasons: Vec<String> = Vec::new();
    if let Some(n) = inputs.num_strategies_tested {
        if n > DSR_EXPLAIN_LARGE_TRIAL_COUNT {
            reasons.push(format!(
                "{} strategy variations were effectively tried (including earlier iterations on this same research thread) — the more that were tried, the higher the bar this Sharpe has to clear to look like more than luck",
                n
            ));
        }
    }
    if let Some(se) = inputs.sharpe_std_error {
        if se > DSR_EXPLAIN_HIGH_SHARPE_STD_ERROR {
            reasons.push(format!(
                "the Sharpe estimate itself is uncertain (standard error {:.2}) — likely too few trades or too short a test period to trust the number yet",
                se
            ));
        }
    }

    let mut explanation = if !reasons.is_empty() {
        Some(format!("Likely driver: {}.", reasons.join("; and ")))
    } else {
        None
    };

    let sharpe = inputs.oos_sharpe_ratio.or(inputs.is_sharpe_ratio);
    if let (Some(bar), Some(s)) = (inputs.expected_max_sharpe_under_null, sharpe) {
        let bar_sentence = format!(
            "Your Sharpe of {:.2} needed to clear ~{:.2} to be considered significant given the search behind it.",
            s, bar
        );
        explanation = Some(match explanation {
            Some(e) => format!("{} {}", e, bar_sentence),
            None => bar_sentence,
        });
    }

    explanation
}

#[cfg(test)]
mod tests {
    use super::*;

    fn promising_base() -> VerdictInputs {
        VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.1),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        }
    }

    // --- compute_verdict ---

    #[test]
    fn returns_promising_when_oos_sharpe_above_1_and_mc_p5_pnl_positive() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(1.5),
            oos_max_drawdown: Some(0.1),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn returns_underperformed_when_oos_sharpe_below_0() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(-0.5),
            oos_max_drawdown: Some(0.1),
            mc_p5_total_pnl: Some(100.0),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn returns_underperformed_when_max_drawdown_above_25pct() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.3),
            mc_p5_total_pnl: Some(100.0),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    // --- user_max_drawdown_tolerance personalization ---

    #[test]
    fn tightens_underperformed_gate_when_user_tolerance_is_stricter() {
        // 18% drawdown clears the platform's 25% ceiling but not this
        // user's stated 10% comfort level.
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.18),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            user_max_drawdown_tolerance: Some(0.10),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn user_tolerance_looser_than_platform_ceiling_is_clamped_not_loosened() {
        // 28% drawdown breaches the platform's 25% ceiling even though the
        // user claimed they could stomach 30% -- the platform's own ceiling
        // still wins, it never loosens.
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.28),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            user_max_drawdown_tolerance: Some(0.30),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn absent_user_tolerance_behaves_exactly_as_before() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.3),
            mc_p5_total_pnl: Some(100.0),
            user_max_drawdown_tolerance: None,
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);

        let promising = promising_base();
        assert_eq!(compute_verdict(&promising), Verdict::Promising);
    }

    #[test]
    fn returns_inconclusive_on_borderline_sharpe_with_mc_robust() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(0.5),
            oos_max_drawdown: Some(0.1),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
    }

    #[test]
    fn returns_inconclusive_when_mc_ran_but_p5_pnl_non_positive_even_with_strong_sharpe() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.1),
            mc_p5_total_pnl: Some(-50.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
    }

    #[test]
    fn allows_promising_on_sharpe_alone_when_mc_did_not_run() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            oos_max_drawdown: Some(0.1),
            monte_carlo_enabled: Some(false),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn falls_back_to_in_sample_sharpe_when_oos_is_missing() {
        let inputs = VerdictInputs {
            is_sharpe_ratio: Some(2.0),
            is_max_drawdown: Some(0.1),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn returns_inconclusive_when_every_signal_is_missing() {
        assert_eq!(compute_verdict(&VerdictInputs::default()), Verdict::Inconclusive);
    }

    // --- robustness gate ---

    #[test]
    fn does_not_block_a_clean_promising_run() {
        let inputs = promising_base();
        assert!(promotion_block_reason(&inputs).is_none());
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn demotes_promising_to_inconclusive_when_overfit() {
        let inputs = VerdictInputs {
            is_overfit: Some(true),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs).unwrap().contains("overfit"));
    }

    #[test]
    fn demotes_promising_to_inconclusive_when_not_statistically_significant() {
        let inputs = VerdictInputs {
            is_statistically_significant: Some(false),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs).unwrap().contains("significant"));
    }

    #[test]
    fn demotes_promising_when_result_quality_is_limited_or_insufficient() {
        let limited = VerdictInputs {
            result_quality: Some("limited".to_string()),
            ..promising_base()
        };
        let insufficient = VerdictInputs {
            result_quality: Some("insufficient".to_string()),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&limited), Verdict::Inconclusive);
        assert_eq!(compute_verdict(&insufficient), Verdict::Inconclusive);
    }

    #[test]
    fn demotes_promising_when_trade_count_is_below_the_minimum() {
        let inputs = VerdictInputs {
            total_trades: Some(VerdictThresholds::PROMISING_MIN_TRADES - 1),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
    }

    #[test]
    fn allows_promising_at_or_above_the_trade_count_minimum() {
        let inputs = VerdictInputs {
            total_trades: Some(VerdictThresholds::PROMISING_MIN_TRADES),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn lets_an_explicit_sufficient_quality_override_a_low_trade_count() {
        let inputs = VerdictInputs {
            result_quality: Some("sufficient".to_string()),
            total_trades: Some(10),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn does_not_fire_on_missing_robustness_data() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(2.0),
            ..Default::default()
        };
        assert!(promotion_block_reason(&inputs).is_none());
    }

    #[test]
    fn keeps_underperformed_unaffected_by_robustness_gates() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(-0.5),
            is_overfit: Some(true),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn demotes_promising_to_inconclusive_when_arbitrage_spread_fragile() {
        let inputs = VerdictInputs {
            arbitrage_spread_fragile: Some(true),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs)
            .unwrap()
            .contains("spread-assumption stress test"));
    }

    #[test]
    fn does_not_block_promising_when_arbitrage_spread_fragile_is_false_or_absent() {
        let false_case = VerdictInputs {
            arbitrage_spread_fragile: Some(false),
            ..promising_base()
        };
        assert!(promotion_block_reason(&false_case).is_none());
        assert!(promotion_block_reason(&promising_base()).is_none());
    }

    #[test]
    fn demotes_promising_when_oos_observations_are_below_the_minimum_track_record_length() {
        let inputs = VerdictInputs {
            oos_min_track_record_length: Some(200),
            oos_n_observations: Some(150),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs)
            .unwrap()
            .contains("minimum track record length"));
    }

    #[test]
    fn allows_promising_when_oos_observations_meet_the_minimum_track_record_length() {
        let inputs = VerdictInputs {
            oos_min_track_record_length: Some(200),
            oos_n_observations: Some(200),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn does_not_fire_the_minimum_track_record_length_gate_on_missing_data() {
        let only_min = VerdictInputs {
            oos_min_track_record_length: Some(200),
            ..promising_base()
        };
        assert!(promotion_block_reason(&only_min).is_none());
        let only_n_obs = VerdictInputs {
            oos_n_observations: Some(50),
            ..promising_base()
        };
        assert!(promotion_block_reason(&only_n_obs).is_none());
    }

    // Design council 2026-08-31: when a thin-sample result has BOTH a crude
    // "insufficient trades" quality tag AND the more specific min-track-record
    // data, the mechanism-adaptive reason must win -- it derives the required
    // observation count from THIS result's own Sharpe standard error instead
    // of citing the same flat trade-count floor regardless of what was tested.
    #[test]
    fn prefers_the_adaptive_min_track_record_reason_over_the_crude_too_few_trades_floor_when_both_apply() {
        let inputs = VerdictInputs {
            result_quality: Some("insufficient".to_string()),
            total_trades: Some(3),
            oos_min_track_record_length: Some(45),
            oos_n_observations: Some(3),
            ..promising_base()
        };
        let reason = promotion_block_reason(&inputs).unwrap();
        assert!(reason.contains("minimum track record length"));
        assert!(!reason.contains("too few trades"));
    }

    #[test]
    fn falls_back_to_the_too_few_trades_floor_when_min_track_record_data_is_absent() {
        let inputs = VerdictInputs {
            result_quality: Some("insufficient".to_string()),
            total_trades: Some(3),
            ..promising_base()
        };
        assert!(promotion_block_reason(&inputs)
            .unwrap()
            .contains("too few trades"));
    }

    // --- dsr_floor_override ---

    #[test]
    fn dsr_floor_override_blocks_promising_between_the_platform_floor_and_the_override() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.97),
            dsr_floor_override: Some(1.2),
            ..promising_base()
        };
        // Clears the platform's own 0.95 floor but not this tenant's 1.2 override.
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs).unwrap().contains("1.2"));
    }

    #[test]
    fn dsr_floor_override_allows_promising_once_it_clears_the_raised_bar() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(1.25),
            dsr_floor_override: Some(1.2),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
        assert!(promotion_block_reason(&inputs).is_none());
    }

    #[test]
    fn dsr_floor_override_below_the_platform_floor_is_clamped_not_loosened() {
        // The DB CHECK constraint (`tenants_dsr_floor_override_min`) should
        // already prevent this, but the gate stays safe even if that
        // invariant is ever violated some other way.
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.6),
            dsr_floor_override: Some(0.1),
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs).unwrap().contains("0.95"));
    }

    #[test]
    fn absent_dsr_floor_override_behaves_exactly_as_before() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.96),
            dsr_floor_override: None,
            ..promising_base()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
        assert!(promotion_block_reason(&inputs).is_none());
    }

    // --- explain_low_dsr ---

    #[test]
    fn explain_low_dsr_returns_none_when_dsr_is_missing() {
        let inputs = VerdictInputs {
            oos_sharpe_ratio: Some(1.5),
            ..Default::default()
        };
        assert_eq!(explain_low_dsr(&inputs), None);
    }

    #[test]
    fn explain_low_dsr_returns_none_when_dsr_already_clears_threshold() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(VerdictThresholds::PROMISING_MIN_DSR),
            num_strategies_tested: Some(5000),
            ..Default::default()
        };
        assert_eq!(explain_low_dsr(&inputs), None);
    }

    #[test]
    fn explain_low_dsr_returns_none_when_neither_heuristic_nor_bar_present() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            ..Default::default()
        };
        assert_eq!(explain_low_dsr(&inputs), None);
    }

    #[test]
    fn explain_low_dsr_flags_a_large_trial_count() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            num_strategies_tested: Some(5000),
            ..Default::default()
        };
        assert!(explain_low_dsr(&inputs).unwrap().contains("5000 strategy variations"));
    }

    #[test]
    fn explain_low_dsr_flags_a_high_sharpe_standard_error() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            sharpe_std_error: Some(0.5),
            ..Default::default()
        };
        let explanation = explain_low_dsr(&inputs).unwrap();
        assert!(explanation.contains("Sharpe estimate itself is uncertain"));
        assert!(explanation.contains("0.50"));
    }

    #[test]
    fn explain_low_dsr_states_the_effective_bar_appended_to_heuristic_reasons() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            num_strategies_tested: Some(5000),
            oos_sharpe_ratio: Some(0.9),
            expected_max_sharpe_under_null: Some(1.8),
            ..Default::default()
        };
        let explanation = explain_low_dsr(&inputs).unwrap();
        assert!(explanation.contains("5000 strategy variations"));
        assert!(explanation.contains("Your Sharpe of 0.90 needed to clear ~1.80"));
    }

    #[test]
    fn explain_low_dsr_states_effective_bar_alone_when_neither_heuristic_fires() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            oos_sharpe_ratio: Some(0.9),
            expected_max_sharpe_under_null: Some(1.8),
            ..Default::default()
        };
        assert_eq!(
            explain_low_dsr(&inputs).unwrap(),
            "Your Sharpe of 0.90 needed to clear ~1.80 to be considered significant given the search behind it."
        );
    }

    #[test]
    fn explain_low_dsr_falls_back_to_in_sample_sharpe_for_the_bar_sentence() {
        let inputs = VerdictInputs {
            deflated_sharpe_ratio: Some(0.2),
            is_sharpe_ratio: Some(0.7),
            expected_max_sharpe_under_null: Some(1.5),
            ..Default::default()
        };
        assert!(explain_low_dsr(&inputs)
            .unwrap()
            .contains("Your Sharpe of 0.70 needed to clear ~1.50"));
    }

    // --- portfolio verdict ---

    #[test]
    fn portfolio_verdict_promising_on_healthy_blend() {
        let inputs = VerdictInputs {
            is_portfolio: Some(true),
            portfolio_sharpe: Some(1.5),
            portfolio_max_drawdown: Some(0.1),
            worst_asset_max_drawdown: Some(0.2),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Promising);
    }

    #[test]
    fn portfolio_verdict_tightens_underperformed_gate_when_user_tolerance_is_stricter() {
        let inputs = VerdictInputs {
            is_portfolio: Some(true),
            portfolio_sharpe: Some(1.5),
            portfolio_max_drawdown: Some(0.18),
            worst_asset_max_drawdown: Some(0.2),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            user_max_drawdown_tolerance: Some(0.10),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn portfolio_verdict_user_tolerance_looser_than_platform_ceiling_is_clamped() {
        let inputs = VerdictInputs {
            is_portfolio: Some(true),
            portfolio_sharpe: Some(1.5),
            portfolio_max_drawdown: Some(0.28),
            worst_asset_max_drawdown: Some(0.2),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            user_max_drawdown_tolerance: Some(0.30),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn portfolio_verdict_underperformed_when_worst_asset_drawdown_exceeds_floor() {
        let inputs = VerdictInputs {
            is_portfolio: Some(true),
            portfolio_sharpe: Some(1.5),
            portfolio_max_drawdown: Some(0.1),
            worst_asset_max_drawdown: Some(0.5),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Underperformed);
    }

    #[test]
    fn portfolio_verdict_demoted_when_assets_fail_their_own_gate() {
        let inputs = VerdictInputs {
            is_portfolio: Some(true),
            portfolio_sharpe: Some(1.5),
            portfolio_max_drawdown: Some(0.1),
            worst_asset_max_drawdown: Some(0.2),
            mc_p5_total_pnl: Some(100.0),
            monte_carlo_enabled: Some(true),
            num_assets_failing_gate: Some(1),
            ..Default::default()
        };
        assert_eq!(compute_verdict(&inputs), Verdict::Inconclusive);
        assert!(promotion_block_reason(&inputs)
            .unwrap()
            .contains("failed their own promotion gate"));
    }
}
