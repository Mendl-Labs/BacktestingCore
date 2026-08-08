//! Strategy Deployment Preparation from Walk-Forward Analysis
//!
//! This module prepares strategies for user-controlled deployment after walk-forward
//! analysis. It does NOT auto-export or auto-deploy - it creates a DeploymentRequest
//! that the user must review and approve.
//!
//! ## Philosophy
//!
//! The walk-forward analysis validates strategy correctness. But the decision to
//! deploy is the USER's choice, not the platform's. This module:
//! 
//! 1. Aggregates walk-forward results into deployment-ready metrics
//! 2. Generates advisory warnings (informational only)
//! 3. Creates a DeploymentRequest for user review
//! 4. Does NOT automatically export or deploy anything

use crate::result::WalkForwardSummary;
use crate::logging_facade::WALKFORWARD_LOGGER;
use crate::{log_info, log_warn};
use strategy::deployment::{
    DeploymentRequest, BacktestMetrics, UserRiskConfig, ApprovalRequirement,
};
use strategy::Strategy as StrategyTrait;
use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

/// Deployment preparation result
#[derive(Debug)]
pub struct DeploymentPreparation {
    /// The deployment request for user review
    pub request: DeploymentRequest,
    /// Summary of what was analyzed
    pub analysis_summary: AnalysisSummary,
}

/// Summary of the walk-forward analysis
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    pub total_windows: usize,
    pub profitable_windows: usize,
    pub average_sharpe: f64,
    pub parameter_stability: f64,
    pub recommendation: AnalysisRecommendation,
}

/// Non-binding recommendation based on analysis (user can ignore)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisRecommendation {
    /// Metrics look solid
    LooksGood,
    /// Metrics are acceptable but have some concerns
    ProceedWithCaution,
    /// Metrics suggest high risk
    HighRisk,
    /// Results are questionable
    Questionable,
}

/// Prepares deployment requests from walk-forward results
pub struct DeploymentPreparer {
    user_id: Uuid,
}

impl DeploymentPreparer {
    pub fn new(user_id: Uuid) -> Self {
        Self { user_id }
    }

    /// Prepare a deployment request from walk-forward results
    /// 
    /// This does NOT deploy anything - it creates a request for user review
    pub fn prepare_deployment(
        &self,
        wf_summary: &WalkForwardSummary,
        _strategy: Arc<dyn StrategyTrait>,
        strategy_name: &str,
        symbol: &str,
        exchange: &str,
        user_config: UserRiskConfig,
    ) -> Result<DeploymentPreparation> {
        log_info!(WALKFORWARD_LOGGER, "📋 Preparing deployment request for user review...");
        
        // 1. Convert walk-forward results to BacktestMetrics
        let metrics = self.aggregate_metrics(wf_summary);
        
        // 2. Create analysis summary
        let analysis_summary = self.create_analysis_summary(wf_summary);
        
        log_info!(WALKFORWARD_LOGGER, "📊 Analysis Summary:");
        log_info!(WALKFORWARD_LOGGER, "   • Windows analyzed: {}", analysis_summary.total_windows);
        log_info!(WALKFORWARD_LOGGER, "   • Profitable windows: {}/{}", 
            analysis_summary.profitable_windows, analysis_summary.total_windows);
        log_info!(WALKFORWARD_LOGGER, "   • Average Sharpe: {:.2}", analysis_summary.average_sharpe);
        log_info!(WALKFORWARD_LOGGER, "   • Parameter stability: {:.2}", analysis_summary.parameter_stability);
        log_info!(WALKFORWARD_LOGGER, "   • Recommendation: {:?}", analysis_summary.recommendation);
        
        // 3. Create deployment request (user must review and approve)
        let request = DeploymentRequest::new(
            self.user_id,
            format!("{}-{}-{}", strategy_name.to_lowercase(), symbol.to_lowercase(), &Uuid::new_v4().to_string()[..8]),
            strategy_name.to_string(),
            symbol.to_string(),
            exchange.to_string(),
            metrics,
            user_config,
        );
        
        // 4. Log advisory warnings (informational)
        if !request.advisory_warnings.is_empty() {
            log_info!(WALKFORWARD_LOGGER, "⚠️ Advisory Warnings ({} total):", request.advisory_warnings.len());
            for warning in &request.advisory_warnings {
                log_info!(WALKFORWARD_LOGGER, "   • [{:?}] {}", warning.severity, warning.message);
            }
            log_info!(WALKFORWARD_LOGGER, "   Note: These are informational only. Deployment decision is yours.");
        }
        
        log_info!(WALKFORWARD_LOGGER, "✅ Deployment request prepared (ID: {})", request.request_id);
        log_info!(WALKFORWARD_LOGGER, "📝 Status: {:?} - awaiting user review", request.status);
        log_info!(WALKFORWARD_LOGGER, "🔐 Required acknowledgments: {}", request.risk_acknowledgments.len());
        
        if request.platform_validation.required_approval != ApprovalRequirement::None {
            log_info!(WALKFORWARD_LOGGER, "👤 Admin approval required: {:?}", 
                request.platform_validation.required_approval);
        }
        
        Ok(DeploymentPreparation {
            request,
            analysis_summary,
        })
    }

    /// Aggregate walk-forward results into BacktestMetrics
    fn aggregate_metrics(&self, wf_summary: &WalkForwardSummary) -> BacktestMetrics {
        let total_windows = wf_summary.window_results.len() as f64;
        
        // Aggregate across all windows
        let total_net_profit: f64 = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.net_profit)
            .sum();
        
        let avg_profit_factor = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.profit_factor)
            .sum::<f64>() / total_windows;
        
        let max_drawdown = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.max_drawdown)
            .fold(0.0, f64::max);
        
        let avg_drawdown = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.max_drawdown)
            .sum::<f64>() / total_windows;
        
        let total_trades: usize = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.number_of_trades)
            .sum();
        
        let avg_win_rate = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.win_rate)
            .sum::<f64>() / total_windows;
        
        let avg_volatility = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.volatility)
            .sum::<f64>() / total_windows;
        
        let avg_trade_return = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.average_trade_return)
            .sum::<f64>() / total_windows;
        
        // Estimate winning/losing trade averages from available metrics
        // Using gross_profit/loss and win_rate to derive these
        let avg_winning_trade = wf_summary.window_results.iter()
            .map(|w| {
                let winning_trades = (w.test_metrics.number_of_trades as f64 * w.test_metrics.win_rate).max(1.0);
                w.test_metrics.gross_profit / winning_trades
            })
            .sum::<f64>() / total_windows.max(1.0);
        
        let avg_losing_trade = wf_summary.window_results.iter()
            .map(|w| {
                let losing_trades = (w.test_metrics.number_of_trades as f64 * (1.0 - w.test_metrics.win_rate)).max(1.0);
                -w.test_metrics.gross_loss.abs() / losing_trades
            })
            .sum::<f64>() / total_windows.max(1.0);
        
        let avg_trade_duration = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.average_trade_duration / 3600.0)
            .sum::<f64>() / total_windows;
        
        // Calculate date range
        let backtest_start = wf_summary.window_results.first()
            .map(|w| w.train_start.and_hms_opt(0, 0, 0).unwrap().and_utc())
            .unwrap_or_else(Utc::now);
        
        let backtest_end = wf_summary.window_results.last()
            .map(|w| w.test_end.and_hms_opt(23, 59, 59).unwrap().and_utc())
            .unwrap_or_else(Utc::now);
        
        let total_days = (backtest_end - backtest_start).num_days() as u32;
        
        // Calculate percentiles from window results
        let sharpe_values: Vec<f64> = wf_summary.window_results.iter()
            .map(|w| w.test_metrics.sharpe_ratio)
            .collect();
        
        let (p5, median, p95) = percentiles(&sharpe_values);
        
        let profitable_windows = wf_summary.window_results.iter()
            .filter(|w| w.test_metrics.net_profit > 0.0)
            .count();
        
        BacktestMetrics {
            sharpe_ratio: wf_summary.average_test_sharpe,
            sortino_ratio: wf_summary.average_test_sharpe * 1.2, // Estimate
            calmar_ratio: if max_drawdown > 0.0 { (total_net_profit / 10000.0) / max_drawdown } else { 0.0 },
            profit_factor: avg_profit_factor,
            win_rate: avg_win_rate,
            total_return_pct: total_net_profit / 10000.0,
            annualized_return_pct: if total_days > 0 { (total_net_profit / 10000.0) * (365.0 / total_days as f64) } else { 0.0 },
            max_drawdown_pct: max_drawdown,
            avg_drawdown_pct: avg_drawdown,
            max_drawdown_duration_days: 0,
            volatility_annualized: avg_volatility * (365.0_f64).sqrt(),
            var_95: max_drawdown * 0.8,
            cvar_95: max_drawdown,
            total_trades: total_trades as u64,
            avg_trade_return_pct: avg_trade_return,
            avg_winning_trade_pct: avg_winning_trade,
            avg_losing_trade_pct: avg_losing_trade,
            largest_win_pct: wf_summary.window_results.iter()
                .map(|w| {
                    let winning_trades = (w.test_metrics.number_of_trades as f64 * w.test_metrics.win_rate).max(1.0);
                    w.test_metrics.gross_profit / winning_trades
                })
                .fold(0.0, f64::max),
            largest_loss_pct: wf_summary.window_results.iter()
                .map(|w| {
                    let losing_trades = (w.test_metrics.number_of_trades as f64 * (1.0 - w.test_metrics.win_rate)).max(1.0);
                    -w.test_metrics.gross_loss.abs() / losing_trades
                })
                .fold(0.0, f64::min),
            avg_trade_duration_hours: avg_trade_duration,
            out_of_sample_sharpe: wf_summary.average_test_sharpe,
            parameter_stability_score: wf_summary.parameter_stability,
            profitable_windows: profitable_windows as u32,
            total_windows: wf_summary.window_results.len() as u32,
            monte_carlo_median_return: median,
            monte_carlo_5th_percentile: p5,
            monte_carlo_95th_percentile: p95,
            monte_carlo_probability_of_profit: profitable_windows as f64 / total_windows,
            backtest_start,
            backtest_end,
            total_days,
        }
    }

    /// Create analysis summary with non-binding recommendation
    fn create_analysis_summary(&self, wf_summary: &WalkForwardSummary) -> AnalysisSummary {
        let total_windows = wf_summary.window_results.len();
        let profitable_windows = wf_summary.window_results.iter()
            .filter(|w| w.test_metrics.net_profit > 0.0 && w.test_metrics.sharpe_ratio > 0.0)
            .count();
        
        let avg_sharpe = wf_summary.average_test_sharpe;
        let param_stability = wf_summary.parameter_stability;
        
        // Non-binding recommendation (user can ignore)
        let recommendation = if avg_sharpe >= 1.5 && param_stability >= 0.8 && profitable_windows >= total_windows * 4 / 5 {
            AnalysisRecommendation::LooksGood
        } else if avg_sharpe >= 1.0 && param_stability >= 0.6 && profitable_windows >= total_windows / 2 {
            AnalysisRecommendation::ProceedWithCaution
        } else if avg_sharpe >= 0.5 && profitable_windows >= total_windows / 3 {
            AnalysisRecommendation::HighRisk
        } else {
            AnalysisRecommendation::Questionable
        };
        
        AnalysisSummary {
            total_windows,
            profitable_windows,
            average_sharpe: avg_sharpe,
            parameter_stability: param_stability,
            recommendation,
        }
    }
}

/// Calculate percentiles from a slice of values
fn percentiles(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let len = sorted.len();
    let p5_idx = (len as f64 * 0.05) as usize;
    let p50_idx = len / 2;
    let p95_idx = (len as f64 * 0.95) as usize;
    
    (
        sorted.get(p5_idx).copied().unwrap_or(0.0),
        sorted.get(p50_idx).copied().unwrap_or(0.0),
        sorted.get(p95_idx.min(len - 1)).copied().unwrap_or(0.0),
    )
}

// ============================================================================
// LEGACY COMPATIBILITY - Deprecated auto-export (redirects to new system)
// ============================================================================

/// Profitability thresholds - DEPRECATED
/// 
/// These thresholds were used for automatic gatekeeping.
/// In the new system, users see all results and decide themselves.
/// Kept for backwards compatibility but no longer used for blocking.
#[deprecated(since = "2.0.0", note = "Use DeploymentPreparer instead - thresholds are now advisory only")]
#[derive(Debug, Clone)]
pub struct ExportThresholds {
    pub min_sharpe_ratio: f64,
    pub min_profit_factor: f64,
    pub min_win_rate: f64,
    pub max_drawdown: f64,
    pub min_number_of_trades: usize,
    pub min_consistency_windows: usize,
}

#[allow(deprecated)]
impl Default for ExportThresholds {
    fn default() -> Self {
        Self {
            min_sharpe_ratio: 1.0,
            min_profit_factor: 1.3,
            min_win_rate: 0.55,
            max_drawdown: 0.15,
            min_number_of_trades: 50,
            min_consistency_windows: 3,
        }
    }
}

/// DEPRECATED - Use DeploymentPreparer instead
#[deprecated(since = "2.0.0", note = "Use DeploymentPreparer::prepare_deployment() instead")]
pub struct AutoExporter {
    #[allow(deprecated)]
    _thresholds: ExportThresholds,
    preparer: DeploymentPreparer,
}

#[allow(deprecated)]
impl AutoExporter {
    pub fn new(thresholds: ExportThresholds) -> Self {
        Self {
            _thresholds: thresholds,
            preparer: DeploymentPreparer::new(Uuid::nil()),
        }
    }

    /// DEPRECATED - Returns true but does NOT auto-deploy
    /// 
    /// Previously this would automatically export strategies that met thresholds.
    /// Now it prepares a deployment request for user review instead.
    #[deprecated(since = "2.0.0", note = "Use DeploymentPreparer::prepare_deployment() instead")]
    pub async fn evaluate_and_export(
        &self,
        wf_summary: &WalkForwardSummary,
        strategy: Arc<dyn StrategyTrait>,
        strategy_name: &str,
    ) -> Result<bool> {
        log_warn!(WALKFORWARD_LOGGER, "⚠️ AutoExporter::evaluate_and_export() is DEPRECATED");
        log_warn!(WALKFORWARD_LOGGER, "   Auto-export has been replaced with user-controlled deployment");
        log_warn!(WALKFORWARD_LOGGER, "   Use DeploymentPreparer::prepare_deployment() instead");
        
        // Prepare deployment request (but don't deploy)
        let preparation = self.preparer.prepare_deployment(
            wf_summary,
            strategy,
            strategy_name,
            "UNKNOWN",
            "UNKNOWN",
            UserRiskConfig::default(),
        )?;
        
        log_info!(WALKFORWARD_LOGGER, "📋 Deployment request created: {}", preparation.request.request_id);
        log_info!(WALKFORWARD_LOGGER, "   User must review and approve before deployment");
        
        Ok(true)
    }
}
