//! Clean and simplified backtesting engine
//! Combines basic and enhanced functionality in a unified interface

use std::time::Instant;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::{BacktestResult, monte_carlo, simulation::SimulationLoop};
use crate::logging_facade::BACKTEST_LOGGER;
use crate::{log_info, log_debug, log_warn, log_error};
use config::{BacktestConfig, AnalysisMode, AnalysisConfig};
use strategy::{StrategyManager, simple_export::ExportConfig};
use orderbook::OrderBook;
use portfoliomanager::PortfolioState;
use riskmanager::RiskManager;
use dataloader::MarketData;
use std::sync::Arc;

/// Progress callback for Monte Carlo and Walk-Forward analysis
/// Parameters: (phase_name, current, total)
pub type ProgressCallback = Arc<dyn Fn(&str, usize, usize) + Send + Sync>;

/// Main backtesting engine with intelligent analysis capabilities
pub struct BacktestEngine {
    config: BacktestConfig,
    analysis_config: AnalysisConfig,
    progress_callback: Option<ProgressCallback>,
    /// Scores GA candidates in the built-in SMA-crossover demo optimizer
    /// (`run_genetic_optimization`). `backtest` ships no built-in formula --
    /// see `genetic::fitness::FitnessFunction`.
    fitness_fn: Arc<dyn genetic::fitness::FitnessFunction>,
}

/// Comprehensive backtest results with optional advanced analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Core backtest metrics
    pub core: BacktestResult,
    /// Monte Carlo simulation results (if performed)
    pub monte_carlo: Option<monte_carlo::MonteCarloResult>,
    /// Walk-forward analysis results (if performed)  
    pub walk_forward: Option<WalkForwardSummary>,
    /// Analysis metadata and recommendations
    pub metadata: AnalysisMetadata,
}

/// Walk-forward analysis summary with full OOS metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WalkForwardSummary {
    /// Number of walk-forward windows completed
    pub periods_tested: usize,
    /// Average out-of-sample performance (return %)
    pub average_performance: f64,
    /// Consistency score (0-1, % of profitable OOS periods)
    pub consistency_score: f64,
    /// Degradation from in-sample to out-of-sample (ratio, higher = more overfit)
    pub out_of_sample_degradation: f64,
    
    // === Full OOS Metrics (primary metrics for production decisions) ===
    /// Out-of-sample total PnL in USD
    #[serde(default)]
    pub oos_total_pnl: f64,
    /// Out-of-sample Sharpe ratio
    #[serde(default)]
    pub oos_sharpe_ratio: f64,
    /// Out-of-sample maximum drawdown (0-1)
    #[serde(default)]
    pub oos_max_drawdown: f64,
    /// Out-of-sample win rate (0-1)
    #[serde(default)]
    pub oos_win_rate: f64,
    /// Out-of-sample profit factor
    #[serde(default)]
    pub oos_profit_factor: f64,
    /// Out-of-sample total trades
    #[serde(default)]
    pub oos_num_trades: usize,
    /// Out-of-sample equity curve
    #[serde(default)]
    pub oos_equity_curve: Vec<f64>,
    /// Buy-and-hold equity curve aligned to the stitched OOS windows.
    /// Same length as `oos_equity_curve`; each window starts from the prior window's
    /// ending equity (offset-chained) so the comparison is apples-to-apples.
    #[serde(default)]
    pub oos_buy_and_hold_equity_curve: Vec<f64>,
    /// OOS Sharpe ratio of the most recent walk-forward window.
    /// Recency gate: if this is negative, the strategy may have degraded recently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_window_oos_sharpe: Option<f64>,
    /// Mann-Kendall τ on the OOS Sharpe time series (-1 to +1).
    /// Negative and significant = declining out-of-sample performance over time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mann_kendall_tau: Option<f64>,
    /// Two-sided p-value for the Mann-Kendall trend test.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mann_kendall_p: Option<f64>,
    /// Information Ratio vs buy-and-hold on OOS data.
    /// Positive = strategy generates active return above passive holding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oos_information_ratio: Option<f64>,
}

/// Analysis execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub analysis_mode: AnalysisMode,
    pub mode_escalated: bool,
    pub total_runtime_seconds: f64,
    pub data_quality_score: f64,
    pub triggers_fired: Vec<String>,
    pub recommendations: Vec<String>,
}

impl BacktestEngine {
    /// Create a new backtesting engine with configuration and a GA fitness
    /// function (used only by the built-in SMA-crossover demo optimizer).
    pub fn new(config: BacktestConfig, fitness_fn: Arc<dyn genetic::fitness::FitnessFunction>) -> Self {
        let analysis_config = config.analysis.clone();

        Self {
            config,
            analysis_config,
            progress_callback: None,
            fitness_fn,
        }
    }
    
    /// Get a reference price from market data (first available price)
    /// Used for converting order_size_pct to asset units (works for any trading pair)
    fn get_reference_price(market_data: &[MarketData]) -> Option<f64> {
        market_data.first().and_then(|data| {
            match data {
                MarketData::Candle(c) => Some(c.close),
                MarketData::Trade(t) => Some(t.price),
                MarketData::PoolSwap(s) => Some(s.price()),
                MarketData::Generic(g) => Some(g.price),
                MarketData::OptionCandle(c) => Some(c.close),
            }
        })
    }
    
    /// Set a progress callback for Monte Carlo and Walk-Forward analysis
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Single source of truth for GA population/generation sizing per analysis
    /// mode. Also drives the trial count fed to the Deflated Sharpe Ratio — the
    /// two values were previously duplicated hardcodes that could drift apart.
    fn ga_population_generations(mode: AnalysisMode) -> (usize, usize) {
        match mode {
            AnalysisMode::Development => (5, 3),
            AnalysisMode::Validation => (20, 10),
            AnalysisMode::Production => (50, 25),
        }
    }

    /// Run backtest with automatic analysis mode selection
    pub async fn run_backtest(
        &self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
    ) -> Result<BacktestResults> {
        self.run_with_export_option(
            strategy_manager,
            orderbook,
            portfolio,
            risk_manager,
            market_data,
            false,
        ).await
    }

    /// Run backtest with export consideration (forces production mode)
    pub async fn run_with_export_option(
        &self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
        export_requested: bool,
    ) -> Result<BacktestResults> {
        let start_time = Instant::now();
        
        log_info!(BACKTEST_LOGGER, "run_with_export_option ENTER | data_points={} export={}", 
            market_data.len(), export_requested);
        
        // Determine analysis mode
        let mut current_mode = self.determine_analysis_mode(export_requested);
        let original_mode = current_mode;
        
        log_info!(BACKTEST_LOGGER, "Starting Backtest Analysis - Mode: {:?}", current_mode);
        
        // PRE-PROCESSING: Analyze and handle data gaps BEFORE optimization/simulation
        let data_quality_report = self.generate_data_quality_report(market_data);
        
        let severe_gap_count = data_quality_report.gaps_detected.iter()
            .filter(|g| matches!(g.severity, crate::types::GapSeverity::Severe | crate::types::GapSeverity::Critical))
            .count();
        log_info!(BACKTEST_LOGGER, "Data Quality: {:.1}/10, {} points, {} gaps ({} severe)", 
            data_quality_report.overall_score, 
            data_quality_report.total_data_points,
            data_quality_report.gaps_detected.len(),
            severe_gap_count
        );
        
        // Apply gap filling strategy if configured
        // Use Cow to avoid cloning when no gap filling is needed (common case).
        let processed_data: std::borrow::Cow<'_, [MarketData]> = if self.config.data.gap_detection.enabled {
            std::borrow::Cow::Owned(self.apply_gap_filling(market_data)?)
        } else {
            std::borrow::Cow::Borrowed(market_data)
        };
        
        if processed_data.len() != market_data.len() {
            log_debug!(BACKTEST_LOGGER, "Gap filling applied: {} → {} data points", 
                market_data.len(), processed_data.len());
        }
        
        // Use processed_data for all subsequent operations
        let market_data = &*processed_data;
        
        // MANDATORY: Run genetic optimization first to find optimal parameters (unless skipped)
        let mut ga_trial_sharpes: Vec<f64> = Vec::new();
        let mut full_data_params: Option<std::collections::HashMap<String, config::ParameterValue>> = None;
        if !self.analysis_config.skip_genetic_optimization {
            log_info!(BACKTEST_LOGGER, "Running genetic optimization");
            let (optimized_params, sharpe_vals) = self.run_genetic_optimization(
                strategy_manager,
                orderbook,
                portfolio,
                risk_manager,
                market_data
            ).await?;
            ga_trial_sharpes = sharpe_vals;

            // Apply optimized parameters to all strategies. Keep a copy so
            // walk-forward (which re-optimizes per window) can restore them.
            strategy_manager.update_all_parameters(optimized_params.clone()).await?;
            full_data_params = Some(optimized_params);
            log_info!(BACKTEST_LOGGER, "Genetic optimization complete");
        } else {
            log_info!(BACKTEST_LOGGER, "Skipping GA (pre-optimized) - proceeding to simulation");
        }
        
        // Run core backtest simulation with optimized parameters
        log_info!(BACKTEST_LOGGER, "PHASE_START | backtest_simulation | data_points={}", market_data.len());
        let sim_start = Instant::now();
        let mut core_result = self.run_core_simulation(
            strategy_manager, 
            orderbook, 
            portfolio, 
            risk_manager, 
            market_data
        ).await?;
        log_info!(BACKTEST_LOGGER, "PHASE_COMPLETE | backtest_simulation | duration={:.1}s trades={}", 
            sim_start.elapsed().as_secs_f64(), core_result.num_trades);
        
        // Attach data quality report to results
        core_result.data_quality = Some(data_quality_report.clone());

        // Deflated Sharpe Ratio — corrects for multiple comparisons introduced
        // by GA parameter search. Stored on the core result so the frontend can
        // gate the Promising verdict independently of the OOS Sharpe.
        if !self.analysis_config.skip_genetic_optimization {
            let ga_trial_count: u32 = {
                let (pop, gens) = Self::ga_population_generations(self.analysis_config.mode);
                (pop * gens) as u32
            };
            let returns = if !core_result.trade_pct_returns.is_empty() {
                core_result.trade_pct_returns.as_slice()
            } else {
                core_result.trade_returns.as_slice()
            };
            core_result.deflated_sharpe =
                metrics::performance::deflated_sharpe_ratio(ga_trial_count, returns);
            core_result.ga_trial_count = Some(ga_trial_count);

            // IS Sharpe distribution across viable GA candidates (pass hard gates)
            if !ga_trial_sharpes.is_empty() {
                let n = ga_trial_sharpes.len() as f64;
                let mean = ga_trial_sharpes.iter().sum::<f64>() / n;
                let variance = ga_trial_sharpes.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
                let pct_above = ga_trial_sharpes.iter().filter(|&&s| s >= 1.0).count() as f64 / n;
                core_result.ga_is_sharpe_mean = Some(mean);
                core_result.ga_is_sharpe_std = Some(variance.sqrt());
                core_result.ga_is_sharpe_pct_above_threshold = Some(pct_above);
            }
        }

        // Auto-escalation logic based on performance
        let mut escalation_triggers = Vec::with_capacity(2);  // Max 2 escalations
        let mut mode_escalated = false;
        
        if current_mode == AnalysisMode::Development {
            if self.should_escalate_to_validation(&core_result) {
                current_mode = AnalysisMode::Validation;
                mode_escalated = true;
                escalation_triggers.push("Poor performance detected - escalated to Validation".to_string());
                log_info!(BACKTEST_LOGGER, "Auto-escalating to Validation mode");
            }
        }
        
        if current_mode == AnalysisMode::Validation {
            if self.should_escalate_to_production(&core_result) {
                current_mode = AnalysisMode::Production;
                mode_escalated = true;
                escalation_triggers.push("Excellent performance detected - escalated to Production".to_string());
                log_info!(BACKTEST_LOGGER, "Auto-escalating to Production mode");
            }
        }
        
        // Run additional analysis based on final mode
        let monte_carlo_result = match current_mode {
            AnalysisMode::Development => None,
            AnalysisMode::Validation | AnalysisMode::Production => {
                let mc_runs = self.analysis_config.monte_carlo_runs;
                log_info!(BACKTEST_LOGGER, "PHASE_START | monte_carlo | runs={}", mc_runs);
                let mc_start = Instant::now();
                let result = self.run_monte_carlo_analysis(&core_result).await?;
                log_info!(BACKTEST_LOGGER, "PHASE_COMPLETE | monte_carlo | duration={:.1}s", mc_start.elapsed().as_secs_f64());
                Some(result)
            }
        };
        
        let walk_forward_result = match current_mode {
            AnalysisMode::Production if self.analysis_config.auto_walk_forward => {
                log_info!(BACKTEST_LOGGER, "PHASE_START | walk_forward | data_points={}", market_data.len());
                let wf_start = Instant::now();
                // Per-window re-optimization on each window's in-sample slice;
                // the full-data parameters are restored afterwards.
                let result = self.run_walk_forward_analysis(
                    strategy_manager, orderbook, portfolio, risk_manager, market_data,
                    full_data_params.clone(),
                ).await?;
                log_info!(BACKTEST_LOGGER, "PHASE_COMPLETE | walk_forward | duration={:.1}s windows={}", 
                    wf_start.elapsed().as_secs_f64(), result.periods_tested);
                Some(result)
            },
            _ => None,
        };
        
        let total_runtime = start_time.elapsed().as_secs_f64();
        
        // Tag the core result with the analysis mode so the frontend can suppress
        // the Promising verdict for development-mode runs without production validation.
        core_result.analysis_mode = Some(match current_mode {
            AnalysisMode::Development => "development".to_string(),
            AnalysisMode::Validation  => "validation".to_string(),
            AnalysisMode::Production  => "production".to_string(),
        });

        // Record which analysis stages ran so the frontend can display them.
        let mut stages: Vec<String> = vec!["in_sample".to_string()];
        if monte_carlo_result.is_some() { stages.push("monte_carlo".to_string()); }
        if walk_forward_result.is_some() { stages.push("walk_forward".to_string()); }
        core_result.analysis_stages_run = stages;

        // Generate analysis metadata
        let metadata = AnalysisMetadata {
            analysis_mode: current_mode,
            mode_escalated: mode_escalated && original_mode != current_mode,
            total_runtime_seconds: total_runtime,
            data_quality_score: self.calculate_data_quality_score(market_data),
            triggers_fired: escalation_triggers,
            recommendations: self.generate_recommendations(&core_result, &monte_carlo_result),
        };

        let results = BacktestResults {
            core: core_result,
            monte_carlo: monte_carlo_result,
            walk_forward: walk_forward_result,
            metadata,
        };
        
        log_info!(BACKTEST_LOGGER, "Backtest completed in {:.1}s", total_runtime);
        
        Ok(results)
    }

    /// Determine initial analysis mode based on configuration and export requirements
    fn determine_analysis_mode(&self, export_requested: bool) -> AnalysisMode {
        if export_requested {
            AnalysisMode::Production
        } else {
            self.analysis_config.mode
        }
    }

    /// Check if performance warrants escalation to validation mode
    fn should_escalate_to_validation(&self, result: &BacktestResult) -> bool {
        if let Some(sharpe) = result.sharpe_ratio {
            sharpe < self.analysis_config.validation_sharpe_threshold
        } else {
            false
        }
    }
    
    /// Check if performance warrants escalation to production mode
    fn should_escalate_to_production(&self, result: &BacktestResult) -> bool {
        if let Some(sharpe) = result.sharpe_ratio {
            sharpe > self.analysis_config.validation_sharpe_threshold * 1.5
        } else {
            false
        }
    }
    
    /// Run the core backtest simulation (internal use for Walk-Forward, etc.)
    async fn run_core_simulation(
        &self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
    ) -> Result<BacktestResult> {
        // Route to LP simulation when LP config is present
        if let Some(ref lp_config) = self.config.lp {
            return self.run_lp_simulation(lp_config.clone(), market_data).await;
        }

        // Run actual simulation using SimulationLoop
        let mut sim_loop = SimulationLoop::new();
        let result = sim_loop.run(
            strategy_manager,
            orderbook,
            portfolio,
            risk_manager,
            market_data,
            &self.config,
        ).await;

        Ok(result)
    }
    
    /// Run the core backtest simulation without analysis (Monte Carlo, Walk-Forward)
    /// This is used by genetic optimization fitness functions for fast evaluation
    pub async fn run_core_simulation_direct(
        &self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
    ) -> Result<BacktestResult> {
        self.run_core_simulation(strategy_manager, orderbook, portfolio, risk_manager, market_data).await
    }

    /// Run LP simulation for AMM liquidity provision backtesting.
    /// Extracts PoolSwapEvents from market data and runs the LP-specific loop.
    async fn run_lp_simulation(
        &self,
        lp_config: config::LpConfig,
        market_data: &[MarketData],
    ) -> Result<BacktestResult> {
        use crate::lp_simulation::LpSimulationLoop;

        let pool_swaps: Vec<&dataloader::PoolSwapEvent> = market_data.iter()
            .filter_map(|d| match d {
                MarketData::PoolSwap(s) => Some(s),
                _ => None,
            })
            .collect();

        if pool_swaps.is_empty() {
            log_warn!(BACKTEST_LOGGER, "LP simulation: no PoolSwap events in market data");
            return Ok(BacktestResult::default());
        }

        log_info!(BACKTEST_LOGGER, "LP simulation: {} pool swap events, pool={}",
            pool_swaps.len(), lp_config.pool_id);

        let mut lp_loop = LpSimulationLoop::new(lp_config);

        // Open initial position on first swap
        lp_loop.open_initial_position(pool_swaps[0]);

        // Collect owned swaps for the simulation (loop expects slice of owned events)
        let owned_swaps: Vec<dataloader::PoolSwapEvent> = pool_swaps.into_iter().cloned().collect();
        let lp_result = lp_loop.run(&owned_swaps);

        log_info!(BACKTEST_LOGGER, "LP simulation complete: fees=${:.2} IL=${:.2} net=${:.2} APR={:.1}%",
            lp_result.total_fees_earned_usd,
            lp_result.total_impermanent_loss_usd,
            lp_result.net_pnl_usd,
            lp_result.annualized_apr);

        // Return a BacktestResult with LP metrics attached
        let mut result = BacktestResult::default();
        result.total_pnl = lp_result.net_pnl_usd;
        result.equity_curve = lp_result.equity_curve.iter().map(|(_, v)| *v).collect();
        result.max_drawdown = lp_result.max_drawdown_pct / 100.0;
        result.lp_metrics = Some(lp_result);
        Ok(result)
    }
    
    /// Run genetic optimization to find optimal strategy parameters.
    ///
    /// Uses a candle-based position simulator with the directional momentum schema.
    /// Evaluates candidates by generating signals from indicator crossovers and
    /// simulating trades with stop-loss/take-profit.
    /// Returns `(best_params, viable_trial_sharpes)` where `viable_trial_sharpes`
    /// is the IS Sharpe for each GA candidate that passed the hard fitness gates
    /// (≥30 trades, drawdown ≤40%). Used to populate `ga_is_sharpe_*` stats.
    async fn run_genetic_optimization(
        &self,
        _strategy_manager: &mut StrategyManager,
        _orderbook: &mut OrderBook,
        _portfolio: &mut PortfolioState,
        _risk_manager: &mut RiskManager,
        market_data: &[MarketData],
    ) -> Result<(std::collections::HashMap<String, config::ParameterValue>, Vec<f64>)> {
        use genetic::{GeneticOptimizer, dynamic_chromosome::{DynamicChromosome, set_dynamic_schema}};
        use genetic::presets;
        use config::GeneticConfig;
        use std::sync::{Arc, Mutex};
        use std::pin::Pin;
        use std::future::Future;
        use crate::candle_sim::{self, CandleSimulator, CandleArrays, EarlyTermConfig, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD};

        // --- 1. Load directional momentum schema ----------
        let schema = presets::directional_momentum_schema();
        set_dynamic_schema(schema.clone());

        let (population_size, generations) = Self::ga_population_generations(self.analysis_config.mode);

        let config = GeneticConfig {
            population_size,
            generations,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            optimization_mode: config::OptimizationMode::SingleObjective,
            use_monte_carlo_fitness: false,
            mc_sample_size: 50,
            top_percentile_threshold: 0.10,
            enable_fitness_sharing: false,
            sharing_radius: 0.15,
            enable_crowding_penalty: false,
            crowding_weight: 0.3,
            min_behavioral_distance: 0.3,
            force_sequential: true,
            convergence_threshold: None,
            random_seed: None,
            check_seed_stability: false,
            random_control: false,
            complexity_penalty_weight: 0.0,
            bayesian_optimization: false,
            embedded_oos_folds: 0,
            embedded_oos_gap_penalty: config::GeneticConfig::default().embedded_oos_gap_penalty,
        };

        // --- 2. Pre-compute candle arrays (shared across all candidates) ----------
        let candles = Arc::new(CandleArrays::from_market_data(market_data));
        let initial_capital = self.config.trading.initial_capital;
        let exchange_fee_config = self.config.trading.exchange_fees.clone()
            .unwrap_or_else(config::ExchangeFeeConfig::default);
        let realism_config = candle_sim::build_realism(&self.config.trading, &self.config.trading.exchange_fees);
        let leverage = self.config.trading.leverage;
        let maintenance_margin_ratio = self.config.trading.maintenance_margin_ratio;
        let liquidation_penalty_bps = self.config.trading.liquidation_penalty_bps;
        let vol_target_cfg = self.config.trading.vol_target_annual.map(|target_annual_vol| {
            portfoliomanager::margin::VolTargetConfig {
                target_annual_vol,
                lookback_bars: self.config.trading.vol_target_lookback_bars,
                bars_per_year: self.config.trading.vol_target_bars_per_year,
                min_scale: self.config.trading.vol_target_min_scale,
                max_scale: self.config.trading.vol_target_max_scale,
            }
        });

        // Collects IS Sharpe for each candidate that passes the hard fitness gates.
        // Cloned into the closure; original kept here so we can read after the run.
        let trial_sharpes: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let trial_sharpes_for_fn = Arc::clone(&trial_sharpes);
        let scorer = Arc::clone(&self.fitness_fn);

        let fitness_fn = Arc::new(move |chromo: &DynamicChromosome| {
            let candles = Arc::clone(&candles);
            let chromo = chromo.clone();
            let exchange_fee_config = exchange_fee_config.clone();
            let fee_rate = exchange_fee_config.taker_fee;
            let initial_capital = initial_capital;
            let realism = realism_config.clone();
            let fixed_leverage = leverage;
            let maintenance_margin_ratio = maintenance_margin_ratio;
            let liquidation_penalty_bps = liquidation_penalty_bps;
            let vol_target_cfg = vol_target_cfg.clone();
            let trial_sharpes = Arc::clone(&trial_sharpes_for_fn);
            let scorer = Arc::clone(&scorer);

            Box::pin(async move {
                let param_map = chromo.to_param_map();

                let fast_period = param_map.get("fast_period")
                    .and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let slow_period = param_map.get("slow_period")
                    .and_then(|v| v.as_u64()).unwrap_or(30) as usize;
                let stop_loss_pct = param_map.get("stop_loss_pct")
                    .and_then(|v| v.as_f64()).unwrap_or(0.03);
                let take_profit_pct = param_map.get("take_profit_pct")
                    .and_then(|v| v.as_f64()).unwrap_or(0.06);
                let max_position_pct = param_map.get("max_position_pct")
                    .and_then(|v| v.as_f64()).unwrap_or(0.5);
                // Genome-evolved leverage, clamped to this exchange's real
                // cap (never allowed to exceed it, regardless of what the
                // GA proposes). Falls back to the job's fixed leverage for
                // schemas that don't declare a "leverage" gene (e.g. custom
                // Python strategies, out of scope for genome evolution).
                let requested_leverage = param_map.get("leverage")
                    .and_then(|v| v.as_f64()).unwrap_or(fixed_leverage);
                let (leverage, _was_capped) = exchange_fee_config.resolve_leverage(requested_leverage);

                // Generate signals: SMA crossover
                let n = candles.len();
                let closes = &candles.closes;
                let mut signals = vec![SIGNAL_HOLD; n];

                if n > slow_period && fast_period < slow_period {
                    let mut fast_sum: f64 = closes[..fast_period].iter().sum();
                    let mut slow_sum: f64 = closes[..slow_period].iter().sum();

                    for i in slow_period..n {
                        // Update running sums
                        fast_sum += closes[i] - closes[i - fast_period];
                        slow_sum += closes[i] - closes[i - slow_period];

                        let fast_ma = fast_sum / fast_period as f64;
                        let slow_ma = slow_sum / slow_period as f64;

                        // Previous bar's MAs
                        let prev_fast_sum = fast_sum - closes[i] + closes[i - fast_period];
                        let prev_slow_sum = slow_sum - closes[i] + closes[i - slow_period];
                        let prev_fast_ma = prev_fast_sum / fast_period as f64;
                        let prev_slow_ma = prev_slow_sum / slow_period as f64;

                        if prev_fast_ma <= prev_slow_ma && fast_ma > slow_ma {
                            signals[i] = SIGNAL_BUY;
                        } else if prev_fast_ma >= prev_slow_ma && fast_ma < slow_ma {
                            signals[i] = SIGNAL_SELL;
                        }
                    }
                }

                // Run candle simulator
                let mut simulator = CandleSimulator::new(initial_capital, fee_rate, config::SlippageModel::default())
                    .with_realism(realism)
                    .with_leverage(leverage, maintenance_margin_ratio, liquidation_penalty_bps);
                if let Some(cfg) = vol_target_cfg {
                    simulator = simulator.with_vol_target(cfg);
                }
                let result = simulator.run(
                    &signals, &candles,
                    Some(stop_loss_pct), Some(take_profit_pct),
                    max_position_pct,
                    Some(&EarlyTermConfig::default()),
                );

                let total_pnl = result.total_pnl;
                let total_fees = result.total_commission;
                let dd_frac = (result.max_drawdown / initial_capital).abs();

                // MinTRL (Bailey & López de Prado, 2014) from this candidate's
                // own dollar-valued trade_returns -- Sharpe-style ratios are
                // scale-invariant to a constant multiplicative rescale, so no
                // conversion to fractional returns is needed here. Caveat:
                // this assumes roughly-constant position sizing across trades.
                let min_trl = if result.trade_returns.len() >= 3 {
                    crate::statistical_significance::compute_sharpe_significance(
                        &result.trade_returns, 0.0, 252.0,
                    ).min_track_record_length
                } else {
                    None
                };
                let data_span_days = match (candles.timestamps_ms.first(), candles.timestamps_ms.last()) {
                    (Some(first), Some(last)) => ((*last - *first) as f64 / 86_400_000.0).max(1.0),
                    _ => 1.0,
                };

                let inputs = genetic::fitness::FitnessInputs {
                    equity_curve: result.equity_curve.clone(),
                    num_trades: result.num_trades,
                    total_liquidations: result.total_liquidations,
                    net_pnl: total_pnl,
                    initial_capital,
                    sharpe: result.sharpe_ratio,
                    sortino: result.sharpe_ratio, // no per-trade downside tracking in CandleSimResult yet
                    profit_factor: result.profit_factor,
                    win_rate: result.win_rate,
                    max_drawdown_frac: dd_frac,
                    max_drawdown_abs: result.max_drawdown,
                    fill_rate: 0.0, // not tracked by the candle simulator
                    total_fees,
                    param_count: param_map.len(),
                    min_trl,
                    data_span_days,
                    ..Default::default()
                };
                let inputs = genetic::fitness::FitnessInputs {
                    max_drawdown_hard_cap: 0.40,
                    leverage_evolved: param_map.contains_key("leverage"),
                    ..inputs
                };
                let fitness_result = scorer.compute(&inputs);

                // Record IS Sharpe for this viable candidate (passed all hard gates)
                if !fitness_result.hard_gated {
                    if let Ok(mut guard) = trial_sharpes.lock() {
                        guard.push(result.sharpe_ratio);
                    }
                }

                fitness_result
            }) as Pin<Box<dyn Future<Output = genetic::FitnessResult> + Send + 'static>>
        });

        log_debug!(BACKTEST_LOGGER, "Genetic: pop={}, gen={}", population_size, generations);

        // --- 3. Run the optimizer ----------
        let optimizer = GeneticOptimizer::<DynamicChromosome> {
            config,
            fitness_fn,
            progress_callback: None,
            job_id: None,
            strategy_registry: None,
            tenant_id: None,
        };

        let (best_chromo, best_fitness) = optimizer.run().await;

        // --- 4. Convert best chromosome → ParameterValue map ----------------
        let best_param_map = best_chromo.to_param_map();

        log_info!(BACKTEST_LOGGER, "Best fitness: {:.4}", best_fitness);

        let mut params = std::collections::HashMap::new();
        for (key, val) in &best_param_map {
            let pv = if let Some(f) = val.as_f64() {
                config::ParameterValue::Float(f)
            } else if let Some(i) = val.as_i64() {
                config::ParameterValue::Float(i as f64)
            } else if let Some(b) = val.as_bool() {
                config::ParameterValue::Bool(b)
            } else if let Some(s) = val.as_str() {
                config::ParameterValue::String(s.to_string())
            } else {
                continue;
            };
            params.insert(key.clone(), pv);
        }

        let sharpe_vals = trial_sharpes.lock().map(|g| g.clone()).unwrap_or_default();
        Ok((params, sharpe_vals))
    }

    /// Run Monte Carlo simulation analysis
    async fn run_monte_carlo_analysis(&self, base_result: &BacktestResult) -> Result<monte_carlo::MonteCarloResult> {
        use crate::logging_facade::BACKTEST_LOGGER;
        use crate::log_info;
        
        let num_runs = self.analysis_config.monte_carlo_runs;
        let num_trades = base_result.num_trades;
        // One clone shared via Arc between the two spawn_blocking closures —
        // BacktestResult carries large equity curves, so avoid cloning twice.
        let base_shared = Arc::new(base_result.clone());
        let base_result = Arc::clone(&base_shared);
        let base_result_bb = base_shared;

        // Log before spawn_blocking (we have Tokio runtime here)
        log_info!(BACKTEST_LOGGER, "Monte Carlo starting | runs={} trades={} shuffle={} bootstrap={}",
                  num_runs, num_trades, num_runs / 2, num_runs - num_runs / 2);

        let start_time = std::time::Instant::now();

        // Run Monte Carlo in blocking thread pool (CPU-intensive, now parallel with Rayon)
        // Note: Logging inside spawn_blocking won't work since there's no Tokio runtime
        let mut mc_result = tokio::task::spawn_blocking(move || {
            monte_carlo::run_monte_carlo_simulation(
                &base_result,
                num_runs
            )
        }).await??;

        let elapsed = start_time.elapsed();

        // Log after spawn_blocking completes (we have Tokio runtime here)
        log_info!(BACKTEST_LOGGER, "Monte Carlo complete | runs={} elapsed={:.2}s mean_sharpe={:.3} var_95=${:.2} cvar_95=${:.2}",
                  num_runs, elapsed.as_secs_f64(), mc_result.mean_metrics.sharpe_ratio,
                  mc_result.var_95, mc_result.cvar_95);

        // Block bootstrap: supplemental run preserving trade-return autocorrelation.
        // Uses half as many runs (supplemental); result drives the bb_p5_total_pnl verdict gate.
        mc_result.bb_p5_total_pnl = tokio::task::spawn_blocking(move || {
            let n_trades = base_result_bb.trade_returns.len();
            monte_carlo::run_block_bootstrap_simulation(
                &base_result_bb,
                (num_runs / 2).max(100),
                monte_carlo::BlockBootstrapConfig {
                    block_size: monte_carlo::BlockBootstrapConfig::optimal_block_size(n_trades),
                    ..monte_carlo::BlockBootstrapConfig::default()
                },
            )
        })
        .await
        .ok()
        .and_then(|r| r.ok())
        .and_then(|r| r.percentile_metrics.get(&5).map(|m| m.net_profit));

        Ok(mc_result)
    }
    
    /// Run walk-forward analysis.
    ///
    /// Each window re-optimizes on its own in-sample (optimization) slice and
    /// is then evaluated on its out-of-sample (test) slice — parameters found
    /// on the FULL dataset are never tested against slices of that same data
    /// (that was pure data leakage). Portfolio, orderbook, and risk state are
    /// reset before every window so no positions/orders leak between windows
    /// or in from the main simulation. `full_data_params` (the top-level GA
    /// result) is restored to the strategy manager afterwards.
    async fn run_walk_forward_analysis(
        &self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
        full_data_params: Option<std::collections::HashMap<String, config::ParameterValue>>,
    ) -> Result<WalkForwardSummary> {
        use walkforward::{WalkForwardConfig, WalkForwardAnalyzer};
        
        // Calculate data duration
        let data_duration_days = if market_data.len() >= 2 {
            let first_ts = match &market_data[0] {
                MarketData::Candle(c) => c.timestamp,
                MarketData::Trade(t) => t.timestamp,
                MarketData::PoolSwap(s) => s.timestamp,
                MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
                MarketData::OptionCandle(c) => c.timestamp,
            };

            let last_ts = match &market_data[market_data.len() - 1] {
                MarketData::Candle(c) => c.timestamp,
                MarketData::Trade(t) => t.timestamp,
                MarketData::PoolSwap(s) => s.timestamp,
                MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
                MarketData::OptionCandle(c) => c.timestamp,
            };
            
            (last_ts - first_ts).num_days()
        } else {
            0
        };
        
        // Dynamically adjust window sizes based on available data
        // Use 2/3 for training, 1/3 for testing
        let optimization_days = (data_duration_days as f64 * 0.67) as i64;
        let test_days = data_duration_days - optimization_days;
        
        // Need at least 3 days total for meaningful walk-forward (2 train, 1 test)
        if data_duration_days < 3 {
            return Ok(WalkForwardSummary {
                periods_tested: 0,
                average_performance: 0.0,
                consistency_score: 0.0,
                out_of_sample_degradation: 0.0,
                ..Default::default()
            });
        }
        
        let start_date = match &market_data[0] {
            MarketData::Candle(c) => c.timestamp,
            MarketData::Trade(t) => t.timestamp,
            MarketData::PoolSwap(s) => s.timestamp,
            MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
            MarketData::OptionCandle(c) => c.timestamp,
        };

        let end_date = match &market_data[market_data.len() - 1] {
            MarketData::Candle(c) => c.timestamp,
            MarketData::Trade(t) => t.timestamp,
            MarketData::PoolSwap(s) => s.timestamp,
            MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
            MarketData::OptionCandle(c) => c.timestamp,
        };
        
        // Use larger step size to create fewer windows (3 windows max for efficiency)
        // Each window tests parameter robustness in different market conditions
        let step_size = (test_days).max(7); // Non-overlapping windows, minimum 7 days

        // Derive embargo from candle interval × max strategy lookback (200 bars for directional_momentum).
        // Prevents the OOS window from warming up indicators on IS data.
        // Capped at 14 days so short WFO windows remain usable.
        // Uses median inter-bar delta (not just first two bars) to avoid atypical opening gaps.
        // delta_secs is hoisted here so it can also drive per-bar risk-free rate in the IR computation below.
        let delta_secs: i64 = if market_data.len() >= 2 {
            let extract_ts = |md: &MarketData| match md {
                MarketData::Candle(c) => c.timestamp,
                MarketData::Trade(t) => t.timestamp,
                MarketData::PoolSwap(s) => s.timestamp,
                MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms)
                    .unwrap_or(chrono::DateTime::UNIX_EPOCH),
                MarketData::OptionCandle(c) => c.timestamp,
            };
            let mut deltas: Vec<i64> = market_data.windows(2)
                .map(|w| (extract_ts(&w[1]) - extract_ts(&w[0])).num_seconds())
                .filter(|&d| d > 0)
                .collect();
            deltas.sort_unstable();
            if deltas.is_empty() { 0 } else { deltas[deltas.len() / 2] }
        } else {
            0
        };
        let embargo_days: i64 = if delta_secs > 0 {
            let bars_per_day = (86_400_i64 / delta_secs).max(1);
            let raw = (200_i64 + bars_per_day - 1) / bars_per_day; // ceil(200 / bars_per_day)
            raw.max(1).min(14)
        } else {
            1
        };

        let wf_config = WalkForwardConfig {
            start_date,
            end_date,
            optimization_window_size: std::time::Duration::from_secs((optimization_days * 24 * 3600) as u64),
            test_window_size: std::time::Duration::from_secs((test_days * 24 * 3600) as u64),
            max_parameters: 5,
            optimization_period_days: optimization_days,
            test_period_days: test_days,
            step_size_days: step_size, // Non-overlapping for efficiency
            min_trades_per_period: 5,
            // Matches the directional-momentum schema actually optimized by
            // run_genetic_optimization (previously listed market-making
            // parameters the schema doesn't contain).
            parameters_to_optimize: vec![
                "fast_period".to_string(),
                "slow_period".to_string(),
                "stop_loss_pct".to_string(),
                "take_profit_pct".to_string(),
                "max_position_pct".to_string(),
            ],
            optimization_metric: "sharpe_ratio".to_string(),
            embargo_days,
            anchored: false,
        };
        
        log_info!(BACKTEST_LOGGER, "Walk-forward config: {} days data, opt={}d test={}d step={}d", 
            data_duration_days, optimization_days, test_days, step_size);
        
        let analyzer = WalkForwardAnalyzer::new(wf_config.clone(), self.config.clone());
        let windows = analyzer.generate_windows();
        
        if windows.is_empty() {
            return Ok(WalkForwardSummary {
                periods_tested: 0,
                average_performance: 0.0,
                consistency_score: 0.0,
                out_of_sample_degradation: 0.0,
                ..Default::default()
            });
        }
        
        // Use full data per window - no sampling
        // Market making requires contiguous data to properly simulate fills
        let total_windows = windows.len();
        log_info!(BACKTEST_LOGGER, "Walk-forward: {} windows, FULL DATA per window (no sampling)", total_windows);
        
        // Helper to extract timestamp from a MarketData event
        let md_ts = |md: &MarketData| -> chrono::DateTime<chrono::Utc> {
            match md {
                MarketData::Candle(c) => c.timestamp,
                MarketData::Trade(t) => t.timestamp,
                MarketData::PoolSwap(s) => s.timestamp,
                MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
                MarketData::OptionCandle(c) => c.timestamp,
            }
        };
        
        // Use binary search (partition_point) for O(log n) window slicing.
        // Data is sorted by timestamp; each window carries BOTH its in-sample
        // (optimization) slice and its out-of-sample (test) slice.
        let window_datasets: Vec<(usize, &[MarketData], &[MarketData])> = windows.iter()
            .enumerate()
            .filter_map(|(window_idx, window)| {
                let is_start = market_data.partition_point(|md| md_ts(md) < window.optimization_window.start_date);
                let is_end = market_data.partition_point(|md| md_ts(md) < window.optimization_window.end_date);
                let test_start = market_data.partition_point(|md| md_ts(md) < window.test_window.start_date);
                let test_end = market_data.partition_point(|md| md_ts(md) < window.test_window.end_date);
                let is_slice = &market_data[is_start..is_end];
                let test_slice = &market_data[test_start..test_end];
                if test_slice.is_empty() {
                    return None;
                }
                Some((window_idx, is_slice, test_slice))
            })
            .collect();

        log_info!(BACKTEST_LOGGER, "Walk-forward: prepared {} valid windows", window_datasets.len());

        // Log window sizes for visibility
        for (idx, is_data, test_data) in &window_datasets {
            log_info!(BACKTEST_LOGGER, "Walk-forward window {}: {} IS events, {} OOS events",
                idx + 1, is_data.len(), test_data.len());
        }

        // Process windows sequentially but with pre-filtered data
        // (Full async parallelism would require cloning strategy state per window)
        let mut out_of_sample_sharpes = Vec::with_capacity(window_datasets.len());
        let mut oos_equity_curves: Vec<Vec<f64>> = Vec::with_capacity(window_datasets.len());
        let mut oos_bh_curves: Vec<Vec<f64>> = Vec::with_capacity(window_datasets.len());
        // Aggregates for the full-OOS summary fields
        let mut oos_total_pnl_sum = 0.0_f64;
        let mut oos_trades_sum = 0_usize;
        let mut oos_gross_profit_sum = 0.0_f64;
        let mut oos_gross_loss_sum = 0.0_f64;
        let mut oos_win_rate_weighted = 0.0_f64;
        let mut oos_win_rate_weight = 0.0_f64;

        for (window_idx, is_data, test_data) in window_datasets {
            let window_start = Instant::now();

            // Report progress
            if let Some(ref callback) = self.progress_callback {
                callback("walk_forward", window_idx + 1, total_windows);
            }

            log_info!(BACKTEST_LOGGER, "Walk-forward window {}/{}: {} IS / {} OOS events - STARTING",
                window_idx + 1, total_windows, is_data.len(), test_data.len());

            // Fresh state per window: the previous window (or the main
            // simulation) must not leak positions, resting orders, or equity
            // history into this window's result.
            *portfolio = PortfolioState::with_balance(self.config.trading.initial_capital);
            orderbook.bids.clear();
            orderbook.asks.clear();
            orderbook.volume_history.clear();
            orderbook.volume_30d = 0.0;
            risk_manager.initialize(self.config.trading.initial_capital);

            // Re-optimize on THIS window's in-sample slice only. Testing
            // parameters that were optimized on the full dataset against a
            // slice of that same dataset is not out-of-sample.
            if !self.analysis_config.skip_genetic_optimization && !is_data.is_empty() {
                let (window_params, _) = self.run_genetic_optimization(
                    strategy_manager, orderbook, portfolio, risk_manager, is_data,
                ).await?;
                strategy_manager.update_all_parameters(window_params).await?;
            }

            // Evaluate the window-optimized parameters on the held-out test slice
            let test_result = self.run_core_simulation(
                strategy_manager,
                orderbook,
                portfolio,
                risk_manager,
                test_data,
            ).await?;

            // Accumulate full-OOS aggregates
            oos_total_pnl_sum += test_result.total_pnl;
            oos_trades_sum += test_result.num_trades;
            oos_gross_profit_sum += test_result.gross_profit.unwrap_or(0.0);
            oos_gross_loss_sum += test_result.gross_loss.unwrap_or(0.0);
            if let Some(wr) = test_result.win_rate {
                let w = test_result.num_trades.max(1) as f64;
                oos_win_rate_weighted += wr * w;
                oos_win_rate_weight += w;
            }
            
            log_info!(BACKTEST_LOGGER, "Walk-forward window {}/{} complete: {:.1}s, sharpe={:.3}, trades={}", 
                window_idx + 1, total_windows, window_start.elapsed().as_secs_f64(),
                test_result.sharpe_ratio.unwrap_or(0.0), test_result.num_trades);
            
            if let Some(test_sharpe) = test_result.sharpe_ratio {
                out_of_sample_sharpes.push(test_sharpe);
            }
            if !test_result.equity_curve.is_empty() {
                // Build a buy-and-hold curve over this window using its tick price series,
                // normalised to the same starting capital as the strategy equity curve.
                let bh_curve: Vec<f64> = if test_result.price_series.len() >= 2 {
                    let p0 = test_result.price_series[0];
                    let cap = test_result.initial_capital;
                    if p0 > 0.0 && cap > 0.0 {
                        // Resample to the same length as the equity curve so the two line up exactly
                        let n = test_result.equity_curve.len();
                        let m = test_result.price_series.len();
                        (0..n).map(|i| {
                            let idx = if n <= 1 { 0 } else { (i * (m - 1)) / (n - 1) };
                            cap * test_result.price_series[idx] / p0
                        }).collect()
                    } else {
                        Vec::new()
                    }
                } else if let (Some(p_start), Some(p_end)) =
                    (test_result.first_trade_price, test_result.last_trade_price)
                {
                    // Fallback: linear interpolation between first and last trade prices
                    let cap = test_result.initial_capital;
                    let n = test_result.equity_curve.len();
                    if p_start > 0.0 && cap > 0.0 && n >= 2 {
                        (0..n).map(|i| {
                            let frac = i as f64 / (n - 1) as f64;
                            let p = p_start + frac * (p_end - p_start);
                            cap * p / p_start
                        }).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                oos_equity_curves.push(test_result.equity_curve);
                oos_bh_curves.push(bh_curve);
            }
        }

        // Restore the full-data parameters — per-window re-optimization above
        // left the strategy manager holding the LAST window's parameters.
        if let Some(params) = full_data_params {
            strategy_manager.update_all_parameters(params).await?;
        }

        let periods_tested = out_of_sample_sharpes.len();
        if periods_tested == 0 {
            return Ok(WalkForwardSummary {
                periods_tested: 0,
                average_performance: 0.0,
                consistency_score: 0.0,
                out_of_sample_degradation: 0.0,
                ..Default::default()
            });
        }
        
        let avg_out_sample = out_of_sample_sharpes.iter().sum::<f64>() / out_of_sample_sharpes.len() as f64;
        
        // Calculate consistency: 1 / (1 + σ) — measures uniformity
        let mean = avg_out_sample;
        let variance = out_of_sample_sharpes.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / out_of_sample_sharpes.len() as f64;
        let std_dev = variance.sqrt();
        let consistency_score = (1.0 / (1.0 + std_dev)).clamp(0.0, 1.0);
        
        // Measure parameter stability across time windows (lower = more stable)
        // High std dev relative to mean indicates parameters don't generalize well
        let degradation = (std_dev / (mean.abs() + 1e-6)).min(1.0);
        
        // Concatenate per-window OOS equity curves into a single curve.
        // Each window starts from its own initial capital; chain them so
        // the end of window N becomes the start of window N+1.
        let oos_equity_curve: Vec<f64> = {
            let mut merged = Vec::new();
            for curve in &oos_equity_curves {
                if curve.len() < 2 { continue; }
                if merged.is_empty() {
                    // First window: take as-is
                    merged.extend_from_slice(curve);
                } else {
                    // Subsequent windows: chain from where the previous ended
                    let prev_end = *merged.last().unwrap();
                    let window_base = curve[0];
                    let offset = prev_end - window_base;
                    // Skip first point (overlaps with previous end)
                    for &val in &curve[1..] {
                        merged.push(val + offset);
                    }
                }
            }
            merged
        };

        // Stitch per-window buy-and-hold curves with the same offset logic so the
        // result lines up index-for-index with `oos_equity_curve`.
        let oos_buy_and_hold_equity_curve: Vec<f64> = {
            let mut merged = Vec::new();
            for (curve, eq_curve) in oos_bh_curves.iter().zip(oos_equity_curves.iter()) {
                if eq_curve.len() < 2 || curve.len() < 2 { continue; }
                if merged.is_empty() {
                    merged.extend_from_slice(curve);
                } else {
                    let prev_end = *merged.last().unwrap();
                    let window_base = curve[0];
                    let offset = prev_end - window_base;
                    for &val in &curve[1..] {
                        merged.push(val + offset);
                    }
                }
            }
            // Only emit if it lines up with the strategy curve length
            if merged.len() == oos_equity_curve.len() { merged } else { Vec::new() }
        };
        
        // Last-window recency: OOS Sharpe from the most recent time window
        let last_window_oos_sharpe = out_of_sample_sharpes.last().copied();

        // Mann-Kendall trend test on the OOS Sharpe time series.
        // τ < 0 and small p-value indicates declining out-of-sample performance.
        let (mann_kendall_tau, mann_kendall_p) = if out_of_sample_sharpes.len() >= 3 {
            let n = out_of_sample_sharpes.len();
            let mut s: i64 = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let diff = out_of_sample_sharpes[j] - out_of_sample_sharpes[i];
                    if diff > 1e-10 { s += 1; }
                    else if diff < -1e-10 { s -= 1; }
                }
            }
            let n_f = n as f64;
            let tau = s as f64 / (n_f * (n_f - 1.0) / 2.0);
            let var_s = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0;
            let z = if s > 0 { (s as f64 - 1.0) / var_s.sqrt() }
                    else if s < 0 { (s as f64 + 1.0) / var_s.sqrt() }
                    else { 0.0 };
            let p = (2.0 * (1.0 - normal_cdf_approx(z.abs()))).min(1.0);
            (Some(tau), Some(p))
        } else {
            (None, None)
        };

        // Information Ratio vs risk-free rate: (strategy excess return) / tracking error.
        // Benchmark is a stablecoin/risk-free yield rather than buy-and-hold; the B&H
        // benchmark is too strict for crypto where passive holding often dominates.
        // Per-bar rf = (1 + rf_annual)^(bar_secs / year_secs) - 1; delta_secs comes
        // from the embargo calculation above.
        let oos_information_ratio: Option<f64> = if oos_equity_curve.len() >= 3 {
            let rf_annual = self.config.trading.risk_free_rate;
            let year_secs = 365.0 * 86_400.0_f64;
            let rf_per_bar = if delta_secs > 0 {
                (1.0 + rf_annual).powf(delta_secs as f64 / year_secs) - 1.0
            } else {
                rf_annual / 252.0  // daily fallback when bar duration is unknown
            };
            let strat: Vec<f64> = oos_equity_curve.windows(2)
                .filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
                .collect();
            if strat.is_empty() {
                None
            } else {
                let active: Vec<f64> = strat.iter().map(|s| s - rf_per_bar).collect();
                let n = active.len() as f64;
                let mean_a = active.iter().sum::<f64>() / n;
                let var_a = active.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n;
                let te = var_a.sqrt();
                if te > 1e-10 { Some(mean_a / te) } else { None }
            }
        } else {
            None
        };

        // Full-OOS aggregates (previously left at Default 0.0 even though
        // downstream consumers prefer them over the in-sample figures).
        let oos_max_drawdown = {
            let mut peak = f64::MIN;
            let mut max_dd = 0.0_f64;
            for &v in &oos_equity_curve {
                if v > peak { peak = v; }
                if peak > 0.0 {
                    let dd = (peak - v) / peak;
                    if dd > max_dd { max_dd = dd; }
                }
            }
            max_dd
        };
        let oos_win_rate = if oos_win_rate_weight > 0.0 {
            oos_win_rate_weighted / oos_win_rate_weight
        } else {
            0.0
        };
        let oos_profit_factor = if oos_gross_loss_sum > 0.0 {
            oos_gross_profit_sum / oos_gross_loss_sum
        } else if oos_gross_profit_sum > 0.0 {
            100.0 // capped: profits with zero losses
        } else {
            0.0
        };

        Ok(WalkForwardSummary {
            periods_tested,
            average_performance: avg_out_sample,
            consistency_score,
            out_of_sample_degradation: degradation,
            oos_total_pnl: oos_total_pnl_sum,
            oos_sharpe_ratio: avg_out_sample,
            oos_max_drawdown,
            oos_win_rate,
            oos_profit_factor,
            oos_num_trades: oos_trades_sum,
            oos_equity_curve,
            oos_buy_and_hold_equity_curve,
            last_window_oos_sharpe,
            mann_kendall_tau,
            mann_kendall_p,
            oos_information_ratio,
        })
    }
    
    /// Generate performance-based recommendations
    fn generate_recommendations(&self, result: &BacktestResult, _mc_result: &Option<monte_carlo::MonteCarloResult>) -> Vec<String> {
        let mut recommendations = Vec::with_capacity(4);  // Typical number of recommendations
        
        if let Some(sharpe) = result.sharpe_ratio {
            if sharpe < 1.0 {
                recommendations.push("Consider adjusting position sizing or risk parameters".to_string());
            } else if sharpe > 2.0 {
                recommendations.push("Strong performance - consider increasing position size".to_string());
            }
        }
        
        recommendations
    }
    
    /// Calculate data quality score for the market data
    fn calculate_data_quality_score(&self, market_data: &[MarketData]) -> f64 {
        if market_data.is_empty() {
            return 0.0;
        }
        
        let mut score: f64 = 10.0;
        
        // Check for sufficient data points
        if market_data.len() < 100 {
            score -= 2.0;
        }
        
        // Analyze gaps in the data
        let gap_config = &self.config.data.gap_detection;
        if gap_config.enabled && market_data.len() >= 2 {
            let gaps = self.detect_data_gaps(market_data);
            
            // Penalize based on number and severity of gaps
            let severe_gaps = gaps.iter().filter(|g| matches!(g.severity, crate::types::GapSeverity::Severe | crate::types::GapSeverity::Critical)).count();
            let moderate_gaps = gaps.iter().filter(|g| g.severity == crate::types::GapSeverity::Moderate).count();
            
            score -= (severe_gaps as f64 * 0.5).min(3.0);
            score -= (moderate_gaps as f64 * 0.2).min(2.0);
            
            // Penalize high gap ratio
            if let Some(first_ts) = self.get_timestamp(&market_data[0]) {
                if let Some(last_ts) = self.get_timestamp(&market_data[market_data.len() - 1]) {
                    let total_duration = (last_ts - first_ts).num_seconds();
                    let gap_duration: i64 = gaps.iter().map(|g| g.duration_seconds).sum();
                    let gap_ratio = gap_duration as f64 / total_duration as f64;
                    
                    if gap_ratio > 0.2 {
                        score -= 2.0;
                    } else if gap_ratio > 0.1 {
                        score -= 1.0;
                    }
                }
            }
        }
        
        score.max(0.0).min(10.0)
    }
    
    /// Detect gaps in market data
    fn detect_data_gaps(&self, market_data: &[MarketData]) -> Vec<crate::types::DataGap> {
        let mut gaps = Vec::with_capacity(50);  // Pre-allocate for typical gap count
        let gap_config = &self.config.data.gap_detection;
        
        if !gap_config.enabled || market_data.len() < 2 {
            return gaps;
        }
        
        let threshold_seconds = (gap_config.expected_frequency_seconds as f64 * gap_config.gap_threshold_multiplier) as i64;
        
        for i in 1..market_data.len() {
            if let (Some(prev_ts), Some(curr_ts)) = (
                self.get_timestamp(&market_data[i - 1]),
                self.get_timestamp(&market_data[i])
            ) {
                let gap_duration = (curr_ts - prev_ts).num_seconds();
                
                if gap_duration > threshold_seconds {
                    let severity = if gap_duration > gap_config.expected_frequency_seconds * 10 {
                        crate::types::GapSeverity::Critical
                    } else if gap_duration > gap_config.expected_frequency_seconds * 5 {
                        crate::types::GapSeverity::Severe
                    } else if gap_duration > gap_config.expected_frequency_seconds * 2 {
                        crate::types::GapSeverity::Moderate
                    } else {
                        crate::types::GapSeverity::Minor
                    };
                    
                    // Log warnings for significant gaps
                    if gap_duration > gap_config.warning_threshold_seconds {
                        log_warn!(BACKTEST_LOGGER, "Data gap detected: {:.1} minutes ({:?})", 
                            gap_duration as f64 / 60.0, &severity);
                    }
                    
                    gaps.push(crate::types::DataGap {
                        start_time: prev_ts,
                        end_time: curr_ts,
                        duration_seconds: gap_duration,
                        severity,
                    });
                }
            }
        }
        
        // Check for critical gaps that should fail the backtest
        if let Some(critical_threshold) = gap_config.critical_threshold_seconds {
            let critical_gaps: Vec<_> = gaps.iter()
                .filter(|g| g.duration_seconds > critical_threshold)
                .collect();
            
            if !critical_gaps.is_empty() {
                log_error!(BACKTEST_LOGGER, "CRITICAL: {} gap(s) exceed {}s threshold", 
                    critical_gaps.len(), critical_threshold);
            }
        }
        
        gaps
    }
    
    /// Generate comprehensive data quality report.
    /// (Previously duplicated under postgres/non-postgres feature gates with
    /// identical bodies, and panicked on empty input via `market_data[0]`.)
    fn generate_data_quality_report(&self, market_data: &[MarketData]) -> crate::types::DataQualityReport {
        if market_data.is_empty() {
            return crate::types::DataQualityReport {
                overall_score: 0.0,
                total_data_points: 0,
                time_span_seconds: 0,
                gaps_detected: Vec::new(),
                average_gap_seconds: 0.0,
                max_gap_seconds: 0,
                gap_ratio: 0.0,
                outliers_detected: 0,
                data_completeness_score: 0.0,
                timestamp_consistency_score: 0.0,
                order_quality: None,
            };
        }

        let gaps = self.detect_data_gaps(market_data);
        let total_data_points = market_data.len();

        let (time_span, avg_gap, max_gap, gap_ratio) = if let (Some(first), Some(last)) = (
            self.get_timestamp(&market_data[0]),
            self.get_timestamp(&market_data[market_data.len() - 1])
        ) {
            let time_span = (last - first).num_seconds();
            let total_gap_time: i64 = gaps.iter().map(|g| g.duration_seconds).sum();
            let avg_gap = if !gaps.is_empty() {
                total_gap_time as f64 / gaps.len() as f64
            } else {
                0.0
            };
            let max_gap = gaps.iter().map(|g| g.duration_seconds).max().unwrap_or(0);
            let gap_ratio = if time_span > 0 {
                total_gap_time as f64 / time_span as f64
            } else {
                0.0
            };
            (time_span, avg_gap, max_gap, gap_ratio)
        } else {
            (0, 0.0, 0, 0.0)
        };

        let overall_score = self.calculate_data_quality_score(market_data);
        let data_completeness_score = ((1.0 - gap_ratio) * 10.0).max(0.0).min(10.0);
        let timestamp_consistency_score = if gaps.is_empty() { 10.0 } else {
            (10.0 - (gaps.len() as f64 * 0.1)).max(0.0)
        };

        crate::types::DataQualityReport {
            overall_score,
            total_data_points,
            time_span_seconds: time_span,
            gaps_detected: gaps,
            average_gap_seconds: avg_gap,
            max_gap_seconds: max_gap,
            gap_ratio,
            outliers_detected: 0, // TODO: implement outlier detection
            data_completeness_score,
            timestamp_consistency_score,
            order_quality: None,
        }
    }
    
    /// Helper to extract timestamp from MarketData
    fn get_timestamp(&self, data: &MarketData) -> Option<chrono::DateTime<chrono::Utc>> {
        match data {
            MarketData::Candle(c) => Some(c.timestamp),
            MarketData::Trade(t) => Some(t.timestamp),
            MarketData::PoolSwap(s) => Some(s.timestamp),
            MarketData::Generic(g) => Some(chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH)),
            MarketData::OptionCandle(c) => Some(c.timestamp),
        }
    }
    
    /// Apply gap filling strategy to market data
    fn apply_gap_filling(&self, market_data: &[MarketData]) -> Result<Vec<MarketData>> {
        use config::GapFillStrategy;
        use chrono::Duration;
        
        let gap_config = &self.config.data.gap_detection;
        
        match gap_config.fill_strategy {
            GapFillStrategy::None => {
                // No filling - return as-is
                Ok(market_data.to_vec())
            }
            GapFillStrategy::ForwardFill => {
                log_info!(BACKTEST_LOGGER, "Applying forward fill strategy");
                let mut filled_data = Vec::new();
                
                for (i, data) in market_data.iter().enumerate() {
                    filled_data.push(data.clone());
                    
                    // Check if there's a gap to the next data point
                    if i < market_data.len() - 1 {
                        if let (Some(current_ts), Some(next_ts)) = (
                            self.get_timestamp(data),
                            self.get_timestamp(&market_data[i + 1])
                        ) {
                            let gap = next_ts - current_ts;
                            let gap_seconds = gap.num_seconds();
                            let threshold = (gap_config.expected_frequency_seconds as f64 * gap_config.gap_threshold_multiplier) as i64;
                            
                            if gap_seconds > threshold {
                                // Fill the gap with forward-filled data
                                let num_fills = (gap_seconds / gap_config.expected_frequency_seconds) - 1;
                                for j in 1..=num_fills {
                                    let fill_timestamp = current_ts + Duration::seconds(gap_config.expected_frequency_seconds * j);
                                    let filled_point = self.create_filled_data_point(data, fill_timestamp);
                                    filled_data.push(filled_point);
                                }
                            }
                        }
                    }
                }
                
                Ok(filled_data)
            }
            GapFillStrategy::LinearInterpolation => {
                log_info!(BACKTEST_LOGGER, "Applying linear interpolation strategy");
                let mut filled_data = Vec::new();
                
                for (i, data) in market_data.iter().enumerate() {
                    filled_data.push(data.clone());
                    
                    if i < market_data.len() - 1 {
                        if let (Some(current_ts), Some(next_ts)) = (
                            self.get_timestamp(data),
                            self.get_timestamp(&market_data[i + 1])
                        ) {
                            let gap = next_ts - current_ts;
                            let gap_seconds = gap.num_seconds();
                            let threshold = (gap_config.expected_frequency_seconds as f64 * gap_config.gap_threshold_multiplier) as i64;
                            
                            if gap_seconds > threshold {
                                // Interpolate between current and next data points
                                let num_fills = (gap_seconds / gap_config.expected_frequency_seconds) - 1;
                                for j in 1..=num_fills {
                                    let fill_timestamp = current_ts + Duration::seconds(gap_config.expected_frequency_seconds * j);
                                    let ratio = j as f64 / (num_fills + 1) as f64;
                                    let filled_point = self.create_interpolated_data_point(data, &market_data[i + 1], fill_timestamp, ratio);
                                    filled_data.push(filled_point);
                                }
                            }
                        }
                    }
                }
                
                Ok(filled_data)
            }
            GapFillStrategy::ExcludeGaps => {
                log_info!(BACKTEST_LOGGER, "Excluding data around gaps");
                let gaps = self.detect_data_gaps(market_data);
                let mut filtered_data = Vec::new();

                for data in market_data {
                    if let Some(ts) = self.get_timestamp(data) {
                        // Exclusive bounds: gap start/end ARE real data points (the
                        // last point before and first point after the gap) — the
                        // previous inclusive check deleted exactly those two valid
                        // points per gap and nothing else.
                        let in_gap = gaps.iter().any(|gap| ts > gap.start_time && ts < gap.end_time);
                        if !in_gap {
                            filtered_data.push(data.clone());
                        }
                    }
                }

                Ok(filtered_data)
            }
        }
    }
    
    /// Create a forward-filled data point
    fn create_filled_data_point(&self, source: &MarketData, timestamp: chrono::DateTime<chrono::Utc>) -> MarketData {
        match source {
            MarketData::Candle(c) => {
                let mut filled = c.clone();
                filled.timestamp = timestamp;
                // Keep same OHLCV but mark it as a filled candle
                filled.volume = 0.0; // Zero volume indicates synthetic data
                MarketData::Candle(filled)
            }
            MarketData::Trade(t) => {
                let mut filled = t.clone();
                filled.timestamp = timestamp;
                filled.quantity = 0.0; // Zero quantity indicates synthetic data
                MarketData::Trade(filled)
            }
            MarketData::PoolSwap(s) => {
                let mut filled = s.clone();
                filled.timestamp = timestamp;
                filled.amount_in = 0.0; // Zero amount indicates synthetic data
                MarketData::PoolSwap(filled)
            }
            MarketData::Generic(g) => {
                let mut filled = g.clone();
                filled.timestamp_ms = timestamp.timestamp_millis();
                filled.quantity = 0.0; // Zero quantity indicates synthetic data
                MarketData::Generic(filled)
            }
            MarketData::OptionCandle(c) => {
                let mut filled = c.clone();
                filled.timestamp = timestamp;
                filled.volume = 0.0; // Zero volume indicates synthetic data
                MarketData::OptionCandle(filled)
            }
        }
    }

    /// Create an interpolated data point between two data points
    fn create_interpolated_data_point(&self, start: &MarketData, end: &MarketData, timestamp: chrono::DateTime<chrono::Utc>, ratio: f64) -> MarketData {
        match (start, end) {
            (MarketData::Candle(c1), MarketData::Candle(c2)) => {
                // Linear interpolation for OHLC prices
                let open_interp = c1.open + (c2.open - c1.open) * ratio;
                let close_interp = c1.close + (c2.close - c1.close) * ratio;
                
                let interpolated = dataloader::Candle {
                    timestamp,
                    symbol: c1.symbol.clone(),
                    exchange: c1.exchange.clone(),
                    open: open_interp,
                    high: if c1.high > c2.high { c1.high } else { c2.high }, // Conservative: use max high
                    low: if c1.low < c2.low { c1.low } else { c2.low },     // Conservative: use min low
                    close: close_interp,
                    volume: 0.0, // Zero volume indicates synthetic data
                    trade_count: 0,
                };
                MarketData::Candle(interpolated)
            }
            (MarketData::Trade(t1), MarketData::Trade(t2)) => {
                let price = t1.price + (t2.price - t1.price) * ratio;
                
                let interpolated = dataloader::TradeData {
                    timestamp,
                    symbol: t1.symbol.clone(),
                    exchange: t1.exchange.clone(),
                    price,
                    quantity: 0.0, // Zero quantity indicates synthetic data
                    side: t1.side,
                    trade_id: timestamp.timestamp(),
                    value: 0.0,
                };
                MarketData::Trade(interpolated)
            }
            _ => {
                // For mismatched types or unsupported types, fall back to forward fill
                self.create_filled_data_point(start, timestamp)
            }
        }
    }
}

impl BacktestResults {
    /// Print comprehensive summary of backtest results
    pub fn print_summary(&self) {
        use crate::logging_facade::BACKTEST_LOGGER;
        
        BACKTEST_LOGGER.info_sync("=== BACKTEST RESULTS SUMMARY ===".to_string());
        
        // Core results
        let mut core_msg = String::from("CORE METRICS: ");
        if let Some(sharpe) = self.core.sharpe_ratio {
            core_msg.push_str(&format!("Sharpe: {:.2}, ", sharpe));
        }
        if let Some(pf) = self.core.profit_factor {
            core_msg.push_str(&format!("PF: {:.2}, ", pf));
        }
        if let Some(wr) = self.core.win_rate {
            core_msg.push_str(&format!("Win Rate: {:.1}%", wr * 100.0));
        }
        BACKTEST_LOGGER.info_sync(core_msg);
        
        // Monte Carlo results
        if let Some(mc) = &self.monte_carlo {
            BACKTEST_LOGGER.info_sync(format!("MONTE CARLO ({} simulations): Mean Sharpe: {:.2}", 
                mc.num_runs, mc.mean_metrics.sharpe_ratio));
        }
        
        // Walk-forward results (average_performance is a Sharpe ratio, not a percentage)
        if let Some(wf) = &self.walk_forward {
            BACKTEST_LOGGER.info_sync(format!("WALK-FORWARD: Periods: {}, Avg OOS Sharpe: {:.2}, Consistency: {:.1}/10",
                wf.periods_tested, wf.average_performance, wf.consistency_score * 10.0));
        }
        
        // Metadata
        BACKTEST_LOGGER.info_sync(format!("METADATA: Mode: {:?}, Escalated: {}, Runtime: {:.1}s, Data Quality: {:.1}/10",
            self.metadata.analysis_mode, self.metadata.mode_escalated, 
            self.metadata.total_runtime_seconds, self.metadata.data_quality_score));
        
        // Data Quality Report
        if let Some(dq) = &self.core.data_quality {
            let mut dq_msg = format!("DATA QUALITY: Score: {:.1}/10, Points: {}, Time Span: {:.1}h, Gaps: {}",
                dq.overall_score, dq.total_data_points, 
                dq.time_span_seconds as f64 / 3600.0, dq.gaps_detected.len());
            if dq.max_gap_seconds > 0 {
                dq_msg.push_str(&format!(", Max Gap: {:.1}m, Gap Ratio: {:.1}%", 
                    dq.max_gap_seconds as f64 / 60.0, dq.gap_ratio * 100.0));
            }
            BACKTEST_LOGGER.info_sync(dq_msg);
        }
        
        // Transaction Cost Analysis
        if let Some(tc) = &self.core.transaction_costs {
            BACKTEST_LOGGER.info_sync(format!("TRANSACTION COSTS: Total: ${:.2}, Commission: ${:.2}, Slippage: ${:.2}, Impact: ${:.2}, Costs/PnL: {:.1}%",
                tc.total_transaction_costs, tc.total_commission, tc.total_slippage, 
                tc.total_market_impact, tc.costs_as_percentage_of_pnl));
        }
        
        // Market Impact Analysis
        if let Some(mi) = &self.core.market_impact {
            if mi.trades_with_impact > 0 {
                let mut mi_msg = format!("MARKET IMPACT: Trades: {}, Avg: {:.2}bps, Max: {:.2}bps, Total Cost: ${:.2}",
                    mi.trades_with_impact, mi.avg_impact_bps, mi.max_impact_bps, mi.total_impact_cost);
                if !mi.liquidity_warnings.is_empty() {
                    mi_msg.push_str(&format!(", Liquidity Warnings: {}", mi.liquidity_warnings.len()));
                }
                BACKTEST_LOGGER.info_sync(mi_msg);
            }
        }
        
        if !self.metadata.triggers_fired.is_empty() {
            BACKTEST_LOGGER.info_sync(format!("TRIGGERS FIRED: {}", self.metadata.triggers_fired.join(", ")));
        }
        
        if !self.metadata.recommendations.is_empty() {
            BACKTEST_LOGGER.info_sync(format!("RECOMMENDATIONS: {}", self.metadata.recommendations.join("; ")));
        }
        
        BACKTEST_LOGGER.info_sync("=== END RESULTS SUMMARY ===".to_string());
    }
    
    /// Check if results meet export criteria
    pub fn should_export(&self, export_config: &ExportConfig) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();
        let mut should_export = true;
        
        // Check core criteria
        if let Some(sharpe) = self.core.sharpe_ratio {
            if sharpe < export_config.min_sharpe_ratio {
                should_export = false;
                reasons.push(format!("Sharpe ratio {:.2} below minimum {:.2}", 
                    sharpe, export_config.min_sharpe_ratio));
            }
        }
        
        if let Some(pf) = self.core.profit_factor {
            if pf < export_config.min_profit_factor {
                should_export = false;
                reasons.push(format!("Profit factor {:.2} below minimum {:.2}", 
                    pf, export_config.min_profit_factor));
            }
        }
        
        // Enhanced criteria
        if self.metadata.data_quality_score < 7.0 {
            should_export = false;
            reasons.push("Data quality score too low".to_string());
        }
        
        if should_export {
            reasons.push("All export criteria met".to_string());
        }
        
        (should_export, reasons)
    }
}

/// Normal CDF: Φ(x) = 0.5·(1 + erf(x/√2)), with erf approximated via
/// Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5×10⁻⁷).
/// Used by the Mann-Kendall p-value computation in run_walk_forward_analysis.
/// NOTE: the erf argument must be x/√2 — an earlier version dropped that
/// scaling and biased the p-values.
fn normal_cdf_approx(x: f64) -> f64 {
    const P: f64 = 0.3275911;
    const A: [f64; 5] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + P * z);
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    let erf_approx = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 { 0.5 * (1.0 + erf_approx) } else { 0.5 * (1.0 - erf_approx) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::{BacktestConfig, AnalysisConfig, AnalysisMode};

    /// Minimal test-only fitness function -- just enough for the GA path to
    /// exercise real code, not meant to demonstrate a real scoring policy.
    struct TestFitnessFunction;
    impl genetic::fitness::FitnessFunction for TestFitnessFunction {
        fn compute(&self, inputs: &genetic::fitness::FitnessInputs) -> genetic::FitnessResult {
            if inputs.num_trades == 0 {
                return genetic::FitnessResult::failure();
            }
            genetic::FitnessResult::with_metrics(
                inputs.sharpe,
                inputs.equity_curve.clone(),
                inputs.net_pnl,
                inputs.sharpe,
                inputs.num_trades as i32,
                inputs.max_drawdown_abs,
                inputs.win_rate,
                inputs.profit_factor,
            )
        }
    }

    fn make_engine() -> BacktestEngine {
        BacktestEngine::new(BacktestConfig::default(), Arc::new(TestFitnessFunction))
    }

    fn make_engine_with_mode(mode: AnalysisMode) -> BacktestEngine {
        let mut config = BacktestConfig::default();
        config.analysis.mode = mode;
        BacktestEngine::new(config, Arc::new(TestFitnessFunction))
    }

    // WalkForwardSummary
    #[test]
    fn test_walk_forward_summary_default() {
        let wfs = WalkForwardSummary::default();
        assert_eq!(wfs.periods_tested, 0);
        assert_eq!(wfs.average_performance, 0.0);
        assert_eq!(wfs.consistency_score, 0.0);
        assert_eq!(wfs.out_of_sample_degradation, 0.0);
        assert_eq!(wfs.oos_total_pnl, 0.0);
        assert_eq!(wfs.oos_sharpe_ratio, 0.0);
        assert_eq!(wfs.oos_max_drawdown, 0.0);
        assert_eq!(wfs.oos_win_rate, 0.0);
        assert_eq!(wfs.oos_profit_factor, 0.0);
        assert_eq!(wfs.oos_num_trades, 0);
        assert!(wfs.oos_equity_curve.is_empty());
    }

    #[test]
    fn test_walk_forward_summary_serde_roundtrip() {
        let wfs = WalkForwardSummary {
            periods_tested: 5,
            average_performance: 1.5,
            consistency_score: 0.8,
            out_of_sample_degradation: 0.2,
            oos_total_pnl: 1000.0,
            oos_sharpe_ratio: 1.8,
            oos_max_drawdown: 0.15,
            oos_win_rate: 0.55,
            oos_profit_factor: 1.6,
            oos_num_trades: 100,
            oos_equity_curve: vec![100.0, 105.0, 110.0],
            oos_buy_and_hold_equity_curve: vec![100.0, 103.0, 108.0],
            last_window_oos_sharpe: None,
            mann_kendall_tau: None,
            mann_kendall_p: None,
            oos_information_ratio: None,
        };
        let json = serde_json::to_string(&wfs).unwrap();
        let deserialized: WalkForwardSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.periods_tested, 5);
        assert_eq!(deserialized.oos_total_pnl, 1000.0);
    }

    // determine_analysis_mode
    #[test]
    fn test_determine_analysis_mode_export_forces_production() {
        let engine = make_engine_with_mode(AnalysisMode::Development);
        assert_eq!(engine.determine_analysis_mode(true), AnalysisMode::Production);
    }

    #[test]
    fn test_determine_analysis_mode_no_export_uses_config() {
        let engine = make_engine_with_mode(AnalysisMode::Validation);
        assert_eq!(engine.determine_analysis_mode(false), AnalysisMode::Validation);
    }

    #[test]
    fn test_determine_analysis_mode_development() {
        let engine = make_engine_with_mode(AnalysisMode::Development);
        assert_eq!(engine.determine_analysis_mode(false), AnalysisMode::Development);
    }

    // should_escalate_to_validation
    #[test]
    fn test_should_escalate_to_validation_poor_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        // Default validation_sharpe_threshold is typically > 0
        result.sharpe_ratio = Some(-1.0);
        assert!(engine.should_escalate_to_validation(&result));
    }

    #[test]
    fn test_should_escalate_to_validation_good_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        result.sharpe_ratio = Some(5.0);
        assert!(!engine.should_escalate_to_validation(&result));
    }

    #[test]
    fn test_should_escalate_to_validation_no_sharpe() {
        let engine = make_engine();
        let result = BacktestResult::default();
        assert!(!engine.should_escalate_to_validation(&result));
    }

    // should_escalate_to_production
    #[test]
    fn test_should_escalate_to_production_excellent_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        result.sharpe_ratio = Some(100.0); // Way above 1.5x threshold
        assert!(engine.should_escalate_to_production(&result));
    }

    #[test]
    fn test_should_escalate_to_production_mediocre_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        result.sharpe_ratio = Some(0.1);
        assert!(!engine.should_escalate_to_production(&result));
    }

    #[test]
    fn test_should_escalate_to_production_no_sharpe() {
        let engine = make_engine();
        let result = BacktestResult::default();
        assert!(!engine.should_escalate_to_production(&result));
    }

    // generate_recommendations
    #[test]
    fn test_generate_recommendations_low_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        result.sharpe_ratio = Some(0.5);
        let recs = engine.generate_recommendations(&result, &None);
        assert!(!recs.is_empty());
        assert!(recs[0].contains("position sizing"));
    }

    #[test]
    fn test_generate_recommendations_high_sharpe() {
        let engine = make_engine();
        let mut result = BacktestResult::default();
        result.sharpe_ratio = Some(2.5);
        let recs = engine.generate_recommendations(&result, &None);
        assert!(!recs.is_empty());
        assert!(recs[0].contains("Strong performance"));
    }

    #[test]
    fn test_generate_recommendations_no_sharpe() {
        let engine = make_engine();
        let result = BacktestResult::default();
        let recs = engine.generate_recommendations(&result, &None);
        assert!(recs.is_empty());
    }

    // BacktestResults::should_export
    #[test]
    fn test_should_export_good_results() {
        let results = BacktestResults {
            core: BacktestResult {
                sharpe_ratio: Some(2.0),
                profit_factor: Some(1.5),
                ..Default::default()
            },
            monte_carlo: None,
            walk_forward: None,
            metadata: AnalysisMetadata {
                analysis_mode: AnalysisMode::Production,
                mode_escalated: false,
                total_runtime_seconds: 10.0,
                data_quality_score: 8.0,
                triggers_fired: vec![],
                recommendations: vec![],
            },
        };
        let export_config = ExportConfig::default();
        let (should, reasons) = results.should_export(&export_config);
        assert!(should, "Should export good results, reasons: {:?}", reasons);
    }

    #[test]
    fn test_should_export_poor_data_quality() {
        let results = BacktestResults {
            core: BacktestResult {
                sharpe_ratio: Some(3.0),
                profit_factor: Some(2.0),
                ..Default::default()
            },
            monte_carlo: None,
            walk_forward: None,
            metadata: AnalysisMetadata {
                analysis_mode: AnalysisMode::Production,
                mode_escalated: false,
                total_runtime_seconds: 10.0,
                data_quality_score: 5.0, // Below 7.0
                triggers_fired: vec![],
                recommendations: vec![],
            },
        };
        let export_config = ExportConfig::default();
        let (should, reasons) = results.should_export(&export_config);
        assert!(!should);
        assert!(reasons.iter().any(|r| r.contains("quality")));
    }

    // AnalysisMetadata serde
    #[test]
    fn test_analysis_metadata_serde() {
        let meta = AnalysisMetadata {
            analysis_mode: AnalysisMode::Production,
            mode_escalated: true,
            total_runtime_seconds: 42.5,
            data_quality_score: 8.5,
            triggers_fired: vec!["test trigger".to_string()],
            recommendations: vec!["test rec".to_string()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: AnalysisMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_runtime_seconds, 42.5);
        assert!(deserialized.mode_escalated);
    }

    // BacktestResults serde
    #[test]
    fn test_backtest_results_serde() {
        let results = BacktestResults {
            core: BacktestResult::default(),
            monte_carlo: None,
            walk_forward: Some(WalkForwardSummary::default()),
            metadata: AnalysisMetadata {
                analysis_mode: AnalysisMode::Development,
                mode_escalated: false,
                total_runtime_seconds: 1.0,
                data_quality_score: 9.0,
                triggers_fired: vec![],
                recommendations: vec![],
            },
        };
        let json = serde_json::to_string(&results).unwrap();
        let deserialized: BacktestResults = serde_json::from_str(&json).unwrap();
        assert!(deserialized.walk_forward.is_some());
    }
}
