/// Genetic Algorithm optimization for trading strategies
/// 
/// Provides async genetic algorithm optimization for finding optimal
/// trading strategy parameters with parallel fitness evaluation.
/// 
/// Includes robustness-weighted fitness using Monte Carlo consistency checks
/// on top performers to prevent overfitting to lucky parameter combinations.
///
/// Performance optimizations:
/// - Adaptive sampling: Coarser data in early generations for faster exploration
/// - Dynamic concurrency: Memory-aware parallel evaluation limits
/// - SIMD utilities: Vectorized calculations for metrics
/// - Diversity preservation: Fitness sharing and crowding penalties

use std::sync::Arc;
use std::collections::HashMap;
use config::GeneticConfig;
use rand::prelude::*;
use ultra_logger::{UltraLogger, LoggerConfig, TransportConfig, ConnectionConfig};
use once_cell::sync::Lazy;

// ---------------------------------------------------------------------------
// Logger and structured logging macros (must be defined before mod declarations)
// ---------------------------------------------------------------------------

/// Create Elasticsearch configuration for genetic module
fn create_elasticsearch_config() -> LoggerConfig {
    // Check if we should use Elasticsearch (environment variable or default)
    let use_elasticsearch = std::env::var("USE_ELASTICSEARCH_LOGGING")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(true); // Default to true for production

    if use_elasticsearch {
        // Get credentials from environment variables set by Kubernetes
        let endpoint = std::env::var("ELASTICSEARCH_ENDPOINT")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_ENDPOINT"))
            .unwrap_or_default();
        let username = std::env::var("ELASTICSEARCH_USERNAME")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_USERNAME"))
            .unwrap_or_default();
        let password = std::env::var("ELASTICSEARCH_PASSWORD")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_PASSWORD"))
            .unwrap_or_default();

        let mut options = HashMap::new();
        options.insert("index".to_string(), "backtestingengine-logs".to_string());

        LoggerConfig {
            level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            transport: TransportConfig {
                transport_type: "elasticsearch".to_string(),
                connection: ConnectionConfig {
                    host: endpoint,
                    port: 443,
                    username: Some(username),
                    password: Some(password),
                    options,
                },
            },
        }
    } else {
        // Fallback to stdout for development
        LoggerConfig::default()
    }
}

/// Logger for genetic algorithm operations with Elasticsearch support
static GENETIC_LOGGER: Lazy<Arc<UltraLogger>> = Lazy::new(|| {
    Arc::new(UltraLogger::with_config(
        "BacktestingEngine-genetic".to_string(),
        create_elasticsearch_config()
    ))
});

/// Log info message using tokio::spawn for async logging (matches worker's pattern)
macro_rules! log_info {
    ($($arg:tt)*) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = crate::GENETIC_LOGGER.clone();
            let msg = format!($($arg)*);
            handle.spawn(async move {
                let _ = logger.info(msg).await;
            });
        }
    }};
}

macro_rules! log_info_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Info, msg, metadata).await;
            });
        }
    }};
}

macro_rules! log_warn_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Warn, msg, metadata).await;
            });
        }
    }};
}

macro_rules! log_error_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Error, msg, metadata).await;
            });
        }
    }};
}

// ---------------------------------------------------------------------------


/// Dynamic chromosome for Python strategy parameter optimization
pub mod dynamic_chromosome;

/// Strategy parameter schema presets (data-driven definitions for built-in strategies)
pub mod presets;

/// GA post-optimization analysis: landscape, sensitivity, overfitting
pub mod landscape;

// Re-export chromosome types for external use

// Re-export dynamic chromosome types
pub use dynamic_chromosome::{
    DynamicChromosome, ParameterSchema, ParameterDef, ParamType, Gene,
    set_dynamic_schema,
};

// Re-export landscape/report types
pub use landscape::{
    GAReport, TopParamSet, LandscapeReport, LandscapePoint,
    SensitivityEntry, ConvergencePoint, OverfitReport,
    generate_landscape_heatmap, sensitivity_analysis, overfitting_gap,
    extract_top_n,
};

/// Portfolio-level genetic algorithm optimization

/// Regime-aware Monte Carlo simulation for robust backtesting
pub mod regime_mc;

/// Diversity-preserving mechanisms for unique strategy generation
/// Prevents strategy crowding and alpha decay when user base grows
pub mod diversity;

// Re-export diversity types
pub use diversity::{
    BehavioralSignature, SignalType, StrategyRegistry, RegisteredStrategy,
    DiversityConfig, DiversityMetrics, apply_fitness_sharing, calculate_adjusted_fitness
};

/// Hidden Markov Model for probabilistic regime detection
pub mod hmm_regime;

/// Adaptive sampling and optimization utilities
pub mod adaptive_sampling;

/// NSGA-II multi-objective Pareto optimizer
pub mod nsga2;

/// Process-pool parallel GA evaluator for Python strategies (GIL bypass)
pub mod worker_pool;

/// Shared GA fitness-scoring function used by every candidate-evaluation
/// call site on the platform (see this module's own doc comment).
pub mod fitness;

pub use regime_mc::{
    RegimeMCConfig, RegimeMCResult, RegimeAwareMC, RegimeInfo,
    MarketRegime, run_regime_aware_monte_carlo,
};

pub use hmm_regime::{
    HMMConfig, HiddenMarkovModel, GaussianEmission, FitResult,
    HMMRegimeTracker, RegimeCharacteristics, RegimeUpdate,
};

pub use adaptive_sampling::{
    AdaptiveSamplingConfig, SampledData, sample_market_data,
    StableRegionDetector, DynamicConcurrency,
};

/// Represents a set of strategy parameters (chromosome)
pub trait Chromosome: Clone + Send + Sync {
    /// Randomly initialize a chromosome
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;
    /// Crossover with another chromosome
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self;
    /// Mutate the chromosome with adaptive mutation rate
    fn mutate(&mut self, rng: &mut impl Rng, mutation_rate: f64);
    /// Calculate distance to another chromosome for diversity metrics
    fn distance(&self, other: &Self) -> f64;

    /// Derive a [`diversity::BehavioralSignature`] describing *how* this
    /// chromosome trades (hold time, frequency, signal type, risk profile, ...),
    /// used by [`diversity::StrategyRegistry::calculate_crowding_penalty`] to
    /// detect crowded/similar deployed strategies.
    ///
    /// Default implementation returns [`diversity::BehavioralSignature::default()`]
    /// (i.e. "unknown"), which yields a neutral/no-op crowding penalty.
    /// [`dynamic_chromosome::DynamicChromosome`] overrides this with a real mapping.
    fn behavioral_signature(&self) -> diversity::BehavioralSignature {
        diversity::BehavioralSignature::default()
    }

    /// Derive a coarse 64-bit hash of this chromosome's parameters, used
    /// alongside [`Chromosome::behavioral_signature`] for crowding detection.
    /// The hash should be stable across near-identical parameter sets (i.e.
    /// values should be quantized) so that minor floating-point differences
    /// don't defeat similarity matching.
    ///
    /// Default implementation returns `0` ("unknown"). See
    /// [`Chromosome::behavioral_signature`] for override guidance.
    fn parameter_hash(&self) -> u64 {
        0
    }

    /// A coarse "how complex is this parameter set" measure, used by the
    /// optional parsimony penalty (`GeneticConfig::complexity_penalty_weight`)
    /// to bias the GA toward simpler strategies when two candidates score
    /// similarly on raw fitness -- fewer tunable parameters means fewer
    /// degrees of freedom available to overfit the fitness landscape.
    ///
    /// Default implementation returns `0.0` (no-op, matches the default
    /// `complexity_penalty_weight: 0.0` so existing behavior is unchanged).
    /// [`dynamic_chromosome::DynamicChromosome`] overrides this with its
    /// schema's parameter count.
    fn complexity(&self) -> f64 {
        0.0
    }
}

/// Sentinel fitness for chromosomes that produced ZERO trades.
///
/// Must be far below any achievable fitness (raw weighted-sum OR z-score
/// normalised) so that a zero-trade chromosome can NEVER outrank a chromosome
/// that actually traded — even a badly losing one. The previous sentinel of
/// -1.0 was NOT a floor: in an all-losing population, z-score normalisation
/// pushed genuinely-trading losers below -1.0, and fitness sharing then
/// divided the crowded zero-trade cluster's -1.0 toward 0, letting no-trade
/// chromosomes win the GA (observed: best_fitness -1.0 → -0.148 across
/// generations was purely a growing no-trade niche, job 64e39acf).
///
/// Kept finite (not NEG_INFINITY) to avoid NaN/inf propagation through
/// sharing / penalty arithmetic.
pub const ZERO_TRADE_FITNESS: f64 = -1_000.0;

/// Hard disqualification fitness for a chromosome whose evaluation triggered
/// one or more forced margin-call liquidations. Distinct from
/// [`ZERO_TRADE_FITNESS`] (a liquidated chromosome may have traded plenty and
/// produced otherwise-trustworthy metrics right up until it blew up) but held
/// to the same floor value so GA ranking stays unambiguous regardless of
/// which of the platform's several fitness formulas evaluated it. Exists
/// because leverage-evolving GA runs need an explicit ruin penalty: raw
/// return/Sharpe-based scoring has no other way to distinguish "leverage
/// that happened to work out in this backtest" from "leverage that will
/// eventually blow the account up" -- a liquidation is a discontinuous,
/// non-recoverable event, not something a weighted average should merely
/// discount.
pub const LIQUIDATION_FITNESS: f64 = -1_000.0;

/// Minimum trade count for a chromosome's metrics (Sharpe, win rate,
/// profit factor, ...) to be considered statistically meaningful rather than
/// luck. A strategy that only fired 2 trades and happened to win both looks
/// like a 100% win-rate genius by the raw weighted-sum formula, but that's
/// noise, not skill -- with n=2 the confidence interval on win rate spans
/// almost the entire [0,1] range. Matches the precedent already used
/// platform-wide for trusting a *final* backtest result at all
/// (`worker::MIN_TRADES_REQUIRED`). Raised 10 -> 30 (quant council
/// 2026-07-28) after auditing a workflow-submitted backtest that completed
/// with only 4 trades: 10 was too low a bar for Sharpe/win-rate/profit
/// factor to be trusted, especially for daily-candle swing strategies where
/// a full backtest year can plausibly produce single-digit trade counts.
pub const MIN_TRADES_FOR_SIGNIFICANCE: usize = 30;

/// Penalty fitness for chromosomes that traded at least once but fewer than
/// [`MIN_TRADES_FOR_SIGNIFICANCE`] times. Scaled by trade count so 9 trades
/// ranks strictly better than 1 (useful for GA diagnostics/debugging), but
/// the whole range stays far below any realistically-achievable weighted
/// fitness (~-3..+3 for the current weight profiles) so these candidates can
/// never win a generation over one with enough trades to trust its metrics --
/// while still ranking strictly above [`ZERO_TRADE_FITNESS`], since trading
/// a little is closer to viable than not trading at all.
pub fn low_trade_count_fitness(trades: usize) -> f64 {
    -100.0 + trades as f64
}

/// Penalty fitness for a chromosome whose trade count falls short of its own
/// **Minimum Track Record Length** (MinTRL — Bailey & López de Prado, 2014):
/// the minimum number of observations needed to reject H0 (true Sharpe <= 0)
/// at 95% confidence, computed from the Sharpe ratio and standard error THIS
/// specific chromosome actually realized.
///
/// This is a DIFFERENT, self-referential concern from
/// [`MIN_TRADES_FOR_SIGNIFICANCE`]: that is a fixed, one-size-fits-all floor
/// (30 trades for everyone); MinTRL is computed per-candidate, so a
/// chromosome with a huge but noisy Sharpe needs MORE trades to be believed
/// than one with a modest, stable Sharpe over the same sample. A candidate
/// can clear the fixed 30-trade floor and still have an unproven Sharpe by
/// its own statistics -- exactly the gap this gate closes.
///
/// `min_trl` is the candidate's own `min_track_record_length` (from
/// `backtest::statistical_significance::compute_sharpe_significance`,
/// computed by the caller from its per-trade returns since `genetic` cannot
/// depend on `backtest` — see this crate's Cargo.toml). Returns:
/// - `None` when MinTRL isn't computable for this candidate (e.g. Sharpe at
///   or below the benchmark, or zero standard error) — callers should fall
///   through to their other gates rather than penalizing on a metric that
///   doesn't apply.
/// - `Some(penalty)` when `trades < min_trl`, scaled so the score improves
///   as `trades` approaches `min_trl` (useful for GA diagnostics), while
///   staying strictly ABOVE [`low_trade_count_fitness`]'s worst case (this
///   candidate already cleared the basic significance floor) and strictly
///   BELOW any realistically-achievable weighted fitness (~-3..+3), so a GA
///   never prefers an unproven Sharpe over a proven one.
pub fn min_track_record_gate_fitness(trades: usize, min_trl: Option<usize>) -> Option<f64> {
    match min_trl {
        Some(trl) if trl > 0 && trades < trl => {
            let progress = (trades as f64 / trl as f64).clamp(0.0, 1.0);
            Some(-50.0 + progress * 40.0) // range: (-50.0, -10.0]
        }
        _ => None,
    }
}

/// Hard disqualification threshold for GA fitness: a candidate whose tunable
/// parameter count, divided by its trade count, exceeds this ratio gets
/// fitness 0.0 regardless of Sharpe -- same "soft-disqualify via the fitness
/// floor" mechanism as the drawdown hard-gate applied alongside it. 0.1 means
/// "require at least ~10 trades per free parameter," the rough
/// observations-per-parameter heuristic used to guard against a GA finding a
/// few-trade, many-parameter fit that looks great in-sample purely because
/// there was nothing left to disconfirm it. This does NOT replace Deflated
/// Sharpe Ratio's own trial-count correction (computed post-hoc elsewhere) --
/// DSR corrects for how much was searched across the whole GA run; this gate
/// corrects for how little data backs the ONE candidate finally selected.
///
/// Lives in `genetic` (rather than `backtest`) so both the in-process fitness
/// closures (`backtest::python_validation`) and the process-pool worker
/// (`program::ga_eval_worker`, the path actually authoritative whenever a
/// `WorkerPool` is spawned -- see quant council 2026-07-27 follow-up) apply
/// the identical gate instead of drifting apart.
pub const MAX_PARAMS_PER_TRADE: f64 = 0.1;

/// True when `param_count` tunable parameters is too many relative to how
/// many `trades` a candidate actually produced -- see [`MAX_PARAMS_PER_TRADE`]'s
/// doc comment.
pub fn exceeds_param_complexity_gate(param_count: usize, trades: usize) -> bool {
    (param_count as f64 / (trades as f64).max(1.0)) > MAX_PARAMS_PER_TRADE
}

/// Fitness value for a candidate disqualified by [`exceeds_param_complexity_gate`].
/// Ordered strictly between [`low_trade_count_fitness`]'s worst case (~-99) and
/// [`min_track_record_gate_fitness`]'s range ((-50.0, -10.0]) -- a candidate with
/// too many free parameters for its trade count is a worse sign than one that
/// merely traded a little, but its problem is different (and less severe) than
/// one that cleared the fixed/self-referential trade-count floors and still has
/// an unproven Sharpe.
pub const PARAM_COMPLEXITY_GATE_FITNESS: f64 = -60.0;

/// Fitness value for a candidate disqualified by a drawdown hard cap (a
/// candidate whose max drawdown exceeded the job's configured tolerance).
/// Ordered strictly between [`min_track_record_gate_fitness`]'s range and any
/// realistically-achievable weighted fitness (~-3..+3 for the current weight
/// profiles) -- excessive drawdown is checked LAST among the hard gates
/// (liquidation/zero-trade/low-trade-count/complexity/MinTRL all fire first
/// when applicable) since it's evaluated only once a candidate's other metrics
/// are already trustworthy enough to say its risk profile, not its sample
/// size, is what disqualifies it.
pub const DRAWDOWN_HARD_CAP_FITNESS: f64 = -5.0;

/// Sign-aware soft penalty factor for trading less often than expected given
/// the backtest's calendar span, using the strategy-profile-specific
/// `min_trades_per_day` weight (0.5-10.0/day depending on preset) rather than
/// a one-size-fits-all constant. This is a DIFFERENT concern from
/// [`MIN_TRADES_FOR_SIGNIFICANCE`]: that answers "can I trust this metric's
/// point estimate" (a function of sample size alone); this answers "has this
/// strategy been exercised across enough of the calendar span to trust it
/// isn't a lucky fit to one narrow regime" (a function of activity density
/// over time) -- see quant council 2026-07-27.
///
/// Returns a factor in `[0.6, 1.0]`. Callers MUST apply it sign-aware:
/// multiply a non-negative fitness, but DIVIDE a negative one -- multiplying
/// a negative fitness by a fraction < 1 moves it toward zero, i.e. rewards
/// under-trading, the same class of bug fixed in `diversity::apply_fitness_sharing`.
pub fn trade_activity_penalty_factor(trades: usize, min_trades_per_day: f64, data_span_days: f64) -> f64 {
    let preferred = (min_trades_per_day * data_span_days).max(1.0);
    if (trades as f64) < preferred {
        (trades as f64 / preferred).max(0.6)
    } else {
        1.0
    }
}

/// Apply [`trade_activity_penalty_factor`] to `fitness` in a sign-aware way.
pub fn apply_trade_activity_penalty(fitness: f64, trades: usize, min_trades_per_day: f64, data_span_days: f64) -> f64 {
    apply_sign_aware_penalty_factor(fitness, trade_activity_penalty_factor(trades, min_trades_per_day, data_span_days))
}

/// Shared sign-aware penalty application: `factor` is a fraction in `(0.0,
/// 1.0]` meaning "keep this much of the fitness" (1.0 = no penalty). Always
/// use this instead of `fitness *= factor` directly -- on a POSITIVE fitness,
/// multiplying by a fraction < 1.0 correctly shrinks it toward zero (a
/// penalty); on a NEGATIVE fitness (e.g. a low-trade-count chromosome's
/// [`low_trade_count_fitness`]/[`ZERO_TRADE_FITNESS`] sentinel), multiplying
/// by the same fraction moves it TOWARD zero too -- which for a negative
/// number means REWARDING it, the exact opposite of the intended penalty.
/// Dividing a negative fitness by `factor` instead correctly pushes it
/// further from zero (more negative), consistent with a positive fitness
/// being pushed toward zero -- both moves make the candidate rank worse.
/// This is the general form of the fix already applied in
/// `diversity::apply_fitness_sharing` (job 64e39acf: a growing zero-trade
/// niche won a GA run because fitness sharing divided its already-negative
/// fitness toward zero via this exact multiplication mistake).
///
/// `factor <= 0.0` is treated as `factor.max(0.01)` to avoid a division blow-up;
/// no caller in this crate currently produces a non-positive factor, but a
/// future one might, and NaN/inf propagating through the rest of the
/// generation's fitness/sharing/sorting arithmetic is worse than a clamp.
pub fn apply_sign_aware_penalty_factor(fitness: f64, factor: f64) -> f64 {
    if factor >= 1.0 {
        return fitness;
    }
    let factor = factor.max(0.01);
    if fitness >= 0.0 {
        fitness * factor
    } else {
        fitness / factor
    }
}

/// Fitness result containing both the raw fitness and equity curve for robustness analysis
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FitnessResult {
    pub fitness: f64,
    pub equity_curve: Vec<f64>,
    /// Additional backtest metrics for reporting
    pub net_pnl: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub num_trades: i32,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    /// Total volume traded in USD
    pub total_volume: f64,
    /// Total fees paid in USD
    pub total_fees: f64,
    /// Best trade P&L in USD
    pub best_trade_pnl: f64,
    /// Worst trade P&L in USD
    pub worst_trade_pnl: f64,
    /// Average trade P&L in USD
    pub avg_trade_pnl: f64,
    /// Average time in trade in milliseconds
    pub avg_time_in_trade_ms: f64,
    /// Average inventory over backtest
    pub avg_inventory: f64,
    /// Maximum inventory held
    pub max_inventory: f64,
    /// Number of inventory zero crossings
    pub zero_crossings: i32,
    /// Average fill rate (0.0-1.0)
    pub avg_fill_rate: f64,
    /// Average queue position
    pub avg_queue_position: f64,
    /// Partial fill percentage (0.0-100.0)
    pub partial_fill_pct: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Fraction of trades whose fill size was capped by available bar volume
    /// rather than filling in full (0.0-1.0).
    pub volume_constrained_pct: f64,

    // === Diversity Metrics (strategy uniqueness) ===
    /// Crowding penalty applied (0.0-0.9) - higher means more similar strategies exist
    pub crowding_penalty: f64,
    /// Novelty bonus from being different from existing strategies
    pub novelty_bonus: f64,
    /// Overall uniqueness score (0.0-1.0) - higher is more unique
    pub uniqueness_score: f64,
    /// Count of similar strategies in registry
    pub similar_strategies_count: i32,

    /// True when `fitness` was set by a hard-disqualification gate (liquidation,
    /// zero/low trade count, param-complexity, MinTRL, drawdown cap) rather than
    /// the weighted-sum formula. Population-level post-processing (z-score
    /// normalization, fitness sharing, crowding, complexity penalty) MUST treat
    /// this candidate's fitness as fixed and skip re-deriving it from metrics --
    /// see `apply_zscore_normalization`'s use of this flag for why.
    #[serde(default)]
    pub hard_gated: bool,
}

impl FitnessResult {
    pub fn new(fitness: f64, equity_curve: Vec<f64>) -> Self {
        Self { 
            fitness, 
            equity_curve,
            net_pnl: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            num_trades: 0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_volume: 0.0,
            total_fees: 0.0,
            best_trade_pnl: 0.0,
            worst_trade_pnl: 0.0,
            avg_trade_pnl: 0.0,
            avg_time_in_trade_ms: 0.0,
            avg_inventory: 0.0,
            max_inventory: 0.0,
            zero_crossings: 0,
            avg_fill_rate: 0.0,
            avg_queue_position: 0.0,
            partial_fill_pct: 0.0,
            avg_latency_ms: 0.0,
            volume_constrained_pct: 0.0,
            // Diversity metrics
            crowding_penalty: 0.0,
            novelty_bonus: 0.0,
            uniqueness_score: 1.0, // Default to fully unique
            similar_strategies_count: 0,
            hard_gated: false,
        }
    }

    /// Create a FitnessResult with full metrics from a backtest
    pub fn with_metrics(
        fitness: f64,
        equity_curve: Vec<f64>,
        net_pnl: f64,
        sharpe_ratio: f64,
        num_trades: i32,
        max_drawdown: f64,
        win_rate: f64,
        profit_factor: f64,
    ) -> Self {
        Self {
            fitness,
            equity_curve,
            net_pnl,
            sharpe_ratio,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            num_trades,
            max_drawdown,
            win_rate,
            profit_factor,
            total_volume: 0.0,
            total_fees: 0.0,
            best_trade_pnl: 0.0,
            worst_trade_pnl: 0.0,
            avg_trade_pnl: 0.0,
            avg_time_in_trade_ms: 0.0,
            avg_inventory: 0.0,
            max_inventory: 0.0,
            zero_crossings: 0,
            avg_fill_rate: 0.0,
            avg_queue_position: 0.0,
            partial_fill_pct: 0.0,
            avg_latency_ms: 0.0,
            volume_constrained_pct: 0.0,
            // Diversity metrics
            crowding_penalty: 0.0,
            novelty_bonus: 0.0,
            uniqueness_score: 1.0,
            similar_strategies_count: 0,
            hard_gated: false,
        }
    }

    /// Create a FitnessResult with ALL metrics from a realistic simulation.
    /// `sortino_ratio`/`calmar_ratio` were added late (both were previously
    /// silently hardcoded to 0.0 here despite the "ALL metrics" doc comment --
    /// found while unifying the platform's GA fitness functions) -- this has
    /// zero external callers today, so the signature change is free.
    #[allow(clippy::too_many_arguments)]
    pub fn with_full_metrics(
        fitness: f64,
        equity_curve: Vec<f64>,
        net_pnl: f64,
        sharpe_ratio: f64,
        sortino_ratio: f64,
        calmar_ratio: f64,
        num_trades: i32,
        max_drawdown: f64,
        win_rate: f64,
        profit_factor: f64,
        total_volume: f64,
        total_fees: f64,
        best_trade_pnl: f64,
        worst_trade_pnl: f64,
        avg_trade_pnl: f64,
        avg_time_in_trade_ms: f64,
        avg_inventory: f64,
        max_inventory: f64,
        zero_crossings: i32,
        avg_fill_rate: f64,
        avg_queue_position: f64,
        partial_fill_pct: f64,
        avg_latency_ms: f64,
        volume_constrained_pct: f64,
        hard_gated: bool,
    ) -> Self {
        Self {
            fitness,
            equity_curve,
            net_pnl,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            num_trades,
            max_drawdown,
            win_rate,
            profit_factor,
            total_volume,
            total_fees,
            best_trade_pnl,
            worst_trade_pnl,
            avg_trade_pnl,
            avg_time_in_trade_ms,
            avg_inventory,
            max_inventory,
            zero_crossings,
            avg_fill_rate,
            avg_queue_position,
            partial_fill_pct,
            avg_latency_ms,
            volume_constrained_pct,
            // Diversity metrics
            crowding_penalty: 0.0,
            novelty_bonus: 0.0,
            uniqueness_score: 1.0,
            similar_strategies_count: 0,
            hard_gated,
        }
    }

    /// Set diversity metrics on an existing FitnessResult
    pub fn with_diversity(
        mut self,
        crowding_penalty: f64,
        novelty_bonus: f64,
        uniqueness_score: f64,
        similar_strategies_count: i32,
    ) -> Self {
        self.crowding_penalty = crowding_penalty;
        self.novelty_bonus = novelty_bonus;
        self.uniqueness_score = uniqueness_score;
        self.similar_strategies_count = similar_strategies_count;
        self
    }
    
    pub fn failure() -> Self {
        Self::default()
    }

    /// Look up a named objective value for multi-objective optimisation.
    pub fn get_objective(&self, name: &str) -> f64 {
        match name {
            "sharpe" | "sharpe_ratio" => self.sharpe_ratio,
            "sortino" | "sortino_ratio" => self.sortino_ratio,
            "calmar" | "calmar_ratio" => self.calmar_ratio,
            "net_pnl" | "pnl" => self.net_pnl,
            "profit_factor" => self.profit_factor,
            "win_rate" => self.win_rate,
            "max_drawdown" | "drawdown" => self.max_drawdown,
            "num_trades" | "trades" => self.num_trades as f64,
            "fill_rate" | "avg_fill_rate" => self.avg_fill_rate,
            "fitness" => self.fitness,
            _ => 0.0,
        }
    }
}

/// Context passed to fitness function for adaptive optimization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FitnessContext {
    /// Current generation (0-indexed)
    pub generation: usize,
    /// Total number of generations
    pub total_generations: usize,
    /// Sample rate for this generation (1 = every tick, N = every Nth tick)
    pub sample_rate: usize,
    /// Whether this is full resolution (final generations)
    pub is_full_resolution: bool,
}

impl FitnessContext {
    /// Create a new context for a given generation
    pub fn new(generation: usize, total_generations: usize, config: &AdaptiveSamplingConfig) -> Self {
        let sample_rate = config.sample_rate_for_generation(generation, total_generations);
        let is_full_resolution = sample_rate == 1;
        Self {
            generation,
            total_generations,
            sample_rate,
            is_full_resolution,
        }
    }
    
    /// Create a context that always uses full resolution
    pub fn full_resolution(generation: usize, total_generations: usize) -> Self {
        Self {
            generation,
            total_generations,
            sample_rate: 1,
            is_full_resolution: true,
        }
    }
}

/// Async fitness function type alias - returns fitness AND equity curve for MC analysis
pub type AsyncFitnessFn<C> = Arc<dyn Fn(&C) -> std::pin::Pin<Box<dyn std::future::Future<Output = FitnessResult> + Send>> + Send + Sync>;

/// Async fitness function with generation context for adaptive sampling
pub type AsyncContextFitnessFn<C> = Arc<dyn Fn(&C, FitnessContext) -> std::pin::Pin<Box<dyn std::future::Future<Output = FitnessResult> + Send>> + Send + Sync>;

/// Progress callback type for reporting generation progress
pub type ProgressCallback = Arc<dyn Fn(usize, usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// Main genetic algorithm optimizer (async fitness)
pub struct GeneticOptimizer<C: Chromosome> {
    pub config: GeneticConfig,
    pub fitness_fn: AsyncFitnessFn<C>,
    pub progress_callback: Option<ProgressCallback>,
    /// Optional job ID for logging correlation
    pub job_id: Option<String>,
    /// Optional strategy registry for diversity-aware fitness sharing
    pub strategy_registry: Option<Arc<diversity::StrategyRegistry>>,
    /// Optional tenant/user ID used when querying `strategy_registry` for
    /// crowding penalties, so that a tenant's own deployed strategies never
    /// count against themselves (see `StrategyRegistry::calculate_crowding_penalty`).
    pub tenant_id: Option<String>,
}

impl<C: Chromosome + 'static> GeneticOptimizer<C> {
    pub async fn run(&self) -> (C, f64) {
        let job_tag = self.job_id.as_ref().map(|id| format!("[Job {}] ", id)).unwrap_or_default();
        let optimization_start = std::time::Instant::now();
        log_info!("{}Starting genetic optimization: pop={} gen={}", 
            job_tag, self.config.population_size, self.config.generations);
        
        let mut rng = self.config.random_seed
            .map(rand::rngs::StdRng::seed_from_u64)
            .unwrap_or_else(rand::rngs::StdRng::from_entropy);
        
        // Initialize population
        let mut population: Vec<C> = (0..self.config.population_size)
            .map(|_| C::random(&mut rng))
            .collect();
        
        // Evaluate initial best (this can take several minutes with full data)
        let mut best = population[0].clone();
        log_info!("{}Evaluating initial chromosome (this may take several minutes)...", job_tag);
        let initial_result = (self.fitness_fn)(&best).await;
        log_info!("{}Initial chromosome evaluated: fitness={:.6}", job_tag, initial_result.fitness);
        let mut best_fitness = initial_result.fitness;
        let mut best_equity_curve = initial_result.equity_curve;
        let mut generations_without_improvement = 0;
        let convergence_threshold: usize = self.config.convergence_threshold.unwrap_or(10);
        const MAX_RESTARTS: usize = 2; // Allow up to 2 restarts on convergence
        let mut restarts = 0_usize;
        
        let elite_size = (self.config.population_size as f64 * 0.05).ceil() as usize;
        
        // Dynamic concurrent evaluations based on available memory
        // Uses DynamicConcurrency to determine optimal parallelism:
        // - Checks MAX_CONCURRENT_EVALS env var first
        // - Then POD_MEMORY_LIMIT_GB for Kubernetes
        // - Falls back to 6 for standard 32GB pods
        let concurrency_calc = DynamicConcurrency::default();
        let max_concurrent_evals = concurrency_calc.get_concurrent_evals();
        let _semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_evals));
        log_info!("{}Using {} concurrent evaluations (dynamic memory-aware)", job_tag, max_concurrent_evals);
        
        // Configure rayon thread pool based on memory-aware concurrency limit
        // This ensures CPU parallelism while respecting memory constraints
        let rayon_threads = max_concurrent_evals.min(num_cpus::get());
        log_info!("{}Rayon thread pool: {} threads (CPUs: {}, memory-limit: {})", 
            job_tag, rayon_threads, num_cpus::get(), max_concurrent_evals);

        for generation in 0..self.config.generations {
            let generation_start = std::time::Instant::now();
            
            // RAYON PARALLEL FITNESS EVALUATION
            // Uses rayon's work-stealing thread pool for CPU-bound simulation
            // Much more efficient than tokio::spawn for CPU-intensive work
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicUsize, Ordering};
            
            let fitness_fn = Arc::clone(&self.fitness_fn);
            let completed = Arc::new(AtomicUsize::new(0));
            let batch_start = std::time::Instant::now();
            let pop_len = population.len();
            let job_tag_clone = job_tag.clone();
            let gen_num = generation + 1;
            let _total_gens = self.config.generations;
            
            // Choose parallel (rayon) or sequential evaluation.
            // Python strategies must use sequential mode because the GIL serialises
            // all Python execution — rayon threads just fight for the GIL, adding
            // context-switch overhead with zero speedup.
            //
            // IMPORTANT: Sequential mode uses .await (not futures::executor::block_on)
            // because the fitness function may use tokio::sync::Mutex internally.
            // block_on() creates a non-tokio executor that cannot wake tokio primitives,
            // causing deadlocks.
            let fitness_results: Vec<FitnessResult> = if self.config.force_sequential {
                log_info!("{}Gen {}/{}: Sequential fitness evaluation for {} chromosomes (force_sequential=true)",
                    job_tag, generation + 1, self.config.generations, population.len());
                let mut results = Vec::with_capacity(population.len());
                for (idx, chromo) in population.iter().enumerate() {
                    let result = fitness_fn(chromo).await;
                    let done = idx + 1;
                    if done % 10 == 0 || done == pop_len {
                        let elapsed = batch_start.elapsed().as_secs_f64();
                        let rate = done as f64 / elapsed.max(0.001);
                        log_info!("{}Gen {}: {}/{} chromosomes evaluated, {:.1}s elapsed, {:.2}/s",
                            job_tag_clone, gen_num, done, pop_len, elapsed, rate);
                    }
                    results.push(result);
                }
                results
            } else {
                // Parallel evaluation using rayon - optimal for CPU-bound Rust strategies.
                // Uses tokio::runtime::Handle to preserve access to tokio primitives
                // (e.g. tokio::sync::Mutex) that the fitness function may use internally.
                let handle = tokio::runtime::Handle::current();
                population
                    .par_iter()
                    .map(|chromo| {
                        let result = handle.block_on(fitness_fn(chromo));
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 10 == 0 || done == pop_len {
                            let elapsed = batch_start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed.max(0.001);
                            log_info!("{}Gen {}: {}/{} chromosomes evaluated, {:.1}s elapsed, {:.2}/s",
                                job_tag_clone, gen_num, done, pop_len, elapsed, rate);
                        }
                        result
                    })
                    .collect()
            };
            
            let gen_elapsed = generation_start.elapsed().as_secs_f64();
            log_info!("{}Gen {}/{}: Fitness evaluation complete in {:.1}s ({:.2} chromosomes/sec)", 
                job_tag, generation + 1, self.config.generations, gen_elapsed, population.len() as f64 / gen_elapsed);
            
            // Extract raw fitness values
            let mut fitnesses: Vec<f64> = fitness_results.iter().map(|r| r.fitness).collect();
            // Only apply in later generations to speed up initial exploration
            // This validates stability and prevents overfitting of top performers
            if self.config.use_monte_carlo_fitness && generation >= self.config.generations / 2 {
                self.apply_robustness_penalty(&mut fitnesses, &fitness_results);
            }
            
            // Apply fitness sharing if diversity preservation is enabled
            // This penalizes strategies that are too similar to each other within the population
            if self.config.enable_fitness_sharing {
                diversity::apply_fitness_sharing(
                    &mut fitnesses,
                    &population,
                    self.config.sharing_radius,
                );
                log_info!("{}Gen {}: Applied intra-population fitness sharing (radius={:.2})", 
                    job_tag, generation + 1, self.config.sharing_radius);
            }
            
            // Apply crowding penalty from registry if available
            // This penalizes strategies similar to already-deployed strategies
            if self.config.enable_crowding_penalty {
                if let Some(ref registry) = self.strategy_registry {
                    for (i, chromo) in population.iter().enumerate() {
                        let sig = chromo.behavioral_signature();
                        let penalty = registry.calculate_crowding_penalty(
                            &sig,
                            chromo.parameter_hash(),
                            self.tenant_id.as_deref().unwrap_or(""),
                        );
                        // Apply penalty: adjusted_fitness = raw_fitness * (1 - penalty * crowding_weight),
                        // applied sign-aware -- see `apply_sign_aware_penalty_factor`'s
                        // doc comment for why a plain `*=` here would reward, not
                        // penalize, an already-negative fitness (e.g. a low-trade-count
                        // chromosome's heavy penalty).
                        let factor = 1.0 - (penalty * self.config.crowding_weight);
                        fitnesses[i] = apply_sign_aware_penalty_factor(fitnesses[i], factor);
                    }
                    log_info!("{}Gen {}: Applied crowding penalty from {} deployed strategies", 
                        job_tag, generation + 1, registry.len());
                }
            }
            
            // Apply parsimony/complexity penalty: subtract a term proportional
            // to each chromosome's parameter count, biasing selection toward
            // simpler strategies when raw fitness is comparable. Disabled
            // (no-op) when complexity_penalty_weight is 0.0 (the default).
            if self.config.complexity_penalty_weight > 0.0 {
                for (i, chromo) in population.iter().enumerate() {
                    fitnesses[i] -= chromo.complexity() * self.config.complexity_penalty_weight;
                }
            }
            
            // 3. Sort population by fitness for elitism
            let mut indexed_population: Vec<(usize, &C, f64, &FitnessResult)> = population.iter()
                .enumerate()
                .zip(fitnesses.iter())
                .zip(fitness_results.iter())
                .map(|(((idx, c), &fit), result)| (idx, c, fit, result))
                .collect();
            indexed_population.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            // Track best fitness BEFORE updating (for convergence check)
            let previous_best_fitness = best_fitness;
            
            // 4. Preserve elite individuals (top 5%)
            let mut new_population = Vec::with_capacity(population.len());
            for i in 0..elite_size {
                new_population.push(indexed_population[i].1.clone());
                if indexed_population[i].2 > best_fitness {
                    best = indexed_population[i].1.clone();
                    best_fitness = indexed_population[i].2;
                    best_equity_curve = indexed_population[i].3.equity_curve.clone();
                }
            }
            
            // Check for improvement (compare against PREVIOUS best, not current)
            const IMPROVEMENT_THRESHOLD: f64 = 0.001; // 0.1% improvement required
            let relative_improvement = if previous_best_fitness > 0.0 {
                (best_fitness - previous_best_fitness) / previous_best_fitness
            } else {
                best_fitness - previous_best_fitness
            };
            
            if relative_improvement > IMPROVEMENT_THRESHOLD {
                generations_without_improvement = 0;
                log_info!("{}Gen {}: Improvement! fitness {:.6} → {:.6} (+{:.2}%)", 
                    job_tag, generation + 1, previous_best_fitness, best_fitness, relative_improvement * 100.0);
            } else {
                generations_without_improvement += 1;
            }
            
            // Calculate population diversity for monitoring
            let diversity = Self::calculate_diversity(&population);
            
            // Early stopping if converged OR if diversity is too low
            if generations_without_improvement >= convergence_threshold {
                if restarts < MAX_RESTARTS {
                    restarts += 1;
                    log_info!("{}Converged at gen {} — restart {}/{}: re-randomising population (keeping best)",
                        job_tag, generation + 1, restarts, MAX_RESTARTS);
                    // Keep the best chromosome, fill rest with new random individuals
                    population = std::iter::once(best.clone())
                        .chain((1..self.config.population_size).map(|_| C::random(&mut rng)))
                        .collect();
                    generations_without_improvement = 0;
                    continue;
                }
                log_info!("{}Converged after {} generations (no improvement for {} generations, {} restarts exhausted)", 
                    job_tag, generation + 1, convergence_threshold, MAX_RESTARTS);
                break;
            }
            
            // If diversity is very low, inject random individuals to maintain exploration
            const MIN_DIVERSITY_THRESHOLD: f64 = 0.01;
            let diversity_injection_count = if diversity < MIN_DIVERSITY_THRESHOLD {
                (self.config.population_size as f64 * 0.1) as usize // Inject 10% new random individuals
            } else {
                0
            };
            
            // 5. Selection (tournament size 3) for remaining population
            while new_population.len() < population.len() - diversity_injection_count {
                let (a, b, c) = (
                    rng.gen_range(0..population.len()),
                    rng.gen_range(0..population.len()),
                    rng.gen_range(0..population.len()),
                );
                let winner_idx = if fitnesses[a] >= fitnesses[b] && fitnesses[a] >= fitnesses[c] {
                    a
                } else if fitnesses[b] >= fitnesses[c] {
                    b
                } else {
                    c
                };
                new_population.push(population[winner_idx].clone());
            }
            
            // 5b. Inject random individuals if diversity is too low
            for _ in 0..diversity_injection_count {
                new_population.push(C::random(&mut rng));
            }
            
            // 6. Crossover (skip elites)
            for i in (elite_size..new_population.len()).step_by(2) {
                if rng.gen_bool(self.config.crossover_rate) && i + 1 < new_population.len() {
                    let child = new_population[i].crossover(&new_population[i + 1], &mut rng);
                    new_population[i] = child;
                }
            }
            
            // 7. Adaptive mutation (skip elites)
            // Adaptive step sizing: larger steps early, smaller as we converge
            // Also increase mutation if improvement rate is slow
            let improvement_rate = if generation > 0 && previous_best_fitness > 0.0 {
                (best_fitness - previous_best_fitness).abs() / previous_best_fitness.abs()
            } else {
                0.1 // Default high improvement rate at start
            };
            
            // If improvement is slow, increase mutation to explore more
            let improvement_factor = if improvement_rate < 0.01 { 1.5 } else { 1.0 };
            
            let adaptive_mutation_rate = (self.config.mutation_rate * 
                (1.0 - (generation as f64 / self.config.generations as f64))) * improvement_factor;
            
            for i in elite_size..new_population.len() {
                if rng.gen_bool(adaptive_mutation_rate) {
                    new_population[i].mutate(&mut rng, adaptive_mutation_rate);
                }
            }
            
            population = new_population;
            
            // Progress callback - called AFTER generation work is complete
            // Reports completed generations accurately (generation 1 complete = 1/total progress)
            if let Some(ref callback) = self.progress_callback {
                callback(generation + 1, self.config.generations).await;
            }
            
            // Progress logging every 5 generations for better visibility
            if (generation + 1) % 5 == 0 || generation == 0 {
                log_info!("{}Generation {}/{}: Best fitness = {:.4}, Diversity = {:.4}, Mutation rate = {:.4}", 
                    job_tag, generation + 1, self.config.generations, best_fitness, diversity, adaptive_mutation_rate);
            }
        }
        
        // Run final robustness check on the best chromosome
        if self.config.use_monte_carlo_fitness && !best_equity_curve.is_empty() {
            let (mean_sharpe, std_sharpe) = Self::run_lightweight_monte_carlo(&best_equity_curve, self.config.mc_sample_size);
            log_info!("{}Best chromosome robustness: mean_sharpe={:.3}, std_sharpe={:.3}", job_tag, mean_sharpe, std_sharpe);
        }
        
        let total_time = optimization_start.elapsed().as_secs_f64();
        log_info!("{}Genetic optimization complete in {:.1}s | best_fitness={:.4}", job_tag, total_time, best_fitness);
        
        (best, best_fitness)
    }
    
    /// Apply robustness penalty to top performers using Monte Carlo consistency check
    /// 
    /// This prevents overfitting by penalizing parameter sets that got lucky.
    /// Only applies to top N% to save computation.
    fn apply_robustness_penalty(&self, fitnesses: &mut [f64], results: &[FitnessResult]) {
        // Find fitness threshold for top N%
        let mut sorted = fitnesses.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((1.0 - self.config.top_percentile_threshold) * sorted.len() as f64) as usize;
        let fitness_threshold = sorted[threshold_idx.min(sorted.len().saturating_sub(1))];
        
        // Use a smaller sample size for per-generation checks (faster)
        // Final robustness check uses the full mc_sample_size
        let quick_sample_size = 10.min(self.config.mc_sample_size);
        
        // Apply MC consistency check only to top performers
        for (i, result) in results.iter().enumerate() {
            if fitnesses[i] >= fitness_threshold && !result.equity_curve.is_empty() {
                let (mean_sharpe, std_sharpe) = Self::run_lightweight_monte_carlo(
                    &result.equity_curve, 
                    quick_sample_size
                );
                
                // Calculate robustness penalty based on coefficient of variation
                // High std relative to mean = unstable = penalty
                let cv = if mean_sharpe.abs() > 0.001 {
                    std_sharpe / mean_sharpe.abs()
                } else {
                    1.0 // Max penalty if mean is near zero
                };
                
                // Penalty ranges from 0 (perfect consistency) to 0.5 (very unstable)
                // CV of 0.5 or higher = 50% penalty
                let penalty = (cv * 0.5).min(0.5);

                // Apply penalty, sign-aware -- see `apply_sign_aware_penalty_factor`'s
                // doc comment. This branch only reaches already-in-the-top-percentile
                // chromosomes, but in a population where every candidate is
                // disqualified for too few trades (all clustered in the
                // heavily-negative penalty range), "top percentile" can still
                // include them.
                fitnesses[i] = apply_sign_aware_penalty_factor(fitnesses[i], 1.0 - penalty);
            }
        }
    }
    
    /// Run lightweight Monte Carlo by shuffling trade returns
    /// Returns (mean_sharpe, std_sharpe)
    fn run_lightweight_monte_carlo(equity_curve: &[f64], num_runs: usize) -> (f64, f64) {
        use rayon::prelude::*;
        
        if equity_curve.len() < 2 {
            return (0.0, 0.0);
        }
        
        // Extract trade returns from equity curve
        let trade_returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        if trade_returns.is_empty() {
            return (0.0, 0.0);
        }
        
        // Run Monte Carlo in parallel
        let sharpe_ratios: Vec<f64> = (0..num_runs)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut shuffled = trade_returns.clone();
                shuffled.shuffle(&mut rng);
                
                // Calculate Sharpe ratio
                let mean_ret: f64 = shuffled.iter().sum::<f64>() / shuffled.len() as f64;
                let variance: f64 = shuffled.iter()
                    .map(|r| (r - mean_ret).powi(2))
                    .sum::<f64>() / (shuffled.len() - 1).max(1) as f64;
                let std_ret = variance.sqrt();
                
                if std_ret > 0.0 {
                    // Note: Using 252 as a standardization factor for relative comparison within GA.
                    // Absolute Sharpe values may differ from properly annualized metrics in backtest.
                    mean_ret / std_ret * 252.0_f64.sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        
        // Calculate mean and std of Sharpe ratios
        let mean_sharpe: f64 = sharpe_ratios.iter().sum::<f64>() / sharpe_ratios.len() as f64;
        let variance: f64 = sharpe_ratios.iter()
            .map(|s| (s - mean_sharpe).powi(2))
            .sum::<f64>() / (sharpe_ratios.len() - 1).max(1) as f64;
        let std_sharpe = variance.sqrt();
        
        (mean_sharpe, std_sharpe)
    }
    
    /// Calculate population diversity as average pairwise distance
    fn calculate_diversity(population: &[C]) -> f64 {
        if population.len() < 2 {
            return 1.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                total_distance += population[i].distance(&population[j]);
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }
}

/// Adaptive Genetic Algorithm optimizer with generation-aware sampling
/// 
/// Uses coarser data sampling in early generations for faster exploration,
/// then switches to full resolution for precise fine-tuning. This provides
/// 1.3-1.5x speedup with minimal accuracy loss.
/// 
/// Also supports dynamic concurrency based on available memory.
pub struct AdaptiveGeneticOptimizer<C: Chromosome> {
    pub config: GeneticConfig,
    pub fitness_fn: AsyncContextFitnessFn<C>,
    pub progress_callback: Option<ProgressCallback>,
    pub job_id: Option<String>,
    pub sampling_config: AdaptiveSamplingConfig,
    /// Per-generation convergence data, populated during run().
    pub convergence_history: Arc<std::sync::Mutex<Vec<landscape::ConvergencePoint>>>,
    /// Optional population-level fitness post-processing hook, applied once
    /// per generation after evaluation and before fitness sharing/crowding.
    /// Typically used for z-score normalization across metrics; `genetic`
    /// ships no built-in implementation since what gets normalized (and how)
    /// is a scoring-policy decision, same as [`fitness::FitnessFunction`].
    /// Receives only non-`hard_gated` results are NOT filtered out for you --
    /// implementations should skip `r.hard_gated` entries themselves (see
    /// the removed `apply_zscore_normalization` for why: a hard-gated
    /// candidate's fitness is a fixed sentinel, not a real metric to
    /// normalize against).
    pub fitness_normalizer: Option<Arc<dyn Fn(&mut [FitnessResult]) + Send + Sync>>,
    /// Optional worker pool for parallel Python strategy evaluation (GIL bypass).
    /// When present, used instead of sequential/rayon paths for batch evaluation.
    pub worker_pool: Option<worker_pool::WorkerPool>,
    /// Bug #29 — Snapshot of the last evaluated generation's (chromosome, fitness)
    /// pairs, sorted by fitness descending. Populated each iteration; consumers
    /// read after `run()` returns to derive `top_n_params` via
    /// `landscape::extract_top_n`.
    pub final_population: Arc<std::sync::Mutex<Vec<(C, f64)>>>,
    /// Optional strategy registry for diversity-aware fitness sharing against
    /// already-deployed strategies (see `GeneticOptimizer::strategy_registry`
    /// for the same mechanism on the sequential optimizer). `None` (the
    /// default) makes `GeneticConfig::enable_crowding_penalty` a no-op, same
    /// as before this field existed.
    pub strategy_registry: Option<Arc<diversity::StrategyRegistry>>,
    /// Tenant/user ID used when querying `strategy_registry`, so a tenant's
    /// own deployed strategies never count against themselves.
    pub tenant_id: Option<String>,
}

impl<C: Chromosome + 'static> AdaptiveGeneticOptimizer<C> {
    /// Create a new adaptive optimizer with default sampling config
    pub fn new(
        config: GeneticConfig,
        fitness_fn: AsyncContextFitnessFn<C>,
    ) -> Self {
        Self {
            config,
            fitness_fn,
            progress_callback: None,
            job_id: None,
            sampling_config: AdaptiveSamplingConfig::default(),
            convergence_history: Arc::new(std::sync::Mutex::new(Vec::new())),
            fitness_normalizer: None,
            worker_pool: None,
            final_population: Arc::new(std::sync::Mutex::new(Vec::new())),
            strategy_registry: None,
            tenant_id: None,
        }
    }

    /// Run optimization with adaptive sampling
    pub async fn run(&mut self) -> (C, f64) {
        let job_tag = self.job_id.as_ref().map(|id| format!("[Job {}] ", id)).unwrap_or_default();
        let optimization_start = std::time::Instant::now();
        
        log_info!("{}Starting ADAPTIVE genetic optimization: pop={} gen={} sampling={:?}", 
            job_tag, self.config.population_size, self.config.generations,
            if self.sampling_config.enabled { "adaptive" } else { "full" });
        
        let mut rng = self.config.random_seed
            .map(rand::rngs::StdRng::seed_from_u64)
            .unwrap_or_else(rand::rngs::StdRng::from_entropy);
        let mut population: Vec<C> = (0..self.config.population_size)
            .map(|_| C::random(&mut rng))
            .collect();
        
        // Evaluate initial best with full resolution
        let mut best = population[0].clone();
        let initial_ctx = FitnessContext::full_resolution(0, self.config.generations);
        log_info!("{}Evaluating initial chromosome (full resolution)...", job_tag);
        let initial_result = (self.fitness_fn)(&best, initial_ctx).await;
        log_info!("{}Initial chromosome evaluated: fitness={:.6}", job_tag, initial_result.fitness);
        
        let mut best_fitness = initial_result.fitness;
        let mut _best_equity_curve = initial_result.equity_curve;
        let mut generations_without_improvement = 0;
        const MAX_RESTARTS: usize = 2;
        let mut restarts = 0_usize;
        let convergence_threshold: usize = self.config.convergence_threshold.unwrap_or(10);
        
        let elite_size = (self.config.population_size as f64 * 0.05).ceil() as usize;
        
        // Dynamic concurrency
        let concurrency_calc = DynamicConcurrency::default();
        let max_concurrent_evals = concurrency_calc.get_concurrent_evals();
        let _semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_evals));
        let rayon_threads = max_concurrent_evals.min(num_cpus::get());
        log_info!("{}Using {} concurrent evaluations, {} rayon threads", job_tag, max_concurrent_evals, rayon_threads);

        for generation in 0..self.config.generations {
            let generation_start = std::time::Instant::now();
            
            // Get sampling context for this generation
            let sample_rate = self.sampling_config.sample_rate_for_generation(
                generation, 
                self.config.generations
            );
            let is_full = sample_rate == 1;
            
            log_info!("{}Gen {}/{}: {} RAYON fitness evaluation for {} chromosomes (sample_rate={}, {} threads)", 
                job_tag, generation + 1, self.config.generations, 
                if is_full { "FULL" } else { "SAMPLED" },
                population.len(), sample_rate, rayon_threads);
            
            // RAYON PARALLEL FITNESS EVALUATION with context
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicUsize, Ordering};
            
            let fitness_fn = Arc::clone(&self.fitness_fn);
            let completed = Arc::new(AtomicUsize::new(0));
            let batch_start = std::time::Instant::now();
            let pop_len = population.len();
            let job_tag_clone = job_tag.clone();
            let gen_num = generation + 1;
            let sampling_config = self.sampling_config.clone();
            let total_gens = self.config.generations;
            
            let fitness_results: Vec<FitnessResult> = if self.worker_pool.is_some() {
                // Process-pool evaluation — parallel Python via separate processes (GIL bypass).
                // Each worker process has its own Python interpreter, giving true OS-level parallelism.
                use std::any::Any;
                let pool = self.worker_pool.as_mut().unwrap();
                let ctx = FitnessContext::new(generation, total_gens, &sampling_config);

                // Downcast population slice to DynamicChromosome (WorkerPool only supports this type)
                let pop_any: &dyn Any = &population;
                if let Some(dyn_pop) = pop_any.downcast_ref::<Vec<DynamicChromosome>>() {
                    log_info!("{}Gen {}/{}: PARALLEL process-pool evaluation for {} chromosomes ({} workers)",
                        job_tag, generation + 1, self.config.generations, population.len(), pool.num_workers());
                    pool.evaluate_batch(dyn_pop, &ctx, &job_tag)
                } else {
                    // Fallback: worker pool set but chromosome type isn't DynamicChromosome.
                    log_info!("{}Gen {}/{}: Worker pool set but chromosome type mismatch, falling back to sequential",
                        job_tag, generation + 1, self.config.generations);
                    let mut results = Vec::with_capacity(population.len());
                    for (idx, chromo) in population.iter().enumerate() {
                        let ctx = FitnessContext::new(generation, total_gens, &sampling_config);
                        let result = fitness_fn(chromo, ctx).await;
                        let done = idx + 1;
                        if done % 10 == 0 || done == pop_len {
                            let elapsed = batch_start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed.max(0.001);
                            log_info!("{}Gen {}: {}/{} chromosomes evaluated, {:.1}s elapsed, {:.2}/s",
                                job_tag_clone, gen_num, done, pop_len, elapsed, rate);
                        }
                        results.push(result);
                    }
                    results
                }
            } else if self.config.force_sequential {
                // Sequential evaluation — required for Python strategies (GIL serialises
                // all Python execution; rayon threads just fight for the GIL).
                // Uses .await directly since we're already in an async context.
                log_info!("{}Gen {}/{}: Sequential fitness evaluation for {} chromosomes (force_sequential=true)",
                    job_tag, generation + 1, self.config.generations, population.len());
                let mut results = Vec::with_capacity(population.len());
                for (idx, chromo) in population.iter().enumerate() {
                    let ctx = FitnessContext::new(generation, total_gens, &sampling_config);
                    let result = fitness_fn(chromo, ctx).await;
                    let done = idx + 1;
                    if done % 10 == 0 || done == pop_len {
                        let elapsed = batch_start.elapsed().as_secs_f64();
                        let rate = done as f64 / elapsed.max(0.001);
                        log_info!("{}Gen {}: {}/{} chromosomes evaluated, {:.1}s elapsed, {:.2}/s",
                            job_tag_clone, gen_num, done, pop_len, elapsed, rate);
                    }
                    results.push(result);
                }
                results
            } else {
                // Parallel evaluation using rayon - optimal for CPU-bound Rust strategies.
                // Uses tokio::runtime::Handle to preserve access to tokio primitives
                // (e.g. tokio::sync::Mutex) that the fitness function may use internally.
                let handle = tokio::runtime::Handle::current();
                population
                    .par_iter()
                    .map(|chromo| {
                        let ctx = FitnessContext::new(generation, total_gens, &sampling_config);
                        
                        let result = handle.block_on(fitness_fn(chromo, ctx));
                        
                        // Update progress counter
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        
                        // Log progress every 10 completions or at end
                        if done % 10 == 0 || done == pop_len {
                            let elapsed = batch_start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed.max(0.001);
                            log_info!("{}Gen {}: {}/{} chromosomes evaluated, {:.1}s elapsed, {:.2}/s", 
                                job_tag_clone, gen_num, done, pop_len, elapsed, rate);
                        }
                        
                        result
                    })
                    .collect()
            };
            
            let gen_elapsed = generation_start.elapsed().as_secs_f64();
            log_info!("{}Gen {}/{}: Fitness evaluation complete in {:.1}s ({:.2} chromosomes/sec)", 
                job_tag, generation + 1, self.config.generations, gen_elapsed, population.len() as f64 / gen_elapsed);
            
            // Optional population-level fitness post-processing (e.g. z-score
            // normalization) -- see `fitness_normalizer`'s doc comment.
            let mut fitness_results = fitness_results;
            if let Some(ref normalizer) = self.fitness_normalizer {
                if fitness_results.len() >= 2 {
                    normalizer(&mut fitness_results);
                }
            }

            let mut fitnesses: Vec<f64> = fitness_results.iter().map(|r| r.fitness).collect();

            // Apply fitness sharing if diversity preservation is enabled (see
            // GeneticOptimizer::run for rationale). This was previously only
            // implemented on the sequential GeneticOptimizer -- AdaptiveGeneticOptimizer
            // (the optimizer actually used for Python custom-strategy GA runs) silently
            // ignored `enable_fitness_sharing` despite it defaulting to `true`.
            if self.config.enable_fitness_sharing {
                diversity::apply_fitness_sharing(
                    &mut fitnesses,
                    &population,
                    self.config.sharing_radius,
                );
                log_info!("{}Gen {}: Applied intra-population fitness sharing (radius={:.2})",
                    job_tag, generation + 1, self.config.sharing_radius);
            }

            // Apply crowding penalty from registry if available (no-op while
            // `strategy_registry` is `None`, same as GeneticOptimizer).
            if self.config.enable_crowding_penalty {
                if let Some(ref registry) = self.strategy_registry {
                    for (i, chromo) in population.iter().enumerate() {
                        let sig = chromo.behavioral_signature();
                        let penalty = registry.calculate_crowding_penalty(
                            &sig,
                            chromo.parameter_hash(),
                            self.tenant_id.as_deref().unwrap_or(""),
                        );
                        fitnesses[i] *= 1.0 - (penalty * self.config.crowding_weight);
                    }
                    log_info!("{}Gen {}: Applied crowding penalty from {} deployed strategies",
                        job_tag, generation + 1, registry.len());
                }
            }

            // Apply parsimony/complexity penalty (see GeneticOptimizer::run for rationale).
            // Disabled (no-op) when complexity_penalty_weight is 0.0 (the default).
            if self.config.complexity_penalty_weight > 0.0 {
                for (i, chromo) in population.iter().enumerate() {
                    fitnesses[i] -= chromo.complexity() * self.config.complexity_penalty_weight;
                }
            }
            
            // Track convergence
            {
                // Bug #28 \u2014 mean fitness over feasible individuals only. Infeasible
                // chromosomes are scored with sentinel (e.g. f64::NEG_INFINITY or 0.0)
                // values that drag the mean toward ~0 / -\u221E and mask actual GA progress.
                let feasible: Vec<f64> = fitnesses.iter()
                    .copied()
                    .filter(|f| f.is_finite() && *f > f64::MIN / 2.0 && *f > ZERO_TRADE_FITNESS)
                    .collect();
                let mean_fitness = if !feasible.is_empty() {
                    feasible.iter().sum::<f64>() / feasible.len() as f64
                } else {
                    0.0
                };
                let current_best = fitnesses.iter().cloned().fold(best_fitness, f64::max);
                if let Ok(mut history) = self.convergence_history.lock() {
                    history.push(landscape::ConvergencePoint {
                        generation: generation + 1,
                        best_fitness: current_best,
                        mean_fitness,
                    });
                }
            }

            // Sort population by fitness
            let mut indexed_population: Vec<(usize, &C, f64, &FitnessResult)> = population.iter()
                .enumerate()
                .zip(fitnesses.iter())
                .zip(fitness_results.iter())
                .map(|(((idx, c), &fit), result)| (idx, c, fit, result))
                .collect();
            indexed_population.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            // Bug #29 — snapshot the sorted (chromosome, fitness) pairs so that
            // after `run()` returns the caller can derive `top_n_params`. We
            // overwrite every generation so the field reflects the final pop.
            if let Ok(mut snap) = self.final_population.lock() {
                snap.clear();
                snap.extend(indexed_population.iter().map(|(_, c, fit, _)| ((*c).clone(), *fit)));
            }
            
            let previous_best_fitness = best_fitness;
            
            // Preserve elites
            let mut new_population = Vec::with_capacity(population.len());
            for i in 0..elite_size {
                new_population.push(indexed_population[i].1.clone());
                if indexed_population[i].2 > best_fitness {
                    best = indexed_population[i].1.clone();
                    best_fitness = indexed_population[i].2;
                    _best_equity_curve = indexed_population[i].3.equity_curve.clone();
                }
            }
            
            // Check improvement
            const IMPROVEMENT_THRESHOLD: f64 = 0.001;
            let relative_improvement = if previous_best_fitness > 0.0 {
                (best_fitness - previous_best_fitness) / previous_best_fitness
            } else {
                best_fitness - previous_best_fitness
            };
            
            if relative_improvement > IMPROVEMENT_THRESHOLD {
                generations_without_improvement = 0;
                log_info!("{}Gen {}: Improvement! fitness {:.6} → {:.6} (+{:.2}%)", 
                    job_tag, generation + 1, previous_best_fitness, best_fitness, relative_improvement * 100.0);
            } else {
                generations_without_improvement += 1;
            }
            
            // Early stopping
            if generations_without_improvement >= convergence_threshold {
                if restarts < MAX_RESTARTS {
                    restarts += 1;
                    log_info!("{}Converged at gen {} — restart {}/{}: re-randomising population (keeping best)",
                        job_tag, generation + 1, restarts, MAX_RESTARTS);
                    population = std::iter::once(best.clone())
                        .chain((1..self.config.population_size).map(|_| C::random(&mut rng)))
                        .collect();
                    generations_without_improvement = 0;
                    continue;
                }
                log_info!("{}Converged at generation {} (no improvement for {} gens, {} restarts exhausted)", 
                    job_tag, generation + 1, convergence_threshold, MAX_RESTARTS);
                break;
            }
            
            // Generate new population. `random_control` (see GeneticConfig's doc
            // comment) swaps this for i.i.d. random draws -- same evaluation
            // budget, no selection pressure, no crossover, no mutation -- so
            // the rest of this loop (elite carryover, convergence tracking,
            // fitness evaluation) stays identical between the guided and
            // control arms and only the search mechanism itself differs.
            if self.config.random_control {
                while new_population.len() < population.len() {
                    new_population.push(C::random(&mut rng));
                }
            } else {
                // Tournament-3 selection for both parents
                let pop_len = indexed_population.len();
                while new_population.len() < population.len() {
                    let select = |rng: &mut rand::rngs::StdRng| -> usize {
                        let (a, b, c) = (
                            rng.gen_range(0..pop_len),
                            rng.gen_range(0..pop_len),
                            rng.gen_range(0..pop_len),
                        );
                        if indexed_population[a].2 >= indexed_population[b].2
                            && indexed_population[a].2 >= indexed_population[c].2 { a }
                        else if indexed_population[b].2 >= indexed_population[c].2 { b }
                        else { c }
                    };
                    let parent1_idx = select(&mut rng);
                    let parent2_idx = select(&mut rng);

                    let mut child = indexed_population[parent1_idx].1.clone();
                    if rng.gen::<f64>() < self.config.crossover_rate {
                        child = child.crossover(indexed_population[parent2_idx].1, &mut rng);
                    }
                    child.mutate(&mut rng, self.config.mutation_rate);
                    new_population.push(child);
                }
            }
            
            population = new_population;
            
            if let Some(ref callback) = self.progress_callback {
                callback(generation + 1, self.config.generations).await;
            }
        }
        
        let total_elapsed = optimization_start.elapsed().as_secs_f64();
        log_info!("{}Adaptive optimization complete in {:.1}s ({:.1} min): best_fitness={:.6}", 
            job_tag, total_elapsed, total_elapsed / 60.0, best_fitness);
        
        (best, best_fitness)
    }

}

#[cfg(test)]
mod sentinel_fitness_tests {
    use super::*;

    /// Regression test for the statistical-significance gate: trade counts
    /// below MIN_TRADES_FOR_SIGNIFICANCE must always score strictly between
    /// the zero-trade floor and a realistically-achievable weighted fitness,
    /// and more trades (still under the threshold) must never score worse.
    #[test]
    fn low_trade_count_fitness_is_ordered_correctly() {
        assert!(ZERO_TRADE_FITNESS < low_trade_count_fitness(1));
        for t in 1..MIN_TRADES_FOR_SIGNIFICANCE {
            assert!(
                low_trade_count_fitness(t) <= low_trade_count_fitness(t + 1),
                "trades={} should not score better than trades={}", t, t + 1
            );
        }
        // Even the best-case low-trade score must stay far below any
        // realistically-achievable weighted-sum fitness (roughly -3..+3 for
        // the current weight profiles).
        assert!(low_trade_count_fitness(MIN_TRADES_FOR_SIGNIFICANCE - 1) < -3.0);
    }

    /// Regression test for the MinTRL gate: must return `None` (no gate) when
    /// MinTRL isn't computable or already cleared, and a penalty strictly
    /// between the low-trade-count tier and any realistic weighted fitness
    /// otherwise -- with the score improving monotonically as `trades`
    /// approaches `min_trl`.
    #[test]
    fn min_track_record_gate_fitness_is_ordered_correctly() {
        // Not computable / already cleared -> no gate.
        assert_eq!(min_track_record_gate_fitness(50, None), None);
        assert_eq!(min_track_record_gate_fitness(100, Some(100)), None);
        assert_eq!(min_track_record_gate_fitness(150, Some(100)), None);

        // Short of MinTRL -> penalized, and strictly worse than any
        // realistically-achievable weighted fitness.
        let far = min_track_record_gate_fitness(10, Some(100)).expect("should gate");
        let close = min_track_record_gate_fitness(90, Some(100)).expect("should gate");
        assert!(far < -3.0 && close < -3.0);
        // Trading count closer to the bar must score no worse than farther away.
        assert!(close >= far);
        // Must stay strictly above the low-trade-count tier's worst case
        // (this candidate already cleared MIN_TRADES_FOR_SIGNIFICANCE).
        assert!(far > low_trade_count_fitness(MIN_TRADES_FOR_SIGNIFICANCE - 1));
        // Must stay strictly below the zero-trade floor's escape velocity.
        assert!(far > ZERO_TRADE_FITNESS);
    }

    /// Regression test: the date-scaled trade-activity penalty must never
    /// reward under-trading for a strategy that is already losing (negative
    /// fitness) -- same bug class as the fitness-sharing fix in
    /// diversity.rs. A penalized negative fitness must be WORSE (more
    /// negative) than the unpenalized value, never better.
    #[test]
    fn trade_activity_penalty_never_rewards_negative_fitness() {
        let min_trades_per_day = 2.0;
        let data_span_days = 30.0; // preferred = 60 trades
        let under_traded = 10usize; // well below preferred
        let raw_negative = -5.0;
        let penalized = apply_trade_activity_penalty(raw_negative, under_traded, min_trades_per_day, data_span_days);
        assert!(
            penalized < raw_negative,
            "penalized fitness ({penalized}) must be worse than raw ({raw_negative}) for a losing, under-traded strategy"
        );
    }

    /// Regression test: for a winning (non-negative) strategy, the penalty
    /// must still behave as a conventional soft multiplier -- strictly worse
    /// than the unpenalized value when under-traded, unchanged when trade
    /// count meets or exceeds the date-scaled preference.
    #[test]
    fn trade_activity_penalty_shrinks_positive_fitness_when_under_traded() {
        let min_trades_per_day = 2.0;
        let data_span_days = 30.0; // preferred = 60 trades
        let raw_positive = 5.0;
        let penalized = apply_trade_activity_penalty(raw_positive, 10, min_trades_per_day, data_span_days);
        assert!(penalized < raw_positive && penalized > 0.0);

        let not_penalized = apply_trade_activity_penalty(raw_positive, 60, min_trades_per_day, data_span_days);
        assert_eq!(not_penalized, raw_positive);
    }

    #[test]
    fn param_complexity_gate_passes_a_well_sampled_candidate() {
        assert!(!exceeds_param_complexity_gate(3, 100));
    }

    /// Regression tests for the shared `apply_sign_aware_penalty_factor` --
    /// the general form of the bug fixed in `diversity::apply_fitness_sharing`
    /// (job 64e39acf) and now also fixed at the crowding-penalty and
    /// robustness-penalty call sites in `GeneticOptimizer`.
    #[test]
    fn sign_aware_penalty_shrinks_positive_fitness() {
        assert_eq!(apply_sign_aware_penalty_factor(10.0, 0.5), 5.0);
    }

    #[test]
    fn sign_aware_penalty_worsens_negative_fitness_not_rewards_it() {
        // A plain `fitness *= factor` here would compute -10.0 * 0.5 = -5.0,
        // i.e. REWARD the already-penalized (e.g. low-trade-count) candidate
        // by moving it toward zero. Dividing instead must push it further
        // negative.
        let penalized = apply_sign_aware_penalty_factor(-10.0, 0.5);
        assert_eq!(penalized, -20.0);
        assert!(penalized < -10.0, "penalty must make a negative fitness worse, not better");
    }

    #[test]
    fn sign_aware_penalty_is_a_no_op_at_or_above_factor_one() {
        assert_eq!(apply_sign_aware_penalty_factor(-10.0, 1.0), -10.0);
        assert_eq!(apply_sign_aware_penalty_factor(10.0, 1.0), 10.0);
        assert_eq!(apply_sign_aware_penalty_factor(-10.0, 1.5), -10.0);
    }

    #[test]
    fn sign_aware_penalty_clamps_non_positive_factor_instead_of_dividing_by_zero() {
        let result = apply_sign_aware_penalty_factor(-10.0, 0.0);
        assert!(result.is_finite());
        assert!(result < -10.0);
    }

    #[test]
    fn sign_aware_penalty_matches_trade_activity_penalty_for_a_disqualified_candidate() {
        // A candidate that already failed MIN_TRADES_FOR_SIGNIFICANCE and then
        // also gets crowding/robustness-penalized must end up STRICTLY worse
        // than its raw low-trade-count sentinel, never better.
        let raw = low_trade_count_fitness(5); // deep in the disqualified range
        let after_crowding_penalty = apply_sign_aware_penalty_factor(raw, 1.0 - 0.3);
        assert!(after_crowding_penalty < raw);
    }

    #[test]
    fn param_complexity_gate_flags_a_thin_sample() {
        assert!(exceeds_param_complexity_gate(6, 20));
    }

    #[test]
    fn param_complexity_gate_is_exclusive_at_the_exact_threshold() {
        assert!(!exceeds_param_complexity_gate(1, 10));
        assert!(exceeds_param_complexity_gate(101, 1000));
    }

    #[test]
    fn param_complexity_gate_treats_zero_trades_as_one_trade() {
        assert!(exceeds_param_complexity_gate(1, 0));
        assert!(!exceeds_param_complexity_gate(0, 0));
    }

    /// PARAM_COMPLEXITY_GATE_FITNESS must sit strictly between the worst
    /// low-trade-count score and MinTRL's range, so a GA never prefers an
    /// overfit-relative-to-sample-size candidate over one that merely traded
    /// a little, nor confuses it with an unproven-Sharpe MinTRL candidate.
    #[test]
    fn param_complexity_gate_fitness_is_ordered_correctly() {
        assert!(PARAM_COMPLEXITY_GATE_FITNESS > low_trade_count_fitness(MIN_TRADES_FOR_SIGNIFICANCE - 1));
        assert!(PARAM_COMPLEXITY_GATE_FITNESS > ZERO_TRADE_FITNESS);
        let mintrl_worst = min_track_record_gate_fitness(1, Some(1000)).expect("should gate");
        assert!(PARAM_COMPLEXITY_GATE_FITNESS < mintrl_worst);
    }

    /// DRAWDOWN_HARD_CAP_FITNESS must sit strictly between MinTRL's range and
    /// any realistically-achievable weighted fitness, since drawdown is
    /// checked last among the hard gates -- only once trade-count/complexity/
    /// MinTRL have already been cleared.
    #[test]
    fn drawdown_hard_cap_fitness_is_ordered_correctly() {
        let mintrl_best = min_track_record_gate_fitness(99, Some(100)).expect("should gate");
        assert!(DRAWDOWN_HARD_CAP_FITNESS > mintrl_best);
        assert!(DRAWDOWN_HARD_CAP_FITNESS < -3.0, "must stay below any realistic weighted fitness (~-3..+3)");
    }

    // NOTE: the z-score-normalization regression test that used to live here
    // now lives alongside `fitness_normalizer`'s concrete implementation
    // (private -- see `program::proprietary_fitness`), since `genetic` no
    // longer ships one to test.
}
