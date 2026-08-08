//! GA fitness-scoring interface.
//!
//! `genetic` deliberately ships no concrete scoring formula -- "what makes a
//! candidate strategy good" is a platform-specific policy decision (which
//! metrics matter, how they're weighted, what gets hard-disqualified), not
//! something a generic GA/backtesting library should dictate. Implement
//! [`FitnessFunction`] with whatever formula fits your use case and pass it
//! to the optimizer.

/// Raw inputs a fitness function can use to score a candidate. No `Option`
/// for anything gate/scoring-relevant -- callers that don't track a field
/// pass its `Default` (0.0/0).
#[derive(Debug, Clone, Default)]
pub struct FitnessInputs {
    pub equity_curve: Vec<f64>,
    pub num_trades: usize,
    pub total_liquidations: usize,
    pub net_pnl: f64,
    pub initial_capital: f64,
    pub sharpe: f64,
    /// Real Sortino where the caller has one; pass `sharpe` again where it
    /// doesn't (e.g. via `sortino.unwrap_or(sharpe)`).
    pub sortino: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    /// Fraction of `initial_capital`, e.g. `0.15` = 15%. Always >= 0.
    pub max_drawdown_frac: f64,
    /// Dollar (quote-currency) drawdown, reported as-is via the returned
    /// `FitnessResult.max_drawdown`.
    pub max_drawdown_abs: f64,
    pub fill_rate: f64,
    pub total_fees: f64,
    pub param_count: usize,
    /// Minimum Track Record Length (Bailey & López de Prado, 2014) for this
    /// candidate, if computable. `None` when it isn't applicable (e.g.
    /// Sharpe at or below the benchmark, or zero standard error).
    pub min_trl: Option<usize>,
    pub data_span_days: f64,
    /// Hard drawdown-disqualification threshold, as a fraction of initial
    /// capital, sourced from the job/user's own configuration (not a
    /// property of the scoring formula itself).
    pub max_drawdown_hard_cap: f64,
    /// True when the candidate's genome evolves its own leverage. Fitness
    /// functions that reward net profit typically want to zero that
    /// component's weight here to avoid reopening a "GA drifts to max
    /// leverage" risk that a liquidation gate alone doesn't fully close.
    pub leverage_evolved: bool,

    // Cosmetic-only fields, reported via the returned `FitnessResult` but
    // not necessarily read by every fitness function.
    pub total_volume: f64,
    pub best_trade_pnl: f64,
    pub worst_trade_pnl: f64,
    pub avg_trade_pnl: f64,
    pub avg_time_in_trade_ms: f64,
    pub avg_inventory: f64,
    pub max_inventory: f64,
    pub zero_crossings: i32,
    pub avg_queue_position: f64,
    pub partial_fill_pct: f64,
    pub avg_latency_ms: f64,
    pub volume_constrained_pct: f64,
}

/// Scores a candidate strategy's simulated performance. Implementations
/// typically apply hard-disqualification gates (e.g. liquidation, too few
/// trades to trust the metrics, excessive drawdown) before falling back to
/// a weighted combination of the input metrics -- see [`FitnessInputs`] and
/// [`crate::FitnessResult`] for the shapes involved. Must be `Send + Sync`
/// since candidate evaluation runs concurrently across the population.
pub trait FitnessFunction: Send + Sync {
    fn compute(&self, inputs: &FitnessInputs) -> crate::FitnessResult;
}
