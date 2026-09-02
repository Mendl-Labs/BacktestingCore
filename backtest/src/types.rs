//! Types and enums for the backtest module
//!
//! Shared types for simulation events, results, and configuration.

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::borrow::Cow;
use std::sync::Arc;

use derivatives::Greeks;

use data_prep::SimulationTick;
use dataloader::{MarketData, TradeData};

// ---------------------------------------------------------------------------
// MarketDataInput — unified input for simulation loops
// ---------------------------------------------------------------------------

/// Allows simulation loops to accept either pre-materialized `&[MarketData]`
/// or a zero-copy `&[SimulationTick]` slice (from mmap'd SimBinCache).
///
/// The `SimTicks` variant avoids a ~2.5 GiB `Vec<MarketData>` allocation for
/// 31M-tick datasets by converting one tick at a time on the stack.
pub enum MarketDataInput<'a> {
    /// Pre-materialized slice (existing path for CSV, Tardis API, uploaded data).
    Slice(&'a [MarketData]),
    /// Zero-copy mmap'd simulation ticks with shared symbol/exchange metadata.
    SimTicks {
        ticks: &'a [SimulationTick],
        symbol: Arc<str>,
        exchange: Arc<str>,
    },
}

impl<'a> MarketDataInput<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            MarketDataInput::Slice(s) => s.len(),
            MarketDataInput::SimTicks { ticks, .. } => ticks.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the input, yielding `Cow<MarketData>`.
    ///
    /// - `Slice` path: yields `Cow::Borrowed` (zero-cost).
    /// - `SimTicks` path: yields `Cow::Owned` with a stack-local `MarketData::Trade`.
    ///   The Arc<str> clone is just a refcount bump — no heap allocation.
    pub fn iter(&self) -> Box<dyn Iterator<Item = Cow<'_, MarketData>> + Send + '_> {
        match self {
            MarketDataInput::Slice(data) => {
                Box::new(data.iter().map(Cow::Borrowed))
            }
            MarketDataInput::SimTicks { ticks, symbol, exchange } => {
                let sym = symbol.clone();
                let exch = exchange.clone();
                Box::new(ticks.iter().map(move |tick| {
                    Cow::Owned(MarketData::Trade(TradeData {
                        timestamp: chrono::DateTime::from_timestamp_millis(tick.timestamp_ms)
                            .unwrap_or(chrono::DateTime::UNIX_EPOCH),
                        symbol: sym.clone(),
                        exchange: exch.clone(),
                        price: tick.price,
                        quantity: tick.quantity,
                        side: if tick.side == 1 {
                            dataloader::tick::Side::Sell
                        } else {
                            dataloader::tick::Side::Buy
                        },
                        trade_id: 0,
                        value: tick.price * tick.quantity,
                    }))
                }))
            }
        }
    }
}

/// A single price level in an L2 orderbook.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub amount: f64,
}

/// An L2 orderbook snapshot with variable-depth bid/ask levels.
///
/// Used as input to the simulation engine for per-tick orderbook context.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BookSnapshot {
    /// Timestamp in microseconds since Unix epoch
    pub timestamp_us: i64,
    /// Bid levels (best bid at index 0)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (best ask at index 0)
    pub asks: Vec<OrderBookLevel>,
}

/// A round-trip trade record linking entry and exit fills.
///
/// Each record represents a complete position lifecycle: opened at entry,
/// closed at exit. Open positions at end-of-backtest have `exit_*` fields
/// set to `None` with unrealized P&L.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub trade_id: usize,
    /// "long" or "short"
    pub side: String,
    /// Entry signal price (before slippage)
    pub entry_signal_price: f64,
    /// Actual entry fill price (after slippage/market impact)
    pub entry_fill_price: f64,
    /// Exit signal price (before slippage). None if still open.
    pub exit_signal_price: Option<f64>,
    /// Actual exit fill price. None if still open.
    pub exit_fill_price: Option<f64>,
    /// Position size
    pub quantity: f64,
    /// Realized P&L (None if still open)
    pub pnl: Option<f64>,
    /// Return percentage (None if still open)
    pub pnl_pct: Option<f64>,
    /// Total commission (entry + exit)
    pub commission: f64,
    /// Total slippage cost (entry + exit, in quote currency)
    pub slippage_cost: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp (None if still open)
    pub exit_time: Option<DateTime<Utc>>,
    /// Duration in seconds (None if still open)
    pub duration_secs: Option<i64>,
    /// Entry liquidity: "maker" or "taker"
    pub entry_liquidity: String,
    /// Exit liquidity (None if still open)
    pub exit_liquidity: Option<String>,
    /// Signal reason at entry
    pub entry_reason: String,
    /// Signal reason at exit
    pub exit_reason: Option<String>,
    /// Maximum Adverse Excursion: worst unrealized loss during this trade (fraction of entry value)
    pub mae: Option<f64>,
    /// Maximum Favorable Excursion: best unrealized gain during this trade (fraction of entry value)
    pub mfe: Option<f64>,
    /// Cross-exchange strategy support (2026-07-22): per-venue leg detail for
    /// a trade that spans 2+ exchanges at once (e.g. a stat-arb pair opened
    /// simultaneously on two venues). Empty for every ordinary single-venue
    /// trade -- the scalar fields above (`entry_fill_price`, `pnl`, etc.)
    /// remain the aggregate/blended view for backward compatibility with
    /// every existing single-venue consumer of `TradeRecord`. Reserved for a
    /// future simultaneous multi-venue EXECUTION model; the current
    /// multi-venue strategy path (`compute_all_signals_multi_venue`) reads
    /// signals from N venues but still executes each trade against one
    /// primary venue, so `legs` is not yet populated by that path either.
    #[serde(default)]
    pub legs: Vec<TradeLeg>,
}

/// One venue's fill within a multi-venue trade. See `TradeRecord::legs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLeg {
    pub exchange: String,
    pub symbol: String,
    /// "long" or "short"
    pub side: String,
    pub fill_price: f64,
    pub quantity: f64,
    /// "maker" or "taker"
    pub liquidity: String,
    pub commission: f64,
    pub slippage_cost: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum BacktestEvent {
    MarketData {
        timestamp: DateTime<Utc>,
        // ... other fields as needed
    },
    OrderSubmitted {
        order_id: Arc<str>,
        timestamp: DateTime<Utc>,
    },
    OrderFilled {
        order_id: Arc<str>,
        fill_price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    },
    // Add more event types as needed
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_pnl: f64,
    pub num_trades: usize,
    /// Number of closed (completed) trades - these contribute to win_rate and realized P&L
    pub closed_trades: usize,
    /// Number of positions still open at end of backtest - contribute to unrealized P&L
    pub open_positions: usize,
    /// Timestamp of the first trade in the backtest data
    pub first_trade_timestamp: Option<DateTime<Utc>>,
    /// Timestamp of the last trade in the backtest data
    pub last_trade_timestamp: Option<DateTime<Utc>>,
    /// Price at the first trade
    pub first_trade_price: Option<f64>,
    /// Price at the last trade
    pub last_trade_price: Option<f64>,
    /// Realized P&L from closed trades only
    pub realized_pnl: Option<f64>,
    /// Unrealized P&L from open positions (mark-to-market)
    pub unrealized_pnl: Option<f64>,
    pub max_drawdown: f64,
    pub sharpe_ratio: Option<f64>,
    pub sortino_ratio: Option<f64>,
    pub calmar_ratio: Option<f64>,
    pub profit_factor: Option<f64>,
    pub win_rate: Option<f64>,
    pub avg_trade_return: Option<f64>,
    pub median_trade_return: Option<f64>,
    pub avg_trade_duration: Option<f64>,
    pub volatility: Option<f64>,
    pub gross_profit: Option<f64>,
    pub gross_loss: Option<f64>,
    pub net_profit: Option<f64>,
    pub equity_curve: Vec<f64>,
    /// Bug #33 — Unix millisecond timestamps parallel to `equity_curve` so the
    /// frontend can plot equity over real dates instead of synthetic indices.
    /// Same length as `equity_curve` when populated; empty if not tracked.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub equity_curve_timestamps: Vec<i64>,
    /// Individual trade P&L in dollars per closed trade — used by Monte Carlo equity simulation
    /// (equity += ret). Do NOT feed to Sharpe or bootstrap functions; use trade_pct_returns.
    #[serde(default)]
    pub trade_returns: Vec<f64>,
    /// Fractional percentage returns per closed trade (pnl_pct = net_pnl / cost_basis).
    /// Used by statistical significance tests and bootstrap confidence intervals.
    /// Populated for Python-simulated strategies; empty for built-in strategies (fallback to trade_returns).
    #[serde(default)]
    pub trade_pct_returns: Vec<f64>,
    /// Per-fill max adverse excursion in dollars (parallel to trade_returns).
    /// Used by Monte Carlo to account for intra-trade drawdowns.
    #[serde(default)]
    pub trade_maes: Vec<f64>,
    pub initial_capital: f64,
    pub data_quality: Option<DataQualityReport>,
    pub transaction_costs: Option<TransactionCostAnalysis>,
    pub market_impact: Option<MarketImpactAnalysis>,
    /// Fraction of trades whose requested fill size was capped by available
    /// bar volume (`fill_volume_fraction`) rather than filling in full. A
    /// nonzero value means the backtest already hit a liquidity ceiling at
    /// its configured position size — a cheap, directional signal for
    /// whether a strategy's edge would survive being deployed at larger
    /// scale. `None` when not tracked by the simulation path used.
    pub volume_constrained_pct: Option<f64>,
    pub inventory_metrics: Option<InventoryMetrics>,
    pub execution_metrics: Option<ExecutionMetrics>,
    pub risk_halt_reason: Option<String>,
    /// Sharpe t-statistic (significance testing)
    pub sharpe_t_stat: Option<f64>,
    /// One-sided p-value for Sharpe significance
    pub significance_pvalue: Option<f64>,
    /// Deflated Sharpe Ratio (Bailey et al., 2014)
    pub deflated_sharpe: Option<f64>,
    /// Ulcer Index — RMS of drawdowns
    pub ulcer_index: Option<f64>,
    /// Pain Index — mean of drawdowns
    pub pain_index: Option<f64>,
    /// Conditional Drawdown-at-Risk at 95% confidence
    pub cdar_95: Option<f64>,
    /// Individual trade records for the trade log
    #[serde(default)]
    pub trade_log: Vec<TradeRecord>,
    /// Per-tick price series for buy-and-hold benchmark computation.
    /// Cleared before serialization (not included in API response).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub price_series: Vec<f64>,
    /// Counts of BUY/SELL/HOLD/CLOSE signals emitted by compute_signals().
    /// None when the per-tick fallback path was used instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_counts: Option<SignalCounts>,
    /// Error message from compute_signals() if it threw a Python exception.
    /// Present only when the vectorized path failed and fell back to per-tick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_signals_error: Option<String>,
    /// Candle bar resolution used when loading data (e.g. 1 = 1-minute bars).
    /// None for tick/trade data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle_interval_minutes: Option<i64>,
    /// Python source code submitted with this job (for debugging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_source_code: Option<String>,
    /// Display name declared by the strategy itself (Python `name()` for custom
    /// strategies). Used by persistence to label the result correctly instead of
    /// falling back to the generic strategy_type ("Custom") or a stale params name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy_display_name: Option<String>,
    /// LP (liquidity provision) backtest metrics — present when running an LP strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_metrics: Option<LpBacktestResult>,
    /// Analysis mode this job ran in: "development", "validation", or "production".
    /// Development mode suppresses the Promising verdict on the frontend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_mode: Option<String>,
    /// Total number of parameter configurations evaluated by the GA (population × generations).
    /// Used to compute the Deflated Sharpe Ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ga_trial_count: Option<u32>,
    /// Mean in-sample Sharpe across all GA trial evaluations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ga_is_sharpe_mean: Option<f64>,
    /// Standard deviation of in-sample Sharpe across all GA trial evaluations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ga_is_sharpe_std: Option<f64>,
    /// Fraction of GA trials that produced IS Sharpe > 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ga_is_sharpe_pct_above_threshold: Option<f64>,
    /// Which analysis stages actually ran for this job.
    /// Ordered: "in_sample", "monte_carlo", "walk_forward" (only those that executed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub analysis_stages_run: Vec<String>,
    /// Per-feature diagnostics (information coefficient + quantile analysis)
    /// when the strategy defines `compute_features()` and
    /// `ENABLE_COMPUTE_FEATURES` is enabled. Diagnostic context only — never
    /// affects trading signals or the pass/fail of the backtest itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_diagnostics: Option<Vec<FeatureDiagnostic>>,
    /// Contract identity + final Greeks snapshot -- present only when this
    /// run traded an option instrument or spread (`PythonSimConfig.
    /// option_instrument`/`option_spread`). `None` for every non-option run,
    /// unchanged from before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_summary: Option<OptionResultSummary>,
}

/// Contract identity + final Greeks snapshot for an option-instrument or
/// option-spread backtest. `contracts` has one entry for a single-leg run,
/// two for a vertical spread (see `OptionSpreadLeg`'s own scope note).
/// `final_greeks` is the portfolio's aggregated Greeks as of the last tick
/// `update_position_greeks` actually ran -- `None` when
/// `PythonSimConfig.underlying_series` wasn't supplied for this run (Greeks
/// were never computed at all in that case), or once every option position
/// has closed (nothing left to aggregate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionResultSummary {
    pub contracts: Vec<OptionContractInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_greeks: Option<Greeks>,
}

/// One option contract's static identity -- strike/expiry/type/ratio the
/// frontend can't derive from the plain trade P&L numbers alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContractInfo {
    /// Exchange contract ticker, e.g. "O:AAPL260501C00130000".
    pub symbol: String,
    pub underlying: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strike: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry: Option<DateTime<Utc>>,
    /// "call" | "put" | "other" (option_instrument/option_spread are only
    /// ever populated with actual options today, but this stays honest
    /// about what InstrumentKind actually said rather than assuming).
    pub contract_type: String,
    pub contract_multiplier: f64,
    /// Signed ratio for this leg (+1 = long, -1 = short). Always 1 for a
    /// single-leg run; the leg's own `OptionSpreadLeg.ratio` for a spread.
    pub ratio: i32,
}

/// IC + quantile analysis for one strategy-declared feature, evaluated
/// against forward returns at the strategy's own holding-period horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDiagnostic {
    pub feature_name: String,
    pub ic: quant_diagnostics::IcResult,
    pub quantile_buckets: Vec<quant_diagnostics::QuantileBucket>,
}

/// Results specific to a liquidity provision backtest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LpBacktestResult {
    pub total_fees_earned_usd: f64,
    pub total_impermanent_loss_usd: f64,
    pub total_gas_costs_usd: f64,
    pub net_pnl_usd: f64,
    pub annualized_apr: f64,
    pub time_in_range_pct: f64,
    pub total_swaps_processed: u64,
    pub fee_earning_swaps: u64,
    pub avg_position_size_usd: f64,
    pub max_drawdown_pct: f64,
    pub equity_curve: Vec<(i64, f64)>,
    pub hodl_comparison: f64,
    pub rebalance_count: u32,
    pub positions_opened: u32,
    pub positions_closed: u32,
}

/// Counts of each signal type produced by the vectorized `compute_signals()` call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalCounts {
    pub buy: usize,
    pub sell: usize,
    pub hold: usize,
    pub close: usize,
}

/// Comprehensive data quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityReport {
    pub overall_score: f64,
    pub total_data_points: usize,
    pub time_span_seconds: i64,
    pub gaps_detected: Vec<DataGap>,
    pub average_gap_seconds: f64,
    pub max_gap_seconds: i64,
    pub gap_ratio: f64, // percentage of time with gaps
    pub outliers_detected: usize,
    pub data_completeness_score: f64,
    pub timestamp_consistency_score: f64,
    pub order_quality: Option<OrderStreamQuality>,
}

/// Information about a detected data gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGap {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_seconds: i64,
    pub severity: GapSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GapSeverity {
    Minor,      // < 2x expected frequency
    Moderate,   // 2-5x expected frequency
    Severe,     // 5-10x expected frequency
    Critical,   // > 10x expected frequency
}

/// Transaction cost breakdown and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCostAnalysis {
    pub total_commission: f64,
    pub total_slippage: f64,
    pub total_market_impact: f64,
    pub total_transaction_costs: f64,
    pub costs_as_percentage_of_pnl: f64,
    pub avg_cost_per_trade: f64,
    pub commission_by_exchange: std::collections::HashMap<String, f64>,
    /// Total gas costs in USD (only populated for DEX backtests)
    #[serde(default)]
    pub total_gas_costs_usd: f64,
    /// Average DEX slippage in basis points (only populated for DEX backtests)
    #[serde(default)]
    pub avg_dex_slippage_bps: f64,
}

/// Order lifecycle and execution quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total orders placed (quotes submitted)
    pub orders_placed: usize,
    /// Orders that received at least one fill
    pub orders_filled: usize,
    /// Orders cancelled before any fill
    pub orders_cancelled: usize,
    /// Total individual fills received (may exceed orders if partial fills)
    pub total_fills: usize,
    /// Fill rate: orders_filled / orders_placed
    pub fill_rate: f64,
    /// Cancellation rate: orders_cancelled / orders_placed
    pub cancellation_rate: f64,
    /// Average fills per filled order (1.0 = no partial fills)
    pub avg_fills_per_order: f64,
    /// Average time from order placement to first fill (seconds)
    pub avg_time_to_fill_seconds: f64,
    /// Median time to fill (seconds)
    pub median_time_to_fill_seconds: f64,
    /// Bid orders placed
    pub bid_orders: usize,
    /// Ask orders placed
    pub ask_orders: usize,
    /// Bid orders filled
    pub bid_fills: usize,
    /// Ask orders filled
    pub ask_fills: usize,
    /// Bid fill rate
    pub bid_fill_rate: f64,
    /// Ask fill rate
    pub ask_fill_rate: f64,
    /// Average quote spread at time of placement (bps)
    pub avg_quote_spread_bps: f64,
    
    // Market-Making Specific Win Rate Metrics
    /// Round-trip win rate: profitable_round_trips / total_round_trips
    pub round_trip_win_rate: Option<f64>,
    /// Total round-trips completed (buy+sell pairs)
    pub total_round_trips: usize,
    /// Profitable round-trips (spread captured > fees)
    pub profitable_round_trips: usize,
    /// Spread capture rate: actual_spread / theoretical_spread
    pub spread_capture_rate: Option<f64>,
    /// Adverse selection rate: % of fills where price moved against us
    pub adverse_selection_rate: Option<f64>,
    /// Average P&L per round-trip in quote currency
    pub avg_pnl_per_round_trip: Option<f64>,
}

/// Inventory management metrics for market making strategies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InventoryMetrics {
    /// Total volume traded (buy + sell) in quote currency (USD)
    pub total_volume_traded: f64,
    /// Average inventory value over the backtest period
    pub avg_inventory_value: f64,
    /// Inventory turnover per day (total_volume / avg_inventory / days)
    pub turnover_per_day: f64,
    /// Number of buy fills
    pub buy_fills: usize,
    /// Number of sell fills
    pub sell_fills: usize,
    /// Total buy volume in quote currency
    pub buy_volume: f64,
    /// Total sell volume in quote currency
    pub sell_volume: f64,
    /// Inventory imbalance ratio (buy_volume - sell_volume) / total_volume
    pub imbalance_ratio: f64,
    
    // === Starting/Ending Position Tracking ===
    /// Starting inventory quantity (base asset units)
    #[serde(default)]
    pub starting_inventory: f64,
    /// Starting inventory value in USD
    #[serde(default)]
    pub starting_inventory_value: f64,
    /// Starting mark price used for valuation
    #[serde(default)]
    pub starting_mark_price: f64,
    /// Ending inventory quantity (base asset units)
    #[serde(default)]
    pub ending_inventory: f64,
    /// Ending inventory value in USD
    #[serde(default)]
    pub ending_inventory_value: f64,
    /// Ending mark price used for valuation
    #[serde(default)]
    pub ending_mark_price: f64,
    /// Starting capital (USD)
    #[serde(default)]
    pub starting_capital: f64,
    /// Ending capital (USD cash after all trades)
    #[serde(default)]
    pub ending_capital: f64,
    /// Final equity (ending_capital + ending_inventory_value)
    #[serde(default)]
    pub final_equity: f64,
}

/// Market impact modeling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactAnalysis {
    pub trades_with_impact: usize,
    pub avg_impact_bps: f64,
    pub max_impact_bps: f64,
    pub total_impact_cost: f64,
    pub liquidity_warnings: Vec<LiquidityWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityWarning {
    pub timestamp: DateTime<Utc>,
    pub order_size: f64,
    pub available_liquidity: f64,
    pub liquidity_ratio: f64, // order_size / available_liquidity
    pub message: String,
}

/// Order stream quality analysis (for market making strategies)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderStreamQuality {
    pub total_orders: usize,
    pub order_gaps: Vec<OrderGap>,
    pub unpaired_quotes: Vec<UnpairedQuote>,
    pub expected_quote_interval_seconds: u64,
    pub avg_quote_interval_seconds: f64,
    pub quote_consistency_score: f64,
    pub missing_quote_cycles: u64,
    pub bid_ask_pair_ratio: f64, // Should be close to 1.0 for market making
}

/// Detected gap in order/quote stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderGap {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_seconds: i64,
    pub missing_quote_cycles: u64,
    pub severity: GapSeverity,
}

/// Quote without matching counterpart (bid without ask, or vice versa)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnpairedQuote {
    pub timestamp: DateTime<Utc>,
    pub order_id: String,
    pub side: String,
    pub price: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TradeRecord tests
    // ========================================================================

    #[test]
    fn test_trade_record_serde_round_trip() {
        let record = TradeRecord {
            trade_id: 1,
            side: "long".to_string(),
            entry_signal_price: 100.0,
            entry_fill_price: 100.05,
            exit_signal_price: Some(110.0),
            exit_fill_price: Some(109.95),
            quantity: 1.0,
            pnl: Some(9.9),
            pnl_pct: Some(9.9),
            commission: 0.1,
            slippage_cost: 0.1,
            entry_time: Utc::now(),
            exit_time: Some(Utc::now()),
            duration_secs: Some(3600),
            entry_liquidity: "taker".to_string(),
            exit_liquidity: Some("maker".to_string()),
            entry_reason: "signal_buy".to_string(),
            exit_reason: Some("signal_sell".to_string()),
            mae: Some(-0.02),
            mfe: Some(0.12),
            legs: Vec::new(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let deser: TradeRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.trade_id, 1);
        assert_eq!(deser.side, "long");
        assert_eq!(deser.entry_fill_price, 100.05);
        assert_eq!(deser.pnl, Some(9.9));
        assert!(deser.legs.is_empty());
    }

    #[test]
    fn test_trade_record_legs_default_to_empty_when_absent_from_json() {
        // Older persisted trade logs (or any producer that predates this
        // field) have no "legs" key at all -- #[serde(default)] must not error.
        let json = r#"{"trade_id":1,"side":"long","entry_signal_price":100.0,
            "entry_fill_price":100.05,"exit_signal_price":null,"exit_fill_price":null,
            "quantity":1.0,"pnl":null,"pnl_pct":null,"commission":0.1,"slippage_cost":0.1,
            "entry_time":"2026-01-01T00:00:00Z","exit_time":null,"duration_secs":null,
            "entry_liquidity":"taker","exit_liquidity":null,"entry_reason":"signal_buy",
            "exit_reason":null,"mae":null,"mfe":null}"#;
        let decoded: TradeRecord = serde_json::from_str(json).unwrap();
        assert!(decoded.legs.is_empty());
    }

    #[test]
    fn test_trade_record_legs_round_trip_a_multi_venue_trade() {
        let leg_a = TradeLeg {
            exchange: "kraken".to_string(),
            symbol: "BTC-USD".to_string(),
            side: "long".to_string(),
            fill_price: 50_000.0,
            quantity: 0.5,
            liquidity: "taker".to_string(),
            commission: 5.0,
            slippage_cost: 1.0,
            timestamp: Utc::now(),
        };
        let leg_b = TradeLeg {
            exchange: "coinbase".to_string(),
            symbol: "BTC-USD".to_string(),
            side: "short".to_string(),
            fill_price: 50_050.0,
            quantity: 0.5,
            liquidity: "taker".to_string(),
            commission: 5.0,
            slippage_cost: 1.0,
            timestamp: Utc::now(),
        };
        let mut record = TradeRecord {
            trade_id: 3,
            side: "long".to_string(),
            entry_signal_price: 50_000.0,
            entry_fill_price: 50_000.0,
            exit_signal_price: None,
            exit_fill_price: None,
            quantity: 0.5,
            pnl: Some(25.0),
            pnl_pct: Some(0.001),
            commission: 10.0,
            slippage_cost: 2.0,
            entry_time: Utc::now(),
            exit_time: None,
            duration_secs: None,
            entry_liquidity: "taker".to_string(),
            exit_liquidity: None,
            entry_reason: "cross_venue_spread".to_string(),
            exit_reason: None,
            mae: None,
            mfe: None,
            legs: Vec::new(),
        };
        record.legs.push(leg_a);
        record.legs.push(leg_b);

        let json = serde_json::to_string(&record).unwrap();
        let deser: TradeRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.legs.len(), 2);
        assert_eq!(deser.legs[0].exchange, "kraken");
        assert_eq!(deser.legs[1].exchange, "coinbase");
    }

    #[test]
    fn test_trade_record_open_position() {
        let record = TradeRecord {
            trade_id: 2,
            side: "short".to_string(),
            entry_signal_price: 200.0,
            entry_fill_price: 200.1,
            exit_signal_price: None,
            exit_fill_price: None,
            quantity: 0.5,
            pnl: None,
            pnl_pct: None,
            commission: 0.05,
            slippage_cost: 0.1,
            entry_time: Utc::now(),
            exit_time: None,
            duration_secs: None,
            entry_liquidity: "maker".to_string(),
            exit_liquidity: None,
            entry_reason: "signal_sell".to_string(),
            exit_reason: None,
            mae: None,
            mfe: None,
            legs: Vec::new(),
        };
        assert!(record.exit_fill_price.is_none());
        assert!(record.pnl.is_none());
        assert!(record.exit_time.is_none());
    }

    // ========================================================================
    // BacktestResult tests
    // ========================================================================

    #[test]
    fn test_backtest_result_default() {
        let result = BacktestResult::default();
        assert_eq!(result.total_pnl, 0.0);
        assert_eq!(result.num_trades, 0);
        assert_eq!(result.closed_trades, 0);
        assert_eq!(result.open_positions, 0);
        assert_eq!(result.max_drawdown, 0.0);
        assert!(result.sharpe_ratio.is_none());
        assert!(result.equity_curve.is_empty());
        assert!(result.trade_returns.is_empty());
        assert!(result.trade_maes.is_empty());
        assert!(result.trade_log.is_empty());
        assert!(result.price_series.is_empty());
    }

    #[test]
    fn test_backtest_result_serde_round_trip() {
        let mut result = BacktestResult::default();
        result.total_pnl = 1000.0;
        result.num_trades = 50;
        result.sharpe_ratio = Some(1.5);
        result.equity_curve = vec![100.0, 101.0, 102.0];
        result.trade_returns = vec![0.01, -0.005, 0.02];

        let json = serde_json::to_string(&result).unwrap();
        let deser: BacktestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.total_pnl, 1000.0);
        assert_eq!(deser.num_trades, 50);
        assert_eq!(deser.sharpe_ratio, Some(1.5));
        assert_eq!(deser.equity_curve.len(), 3);
    }

    #[test]
    fn test_backtest_result_price_series_skipped_when_empty() {
        let result = BacktestResult::default();
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("price_series"));
    }

    // ========================================================================
    // BacktestEvent tests
    // ========================================================================

    #[test]
    fn test_backtest_event_market_data() {
        let event = BacktestEvent::MarketData {
            timestamp: Utc::now(),
        };
        match event {
            BacktestEvent::MarketData { timestamp } => {
                assert!(timestamp <= Utc::now());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_backtest_event_order_filled() {
        let event = BacktestEvent::OrderFilled {
            order_id: Arc::from("order-1"),
            fill_price: 50000.0,
            quantity: 0.1,
            timestamp: Utc::now(),
        };
        match event {
            BacktestEvent::OrderFilled { fill_price, quantity, .. } => {
                assert_eq!(fill_price, 50000.0);
                assert_eq!(quantity, 0.1);
            }
            _ => panic!("Wrong variant"),
        }
    }

    // ========================================================================
    // GapSeverity tests
    // ========================================================================

    #[test]
    fn test_gap_severity_serde() {
        let severities = vec![
            GapSeverity::Minor,
            GapSeverity::Moderate,
            GapSeverity::Severe,
            GapSeverity::Critical,
        ];
        for sev in severities {
            let json = serde_json::to_string(&sev).unwrap();
            let deser: GapSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(sev, deser);
        }
    }

    // ========================================================================
    // ExecutionMetrics tests
    // ========================================================================

    #[test]
    fn test_execution_metrics_default() {
        let m = ExecutionMetrics::default();
        assert_eq!(m.orders_placed, 0);
        assert_eq!(m.orders_filled, 0);
        assert_eq!(m.fill_rate, 0.0);
        assert!(m.round_trip_win_rate.is_none());
    }

    #[test]
    fn test_execution_metrics_serde() {
        let m = ExecutionMetrics {
            orders_placed: 100,
            orders_filled: 80,
            fill_rate: 0.8,
            total_fills: 95,
            ..ExecutionMetrics::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let deser: ExecutionMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.orders_placed, 100);
        assert_eq!(deser.fill_rate, 0.8);
    }

    // ========================================================================
    // InventoryMetrics tests
    // ========================================================================

    #[test]
    fn test_inventory_metrics_default() {
        let m = InventoryMetrics::default();
        assert_eq!(m.total_volume_traded, 0.0);
        assert_eq!(m.buy_fills, 0);
        assert_eq!(m.sell_fills, 0);
        assert_eq!(m.starting_inventory, 0.0);
        assert_eq!(m.ending_inventory, 0.0);
    }

    #[test]
    fn test_inventory_metrics_serde() {
        let m = InventoryMetrics {
            total_volume_traded: 1_000_000.0,
            buy_fills: 500,
            sell_fills: 480,
            buy_volume: 500_000.0,
            sell_volume: 480_000.0,
            imbalance_ratio: 0.02,
            ..InventoryMetrics::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let deser: InventoryMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.total_volume_traded, 1_000_000.0);
        assert_eq!(deser.buy_fills, 500);
    }

    // ========================================================================
    // DataQualityReport tests
    // ========================================================================

    #[test]
    fn test_data_quality_report_serde() {
        let report = DataQualityReport {
            overall_score: 0.95,
            total_data_points: 1_000_000,
            time_span_seconds: 86400,
            gaps_detected: vec![],
            average_gap_seconds: 0.1,
            max_gap_seconds: 5,
            gap_ratio: 0.001,
            outliers_detected: 3,
            data_completeness_score: 0.999,
            timestamp_consistency_score: 0.998,
            order_quality: None,
        };
        let json = serde_json::to_string(&report).unwrap();
        let deser: DataQualityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.overall_score, 0.95);
        assert_eq!(deser.total_data_points, 1_000_000);
    }

    // ========================================================================
    // TransactionCostAnalysis tests
    // ========================================================================

    #[test]
    fn test_transaction_cost_analysis_serde() {
        let tca = TransactionCostAnalysis {
            total_commission: 50.0,
            total_slippage: 25.0,
            total_market_impact: 10.0,
            total_transaction_costs: 85.0,
            costs_as_percentage_of_pnl: 8.5,
            avg_cost_per_trade: 1.7,
            commission_by_exchange: std::collections::HashMap::from([
                ("Kraken".to_string(), 30.0),
                ("Binance".to_string(), 20.0),
            ]),
            total_gas_costs_usd: 0.0,
            avg_dex_slippage_bps: 0.0,
        };
        let json = serde_json::to_string(&tca).unwrap();
        let deser: TransactionCostAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.total_transaction_costs, 85.0);
    }
}
