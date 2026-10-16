//! Core strategy traits and types
//!
//! This module defines the fundamental Strategy trait that all trading strategies
//! must implement, along with supporting types for strategy execution context.

use crate::signal::Signal;
use portfoliomanager::PortfolioState;
use riskmanager::RiskLimits;
use anyhow::Result;
use async_trait::async_trait;
use config::ParameterValue;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, collections::HashMap};

pub use derivatives::{DerivativeMetadata, FundingRate, FuturesCurve, Greeks, InstrumentKind, IvSurface};

/// Result type for strategy operations
pub type StrategyResult<T> = Result<T, StrategyError>;

/// Lightweight snapshot of portfolio state for strategy context.
/// Avoids cloning Vec<Position> on every tick — only copies scalar fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub balance: f64,
    pub net_position: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    /// Number of currently open (non-closed) positions.
    pub position_count: usize,
}

impl From<&PortfolioState> for PortfolioSnapshot {
    fn from(ps: &PortfolioState) -> Self {
        PortfolioSnapshot {
            timestamp: ps.timestamp,
            balance: ps.balance,
            net_position: ps.net_position,
            unrealized_pnl: ps.unrealized_pnl,
            realized_pnl: ps.realized_pnl,
            fees_paid: ps.fees_paid,
            position_count: ps.positions.iter().filter(|p| p.close_time.is_none()).count(),
        }
    }
}

/// One venue's aligned price/volume/OHLC series, as passed to
/// `Strategy::compute_all_signals_multi_venue`. Same array shapes
/// `compute_all_signals` uses for its single flat series — just one bundle
/// per declared `(symbol, exchange)` venue instead of one overall.
#[derive(Debug, Clone, Default)]
pub struct VenueSeries {
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
}

/// How a pair strategy's hedge ratio is determined. See `PairSpec`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HedgeRatioMode {
    /// A fixed ratio the strategy author supplies once (e.g. from an offline
    /// `get_pair_cointegration` call) and never recomputes during the run.
    Fixed(f64),
    /// Recompute the hedge ratio periodically from the strategy's own
    /// `compute_all_signals_multi_venue` data (a rolling OLS re-estimate),
    /// matching how real pairs traders re-estimate on a rolling window
    /// rather than trusting one entry-time estimate for an entire deployment.
    Dynamic,
}

/// A strategy's declaration that it trades the spread between exactly two
/// symbols. Returned by `Strategy::declares_pair_trade`; see that method's
/// doc comment for how the engine interprets the joint signal this implies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairSpec {
    pub symbol_a: String,
    pub exchange_a: String,
    pub symbol_b: String,
    pub exchange_b: String,
    pub hedge_ratio_mode: HedgeRatioMode,
    /// When leg A trades an option contract rather than `symbol_a` itself
    /// (spot/futures), this carries that contract's strike/expiry/kind.
    /// `symbol_a`/`exchange_a` still name the *underlying* in this case,
    /// matching `DerivativeMetadata.underlying`'s convention elsewhere.
    /// `#[serde(default)]` keeps this field backward-compatible with
    /// `PairSpec` values already serialized by the live SignalEngine
    /// deployment before this field existed.
    #[serde(default)]
    pub option_leg_a: Option<DerivativeMetadata>,
    /// Same as `option_leg_a`, for leg B / `symbol_b`.
    #[serde(default)]
    pub option_leg_b: Option<DerivativeMetadata>,
}

/// One leg of a basket cointegration trade. See `BasketSpec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasketLeg {
    pub symbol: String,
    pub exchange: String,
    /// When this leg trades an option contract rather than `symbol` itself,
    /// carries that contract's strike/expiry/kind — see `PairSpec::option_leg_a`'s
    /// doc comment for the same convention. `#[serde(default)]` for the same
    /// backward-compatibility reason.
    #[serde(default)]
    pub option_instrument: Option<DerivativeMetadata>,
}

/// How a basket strategy's weights are determined -- the N-way
/// generalization of `HedgeRatioMode`. `Fixed` weights are normalized so
/// `weights[0] == 1.0` (matching `quant_diagnostics::normalize_cointegrating_vector`'s
/// convention): "1 unit of `legs[0]` per `weights[i]` units of `legs[i]`",
/// sign included (a negative weight means that leg is on the opposite side
/// of `legs[0]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BasketWeightMode {
    Fixed(Vec<f64>),
    Dynamic,
}

/// A strategy's declaration that it trades a stationary linear combination
/// of 3+ symbols (basket statistical arbitrage) -- the N-way generalization
/// of `PairSpec`, backed by `get_basket_cointegration`'s Johansen procedure
/// rather than `get_pair_cointegration`'s Engle-Granger test. Returned by
/// `Strategy::declares_basket_trade`. Kept as a genuinely separate type
/// from `PairSpec` (not "PairSpec with N legs") since the pairs path is
/// already fully built end-to-end (backtest execution, live deployment) and
/// deliberately left untouched -- this is additive, mirroring how
/// `declares_pair_trade` itself was added alongside `accepts_multi_venue`
/// without changing that mechanism.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasketSpec {
    pub legs: Vec<BasketLeg>,
    pub weight_mode: BasketWeightMode,
}

/// Strategy-specific error types
#[derive(Debug, thiserror::Error)]
pub enum StrategyError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameter: {parameter} = {value}")]
    InvalidParameter { parameter: String, value: String },
    
    #[error("Market data error: {0}")]
    MarketDataError(String),
    
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    
    #[error("Strategy not initialized")]
    NotInitialized,
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Calculation error: {0}")]
    CalculationError(String),

    /// Returned by the executor when an OptionSpread cannot be filled atomically.
    #[error("Multi-leg order rejected: {reason}")]
    MultiLegRejected {
        /// Human-readable description of which leg failed and why.
        reason: String,
        /// Number of legs that would have filled before rejection.
        filled_legs: usize,
        /// Total number of legs in the spread.
        total_legs: usize,
    },
}

/// Context information provided to strategies during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyContext {
    /// Current timestamp
    pub timestamp: DateTime<Utc>,
    /// Lightweight portfolio snapshot (avoids Vec<Position> clone per tick)
    pub portfolio_state: PortfolioSnapshot,
    /// Historical price data window
    pub price_history: Vec<PricePoint>,
    /// Volume history
    pub volume_history: Vec<VolumePoint>,
    /// Current spread information
    pub spread: Option<f64>,
    /// Market session information
    pub market_session: MarketSession,
    /// Orderbook snapshot (when available — trade-level or L2 data)
    #[serde(default)]
    pub orderbook_snapshot: Option<OrderbookSnapshot>,
    /// Supplementary user data (from uploaded feature files, time-aligned)
    #[serde(default)]
    pub custom_data: HashMap<String, f64>,
    /// Multi-asset context — latest price + position per asset in the portfolio
    #[serde(default)]
    pub assets: HashMap<String, AssetSnapshot>,
    /// Current tick number (0-indexed, monotonically increasing)
    #[serde(default)]
    pub tick_number: u64,
    /// Elapsed seconds since backtest start
    #[serde(default)]
    pub elapsed_seconds: f64,

    // ── Derivatives market data ─────────────────────────────────────────

    /// Implied volatility surface for the primary underlying (options strategies).
    /// Populated when `DataRequirements::needs_iv_surface` is true.
    #[serde(default)]
    pub iv_surface: Option<IvSurface>,

    /// Futures term structure for the primary underlying.
    /// Populated when `DataRequirements::needs_futures_curve` is true.
    #[serde(default)]
    pub futures_curve: Option<FuturesCurve>,

    /// Funding rates keyed by perpetual symbol.
    /// Populated when `DataRequirements::needs_funding_rates` is true.
    #[serde(default)]
    pub funding_rates: Option<HashMap<String, FundingRate>>,

    /// Aggregate portfolio-level Greeks, pre-computed by the portfolio manager.
    /// `None` when the portfolio holds no derivative positions.
    #[serde(default)]
    pub portfolio_greeks: Option<Greeks>,

    // ── DEX / LP pool state ────────────────────────────────────────────

    /// Current pool state snapshot for LP strategies.
    /// Populated when running an LP backtest (`config.lp.is_some()`).
    #[serde(default)]
    pub pool_state: Option<PoolStateSnapshot>,
}

/// Orderbook snapshot passed to Python strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    /// Top N bid levels as `[price, size]` pairs (sorted best-first)
    pub bids: Vec<[f64; 2]>,
    /// Top N ask levels as `[price, size]` pairs (sorted best-first)
    pub asks: Vec<[f64; 2]>,
    /// Best bid - best ask
    pub spread: f64,
    /// (best_bid + best_ask) / 2
    pub mid_price: f64,
    /// Total bid liquidity (sum of bid sizes)
    pub bid_depth: f64,
    /// Total ask liquidity (sum of ask sizes)
    pub ask_depth: f64,
    /// (bid_depth - ask_depth) / (bid_depth + ask_depth)
    pub imbalance: f64,
}

/// Per-asset snapshot for multi-asset strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSnapshot {
    /// Latest price
    pub price: f64,
    /// Current net position quantity
    pub position: f64,
}

/// Pool state snapshot for LP strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStateSnapshot {
    pub pool_id: String,
    pub current_price: f64,
    pub current_tick: i32,
    pub total_liquidity: f64,
    pub reserve_a: f64,
    pub reserve_b: f64,
    pub fee_tier_bps: u32,
    pub volume_usd_24h: f64,
    pub my_positions: Vec<LpPositionSummary>,
}

/// Summary of an active LP position (passed into strategy context)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpPositionSummary {
    pub position_id: String,
    pub liquidity_units: f64,
    pub tick_lower: Option<i32>,
    pub tick_upper: Option<i32>,
    pub in_range: bool,
    pub fees_earned_usd: f64,
    pub impermanent_loss_usd: f64,
    pub net_value_usd: f64,
}

/// Price point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub source: PriceSource,
}

/// Volume point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePoint {
    pub timestamp: DateTime<Utc>,
    pub volume: f64,
    pub side: Option<OrderSide>,
}

/// Source of price data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriceSource {
    Trade,
    BestBid,
    BestAsk,
    MidPrice,
    VWAP,
}

/// Order side for volume tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Market session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSession {
    pub is_open: bool,
    pub session_type: SessionType,
    pub next_open: Option<DateTime<Utc>>,
    pub next_close: Option<DateTime<Utc>>,
}

/// Type of market session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    PreMarket,
    Regular,
    PostMarket,
    Closed,
}

/// Strategy category for data loading optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyCategory {
    /// Tick-level strategies needing order flow, trades, and orderbook data
    MarketMaking,
    /// Momentum/trend strategies work best with candle data
    Momentum,
    /// Arbitrage strategies need real-time data from multiple sources
    Arbitrage,
    /// Statistical arbitrage using mean reversion
    StatisticalArbitrage,
    /// Options strategies (calls, puts, spreads) — needs IV surface
    Options,
    /// Futures strategies (dated contracts, basis trading) — needs futures curve
    Futures,
    /// Volatility-focused strategies (vol surface arbitrage, vega-neutral)
    Volatility,
    /// Multi-leg spread strategies (straddles, condors, butterflies)
    Spread,
    /// AMM liquidity provision (concentrated, constant product, stable)
    LiquidityProvision,
    /// Custom strategy with explicit data requirements
    Custom,
}

/// Data requirements for a strategy
/// 
/// Strategies declare what data they need, and the data loader
/// intelligently fetches the appropriate data types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequirements {
    /// Strategy category (determines default data types)
    pub category: StrategyCategory,
    
    /// Whether strategy needs historical order flow data
    pub needs_order_flow: bool,
    
    /// Whether strategy needs trade/tick data
    pub needs_trades: bool,
    
    /// Whether strategy needs orderbook snapshots
    pub needs_orderbook_snapshots: bool,
    
    /// Whether strategy needs OHLCV candle data
    pub needs_candles: bool,
    
    /// Minimum lookback window (number of data points)
    pub lookback_window: usize,
    
    /// Whether strategy needs real-time orderbook depth
    pub needs_depth: bool,
    
    /// Maximum data age in seconds (for freshness requirements)
    pub max_data_age_secs: Option<u64>,

    /// Whether strategy needs the implied volatility surface.
    /// Set to `true` for options and volatility strategies.
    pub needs_iv_surface: bool,

    /// Whether strategy needs the futures term structure / basis curve.
    pub needs_futures_curve: bool,

    /// Whether strategy needs current perpetual funding rates.
    pub needs_funding_rates: bool,
}

impl DataRequirements {
    /// Create requirements for a market-making strategy
    /// Market makers need order flow, trades, and orderbook data
    pub fn market_making() -> Self {
        Self {
            category: StrategyCategory::MarketMaking,
            needs_order_flow: true,
            needs_trades: true,
            needs_orderbook_snapshots: true,
            needs_candles: false,
            lookback_window: 1000,
            needs_depth: true,
            max_data_age_secs: Some(60),
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }

    /// Create requirements for an options strategy.
    /// Needs IV surface, trades, and orderbook snapshots.
    pub fn options() -> Self {
        Self {
            category: StrategyCategory::Options,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: true,
            needs_candles: true,
            lookback_window: 200,
            needs_depth: false,
            max_data_age_secs: Some(300),
            needs_iv_surface: true,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }

    /// Create requirements for a futures / basis trading strategy.
    pub fn futures() -> Self {
        Self {
            category: StrategyCategory::Futures,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: false,
            needs_candles: true,
            lookback_window: 200,
            needs_depth: false,
            max_data_age_secs: None,
            needs_iv_surface: false,
            needs_futures_curve: true,
            needs_funding_rates: false,
        }
    }

    /// Create requirements for a perpetual / funding-rate strategy.
    pub fn perpetual() -> Self {
        Self {
            category: StrategyCategory::Futures,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: false,
            needs_candles: false,
            lookback_window: 100,
            needs_depth: false,
            max_data_age_secs: Some(60),
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: true,
        }
    }

    /// Create requirements for a volatility / IV-surface strategy.
    pub fn volatility() -> Self {
        Self {
            category: StrategyCategory::Volatility,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: true,
            needs_candles: true,
            lookback_window: 500,
            needs_depth: false,
            max_data_age_secs: Some(300),
            needs_iv_surface: true,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }
    
    /// Create requirements for a momentum/trend strategy
    /// Momentum strategies work with candle data
    pub fn momentum() -> Self {
        Self {
            category: StrategyCategory::Momentum,
            needs_order_flow: false,
            needs_trades: false,
            needs_orderbook_snapshots: false,
            needs_candles: true,
            lookback_window: 200,
            needs_depth: false,
            max_data_age_secs: None,
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }

    /// Create requirements for an arbitrage strategy
    pub fn arbitrage() -> Self {
        Self {
            category: StrategyCategory::Arbitrage,
            needs_order_flow: true,
            needs_trades: true,
            needs_orderbook_snapshots: true,
            needs_candles: false,
            lookback_window: 100,
            needs_depth: true,
            max_data_age_secs: Some(1),
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }
    
    /// Create requirements for statistical arbitrage
    pub fn stat_arb() -> Self {
        Self {
            category: StrategyCategory::StatisticalArbitrage,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: false,
            needs_candles: true,
            lookback_window: 500,
            needs_depth: false,
            max_data_age_secs: None,
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }

    /// Create requirements for an LP / liquidity provision strategy.
    /// LP strategies consume pool swap events, not traditional trades.
    pub fn liquidity_provision() -> Self {
        Self {
            category: StrategyCategory::LiquidityProvision,
            needs_order_flow: false,
            needs_trades: false,
            needs_orderbook_snapshots: false,
            needs_candles: false,
            lookback_window: 100,
            needs_depth: false,
            max_data_age_secs: None,
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }
}

impl Default for DataRequirements {
    fn default() -> Self {
        // Default to momentum/candle-based requirements — Custom/Python
        // strategies (the only supported strategy type) are candle-driven,
        // not tick-level market-making style order flow consumers.
        Self::momentum()
    }
}

/// Core strategy trait that all trading strategies must implement
#[async_trait]
pub trait Strategy: Send + Sync + Debug {
    /// Get the strategy name
    fn name(&self) -> &str;
    
    /// Get the strategy version
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    /// Get strategy description
    fn description(&self) -> &str {
        "Trading strategy"
    }
    
    /// Initialize the strategy with parameters
    async fn initialize(&mut self, parameters: HashMap<String, ParameterValue>) -> StrategyResult<()>;
    
    /// Generate trading signals based on market data (per-tick fallback path).
    ///
    /// For Python strategies this is a stub — see `compute_all_signals()` for the
    /// vectorized bulk path that replaces the per-tick PyO3 call.
    async fn generate_signals(
        &mut self,
        market_data: &dataloader::MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>>;

    /// Bulk-vectorized signal generation for the entire tick array.
    ///
    /// Called ONCE before the simulation tick loop with the complete price/volume/
    /// timestamp arrays.  Returns `Some(signals)` where `signals[i]` encodes the
    /// action for tick i using the int8 convention:
    ///   1  = BUY   (long entry / increase)
    ///  -1  = SELL  (short entry / increase)
    ///   0  = HOLD  (no change)
    ///   2  = CLOSE (close any open position)
    ///
    /// `prices` is the close series. `opens`/`highs`/`lows` carry the rest of the
    /// OHLC bar; for tick/trade data (no real candles) they degenerate to the
    /// trade price. They are only forwarded to Python strategies that opt in by
    /// declaring the extra parameters (see `compute_signals_accepts_ohlc`).
    ///
    /// Returns `None` if the strategy does not implement the vectorized API —
    /// the caller falls back to the per-tick `generate_signals()` path.
    async fn compute_all_signals(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> StrategyResult<Option<Vec<i8>>> {
        let _ = (prices, volumes, timestamps, opens, highs, lows);
        Ok(None)
    }

    /// Whether this strategy's `compute_signals` accepts the optional OHLC
    /// arrays (opens/highs/lows). Default `false` — the engine then avoids
    /// building those arrays. Python strategies override this after inspecting
    /// the user's `compute_signals` signature during initialization.
    fn compute_signals_accepts_ohlc(&self) -> bool {
        false
    }

    /// Cross-exchange strategy support (2026-07-22): bulk-vectorized signal
    /// generation for a strategy that reads price/volume/timestamp data from
    /// TWO OR MORE venues at once (e.g. a stat-arb strategy comparing BTC-USD
    /// on Kraken against BTC-USD on Coinbase), as opposed to `compute_all_signals`'s
    /// single flat series. Keyed by `(symbol, exchange)` — see `StrategyConfig::venues`
    /// for how a strategy declares which venues it wants.
    ///
    /// Additive, not a replacement: mirrors `compute_all_signals`'s own
    /// default-`Ok(None)`-means-"not implemented" pattern exactly, so a
    /// single-venue strategy is entirely unaffected — the caller only builds
    /// the multi-venue bundle and calls this method when `accepts_multi_venue()`
    /// returns `true`; otherwise it falls back to `compute_all_signals` as before.
    ///
    /// V1 scope (see docs/PLAN or the cross-exchange implementation plan):
    /// vectorized/backtest-only, requires identical candle_interval_minutes
    /// across every declared venue (exact-timestamp alignment, no fuzzy/
    /// nearest-match join) — no live per-tick multi-venue trading yet.
    async fn compute_all_signals_multi_venue(
        &mut self,
        venues: &HashMap<(String, String), VenueSeries>,
    ) -> StrategyResult<Option<Vec<i8>>> {
        let _ = venues;
        Ok(None)
    }

    /// Whether this strategy implements `compute_all_signals_multi_venue`.
    /// Default `false` — the engine then never builds the (more expensive,
    /// per-venue) data bundle at all. Python strategies override this after
    /// inspecting the user's strategy code during initialization, mirroring
    /// `compute_signals_accepts_ohlc`'s exact pattern.
    fn accepts_multi_venue(&self) -> bool {
        false
    }

    /// Pairs-trading / statistical-arbitrage support: a strategy that trades
    /// the spread between exactly two symbols declares itself here. Its
    /// `compute_all_signals_multi_venue` output (built from the same
    /// `(symbol, exchange)`-keyed `VenueSeries` bundle multi-venue strategies
    /// already use) is interpreted as ONE joint spread signal — 1 = enter
    /// long-spread (long `symbol_a`, short `symbol_b`), -1 = enter
    /// short-spread, 0 = hold, 2 = close — rather than each venue's own
    /// independent directional signal. The engine routes strategies that
    /// return `Some` here to a dedicated two-leg execution path
    /// (`backtest::pair_simulation::run_pair_backtest`) instead of the
    /// ordinary single-primary-venue multi-venue path, since a pair needs
    /// BOTH legs to actually fill, not just the first-declared venue.
    ///
    /// Default `None` — every existing multi-venue strategy (cross-exchange
    /// arbitrage of one symbol across venues) is entirely unaffected; this is
    /// an additive, opt-in declaration, mirroring `accepts_multi_venue`'s own
    /// default-`false`-means-"not implemented" pattern. Python strategies
    /// override this after inspecting the user's code for a `pair_spec()`
    /// method during initialization.
    fn declares_pair_trade(&self) -> Option<PairSpec> {
        None
    }

    /// Why `declares_pair_trade` returned `None`, when `pair_spec()` exists
    /// but its return value didn't parse -- distinct from the method simply
    /// not being defined. `None` here (the default) covers both "no
    /// pair_spec() at all" and "not a Python strategy"; only `PythonStrategy`
    /// can distinguish further, since only it actually attempts the parse.
    fn pair_spec_error(&self) -> Option<String> {
        None
    }

    /// Basket statistical-arbitrage support: a strategy that trades a
    /// stationary linear combination of 3+ symbols declares itself here --
    /// the N-way generalization of `declares_pair_trade`. Its
    /// `compute_all_signals_multi_venue` output is interpreted the same way
    /// a pair's is (1 = enter basket long-combination, -1 = enter short,
    /// 2 = close), just spanning `legs.len()` legs instead of 2. Routed to
    /// `backtest::basket_simulation::run_basket_backtest`.
    ///
    /// Default `None` -- additive and independent of `declares_pair_trade`;
    /// a strategy should implement at most one of the two.
    fn declares_basket_trade(&self) -> Option<BasketSpec> {
        None
    }

    /// Mirrors `pair_spec_error`'s exact purpose, for `basket_spec()`.
    fn basket_spec_error(&self) -> Option<String> {
        None
    }

    /// Per-trade dynamic position sizing (2026-07-25): optional bulk-vectorized
    /// sizing alongside `compute_all_signals`. Returns one fraction of equity
    /// `(0.0, 1.0]` per tick, consulted only for ticks where `compute_all_signals`
    /// returned BUY(1)/SELL(-1) — ignored for HOLD(0)/CLOSE(2). Default: not
    /// supported, mirroring `compute_all_signals_multi_venue`'s own
    /// default-`Ok(None)` pattern. Unlike the signal contract, an invalid or
    /// missing result here must NEVER fail the backtest — the caller falls
    /// back to the flat `position_size_pct` (or per-tick, for individual NaN
    /// entries) instead.
    async fn compute_all_position_sizes(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> StrategyResult<Option<Vec<f64>>> {
        let _ = (prices, volumes, timestamps);
        Ok(None)
    }

    /// Whether this strategy implements `compute_position_sizes`. Default
    /// `false` — the engine then never makes the extra Python call. Python
    /// strategies override this after a simple `hasattr` check during
    /// initialization, mirroring `accepts_multi_venue`'s exact pattern.
    fn accepts_position_sizes(&self) -> bool {
        false
    }

    /// Whether this strategy implements `generate_signals`. Default `false`
    /// — the engine then always uses the vectorized `compute_signals()` bulk
    /// path and never calls this per-tick, context-carrying method. Python
    /// strategies override this after a simple `hasattr` check during
    /// initialization, mirroring `accepts_position_sizes`'s exact pattern.
    fn defines_generate_signals(&self) -> bool {
        false
    }

    /// Optional bulk-vectorized feature computation, alongside
    /// `compute_all_signals`. Returns `Ok(Some(features))` mapping a feature
    /// name to a value per tick (aligned with `prices`) when the strategy
    /// defines `compute_features()` and the caller has it enabled. Default:
    /// not supported. Diagnostic only — never affects trading signals.
    async fn compute_all_features(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> StrategyResult<Option<HashMap<String, Vec<f64>>>> {
        let _ = (prices, volumes, timestamps);
        Ok(None)
    }

    /// Update strategy state with new market data
    async fn update_state(
        &mut self,
        market_data: &dataloader::MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<()> {
        // Default implementation does nothing
        let _ = (market_data, context);
        Ok(())
    }
    
    /// Get current strategy parameters
    fn get_parameters(&self) -> HashMap<String, ParameterValue>;
    
    /// Update strategy parameters
    async fn update_parameters(&mut self, parameters: HashMap<String, ParameterValue>) -> StrategyResult<()>;
    
    /// Get strategy risk limits
    fn get_risk_limits(&self) -> RiskLimits;
    
    /// Check if strategy is ready to generate signals
    fn is_ready(&self) -> bool {
        true
    }
    
    /// Get required data window size (number of historical data points needed)
    fn required_data_window(&self) -> usize {
        100 // Default window size
    }

    /// Strategy-declared max leverage cap (e.g. from a Python strategy's
    /// `get_risk_limits().max_leverage`). `f64::INFINITY` (no additional cap)
    /// is the safe default for any strategy that doesn't declare one --
    /// callers compose this with the platform/exchange leverage cap (the
    /// lower of the two wins), never using it to raise leverage above what's
    /// configured. Deliberately not `1.0`: nothing distinguishes "never
    /// declared a cap" from "explicitly declared 1x", so a numeric default
    /// here would silently clamp every strategy down to 1x leverage.
    fn declared_max_leverage(&self) -> f64 {
        f64::INFINITY
    }
    
    /// Get data requirements for this strategy
    /// 
    /// This tells the data loader what types of data the strategy needs.
    /// Custom/Python strategies are candle-driven by default; strategies
    /// needing tick-level order flow can override this explicitly.
    fn data_requirements(&self) -> DataRequirements {
        DataRequirements::default()
    }
    
    /// Get strategy performance metrics (if available)
    fn get_metrics(&self) -> Option<StrategyMetrics> {
        None
    }
    
    /// Handle strategy warmup period
    async fn warmup(
        &mut self,
        historical_data: &[dataloader::MarketData],
        context: &StrategyContext,
    ) -> StrategyResult<()> {
        // Default implementation processes all historical data
        for data in historical_data {
            self.update_state(data, context).await?;
        }
        Ok(())
    }
    
    /// Reset strategy state
    async fn reset(&mut self) -> StrategyResult<()> {
        // Default implementation does nothing
        Ok(())
    }
    
    /// Validate strategy configuration
    fn validate_config(&self, parameters: &HashMap<String, ParameterValue>) -> StrategyResult<()> {
        // Default implementation accepts all parameters
        let _ = parameters;
        Ok(())
    }

    // ── Derivatives lifecycle hooks ───────────────────────────────────

    /// Called when one or more derivative positions are approaching expiry.
    ///
    /// The executor calls this once per expiring instrument, typically within
    /// one hour of expiry. Return signals to roll, close, or exercise the
    /// position. Default implementation does nothing.
    async fn on_expiry(
        &mut self,
        expired: &DerivativeMetadata,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let _ = (expired, context);
        Ok(vec![])
    }

    /// Called when a perpetual funding rate changes materially.
    ///
    /// Funding-rate strategies override this to re-assess sizing.
    /// Default implementation does nothing.
    async fn on_funding_rate_change(
        &mut self,
        symbol: &str,
        rate: &FundingRate,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let _ = (symbol, rate, context);
        Ok(vec![])
    }

    /// Declare the derivative instruments this strategy trades.
    ///
    /// Used by the executor and data loader to pre-fetch instrument metadata
    /// and subscribe to the right Deribit channels.  
    /// Default: no specific instruments (implies spot / dynamic instrument selection).
    fn required_instruments(&self) -> Vec<InstrumentKind> {
        vec![]
    }

    /// Desired net portfolio delta after rebalancing.
    ///
    /// When `Some(target)`, the executor will emit a `HedgeDelta` signal
    /// after each tick if the actual portfolio delta deviates beyond the
    /// strategy's configured tolerance.  
    /// Default: `None` (no automatic delta hedging).
    fn target_net_delta(&self) -> Option<f64> {
        None
    }
}

/// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    /// Total number of signals generated
    pub total_signals: u64,
    /// Number of buy signals
    pub buy_signals: u64,
    /// Number of sell signals
    pub sell_signals: u64,
    /// Average signal strength
    pub avg_signal_strength: f64,
    /// Strategy uptime
    pub uptime_percentage: f64,
    /// Last signal timestamp
    pub last_signal_time: Option<DateTime<Utc>>,
    /// Strategy-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl StrategyMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            total_signals: 0,
            buy_signals: 0,
            sell_signals: 0,
            avg_signal_strength: 0.0,
            uptime_percentage: 100.0,
            last_signal_time: None,
            custom_metrics: HashMap::new(),
        }
    }
    
    /// Update metrics with a new signal
    pub fn record_signal(&mut self, signal: &Signal) {
        self.total_signals += 1;
        
        match signal.signal_type {
            crate::signal::SignalType::Buy => self.buy_signals += 1,
            crate::signal::SignalType::Sell => self.sell_signals += 1,
            _ => {}
        }
        
        // Update average signal strength
        let strength_value = match signal.strength {
            crate::signal::SignalStrength::Weak => 1.0,
            crate::signal::SignalStrength::Medium => 2.0,
            crate::signal::SignalStrength::Strong => 3.0,
            crate::signal::SignalStrength::Custom(val) => val,
        };
        
        self.avg_signal_strength = ((self.avg_signal_strength * (self.total_signals - 1) as f64) 
                                   + strength_value) / self.total_signals as f64;
        
        self.last_signal_time = Some(signal.timestamp);
    }
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{SignalType, SignalStrength};

    #[test]
    fn test_strategy_metrics_creation() {
        let metrics = StrategyMetrics::new();
        assert_eq!(metrics.total_signals, 0);
        assert_eq!(metrics.buy_signals, 0);
        assert_eq!(metrics.sell_signals, 0);
        assert_eq!(metrics.avg_signal_strength, 0.0);
    }

    #[test]
    fn test_strategy_metrics_signal_recording() {
        let mut metrics = StrategyMetrics::new();
        
        let signal = Signal {
            id: "test".into(),
            timestamp: Utc::now(),
            signal_type: SignalType::Buy,
            strength: SignalStrength::Strong,
            price: Some(100.0),
            quantity: Some(10.0),
            reason: crate::signal::SignalReason::TechnicalAnalysis("test".to_string()),
            metadata: HashMap::new(),
            numeric_metadata: HashMap::new(),
            is_limit: false,
        };
        
        metrics.record_signal(&signal);
        
        assert_eq!(metrics.total_signals, 1);
        assert_eq!(metrics.buy_signals, 1);
        assert_eq!(metrics.avg_signal_strength, 3.0); // Strong signal = 3.0
        assert!(metrics.last_signal_time.is_some());
    }
}
