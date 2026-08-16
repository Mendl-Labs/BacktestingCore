use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use thiserror::Error;

/// Variable latency simulation with jitter and spikes
pub mod variable_latency;

pub use variable_latency::{VariableLatencyConfig, LatencySimulator, TimeOfDayLatencyPattern};

/// Configuration validation errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },
    
    #[error("Invalid value for {field}: {value} (reason: {reason})")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },
    
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Main backtesting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BacktestConfig {
    pub database: DatabaseConfig,
    pub trading: TradingConfig,
    pub data: DataConfig,
    pub output: OutputConfig,
    pub analysis: AnalysisConfig,
    /// Liquidity provision config — when present, runs the LP simulation loop instead of directional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp: Option<LpConfig>,
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: Option<u32>,
    pub connection_timeout: Option<u64>, // seconds
}

/// Trading-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub commission_rate: f64,      // e.g., 0.001 for 0.1% (fallback for simple fee)
    pub slippage_bps: f64,         // basis points (legacy — prefer slippage_model)
    /// Slippage model for candle-based simulation. Defaults to VolumeAdjusted.
    /// When set, overrides the flat `slippage_bps` field.
    #[serde(default)]
    pub slippage_model: SlippageModel,
    pub initial_capital: f64,
    pub max_position_size: Option<f64>, // as percentage of portfolio
    pub risk_free_rate: f64,       // for Sharpe ratio calculation
    pub exchange_fees: Option<ExchangeFeeConfig>, // Maker/taker fee configuration
    pub routing_mode: Option<RoutingMode>, // Multi-exchange routing configuration
    /// Simulated network/exchange latency in milliseconds
    /// Orders placed within this time window of a trade won't be filled
    /// This simulates realistic order propagation delay (default: 50ms)
    #[serde(default = "default_latency_ms")]
    pub latency_ms: u64,
    /// Latency jitter percentage (0.0-1.0)
    /// Adds random variation to latency: actual = latency_ms * (1 + jitter * random(-1, 1))
    /// Default: 0.2 (20% variation)
    #[serde(default = "default_latency_jitter")]
    pub latency_jitter_pct: f64,
    /// Minimum latency floor in milliseconds (even with negative jitter)
    #[serde(default = "default_min_latency_ms")]
    pub min_latency_ms: u64,
    /// Latency spike probability (0.0-1.0)
    /// Probability of experiencing a latency spike (3-10x normal latency)
    #[serde(default = "default_latency_spike_prob")]
    pub latency_spike_probability: f64,
    /// Number of historical price/volume ticks to pass to strategies.
    /// Larger windows give strategies more data but use more memory.
    /// Default: 1000
    #[serde(default = "default_lookback_window")]
    pub lookback_window: usize,
    /// Performance optimization settings
    #[serde(default)]
    pub optimization: PerformanceOptimization,
    /// Maximum fraction of bar volume that can be filled per tick (default: 0.10 = 10%).
    /// Prevents unrealistic fills that exceed available liquidity.
    #[serde(default = "default_fill_volume_fraction")]
    pub fill_volume_fraction: f64,
    /// Base spread in basis points for the synthetic orderbook (default: 5.0 bps).
    /// Used to model implied bid/ask when only candle data is available.
    #[serde(default = "default_synthetic_spread_bps")]
    pub synthetic_spread_bps: f64,
    /// Number of depth levels in the synthetic orderbook (default: 5).
    #[serde(default = "default_synthetic_depth_levels")]
    pub synthetic_depth_levels: usize,
    /// Require price to cross *through* the limit level for a fill (default: true).
    /// When true, buy limits require price < order.price (not <=), preventing "touch fills"
    /// that assume perfect queue priority.
    #[serde(default = "default_true")]
    pub adverse_selection: bool,
    /// Use synthetic orderbook for taker (market) fills (default: true).
    /// When enabled, taker slippage is computed by walking through depth levels
    /// generated from OHLCV data (real size-scaling) rather than using a flat
    /// bps adjustment that's identical regardless of order size.
    #[serde(default = "default_true")]
    pub use_synthetic_book: bool,
    /// DEX-specific cost model (gas + AMM/CLOB slippage). When enabled, this
    /// replaces the standard slippage_bps for Sui DEX backtests.
    #[serde(default)]
    pub dex_costs: Option<DexCostConfig>,
    /// Leverage multiplier for margined positions. `1.0` (default) = no
    /// leverage -- behavior-identical to the pre-leverage engine (see
    /// `portfoliomanager::margin::size_leveraged_order`'s doc comment for
    /// why leverage=1.0 is exactly the old sizing formula, not a special case).
    #[serde(default = "default_leverage")]
    pub leverage: f64,
    /// Flat maintenance-margin ratio against notional at mark price, used to
    /// compute the liquidation trigger for leveraged positions (isolated
    /// margin, v1 -- no tiered curve). Default 0.5%.
    #[serde(default = "default_maintenance_margin_ratio")]
    pub maintenance_margin_ratio: f64,
    /// Extra adverse slippage (in basis points) applied to a liquidation's
    /// exit price on top of the theoretical trigger price, matching how real
    /// exchanges execute liquidations worse than the exact trigger and
    /// charge a liquidation fee. Default 50bps.
    #[serde(default = "default_liquidation_penalty_bps")]
    pub liquidation_penalty_bps: f64,
    /// Volatility-targeting position-size overlay (opt-in; `None` disables
    /// it entirely -- zero behavior change from before this feature
    /// existed). When set to e.g. `0.15`, position size is scaled inversely
    /// to the asset's own trailing realized volatility so the position's
    /// expected annualized volatility sits near this target instead of
    /// scaling only with notional -- see
    /// `portfoliomanager::margin::VolTargetConfig`'s doc comment for the
    /// literature this is based on. A pure risk overlay: layers on top of
    /// any existing entry/exit logic without changing it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vol_target_annual: Option<f64>,
    /// Trailing bars used to estimate realized volatility for the overlay
    /// above. Ignored when `vol_target_annual` is `None`. Default 20.
    #[serde(default = "default_vol_target_lookback_bars")]
    pub vol_target_lookback_bars: usize,
    /// Bars per year for this candle granularity, used to annualize the
    /// trailing per-bar realized volatility. Default 365.0 (daily bars --
    /// this platform's standard crypto annualization convention). Override
    /// this for intraday granularities (e.g. hourly bars: `365.0 * 24.0`) --
    /// getting it wrong doesn't crash anything, it just biases the scale
    /// factor consistently in one direction.
    #[serde(default = "default_vol_target_bars_per_year")]
    pub vol_target_bars_per_year: f64,
    /// Floor on the vol-targeting scale multiplier. Default 0.25 (never
    /// size below 1/4 of the stated `position_size_pct`).
    #[serde(default = "default_vol_target_min_scale")]
    pub vol_target_min_scale: f64,
    /// Ceiling on the vol-targeting scale multiplier. Default 3.0 (never
    /// size above 3x the stated `position_size_pct`).
    #[serde(default = "default_vol_target_max_scale")]
    pub vol_target_max_scale: f64,
}

fn default_lookback_window() -> usize { 1000 }
fn default_fill_volume_fraction() -> f64 { 0.10 }
fn default_synthetic_spread_bps() -> f64 { 5.0 }
fn default_synthetic_depth_levels() -> usize { 5 }
fn default_leverage() -> f64 { 1.0 }
/// Public so GA fitness paths that construct a `CandleSimulator` without a
/// full `TradingConfig` (e.g. `distributed::ga_worker::evaluate_candle_based`)
/// can apply the platform's standard default rather than inventing their own.
pub fn default_maintenance_margin_ratio() -> f64 { 0.005 }
pub fn default_liquidation_penalty_bps() -> f64 { 50.0 }
pub fn default_vol_target_lookback_bars() -> usize { 20 }
pub fn default_vol_target_bars_per_year() -> f64 { 365.0 }
pub fn default_vol_target_min_scale() -> f64 { 0.25 }
pub fn default_vol_target_max_scale() -> f64 { 3.0 }

/// DEX cost model configuration for Sui-based DEX backtests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexCostConfig {
    /// Whether to apply DEX cost model (gas + slippage)
    #[serde(default)]
    pub enabled: bool,
    /// Gas cost per on-chain transaction in native token (SUI). Default: 0.001 SUI.
    #[serde(default = "default_gas_cost")]
    pub gas_cost_per_tx: f64,
    /// Price of the native token in USD (SUI/USD) for gas cost conversion.
    #[serde(default = "default_native_token_price")]
    pub native_token_price_usd: f64,
    /// Slippage model to apply
    #[serde(default)]
    pub slippage_model: DexSlippageModel,
    /// Pool total value locked in USD (for AMM slippage calculation)
    #[serde(default = "default_pool_liquidity")]
    pub pool_liquidity_usd: f64,
    /// Concentration factor for Cetus concentrated-liquidity pools (0.0-1.0).
    /// Lower = more concentrated = less slippage. Default: 0.4.
    #[serde(default = "default_concentration_factor")]
    pub concentration_factor: f64,
    /// Simulated execution delay for on-chain finality in milliseconds.
    /// Sui typically finalizes in ~400ms.
    #[serde(default = "default_execution_delay")]
    pub execution_delay_ms: u64,
}

fn default_gas_cost() -> f64 { 0.001 }
fn default_native_token_price() -> f64 { 1.0 }
fn default_pool_liquidity() -> f64 { 1_000_000.0 }
fn default_concentration_factor() -> f64 { 0.4 }
fn default_execution_delay() -> u64 { 400 }

impl Default for DexCostConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gas_cost_per_tx: default_gas_cost(),
            native_token_price_usd: default_native_token_price(),
            slippage_model: DexSlippageModel::default(),
            pool_liquidity_usd: default_pool_liquidity(),
            concentration_factor: default_concentration_factor(),
            execution_delay_ms: default_execution_delay(),
        }
    }
}

/// DEX slippage model variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DexSlippageModel {
    /// Fixed basis-point slippage (simple model)
    FixedBps {
        #[serde(default = "default_fixed_slippage_bps")]
        bps: u32,
    },
    /// Constant-product AMM: slippage = trade_size / (reserve + trade_size)
    ConstantProduct,
    /// Concentrated-liquidity AMM (Cetus): reduced slippage within active range
    ConcentratedLiquidity,
    /// CLOB (DeepBook): near-zero slippage for limit orders, minimal for market orders
    Orderbook {
        #[serde(default = "default_orderbook_spread_bps")]
        spread_bps: u32,
    },
}

impl Default for DexSlippageModel {
    fn default() -> Self {
        Self::FixedBps { bps: default_fixed_slippage_bps() }
    }
}

fn default_fixed_slippage_bps() -> u32 { 30 }
fn default_orderbook_spread_bps() -> u32 { 5 }

/// Slippage model for candle-based (non-orderbook) backtesting.
/// Determines how execution slippage is estimated when only OHLCV data is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    /// Fixed basis-point slippage regardless of bar conditions (legacy behavior).
    Flat { bps: f64 },
    /// Volume-adjusted: penalizes fills on low-volume bars.
    /// effective_bps = base_bps * sqrt(avg_volume / max(bar_volume, epsilon))
    /// Capped at max_multiplier * base_bps to prevent blow-up on near-zero volume.
    VolumeAdjusted {
        base_bps: f64,
        max_multiplier: f64,
    },
}

impl Default for SlippageModel {
    fn default() -> Self {
        SlippageModel::VolumeAdjusted { base_bps: 3.0, max_multiplier: 10.0 }
    }
}

impl SlippageModel {
    /// Compute effective slippage in basis points for a given bar.
    pub fn effective_bps(&self, bar_volume: f64, avg_volume: f64) -> f64 {
        match self {
            SlippageModel::Flat { bps } => *bps,
            SlippageModel::VolumeAdjusted { base_bps, max_multiplier } => {
                let vol_ratio = if bar_volume > 0.0 && avg_volume > 0.0 {
                    (avg_volume / bar_volume).sqrt()
                } else {
                    *max_multiplier
                };
                base_bps * vol_ratio.min(*max_multiplier)
            }
        }
    }
}

impl DexCostConfig {
    /// Calculate slippage in basis points for a given trade size in USD.
    pub fn calculate_slippage_bps(&self, trade_size_usd: f64) -> f64 {
        match &self.slippage_model {
            DexSlippageModel::FixedBps { bps } => *bps as f64,
            DexSlippageModel::ConstantProduct => {
                let reserve = self.pool_liquidity_usd / 2.0;
                let slippage_fraction = trade_size_usd / (reserve + trade_size_usd);
                slippage_fraction * 10_000.0
            }
            DexSlippageModel::ConcentratedLiquidity => {
                let effective_liquidity = self.pool_liquidity_usd / self.concentration_factor.max(0.01);
                let reserve = effective_liquidity / 2.0;
                let slippage_fraction = trade_size_usd / (reserve + trade_size_usd);
                slippage_fraction * 10_000.0
            }
            DexSlippageModel::Orderbook { spread_bps } => {
                *spread_bps as f64 / 2.0
            }
        }
    }

    /// Calculate gas cost in USD for a single transaction.
    pub fn gas_cost_usd(&self) -> f64 {
        self.gas_cost_per_tx * self.native_token_price_usd
    }
}

/// Configuration for liquidity provision backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpConfig {
    pub pool_type: LpPoolType,
    pub pool_id: String,
    pub exchange: String,
    pub token_a: String,
    pub token_b: String,
    pub fee_tier_bps: u32,
    pub initial_deposit_usd: f64,
    #[serde(default = "default_deposit_ratio")]
    pub deposit_ratio: f64,
    #[serde(default = "default_lp_gas_cost")]
    pub gas_cost_per_operation: f64,
    #[serde(default = "default_sui_price")]
    pub native_token_price_usd: f64,
    pub range_width_ticks: Option<u32>,
    pub amplification_factor: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LpPoolType {
    ConstantProduct,
    ConcentratedLiquidity,
    StableSwap,
}

fn default_deposit_ratio() -> f64 { 0.5 }
fn default_lp_gas_cost() -> f64 { 0.001 }
fn default_sui_price() -> f64 { 1.0 }

/// Performance optimization settings for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimization {
    /// Enable adaptive sampling in GA (coarser data in early generations)
    /// Provides 1.3-1.5x speedup with minimal accuracy loss
    #[serde(default = "default_true")]
    pub adaptive_sampling_enabled: bool,
    /// Enable stable region skipping (skip strategy recalc when price unchanged)
    /// Provides 1.2-1.3x speedup depending on market activity
    #[serde(default = "default_true")]
    pub stable_region_skipping: bool,
    /// Price change threshold for stable region detection (as ratio, e.g., 0.0001 = 1bp)
    #[serde(default = "default_stable_threshold")]
    pub stable_region_threshold: f64,
    /// Maximum ticks to skip between forced strategy updates
    #[serde(default = "default_max_skip_ticks")]
    pub max_skip_ticks: usize,
    /// Override for MAX_CONCURRENT_EVALS (0 = auto-detect from memory)
    #[serde(default)]
    pub max_concurrent_evals: usize,
    /// Number of ticks to reuse Python-generated quotes before recalculating
    /// (0 = disabled, rely solely on stable region detection)
    #[serde(default = "default_quote_validity_ticks")]
    pub quote_validity_ticks: usize,
}

fn default_true() -> bool { true }
fn default_stable_threshold() -> f64 { 0.0001 } // 1 basis point
fn default_max_skip_ticks() -> usize { 100 }
fn default_quote_validity_ticks() -> usize { 50 }

impl Default for PerformanceOptimization {
    fn default() -> Self {
        Self {
            adaptive_sampling_enabled: true,
            stable_region_skipping: true,
            stable_region_threshold: 0.0001,
            max_skip_ticks: 100,
            max_concurrent_evals: 0, // Auto-detect
            quote_validity_ticks: 50,
        }
    }
}

fn default_latency_ms() -> u64 { 10 }
fn default_latency_jitter() -> f64 { 0.2 }
fn default_min_latency_ms() -> u64 { 1 }
fn default_latency_spike_prob() -> f64 { 0.01 }

/// Routing mode for multi-exchange strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingMode {
    /// Single exchange trading (default)
    SingleExchange {
        exchange: String,
    },
    /// Smart order routing across multiple exchanges
    SmartRouting {
        exchanges: Vec<String>,
        algorithm: RoutingAlgorithm,
        max_venues_per_order: usize,
        cross_exchange_latency_ms: f64,
    },
    /// Arbitrage-specific routing
    Arbitrage {
        exchanges: Vec<String>,
        min_profit_bps: f64,
        max_holding_time_seconds: u64,
        risk_limits: ArbitrageRiskLimits,
    },
}

/// Smart routing algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    /// Time-Weighted Average Price
    TWAP,
    /// Volume-Weighted Average Price  
    VWAP,
    /// Implementation Shortfall
    ImplementationShortfall,
    /// Best execution across venues
    BestExecution,
}

/// Arbitrage-specific risk limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageRiskLimits {
    pub max_open_positions: u32,
    pub max_daily_loss: f64,
    pub max_exchange_exposure: f64,
    pub min_liquidity_ratio: f64,
}

impl Default for ArbitrageRiskLimits {
    fn default() -> Self {
        Self {
            max_open_positions: 5,
            max_daily_loss: 1000.0,
            max_exchange_exposure: 10000.0,
            min_liquidity_ratio: 5.0,
        }
    }
}

/// Exchange-specific fee configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFeeConfig {
    pub exchange: String,          // "kraken", "coinbase", "binance", etc.
    pub maker_fee: f64,           // e.g., 0.0002 (0.02% for Kraken at high volume)
    pub taker_fee: f64,           // e.g., 0.0012 (0.12% for Kraken at high volume)
    pub volume_tiers: Vec<VolumeTier>,
    /// Starting 30-day volume to simulate an established market maker
    /// Set this to skip the highest fee tiers and use more realistic fees
    /// e.g., 1_000_000.0 for $1M 30-day volume = 0.06% maker / 0.16% taker on Kraken
    #[serde(default)]
    pub starting_30d_volume: f64,
    
    // === Order Size Limits ===
    /// Minimum order size in base currency (e.g., 0.0001 BTC for Kraken)
    #[serde(default = "default_min_order_size")]
    pub min_order_size: f64,
    /// Maximum order size in base currency
    #[serde(default = "default_max_order_size")]
    pub max_order_size: f64,
    
    // === Latency Modeling ===
    /// Mean latency in milliseconds for order execution
    #[serde(default = "default_latency_mean")]
    pub latency_mean_ms: f64,
    /// Standard deviation of latency (for jitter simulation)
    #[serde(default = "default_latency_std")]
    pub latency_std_ms: f64,
    
    // === Rate Limiting ===
    /// Maximum orders per second allowed by exchange
    #[serde(default = "default_max_orders_per_second")]
    pub max_orders_per_second: f64,
    /// Cooldown period in ms after rate limit hit
    #[serde(default = "default_rate_limit_cooldown")]
    pub rate_limit_cooldown_ms: i64,
    
    // === Funding Rate (for perpetuals) ===
    /// Funding rate per 8-hour interval (e.g., 0.0001 = 0.01%)
    #[serde(default = "default_funding_rate")]
    pub funding_rate_8h: f64,
    /// Funding interval in milliseconds (default 8 hours)
    #[serde(default = "default_funding_interval")]
    pub funding_interval_ms: i64,
    
    // === Swap/Overnight Fee (for prop firms or CFDs) ===
    /// Daily swap fee on open positions as a fraction of notional.
    /// 0.0 for spot exchanges (no overnight fee), >0 for prop firms.
    /// e.g., 0.00033 = 0.033%/day for Breakout prop firm.
    #[serde(default)]
    pub swap_fee_daily: f64,

    // === Leverage ===
    /// Maximum leverage this venue realistically allows (e.g. 4x Reg-T
    /// equities margin, 50x Oanda forex, 100x crypto derivatives, 1x for a
    /// spot-only exchange with no margin/futures product). Used to cap a
    /// user's requested `leverage` per-exchange (see
    /// `program/src/worker.rs`'s leverage resolution) rather than a single
    /// flat platform-wide number for every venue.
    #[serde(default = "default_max_leverage")]
    pub max_leverage: f64,
}

// Default value functions for serde
fn default_min_order_size() -> f64 { 0.0001 }  // 0.0001 BTC typical minimum
fn default_max_order_size() -> f64 { 100.0 }   // 100 BTC typical maximum
fn default_latency_mean() -> f64 { 10.0 }
fn default_latency_std() -> f64 { 3.0 }
fn default_max_orders_per_second() -> f64 { 10.0 }
fn default_rate_limit_cooldown() -> i64 { 1000 }
fn default_funding_rate() -> f64 { 0.0001 }
fn default_funding_interval() -> i64 { 8 * 60 * 60 * 1000 }
fn default_max_leverage() -> f64 { 3.0 }

impl Default for ExchangeFeeConfig {
    /// Default to Kraken fees starting at 0 volume (realistic for new market makers)
    /// Fees will decrease as 30-day volume builds during the backtest
    fn default() -> Self {
        Self::kraken()
    }
}

impl ExchangeFeeConfig {
    /// Kraken's complete fee tier structure (as of 2026)
    pub fn kraken_fee_tiers() -> Vec<VolumeTier> {
        vec![
            VolumeTier { min_volume_30d: 0.0,           maker_fee: 0.0025, taker_fee: 0.0040 },
            VolumeTier { min_volume_30d: 10_000.0,      maker_fee: 0.0020, taker_fee: 0.0035 },
            VolumeTier { min_volume_30d: 50_000.0,      maker_fee: 0.0014, taker_fee: 0.0024 },
            VolumeTier { min_volume_30d: 100_000.0,     maker_fee: 0.0012, taker_fee: 0.0022 },
            VolumeTier { min_volume_30d: 250_000.0,     maker_fee: 0.0010, taker_fee: 0.0020 },
            VolumeTier { min_volume_30d: 500_000.0,     maker_fee: 0.0008, taker_fee: 0.0018 },
            VolumeTier { min_volume_30d: 1_000_000.0,   maker_fee: 0.0006, taker_fee: 0.0016 },
            VolumeTier { min_volume_30d: 2_500_000.0,   maker_fee: 0.0004, taker_fee: 0.0014 },
            VolumeTier { min_volume_30d: 5_000_000.0,   maker_fee: 0.0002, taker_fee: 0.0012 },
            VolumeTier { min_volume_30d: 10_000_000.0,  maker_fee: 0.0000, taker_fee: 0.0010 },
            VolumeTier { min_volume_30d: 100_000_000.0, maker_fee: 0.0000, taker_fee: 0.0008 },
            VolumeTier { min_volume_30d: 500_000_000.0, maker_fee: 0.0000, taker_fee: 0.0005 },
        ]
    }
    
    /// Create Kraken exchange configuration (starting from 0 volume)
    pub fn kraken() -> Self {
        Self {
            exchange: "kraken".to_string(),
            maker_fee: 0.0025,  // 0.25% at 0 volume tier
            taker_fee: 0.0040,  // 0.40% at 0 volume tier
            volume_tiers: Self::kraken_fee_tiers(),
            starting_30d_volume: 0.0,
            // Kraken BTC/USD min: 0.0001 BTC, max: varies by pair
            min_order_size: 0.0001,
            max_order_size: 100.0,
            // Kraken REST API: ~50-100ms, Websocket: ~10-20ms
            latency_mean_ms: 15.0,
            latency_std_ms: 5.0,
            // Kraken rate limits: 15/sec burst for adding orders
            max_orders_per_second: 15.0,
            rate_limit_cooldown_ms: 1000,
            // Spot trading - no funding rates (only applies to perpetuals)
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            // Spot - no swap/overnight fees
            swap_fee_daily: 0.0,
            // Verified 2026-07: Kraken spot margin trading (what this preset's
            // fees model) caps at 10x. Kraken's separate derivatives product
            // supports up to 50x-100x on BTC/ETH perpetuals, but that's a
            // different product from the spot fee schedule modeled here.
            max_leverage: 10.0,
        }
    }

    /// Create Kraken configuration for an established market maker
    /// At $5M 30-day volume: 0.02% maker, 0.12% taker
    /// This is more realistic for backtesting as most serious MMs operate at volume discounts
    pub fn kraken_established() -> Self {
        let mut config = Self::kraken();
        config.starting_30d_volume = 5_000_000.0;  // $5M 30-day volume
        config.maker_fee = 0.0002;  // 0.02% at $5M tier
        config.taker_fee = 0.0012;  // 0.12% at $5M tier
        config
    }
    
    /// Create Kraken configuration for co-located infrastructure (e.g., Beeks Group)
    /// Sub-millisecond latency to Kraken's matching engine
    /// Use this with FillModelType::CoLocated for accurate simulation
    pub fn kraken_colocated() -> Self {
        let mut config = Self::kraken();
        // Co-located latency: ~0.5ms mean with minimal jitter
        config.latency_mean_ms = 0.5;
        config.latency_std_ms = 0.1;
        // Co-located can handle higher order rates
        config.max_orders_per_second = 50.0;
        config
    }
    
    /// Create Kraken configuration for co-located AND established market maker
    /// Best case: low fees + low latency
    pub fn kraken_colocated_established() -> Self {
        let mut config = Self::kraken_colocated();
        config.starting_30d_volume = 5_000_000.0;  // $5M 30-day volume
        config.maker_fee = 0.0002;  // 0.02% at $5M tier
        config.taker_fee = 0.0012;  // 0.12% at $5M tier
        config
    }
    
    /// Create Binance (global) exchange configuration
    pub fn binance() -> Self {
        Self {
            exchange: "binance".to_string(),
            maker_fee: 0.0010,  // 0.10% Regular user
            taker_fee: 0.0010,  // 0.10% Regular user
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,           maker_fee: 0.0010, taker_fee: 0.0010 },  // Regular
                VolumeTier { min_volume_30d: 1_000_000.0,   maker_fee: 0.0009, taker_fee: 0.0010 },  // VIP 1
                VolumeTier { min_volume_30d: 5_000_000.0,   maker_fee: 0.0008, taker_fee: 0.0010 },  // VIP 2
                VolumeTier { min_volume_30d: 20_000_000.0,  maker_fee: 0.0004, taker_fee: 0.0006 },  // VIP 3
                VolumeTier { min_volume_30d: 75_000_000.0,  maker_fee: 0.0004, taker_fee: 0.00052 }, // VIP 4
                VolumeTier { min_volume_30d: 150_000_000.0, maker_fee: 0.00025, taker_fee: 0.00031 }, // VIP 5
                VolumeTier { min_volume_30d: 400_000_000.0, maker_fee: 0.0002, taker_fee: 0.00029 }, // VIP 6
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.00001,
            max_order_size: 500.0,
            latency_mean_ms: 5.0,
            latency_std_ms: 2.0,
            max_orders_per_second: 10.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0001,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Binance futures caps at 125x on BTC/ETH
            // majors (lower for altcoins, and new accounts ramp up from 20x
            // over their first ~60 days) -- 125x is the well-known headline
            // figure for this platform's single "binance" preset.
            max_leverage: 125.0,
        }
    }

    /// Create Binance.US exchange configuration
    /// Updated 2026: Tier 0 pairs have 0% maker / 0.01% taker
    /// Tier I pairs have volume-based fees (much higher than old structure)
    pub fn binance_us() -> Self {
        Self {
            exchange: "binance_us".to_string(),
            maker_fee: 0.0038,  // 0.38% base (Tier I, VIP 1)
            taker_fee: 0.0057,  // 0.57% base (Tier I, VIP 1)
            volume_tiers: Self::binance_us_fee_tiers(),
            starting_30d_volume: 0.0,
            // Binance.US BTC/USD min: 0.00001 BTC
            min_order_size: 0.00001,
            max_order_size: 500.0,
            // Binance.US: Low latency (~10ms from US)
            latency_mean_ms: 10.0,
            latency_std_ms: 3.0,
            // Binance.US rate limits: 10 orders/sec for Spot
            max_orders_per_second: 10.0,
            rate_limit_cooldown_ms: 1000,
            // Spot only - no funding
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Binance.US offers no margin/futures product -- materially
            // more restricted than global Binance.
            max_leverage: 10.0,
        }
    }

    /// Binance.US fee tiers (as of 2026) - Tier I pairs only
    /// Tier 0 pairs (BTC, ETH, etc.) have 0% maker / 0.01% taker regardless of volume
    /// Note: These are MUCH higher than the old 0.10% flat structure
    pub fn binance_us_fee_tiers() -> Vec<VolumeTier> {
        vec![
            VolumeTier { min_volume_30d: 0.0,          maker_fee: 0.0038, taker_fee: 0.0057 },   // VIP 1: <$10K
            VolumeTier { min_volume_30d: 10_000.0,     maker_fee: 0.002375, taker_fee: 0.0038 }, // VIP 2: $10K-$50K
            VolumeTier { min_volume_30d: 50_000.0,     maker_fee: 0.001425, taker_fee: 0.002375 }, // VIP 3: $50K-$100K
            VolumeTier { min_volume_30d: 100_000.0,    maker_fee: 0.00095, taker_fee: 0.0019 },  // VIP 4: $100K-$1M
            VolumeTier { min_volume_30d: 1_000_000.0,  maker_fee: 0.00076, taker_fee: 0.00171 }, // VIP 5: $1M-$20M
            VolumeTier { min_volume_30d: 20_000_000.0, maker_fee: 0.000475, taker_fee: 0.001425 }, // VIP 6: $20M-$100M
            VolumeTier { min_volume_30d: 100_000_000.0, maker_fee: 0.00019, taker_fee: 0.00095 }, // VIP 7: $100M-$300M
        ]
    }
    
    /// Binance.US Tier 0 pairs (BTC, ETH, major coins) - FREE maker fees!
    /// Use this for backtesting BTC/USD on Binance.US
    pub fn binance_us_tier0() -> Self {
        Self {
            exchange: "binance_us_tier0".to_string(),
            maker_fee: 0.0000,  // 0% maker - FREE!
            taker_fee: 0.0001,  // 0.01% taker
            volume_tiers: vec![
                // Tier 0 has flat fees regardless of volume
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0000, taker_fee: 0.0001 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.00001,
            max_order_size: 500.0,
            latency_mean_ms: 10.0,
            latency_std_ms: 3.0,
            max_orders_per_second: 10.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            max_leverage: 10.0,
        }
    }

    /// Binance.US with BNB fee discount (5% off, not 25%)
    pub fn binance_us_bnb() -> Self {
        let mut config = Self::binance_us();
        config.maker_fee = 0.00361;  // 0.361% with 5% BNB discount
        config.taker_fee = 0.005415; // 0.5415% with 5% BNB discount
        // Apply 5% discount to all tiers
        config.volume_tiers = vec![
            VolumeTier { min_volume_30d: 0.0,          maker_fee: 0.00361, taker_fee: 0.005415 },
            VolumeTier { min_volume_30d: 10_000.0,     maker_fee: 0.0022563, taker_fee: 0.00361 },
            VolumeTier { min_volume_30d: 50_000.0,     maker_fee: 0.0013538, taker_fee: 0.0022563 },
            VolumeTier { min_volume_30d: 100_000.0,    maker_fee: 0.0009025, taker_fee: 0.001805 },
            VolumeTier { min_volume_30d: 1_000_000.0,  maker_fee: 0.000722, taker_fee: 0.0016245 },
            VolumeTier { min_volume_30d: 20_000_000.0, maker_fee: 0.00045125, taker_fee: 0.0013538 },
            VolumeTier { min_volume_30d: 100_000_000.0, maker_fee: 0.0001805, taker_fee: 0.0009025 },
        ];
        config
    }
    
    /// Create Coinbase Advanced Trade configuration (spot)
    /// Updated 2026: fees are significantly higher at low volumes than the old Pro structure
    pub fn coinbase() -> Self {
        Self {
            exchange: "coinbase".to_string(),
            maker_fee: 0.0060,  // 0.60% at lowest tier
            taker_fee: 0.0120,  // 1.20% at lowest tier
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,           maker_fee: 0.0060, taker_fee: 0.0120 },
                VolumeTier { min_volume_30d: 10_000.0,      maker_fee: 0.0040, taker_fee: 0.0080 },
                VolumeTier { min_volume_30d: 25_000.0,      maker_fee: 0.0025, taker_fee: 0.0050 },
                VolumeTier { min_volume_30d: 75_000.0,      maker_fee: 0.00125, taker_fee: 0.0025 },
                VolumeTier { min_volume_30d: 250_000.0,     maker_fee: 0.00075, taker_fee: 0.0015 },
                VolumeTier { min_volume_30d: 500_000.0,     maker_fee: 0.0006, taker_fee: 0.00125 },
                VolumeTier { min_volume_30d: 1_000_000.0,   maker_fee: 0.0005, taker_fee: 0.0010 },
                VolumeTier { min_volume_30d: 5_000_000.0,   maker_fee: 0.0004, taker_fee: 0.00085 },
                VolumeTier { min_volume_30d: 10_000_000.0,  maker_fee: 0.00025, taker_fee: 0.00065 },
                VolumeTier { min_volume_30d: 20_000_000.0,  maker_fee: 0.0001, taker_fee: 0.0005 },
                VolumeTier { min_volume_30d: 50_000_000.0,  maker_fee: 0.0000, taker_fee: 0.00035 },
                VolumeTier { min_volume_30d: 100_000_000.0, maker_fee: 0.0000, taker_fee: 0.00025 },
                VolumeTier { min_volume_30d: 250_000_000.0, maker_fee: 0.0000, taker_fee: 0.0002 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.0001,
            max_order_size: 200.0,
            latency_mean_ms: 20.0,
            latency_std_ms: 8.0,
            max_orders_per_second: 10.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Coinbase's core spot exchange (what this
            // preset's fees model) genuinely offers no leverage -- US spot
            // crypto leverage is prohibited for most investors. A separate
            // Coinbase Advanced/Derivatives product does offer perpetual
            // futures up to 25x-50x, but that's a different product from
            // the spot fee schedule modeled here.
            max_leverage: 1.0,
        }
    }

    /// Create DeepBook (SUI CLOB) exchange configuration
    /// DeepBook has very competitive fees for market makers
    pub fn deepbook() -> Self {
        Self {
            exchange: "deepbook".to_string(),
            maker_fee: 0.0005,  // 5 bps maker - very competitive
            taker_fee: 0.0010,  // 10 bps taker
            volume_tiers: vec![
                // DeepBook has flat fees (no volume tiers currently)
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0005, taker_fee: 0.0010 },
            ],
            starting_30d_volume: 0.0,
            // DeepBook minimum lot sizes (SUI-based)
            min_order_size: 0.001,
            max_order_size: 1000.0,
            // DeepBook on SUI: Very fast on-chain (~400ms finality)
            latency_mean_ms: 400.0,
            latency_std_ms: 100.0,
            // No traditional rate limits (gas limited)
            max_orders_per_second: 100.0,
            rate_limit_cooldown_ms: 100,
            // No funding on spot CLOB
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Spot CLOB -- no margin/leverage product.
            max_leverage: 1.0,
        }
    }

    /// Create Cetus (SUI AMM) exchange configuration
    pub fn cetus() -> Self {
        Self {
            exchange: "cetus".to_string(),
            maker_fee: 0.0030,  // 30 bps - AMM pool fee
            taker_fee: 0.0030,  // 30 bps - same for AMM
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0030, taker_fee: 0.0030 },
            ],
            starting_30d_volume: 0.0,
            // Cetus minimum swap sizes
            min_order_size: 0.001,
            max_order_size: 1000.0,
            // Cetus on SUI: Similar to DeepBook
            latency_mean_ms: 400.0,
            latency_std_ms: 100.0,
            // No traditional rate limits (gas limited)
            max_orders_per_second: 100.0,
            rate_limit_cooldown_ms: 100,
            // No funding on AMM
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // AMM pool swaps -- no margin/leverage product.
            max_leverage: 1.0,
        }
    }

    /// Create Bitfinex exchange configuration (spot)
    ///
    /// Verified 2026-07 via bitfinex.com/fees: Bitfinex eliminated ALL maker and
    /// taker fees for spot and margin trading on 2025-12-17 (no volume tiers, no
    /// token-holding conditions). NOTE: backtests over pre-Dec-2025 data will
    /// understate historical costs (the old schedule was 0.10% maker / 0.20%
    /// taker at entry tier); override fees manually for historical accuracy.
    pub fn bitfinex() -> Self {
        Self {
            exchange: "bitfinex".to_string(),
            maker_fee: 0.0,
            taker_fee: 0.0,
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0, taker_fee: 0.0 },
            ],
            starting_30d_volume: 0.0,
            // Bitfinex min order ≈ $5 equivalent
            min_order_size: 0.0001,
            max_order_size: 500.0,
            // Bitfinex REST: ~15-30ms, Websocket: ~10ms
            latency_mean_ms: 20.0,
            latency_std_ms: 8.0,
            max_orders_per_second: 10.0,
            rate_limit_cooldown_ms: 1000,
            // Spot — no funding
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Bitfinex spot margin trading caps at 10x
            // (its separate derivatives product goes to 100x, a different
            // product from the spot fee schedule modeled here).
            max_leverage: 10.0,
        }
    }

    /// Create Bitstamp exchange configuration (spot)
    ///
    /// Verified 2026-07 against the published "Bitstamp by Robinhood" fee
    /// schedule: 0.30% maker / 0.40% taker below $10K 30-day volume, stepping
    /// down to 0.00% maker / 0.03% taker above $1B. Anchor points confirmed:
    /// entry 0.30/0.40, $5M tier 0.03/0.12, top tier 0.00/0.03.
    pub fn bitstamp() -> Self {
        Self {
            exchange: "bitstamp".to_string(),
            maker_fee: 0.0030,  // 0.30% at lowest tier
            taker_fee: 0.0040,  // 0.40% at lowest tier
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,             maker_fee: 0.0030,  taker_fee: 0.0040 },
                VolumeTier { min_volume_30d: 10_000.0,        maker_fee: 0.0020,  taker_fee: 0.0030 },
                VolumeTier { min_volume_30d: 100_000.0,       maker_fee: 0.0010,  taker_fee: 0.0020 },
                VolumeTier { min_volume_30d: 500_000.0,       maker_fee: 0.0008,  taker_fee: 0.0018 },
                VolumeTier { min_volume_30d: 1_500_000.0,     maker_fee: 0.0006,  taker_fee: 0.0016 },
                VolumeTier { min_volume_30d: 5_000_000.0,     maker_fee: 0.0003,  taker_fee: 0.0012 },
                VolumeTier { min_volume_30d: 20_000_000.0,    maker_fee: 0.0002,  taker_fee: 0.0010 },
                VolumeTier { min_volume_30d: 50_000_000.0,    maker_fee: 0.0001,  taker_fee: 0.0008 },
                VolumeTier { min_volume_30d: 100_000_000.0,   maker_fee: 0.00005, taker_fee: 0.0006 },
                VolumeTier { min_volume_30d: 500_000_000.0,   maker_fee: 0.0,     taker_fee: 0.0005 },
                VolumeTier { min_volume_30d: 1_000_000_000.0, maker_fee: 0.0,     taker_fee: 0.0003 },
            ],
            starting_30d_volume: 0.0,
            // Bitstamp min order ≈ $10 equivalent
            min_order_size: 0.0002,
            max_order_size: 200.0,
            // Bitstamp REST: ~30-60ms (EU-based)
            latency_mean_ms: 40.0,
            latency_std_ms: 15.0,
            // Bitstamp rate limit: 400 requests per second (burst); orders lower
            max_orders_per_second: 8.0,
            rate_limit_cooldown_ms: 1000,
            // Spot — no funding
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: sources conflict on whether Bitstamp
            // currently offers any spot margin/leverage at all (some say no
            // margin/futures product exists; others cite up to 5x). Kept
            // conservative given the ambiguity.
            max_leverage: 2.0,
        }
    }

    /// Create Bybit exchange configuration (spot)
    pub fn bybit() -> Self {
        Self {
            exchange: "bybit".to_string(),
            maker_fee: 0.0010,  // 0.10% at $0 volume
            taker_fee: 0.0010,  // 0.10% at $0 volume
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,          maker_fee: 0.0010, taker_fee: 0.0010 },
                VolumeTier { min_volume_30d: 100_000.0,    maker_fee: 0.0006, taker_fee: 0.0008 },
                VolumeTier { min_volume_30d: 500_000.0,    maker_fee: 0.0004, taker_fee: 0.0006 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.00001,
            max_order_size: 500.0,
            latency_mean_ms: 8.0,
            latency_std_ms: 3.0,
            max_orders_per_second: 20.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Bybit caps at 100x on BTC/ETH majors (lower,
            // risk-tiered down for larger positions and altcoins).
            max_leverage: 100.0,
        }
    }

    /// Create OKX exchange configuration (spot)
    pub fn okx() -> Self {
        Self {
            exchange: "okex".to_string(),
            maker_fee: 0.0008,  // 0.08% at $0 volume
            taker_fee: 0.0010,  // 0.10% at $0 volume
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,          maker_fee: 0.0008, taker_fee: 0.0010 },
                VolumeTier { min_volume_30d: 100_000.0,    maker_fee: 0.0006, taker_fee: 0.0008 },
                VolumeTier { min_volume_30d: 1_000_000.0,  maker_fee: 0.0004, taker_fee: 0.0006 },
                VolumeTier { min_volume_30d: 5_000_000.0,  maker_fee: 0.0002, taker_fee: 0.0005 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.0001,
            max_order_size: 300.0,
            latency_mean_ms: 8.0,
            latency_std_ms: 3.0,
            max_orders_per_second: 20.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: OKX caps at 100x-125x on BTC/ETH majors
            // (lower for altcoins); 100x is the conservative-side figure.
            max_leverage: 100.0,
        }
    }

    /// Create Gemini ActiveTrader configuration (spot)
    /// Updated 2026: Gemini restructured fees — much higher at low volume, competitive at high volume
    pub fn gemini() -> Self {
        Self {
            exchange: "gemini".to_string(),
            maker_fee: 0.0060,  // 0.60% at lowest tier (ActiveTrader)
            taker_fee: 0.0120,  // 1.20% at lowest tier (ActiveTrader)
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0,           maker_fee: 0.0060, taker_fee: 0.0120 },
                VolumeTier { min_volume_30d: 10_000.0,      maker_fee: 0.0040, taker_fee: 0.0080 },
                VolumeTier { min_volume_30d: 25_000.0,      maker_fee: 0.0025, taker_fee: 0.0050 },
                VolumeTier { min_volume_30d: 75_000.0,      maker_fee: 0.00125, taker_fee: 0.0025 },
                VolumeTier { min_volume_30d: 250_000.0,     maker_fee: 0.00075, taker_fee: 0.0015 },
                VolumeTier { min_volume_30d: 500_000.0,     maker_fee: 0.0006, taker_fee: 0.00125 },
                VolumeTier { min_volume_30d: 1_000_000.0,   maker_fee: 0.0005, taker_fee: 0.0010 },
                VolumeTier { min_volume_30d: 5_000_000.0,   maker_fee: 0.0004, taker_fee: 0.00085 },
                VolumeTier { min_volume_30d: 10_000_000.0,  maker_fee: 0.00025, taker_fee: 0.00065 },
                VolumeTier { min_volume_30d: 20_000_000.0,  maker_fee: 0.0001, taker_fee: 0.0005 },
                VolumeTier { min_volume_30d: 50_000_000.0,  maker_fee: 0.0000, taker_fee: 0.00035 },
                VolumeTier { min_volume_30d: 100_000_000.0, maker_fee: 0.0000, taker_fee: 0.00025 },
                VolumeTier { min_volume_30d: 250_000_000.0, maker_fee: 0.0000, taker_fee: 0.0002 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.00001,
            max_order_size: 100.0,
            latency_mean_ms: 20.0,
            latency_std_ms: 8.0,
            max_orders_per_second: 5.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Gemini margin trading caps at 5x (long-only
            // currently, curated pairs, ECP-eligible clients only).
            max_leverage: 5.0,
        }
    }

    /// Create Deribit exchange configuration (perpetuals/options)
    pub fn deribit() -> Self {
        Self {
            exchange: "deribit".to_string(),
            maker_fee: 0.0002,  // 0.02% maker
            taker_fee: 0.0005,  // 0.05% taker
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0002, taker_fee: 0.0005 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.001,
            max_order_size: 500.0,
            latency_mean_ms: 5.0,
            latency_std_ms: 2.0,
            max_orders_per_second: 20.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0001,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: Deribit perpetuals/futures cap at 50x --
            // deliberately conservative vs. peers since Deribit's futures
            // exist mainly to delta-hedge options books, not for outright
            // leveraged speculation. Options leverage is variable/embedded
            // in the premium, not a linear multiple like futures.
            max_leverage: 50.0,
        }
    }

    /// Create OANDA forex configuration (spread-based, no commission)
    /// OANDA charges no commission on standard accounts — cost is embedded in the spread.
    /// Typical EUR/USD spread: ~1.0-1.3 pips (0.00010-0.00013 of price).
    /// We model this as a taker fee equivalent to the half-spread cost.
    pub fn oanda() -> Self {
        Self {
            exchange: "oanda".to_string(),
            // No explicit commission — spread cost modeled as taker fee
            // ~1.2 pip typical spread on EUR/USD = 0.012% round-trip equivalent
            maker_fee: 0.0,       // No maker rebate in forex spread model
            taker_fee: 0.00006,   // Half-spread cost (~0.6 pip per side on majors)
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0, taker_fee: 0.00006 },
            ],
            starting_30d_volume: 0.0,
            // Forex: standard lot = 100,000 units, micro lot = 1,000
            min_order_size: 1.0,       // 1 unit minimum
            max_order_size: 10_000_000.0, // 10M units
            // OANDA REST: ~20-50ms
            latency_mean_ms: 30.0,
            latency_std_ms: 10.0,
            max_orders_per_second: 30.0,
            rate_limit_cooldown_ms: 500,
            // No funding rate — forex uses overnight swap rates instead
            funding_rate_8h: 0.0,
            funding_interval_ms: 24 * 60 * 60 * 1000, // Daily rollover
            // Overnight swap/rollover fee varies by pair; use approximate average
            // Actual swap depends on interest rate differential between currencies
            swap_fee_daily: 0.0,  // Set to 0 — handled separately by forex swap logic
            // Verified 2026-07: CFTC/NFA caps US retail forex leverage at
            // 50:1 on major pairs, 20:1 on others; Oanda's US entity follows
            // this exactly. Lower under ESMA/ASIC rules, higher in some
            // offshore regimes -- 50x is the correct figure for Oanda's US arm.
            max_leverage: 50.0,
        }
    }

    /// Create Alpaca configuration for US equities
    /// Commission-free stock trading. Only costs are SEC and FINRA regulatory fees on sells.
    /// SEC fee: ~$8.00 per $1M of sell proceeds (0.0008%)
    /// FINRA TAF: $0.000119 per share on sells (capped at $5.95/trade)
    pub fn alpaca() -> Self {
        Self {
            exchange: "alpaca".to_string(),
            maker_fee: 0.0,        // Commission-free
            taker_fee: 0.000008,   // SEC fee ~$8/$1M on sells only; approximate as tiny taker cost
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0, taker_fee: 0.000008 },
            ],
            starting_30d_volume: 0.0,
            // Equities: fractional shares supported (min $1)
            min_order_size: 1.0,       // 1 share (or $1 for fractional)
            max_order_size: 1_000_000.0,
            // Alpaca REST API: ~30-50ms typical
            latency_mean_ms: 40.0,
            latency_std_ms: 15.0,
            // Alpaca rate limit: 200 req/min = ~3.3/sec
            max_orders_per_second: 3.0,
            rate_limit_cooldown_ms: 2000,
            // No funding for equities (no perpetuals)
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            swap_fee_daily: 0.0,
            // Verified 2026-07: standard Reg-T margin gives 2x buying power;
            // Pattern-Day-Trader-flagged accounts get up to 4x intraday
            // buying power. 4x is the representative figure for this
            // platform's backtest-only simulated leverage.
            max_leverage: 4.0,
        }
    }

    /// Create Breakout prop firm configuration
    /// Breakout routes through Kraken but has its own fee structure + daily swap fee.
    pub fn breakout() -> Self {
        Self {
            exchange: "breakout".to_string(),
            maker_fee: 0.0004,  // 0.04% maker
            taker_fee: 0.0004,  // 0.04% taker
            volume_tiers: vec![
                VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0004, taker_fee: 0.0004 },
            ],
            starting_30d_volume: 0.0,
            min_order_size: 0.0001,
            max_order_size: 100.0,
            latency_mean_ms: 15.0,  // Routes through Kraken
            latency_std_ms: 5.0,
            max_orders_per_second: 15.0,
            rate_limit_cooldown_ms: 1000,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            // Breakout charges 0.033% daily swap on overnight positions
            swap_fee_daily: 0.00033,
            // Verified 2026-07: Breakout (crypto prop firm) funded accounts
            // provide 5x buying power on BTC/ETH, 2x on altcoins -- 5x is
            // the representative figure for this crypto-focused preset.
            max_leverage: 5.0,
        }
    }

    /// Create configuration for a specific exchange by name
    /// Supports all major CEX exchanges, prop firms, and DEX protocols.
    pub fn for_exchange(exchange: &str) -> Self {
        match exchange.to_lowercase().as_str() {
            "massive" | "exchange" => Self::kraken(), // Massive aggregates multiple venues; use kraken-like fee model as baseline
            "kraken" | "krkn" => Self::kraken(),
            "kraken_colocated" | "kraken-colocated" => Self::kraken_colocated(),
            "kraken_established" | "kraken-established" => Self::kraken_established(),
            "kraken_colocated_established" | "kraken-colocated-established" => Self::kraken_colocated_established(),
            "binance" => Self::binance(),
            "binance_us" | "binance-us" | "binanceus" | "bnus" => Self::binance_us(),
            "binance_us_bnb" | "binance-us-bnb" => Self::binance_us_bnb(),
            "coinbase" | "cbse" => Self::coinbase(),
            "bitfinex" | "bfnx" => Self::bitfinex(),
            "bitstamp" | "bstp" => Self::bitstamp(),
            "bybit" | "bybt" => Self::bybit(),
            "okex" | "okx" => Self::okx(),
            "gemini" => Self::gemini(),
            "deribit" => Self::deribit(),
            // Forex: "fx" is the exchange id the platform's forex asset class
            // uses (Polygon feed) — costs follow the OANDA spread model.
            "oanda" | "oanda_forex" | "forex" | "fx" => Self::oanda(),
            // US equities: "consolidated" (NBBO) and the individual listing
            // venues all execute through Alpaca's commission-free model.
            "alpaca" | "alpaca_stocks" | "stocks" | "equities"
            | "consolidated" | "nasdaq" | "nyse" | "nyse_arca" | "nyse-arca" | "amex" => Self::alpaca(),
            "breakout" => Self::breakout(),
            "deepbook" | "sui" => Self::deepbook(),
            "cetus" => Self::cetus(),
            _ => {
                // Unknown exchange — surface it loudly so silent fee mispricing
                // can't recur (this generic fallback was previously applied to
                // whole asset classes without any signal).
                eprintln!(
                    "[config] WARNING: no fee preset for exchange '{}' — applying generic fallback \
                     (0.10% maker / 0.15% taker, no volume tiers). Add a preset in \
                     ExchangeFeeConfig::for_exchange if this venue is traded regularly.",
                    exchange
                );
                // Default to generic conservative fees with typical CEX parameters
                Self {
                    exchange: exchange.to_string(),
                    maker_fee: 0.0010,
                    taker_fee: 0.0015,
                    volume_tiers: vec![],
                    starting_30d_volume: 0.0,
                    // Conservative defaults for unknown exchanges
                    min_order_size: 0.0001,
                    max_order_size: 100.0,
                    latency_mean_ms: 50.0,
                    latency_std_ms: 20.0,
                    max_orders_per_second: 5.0,
                    rate_limit_cooldown_ms: 2000,
                    funding_rate_8h: 0.0001,
                    funding_interval_ms: 8 * 60 * 60 * 1000,
                    swap_fee_daily: 0.0,
                    // Conservative fallback for an unrecognized venue -- matches
                    // the platform's pre-per-exchange-cap global default, so an
                    // unknown exchange keeps prior behavior rather than becoming
                    // more permissive or more restrictive by surprise.
                    max_leverage: 3.0,
                }
            }
        }
    }

    /// Cap a requested leverage against this exchange's real limit, floored
    /// at 1.0 (there is no such thing as <1x margin in this platform's
    /// model). Returns `(effective_leverage, was_capped)` so callers can log
    /// when a request got reduced. Shared by every leverage-resolution call
    /// site (job-level fixed leverage in `program::worker::resolve_leverage`,
    /// and per-chromosome genome-evolved leverage in the GA fitness paths) so
    /// the cap logic exists in exactly one place.
    pub fn resolve_leverage(&self, requested: f64) -> (f64, bool) {
        let capped = requested.min(self.max_leverage).max(1.0);
        (capped, capped < requested)
    }
}

/// Volume-based fee tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeTier {
    pub min_volume_30d: f64,      // Minimum 30-day volume in USD
    pub maker_fee: f64,
    pub taker_fee: f64,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub default_timeframe: String, // e.g., "1d", "1h", "5m"
    pub data_sources: Vec<String>, // e.g., ["yahoo", "alpha_vantage"]
    pub cache_enabled: bool,
    pub cache_duration_hours: u32,
    pub gap_detection: GapDetectionConfig,
    pub market_impact: MarketImpactConfig,
}

/// Gap detection and handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapDetectionConfig {
    /// Enable gap detection
    pub enabled: bool,
    /// Expected data frequency in seconds (e.g., 60 for 1-minute data)
    pub expected_frequency_seconds: i64,
    /// Gap threshold multiplier (gap = frequency * threshold)
    pub gap_threshold_multiplier: f64,
    /// Gap filling strategy
    pub fill_strategy: GapFillStrategy,
    /// Warn if gap exceeds this duration (seconds)
    pub warning_threshold_seconds: i64,
    /// Fail backtest if gap exceeds this duration (seconds)
    pub critical_threshold_seconds: Option<i64>,
    /// Order stream detection (for market making strategies)
    pub order_detection: Option<OrderDetectionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GapFillStrategy {
    /// Do nothing - process data as-is
    None,
    /// Forward fill - use last known values
    ForwardFill,
    /// Linear interpolation between points
    LinearInterpolation,
    /// Mark gap periods and exclude from analysis
    ExcludeGaps,
}

/// Market impact modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactConfig {
    /// Enable market impact modeling
    pub enabled: bool,
    /// Impact model type
    pub model: MarketImpactModel,
    /// Minimum order size ratio to trigger impact (order_size / avg_volume)
    pub min_impact_ratio: f64,
    /// Linear impact coefficient (basis points per 1% of volume)
    pub linear_coefficient: f64,
    /// Square root impact coefficient
    pub sqrt_coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketImpactModel {
    /// No market impact
    None,
    /// Linear model: impact = coefficient * (order_size / volume)
    Linear,
    /// Square root model: impact = coefficient * sqrt(order_size / volume)
    SquareRoot,
    /// Almgren-Chriss model (more sophisticated)
    AlmgrenChriss,
}

/// Order stream detection configuration (for market making strategies)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderDetectionConfig {
    /// Enable order stream analysis
    pub enabled: bool,
    /// Expected quote update interval in milliseconds (your strategy's quote refresh rate)
    /// For co-located Kraken: 10-100ms, typical HFT market making: 50-200ms
    pub expected_quote_interval_ms: u64,
    /// Require bid-ask pairs (both sides present)
    pub require_paired_quotes: bool,
    /// Tolerance window for pairing (milliseconds)
    pub pairing_window_ms: u64,
    /// Maximum gap before warning (in number of missing quote cycles)
    pub warning_missing_cycles: u64,
}

impl Default for OrderDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            expected_quote_interval_ms: 50, // 50ms for co-located market making
            require_paired_quotes: true,
            pairing_window_ms: 20, // 20ms tolerance for bid-ask pairing
            warning_missing_cycles: 10, // Warn after 500ms of missing quotes (10 * 50ms)
        }
    }
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            expected_frequency_seconds: 60, // 1-minute data
            gap_threshold_multiplier: 2.0,  // gaps > 2x frequency
            fill_strategy: GapFillStrategy::ForwardFill,
            warning_threshold_seconds: 300, // 5 minutes
            critical_threshold_seconds: Some(3600), // 1 hour
            order_detection: None, // Disabled by default, enable for market making
        }
    }
}

impl Default for MarketImpactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: MarketImpactModel::SquareRoot,
            min_impact_ratio: 0.01, // 1% of volume
            linear_coefficient: 10.0, // 10 bps per 1% of volume
            sqrt_coefficient: 5.0,
        }
    }
}

/// Output and reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub results_dir: String,
    pub reports_dir: String,
    pub charts_enabled: bool,
    pub export_formats: Vec<String>, // e.g., ["json", "csv", "html"]
}

/// Monte Carlo simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of Monte Carlo simulation runs
    pub num_simulations: usize,
    /// Confidence level for percentile calculations
    pub confidence_level: f64,
    /// Random seed for reproducibility (optional)
    pub random_seed: Option<u64>,
    /// Bootstrap sample size as fraction of original data
    pub bootstrap_fraction: f64,
    /// Whether to bootstrap trade returns or equity curve
    pub bootstrap_method: BootstrapMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BootstrapMethod {
    /// Bootstrap individual trade returns
    #[serde(rename = "trades")]
    TradeReturns,
    /// Bootstrap blocks of equity curve data
    #[serde(rename = "blocks")]
    BlockBootstrap,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_simulations: 500,
            confidence_level: 0.95,
            random_seed: None,
            bootstrap_fraction: 1.0,
            bootstrap_method: BootstrapMethod::TradeReturns,
        }
    }
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_trades: bool,
    pub include_charts: bool,
    pub template_dir: Option<PathBuf>,
    pub custom_css: Option<String>,
    /// Whether to persist report metadata to database (enabled by default for historical tracking)
    pub persist_to_database: bool,
    /// Database connection for persistence (if enabled, reads from DATABASE_URL env var)
    pub database_url: Option<String>,
    /// User or system generating the report
    pub generated_by: Option<String>,
    /// Source of report generation
    pub generation_source: String,
}

/// Supported report formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFormat {
    Html,
    Json,
    Csv,
    Pdf,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportFormat::Html => write!(f, "html"),
            ReportFormat::Json => write!(f, "json"),
            ReportFormat::Csv => write!(f, "csv"),
            ReportFormat::Pdf => write!(f, "pdf"),
        }
    }
}

/// Optimization mode for genetic algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMode {
    /// Single weighted-sum fitness (default).
    SingleObjective,
    /// Multi-objective NSGA-II with named objectives.
    MultiObjective {
        objectives: Vec<ObjectiveDef>,
    },
}

impl Default for OptimizationMode {
    fn default() -> Self { Self::SingleObjective }
}

/// Definition of an objective for multi-objective optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveDef {
    pub field: String,
    pub direction: ObjectiveDirection,
}

/// Maximize or minimize.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

/// Genetic algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    
    /// Optimization mode: single-objective (default) or multi-objective.
    #[serde(default)]
    pub optimization_mode: OptimizationMode,
    
    // Lazy evaluation parameters for Monte Carlo consistency checking
    #[serde(default)]
    pub use_monte_carlo_fitness: bool,     // Enable MC consistency in fitness
    #[serde(default = "default_mc_sample_size")]
    pub mc_sample_size: usize,             // Number of MC runs for fitness (50-100)
    #[serde(default = "default_top_percentile")]
    pub top_percentile_threshold: f64,     // Only evaluate top N% (e.g., 0.10 for top 10%)
    
    // Performance: force sequential evaluation (no rayon parallelism).
    // Required for Python strategies where the GIL serialises all threads.
    #[serde(default)]
    pub force_sequential: bool,

    // Early-stop: number of generations without improvement before triggering
    // convergence restart (or full stop if restarts exhausted).  None = default (10).
    #[serde(default)]
    pub convergence_threshold: Option<usize>,

    // Diversity preservation settings (prevents strategy crowding)
    #[serde(default)]
    pub enable_fitness_sharing: bool,      // Penalize similar individuals in population
    #[serde(default = "default_sharing_radius")]
    pub sharing_radius: f64,               // Distance threshold for fitness sharing
    #[serde(default)]
    pub enable_crowding_penalty: bool,     // Penalize similarity to deployed strategies
    #[serde(default = "default_crowding_weight")]
    pub crowding_weight: f64,              // Weight of crowding penalty (0.0-1.0)
    #[serde(default = "default_min_behavioral_distance")]
    pub min_behavioral_distance: f64,      // Min required distance from deployed strategies

    /// Fixed RNG seed for reproducible runs. When set, both `GeneticOptimizer`
    /// and `AdaptiveGeneticOptimizer` seed their `StdRng` from this value
    /// instead of `from_entropy()`. Primarily used to run the same GA twice
    /// with two different explicit seeds and compare the resulting best
    /// parameters -- a large divergence between runs is a concrete signal
    /// that the "best" parameters found are fragile/overfit to noise in the
    /// fitness landscape rather than a real, stable optimum. `None` (default)
    /// preserves the existing non-deterministic behavior.
    #[serde(default)]
    pub random_seed: Option<u64>,

    /// When `true` (and `random_seed` is set), the caller runs the GA a
    /// second time with a different fixed seed
    /// (`random_seed.wrapping_add(SEED_STABILITY_OFFSET)` -- see
    /// `python_validation.rs`) purely to compare the two runs' best
    /// parameters via `Chromosome::distance()`. Doubles GA wall-clock cost,
    /// so it defaults to `false` and should only be enabled when overfitting-
    /// fragility evidence is worth the extra run (e.g. a final validation
    /// pass before deployment, not every exploratory GA call).
    #[serde(default)]
    pub check_seed_stability: bool,

    /// When `true`, every GA population-generation step (in
    /// `AdaptiveGeneticOptimizer::run()`, including per-window
    /// re-optimization inside walk-forward) draws each non-elite candidate
    /// via `Chromosome::random()` instead of tournament selection +
    /// crossover + mutation -- i.e. this becomes a same-budget, i.i.d.
    /// random-search control arm rather than a guided evolutionary search.
    /// Elite carryover (remembering the best chromosome seen so far) is
    /// left untouched so both arms share identical bookkeeping and only the
    /// search mechanism itself differs -- the one thing under test.
    ///
    /// Exists to falsify (or confirm) the standing assumption that guided
    /// GA search finds real, OOS-generalizing alpha rather than just
    /// overfitting noise faster than random sampling would. A job submitted
    /// with this set to `true` runs through the exact same production
    /// pipeline (walk-forward, DSR, everything) as a normal job -- compare
    /// its persisted OOS metrics against a paired guided run on the same
    /// asset/params/seed to get an apples-to-apples answer. `false` by
    /// default preserves all existing behavior.
    #[serde(default)]
    pub random_control: bool,

    /// Weight (0.0 = disabled, the default) for a parsimony/complexity
    /// penalty applied on top of the normal fitness score:
    /// `fitness -= chromosome.complexity() * complexity_penalty_weight`.
    /// `Chromosome::complexity()` defaults to `0.0` (no-op) and is overridden
    /// by `DynamicChromosome` to return its schema's parameter count --
    /// simpler strategies (fewer tunable parameters) are statistically more
    /// likely to reflect a real, broad effect rather than overfit noise.
    /// Kept separate from `FitnessWeights` (whose fields must sum to 1.0)
    /// since this is a penalty, not a weighted component of the score.
    #[serde(default)]
    pub complexity_penalty_weight: f64,
}

fn default_mc_sample_size() -> usize { 50 }
fn default_top_percentile() -> f64 { 0.10 }
fn default_sharing_radius() -> f64 { 0.15 }
fn default_crowding_weight() -> f64 { 0.3 }
fn default_min_behavioral_distance() -> f64 { 0.3 }

/// Configuration for order book reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookConfig {
    /// Maximum number of price levels to maintain on each side
    pub max_levels: Option<usize>,
    /// Whether to maintain detailed order information
    pub track_individual_orders: bool,
    /// Snapshot validation frequency (in events)
    pub snapshot_validation_frequency: Option<usize>,
    /// Whether to enable strict validation
    pub strict_validation: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub max_parallel: usize,
    pub retry_attempts: usize,
}

/// Strategy configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Unique strategy identifier
    pub id: String,
    /// Human-readable strategy name
    pub name: String,
    /// Strategy type/category
    pub strategy_type: String,
    /// Symbol to trade
    pub symbol: String,
    /// Exchange to trade on
    pub exchange: String,
    /// Strategy-specific parameters
    pub parameters: HashMap<String, ParameterValue>,
    /// Risk management settings
    pub risk_limits: RiskLimits,
    /// Whether the strategy is enabled
    pub enabled: bool,
}

/// Parameter value that can hold different types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Int(i64),       // Renamed from Integer for consistency
    Float(f64),
    String(String),
    Bool(bool),     // Renamed from Boolean for consistency
    Array(Vec<ParameterValue>), // Added for array support
}

impl ParameterValue {
    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    
    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get as array
    pub fn as_array(&self) -> Option<&Vec<ParameterValue>> {
        match self {
            ParameterValue::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ParameterValue::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Risk management limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size
    pub max_position_size: f64,
    /// Maximum order size
    pub max_order_size: f64,
    /// Maximum daily loss threshold
    pub max_daily_loss: f64,
    /// Maximum portfolio value at risk
    pub max_var: Option<f64>,
    /// Stop loss percentage
    pub stop_loss_pct: Option<f64>,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            database: "trading_platform".to_string(),
            username: "postgres".to_string(),
            password: "password".to_string(),
            max_connections: Some(10),
            connection_timeout: Some(30),
        }
    }
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            commission_rate: 0.001,     // 0.1% (fallback, use exchange_fees instead)
            slippage_bps: 2.0,          // 2 basis points for co-located
            initial_capital: 100000.0,  // $100,000
            max_position_size: Some(0.1), // 10%
            risk_free_rate: 0.02,       // 2%
            exchange_fees: Some(ExchangeFeeConfig::default()), // Kraken at $5M volume
            routing_mode: None,         // Single exchange by default
            latency_ms: 10,             // 10ms simulated latency (realistic for co-located)
            latency_jitter_pct: 0.2,    // 20% variation
            min_latency_ms: 1,          // Minimum 1ms even with negative jitter
            latency_spike_probability: 0.01, // 1% chance of latency spike
            lookback_window: 1000,
            optimization: PerformanceOptimization::default(),
            fill_volume_fraction: 0.10,
            synthetic_spread_bps: 5.0,
            synthetic_depth_levels: 5,
            adverse_selection: true,
            use_synthetic_book: true,
            dex_costs: None,
            slippage_model: SlippageModel::default(),
            leverage: default_leverage(),
            maintenance_margin_ratio: default_maintenance_margin_ratio(),
            liquidation_penalty_bps: default_liquidation_penalty_bps(),
            vol_target_annual: None,
            vol_target_lookback_bars: default_vol_target_lookback_bars(),
            vol_target_bars_per_year: default_vol_target_bars_per_year(),
            vol_target_min_scale: default_vol_target_min_scale(),
            vol_target_max_scale: default_vol_target_max_scale(),
        }
    }
}

impl ExchangeFeeConfig {
    /// Kraken exchange fee configuration starting at 0 volume (new market maker)
    /// Delegates to the main kraken() factory method
    pub fn kraken_default() -> Self {
        Self::kraken()
    }

    /// Coinbase Pro exchange fee configuration
    /// Delegates to the main coinbase() factory method
    pub fn coinbase_default() -> Self {
        Self::coinbase()
    }

    /// Get applicable fee rate based on volume and liquidity type
    pub fn get_fee_rate(&self, volume_30d: f64, is_maker: bool) -> f64 {
        // Create a fallback tier
        let fallback_tier = VolumeTier {
            min_volume_30d: 0.0,
            maker_fee: self.maker_fee,
            taker_fee: self.taker_fee,
        };

        // Find the highest applicable tier
        let applicable_tier = self.volume_tiers
            .iter()
            .filter(|tier| volume_30d >= tier.min_volume_30d)
            .max_by(|a, b| a.min_volume_30d.partial_cmp(&b.min_volume_30d).unwrap())
            .unwrap_or(&fallback_tier);

        if is_maker {
            applicable_tier.maker_fee
        } else {
            applicable_tier.taker_fee
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            default_timeframe: "1d".to_string(),
            data_sources: vec!["yahoo".to_string()],
            cache_enabled: true,
            cache_duration_hours: 24,
            gap_detection: GapDetectionConfig::default(),
            market_impact: MarketImpactConfig::default(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            results_dir: "./results".to_string(),
            reports_dir: "./reports".to_string(),
            charts_enabled: true,
            export_formats: vec!["json".to_string(), "csv".to_string()],
        }
    }
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./reports"),
            formats: vec![ReportFormat::Html, ReportFormat::Json],
            include_trades: true,
            include_charts: true,
            template_dir: None,
            custom_css: None,
            persist_to_database: false,
            database_url: None,
            generated_by: None,
            generation_source: "cli".to_string(),
        }
    }
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.15,
            optimization_mode: OptimizationMode::SingleObjective,
            use_monte_carlo_fitness: true,      // Enabled by default for robustness
            mc_sample_size: 50,                 // Fast MC for fitness eval
            top_percentile_threshold: 0.10,     // Only check top 10%
            force_sequential: true,             // Sequential by default (Python GIL safety)
            convergence_threshold: None,         // Use default (10 generations)
            // Diversity preservation (ENABLED by default to prevent strategy crowding)
            enable_fitness_sharing: true,       // Penalize similar individuals in population
            sharing_radius: 0.15,               // Sharing radius for intra-population diversity
            enable_crowding_penalty: true,      // Penalize similarity to deployed strategies globally
            crowding_weight: 0.3,               // Max 30% fitness reduction from crowding
            min_behavioral_distance: 0.3,       // Reject strategies < 30% different from deployed
            random_seed: None,                  // Non-deterministic by default
            check_seed_stability: false,        // Disabled by default (doubles GA cost)
            random_control: false,              // Disabled by default (guided search is normal behavior)
            complexity_penalty_weight: 0.0,      // Disabled by default (no behavior change)
        }
    }
}

impl Default for OrderBookConfig {
    fn default() -> Self {
        Self {
            max_levels: Some(100), // Keep top 100 levels by default
            track_individual_orders: true,
            snapshot_validation_frequency: Some(1000),
            strict_validation: true,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_parallel: 4,
            retry_attempts: 3,
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            name: "Default Strategy".to_string(),
            strategy_type: "basic".to_string(),
            symbol: "BTCUSD".to_string(),
            exchange: "default".to_string(),
            parameters: HashMap::new(),
            risk_limits: RiskLimits::default(),
            enabled: true,
        }
    }
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_size: 1.0,
            max_order_size: 0.1,
            max_daily_loss: 1000.0,
            max_var: None,
            stop_loss_pct: None,
        }
    }
}

impl BacktestConfig {
    /// Load configuration from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)?;
        
        let config: BacktestConfig = serde_yaml::from_str(&contents)?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to a YAML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let contents = serde_yaml::to_string(self)?;
        
        fs::write(path, contents)?;
        
        Ok(())
    }
    
    /// Create a default configuration file if it doesn't exist
    pub fn create_default_if_missing<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        
        if path.exists() {
            Self::from_file(path)
        } else {
            let default_config = Self::default();
            default_config.to_file(path)?;
            Ok(default_config)
        }
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate database config
        if self.database.host.is_empty() {
            return Err(ConfigError::MissingField {
                field: "database.host".to_string(),
            });
        }
        
        if self.database.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "database.port".to_string(),
                value: "0".to_string(),
                reason: "Port must be greater than 0".to_string(),
            });
        }
        
        // Validate trading config
        if self.trading.commission_rate < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "trading.commission_rate".to_string(),
                value: self.trading.commission_rate.to_string(),
                reason: "Commission rate cannot be negative".to_string(),
            });
        }
        
        if self.trading.initial_capital <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "trading.initial_capital".to_string(),
                value: self.trading.initial_capital.to_string(),
                reason: "Initial capital must be positive".to_string(),
            });
        }
        
        if let Some(max_pos) = self.trading.max_position_size {
            if max_pos <= 0.0 || max_pos > 1.0 {
                return Err(ConfigError::InvalidValue {
                    field: "trading.max_position_size".to_string(),
                    value: max_pos.to_string(),
                    reason: "Max position size must be between 0 and 1".to_string(),
                });
            }
        }
        
        // Validate data config
        if self.data.default_timeframe.is_empty() {
            return Err(ConfigError::MissingField {
                field: "data.default_timeframe".to_string(),
            });
        }
        
        if self.data.data_sources.is_empty() {
            return Err(ConfigError::MissingField {
                field: "data.data_sources".to_string(),
            });
        }
        
        // Validate output config
        if self.output.results_dir.is_empty() {
            return Err(ConfigError::MissingField {
                field: "output.results_dir".to_string(),
            });
        }
        
        if self.output.export_formats.is_empty() {
            return Err(ConfigError::MissingField {
                field: "output.export_formats".to_string(),
            });
        }
        
        Ok(())
    }
}

/// Analysis configuration for Monte Carlo and Walk-Forward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis mode: "development", "validation", "production"
    pub mode: AnalysisMode,
    /// Skip genetic optimization (used when worker already did it)
    pub skip_genetic_optimization: bool,
    /// Enable automatic Monte Carlo simulation
    pub auto_monte_carlo: bool,
    /// Number of Monte Carlo runs (default: 500)
    pub monte_carlo_runs: usize,
    /// Enable automatic walk-forward analysis  
    pub auto_walk_forward: bool,
    /// Walk-forward training window size in days
    pub wf_training_days: i64,
    /// Walk-forward test window size in days
    pub wf_test_days: i64,
    /// Walk-forward step size in days
    pub wf_step_days: i64,
    /// Walk-forward embargo gap between training and testing (days)
    pub wf_embargo_days: i64,
    /// Anchored walk-forward: training always starts from beginning of data
    pub wf_anchored: bool,
    /// Minimum backtest duration to trigger auto-analysis (days)
    pub min_duration_for_auto: i64,
    /// Minimum Sharpe ratio to trigger validation mode
    pub validation_sharpe_threshold: f64,
    /// Minimum profit factor to trigger validation mode
    pub validation_pf_threshold: f64,
    /// Minimum number of trades to trigger validation mode
    pub validation_min_trades: u64,
    /// GA seeding configuration for strategy optimization
    pub ga_seed_params: Option<GASeedParams>,
}

/// Baseline parameters to seed genetic algorithm population
/// Based on proven market-making parameters to give GA a good starting point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GASeedParams {
    /// Risk aversion parameter (e.g., 1.5)
    pub risk_aversion: f64,
    /// Order size percentage (e.g., 0.02 = 2%)
    pub order_size_pct: f64,
    /// Inventory target (e.g., 0.0 = neutral)
    pub inventory_target: f64,
    /// Window in milliseconds (e.g., 60000 = 1 minute)
    pub window_ms: u64,
    /// Min quote lifetime in milliseconds (e.g., 200)
    pub min_quote_lifetime_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
pub enum AnalysisMode {
    /// Fast development mode - basic backtest only
    #[serde(rename = "development")]
    Development,
    /// Validation mode - basic + Monte Carlo
    #[serde(rename = "validation")]
    Validation, 
    /// Production mode - full analysis (basic + MC + walk-forward)
    #[serde(rename = "production")]
    Production,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        // Default to Production mode with full analysis pipeline:
        // 1. Genetic optimization (mandatory before backtesting)
        // 2. Walk-forward analysis (out-of-sample validation)
        // 3. Monte Carlo simulation (parameter robustness)
        Self {
            mode: AnalysisMode::Production,
            skip_genetic_optimization: false,
            auto_monte_carlo: true,
            monte_carlo_runs: 500,
            auto_walk_forward: true,
            wf_training_days: 252,  // 1 year training
            wf_test_days: 63,       // 3 months testing
            wf_step_days: 21,       // 1 month rolling steps
            wf_embargo_days: 0,     // No embargo gap by default
            wf_anchored: false,     // Rolling mode by default
            min_duration_for_auto: 90, // Minimum 90 days for full analysis
            validation_sharpe_threshold: 1.2,
            validation_pf_threshold: 1.4,
            validation_min_trades: 50,
            ga_seed_params: Some(GASeedParams {
                risk_aversion: 1.5,
                order_size_pct: 0.02,          // 2%
                inventory_target: 0.0,          // Neutral
                window_ms: 60_000,              // 1 minute
                min_quote_lifetime_ms: 200,     // 200ms
            }),
        }
    }
}

impl AnalysisConfig {
    pub fn development() -> Self {
        Self {
            mode: AnalysisMode::Development,
            auto_monte_carlo: false,
            auto_walk_forward: false,
            ..Default::default()
        }
    }
    
    pub fn validation() -> Self {
        Self {
            mode: AnalysisMode::Validation,
            auto_monte_carlo: true,
            auto_walk_forward: false,
            monte_carlo_runs: 250, // Faster for validation
            ..Default::default()
        }
    }
    
    pub fn production() -> Self {
        Self {
            mode: AnalysisMode::Production,
            auto_monte_carlo: true,
            auto_walk_forward: true,
            monte_carlo_runs: 1000, // More thorough for production
            ..Default::default()
        }
    }

    /// Check if auto-escalation to validation mode should trigger
    pub fn should_escalate_to_validation(&self, 
        duration_days: i64, 
        sharpe_ratio: f64, 
        profit_factor: f64, 
        num_trades: u64
    ) -> bool {
        matches!(self.mode, AnalysisMode::Development) &&
        duration_days >= self.min_duration_for_auto &&
        sharpe_ratio >= self.validation_sharpe_threshold &&
        profit_factor >= self.validation_pf_threshold &&
        num_trades >= self.validation_min_trades
    }

    /// Check if auto-escalation to production mode should trigger
    pub fn should_escalate_to_production(&self, 
        mc_confidence_95th: f64,
        export_requested: bool
    ) -> bool {
        matches!(self.mode, AnalysisMode::Validation) &&
        mc_confidence_95th > 1.0 && // Positive expected returns at 95th percentile
        export_requested
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    // ── Exchange fee routing: every id the Massive/Polygon feeds emit must
    //    resolve to the correct preset, not the generic fallback. ──────────

    #[test]
    fn test_fee_routing_polygon_crypto_venues() {
        // Live Polygon crypto exchange list: Coinbase, Bitfinex, Bitstamp, Binance, Kraken
        assert_eq!(ExchangeFeeConfig::for_exchange("coinbase").exchange, "coinbase");
        assert_eq!(ExchangeFeeConfig::for_exchange("bitfinex").exchange, "bitfinex");
        assert_eq!(ExchangeFeeConfig::for_exchange("bitstamp").exchange, "bitstamp");
        assert_eq!(ExchangeFeeConfig::for_exchange("binance").exchange, "binance");
        assert_eq!(ExchangeFeeConfig::for_exchange("kraken").exchange, "kraken");
    }

    #[test]
    fn test_fee_routing_stocks_use_alpaca() {
        // Stock asset class uses "consolidated"; individual venues also appear
        for id in ["consolidated", "nasdaq", "nyse", "nyse_arca", "nyse-arca", "amex"] {
            let cfg = ExchangeFeeConfig::for_exchange(id);
            assert_eq!(cfg.exchange, "alpaca", "stock id '{}' must route to alpaca", id);
            assert_eq!(cfg.maker_fee, 0.0, "equities are commission-free");
        }
    }

    #[test]
    fn test_fee_routing_forex_uses_oanda() {
        let cfg = ExchangeFeeConfig::for_exchange("fx");
        assert_eq!(cfg.exchange, "oanda");
        assert!(cfg.taker_fee < 0.0005, "forex costs are spread-based, sub-5bps");
    }

    // ── Per-exchange leverage caps: real, venue-verified figures (2026-07),
    //    not one flat platform-wide number for every asset class. ──────────

    #[test]
    fn test_max_leverage_per_exchange() {
        // (exchange_id, expected_max_leverage)
        let cases: &[(&str, f64)] = &[
            ("kraken", 10.0),
            ("binance", 125.0),
            ("binance_us", 10.0),
            ("coinbase", 1.0),
            ("bitfinex", 10.0),
            ("bitstamp", 2.0),
            ("bybit", 100.0),
            ("okx", 100.0),
            ("gemini", 5.0),
            ("deribit", 50.0),
            ("oanda", 50.0),
            ("alpaca", 4.0),
            ("breakout", 5.0),
            ("deepbook", 1.0),
            ("cetus", 1.0),
        ];
        for (id, expected) in cases {
            let cfg = ExchangeFeeConfig::for_exchange(id);
            assert_eq!(cfg.max_leverage, *expected, "unexpected max_leverage for '{}'", id);
        }
    }

    #[test]
    fn resolve_leverage_clamps_above_cap() {
        let fc = ExchangeFeeConfig::for_exchange("alpaca"); // max_leverage = 4.0
        let (capped, was_capped) = fc.resolve_leverage(50.0);
        assert_eq!(capped, 4.0);
        assert!(was_capped);
    }

    #[test]
    fn resolve_leverage_passes_through_when_within_cap() {
        let fc = ExchangeFeeConfig::for_exchange("oanda"); // max_leverage = 50.0
        assert_eq!(fc.resolve_leverage(10.0), (10.0, false));
    }

    #[test]
    fn resolve_leverage_never_drops_below_one() {
        let fc = ExchangeFeeConfig::for_exchange("kraken");
        assert_eq!(fc.resolve_leverage(0.5), (1.0, false));
    }

    #[test]
    fn test_max_leverage_stocks_and_forex_aliases_match_base_venue() {
        // Aliases must resolve to the SAME leverage cap as their base venue,
        // not silently fall back to something else.
        assert_eq!(ExchangeFeeConfig::for_exchange("consolidated").max_leverage, 4.0);
        assert_eq!(ExchangeFeeConfig::for_exchange("nasdaq").max_leverage, 4.0);
        assert_eq!(ExchangeFeeConfig::for_exchange("fx").max_leverage, 50.0);
        assert_eq!(ExchangeFeeConfig::for_exchange("forex").max_leverage, 50.0);
    }

    #[test]
    fn test_max_leverage_unknown_exchange_uses_conservative_fallback() {
        // Matches the pre-per-exchange-cap global default (3.0x) so an
        // unrecognized venue keeps prior behavior rather than becoming more
        // permissive or more restrictive by surprise.
        let cfg = ExchangeFeeConfig::for_exchange("some-made-up-venue");
        assert_eq!(cfg.max_leverage, 3.0);
    }

    #[test]
    fn test_max_leverage_kraken_variants_inherit_base_cap() {
        // kraken_established()/kraken_colocated()/etc. build on top of
        // kraken() and don't override leverage -- confirm they inherit it
        // rather than silently defaulting to something else.
        assert_eq!(ExchangeFeeConfig::kraken_established().max_leverage, 10.0);
        assert_eq!(ExchangeFeeConfig::kraken_colocated().max_leverage, 10.0);
        assert_eq!(ExchangeFeeConfig::kraken_colocated_established().max_leverage, 10.0);
    }

    #[test]
    fn test_bitfinex_zero_fee_schedule() {
        // Bitfinex eliminated spot maker/taker fees on 2025-12-17
        let cfg = ExchangeFeeConfig::for_exchange("bitfinex");
        assert_eq!(cfg.maker_fee, 0.0);
        assert_eq!(cfg.taker_fee, 0.0);
    }

    #[test]
    fn test_bitstamp_tier_anchors() {
        let cfg = ExchangeFeeConfig::for_exchange("bitstamp");
        // Entry tier: 0.30% maker / 0.40% taker
        assert!((cfg.get_fee_rate(0.0, true) - 0.0030).abs() < 1e-9);
        assert!((cfg.get_fee_rate(0.0, false) - 0.0040).abs() < 1e-9);
        // $5M tier: 0.03% maker / 0.12% taker
        assert!((cfg.get_fee_rate(5_000_000.0, true) - 0.0003).abs() < 1e-9);
        assert!((cfg.get_fee_rate(5_000_000.0, false) - 0.0012).abs() < 1e-9);
        // Above $1B: 0.00% maker / 0.03% taker
        assert!((cfg.get_fee_rate(2_000_000_000.0, true) - 0.0).abs() < 1e-9);
        assert!((cfg.get_fee_rate(2_000_000_000.0, false) - 0.0003).abs() < 1e-9);
    }

    #[test]
    fn test_unknown_exchange_falls_back_to_generic() {
        let cfg = ExchangeFeeConfig::for_exchange("some-new-venue");
        assert_eq!(cfg.exchange, "some-new-venue");
        assert!((cfg.maker_fee - 0.0010).abs() < 1e-9);
        assert!((cfg.taker_fee - 0.0015).abs() < 1e-9);
    }

    #[test]
    fn test_default_config_creation() {
        let config = BacktestConfig::default();
        assert_eq!(config.database.host, "localhost");
        assert_eq!(config.trading.initial_capital, 100000.0);
        assert_eq!(config.data.default_timeframe, "1d");
    }

    #[test]
    fn test_config_validation() {
        let mut config = BacktestConfig::default();
        
        // Test valid config
        assert!(config.validate().is_ok());
        
        // Test invalid commission rate
        config.trading.commission_rate = -0.1;
        assert!(config.validate().is_err());
        
        // Reset and test invalid capital
        config.trading.commission_rate = 0.001;
        config.trading.initial_capital = -1000.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_yaml_serialization() {
        let config = BacktestConfig::default();
        let yaml_str = serde_yaml::to_string(&config).unwrap();
        
        // Should contain expected fields
        assert!(yaml_str.contains("database"));
        assert!(yaml_str.contains("trading"));
        assert!(yaml_str.contains("data"));
        assert!(yaml_str.contains("output"));
        
        // Should be able to deserialize back
        let parsed: BacktestConfig = serde_yaml::from_str(&yaml_str).unwrap();
        assert_eq!(parsed.database.host, config.database.host);
    }

    #[test]
    fn test_file_operations() {
        let temp_path = PathBuf::from("test_config.yaml");
        let config = BacktestConfig::default();
        
        // Test saving
        config.to_file(&temp_path).unwrap();
        assert!(temp_path.exists());
        
        // Test loading
        let loaded_config = BacktestConfig::from_file(&temp_path).unwrap();
        assert_eq!(loaded_config.database.host, config.database.host);
        assert_eq!(loaded_config.trading.initial_capital, config.trading.initial_capital);
        
        // Cleanup
        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn genetic_config_default_keeps_random_control_off() {
        // Regression guard: random_control must default to false so every
        // existing job (which never sets it) keeps running guided GA search,
        // not the random-perturbation control arm.
        assert!(!GeneticConfig::default().random_control);
    }

    #[test]
    fn genetic_config_random_control_round_trips_through_json() {
        // The job-submission path threads this through BacktestJobParams's
        // JSONB params_json blob -- confirm it actually survives a
        // serialize/deserialize cycle rather than silently dropping.
        let config = GeneticConfig { random_control: true, ..GeneticConfig::default() };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: GeneticConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.random_control);
    }
}
