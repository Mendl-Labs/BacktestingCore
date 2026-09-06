//! Python directional strategy simulation (BaseStrategy subclass).
//!
//! Used by: `program/src/worker.rs` when `StrategyType::Custom`.
//! Handles Buy/Sell/Hold/Close signals with maker and taker fills.
//! Supports GA optimization if the Python class defines `parameter_space()`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use crate::{log_debug, log_info, log_warn};
use crate::logging_facade::BACKTEST_LOGGER;

use config::{BacktestConfig, ExchangeFeeConfig, ParameterValue};
use dataloader::MarketData;
use derivatives::DerivativeMetadata;
use orderbook::{BookSide, Fill, LiquidityType};
use portfoliomanager::PortfolioState;
use signal::{Signal, SignalStrength, SignalReason, SignalType, SpreadLeg};
use strategy::traits::{
    MarketSession, PricePoint, PriceSource, SessionType, StrategyContext, VolumePoint,
    PortfolioSnapshot,
};
use strategy::Strategy; // trait import — required for .initialize() / .generate_signals()
use strategy::executor::{self, StrategyExecutor, ExecutionTier};

use crate::types::{BacktestResult, TradeRecord, MarketDataInput, BookSnapshot, OrderBookLevel};
use crate::python_validation::OptionSpreadLeg;
use crate::rolling_window::RollingWindow;
use genetic::adaptive_sampling::StableRegionDetector;
use metrics::significance;
use metrics::risk;
use crate::synthetic_book::{SyntheticOrderBook, SyntheticBookConfig};
use strategy::traits::OrderbookSnapshot;

// ---------------------------------------------------------------------------
// Orderbook snapshot lookup
// ---------------------------------------------------------------------------

/// Convert a `BookSnapshot` into the strategy-facing `OrderbookSnapshot`.
fn tardis_to_orderbook(snap: &BookSnapshot) -> OrderbookSnapshot {
    let bids: Vec<[f64; 2]> = snap.bids.iter()
        .filter(|l| l.price > 0.0)
        .map(|l| [l.price, l.amount])
        .collect();
    let asks: Vec<[f64; 2]> = snap.asks.iter()
        .filter(|l| l.price > 0.0)
        .map(|l| [l.price, l.amount])
        .collect();
    let best_bid = bids.first().map(|l| l[0]).unwrap_or(0.0);
    let best_ask = asks.first().map(|l| l[0]).unwrap_or(0.0);
    let spread = best_ask - best_bid;
    let mid_price = (best_bid + best_ask) / 2.0;
    let bid_depth: f64 = bids.iter().map(|l| l[1]).sum();
    let ask_depth: f64 = asks.iter().map(|l| l[1]).sum();
    let total = bid_depth + ask_depth;
    let imbalance = if total > 0.0 { (bid_depth - ask_depth) / total } else { 0.0 };
    OrderbookSnapshot { bids, asks, spread, mid_price, bid_depth, ask_depth, imbalance }
}

/// Binary-search a pre-sorted `(timestamp_ms, value)` series for the value
/// observable as-of the given tick timestamp: the last entry at-or-before it.
///
/// Unlike `find_nearest_snapshot` below, this deliberately returns `None`
/// (not the earliest entry) when every entry is still in the future relative
/// to `timestamp` -- an orderbook snapshot approximates market microstructure
/// and a same-day fallback is harmless, but a user-supplied alt-data series
/// (e.g. insider-buying intensity) is a candidate alpha signal, and silently
/// exposing tomorrow's value on today's tick would be real lookahead bias.
pub(crate) fn find_asof_value(series: &[(i64, f64)], timestamp: &chrono::DateTime<chrono::Utc>) -> Option<f64> {
    if series.is_empty() {
        return None;
    }
    let tick_ms = timestamp.timestamp_millis();
    let idx = series.partition_point(|(ts, _)| *ts <= tick_ms);
    if idx == 0 {
        None
    } else {
        Some(series[idx - 1].1)
    }
}

/// Binary-search the pre-sorted snapshots for the entry whose `timestamp_us`
/// is closest to (but not after) the given tick timestamp.
pub(crate) fn find_nearest_snapshot(
    snapshots: &[BookSnapshot],
    timestamp: &chrono::DateTime<chrono::Utc>,
) -> Option<OrderbookSnapshot> {
    if snapshots.is_empty() {
        return None;
    }
    let tick_us = timestamp.timestamp_millis() * 1000;
    let idx = snapshots.partition_point(|s| s.timestamp_us <= tick_us);
    // partition_point returns the first index where predicate is false,
    // so idx-1 is the last snapshot at-or-before the tick.
    if idx == 0 {
        // All snapshots are after this tick — use the earliest one.
        Some(tardis_to_orderbook(&snapshots[0]))
    } else {
        Some(tardis_to_orderbook(&snapshots[idx - 1]))
    }
}

/// Effective leverage for a run: the lower of what was requested (already
/// exchange-capped upstream by `worker::resolve_leverage`) and what the
/// strategy itself declares via `get_risk_limits().max_leverage`
/// (`Strategy::declared_max_leverage`, `f64::INFINITY` -- no additional cap --
/// for a strategy that doesn't declare one). A strategy-declared cap can only
/// ever REDUCE leverage below the configured/exchange-capped value, never
/// raise it above what was requested.
fn effective_leverage(configured: f64, strategy_declared_cap: f64) -> f64 {
    configured.min(strategy_declared_cap).max(1.0)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for a Python directional simulation run.
pub struct PythonSimConfig {
    /// User Python source code (the strategy file contents).
    pub python_source: String,
    /// Backtest configuration (initial capital, fees, latency, etc.).
    pub backtest_config: BacktestConfig,
    /// Optional exchange fee config for maker/taker fee lookup.
    pub fee_config: Option<ExchangeFeeConfig>,
    /// Optional user-uploaded alt-data series: key (e.g. filename stem) →
    /// `(timestamp_ms, value)` points, sorted ascending by timestamp.
    /// Looked up per tick via `find_asof_value` (last value at-or-before the
    /// tick's own timestamp -- never a future value), so a key is simply
    /// absent from `StrategyContext.custom_data` before its series starts.
    pub supplementary_data: HashMap<String, Vec<(i64, f64)>>,
    /// Optional pre-parsed parameters to pass to the strategy.
    pub parameters: HashMap<String, ParameterValue>,
    /// Optional progress callback: receives (current_tick, total_ticks).
    pub progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
    /// Optional risk manager for position/drawdown limits (opt-in).
    pub risk_manager: Option<riskmanager::RiskManager>,
    /// Maximum trade log records. `None` = unlimited (final backtest),
    /// `Some(500)` = capped (GA evaluation).
    pub max_trade_log_size: Option<usize>,
    /// Pre-loaded orderbook snapshots (sorted by `timestamp_us`).
    /// When present, the simulation attaches the nearest snapshot to each tick's
    /// `StrategyContext.orderbook_snapshot`.
    pub orderbook_snapshots: Option<Vec<BookSnapshot>>,
    /// Cross-exchange strategy support (2026-07-22): when set, the `market_data`
    /// passed to `run`/`run_with_input` MUST be the PRIMARY venue's own aligned
    /// candle series (same order, same length as every venue's series here) --
    /// the vectorized bulk-signal path calls `compute_all_signals_multi_venue`
    /// with this bundle instead of the normal flat-array `compute_all_signals`
    /// when the executor's `accepts_multi_venue()` is true, falling back to the
    /// normal single-venue path otherwise. `None` (the default) is the existing,
    /// unaffected single-venue behavior.
    pub multi_venue_data: Option<HashMap<(String, String), strategy::traits::VenueSeries>>,
    /// Pre-computed historical options-derived IV overlay, sorted by
    /// `IvSurface.timestamp` -- populated into `StrategyContext.iv_surface`
    /// at each tick via a nearest-at-or-before lookup (mirrors
    /// `find_nearest_snapshot`'s binary-search shape). `None` (the default,
    /// used at every existing call site) keeps `iv_surface` `None` for
    /// every tick exactly as before -- zero behavior change for any
    /// strategy that doesn't opt in via `data_requirements().needs_iv_surface`.
    pub historical_iv_surfaces: Option<Vec<derivatives::IvSurface>>,
    /// When set, this run is trading a single pre-resolved option contract
    /// rather than the underlying spot asset -- `market_data` must already
    /// be that contract's OWN OHLCV series (see `options_backtest_data` in
    /// the `program` crate, which resolves a job's `OptionLeg` into this
    /// exact shape). The strategy's Python `compute_signals()` still only
    /// ever returns the plain int8 BUY/SELL/HOLD/CLOSE vocabulary (the SDK
    /// has no way to embed contract metadata in a signal) -- `i8_to_signals`
    /// reinterprets a bulk BUY/SELL code as `SignalType::BuyOption`/
    /// `SellOption` on THIS instrument when set, instead of a plain spot
    /// `Buy`/`Sell`, so the strategy's existing directional logic (when to
    /// be long/flat) drives the option position without the strategy code
    /// itself needing to know it's trading an option at all. CLOSE needs no
    /// equivalent handling: `close_all_positions` already closes each open
    /// position using ITS OWN recorded instrument (set when the position
    /// was opened), not a signal-level one. `None` (the default) is the
    /// existing, unaffected spot/futures behavior.
    pub option_instrument: Option<DerivativeMetadata>,
    /// When set, this run is trading a pre-configured multi-leg option
    /// spread (a vertical, currently -- 2 legs, see `OptionSpreadLeg`'s doc
    /// comment for the exact scope) instead of a single option or the
    /// underlying spot asset. Mutually exclusive with `option_instrument`
    /// (a run is single-leg or multi-leg, never both) -- when both are
    /// `Some`, `option_spread` takes precedence. `None` (the default) is
    /// the existing, unaffected behavior.
    pub option_spread: Option<Vec<OptionSpreadLeg>>,
    /// The underlying's own spot price at each tick, aligned index-for-index
    /// with `market_data`/`input` -- required to price Greeks at all when
    /// `option_instrument`/`option_spread` is set, since in that mode
    /// `current_price` in the tick loop is the OPTION CONTRACT's own price,
    /// not the underlying's (see `option_instrument`'s doc comment: the
    /// whole point of that field is trading the contract's own OHLCV
    /// series). `None` (the default) leaves Greeks unpopulated exactly as
    /// before this field existed -- `update_position_greeks` is simply never
    /// called, matching every existing call site.
    pub underlying_series: Option<Vec<f64>>,
}

/// Binary-search the pre-sorted historical IV series for the entry whose
/// `timestamp` is closest to (but not after) the given tick timestamp --
/// same shape as `find_nearest_snapshot` above, applied to
/// `IvSurface.timestamp` instead of `BookSnapshot.timestamp_us`.
fn find_nearest_iv_surface<'a>(
    surfaces: &'a [derivatives::IvSurface],
    timestamp: &chrono::DateTime<chrono::Utc>,
) -> Option<&'a derivatives::IvSurface> {
    if surfaces.is_empty() {
        return None;
    }
    let idx = surfaces.partition_point(|s| s.timestamp <= *timestamp);
    if idx == 0 {
        // Every surface is after this tick -- no historical coverage yet.
        None
    } else {
        Some(&surfaces[idx - 1])
    }
}

/// Annualized trailing realized-vol series for `update_position_greeks`'s
/// fallback vol source (see that call site's doc comment for when the
/// fallback is actually used vs. the real IV surface). `series` is the
/// underlying's own spot price at each tick (`PythonSimConfig.underlying_series`);
/// `timestamps_ms` its aligned tick timestamps, used only to derive the real
/// bars-per-year annualization factor from the data's own actual cadence
/// (`metrics::significance::observations_per_year_from_span`) rather than a
/// hardcoded 252/365. Empty when `series` has fewer than 2 points.
fn annualized_underlying_realized_vol(series: &[f64], timestamps_ms: &[i64], lookback: usize) -> Vec<Option<f64>> {
    if series.len() < 2 {
        return Vec::new();
    }
    let annualization = if timestamps_ms.len() >= 2 {
        let span_secs = (timestamps_ms[timestamps_ms.len() - 1] - timestamps_ms[0]) as f64 / 1000.0;
        significance::observations_per_year_from_span(timestamps_ms.len(), Some(span_secs))
    } else {
        365.0
    };
    portfoliomanager::margin::trailing_realized_vol_series(series, lookback)
        .into_iter()
        .map(|v| v.map(|std| std * annualization.sqrt()))
        .collect()
}

/// Pick the vol to price `instrument`'s Greeks with at this tick: the real
/// historical IV surface's nearest point for this exact strike/expiry when
/// available, falling back to `fallback_vol` (the precomputed annualized
/// realized-vol reading for this tick, from `annualized_underlying_realized_vol`)
/// otherwise. `None` when neither source has a usable (finite, positive)
/// reading -- callers must skip `update_position_greeks` entirely in that
/// case rather than pricing off a zero/garbage vol.
fn select_greeks_vol(iv_surface: Option<&derivatives::IvSurface>, instrument: &DerivativeMetadata, fallback_vol: Option<f64>) -> Option<f64> {
    iv_surface
        .and_then(|surf| {
            let strike = instrument.instrument_kind.strike()?;
            let expiry = instrument.instrument_kind.expiry()?;
            surf.nearest_iv(strike, expiry).map(|p| p.iv)
        })
        .or(fallback_vol)
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Whether `position` is exempt from the engine's leveraged-futures-style
/// margin-call check (`portfoliomanager::margin::is_liquidated`, applied at
/// this function's call site in the tick loop). `true` for an already-closed
/// position, or an option (long or short) -- see that call site's own doc
/// comment for the full reasoning on why options are exempt in BOTH
/// directions despite a short option's real unbounded loss potential
/// (no real option margin-call formula exists in this platform; the
/// account-level `risk_manager` drawdown gate is the applicable backstop
/// instead). `false` for futures/perps (2026-09-02 fix): unlike options,
/// they use the exact leveraged-margin accounting `is_liquidated` models.
fn is_exempt_from_leverage_liquidation_check(position: &portfoliomanager::Position) -> bool {
    position.close_time.is_some()
        || position.instrument.as_ref().map(|i| i.instrument_kind.is_option()).unwrap_or(false)
}

/// Run a directional Python strategy simulation.
///
/// Creates a `PythonStrategy`, iterates through market data tick-by-tick,
/// calls `generate_signals()` on each tick, and executes fills against a
/// simple portfolio model (taker = immediate @ market price, maker = fills
/// only when subsequent trade crosses our limit price).
pub async fn run(
    market_data: &[MarketData],
    config: PythonSimConfig,
) -> Result<PythonSimResult> {
    run_with_input(MarketDataInput::Slice(market_data), config).await
}

/// Zero-copy entry point for SimBinCache mmap'd ticks.
/// Avoids ~2.5 GiB `Vec<MarketData>` allocation for large datasets.
pub async fn run_sim_ticks(
    ticks: &[data_prep::SimulationTick],
    symbol: Arc<str>,
    exchange: Arc<str>,
    config: PythonSimConfig,
) -> Result<PythonSimResult> {
    run_with_input(MarketDataInput::SimTicks { ticks, symbol, exchange }, config).await
}

async fn run_with_input(
    input: MarketDataInput<'_>,
    config: PythonSimConfig,
) -> Result<PythonSimResult> {
    if input.is_empty() {
        anyhow::bail!("No market data provided for Python simulation");
    }

    log_debug!(BACKTEST_LOGGER, "python_simulation::run() starting with {} data points", input.len());

    // Build the best executor for this strategy (Tier 0–3).
    // The executor analyses state_schema(), model(), features() to determine
    // the optimal execution tier automatically.
    let (mut executor, tier_analysis) = match executor::build_executor(
        &config.python_source,
        config.parameters.clone(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return Err(e).context("Failed to build strategy executor");
        }
    };

    let execution_tier = tier_analysis.tier;
    log_info!(BACKTEST_LOGGER, "Strategy executor: {} — {}", execution_tier, tier_analysis.reason);

    // Capture the strategy's self-declared name() (custom Python strategies) so
    // persistence can label the result correctly instead of "Custom". Treated as
    // absent when it is the generic placeholder.
    let strategy_display_name = executor
        .strategy_name()
        .filter(|n| !n.is_empty() && n != "PythonStrategy");

    let initial_capital = config.backtest_config.trading.initial_capital;
    let leverage = effective_leverage(config.backtest_config.trading.leverage, executor.declared_max_leverage());
    let maintenance_margin_ratio = config.backtest_config.trading.maintenance_margin_ratio;
    let liquidation_penalty_bps = config.backtest_config.trading.liquidation_penalty_bps;
    let mut portfolio = PortfolioState::with_balance(initial_capital).with_leverage(leverage);

    // Volatility-targeting position-size overlay (opt-in via
    // `backtest_config.trading.vol_target_annual`; `None` -- the default --
    // is fully behavior-preserving). See
    // `portfoliomanager::margin::VolTargetConfig`'s doc comment.
    let vol_target_cfg = config.backtest_config.trading.vol_target_annual.map(|target_annual_vol| {
        portfoliomanager::margin::VolTargetConfig {
            target_annual_vol,
            lookback_bars: config.backtest_config.trading.vol_target_lookback_bars,
            bars_per_year: config.backtest_config.trading.vol_target_bars_per_year,
            min_scale: config.backtest_config.trading.vol_target_min_scale,
            max_scale: config.backtest_config.trading.vol_target_max_scale,
        }
    });

    // Per-entry position size (fraction of equity) for signals that omit an
    // explicit quantity. Honors a strategy-declared `position_size_pct` /
    // `risk_per_trade` (GA-tunable); falls back to 2%.
    let position_size_pct = resolve_position_size_pct(&config);
    log_info!(
        BACKTEST_LOGGER,
        "Position sizing: {:.2}% of equity per entry (auto-size for signals without explicit quantity)",
        position_size_pct * 100.0
    );

    // Fee configuration
    let maker_fee = config
        .fee_config
        .as_ref()
        .map(|f| f.maker_fee)
        .unwrap_or(0.0005); // 5 bps default
    let taker_fee = config
        .fee_config
        .as_ref()
        .map(|f| f.taker_fee)
        .unwrap_or(0.001); // 10 bps default

    // Fill realism config
    let slippage_bps = config.backtest_config.trading.slippage_bps;
    let latency_ms = config.backtest_config.trading.latency_ms;
    let adverse_selection = config.backtest_config.trading.adverse_selection;
    let fill_volume_fraction = config.backtest_config.trading.fill_volume_fraction;
    let synthetic_book_cfg = if config.backtest_config.trading.use_synthetic_book {
        Some(SyntheticBookConfig {
            spread_bps: config.backtest_config.trading.synthetic_spread_bps,
            depth_levels: config.backtest_config.trading.synthetic_depth_levels,
            ..Default::default()
        })
    } else {
        None
    };

    // Adaptive equity curve sampling — avoid 240 MB for 31M-tick datasets.
    let equity_sample_interval = if input.len() > 1_000_000 {
        (input.len() / 10_000).max(1)
    } else {
        1
    };
    let equity_capacity = (input.len() / equity_sample_interval) + 100;

    // Tracking state
    let mut equity_curve: Vec<f64> = Vec::with_capacity(equity_capacity);
    // Bug #33 — parallel timestamps (Unix ms) for equity_curve, populated at the
    // same sample cadence so downstream consumers can plot equity over real dates.
    let mut equity_curve_timestamps: Vec<i64> = Vec::with_capacity(equity_capacity);
    let mut price_series: Vec<f64> = Vec::with_capacity(input.len());
    let mut trade_returns: Vec<f64> = Vec::new();
    let mut trade_maes: Vec<f64> = Vec::new();
    let mut trade_log: Vec<TradeRecord> = Vec::new();
    let mut open_positions: Vec<OpenPosition> = Vec::new();
    let mut next_trade_id: usize = 1;
    let max_trade_log_size = config.max_trade_log_size;
    let mut total_commission = 0.0_f64;
    let mut num_trades = 0_usize;
    let mut volume_constrained_fills = 0_usize;
    let required_window = executor.required_data_window();
    let lookback = config
        .backtest_config
        .trading
        .lookback_window
        .max(required_window);
    if lookback > config.backtest_config.trading.lookback_window {
        log_info!(
            BACKTEST_LOGGER,
            "Python strategy requires larger lookback window: configured={} required={} using={}",
            config.backtest_config.trading.lookback_window,
            required_window,
            lookback
        );
    }
    let mut price_window = RollingWindow::<PricePoint>::new(lookback);
    let mut volume_window = RollingWindow::<VolumePoint>::new(lookback);
    let mut pending_limits: Vec<PendingLimit> = Vec::new();
    let mut first_price: Option<f64> = None;
    let mut last_price = 0.0_f64;
    let total_ticks = input.len();

    // Optional risk manager (opt-in)
    let mut risk_manager = config.risk_manager;
    let mut trading_halted = false;
    if let Some(ref mut rm) = risk_manager {
        rm.initialize(initial_capital);
    }

    // Diagnostic counters for debugging zero-trade issues
    let mut diag_buy_signals = 0_usize;
    let mut diag_sell_signals = 0_usize;
    let mut diag_close_signals = 0_usize;
    let mut diag_hold_signals = 0_usize;
    let mut diag_zero_qty_skips = 0_usize;
    let mut diag_error_ticks = 0_usize;
    let mut diag_empty_signal_ticks = 0_usize;
    let mut diag_first_signals_logged = 0_usize;
    // Per-tick `executor.on_tick()` error tracking (per-tick fallback path,
    // used when the bulk `precomputed_signals` path isn't available). Root
    // cause of a real production bug: a strategy whose Python execution
    // genuinely fails on EVERY attempted tick (e.g. failed to initialize
    // for a reason that only surfaces once real ticks run, not at
    // build_executor() time) previously produced a "completed" backtest
    // with 0 trades and no error at all -- the `.unwrap_or_else` below only
    // logged a throttled warning (first 5 ticks), never surfaced to the
    // user. Track total attempts vs. errors so a 100%-error run can be
    // reported honestly instead of silently.
    let mut diag_on_tick_attempts = 0_usize;
    let mut first_on_tick_error: Option<String> = None;

    // Pre-allocate reusable buffers for StrategyContext (avoids ~12GB alloc churn)
    let mut price_buf: Vec<PricePoint> = Vec::with_capacity(lookback);
    let mut volume_buf: Vec<VolumePoint> = Vec::with_capacity(lookback);
    let supplementary_data = config.supplementary_data;
    let ob_snapshots = config.orderbook_snapshots;
    // Reusable custom_data HashMap — cleared and refilled each tick instead of cloning.
    let mut custom_data_buf: HashMap<String, f64> = HashMap::with_capacity(
        supplementary_data.len() + 64, // room for state + indicator keys
    );

    // Stable region detector: skip Python strategy call when price hasn't moved significantly.
    // Disabled unconditionally for strategies that define generate_signals() -- they read
    // context (e.g. context.iv_surface) that can change independently of price, so skipping
    // ticks on a stable price would silently starve them of that context.
    let opt = &config.backtest_config.trading.optimization;
    let mut stable_detector = if opt.stable_region_skipping && !executor.defines_generate_signals() {
        StableRegionDetector::new(opt.stable_region_threshold, opt.max_skip_ticks)
    } else {
        StableRegionDetector::disabled()
    };
    let mut last_signals: Vec<Signal> = Vec::new();
    let mut stable_region_skips = 0_usize;

    log_info!(BACKTEST_LOGGER, "Starting Python directional simulation: {} ticks, ${:.0} capital", total_ticks, initial_capital);

    // ── Vectorized bulk-signal path ────────────────────────────────────────
    // Try to call Python ONCE with the full tick array (1 FFI crossing vs. N).
    // Falls back to per-tick path when not supported or on error.
    // `computed_features`/`captured_prices_for_features`/`captured_timestamps_for_features`
    // carry the optional compute_features() output (and the aligned price/ts
    // series needed to later compute forward returns) out of this block —
    // populated only when ENABLE_COMPUTE_FEATURES is on, so there's no cost
    // when it's off (the default).
    let mut computed_features: Option<HashMap<String, Vec<f64>>> = None;
    let mut captured_prices_for_features: Vec<f64> = Vec::new();
    let mut captured_timestamps_for_features: Vec<i64> = Vec::new();
    // Per-trade dynamic position sizing (2026-07-25): `sizes[i]` is a fraction
    // of equity `(0.0, 1.0]` for tick i, populated inside the block below
    // (best-effort, alongside the signals dispatch) when the strategy defines
    // `compute_position_sizes`. `None` (the strategy doesn't implement it, an
    // exception, or a wrong-length result) means every tick falls back to the
    // flat `position_size_pct` exactly as before this feature existed.
    let mut precomputed_sizes: Option<Vec<f64>> = None;
    // Volatility-targeting overlay (opt-in): the whole trailing realized-vol
    // series is precomputed once here (alongside `precomputed_sizes`, same
    // reasoning -- avoid recomputing a rolling window inside the hot
    // per-tick loop below), causal by construction (`trailing_realized_vol_series`
    // never looks at a bar's own close), and `None` end-to-end when
    // `vol_target_cfg` is `None`.
    let mut precomputed_trailing_vol: Option<Vec<Option<f64>>> = None;
    // Greeks/mark-to-market wiring (2026-09-02): fallback realized-vol
    // series, used per-tick to price an option position's Greeks whenever
    // no historical IV surface point is available for that tick/strike
    // (see the tick loop's `update_position_greeks` call below). `20` bars
    // is a conventional realized-vol lookback (roughly a trading month at
    // daily granularity); annualized by the data's own ACTUAL bar cadence
    // (`observations_per_year_from_span`, populated just below once
    // `timestamps_arr` is built) rather than a hardcoded 252/365, matching
    // how Sharpe/DSR annualization already works elsewhere in this
    // pipeline. Empty (the default) whenever no option instrument/spread is
    // configured, or no `underlying_series` was supplied -- `None` end to
    // end, `update_position_greeks` is simply never called.
    const GREEKS_REALIZED_VOL_LOOKBACK: usize = 20;
    let mut underlying_realized_vol: Vec<Option<f64>> = Vec::new();
    let (precomputed_signals, compute_signals_error): (Option<Vec<i8>>, Option<String>) = {
        let mut prices_arr: Vec<f64>    = Vec::with_capacity(total_ticks);
        let mut volumes_arr: Vec<f64>   = Vec::with_capacity(total_ticks);
        let mut timestamps_arr: Vec<i64> = Vec::with_capacity(total_ticks);
        // Only build OHLC arrays when the strategy opted in (avoids per-eval
        // allocation churn on the GA hot path for close-only strategies).
        let wants_ohlc = executor.compute_signals_accepts_ohlc();
        let mut opens_arr:  Vec<f64> = if wants_ohlc { Vec::with_capacity(total_ticks) } else { Vec::new() };
        let mut highs_arr:  Vec<f64> = if wants_ohlc { Vec::with_capacity(total_ticks) } else { Vec::new() };
        let mut lows_arr:   Vec<f64> = if wants_ohlc { Vec::with_capacity(total_ticks) } else { Vec::new() };
        for data_cow in input.iter() {
            if wants_ohlc {
                let (o, h, l, c, v, ts) = extract_ohlcv_ts(&*data_cow);
                prices_arr.push(c);
                volumes_arr.push(v);
                timestamps_arr.push(ts.timestamp_millis());
                opens_arr.push(o);
                highs_arr.push(h);
                lows_arr.push(l);
            } else {
                let (p, v, ts) = extract_price_volume_ts(&*data_cow);
                prices_arr.push(p);
                volumes_arr.push(v);
                timestamps_arr.push(ts.timestamp_millis());
            }
        }
        if let Some(series) = config.underlying_series.as_ref() {
            if config.option_instrument.is_some() || config.option_spread.is_some() {
                underlying_realized_vol = annualized_underlying_realized_vol(series, &timestamps_arr, GREEKS_REALIZED_VOL_LOOKBACK);
            }
        }
        let (features, features_error) = executor.compute_all_features(&prices_arr, &volumes_arr, &timestamps_arr).await;
        if let Some(ref e) = features_error {
            log_warn!(BACKTEST_LOGGER, "compute_features() failed (diagnostic only, ignored): {}", e);
        }
        if let Some(cfg) = &vol_target_cfg {
            precomputed_trailing_vol = Some(portfoliomanager::margin::trailing_realized_vol_series(&prices_arr, cfg.lookback_bars));
        }
        if features.is_some() {
            captured_prices_for_features = prices_arr.clone();
            captured_timestamps_for_features = timestamps_arr.clone();
        }
        computed_features = features;
        // Cross-exchange strategy support (2026-07-22): when the caller supplied
        // a multi-venue bundle AND the executor supports it, compute signals
        // from ALL declared venues at once instead of just this flat series.
        // `input`/`total_ticks` must be the PRIMARY venue's own aligned series
        // in this mode (the caller's responsibility) so the signal index
        // returned here still lines up 1:1 with the tick loop below.
        let raw_signals_result: (Option<Vec<i8>>, Option<String>) = if executor.defines_generate_signals() {
            // Strategy defines generate_signals() -- skip the vectorized bulk path
            // entirely so the per-tick loop below dispatches to it with the full
            // StrategyContext instead of ever using precomputed array signals.
            (None, None)
        } else {
            match &config.multi_venue_data {
                Some(venues) if executor.accepts_multi_venue() => {
                    executor.compute_all_signals_multi_venue(venues).await
                }
                _ => {
                    executor.compute_all_signals(&prices_arr, &volumes_arr, &timestamps_arr, &opens_arr, &highs_arr, &lows_arr).await
                }
            }
        };

        // Per-trade dynamic position sizing (2026-07-25): best-effort fetch,
        // never fails the run. A wrong-length array, an exception, or the
        // strategy simply not implementing compute_position_sizes all fall
        // back to `None` here, which resolve_position_size_pct's flat value
        // covers below. See docs/PLAN or the position-sizing implementation
        // plan for the full design.
        if executor.accepts_position_sizes() {
            match executor.compute_all_position_sizes(&prices_arr, &volumes_arr, &timestamps_arr).await {
                (Some(sizes), _) if sizes.len() == total_ticks => precomputed_sizes = Some(sizes),
                (Some(sizes), _) => {
                    log_warn!(
                        BACKTEST_LOGGER,
                        "compute_position_sizes() returned {} values, expected {} — falling back to position_size_pct for this run",
                        sizes.len(), total_ticks
                    );
                }
                (_, err) => {
                    if let Some(e) = err {
                        log_warn!(BACKTEST_LOGGER, "compute_position_sizes() failed: {} — falling back to position_size_pct", e);
                    }
                }
            }
        }

        match raw_signals_result {
            (Some(sigs), _) if sigs.len() == total_ticks => {
                log_info!(BACKTEST_LOGGER, "Vectorized bulk signals: {} ticks computed with 1 FFI call", total_ticks);
                (Some(sigs), None)
            }
            (Some(sigs), _) => {
                // compute_signals() returned an array of the WRONG LENGTH.
                // The most common cause: `return sig[warmup:]` instead of `return sig`.
                // Silently falling through to the per-tick path here produces 0 trades
                // with no error message — the user thinks their strategy ran but it didn't.
                // Fail loudly instead.
                let msg = format!(
                    "compute_signals() returned {} signals but expected {}. \
                     The array must have exactly one entry per tick. \
                     Common fix: return sig (full-length zeros array) instead of sig[warmup:].",
                    sigs.len(), total_ticks
                );
                (None, Some(msg))
            }
            (_, err) => {
                if let Some(ref e) = err {
                    log_warn!(BACKTEST_LOGGER, "Vectorized path failed: {}", e);
                }
                (None, err)
            }
        }
    };

    // Early abort: if compute_signals() failed there is nothing to fall back to.
    // Continuing would run every tick as a silent HOLD, producing 0 trades and
    // misleading the user into thinking the strategy ran correctly.
    if precomputed_signals.is_none() {
        if let Some(ref e) = compute_signals_error {
            anyhow::bail!(
                "Strategy compute_signals() failed: {}. Fix the error in your strategy code.",
                e
            );
        }
    }

    let signal_counts: Option<crate::types::SignalCounts> = precomputed_signals.as_ref().map(|sigs| {
        let mut counts = crate::types::SignalCounts::default();
        for &s in sigs {
            match s {
                1  => counts.buy  += 1,
                -1 => counts.sell += 1,
                2  => counts.close += 1,
                _  => counts.hold += 1,
            }
        }
        counts
    });
    // `signal_counts` from the precomputed (vectorized) path is populated above;
    // the per-tick `generate_signals` path counts via `diag_*` and we merge those
    // into a `SignalCounts` after the dispatch loop completes (see below).
    let mut signal_counts = signal_counts;
    let risk_free_rate = config.backtest_config.trading.risk_free_rate;

    for (tick_idx, data_cow) in input.iter().enumerate() {
        let data = &*data_cow;
        // -- Extract current price/volume from market data --
        let (current_price, current_volume, timestamp) = extract_price_volume_ts(data);
        if current_price <= 0.0 {
            continue;
        }
        if first_price.is_none() {
            first_price = Some(current_price);
        }
        last_price = current_price;
        price_series.push(current_price);

        // Update MAE/MFE for all open positions
        for pos in &mut open_positions {
            let unrealized = if pos.side == "long" {
                (current_price - pos.entry_fill_price) / pos.entry_fill_price
            } else {
                (pos.entry_fill_price - current_price) / pos.entry_fill_price
            };
            if unrealized < pos.worst_unrealized {
                pos.worst_unrealized = unrealized;
            }
            if unrealized > pos.best_unrealized {
                pos.best_unrealized = unrealized;
            }
        }

        // -- Check pending limit orders for fills --
        let filled = fill_pending_limits(
            &mut pending_limits,
            current_price,
            &mut portfolio,
            &mut total_commission,
            maker_fee,
            &mut num_trades,
            &mut trade_returns,
            &mut trade_maes,
            &mut trade_log,
            &mut open_positions,
            &mut next_trade_id,
            max_trade_log_size,
            timestamp,
            latency_ms,
            adverse_selection,
            current_volume,
            fill_volume_fraction,
            &mut volume_constrained_fills,
        );
        if filled > 0 {
            log_info!(BACKTEST_LOGGER, "Filled {} pending limit order(s) at tick {}", filled, tick_idx);
        }

        // -- Build history arrays for StrategyContext --
        price_window.push(PricePoint {
            timestamp,
            price: current_price,
            source: PriceSource::Trade,
        });
        volume_window.push(VolumePoint {
            timestamp,
            volume: current_volume,
            side: None,
        });

        // -- Construct StrategyContext (reuse buffers to avoid allocations) --
        price_buf.clear();
        let (s1, s2) = price_window.as_slices();
        price_buf.extend_from_slice(s1);
        price_buf.extend_from_slice(s2);
        
        volume_buf.clear();
        let (v1, v2) = volume_window.as_slices();
        volume_buf.extend_from_slice(v1);
        volume_buf.extend_from_slice(v2);

        // Pre-enrich: let the executor inject state/indicators into custom_data
        // BEFORE building StrategyContext, so on_tick avoids cloning ~48 KB of
        // price/volume history.
        custom_data_buf.clear();
        for (k, series) in &supplementary_data {
            if let Some(v) = find_asof_value(series, &timestamp) {
                custom_data_buf.insert(k.clone(), v);
            }
        }
        executor.pre_enrich(data, &mut custom_data_buf);
        
        let ob_snap = ob_snapshots.as_deref().and_then(|s| find_nearest_snapshot(s, &timestamp));
        let iv_surface_for_tick = config.historical_iv_surfaces.as_deref()
            .and_then(|s| find_nearest_iv_surface(s, &timestamp))
            .cloned();

        // Greeks/mark-to-market wiring (2026-09-02): needs the underlying's
        // OWN spot at this tick (`config.underlying_series`, NOT
        // `current_price` -- see that field's doc comment for why those
        // differ whenever an option instrument/spread is configured) and a
        // vol reading -- prefer the real historical IV surface for this
        // exact strike/expiry when available, falling back to the
        // precomputed annualized realized-vol estimate otherwise. Updates
        // `position.greeks`/`mark_price`/`unrealized_pnl` for every open
        // position on that symbol via `apply_derivative_fill`'s own bounds
        // (a no-op if nothing is open on it yet, e.g. before entry). A
        // no-op end to end whenever `underlying_series` wasn't supplied.
        if let Some(spot) = config.underlying_series.as_ref()
            .and_then(|s| s.get(tick_idx).copied())
            .filter(|s| s.is_finite() && *s > 0.0)
        {
            let fallback_vol = underlying_realized_vol.get(tick_idx).copied().flatten();
            if let Some(instr) = config.option_instrument.as_ref() {
                if let Some(vol) = select_greeks_vol(iv_surface_for_tick.as_ref(), instr, fallback_vol) {
                    portfolio.update_position_greeks(&instr.symbol, spot, risk_free_rate, vol, timestamp);
                }
            }
            if let Some(legs) = config.option_spread.as_ref() {
                for leg in legs {
                    if let Some(vol) = select_greeks_vol(iv_surface_for_tick.as_ref(), &leg.instrument, fallback_vol) {
                        portfolio.update_position_greeks(&leg.instrument.symbol, spot, risk_free_rate, vol, timestamp);
                    }
                }
            }
        }

        let context = StrategyContext {
            timestamp,
            portfolio_state: PortfolioSnapshot::from(&portfolio),
            price_history: std::mem::take(&mut price_buf),
            volume_history: std::mem::take(&mut volume_buf),
            spread: ob_snap.as_ref().map(|s| s.spread),
            market_session: MarketSession {
                is_open: true,
                session_type: SessionType::Regular,
                next_open: None,
                next_close: None,
            },
            orderbook_snapshot: ob_snap,
            custom_data: std::mem::take(&mut custom_data_buf),
            assets: HashMap::new(),
            // Historical IV overlay (2026-08-29): populated from
            // `config.historical_iv_surfaces` when the caller pre-computed
            // one (self-computed via Black-Scholes inversion of historical
            // option-contract trade prices against the underlying's own
            // historical spot -- no vendor supplies historical IV/quotes
            // directly, confirmed this session). `find_nearest_iv_surface`
            // deliberately returns `None` rather than the nearest FOLLOWING
            // point when every surface is after this tick, to avoid leaking
            // future information into the backtest. `None` (the default,
            // every existing call site) is unchanged prior behavior.
            iv_surface: iv_surface_for_tick,
            futures_curve: None,
            funding_rates: None,
            // portfolio_greeks aggregates whatever positions already have
            // `.greeks` populated -- real per-tick Greeks (2026-09-02) once
            // `update_position_greeks` was called just above for this tick
            // (needs `config.underlying_series` to be set); otherwise still
            // empty/zero exactly as before that wiring existed.
            portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
            tick_number: tick_idx as u64,
            elapsed_seconds: 0.0,
            pool_state: None,
        };
        
        // Reclaim buffers after context is consumed (below)

        // -- Generate signals via executor (Tier 0–3) with stable region optimization --
        let signals = if let Some(ref sigs) = precomputed_signals {
            // Bulk path: convert precomputed i8 → Signal (no Python call this tick).
            last_signals = i8_to_signals(sigs[tick_idx], current_price, tick_idx, config.option_instrument.as_ref(), config.option_spread.as_deref());
            &last_signals
        } else if stable_detector.should_process(current_price) {
            // Price moved significantly or max skip reached — call Python strategy
            diag_on_tick_attempts += 1;
            let new_signals = executor
                .on_tick(data, &context)
                .await
                .unwrap_or_else(|e| {
                    if diag_error_ticks < 5 {
                        log_warn!(BACKTEST_LOGGER, "Executor on_tick error at tick {}: {:?}", tick_idx, e);
                    }
                    if first_on_tick_error.is_none() {
                        first_on_tick_error = Some(e.to_string());
                    }
                    diag_error_ticks += 1;
                    Vec::new()
                });
            last_signals = new_signals;
            &last_signals
        } else {
            // Stable region — reuse cached signals, skip expensive Python call
            stable_region_skips += 1;
            &last_signals
        };

        if signals.is_empty() {
            diag_empty_signal_ticks += 1;
        }

        // Reclaim Vec buffers from the consumed context to avoid re-allocation
        price_buf = context.price_history;
        volume_buf = context.volume_history;
        custom_data_buf = context.custom_data;

        // -- Risk manager per-tick update and global check --
        if let Some(ref mut rm) = risk_manager {
            let current_equity = portfolio.equity_at(current_price);
            rm.update_equity(current_equity);
            let current_drawdown = rm.get_current_drawdown(current_equity);
            let risk_metrics = riskmanager::RiskMetrics {
                current_position: portfolio.net_position,
                current_order_size: 0.0,
                current_inventory_skew: 0.0,
                current_volatility: 0.0,
                current_drawdown,
                unrealized_pnl: portfolio.unrealized_pnl,
                daily_pnl: current_equity - initial_capital,
                portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
                current_var: 0.0,
                current_cvar: 0.0,
                correlated_exposure: 0.0,
                sector_exposures: HashMap::new(),
                volatility_regime: riskmanager::VolatilityRegime::Normal,
                baseline_volatility: 0.0,
            };
            if rm.check(&risk_metrics).is_err() {
                trading_halted = true;
            }
        }

        // -- Margin call / liquidation check for leveraged positions --
        // Enforced independent of whether the optional `risk_manager`
        // feature is enabled for this run -- liquidation is a fundamental
        // mechanic of leverage, not an opt-in risk check. Checked once per
        // tick against the bar's intrabar extreme (bar low for longs, bar
        // high for shorts) so a liquidation isn't missed when price gaps
        // through the trigger within a single bar -- mirrors how this
        // engine's stop-loss/take-profit logic already uses intrabar
        // extremes rather than just the close. No-op at leverage=1.0
        // (`margin::is_liquidated` always returns false).
        //
        // Exemption scope (2026-09-02, narrowed from "any derivative"):
        // ONLY options are exempt here, long or short -- `is_liquidated`
        // models a leveraged-futures-style margin call (a fixed % adverse
        // move against `entry_price`, scaled by `leverage`/
        // `maintenance_margin_ratio`), which doesn't correspond to how a
        // real broker's option margin requirement actually works (a
        // function of the option's own current value plus a percentage of
        // the underlying -- this platform has no such model, and applying
        // the leveraged-futures formula to an option's PREMIUM would
        // produce an economically meaningless trigger price unrelated to
        // real risk). A long option's max loss is bounded by the premium
        // paid regardless, so no circuit breaker is needed there either
        // way. A short/written option's loss is genuinely unbounded and
        // DOES need one -- that's now the account-level `risk_manager`
        // drawdown gate (`equity_at`/`get_current_drawdown`, both fixed to
        // correctly reflect option P&L without an artificial 100% cap
        // earlier this session), not this per-position intrabar check.
        // Futures/perps were PREVIOUSLY also swept into this exemption via
        // the old blanket `instrument.is_some()` condition -- that was a
        // real bug: unlike options, futures/perps DO use the exact
        // leveraged-margin accounting `is_liquidated` models (real
        // `margin_posted`, same `entry_price`/`leverage` semantics as a
        // plain leveraged spot position), so they're no longer exempt.
        if leverage > 1.0 {
            let ohlc = extract_candle_ohlcv(data);
            let triggered = portfolio.positions.iter().any(|p| {
                if is_exempt_from_leverage_liquidation_check(p) {
                    return false;
                }
                let intrabar_extreme = match (p.side, ohlc) {
                    (portfoliomanager::PositionSide::Long, Some((_, _, low, _))) => low,
                    (portfoliomanager::PositionSide::Short, Some((_, high, _, _))) => high,
                    (_, None) => current_price,
                };
                portfoliomanager::margin::is_liquidated(
                    intrabar_extreme, p.entry_price, p.side, leverage, maintenance_margin_ratio,
                )
            });
            if triggered {
                let equity = portfolio.equity_at(current_price);
                let margin_used = portfolio.margin_used();
                let maintenance_margin = portfoliomanager::margin::maintenance_margin(
                    portfolio.net_position.abs() * current_price,
                    maintenance_margin_ratio,
                );
                if matches!(
                    riskmanager::check_margin(equity, margin_used, maintenance_margin),
                    riskmanager::RiskAction::CloseAllPositions(_)
                ) {
                    let liquidation_side = if portfolio.net_position > 0.0 {
                        portfoliomanager::PositionSide::Long
                    } else {
                        portfoliomanager::PositionSide::Short
                    };
                    let liq_exit_price = portfoliomanager::margin::apply_liquidation_penalty(
                        current_price, liquidation_side, liquidation_penalty_bps,
                    );
                    log_warn!(BACKTEST_LOGGER, "Margin call at tick {}: equity={:.2} margin_used={:.2} maintenance_margin={:.2} -- liquidating at {:.4}", tick_idx, equity, margin_used, maintenance_margin, liq_exit_price);
                    close_all_positions(
                        &mut portfolio,
                        liq_exit_price,
                        &mut total_commission,
                        taker_fee,
                        &mut num_trades,
                        &mut trade_returns,
                        &mut trade_maes,
                        &mut trade_log,
                        &mut open_positions,
                        max_trade_log_size,
                        timestamp,
                        slippage_bps,
                        ohlc,
                        synthetic_book_cfg.as_ref(),
                        "margin_call_liquidation",
                    );
                }
            }
        }

        if trading_halted {
            // Record equity but skip signal processing
            if tick_idx % equity_sample_interval == 0 {
                let eq = portfolio.equity_at(current_price);
                equity_curve.push(eq);
                equity_curve_timestamps.push(timestamp.timestamp_millis());
            }
            continue;
        }

        // -- Process signals --
        for signal in signals {
            let sig_price = signal
                .price
                .unwrap_or(current_price);
            let sig_qty = signal.quantity.unwrap_or_else(|| {
                // Auto-size: risk `position_size_pct` of equity per trade when
                // the strategy omits an explicit quantity (e.g. the vectorized
                // direction-only path). Notional (and therefore quantity)
                // scales with leverage; at leverage=1.0 this is identical to
                // the pre-leverage formula (see portfoliomanager::margin).
                //
                // Per-trade dynamic position sizing (2026-07-25): a strategy
                // defining `compute_position_sizes` gets its own fraction for
                // this tick instead of the flat run-wide default, as long as
                // it's a valid fraction of equity. Anything else (no
                // precomputed sizes at all, index out of range, NaN/<=0/>1)
                // falls back to `position_size_pct` exactly as before this
                // feature existed.
                let effective_pct = precomputed_sizes.as_ref()
                    .and_then(|s| s.get(tick_idx).copied())
                    .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
                    .unwrap_or(position_size_pct);
                // Volatility-targeting overlay (opt-in): scales whatever
                // `effective_pct` resolved to above -- flat default or a
                // strategy's own `compute_position_sizes` fraction -- by
                // trailing realized vol vs. the configured annual target.
                // A no-op (returns `effective_pct` unchanged) whenever
                // `vol_target_cfg` is `None` or no trailing vol reading is
                // available yet for this tick.
                let effective_pct = match (&vol_target_cfg, &precomputed_trailing_vol) {
                    (Some(cfg), Some(series)) => portfoliomanager::margin::volatility_scaled_position_size_pct(
                        effective_pct, series.get(tick_idx).copied().flatten(), cfg,
                    ),
                    _ => effective_pct,
                };
                let equity = portfolio.equity_at(current_price);
                let (_, _, qty) = portfoliomanager::margin::size_leveraged_order(
                    equity, effective_pct, leverage, current_price,
                );
                qty.max(0.0)
            });

            // Diagnostic: count signal types and log first few non-hold signals
            match &signal.signal_type {
                SignalType::Buy
                | SignalType::ScaleIn
                | SignalType::BuyOption { .. }
                | SignalType::BuyFuture { .. } => diag_buy_signals += 1,
                SignalType::Sell
                | SignalType::ScaleOut
                | SignalType::SellOption { .. }
                | SignalType::SellFuture { .. } => diag_sell_signals += 1,
                SignalType::Close => diag_close_signals += 1,
                SignalType::Hold | SignalType::CancelQuotes => diag_hold_signals += 1,
                _ => {}
            }
            if !matches!(signal.signal_type, SignalType::Hold) && diag_first_signals_logged < 10 {
                log_debug!(BACKTEST_LOGGER, "tick={} signal={:?} qty={:?} price={:?} reason={:?}", tick_idx, signal.signal_type, signal.quantity, signal.price, signal.reason);
                diag_first_signals_logged += 1;
            }

            match &signal.signal_type {
                SignalType::Buy
                | SignalType::Sell
                | SignalType::ScaleIn
                | SignalType::ScaleOut
                | SignalType::BuyOption { .. }
                | SignalType::SellOption { .. }
                | SignalType::BuyFuture { .. }
                | SignalType::SellFuture { .. }
                | SignalType::ExerciseOption { .. }
                | SignalType::HedgeDelta { .. } => {
                    let Some(directional_signal) = to_directional_signal(&signal.signal_type) else {
                        continue;
                    };
                    let instrument = extract_instrument(&signal.signal_type);
                    if sig_qty <= 0.0 {
                        diag_zero_qty_skips += 1;
                        if diag_zero_qty_skips <= 5 {
                            log_debug!(BACKTEST_LOGGER, "Skipping signal {:?} at tick {} — quantity is 0 or negative", signal.signal_type, tick_idx);
                        }
                        continue;
                    }

                    // Per-order risk check (opt-in)
                    let mut order_qty = sig_qty;
                    if let Some(ref rm) = risk_manager {
                        let current_equity = portfolio.equity_at(current_price);
                        let current_drawdown = rm.get_current_drawdown(current_equity);
                        let risk_metrics = riskmanager::RiskMetrics {
                            current_position: portfolio.net_position,
                            current_order_size: sig_qty,
                            current_inventory_skew: 0.0,
                            current_volatility: 0.0,
                            current_drawdown,
                            unrealized_pnl: portfolio.unrealized_pnl,
                            daily_pnl: current_equity - initial_capital,
                            portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
                            current_var: 0.0,
                            current_cvar: 0.0,
                            correlated_exposure: 0.0,
                            sector_exposures: HashMap::new(),
                            volatility_regime: riskmanager::VolatilityRegime::Normal,
                            baseline_volatility: 0.0,
                        };
                        match rm.check_order(sig_qty, portfolio.net_position, &risk_metrics) {
                            riskmanager::RiskAction::Proceed => {}
                            riskmanager::RiskAction::ReducePosition(allowed) => {
                                order_qty = allowed;
                                if order_qty <= 0.0 { continue; }
                            }
                            riskmanager::RiskAction::ScalePosition(factor, _) => {
                                order_qty *= factor;
                                if order_qty <= 0.0 { continue; }
                            }
                            riskmanager::RiskAction::RejectOrder(_) => { continue; }
                            riskmanager::RiskAction::CloseAllPositions(_)
                            | riskmanager::RiskAction::HaltTrading(_) => {
                                trading_halted = true;
                                continue;
                            }
                        }
                    }

                    // Aggregate exposure cap (2026-08): size_leveraged_order/the
                    // per-order risk check above only bound THIS order's own size --
                    // neither knows whether the strategy is pyramiding into an
                    // already-open position via repeated BUY/ScaleIn (or SELL/
                    // ScaleOut when short) signals. Confirmed live in production
                    // (job 50ac25dd): uncapped pyramiding combined with a pairs
                    // hedge ratio produced a single trade losing 132% of account
                    // equity. Only clamp when this order INCREASES absolute exposure
                    // in the current direction (opening or adding) -- a reducing/
                    // covering order in the opposite direction is always safe to
                    // let through unclamped, since it shrinks net exposure.
                    let is_increasing = match directional_signal {
                        SignalType::Buy => portfolio.net_position >= 0.0,
                        SignalType::Sell => portfolio.net_position <= 0.0,
                        _ => false,
                    };
                    if is_increasing {
                        let equity_now = portfolio.equity_at(current_price);
                        let existing_notional = portfolio.net_position.abs() * current_price;
                        order_qty = portfoliomanager::margin::clamp_aggregate_position_qty(
                            order_qty, current_price, existing_notional, equity_now, leverage,
                        );
                        if order_qty <= 0.0 { continue; }
                    }

                    if signal.is_limit {
                        // Maker/passive — queue the order, fill only when price crosses
                        pending_limits.push(PendingLimit {
                            signal_type: directional_signal.clone(),
                            price: sig_price,
                            remaining_quantity: order_qty,
                            placed_at: timestamp,
                            reason: signal.reason.description(),
                            instrument: instrument.clone(),
                        });
                    } else {
                        // Taker/market — fill immediately at current price
                        execute_taker_fill(
                            &directional_signal,
                            current_price,
                            order_qty,
                            &mut portfolio,
                            &mut total_commission,
                            taker_fee,
                            &mut num_trades,
                            &mut trade_returns,
                            &mut trade_maes,
                            &mut trade_log,
                            &mut open_positions,
                            &mut next_trade_id,
                            max_trade_log_size,
                            timestamp,
                            slippage_bps,
                            extract_candle_ohlcv(data),
                            synthetic_book_cfg.as_ref(),
                            &signal.reason.description(),
                            instrument.as_ref(),
                        );
                    }
                }
                SignalType::OptionSpread { legs } => {
                    if legs.is_empty() {
                        log_warn!(BACKTEST_LOGGER, "Rejecting OptionSpread at tick {}: no legs", tick_idx);
                        continue;
                    }

                    // All-or-nothing: validate and risk-check all legs before any execution.
                    let mut leg_plan: Vec<(SignalType, f64, f64, bool, Option<DerivativeMetadata>)> = Vec::with_capacity(legs.len());
                    let mut rejected = false;

                    for leg in legs {
                        if leg.ratio == 0 {
                            rejected = true;
                            break;
                        }

                        let Some(dir_signal) = to_directional_signal(&leg.signal_type) else {
                            rejected = true;
                            break;
                        };
                        let leg_instrument = extract_instrument(&leg.signal_type);

                        let leg_qty = sig_qty * (leg.ratio.abs() as f64);
                        if leg_qty <= 0.0 {
                            rejected = true;
                            break;
                        }

                        // A real explicit limit_price always wins. Otherwise,
                        // for a TAKER fill, this leg's own price (looked up
                        // from config.option_spread by instrument symbol at
                        // this exact tick_idx) is the correct fill price --
                        // sig_price is the PRIMARY series' own price and is
                        // only correct for whichever leg happens to BE the
                        // primary series; every other leg needs its own.
                        let leg_price = leg.limit_price.unwrap_or_else(|| {
                            spread_leg_price_at(config.option_spread.as_deref(), leg_instrument.as_ref(), tick_idx, sig_price)
                        });
                        let is_limit_leg = leg.limit_price.is_some() || signal.is_limit;

                        if let Some(ref rm) = risk_manager {
                            let current_equity = portfolio.equity_at(current_price);
                            let current_drawdown = rm.get_current_drawdown(current_equity);
                            let risk_metrics = riskmanager::RiskMetrics {
                                current_position: portfolio.net_position,
                                current_order_size: leg_qty,
                                current_inventory_skew: 0.0,
                                current_volatility: 0.0,
                                current_drawdown,
                                unrealized_pnl: portfolio.unrealized_pnl,
                                daily_pnl: current_equity - initial_capital,
                                portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
                                current_var: 0.0,
                                current_cvar: 0.0,
                                correlated_exposure: 0.0,
                                sector_exposures: HashMap::new(),
                                volatility_regime: riskmanager::VolatilityRegime::Normal,
                                baseline_volatility: 0.0,
                            };
                            if !matches!(rm.check_order(leg_qty, portfolio.net_position, &risk_metrics), riskmanager::RiskAction::Proceed) {
                                rejected = true;
                                break;
                            }
                        }

                        leg_plan.push((dir_signal, leg_qty, leg_price, is_limit_leg, leg_instrument));
                    }

                    if rejected {
                        log_warn!(BACKTEST_LOGGER, "Rejecting OptionSpread at tick {}: failed leg validation/risk check", tick_idx);
                        continue;
                    }

                    for (dir_signal, leg_qty, leg_price, is_limit_leg, leg_instrument) in leg_plan {
                        if is_limit_leg {
                            pending_limits.push(PendingLimit {
                                signal_type: dir_signal,
                                price: leg_price,
                                remaining_quantity: leg_qty,
                                placed_at: timestamp,
                                reason: signal.reason.description(),
                                instrument: leg_instrument,
                            });
                        } else {
                            execute_taker_fill(
                                &dir_signal,
                                leg_price,
                                leg_qty,
                                &mut portfolio,
                                &mut total_commission,
                                taker_fee,
                                &mut num_trades,
                                &mut trade_returns,
                                &mut trade_maes,
                                &mut trade_log,
                                &mut open_positions,
                                &mut next_trade_id,
                                max_trade_log_size,
                                timestamp,
                                slippage_bps,
                                extract_candle_ohlcv(data),
                                synthetic_book_cfg.as_ref(),
                                &signal.reason.description(),
                                leg_instrument.as_ref(),
                            );
                        }
                    }
                }
                SignalType::Close => {
                    // Close all open positions at market (taker)
                    close_all_positions(
                        &mut portfolio,
                        current_price,
                        &mut total_commission,
                        taker_fee,
                        &mut num_trades,
                        &mut trade_returns,
                        &mut trade_maes,
                        &mut trade_log,
                        &mut open_positions,
                        max_trade_log_size,
                        timestamp,
                        slippage_bps,
                        extract_candle_ohlcv(data),
                        synthetic_book_cfg.as_ref(),
                        &signal.reason.description(),
                    );
                }
                SignalType::StopLoss | SignalType::TakeProfit => {
                    // Emergency exits always use taker fill
                    let exit_side = if portfolio.net_position > 0.0 {
                        SignalType::Sell
                    } else {
                        SignalType::Buy
                    };
                    let exit_qty = portfolio.net_position.abs();
                    if exit_qty > 0.0 {
                        // `instrument: None` -- this emergency exit closes the
                        // aggregate net_position as a single spot-style fill,
                        // same as before this change. It doesn't (and, being
                        // a single net-position exit rather than a per-
                        // position one, structurally can't yet) distinguish
                        // which specific option contract to close when
                        // multiple are open simultaneously -- unlike
                        // `close_all_positions`, which already closes each
                        // position individually and does thread its
                        // instrument through.
                        execute_taker_fill(
                            &exit_side,
                            current_price,
                            exit_qty,
                            &mut portfolio,
                            &mut total_commission,
                            taker_fee,
                            &mut num_trades,
                            &mut trade_returns,
                            &mut trade_maes,
                            &mut trade_log,
                            &mut open_positions,
                            &mut next_trade_id,
                            max_trade_log_size,
                            timestamp,
                            slippage_bps,
                            extract_candle_ohlcv(data),
                            synthetic_book_cfg.as_ref(),
                            &signal.reason.description(),
                            None,
                        );
                    }
                }
                SignalType::Hold | SignalType::CancelQuotes => {
                    // CancelQuotes in a directional strategy context = clear pending limits
                    if matches!(signal.signal_type, SignalType::CancelQuotes) {
                        pending_limits.clear();
                    }
                }
                SignalType::TwoSidedQuote { .. } => {
                    // Directional strategies shouldn't normally emit quotes,
                    // but if they do, treat them as two pending limits
                    if let SignalType::TwoSidedQuote {
                        bid_price,
                        ask_price,
                        bid_size,
                        ask_size,
                    } = &signal.signal_type
                    {
                        pending_limits.push(PendingLimit {
                            signal_type: SignalType::Buy,
                            price: *bid_price,
                            remaining_quantity: *bid_size,
                            placed_at: timestamp,
                            reason: signal.reason.description(),
                            instrument: None,
                        });
                        pending_limits.push(PendingLimit {
                            signal_type: SignalType::Sell,
                            price: *ask_price,
                            remaining_quantity: *ask_size,
                            placed_at: timestamp,
                            reason: signal.reason.description(),
                            instrument: None,
                        });
                    }
                }
                SignalType::CrossAsset { .. }
                | SignalType::Custom(_)
                | SignalType::RollContract { .. }
                | SignalType::AddLiquidity { .. }
                | SignalType::RemoveLiquidity { .. }
                | SignalType::Rebalance { .. } => {
                    // Not supported in single-symbol simulation
                }
            }
        }

        // -- Expiration handling: force-close any option contract whose
        // expiry has passed as of this tick's timestamp. --
        //
        // This uses the contract's own last traded price (`current_price` in
        // this single-instrument loop IS the option contract's own OHLCV
        // series, per Phase 0 -- `MarketData::OptionCandle`/
        // `fetch_option_aggregates`) as the closing price, NOT a Black-
        // Scholes strike-vs-underlying-spot intrinsic-value settlement.
        // `PortfolioState::settle_expired_options` (the intrinsic-value
        // version) exists and is unit-tested in `portfoliomanager`, but needs
        // a genuine underlying-spot series as input -- this single-contract
        // candle loop doesn't carry one. Closing at last market price is the
        // honest choice for what data is actually available here: a real
        // option's quoted price already converges toward intrinsic value as
        // expiration approaches, so this is a reasonable approximation, not
        // a silent shortcut. Wiring true intrinsic-value settlement is a
        // natural follow-up once a portfolio job tracks the underlying's own
        // series alongside each contract.
        let expired: Vec<(portfoliomanager::PositionSide, f64, DerivativeMetadata)> = portfolio
            .positions
            .iter()
            .filter(|p| p.close_time.is_none())
            .filter_map(|p| {
                let instr = p.instrument.as_ref()?;
                let expiry = instr.instrument_kind.expiry()?;
                if expiry <= timestamp {
                    Some((p.side.clone(), p.quantity, instr.clone()))
                } else {
                    None
                }
            })
            .collect();
        for (side, qty, instr) in expired {
            let exit_signal = match side {
                portfoliomanager::PositionSide::Long => SignalType::Sell,
                portfoliomanager::PositionSide::Short => SignalType::Buy,
            };
            let mut dummy_id = 0usize;
            execute_taker_fill(
                &exit_signal,
                current_price,
                qty,
                &mut portfolio,
                &mut total_commission,
                taker_fee,
                &mut num_trades,
                &mut trade_returns,
                &mut trade_maes,
                &mut trade_log,
                &mut open_positions,
                &mut dummy_id,
                max_trade_log_size,
                timestamp,
                slippage_bps,
                extract_candle_ohlcv(data),
                synthetic_book_cfg.as_ref(),
                "contract_expiration",
                Some(&instr),
            );
        }

        // -- Update portfolio mark-to-market --
        portfolio.update_unrealized_pnl(current_price);
        let total_value = portfolio.get_total_value();
        if tick_idx % equity_sample_interval == 0 || tick_idx == total_ticks - 1 {
            equity_curve.push(total_value);
            equity_curve_timestamps.push(timestamp.timestamp_millis());
        }

        // -- Progress callback --
        if let Some(ref cb) = config.progress_callback {
            if tick_idx % 1000 == 0 || tick_idx == total_ticks - 1 {
                cb(tick_idx + 1, total_ticks);
            }
        }
    }

    // -- Build result --
    let final_equity = equity_curve.last().copied().unwrap_or(initial_capital);
    let total_pnl = final_equity - initial_capital;

    // Close out remaining open positions as "still open" trade records
    for pos in open_positions.drain(..) {
        let should_log = match max_trade_log_size {
            Some(max) => trade_log.len() < max,
            None => true,
        };
        if should_log {
            // Compute unrealized P&L at final price
            let unrealized_pnl = if pos.side == "long" {
                (last_price - pos.entry_fill_price) * pos.quantity - pos.entry_commission
            } else {
                (pos.entry_fill_price - last_price) * pos.quantity - pos.entry_commission
            };
            let unrealized_pct = unrealized_pnl / (pos.entry_fill_price * pos.quantity);
            trade_log.push(TradeRecord {
                trade_id: pos.trade_id,
                side: pos.side,
                entry_signal_price: pos.entry_signal_price,
                entry_fill_price: pos.entry_fill_price,
                exit_signal_price: None,
                exit_fill_price: None,
                quantity: pos.quantity,
                pnl: Some(unrealized_pnl),
                pnl_pct: Some(unrealized_pct),
                commission: pos.entry_commission,
                slippage_cost: pos.entry_slippage,
                entry_time: pos.entry_time,
                exit_time: None,
                duration_secs: None,
                entry_liquidity: pos.entry_liquidity,
                exit_liquidity: None,
                entry_reason: pos.entry_reason,
                exit_reason: None,
                mae: Some(pos.worst_unrealized),
                mfe: Some(pos.best_unrealized),
                legs: Vec::new(),
            });
        }
    }

    log_info!(BACKTEST_LOGGER, "Python directional simulation complete: {} trades, PnL=${:.2}, equity=${:.2}", num_trades, total_pnl, final_equity);
    log_info!(BACKTEST_LOGGER, "Signal Summary ({} ticks): Buy={} Sell={} Close={} Hold={} ZeroQty={} Errors={} Trades={}", total_ticks, diag_buy_signals, diag_sell_signals, diag_close_signals, diag_hold_signals, diag_zero_qty_skips, diag_error_ticks, num_trades);

    // Hard-fail on a 100%-error-rate run instead of silently "completing"
    // with 0 trades. Previously the per-tick `on_tick()` error path only
    // logged a throttled warning (see diag_error_ticks above) -- a strategy
    // that failed on literally every attempted tick (e.g. an initialization
    // failure that only manifests once real tick execution runs) produced a
    // "completed" backtest indistinguishable from a legitimate strategy
    // that simply chose not to trade. Only fires when EVERY attempt failed
    // (never a partial/transient-error case, which can still produce real
    // trades from its successful ticks and shouldn't be retroactively
    // invalidated) and the run genuinely produced nothing.
    if diag_on_tick_attempts > 0 && diag_error_ticks == diag_on_tick_attempts && num_trades == 0 {
        let msg = first_on_tick_error.unwrap_or_else(|| "unknown error".to_string());
        anyhow::bail!(
            "Strategy execution failed on every attempted tick ({} of {}): {}. Fix the error in your strategy code.",
            diag_error_ticks, diag_on_tick_attempts, msg
        );
    }
    if stable_region_skips > 0 {
        log_info!(BACKTEST_LOGGER, "Stable region optimization: skipped {} of {} ticks ({:.1}%)", stable_region_skips, total_ticks, (stable_region_skips as f64 / total_ticks as f64) * 100.0);
    }

    // If the vectorized path didn't precompute signal_counts (i.e. this is the
    // per-tick `generate_signals` runtime path), populate it from the diagnostic
    // counters so the persisted BacktestResult always carries a non-null breakdown.
    if signal_counts.is_none() {
        signal_counts = Some(crate::types::SignalCounts {
            buy:   diag_buy_signals   as usize,
            sell:  diag_sell_signals  as usize,
            close: diag_close_signals as usize,
            hold:  diag_hold_signals  as usize,
        });
    }

    Ok(PythonSimResult {
        backtest_result: build_backtest_result(
            total_pnl,
            num_trades,
            &equity_curve,
            &equity_curve_timestamps,
            &trade_returns,
            &trade_maes,
            trade_log,
            initial_capital,
            total_commission,
            first_price,
            last_price,
            &portfolio,
            price_series,
            signal_counts,
            compute_signals_error,
            strategy_display_name,
            computed_features,
            captured_prices_for_features,
            captured_timestamps_for_features,
            volume_constrained_fills,
            config.option_instrument.as_ref(),
            config.option_spread.as_deref(),
        ),
        total_commission,
        pending_limits_at_end: pending_limits.len(),
        execution_tier,
    })
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a Python directional simulation.
pub struct PythonSimResult {
    /// Standard backtest result (compatible with all downstream consumers).
    pub backtest_result: BacktestResult,
    /// Total commission/fees paid.
    pub total_commission: f64,
    /// Number of unfilled limit orders remaining at end.
    pub pending_limits_at_end: usize,
    /// The execution tier used for this run.
    pub execution_tier: ExecutionTier,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A pending limit order waiting for the market to cross its price.
struct PendingLimit {
    signal_type: SignalType,
    price: f64,
    /// Remaining quantity (for partial fills when volume-constrained).
    remaining_quantity: f64,
    /// Timestamp when the order was placed (used for latency gating).
    placed_at: DateTime<Utc>,
    /// Signal reason string.
    reason: String,
    /// Derivative contract metadata, carried from the originating
    /// `BuyOption`/`SellOption`/etc. signal so the eventual fill can be
    /// applied via `apply_derivative_fill` (multiplier-aware, correct
    /// long/short bookkeeping) instead of the plain spot fill path.
    /// `None` for ordinary Buy/Sell signals.
    instrument: Option<DerivativeMetadata>,
}

/// An open position being tracked for round-trip trade log.
struct OpenPosition {
    trade_id: usize,
    side: String,
    entry_signal_price: f64,
    entry_fill_price: f64,
    quantity: f64,
    entry_commission: f64,
    entry_slippage: f64,
    entry_time: DateTime<Utc>,
    entry_liquidity: String,
    entry_reason: String,
    /// Tracks worst unrealized loss during this position (fraction).
    worst_unrealized: f64,
    /// Tracks best unrealized gain during this position (fraction).
    best_unrealized: f64,
}

/// Extract (price, volume, timestamp) from any MarketData variant.
fn extract_price_volume_ts(data: &MarketData) -> (f64, f64, DateTime<Utc>) {
    match data {
        MarketData::Trade(t) => (
            t.price,
            t.quantity,
            t.timestamp,
        ),
        MarketData::Candle(c) => (
            c.close,
            c.volume,
            c.timestamp,
        ),
        MarketData::PoolSwap(s) => (
            s.price(),
            s.amount_in,
            s.timestamp,
        ),
        MarketData::Generic(g) => (
            g.price,
            g.quantity,
            DateTime::from_timestamp_millis(g.timestamp_ms)
                .unwrap_or_else(|| Utc::now()),
        ),
        MarketData::OptionCandle(c) => (
            c.close,
            c.volume,
            c.timestamp,
        ),
    }
}

/// Extract full OHLCV plus timestamp from a market data point.
///
/// Returns `(open, high, low, close, volume, timestamp)`. For data that has no
/// real candle (trades / generic ticks without OHLC columns) the bar
/// degenerates to the tick price — open=high=low=close=price — so close-bar
/// behavior is preserved while candle datasets surface true OHLC.
fn extract_ohlcv_ts(data: &MarketData) -> (f64, f64, f64, f64, f64, DateTime<Utc>) {
    match data {
        MarketData::Trade(t) => (
            t.price, t.price, t.price, t.price,
            t.quantity,
            t.timestamp,
        ),
        MarketData::Candle(c) => (
            c.open, c.high, c.low, c.close,
            c.volume,
            c.timestamp,
        ),
        MarketData::PoolSwap(s) => {
            let p = s.price();
            (
                p, p, p, p,
                s.amount_in,
                s.timestamp,
            )
        }
        MarketData::Generic(g) => {
            let close = g.close.unwrap_or(g.price);
            (
                g.open.unwrap_or(close),
                g.high.unwrap_or(close),
                g.low.unwrap_or(close),
                close,
                g.volume.unwrap_or(g.quantity),
                DateTime::from_timestamp_millis(g.timestamp_ms)
                    .unwrap_or_else(|| Utc::now()),
            )
        }
        MarketData::OptionCandle(c) => (
            c.open, c.high, c.low, c.close,
            c.volume,
            c.timestamp,
        ),
    }
}

/// Resolve the RUN-WIDE FALLBACK position size (as a fraction of equity) used
/// when a `Signal` carries no explicit quantity — i.e. the vectorized
/// direction-only (`compute_signals`) path and any `generate_signals` signal
/// that omits `quantity=`. As of 2026-07-25 this is genuinely just the
/// fallback: a strategy defining `compute_position_sizes` gets a per-tick
/// fraction from that instead (see the `precomputed_sizes`/`effective_pct`
/// resolution around the `sig_qty` closure), and only falls through to this
/// flat value for ticks where the per-trade result is missing or invalid.
///
/// Resolution order (first match wins):
///   1. `config.parameters["position_size_pct"]` or `["risk_per_trade"]` —
///      a GA-injected value or an explicit override.
///   2. The strategy's `parameter_space()` default for those keys, so plain
///      and quick (non-GA) runs honor an AI-declared sizing knob. Only consulted
///      for non-GA runs (`max_trade_log_size == None`) to avoid re-instantiating
///      Python on every GA individual evaluation.
///   3. Fallback `0.02` (2% of equity) — preserves historical behavior for
///      strategies that declare no sizing parameter.
///
/// The result is clamped to `(0.0, 1.0]` (a single entry may risk at most 100%
/// of current equity).
fn resolve_position_size_pct(config: &PythonSimConfig) -> f64 {
    const DEFAULT_PCT: f64 = 0.02;

    let from_override = config
        .parameters
        .get("position_size_pct")
        .or_else(|| config.parameters.get("risk_per_trade"))
        .and_then(|v| v.as_float());

    let resolved = from_override.or_else(|| {
        // GA inner evaluations (capped trade log) carry injected params already;
        // skip Python introspection to keep per-individual evaluation cheap.
        if config.max_trade_log_size.is_some() {
            return None;
        }
        #[cfg(feature = "python")]
        {
            crate::python_validation::extract_parameter_schema(&config.python_source).and_then(
                |schema| {
                    let defaults = schema.to_default_map();
                    defaults
                        .get("position_size_pct")
                        .or_else(|| defaults.get("risk_per_trade"))
                        .and_then(|v| v.as_f64())
                },
            )
        }
        #[cfg(not(feature = "python"))]
        {
            None
        }
    });

    match resolved {
        Some(p) if p.is_finite() && p > 0.0 => p.min(1.0),
        _ => DEFAULT_PCT,
    }
}

/// Convert a bulk-signal i8 code to a `Vec<Signal>`.
///
/// Signal encoding: 1=BUY, -1=SELL, 0=HOLD, 2=CLOSE
///
/// `option_spread`, when set (and non-empty), takes precedence over
/// `option_instrument`: BUY(1) reinterprets as `SignalType::OptionSpread`
/// entering every leg AS CONFIGURED (a leg with positive `ratio` opens long
/// via `BuyOption`, negative opens short via `SellOption`), and SELL(-1) as
/// the MIRROR -- every leg's direction and ratio sign negated, so the same
/// spread closes (or, for a leg `apply_derivative_fill` finds no matching
/// open position for, opens the inverse) exactly the way a plain SELL
/// closes a plain BUY. Each leg's fill price comes from ITS OWN `prices[
/// tick_idx]`, not the shared `price` parameter -- see `OptionSpreadLeg`'s
/// doc comment for why every leg needs its own aligned series. Falls back
/// to `price` (with a loud warning) if a leg's series is shorter than
/// `tick_idx` requires -- callers are responsible for aligning every leg's
/// series to the primary market_data length; this is a defensive
/// fallback for a caller bug, not expected in practice.
///
/// `option_instrument`, when set and `option_spread` is not, reinterprets
/// BUY(1)/SELL(-1) as `SignalType::BuyOption`/`SellOption` on that single
/// contract instead of plain spot `Buy`/`Sell` -- see `PythonSimConfig.
/// option_instrument`'s doc comment for the full rationale.
///
/// CLOSE(2) is deliberately unchanged in every case: `close_all_positions`
/// already closes each open position (single-leg or spread-leg alike)
/// using its own recorded instrument, not a signal-level one.
fn i8_to_signals(
    code: i8,
    price: f64,
    tick_idx: usize,
    option_instrument: Option<&DerivativeMetadata>,
    option_spread: Option<&[OptionSpreadLeg]>,
) -> Vec<Signal> {
    let reason = SignalReason::TechnicalAnalysis("vectorized".into());

    let spread_leg_price = |leg: &OptionSpreadLeg| -> f64 {
        leg.prices.get(tick_idx).copied().unwrap_or_else(|| {
            log_warn!(
                BACKTEST_LOGGER,
                "OptionSpreadLeg '{}' has no price at tick {} (series len {}) -- falling back to the primary series' price, which is WRONG for this contract; the caller must align every leg's series to market_data's length",
                leg.instrument.symbol, tick_idx, leg.prices.len()
            );
            price
        })
    };
    let build_spread_signal = |legs: &[OptionSpreadLeg], entering: bool| -> Signal {
        let spread_legs: Vec<SpreadLeg> = legs.iter().map(|leg| {
            let leg_price = spread_leg_price(leg);
            let opens_long = if entering { leg.ratio > 0 } else { leg.ratio < 0 };
            let effective_ratio = if entering { leg.ratio } else { -leg.ratio };
            let signal_type = if opens_long {
                SignalType::BuyOption { instrument: leg.instrument.clone(), premium: Some(leg_price) }
            } else {
                SignalType::SellOption { instrument: leg.instrument.clone(), premium: Some(leg_price) }
            };
            SpreadLeg { signal_type, ratio: effective_ratio, limit_price: None }
        }).collect();
        Signal::new("bulk", SignalType::OptionSpread { legs: spread_legs }, SignalStrength::Medium, reason.clone())
    };

    match code {
        1 => {
            if let Some(legs) = option_spread.filter(|l| !l.is_empty()) {
                return vec![build_spread_signal(legs, true).with_price(price)];
            }
            vec![match option_instrument {
                Some(instr) => Signal::new("bulk", SignalType::BuyOption { instrument: instr.clone(), premium: None }, SignalStrength::Medium, reason),
                None => Signal::buy("bulk", SignalStrength::Medium, reason),
            }.with_price(price)]
        }
        -1 => {
            if let Some(legs) = option_spread.filter(|l| !l.is_empty()) {
                return vec![build_spread_signal(legs, false).with_price(price)];
            }
            vec![match option_instrument {
                Some(instr) => Signal::new("bulk", SignalType::SellOption { instrument: instr.clone(), premium: None }, SignalStrength::Medium, reason),
                None => Signal::sell("bulk", SignalStrength::Medium, reason),
            }.with_price(price)]
        }
        2  => vec![Signal::close("bulk", SignalStrength::Medium, reason).with_price(price)],
        _  => Vec::new(), // HOLD or any unknown code
    }
}

/// Extract candle OHLCV as (close, high, low, volume) if the data is a candle.
fn extract_candle_ohlcv(data: &MarketData) -> Option<(f64, f64, f64, f64)> {
    match data {
        MarketData::Candle(c) => Some((c.close, c.high, c.low, c.volume)),
        _ => None,
    }
}

/// Map derivative-aware signal variants to directional base signals used by
/// the single-symbol directional simulator.
fn to_directional_signal(signal_type: &SignalType) -> Option<SignalType> {
    match signal_type {
        SignalType::Buy
        | SignalType::ScaleIn
        | SignalType::BuyOption { .. }
        | SignalType::BuyFuture { .. } => Some(SignalType::Buy),
        SignalType::Sell
        | SignalType::ScaleOut
        | SignalType::SellOption { .. }
        | SignalType::SellFuture { .. }
        | SignalType::ExerciseOption { .. } => Some(SignalType::Sell),
        SignalType::HedgeDelta {
            target_delta,
            current_delta,
            ..
        } => {
            if current_delta > target_delta {
                Some(SignalType::Sell)
            } else {
                Some(SignalType::Buy)
            }
        }
        _ => None,
    }
}

/// Which round-trip action a Buy/Sell(-family) fill represents, given the
/// side of whatever position (if any) `open_positions` currently has open.
/// Real bug (confirmed live, 2026-09-02, harvesting-skill eval): the two
/// callers of this (`execute_taker_fill`, `fill_pending_limits`) used to
/// classify purely off the signal type -- ANY Buy/ScaleIn was "open a new
/// long", ANY Sell/ScaleOut was "close" -- which silently dropped a bare
/// Sell arriving while flat from the round-trip ledger entirely (it matched
/// neither "close" (nothing open) nor "entry" (not a Buy)), even though
/// `PortfolioState::apply_fill_executed` already opens a real short for it
/// (see that function's own doc comment -- the portfolio-level short-open
/// bug was already fixed; this reporting-layer companion never was). A
/// later Buy meant to cover that (invisible-to-the-ledger) short then got
/// misclassified as a fresh "long" entry instead of a close, corrupting
/// `trade_log`'s `side`/`pnl` for every strategy that ever goes short.
fn classify_round_trip_action(is_buy_direction: bool, open_positions: &[OpenPosition]) -> RoundTripAction {
    match (is_buy_direction, open_positions.last().map(|p| p.side.as_str())) {
        (true, Some("short")) => RoundTripAction::Close,
        (false, Some("long")) => RoundTripAction::Close,
        (true, _) => RoundTripAction::OpenLong,
        (false, _) => RoundTripAction::OpenShort,
    }
}

enum RoundTripAction {
    OpenLong,
    OpenShort,
    Close,
}

/// Pull the `DerivativeMetadata` out of a signal that carries one, so the
/// fill path can apply it via `apply_derivative_fill` (multiplier-aware,
/// long/short-correct) instead of collapsing straight to a plain spot fill.
/// `None` for ordinary Buy/Sell/ScaleIn/ScaleOut and non-derivative signals.
fn extract_instrument(signal_type: &SignalType) -> Option<DerivativeMetadata> {
    match signal_type {
        SignalType::BuyOption { instrument, .. }
        | SignalType::SellOption { instrument, .. }
        | SignalType::BuyFuture { instrument }
        | SignalType::SellFuture { instrument }
        | SignalType::ExerciseOption { instrument } => Some(instrument.clone()),
        _ => None,
    }
}

/// Look up an `OptionSpread` leg's own price at `tick_idx`, matched against
/// `option_spread` by instrument symbol. Falls back to `fallback` (with a
/// loud warning) when `option_spread` is `None`, the symbol isn't found, or
/// the matched leg's series doesn't reach `tick_idx` -- each of these means
/// this fill is either not actually part of a configured spread (a plain
/// single-leg BuyOption/SellOption reusing the same taker-fill machinery,
/// where `fallback` -- `sig_price` at the call site -- is already correct)
/// or a genuine caller alignment bug, never something to silently mis-price
/// without at least logging it.
fn spread_leg_price_at(
    option_spread: Option<&[OptionSpreadLeg]>,
    instrument: Option<&DerivativeMetadata>,
    tick_idx: usize,
    fallback: f64,
) -> f64 {
    let (Some(legs), Some(instr)) = (option_spread, instrument) else {
        return fallback;
    };
    match legs.iter().find(|l| l.instrument.symbol == instr.symbol) {
        Some(leg) => leg.prices.get(tick_idx).copied().unwrap_or_else(|| {
            log_warn!(
                BACKTEST_LOGGER,
                "OptionSpreadLeg '{}' has no price at tick {} (series len {}) -- falling back to {}, which is WRONG for this contract",
                instr.symbol, tick_idx, leg.prices.len(), fallback
            );
            fallback
        }),
        None => fallback,
    }
}

/// Commission for one fill. An option instrument (`is_option()`) uses the
/// real per-contract Alpaca fee schedule (`config::OptionsFeeConfig`) --
/// flat regulatory pass-throughs, not a percentage of premium -- rather
/// than `generic_fee_rate` (this platform's only other broker integration
/// for the same underlying asset class; see `OptionsFeeConfig::alpaca`'s
/// doc comment for why). Any other instrument (a future, say) still applies
/// `generic_fee_rate` but against the correctly-scaled notional (`price *
/// qty * contract_multiplier`) -- fixes a second, compounding bug this fee
/// model shared with options: `fill_value` was previously computed as
/// `price * qty` with no multiplier at all, understating notional by
/// exactly the multiplier factor for every derivative instrument, not just
/// options. A plain (non-derivative) fill keeps the pre-existing `price *
/// qty * generic_fee_rate` behavior exactly, since its implicit multiplier
/// is 1.
fn compute_commission(price: f64, qty: f64, is_buy: bool, instrument: Option<&DerivativeMetadata>, generic_fee_rate: f64) -> f64 {
    match instrument {
        Some(instr) if instr.instrument_kind.is_option() => {
            config::OptionsFeeConfig::alpaca().commission_for_fill(price, qty, instr.contract_multiplier, is_buy)
        }
        Some(instr) => price * qty * instr.contract_multiplier * generic_fee_rate,
        None => price * qty * generic_fee_rate,
    }
}

/// Execute a taker (market) fill immediately, applying slippage.
/// Tracks round-trip trades: Buy/ScaleIn opens a position, Sell/ScaleOut closes it.
fn execute_taker_fill(
    signal_type: &SignalType,
    fill_price: f64,
    fill_qty: f64,
    portfolio: &mut PortfolioState,
    total_commission: &mut f64,
    taker_fee: f64,
    num_trades: &mut usize,
    trade_returns: &mut Vec<f64>,
    trade_maes: &mut Vec<f64>,
    trade_log: &mut Vec<TradeRecord>,
    open_positions: &mut Vec<OpenPosition>,
    next_trade_id: &mut usize,
    max_trade_log_size: Option<usize>,
    timestamp: DateTime<Utc>,
    slippage_bps: f64,
    candle_ohlcv: Option<(f64, f64, f64, f64)>,
    synthetic_book_config: Option<&SyntheticBookConfig>,
    signal_reason: &str,
    instrument: Option<&DerivativeMetadata>,
) {
    let (adjusted_price, actual_slippage_bps) = match (candle_ohlcv, synthetic_book_config) {
        (Some((close, high, low, volume)), Some(cfg)) => {
            let book = SyntheticOrderBook::from_candle(close, high, low, volume, cfg);
            let side = match signal_type {
                SignalType::Buy | SignalType::ScaleIn => orderbook::BookSide::Bid,
                _ => orderbook::BookSide::Ask,
            };
            book.calculate_slippage(fill_qty, side)
        }
        _ => {
            let factor = slippage_bps / 10_000.0;
            let price = match signal_type {
                SignalType::Buy | SignalType::ScaleIn => fill_price * (1.0 + factor),
                _ => fill_price * (1.0 - factor),
            };
            (price, slippage_bps)
        }
    };

    let fill = build_fill(
        signal_type,
        adjusted_price,
        fill_qty,
        LiquidityType::Taker,
        timestamp,
        Some(fill_price),
        Some(actual_slippage_bps),
    );
    // Bug #34 fix: execution may be capped by available balance (buys) or
    // inventory (sells), or hit no inventory at all. Charge commission and
    // record the trade based on the quantity that ACTUALLY executed, never the
    // requested quantity — otherwise phantom fees on unexecuted/partial fills
    // depress equity below the sum of realized round-trip P&L (the equity-curve
    // reconciliation gap).
    //
    // `instrument.is_some()` means this fill came from a BuyOption/SellOption/
    // BuyFuture/SellFuture/ExerciseOption signal — apply it through
    // `apply_derivative_fill` (multiplier-aware, correctly opens/covers a
    // short rather than assuming every Ask-side fill closes a long) instead
    // of the plain spot path.
    let executed_qty = match instrument {
        Some(instr) => portfolio.apply_derivative_fill(&fill, instr),
        None => portfolio.apply_fills_executed(&[fill]),
    };
    let executed_qty = match executed_qty {
        Ok(q) => q,
        Err(e) => {
            log_warn!(BACKTEST_LOGGER, "Taker fill error: {}", e);
            0.0
        }
    };
    if executed_qty < 1e-10 {
        // Nothing executed — no trade, no fees.
        return;
    }
    let fill_qty = executed_qty;

    let fill_value = adjusted_price * fill_qty;
    let is_buy = matches!(signal_type, SignalType::Buy | SignalType::ScaleIn);
    let commission = compute_commission(adjusted_price, fill_qty, is_buy, instrument, taker_fee);
    *total_commission += commission;
    *num_trades += 1;
    portfolio.fees_paid += commission;
    portfolio.balance -= commission;

    // Track trade return
    let pnl_pct = if portfolio.balance > 0.0 {
        let direction = if matches!(signal_type, SignalType::Buy | SignalType::ScaleIn) {
            -1.0
        } else {
            1.0
        };
        (fill_value * direction) / portfolio.get_total_value()
    } else {
        0.0
    };
    trade_returns.push(pnl_pct);

    let is_buy_direction = matches!(signal_type, SignalType::Buy | SignalType::ScaleIn);
    let action = classify_round_trip_action(is_buy_direction, open_positions);
    let is_exit = matches!(action, RoundTripAction::Close);

    // MAE: for exits, convert the round-trip's fractional MAE to dollars;
    // for entries, no completed trade yet → 0.
    let mae_dollars = if is_exit && !open_positions.is_empty() {
        let pos = &open_positions[open_positions.len() - 1];
        pos.worst_unrealized.abs() * pos.entry_fill_price * pos.quantity
    } else {
        0.0
    };
    trade_maes.push(mae_dollars);
    let slippage_cost = (adjusted_price - fill_price).abs() * fill_qty;

    let should_log = match max_trade_log_size {
        Some(max) => trade_log.len() < max,
        None => true,
    };

    if is_exit && !open_positions.is_empty() {
        // Close the oldest matching open position (FIFO)
        if let Some(pos) = open_positions.pop() {
            if should_log {
                let exit_slippage = slippage_cost;
                let raw_pnl = if pos.side == "long" {
                    (adjusted_price - pos.entry_fill_price) * pos.quantity
                } else {
                    (pos.entry_fill_price - adjusted_price) * pos.quantity
                };
                let total_comm = pos.entry_commission + commission;
                let net_pnl = raw_pnl - total_comm;
                let net_pnl_pct = net_pnl / (pos.entry_fill_price * pos.quantity);
                let duration = (timestamp - pos.entry_time).num_seconds();

                trade_log.push(TradeRecord {
                    trade_id: pos.trade_id,
                    side: pos.side,
                    entry_signal_price: pos.entry_signal_price,
                    entry_fill_price: pos.entry_fill_price,
                    exit_signal_price: Some(fill_price),
                    exit_fill_price: Some(adjusted_price),
                    quantity: pos.quantity,
                    pnl: Some(net_pnl),
                    pnl_pct: Some(net_pnl_pct),
                    commission: total_comm,
                    slippage_cost: pos.entry_slippage + exit_slippage,
                    entry_time: pos.entry_time,
                    exit_time: Some(timestamp),
                    duration_secs: Some(duration),
                    entry_liquidity: pos.entry_liquidity,
                    exit_liquidity: Some("taker".to_string()),
                    entry_reason: pos.entry_reason,
                    exit_reason: Some(signal_reason.to_string()),
                    mae: Some(pos.worst_unrealized),
                    mfe: Some(pos.best_unrealized),
                    legs: Vec::new(),
                });
            }
        }
    } else {
        // Open a new position for round-trip tracking -- long for a bare
        // Buy/ScaleIn, short for a bare Sell/ScaleOut (see
        // `classify_round_trip_action`'s doc comment for why this can't
        // just assume "long").
        let side = if matches!(action, RoundTripAction::OpenShort) { "short" } else { "long" };
        open_positions.push(OpenPosition {
            trade_id: *next_trade_id,
            side: side.to_string(),
            entry_signal_price: fill_price,
            entry_fill_price: adjusted_price,
            quantity: fill_qty,
            entry_commission: commission,
            entry_slippage: slippage_cost,
            entry_time: timestamp,
            entry_liquidity: "taker".to_string(),
            entry_reason: signal_reason.to_string(),
            worst_unrealized: 0.0,
            best_unrealized: 0.0,
        });
        *next_trade_id += 1;
    }
}
fn fill_pending_limits(
    pending: &mut Vec<PendingLimit>,
    current_price: f64,
    portfolio: &mut PortfolioState,
    total_commission: &mut f64,
    maker_fee: f64,
    num_trades: &mut usize,
    trade_returns: &mut Vec<f64>,
    trade_maes: &mut Vec<f64>,
    trade_log: &mut Vec<TradeRecord>,
    open_positions: &mut Vec<OpenPosition>,
    next_trade_id: &mut usize,
    max_trade_log_size: Option<usize>,
    timestamp: DateTime<Utc>,
    latency_ms: u64,
    adverse_selection: bool,
    current_volume: f64,
    fill_volume_fraction: f64,
    volume_constrained: &mut usize,
) -> usize {
    let mut filled_count = 0;
    let latency_duration = Duration::milliseconds(latency_ms as i64);
    // Maximum fillable volume this tick
    let max_fill_volume = current_volume * fill_volume_fraction;
    let mut volume_filled_this_tick = 0.0_f64;

    let mut i = 0;
    while i < pending.len() {
        let order = &pending[i];

        // Latency gating: skip orders placed too recently
        if timestamp - order.placed_at < latency_duration {
            i += 1;
            continue;
        }

        let should_fill = if adverse_selection {
            // Adverse selection: price must cross THROUGH the limit (strict inequality)
            match &order.signal_type {
                SignalType::Buy | SignalType::ScaleIn => current_price < order.price,
                SignalType::Sell | SignalType::ScaleOut => current_price > order.price,
                _ => false,
            }
        } else {
            // Legacy: fill at or through the limit
            match &order.signal_type {
                SignalType::Buy | SignalType::ScaleIn => current_price <= order.price,
                SignalType::Sell | SignalType::ScaleOut => current_price >= order.price,
                _ => false,
            }
        };

        if should_fill {
            // Volume constraint: cap fill quantity at remaining fillable volume
            let available_volume = (max_fill_volume - volume_filled_this_tick).max(0.0);
            if available_volume <= 0.0 {
                // No more volume available this tick — stop filling
                *volume_constrained += 1;
                i += 1;
                continue;
            }
            let fill_qty = order.remaining_quantity.min(available_volume);
            if fill_qty < order.remaining_quantity {
                *volume_constrained += 1;
            }
            volume_filled_this_tick += fill_qty;

            let fill = build_fill(
                &order.signal_type,
                order.price,
                fill_qty,
                LiquidityType::Maker,
                timestamp,
                None,
                None,
            );
            // Bug #34 fix: charge commission and record the round-trip on the
            // quantity that ACTUALLY executed against the portfolio (capped by
            // available balance / inventory), not the volume-matched request.
            // Phantom fees on unexecuted fills are what break equity-curve
            // reconciliation against summed round-trip P&L.
            //
            // See `execute_taker_fill`'s matching comment: an order carrying
            // `instrument` came from a BuyOption/SellOption/etc. signal and
            // must go through `apply_derivative_fill`, not the plain spot path.
            let executed_qty = match order.instrument.as_ref() {
                Some(instr) => portfolio.apply_derivative_fill(&fill, instr),
                None => portfolio.apply_fills_executed(&[fill]),
            };
            let executed_qty = match executed_qty {
                Ok(q) => q,
                Err(e) => {
                    log_warn!(BACKTEST_LOGGER, "Maker fill error: {}", e);
                    0.0
                }
            };

            if executed_qty >= 1e-10 {
                let exec_fill_qty = executed_qty;
                let is_buy = matches!(order.signal_type, SignalType::Buy | SignalType::ScaleIn);
                let commission = compute_commission(order.price, exec_fill_qty, is_buy, order.instrument.as_ref(), maker_fee);
                *total_commission += commission;
                *num_trades += 1;
                portfolio.fees_paid += commission;
                portfolio.balance -= commission;

                trade_returns.push(0.0); // Actual P&L tracked via portfolio

                // Round-trip tracking for maker fills -- see
                // `classify_round_trip_action`'s doc comment for why this
                // can't classify off the signal type alone.
                let is_buy_direction = matches!(order.signal_type, SignalType::Buy | SignalType::ScaleIn);
                let action = classify_round_trip_action(is_buy_direction, open_positions);
                let is_exit = matches!(action, RoundTripAction::Close);

                // MAE for maker fills
                let mae_dollars = if is_exit && !open_positions.is_empty() {
                    let pos = &open_positions[open_positions.len() - 1];
                    pos.worst_unrealized.abs() * pos.entry_fill_price * pos.quantity
                } else {
                    0.0
                };
                trade_maes.push(mae_dollars);

                let should_log = match max_trade_log_size {
                    Some(max) => trade_log.len() < max,
                    None => true,
                };

                if is_exit && !open_positions.is_empty() {
                    if let Some(pos) = open_positions.pop() {
                        if should_log {
                            let raw_pnl = if pos.side == "long" {
                                (order.price - pos.entry_fill_price) * pos.quantity
                            } else {
                                (pos.entry_fill_price - order.price) * pos.quantity
                            };
                            let total_comm = pos.entry_commission + commission;
                            let net_pnl = raw_pnl - total_comm;
                            let net_pnl_pct = net_pnl / (pos.entry_fill_price * pos.quantity);
                            let duration = (timestamp - pos.entry_time).num_seconds();

                            trade_log.push(TradeRecord {
                                trade_id: pos.trade_id,
                                side: pos.side,
                                entry_signal_price: pos.entry_signal_price,
                                entry_fill_price: pos.entry_fill_price,
                                exit_signal_price: Some(order.price),
                                exit_fill_price: Some(order.price),
                                quantity: pos.quantity,
                                pnl: Some(net_pnl),
                                pnl_pct: Some(net_pnl_pct),
                                commission: total_comm,
                                slippage_cost: pos.entry_slippage,
                                entry_time: pos.entry_time,
                                exit_time: Some(timestamp),
                                duration_secs: Some(duration),
                                entry_liquidity: pos.entry_liquidity,
                                exit_liquidity: Some("maker".to_string()),
                                entry_reason: pos.entry_reason,
                                exit_reason: Some(order.reason.clone()),
                                mae: Some(pos.worst_unrealized),
                                mfe: Some(pos.best_unrealized),
                                legs: Vec::new(),
                            });
                        }
                    }
                } else {
                    let side = if matches!(action, RoundTripAction::OpenShort) { "short" } else { "long" };
                    open_positions.push(OpenPosition {
                        trade_id: *next_trade_id,
                        side: side.to_string(),
                        entry_signal_price: order.price,
                        entry_fill_price: order.price,
                        quantity: exec_fill_qty,
                        entry_commission: commission,
                        entry_slippage: 0.0,
                        entry_time: order.placed_at,
                        entry_liquidity: "maker".to_string(),
                        entry_reason: order.reason.clone(),
                        worst_unrealized: 0.0,
                        best_unrealized: 0.0,
                    });
                    *next_trade_id += 1;
                }
            }
            filled_count += 1;

            // Check if order is fully filled or partially filled
            let leftover = order.remaining_quantity - fill_qty;
            if leftover <= 1e-12 {
                // Fully filled — remove
                pending.swap_remove(i);
            } else {
                // Partial fill — update remaining quantity
                pending[i].remaining_quantity = leftover;
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    filled_count
}

/// Close all open positions at market price (taker).
fn close_all_positions(
    portfolio: &mut PortfolioState,
    current_price: f64,
    total_commission: &mut f64,
    taker_fee: f64,
    num_trades: &mut usize,
    trade_returns: &mut Vec<f64>,
    trade_maes: &mut Vec<f64>,
    trade_log: &mut Vec<TradeRecord>,
    open_positions: &mut Vec<OpenPosition>,
    max_trade_log_size: Option<usize>,
    timestamp: DateTime<Utc>,
    slippage_bps: f64,
    candle_ohlcv: Option<(f64, f64, f64, f64)>,
    synthetic_book_config: Option<&SyntheticBookConfig>,
    signal_reason: &str,
) {
    let positions: Vec<_> = portfolio
        .positions
        .iter()
        .filter(|p| p.close_time.is_none())
        .map(|p| (p.side.clone(), p.quantity, p.instrument.clone()))
        .collect();

    for (side, qty, instrument) in positions {
        let exit_signal = match side {
            portfoliomanager::PositionSide::Long => SignalType::Sell,
            portfoliomanager::PositionSide::Short => SignalType::Buy,
        };
        // Note: close_all_positions doesn't open new positions, so we pass a
        // dummy next_trade_id that won't be used for entries.
        let mut dummy_id = 0usize;
        execute_taker_fill(
            &exit_signal,
            current_price,
            qty,
            portfolio,
            total_commission,
            taker_fee,
            num_trades,
            trade_returns,
            trade_maes,
            trade_log,
            open_positions,
            &mut dummy_id,
            max_trade_log_size,
            timestamp,
            slippage_bps,
            candle_ohlcv,
            synthetic_book_config,
            signal_reason,
            instrument.as_ref(),
        );
    }
}

/// Build an `orderbook::Fill` from signal parameters.
///
/// Fee amount/rate are set to zero here — fees are tracked and deducted
/// separately by the caller to ensure consistent accounting.
fn build_fill(
    signal_type: &SignalType,
    price: f64,
    quantity: f64,
    liquidity_type: LiquidityType,
    timestamp: DateTime<Utc>,
    base_price: Option<f64>,
    slippage_bps_val: Option<f64>,
) -> Fill {
    Fill {
        order_id: Arc::from(uuid::Uuid::new_v4().to_string().as_str()),
        side: match signal_type {
            SignalType::Buy | SignalType::ScaleIn => BookSide::Bid,
            _ => BookSide::Ask,
        },
        price,
        base_price,
        quantity,
        liquidity_type,
        fee_rate: 0.0,
        fee_amount: 0.0,
        slippage_bps: slippage_bps_val,
        timestamp,
    }
}

/// Fraction of positive-return observations in `primary_returns` (closed
/// round-trip returns), falling back to `fallback_returns` (equity-curve
/// bar-to-bar returns) when `primary_returns` is empty -- a strategy that
/// never explicitly exits a position closes zero round-trips even though the
/// equity curve may show a real, large mark-to-market gain.
fn win_rate_with_fallback(primary_returns: &[f64], fallback_returns: &[f64]) -> Option<f64> {
    let source = if !primary_returns.is_empty() { primary_returns } else { fallback_returns };
    if source.is_empty() {
        None
    } else {
        Some(source.iter().filter(|r| **r > 0.0).count() as f64 / source.len() as f64)
    }
}

/// Gross profit / gross loss (absolute value) from closed round-trip dollar
/// PnLs, falling back to per-bar equity-curve dollar deltas when there are no
/// closed round-trips at all (same rationale as `win_rate_with_fallback`).
fn gross_profit_loss_with_fallback(trade_log: &[TradeRecord], equity_curve: &[f64]) -> (f64, f64) {
    if trade_log.iter().any(|t| t.exit_time.is_some()) {
        let gp: f64 = trade_log.iter()
            .filter(|t| t.exit_time.is_some())
            .filter_map(|t| t.pnl)
            .filter(|p| *p > 0.0)
            .sum();
        let gl: f64 = trade_log.iter()
            .filter(|t| t.exit_time.is_some())
            .filter_map(|t| t.pnl)
            .filter(|p| *p < 0.0)
            .sum::<f64>()
            .abs();
        (gp, gl)
    } else {
        let deltas: Vec<f64> = equity_curve.windows(2).map(|w| w[1] - w[0]).collect();
        let gp: f64 = deltas.iter().filter(|d| **d > 0.0).sum();
        let gl: f64 = deltas.iter().filter(|d| **d < 0.0).sum::<f64>().abs();
        (gp, gl)
    }
}

/// Build a standard `BacktestResult` from simulation state.
fn build_backtest_result(
    total_pnl: f64,
    num_trades: usize,
    equity_curve: &[f64],
    equity_curve_timestamps: &[i64],
    _trade_returns: &[f64],
    _trade_maes: &[f64],
    trade_log: Vec<TradeRecord>,
    initial_capital: f64,
    total_commission: f64,
    first_price: Option<f64>,
    last_price: f64,
    portfolio: &PortfolioState,
    price_series: Vec<f64>,
    signal_counts: Option<crate::types::SignalCounts>,
    compute_signals_error: Option<String>,
    strategy_display_name: Option<String>,
    computed_features: Option<HashMap<String, Vec<f64>>>,
    captured_prices_for_features: Vec<f64>,
    captured_timestamps_for_features: Vec<i64>,
    volume_constrained_fills: usize,
    option_instrument: Option<&DerivativeMetadata>,
    option_spread: Option<&[OptionSpreadLeg]>,
) -> BacktestResult {
    // Use actual round-trip percentage returns from trade_log instead of per-fill
    // pseudo-returns in trade_returns (which are just cash flow proportions, not real P&L).
    //
    // CRITICAL: filter to CLOSED round-trips only (`exit_time.is_some()`). At the end
    // of a backtest, still-open positions are flushed to `trade_log` as synthetic
    // mark-to-market records carrying `pnl: Some(unrealized_pnl)` but `exit_time: None`
    // (see open-position flush above). Including those unrealized marks here let them
    // leak into `trade_returns`/`trade_pct_returns`, which Monte Carlo sums additively
    // (equity += ret). In walk-forward, every window boundary leaves a position open,
    // so the combined OOS result accumulated hundreds of phantom unrealized marks —
    // inflating the MC trade count and P&L distribution far above the realized result
    // (Bug #16: MC band $32k-$46k vs realized ~break-even). The closed-trade filter
    // here matches the trade-count filter in worker.rs (Bug #17) so the realized P&L,
    // trade count, win rate, and Monte Carlo base all reconcile.
    let actual_returns: Vec<f64> = trade_log.iter()
        .filter(|t| t.exit_time.is_some())
        .filter_map(|t| t.pnl_pct)
        .collect();

    // Dollar P&Ls from closed round-trips for Monte Carlo (which does equity += ret)
    let dollar_returns: Vec<f64> = trade_log.iter()
        .filter(|t| t.exit_time.is_some())
        .filter_map(|t| t.pnl)
        .collect();

    // MAEs aligned to closed round-trips: convert fractional MAE to dollar MAE
    let roundtrip_maes: Vec<f64> = trade_log.iter()
        .filter(|t| t.exit_time.is_some() && t.pnl.is_some())
        .map(|t| {
            t.mae.unwrap_or(0.0).abs() * t.entry_fill_price * t.quantity
        })
        .collect();

    // Fall back to equity-curve percentage returns if no closed trades exist
    let returns = if !actual_returns.is_empty() {
        actual_returns.clone()
    } else if equity_curve.len() > 1 {
        equity_curve.windows(2)
            .filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
            .collect()
    } else {
        Vec::new()
    };

    // Determine the annualization factor for risk metrics.
    //
    // `returns` is either (a) per-round-trip percentage returns or (b) per-bar
    // equity-curve percentage returns. Naively multiplying std/mean by sqrt(365)
    // assumes ONE observation per day, which is almost never true:
    //   - For per-trade returns at 64 trades over ~6 months, the true rate is
    //     ~0.35 trades/day, so sqrt(365) over-annualizes ~32x.
    //   - For per-bar returns at minute cadence, the true rate is ~1440/day,
    //     so sqrt(365) under-annualizes ~2x.
    // Compute the observation rate from the actual backtest time span when
    // available and fall back to 365 only when the span is unknown.
    let backtest_span_secs: Option<f64> = {
        let first_ts = trade_log.first().map(|t| t.entry_time);
        let last_ts = trade_log.last().and_then(|t| t.exit_time)
            .or_else(|| trade_log.last().map(|t| t.entry_time));
        match (first_ts, last_ts) {
            (Some(a), Some(b)) if b > a => Some((b - a).num_seconds() as f64),
            _ => None,
        }
    };
    let observations_per_year: f64 = significance::observations_per_year_from_span(returns.len(), backtest_span_secs);
    let annualization_factor = observations_per_year.sqrt();

    let sharpe = if returns.len() >= 2 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        let std = variance.sqrt();
        if std > 0.0 {
            // Clamp at ±20: prevents OOS Sharpe blowup when std dev ≈ 0 (all losses equal magnitude)
            Some(((mean / std) * annualization_factor).clamp(-20.0, 20.0))
        } else {
            Some(0.0)
        }
    } else {
        None
    };

    let max_drawdown = compute_max_drawdown(equity_curve);

    // A strategy that only ever scales into a position and never explicitly
    // exits (e.g. buy/hold-only signal logic) closes zero round-trips, so
    // `actual_returns` is empty even though the equity curve shows a real,
    // possibly large mark-to-market gain -- reporting a flat 0% win rate for
    // a genuinely profitable run is a direct contradiction with the equity
    // curve/Sharpe (which already fall back to per-bar equity returns, see
    // `returns` above). Fall back to the SAME per-bar equity returns here so
    // win rate reflects "fraction of profitable observation periods" instead
    // of silently defaulting to zero. This does not touch `trade_returns`/
    // `trade_pct_returns` (Monte Carlo's input), which must stay closed-
    // round-trips-only per Bug #16 above.
    let win_rate = win_rate_with_fallback(&actual_returns, &returns);

    // Sortino ratio: mean / downside_std * sqrt(365)
    let sortino = if !returns.is_empty() {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_var = returns
            .iter()
            .filter(|r| **r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let downside_std = downside_var.sqrt();
        if downside_std > 0.0 {
            Some(((mean / downside_std) * annualization_factor).clamp(-20.0, 20.0))
        } else {
            Some(0.0)
        }
    } else {
        None
    };

    // Volatility: annualized standard deviation of returns
    let volatility = if !returns.is_empty() {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        Some(var.sqrt() * annualization_factor)
    } else {
        None
    };

    // Average and median trade return (from actual round-trip returns)
    let avg_trade_return = if !actual_returns.is_empty() {
        Some(actual_returns.iter().sum::<f64>() / actual_returns.len() as f64)
    } else {
        None
    };

    let median_trade_return = if !actual_returns.is_empty() {
        let mut sorted = actual_returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    } else {
        None
    };

    // Average trade duration (seconds) from closed trades
    let avg_trade_duration = {
        let durations: Vec<f64> = trade_log
            .iter()
            .filter_map(|t| t.duration_secs.map(|d| d as f64))
            .collect();
        if !durations.is_empty() {
            Some(durations.iter().sum::<f64>() / durations.len() as f64)
        } else {
            None
        }
    };

    // Per-feature diagnostics (IC + quantile analysis vs. forward returns) —
    // only when the strategy defined compute_features() and
    // ENABLE_COMPUTE_FEATURES is on. Horizon comes from the strategy's own
    // average holding period (never a fixed default): avg_trade_duration
    // converted from seconds into a bar count via the data's own average bar
    // spacing, falling back to a 1-bar horizon when no trades closed yet or
    // the series is too short to derive bar spacing.
    let feature_diagnostics = crate::feature_diagnostics::compute_feature_diagnostics(
        &computed_features,
        &captured_prices_for_features,
        &captured_timestamps_for_features,
        avg_trade_duration,
    );

    // Gross profit / loss / profit factor from CLOSED round-trip PnLs only.
    // Excludes the synthetic open-position mark-to-market records (exit_time: None)
    // so profit_factor reconciles with realized P&L (Bug #16). Falls back to
    // per-bar equity-curve dollar deltas when there are no closed round-trips
    // (see the win_rate fallback above for why) -- local to this metric only,
    // does not touch `trade_returns` (Monte Carlo's input, Bug #16).
    let (gross_profit, gross_loss_abs) = gross_profit_loss_with_fallback(&trade_log, equity_curve);
    let profit_factor = if gross_loss_abs > 0.0 {
        Some(gross_profit / gross_loss_abs)
    } else if gross_profit > 0.0 {
        Some(f64::INFINITY)
    } else {
        Some(0.0)
    };

    // Calmar ratio: annualized_return / max_drawdown
    let calmar = if max_drawdown > 0.0 && !equity_curve.is_empty() {
        let final_eq = *equity_curve.last().unwrap();
        let total_return = final_eq / initial_capital;
        let years = if let (Some(first), Some(last)) = (
            trade_log.first().map(|t| t.entry_time),
            trade_log.last().and_then(|t| t.exit_time).or_else(|| trade_log.last().map(|t| t.entry_time)),
        ) {
            (last - first).num_seconds() as f64 / (365.25 * 86400.0)
        } else {
            1.0
        };
        if years > 0.0 && total_return > 0.0 {
            let annualized = total_return.powf(1.0 / years) - 1.0;
            Some(annualized / max_drawdown)
        } else {
            None
        }
    } else {
        None
    };

    // Count actual closed round-trips from trade_log (not fill count)
    let closed_round_trips = trade_log.iter().filter(|t| t.exit_time.is_some()).count();

    // Realized P&L NET of fees. `portfolio.realized_pnl` is GROSS (commissions are
    // deducted from `balance`/equity but NOT from the realized accumulator), which
    // mislabels the figure and fails to reconcile with `total_pnl`. Each closed
    // round-trip's `pnl` already has entry+exit commission subtracted, so summing
    // the net trade ledger makes realized_pnl reconcile with the trade breakdown
    // and, when the book is flat at end, with total_pnl (final_equity - initial).
    // (The full result path logs every trade; GA evaluations cap the log but do not
    // persist realized_pnl, and use net_profit/total_pnl for fitness.)
    let realized_pnl_net: f64 = trade_log
        .iter()
        .filter(|t| t.exit_time.is_some())
        .filter_map(|t| t.pnl)
        .sum();

    BacktestResult {
        total_pnl,
        num_trades,
        closed_trades: closed_round_trips,
        open_positions: portfolio
            .positions
            .iter()
            .filter(|p| p.close_time.is_none())
            .count(),
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        calmar_ratio: calmar,
        volatility,
        avg_trade_return,
        median_trade_return,
        avg_trade_duration,
        profit_factor,
        gross_profit: Some(gross_profit),
        gross_loss: Some(gross_loss_abs),
        net_profit: Some(total_pnl),
        max_drawdown,
        win_rate,
        equity_curve: equity_curve.to_vec(),
        equity_curve_timestamps: equity_curve_timestamps.to_vec(),
        trade_returns: dollar_returns,
        trade_pct_returns: actual_returns,
        trade_maes: roundtrip_maes,
        initial_capital,
        realized_pnl: Some(realized_pnl_net),
        unrealized_pnl: Some(portfolio.unrealized_pnl),
        first_trade_price: first_price,
        last_trade_price: Some(last_price),
        first_trade_timestamp: trade_log.first().map(|t| t.entry_time),
        last_trade_timestamp: trade_log.last().and_then(|t| t.exit_time).or_else(|| trade_log.last().map(|t| t.entry_time)),
        sharpe_t_stat: sharpe.map(|s| {
            let n = returns.len().max(2);
            let (t, _) = significance::sharpe_significance(s, n, observations_per_year);
            t
        }),
        significance_pvalue: sharpe.map(|s| {
            let n = returns.len().max(2);
            let (_, p) = significance::sharpe_significance(s, n, observations_per_year);
            p
        }),
        deflated_sharpe: sharpe.map(|s| {
            let n = returns.len().max(2);
            significance::deflated_sharpe_ratio(s, 1, n, 1.0, observations_per_year)
        }),
        ulcer_index: risk::ulcer_index(equity_curve),
        pain_index: risk::pain_index(equity_curve),
        cdar_95: risk::conditional_drawdown_at_risk(equity_curve, 0.95),
        trade_log,
        price_series,
        transaction_costs: Some(crate::types::TransactionCostAnalysis {
            total_commission,
            total_slippage: 0.0,
            total_market_impact: 0.0,
            total_transaction_costs: total_commission,
            costs_as_percentage_of_pnl: if total_pnl.abs() > 0.0 {
                total_commission / total_pnl.abs()
            } else {
                0.0
            },
            avg_cost_per_trade: if num_trades > 0 {
                total_commission / num_trades as f64
            } else {
                0.0
            },
            commission_by_exchange: std::collections::HashMap::new(),
            total_gas_costs_usd: 0.0,
            avg_dex_slippage_bps: 0.0,
        }),
        signal_counts,
        compute_signals_error,
        strategy_display_name,
        feature_diagnostics,
        volume_constrained_pct: if num_trades > 0 {
            Some(volume_constrained_fills as f64 / num_trades as f64)
        } else {
            None
        },
        option_summary: build_option_summary(option_instrument, option_spread, portfolio),
        ..Default::default()
    }
}

/// Build the `BacktestResult.option_summary` field: contract identity for
/// every leg this run traded (single instrument or spread), plus the
/// portfolio's final aggregated Greeks snapshot. `None` when this run
/// wasn't an options run at all (neither `option_instrument` nor
/// `option_spread` set) -- unchanged, existing behavior.
fn build_option_summary(
    option_instrument: Option<&DerivativeMetadata>,
    option_spread: Option<&[OptionSpreadLeg]>,
    portfolio: &PortfolioState,
) -> Option<crate::types::OptionResultSummary> {
    let contracts: Vec<crate::types::OptionContractInfo> = if let Some(legs) = option_spread {
        legs.iter().map(|leg| option_contract_info(&leg.instrument, leg.ratio)).collect()
    } else {
        vec![option_contract_info(option_instrument?, 1)]
    };
    let greeks = portfolio.get_portfolio_greeks();
    let final_greeks = if greeks == derivatives::Greeks::default() { None } else { Some(greeks) };
    Some(crate::types::OptionResultSummary { contracts, final_greeks })
}

fn option_contract_info(instrument: &DerivativeMetadata, ratio: i32) -> crate::types::OptionContractInfo {
    let contract_type = match instrument.instrument_kind {
        derivatives::InstrumentKind::Call { .. } => "call",
        derivatives::InstrumentKind::Put { .. } => "put",
        _ => "other",
    }.to_string();
    crate::types::OptionContractInfo {
        symbol: instrument.symbol.clone(),
        underlying: instrument.underlying.clone(),
        strike: instrument.instrument_kind.strike(),
        expiry: instrument.instrument_kind.expiry(),
        contract_type,
        contract_multiplier: instrument.contract_multiplier,
        ratio,
    }
}

/// Compute maximum drawdown from equity curve (as a fraction, e.g. 0.15 = 15%).
fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = (peak - val) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_trade(pnl: Option<f64>, exit_time: Option<DateTime<Utc>>) -> TradeRecord {
        TradeRecord {
            trade_id: 0,
            side: "long".to_string(),
            entry_signal_price: 100.0,
            entry_fill_price: 100.0,
            exit_signal_price: exit_time.map(|_| 105.0),
            exit_fill_price: exit_time.map(|_| 105.0),
            quantity: 1.0,
            pnl,
            pnl_pct: pnl.map(|p| p / 100.0),
            commission: 0.0,
            slippage_cost: 0.0,
            entry_time: chrono::Utc::now(),
            exit_time,
            duration_secs: None,
            entry_liquidity: "taker".to_string(),
            exit_liquidity: exit_time.map(|_| "taker".to_string()),
            entry_reason: "signal".to_string(),
            exit_reason: exit_time.map(|_| "signal".to_string()),
            mae: None,
            mfe: None,
            legs: Vec::new(),
        }
    }

    #[test]
    fn find_asof_value_returns_last_value_at_or_before_tick() {
        let series = vec![(1_000_i64, 10.0), (2_000, 20.0), (3_000, 30.0)];
        let ts = |ms: i64| chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).unwrap();
        assert_eq!(find_asof_value(&series, &ts(2_500)), Some(20.0));
        assert_eq!(find_asof_value(&series, &ts(2_000)), Some(20.0)); // exact match is at-or-before
    }

    #[test]
    fn find_asof_value_never_returns_a_future_value() {
        let series = vec![(1_000_i64, 10.0), (2_000, 20.0)];
        let ts = |ms: i64| chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).unwrap();
        // Every point is still in the future relative to this tick -- must be
        // None, not the earliest point (that would be lookahead bias).
        assert_eq!(find_asof_value(&series, &ts(500)), None);
    }

    #[test]
    fn find_asof_value_empty_series_is_none() {
        let ts = chrono::Utc::now();
        assert_eq!(find_asof_value(&[], &ts), None);
    }

    #[test]
    fn find_asof_value_after_last_point_returns_last_value() {
        let series = vec![(1_000_i64, 10.0), (2_000, 20.0)];
        let ts = |ms: i64| chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).unwrap();
        assert_eq!(find_asof_value(&series, &ts(999_999)), Some(20.0));
    }

    #[test]
    fn win_rate_with_fallback_uses_primary_when_present() {
        let primary = vec![0.01, -0.01, 0.02];
        let fallback = vec![-0.5, -0.5];
        let wr = win_rate_with_fallback(&primary, &fallback).unwrap();
        assert!((wr - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn win_rate_with_fallback_uses_equity_curve_when_no_closed_trades() {
        // A strategy that never closes a position: no primary (closed
        // round-trip) returns, but the equity curve shows real profit.
        let primary: Vec<f64> = vec![];
        let fallback = vec![0.01, 0.02, -0.005, 0.01];
        let wr = win_rate_with_fallback(&primary, &fallback).unwrap();
        assert!((wr - 0.75).abs() < 1e-9);
    }

    #[test]
    fn win_rate_with_fallback_is_none_when_both_empty() {
        assert!(win_rate_with_fallback(&[], &[]).is_none());
    }

    #[test]
    fn gross_profit_loss_uses_closed_trades_when_present() {
        let trades = vec![
            mk_trade(Some(100.0), Some(chrono::Utc::now())),
            mk_trade(Some(-40.0), Some(chrono::Utc::now())),
            // Still-open position flushed as an unrealized mark -- must be
            // excluded even though it has a `pnl` value (Bug #16).
            mk_trade(Some(9999.0), None),
        ];
        let equity_curve = vec![1000.0, 1100.0];
        let (gp, gl) = gross_profit_loss_with_fallback(&trades, &equity_curve);
        assert!((gp - 100.0).abs() < 1e-9);
        assert!((gl - 40.0).abs() < 1e-9);
    }

    #[test]
    fn gross_profit_loss_falls_back_to_equity_curve_when_no_closed_trades() {
        // Every trade is still open (exit_time: None) -- e.g. a strategy that
        // only ever scales into a position and never explicitly exits.
        let trades = vec![mk_trade(Some(500.0), None)];
        let equity_curve = vec![1000.0, 1050.0, 1030.0, 1080.0];
        let (gp, gl) = gross_profit_loss_with_fallback(&trades, &equity_curve);
        // Deltas: +50, -20, +50 -> gross_profit=100, gross_loss=20
        assert!((gp - 100.0).abs() < 1e-9);
        assert!((gl - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_nearest_snapshot_empty() {
        let snapshots: Vec<BookSnapshot> = vec![];
        let ts = chrono::Utc::now();
        assert!(find_nearest_snapshot(&snapshots, &ts).is_none());
    }

    #[test]
    fn test_find_nearest_snapshot_single() {
        let snap = BookSnapshot {
            timestamp_us: 1_000_000_000, // 1000s
            bids: vec![OrderBookLevel { price: 100.0, amount: 1.0 }],
            asks: vec![OrderBookLevel { price: 101.0, amount: 1.0 }],
        };
        let ts = chrono::DateTime::from_timestamp_millis(2_000_000).unwrap(); // 2000s
        let result = find_nearest_snapshot(&[snap], &ts);
        assert!(result.is_some());
        let ob = result.unwrap();
        assert!((ob.bids[0][0] - 100.0).abs() < f64::EPSILON);
        assert!((ob.asks[0][0] - 101.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_nearest_snapshot_before_all() {
        let snap = BookSnapshot {
            timestamp_us: 5_000_000_000, // 5000s in us
            bids: vec![OrderBookLevel { price: 50.0, amount: 2.0 }],
            asks: vec![OrderBookLevel { price: 51.0, amount: 2.0 }],
        };
        // Timestamp well before the snapshot
        let ts = chrono::DateTime::from_timestamp_millis(1_000).unwrap();
        let result = find_nearest_snapshot(&[snap], &ts);
        assert!(result.is_some()); // Should return earliest snapshot
    }

    #[test]
    fn test_tardis_to_orderbook_basic() {
        let snap = BookSnapshot {
            timestamp_us: 1_000_000,
            bids: vec![
                OrderBookLevel { price: 100.0, amount: 5.0 },
                OrderBookLevel { price: 99.0, amount: 3.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, amount: 4.0 },
                OrderBookLevel { price: 102.0, amount: 2.0 },
            ],
        };
        let ob = tardis_to_orderbook(&snap);
        assert_eq!(ob.bids.len(), 2);
        assert_eq!(ob.asks.len(), 2);
        assert!((ob.spread - 1.0).abs() < f64::EPSILON); // 101 - 100
        assert!((ob.mid_price - 100.5).abs() < f64::EPSILON);
        assert!((ob.bid_depth - 8.0).abs() < f64::EPSILON); // 5+3
        assert!((ob.ask_depth - 6.0).abs() < f64::EPSILON); // 4+2
    }

    #[test]
    fn test_tardis_to_orderbook_filters_zero_prices() {
        let snap = BookSnapshot {
            timestamp_us: 1_000_000,
            bids: vec![
                OrderBookLevel { price: 100.0, amount: 5.0 },
                OrderBookLevel { price: 0.0, amount: 3.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 0.0, amount: 4.0 },
                OrderBookLevel { price: 102.0, amount: 2.0 },
            ],
        };
        let ob = tardis_to_orderbook(&snap);
        assert_eq!(ob.bids.len(), 1); // zero price filtered
        assert_eq!(ob.asks.len(), 1); // zero price filtered
    }

    #[test]
    fn test_tardis_to_orderbook_empty() {
        let snap = BookSnapshot {
            timestamp_us: 1_000_000,
            bids: vec![],
            asks: vec![],
        };
        let ob = tardis_to_orderbook(&snap);
        assert!(ob.bids.is_empty());
        assert!(ob.asks.is_empty());
        assert_eq!(ob.mid_price, 0.0);
        assert_eq!(ob.spread, 0.0);
        assert_eq!(ob.bid_depth, 0.0);
        assert_eq!(ob.ask_depth, 0.0);
    }

    #[test]
    fn test_tardis_to_orderbook_imbalance() {
        // All depth on bid side
        let snap = BookSnapshot {
            timestamp_us: 1_000_000,
            bids: vec![OrderBookLevel { price: 100.0, amount: 10.0 }],
            asks: vec![OrderBookLevel { price: 101.0, amount: 0.0001 }],
        };
        let ob = tardis_to_orderbook(&snap);
        assert!(ob.imbalance > 0.0); // bid-heavy
    }

    #[test]
    fn test_compute_max_drawdown_empty() {
        let dd = compute_max_drawdown(&[]);
        assert_eq!(dd, 0.0);
    }

    #[test]
    fn test_compute_max_drawdown_monotonic_up() {
        let equity = vec![100.0, 110.0, 120.0, 130.0];
        let dd = compute_max_drawdown(&equity);
        assert_eq!(dd, 0.0);
    }

    #[test]
    fn test_compute_max_drawdown_with_drawdown() {
        let equity = vec![100.0, 110.0, 88.0, 95.0, 105.0];
        let dd = compute_max_drawdown(&equity);
        // Peak = 110, trough = 88, drawdown = 22/110 = 0.2
        assert!((dd - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_max_drawdown_full_loss() {
        let equity = vec![100.0, 50.0, 25.0, 10.0];
        let dd = compute_max_drawdown(&equity);
        // Peak = 100, min = 10, dd = 90/100 = 0.9
        assert!((dd - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_compute_max_drawdown_single_value() {
        let equity = vec![100.0];
        let dd = compute_max_drawdown(&equity);
        assert_eq!(dd, 0.0);
    }

    // -- effective_leverage --------------------------------------------------

    #[test]
    fn effective_leverage_uses_configured_value_when_strategy_declares_no_cap() {
        // A strategy that never overrides get_risk_limits() declares
        // f64::INFINITY (no additional cap) -- the job's configured/
        // exchange-capped leverage must pass through unchanged.
        assert_eq!(effective_leverage(10.0, f64::INFINITY), 10.0);
        assert_eq!(effective_leverage(1.0, f64::INFINITY), 1.0);
    }

    #[test]
    fn effective_leverage_strategy_cap_reduces_below_configured() {
        assert_eq!(effective_leverage(10.0, 3.0), 3.0);
    }

    #[test]
    fn effective_leverage_strategy_cap_above_configured_has_no_effect() {
        // A strategy cap can only ever REDUCE leverage, never raise it above
        // what the job itself requested.
        assert_eq!(effective_leverage(3.0, 10.0), 3.0);
    }

    #[test]
    fn effective_leverage_never_drops_below_one() {
        assert_eq!(effective_leverage(0.5, f64::INFINITY), 1.0);
        assert_eq!(effective_leverage(10.0, 0.5), 1.0);
    }

    #[test]
    fn test_python_sim_config_construction() {
        let config = PythonSimConfig {
            python_source: "class MyStrat: pass".into(),
            backtest_config: config::BacktestConfig::default(),
            fee_config: None,
            supplementary_data: HashMap::new(),
            parameters: HashMap::new(),
            progress_callback: None,
            risk_manager: None,
            max_trade_log_size: Some(500),
            orderbook_snapshots: None,
            historical_iv_surfaces: None,
            multi_venue_data: None,
            option_instrument: None,
            option_spread: None,
            underlying_series: None,
        };
        assert_eq!(config.python_source, "class MyStrat: pass");
        assert_eq!(config.max_trade_log_size, Some(500));
        assert!(config.fee_config.is_none());
    }

    #[test]
    fn test_python_sim_config_unlimited_trade_log() {
        let config = PythonSimConfig {
            python_source: String::new(),
            backtest_config: config::BacktestConfig::default(),
            fee_config: None,
            supplementary_data: HashMap::new(),
            parameters: HashMap::new(),
            progress_callback: None,
            risk_manager: None,
            max_trade_log_size: None,
            orderbook_snapshots: None,
            historical_iv_surfaces: None,
            multi_venue_data: None,
            option_instrument: None,
            option_spread: None,
            underlying_series: None,
        };
        assert!(config.max_trade_log_size.is_none());
    }

    // ---- classify_round_trip_action (real bug, 2026-09-02: a bare Sell
    // arriving while flat used to vanish from the round-trip ledger, and a
    // Buy meant to cover an open short got recorded as a fresh long) ----

    fn mk_open_position(side: &str) -> OpenPosition {
        OpenPosition {
            trade_id: 1,
            side: side.to_string(),
            entry_signal_price: 100.0,
            entry_fill_price: 100.0,
            quantity: 1.0,
            entry_commission: 0.0,
            entry_slippage: 0.0,
            entry_time: chrono::Utc::now(),
            entry_liquidity: "taker".to_string(),
            entry_reason: "signal".to_string(),
            worst_unrealized: 0.0,
            best_unrealized: 0.0,
        }
    }

    #[test]
    fn classify_round_trip_action_buy_from_flat_opens_long() {
        let open_positions: Vec<OpenPosition> = Vec::new();
        assert!(matches!(classify_round_trip_action(true, &open_positions), RoundTripAction::OpenLong));
    }

    #[test]
    fn classify_round_trip_action_sell_from_flat_opens_short() {
        // The core bug: a bare Sell with nothing open used to match neither
        // "close" (nothing to close) nor "entry" (not a Buy), so it was
        // silently dropped from the round-trip ledger even though the
        // portfolio itself opens a real short for it.
        let open_positions: Vec<OpenPosition> = Vec::new();
        assert!(matches!(classify_round_trip_action(false, &open_positions), RoundTripAction::OpenShort));
    }

    #[test]
    fn classify_round_trip_action_sell_closes_open_long() {
        let open_positions = vec![mk_open_position("long")];
        assert!(matches!(classify_round_trip_action(false, &open_positions), RoundTripAction::Close));
    }

    #[test]
    fn classify_round_trip_action_buy_covers_open_short() {
        // The other half of the bug: a Buy meant to cover an open short
        // used to be misread as opening a brand-new "long".
        let open_positions = vec![mk_open_position("short")];
        assert!(matches!(classify_round_trip_action(true, &open_positions), RoundTripAction::Close));
    }

    #[test]
    fn classify_round_trip_action_buy_adds_to_open_long() {
        let open_positions = vec![mk_open_position("long")];
        assert!(matches!(classify_round_trip_action(true, &open_positions), RoundTripAction::OpenLong));
    }

    #[test]
    fn classify_round_trip_action_sell_adds_to_open_short() {
        let open_positions = vec![mk_open_position("short")];
        assert!(matches!(classify_round_trip_action(false, &open_positions), RoundTripAction::OpenShort));
    }

    // ---- Options signal routing: to_directional_signal / extract_instrument ----

    fn test_call_instrument() -> DerivativeMetadata {
        DerivativeMetadata::new(
            "O:SPY-TEST-C",
            "SPY",
            derivatives::InstrumentKind::Call {
                strike: 450.0,
                // Fixed, not `Utc::now() + ...`: several tests build this
                // twice (once via a shared fixture helper, once directly for
                // an equality assertion) and compare the two `DerivativeMetadata`
                // values for exact equality -- a `Utc::now()`-based expiry
                // makes the two calls' timestamps differ by whatever wall-clock
                // time elapsed between them (even a few microseconds), which
                // fails `PartialEq` on the embedded `DateTime<Utc>`. Confirmed
                // as a real, live flake once these `python`-feature tests were
                // actually run (not just compile-checked) for the first time,
                // 2026-09-02.
                expiry: chrono::DateTime::parse_from_rfc3339("2026-10-02T00:00:00Z").unwrap().to_utc(),
            },
            100.0,
            "USD",
            "test",
        )
    }

    // -- BacktestResult.option_summary (build_option_summary / option_contract_info) --

    #[test]
    fn build_option_summary_is_none_for_a_non_option_run() {
        let portfolio = PortfolioState::with_balance(100_000.0);
        assert!(build_option_summary(None, None, &portfolio).is_none());
    }

    #[test]
    fn build_option_summary_reports_the_single_leg_contract_identity() {
        let instrument = test_call_instrument();
        let portfolio = PortfolioState::with_balance(100_000.0);
        let summary = build_option_summary(Some(&instrument), None, &portfolio)
            .expect("a single-leg option run should produce a summary");
        assert_eq!(summary.contracts.len(), 1);
        let contract = &summary.contracts[0];
        assert_eq!(contract.symbol, "O:SPY-TEST-C");
        assert_eq!(contract.underlying, "SPY");
        assert_eq!(contract.strike, Some(450.0));
        assert_eq!(contract.contract_type, "call");
        assert_eq!(contract.contract_multiplier, 100.0);
        assert_eq!(contract.ratio, 1);
        // No Greeks were ever computed for this portfolio (no fill, no
        // update_position_greeks call) -- must stay None, not a spurious
        // all-zero Greeks value.
        assert!(summary.final_greeks.is_none());
    }

    #[test]
    fn build_option_summary_reports_both_legs_of_a_spread_with_their_own_ratios() {
        let call = test_call_instrument();
        let put = test_put_instrument();
        let legs = vec![
            OptionSpreadLeg { instrument: call, ratio: 1, prices: vec![5.0] },
            OptionSpreadLeg { instrument: put, ratio: -1, prices: vec![2.0] },
        ];
        let portfolio = PortfolioState::with_balance(100_000.0);
        let summary = build_option_summary(None, Some(&legs), &portfolio)
            .expect("a spread run should produce a summary");
        assert_eq!(summary.contracts.len(), 2);
        assert_eq!(summary.contracts[0].contract_type, "call");
        assert_eq!(summary.contracts[0].ratio, 1);
        assert_eq!(summary.contracts[1].contract_type, "put");
        assert_eq!(summary.contracts[1].ratio, -1);
    }

    #[test]
    fn build_option_summary_prefers_a_nonzero_final_greeks_snapshot_when_available() {
        let instrument = test_call_instrument();
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let fill = orderbook::Fill {
            order_id: std::sync::Arc::from("test"),
            side: orderbook::BookSide::Bid,
            price: 5.0,
            base_price: None,
            quantity: 1.0,
            liquidity_type: orderbook::LiquidityType::Taker,
            fee_rate: 0.0,
            fee_amount: 0.0,
            slippage_bps: None,
            timestamp: chrono::Utc::now(),
        };
        portfolio.apply_derivative_fill(&fill, &instrument).unwrap();
        portfolio.update_position_greeks(&instrument.symbol, 460.0, 0.02, 0.30, chrono::Utc::now());

        let summary = build_option_summary(Some(&instrument), None, &portfolio).unwrap();
        let greeks = summary.final_greeks.expect("a marked option position should produce real Greeks");
        assert_ne!(greeks, derivatives::Greeks::default());
    }

    #[test]
    fn i8_to_signals_wraps_buy_and_sell_as_options_when_instrument_is_set() {
        let instrument = test_call_instrument();
        let buy_signals = i8_to_signals(1, 5.0, 0, Some(&instrument), None);
        assert_eq!(buy_signals.len(), 1);
        match &buy_signals[0].signal_type {
            SignalType::BuyOption { instrument: instr, premium } => {
                assert_eq!(instr, &instrument);
                assert!(premium.is_none(), "premium is filled in from the fill price downstream, not set here");
            }
            other => panic!("expected BuyOption, got {other:?}"),
        }

        let sell_signals = i8_to_signals(-1, 5.0, 0, Some(&instrument), None);
        assert_eq!(sell_signals.len(), 1);
        match &sell_signals[0].signal_type {
            SignalType::SellOption { instrument: instr, .. } => assert_eq!(instr, &instrument),
            other => panic!("expected SellOption, got {other:?}"),
        }
    }

    #[test]
    fn i8_to_signals_stays_plain_buy_sell_without_an_instrument() {
        let buy_signals = i8_to_signals(1, 100.0, 0, None, None);
        assert_eq!(buy_signals.len(), 1);
        assert_eq!(buy_signals[0].signal_type, SignalType::Buy);

        let sell_signals = i8_to_signals(-1, 100.0, 0, None, None);
        assert_eq!(sell_signals.len(), 1);
        assert_eq!(sell_signals[0].signal_type, SignalType::Sell);
    }

    #[test]
    fn i8_to_signals_close_is_unaffected_by_option_instrument() {
        // CLOSE needs no option-aware wrapping: close_all_positions already
        // closes each open position using its OWN recorded instrument, not
        // a signal-level one -- see PythonSimConfig.option_instrument's doc
        // comment.
        let instrument = test_call_instrument();
        let with_instrument = i8_to_signals(2, 5.0, 0, Some(&instrument), None);
        let without_instrument = i8_to_signals(2, 5.0, 0, None, None);
        assert_eq!(with_instrument.len(), 1);
        assert_eq!(without_instrument.len(), 1);
        assert_eq!(with_instrument[0].signal_type, SignalType::Close);
        assert_eq!(without_instrument[0].signal_type, SignalType::Close);
    }

    #[test]
    fn i8_to_signals_hold_and_unknown_codes_produce_nothing_regardless_of_instrument() {
        let instrument = test_call_instrument();
        assert!(i8_to_signals(0, 5.0, 0, Some(&instrument), None).is_empty());
        assert!(i8_to_signals(99, 5.0, 0, Some(&instrument), None).is_empty());
    }

    // ---- i8_to_signals: OptionSpread construction ----

    fn test_put_instrument() -> DerivativeMetadata {
        DerivativeMetadata::new(
            "O:SPY-TEST-P",
            "SPY",
            derivatives::InstrumentKind::Put {
                strike: 460.0,
                // Fixed for the same reason as test_call_instrument's expiry.
                expiry: chrono::DateTime::parse_from_rfc3339("2026-10-02T00:00:00Z").unwrap().to_utc(),
            },
            100.0,
            "USD",
            "test",
        )
    }

    fn test_position(side: portfoliomanager::PositionSide, instrument: Option<DerivativeMetadata>, closed: bool) -> portfoliomanager::Position {
        let now = chrono::Utc::now();
        portfoliomanager::Position {
            symbol: instrument.as_ref().map(|i| i.symbol.clone()).unwrap_or_else(|| "TEST".to_string()),
            side,
            quantity: 1.0,
            entry_price: 100.0,
            mark_price: None,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            open_time: now,
            close_time: if closed { Some(now) } else { None },
            instrument,
            greeks: None,
            margin_posted: 0.0,
        }
    }

    // -- leverage liquidation exemption (is_exempt_from_leverage_liquidation_check) --

    #[test]
    fn liquidation_exemption_a_closed_position_is_exempt_regardless_of_instrument() {
        let pos = test_position(portfoliomanager::PositionSide::Long, None, true);
        assert!(is_exempt_from_leverage_liquidation_check(&pos));
    }

    #[test]
    fn liquidation_exemption_a_plain_leveraged_position_is_not_exempt() {
        let pos = test_position(portfoliomanager::PositionSide::Long, None, false);
        assert!(!is_exempt_from_leverage_liquidation_check(&pos));
    }

    #[test]
    fn liquidation_exemption_a_long_option_is_exempt() {
        let pos = test_position(portfoliomanager::PositionSide::Long, Some(test_call_instrument()), false);
        assert!(is_exempt_from_leverage_liquidation_check(&pos));
    }

    #[test]
    fn liquidation_exemption_a_short_written_option_is_also_exempt() {
        // Deliberate: a short option's unbounded loss potential is real, but
        // this specific leveraged-futures-style formula still doesn't model
        // real option margin-call mechanics for it -- see this function's
        // doc comment for why the account-level drawdown gate is the
        // applicable backstop instead, not this per-position check.
        let pos = test_position(portfoliomanager::PositionSide::Short, Some(test_call_instrument()), false);
        assert!(is_exempt_from_leverage_liquidation_check(&pos));
    }

    #[test]
    fn liquidation_exemption_a_future_position_is_not_exempt() {
        // Regression test for the 2026-09-02 fix: futures/perps were
        // previously (incorrectly) swept into a blanket "any derivative"
        // exemption -- they use the exact leveraged-margin accounting this
        // check models, so they must NOT be exempt.
        let future = DerivativeMetadata::new(
            "BTC-PERP", "BTC",
            derivatives::InstrumentKind::Perpetual,
            1.0, "USD", "test",
        );
        let pos = test_position(portfoliomanager::PositionSide::Long, Some(future), false);
        assert!(!is_exempt_from_leverage_liquidation_check(&pos));
    }

    // -- greeks wiring (select_greeks_vol / annualized_underlying_realized_vol) --

    #[test]
    fn select_greeks_vol_prefers_the_real_iv_surface_when_available() {
        let instrument = test_call_instrument();
        let strike = instrument.instrument_kind.strike().unwrap();
        let expiry = instrument.instrument_kind.expiry().unwrap();
        let surface = derivatives::IvSurface {
            underlying: "SPY".to_string(),
            timestamp: chrono::Utc::now(),
            points: vec![derivatives::IvPoint { strike, expiry, iv: 0.42, bid_iv: 0.41, ask_iv: 0.43 }],
        };
        let vol = select_greeks_vol(Some(&surface), &instrument, Some(0.99));
        assert_eq!(vol, Some(0.42), "the real IV surface reading must win over the fallback");
    }

    #[test]
    fn select_greeks_vol_falls_back_to_realized_vol_without_a_surface() {
        let instrument = test_call_instrument();
        let vol = select_greeks_vol(None, &instrument, Some(0.35));
        assert_eq!(vol, Some(0.35));
    }

    #[test]
    fn select_greeks_vol_falls_back_when_the_surface_has_no_points() {
        let instrument = test_call_instrument();
        let empty_surface = derivatives::IvSurface {
            underlying: "SPY".to_string(),
            timestamp: chrono::Utc::now(),
            points: Vec::new(),
        };
        let vol = select_greeks_vol(Some(&empty_surface), &instrument, Some(0.28));
        assert_eq!(vol, Some(0.28));
    }

    #[test]
    fn select_greeks_vol_returns_none_when_no_usable_source_exists() {
        let instrument = test_call_instrument();
        assert_eq!(select_greeks_vol(None, &instrument, None), None);
        // A non-finite or non-positive fallback must not be treated as usable.
        assert_eq!(select_greeks_vol(None, &instrument, Some(f64::NAN)), None);
        assert_eq!(select_greeks_vol(None, &instrument, Some(-0.1)), None);
        assert_eq!(select_greeks_vol(None, &instrument, Some(0.0)), None);
    }

    #[test]
    fn annualized_underlying_realized_vol_is_empty_for_too_short_a_series() {
        assert!(annualized_underlying_realized_vol(&[100.0], &[0], 20).is_empty());
        assert!(annualized_underlying_realized_vol(&[], &[], 20).is_empty());
    }

    #[test]
    fn annualized_underlying_realized_vol_scales_by_the_real_bars_per_year_annualization() {
        // A varying price series long enough for `lookback=3` to produce
        // real readings.
        let series = vec![100.0, 102.0, 99.0, 103.0, 98.0, 105.0, 97.0, 104.0, 101.0, 103.0];
        let n = series.len();
        // One year of daily bars -> annualization factor near `n` itself
        // (observations_per_year_from_span(n, one_year_secs) ~= n).
        let one_year_ms: i64 = (365.25 * 86_400.0 * 1000.0) as i64;
        let timestamps: Vec<i64> = (0..n as i64).map(|i| i * (one_year_ms / n as i64)).collect();

        let annualized = annualized_underlying_realized_vol(&series, &timestamps, 3);
        let raw = portfoliomanager::margin::trailing_realized_vol_series(&series, 3);
        assert_eq!(annualized.len(), raw.len());

        // Every populated entry must be the raw per-bar std scaled by the
        // SAME annualization factor (not an arbitrary/hardcoded one) --
        // confirm the ratio is constant across every populated index.
        let ratios: Vec<f64> = annualized.iter().zip(raw.iter())
            .filter_map(|(a, r)| match (a, r) {
                (Some(a), Some(r)) if *r > 0.0 => Some(a / r),
                _ => None,
            })
            .collect();
        assert!(!ratios.is_empty(), "expected at least one populated reading from this fixture");
        let first = ratios[0];
        for ratio in &ratios {
            assert!((ratio - first).abs() < 1e-6, "annualization factor must be identical across ticks: {:?}", ratios);
        }
        assert!(first > 1.0, "annualizing ~daily bars over a year should scale vol up materially, got factor {}", first);
    }

    fn vertical_spread_legs() -> Vec<OptionSpreadLeg> {
        vec![
            OptionSpreadLeg { instrument: test_call_instrument(), ratio: 1, prices: vec![5.0, 5.5, 4.5] },
            OptionSpreadLeg { instrument: test_put_instrument(), ratio: -1, prices: vec![2.0, 1.8, 2.2] },
        ]
    }

    #[test]
    fn i8_to_signals_buy_enters_spread_legs_as_configured() {
        let legs = vertical_spread_legs();
        let signals = i8_to_signals(1, 999.0, 1, None, Some(&legs));
        assert_eq!(signals.len(), 1);
        match &signals[0].signal_type {
            SignalType::OptionSpread { legs: spread_legs } => {
                assert_eq!(spread_legs.len(), 2);
                match &spread_legs[0].signal_type {
                    SignalType::BuyOption { instrument, premium } => {
                        assert_eq!(instrument, &test_call_instrument());
                        // tick_idx=1 -> this leg's own price (5.5), not the
                        // shared `price` param (999.0) passed in.
                        assert_eq!(*premium, Some(5.5));
                    }
                    other => panic!("expected BuyOption for the +1 ratio leg, got {other:?}"),
                }
                assert_eq!(spread_legs[0].ratio, 1);
                match &spread_legs[1].signal_type {
                    SignalType::SellOption { instrument, premium } => {
                        assert_eq!(instrument, &test_put_instrument());
                        assert_eq!(*premium, Some(1.8));
                    }
                    other => panic!("expected SellOption for the -1 ratio leg, got {other:?}"),
                }
                assert_eq!(spread_legs[1].ratio, -1);
            }
            other => panic!("expected OptionSpread, got {other:?}"),
        }
    }

    #[test]
    fn i8_to_signals_sell_mirrors_every_leg_to_close_or_reverse() {
        let legs = vertical_spread_legs();
        let signals = i8_to_signals(-1, 999.0, 0, None, Some(&legs));
        match &signals[0].signal_type {
            SignalType::OptionSpread { legs: spread_legs } => {
                // The +1 ratio leg (entered via BuyOption) must now close
                // via SellOption with a negated ratio; the -1 ratio leg
                // (entered via SellOption) must now close via BuyOption.
                assert!(matches!(spread_legs[0].signal_type, SignalType::SellOption { .. }));
                assert_eq!(spread_legs[0].ratio, -1);
                assert!(matches!(spread_legs[1].signal_type, SignalType::BuyOption { .. }));
                assert_eq!(spread_legs[1].ratio, 1);
            }
            other => panic!("expected OptionSpread, got {other:?}"),
        }
    }

    #[test]
    fn i8_to_signals_option_spread_takes_precedence_over_option_instrument() {
        let legs = vertical_spread_legs();
        let single = test_call_instrument();
        let signals = i8_to_signals(1, 999.0, 0, Some(&single), Some(&legs));
        assert!(matches!(signals[0].signal_type, SignalType::OptionSpread { .. }));
    }

    #[test]
    fn i8_to_signals_empty_option_spread_falls_back_to_single_leg_or_plain() {
        let empty: Vec<OptionSpreadLeg> = Vec::new();
        let signals = i8_to_signals(1, 100.0, 0, None, Some(&empty));
        assert_eq!(signals[0].signal_type, SignalType::Buy);
    }

    #[test]
    fn to_directional_signal_maps_buy_option_to_buy() {
        let sig = SignalType::BuyOption { instrument: test_call_instrument(), premium: None };
        assert_eq!(to_directional_signal(&sig), Some(SignalType::Buy));
    }

    #[test]
    fn to_directional_signal_maps_sell_option_to_sell() {
        let sig = SignalType::SellOption { instrument: test_call_instrument(), premium: None };
        assert_eq!(to_directional_signal(&sig), Some(SignalType::Sell));
    }

    #[test]
    fn to_directional_signal_maps_exercise_option_to_sell() {
        // Exercising closes the long option position -- same direction as
        // an ordinary sell-to-close.
        let sig = SignalType::ExerciseOption { instrument: test_call_instrument() };
        assert_eq!(to_directional_signal(&sig), Some(SignalType::Sell));
    }

    #[test]
    fn extract_instrument_returns_none_for_plain_buy_sell() {
        assert!(extract_instrument(&SignalType::Buy).is_none());
        assert!(extract_instrument(&SignalType::Sell).is_none());
        assert!(extract_instrument(&SignalType::Hold).is_none());
    }

    #[test]
    fn extract_instrument_returns_metadata_for_option_signals() {
        let instrument = test_call_instrument();
        let buy = SignalType::BuyOption { instrument: instrument.clone(), premium: Some(5.0) };
        let sell = SignalType::SellOption { instrument: instrument.clone(), premium: None };
        let exercise = SignalType::ExerciseOption { instrument: instrument.clone() };

        assert_eq!(extract_instrument(&buy), Some(instrument.clone()));
        assert_eq!(extract_instrument(&sell), Some(instrument.clone()));
        assert_eq!(extract_instrument(&exercise), Some(instrument));
    }

    // ---- compute_commission: real per-contract options fees, not a % of premium ----

    #[test]
    fn compute_commission_option_buy_ignores_the_generic_percentage_fee_rate() {
        let instrument = test_call_instrument();
        // A generic_fee_rate of 10% would dominate a percentage-based calc
        // (0.10 * $5 * 1 contract = $0.50) -- for an option this must be
        // completely ignored in favor of the flat per-contract schedule.
        let commission = compute_commission(5.0, 1.0, true, Some(&instrument), 0.10);
        let expected = config::OptionsFeeConfig::alpaca().commission_for_fill(5.0, 1.0, 100.0, true);
        assert!((commission - expected).abs() < 1e-9, "commission={commission}, expected={expected}");
        assert!(commission < 0.10, "must not be anywhere near the 10% generic rate: {commission}");
    }

    #[test]
    fn compute_commission_option_sell_includes_sec_fee_and_taf() {
        let instrument = test_call_instrument();
        let buy_commission = compute_commission(5.0, 1.0, true, Some(&instrument), 0.001);
        let sell_commission = compute_commission(5.0, 1.0, false, Some(&instrument), 0.001);
        assert!(sell_commission > buy_commission, "sell must add SEC fee + TAF on top of the both-ways fees");
    }

    #[test]
    fn compute_commission_non_derivative_fill_uses_the_generic_rate_unscaled() {
        // No instrument at all (a plain equity/crypto/forex fill) -- must
        // keep the pre-existing price * qty * generic_fee_rate behavior
        // exactly, with no multiplier applied.
        let commission = compute_commission(100.0, 2.0, true, None, 0.001);
        assert!((commission - 100.0 * 2.0 * 0.001).abs() < 1e-9, "commission={commission}");
    }

    #[test]
    fn compute_commission_non_option_derivative_still_scales_by_contract_multiplier() {
        // A future (not an option) with a real multiplier still applies the
        // generic percentage fee, but against the correctly-scaled notional
        // -- fixes the second bug this fee model shared with options
        // (fill_value previously had no multiplier applied at all).
        let future = DerivativeMetadata::new(
            "TEST-FUT",
            "BTC-USD",
            derivatives::InstrumentKind::Future { expiry: chrono::Utc::now() + chrono::Duration::days(30) },
            5.0, // e.g. a micro future's contract multiplier
            "USD",
            "test",
        );
        let commission = compute_commission(100.0, 2.0, true, Some(&future), 0.001);
        assert!((commission - 100.0 * 2.0 * 5.0 * 0.001).abs() < 1e-9, "commission={commission}");
    }
}
