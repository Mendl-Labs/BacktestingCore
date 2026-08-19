//! Primary SimulationLoop for built-in Rust strategies.
//!
//! Used by: `BacktestEngine::run()` for non-Python strategies.
//! NOT used for Python strategies — see `python_simulation.rs`.

use strategy::{StrategyManager, StrategyContext, PortfolioSnapshot};
use strategy::traits::{MarketSession, SessionType, PricePoint, VolumePoint, PriceSource};
use orderbook::OrderBook;
use orderbook::{OrderBookEvent, BookSide};
use genetic::adaptive_sampling::StableRegionDetector;
use portfoliomanager::PortfolioState;
use riskmanager::{RiskManager, RiskMetrics, RiskAction};
use dataloader::MarketData;
use crate::types::{BacktestEvent, BacktestResult};
use crate::collectors::{self, FillEvent, OrderEvent, MarketTradeEvent, MetricCollector};
use crate::logging_facade::BACKTEST_LOGGER;
use metrics::{performance, risk, trade_stats};
use rustc_hash::{FxHashMap, FxHashSet};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use crate::rolling_window::RollingWindow;

/// Tracks individual order lifecycle for execution metrics
#[allow(dead_code)]
struct OrderTracker {
    order_id: Arc<str>,
    side: orderbook::BookSide,
    placed_at: DateTime<Utc>,
    price: f64,
    filled: bool,
    fill_count: usize,
    first_fill_at: Option<DateTime<Utc>>,
}

pub struct SimulationLoop {
    // Track 30-day rolling volume per exchange for fee tiers.
    // VecDeque + running sum keeps updates O(1) amortized per fill instead of
    // an O(n) retain+sum over the whole history.
    trade_history_by_exchange: FxHashMap<String, std::collections::VecDeque<(chrono::DateTime<chrono::Utc>, f64)>>,
    current_30d_volume_by_exchange: FxHashMap<String, f64>,
    // Track current active order IDs to cancel before placing new orders
    // Using HashSet for O(1) contains() check instead of O(n) Vec lookup
    active_order_ids: FxHashSet<Arc<str>>,
    // Track rolling market volume for realistic market impact calculation
    // (VecDeque + running sum for O(1) amortized updates)
    market_volume_window: std::collections::VecDeque<(chrono::DateTime<chrono::Utc>, f64)>,
    market_volume_sum: f64,
    rolling_avg_volume: f64, // Average volume over rolling window
    // Track order lifecycle for execution metrics (read during event loop for missed fill and latency gating).
    // Trackers are removed once an order fills or is cancelled; lifetime fill
    // stats live in the counters below so the map stays bounded.
    order_trackers: FxHashMap<Arc<str>, OrderTracker>, // order_id -> tracker
    // Lifetime fill counters (survive tracker removal)
    orders_filled_count: usize,
    bid_orders_filled_count: usize,
    ask_orders_filled_count: usize,
    // Track quote spreads (read during loop via .last())
    quote_spreads_bps: Vec<f64>, // spread at time of quote placement
    // Performance optimization: track skipped stable regions
    stable_region_skips: usize,
    // Pluggable metric collectors (lean for GA, full for final eval)
    collectors: Vec<Box<dyn MetricCollector>>,
}

impl SimulationLoop {
    pub fn new() -> Self {
        Self {
            trade_history_by_exchange: FxHashMap::default(),
            current_30d_volume_by_exchange: FxHashMap::default(),
            active_order_ids: FxHashSet::default(),
            market_volume_window: std::collections::VecDeque::with_capacity(1000), // Track last ~1000 trades
            market_volume_sum: 0.0,
            rolling_avg_volume: 0.0,
            order_trackers: FxHashMap::default(),
            orders_filled_count: 0,
            bid_orders_filled_count: 0,
            ask_orders_filled_count: 0,
            quote_spreads_bps: Vec::with_capacity(1000),
            stable_region_skips: 0,
            collectors: collectors::full_collectors(0.0),
        }
    }

    /// Lean constructor for GA inner loop — minimal metric overhead.
    /// Only tracks timestamps; ~4x less per-tick work than full.
    pub fn new_lean() -> Self {
        let mut s = Self::new();
        s.collectors = collectors::lean_collectors();
        s
    }

    /// Full constructor for final evaluation — all diagnostic collectors.
    pub fn new_full(fee_rate: f64) -> Self {
        let mut s = Self::new();
        s.collectors = collectors::full_collectors(fee_rate);
        s
    }
    
    /// Initialize starting 30-day volume for an exchange
    /// This simulates an established market maker who already has volume history
    /// allowing them to start at a lower fee tier
    pub fn initialize_starting_volume(&mut self, config: &config::BacktestConfig) {
        if let Some(ref fee_config) = config.trading.exchange_fees {
            if fee_config.starting_30d_volume > 0.0 {
                self.current_30d_volume_by_exchange.insert(
                    fee_config.exchange.clone(),
                    fee_config.starting_30d_volume,
                );
            }
        }
    }
    
    /// Update rolling market volume from observed trades
    /// Uses a 1-hour window to calculate average trade size.
    /// O(1) amortized: running sum maintained as entries enter/leave the window.
    fn update_market_volume(&mut self, timestamp: chrono::DateTime<chrono::Utc>, trade_volume_usd: f64) {
        use chrono::Duration;

        // Add new trade
        self.market_volume_window.push_back((timestamp, trade_volume_usd));
        self.market_volume_sum += trade_volume_usd;

        // Remove trades older than 1 hour (data is time-ordered, so pop from front)
        let cutoff = timestamp - Duration::hours(1);
        while let Some(&(ts, vol)) = self.market_volume_window.front() {
            if ts >= cutoff {
                break;
            }
            self.market_volume_sum -= vol;
            self.market_volume_window.pop_front();
        }

        // Average volume per minute (for market impact comparison)
        if !self.market_volume_window.is_empty() {
            let window_minutes = 60.0; // 1 hour = 60 minutes
            self.rolling_avg_volume = self.market_volume_sum.max(0.0) / window_minutes;
        }
    }
    
    /// Get the current average market volume (per minute)
    /// Returns a reasonable default if no data yet
    fn get_avg_volume(&self) -> f64 {
        if self.rolling_avg_volume > 0.0 {
            self.rolling_avg_volume
        } else {
            // Default fallback: assume $100k per minute for BTC (reasonable for major exchanges)
            // This is conservative - will result in some market impact
            100_000.0
        }
    }
    
    /// Update 30-day rolling volume window for specific exchange.
    /// O(1) amortized: running sum maintained as entries enter/leave the window.
    fn update_30d_volume(&mut self, exchange: &str, current_time: chrono::DateTime<chrono::Utc>, trade_volume_usd: f64) {
        use chrono::Duration;

        // Get or create trade history for this exchange
        let trade_history = self.trade_history_by_exchange
            .entry(exchange.to_string())
            .or_insert_with(std::collections::VecDeque::new);

        // Add new trade and update running sum
        trade_history.push_back((current_time, trade_volume_usd));
        let volume_entry = self.current_30d_volume_by_exchange
            .entry(exchange.to_string())
            .or_insert(0.0);
        *volume_entry += trade_volume_usd;

        // Expire trades older than 30 days (data is time-ordered, pop from front)
        let cutoff_time = current_time - Duration::days(30);
        while let Some(&(ts, vol)) = trade_history.front() {
            if ts >= cutoff_time {
                break;
            }
            *volume_entry -= vol;
            trade_history.pop_front();
        }
        *volume_entry = volume_entry.max(0.0);
    }
    
    /// Get applicable fee rate based on exchange's 30-day volume
    fn get_fee_rate(&self, config: &config::BacktestConfig, exchange: &str, is_maker: bool) -> f64 {
        if let Some(ref fee_config) = config.trading.exchange_fees {
            // Only use tiered fees if the exchange matches
            if fee_config.exchange.to_lowercase() == exchange.to_lowercase() {
                let volume = self.current_30d_volume_by_exchange.get(exchange).copied().unwrap_or(0.0);
                fee_config.get_fee_rate(volume, is_maker)
            } else {
                // Different exchange, use generic fees or create config for that exchange
                let exchange_config = config::ExchangeFeeConfig::for_exchange(exchange);
                let volume = self.current_30d_volume_by_exchange.get(exchange).copied().unwrap_or(0.0);
                exchange_config.get_fee_rate(volume, is_maker)
            }
        } else {
            // Fallback to simple commission rate
            config.trading.commission_rate
        }
    }

    fn is_dex_exchange(exchange: &str) -> bool {
        let lower = exchange.to_lowercase();
        lower == "cetus" || lower == "deepbook" || lower == "cetus_amm"
            || lower == "deepbookv2" || lower == "deep_book"
    }

    fn calculate_dex_costs(
        dex_config: &config::DexCostConfig,
        fill_value: f64,
    ) -> (f64, f64) {
        let slippage_bps = dex_config.calculate_slippage_bps(fill_value);
        let slippage_cost = fill_value * (slippage_bps / 10_000.0);
        let gas_cost = dex_config.gas_cost_usd();
        (slippage_cost, gas_cost)
    }
    
    /// Track a fill for order lifecycle metrics.
    /// Updates local order_trackers and delegates time-to-fill tracking to ExecutionQualityCollector.
    fn track_fill(&mut self, order_id: &str, fill_time: DateTime<Utc>) {
        if let Some(tracker) = self.order_trackers.get_mut(order_id) {
            if !tracker.filled {
                tracker.filled = true;
                tracker.first_fill_at = Some(fill_time);
                // Lifetime counters — trackers are pruned once orders complete,
                // so end-of-run stats read these instead of the tracker map.
                self.orders_filled_count += 1;
                match tracker.side {
                    orderbook::BookSide::Bid => self.bid_orders_filled_count += 1,
                    orderbook::BookSide::Ask => self.ask_orders_filled_count += 1,
                }
            }
            tracker.fill_count += 1;
        }
        // Delegate time-to-fill tracking to ExecutionQualityCollector
        if let Some(exec) = collectors::get_collector_mut::<collectors::ExecutionQualityCollector>(&mut self.collectors) {
            exec.track_fill(order_id, fill_time);
        }
    }
    
    // Note: Slippage is now calculated internally by the orderbook during order matching.
    // The simulation tracks actual slippage from fill.slippage_bps in the fills loop.
    
    /// Calculate market impact using configured model.
    /// Delegates recording to TransactionCostCollector.
    fn calculate_market_impact(&mut self, order_size: f64, price: f64, avg_volume: f64, config: &config::BacktestConfig, timestamp: chrono::DateTime<chrono::Utc>) -> f64 {
        let impact_config = &config.data.market_impact;

        // Pure arithmetic lives on MarketImpactConfig::impact_bps so the
        // pre-submission cost-hurdle gate (asset_diagnostics_service.rs) can
        // reuse the exact same real, volume-aware model instead of a flat
        // per-asset-class guess -- see that gate's own doc comment.
        let Some(impact_bps) = impact_config.impact_bps(order_size, avg_volume) else {
            return 0.0;
        };

        // Delegate impact recording to the TransactionCostCollector
        if let Some(tc) = collectors::get_collector_mut::<collectors::TransactionCostCollector>(&mut self.collectors) {
            tc.record_market_impact(impact_bps, price, order_size, avg_volume, timestamp);
        }

        // Return the computed impact cost
        (impact_bps / 10000.0) * price * order_size
    }

    pub async fn run(
        &mut self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        market_data: &[MarketData],
        config: &config::BacktestConfig,
    ) -> BacktestResult {
    let mut events: Vec<BacktestEvent> = Vec::with_capacity(market_data.len());
    let mut result = BacktestResult::default();
    
    // Track if trading has been halted due to risk limits
    let mut trading_halted = false;
    let mut halt_reason: Option<String> = None;

    // Collectors for metrics (pre-allocated)
    let mut trade_returns: Vec<f64> = Vec::with_capacity(100);
    let mut trade_durations: Vec<f64> = Vec::with_capacity(100);

    // Adaptive equity curve sampling — avoid 240 MB for 31M-tick datasets.
    // Downsample to ~10K points for large datasets; full resolution for small ones.
    let equity_sample_interval = if market_data.len() > 1_000_000 {
        (market_data.len() / 10_000).max(1)
    } else {
        1
    };
    let equity_capacity = (market_data.len() / equity_sample_interval) + 100;
    let mut equity_curve: Vec<f64> = Vec::with_capacity(equity_capacity);
    // Unix-ms timestamps parallel to equity_curve samples (Bug #33 — lets the
    // frontend plot equity over real dates). The initial-capital point gets the
    // first sample's timestamp prepended after the loop.
    let mut equity_curve_timestamps: Vec<i64> = Vec::with_capacity(equity_capacity);

    // Start equity curve with initial capital
    equity_curve.push(config.trading.initial_capital);
    
    // Initialize risk manager with starting equity
    risk_manager.initialize(config.trading.initial_capital);
    
    // Track current market price for unrealized P&L calculation
    let mut current_market_price: Option<f64> = None;

    // Progress tracking for debugging stalls
    let total_items = market_data.len();
    let mut processed_count = 0_usize;
    let sim_start = std::time::Instant::now();
    // Log more frequently for large datasets to show progress
    let log_interval = if total_items > 1_000_000 {
        total_items / 20  // Every 5% for large datasets
    } else if total_items > 100_000 {
        total_items / 10  // Every 10% for medium datasets
    } else {
        (total_items / 10).max(1000)  // Every 10% or 1000 for small datasets
    };
    
    log_info!(BACKTEST_LOGGER, "SIMULATION_START | items={} log_interval={} ({}%)", 
        total_items, log_interval, if total_items > 0 { log_interval * 100 / total_items } else { 0 });
    
    // Early stopping for GA fitness - abort clearly failing strategies
    // More aggressive for GA mode (skip_genetic_optimization = true means we're IN a GA eval)
    let is_ga_fitness_mode = config.analysis.skip_genetic_optimization;
    let early_stop_threshold = if is_ga_fitness_mode {
        config.trading.initial_capital * 0.7 // Stop at 30% loss in GA mode (faster iterations)
    } else {
        config.trading.initial_capital * 0.5 // Stop at 50% loss normally
    };
    let early_stop_check_interval = if is_ga_fitness_mode {
        5_000_usize // Check every 5K events in GA mode
    } else {
        10_000_usize // Check every 10K events normally  
    };
    let mut _early_stopped = false;
    
    // Stable region detector: skip strategy recalculation when price hasn't moved significantly
    // This provides 1.2-1.3x speedup by avoiding redundant computations
    let opt = &config.trading.optimization;
    let mut stable_detector = if opt.stable_region_skipping {
        StableRegionDetector::new(opt.stable_region_threshold, opt.max_skip_ticks)
    } else {
        StableRegionDetector::disabled()
    };
    let mut last_signals = Vec::new(); // Cache signals from last strategy run
    
    // Pre-cache exchange name as Arc<str> — clone is just a refcount bump
    let exchange_name: Arc<str> = Arc::from(orderbook.exchange.as_str());

    // Rolling windows for price/volume history passed to strategies
    let lookback = config.trading.lookback_window;
    let mut price_window = RollingWindow::<PricePoint>::new(lookback);
    let mut volume_window = RollingWindow::<VolumePoint>::new(lookback);

    // Pre-allocate reusable buffers for StrategyContext (avoids Vec allocation per tick)
    let mut price_buf: Vec<PricePoint> = Vec::with_capacity(lookback);
    let mut volume_buf: Vec<VolumePoint> = Vec::with_capacity(lookback);

    // Track equity at the start of each UTC day so RiskMetrics.daily_pnl is an
    // actual daily figure (it was previously lifetime PnL, which made
    // daily-loss limits misfire on multi-day backtests).
    let mut current_day: Option<i64> = None;
    let mut day_start_equity = config.trading.initial_capital;

    // Cap the per-signal order-size diagnostic to the first few signals —
    // it used to log unconditionally on every signal in the hot loop.
    let mut order_size_diag_count = 0usize;

    for data in market_data {
            // Progress logging at intervals with ETA
            processed_count += 1;
            if processed_count % log_interval == 0 {
                let elapsed = sim_start.elapsed().as_secs_f64();
                let rate = processed_count as f64 / elapsed;
                let remaining = (total_items - processed_count) as f64 / rate;
                log_info!(BACKTEST_LOGGER, "SIM_PROGRESS | {}/{} ({:.0}%) | elapsed={:.1}s | rate={:.0}/s | ETA={:.0}s", 
                    processed_count, total_items, (processed_count as f64 / total_items as f64) * 100.0, 
                    elapsed, rate, remaining);
            }
            
            // Early stopping check for failing strategies (speeds up GA)
            if processed_count % early_stop_check_interval == 0 && !equity_curve.is_empty() {
                let current_equity = *equity_curve.last().unwrap_or(&config.trading.initial_capital);
                if current_equity < early_stop_threshold {
                    log_debug!(BACKTEST_LOGGER, "EARLY_STOP | equity={:.2} < threshold={:.2} | processed {}/{}", 
                        current_equity, early_stop_threshold, processed_count, total_items);
                    _early_stopped = true;
                    break;
                }
            }
            
            // 1. Update orderbook with new market data if needed
            match &data {
                MarketData::Trade(trade) => {
                    // Update current market price from trade
                    current_market_price = Some(trade.price);
                    
                    let trade_price_f64 = trade.price;
                    
                    // Use cached exchange_name instead of cloning
                    let exchange = &exchange_name;
                    
                    // Track market volume using pre-computed value
                    self.update_market_volume(trade.timestamp, trade.value);
                    
                    // Convert TradeData side to BookSide
                    // The trade side is the AGGRESSOR's side (the taker who initiated):
                    // - Buy means a buyer aggressed -> fills our ASK orders
                    // - Sell means a seller aggressed -> fills our BID orders
                    let aggressor_side = if trade.side.is_buy() {
                        BookSide::Bid   // Aggressor is buyer
                    } else {
                        BookSide::Ask   // Aggressor is seller
                    };
                    
                    // Dispatch market trade event to collectors
                    collectors::dispatch_market_trade(&mut self.collectors, &MarketTradeEvent {
                        price: trade_price_f64, quantity: trade.quantity,
                        side: aggressor_side, timestamp: trade.timestamp,
                    });
                    
                    // Track missed fill opportunities BEFORE matching
                    // This helps diagnose why we might have 0 bid fills or 0 ask fills
                    let trade_price_for_tracking = trade_price_f64;
                    if matches!(aggressor_side, BookSide::Ask) {
                        // Sell trade - could fill our bids if trade_price <= bid_price
                        // Find our best bid from trackers
                        if let Some(our_best_bid) = self.order_trackers.values()
                            .filter(|t| matches!(t.side, orderbook::BookSide::Bid))
                            .map(|t| t.price)
                            .reduce(f64::max) 
                        {
                            if trade_price_for_tracking > our_best_bid {
                                // Trade happened above our bid - missed opportunity
                                collectors::dispatch_missed_fill(&mut self.collectors, BookSide::Bid, trade_price_for_tracking - our_best_bid);
                            }
                        }
                    } else {
                        // Buy trade - could fill our asks if trade_price >= ask_price
                        // Find our best ask from trackers
                        if let Some(our_best_ask) = self.order_trackers.values()
                            .filter(|t| matches!(t.side, orderbook::BookSide::Ask))
                            .map(|t| t.price)
                            .reduce(f64::min) 
                        {
                            if trade_price_for_tracking < our_best_ask {
                                // Trade happened below our ask - missed opportunity
                                collectors::dispatch_missed_fill(&mut self.collectors, BookSide::Ask, our_best_ask - trade_price_for_tracking);
                            }
                        }
                    }
                    
                    // Execute the trade against our resting orders and get fills
                    let trade_fills = orderbook.execute_trade_with_fills(
                        aggressor_side,
                        trade.price,
                        trade.quantity,
                        trade.timestamp,
                    );
                    
                    #[cfg(feature = "debug_logging")]
                    static TRADE_DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    #[cfg(feature = "debug_logging")]
                    let debug_count = TRADE_DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    
                    // Get trade price early for missed opportunity tracking (use pre-computed)
                    let _trade_price = trade_price_f64;
                    
                    // Filter fills to only include OUR strategy's orders
                    // Also apply latency simulation: orders placed within latency_ms can't be filled yet
                    let latency_ms = config.trading.latency_ms as i64;
                    let latency_cutoff = trade.timestamp - chrono::Duration::milliseconds(latency_ms);

                    // Track latency rejections separately for debugging
                    let mut latency_rejected_bid = 0usize;
                    let mut latency_rejected_ask = 0usize;
                    let mut latency_rejected_ids: Vec<Arc<str>> = Vec::new();

                    let mut our_fills: Vec<orderbook::Fill> = Vec::new();
                    for fill in trade_fills {
                        // Must be our order
                        if !self.active_order_ids.contains(&fill.order_id) {
                            continue;
                        }
                        // Latency simulation: order must have been placed before the cutoff
                        // (i.e., order has had time to propagate to exchange)
                        if let Some(tracker) = self.order_trackers.get(&*fill.order_id) {
                            if tracker.placed_at > latency_cutoff {
                                // Track latency rejection by side
                                match tracker.side {
                                    orderbook::BookSide::Bid => latency_rejected_bid += 1,
                                    orderbook::BookSide::Ask => latency_rejected_ask += 1,
                                }
                                latency_rejected_ids.push(fill.order_id.clone());
                                continue;
                            }
                            our_fills.push(fill);
                        }
                        // No tracker found: skip this fill
                    }

                    // A latency-rejected fill means the orderbook already consumed the
                    // resting order — it can never fill again. Drop it from tracking so
                    // the active set and tracker map stay consistent with the book
                    // (previously it lingered as "active", skewing missed-fill
                    // diagnostics and leaking tracker entries).
                    for order_id in &latency_rejected_ids {
                        self.active_order_ids.remove(order_id);
                        self.order_trackers.remove(order_id.as_ref() as &str);
                    }

                    // Dispatch latency rejections to collectors
                    for _ in 0..latency_rejected_bid {
                        collectors::dispatch_latency_rejection(&mut self.collectors, BookSide::Bid);
                    }
                    for _ in 0..latency_rejected_ask {
                        collectors::dispatch_latency_rejection(&mut self.collectors, BookSide::Ask);
                    }
                    
                    #[cfg(feature = "debug_logging")]
                    if debug_count < 10 && !self.active_order_ids.is_empty() {
                        let trade_price = trade_price_f64;
                        let best_bid = orderbook.best_bid();
                        let best_ask = orderbook.best_ask();
                        // Strategy's bid/ask from the trackers
                        let our_bid = self.order_trackers.values()
                            .filter(|t| matches!(t.side, orderbook::BookSide::Bid))
                            .map(|t| t.price)
                            .reduce(f64::max);
                        let our_ask = self.order_trackers.values()
                            .filter(|t| matches!(t.side, orderbook::BookSide::Ask))
                            .map(|t| t.price)
                            .reduce(f64::min);
                        log_info!(BACKTEST_LOGGER,
                            "TRADE_VS_ORDERS | trade=${:.2} side={:?} | our_bid=${:.2} our_ask=${:.2} | book_bid=${:.2} book_ask=${:.2} | our_fills={} latency_rejected={}",
                            trade_price, trade.side,
                            our_bid.unwrap_or(0.0), our_ask.unwrap_or(0.0),
                            best_bid.unwrap_or(0.0), best_ask.unwrap_or(0.0), 
                            our_fills.len(), latency_rejected_bid + latency_rejected_ask);
                    }
                    
                    // Apply fills one at a time so commission is charged on the quantity
                    // actually executed (the portfolio may cap execution by available
                    // balance/inventory), and so the SAME commission both hits equity
                    // (via fill.fee_amount) and is reported to collectors. Previously the
                    // orderbook's independent fee state was charged to the portfolio
                    // while a separately-computed commission was reported — the two
                    // could silently diverge.
                    for mut fill in our_fills {
                        // Our resting orders are MAKERS
                        let fee_rate = self.get_fee_rate(config, exchange, true);
                        fill.fee_rate = fee_rate;
                        fill.fee_amount = fill.price * fill.quantity * fee_rate;

                        let requested_quantity = fill.quantity;
                        let executed_quantity = match portfolio.apply_fill_executed(&fill) {
                            Ok(q) => q,
                            Err(e) => {
                                log_error!(BACKTEST_LOGGER, "Error applying trade fill: {}", e);
                                0.0
                            }
                        };

                        // The order was consumed from the book either way — stop tracking it.
                        self.active_order_ids.remove(&fill.order_id);

                        if executed_quantity <= 0.0 || requested_quantity <= 0.0 {
                            self.order_trackers.remove(fill.order_id.as_ref() as &str);
                            continue;
                        }

                        let execution_ratio = executed_quantity / requested_quantity;
                        let fill_value = fill.price * executed_quantity;
                        let commission = fill.fee_amount * execution_ratio;

                        // Track order lifecycle (reads tracker side — must precede removal)
                        self.track_fill(&fill.order_id, trade.timestamp);
                        self.order_trackers.remove(fill.order_id.as_ref() as &str);

                        // Track commission by fee tier (convert to bps for bucketing)
                        let tier_bps = (fee_rate * 10000.0).round() as u32;

                        // Update volume tracking for this exchange (executed value only)
                        self.update_30d_volume(exchange, trade.timestamp, fill_value);

                        collectors::dispatch_fill(&mut self.collectors, &FillEvent {
                            order_id: fill.order_id.clone(),
                            side: fill.side,
                            price: fill.price,
                            quantity: executed_quantity,
                            commission,
                            slippage: 0.0,
                            exchange: exchange.clone(),
                            fee_tier_bps: tier_bps,
                            timestamp: trade.timestamp,
                        });
                    }
                }
                MarketData::Candle(candle) => {
                    // Update current market price from candle close
                    current_market_price = Some(candle.close);
                }
                #[allow(unreachable_patterns)]
                _ => {
                    // Skip unknown/feature-gated variants
                }
            }

            // 2. Build strategy context (timestamp, portfolio, etc.)
            let timestamp = match &data {
                MarketData::Candle(c) => c.timestamp,
                MarketData::Trade(t) => t.timestamp,
                MarketData::PoolSwap(s) => s.timestamp,
                MarketData::Generic(g) => chrono::DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(chrono::DateTime::UNIX_EPOCH),
                MarketData::OptionCandle(c) => c.timestamp,
            };

            // Push into rolling windows for price/volume history
            if let Some(price) = current_market_price {
                price_window.push(PricePoint {
                    timestamp,
                    price,
                    source: match &data {
                        MarketData::Trade(_) => PriceSource::Trade,
                        _ => PriceSource::MidPrice,
                    },
                });
            }
            match &data {
                MarketData::Trade(t) => {
                    volume_window.push(VolumePoint {
                        timestamp,
                        volume: t.quantity,
                        side: None,
                    });
                }
                MarketData::Candle(c) => {
                    volume_window.push(VolumePoint {
                        timestamp,
                        volume: c.volume,
                        side: None,
                    });
                }
                _ => {}
            }

            // -- Reuse buffers for price/volume history (avoids Vec alloc per tick) --
            price_buf.clear();
            let (s1, s2) = price_window.as_slices();
            price_buf.extend_from_slice(s1);
            price_buf.extend_from_slice(s2);

            volume_buf.clear();
            let (v1, v2) = volume_window.as_slices();
            volume_buf.extend_from_slice(v1);
            volume_buf.extend_from_slice(v2);

            let context = StrategyContext {
                timestamp,
                portfolio_state: PortfolioSnapshot::from(&*portfolio),
                price_history: std::mem::take(&mut price_buf),
                volume_history: std::mem::take(&mut volume_buf),
                spread: {
                    // Calculate spread from orderbook best bid/ask
                    match (orderbook.best_bid(), orderbook.best_ask()) {
                        (Some(bid), Some(ask)) => Some(ask - bid),
                        _ => None,
                    }
                },
                market_session: MarketSession {
                    is_open: true,
                    session_type: SessionType::Regular,
                    next_open: None,
                    next_close: None,
                }, // Default session
                orderbook_snapshot: None,
                custom_data: std::collections::HashMap::new(),
                assets: std::collections::HashMap::new(),
                iv_surface: None,
                futures_curve: None,
                funding_rates: None,
                portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
                tick_number: processed_count as u64,
                elapsed_seconds: 0.0,
                pool_state: None,
            };
            
            // Skip signal processing if trading has been halted
            if trading_halted {
                // Reclaim buffers before continuing
                price_buf = context.price_history;
                volume_buf = context.volume_history;
                // Still update equity curve even when halted
                if let Some(price) = current_market_price {
                    portfolio.update_unrealized_pnl(price);
                }
                let total_portfolio_value = portfolio.get_total_value();
                if processed_count % equity_sample_interval == 0 || processed_count == total_items {
                    equity_curve.push(total_portfolio_value);
                    equity_curve_timestamps.push(timestamp.timestamp_millis());
                }
                continue;
            }

            // 3. Run strategies with stable region optimization
            // Extract current price for stable region detection
            let current_price_f64 = current_market_price
                .unwrap_or(0.0);
            
            // Check if we should run strategy or reuse cached signals
            let signals = if stable_detector.should_process(current_price_f64) {
                // Price moved significantly or max skip reached - run strategy
                last_signals = strategy_manager.process_market_data(&data, &context).await.unwrap_or_default();
                &last_signals
            } else {
                // Stable region - reuse cached signals (skip expensive strategy computation)
                self.stable_region_skips += 1;
                &last_signals
            };

            // Reclaim Vec buffers from context to avoid re-allocation next tick
            price_buf = context.price_history;
            volume_buf = context.volume_history;
            
            // Cancel previous active orders before placing new quotes
            // This prevents order accumulation that leads to incorrect fill volumes
            if !signals.is_empty() {
                for order_id in &self.active_order_ids {
                    let cancel_event = OrderBookEvent::CancelOrder { order_id: order_id.clone(), timestamp };
                    let _ = orderbook.apply_event(cancel_event);
                    // Track cancellation if order was never filled
                    if let Some(tracker) = self.order_trackers.get(order_id.as_ref() as &str) {
                        if !tracker.filled {
                            collectors::dispatch_order_cancelled(&mut self.collectors, order_id, timestamp);
                        }
                    }
                    self.order_trackers.remove(order_id.as_ref() as &str);
                }
                self.active_order_ids.clear();
            }
            
            // Calculate current risk metrics for risk checks
            let current_equity = portfolio.get_total_value();
            let current_position = portfolio.positions.iter()
                .filter(|p| p.close_time.is_none())
                .map(|p| {
                    let qty = p.quantity;
                    match p.side {
                        portfoliomanager::PositionSide::Long => qty,
                        portfoliomanager::PositionSide::Short => -qty,
                    }
                })
                .sum::<f64>();
            
            // Update risk manager's equity tracking
            risk_manager.update_equity(current_equity);
            let current_drawdown = risk_manager.get_current_drawdown(current_equity);

            // Roll the daily P&L base at UTC day boundaries
            let day_number = timestamp.timestamp().div_euclid(86_400);
            if current_day != Some(day_number) {
                current_day = Some(day_number);
                day_start_equity = current_equity;
            }

            // Build risk metrics
            let risk_metrics = RiskMetrics {
                current_position,
                current_order_size: 0.0,
                current_inventory_skew: 0.0,
                current_volatility: 0.0,
                current_drawdown,
                unrealized_pnl: portfolio.positions.iter()
                    .filter(|p| p.close_time.is_none())
                    .map(|p| p.unrealized_pnl)
                    .sum(),
                daily_pnl: current_equity - day_start_equity,
                portfolio_greeks: Some(portfolio.get_portfolio_greeks()),
                // VaR metrics - defaults for simulation
                current_var: 0.0,
                current_cvar: 0.0,
                // Correlation metrics
                correlated_exposure: 0.0,
                sector_exposures: std::collections::HashMap::new(),
                // Volatility regime
                volatility_regime: riskmanager::VolatilityRegime::Normal,
                baseline_volatility: 0.0,
            };
            
            // Check global risk limits before processing signals
            if let Err(reason) = risk_manager.check(&risk_metrics) {
                log_error!(BACKTEST_LOGGER, "Risk limit exceeded: {}", reason);
                trading_halted = true;
                halt_reason = Some(reason);
                continue;
            }
            
            for (_strategy_id, sigs) in signals.iter() {
                // Track bid/ask prices for spread calculation
                let mut last_bid_price: Option<f64> = None;
                let mut last_ask_price: Option<f64> = None;
                
                for signal in sigs {
                    // Extract order details for cost calculation
                    let order_price = signal.price
                        .unwrap_or(0.0);
                    let mut order_size = signal.quantity
                        .unwrap_or(0.0);
                    // Use pre-cached exchange_name — Arc::clone is just a refcount bump
                    let exchange = &exchange_name;
                    
                    // Track bid/ask prices for spread calculation
                    match signal.signal_type {
                        signal::SignalType::Buy | signal::SignalType::ScaleIn => {
                            last_bid_price = Some(order_price);
                        }
                        signal::SignalType::Sell | signal::SignalType::ScaleOut => {
                            last_ask_price = Some(order_price);
                        }
                        _ => {}
                    }
                    
                    // Calculate and store quote spread when we have both bid and ask
                    if let (Some(bid), Some(ask)) = (last_bid_price, last_ask_price) {
                        if bid > 0.0 && ask > 0.0 {
                            let mid = (bid + ask) / 2.0;
                            let spread_bps = (ask - bid) / mid * 10000.0;
                            self.quote_spreads_bps.push(spread_bps);
                            
                            // Log quote vs market price for debugging (first 10 quotes)
                            if self.quote_spreads_bps.len() <= 10 {
                                let market_price = current_market_price
                                    .unwrap_or(0.0);
                                let bid_distance_bps = if market_price > 0.0 { (market_price - bid) / market_price * 10000.0 } else { 0.0 };
                                let ask_distance_bps = if market_price > 0.0 { (ask - market_price) / market_price * 10000.0 } else { 0.0 };
                                log_info!(BACKTEST_LOGGER, 
                                    "QUOTE_DEBUG | market=${:.2} bid=${:.2} ask=${:.2} spread={:.1}bps bid_dist={:.1}bps ask_dist={:.1}bps",
                                    market_price, bid, ask, spread_bps, bid_distance_bps, ask_distance_bps);
                            }
                            
                            // Reset for next pair
                            last_bid_price = None;
                            last_ask_price = None;
                        }
                    }
                    
                    // Debug: Log first few order sizes from strategy with full context
                    if order_size_diag_count < 5 {
                        order_size_diag_count += 1;
                        let order_value = order_price * order_size;
                        let expected_value_2pct = config.trading.initial_capital * 0.02; // Expected at 2%
                        // Calculate implied order_size_pct from actual order value
                        let implied_pct = if config.trading.initial_capital > 0.0 { order_value / config.trading.initial_capital * 100.0 } else { 0.0 };
                        let portfolio_balance = portfolio.balance;
                        let portfolio_position = portfolio.net_position;
                        log_info!(BACKTEST_LOGGER,
                            "ORDER_SIZE_DIAG | portfolio: balance=${:.2} position={:.6} | order: price=${:.2} size={:.6} value=${:.2} | config: initial_cap=${:.0} expected_2pct=${:.2} implied_pct={:.2}%",
                            portfolio_balance, portfolio_position, order_price, order_size, order_value,
                            config.trading.initial_capital, expected_value_2pct, implied_pct);
                    }
                    
                    // Check if this specific order should be allowed
                    match risk_manager.check_order(order_size, current_position, &risk_metrics) {
                        RiskAction::Proceed => {
                            // Order is allowed, continue
                        }
                        RiskAction::ReducePosition(allowed_size) => {
                            log_info!(BACKTEST_LOGGER, "Risk: reducing order size from {} to {}", order_size, allowed_size);
                            order_size = allowed_size;
                            if order_size <= 0.0 {
                                continue; // Skip this order entirely
                            }
                        }
                        RiskAction::ScalePosition(scale_factor, reason) => {
                            log_info!(BACKTEST_LOGGER, "Risk: scaling position by {:.2}x - {}", scale_factor, reason);
                            order_size *= scale_factor;
                            if order_size <= 0.0 {
                                continue; // Skip this order entirely
                            }
                        }
                        RiskAction::RejectOrder(reason) => {
                            log_info!(BACKTEST_LOGGER, "Risk: rejecting order - {}", reason);
                            continue;
                        }
                        RiskAction::CloseAllPositions(reason) => {
                            log_error!(BACKTEST_LOGGER, "Risk: closing all positions - {}", reason);
                            // TODO: Implement position closing logic
                            trading_halted = true;
                            halt_reason = Some(reason);
                            continue;
                        }
                        RiskAction::HaltTrading(reason) => {
                            log_error!(BACKTEST_LOGGER, "Risk: halting trading - {}", reason);
                            trading_halted = true;
                            halt_reason = Some(reason);
                            continue;
                        }
                    }
                    
                    // Note: Slippage is now handled internally by the orderbook during order matching
                    // We track the actual slippage from fills below

                    // Market impact applies to MARKET orders only (takers move the market;
                    // resting limit orders are makers). The cost is charged into the fills'
                    // fee below so the reported impact reconciles with equity — it was
                    // previously computed and reported but never applied to P&L.
                    let is_limit_order = signal.price.is_some();
                    let impact_cost = if is_limit_order {
                        0.0
                    } else {
                        let avg_volume = self.get_avg_volume();
                        self.calculate_market_impact(order_size, order_price, avg_volume, config, timestamp)
                    };
                    
                    // Track this order ID for later cancellation (O(1) insert with HashSet)
                    self.active_order_ids.insert(signal.id.clone());
                    
                    // Determine order side from signal type
                    let order_side = match signal.signal_type {
                        signal::SignalType::Buy | signal::SignalType::ScaleIn => orderbook::BookSide::Bid,
                        _ => orderbook::BookSide::Ask,
                    };
                    
                    // Track order placement for execution metrics
                    self.order_trackers.insert(signal.id.clone(), OrderTracker {
                        order_id: signal.id.clone(),
                        side: order_side,
                        placed_at: timestamp,
                        price: order_price,
                        filled: false,
                        fill_count: 0,
                        first_fill_at: None,
                    });
                    // Dispatch order placement to collectors
                    let spread_bps_val = self.quote_spreads_bps.last().copied().unwrap_or(0.0);
                    collectors::dispatch_order_placed(&mut self.collectors, &OrderEvent {
                        order_id: signal.id.clone(),
                        side: order_side,
                        price: order_price,
                        spread_bps: spread_bps_val,
                        timestamp,
                    });
                    
                    // 4. Submit order to orderbook
                    let _ = orderbook.submit_order_from_signal(&signal, timestamp);

                    // 5. Simulate order matching/fills
                    // For LIMIT orders (with price): DON'T immediately match - they rest on the book
                    //   and only get filled when external market trades hit them via execute_trade_with_fills()
                    // For MARKET orders (without price): Immediately match against resting orders
                    let fills = if is_limit_order {
                        // Limit orders rest on the book - no immediate matching
                        // They will be filled by market trades in the MarketData::Trade handler
                        Vec::new()
                    } else {
                        // Market orders immediately sweep the book
                        orderbook.match_all_orders(timestamp)
                    };
                    
                    // 6. Charge and report costs from fills.
                    // Applied per fill so commission is charged on the quantity actually
                    // executed, and the SAME cost hits both equity (fill.fee_amount) and
                    // the collectors. Market impact is distributed across fills by value.
                    // Exchange slippage from fill.slippage_bps is already embedded in the
                    // fill price, so it is reported but NOT charged again; DEX model
                    // slippage is a real cost not in the price, so it IS charged.
                    let total_fill_value: f64 = fills.iter().map(|f| f.price * f.quantity).sum();
                    for mut fill in fills {
                        let requested_value = fill.price * fill.quantity;

                        let dex_enabled = Self::is_dex_exchange(&exchange)
                            && config.trading.dex_costs.as_ref().map(|d| d.enabled).unwrap_or(false);
                        let (commission_req, slippage_req, tier_bps, slippage_is_charged) = if dex_enabled {
                            let dex_config = config.trading.dex_costs.as_ref().unwrap();
                            let (slippage, gas) = Self::calculate_dex_costs(dex_config, requested_value);
                            let slippage_bps = dex_config.calculate_slippage_bps(requested_value);
                            (gas, slippage, slippage_bps.round() as u32, true)
                        } else {
                            // Immediate fills only occur for market orders → taker fees.
                            let fee_rate = self.get_fee_rate(config, &exchange, false);
                            let comm = requested_value * fee_rate;
                            let slip = fill.slippage_bps.map(|bps| requested_value * (bps / 10000.0)).unwrap_or(0.0);
                            (comm, slip, (fee_rate * 10000.0).round() as u32, false)
                        };

                        let impact_share = if total_fill_value > 0.0 {
                            impact_cost * (requested_value / total_fill_value)
                        } else {
                            0.0
                        };
                        fill.fee_amount = commission_req
                            + impact_share
                            + if slippage_is_charged { slippage_req } else { 0.0 };

                        let requested_quantity = fill.quantity;
                        let executed_quantity = match portfolio.apply_fill_executed(&fill) {
                            Ok(q) => q,
                            Err(e) => {
                                log_error!(BACKTEST_LOGGER, "Error applying fill: {}", e);
                                0.0
                            }
                        };
                        if executed_quantity <= 0.0 || requested_quantity <= 0.0 {
                            continue;
                        }
                        let execution_ratio = executed_quantity / requested_quantity;
                        let fill_value = fill.price * executed_quantity;

                        // Update volume tracking for this exchange (executed value only)
                        self.update_30d_volume(&exchange, timestamp, fill_value);

                        // Dispatch immediate fill to collectors (executed amounts)
                        collectors::dispatch_fill(&mut self.collectors, &FillEvent {
                            order_id: fill.order_id.clone(),
                            side: order_side,
                            price: fill.price,
                            quantity: executed_quantity,
                            commission: commission_req * execution_ratio,
                            slippage: slippage_req * execution_ratio,
                            exchange: exchange_name.clone(),
                            fee_tier_bps: tier_bps,
                            timestamp,
                        });
                    }

                    // Update risk manager equity after fills
                    let new_equity = portfolio.get_total_value();
                    risk_manager.update_equity(new_equity);

                    // 7. Record event
                    events.push(BacktestEvent::OrderSubmitted {
                        order_id: signal.id.clone(),
                        timestamp,
                    });
                }
            }

            // Update unrealized P&L with current market price before collecting equity
            if let Some(price) = current_market_price {
                portfolio.update_unrealized_pnl(price);
                
                // Sample inventory value and dispatch to collectors
                let inventory_value = portfolio.get_total_value() - portfolio.balance;
                let net_pos = portfolio.net_position;
                for c in self.collectors.iter_mut() {
                    c.on_inventory_sample(net_pos, timestamp);
                    c.on_inventory_value(inventory_value);
                }
            }
            
            // Collect TOTAL portfolio value (balance + unrealized P&L) at each step
            // This correctly reflects the portfolio's worth including open positions
            let total_portfolio_value = portfolio.get_total_value();
            if processed_count % equity_sample_interval == 0 || processed_count == total_items {
                equity_curve.push(total_portfolio_value);
                equity_curve_timestamps.push(timestamp.timestamp_millis());
            }
        }

        // ========== DETAILED FILL DIAGNOSTICS ==========
        // Collector snapshots for diagnostic logging
        let tc = collectors::get_collector::<collectors::TransactionCostCollector>(&self.collectors);
        let inv = collectors::get_collector::<collectors::InventoryCollector>(&self.collectors);
        let exec = collectors::get_collector::<collectors::ExecutionQualityCollector>(&self.collectors);
        let missed = collectors::get_collector::<collectors::MissedOpportunityCollector>(&self.collectors);
        let rt = collectors::get_collector::<collectors::RoundTripCollector>(&self.collectors);
        let ts = collectors::get_collector::<collectors::TimestampCollector>(&self.collectors);

        let total_fill_count = exec.map(|c| c.total_fill_count).unwrap_or(0);
        let total_order_count = exec.map(|c| c.total_order_count).unwrap_or(0);
        let buy_fills = inv.map(|c| c.buy_fills).unwrap_or(0);
        let sell_fills = inv.map(|c| c.sell_fills).unwrap_or(0);
        let latency_rejections = exec.map(|c| c.latency_rejections).unwrap_or(0);
        let latency_rejections_bid = exec.map(|c| c.latency_rejections_bid).unwrap_or(0);
        let latency_rejections_ask = exec.map(|c| c.latency_rejections_ask).unwrap_or(0);
        let orders_cancelled = exec.map(|c| c.orders_cancelled).unwrap_or(0);
        let missed_bid_opportunities = missed.map(|c| c.missed_bid_opportunities).unwrap_or(0);
        let missed_ask_opportunities = missed.map(|c| c.missed_ask_opportunities).unwrap_or(0);
        let bid_price_gap_sum = missed.map(|c| c.bid_price_gap_sum).unwrap_or(0.0);
        let ask_price_gap_sum = missed.map(|c| c.ask_price_gap_sum).unwrap_or(0.0);
        let market_buy_trades = inv.map(|c| c.market_buy_trades).unwrap_or(0);
        let market_sell_trades = inv.map(|c| c.market_sell_trades).unwrap_or(0);
        let buy_volume = inv.map(|c| c.buy_volume).unwrap_or(0.0);
        let sell_volume = inv.map(|c| c.sell_volume).unwrap_or(0.0);
        let buy_qty = inv.map(|c| c.buy_qty).unwrap_or(0.0);
        let sell_qty = inv.map(|c| c.sell_qty).unwrap_or(0.0);
        let max_inventory = inv.map(|c| c.max_inventory).unwrap_or(0.0);
        let min_inventory = inv.map(|c| c.min_inventory).unwrap_or(0.0);
        let total_commission = tc.map(|c| c.total_commission).unwrap_or(0.0);
        let bid_orders_placed = exec.map(|c| c.bid_orders_placed).unwrap_or(0);
        let ask_orders_placed = exec.map(|c| c.ask_orders_placed).unwrap_or(0);

        // Log comprehensive diagnostics to understand why fills aren't happening
        log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | total_orders={} total_fills={} buy_fills={} sell_fills={} latency_rejected={} (bid={} ask={})",
            total_order_count, total_fill_count, buy_fills, sell_fills,
            latency_rejections, latency_rejections_bid, latency_rejections_ask);
        log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | orders_cancelled={} missed_bid_opps={} missed_ask_opps={}",
            orders_cancelled, missed_bid_opportunities, missed_ask_opportunities);
        log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | active_orders_remaining={} order_trackers_remaining={}",
            self.active_order_ids.len(), self.order_trackers.len());
        
        // Analyze quote spreads if we have them
        if !self.quote_spreads_bps.is_empty() {
            let avg_spread = self.quote_spreads_bps.iter().sum::<f64>() / self.quote_spreads_bps.len() as f64;
            let min_spread = self.quote_spreads_bps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_spread = self.quote_spreads_bps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | quote_spreads: count={} avg={:.1}bps min={:.1}bps max={:.1}bps",
                self.quote_spreads_bps.len(), avg_spread, min_spread, max_spread);
        } else {
            log_warn!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | NO QUOTES GENERATED - strategy didn't produce any bid/ask pairs!");
        }
        
        // Analyze price gaps for missed opportunities
        if missed_bid_opportunities > 0 || missed_ask_opportunities > 0 {
            let avg_bid_gap = if missed_bid_opportunities > 0 { 
                bid_price_gap_sum / missed_bid_opportunities as f64 
            } else { 0.0 };
            let avg_ask_gap = if missed_ask_opportunities > 0 { 
                ask_price_gap_sum / missed_ask_opportunities as f64 
            } else { 0.0 };
            log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | price_gaps: avg_bid_gap=${:.2} avg_ask_gap=${:.2} (quotes too far from market)",
                avg_bid_gap, avg_ask_gap);
        }
        
        // Log market trade stats for context
        log_info!(BACKTEST_LOGGER, "FILL_DIAGNOSTICS | market_trades: buys={} sells={} (potential opportunities)",
            market_buy_trades, market_sell_trades);

        // Collect trade returns and durations from closed positions (do this after market data processing)
        let mut closed_position_count = 0;
        let mut open_position_count = 0;
        let mut total_unrealized_pnl = 0.0;
        
        for pos in &portfolio.positions {
            if let Some(close_time) = pos.close_time {
                // Only consider closed positions for trade statistics
                closed_position_count += 1;
                trade_returns.push(pos.realized_pnl);
                let duration = (close_time - pos.open_time).num_seconds() as f64;
                trade_durations.push(duration);
            } else {
                // Track open positions and their unrealized P&L
                open_position_count += 1;
                total_unrealized_pnl += pos.unrealized_pnl;
            }
        }
        
        // Log position breakdown for debugging
        log_info!(BACKTEST_LOGGER, "Position breakdown: closed={} open={} unrealized_pnl={:.2}",
            closed_position_count, open_position_count, total_unrealized_pnl);

        // Compute metrics using collected data
        // NOTE: These metrics are based on CLOSED positions only.
        // If there are open positions with unrealized P&L, the total_pnl may differ from net_profit.
        
        // Some strategies (e.g. those continuously adjusting inventory via quotes)
        // generate fills without ever closing a position. Fall back to counting
        // fills as trades in that case.
        // A "round-trip" is min(buy_fills, sell_fills) - completed inventory cycles
        let fill_round_trips = buy_fills.min(sell_fills);
        let total_fills = buy_fills + sell_fills;
        
        // Store info for later win rate calculation (after total_pnl is calculated)
        let is_fills_only_mode = closed_position_count == 0 && total_fills > 0;
        
        log_info!(BACKTEST_LOGGER, "Fills-only mode check: closed={} total_fills={} is_fills_only_mode={}",
            closed_position_count, total_fills, is_fills_only_mode);
        
        // Use fill-based metrics if we have fills but no closed positions
        // This handles the case where a strategy adjusts inventory without closing positions
        if is_fills_only_mode {
            // Fills-only mode: count fills as trades
            result.num_trades = fill_round_trips;
            
            log_info!(BACKTEST_LOGGER, "Fills-only mode: using fills as trades | round_trips={} total_fills={} buy={} sell={}",
                fill_round_trips, total_fills, buy_fills, sell_fills);
            
            // Calculate metrics from fills rather than closed positions
            // We'll populate these after total_pnl is calculated (below)
        } else {
            // Traditional mode: use closed positions
            result.num_trades = performance::number_of_trades(&trade_returns);
            
            // Traditional metrics from closed positions
            result.gross_profit = Some(performance::gross_profit(&trade_returns));
            result.gross_loss = Some(performance::gross_loss(&trade_returns));
            result.net_profit = Some(performance::net_profit(&trade_returns));
            result.avg_trade_return = performance::average_trade_return(&trade_returns);
            result.median_trade_return = performance::median_trade_return(&trade_returns);
        }
        
        // For Sharpe ratio, if we have very few closed trades but significant unrealized P&L,
        // use the equity curve returns instead for a more representative metric
        let use_equity_returns = trade_returns.len() < 10 && open_position_count > 0;

        // Equity-curve samples per day, from the actual data time range.
        // Used by both the equity-curve Sharpe and the Calmar annualization.
        let periods_per_day = if let Some(tc) = ts {
            if let (Some(start), Some(end)) = (tc.first_trade_timestamp, tc.last_trade_timestamp) {
                let total_secs = (end - start).num_seconds().max(1) as f64;
                let total_days = total_secs / 86400.0;
                if total_days > 0.0 && equity_curve.len() > 1 {
                    ((equity_curve.len() as f64) / total_days).max(1.0) as usize
                } else {
                    equity_curve.len().max(1) // fallback: 1 day of data
                }
            } else {
                equity_curve.len().max(1)
            }
        } else {
            equity_curve.len().max(1)
        };

        // Calculate Sharpe ratio properly:
        // - For equity curve returns: use the dedicated function with proper annualization
        // - For trade returns: the function now handles dollar vs percentage returns
        result.sharpe_ratio = if use_equity_returns {
            performance::sharpe_ratio_from_equity_curve(&equity_curve, config.trading.risk_free_rate, periods_per_day)
        } else {
            // Use trade returns (the function now handles dollar amounts properly)
            performance::sharpe_ratio(&trade_returns, config.trading.risk_free_rate)
        };
        
        result.profit_factor = performance::profit_factor(
            result.gross_profit.unwrap_or(0.0),
            result.gross_loss.unwrap_or(0.0),
        );
        
        // Calculate win rate - only override if not already set by fills-only logic
        if result.win_rate.is_none() {
            result.win_rate = if closed_position_count > 0 {
                trade_stats::win_rate(&trade_returns)
            } else {
                // No closed trades and no fills - win rate is undefined
                None
            };
        }
        result.avg_trade_duration = trade_stats::average_trade_duration(&trade_durations);
        
        // Populate closed/open position counts and realized/unrealized PnL for transparency
        result.closed_trades = closed_position_count;
        result.open_positions = open_position_count;
        let realized_pnl = performance::net_profit(&trade_returns);
        result.realized_pnl = Some(realized_pnl);
        result.unrealized_pnl = Some(total_unrealized_pnl);
        
        // Log warning if PnL is primarily from unrealized gains
        if total_unrealized_pnl.abs() > realized_pnl.abs() && open_position_count > 0 {
            log_info!(BACKTEST_LOGGER, "WARNING: PnL dominated by unrealized gains. realized={:.2} unrealized={:.2}",
                realized_pnl, total_unrealized_pnl);
        }
        
        // Calculate volatility using percentage returns from equity curve (not dollar returns)
        // This gives a proper annualizable volatility metric
        let pct_returns: Vec<f64> = if equity_curve.len() > 1 {
            equity_curve.windows(2)
                .filter_map(|w| {
                    if w[0] > 0.0 {
                        Some((w[1] - w[0]) / w[0])
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        result.volatility = risk::volatility(&pct_returns);
        result.max_drawdown = risk::max_drawdown(&equity_curve)
            .unwrap_or(0.0);
        
        // Calculate Sortino ratio using percentage returns (same as Sharpe for consistency)
        // Sortino uses downside deviation instead of total volatility
        result.sortino_ratio = risk::sortino_ratio(&pct_returns, config.trading.risk_free_rate);
        
        // Calculate Calmar ratio (annualized return / max drawdown)
        // Uses percentage-based annualized return for proper ratio
        let max_dd = result.max_drawdown;
        if !pct_returns.is_empty() && max_dd > 0.0 {
            // Annualize the mean per-SAMPLE return using the actual sample
            // frequency (periods_per_day × 365 for crypto). Equity samples are
            // ticks (or downsampled ticks), not days — multiplying the per-tick
            // mean by 365 alone massively understated the annualized return.
            let mean_return = pct_returns.iter().sum::<f64>() / pct_returns.len() as f64;
            let annualized_return = mean_return * periods_per_day as f64 * 365.0;
            result.calmar_ratio = risk::calmar_ratio(annualized_return, max_dd);
        }
        
        // Record risk halt reason if trading was stopped
        result.risk_halt_reason = halt_reason;
            
        // Debug: Log portfolio state before calculating PnL
        log_info!(BACKTEST_LOGGER, "Portfolio state: balance={} positions={} open_positions={}",
            portfolio.balance,
            portfolio.positions.len(),
            portfolio.positions.iter().filter(|p| p.close_time.is_none()).count()
        );
        
        for (i, pos) in portfolio.positions.iter().enumerate() {
            let mark = pos.mark_price.unwrap_or(0.0);
            log_info!(BACKTEST_LOGGER, "Position {}: side={:?} qty={} entry={} mark={} closed={}",
                i,
                pos.side,
                pos.quantity,
                pos.entry_price,
                mark,
                pos.close_time.is_some()
            );
        }
            
        // Calculate total P&L: ending portfolio value - initial capital
        // Use get_total_value() which correctly includes position market values
        let final_portfolio_value = portfolio.get_total_value();
        
        log_info!(BACKTEST_LOGGER, "PnL calc: final_value={} initial={} pnl={}",
            final_portfolio_value,
            config.trading.initial_capital,
            final_portfolio_value - config.trading.initial_capital
        );
        
        result.total_pnl = final_portfolio_value - config.trading.initial_capital;

        // Fills-only metrics (after total_pnl is set).
        // NOTE: win_rate is NOT estimated here. It is set below from the
        // MEASURED round-trip win rate (RoundTripCollector data); if no
        // round-trips completed it stays None (undefined) — previously a
        // fabricated estimate derived from net PnL was reported as if measured.
        if is_fills_only_mode {
            let total_pnl_f64 = result.total_pnl;

            // Calculate fills-only specific metrics
            // gross_profit = spread captured (pnl + commission), gross_loss = commission paid
            let gross_pnl = total_pnl_f64 + total_commission; // Total spread captured before fees
            if gross_pnl > 0.0 {
                result.gross_profit = Some(gross_pnl);
                result.gross_loss = Some(total_commission);
            } else {
                // Lost money even before commission - adverse selection
                result.gross_profit = Some(0.0);
                result.gross_loss = Some(-total_pnl_f64); // Total loss
            }
            result.net_profit = Some(total_pnl_f64);
            
            // Avg return per round-trip
            if fill_round_trips > 0 {
                result.avg_trade_return = Some(total_pnl_f64 / fill_round_trips as f64);
            }
            
            // Profit factor: gross_profit / gross_loss
            if total_commission > 0.0 && gross_pnl > 0.0 {
                result.profit_factor = Some(gross_pnl / total_commission);
            }
            
            log_info!(BACKTEST_LOGGER, "Fills-only metrics: gross_profit={:.2} gross_loss={:.2} net_profit={:.2} avg_return_per_rt={:.2} profit_factor={:.2}",
                result.gross_profit.unwrap_or(0.0), result.gross_loss.unwrap_or(0.0), 
                result.net_profit.unwrap_or(0.0), result.avg_trade_return.unwrap_or(0.0),
                result.profit_factor.unwrap_or(0.0));
                
            log_info!(BACKTEST_LOGGER, "Final MM result: win_rate={:?} num_trades={}",
                result.win_rate, result.num_trades);
        }

        // Include equity curve data for visualization
        result.equity_curve = equity_curve;
        // Timestamps parallel to the equity curve (Bug #33). The curve's first
        // point (initial capital) predates the first sample, so reuse the first
        // sample's timestamp for it.
        if !equity_curve_timestamps.is_empty() {
            let first_ts = equity_curve_timestamps[0];
            equity_curve_timestamps.insert(0, first_ts);
        }
        if equity_curve_timestamps.len() == result.equity_curve.len() {
            result.equity_curve_timestamps = equity_curve_timestamps;
        }
        // Store individual trade returns for Monte Carlo simulation (actual trade-level P&L)
        result.trade_returns = trade_returns;
        result.initial_capital = config.trading.initial_capital;
        
        // NOTE: transaction_costs and market_impact are populated by
        // TransactionCostCollector::finalize() via collectors::finalize_all()
        // at the end of this method. No inline write needed.
        
        // NOTE: inventory_metrics, execution_metrics, round-trip metrics are all
        // populated by their respective collectors via finalize_all().
        // The dead metric calculations that used to be here have been removed.
        
        // Execution metrics from collectors for diagnostic logging.
        // Uses lifetime counters — trackers are pruned when orders complete.
        let orders_placed = bid_orders_placed + ask_orders_placed;
        let orders_filled = self.orders_filled_count;
        
        let fill_rate = if orders_placed > 0 { orders_filled as f64 / orders_placed as f64 } else { 0.0 };
        let cancellation_rate = if orders_placed > 0 { orders_cancelled as f64 / orders_placed as f64 } else { 0.0 };
        
        // Time to fill statistics from collector
        let time_to_fill_samples = exec.map(|c| &c.time_to_fill_samples[..]).unwrap_or(&[]);
        let avg_time_to_fill = if !time_to_fill_samples.is_empty() {
            time_to_fill_samples.iter().sum::<f64>() / time_to_fill_samples.len() as f64
        } else { 0.0 };
        
        let median_time_to_fill = if !time_to_fill_samples.is_empty() {
            let mut sorted = time_to_fill_samples.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 && sorted.len() > 1 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        } else { 0.0 };
        
        // Bid/ask fill rates (lifetime counters — see track_fill)
        let bid_fills_count = self.bid_orders_filled_count;
        let ask_fills_count = self.ask_orders_filled_count;
        let bid_fill_rate = if bid_orders_placed > 0 { bid_fills_count as f64 / bid_orders_placed as f64 } else { 0.0 };
        let ask_fill_rate = if ask_orders_placed > 0 { ask_fills_count as f64 / ask_orders_placed as f64 } else { 0.0 };
        
        // Average quote spread (from tracked spreads — stays on struct)
        let avg_quote_spread_bps = if !self.quote_spreads_bps.is_empty() {
            self.quote_spreads_bps.iter().sum::<f64>() / self.quote_spreads_bps.len() as f64
        } else { 0.0 };
        
        // Round-trip diagnostics from collector
        let rt_buy_prices = rt.map(|c| &c.buy_fill_prices[..]).unwrap_or(&[]);
        let rt_sell_prices = rt.map(|c| &c.sell_fill_prices[..]).unwrap_or(&[]);
        let total_round_trips = rt_buy_prices.len().min(rt_sell_prices.len());
        let mut profitable_round_trips = 0_usize;
        let mut total_round_trip_pnl = 0.0_f64;
        let rt_fee_rate = rt.map(|c| c.fee_rate).unwrap_or(config.trading.commission_rate);
        
        for i in 0..total_round_trips {
            let buy_price = rt_buy_prices[i];
            let sell_price = rt_sell_prices[i];
            let net_pnl = (sell_price - buy_price) - rt_fee_rate * (buy_price + sell_price);
            total_round_trip_pnl += net_pnl;
            if net_pnl > 0.0 { profitable_round_trips += 1; }
        }
        
        let round_trip_win_rate = if total_round_trips > 0 {
            Some(profitable_round_trips as f64 / total_round_trips as f64)
        } else { None };

        // Fills-only mode: report the measured round-trip win rate as the
        // top-level win rate (None if no round-trips completed).
        if is_fills_only_mode && result.win_rate.is_none() {
            result.win_rate = round_trip_win_rate;
        }
        let avg_pnl_per_round_trip = if total_round_trips > 0 {
            Some(total_round_trip_pnl / total_round_trips as f64)
        } else { None };
        let spread_capture_rate = if total_round_trips > 0 && avg_quote_spread_bps > 0.0 {
            let avg_price = (rt_buy_prices.iter().sum::<f64>() + rt_sell_prices.iter().sum::<f64>())
                / (rt_buy_prices.len() + rt_sell_prices.len()) as f64;
            let avg_spread_captured_bps = if avg_price > 0.0 {
                (total_round_trip_pnl / total_round_trips as f64) / avg_price * 10000.0
            } else { 0.0 };
            Some((avg_spread_captured_bps / avg_quote_spread_bps).clamp(0.0, 2.0))
        } else { None };
        let adverse_selection_rate = if total_round_trips > 0 {
            Some((total_round_trips - profitable_round_trips) as f64 / total_round_trips as f64)
        } else { None };
        
        log_info!(BACKTEST_LOGGER, "Round-trip metrics | total={} profitable={} win_rate={:.1}% avg_pnl=${:.4} spread_capture={:.1}% adverse_sel={:.1}%",
            total_round_trips, profitable_round_trips, 
            round_trip_win_rate.unwrap_or(0.0) * 100.0,
            avg_pnl_per_round_trip.unwrap_or(0.0),
            spread_capture_rate.unwrap_or(0.0) * 100.0,
            adverse_selection_rate.unwrap_or(0.0) * 100.0);
        
        // Log execution metrics summary
        log_info!(BACKTEST_LOGGER, "Execution metrics | orders_placed={} filled={} cancelled={} fill_rate={:.1}% cancel_rate={:.1}%",
            orders_placed, orders_filled, orders_cancelled, fill_rate * 100.0, cancellation_rate * 100.0);
        log_info!(BACKTEST_LOGGER, "Execution metrics | avg_time_to_fill={:.3}s median={:.3}s bid_fill_rate={:.1}% ask_fill_rate={:.1}%",
            avg_time_to_fill, median_time_to_fill, bid_fill_rate * 100.0, ask_fill_rate * 100.0);
        
        // Always log latency simulation settings and rejection stats
        let total_potential = orders_filled + latency_rejections;
        let rejection_pct = if total_potential > 0 { 
            latency_rejections as f64 / total_potential as f64 * 100.0 
        } else { 
            0.0 
        };
        log_info!(BACKTEST_LOGGER, "Latency simulation | latency_ms={} rejections={} (bid={} ask={}) rejection_rate={:.1}%",
            config.trading.latency_ms, latency_rejections, 
            latency_rejections_bid, latency_rejections_ask, rejection_pct);
        
        // Log average quoted spread - key diagnostic for fill rate
        log_info!(BACKTEST_LOGGER, "Spread analysis | avg_quoted_spread_bps={:.2} samples={} | wider spreads = fewer fills but higher profit/fill",
            avg_quote_spread_bps, self.quote_spreads_bps.len());
        
        // Log market data trade direction distribution
        let total_market_trades = market_buy_trades + market_sell_trades;
        let buy_pct = if total_market_trades > 0 { market_buy_trades as f64 / total_market_trades as f64 * 100.0 } else { 0.0 };
        let sell_pct = if total_market_trades > 0 { market_sell_trades as f64 / total_market_trades as f64 * 100.0 } else { 0.0 };
        log_info!(BACKTEST_LOGGER, "Market trades | total={} buys={} ({:.1}%) sells={} ({:.1}%) | buys->fill asks, sells->fill bids",
            total_market_trades, market_buy_trades, buy_pct, market_sell_trades, sell_pct);
        
        // Log missed opportunity analysis - KEY DIAGNOSTIC for fill rate issues
        let avg_bid_gap_diag = if missed_bid_opportunities > 0 { 
            bid_price_gap_sum / missed_bid_opportunities as f64 
        } else { 0.0 };
        let avg_ask_gap_diag = if missed_ask_opportunities > 0 { 
            ask_price_gap_sum / missed_ask_opportunities as f64 
        } else { 0.0 };
        log_info!(BACKTEST_LOGGER, 
            "MISSED_FILLS | bid_opportunities_missed={} avg_gap=${:.4} | ask_opportunities_missed={} avg_gap=${:.4}",
            missed_bid_opportunities, avg_bid_gap_diag, 
            missed_ask_opportunities, avg_ask_gap_diag);
        
        // Calculate what percentage of sell trades we could have captured if our bids were higher
        let potential_bid_fills = market_sell_trades;
        let missed_bid_pct = if potential_bid_fills > 0 {
            missed_bid_opportunities as f64 / potential_bid_fills as f64 * 100.0
        } else { 0.0 };
        let captured_bid_fills = result.execution_metrics.as_ref().map(|m| m.bid_fills).unwrap_or(0);
        let captured_bid_pct = if potential_bid_fills > 0 {
            captured_bid_fills as f64 / potential_bid_fills as f64 * 100.0
        } else { 0.0 };
        log_info!(BACKTEST_LOGGER,
            "BID_FILL_RATE | sell_trades={} captured={} ({:.2}%) missed={} ({:.2}%) | avg_gap=${:.4} = how much higher bid needs to be",
            potential_bid_fills, captured_bid_fills, captured_bid_pct, 
            missed_bid_opportunities, missed_bid_pct, avg_bid_gap_diag);
        
        // Log data date range and market trend from TimestampCollector
        let first_trade_timestamp = ts.and_then(|c| c.first_trade_timestamp);
        let last_trade_timestamp = ts.and_then(|c| c.last_trade_timestamp);
        let first_trade_price = ts.and_then(|c| c.first_trade_price);
        let last_trade_price = ts.and_then(|c| c.last_trade_price);
        
        if let (Some(start_ts), Some(end_ts)) = (first_trade_timestamp, last_trade_timestamp) {
            let duration = end_ts - start_ts;
            let hours = duration.num_hours();
            let minutes = duration.num_minutes() % 60;
            log_info!(BACKTEST_LOGGER, "Data range | start={} end={} duration={}h {}m",
                start_ts.format("%Y-%m-%d %H:%M:%S UTC"), 
                end_ts.format("%Y-%m-%d %H:%M:%S UTC"),
                hours, minutes);
            
            // Log market trend (price change during backtest)
            if let (Some(start_price), Some(end_price)) = (first_trade_price, last_trade_price) {
                let price_change = end_price - start_price;
                let price_change_pct = (price_change / start_price) * 100.0;
                let trend = if price_change_pct > 1.0 { "UPTREND" } 
                           else if price_change_pct < -1.0 { "DOWNTREND" } 
                           else { "SIDEWAYS" };
                log_info!(BACKTEST_LOGGER, "Market trend | start_price=${:.2} end_price=${:.2} change={:.2}% ({}) | {} bids hit more, {} asks hit more",
                    start_price, end_price, price_change_pct, trend,
                    if price_change_pct > 0.0 { "uptrend" } else { "downtrend" },
                    if price_change_pct > 0.0 { "downtrend" } else { "uptrend" });
            }
        } else {
            log_info!(BACKTEST_LOGGER, "Data range | No trades processed in market data");
        }
        
        // Log inventory metrics summary (from collector)
        let total_volume = buy_volume + sell_volume;
        let final_inventory = buy_qty - sell_qty;
        log_info!(BACKTEST_LOGGER, "Inventory position | final_qty={:.6} max_long={:.6} max_short={:.6} | asymmetric if not near zero",
            final_inventory, max_inventory, min_inventory);
        log_info!(BACKTEST_LOGGER, "Inventory metrics | total_volume=${:.2} buy=${:.2} sell=${:.2}",
            total_volume, buy_volume, sell_volume);
        
        // Log fee tier progression for debugging commission issues
        for (exchange, volume) in &self.current_30d_volume_by_exchange {
            let fee_rate_val = if let Some(ref fee_config) = config.trading.exchange_fees {
                fee_config.get_fee_rate(*volume, true) // maker rate
            } else {
                config.trading.commission_rate
            };
            log_info!(BACKTEST_LOGGER, "Fee tier summary | exchange={} final_30d_volume=${:.0} final_maker_fee_bps={:.1} total_commission=${:.2}",
                exchange, volume, fee_rate_val * 10000.0, total_commission);
        }
        
        // Log commission breakdown by fee tier (from collector)
        if let Some(tc_ref) = tc {
            let mut tier_breakdown: Vec<_> = tc_ref.commission_by_tier_bps.iter().collect();
            tier_breakdown.sort_by_key(|(tier, _)| **tier);
            for (tier_bps, commission) in &tier_breakdown {
                let fills = tc_ref.fills_by_tier_bps.get(tier_bps).unwrap_or(&0);
                let avg_per_fill = if *fills > 0 { *commission / *fills as f64 } else { 0.0 };
                log_info!(BACKTEST_LOGGER, "Commission by tier | fee_bps={} fills={} commission=${:.2} avg_per_fill=${:.2}",
                    tier_bps, fills, commission, avg_per_fill);
            }
        }
        
        // Log fill count distribution (from collector)
        if let Some(exec_ref) = exec {
            let mut fill_dist: Vec<_> = exec_ref.fill_count_histogram.iter().collect();
            fill_dist.sort_by_key(|(count, _)| **count);
            let fill_dist_str: String = fill_dist.iter()
                .map(|(fills, orders)| format!("{}fills:{}orders", fills, orders))
                .collect::<Vec<_>>()
                .join(", ");
            log_info!(BACKTEST_LOGGER, "Fill distribution | total_orders={} total_fills={} avg_fills_per_order={:.2} distribution=[{}]",
                total_order_count, total_fill_count, 
                if total_order_count > 0 { total_fill_count as f64 / total_order_count as f64 } else { 0.0 },
                fill_dist_str);
        }

        // Diagnostic: Calculate average fill value from commission data (from collector)
        if let Some(tc_ref) = tc {
            let total_fills_diag = tc_ref.fills_by_tier_bps.values().sum::<u32>();
            log_info!(BACKTEST_LOGGER, "Order size diagnostic START | total_fills={} total_commission=${:.2} fills_by_tier_count={}",
                total_fills_diag, total_commission, tc_ref.fills_by_tier_bps.len());
            
            if total_fills_diag > 0 {
                let fee_rate_diag = if let Some((&tier_bps, _)) = tc_ref.fills_by_tier_bps.iter().max_by_key(|(_, count)| *count) {
                    if tier_bps > 0 {
                        tier_bps as f64 / 10000.0
                    } else {
                        config.trading.commission_rate.max(0.001)
                    }
                } else {
                    config.trading.commission_rate.max(0.001)
                };
                
                let estimated_total_fill_value = if total_commission > 0.0 && fee_rate_diag > 0.0 {
                    total_commission / fee_rate_diag
                } else { 0.0 };
                let avg_fill_value = if total_fills_diag > 0 { estimated_total_fill_value / total_fills_diag as f64 } else { 0.0 };
                let expected_order_value = config.trading.initial_capital * 0.02;
                log_info!(BACKTEST_LOGGER, 
                    "Order size diagnostic | total_fills={} commission=${:.2} fee_rate={:.4}% => estimated_fill_value=${:.2}/fill (expected ~${:.2} at 2% order_size_pct)",
                    total_fills_diag, total_commission, fee_rate_diag * 100.0, avg_fill_value, expected_order_value);
            }
        }
        
        // Log performance optimization stats
        let skip_pct = if processed_count > 0 {
            (self.stable_region_skips as f64 / processed_count as f64) * 100.0
        } else { 0.0 };
        log_info!(BACKTEST_LOGGER, "PERF_OPT | stable_region_skips={} ({:.1}% of {} ticks) | strategy calls saved",
            self.stable_region_skips, skip_pct, processed_count);

        // Run pluggable collectors to populate result sections
        collectors::finalize_all(&self.collectors, &mut result);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_loop_new() {
        let sim = SimulationLoop::new();
        assert!(sim.trade_history_by_exchange.is_empty());
        assert!(sim.current_30d_volume_by_exchange.is_empty());
        assert!(sim.active_order_ids.is_empty());
        assert!(sim.market_volume_window.is_empty());
        assert_eq!(sim.rolling_avg_volume, 0.0);
        assert!(sim.order_trackers.is_empty());
        assert!(sim.quote_spreads_bps.is_empty());
        assert_eq!(sim.stable_region_skips, 0);
        assert!(!sim.collectors.is_empty());
    }

    #[test]
    fn test_simulation_loop_new_lean() {
        let sim = SimulationLoop::new_lean();
        assert!(!sim.collectors.is_empty());
    }

    #[test]
    fn test_simulation_loop_new_full() {
        let sim = SimulationLoop::new_full(0.001);
        assert!(!sim.collectors.is_empty());
    }

    #[test]
    fn test_get_avg_volume_default() {
        let sim = SimulationLoop::new();
        // Default fallback should return 100_000.0 when no data
        assert_eq!(sim.get_avg_volume(), 100_000.0);
    }

    #[test]
    fn test_get_avg_volume_with_data() {
        let mut sim = SimulationLoop::new();
        sim.rolling_avg_volume = 50_000.0;
        assert_eq!(sim.get_avg_volume(), 50_000.0);
    }

    #[test]
    fn test_update_market_volume() {
        let mut sim = SimulationLoop::new();
        let ts = chrono::Utc::now();
        sim.update_market_volume(ts, 1000.0);
        assert_eq!(sim.market_volume_window.len(), 1);
        assert!(sim.rolling_avg_volume > 0.0);
    }

    #[test]
    fn test_update_market_volume_window_retention() {
        let mut sim = SimulationLoop::new();
        let ts = chrono::Utc::now();
        // Add two volumes at current time - both should be retained
        sim.update_market_volume(ts, 1000.0);
        sim.update_market_volume(ts, 2000.0);
        assert_eq!(sim.market_volume_window.len(), 2);
        // rolling_avg_volume = total / 60 minutes
        assert!((sim.rolling_avg_volume - 3000.0 / 60.0).abs() < 0.01);
    }

    #[test]
    fn test_update_30d_volume() {
        let mut sim = SimulationLoop::new();
        let ts = chrono::Utc::now();
        sim.update_30d_volume("massive", ts, 50_000.0);
        assert_eq!(*sim.current_30d_volume_by_exchange.get("massive").unwrap(), 50_000.0);
        sim.update_30d_volume("massive", ts, 30_000.0);
        assert_eq!(*sim.current_30d_volume_by_exchange.get("massive").unwrap(), 80_000.0);
    }

    #[test]
    fn test_get_fee_rate_fallback() {
        let sim = SimulationLoop::new();
        let config = config::BacktestConfig::default();
        let fee = sim.get_fee_rate(&config, "massive", true);
        // Should return something reasonable (either from exchange config or commission_rate)
        assert!(fee >= 0.0);
    }

    #[test]
    fn test_initialize_starting_volume() {
        let mut sim = SimulationLoop::new();
        let mut config = config::BacktestConfig::default();
        // With no exchange_fees configured, should be a no-op
        sim.initialize_starting_volume(&config);
        assert!(sim.current_30d_volume_by_exchange.is_empty());
    }

    #[test]
    fn test_calculate_market_impact_disabled() {
        let mut sim = SimulationLoop::new();
        let mut config = config::BacktestConfig::default();
        config.data.market_impact.enabled = false;
        let ts = chrono::Utc::now();
        let impact = sim.calculate_market_impact(100.0, 50000.0, 100_000.0, &config, ts);
        assert_eq!(impact, 0.0);
    }

    #[test]
    fn test_calculate_market_impact_zero_volume() {
        let mut sim = SimulationLoop::new();
        let config = config::BacktestConfig::default();
        let ts = chrono::Utc::now();
        let impact = sim.calculate_market_impact(100.0, 50000.0, 0.0, &config, ts);
        assert_eq!(impact, 0.0);
    }

    #[test]
    fn test_calculate_market_impact_delegates_to_config_impact_bps() {
        // Regression guard for the impact_bps() extraction: the method's
        // observable output must still equal the same formula, just
        // computed through MarketImpactConfig::impact_bps now.
        let mut sim = SimulationLoop::new();
        let config = config::BacktestConfig::default();
        let ts = chrono::Utc::now();
        let order_size = 5000.0;
        let price = 50000.0;
        let avg_volume = 100_000.0;

        let impact = sim.calculate_market_impact(order_size, price, avg_volume, &config, ts);

        let expected_bps = config.data.market_impact
            .impact_bps(order_size, avg_volume)
            .unwrap_or(0.0);
        let expected = (expected_bps / 10000.0) * price * order_size;
        assert!((impact - expected).abs() < 1e-9);
        assert!(impact > 0.0, "sanity check: this order size should be large enough to trigger impact");
    }

    #[test]
    fn test_active_order_ids_tracking() {
        let mut sim = SimulationLoop::new();
        let id: Arc<str> = Arc::from("order-1");
        sim.active_order_ids.insert(id.clone());
        assert!(sim.active_order_ids.contains(&id));
        sim.active_order_ids.remove(&id);
        assert!(!sim.active_order_ids.contains(&id));
    }

    #[test]
    fn test_order_tracker_struct() {
        let tracker = OrderTracker {
            order_id: Arc::from("test-order"),
            side: orderbook::BookSide::Bid,
            placed_at: chrono::Utc::now(),
            price: 50000.0,
            filled: false,
            fill_count: 0,
            first_fill_at: None,
        };
        assert!(!tracker.filled);
        assert_eq!(tracker.fill_count, 0);
        assert!(tracker.first_fill_at.is_none());
    }

    #[test]
    fn test_track_fill() {
        let mut sim = SimulationLoop::new();
        let id: Arc<str> = Arc::from("order-42");
        let now = chrono::Utc::now();
        sim.order_trackers.insert(id.clone(), OrderTracker {
            order_id: id.clone(),
            side: orderbook::BookSide::Bid,
            placed_at: now,
            price: 100.0,
            filled: false,
            fill_count: 0,
            first_fill_at: None,
        });
        sim.track_fill("order-42", now);
        let tracker = sim.order_trackers.get("order-42").unwrap();
        assert!(tracker.filled);
        assert_eq!(tracker.fill_count, 1);
        assert!(tracker.first_fill_at.is_some());
    }

    #[test]
    fn test_track_fill_multiple() {
        let mut sim = SimulationLoop::new();
        let id: Arc<str> = Arc::from("order-99");
        let now = chrono::Utc::now();
        sim.order_trackers.insert(id.clone(), OrderTracker {
            order_id: id.clone(),
            side: orderbook::BookSide::Ask,
            placed_at: now,
            price: 200.0,
            filled: false,
            fill_count: 0,
            first_fill_at: None,
        });
        sim.track_fill("order-99", now);
        sim.track_fill("order-99", now);
        let tracker = sim.order_trackers.get("order-99").unwrap();
        assert!(tracker.filled);
        assert_eq!(tracker.fill_count, 2);
    }

    #[test]
    fn test_track_fill_nonexistent_order() {
        let mut sim = SimulationLoop::new();
        let now = chrono::Utc::now();
        // Should not panic
        sim.track_fill("nonexistent", now);
    }

    #[test]
    fn test_quote_spreads_tracking() {
        let mut sim = SimulationLoop::new();
        sim.quote_spreads_bps.push(5.0);
        sim.quote_spreads_bps.push(10.0);
        assert_eq!(sim.quote_spreads_bps.len(), 2);
        assert_eq!(*sim.quote_spreads_bps.last().unwrap(), 10.0);
    }
}

impl SimulationLoop {
    /// Run simulation using a streaming TickSource (memory-efficient)
    ///
    /// Converts ticks to `MarketData::Trade` with zero-copy f64 fields
    /// (uses `Default::default()` for legacy BD fields since `run()` only reads
    /// the pre-computed f64 fields). Uses empty strings for symbol/exchange
    /// (the simulation reads these from `OrderBook` config, not from trade data).
    ///
    /// Memory: ~48 bytes/tick vs ~400 bytes/tick in the old implementation.
    ///
    /// # Future optimization
    /// The event loop should be refactored to accept `&[SimulationTick]` directly,
    /// eliminating this intermediate `Vec<MarketData>` materialization entirely.
    pub async fn run_streaming<S: dataloader::TickSource>(
        &mut self,
        strategy_manager: &mut StrategyManager,
        orderbook: &mut OrderBook,
        portfolio: &mut PortfolioState,
        risk_manager: &mut RiskManager,
        source: &S,
        _symbol: &str,
        _exchange: &str,
        config: &config::BacktestConfig,
    ) -> BacktestResult {
        use dataloader::Tick;
        
        let total_items = source.len();
        
        // Symbol/exchange are never read from trade data — simulation uses OrderBook config.
        
        let market_data: Vec<MarketData> = (0..total_items)
            .filter_map(|i| {
                source.get(i).map(|tick| {
                    let price = tick.price();
                    let quantity = tick.quantity();
                    let side_val = tick.side();
                    
                    MarketData::Trade(dataloader::TradeData {
                        timestamp: tick.timestamp(),
                        symbol: "".into(),      // Not read by simulation (uses OrderBook config)
                        exchange: "".into(),     // Not read by simulation (uses OrderBook config)
                        price,
                        quantity,
                        side: side_val,
                        trade_id: 0,
                        value: price * quantity,
                    })
                })
            })
            .collect();
        
        log_info!(BACKTEST_LOGGER, "STREAMING_CONVERTED | {} ticks → {} MarketData",
            total_items, market_data.len());
        
        self.run(
            strategy_manager,
            orderbook,
            portfolio,
            risk_manager,
            &market_data,
            config,
        ).await
    }
}
