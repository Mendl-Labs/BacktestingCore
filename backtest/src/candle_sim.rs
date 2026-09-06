//! Lightweight candle-based position simulator for GA fitness evaluation.
//!
//! Takes a signal array (1=BUY, -1=SELL, 0=HOLD, 2=CLOSE) and OHLCV data,
//! simulates positions with commission/slippage, and returns metrics.
//! Designed for ~0.1ms evaluation of 10K candles.

use std::sync::Arc;
use serde::{Serialize, Deserialize};
use config::SlippageModel;

/// Unrealized P&L for a position sized `qty` (positive = long, negative =
/// short, zero = flat) entered at `entry_price`, marked at `mark_price`.
#[inline]
fn unrealized_pnl(qty: f64, entry_price: f64, mark_price: f64) -> f64 {
    if qty > 0.0 {
        (mark_price - entry_price) * qty
    } else if qty < 0.0 {
        (entry_price - mark_price) * qty.abs()
    } else {
        0.0
    }
}

/// Build a RealismConfig from trading config + optional exchange fee config.
pub fn build_realism(trading: &config::TradingConfig, fee: &Option<config::ExchangeFeeConfig>) -> RealismConfig {
    RealismConfig {
        intrabar_path: true,
        market_impact_k: 0.3,
        spread_from_range: true,
        spread_cap_bps: trading.synthetic_spread_bps,
        adverse_selection: trading.adverse_selection,
        adverse_selection_max_bps: 20.0,
        latency_mean_ms: fee.as_ref().map(|f| f.latency_mean_ms).unwrap_or(0.0),
        latency_std_ms: fee.as_ref().map(|f| f.latency_std_ms).unwrap_or(0.0),
        fee_tiers: fee.as_ref().map(|f| f.volume_tiers.clone()),
        starting_30d_volume: fee.as_ref().map(|f| f.starting_30d_volume).unwrap_or(0.0),
    }
}

/// Configuration for the candle simulator.
#[derive(Debug, Clone)]
pub struct CandleSimulator {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage_model: SlippageModel,
    pub fill_volume_fraction: f64,
    pub funding_rate_8h: f64,
    pub funding_interval_ms: i64,
    pub realism: RealismConfig,
    /// When true (default), track equity at every bar for curve display.
    /// Set false for GA fitness evaluations that only need scalar metrics — saves
    /// ~40 KB of Vec allocations per chromosome and avoids serialisation overhead.
    pub track_equity_curve: bool,
    /// Leverage multiplier for margined positions. `1.0` (default) = no
    /// leverage -- behavior-identical to this simulator before leverage
    /// existed (see `portfoliomanager::margin`'s module docs).
    pub leverage: f64,
    /// Flat maintenance-margin ratio used to compute the liquidation
    /// trigger for leveraged positions (isolated margin, v1).
    pub maintenance_margin_ratio: f64,
    /// Extra adverse slippage (bps) applied to a liquidation's exit price
    /// on top of the theoretical trigger price.
    pub liquidation_penalty_bps: f64,
    /// Volatility-targeting position-size overlay (opt-in; `None` -- the
    /// default -- is fully behavior-preserving). See
    /// `portfoliomanager::margin::VolTargetConfig`'s doc comment.
    pub vol_target: Option<portfoliomanager::margin::VolTargetConfig>,
}

/// OHLCV candle arrays extracted from market data (shared across GA candidates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleArrays {
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
    pub timestamps_ms: Vec<i64>,
}

impl CandleArrays {
    pub fn from_market_data(data: &[dataloader::MarketData]) -> Self {
        let mut opens = Vec::with_capacity(data.len());
        let mut highs = Vec::with_capacity(data.len());
        let mut lows = Vec::with_capacity(data.len());
        let mut closes = Vec::with_capacity(data.len());
        let mut volumes = Vec::with_capacity(data.len());
        let mut timestamps_ms = Vec::with_capacity(data.len());

        for md in data {
            match md {
                dataloader::MarketData::Candle(c) => {
                    opens.push(c.open);
                    highs.push(c.high);
                    lows.push(c.low);
                    closes.push(c.close);
                    volumes.push(c.volume);
                    timestamps_ms.push(c.timestamp.timestamp_millis());
                }
                dataloader::MarketData::Trade(t) => {
                    opens.push(t.price);
                    highs.push(t.price);
                    lows.push(t.price);
                    closes.push(t.price);
                    volumes.push(t.quantity);
                    timestamps_ms.push(t.timestamp.timestamp_millis());
                }
                dataloader::MarketData::Generic(g) => {
                    opens.push(g.price);
                    highs.push(g.price);
                    lows.push(g.price);
                    closes.push(g.price);
                    volumes.push(g.quantity);
                    timestamps_ms.push(g.timestamp_ms);
                }
                dataloader::MarketData::PoolSwap(s) => {
                    let p = s.price();
                    opens.push(p);
                    highs.push(p);
                    lows.push(p);
                    closes.push(p);
                    volumes.push(s.amount_in);
                    timestamps_ms.push(s.timestamp.timestamp_millis());
                }
                dataloader::MarketData::OptionCandle(c) => {
                    // Structurally identical OHLCV to a plain Candle -- this
                    // lets option-contract bars flow through the same
                    // CandleArrays the rest of the simulation engine uses.
                    // Strike/expiration/contract_type aren't carried here;
                    // the options P&L engine (a later phase) reads those
                    // from the contract metadata directly, not from this
                    // array.
                    opens.push(c.open);
                    highs.push(c.high);
                    lows.push(c.low);
                    closes.push(c.close);
                    volumes.push(c.volume);
                    timestamps_ms.push(c.timestamp.timestamp_millis());
                }
            }
        }

        Self { opens, highs, lows, closes, volumes, timestamps_ms }
    }

    pub fn len(&self) -> usize {
        self.closes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.closes.is_empty()
    }
}

/// Advanced realism configuration — all features disabled by default for zero behavior change.
#[derive(Debug, Clone)]
pub struct RealismConfig {
    pub intrabar_path: bool,
    pub market_impact_k: f64,
    pub spread_from_range: bool,
    pub spread_cap_bps: f64,
    pub adverse_selection: bool,
    pub adverse_selection_max_bps: f64,
    pub latency_mean_ms: f64,
    pub latency_std_ms: f64,
    pub fee_tiers: Option<Vec<config::VolumeTier>>,
    pub starting_30d_volume: f64,
}

impl Default for RealismConfig {
    fn default() -> Self {
        Self {
            intrabar_path: false,
            market_impact_k: 0.0,
            spread_from_range: false,
            spread_cap_bps: 10.0,
            adverse_selection: false,
            adverse_selection_max_bps: 20.0,
            latency_mean_ms: 0.0,
            latency_std_ms: 0.0,
            fee_tiers: None,
            starting_30d_volume: 0.0,
        }
    }
}

/// Configuration for early termination of bad chromosomes.
#[derive(Debug, Clone)]
pub struct EarlyTermConfig {
    pub max_loss_pct: f64,
    pub check_interval: usize,
}

impl Default for EarlyTermConfig {
    fn default() -> Self {
        Self { max_loss_pct: 0.30, check_interval: 500 }
    }
}

/// Result of a candle simulation run.
#[derive(Debug, Clone)]
pub struct CandleSimResult {
    pub equity_curve: Vec<f64>,
    pub total_pnl: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
    pub trade_returns: Vec<f64>,
    pub total_commission: f64,
    pub total_funding_paid: f64,
    pub volume_constrained_skips: usize,
    pub early_terminated: bool,
    pub total_spread_cost: f64,
    pub total_market_impact_cost: f64,
    pub total_adverse_selection_cost: f64,
    pub latency_rejected_signals: usize,
    pub fee_tier_changes: usize,
    /// Number of times a leveraged position was force-closed by a margin
    /// call. Always `0` at `leverage=1.0`.
    pub total_liquidations: usize,
}

/// Signal codes matching Python convention.
pub const SIGNAL_BUY: i8 = 1;
pub const SIGNAL_SELL: i8 = -1;
pub const SIGNAL_HOLD: i8 = 0;
pub const SIGNAL_CLOSE: i8 = 2;

impl CandleSimulator {
    pub fn new(initial_capital: f64, commission_rate: f64, slippage_model: SlippageModel) -> Self {
        Self {
            initial_capital,
            commission_rate,
            slippage_model,
            fill_volume_fraction: 0.10,
            funding_rate_8h: 0.0,
            funding_interval_ms: 8 * 60 * 60 * 1000,
            realism: RealismConfig::default(),
            track_equity_curve: true,
            leverage: 1.0,
            maintenance_margin_ratio: 0.005,
            liquidation_penalty_bps: 50.0,
            vol_target: None,
        }
    }

    pub fn with_skip_equity_curve(mut self) -> Self {
        self.track_equity_curve = false;
        self
    }

    pub fn with_fill_volume_fraction(mut self, frac: f64) -> Self {
        self.fill_volume_fraction = frac;
        self
    }

    pub fn with_funding_rate(mut self, rate_8h: f64, interval_ms: i64) -> Self {
        self.funding_rate_8h = rate_8h;
        self.funding_interval_ms = interval_ms;
        self
    }

    pub fn with_realism(mut self, realism: RealismConfig) -> Self {
        self.realism = realism;
        self
    }

    pub fn with_leverage(mut self, leverage: f64, maintenance_margin_ratio: f64, liquidation_penalty_bps: f64) -> Self {
        self.leverage = leverage.max(1.0);
        self.maintenance_margin_ratio = maintenance_margin_ratio;
        self.liquidation_penalty_bps = liquidation_penalty_bps;
        self
    }

    /// Enable the volatility-targeting position-size overlay (opt-in; not
    /// calling this at all is fully behavior-preserving). See
    /// `portfoliomanager::margin::VolTargetConfig`'s doc comment for what
    /// this scales and why.
    pub fn with_vol_target(mut self, cfg: portfoliomanager::margin::VolTargetConfig) -> Self {
        self.vol_target = Some(cfg);
        self
    }

    /// Run position simulation from a signal array.
    ///
    /// Fills execute at the next bar's open price (with volume-adjusted slippage).
    /// Stop-loss/take-profit are checked against each bar's high/low with gap-through detection.
    pub fn run(
        &self,
        signals: &[i8],
        candles: &CandleArrays,
        stop_loss_pct: Option<f64>,
        take_profit_pct: Option<f64>,
        position_size_pct: f64,
        early_term: Option<&EarlyTermConfig>,
    ) -> CandleSimResult {
        let n = signals.len().min(candles.len());
        if n == 0 {
            return CandleSimResult::empty(self.initial_capital);
        }

        let avg_volume = candles.volumes.iter().sum::<f64>() / n as f64;

        // Volatility-targeting overlay (opt-in): precompute the whole
        // trailing realized-vol series once up front rather than
        // recomputing a rolling window inside the hot per-bar loop below.
        let vol_target_series: Option<Vec<Option<f64>>> = self
            .vol_target
            .as_ref()
            .map(|cfg| portfoliomanager::margin::trailing_realized_vol_series(&candles.closes, cfg.lookback_bars));

        let mut cash = self.initial_capital;
        let mut position_qty: f64 = 0.0;
        let mut entry_price: f64 = 0.0;
        // Margin currently posted (locked up) for the open position. `0.0`
        // when flat. At leverage=1.0, margin_used == the position's full
        // notional at entry -- see portfoliomanager::margin's module docs
        // for why this makes leverage=1.0 behavior-identical to before
        // leverage existed.
        let mut margin_used: f64 = 0.0;
        let mut total_liquidations = 0usize;
        let mut equity_curve = Vec::with_capacity(n);
        let mut trade_returns = Vec::new();
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut gross_profit = 0.0f64;
        let mut gross_loss = 0.0f64;
        let mut total_commission = 0.0f64;
        let mut total_funding_paid = 0.0f64;
        let mut volume_constrained_skips = 0usize;
        let mut peak_equity = self.initial_capital;
        let mut max_drawdown = 0.0f64;
        let mut early_terminated = false;
        let mut last_funding_ts = candles.timestamps_ms[0];

        // Realism tracking
        let mut total_spread_cost = 0.0f64;
        let mut total_market_impact_cost = 0.0f64;
        let mut total_adverse_selection_cost = 0.0f64;
        let mut latency_rejected_signals = 0usize;
        let mut fee_tier_changes = 0usize;

        // Fee tier state
        let mut rolling_30d_volume = self.realism.starting_30d_volume;
        let mut effective_commission = self.commission_rate;
        let mut last_tier_check_ts = candles.timestamps_ms[0];
        let day_ms: i64 = 24 * 60 * 60 * 1000;

        for i in 0..n {
            let close = candles.closes[i];
            let high = candles.highs[i];
            let low = candles.lows[i];
            let open = candles.opens[i];
            let volume = candles.volumes[i];
            let ts = candles.timestamps_ms[i];

            // Fee tier decay (Item 8)
            if self.realism.fee_tiers.is_some() {
                let elapsed = ts - last_tier_check_ts;
                if elapsed >= day_ms {
                    let days = elapsed / day_ms;
                    let decay = ((30 - days.min(30)) as f64) / 30.0;
                    rolling_30d_volume *= decay;
                    last_tier_check_ts += days * day_ms;

                    if let Some(ref tiers) = self.realism.fee_tiers {
                        let new_commission = tiers.iter()
                            .rev()
                            .find(|t| rolling_30d_volume >= t.min_volume_30d)
                            .map(|t| t.taker_fee)
                            .unwrap_or(self.commission_rate);
                        if (new_commission - effective_commission).abs() > 1e-10 {
                            effective_commission = new_commission;
                            fee_tier_changes += 1;
                        }
                    }
                }
            }

            // Apply funding rate for perpetuals
            if position_qty != 0.0 && self.funding_rate_8h > 0.0 {
                let elapsed_ms = ts - last_funding_ts;
                if elapsed_ms >= self.funding_interval_ms {
                    let periods = elapsed_ms / self.funding_interval_ms;
                    let position_value = position_qty.abs() * close;
                    let funding_cost = position_value * self.funding_rate_8h * periods as f64;
                    cash -= funding_cost;
                    total_funding_paid += funding_cost;
                    last_funding_ts += periods * self.funding_interval_ms;
                }
            }

            // Margin call / liquidation check for leveraged positions.
            // Checked against the bar's intrabar extreme (bar low for longs,
            // bar high for shorts) BEFORE the strategy's own stop-loss/
            // take-profit below, since a margin call is an involuntary,
            // exchange-forced event that takes priority over a voluntary
            // exit rule. No-op at leverage=1.0 (`margin::is_liquidated`
            // always returns false, so this whole block is skipped).
            if position_qty != 0.0 && self.leverage > 1.0 {
                let side = if position_qty > 0.0 {
                    portfoliomanager::PositionSide::Long
                } else {
                    portfoliomanager::PositionSide::Short
                };
                let intrabar_extreme = match side {
                    portfoliomanager::PositionSide::Long => low,
                    portfoliomanager::PositionSide::Short => high,
                };
                if portfoliomanager::margin::is_liquidated(
                    intrabar_extreme, entry_price, side, self.leverage, self.maintenance_margin_ratio,
                ) {
                    let liq_price = portfoliomanager::margin::liquidation_price(
                        entry_price, side, self.leverage, self.maintenance_margin_ratio,
                    );
                    let liq_exit_price = portfoliomanager::margin::apply_liquidation_penalty(
                        liq_price, side, self.liquidation_penalty_bps,
                    );
                    let is_long = side == portfoliomanager::PositionSide::Long;
                    let ohlc = [open, high, low, close];
                    let fp = self.apply_fill_costs(
                        liq_exit_price, !is_long, volume, avg_volume,
                        position_qty.abs() * liq_exit_price, ohlc, false,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    self.close_position_with_commission(
                        &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                        fp, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                        &mut gross_profit, &mut gross_loss, &mut total_commission,
                        &mut rolling_30d_volume,
                    );
                    total_liquidations += 1;
                }
            }

            // Check stop-loss / take-profit on current bar
            if position_qty != 0.0 {
                let is_long = position_qty > 0.0;
                let ohlc = [open, high, low, close];

                let mut sl_triggered = false;
                let mut tp_triggered = false;
                let mut sl_base_price = 0.0f64;
                let mut tp_price_level = 0.0f64;

                if let Some(sl) = stop_loss_pct {
                    let sl_price = if is_long {
                        entry_price * (1.0 - sl)
                    } else {
                        entry_price * (1.0 + sl)
                    };
                    let triggered = if is_long { low <= sl_price } else { high >= sl_price };
                    if triggered {
                        sl_base_price = if is_long && open < sl_price {
                            open
                        } else if !is_long && open > sl_price {
                            open
                        } else {
                            sl_price
                        };
                        sl_triggered = true;
                    }
                }

                if let Some(tp) = take_profit_pct {
                    let tp_price = if is_long {
                        entry_price * (1.0 + tp)
                    } else {
                        entry_price * (1.0 - tp)
                    };
                    let triggered = if is_long { high >= tp_price } else { low <= tp_price };
                    if triggered {
                        tp_price_level = tp_price;
                        tp_triggered = true;
                    }
                }

                // Item 1: Intra-bar path — determine which triggers first.
                // Arbitrate BEFORE computing fill costs so the discarded trigger
                // doesn't inflate the spread/impact/adverse-selection totals.
                if sl_triggered && tp_triggered && self.realism.intrabar_path {
                    let up_first = (high - open) < (open - low);
                    let tp_fires_first = if is_long { up_first } else { !up_first };
                    if tp_fires_first {
                        sl_triggered = false;
                    } else {
                        tp_triggered = false;
                    }
                }

                if sl_triggered {
                    let sl_fill = self.apply_fill_costs(
                        sl_base_price, !is_long, volume, avg_volume,
                        position_qty.abs() * sl_base_price, ohlc, false,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    self.close_position_with_commission(
                        &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                        sl_fill, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                        &mut gross_profit, &mut gross_loss, &mut total_commission,
                        &mut rolling_30d_volume,
                    );
                } else if tp_triggered {
                    let tp_fill = self.apply_fill_costs(
                        tp_price_level, !is_long, volume, avg_volume,
                        position_qty.abs() * tp_price_level, ohlc, false,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    self.close_position_with_commission(
                        &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                        tp_fill, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                        &mut gross_profit, &mut gross_loss, &mut total_commission,
                        &mut rolling_30d_volume,
                    );
                }
            }

            // Process signal (fill at next bar's open)
            let fill_price_base = if i + 1 < n { candles.opens[i + 1] } else { close };
            let fill_volume = if i + 1 < n { candles.volumes[i + 1] } else { volume };
            let fill_ohlc = if i + 1 < n {
                [candles.opens[i + 1], candles.highs[i + 1], candles.lows[i + 1], candles.closes[i + 1]]
            } else {
                [open, high, low, close]
            };

            // Item 6: Latency rejection
            let signal = if self.realism.latency_mean_ms > 0.0 && i > 0 {
                let bar_dur_ms = (ts - candles.timestamps_ms[i.saturating_sub(1)]) as f64;
                let jitter = ((i.wrapping_mul(7919) % 1000) as f64 / 500.0 - 1.0) * self.realism.latency_std_ms;
                let latency = (self.realism.latency_mean_ms + jitter).max(0.0);
                if latency > bar_dur_ms && signals[i] != SIGNAL_HOLD {
                    latency_rejected_signals += 1;
                    SIGNAL_HOLD
                } else {
                    signals[i]
                }
            } else {
                signals[i]
            };

            match signal {
                SIGNAL_BUY if position_qty <= 0.0 => {
                    if position_qty < 0.0 {
                        let fp = self.apply_fill_costs(
                            fill_price_base, true, fill_volume, avg_volume,
                            position_qty.abs() * fill_price_base, fill_ohlc, false,
                            &mut total_spread_cost, &mut total_market_impact_cost,
                            &mut total_adverse_selection_cost,
                        );
                        self.close_position_with_commission(
                            &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                            fp, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                            &mut gross_profit, &mut gross_loss, &mut total_commission,
                            &mut rolling_30d_volume,
                        );
                    }
                    // Notional (and therefore quantity) scales with leverage;
                    // at leverage=1.0 this is identical to the pre-leverage
                    // `equity * position_size_pct` formula (see
                    // portfoliomanager::margin's module docs).
                    let equity = cash + margin_used + unrealized_pnl(position_qty, entry_price, fill_price_base);
                    let effective_position_size_pct = match (&self.vol_target, &vol_target_series) {
                        (Some(cfg), Some(series)) => portfoliomanager::margin::volatility_scaled_position_size_pct(
                            position_size_pct, series.get(i).copied().flatten(), cfg,
                        ),
                        _ => position_size_pct,
                    };
                    let (notional, _, _) = portfoliomanager::margin::size_leveraged_order(
                        equity, effective_position_size_pct, self.leverage, fill_price_base,
                    );
                    let mut size_value = notional;
                    let fp = self.apply_fill_costs(
                        fill_price_base, true, fill_volume, avg_volume,
                        size_value, fill_ohlc, true,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    let max_fill_value = fill_volume * self.fill_volume_fraction * fp;
                    if max_fill_value > 0.0 && size_value > max_fill_value {
                        size_value = max_fill_value;
                        volume_constrained_skips += 1;
                    } else if fill_volume <= 0.0 {
                        volume_constrained_skips += 1;
                    }
                    if size_value > 0.0 && fill_volume > 0.0 {
                        let qty = size_value / fp;
                        let commission = size_value * effective_commission;
                        let margin_posted = size_value / self.leverage.max(1.0);
                        cash -= margin_posted + commission;
                        margin_used += margin_posted;
                        position_qty = qty;
                        entry_price = fp;
                        total_commission += commission;
                        rolling_30d_volume += size_value;
                    }
                }
                SIGNAL_SELL if position_qty >= 0.0 => {
                    if position_qty > 0.0 {
                        let fp = self.apply_fill_costs(
                            fill_price_base, false, fill_volume, avg_volume,
                            position_qty.abs() * fill_price_base, fill_ohlc, false,
                            &mut total_spread_cost, &mut total_market_impact_cost,
                            &mut total_adverse_selection_cost,
                        );
                        self.close_position_with_commission(
                            &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                            fp, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                            &mut gross_profit, &mut gross_loss, &mut total_commission,
                            &mut rolling_30d_volume,
                        );
                    }
                    // Shorting a margined/leveraged instrument requires
                    // posting margin, same as opening a long -- NOT the
                    // classic "receive proceeds now" stock-borrow model,
                    // which doesn't scale sensibly under leverage. See
                    // close_position_with_commission's doc comment: this
                    // stays algebraically equivalent to the old formula at
                    // leverage=1.0 (verified via the equity conservation
                    // property in portfoliomanager::margin's module docs).
                    let equity = cash + margin_used + unrealized_pnl(position_qty, entry_price, fill_price_base);
                    let effective_position_size_pct = match (&self.vol_target, &vol_target_series) {
                        (Some(cfg), Some(series)) => portfoliomanager::margin::volatility_scaled_position_size_pct(
                            position_size_pct, series.get(i).copied().flatten(), cfg,
                        ),
                        _ => position_size_pct,
                    };
                    let (notional, _, _) = portfoliomanager::margin::size_leveraged_order(
                        equity, effective_position_size_pct, self.leverage, fill_price_base,
                    );
                    let mut size_value = notional;
                    let fp = self.apply_fill_costs(
                        fill_price_base, false, fill_volume, avg_volume,
                        size_value, fill_ohlc, true,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    let max_fill_value = fill_volume * self.fill_volume_fraction * fp;
                    if max_fill_value > 0.0 && size_value > max_fill_value {
                        size_value = max_fill_value;
                        volume_constrained_skips += 1;
                    } else if fill_volume <= 0.0 {
                        volume_constrained_skips += 1;
                    }
                    if size_value > 0.0 && fill_volume > 0.0 {
                        let qty = size_value / fp;
                        let commission = size_value * effective_commission;
                        let margin_posted = size_value / self.leverage.max(1.0);
                        cash -= margin_posted + commission;
                        margin_used += margin_posted;
                        position_qty = -qty;
                        entry_price = fp;
                        total_commission += commission;
                        rolling_30d_volume += size_value;
                    }
                }
                SIGNAL_CLOSE if position_qty != 0.0 => {
                    let is_long = position_qty > 0.0;
                    let fp = self.apply_fill_costs(
                        fill_price_base, !is_long, fill_volume, avg_volume,
                        position_qty.abs() * fill_price_base, fill_ohlc, false,
                        &mut total_spread_cost, &mut total_market_impact_cost,
                        &mut total_adverse_selection_cost,
                    );
                    self.close_position_with_commission(
                        &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                        fp, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                        &mut gross_profit, &mut gross_loss, &mut total_commission,
                        &mut rolling_30d_volume,
                    );
                }
                _ => {}
            }

            // Record equity (skip when caller only needs scalar metrics)
            let equity = cash + margin_used + unrealized_pnl(position_qty, entry_price, close);
            if self.track_equity_curve {
                equity_curve.push(equity);
            }

            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = (peak_equity - equity) / peak_equity;
            if dd > max_drawdown {
                max_drawdown = dd;
            }

            if let Some(et) = early_term {
                if i > 0 && i % et.check_interval == 0
                    && equity < self.initial_capital * (1.0 - et.max_loss_pct)
                {
                    early_terminated = true;
                    break;
                }
            }
        }

        // Close any remaining position at last close
        if position_qty != 0.0 {
            let last_close = candles.closes[n - 1];
            let last_volume = candles.volumes[n - 1];
            let is_long = position_qty > 0.0;
            let ohlc = [candles.opens[n-1], candles.highs[n-1], candles.lows[n-1], last_close];
            let fp = self.apply_fill_costs(
                last_close, !is_long, last_volume, avg_volume,
                position_qty.abs() * last_close, ohlc, false,
                &mut total_spread_cost, &mut total_market_impact_cost,
                &mut total_adverse_selection_cost,
            );
            self.close_position_with_commission(
                &mut cash, &mut position_qty, &mut entry_price, &mut margin_used,
                fp, effective_commission, &mut trade_returns, &mut wins, &mut losses,
                &mut gross_profit, &mut gross_loss, &mut total_commission,
                &mut rolling_30d_volume,
            );
            if self.track_equity_curve {
                if let Some(last) = equity_curve.last_mut() {
                    *last = cash;
                }
            }
        }

        let num_trades = wins + losses;
        let win_rate = if num_trades > 0 { wins as f64 / num_trades as f64 } else { 0.0 };
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { if gross_profit > 0.0 { 10.0 } else { 0.0 } };
        let total_pnl = cash - self.initial_capital;
        let sharpe_ratio = Self::compute_sharpe(&trade_returns);

        CandleSimResult {
            equity_curve,
            total_pnl,
            num_trades,
            win_rate,
            // `max_drawdown` is a fraction (e.g. 0.15 = 15%), matching every
            // other engine path -- see python_simulation.rs's own
            // compute_max_drawdown doc comment. Bug fixed 2026-09-06: this
            // used to multiply by `self.initial_capital`, turning the
            // fraction back into a dollar figure and silently breaking every
            // downstream consumer (Calmar ratio, risk gates, UI %) that
            // expects [0,1].
            max_drawdown,
            sharpe_ratio,
            profit_factor,
            trade_returns,
            total_commission,
            total_funding_paid,
            volume_constrained_skips,
            early_terminated,
            total_spread_cost,
            total_market_impact_cost,
            total_adverse_selection_cost,
            latency_rejected_signals,
            fee_tier_changes,
            total_liquidations,
        }
    }

    #[inline]
    fn apply_fill_costs(
        &self, price: f64, is_buy: bool, bar_volume: f64, avg_volume: f64,
        order_value: f64, ohlc: [f64; 4], is_entry: bool,
        spread_cost: &mut f64, impact_cost: &mut f64, adverse_cost: &mut f64,
    ) -> f64 {
        let mut bps = self.slippage_model.effective_bps(bar_volume, avg_volume);

        // Item 2: Market impact (sqrt model)
        if self.realism.market_impact_k > 0.0 && bar_volume > 0.0 {
            let vwap = (ohlc[0] + ohlc[1] + ohlc[2] + ohlc[3]) * 0.25;
            let participation = if vwap > 0.0 { order_value / (bar_volume * vwap) } else { 0.0 };
            let impact_bps = self.realism.market_impact_k * participation.min(1.0).sqrt() * 10_000.0;
            *impact_cost += price * impact_bps / 10_000.0 * (order_value / price).max(1.0);
            bps += impact_bps;
        }

        // Item 3: Spread from range (entry only)
        if is_entry && self.realism.spread_from_range && ohlc[3] > 0.0 {
            let range_bps = (ohlc[1] - ohlc[2]) / ohlc[3] * 10_000.0;
            let spread_bps = range_bps.min(self.realism.spread_cap_bps) * 0.5;
            *spread_cost += price * spread_bps / 10_000.0 * (order_value / price).max(1.0);
            bps += spread_bps;
        }

        // Item 4: Adverse selection (entry only)
        if is_entry && self.realism.adverse_selection && ohlc[0] > 0.0 {
            let bar_return = (ohlc[3] - ohlc[0]) / ohlc[0];
            let adversely_selected = (is_buy && bar_return > 0.0) || (!is_buy && bar_return < 0.0);
            if adversely_selected {
                let adv_bps = (bar_return.abs() * 10_000.0).min(self.realism.adverse_selection_max_bps);
                *adverse_cost += price * adv_bps / 10_000.0 * (order_value / price).max(1.0);
                bps += adv_bps;
            }
        }

        let slip = price * bps / 10_000.0;
        if is_buy { price + slip } else { price - slip }
    }

    #[inline]
    /// Closes the entire remaining position (this simulator never partially
    /// closes -- a signal either fully exits or leaves the position alone).
    /// Credits back the margin posted for it plus P&L, symmetric for long
    /// and short (see `unrealized_pnl`/module docs on `margin_used`) --
    /// at `leverage=1.0`, `margin_used` equals the position's full entry
    /// notional, so this is algebraically identical to the pre-leverage
    /// `exit_value`-based formula this function used to have.
    fn close_position_with_commission(
        &self,
        cash: &mut f64,
        position_qty: &mut f64,
        entry_price: &mut f64,
        margin_used: &mut f64,
        fill_price: f64,
        commission_rate: f64,
        trade_returns: &mut Vec<f64>,
        wins: &mut usize,
        losses: &mut usize,
        gross_profit: &mut f64,
        gross_loss: &mut f64,
        total_commission: &mut f64,
        rolling_30d_volume: &mut f64,
    ) {
        let qty = position_qty.abs();
        let exit_value = qty * fill_price;
        let commission = exit_value * commission_rate;

        let pnl = unrealized_pnl(*position_qty, *entry_price, fill_price) - commission;

        *cash += *margin_used + pnl;
        *margin_used = 0.0;

        *total_commission += commission;
        *rolling_30d_volume += exit_value;
        trade_returns.push(pnl);

        if pnl > 0.0 {
            *wins += 1;
            *gross_profit += pnl;
        } else {
            *losses += 1;
            *gross_loss += pnl.abs();
        }

        *position_qty = 0.0;
        *entry_price = 0.0;
    }

    fn compute_sharpe(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        if std_dev < 1e-10 {
            return 0.0;
        }
        // Annualize with 365 (crypto trades every day) — consistent with the
        // Monte Carlo module's SQRT_365 so GA fitness and MC report the same
        // Sharpe scale. Assumes roughly one trade per day.
        (mean / std_dev) * (365.0_f64).sqrt()
    }
}

impl CandleSimResult {
    pub fn empty(initial_capital: f64) -> Self {
        Self {
            equity_curve: vec![initial_capital],
            total_pnl: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: 0.0,
            profit_factor: 0.0,
            trade_returns: Vec::new(),
            total_commission: 0.0,
            total_funding_paid: 0.0,
            volume_constrained_skips: 0,
            early_terminated: false,
            total_spread_cost: 0.0,
            total_market_impact_cost: 0.0,
            total_adverse_selection_cost: 0.0,
            latency_rejected_signals: 0,
            fee_tier_changes: 0,
            total_liquidations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(closes: &[f64]) -> CandleArrays {
        CandleArrays {
            opens: closes.to_vec(),
            highs: closes.iter().map(|c| c * 1.01).collect(),
            lows: closes.iter().map(|c| c * 0.99).collect(),
            closes: closes.to_vec(),
            volumes: vec![1000.0; closes.len()],
            timestamps_ms: (0..closes.len() as i64).map(|i| i * 86_400_000).collect(),
        }
    }

    fn flat_sim(capital: f64, commission: f64, slippage_bps: f64) -> CandleSimulator {
        CandleSimulator::new(capital, commission, SlippageModel::Flat { bps: slippage_bps })
    }

    #[test]
    fn empty_signals_no_trades() {
        let sim = flat_sim(10000.0, 0.001, 1.0);
        let candles = make_candles(&[100.0, 101.0, 102.0]);
        let signals = vec![SIGNAL_HOLD; 3];
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert_eq!(result.num_trades, 0);
        assert!((result.total_pnl).abs() < 1e-10);
    }

    #[test]
    fn buy_and_sell_produces_trade() {
        let sim = flat_sim(10000.0, 0.0, 0.0);
        let candles = make_candles(&[100.0, 105.0, 110.0, 108.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_CLOSE, SIGNAL_HOLD];
        let result = sim.run(&signals, &candles, None, None, 1.0, None);
        assert_eq!(result.num_trades, 1);
        assert!(result.total_pnl > 0.0);
    }

    #[test]
    fn max_drawdown_is_a_fraction_not_a_dollar_amount() {
        // Regression test for a 2026-09-06 bug: max_drawdown used to be
        // multiplied by `self.initial_capital`, turning the [0,1] fraction
        // into a dollar figure. A large adverse move while holding a full-size
        // position should still report a drawdown no greater than 1.0.
        let sim = flat_sim(10000.0, 0.0, 0.0);
        let candles = make_candles(&[100.0, 50.0, 60.0, 60.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_CLOSE];
        let result = sim.run(&signals, &candles, None, None, 1.0, None);
        assert!(result.max_drawdown > 0.0, "expected a real drawdown from the 50% adverse move");
        assert!(
            result.max_drawdown <= 1.0,
            "max_drawdown must be a [0,1] fraction, got {} (looks dollar-scaled)",
            result.max_drawdown
        );
    }

    #[test]
    fn commission_reduces_pnl() {
        let sim_no_fee = flat_sim(10000.0, 0.0, 0.0);
        let sim_fee = flat_sim(10000.0, 0.01, 0.0);
        let candles = make_candles(&[100.0, 110.0, 120.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD];
        let r1 = sim_no_fee.run(&signals, &candles, None, None, 1.0, None);
        let r2 = sim_fee.run(&signals, &candles, None, None, 1.0, None);
        assert!(r1.total_pnl > r2.total_pnl);
    }

    #[test]
    fn stop_loss_triggers() {
        let sim = flat_sim(10000.0, 0.0, 0.0);
        let mut candles = make_candles(&[100.0, 95.0, 90.0, 85.0]);
        candles.lows[1] = 94.0;
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD];
        let result = sim.run(&signals, &candles, Some(0.05), None, 1.0, None);
        assert_eq!(result.num_trades, 1);
        assert!(result.total_pnl < 0.0);
    }

    #[test]
    fn volume_adjusted_slippage_thin_bar_costs_more() {
        let sim = CandleSimulator::new(10000.0, 0.0, SlippageModel::VolumeAdjusted { base_bps: 5.0, max_multiplier: 10.0 })
            .with_fill_volume_fraction(1.0); // disable volume constraint for this test
        // Scenario 1: entry fills on thick bar (bar 1 = 10000 vol), exit on thick bar (bar 2 = 10000)
        let mut candles1 = make_candles(&[100.0, 100.0, 100.0]);
        candles1.volumes = vec![10000.0, 10000.0, 10000.0];
        let signals1 = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        // Scenario 2: entry fills on thin bar (bar 1 = 100 vol), exit on thick bar (bar 2 = 10000)
        let mut candles2 = make_candles(&[100.0, 100.0, 100.0]);
        candles2.volumes = vec![10000.0, 100.0, 10000.0];
        let signals2 = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        let r1 = sim.run(&signals1, &candles1, None, None, 0.5, None);
        let r2 = sim.run(&signals2, &candles2, None, None, 0.5, None);
        // Thin entry bar should cost more (lower PnL due to higher slippage)
        assert!(r1.total_pnl > r2.total_pnl, "thick bar pnl {} should > thin bar pnl {}", r1.total_pnl, r2.total_pnl);
    }

    #[test]
    fn volume_constraint_caps_position() {
        let sim = CandleSimulator::new(100000.0, 0.0, SlippageModel::Flat { bps: 0.0 })
            .with_fill_volume_fraction(0.10);
        let mut candles = make_candles(&[100.0, 100.0, 100.0]);
        // Very low volume: 10 units * $100 * 10% = $100 max fill
        candles.volumes = vec![10.0, 10.0, 10.0];
        let signals = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        let result = sim.run(&signals, &candles, None, None, 1.0, None);
        assert!(result.volume_constrained_skips > 0);
    }

    #[test]
    fn funding_rate_deducted() {
        let sim = CandleSimulator::new(10000.0, 0.0, SlippageModel::Flat { bps: 0.0 })
            .with_funding_rate(0.0001, 8 * 60 * 60 * 1000);
        let mut candles = make_candles(&[100.0; 10]);
        candles.timestamps_ms = (0..10).map(|i| i * 8 * 60 * 60 * 1000).collect();
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD,
                          SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_CLOSE];
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert!(result.total_funding_paid > 0.0);
        assert!(result.total_pnl < 0.0);
    }

    #[test]
    fn test_default_realism_matches_legacy() {
        let sim = flat_sim(10000.0, 0.001, 5.0);
        let candles = make_candles(&[100.0, 105.0, 110.0, 108.0, 112.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_HOLD, SIGNAL_CLOSE];
        let result = sim.run(&signals, &candles, Some(0.03), Some(0.05), 0.5, None);
        assert_eq!(result.total_spread_cost, 0.0);
        assert_eq!(result.total_market_impact_cost, 0.0);
        assert_eq!(result.total_adverse_selection_cost, 0.0);
        assert_eq!(result.latency_rejected_signals, 0);
        assert_eq!(result.fee_tier_changes, 0);
    }

    #[test]
    fn test_intrabar_path_tp_first() {
        let sim = flat_sim(10000.0, 0.0, 0.0).with_realism(RealismConfig {
            intrabar_path: true,
            ..Default::default()
        });
        let mut candles = make_candles(&[100.0, 100.0, 100.0]);
        // Bar 1: high-open=8 < open-low=12 → up first → TP fires first for long
        candles.opens[1] = 100.0;
        candles.highs[1] = 108.0;
        candles.lows[1] = 88.0;
        candles.closes[1] = 100.0;
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD];
        // Entry at bar 1 open = 100. SL at 5% = 95, TP at 5% = 105
        // Bar 1: high=108 >= 105 (TP), low=88 <= 95 (SL). Both trigger.
        let result = sim.run(&signals, &candles, Some(0.05), Some(0.05), 1.0, None);
        assert_eq!(result.num_trades, 1);
        assert!(result.total_pnl > 0.0, "TP should fire first, pnl={}", result.total_pnl);
    }

    #[test]
    fn test_market_impact_proportional_to_size() {
        let realism = RealismConfig {
            market_impact_k: 0.3,
            ..Default::default()
        };
        let sim_small = flat_sim(10000.0, 0.0, 0.0).with_realism(realism.clone());
        let sim_large = flat_sim(100000.0, 0.0, 0.0).with_realism(realism);
        let candles = make_candles(&[100.0, 100.0, 100.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        let r_small = sim_small.run(&signals, &candles, None, None, 1.0, None);
        let r_large = sim_large.run(&signals, &candles, None, None, 1.0, None);
        assert!(r_large.total_market_impact_cost > r_small.total_market_impact_cost);
    }

    #[test]
    fn test_spread_from_range_volatile_bar() {
        let realism = RealismConfig {
            spread_from_range: true,
            spread_cap_bps: 50.0,
            ..Default::default()
        };
        let sim = flat_sim(10000.0, 0.0, 0.0).with_realism(realism);
        let mut candles = make_candles(&[100.0, 100.0, 100.0]);
        candles.highs[1] = 110.0;
        candles.lows[1] = 90.0;
        candles.closes[1] = 100.0;
        let signals = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert!(result.total_spread_cost > 0.0, "spread cost should be > 0");
    }

    #[test]
    fn test_adverse_selection_buy_green_bar() {
        let realism = RealismConfig {
            adverse_selection: true,
            adverse_selection_max_bps: 20.0,
            ..Default::default()
        };
        let sim = flat_sim(10000.0, 0.0, 0.0).with_realism(realism);
        let mut candles_green = make_candles(&[100.0, 105.0, 105.0]);
        candles_green.opens[1] = 100.0;
        candles_green.closes[1] = 105.0;
        let mut candles_red = make_candles(&[100.0, 95.0, 95.0]);
        candles_red.opens[1] = 100.0;
        candles_red.closes[1] = 95.0;
        let signals = vec![SIGNAL_BUY, SIGNAL_CLOSE, SIGNAL_HOLD];
        let r_green = sim.run(&signals, &candles_green, None, None, 0.5, None);
        let r_red = sim.run(&signals, &candles_red, None, None, 0.5, None);
        assert!(r_green.total_adverse_selection_cost > 0.0);
        assert_eq!(r_red.total_adverse_selection_cost, 0.0);
    }

    #[test]
    fn test_latency_rejection_stale_signal() {
        let realism = RealismConfig {
            latency_mean_ms: 100_000.0,
            latency_std_ms: 0.0,
            ..Default::default()
        };
        let sim = flat_sim(10000.0, 0.0, 0.0).with_realism(realism);
        let mut candles = make_candles(&[100.0, 105.0, 110.0]);
        candles.timestamps_ms = vec![0, 50_000, 100_000]; // 50s bars, latency=100s > bar
        let signals = vec![SIGNAL_BUY, SIGNAL_BUY, SIGNAL_CLOSE];
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert!(result.latency_rejected_signals > 0);
    }

    #[test]
    fn test_fee_tier_decay_no_trading() {
        let realism = RealismConfig {
            fee_tiers: Some(vec![
                config::VolumeTier { min_volume_30d: 0.0, maker_fee: 0.0040, taker_fee: 0.0040 },
                config::VolumeTier { min_volume_30d: 100_000.0, maker_fee: 0.0010, taker_fee: 0.0010 },
            ]),
            starting_30d_volume: 200_000.0,
            ..Default::default()
        };
        let sim = flat_sim(10000.0, 0.0010, 0.0).with_realism(realism);
        let n = 35;
        let mut candles = make_candles(&vec![100.0; n]);
        candles.timestamps_ms = (0..n as i64).map(|i| i * 86_400_000).collect();
        let mut signals = vec![SIGNAL_HOLD; n];
        signals[0] = SIGNAL_BUY;
        signals[n - 1] = SIGNAL_CLOSE;
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert!(result.fee_tier_changes > 0, "fee tier should change after volume decay");
    }

    // ---- Leverage/margin (GA-fitness engine parity with portfoliomanager) ----

    #[test]
    fn leverage_one_is_explicitly_a_noop() {
        // Setting leverage=1.0 explicitly must produce byte-identical results
        // to the default (unset) simulator -- the whole point of the
        // leverage=1.0 conservation property.
        let candles = make_candles(&[100.0, 110.0, 105.0, 120.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_CLOSE];
        let default_sim = flat_sim(10_000.0, 0.001, 5.0);
        let explicit_sim = flat_sim(10_000.0, 0.001, 5.0).with_leverage(1.0, 0.005, 50.0);
        let r1 = default_sim.run(&signals, &candles, None, None, 0.5, None);
        let r2 = explicit_sim.run(&signals, &candles, None, None, 0.5, None);
        assert_eq!(r1.total_pnl, r2.total_pnl);
        assert_eq!(r1.equity_curve, r2.equity_curve);
        assert_eq!(r1.num_trades, r2.num_trades);
        assert_eq!(r2.total_liquidations, 0);
    }

    #[test]
    fn leverage_scales_pnl_proportionally_to_notional() {
        // Same price path, same position_size_pct: 4x leverage should
        // produce ~4x the total_pnl of 1x, since notional (and therefore
        // absolute P&L for the same % price move) scales with leverage
        // while margin posted (and therefore risk of ruin) does not change
        // the SIZING inputs. Volume constraint disabled so the larger
        // leveraged notional isn't fill-capped by available bar volume
        // (that's correct, separate behavior -- see
        // `high_leverage_triggers_liquidation_on_sharp_adverse_move` for
        // the interaction this test intentionally avoids).
        let candles = make_candles(&[100.0, 110.0, 120.0]);
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_CLOSE];
        let sim_1x = flat_sim(100_000.0, 0.0, 0.0).with_fill_volume_fraction(1.0);
        let sim_4x = flat_sim(100_000.0, 0.0, 0.0).with_leverage(4.0, 0.005, 50.0).with_fill_volume_fraction(1.0);
        let r1 = sim_1x.run(&signals, &candles, None, None, 0.1, None);
        let r4 = sim_4x.run(&signals, &candles, None, None, 0.1, None);
        assert!((r4.total_pnl - r1.total_pnl * 4.0).abs() < 1.0,
            "4x leverage pnl {} should be ~4x the 1x pnl {}", r4.total_pnl, r1.total_pnl);
    }

    #[test]
    fn high_leverage_triggers_liquidation_on_sharp_adverse_move() {
        // A ~15% adverse move against a 10x long should blow through
        // maintenance margin (10x needs only a ~10% adverse move to wipe
        // out posted margin) and force-liquidate; the identical price path
        // at leverage=1x should never liquidate.
        let mut candles = make_candles(&[100.0, 100.0, 85.0, 85.0]);
        candles.lows[2] = 84.0; // intrabar extreme dips further
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD];

        let sim_10x = flat_sim(100_000.0, 0.0, 0.0).with_leverage(10.0, 0.005, 50.0);
        let result_10x = sim_10x.run(&signals, &candles, None, None, 0.5, None);
        assert!(result_10x.total_liquidations > 0,
            "10x leverage should be liquidated by a ~15% adverse move");

        let sim_1x = flat_sim(100_000.0, 0.0, 0.0);
        let result_1x = sim_1x.run(&signals, &candles, None, None, 0.5, None);
        assert_eq!(result_1x.total_liquidations, 0,
            "unleveraged (1x) positions should never liquidate");
    }

    #[test]
    fn liquidation_cash_never_goes_more_negative_than_pre_liquidation_equity() {
        // Isolated margin's core guarantee: even a severe adverse move
        // shouldn't leave the simulator with wildly negative cash -- the
        // position's own posted margin absorbs the loss.
        let mut candles = make_candles(&[100.0, 100.0, 50.0, 50.0]); // 50% crash
        candles.lows[2] = 40.0;
        let signals = vec![SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_HOLD, SIGNAL_HOLD];
        let sim = flat_sim(100_000.0, 0.0, 0.0).with_leverage(10.0, 0.005, 50.0);
        let result = sim.run(&signals, &candles, None, None, 0.5, None);
        assert!(result.total_liquidations > 0);
        // Losing the full 10,000 posted margin (10x on a 10% position_size_pct
        // slice of 100k) is the worst case -- total_pnl should not be
        // catastrophically beyond that (no unbounded negative cash).
        assert!(result.total_pnl > -15_000.0,
            "liquidation loss should be roughly bounded by posted margin, got {}", result.total_pnl);
    }
}
