//! Portfolio Manager Module
//!
//! Provides portfolio state tracking, position management, and P&L calculation for trading systems.

pub mod lp_position;
pub mod margin;

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use orderbook::{Fill, BookSide};
use derivatives::{DerivativeMetadata, Greeks};
use greeks::{aggregate_greeks, bs_greeks, bs_price};
use anyhow::Result;

/// Represents the state of the trading portfolio at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub timestamp: DateTime<Utc>,
    pub balance: f64,              // Total account balance (quote currency)
    pub net_position: f64,             // Net position (base asset, e.g. BTC)
    pub positions: Vec<Position>,      // Open positions
    pub realized_pnl: f64,         // Realized profit and loss
    pub unrealized_pnl: f64,       // Unrealized profit and loss
    pub fees_paid: f64,            // Total fees paid
    /// Leverage multiplier applied to new position sizing/margin (see
    /// `portfoliomanager::margin`). `1.0` (default) = no leverage --
    /// behavior-identical to this portfolio before leverage support existed.
    #[serde(default = "default_leverage")]
    pub leverage: f64,
}

fn default_leverage() -> f64 { 1.0 }

impl Default for PortfolioState {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            balance: 0.0,
            net_position: 0.0,
            positions: Vec::new(),
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            fees_paid: 0.0,
            leverage: default_leverage(),
        }
    }
}

/// Represents an open or closed position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,                // Trading pair (e.g. BTC/USDT)
    pub side: PositionSide,            // Long or Short
    pub quantity: f64,                 // Position size (base asset)
    pub entry_price: f64,              // Entry price
    pub mark_price: Option<f64>,       // Current mark price
    pub realized_pnl: f64,            // Realized P&L for this position
    pub unrealized_pnl: f64,          // Unrealized P&L for this position
    pub open_time: DateTime<Utc>,
    pub close_time: Option<DateTime<Utc>>,
    /// Derivative contract metadata. `None` for spot positions.
    #[serde(default)]
    pub instrument: Option<DerivativeMetadata>,
    /// Per-contract Greeks snapshot. `None` for spot or when not yet computed.
    #[serde(default)]
    pub greeks: Option<Greeks>,
    /// Cash actually posted as margin for this position (quote currency).
    /// `0.0` for option positions (which use full-premium accounting via
    /// `margin_requirement()`, untouched by leverage) and for any position
    /// opened at `leverage = 1.0` where margin equals the full notional
    /// debit -- see `portfoliomanager::margin` for the leverage math.
    /// `#[serde(default)]` so persisted pre-leverage state deserializes as `0.0`.
    #[serde(default)]
    pub margin_posted: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

impl PortfolioState {
    /// Create a new portfolio with an initial balance (quote asset only)
    pub fn with_balance(balance: f64) -> Self {
        Self {
            balance,
            ..Default::default()
        }
    }
    
    /// Create a new portfolio for market making with both quote and base assets
    /// 
    /// For proper market making, you need inventory on BOTH sides:
    /// - quote_balance: USD/USDT to buy the base asset (fund bid orders)
    /// - base_quantity: BTC/ETH to sell (fund ask orders)
    /// 
    /// Typically split 50/50 by value for neutral market making.
    pub fn with_inventory(
        symbol: String,
        quote_balance: f64,
        base_quantity: f64,
        base_price: f64,
        open_time: DateTime<Utc>,
    ) -> Self {
        let mut portfolio = Self {
            balance: quote_balance,
            net_position: base_quantity,
            ..Default::default()
        };
        
        // Create initial long position for the base asset inventory
        if base_quantity > 0.0 {
            portfolio.positions.push(Position {
                symbol,
                side: PositionSide::Long,
                quantity: base_quantity,
                entry_price: base_price,
                mark_price: Some(base_price),
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                open_time,
                close_time: None,
                instrument: None,
                greeks: None,
                margin_posted: 0.0,
            });
        }

        portfolio
    }

    /// Apply multiple fills to the portfolio
    pub fn apply_fills(&mut self, fills: &[Fill]) -> Result<()> {
        for fill in fills {
            self.apply_fill(fill)?;
        }
        Ok(())
    }

    /// Like `apply_fills` but returns the total actually executed quantity across
    /// all fills. Execution may be less than requested when capped by available
    /// balance (buys) or inventory (sells). Callers should charge commission on
    /// the executed quantity, not the requested quantity.
    pub fn apply_fills_executed(&mut self, fills: &[Fill]) -> Result<f64> {
        let mut total_executed = 0.0;
        for fill in fills {
            total_executed += self.apply_fill_executed(fill)?;
        }
        Ok(total_executed)
    }

    /// Get the available quantity for a given symbol and side
    /// For sells: returns the quantity of Long positions we can sell
    /// For buys: returns max quantity affordable with current balance
    fn get_available_quantity(&self, symbol: &str, side: &BookSide, price: f64) -> f64 {
        match side {
            BookSide::Ask => {
                // For selling, we can only sell what we have in Long positions
                self.positions.iter()
                    .filter(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long)
                    .map(|p| p.quantity.max(0.0)) // Ensure no negative quantities leak through
                    .sum()
            }
            BookSide::Bid => {
                // For buying, limited by available balance
                // Calculate max quantity we can afford: balance / price
                if price <= 0.0 {
                    return 0.0;
                }
                if self.balance <= 0.0 {
                    return 0.0; // No balance, can't buy
                }
                // Allow buying up to 95% of balance to leave room for fees
                (self.balance * 0.95) / price
            }
        }
    }

    /// Apply a single fill to the portfolio with proper validation
    /// 
    /// This method ensures:
    /// 1. You cannot sell more than you own (prevents negative Long positions)
    /// 2. Balance and position updates are atomic and consistent
    /// 3. Fees are proportionally adjusted if quantity is reduced
    pub fn apply_fill(&mut self, fill: &Fill) -> Result<()> {
        self.apply_fill_executed(fill).map(|_| ())
    }

    /// Like `apply_fill` but returns the actually executed quantity (which may be
    /// less than `fill.quantity` when capped by available balance or inventory,
    /// or `0.0` when nothing could be executed).
    ///
    /// Distinguishes four cases based on what's already open for `symbol`:
    /// buy-to-open-long, buy-to-cover-an-existing-short, sell-to-close-an-
    /// existing-long, and sell-to-open-a-new-short -- full long+short
    /// mechanics, not long-only. (Previously the Ask branch unconditionally
    /// assumed an existing long and silently executed zero quantity with
    /// none open -- ordinary directional strategies could never actually
    /// open a short position through this path, contradicting the SDK's own
    /// documented `SELL = -1 // open or increase short position` contract.)
    ///
    /// Opening a position debits `margin_posted = trade_value / leverage`
    /// from balance rather than the full notional (see `portfoliomanager::margin`);
    /// at `leverage = 1.0` this is exactly the full notional, unchanged from
    /// before this method supported leverage. Opening a fresh short requires
    /// margin the same way a long does -- unlike writing an option
    /// (`apply_derivative_fill`'s premium-credit branch), shorting a spot/
    /// futures-style instrument doesn't pay you money upfront.
    pub fn apply_fill_executed(&mut self, fill: &Fill) -> Result<f64> {
        let symbol = "BTC/USD".to_string(); // TODO: Get from fill/order in production

        self.timestamp = fill.timestamp;
        let requested_quantity = fill.quantity;
        if requested_quantity <= 0.0 {
            return Ok(0.0); // Nothing to do
        }

        let has_open_long = self.positions.iter()
            .any(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long);
        let has_open_short = self.positions.iter()
            .any(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Short);
        let leverage = self.leverage.max(1.0);

        match fill.side {
            BookSide::Bid if has_open_short => {
                // Buy-to-cover an existing short.
                let available = self.get_available_short_quantity(&symbol);
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                let (realized, margin_released) = self.reduce_short_position(&symbol, executed_quantity, fill.price, fill.timestamp);
                self.balance += margin_released + realized - fee_amount;
                self.realized_pnl += realized;
                self.net_position += executed_quantity;
                Ok(executed_quantity)
            }
            BookSide::Bid => {
                // Buy-to-open (or add to) a long. Buying power scales with
                // leverage -- NOT via the shared `get_available_quantity`
                // (that helper stays unleveraged since `apply_derivative_fill`
                // also uses it for the options case, which leverage never applies to).
                if fill.price <= 0.0 || self.balance <= 0.0 {
                    return Ok(0.0);
                }
                let available = (self.balance * 0.95 * leverage) / fill.price;
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let trade_value = fill.price * executed_quantity;
                let margin_posted = trade_value / leverage;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                self.balance -= margin_posted;
                self.balance -= fee_amount;
                self.net_position += executed_quantity;
                self.add_to_long_position(&symbol, executed_quantity, fill.price, fill.timestamp, margin_posted);
                Ok(executed_quantity)
            }
            BookSide::Ask if has_open_long => {
                // Sell-to-close an existing long.
                let available = self.get_available_quantity(&symbol, &fill.side, fill.price);
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                let (realized, margin_released) = self.reduce_long_position(&symbol, executed_quantity, fill.price, fill.timestamp);
                self.balance += margin_released + realized - fee_amount;
                self.realized_pnl += realized;
                self.net_position -= executed_quantity;
                Ok(executed_quantity)
            }
            BookSide::Ask => {
                // Sell-to-open (or add to) a new Short. Capped by margin/
                // buying power the same way as opening a long -- NOT
                // uncapped like an option premium credit.
                if fill.price <= 0.0 {
                    return Ok(0.0);
                }
                let available = ((self.balance * 0.95 * leverage) / fill.price).max(0.0);
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let trade_value = fill.price * executed_quantity;
                let margin_posted = trade_value / leverage;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                self.balance -= margin_posted;
                self.balance -= fee_amount;
                self.net_position -= executed_quantity;
                self.add_to_short_position_with_instrument(&symbol, executed_quantity, fill.price, fill.timestamp, None, margin_posted);
                Ok(executed_quantity)
            }
        }
    }

    /// Add to an existing Long position or create a new one
    /// Uses weighted average for entry price when adding to existing position.
    /// `margin_posted` is the cash actually debited for this fill (see
    /// `portfoliomanager::margin`) -- summed into the position's running
    /// total so a later partial/full close knows how much margin to release.
    fn add_to_long_position(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
        margin_posted: f64,
    ) {
        // Find existing open Long position
        let existing = self.positions.iter_mut()
            .find(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long);

        match existing {
            Some(position) => {
                // Update with weighted average entry price
                let old_value = position.entry_price * position.quantity;
                let new_value = price * quantity;
                let new_quantity = position.quantity + quantity;

                if new_quantity > 0.0 {
                    position.entry_price = (old_value + new_value) / new_quantity;
                }
                position.quantity = new_quantity;
                position.mark_price = Some(price);
                position.margin_posted += margin_posted;
            }
            None => {
                // Create new Long position
                self.positions.push(Position {
                    symbol: symbol.to_string(),
                    side: PositionSide::Long,
                    quantity,
                    entry_price: price,
                    mark_price: Some(price),
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    open_time: timestamp,
                    close_time: None,
                    margin_posted,
                    instrument: None,
                    greeks: None,
                });
            }
        }
    }

    /// Reduce a Long position by selling
    /// Returns the realized P&L from the sale
    ///
    /// INVARIANT: quantity must be <= available position quantity
    /// This is enforced by apply_fill() before calling this method
    ///
    /// Returns `(realized_pnl, margin_released)` -- `margin_released` is the
    /// slice of the position's posted margin proportional to the fraction
    /// closed (e.g. closing half a position releases half its margin_posted),
    /// which the caller credits back to `balance` instead of the raw trade
    /// value once leverage is in play (see `portfoliomanager::margin`).
    fn reduce_long_position(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>
    ) -> (f64, f64) {
        let mut remaining_to_sell = quantity;
        let mut total_realized_pnl = 0.0;
        let mut total_margin_released = 0.0;

        // Process positions in order (FIFO)
        for position in self.positions.iter_mut()
            .filter(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long)
        {
            if remaining_to_sell <= 0.0 {
                break;
            }

            // How much can we sell from this position?
            let sell_from_this = remaining_to_sell.min(position.quantity);

            if sell_from_this > 0.0 {
                // Calculate realized P&L: (sell_price - entry_price) * quantity
                let pnl = (price - position.entry_price) * sell_from_this;
                let margin_fraction = if position.quantity > 0.0 { sell_from_this / position.quantity } else { 0.0 };
                let margin_released = position.margin_posted * margin_fraction;

                position.realized_pnl += pnl;
                total_realized_pnl += pnl;
                position.margin_posted -= margin_released;
                total_margin_released += margin_released;

                position.quantity -= sell_from_this;
                remaining_to_sell -= sell_from_this;

                // Close position if fully sold (use epsilon for float comparison)
                if position.quantity < 1e-10 {
                    position.quantity = 0.0;
                    position.close_time = Some(timestamp);
                }
            }
        }

        (total_realized_pnl, total_margin_released)
    }

    /// Get total portfolio value including the market value of all open positions.
    ///
    /// Two valuation models coexist, matched per-position by whether it's an
    /// option:
    /// - Options (untouched by leverage -- see module docs): the pre-existing
    ///   credit-premium model, `+price*qty` for a long holding, `-price*qty`
    ///   for a short (written) position.
    /// - Everything else (spot and futures/perps, with or without leverage):
    ///   `margin_posted + unrealized_pnl`. At `leverage = 1.0` this is
    ///   algebraically identical to the old `mark_price * quantity` formula
    ///   for longs (`margin_posted` is the full entry notional, so
    ///   `margin_posted + (mark-entry)*qty == mark*qty`) -- the old formula's
    ///   short branch was never actually exercised before this feature
    ///   (opening a fresh short via the plain fill path didn't work), so
    ///   there's no prior behavior to preserve there.
    pub fn get_total_value(&self) -> f64 {
        let mut position_value = 0.0;

        for position in &self.positions {
            if position.close_time.is_none() && position.quantity > 0.0 {
                let is_option = position.instrument.as_ref()
                    .map(|i| i.instrument_kind.is_option())
                    .unwrap_or(false);
                if is_option {
                    let price = position.mark_price.unwrap_or(position.entry_price);
                    match position.side {
                        PositionSide::Long => position_value += price * position.quantity,
                        PositionSide::Short => position_value -= price * position.quantity,
                    }
                } else {
                    position_value += position.margin_posted + position.unrealized_pnl;
                }
            }
        }

        self.balance + position_value
    }

    /// Update unrealized P&L for all open positions based on current market price
    pub fn update_unrealized_pnl(&mut self, current_price: f64) {
        let mut total_unrealized = 0.0;
        
        for position in &mut self.positions {
            if position.close_time.is_none() && position.quantity > 0.0 {
                position.mark_price = Some(current_price);
                
                let pnl = match position.side {
                    PositionSide::Long => {
                        (current_price - position.entry_price) * position.quantity
                    }
                    PositionSide::Short => {
                        (position.entry_price - current_price) * position.quantity
                    }
                };
                
                position.unrealized_pnl = pnl;
                total_unrealized += pnl;
            }
        }

        self.unrealized_pnl = total_unrealized;
    }

    /// Set the leverage multiplier applied to subsequent fills' margin
    /// requirements. `1.0` = no leverage.
    pub fn with_leverage(mut self, leverage: f64) -> Self {
        self.leverage = leverage.max(1.0);
        self
    }

    /// Total margin currently posted (locked up) across all open,
    /// non-option positions. `0.0` for option positions (see
    /// [`get_total_value`](Self::get_total_value)'s doc comment).
    pub fn margin_used(&self) -> f64 {
        self.positions.iter()
            .filter(|p| p.close_time.is_none())
            .map(|p| p.margin_posted)
            .sum()
    }

    /// Unrealized P&L across all open, non-option positions if marked at
    /// `current_price`. Read-only sibling to [`update_unrealized_pnl`], for
    /// use mid-tick when a fresh mark-to-market equity figure is needed
    /// without waiting for the once-per-tick `update_unrealized_pnl` call
    /// (or mutating position state).
    pub fn unrealized_pnl_at(&self, current_price: f64) -> f64 {
        self.positions.iter()
            .filter(|p| p.close_time.is_none() && p.quantity > 0.0 && p.instrument.is_none())
            .map(|p| match p.side {
                PositionSide::Long => (current_price - p.entry_price) * p.quantity,
                PositionSide::Short => (p.entry_price - current_price) * p.quantity,
            })
            .sum()
    }

    /// Mark-to-market equity at `current_price`: free cash + margin currently
    /// posted for open positions + unrealized P&L. This is the leverage-
    /// correct successor to the naive `balance + net_position * current_price`
    /// formula, which overstates equity once a position is only margin-backed
    /// rather than fully owned (see `portfoliomanager::margin`'s module docs).
    pub fn equity_at(&self, current_price: f64) -> f64 {
        self.balance + self.margin_used() + self.unrealized_pnl_at(current_price)
    }

    /// Apply a derivative fill, scaling balance changes by the contract multiplier and
    /// tagging the position with `instrument` metadata.
    ///
    /// Unlike [`apply_fill_executed`] (which only ever opens/closes a Long), this
    /// distinguishes four cases based on what's already open for `instrument.symbol`:
    /// buy-to-open-long, buy-to-cover-an-existing-short, sell-to-close-an-existing-
    /// long, and sell-to-open-a-new-short (writing/selling an option with no prior
    /// inventory) -- full long+short mechanics, not long-only.
    ///
    /// The position symbol is taken from `instrument.symbol`. For options this debits
    /// the premium × multiplier from balance on entry (buy) and credits it on entry
    /// (sell/write) or exit (close), scaled by `instrument.contract_multiplier`.
    ///
    /// Returns the actually-executed quantity (may be less than `fill.quantity`
    /// when capped by available balance/inventory, or `0.0` when nothing could
    /// be executed) -- mirrors [`apply_fill_executed`]'s contract so callers can
    /// charge commission and record trades on what actually happened, not what
    /// was requested.
    ///
    /// Leverage/margin only applies to non-option instruments (futures/perps):
    /// options keep their pre-existing full-premium accounting untouched --
    /// buying an option always debits the full premium (no margin concept for
    /// the holder), and writing/selling one always credits premium uncapped
    /// by balance/inventory (unrelated to leverage). Futures/perps instead
    /// debit `margin_posted = trade_value / leverage` on open (both long AND
    /// short -- shorting a future/perp requires posting margin, it doesn't
    /// pay you a premium), capped by available margin/buying power.
    pub fn apply_derivative_fill(&mut self, fill: &Fill, instrument: &DerivativeMetadata) -> Result<f64> {
        let symbol = instrument.symbol.clone();
        let multiplier = instrument.contract_multiplier;
        let is_option = instrument.instrument_kind.is_option();
        let leverage = self.leverage.max(1.0);

        self.timestamp = fill.timestamp;
        let requested_quantity = fill.quantity;
        if requested_quantity <= 0.0 {
            return Ok(0.0);
        }

        let has_open_long = self.positions.iter()
            .any(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long);
        let has_open_short = self.positions.iter()
            .any(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Short);

        match &fill.side {
            BookSide::Bid if has_open_short => {
                // Buy-to-cover an existing short.
                let available = self.get_available_short_quantity(&symbol);
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let trade_value = fill.price * executed_quantity * multiplier;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                let (realized, margin_released) = self.reduce_short_position(&symbol, executed_quantity, fill.price, fill.timestamp);
                if is_option {
                    self.balance -= trade_value;
                } else {
                    self.balance += margin_released;
                }
                self.balance -= fee_amount;
                self.net_position += executed_quantity;
                self.realized_pnl += realized;
                Ok(executed_quantity)
            }
            BookSide::Bid => {
                // Buy-to-open (or add to) a long.
                let available = if is_option {
                    self.get_available_quantity(&symbol, &fill.side, fill.price)
                } else {
                    ((self.balance * 0.95 * leverage) / (fill.price * multiplier)).max(0.0)
                };
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let trade_value = fill.price * executed_quantity * multiplier;
                let margin_posted = if is_option { trade_value } else { trade_value / leverage };
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                self.balance -= margin_posted;
                self.balance -= fee_amount;
                self.net_position += executed_quantity;
                self.add_to_long_position_with_instrument(
                    &symbol,
                    executed_quantity,
                    fill.price,
                    fill.timestamp,
                    Some(instrument.clone()),
                    if is_option { 0.0 } else { margin_posted },
                );
                Ok(executed_quantity)
            }
            BookSide::Ask if has_open_long => {
                // Sell-to-close an existing long.
                let available = self.get_available_quantity(&symbol, &fill.side, fill.price);
                let executed_quantity = requested_quantity.min(available);
                if executed_quantity < 1e-10 {
                    return Ok(0.0);
                }
                let execution_ratio = executed_quantity / requested_quantity;
                let trade_value = fill.price * executed_quantity * multiplier;
                let fee_amount = fill.fee_amount * execution_ratio;
                self.fees_paid += fee_amount;
                let (realized, margin_released) = self.reduce_long_position(&symbol, executed_quantity, fill.price, fill.timestamp);
                if is_option {
                    self.balance += trade_value;
                } else {
                    self.balance += margin_released;
                }
                self.balance -= fee_amount;
                self.net_position -= executed_quantity;
                self.realized_pnl += realized;
                Ok(executed_quantity)
            }
            BookSide::Ask => {
                // Sell-to-open (write) a new short, or add to an existing one.
                // Options: unchanged -- not capped by balance/inventory,
                // writing an option credits premium rather than requiring it
                // upfront. Futures/perps: capped by margin/buying power and
                // debits margin, same treatment as opening a long.
                if is_option {
                    let executed_quantity = requested_quantity;
                    let trade_value = fill.price * executed_quantity * multiplier;
                    let fee_amount = fill.fee_amount;
                    self.fees_paid += fee_amount;
                    self.balance += trade_value;
                    self.balance -= fee_amount;
                    self.net_position -= executed_quantity;
                    self.add_to_short_position_with_instrument(
                        &symbol, executed_quantity, fill.price, fill.timestamp,
                        Some(instrument.clone()), 0.0,
                    );
                    Ok(executed_quantity)
                } else {
                    if fill.price <= 0.0 {
                        return Ok(0.0);
                    }
                    let available = ((self.balance * 0.95 * leverage) / (fill.price * multiplier)).max(0.0);
                    let executed_quantity = requested_quantity.min(available);
                    if executed_quantity < 1e-10 {
                        return Ok(0.0);
                    }
                    let execution_ratio = executed_quantity / requested_quantity;
                    let trade_value = fill.price * executed_quantity * multiplier;
                    let margin_posted = trade_value / leverage;
                    let fee_amount = fill.fee_amount * execution_ratio;
                    self.fees_paid += fee_amount;
                    self.balance -= margin_posted;
                    self.balance -= fee_amount;
                    self.net_position -= executed_quantity;
                    self.add_to_short_position_with_instrument(
                        &symbol, executed_quantity, fill.price, fill.timestamp,
                        Some(instrument.clone()), margin_posted,
                    );
                    Ok(executed_quantity)
                }
            }
        }
    }

    /// Quantity of open Short positions on `symbol` available to buy-to-cover.
    /// Parallel to [`get_available_quantity`]'s `BookSide::Ask` (sell-to-close-
    /// long) branch, but for the short side.
    fn get_available_short_quantity(&self, symbol: &str) -> f64 {
        self.positions.iter()
            .filter(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Short)
            .map(|p| p.quantity.max(0.0))
            .sum()
    }

    /// Like [`add_to_long_position_with_instrument`] but opens/adds to a Short
    /// position. Weighted-average entry price, same as the long side.
    /// `margin_posted` is summed into the position's running total (`0.0`
    /// for option writes, which use full-premium accounting instead).
    fn add_to_short_position_with_instrument(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
        instrument: Option<DerivativeMetadata>,
        margin_posted: f64,
    ) {
        let existing = self.positions.iter_mut()
            .find(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Short);

        match existing {
            Some(position) => {
                let old_value = position.entry_price * position.quantity;
                let new_value = price * quantity;
                let new_quantity = position.quantity + quantity;
                if new_quantity > 0.0 {
                    position.entry_price = (old_value + new_value) / new_quantity;
                }
                position.quantity = new_quantity;
                position.mark_price = Some(price);
                position.margin_posted += margin_posted;
                if position.instrument.is_none() {
                    position.instrument = instrument;
                }
            }
            None => {
                self.positions.push(Position {
                    symbol: symbol.to_string(),
                    side: PositionSide::Short,
                    quantity,
                    entry_price: price,
                    mark_price: Some(price),
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    open_time: timestamp,
                    close_time: None,
                    instrument,
                    greeks: None,
                    margin_posted,
                });
            }
        }
    }

    /// Reduce (buy-to-cover) a Short position. Returns `(realized_pnl,
    /// margin_released)` -- for a short, realized P&L is `(entry_price -
    /// cover_price) * quantity` (profit when bought back cheaper than the
    /// premium/margin originally posted); `margin_released` is the slice of
    /// posted margin proportional to the fraction covered (see
    /// [`reduce_long_position`](Self::reduce_long_position)'s doc comment).
    fn reduce_short_position(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
    ) -> (f64, f64) {
        let mut remaining_to_cover = quantity;
        let mut total_realized_pnl = 0.0;
        let mut total_margin_released = 0.0;

        for position in self.positions.iter_mut()
            .filter(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Short)
        {
            if remaining_to_cover <= 0.0 {
                break;
            }
            let cover_from_this = remaining_to_cover.min(position.quantity);
            if cover_from_this > 0.0 {
                let pnl = (position.entry_price - price) * cover_from_this;
                let margin_fraction = if position.quantity > 0.0 { cover_from_this / position.quantity } else { 0.0 };
                let margin_released = position.margin_posted * margin_fraction;

                position.realized_pnl += pnl;
                total_realized_pnl += pnl;
                position.margin_posted -= margin_released;
                total_margin_released += margin_released;

                position.quantity -= cover_from_this;
                remaining_to_cover -= cover_from_this;
                if position.quantity < 1e-10 {
                    position.quantity = 0.0;
                    position.close_time = Some(timestamp);
                }
            }
        }

        (total_realized_pnl, total_margin_released)
    }

    /// Like [`add_to_long_position`] but tags a new position with derivative metadata.
    /// Existing positions keep their existing `instrument` field unchanged.
    /// `margin_posted` is summed into the position's running total (`0.0`
    /// for option purchases, which use full-premium accounting instead).
    fn add_to_long_position_with_instrument(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
        instrument: Option<DerivativeMetadata>,
        margin_posted: f64,
    ) {
        let existing = self.positions.iter_mut()
            .find(|p| p.symbol == symbol && p.close_time.is_none() && p.side == PositionSide::Long);

        match existing {
            Some(position) => {
                let old_value = position.entry_price * position.quantity;
                let new_value = price * quantity;
                let new_quantity = position.quantity + quantity;
                if new_quantity > 0.0 {
                    position.entry_price = (old_value + new_value) / new_quantity;
                }
                position.quantity = new_quantity;
                position.mark_price = Some(price);
                position.margin_posted += margin_posted;
                if position.instrument.is_none() {
                    position.instrument = instrument;
                }
            }
            None => {
                self.positions.push(Position {
                    symbol: symbol.to_string(),
                    side: PositionSide::Long,
                    quantity,
                    entry_price: price,
                    mark_price: Some(price),
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    open_time: timestamp,
                    close_time: None,
                    instrument,
                    greeks: None,
                    margin_posted,
                });
            }
        }
    }

    /// Recompute Black-Scholes Greeks and mark price for the open position on `symbol`.
    ///
    /// No-op for positions without `DerivativeMetadata` or for non-option instruments.
    /// Call this whenever you have a new IV quote for the instrument.
    ///
    /// - `spot` — current underlying price.
    /// - `rate` — continuously compounded risk-free rate (e.g. `0.05`).
    /// - `vol`  — current implied volatility, annualised (e.g. `0.80`).
    /// - `as_of` — the point in time "now" means. Callers driving a live feed
    ///   pass `Utc::now()`; a backtest MUST pass the simulated bar's own
    ///   timestamp, not wall-clock time, or every time-to-expiry computation
    ///   silently measures against the real present instead of the simulated
    ///   date.
    pub fn update_position_greeks(&mut self, symbol: &str, spot: f64, rate: f64, vol: f64, as_of: DateTime<Utc>) {
        let now = as_of;
        for position in self.positions.iter_mut()
            .filter(|p| p.symbol == symbol && p.close_time.is_none())
        {
            if let Some(instr) = &position.instrument {
                if let Some(tte) = instr.instrument_kind.time_to_expiry(now) {
                    if tte <= 0.0 {
                        continue;
                    }
                    let g = bs_greeks(&instr.instrument_kind, spot, rate, vol, tte);
                    let mark = bs_price(&instr.instrument_kind, spot, rate, vol, tte);
                    position.greeks = Some(g);
                    if mark > 0.0 {
                        let multiplier = instr.contract_multiplier;
                        let pnl = match position.side {
                            PositionSide::Long => {
                                (mark - position.entry_price) * position.quantity * multiplier
                            }
                            PositionSide::Short => {
                                (position.entry_price - mark) * position.quantity * multiplier
                            }
                        };
                        position.mark_price = Some(mark);
                        position.unrealized_pnl = pnl;
                    }
                }
            }
        }
    }

    /// Aggregate Greeks across all open positions that have Greeks populated.
    ///
    /// Call [`update_position_greeks`] for each instrument before calling this to
    /// ensure the result reflects current market conditions.
    pub fn get_portfolio_greeks(&self) -> Greeks {
        let inputs: Vec<(f64, f64, Greeks)> = self.positions.iter()
            .filter(|p| p.close_time.is_none() && p.greeks.is_some())
            .map(|p| {
                let multiplier = p.instrument.as_ref()
                    .map(|i| i.contract_multiplier)
                    .unwrap_or(1.0);
                (p.quantity, multiplier, p.greeks.clone().unwrap())
            })
            .collect();
        aggregate_greeks(&inputs)
    }

    /// Return references to all open positions whose instrument expires within
    /// `within` of `as_of`. Useful for pre-expiry roll or exercise logic. `as_of`
    /// must be the simulated/current timestamp, not wall-clock time, in a
    /// backtest context (see [`update_position_greeks`]'s doc comment).
    pub fn get_expiring_positions(&self, as_of: DateTime<Utc>, within: chrono::Duration) -> Vec<&Position> {
        let cutoff = as_of + within;
        self.positions.iter()
            .filter(|p| p.close_time.is_none())
            .filter(|p| {
                p.instrument
                    .as_ref()
                    .and_then(|i| i.instrument_kind.expiry())
                    .map(|exp| exp <= cutoff)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Cash-settle every open option position on `symbol` whose expiry is at
    /// or before `as_of`. In-the-money contracts auto-exercise/get assigned
    /// for cash intrinsic value (`max(spot-strike,0)*multiplier` for calls,
    /// `max(strike-spot,0)*multiplier` for puts); out-of-the-money contracts
    /// expire worthless. No-op for non-option positions or contracts not yet
    /// expired.
    ///
    /// European-exercise-at-expiration only -- matches the Black-Scholes
    /// model this crate already uses for marking (`greeks::bs_price`/
    /// `bs_greeks` assume European exercise). Early/American-style assignment
    /// is not modeled; this is the standard backtesting approximation, not an
    /// oversight.
    pub fn settle_expired_options(&mut self, symbol: &str, spot: f64, as_of: DateTime<Utc>) {
        for position in self.positions.iter_mut()
            .filter(|p| p.symbol == symbol && p.close_time.is_none())
        {
            let Some(instrument) = &position.instrument else { continue };
            if !instrument.instrument_kind.is_option() {
                continue;
            }
            let Some(expiry) = instrument.instrument_kind.expiry() else { continue };
            if expiry > as_of {
                continue;
            }

            let intrinsic = match &instrument.instrument_kind {
                derivatives::InstrumentKind::Call { strike, .. } => (spot - strike).max(0.0),
                derivatives::InstrumentKind::Put { strike, .. } => (strike - spot).max(0.0),
                _ => 0.0,
            };
            let multiplier = instrument.contract_multiplier;
            let settlement_value = intrinsic * position.quantity * multiplier;

            // Long: receives intrinsic value (0 if OTM/worthless).
            // Short: pays out intrinsic value (assigned if the holder exercises).
            let pnl = match position.side {
                PositionSide::Long => {
                    self.balance += settlement_value;
                    settlement_value - position.entry_price * position.quantity * multiplier
                }
                PositionSide::Short => {
                    self.balance -= settlement_value;
                    position.entry_price * position.quantity * multiplier - settlement_value
                }
            };

            position.realized_pnl += pnl;
            position.mark_price = Some(intrinsic);
            position.unrealized_pnl = 0.0;
            position.close_time = Some(as_of);
            self.realized_pnl += pnl;
        }
    }

    /// Approximate margin/collateral required to carry all open short option
    /// positions, in quote currency. `spot_prices` maps each underlying symbol
    /// to its current price (only underlyings with an open short option need
    /// an entry).
    ///
    /// Uses the standard simplified Reg-T formula for naked options -- a real
    /// broker's margin rules are far more nuanced (portfolio margin, broker-
    /// specific add-ons) than any backtesting engine can replicate exactly;
    /// this is a reasonable, clearly-approximate estimate, not a live margin
    /// calculation:
    ///
    /// `max(0.20 * spot * multiplier - otm_amount * multiplier, 0.10 * strike * multiplier) + premium_received`
    ///
    /// A short call fully covered by an existing long position of at least
    /// the required shares in the same underlying needs zero additional
    /// margin (the shares themselves are the collateral). Cash-secured puts
    /// are NOT specially detected -- this engine doesn't track cash reserved
    /// specifically for a put's strike separately from the general balance,
    /// so a short put is always treated as naked here. That's a real
    /// simplification, called out rather than silently assumed away.
    pub fn margin_requirement(&self, spot_prices: &std::collections::HashMap<String, f64>) -> f64 {
        let mut total = 0.0;
        for position in self.positions.iter()
            .filter(|p| p.close_time.is_none() && p.side == PositionSide::Short)
        {
            let Some(instrument) = &position.instrument else { continue };
            if !instrument.instrument_kind.is_option() {
                continue;
            }
            let Some(&spot) = spot_prices.get(&instrument.underlying) else { continue };
            let multiplier = instrument.contract_multiplier;
            let strike = instrument.instrument_kind.strike().unwrap_or(0.0);

            let is_covered_call = matches!(instrument.instrument_kind, derivatives::InstrumentKind::Call { .. })
                && self.positions.iter().any(|p| {
                    p.symbol == instrument.underlying
                        && p.close_time.is_none()
                        && p.side == PositionSide::Long
                        && p.quantity >= position.quantity * multiplier
                });
            if is_covered_call {
                continue;
            }

            let otm_amount = match &instrument.instrument_kind {
                derivatives::InstrumentKind::Call { .. } => (strike - spot).max(0.0),
                derivatives::InstrumentKind::Put { .. } => (spot - strike).max(0.0),
                _ => 0.0,
            };
            let premium_received = position.entry_price * position.quantity * multiplier;
            let per_contract_margin = (0.20 * spot * multiplier - otm_amount * multiplier)
                .max(0.10 * strike * multiplier);
            total += (per_contract_margin * position.quantity + premium_received).max(0.0);
        }
        total
    }

    /// Get a summary of open positions for debugging
    pub fn get_position_summary(&self) -> String {
        let open_positions: Vec<_> = self.positions.iter()
            .filter(|p| p.close_time.is_none())
            .collect();
        
        if open_positions.is_empty() {
            return "No open positions".to_string();
        }
        
        open_positions.iter()
            .map(|p| format!("{} {:?} qty={:.8} entry={}", p.symbol, p.side, p.quantity, p.entry_price))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orderbook::LiquidityType;
    use std::sync::Arc;

    fn create_fill(side: BookSide, price: f64, quantity: f64) -> Fill {
        Fill {
            order_id: Arc::from("test"),
            side,
            price,
            base_price: None,
            quantity,
            liquidity_type: LiquidityType::Maker,
            fee_rate: 0.001,
            fee_amount: price * quantity * 0.001,
            slippage_bps: None,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_cannot_sell_more_than_owned() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        
        // Buy 1 BTC at $50,000 (well within $100k balance)
        let buy_fill = create_fill(BookSide::Bid, 50000.0, 1.0);
        portfolio.apply_fill(&buy_fill).unwrap();
        
        // Should have bought ~1 BTC (might be slightly less due to 95% balance limit)
        assert!(portfolio.net_position > 0.9, "Should have bought close to 1 BTC, got {}", portfolio.net_position);
        
        // Try to sell 2 BTC (should only sell what we have)
        let sell_fill = create_fill(BookSide::Ask, 51000.0, 2.0);
        portfolio.apply_fill(&sell_fill).unwrap();
        
        // Should have sold only what we had
        assert!(portfolio.net_position.abs() < 1e-10, "Net position should be ~0, got {}", portfolio.net_position);
        
        // Position should be closed, not negative
        let open_positions: Vec<_> = portfolio.positions.iter()
            .filter(|p| p.close_time.is_none())
            .collect();
        assert!(open_positions.is_empty(), "Should have no open positions");
        
        // No position should have negative quantity
        for pos in &portfolio.positions {
            assert!(pos.quantity >= 0.0, "Position quantity should never be negative: {}", pos.quantity);
        }
    }

    #[test]
    fn test_sell_without_position_opens_short() {
        // Regression: this used to be a documented no-op (selling with no
        // inventory silently executed zero) -- but that meant ordinary
        // directional strategies could never open a short at all, contradicting
        // the SDK's own documented `SELL = -1 // open or increase short
        // position` contract. Selling with no open long now opens a fresh
        // Short, margin-capped by buying power exactly like opening a long.
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let initial_balance = portfolio.balance;

        let sell_fill = create_fill(BookSide::Ask, 50000.0, 1.0);
        portfolio.apply_fill(&sell_fill).unwrap();

        assert!(portfolio.balance < initial_balance, "opening a short should debit margin from balance");
        assert!((portfolio.net_position - (-1.0)).abs() < 1e-6, "net_position should reflect the new short, got {}", portfolio.net_position);
        assert_eq!(portfolio.positions.len(), 1);
        assert_eq!(portfolio.positions[0].side, PositionSide::Short);
        assert!(portfolio.positions[0].margin_posted > 0.0);
    }

    #[test]
    fn test_buy_and_partial_sell() {
        let mut portfolio = PortfolioState::with_balance(200_000.0);
        
        // Buy 2 BTC at $50,000 (=$100k, within $200k balance)
        let buy_fill = create_fill(BookSide::Bid, 50000.0, 2.0);
        portfolio.apply_fill(&buy_fill).unwrap();
        
        assert!((portfolio.net_position - 2.0).abs() < 0.1, "Should have ~2 BTC, got {}", portfolio.net_position);
        
        // Sell 1 BTC at $51,000
        let sell_fill = create_fill(BookSide::Ask, 51000.0, 1.0);
        portfolio.apply_fill(&sell_fill).unwrap();
        
        // Should still have ~1 BTC
        assert!((portfolio.net_position - 1.0).abs() < 0.1, "Should have ~1 BTC, got {}", portfolio.net_position);
        
        // Realized P&L should be positive (sold higher than bought)
        assert!(portfolio.realized_pnl > 0.0);
    }

    #[test]
    fn test_cannot_buy_more_than_balance_allows() {
        let mut portfolio = PortfolioState::with_balance(10_000.0);
        
        // Try to buy 1 BTC at $50,000 (but only have $10k)
        let buy_fill = create_fill(BookSide::Bid, 50000.0, 1.0);
        portfolio.apply_fill(&buy_fill).unwrap();
        
        // Should only buy what we can afford: $10k * 0.95 / $50k = 0.19 BTC
        assert!(portfolio.net_position < 0.2, "Should only buy ~0.19 BTC with $10k, got {}", portfolio.net_position);
        assert!(portfolio.net_position > 0.1, "Should have bought some BTC, got {}", portfolio.net_position);
        
        // Balance should never go negative
        assert!(portfolio.balance >= 0.0, "Balance should never be negative: {}", portfolio.balance);
    }

    #[test]
    fn test_balance_never_goes_negative() {
        let mut portfolio = PortfolioState::with_balance(10_000.0);
        
        // Try to buy massive amounts repeatedly
        for _ in 0..10 {
            let buy_fill = create_fill(BookSide::Bid, 90000.0, 100.0); // Try to buy $9M worth
            portfolio.apply_fill(&buy_fill).unwrap();
        }
        
        // Balance should never go negative
        assert!(portfolio.balance >= -0.01, 
            "Balance should never be significantly negative: {}", portfolio.balance);
        
        // Position should be limited by what we could afford
        assert!(portfolio.net_position < 0.2, 
            "Position should be limited to what $10k can buy at $90k/BTC: {}", portfolio.net_position);
    }

    #[test]
    fn test_apply_fill_executed_reports_capped_quantity() {
        // Bug #34 regression: commission must be charged on the executed quantity,
        // not the requested quantity. apply_fill_executed surfaces exactly how much
        // executed so callers can scope fees correctly.

        // Buy capped by balance: $10k * 0.95 / $50k ≈ 0.19 BTC executed out of 1.0 requested.
        let mut portfolio = PortfolioState::with_balance(10_000.0);
        let buy_fill = create_fill(BookSide::Bid, 50000.0, 1.0);
        let executed = portfolio.apply_fill_executed(&buy_fill).unwrap();
        assert!(executed > 0.1 && executed < 0.2,
            "Executed buy should be capped to ~0.19 BTC, got {}", executed);
        assert!((executed - portfolio.net_position).abs() < 1e-9,
            "Reported executed qty {} should match position delta {}", executed, portfolio.net_position);

        // Sell with no inventory: opens a fresh Short (see
        // test_sell_without_position_opens_short), capped by margin/buying
        // power the same way the buy side is -- same ~0.19 BTC cap.
        let mut empty = PortfolioState::with_balance(10_000.0);
        let sell_fill = create_fill(BookSide::Ask, 50000.0, 1.0);
        let executed_sell = empty.apply_fill_executed(&sell_fill).unwrap();
        assert!(executed_sell > 0.1 && executed_sell < 0.2,
            "Executed short-open should be capped to ~0.19 BTC, got {}", executed_sell);
        assert!((executed_sell - empty.net_position.abs()).abs() < 1e-9,
            "Reported executed qty {} should match position delta {}", executed_sell, empty.net_position);
    }

    // ---- Options: apply_derivative_fill, settlement, margin ----

    fn call_instrument(strike: f64, expiry: DateTime<Utc>) -> DerivativeMetadata {
        DerivativeMetadata::new(
            "O:SPY-TEST-C",
            "SPY",
            derivatives::InstrumentKind::Call { strike, expiry },
            100.0,
            "USD",
            "test",
        )
    }

    fn put_instrument(strike: f64, expiry: DateTime<Utc>) -> DerivativeMetadata {
        DerivativeMetadata::new(
            "O:SPY-TEST-P",
            "SPY",
            derivatives::InstrumentKind::Put { strike, expiry },
            100.0,
            "USD",
            "test",
        )
    }

    #[test]
    fn apply_derivative_fill_buy_to_open_long_debits_premium_times_multiplier() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        let instrument = call_instrument(450.0, expiry);
        let fill = create_fill(BookSide::Bid, 5.0, 1.0); // $5 premium, 1 contract
        let executed = portfolio.apply_derivative_fill(&fill, &instrument).unwrap();

        assert_eq!(executed, 1.0);
        // $5 premium * 1 contract * 100 multiplier = $500 debited (plus fee).
        assert!((100_000.0 - portfolio.balance - 500.0).abs() < 1.0,
            "balance should be debited ~$500 for the premium, got balance={}", portfolio.balance);
        assert_eq!(portfolio.positions.len(), 1);
        assert_eq!(portfolio.positions[0].side, PositionSide::Long);
    }

    #[test]
    fn apply_derivative_fill_sell_to_open_short_credits_premium_uncapped() {
        // Writing/selling a naked option has no existing long inventory to cap
        // against -- this must NOT silently execute as zero the way the plain
        // apply_fill_executed path would (it only ever reduces an existing long).
        let mut portfolio = PortfolioState::with_balance(10_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        let instrument = call_instrument(450.0, expiry);
        let fill = create_fill(BookSide::Ask, 5.0, 2.0); // write 2 contracts at $5
        let executed = portfolio.apply_derivative_fill(&fill, &instrument).unwrap();

        assert_eq!(executed, 2.0, "writing a new short must not be capped to 0 by 'no inventory'");
        assert!(portfolio.balance > 10_000.0, "premium received should credit balance, got {}", portfolio.balance);
        assert_eq!(portfolio.positions.len(), 1);
        assert_eq!(portfolio.positions[0].side, PositionSide::Short);
        assert_eq!(portfolio.positions[0].quantity, 2.0);
    }

    #[test]
    fn apply_derivative_fill_buy_to_cover_short_realizes_pnl() {
        let mut portfolio = PortfolioState::with_balance(10_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        let instrument = call_instrument(450.0, expiry);

        // Write at $5, buy back at $2 -- should realize a profit.
        let open_fill = create_fill(BookSide::Ask, 5.0, 1.0);
        portfolio.apply_derivative_fill(&open_fill, &instrument).unwrap();
        let cover_fill = create_fill(BookSide::Bid, 2.0, 1.0);
        let executed = portfolio.apply_derivative_fill(&cover_fill, &instrument).unwrap();

        assert_eq!(executed, 1.0);
        assert!(portfolio.realized_pnl > 0.0, "buying back cheaper than premium received should realize a profit, got {}", portfolio.realized_pnl);
        assert!(portfolio.positions[0].close_time.is_some(), "fully covered short should be closed");
    }

    // ---- Leverage/margin ----

    fn perpetual_instrument() -> DerivativeMetadata {
        DerivativeMetadata::new(
            "BTC-PERP", "BTC", derivatives::InstrumentKind::Perpetual, 1.0, "USD", "test",
        )
    }

    #[test]
    fn apply_fill_executed_leverage_one_matches_pre_leverage_debit() {
        // The critical parity property: leverage=1.0 must debit the FULL
        // trade value, exactly like this method did before leverage existed.
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        assert_eq!(portfolio.leverage, 1.0);
        let buy_fill = create_fill(BookSide::Bid, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&buy_fill).unwrap();

        let expected_debit = 50_000.0 + buy_fill.fee_amount;
        assert!((portfolio.balance - (100_000.0 - expected_debit)).abs() < 1e-6);
        assert!((portfolio.positions[0].margin_posted - 50_000.0).abs() < 1e-6);
    }

    #[test]
    fn apply_fill_executed_leverage_scales_notional_debits_only_margin() {
        // At 4x leverage, opening the SAME quantity only debits 1/4 the cash
        // (margin), not the full notional -- see portfoliomanager::margin.
        let mut portfolio = PortfolioState::with_balance(100_000.0).with_leverage(4.0);
        let buy_fill = create_fill(BookSide::Bid, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&buy_fill).unwrap();

        let expected_margin = 50_000.0 / 4.0;
        assert!((portfolio.positions[0].margin_posted - expected_margin).abs() < 1e-6);
        let expected_debit = expected_margin + buy_fill.fee_amount;
        assert!((portfolio.balance - (100_000.0 - expected_debit)).abs() < 1e-6);
    }

    #[test]
    fn apply_fill_executed_leverage_increases_buying_power() {
        // Buying power scales with leverage: at 4x, the same $10k balance
        // should afford ~4x the quantity it would unleveraged.
        let mut unleveraged = PortfolioState::with_balance(10_000.0);
        let mut leveraged = PortfolioState::with_balance(10_000.0).with_leverage(4.0);
        let fill = create_fill(BookSide::Bid, 50_000.0, 10.0); // far more than either can afford
        let executed_1x = unleveraged.apply_fill_executed(&fill).unwrap();
        let executed_4x = leveraged.apply_fill_executed(&fill).unwrap();

        assert!((executed_4x - executed_1x * 4.0).abs() < 1e-6,
            "4x leverage should afford ~4x the quantity: 1x={executed_1x}, 4x={executed_4x}");
    }

    #[test]
    fn apply_fill_executed_closing_leveraged_position_releases_margin_plus_pnl() {
        let mut portfolio = PortfolioState::with_balance(100_000.0).with_leverage(2.0);
        let buy_fill = create_fill(BookSide::Bid, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&buy_fill).unwrap();
        let balance_after_open = portfolio.balance;
        let margin_posted = portfolio.positions[0].margin_posted;
        assert!((margin_posted - 25_000.0).abs() < 1e-6); // 50,000/2

        // Close at a $1,000 profit (price rose from 50,000 to 51,000).
        let sell_fill = create_fill(BookSide::Ask, 51_000.0, 1.0);
        portfolio.apply_fill_executed(&sell_fill).unwrap();

        let expected_credit = margin_posted + 1_000.0 - sell_fill.fee_amount;
        assert!((portfolio.balance - (balance_after_open + expected_credit)).abs() < 1e-6,
            "closing should return margin + pnl, not the full notional");
        assert!(portfolio.positions[0].close_time.is_some());
    }

    #[test]
    fn apply_derivative_fill_future_short_requires_margin_not_premium_credit() {
        // Unlike writing an OPTION (credits premium, uncapped), shorting a
        // future/perp requires posting margin -- same treatment as a long.
        let mut portfolio = PortfolioState::with_balance(100_000.0).with_leverage(2.0);
        let instrument = perpetual_instrument();
        let initial_balance = portfolio.balance;

        let short_fill = create_fill(BookSide::Ask, 50_000.0, 1.0);
        portfolio.apply_derivative_fill(&short_fill, &instrument).unwrap();

        assert!(portfolio.balance < initial_balance,
            "opening a future short should DEBIT margin, not credit premium");
        let expected_margin = 50_000.0 / 2.0;
        assert!((portfolio.positions[0].margin_posted - expected_margin).abs() < 1e-6);
    }

    #[test]
    fn apply_derivative_fill_option_unaffected_by_leverage_setting() {
        // Options keep their pre-existing full-premium accounting regardless
        // of the portfolio's leverage setting -- leverage is unrelated to
        // this feature (see module docs on apply_derivative_fill).
        let mut leveraged = PortfolioState::with_balance(100_000.0).with_leverage(5.0);
        let mut unleveraged = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        let instrument = call_instrument(450.0, expiry);

        let buy_fill = create_fill(BookSide::Bid, 5.0, 1.0);
        leveraged.apply_derivative_fill(&buy_fill, &instrument).unwrap();
        unleveraged.apply_derivative_fill(&buy_fill, &instrument).unwrap();

        assert!((leveraged.balance - unleveraged.balance).abs() < 1e-9,
            "option premium debit should be identical regardless of leverage");
        assert_eq!(leveraged.positions[0].margin_posted, 0.0);
    }

    #[test]
    fn get_total_value_matches_equity_at_for_a_simple_leveraged_long() {
        let mut portfolio = PortfolioState::with_balance(100_000.0).with_leverage(3.0);
        let buy_fill = create_fill(BookSide::Bid, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&buy_fill).unwrap();
        portfolio.update_unrealized_pnl(52_000.0);

        let total_value = portfolio.get_total_value();
        let equity_at = portfolio.equity_at(52_000.0);
        assert!((total_value - equity_at).abs() < 1e-6,
            "get_total_value and equity_at should agree: {total_value} vs {equity_at}");
    }

    #[test]
    fn margin_used_sums_across_open_positions_only() {
        let mut portfolio = PortfolioState::with_balance(200_000.0).with_leverage(2.0);
        let buy_fill = create_fill(BookSide::Bid, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&buy_fill).unwrap();
        assert!((portfolio.margin_used() - 25_000.0).abs() < 1e-6);

        // Fully close it -- margin_used should drop back to zero.
        let sell_fill = create_fill(BookSide::Ask, 50_000.0, 1.0);
        portfolio.apply_fill_executed(&sell_fill).unwrap();
        assert!(portfolio.margin_used().abs() < 1e-6);
    }

    #[test]
    fn settle_expired_options_itm_call_pays_intrinsic_value() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() - chrono::Duration::days(1); // already expired
        let instrument = call_instrument(450.0, expiry);
        let fill = create_fill(BookSide::Bid, 5.0, 1.0);
        portfolio.apply_derivative_fill(&fill, &instrument).unwrap();

        let balance_before = portfolio.balance;
        portfolio.settle_expired_options("O:SPY-TEST-C", 460.0, Utc::now());

        // ITM by $10, * 100 multiplier = $1000 settlement credit.
        assert!((portfolio.balance - balance_before - 1000.0).abs() < 1.0,
            "ITM call should settle for $1000 intrinsic value, balance delta={}", portfolio.balance - balance_before);
        assert!(portfolio.positions[0].close_time.is_some());
    }

    #[test]
    fn settle_expired_options_otm_put_expires_worthless() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() - chrono::Duration::days(1);
        let instrument = put_instrument(400.0, expiry); // spot 460 > strike 400 => OTM put
        let fill = create_fill(BookSide::Bid, 5.0, 1.0);
        portfolio.apply_derivative_fill(&fill, &instrument).unwrap();

        let balance_before = portfolio.balance;
        portfolio.settle_expired_options("O:SPY-TEST-P", 460.0, Utc::now());

        assert_eq!(portfolio.balance, balance_before, "OTM option should settle for zero, not credit/debit anything");
        assert!(portfolio.positions[0].realized_pnl < 0.0, "long OTM option should realize the premium paid as a loss");
    }

    #[test]
    fn settle_expired_options_is_noop_before_expiry() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30); // not yet expired
        let instrument = call_instrument(450.0, expiry);
        let fill = create_fill(BookSide::Bid, 5.0, 1.0);
        portfolio.apply_derivative_fill(&fill, &instrument).unwrap();

        portfolio.settle_expired_options("O:SPY-TEST-C", 460.0, Utc::now());
        assert!(portfolio.positions[0].close_time.is_none(), "should not settle a contract that hasn't expired yet");
    }

    #[test]
    fn margin_requirement_is_zero_for_covered_call() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        // Own 100 shares of SPY (covers 1 short call contract at multiplier 100).
        let shares_fill = create_fill(BookSide::Bid, 440.0, 100.0);
        portfolio.apply_fill_executed(&shares_fill).unwrap();
        // "BTC/USD" is apply_fill_executed's hardcoded symbol (pre-existing,
        // unrelated to options) -- rename the resulting position to SPY so the
        // covered-call underlying match can find it.
        portfolio.positions[0].symbol = "SPY".to_string();

        let instrument = call_instrument(450.0, expiry);
        let short_fill = create_fill(BookSide::Ask, 5.0, 1.0);
        portfolio.apply_derivative_fill(&short_fill, &instrument).unwrap();

        let mut spot_prices = std::collections::HashMap::new();
        spot_prices.insert("SPY".to_string(), 440.0);
        assert_eq!(portfolio.margin_requirement(&spot_prices), 0.0,
            "a short call fully covered by >= 100 shares of the underlying needs no additional margin");
    }

    #[test]
    fn margin_requirement_is_positive_for_naked_short_call() {
        let mut portfolio = PortfolioState::with_balance(100_000.0);
        let expiry = Utc::now() + chrono::Duration::days(30);
        let instrument = call_instrument(450.0, expiry);
        let short_fill = create_fill(BookSide::Ask, 5.0, 1.0);
        portfolio.apply_derivative_fill(&short_fill, &instrument).unwrap();

        let mut spot_prices = std::collections::HashMap::new();
        spot_prices.insert("SPY".to_string(), 440.0);
        assert!(portfolio.margin_requirement(&spot_prices) > 0.0,
            "a naked short call must require positive margin");
    }
}
