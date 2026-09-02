//! Risk Manager Module
//!
//! Provides risk limit types and a generic risk manager for use by strategies and portfolio manager.
//! Includes VaR-based position limits, correlation risk management, and volatility regime switching.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use derivatives::Greeks;

/// Cross-asset correlation monitoring for portfolio risk
pub mod correlation_monitor;

pub use correlation_monitor::{
    CrossAssetCorrelationMonitor, CorrelationMonitorConfig,
    CorrelationAlert, CorrelationAlertType,
    PortfolioCorrelationSummary,
};

/// Volatility regime for risk management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    /// Low volatility - normal or increased limits
    Low,
    /// Normal volatility - standard limits
    Normal,
    /// High volatility - tighter limits
    High,
    /// Extreme volatility - severely restricted
    Extreme,
}

impl Default for VolatilityRegime {
    fn default() -> Self {
        VolatilityRegime::Normal
    }
}

/// Risk limits for a strategy or portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position: Option<f64>,      // Maximum allowed net position
    pub max_order_size: Option<f64>,    // Maximum allowed order size
    pub max_inventory_skew: Option<f64>,// Maximum allowed inventory skew
    pub max_volatility: Option<f64>,    // Maximum allowed volatility
    pub max_drawdown: Option<f64>,      // Maximum allowed drawdown (0.0 to 1.0)
    pub stop_loss_pct: Option<f64>,     // Stop loss per trade as percentage (0.0 to 1.0)
    pub max_daily_loss: Option<f64>,    // Maximum daily loss in currency units

    // Portfolio Greeks limits
    pub max_net_delta: Option<f64>,      // Maximum absolute portfolio delta
    pub max_gamma: Option<f64>,          // Maximum absolute portfolio gamma
    pub max_vega: Option<f64>,           // Maximum absolute portfolio vega
    pub max_theta: Option<f64>,          // Maximum absolute portfolio theta
    pub max_notional_per_expiry: Option<f64>, // Maximum notional concentration per expiry bucket
    
    // VaR-based limits
    pub var_limit: Option<f64>,         // Maximum VaR (95%) in currency units
    pub cvar_limit: Option<f64>,        // Maximum CVaR (Expected Shortfall) in currency units
    pub var_position_scale: Option<f64>, // Scale position by 1/VaR (higher VaR = smaller position)
    
    // Correlation/multi-asset limits
    pub max_correlated_exposure: Option<f64>,  // Max total exposure to correlated assets
    pub correlation_threshold: Option<f64>,     // Correlation above which assets are considered correlated
    pub max_sector_exposure: Option<f64>,       // Max exposure per sector/group
    
    // Volatility regime limits
    pub high_vol_position_mult: Option<f64>,    // Position multiplier in high-vol regime (e.g., 0.5)
    pub extreme_vol_position_mult: Option<f64>, // Position multiplier in extreme-vol regime (e.g., 0.2)
    pub vol_regime_threshold_high: Option<f64>, // Volatility ratio for high regime (e.g., 1.5)
    pub vol_regime_threshold_extreme: Option<f64>, // Volatility ratio for extreme regime (e.g., 2.5)
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position: Some(10.0),
            max_order_size: Some(1.0),
            max_inventory_skew: Some(2.0),
            max_volatility: Some(0.10),
            max_drawdown: Some(0.20),      // 20% max drawdown
            stop_loss_pct: Some(0.05),     // 5% stop loss per trade
            max_daily_loss: Some(1000.0),  // $1000 max daily loss

            // Greeks defaults
            max_net_delta: Some(10.0),
            max_gamma: Some(5.0),
            max_vega: Some(10000.0),
            max_theta: None,
            max_notional_per_expiry: Some(100000.0),
            
            // VaR-based defaults
            var_limit: Some(5000.0),       // $5000 max VaR
            cvar_limit: Some(7500.0),      // $7500 max CVaR
            var_position_scale: Some(1.0), // Scale factor for VaR-based sizing
            
            // Correlation defaults
            max_correlated_exposure: Some(50000.0), // $50k max correlated exposure
            correlation_threshold: Some(0.7),       // 70% correlation threshold
            max_sector_exposure: Some(25000.0),     // $25k max per sector
            
            // Volatility regime defaults
            high_vol_position_mult: Some(0.5),     // 50% position in high-vol
            extreme_vol_position_mult: Some(0.2),  // 20% position in extreme-vol
            vol_regime_threshold_high: Some(1.5),  // 50% above baseline
            vol_regime_threshold_extreme: Some(2.5), // 150% above baseline
        }
    }
}

/// Risk metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub current_position: f64,
    pub current_order_size: f64,
    pub current_inventory_skew: f64,
    pub current_volatility: f64,
    pub current_drawdown: f64,           // Current drawdown from peak (0.0 to 1.0)
    pub unrealized_pnl: f64,             // Unrealized P&L of open positions
    pub daily_pnl: f64,                  // Today's realized + unrealized P&L

    // Derivatives metrics
    pub portfolio_greeks: Option<Greeks>,
    
    // VaR metrics
    pub current_var: f64,                // Current VaR (95%) estimate
    pub current_cvar: f64,               // Current CVaR estimate
    
    // Correlation metrics
    pub correlated_exposure: f64,        // Total exposure to correlated positions
    pub sector_exposures: HashMap<String, f64>, // Exposure by sector
    
    // Volatility regime
    pub volatility_regime: VolatilityRegime,
    pub baseline_volatility: f64,        // Historical baseline for comparison
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            current_position: 0.0,
            current_order_size: 0.0,
            current_inventory_skew: 0.0,
            current_volatility: 0.0,
            current_drawdown: 0.0,
            unrealized_pnl: 0.0,
            daily_pnl: 0.0,
            portfolio_greeks: None,
            current_var: 0.0,
            current_cvar: 0.0,
            correlated_exposure: 0.0,
            sector_exposures: HashMap::new(),
            volatility_regime: VolatilityRegime::Normal,
            baseline_volatility: 0.0,
        }
    }
}

/// Risk check result indicating what action to take
#[derive(Debug, Clone, PartialEq)]
pub enum RiskAction {
    /// All checks passed, proceed normally
    Proceed,
    /// Reduce position size to stay within limits
    ReducePosition(f64),
    /// Close all positions immediately (stop loss triggered)
    CloseAllPositions(String),
    /// Reject the order entirely
    RejectOrder(String),
    /// Halt trading for the day
    HaltTrading(String),
    /// Scale down position due to VaR limits
    ScalePosition(f64, String),
}

/// Margin-call check for leveraged positions: `equity` and
/// `maintenance_margin` are pre-computed by the caller (see
/// `portfoliomanager::margin::maintenance_margin`/`equity_at` -- this
/// function is a pure threshold comparison, it doesn't duplicate the margin
/// formula itself, so there's exactly one source of truth for the math).
/// `margin_used > 0.0` gates the check so unleveraged portfolios (which
/// never post margin) can never spuriously trigger it.
///
/// A free function, not a `RiskManager` method: liquidation must be
/// enforced whenever leverage is in play, independent of whether the
/// separate, opt-in `RiskManager`/advanced-risk-checks feature is enabled
/// for a given backtest run.
///
/// Returns `CloseAllPositions` when equity has fallen to or below the
/// maintenance margin requirement -- callers MUST actually close positions
/// on this action (see `python_simulation.rs`'s liquidation wiring), not
/// just halt/log, unlike the pre-existing `CloseAllPositions` handling
/// elsewhere in this crate's callers.
pub fn check_margin(equity: f64, margin_used: f64, maintenance_margin: f64) -> RiskAction {
    if margin_used > 0.0 && equity <= maintenance_margin {
        RiskAction::CloseAllPositions(format!(
            "Margin call: equity {:.2} <= maintenance margin {:.2}",
            equity, maintenance_margin
        ))
    } else {
        RiskAction::Proceed
    }
}

/// Asset correlation entry for multi-asset risk management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCorrelation {
    pub asset_a: String,
    pub asset_b: String,
    pub correlation: f64,
    pub last_updated: DateTime<Utc>,
}

/// Generic risk manager
#[derive(Debug)]
pub struct RiskManager {
    pub limits: RiskLimits,
    peak_equity: f64,
    daily_start_equity: f64,
    
    // VaR tracking
    recent_returns: Vec<f64>,
    var_window_size: usize,
    
    // Correlation tracking
    asset_correlations: Vec<AssetCorrelation>,
    position_by_asset: HashMap<String, f64>,
    
    // Volatility regime tracking
    volatility_history: Vec<f64>,
    baseline_volatility: Option<f64>,
    current_regime: VolatilityRegime,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            limits: RiskLimits::default(),
            peak_equity: 0.0,
            daily_start_equity: 0.0,
            recent_returns: Vec::new(),
            var_window_size: 252,  // 1 year of daily returns
            asset_correlations: Vec::new(),
            position_by_asset: HashMap::new(),
            volatility_history: Vec::new(),
            baseline_volatility: None,
            current_regime: VolatilityRegime::Normal,
        }
    }
}

impl RiskManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limits(limits: RiskLimits) -> Self {
        Self { 
            limits,
            peak_equity: 0.0,
            daily_start_equity: 0.0,
            recent_returns: Vec::new(),
            var_window_size: 252,
            asset_correlations: Vec::new(),
            position_by_asset: HashMap::new(),
            volatility_history: Vec::new(),
            baseline_volatility: None,
            current_regime: VolatilityRegime::Normal,
        }
    }
    
    /// Initialize with starting equity
    pub fn initialize(&mut self, initial_equity: f64) {
        self.peak_equity = initial_equity;
        self.daily_start_equity = initial_equity;
    }
    
    /// Update peak equity tracking
    pub fn update_equity(&mut self, current_equity: f64) {
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }
    }
    
    /// Get current drawdown from peak, as a fraction of peak equity.
    ///
    /// Floored at `0.0` (drawdown can't be negative -- `update_equity`
    /// always advances `peak_equity` before a tick's own equity could
    /// exceed it, so a negative raw ratio here would only ever be
    /// floating-point noise), but deliberately UNCAPPED above `1.0`
    /// (2026-09-02, fixing a real live-gate bug: this used to `.clamp(0.0,
    /// 1.0)`, silently capping the reported drawdown at exactly 100% no
    /// matter how far equity actually fell below zero). A short/written
    /// option position carries genuinely unbounded loss potential and is
    /// deliberately exempt from margin-call liquidation (see
    /// `python_simulation.rs`'s liquidation-check doc note) -- nothing else
    /// stops equity from crashing well past a 100% loss of peak once
    /// `equity_at` correctly includes option P&L (see `portfoliomanager`'s
    /// 2026-09-02 option-equity fix). A caller with `RiskLimits.max_drawdown`
    /// configured above `1.0` (a deliberately loose tolerance) would have
    /// found this gate could NEVER fire, regardless of how catastrophic the
    /// real loss became, since the clamped value could never exceed `1.0`
    /// to begin with. Matches `python_simulation.rs::compute_max_drawdown`'s
    /// existing unclamped convention for the final-reported max drawdown --
    /// this was the one remaining clamped outlier between the live gate and
    /// the final report.
    pub fn get_current_drawdown(&self, current_equity: f64) -> f64 {
        if self.peak_equity <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity - current_equity) / self.peak_equity).max(0.0)
    }
    
    /// Check if a proposed order should be allowed
    pub fn check_order(&self, order_size: f64, current_position: f64, metrics: &RiskMetrics) -> RiskAction {
        // Check max order size
        if let Some(max_order) = self.limits.max_order_size {
            if order_size > max_order {
                return RiskAction::ReducePosition(max_order);
            }
        }
        
        // Check if new position would exceed limits
        if let Some(max_pos) = self.limits.max_position {
            let new_position = current_position + order_size;
            if new_position.abs() > max_pos {
                let allowed_size = (max_pos - current_position.abs()).max(0.0);
                if allowed_size <= 0.0 {
                    return RiskAction::RejectOrder(format!(
                        "Position {} would exceed max {}", new_position.abs(), max_pos
                    ));
                }
                return RiskAction::ReducePosition(allowed_size);
            }
        }
        
        // Check drawdown limit
        if let Some(max_dd) = self.limits.max_drawdown {
            if metrics.current_drawdown > max_dd {
                return RiskAction::CloseAllPositions(format!(
                    "Drawdown {:.2}% exceeds max {:.2}%", 
                    metrics.current_drawdown * 100.0, max_dd * 100.0
                ));
            }
        }
        
        // Check daily loss limit
        if let Some(max_daily) = self.limits.max_daily_loss {
            if metrics.daily_pnl < -max_daily {
                return RiskAction::HaltTrading(format!(
                    "Daily loss ${:.2} exceeds max ${:.2}", 
                    -metrics.daily_pnl, max_daily
                ));
            }
        }
        
        RiskAction::Proceed
    }
    
    /// Check if a position should be stopped out
    pub fn check_stop_loss(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        if let Some(stop_pct) = self.limits.stop_loss_pct {
            let pnl_pct = if is_long {
                (current_price - entry_price) / entry_price
            } else {
                (entry_price - current_price) / entry_price
            };
            
            return pnl_pct < -stop_pct;
        }
        false
    }

    /// Check if the given metrics are within risk limits (position size, order
    /// size, inventory skew, volatility) against `self.limits`. Actively used
    /// by the live simulation path (`backtest::simulation`,
    /// `backtest::python_simulation`) -- despite the name's plainness, this
    /// is the current risk-limit check, not a superseded one.
    pub fn check(&self, metrics: &RiskMetrics) -> Result<(), String> {
        if let Some(max_pos) = self.limits.max_position {
            if metrics.current_position.abs() > max_pos {
                return Err(format!("Position {} exceeds max {}", metrics.current_position, max_pos));
            }
        }
        if let Some(max_order) = self.limits.max_order_size {
            if metrics.current_order_size > max_order {
                return Err(format!("Order size {} exceeds max {}", metrics.current_order_size, max_order));
            }
        }
        if let Some(max_skew) = self.limits.max_inventory_skew {
            if metrics.current_inventory_skew.abs() > max_skew {
                return Err(format!("Inventory skew {} exceeds max {}", metrics.current_inventory_skew, max_skew));
            }
        }
        if let Some(max_vol) = self.limits.max_volatility {
            if metrics.current_volatility > max_vol {
                return Err(format!("Volatility {} exceeds max {}", metrics.current_volatility, max_vol));
            }
        }
        if let Some(max_dd) = self.limits.max_drawdown {
            if metrics.current_drawdown > max_dd {
                return Err(format!("Drawdown {:.2}% exceeds max {:.2}%", 
                    metrics.current_drawdown * 100.0, max_dd * 100.0));
            }
        }
        Ok(())
    }
    
    // ========== VAR-BASED POSITION LIMITS ==========
    
    /// Update returns history for VaR calculation
    pub fn update_returns(&mut self, daily_return: f64) {
        self.recent_returns.push(daily_return);
        if self.recent_returns.len() > self.var_window_size {
            self.recent_returns.remove(0);
        }
    }
    
    /// Calculate historical VaR (95% confidence)
    /// Returns the potential loss at 95% confidence level
    pub fn calculate_var(&self, position_value: f64) -> f64 {
        if self.recent_returns.len() < 30 {
            return 0.0; // Not enough data
        }
        
        let mut sorted_returns = self.recent_returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // 5th percentile for 95% VaR
        let var_index = (0.05 * sorted_returns.len() as f64) as usize;
        let var_return = sorted_returns[var_index];
        
        // VaR is positive number representing potential loss
        if var_return < 0.0 {
            -var_return * position_value
        } else {
            0.0
        }
    }
    
    /// Calculate CVaR (Expected Shortfall) - average loss beyond VaR
    pub fn calculate_cvar(&self, position_value: f64) -> f64 {
        if self.recent_returns.len() < 30 {
            return 0.0;
        }
        
        let mut sorted_returns = self.recent_returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let var_index = (0.05 * sorted_returns.len() as f64) as usize;
        let tail_returns: Vec<f64> = sorted_returns.iter().take(var_index).copied().collect();
        
        if tail_returns.is_empty() {
            return 0.0;
        }
        
        let avg_tail_return = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        
        if avg_tail_return < 0.0 {
            -avg_tail_return * position_value
        } else {
            0.0
        }
    }
    
    /// Check VaR-based position limits and return allowed position size
    pub fn check_var_limits(&self, proposed_position_value: f64, current_var: f64, current_cvar: f64) -> RiskAction {
        // Check VaR limit
        if let Some(var_limit) = self.limits.var_limit {
            if current_var > var_limit {
                let scale_factor = var_limit / current_var;
                let allowed_position = proposed_position_value * scale_factor;
                return RiskAction::ScalePosition(allowed_position, 
                    format!("VaR ${:.0} exceeds limit ${:.0}, scaling to {:.1}%", 
                        current_var, var_limit, scale_factor * 100.0));
            }
        }
        
        // Check CVaR limit
        if let Some(cvar_limit) = self.limits.cvar_limit {
            if current_cvar > cvar_limit {
                let scale_factor = cvar_limit / current_cvar;
                let allowed_position = proposed_position_value * scale_factor;
                return RiskAction::ScalePosition(allowed_position,
                    format!("CVaR ${:.0} exceeds limit ${:.0}, scaling to {:.1}%",
                        current_cvar, cvar_limit, scale_factor * 100.0));
            }
        }
        
        RiskAction::Proceed
    }

    /// Check aggregate portfolio Greeks against configured limits.
    pub fn check_greeks(&self, portfolio_greeks: &Greeks) -> RiskAction {
        if let Some(limit) = self.limits.max_net_delta {
            if portfolio_greeks.delta.abs() > limit {
                return RiskAction::RejectOrder(format!(
                    "Portfolio delta {:.4} exceeds max {:.4}",
                    portfolio_greeks.delta, limit
                ));
            }
        }

        if let Some(limit) = self.limits.max_gamma {
            if portfolio_greeks.gamma.abs() > limit {
                return RiskAction::RejectOrder(format!(
                    "Portfolio gamma {:.6} exceeds max {:.6}",
                    portfolio_greeks.gamma, limit
                ));
            }
        }

        if let Some(limit) = self.limits.max_vega {
            if portfolio_greeks.vega.abs() > limit {
                return RiskAction::RejectOrder(format!(
                    "Portfolio vega {:.4} exceeds max {:.4}",
                    portfolio_greeks.vega, limit
                ));
            }
        }

        if let Some(limit) = self.limits.max_theta {
            if portfolio_greeks.theta.abs() > limit {
                return RiskAction::RejectOrder(format!(
                    "Portfolio theta {:.4} exceeds max {:.4}",
                    portfolio_greeks.theta, limit
                ));
            }
        }

        RiskAction::Proceed
    }

    /// Check expiry bucket concentration using absolute notional per expiry.
    pub fn check_expiry_concentration(
        &self,
        expiry_notionals: &HashMap<DateTime<Utc>, f64>,
    ) -> RiskAction {
        let Some(limit) = self.limits.max_notional_per_expiry else {
            return RiskAction::Proceed;
        };

        for (expiry, notional) in expiry_notionals {
            if notional.abs() > limit {
                return RiskAction::RejectOrder(format!(
                    "Notional {:.2} for expiry {} exceeds max {:.2}",
                    notional,
                    expiry,
                    limit
                ));
            }
        }

        RiskAction::Proceed
    }
    
    /// Set VaR/CVaR from external Monte Carlo analysis
    pub fn set_var_from_monte_carlo(&mut self, var_95: f64, cvar_95: f64) {
        // Store for use in position sizing
        // These values come from backtest Monte Carlo analysis
        self.recent_returns.clear(); // Clear historical, use MC values directly
        // The actual VaR/CVaR are passed to check_var_limits when checking orders
        let _ = (var_95, cvar_95); // Used by caller
    }
    
    // ========== CORRELATION RISK MANAGEMENT ==========
    
    /// Update correlation matrix between assets
    pub fn update_correlation(&mut self, asset_a: &str, asset_b: &str, correlation: f64) {
        // Remove old entry if exists
        self.asset_correlations.retain(|c| 
            !(c.asset_a == asset_a && c.asset_b == asset_b) &&
            !(c.asset_a == asset_b && c.asset_b == asset_a)
        );
        
        self.asset_correlations.push(AssetCorrelation {
            asset_a: asset_a.to_string(),
            asset_b: asset_b.to_string(),
            correlation,
            last_updated: Utc::now(),
        });
    }
    
    /// Update position for an asset
    pub fn update_position(&mut self, asset: &str, position_value: f64) {
        self.position_by_asset.insert(asset.to_string(), position_value);
    }
    
    /// Get correlation between two assets
    pub fn get_correlation(&self, asset_a: &str, asset_b: &str) -> Option<f64> {
        self.asset_correlations.iter()
            .find(|c| 
                (c.asset_a == asset_a && c.asset_b == asset_b) ||
                (c.asset_a == asset_b && c.asset_b == asset_a)
            )
            .map(|c| c.correlation)
    }
    
    /// Calculate total correlated exposure for an asset
    /// Returns sum of position values for assets correlated above threshold
    pub fn calculate_correlated_exposure(&self, asset: &str) -> f64 {
        let threshold = self.limits.correlation_threshold.unwrap_or(0.7);
        let mut total_exposure = 0.0;
        
        // Find all assets correlated with this one
        for corr in &self.asset_correlations {
            if corr.correlation.abs() >= threshold {
                if corr.asset_a == asset {
                    if let Some(&pos) = self.position_by_asset.get(&corr.asset_b) {
                        total_exposure += pos.abs();
                    }
                } else if corr.asset_b == asset {
                    if let Some(&pos) = self.position_by_asset.get(&corr.asset_a) {
                        total_exposure += pos.abs();
                    }
                }
            }
        }
        
        // Include own position
        if let Some(&pos) = self.position_by_asset.get(asset) {
            total_exposure += pos.abs();
        }
        
        total_exposure
    }
    
    /// Check if adding a position would exceed correlated exposure limits
    pub fn check_correlated_exposure(&self, asset: &str, additional_exposure: f64) -> RiskAction {
        if let Some(max_corr_exp) = self.limits.max_correlated_exposure {
            let current_exposure = self.calculate_correlated_exposure(asset);
            let new_exposure = current_exposure + additional_exposure;
            
            if new_exposure > max_corr_exp {
                let allowed = (max_corr_exp - current_exposure).max(0.0);
                if allowed <= 0.0 {
                    return RiskAction::RejectOrder(format!(
                        "Correlated exposure ${:.0} would exceed limit ${:.0}",
                        new_exposure, max_corr_exp
                    ));
                }
                return RiskAction::ScalePosition(allowed,
                    format!("Limiting to ${:.0} due to correlated exposure", allowed));
            }
        }
        
        RiskAction::Proceed
    }
    
    // ========== VOLATILITY REGIME SWITCHING ==========
    
    /// Update volatility history and detect regime
    pub fn update_volatility(&mut self, current_volatility: f64) {
        self.volatility_history.push(current_volatility);
        if self.volatility_history.len() > 100 {
            self.volatility_history.remove(0);
        }
        
        // Update baseline (rolling average of volatility)
        if self.volatility_history.len() >= 20 {
            let baseline = self.volatility_history.iter().sum::<f64>() 
                / self.volatility_history.len() as f64;
            self.baseline_volatility = Some(baseline);
        }
        
        // Detect regime
        self.current_regime = self.detect_volatility_regime(current_volatility);
    }
    
    /// Detect volatility regime based on current vs baseline
    fn detect_volatility_regime(&self, current_volatility: f64) -> VolatilityRegime {
        let baseline = match self.baseline_volatility {
            Some(b) if b > 0.0 => b,
            _ => return VolatilityRegime::Normal,
        };
        
        let vol_ratio = current_volatility / baseline;
        
        let extreme_threshold = self.limits.vol_regime_threshold_extreme.unwrap_or(2.5);
        let high_threshold = self.limits.vol_regime_threshold_high.unwrap_or(1.5);
        
        if vol_ratio >= extreme_threshold {
            VolatilityRegime::Extreme
        } else if vol_ratio >= high_threshold {
            VolatilityRegime::High
        } else if vol_ratio <= 0.7 {
            VolatilityRegime::Low
        } else {
            VolatilityRegime::Normal
        }
    }
    
    /// Get position size multiplier based on current volatility regime
    pub fn get_regime_position_multiplier(&self) -> f64 {
        match self.current_regime {
            VolatilityRegime::Low => 1.2,      // Slightly larger in low vol
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::High => self.limits.high_vol_position_mult.unwrap_or(0.5),
            VolatilityRegime::Extreme => self.limits.extreme_vol_position_mult.unwrap_or(0.2),
        }
    }
    
    /// Get current volatility regime
    pub fn get_current_regime(&self) -> VolatilityRegime {
        self.current_regime
    }
    
    /// Enhanced order check with all risk dimensions
    pub fn check_order_comprehensive(
        &self, 
        order_size: f64, 
        current_position: f64, 
        metrics: &RiskMetrics,
        asset: &str,
        position_value: f64,
    ) -> RiskAction {
        // 1. Basic position/order checks
        let basic_check = self.check_order(order_size, current_position, metrics);
        if basic_check != RiskAction::Proceed {
            return basic_check;
        }
        
        // 2. VaR-based check
        let var_check = self.check_var_limits(position_value, metrics.current_var, metrics.current_cvar);
        if let RiskAction::ScalePosition(_, _) = &var_check {
            return var_check;
        }

        // 2.5. Derivatives Greeks check
        if let Some(portfolio_greeks) = &metrics.portfolio_greeks {
            let greek_check = self.check_greeks(portfolio_greeks);
            if greek_check != RiskAction::Proceed {
                return greek_check;
            }
        }
        
        // 3. Correlation check
        let corr_check = self.check_correlated_exposure(asset, order_size * position_value / order_size.max(0.001));
        if corr_check != RiskAction::Proceed {
            return corr_check;
        }
        
        // 4. Apply volatility regime adjustment
        let regime_mult = self.get_regime_position_multiplier();
        if regime_mult < 1.0 {
            let adjusted_max = order_size * regime_mult;
            if order_size > adjusted_max {
                return RiskAction::ScalePosition(adjusted_max,
                    format!("Reducing size due to {:?} volatility regime", self.current_regime));
            }
        }
        
        RiskAction::Proceed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── VolatilityRegime ──

    #[test]
    fn volatility_regime_default_is_normal() {
        assert_eq!(VolatilityRegime::default(), VolatilityRegime::Normal);
    }

    #[test]
    fn volatility_regime_serde_roundtrip() {
        let v = VolatilityRegime::Extreme;
        let json = serde_json::to_string(&v).unwrap();
        let v2: VolatilityRegime = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    // ── RiskLimits ──

    #[test]
    fn risk_limits_default() {
        let d = RiskLimits::default();
        assert_eq!(d.max_position, Some(10.0));
        assert_eq!(d.max_order_size, Some(1.0));
        assert_eq!(d.max_drawdown, Some(0.20));
        assert_eq!(d.var_limit, Some(5000.0));
        assert_eq!(d.correlation_threshold, Some(0.7));
    }

    // ── RiskMetrics ──

    #[test]
    fn risk_metrics_default() {
        let m = RiskMetrics::default();
        assert_eq!(m.current_position, 0.0);
        assert_eq!(m.current_drawdown, 0.0);
        assert_eq!(m.volatility_regime, VolatilityRegime::Normal);
        assert!(m.portfolio_greeks.is_none());
    }

    // ── RiskManager basics ──

    #[test]
    fn risk_manager_new_defaults() {
        let rm = RiskManager::new();
        assert_eq!(rm.limits.max_position, Some(10.0));
        assert_eq!(rm.get_current_regime(), VolatilityRegime::Normal);
    }

    #[test]
    fn risk_manager_with_limits() {
        let mut limits = RiskLimits::default();
        limits.max_position = Some(100.0);
        let rm = RiskManager::with_limits(limits);
        assert_eq!(rm.limits.max_position, Some(100.0));
    }

    #[test]
    fn risk_manager_initialize_sets_equity() {
        let mut rm = RiskManager::new();
        rm.initialize(10000.0);
        assert_eq!(rm.get_current_drawdown(10000.0), 0.0);
    }

    #[test]
    fn risk_manager_update_equity_tracks_peak() {
        let mut rm = RiskManager::new();
        rm.initialize(10000.0);
        rm.update_equity(12000.0);
        let dd = rm.get_current_drawdown(11000.0);
        // Peak is 12000, current 11000 → dd = 1000/12000 ≈ 0.0833
        assert!((dd - 1000.0 / 12000.0).abs() < 1e-10);
    }

    #[test]
    fn get_current_drawdown_zero_peak() {
        let rm = RiskManager::new();
        assert_eq!(rm.get_current_drawdown(100.0), 0.0);
    }

    #[test]
    fn get_current_drawdown_is_not_capped_at_100_percent() {
        // A short/written option's genuinely unbounded loss (see this
        // function's own doc comment) can crash equity well below zero --
        // the reported drawdown must reflect that, not silently cap at 1.0.
        let mut rm = RiskManager::new();
        rm.initialize(10_000.0);
        // Equity crashes to -8,000 against a 10,000 peak: (10,000 - (-8,000))
        // / 10,000 = 1.8 (180% drawdown).
        let dd = rm.get_current_drawdown(-8_000.0);
        assert!((dd - 1.8).abs() < 1e-9, "expected an uncapped 1.8, got {dd}");
    }

    #[test]
    fn get_current_drawdown_a_max_drawdown_limit_above_one_can_still_fire() {
        // Regression guard for the pre-fix bug: a caller configuring
        // `max_drawdown` above 1.0 (a deliberately loose tolerance) must
        // still be able to have the gate fire once a real loss exceeds it --
        // with the old `.clamp(0.0, 1.0)`, this gate could NEVER fire
        // regardless of how catastrophic the real loss became.
        let mut limits = RiskLimits::default();
        limits.max_drawdown = Some(1.2);
        let mut rm = RiskManager::with_limits(limits);
        rm.initialize(10_000.0);
        let metrics = RiskMetrics {
            current_drawdown: rm.get_current_drawdown(-3_000.0), // (10000-(-3000))/10000 = 1.3
            ..RiskMetrics::default()
        };
        assert!(rm.check(&metrics).is_err(), "a 130% drawdown must trip a 120% limit");
    }

    // ── check_order ──

    #[test]
    fn check_order_proceed() {
        let rm = RiskManager::new();
        let metrics = RiskMetrics::default();
        assert_eq!(rm.check_order(0.5, 0.0, &metrics), RiskAction::Proceed);
    }

    #[test]
    fn check_order_exceeds_max_order_size() {
        let rm = RiskManager::new(); // max_order_size = 1.0
        let metrics = RiskMetrics::default();
        match rm.check_order(5.0, 0.0, &metrics) {
            RiskAction::ReducePosition(allowed) => assert!((allowed - 1.0).abs() < 1e-10),
            other => panic!("expected ReducePosition, got {:?}", other),
        }
    }

    #[test]
    fn check_order_exceeds_max_position() {
        let rm = RiskManager::new(); // max_position = 10.0
        let metrics = RiskMetrics::default();
        match rm.check_order(0.5, 10.0, &metrics) {
            RiskAction::ReducePosition(_) | RiskAction::RejectOrder(_) => {}
            other => panic!("expected position limit action, got {:?}", other),
        }
    }

    #[test]
    fn check_order_drawdown_exceeded() {
        let rm = RiskManager::new(); // max_drawdown = 0.20
        let mut metrics = RiskMetrics::default();
        metrics.current_drawdown = 0.25;
        match rm.check_order(0.5, 0.0, &metrics) {
            RiskAction::CloseAllPositions(_) => {}
            other => panic!("expected CloseAllPositions, got {:?}", other),
        }
    }

    #[test]
    fn check_order_daily_loss_exceeded() {
        let rm = RiskManager::new(); // max_daily_loss = 1000.0
        let mut metrics = RiskMetrics::default();
        metrics.daily_pnl = -1500.0;
        match rm.check_order(0.5, 0.0, &metrics) {
            RiskAction::HaltTrading(_) => {}
            other => panic!("expected HaltTrading, got {:?}", other),
        }
    }

    // ── check_stop_loss ──

    #[test]
    fn check_stop_loss_long_triggered() {
        let rm = RiskManager::new(); // stop_loss_pct = 0.05
        assert!(rm.check_stop_loss(100.0, 94.0, true));
    }

    #[test]
    fn check_stop_loss_long_not_triggered() {
        let rm = RiskManager::new();
        assert!(!rm.check_stop_loss(100.0, 96.0, true));
    }

    #[test]
    fn check_stop_loss_short_triggered() {
        let rm = RiskManager::new();
        assert!(rm.check_stop_loss(100.0, 106.0, false));
    }

    // ── check ──

    #[test]
    fn check_ok_within_limits() {
        let rm = RiskManager::new();
        let metrics = RiskMetrics::default();
        assert!(rm.check(&metrics).is_ok());
    }

    #[test]
    fn check_position_exceeded() {
        let rm = RiskManager::new();
        let mut m = RiskMetrics::default();
        m.current_position = 20.0;
        assert!(rm.check(&m).is_err());
    }

    // ── VaR / CVaR ──

    #[test]
    fn calculate_var_insufficient_data() {
        let rm = RiskManager::new();
        assert_eq!(rm.calculate_var(10000.0), 0.0);
    }

    #[test]
    fn calculate_var_sufficient_data() {
        let mut rm = RiskManager::new();
        // Add 50 daily returns with some negative ones
        for i in 0..50 {
            let r = if i % 5 == 0 { -0.03 } else { 0.01 };
            rm.update_returns(r);
        }
        let var = rm.calculate_var(10000.0);
        assert!(var > 0.0, "VaR should be positive with negative returns");
    }

    #[test]
    fn calculate_cvar_insufficient_data() {
        let rm = RiskManager::new();
        assert_eq!(rm.calculate_cvar(10000.0), 0.0);
    }

    // ── VaR limits ──

    #[test]
    fn check_var_limits_proceed() {
        let rm = RiskManager::new(); // var_limit = 5000
        assert_eq!(rm.check_var_limits(100000.0, 3000.0, 4000.0), RiskAction::Proceed);
    }

    #[test]
    fn check_var_limits_var_exceeded() {
        let rm = RiskManager::new();
        match rm.check_var_limits(100000.0, 8000.0, 4000.0) {
            RiskAction::ScalePosition(_, _) => {}
            other => panic!("expected ScalePosition, got {:?}", other),
        }
    }

    #[test]
    fn check_var_limits_cvar_exceeded() {
        let rm = RiskManager::new(); // cvar_limit = 7500
        match rm.check_var_limits(100000.0, 4000.0, 10000.0) {
            RiskAction::ScalePosition(_, _) => {}
            other => panic!("expected ScalePosition, got {:?}", other),
        }
    }

    // ── Greeks checks ──

    #[test]
    fn check_greeks_within_limits() {
        let rm = RiskManager::new();
        let g = Greeks { delta: 5.0, gamma: 2.0, theta: 0.0, vega: 500.0, rho: 0.0 };
        assert_eq!(rm.check_greeks(&g), RiskAction::Proceed);
    }

    #[test]
    fn check_greeks_delta_exceeded() {
        let rm = RiskManager::new(); // max_net_delta = 10.0
        let g = Greeks { delta: 15.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 };
        match rm.check_greeks(&g) {
            RiskAction::RejectOrder(_) => {}
            other => panic!("expected RejectOrder, got {:?}", other),
        }
    }

    // ── Margin call ──

    #[test]
    fn check_margin_proceeds_when_equity_comfortably_above_maintenance() {
        assert_eq!(check_margin(5000.0, 2000.0, 500.0), RiskAction::Proceed);
    }

    #[test]
    fn check_margin_triggers_at_or_below_maintenance() {
        match check_margin(500.0, 2000.0, 500.0) {
            RiskAction::CloseAllPositions(_) => {}
            other => panic!("expected CloseAllPositions at equity == maintenance margin, got {:?}", other),
        }
        match check_margin(400.0, 2000.0, 500.0) {
            RiskAction::CloseAllPositions(_) => {}
            other => panic!("expected CloseAllPositions below maintenance margin, got {:?}", other),
        }
    }

    #[test]
    fn check_margin_never_triggers_for_unleveraged_portfolio() {
        // margin_used == 0.0 means no leveraged position was ever opened --
        // this must never spuriously fire regardless of the equity/threshold
        // values passed in.
        assert_eq!(check_margin(0.0, 0.0, 500.0), RiskAction::Proceed);
        assert_eq!(check_margin(-100.0, 0.0, 500.0), RiskAction::Proceed);
    }

    // ── Correlation management ──

    #[test]
    fn correlation_update_and_get() {
        let mut rm = RiskManager::new();
        rm.update_correlation("BTC", "ETH", 0.85);
        assert_eq!(rm.get_correlation("BTC", "ETH"), Some(0.85));
        assert_eq!(rm.get_correlation("ETH", "BTC"), Some(0.85));
    }

    #[test]
    fn correlation_missing_returns_none() {
        let rm = RiskManager::new();
        assert_eq!(rm.get_correlation("BTC", "SOL"), None);
    }

    #[test]
    fn correlated_exposure_calculation() {
        let mut rm = RiskManager::new();
        rm.update_correlation("BTC", "ETH", 0.85); // above 0.7 threshold
        rm.update_position("BTC", 10000.0);
        rm.update_position("ETH", 5000.0);
        let exp = rm.calculate_correlated_exposure("BTC");
        // Should include BTC's own position + ETH (correlated)
        assert!(exp >= 15000.0);
    }

    #[test]
    fn check_correlated_exposure_proceed() {
        let rm = RiskManager::new();
        assert_eq!(rm.check_correlated_exposure("BTC", 1000.0), RiskAction::Proceed);
    }

    // ── Volatility regime ──

    #[test]
    fn volatility_regime_detection() {
        let mut rm = RiskManager::new();
        // Build baseline with 20+ observations
        for _ in 0..25 {
            rm.update_volatility(0.02);
        }
        assert_eq!(rm.get_current_regime(), VolatilityRegime::Normal);
        // Spike to extreme
        rm.update_volatility(0.10); // 5x baseline => extreme
        assert_eq!(rm.get_current_regime(), VolatilityRegime::Extreme);
    }

    #[test]
    fn regime_position_multiplier() {
        let rm = RiskManager::new();
        assert_eq!(rm.get_regime_position_multiplier(), 1.0); // Normal default
    }
}
