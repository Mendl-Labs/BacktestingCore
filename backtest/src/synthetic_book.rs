//! Synthetic orderbook module for candle-based simulations.
//!
//! When only OHLCV data is available (no L2/trade-by-trade), this module
//! generates an implied bid/ask spread and depth profile from each candle.
//! The simulator uses these to model realistic slippage and market impact.

use orderbook::BookSide;

/// Configuration for the synthetic orderbook generator.
#[derive(Debug, Clone)]
pub struct SyntheticBookConfig {
    /// Base half-spread in basis points (default: 5.0 bps).
    pub spread_bps: f64,
    /// Number of depth levels to generate on each side (default: 5).
    pub depth_levels: usize,
    /// Exponential decay factor for volume at deeper levels (default: 0.5).
    pub depth_decay: f64,
    /// Multiplier applied to (intrabar range / close) to widen the spread
    /// during volatile bars (default: 2.0).
    pub volatility_multiplier: f64,
    /// Hard ceiling on how many bps of HALF-spread the volatility-widening
    /// component alone may contribute, regardless of how wide the candle's
    /// own high-low range is (default: 50.0 bps -- a 0.5% half-spread, i.e.
    /// up to ~1% of additional full-spread width from volatility on top of
    /// the base `spread_bps`).
    ///
    /// Without this cap, `volatility_multiplier` scales the ENTIRE intrabar
    /// range into the spread with no limit: for a calm instrument (~0.1-0.5%
    /// daily range) that's a plausible sub-1% spread, but for a single wide-
    /// range bar (routine for volatile EM FX pairs like USD-TRY, or any
    /// instrument during a high-volatility day) it can blow the spread out
    /// to several percent of price on EACH side. That made every overnight
    /// round trip (buy at the ask on day N, sell at the bid on day N+1) lose
    /// roughly the sum of two days' full trading ranges, dwarfing any real
    /// signal edge and mechanically producing a 0% win rate regardless of
    /// true market direction -- confirmed live in production (backtest
    /// results 71715f38/8d75876a, USD-TRY/oanda, 0/34 and 0/140 winning
    /// trades) once `use_synthetic_book` defaulted to `true` for all
    /// Python-strategy backtests. Mirrors candle_sim.rs's sibling GA-path
    /// model (`RealismConfig::spread_cap_bps`), which has always capped its
    /// own equivalent range-derived spread term for the same reason.
    pub max_volatility_widening_bps: f64,
}

impl Default for SyntheticBookConfig {
    fn default() -> Self {
        Self {
            spread_bps: 5.0,
            depth_levels: 5,
            depth_decay: 0.5,
            volatility_multiplier: 2.0,
            max_volatility_widening_bps: 50.0,
        }
    }
}

/// A single level in the synthetic depth.
#[derive(Debug, Clone)]
pub struct DepthLevel {
    pub price: f64,
    pub volume: f64,
}

/// An instantaneous implied orderbook snapshot for a single candle.
pub struct SyntheticOrderBook {
    pub mid_price: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_levels: Vec<DepthLevel>,
    pub ask_levels: Vec<DepthLevel>,
}

impl SyntheticOrderBook {
    /// Build a synthetic orderbook from a single OHLCV candle.
    ///
    /// - `close`: candle close price (used as mid-price proxy).
    /// - `high`, `low`: intrabar extremes — wider range implies wider spread.
    /// - `volume`: total bar volume — distributed across depth levels.
    /// - `config`: tuning parameters.
    pub fn from_candle(
        close: f64,
        high: f64,
        low: f64,
        volume: f64,
        config: &SyntheticBookConfig,
    ) -> Self {
        // Intrabar volatility as ratio of range to close
        let intrabar_range_ratio = if close > 0.0 {
            (high - low) / close
        } else {
            0.0
        };

        // Effective half-spread: base spread + volatility-widened component,
        // the latter capped at `max_volatility_widening_bps` so a single
        // wide-range bar can't blow the spread out to several percent of
        // price (see that field's doc comment for the production incident
        // this closes).
        let base_half_spread = close * config.spread_bps / 20_000.0;
        let vol_component_uncapped = close * intrabar_range_ratio * config.volatility_multiplier / 2.0;
        let max_vol_component = close * config.max_volatility_widening_bps / 10_000.0;
        let vol_component = vol_component_uncapped.min(max_vol_component);
        let half_spread = base_half_spread + vol_component;

        let best_bid = close - half_spread;
        let best_ask = close + half_spread;

        // Distribute volume across depth levels with exponential decay.
        // Each side gets half the bar volume, spread across levels.
        let half_volume = volume / 2.0;
        let decay = config.depth_decay;

        // Geometric series normalisation: sum = (1 - r^n) / (1 - r)
        let geo_sum = if (decay - 1.0).abs() < 1e-12 {
            config.depth_levels as f64
        } else {
            (1.0 - decay.powi(config.depth_levels as i32)) / (1.0 - decay)
        };

        let base_level_volume = if geo_sum > 0.0 {
            half_volume / geo_sum
        } else {
            half_volume
        };

        // Tick size for spacing levels (one spread-width per level)
        let tick = half_spread.max(close * 0.0001); // at least 1 bps spacing

        let mut bid_levels = Vec::with_capacity(config.depth_levels);
        let mut ask_levels = Vec::with_capacity(config.depth_levels);

        for i in 0..config.depth_levels {
            let level_volume = base_level_volume * decay.powi(i as i32);
            bid_levels.push(DepthLevel {
                price: best_bid - tick * i as f64,
                volume: level_volume,
            });
            ask_levels.push(DepthLevel {
                price: best_ask + tick * i as f64,
                volume: level_volume,
            });
        }

        Self {
            mid_price: close,
            best_bid,
            best_ask,
            bid_levels,
            ask_levels,
        }
    }

    /// Calculate effective fill price and slippage for a market order that
    /// walks through the synthetic depth levels.
    ///
    /// Returns `(fill_price, slippage_bps)`.
    pub fn calculate_slippage(&self, order_size: f64, side: BookSide) -> (f64, f64) {
        let levels = match side {
            BookSide::Bid => &self.ask_levels, // buying = taking from asks
            BookSide::Ask => &self.bid_levels, // selling = taking from bids
        };

        if levels.is_empty() || order_size <= 0.0 {
            let price = match side {
                BookSide::Bid => self.best_ask,
                BookSide::Ask => self.best_bid,
            };
            return (price, 0.0);
        }

        let mut remaining = order_size;
        let mut cost = 0.0_f64;

        for level in levels {
            if remaining <= 0.0 {
                break;
            }
            let fill_at_level = remaining.min(level.volume);
            cost += fill_at_level * level.price;
            remaining -= fill_at_level;
        }

        // If order exceeds all depth, fill remainder at worst level
        if remaining > 0.0 {
            if let Some(worst) = levels.last() {
                cost += remaining * worst.price;
            }
        }

        let fill_price = cost / order_size;
        let reference = self.mid_price;
        let slippage_bps = if reference > 0.0 {
            ((fill_price - reference).abs() / reference) * 10_000.0
        } else {
            0.0
        };

        (fill_price, slippage_bps)
    }

    /// Estimate market impact using a square-root model:
    /// `impact_bps = k * sqrt(order_size / bar_volume) * 10000`
    ///
    /// `k` is calibrated to the current spread (wider spread → more impact).
    pub fn calculate_market_impact(&self, order_size: f64, bar_volume: f64) -> f64 {
        if bar_volume <= 0.0 || order_size <= 0.0 {
            return 0.0;
        }
        // Impact constant proportional to half-spread
        let half_spread_bps = if self.mid_price > 0.0 {
            ((self.best_ask - self.best_bid) / (2.0 * self.mid_price)) * 10_000.0
        } else {
            0.0
        };
        // Square-root law: impact grows with sqrt of participation rate
        let participation = (order_size / bar_volume).min(1.0);
        half_spread_bps * participation.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wider_spread_for_volatile_candle() {
        // Both ranges are kept well under the default volatility-widening
        // cap (50 bps half-spread, see SyntheticBookConfig::max_volatility_widening_bps)
        // so this test demonstrates genuine, uncapped differentiation
        // between a calmer and a more volatile (but still realistic) bar --
        // the capped/extreme case is covered separately below.
        let config = SyntheticBookConfig::default();
        let calm = SyntheticOrderBook::from_candle(100.0, 100.05, 99.95, 1000.0, &config); // 0.1% range
        let wild = SyntheticOrderBook::from_candle(100.0, 100.2, 99.8, 1000.0, &config);    // 0.4% range
        // Wild candle should have a wider spread
        let calm_spread = calm.best_ask - calm.best_bid;
        let wild_spread = wild.best_ask - wild.best_bid;
        assert!(wild_spread > calm_spread, "volatile candle should widen spread");
    }

    #[test]
    fn test_extreme_range_candle_spread_is_capped_not_unbounded() {
        // Regression test for the production incident this closes: an
        // 8%-range bar (routine for a volatile EM FX pair like USD-TRY, or
        // any instrument during a high-volatility day) previously produced
        // a half-spread roughly equal to the ENTIRE intrabar range (~8% of
        // price) with no ceiling -- meaning an overnight round trip (buy at
        // the ask on day N, sell at the bid on day N+1) could lose well over
        // 10% of price on spread cost alone, dwarfing any real signal edge
        // and guaranteeing a loss regardless of true market direction (see
        // backtest results 71715f38/8d75876a, USD-TRY/oanda, 0% win rate
        // across 34 and 140 trades respectively).
        let config = SyntheticBookConfig::default();
        let extreme = SyntheticOrderBook::from_candle(100.0, 104.0, 96.0, 1000.0, &config); // 8% range
        let full_spread_pct = (extreme.best_ask - extreme.best_bid) / extreme.mid_price;
        assert!(
            full_spread_pct < 0.02,
            "an extreme-range bar's spread must stay bounded (got {:.4}%, expected < 2%)",
            full_spread_pct * 100.0
        );
    }

    #[test]
    fn test_slippage_increases_with_size() {
        let config = SyntheticBookConfig::default();
        let book = SyntheticOrderBook::from_candle(100.0, 101.0, 99.0, 1000.0, &config);
        let (_, slip_small) = book.calculate_slippage(10.0, BookSide::Bid);
        let (_, slip_large) = book.calculate_slippage(500.0, BookSide::Bid);
        assert!(slip_large >= slip_small, "larger orders should have >= slippage");
    }

    #[test]
    fn test_market_impact_sqrt_law() {
        let config = SyntheticBookConfig::default();
        let book = SyntheticOrderBook::from_candle(100.0, 101.0, 99.0, 1000.0, &config);
        let impact_small = book.calculate_market_impact(10.0, 1000.0);
        let impact_large = book.calculate_market_impact(100.0, 1000.0);
        // 10x order should have ~sqrt(10) ≈ 3.16x impact
        let ratio = impact_large / impact_small;
        assert!(ratio > 2.5 && ratio < 4.0, "sqrt-law check: ratio={:.2}", ratio);
    }

    #[test]
    fn test_zero_volume_candle() {
        let config = SyntheticBookConfig::default();
        let book = SyntheticOrderBook::from_candle(100.0, 100.0, 100.0, 0.0, &config);
        // Should still produce valid BBO
        assert!(book.best_bid < book.best_ask);
        assert_eq!(book.calculate_market_impact(10.0, 0.0), 0.0);
    }
}
