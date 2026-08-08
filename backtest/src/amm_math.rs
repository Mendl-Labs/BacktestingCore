//! AMM math primitives for liquidity provision backtesting.
//!
//! Pure functions — no state, easily unit-tested with known analytical values.

pub mod constant_product {
    /// Fraction of a swap's fee earned by an LP with the given pool share.
    pub fn fee_share(pool_share: f64, swap_fee: f64) -> f64 {
        pool_share * swap_fee
    }

    /// Impermanent loss for a constant-product (x*y=k) pool.
    ///
    /// `price_ratio` = current_price / entry_price.
    /// Returns a negative value representing loss vs. HODL (e.g., -0.0572 for 2x price move).
    pub fn impermanent_loss(price_ratio: f64) -> f64 {
        if price_ratio <= 0.0 {
            return -1.0;
        }
        let sqrt_r = price_ratio.sqrt();
        2.0 * sqrt_r / (1.0 + price_ratio) - 1.0
    }

    /// Value of an LP position in quote terms given pool reserves and share.
    pub fn position_value(reserve_a: f64, reserve_b: f64, pool_share: f64, price_a_in_b: f64) -> f64 {
        pool_share * (reserve_a * price_a_in_b + reserve_b)
    }

    /// Token amounts for an LP with given share of a constant-product pool.
    pub fn token_amounts(reserve_a: f64, reserve_b: f64, pool_share: f64) -> (f64, f64) {
        (reserve_a * pool_share, reserve_b * pool_share)
    }
}

pub mod concentrated {
    /// Whether the current tick falls within the LP's range [tick_lower, tick_upper).
    pub fn is_in_range(current_tick: i32, tick_lower: i32, tick_upper: i32) -> bool {
        current_tick >= tick_lower && current_tick < tick_upper
    }

    /// Fee earned by a concentrated LP position from a single swap.
    ///
    /// Only call this when the swap is in-range (check `is_in_range` first).
    pub fn fee_share(lp_liquidity: f64, total_liquidity: f64, swap_fee: f64) -> f64 {
        if total_liquidity <= 0.0 {
            return 0.0;
        }
        (lp_liquidity / total_liquidity) * swap_fee
    }

    /// Convert a tick index to a price (base formula: price = 1.0001^tick).
    pub fn tick_to_price(tick: i32) -> f64 {
        1.0001_f64.powi(tick)
    }

    /// Convert a price to the nearest tick index.
    pub fn price_to_tick(price: f64) -> i32 {
        if price <= 0.0 {
            return i32::MIN;
        }
        (price.ln() / 1.0001_f64.ln()).round() as i32
    }

    /// sqrt(price) from tick index.
    pub fn tick_to_sqrt_price(tick: i32) -> f64 {
        1.0001_f64.powi(tick).sqrt()
    }

    /// Compute token amounts from liquidity, current sqrt_price, and range bounds.
    ///
    /// Returns (amount_a, amount_b) the LP holds given the current price.
    pub fn amounts_from_liquidity(
        liquidity: f64,
        sqrt_price: f64,
        sqrt_lower: f64,
        sqrt_upper: f64,
    ) -> (f64, f64) {
        if sqrt_price <= sqrt_lower {
            // All in token A (below range)
            let amount_a = liquidity * (1.0 / sqrt_lower - 1.0 / sqrt_upper);
            (amount_a, 0.0)
        } else if sqrt_price >= sqrt_upper {
            // All in token B (above range)
            let amount_b = liquidity * (sqrt_upper - sqrt_lower);
            (0.0, amount_b)
        } else {
            // In range — split between both tokens
            let amount_a = liquidity * (1.0 / sqrt_price - 1.0 / sqrt_upper);
            let amount_b = liquidity * (sqrt_price - sqrt_lower);
            (amount_a, amount_b)
        }
    }

    /// Compute liquidity from desired token amounts and price range.
    ///
    /// Returns the minimum liquidity satisfiable by both token amounts.
    pub fn liquidity_from_amounts(
        sqrt_price: f64,
        sqrt_lower: f64,
        sqrt_upper: f64,
        amount_a: f64,
        amount_b: f64,
    ) -> f64 {
        if sqrt_price <= sqrt_lower {
            amount_a / (1.0 / sqrt_lower - 1.0 / sqrt_upper)
        } else if sqrt_price >= sqrt_upper {
            amount_b / (sqrt_upper - sqrt_lower)
        } else {
            let l_a = amount_a / (1.0 / sqrt_price - 1.0 / sqrt_upper);
            let l_b = amount_b / (sqrt_price - sqrt_lower);
            l_a.min(l_b)
        }
    }

    /// Impermanent loss for a concentrated liquidity position.
    ///
    /// More severe than constant product because capital is concentrated in a tighter range.
    pub fn impermanent_loss(
        entry_sqrt_price: f64,
        current_sqrt_price: f64,
        sqrt_lower: f64,
        sqrt_upper: f64,
    ) -> f64 {
        let liquidity = 1.0; // Normalize

        let (entry_a, entry_b) = amounts_from_liquidity(liquidity, entry_sqrt_price, sqrt_lower, sqrt_upper);
        let (current_a, current_b) = amounts_from_liquidity(liquidity, current_sqrt_price, sqrt_lower, sqrt_upper);

        let entry_price = entry_sqrt_price * entry_sqrt_price;
        let current_price = current_sqrt_price * current_sqrt_price;

        let hodl_value = entry_a * current_price + entry_b;
        let lp_value = current_a * current_price + current_b;

        if hodl_value == 0.0 {
            return 0.0;
        }
        (lp_value - hodl_value) / hodl_value
    }
}

pub mod stable_swap {
    /// Fee earned by a stable-swap LP (proportional to pool share, same as constant product).
    pub fn fee_share(pool_share: f64, swap_fee: f64) -> f64 {
        pool_share * swap_fee
    }

    /// Impermanent loss for a stable swap pool.
    ///
    /// Stable pools have much lower IL near peg due to the amplification factor.
    /// `price_ratio` = current_price / entry_price (should be near 1.0 for stables).
    /// `amplification` = A parameter (typically 100-2000 for Curve-style pools).
    pub fn impermanent_loss(price_ratio: f64, amplification: f64) -> f64 {
        if price_ratio <= 0.0 || amplification <= 0.0 {
            return -1.0;
        }
        // Approximation: IL ≈ IL_constant_product / (1 + A * (1 - |1 - r|))
        // This dampens IL near the peg (r ≈ 1) proportional to A.
        let cp_il = super::constant_product::impermanent_loss(price_ratio);
        let depeg = (1.0 - price_ratio).abs();
        let damping = 1.0 + amplification * (1.0 - depeg).max(0.0);
        cp_il / damping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp_il_no_price_change() {
        let il = constant_product::impermanent_loss(1.0);
        assert!((il - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cp_il_2x_price() {
        // Known value: 2*sqrt(2)/(1+2) - 1 ≈ -0.05719
        let il = constant_product::impermanent_loss(2.0);
        assert!((il - (-0.05719)).abs() < 0.001);
    }

    #[test]
    fn test_cp_il_half_price() {
        // IL is symmetric: halving and doubling give same IL
        let il = constant_product::impermanent_loss(0.5);
        assert!((il - (-0.05719)).abs() < 0.001);
    }

    #[test]
    fn test_cp_fee_share() {
        let fee = constant_product::fee_share(0.1, 5.0);
        assert!((fee - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_concentrated_in_range() {
        assert!(concentrated::is_in_range(100, 50, 200));
        assert!(!concentrated::is_in_range(200, 50, 200)); // upper is exclusive
        assert!(!concentrated::is_in_range(49, 50, 200));
    }

    #[test]
    fn test_concentrated_tick_price_roundtrip() {
        let tick = 1000;
        let price = concentrated::tick_to_price(tick);
        let recovered = concentrated::price_to_tick(price);
        assert_eq!(recovered, tick);
    }

    #[test]
    fn test_concentrated_amounts_below_range() {
        let (a, b) = concentrated::amounts_from_liquidity(1000.0, 0.9, 1.0, 2.0);
        assert!(a > 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_concentrated_amounts_above_range() {
        let (a, b) = concentrated::amounts_from_liquidity(1000.0, 2.1, 1.0, 2.0);
        assert_eq!(a, 0.0);
        assert!(b > 0.0);
    }

    #[test]
    fn test_concentrated_amounts_in_range() {
        let (a, b) = concentrated::amounts_from_liquidity(1000.0, 1.5, 1.0, 2.0);
        assert!(a > 0.0);
        assert!(b > 0.0);
    }

    #[test]
    fn test_stable_swap_near_peg() {
        // Near peg (ratio=1.001), IL should be near zero with high A
        let il = stable_swap::impermanent_loss(1.001, 1000.0);
        assert!(il.abs() < 0.0001);
    }

    #[test]
    fn test_stable_swap_depeg() {
        // With major depeg and low A, IL approaches constant product IL
        let il_stable = stable_swap::impermanent_loss(0.5, 1.0);
        let il_cp = constant_product::impermanent_loss(0.5);
        // With A=1, damping is minimal
        assert!((il_stable - il_cp / 2.0).abs() < 0.01);
    }
}
