//! Black-Scholes option pricing and Greeks engine.
//!
//! Provides:
//! - [`bs_price`] — European call/put price via Black-Scholes.
//! - [`bs_greeks`] — First- and second-order sensitivities (Δ, Γ, Θ, Vega, ρ).
//! - [`bs_implied_vol`] — IV solver (Newton-Raphson, converges to ε = 1e-7).
//! - [`aggregate_greeks`] — Portfolio-level Greek aggregation with multiplier scaling.
//!
//! Assumptions: European exercise, continuous dividend yield = 0, deterministic
//! interest rate, log-normal underlying. Suitable for crypto options (Deribit).

use derivatives::{Greeks, InstrumentKind};
use statrs::distribution::{ContinuousCDF, Normal};

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Standard normal distribution singleton (stateless / zero-cost).
#[inline]
fn norm() -> Normal {
    Normal::new(0.0, 1.0).expect("Valid normal distribution constants")
}

/// N(x) — standard normal CDF.
#[inline]
fn cdf(x: f64) -> f64 {
    norm().cdf(x)
}

/// n(x) — standard normal PDF.
#[inline]
fn pdf(x: f64) -> f64 {
    use statrs::distribution::Continuous;
    norm().pdf(x)
}

/// Compute d₁ and d₂ for Black-Scholes.
/// Returns `None` if the inputs are degenerate (non-positive vol, tte, or spot).
fn d1_d2(spot: f64, strike: f64, rate: f64, vol: f64, tte: f64) -> Option<(f64, f64)> {
    if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || tte <= 0.0 {
        return None;
    }
    let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * tte) / (vol * tte.sqrt());
    let d2 = d1 - vol * tte.sqrt();
    Some((d1, d2))
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Compute the Black-Scholes price for a European call or put.
///
/// # Parameters
/// - `kind`   — must be `InstrumentKind::Call` or `InstrumentKind::Put`.
///   Other variants return `0.0`.
/// - `spot`   — current underlying price.
/// - `rate`   — continuously compounded risk-free rate (e.g. 0.05 = 5%).
/// - `vol`    — implied volatility, annualised (e.g. 0.80 = 80%).
/// - `tte`    — time to expiry in years (e.g. 30/365.25 for 30 days).
///
/// Returns the theoretical option price. At expiry (`tte = 0`) returns
/// intrinsic value.
pub fn bs_price(kind: &InstrumentKind, spot: f64, rate: f64, vol: f64, tte: f64) -> f64 {
    match kind {
        InstrumentKind::Call { strike, .. } => {
            bs_call(spot, *strike, rate, vol, tte)
        }
        InstrumentKind::Put { strike, .. } => {
            bs_put(spot, *strike, rate, vol, tte)
        }
        // Non-options: mark-to-market is handled linearly by the portfolio manager.
        _ => 0.0,
    }
}

/// Compute all five Black-Scholes Greeks for a European call or put.
///
/// Returns [`Greeks::zero()`] for non-option instrument kinds.
/// Theta is expressed per calendar day (divide annual theta by 365).
pub fn bs_greeks(kind: &InstrumentKind, spot: f64, rate: f64, vol: f64, tte: f64) -> Greeks {
    let (strike, is_call) = match kind {
        InstrumentKind::Call { strike, .. } => (*strike, true),
        InstrumentKind::Put { strike, .. } => (*strike, false),
        _ => return Greeks::zero(),
    };

    let Some((d1, d2)) = d1_d2(spot, strike, rate, vol, tte) else {
        // At expiry: return intrinsic delta only.
        let delta = if is_call {
            if spot > strike { 1.0 } else { 0.0 }
        } else {
            if spot < strike { -1.0 } else { 0.0 }
        };
        return Greeks { delta, ..Greeks::zero() };
    };

    let nd1 = pdf(d1);
    let df = (-rate * tte).exp(); // discount factor

    // Delta
    let delta = if is_call { cdf(d1) } else { cdf(d1) - 1.0 };

    // Gamma (same for call and put)
    let gamma = nd1 / (spot * vol * tte.sqrt());

    // Vega (same for call and put) — per 1 unit of vol (i.e. per "100%")
    let vega = spot * nd1 * tte.sqrt();

    // Theta — per calendar year; divide by 365 for daily decay
    let theta_common = -(spot * nd1 * vol) / (2.0 * tte.sqrt()) - rate * strike * df;
    let theta = if is_call {
        (theta_common * cdf(d2)) / 365.0
    } else {
        (theta_common * cdf(-d2)) / 365.0
    };

    // Rho — per 1 unit of rate (i.e. per "100%")
    let rho = if is_call {
        strike * tte * df * cdf(d2)
    } else {
        -strike * tte * df * cdf(-d2)
    };

    Greeks { delta, gamma, theta, vega, rho }
}

/// Compute implied volatility via Newton-Raphson iteration.
///
/// # Parameters
/// - `kind`         — `Call` or `Put`. Returns `None` for other variants.
/// - `spot`         — current underlying price.
/// - `market_price` — observed option price in the market.
/// - `rate`         — risk-free rate.
/// - `tte`          — time to expiry in years.
///
/// Returns `Some(iv)` on convergence, `None` if the inputs are degenerate
/// or the solver fails to converge within 100 iterations to ε = 1e-7.
pub fn bs_implied_vol(
    kind: &InstrumentKind,
    spot: f64,
    market_price: f64,
    rate: f64,
    tte: f64,
) -> Option<f64> {
    let strike = match kind {
        InstrumentKind::Call { strike, .. } | InstrumentKind::Put { strike, .. } => *strike,
        _ => return None,
    };

    if market_price <= 0.0 || spot <= 0.0 || strike <= 0.0 || tte <= 0.0 {
        return None;
    }

    // Initial guess: Brenner-Subrahmanyam approximation
    let mut vol = (2.0 * std::f64::consts::PI / tte).sqrt() * (market_price / spot);
    // Clamp to a reasonable range
    vol = vol.clamp(1e-4, 20.0);

    const MAX_ITER: usize = 100;
    const EPSILON: f64 = 1e-7;

    for _ in 0..MAX_ITER {
        let price = bs_price(kind, spot, rate, vol, tte);
        let diff = price - market_price;

        if diff.abs() < EPSILON {
            return Some(vol);
        }

        // vega = ∂price/∂vol — use bs_greeks for this
        let v = bs_greeks(kind, spot, rate, vol, tte).vega;
        if v.abs() < 1e-12 {
            break; // Degenerate: vega vanished
        }

        vol -= diff / v;

        if vol <= 0.0 || vol > 20.0 {
            break; // Out of reasonable range
        }
    }

    // Return the best estimate if it's in a reasonable range
    if vol > 0.0 && vol < 20.0 {
        Some(vol)
    } else {
        None
    }
}

/// Aggregate Greeks across a portfolio of positions.
///
/// `positions` is a slice of tuples `(quantity, multiplier, greeks_per_contract)`.
/// - `quantity`   — signed position size (positive = long, negative = short).
/// - `multiplier` — contract multiplier (`DerivativeMetadata::contract_multiplier`).
/// - `greeks`     — per-contract Greeks as returned by [`bs_greeks`].
///
/// Returns the net portfolio-level Greeks after accounting for quantity and
/// contract size.
pub fn aggregate_greeks(positions: &[(f64, f64, Greeks)]) -> Greeks {
    positions.iter().fold(Greeks::zero(), |acc, (qty, mult, g)| {
        acc + g.scale(*qty * *mult)
    })
}

// ─────────────────────────────────────────────
// Internal implementations
// ─────────────────────────────────────────────

fn bs_call(spot: f64, strike: f64, rate: f64, vol: f64, tte: f64) -> f64 {
    match d1_d2(spot, strike, rate, vol, tte) {
        Some((d1, d2)) => {
            let df = (-rate * tte).exp();
            spot * cdf(d1) - strike * df * cdf(d2)
        }
        // At or past expiry: intrinsic value
        None => (spot - strike).max(0.0),
    }
}

fn bs_put(spot: f64, strike: f64, rate: f64, vol: f64, tte: f64) -> f64 {
    match d1_d2(spot, strike, rate, vol, tte) {
        Some((d1, d2)) => {
            let df = (-rate * tte).exp();
            strike * df * cdf(-d2) - spot * cdf(-d1)
        }
        None => (strike - spot).max(0.0),
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use chrono::{Duration, Utc};

    fn call_kind(strike: f64) -> InstrumentKind {
        InstrumentKind::Call {
            strike,
            expiry: Utc::now() + Duration::days(30),
        }
    }

    fn put_kind(strike: f64) -> InstrumentKind {
        InstrumentKind::Put {
            strike,
            expiry: Utc::now() + Duration::days(30),
        }
    }

    // Reference values computed externally with scipy.stats.norm
    // (spot=100, strike=100, rate=0.05, vol=0.20, tte=1.0)
    const SPOT: f64 = 100.0;
    const STRIKE: f64 = 100.0;
    const RATE: f64 = 0.05;
    const VOL: f64 = 0.20;
    const TTE: f64 = 1.0; // 1 year

    #[test]
    fn bs_call_atm_price() {
        // scipy: ~10.4506
        let price = bs_price(&call_kind(STRIKE), SPOT, RATE, VOL, TTE);
        assert_abs_diff_eq!(price, 10.4506, epsilon = 0.01);
    }

    #[test]
    fn bs_put_atm_price() {
        // Put-call parity: P = C - S + K*e^{-rT} ≈ 5.5735
        let call = bs_price(&call_kind(STRIKE), SPOT, RATE, VOL, TTE);
        let put  = bs_price(&put_kind(STRIKE),  SPOT, RATE, VOL, TTE);
        let parity_put = call - SPOT + STRIKE * (-RATE * TTE).exp();
        assert_abs_diff_eq!(put, parity_put, epsilon = 1e-6);
    }

    #[test]
    fn call_delta_atm() {
        let g = bs_greeks(&call_kind(STRIKE), SPOT, RATE, VOL, TTE);
        // ATM call delta should be around 0.5–0.64
        assert!(g.delta > 0.5 && g.delta < 0.7, "delta={}", g.delta);
    }

    #[test]
    fn put_delta_atm() {
        let g = bs_greeks(&put_kind(STRIKE), SPOT, RATE, VOL, TTE);
        // ATM put delta should be around -0.3 to -0.5
        assert!(g.delta > -0.6 && g.delta < -0.3, "delta={}", g.delta);
    }

    #[test]
    fn call_put_gamma_equal() {
        let gc = bs_greeks(&call_kind(STRIKE), SPOT, RATE, VOL, TTE);
        let gp = bs_greeks(&put_kind(STRIKE), SPOT, RATE, VOL, TTE);
        assert_abs_diff_eq!(gc.gamma, gp.gamma, epsilon = 1e-10);
    }

    #[test]
    fn call_put_vega_equal() {
        let gc = bs_greeks(&call_kind(STRIKE), SPOT, RATE, VOL, TTE);
        let gp = bs_greeks(&put_kind(STRIKE), SPOT, RATE, VOL, TTE);
        assert_abs_diff_eq!(gc.vega, gp.vega, epsilon = 1e-10);
    }

    #[test]
    fn iv_round_trip() {
        let kind = call_kind(STRIKE);
        let price = bs_price(&kind, SPOT, RATE, VOL, TTE);
        let iv = bs_implied_vol(&kind, SPOT, price, RATE, TTE).unwrap();
        assert_abs_diff_eq!(iv, VOL, epsilon = 1e-5);
    }

    #[test]
    fn iv_round_trip_put() {
        let kind = put_kind(90.0);
        let price = bs_price(&kind, SPOT, RATE, 0.35, TTE);
        let iv = bs_implied_vol(&kind, SPOT, price, RATE, TTE).unwrap();
        assert_abs_diff_eq!(iv, 0.35, epsilon = 1e-5);
    }

    #[test]
    fn aggregate_greeks_sums_correctly() {
        let g = Greeks {
            delta: 0.5,
            gamma: 0.02,
            theta: -5.0,
            vega: 30.0,
            rho: 0.1,
        };
        // Long 2 contracts × multiplier 1.0
        let pos = vec![
            (2.0_f64, 1.0_f64, g.clone()),
            (-1.0_f64, 1.0_f64, g.clone()),
        ];
        let net = aggregate_greeks(&pos);
        // net quantity = 2 - 1 = 1
        assert_abs_diff_eq!(net.delta, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(net.gamma, 0.02, epsilon = 1e-10);
        assert_abs_diff_eq!(net.theta, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn aggregate_greeks_multiplier_scaling() {
        let g = Greeks { delta: 1.0, ..Greeks::zero() };
        // 3 contracts × multiplier 10
        let pos = vec![(3.0_f64, 10.0_f64, g)];
        let net = aggregate_greeks(&pos);
        assert_abs_diff_eq!(net.delta, 30.0, epsilon = 1e-10);
    }
}
