//! Integration tests for Greeks crate — covers gaps in inline unit tests.
//!
//! Covers:
//!   - bs_price: OTM/ITM/expired/degenerate/non-option inputs
//!   - bs_greeks: theta, rho, deep ITM/OTM, expired, non-option
//!   - bs_implied_vol: convergence failures, non-option, deep ITM/OTM, high vol
//!   - aggregate_greeks: empty, single, all-fields, zero-quantity
//!   - put-call parity (price & delta)

use greeks::{bs_price, bs_greeks, bs_implied_vol, aggregate_greeks};
use derivatives::{Greeks, InstrumentKind};
use chrono::{Duration, Utc};

// ─── Helpers ───────────────────────────────────────────────

fn call(strike: f64) -> InstrumentKind {
    InstrumentKind::Call {
        strike,
        expiry: Utc::now() + Duration::days(30),
    }
}

fn put(strike: f64) -> InstrumentKind {
    InstrumentKind::Put {
        strike,
        expiry: Utc::now() + Duration::days(30),
    }
}

// Standard test params
const S: f64 = 100.0;
const K: f64 = 100.0;
const R: f64 = 0.05;
const V: f64 = 0.20;
const T: f64 = 1.0;

// ============================================================================
// bs_price — OTM / ITM / expired / non-option / degenerate
// ============================================================================

#[test]
fn price_itm_call() {
    // strike=80 (deep ITM), should be > intrinsic (20.0)
    let p = bs_price(&call(80.0), S, R, V, T);
    assert!(p > 20.0, "ITM call price {p} > intrinsic 20");
}

#[test]
fn price_otm_call() {
    // strike=120 (OTM), should be > 0 but < ATM
    let atm = bs_price(&call(K), S, R, V, T);
    let otm = bs_price(&call(120.0), S, R, V, T);
    assert!(otm > 0.0);
    assert!(otm < atm, "OTM call price {otm} < ATM {atm}");
}

#[test]
fn price_itm_put() {
    // strike=120, ITM put — intrinsic = 20
    // Note: with positive rates, put time value can be negative (discounting effect)
    // so price may be slightly below intrinsic
    let p = bs_price(&put(120.0), S, R, V, T);
    assert!(p > 15.0, "ITM put price {p} should be substantial");
    // But should be close to intrinsic
    assert!((p - 20.0).abs() < 5.0, "ITM put price {p} should be near intrinsic 20");
}

#[test]
fn price_otm_put() {
    let atm = bs_price(&put(K), S, R, V, T);
    let otm = bs_price(&put(80.0), S, R, V, T);
    assert!(otm > 0.0);
    assert!(otm < atm);
}

#[test]
fn price_expired_call_itm() {
    // At expiry (tte=0), ITM call = intrinsic
    let p = bs_price(&call(90.0), S, R, V, 0.0);
    assert!((p - 10.0).abs() < 1e-10);
}

#[test]
fn price_expired_call_otm() {
    let p = bs_price(&call(110.0), S, R, V, 0.0);
    assert!((p - 0.0).abs() < 1e-10);
}

#[test]
fn price_expired_put_itm() {
    let p = bs_price(&put(110.0), S, R, V, 0.0);
    assert!((p - 10.0).abs() < 1e-10);
}

#[test]
fn price_expired_put_otm() {
    let p = bs_price(&put(90.0), S, R, V, 0.0);
    assert!((p - 0.0).abs() < 1e-10);
}

#[test]
fn price_non_option_returns_zero() {
    assert_eq!(bs_price(&InstrumentKind::Spot, S, R, V, T), 0.0);
    assert_eq!(bs_price(&InstrumentKind::Perpetual, S, R, V, T), 0.0);
    let fut = InstrumentKind::Future { expiry: Utc::now() + Duration::days(30) };
    assert_eq!(bs_price(&fut, S, R, V, T), 0.0);
}

#[test]
fn price_put_call_parity() {
    // C - P = S - K*e^(-rT)
    let c = bs_price(&call(K), S, R, V, T);
    let p = bs_price(&put(K), S, R, V, T);
    let parity = S - K * (-R * T).exp();
    assert!((c - p - parity).abs() < 0.01, "Put-call parity violated: C-P={}, expected {parity}", c - p);
}

// ============================================================================
// bs_greeks — theta, rho, deep ITM/OTM, expired, non-option
// ============================================================================

#[test]
fn greeks_theta_call_negative() {
    let g = bs_greeks(&call(K), S, R, V, T);
    // Theta should be negative for a long call (time decay)
    assert!(g.theta < 0.0, "Call theta should be negative, got {}", g.theta);
}

#[test]
fn greeks_theta_put_negative() {
    let g = bs_greeks(&put(K), S, R, V, T);
    assert!(g.theta < 0.0, "Put theta should be negative, got {}", g.theta);
}

#[test]
fn greeks_rho_call_positive() {
    let g = bs_greeks(&call(K), S, R, V, T);
    // Call rho is positive (higher rates → higher call value)
    assert!(g.rho > 0.0, "Call rho should be positive, got {}", g.rho);
}

#[test]
fn greeks_rho_put_negative() {
    let g = bs_greeks(&put(K), S, R, V, T);
    assert!(g.rho < 0.0, "Put rho should be negative, got {}", g.rho);
}

#[test]
fn greeks_deep_itm_call_delta_near_one() {
    let g = bs_greeks(&call(50.0), S, R, V, T); // strike=50, very deep ITM
    assert!(g.delta > 0.95, "Deep ITM call delta {}, expected near 1.0", g.delta);
}

#[test]
fn greeks_deep_otm_call_delta_near_zero() {
    let g = bs_greeks(&call(200.0), S, R, V, T);
    assert!(g.delta < 0.05, "Deep OTM call delta {}, expected near 0.0", g.delta);
}

#[test]
fn greeks_deep_itm_put_delta_near_neg_one() {
    let g = bs_greeks(&put(200.0), S, R, V, T);
    assert!(g.delta < -0.95, "Deep ITM put delta {}, expected near -1.0", g.delta);
}

#[test]
fn greeks_deep_otm_put_delta_near_zero() {
    let g = bs_greeks(&put(50.0), S, R, V, T);
    assert!(g.delta > -0.05, "Deep OTM put delta {}, expected near 0.0", g.delta);
}

#[test]
fn greeks_gamma_reference_value() {
    // Gamma for ATM: n(d1) / (S * vol * sqrt(T))
    let g = bs_greeks(&call(K), S, R, V, T);
    // scipy reference: ~0.0188
    assert!((g.gamma - 0.0188).abs() < 0.002, "Gamma = {}", g.gamma);
}

#[test]
fn greeks_vega_reference_value() {
    let g = bs_greeks(&call(K), S, R, V, T);
    // Vega for ATM 1yr: S * n(d1) * sqrt(T) ≈ 37.52
    assert!((g.vega - 37.52).abs() < 1.0, "Vega = {}", g.vega);
}

#[test]
fn greeks_expired_call_itm_delta() {
    let g = bs_greeks(&call(90.0), S, R, V, 0.0);
    assert!((g.delta - 1.0).abs() < 1e-10, "Expired ITM call delta should be 1.0");
    assert!((g.gamma).abs() < 1e-10);
    assert!((g.vega).abs() < 1e-10);
}

#[test]
fn greeks_expired_call_otm_delta() {
    let g = bs_greeks(&call(110.0), S, R, V, 0.0);
    assert!((g.delta - 0.0).abs() < 1e-10, "Expired OTM call delta should be 0.0");
}

#[test]
fn greeks_expired_put_itm_delta() {
    let g = bs_greeks(&put(110.0), S, R, V, 0.0);
    assert!((g.delta - (-1.0)).abs() < 1e-10);
}

#[test]
fn greeks_expired_put_otm_delta() {
    let g = bs_greeks(&put(90.0), S, R, V, 0.0);
    assert!(g.delta.abs() < 1e-10);
}

#[test]
fn greeks_non_option_returns_zero() {
    let g = bs_greeks(&InstrumentKind::Spot, S, R, V, T);
    assert!((g.delta).abs() < 1e-10);
    assert!((g.gamma).abs() < 1e-10);
    assert!((g.theta).abs() < 1e-10);
    assert!((g.vega).abs() < 1e-10);
    assert!((g.rho).abs() < 1e-10);
}

#[test]
fn greeks_put_call_delta_parity() {
    // delta_call - delta_put ≈ 1.0 (approximately, due to continuous dividends = 0)
    let gc = bs_greeks(&call(K), S, R, V, T);
    let gp = bs_greeks(&put(K), S, R, V, T);
    assert!((gc.delta - gp.delta - 1.0).abs() < 0.01, 
        "Call delta {} - Put delta {} should ≈ 1.0", gc.delta, gp.delta);
}

// ============================================================================
// bs_implied_vol — convergence failures, non-option, edge cases
// ============================================================================

#[test]
fn iv_non_option_returns_none() {
    assert!(bs_implied_vol(&InstrumentKind::Spot, S, 5.0, R, T).is_none());
}

#[test]
fn iv_negative_price_returns_none() {
    assert!(bs_implied_vol(&call(K), S, -1.0, R, T).is_none());
}

#[test]
fn iv_zero_price_returns_none() {
    assert!(bs_implied_vol(&call(K), S, 0.0, R, T).is_none());
}

#[test]
fn iv_zero_spot_returns_none() {
    assert!(bs_implied_vol(&call(K), 0.0, 5.0, R, T).is_none());
}

#[test]
fn iv_zero_tte_returns_none() {
    assert!(bs_implied_vol(&call(K), S, 5.0, R, 0.0).is_none());
}

#[test]
fn iv_round_trip_otm_call() {
    let vol = 0.30;
    let price = bs_price(&call(120.0), S, R, vol, T);
    let iv = bs_implied_vol(&call(120.0), S, price, R, T);
    assert!(iv.is_some(), "IV solver should converge for OTM call");
    assert!((iv.unwrap() - vol).abs() < 0.01, "IV {} ≈ {vol}", iv.unwrap());
}

#[test]
fn iv_round_trip_itm_call() {
    let vol = 0.25;
    let price = bs_price(&call(80.0), S, R, vol, T);
    let iv = bs_implied_vol(&call(80.0), S, price, R, T);
    assert!(iv.is_some());
    assert!((iv.unwrap() - vol).abs() < 0.01);
}

#[test]
fn iv_round_trip_high_vol() {
    // Crypto-like volatility
    let vol = 1.50;
    let price = bs_price(&call(K), S, R, vol, T);
    let iv = bs_implied_vol(&call(K), S, price, R, T);
    assert!(iv.is_some(), "IV solver should converge for high vol");
    assert!((iv.unwrap() - vol).abs() < 0.05, "IV {} ≈ {vol}", iv.unwrap());
}

#[test]
fn iv_round_trip_short_tte() {
    let vol = 0.20;
    let tte = 1.0 / 365.0; // 1 day
    let price = bs_price(&call(K), S, R, vol, tte);
    if price > 1e-7 {
        let iv = bs_implied_vol(&call(K), S, price, R, tte);
        // May or may not converge for very short TTE — just ensure no panic
        if let Some(iv_val) = iv {
            assert!(iv_val > 0.0);
        }
    }
}

// ============================================================================
// aggregate_greeks — edge cases
// ============================================================================

#[test]
fn aggregate_empty_positions() {
    let g = aggregate_greeks(&[]);
    assert!((g.delta).abs() < 1e-10);
    assert!((g.gamma).abs() < 1e-10);
    assert!((g.theta).abs() < 1e-10);
    assert!((g.vega).abs() < 1e-10);
    assert!((g.rho).abs() < 1e-10);
}

#[test]
fn aggregate_single_position() {
    let g = Greeks { delta: 0.5, gamma: 0.02, theta: -0.01, vega: 30.0, rho: 5.0 };
    let result = aggregate_greeks(&[(2.0, 1.0, g)]);
    assert!((result.delta - 1.0).abs() < 1e-10);
    assert!((result.gamma - 0.04).abs() < 1e-10);
    assert!((result.theta - (-0.02)).abs() < 1e-10);
    assert!((result.vega - 60.0).abs() < 1e-10);
    assert!((result.rho - 10.0).abs() < 1e-10);
}

#[test]
fn aggregate_all_five_fields() {
    let g1 = Greeks { delta: 0.6, gamma: 0.02, theta: -0.05, vega: 40.0, rho: 8.0 };
    let g2 = Greeks { delta: -0.4, gamma: 0.01, theta: -0.03, vega: 20.0, rho: -3.0 };
    // Long 10 of g1 (mult=1), Short 5 of g2 (mult=2)
    let result = aggregate_greeks(&[(10.0, 1.0, g1), (-5.0, 2.0, g2)]);
    // g1: 10*1*0.6=6, 10*1*0.02=0.2, 10*1*(-0.05)=-0.5, 10*1*40=400, 10*1*8=80
    // g2: -5*2*(-0.4)=4, -5*2*0.01=-0.1, -5*2*(-0.03)=0.3, -5*2*20=-200, -5*2*(-3)=30
    assert!((result.delta - 10.0).abs() < 1e-10);
    assert!((result.gamma - 0.1).abs() < 1e-10);
    assert!((result.theta - (-0.2)).abs() < 1e-10);
    assert!((result.vega - 200.0).abs() < 1e-10);
    assert!((result.rho - 110.0).abs() < 1e-10);
}

#[test]
fn aggregate_zero_quantity_no_effect() {
    let g = Greeks { delta: 0.5, gamma: 0.02, theta: -0.01, vega: 30.0, rho: 5.0 };
    let result = aggregate_greeks(&[(0.0, 1.0, g)]);
    assert!((result.delta).abs() < 1e-10);
}

#[test]
fn aggregate_zero_multiplier_no_effect() {
    let g = Greeks { delta: 0.5, gamma: 0.02, theta: -0.01, vega: 30.0, rho: 5.0 };
    let result = aggregate_greeks(&[(10.0, 0.0, g)]);
    assert!((result.delta).abs() < 1e-10);
}

// ============================================================================
// Numerical stability — very short TTE, very high vol
// ============================================================================

#[test]
fn greeks_very_short_tte() {
    // 1 minute to expiry
    let tte = 1.0 / (365.25 * 24.0 * 60.0);
    let g = bs_greeks(&call(K), S, R, V, tte);
    // Should not panic, delta should be ~0.5 for ATM
    assert!(g.delta.is_finite());
    assert!(g.gamma.is_finite());
}

#[test]
fn greeks_very_high_vol() {
    // Crypto-like: 300% annualized
    let g = bs_greeks(&call(K), S, R, 3.0, T);
    assert!(g.delta.is_finite());
    assert!(g.gamma.is_finite());
    assert!(g.vega.is_finite());
    // Delta for very high vol ATM call should be close to 1 (option is almost always ITM)
    assert!(g.delta > 0.5);
}

#[test]
fn price_very_high_vol_positive() {
    let p = bs_price(&call(K), S, R, 3.0, T);
    assert!(p > 0.0);
    assert!(p.is_finite());
}
