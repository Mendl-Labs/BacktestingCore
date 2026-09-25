//! Hand-computed cases for every function of the crate. Each expected number below was worked out on paper (the
//! arithmetic is in the comment next to it), independently of the implementation.

mod common;

use common::*;
use portfolio_construct::*;

// ---------------------------------------------------------------------------------------------------------------
// capital base, sizing, risk scale
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn capital_base_is_the_minimum_of_equity_and_the_allocation() {
    assert_eq!(capital_base(100.0, Some(50.0)), 50.0);
    assert_eq!(capital_base(100.0, Some(200.0)), 100.0);
    assert_eq!(capital_base(100.0, None), 100.0);
    assert_eq!(capital_base(100.0, Some(100.0)), 100.0);
}

#[test]
fn capital_base_reaches_the_targets_through_construct() {
    // One instrument at price 10, share 1, weight 0.5.
    let inst = vec![InstrumentFacts::new("X", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.5)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.filter = TradeFilter::NONE;
    c.equity = 1000.0;
    c.allocated = Some(400.0);
    assert_eq!(c.ok().target_notional[0], 200.0); // min(1000, 400) * 0.5
    c.allocated = Some(4000.0);
    assert_eq!(c.ok().target_notional[0], 500.0); // min(1000, 4000) * 0.5
    c.allocated = None;
    assert_eq!(c.ok().target_notional[0], 500.0); // equity * 0.5
}

#[test]
fn weights_sizing_share_weight_capital_and_both_risk_factors() {
    // equity 10000, share 0.6, weight 0.5: 10000 * (0.6 * 0.5) = 3000. Risk: approval 0.8 * ladder 0.5 = 0.4 -> 1200.
    let inst = vec![InstrumentFacts::new("X", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::long_only("s", 0.6, vec![(0, 0.5)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.equity = 10000.0;
    c.allocated = None;
    let out = c.ok();
    assert_close(out.target_notional[0], 3000.0, 1e-9, "3000");
    assert_eq!(out.trades[0].quantity, out.target_notional[0] / 10.0);
    c.rs = RiskScale::new(0.8, 0.5);
    let out = c.ok();
    assert_close(out.target_notional[0], 1200.0, 1e-9, "1200");
    assert_eq!(out.risk_scale_applied, 0.4);
    // The ladder alone (approval constant 1): 0.5 -> 1500. The approval constant alone: 0.8 -> 2400.
    c.rs = RiskScale::new(1.0, 0.5);
    assert_close(c.ok().target_notional[0], 1500.0, 1e-9, "ladder only");
    c.rs = RiskScale::new(0.8, 1.0);
    assert_close(c.ok().target_notional[0], 2400.0, 1e-9, "approval only");
}

#[test]
fn opposite_sleeves_net_by_the_signed_sum_not_the_absolute_sum() {
    // Sleeve a: share 0.6, weight +0.5 -> +0.30. Sleeve b: share 0.4, weight -0.25 -> -0.10. Net +0.20 of 10000 = 2000
    // (an absolute sum would give 0.40 -> 4000, a sleeve-b-ignored book 3000).
    let inst = vec![InstrumentFacts::new("X", "v", "c", 10.0)];
    let sleeves = vec![
        SleeveTargets::signed("a", 0.6, 1.0, vec![(0, 0.5)]),
        SleeveTargets::signed("b", 0.4, 1.0, vec![(0, -0.25)]),
    ];
    let mut c = Case::new(inst, sleeves);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.equity = 10000.0;
    c.allocated = None;
    let out = c.ok();
    assert_close(out.target_notional[0], 2000.0, 1e-9, "net");
    assert_close(out.gross, 2000.0, 1e-9, "gross is of the NET target, not of the sleeves");
}

#[test]
fn approval_time_risk_scale_r2() {
    // c = min(1, 0.9 * cap / 4.41). cap 3 -> 2.7 / 4.41 = 0.612244897959...; cap 2 -> 1.8 / 4.41 = 0.408163265306...
    assert_close(approval_risk_scale(3.0), 2.7 / 4.41, 1e-15, "cap 3");
    assert_close(approval_risk_scale(3.0), 0.6122448979591837, 1e-15, "cap 3 value");
    assert_close(approval_risk_scale(2.0), 0.40816326530612246, 1e-15, "cap 2 value");
    assert_eq!(approval_risk_scale(5.0), 1.0, "a cap above 4.9 needs no scaling");
    assert_eq!(approval_risk_scale(4.9), 1.0, "0.9 * 4.9 = 4.41: exactly no scaling");
    assert!(approval_risk_scale(4.8) < 1.0);
    assert_eq!(R2_REFERENCE_P90_GROSS, 4.41);
    assert_eq!(R2_HEADROOM, 0.9);
}

#[test]
fn r2_scale_makes_a_book_that_needs_more_gross_than_the_cap_fit() {
    // A signed sleeve that wants gross 4.41x equity under a cap of 3x: refused unscaled; at c = 0.9*3/4.41 the scaled
    // gross is 0.9 * 3 = 2.7x <= 3x. The refusal is not clipped: it is a refusal until an approval records the scale.
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::signed("s", 1.0, 3.0, vec![(0, 2.205), (1, -2.205)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.equity = 1000.0;
    c.allocated = None;
    c.limits = Limits::unlimited().with_max_gross(3.0).with_leverage_max_gross(3.0);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { .. })), "4.41x > 3x is refused, not clipped");
    c.rs = RiskScale::new(approval_risk_scale(3.0), 1.0);
    let out = c.ok();
    assert_close(out.gross, 2700.0, 1e-9, "0.9 * 3 * equity");
}

// ---------------------------------------------------------------------------------------------------------------
// trade filter thresholds at the boundary (10 units, 2%)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn trade_filter_absolute_threshold_boundary() {
    let f = TradeFilter::new(10.0, 0.0);
    assert_eq!(f.check(1000.0, 990.0), None, "delta exactly 10 trades");
    assert!(matches!(f.check(1000.0, 990.01), Some(SkipReason::BelowMinAbs { min, .. }) if min == 10.0), "delta 9.99 is dropped");
    assert_eq!(f.check(1000.0, 989.99), None);
    assert!(matches!(f.check(990.01, 1000.0), Some(SkipReason::BelowMinAbs { .. })), "a reduction is filtered the same way");
}

#[test]
fn trade_filter_percentage_threshold_is_a_fraction_of_the_target() {
    let f = TradeFilter::new(0.0, 0.02);
    assert_eq!(f.check(1000.0, 980.0), None, "delta 20 = 2% of the target trades");
    assert!(matches!(f.check(1000.0, 980.01), Some(SkipReason::BelowMinPct { min, .. }) if (min - 20.0).abs() < 1e-12));
    // Reference is the TARGET, not the current holding: min_pct 0.5, target 1000, current 600: delta 400 < 0.5 * 1000.
    let half = TradeFilter::new(0.0, 0.5);
    assert!(matches!(half.check(1000.0, 600.0), Some(SkipReason::BelowMinPct { min, .. }) if min == 500.0));
    assert_eq!(half.check(1000.0, 500.0), None, "delta 500 = 0.5 * 1000 trades");
    // Target zero: the reference is the current value; a full exit is never exempt but always passes a pct < 1.
    assert_eq!(half.check(0.0, 100.0), None);
}

#[test]
fn trade_filter_defaults_and_validation() {
    assert_eq!(TradeFilter::PLANNER_DEFAULT, TradeFilter::new(10.0, 0.02));
    assert_eq!((DEFAULT_MIN_TRADE_ABS, DEFAULT_MIN_TRADE_PCT), (10.0, 0.02));
    assert!(TradeFilter::new(-1.0, 0.0).validate().is_err());
    assert!(TradeFilter::new(0.0, 1.0).validate().is_err());
    assert!(TradeFilter::new(0.0, -0.1).validate().is_err());
    assert!(TradeFilter::new(f64::NAN, 0.0).validate().is_err());
    assert!(TradeFilter::NONE.validate().is_ok());
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]);
    c.filter = TradeFilter::new(0.0, 1.0);
    assert!(matches!(c.run(), Err(ConstructRefusal::Invalid(InputError::BadParam(_)))));
}

#[test]
fn a_dropped_trade_keeps_the_holding_and_records_why() {
    // Target 1000, held 1.98 * 500 = 990, filter 10 / 2%: delta 10 passes abs but is below 2% of 1000 = 20 -> dropped.
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]).held(SPY, 1.98).with_cash(4010.0);
    c.filter = TradeFilter::PLANNER_DEFAULT;
    let out = c.ok();
    assert!(out.trades.is_empty());
    assert!(matches!(skip_reason(&out, "SPY"), SkipReason::BelowMinPct { delta, min } if (*delta - 10.0).abs() < 1e-9 && (*min - 20.0).abs() < 1e-9));
}

// ---------------------------------------------------------------------------------------------------------------
// rounding: 8 decimals toward zero, venue lot rules
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn rounding_toward_zero_at_eight_decimals() {
    assert_eq!(round_toward_zero_dp(2469.135782468, 8), 2469.13578246);
    assert_eq!(round_toward_zero_dp(-2469.135782468, 8), -2469.13578246, "a short is not made bigger");
    assert_eq!(round_toward_zero_dp(0.999999999, 8), 0.99999999);
    assert_eq!(round_toward_zero_dp(-0.999999999, 8), -0.99999999);
    assert_eq!(round_toward_zero_dp(1234.567891234, 8), 1234.56789123);
    assert_eq!(round_toward_zero_dp(0.0, 8), 0.0);
    // Decimal ties: 0.1 + 0.2 = 0.30000000000000004 and 0.3 both land on 0.3; a value 4 ulps under a quantum counts as on it.
    assert_eq!(round_toward_zero_dp(0.1 + 0.2, 8), 0.3);
    assert_eq!(round_toward_zero_dp(-(0.1 + 0.2), 8), -0.3);
    assert_eq!(round_toward_zero_dp(0.29999999999999993, 8), 0.3);
    // 1 quantum is not swallowed: 0.299999995 is below 0.3 at 8 decimals.
    assert_eq!(round_toward_zero_dp(0.299999995, 8), 0.29999999);
    assert_eq!(floor_dp(5.263157894736842, 9), 5.263157894);
    assert_eq!(floor_dp(2.5, 0), 2.0);
    assert_eq!(ceil_dp(1.250000001, 8), 1.25000001);
    assert_eq!(ceil_dp(1.25, 8), 1.25);
    assert_eq!(ceil_dp(1.25 + 4.0 * f64::EPSILON, 8), 1.25, "an exact decimal tie is not pushed up by float noise");
}

#[test]
fn quantities_round_down_to_the_lot_and_never_up() {
    let r = LotRounder::new().with("X", LotRule::new(2)).with("W", LotRule::new(0));
    assert_eq!(r.round_quantity("X", Side::Buy, 1.239, 1.0), Ok(1.23));
    assert_eq!(r.round_quantity("X", Side::Sell, 1.2399999, 1.0), Ok(1.23));
    assert_eq!(r.round_quantity("X", Side::Buy, 1.24, 1.0), Ok(1.24));
    assert_eq!(r.round_quantity("W", Side::Buy, 2.999999, 1.0), Ok(2.0));
    assert_eq!(r.round_quantity("W", Side::Buy, 0.999999, 1.0), Err(SizeRefusal::RoundsToZero));
    assert_eq!(r.round_quantity("Y", Side::Buy, 1.0, 1.0), Err(SizeRefusal::UnknownInstrument("Y".into())));
    assert_eq!(ExactUnits.round_quantity("anything", Side::Buy, 1.23456789012, 1.0), Ok(1.23456789012));
    let m = LotRounder::new().with("M", LotRule::new(1).with_min_quantity(0.5).with_min_notional(20.0).with_max_order_units(10.0));
    assert_eq!(m.round_quantity("M", Side::Buy, 0.49, 100.0), Err(SizeRefusal::BelowMinQuantity { min: 0.5 }));
    assert_eq!(m.round_quantity("M", Side::Buy, 0.5, 39.0), Err(SizeRefusal::BelowMinCost { min: 20.0 }));
    assert_eq!(m.round_quantity("M", Side::Buy, 0.5, 40.0), Ok(0.5));
    assert!(matches!(m.round_quantity("M", Side::Buy, 10.1, 100.0), Err(SizeRefusal::Other(_))));
}

#[test]
fn a_full_exit_sells_exactly_the_held_quantity_with_no_dust() {
    // 0.7 units at 30.3: (0.7 * 30.3) / 30.3 = 0.7000000000000001 in f64, so sizing an exit as |delta| / price would leave
    // or overshoot float dust. The exit must be exactly the held quantity. (No rounder: exact fractional units.)
    let inst = vec![InstrumentFacts::new("X", "v", "c", 30.3).with_held(0.7)];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.0)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.filter = TradeFilter::NONE;
    let out = c.ok();
    assert_eq!(out.trades.len(), 1);
    assert_eq!(out.trades[0].quantity, 0.7);
    assert_eq!(out.trades[0].side, Side::Sell);
}

// ---------------------------------------------------------------------------------------------------------------
// whole-book refusals: gross-cap boundary at 1e-9, position, class, net, shorting
// ---------------------------------------------------------------------------------------------------------------

fn two_long(cap_gross: f64, w: f64) -> Case {
    // equity 1000, share 1, weights (w, w): gross = 2000 * w against cap = cap_gross * 1000.
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::signed("s", 1.0, 3.0, vec![(0, w), (1, w)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.equity = 1000.0;
    c.allocated = None;
    c.limits = Limits::unlimited().with_max_gross(cap_gross).with_leverage_max_gross(cap_gross);
    c
}

#[test]
fn gross_cap_boundary_permits_the_cap_and_refuses_the_whole_book_above_it() {
    // cap 1.5x = 1500; weights 0.75 each = 1500 exactly: permitted.
    let out = two_long(1.5, 0.75).ok();
    assert_eq!(out.gross, 1500.0);
    // 1e-9 relative over the cap: the WHOLE book is refused, so there is no plan at all (no partial trades).
    let over = two_long(1.5, 0.75 * (1.0 + 1e-9)).run();
    match over {
        Err(ConstructRefusal::GrossAboveCap { gross, cap }) => {
            assert_eq!(cap, 1500.0);
            assert!(gross > 1500.0 && gross < 1500.000002);
        }
        other => panic!("expected the whole book to be refused, got {other:?}"),
    }
    // Float noise under the 1e-12 tolerance is not a breach (the exact decimal is exactly at the cap).
    let noise = two_long(1.5, 0.75 * (1.0 + 1e-14)).ok();
    assert!(noise.gross >= 1500.0);
    // 1e-11 over is a breach (above the tolerance).
    assert!(matches!(two_long(1.5, 0.75 * (1.0 + 1e-11)).run(), Err(ConstructRefusal::GrossAboveCap { .. })));
}

#[test]
fn gross_cap_permits_a_tie_that_float_noise_would_break() {
    // Weights 0.1 / 0.55 / 0.55 of 12345 give 14814.000000000002 in f64 (exactly 14814 in decimal) against the cap
    // 1.2 * 12345 = 14814.0. The exact-decimal planner permits it; so must the f64 spec.
    let cb = 12345.0;
    let mut ts = [cb * 0.1, cb * 0.55, cb * 0.55];
    ts.sort_by(f64::total_cmp);
    let noisy = ts[0] + ts[1] + ts[2];
    assert!(noisy > 1.2 * cb, "premise: the f64 gross ({noisy}) exceeds the cap ({}) by noise", 1.2 * cb);
    let inst = vec![
        InstrumentFacts::new("A", "v", "c", 10.0),
        InstrumentFacts::new("B", "v", "c", 10.0),
        InstrumentFacts::new("C", "v", "c", 10.0),
    ];
    let sleeve = vec![SleeveTargets::signed("s", 1.0, 1.0, vec![(0, 0.1), (1, 0.55), (2, 0.55)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.equity = cb;
    c.allocated = None;
    c.limits = Limits::unlimited().with_max_gross(1.2).with_leverage_max_gross(1.2);
    assert!(c.run().is_ok());
}

#[test]
fn gross_cap_is_the_smaller_of_max_gross_and_leverage() {
    let mut c = two_long(10.0, 0.75); // gross 1500
    c.limits = Limits::unlimited().with_max_gross(2.0).with_leverage_max_gross(1.0); // effective cap 1.0 -> 1000
    assert_eq!(c.limits.effective_max_gross(), 1.0);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { cap, .. }) if cap == 1000.0));
    c.limits = Limits::unlimited().with_max_gross(1.0).with_leverage_max_gross(2.0);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { cap, .. }) if cap == 1000.0));
}

#[test]
fn a_gross_breach_is_refused_not_clipped_even_when_clipping_would_fit() {
    // gross 1600 vs cap 1500: clipping every target by 1500/1600 would fit; the spec refuses.
    let c = two_long(1.5, 0.8);
    match c.run() {
        Err(ConstructRefusal::GrossAboveCap { gross, cap }) => assert_eq!((gross, cap), (1600.0, 1500.0)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn position_class_and_net_caps_refuse_the_whole_book() {
    let inst = vec![
        InstrumentFacts::new("A", "v", "eq", 10.0),
        InstrumentFacts::new("B", "v", "eq", 10.0),
        InstrumentFacts::new("C", "v", "fx", 10.0),
    ];
    let mk = |limits: Limits, w: [f64; 3]| {
        let sleeve = vec![SleeveTargets::signed("s", 1.0, 3.0, vec![(0, w[0]), (1, w[1]), (2, w[2])])];
        let mut c = Case::new(inst.clone(), sleeve);
        c.funding = Funding::Unconstrained;
        c.rounder = None;
        c.target_dp = None;
        c.filter = TradeFilter::NONE;
        c.equity = 1000.0;
        c.allocated = None;
        c.limits = limits;
        c
    };
    // per position: cap 0.4 of 1000 = 400; B at 0.5 -> 500.
    let pos = Limits::unlimited().with_max_position(0.4);
    assert!(mk(pos.clone(), [0.4, 0.4, 0.4]).run().is_ok(), "exactly at the cap is fine");
    match mk(pos, [0.3, 0.5, 0.3]).run() {
        Err(ConstructRefusal::PositionAboveCap { symbol, notional, cap, .. }) => assert_eq!((symbol.as_str(), notional, cap), ("B", 500.0, 400.0)),
        other => panic!("{other:?}"),
    }
    // asset class: eq cap 0.6 of 1000 = 600; A + B = 700.
    let class = Limits::unlimited().with_class_cap("EQ", 0.6);
    assert!(mk(class.clone(), [0.3, 0.3, 0.9]).run().is_ok());
    match mk(class, [0.3, 0.4, 0.1]).run() {
        Err(ConstructRefusal::ClassAboveCap { class, gross, cap }) => assert_eq!((class.as_str(), gross, cap), ("eq", 700.0, 600.0)),
        other => panic!("{other:?}"),
    }
    // net: cap 0.5 of 1000 = 500; +0.5 +0.4 -0.2 = +0.7 -> 700 refused; +0.5 -0.2 -0.2 = 0.1 fine.
    let net = Limits::unlimited().with_max_net(0.5);
    assert!(mk(net.clone(), [0.5, -0.2, -0.2]).run().is_ok());
    match mk(net.clone(), [0.5, 0.4, -0.2]).run() {
        Err(ConstructRefusal::NetAboveCap { net, cap }) => assert_eq!((net, cap), (700.0, 500.0)),
        other => panic!("{other:?}"),
    }
    // net is on the SIGNED sum: a short book breaches it too.
    assert!(matches!(mk(net, [-0.5, -0.4, 0.0]).run(), Err(ConstructRefusal::NetAboveCap { .. })));
}

#[test]
fn shorting_forbidden_refuses_a_book_with_any_negative_target() {
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::signed("s", 1.0, 1.0, vec![(0, 0.3), (1, -0.1)])];
    let mut c = Case::new(inst, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.equity = 1000.0;
    c.allocated = None;
    c.limits = Limits::unlimited().with_shorting(false);
    match c.run() {
        Err(ConstructRefusal::ShortingForbidden { symbol, target, .. }) => assert_eq!((symbol.as_str(), target), ("B", -100.0)),
        other => panic!("{other:?}"),
    }
    c.limits = Limits::unlimited().with_shorting(true);
    assert!(c.run().is_ok());
    // All-long is fine even with shorting off.
    c.sleeves = vec![SleeveTargets::signed("s", 1.0, 1.0, vec![(0, 0.3), (1, 0.1)])];
    c.limits = Limits::unlimited().with_shorting(false);
    assert!(c.run().is_ok());
}

#[test]
fn binding_constraints_are_reported_closest_first() {
    let mut c = two_long(1.5, 0.75); // gross exactly at the cap
    c.limits = Limits::unlimited().with_max_gross(1.5).with_leverage_max_gross(1.5).with_max_position(1.0).with_class_cap("c", 2.0);
    let out = c.ok();
    let first = &out.binding[0];
    assert_eq!((first.kind, first.bound), (BindingKind::Gross, true));
    assert_eq!((first.used, first.cap), (1500.0, 1500.0));
    assert!(out.binding.windows(2).all(|w| w[0].ratio() >= w[1].ratio()));
    // Position: largest target 750 of a cap of 1000 -> ratio 0.75; class: 1500 of 2000 -> 0.75; neither bound.
    let pos = out.binding.iter().find(|b| b.kind == BindingKind::Position).unwrap();
    assert_eq!((pos.used, pos.cap, pos.bound), (750.0, 1000.0, false));
    let cls = out.binding.iter().find(|b| b.kind == BindingKind::AssetClass).unwrap();
    assert_eq!((cls.label.as_str(), cls.used, cls.cap), ("c", 1500.0, 2000.0));
    // The allocation cap shows as a bound capital base when it binds.
    let mut c = two_long(10.0, 0.1);
    c.equity = 1000.0;
    c.allocated = Some(500.0);
    let out = c.ok();
    let cb = out.binding.iter().find(|b| b.kind == BindingKind::CapitalBase).unwrap();
    assert_eq!((cb.used, cb.cap), (1000.0, 500.0));
    assert_eq!(out.capital_base, 500.0);
}

// ---------------------------------------------------------------------------------------------------------------
// margin models (R3 OANDA, R4 Alpaca Reg T)
// ---------------------------------------------------------------------------------------------------------------

fn oanda_book() -> Vec<InstrumentFacts> {
    vec![
        InstrumentFacts::new("EUR_USD", "oanda", "fx", 1.0).with_margin_rate(0.02),
        InstrumentFacts::new("USD_JPY", "oanda", "fx", 1.0).with_margin_rate(0.05),
    ]
}

fn margin_case(instruments: Vec<InstrumentFacts>, model: Box<dyn MarginModel>, w: [f64; 2], equity: f64) -> Case {
    let sleeve = vec![SleeveTargets::signed("s", 1.0, 3.0, vec![(0, w[0]), (1, w[1])])];
    let mut c = Case::new(instruments, sleeve);
    c.funding = Funding::Unconstrained;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.equity = equity;
    c.allocated = None;
    c.margin = model;
    c
}

#[test]
fn oanda_margin_is_per_position_with_no_offset() {
    // NAV 10000. Long 100000 EUR_USD (weight +10) is not allowed by max_abs 3, so use notional 30000 = weight 3 and
    // short 20000 USD_JPY (weight -2): margin = 0.02 * 30000 + 0.05 * 20000 = 600 + 1000 = 1600. A netting model would
    // give 0.02 * 10000 or less; OANDA charges every position (measured 2026-09-23).
    let c = margin_case(oanda_book(), Box::new(OandaMargin::r3_default()), [3.0, -2.0], 10000.0);
    let out = c.ok();
    assert_close(out.margin_used, 1600.0, 1e-9, "margin");
    assert_close(OandaMargin::margin_call_percent(out.margin_used, 10000.0), 0.16, 1e-12, "call percent");
    assert_close(OandaMargin::margin_closeout_percent(out.margin_used, 10000.0), 0.08, 1e-12, "closeout percent");
    assert_eq!(out.binding.iter().find(|b| b.kind == BindingKind::Margin).unwrap().cap, 5000.0);
}

#[test]
fn oanda_margin_ceiling_is_fifty_percent_of_nav_and_refuses_the_whole_book() {
    // NAV 10000, ceiling 5000. EUR_USD +3 (30000 -> 600) and USD_JPY -3 (30000 -> 1500): 2100, fine.
    // Rates are per position: a book of 0.02 * 30000 + 0.05 * 30000 stays far below. To reach the ceiling use a
    // custom ceiling of 0.2: 2000 < 2100 -> refused.
    let ok = margin_case(oanda_book(), Box::new(OandaMargin::r3_default()), [3.0, -3.0], 10000.0).ok();
    assert_close(ok.margin_used, 2100.0, 1e-9, "margin");
    let tight = margin_case(oanda_book(), Box::new(OandaMargin::with_ceiling(0.2)), [3.0, -3.0], 10000.0);
    match tight.run() {
        Err(ConstructRefusal::MarginExceeded { model, used, ceiling }) => {
            assert_eq!(model, "oanda");
            assert_close(used, 2100.0, 1e-9, "used");
            assert_close(ceiling, 2000.0, 1e-9, "ceiling");
        }
        other => panic!("{other:?}"),
    }
    // Exactly at the ceiling passes: ceiling 0.21 -> 2100.
    assert!(margin_case(oanda_book(), Box::new(OandaMargin::with_ceiling(0.21)), [3.0, -3.0], 10000.0).run().is_ok());
    assert_eq!(OANDA_DEFAULT_MARGIN_CEILING, 0.5);
    // A missing marginRate fails closed (no guess).
    let mut inst = oanda_book();
    inst[1].margin_rate = None;
    assert!(matches!(
        margin_case(inst, Box::new(OandaMargin::r3_default()), [1.0, -1.0], 10000.0).run(),
        Err(ConstructRefusal::Invalid(InputError::MarginRateUnknown(s))) if s == "USD_JPY"
    ));
}

#[test]
fn oanda_buying_power_records_the_binding_factor() {
    let m = OandaMargin::r3_default();
    let eur = &oanda_book()[0];
    let jpy = &oanda_book()[1];
    // NAV 10000, margin used 4500: available 5500, ceiling room 500 (0.5 * 10000 - 4500). The ceiling binds:
    // notional 500 / 0.02 = 25000 in EUR_USD, 500 / 0.05 = 10000 in USD_JPY.
    let bp = m.buying_power(10000.0, 4500.0, eur).unwrap();
    assert_eq!(bp.bound_by, BuyingPowerFactor::Ceiling);
    assert_close(bp.notional, 25000.0, 1e-9, "EUR");
    assert_close(m.buying_power(10000.0, 4500.0, jpy).unwrap().notional, 10000.0, 1e-9, "JPY");
    // With a ceiling above NAV the margin actually available binds: 5500 / 0.02 = 275000.
    let loose = OandaMargin::with_ceiling(2.0);
    let bp = loose.buying_power(10000.0, 4500.0, eur).unwrap();
    assert_eq!(bp.bound_by, BuyingPowerFactor::MarginAvailable);
    assert_close(bp.notional, 275000.0, 1e-6, "available");
    // Over the ceiling: nothing left.
    assert_eq!(m.buying_power(10000.0, 6000.0, eur).unwrap().notional, 0.0);
}

fn alpaca_book() -> Vec<InstrumentFacts> {
    vec![
        InstrumentFacts::new("SPY", "alpaca", "us_etf", 100.0).with_margin_rate(0.25), // asset requirement below the floor
        InstrumentFacts::new("XYZ", "alpaca", "us_etf", 100.0).with_margin_rate(0.75), // above the floor
        InstrumentFacts::new("BTC", "alpaca", "crypto", 100.0).with_marginable(false),
        InstrumentFacts::new("NOREQ", "alpaca", "us_etf", 100.0), // no stated requirement
    ]
}

#[test]
fn alpaca_regt_margin_rate_is_max_of_asset_requirement_and_half() {
    let m = AlpacaRegT::r4_default();
    let b = alpaca_book();
    assert_eq!(m.rate(&b[0]), Some(0.5), "max(0.25, 0.50)");
    assert_eq!(m.rate(&b[1]), Some(0.75), "max(0.75, 0.50)");
    assert_eq!(m.rate(&b[2]), Some(1.0), "non-marginable: 100%");
    assert_eq!(m.rate(&b[3]), Some(0.5), "no requirement stated: the 0.50 floor");
    assert_eq!(ALPACA_MIN_OPENING_MARGIN_RATE, 0.5);
    // margin_used: 30000 SPY long (15000) + 20000 XYZ short (15000) + 10000 BTC (10000, non-marginable) = 40000.
    let used = m.margin_used(&[(&b[0], 30000.0), (&b[1], -20000.0), (&b[2], 10000.0)]).unwrap();
    assert_close(used, 40000.0, 1e-9, "margin used");
}

#[test]
fn alpaca_regt_refuses_a_book_whose_initial_margin_exceeds_equity() {
    // equity 20000: 2x gross of marginable names at 0.5 = 20000 margin = equity -> allowed (tie).
    let inst = || vec![InstrumentFacts::new("A", "alpaca", "us_etf", 10.0), InstrumentFacts::new("B", "alpaca", "us_etf", 10.0)];
    let mk = |w: f64| margin_case(inst(), Box::new(AlpacaRegT::r4_default()), [w, w], 20000.0);
    assert_close(mk(1.0).ok().margin_used, 20000.0, 1e-9, "at the ceiling");
    match mk(1.0 + 1e-9).run() {
        Err(ConstructRefusal::MarginExceeded { model, .. }) => assert_eq!(model, "alpaca_regt"),
        other => panic!("{other:?}"),
    }
    // With the floor dropped (rate = asset requirement 0.25) 2x gross would pass; the R4 floor makes it refuse at 2.4x.
    let mut inst2 = inst();
    inst2[0].margin_rate = Some(0.25);
    inst2[1].margin_rate = Some(0.25);
    let c = margin_case(inst2, Box::new(AlpacaRegT::r4_default()), [1.2, 1.2], 20000.0); // gross 48000, margin 24000
    assert!(matches!(c.run(), Err(ConstructRefusal::MarginExceeded { .. })));
    // A non-marginable instrument is 100%: 25000 of it on 20000 equity refuses.
    let nm = vec![InstrumentFacts::new("N", "alpaca", "crypto", 10.0).with_marginable(false), InstrumentFacts::new("B", "alpaca", "us_etf", 10.0)];
    assert!(matches!(
        margin_case(nm, Box::new(AlpacaRegT::r4_default()), [1.25, 0.0], 20000.0).run(),
        Err(ConstructRefusal::MarginExceeded { .. })
    ));
}

#[test]
fn alpaca_broker_buying_power_is_the_smaller_of_the_two_figures() {
    // R4: use the smaller of buying_power and regt_buying_power (measured 4x = 400000 and 2x = 200000 on a 100000 account).
    let m = AlpacaRegT::with_broker_figures(400000.0, 200000.0);
    assert_eq!(m.broker_buying_power, Some(200000.0));
    let i = InstrumentFacts::new("SPY", "alpaca", "us_etf", 100.0);
    // equity 100000, nothing used, rate 0.5: model room = 200000; the broker figure 200000 ties (model factor).
    assert_close(m.buying_power(100000.0, 0.0, &i).unwrap().notional, 200000.0, 1e-9, "room");
    let smaller = AlpacaRegT::with_broker_figures(150000.0, 200000.0);
    let bp = smaller.buying_power(100000.0, 0.0, &i).unwrap();
    assert_eq!((bp.bound_by, bp.notional), (BuyingPowerFactor::BrokerFigure, 150000.0));
    // Margin already used shrinks the room: used 60000 -> (100000 - 60000) / 0.5 = 80000.
    let bp = AlpacaRegT::r4_default().buying_power(100000.0, 60000.0, &i).unwrap();
    assert_eq!((bp.bound_by, bp.notional), (BuyingPowerFactor::MarginAvailable, 80000.0));
}

#[test]
fn no_margin_model_uses_no_margin_and_has_no_ceiling() {
    let c = margin_case(oanda_book(), Box::new(NoMargin), [3.0, -3.0], 10000.0);
    let out = c.ok();
    assert_eq!(out.margin_used, 0.0);
    assert!(out.binding.iter().all(|b| b.kind != BindingKind::Margin));
}

// ---------------------------------------------------------------------------------------------------------------
// funding: reserve, fees, common factor
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn cash_budget_scales_every_increase_by_one_common_factor_with_a_fee_reserve() {
    // Two buys of 500 each (fee 0.2% = 1 each), equity 1000, cash 800, reserve 10% of 1000 = 100 -> available 700.
    // Total cost 1002 > 700: usable = 700 - 2e-8; factor = usable / 1002 = 0.6986...; each buys factor * 50 units.
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.5), (1, 0.5)])];
    let mut c = Case::new(inst, sleeve);
    c.equity = 1000.0;
    c.allocated = None;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.funding = Funding::Cash { cash: 800.0, reserve_fraction: 0.1, fee_rate: 0.002, credit_sell_proceeds: true };
    let out = c.ok();
    let factor = (700.0 - 2e-8) / 1002.0;
    assert_close(out.trades[0].quantity, 50.0 * factor, 1e-9, "A");
    assert_eq!(out.trades[0].quantity, out.trades[1].quantity, "one common factor");
    let spent: f64 = out.trades.iter().map(|t| t.notional + t.est_fee).sum();
    assert!(spent <= 700.0 && spent > 699.999, "{spent}");
    assert_close(out.funding_left.unwrap(), 800.0 - spent, 1e-9, "cash left");
    // Ample cash: nothing is scaled and the fee is ceil8(500 * 0.002) = 1.
    c.funding = Funding::Cash { cash: 5000.0, reserve_fraction: 0.1, fee_rate: 0.002, credit_sell_proceeds: true };
    let out = c.ok();
    assert_eq!((out.trades[0].quantity, out.trades[0].est_fee), (50.0, 1.0));
    assert_close(out.funding_left.unwrap(), 5000.0 - 1002.0, 1e-9, "left");
}

#[test]
fn reserve_is_rounded_up_to_eight_decimals() {
    // reserve = ceil8(0.05 * 3333.333333335) = ceil8(166.66666666675) = 166.66666667 (planner test `reserve_rounds_up`).
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0)];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 1.0)])];
    let mut c = Case::new(inst, sleeve);
    c.equity = 3333.333333335;
    c.allocated = None;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    // cash exactly reserve + 100: available must be 100 - 1.5e-9 slack-free: buy 100 of cost.
    c.funding = Funding::Cash { cash: 166.66666667 + 100.0, reserve_fraction: 0.05, fee_rate: 0.0, credit_sell_proceeds: true };
    let out = c.ok();
    // available = 266.66666667 - 166.66666667 = 100, less the 1e-8 per-buy slack: the notional is 99.99999999. A reserve
    // rounded DOWN (166.66666666) would leave 100.00000001 and buy 100.00000000.
    let n = out.trades[0].notional;
    assert!(n < 99.999999995 && n > 99.99999, "{n}");
}

// ---------------------------------------------------------------------------------------------------------------
// cadence flags and scope
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn an_out_of_scope_instrument_counts_in_the_checks_but_is_not_traded() {
    let inst = vec![
        InstrumentFacts::new("A", "v", "c", 10.0),
        InstrumentFacts::new("B", "v", "c", 10.0).with_in_scope(false),
    ];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.5), (1, 0.5)])];
    let mut c = Case::new(inst, sleeve);
    c.equity = 1000.0;
    c.allocated = None;
    c.rounder = None;
    c.target_dp = None;
    c.filter = TradeFilter::NONE;
    c.funding = Funding::Unconstrained;
    let out = c.ok();
    assert_eq!(out.trades.len(), 1);
    assert_eq!(out.trades[0].symbol, "A");
    assert_eq!(out.gross, 1000.0, "B's target still counts toward gross");
    assert_eq!(out.target_notional[1], 500.0);
    // A cap on the whole book still refuses because of the out-of-scope instrument's target.
    c.limits = Limits::unlimited().with_max_gross(0.9);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { .. })));
}

#[test]
fn an_instrument_without_a_price_is_skipped_and_contributes_nothing() {
    let inst = vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", 10.0).without_price()];
    let sleeve = vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.5), (1, 0.5)])];
    let mut c = Case::new(inst, sleeve);
    c.equity = 1000.0;
    c.allocated = None;
    c.rounder = None;
    c.filter = TradeFilter::NONE;
    c.funding = Funding::Unconstrained;
    let out = c.ok();
    assert_eq!(*skip_reason(&out, "B"), SkipReason::NoPrice);
    assert_eq!(out.gross, 500.0);
    assert_eq!(out.target_notional[1], 0.0);
    assert!(out.named[1]);
    // A non-positive or NaN price is the same.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut c2 = Case::new(
            vec![InstrumentFacts::new("A", "v", "c", 10.0), InstrumentFacts::new("B", "v", "c", bad)],
            vec![SleeveTargets::long_only("s", 1.0, vec![(0, 0.5), (1, 0.5)])],
        );
        c2.rounder = None;
        c2.funding = Funding::Unconstrained;
        assert_eq!(*skip_reason(&c2.ok(), "B"), SkipReason::NoPrice, "{bad}");
    }
}

#[test]
fn projected_gross_includes_unmanaged_positions_and_drives_needs_margin() {
    // Targets 1000 (long only); unmanaged 4500 on equity 5000: projected 5500 > equity -> needs margin.
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]);
    c.unmanaged = 4500.0;
    let out = c.ok();
    assert_eq!(out.projected_gross, 5500.0);
    assert!(out.needs_margin);
    c.unmanaged = 3000.0;
    let out = c.ok();
    assert_eq!(out.projected_gross, 4000.0);
    assert!(!out.needs_margin);
    // With a signed sleeve and a cash budget the levered projection is refused.
    let mut s = Case::planner(vec![SleeveTargets::signed("ls", 1.0, 2.0, vec![(SPY, 0.2)])]);
    s.unmanaged = 4500.0;
    assert_eq!(s.run(), Err(ConstructRefusal::BuyingPowerRequired { gross: 5500.0 }));
}
