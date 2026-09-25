//! Golden vectors taken from the live planner's own tests and reproduced with the f64 `construct`
//! (design PF2: "`construct` vs planner golden vectors exported from SE").
//!
//! Provenance. Every case below is transcribed BY HAND from `Mendl-Labs/SignalEngine` `main` (4937a43),
//! `crates/rebalancer-core/tests/planner.rs`, `tests/signed.rs`, `tests/oanda_rules.rs` and `src/planner.rs` doc
//! examples. Inputs (prices, holdings, shares, weights, caps, cash) and the asserted outputs (quantities, target
//! notionals, buying power left, skip reasons, refusal values) are the planner tests' own numbers. The Decimal
//! expectations are compared here with f64 `==` after the 8-decimal quantum where the planner asserts an exact decimal
//! (the f64 spelling of `5.263157894` is the same double), and with an explicit tolerance where the planner asserts an
//! inequality.
//!
//! What is mirrored and what is not (the full list is in the crate README):
//! * mirrored: sizing, capital base, risk scale, min-trade thresholds, rounding by lot rules, sells-first ordering,
//!   cash budget with reserve and fees (common factor), crediting sell proceeds, buying-power budget, signed sleeves,
//!   gross-cap refusal (value and cap), crossing zero as two legs, held shorts, zero targets, input validation,
//!   toward-zero rounding of shorts;
//! * NOT mirrored (no counterpart in a pure sizing crate): everything the guard decides (denials, halted account, stale
//!   price, orders per day, turnover, universe, mandate standing), client tags and the inputs digest, the
//!   `reference-rules` decision conversion, the adapters' own `prepare_order` (a table stands in for it), and the 1,500
//!   hash-only cases of `long_only_golden.rs` (they are SHA-256 hashes of plans, not extractable inputs and outputs).

mod common;

use common::*;
use portfolio_construct::*;

// ---------------------------------------------------------------------------------------------------------------
// Basic plans (planner.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn planner_flat_account_buys_are_scaled_to_the_cash_above_the_reserve() {
    let out = Case::planner(both_sleeves()).ok();
    assert_eq!(out.trades.len(), 7, "{out:#?}");
    assert!(out.trades.iter().all(|t| t.side == Side::Buy));
    let total: f64 = out.trades.iter().map(|t| t.notional + t.est_fee).sum();
    assert!(total <= 4750.0, "buys {total} must fit cash 5000 minus the 250 reserve");
    assert!(total > 4740.0, "and use nearly all of it, got {total}");
    assert!(trade(&out, "SPY").quantity < 1.0);
    assert!(trade(&out, "ETH/USD").quantity < 0.41666666);
    assert_eq!(out.capital_base, 5000.0);
}

#[test]
fn planner_when_cash_is_ample_buys_are_the_exact_floor_of_target_over_price() {
    let mut c = Case::planner(both_sleeves());
    c.equity = 50000.0;
    let out = c.with_cash(50000.0).ok();
    assert_eq!(trade(&out, "SPY").quantity, 1.0);
    assert_eq!(trade(&out, "EFA").quantity, 6.25);
    assert_eq!(trade(&out, "IEF").quantity, 5.263157894);
    assert_eq!(trade(&out, "DBC").quantity, 20.0);
    assert_eq!(trade(&out, "VNQ").quantity, 5.555555555);
    assert_eq!(trade(&out, "BTC/USD").quantity, 0.02083333);
    assert_eq!(trade(&out, "ETH/USD").quantity, 0.41666666);
    for t in &out.trades {
        assert!(t.notional <= 1250.0, "{t:?}");
    }
    assert_eq!(trade(&out, "SPY").est_fee, 1.25);
}

#[test]
fn planner_sells_come_first_and_full_exits_sell_exactly_the_held_quantity() {
    // Holds SPY 4 (2000) and VNQ 3 (270); the ETF sleeve is 100% of capital and wants VNQ at zero.
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.2, 0.2, 0.2, 0.0])]).held(SPY, 4.0).held(VNQ, 3.0).with_cash(2730.0);
    c.equity = 5000.0;
    let out = c.ok();
    let sides: Vec<Side> = out.trades.iter().map(|t| t.side).collect();
    assert_eq!(sides, vec![Side::Sell, Side::Sell, Side::Buy, Side::Buy, Side::Buy], "{out:#?}");
    let symbols: Vec<&str> = out.trades.iter().map(|t| t.symbol.as_str()).collect();
    assert_eq!(symbols, vec!["SPY", "VNQ", "DBC", "EFA", "IEF"]);
    assert_eq!(trade(&out, "SPY").quantity, 2.0); // 2000 -> 1000
    assert_eq!(trade(&out, "VNQ").quantity, 3.0); // full exit sells exactly the held quantity
}

#[test]
fn planner_a_position_already_at_target_produces_no_order() {
    let c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]).held(SPY, 2.0).with_cash(4000.0);
    let out = c.ok();
    assert!(out.trades.is_empty() && out.skipped.is_empty(), "{out:#?}");
    let spy = line(&out, "SPY");
    assert_eq!(spy.target_notional, 1000.0);
    assert_eq!(spy.current_notional, 1000.0);
}

// ---------------------------------------------------------------------------------------------------------------
// Minimum trade thresholds (planner.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn planner_min_trade_abs_boundary() {
    // Target SPY 1000; hold 1.98 shares (990): delta exactly 10.
    let sleeves = vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])];
    let mut c = Case::planner(sleeves.clone()).held(SPY, 1.98).with_cash(4010.0);
    c.filter = TradeFilter::new(10.0, 0.0);
    assert_eq!(trade(&c.ok(), "SPY").quantity, 0.02, "a delta exactly at the minimum trades");
    let mut c = Case::planner(sleeves).held(SPY, 1.98).with_cash(4010.0);
    c.filter = TradeFilter::new(10.01, 0.0);
    let out = c.ok();
    assert!(out.trades.is_empty());
    assert!(matches!(skip_reason(&out, "SPY"), SkipReason::BelowMinAbs { .. }));
}

#[test]
fn planner_min_trade_pct_boundary_is_a_fraction_of_the_target() {
    // Target SPY 1000, threshold 2% = 20. Hold 980 -> delta 20 trades; 980.01 -> 19.99 does not.
    let sleeves = vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])];
    let mut c = Case::planner(sleeves.clone()).held(SPY, 1.96).with_cash(4020.0);
    c.filter = TradeFilter::new(0.0, 0.02);
    assert_eq!(trade(&c.ok(), "SPY").quantity, 0.04);
    let mut c = Case::planner(sleeves).held(SPY, 1.96002).with_cash(4020.0);
    c.filter = TradeFilter::new(0.0, 0.02);
    let out = c.ok();
    assert!(out.trades.is_empty());
    assert!(matches!(skip_reason(&out, "SPY"), SkipReason::BelowMinPct { .. }));
}

#[test]
fn planner_a_full_exit_uses_the_current_value_as_the_percentage_reference() {
    // Target 0 but held 1000: the whole position sells.
    let sleeves = vec![etf(1.0, [0.0; 5])];
    let out = Case::planner(sleeves.clone()).held(SPY, 2.0).with_cash(4000.0).ok();
    assert_eq!(trade(&out, "SPY").quantity, 2.0);
    // Dust below the absolute minimum stays (no order).
    let out = Case::planner(sleeves).held(SPY, 0.01).with_cash(4995.0).ok();
    assert!(out.trades.is_empty());
    assert!(matches!(skip_reason(&out, "SPY"), SkipReason::BelowMinAbs { .. }));
}

// ---------------------------------------------------------------------------------------------------------------
// Venue rules (planner.rs, oanda_rules.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn planner_whole_share_assets_round_down_and_tiny_targets_are_refused() {
    let whole = LotRounder::new()
        .with("SPY", LotRule::new(0).with_min_notional(1.0))
        .with("EFA", LotRule::new(9))
        .with("IEF", LotRule::new(9))
        .with("DBC", LotRule::new(9))
        .with("VNQ", LotRule::new(9));
    let mut c = Case::planner(vec![etf(0.5, [0.5, 0.0, 0.0, 0.0, 0.0])]);
    c.equity = 50000.0;
    c.rounder = Some(whole.clone());
    let out = c.with_cash(50000.0).ok();
    assert_eq!(trade(&out, "SPY").quantity, 2.0, "2.5 shares round DOWN to 2");
    // 5000 * 0.001 * 0.5 = 2.5 USD -> 0.005 shares -> rounds to zero.
    let mut c = Case::planner(vec![etf(0.001, [0.5, 0.0, 0.0, 0.0, 0.0])]);
    c.equity = 50000.0;
    c.rounder = Some(whole);
    c.filter = TradeFilter::new(0.0, 0.02);
    let out = c.with_cash(50000.0).ok();
    assert!(out.trades.is_empty());
    assert_eq!(skip_reason(&out, "SPY"), &SkipReason::VenueRefused(SizeRefusal::RoundsToZero));
    // Missing rule row: the instrument is refused, never guessed.
    let mut c = Case::planner(vec![etf(0.5, [0.5, 0.0, 0.0, 0.0, 0.0])]);
    c.equity = 50000.0;
    c.rounder = Some(LotRounder::new().with("IEF", LotRule::new(9)));
    let out = c.with_cash(50000.0).ok();
    assert!(matches!(skip_reason(&out, "SPY"), SkipReason::VenueRefused(SizeRefusal::UnknownInstrument(_))));
}

#[test]
fn planner_kraken_minimum_volume_is_respected_not_bumped_up() {
    // 5000 * 0.001 * 0.5 = 2.5 USD of BTC = 0.00004166 BTC, below the 0.0001 minimum.
    let mut c = Case::planner(vec![crypto(0.001, 0.5, 0.0)]);
    c.filter = TradeFilter::new(0.0, 0.02);
    let out = c.ok();
    assert!(out.trades.is_empty(), "{out:#?}");
    match skip_reason(&out, "BTC/USD") {
        SkipReason::VenueRefused(SizeRefusal::BelowMinQuantity { min }) => assert_eq!(*min, 0.0001),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn planner_a_venue_that_rounds_up_is_a_hard_error() {
    struct RoundsUp;
    impl QuantityRounder for RoundsUp {
        fn round_quantity(&self, _: &str, _: Side, q: f64, _: f64) -> Result<f64, SizeRefusal> {
            Ok(q + 1.0)
        }
    }
    let r = RoundsUp;
    let c = Case::planner(both_sleeves());
    let err = construct(&ConstructInputs {
        equity: c.equity,
        allocated_capital: c.allocated,
        sleeves: &c.sleeves,
        risk_scale: c.rs,
        limits: &c.limits,
        instruments: &c.instruments,
        margin: &NoMargin,
        trade_filter: c.filter,
        rounding: Some(&r),
        funding: c.funding,
        target_dp: c.target_dp,
        unmanaged_gross: 0.0,
    })
    .unwrap_err();
    assert!(matches!(err, ConstructRefusal::VenueRoundedUp { .. }), "{err:?}");
}

#[test]
fn planner_oanda_rule_table_rounds_down_and_refuses_with_reasons() {
    let t = LotRounder::new()
        .with("EUR_USD", LotRule::new(0).with_min_quantity(1.0).with_max_order_units(1_000_000.0))
        .with("DE30_EUR", LotRule::new(1).with_min_quantity(0.1).with_max_order_units(2500.0));
    let px = 1.10;
    assert_eq!(t.round_quantity("EUR_USD", Side::Buy, 1234.99, px), Ok(1234.0));
    assert_eq!(t.round_quantity("EUR_USD", Side::Sell, 1234.99, px), Ok(1234.0), "a magnitude for both sides");
    assert_eq!(t.round_quantity("eur_usd", Side::Buy, 50.0, px), Ok(50.0));
    assert_eq!(t.round_quantity("DE30_EUR", Side::Buy, 2.57, 18000.0), Ok(2.5));
    assert_eq!(t.round_quantity("EUR_USD", Side::Buy, 0.4, px), Err(SizeRefusal::RoundsToZero));
    assert_eq!(t.round_quantity("DE30_EUR", Side::Buy, 0.05, px), Err(SizeRefusal::RoundsToZero));
    assert!(matches!(t.round_quantity("EUR_USD", Side::Buy, 2_000_000.0, px), Err(SizeRefusal::Other(m)) if m.contains("maximumOrderUnits")));
    assert_eq!(t.round_quantity("GBP_USD", Side::Buy, 1000.0, px), Err(SizeRefusal::UnknownInstrument("GBP_USD".into())));
    let min10 = LotRounder::new().with("EUR_USD", LotRule::new(0).with_min_quantity(10.0));
    assert_eq!(min10.round_quantity("EUR_USD", Side::Buy, 9.0, 1.0), Err(SizeRefusal::BelowMinQuantity { min: 10.0 }));
    assert_eq!(min10.round_quantity("EUR_USD", Side::Buy, 10.0, 1.0), Ok(10.0));
    // The reference price does not matter when there is no minimum notional.
    for px in [0.0001, 1.1, 100000.0] {
        assert_eq!(t.round_quantity("EUR_USD", Side::Buy, 500.0, px), Ok(500.0), "{px}");
    }
}

#[test]
fn planner_kraken_and_alpaca_shaped_rules() {
    let r = planner_rounder();
    assert_eq!(r.round_quantity("BTC/USD", Side::Buy, 0.123456789, 60000.0), Ok(0.12345678));
    assert_eq!(r.round_quantity("BTC/USD", Side::Buy, 0.00009, 60000.0), Err(SizeRefusal::BelowMinQuantity { min: 0.0001 }));
    assert!(matches!(r.round_quantity("DOGE/USD", Side::Buy, 1.0, 1.0), Err(SizeRefusal::UnknownInstrument(_))));
    // costmin 0.5: 0.0001 BTC at a price of 100 is worth 0.01.
    assert_eq!(r.round_quantity("BTC/USD", Side::Buy, 0.0001, 100.0), Err(SizeRefusal::BelowMinCost { min: 0.5 }));
    assert_eq!(r.round_quantity("SPY", Side::Buy, 1.2345678919, 500.0), Ok(1.234567891));
    assert_eq!(r.round_quantity("SPY", Side::Buy, 0.001, 500.0), Err(SizeRefusal::BelowMinCost { min: 1.0 }));
}

// ---------------------------------------------------------------------------------------------------------------
// risk_scale, sleeve shares, capital base (planner.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn planner_risk_scale_multiplies_every_target() {
    let sleeves = vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])];
    let mut c = Case::planner(sleeves.clone());
    c.equity = 50000.0;
    let full = c.with_cash(50000.0).ok();
    assert_eq!(trade(&full, "SPY").quantity, 2.0); // 1000
    let mut c = Case::planner(sleeves);
    c.equity = 50000.0;
    c.rs = RiskScale::new(0.5, 1.0);
    let half = c.with_cash(50000.0).ok();
    assert_eq!(trade(&half, "SPY").quantity, 1.0); // 500
    assert_eq!(half.risk_scale_applied, 0.5);
}

#[test]
fn planner_risk_scale_below_the_held_value_sells() {
    // Holding 2 SPY (1000) at scale 0.5 -> target 500: sell 1.
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]).held(SPY, 2.0).with_cash(4000.0);
    c.rs = RiskScale::new(0.5, 1.0);
    let out = c.ok();
    assert_eq!((out.trades[0].side, out.trades[0].quantity), (Side::Sell, 1.0));
}

#[test]
fn planner_risk_scale_must_be_in_the_open_closed_unit_interval() {
    for bad in [0.0, -0.1, 1.0001, 2.0] {
        let mut c = Case::planner(both_sleeves());
        c.rs = RiskScale::new(bad, 1.0);
        assert!(matches!(c.run(), Err(ConstructRefusal::Invalid(InputError::BadRiskScale { .. }))), "{bad}");
    }
    for good in [1.0, 0.000001] {
        let mut c = Case::planner(both_sleeves());
        c.rs = RiskScale::new(good, 1.0);
        assert!(c.run().is_ok(), "{good}");
    }
    // Deviation from the planner (ledger item 5): the ladder factor may be 0 (halted: flatten), but never above 1.
    let mut c = Case::planner(both_sleeves());
    c.rs = RiskScale::new(1.0, 1.0001);
    assert!(matches!(c.run(), Err(ConstructRefusal::Invalid(InputError::BadRiskScale { .. }))));
}

#[test]
fn planner_sleeve_input_validation() {
    let run = |sleeves: Vec<SleeveTargets>| Case::planner(sleeves).run().unwrap_err();
    let inv = |e: InputError| ConstructRefusal::Invalid(e);
    // shares above 1 in total
    assert!(matches!(run(vec![etf(0.6, [0.2; 5]), crypto(0.5, 0.5, 0.5)]), ConstructRefusal::Invalid(InputError::SharesExceedOne(_))));
    // a share of 0 or above 1
    assert!(matches!(run(vec![etf(0.0, [0.2; 5])]), ConstructRefusal::Invalid(InputError::BadShare(_))));
    assert!(matches!(run(vec![etf(1.1, [0.2; 5])]), ConstructRefusal::Invalid(InputError::BadShare(_))));
    // weights above 1 in total, or a single weight outside [0, 1]
    assert!(matches!(run(vec![etf(1.0, [0.3; 5])]), ConstructRefusal::Invalid(InputError::WeightsExceedOne(_))));
    assert!(matches!(run(vec![etf(1.0, [-0.1, 0.0, 0.0, 0.0, 0.0])]), ConstructRefusal::Invalid(InputError::BadWeight { .. })));
    // duplicate sleeve id, duplicate symbol within a sleeve
    let mut b = etf(0.5, [0.2; 5]);
    b.id = "etf".into();
    assert!(matches!(run(vec![etf(0.5, [0.2; 5]), b]), ConstructRefusal::Invalid(InputError::DuplicateSleeve(_))));
    let mut dup = etf(1.0, [0.1; 5]);
    dup.weights.push((SPY, 0.1));
    assert_eq!(run(vec![dup]), inv(InputError::DuplicateWeight { sleeve: "etf".into(), symbol: "SPY".into() }));
    // shares summing to exactly 1 are fine
    assert!(Case::planner(both_sleeves()).run().is_ok());
    // an index outside the instrument list
    let bad = SleeveTargets::long_only("x", 1.0, vec![(99, 0.1)]);
    assert!(matches!(run(vec![bad]), ConstructRefusal::Invalid(InputError::UnknownInstrument { .. })));
}

#[test]
fn planner_an_instrument_in_two_sleeves_gets_the_sum_of_share_times_weight() {
    let a = SleeveTargets::long_only("a", 0.5, vec![(SPY, 0.4)]);
    let b = SleeveTargets::long_only("b", 0.5, vec![(SPY, 0.2)]);
    // SPY: 0.5*0.4 + 0.5*0.2 = 0.3 of 5000 = 1500 = 3 shares. Listed in the opposite order on purpose.
    let mut c = Case::planner(vec![b, a]);
    c.equity = 50000.0;
    let out = c.with_cash(50000.0).ok();
    assert_eq!(out.trades.len(), 1);
    assert_eq!(out.trades[0].quantity, 3.0);
    assert_eq!(line(&out, "SPY").sleeves, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn planner_capital_base_is_the_smaller_of_broker_equity_and_the_allocation() {
    let sleeves = vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])];
    // Equity above the allocation: sized on 5000 (allocation), not 20000.
    let mut c = Case::planner(sleeves.clone());
    c.equity = 20000.0;
    let out = c.with_cash(20000.0).ok();
    assert_eq!(out.capital_base, 5000.0);
    assert_eq!(trade(&out, "SPY").quantity, 2.0);
    // Equity below the allocation (after a loss): sized on equity.
    let mut c = Case::planner(sleeves.clone());
    c.equity = 2500.0;
    let out = c.with_cash(2500.0).ok();
    assert_eq!(out.capital_base, 2500.0);
    assert_eq!(trade(&out, "SPY").quantity, 1.0);
    // Non-positive equity is refused.
    let mut c = Case::planner(sleeves);
    c.equity = 0.0;
    assert!(matches!(c.run(), Err(ConstructRefusal::Invalid(InputError::EquityInvalid(_)))));
}

// ---------------------------------------------------------------------------------------------------------------
// Cash handling (planner.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn planner_no_cash_above_the_reserve_means_no_buys() {
    // 4800 of unmanaged QQQ, 200 cash: below the 250 reserve.
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]).with_cash(200.0);
    c.unmanaged = 4800.0;
    let out = c.ok();
    assert!(out.trades.is_empty());
    assert_eq!(skip_reason(&out, "SPY"), &SkipReason::NoCashAvailable);
    assert!(out.lines.iter().all(|l| l.symbol != "QQQ"));
}

#[test]
fn planner_sell_proceeds_can_fund_buys_only_when_credited() {
    // Hold 4 SPY (2000) and 2750 cash; sleeve wants EFA 950 (11.875 sh) and SPY 0 -> sells 4 SPY.
    let sleeves = vec![etf(1.0, [0.0, 0.2, 0.0, 0.0, 0.0])];
    let mk = || {
        let mut c = Case::planner(sleeves.clone()).held(SPY, 4.0);
        c.equity = 4750.0;
        c
    };
    assert_eq!(mk().with_cash(2750.0).ok().trades.len(), 2);
    assert_eq!(mk().with_cash(2750.0).with_credit(false).ok().trades.len(), 2, "cash 2750 alone already covers 950");
    // Make the starting cash too small to cover the buy without the sale proceeds (QQQ is unmanaged: 2150).
    let tight = |credit: bool| {
        let mut c = mk().with_cash(600.0).with_credit(credit);
        c.unmanaged = 2150.0;
        c
    };
    let with = tight(true).ok();
    assert_eq!(trade(&with, "EFA").quantity, 11.875, "sale proceeds fund the full 950 buy: {with:#?}");
    let without = tight(false).ok();
    let spent: f64 = without.trades.iter().filter(|t| t.side == Side::Buy).map(|t| t.notional + t.est_fee).sum();
    assert!(spent <= 362.5, "600 cash - 237.5 reserve: {spent}");
    assert!(spent > 300.0, "but it still buys what fits: {spent}");
}

#[test]
fn planner_the_cash_reserve_is_a_fraction_of_the_capital_base_and_the_cash_is_the_actual_cash() {
    // Equity 20000, cash 300, allocation 5000: reserve = 5% of 5000 = 250, so 50 is available.
    let mk = |cash: f64| {
        let mut c = Case::planner(vec![crypto(1.0, 0.0, 0.5)]).with_cash(cash); // ETH target = 0.5 * 5000 = 2500
        c.equity = 20000.0;
        c
    };
    let out = mk(300.0).ok();
    assert_eq!(out.capital_base, 5000.0);
    let eth = trade(&out, "ETH/USD");
    let spent = eth.notional + eth.est_fee;
    assert!(spent <= 50.0, "only 300 - 250 = 50 is available, planned {spent}");
    assert!(spent > 49.0, "and the plan uses it: {spent}");
    // Actual cash still binds when the reserve is met: cash 250 exactly leaves nothing to spend.
    let out = mk(250.0).ok();
    assert!(out.trades.is_empty());
    assert_eq!(skip_reason(&out, "ETH/USD"), &SkipReason::NoCashAvailable);
}

#[test]
fn planner_the_plan_is_stable_under_input_ordering() {
    // Holds SPY 4 and VNQ 3. Reverse the sleeves, the weights, and the instrument list (indices remapped).
    let base = Case::planner(both_sleeves()).held(SPY, 4.0).held(VNQ, 3.0).with_cash(2730.0);
    let a = base.ok();
    let n = base.instruments.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.reverse(); // new index p -> old index perm[p]; old index o -> new index (n-1-o)
    let remap = |o: usize| n - 1 - o;
    let mut sleeves = both_sleeves();
    sleeves.reverse();
    for s in &mut sleeves {
        s.weights = s.weights.iter().rev().map(|&(j, w)| (remap(j), w)).collect();
    }
    let mut c = Case::new(perm.iter().map(|&o| base.instruments[o].clone()).collect(), sleeves).with_cash(2730.0);
    c.equity = base.equity;
    let b = c.ok();
    assert_eq!(brief(&a), brief(&b));
    for l in &a.lines {
        assert_eq!(l.target_notional, line(&b, &l.symbol).target_notional, "{}", l.symbol);
    }
    assert_eq!((a.gross, a.net, a.funding_left), (b.gross, b.net, b.funding_left));
}

// ---------------------------------------------------------------------------------------------------------------
// Signed sleeves (signed.rs). Mandate there: allocated 20000, shorting ON, leverage 3x, max_gross 3, max_net 3,
// max_position 3, 5% reserve, fee 0.25%. Prices SPY 500, EFA 80, IEF 95, DBC 25, VNQ 90.
// ---------------------------------------------------------------------------------------------------------------

fn signed_case(weights: &[(usize, f64)]) -> Case {
    let mut c = Case::planner(vec![SleeveTargets::signed("ls", 1.0, 2.0, weights.to_vec())]);
    c.equity = 20000.0;
    c.allocated = Some(20000.0);
    c.limits = Limits::unlimited().with_max_gross(3.0).with_leverage_max_gross(3.0).with_max_net(3.0).with_max_position(3.0);
    c.funding = Funding::BuyingPower { buying_power: 100000.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
    c
}

#[test]
fn signed_long_and_short_targets_are_planned_and_buying_power_left_matches() {
    // SPY +0.1 -> +2000 (4 shares), EFA -0.1 -> -2000 (25 shares).
    let out = signed_case(&[(SPY, 0.1), (EFA, -0.1)]).ok();
    assert_eq!(brief(&out), vec![("EFA".to_string(), Side::Sell, 25.0), ("SPY".to_string(), Side::Buy, 4.0)], "increases are ordered by (venue, symbol)");
    assert_eq!(line(&out, "EFA").target_notional, -2000.0);
    assert_eq!(line(&out, "SPY").target_notional, 2000.0);
    assert!(out.needs_margin, "a short target needs margin");
    assert_eq!(out.gross, 4000.0);
    // buying power: 100000 - (2000 + 5) - (2000 + 5)
    assert_eq!(out.funding_left, Some(95990.0));
}

#[test]
fn signed_gross_above_one_times_equity_is_planned_up_to_the_cap_and_refused_above_it() {
    // +-0.9 of 20000 = 18000 each: gross 36000 = 1.8x equity.
    let out = signed_case(&[(SPY, 0.9), (EFA, -0.9)]).ok();
    assert_eq!(trade(&out, "SPY").quantity, 36.0);
    assert_eq!(trade(&out, "EFA").quantity, 225.0);
    assert_eq!(out.projected_gross, 36000.0);
    // Exactly at the cap (3x = 60000): allowed.
    let mut at_cap = signed_case(&[(SPY, 1.0), (EFA, -1.0), (IEF, 1.0)]);
    at_cap.funding = Funding::BuyingPower { buying_power: 200000.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
    assert_eq!(at_cap.ok().gross, 60000.0);
    // One notch over: the WHOLE book is refused (no partial book).
    let mut over = signed_case(&[(SPY, 1.0), (EFA, -1.0), (IEF, 1.01)]);
    over.funding = Funding::BuyingPower { buying_power: 200000.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
    match over.run().unwrap_err() {
        ConstructRefusal::GrossAboveCap { gross, cap } => {
            assert_close(gross, 60200.0, 1e-8, "gross");
            assert_eq!(cap, 60000.0);
        }
        e => panic!("{e:?}"),
    }
}

#[test]
fn signed_a_no_leverage_mandate_refuses_a_levered_plan() {
    let mut c = signed_case(&[(SPY, 0.7), (DBC, 0.7)]);
    c.limits = Limits::unlimited().with_max_gross(1.0).with_leverage_max_gross(1.0).with_max_net(1.0).with_max_position(1.0);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { .. })));
}

#[test]
fn signed_max_abs_weight_bounds_a_signed_sleeve_and_replaces_the_unit_rules_only_for_it() {
    let run = |sleeves: Vec<SleeveTargets>| {
        let mut c = Case::planner(sleeves);
        c.equity = 20000.0;
        c.allocated = Some(20000.0);
        c.funding = Funding::BuyingPower { buying_power: 100000.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
        c.run()
    };
    let ls = |max: f64, w: &[(usize, f64)]| vec![SleeveTargets::signed("ls", 1.0, max, w.to_vec())];
    assert!(run(ls(2.0, &[(SPY, 2.0)])).is_ok());
    assert!(run(ls(2.0, &[(SPY, -2.0)])).is_ok());
    for bad in [2.000000001, -2.000000001, 3.0, -5.0] {
        let e = run(ls(2.0, &[(SPY, bad)])).unwrap_err();
        assert_eq!(e, ConstructRefusal::Invalid(InputError::BadSignedWeight { sleeve: "ls".into(), symbol: "SPY".into(), max: 2.0 }), "{bad}");
    }
    // No sum rule for a signed sleeve.
    assert!(run(ls(2.0, &[(SPY, 1.5), (EFA, -1.5)])).is_ok());
    // The bound itself: (0, MAX_ABS_WEIGHT_CAP].
    assert_eq!(MAX_ABS_WEIGHT_CAP, 3.0);
    for bad in [0.0, -1.0, 3.000000001, 4.0] {
        assert!(matches!(run(ls(bad, &[(SPY, 0.1)])), Err(ConstructRefusal::Invalid(InputError::BadMaxAbsWeight { .. }))), "{bad}");
    }
    assert!(run(ls(3.0, &[(SPY, 0.0)])).is_ok(), "the hard cap itself is allowed");
    // Per sleeve: a long-only sleeve next to a signed one keeps today's rules (share 0.5 each).
    let mixed = |w: &[(usize, f64)]| {
        vec![
            SleeveTargets::signed("ls", 0.5, 2.0, vec![(SPY, 1.5)]),
            SleeveTargets::long_only("etf", 0.5, w.to_vec()),
        ]
    };
    assert!(matches!(run(mixed(&[(EFA, -0.1)])), Err(ConstructRefusal::Invalid(InputError::BadWeight { .. }))));
    assert!(matches!(run(mixed(&[(EFA, 1.1)])), Err(ConstructRefusal::Invalid(InputError::BadWeight { .. }))));
    assert!(matches!(run(mixed(&[(EFA, 0.6), (IEF, 0.6)])), Err(ConstructRefusal::Invalid(InputError::WeightsExceedOne(_)))));
}

#[test]
fn signed_a_short_within_the_gross_cap_is_not_leverage_even_at_one_x() {
    let mut c = signed_case(&[(SPY, 0.1), (EFA, -0.1)]);
    c.limits = Limits::unlimited().with_max_gross(1.0).with_leverage_max_gross(1.0).with_max_net(1.0).with_max_position(1.0);
    let out = c.ok();
    assert_eq!(brief(&out), vec![("EFA".to_string(), Side::Sell, 25.0), ("SPY".to_string(), Side::Buy, 4.0)]);
}

#[test]
fn signed_a_margin_plan_without_buying_power_is_refused() {
    let cash_plan = |w: &[(usize, f64)]| {
        let mut c = signed_case(w);
        c.funding = Funding::Cash { cash: 20000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
        c.run()
    };
    // A short target.
    match cash_plan(&[(SPY, 0.1), (EFA, -0.1)]) {
        Err(ConstructRefusal::BuyingPowerRequired { .. }) => {}
        other => panic!("{other:?}"),
    }
    // A levered long-only book (gross 1.8x equity).
    assert_eq!(cash_plan(&[(SPY, 0.9), (EFA, 0.9)]), Err(ConstructRefusal::BuyingPowerRequired { gross: 36000.0 }));
    // Not a margin plan: long only, gross within equity -> plain cash rules.
    let out = cash_plan(&[(SPY, 0.5), (EFA, 0.3)]).unwrap();
    assert!(!out.needs_margin);
    assert_eq!(out.trades.len(), 2);
    // A negative figure is a caller bug.
    let mut c = signed_case(&[(SPY, 0.1)]);
    c.funding = Funding::BuyingPower { buying_power: -1.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
    assert!(matches!(c.run(), Err(ConstructRefusal::Invalid(InputError::BadParam(_)))));
}

#[test]
fn signed_buying_power_limits_increases_and_scales_them_by_one_common_factor() {
    // Wants 2005 + 2005 of increases; buying power 3000 less the 1000 reserve leaves 2000.
    let mk = |bp: f64| {
        let mut c = signed_case(&[(SPY, 0.1), (EFA, -0.1)]);
        c.funding = Funding::BuyingPower { buying_power: bp, reserve_fraction: 0.05, fee_rate: 0.0025 };
        c
    };
    let out = mk(3000.0).ok();
    let spent: f64 = out.trades.iter().map(|t| t.notional + t.est_fee).sum();
    assert!(spent <= 2000.0, "increases plus fees {spent} must fit buying power minus reserve");
    assert!(spent > 1999.0, "and use nearly all of it, got {spent}");
    let (efa, spy) = (trade(&out, "EFA"), trade(&out, "SPY"));
    assert!(efa.quantity < 25.0 && spy.quantity < 4.0);
    assert!((efa.notional / 2000.0 - spy.notional / 2000.0).abs() < 0.001, "one common factor");
    assert!(out.funding_left.unwrap() >= 1000.0, "the reserve stays uncommitted");
    // No room at all (buying power below the reserve): every increase is skipped, none is invented.
    let none = mk(900.0).ok();
    assert!(none.trades.is_empty());
    assert!(none.skipped.iter().all(|s| s.reason == SkipReason::NoCashAvailable), "{:?}", none.skipped);
}

#[test]
fn signed_with_buying_power_cash_is_not_the_budget() {
    // The broker reports 100000 of buying power (a margin book) whatever the cash: sized on buying power.
    let out = signed_case(&[(SPY, 0.9), (EFA, -0.9)]).ok();
    assert_eq!(out.trades.len(), 2);
    assert_eq!(trade(&out, "SPY").quantity, 36.0);
}

#[test]
fn signed_long_to_short_is_a_close_leg_then_an_open_leg() {
    // Holds SPY 4 (2000 long); the target is -2000 (-0.1).
    let out = signed_case(&[(SPY, -0.1)]).held(SPY, 4.0).ok();
    assert_eq!(brief(&out), vec![("SPY".to_string(), Side::Sell, 4.0), ("SPY".to_string(), Side::Sell, 4.0)], "sell to close, then sell to open");
    assert_eq!((out.trades[0].leg, out.trades[0].reducing), (LegKind::Close, true));
    assert_eq!((out.trades[1].leg, out.trades[1].reducing), (LegKind::Open, false));
}

#[test]
fn signed_short_to_long_is_a_cover_then_a_buy_and_needs_no_margin() {
    // Holds EFA -25 (-2000 short); the target is +2000. Cash funding: not a margin plan.
    let mut c = signed_case(&[(EFA, 0.1)]).held(EFA, -25.0);
    c.equity = 22000.0;
    c.funding = Funding::Cash { cash: 22000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
    let out = c.ok();
    assert_eq!(brief(&out), vec![("EFA".to_string(), Side::Buy, 25.0), ("EFA".to_string(), Side::Buy, 25.0)]);
    assert!(!out.needs_margin);
}

#[test]
fn signed_a_venue_refusal_of_the_open_leg_leaves_the_account_flat() {
    // SPY trades in whole shares; the short target is 0.8 of a share, which rounds to zero: close only.
    let mut c = signed_case(&[(SPY, -0.02)]).held(SPY, 4.0);
    c.rounder = Some(planner_rounder().with("SPY", LotRule::new(0).with_min_notional(1.0)));
    c.filter = TradeFilter::new(0.0, 0.02);
    let out = c.ok();
    assert_eq!(brief(&out), vec![("SPY".to_string(), Side::Sell, 4.0)]);
    assert!(out.skipped.iter().any(|s| s.symbol == "SPY" && s.reason == SkipReason::VenueRefused(SizeRefusal::RoundsToZero)));
}

#[test]
fn signed_a_held_short_is_managed_in_a_signed_sleeve_and_still_skipped_in_a_long_only_one() {
    // Signed sleeve owns EFA: already at its -2000 target -> nothing to do, NOT skipped as a held short. The long-only
    // sleeve's SPY short is untouched, exactly as before.
    let mut c = Case::planner(vec![
        SleeveTargets::signed("ls", 0.5, 2.0, vec![(EFA, -0.2)]),
        SleeveTargets::long_only("etf", 0.5, vec![(SPY, 0.1)]),
    ])
    .held(EFA, -25.0)
    .held(SPY, -1.0);
    c.equity = 22500.0;
    c.allocated = Some(20000.0);
    c.funding = Funding::BuyingPower { buying_power: 100000.0, reserve_fraction: 0.05, fee_rate: 0.0025 };
    let out = c.ok();
    assert!(out.trades.is_empty(), "{out:#?}");
    assert_eq!(line(&out, "EFA").held_units, -25.0);
    assert!(!out.skipped.iter().any(|s| s.symbol == "EFA"));
    assert!(out.skipped.iter().any(|s| s.symbol == "SPY" && s.reason == SkipReason::ShortPositionHeld));
    // A smaller short target buys back part of it (a reduction): 12.5 of 25 shares.
    let mut c = signed_case(&[(EFA, -0.05)]).held(EFA, -25.0);
    c.equity = 22000.0;
    let out = c.ok();
    assert_eq!(brief(&out), vec![("EFA".to_string(), Side::Buy, 12.5)]);
    // A zero target covers all of it, and needs no buying power.
    let mut c = signed_case(&[(EFA, 0.0)]).held(EFA, -25.0);
    c.equity = 22000.0;
    c.funding = Funding::Cash { cash: 22000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
    assert_eq!(brief(&c.ok()), vec![("EFA".to_string(), Side::Buy, 25.0)]);
    // The same account in a plan with NO signed sleeve: refused, as it always was.
    let mut c = Case::planner(vec![SleeveTargets::long_only("etf", 1.0, vec![(EFA, 0.1)])]).held(EFA, -25.0);
    c.equity = 22000.0;
    c.allocated = Some(20000.0);
    c.funding = Funding::Cash { cash: 22000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
    let out = c.ok();
    assert!(out.trades.is_empty());
    assert_eq!(out.skipped[0].reason, SkipReason::ShortPositionHeld);
}

#[test]
fn signed_zero_targets_close_everything_without_needing_margin_or_buying_power() {
    let mk = || {
        let mut c = signed_case(&[(SPY, 0.0), (EFA, 0.0), (IEF, 0.0)]);
        c.funding = Funding::Cash { cash: 20000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
        c
    };
    let out = mk().held(SPY, 4.0).held(EFA, -25.0).ok();
    assert!(!out.needs_margin);
    assert_eq!(brief(&out), vec![("EFA".to_string(), Side::Buy, 25.0), ("SPY".to_string(), Side::Sell, 4.0)], "both are reductions: ordered by symbol");
    // Flat account, zero targets: nothing at all.
    let out = mk().ok();
    assert!(out.trades.is_empty() && out.skipped.is_empty());
}

#[test]
fn signed_risk_scale_multiplies_signed_targets_and_rounds_toward_zero() {
    let w = 0.1234567891234;
    let run = |scale: f64| {
        let mut c = signed_case(&[(SPY, w), (EFA, -w)]);
        c.rs = RiskScale::new(scale, 1.0);
        c.filter = TradeFilter::NONE;
        c.ok()
    };
    let full = run(1.0);
    // 20000 * 0.1234567891234 = 2469.135782468: the long floors to ...46, the short rounds toward zero to -...46 (a
    // plain floor would make it -...47, i.e. a slightly bigger short).
    assert_eq!(line(&full, "SPY").target_notional, 2469.13578246);
    assert_eq!(line(&full, "EFA").target_notional, -2469.13578246);
    let half = run(0.5);
    assert_eq!(line(&half, "SPY").target_notional, 1234.56789123);
    assert_eq!(line(&half, "EFA").target_notional, -1234.56789123);
    for t in &full.trades {
        assert!(t.notional <= 2469.13578246 + 1e-9, "{t:?}");
    }
    assert!(trade(&half, "EFA").quantity < trade(&full, "EFA").quantity);
    assert_eq!(trade(&full, "EFA").side, Side::Sell);
}

// ---------------------------------------------------------------------------------------------------------------
// Known deviations, recorded as tests so a change to either side is noticed (design 5.4 test 5).
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn deviation_shorting_off_refuses_the_whole_book_where_the_planner_denies_one_order() {
    // signed.rs `shorting_off_in_the_mandate_denies_the_short_with_the_existing_code`: the planner places the SPY buy and
    // denies the EFA short (SHORTING_FORBIDDEN). Council R1 (all-or-nothing) says the whole signed book must be refused;
    // `construct` implements R1, the planner does not yet. Under `PlannerFaithful` the guard's job is left to the guard.
    let mut c = signed_case(&[(SPY, 0.1), (EFA, -0.1)]);
    c.limits = c.limits.clone().with_shorting(false);
    match c.run() {
        Err(ConstructRefusal::ShortingForbidden { symbol, .. }) => assert_eq!(symbol, "EFA"),
        other => panic!("{other:?}"),
    }
    let mut c = signed_case(&[(SPY, 0.1), (EFA, -0.1)]);
    c.limits = c.limits.clone().with_shorting(false).with_policy(LimitPolicy::PlannerFaithful);
    assert_eq!(c.ok().trades.len(), 2, "planner-faithful mode does not decide shorting: the guard does");
}

#[test]
fn deviation_long_only_gross_cap_is_checked_at_plan_level_only_under_r1() {
    // planner.rs: a long-only plan never gets `GrossAboveCap` (its weights and shares are bounded instead, and the guard
    // denies single orders). `construct` refuses the whole book under R1/R2 and mirrors the planner under PlannerFaithful.
    let mk = |policy: LimitPolicy| {
        let mut c = Case::planner(vec![etf(1.0, [0.2; 5])]);
        c.limits = Limits::long_only_unit().with_max_gross(0.5).with_policy(policy);
        c.funding = Funding::Cash { cash: 5000.0, reserve_fraction: 0.05, fee_rate: 0.0025, credit_sell_proceeds: true };
        c
    };
    assert!(matches!(mk(LimitPolicy::RefuseWholeBook).run(), Err(ConstructRefusal::GrossAboveCap { .. })));
    assert!(mk(LimitPolicy::PlannerFaithful).run().is_ok());
    // A signed plan is refused by both.
    let mut c = signed_case(&[(SPY, 1.0), (EFA, 1.0)]);
    c.limits = c.limits.clone().with_max_gross(1.0).with_leverage_max_gross(1.0).with_policy(LimitPolicy::PlannerFaithful);
    assert!(matches!(c.run(), Err(ConstructRefusal::GrossAboveCap { .. })));
}
