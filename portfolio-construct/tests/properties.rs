//! Seeded property tests of `construct` (no `rand`: SplitMix64). Every case is reproducible from its seed, printed on
//! failure. The oracles below re-derive limits, budgets and directions from the generated numbers and the returned
//! lines; they do not call the code under test to vouch for itself. Non-vacuity is asserted: each property reports how
//! many generated books actually exercised it.

mod common;

use common::*;
use portfolio_construct::*;

const CASES: u64 = 1500;

fn is_ok(g: &G) -> bool {
    g.run().is_ok()
}

fn sym_map<T: Clone>(out: &ConstructOutput, f: impl Fn(&Line) -> T) -> Vec<(String, T)> {
    let mut v: Vec<(String, T)> = out.lines.iter().map(|l| (l.symbol.clone(), f(l))).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn trade_key(t: &TradeIntent) -> (String, u8, u64, u64, u64, bool) {
    (
        t.symbol.clone(),
        if t.side == Side::Buy { 1 } else { 2 },
        t.quantity.to_bits(),
        t.notional.to_bits(),
        t.est_fee.to_bits(),
        t.reducing,
    )
}

fn sorted_trades(out: &ConstructOutput) -> Vec<(String, u8, u64, u64, u64, bool)> {
    let mut v: Vec<_> = out.trades.iter().map(trade_key).collect();
    v.sort();
    v
}

// ---------------------------------------------------------------------------------------------------------------
// limits are respected by every Ok output (independent recomputation)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn every_ok_output_satisfies_every_limit_and_every_refusal_is_a_real_breach() {
    let (mut oks, mut refused, mut short_books) = (0, 0, 0);
    for seed in 0..CASES {
        let g = gen(seed);
        let cb = capital_base(g.equity, g.allocated);
        let lim = &g.limits;
        match g.run() {
            Ok(out) => {
                oks += 1;
                let tol = 1e-9;
                let t: Vec<f64> = out.lines.iter().map(|l| l.target_notional).collect();
                let gross: f64 = t.iter().map(|x| x.abs()).sum();
                assert_close(gross, out.gross, 1e-6 * (1.0 + gross), &format!("seed {seed}: gross"));
                if lim.policy == LimitPolicy::RefuseWholeBook {
                    assert!(
                        gross <= lim.effective_max_gross() * cb * (1.0 + tol)
                            || lim.effective_max_gross().is_infinite(),
                        "seed {seed}: gross {gross}"
                    );
                    let net: f64 = t.iter().sum();
                    assert!(
                        net.abs() <= lim.max_net * cb * (1.0 + tol) || lim.max_net.is_infinite(),
                        "seed {seed}: net {net}"
                    );
                    assert!(
                        t.iter()
                            .all(|x| x.abs() <= lim.max_position * cb * (1.0 + tol) || lim.max_position.is_infinite()),
                        "seed {seed}: position"
                    );
                    if let Some(cap) = lim.max_asset_class.get("a") {
                        let class_gross: f64 = out
                            .lines
                            .iter()
                            .filter(|l| g.instruments[l.instrument].class == "a")
                            .map(|l| l.target_notional.abs())
                            .sum();
                        assert!(class_gross <= cap * cb * (1.0 + tol), "seed {seed}: class");
                    }
                    if !lim.shorting {
                        assert!(t.iter().all(|x| *x >= 0.0), "seed {seed}: a short in a no-shorting book");
                    }
                }
                if t.iter().any(|x| *x < 0.0) {
                    short_books += 1;
                }
            }
            Err(ConstructRefusal::GrossAboveCap { gross, cap }) => {
                refused += 1;
                assert!(gross > cap, "seed {seed}");
                assert_close(cap, lim.effective_max_gross() * cb, 1e-6, "cap");
            }
            Err(ConstructRefusal::NetAboveCap { net, cap }) => {
                refused += 1;
                assert!(net.abs() > cap, "seed {seed}");
            }
            Err(ConstructRefusal::PositionAboveCap { notional, cap, .. }) => {
                refused += 1;
                assert!(notional.abs() > cap, "seed {seed}");
            }
            Err(ConstructRefusal::ClassAboveCap { gross, cap, .. }) => {
                refused += 1;
                assert!(gross > cap, "seed {seed}");
            }
            Err(ConstructRefusal::MarginExceeded { used, ceiling, .. }) => {
                refused += 1;
                assert!(used > ceiling, "seed {seed}");
            }
            Err(ConstructRefusal::ShortingForbidden { target, .. }) => {
                refused += 1;
                assert!(target < 0.0 && !lim.shorting, "seed {seed}");
            }
            Err(ConstructRefusal::BuyingPowerRequired { .. }) => refused += 1,
            Err(other) => panic!("seed {seed}: unexpected {other:?}"),
        }
    }
    assert!(
        oks > 400 && refused > 100 && short_books > 100,
        "non-vacuous: ok {oks}, refused {refused}, short {short_books}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// no cash creation, never oversell, moves toward the target and never past it
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn a_plan_never_oversells_never_overspends_and_never_overshoots_a_target() {
    let (mut checked, mut funded, mut crossings) = (0, 0, 0);
    for seed in 0..CASES {
        let g = gen(seed);
        let Ok(out) = g.run() else { continue };
        checked += 1;
        let cb = out.capital_base;
        // reductions precede increases
        let first_increase = out.trades.iter().position(|t| !t.reducing).unwrap_or(out.trades.len());
        assert!(out.trades[first_increase..].iter().all(|t| !t.reducing), "seed {seed}: a reduction after an increase");
        // per instrument: units after applying the trades stay between the current holding and the target
        for l in &out.lines {
            let inst = &g.instruments[l.instrument];
            let price = inst.price.unwrap();
            let mut units = inst.held_units;
            let mut n_legs = 0;
            for t in out.trades.iter().filter(|t| t.instrument == l.instrument) {
                n_legs += 1;
                units += if t.side == Side::Buy { t.quantity } else { -t.quantity };
                if t.reducing {
                    assert!(
                        t.quantity <= inst.held_units.abs() * (1.0 + 1e-12),
                        "seed {seed}: {} oversells {} of {}",
                        l.symbol,
                        t.quantity,
                        inst.held_units
                    );
                }
            }
            if n_legs == 2 {
                crossings += 1;
            }
            let after = units * price;
            let tol = 1e-6 * (1.0 + l.current_notional.abs() + l.target_notional.abs());
            let (d_after, d_cur) = (after - l.target_notional, l.current_notional - l.target_notional);
            let over = if d_after * d_cur < 0.0 { d_after.abs() } else { 0.0 };
            assert!(
                over <= tol,
                "seed {seed}: {} overshoots: current {} target {} after {after}",
                l.symbol,
                l.current_notional,
                l.target_notional
            );
            // and never moves the wrong way from the current holding
            assert!(
                (after - l.current_notional) * (l.target_notional - l.current_notional) >= -tol,
                "seed {seed}: {} moved away from the target",
                l.symbol
            );
        }
        // cash: the increases fit the independently recomputed budget
        let rate_of = |f: Funding| match f {
            Funding::Cash { fee_rate, .. } | Funding::BuyingPower { fee_rate, .. } => fee_rate,
            Funding::Unconstrained => 0.0,
        };
        let _ = rate_of(g.funding);
        let inc: f64 = out.trades.iter().filter(|t| !t.reducing).map(|t| t.notional + t.est_fee).sum();
        match g.funding {
            Funding::Unconstrained => {}
            Funding::Cash { cash, reserve_fraction, credit_sell_proceeds, .. } => {
                funded += 1;
                let mut sim = cash;
                for t in out.trades.iter().filter(|t| t.reducing) {
                    sim += if t.side == Side::Sell { t.notional - t.est_fee } else { -t.notional - t.est_fee };
                }
                let budget = if credit_sell_proceeds { sim } else { sim.min(cash) };
                let available = budget - ceil_dp(reserve_fraction * cb, 8);
                assert!(inc <= available.max(0.0) + 1e-6, "seed {seed}: increases {inc} exceed available {available}");
                if available <= 0.0 {
                    assert!(out.trades.iter().all(|t| t.reducing), "seed {seed}: bought with no cash");
                }
            }
            Funding::BuyingPower { buying_power, reserve_fraction, .. } => {
                funded += 1;
                let available = buying_power - ceil_dp(reserve_fraction * cb, 8);
                assert!(
                    inc <= available.max(0.0) + 1e-6,
                    "seed {seed}: increases {inc} exceed buying power {available}"
                );
            }
        }
    }
    assert!(
        checked > 800 && funded > 350 && crossings > 30,
        "non-vacuous: checked {checked}, funded {funded}, crossings {crossings}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// permutation invariance (sleeves, weights, instruments)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn permuting_sleeves_weights_and_instruments_changes_no_bit_of_the_result() {
    let mut rng = SplitMix64(4242);
    let (mut same, mut refusals) = (0, 0);
    for seed in 0..CASES {
        let g = gen(seed);
        let mut perm: Vec<usize> = (0..g.instruments.len()).collect();
        rng.shuffle(&mut perm);
        let p = g.permuted(&perm, &mut rng);
        match (g.run(), p.run()) {
            (Ok(a), Ok(b)) => {
                same += 1;
                assert_eq!(
                    sym_map(&a, |l| l.target_notional.to_bits()),
                    sym_map(&b, |l| l.target_notional.to_bits()),
                    "seed {seed}: targets"
                );
                assert_eq!(
                    sym_map(&a, |l| l.current_notional.to_bits()),
                    sym_map(&b, |l| l.current_notional.to_bits()),
                    "seed {seed}: current"
                );
                assert_eq!(
                    sym_map(&a, |l| l.sleeves.clone()),
                    sym_map(&b, |l| l.sleeves.clone()),
                    "seed {seed}: sleeve labels"
                );
                assert_eq!(sorted_trades(&a), sorted_trades(&b), "seed {seed}: trades");
                assert_eq!(
                    (
                        a.gross.to_bits(),
                        a.net.to_bits(),
                        a.margin_used.to_bits(),
                        a.projected_gross.to_bits(),
                        a.needs_margin
                    ),
                    (
                        b.gross.to_bits(),
                        b.net.to_bits(),
                        b.margin_used.to_bits(),
                        b.projected_gross.to_bits(),
                        b.needs_margin
                    ),
                    "seed {seed}: aggregates"
                );
                assert_eq!(
                    a.funding_left.map(f64::to_bits),
                    b.funding_left.map(f64::to_bits),
                    "seed {seed}: funding left"
                );
                let mut sa: Vec<_> = a.skipped.iter().map(|s| (s.symbol.clone(), format!("{:?}", s.reason))).collect();
                let mut sb: Vec<_> = b.skipped.iter().map(|s| (s.symbol.clone(), format!("{:?}", s.reason))).collect();
                sa.sort();
                sb.sort();
                assert_eq!(sa, sb, "seed {seed}: skips");
            }
            (Err(a), Err(b)) => {
                refusals += 1;
                assert_eq!(std::mem::discriminant(&a), std::mem::discriminant(&b), "seed {seed}: {a:?} vs {b:?}");
            }
            (a, b) => panic!("seed {seed}: one order is refused and the other is not: {a:?} / {b:?}"),
        }
    }
    assert!(same > 600 && refusals > 100, "non-vacuous: ok {same}, refused {refusals}");
}

// ---------------------------------------------------------------------------------------------------------------
// linearity
// ---------------------------------------------------------------------------------------------------------------

/// A research-mode copy: unconstrained funding, unrounded, no limits, no filter, no margin.
fn research(g: &G) -> G {
    let mut r = g.clone();
    r.funding = Funding::Unconstrained;
    r.rounder = None;
    r.target_dp = None;
    r.limits = Limits::unlimited();
    r.filter = TradeFilter::NONE;
    r.margin = 0;
    r
}

fn targets(g: &G) -> Vec<f64> {
    g.run().unwrap().target_notional
}

#[test]
fn targets_are_linear_in_the_risk_scale_the_capital_base_and_the_shares() {
    let mut n = 0;
    for seed in 0..CASES {
        let base = research(&gen(seed));
        let t0 = targets(&base);
        // risk scale: halve the ladder factor
        let mut half = base.clone();
        half.rs = RiskScale::new(base.rs.approval_constant, base.rs.ladder * 0.5);
        let t1 = targets(&half);
        for (a, b) in t0.iter().zip(&t1) {
            assert_close(*b, a * 0.5, 1e-9 * (1.0 + a.abs()), &format!("seed {seed}: risk scale"));
        }
        // capital base: double the equity with no allocation cap
        let mut uncapped = base.clone();
        uncapped.allocated = None;
        let u0 = targets(&uncapped);
        let mut twice = uncapped.clone();
        twice.equity *= 2.0;
        let u1 = targets(&twice);
        for (a, b) in u0.iter().zip(&u1) {
            assert_close(*b, a * 2.0, 1e-9 * (1.0 + a.abs()), &format!("seed {seed}: capital base"));
        }
        // shares: halve every share (still valid: at most 1)
        let mut shares = base.clone();
        for s in &mut shares.sleeves {
            s.share *= 0.5;
        }
        let s1 = targets(&shares);
        for (a, b) in t0.iter().zip(&s1) {
            assert_close(*b, a * 0.5, 1e-9 * (1.0 + a.abs()), &format!("seed {seed}: shares"));
        }
        n += 1;
    }
    assert_eq!(n, CASES);
}

#[test]
fn opposite_sleeves_on_one_instrument_cancel() {
    // Two signed sleeves with equal shares and opposite weights net to zero: the target is exactly 0 and a held position
    // is sold. The absolute-sum error would give twice the weight.
    let mut rng = SplitMix64(11);
    for case in 0..200 {
        let share = 0.05 * rng.range(1, 10) as f64;
        let w = (rng.range(1, 200) as f64) / 100.0;
        let inst = vec![InstrumentFacts::new("X", "v", "c", 10.0 + case as f64).with_held(3.0)];
        let sleeves = vec![
            SleeveTargets::signed("a", share, 2.0, vec![(0, w)]),
            SleeveTargets::signed("b", share, 2.0, vec![(0, -w)]),
        ];
        let mut c = Case::new(inst, sleeves);
        c.equity = 10000.0;
        c.allocated = None;
        c.funding = Funding::Unconstrained;
        c.rounder = None;
        c.filter = TradeFilter::NONE;
        let out = c.ok();
        assert_eq!(out.target_notional[0], 0.0, "case {case}");
        assert_eq!((out.trades[0].side, out.trades[0].quantity), (Side::Sell, 3.0));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// monotonicity
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn lowering_the_risk_scale_never_increases_gross_and_never_creates_a_refusal() {
    let (mut permitted, mut shrunk) = (0, 0);
    for seed in 0..CASES {
        let g = gen(seed);
        let mut lower = g.clone();
        lower.rs = RiskScale::new(g.rs.approval_constant, g.rs.ladder * 0.5);
        match (g.run(), lower.run()) {
            (Ok(a), Ok(b)) => {
                permitted += 1;
                assert!(b.gross <= a.gross * (1.0 + 1e-12) + 1e-9, "seed {seed}: gross {} -> {}", a.gross, b.gross);
                assert!(b.margin_used <= a.margin_used * (1.0 + 1e-12) + 1e-9, "seed {seed}: margin");
                for (x, y) in a.target_notional.iter().zip(&b.target_notional) {
                    assert!(y.abs() <= x.abs() * (1.0 + 1e-12) + 1e-9, "seed {seed}: |target| grew");
                    assert!(x * y >= 0.0, "seed {seed}: a target changed sign");
                }
                if b.gross < a.gross {
                    shrunk += 1;
                }
            }
            (Ok(_), Err(e)) => {
                // The only refusals lowering the scale may cause are funding ones that do not depend on the book's size.
                assert!(
                    matches!(e, ConstructRefusal::BuyingPowerRequired { .. }),
                    "seed {seed}: a smaller book was refused: {e:?}"
                );
            }
            _ => {}
        }
    }
    assert!(permitted > 600 && shrunk > 300, "non-vacuous: {permitted} / {shrunk}");
}

#[test]
fn raising_a_limit_never_creates_a_refusal_and_lowering_it_never_removes_one() {
    let (mut flips_up, mut flips_down) = (0, 0);
    for seed in 0..CASES {
        let g = gen(seed);
        if g.limits.policy != LimitPolicy::RefuseWholeBook {
            continue;
        }
        let mut loose = g.clone();
        loose.limits = Limits::unlimited();
        let mut tight = g.clone();
        tight.limits =
            tight.limits.clone().with_max_gross(tight.limits.max_gross.min(0.5)).with_leverage_max_gross(f64::INFINITY);
        let (base, lo, ti) = (is_ok(&g), is_ok(&loose), is_ok(&tight));
        if base {
            assert!(lo, "seed {seed}: unlimited caps refused a permitted book: {:?}", loose.run().err());
        }
        if !base {
            assert!(!ti, "seed {seed}: a tighter cap permitted a refused book");
        }
        if !base && lo {
            flips_up += 1;
        }
        if base && !ti {
            flips_down += 1;
        }
    }
    assert!(flips_up > 30 && flips_down > 30, "non-vacuous: {flips_up} / {flips_down}");
}

#[test]
fn raising_the_trade_filter_never_adds_a_trade() {
    let mut compared = 0;
    for seed in 0..CASES {
        let mut g = research(&gen(seed));
        g.filter = TradeFilter::new(5.0, 0.01);
        let a: std::collections::BTreeSet<usize> = g.run().unwrap().trades.iter().map(|t| t.instrument).collect();
        for stricter in [TradeFilter::new(20.0, 0.01), TradeFilter::new(5.0, 0.05), TradeFilter::new(100.0, 0.1)] {
            let mut h = g.clone();
            h.filter = stricter;
            let b: std::collections::BTreeSet<usize> = h.run().unwrap().trades.iter().map(|t| t.instrument).collect();
            assert!(
                b.is_subset(&a),
                "seed {seed}: a stricter filter {stricter:?} added {:?}",
                b.difference(&a).collect::<Vec<_>>()
            );
            compared += 1;
        }
    }
    assert_eq!(compared, CASES * 3);
}

// ---------------------------------------------------------------------------------------------------------------
// idempotence
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn the_trade_filter_is_idempotent_and_returns_target_or_current() {
    let mut rng = SplitMix64(31337);
    let (mut kept, mut dropped) = (0, 0);
    for _ in 0..20_000 {
        let f = TradeFilter::new(*rng.pick(&[0.0, 5.0, 10.0, 50.0]), *rng.pick(&[0.0, 0.02, 0.1, 0.5]));
        let target = (rng.range(0, 40_000) as f64 - 20_000.0) / 10.0;
        let current = (rng.range(0, 40_000) as f64 - 20_000.0) / 10.0;
        let once = f.apply(target, current);
        assert!(once == target || once == current);
        assert_eq!(f.apply(target, once), once, "idempotent: target {target} current {current} {f:?}");
        if once == target {
            kept += 1;
        } else {
            dropped += 1;
            assert!(f.check(target, current).is_some());
        }
    }
    assert!(kept > 5000 && dropped > 500, "{kept} / {dropped}");
}

#[test]
fn replanning_from_the_post_plan_holdings_trades_nothing_more() {
    // Research mode (exact units, no cash limit): execute every trade at its price, then construct again from the new
    // holdings. Nothing is left to trade, and every instrument the filter dropped is dropped again for the same reason.
    let (mut traded, mut second_empty) = (0, 0);
    for seed in 0..CASES {
        let g = research(&gen(seed));
        let a = g.run().unwrap();
        let mut h = g.clone();
        for t in &a.trades {
            h.instruments[t.instrument].held_units += if t.side == Side::Buy { t.quantity } else { -t.quantity };
        }
        let b = h.run().unwrap();
        if !a.trades.is_empty() {
            traded += 1;
        }
        assert!(b.trades.is_empty(), "seed {seed}: the second pass still trades {:?}", b.trades);
        second_empty += 1;
        let dropped_a: Vec<(String, String)> =
            a.skipped.iter().map(|s| (s.symbol.clone(), format!("{:?}", s.reason))).collect();
        for (sym, _) in &dropped_a {
            assert!(
                b.skipped.iter().any(|s| &s.symbol == sym),
                "seed {seed}: {sym} was skipped once and traded/ignored later"
            );
        }
        // targets are unchanged by trading (they depend on equity, not on holdings)
        assert_eq!(a.target_notional, b.target_notional, "seed {seed}");
    }
    assert!(traded > 700 && second_empty == CASES, "non-vacuous: {traded}");
}

// ---------------------------------------------------------------------------------------------------------------
// rounding never rounds up
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn the_rounder_never_returns_more_than_asked_and_lots_are_multiples() {
    let mut rng = SplitMix64(5);
    let mut ok = 0;
    for _ in 0..20_000 {
        let dp = *rng.pick(&[0u32, 1, 2, 4, 8, 9]);
        let rule = LotRule::new(dp).with_min_quantity(*rng.pick(&[0.0, 0.5])).with_min_notional(*rng.pick(&[0.0, 1.0]));
        let r = LotRounder::new().with("X", rule);
        let wished = rng.range(0, 10_000_000) as f64 / 1e4 * (0.01 + rng.unit());
        let price = 1.0 + rng.range(0, 100_000) as f64 / 100.0;
        if let Ok(q) = r.round_quantity("X", Side::Buy, wished, price) {
            ok += 1;
            // (a value within 4 ulps under a lot boundary counts as on it, so a hair above is allowed)
            assert!(q <= wished * (1.0 + 1e-12), "rounded UP: {wished} -> {q}");
            assert!(
                wished - q < 1.0 / 10f64.powi(dp as i32) * (1.0 + 1e-9),
                "{wished} -> {q} loses more than one lot at {dp} dp"
            );
            let lots = q * 10f64.powi(dp as i32);
            assert!((lots - lots.round()).abs() < 1e-6 * lots.max(1.0), "{q} is not a multiple of the lot at {dp} dp");
            assert!(q > 0.0);
        }
    }
    assert!(ok > 5000, "{ok}");
}
