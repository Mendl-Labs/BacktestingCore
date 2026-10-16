//! The drawdown ladder and daily-loss limit (f64).
//!
//! Golden vectors are transcribed from SignalEngine `crates/rebalancer-risk/tests/overlay.rs` (main, 4937a43): the
//! baseline policy (daily loss 3%, shrink 0.5 at 10%, halt at 20%, recovery fraction 0.5), the three-rung policy
//! (0.75 at 5%, 0.5 at 10%, halt at 20%) and the exact boundary equities of those tests. The decisive independent check is
//! the exact-integer oracle at the bottom: 10,000 generated equity paths through the f64 ladder and through an oracle
//! that decides every comparison in integer cents and basis points (no rounding anywhere), which must agree step by step.

mod common;

use common::*;
use portfolio_construct::*;

fn shrink(at: f64, scale: f64) -> Rung {
    Rung { at, action: RungAction::Shrink { scale } }
}
fn halt(at: f64) -> Rung {
    Rung { at, action: RungAction::HaltFlatten }
}

fn baseline() -> Ladder {
    Ladder::new(0.03, vec![shrink(0.1, 0.5), halt(0.2)], 0.5).unwrap()
}

fn three_rung() -> Ladder {
    Ladder::new(0.03, vec![shrink(0.05, 0.75), shrink(0.1, 0.5), halt(0.2)], 0.5).unwrap()
}

fn kind(d: &LadderDecision) -> &'static str {
    match d.action {
        LadderAction::None => "none",
        LadderAction::Shrink { .. } => "shrink",
        LadderAction::HaltFlatten => "halt",
    }
}

/// An Active state whose mark is `hwm`, evaluated at `equity` with the day starting at `equity` (a pure drawdown test).
fn eval(l: &Ladder, hwm: f64, equity: f64) -> LadderDecision {
    let mut st = LadderState { hwm: Some(hwm), status: LadderStatus::Active };
    l.step(&mut st, equity, equity)
}

/// A pure daily-loss test: mark and day start are both `day_start`.
fn eval_day(l: &Ladder, day_start: f64, equity: f64) -> LadderDecision {
    let mut st = LadderState { hwm: Some(day_start), status: LadderStatus::Active };
    l.step(&mut st, equity, day_start)
}

/// The Shrunk state of the live `shrunk_state` helper: hwm 10000, equity 9000 -> shrink rung 0.
fn shrunk(l: &Ladder) -> LadderState {
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 9000.0, 9000.0);
    assert_eq!(d.action, LadderAction::Shrink { scale: 0.5 });
    assert_eq!(st.status, LadderStatus::Shrunk(0));
    st
}

fn step_at(l: &Ladder, st: &LadderState, equity: f64) -> LadderDecision {
    let mut s = st.clone();
    l.step(&mut s, equity, equity)
}

// ---------------------------------------------------------------------------------------------------------------
// golden: boundaries (overlay.rs)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn ladder_rung_one_triggers_exactly_at_the_boundary() {
    let l = baseline();
    for (equity, want) in
        [(10000.0, "none"), (9000.01, "none"), (9000.0, "shrink"), (8999.99, "shrink"), (8000.01, "shrink")]
    {
        let d = eval(&l, 10000.0, equity);
        assert_eq!(kind(&d), want, "equity {equity}: {d:?}");
    }
    let at = eval(&l, 10000.0, 9000.0);
    assert_eq!(at.scale, 0.5);
    assert_eq!(at.action, LadderAction::Shrink { scale: 0.5 });
    assert_eq!(at.code, LadderCode::DrawdownShrink);
    assert_eq!(at.rung, Some(0));
    assert_eq!(eval(&l, 10000.0, 9000.01).scale, 1.0);
}

#[test]
fn ladder_halt_rung_triggers_exactly_at_the_boundary() {
    let l = baseline();
    assert_eq!(kind(&eval(&l, 10000.0, 8000.01)), "shrink", "just above the halt rung is still only a shrink");
    let at = eval(&l, 10000.0, 8000.0);
    assert_eq!(at.action, LadderAction::HaltFlatten);
    assert_eq!(at.scale, 0.0);
    assert_eq!(at.code, LadderCode::DrawdownHalt);
    assert_eq!(kind(&eval(&l, 10000.0, 7999.99)), "halt");
    assert_eq!(kind(&eval(&l, 10000.0, 1.0)), "halt");
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    l.step(&mut st, 8000.0, 8000.0);
    assert_eq!(st.status, LadderStatus::Halted(LadderHalt::DrawdownLadder));
}

#[test]
fn boundaries_scale_with_the_high_water_mark_not_with_a_constant() {
    let l = baseline();
    // hwm 12345.67: 10% = 1234.567, so the trigger equity is 11111.103 (a decimal tie that float noise can break).
    assert_eq!(kind(&eval(&l, 12345.67, 11111.104)), "none");
    assert_eq!(kind(&eval(&l, 12345.67, 11111.103)), "shrink");
    // 20% = 2469.134 -> 9876.536
    assert_eq!(kind(&eval(&l, 12345.67, 9876.537)), "shrink");
    assert_eq!(kind(&eval(&l, 12345.67, 9876.536)), "halt");
}

#[test]
fn a_new_high_is_not_a_drawdown_and_moves_the_mark() {
    let l = baseline();
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 11000.0, 11000.0);
    assert_eq!(d.action, LadderAction::None);
    assert_eq!(st.hwm, Some(11000.0));
    // 10% below the NEW mark is 9900.
    assert_eq!(kind(&step_at(&l, &st, 9900.0)), "shrink");
    assert_eq!(kind(&step_at(&l, &st, 9900.01)), "none");
}

#[test]
fn the_high_water_mark_only_ratchets_up_and_freezes_when_halted() {
    let l = baseline();
    let mut st = LadderState::new();
    assert_eq!(st.hwm, None);
    l.step(&mut st, 10000.0, 10000.0);
    assert_eq!(st.hwm, Some(10000.0));
    l.step(&mut st, 9800.0, 10000.0);
    assert_eq!(st.hwm, Some(10000.0), "a fall never lowers it");
    l.step(&mut st, 10500.0, 10000.0);
    assert_eq!(st.hwm, Some(10500.0));
    // Halt (equity 7000: -33%), then a huge equity: the mark stays frozen.
    l.step(&mut st, 7000.0, 7000.0);
    assert!(st.is_halted());
    l.step(&mut st, 20000.0, 7000.0);
    assert_eq!(st.hwm, Some(10500.0), "frozen while halted");
}

// ---------------------------------------------------------------------------------------------------------------
// golden: daily loss (3% of day-start equity)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn daily_loss_halts_exactly_at_the_limit() {
    let l = baseline();
    assert_eq!(kind(&eval_day(&l, 10000.0, 9700.01)), "none"); // 2.9999% (and drawdown 3% < 10%)
    let at = eval_day(&l, 10000.0, 9700.0);
    assert_eq!(at.action, LadderAction::HaltFlatten, "{at:?}");
    assert_eq!(at.code, LadderCode::DailyLossHalt);
    assert_eq!(kind(&eval_day(&l, 10000.0, 9699.99)), "halt");
    assert_eq!(kind(&eval_day(&l, 10000.0, 10500.0)), "none", "gains never trip it");
}

#[test]
fn daily_loss_is_measured_from_the_days_start_not_from_the_high_water_mark() {
    let l = baseline();
    // hwm 12000 but the day started at 10000: a fall to 9700 is a 3% daily loss (drawdown 19.2% is only a shrink).
    let mut st = LadderState { hwm: Some(12000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 9700.0, 10000.0);
    assert_eq!(d.action, LadderAction::HaltFlatten);
    assert_eq!(st.status, LadderStatus::Halted(LadderHalt::DailyLoss));
    // A new day re-bases: the same 9700 with a day start of 9700 is not a daily loss.
    let mut st = LadderState { hwm: Some(12000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 9700.0, 9700.0);
    assert_ne!(d.code, LadderCode::DailyLossHalt);
}

#[test]
fn both_limits_hit_reports_both_and_prefers_the_daily_loss_reason() {
    let l = baseline();
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 7900.0, 10000.0); // -21%: daily loss AND the halt rung
    assert_eq!(d.code, LadderCode::DailyLossHalt);
    assert!(d.both_halts);
    assert_eq!(st.status, LadderStatus::Halted(LadderHalt::DailyLoss));
    // Only the drawdown halt: the daily loss is within the limit.
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 7900.0, 7950.0);
    assert_eq!((d.code, d.both_halts), (LadderCode::DrawdownHalt, false));
}

// ---------------------------------------------------------------------------------------------------------------
// golden: fail closed, sticky halts
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn non_positive_or_non_finite_equity_halts_fail_closed() {
    let l = baseline();
    for e in [0.0, -1.0, -0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
        let d = l.step(&mut st, e, 10000.0);
        assert_eq!(d.action, LadderAction::HaltFlatten, "{e}");
        assert_eq!(d.code, LadderCode::EquityInvalid);
        assert_eq!(st.status, LadderStatus::Halted(LadderHalt::EquityInvalid));
    }
}

#[test]
fn a_halted_account_gets_scale_zero_whatever_the_equity() {
    let l = baseline();
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    l.step(&mut st, 7000.0, 7000.0);
    assert!(st.is_halted());
    for equity in [1.0, 7000.0, 10000.0, 50000.0, 1_000_000.0] {
        let mut s = st.clone();
        let d = l.step(&mut s, equity, equity);
        assert_eq!(d.scale, 0.0, "{equity}");
        assert_eq!(d.code, LadderCode::AlreadyHalted);
        assert_eq!(d.action, LadderAction::None);
        assert_eq!(s, st, "a halt is sticky: nothing changes");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// golden: shrink recovery and hysteresis
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn a_shrink_is_released_only_when_the_drawdown_is_back_within_half_of_the_rung() {
    let l = baseline(); // recovery fraction 0.5: release at drawdown <= 5% (equity >= 9500)
    let st = shrunk(&l);
    assert_eq!(st.status, LadderStatus::Shrunk(0));
    for equity in [9000.0, 9100.0, 9400.0, 9499.99] {
        let d = step_at(&l, &st, equity);
        assert_eq!(d.action, LadderAction::Shrink { scale: 0.5 }, "equity {equity}");
        assert!(matches!(d.code, LadderCode::ShrinkHeld | LadderCode::DrawdownShrink), "{:?}", d.code);
    }
    let d = step_at(&l, &st, 9500.0); // exactly at half the rung: released (inclusive)
    assert_eq!(d.action, LadderAction::None, "{d:?}");
    assert_eq!(d.code, LadderCode::Recovered);
    assert_eq!(d.scale, 1.0);
    // The same equity from an ACTIVE state is simply "no action": the hysteresis only holds an existing shrink.
    assert_eq!(kind(&eval(&l, 10000.0, 9400.0)), "none");
}

#[test]
fn shrink_holds_while_recovering_then_returns_to_active() {
    let l = baseline();
    let mut st = shrunk(&l);
    let d = l.step(&mut st, 9300.0, 9300.0);
    assert_eq!(st.status, LadderStatus::Shrunk(0));
    assert_eq!(d.code, LadderCode::ShrinkHeld);
    let d = l.step(&mut st, 9600.0, 9600.0);
    assert_eq!(st.status, LadderStatus::Active);
    assert_eq!(d.scale, 1.0);
    assert_eq!(d.code, LadderCode::Recovered);
}

#[test]
fn the_recovery_fraction_is_a_documented_parameter() {
    let strict = Ladder::new(0.03, vec![shrink(0.1, 0.5), halt(0.2)], 0.2).unwrap(); // release at drawdown <= 2% (>= 9800)
    let st = shrunk(&strict);
    assert_eq!(kind(&step_at(&strict, &st, 9799.99)), "shrink");
    assert_eq!(kind(&step_at(&strict, &st, 9800.0)), "none");
    let none = Ladder::new(0.03, vec![shrink(0.1, 0.5), halt(0.2)], 1.0).unwrap(); // no hysteresis
    let st = shrunk(&none);
    assert_eq!(kind(&step_at(&none, &st, 9000.0)), "shrink", "at the trigger the raw rung hits again");
    assert_eq!(kind(&step_at(&none, &st, 9000.01)), "none");
    assert!(Ladder::new(0.03, vec![halt(0.2)], 0.0).is_err());
    assert!(Ladder::new(0.03, vec![halt(0.2)], 1.01).is_err());
    assert_eq!(baseline().recovery_fraction(), DEFAULT_RECOVERY_FRACTION);
}

#[test]
fn several_shrink_rungs_step_down_one_at_a_time() {
    let l = three_rung(); // 5% -> 0.75, 10% -> 0.5, 20% halt; recovery 0.5
    let mut st = LadderState { hwm: Some(10000.0), status: LadderStatus::Active };
    let d = l.step(&mut st, 8900.0, 8900.0); // -11%
    assert_eq!(d.rung, Some(1));
    assert_eq!(d.scale, 0.5);
    // Drawdown 4%: rung 1 (trigger 10%, release <= 5%) is released, rung 0 (trigger 5%, release <= 2.5%) is not.
    let d = l.step(&mut st, 9600.0, 9600.0);
    assert_eq!(d.code, LadderCode::PartialRecovery, "{d:?}");
    assert_eq!(d.rung, Some(0));
    assert_eq!(d.scale, 0.75);
    assert_eq!(st.status, LadderStatus::Shrunk(0));
    // Drawdown 2.5%: rung 0 released too.
    let d = l.step(&mut st, 9750.0, 9750.0);
    assert_eq!(d.code, LadderCode::Recovered);
    assert_eq!(st.status, LadderStatus::Active);
    // Jumping straight from Active to the second rung is one decision.
    let d = l.step(&mut st, 9000.0, 9000.0);
    assert_eq!(d.rung, Some(1));
    assert_eq!(d.scale, 0.5);
}

// ---------------------------------------------------------------------------------------------------------------
// golden: policy construction (overlay.rs `policy_new_enforces_the_ladder_invariants`)
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn ladder_new_enforces_the_ladder_invariants() {
    let ok = |r: Vec<Rung>| Ladder::new(0.03, r, 0.5);
    assert!(ok(vec![shrink(0.1, 0.5), halt(0.2)]).is_ok());
    assert!(ok(vec![halt(0.2)]).is_ok(), "a ladder may be a single halt rung");
    assert!(ok(vec![]).is_err(), "empty ladder");
    assert!(ok(vec![shrink(0.1, 0.5)]).is_err(), "the last rung must halt");
    assert!(ok(vec![halt(0.1), halt(0.2)]).is_err(), "only the last rung may halt");
    assert!(ok(vec![shrink(0.2, 0.5), halt(0.2)]).is_err(), "strictly ascending");
    assert!(ok(vec![shrink(0.3, 0.5), shrink(0.2, 0.4), halt(0.4)]).is_err(), "ascending order");
    assert!(ok(vec![shrink(0.1, 0.5), shrink(0.15, 0.6), halt(0.2)]).is_err(), "no scaling back up");
    assert!(ok(vec![shrink(0.1, 1.0), halt(0.2)]).is_err(), "scale must be below 1");
    assert!(ok(vec![shrink(0.1, 0.0), halt(0.2)]).is_err(), "scale must be above 0");
    assert!(ok(vec![shrink(0.1, 0.5), halt(1.5)]).is_err(), "rung above 1");
    assert!(ok(vec![shrink(0.0, 0.5), halt(0.2)]).is_err(), "rung at 0");
    assert!(Ladder::new(0.0, vec![halt(0.2)], 0.5).is_err());
    assert!(Ladder::new(1.5, vec![halt(0.2)], 0.5).is_err());
    assert!(Ladder::new(f64::NAN, vec![halt(0.2)], 0.5).is_err());
    let l = baseline();
    assert_eq!((l.daily_loss_limit(), l.rungs().len()), (0.03, 2));
}

// ---------------------------------------------------------------------------------------------------------------
// hand-computed and property tests
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn the_ladder_scale_shrinks_the_book_through_construct() {
    // Drawdown 10% on the baseline ladder gives scale 0.5; construct halves every target.
    let l = baseline();
    let d = eval(&l, 10000.0, 9000.0);
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]);
    c.equity = 50000.0;
    c.rs = RiskScale::new(1.0, d.scale);
    let out = c.with_cash(50000.0).ok();
    assert_eq!(line(&out, "SPY").target_notional, 500.0);
    // A halt (scale 0) flattens: every target is zero and a held position is sold.
    let h = eval(&l, 10000.0, 7000.0);
    let mut c = Case::planner(vec![etf(1.0, [0.2, 0.0, 0.0, 0.0, 0.0])]).held(SPY, 2.0);
    c.equity = 50000.0;
    c.rs = RiskScale::new(1.0, h.scale);
    let out = c.with_cash(50000.0).ok();
    assert_eq!(line(&out, "SPY").target_notional, 0.0);
    assert_eq!((out.trades[0].side, out.trades[0].quantity), (Side::Sell, 2.0));
}

#[test]
fn monotone_lower_equity_never_yields_a_larger_scale() {
    // For a fixed prior state, over a dense grid of equities.
    let l = three_rung();
    let priors = [
        LadderState { hwm: Some(10000.0), status: LadderStatus::Active },
        LadderState { hwm: Some(10000.0), status: LadderStatus::Shrunk(0) },
        LadderState { hwm: Some(10000.0), status: LadderStatus::Shrunk(1) },
    ];
    for prior in &priors {
        let mut last = f64::INFINITY;
        // Ascending equity => non-decreasing scale.
        let mut prev_scale = -1.0;
        for cents in (100..=1_100_000).step_by(37) {
            let e = f64::from(cents) / 100.0;
            let mut st = prior.clone();
            let d = l.step(&mut st, e, e);
            assert!(d.scale >= prev_scale, "equity {e}: scale {} after {prev_scale} from {prior:?}", d.scale);
            prev_scale = d.scale;
            last = last.min(d.scale);
        }
        assert_eq!(last, 0.0, "the deepest equities halt");
        assert_eq!(prev_scale, 1.0, "the highest equities are full size");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// the exact-integer oracle: 10,000 generated paths, step-by-step agreement
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum OStatus {
    Active,
    Shrunk(usize),
    Halted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ODecision {
    scale_idx: i32, // -1 = full size (1.0), -2 = halted (0.0), k = shrink rung k's scale
    code: LadderCode,
    rung: Option<usize>,
}

/// Rungs in basis points of the mark: (at_bp, Some(scale index) for shrink | None for halt); daily loss in bp; recovery as
/// a rational num/den. Every comparison is integer arithmetic on cents (i128), so nothing rounds.
struct Oracle {
    daily_bp: i128,
    rungs: Vec<(i128, bool)>, // (at_bp, is_halt)
    rec_num: i128,
    rec_den: i128,
}

impl Oracle {
    fn breached(&self, reference: i128, equity: i128, bp: i128) -> bool {
        (reference - equity) * 10000 >= bp * reference
    }
    fn recovered(&self, reference: i128, equity: i128, bp: i128) -> bool {
        (reference - equity) * 10000 * self.rec_den <= bp * self.rec_num * reference
    }
    fn step(&self, status: &mut OStatus, hwm: &mut Option<i128>, equity: i128, day_start: i128) -> ODecision {
        if *status == OStatus::Halted {
            return ODecision { scale_idx: -2, code: LadderCode::AlreadyHalted, rung: None };
        }
        if equity <= 0 {
            *status = OStatus::Halted;
            return ODecision { scale_idx: -2, code: LadderCode::EquityInvalid, rung: None };
        }
        let h = match *hwm {
            Some(h) if h >= equity => h,
            _ => equity,
        };
        *hwm = Some(h);
        let daily = day_start > 0 && self.breached(day_start, equity, self.daily_bp);
        let mut raw: Option<usize> = None;
        for (i, (bp, _)) in self.rungs.iter().enumerate() {
            if self.breached(h, equity, *bp) {
                raw = Some(i);
            }
        }
        let dd_halt = raw.is_some_and(|i| self.rungs[i].1);
        if daily || dd_halt {
            *status = OStatus::Halted;
            let code = if daily { LadderCode::DailyLossHalt } else { LadderCode::DrawdownHalt };
            return ODecision { scale_idx: -2, code, rung: None };
        }
        let last_shrink = self.rungs.len().checked_sub(2);
        let entered = if let OStatus::Shrunk(r) = *status { Some(r) } else { None };
        let mut held = match (entered, last_shrink) {
            (Some(c), Some(l)) => Some(c.min(l)),
            _ => None,
        };
        while let Some(c) = held {
            if self.recovered(h, equity, self.rungs[c].0) {
                held = c.checked_sub(1);
            } else {
                break;
            }
        }
        let target = match (raw, held) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        match target {
            None => {
                *status = OStatus::Active;
                let code = if entered.is_some() { LadderCode::Recovered } else { LadderCode::NoAction };
                ODecision { scale_idx: -1, code, rung: None }
            }
            Some(i) => {
                let newly = raw == Some(i) && entered != Some(i);
                let stepped_down = entered.is_some_and(|c| i < c);
                let code = if newly {
                    LadderCode::DrawdownShrink
                } else if stepped_down {
                    LadderCode::PartialRecovery
                } else {
                    LadderCode::ShrinkHeld
                };
                *status = OStatus::Shrunk(i);
                ODecision { scale_idx: i as i32, code, rung: Some(i) }
            }
        }
    }
}

#[test]
fn ten_thousand_generated_paths_agree_with_the_exact_integer_oracle_step_by_step() {
    // Ladder: daily 3% (300 bp), shrink 0.75 at 5% (500 bp), shrink 0.5 at 10% (1000 bp), halt at 20% (2000 bp), recovery 1/2.
    let ladder = three_rung();
    let scales = [0.75, 0.5];
    let oracle =
        Oracle { daily_bp: 300, rungs: vec![(500, false), (1000, false), (2000, true)], rec_num: 1, rec_den: 2 };
    let mut rng = SplitMix64(0xC0FFEE);
    let (mut ties, mut steps, mut halts, mut shrinks, mut recoveries) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for path in 0..10_000u64 {
        // Marks are multiples of 10000 cents half the time, so that basis-point thresholds land exactly on cents.
        let start_cents: i128 = if rng.chance(50) {
            10000 * i128::from(rng.range(50, 500))
        } else {
            i128::from(rng.range(100_000, 5_000_000))
        };
        let mut equity = start_cents;
        let mut day_start = equity;
        let mut ost = OStatus::Active;
        let mut ohwm: Option<i128> = None;
        let mut fst = LadderState::new();
        for step in 0..40 {
            if step % 5 == 0 {
                day_start = equity;
            }
            // Next equity: a random move, or (a quarter of the time) exactly ON a rung/release threshold of the current mark.
            let mark = ohwm.unwrap_or(equity).max(equity);
            equity = if rng.chance(25) && mark % 10000 == 0 {
                let bps = *rng.pick(&[300i128, 500, 1000, 2000, 250, 500 * 5 / 10, 100, 1000 / 2]);
                mark - mark * bps / 10000
            } else {
                let mv = i128::from(rng.range(0, 700)) - 330; // -3.30% .. +3.70%
                (equity + equity * mv / 10000).max(1)
            };
            let od = oracle.step(&mut ost, &mut ohwm, equity, day_start);
            let eq_f = equity as f64 / 100.0;
            let d = ladder.step(&mut fst, eq_f, day_start as f64 / 100.0);
            let want_scale = match od.scale_idx {
                -1 => 1.0,
                -2 => 0.0,
                k => scales[k as usize],
            };
            assert_eq!(
                (d.scale, d.code, d.rung),
                (want_scale, od.code, od.rung),
                "path {path} step {step}: equity {equity} cents, day_start {day_start}, hwm {ohwm:?}, oracle {od:?}, float {d:?}"
            );
            let want_status = match ost {
                OStatus::Active => LadderStatus::Active,
                OStatus::Shrunk(r) => LadderStatus::Shrunk(r),
                OStatus::Halted => fst.status,
            };
            assert_eq!(fst.status, want_status, "path {path} step {step}");
            if ost == OStatus::Halted {
                assert!(fst.is_halted());
                halts += 1;
                break;
            }
            steps += 1;
            if od.code == LadderCode::DrawdownShrink {
                shrinks += 1;
            }
            if od.code == LadderCode::Recovered {
                recoveries += 1;
            }
            if mark % 10000 == 0 && ((mark - equity) * 10000) % mark == 0 {
                ties += 1;
            }
        }
    }
    // The generator really exercised the interesting cases (a vacuous pass would be worthless).
    assert!(halts > 1500, "halts {halts}");
    assert!(shrinks > 1500, "shrinks {shrinks}");
    assert!(recoveries > 300, "recoveries {recoveries}");
    assert!(ties > 300, "exact-boundary steps {ties}");
    assert!(steps > 100_000, "steps {steps}");
}
