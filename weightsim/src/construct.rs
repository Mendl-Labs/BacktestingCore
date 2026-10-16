//! MINIMAL portfolio construction: the PF2 BOUNDARY (design 3.2, 6.2 rows PF1/PF2).
//!
//! The book simulator ([`crate::simulate_book`]) never combines sleeves, sizes, filters or budgets trades itself: it
//! hands a plain-data [`ConstructInputs`] to a [`Construct`] and gets a [`ConstructOutput`] (or a whole-book
//! [`ConstructRefusal`]) back. The implementation here, [`MinimalConstruct`], is exactly what the PF0 book key
//! (`AMENDMENT_12.md`) pins and nothing more; PF2's shared crate `portfolio-construct` replaces it. Nothing else in
//! `weightsim` needs to change for that: `simulate_book_with` takes any `&dyn Construct`.
//!
//! # What PF2 must provide (exact signatures)
//!
//! ```text
//! pub trait Construct: Send + Sync {
//!     fn construct(&self, i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal>;
//!     fn inverse_vol_shares(&self, gross_returns: &[&[f64]], lookback: usize, total: f64) -> Option<Vec<f64>>;
//! }
//! ```
//! with the data types of this module (`ConstructInputs`, `SleeveTarget`, `ConstructPolicy`, `TradeFilter`,
//! `CashPolicy`, `ConstructOutput`, `Skip`, `SkipReason`, `ConstructRefusal`). All f64, all pure (no clock, no I/O).
//! `weightsim` is zero-dependency, so PF2 either (a) exports these types itself and this module becomes a re-export,
//! or (b) keeps its own richer types and provides an adapter implementing this trait. Either way the semantics below
//! are the specification, and the per-bar identity to the book key (1e-9) is the acceptance test.
//!
//! # Order of operations (the design's planner order, restricted to what the key pins)
//! 1. combine: `raw_j = SUM_s share_s * w_(s,j)` over the sleeves that have a standing target, sleeves in list order,
//!    signed (opposite sleeves NET), starting from the first term (so one sleeve with share 1 is the identity even for
//!    a signed zero);
//! 2. risk scale: `w_j = raw_j * risk_scale` (a fraction of the capital base);
//! 3. capital base `= min(equity, allocated_capital)` (`allocated_capital = None` means equity), target notional
//!    `= capital_base * w_j`;
//! 4. whole-book gross cap: if `SUM_j |w_j| > max_gross * (1 + 1e-12)` the WHOLE book is refused (never clipped);
//! 5. trade filter on the planned instruments: drop a trade when `|delta| < min_abs` or
//!    `|delta| < min_pct * |target|` (`|current|` when the target is 0), `delta = target - held notional`;
//! 6. cash policy: `Certification` executes every wanted trade in full (the cost is deducted from equity and cash is
//!    the residual, the T0/T1 convention); `Budget` sells first with the proceeds credited, then scales ALL
//!    increases by one common factor so that notional plus fee fits the cash left (reserve 0). Long-only books only.
//!
//! Equity and every currency amount are in the account's units; the book key runs at initial equity 1.0 and converts
//! the planner's currency thresholds (`min_abs`, `allocated_capital`) to those units before calling.

/// Relative tolerance of the gross-cap refusal: refuse iff `gross > cap * (1 + GROSS_CAP_REL_TOL)`. The same value
/// `simulate` uses for `SimConfig::max_gross`, so one-sleeve books agree with it on the boundary.
pub const GROSS_CAP_REL_TOL: f64 = 1e-12;

/// The planner's minimum-trade filter. `min_abs` is in the account's currency units (the same units as equity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeFilter {
    pub min_abs: f64,
    pub min_pct: f64,
}

/// How the cash of the account limits increases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CashPolicy {
    /// T0/T1: every wanted trade executes at its full target; cost comes out of equity; cash is the residual and can
    /// be a few bps negative on a fully invested book.
    Certification,
    /// Planner: sells first (proceeds credited), then all increases scaled by ONE common factor so that
    /// notional + fee fits the cash left. Long-only books only.
    Budget,
}

/// Book-level construction parameters (constant over a run, except that the caller multiplies the overlay scale into
/// `risk_scale` per bar).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstructPolicy {
    /// Multiplier applied to every combined weight (approval constant times the overlay's current scale).
    pub risk_scale: f64,
    pub allocated_capital: Option<f64>,
    pub max_gross: Option<f64>,
    pub trade_filter: Option<TradeFilter>,
    pub cash_policy: CashPolicy,
    /// Cost per unit traded notional; used by `CashPolicy::Budget` to size buys (the account charges the cost itself).
    pub fee_rate: f64,
}

/// One sleeve's standing target, for the sleeves that have one. `weights[k]` belongs to instrument `instruments[k]`.
#[derive(Clone, Copy, Debug)]
pub struct SleeveTarget<'a> {
    pub share: f64,
    pub instruments: &'a [usize],
    pub weights: &'a [f64],
}

/// Everything `construct` may look at. All slices are indexed by instrument (book order) unless stated.
#[derive(Clone, Copy, Debug)]
pub struct ConstructInputs<'a> {
    /// Pre-cost equity at this bar's marks.
    pub equity: f64,
    /// Cash before trading.
    pub cash: f64,
    /// Valuation price per instrument (the carried close where the market is closed).
    pub marks: &'a [f64],
    /// Held units per instrument before trading.
    pub units: &'a [f64],
    /// Sleeves with a standing target, in book order.
    pub sleeves: &'a [SleeveTarget<'a>],
    /// Instruments the driver run may trade this bar.
    pub planned: &'a [bool],
    pub policy: &'a ConstructPolicy,
}

/// Why a planned instrument was not traded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The trade filter is on and the target equals the held notional exactly.
    NoChange,
    BelowMinAbs,
    BelowMinPct,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Skip {
    pub instrument: usize,
    pub reason: SkipReason,
}

/// Result of a successful construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstructOutput {
    /// Combined target per instrument as a fraction of the capital base, risk scale included; `None` when no sleeve
    /// with a standing target owns the instrument.
    pub target_weight: Vec<Option<f64>>,
    pub capital_base: f64,
    /// Units after trading.
    pub units_new: Vec<f64>,
    /// Traded notional per instrument (always >= 0).
    pub traded: Vec<f64>,
    /// Sum of `traded`, accumulated in instrument order.
    pub traded_total: f64,
    pub skipped: Vec<Skip>,
}

/// A whole-book refusal (design R1/R2): nothing is traded on this bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstructRefusal {
    /// `SUM |weight|` (fractions of the capital base) exceeds the cap.
    GrossAboveCap { gross: f64, cap: f64 },
    /// `CashPolicy::Budget` is defined for long-only books; a short target or holding was met.
    BudgetNeedsLongOnly { instrument: usize },
}

/// The PF2 boundary.
pub trait Construct: Send + Sync {
    fn construct(&self, i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal>;

    /// Frozen inverse-volatility shares: for each sleeve the last `lookback` values of `gross_returns[s]` (the series
    /// is already cut at the review bar; fewer than `lookback` values, or a zero/NaN standard deviation, means
    /// `None` = keep the current shares). Shares are `total * (1/sd_s) / SUM_s(1/sd_s)`, with `sd` the sample
    /// standard deviation (ddof 1, two-pass, sequential sums).
    fn inverse_vol_shares(&self, gross_returns: &[&[f64]], lookback: usize, total: f64) -> Option<Vec<f64>>;
}

/// The minimal implementation the book key pins.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinimalConstruct;

/// `min(equity, allocated)`; `None` = no cap.
pub fn capital_base(equity: f64, allocated: Option<f64>) -> f64 {
    match allocated {
        None => equity,
        Some(a) => {
            if a < equity {
                a
            } else {
                equity
            }
        }
    }
}

/// The 2% band's reference: a fraction of the TARGET (of the current value when the target is 0).
fn filter_reference(target: f64, current: f64) -> f64 {
    if target == 0.0 {
        current.abs()
    } else {
        target.abs()
    }
}

impl Construct for MinimalConstruct {
    fn construct(&self, i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal> {
        let n = i.marks.len();
        let p = i.policy;
        // (1) combine, signed, sleeves in list order, starting from the first term.
        let mut raw: Vec<Option<f64>> = vec![None; n];
        for s in i.sleeves {
            for (k, &j) in s.instruments.iter().enumerate() {
                let term = s.share * s.weights[k];
                raw[j] = Some(match raw[j] {
                    None => term,
                    Some(a) => a + term,
                });
            }
        }
        // (2) risk scale.
        let weight: Vec<Option<f64>> = raw.iter().map(|r| r.map(|v| v * p.risk_scale)).collect();
        // (3) capital base.
        let cb = capital_base(i.equity, p.allocated_capital);
        // (4) whole-book gross cap, on the weights (fractions of the capital base).
        if let Some(cap) = p.max_gross {
            let mut gross = 0.0;
            for w in weight.iter().flatten() {
                gross += w.abs();
            }
            if gross > cap * (1.0 + GROSS_CAP_REL_TOL) {
                return Err(ConstructRefusal::GrossAboveCap { gross, cap });
            }
        }
        // (5) wanted trades on the planned, named instruments.
        let mut wanted: Vec<Option<f64>> = vec![None; n];
        let mut skipped = Vec::new();
        for j in 0..n {
            if !i.planned[j] {
                continue;
            }
            let w = match weight[j] {
                Some(w) => w,
                None => continue,
            };
            let notional = cb * w;
            let price = i.marks[j];
            if let Some(f) = p.trade_filter {
                let current = i.units[j] * price;
                let delta = notional - current;
                if delta == 0.0 {
                    skipped.push(Skip { instrument: j, reason: SkipReason::NoChange });
                    continue;
                }
                let ad = delta.abs();
                if ad < f.min_abs {
                    skipped.push(Skip { instrument: j, reason: SkipReason::BelowMinAbs });
                    continue;
                }
                if ad < f.min_pct * filter_reference(notional, current) {
                    skipped.push(Skip { instrument: j, reason: SkipReason::BelowMinPct });
                    continue;
                }
            }
            wanted[j] = Some(notional / price);
        }
        // (6) cash policy.
        let mut units_new = i.units.to_vec();
        let mut traded = vec![0.0; n];
        let mut traded_total = 0.0;
        match p.cash_policy {
            CashPolicy::Certification => {
                for j in 0..n {
                    if let Some(w) = wanted[j] {
                        units_new[j] = w;
                        let tr = (units_new[j] - i.units[j]).abs() * i.marks[j];
                        traded[j] = tr;
                        traded_total += tr;
                    }
                }
            }
            CashPolicy::Budget => {
                let mut cash_run = i.cash;
                let mut sells: Vec<(usize, f64)> = Vec::new();
                let mut buys: Vec<(usize, f64)> = Vec::new();
                for j in 0..n {
                    if let Some(w) = wanted[j] {
                        if !(w >= 0.0 && i.units[j] >= 0.0) {
                            return Err(ConstructRefusal::BudgetNeedsLongOnly { instrument: j });
                        }
                        if w < i.units[j] {
                            sells.push((j, i.units[j] - w));
                        } else if w > i.units[j] {
                            buys.push((j, w - i.units[j]));
                        }
                    }
                }
                for &(j, q) in &sells {
                    let ns = q * i.marks[j];
                    cash_run = cash_run + ns - p.fee_rate * ns;
                    traded_total += ns;
                    traded[j] = ns;
                    units_new[j] = wanted[j].expect("sell has a wanted target");
                }
                let mut needed = 0.0;
                for &(j, q) in &buys {
                    let nb = q * i.marks[j];
                    needed += nb + p.fee_rate * nb;
                }
                let avail = cash_run;
                let factor = if needed <= avail {
                    1.0
                } else if avail > 0.0 {
                    avail / needed
                } else {
                    0.0
                };
                for &(j, q) in &buys {
                    let qq = q * factor;
                    let nb = qq * i.marks[j];
                    cash_run -= nb + p.fee_rate * nb;
                    traded_total += nb;
                    traded[j] = nb;
                    units_new[j] =
                        if factor == 1.0 { wanted[j].expect("buy has a wanted target") } else { i.units[j] + qq };
                }
            }
        }
        Ok(ConstructOutput { target_weight: weight, capital_base: cb, units_new, traded, traded_total, skipped })
    }

    fn inverse_vol_shares(&self, gross_returns: &[&[f64]], lookback: usize, total: f64) -> Option<Vec<f64>> {
        let mut sds = Vec::with_capacity(gross_returns.len());
        for r in gross_returns {
            if r.len() < lookback {
                return None;
            }
            let win = &r[r.len() - lookback..];
            let sd = crate::metrics::std_ddof1(win);
            if !(sd > 0.0) {
                return None;
            }
            sds.push(sd);
        }
        let invs: Vec<f64> = sds.iter().map(|sd| 1.0 / sd).collect();
        let mut tot = 0.0;
        for v in &invs {
            tot += v;
        }
        Some(invs.iter().map(|v| total * (v / tot)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ConstructPolicy {
        ConstructPolicy {
            risk_scale: 1.0,
            allocated_capital: None,
            max_gross: None,
            trade_filter: None,
            cash_policy: CashPolicy::Certification,
            fee_rate: 0.0,
        }
    }

    fn run(
        equity: f64,
        cash: f64,
        marks: &[f64],
        units: &[f64],
        sleeves: &[SleeveTarget<'_>],
        planned: &[bool],
        p: &ConstructPolicy,
    ) -> Result<ConstructOutput, ConstructRefusal> {
        MinimalConstruct.construct(&ConstructInputs { equity, cash, marks, units, sleeves, planned, policy: p })
    }

    #[test]
    fn combine_is_a_signed_sum_that_nets_opposite_sleeves() {
        let a = SleeveTarget { share: 0.5, instruments: &[0, 1], weights: &[0.6, 0.4] };
        let b = SleeveTarget { share: 0.5, instruments: &[0, 2], weights: &[-0.8, 0.2] };
        let out = run(1.0, 1.0, &[1.0, 1.0, 1.0], &[0.0; 3], &[a, b], &[true; 3], &policy()).unwrap();
        // X: 0.5*0.6 + 0.5*(-0.8) = -0.1 (an absolute sum would give 0.7)
        assert!((out.target_weight[0].unwrap() - (-0.1)).abs() < 1e-15);
        assert!((out.target_weight[1].unwrap() - 0.2).abs() < 1e-15);
        assert!((out.target_weight[2].unwrap() - 0.1).abs() < 1e-15);
        assert_eq!(out.traded_total, out.traded.iter().sum::<f64>());
    }

    #[test]
    fn one_sleeve_share_one_is_the_identity_even_for_negative_zero() {
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[-0.0] };
        let out = run(1.0, 1.0, &[2.0], &[0.0], &[a], &[true], &policy()).unwrap();
        assert_eq!(out.target_weight[0].unwrap().to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn risk_scale_multiplies_the_weight_then_the_capital_base_applies() {
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[0.5] };
        let p = ConstructPolicy { risk_scale: 0.8, allocated_capital: Some(1.2), ..policy() };
        // equity 2.0 > allocated 1.2: capital base 1.2; notional = 1.2 * (0.5*0.8) = 0.48; units = 0.48 / 4.0
        let out = run(2.0, 2.0, &[4.0], &[0.0], &[a], &[true], &p).unwrap();
        assert_eq!(out.capital_base, 1.2);
        assert!((out.units_new[0] - 0.12).abs() < 1e-15);
        // equity below the allocation: the capital base is equity
        let out = run(1.0, 1.0, &[4.0], &[0.0], &[a], &[true], &p).unwrap();
        assert_eq!(out.capital_base, 1.0);
        assert!((out.units_new[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn gross_cap_refuses_never_clips_and_the_boundary_is_pinned() {
        let a = SleeveTarget { share: 1.0, instruments: &[0, 1], weights: &[0.6, -0.4] };
        let mk = |cap: f64| ConstructPolicy { max_gross: Some(cap), ..policy() };
        let go = |cap: f64| run(1.0, 1.0, &[1.0, 1.0], &[0.0; 2], &[a], &[true; 2], &mk(cap));
        // gross = 1.0 exactly
        assert!(go(1.0).is_ok(), "gross equal to the cap is permitted");
        // cap just below gross by 1e-9 relative: refuse; just above: permit
        assert!(matches!(go(1.0 / (1.0 + 1e-9)), Err(ConstructRefusal::GrossAboveCap { .. })));
        assert!(go(1.0 / (1.0 - 1e-9)).is_ok());
        // the pinned tolerance is 1e-12 relative
        assert!(go(1.0 / (1.0 + 0.5e-12)).is_ok());
        assert!(matches!(go(1.0 / (1.0 + 2e-12)), Err(ConstructRefusal::GrossAboveCap { .. })));
        // a refusal carries the numbers
        match go(0.5) {
            Err(ConstructRefusal::GrossAboveCap { gross, cap }) => {
                assert!((gross - 1.0).abs() < 1e-15);
                assert_eq!(cap, 0.5);
            }
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn trade_filter_uses_the_target_as_reference_and_the_absolute_floor() {
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[0.5] };
        let mut p = policy();
        p.trade_filter = Some(TradeFilter { min_abs: 0.01, min_pct: 0.02 });
        let go = |units0: f64, p: &ConstructPolicy| run(1.0, 0.5, &[1.0], &[units0], &[a], &[true], p).unwrap();
        // target notional 0.5; held 0.495: delta 0.005 < 2% of 0.5 = 0.01 and < min_abs 0.01 -> skipped
        let o = go(0.495, &p);
        assert_eq!(o.units_new[0], 0.495);
        assert_eq!(o.skipped, vec![Skip { instrument: 0, reason: SkipReason::BelowMinAbs }]);
        // relative band binds: min_abs tiny, delta 0.005 < 0.02*0.5
        p.trade_filter = Some(TradeFilter { min_abs: 1e-6, min_pct: 0.02 });
        let o = go(0.495, &p);
        assert_eq!(o.skipped, vec![Skip { instrument: 0, reason: SkipReason::BelowMinPct }]);
        // large delta trades
        let o = go(0.4, &p);
        assert_eq!(o.units_new[0], 0.5);
        assert!(o.skipped.is_empty());
        // reference is the TARGET, not the current value: target 0.5, current 0.505 -> delta 0.005 = 1% of the target
        // (skipped at 2%), but 0.99% of current (also skipped); choose numbers where they differ:
        // current 0.25 -> delta 0.25 vs 2% of 0.5 = 0.01: trades regardless.
        let o = go(0.25, &p);
        assert_eq!(o.units_new[0], 0.5);
        // exact no-change
        let o = go(0.5, &p);
        assert_eq!(o.skipped, vec![Skip { instrument: 0, reason: SkipReason::NoChange }]);
    }

    #[test]
    fn filter_reference_is_current_when_the_target_is_zero() {
        // target 0, current 1.0: 2% of |current| = 0.02; a min_abs above the delta skips first; with min_abs tiny the
        // relative test (delta 1.0 vs 0.02) lets it through and the position closes.
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[0.0] };
        let mut p = policy();
        p.trade_filter = Some(TradeFilter { min_abs: 1e-9, min_pct: 0.02 });
        let o = run(1.0, 0.0, &[1.0], &[1.0], &[a], &[true], &p).unwrap();
        assert_eq!(o.units_new[0], 0.0);
        assert_eq!(o.traded[0], 1.0);
    }

    #[test]
    fn unplanned_and_unnamed_instruments_are_untouched() {
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[0.5] };
        let out = run(1.0, 1.0, &[1.0, 1.0], &[0.0, 0.7], &[a], &[false, true], &policy()).unwrap();
        assert_eq!(out.units_new, vec![0.0, 0.7]);
        assert_eq!(out.traded, vec![0.0, 0.0]);
        assert_eq!(out.target_weight, vec![Some(0.5), None]);
    }

    #[test]
    fn budget_policy_sells_first_then_scales_all_buys_by_one_factor() {
        // Two instruments, price 1. Held: A 0.6, B 0.0. Cash 0.1. Targets: A 0.2 (sell 0.4), B 0.9 (buy 0.9).
        let a = SleeveTarget { share: 1.0, instruments: &[0, 1], weights: &[0.2, 0.9] };
        let p = ConstructPolicy { cash_policy: CashPolicy::Budget, fee_rate: 0.001, ..policy() };
        let o = run(1.0, 0.1, &[1.0, 1.0], &[0.6, 0.0], &[a], &[true, true], &p).unwrap();
        // proceeds 0.4 - 0.0004 = 0.3996 -> cash 0.4996; buy 0.9 needs 0.9009 -> factor 0.4996/0.9009
        let factor = 0.4996 / 0.9009;
        assert!((o.units_new[1] - 0.9 * factor).abs() < 1e-12);
        assert_eq!(o.units_new[0], 0.2);
        // traded = sell 0.4 + scaled buy
        assert!((o.traded_total - (0.4 + 0.9 * factor)).abs() < 1e-12);
        // enough cash: no scaling
        let o = run(1.0, 1.0, &[1.0, 1.0], &[0.6, 0.0], &[a], &[true, true], &p).unwrap();
        assert_eq!(o.units_new[1], 0.9);
        // no cash at all and nothing to sell: buys scale to zero
        let b = SleeveTarget { share: 1.0, instruments: &[0], weights: &[0.9] };
        let o = run(1.0, 0.0, &[1.0], &[0.0], &[b], &[true], &p).unwrap();
        assert_eq!(o.units_new[0], 0.0);
    }

    #[test]
    fn budget_policy_refuses_shorts() {
        let a = SleeveTarget { share: 1.0, instruments: &[0], weights: &[-0.3] };
        let p = ConstructPolicy { cash_policy: CashPolicy::Budget, ..policy() };
        assert_eq!(
            run(1.0, 1.0, &[1.0], &[0.0], &[a], &[true], &p),
            Err(ConstructRefusal::BudgetNeedsLongOnly { instrument: 0 })
        );
    }

    #[test]
    fn inverse_vol_shares_match_a_hand_computation() {
        let calm = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01];
        let wild = [0.03, -0.03, 0.03, -0.03, 0.03, -0.03];
        // sd(calm) = 0.01*sqrt(6/5), sd(wild) = 3x -> inverse-vol shares 0.75 / 0.25 of the total
        let s = MinimalConstruct.inverse_vol_shares(&[&calm, &wild], 6, 1.0).unwrap();
        assert!((s[0] - 0.75).abs() < 1e-12 && (s[1] - 0.25).abs() < 1e-12, "{s:?}");
        let s = MinimalConstruct.inverse_vol_shares(&[&calm, &wild], 6, 0.8).unwrap();
        assert!((s[0] - 0.6).abs() < 1e-12 && (s[1] - 0.2).abs() < 1e-12);
        // only the LAST `lookback` observations count
        let long: Vec<f64> = [0.5, -0.5].iter().chain(calm.iter()).copied().collect();
        let s2 = MinimalConstruct.inverse_vol_shares(&[&long, &wild], 6, 1.0).unwrap();
        assert!((s2[0] - 0.75).abs() < 1e-12);
        // too short a window, or zero volatility: keep current shares (None)
        assert!(MinimalConstruct.inverse_vol_shares(&[&calm[..5], &wild], 6, 1.0).is_none());
        assert!(MinimalConstruct.inverse_vol_shares(&[&[0.0; 6], &wild], 6, 1.0).is_none());
    }
}
