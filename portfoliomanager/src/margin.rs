//! Pure margin/leverage math shared by every P&L engine that supports
//! leveraged positions (`backtest::python_simulation` and `backtest::candle_sim`).
//!
//! Kept as one shared module rather than duplicated per-engine specifically
//! to prevent the two-engine-drift class of bug this platform has hit
//! repeatedly (GA-fitness scoring under different economics than the final
//! backtest that reports on the same strategy).
//!
//! v1 scope: isolated margin only (per-position, no cross-margining), flat
//! maintenance-margin ratio (no tiered curve), single quote-currency balance.

use crate::PositionSide;

/// notional / leverage. `leverage` is floored at 1.0 -- there is no such
/// thing as <1x margin in this platform's model.
pub fn initial_margin(notional: f64, leverage: f64) -> f64 {
    notional / leverage.max(1.0)
}

/// Flat maintenance-margin requirement against notional at the current mark
/// price (isolated margin, v1 -- no tiered curve).
pub fn maintenance_margin(notional_at_mark: f64, maintenance_margin_ratio: f64) -> f64 {
    notional_at_mark * maintenance_margin_ratio
}

/// Theoretical liquidation price for an isolated-margin position, using the
/// standard linear approximation real exchanges (e.g. Binance USDT-M
/// futures) publish for isolated margin:
///   long:  entry * (1 - 1/leverage + maintenance_margin_ratio)
///   short: entry * (1 + 1/leverage - maintenance_margin_ratio)
///
/// `leverage <= 1.0` disables liquidation entirely: longs return `0.0`
/// (price can never go negative, so this never triggers) and shorts return
/// `f64::INFINITY` (likewise never triggers) -- leverage-off semantics fall
/// out of the formula with no special-casing at call sites.
pub fn liquidation_price(
    entry_price: f64,
    side: PositionSide,
    leverage: f64,
    maintenance_margin_ratio: f64,
) -> f64 {
    if leverage <= 1.0 {
        return match side {
            PositionSide::Long => 0.0,
            PositionSide::Short => f64::INFINITY,
        };
    }
    match side {
        PositionSide::Long => entry_price * (1.0 - 1.0 / leverage + maintenance_margin_ratio),
        PositionSide::Short => entry_price * (1.0 + 1.0 / leverage - maintenance_margin_ratio),
    }
}

/// Whether a position should be liquidated, given the current bar's intrabar
/// extreme price -- pass the bar LOW for longs and the bar HIGH for shorts,
/// mirroring how this platform's existing stop-loss/take-profit checks
/// already use intrabar extremes rather than just the close, so a
/// liquidation isn't missed when price gaps through the trigger mid-bar.
pub fn is_liquidated(
    intrabar_extreme_price: f64,
    entry_price: f64,
    side: PositionSide,
    leverage: f64,
    maintenance_margin_ratio: f64,
) -> bool {
    if leverage <= 1.0 {
        return false;
    }
    let liq_price = liquidation_price(entry_price, side, leverage, maintenance_margin_ratio);
    match side {
        PositionSide::Long => intrabar_extreme_price <= liq_price,
        PositionSide::Short => intrabar_extreme_price >= liq_price,
    }
}

/// Size a new order under leverage.
///
/// `margin_posted = equity * position_size_pct` always -- identical to this
/// platform's pre-leverage sizing formula. This identity (not a coincidence)
/// is what makes `leverage = 1.0` behavior-preserving: only the notional
/// (and therefore quantity) scales with leverage, the cash actually debited
/// from free balance does not change.
///
/// Returns `(notional, margin_posted, qty)`.
pub fn size_leveraged_order(
    equity: f64,
    position_size_pct: f64,
    leverage: f64,
    fill_price: f64,
) -> (f64, f64, f64) {
    let leverage = leverage.max(1.0);
    let notional = equity * position_size_pct * leverage;
    let margin_posted = initial_margin(notional, leverage);
    let qty = if fill_price > 0.0 { notional / fill_price } else { 0.0 };
    (notional, margin_posted, qty)
}

/// Hard ceiling on TOTAL notional exposure of one logical position (existing
/// quantity already held, plus every leg of a new order being sized against
/// it), as a fraction of `equity * leverage` -- deliberately matching
/// `size_leveraged_order`'s own `notional = equity * position_size_pct *
/// leverage` formula at `position_size_pct = 1.0`, so this cap allows exactly
/// what ONE single max-sized order at the configured leverage would already
/// be entitled to, no more. `size_leveraged_order` above only bounds a
/// single order's own notional -- it has no idea whether the strategy is
/// pyramiding into an already-open position (repeated BUY/ScaleIn signals)
/// or, for pairs, scaling a second leg by a hedge ratio independent of the
/// first leg's own cap. Neither case is caught by the existing `<=1.0`
/// per-call clamp on `position_size_pct`, and a flat (leverage-blind) equity
/// cap would incorrectly clip legitimate leveraged notional -- confirmed via
/// a regression this fix introduced and then had to correct
/// (`leverage_scales_pnl_proportionally_to_notional` in
/// `backtest::pair_simulation`, which asserts 4x leverage produces 4x pnl on
/// an identical trade -- a leverage-blind cap silently clipped the 4x case
/// back down to the 1x case's notional).
///
/// Confirmed live in production: an ADA-USD/XRP-USD pairs strategy
/// (job 50ac25dd) combined a flat per-leg sizing fraction with a hedge ratio
/// of 1.83 on leg B and GA-evolved pyramiding (`max_inventory_multiple: 5.0`)
/// -- none of those three multipliers were cross-checked against each other
/// or against total account exposure, and the account saw a single trade
/// lose 132% of capital (max drawdown 413%) at leverage=1.0, where this cap
/// and a flat equity cap agree. This cap is name-agnostic (it reads notional
/// in dollar terms, not strategy-chosen parameter names like
/// `risk_per_trade`), so it catches every strategy regardless of how it
/// structures its own sizing logic.
pub const MAX_AGGREGATE_POSITION_EQUITY_PCT: f64 = 1.0;

/// Clamp `proposed_qty` (a new order's quantity, at `fill_price`) so that
/// `existing_notional + proposed_notional` never exceeds
/// `equity * MAX_AGGREGATE_POSITION_EQUITY_PCT * leverage.max(1.0)`.
/// `existing_notional` is the dollar exposure already held on this
/// instrument (e.g. `portfolio.net_position.abs() * current_price` for a
/// single-asset position, or the sum of both legs' notional for a pair).
/// `leverage` must match whatever `size_leveraged_order` used to size
/// existing/prior orders on this same position, or the cap will be
/// inconsistent with what was legitimately already opened. Returns
/// `proposed_qty` unchanged whenever there's enough headroom, `0.0` if the
/// position is already at or past the cap, and a proportionally reduced
/// quantity otherwise -- never a negative or NaN quantity.
pub fn clamp_aggregate_position_qty(
    proposed_qty: f64,
    fill_price: f64,
    existing_notional: f64,
    equity: f64,
    leverage: f64,
) -> f64 {
    if !proposed_qty.is_finite() || proposed_qty <= 0.0 || fill_price <= 0.0 {
        return 0.0;
    }
    let leverage = leverage.max(1.0);
    let max_total_notional = (equity * MAX_AGGREGATE_POSITION_EQUITY_PCT * leverage).max(0.0);
    let existing_notional = existing_notional.max(0.0);
    let headroom_notional = (max_total_notional - existing_notional).max(0.0);
    let proposed_notional = proposed_qty * fill_price;
    if proposed_notional <= headroom_notional {
        return proposed_qty;
    }
    (headroom_notional / fill_price).max(0.0)
}

/// Two-leg counterpart to `clamp_aggregate_position_qty`, for pairs/spread
/// strategies where a second leg's quantity is derived as a hedge-ratio
/// MULTIPLE of the first leg's quantity (`qty_b = hedge_ratio.abs() * qty_a`)
/// rather than independently sized. Whenever `|hedge_ratio| > 1` -- routine
/// for pairs whose legs trade at different price scales -- the combined
/// notional across both legs silently exceeds what sizing leg A alone would
/// suggest, with no independent cap on leg B. Scales BOTH legs down together
/// (preserving the hedge ratio / relative sizing between them) if their
/// combined notional would exceed `equity * MAX_AGGREGATE_POSITION_EQUITY_PCT
/// * leverage.max(1.0)` -- see `clamp_aggregate_position_qty`'s doc comment
/// for why this must scale with leverage rather than being a flat equity
/// cap. Returns legs unchanged otherwise. Never returns negative or NaN
/// quantities.
pub fn scale_paired_legs_to_cap(
    qty_a: f64,
    price_a: f64,
    qty_b: f64,
    price_b: f64,
    equity: f64,
    leverage: f64,
) -> (f64, f64) {
    let qty_a = if qty_a.is_finite() && qty_a > 0.0 { qty_a } else { 0.0 };
    let qty_b = if qty_b.is_finite() && qty_b > 0.0 { qty_b } else { 0.0 };
    let price_a = if price_a.is_finite() && price_a > 0.0 { price_a } else { 0.0 };
    let price_b = if price_b.is_finite() && price_b > 0.0 { price_b } else { 0.0 };
    let leverage = leverage.max(1.0);
    let combined_notional = qty_a * price_a + qty_b * price_b;
    let max_combined_notional = (equity * MAX_AGGREGATE_POSITION_EQUITY_PCT * leverage).max(0.0);
    if combined_notional <= max_combined_notional || combined_notional <= 0.0 {
        return (qty_a, qty_b);
    }
    let scale = (max_combined_notional / combined_notional).max(0.0);
    (qty_a * scale, qty_b * scale)
}

/// Adjust a theoretical liquidation trigger price by an adverse-slippage
/// penalty, matching how real exchanges execute liquidations worse than the
/// exact trigger price (and charge a liquidation fee on top).
pub fn apply_liquidation_penalty(theoretical_liq_price: f64, side: PositionSide, penalty_bps: f64) -> f64 {
    let factor = penalty_bps / 10_000.0;
    match side {
        // Closing a long liquidation is a sell -- executes worse (lower).
        PositionSide::Long => theoretical_liq_price * (1.0 - factor),
        // Closing a short liquidation is a buy-to-cover -- executes worse (higher).
        PositionSide::Short => theoretical_liq_price * (1.0 + factor),
    }
}

/// Cash returned to free balance when a leveraged position is liquidated,
/// floored at `0.0` so a single liquidated position can never drag the rest
/// of the portfolio's cash negative -- isolated margin's core guarantee.
pub fn liquidation_cash_return(margin_posted: f64, realized_pnl: f64) -> f64 {
    (margin_posted + realized_pnl).max(0.0)
}

/// Volatility-targeting position-size overlay config (opt-in; `None`
/// anywhere in the pipeline is fully behavior-preserving). Scales whatever
/// `position_size_pct` a strategy/config would otherwise use, inversely to
/// the asset's own trailing realized volatility, so the position's expected
/// annualized volatility (not just its notional) stays roughly constant
/// across calm and turbulent periods -- see Moreira & Muir 2017
/// ("volatility-managed portfolios") and Barroso & Santa-Clara 2015 for the
/// momentum-specific case. This is a risk overlay, not a new signal: it
/// layers on top of ANY existing entry/exit logic without touching it.
#[derive(Debug, Clone, Copy)]
pub struct VolTargetConfig {
    /// Desired annualized volatility of the sized position, e.g. `0.15` for
    /// a 15% annualized target.
    pub target_annual_vol: f64,
    /// Number of trailing bars' returns used to estimate realized volatility.
    pub lookback_bars: usize,
    /// Bars per year for this candle granularity, used to annualize the
    /// trailing per-bar realized volatility (e.g. `365.0` for daily bars,
    /// `365.0 * 24.0` for hourly). Getting this wrong doesn't crash
    /// anything -- it consistently biases the scale factor in one
    /// direction rather than producing a nonsensical result.
    pub bars_per_year: f64,
    /// Floor on the resulting multiplier -- prevents an unusually calm
    /// trailing window from levering the position size up unboundedly.
    pub min_scale: f64,
    /// Ceiling on the resulting multiplier -- prevents an unusually quiet
    /// trailing window from oversizing the next position.
    pub max_scale: f64,
}

/// Rolling, strictly-causal realized-volatility series over `closes`: entry
/// `i` uses only returns computed from bars strictly before `i`, matching
/// how a real strategy only knows trailing history (never the current
/// bar's own close) when it decides that bar's position size. Returns
/// `None` until enough trailing history exists, and whenever the trailing
/// return window is degenerate (fewer than 2 usable returns).
pub fn trailing_realized_vol_series(closes: &[f64], lookback: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if lookback < 2 || n < lookback + 2 {
        return out;
    }
    // returns[k] = simple return from closes[k] to closes[k+1] -- "known"
    // only once bar k+1 has printed.
    let returns: Vec<f64> = closes
        .windows(2)
        .map(|w| if w[0] > 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
        .collect();

    for i in (lookback + 1)..n {
        let end = i - 1; // exclusive -- last known return index is end-1
        let start = end - lookback;
        let window = &returns[start..end];
        if window.len() < 2 {
            continue;
        }
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (window.len() as f64 - 1.0);
        out[i] = Some(variance.sqrt());
    }
    out
}

/// Scale `base_position_size_pct` inversely to trailing realized volatility
/// so the position's expected annualized volatility sits near
/// `cfg.target_annual_vol`. Falls back to the unscaled
/// `base_position_size_pct` whenever the trailing vol reading is
/// missing/degenerate -- never blocks a trade for lack of a vol estimate,
/// it just leaves sizing at the strategy's own stated default for that bar.
pub fn volatility_scaled_position_size_pct(
    base_position_size_pct: f64,
    trailing_realized_vol_per_bar: Option<f64>,
    cfg: &VolTargetConfig,
) -> f64 {
    let per_bar = match trailing_realized_vol_per_bar {
        Some(v) if v > 0.0 && v.is_finite() => v,
        _ => return base_position_size_pct,
    };
    if cfg.target_annual_vol <= 0.0 || cfg.bars_per_year <= 0.0 {
        return base_position_size_pct;
    }
    let realized_annual_vol = per_bar * cfg.bars_per_year.sqrt();
    if !realized_annual_vol.is_finite() || realized_annual_vol <= 0.0 {
        return base_position_size_pct;
    }
    let raw_scale = cfg.target_annual_vol / realized_annual_vol;
    let min_scale = cfg.min_scale.max(0.0);
    let max_scale = cfg.max_scale.max(min_scale);
    let scale = raw_scale.clamp(min_scale, max_scale);
    (base_position_size_pct * scale).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // -- initial_margin --------------------------------------------------

    #[test]
    fn initial_margin_divides_notional_by_leverage() {
        assert!((initial_margin(1000.0, 4.0) - 250.0).abs() < EPS);
    }

    #[test]
    fn initial_margin_clamps_leverage_floor_at_one() {
        // Leverage < 1.0 must not act as a discount -- no free lunch below 1x.
        assert!((initial_margin(1000.0, 0.5) - 1000.0).abs() < EPS);
        assert!((initial_margin(1000.0, 1.0) - 1000.0).abs() < EPS);
    }

    // -- maintenance_margin -----------------------------------------------

    #[test]
    fn maintenance_margin_is_flat_ratio_of_notional() {
        assert!((maintenance_margin(10_000.0, 0.005) - 50.0).abs() < EPS);
    }

    // -- liquidation_price --------------------------------------------------

    #[test]
    fn liquidation_price_leverage_one_never_triggers() {
        assert_eq!(liquidation_price(100.0, PositionSide::Long, 1.0, 0.005), 0.0);
        assert_eq!(liquidation_price(100.0, PositionSide::Short, 1.0, 0.005), f64::INFINITY);
        // Also below 1.0 (clamped the same way).
        assert_eq!(liquidation_price(100.0, PositionSide::Long, 0.5, 0.005), 0.0);
        assert_eq!(liquidation_price(100.0, PositionSide::Short, 0.5, 0.005), f64::INFINITY);
    }

    #[test]
    fn liquidation_price_long_matches_hand_calculation() {
        // 2x leverage, zero maintenance margin: a long liquidates exactly
        // when it has lost 1/leverage = 50% of entry price -- the point at
        // which the posted margin is fully wiped out.
        let liq = liquidation_price(100.0, PositionSide::Long, 2.0, 0.0);
        assert!((liq - 50.0).abs() < EPS, "liq={liq}");
    }

    #[test]
    fn liquidation_price_short_matches_hand_calculation() {
        // 2x leverage, zero maintenance margin: a short liquidates exactly
        // when price has risen 1/leverage = 50% above entry.
        let liq = liquidation_price(100.0, PositionSide::Short, 2.0, 0.0);
        assert!((liq - 150.0).abs() < EPS, "liq={liq}");
    }

    #[test]
    fn liquidation_price_maintenance_margin_moves_trigger_closer_to_entry() {
        // A nonzero maintenance margin requirement means the position gets
        // liquidated *before* margin is fully wiped out -- long liq price
        // should be higher (closer to entry) than the zero-mmr case, and
        // short liq price should be lower (closer to entry).
        let long_zero_mmr = liquidation_price(100.0, PositionSide::Long, 2.0, 0.0);
        let long_with_mmr = liquidation_price(100.0, PositionSide::Long, 2.0, 0.02);
        assert!(long_with_mmr > long_zero_mmr);

        let short_zero_mmr = liquidation_price(100.0, PositionSide::Short, 2.0, 0.0);
        let short_with_mmr = liquidation_price(100.0, PositionSide::Short, 2.0, 0.02);
        assert!(short_with_mmr < short_zero_mmr);
    }

    #[test]
    fn liquidation_price_higher_leverage_triggers_closer_to_entry() {
        let liq_5x = liquidation_price(100.0, PositionSide::Long, 5.0, 0.005);
        let liq_20x = liquidation_price(100.0, PositionSide::Long, 20.0, 0.005);
        // Higher leverage => smaller adverse move needed => higher (closer
        // to entry) liquidation price for a long.
        assert!(liq_20x > liq_5x);
    }

    // -- is_liquidated --------------------------------------------------

    #[test]
    fn is_liquidated_long_triggers_at_or_below_liq_price() {
        let liq = liquidation_price(100.0, PositionSide::Long, 2.0, 0.0); // 50.0
        assert!(is_liquidated(liq, 100.0, PositionSide::Long, 2.0, 0.0));
        assert!(is_liquidated(liq - 0.01, 100.0, PositionSide::Long, 2.0, 0.0));
        assert!(!is_liquidated(liq + 0.01, 100.0, PositionSide::Long, 2.0, 0.0));
    }

    #[test]
    fn is_liquidated_short_triggers_at_or_above_liq_price() {
        let liq = liquidation_price(100.0, PositionSide::Short, 2.0, 0.0); // 150.0
        assert!(is_liquidated(liq, 100.0, PositionSide::Short, 2.0, 0.0));
        assert!(is_liquidated(liq + 0.01, 100.0, PositionSide::Short, 2.0, 0.0));
        assert!(!is_liquidated(liq - 0.01, 100.0, PositionSide::Short, 2.0, 0.0));
    }

    #[test]
    fn is_liquidated_leverage_one_never_triggers_regardless_of_price() {
        assert!(!is_liquidated(0.01, 100.0, PositionSide::Long, 1.0, 0.005));
        assert!(!is_liquidated(1_000_000.0, 100.0, PositionSide::Short, 1.0, 0.005));
    }

    // -- size_leveraged_order --------------------------------------------------

    #[test]
    fn size_leveraged_order_leverage_one_matches_legacy_sizing_formula() {
        // The critical property: at leverage=1.0, margin_posted (what's
        // actually debited from cash) must equal exactly what the
        // pre-leverage code computed as `size_value = equity * position_size_pct`.
        let (notional, margin_posted, qty) = size_leveraged_order(10_000.0, 0.1, 1.0, 50.0);
        assert!((notional - 1_000.0).abs() < EPS);
        assert!((margin_posted - 1_000.0).abs() < EPS);
        assert!((qty - 20.0).abs() < EPS);
    }

    #[test]
    fn size_leveraged_order_scales_notional_not_margin() {
        // At leverage=4x: notional quadruples but margin_posted (the actual
        // cash debit) stays identical to the unleveraged case -- leverage
        // buys more exposure per dollar of margin, it doesn't change what's
        // debited for the same position_size_pct.
        let (notional_1x, margin_1x, _) = size_leveraged_order(10_000.0, 0.1, 1.0, 50.0);
        let (notional_4x, margin_4x, qty_4x) = size_leveraged_order(10_000.0, 0.1, 4.0, 50.0);
        assert!((notional_4x - notional_1x * 4.0).abs() < EPS);
        assert!((margin_4x - margin_1x).abs() < EPS);
        assert!((qty_4x - notional_4x / 50.0).abs() < EPS);
    }

    #[test]
    fn size_leveraged_order_below_one_clamps_to_one() {
        let (notional_half, _, _) = size_leveraged_order(10_000.0, 0.1, 0.5, 50.0);
        let (notional_one, _, _) = size_leveraged_order(10_000.0, 0.1, 1.0, 50.0);
        assert!((notional_half - notional_one).abs() < EPS);
    }

    #[test]
    fn size_leveraged_order_zero_price_returns_zero_qty_no_panic() {
        let (_, _, qty) = size_leveraged_order(10_000.0, 0.1, 2.0, 0.0);
        assert_eq!(qty, 0.0);
    }

    // -- clamp_aggregate_position_qty --------------------------------------------------

    #[test]
    fn clamp_aggregate_position_qty_passes_through_when_within_headroom() {
        // 10,000 equity, 0 existing exposure, proposing 50 units at $10 =
        // $500 notional -- well under the $10,000 cap, no clamping.
        let qty = clamp_aggregate_position_qty(50.0, 10.0, 0.0, 10_000.0, 1.0);
        assert!((qty - 50.0).abs() < EPS);
    }

    #[test]
    fn clamp_aggregate_position_qty_reduces_when_exceeding_combined_cap() {
        // 10,000 equity, already holding $9,000 notional, proposing another
        // 200 units at $10 = $2,000 more (would total $11,000, over cap) --
        // should clamp to exactly the remaining $1,000 of headroom = 100 units.
        let qty = clamp_aggregate_position_qty(200.0, 10.0, 9_000.0, 10_000.0, 1.0);
        assert!((qty - 100.0).abs() < EPS);
    }

    #[test]
    fn clamp_aggregate_position_qty_returns_zero_when_already_at_cap() {
        let qty = clamp_aggregate_position_qty(50.0, 10.0, 10_000.0, 10_000.0, 1.0);
        assert_eq!(qty, 0.0);
    }

    #[test]
    fn clamp_aggregate_position_qty_this_is_the_regression_case_from_the_ada_xrp_incident() {
        // Reproduces the shape of job 50ac25dd: existing pyramided exposure
        // already near the account's equity, a new order that alone would be
        // fine in isolation but pushes combined exposure past 100% of
        // capital -- must be clamped, not passed through unchanged.
        let equity = 10_000.0;
        let existing_notional = 8_500.0; // already heavily pyramided
        let proposed_qty = 500.0;
        let fill_price = 10.0; // proposed notional = $5,000, would total $13,500
        let qty = clamp_aggregate_position_qty(proposed_qty, fill_price, existing_notional, equity, 1.0);
        let resulting_total = existing_notional + qty * fill_price;
        assert!(resulting_total <= equity + EPS, "combined notional {resulting_total} must not exceed equity {equity}");
        assert!(qty < proposed_qty);
    }

    #[test]
    fn clamp_aggregate_position_qty_zero_or_negative_inputs_return_zero_no_panic() {
        assert_eq!(clamp_aggregate_position_qty(0.0, 10.0, 0.0, 10_000.0, 1.0), 0.0);
        assert_eq!(clamp_aggregate_position_qty(-5.0, 10.0, 0.0, 10_000.0, 1.0), 0.0);
        assert_eq!(clamp_aggregate_position_qty(50.0, 0.0, 0.0, 10_000.0, 1.0), 0.0);
        assert_eq!(clamp_aggregate_position_qty(f64::NAN, 10.0, 0.0, 10_000.0, 1.0), 0.0);
    }

    #[test]
    fn clamp_aggregate_position_qty_scales_headroom_with_leverage_not_just_equity() {
        // This is the regression this fix itself introduced and then had to
        // correct: a flat (leverage-blind) equity cap clipped legitimate 4x
        // leveraged notional back down to the 1x case, breaking
        // `leverage_scales_pnl_proportionally_to_notional` in
        // `backtest::pair_simulation`. At 4x leverage, an order whose notional
        // is 4x equity is exactly at the cap (equity * 1.0 * 4.0), not over it.
        let equity = 10_000.0;
        let fill_price = 10.0;
        let proposed_qty = 4_000.0; // notional = $40,000 = equity * 4x leverage
        let qty_1x = clamp_aggregate_position_qty(proposed_qty, fill_price, 0.0, equity, 1.0);
        let qty_4x = clamp_aggregate_position_qty(proposed_qty, fill_price, 0.0, equity, 4.0);
        assert!(qty_1x < proposed_qty, "1x leverage must still clamp a 4x-equity-sized order");
        assert!((qty_4x - proposed_qty).abs() < EPS, "4x leverage must allow the full 4x-equity notional through unclamped");
    }

    // -- scale_paired_legs_to_cap --------------------------------------------------

    #[test]
    fn scale_paired_legs_to_cap_passes_through_when_within_headroom() {
        // qty_a=10 @ $100 = $1,000; qty_b (hedge_ratio=1.8) = 18 @ $50 = $900.
        // Combined $1,900, well under $10,000 equity -- no scaling.
        let (qty_a, qty_b) = scale_paired_legs_to_cap(10.0, 100.0, 18.0, 50.0, 10_000.0, 1.0);
        assert!((qty_a - 10.0).abs() < EPS);
        assert!((qty_b - 18.0).abs() < EPS);
    }

    #[test]
    fn scale_paired_legs_to_cap_this_is_the_regression_case_from_the_ada_xrp_incident() {
        // Reproduces the shape of job 50ac25dd: leg A sized reasonably on its
        // own, but leg B scaled up by a hedge ratio > 1 pushes combined
        // notional over equity. Must scale BOTH legs down together, preserving
        // their ratio, so combined notional lands at or under the cap.
        let equity = 10_000.0;
        let qty_a = 60.0;
        let price_a = 100.0; // leg A notional = $6,000
        let hedge_ratio = 1.83_f64;
        let qty_b = hedge_ratio * qty_a; // 109.8
        let price_b = 40.0; // leg B notional = $4,392 -> combined $10,392, over cap

        let (scaled_a, scaled_b) = scale_paired_legs_to_cap(qty_a, price_a, qty_b, price_b, equity, 1.0);
        let combined = scaled_a * price_a + scaled_b * price_b;
        assert!(combined <= equity + EPS, "combined notional {combined} must not exceed equity {equity}");
        // Hedge ratio (relative sizing between legs) must be preserved.
        assert!((scaled_b / scaled_a - hedge_ratio).abs() < 1e-6);
        assert!(scaled_a < qty_a);
    }

    #[test]
    fn scale_paired_legs_to_cap_scales_headroom_with_leverage() {
        // Same combined notional as the ADA/XRP regression case ($10,392,
        // over the 1x cap of $10,000) but at 4x leverage the cap is $40,000 --
        // well within headroom, so both legs must pass through unscaled.
        let equity = 10_000.0;
        let qty_a = 60.0;
        let price_a = 100.0;
        let hedge_ratio = 1.83_f64;
        let qty_b = hedge_ratio * qty_a;
        let price_b = 40.0;
        let (scaled_a, scaled_b) = scale_paired_legs_to_cap(qty_a, price_a, qty_b, price_b, equity, 4.0);
        assert!((scaled_a - qty_a).abs() < EPS);
        assert!((scaled_b - qty_b).abs() < EPS);
    }

    #[test]
    fn scale_paired_legs_to_cap_zero_or_invalid_inputs_return_zero_no_panic() {
        let (a, b) = scale_paired_legs_to_cap(0.0, 100.0, 0.0, 50.0, 10_000.0, 1.0);
        assert_eq!(a, 0.0);
        assert_eq!(b, 0.0);
        let (a, b) = scale_paired_legs_to_cap(f64::NAN, 100.0, 10.0, 50.0, 10_000.0, 1.0);
        assert_eq!(a, 0.0);
        assert!(b >= 0.0);
        let (a, b) = scale_paired_legs_to_cap(10.0, 100.0, 10.0, 50.0, 0.0, 1.0);
        assert_eq!(a, 0.0);
        assert_eq!(b, 0.0);
    }

    // -- apply_liquidation_penalty --------------------------------------------------

    #[test]
    fn apply_liquidation_penalty_worsens_long_exit_price() {
        let adjusted = apply_liquidation_penalty(100.0, PositionSide::Long, 50.0);
        assert!(adjusted < 100.0);
        assert!((adjusted - 99.5).abs() < EPS);
    }

    #[test]
    fn apply_liquidation_penalty_worsens_short_exit_price() {
        let adjusted = apply_liquidation_penalty(100.0, PositionSide::Short, 50.0);
        assert!(adjusted > 100.0);
        assert!((adjusted - 100.5).abs() < EPS);
    }

    #[test]
    fn apply_liquidation_penalty_zero_bps_is_noop() {
        assert_eq!(apply_liquidation_penalty(100.0, PositionSide::Long, 0.0), 100.0);
        assert_eq!(apply_liquidation_penalty(100.0, PositionSide::Short, 0.0), 100.0);
    }

    // -- liquidation_cash_return --------------------------------------------------

    #[test]
    fn liquidation_cash_return_normal_case() {
        assert!((liquidation_cash_return(100.0, -30.0) - 70.0).abs() < EPS);
    }

    #[test]
    fn liquidation_cash_return_floors_at_zero_never_goes_negative() {
        // Loss exceeds posted margin (e.g. a large penalty-adjusted slippage
        // gap) -- isolated margin means the account never owes more than
        // what was posted for this specific position.
        assert_eq!(liquidation_cash_return(100.0, -150.0), 0.0);
    }

    #[test]
    fn liquidation_cash_return_profit_case() {
        assert!((liquidation_cash_return(100.0, 25.0) - 125.0).abs() < EPS);
    }

    // -- trailing_realized_vol_series --------------------------------------------------

    #[test]
    fn trailing_realized_vol_series_none_before_enough_history() {
        let closes = vec![100.0, 101.0, 102.0, 103.0];
        let series = trailing_realized_vol_series(&closes, 5);
        assert!(series.iter().all(|v| v.is_none()));
    }

    #[test]
    fn trailing_realized_vol_series_is_causal_never_uses_current_bar() {
        // Constant returns until a single huge move at the very last bar --
        // if the estimate at the last bar's OWN index used that bar's
        // return, it would spike; causal correctness means it must not.
        let mut closes = vec![100.0];
        for _ in 0..10 {
            let last = *closes.last().unwrap();
            closes.push(last * 1.01);
        }
        let steady_vol_at_second_to_last = trailing_realized_vol_series(&closes, 3);
        // Now append one huge outlier return and recompute the series up to
        // (not including) that outlier bar's own index -- must be identical
        // to the version without the outlier appended, since a causal
        // series at index i never looks at closes[i] itself... but the
        // outlier IS closes[i], so the entry AT the outlier's own index
        // must match what it would have been using only prior history.
        let mut closes_with_outlier = closes.clone();
        closes_with_outlier.push(closes.last().unwrap() * 5.0);
        let series_with_outlier = trailing_realized_vol_series(&closes_with_outlier, 3);
        let last_idx = closes.len() - 1;
        assert_eq!(
            series_with_outlier[last_idx].map(|v| (v * 1e9).round()),
            steady_vol_at_second_to_last[last_idx].map(|v| (v * 1e9).round()),
        );
    }

    #[test]
    fn trailing_realized_vol_series_positive_for_varying_returns() {
        let closes = vec![100.0, 102.0, 99.0, 103.0, 98.0, 105.0, 97.0, 104.0];
        let series = trailing_realized_vol_series(&closes, 3);
        let any_some = series.iter().any(|v| v.is_some());
        assert!(any_some);
        for v in series.iter().flatten() {
            assert!(*v > 0.0);
            assert!(v.is_finite());
        }
    }

    // -- volatility_scaled_position_size_pct --------------------------------------------------

    fn default_vol_cfg() -> VolTargetConfig {
        VolTargetConfig {
            target_annual_vol: 0.20,
            lookback_bars: 20,
            bars_per_year: 365.0,
            min_scale: 0.25,
            max_scale: 3.0,
        }
    }

    #[test]
    fn vol_scaled_sizing_falls_back_when_no_vol_reading() {
        let cfg = default_vol_cfg();
        let scaled = volatility_scaled_position_size_pct(0.1, None, &cfg);
        assert_eq!(scaled, 0.1);
    }

    #[test]
    fn vol_scaled_sizing_scales_up_in_calm_markets() {
        let cfg = default_vol_cfg();
        // Realized annual vol = 0.005 * sqrt(365) ~= 0.0955 -- well below the
        // 0.20 target, so the scale factor should be > 1 (up to the cap).
        let scaled = volatility_scaled_position_size_pct(0.1, Some(0.005), &cfg);
        assert!(scaled > 0.1);
        assert!(scaled <= 0.1 * cfg.max_scale + EPS);
    }

    #[test]
    fn vol_scaled_sizing_scales_down_in_turbulent_markets() {
        let cfg = default_vol_cfg();
        // Realized annual vol = 0.05 * sqrt(365) ~= 0.955 -- far above the
        // 0.20 target, so the scale factor should be well below 1.
        let scaled = volatility_scaled_position_size_pct(0.1, Some(0.05), &cfg);
        assert!(scaled < 0.1);
        assert!(scaled >= 0.1 * cfg.min_scale - EPS);
    }

    #[test]
    fn vol_scaled_sizing_respects_min_and_max_scale_clamps() {
        let mut cfg = default_vol_cfg();
        cfg.min_scale = 0.5;
        cfg.max_scale = 2.0;
        let scaled_up = volatility_scaled_position_size_pct(0.1, Some(1e-6), &cfg);
        assert!((scaled_up - 0.1 * 2.0).abs() < 1e-6);
        let scaled_down = volatility_scaled_position_size_pct(0.1, Some(10.0), &cfg);
        assert!((scaled_down - 0.1 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn vol_scaled_sizing_disabled_target_falls_back() {
        let mut cfg = default_vol_cfg();
        cfg.target_annual_vol = 0.0;
        let scaled = volatility_scaled_position_size_pct(0.1, Some(0.02), &cfg);
        assert_eq!(scaled, 0.1);
    }
}
