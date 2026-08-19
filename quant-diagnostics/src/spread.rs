//! Corwin & Schultz (2012) high-low bid-ask spread estimator: recovers an
//! implied effective spread from OHLC high/low data alone, using
//! consecutive 2-bar windows. No live quote feed needed.
//!
//! Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask
//! Spreads from Daily High and Low Prices." Journal of Finance, 67(2).

/// 3 - 2*sqrt(2), the constant denominator in the Corwin-Schultz alpha term.
const K: f64 = 3.0 - std::f64::consts::SQRT_2 * 2.0;

/// Average Corwin-Schultz spread (fraction of price, e.g. 0.0012 == 12bps)
/// across every consecutive 2-bar window in `bars` (chronological `(high,
/// low)` pairs). `None` when there are fewer than 2 bars, or every window
/// was degenerate (non-positive or inverted high/low) -- "no estimate
/// available", never "spread is zero".
pub fn corwin_schultz_spread(bars: &[(f64, f64)]) -> Option<f64> {
    if bars.len() < 2 {
        return None;
    }
    let mut spreads = Vec::with_capacity(bars.len() - 1);
    for w in bars.windows(2) {
        let (h1, l1) = w[0];
        let (h2, l2) = w[1];
        // Defensive: ln() of non-positive is undefined, and h < l is not a
        // physically valid bar. Skip only this window, not the whole series.
        if h1 <= 0.0 || l1 <= 0.0 || h2 <= 0.0 || l2 <= 0.0 || h1 < l1 || h2 < l2 {
            continue;
        }
        let beta = (h1 / l1).ln().powi(2) + (h2 / l2).ln().powi(2);
        let gamma = (h1.max(h2) / l1.min(l2)).ln().powi(2);
        let alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / K - (gamma / K).sqrt();
        // Known artifact of the estimator: alpha can go negative. Clip to 0
        // PER-WINDOW, before computing that window's spread -- not once on
        // the pooled average (clipping the average would let one very-
        // negative window silently cancel a legitimately positive one).
        let alpha = alpha.max(0.0);
        let e_alpha = alpha.exp();
        spreads.push(2.0 * (e_alpha - 1.0) / (1.0 + e_alpha));
    }
    if spreads.is_empty() {
        return None;
    }
    Some(spreads.iter().sum::<f64>() / spreads.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::corwin_schultz_spread;

    #[test]
    fn empty_bars_yield_none() {
        assert_eq!(corwin_schultz_spread(&[]), None);
    }

    #[test]
    fn single_bar_yields_none() {
        assert_eq!(corwin_schultz_spread(&[(101.0, 99.0)]), None);
    }

    #[test]
    fn known_input_known_output_regression() {
        // Hand-verified against the formula directly (not just "runs
        // without panicking") -- pins the exact arithmetic, not just the
        // qualitative shape.
        let bars = [(101.0, 99.0), (102.0, 98.0)];
        let result = corwin_schultz_spread(&bars).expect("two valid bars produce one window");
        assert!(
            (result - 0.011397607119481453).abs() < 1e-9,
            "expected ~0.0113976, got {result}"
        );
    }

    #[test]
    fn degenerate_zero_range_bar_does_not_panic_or_nan() {
        // H == L on both bars -- beta=0, gamma=0, alpha clipped to 0 ->
        // spread 0. Proves no special-case guard is needed for this case.
        let bars = [(100.0, 100.0), (100.0, 100.0)];
        let result = corwin_schultz_spread(&bars).expect("degenerate but not invalid");
        assert!(result.is_finite());
        assert!((result - 0.0).abs() < 1e-12);
    }

    #[test]
    fn non_positive_or_inverted_bar_is_skipped_not_fatal() {
        // Middle bar is invalid (negative high/low) -- windows touching it
        // are skipped, but the series as a whole still yields an estimate
        // from the remaining valid window, not None.
        let bars = [(100.0, 99.0), (101.0, 99.5), (-1.0, -2.0), (103.0, 100.0)];
        assert!(corwin_schultz_spread(&bars).is_some());
    }

    #[test]
    fn calm_series_produces_smaller_spread_than_volatile_series() {
        // The actual behavioral claim this estimator exists to deliver:
        // a tight-range (calm/liquid) series should read as materially
        // cheaper to trade than a wide-range (volatile/thin) one.
        let calm = [(100.5, 99.5), (100.6, 99.6), (100.4, 99.4)];
        let volatile = [(105.0, 95.0), (107.0, 93.0), (104.0, 96.0)];
        let calm_spread = corwin_schultz_spread(&calm).expect("valid calm series");
        let volatile_spread = corwin_schultz_spread(&volatile).expect("valid volatile series");
        assert!(
            calm_spread < volatile_spread,
            "expected calm ({calm_spread}) < volatile ({volatile_spread})"
        );
    }
}
