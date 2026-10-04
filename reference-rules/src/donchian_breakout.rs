//! Donchian channel breakout sleeve: single asset, daily, OHLC (the first primitive in this crate that needs
//! high/low, not just close -- see [`HlcSeries`] below).
//!
//! # Rule
//! Parameter: channel length `N` ([`DONCHIAN_DEFAULT_N`] = 20). For decision day `t`, the upper channel is
//! `max(high)` and the lower channel is `min(low)` over the `N` bars strictly PRECEDING today -- bars
//! `[t-N, t-1]`, excluding today's own high/low (no lookahead into the bar being decided). Weight 1.0 if today's
//! close is strictly above the upper channel (breakout up); weight 0.0 if today's close is strictly below the
//! lower channel (breakout down); otherwise HOLD -- the weight stays whatever it was on the previous decision.
//! This is NOT a pure function of today's bar alone: on a non-breakout day the result depends on the prior
//! decision, exactly like the pairs primitives' stateful hold (see `weightsim::stateful`'s doc comment: "the
//! simulator owns the state ... a rule never stores anything between decisions itself"). Fewer than `N+1` bars of
//! history (today plus its full preceding window) is a SILENT SKIP: flat (weight 0.0), not a refusal.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *One channel per decision day.* The spec's own wording describes "today's close vs. yesterday's upper
//!    channel value" and "upper channel = max(high over bars `[t-N, t-1]`)" as the SAME quantity for decision day
//!    `t`: there is exactly one channel computed from the `N` bars strictly before `t`, known causally as of
//!    yesterday's close. This file computes one channel per `t`, never two.
//! 2. *Exact arithmetic is NOT needed here, unlike `compare_to_mean` (`exact.rs`).* `compare_to_mean` exists
//!    because computing an arithmetic MEAN in binary floating point (`sum / n`) can round a true tie into a false
//!    "above" or "below" -- the mean itself is a new value that didn't exist in the input. A channel bound here is
//!    a MAX or MIN of the window: `f64::max`/`f64::min` (via `Iterator::fold`) always return one of the window's
//!    OWN input values, bit-for-bit, with no arithmetic performed on it at all. Comparing `close > upper` is then a
//!    plain IEEE-754 comparison of two values neither of which was computed by summation or division; there is no
//!    rounding step that could manufacture a false tie-break. Plain `f64` comparison is therefore exact for this
//!    primitive, and the `exact` module (private to this crate, mean-comparison only) is not used here.
//! 3. *Function shape: whole series in, whole decision vector out*, the same shape `turn_of_month` and
//!    `day_of_week` use (not a per-bar function taking an explicit previous-weight argument). Chosen because the
//!    hold-chain state (today's weight depends on yesterday's weight on a non-breakout day) is naturally threaded
//!    as a single local variable while scanning the series forward once; a per-bar signature would just move that
//!    same accumulator into the caller with no behavioral difference, at the cost of every caller having to get
//!    the seed value right. See [`decide_donchian_breakout`].
//! 4. *"Flat = 0.0" is both the pre-history skip value AND the hold-chain's seed -- not a separate marker.* Before
//!    the first possible decision (bars `0..N`, fewer than `N+1` bars visible), the function reports weight 0.0.
//!    From bar `N` onward the hold-chain carries forward whatever the last decision was, starting from that same
//!    0.0. A separate "undecided" marker (e.g. `Option<f64>` or `NaN`) was considered and rejected: operationally
//!    there is no behavioral difference between "we have no opinion yet" and "we decided to be flat" -- both mean
//!    zero capital allocated on that bar, and both the bar immediately after the skip region AND every pre-history
//!    bar need the hold-chain to start from exactly the same value for the rule to have one consistent definition
//!    of "the weight before the first decision". Reusing the crate's existing flat convention (0.0, as
//!    `day_of_week`/`turn_of_month` already use for "not applicable") is the simpler, already-established choice.
//! 5. *Series type: minimal HLC, not a general OHLC type.* The spec's own decision rule (window of highs/lows,
//!    comparison against today's close) never reads an "open" value anywhere; building a 4-field OHLC type whose
//!    `open` is accepted, validated, and then never consulted again would be ceremony with no caller benefit (and
//!    an unused-field lint risk). [`HlcSeries`] therefore carries only high/low/close -- the minimal fields this
//!    primitive actually needs -- named accordingly rather than "Ohlc" so the type's own name does not overclaim
//!    what it stores. If a future primitive in this crate needs `open` too, a true OHLC type can be added then;
//!    this file does not block that.
//! 6. *Validation discipline mirrors [`crate::series::PriceSeries`]*: non-empty, strictly ascending dates, a valid
//!    symbol, every value finite. Extended for OHLC-specific invariants: `high >= low` (a bar whose high is below
//!    its low is corrupt data, not a rare-but-legal bar) and `low <= close <= high` (a close outside its own
//!    bar's range is corrupt data -- the "sane range" check the task left optional; chosen because it is a real
//!    invariant of any genuine OHLC bar, not a judgment call about typical price behavior). Like `PriceSeries`,
//!    values must be STRICTLY positive (not merely `>= 0`): this crate treats a zero or negative price as invalid
//!    everywhere else, and an asset's high/low/close are no exception.

use chrono::NaiveDate;
use std::fmt;

/// Default Donchian channel length in bars.
pub const DONCHIAN_DEFAULT_N: usize = 20;

/// Why [`HlcSeries::new`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum DonchianError {
    /// Symbol is empty or contains whitespace/control characters.
    InvalidSymbol { symbol: String },
    /// `dates`/`highs`/`lows`/`closes` passed to `HlcSeries::new` differ in length.
    LengthMismatch {
        symbol: String,
        dates: usize,
        highs: usize,
        lows: usize,
        closes: usize,
    },
    /// A series with no bars.
    EmptySeries { symbol: String },
    /// Dates are not strictly ascending (a duplicate date counts as non-monotonic).
    NonMonotonic {
        symbol: String,
        index: usize,
        previous: NaiveDate,
        current: NaiveDate,
    },
    /// A high, low or close is NaN, infinite, zero or negative.
    InvalidValue {
        symbol: String,
        date: NaiveDate,
        field: &'static str,
        value: f64,
    },
    /// `high < low` on the same bar.
    HighBelowLow {
        symbol: String,
        date: NaiveDate,
        high: f64,
        low: f64,
    },
    /// `close` outside its own bar's `[low, high]` range.
    CloseOutsideRange {
        symbol: String,
        date: NaiveDate,
        close: f64,
        low: f64,
        high: f64,
    },
}

impl fmt::Display for DonchianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DonchianError {}

/// Validated daily high/low/close series of one instrument (no `open`: see module docs, choice 5).
/// Invariants (enforced by `new`): at least one bar, dates strictly ascending, every value finite and strictly
/// positive, `high >= low`, `low <= close <= high`, symbol non-empty without whitespace or control characters.
#[derive(Debug, Clone, PartialEq)]
pub struct HlcSeries {
    symbol: String,
    dates: Vec<NaiveDate>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
}

impl HlcSeries {
    pub fn new(
        symbol: impl Into<String>,
        dates: Vec<NaiveDate>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> Result<Self, DonchianError> {
        let symbol = symbol.into();
        if symbol.is_empty() || symbol.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(DonchianError::InvalidSymbol { symbol });
        }
        if dates.len() != highs.len() || dates.len() != lows.len() || dates.len() != closes.len() {
            return Err(DonchianError::LengthMismatch {
                symbol,
                dates: dates.len(),
                highs: highs.len(),
                lows: lows.len(),
                closes: closes.len(),
            });
        }
        if dates.is_empty() {
            return Err(DonchianError::EmptySeries { symbol });
        }
        for i in 0..dates.len() {
            let date = dates[i];
            let (high, low, close) = (highs[i], lows[i], closes[i]);
            for (field, value) in [("high", high), ("low", low), ("close", close)] {
                if !(value.is_finite() && value > 0.0) {
                    return Err(DonchianError::InvalidValue {
                        symbol,
                        date,
                        field,
                        value,
                    });
                }
            }
            if high < low {
                return Err(DonchianError::HighBelowLow {
                    symbol,
                    date,
                    high,
                    low,
                });
            }
            if close < low || close > high {
                return Err(DonchianError::CloseOutsideRange {
                    symbol,
                    date,
                    close,
                    low,
                    high,
                });
            }
            if i > 0 && dates[i] <= dates[i - 1] {
                return Err(DonchianError::NonMonotonic {
                    symbol,
                    index: i,
                    previous: dates[i - 1],
                    current: dates[i],
                });
            }
        }
        Ok(Self {
            symbol,
            dates,
            highs,
            lows,
            closes,
        })
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    pub fn dates(&self) -> &[NaiveDate] {
        &self.dates
    }
    pub fn highs(&self) -> &[f64] {
        &self.highs
    }
    pub fn lows(&self) -> &[f64] {
        &self.lows
    }
    pub fn closes(&self) -> &[f64] {
        &self.closes
    }
    pub fn len(&self) -> usize {
        self.dates.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }
}

/// Decide the whole-series Donchian breakout weight vector for `series` with channel length `n` (use
/// [`DONCHIAN_DEFAULT_N`] for the default).
///
/// For bar `t` with `t < n` (fewer than `n+1` bars visible: today plus a full preceding window), the result is
/// `0.0` -- silent skip, flat (module docs, choice 4). For `t >= n`, the channel is `max(highs[t-n..t])` /
/// `min(lows[t-n..t])` (bars `t-n ..= t-1`, excluding today): weight flips to `1.0` if `closes[t]` is strictly
/// above the upper channel, flips to `0.0` if strictly below the lower channel, otherwise HOLDS the previous bar's
/// weight (which, for the first decision at `t == n`, is the `0.0` flat seed -- see choice 4).
///
/// # Panics
/// Panics if `n == 0` (a zero-length channel is not a meaningful parameter).
pub fn decide_donchian_breakout(series: &HlcSeries, n: usize) -> Vec<f64> {
    assert!(n >= 1, "channel length n must be >= 1");
    let highs = series.highs();
    let lows = series.lows();
    let closes = series.closes();
    let len = series.len();
    let mut weights = vec![0.0; len];
    let mut current = 0.0; // flat seed; see module docs choice 4.
    for t in 0..len {
        if t < n {
            // Fewer than n+1 bars visible (indices 0..=t is only t+1 <= n bars): silent skip, flat.
            weights[t] = 0.0;
            continue;
        }
        let window_start = t - n;
        let upper = highs[window_start..t]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let lower = lows[window_start..t]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let close = closes[t];
        if close > upper {
            current = 1.0;
        } else if close < lower {
            current = 0.0;
        }
        // else: HOLD, `current` unchanged.
        weights[t] = current;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn series(bars: &[(NaiveDate, f64, f64, f64)], // (date, high, low, close)
    ) -> HlcSeries {
        let dates = bars.iter().map(|b| b.0).collect();
        let highs = bars.iter().map(|b| b.1).collect();
        let lows = bars.iter().map(|b| b.2).collect();
        let closes = bars.iter().map(|b| b.3).collect();
        HlcSeries::new("TEST", dates, highs, lows, closes).unwrap()
    }

    #[test]
    fn default_n_is_twenty() {
        assert_eq!(DONCHIAN_DEFAULT_N, 20);
    }

    #[test]
    fn insufficient_history_is_silent_skip_flat() {
        // n = 3 needs n+1 = 4 bars for the first decision; only 3 bars supplied, so every bar is flat.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 10.0, 5.0, 7.0),
        ]);
        let w = decide_donchian_breakout(&s, 3);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn breakout_up_flips_weight_to_one() {
        // n = 2. Bars 0,1 build the window (high 10, low 5 each); bar 2 is the first possible decision
        // (t == n == 2, exactly n+1 = 3 bars visible). Its close (11) is strictly above the window's upper
        // channel (10), so weight flips from the flat 0.0 seed to 1.0.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 12.0, 4.0, 11.0),
        ]);
        let w = decide_donchian_breakout(&s, 2);
        assert_eq!(w, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn breakout_down_flips_weight_to_zero_after_holding_at_one() {
        // n = 2. Bar 2 breaks out up (weight -> 1.0, as in the previous test). Bar 3's window is
        // [bars 1,2] = highs(10,12)->upper 12, lows(5,4)->lower 4. Bar 3's close (3) is strictly below the
        // lower channel (4), so weight flips from the held 1.0 down to 0.0 -- a genuine down-breakout, not
        // merely "stayed at the initial flat seed".
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 12.0, 4.0, 11.0),
            (d(2024, 1, 4), 5.0, 2.0, 3.0),
        ]);
        let w = decide_donchian_breakout(&s, 2);
        assert_eq!(w, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn hold_case_stays_at_one_between_channel_bounds() {
        // Same setup through bar 2 (weight -> 1.0). Bar 3's window is [bars 1,2]: upper 12, lower 4. Bar 3's
        // close (8) is strictly between them, so the rule HOLDS: weight stays 1.0, not reset to 0.0.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 12.0, 4.0, 11.0),
            (d(2024, 1, 4), 10.0, 6.0, 8.0),
        ]);
        let w = decide_donchian_breakout(&s, 2);
        assert_eq!(w, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn hold_case_stays_at_zero_when_never_broken_out() {
        // n = 2. Bar 2's window is [bars 0,1]: upper 10, lower 5. Bar 2's close (8) is strictly between them,
        // so the rule HOLDS at the flat 0.0 seed -- this is a genuine "decided to stay flat" outcome (bar 2 IS
        // a valid decision, t == n), not the pre-history skip of `insufficient_history_is_silent_skip_flat`.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 9.0, 6.0, 8.0),
        ]);
        let w = decide_donchian_breakout(&s, 2);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn window_excludes_the_bar_exactly_n_plus_one_back_off_by_one_boundary() {
        // n = 2. Bar 0 is a huge outlier (high 1000) that is exactly n+1 = 3 bars back from decision day t = 3.
        // Bar 3's CORRECT window is [bars 1,2] (upper = max(10,12) = 12, lower = min(5,6) = 5); bar 0 must be
        // excluded. Bar 3's close (15) is strictly above 12, so the correct result is a breakout up (1.0). If
        // an off-by-one bug widened the window to include bar 0 (upper would become 1000), bar 3's close (15)
        // would NOT clear it and the (wrong) result would stay 0.0 -- this test fails under that bug.
        let s = series(&[
            (d(2024, 1, 1), 1000.0, 999.0, 999.5), // t-n-1 = 0: must be excluded from bar 3's window
            (d(2024, 1, 2), 10.0, 5.0, 7.0),
            (d(2024, 1, 3), 12.0, 6.0, 7.0),
            (d(2024, 1, 4), 16.0, 14.0, 15.0),
        ]);
        let w = decide_donchian_breakout(&s, 2);
        assert_eq!(w[2], 0.0); // bar 2's own decision: window [0,1], upper 1000, lower 5 -> close 7, HOLD at flat.
        assert_eq!(w[3], 1.0); // bar 3: window [1,2] excludes bar 0 -> breakout up.
    }

    #[test]
    fn series_rejects_empty_non_monotonic_and_length_mismatch() {
        assert!(matches!(
            HlcSeries::new("X", vec![], vec![], vec![], vec![]),
            Err(DonchianError::EmptySeries { .. })
        ));
        assert!(matches!(
            HlcSeries::new(
                "X",
                vec![d(2024, 1, 2), d(2024, 1, 1)],
                vec![10.0, 10.0],
                vec![5.0, 5.0],
                vec![7.0, 7.0],
            ),
            Err(DonchianError::NonMonotonic { .. })
        ));
        assert!(matches!(
            HlcSeries::new(
                "X",
                vec![d(2024, 1, 1)],
                vec![10.0, 10.0],
                vec![5.0],
                vec![7.0],
            ),
            Err(DonchianError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn series_rejects_high_below_low_and_close_outside_range() {
        assert!(matches!(
            HlcSeries::new("X", vec![d(2024, 1, 1)], vec![4.0], vec![5.0], vec![4.5]),
            Err(DonchianError::HighBelowLow { .. })
        ));
        assert!(matches!(
            HlcSeries::new("X", vec![d(2024, 1, 1)], vec![10.0], vec![5.0], vec![11.0]),
            Err(DonchianError::CloseOutsideRange { .. })
        ));
        assert!(matches!(
            HlcSeries::new("X", vec![d(2024, 1, 1)], vec![10.0], vec![5.0], vec![4.0]),
            Err(DonchianError::CloseOutsideRange { .. })
        ));
    }

    #[test]
    fn series_rejects_non_positive_and_non_finite_values() {
        assert!(matches!(
            HlcSeries::new("X", vec![d(2024, 1, 1)], vec![0.0], vec![0.0], vec![0.0]),
            Err(DonchianError::InvalidValue { .. })
        ));
        assert!(matches!(
            HlcSeries::new(
                "X",
                vec![d(2024, 1, 1)],
                vec![f64::NAN],
                vec![5.0],
                vec![7.0],
            ),
            Err(DonchianError::InvalidValue { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "channel length n must be >= 1")]
    fn zero_length_channel_panics() {
        let s = series(&[(d(2024, 1, 1), 10.0, 5.0, 7.0)]);
        let _ = decide_donchian_breakout(&s, 0);
    }
}
