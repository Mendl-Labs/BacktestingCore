//! Volume-confirmed Donchian breakout sleeve: single asset, daily OHLCV. Same price-channel breakout logic as
//! [`crate::donchian_breakout`], gated by a volume-confirmation condition -- see that module's doc comment first,
//! this one only documents what is DIFFERENT.
//!
//! # Rule
//! Two parameters: a price-channel length ([`VCB_DEFAULT_PRICE_N`] = 20) and a trailing-volume-average window
//! length ([`VCB_DEFAULT_VOLUME_N`] = 20) -- see choice 1 below for why these are two independent parameters
//! rather than one shared one, even though their defaults happen to coincide. For decision day `t`: the price
//! channel is computed exactly as `donchian_breakout` computes it, over the `price_n` bars strictly preceding
//! today (`[t-price_n, t-1]`); the trailing volume average is the MEAN (not sum) of `volumes[t-volume_n..t]`,
//! the `volume_n` bars strictly preceding today, excluding today's own volume -- the same "no lookahead into the
//! bar being decided" discipline as the price channel, just applied to a different column. A breakout signal
//! actually FIRES (weight flips to `1.0` or `0.0`) only if BOTH: (a) today's close clears the price channel
//! (strictly above the upper bound, or strictly below the lower bound), AND (b) today's volume strictly exceeds
//! [`VCB_VOLUME_MULTIPLIER`] (1.5) times the trailing volume average. If the price condition holds but the volume
//! condition does not (or vice versa, or neither holds), the weight HOLDS -- stays whatever it was on the
//! previous decision, the same stateful hold-chain convention as `donchian_breakout`, seeded at the same flat
//! `0.0`. Volume alone never flips anything: a volume spike with no price breakout is always a hold, because the
//! `AND` in the previous sentence requires the price condition regardless of how large the volume condition's
//! margin is.
//!
//! Insufficient history is governed by WHICHEVER of the two windows needs more bars: the function cannot make its
//! first real decision until BOTH the price channel and the volume average have their full window, i.e. not
//! before bar index `max(price_n, volume_n)` (today plus a full preceding window of the LARGER requirement). Every
//! bar before that is a silent skip: flat (weight `0.0`), exactly like `donchian_breakout`'s pre-history
//! convention, just with `n` replaced by `max(price_n, volume_n)`.
//!
//! # Interpretation choices a reviewer must confirm
//! 1. *Two independent lookback parameters, not one shared parameter.* The spec names the price channel's window
//!    as `[t-N, t-1]` and the volume average's window as `[t-20, t-1]` in separate sentences, and both default to
//!    20 only because the task's own worked example happens to pick the same number for both -- nothing in the
//!    rule's definition requires them to move together. A caller may reasonably want a fast 10-bar volume filter
//!    layered on a slow 50-bar price channel (or the reverse), and collapsing them into one parameter would make
//!    that an impossible request without duplicating this entire file. [`decide_volume_confirmed_breakout`]
//!    therefore takes `price_n` and `volume_n` as two separate arguments; [`VCB_DEFAULT_PRICE_N`] and
//!    [`VCB_DEFAULT_VOLUME_N`] are two separate constants (not one shared constant reused twice), even though both
//!    equal 20 today, so that a future change to one default is not a textual no-op that silently also changes
//!    the other.
//! 2. *Insufficient-history threshold is `max(price_n, volume_n)`, not `min` and not a sum.* Both windows must be
//!    FULLY available before the rule can evaluate its `AND` condition at all (a partial volume average or a
//!    partial price channel is not "the rule decided to hold", it is "the rule had no basis to decide"); using
//!    `min` would let the function read past the start of whichever vector backs the larger window (the
//!    donchian-style slice `t-n..t` would underflow `usize` subtraction, or silently read a too-short window), and
//!    there is no reason to add the two windows together since they look at the same calendar days, just two
//!    different columns of the same bars. See `insufficient_history_uses_max_not_min_of_the_two_windows` below for
//!    a test that only passes under `max`.
//! 3. *Series type: `OhlcvSeries` wraps [`HlcSeries`] plus an independent `volumes: Vec<f64>`, rather than
//!    redeclaring high/low/close fields.* `donchian_breakout::HlcSeries` already owns the exact validation this
//!    primitive needs for price (non-empty, strictly ascending dates, valid symbol, every value finite, strictly
//!    positive, `high >= low`, `low <= close <= high`); duplicating that logic field-for-field here would be a
//!    second copy of the same invariants to keep in sync. Composing (`OhlcvSeries { hlc: HlcSeries, volumes:
//!    Vec<f64> }`) reuses that validation by construction and adds exactly the one new invariant this primitive
//!    needs: `volumes` must be the same length as the price series, and every volume finite and `>= 0.0`.
//! 4. *Volume validation is `>= 0.0`, not `> 0.0` like price fields.* A zero-volume bar (no shares/contracts
//!    traded that session) is a real, legal market condition -- illiquid days happen -- unlike a zero or negative
//!    price, which `donchian_breakout`'s own validation (and every other price series in this crate) treats as
//!    corrupt data. Volume must still be finite (no NaN/infinite): a non-finite volume cannot be averaged or
//!    compared meaningfully, so that case is refused rather than silently coerced.
//! 5. *Volume average is a MEAN, not a sum*, and is compared with a strict `>` against `VCB_VOLUME_MULTIPLIER *`
//!    that mean -- stated explicitly because the spec's own wording ("average volume") and the multiplier (1.5x)
//!    only make sense against a per-bar average, not an accumulated total that grows with the window length.
//!    Unlike `compare_to_mean` (`exact.rs`), this comparison is not promoted to exact rational arithmetic: the
//!    1.5x threshold is itself an approximate, judgment-call multiplier (not a structural tie the way
//!    `close == SMA` is for the ETF/crypto trend rules), so an IEEE-754 `sum / n` followed by a plain `f64 >`
//!    comparison is an accepted, deliberate source of (extremely rare, economically meaningless) rounding noise
//!    at the exact 1.5x boundary -- the same tradeoff every other mean-based comparison outside `exact.rs` already
//!    makes in this crate.
//! 6. *Function shape and hold-chain seed are unchanged from `donchian_breakout`*: whole series in, whole weight
//!    vector out, one local `current` accumulator threaded across the scan, seeded at the flat `0.0` that is also
//!    the pre-history skip value -- see `donchian_breakout`'s module doc, choices 3 and 4, which apply here
//!    verbatim.

use chrono::NaiveDate;
use std::fmt;

use crate::donchian_breakout::{DonchianError, HlcSeries};

/// Default price-channel length in bars (see module docs, choice 1: independent from [`VCB_DEFAULT_VOLUME_N`]).
pub const VCB_DEFAULT_PRICE_N: usize = 20;

/// Default trailing-volume-average window length in bars (see module docs, choice 1: independent from
/// [`VCB_DEFAULT_PRICE_N`]).
pub const VCB_DEFAULT_VOLUME_N: usize = 20;

/// Volume confirmation multiplier: today's volume must strictly exceed this many times the trailing average for
/// a price breakout to actually fire.
pub const VCB_VOLUME_MULTIPLIER: f64 = 1.5;

/// Why [`OhlcvSeries::new`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum VolumeConfirmedBreakoutError {
    /// The underlying high/low/close series (dates, symbol, OHLC invariants) failed [`HlcSeries::new`]'s own
    /// validation; see that error's variants for the specific reason.
    Hlc(DonchianError),
    /// `volumes` passed to `OhlcvSeries::new` has a different length than the price series.
    VolumeLengthMismatch {
        symbol: String,
        bars: usize,
        volumes: usize,
    },
    /// A volume is NaN, infinite, or negative (zero is legal: see module docs, choice 4).
    InvalidVolume {
        symbol: String,
        date: NaiveDate,
        value: f64,
    },
}

impl fmt::Display for VolumeConfirmedBreakoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for VolumeConfirmedBreakoutError {}

impl From<DonchianError> for VolumeConfirmedBreakoutError {
    fn from(e: DonchianError) -> Self {
        VolumeConfirmedBreakoutError::Hlc(e)
    }
}

/// Validated daily OHLCV series of one instrument: an [`HlcSeries`] (high/low/close, with that type's own
/// invariants -- see module docs, choice 3) plus an independent `volumes` column, one entry per bar, each finite
/// and `>= 0.0` (module docs, choice 4).
#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvSeries {
    hlc: HlcSeries,
    volumes: Vec<f64>,
}

impl OhlcvSeries {
    pub fn new(
        symbol: impl Into<String>,
        dates: Vec<NaiveDate>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> Result<Self, VolumeConfirmedBreakoutError> {
        let symbol = symbol.into();
        let hlc = HlcSeries::new(symbol.clone(), dates, highs, lows, closes)?;
        if volumes.len() != hlc.len() {
            return Err(VolumeConfirmedBreakoutError::VolumeLengthMismatch {
                symbol,
                bars: hlc.len(),
                volumes: volumes.len(),
            });
        }
        for (i, &value) in volumes.iter().enumerate() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(VolumeConfirmedBreakoutError::InvalidVolume {
                    symbol,
                    date: hlc.dates()[i],
                    value,
                });
            }
        }
        Ok(Self { hlc, volumes })
    }

    pub fn symbol(&self) -> &str {
        self.hlc.symbol()
    }
    pub fn dates(&self) -> &[NaiveDate] {
        self.hlc.dates()
    }
    pub fn highs(&self) -> &[f64] {
        self.hlc.highs()
    }
    pub fn lows(&self) -> &[f64] {
        self.hlc.lows()
    }
    pub fn closes(&self) -> &[f64] {
        self.hlc.closes()
    }
    pub fn volumes(&self) -> &[f64] {
        &self.volumes
    }
    pub fn len(&self) -> usize {
        self.hlc.len()
    }
    pub fn is_empty(&self) -> bool {
        self.hlc.is_empty()
    }
}

/// Decide the whole-series volume-confirmed Donchian breakout weight vector for `series`, with price-channel
/// length `price_n` and volume-average window length `volume_n` (use [`VCB_DEFAULT_PRICE_N`] /
/// [`VCB_DEFAULT_VOLUME_N`] for the defaults -- two independent parameters, module docs choice 1).
///
/// For bar `t` with `t < max(price_n, volume_n)` (fewer bars visible than the larger of the two full preceding
/// windows needs), the result is `0.0` -- silent skip, flat (module docs, choice 2). For `t >= max(price_n,
/// volume_n)`: the price channel is `max(highs[t-price_n..t])` / `min(lows[t-price_n..t])`, and the trailing
/// volume average is `mean(volumes[t-volume_n..t])` (module docs, choice 5). Weight flips to `1.0` if `closes[t]`
/// is strictly above the upper channel AND `volumes[t] > VCB_VOLUME_MULTIPLIER * trailing_volume_average`; flips
/// to `0.0` if `closes[t]` is strictly below the lower channel AND the same volume condition holds; otherwise
/// HOLDS the previous bar's weight (which, for the first decision, is the `0.0` flat seed).
///
/// # Panics
/// Panics if `price_n == 0` or `volume_n == 0` (a zero-length window is not a meaningful parameter).
pub fn decide_volume_confirmed_breakout(series: &OhlcvSeries, price_n: usize, volume_n: usize) -> Vec<f64> {
    assert!(price_n >= 1, "price_n must be >= 1");
    assert!(volume_n >= 1, "volume_n must be >= 1");
    let highs = series.highs();
    let lows = series.lows();
    let closes = series.closes();
    let volumes = series.volumes();
    let len = series.len();
    let min_history = price_n.max(volume_n);
    let mut weights = vec![0.0; len];
    let mut current = 0.0; // flat seed; same convention as donchian_breakout.
    for t in 0..len {
        if t < min_history {
            // Fewer bars visible than the larger of the two full preceding windows needs: silent skip, flat.
            weights[t] = 0.0;
            continue;
        }
        let price_window_start = t - price_n;
        let upper = highs[price_window_start..t]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let lower = lows[price_window_start..t]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let volume_window_start = t - volume_n;
        let trailing_volume_avg: f64 =
            volumes[volume_window_start..t].iter().sum::<f64>() / volume_n as f64;
        let volume_confirmed = volumes[t] > VCB_VOLUME_MULTIPLIER * trailing_volume_avg;

        let close = closes[t];
        if close > upper && volume_confirmed {
            current = 1.0;
        } else if close < lower && volume_confirmed {
            current = 0.0;
        }
        // else: HOLD (price condition false, or volume condition false, or both), `current` unchanged.
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

    fn series(
        bars: &[(NaiveDate, f64, f64, f64, f64)], // (date, high, low, close, volume)
    ) -> OhlcvSeries {
        let dates = bars.iter().map(|b| b.0).collect();
        let highs = bars.iter().map(|b| b.1).collect();
        let lows = bars.iter().map(|b| b.2).collect();
        let closes = bars.iter().map(|b| b.3).collect();
        let volumes = bars.iter().map(|b| b.4).collect();
        OhlcvSeries::new("TEST", dates, highs, lows, closes, volumes).unwrap()
    }

    #[test]
    fn default_constants() {
        assert_eq!(VCB_DEFAULT_PRICE_N, 20);
        assert_eq!(VCB_DEFAULT_VOLUME_N, 20);
        assert_eq!(VCB_VOLUME_MULTIPLIER, 1.5);
    }

    #[test]
    fn price_breakout_with_volume_confirmation_flips_weight() {
        // price_n = volume_n = 2. Bars 0,1 build both windows: high10/low5/close7, volume100 each. Bar 2 is the
        // first decision (t == 2 == max(2,2)): its close (11) clears the price channel's upper bound (10), AND
        // its volume (300) strictly exceeds 1.5x the trailing average (1.5 * 100 = 150) -- both conditions hold,
        // so the weight flips from the flat 0.0 seed to 1.0.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 3), 12.0, 4.0, 11.0, 300.0),
        ]);
        let w = decide_volume_confirmed_breakout(&s, 2, 2);
        assert_eq!(w, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn price_breakout_without_volume_confirmation_holds_unchanged() {
        // Identical price action to the previous test (bar 2's close 11 clears the same upper bound of 10), but
        // bar 2's volume is only 120 -- NOT > 1.5 * 100 = 150. The price condition alone is not enough: the
        // volume condition fails, so the breakout does NOT fire and the weight HOLDS at the flat 0.0 seed, even
        // though the price action is identical to the confirmed case above. This is the primitive's whole point.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 3), 12.0, 4.0, 11.0, 120.0),
        ]);
        let w = decide_volume_confirmed_breakout(&s, 2, 2);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn volume_spike_with_no_price_breakout_holds_unchanged() {
        // Bar 2's close (8) stays strictly between the price channel's bounds (5, 10) -- no price breakout at
        // all -- even though its volume (1000) massively exceeds 1.5x the trailing average (150). Volume alone
        // never flips anything: the weight holds at the flat 0.0 seed.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 3), 9.0, 6.0, 8.0, 1000.0),
        ]);
        let w = decide_volume_confirmed_breakout(&s, 2, 2);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn insufficient_history_uses_max_not_min_of_the_two_windows() {
        // price_n = 2, volume_n = 4 -- deliberately different, so max(2,4) = 4 and min(2,4) = 2 disagree on when
        // the first real decision may happen. Bar 3's price action (close 15) breaks out above the price
        // channel formed by bars [1,2] (upper 10), and bar 3's own volume (10000) is huge -- under a (buggy)
        // `min`-based threshold, t = 3 (>= min = 2) would be treated as a valid decision day and would flip the
        // weight to 1.0. Under the correct `max`-based threshold, t = 3 (< max = 4) is still in the silent-skip
        // region (the volume window has not yet accumulated its own full 4 bars), so the weight must stay flat.
        // Bar 4 is the first real decision (t == 4 == max(2,4)): its price action (close 7) sits strictly between
        // the channel formed by bars [2,3] (upper 20, lower 3), so it HOLDS at the flat 0.0 seed regardless.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 3), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 4), 20.0, 3.0, 15.0, 10000.0),
            (d(2024, 1, 5), 10.0, 5.0, 7.0, 100.0),
        ]);
        let w = decide_volume_confirmed_breakout(&s, 2, 4);
        assert_eq!(w, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn volume_window_excludes_the_bar_exactly_n_plus_one_back_off_by_one_boundary() {
        // price_n = volume_n = 2. Bar 0 has a huge outlier volume (100000), exactly n+1 = 3 bars back from
        // decision day t = 3 -- it must be EXCLUDED from bar 3's trailing volume window (which is bars [1,2]).
        // Bar 3's close (15) clears the price channel formed by bars [1,2] (upper 10), and its volume (300)
        // strictly exceeds 1.5x the CORRECT trailing average (100, from bars [1,2] only: 1.5 * 100 = 150), so the
        // weight flips to 1.0. If an off-by-one bug widened the volume window to include bar 0 (average would
        // balloon to ~50050, 1.5x of which is ~75075), bar 3's volume (300) would NOT clear it and the (wrong)
        // result would stay at the flat 0.0 -- this test fails under that bug.
        let s = series(&[
            (d(2024, 1, 1), 10.0, 5.0, 7.0, 100000.0), // t-n-1 = 0: must be excluded from bar 3's volume window
            (d(2024, 1, 2), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 3), 10.0, 5.0, 7.0, 100.0),
            (d(2024, 1, 4), 20.0, 3.0, 15.0, 300.0),
        ]);
        let w = decide_volume_confirmed_breakout(&s, 2, 2);
        assert_eq!(w[2], 0.0); // bar 2's own decision: close 7 between 5 and 10 -- hold at flat regardless.
        assert_eq!(w[3], 1.0); // bar 3: volume window [1,2] excludes bar 0 -> confirmed breakout up.
    }

    #[test]
    fn series_rejects_volume_length_mismatch() {
        assert!(matches!(
            OhlcvSeries::new(
                "X",
                vec![d(2024, 1, 1), d(2024, 1, 2)],
                vec![10.0, 10.0],
                vec![5.0, 5.0],
                vec![7.0, 7.0],
                vec![100.0],
            ),
            Err(VolumeConfirmedBreakoutError::VolumeLengthMismatch { .. })
        ));
    }

    #[test]
    fn series_rejects_negative_and_non_finite_volume_but_allows_zero() {
        assert!(matches!(
            OhlcvSeries::new(
                "X",
                vec![d(2024, 1, 1)],
                vec![10.0],
                vec![5.0],
                vec![7.0],
                vec![-1.0],
            ),
            Err(VolumeConfirmedBreakoutError::InvalidVolume { .. })
        ));
        assert!(matches!(
            OhlcvSeries::new(
                "X",
                vec![d(2024, 1, 1)],
                vec![10.0],
                vec![5.0],
                vec![7.0],
                vec![f64::NAN],
            ),
            Err(VolumeConfirmedBreakoutError::InvalidVolume { .. })
        ));
        // Zero volume is legal (a genuinely illiquid day): module docs, choice 4.
        assert!(OhlcvSeries::new(
            "X",
            vec![d(2024, 1, 1)],
            vec![10.0],
            vec![5.0],
            vec![7.0],
            vec![0.0],
        )
        .is_ok());
    }

    #[test]
    fn series_propagates_hlc_validation_errors() {
        // high < low is an HlcSeries-level invariant; OhlcvSeries must surface it, wrapped, not swallow it.
        assert!(matches!(
            OhlcvSeries::new(
                "X",
                vec![d(2024, 1, 1)],
                vec![4.0],
                vec![5.0],
                vec![4.5],
                vec![100.0],
            ),
            Err(VolumeConfirmedBreakoutError::Hlc(DonchianError::HighBelowLow { .. }))
        ));
    }

    #[test]
    #[should_panic(expected = "price_n must be >= 1")]
    fn zero_price_n_panics() {
        let s = series(&[(d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0)]);
        let _ = decide_volume_confirmed_breakout(&s, 0, 2);
    }

    #[test]
    #[should_panic(expected = "volume_n must be >= 1")]
    fn zero_volume_n_panics() {
        let s = series(&[(d(2024, 1, 1), 10.0, 5.0, 7.0, 100.0)]);
        let _ = decide_volume_confirmed_breakout(&s, 2, 0);
    }
}
