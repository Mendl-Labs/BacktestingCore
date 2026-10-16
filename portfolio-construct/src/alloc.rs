//! Allocators: how sleeve shares are chosen (design 3.2 `AllocatorSpec`, 4.2; quant ruling R-P6).
//!
//! The evidence-grade menu is `Fixed`, `Equal` and `InverseVol`, always as STATIC shares: they change only when the
//! simulator calls [`AllocatorState::review`] (a declared review date, or a walk-forward fold start), so the live
//! rebalancer can reproduce them as a plan revision. Everything adaptive beyond that (min-variance, HRP, per-bar
//! dynamic weights) is research-only and is not in this crate.
//!
//! `InverseVol` definition (the pre-registered reference of the PF0 book key, Amendment 12 section 4): for each sleeve
//! take the last `lookback_bars` returns of its own calendar visible at the review, the sample standard deviation with
//! ddof 1 (sequential sums), and set `share_i = total * (1/sd_i) / SUM_k (1/sd_k)`. A review that lacks history
//! (fewer than `lookback_bars` visible returns for ANY sleeve) or meets a non-positive or non-finite deviation leaves the
//! shares unchanged (they stay at their previous static values). `floor` is a lower bound on each deviation (`sd_i =
//! max(sd_i, floor)`, default 0 = the key's definition); it lets a zero-variance sleeve receive a finite weight instead
//! of blocking the review.
//!
//! Allocators see only the past: a review receives each sleeve's return series and the NUMBER of returns visible at the
//! review close, and reads nothing beyond that index (poisoning test: values after it may be garbage or NaN without
//! changing a bit of the result).

/// When an `InverseVol` allocator's shares may change. Recorded in provenance; the mechanics are the same, the caller
/// decides when to call `review`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezeRule {
    /// Re-estimated at declared review dates (calendar month-ends in the key), static between them.
    AtReviewDates,
    /// Estimated once at the start of each walk-forward fold from data before the fold, then frozen for the fold.
    AtFoldStart,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AllocatorSpec {
    /// Owner- or plan-chosen shares, each in `(0, 1]`, summing to at most `total`.
    Fixed(Vec<f64>),
    /// `total / n` each.
    Equal,
    /// Static inverse-volatility shares, see the module docs.
    InverseVol { lookback_bars: usize, floor: f64, freeze: FreezeRule },
}

/// Why an allocator could not be built or reviewed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// No sleeves, or `Fixed` has the wrong length.
    BadLength,
    BadShare,
    SharesExceedTotal,
    BadParam(&'static str),
}

/// Why a review left the shares unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldReason {
    /// The allocator is `Fixed` or `Equal`: reviews never change anything.
    Static,
    /// Some sleeve has fewer than `lookback_bars` visible returns.
    NotEnoughHistory,
    /// Some sleeve's deviation is zero/negative/non-finite (or a return in the window is not finite).
    DegenerateVolatility,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewOutcome {
    Updated,
    Held(HoldReason),
}

/// The current shares of one allocator instance.
#[derive(Clone, Debug, PartialEq)]
pub struct AllocatorState {
    spec: AllocatorSpec,
    total: f64,
    shares: Vec<f64>,
}

impl AllocatorState {
    /// `n` sleeves, shares summing to (at most) `total`. `Fixed` uses its own vector, `Equal` and `InverseVol` start at
    /// `total / n` (the key's initial 0.5/0.5).
    pub fn new(spec: AllocatorSpec, n: usize, total: f64) -> Result<Self, AllocError> {
        if n == 0 {
            return Err(AllocError::BadLength);
        }
        if !(total.is_finite() && total > 0.0 && total <= 1.0) {
            return Err(AllocError::BadParam("total must be in (0, 1]"));
        }
        let shares = match &spec {
            AllocatorSpec::Fixed(v) => {
                if v.len() != n {
                    return Err(AllocError::BadLength);
                }
                if v.iter().any(|s| !(s.is_finite() && *s > 0.0 && *s <= 1.0)) {
                    return Err(AllocError::BadShare);
                }
                let mut sum = 0.0;
                for s in v {
                    sum += s;
                }
                if sum > total + total * 1e-12 {
                    return Err(AllocError::SharesExceedTotal);
                }
                v.clone()
            }
            AllocatorSpec::Equal => vec![total / n as f64; n],
            AllocatorSpec::InverseVol { lookback_bars, floor, .. } => {
                if *lookback_bars < 2 {
                    return Err(AllocError::BadParam("lookback_bars must be at least 2"));
                }
                if !(floor.is_finite() && *floor >= 0.0) {
                    return Err(AllocError::BadParam("floor must be finite and not negative"));
                }
                vec![total / n as f64; n]
            }
        };
        Ok(AllocatorState { spec, total, shares })
    }

    pub fn spec(&self) -> &AllocatorSpec {
        &self.spec
    }

    /// The shares in force (sleeve order as given to `new` and `review`).
    pub fn shares(&self) -> &[f64] {
        &self.shares
    }

    /// A declared review. `sleeve_returns[s]` is sleeve `s`'s return series on its OWN calendar and `visible[s]` the
    /// number of those returns that are visible at the review close; nothing at or beyond `visible[s]` is read.
    pub fn review(&mut self, sleeve_returns: &[&[f64]], visible: &[usize]) -> ReviewOutcome {
        let AllocatorSpec::InverseVol { lookback_bars, floor, .. } = &self.spec else {
            return ReviewOutcome::Held(HoldReason::Static);
        };
        let (lookback, floor) = (*lookback_bars, *floor);
        if sleeve_returns.len() != self.shares.len() || visible.len() != self.shares.len() {
            return ReviewOutcome::Held(HoldReason::NotEnoughHistory);
        }
        let mut inv: Vec<f64> = Vec::with_capacity(self.shares.len());
        for (rets, &vis) in sleeve_returns.iter().zip(visible) {
            let vis = vis.min(rets.len());
            if vis < lookback {
                return ReviewOutcome::Held(HoldReason::NotEnoughHistory);
            }
            let window = &rets[vis - lookback..vis];
            if window.iter().any(|x| !x.is_finite()) {
                return ReviewOutcome::Held(HoldReason::DegenerateVolatility);
            }
            let mut sd = sample_std(window);
            if floor > 0.0 && sd < floor {
                sd = floor;
            }
            if !(sd > 0.0 && sd.is_finite()) {
                return ReviewOutcome::Held(HoldReason::DegenerateVolatility);
            }
            inv.push(1.0 / sd);
        }
        let mut tot = 0.0;
        for v in &inv {
            tot += v;
        }
        self.shares = inv.iter().map(|v| self.total * (v / tot)).collect();
        ReviewOutcome::Updated
    }
}

/// Sample standard deviation, ddof 1, sequential sums (the key's `std1`).
pub fn sample_std(xs: &[f64]) -> f64 {
    let n = xs.len();
    let mut sum = 0.0;
    for x in xs {
        sum += x;
    }
    let mean = sum / n as f64;
    let mut v = 0.0;
    for x in xs {
        v += (x - mean) * (x - mean);
    }
    (v / (n - 1) as f64).sqrt()
}
