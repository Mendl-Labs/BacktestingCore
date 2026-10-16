//! Book-level trial ledger, sealed-holdout bookkeeping and effective breadth (design 4.4, Appendix A rulings).
//!
//! **Trial ledger.** One ledger per book lineage counts every choice made on the book, because the deflated Sharpe is
//! only as honest as its trial count (memory: three separate DSR trial-count bugs). The effective number of trials is
//!
//! ```text
//! K_effective = configs x universe_variants x allocator_variants
//!             + screens + candidates_screened + lineage_prior_trials + concurrent_tenant_trials
//! ```
//!
//! `configs` are UNIQUE parameter configurations: recording the same id twice does not add a trial, recording a new id
//! does. A configuration that blew up or errored still counts ([`TrialLedger::record_failed_config`]); it just has no
//! Sharpe. The dispersion of the trial Sharpe ratios is a ROBUST scale (`1.4826 x MAD`), not the plain standard
//! deviation, because blown-up configurations inflated the plain estimate to 7.47 against a 1.27 null bar (memory
//! 2026-09-18, finding 7).
//!
//! **Sealed holdout.** The tail `max(fraction, min_bars)` of the calendar is looked at once; every later look is
//! reported as post hoc ([`HoldoutLook::post_hoc`]). Time is supplied by the caller as milliseconds; the crate never
//! reads a clock.
//!
//! **Effective breadth.** A hub instrument shared by many legs is one bet, not many (shared-leg pool artifact, memory
//! 2026-09-19): [`independent_leg_groups`] counts connected components of the leg/instrument graph, and
//! [`effective_breadth_from_correlation`] is the participation ratio `N^2 / sum_ij rho_ij^2`.

use crate::dsr::{deflated_sharpe_from_returns, DsrResult};
use crate::error::{invalid, EvalError, Result};
use crate::stats;
use std::collections::BTreeMap;
use std::ops::Range;

/// What [`TrialLedger::record_config`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordOutcome {
    /// A new configuration with a finite Sharpe was added (K increased by one, times the variant factors).
    New,
    /// A new configuration without a usable Sharpe was added: it counts as a trial but not in the dispersion.
    NewFailed,
    /// The id was already in the ledger: nothing was added. `upgraded` is true when a previously failed trial now has a
    /// finite Sharpe (a retry of the same trial).
    Duplicate { upgraded: bool },
}

/// Summary of a ledger for reporting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LedgerSummary {
    pub configs: usize,
    pub failed_configs: usize,
    pub universe_variants: usize,
    pub allocator_variants: usize,
    pub screens: usize,
    pub candidates_screened: usize,
    pub lineage_prior: usize,
    pub concurrent_tenant: usize,
    pub k_effective: usize,
    /// `1.4826 x MAD` of the finite trial Sharpe ratios (annualised units), `None` with fewer than two.
    pub dispersion_robust: Option<f64>,
    /// Plain sample standard deviation of the same Sharpe ratios, for comparison.
    pub dispersion_plain: Option<f64>,
}

/// Per-lineage trial ledger.
#[derive(Clone, Debug)]
pub struct TrialLedger {
    trials: BTreeMap<String, Option<f64>>,
    universe_variants: usize,
    allocator_variants: usize,
    screens: usize,
    candidates_screened: usize,
    lineage_prior: usize,
    concurrent_tenant: usize,
}

impl Default for TrialLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl TrialLedger {
    /// Empty ledger with one universe variant and one allocator variant.
    pub fn new() -> Self {
        TrialLedger {
            trials: BTreeMap::new(),
            universe_variants: 1,
            allocator_variants: 1,
            screens: 0,
            candidates_screened: 0,
            lineage_prior: 0,
            concurrent_tenant: 0,
        }
    }

    /// Record a configuration evaluated on the book with its ANNUALISED Sharpe. A non-finite Sharpe is recorded as a
    /// failed trial (still counted).
    pub fn record_config(&mut self, id: &str, annual_sharpe: f64) -> Result<RecordOutcome> {
        if id.is_empty() {
            return Err(invalid("id", "configuration id must not be empty"));
        }
        let value = if annual_sharpe.is_finite() { Some(annual_sharpe) } else { None };
        match self.trials.get_mut(id) {
            Some(existing) => {
                if existing.is_none() && value.is_some() {
                    *existing = value;
                    Ok(RecordOutcome::Duplicate { upgraded: true })
                } else {
                    Ok(RecordOutcome::Duplicate { upgraded: false })
                }
            }
            None => {
                self.trials.insert(id.to_string(), value);
                Ok(if value.is_some() { RecordOutcome::New } else { RecordOutcome::NewFailed })
            }
        }
    }

    /// Record a configuration that errored or was rejected before a Sharpe existed; it still counts as a trial.
    pub fn record_failed_config(&mut self, id: &str) -> Result<RecordOutcome> {
        self.record_config(id, f64::NAN)
    }

    /// Number of universe variants tried (at least 1).
    pub fn set_universe_variants(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(invalid("universe_variants", "must be at least 1"));
        }
        self.universe_variants = n;
        Ok(())
    }

    /// Number of allocator variants tried (at least 1).
    pub fn set_allocator_variants(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(invalid("allocator_variants", "must be at least 1"));
        }
        self.allocator_variants = n;
        Ok(())
    }

    /// Screens run before any configuration was chosen (e.g. cointegration screens).
    pub fn add_screens(&mut self, n: usize) {
        self.screens = self.screens.saturating_add(n);
    }

    /// Candidates screened for marginal contribution (also the family size for the BH correction).
    pub fn add_candidates_screened(&mut self, n: usize) {
        self.candidates_screened = self.candidates_screened.saturating_add(n);
    }

    /// Trials inherited from earlier runs of the same lineage.
    pub fn set_lineage_prior(&mut self, n: usize) {
        self.lineage_prior = n;
    }

    /// Trials run concurrently by the same tenant.
    pub fn set_concurrent_tenant(&mut self, n: usize) {
        self.concurrent_tenant = n;
    }

    /// Unique configurations recorded (finite or failed).
    pub fn n_configs(&self) -> usize {
        self.trials.len()
    }

    /// `K_effective` (see the module docs), saturating.
    pub fn k_effective(&self) -> usize {
        self.trials
            .len()
            .saturating_mul(self.universe_variants)
            .saturating_mul(self.allocator_variants)
            .saturating_add(self.screens)
            .saturating_add(self.candidates_screened)
            .saturating_add(self.lineage_prior)
            .saturating_add(self.concurrent_tenant)
    }

    /// The finite trial Sharpe ratios in id order.
    pub fn sharpes(&self) -> Vec<f64> {
        self.trials.values().filter_map(|v| *v).collect()
    }

    /// Robust dispersion `1.4826 x MAD` of the trial Sharpe ratios (annualised units).
    pub fn dispersion_robust(&self) -> Result<f64> {
        let s = self.sharpes();
        if s.len() < 2 {
            return Err(EvalError::TooShort { what: "trial Sharpe dispersion", need: 2, got: s.len() });
        }
        stats::mad_scale(&s)
    }

    /// Plain sample standard deviation of the trial Sharpe ratios (reported for comparison, not used by the DSR).
    pub fn dispersion_plain(&self) -> Result<f64> {
        let s = self.sharpes();
        if s.len() < 2 {
            return Err(EvalError::TooShort { what: "trial Sharpe dispersion", need: 2, got: s.len() });
        }
        stats::std_dev(&s)
    }

    /// Everything the result JSON reports about the ledger.
    pub fn summary(&self) -> LedgerSummary {
        LedgerSummary {
            configs: self.trials.len(),
            failed_configs: self.trials.values().filter(|v| v.is_none()).count(),
            universe_variants: self.universe_variants,
            allocator_variants: self.allocator_variants,
            screens: self.screens,
            candidates_screened: self.candidates_screened,
            lineage_prior: self.lineage_prior,
            concurrent_tenant: self.concurrent_tenant,
            k_effective: self.k_effective(),
            dispersion_robust: self.dispersion_robust().ok(),
            dispersion_plain: self.dispersion_plain().ok(),
        }
    }

    /// Deflated Sharpe ratio of the selected configuration's OUT-OF-SAMPLE portfolio returns, using this ledger's
    /// `K_effective` and robust dispersion.
    pub fn deflated_sharpe_of(
        &self,
        oos_returns: &[f64],
        periods_per_year: f64,
        floor_at_normal: bool,
    ) -> Result<DsrResult> {
        let k = self.k_effective();
        if k == 0 {
            return Err(invalid("ledger", "the ledger is empty: the selected configuration must be recorded first"));
        }
        let disp = self.dispersion_robust()?;
        deflated_sharpe_from_returns(oos_returns, k, disp, periods_per_year, floor_at_normal)
    }
}

/// Tail range reserved as the sealed holdout: the last `max(ceil(fraction x n_bars), min_bars)` bars. At least one bar
/// must remain for training.
pub fn holdout_tail(n_bars: usize, fraction: f64, min_bars: usize) -> Result<Range<usize>> {
    if !(fraction.is_finite() && fraction > 0.0 && fraction < 1.0) {
        return Err(invalid("fraction", format!("must be in (0, 1), got {fraction}")));
    }
    let by_fraction = (fraction * n_bars as f64).ceil() as usize;
    let len = by_fraction.max(min_bars);
    if len == 0 || len >= n_bars {
        return Err(EvalError::TooShort {
            what: "calendar for a sealed holdout plus training data",
            need: len.saturating_add(1),
            got: n_bars,
        });
    }
    Ok(n_bars - len..n_bars)
}

/// What a look at the sealed holdout means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldoutLook {
    /// This was the first look (the holdout was unspent).
    pub first_look: bool,
    /// Any later look: the holdout was already spent, so the result is post hoc and must be labelled as such.
    pub post_hoc: bool,
    /// Total looks including this one.
    pub looks: u32,
    /// When the first look happened (milliseconds supplied by the caller).
    pub spent_at_ms: i64,
}

/// A sealed holdout window with a spent flag.
#[derive(Clone, Debug)]
pub struct SealedHoldout {
    range: Range<usize>,
    spent_at_ms: Option<i64>,
    looks: u32,
}

impl SealedHoldout {
    /// A holdout over a non-empty bar range.
    pub fn new(range: Range<usize>) -> Result<Self> {
        if range.start >= range.end {
            return Err(invalid("range", "the sealed holdout must contain at least one bar"));
        }
        Ok(SealedHoldout { range, spent_at_ms: None, looks: 0 })
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn is_spent(&self) -> bool {
        self.spent_at_ms.is_some()
    }
    pub fn spent_at_ms(&self) -> Option<i64> {
        self.spent_at_ms
    }
    pub fn looks(&self) -> u32 {
        self.looks
    }

    /// Look at the holdout at time `now_ms`. The first look spends it; every later look is flagged `post_hoc`.
    pub fn look(&mut self, now_ms: i64) -> HoldoutLook {
        self.looks = self.looks.saturating_add(1);
        match self.spent_at_ms {
            None => {
                self.spent_at_ms = Some(now_ms);
                HoldoutLook { first_look: true, post_hoc: false, looks: self.looks, spent_at_ms: now_ms }
            }
            Some(t) => HoldoutLook { first_look: false, post_hoc: true, looks: self.looks, spent_at_ms: t },
        }
    }

    /// The holdout must not overlap any range used for training, tuning or selection.
    pub fn verify_disjoint(&self, used: &[Range<usize>]) -> Result<()> {
        for r in used {
            if r.start < r.end && r.start < self.range.end && self.range.start < r.end {
                return Err(EvalError::HoldoutOverlap {
                    holdout_start: self.range.start,
                    holdout_end: self.range.end,
                    other_start: r.start,
                    other_end: r.end,
                });
            }
        }
        Ok(())
    }
}

/// Participation-ratio effective number of independent bets from a correlation matrix (row-major rows):
/// `N^2 / sum_ij rho_ij^2 = (sum lambda)^2 / sum lambda^2`. Identity gives `N`, all-ones gives 1.
pub fn effective_breadth_from_correlation(corr: &[Vec<f64>]) -> Result<f64> {
    let n = corr.len();
    if n == 0 {
        return Err(EvalError::TooShort { what: "correlation matrix", need: 1, got: 0 });
    }
    for row in corr {
        if row.len() != n {
            return Err(EvalError::LengthMismatch { what: "correlation matrix row", left: n, right: row.len() });
        }
    }
    let mut ss = 0.0;
    for (i, row) in corr.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(EvalError::NonFinite { what: "correlation matrix", index: i * n + j });
            }
            if v.abs() > 1.0 + 1e-9 {
                return Err(invalid("corr", format!("entry ({i}, {j}) = {v} is outside [-1, 1]")));
            }
            if (v - corr[j][i]).abs() > 1e-9 {
                return Err(invalid("corr", "matrix is not symmetric"));
            }
            ss += v * v;
        }
        if (row[i] - 1.0).abs() > 1e-9 {
            return Err(invalid("corr", format!("diagonal entry {i} is not 1")));
        }
    }
    Ok((n * n) as f64 / ss)
}

/// Number of independent leg groups: legs (each a list of instrument ids) that share any instrument, directly or
/// through a chain, form one group. A hub instrument used by many pairs makes them ONE bet.
pub fn independent_leg_groups(legs: &[Vec<u32>]) -> usize {
    let n = legs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut owner: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, leg) in legs.iter().enumerate() {
        for inst in leg {
            match owner.get(inst) {
                Some(&j) => {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
                None => {
                    owner.insert(*inst, i);
                }
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}
