//! The output of a book run: full-resolution, column-oriented, on the ACCOUNT clock, plus the derived checks
//! (attribution identity), metrics, the series digest and the one-sleeve view that is bit-identical to `simulate`.
//!
//! Conventions (the same as `SimResult`, so a one-sleeve book can be compared cell by cell):
//! * every per-bar vector has one entry per account bar (`times`); bar 0 is the account's first bar (initial build,
//!   return 0);
//! * `ret[k]` is the return dated at the close of bar `k` (earned over `(k-1, k]`), post-cost in this run;
//! * `target_weights[k]` is the combined standing target (`SUM share*w * risk_scale`, a fraction of the capital base)
//!   AFTER bar `k`'s decisions and share review; `held_weights[k]` are the post-trade weights `units*mark/equity`;
//!   the PF0 book key prints the same quantities one row later (its row `k` shows what was in force ENTERING bar `k`);
//! * flat matrices are `[bar * n + j]` (instruments) or `[bar * S + s]` (sleeves).

use crate::bartime::BarTime;
use crate::book::BookRefusal;
use crate::date::Date;
use crate::metrics::{answer_key_metrics, cumprod_one_plus, Metrics, METRIC_DEFINITIONS};
use crate::rule::RefusalKind;
use crate::sha256::{to_hex, Sha256};
use crate::sim::{digest, exposure_stats, ExposureStats, Refusal, SimResult, Window};

/// Name recorded with the book-level metrics: `answer_key_v1`'s formulas with `ppy = n_returns / years` on the ACCOUNT
/// clock (a crypto-plus-ETF book annualises on 365-day bars; each sleeve is also reported on its own calendar).
pub const BOOK_METRIC_DEFINITIONS: &str = "account_clock_v1";

/// One sleeve's shadow (unit-capital) curve on its OWN calendar: one row per own bar after the shadow's first bar.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowSeries {
    pub dates: Vec<Date>,
    /// Returns of the hypothetical sleeve-alone account with zero cost (the allocator's input).
    pub gross: Vec<f64>,
    /// Returns of the same account charged this run's cost preset (its own cost, never the joint account's).
    pub cost: Vec<f64>,
}

/// Result of [`crate::attribution_report`]-style checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttributionReport {
    /// `max_k | ret_k - (SUM_j w_(j,k-1) r_(j,k) + financing_k/E_(k-1) - cost_k/E_(k-1)) |`.
    pub max_abs_err_total: f64,
    /// `max_k | SUM_s contrib_(s,k) - SUM_j w_(j,k-1) r_(j,k) |`.
    pub max_abs_err_contrib_sum: f64,
    pub bars_checked: usize,
}

/// Full-resolution result of one book run. See the module docs for the conventions.
#[derive(Clone, Debug)]
pub struct BookResult {
    pub instruments: Vec<String>,
    pub sleeve_ids: Vec<String>,
    /// `(rule id, impl version)` per sleeve.
    pub sleeve_rules: Vec<(String, String)>,
    /// Instrument indices of each sleeve.
    pub sleeve_universes: Vec<Vec<usize>>,
    pub cost_model_id: &'static str,
    pub metric_definitions: &'static str,
    /// Account clock.
    pub times: Vec<BarTime>,
    /// Index of each account bar on the panel's union clock.
    pub clock_index: Vec<usize>,
    pub ret: Vec<f64>,
    pub ret_pre_cost: Vec<f64>,
    /// Post-cost equity at the close.
    pub equity: Vec<f64>,
    /// Pre-cost equity at the close (marked, before this bar's trading cost).
    pub equity_pre: Vec<f64>,
    pub cash: Vec<f64>,
    pub cost: Vec<f64>,
    pub traded_notional: Vec<f64>,
    /// Financing credited to cash over `(k-1, k]` (negative = paid).
    pub financing: Vec<f64>,
    pub gross_exposure: Vec<f64>,
    pub net_exposure: Vec<f64>,
    /// Risk scale in force on the bar: the constant approval scale times the overlay's scale.
    pub risk_scale: Vec<f64>,
    /// At least one sleeve was due (a driver run).
    pub run: Vec<bool>,
    /// The whole book was refused on this bar (nothing traded).
    pub book_refused: Vec<bool>,
    /// The overlay had halted the account by this bar.
    pub halted: Vec<bool>,
    // [bar * S + s]
    /// A decision of the sleeve succeeded on this bar.
    pub decision: Vec<bool>,
    /// The sleeve's rule refused on this bar.
    pub rule_refused: Vec<bool>,
    /// The sleeve had a bar (all its instruments priced) on this account bar.
    pub sleeve_open: Vec<bool>,
    /// The sleeve was due (a new target effective, or `EveryBar` policy) and open.
    pub due: Vec<bool>,
    /// The sleeve's instruments were planned (re-targeted) this bar.
    pub planned: Vec<bool>,
    /// Share in force after this bar's review.
    pub share: Vec<f64>,
    /// The sleeve's contribution `SUM_j w_(j,k-1) r_(j,k)` (gross of this bar's cost; shared instruments are split by
    /// the sleeves' standing components).
    pub contrib: Vec<f64>,
    /// Return of the sleeve's shadow (unit-capital, zero-cost) account on this bar; exactly 0.0 where the sleeve has no
    /// own bar and on the shadow's first bar.
    pub shadow_ret_gross: Vec<f64>,
    /// Same, charged this run's cost preset.
    pub shadow_ret_cost: Vec<f64>,
    // [bar * n + j]
    /// Valuation price (carried close where the market is closed; 0.0 before the instrument was first marked).
    pub marks: Vec<f64>,
    pub units: Vec<f64>,
    pub target_weights: Vec<f64>,
    pub held_weights: Vec<f64>,
    /// Notional traded on the bar per instrument.
    pub traded_by_instrument: Vec<f64>,
    pub refusals: Vec<BookRefusal>,
    pub window: Option<Window>,
    /// Per sleeve, per asset of its universe: decision-to-decision sign changes (T1's `signal_flips`).
    pub signal_flips: Vec<Vec<u64>>,
    /// Per instrument: bars on which the position changed.
    pub fills_per_instrument: Vec<u64>,
    pub rebalance_bars: u64,
    pub gross_exposure_stats: Option<ExposureStats>,
    pub net_exposure_stats: Option<ExposureStats>,
    /// Per sleeve.
    pub shadows: Vec<ShadowSeries>,
    /// Account bar at which the overlay halted the book.
    pub halted_at: Option<usize>,
    /// SHA-256 over the canonical series (every column above, bit patterns).
    pub series_sha256: String,
}

impl BookResult {
    pub fn n_bars(&self) -> usize {
        self.times.len()
    }
    pub fn n_instruments(&self) -> usize {
        self.instruments.len()
    }
    pub fn n_sleeves(&self) -> usize {
        self.sleeve_ids.len()
    }
    /// Row `k` of a `[bar * n + j]` matrix.
    pub fn inst_row<'a>(&self, flat: &'a [f64], k: usize) -> &'a [f64] {
        let n = self.n_instruments();
        &flat[k * n..(k + 1) * n]
    }
    /// Row `k` of a `[bar * S + s]` matrix.
    pub fn sleeve_row<'a, T>(&self, flat: &'a [T], k: usize) -> &'a [T] {
        let s = self.n_sleeves();
        &flat[k * s..(k + 1) * s]
    }
    pub fn dates(&self) -> Vec<Date> {
        self.times.iter().map(|t| t.date()).collect()
    }
    pub fn window_dates(&self) -> Vec<Date> {
        match self.window {
            Some(w) => self.times[w.first_bar..=w.last_bar].iter().map(|t| t.date()).collect(),
            None => Vec::new(),
        }
    }
    pub fn window_returns(&self) -> &[f64] {
        match self.window {
            Some(w) => &self.ret[w.first_bar..=w.last_bar],
            None => &[],
        }
    }
    /// `cumprod(1 + r)` over the counted returns.
    pub fn window_equity(&self) -> Vec<f64> {
        cumprod_one_plus(self.window_returns())
    }
    /// Book metrics over the counted window, `account_clock_v1` (`ppy = n / years` on the account clock).
    pub fn metrics(&self) -> Option<Metrics> {
        answer_key_metrics(&self.window_dates(), self.window_returns())
    }
    /// Sleeve `s` on its OWN calendar (`answer_key_v1`): its shadow curve, gross or charged the run's cost preset.
    pub fn sleeve_metrics(&self, s: usize, gross: bool) -> Option<Metrics> {
        let sh = &self.shadows[s];
        answer_key_metrics(&sh.dates, if gross { &sh.gross } else { &sh.cost })
    }
    pub fn total_cost(&self) -> f64 {
        let mut s = 0.0;
        for &c in &self.cost {
            s += c;
        }
        s
    }
    pub fn total_traded_notional(&self) -> f64 {
        let mut s = 0.0;
        for &c in &self.traded_notional {
            s += c;
        }
        s
    }
    /// Refusals of one kind of code (e.g. `"gross_above_cap"`).
    pub fn refusals_with_code(&self, code: &str) -> Vec<&BookRefusal> {
        self.refusals.iter().filter(|r| r.code == code).collect()
    }

    /// The attribution identity (design 3.3), recomputed from the stored columns only:
    /// `ret_k = SUM_j w_(j,k-1) r_(j,k) + financing_k / E_(k-1) - cost_k / E_(k-1)` with `w` the post-trade weights of
    /// the previous bar and `r` the carry-aware close-to-close return of each instrument's valuation price, and the
    /// per-sleeve contributions summing to the total.
    pub fn attribution_report(&self) -> AttributionReport {
        let n = self.n_instruments();
        let s_n = self.n_sleeves();
        let mut e_total = 0.0f64;
        let mut e_sum = 0.0f64;
        for k in 1..self.n_bars() {
            let e_prev = self.equity[k - 1];
            let mut sw = 0.0;
            for j in 0..n {
                let m0 = self.marks[(k - 1) * n + j];
                let m1 = self.marks[k * n + j];
                let r = if m0 > 0.0 { m1 / m0 - 1.0 } else { 0.0 };
                sw += self.held_weights[(k - 1) * n + j] * r;
            }
            let ident = sw + self.financing[k] / e_prev - self.cost[k] / e_prev;
            e_total = e_total.max((self.ret[k] - ident).abs());
            let mut cs = 0.0;
            for s in 0..s_n {
                cs += self.contrib[k * s_n + s];
            }
            e_sum = e_sum.max((cs - sw).abs());
        }
        AttributionReport {
            max_abs_err_total: e_total,
            max_abs_err_contrib_sum: e_sum,
            bars_checked: self.n_bars().saturating_sub(1),
        }
    }

    /// For a book of exactly one sleeve whose universe is every instrument in order: the T1 [`SimResult`] of the same
    /// run, with the same series digest as `simulate` would give (`answer_key_v1` metric label, the T1 view). `None`
    /// for any other book.
    pub fn one_sleeve_sim_result(&self) -> Option<SimResult> {
        let n = self.n_instruments();
        if self.n_sleeves() != 1 || self.sleeve_universes[0] != (0..n).collect::<Vec<_>>() {
            return None;
        }
        let dates = self.dates();
        let mut res = SimResult {
            rule_id: self.sleeve_rules[0].0.clone(),
            rule_impl_version: self.sleeve_rules[0].1.clone(),
            symbols: self.instruments.clone(),
            cost_model_id: self.cost_model_id,
            metric_definitions: METRIC_DEFINITIONS,
            dates: dates.clone(),
            ret: self.ret.clone(),
            ret_pre_cost: self.ret_pre_cost.clone(),
            equity: self.equity.clone(),
            cash: self.cash.clone(),
            cost: self.cost.clone(),
            traded_notional: self.traded_notional.clone(),
            financing: self.financing.clone(),
            gross_exposure: self.gross_exposure.clone(),
            net_exposure: self.net_exposure.clone(),
            decision: self.decision.clone(),
            refused: self.rule_refused.clone(),
            target_weights: self.target_weights.clone(),
            held_weights: self.held_weights.clone(),
            units: self.units.clone(),
            refusals: self
                .refusals
                .iter()
                .filter(|r| r.sleeve == Some(0) && r.code != "gross_above_cap" && r.code != "data_gap")
                .map(|r| Refusal {
                    bar: r.bar,
                    date: r.time.date(),
                    kind: r.kind,
                    code: r.code,
                    message: r.message.clone(),
                })
                .collect(),
            window: self.window,
            signal_flips: self.signal_flips[0].clone(),
            fills_per_asset: self.fills_per_instrument.clone(),
            rebalance_bars: self.rebalance_bars,
            gross_exposure_stats: None,
            net_exposure_stats: None,
            series_sha256: String::new(),
        };
        if let Some(w) = res.window {
            res.gross_exposure_stats = Some(exposure_stats(&res.gross_exposure[w.first_bar..=w.last_bar]));
            res.net_exposure_stats = Some(exposure_stats(&res.net_exposure[w.first_bar..=w.last_bar]));
        }
        res.series_sha256 = digest(&res);
        Some(res)
    }
}

/// Kind of a refusal as text, for diagnostics.
pub fn refusal_kind_name(k: RefusalKind) -> &'static str {
    match k {
        RefusalKind::Warmup => "warmup",
        RefusalKind::Data => "data",
        RefusalKind::Other => "other",
    }
}

pub(crate) fn book_digest(r: &BookResult) -> String {
    let mut h = Sha256::new();
    h.update(b"weightsim-book-series-v1\n");
    for (id, v) in r.sleeve_ids.iter().zip(&r.sleeve_rules) {
        h.update(id.as_bytes());
        h.update(b"|");
        h.update(v.0.as_bytes());
        h.update(b"|");
        h.update(v.1.as_bytes());
        h.update(b";");
    }
    h.update(b"\n");
    for s in &r.instruments {
        h.update(s.as_bytes());
        h.update(b",");
    }
    h.update(b"\n");
    h.update(r.cost_model_id.as_bytes());
    h.update(b"\n");
    h.update(r.metric_definitions.as_bytes());
    h.update(b"\n");
    let n = r.n_instruments();
    let s_n = r.n_sleeves();
    let f = |h: &mut Sha256, v: f64| h.update(&v.to_bits().to_be_bytes());
    for k in 0..r.n_bars() {
        h.update(&r.times[k].ms().to_be_bytes());
        h.update(&(r.clock_index[k] as u64).to_be_bytes());
        for col in [
            &r.ret,
            &r.ret_pre_cost,
            &r.equity,
            &r.equity_pre,
            &r.cash,
            &r.cost,
            &r.traded_notional,
            &r.financing,
            &r.gross_exposure,
            &r.net_exposure,
            &r.risk_scale,
        ] {
            f(&mut h, col[k]);
        }
        h.update(&[u8::from(r.run[k]), u8::from(r.book_refused[k]), u8::from(r.halted[k])]);
        for s in 0..s_n {
            let i = k * s_n + s;
            h.update(&[
                u8::from(r.decision[i]),
                u8::from(r.rule_refused[i]),
                u8::from(r.sleeve_open[i]),
                u8::from(r.due[i]),
                u8::from(r.planned[i]),
            ]);
            for col in [&r.share, &r.contrib, &r.shadow_ret_gross, &r.shadow_ret_cost] {
                f(&mut h, col[i]);
            }
        }
        for j in 0..n {
            let i = k * n + j;
            for col in [&r.marks, &r.units, &r.target_weights, &r.held_weights, &r.traded_by_instrument] {
                f(&mut h, col[i]);
            }
        }
    }
    h.update(b"refusals\n");
    for rf in &r.refusals {
        h.update(&(rf.bar as u64).to_be_bytes());
        h.update(&rf.sleeve.map_or(u64::MAX, |s| s as u64).to_be_bytes());
        h.update(rf.code.as_bytes());
        h.update(b";");
    }
    to_hex(&h.finalize())
}
