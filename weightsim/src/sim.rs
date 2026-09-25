//! The simulator (design 2.1). One function, [`simulate`], generic over a [`WeightRule`].
//!
//! Per-bar algorithm (identical to the design's pseudo-code; every line below is numbered so tests can cite it):
//!
//! ```text
//! for t in calendar:
//!    (1) if t > 0: accrue financing on the (t-1)-close notionals; equity_pre = cash + sum(units[i] * P[i][t])
//!        else     : equity_pre = cash
//!    (2) if decision_bar(t) and t+1 >= min_history_bars: ask the rule with a view of bars 0..=t
//!            Ok(w)  -> scaled = w * risk_scale; refuse-not-clip max_gross check; effective at bar t + delay
//!            Err(e) -> Abort (fail) or HoldPrevious (record, standing target untouched)
//!    (3) pending targets whose effective bar is t become the standing target
//!    (4) rebalance if due (OnDecision: only on the bar a target became effective; EveryBar: every bar with a target):
//!            units'[i] = target[i] * equity_pre / P[i][t]
//!            traded    = sum |units'[i] - units[i]| * P[i][t]
//!            cost      = traded * rate
//!            cash      = equity_pre - sum(units'[i] * P[i][t]) - cost
//!    (5) equity[t] = equity_pre - cost;  ret_pre_cost[t] = equity_pre / equity[t-1] - 1;  ret[t] = equity[t] / equity[t-1] - 1
//! ```
//!
//! Only `+ - * /` and `sqrt` are used, all sums are sequential left-to-right, there is no parallelism and no libm call
//! in this file, so results are bit-reproducible across x86 and ARM.

use crate::costs::{CostModel, Financing};
use crate::date::Date;
use crate::metrics::{
    answer_key_metrics, cumprod_one_plus, median, percentile_nearest_rank, Metrics, METRIC_DEFINITIONS,
};
use crate::panel::{HistoryView, Panel};
use crate::rule::{OnRefusal, RebalancePolicy, RefusalKind, RuleRefusal, WeightRule};
use crate::sha256::{to_hex, Sha256};
use std::collections::VecDeque;
use std::fmt;

/// Run configuration. There is deliberately nothing here that a rule can override.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// First date whose return is counted (the rule still sees the earlier warm-up history). `None` = as soon as the
    /// first target is effective.
    pub start: Option<Date>,
    /// Last date whose return is counted. `None` = end of panel.
    pub end: Option<Date>,
    /// Starting capital (returns are scale-free, so 1.0 is the convention).
    pub initial_equity: f64,
    pub cost: CostModel,
    pub financing: Financing,
    pub on_refusal: OnRefusal,
    /// Bars between a decision and its fill. 0 reproduces the key (design S-6); 1 is the `delay1` sensitivity.
    pub execution_delay_bars: usize,
    /// Constant multiplier applied to every target (design 2.7, R2). 1.0 in replication.
    pub risk_scale: f64,
    /// Refuse-not-clip cap on sum |weight| (design S-10). `None` in replication.
    pub max_gross: Option<f64>,
}

impl Default for SimConfig {
    /// The replication configuration: zero cost, no financing, `Abort` on refusal, no delay, no scaling, no cap.
    fn default() -> Self {
        SimConfig {
            start: None,
            end: None,
            initial_equity: 1.0,
            cost: CostModel::ZERO,
            financing: Financing::None,
            on_refusal: OnRefusal::Abort,
            execution_delay_bars: 0,
            risk_scale: 1.0,
            max_gross: None,
        }
    }
}

impl SimConfig {
    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if !(self.initial_equity.is_finite() && self.initial_equity > 0.0) {
            return Err(SimError::BadConfig("initial_equity must be finite and > 0".into()));
        }
        if !self.cost.is_valid() {
            return Err(SimError::BadConfig("cost components must be finite and >= 0".into()));
        }
        if !self.risk_scale.is_finite() {
            return Err(SimError::BadConfig("risk_scale must be finite".into()));
        }
        if let Some(g) = self.max_gross {
            if !(g.is_finite() && g > 0.0) {
                return Err(SimError::BadConfig("max_gross must be finite and > 0".into()));
            }
        }
        if let (Some(s), Some(e)) = (self.start, self.end) {
            if e < s {
                return Err(SimError::BadConfig("end before start".into()));
            }
        }
        Ok(())
    }
}

/// Reasons a run fails.
#[derive(Clone, Debug, PartialEq)]
pub enum SimError {
    BadConfig(String),
    UniverseMismatch { rule: Vec<String>, panel: Vec<String> },
    RuleRefused { date: Date, refusal: RuleRefusal },
    InvalidWeights { date: Date, reason: String },
    MaxGrossBreached { date: Date, gross: f64, limit: f64 },
    NonPositiveEquity { date: Date, equity: f64 },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::BadConfig(m) => write!(f, "bad config: {m}"),
            SimError::UniverseMismatch { rule, panel } => {
                write!(f, "rule universe {rule:?} does not match panel symbols {panel:?}")
            }
            SimError::RuleRefused { date, refusal } => write!(f, "rule refused on {date}: {refusal}"),
            SimError::InvalidWeights { date, reason } => write!(f, "invalid weights on {date}: {reason}"),
            SimError::MaxGrossBreached { date, gross, limit } => {
                write!(f, "gross exposure {gross} exceeds limit {limit} on {date} (refuse, not clip)")
            }
            SimError::NonPositiveEquity { date, equity } => write!(f, "equity {equity} <= 0 on {date}"),
        }
    }
}

impl std::error::Error for SimError {}

/// A refusal recorded during a run (design S-5).
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub bar: usize,
    pub date: Date,
    pub kind: RefusalKind,
    pub code: &'static str,
    pub message: String,
}

/// Inclusive bar range whose returns are counted in the metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub first_bar: usize,
    pub last_bar: usize,
}

/// Distribution of an exposure series over the counted window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureStats {
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}

/// Full-resolution, column-oriented result. Flat matrices are `[bar * n_assets + asset]`.
#[derive(Clone, Debug)]
pub struct SimResult {
    pub rule_id: String,
    pub rule_impl_version: String,
    pub symbols: Vec<String>,
    pub cost_model_id: &'static str,
    pub metric_definitions: &'static str,
    pub dates: Vec<Date>,
    /// Realised return dated t (post-cost in this run). `ret[0] == 0`.
    pub ret: Vec<f64>,
    /// Return dated t before this bar's rebalance cost.
    pub ret_pre_cost: Vec<f64>,
    /// Post-cost equity at the close of t.
    pub equity: Vec<f64>,
    pub cash: Vec<f64>,
    /// Cost paid at the close of t.
    pub cost: Vec<f64>,
    /// Traded notional at the close of t.
    pub traded_notional: Vec<f64>,
    /// Financing accrued into cash over (t-1, t].
    pub financing: Vec<f64>,
    pub gross_exposure: Vec<f64>,
    pub net_exposure: Vec<f64>,
    /// A decision succeeded on this bar.
    pub decision: Vec<bool>,
    /// The rule refused on this bar.
    pub refused: Vec<bool>,
    /// Standing (effective) target after step (3); zeros while no target exists.
    pub target_weights: Vec<f64>,
    /// Post-trade held weights `units * P / equity`.
    pub held_weights: Vec<f64>,
    /// Post-trade units.
    pub units: Vec<f64>,
    pub refusals: Vec<Refusal>,
    pub window: Option<Window>,
    /// Per asset: decision-to-decision changes of sign(target) among successful decisions dated in
    /// `[max(start, first decision), end]`, excluding the first (the key's `signal_flips`).
    pub signal_flips: Vec<u64>,
    /// Per asset: number of bars on which the position changed.
    pub fills_per_asset: Vec<u64>,
    pub rebalance_bars: u64,
    pub gross_exposure_stats: Option<ExposureStats>,
    pub net_exposure_stats: Option<ExposureStats>,
    /// SHA-256 over the canonical series (all columns above, bit patterns). Two runs must give the same digest.
    pub series_sha256: String,
}

impl SimResult {
    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }
    pub fn n_bars(&self) -> usize {
        self.dates.len()
    }
    /// Row `t` of a flat `[bar * n_assets + asset]` matrix.
    pub fn row<'a>(&self, flat: &'a [f64], t: usize) -> &'a [f64] {
        let k = self.n_assets();
        &flat[t * k..(t + 1) * k]
    }
    pub fn window_dates(&self) -> &[Date] {
        match self.window {
            Some(w) => &self.dates[w.first_bar..=w.last_bar],
            None => &[],
        }
    }
    pub fn window_returns(&self) -> &[f64] {
        match self.window {
            Some(w) => &self.ret[w.first_bar..=w.last_bar],
            None => &[],
        }
    }
    /// `cumprod(1 + r)` over the counted returns (no initial point), i.e. equity rebased to 1.0 at the window start.
    pub fn window_equity(&self) -> Vec<f64> {
        cumprod_one_plus(self.window_returns())
    }
    /// `answer_key_v1` metrics over the counted window.
    pub fn metrics(&self) -> Option<Metrics> {
        answer_key_metrics(self.window_dates(), self.window_returns())
    }
    /// Total cost over the whole run, summed sequentially.
    pub fn total_cost(&self) -> f64 {
        let mut s = 0.0;
        for &c in &self.cost {
            s += c;
        }
        s
    }
    /// Total traded notional over the whole run, summed sequentially.
    pub fn total_traded_notional(&self) -> f64 {
        let mut s = 0.0;
        for &c in &self.traded_notional {
            s += c;
        }
        s
    }
}

pub(crate) fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

pub(crate) fn exposure_stats(x: &[f64]) -> ExposureStats {
    let mut s = 0.0;
    let mut mx = f64::NEG_INFINITY;
    for &v in x {
        s += v;
        if v > mx {
            mx = v;
        }
    }
    ExposureStats { mean: s / x.len() as f64, median: median(x), p90: percentile_nearest_rank(x, 0.9), max: mx }
}

/// Run `rule` over `panel` under `cfg`.
pub fn simulate<R: WeightRule + ?Sized>(panel: &Panel, rule: &R, cfg: &SimConfig) -> Result<SimResult, SimError> {
    cfg.validate()?;
    let k = panel.n_assets();
    let n = panel.n_bars();
    let universe = rule.universe();
    if universe.len() != k || universe.iter().zip(panel.symbols()).any(|(a, b)| *a != b.as_str()) {
        return Err(SimError::UniverseMismatch {
            rule: universe.iter().map(|s| (*s).to_string()).collect(),
            panel: panel.symbols().to_vec(),
        });
    }
    let schedule = rule.decision_schedule();
    let policy = rule.rebalance_policy();
    let min_hist = rule.min_history_bars();
    let dates = panel.dates();
    let rate = cfg.cost.rate();

    let mut units = vec![0.0f64; k];
    let mut cash = cfg.initial_equity;
    let mut equity_prev = cfg.initial_equity;
    let mut standing: Option<Vec<f64>> = None;
    let mut first_effective: Option<usize> = None;
    let mut decided_any = false;
    let mut pending: VecDeque<(usize, Vec<f64>)> = VecDeque::new();
    let mut decision_bars: Vec<usize> = Vec::new();
    let mut decision_signs: Vec<Vec<i8>> = Vec::new();

    let mut res = SimResult {
        rule_id: rule.id().to_string(),
        rule_impl_version: rule.impl_version(),
        symbols: panel.symbols().to_vec(),
        cost_model_id: cfg.cost.id,
        metric_definitions: METRIC_DEFINITIONS,
        dates: dates.to_vec(),
        ret: Vec::with_capacity(n),
        ret_pre_cost: Vec::with_capacity(n),
        equity: Vec::with_capacity(n),
        cash: Vec::with_capacity(n),
        cost: Vec::with_capacity(n),
        traded_notional: Vec::with_capacity(n),
        financing: Vec::with_capacity(n),
        gross_exposure: Vec::with_capacity(n),
        net_exposure: Vec::with_capacity(n),
        decision: Vec::with_capacity(n),
        refused: Vec::with_capacity(n),
        target_weights: Vec::with_capacity(n * k),
        held_weights: Vec::with_capacity(n * k),
        units: Vec::with_capacity(n * k),
        refusals: Vec::new(),
        window: None,
        signal_flips: vec![0; k],
        fills_per_asset: vec![0; k],
        rebalance_bars: 0,
        gross_exposure_stats: None,
        net_exposure_stats: None,
        series_sha256: String::new(),
    };

    for t in 0..n {
        let date = dates[t];

        // (1) mark to market, financing first.
        let mut fin = 0.0;
        let equity_pre;
        if t == 0 {
            equity_pre = cash;
        } else {
            let mut long_v = 0.0;
            let mut short_v = 0.0;
            for i in 0..k {
                let v = units[i] * panel.closes(i)[t - 1];
                if v > 0.0 {
                    long_v += v;
                } else {
                    short_v += -v;
                }
            }
            fin = cfg.financing.accrual(dates[t - 1].days_until(date), cash, long_v, short_v);
            cash += fin;
            let mut invested = 0.0;
            for i in 0..k {
                invested += units[i] * panel.closes(i)[t];
            }
            equity_pre = cash + invested;
        }
        if !(equity_pre > 0.0) {
            return Err(SimError::NonPositiveEquity { date, equity: equity_pre });
        }

        // (2) decision.
        let mut decision_ok = false;
        let mut refused = false;
        if schedule.is_decision_bar(dates, t) && t + 1 >= min_hist {
            let view = HistoryView::new(panel, t);
            match rule.target_weights(&view) {
                Ok(w) => {
                    if w.len() != k {
                        return Err(SimError::InvalidWeights {
                            date,
                            reason: format!("{} weights for {} assets", w.len(), k),
                        });
                    }
                    if let Some(bad) = w.iter().position(|v| !v.is_finite()) {
                        return Err(SimError::InvalidWeights { date, reason: format!("weight {bad} is not finite") });
                    }
                    let scaled: Vec<f64> = w.iter().map(|v| v * cfg.risk_scale).collect();
                    if let Some(limit) = cfg.max_gross {
                        let mut gross = 0.0;
                        for v in &scaled {
                            gross += v.abs();
                        }
                        if gross > limit * (1.0 + 1e-12) {
                            return Err(SimError::MaxGrossBreached { date, gross, limit });
                        }
                    }
                    decided_any = true;
                    decision_ok = true;
                    decision_bars.push(t);
                    decision_signs.push(scaled.iter().map(|v| sign(*v)).collect());
                    pending.push_back((t + cfg.execution_delay_bars, scaled));
                }
                Err(refusal) => {
                    let tolerated_warmup = refusal.kind == RefusalKind::Warmup && !decided_any;
                    if cfg.on_refusal == OnRefusal::Abort && !tolerated_warmup {
                        return Err(SimError::RuleRefused { date, refusal });
                    }
                    refused = true;
                    res.refusals.push(Refusal {
                        bar: t,
                        date,
                        kind: refusal.kind,
                        code: refusal.code,
                        message: refusal.message,
                    });
                }
            }
        }

        // (3) targets that become effective at this bar.
        let mut newly_effective = false;
        while let Some(front) = pending.front() {
            if front.0 == t {
                let (_, w) = pending.pop_front().expect("front exists");
                standing = Some(w);
                newly_effective = true;
                if first_effective.is_none() {
                    first_effective = Some(t);
                }
            } else {
                break;
            }
        }

        // (4) rebalance.
        let due = standing.is_some()
            && match policy {
                RebalancePolicy::OnDecision => newly_effective,
                RebalancePolicy::EveryBar => true,
            };
        let mut traded = 0.0;
        let mut cost = 0.0;
        if due {
            let w = standing.as_ref().expect("due implies a standing target");
            let mut any_fill = false;
            for i in 0..k {
                let p = panel.closes(i)[t];
                let nu = w[i] * equity_pre / p;
                traded += (nu - units[i]).abs() * p;
                if nu != units[i] {
                    res.fills_per_asset[i] += 1;
                    any_fill = true;
                }
                units[i] = nu;
            }
            if any_fill {
                res.rebalance_bars += 1;
            }
            let mut invested = 0.0;
            for i in 0..k {
                invested += units[i] * panel.closes(i)[t];
            }
            cost = traded * rate;
            cash = equity_pre - invested - cost;
        }

        // (5) equity and returns.
        let equity = equity_pre - cost;
        if !(equity > 0.0) {
            return Err(SimError::NonPositiveEquity { date, equity });
        }
        let (ret_pre, ret) =
            if t == 0 { (0.0, 0.0) } else { (equity_pre / equity_prev - 1.0, equity / equity_prev - 1.0) };

        let mut gross_e = 0.0;
        let mut net_e = 0.0;
        for i in 0..k {
            let hw = units[i] * panel.closes(i)[t] / equity;
            gross_e += hw.abs();
            net_e += hw;
            res.held_weights.push(hw);
            res.units.push(units[i]);
            res.target_weights.push(standing.as_ref().map_or(0.0, |w| w[i]));
        }

        res.ret.push(ret);
        res.ret_pre_cost.push(ret_pre);
        res.equity.push(equity);
        res.cash.push(cash);
        res.cost.push(cost);
        res.traded_notional.push(traded);
        res.financing.push(fin);
        res.gross_exposure.push(gross_e);
        res.net_exposure.push(net_e);
        res.decision.push(decision_ok);
        res.refused.push(refused);
        equity_prev = equity;
    }

    // Counted window (design S-7): returns dated in [start, end], from the bar after the first fill.
    let start_idx = match cfg.start {
        Some(s) => dates.iter().position(|&d| d >= s),
        None => Some(0),
    };
    let end_idx = match cfg.end {
        Some(e) => dates.iter().rposition(|&d| d <= e),
        None => Some(n - 1),
    };
    if let (Some(fe), Some(si), Some(ei)) = (first_effective, start_idx, end_idx) {
        let i0 = (fe + 1).max(si);
        if i0 <= ei {
            res.window = Some(Window { first_bar: i0, last_bar: ei });
        }
    }

    // Signal flips (the key's counter).
    if let Some(&first_dec) = decision_bars.first() {
        let from = match cfg.start {
            Some(s) => s.max(dates[first_dec]),
            None => dates[first_dec],
        };
        let mut prev: Option<&Vec<i8>> = None;
        for (bar, signs) in decision_bars.iter().zip(&decision_signs) {
            let d = dates[*bar];
            if d < from || cfg.end.is_some_and(|e| d > e) {
                continue;
            }
            if let Some(p) = prev {
                for i in 0..k {
                    if p[i] != signs[i] {
                        res.signal_flips[i] += 1;
                    }
                }
            }
            prev = Some(signs);
        }
    }

    if let Some(w) = res.window {
        res.gross_exposure_stats = Some(exposure_stats(&res.gross_exposure[w.first_bar..=w.last_bar]));
        res.net_exposure_stats = Some(exposure_stats(&res.net_exposure[w.first_bar..=w.last_bar]));
    }

    res.series_sha256 = digest(&res);
    Ok(res)
}

/// Run the same decisions twice (design S-11): `gross` under `CostModel::ZERO` and `Financing::None`, `net` under
/// `cfg` as given. They are two executions of the same code, not a decomposition, because costs change equity and
/// therefore unit counts.
pub fn simulate_gross_and_net<R: WeightRule + ?Sized>(
    panel: &Panel,
    rule: &R,
    cfg: &SimConfig,
) -> Result<(SimResult, SimResult), SimError> {
    let mut gross_cfg = cfg.clone();
    gross_cfg.cost = CostModel::ZERO;
    gross_cfg.financing = Financing::None;
    let gross = simulate(panel, rule, &gross_cfg)?;
    let net = simulate(panel, rule, cfg)?;
    Ok((gross, net))
}

pub(crate) fn digest(r: &SimResult) -> String {
    let mut h = Sha256::new();
    h.update(b"weightsim-series-v1\n");
    h.update(r.rule_id.as_bytes());
    h.update(b"\n");
    h.update(r.rule_impl_version.as_bytes());
    h.update(b"\n");
    for s in &r.symbols {
        h.update(s.as_bytes());
        h.update(b",");
    }
    h.update(b"\n");
    h.update(r.cost_model_id.as_bytes());
    h.update(b"\n");
    h.update(r.metric_definitions.as_bytes());
    h.update(b"\n");
    let k = r.n_assets();
    let f = |h: &mut Sha256, v: f64| h.update(&v.to_bits().to_be_bytes());
    for t in 0..r.n_bars() {
        let d = r.dates[t];
        h.update(&d.year().to_be_bytes());
        h.update(&[d.month(), d.day()]);
        for col in [
            &r.ret,
            &r.ret_pre_cost,
            &r.equity,
            &r.cash,
            &r.cost,
            &r.traded_notional,
            &r.financing,
            &r.gross_exposure,
            &r.net_exposure,
        ] {
            f(&mut h, col[t]);
        }
        h.update(&[u8::from(r.decision[t]), u8::from(r.refused[t])]);
        for flat in [&r.target_weights, &r.held_weights, &r.units] {
            for i in 0..k {
                f(&mut h, flat[t * k + i]);
            }
        }
    }
    to_hex(&h.finalize())
}
