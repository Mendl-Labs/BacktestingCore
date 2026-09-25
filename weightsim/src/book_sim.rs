//! The book simulator (design 3.3): one joint account on the union clock, several sleeves, shares, an allocator,
//! per-sleeve shadow curves, attribution, an overlay hook, and the legacy `IndependentSubAccounts` emulation.
//!
//! Per account bar `k` (clock index `u`), in this order (every step is numbered so tests and the mutation list can
//! cite it):
//!
//! ```text
//! (1) availability: which sleeves have a bar (all instruments priced); a missing bar of an OPEN market is a data
//!     gap (refusal / error), a declared closure is a carry
//! (2) marks: an instrument is valued at its close when an owning sleeve has a bar, else at its carried last close
//! (3) shadows: every sleeve's unit-capital accounts (gross and cost) mark to market on the sleeve's OWN bars
//! (4) joint account: financing, E_pre = cash + SUM units*mark (no trading yet); contributions of the previous bar's
//!     weights; the overlay steps on E_pre and scales the risk
//! (5) decisions: each sleeve with an own bar asks its rule (own-calendar schedule and history), targets become
//!     effective after `execution_delay_bars` own bars
//! (6) allocator review (calendar month-end of the account clock): shares from past shadow returns only
//! (7) cadence: due sleeves, driver run, planned sleeves
//! (8) construction (crate::construct) -> units, cost; a whole-book refusal trades nothing
//! (9) shadows trade, everything is recorded
//! ```
//!
//! Only `+ - * /` and `sqrt` are used, all sums are sequential, no parallelism: results are bit-reproducible.

use crate::bartime::BarTime;
use crate::book::{
    AccountMode, AllocatorSpec, Book, BookCadence, BookConfig, BookError, BookRefusal, OverlayInput, ShareSpec,
};
use crate::book_panel::{Availability, BookPanel, SleeveCalendar};
use crate::book_result::{book_digest, BookResult, ShadowSeries, BOOK_METRIC_DEFINITIONS};
use crate::construct::{
    CashPolicy, Construct, ConstructInputs, ConstructPolicy, ConstructRefusal, MinimalConstruct, SleeveTarget,
};
use crate::costs::{CostModel, Financing};
use crate::date::Date;
use crate::panel::{HistoryView, Panel};
use crate::rule::{DecisionSchedule, OnRefusal, RebalancePolicy, RefusalKind};
use crate::sim::{exposure_stats, sign, SimError, Window};
use crate::stateful::{DataNeed, RuleRun, VolScaling};
use std::collections::VecDeque;

const NO_OWN: usize = usize::MAX;

/// Run `book` over `panel` under `cfg` with the minimal construction ([`MinimalConstruct`]).
pub fn simulate_book(panel: &BookPanel, book: &Book, cfg: &BookConfig) -> Result<BookResult, BookError> {
    simulate_book_with(panel, book, cfg, &MinimalConstruct)
}

/// Design S-11 for books: `gross` under `CostModel::ZERO` and `Financing::None`, `net` under `cfg` as given; two
/// executions of the same code, because costs change equity and therefore unit counts.
pub fn simulate_book_gross_and_net(
    panel: &BookPanel,
    book: &Book,
    cfg: &BookConfig,
) -> Result<(BookResult, BookResult), BookError> {
    let mut gross_cfg = cfg.clone();
    gross_cfg.sim.cost = CostModel::ZERO;
    gross_cfg.sim.financing = Financing::None;
    let gross = simulate_book(panel, book, &gross_cfg)?;
    let net = simulate_book(panel, book, cfg)?;
    Ok((gross, net))
}

/// One unit-capital account of a sleeve on its own calendar.
struct Shadow {
    rate: f64,
    financing: Financing,
    units: Vec<f64>,
    cash: f64,
    equity: f64,
    e_prev: f64,
    started: bool,
    // scratch of the current bar
    equity_pre: f64,
    first: bool,
}

impl Shadow {
    fn new(k: usize, rate: f64, financing: Financing) -> Shadow {
        Shadow {
            rate,
            financing,
            units: vec![0.0; k],
            cash: 1.0,
            equity: 1.0,
            e_prev: 1.0,
            started: false,
            equity_pre: 1.0,
            first: false,
        }
    }

    /// Mark to market at own bar `t`. Returns `(financing credit, return before cost)`; the return is 0 on the first bar.
    fn mark(&mut self, panel: &Panel, t: usize) -> Result<(f64, f64), ()> {
        let k = self.units.len();
        let mut fin = 0.0;
        if !self.started {
            self.started = true;
            self.first = true;
            self.equity_pre = self.cash;
        } else {
            self.first = false;
            let mut long_v = 0.0;
            let mut short_v = 0.0;
            for i in 0..k {
                let v = self.units[i] * panel.closes(i)[t - 1];
                if v > 0.0 {
                    long_v += v;
                } else {
                    short_v += -v;
                }
            }
            fin = self.financing.accrual(panel.dates()[t - 1].days_until(panel.dates()[t]), self.cash, long_v, short_v);
            self.cash += fin;
            let mut invested = 0.0;
            for i in 0..k {
                invested += self.units[i] * panel.closes(i)[t];
            }
            self.equity_pre = self.cash + invested;
        }
        if !(self.equity_pre > 0.0) {
            return Err(());
        }
        let ret_pre = if self.first { 0.0 } else { self.equity_pre / self.e_prev - 1.0 };
        Ok((fin, ret_pre))
    }

    /// Trade to `w` (the sleeve's own weights, fractions of the shadow's equity) at own bar `t`. Returns
    /// `(traded notional, cost)`.
    fn trade(&mut self, panel: &Panel, t: usize, w: &[f64]) -> (f64, f64) {
        let k = self.units.len();
        let mut traded = 0.0;
        for i in 0..k {
            let p = panel.closes(i)[t];
            let nu = w[i] * self.equity_pre / p;
            traded += (nu - self.units[i]).abs() * p;
            self.units[i] = nu;
        }
        let mut invested = 0.0;
        for i in 0..k {
            invested += self.units[i] * panel.closes(i)[t];
        }
        let cost = traded * self.rate;
        self.cash = self.equity_pre - invested - cost;
        (traded, cost)
    }

    /// Close the bar: equity is post-cost; returns the post-cost return (0 on the first bar).
    fn finish(&mut self, cost: f64) -> f64 {
        self.equity = self.equity_pre - cost;
        let r = if self.first { 0.0 } else { self.equity / self.e_prev - 1.0 };
        self.e_prev = self.equity;
        r
    }
}

struct SleeveRt<'a> {
    cal: SleeveCalendar,
    own_of_union: Vec<usize>,
    run: Box<dyn RuleRun + 'a>,
    schedule: DecisionSchedule,
    policy: RebalancePolicy,
    min_hist: usize,
    standing: Option<Vec<f64>>,
    pending: VecDeque<(usize, Vec<f64>)>,
    decided_any: bool,
    share: f64,
    decision_bars: Vec<usize>,
    decision_signs: Vec<Vec<i8>>,
    sh_g: Shadow,
    sh_c: Shadow,
    sh_dates: Vec<Date>,
    sh_gross: Vec<f64>,
    sh_cost: Vec<f64>,
    first_effective: Option<usize>,
}

fn month_end_review(times: &[BarTime], u: usize) -> bool {
    u + 1 == times.len() || !times[u + 1].date().same_month(times[u].date())
}

/// Run `book` with a caller-supplied construction (the PF2 boundary, see [`crate::construct`]).
pub fn simulate_book_with(
    panel: &BookPanel,
    book: &Book,
    cfg: &BookConfig,
    construct: &dyn Construct,
) -> Result<BookResult, BookError> {
    cfg.sim.validate()?;
    let n = panel.n_instruments();
    let s_n = book.sleeves.len();
    if s_n == 0 {
        return Err(BookError::BadBook("a book needs at least one sleeve".into()));
    }
    // ---- validation of the book
    for (a, sl) in book.sleeves.iter().enumerate() {
        if sl.id.is_empty() || book.sleeves[..a].iter().any(|o| o.id == sl.id) {
            return Err(BookError::BadBook(format!("sleeve id `{}` is empty or duplicated", sl.id)));
        }
        if sl.universe.is_empty() {
            return Err(BookError::BadBook(format!("sleeve {} has an empty universe", sl.id)));
        }
        for (b, &i) in sl.universe.iter().enumerate() {
            if i >= n || sl.universe[..b].contains(&i) {
                return Err(BookError::BadBook(format!(
                    "sleeve {}: universe index {i} out of range or repeated",
                    sl.id
                )));
            }
        }
        let want: Vec<&str> = sl.rule.universe().to_vec();
        let have: Vec<&str> = sl.universe.iter().map(|&i| panel.instruments()[i].as_str()).collect();
        if want != have {
            return Err(BookError::Sim(SimError::UniverseMismatch {
                rule: want.iter().map(|s| (*s).to_string()).collect(),
                panel: have.iter().map(|s| (*s).to_string()).collect(),
            }));
        }
        if sl.rule.data_need() == DataNeed::AvailabilityMasked {
            return Err(BookError::Unsupported(format!(
                "sleeve {}: DataNeed::AvailabilityMasked is not implemented in weightsim 0.2 (needs the masked history view of PF4)",
                sl.id
            )));
        }
        let init = sl.share.initial();
        if !(init.is_finite() && init > 0.0) {
            return Err(BookError::BadBook(format!("sleeve {}: share must be finite and > 0", sl.id)));
        }
        match (sl.share, book.allocator) {
            (ShareSpec::Fixed(_), AllocatorSpec::Fixed) => {}
            (ShareSpec::Allocated { .. }, AllocatorSpec::InverseVol { .. }) => {}
            (ShareSpec::Fixed(_), AllocatorSpec::InverseVol { .. }) => {
                return Err(BookError::BadBook(format!(
                    "sleeve {}: a Fixed share under an InverseVol allocator",
                    sl.id
                )))
            }
            (ShareSpec::Allocated { .. }, AllocatorSpec::Fixed) => {
                return Err(BookError::BadBook(format!("sleeve {}: an Allocated share needs an allocator", sl.id)))
            }
        }
        if sl.rule.vol_scaling() == VolScaling::Internal && matches!(book.allocator, AllocatorSpec::InverseVol { .. }) {
            return Err(BookError::BadBook(format!(
                "sleeve {} scales its own volatility (VolScaling::Internal); a volatility-based allocator would scale it twice",
                sl.id
            )));
        }
    }
    if let AllocatorSpec::InverseVol { lookback_bars, total } = book.allocator {
        if lookback_bars < 2 || !(total.is_finite() && total > 0.0) {
            return Err(BookError::BadBook("InverseVol needs lookback_bars >= 2 and a finite total > 0".into()));
        }
    }
    if let Some(a) = cfg.allocated_capital {
        if !(a.is_finite() && a > 0.0) {
            return Err(BookError::Sim(SimError::BadConfig("allocated_capital must be finite and > 0".into())));
        }
    }
    if let Some(f) = cfg.trade_filter {
        if !(f.min_abs.is_finite() && f.min_abs >= 0.0 && f.min_pct.is_finite() && f.min_pct >= 0.0) {
            return Err(BookError::Sim(SimError::BadConfig("trade_filter thresholds must be finite and >= 0".into())));
        }
    }
    let joint = cfg.mode == AccountMode::Joint;
    if !joint {
        let mut why = Vec::new();
        if cfg.allocated_capital.is_some() {
            why.push("allocated_capital");
        }
        if cfg.trade_filter.is_some() {
            why.push("trade_filter");
        }
        if cfg.sim.max_gross.is_some() {
            why.push("max_gross");
        }
        if book.overlay.is_some() {
            why.push("overlay");
        }
        if book.allocator != AllocatorSpec::Fixed {
            why.push("a non-Fixed allocator");
        }
        if cfg.cadence != BookCadence::PerSleeve {
            why.push("cadence AllSleevesOnAnyDue");
        }
        if cfg.cash_policy != CashPolicy::Certification {
            why.push("CashPolicy::Budget");
        }
        if cfg.trade_on_closed_market {
            why.push("trade_on_closed_market");
        }
        if !why.is_empty() {
            return Err(BookError::Unsupported(format!(
                "AccountMode::IndependentSubAccounts (legacy emulation) has no joint construction: {}",
                why.join(", ")
            )));
        }
    }

    // ---- clock and calendars
    let times = panel.times();
    let n_union = times.len();
    let a0 = match cfg.account_start {
        None => 0,
        Some(t) => times
            .iter()
            .position(|x| *x >= t)
            .ok_or_else(|| BookError::BadBook("account_start is after the last bar".into()))?,
    };
    let nb = n_union - a0;
    let rate = cfg.sim.cost.rate();

    let mut sl: Vec<SleeveRt<'_>> = Vec::with_capacity(s_n);
    for spec in &book.sleeves {
        let cal = panel.sleeve_calendar(&spec.universe)?;
        let mut own_of_union = vec![NO_OWN; n_union];
        for (t, &u) in cal.union_index.iter().enumerate() {
            own_of_union[u] = t;
        }
        let k = spec.universe.len();
        sl.push(SleeveRt {
            cal,
            own_of_union,
            run: spec.rule.start(),
            schedule: spec.rule.decision_schedule(),
            policy: spec.effective_policy(),
            min_hist: spec.rule.min_history_bars(),
            standing: None,
            pending: VecDeque::new(),
            decided_any: false,
            share: spec.share.initial(),
            decision_bars: Vec::new(),
            decision_signs: Vec::new(),
            sh_g: Shadow::new(k, 0.0, Financing::None),
            sh_c: Shadow::new(k, rate, cfg.sim.financing),
            sh_dates: Vec::new(),
            sh_gross: Vec::new(),
            sh_cost: Vec::new(),
            first_effective: None,
        });
    }
    let owners: Vec<Vec<usize>> =
        (0..n).map(|j| (0..s_n).filter(|&s| book.sleeves[s].universe.contains(&j)).collect()).collect();
    let mut overlay_run = book.overlay.as_ref().map(|o| o.start());

    // ---- state
    let e0 = cfg.sim.initial_equity;
    let mut units = vec![0.0f64; n];
    let mut cash = e0;
    let mut mark = vec![0.0f64; n];
    let mut marked = vec![false; n];
    let mut e_prev = e0;
    let mut held_prev = vec![0.0f64; n];
    let mut halted = false;
    let mut halted_at: Option<usize> = None;
    let mut fills = vec![0u64; n];
    let mut rebalance_bars = 0u64;
    let on_refusal = cfg.sim.on_refusal;
    let delay = cfg.sim.execution_delay_bars;

    // ---- outputs
    let mut res = BookResult {
        instruments: panel.instruments().to_vec(),
        sleeve_ids: book.sleeves.iter().map(|s| s.id.clone()).collect(),
        sleeve_rules: book.sleeves.iter().map(|s| (s.rule.id().to_string(), s.rule.impl_version())).collect(),
        sleeve_universes: book.sleeves.iter().map(|s| s.universe.clone()).collect(),
        cost_model_id: cfg.sim.cost.id,
        metric_definitions: BOOK_METRIC_DEFINITIONS,
        times: Vec::with_capacity(nb),
        clock_index: Vec::with_capacity(nb),
        ret: Vec::with_capacity(nb),
        ret_pre_cost: Vec::with_capacity(nb),
        equity: Vec::with_capacity(nb),
        equity_pre: Vec::with_capacity(nb),
        cash: Vec::with_capacity(nb),
        cost: Vec::with_capacity(nb),
        traded_notional: Vec::with_capacity(nb),
        financing: Vec::with_capacity(nb),
        gross_exposure: Vec::with_capacity(nb),
        net_exposure: Vec::with_capacity(nb),
        risk_scale: Vec::with_capacity(nb),
        run: Vec::with_capacity(nb),
        book_refused: Vec::with_capacity(nb),
        halted: Vec::with_capacity(nb),
        decision: Vec::with_capacity(nb * s_n),
        rule_refused: Vec::with_capacity(nb * s_n),
        sleeve_open: Vec::with_capacity(nb * s_n),
        due: Vec::with_capacity(nb * s_n),
        planned: Vec::with_capacity(nb * s_n),
        share: Vec::with_capacity(nb * s_n),
        contrib: Vec::with_capacity(nb * s_n),
        shadow_ret_gross: Vec::with_capacity(nb * s_n),
        shadow_ret_cost: Vec::with_capacity(nb * s_n),
        marks: Vec::with_capacity(nb * n),
        units: Vec::with_capacity(nb * n),
        target_weights: Vec::with_capacity(nb * n),
        held_weights: Vec::with_capacity(nb * n),
        traded_by_instrument: Vec::with_capacity(nb * n),
        refusals: Vec::new(),
        window: None,
        signal_flips: book.sleeves.iter().map(|s| vec![0; s.universe.len()]).collect(),
        fills_per_instrument: vec![0; n],
        rebalance_bars: 0,
        gross_exposure_stats: None,
        net_exposure_stats: None,
        shadows: Vec::new(),
        halted_at: None,
        series_sha256: String::new(),
    };

    let sub_cap: Vec<f64> = book.sleeves.iter().map(|s| s.share.initial() * e0).collect();
    let idle_cash = if joint { 0.0 } else { e0 - sub_cap.iter().fold(0.0, |a, c| a + c) };

    for u in a0..n_union {
        let k = u - a0;
        let time = times[u];
        let date = time.date();

        // (1) availability
        let own: Vec<usize> = (0..s_n).map(|s| sl[s].own_of_union[u]).collect();
        let has: Vec<bool> = own.iter().map(|&o| o != NO_OWN).collect();
        for s in 0..s_n {
            if has[s] {
                continue;
            }
            for &i in &book.sleeves[s].universe {
                if panel.availability(i, u) == Availability::Gap {
                    if on_refusal == OnRefusal::Abort {
                        return Err(BookError::DataGap {
                            sleeve: book.sleeves[s].id.clone(),
                            instrument: panel.instruments()[i].clone(),
                            time,
                        });
                    }
                    res.refusals.push(BookRefusal {
                        bar: k,
                        time,
                        sleeve: Some(s),
                        kind: RefusalKind::Data,
                        code: "data_gap",
                        message: format!(
                            "instrument {} has no bar although its market is open",
                            panel.instruments()[i]
                        ),
                    });
                }
            }
        }

        // (2) marks
        let prev_mark = mark.clone();
        let prev_marked = marked.clone();
        let mut owner_open = vec![false; n];
        for j in 0..n {
            if owners[j].iter().any(|&s| has[s]) {
                owner_open[j] = true;
                mark[j] = panel.close(j)[u].expect("an open sleeve has every close");
                marked[j] = true;
            }
        }

        // (3) shadows: mark to market on the sleeve's own bars
        let mut sh_ret_g = vec![0.0f64; s_n];
        let mut sh_fin_c = vec![0.0f64; s_n];
        for s in 0..s_n {
            if !has[s] {
                continue;
            }
            let t = own[s];
            let r = &mut sl[s];
            let (_, rg) = r
                .sh_g
                .mark(&r.cal.panel, t)
                .map_err(|_| BookError::Sim(SimError::NonPositiveEquity { date, equity: r.sh_g.equity_pre }))?;
            let (fc, _) = r
                .sh_c
                .mark(&r.cal.panel, t)
                .map_err(|_| BookError::Sim(SimError::NonPositiveEquity { date, equity: r.sh_c.equity_pre }))?;
            sh_fin_c[s] = fc;
            if !r.sh_g.first {
                r.sh_dates.push(date);
                r.sh_gross.push(rg);
                sh_ret_g[s] = rg;
            }
        }

        // (4) the account: financing, E_pre
        let mut fin = 0.0;
        let equity_pre;
        if joint {
            if k == 0 {
                equity_pre = cash;
            } else {
                let mut long_v = 0.0;
                let mut short_v = 0.0;
                for j in 0..n {
                    let v = units[j] * prev_mark[j];
                    if v > 0.0 {
                        long_v += v;
                    } else {
                        short_v += -v;
                    }
                }
                fin = cfg.sim.financing.accrual(times[u - 1].date().days_until(date), cash, long_v, short_v);
                cash += fin;
                let mut invested = 0.0;
                for j in 0..n {
                    invested += units[j] * mark[j];
                }
                equity_pre = cash + invested;
            }
        } else {
            // legacy sum of sub-accounts: aggregate the (pre-trade) sub-account state
            if k == 0 {
                equity_pre = cash;
            } else {
                let mut c = idle_cash;
                for s in 0..s_n {
                    c += sub_cap[s] * sl[s].sh_c.cash;
                    fin += sub_cap[s] * sh_fin_c[s];
                }
                cash = c;
                for j in 0..n {
                    units[j] = 0.0;
                }
                for s in 0..s_n {
                    for (i, &j) in book.sleeves[s].universe.iter().enumerate() {
                        units[j] += sub_cap[s] * sl[s].sh_c.units[i];
                    }
                }
                let mut invested = 0.0;
                for j in 0..n {
                    invested += units[j] * mark[j];
                }
                equity_pre = cash + invested;
            }
        }
        if !(equity_pre > 0.0) {
            return Err(BookError::Sim(SimError::NonPositiveEquity { date, equity: equity_pre }));
        }

        // contributions of the weights held ENTERING this bar (before any decision of this bar)
        let mut contrib = vec![0.0f64; s_n];
        if k > 0 {
            if joint {
                for j in 0..n {
                    if !prev_marked[j] || prev_mark[j] <= 0.0 {
                        continue;
                    }
                    let r = mark[j] / prev_mark[j] - 1.0;
                    let piece = held_prev[j] * r;
                    match owners[j].len() {
                        0 => {}
                        1 => contrib[owners[j][0]] += piece,
                        _ => {
                            let mut tot = 0.0;
                            let mut comp = vec![0.0f64; owners[j].len()];
                            for (q, &s) in owners[j].iter().enumerate() {
                                if let Some(w) = &sl[s].standing {
                                    let pos = book.sleeves[s].universe.iter().position(|&x| x == j).expect("owner");
                                    comp[q] = (sl[s].share * w[pos]).abs();
                                    tot += comp[q];
                                }
                            }
                            if tot > 0.0 {
                                for (q, &s) in owners[j].iter().enumerate() {
                                    contrib[s] += piece * (comp[q] / tot);
                                }
                            } else {
                                contrib[owners[j][0]] += piece;
                            }
                        }
                    }
                }
            } else {
                for s in 0..s_n {
                    for (i, &j) in book.sleeves[s].universe.iter().enumerate() {
                        if !prev_marked[j] || prev_mark[j] <= 0.0 {
                            continue;
                        }
                        // sub-account units are as of the end of the previous bar only if it did not trade since;
                        // shadows have not traded yet on this bar (step 9), so their units are the held ones
                        let r = mark[j] / prev_mark[j] - 1.0;
                        contrib[s] += (sub_cap[s] * sl[s].sh_c.units[i] * prev_mark[j] / e_prev) * r;
                    }
                }
            }
        }

        // overlay
        let mut overlay_scale = 1.0;
        if let Some(o) = overlay_run.as_mut() {
            let dec = o.step(&OverlayInput { bar: k, time, equity: equity_pre, initial_equity: e0 });
            if !dec.scale.is_finite() {
                return Err(BookError::Sim(SimError::BadConfig("overlay returned a non-finite scale".into())));
            }
            overlay_scale = dec.scale;
            if dec.halt && !halted {
                halted = true;
                halted_at = Some(k);
            }
        }
        let rs_eff = cfg.sim.risk_scale * overlay_scale;

        // (5) decisions
        let mut decision_ok = vec![false; s_n];
        let mut rule_refused = vec![false; s_n];
        let mut newly = vec![false; s_n];
        for s in 0..s_n {
            if !has[s] {
                continue;
            }
            let t = own[s];
            let r = &mut sl[s];
            let kk = book.sleeves[s].universe.len();
            if r.schedule.is_decision_bar(r.cal.panel.dates(), t) && t + 1 >= r.min_hist {
                let view = HistoryView::new(&r.cal.panel, t);
                match r.run.step(&view) {
                    Ok(w) => {
                        if w.len() != kk {
                            return Err(BookError::Sim(SimError::InvalidWeights {
                                date,
                                reason: format!("{} weights for {} assets", w.len(), kk),
                            }));
                        }
                        if let Some(bad) = w.iter().position(|v| !v.is_finite()) {
                            return Err(BookError::Sim(SimError::InvalidWeights {
                                date,
                                reason: format!("weight {bad} is not finite"),
                            }));
                        }
                        r.decided_any = true;
                        decision_ok[s] = true;
                        r.decision_bars.push(k);
                        r.decision_signs.push(w.iter().map(|v| sign(*v * cfg.sim.risk_scale)).collect());
                        r.pending.push_back((t + delay, w));
                    }
                    Err(refusal) => {
                        let tolerated_warmup = refusal.kind == RefusalKind::Warmup && !r.decided_any;
                        if on_refusal == OnRefusal::Abort && !tolerated_warmup {
                            return Err(BookError::Sim(SimError::RuleRefused { date, refusal }));
                        }
                        rule_refused[s] = true;
                        res.refusals.push(BookRefusal {
                            bar: k,
                            time,
                            sleeve: Some(s),
                            kind: refusal.kind,
                            code: refusal.code,
                            message: refusal.message,
                        });
                    }
                }
            }
            while let Some(front) = r.pending.front() {
                if front.0 == t {
                    let (_, w) = r.pending.pop_front().expect("front exists");
                    r.standing = Some(w);
                    newly[s] = true;
                    if r.first_effective.is_none() {
                        r.first_effective = Some(k);
                    }
                } else {
                    break;
                }
            }
        }

        // (6) allocator review (static shares, past shadow returns only)
        if let AllocatorSpec::InverseVol { lookback_bars, total } = book.allocator {
            if month_end_review(times, u) {
                let windows: Vec<&[f64]> = sl.iter().map(|r| r.sh_gross.as_slice()).collect();
                if let Some(new_shares) = construct.inverse_vol_shares(&windows, lookback_bars, total) {
                    for s in 0..s_n {
                        sl[s].share = new_shares[s];
                    }
                }
            }
        }

        // (7) cadence
        let due: Vec<bool> =
            (0..s_n).map(|s| has[s] && (sl[s].policy == RebalancePolicy::EveryBar || newly[s])).collect();
        let run = due.iter().any(|d| *d);
        let planned: Vec<bool> = match cfg.cadence {
            BookCadence::PerSleeve => due.clone(),
            BookCadence::AllSleevesOnAnyDue => {
                (0..s_n).map(|s| run && (has[s] || cfg.trade_on_closed_market)).collect()
            }
        };

        // (8) construction and trade
        let mut cost = 0.0;
        let mut traded = 0.0;
        let mut traded_by = vec![0.0f64; n];
        let mut book_refused = false;
        let mut target_w: Vec<f64> = vec![0.0; n];
        if joint {
            if planned.iter().any(|p| *p) || halted {
                let mut plan_j = vec![false; n];
                for s in 0..s_n {
                    if planned[s] {
                        for &j in &book.sleeves[s].universe {
                            plan_j[j] = true;
                        }
                    }
                }
                let policy = ConstructPolicy {
                    risk_scale: rs_eff,
                    allocated_capital: cfg.allocated_capital,
                    max_gross: cfg.sim.max_gross,
                    trade_filter: cfg.trade_filter,
                    cash_policy: cfg.cash_policy,
                    fee_rate: rate,
                };
                let flatten_policy = ConstructPolicy {
                    max_gross: None,
                    trade_filter: None,
                    cash_policy: CashPolicy::Certification,
                    ..policy
                };
                let zero_w: Vec<Vec<f64>> = book.sleeves.iter().map(|sp| vec![0.0; sp.universe.len()]).collect();
                let targets: Vec<SleeveTarget<'_>> = (0..s_n)
                    .filter(|&s| sl[s].standing.is_some())
                    .map(|s| SleeveTarget {
                        share: sl[s].share,
                        instruments: &book.sleeves[s].universe,
                        weights: if halted { &zero_w[s] } else { sl[s].standing.as_ref().expect("filtered") },
                    })
                    .collect();
                let owner_open_plan: Vec<bool> = if halted { owner_open.clone() } else { plan_j.clone() };
                let out = construct.construct(&ConstructInputs {
                    equity: equity_pre,
                    cash,
                    marks: &mark,
                    units: &units,
                    sleeves: &targets,
                    planned: &owner_open_plan,
                    policy: if halted { &flatten_policy } else { &policy },
                });
                match out {
                    Err(ConstructRefusal::GrossAboveCap { gross, cap }) => {
                        if on_refusal == OnRefusal::Abort {
                            return Err(BookError::Sim(SimError::MaxGrossBreached { date, gross, limit: cap }));
                        }
                        book_refused = true;
                        res.refusals.push(BookRefusal {
                            bar: k,
                            time,
                            sleeve: None,
                            kind: RefusalKind::Other,
                            code: "gross_above_cap",
                            message: format!(
                                "gross {gross} exceeds cap {cap}: the whole book is refused, nothing traded"
                            ),
                        });
                        // the standing weights are still reported
                        for (j, w) in target_w.iter_mut().enumerate() {
                            *w = combined_weight(&book.sleeves, &sl, j, rs_eff, &owners);
                        }
                    }
                    Err(other) => return Err(BookError::Construct(other)),
                    Ok(o) => {
                        for j in 0..n {
                            target_w[j] = o.target_weight[j].unwrap_or(0.0);
                        }
                        let units_new = o.units_new;
                        traded_by = o.traded;
                        traded = o.traded_total;
                        let mut any_fill = false;
                        for j in 0..n {
                            if units_new[j] != units[j] {
                                fills[j] += 1;
                                any_fill = true;
                            }
                        }
                        if any_fill {
                            rebalance_bars += 1;
                        }
                        let mut mv2 = 0.0;
                        for j in 0..n {
                            mv2 += units_new[j] * mark[j];
                        }
                        cost = traded * rate;
                        cash = equity_pre - mv2 - cost;
                        units = units_new.clone();
                    }
                }
            } else {
                for (j, w) in target_w.iter_mut().enumerate() {
                    *w = combined_weight(&book.sleeves, &sl, j, rs_eff, &owners);
                }
            }
        }

        // (9) shadows trade and close the bar
        let mut sh_ret_c = vec![0.0f64; s_n];
        let mut sub_cost = 0.0;
        let mut sub_traded = 0.0;
        let mut sub_traded_by = vec![0.0f64; n];
        for s in 0..s_n {
            if !has[s] {
                continue;
            }
            let t = own[s];
            let r = &mut sl[s];
            let mut cost_g = 0.0;
            let mut cost_c = 0.0;
            if due[s] {
                if let Some(w) = r.standing.clone() {
                    let (_, cg) = r.sh_g.trade(&r.cal.panel, t, &w);
                    cost_g = cg;
                    let units_before: Vec<f64> = r.sh_c.units.clone();
                    let (tr_c, cc) = r.sh_c.trade(&r.cal.panel, t, &w);
                    cost_c = cc;
                    if !joint {
                        sub_traded += sub_cap[s] * tr_c;
                        sub_cost += sub_cap[s] * cc;
                        for (i, &j) in book.sleeves[s].universe.iter().enumerate() {
                            sub_traded_by[j] +=
                                sub_cap[s] * (r.sh_c.units[i] - units_before[i]).abs() * r.cal.panel.closes(i)[t];
                        }
                    }
                }
            }
            let _ = r.sh_g.finish(cost_g);
            let rc = r.sh_c.finish(cost_c);
            if !r.sh_c.first {
                r.sh_cost.push(rc);
                sh_ret_c[s] = rc;
            }
        }
        if !joint {
            // the aggregate after the sub-accounts traded
            let mut c = idle_cash;
            for s in 0..s_n {
                c += sub_cap[s] * sl[s].sh_c.cash;
            }
            cash = c;
            for j in 0..n {
                units[j] = 0.0;
            }
            for s in 0..s_n {
                for (i, &j) in book.sleeves[s].universe.iter().enumerate() {
                    units[j] += sub_cap[s] * sl[s].sh_c.units[i];
                }
            }
            cost = sub_cost;
            traded = sub_traded;
            traded_by = sub_traded_by;
            for (j, w) in target_w.iter_mut().enumerate() {
                *w = combined_weight(&book.sleeves, &sl, j, rs_eff, &owners);
            }
            let mut any_fill = false;
            for j in 0..n {
                if traded_by[j] > 0.0 {
                    fills[j] += 1;
                    any_fill = true;
                }
            }
            if any_fill {
                rebalance_bars += 1;
            }
        }

        // record
        let equity = equity_pre - cost;
        if !(equity > 0.0) {
            return Err(BookError::Sim(SimError::NonPositiveEquity { date, equity }));
        }
        let (ret_pre, ret) = if k == 0 { (0.0, 0.0) } else { (equity_pre / e_prev - 1.0, equity / e_prev - 1.0) };
        let mut gross_e = 0.0;
        let mut net_e = 0.0;
        for j in 0..n {
            let hw = units[j] * mark[j] / equity;
            gross_e += hw.abs();
            net_e += hw;
            held_prev[j] = hw;
            res.held_weights.push(hw);
            res.units.push(units[j]);
            res.marks.push(mark[j]);
            res.target_weights.push(target_w[j]);
            res.traded_by_instrument.push(traded_by[j]);
        }
        res.times.push(time);
        res.clock_index.push(u);
        res.ret.push(ret);
        res.ret_pre_cost.push(ret_pre);
        res.equity.push(equity);
        res.equity_pre.push(equity_pre);
        res.cash.push(cash);
        res.cost.push(cost);
        res.traded_notional.push(traded);
        res.financing.push(fin);
        res.gross_exposure.push(gross_e);
        res.net_exposure.push(net_e);
        res.risk_scale.push(rs_eff);
        res.run.push(run);
        res.book_refused.push(book_refused);
        res.halted.push(halted);
        for s in 0..s_n {
            res.decision.push(decision_ok[s]);
            res.rule_refused.push(rule_refused[s]);
            res.sleeve_open.push(has[s]);
            res.due.push(due[s]);
            res.planned.push(planned[s]);
            res.share.push(sl[s].share);
            res.contrib.push(contrib[s]);
            res.shadow_ret_gross.push(sh_ret_g[s]);
            res.shadow_ret_cost.push(sh_ret_c[s]);
        }
        e_prev = equity;
    }

    // ---- window (design S-7): counted returns are dated in [start, end], from the bar after the first fill
    let dates: Vec<Date> = res.times.iter().map(|t| t.date()).collect();
    let start_idx = match cfg.sim.start {
        Some(s) => dates.iter().position(|&d| d >= s),
        None => Some(0),
    };
    let end_idx = match cfg.sim.end {
        Some(e) => dates.iter().rposition(|&d| d <= e),
        None => Some(nb - 1),
    };
    let first_effective = sl.iter().filter_map(|r| r.first_effective).min();
    if let (Some(fe), Some(si), Some(ei)) = (first_effective, start_idx, end_idx) {
        let i0 = (fe + 1).max(si);
        if i0 <= ei {
            res.window = Some(Window { first_bar: i0, last_bar: ei });
        }
    }
    // signal flips per sleeve (T1's counter)
    for (s, r) in sl.iter().enumerate() {
        if let Some(&first_dec) = r.decision_bars.first() {
            let from = match cfg.sim.start {
                Some(st) => st.max(dates[first_dec]),
                None => dates[first_dec],
            };
            let mut prev: Option<&Vec<i8>> = None;
            for (bar, signs) in r.decision_bars.iter().zip(&r.decision_signs) {
                let d = dates[*bar];
                if d < from || cfg.sim.end.is_some_and(|e| d > e) {
                    continue;
                }
                if let Some(p) = prev {
                    for i in 0..signs.len() {
                        if p[i] != signs[i] {
                            res.signal_flips[s][i] += 1;
                        }
                    }
                }
                prev = Some(signs);
            }
        }
    }
    if let Some(w) = res.window {
        res.gross_exposure_stats = Some(exposure_stats(&res.gross_exposure[w.first_bar..=w.last_bar]));
        res.net_exposure_stats = Some(exposure_stats(&res.net_exposure[w.first_bar..=w.last_bar]));
    }
    res.shadows = sl
        .iter()
        .map(|r| ShadowSeries { dates: r.sh_dates.clone(), gross: r.sh_gross.clone(), cost: r.sh_cost.clone() })
        .collect();
    res.fills_per_instrument = fills;
    res.rebalance_bars = rebalance_bars;
    res.halted_at = halted_at;
    res.series_sha256 = book_digest(&res);
    Ok(res)
}

/// The combined standing weight of instrument `j` as construct would report it (`SUM share*w * risk_scale`), used
/// for the recorded target on bars where nothing was constructed.
fn combined_weight(
    specs: &[crate::book::SleeveSpec],
    sl: &[SleeveRt<'_>],
    j: usize,
    rs: f64,
    owners: &[Vec<usize>],
) -> f64 {
    let mut acc: Option<f64> = None;
    for &s in &owners[j] {
        if let Some(w) = &sl[s].standing {
            let pos = specs[s].universe.iter().position(|&x| x == j).expect("owner");
            let term = sl[s].share * w[pos];
            acc = Some(match acc {
                None => term,
                Some(a) => a + term,
            });
        }
    }
    acc.map_or(0.0, |v| v * rs)
}
