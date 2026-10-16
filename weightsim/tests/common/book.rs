//! Shared helpers of the BOOK tests: the synthetic book fixture (generated once by `book_fixtures/gen_book_key.py`, an
//! independent Python engine that mirrors the PF0 book key and is itself proven equal to the real key), the test rules
//! whose decisions the generator's Python rules reproduce, the per-bar key reader and the comparison of a Rust book run
//! with a key file (the same column semantics as `book_key.py`'s `table()`, see `compare_book_to_key`).
#![allow(dead_code, clippy::needless_range_loop, clippy::manual_is_multiple_of, clippy::field_reassign_with_default)]

use super::*;

pub const BOOK_CSV: &str = include_str!("../book_fixtures/synthetic_book_candles.csv");
pub const NETTING_CSV: &str = include_str!("../book_fixtures/synthetic_netting_candles.csv");
pub const BOOK_META: &str = include_str!("../book_fixtures/synthetic_book_meta.csv");
pub const BOOK_DECISIONS: &str = include_str!("../book_fixtures/synthetic_book_decisions.csv");
pub const BOOK_F1: &str = include_str!("../book_fixtures/synthetic_book_f1.csv");

pub const E: [&str; 3] = ["E1", "E2", "E3"];
pub const C: [&str; 2] = ["C1", "C2"];
pub const ETF_HOLIDAYS: [&str; 5] = ["2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04"];
pub const CAPITAL0: f64 = 100000.0;

pub fn meta(name: &str) -> String {
    for line in BOOK_META.lines().skip(1) {
        let p: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        if p[0] == name {
            return p[1].to_string();
        }
    }
    panic!("no meta {name}");
}

pub fn meta_f(name: &str) -> f64 {
    meta(name).parse().unwrap()
}

// --------------------------------------------------------------------------------------------------- csv
pub struct Csv {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

pub fn parse_csv(text: &str) -> Csv {
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().trim_end_matches('\r').split(',').map(str::to_string).collect();
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| l.trim_end_matches('\r').split(',').map(str::to_string).collect())
        .collect();
    Csv { header, rows }
}

impl Csv {
    pub fn col(&self, name: &str) -> usize {
        self.header.iter().position(|h| h == name).unwrap_or_else(|| panic!("no column {name}"))
    }
    pub fn f(&self, r: usize, name: &str) -> f64 {
        self.rows[r][self.col(name)].parse().unwrap()
    }
    pub fn s(&self, r: usize, name: &str) -> &str {
        &self.rows[r][self.col(name)]
    }
}

// --------------------------------------------------------------------------------------------------- panels
pub fn etf_session() -> SessionKind {
    SessionKind::exchange("test_us", ETF_HOLIDAYS.iter().map(|s| d(s)).collect())
}

/// The synthetic ETF (weekday, holidays declared) and crypto (7-day) instruments; the crypto data ends at the end of
/// the key window, the ETF data runs on for two more weeks.
pub fn full_book_panel() -> BookPanel {
    let inst: Vec<(&str, SessionKind)> =
        E.iter().map(|s| (*s, etf_session())).chain(C.iter().map(|s| (*s, SessionKind::Continuous))).collect();
    BookPanel::from_long_csv(BOOK_CSV, &inst).unwrap()
}

pub fn truncate_after(panel: &BookPanel, last_date: &str) -> BookPanel {
    let end = BarTime::from_date(d(last_date));
    let idx = panel.times().iter().rposition(|t| *t <= end).unwrap();
    panel.truncated(idx + 1)
}

/// The two-sleeve key window: cut where the crypto data ends (the key's `end = earliest last bar`).
pub fn key_book_panel() -> BookPanel {
    truncate_after(&full_book_panel(), &meta("end"))
}

pub fn etf_only_panel() -> BookPanel {
    let inst: Vec<(&str, SessionKind)> = E.iter().map(|s| (*s, etf_session())).collect();
    BookPanel::from_long_csv(BOOK_CSV, &inst).unwrap()
}

pub fn netting_panel() -> BookPanel {
    BookPanel::from_long_csv(
        NETTING_CSV,
        &[("X", SessionKind::Continuous), ("Y", SessionKind::Continuous), ("Z", SessionKind::Continuous)],
    )
    .unwrap()
}

// --------------------------------------------------------------------------------------------------- rules
/// Monthly ETF-like rule (the counterpart of the S1 test rule): month-end close above the SMA of the last 3 month-end
/// closes (the current month-end included) => 30% of the sleeve in that instrument, drifting between month-ends.
pub struct BookEtfRule;

impl WeightRule for BookEtfRule {
    fn id(&self) -> &'static str {
        "book_etf_test"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &E
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::LastBarOfMonth
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::OnDecision
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let dates = h.dates();
        let mut me: Vec<usize> = Vec::new();
        for i in 0..dates.len() {
            if i + 1 == dates.len() || !dates[i].same_month(dates[i + 1]) {
                me.push(i);
            }
        }
        if me.len() < 3 {
            return Err(RuleRefusal::warmup("need 3 month-end closes"));
        }
        let last3 = &me[me.len() - 3..];
        let mut w = Vec::with_capacity(3);
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &i in last3 {
                s += c[i];
            }
            w.push(if c[c.len() - 1] > s / 3.0 { 0.3 } else { 0.0 });
        }
        Ok(w)
    }
}

/// Daily crypto-like rule: close above the SMA10 (current bar included) => 50% of the sleeve per coin, EveryBar.
pub struct BookCryRule;

impl WeightRule for BookCryRule {
    fn id(&self) -> &'static str {
        "book_cry_test"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &C
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        10
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let mut w = Vec::with_capacity(2);
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &v in &c[c.len() - 10..] {
                s += v;
            }
            w.push(if c[c.len() - 1] > s / 10.0 { 0.5 } else { 0.0 });
        }
        Ok(w)
    }
}

/// The two sleeves of the netting fixture: opposite signs on the shared instrument X, both EveryBar.
pub struct NetRuleA;
pub struct NetRuleB;

impl WeightRule for NetRuleA {
    fn id(&self) -> &'static str {
        "net_a"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["X", "Y"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let i = h.len() - 1;
        Ok(if (i / 7) % 2 == 0 { vec![0.6, 0.4] } else { vec![0.3, 0.7] })
    }
}

impl WeightRule for NetRuleB {
    fn id(&self) -> &'static str {
        "net_b"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &["X", "Z"]
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let i = h.len() - 1;
        Ok(if (i / 5) % 3 != 1 { vec![-0.8, 0.2] } else { vec![0.5, -0.3] })
    }
}

/// A rule that replays recorded decisions (date -> weights): used for the REAL-data comparison, where the sleeve weights are
/// the T0 key's decisions (the book key does not re-derive rule logic either; it tests the ACCOUNT layer).
pub struct ScriptedRule {
    pub id: &'static str,
    pub universe: Vec<&'static str>,
    pub schedule: DecisionSchedule,
    pub policy: RebalancePolicy,
    pub decisions: std::collections::HashMap<Date, Vec<f64>>,
}

impl WeightRule for ScriptedRule {
    fn id(&self) -> &'static str {
        self.id
    }
    fn impl_version(&self) -> String {
        "scripted".into()
    }
    fn universe(&self) -> &[&'static str] {
        &self.universe
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.schedule
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.policy
    }
    fn min_history_bars(&self) -> usize {
        1
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        match self.decisions.get(&h.date()) {
            Some(w) => Ok(w.clone()),
            None => Err(RuleRefusal::warmup("no recorded decision for this date")),
        }
    }
}

/// Parse a T0 `*_decisions.csv` (`date,status,reason,w_<sym>...`, every row `ok`).
pub fn read_decisions(text: &str, n_assets: usize) -> std::collections::HashMap<Date, Vec<f64>> {
    let mut out = std::collections::HashMap::new();
    for line in text.lines().skip(1) {
        let p: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        if p.len() < 3 + n_assets {
            continue;
        }
        assert_eq!(p[1], "ok", "T0 decision rows are all ok");
        out.insert(d(p[0]), p[3..3 + n_assets].iter().map(|x| x.parse().unwrap()).collect());
    }
    out
}

// --------------------------------------------------------------------------------------------------- cases
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub shares: (f64, f64),
    pub cadence: BookCadence,
    pub filter: bool,
    pub budget: bool,
    pub risk_scale: f64,
    pub allocated_currency: Option<f64>,
    pub max_gross: Option<f64>,
    pub invvol_lookback: Option<usize>,
    pub etf_only: bool,
    pub etf_every_bar: bool,
}

impl Case {
    fn base(name: &'static str) -> Case {
        Case {
            name,
            shares: (0.6, 0.4),
            cadence: BookCadence::PerSleeve,
            filter: false,
            budget: false,
            risk_scale: 1.0,
            allocated_currency: None,
            max_gross: None,
            invvol_lookback: None,
            etf_only: false,
            etf_every_bar: false,
        }
    }
}

/// The synthetic counterparts of the key's named configurations (parameters as in `gen_book_key.py`).
pub fn cases() -> Vec<Case> {
    let live = |name| Case { cadence: BookCadence::AllSleevesOnAnyDue, filter: true, budget: true, ..Case::base(name) };
    vec![
        Case::base("book_cert_60_40"),
        live("book_live_60_40"),
        Case { cadence: BookCadence::PerSleeve, ..live("book_due_filter_60_40") },
        Case {
            shares: (0.5, 0.3),
            risk_scale: meta_f("risk_scale_scaled"),
            allocated_currency: Some(meta_f("allocated_capital_currency")),
            ..live("book_scaled_50_30")
        },
        Case { max_gross: Some(meta_f("max_gross")), ..Case::base("book_grosscap_60_40") },
        Case {
            shares: (0.5, 0.5),
            invvol_lookback: Some(meta_f("invvol_lookback") as usize),
            ..Case::base("book_invvol")
        },
        Case {
            shares: (1.0, 0.0),
            filter: true,
            etf_only: true,
            etf_every_bar: true,
            ..Case::base("single_etf_everybar_filter")
        },
    ]
}

pub fn case(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

pub fn instrument_index(p: &BookPanel, names: &[&str]) -> Vec<usize> {
    names.iter().map(|n| p.instrument_index(n).unwrap()).collect()
}

/// Panel, book and (gross-cost-free) configuration for a case; the caller sets the cost for the net run.
pub fn build_case(c: &Case) -> (BookPanel, Book, BookConfig) {
    let panel = if c.etf_only { etf_only_panel() } else { key_book_panel() };
    let book = book_for_case(&panel, c);
    (panel, book, config_for_case(c))
}

pub fn book_for_case(panel: &BookPanel, c: &Case) -> Book {
    let etf_idx = instrument_index(panel, &E);
    let alloc = c.invvol_lookback.map(|l| AllocatorSpec::InverseVol { lookback_bars: l, total: 1.0 });
    let share = |v: f64| if alloc.is_some() { ShareSpec::Allocated { initial: v } } else { ShareSpec::Fixed(v) };
    let mut etf = SleeveSpec::from_rule("etf", BookEtfRule, etf_idx, share(c.shares.0));
    if c.etf_every_bar {
        etf = etf.with_policy(RebalancePolicy::EveryBar);
    }
    let mut sleeves = vec![etf];
    if !c.etf_only {
        sleeves.push(SleeveSpec::from_rule("cry", BookCryRule, instrument_index(panel, &C), share(c.shares.1)));
    }
    let mut b = Book::new(sleeves);
    if let Some(a) = alloc {
        b = b.with_allocator(a);
    }
    b
}

pub fn config_for_case(c: &Case) -> BookConfig {
    let mut cfg = BookConfig::default();
    cfg.sim.on_refusal = OnRefusal::HoldPrevious;
    cfg.sim.risk_scale = c.risk_scale;
    cfg.sim.max_gross = c.max_gross;
    cfg.sim.cost = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;
    cfg.account_start = Some(BarTime::from_date(d(&meta("start_bar"))));
    cfg.cadence = c.cadence;
    if c.filter {
        cfg.trade_filter = Some(TradeFilter { min_abs: meta_f("min_abs") / CAPITAL0, min_pct: meta_f("min_pct") });
    }
    if c.budget {
        cfg.cash_policy = CashPolicy::Budget;
    }
    cfg.allocated_capital = c.allocated_currency.map(|a| a / CAPITAL0);
    cfg
}

pub fn netting_book(panel: &BookPanel) -> Book {
    Book::new(vec![
        SleeveSpec::from_rule("a", NetRuleA, instrument_index(panel, &["X", "Y"]), ShareSpec::Fixed(0.5)),
        SleeveSpec::from_rule("b", NetRuleB, instrument_index(panel, &["X", "Z"]), ShareSpec::Fixed(0.5)),
    ])
}

pub fn netting_config() -> BookConfig {
    let mut cfg = BookConfig::default();
    cfg.sim.on_refusal = OnRefusal::HoldPrevious;
    cfg.sim.cost = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE;
    cfg
}

// --------------------------------------------------------------------------------------------------- key comparison
pub struct KeyCompare {
    pub rows: usize,
    pub cells: usize,
    pub cells_not_bit_identical: usize,
    pub worst: f64,
    pub worst_col: String,
    /// Largest difference on the columns that must be exact (flags).
    pub worst_flag: f64,
    pub by_col: Vec<(String, f64)>,
}

/// One column of the key layout computed from a gross run `g` and a net run `n` (same book, `n` charged the
/// certification preset), with the key's semantics: key row `r` is account bar `k = r + 1`; `w_target`, `w_held`,
/// `gross_exposure`, `net_exposure`, `cash_frac` and `share_*` are what was in force ENTERING bar `k` (= my bar `k-1`);
/// `cost` and `turnover` are divided by the pre-cost equity of the bar, `cost_prev_eq` by the previous equity;
/// `traded_gross_<sleeve>` is the traded notional of the instruments the sleeve owns FIRST, over pre-cost equity.
pub fn key_column(name: &str, g: &BookResult, n: &BookResult) -> Vec<f64> {
    let s_n = g.n_sleeves();
    let ni = g.n_instruments();
    let first_owner: Vec<usize> =
        (0..ni).map(|j| (0..s_n).find(|&s| g.sleeve_universes[s].contains(&j)).unwrap_or(usize::MAX)).collect();
    let sleeve_of = |prefix: &str| -> usize {
        let sid = &name[prefix.len()..];
        g.sleeve_ids.iter().position(|x| x == sid).unwrap_or_else(|| panic!("unknown sleeve in column {name}"))
    };
    let inst_of = |prefix: &str| -> usize {
        let sym = &name[prefix.len()..];
        g.instruments.iter().position(|x| x == sym).unwrap_or_else(|| panic!("unknown instrument in column {name}"))
    };
    let flag = |b: bool| f64::from(u8::from(b));
    (1..g.n_bars())
        .map(|k| match name {
            "ret_gross" => g.ret[k],
            "ret_net" => n.ret[k],
            "equity_gross" => g.equity[k],
            "equity_net" => n.equity[k],
            "cost" => n.cost[k] / n.equity_pre[k],
            "turnover" => n.traded_notional[k] / n.equity_pre[k],
            "cost_prev_eq" => n.cost[k] / n.equity[k - 1],
            "gross_exposure" => g.gross_exposure[k - 1],
            "net_exposure" => g.net_exposure[k - 1],
            "cash_frac" => g.cash[k - 1] / g.equity[k - 1],
            "run" => flag(g.run[k]),
            "refused" => flag(g.book_refused[k]),
            "decision" => flag(g.decision[k * s_n]),
            "excluded" => 0.0,
            _ if name.starts_with("decision_") => flag(g.decision[k * s_n + sleeve_of("decision_")]),
            _ if name.starts_with("planned_") => flag(g.planned[k * s_n + sleeve_of("planned_")]),
            _ if name.starts_with("share_") => g.share[(k - 1) * s_n + sleeve_of("share_")],
            _ if name.starts_with("traded_gross_") => {
                let s = sleeve_of("traded_gross_");
                let mut t = 0.0;
                for j in 0..ni {
                    if first_owner[j] == s {
                        t += g.traded_by_instrument[k * ni + j];
                    }
                }
                t / g.equity_pre[k]
            }
            _ if name.starts_with("w_target_") => g.target_weights[(k - 1) * ni + inst_of("w_target_")],
            _ if name.starts_with("w_held_") => g.held_weights[(k - 1) * ni + inst_of("w_held_")],
            _ if name.starts_with("contrib_") => g.contrib[k * s_n + sleeve_of("contrib_")],
            _ if name.starts_with("shadow_ret_gross_") => g.shadow_ret_gross[k * s_n + sleeve_of("shadow_ret_gross_")],
            _ if name.starts_with("shadow_ret_net_") => n.shadow_ret_cost[k * s_n + sleeve_of("shadow_ret_net_")],
            other => panic!("key column {other} has no mapping"),
        })
        .collect()
}

/// Compare a gross run and a net run with a per-bar key file (see [`key_column`]) cell by cell.
pub fn compare_book_to_key(key: &Csv, g: &BookResult, n: &BookResult) -> KeyCompare {
    assert_eq!(key.rows.len() + 1, g.n_bars(), "key rows = account bars minus the build bar");
    let mut by_col: Vec<(String, f64)> = Vec::new();
    let (mut cells, mut nbi, mut worst, mut worst_col, mut worst_flag) =
        (0usize, 0usize, 0.0f64, String::new(), 0.0f64);
    for (ci, name) in key.header.iter().enumerate() {
        if name == "date" {
            for r in 0..key.rows.len() {
                assert_eq!(key.rows[r][ci], g.times[r + 1].date().to_string(), "date at row {r}");
            }
            continue;
        }
        let is_flag =
            name == "run" || name == "refused" || name.starts_with("decision") || name.starts_with("planned_");
        let mine = key_column(name, g, n);
        let mut col_worst = 0.0f64;
        for r in 0..key.rows.len() {
            let want: f64 = key.rows[r][ci].parse().unwrap();
            let diff = (mine[r] - want).abs();
            cells += 1;
            if mine[r] != want {
                nbi += 1;
            }
            col_worst = col_worst.max(diff);
            if is_flag {
                worst_flag = worst_flag.max(diff);
            }
        }
        if col_worst > worst {
            worst = col_worst;
            worst_col = name.clone();
        }
        by_col.push((name.clone(), col_worst));
    }
    KeyCompare { rows: key.rows.len(), cells, cells_not_bit_identical: nbi, worst, worst_col, worst_flag, by_col }
}

// --------------------------------------------------------------------------------------------------- F1 (cadence) numbers
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut sa, mut sb) = (0.0, 0.0);
    for i in 0..n {
        sa += a[i];
        sb += b[i];
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
    for i in 0..n {
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
        sab += (a[i] - ma) * (b[i] - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// The finding-F1 statistics of `gen_book_key.py::f1_stats` / `book_key.py::compare_configs` for X = only-due and Y =
/// all-on-any-due runs `(gross, net)` of the same book; `etf_syms` are the ETF instruments' column suffixes.
pub fn f1_stats(
    x: (&BookResult, &BookResult),
    y: (&BookResult, &BookResult),
    etf_syms: &[&str],
) -> Vec<(&'static str, f64)> {
    let col = |r: (&BookResult, &BookResult), name: &str| key_column(name, r.0, r.1);
    let n = x.0.n_bars() - 1;
    let sum = |v: &[f64]| {
        let mut s = 0.0;
        for a in v {
            s += a;
        }
        s
    };
    let l1: Vec<f64> = {
        let wx: Vec<Vec<f64>> = etf_syms.iter().map(|s| col(x, &format!("w_held_{s}"))).collect();
        let wy: Vec<Vec<f64>> = etf_syms.iter().map(|s| col(y, &format!("w_held_{s}"))).collect();
        (0..n)
            .map(|k| {
                let mut t = 0.0;
                for i in 0..etf_syms.len() {
                    t += (wy[i][k] - wx[i][k]).abs();
                }
                t
            })
            .collect()
    };
    let rg_x = col(x, "ret_gross");
    let rg_y = col(y, "ret_gross");
    let dg: Vec<f64> = (0..n).map(|k| rg_y[k] - rg_x[k]).collect();
    let dates: Vec<Date> = (1..=n).map(|k| x.0.times[k].date()).collect();
    let rn_x = col(x, "ret_net");
    let rn_y = col(y, "ret_net");
    let mx = answer_key_metrics(&dates, &rn_x).unwrap();
    let my = answer_key_metrics(&dates, &rn_y).unwrap();
    let yrs = mx.years;
    let tr_x = col(x, "traded_gross_etf");
    let tr_y = col(y, "traded_gross_etf");
    let cnt = |v: &[f64], pred: &dyn Fn(f64) -> bool| v.iter().filter(|a| pred(**a)).count() as f64;
    vec![
        ("bars", n as f64),
        ("etf_planned_bars_X", sum(&col(x, "planned_etf"))),
        ("etf_planned_bars_Y", sum(&col(y, "planned_etf"))),
        ("etf_decision_bars", sum(&col(x, "decision_etf"))),
        ("etf_trade_bars_gross_X", cnt(&tr_x, &|a| a > 0.0)),
        ("etf_trade_bars_gross_Y", cnt(&tr_y, &|a| a > 0.0)),
        ("etf_weight_gap_L1_max", l1.iter().cloned().fold(f64::MIN, f64::max)),
        ("etf_weight_gap_L1_mean", sum(&l1) / n as f64),
        ("etf_weight_gap_bars_gt_1e-9", cnt(&l1, &|a| a > 1e-9)),
        ("etf_weight_gap_bars_gt_1e-2", cnt(&l1, &|a| a > 1e-2)),
        ("ret_gross_diff_max_abs", dg.iter().map(|a| a.abs()).fold(0.0, f64::max)),
        ("ret_gross_diff_mean_abs", dg.iter().map(|a| a.abs()).sum::<f64>() / n as f64),
        ("ret_gross_diff_bars_gt_1e-9", cnt(&dg, &|a| a.abs() > 1e-9)),
        ("ret_gross_corr", pearson(&rg_x, &rg_y)),
        ("ret_net_corr", pearson(&rn_x, &rn_y)),
        ("net_d_sharpe_Y_minus_X", my.sharpe - mx.sharpe),
        ("net_d_cagr_pp_Y_minus_X", 100.0 * (my.cagr - mx.cagr)),
        ("turnover_per_year_X", sum(&col(x, "turnover")) / yrs),
        ("turnover_per_year_Y", sum(&col(y, "turnover")) / yrs),
        ("cost_bps_per_year_X", 10000.0 * sum(&col(x, "cost")) / yrs),
        ("cost_bps_per_year_Y", 10000.0 * sum(&col(y, "cost")) / yrs),
    ]
}
