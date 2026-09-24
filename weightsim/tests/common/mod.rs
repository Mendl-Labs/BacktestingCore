//! Shared helpers for the integration tests: hand-written TEST rules (the library rules come in T3), a deterministic
//! synthetic price generator, and readers for the committed answer-key fixtures.
#![allow(dead_code, clippy::needless_range_loop, clippy::manual_is_multiple_of)]

use weightsim::*;

pub const ETF: [&str; 5] = ["SPY", "EFA", "IEF", "DBC", "VNQ"];
pub const CRY: [&str; 2] = ["BTC", "ETH"];

pub fn d(s: &str) -> Date {
    Date::parse(s).unwrap()
}

pub const SYNTH_CSV: &str = include_str!("../fixtures/synthetic_ladder_candles.csv");
pub const KEY_S1_RETURNS: &str = include_str!("../fixtures/key_S1_returns.csv");
pub const KEY_S1_SIGNALS: &str = include_str!("../fixtures/key_S1_signals.csv");
pub const KEY_S3_RETURNS: &str = include_str!("../fixtures/key_S3_returns.csv");
pub const KEY_S3_SIGNALS: &str = include_str!("../fixtures/key_S3_signals.csv");
pub const KEY_METRICS: &str = include_str!("../fixtures/key_metrics.csv");

// --------------------------------------------------------------------------------------------- deterministic data

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1) from integer arithmetic only (bit-reproducible on every platform).
pub fn uniform(seed: u64, a: u64, b: u64) -> f64 {
    let h = splitmix64(seed ^ splitmix64((a << 32) | b));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// `n` weekday dates starting at the first weekday on or after `start`.
pub fn weekdays(start: Date, n: usize) -> Vec<Date> {
    let mut out = Vec::with_capacity(n);
    let mut cur = start;
    while out.len() < n {
        if cur.weekday() < 5 {
            out.push(cur);
        }
        cur = cur.add_days(1);
    }
    out
}

/// Synthetic panel: weekday calendar, prices follow `p *= 1 + drift_i + noise` with only + - * (no libm), so the
/// panel is bit-identical everywhere. Per-asset slow drift regimes make SMA rules flip.
pub fn synth_panel(symbols: &[&str], n_bars: usize, seed: u64, start: &str) -> Panel {
    let dates = weekdays(d(start), n_bars);
    let mut closes = Vec::new();
    for (i, _) in symbols.iter().enumerate() {
        let mut p = 100.0 + 25.0 * i as f64;
        let mut col = Vec::with_capacity(n_bars);
        for t in 0..n_bars {
            let regime = if ((t / (60 + 17 * i)) % 2) == 0 { 0.0015 } else { -0.0015 };
            let noise = (uniform(seed, i as u64, t as u64) - 0.5) * 0.03;
            p *= 1.0 + regime + noise;
            col.push(p);
        }
        closes.push(col);
    }
    Panel::new(symbols.iter().map(|s| s.to_string()).collect(), dates, closes).unwrap()
}

// -------------------------------------------------------------------------------------------------- test rules

type RuleFn = dyn Fn(&HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> + Send + Sync;

/// A closure-backed test rule.
pub struct FnRule {
    pub universe: Vec<&'static str>,
    pub schedule: DecisionSchedule,
    pub policy: RebalancePolicy,
    pub min_hist: usize,
    pub f: Box<RuleFn>,
}

impl FnRule {
    pub fn new(
        universe: &[&'static str],
        schedule: DecisionSchedule,
        policy: RebalancePolicy,
        min_hist: usize,
        f: impl Fn(&HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> + Send + Sync + 'static,
    ) -> FnRule {
        FnRule { universe: universe.to_vec(), schedule, policy, min_hist, f: Box::new(f) }
    }
}

impl WeightRule for FnRule {
    fn id(&self) -> &'static str {
        "test_fn_rule"
    }
    fn impl_version(&self) -> String {
        "test".into()
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
        self.min_hist
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        (self.f)(h)
    }
}

/// Constant weights, every decision.
pub fn const_rule(
    universe: &[&'static str],
    w: Vec<f64>,
    schedule: DecisionSchedule,
    policy: RebalancePolicy,
) -> FnRule {
    FnRule::new(universe, schedule, policy, 1, move |_| Ok(w.clone()))
}

/// Test re-implementation of shadow.py S1 (Faber ETF trend): month-end close > SMA10 of month-end closes (current
/// month-end included) => 20% of the sleeve; drifting between month-ends. Not the library rule (that is T3).
pub struct S1TestRule;

impl WeightRule for S1TestRule {
    fn id(&self) -> &'static str {
        "test_s1"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &ETF
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
        // Month-end bars visible so far. The current bar t is the last visible bar and (because the schedule only
        // asks on month-ends) is itself a month-end, so it is included, exactly as in shadow.py.
        let dates = h.dates();
        let mut me: Vec<usize> = Vec::new();
        for i in 0..dates.len() {
            if i + 1 == dates.len() || !dates[i].same_month(dates[i + 1]) {
                me.push(i);
            }
        }
        if me.len() < 10 {
            return Err(RuleRefusal::warmup("need 10 month-end closes"));
        }
        let last10 = &me[me.len() - 10..];
        let mut w = Vec::with_capacity(5);
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &i in last10 {
                s += c[i];
            }
            let sma = s / 10.0;
            w.push(if c[c.len() - 1] > sma { 0.2 } else { 0.0 });
        }
        Ok(w)
    }
}

/// Test re-implementation of shadow.py S3 (crypto trend): close > SMA100 (current bar included) => 50% of the sleeve
/// per coin, daily rebalanced to fixed weights.
pub struct S3TestRule;

impl WeightRule for S3TestRule {
    fn id(&self) -> &'static str {
        "test_s3"
    }
    fn impl_version(&self) -> String {
        "test".into()
    }
    fn universe(&self) -> &[&'static str] {
        &CRY
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        DecisionSchedule::Daily
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        RebalancePolicy::EveryBar
    }
    fn min_history_bars(&self) -> usize {
        100
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        let mut w = Vec::with_capacity(2);
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &v in &c[c.len() - 100..] {
                s += v;
            }
            let sma = s / 100.0;
            w.push(if c[c.len() - 1] > sma { 0.5 } else { 0.0 });
        }
        Ok(w)
    }
}

// ------------------------------------------------------------------------------------------- fixture readers

/// Parse `date,ret,equity` (key files) or `date,ret` / `,ret` (shadow files): returns (dates, ret, equity-if-present).
pub fn parse_returns(text: &str) -> (Vec<Date>, Vec<f64>, Vec<f64>) {
    let mut dates = Vec::new();
    let mut ret = Vec::new();
    let mut eq = Vec::new();
    for line in text.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split(',').collect();
        dates.push(Date::parse(p[0]).unwrap());
        ret.push(p[1].parse::<f64>().unwrap());
        if p.len() > 2 {
            eq.push(p[2].parse::<f64>().unwrap());
        }
    }
    (dates, ret, eq)
}

/// Parse a signals file (`date,<sym>...`, values 0/1): returns (dates, rows).
pub fn parse_signals(text: &str) -> (Vec<Date>, Vec<Vec<f64>>) {
    let mut dates = Vec::new();
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split(',').collect();
        dates.push(Date::parse(p[0]).unwrap());
        rows.push(p[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect());
    }
    (dates, rows)
}

/// `key_metrics.csv` lookup.
pub fn key_metric(sleeve: &str, metric: &str) -> f64 {
    for line in KEY_METRICS.lines().skip(1) {
        let p: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        if p[0] == sleeve && p[1] == metric {
            return p[2].parse().unwrap();
        }
    }
    panic!("no key metric {sleeve}.{metric}");
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

pub fn s1_panel() -> Panel {
    Panel::from_long_csv(SYNTH_CSV, &ETF).unwrap()
}

pub fn s3_panel() -> Panel {
    Panel::from_long_csv(SYNTH_CSV, &CRY).unwrap()
}

pub fn s1_config() -> SimConfig {
    SimConfig::default()
}

pub fn s3_config() -> SimConfig {
    SimConfig { start: Some(d("2016-01-01")), end: Some(d("2020-12-31")), ..SimConfig::default() }
}
