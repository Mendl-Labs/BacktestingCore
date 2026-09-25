//! Shared helpers for the always-on tests: the synthetic price panel and the independent Python-generated key of
//! `weightsim` (`../weightsim/tests/fixtures/`, public and synthetic), independent hand-written ORACLE rules, and an
//! in-memory synthetic answer-key fixture set in the exact layout of the real (private) one, so the ladder logic can
//! be exercised, and mutation-tested, without any vendor data.
#![allow(dead_code, clippy::needless_range_loop)]

use std::collections::BTreeMap;

use weightsim::*;
use weightsim_rules::ladder::checks::compare;
use weightsim_rules::ladder::fixtures::{Fixtures, Pins, F_CANDLES, F_KEY_METRICS, F_MANIFEST, F_MUTANTS};
use weightsim_rules::ladder::mutants::{run_mutant, Mutant, MutantSleeve};
use weightsim_rules::ladder::runner::key_rows_for_mutants;

pub const ETF: [&str; 5] = ["SPY", "EFA", "IEF", "DBC", "VNQ"];
pub const CRY: [&str; 2] = ["BTC", "ETH"];

pub const SYNTH_CSV: &str = include_str!("../../../weightsim/tests/fixtures/synthetic_ladder_candles.csv");
pub const PY_KEY_S1_RETURNS: &str = include_str!("../../../weightsim/tests/fixtures/key_S1_returns.csv");
pub const PY_KEY_S1_SIGNALS: &str = include_str!("../../../weightsim/tests/fixtures/key_S1_signals.csv");
pub const PY_KEY_S3_RETURNS: &str = include_str!("../../../weightsim/tests/fixtures/key_S3_returns.csv");
pub const PY_KEY_S3_SIGNALS: &str = include_str!("../../../weightsim/tests/fixtures/key_S3_signals.csv");
pub const PY_KEY_METRICS: &str = include_str!("../../../weightsim/tests/fixtures/key_metrics.csv");

pub fn d(s: &str) -> Date {
    Date::parse(s).unwrap()
}

pub fn s1_panel() -> Panel {
    Panel::from_long_csv(SYNTH_CSV, &ETF).unwrap()
}

pub fn s3_panel() -> Panel {
    Panel::from_long_csv(SYNTH_CSV, &CRY).unwrap()
}

// ------------------------------------------------------------------------------------------------ oracle rules

/// Independent re-implementation of `shadow.py` S1 (not the library rule): month-end close > mean of the last 10
/// month-end closes (current one included) => 20% of the sleeve.
pub struct OracleS1;

impl WeightRule for OracleS1 {
    fn id(&self) -> &'static str {
        "oracle_s1"
    }
    fn impl_version(&self) -> String {
        "oracle".into()
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
        let mut w = Vec::new();
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &i in last10 {
                s += c[i];
            }
            w.push(if c[c.len() - 1] > s / 10.0 { 0.2 } else { 0.0 });
        }
        Ok(w)
    }
}

/// Independent re-implementation of `shadow.py` S3: close > mean of the last 100 closes (today included) => 50%.
pub struct OracleS3;

impl WeightRule for OracleS3 {
    fn id(&self) -> &'static str {
        "oracle_s3"
    }
    fn impl_version(&self) -> String {
        "oracle".into()
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
        let mut w = Vec::new();
        for a in 0..h.n_assets() {
            let c = h.closes(a);
            let mut s = 0.0;
            for &v in &c[c.len() - 100..] {
                s += v;
            }
            w.push(if c[c.len() - 1] > s / 100.0 { 0.5 } else { 0.0 });
        }
        Ok(w)
    }
}

// ------------------------------------------------------------------------------------------- csv/json helpers

pub fn parse_returns(text: &str) -> (Vec<Date>, Vec<f64>) {
    let mut dates = Vec::new();
    let mut ret = Vec::new();
    for line in text.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split(',').collect();
        dates.push(Date::parse(p[0]).unwrap());
        ret.push(p[1].parse::<f64>().unwrap());
    }
    (dates, ret)
}

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

pub fn py_key_metric(sleeve: &str, metric: &str) -> f64 {
    for line in PY_KEY_METRICS.lines().skip(1) {
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

// -------------------------------------------------------------------------------- synthetic answer-key fixtures

fn s1_cfg(cost: CostModel) -> SimConfig {
    SimConfig { cost, ..SimConfig::default() }
}

fn s3_cfg(cost: CostModel) -> SimConfig {
    SimConfig { start: Some(d("2016-01-01")), end: Some(d("2020-12-31")), cost, ..SimConfig::default() }
}

fn f(v: f64) -> String {
    // `{:?}` prints the shortest string that round-trips, so the parse in the loader is bit-exact.
    format!("{v:?}")
}

/// The per-bar key CSV (Amendment 11 layout) of an ORACLE rule run under the simulator. Alignment is written out
/// independently of the crate's `rows_from_sim`: row `t` carries the simulator's `ret/equity/cost/traded` at `t` and
/// the standing target and held weights at `t - 1`.
fn perbar_csv<R: WeightRule>(
    panel: &Panel,
    rule: R,
    cfg_of: fn(CostModel) -> SimConfig,
    syms: &[&str],
) -> (String, u64, SimResult, SimResult) {
    // The key ledger starts flat at the bar before the window and enters its first position there. Find the window
    // with an ungated zero-cost run, then re-run the oracle gated at that entry bar (the oracle's own gate, independent
    // of the crate's `FlatUntil`).
    let plain = simulate(panel, &rule, &cfg_of(CostModel::ZERO)).unwrap();
    let pw = plain.window.expect("oracle run has a window");
    let (first, entry) = (plain.dates[pw.first_bar], plain.dates[pw.first_bar - 1]);
    let gated = Gated { inner: rule, from: entry };
    let cfg = SimConfig { start: Some(first), ..cfg_of(CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE) };
    let (g, n) = simulate_gross_and_net(panel, &gated, &cfg).unwrap();
    let w = g.window.expect("gated oracle run has a window");
    assert_eq!(g.dates[w.first_bar], first, "the gated run must open its window on the same bar");
    let mut out = String::from(
        "date,ret_gross,ret_net,equity_gross,equity_net,cost,turnover,gross_exposure,net_exposure,decision,excluded",
    );
    for s in syms {
        out.push_str(&format!(",w_target_{s}"));
    }
    for s in syms {
        out.push_str(&format!(",w_held_{s}"));
    }
    out.push('\n');
    for i in w.first_bar..=w.last_bar {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},0",
            g.dates[i],
            f(g.ret[i]),
            f(n.ret[i]),
            f(g.equity[i]),
            f(n.equity[i]),
            // cost and turnover as fractions of the pre-cost equity at the rebalance
            f(n.cost[i] / (n.equity[i] + n.cost[i])),
            f(n.traded_notional[i] / (n.equity[i] + n.cost[i])),
            f(g.gross_exposure[i - 1]),
            f(g.net_exposure[i - 1]),
            u8::from(g.decision[i]),
        ));
        for a in 0..syms.len() {
            out.push_str(&format!(",{}", f(g.row(&g.target_weights, i - 1)[a])));
        }
        for a in 0..syms.len() {
            out.push_str(&format!(",{}", f(g.row(&g.held_weights, i - 1)[a])));
        }
        out.push('\n');
    }
    // The key's trade counter: sign changes of the target between successive decisions, from the entry decision on.
    let decs: Vec<usize> = (0..g.n_bars()).filter(|&t| g.decision[t] && g.dates[t] >= entry).collect();
    let mut flips = 0u64;
    for pair in decs.windows(2) {
        for a in 0..syms.len() {
            let sign = |t: usize| g.row(&g.target_weights, t)[a].partial_cmp(&0.0).unwrap() as i8;
            if sign(pair[0]) != sign(pair[1]) {
                flips += 1;
            }
        }
    }
    (out, flips, g, n)
}

/// The oracle's own entry gate (independent of the crate's `FlatUntil`): cash until `from`.
pub struct Gated<R: WeightRule> {
    pub inner: R,
    pub from: Date,
}

impl<R: WeightRule> WeightRule for Gated<R> {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn impl_version(&self) -> String {
        "gated oracle".into()
    }
    fn universe(&self) -> &[&'static str] {
        self.inner.universe()
    }
    fn decision_schedule(&self) -> DecisionSchedule {
        self.inner.decision_schedule()
    }
    fn rebalance_policy(&self) -> RebalancePolicy {
        self.inner.rebalance_policy()
    }
    fn min_history_bars(&self) -> usize {
        self.inner.min_history_bars()
    }
    fn target_weights(&self, h: &HistoryView<'_>) -> Result<Vec<f64>, RuleRefusal> {
        if h.date() < self.from {
            Ok(vec![0.0; h.n_assets()])
        } else {
            self.inner.target_weights(h)
        }
    }
}

fn metrics_json(g: &SimResult, n: &SimResult) -> (String, String) {
    let one = |s: &SimResult| {
        let m = s.metrics().unwrap();
        format!(
            "{{\"cagr\": {}, \"max_drawdown\": {}, \"obs\": {}, \"sharpe\": {}, \"vol\": {}}}",
            f(m.cagr),
            f(m.max_drawdown),
            m.n,
            f(m.sharpe),
            f(m.vol)
        )
    };
    (one(g), one(n))
}

fn returns_csv(dates: &[Date], rets: &[f64], header: &str) -> String {
    let mut out = format!("{header}\n");
    for (dt, r) in dates.iter().zip(rets) {
        out.push_str(&format!("{dt},{}\n", f(*r)));
    }
    out
}

/// The set of files of a synthetic fixture directory: name -> bytes. The manifest is added by [`with_manifest`].
pub struct SyntheticFiles {
    pub files: BTreeMap<String, Vec<u8>>,
}

pub fn synthetic_files() -> SyntheticFiles {
    let etf = s1_panel();
    let cry = s3_panel();
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    files.insert(F_CANDLES.into(), SYNTH_CSV.as_bytes().to_vec());

    let (s1_csv, flips1, g1, n1) = perbar_csv(&etf, OracleS1, s1_cfg, &ETF);
    let (s3_csv, flips3, g3, n3) = perbar_csv(&cry, OracleS3, s3_cfg, &CRY);
    files.insert("key/S1_etf_trend_faber_perbar.csv".into(), s1_csv.into_bytes());
    files.insert("key/S3_crypto_trend_100d_perbar.csv".into(), s3_csv.into_bytes());

    let (m1g, m1n) = metrics_json(&g1, &n1);
    let (m3g, m3n) = metrics_json(&g3, &n3);
    let metrics = format!(
        "{{\n \"S1\": {{\"flips\": {flips1}, \"gross\": {m1g}, \"net\": {m1n}}},\n \"S3\": {{\"flips\": {flips3}, \"gross\": {m3g}, \"net\": {m3n}}}\n}}\n"
    );
    files.insert(F_KEY_METRICS.into(), metrics.into_bytes());

    // The shadow's saved returns: the committed PYTHON key of weightsim (independent of every Rust code path here).
    let (sd1, sr1) = parse_returns(PY_KEY_S1_RETURNS);
    let (sd3, sr3) = parse_returns(PY_KEY_S3_RETURNS);
    files.insert("shadow_saved/shadow_S1_daily_returns.csv".into(), returns_csv(&sd1, &sr1, ",ret").into_bytes());
    files.insert("shadow_saved/shadow_S3_daily_returns.csv".into(), returns_csv(&sd3, &sr3, "date,ret").into_bytes());

    // Placeholder mutants file; `with_expected_mutants` replaces it.
    files.insert(F_MUTANTS.into(), b"{\"mutants\": {\"placeholder\": {\"caught_by\": [], \"escapes_tier1\": false, \"common_days\": 0, \"corr\": 0, \"d_sharpe\": 0, \"d_cagr_pp\": 0, \"mutant_sharpe\": 0, \"tier2\": {\"max_abs_return_diff\": 0}, \"tier3\": null}}}\n".to_vec());
    SyntheticFiles { files }
}

/// Add `MANIFEST.json` (sha256 and size of every file) and return the pins that match it.
pub fn with_manifest(mut s: SyntheticFiles) -> (SyntheticFiles, String, String) {
    s.files.remove(F_MANIFEST);
    let mut body = String::from("{\n \"files\": {\n");
    let mut first = true;
    for (name, bytes) in &s.files {
        if !first {
            body.push_str(",\n");
        }
        first = false;
        body.push_str(&format!("  \"{name}\": {{\"bytes\": {}, \"sha256\": \"{}\"}}", bytes.len(), sha256_hex(bytes)));
    }
    body.push_str("\n },\n \"schema\": \"synthetic_manifest\"\n}\n");
    let manifest_sha = sha256_hex(body.as_bytes());
    let candles_sha = sha256_hex(&s.files[F_CANDLES]);
    s.files.insert(F_MANIFEST.into(), body.into_bytes());
    (s, manifest_sha, candles_sha)
}

pub fn load(
    files: &SyntheticFiles,
    manifest_sha: &str,
    candles_sha: &str,
) -> Result<Fixtures, weightsim_rules::ladder::LadderError> {
    let reader = |name: &str| -> Result<Vec<u8>, String> {
        files.files.get(name).cloned().ok_or_else(|| format!("no such file {name}"))
    };
    Fixtures::load(&reader, Pins { manifest_sha256: manifest_sha, candles_sha256: candles_sha })
}

fn json_f(v: f64) -> String {
    assert!(v.is_finite(), "expected a finite number in the synthetic mutants file");
    format!("{v:?}")
}

fn opt_json(v: Option<f64>) -> String {
    v.map_or("null".to_string(), json_f)
}

/// Build the synthetic `mutants.json` from THIS crate's own mutant runs (so it is self-consistent, not independent;
/// what it exercises is the loader, the comparison machinery and the mismatch detection).
pub fn with_expected_mutants(base: SyntheticFiles) -> (SyntheticFiles, String, String) {
    let (with_m, ms, cs) = with_manifest(base);
    let fx = load(&with_m, &ms, &cs).expect("synthetic fixtures load");
    let mut body = String::from("{\n \"mutants\": {\n");
    let mut first = true;
    for m in Mutant::ALL {
        let key = match m.sleeve() {
            MutantSleeve::S1 => &fx.s1,
            MutantSleeve::S3 => &fx.s3,
        };
        let run = run_mutant(&fx, m).unwrap();
        let c = compare(&key_rows_for_mutants(key), &run.rows).unwrap();
        let caught: Vec<String> = c.failed_tiers().iter().map(|t| format!("\"{t}\"")).collect();
        let t3 = match &c.tier3 {
            Some(t) => format!(
                "{{\"agreement\": {}, \"cells\": {}, \"disagreeing_cells\": {}}}",
                json_f(t.agreement),
                t.cells,
                t.disagreeing
            ),
            None => "null".to_string(),
        };
        if !first {
            body.push_str(",\n");
        }
        first = false;
        body.push_str(&format!(
            "  \"{}\": {{\"caught_by\": [{}], \"common_days\": {}, \"corr\": {}, \"d_cagr_pp\": {}, \"d_sharpe\": {}, \"escapes_tier1\": {}, \"mutant_sharpe\": {}, \"tier2\": {{\"max_abs_return_diff\": {}, \"max_abs_w_held_diff\": {}, \"max_abs_w_target_diff\": {}}}, \"tier3\": {}}}",
            m.name(),
            caught.join(", "),
            c.common_days,
            json_f(c.corr),
            json_f(c.d_cagr_pp),
            json_f(c.d_sharpe),
            c.bands_pass,
            json_f(c.run_sharpe),
            json_f(c.max_abs_ret_diff),
            opt_json(c.max_abs_w_held_diff),
            opt_json(c.max_abs_w_target_diff),
            t3
        ));
    }
    body.push_str("\n }\n}\n");
    let mut files = with_m;
    files.files.insert(F_MUTANTS.into(), body.into_bytes());
    with_manifest(files)
}

/// A ready synthetic fixture set: (files, manifest sha, candles sha).
pub fn synthetic_ready() -> (SyntheticFiles, String, String) {
    with_expected_mutants(synthetic_files())
}

static READY: std::sync::OnceLock<(SyntheticFiles, String, String)> = std::sync::OnceLock::new();

/// The synthetic fixture set, built once per test binary.
pub fn ready() -> &'static (SyntheticFiles, String, String) {
    READY.get_or_init(synthetic_ready)
}

/// A fresh, verified copy of the synthetic fixtures.
pub fn fixtures() -> Fixtures {
    let (files, ms, cs) = ready();
    load(files, ms, cs).expect("synthetic fixtures load")
}
