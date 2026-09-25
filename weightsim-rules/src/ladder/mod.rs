//! The replication ladder: certify the backtester (weightsim + these adapters) against the pinned answer key.
//!
//! Implements Stage T3 of `product-mandate/BACKTESTER_TRUTH_DESIGN.md` (sections 1.4, 4.4, 6.1) and the checks of
//! pre-registration Amendment 11 exactly as written: Tier I (bands, gross and net), Tier II (per-bar identity),
//! Tier III (weight agreement), Tier IV (the eight named mutants must each FAIL certification, and the tier that catches
//! each must match `mutants.json`), plus the Layer C canaries and the causality and determinism harnesses of
//! `weightsim::harness` run on the real fixture.
//!
//! Everything here is a pure function of the fixture bytes: the only I/O is reading the fixture files, through
//! [`Fixtures`]. [`run_ladder`] returns a [`LadderReport`] whose failed checks are recorded, not thrown;
//! [`self_test`] turns any failed check into an `Err`.

pub mod checks;
pub mod fixtures;
pub mod json;
pub mod mutants;
pub mod runner;

use std::fmt;
use std::path::Path;

use weightsim::harness::{check_cost_identity, check_determinism, check_poisoning, check_rule_truncation};
use weightsim::{answer_key_metrics, sha256_hex, CostModel, Date, Panel, WeightRule};

pub use checks::{Comparison, SeriesRows, Tier3};
pub use fixtures::{Fixtures, LadderError, Pins};
pub use mutants::Mutant;

use crate::adapters::{CryptoTrendRule, EtfTrendRule, FlatUntil};
use checks::{compare, trades_within_band};
use fixtures::{ExpectedMutant, RecordedMetrics, SleeveKey};
use runner::{
    entry_date, flips_by_key_convention, key_rows, key_rows_for_mutants, rows_from_sim, run_gross_and_net,
    sleeve_config, BaseRuns, Basis,
};

/// Layer C canary: the same-day-peek S3 mutant's own Sharpe must read 3.39 +/- 0.05 (Amendment 1).
pub const CANARY_PEEK_SHARPE: (f64, f64) = (3.39, 0.05);
/// Layer C canary: the extra-delay S3 mutant's own Sharpe must read 2.16 +/- 0.05.
pub const CANARY_DELAY_SHARPE: (f64, f64) = (2.16, 0.05);
/// Layer C: net-versus-gross final equity within 10% of turnover x cost.
pub const COST_IDENTITY_MAX_REL_GAP: f64 = 0.10;

/// The key's per-bar files must reproduce the metrics recorded next to them (`key_metrics.json`, 12 decimals) to this
/// absolute tolerance. A consistency cross-check of the key itself, not one of the tiers.
pub const KEY_SELF_CONSISTENCY_TOL: f64 = 1e-8;

/// The Layer C cost identity: every bar's cost equals rate x traded notional (exactly, to rounding), the total agrees
/// with rate x total traded to 1e-12 relative, and the net-versus-gross drag is within 10% of turnover x cost. A NaN
/// gap (nothing was traded) is not a pass.
pub fn cost_identity_holds(max_bar_error: f64, total_rel_error: f64, relative_gap: f64) -> bool {
    max_bar_error <= 1e-15 && total_rel_error <= 1e-12 && relative_gap <= COST_IDENTITY_MAX_REL_GAP
}

/// One named pass/fail line of the ladder.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Tier I-III numbers of one basis (gross or net) of one sleeve.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisReport {
    pub basis: &'static str,
    pub cmp: Comparison,
    /// The run's signal flips (the key's trade counter), summed over assets, and the key's.
    pub flips_run: u64,
    pub flips_key: u64,
    pub trades_rel_gap: f64,
    pub trades_ok: bool,
    /// Tier I = the three return bands AND the trades band.
    pub tier1_pass: bool,
    pub tier2_pass: bool,
    pub tier3_pass: bool,
    /// The simulator's own counted window equals the key's bars exactly.
    pub window_matches_key: bool,
    pub series_sha256: String,
}

/// Everything about one sleeve (S1 or S3).
#[derive(Clone, Debug, PartialEq)]
pub struct SleeveReport {
    pub code: &'static str,
    pub rule_id: &'static str,
    pub gross: BasisReport,
    pub net: BasisReport,
    /// Largest |gross return - the shadow's saved return| over the key's bars (Tier II against the original key).
    pub vs_shadow_saved_max_abs: f64,
    /// Largest difference between the key's recorded metrics (`key_metrics.json`) and the metrics recomputed from the
    /// key's own per-bar returns, both bases.
    pub key_self_consistency_max_abs: f64,
    pub cost_max_bar_error: f64,
    pub cost_total_vs_rate_times_traded_rel: f64,
    pub cost_actual_drag: f64,
    pub cost_predicted_drag: f64,
    pub cost_relative_gap: f64,
    pub poisoning_bar: usize,
    pub poisoning_mismatches: usize,
    pub truncation_samples: usize,
    pub truncation_disagreements: usize,
    pub determinism_mismatches: usize,
}

/// One mutant's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct MutantReport {
    pub name: &'static str,
    pub description: &'static str,
    pub cmp: Comparison,
    /// Tiers this mutant fails (vocabulary of `mutants.json`); empty means it was NOT caught.
    pub caught_by: Vec<&'static str>,
    pub flips: Option<u64>,
    pub series_sha256: Option<String>,
    /// What `mutants.json` says (None when the fixture set has no such entry).
    pub expected_caught_by: Option<Vec<String>>,
    /// Disagreements with `mutants.json` (empty = the mutant reproduces the recorded numbers and tiers).
    pub mismatches: Vec<String>,
}

impl MutantReport {
    pub fn caught(&self) -> bool {
        !self.caught_by.is_empty()
    }
}

/// The full ladder outcome.
#[derive(Clone, Debug)]
pub struct LadderReport {
    pub manifest_sha256: String,
    pub candles_sha256: String,
    pub verified_files: usize,
    pub sleeves: Vec<SleeveReport>,
    pub mutants: Vec<MutantReport>,
    pub checks: Vec<Check>,
    /// sha256 over the base runs' series digests, the mutant runs' series digests and every check's name and verdict.
    pub digest: String,
}

impl LadderReport {
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|c| c.passed)
    }
    pub fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }
    /// The four base-run series digests, in order S1 gross, S1 net, S3 gross, S3 net.
    pub fn series_digests(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for s in &self.sleeves {
            out.push((format!("{} gross", s.code), s.gross.series_sha256.clone()));
            out.push((format!("{} net", s.code), s.net.series_sha256.clone()));
        }
        out
    }
}

/// Why `self_test` failed.
#[derive(Debug)]
pub enum SelfTestError {
    /// The fixtures or a simulation could not be used at all.
    Infrastructure(LadderError),
    /// The ladder ran and at least one check failed.
    Failed(Box<LadderReport>),
}

impl fmt::Display for SelfTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelfTestError::Infrastructure(e) => write!(f, "ladder self-test could not run: {e}"),
            SelfTestError::Failed(r) => {
                writeln!(f, "ladder self-test FAILED ({} failed checks):", r.failures().len())?;
                for c in r.failures() {
                    writeln!(f, "  - {}: {}", c.name, c.detail)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for SelfTestError {}

/// Run the whole ladder on verified fixtures and return `Ok(report)` only if EVERY check passed.
pub fn self_test(fx: &Fixtures) -> Result<LadderReport, SelfTestError> {
    let report = run_ladder(fx).map_err(SelfTestError::Infrastructure)?;
    if report.passed() {
        Ok(report)
    } else {
        Err(SelfTestError::Failed(Box::new(report)))
    }
}

/// `Fixtures::from_dir` (real pins) followed by [`self_test`].
pub fn self_test_dir(dir: &Path) -> Result<LadderReport, SelfTestError> {
    let fx = Fixtures::from_dir(dir).map_err(SelfTestError::Infrastructure)?;
    self_test(&fx)
}

struct Collector {
    checks: Vec<Check>,
}

impl Collector {
    fn push(&mut self, name: impl Into<String>, passed: bool, detail: impl Into<String>) {
        self.checks.push(Check { name: name.into(), passed, detail: detail.into() });
    }
}

fn recorded_max_diff(key: &SleeveKey, recorded: &[RecordedMetrics; 2]) -> Result<f64, LadderError> {
    let dates: Vec<Date> = key.bars.iter().map(|b| b.date).collect();
    let mut worst = 0.0f64;
    for (i, pick) in [(0usize, true), (1usize, false)] {
        let rets: Vec<f64> = key.bars.iter().map(|b| if pick { b.ret_gross } else { b.ret_net }).collect();
        let m = answer_key_metrics(&dates, &rets)
            .ok_or_else(|| LadderError::Inconsistent("key metrics undefined".into()))?;
        let r = &recorded[i];
        for d in [
            (m.n as f64 - r.obs).abs(),
            (m.cagr - r.cagr).abs(),
            (m.vol - r.vol).abs(),
            (m.sharpe - r.sharpe).abs(),
            (m.max_drawdown - r.max_drawdown).abs(),
        ] {
            if d.is_nan() {
                return Ok(f64::NAN);
            }
            worst = worst.max(d);
        }
    }
    Ok(worst)
}

fn shadow_max_diff(gross_rows: &SeriesRows, shadow: &[(Date, f64)]) -> f64 {
    let mut worst = 0.0f64;
    let (mut i, mut j) = (0, 0);
    let mut matched = 0usize;
    while i < gross_rows.dates.len() && j < shadow.len() {
        if gross_rows.dates[i] == shadow[j].0 {
            let d = (gross_rows.ret[i] - shadow[j].1).abs();
            if d.is_nan() {
                return f64::NAN;
            }
            worst = worst.max(d);
            matched += 1;
            i += 1;
            j += 1;
        } else if gross_rows.dates[i] < shadow[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    // The shadow file must cover the key's bars exactly (same count), otherwise the comparison is meaningless.
    if matched != gross_rows.dates.len() || matched != shadow.len() {
        return f64::INFINITY;
    }
    worst
}

fn basis_report(
    sim: &weightsim::SimResult,
    rows: &SeriesRows,
    key: &SleeveKey,
    basis: Basis,
) -> Result<BasisReport, LadderError> {
    let cmp = compare(&key_rows(key, basis), rows)?;
    let key_dates: Vec<Date> = key.bars.iter().map(|b| b.date).collect();
    let flips_run = flips_by_key_convention(sim, key_dates[0], key_dates[key_dates.len() - 1]);
    let (trades_rel_gap, trades_ok) = trades_within_band(flips_run, key.flips);
    Ok(BasisReport {
        basis: basis.name(),
        flips_run,
        flips_key: key.flips,
        trades_rel_gap,
        trades_ok,
        tier1_pass: cmp.bands_pass && trades_ok,
        tier2_pass: cmp.tier2_pass,
        tier3_pass: cmp.tier3_pass(),
        window_matches_key: sim.window_dates() == key_dates.as_slice(),
        series_sha256: sim.series_sha256.clone(),
        cmp,
    })
}

/// Run the per-sleeve checks (Tiers I-III gross and net, the shadow and key cross-checks, cost identity, causality,
/// determinism) for ANY rule on the fixtures, and return the sleeve report with its checks. `run_ladder` does this for
/// the two library rules; it is public so a test can hand it a deliberately wrong or leaky rule and watch each check
/// fail (a certification is only worth something if it can fail).
pub fn certify_sleeve<R, F>(
    fx: &Fixtures,
    sleeve: mutants::MutantSleeve,
    make_rule: &F,
) -> Result<(SleeveReport, Vec<Check>), LadderError>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    let mut col = Collector { checks: Vec::new() };
    let (panel, key, recorded, shadow) = match sleeve {
        mutants::MutantSleeve::S1 => (&fx.etf_panel, &fx.s1, &fx.recorded_s1, &fx.shadow_s1),
        mutants::MutantSleeve::S3 => (&fx.crypto_panel, &fx.s3, &fx.recorded_s3, &fx.shadow_s3),
    };
    let (report, _) = run_sleeve(&mut col, panel, make_rule, key, recorded, shadow)?;
    Ok((report, col.checks))
}

/// Everything about one sleeve except the mutants.
#[allow(clippy::too_many_arguments)]
fn run_sleeve<R, F>(
    col: &mut Collector,
    panel: &Panel,
    make_rule: &F,
    key: &SleeveKey,
    recorded: &[RecordedMetrics; 2],
    shadow: &[(Date, f64)],
) -> Result<(SleeveReport, BaseRuns), LadderError>
where
    R: WeightRule,
    F: Fn(&Panel) -> R,
{
    let code = key.code;
    let rule = make_rule(panel);
    let runs = run_gross_and_net(panel, &rule, key)?;
    let (start, end) = (key.bars[0].date, key.bars[key.bars.len() - 1].date);
    let g_rows = rows_from_sim(&runs.gross, start, end, true)?;
    let n_rows = rows_from_sim(&runs.net, start, end, true)?;
    let gross = basis_report(&runs.gross, &g_rows, key, Basis::Gross)?;
    let net = basis_report(&runs.net, &n_rows, key, Basis::Net)?;

    for b in [&gross, &net] {
        let c = &b.cmp;
        let tag = format!("{code}.{}", b.basis);
        col.push(
            format!("{tag}.window"),
            b.window_matches_key && c.covers_key_exactly(),
            format!(
                "key {} bars, run {} bars, common {}, simulator window equals key bars: {}",
                c.key_days, c.run_days, c.common_days, b.window_matches_key
            ),
        );
        col.push(
            format!("{tag}.tier1.bands"),
            c.bands_pass,
            format!(
                "corr {:.6} (>= 0.99), dSharpe {:+.6} (<= 0.05), dCAGR {:+.6} pp (<= 0.5)",
                c.corr, c.d_sharpe, c.d_cagr_pp
            ),
        );
        col.push(
            format!("{tag}.tier1.trades"),
            b.trades_ok,
            format!("signal flips run {} vs key {} (gap {:.4}, <= 0.05)", b.flips_run, b.flips_key, b.trades_rel_gap),
        );
        col.push(
            format!("{tag}.tier2.identity"),
            c.tier2_pass,
            format!(
                "max|ret| {:e}, max|equity| {}, max|cost| {}, max|traded| {}, max|w_target| {}, max|w_held| {} (all <= 1e-9)",
                c.max_abs_ret_diff,
                fmt_opt(c.max_abs_equity_diff),
                fmt_opt(c.max_abs_cost_diff),
                fmt_opt(c.max_abs_traded_diff),
                fmt_opt(c.max_abs_w_target_diff),
                fmt_opt(c.max_abs_w_held_diff)
            ),
        );
        col.push(
            format!("{tag}.tier3.weights"),
            b.tier3_pass && c.tier3.is_some(),
            match &c.tier3 {
                Some(t) => format!("{}/{} cells agree ({:.6}, >= 0.98)", t.cells - t.disagreeing, t.cells, t.agreement),
                None => "no weights compared".to_string(),
            },
        );
    }
    col.push(
        format!("{code}.trades.exact"),
        gross.flips_run == key.flips && net.flips_run == key.flips,
        format!("signal flips gross {} net {} vs key {}", gross.flips_run, net.flips_run, key.flips),
    );

    // Tier II against the ORIGINAL key (the shadow's saved returns).
    let vs_shadow = shadow_max_diff(&g_rows, shadow);
    col.push(
        format!("{code}.gross.vs_shadow_saved"),
        vs_shadow <= checks::TIER2_TOL,
        format!("max|gross ret - shadow saved ret| = {vs_shadow:e} over {} bars", g_rows.dates.len()),
    );

    // The key must equal its own recorded metrics.
    let self_diff = recorded_max_diff(key, recorded)?;
    col.push(
        format!("{code}.key_self_consistency"),
        self_diff <= KEY_SELF_CONSISTENCY_TOL,
        format!("key per-bar returns vs key_metrics.json, max abs difference {self_diff:e} (<= {KEY_SELF_CONSISTENCY_TOL:e})"),
    );

    // Layer C (d): cost identity.
    let ci = check_cost_identity(&runs.gross, &runs.net, &CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE);
    let total_rel = (ci.total_cost - ci.rate_times_total_traded).abs() / ci.rate_times_total_traded.abs().max(1e-300);
    col.push(
        format!("{code}.cost_identity"),
        cost_identity_holds(ci.max_bar_cost_error, total_rel, ci.relative_gap),
        format!(
            "max per-bar |cost - 0.001 x traded| {:e}; total cost vs rate x traded rel {:e}; drag {:.6} vs turnover x cost {:.6} (gap {:.4}, <= {})",
            ci.max_bar_cost_error, total_rel, ci.actual_drag, ci.predicted_drag, ci.relative_gap, COST_IDENTITY_MAX_REL_GAP
        ),
    );

    // Causality on the real fixture: poisoning of the whole simulator, truncation of the rule, determinism.
    let cfg = sleeve_config(key, CostModel::ZERO);
    let n = panel.n_bars();
    // Poison the future at three cut points: a leaky decision only shows on the last kept bar, and one cut can miss it.
    let poison_bars = [n / 4, n / 2, 3 * n / 4];
    let mut poison_mismatches = 0usize;
    let mut poison_clean = true;
    for &cut in &poison_bars {
        let pr = check_poisoning(make_rule, panel, &cfg, cut, 0x5EED_1AD0 + cut as u64)
            .map_err(|e| LadderError::Sim(format!("poisoning run: {e}")))?;
        poison_clean = poison_clean && pr.is_clean() && pr.compared_through_bar == cut;
        poison_mismatches += pr.mismatches.len();
    }
    col.push(
        format!("{code}.causality.poisoning"),
        poison_clean,
        format!(
            "prices after bars {poison_bars:?} replaced by garbage: {poison_mismatches} mismatching fields dated <= the cut"
        ),
    );
    let lo = 300.min(n / 2);
    let samples: Vec<usize> = (0..40).map(|i| lo + i * (n - 1 - lo) / 40).collect();
    let bad = check_rule_truncation(make_rule, panel, &samples);
    col.push(
        format!("{code}.causality.truncation"),
        bad.is_empty(),
        format!("{} sampled decision bars, {} disagree with the truncated panel", samples.len(), bad.len()),
    );
    let det =
        check_determinism(make_rule, panel, &cfg, 2).map_err(|e| LadderError::Sim(format!("determinism run: {e}")))?;
    col.push(
        format!("{code}.determinism"),
        det.is_empty(),
        format!("second execution differs in {} fields (series digest included)", det.len()),
    );

    let report = SleeveReport {
        code,
        rule_id: key.rule_id,
        gross,
        net,
        vs_shadow_saved_max_abs: vs_shadow,
        key_self_consistency_max_abs: self_diff,
        cost_max_bar_error: ci.max_bar_cost_error,
        cost_total_vs_rate_times_traded_rel: total_rel,
        cost_actual_drag: ci.actual_drag,
        cost_predicted_drag: ci.predicted_drag,
        cost_relative_gap: ci.relative_gap,
        poisoning_bar: n / 2,
        poisoning_mismatches: poison_mismatches,
        truncation_samples: samples.len(),
        truncation_disagreements: bad.len(),
        determinism_mismatches: det.len(),
    };
    Ok((report, runs))
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:e}"),
        None => "n/a".to_string(),
    }
}

/// Compare a mutant's numbers with `mutants.json`. Returns the list of disagreements (empty = reproduced).
fn mismatches_with_expected(cmp: &Comparison, caught_by: &[&'static str], e: &ExpectedMutant) -> Vec<String> {
    let mut out = Vec::new();
    let mut got: Vec<&str> = caught_by.to_vec();
    got.sort_unstable();
    let mut want: Vec<&str> = e.caught_by.iter().map(String::as_str).collect();
    want.sort_unstable();
    if got != want {
        out.push(format!("caught_by {got:?}, mutants.json says {want:?}"));
    }
    if cmp.bands_pass != e.escapes_tier1 {
        out.push(format!(
            "passes every Tier I band: {}, mutants.json escapes_tier1 = {}",
            cmp.bands_pass, e.escapes_tier1
        ));
    }
    if cmp.common_days != e.common_days {
        out.push(format!("common days {}, mutants.json {}", cmp.common_days, e.common_days));
    }
    let close = |name: &str, got: f64, want: f64, tol: f64, out: &mut Vec<String>| {
        if !((got - want).abs() <= tol) {
            out.push(format!("{name} {got}, mutants.json {want} (tol {tol})"));
        }
    };
    close("corr", cmp.corr, e.corr, 2e-6, &mut out);
    close("dSharpe", cmp.d_sharpe, e.d_sharpe, 2e-6, &mut out);
    close("dCAGR pp", cmp.d_cagr_pp, e.d_cagr_pp, 5e-6, &mut out);
    close("mutant Sharpe", cmp.run_sharpe, e.mutant_sharpe, 2e-6, &mut out);
    close("max|ret diff|", cmp.max_abs_ret_diff, e.max_abs_return_diff, 1e-9 + 1e-6 * e.max_abs_return_diff, &mut out);
    match (cmp.max_abs_w_target_diff, e.max_abs_w_target_diff) {
        (Some(g), Some(w)) => close("max|w_target diff|", g, w, 1e-9 + 1e-6 * w, &mut out),
        (None, None) => {}
        (g, w) => out.push(format!("w_target diff {g:?}, mutants.json {w:?}")),
    }
    match (cmp.max_abs_w_held_diff, e.max_abs_w_held_diff) {
        (Some(g), Some(w)) => close("max|w_held diff|", g, w, 1e-9 + 1e-6 * w, &mut out),
        (None, None) => {}
        (g, w) => out.push(format!("w_held diff {g:?}, mutants.json {w:?}")),
    }
    match (&cmp.tier3, &e.tier3) {
        (Some(t), Some((agree, dis, cells))) => {
            if t.cells != *cells || t.disagreeing != *dis || (t.agreement - agree).abs() > 2e-6 {
                out.push(format!(
                    "Tier III {}/{} cells ({:.6}), mutants.json {}/{} ({:.6})",
                    t.cells - t.disagreeing,
                    t.cells,
                    t.agreement,
                    cells - dis,
                    cells,
                    agree
                ));
            }
        }
        (None, None) => {}
        (g, w) => out.push(format!("Tier III {g:?}, mutants.json {w:?}")),
    }
    out
}

/// Switches of a ladder run. The default is the certification (everything on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LadderOptions {
    /// Check the Layer C canaries (same-day-peek S3 Sharpe 3.39 +/- 0.05, extra-delay 2.16 +/- 0.05). They are numbers
    /// of the REAL pinned data; synthetic fixture sets switch them off.
    pub check_canaries: bool,
}

impl Default for LadderOptions {
    fn default() -> Self {
        LadderOptions { check_canaries: true }
    }
}

/// Run the whole ladder (S1 and S3, gross and net, Tiers I-III, the eight mutants, canaries, causality) on verified
/// fixtures. Failed checks are recorded in the report; only an unusable fixture or a simulation error is an `Err`.
pub fn run_ladder(fx: &Fixtures) -> Result<LadderReport, LadderError> {
    run_ladder_with(fx, &LadderOptions::default())
}

/// [`run_ladder`] with explicit options.
pub fn run_ladder_with(fx: &Fixtures, opts: &LadderOptions) -> Result<LadderReport, LadderError> {
    let mut col = Collector { checks: Vec::new() };
    col.push(
        "fixtures.verified",
        true,
        format!(
            "MANIFEST.json sha256 {} matches its pin; {} manifest files verified; candles sha256 {}",
            fx.manifest_sha256,
            fx.verified_files.len(),
            fx.candles_sha256
        ),
    );

    // Each sleeve starts flat at the bar before its window (the key ledger's convention, see `FlatUntil`).
    let entry1 = entry_date(&fx.etf_panel, &fx.s1)?;
    let entry3 = entry_date(&fx.crypto_panel, &fx.s3)?;
    let (s1, _) = run_sleeve(
        &mut col,
        &fx.etf_panel,
        &|_p: &Panel| FlatUntil::new(EtfTrendRule, entry1),
        &fx.s1,
        &fx.recorded_s1,
        &fx.shadow_s1,
    )?;
    let (s3, _) = run_sleeve(
        &mut col,
        &fx.crypto_panel,
        &|_p: &Panel| FlatUntil::new(CryptoTrendRule, entry3),
        &fx.s3,
        &fx.recorded_s3,
        &fx.shadow_s3,
    )?;

    // Tier IV
    let mut mutant_reports = Vec::new();
    for m in Mutant::ALL {
        let key = match m.sleeve() {
            mutants::MutantSleeve::S1 => &fx.s1,
            mutants::MutantSleeve::S3 => &fx.s3,
        };
        let run = mutants::run_mutant(fx, m)?;
        let cmp = compare(&key_rows_for_mutants(key), &run.rows)?;
        let caught_by = cmp.failed_tiers();
        let expected = fx.expected_mutants.iter().find(|e| e.name == m.name());
        let mismatches = match expected {
            Some(e) => mismatches_with_expected(&cmp, &caught_by, e),
            None => vec!["no entry in mutants.json".to_string()],
        };
        col.push(
            format!("mutant.{}.caught", m.name()),
            !caught_by.is_empty(),
            format!(
                "{}: caught by {:?}; corr {:.4}, dSharpe {:+.3}, dCAGR {:+.2} pp, max|ret diff| {:.3e}",
                m.description(),
                caught_by,
                cmp.corr,
                cmp.d_sharpe,
                cmp.d_cagr_pp,
                cmp.max_abs_ret_diff
            ),
        );
        if cmp.bands_pass {
            col.push(
                format!("mutant.{}.tier2_catches_band_escaper", m.name()),
                !cmp.tier2_pass,
                "passes every Tier I band, so Tier II MUST catch it (Amendment 11 requirement b)".to_string(),
            );
        }
        col.push(
            format!("mutant.{}.matches_mutants_json", m.name()),
            mismatches.is_empty(),
            if mismatches.is_empty() {
                "caught_by, Tier I verdict, common days, corr, dSharpe, dCAGR, Sharpe, diffs and Tier III cells reproduce mutants.json"
                    .to_string()
            } else {
                mismatches.join("; ")
            },
        );
        mutant_reports.push(MutantReport {
            name: m.name(),
            description: m.description(),
            caught_by,
            flips: run.flips,
            series_sha256: run.series_sha256,
            expected_caught_by: expected.map(|e| e.caught_by.clone()),
            mismatches,
            cmp,
        });
    }
    col.push(
        "tier4.all_mutants_caught",
        mutant_reports.iter().all(MutantReport::caught) && mutant_reports.len() == Mutant::ALL.len(),
        format!(
            "{} of {} mutants fail certification",
            mutant_reports.iter().filter(|m| m.caught()).count(),
            Mutant::ALL.len()
        ),
    );
    if opts.check_canaries {
        for (check, mutant, (centre, tol)) in [
            ("canary.s3_same_day_peek_sharpe", Mutant::S3SameDayPeek, CANARY_PEEK_SHARPE),
            ("canary.s3_extra_delay_sharpe", Mutant::S3ExtraDelay, CANARY_DELAY_SHARPE),
        ] {
            match mutant_reports.iter().find(|m| m.name == mutant.name()) {
                Some(m) => col.push(
                    check,
                    (m.cmp.run_sharpe - centre).abs() <= tol,
                    format!("mutant Sharpe {:.4}, must read {centre} +/- {tol}", m.cmp.run_sharpe),
                ),
                None => col.push(check, false, "the mutant was not run"),
            }
        }
    }

    let mut h = String::new();
    h.push_str("weightsim-rules-ladder-v1\n");
    h.push_str(&fx.manifest_sha256);
    h.push('\n');
    h.push_str(&fx.candles_sha256);
    h.push('\n');
    for s in [&s1, &s3] {
        h.push_str(&format!("{} {} {}\n", s.code, s.gross.series_sha256, s.net.series_sha256));
    }
    for m in &mutant_reports {
        h.push_str(&format!("{} {:?} {}\n", m.name, m.caught_by, m.series_sha256.as_deref().unwrap_or("-")));
    }
    for c in &col.checks {
        h.push_str(&format!("{} {}\n", c.name, u8::from(c.passed)));
    }
    let digest = sha256_hex(h.as_bytes());

    Ok(LadderReport {
        manifest_sha256: fx.manifest_sha256.clone(),
        candles_sha256: fx.candles_sha256.clone(),
        verified_files: fx.verified_files.len(),
        sleeves: vec![s1, s3],
        mutants: mutant_reports,
        checks: col.checks,
        digest,
    })
}

impl fmt::Display for LadderReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "REPLICATION LADDER (weightsim-rules {})", env!("CARGO_PKG_VERSION"))?;
        writeln!(f, "  manifest sha256 {}", self.manifest_sha256)?;
        writeln!(f, "  candles  sha256 {}  ({} manifest files verified)", self.candles_sha256, self.verified_files)?;
        for s in &self.sleeves {
            writeln!(
                f,
                "\n{} {} :: vs shadow saved returns max|diff| {:e}; key self-consistency {:e}",
                s.code, s.rule_id, s.vs_shadow_saved_max_abs, s.key_self_consistency_max_abs
            )?;
            for b in [&s.gross, &s.net] {
                let c = &b.cmp;
                writeln!(
                    f,
                    "  {:5} bars {} | corr {:.6} dSharpe {:+.6} dCAGR {:+.6}pp (Sharpe key {:.6} run {:.6}; CAGR key {:.6} run {:.6}) | flips run {} key {} | tier I {} II {} III {}",
                    b.basis,
                    c.common_days,
                    c.corr,
                    c.d_sharpe,
                    c.d_cagr_pp,
                    c.key_sharpe,
                    c.run_sharpe,
                    c.key_cagr,
                    c.run_cagr,
                    b.flips_run,
                    b.flips_key,
                    verdict(b.tier1_pass),
                    verdict(b.tier2_pass),
                    verdict(b.tier3_pass)
                )?;
                writeln!(
                    f,
                    "        tier II max|ret| {:e} max|equity| {} max|cost| {} max|traded| {} max|w_target| {} max|w_held| {}",
                    c.max_abs_ret_diff,
                    fmt_opt(c.max_abs_equity_diff),
                    fmt_opt(c.max_abs_cost_diff),
                    fmt_opt(c.max_abs_traded_diff),
                    fmt_opt(c.max_abs_w_target_diff),
                    fmt_opt(c.max_abs_w_held_diff)
                )?;
                if let Some(t) = &c.tier3 {
                    writeln!(
                        f,
                        "        tier III {}/{} cells agree ({:.6})",
                        t.cells - t.disagreeing,
                        t.cells,
                        t.agreement
                    )?;
                }
                writeln!(f, "        series sha256 {}", b.series_sha256)?;
            }
            writeln!(
                f,
                "  cost identity: max per-bar error {:e}; drag {:.6} vs turnover x cost {:.6} (gap {:.4}); causality: poisoning at bar {} -> {} mismatches, truncation {}/{} disagree, determinism {} mismatches",
                s.cost_max_bar_error,
                s.cost_actual_drag,
                s.cost_predicted_drag,
                s.cost_relative_gap,
                s.poisoning_bar,
                s.poisoning_mismatches,
                s.truncation_disagreements,
                s.truncation_samples,
                s.determinism_mismatches
            )?;
        }
        writeln!(f, "\nTIER IV (each mutant must FAIL certification; tiers must match mutants.json)")?;
        for m in &self.mutants {
            writeln!(
                f,
                "  {:26} caught by {:?} | corr {:.4} dSharpe {:+.3} dCAGR {:+.2}pp | max|ret| {:.3e} | Sharpe {:.3} | {}",
                m.name,
                m.caught_by,
                m.cmp.corr,
                m.cmp.d_sharpe,
                m.cmp.d_cagr_pp,
                m.cmp.max_abs_ret_diff,
                m.cmp.run_sharpe,
                if m.mismatches.is_empty() { "matches mutants.json".to_string() } else { format!("MISMATCH: {}", m.mismatches.join("; ")) }
            )?;
        }
        writeln!(f, "\nCHECKS ({} total, {} failed)", self.checks.len(), self.failures().len())?;
        for c in &self.checks {
            writeln!(f, "  [{}] {} :: {}", if c.passed { "PASS" } else { "FAIL" }, c.name, c.detail)?;
        }
        writeln!(f, "\nladder digest {}", self.digest)?;
        writeln!(f, "VERDICT: {}", if self.passed() { "PASS" } else { "FAIL" })
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_identity_boundaries() {
        assert!(cost_identity_holds(0.0, 0.0, 0.0));
        // per-bar error: exactly 1e-15 passes, more fails
        assert!(cost_identity_holds(1e-15, 0.0, 0.0));
        assert!(!cost_identity_holds(1.5e-15, 0.0, 0.0));
        // total relative error: exactly 1e-12 passes, more fails
        assert!(cost_identity_holds(0.0, 1e-12, 0.0));
        assert!(!cost_identity_holds(0.0, 1.5e-12, 0.0));
        // relative drag gap: 10% inclusive
        assert!(cost_identity_holds(0.0, 0.0, 0.1));
        assert!(cost_identity_holds(0.0, 0.0, 0.0999));
        assert!(!cost_identity_holds(0.0, 0.0, 0.1001));
        assert!(!cost_identity_holds(0.0, 0.0, f64::NAN));
        assert!(!cost_identity_holds(f64::NAN, 0.0, 0.0));
        assert!(!cost_identity_holds(0.0, f64::NAN, 0.0));
        assert_eq!(COST_IDENTITY_MAX_REL_GAP, 0.10);
    }
}
