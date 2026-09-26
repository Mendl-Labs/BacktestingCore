//! Replicate a library rule on the pinned fixtures and independently RE-VERIFY a stored run (stage T4, slice C1 of
//! `product-mandate/T4_FIXED_RULE_ENDPOINT_PLAN.md`; design `BACKTESTER_TRUTH_DESIGN.md` 3.6, 3.7, 4.2 and 4.4).
//!
//! * [`replicate`] / [`replicate_with`]: run one library rule on the ladder's own base configuration (the sleeve starts
//!   flat at the bar before the key's window, gross run at zero cost, net run at the cost preset, no delay, no scaling,
//!   no cap, no financing) and package the result: the FULL-RESOLUTION series of both runs as plain
//!   [`SeriesColumns`] (what a run store serialises), the two series digests, the refusals and a [`RunSummary`] per
//!   basis.
//! * [`verify`] / [`verify_with`]: given only stored columns (and, optionally, the numbers the store claims for them),
//!   decide whether the run is exactly what this platform computes. Nothing stored is trusted:
//!   1. every column is shape- and finiteness-checked before it is indexed or hashed;
//!   2. rule id, rule implementation version (which contains the `weightsim-rules` crate version), symbols, cost preset
//!      and metric-definition label are compared with what the platform would stamp;
//!   3. the digest of the stored columns is recomputed and compared with the claimed digest, if one is given;
//!   4. the rule is RE-RUN from the fixtures and the stored columns must be bit-identical to the re-run (same digest);
//!   5. the counted window, the answer-key metrics (Sharpe, CAGR, volatility, maximum drawdown, `ppy = n / years`,
//!      ddof 1) and the signal flips are recomputed FROM THE STORED COLUMNS, never read from storage, and compared with
//!      any claimed summary;
//!   6. Tiers I-III of pre-registration Amendment 11 are evaluated on the stored columns, gross and net, with the
//!      certification's own comparison code; Tier IV (the eight named mutants must each fail) is evaluated for the
//!      rule's sleeve unless switched off.
//!
//! Every failure is a typed [`VerifyError`]; a run that is genuine but does not reproduce the key is
//! [`VerifyError::TierFailed`] (it carries the full numbers).
//!
//! The analysis of step 5 and 6 is the certified code path itself (`rows_from_sim`, `basis_report`, `compare`): the
//! stored columns are wrapped in an analysis-only `SimResult` (fields the digest does not cover are left empty and the
//! counted window is DERIVED by `rows_from_sim`'s own rule, not read), so a verified run and a certified run cannot
//! disagree about what a tier means.

use std::collections::BTreeMap;
use std::fmt;

use weightsim::{
    simulate_gross_and_net, CostModel, Date, DecisionSchedule, Metrics, Panel, RebalancePolicy, Refusal, SimResult,
    WeightRule, Window, METRIC_DEFINITIONS,
};

pub use weightsim::{ColumnsError, SeriesColumns};

use super::checks::{compare, CAUGHT_TIER1, CAUGHT_TIER2, CAUGHT_TIER3};
use super::fixtures::{Fixtures, LadderError, SleeveKey};
use super::mutants::{run_mutant, Mutant, MutantSleeve};
use super::runner::{entry_date, flips_by_key_convention, key_rows_for_mutants, rows_from_sim, sleeve_config, Basis};
use super::{basis_report, mismatches_with_expected, BasisReport};
use crate::adapters::{CryptoTrendRule, EtfTrendRule, FlatUntil};

/// The cost preset of the net run of a replication unless the request names another: the certification's flat 10 bps
/// per side (the answer key's own cost model).
pub const DEFAULT_COST_MODEL_ID: &str = CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE.id;

/// The cost preset id of a gross run (always zero cost).
pub const GROSS_COST_MODEL_ID: &str = CostModel::ZERO.id;

/// Absolute tolerance for a claimed CAGR. It is the only summary number that goes through `powf`, whose last bit is
/// not specified across platforms; every other claimed number must match bit for bit.
pub const CLAIMED_CAGR_TOL: f64 = 1e-12;

/// The library rules this module can replicate, by id.
pub const LIBRARY_RULE_IDS: [&str; 2] = ["etf_trend_faber", "crypto_trend_100d"];

// ------------------------------------------------------------------------------------------------------ rule facts

/// Compile-time facts of a library rule, so a caller's registry is a lookup and not a second copy of them.
#[derive(Clone, Debug, PartialEq)]
pub struct RuleFacts {
    pub id: &'static str,
    /// The answer-key sleeve the rule is certified against (`S1` or `S3`).
    pub sleeve_code: &'static str,
    pub universe: Vec<&'static str>,
    pub declared_parameters: BTreeMap<&'static str, String>,
    pub decision_schedule: DecisionSchedule,
    pub rebalance_policy: RebalancePolicy,
    pub min_history_bars: usize,
}

fn facts_of<R: WeightRule>(rule: &R, sleeve_code: &'static str) -> RuleFacts {
    RuleFacts {
        id: rule.id(),
        sleeve_code,
        universe: rule.universe().to_vec(),
        declared_parameters: rule.declared_parameters(),
        decision_schedule: rule.decision_schedule(),
        rebalance_policy: rule.rebalance_policy(),
        min_history_bars: rule.min_history_bars(),
    }
}

/// The facts of a library rule, or `None` for an id this crate does not implement.
pub fn rule_facts(rule_id: &str) -> Option<RuleFacts> {
    if rule_id == EtfTrendRule.id() {
        Some(facts_of(&EtfTrendRule, "S1"))
    } else if rule_id == CryptoTrendRule.id() {
        Some(facts_of(&CryptoTrendRule, "S3"))
    } else {
        None
    }
}

/// A library rule wired to its panel and key on a fixture set.
struct Prepared<'a> {
    rule: Box<dyn WeightRule>,
    panel: &'a Panel,
    key: &'a SleeveKey,
    sleeve: MutantSleeve,
}

fn library_rule<'a>(fx: &'a Fixtures, rule_id: &str) -> Result<Prepared<'a>, ReplicateError> {
    // Each sleeve starts flat at the bar before its window (the key ledger's convention, see `FlatUntil`).
    if rule_id == EtfTrendRule.id() {
        let entry = entry_date(&fx.etf_panel, &fx.s1)?;
        Ok(Prepared {
            rule: Box::new(FlatUntil::new(EtfTrendRule, entry)),
            panel: &fx.etf_panel,
            key: &fx.s1,
            sleeve: MutantSleeve::S1,
        })
    } else if rule_id == CryptoTrendRule.id() {
        let entry = entry_date(&fx.crypto_panel, &fx.s3)?;
        Ok(Prepared {
            rule: Box::new(FlatUntil::new(CryptoTrendRule, entry)),
            panel: &fx.crypto_panel,
            key: &fx.s3,
            sleeve: MutantSleeve::S3,
        })
    } else {
        Err(ReplicateError::UnknownRule { rule_id: rule_id.to_string() })
    }
}

// --------------------------------------------------------------------------------------------------------- summary

/// The headline numbers of one basis, in the answer key's definitions (`answer_key_v1`): `ppy = n / years`, sample
/// standard deviation (ddof 1), `max_drawdown` from `cumprod(1 + r)` with no initial point, over the counted window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunSummary {
    pub n: usize,
    pub first_date: Date,
    pub last_date: Date,
    pub sharpe: f64,
    pub cagr: f64,
    pub vol: f64,
    pub max_drawdown: f64,
    /// The key's trade counter (signal flips, summed over assets).
    pub flips: u64,
}

fn summarize(m: &Metrics, flips: u64) -> RunSummary {
    RunSummary {
        n: m.n,
        first_date: m.first_date,
        last_date: m.last_date,
        sharpe: m.sharpe,
        cagr: m.cagr,
        vol: m.vol,
        max_drawdown: m.max_drawdown,
        flips,
    }
}

// ------------------------------------------------------------------------------------------------------ replicate

/// What a replication runs under. There is deliberately one knob: the cost preset of the NET run, selected by id
/// (no free numeric fields). Everything else is the ladder's fixed configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicationConfig {
    pub cost_model_id: String,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        ReplicationConfig { cost_model_id: DEFAULT_COST_MODEL_ID.to_string() }
    }
}

/// Why a rule could not be replicated.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplicateError {
    UnknownRule {
        rule_id: String,
    },
    UnknownCostPreset {
        cost_model_id: String,
    },
    /// The fixtures cannot express the sleeve or the simulation failed.
    Ladder(LadderError),
}

impl From<LadderError> for ReplicateError {
    fn from(e: LadderError) -> Self {
        ReplicateError::Ladder(e)
    }
}

impl fmt::Display for ReplicateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplicateError::UnknownRule { rule_id } => write!(f, "unknown library rule `{rule_id}`"),
            ReplicateError::UnknownCostPreset { cost_model_id } => write!(f, "unknown cost preset `{cost_model_id}`"),
            ReplicateError::Ladder(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ReplicateError {}

/// A replication, packaged: everything a run store keeps, and nothing that a verifier has to trust.
#[derive(Clone, Debug)]
pub struct ReplicationRun {
    pub rule_id: String,
    pub cost_model_id: String,
    /// What the run stamped as its rule implementation version (contains the `weightsim-rules` crate version).
    pub rule_impl_version: String,
    /// Full-resolution series, gross (zero cost) and net (`cost_model_id`).
    pub gross: SeriesColumns,
    pub net: SeriesColumns,
    /// The simulator's own series digests of the two runs.
    pub gross_sha256: String,
    pub net_sha256: String,
    pub gross_summary: RunSummary,
    pub net_summary: RunSummary,
    /// Refusals recorded during the run (warm-up refusals before the first decision; the ladder aborts on any other).
    pub refusals: Vec<Refusal>,
    /// The fixture set the run used (manifest and candles sha256).
    pub manifest_sha256: String,
    pub candles_sha256: String,
}

impl ReplicationRun {
    /// The digests, summaries and fixture identity of this run as a verifier's claims (what a store would keep next
    /// to the columns).
    pub fn claims(&self) -> Claims {
        Claims {
            gross_sha256: Some(self.gross_sha256.clone()),
            net_sha256: Some(self.net_sha256.clone()),
            gross_summary: Some(self.gross_summary),
            net_summary: Some(self.net_summary),
            manifest_sha256: Some(self.manifest_sha256.clone()),
            candles_sha256: Some(self.candles_sha256.clone()),
        }
    }
}

/// [`replicate_with`] under the default configuration (net = certification 10 bps per side).
pub fn replicate(fx: &Fixtures, rule_id: &str) -> Result<ReplicationRun, ReplicateError> {
    replicate_with(fx, rule_id, &ReplicationConfig::default())
}

/// Run a library rule on the fixtures under the ladder's base configuration and package the result.
pub fn replicate_with(
    fx: &Fixtures,
    rule_id: &str,
    config: &ReplicationConfig,
) -> Result<ReplicationRun, ReplicateError> {
    let cost = CostModel::by_id(&config.cost_model_id)
        .ok_or_else(|| ReplicateError::UnknownCostPreset { cost_model_id: config.cost_model_id.clone() })?;
    let p = library_rule(fx, rule_id)?;
    let cfg = sleeve_config(p.key, cost);
    let (gross, net) = simulate_gross_and_net(p.panel, &*p.rule, &cfg)
        .map_err(|e| ReplicateError::Ladder(LadderError::Sim(e.to_string())))?;
    let (first, last) = key_span(p.key);
    let summary_of = |sim: &SimResult| -> Result<RunSummary, ReplicateError> {
        let m = sim
            .metrics()
            .ok_or_else(|| ReplicateError::Ladder(LadderError::Sim("the run has no counted window".into())))?;
        Ok(summarize(&m, flips_by_key_convention(sim, first, last)))
    };
    Ok(ReplicationRun {
        rule_id: gross.rule_id.clone(),
        cost_model_id: config.cost_model_id.clone(),
        rule_impl_version: gross.rule_impl_version.clone(),
        gross_summary: summary_of(&gross)?,
        net_summary: summary_of(&net)?,
        gross_sha256: gross.series_sha256.clone(),
        net_sha256: net.series_sha256.clone(),
        gross: gross.series_columns(),
        net: net.series_columns(),
        refusals: gross.refusals.clone(),
        manifest_sha256: fx.manifest_sha256.clone(),
        candles_sha256: fx.candles_sha256.clone(),
    })
}

fn key_span(key: &SleeveKey) -> (Date, Date) {
    (key.bars[0].date, key.bars[key.bars.len() - 1].date)
}

// -------------------------------------------------------------------------------------------------------- verify

/// What a run store claims about a stored run. Every claim is optional and every given claim is CHECKED against a
/// recomputation, never used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Claims {
    pub gross_sha256: Option<String>,
    pub net_sha256: Option<String>,
    pub gross_summary: Option<RunSummary>,
    pub net_summary: Option<RunSummary>,
    /// The fixture set the run says it used.
    pub manifest_sha256: Option<String>,
    pub candles_sha256: Option<String>,
}

/// One stored run to verify.
#[derive(Clone, Debug)]
pub struct VerifyRequest<'a> {
    /// The library rule the run is supposed to be a run of (the entry's rule, chosen by the caller, not by the run).
    pub rule_id: &'a str,
    pub config: ReplicationConfig,
    pub gross: &'a SeriesColumns,
    pub net: &'a SeriesColumns,
    pub claims: Claims,
}

impl<'a> VerifyRequest<'a> {
    /// Default configuration, no claims.
    pub fn new(rule_id: &'a str, gross: &'a SeriesColumns, net: &'a SeriesColumns) -> Self {
        VerifyRequest { rule_id, config: ReplicationConfig::default(), gross, net, claims: Claims::default() }
    }
}

/// Switches of a verification. The default is everything on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyOptions {
    /// Also run Tier IV (the rule's sleeve's named mutants must each fail certification and reproduce
    /// `mutants.json`). Tier IV is a property of the certifying backtester, not of the run: a caller that binds runs
    /// to a separately verified self-test digest may switch it off to keep a verification to two simulations.
    pub tier4: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions { tier4: true }
    }
}

/// The first cell where two same-shaped series differ, by bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Difference {
    pub column: &'static str,
    pub bar: usize,
}

/// A tier (or window rule) a verified run does not satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierFailure {
    /// `gross`, `net` or `tier4`.
    pub scope: &'static str,
    /// `window`, `tier1_bands`, `tier1_trades`, `tier2_identity`, `tier3_weights`, `mutants_caught` or
    /// `matches_mutants_json`.
    pub tier: &'static str,
}

/// Why a stored run is not accepted. Each variant is one tamper class (or one honest failure to reproduce the key).
#[derive(Clone, Debug, PartialEq)]
pub enum VerifyError {
    UnknownRule {
        rule_id: String,
    },
    UnknownCostPreset {
        cost_model_id: String,
    },
    /// The platform could not re-run the rule on the fixtures at all.
    Rerun(LadderError),
    /// The stored run says it used another fixture set than the one verifying it.
    FixtureMismatch {
        what: &'static str,
        claimed: String,
        actual: String,
    },
    /// Columns are the wrong shape (a truncated or padded column, unsorted dates, no bars).
    Shape {
        basis: &'static str,
        error: ColumnsError,
    },
    /// A NaN or infinity in a series.
    NonFinite {
        basis: &'static str,
        column: &'static str,
        bar: usize,
    },
    RuleIdMismatch {
        basis: &'static str,
        expected: String,
        found: String,
    },
    /// The run was made by another implementation version (contains the `weightsim-rules` crate version and the
    /// flat-until date) than this platform's.
    ImplVersionMismatch {
        basis: &'static str,
        expected: String,
        found: String,
    },
    SymbolsMismatch {
        basis: &'static str,
        expected: Vec<String>,
        found: Vec<String>,
    },
    /// The run was made under another cost preset (or the gross and net series were swapped).
    CostModelMismatch {
        basis: &'static str,
        expected: String,
        found: String,
    },
    MetricDefinitionsMismatch {
        basis: &'static str,
        expected: String,
        found: String,
    },
    /// The digest the store claims is not the digest of the stored columns (the columns were altered after the digest
    /// was taken, or the digest was).
    ClaimedDigestMismatch {
        basis: &'static str,
        claimed: String,
        recomputed: String,
    },
    /// Fewer bars than the platform's run.
    Truncated {
        basis: &'static str,
        expected_bars: usize,
        found_bars: usize,
    },
    /// More bars than the platform's run.
    ExtraBars {
        basis: &'static str,
        expected_bars: usize,
        found_bars: usize,
    },
    /// The stored series is not the series the platform computes: its digest differs from the re-run's.
    DigestMismatch {
        basis: &'static str,
        stored: String,
        platform: String,
        first_difference: Option<Difference>,
    },
    /// A summary number the store claims is not what the stored columns give.
    ClaimedMetricsMismatch {
        basis: &'static str,
        metric: &'static str,
        claimed: f64,
        recomputed: f64,
    },
    /// The comparison with the key could not be computed (for instance too few common days).
    Analysis(LadderError),
    /// The run is genuine and consistent but does not satisfy every tier. Carries the full recomputed numbers.
    TierFailed {
        failures: Vec<TierFailure>,
        run: Box<VerifiedRun>,
    },
}

impl VerifyError {
    /// A stable machine code per class (for audit records and API errors).
    pub fn code(&self) -> &'static str {
        match self {
            VerifyError::UnknownRule { .. } => "unknown_rule",
            VerifyError::UnknownCostPreset { .. } => "unknown_cost_preset",
            VerifyError::Rerun(_) => "rerun_failed",
            VerifyError::FixtureMismatch { .. } => "fixture_mismatch",
            VerifyError::Shape { .. } => "shape",
            VerifyError::NonFinite { .. } => "non_finite",
            VerifyError::RuleIdMismatch { .. } => "rule_id_mismatch",
            VerifyError::ImplVersionMismatch { .. } => "impl_version_mismatch",
            VerifyError::SymbolsMismatch { .. } => "symbols_mismatch",
            VerifyError::CostModelMismatch { .. } => "cost_model_mismatch",
            VerifyError::MetricDefinitionsMismatch { .. } => "metric_definitions_mismatch",
            VerifyError::ClaimedDigestMismatch { .. } => "claimed_digest_mismatch",
            VerifyError::Truncated { .. } => "truncated",
            VerifyError::ExtraBars { .. } => "extra_bars",
            VerifyError::DigestMismatch { .. } => "digest_mismatch",
            VerifyError::ClaimedMetricsMismatch { .. } => "claimed_metrics_mismatch",
            VerifyError::Analysis(_) => "analysis_failed",
            VerifyError::TierFailed { .. } => "tier_failed",
        }
    }
}

impl From<ReplicateError> for VerifyError {
    fn from(e: ReplicateError) -> Self {
        match e {
            ReplicateError::UnknownRule { rule_id } => VerifyError::UnknownRule { rule_id },
            ReplicateError::UnknownCostPreset { cost_model_id } => VerifyError::UnknownCostPreset { cost_model_id },
            ReplicateError::Ladder(e) => VerifyError::Rerun(e),
        }
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        match self {
            VerifyError::UnknownRule { rule_id } => write!(f, "`{rule_id}` is not a library rule"),
            VerifyError::UnknownCostPreset { cost_model_id } => write!(f, "`{cost_model_id}` is not a cost preset"),
            VerifyError::Rerun(e) => write!(f, "{e}"),
            VerifyError::FixtureMismatch { what, claimed, actual } => {
                write!(f, "{what}: the run claims {claimed}, the fixtures are {actual}")
            }
            VerifyError::Shape { basis, error } => write!(f, "{basis} series: {error}"),
            VerifyError::NonFinite { basis, column, bar } => {
                write!(f, "{basis} series: `{column}` is not finite at bar {bar}")
            }
            VerifyError::RuleIdMismatch { basis, expected, found }
            | VerifyError::ImplVersionMismatch { basis, expected, found }
            | VerifyError::CostModelMismatch { basis, expected, found }
            | VerifyError::MetricDefinitionsMismatch { basis, expected, found } => {
                write!(f, "{basis} series: expected `{expected}`, found `{found}`")
            }
            VerifyError::SymbolsMismatch { basis, expected, found } => {
                write!(f, "{basis} series: expected symbols {expected:?}, found {found:?}")
            }
            VerifyError::ClaimedDigestMismatch { basis, claimed, recomputed } => {
                write!(f, "{basis} series: claimed digest {claimed}, the columns hash to {recomputed}")
            }
            VerifyError::Truncated { basis, expected_bars, found_bars }
            | VerifyError::ExtraBars { basis, expected_bars, found_bars } => {
                write!(f, "{basis} series has {found_bars} bars, the platform's run has {expected_bars}")
            }
            VerifyError::DigestMismatch { basis, stored, platform, first_difference } => {
                write!(f, "{basis} series digest {stored} differs from the platform's {platform}")?;
                if let Some(d) = first_difference {
                    write!(f, " (first difference: `{}` at bar {})", d.column, d.bar)?;
                }
                Ok(())
            }
            VerifyError::ClaimedMetricsMismatch { basis, metric, claimed, recomputed } => {
                write!(f, "{basis} {metric}: claimed {claimed}, recomputed from the columns {recomputed}")
            }
            VerifyError::Analysis(e) => write!(f, "{e}"),
            VerifyError::TierFailed { failures, .. } => {
                let names: Vec<String> = failures.iter().map(|t| format!("{}.{}", t.scope, t.tier)).collect();
                write!(f, "the run does not satisfy: {}", names.join(", "))
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// One basis of a verified run: everything is recomputed from the stored columns.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedBasis {
    /// Tiers I-III against the key, the flips, the window check, and the digest RECOMPUTED from the stored columns.
    pub report: BasisReport,
    /// The counted window (bar indices into the stored columns), derived, never read.
    pub window: Window,
    /// `answer_key_v1` metrics over the counted window of the stored columns.
    pub metrics: Metrics,
    pub summary: RunSummary,
}

/// Tier IV outcome for one mutant.
#[derive(Clone, Debug, PartialEq)]
pub struct Tier4Mutant {
    pub name: &'static str,
    pub caught_by: Vec<&'static str>,
    /// Disagreements with `mutants.json` (empty = reproduced).
    pub mismatches: Vec<String>,
}

/// Tier IV for one sleeve: every named mutant of the sleeve must fail certification, in the tiers `mutants.json`
/// records.
#[derive(Clone, Debug, PartialEq)]
pub struct Tier4Report {
    pub sleeve_code: &'static str,
    pub mutants: Vec<Tier4Mutant>,
}

impl Tier4Report {
    pub fn all_caught(&self) -> bool {
        !self.mutants.is_empty() && self.mutants.iter().all(|m| !m.caught_by.is_empty())
    }
    pub fn all_match_recorded(&self) -> bool {
        !self.mutants.is_empty() && self.mutants.iter().all(|m| m.mismatches.is_empty())
    }
    pub fn passed(&self) -> bool {
        self.all_caught() && self.all_match_recorded()
    }
}

/// Run the named mutants of one sleeve against the fixtures' key (Tier IV). Pure and deterministic.
pub fn tier4(fx: &Fixtures, sleeve: MutantSleeve) -> Result<Tier4Report, LadderError> {
    let key = match sleeve {
        MutantSleeve::S1 => &fx.s1,
        MutantSleeve::S3 => &fx.s3,
    };
    let mut mutants = Vec::new();
    for m in Mutant::ALL {
        if m.sleeve() != sleeve {
            continue;
        }
        let run = run_mutant(fx, m)?;
        let cmp = compare(&key_rows_for_mutants(key), &run.rows)?;
        let caught_by = cmp.failed_tiers();
        let mismatches = match fx.expected_mutants.iter().find(|e| e.name == m.name()) {
            Some(e) => mismatches_with_expected(&cmp, &caught_by, e),
            None => vec!["no entry in mutants.json".to_string()],
        };
        mutants.push(Tier4Mutant { name: m.name(), caught_by, mismatches });
    }
    Ok(Tier4Report { sleeve_code: key.code, mutants })
}

/// A run that was re-derived from its stored columns. Only [`verify`] and [`verify_with`] construct one, and only
/// inside `Err(VerifyError::TierFailed)` can it carry a failing tier.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRun {
    pub rule_id: String,
    /// The platform's implementation version, equal to the stored run's.
    pub rule_impl_version: String,
    pub sleeve_code: &'static str,
    pub cost_model_id: String,
    pub manifest_sha256: String,
    pub candles_sha256: String,
    pub gross: VerifiedBasis,
    pub net: VerifiedBasis,
    /// `None` when Tier IV was switched off.
    pub tier4: Option<Tier4Report>,
}

impl VerifiedRun {
    /// Every window rule and tier this run does not satisfy (empty for an accepted run).
    pub fn tier_failures(&self) -> Vec<TierFailure> {
        let mut out = Vec::new();
        for (scope, b) in [("gross", &self.gross), ("net", &self.net)] {
            let r = &b.report;
            if !(r.window_matches_key && r.cmp.covers_key_exactly()) {
                out.push(TierFailure { scope, tier: "window" });
            }
            if !r.cmp.bands_pass {
                out.push(TierFailure { scope, tier: CAUGHT_TIER1 });
            }
            if !r.trades_ok {
                out.push(TierFailure { scope, tier: "tier1_trades" });
            }
            if !r.tier2_pass {
                out.push(TierFailure { scope, tier: CAUGHT_TIER2 });
            }
            if !r.tier3_pass {
                out.push(TierFailure { scope, tier: CAUGHT_TIER3 });
            }
        }
        if let Some(t4) = &self.tier4 {
            if !t4.all_caught() {
                out.push(TierFailure { scope: "tier4", tier: "mutants_caught" });
            }
            if !t4.all_match_recorded() {
                out.push(TierFailure { scope: "tier4", tier: "matches_mutants_json" });
            }
        }
        out
    }
}

/// [`verify_with`] under the default configuration, no claims and every tier on.
pub fn verify(
    fx: &Fixtures,
    rule_id: &str,
    gross: &SeriesColumns,
    net: &SeriesColumns,
) -> Result<VerifiedRun, VerifyError> {
    verify_with(fx, &VerifyRequest::new(rule_id, gross, net), &VerifyOptions::default())
}

/// Re-verify a stored run against the fixtures. See the module documentation for the exact sequence of checks. `Ok`
/// only if every check passed AND the run satisfies Tiers I-III (and IV when on); the accepted run is returned with
/// every number recomputed from the stored columns.
pub fn verify_with(fx: &Fixtures, req: &VerifyRequest<'_>, opts: &VerifyOptions) -> Result<VerifiedRun, VerifyError> {
    let cost = CostModel::by_id(&req.config.cost_model_id)
        .ok_or_else(|| VerifyError::UnknownCostPreset { cost_model_id: req.config.cost_model_id.clone() })?;
    let prepared = library_rule(fx, req.rule_id)?;
    let expected_version = prepared.rule.impl_version();

    // 1. the fixture set the run says it used
    for (what, claimed, actual) in [
        ("manifest_sha256", &req.claims.manifest_sha256, &fx.manifest_sha256),
        ("candles_sha256", &req.claims.candles_sha256, &fx.candles_sha256),
    ] {
        if let Some(c) = claimed {
            if !c.eq_ignore_ascii_case(actual) {
                return Err(VerifyError::FixtureMismatch { what, claimed: c.clone(), actual: actual.clone() });
            }
        }
    }

    // 2. shape, finiteness and identity of each basis, before anything is indexed
    let bases: [(&'static str, &SeriesColumns, &'static str, Option<&String>); 2] = [
        ("gross", req.gross, GROSS_COST_MODEL_ID, req.claims.gross_sha256.as_ref()),
        ("net", req.net, cost.id, req.claims.net_sha256.as_ref()),
    ];
    for (basis, cols, cost_id, claimed_digest) in bases {
        cols.validate_shape().map_err(|error| VerifyError::Shape { basis, error })?;
        if let Some((column, bar)) = cols.first_non_finite() {
            return Err(VerifyError::NonFinite { basis, column, bar });
        }
        let wrong = |expected: &str, found: &str| (expected.to_string(), found.to_string());
        if cols.rule_id != req.rule_id {
            let (expected, found) = wrong(req.rule_id, &cols.rule_id);
            return Err(VerifyError::RuleIdMismatch { basis, expected, found });
        }
        if cols.rule_impl_version != expected_version {
            let (expected, found) = wrong(&expected_version, &cols.rule_impl_version);
            return Err(VerifyError::ImplVersionMismatch { basis, expected, found });
        }
        if cols.symbols.iter().map(String::as_str).ne(prepared.rule.universe().iter().copied()) {
            return Err(VerifyError::SymbolsMismatch {
                basis,
                expected: prepared.rule.universe().iter().map(|s| (*s).to_string()).collect(),
                found: cols.symbols.clone(),
            });
        }
        if cols.cost_model_id != cost_id {
            let (expected, found) = wrong(cost_id, &cols.cost_model_id);
            return Err(VerifyError::CostModelMismatch { basis, expected, found });
        }
        if cols.metric_definitions != METRIC_DEFINITIONS {
            let (expected, found) = wrong(METRIC_DEFINITIONS, &cols.metric_definitions);
            return Err(VerifyError::MetricDefinitionsMismatch { basis, expected, found });
        }
        // 3. the claimed digest is the digest of the stored columns
        if let Some(claimed) = claimed_digest {
            let recomputed = cols.digest().map_err(|error| VerifyError::Shape { basis, error })?;
            if !claimed.eq_ignore_ascii_case(&recomputed) {
                return Err(VerifyError::ClaimedDigestMismatch { basis, claimed: claimed.clone(), recomputed });
            }
        }
    }

    // 4. the columns are exactly what the platform computes: re-run the rule from the fixtures
    let platform = replicate_with(fx, req.rule_id, &req.config)?;
    let mut recomputed_digests: Vec<String> = Vec::with_capacity(2);
    for (basis, cols, expected_cols, platform_digest) in [
        ("gross", req.gross, &platform.gross, &platform.gross_sha256),
        ("net", req.net, &platform.net, &platform.net_sha256),
    ] {
        let expected_bars = expected_cols.n_bars();
        let found_bars = cols.n_bars();
        if found_bars < expected_bars {
            return Err(VerifyError::Truncated { basis, expected_bars, found_bars });
        }
        if found_bars > expected_bars {
            return Err(VerifyError::ExtraBars { basis, expected_bars, found_bars });
        }
        let stored = cols.digest().map_err(|error| VerifyError::Shape { basis, error })?;
        if stored != *platform_digest {
            return Err(VerifyError::DigestMismatch {
                basis,
                stored,
                platform: platform_digest.clone(),
                first_difference: first_difference(cols, expected_cols),
            });
        }
        recomputed_digests.push(stored);
    }

    // 5. window, metrics and flips from the STORED columns; claimed summaries are checked, not used
    let net_digest = recomputed_digests.pop().expect("two digests");
    let gross_digest = recomputed_digests.pop().expect("two digests");
    let gross = basis_from_columns(req.gross, GROSS_COST_MODEL_ID, gross_digest, prepared.key, Basis::Gross)
        .map_err(VerifyError::Analysis)?;
    let net =
        basis_from_columns(req.net, cost.id, net_digest, prepared.key, Basis::Net).map_err(VerifyError::Analysis)?;
    for (basis, verified, claimed) in
        [("gross", &gross, &req.claims.gross_summary), ("net", &net, &req.claims.net_summary)]
    {
        if let Some(c) = claimed {
            check_claimed_summary(basis, c, &verified.summary)?;
        }
    }

    // 6. the tiers
    let tier4 = if opts.tier4 { Some(tier4(fx, prepared.sleeve).map_err(VerifyError::Analysis)?) } else { None };
    let run = VerifiedRun {
        rule_id: req.rule_id.to_string(),
        rule_impl_version: expected_version,
        sleeve_code: prepared.key.code,
        cost_model_id: req.config.cost_model_id.clone(),
        manifest_sha256: fx.manifest_sha256.clone(),
        candles_sha256: fx.candles_sha256.clone(),
        gross,
        net,
        tier4,
    };
    let failures = run.tier_failures();
    if failures.is_empty() {
        Ok(run)
    } else {
        Err(VerifyError::TierFailed { failures, run: Box::new(run) })
    }
}

/// What ONE stored series says against the answer key: the counted window, the metrics and the flips recomputed from
/// the columns, and Tiers I-III of `basis` (`Basis::Gross` against the key's gross columns, `Basis::Net` against its net
/// ones). This is the analysis half of [`verify`] WITHOUT the re-run: it establishes what the numbers say, NOT that this
/// platform produced them (a one-ulp edit passes every tier; only the digest comparison of [`verify`] sees it). Shape and
/// finiteness are checked first and are typed errors.
pub fn analyze_columns(
    fx: &Fixtures,
    rule_id: &str,
    basis: Basis,
    cols: &SeriesColumns,
) -> Result<VerifiedBasis, VerifyError> {
    let prepared = library_rule(fx, rule_id)?;
    let name = basis.name();
    cols.validate_shape().map_err(|error| VerifyError::Shape { basis: name, error })?;
    if let Some((column, bar)) = cols.first_non_finite() {
        return Err(VerifyError::NonFinite { basis: name, column, bar });
    }
    let digest = cols.digest().map_err(|error| VerifyError::Shape { basis: name, error })?;
    let cost_id = match basis {
        Basis::Gross => GROSS_COST_MODEL_ID,
        Basis::Net => DEFAULT_COST_MODEL_ID,
    };
    basis_from_columns(cols, cost_id, digest, prepared.key, basis).map_err(VerifyError::Analysis)
}

/// Wrap stored columns in an analysis-only `SimResult` (fields the digest does not cover are left empty), derive the
/// counted window with `rows_from_sim`'s own rule, and run the certification's per-basis analysis on it.
fn basis_from_columns(
    c: &SeriesColumns,
    cost_model_id: &'static str,
    series_sha256: String,
    key: &SleeveKey,
    basis: Basis,
) -> Result<VerifiedBasis, LadderError> {
    let k = c.n_assets();
    let mut sim = SimResult {
        rule_id: c.rule_id.clone(),
        rule_impl_version: c.rule_impl_version.clone(),
        symbols: c.symbols.clone(),
        cost_model_id,
        metric_definitions: METRIC_DEFINITIONS,
        dates: c.dates.clone(),
        ret: c.ret.clone(),
        ret_pre_cost: c.ret_pre_cost.clone(),
        equity: c.equity.clone(),
        cash: c.cash.clone(),
        cost: c.cost.clone(),
        traded_notional: c.traded_notional.clone(),
        financing: c.financing.clone(),
        gross_exposure: c.gross_exposure.clone(),
        net_exposure: c.net_exposure.clone(),
        decision: c.decision.clone(),
        refused: c.refused.clone(),
        target_weights: c.target_weights.clone(),
        held_weights: c.held_weights.clone(),
        units: c.units.clone(),
        refusals: Vec::new(),
        window: None,
        signal_flips: vec![0; k],
        fills_per_asset: vec![0; k],
        rebalance_bars: 0,
        gross_exposure_stats: None,
        net_exposure_stats: None,
        series_sha256,
    };
    let (start, end) = key_span(key);
    let rows = rows_from_sim(&sim, start, end, true)?;
    let first_bar = c.dates.iter().position(|d| *d == rows.dates[0]).ok_or_else(|| {
        LadderError::Inconsistent("the counted window opens on a date that is not in the series".into())
    })?;
    let window = Window { first_bar, last_bar: first_bar + rows.dates.len() - 1 };
    sim.window = Some(window);
    let report = basis_report(&sim, &rows, key, basis)?;
    let metrics = sim
        .metrics()
        .ok_or_else(|| LadderError::Inconsistent("the metrics are undefined over the counted window".into()))?;
    let summary = summarize(&metrics, report.flips_run);
    Ok(VerifiedBasis { report, window, metrics, summary })
}

/// Every claimed number must equal the recomputed one bit for bit (both NaN counts as equal), except the CAGR, which
/// goes through `powf` and is compared to [`CLAIMED_CAGR_TOL`].
fn check_claimed_summary(basis: &'static str, claimed: &RunSummary, got: &RunSummary) -> Result<(), VerifyError> {
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    let mismatch = |metric: &'static str, claimed: f64, recomputed: f64| {
        Err(VerifyError::ClaimedMetricsMismatch { basis, metric, claimed, recomputed })
    };
    if claimed.n != got.n {
        return mismatch("n", claimed.n as f64, got.n as f64);
    }
    if claimed.first_date != got.first_date {
        return mismatch(
            "first_date",
            claimed.first_date.days_since_epoch() as f64,
            got.first_date.days_since_epoch() as f64,
        );
    }
    if claimed.last_date != got.last_date {
        return mismatch(
            "last_date",
            claimed.last_date.days_since_epoch() as f64,
            got.last_date.days_since_epoch() as f64,
        );
    }
    if claimed.flips != got.flips {
        return mismatch("flips", claimed.flips as f64, got.flips as f64);
    }
    if !same(claimed.sharpe, got.sharpe) {
        return mismatch("sharpe", claimed.sharpe, got.sharpe);
    }
    if !same(claimed.vol, got.vol) {
        return mismatch("vol", claimed.vol, got.vol);
    }
    if !same(claimed.max_drawdown, got.max_drawdown) {
        return mismatch("max_drawdown", claimed.max_drawdown, got.max_drawdown);
    }
    if !((claimed.cagr - got.cagr).abs() <= CLAIMED_CAGR_TOL) {
        return mismatch("cagr", claimed.cagr, got.cagr);
    }
    Ok(())
}

/// The first cell where two SAME-SHAPED series differ (dates, then each per-bar column, the flags, then the
/// matrices). `None` if they are identical.
fn first_difference(a: &SeriesColumns, b: &SeriesColumns) -> Option<Difference> {
    if let Some(bar) = a.dates.iter().zip(&b.dates).position(|(x, y)| x != y) {
        return Some(Difference { column: "dates", bar });
    }
    let k = a.n_assets().max(1);
    let floats: [(&'static str, &[f64], &[f64], usize); 12] = [
        ("ret", &a.ret, &b.ret, 1),
        ("ret_pre_cost", &a.ret_pre_cost, &b.ret_pre_cost, 1),
        ("equity", &a.equity, &b.equity, 1),
        ("cash", &a.cash, &b.cash, 1),
        ("cost", &a.cost, &b.cost, 1),
        ("traded_notional", &a.traded_notional, &b.traded_notional, 1),
        ("financing", &a.financing, &b.financing, 1),
        ("gross_exposure", &a.gross_exposure, &b.gross_exposure, 1),
        ("net_exposure", &a.net_exposure, &b.net_exposure, 1),
        ("target_weights", &a.target_weights, &b.target_weights, k),
        ("held_weights", &a.held_weights, &b.held_weights, k),
        ("units", &a.units, &b.units, k),
    ];
    for (column, x, y, per) in floats {
        if let Some(i) = x.iter().zip(y).position(|(p, q)| p.to_bits() != q.to_bits()) {
            return Some(Difference { column, bar: i / per });
        }
    }
    for (column, x, y) in [("decision", &a.decision, &b.decision), ("refused", &a.refused, &b.refused)] {
        if let Some(bar) = x.iter().zip(y).position(|(p, q)| p != q) {
            return Some(Difference { column, bar });
        }
    }
    None
}
