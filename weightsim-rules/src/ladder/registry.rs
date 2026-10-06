//! The library rule registry (gap-closure plan W3.1): ONE reviewed constant table that says, for every rule this crate
//! can certify, which compiled rule it is, which pinned panel it runs on, which answer-key sleeve certifies it, which
//! named mutants must fail for it (Tier IV) and which Layer C canaries it carries. `rule_facts`, `replicate`, `verify`,
//! `tier4`, `certify_sleeve` and `run_ladder` are lookups over this table; nothing else in the ladder names a rule.
//!
//! The same discipline as the Engine's `CERTIFIED_LADDER_DIGESTS`: adding a rule is a human-reviewed edit of
//! [`LIBRARY_RULES`], and the ladder's digest over the two existing entries is pinned by `tests/registry.rs` (synthetic
//! fixtures, always on) and `tests/ladder_real.rs` (real fixtures, env-gated), so a registry change that alters
//! behaviour cannot pass unnoticed.
//!
//! The registry is keyed by [`RuleSpec`] (signal x universe x weighting x cadence x overlay, the typed description of a
//! rule, plan 9.9 / W3.6): [`Registry::get_spec`] is the primary lookup, and the library id string is carried on each
//! entry as a view ([`Registry::get`], `verify::LIBRARY_RULE_IDS`). An entry also carries a factory for the compiled
//! rule, the panel it runs on, a reference to its key files inside whichever fixture set is loaded, its own mutant list
//! and its own canaries. Only the two certified rules are registered; the OHLCV grammar that would build more specs is
//! NOT built here.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use reference_rules::{
    CRYPTO_SMA_DAYS, CRYPTO_SYMBOLS, CRYPTO_WEIGHT_PER_INSTRUMENT, ETF_SMA_MONTH_ENDS, ETF_SYMBOLS,
    ETF_WEIGHT_PER_INSTRUMENT,
};
use weightsim::{Date, DecisionSchedule, Panel, RebalancePolicy, WeightRule};

use super::fixtures::{
    Fixtures, LadderError, RecordedMetrics, SleeveKey, F_S1_PERBAR, F_S3_PERBAR, F_SHADOW_S1, F_SHADOW_S3,
};
use super::mutants::{Mutant, MutantSleeve};
use super::{CANARY_DELAY_SHARPE, CANARY_PEEK_SHARPE};
use crate::adapters::{CryptoTrendRule, EtfTrendRule};

/// What a rule's signal reads. The sampling of the signal is the rule's decision schedule ([`CadenceSpec::decision`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SignalSpec {
    /// The close is strictly above the simple average of the last `window` closes sampled on the decision bars (the
    /// current one included).
    CloseAboveSma { window: usize },
}

/// Which instruments a rule trades.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniverseSpec {
    /// A fixed, ordered list of symbols.
    Fixed(&'static [&'static str]),
}

/// How the sleeve is weighted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightingSpec {
    /// Every instrument whose signal is on gets `weight` of the sleeve; the rest is cash.
    FixedPerInstrument { weight: f64 },
}

/// When the rule decides, and how the book is traded between decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CadenceSpec {
    pub decision: DecisionSchedule,
    pub rebalance: RebalancePolicy,
}

/// What is applied on top of the weighted book. None is registered yet. The ladder's flat-start entry gate
/// (`adapters::FlatUntil`) is a run-time harness applied to every rule, not an overlay of the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlaySpec {
    None,
}

/// The typed description of a library rule: the key of the registry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleSpec {
    pub signal: SignalSpec,
    pub universe: UniverseSpec,
    pub weighting: WeightingSpec,
    pub cadence: CadenceSpec,
    pub overlay: OverlaySpec,
}

/// Which pinned price panel of the fixture set a rule runs on. The loader builds one panel per universe from the
/// single candles file; a future universe adds a variant here and a panel in the loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelSpec {
    /// The five ETFs (`reference_rules::ETF_SYMBOLS`).
    Etf,
    /// BTC and ETH (`reference_rules::CRYPTO_SYMBOLS`).
    Crypto,
}

/// Where a rule's answer key lives inside a fixture set: the sleeve code (`S1`, `S3`) the ladder reports under, and
/// the per-bar key and shadow files the manifest must list for it. The fixture set's own pins (manifest and candles
/// sha256) are a property of the set, not of one rule, and stay in [`super::fixtures::Pins`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyRef {
    pub sleeve_code: &'static str,
    pub perbar_file: &'static str,
    pub shadow_file: &'static str,
}

/// A Layer C canary: a named mutant's OWN Sharpe on the real pinned data must read `centre +/- tol`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Canary {
    /// The ladder check name (`canary.<...>`).
    pub check: &'static str,
    pub mutant: Mutant,
    /// `(centre, tolerance)`.
    pub sharpe: (f64, f64),
}

/// One library rule the ladder can certify, replicate and verify.
#[derive(Clone, Copy, Debug)]
pub struct RegisteredRule {
    /// The library id (`WeightRule::id` of the compiled rule).
    pub id: &'static str,
    /// The typed description of the rule. The registry is keyed by it ([`Registry::get_spec`]).
    pub spec: RuleSpec,
    /// Builds the compiled rule, bare. The ladder applies its own `FlatUntil` entry gate at run time.
    pub factory: fn() -> Box<dyn WeightRule>,
    pub panel: PanelSpec,
    pub key: KeyRef,
    /// The named mutants of this rule's sleeve (Tier IV), in `Mutant::ALL` order.
    pub mutants: &'static [Mutant],
    pub canaries: &'static [Canary],
}

impl RegisteredRule {
    /// The implementation version the compiled rule stamps on its runs (derived from the factory, never copied).
    pub fn impl_version(&self) -> String {
        (self.factory)().impl_version()
    }

    /// The sleeve enum of this rule's key, for the mutant and Tier IV APIs that take one.
    pub fn sleeve(&self) -> Result<MutantSleeve, RegistryError> {
        MutantSleeve::from_code(self.key.sleeve_code)
            .ok_or_else(|| RegistryError::UnknownSleeve { sleeve_code: self.key.sleeve_code.to_string() })
    }

    /// Everything a fixture set holds for this rule, checked to belong to it: the key's rule id must be this id and
    /// the panel's symbols must be the rule's universe.
    pub fn fixture<'a>(&self, fx: &'a Fixtures) -> Result<SleeveFixture<'a>, LadderError> {
        let sf = fx.sleeve(self.key.sleeve_code).ok_or_else(|| {
            LadderError::Inconsistent(format!(
                "the fixture set has no sleeve `{}` for `{}`",
                self.key.sleeve_code, self.id
            ))
        })?;
        if sf.key.rule_id != self.id {
            return Err(LadderError::Inconsistent(format!(
                "sleeve `{}` is the key of `{}`, not of `{}`",
                self.key.sleeve_code, sf.key.rule_id, self.id
            )));
        }
        let universe = (self.factory)().universe().to_vec();
        if fx.panel(self.panel).symbols().iter().map(String::as_str).ne(universe.iter().copied()) {
            return Err(LadderError::Inconsistent(format!(
                "panel {:?} has symbols {:?}, `{}` needs {:?}",
                self.panel,
                fx.panel(self.panel).symbols(),
                self.id,
                universe
            )));
        }
        Ok(sf)
    }
}

/// Everything a fixture set holds for one answer-key sleeve, by reference.
#[derive(Clone, Copy, Debug)]
pub struct SleeveFixture<'a> {
    pub panel: &'a Panel,
    pub key: &'a SleeveKey,
    pub recorded: &'a [RecordedMetrics; 2],
    pub shadow: &'a [(Date, f64)],
}

fn make_etf() -> Box<dyn WeightRule> {
    Box::new(EtfTrendRule)
}

fn make_crypto() -> Box<dyn WeightRule> {
    Box::new(CryptoTrendRule)
}

/// The reviewed table. Order matters: it is the order sleeves are run and reported in, which the ladder digest covers.
pub const LIBRARY_RULES: [RegisteredRule; 2] = [
    RegisteredRule {
        id: "etf_trend_faber",
        spec: RuleSpec {
            signal: SignalSpec::CloseAboveSma { window: ETF_SMA_MONTH_ENDS },
            universe: UniverseSpec::Fixed(&ETF_SYMBOLS),
            weighting: WeightingSpec::FixedPerInstrument { weight: ETF_WEIGHT_PER_INSTRUMENT },
            cadence: CadenceSpec { decision: DecisionSchedule::LastBarOfMonth, rebalance: RebalancePolicy::OnDecision },
            overlay: OverlaySpec::None,
        },
        factory: make_etf,
        panel: PanelSpec::Etf,
        key: KeyRef { sleeve_code: "S1", perbar_file: F_S1_PERBAR, shadow_file: F_SHADOW_S1 },
        mutants: &[Mutant::S1SmaExcludesCurrent, Mutant::S1OneBarLate, Mutant::S1WrongRebalanceMode],
        canaries: &[],
    },
    RegisteredRule {
        id: "crypto_trend_100d",
        spec: RuleSpec {
            signal: SignalSpec::CloseAboveSma { window: CRYPTO_SMA_DAYS },
            universe: UniverseSpec::Fixed(&CRYPTO_SYMBOLS),
            weighting: WeightingSpec::FixedPerInstrument { weight: CRYPTO_WEIGHT_PER_INSTRUMENT },
            cadence: CadenceSpec { decision: DecisionSchedule::Daily, rebalance: RebalancePolicy::EveryBar },
            overlay: OverlaySpec::None,
        },
        factory: make_crypto,
        panel: PanelSpec::Crypto,
        key: KeyRef { sleeve_code: "S3", perbar_file: F_S3_PERBAR, shadow_file: F_SHADOW_S3 },
        mutants: &[
            Mutant::S3SameDayPeek,
            Mutant::S3ExtraDelay,
            Mutant::S3SmaExcludesToday,
            Mutant::S3HalfSizing,
            Mutant::S3DriftingSubaccounts,
        ],
        canaries: &[
            Canary {
                check: "canary.s3_same_day_peek_sharpe",
                mutant: Mutant::S3SameDayPeek,
                sharpe: CANARY_PEEK_SHARPE,
            },
            Canary { check: "canary.s3_extra_delay_sharpe", mutant: Mutant::S3ExtraDelay, sharpe: CANARY_DELAY_SHARPE },
        ],
    },
];

/// Why a registry lookup or construction failed. (`PartialEq` only: a [`RuleSpec`] holds a weight, an `f64`.)
#[derive(Clone, Debug, PartialEq)]
pub enum RegistryError {
    UnknownRule {
        rule_id: String,
    },
    DuplicateRule {
        rule_id: String,
    },
    /// Two entries claim the same answer-key sleeve.
    DuplicateSleeve {
        sleeve_code: String,
    },
    /// Two entries carry the same [`RuleSpec`].
    DuplicateSpec {
        spec: RuleSpec,
    },
    /// No entry carries this [`RuleSpec`].
    UnknownSpec {
        spec: RuleSpec,
    },
    /// No entry is certified against this sleeve.
    UnknownSleeve {
        sleeve_code: String,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::UnknownRule { rule_id } => write!(f, "`{rule_id}` is not a registered library rule"),
            RegistryError::DuplicateRule { rule_id } => write!(f, "`{rule_id}` is registered twice"),
            RegistryError::DuplicateSleeve { sleeve_code } => write!(f, "sleeve `{sleeve_code}` is claimed twice"),
            RegistryError::DuplicateSpec { spec } => write!(f, "rule spec {spec:?} is registered twice"),
            RegistryError::UnknownSpec { spec } => write!(f, "no registered rule has spec {spec:?}"),
            RegistryError::UnknownSleeve { sleeve_code } => {
                write!(f, "no registered rule is certified on `{sleeve_code}`")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// A validated set of registered rules (ids, sleeve codes and specs unique), in table order.
#[derive(Clone, Debug)]
pub struct Registry {
    rules: Vec<RegisteredRule>,
}

impl Registry {
    /// Build from a table, refusing a duplicate id, a duplicate sleeve or a duplicate spec.
    pub fn from_rules(rules: Vec<RegisteredRule>) -> Result<Registry, RegistryError> {
        let mut ids: BTreeMap<&str, ()> = BTreeMap::new();
        let mut sleeves: BTreeMap<&str, ()> = BTreeMap::new();
        let mut specs: Vec<RuleSpec> = Vec::with_capacity(rules.len());
        for r in &rules {
            if ids.insert(r.id, ()).is_some() {
                return Err(RegistryError::DuplicateRule { rule_id: r.id.to_string() });
            }
            if sleeves.insert(r.key.sleeve_code, ()).is_some() {
                return Err(RegistryError::DuplicateSleeve { sleeve_code: r.key.sleeve_code.to_string() });
            }
            if specs.contains(&r.spec) {
                return Err(RegistryError::DuplicateSpec { spec: r.spec });
            }
            specs.push(r.spec);
        }
        Ok(Registry { rules })
    }

    /// The library registry, built once from [`LIBRARY_RULES`].
    pub fn library() -> &'static Registry {
        static LIBRARY: OnceLock<Registry> = OnceLock::new();
        LIBRARY.get_or_init(|| Registry::from_rules(LIBRARY_RULES.to_vec()).expect("LIBRARY_RULES is a reviewed table"))
    }

    /// The entry with this library id (the view of the registry by id string).
    pub fn get(&self, rule_id: &str) -> Result<&RegisteredRule, RegistryError> {
        self.rules
            .iter()
            .find(|r| r.id == rule_id)
            .ok_or_else(|| RegistryError::UnknownRule { rule_id: rule_id.to_string() })
    }

    /// The entry carrying this [`RuleSpec`] (the registry's own key).
    pub fn get_spec(&self, spec: &RuleSpec) -> Result<&RegisteredRule, RegistryError> {
        self.rules.iter().find(|r| r.spec == *spec).ok_or(RegistryError::UnknownSpec { spec: *spec })
    }

    /// The entry certified against a sleeve.
    pub fn by_sleeve(&self, sleeve: MutantSleeve) -> Result<&RegisteredRule, RegistryError> {
        self.rules
            .iter()
            .find(|r| r.key.sleeve_code == sleeve.code())
            .ok_or_else(|| RegistryError::UnknownSleeve { sleeve_code: sleeve.code().to_string() })
    }

    pub fn iter(&self) -> impl Iterator<Item = &RegisteredRule> {
        self.rules.iter()
    }

    pub fn ids(&self) -> Vec<&'static str> {
        self.rules.iter().map(|r| r.id).collect()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}
