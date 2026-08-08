//! Dynamic chromosome for strategy-agnostic parameter optimization.
//!
//! `DynamicChromosome` is the primary chromosome type used by the distributed GA
//! pipeline. Instead of hard-coded Rust fields (like the legacy `MarketMakingChromosome`),
//! all parameter definitions are stored in a shared `ParameterSchema`, and each
//! chromosome instance holds a `Vec<Gene>` matching the schema order.
//!
//! Built-in strategies get their schema from [`presets::schema_for_strategy()`],
//! while custom/Python strategies declare theirs via `parameter_space()` dicts.
//!
//! # Key conversions
//!
//! - [`DynamicChromosome::to_param_map()`] — genes → `HashMap<String, serde_json::Value>`
//! - [`DynamicChromosome::from_param_map()`] — `HashMap` → genes (fills missing with defaults)
//! - [`DynamicChromosome::to_f64_map()`] — genes → `HashMap<String, f64>`
//!
//! Implements the `Chromosome` trait: random, crossover, mutate, distance.
//! Derives `Serialize`/`Deserialize` for network transport (distributed GA).

use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::Chromosome;

// ---------------------------------------------------------------------------
// Gene & parameter types
// ---------------------------------------------------------------------------

/// A single gene value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gene {
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl Gene {
    /// Normalize to \[0, 1\] given the parameter definition.
    pub fn normalized(&self, def: &ParameterDef) -> f64 {
        match (self, &def.param_type) {
            (Gene::Float(v), ParamType::Float { min, max }) => {
                if (max - min).abs() < f64::EPSILON {
                    0.5
                } else {
                    (v - min) / (max - min)
                }
            }
            (Gene::Int(v), ParamType::Int { min, max }) => {
                if max == min {
                    0.5
                } else {
                    (*v - min) as f64 / (max - min) as f64
                }
            }
            (Gene::Bool(v), ParamType::Bool) => if *v { 1.0 } else { 0.0 },
            _ => 0.5,
        }
    }

    /// Convert to f64 for simple numeric use.
    pub fn as_f64(&self) -> f64 {
        match self {
            Gene::Float(v) => *v,
            Gene::Int(v) => *v as f64,
            Gene::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }
}

/// Parameter type with bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamType {
    Float { min: f64, max: f64 },
    Int { min: i64, max: i64 },
    Bool,
}

/// Definition of a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDef {
    /// Parameter name (must match Python strategy attribute).
    pub name: String,
    /// Type and bounds.
    pub param_type: ParamType,
    /// Default value.
    pub default: Gene,
}

impl ParameterDef {
    /// Generate a random gene within bounds.
    pub fn random_gene<R: Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        match &self.param_type {
            ParamType::Float { min, max } => Gene::Float(rng.gen_range(*min..=*max)),
            ParamType::Int { min, max } => Gene::Int(rng.gen_range(*min..=*max)),
            ParamType::Bool => Gene::Bool(rng.gen_bool(0.5)),
        }
    }

    /// Clamp a gene to valid bounds.
    pub fn enforce_bounds(&self, gene: &mut Gene) {
        match (&mut *gene, &self.param_type) {
            (Gene::Float(v), ParamType::Float { min, max }) => {
                *v = v.clamp(*min, *max);
            }
            (Gene::Int(v), ParamType::Int { min, max }) => {
                *v = (*v).clamp(*min, *max);
            }
            _ => {} // Bool has no bounds
        }
    }
}

/// Schema describing all parameters for a Python strategy.
/// Shared across the entire population via `Arc`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub params: Vec<ParameterDef>,
}

impl ParameterSchema {
    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Create a default chromosome from this schema.
    pub fn default_chromosome(self: &Arc<Self>) -> DynamicChromosome {
        let genes = self.params.iter().map(|d| d.default.clone()).collect();
        DynamicChromosome {
            genes,
            schema: self.clone(),
        }
    }

    /// Convert schema defaults to a `HashMap<String, serde_json::Value>`.
    ///
    /// Useful for merging optimizable parameter defaults with non-optimized
    /// strategy constants.
    pub fn to_default_map(&self) -> HashMap<String, serde_json::Value> {
        self.params
            .iter()
            .map(|def| {
                let val = match &def.default {
                    Gene::Float(v) => serde_json::Value::from(*v),
                    Gene::Int(v) => serde_json::Value::from(*v),
                    Gene::Bool(v) => serde_json::Value::from(*v),
                };
                (def.name.clone(), val)
            })
            .collect()
    }

    /// Find the indices of the two parameters with the widest normalized ranges
    /// (useful for landscape heatmaps). Booleans are deprioritized.
    #[deprecated(note = "Use top_pairs_by_sensitivity() for more meaningful pair selection")]
    pub fn top_two_continuous_indices(&self) -> (usize, usize) {
        let mut scored: Vec<(usize, f64)> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let score = match &d.param_type {
                    ParamType::Float { min, max } => (max - min).abs(),
                    ParamType::Int { min, max } => (max - min) as f64,
                    ParamType::Bool => 0.0, // deprioritize
                };
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let a = scored.first().map(|(i, _)| *i).unwrap_or(0);
        let b = scored.get(1).map(|(i, _)| *i).unwrap_or(if a == 0 { 1.min(self.params.len().saturating_sub(1)) } else { 0 });
        (a, b)
    }

    /// Select up to `max_pairs` parameter index pairs ranked by combined
    /// sensitivity score. Booleans are excluded. Falls back to range-width
    /// ranking if `sensitivity` is empty.
    pub fn top_pairs_by_sensitivity(
        &self,
        sensitivity: &[crate::landscape::SensitivityEntry],
        max_pairs: usize,
    ) -> Vec<(usize, usize)> {
        // Build index→score map from sensitivity entries
        let score_map: HashMap<&str, f64> = sensitivity
            .iter()
            .map(|s| (s.param_name.as_str(), s.sensitivity_score))
            .collect();

        // Collect non-bool param indices with their sensitivity scores
        let mut candidates: Vec<(usize, f64)> = self
            .params
            .iter()
            .enumerate()
            .filter(|(_, d)| !matches!(d.param_type, ParamType::Bool))
            .map(|(i, d)| {
                let score = score_map.get(d.name.as_str()).copied().unwrap_or(0.0);
                (i, score)
            })
            .collect();

        // Sort by sensitivity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K candidates (at most 6) to limit combinatorial pairs
        let k = candidates.len().min(6);
        let top = &candidates[..k];

        // Generate all pairs, rank by combined sensitivity
        let mut pairs: Vec<((usize, usize), f64)> = Vec::new();
        for i in 0..top.len() {
            for j in (i + 1)..top.len() {
                pairs.push(((top[i].0, top[j].0), top[i].1 + top[j].1));
            }
        }
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        pairs.into_iter().take(max_pairs).map(|(pair, _)| pair).collect()
    }

    /// Build a `ParameterSchema` from a raw dict typically extracted from Python.
    ///
    /// Expected format per entry:
    /// ```json
    /// { "type": "float"|"int"|"bool", "min": N, "max": N, "default": N }
    /// ```
    pub fn from_raw_dict(dict: &HashMap<String, HashMap<String, serde_json::Value>>) -> Result<Arc<Self>, String> {
        let mut params = Vec::with_capacity(dict.len());

        // Stable ordering: sort by parameter name
        let mut keys: Vec<&String> = dict.keys().collect();
        keys.sort();

        for key in keys {
            let entry = &dict[key];
            let type_str = entry
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("parameter '{}' missing 'type'", key))?;

            let (param_type, default) = match type_str {
                "float" => {
                    let min = entry.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let max = entry.get("max").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let def = entry.get("default").and_then(|v| v.as_f64()).unwrap_or((min + max) / 2.0);
                    (ParamType::Float { min, max }, Gene::Float(def))
                }
                "int" | "integer" => {
                    // Use as_i64() first; fall back to rounding as_f64() in case the
                    // JSON was produced from a Python int serialized as a float (1000.0).
                    let json_as_i64 = |v: &serde_json::Value| -> Option<i64> {
                        v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64))
                    };
                    let min = entry.get("min").and_then(json_as_i64).unwrap_or(0);
                    let max = entry.get("max").and_then(json_as_i64).unwrap_or(100);
                    let def = entry.get("default").and_then(json_as_i64).unwrap_or((min + max) / 2);
                    (ParamType::Int { min, max }, Gene::Int(def))
                }
                "bool" | "boolean" => {
                    let def = entry.get("default").and_then(|v| v.as_bool()).unwrap_or(false);
                    (ParamType::Bool, Gene::Bool(def))
                }
                other => return Err(format!("parameter '{}' has unsupported type '{}'", key, other)),
            };

            params.push(ParameterDef {
                name: key.clone(),
                param_type,
                default,
            });
        }

        if params.is_empty() {
            return Err("parameter_space() returned empty dict".into());
        }

        Ok(Arc::new(ParameterSchema { params }))
    }
}

// ---------------------------------------------------------------------------
// DynamicChromosome
// ---------------------------------------------------------------------------

/// A chromosome whose genes are dynamically defined by a Python strategy's
/// `parameter_space()` method rather than hardcoded Rust fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicChromosome {
    /// Gene values, in the same order as `schema.params`.
    pub genes: Vec<Gene>,
    /// Shared parameter definitions (Arc for cheap clone across population).
    pub schema: Arc<ParameterSchema>,
}

impl DynamicChromosome {
    /// Convert genes to a `HashMap<String, serde_json::Value>` suitable for
    /// passing to Python's `update_parameters()`.
    pub fn to_param_map(&self) -> HashMap<String, serde_json::Value> {
        self.schema
            .params
            .iter()
            .zip(self.genes.iter())
            .map(|(def, gene)| {
                let val = match gene {
                    Gene::Float(v) => serde_json::Value::from(*v),
                    Gene::Int(v) => serde_json::Value::from(*v),
                    Gene::Bool(v) => serde_json::Value::from(*v),
                };
                (def.name.clone(), val)
            })
            .collect()
    }

    /// Convert genes to a flat `HashMap<String, f64>` for numeric use.
    pub fn to_f64_map(&self) -> HashMap<String, f64> {
        self.schema
            .params
            .iter()
            .zip(self.genes.iter())
            .map(|(def, gene)| (def.name.clone(), gene.as_f64()))
            .collect()
    }

    /// Create from a `HashMap<String, serde_json::Value>` (e.g. from JSON),
    /// filling missing keys with defaults.
    pub fn from_param_map(
        map: &HashMap<String, serde_json::Value>,
        schema: &Arc<ParameterSchema>,
    ) -> Self {
        let genes = schema
            .params
            .iter()
            .map(|def| {
                if let Some(val) = map.get(&def.name) {
                    match &def.param_type {
                        ParamType::Float { .. } => {
                            Gene::Float(val.as_f64().unwrap_or(def.default.as_f64()))
                        }
                        ParamType::Int { .. } => {
                            Gene::Int(val.as_i64().unwrap_or_else(|| match &def.default {
                                Gene::Int(v) => *v,
                                _ => 0,
                            }))
                        }
                        ParamType::Bool => Gene::Bool(val.as_bool().unwrap_or_else(|| {
                            matches!(&def.default, Gene::Bool(true))
                        })),
                    }
                } else {
                    def.default.clone()
                }
            })
            .collect();

        let mut chr = DynamicChromosome {
            genes,
            schema: schema.clone(),
        };
        chr.enforce_bounds();
        chr
    }

    /// Enforce all bounds.
    pub fn enforce_bounds(&mut self) {
        for (gene, def) in self.genes.iter_mut().zip(self.schema.params.iter()) {
            def.enforce_bounds(gene);
        }
    }
}

// ---------------------------------------------------------------------------
// Chromosome trait implementation
// ---------------------------------------------------------------------------

// We need a schema to create random chromosomes. Since the trait method
// `random()` has no `self`, we use a thread-local to pass the schema.
// This is set before the GA optimizer creates its initial population.
thread_local! {
    static DYNAMIC_SCHEMA: std::cell::RefCell<Option<Arc<ParameterSchema>>> = const { std::cell::RefCell::new(None) };
}

/// Set the schema for the current thread. Must be called before
/// `GeneticOptimizer::run()` on each thread that will create chromosomes.
pub fn set_dynamic_schema(schema: Arc<ParameterSchema>) {
    DYNAMIC_SCHEMA.with(|s| {
        *s.borrow_mut() = Some(schema);
    });
}

/// Get the current thread's schema. Panics if not set.
fn get_dynamic_schema() -> Arc<ParameterSchema> {
    DYNAMIC_SCHEMA.with(|s| {
        s.borrow()
            .as_ref()
            .expect("DynamicChromosome schema not set — call set_dynamic_schema() before GA run")
            .clone()
    })
}

impl Chromosome for DynamicChromosome {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let schema = get_dynamic_schema();
        let genes = schema.params.iter().map(|def| def.random_gene(rng)).collect();
        DynamicChromosome { genes, schema }
    }

    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let genes = self
            .genes
            .iter()
            .zip(other.genes.iter())
            .map(|(a, b)| {
                if rng.gen_bool(0.5) {
                    a.clone()
                } else {
                    b.clone()
                }
            })
            .collect();

        let mut child = DynamicChromosome {
            genes,
            schema: self.schema.clone(),
        };
        child.enforce_bounds();
        child
    }

    fn mutate(&mut self, rng: &mut impl Rng, mutation_rate: f64) {
        for (gene, def) in self.genes.iter_mut().zip(self.schema.params.iter()) {
            if rng.gen::<f64>() < mutation_rate {
                match (&mut *gene, &def.param_type) {
                    (Gene::Float(v), ParamType::Float { min, max }) => {
                        let range = max - min;
                        if let Ok(normal) = Normal::new(0.0, range * 0.1) {
                            *v += normal.sample(rng);
                            *v = v.clamp(*min, *max);
                        }
                    }
                    (Gene::Int(v), ParamType::Int { min, max }) => {
                        let range = (max - min) as f64;
                        if let Ok(normal) = Normal::new(0.0, range * 0.1) {
                            let delta = normal.sample(rng).round() as i64;
                            *v = (*v + delta).clamp(*min, *max);
                        }
                    }
                    (Gene::Bool(v), ParamType::Bool) => {
                        *v = !*v; // bit flip
                    }
                    _ => {}
                }
            }
        }
    }

    fn distance(&self, other: &Self) -> f64 {
        // Normalized Euclidean distance: each gene divided by its range
        let sum_sq: f64 = self
            .genes
            .iter()
            .zip(other.genes.iter())
            .zip(self.schema.params.iter())
            .map(|((a, b), def)| {
                let diff = a.normalized(def) - b.normalized(def);
                diff * diff
            })
            .sum();

        (sum_sq / self.genes.len().max(1) as f64).sqrt()
    }

    fn behavioral_signature(&self) -> crate::diversity::BehavioralSignature {
        use crate::diversity::{BehavioralSignature, SignalType};

        // `DynamicChromosome` is schema-driven (parameter names vary per Python
        // strategy / built-in preset), so unlike `MarketMakingChromosome` there
        // are no fixed struct fields to read directly. Instead, look up a set
        // of conventional parameter names across known presets and fall back to
        // neutral defaults when a given schema doesn't define that concept.
        let map = self.to_f64_map();
        let get = |names: &[&str]| -> Option<f64> {
            names.iter().find_map(|n| map.get(*n).copied())
        };
        let names: Vec<&str> = self.schema.params.iter().map(|d| d.name.as_str()).collect();
        let has_any = |needles: &[&str]| names.iter().any(|n| needles.iter().any(|needle| n.contains(needle)));

        let hold_time_bucket = get(&[
            "time_horizon_minutes",
            "holding_period_minutes",
            "hold_time_minutes",
            "slow_period",
            "slow_window",
        ])
        .map(|v| {
            if v < 2.0 {
                0
            } else if v < 10.0 {
                1
            } else if v < 60.0 {
                2
            } else {
                3
            }
        })
        .unwrap_or(1); // unknown: assume moderate (matches BehavioralSignature::default's implicit "0" only via explicit fallback)

        let frequency_bucket = get(&["max_trades_per_minute", "trade_frequency", "fast_period", "fast_window"])
            .map(|v| {
                if v < 1.0 {
                    0
                } else if v < 10.0 {
                    1
                } else if v < 60.0 {
                    2
                } else {
                    3
                }
            })
            .unwrap_or(1);

        let signal_type = if has_any(&["spread", "inventory", "quote", "market_making"]) {
            SignalType::MarketMaking
        } else if has_any(&["breakout", "channel", "donchian"]) {
            SignalType::Breakout
        } else if has_any(&["arbitrage", "arb_", "basis"]) {
            SignalType::Arbitrage
        } else {
            SignalType::Custom
        };

        let sizing_style: u8 = if has_any(&["kelly"]) {
            1
        } else if has_any(&["vol_target", "volatility_scale", "atr_multiple"]) {
            2
        } else {
            0
        };

        // No per-hour gene exists in typical schemas; assume always-on.
        let active_hours: u32 = 0x00FF_FFFF;

        // Asset isn't a chromosome field — combine externally if asset-aware
        // crowding is needed.
        let asset_hash: u64 = 0;

        let risk_profile = get(&["risk_aversion"])
            .map(|v| if v > 0.6 { 0 } else if v > 0.2 { 1 } else { 2 })
            .unwrap_or(1);

        let reward_risk_bucket = match (get(&["take_profit_pct", "tp_pct"]), get(&["stop_loss_pct", "sl_pct"])) {
            (Some(tp), Some(sl)) if sl > 0.0 => {
                let ratio = tp / sl;
                if ratio < 1.0 {
                    0
                } else if ratio < 2.0 {
                    1
                } else if ratio < 3.0 {
                    2
                } else {
                    3
                }
            }
            _ => get(&["fee_spread_multiplier"])
                .map(|v| if v < 3.25 { 0 } else if v < 4.0 { 1 } else if v < 5.0 { 2 } else { 3 })
                .unwrap_or(1),
        };

        BehavioralSignature {
            hold_time_bucket,
            frequency_bucket,
            signal_type,
            sizing_style,
            active_hours,
            asset_hash,
            risk_profile,
            reward_risk_bucket,
        }
    }

    fn parameter_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Sort by parameter name so hash is stable regardless of schema
        // ordering, then quantize floats to tolerate GA mutation-level noise.
        let mut entries: Vec<(String, f64)> = self.to_f64_map().into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut hasher = DefaultHasher::new();
        for (name, value) in entries {
            name.hash(&mut hasher);
            ((value * 1_000.0).round() as i64).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Complexity = number of tunable parameters in the schema. Used by the
    /// optional GA parsimony penalty (`GeneticConfig::complexity_penalty_weight`)
    /// to bias selection toward strategies with fewer degrees of freedom when
    /// raw fitness is comparable.
    fn complexity(&self) -> f64 {
        self.schema.params.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> Arc<ParameterSchema> {
        Arc::new(ParameterSchema {
            params: vec![
                ParameterDef {
                    name: "fast_window".into(),
                    param_type: ParamType::Int { min: 5, max: 100 },
                    default: Gene::Int(10),
                },
                ParameterDef {
                    name: "slow_window".into(),
                    param_type: ParamType::Int { min: 20, max: 200 },
                    default: Gene::Int(50),
                },
                ParameterDef {
                    name: "threshold".into(),
                    param_type: ParamType::Float { min: 0.0001, max: 0.01 },
                    default: Gene::Float(0.002),
                },
                ParameterDef {
                    name: "use_volume".into(),
                    param_type: ParamType::Bool,
                    default: Gene::Bool(false),
                },
            ],
        })
    }

    #[test]
    fn test_random_within_bounds() {
        let schema = sample_schema();
        set_dynamic_schema(schema.clone());

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let chr = DynamicChromosome::random(&mut rng);
            assert_eq!(chr.genes.len(), 4);
            match &chr.genes[0] {
                Gene::Int(v) => assert!(*v >= 5 && *v <= 100),
                _ => panic!("expected Int"),
            }
            match &chr.genes[2] {
                Gene::Float(v) => assert!(*v >= 0.0001 && *v <= 0.01),
                _ => panic!("expected Float"),
            }
        }
    }

    #[test]
    fn test_crossover_preserves_schema() {
        let schema = sample_schema();
        set_dynamic_schema(schema.clone());

        let mut rng = rand::thread_rng();
        let a = DynamicChromosome::random(&mut rng);
        let b = DynamicChromosome::random(&mut rng);
        let child = a.crossover(&b, &mut rng);
        assert_eq!(child.genes.len(), 4);
    }

    #[test]
    fn test_distance_same_is_zero() {
        let schema = sample_schema();
        set_dynamic_schema(schema.clone());

        let mut rng = rand::thread_rng();
        let a = DynamicChromosome::random(&mut rng);
        let d = a.distance(&a);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_to_param_map_roundtrip() {
        let schema = sample_schema();
        let chr = schema.default_chromosome();

        let map = chr.to_param_map();
        assert_eq!(map["fast_window"], serde_json::Value::from(10));
        assert_eq!(map["threshold"], serde_json::Value::from(0.002));
        assert_eq!(map["use_volume"], serde_json::Value::from(false));

        let chr2 = DynamicChromosome::from_param_map(&map, &schema);
        assert!(chr.distance(&chr2) < 1e-10);
    }

    #[test]
    fn test_from_raw_dict() {
        let mut dict: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();

        let mut fast = HashMap::new();
        fast.insert("type".into(), serde_json::Value::from("int"));
        fast.insert("min".into(), serde_json::Value::from(5));
        fast.insert("max".into(), serde_json::Value::from(100));
        fast.insert("default".into(), serde_json::Value::from(10));
        dict.insert("fast_window".into(), fast);

        let mut thresh = HashMap::new();
        thresh.insert("type".into(), serde_json::Value::from("float"));
        thresh.insert("min".into(), serde_json::Value::from(0.0001));
        thresh.insert("max".into(), serde_json::Value::from(0.01));
        thresh.insert("default".into(), serde_json::Value::from(0.002));
        dict.insert("threshold".into(), thresh);

        let schema = ParameterSchema::from_raw_dict(&dict).unwrap();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema.params[0].name, "fast_window"); // alphabetical
        assert_eq!(schema.params[1].name, "threshold");
    }
}
