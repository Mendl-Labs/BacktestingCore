//! Typed state management for Python strategies with `state_schema()`.
//!
//! The [`StateStore`] holds all declared state variables in native Rust containers,
//! eliminating Python dict conversion on the hot path. State is constructed once
//! from the Python `state_schema()` dict at job start, then updated tick-by-tick
//! entirely in Rust.
//!
//! ## Supported state types
//!
//! | Python type        | Rust container                       |
//! |--------------------|--------------------------------------|
//! | `bool`             | `bool`                               |
//! | `int`              | `i64`                                |
//! | `float`            | `f64`                                |
//! | `enum`             | `(u8, Vec<String>)` — index + names  |
//! | `set[int]`         | `BoundedHashSet<i64>`                |
//! | `window[float]`    | `BoundedDeque<f64>`                  |

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BoundedDeque — fixed-capacity sliding window with O(1) mean/std
// ---------------------------------------------------------------------------

/// A fixed-capacity `VecDeque<f64>` that auto-evicts the oldest element on push
/// when at capacity. Used for `window[float]` state fields.
///
/// Maintains running `sum` and `sum_sq` accumulators so that `mean()` and
/// `std()` are O(1) instead of O(n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedDeque<T = f64> {
    inner: VecDeque<T>,
    capacity: usize,
    /// Running sum of all elements (for O(1) mean).
    #[serde(default)]
    running_sum: f64,
    /// Running sum of squares (for O(1) std via Var = E[X²] - E[X]²).
    #[serde(default)]
    running_sum_sq: f64,
}

impl BoundedDeque<f64> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(capacity),
            capacity,
            running_sum: 0.0,
            running_sum_sq: 0.0,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.inner.len() == self.capacity {
            if let Some(evicted) = self.inner.pop_front() {
                self.running_sum -= evicted;
                self.running_sum_sq -= evicted * evicted;
            }
        }
        self.running_sum += value;
        self.running_sum_sq += value * value;
        self.inner.push_back(value);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn back(&self) -> Option<&f64> {
        self.inner.back()
    }

    pub fn front(&self) -> Option<&f64> {
        self.inner.front()
    }

    /// Get the Nth element from the back (0 = last, 1 = second-to-last, …).
    pub fn nth_from_back(&self, n: usize) -> Option<&f64> {
        if n >= self.inner.len() {
            return None;
        }
        self.inner.get(self.inner.len() - 1 - n)
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.inner.iter()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
    }

    /// Return last `n` elements as a slice-like iterator.
    pub fn tail(&self, n: usize) -> impl Iterator<Item = &f64> {
        let skip = self.inner.len().saturating_sub(n);
        self.inner.iter().skip(skip)
    }

    /// Mean of the last `n` elements (or all if fewer than `n`).
    pub fn tail_mean(&self, n: usize) -> f64 {
        let count = self.inner.len().min(n);
        if count == 0 {
            return 0.0;
        }
        // When n >= len, use the O(1) running sum
        if count == self.inner.len() {
            return self.running_sum / count as f64;
        }
        let sum: f64 = self.tail(n).sum();
        sum / count as f64
    }

    /// Population standard deviation of the last `n` elements.
    pub fn tail_std(&self, n: usize) -> f64 {
        let count = self.inner.len().min(n);
        if count < 2 {
            return 0.0;
        }
        // When n >= len, use O(1) accumulators
        if count == self.inner.len() {
            return self.std();
        }
        let mean = self.tail_mean(n);
        let variance = self.tail(n).map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        variance.sqrt()
    }

    /// Max of the last `n` elements.
    pub fn tail_max(&self, n: usize) -> f64 {
        self.tail(n).copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Min of the last `n` elements.
    pub fn tail_min(&self, n: usize) -> f64 {
        self.tail(n).copied().fold(f64::INFINITY, f64::min)
    }

    /// Sum of the last `n` elements.
    pub fn tail_sum(&self, n: usize) -> f64 {
        self.tail(n).sum()
    }

    /// Mean of all elements — O(1) via running accumulator.
    pub fn mean(&self) -> f64 {
        if self.inner.is_empty() {
            return 0.0;
        }
        self.running_sum / self.inner.len() as f64
    }

    /// Population standard deviation of all elements — O(1) via running accumulators.
    /// Uses Var(X) = E[X²] − (E[X])².
    pub fn std(&self) -> f64 {
        if self.inner.len() < 2 {
            return 0.0;
        }
        let n = self.inner.len() as f64;
        let mean = self.running_sum / n;
        let variance = (self.running_sum_sq / n) - (mean * mean);
        // Guard against floating-point rounding producing tiny negatives
        if variance <= 0.0 { 0.0 } else { variance.sqrt() }
    }
}

// ---------------------------------------------------------------------------
// BoundedHashSet — capacity-limited set with FIFO eviction
// ---------------------------------------------------------------------------

/// A capacity-limited `HashSet` with FIFO eviction. When capacity is reached,
/// the oldest inserted element is removed before inserting the new one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedHashSet<T: std::hash::Hash + Eq + Clone> {
    set: HashSet<T>,
    insertion_order: VecDeque<T>,
    capacity: usize,
}

impl<T: std::hash::Hash + Eq + Clone> BoundedHashSet<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            set: HashSet::with_capacity(capacity),
            insertion_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.set.contains(&value) {
            return false; // already present
        }
        if self.set.len() == self.capacity {
            // Evict oldest
            if let Some(oldest) = self.insertion_order.pop_front() {
                self.set.remove(&oldest);
            }
        }
        self.insertion_order.push_back(value.clone());
        self.set.insert(value)
    }

    pub fn contains(&self, value: &T) -> bool {
        self.set.contains(value)
    }

    pub fn remove(&mut self, value: &T) -> bool {
        if self.set.remove(value) {
            self.insertion_order.retain(|v| v != value);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    pub fn clear(&mut self) {
        self.set.clear();
        self.insertion_order.clear();
    }
}

// ---------------------------------------------------------------------------
// StateType — supported field types in state_schema()
// ---------------------------------------------------------------------------

/// The types a state field can have.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateType {
    Bool,
    Int,
    Float,
    /// Enum with named variants (stored as variant index `u8`).
    Enum { values: Vec<String> },
    /// Capacity-bounded set of integers.
    IntSet { capacity: usize },
    /// Capacity-bounded sliding window of floats.
    FloatWindow { capacity: usize },
}

// ---------------------------------------------------------------------------
// FieldDef — a single field definition in the schema
// ---------------------------------------------------------------------------

/// One field in a `state_schema()` declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub state_type: StateType,
    pub default_json: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// StateSchema — the full schema parsed from Python
// ---------------------------------------------------------------------------

/// The full state schema parsed from a Python strategy's `state_schema()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSchema {
    pub fields: Vec<FieldDef>,
}

impl StateSchema {
    /// Parse a raw dict from Python `state_schema()` into a validated schema.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "my_bool":    {"type": "bool",   "default": false},
    ///   "my_int":     {"type": "int",    "default": 0},
    ///   "my_float":   {"type": "float",  "default": 0.0},
    ///   "my_enum":    {"type": "enum",   "values": ["flat","long","short"], "default": "flat"},
    ///   "my_set":     {"type": "set[int]",     "capacity": 100},
    ///   "my_window":  {"type": "window[float]","capacity": 50, "default": null}
    /// }
    /// ```
    pub fn from_raw_dict(
        raw: &HashMap<String, HashMap<String, serde_json::Value>>,
    ) -> Result<Self, String> {
        let mut fields = Vec::with_capacity(raw.len());

        for (name, props) in raw {
            let type_str = props
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Field '{}': missing 'type'", name))?;

            let state_type = match type_str {
                "bool" => StateType::Bool,
                "int" => StateType::Int,
                "float" => StateType::Float,
                "enum" => {
                    let values = props
                        .get("values")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| format!("Field '{}': enum requires 'values' array", name))?
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .map(|s| s.to_string())
                                .ok_or_else(|| format!("Field '{}': enum values must be strings", name))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if values.is_empty() {
                        return Err(format!("Field '{}': enum must have at least one value", name));
                    }
                    StateType::Enum { values }
                }
                "set[int]" => {
                    let capacity = props
                        .get("capacity")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1000) as usize;
                    StateType::IntSet { capacity }
                }
                "window[float]" => {
                    let capacity = props
                        .get("capacity")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(100) as usize;
                    StateType::FloatWindow { capacity }
                }
                other => {
                    return Err(format!(
                        "Field '{}': unsupported type '{}'. Supported: bool, int, float, enum, set[int], window[float]",
                        name, other
                    ));
                }
            };

            let default_json = props.get("default").cloned();

            fields.push(FieldDef {
                name: name.clone(),
                state_type,
                default_json,
            });
        }

        // Sort by name for deterministic ordering
        fields.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(StateSchema { fields })
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StateStore — the runtime state container
// ---------------------------------------------------------------------------

/// Runtime container holding all typed state variables for a strategy.
///
/// Constructed once from [`StateSchema`] at job start, then mutated
/// tick-by-tick in pure Rust (no Python, no GIL).
#[derive(Debug, Clone)]
pub struct StateStore {
    pub bools: HashMap<String, bool>,
    pub ints: HashMap<String, i64>,
    pub floats: HashMap<String, f64>,
    /// `(current_variant_index, variant_names)`.
    pub enums: HashMap<String, (u8, Vec<String>)>,
    pub int_sets: HashMap<String, BoundedHashSet<i64>>,
    pub float_windows: HashMap<String, BoundedDeque<f64>>,
}

impl StateStore {
    /// Construct a [`StateStore`] from a [`StateSchema`], applying defaults.
    pub fn from_schema(schema: &StateSchema) -> Self {
        let mut store = StateStore {
            bools: HashMap::new(),
            ints: HashMap::new(),
            floats: HashMap::new(),
            enums: HashMap::new(),
            int_sets: HashMap::new(),
            float_windows: HashMap::new(),
        };

        for field in &schema.fields {
            match &field.state_type {
                StateType::Bool => {
                    let default = field
                        .default_json
                        .as_ref()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    store.bools.insert(field.name.clone(), default);
                }
                StateType::Int => {
                    let default = field
                        .default_json
                        .as_ref()
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    store.ints.insert(field.name.clone(), default);
                }
                StateType::Float => {
                    let default = field
                        .default_json
                        .as_ref()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    store.floats.insert(field.name.clone(), default);
                }
                StateType::Enum { values } => {
                    let default_idx = if let Some(def) = field.default_json.as_ref().and_then(|v| v.as_str()) {
                        values.iter().position(|v| v == def).unwrap_or(0) as u8
                    } else {
                        0
                    };
                    store.enums.insert(field.name.clone(), (default_idx, values.clone()));
                }
                StateType::IntSet { capacity } => {
                    store.int_sets.insert(field.name.clone(), BoundedHashSet::new(*capacity));
                }
                StateType::FloatWindow { capacity } => {
                    store.float_windows.insert(field.name.clone(), BoundedDeque::new(*capacity));
                }
            }
        }

        store
    }

    // ---- Bool accessors ----

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bools.get(name).copied()
    }

    pub fn set_bool(&mut self, name: &str, value: bool) -> bool {
        if let Some(v) = self.bools.get_mut(name) {
            *v = value;
            true
        } else {
            false
        }
    }

    // ---- Int accessors ----

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.ints.get(name).copied()
    }

    pub fn set_int(&mut self, name: &str, value: i64) -> bool {
        if let Some(v) = self.ints.get_mut(name) {
            *v = value;
            true
        } else {
            false
        }
    }

    // ---- Float accessors ----

    pub fn get_float(&self, name: &str) -> Option<f64> {
        self.floats.get(name).copied()
    }

    pub fn set_float(&mut self, name: &str, value: f64) -> bool {
        if let Some(v) = self.floats.get_mut(name) {
            *v = value;
            true
        } else {
            false
        }
    }

    // ---- Enum accessors ----

    /// Get the current enum variant name.
    pub fn get_enum(&self, name: &str) -> Option<&str> {
        self.enums.get(name).and_then(|(idx, variants)| {
            variants.get(*idx as usize).map(|s| s.as_str())
        })
    }

    /// Get the current enum variant index.
    pub fn get_enum_index(&self, name: &str) -> Option<u8> {
        self.enums.get(name).map(|(idx, _)| *idx)
    }

    /// Set enum by variant name. Returns false if name not found or variant unknown.
    pub fn set_enum(&mut self, name: &str, variant: &str) -> bool {
        if let Some((idx, variants)) = self.enums.get_mut(name) {
            if let Some(pos) = variants.iter().position(|v| v == variant) {
                *idx = pos as u8;
                return true;
            }
        }
        false
    }

    /// Set enum by index.
    pub fn set_enum_index(&mut self, name: &str, index: u8) -> bool {
        if let Some((idx, variants)) = self.enums.get_mut(name) {
            if (index as usize) < variants.len() {
                *idx = index;
                return true;
            }
        }
        false
    }

    // ---- Window accessors ----

    pub fn window_push(&mut self, name: &str, value: f64) -> bool {
        if let Some(w) = self.float_windows.get_mut(name) {
            w.push(value);
            true
        } else {
            false
        }
    }

    pub fn window_mean(&self, name: &str) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.mean())
    }

    pub fn window_std(&self, name: &str) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.std())
    }

    pub fn window_tail_mean(&self, name: &str, n: usize) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.tail_mean(n))
    }

    pub fn window_tail_std(&self, name: &str, n: usize) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.tail_std(n))
    }

    pub fn window_tail_max(&self, name: &str, n: usize) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.tail_max(n))
    }

    pub fn window_tail_min(&self, name: &str, n: usize) -> Option<f64> {
        self.float_windows.get(name).map(|w| w.tail_min(n))
    }

    pub fn window_last(&self, name: &str) -> Option<f64> {
        self.float_windows.get(name).and_then(|w| w.back().copied())
    }

    pub fn window_nth_from_back(&self, name: &str, n: usize) -> Option<f64> {
        self.float_windows
            .get(name)
            .and_then(|w| w.nth_from_back(n).copied())
    }

    pub fn window_len(&self, name: &str) -> Option<usize> {
        self.float_windows.get(name).map(|w| w.len())
    }

    // ---- Set accessors ----

    pub fn set_insert(&mut self, name: &str, value: i64) -> bool {
        if let Some(s) = self.int_sets.get_mut(name) {
            s.insert(value);
            true
        } else {
            false
        }
    }

    pub fn set_contains(&self, name: &str, value: i64) -> bool {
        self.int_sets
            .get(name)
            .map(|s| s.contains(&value))
            .unwrap_or(false)
    }

    pub fn set_remove(&mut self, name: &str, value: i64) -> bool {
        if let Some(s) = self.int_sets.get_mut(name) {
            s.remove(&value)
        } else {
            false
        }
    }

    pub fn set_len(&self, name: &str) -> Option<usize> {
        self.int_sets.get(name).map(|s| s.len())
    }

    // ---- Bulk export (for sending to Python in Tier 1/2) ----

    /// Export all state as a flat JSON dict for Python interop.
    pub fn to_json_map(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();

        for (k, v) in &self.bools {
            map.insert(k.clone(), serde_json::Value::Bool(*v));
        }
        for (k, v) in &self.ints {
            map.insert(k.clone(), serde_json::json!(*v));
        }
        for (k, v) in &self.floats {
            map.insert(k.clone(), serde_json::json!(*v));
        }
        for (k, (idx, variants)) in &self.enums {
            if let Some(name) = variants.get(*idx as usize) {
                map.insert(k.clone(), serde_json::Value::String(name.clone()));
            }
        }
        for (k, w) in &self.float_windows {
            let arr: Vec<f64> = w.iter().copied().collect();
            map.insert(k.clone(), serde_json::json!(arr));
        }
        for (k, s) in &self.int_sets {
            let arr: Vec<i64> = s.insertion_order.iter().copied().collect();
            map.insert(k.clone(), serde_json::json!(arr));
        }

        serde_json::Value::Object(map)
    }

    /// Update state from a JSON dict returned by Python.
    /// Only updates fields that exist in the schema.
    pub fn update_from_json(&mut self, json: &serde_json::Value) {
        if let Some(obj) = json.as_object() {
            for (k, v) in obj {
                // Try each type
                if let Some(b) = v.as_bool() {
                    self.set_bool(k, b);
                } else if let Some(i) = v.as_i64() {
                    // Could be int or enum index
                    if self.ints.contains_key(k) {
                        self.set_int(k, i);
                    }
                } else if let Some(f) = v.as_f64() {
                    self.set_float(k, f);
                } else if let Some(s) = v.as_str() {
                    // Enum variant by name
                    self.set_enum(k, s);
                }
                // Arrays (windows, sets) not updated from Python — they're append-only in Rust
            }
        }
    }

    /// Reset all state to defaults from schema.
    pub fn reset(&mut self, schema: &StateSchema) {
        *self = Self::from_schema(schema);
    }
}

// ---------------------------------------------------------------------------
// PyO3 extraction
// ---------------------------------------------------------------------------

/// Extract `state_schema()` from a Python strategy source via PyO3.
/// Returns `None` if the strategy doesn't define `state_schema()`.
#[cfg(feature = "python")]
pub fn extract_state_schema(python_source: &str) -> Option<StateSchema> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyModule};
    use std::collections::HashMap as StdHashMap;

    Python::with_gil(|py| {
        // Inject SDK so `from trading_platform import ...` works
        crate::strategies::python_strategy::inject_sdk(py).ok()?;

        // Unique name per call -- see python_strategy.rs's
        // unique_module_name doc comment for why a fixed literal here leaked
        // a previous job's stale Strategy class into this compilation.
        let module =
            PyModule::from_code_bound(py, python_source, "strategy.py", &crate::strategies::python_strategy::unique_module_name()).ok()?;

        let strategy_class = module.getattr("Strategy").ok()?;
        let instance = strategy_class.call0().ok()?;

        if !instance.hasattr("state_schema").unwrap_or(false) {
            return None;
        }

        let schema_result = instance.call_method0("state_schema").ok()?;
        let schema_dict = schema_result.downcast::<PyDict>().ok()?;

        let mut raw: StdHashMap<String, StdHashMap<String, serde_json::Value>> = StdHashMap::new();

        for (key, value) in schema_dict.iter() {
            let field_name: String = key.extract().ok()?;
            let field_dict = value.downcast::<PyDict>().ok()?;

            let mut entry: StdHashMap<String, serde_json::Value> = StdHashMap::new();
            for (k, v) in field_dict.iter() {
                let k_str: String = k.extract().ok()?;
                let val = py_value_to_json(&v);
                entry.insert(k_str, val);
            }

            raw.insert(field_name, entry);
        }

        match StateSchema::from_raw_dict(&raw) {
            Ok(schema) => {
                crate::log_info!(crate::logging_facade::STRATEGY_LOGGER, "[STATE_STORE] Extracted state_schema: {} fields", schema.len());
                Some(schema)
            }
            Err(e) => {
                crate::log_warn!(crate::logging_facade::STRATEGY_LOGGER, "[STATE_STORE] Failed to parse state_schema: {}", e);
                None
            }
        }
    })
}

/// Stub for non-python builds.
#[cfg(not(feature = "python"))]
pub fn extract_state_schema(_python_source: &str) -> Option<StateSchema> {
    None
}

#[cfg(feature = "python")]
fn py_value_to_json(value: &pyo3::Bound<'_, pyo3::PyAny>) -> serde_json::Value {
    use pyo3::types::PyAnyMethods;

    // Try list/array first (for enum values, etc.)
    if let Ok(list) = value.downcast::<pyo3::types::PyList>() {
        use pyo3::types::PyListMethods;
        let items: Vec<serde_json::Value> = list
            .iter()
            .map(|item| py_value_to_json(&item))
            .collect();
        return serde_json::Value::Array(items);
    }

    if let Ok(v) = value.extract::<bool>() {
        serde_json::Value::Bool(v)
    } else if let Ok(v) = value.extract::<i64>() {
        serde_json::json!(v)
    } else if let Ok(v) = value.extract::<f64>() {
        serde_json::json!(v)
    } else if let Ok(v) = value.extract::<String>() {
        serde_json::Value::String(v)
    } else if value.is_none() {
        serde_json::Value::Null
    } else {
        serde_json::Value::Null
    }
}

// ---------------------------------------------------------------------------
// Extract model() and features() declarations
// ---------------------------------------------------------------------------

/// ONNX model reference parsed from Python `model()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    /// Path to the .onnx file (relative to upload directory).
    pub path: String,
    /// Runtime identifier (currently only "onnx").
    pub runtime: String,
    /// Optional output mapping: signal thresholds or class map.
    pub output_mapping: Option<OutputMapping>,
}

/// How to map ONNX output tensor to signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputMapping {
    /// Scalar output: `> buy_threshold → Buy`, `< sell_threshold → Sell`, else `Hold`.
    Threshold {
        buy_threshold: f64,
        sell_threshold: f64,
    },
    /// N-class classification: argmax over class names.
    Argmax { classes: Vec<String> },
}

/// A feature expression from Python `features()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExpr {
    pub name: String,
    pub expr: String,
}

/// Extract `model()` from a Python strategy source.
#[cfg(feature = "python")]
pub fn extract_model_ref(python_source: &str) -> Option<ModelRef> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyModule};

    Python::with_gil(|py| {
        // Inject SDK so `from trading_platform import ...` works
        crate::strategies::python_strategy::inject_sdk(py).ok()?;

        // Unique name per call -- see python_strategy.rs's
        // unique_module_name doc comment for why a fixed literal here leaked
        // a previous job's stale Strategy class into this compilation.
        let module =
            PyModule::from_code_bound(py, python_source, "strategy.py", &crate::strategies::python_strategy::unique_module_name()).ok()?;

        let strategy_class = module.getattr("Strategy").ok()?;
        let instance = strategy_class.call0().ok()?;

        if !instance.hasattr("model").unwrap_or(false) {
            return None;
        }

        let model_result = instance.call_method0("model").ok()?;
        let model_dict = model_result.downcast::<PyDict>().ok()?;

        let path: String = model_dict.get_item("path").ok()??.extract().ok()?;
        let runtime: String = model_dict
            .get_item("runtime")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "onnx".to_string());

        // Parse output_mapping if present
        let output_mapping = if let Ok(Some(mapping)) = model_dict.get_item("output_mapping") {
            if let Ok(mapping_dict) = mapping.downcast::<PyDict>() {
                if let Ok(Some(classes_val)) = mapping_dict.get_item("classes") {
                    // Argmax mapping
                    if let Ok(list) = classes_val.downcast::<pyo3::types::PyList>() {
                        let classes: Vec<String> = list
                            .iter()
                            .filter_map(|item| item.extract::<String>().ok())
                            .collect();
                        Some(OutputMapping::Argmax { classes })
                    } else {
                        None
                    }
                } else {
                    // Threshold mapping
                    let buy_threshold = mapping_dict
                        .get_item("buy_threshold")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f64>().ok())
                        .unwrap_or(0.5);
                    let sell_threshold = mapping_dict
                        .get_item("sell_threshold")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f64>().ok())
                        .unwrap_or(-0.5);
                    Some(OutputMapping::Threshold {
                        buy_threshold,
                        sell_threshold,
                    })
                }
            } else {
                None
            }
        } else {
            None
        };

        crate::log_info!(crate::logging_facade::STRATEGY_LOGGER, "[STATE_STORE] Extracted model(): path={}, runtime={}", path, runtime);

        Some(ModelRef {
            path,
            runtime,
            output_mapping,
        })
    })
}

#[cfg(not(feature = "python"))]
pub fn extract_model_ref(_python_source: &str) -> Option<ModelRef> {
    None
}

/// Extract `features()` from a Python strategy source.
#[cfg(feature = "python")]
pub fn extract_features(python_source: &str) -> Option<Vec<FeatureExpr>> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyList, PyModule};

    Python::with_gil(|py| {
        // Inject SDK so `from trading_platform import ...` works
        crate::strategies::python_strategy::inject_sdk(py).ok()?;

        // Unique name per call -- see python_strategy.rs's
        // unique_module_name doc comment for why a fixed literal here leaked
        // a previous job's stale Strategy class into this compilation.
        let module =
            PyModule::from_code_bound(py, python_source, "strategy.py", &crate::strategies::python_strategy::unique_module_name()).ok()?;

        let strategy_class = module.getattr("Strategy").ok()?;
        let instance = strategy_class.call0().ok()?;

        if !instance.hasattr("features").unwrap_or(false) {
            return None;
        }

        let features_result = instance.call_method0("features").ok()?;
        let features_list = features_result.downcast::<PyList>().ok()?;

        let mut exprs = Vec::new();
        for item in features_list.iter() {
            let item_dict = item.downcast::<PyDict>().ok()?;
            let name: String = item_dict.get_item("name").ok()??.extract().ok()?;
            let expr: String = item_dict.get_item("expr").ok()??.extract().ok()?;
            exprs.push(FeatureExpr { name, expr });
        }

        crate::log_info!(crate::logging_facade::STRATEGY_LOGGER, "[STATE_STORE] Extracted features(): {} expressions", exprs.len());

        Some(exprs)
    })
}

#[cfg(not(feature = "python"))]
pub fn extract_features(_python_source: &str) -> Option<Vec<FeatureExpr>> {
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_deque_eviction() {
        let mut d = BoundedDeque::new(3);
        d.push(1.0);
        d.push(2.0);
        d.push(3.0);
        assert_eq!(d.len(), 3);
        d.push(4.0); // evicts 1.0
        assert_eq!(d.len(), 3);
        assert_eq!(d.front(), Some(&2.0));
        assert_eq!(d.back(), Some(&4.0));
    }

    #[test]
    fn test_bounded_deque_tail_mean() {
        let mut d = BoundedDeque::new(10);
        for i in 1..=5 {
            d.push(i as f64);
        }
        assert!((d.tail_mean(3) - 4.0).abs() < 1e-10); // mean(3,4,5)=4
        assert!((d.mean() - 3.0).abs() < 1e-10); // mean(1,2,3,4,5)=3
    }

    #[test]
    fn test_bounded_deque_tail_std() {
        let mut d = BoundedDeque::new(10);
        d.push(2.0);
        d.push(4.0);
        d.push(4.0);
        d.push(4.0);
        d.push(5.0);
        d.push(5.0);
        d.push(7.0);
        d.push(9.0);
        // std of all: population std of [2,4,4,4,5,5,7,9] = 2.0
        assert!((d.std() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_hashset_eviction() {
        let mut s = BoundedHashSet::new(3);
        s.insert(1);
        s.insert(2);
        s.insert(3);
        assert_eq!(s.len(), 3);
        s.insert(4); // evicts 1
        assert_eq!(s.len(), 3);
        assert!(!s.contains(&1));
        assert!(s.contains(&4));
    }

    #[test]
    fn test_bounded_hashset_no_duplicates() {
        let mut s = BoundedHashSet::new(5);
        assert!(s.insert(1));
        assert!(!s.insert(1)); // duplicate
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_state_schema_from_raw() {
        let mut raw = HashMap::new();

        let mut bool_field = HashMap::new();
        bool_field.insert("type".into(), serde_json::json!("bool"));
        bool_field.insert("default".into(), serde_json::json!(true));
        raw.insert("is_active".into(), bool_field);

        let mut float_field = HashMap::new();
        float_field.insert("type".into(), serde_json::json!("float"));
        float_field.insert("default".into(), serde_json::json!(1.5));
        raw.insert("threshold".into(), float_field);

        let mut enum_field = HashMap::new();
        enum_field.insert("type".into(), serde_json::json!("enum"));
        enum_field.insert("values".into(), serde_json::json!(["flat", "long", "short"]));
        enum_field.insert("default".into(), serde_json::json!("flat"));
        raw.insert("position".into(), enum_field);

        let mut window_field = HashMap::new();
        window_field.insert("type".into(), serde_json::json!("window[float]"));
        window_field.insert("capacity".into(), serde_json::json!(50));
        raw.insert("prices".into(), window_field);

        let schema = StateSchema::from_raw_dict(&raw).unwrap();
        assert_eq!(schema.len(), 4);

        let store = StateStore::from_schema(&schema);
        assert_eq!(store.get_bool("is_active"), Some(true));
        assert_eq!(store.get_float("threshold"), Some(1.5));
        assert_eq!(store.get_enum("position"), Some("flat"));
        assert_eq!(store.window_len("prices"), Some(0));
    }

    #[test]
    fn test_state_store_mutations() {
        let mut raw = HashMap::new();

        let mut f = HashMap::new();
        f.insert("type".into(), serde_json::json!("float"));
        raw.insert("x".into(), f);

        let mut e = HashMap::new();
        e.insert("type".into(), serde_json::json!("enum"));
        e.insert("values".into(), serde_json::json!(["a", "b", "c"]));
        raw.insert("state".into(), e);

        let mut w = HashMap::new();
        w.insert("type".into(), serde_json::json!("window[float]"));
        w.insert("capacity".into(), serde_json::json!(3));
        raw.insert("buf".into(), w);

        let schema = StateSchema::from_raw_dict(&raw).unwrap();
        let mut store = StateStore::from_schema(&schema);

        // Float
        store.set_float("x", 42.0);
        assert_eq!(store.get_float("x"), Some(42.0));

        // Enum
        store.set_enum("state", "b");
        assert_eq!(store.get_enum("state"), Some("b"));
        assert_eq!(store.get_enum_index("state"), Some(1));

        // Window
        store.window_push("buf", 1.0);
        store.window_push("buf", 2.0);
        store.window_push("buf", 3.0);
        assert_eq!(store.window_len("buf"), Some(3));
        assert!((store.window_mean("buf").unwrap() - 2.0).abs() < 1e-10);
        store.window_push("buf", 4.0); // evicts 1.0
        assert_eq!(store.window_len("buf"), Some(3));
        assert!((store.window_mean("buf").unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_store_json_roundtrip() {
        let mut raw = HashMap::new();
        let mut f = HashMap::new();
        f.insert("type".into(), serde_json::json!("float"));
        raw.insert("x".into(), f);

        let mut b = HashMap::new();
        b.insert("type".into(), serde_json::json!("bool"));
        raw.insert("flag".into(), b);

        let schema = StateSchema::from_raw_dict(&raw).unwrap();
        let mut store = StateStore::from_schema(&schema);
        store.set_float("x", 3.14);
        store.set_bool("flag", true);

        let json = store.to_json_map();
        assert_eq!(json["x"], 3.14);
        assert_eq!(json["flag"], true);

        // Update from modified JSON
        let modified = serde_json::json!({"x": 2.72, "flag": false});
        store.update_from_json(&modified);
        assert_eq!(store.get_float("x"), Some(2.72));
        assert_eq!(store.get_bool("flag"), Some(false));
    }
}
