//! Strategy execution tiers — from pure-Rust (Tier 0) to full-Python (Tier 3).
//!
//! The platform analyses a Python strategy's declarations (`state_schema()`,
//! `model()`, `features()`) at job start and constructs the highest-performance
//! executor the strategy qualifies for:
//!
//! | Tier | State        | Inference              | Hot Path            |
//! |------|-------------|------------------------|---------------------|
//! | 0    | Rust `StateStore` | `ort` ONNX        | Pure Rust           |
//! | 1    | Rust `StateStore` | PyO3 signals       | Rust state + Python |
//! | 2    | Rust `StateStore` | PyO3 signals       | Typed state xfer    |
//! | 3    | Python-internal   | PyO3 full bridge   | Full Python         |

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::{log_info, log_warn};
use crate::logging_facade::STRATEGY_LOGGER;

use dataloader::MarketData;
use signal::Signal;

use crate::state_store::{
    extract_features, extract_model_ref, extract_state_schema, FeatureExpr,
    ModelRef, OutputMapping, StateSchema, StateStore,
};
use crate::feature_compiler::{self, CompiledFeature, CompilationResult};
use crate::indicator_registry::IndicatorBank;
#[cfg(feature = "python")]
use crate::indicator_registry::extract_indicators;
use crate::traits::{Strategy, StrategyContext, StrategyError, StrategyResult, VenueSeries};

// ---------------------------------------------------------------------------
// ExecutionTier enum
// ---------------------------------------------------------------------------

/// The execution tier assigned to a strategy based on its Python declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionTier {
    /// Pure Rust + ONNX — no Python on the hot path.
    Tier0,
    /// Rust `StateStore` + Python `generate_signals()`.
    Tier1,
    /// Rust `StateStore` + typed state transfer to Python.
    Tier2,
    /// Declarative MM pricing — pure Rust formula evaluation.
    Tier0_5,
    /// Full Python (existing `PythonStrategy` path).
    Tier3,
}

impl std::fmt::Display for ExecutionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionTier::Tier0 => write!(f, "Tier 0 (pure Rust + ONNX)"),
            ExecutionTier::Tier0_5 => write!(f, "Tier 0.5 (declarative MM pricing)"),
            ExecutionTier::Tier1 => write!(f, "Tier 1 (Rust state + Python logic)"),
            ExecutionTier::Tier2 => write!(f, "Tier 2 (typed state transfer)"),
            ExecutionTier::Tier3 => write!(f, "Tier 3 (full Python)"),
        }
    }
}

// ---------------------------------------------------------------------------
// StrategyExecutor trait
// ---------------------------------------------------------------------------

/// Trait for tick-by-tick strategy execution at any tier.
///
/// Implementors manage their own state and return signals per tick.
#[async_trait]
pub trait StrategyExecutor: Send + Sync {
    /// Which tier this executor is running at.
    fn tier(&self) -> ExecutionTier;

    /// Process one tick and return signals (may be empty for Hold).
    async fn on_tick(
        &mut self,
        data: &MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>>;

    /// Pre-enrich `custom_data` with state values and/or indicator outputs.
    ///
    /// When called before building a `StrategyContext`, the executor injects
    /// all enrichments directly into `custom_data` so that `on_tick` can
    /// skip its internal `context.clone()` — avoiding ~48 KB of copy per tick
    /// for the `price_history` and `volume_history` Vecs.
    ///
    /// Simulation loops that call this MUST set the returned `custom_data`
    /// on the `StrategyContext` they build.  Callers that do NOT call this
    /// method get the same behaviour as before (the executor clones internally).
    fn pre_enrich(&mut self, _data: &MarketData, _custom_data: &mut HashMap<String, f64>) {}

    /// Bulk-vectorized signal generation for the entire tick array.
    ///
    /// Returns `(Some(signals), None)` on success, where `signals[i]` encodes the
    /// action for tick i: 1 = BUY, -1 = SELL, 0 = HOLD, 2 = CLOSE.
    ///
    /// `prices` is the close series; `opens`/`highs`/`lows` carry the rest of the
    /// OHLC bar (degenerate to the trade price for tick data). They are only
    /// forwarded to strategies that opt in via `compute_signals_accepts_ohlc()`.
    ///
    /// Returns `(None, Some(error))` when a Python exception occurred.
    /// Returns `(None, None)` when not supported — caller falls back to per-tick `on_tick()`.
    async fn compute_all_signals(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> (Option<Vec<i8>>, Option<String>) {
        let _ = (prices, volumes, timestamps, opens, highs, lows);
        (None, None)
    }

    /// Whether the underlying strategy's `compute_signals` accepts the optional
    /// OHLC arrays. When `false`, the caller can skip building opens/highs/lows.
    fn compute_signals_accepts_ohlc(&self) -> bool {
        false
    }

    /// Cross-exchange strategy support (2026-07-22): executor-level mirror of
    /// `Strategy::compute_all_signals_multi_venue`, translated into this
    /// trait's flatter `(Option<Vec<i8>>, Option<String>)` shape exactly like
    /// `compute_all_signals` above. Default: not supported — only
    /// `PythonExecutor` (Tier 3, the only tier where custom multi-venue
    /// Python logic makes sense in V1) overrides this.
    async fn compute_all_signals_multi_venue(
        &mut self,
        venues: &HashMap<(String, String), VenueSeries>,
    ) -> (Option<Vec<i8>>, Option<String>) {
        let _ = venues;
        (None, None)
    }

    /// Whether this executor supports `compute_all_signals_multi_venue`.
    fn accepts_multi_venue(&self) -> bool {
        false
    }

    /// Executor-level mirror of `Strategy::declares_pair_trade`. Default:
    /// not supported — only executors wrapping a `PythonStrategy` (Hybrid/
    /// Python tiers) can ever return `Some`, since `pair_spec()` is a Python-
    /// authored declaration; `NativeExecutor` (Tier0) has no Python object
    /// to ask.
    fn declares_pair_trade(&self) -> Option<crate::traits::PairSpec> {
        None
    }

    /// Executor-level mirror of `Strategy::pair_spec_error`.
    fn pair_spec_error(&self) -> Option<String> {
        None
    }

    /// Executor-level mirror of `Strategy::declares_basket_trade`. Default:
    /// not supported, same reasoning as `declares_pair_trade` above.
    fn declares_basket_trade(&self) -> Option<crate::traits::BasketSpec> {
        None
    }

    /// Executor-level mirror of `Strategy::basket_spec_error`.
    fn basket_spec_error(&self) -> Option<String> {
        None
    }

    /// Per-trade dynamic position sizing (2026-07-25): executor-level mirror
    /// of `Strategy::compute_all_position_sizes`, translated into this
    /// trait's flatter `(Option<Vec<f64>>, Option<String>)` shape exactly
    /// like `compute_all_signals` above. Default: not supported — only
    /// `HybridExecutor`/`PythonExecutor` (the tiers with an underlying
    /// `PythonStrategy`) override this.
    async fn compute_all_position_sizes(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<Vec<f64>>, Option<String>) {
        let _ = (prices, volumes, timestamps);
        (None, None)
    }

    /// Whether this executor supports `compute_all_position_sizes`.
    fn accepts_position_sizes(&self) -> bool {
        false
    }

    /// Whether the underlying strategy defines `generate_signals` and should
    /// be dispatched per-tick with the full `StrategyContext` instead of via
    /// the vectorized `compute_all_signals` bulk path.
    fn defines_generate_signals(&self) -> bool {
        false
    }

    /// Optional bulk-vectorized feature computation, alongside `compute_all_signals`.
    ///
    /// Returns `(Some(features), None)` on success, where `features` maps a
    /// feature name to a value per tick, aligned 1:1 with `prices`. Returns
    /// `(None, Some(error))` on a Python exception. Returns `(None, None)`
    /// when the strategy has no `compute_features()` method, when the
    /// feature is disabled (`ENABLE_COMPUTE_FEATURES`), or when the executor
    /// doesn't support it at all — this is diagnostic context only and never
    /// affects trading signals or the pass/fail of a backtest.
    async fn compute_all_features(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<HashMap<String, Vec<f64>>>, Option<String>) {
        let _ = (prices, volumes, timestamps);
        (None, None)
    }

    /// Display name declared by the underlying strategy (Python `name()` for
    /// custom strategies). `None` for executors without a self-describing
    /// strategy. Used by persistence to label results correctly instead of
    /// falling back to the generic strategy type ("Custom").
    fn strategy_name(&self) -> Option<String> {
        None
    }

    /// Strategy-declared max leverage cap (see `Strategy::declared_max_leverage`).
    /// `f64::INFINITY` (no additional cap) for executors without an
    /// underlying strategy that could declare one.
    fn declared_max_leverage(&self) -> f64 {
        f64::INFINITY
    }

    /// Get a snapshot of the Rust-managed state (None for Tier 3).
    fn state_snapshot(&self) -> Option<&StateStore>;

    /// Human-readable description of the execution tier and reason.
    fn tier_description(&self) -> String;

    /// Historical points needed before strategy can emit valid signals.
    ///
    /// Default remains conservative for executors that don't provide
    /// strategy-specific requirements.
    fn required_data_window(&self) -> usize {
        100
    }
}

// ---------------------------------------------------------------------------
// TierAnalysis — result of analysing a Python strategy
// ---------------------------------------------------------------------------

/// The result of analysing a Python strategy's declarations to determine
/// the best execution tier.
#[derive(Debug)]
pub struct TierAnalysis {
    pub tier: ExecutionTier,
    pub state_schema: Option<StateSchema>,
    pub model_ref: Option<ModelRef>,
    pub feature_exprs: Option<Vec<FeatureExpr>>,
    pub compiled_features: Option<Vec<CompiledFeature>>,
    pub compile_errors: Vec<String>,
    pub reason: String,
}

/// Analyse a Python strategy source to determine the best execution tier
/// and extract all declarations needed to construct an executor.
pub fn analyse_strategy(python_source: &str) -> TierAnalysis {
    // Step 1: Extract state_schema()
    let state_schema = extract_state_schema(python_source);
    if state_schema.is_none() {
        return TierAnalysis {
            tier: ExecutionTier::Tier3,
            state_schema: None,
            model_ref: None,
            feature_exprs: None,
            compiled_features: None,
            compile_errors: vec![],
            reason: "No state_schema() defined — full Python execution".into(),
        };
    }
    let schema = state_schema.unwrap();

    // Step 1.5: Check for pricing() — if present, Tier 0.5 declarative MM pricing
    let pricing_raw = crate::mm_pricing::extract_pricing(python_source);
    if let Some(ref raw) = pricing_raw {
        match crate::mm_pricing::PricingFormula::compile(raw) {
            Ok(_) => {
                return TierAnalysis {
                    tier: ExecutionTier::Tier0_5,
                    state_schema: Some(schema),
                    model_ref: None,
                    feature_exprs: None,
                    compiled_features: None,
                    compile_errors: vec![],
                    reason: "state_schema() + pricing() → declarative MM pricing (pure Rust)".into(),
                };
            }
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "[EXECUTOR] pricing() compile error: {} — falling through", e);
            }
        }
    }

    // Step 2: Extract model()
    let model_ref = extract_model_ref(python_source);

    // Step 3: Extract features() if model() exists
    let feature_exprs = if model_ref.is_some() {
        extract_features(python_source)
    } else {
        None
    };

    // Step 4: Compile feature expressions
    let (compiled_features, compile_errors, all_compiled) = if let Some(ref exprs) = feature_exprs {
        let pairs: Vec<(String, String)> = exprs
            .iter()
            .map(|fe| (fe.name.clone(), fe.expr.clone()))
            .collect();
        let result: CompilationResult = feature_compiler::compile_features(&pairs);
        let errors: Vec<String> = result.errors.iter().map(|e| e.to_string()).collect();
        (Some(result.compiled), errors, result.all_compiled)
    } else {
        (None, vec![], false)
    };

    // Step 5: Determine tier
    let tier = if model_ref.is_some() && feature_exprs.is_some() && all_compiled {
        ExecutionTier::Tier0
    } else if model_ref.is_some() && feature_exprs.is_some() && !all_compiled {
        // Model present but some features didn't compile — Tier 1
        ExecutionTier::Tier1
    } else {
        // state_schema present but no model — Tier 1 (Rust state + Python logic)
        ExecutionTier::Tier1
    };

    let reason = match tier {
        ExecutionTier::Tier0 => format!(
            "state_schema() + model() + all {} features compiled → pure Rust + ONNX",
            compiled_features.as_ref().map(|c| c.len()).unwrap_or(0)
        ),
        ExecutionTier::Tier0_5 => "state_schema() + pricing() → declarative MM pricing".into(),
        ExecutionTier::Tier1 => {
            if !compile_errors.is_empty() {
                format!(
                    "state_schema() present, {} feature compile errors → Rust state + Python logic",
                    compile_errors.len()
                )
            } else {
                "state_schema() present, no model() → Rust state + Python logic".into()
            }
        }
        ExecutionTier::Tier2 => "state_schema() with typed state transfer".into(),
        ExecutionTier::Tier3 => "Full Python execution".into(),
    };

    log_info!(STRATEGY_LOGGER, "[EXECUTOR] Strategy analysis: {} — {}", tier, reason);

    TierAnalysis {
        tier,
        state_schema: Some(schema),
        model_ref,
        feature_exprs,
        compiled_features,
        compile_errors,
        reason,
    }
}

// ---------------------------------------------------------------------------
// NativeExecutor — Tier 0: Pure Rust + ONNX
// ---------------------------------------------------------------------------

/// Tier 0 executor: pure Rust state management + compiled features + ONNX inference.
/// No Python on the hot path.
pub struct NativeExecutor {
    pub state_store: StateStore,
    pub schema: StateSchema,
    pub compiled_features: Vec<CompiledFeature>,
    pub model_ref: ModelRef,
    /// ONNX session (loaded via ort crate). For now we store the path and
    /// defer actual ONNX loading to when the `onnx` feature is enabled.
    pub onnx_path: String,
    /// Signal thresholds for mapping ONNX output.
    pub output_mapping: OutputMapping,
    /// Auto-update: which state windows to push price/volume to each tick.
    pub price_windows: Vec<String>,
    pub volume_windows: Vec<String>,
}

impl NativeExecutor {
    /// Extract price and volume from MarketData.
    fn extract_price_volume(data: &MarketData) -> (f64, f64) {
        match data {
            MarketData::Trade(t) => (t.price, t.quantity),
            MarketData::Candle(c) => (
                c.close,
                c.volume,
            ),
            MarketData::PoolSwap(s) => (s.price(), s.amount_in),
            MarketData::Generic(g) => (g.price, g.quantity),
            MarketData::OptionCandle(c) => (c.close, c.volume),
        }
    }

    /// Auto-push price/volume to windows whose names contain "price" or "volume".
    fn auto_update_state(&mut self, price: f64, volume: f64) {
        for name in &self.price_windows {
            self.state_store.window_push(name, price);
        }
        for name in &self.volume_windows {
            self.state_store.window_push(name, volume);
        }
    }

    /// Map ONNX output to signals using the output mapping configuration.
    #[allow(dead_code)]
    fn map_output_to_signals(
        &self,
        output: &[f64],
        context: &StrategyContext,
    ) -> Vec<Signal> {
        use chrono::Utc;

        match &self.output_mapping {
            OutputMapping::Threshold {
                buy_threshold,
                sell_threshold,
            } => {
                if output.is_empty() {
                    return vec![];
                }
                let val = output[0];
                let signal_type = if val > *buy_threshold {
                    signal::SignalType::Buy
                } else if val < *sell_threshold {
                    signal::SignalType::Sell
                } else {
                    return vec![]; // Hold
                };

                vec![Signal {
                    id: format!("onnx_{}", context.tick_number).into(),
                    timestamp: Utc::now(),
                    signal_type,
                    strength: signal::SignalStrength::Strong,
                    price: None,
                    quantity: None,
                    reason: signal::SignalReason::Custom(
                        "ONNX".into(),
                        format!("output={:.4}", val),
                    ),
                    metadata: HashMap::new(),
                    numeric_metadata: {
                        let mut m = HashMap::new();
                        m.insert("onnx_output".into(), val);
                        m
                    },
                    is_limit: false,
                }]
            }
            OutputMapping::Argmax { classes } => {
                if output.is_empty() || classes.is_empty() {
                    return vec![];
                }
                // Find argmax
                let (max_idx, max_val) = output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                let class_name = classes
                    .get(max_idx)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into());

                let signal_type = match class_name.to_lowercase().as_str() {
                    "buy" | "long" | "up" => signal::SignalType::Buy,
                    "sell" | "short" | "down" => signal::SignalType::Sell,
                    "hold" | "neutral" | "flat" => return vec![],
                    _ => return vec![],
                };

                vec![Signal {
                    id: format!("onnx_{}", context.tick_number).into(),
                    timestamp: Utc::now(),
                    signal_type,
                    strength: signal::SignalStrength::Strong,
                    price: None,
                    quantity: None,
                    reason: signal::SignalReason::Custom(
                        "ONNX".into(),
                        format!("class={}, prob={:.4}", class_name, max_val),
                    ),
                    metadata: HashMap::new(),
                    numeric_metadata: {
                        let mut m = HashMap::new();
                        m.insert("onnx_confidence".into(), *max_val);
                        m
                    },
                    is_limit: false,
                }]
            }
        }
    }
}

#[async_trait]
impl StrategyExecutor for NativeExecutor {
    fn tier(&self) -> ExecutionTier {
        ExecutionTier::Tier0
    }

    async fn on_tick(
        &mut self,
        data: &MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let (price, volume) = Self::extract_price_volume(data);
        if price <= 0.0 {
            return Ok(vec![]);
        }

        // 1. Auto-update state windows
        self.auto_update_state(price, volume);

        // 2. Compile feature vector
        let feature_vec = feature_compiler::evaluate_features(
            &self.compiled_features,
            &self.state_store,
            context,
        );

        // 3. ONNX inference
        // NOTE: Actual ort::Session integration requires the `onnx` feature.
        // For now, we use the feature vector length as a placeholder that
        // will be wired to ort when the dependency is added.
        // In production: let output = self.session.run(input_tensor)?;
        let _ = &feature_vec; // used by ONNX session

        // Placeholder: return empty signals until ONNX is wired
        // Real implementation will call: self.ort_session.run(...)
        log_warn!(STRATEGY_LOGGER, "[NATIVE_EXECUTOR] ONNX inference not yet wired — {} features computed, returning Hold", feature_vec.len());
        Ok(vec![])
    }

    fn state_snapshot(&self) -> Option<&StateStore> {
        Some(&self.state_store)
    }

    fn tier_description(&self) -> String {
        format!(
            "Tier 0 (pure Rust + ONNX): {} compiled features, model={}",
            self.compiled_features.len(),
            self.onnx_path
        )
    }
}

// ---------------------------------------------------------------------------
// HybridExecutor — Tier 1: Rust StateStore + Python generate_signals()
// ---------------------------------------------------------------------------

/// Tier 1 executor: state managed in Rust, signal logic delegated to Python.
///
/// State windows are auto-updated per tick in Rust. The state is serialized
/// to JSON and injected into the Python context before calling
/// `generate_signals()`. State changes from Python are written back.
pub struct HybridExecutor {
    pub state_store: StateStore,
    pub schema: StateSchema,
    /// The underlying Python strategy (used only for generate_signals).
    pub py_strategy: Box<dyn Strategy>,
    /// Window names to auto-push price/volume data.
    pub price_windows: Vec<String>,
    pub volume_windows: Vec<String>,
    /// Pre-computed Rust indicators (from strategy's `indicators()` declaration).
    pub indicator_bank: Option<IndicatorBank>,
    /// Set by `pre_enrich`, cleared by `on_tick`.
    pre_enriched: bool,
    /// Pre-computed "state.{key}" strings for inject_state() — avoids format!() per tick.
    state_key_cache: StateKeyCache,
    /// Pre-computed "ind.{key}" strings for indicator injection.
    indicator_prefix_cache: HashMap<String, String>,
}

/// Pre-computed formatted key strings for `inject_state()`,
/// eliminating ~15 `format!()` allocations per tick.
struct StateKeyCache {
    float_keys: Vec<(String, String)>,   // (original_key, "state.{key}")
    bool_keys: Vec<(String, String)>,
    int_keys: Vec<(String, String)>,
    enum_keys: Vec<(String, String)>,
    /// Window keys: (original_key, mean_key, std_key, len_key, last_key)
    window_keys: Vec<(String, String, String, String, String)>,
}

impl StateKeyCache {
    #[allow(dead_code)]
    fn from_schema(schema: &StateSchema) -> Self {
        use crate::state_store::StateType;
        let mut float_keys = Vec::new();
        let mut bool_keys = Vec::new();
        let mut int_keys = Vec::new();
        let mut enum_keys = Vec::new();
        let mut window_keys = Vec::new();
        for field in &schema.fields {
            let name = &field.name;
            match &field.state_type {
                StateType::Float => float_keys.push((name.clone(), format!("state.{}", name))),
                StateType::Bool => bool_keys.push((name.clone(), format!("state.{}", name))),
                StateType::Int => int_keys.push((name.clone(), format!("state.{}", name))),
                StateType::Enum { .. } => enum_keys.push((name.clone(), format!("state.{}", name))),
                StateType::IntSet { .. } => {},
                StateType::FloatWindow { .. } => window_keys.push((
                    name.clone(),
                    format!("state.{}.mean", name),
                    format!("state.{}.std", name),
                    format!("state.{}.len", name),
                    format!("state.{}.last", name),
                )),
            }
        }
        Self { float_keys, bool_keys, int_keys, enum_keys, window_keys }
    }
}

impl HybridExecutor {
    fn extract_price_volume(data: &MarketData) -> (f64, f64) {
        match data {
            MarketData::Trade(t) => (t.price, t.quantity),
            MarketData::Candle(c) => (
                c.close,
                c.volume,
            ),
            MarketData::PoolSwap(s) => (s.price(), s.amount_in),
            MarketData::Generic(g) => (g.price, g.quantity),
            MarketData::OptionCandle(c) => (c.close, c.volume),
        }
    }

    /// Inject state-store values into a `custom_data` map using pre-computed keys.
    /// Uses `get_mut` to update existing entries in-place, avoiding String::clone()
    /// on ticks after the first (when keys already exist in the map).
    fn inject_state(&self, custom_data: &mut HashMap<String, f64>) {
        for (orig, cached) in &self.state_key_cache.float_keys {
            if let Some(v) = self.state_store.floats.get(orig) {
                if let Some(slot) = custom_data.get_mut(cached.as_str()) {
                    *slot = *v;
                } else {
                    custom_data.insert(cached.clone(), *v);
                }
            }
        }
        for (orig, cached) in &self.state_key_cache.bool_keys {
            if let Some(v) = self.state_store.bools.get(orig) {
                let val = if *v { 1.0 } else { 0.0 };
                if let Some(slot) = custom_data.get_mut(cached.as_str()) {
                    *slot = val;
                } else {
                    custom_data.insert(cached.clone(), val);
                }
            }
        }
        for (orig, cached) in &self.state_key_cache.int_keys {
            if let Some(v) = self.state_store.ints.get(orig) {
                let val = *v as f64;
                if let Some(slot) = custom_data.get_mut(cached.as_str()) {
                    *slot = val;
                } else {
                    custom_data.insert(cached.clone(), val);
                }
            }
        }
        for (orig, cached) in &self.state_key_cache.enum_keys {
            if let Some((idx, _)) = self.state_store.enums.get(orig) {
                let val = *idx as f64;
                if let Some(slot) = custom_data.get_mut(cached.as_str()) {
                    *slot = val;
                } else {
                    custom_data.insert(cached.clone(), val);
                }
            }
        }
        for (orig, mean_k, std_k, len_k, last_k) in &self.state_key_cache.window_keys {
            if let Some(w) = self.state_store.float_windows.get(orig) {
                let mean_v = w.mean();
                let std_v = w.std();
                let len_v = w.len() as f64;
                if let Some(slot) = custom_data.get_mut(mean_k.as_str()) {
                    *slot = mean_v;
                } else {
                    custom_data.insert(mean_k.clone(), mean_v);
                }
                if let Some(slot) = custom_data.get_mut(std_k.as_str()) {
                    *slot = std_v;
                } else {
                    custom_data.insert(std_k.clone(), std_v);
                }
                if let Some(slot) = custom_data.get_mut(len_k.as_str()) {
                    *slot = len_v;
                } else {
                    custom_data.insert(len_k.clone(), len_v);
                }
                if let Some(last) = w.back() {
                    if let Some(slot) = custom_data.get_mut(last_k.as_str()) {
                        *slot = *last;
                    } else {
                        custom_data.insert(last_k.clone(), *last);
                    }
                }
            }
        }
    }
}

#[async_trait]
impl StrategyExecutor for HybridExecutor {
    fn tier(&self) -> ExecutionTier {
        ExecutionTier::Tier1
    }

    fn pre_enrich(&mut self, data: &MarketData, custom_data: &mut HashMap<String, f64>) {
        let (price, volume) = Self::extract_price_volume(data);

        // Update state windows in Rust (no Python, no GIL).
        for name in &self.price_windows {
            self.state_store.window_push(name, price);
        }
        for name in &self.volume_windows {
            self.state_store.window_push(name, volume);
        }

        // Inject state values into custom_data.
        self.inject_state(custom_data);

        // Update & inject Rust indicators.
        if let Some(ref mut bank) = self.indicator_bank {
            bank.update_all(price, volume);
            bank.inject_into(custom_data, &self.indicator_prefix_cache);
        }

        self.pre_enriched = true;
    }

    async fn on_tick(
        &mut self,
        data: &MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let (price, volume) = Self::extract_price_volume(data);

        if !self.pre_enriched {
            // Legacy path: caller didn't call pre_enrich — update Rust-side state first.
            for name in &self.price_windows {
                self.state_store.window_push(name, price);
            }
            for name in &self.volume_windows {
                self.state_store.window_push(name, volume);
            }
            if let Some(ref mut bank) = self.indicator_bank {
                bank.update_all(price, volume);
            }
        }
        self.pre_enriched = false;

        if self.py_strategy.defines_generate_signals() {
            return self.py_strategy.generate_signals(data, context).await;
        }

        // Delegate to the same compute_all_signals API used by the vectorized path,
        // passing a single-element slice so the Python strategy receives a 1-tick batch.
        let ts = context.timestamp.timestamp_millis();
        match self.py_strategy.compute_all_signals(&[price], &[volume], &[ts], &[price], &[price], &[price]).await {
            Ok(Some(sigs)) if !sigs.is_empty() => Ok(i8_to_signal_vec(sigs[0], price)),
            Ok(_) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    fn state_snapshot(&self) -> Option<&StateStore> {
        Some(&self.state_store)
    }

    async fn compute_all_signals(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> (Option<Vec<i8>>, Option<String>) {
        match self.py_strategy.compute_all_signals(prices, volumes, timestamps, opens, highs, lows).await {
            Ok(signals) => (signals, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_signals failed (HybridExecutor), falling back to per-tick: {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    async fn compute_all_signals_multi_venue(
        &mut self,
        venues: &HashMap<(String, String), VenueSeries>,
    ) -> (Option<Vec<i8>>, Option<String>) {
        match self.py_strategy.compute_all_signals_multi_venue(venues).await {
            Ok(signals) => (signals, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_signals_multi_venue failed (HybridExecutor): {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    fn accepts_multi_venue(&self) -> bool {
        self.py_strategy.accepts_multi_venue()
    }

    fn declares_pair_trade(&self) -> Option<crate::traits::PairSpec> {
        self.py_strategy.declares_pair_trade()
    }

    fn pair_spec_error(&self) -> Option<String> {
        self.py_strategy.pair_spec_error()
    }

    fn declares_basket_trade(&self) -> Option<crate::traits::BasketSpec> {
        self.py_strategy.declares_basket_trade()
    }

    fn basket_spec_error(&self) -> Option<String> {
        self.py_strategy.basket_spec_error()
    }

    fn compute_signals_accepts_ohlc(&self) -> bool {
        self.py_strategy.compute_signals_accepts_ohlc()
    }

    async fn compute_all_position_sizes(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<Vec<f64>>, Option<String>) {
        match self.py_strategy.compute_all_position_sizes(prices, volumes, timestamps).await {
            Ok(sizes) => (sizes, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_position_sizes failed (HybridExecutor), falling back to position_size_pct: {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    fn accepts_position_sizes(&self) -> bool {
        self.py_strategy.accepts_position_sizes()
    }

    fn defines_generate_signals(&self) -> bool {
        self.py_strategy.defines_generate_signals()
    }

    async fn compute_all_features(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<HashMap<String, Vec<f64>>>, Option<String>) {
        match self.py_strategy.compute_all_features(prices, volumes, timestamps).await {
            Ok(features) => (features, None),
            Err(e) => (None, Some(format!("{}", e))),
        }
    }

    fn strategy_name(&self) -> Option<String> {
        Some(self.py_strategy.name().to_string())
    }

    fn declared_max_leverage(&self) -> f64 {
        self.py_strategy.declared_max_leverage()
    }

    fn tier_description(&self) -> String {
        let ind_count = self.indicator_bank.as_ref().map_or(0, |b| b.len());
        format!(
            "Tier 1 (Rust state + Python logic): {} state fields, {} price windows, {} volume windows, {} indicators",
            self.schema.fields.len(),
            self.price_windows.len(),
            self.volume_windows.len(),
            ind_count,
        )
    }

    fn required_data_window(&self) -> usize {
        self.py_strategy.required_data_window()
    }
}

// ---------------------------------------------------------------------------
// PythonExecutor — Tier 3: Full Python (wraps existing PythonStrategy)
// ---------------------------------------------------------------------------

/// Tier 3 executor: wraps an existing `PythonStrategy` with no Rust-side
/// state management. This is the fallback when no `state_schema()` is defined.
pub struct PythonExecutor {
    pub py_strategy: Box<dyn Strategy>,
    /// Pre-computed Rust indicators (from strategy's `indicators()` declaration).
    pub indicator_bank: Option<IndicatorBank>,
    /// Set by `pre_enrich`, cleared by `on_tick`.
    pre_enriched: bool,
    /// Pre-computed "ind.{key}" strings for indicator injection.
    indicator_prefix_cache: HashMap<String, String>,
}

#[async_trait]
impl StrategyExecutor for PythonExecutor {
    fn tier(&self) -> ExecutionTier {
        ExecutionTier::Tier3
    }

    fn pre_enrich(&mut self, data: &MarketData, custom_data: &mut HashMap<String, f64>) {
        if let Some(ref mut bank) = self.indicator_bank {
            let (price, volume) = HybridExecutor::extract_price_volume(data);
            bank.update_all(price, volume);
            bank.inject_into(custom_data, &self.indicator_prefix_cache);
        }
        self.pre_enriched = true;
    }

    async fn on_tick(
        &mut self,
        data: &MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let (price, volume) = HybridExecutor::extract_price_volume(data);

        if !self.pre_enriched {
            // Legacy path: update Rust-side indicators before calling Python.
            if let Some(ref mut bank) = self.indicator_bank {
                bank.update_all(price, volume);
            }
        }
        self.pre_enriched = false;

        if self.py_strategy.defines_generate_signals() {
            return self.py_strategy.generate_signals(data, context).await;
        }

        let ts = context.timestamp.timestamp_millis();
        match self.py_strategy.compute_all_signals(&[price], &[volume], &[ts], &[price], &[price], &[price]).await {
            Ok(Some(sigs)) if !sigs.is_empty() => Ok(i8_to_signal_vec(sigs[0], price)),
            Ok(_) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    fn state_snapshot(&self) -> Option<&StateStore> {
        None
    }

    async fn compute_all_signals(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> (Option<Vec<i8>>, Option<String>) {
        match self.py_strategy.compute_all_signals(prices, volumes, timestamps, opens, highs, lows).await {
            Ok(signals) => (signals, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_signals failed (PythonExecutor), falling back to per-tick: {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    async fn compute_all_signals_multi_venue(
        &mut self,
        venues: &HashMap<(String, String), VenueSeries>,
    ) -> (Option<Vec<i8>>, Option<String>) {
        match self.py_strategy.compute_all_signals_multi_venue(venues).await {
            Ok(signals) => (signals, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_signals_multi_venue failed (PythonExecutor): {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    fn accepts_multi_venue(&self) -> bool {
        self.py_strategy.accepts_multi_venue()
    }

    fn declares_pair_trade(&self) -> Option<crate::traits::PairSpec> {
        self.py_strategy.declares_pair_trade()
    }

    fn pair_spec_error(&self) -> Option<String> {
        self.py_strategy.pair_spec_error()
    }

    fn declares_basket_trade(&self) -> Option<crate::traits::BasketSpec> {
        self.py_strategy.declares_basket_trade()
    }

    fn basket_spec_error(&self) -> Option<String> {
        self.py_strategy.basket_spec_error()
    }

    fn compute_signals_accepts_ohlc(&self) -> bool {
        self.py_strategy.compute_signals_accepts_ohlc()
    }

    async fn compute_all_position_sizes(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<Vec<f64>>, Option<String>) {
        match self.py_strategy.compute_all_position_sizes(prices, volumes, timestamps).await {
            Ok(sizes) => (sizes, None),
            Err(e) => {
                log_warn!(STRATEGY_LOGGER, "compute_all_position_sizes failed (PythonExecutor), falling back to position_size_pct: {}", e);
                (None, Some(format!("{}", e)))
            }
        }
    }

    fn accepts_position_sizes(&self) -> bool {
        self.py_strategy.accepts_position_sizes()
    }

    fn defines_generate_signals(&self) -> bool {
        self.py_strategy.defines_generate_signals()
    }

    async fn compute_all_features(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> (Option<HashMap<String, Vec<f64>>>, Option<String>) {
        match self.py_strategy.compute_all_features(prices, volumes, timestamps).await {
            Ok(features) => (features, None),
            Err(e) => (None, Some(format!("{}", e))),
        }
    }

    fn strategy_name(&self) -> Option<String> {
        Some(self.py_strategy.name().to_string())
    }

    fn declared_max_leverage(&self) -> f64 {
        self.py_strategy.declared_max_leverage()
    }

    fn tier_description(&self) -> String {
        let ind_count = self.indicator_bank.as_ref().map_or(0, |b| b.len());
        if ind_count > 0 {
            format!("Tier 3 (full Python): no state_schema() — {} Rust indicators pre-computed", ind_count)
        } else {
            "Tier 3 (full Python): no state_schema() — all state management in Python".into()
        }
    }

    fn required_data_window(&self) -> usize {
        self.py_strategy.required_data_window()
    }
}

// ---------------------------------------------------------------------------
// Per-tick signal helpers
// ---------------------------------------------------------------------------

/// Convert a single bulk-signal i8 code to a `Vec<Signal>`.
///
/// Signal encoding matches `compute_signals()` contract: 1=BUY, -1=SELL, 2=CLOSE, 0=HOLD.
/// Used by `on_tick` implementations that call `compute_all_signals` with a 1-tick slice.
fn i8_to_signal_vec(code: i8, price: f64) -> Vec<Signal> {
    let reason = signal::SignalReason::TechnicalAnalysis("per-tick".into());
    match code {
        1  => vec![Signal::buy("tick", signal::SignalStrength::Medium, reason).with_price(price)],
        -1 => vec![Signal::sell("tick", signal::SignalStrength::Medium, reason).with_price(price)],
        2  => vec![Signal::close("tick", signal::SignalStrength::Medium, reason).with_price(price)],
        _  => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Executor construction — the graph extraction pipeline
// ---------------------------------------------------------------------------

/// Construct the best executor for a given Python strategy source.
///
/// This is the "graph extraction pipeline" described in Phase 1.11:
/// 1. Parse Python source → extract state_schema(), model(), features()
/// 2. Compile feature expressions
/// 3. Determine tier
/// 4. Construct appropriate executor
///
/// Returns `(executor, tier_analysis)`.
#[cfg(feature = "python")]
pub async fn build_executor(
    python_source: &str,
    parameters: HashMap<String, config::ParameterValue>,
) -> Result<(Box<dyn StrategyExecutor>, TierAnalysis), StrategyError> {
    use crate::state_store::StateType;

    let analysis = analyse_strategy(python_source);

    match analysis.tier {
        ExecutionTier::Tier0 => {
            let schema = analysis
                .state_schema
                .as_ref()
                .expect("Tier0 requires state_schema");
            let model_ref = analysis
                .model_ref
                .as_ref()
                .expect("Tier0 requires model");
            let state_store = StateStore::from_schema(schema);

            // Identify price/volume windows by name heuristic
            let (price_windows, volume_windows) = classify_windows(schema);

            let output_mapping = model_ref.output_mapping.clone().unwrap_or(
                OutputMapping::Threshold {
                    buy_threshold: 0.5,
                    sell_threshold: -0.5,
                },
            );

            let executor = NativeExecutor {
                state_store,
                schema: schema.clone(),
                compiled_features: analysis.compiled_features.clone().unwrap_or_default(),
                model_ref: model_ref.clone(),
                onnx_path: model_ref.path.clone(),
                output_mapping,
                price_windows,
                volume_windows,
            };

            log_info!(STRATEGY_LOGGER, "[EXECUTOR] Built {}", executor.tier_description());
            Ok((Box::new(executor), analysis))
        }
        ExecutionTier::Tier0_5 => {
            let schema = analysis
                .state_schema
                .as_ref()
                .expect("Tier0.5 requires state_schema");
            let state_store = StateStore::from_schema(schema);
            let (price_windows, volume_windows) = classify_windows(schema);

            // Extract indicators() declaration (opt-in pre-computation)
            let indicator_bank = extract_indicators(python_source)
                .map(IndicatorBank::new);

            // Build the compiled pricing formula
            let pricing_raw = crate::mm_pricing::extract_pricing(python_source)
                .ok_or_else(|| StrategyError::ConfigurationError(
                    "Tier0.5 requires pricing() — extraction failed".into()
                ))?;
            let formula = crate::mm_pricing::PricingFormula::compile(&pricing_raw)
                .map_err(|e| StrategyError::ConfigurationError(
                    format!("pricing() compile error: {}", e)
                ))?;

            let executor = crate::mm_pricing::MMPricingExecutor {
                state_store,
                schema: schema.clone(),
                formula,
                indicator_bank,
                price_windows,
                volume_windows,
                pre_enriched: false,
            };

            log_info!(STRATEGY_LOGGER, "[EXECUTOR] Built {}", executor.tier_description());
            Ok((Box::new(executor), analysis))
        }
        ExecutionTier::Tier1 | ExecutionTier::Tier2 => {
            let schema = analysis
                .state_schema
                .as_ref()
                .expect("Tier1/2 requires state_schema");
            let state_store = StateStore::from_schema(schema);
            let (price_windows, volume_windows) = classify_windows(schema);

            // Extract indicators() declaration (opt-in pre-computation)
            let indicator_bank = extract_indicators(python_source)
                .map(IndicatorBank::new);

            // Build PythonStrategy
            let mut py = crate::strategies::PythonStrategy::new(python_source.to_string());
            py.initialize(parameters)
                .await
                .map_err(|e| StrategyError::ConfigurationError(format!("init failed: {}", e)))?;

            let state_key_cache = StateKeyCache::from_schema(&schema);
            let indicator_prefix_cache = indicator_bank.as_ref()
                .map(|b| b.build_prefix_cache())
                .unwrap_or_default();
            let executor = HybridExecutor {
                state_store,
                schema: schema.clone(),
                py_strategy: Box::new(py),
                price_windows,
                volume_windows,
                indicator_bank,
                pre_enriched: false,
                state_key_cache,
                indicator_prefix_cache,
            };

            log_info!(STRATEGY_LOGGER, "[EXECUTOR] Built {}", executor.tier_description());
            Ok((Box::new(executor), analysis))
        }
        ExecutionTier::Tier3 => {
            // Extract indicators() declaration (opt-in pre-computation)
            let indicator_bank = extract_indicators(python_source)
                .map(IndicatorBank::new);

            // Build PythonStrategy
            let mut py = crate::strategies::PythonStrategy::new(python_source.to_string());
            py.initialize(parameters)
                .await
                .map_err(|e| StrategyError::ConfigurationError(format!("init failed: {}", e)))?;

            let indicator_prefix_cache = indicator_bank.as_ref()
                .map(|b| b.build_prefix_cache())
                .unwrap_or_default();
            let executor = PythonExecutor {
                py_strategy: Box::new(py),
                indicator_bank,
                pre_enriched: false,
                indicator_prefix_cache,
            };

            log_info!(STRATEGY_LOGGER, "[EXECUTOR] Built {}", executor.tier_description());
            Ok((Box::new(executor), analysis))
        }
    }
}

/// Build a Tier 3 (full Python) executor — used as fallback.
#[cfg(not(feature = "python"))]
pub async fn build_executor(
    _python_source: &str,
    _parameters: HashMap<String, config::ParameterValue>,
) -> Result<(Box<dyn StrategyExecutor>, TierAnalysis), StrategyError> {
    Err(StrategyError::ConfigurationError(
        "Python feature not enabled".into(),
    ))
}

/// Classify windows as price vs volume based on name heuristics.
#[allow(dead_code)]
fn classify_windows(schema: &StateSchema) -> (Vec<String>, Vec<String>) {
    use crate::state_store::StateType;

    let mut price_windows = Vec::new();
    let mut volume_windows = Vec::new();

    for field in &schema.fields {
        if matches!(field.state_type, StateType::FloatWindow { .. }) {
            let lower = field.name.to_lowercase();
            if lower.contains("price") || lower.contains("close") || lower.contains("mid") {
                price_windows.push(field.name.clone());
            } else if lower.contains("volume") || lower.contains("vol") || lower.contains("qty")
            {
                volume_windows.push(field.name.clone());
            }
            // else: user manages manually via update_state() in Python
        }
    }

    (price_windows, volume_windows)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_tier_display() {
        assert_eq!(format!("{}", ExecutionTier::Tier0), "Tier 0 (pure Rust + ONNX)");
        assert_eq!(format!("{}", ExecutionTier::Tier3), "Tier 3 (full Python)");
    }

    #[test]
    fn test_classify_windows() {
        let mut raw = HashMap::new();

        let mut w1 = HashMap::new();
        w1.insert("type".into(), serde_json::json!("window[float]"));
        w1.insert("capacity".into(), serde_json::json!(50));
        raw.insert("recent_prices".into(), w1);

        let mut w2 = HashMap::new();
        w2.insert("type".into(), serde_json::json!("window[float]"));
        w2.insert("capacity".into(), serde_json::json!(50));
        raw.insert("volume_history".into(), w2);

        let mut w3 = HashMap::new();
        w3.insert("type".into(), serde_json::json!("window[float]"));
        w3.insert("capacity".into(), serde_json::json!(50));
        raw.insert("custom_series".into(), w3);

        let schema = StateSchema::from_raw_dict(&raw).unwrap();
        let (price_w, vol_w) = classify_windows(&schema);

        assert!(price_w.contains(&"recent_prices".to_string()));
        assert!(vol_w.contains(&"volume_history".to_string()));
        assert!(!price_w.contains(&"custom_series".to_string()));
        assert!(!vol_w.contains(&"custom_series".to_string()));
    }

    #[test]
    fn test_output_mapping_threshold() {
        let executor = NativeExecutor {
            state_store: StateStore::from_schema(&StateSchema { fields: vec![] }),
            schema: StateSchema { fields: vec![] },
            compiled_features: vec![],
            model_ref: ModelRef {
                path: "model.onnx".into(),
                runtime: "onnx".into(),
                output_mapping: None,
            },
            onnx_path: "model.onnx".into(),
            output_mapping: OutputMapping::Threshold {
                buy_threshold: 0.5,
                sell_threshold: -0.5,
            },
            price_windows: vec![],
            volume_windows: vec![],
        };

        let ctx = crate::traits::StrategyContext {
            timestamp: chrono::Utc::now(),
            portfolio_state: Default::default(),
            price_history: vec![],
            volume_history: vec![],
            spread: None,
            market_session: crate::traits::MarketSession {
                is_open: true,
                session_type: crate::traits::SessionType::Regular,
                next_open: None,
                next_close: None,
            },
            orderbook_snapshot: None,
            custom_data: HashMap::new(),
            assets: HashMap::new(),
            tick_number: 1,
            elapsed_seconds: 0.0,
            iv_surface: None,
            futures_curve: None,
            funding_rates: None,
            portfolio_greeks: None,
            pool_state: None,
        };

        // Buy signal
        let signals = executor.map_output_to_signals(&[0.8], &ctx);
        assert_eq!(signals.len(), 1);
        assert!(matches!(signals[0].signal_type, signal::SignalType::Buy));

        // Sell signal
        let signals = executor.map_output_to_signals(&[-0.7], &ctx);
        assert_eq!(signals.len(), 1);
        assert!(matches!(signals[0].signal_type, signal::SignalType::Sell));

        // Hold
        let signals = executor.map_output_to_signals(&[0.1], &ctx);
        assert!(signals.is_empty());
    }

    #[test]
    fn native_executor_declared_max_leverage_defaults_to_no_cap() {
        // NativeExecutor has no underlying Python strategy that could declare
        // a leverage cap -- it must inherit the trait's own no-cap default
        // (infinity), never a numeric value that would silently constrain
        // leverage for pure Rust/ONNX strategies.
        let executor = NativeExecutor {
            state_store: StateStore::from_schema(&StateSchema { fields: vec![] }),
            schema: StateSchema { fields: vec![] },
            compiled_features: vec![],
            model_ref: ModelRef {
                path: "model.onnx".into(),
                runtime: "onnx".into(),
                output_mapping: None,
            },
            onnx_path: "model.onnx".into(),
            output_mapping: OutputMapping::Threshold {
                buy_threshold: 0.5,
                sell_threshold: -0.5,
            },
            price_windows: vec![],
            volume_windows: vec![],
        };
        assert!(executor.declared_max_leverage().is_infinite());
    }

    #[test]
    fn test_output_mapping_argmax() {
        let executor = NativeExecutor {
            state_store: StateStore::from_schema(&StateSchema { fields: vec![] }),
            schema: StateSchema { fields: vec![] },
            compiled_features: vec![],
            model_ref: ModelRef {
                path: "model.onnx".into(),
                runtime: "onnx".into(),
                output_mapping: None,
            },
            onnx_path: "model.onnx".into(),
            output_mapping: OutputMapping::Argmax {
                classes: vec!["hold".into(), "buy".into(), "sell".into()],
            },
            price_windows: vec![],
            volume_windows: vec![],
        };

        let ctx = crate::traits::StrategyContext {
            timestamp: chrono::Utc::now(),
            portfolio_state: Default::default(),
            price_history: vec![],
            volume_history: vec![],
            spread: None,
            market_session: crate::traits::MarketSession {
                is_open: true,
                session_type: crate::traits::SessionType::Regular,
                next_open: None,
                next_close: None,
            },
            orderbook_snapshot: None,
            custom_data: HashMap::new(),
            assets: HashMap::new(),
            tick_number: 1,
            elapsed_seconds: 0.0,
            iv_surface: None,
            futures_curve: None,
            funding_rates: None,
            portfolio_greeks: None,
            pool_state: None,
        };

        // Buy predicted
        let signals = executor.map_output_to_signals(&[0.1, 0.8, 0.1], &ctx);
        assert_eq!(signals.len(), 1);
        assert!(matches!(signals[0].signal_type, signal::SignalType::Buy));

        // Hold predicted
        let signals = executor.map_output_to_signals(&[0.9, 0.05, 0.05], &ctx);
        assert!(signals.is_empty());
    }
}
