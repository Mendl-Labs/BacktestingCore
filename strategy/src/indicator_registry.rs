//! Indicator registry for Rust-side pre-computation of technical indicators.
//!
//! Strategies that declare an `indicators()` method get their indicators
//! computed in Rust and injected into the Python context, eliminating
//! per-tick Python indicator math.
//!
//! ## Usage
//!
//! Python strategies declare which indicators they need:
//! ```python
//! def indicators(self):
//!     return {
//!         "fast_sma": ("sma", {"period": 10}),
//!         "slow_sma": ("sma", {"period": 50}),
//!         "rsi_14":   ("rsi", {"period": 14}),
//!     }
//! ```
//!
//! The Rust engine:
//! 1. Extracts this once at job start
//! 2. Creates an `IndicatorBank` with stateful Rust indicators
//! 3. Each tick: `bank.update_all(price, volume)` (pure Rust, no Python)
//! 4. Injects pre-computed values into the context dict before calling Python

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::indicators::{
    SimpleMovingAverage, ExponentialMovingAverage, RelativeStrengthIndex,
    MACD, BollingerBands, AverageTrueRange,
};

// ---------------------------------------------------------------------------
// IndicatorSpec — describes what indicator to create
// ---------------------------------------------------------------------------

/// Specification for a single indicator, parsed from Python's `indicators()` dict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorSpec {
    Sma { period: usize },
    Ema { period: usize },
    Rsi { period: usize },
    Macd { fast: usize, slow: usize, signal: usize },
    BollingerBands { period: usize, std_dev: f64 },
    Atr { period: usize },
    Stddev { period: usize },
    Vwap { lookback: Option<usize> },
}

// ---------------------------------------------------------------------------
// IndicatorOutput — the computed value(s)
// ---------------------------------------------------------------------------

/// Output from a single indicator computation. Scalar indicators produce `Single`,
/// multi-value indicators (MACD, Bollinger) produce `Multi`.
#[derive(Debug, Clone)]
pub enum IndicatorOutput {
    /// Not yet ready (warmup period not met).
    Pending,
    /// Single scalar value (SMA, EMA, RSI, ATR, Stddev).
    Single(f64),
    /// Named multi-values (MACD → macd/signal/histogram, Bollinger → upper/middle/lower).
    Multi(Vec<(&'static str, f64)>),
}

// ---------------------------------------------------------------------------
// IndicatorInstance — wraps a stateful Rust indicator
// ---------------------------------------------------------------------------

/// A live indicator instance with its spec and stateful computation.
pub struct IndicatorInstance {
    pub spec: IndicatorSpec,
    inner: IndicatorInner,
}

enum IndicatorInner {
    Sma(SimpleMovingAverage),
    Ema(ExponentialMovingAverage),
    Rsi(RelativeStrengthIndex),
    Macd(MACD),
    Bollinger(BollingerBands),
    Atr(AverageTrueRange),
    Stddev(StddevIndicator),
    Vwap(VwapIndicator),
}

/// Simple standard deviation indicator using a sliding window.
/// Uses running sum / sum-of-squares for O(1) computation per tick.
struct StddevIndicator {
    period: usize,
    values: std::collections::VecDeque<f64>,
    running_sum: f64,
    running_sum_sq: f64,
}

impl StddevIndicator {
    fn new(period: usize) -> Self {
        Self {
            period,
            values: std::collections::VecDeque::with_capacity(period),
            running_sum: 0.0,
            running_sum_sq: 0.0,
        }
    }

    fn next(&mut self, price: f64) -> Option<f64> {
        if self.values.len() == self.period {
            if let Some(evicted) = self.values.pop_front() {
                self.running_sum -= evicted;
                self.running_sum_sq -= evicted * evicted;
            }
        }
        self.values.push_back(price);
        self.running_sum += price;
        self.running_sum_sq += price * price;

        if self.values.len() == self.period {
            let n = self.period as f64;
            let mean = self.running_sum / n;
            let variance = (self.running_sum_sq / n) - (mean * mean);
            Some(if variance <= 0.0 { 0.0 } else { variance.sqrt() })
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.values.clear();
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
    }
}

/// VWAP indicator accumulating price*volume / volume.
/// Uses running sums for O(1) computation per tick.
struct VwapIndicator {
    lookback: Option<usize>,
    prices: std::collections::VecDeque<f64>,
    volumes: std::collections::VecDeque<f64>,
    running_pv_sum: f64,
    running_vol_sum: f64,
}

impl VwapIndicator {
    fn new(lookback: Option<usize>) -> Self {
        let cap = lookback.unwrap_or(1024);
        Self {
            lookback,
            prices: std::collections::VecDeque::with_capacity(cap),
            volumes: std::collections::VecDeque::with_capacity(cap),
            running_pv_sum: 0.0,
            running_vol_sum: 0.0,
        }
    }

    fn next(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.prices.push_back(price);
        self.volumes.push_back(volume);
        self.running_pv_sum += price * volume;
        self.running_vol_sum += volume;
        if let Some(lb) = self.lookback {
            while self.prices.len() > lb {
                let evicted_p = self.prices.pop_front().unwrap_or(0.0);
                let evicted_v = self.volumes.pop_front().unwrap_or(0.0);
                self.running_pv_sum -= evicted_p * evicted_v;
                self.running_vol_sum -= evicted_v;
            }
        }
        if self.running_vol_sum == 0.0 {
            return None;
        }
        Some(self.running_pv_sum / self.running_vol_sum)
    }

    fn reset(&mut self) {
        self.prices.clear();
        self.volumes.clear();
        self.running_pv_sum = 0.0;
        self.running_vol_sum = 0.0;
    }
}

impl IndicatorInstance {
    /// Create an indicator instance from a spec.
    /// Returns `None` if the spec has invalid parameters.
    pub fn from_spec(spec: &IndicatorSpec) -> Option<Self> {
        let inner = match spec {
            IndicatorSpec::Sma { period } => {
                IndicatorInner::Sma(SimpleMovingAverage::new(*period).ok()?)
            }
            IndicatorSpec::Ema { period } => {
                IndicatorInner::Ema(ExponentialMovingAverage::new(*period).ok()?)
            }
            IndicatorSpec::Rsi { period } => {
                IndicatorInner::Rsi(RelativeStrengthIndex::new(*period).ok()?)
            }
            IndicatorSpec::Macd { fast, slow, signal } => {
                IndicatorInner::Macd(MACD::with_params(*fast, *slow, *signal).ok()?)
            }
            IndicatorSpec::BollingerBands { period, std_dev } => {
                IndicatorInner::Bollinger(BollingerBands::with_params(*period, *std_dev).ok()?)
            }
            IndicatorSpec::Atr { period } => {
                IndicatorInner::Atr(AverageTrueRange::with_period(*period).ok()?)
            }
            IndicatorSpec::Stddev { period } => {
                if *period == 0 { return None; }
                IndicatorInner::Stddev(StddevIndicator::new(*period))
            }
            IndicatorSpec::Vwap { lookback } => {
                IndicatorInner::Vwap(VwapIndicator::new(*lookback))
            }
        };
        Some(Self { spec: spec.clone(), inner })
    }

    /// Feed a new price (and volume) and compute the indicator value.
    pub fn update(&mut self, price: f64, volume: f64) -> IndicatorOutput {
        match &mut self.inner {
            IndicatorInner::Sma(ind) => match ind.next(price) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Ema(ind) => match ind.next(price) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Rsi(ind) => match ind.next(price) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Macd(ind) => match ind.next(price) {
                Some(r) => IndicatorOutput::Multi(vec![
                    ("macd", r.macd),
                    ("signal", r.signal.unwrap_or(0.0)),
                    ("histogram", r.histogram.unwrap_or(0.0)),
                ]),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Bollinger(ind) => match ind.next(price) {
                Some(r) => IndicatorOutput::Multi(vec![
                    ("upper", r.upper),
                    ("middle", r.middle),
                    ("lower", r.lower),
                    ("bandwidth", r.bandwidth),
                ]),
                None => IndicatorOutput::Pending,
            },
            // ATR uses simplified price-only approximation (matching Python SDK context.atr())
            IndicatorInner::Atr(ind) => match ind.next(price, price, price) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Stddev(ind) => match ind.next(price) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
            IndicatorInner::Vwap(ind) => match ind.next(price, volume) {
                Some(v) => IndicatorOutput::Single(v),
                None => IndicatorOutput::Pending,
            },
        }
    }

    /// Reset indicator state (for new simulation run).
    pub fn reset(&mut self) {
        match &mut self.inner {
            IndicatorInner::Sma(ind) => ind.reset(),
            IndicatorInner::Ema(ind) => ind.reset(),
            IndicatorInner::Rsi(ind) => ind.reset(),
            IndicatorInner::Macd(ind) => ind.reset(),
            IndicatorInner::Bollinger(ind) => ind.reset(),
            IndicatorInner::Atr(ind) => ind.reset(),
            IndicatorInner::Stddev(ind) => ind.reset(),
            IndicatorInner::Vwap(ind) => ind.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// IndicatorBank — collection of named indicator instances
// ---------------------------------------------------------------------------

/// Pre-computed output key strings for an indicator, avoiding per-tick format!()/clone().
enum IndicatorKeys {
    /// Single-value indicator: one pre-computed key string.
    Single(String),
    /// Multi-value indicator: pre-computed "{name}.{suffix}" for each known suffix.
    /// MACD → ["x.macd", "x.signal", "x.histogram"], Bollinger → ["x.upper", "x.middle", "x.lower", "x.bandwidth"]
    Multi(Vec<String>),
}

/// A bank of indicator instances, keyed by user-chosen names.
/// Created once at job start, updated every tick in pure Rust.
pub struct IndicatorBank {
    indicators: Vec<(IndicatorKeys, IndicatorInstance)>,
    /// Cached output map, updated each tick.
    values: HashMap<String, f64>,
}

impl IndicatorBank {
    /// Create a bank from a map of {name → spec}.
    /// Silently skips indicators with invalid parameters.
    pub fn new(specs: HashMap<String, IndicatorSpec>) -> Self {
        let mut indicators = Vec::with_capacity(specs.len());
        for (name, spec) in specs {
            if let Some(instance) = IndicatorInstance::from_spec(&spec) {
                // Pre-compute key strings based on indicator type.
                let keys = match &spec {
                    IndicatorSpec::Macd { .. } => IndicatorKeys::Multi(vec![
                        format!("{}.macd", name),
                        format!("{}.signal", name),
                        format!("{}.histogram", name),
                    ]),
                    IndicatorSpec::BollingerBands { .. } => IndicatorKeys::Multi(vec![
                        format!("{}.upper", name),
                        format!("{}.middle", name),
                        format!("{}.lower", name),
                        format!("{}.bandwidth", name),
                    ]),
                    _ => IndicatorKeys::Single(name),
                };
                indicators.push((keys, instance));
            }
        }
        let capacity = indicators.len() * 4; // multi-value indicators expand
        Self {
            indicators,
            values: HashMap::with_capacity(capacity),
        }
    }

    /// Update all indicators with the current tick's price and volume.
    /// This is called once per tick, entirely in Rust — no Python, no GIL.
    pub fn update_all(&mut self, price: f64, volume: f64) {
        self.values.clear();
        for (keys, instance) in &mut self.indicators {
            match instance.update(price, volume) {
                IndicatorOutput::Pending => {}
                IndicatorOutput::Single(v) => {
                    if let IndicatorKeys::Single(ref k) = keys {
                        self.values.insert(k.clone(), v);
                    }
                }
                IndicatorOutput::Multi(pairs) => {
                    if let IndicatorKeys::Multi(ref cached_keys) = keys {
                        // cached_keys and pairs are in the same order (set at construction).
                        for (cached_key, (_suffix, v)) in cached_keys.iter().zip(pairs.iter()) {
                            self.values.insert(cached_key.clone(), *v);
                        }
                    }
                }
            }
        }
    }

    /// Get the current flat map of indicator values.
    /// Keys: `"fast_sma"` for single-value, `"my_macd.signal"` for multi-value.
    pub fn values(&self) -> &HashMap<String, f64> {
        &self.values
    }

    /// Inject current indicator values into `custom_data` with an `"ind."` prefix.
    /// Uses a pre-built prefix cache to avoid `format!("ind.{}", k)` per tick.
    /// Uses `get_mut` to update existing entries in-place, avoiding String::clone()
    /// after the first tick.
    pub fn inject_into(&self, custom_data: &mut HashMap<String, f64>, prefix_cache: &HashMap<String, String>) {
        for (k, v) in &self.values {
            if let Some(prefixed) = prefix_cache.get(k) {
                if let Some(slot) = custom_data.get_mut(prefixed.as_str()) {
                    *slot = *v;
                } else {
                    custom_data.insert(prefixed.clone(), *v);
                }
            } else {
                // Fallback for any key not in cache (shouldn't happen in practice).
                custom_data.insert(format!("ind.{}", k), *v);
            }
        }
    }

    /// Build a prefix cache mapping each indicator key to `"ind.{key}"`.
    /// Call once at startup, then pass to `inject_into()` each tick.
    pub fn build_prefix_cache(&self) -> HashMap<String, String> {
        let mut cache = HashMap::with_capacity(self.indicators.len() * 4);
        for (keys, _) in &self.indicators {
            match keys {
                IndicatorKeys::Single(ref name) => {
                    cache.insert(name.clone(), format!("ind.{}", name));
                }
                IndicatorKeys::Multi(ref cached_keys) => {
                    for k in cached_keys {
                        cache.insert(k.clone(), format!("ind.{}", k));
                    }
                }
            }
        }
        cache
    }

    /// Reset all indicators (for a new simulation run).
    pub fn reset(&mut self) {
        self.values.clear();
        for (_, instance) in &mut self.indicators {
            instance.reset();
        }
    }

    /// Number of indicator instances.
    pub fn len(&self) -> usize {
        self.indicators.len()
    }

    /// Whether the bank has no indicators.
    pub fn is_empty(&self) -> bool {
        self.indicators.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Python extraction — parse indicators() from strategy source
// ---------------------------------------------------------------------------

/// Extract the `indicators()` declaration from a Python strategy.
///
/// Returns `None` if the strategy doesn't define `indicators()`.
/// Returns `Some(map)` with {name → IndicatorSpec} on success.
#[cfg(feature = "python")]
pub fn extract_indicators(python_source: &str) -> Option<HashMap<String, IndicatorSpec>> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};

    Python::with_gil(|py| {
        // Inject SDK so `from trading_platform import ...` works
        crate::strategies::python_strategy::inject_sdk(py).ok()?;

        // Unique name per call -- a fixed literal here let a previous job's
        // stale module (and its Strategy class) leak into this compilation
        // via a shared sys.modules entry; see python_strategy.rs's
        // unique_module_name doc comment for the full incident writeup.
        let module = PyModule::from_code_bound(
            py, python_source, "strategy.py", &crate::strategies::python_strategy::unique_module_name(),
        ).ok()?;

        let strategy_class = module.getattr("Strategy").ok()?;
        let instance = strategy_class.call0().ok()?;

        if !instance.hasattr("indicators").unwrap_or(false) {
            return None;
        }

        let result = instance.call_method0("indicators").ok()?;
        let dict = result.downcast::<PyDict>().ok()?;

        let mut specs: HashMap<String, IndicatorSpec> = HashMap::new();

        for (key, value) in dict.iter() {
            let name: String = key.extract().ok()?;

            // Expected format: ("indicator_type", {"param": value, ...})
            let tuple = value.downcast::<PyTuple>().ok()?;
            if tuple.len() < 2 { continue; }

            let indicator_type: String = tuple.get_item(0).ok()?.extract().ok()?;
            let params = tuple.get_item(1).ok()?;
            let params_dict = params.downcast::<PyDict>().ok()?;

            let spec = parse_indicator_spec(&indicator_type, params_dict)?;
            specs.insert(name, spec);
        }

        if specs.is_empty() {
            None
        } else {
            crate::log_info!(
                crate::logging_facade::STRATEGY_LOGGER,
                "[INDICATOR_REGISTRY] Extracted indicators(): {} indicators: {:?}",
                specs.len(),
                specs.keys().collect::<Vec<_>>()
            );
            Some(specs)
        }
    })
}

/// Stub for non-python builds.
#[cfg(not(feature = "python"))]
pub fn extract_indicators(_python_source: &str) -> Option<HashMap<String, IndicatorSpec>> {
    None
}

/// Parse a single indicator spec from Python dict params.
#[cfg(feature = "python")]
fn parse_indicator_spec(
    indicator_type: &str,
    params: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> Option<IndicatorSpec> {
    use pyo3::types::PyAnyMethods;
    use pyo3::types::PyDictMethods;

    let get_usize = |key: &str, default: usize| -> usize {
        params.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(default)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        params.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(default)
    };

    match indicator_type.to_lowercase().as_str() {
        "sma" => Some(IndicatorSpec::Sma { period: get_usize("period", 20) }),
        "ema" => Some(IndicatorSpec::Ema { period: get_usize("period", 20) }),
        "rsi" => Some(IndicatorSpec::Rsi { period: get_usize("period", 14) }),
        "macd" => Some(IndicatorSpec::Macd {
            fast: get_usize("fast", 12),
            slow: get_usize("slow", 26),
            signal: get_usize("signal", 9),
        }),
        "bollinger" | "bollinger_bands" | "bb" => Some(IndicatorSpec::BollingerBands {
            period: get_usize("period", 20),
            std_dev: get_f64("std_dev", 2.0),
        }),
        "atr" => Some(IndicatorSpec::Atr { period: get_usize("period", 14) }),
        "stddev" | "std" => Some(IndicatorSpec::Stddev { period: get_usize("period", 20) }),
        "vwap" => {
            let lookback = params.get_item("lookback").ok().flatten()
                .and_then(|v| v.extract::<usize>().ok());
            Some(IndicatorSpec::Vwap { lookback })
        }
        _ => {
            crate::log_warn!(
                crate::logging_facade::STRATEGY_LOGGER,
                "[INDICATOR_REGISTRY] Unknown indicator type: '{}' — skipping",
                indicator_type
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_bank() {
        let mut specs = HashMap::new();
        specs.insert("sma_3".into(), IndicatorSpec::Sma { period: 3 });
        let mut bank = IndicatorBank::new(specs);

        bank.update_all(1.0, 100.0);
        assert!(bank.values().is_empty()); // warmup

        bank.update_all(2.0, 100.0);
        assert!(bank.values().is_empty()); // still warming

        bank.update_all(3.0, 100.0);
        assert!((bank.values()["sma_3"] - 2.0).abs() < 1e-10);

        bank.update_all(4.0, 100.0);
        assert!((bank.values()["sma_3"] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_bank() {
        let mut specs = HashMap::new();
        specs.insert("ema_3".into(), IndicatorSpec::Ema { period: 3 });
        let mut bank = IndicatorBank::new(specs);

        // EMA initializes on first price
        bank.update_all(10.0, 100.0);
        assert!((bank.values()["ema_3"] - 10.0).abs() < 1e-10);

        // multiplier = 2/(3+1) = 0.5
        bank.update_all(20.0, 100.0);
        // 20*0.5 + 10*0.5 = 15.0
        assert!((bank.values()["ema_3"] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_bank() {
        let mut specs = HashMap::new();
        specs.insert("rsi_3".into(), IndicatorSpec::Rsi { period: 3 });
        let mut bank = IndicatorBank::new(specs);

        // RSI needs period+1 prices to produce a value
        bank.update_all(100.0, 1.0); // first price, no output
        assert!(bank.values().is_empty());

        bank.update_all(101.0, 1.0); // gain
        bank.update_all(102.0, 1.0); // gain
        bank.update_all(103.0, 1.0); // gain → 3 gains, 0 losses → RSI=100
        assert!((bank.values()["rsi_3"] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_macd_bank() {
        let mut specs = HashMap::new();
        specs.insert("my_macd".into(), IndicatorSpec::Macd { fast: 3, slow: 5, signal: 3 });
        let mut bank = IndicatorBank::new(specs);

        // Feed enough prices for EMAs to produce values
        for i in 0..10 {
            bank.update_all(100.0 + i as f64, 100.0);
        }

        // MACD should have multi-value output
        assert!(bank.values().contains_key("my_macd.macd"));
        assert!(bank.values().contains_key("my_macd.signal"));
        assert!(bank.values().contains_key("my_macd.histogram"));
    }

    #[test]
    fn test_bollinger_bank() {
        let mut specs = HashMap::new();
        specs.insert("bb".into(), IndicatorSpec::BollingerBands { period: 3, std_dev: 2.0 });
        let mut bank = IndicatorBank::new(specs);

        bank.update_all(10.0, 1.0);
        bank.update_all(10.0, 1.0);
        bank.update_all(10.0, 1.0);

        // All same price → std_dev=0 → upper=middle=lower
        assert!((bank.values()["bb.middle"] - 10.0).abs() < 1e-10);
        assert!((bank.values()["bb.upper"] - 10.0).abs() < 1e-10);
        assert!((bank.values()["bb.lower"] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_bank() {
        let mut specs = HashMap::new();
        specs.insert("std_3".into(), IndicatorSpec::Stddev { period: 3 });
        let mut bank = IndicatorBank::new(specs);

        bank.update_all(1.0, 1.0);
        bank.update_all(2.0, 1.0);
        bank.update_all(3.0, 1.0);

        // mean=2, variance=((1-2)²+(2-2)²+(3-2)²)/3 = 2/3, std=sqrt(2/3)
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((bank.values()["std_3"] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vwap_bank() {
        let mut specs = HashMap::new();
        specs.insert("vwap".into(), IndicatorSpec::Vwap { lookback: Some(3) });
        let mut bank = IndicatorBank::new(specs);

        bank.update_all(100.0, 10.0);
        // VWAP = 100*10/10 = 100
        assert!((bank.values()["vwap"] - 100.0).abs() < 1e-10);

        bank.update_all(200.0, 20.0);
        // VWAP = (100*10 + 200*20) / 30 = 5000/30 ≈ 166.667
        assert!((bank.values()["vwap"] - 166.6666666667).abs() < 1e-6);
    }

    #[test]
    fn test_bank_reset() {
        let mut specs = HashMap::new();
        specs.insert("sma_2".into(), IndicatorSpec::Sma { period: 2 });
        let mut bank = IndicatorBank::new(specs);

        bank.update_all(10.0, 1.0);
        bank.update_all(20.0, 1.0);
        assert_eq!(bank.values().len(), 1);

        bank.reset();
        assert!(bank.values().is_empty());

        // After reset, needs warmup again
        bank.update_all(30.0, 1.0);
        assert!(bank.values().is_empty());

        bank.update_all(40.0, 1.0);
        assert!((bank.values()["sma_2"] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_spec_skipped() {
        let mut specs = HashMap::new();
        specs.insert("bad_sma".into(), IndicatorSpec::Sma { period: 0 }); // invalid
        specs.insert("good_sma".into(), IndicatorSpec::Sma { period: 5 });
        let bank = IndicatorBank::new(specs);

        // Invalid spec silently skipped
        assert_eq!(bank.len(), 1);
    }
}
