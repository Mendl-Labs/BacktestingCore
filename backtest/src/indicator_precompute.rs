//! Pre-computes indicator arrays for all parameter values in a GA schema.
//!
//! For a momentum crossover with `fast_period: 5-50` and `slow_period: 20-200`,
//! this pre-computes 227 SMA arrays once (~50ms), then GA candidates look up
//! the arrays by parameter value — no per-candidate computation.

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use strategy::indicator_registry::{IndicatorSpec, IndicatorInstance, IndicatorOutput};
use genetic::dynamic_chromosome::ParameterSchema;
use genetic::dynamic_chromosome::ParamType;

use crate::candle_sim::CandleArrays;

/// Pre-computed indicator arrays keyed by (indicator_name, param_value).
/// For SMA with period 10: key = ("fast_sma", 10)
#[derive(Debug, Clone)]
pub struct PrecomputedIndicators {
    /// Single-value indicators: (name, period) → array of f64 (NaN where pending)
    pub single_arrays: HashMap<(String, u32), Vec<f64>>,
    /// Multi-value indicators: (name, period, suffix) → array of f64
    pub multi_arrays: HashMap<(String, u32, String), Vec<f64>>,
    /// Shared candle data
    pub candles: Arc<CandleArrays>,
}

/// Maps an indicator name declared in Python's `indicators()` to the chromosome
/// parameter that controls its period/param value.
#[derive(Debug, Clone)]
pub struct IndicatorParamBinding {
    pub indicator_name: String,
    pub spec_template: IndicatorSpec,
    pub param_key: String,
}

impl PrecomputedIndicators {
    /// Pre-compute all indicator variants for the given bindings and parameter ranges.
    ///
    /// For each binding (indicator + param_key), iterates over all integer values
    /// in the schema's range for that parameter and computes the full indicator array.
    pub fn from_candles_and_bindings(
        candles: Arc<CandleArrays>,
        schema: &ParameterSchema,
        bindings: &[IndicatorParamBinding],
    ) -> Self {
        let n = candles.len();
        let mut single_arrays = HashMap::new();
        let mut multi_arrays = HashMap::new();

        for binding in bindings {
            let param_def = schema.params.iter().find(|p| p.name == binding.param_key);
            let (min_val, max_val) = match param_def.map(|p| &p.param_type) {
                Some(ParamType::Int { min, max }) => (*min as u32, *max as u32),
                Some(ParamType::Float { min, max }) => (*min as u32, *max as u32),
                _ => continue,
            };

            for period in min_val..=max_val {
                let spec = substitute_period(&binding.spec_template, period as usize);
                let mut instance = match IndicatorInstance::from_spec(&spec) {
                    Some(i) => i,
                    None => continue,
                };

                let mut values = Vec::with_capacity(n);
                let mut multi_buffers: HashMap<String, Vec<f64>> = HashMap::new();

                for i in 0..n {
                    let price = candles.closes[i];
                    let volume = candles.volumes[i];
                    match instance.update(price, volume) {
                        IndicatorOutput::Pending => {
                            values.push(f64::NAN);
                        }
                        IndicatorOutput::Single(v) => {
                            values.push(v);
                        }
                        IndicatorOutput::Multi(pairs) => {
                            values.push(f64::NAN); // multi doesn't use single_arrays
                            for (suffix, v) in pairs {
                                multi_buffers.entry(suffix.to_string())
                                    .or_insert_with(|| {
                                        let mut buf = vec![f64::NAN; i];
                                        buf.reserve(n - i);
                                        buf
                                    })
                                    .push(v);
                            }
                        }
                    }
                }

                // Pad multi_buffers that started late
                for buf in multi_buffers.values_mut() {
                    while buf.len() < n {
                        buf.push(f64::NAN);
                    }
                }

                if !values.iter().all(|v| v.is_nan()) {
                    single_arrays.insert((binding.indicator_name.clone(), period), values);
                }
                for (suffix, buf) in multi_buffers {
                    if !buf.iter().all(|v| v.is_nan()) {
                        multi_arrays.insert(
                            (binding.indicator_name.clone(), period, suffix),
                            buf,
                        );
                    }
                }
            }
        }

        Self { single_arrays, multi_arrays, candles }
    }

    /// Look up a single-value indicator array by name and period.
    pub fn get_single(&self, name: &str, period: u32) -> Option<&[f64]> {
        self.single_arrays.get(&(name.to_string(), period)).map(|v| v.as_slice())
    }

    /// Look up a multi-value indicator array by name, period, and suffix.
    pub fn get_multi(&self, name: &str, period: u32, suffix: &str) -> Option<&[f64]> {
        self.multi_arrays.get(&(name.to_string(), period, suffix.to_string())).map(|v| v.as_slice())
    }

    /// Convert to a serializable form (excludes candle data).
    pub fn to_serializable(&self) -> SerializedPrecomputed {
        let single_arrays = self.single_arrays.iter()
            .map(|((name, period), arr)| {
                (format!("{}:{}", name, period), arr.clone())
            })
            .collect();
        let multi_arrays = self.multi_arrays.iter()
            .map(|((name, period, suffix), arr)| {
                (format!("{}:{}:{}", name, period, suffix), arr.clone())
            })
            .collect();
        SerializedPrecomputed { single_arrays, multi_arrays }
    }

    /// Reconstruct from serialized form + candle data.
    pub fn from_serializable(s: SerializedPrecomputed, candles: Arc<CandleArrays>) -> Self {
        let single_arrays = s.single_arrays.into_iter()
            .filter_map(|(key, arr)| {
                let parts: Vec<&str> = key.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let period: u32 = parts[1].parse().ok()?;
                    Some(((parts[0].to_string(), period), arr))
                } else {
                    None
                }
            })
            .collect();
        let multi_arrays = s.multi_arrays.into_iter()
            .filter_map(|(key, arr)| {
                let parts: Vec<&str> = key.splitn(3, ':').collect();
                if parts.len() == 3 {
                    let period: u32 = parts[1].parse().ok()?;
                    Some(((parts[0].to_string(), period, parts[2].to_string()), arr))
                } else {
                    None
                }
            })
            .collect();
        Self { single_arrays, multi_arrays, candles }
    }
}

/// Serializable snapshot of precomputed indicators (no Arc<CandleArrays>).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPrecomputed {
    pub single_arrays: HashMap<String, Vec<f64>>,
    pub multi_arrays: HashMap<String, Vec<f64>>,
}

/// Create a new IndicatorSpec with the given period substituted in.
fn substitute_period(template: &IndicatorSpec, period: usize) -> IndicatorSpec {
    match template {
        IndicatorSpec::Sma { .. } => IndicatorSpec::Sma { period },
        IndicatorSpec::Ema { .. } => IndicatorSpec::Ema { period },
        IndicatorSpec::Rsi { .. } => IndicatorSpec::Rsi { period },
        IndicatorSpec::Macd { fast, slow, signal } => {
            // For MACD, period substitutes the 'fast' param; slow/signal scale proportionally
            let ratio_slow = *slow as f64 / *fast as f64;
            let ratio_sig = *signal as f64 / *fast as f64;
            IndicatorSpec::Macd {
                fast: period,
                slow: (period as f64 * ratio_slow).round() as usize,
                signal: (period as f64 * ratio_sig).round() as usize,
            }
        }
        IndicatorSpec::BollingerBands { std_dev, .. } => {
            IndicatorSpec::BollingerBands { period, std_dev: *std_dev }
        }
        IndicatorSpec::Atr { .. } => IndicatorSpec::Atr { period },
        IndicatorSpec::Stddev { .. } => IndicatorSpec::Stddev { period },
        IndicatorSpec::Vwap { .. } => IndicatorSpec::Vwap { lookback: Some(period) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_sim::CandleArrays;

    #[test]
    fn precompute_sma_range() {
        let closes: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let candles = Arc::new(CandleArrays {
            opens: closes.clone(),
            highs: closes.iter().map(|c| c + 1.0).collect(),
            lows: closes.iter().map(|c| c - 1.0).collect(),
            closes: closes.clone(),
            volumes: vec![100.0; 100],
            timestamps_ms: (0..100).map(|i| i * 86_400_000).collect(),
        });

        let schema = ParameterSchema {
            params: vec![
                genetic::dynamic_chromosome::ParameterDef {
                    name: "fast_period".into(),
                    param_type: ParamType::Int { min: 5, max: 10 },
                    default: genetic::dynamic_chromosome::Gene::Int(7),
                },
            ],
        };

        let bindings = vec![IndicatorParamBinding {
            indicator_name: "fast_sma".into(),
            spec_template: IndicatorSpec::Sma { period: 0 },
            param_key: "fast_period".into(),
        }];

        let precomputed = PrecomputedIndicators::from_candles_and_bindings(
            candles, &schema, &bindings,
        );

        // Should have SMA arrays for periods 5..=10
        for period in 5..=10u32 {
            let arr = precomputed.get_single("fast_sma", period);
            assert!(arr.is_some(), "Missing SMA for period {}", period);
            let arr = arr.unwrap();
            assert_eq!(arr.len(), 100);
            // First (period-1) values are NaN (warmup)
            assert!(arr[period as usize - 2].is_nan());
            // Value at index=period-1 should be the average of 1..=period
            let expected = (1..=period).sum::<u32>() as f64 / period as f64;
            assert!((arr[period as usize - 1] - expected).abs() < 1e-10);
        }
    }
}
