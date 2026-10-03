//! W2.1 / G1: the causality-by-truncation gate against real Python strategies through PyO3.
//!
//! Run with: `cargo test -p backtest --features python -- causality_truncation`
//! (needs a linkable Python; on the ARM64 Windows box only `cargo check --features python --tests` works, so these
//! run in CI / WSL.) The pure-Rust sampler tests live in `backtest/src/causality_gate.rs`.

#![cfg(feature = "python")]

use std::collections::HashMap;

use backtest::causality_gate::{
    causality_truncation_check, sample_truncation_bars, CausalityGateConfig, LookaheadScanMode,
};
use chrono::{TimeZone, Utc};
use dataloader::{Candle, MarketData};

/// The benchmark's defect class, twice over: a "20-bar SMA" anchored to the END of the array (so every bar's signal
/// depends on how many bars follow it) and the last available bar treated as a decision point.
const END_OF_ARRAY_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "EndOfArrayDefect"

    def compute_signals(self, prices, volumes, timestamps):
        n = len(prices)
        sig = np.zeros(n, dtype=np.int8)
        # DEFECT 1: the window is anchored to the end of the array, not to each bar.
        sma_end = prices[-20:].mean() if n >= 20 else prices.mean()
        sig[prices > sma_end] = 1
        sig[prices <= sma_end] = -1
        # DEFECT 2: the last available bar is a decision point.
        sig[n - 1] = 2
        return sig
"#;

/// A causal 20-bar SMA cross: bar t's signal uses bars <= t only. Prefix sums are sequential, so truncation cannot
/// change a single bit of any earlier value.
const CAUSAL_SMA_CROSS_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "CausalSmaCross"

    def compute_signals(self, prices, volumes, timestamps):
        n = len(prices)
        w = 20
        sig = np.zeros(n, dtype=np.int8)
        if n <= w:
            return sig
        cs = np.concatenate(([0.0], np.cumsum(prices)))
        # sma[i] = mean of prices[i-w+1 ..= i], defined for i >= w-1
        sma = (cs[w:] - cs[:-w]) / w
        for i in range(w, n):
            above = prices[i] > sma[i - w + 1]
            prev_above = prices[i - 1] > sma[i - w]
            if above and not prev_above:
                sig[i] = 1
            elif prev_above and not above:
                sig[i] = -1
        return sig
"#;

fn synthetic_candles(n: usize) -> Vec<MarketData> {
    let mut price = 100.0_f64;
    (0..n)
        .map(|i| {
            let trend = if (i / 60) % 2 == 0 { 0.08 } else { -0.07 };
            let noise = ((i * 7 + 3) % 11) as f64 * 0.05 - 0.25;
            price = (price + trend + noise).max(10.0);
            MarketData::Candle(Candle {
                timestamp: Utc
                    .timestamp_opt(1_700_000_000 + i as i64 * 3600, 0)
                    .unwrap(),
                symbol: "BTC/USD".into(),
                exchange: "test".into(),
                open: price,
                high: price * 1.002,
                low: price * 0.998,
                close: price,
                volume: 1000.0,
                trade_count: 1,
            })
        })
        .collect()
}

fn cfg() -> CausalityGateConfig {
    CausalityGateConfig {
        mode: LookaheadScanMode::Sampled,
        ..CausalityGateConfig::default()
    }
}

#[tokio::test]
async fn end_of_array_defect_fails_at_the_last_decision_bar_and_at_sampled_interior_bars() {
    let data = synthetic_candles(600);
    let r = causality_truncation_check(&data, END_OF_ARRAY_STRATEGY, HashMap::new(), &cfg())
        .await
        .unwrap();
    assert!(r.applicable && !r.nondeterministic);
    assert!(!r.passed(), "{}", r.message);
    assert_eq!(r.bars_checked, r.sampled_bars.len());
    // Bar n-2 is the last bar at which "one later bar exists": DEFECT 2 marks it 2 when the series ends there and
    // DEFECT 1 / the full run give 1 or -1. It is always sampled and must always be a violation.
    let n = data.len();
    assert!(r.sampled_bars.contains(&(n - 1)) && r.sampled_bars.contains(&(n - 2)));
    let at_n2 = r
        .violations
        .iter()
        .find(|v| v.bar_index == n - 2)
        .expect("violation at n-2");
    assert_eq!(at_n2.truncated_signal, 2.0);
    assert_ne!(at_n2.full_signal, 2.0);
    // Every interior sampled bar is a violation (DEFECT 2 alone guarantees it: the truncated run ends at t).
    let interior: Vec<usize> = r
        .sampled_bars
        .iter()
        .copied()
        .filter(|&b| b < n - 1)
        .collect();
    assert!(!interior.is_empty());
    for b in &interior {
        assert!(
            r.violations.iter().any(|v| v.bar_index == *b),
            "bar {b} should be a violation"
        );
    }
    // Truncating at the final bar is the full series itself, so n-1 can never be a violation by construction.
    assert!(!r.violations.iter().any(|v| v.bar_index == n - 1));
    assert!(r
        .violations
        .iter()
        .all(|v| v.asset == "BTC/USD" && r.sampled_bars.contains(&v.bar_index)));
    assert!(r.message.contains("end-of-array"), "{}", r.message);
}

#[tokio::test]
async fn a_causal_sma_cross_passes_at_every_sampled_bar() {
    let data = synthetic_candles(600);
    let r = causality_truncation_check(&data, CAUSAL_SMA_CROSS_STRATEGY, HashMap::new(), &cfg())
        .await
        .unwrap();
    assert!(r.applicable && !r.nondeterministic);
    assert!(r.passed(), "{:?}", r.violations);
    assert!(r.violations.is_empty());
    assert_eq!(r.bars_checked, r.sampled_bars.len());
    assert!(r.sampled_bars.contains(&599));
    assert!(!r.budget_exhausted);
}

#[tokio::test]
async fn the_sampled_set_includes_the_last_bar_and_is_deterministic_across_runs() {
    let data = synthetic_candles(600);
    let a = causality_truncation_check(&data, CAUSAL_SMA_CROSS_STRATEGY, HashMap::new(), &cfg())
        .await
        .unwrap();
    let b = causality_truncation_check(&data, CAUSAL_SMA_CROSS_STRATEGY, HashMap::new(), &cfg())
        .await
        .unwrap();
    assert_eq!(a.sampled_bars, b.sampled_bars);
    assert!(a.sampled_bars.contains(&(data.len() - 1)));
    assert!(a.sampled_bars.len() <= cfg().sample_bars_per_asset);
    // The report's sample is the pure sampler's sample for the same (n, K, first_bar, seed).
    let first_bar = a.sampled_bars[0];
    assert_eq!(
        a.sampled_bars,
        sample_truncation_bars(data.len(), cfg().sample_bars_per_asset, first_bar, a.seed)
    );
}

#[tokio::test]
async fn full_mode_checks_every_bar_from_the_declared_window() {
    let data = synthetic_candles(300);
    let full = CausalityGateConfig {
        mode: LookaheadScanMode::Full,
        ..CausalityGateConfig::default()
    };
    let r = causality_truncation_check(&data, END_OF_ARRAY_STRATEGY, HashMap::new(), &full)
        .await
        .unwrap();
    let first = r.sampled_bars[0];
    assert_eq!(r.sampled_bars, (first..300).collect::<Vec<_>>());
    assert!(!r.passed());
    assert_eq!(
        r.violations.len(),
        r.sampled_bars.len() - 1,
        "every bar but the last"
    );
}

#[tokio::test]
async fn the_pipeline_blocks_on_the_defect_with_a_typed_report() {
    use backtest::python_validation::{run_validation_pipeline, ValidationConfig};
    let data = synthetic_candles(600);
    let config = ValidationConfig {
        python_source: END_OF_ARRAY_STRATEGY.to_string(),
        ..ValidationConfig::default()
    };
    let v = run_validation_pipeline(&data, config).await.unwrap();
    assert_eq!(v.verdict, "fail");
    assert_eq!(v.stages_completed, 0);
    assert_eq!(v.look_ahead_detected, Some(true));
    let report = v.causality_truncation.expect("typed report");
    assert!(!report.passed());
    assert!(v
        .stage_verdicts
        .iter()
        .any(|s| s.name == "Causality (truncation)" && !s.passed));
    assert!(v.summary.contains("Causality gate failed"));
}
