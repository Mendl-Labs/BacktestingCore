//! Python FFI integration test — verifies a Python strategy can be loaded,
//! executed, and produce meaningful results through PyO3.
//!
//! Run with: cargo test -p backtest --features python -- python_ffi

#![cfg(feature = "python")]

use std::collections::HashMap;

use backtest::python_simulation::{self, PythonSimConfig};
use config::BacktestConfig;
use dataloader::{Candle, MarketData};
use chrono::{DateTime, Utc, TimeZone};

/// Minimal Python strategy: buys when price crosses above 20-period SMA,
/// sells when it crosses below.
const PYTHON_STRATEGY: &str = r#"
class Strategy:
    def __init__(self, parameters):
        self.lookback = int(parameters.get("lookback", 20))
        self.prices = []

    def on_candle(self, candle, position, portfolio):
        self.prices.append(candle["close"])
        if len(self.prices) < self.lookback:
            return None

        sma = sum(self.prices[-self.lookback:]) / self.lookback
        current = candle["close"]

        if current > sma and position <= 0:
            return {"action": "buy", "quantity": 1.0}
        elif current < sma and position > 0:
            return {"action": "sell", "quantity": 1.0}
        return None
"#;

/// Generate synthetic market data with a trend + noise pattern.
fn generate_synthetic_data(n: usize) -> Vec<MarketData> {
    let mut data = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        // Trending pattern to generate trades
        let trend = if i < n / 3 {
            0.1 // uptrend
        } else if i < 2 * n / 3 {
            -0.15 // downtrend
        } else {
            0.08 // uptrend
        };
        let noise = ((i * 7 + 3) % 11) as f64 * 0.05 - 0.25;
        price = (price + trend + noise).max(10.0);

        let high = price * 1.005;
        let low = price * 0.995;
        let open = price + (if i % 2 == 0 { 0.1 } else { -0.1 });
        let volume = 1000.0 + (i % 50) as f64 * 10.0;
        let ts = Utc.timestamp_opt(1_700_000_000 + (i as i64 * 60), 0).unwrap();

        data.push(MarketData::Candle(Candle {
            timestamp: ts,
            symbol: "BTC/USD".into(),
            exchange: "test".into(),
            open,
            high,
            low,
            close: price,
            volume,
            trade_count: 10,
        }));
    }
    data
}

#[tokio::test]
async fn python_strategy_produces_trades_and_metrics() {
    let market_data = generate_synthetic_data(500);

    let mut backtest_config = BacktestConfig::default();
    backtest_config.trading.initial_capital = 100_000.0;

    let config = PythonSimConfig {
        python_source: PYTHON_STRATEGY.to_string(),
        backtest_config,
        fee_config: None,
        supplementary_data: HashMap::new(),
        parameters: {
            let mut p = HashMap::new();
            p.insert("lookback".to_string(), config::ParameterValue::Int(20));
            p
        },
        progress_callback: None,
        risk_manager: None,
        max_trade_log_size: None,
        orderbook_snapshots: None,
        multi_venue_data: None,
        option_instrument: None,
    };

    let result = python_simulation::run(&market_data, config)
        .await
        .expect("Python simulation should succeed");

    let br = &result.backtest_result;

    // Strategy should have traded
    assert!(br.num_trades > 0, "Expected trades, got {}", br.num_trades);

    // Equity curve should have entries
    assert!(
        br.equity_curve.len() > 1,
        "Equity curve too short: {}",
        br.equity_curve.len()
    );

    // Significance fields should be populated (wired in A.1)
    assert!(
        br.sharpe_t_stat.is_some(),
        "sharpe_t_stat should be Some after A.1 wiring"
    );
    assert!(
        br.significance_pvalue.is_some(),
        "significance_pvalue should be Some"
    );
    assert!(
        br.deflated_sharpe.is_some(),
        "deflated_sharpe should be Some"
    );

    // Risk metrics should be populated (wired in A.2)
    assert!(br.ulcer_index.is_some(), "ulcer_index should be Some");
    assert!(br.pain_index.is_some(), "pain_index should be Some");
    assert!(br.cdar_95.is_some(), "cdar_95 should be Some");

    // Sanity: p-value must be in [0, 1]
    if let Some(p) = br.significance_pvalue {
        assert!(
            (0.0..=1.0).contains(&p),
            "p-value out of range: {}",
            p
        );
    }
}

// ── Per-trade dynamic position sizing (compute_position_sizes) ─────────────
//
// Flat (constant) price series so equity barely drifts between the two
// entries in each test below (only commission/slippage move it slightly) —
// this keeps the fill-quantity ratio between a "small" and "large" sizing
// regime close to the ratio of the fractions themselves, without needing to
// hand-replicate the engine's own sizing formula in the assertion.
fn generate_flat_data(n: usize, price: f64) -> Vec<MarketData> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let ts = Utc.timestamp_opt(1_700_000_000 + (i as i64 * 60), 0).unwrap();
        data.push(MarketData::Candle(Candle {
            timestamp: ts,
            symbol: "BTC/USD".into(),
            exchange: "test".into(),
            open: price,
            high: price,
            low: price,
            close: price,
            volume: 1000.0,
            trade_count: 10,
        }));
    }
    data
}

/// BUY at tick 0, CLOSE at tick 10, BUY at tick 15, CLOSE at tick 25, HOLD elsewhere.
const SIZING_SIGNAL_PATTERN: &str = r#"
def _signal_pattern(n):
    sig = [0] * n
    if n > 0:
        sig[0] = 1
    if n > 10:
        sig[10] = 2
    if n > 15:
        sig[15] = 1
    if n > 25:
        sig[25] = 2
    return sig
"#;

fn python_strategy_with_varying_sizes() -> String {
    format!(
        r#"
import numpy as np
from trading_platform import BaseStrategy
{pattern}

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "VaryingSizeStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.array(_signal_pattern(len(prices)), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        # Small fraction for the first entry (tick 0), large for the second (tick 15).
        sizes = np.full(len(prices), 0.05)
        if len(prices) > 15:
            sizes[15] = 0.5
        return sizes
"#,
        pattern = SIZING_SIGNAL_PATTERN
    )
}

fn python_strategy_no_sizing_method() -> String {
    format!(
        r#"
import numpy as np
from trading_platform import BaseStrategy
{pattern}

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "NoSizingStrategy"

    def parameter_space(self):
        return {{"risk_per_trade": {{"type": "float", "min": 0.01, "max": 0.1, "default": 0.02}}}}

    def compute_signals(self, prices, volumes, timestamps):
        return np.array(_signal_pattern(len(prices)), dtype=np.int8)
"#,
        pattern = SIZING_SIGNAL_PATTERN
    )
}

fn python_strategy_matching_default_sizing() -> String {
    format!(
        r#"
import numpy as np
from trading_platform import BaseStrategy
{pattern}

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "MatchingDefaultSizingStrategy"

    def parameter_space(self):
        return {{"risk_per_trade": {{"type": "float", "min": 0.01, "max": 0.1, "default": 0.02}}}}

    def compute_signals(self, prices, volumes, timestamps):
        return np.array(_signal_pattern(len(prices)), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        # Every entry uses exactly the same 2% that the flat position_size_pct
        # default would use anyway -- resulting quantities must match the
        # no-compute_position_sizes baseline exactly.
        return np.full(len(prices), 0.02)
"#,
        pattern = SIZING_SIGNAL_PATTERN
    )
}

fn python_strategy_wrong_length_sizing() -> String {
    format!(
        r#"
import numpy as np
from trading_platform import BaseStrategy
{pattern}

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WrongLengthSizingStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.array(_signal_pattern(len(prices)), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        # Deliberately wrong length -- must fall back to position_size_pct
        # for the WHOLE run rather than failing the backtest.
        return np.full(3, 0.5)
"#,
        pattern = SIZING_SIGNAL_PATTERN
    )
}

fn run_sizing_config(python_source: String) -> PythonSimConfig {
    let mut backtest_config = BacktestConfig::default();
    backtest_config.trading.initial_capital = 100_000.0;
    PythonSimConfig {
        python_source,
        backtest_config,
        fee_config: None,
        supplementary_data: HashMap::new(),
        parameters: HashMap::new(),
        progress_callback: None,
        risk_manager: None,
        max_trade_log_size: None,
        orderbook_snapshots: None,
        multi_venue_data: None,
        option_instrument: None,
    }
}

#[tokio::test]
async fn compute_position_sizes_scales_fill_quantity_across_entries() {
    let market_data = generate_flat_data(30, 100.0);
    let config = run_sizing_config(python_strategy_with_varying_sizes());

    let result = python_simulation::run(&market_data, config)
        .await
        .expect("Python simulation with compute_position_sizes should succeed");

    let trades = &result.backtest_result.trade_log;
    assert!(trades.len() >= 2, "expected at least 2 entries, got {}", trades.len());

    let small_qty = trades[0].quantity;
    let large_qty = trades[1].quantity;
    assert!(small_qty > 0.0 && large_qty > 0.0, "both entries should have positive size");

    let ratio = large_qty / small_qty;
    assert!(
        (7.0..=13.0).contains(&ratio),
        "expected the 0.5-fraction entry to be ~10x the 0.05-fraction entry, got ratio {:.2} ({} vs {})",
        ratio, small_qty, large_qty
    );
}

#[tokio::test]
async fn compute_position_sizes_matching_default_matches_no_sizing_baseline() {
    let market_data = generate_flat_data(30, 100.0);

    let baseline_config = run_sizing_config(python_strategy_no_sizing_method());
    let baseline = python_simulation::run(&market_data, baseline_config)
        .await
        .expect("baseline (no compute_position_sizes) simulation should succeed");

    let matching_config = run_sizing_config(python_strategy_matching_default_sizing());
    let matching = python_simulation::run(&market_data, matching_config)
        .await
        .expect("compute_position_sizes matching the default simulation should succeed");

    assert!(!baseline.backtest_result.trade_log.is_empty());
    assert_eq!(
        baseline.backtest_result.trade_log.len(),
        matching.backtest_result.trade_log.len(),
        "adding compute_position_sizes that mirrors the existing default must not change trade count"
    );
    for (b, m) in baseline.backtest_result.trade_log.iter().zip(matching.backtest_result.trade_log.iter()) {
        assert!(
            (b.quantity - m.quantity).abs() < 1e-6,
            "fill quantity should be unchanged when compute_position_sizes returns the same fraction \
             as the position_size_pct default: baseline={} matching={}",
            b.quantity, m.quantity
        );
    }
}

#[tokio::test]
async fn compute_position_sizes_wrong_length_falls_back_without_failing_backtest() {
    let market_data = generate_flat_data(30, 100.0);
    let config = run_sizing_config(python_strategy_wrong_length_sizing());

    let result = python_simulation::run(&market_data, config)
        .await
        .expect("a wrong-length compute_position_sizes output must fall back, not fail the backtest");

    assert!(
        result.backtest_result.num_trades > 0,
        "expected the backtest to still complete with trades via the position_size_pct fallback, got {}",
        result.backtest_result.num_trades
    );
}
