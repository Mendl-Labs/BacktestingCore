# BacktestingCore

**Open-source backtesting engine core** — event-driven simulation, walk-forward analysis, genetic strategy optimization, portfolio/risk management, and Python-strategy execution for the TradingPlatform.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Workspace Crates](#workspace-crates)
- [What This Is Not](#what-this-is-not)
- [Extension Points](#extension-points)
- [Quick Start](#quick-start)
- [Sibling Repositories](#sibling-repositories)
- [Testing & Benchmarks](#testing--benchmarks)
- [License](#license)

---

## Overview

BacktestingCore is the generic simulation and analysis infrastructure extracted from the platform's private backtesting SaaS. It:

1. **Simulates** strategy execution against historical tick/bar data with a lock-free orderbook and configurable slippage/commission models
2. **Runs Python strategies** via PyO3 — strategy logic is authored in Python, executed against Rust-speed simulation internals
3. **Optimizes parameters** with a genetic algorithm (GA), scored by a pluggable fitness function you supply
4. **Validates robustness** via walk-forward analysis (train/test windowing with a purge gap to prevent lookahead leakage)
5. **Manages portfolios** across multiple simultaneous assets/strategies with rebalancing and cross-asset risk limits
6. **Prices derivatives** (Black-Scholes, Greeks) for options/futures-aware strategies
7. **Routes orders** across simulated venues (TWAP/VWAP/Implementation-Shortfall algorithms, cross-venue arbitrage detection)

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| **Strategy language** | Python (via PyO3) |
| **Simulation** | Event-driven, tick or bar granularity |
| **Optimization** | Genetic algorithm, pluggable fitness function |
| **Data format** | Memory-mappable binary (via `data_prep`) for fast repeated backtests |
| **Validation** | Walk-forward with purge gap, bootstrap confidence intervals |

---

## Architecture

### Data & Execution Flow

```
Historical data (PostgreSQL / files)
        │
        ▼
┌───────────────┐   converts to lightweight SimulationTick,
│   data_prep    │   optionally precomputes features (volatility, EMA, ...),
│  ("librarian") │   writes memory-mappable binary
└───────┬────────┘
        │
        ▼
┌───────────────┐        ┌────────────┐        ┌──────────────┐
│  dataloader    │───────▶│  backtest  │◀──────▶│  orderbook   │
│ (streams data) │        │ (sim core) │        │ (lock-free)  │
└───────────────┘        └─────┬──────┘        └──────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐   ┌──────────────┐   ┌──────────────┐
       │  strategy   │   │ portfoliomanager│  │  riskmanager  │
       │ (PyO3 exec) │   │ (positions/P&L) │  │ (VaR, limits) │
       └────────────┘   └──────────────┘   └──────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │ metrics   │  Sharpe/Sortino/Calmar, drawdown,
                          │           │  VaR/ES, slippage, order stats
                          └──────────┘
```

### Optimization & Validation

```
┌────────────┐   evaluates population against    ┌──────────────────┐
│   genetic   │───────────────────────────────────▶│  backtest (sim)  │
│    (GA)     │◀────────────────────────────────────│                  │
└─────┬──────┘   FitnessFunction (caller-supplied)  └──────────────────┘
      │
      ▼
┌────────────┐
│ walkforward │  train/test windowing with purge gap,
│             │  out-of-sample validation of GA winners
└────────────┘
      │
      ▼
┌───────────────────┐
│ quant-diagnostics   │  variance-ratio test, cross-sectional checks —
│                     │  pure data-in/data-out, shared with SignalEngine
└───────────────────┘
```

---

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `backtest` | Core simulation engine — event loop, PyO3 strategy execution bridge |
| `orderbook` | Lock-free limit orderbook for simulated matching |
| `dataloader` | Streams prepared historical data into the simulator |
| `data_prep` | Converts raw historical data (from PostgreSQL) into memory-mappable binary format, optionally precomputing features |
| `walkforward` | Train/test window generation with a purge gap; out-of-sample validation reporting |
| `metrics` | Performance metrics: Sharpe, Sortino, Calmar, drawdown, VaR/ES, slippage, order statistics |
| `derivatives` | Shared option/future/perpetual instrument types — foundation for `greeks` and derivative-aware strategies |
| `greeks` | Black-Scholes pricing and Greeks (Δ, Γ, Θ, Vega, ρ), implied-vol solver, portfolio-level Greek aggregation |
| `signal` | Trading signal type definitions shared across the simulation pipeline |
| `smartrouter` | Multi-venue simulated order routing — best-venue selection, cross-venue arbitrage detection, TWAP/VWAP/IS execution algorithms |
| `portfoliomanager` | Portfolio state tracking, position management, P&L calculation across multiple assets |
| `riskmanager` | Risk-limit types and enforcement — VaR-based position limits, correlation risk, volatility-regime switching |
| `genetic` | Genetic-algorithm optimizer; ships a scoring **interface** (`FitnessFunction`) but no concrete fitness formula |
| `quant-diagnostics` | Pure-Rust statistical diagnostics (e.g. variance-ratio test) with no BacktestingEngine/SignalEngine-specific dependencies, so both research-time and live-time callers share one implementation |
| `strategy` | Strategy execution framework — indicator library, feature compiler, Python strategy-source validation, deployment-readiness types |
| `config` | Configuration management |

---

## What This Is Not

This crate ships **infrastructure, not policy or proprietary strategy logic**. Notably absent, by design:

- **A fitness formula.** `genetic::FitnessFunction` is a trait with zero default implementation — "what makes a candidate strategy good" (which metrics matter, how they're weighted, hard-disqualification rules) is a platform-specific decision you supply.
- **Deployment/approval workflow enforcement, database export, or publishing to a live-trading system.** `strategy::deployment` defines generic advisory-warning types (used internally by `walkforward`'s auto-export reporting) but does not implement an approval pipeline, admin gating, or a path to production.
- **Multi-tenant billing, usage metering, or subscription tiers.** None of these crates know what a "tenant" or "plan" is.

If you're building a hosted product on top of this, those pieces belong in your own private layer, constructed at your binary's entry point and passed in via the extension points below — mirroring how the platform's own private SaaS binary consumes this crate.

---

## Extension Points

| Trait / Seam | Where | You supply |
|---|---|---|
| `genetic::fitness::FitnessFunction` | `genetic` crate | Concrete scoring formula (weights, disqualification gates) — construct it and pass it into the GA optimizer |
| `strategy::deployment` advisory types | `strategy` crate | Your own approval/deployment pipeline built around the generic warning types |

---

## Quick Start

### Prerequisites

- Rust 1.82+
- Python 3.10+ (for PyO3 strategy execution)
- PostgreSQL (only if using `data_prep`'s database-backed ingestion — the `databaseschema` dependency is optional/feature-gated)

### Build

```powershell
cd BacktestingCore
cargo build --workspace --release
```

### Run a backtest

```powershell
# Prepare historical data into binary format
cargo run --release -p data_prep -- --symbol BTCUSD --exchange kraken --start 2024-01-01 --end 2024-06-01

# Run the simulator against a Python strategy
cargo run --release -p backtest -- --strategy path/to/strategy.py --data path/to/prepared.bin
```

### Run genetic optimization

```rust
use genetic::{GeneticOptimizer, fitness::{FitnessFunction, FitnessInputs}};

struct MyFitness;
impl FitnessFunction for MyFitness {
    fn compute(&self, inputs: &FitnessInputs) -> genetic::FitnessResult {
        // your scoring formula
        # unimplemented!()
    }
}

let optimizer = GeneticOptimizer::new(MyFitness, /* population/generation config */);
let best = optimizer.run(/* ... */);
```

---

## Sibling Repositories

Several crates have optional/feature-gated path dependencies on sibling repositories, expected to sit alongside `BacktestingCore` on disk:

```
TradingPlatform/
├── BacktestingCore/     (this repo)
├── databaseschema/       (optional — result/job persistence)
└── LoggingEngine/        (ultra-logger — structured logging)
```

Building without them is possible where the dependency is `optional = true` (disable the corresponding feature); `ultra-logger` is a hard dependency for logging output.

---

## Testing & Benchmarks

```powershell
# Unit + integration tests
cargo test --workspace

# Benchmarks (per-crate, where present)
cargo bench -p genetic
cargo bench -p dataloader
cargo bench -p portfoliomanager
cargo bench -p riskmanager
cargo bench -p signal
cargo bench -p smartrouter
cargo bench -p strategy
cargo bench -p config
```

---

## License

Functional Source License, Version 1.1, ALv2 Future License (FSL-1.1-ALv2) — see [LICENSE](LICENSE). Free for internal use, non-commercial research/education, and professional services; converts to Apache License 2.0 two years after each version's release. In short: use it, modify it, build a product on top of it — you just can't resell this engine itself (or a thin wrapper around it) as a directly competing hosted backtesting service.
