//! Backtesting Engine Library
//!
//! Core simulation and analysis modules for the backtesting platform.
//! All strategy execution is Python-based via PyO3.
//!
//! # Simulation Routing
//!
//! ```text
//! User submits backtest job
//!   │
//!   ├─ StrategyType::Custom (directional Python)
//!   │    └─ python_simulation::run()
//!   │         └─ Vectorized: compute_signals(prices, volumes, timestamps, params) → fills → BacktestResult
//!   │
//!   └─ GA Optimization (two-tier)
//!        ├─ Fast Tier (Rust-native, parallel via rayon)
//!        │    └─ PrecomputedIndicators + SignalRuleSet → CandleSimulator → FitnessResult
//!        └─ Standard Tier (Python vectorized + Rust position sim)
//!             └─ compute_signals() → CandleSimulator → FitnessResult
//! ```

// Logging must be defined first since other modules use it
#[macro_use]
pub mod logging_facade;

// Core modules
pub mod types;
pub mod verdict;
pub mod feature_diagnostics;
pub mod collectors;
pub mod rolling_window;
pub mod simulation;
pub mod engine;
pub mod html_export;
pub mod monte_carlo;
pub mod statistical_significance;
pub mod portfolio_simulation;
pub mod latency;
pub mod portfolio_risk;
pub mod rebalancing;
pub mod accuracy_validation;
pub mod synthetic_book;
pub mod amm_math;
pub mod lp_simulation;
pub mod candle_sim;
pub mod indicator_precompute;
pub mod fast_signal_eval;
pub mod pair_simulation;
pub mod basket_simulation;
pub mod cross_sectional_simulation;

// Python strategy simulation (requires python feature)
#[cfg(feature = "python")]
pub mod python_simulation;
pub mod python_validation;

// Re-export commonly used types at crate root
pub use types::{BacktestResult, BacktestEvent, MarketDataInput};
pub use engine::{BacktestEngine, BacktestResults, WalkForwardSummary, AnalysisMetadata, ProgressCallback};
pub use html_export::HtmlExporter;
pub use simulation::SimulationLoop;
pub use monte_carlo::{MonteCarloResult, run_monte_carlo_simulation};
pub use python_validation::{ValidationConfig, ValidationResult, WalkForwardResult, extract_parameter_schema};
pub use portfolio_simulation::{
    PortfolioSimulator,
    PortfolioSimResult,
    SymbolResult,
    AssetStrategyParams,
};

pub use statistical_significance::{
    SignificanceResult,
    compute_sharpe_significance,
    expected_max_sharpe_under_null,
};

// Python strategy simulation exports
#[cfg(feature = "python")]
pub use python_simulation::{PythonSimConfig, PythonSimResult};
