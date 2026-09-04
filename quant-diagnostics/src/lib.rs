//! Pure-Rust quantitative diagnostics shared between research-time
//! (BacktestingEngine) and, in the future, live-time (SignalEngine) callers.
//!
//! No BacktestingEngine- or SignalEngine-specific dependencies — this crate
//! is deliberately data-in/data-out so both sides can share one
//! implementation instead of risking silent divergence.

pub mod variance_ratio;
pub mod ic;
pub mod quantile;
pub mod volatility_regime;
pub mod seasonality;
pub mod multiple_testing;
pub mod cross_sectional;
pub mod cointegration;
pub mod linalg;
pub mod johansen;
pub mod spread;
pub mod vrp;

pub use variance_ratio::{variance_ratio as compute_variance_ratio, VrResult};
pub use ic::{compute_ic, IcResult};
pub use quantile::{quantile_analysis, QuantileBucket};
pub use volatility_regime::{rolling_volatility, volatility_tercile_regimes, VolatilityRegime};
pub use vrp::{vrp_significance, VrpResult};
pub use seasonality::{seasonality_by_bucket, SeasonalityBucket};
pub use multiple_testing::benjamini_hochberg;
pub use cross_sectional::{
    cross_sectional_rank_spread, cross_sectional_rank_spread_by_class, cross_sectional_volatility_rank_spread,
    effective_breadth, pearson_correlation, CrossSectionalRankResult, RebalancePeriod,
};
pub use cointegration::{
    adf_test, engle_granger_test, half_life, ols_simple, spread_zscore, AdfResult,
    CointegrationResult, CriticalValues, OlsResult,
};
pub use johansen::{johansen_test, normalize_cointegrating_vector, JohansenResult};
pub use spread::corwin_schultz_spread;
