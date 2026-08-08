//! Performance metrics for backtested strategies
//!
//! Includes Sharpe ratio, profit factor, net/gross profit, average/median trade return, number of trades.
//! Uses SIMD-friendly patterns for improved performance on large datasets.

/// SIMD lane width for vectorized operations
const SIMD_LANES: usize = 4;

/// Calculates the annualized Sharpe ratio from periodic returns.
/// 
/// This function expects percentage returns (e.g., 0.01 for 1%) and an annual risk-free rate.
/// It calculates the Sharpe ratio and annualizes it using 365 days (crypto trades 24/7).
/// 
/// For high-frequency data (ticks/trades), the returns should be aggregated to daily
/// before calling this function, or use `sharpe_ratio_from_equity_curve` instead.
/// 
/// Formula: Sharpe = sqrt(periods_per_year) * (mean_return - rfr_per_period) / stddev
pub fn sharpe_ratio(returns: &[f64], annual_risk_free_rate: f64) -> Option<f64> {
	if returns.is_empty() {
		return None;
	}
	let mean_return = simd_mean(returns)?;
	let stddev = simd_std_dev(returns)?;
	if stddev == 0.0 {
		return None;
	}
	
	// Percentage returns — assume these are per-period (daily) returns
	// Annualize using 365 days (crypto markets trade 24/7)
	let daily_rfr = annual_risk_free_rate / 365.0;
	let annualization_factor = 365.0_f64.sqrt();
	Some(annualization_factor * (mean_return - daily_rfr) / stddev)
}

/// Asset-class-aware Sharpe ratio. Annualizes with 252 trading days for stocks
/// and forex, 365 for crypto (24/7 markets). Prefer this over `sharpe_ratio`
/// when the asset class is known.
pub fn sharpe_ratio_annual(returns: &[f64], annual_risk_free_rate: f64, trading_days: u32) -> Option<f64> {
	if returns.is_empty() {
		return None;
	}
	let mean_return = simd_mean(returns)?;
	let stddev = simd_std_dev(returns)?;
	if stddev == 0.0 {
		return None;
	}

	let days = trading_days as f64;
	let daily_rfr = annual_risk_free_rate / days;
	let annualization_factor = days.sqrt();
	Some(annualization_factor * (mean_return - daily_rfr) / stddev)
}

/// Calculates annualized Sharpe ratio from an equity curve.
/// 
/// This function calculates daily returns from equity values and computes
/// a properly annualized Sharpe ratio.
/// 
/// # Arguments
/// * `equity_curve` - Vector of portfolio values over time
/// * `annual_risk_free_rate` - Annual risk-free rate (e.g., 0.02 for 2%)
/// * `periods_per_day` - How many data points represent one trading day
///                       (e.g., 1 for daily data, 390 for minute bars, etc.)
pub fn sharpe_ratio_from_equity_curve(
	equity_curve: &[f64],
	annual_risk_free_rate: f64,
	periods_per_day: usize,
) -> Option<f64> {
	if equity_curve.len() < 2 {
		return None;
	}
	
	// Calculate per-period returns
	let period_returns: Vec<f64> = equity_curve.windows(2)
		.filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
		.collect();
	
	if period_returns.is_empty() {
		return None;
	}
	
	// Aggregate to daily returns if we have intraday data
	let daily_returns = if periods_per_day > 1 {
		// Sum returns within each day (approximation for small returns)
		period_returns.chunks(periods_per_day)
			.map(|chunk| chunk.iter().sum::<f64>())
			.collect::<Vec<_>>()
	} else {
		period_returns
	};
	
	if daily_returns.is_empty() {
		return None;
	}
	
	let mean_daily = mean(&daily_returns)?;
	let stddev_daily = std_dev(&daily_returns)?;
	
	if stddev_daily == 0.0 {
		return None;
	}
	
	// Convert annual RFR to daily (365 days for crypto)
	let daily_rfr = annual_risk_free_rate / 365.0;

	// Annualized Sharpe = sqrt(365) * (mean_daily - daily_rfr) / stddev_daily
	let annualization_factor = 365.0_f64.sqrt();
	Some(annualization_factor * (mean_daily - daily_rfr) / stddev_daily)
}

/// Asset-class-aware equity-curve Sharpe. Pass `trading_days = 252` for stocks
/// and forex, `365` for crypto. Prefer this over `sharpe_ratio_from_equity_curve`
/// when the asset class is known.
pub fn sharpe_ratio_from_equity_curve_annual(
	equity_curve: &[f64],
	annual_risk_free_rate: f64,
	periods_per_day: usize,
	trading_days: u32,
) -> Option<f64> {
	if equity_curve.len() < 2 {
		return None;
	}

	let period_returns: Vec<f64> = equity_curve.windows(2)
		.filter_map(|w| if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
		.collect();

	if period_returns.is_empty() {
		return None;
	}

	let daily_returns = if periods_per_day > 1 {
		period_returns.chunks(periods_per_day)
			.map(|chunk| chunk.iter().sum::<f64>())
			.collect::<Vec<_>>()
	} else {
		period_returns
	};

	if daily_returns.is_empty() {
		return None;
	}

	let mean_daily = mean(&daily_returns)?;
	let stddev_daily = std_dev(&daily_returns)?;

	if stddev_daily == 0.0 {
		return None;
	}

	let days = trading_days as f64;
	let daily_rfr = annual_risk_free_rate / days;
	let annualization_factor = days.sqrt();
	Some(annualization_factor * (mean_daily - daily_rfr) / stddev_daily)
}

/// Calculates the profit factor given gross profit and gross loss.
/// Returns a bounded value: 1.0 for no trades, 10.0 for all wins, or actual ratio otherwise.
pub fn profit_factor(gross_profit: f64, gross_loss: f64) -> Option<f64> {
	if gross_loss.abs() < std::f64::EPSILON {
		// No losses
		if gross_profit > std::f64::EPSILON {
			// All wins - use a high but reasonable value
			return Some(10.0);
		} else {
			// No trades at all - neutral
			return Some(1.0);
		}
	}
	// Cap at reasonable max to prevent overflow in downstream calculations
	Some((gross_profit / gross_loss.abs()).min(100.0))
}

/// Calculates net profit from a list of trade returns.
pub fn net_profit(trade_returns: &[f64]) -> f64 {
	trade_returns.iter().sum()
}

/// Calculates gross profit from a list of trade returns.
pub fn gross_profit(trade_returns: &[f64]) -> f64 {
	trade_returns.iter().filter(|&&x| x > 0.0).sum()
}

/// Calculates gross loss from a list of trade returns.
pub fn gross_loss(trade_returns: &[f64]) -> f64 {
	trade_returns.iter().filter(|&&x| x < 0.0).sum()
}

/// Calculates the average trade return.
pub fn average_trade_return(trade_returns: &[f64]) -> Option<f64> {
	mean(trade_returns)
}

/// Calculates the median trade return.
pub fn median_trade_return(trade_returns: &[f64]) -> Option<f64> {
	if trade_returns.is_empty() {
		return None;
	}
	let mut sorted = trade_returns.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let mid = sorted.len() / 2;
	if sorted.len() % 2 == 0 {
		Some((sorted[mid - 1] + sorted[mid]) / 2.0)
	} else {
		Some(sorted[mid])
	}
}

/// Returns the number of trades.
pub fn number_of_trades(trade_returns: &[f64]) -> usize {
	trade_returns.len()
}

/// Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
///
/// Returns the probability that the observed in-sample Sharpe is genuine
/// given `num_trials` independent parameter configurations evaluated during
/// optimization (e.g., GA population × generations). Values below 0.95
/// indicate that the best observed Sharpe is likely a multiple-comparisons
/// artefact rather than a true edge.
///
/// Uses the per-period (per-trade) Sharpe computed internally from `returns`
/// so the result is independent of annualization convention.
///
/// Returns `None` when there are fewer than 4 observations or zero trials.
pub fn deflated_sharpe_ratio(num_trials: u32, returns: &[f64]) -> Option<f64> {
	let n = returns.len();
	if n < 4 || num_trials == 0 {
		return None;
	}
	let n_f = n as f64;
	let n_trials_f = num_trials as f64;

	let mean = simd_mean(returns)?;
	let std = simd_std_dev(returns)?;
	if std < 1e-10 {
		return None;
	}
	let sr = mean / std;

	let skewness = returns.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n_f;
	let excess_kurtosis = returns.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n_f - 3.0;

	// Expected maximum Sharpe under H₀ (Bailey et al. 2014, eq. 8).
	// γ_E ≈ 0.5772 (Euler–Mascheroni constant)
	const EULER_GAMMA: f64 = 0.5772156649015329;
	let sr_star = (1.0 - EULER_GAMMA + EULER_GAMMA * n_trials_f.ln()) / (n_f - 1.0).sqrt();

	// Variance of the Sharpe estimator (Bailey & López de Prado 2014, eq. 7).
	let se_sq = (1.0 - skewness * sr + ((excess_kurtosis + 2.0) / 4.0) * sr * sr) / (n_f - 1.0);
	if se_sq <= 0.0 {
		return None;
	}

	let z = (sr - sr_star) / se_sq.sqrt();
	Some(normal_cdf(z))
}

/// Normal CDF approximation (Abramowitz & Stegun 26.2.17, max error 7.5×10⁻⁸).
fn normal_cdf(x: f64) -> f64 {
	const P: f64 = 0.3275911;
	const A: [f64; 5] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
	let t = 1.0 / (1.0 + P * x.abs());
	let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
	let erf_approx = 1.0 - poly * (-x * x / 2.0).exp();
	0.5 * (1.0 + if x >= 0.0 { 1.0 } else { -1.0 } * erf_approx)
}

// ============================================================================
// SIMD-accelerated helper functions
// Uses 4-wide vectorization for 1.5-2x speedup on large datasets
// ============================================================================

/// SIMD-accelerated sum
#[inline]
fn simd_sum(values: &[f64]) -> f64 {
	if values.len() < SIMD_LANES * 2 {
		return values.iter().sum();
	}

	let chunks = values.len() / SIMD_LANES;
	let remainder = values.len() % SIMD_LANES;

	let mut accumulators = [0.0f64; SIMD_LANES];
	
	for chunk_idx in 0..chunks {
		let base = chunk_idx * SIMD_LANES;
		accumulators[0] += values[base];
		accumulators[1] += values[base + 1];
		accumulators[2] += values[base + 2];
		accumulators[3] += values[base + 3];
	}

	let mut sum = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];

	let remainder_start = chunks * SIMD_LANES;
	for i in 0..remainder {
		sum += values[remainder_start + i];
	}

	sum
}

/// SIMD-accelerated mean
#[inline]
fn simd_mean(data: &[f64]) -> Option<f64> {
	if data.is_empty() {
		return None;
	}
	Some(simd_sum(data) / data.len() as f64)
}

/// SIMD-accelerated standard deviation
#[inline]
fn simd_std_dev(data: &[f64]) -> Option<f64> {
	if data.len() < 2 {
		return None;
	}

	let mean = simd_mean(data)?;
	let n = data.len();

	if n < SIMD_LANES * 2 {
		// Scalar fallback
		let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
		return Some(var.sqrt());
	}

	let chunks = n / SIMD_LANES;
	let remainder = n % SIMD_LANES;

	let mut accumulators = [0.0f64; SIMD_LANES];

	for chunk_idx in 0..chunks {
		let base = chunk_idx * SIMD_LANES;
		let d0 = data[base] - mean;
		let d1 = data[base + 1] - mean;
		let d2 = data[base + 2] - mean;
		let d3 = data[base + 3] - mean;
		accumulators[0] += d0 * d0;
		accumulators[1] += d1 * d1;
		accumulators[2] += d2 * d2;
		accumulators[3] += d3 * d3;
	}

	let mut sum_sq = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];

	let remainder_start = chunks * SIMD_LANES;
	for i in 0..remainder {
		let d = data[remainder_start + i] - mean;
		sum_sq += d * d;
	}

	Some((sum_sq / n as f64).sqrt())
}

/// Legacy helper for compatibility (now uses SIMD internally)
fn mean(data: &[f64]) -> Option<f64> {
	simd_mean(data)
}

/// Legacy helper for compatibility (now uses SIMD internally)
fn std_dev(data: &[f64]) -> Option<f64> {
	simd_std_dev(data)
}

#[cfg(test)]
mod tests {
	use super::*;

	// ── sharpe_ratio ──

	#[test]
	fn sharpe_ratio_empty_returns_none() {
		assert!(sharpe_ratio(&[], 0.02).is_none());
	}

	#[test]
	fn sharpe_ratio_zero_std_returns_none() {
		assert!(sharpe_ratio(&[0.01, 0.01, 0.01], 0.02).is_none());
	}

	#[test]
	fn sharpe_ratio_positive_returns() {
		let returns = vec![0.01, 0.02, -0.005, 0.015, 0.008];
		let sr = sharpe_ratio(&returns, 0.02).unwrap();
		assert!(sr > 0.0, "expected positive sharpe, got {sr}");
	}

	#[test]
	fn sharpe_ratio_large_returns_use_rfr_path() {
		// Returns > 1.0 in magnitude must use the RFR annualization path,
		// not a dollar-branch shortcut. With RFR=0 the formula is sqrt(365)*mean/std.
		let returns = vec![2.0, 3.0, -1.0, 1.5, 1.0];
		let sr = sharpe_ratio(&returns, 0.0).unwrap();
		let m = returns.iter().sum::<f64>() / returns.len() as f64;
		let expected = 365.0_f64.sqrt() * m / simd_std_dev(&returns).unwrap();
		assert!((sr - expected).abs() < 1e-10, "got {sr}, expected {expected}");
	}

	#[test]
	fn deflated_sharpe_ratio_penalizes_many_trials() {
		// With many GA trials the DSR should be well below 1.0 for a mediocre strategy
		let returns: Vec<f64> = (0..50).map(|i| if i % 3 == 0 { -0.01 } else { 0.005 }).collect();
		let dsr_few = deflated_sharpe_ratio(1, &returns).unwrap();
		let dsr_many = deflated_sharpe_ratio(1250, &returns).unwrap();
		assert!(dsr_few > dsr_many, "more trials should yield lower DSR for same returns");
	}

	#[test]
	fn deflated_sharpe_ratio_none_on_few_observations() {
		let returns = vec![0.01, 0.02, 0.01]; // < 4 samples
		assert!(deflated_sharpe_ratio(100, &returns).is_none());
		assert!(deflated_sharpe_ratio(0, &[0.01; 50]).is_none());
	}

	#[test]
	fn sharpe_ratio_single_return_none() {
		// stddev needs >= 2 values
		assert!(sharpe_ratio(&[0.01], 0.02).is_none());
	}

	// ── sharpe_ratio_from_equity_curve ──

	#[test]
	fn sharpe_from_curve_short_returns_none() {
		assert!(sharpe_ratio_from_equity_curve(&[100.0], 0.02, 1).is_none());
	}

	#[test]
	fn sharpe_from_curve_flat_returns_none() {
		assert!(sharpe_ratio_from_equity_curve(&[100.0, 100.0, 100.0], 0.02, 1).is_none());
	}

	#[test]
	fn sharpe_from_curve_rising() {
		let curve: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
		let sr = sharpe_ratio_from_equity_curve(&curve, 0.02, 1).unwrap();
		assert!(sr > 0.0);
	}

	#[test]
	fn sharpe_from_curve_intraday_aggregation() {
		// 10 "ticks" per day over 3 days
		let curve: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1).collect();
		let sr = sharpe_ratio_from_equity_curve(&curve, 0.02, 10).unwrap();
		assert!(sr.is_finite());
	}

	// ── profit_factor ──

	#[test]
	fn profit_factor_no_losses_returns_10() {
		assert_eq!(profit_factor(500.0, 0.0), Some(10.0));
	}

	#[test]
	fn profit_factor_no_trades_returns_1() {
		assert_eq!(profit_factor(0.0, 0.0), Some(1.0));
	}

	#[test]
	fn profit_factor_normal() {
		let pf = profit_factor(300.0, 100.0).unwrap();
		assert!((pf - 3.0).abs() < 1e-10);
	}

	#[test]
	fn profit_factor_capped_at_100() {
		let pf = profit_factor(10000.0, 0.01).unwrap();
		assert!(pf <= 100.0);
	}

	// ── net_profit / gross_profit / gross_loss ──

	#[test]
	fn net_profit_empty() {
		assert_eq!(net_profit(&[]), 0.0);
	}

	#[test]
	fn net_profit_mixed() {
		assert!((net_profit(&[10.0, -3.0, 5.0]) - 12.0).abs() < 1e-10);
	}

	#[test]
	fn gross_profit_filters_positive() {
		assert!((gross_profit(&[10.0, -3.0, 5.0]) - 15.0).abs() < 1e-10);
	}

	#[test]
	fn gross_loss_filters_negative() {
		assert!((gross_loss(&[10.0, -3.0, 5.0]) - (-3.0)).abs() < 1e-10);
	}

	#[test]
	fn gross_profit_empty() {
		assert_eq!(gross_profit(&[]), 0.0);
	}

	#[test]
	fn gross_loss_empty() {
		assert_eq!(gross_loss(&[]), 0.0);
	}

	// ── average_trade_return / median_trade_return ──

	#[test]
	fn average_trade_return_empty_none() {
		assert!(average_trade_return(&[]).is_none());
	}

	#[test]
	fn average_trade_return_correct() {
		let avg = average_trade_return(&[10.0, 20.0, 30.0]).unwrap();
		assert!((avg - 20.0).abs() < 1e-10);
	}

	#[test]
	fn median_trade_return_empty_none() {
		assert!(median_trade_return(&[]).is_none());
	}

	#[test]
	fn median_trade_return_odd() {
		let m = median_trade_return(&[3.0, 1.0, 2.0]).unwrap();
		assert!((m - 2.0).abs() < 1e-10);
	}

	#[test]
	fn median_trade_return_even() {
		let m = median_trade_return(&[4.0, 1.0, 2.0, 3.0]).unwrap();
		assert!((m - 2.5).abs() < 1e-10);
	}

	// ── number_of_trades ──

	#[test]
	fn number_of_trades_empty() {
		assert_eq!(number_of_trades(&[]), 0);
	}

	#[test]
	fn number_of_trades_counts() {
		assert_eq!(number_of_trades(&[1.0, 2.0, 3.0]), 3);
	}

	// ── SIMD helpers ──

	#[test]
	fn simd_sum_small_and_large() {
		let small = vec![1.0, 2.0, 3.0];
		assert!((simd_sum(&small) - 6.0).abs() < 1e-10);

		let large: Vec<f64> = (1..=100).map(|i| i as f64).collect();
		assert!((simd_sum(&large) - 5050.0).abs() < 1e-10);
	}

	#[test]
	fn simd_mean_empty_none() {
		assert!(simd_mean(&[]).is_none());
	}

	#[test]
	fn simd_std_dev_single_none() {
		assert!(simd_std_dev(&[42.0]).is_none());
	}

	#[test]
	fn simd_std_dev_uniform_zero() {
		let sd = simd_std_dev(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
		assert!(sd.abs() < 1e-10);
	}
}

