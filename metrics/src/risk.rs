//! Risk metrics for backtested strategies
//!
//! Includes maximum drawdown, volatility, Sortino ratio, and Calmar ratio.
//! Uses SIMD-friendly patterns for improved performance on large datasets.

/// SIMD lane width for vectorized operations
const SIMD_LANES: usize = 4;

/// Calculates the maximum drawdown from a list of equity values.
/// Returns the maximum drawdown as a decimal (0.0 to 1.0).
/// If equity goes negative (which indicates a bug), clamps to 1.0 (100% loss).
pub fn max_drawdown(equity_curve: &[f64]) -> Option<f64> {
	if equity_curve.is_empty() {
		return None;
	}
	let mut max_drawdown = 0.0;
	let mut peak = equity_curve[0];
	for &value in equity_curve {
		if value > peak {
			peak = value;
		}
		// Guard against division by zero or negative peak
		if peak <= 0.0 {
			continue;
		}
		let drawdown = (peak - value) / peak;
		// Clamp drawdown to valid range [0.0, 1.0]
		let drawdown = drawdown.clamp(0.0, 1.0);
		if drawdown > max_drawdown {
			max_drawdown = drawdown;
		}
	}
	Some(max_drawdown)
}

/// Calculates the volatility (standard deviation of returns).
/// Uses SIMD-accelerated computation for large datasets.
pub fn volatility(returns: &[f64]) -> Option<f64> {
	if returns.is_empty() {
		return None;
	}
	simd_std_dev(returns)
}

/// Calculates the downside deviation (standard deviation of negative returns only).
/// Used in Sortino ratio calculation.
pub fn downside_deviation(returns: &[f64], target_return: f64) -> Option<f64> {
	if returns.is_empty() {
		return None;
	}
	
	let downside_returns: Vec<f64> = returns
		.iter()
		.filter(|&&r| r < target_return)
		.map(|&r| (r - target_return).powi(2))
		.collect();
	
	if downside_returns.is_empty() {
		return Some(0.0); // No downside returns
	}
	
	let sum: f64 = downside_returns.iter().sum();
	Some((sum / returns.len() as f64).sqrt())
}

/// Calculates the annualized Sortino ratio.
/// Sortino = sqrt(periods_per_year) * (Mean Return - Risk Free Rate) / Downside Deviation
/// Unlike Sharpe, only penalizes downside volatility.
/// Annualized using 365 days (crypto markets trade 24/7).
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64) -> Option<f64> {
	if returns.is_empty() {
		return None;
	}
	
	let mean_return = mean(returns)?;
	let daily_rfr = risk_free_rate / 365.0;
	let dd = downside_deviation(returns, daily_rfr)?;
	
	if dd == 0.0 {
		// No downside deviation - if positive returns, cap at 99.99
		// (f64::MAX produces absurd display values in the UI)
		if mean_return > daily_rfr {
			return Some(99.99);
		}
		return None;
	}
	
	let annualization_factor = 365.0_f64.sqrt();
	Some(annualization_factor * (mean_return - daily_rfr) / dd)
}

/// Calculates the Calmar ratio.
/// Calmar = Annualized Return / Maximum Drawdown
/// Measures return relative to worst-case drawdown.
pub fn calmar_ratio(annualized_return: f64, max_drawdown: f64) -> Option<f64> {
	if max_drawdown == 0.0 {
		if annualized_return > 0.0 {
			// Cap at 99.99 instead of f64::MAX to avoid absurd UI values
			return Some(99.99);
		}
		return None;
	}
	
	Some(annualized_return / max_drawdown)
}

/// Calculates the current drawdown from peak.
pub fn current_drawdown(equity_curve: &[f64]) -> Option<f64> {
	if equity_curve.is_empty() {
		return None;
	}
	
	let peak = equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let current = *equity_curve.last()?;
	
	if peak <= 0.0 {
		return None;
	}
	
	Some(((peak - current) / peak).clamp(0.0, 1.0))
}

/// Calculates the average drawdown over the equity curve.
pub fn average_drawdown(equity_curve: &[f64]) -> Option<f64> {
	if equity_curve.is_empty() {
		return None;
	}
	
	let mut peak = equity_curve[0];
	let mut total_dd = 0.0;
	let mut count = 0;
	
	for &value in equity_curve {
		if value > peak {
			peak = value;
		}
		if peak > 0.0 {
			let dd = (peak - value) / peak;
			total_dd += dd.clamp(0.0, 1.0);
			count += 1;
		}
	}
	
	if count == 0 {
		return None;
	}
	
	Some(total_dd / count as f64)
}

// ============================================================================
// SIMD-accelerated helper functions
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
	if data.is_empty() {
		return None;
	}
	if data.len() == 1 {
		// Single value has zero volatility
		return Some(0.0);
	}

	let mean = simd_mean(data)?;
	let n = data.len();

	if n < SIMD_LANES * 2 {
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

// ============================================================================
// Drawdown distribution metrics
// ============================================================================

/// Compute the percentage drawdown series from an equity curve.
///
/// Each element is the drawdown at that point as a non-negative fraction
/// (0.0 = at peak, 0.15 = 15% below peak).
fn drawdown_series(equity_curve: &[f64]) -> Vec<f64> {
	let mut peak = f64::NEG_INFINITY;
	equity_curve
		.iter()
		.map(|&val| {
			if val > peak {
				peak = val;
			}
			if peak > 0.0 {
				((peak - val) / peak).max(0.0)
			} else {
				0.0
			}
		})
		.collect()
}

/// Ulcer Index — RMS of percentage drawdowns.
///
/// Measures the depth **and** duration of drawdowns.  Lower is better.
/// Unlike max drawdown or average drawdown, it penalises deep drawdowns
/// quadratically.
pub fn ulcer_index(equity_curve: &[f64]) -> Option<f64> {
	if equity_curve.is_empty() {
		return None;
	}
	let dd = drawdown_series(equity_curve);
	let sum_sq: f64 = dd.iter().map(|d| d * d).sum();
	Some((sum_sq / dd.len() as f64).sqrt())
}

/// Ulcer Performance Index (UPI) = (total return − risk-free) / Ulcer Index.
///
/// Martin ratio variant — measures return per unit of drawdown pain.
pub fn ulcer_performance_index(equity_curve: &[f64], risk_free_rate: f64) -> Option<f64> {
	if equity_curve.len() < 2 {
		return None;
	}
	let first = equity_curve[0];
	let last = *equity_curve.last()?;
	if first <= 0.0 {
		return None;
	}
	let total_return = (last - first) / first;
	let ui = ulcer_index(equity_curve)?;
	if ui < 1e-12 {
		if total_return > risk_free_rate {
			return Some(99.99);
		}
		return None;
	}
	Some((total_return - risk_free_rate) / ui)
}

/// Pain Index — mean of percentage drawdowns.
///
/// Similar to Ulcer Index but uses arithmetic mean instead of RMS,
/// so deep drawdowns are not exaggerated as much.
pub fn pain_index(equity_curve: &[f64]) -> Option<f64> {
	if equity_curve.is_empty() {
		return None;
	}
	let dd = drawdown_series(equity_curve);
	let sum: f64 = dd.iter().sum();
	Some(sum / dd.len() as f64)
}

/// Pain Ratio = (total return − risk-free) / Pain Index.
pub fn pain_ratio(equity_curve: &[f64], risk_free_rate: f64) -> Option<f64> {
	if equity_curve.len() < 2 {
		return None;
	}
	let first = equity_curve[0];
	let last = *equity_curve.last()?;
	if first <= 0.0 {
		return None;
	}
	let total_return = (last - first) / first;
	let pi = pain_index(equity_curve)?;
	if pi < 1e-12 {
		if total_return > risk_free_rate {
			return Some(99.99);
		}
		return None;
	}
	Some((total_return - risk_free_rate) / pi)
}

/// Conditional Drawdown-at-Risk (CDaR) at confidence level `alpha`.
///
/// CDaR is the mean of the worst `(1 − alpha)` fraction of drawdowns —
/// analogous to CVaR / Expected Shortfall but applied to the drawdown
/// distribution rather than P&L.
///
/// `confidence` should be in (0, 1), e.g. 0.95 for CDaR₉₅.
pub fn conditional_drawdown_at_risk(equity_curve: &[f64], confidence: f64) -> Option<f64> {
	if equity_curve.is_empty() || confidence <= 0.0 || confidence >= 1.0 {
		return None;
	}
	let mut dd = drawdown_series(equity_curve);
	dd.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending
	let tail_count = ((1.0 - confidence) * dd.len() as f64).ceil().max(1.0) as usize;
	let tail_count = tail_count.min(dd.len());
	let tail_sum: f64 = dd[..tail_count].iter().sum();
	Some(tail_sum / tail_count as f64)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_ulcer_index_flat_curve() {
		let eq = vec![100.0; 10];
		assert_eq!(ulcer_index(&eq), Some(0.0));
	}

	#[test]
	fn test_ulcer_index_drawdown() {
		let eq = vec![100.0, 90.0, 80.0, 90.0, 100.0];
		let ui = ulcer_index(&eq).unwrap();
		assert!(ui > 0.0, "ui = {}", ui);
	}

	#[test]
	fn test_pain_index_flat() {
		let eq = vec![100.0; 10];
		assert_eq!(pain_index(&eq), Some(0.0));
	}

	#[test]
	fn test_cdar_worst_tail() {
		let eq = vec![100.0, 95.0, 90.0, 85.0, 80.0, 85.0, 90.0, 95.0, 100.0, 100.0];
		let cdar = conditional_drawdown_at_risk(&eq, 0.50).unwrap();
		// Worst 50% of drawdowns should be the deeper ones
		assert!(cdar > 0.10, "cdar = {}", cdar);
	}

	#[test]
	fn test_pain_ratio() {
		let eq = vec![100.0, 95.0, 90.0, 110.0, 120.0];
		let pr = pain_ratio(&eq, 0.0).unwrap();
		assert!(pr > 0.0, "pain_ratio = {}", pr);
	}
}

