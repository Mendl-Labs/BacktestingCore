//! Trade statistics for backtested strategies
//
// Includes win rate, average trade duration.

/// Calculates the win rate from a list of trade returns.
pub fn win_rate(trade_returns: &[f64]) -> Option<f64> {
	if trade_returns.is_empty() {
		return None;
	}
	let wins = trade_returns.iter().filter(|&&x| x > 0.0).count();
	Some(wins as f64 / trade_returns.len() as f64)
}

/// Calculates the average trade duration from a list of trade durations (in seconds).
pub fn average_trade_duration(trade_durations: &[f64]) -> Option<f64> {
	if trade_durations.is_empty() {
		return None;
	}
	Some(trade_durations.iter().sum::<f64>() / trade_durations.len() as f64)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn win_rate_empty_none() {
		assert!(win_rate(&[]).is_none());
	}

	#[test]
	fn win_rate_all_wins() {
		assert!((win_rate(&[1.0, 2.0, 0.5]).unwrap() - 1.0).abs() < 1e-10);
	}

	#[test]
	fn win_rate_all_losses() {
		assert!((win_rate(&[-1.0, -0.5]).unwrap()).abs() < 1e-10);
	}

	#[test]
	fn win_rate_mixed() {
		// 2 wins out of 4 (zero is NOT a win)
		let wr = win_rate(&[1.0, -1.0, 0.0, 2.0]).unwrap();
		assert!((wr - 0.5).abs() < 1e-10);
	}

	#[test]
	fn average_trade_duration_empty_none() {
		assert!(average_trade_duration(&[]).is_none());
	}

	#[test]
	fn average_trade_duration_single() {
		assert!((average_trade_duration(&[120.0]).unwrap() - 120.0).abs() < 1e-10);
	}

	#[test]
	fn average_trade_duration_multiple() {
		let avg = average_trade_duration(&[60.0, 120.0, 180.0]).unwrap();
		assert!((avg - 120.0).abs() < 1e-10);
	}
}

