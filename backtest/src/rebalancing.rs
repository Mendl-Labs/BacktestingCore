use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RebalanceStrategy {
    None,
    Threshold { drift_pct: f64 },
    Calendar { frequency: RebalanceFreq },
    ThresholdOrCalendar { drift_pct: f64, frequency: RebalanceFreq },
}

impl Default for RebalanceStrategy {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RebalanceFreq {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
}

impl RebalanceFreq {
    pub fn interval_seconds(&self) -> i64 {
        match self {
            Self::Daily => 86_400,
            Self::Weekly => 604_800,
            Self::Monthly => 2_592_000,
            Self::Quarterly => 7_776_000,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RebalanceTrigger {
    Drift,
    Calendar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceTrade {
    pub symbol: String,
    pub side: RebalanceSide,
    pub notional: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RebalanceSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceEvent {
    pub timestamp_idx: usize,
    pub trigger: RebalanceTrigger,
    pub trades: Vec<RebalanceTrade>,
    pub total_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceSimResult {
    pub adjusted_equity_curve: Vec<f64>,
    pub events: Vec<RebalanceEvent>,
    pub total_rebalance_cost: f64,
    pub num_rebalances: usize,
    pub weight_drift_series: Vec<Vec<f64>>,
}

pub fn simulate_rebalancing(
    equity_curves: &[Vec<f64>],
    target_weights: &[f64],
    strategy: &RebalanceStrategy,
    maker_fee: f64,
    slippage_bps: f64,
    timestamps_per_day: usize,
) -> RebalanceSimResult {
    if matches!(strategy, RebalanceStrategy::None) || equity_curves.is_empty() {
        let max_len = equity_curves.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut combined = vec![0.0; max_len];
        for curve in equity_curves {
            for (i, v) in curve.iter().enumerate() {
                combined[i] += v;
            }
            if let Some(&last) = curve.last() {
                for i in curve.len()..max_len {
                    combined[i] += last;
                }
            }
        }
        return RebalanceSimResult {
            adjusted_equity_curve: combined,
            events: Vec::new(),
            total_rebalance_cost: 0.0,
            num_rebalances: 0,
            weight_drift_series: Vec::new(),
        };
    }

    let n_assets = equity_curves.len();
    let max_len = equity_curves.iter().map(|c| c.len()).max().unwrap_or(0);

    let mut asset_values: Vec<f64> = (0..n_assets)
        .map(|i| equity_curves[i].first().copied().unwrap_or(0.0))
        .collect();

    let mut combined_equity = Vec::with_capacity(max_len);
    let mut events = Vec::new();
    let mut weight_drift_series: Vec<Vec<f64>> = Vec::with_capacity(max_len);
    let mut total_rebalance_cost = 0.0;
    let mut last_rebalance_idx: usize = 0;

    let calendar_interval = match strategy {
        RebalanceStrategy::Calendar { frequency } |
        RebalanceStrategy::ThresholdOrCalendar { frequency, .. } => {
            frequency.interval_seconds() as usize / (86_400 / timestamps_per_day.max(1))
        }
        _ => usize::MAX,
    };

    let drift_threshold = match strategy {
        RebalanceStrategy::Threshold { drift_pct } |
        RebalanceStrategy::ThresholdOrCalendar { drift_pct, .. } => *drift_pct,
        _ => f64::MAX,
    };

    for idx in 0..max_len {
        for i in 0..n_assets {
            let curve = &equity_curves[i];
            if idx < curve.len() {
                asset_values[i] = curve[idx];
            }
        }

        let total_value: f64 = asset_values.iter().sum();
        combined_equity.push(total_value);

        if total_value <= 0.0 {
            weight_drift_series.push(vec![0.0; n_assets]);
            continue;
        }

        let actual_weights: Vec<f64> = asset_values.iter().map(|v| v / total_value).collect();
        let drifts: Vec<f64> = actual_weights.iter().zip(target_weights.iter())
            .map(|(a, t)| (a - t).abs())
            .collect();
        let max_drift = drifts.iter().cloned().fold(0.0_f64, f64::max);
        weight_drift_series.push(drifts);

        let should_rebalance = if idx == 0 {
            false
        } else {
            let drift_triggered = max_drift >= drift_threshold;
            let calendar_triggered = (idx - last_rebalance_idx) >= calendar_interval;
            match strategy {
                RebalanceStrategy::Threshold { .. } => drift_triggered,
                RebalanceStrategy::Calendar { .. } => calendar_triggered,
                RebalanceStrategy::ThresholdOrCalendar { .. } => drift_triggered || calendar_triggered,
                RebalanceStrategy::None => false,
            }
        };

        if should_rebalance {
            let trigger = if max_drift >= drift_threshold {
                RebalanceTrigger::Drift
            } else {
                RebalanceTrigger::Calendar
            };

            let mut trades = Vec::new();
            let mut event_cost = 0.0;

            for i in 0..n_assets {
                let target_value = total_value * target_weights[i];
                let diff = target_value - asset_values[i];

                if diff.abs() < 1.0 {
                    continue;
                }

                let side = if diff > 0.0 { RebalanceSide::Buy } else { RebalanceSide::Sell };
                let notional = diff.abs();
                let price = if !equity_curves[i].is_empty() && idx < equity_curves[i].len() {
                    equity_curves[i][idx] / target_weights[i].max(0.001)
                } else {
                    1.0
                };
                let commission = notional * maker_fee;
                let slippage = notional * slippage_bps / 10_000.0;
                let cost = commission + slippage;

                event_cost += cost;
                trades.push(RebalanceTrade {
                    symbol: format!("asset_{}", i),
                    side,
                    notional,
                    price,
                    commission,
                    slippage,
                });

                asset_values[i] = target_value - if diff > 0.0 { cost / 2.0 } else { -cost / 2.0 };
            }

            total_rebalance_cost += event_cost;
            last_rebalance_idx = idx;

            events.push(RebalanceEvent {
                timestamp_idx: idx,
                trigger,
                trades,
                total_cost: event_cost,
            });

            let adjusted_total: f64 = asset_values.iter().sum();
            if let Some(last) = combined_equity.last_mut() {
                *last = adjusted_total;
            }
        }
    }

    let num_rebalances = events.len();

    RebalanceSimResult {
        adjusted_equity_curve: combined_equity,
        events,
        total_rebalance_cost,
        num_rebalances,
        weight_drift_series,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_rebalancing_returns_summed_curves() {
        let curves = vec![
            vec![5000.0, 5100.0, 5200.0],
            vec![5000.0, 4900.0, 4800.0],
        ];
        let weights = vec![0.5, 0.5];
        let result = simulate_rebalancing(&curves, &weights, &RebalanceStrategy::None, 0.001, 5.0, 1);

        assert_eq!(result.adjusted_equity_curve, vec![10000.0, 10000.0, 10000.0]);
        assert_eq!(result.num_rebalances, 0);
    }

    #[test]
    fn threshold_triggers_when_drift_exceeded() {
        let curves = vec![
            vec![5000.0, 6000.0, 7000.0, 8000.0],
            vec![5000.0, 4000.0, 3000.0, 2000.0],
        ];
        let weights = vec![0.5, 0.5];
        let strategy = RebalanceStrategy::Threshold { drift_pct: 0.05 };
        let result = simulate_rebalancing(&curves, &weights, &strategy, 0.001, 5.0, 1);

        assert!(result.num_rebalances > 0);
        assert!(result.total_rebalance_cost > 0.0);
    }

    #[test]
    fn calendar_triggers_at_interval() {
        let curve_a: Vec<f64> = (0..100).map(|i| 5000.0 + i as f64 * 10.0).collect();
        let curve_b: Vec<f64> = (0..100).map(|i| 5000.0 - i as f64 * 5.0).collect();
        let curves = vec![curve_a, curve_b];
        let weights = vec![0.5, 0.5];
        let strategy = RebalanceStrategy::Calendar { frequency: RebalanceFreq::Daily };
        let result = simulate_rebalancing(&curves, &weights, &strategy, 0.001, 5.0, 1);

        assert!(result.num_rebalances > 0);
    }
}
