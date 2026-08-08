//! Reference signal construction for a delta-hedged covered call workflow.
//!
//! Run with:
//! cargo run -p backtest --example delta_hedged_covered_call

use chrono::{Duration, Utc};
use signal::{
    DerivativeMetadata, InstrumentKind, Signal, SignalReason, SignalStrength, SignalType,
};
use std::collections::HashMap;

fn main() {
    let expiry = Utc::now() + Duration::days(30);
    let covered_call = DerivativeMetadata {
        symbol: "BTC-30APR26-90000-C".to_string(),
        underlying: "BTC".to_string(),
        instrument_kind: InstrumentKind::Call {
            strike: 90_000.0,
            expiry,
        },
        contract_multiplier: 1.0,
        settlement_currency: "USD".to_string(),
        exchange: "massive".to_string(),
    };

    let short_call_signal = Signal {
        id: "covered_call_1".into(),
        timestamp: Utc::now(),
        signal_type: SignalType::SellOption {
            instrument: covered_call,
            premium: Some(2_250.0),
        },
        strength: SignalStrength::Strong,
        price: Some(2_250.0),
        quantity: Some(1.0),
        reason: SignalReason::Custom(
            "covered_call".into(),
            "Harvest option premium against spot inventory".into(),
        ),
        metadata: HashMap::new(),
        numeric_metadata: HashMap::new(),
        is_limit: true,
    };

    let delta_hedge_signal = Signal {
        id: "delta_hedge_1".into(),
        timestamp: Utc::now(),
        signal_type: SignalType::HedgeDelta {
            target_delta: 0.0,
            current_delta: -0.42,
            hedge_instrument: "BTC-PERPETUAL".to_string(),
        },
        strength: SignalStrength::Medium,
        price: None,
        quantity: Some(0.42),
        reason: SignalReason::Custom(
            "delta_hedge".into(),
            "Neutralize short call directional exposure".into(),
        ),
        metadata: HashMap::new(),
        numeric_metadata: HashMap::new(),
        is_limit: false,
    };

    println!("Covered-call signal valid: {}", short_call_signal.validate().is_valid);
    println!("Delta-hedge signal valid: {}", delta_hedge_signal.validate().is_valid);
    println!("Signals ready for execution pipeline.");
}
