//! Reference signal construction for funding-rate arbitrage using
//! perpetuals versus dated futures.
//!
//! Run with:
//! cargo run -p backtest --example funding_rate_arbitrage

use chrono::{Duration, Utc};
use signal::{
    DerivativeMetadata, InstrumentKind, Signal, SignalReason, SignalStrength, SignalType,
    SpreadLeg,
};
use std::collections::HashMap;

fn main() {
    let quarterly_future = DerivativeMetadata {
        symbol: "BTC-30JUN26".to_string(),
        underlying: "BTC".to_string(),
        instrument_kind: InstrumentKind::Future {
            expiry: Utc::now() + Duration::days(90),
        },
        contract_multiplier: 1.0,
        settlement_currency: "USD".to_string(),
        exchange: "massive".to_string(),
    };

    let perp = DerivativeMetadata {
        symbol: "BTC-PERPETUAL".to_string(),
        underlying: "BTC".to_string(),
        instrument_kind: InstrumentKind::Perpetual,
        contract_multiplier: 1.0,
        settlement_currency: "USD".to_string(),
        exchange: "massive".to_string(),
    };

    let spread_signal = Signal {
        id: "funding_arb_1".into(),
        timestamp: Utc::now(),
        signal_type: SignalType::OptionSpread {
            legs: vec![
                SpreadLeg {
                    signal_type: SignalType::SellFuture {
                        instrument: perp,
                    },
                    ratio: 1,
                    limit_price: None,
                },
                SpreadLeg {
                    signal_type: SignalType::BuyFuture {
                        instrument: quarterly_future,
                    },
                    ratio: 1,
                    limit_price: None,
                },
            ],
        },
        strength: SignalStrength::Strong,
        price: None,
        quantity: Some(1.0),
        reason: SignalReason::Custom(
            "funding_arb".into(),
            "Capture positive perp funding while offsetting delta with dated future".into(),
        ),
        metadata: HashMap::new(),
        numeric_metadata: HashMap::new(),
        is_limit: false,
    };

    println!("Funding-rate arb spread valid: {}", spread_signal.validate().is_valid);
    println!("Signal uses all-or-nothing multi-leg semantics.");
}
