use chrono::{Duration, Utc};
use derivatives::{DerivativeMetadata, InstrumentKind};
use signal::*;

// ═══════════════════════════════════════════════════════════════════
// SignalBuilder
// ═══════════════════════════════════════════════════════════════════

#[test]
fn builder_defaults() {
    let sig = SignalBuilder::new("b1", SignalType::Buy).build();
    assert_eq!(&*sig.id, "b1");
    assert_eq!(sig.signal_type, SignalType::Buy);
    assert_eq!(sig.strength, SignalStrength::Medium);
    assert!(sig.price.is_none());
    assert!(sig.quantity.is_none());
    assert!(!sig.is_limit);
}

#[test]
fn builder_full_chain() {
    let ts = Utc::now() - Duration::hours(1);
    let sig = SignalBuilder::new("b2", SignalType::Sell)
        .strength(SignalStrength::Strong)
        .reason(SignalReason::Momentum("breakout".into()))
        .price(105.0)
        .quantity(50.0)
        .timestamp(ts)
        .metadata("key".into(), SignalMetadata::Text("val".into()))
        .build();

    assert_eq!(&*sig.id, "b2");
    assert_eq!(sig.signal_type, SignalType::Sell);
    assert_eq!(sig.strength, SignalStrength::Strong);
    assert_eq!(sig.price, Some(105.0));
    assert_eq!(sig.quantity, Some(50.0));
    assert_eq!(sig.timestamp, ts);
    assert_eq!(
        sig.metadata.get("key"),
        Some(&SignalMetadata::Text("val".into()))
    );
}

#[test]
fn builder_no_timestamp_uses_now() {
    let before = Utc::now();
    let sig = SignalBuilder::new("b3", SignalType::Hold).build();
    let after = Utc::now();
    assert!(sig.timestamp >= before && sig.timestamp <= after);
}

// ═══════════════════════════════════════════════════════════════════
// Signal constructors (buy, sell, hold, close, two_sided_quote, cancel_quotes)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn buy_constructor() {
    let s = Signal::buy("s1", SignalStrength::Weak, SignalReason::PriceAction("dip".into()));
    assert_eq!(s.signal_type, SignalType::Buy);
    assert_eq!(s.strength, SignalStrength::Weak);
}

#[test]
fn sell_constructor() {
    let s = Signal::sell("s2", SignalStrength::Strong, SignalReason::Momentum("top".into()));
    assert_eq!(s.signal_type, SignalType::Sell);
}

#[test]
fn hold_constructor() {
    let s = Signal::hold("s3", SignalReason::RiskManagement("pause".into()));
    assert_eq!(s.signal_type, SignalType::Hold);
    assert_eq!(s.strength, SignalStrength::Medium);
}

#[test]
fn close_constructor() {
    let s = Signal::close("s4", SignalStrength::Strong, SignalReason::TechnicalAnalysis("exit".into()));
    assert_eq!(s.signal_type, SignalType::Close);
}

#[test]
fn two_sided_quote_constructor() {
    let s = Signal::two_sided_quote(
        "mm1",
        99.0, 101.0, 10.0, 10.0,
        SignalStrength::Medium,
        SignalReason::Custom("MM".into(), "spread".into()),
    );
    match &s.signal_type {
        SignalType::TwoSidedQuote { bid_price, ask_price, bid_size, ask_size } => {
            assert_eq!(*bid_price, 99.0);
            assert_eq!(*ask_price, 101.0);
            assert_eq!(*bid_size, 10.0);
            assert_eq!(*ask_size, 10.0);
        }
        _ => panic!("expected TwoSidedQuote"),
    }
}

#[test]
fn cancel_quotes_constructor() {
    let s = Signal::cancel_quotes("cq1", SignalReason::RiskManagement("risk".into()));
    assert_eq!(s.signal_type, SignalType::CancelQuotes);
    assert_eq!(s.strength, SignalStrength::Strong);
}

// ═══════════════════════════════════════════════════════════════════
// Builder methods: with_price, with_quantity, with_metadata, with_numeric,
// with_limit, with_timestamp
// ═══════════════════════════════════════════════════════════════════

#[test]
fn with_price_and_quantity() {
    let s = Signal::buy("wp", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(50.0)
        .with_quantity(100.0);
    assert_eq!(s.price, Some(50.0));
    assert_eq!(s.quantity, Some(100.0));
}

#[test]
fn with_metadata_and_numeric() {
    let s = Signal::buy("wm", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_metadata("tag".into(), SignalMetadata::Boolean(true))
        .with_numeric("confidence", 0.95);
    assert_eq!(
        s.metadata.get("tag"),
        Some(&SignalMetadata::Boolean(true))
    );
    assert_eq!(s.numeric_metadata.get("confidence"), Some(&0.95));
}

#[test]
fn with_limit() {
    let s = Signal::buy("wl", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_limit(true);
    assert!(s.is_limit);
}

#[test]
fn with_timestamp() {
    let ts = Utc::now() - Duration::days(1);
    let s = Signal::buy("wt", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_timestamp(ts);
    assert_eq!(s.timestamp, ts);
}

// ═══════════════════════════════════════════════════════════════════
// is_entry_signal / is_exit_signal / is_scaling_signal
// ═══════════════════════════════════════════════════════════════════

fn make_option_instrument() -> DerivativeMetadata {
    DerivativeMetadata::new(
        "BTC-25MAR26-100000-C", "BTC",
        InstrumentKind::Call { strike: 100000.0, expiry: Utc::now() + Duration::days(30) },
        1.0, "USD", "deribit",
    )
}

fn make_future_instrument() -> DerivativeMetadata {
    DerivativeMetadata::new(
        "BTC-PERP", "BTC",
        InstrumentKind::Perpetual,
        1.0, "USD", "deribit",
    )
}

#[test]
fn entry_signal_basic_types() {
    let buy = Signal::buy("e1", SignalStrength::Medium, SignalReason::PriceAction("b".into()));
    let sell = Signal::sell("e2", SignalStrength::Medium, SignalReason::PriceAction("s".into()));
    assert!(buy.is_entry_signal());
    assert!(sell.is_entry_signal());
    assert!(!buy.is_exit_signal());
    assert!(!buy.is_scaling_signal());
}

#[test]
fn entry_signal_two_sided_quote() {
    let s = Signal::two_sided_quote("e3", 99.0, 101.0, 5.0, 5.0,
        SignalStrength::Medium, SignalReason::Custom("MM".into(), "q".into()));
    assert!(s.is_entry_signal());
}

#[test]
fn entry_signal_derivative_types() {
    let inst = make_option_instrument();
    let types = vec![
        SignalType::BuyOption { instrument: inst.clone(), premium: Some(5.0) },
        SignalType::SellOption { instrument: inst.clone(), premium: None },
        SignalType::BuyFuture { instrument: make_future_instrument() },
        SignalType::SellFuture { instrument: make_future_instrument() },
        SignalType::OptionSpread { legs: vec![SpreadLeg {
            signal_type: SignalType::BuyOption { instrument: inst.clone(), premium: None },
            ratio: 1, limit_price: None,
        }]},
        SignalType::HedgeDelta { target_delta: 0.0, current_delta: 0.5, hedge_instrument: "BTC".into() },
    ];
    for st in types {
        let sig = Signal::new("ed", st, SignalStrength::Medium, SignalReason::PriceAction("d".into()));
        assert!(sig.is_entry_signal(), "Expected entry for {:?}", sig.signal_type);
    }
}

#[test]
fn exit_signal_types() {
    let inst = make_option_instrument();
    let types = vec![
        SignalType::Close,
        SignalType::StopLoss,
        SignalType::TakeProfit,
        SignalType::CancelQuotes,
        SignalType::ExerciseOption { instrument: inst.clone() },
        SignalType::RollContract {
            from_symbol: "X".into(), to_symbol: "Y".into(),
            from_instrument: inst.clone(), to_instrument: inst.clone(),
        },
    ];
    for st in types {
        let sig = Signal::new("ex", st, SignalStrength::Medium, SignalReason::PriceAction("x".into()));
        assert!(sig.is_exit_signal(), "Expected exit for {:?}", sig.signal_type);
        assert!(!sig.is_entry_signal());
    }
}

#[test]
fn scaling_signal_types() {
    for st in [SignalType::ScaleIn, SignalType::ScaleOut] {
        let sig = Signal::new("sc", st, SignalStrength::Medium, SignalReason::PriceAction("s".into()));
        assert!(sig.is_scaling_signal());
        assert!(!sig.is_entry_signal());
        assert!(!sig.is_exit_signal());
    }
}

#[test]
fn hold_is_none_of_the_three() {
    let s = Signal::hold("h1", SignalReason::PriceAction("wait".into()));
    assert!(!s.is_entry_signal());
    assert!(!s.is_exit_signal());
    assert!(!s.is_scaling_signal());
}

#[test]
fn custom_is_none_of_the_three() {
    let s = Signal::new("cu", SignalType::Custom("audit".into()),
        SignalStrength::Medium, SignalReason::Custom("sys".into(), "check".into()));
    assert!(!s.is_entry_signal());
    assert!(!s.is_exit_signal());
    assert!(!s.is_scaling_signal());
}

// ═══════════════════════════════════════════════════════════════════
// strength_value
// ═══════════════════════════════════════════════════════════════════

#[test]
fn strength_value_weak() {
    let s = Signal::buy("sv1", SignalStrength::Weak, SignalReason::PriceAction("w".into()));
    assert!((s.strength_value() - 0.33).abs() < 1e-9);
}

#[test]
fn strength_value_medium() {
    let s = Signal::buy("sv2", SignalStrength::Medium, SignalReason::PriceAction("m".into()));
    assert!((s.strength_value() - 0.66).abs() < 1e-9);
}

#[test]
fn strength_value_strong() {
    let s = Signal::buy("sv3", SignalStrength::Strong, SignalReason::PriceAction("s".into()));
    assert!((s.strength_value() - 1.0).abs() < 1e-9);
}

#[test]
fn strength_value_custom() {
    let s = Signal::new("sv4", SignalType::Buy, SignalStrength::Custom(0.42),
        SignalReason::PriceAction("c".into()));
    assert!((s.strength_value() - 0.42).abs() < 1e-9);
}

// ═══════════════════════════════════════════════════════════════════
// validate()
// ═══════════════════════════════════════════════════════════════════

#[test]
fn validate_valid_entry_with_price_and_quantity() {
    let s = Signal::buy("v1", SignalStrength::Medium, SignalReason::PriceAction("ok".into()))
        .with_price(100.0)
        .with_quantity(10.0);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.errors.is_empty());
    assert!(v.warnings.is_empty());
}

#[test]
fn validate_empty_id() {
    let s = Signal::buy("", SignalStrength::Medium, SignalReason::PriceAction("e".into()))
        .with_price(100.0)
        .with_quantity(10.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("ID cannot be empty")));
}

#[test]
fn validate_entry_without_price_warns() {
    let s = Signal::buy("v2", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_quantity(10.0);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.iter().any(|w| w.contains("without suggested price")));
}

#[test]
fn validate_entry_without_quantity_warns() {
    let s = Signal::buy("v3", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(100.0);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.iter().any(|w| w.contains("without suggested quantity")));
}

#[test]
fn validate_negative_price() {
    let s = Signal::buy("v4", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(-5.0)
        .with_quantity(10.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("price must be positive")));
}

#[test]
fn validate_zero_price() {
    let s = Signal::buy("v5", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(0.0)
        .with_quantity(10.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("price must be positive")));
}

#[test]
fn validate_negative_quantity() {
    let s = Signal::buy("v6", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(100.0)
        .with_quantity(-1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("quantity must be positive")));
}

#[test]
fn validate_zero_quantity() {
    let s = Signal::buy("v7", SignalStrength::Medium, SignalReason::PriceAction("b".into()))
        .with_price(100.0)
        .with_quantity(0.0);
    let v = s.validate();
    assert!(!v.is_valid);
}

#[test]
fn validate_option_spread_empty_legs() {
    let s = Signal::new("v8", SignalType::OptionSpread { legs: vec![] },
        SignalStrength::Medium, SignalReason::PriceAction("d".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("at least one leg")));
}

#[test]
fn validate_option_spread_zero_ratios() {
    let legs = vec![
        SpreadLeg {
            signal_type: SignalType::BuyOption {
                instrument: make_option_instrument(),
                premium: None,
            },
            ratio: 0,
            limit_price: None,
        },
    ];
    let s = Signal::new("v9", SignalType::OptionSpread { legs },
        SignalStrength::Medium, SignalReason::PriceAction("d".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("non-zero ratios")));
}

#[test]
fn validate_option_spread_valid_legs() {
    let legs = vec![
        SpreadLeg {
            signal_type: SignalType::BuyOption {
                instrument: make_option_instrument(),
                premium: Some(5.0),
            },
            ratio: 1,
            limit_price: Some(5.0),
        },
        SpreadLeg {
            signal_type: SignalType::SellOption {
                instrument: make_option_instrument(),
                premium: Some(3.0),
            },
            ratio: -1,
            limit_price: Some(3.0),
        },
    ];
    let s = Signal::new("v10", SignalType::OptionSpread { legs },
        SignalStrength::Medium, SignalReason::PriceAction("spread".into()))
        .with_price(2.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(v.is_valid);
}

#[test]
fn validate_hedge_delta_empty_instrument() {
    let s = Signal::new("v11",
        SignalType::HedgeDelta { target_delta: 0.0, current_delta: 0.5, hedge_instrument: "".into() },
        SignalStrength::Medium, SignalReason::PriceAction("h".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("hedge_instrument cannot be empty")));
}

#[test]
fn validate_hedge_delta_same_target_and_current_warns() {
    let s = Signal::new("v12",
        SignalType::HedgeDelta { target_delta: 0.5, current_delta: 0.5, hedge_instrument: "BTC".into() },
        SignalStrength::Medium, SignalReason::PriceAction("h".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.iter().any(|w| w.contains("no action needed")));
}

#[test]
fn validate_roll_contract_same_symbols() {
    let inst = make_option_instrument();
    let s = Signal::new("v13",
        SignalType::RollContract {
            from_symbol: "X".into(), to_symbol: "X".into(),
            from_instrument: inst.clone(), to_instrument: inst,
        },
        SignalStrength::Medium, SignalReason::PriceAction("r".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.iter().any(|e| e.contains("must differ")));
}

#[test]
fn validate_roll_contract_different_symbols() {
    let inst = make_option_instrument();
    let inst2 = make_option_instrument();
    let s = Signal::new("v14",
        SignalType::RollContract {
            from_symbol: "A".into(), to_symbol: "B".into(),
            from_instrument: inst, to_instrument: inst2,
        },
        SignalStrength::Medium, SignalReason::PriceAction("r".into()))
        .with_price(100.0)
        .with_quantity(1.0);
    let v = s.validate();
    assert!(v.is_valid);
}

#[test]
fn validate_future_timestamp_warns() {
    let future_ts = Utc::now() + Duration::seconds(60);
    let s = Signal::buy("v15", SignalStrength::Medium, SignalReason::PriceAction("f".into()))
        .with_price(100.0)
        .with_quantity(10.0)
        .with_timestamp(future_ts);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.iter().any(|w| w.contains("future")));
}

#[test]
fn validate_past_timestamp_no_warning() {
    let past_ts = Utc::now() - Duration::hours(1);
    let s = Signal::buy("v16", SignalStrength::Medium, SignalReason::PriceAction("p".into()))
        .with_price(100.0)
        .with_quantity(10.0)
        .with_timestamp(past_ts);
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.is_empty());
}

#[test]
fn validate_non_entry_without_price_no_warning() {
    let s = Signal::hold("v17", SignalReason::PriceAction("wait".into()));
    let v = s.validate();
    assert!(v.is_valid);
    assert!(v.warnings.is_empty());
}

#[test]
fn validate_multiple_errors_accumulated() {
    let s = Signal::new("",
        SignalType::OptionSpread { legs: vec![] },
        SignalStrength::Medium, SignalReason::PriceAction("bad".into()))
        .with_price(-1.0)
        .with_quantity(-1.0);
    let v = s.validate();
    assert!(!v.is_valid);
    assert!(v.errors.len() >= 3); // empty ID + negative price + negative qty + empty legs
}

// ═══════════════════════════════════════════════════════════════════
// SpreadLeg struct
// ═══════════════════════════════════════════════════════════════════

#[test]
fn spread_leg_construction() {
    let leg = SpreadLeg {
        signal_type: SignalType::BuyOption {
            instrument: make_option_instrument(),
            premium: Some(5.0),
        },
        ratio: 2,
        limit_price: Some(4.5),
    };
    assert_eq!(leg.ratio, 2);
    assert_eq!(leg.limit_price, Some(4.5));
}

#[test]
fn spread_leg_negative_ratio() {
    let leg = SpreadLeg {
        signal_type: SignalType::SellOption {
            instrument: make_option_instrument(),
            premium: None,
        },
        ratio: -1,
        limit_price: None,
    };
    assert_eq!(leg.ratio, -1);
}

// ═══════════════════════════════════════════════════════════════════
// SignalReason::description
// ═══════════════════════════════════════════════════════════════════

#[test]
fn signal_reason_descriptions() {
    let cases = vec![
        (SignalReason::TechnicalAnalysis("RSI".into()), "Technical Analysis: RSI"),
        (SignalReason::FundamentalAnalysis("PE".into()), "Fundamental Analysis: PE"),
        (SignalReason::RiskManagement("stop".into()), "Risk Management: stop"),
        (SignalReason::RegimeChange("vol".into()), "Regime Change: vol"),
        (SignalReason::VolumeAnalysis("spike".into()), "Volume Analysis: spike"),
        (SignalReason::PriceAction("breakout".into()), "Price Action: breakout"),
        (SignalReason::MeanReversion("z-score".into()), "Mean Reversion: z-score"),
        (SignalReason::Momentum("trend".into()), "Momentum: trend"),
        (SignalReason::Arbitrage("cross".into()), "Arbitrage: cross"),
        (SignalReason::Custom("My".into(), "reason".into()), "My: reason"),
    ];
    for (reason, expected) in cases {
        assert_eq!(reason.description(), expected);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SignalMetadata::as_float / as_string
// ═══════════════════════════════════════════════════════════════════

#[test]
fn metadata_as_float() {
    assert_eq!(SignalMetadata::Number(3.14).as_float(), Some(3.14));
    assert_eq!(SignalMetadata::Percentage(0.5).as_float(), Some(0.5));
    assert_eq!(SignalMetadata::Boolean(true).as_float(), Some(1.0));
    assert_eq!(SignalMetadata::Boolean(false).as_float(), Some(0.0));
    assert_eq!(SignalMetadata::Price(99.9).as_float(), Some(99.9));
    assert_eq!(SignalMetadata::Text("hi".into()).as_float(), None);
    assert_eq!(SignalMetadata::Timestamp(Utc::now()).as_float(), None);
}

#[test]
fn metadata_as_string() {
    assert_eq!(SignalMetadata::Number(1.5).as_string(), "1.5");
    assert_eq!(SignalMetadata::Text("hello".into()).as_string(), "hello");
    assert_eq!(SignalMetadata::Boolean(true).as_string(), "true");
    assert_eq!(SignalMetadata::Price(42.0).as_string(), "42");
    assert_eq!(SignalMetadata::Percentage(0.1234).as_string(), "12.34%");
}

// ═══════════════════════════════════════════════════════════════════
// CrossAsset signal type
// ═══════════════════════════════════════════════════════════════════

#[test]
fn cross_asset_signal() {
    let s = Signal::new("ca1",
        SignalType::CrossAsset {
            target_symbol: "ETH".into(),
            target_exchange: "binance".into(),
            action: Box::new(SignalType::Buy),
        },
        SignalStrength::Strong,
        SignalReason::Arbitrage("cross-exchange".into()),
    );
    match &s.signal_type {
        SignalType::CrossAsset { target_symbol, target_exchange, action } => {
            assert_eq!(target_symbol, "ETH");
            assert_eq!(target_exchange, "binance");
            assert_eq!(**action, SignalType::Buy);
        }
        _ => panic!("expected CrossAsset"),
    }
    // CrossAsset itself is not classified as entry/exit/scaling
    assert!(!s.is_entry_signal());
    assert!(!s.is_exit_signal());
    assert!(!s.is_scaling_signal());
}

// ═══════════════════════════════════════════════════════════════════
// Serialization round-trip
// ═══════════════════════════════════════════════════════════════════

#[test]
fn signal_serialize_deserialize_round_trip() {
    let s = Signal::buy("rt1", SignalStrength::Strong, SignalReason::PriceAction("b".into()))
        .with_price(100.0)
        .with_quantity(10.0)
        .with_numeric("vol", 0.3)
        .with_limit(true);

    let json = serde_json::to_string(&s).unwrap();
    let deser: Signal = serde_json::from_str(&json).unwrap();

    assert_eq!(&*deser.id, "rt1");
    assert_eq!(deser.signal_type, SignalType::Buy);
    assert_eq!(deser.strength, SignalStrength::Strong);
    assert_eq!(deser.price, Some(100.0));
    assert_eq!(deser.quantity, Some(10.0));
    assert_eq!(deser.numeric_metadata.get("vol"), Some(&0.3));
    assert!(deser.is_limit);
}

#[test]
fn derivative_signal_serialize_deserialize() {
    let inst = make_option_instrument();
    let s = Signal::new("rt2",
        SignalType::BuyOption { instrument: inst, premium: Some(5.0) },
        SignalStrength::Medium,
        SignalReason::TechnicalAnalysis("vol smile".into()),
    ).with_price(5.0).with_quantity(10.0);

    let json = serde_json::to_string(&s).unwrap();
    let deser: Signal = serde_json::from_str(&json).unwrap();

    match &deser.signal_type {
        SignalType::BuyOption { premium, instrument } => {
            assert_eq!(*premium, Some(5.0));
            assert_eq!(instrument.underlying, "BTC");
        }
        _ => panic!("expected BuyOption"),
    }
}

#[test]
fn spread_signal_serialize_deserialize() {
    let legs = vec![
        SpreadLeg {
            signal_type: SignalType::BuyOption {
                instrument: make_option_instrument(),
                premium: Some(5.0),
            },
            ratio: 1,
            limit_price: Some(5.0),
        },
        SpreadLeg {
            signal_type: SignalType::SellOption {
                instrument: make_option_instrument(),
                premium: Some(3.0),
            },
            ratio: -1,
            limit_price: Some(3.0),
        },
    ];
    let s = Signal::new("rt3",
        SignalType::OptionSpread { legs },
        SignalStrength::Strong,
        SignalReason::Custom("Spread".into(), "vertical".into()),
    ).with_price(2.0).with_quantity(1.0);

    let json = serde_json::to_string(&s).unwrap();
    let deser: Signal = serde_json::from_str(&json).unwrap();

    match &deser.signal_type {
        SignalType::OptionSpread { legs } => assert_eq!(legs.len(), 2),
        _ => panic!("expected OptionSpread"),
    }
}
