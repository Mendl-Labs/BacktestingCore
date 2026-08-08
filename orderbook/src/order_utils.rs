//! Utilities for order book and signal conversion

use chrono::{DateTime, Utc};
use signal::{Signal, SignalType};
use crate::types::{OrderBookEvent, BookSide};

/// Convert a Signal to an OrderBookEvent::NewOrder
pub fn signal_to_order_event(signal: &Signal, timestamp: DateTime<Utc>) -> OrderBookEvent {
    OrderBookEvent::NewOrder {
        order_id: signal.id.clone(),
        side: match signal.signal_type {
            SignalType::Buy => BookSide::Bid,
            SignalType::Sell => BookSide::Ask,
            _ => BookSide::Bid, // Default for other signal types
        },
        price: signal.price.unwrap_or(0.0),
        quantity: signal.quantity.unwrap_or(0.0),
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signal::{SignalStrength, SignalReason};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_signal(signal_type: SignalType, price: Option<f64>, quantity: Option<f64>) -> Signal {
        Signal {
            id: Arc::from("test-signal-1"),
            timestamp: Utc::now(),
            signal_type,
            strength: SignalStrength::Strong,
            price,
            quantity,
            reason: SignalReason::Custom("test".to_string(), "test_detail".to_string()),
            metadata: HashMap::new(),
            numeric_metadata: HashMap::new(),
            is_limit: false,
        }
    }

    #[test]
    fn test_buy_signal_maps_to_bid() {
        let signal = make_signal(SignalType::Buy, Some(100.0), Some(1.0));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { side, price, quantity, .. } => {
                assert_eq!(side, BookSide::Bid);
                assert_eq!(price, 100.0);
                assert_eq!(quantity, 1.0);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_sell_signal_maps_to_ask() {
        let signal = make_signal(SignalType::Sell, Some(200.5), Some(2.5));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { side, price, quantity, .. } => {
                assert_eq!(side, BookSide::Ask);
                assert_eq!(price, 200.5);
                assert_eq!(quantity, 2.5);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_hold_signal_defaults_to_bid() {
        let signal = make_signal(SignalType::Hold, Some(50.0), Some(0.5));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { side, .. } => {
                assert_eq!(side, BookSide::Bid);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_close_signal_defaults_to_bid() {
        let signal = make_signal(SignalType::Close, Some(50.0), Some(0.5));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { side, .. } => {
                assert_eq!(side, BookSide::Bid);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_none_price_defaults_to_zero() {
        let signal = make_signal(SignalType::Buy, None, Some(1.0));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { price, .. } => {
                assert_eq!(price, 0.0);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_none_quantity_defaults_to_zero() {
        let signal = make_signal(SignalType::Sell, Some(100.0), None);
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { quantity, .. } => {
                assert_eq!(quantity, 0.0);
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_order_id_matches_signal_id() {
        let signal = make_signal(SignalType::Buy, Some(100.0), Some(1.0));
        let ts = Utc::now();
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { order_id, .. } => {
                assert_eq!(&*order_id, "test-signal-1");
            }
            _ => panic!("Expected NewOrder"),
        }
    }

    #[test]
    fn test_timestamp_propagation() {
        let signal = make_signal(SignalType::Buy, Some(100.0), Some(1.0));
        let ts = chrono::DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z").unwrap().with_timezone(&Utc);
        let event = signal_to_order_event(&signal, ts);
        match event {
            OrderBookEvent::NewOrder { timestamp, .. } => {
                assert_eq!(timestamp, ts);
            }
            _ => panic!("Expected NewOrder"),
        }
    }
}
