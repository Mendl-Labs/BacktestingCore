//! Integration tests for the portfoliomanager module

use portfoliomanager::{PortfolioState, Position, PositionSide};
use chrono::Utc;

#[test]
fn test_portfolio_state_default() {
    let state = PortfolioState::default();
    
    assert_eq!(state.balance, 0.0);
    assert_eq!(state.net_position, 0.0);
    assert!(state.positions.is_empty());
    assert_eq!(state.realized_pnl, 0.0);
    assert_eq!(state.unrealized_pnl, 0.0);
    assert_eq!(state.fees_paid, 0.0);
}

#[test]
fn test_portfolio_state_creation() {
    let timestamp = Utc::now();
    let state = PortfolioState {
        timestamp,
        balance: 10000.0,
        net_position: 1.0,
        positions: Vec::new(),
        realized_pnl: 100.0,
        unrealized_pnl: 50.0,
        fees_paid: 10.0,
        leverage: 1.0,
    };
    
    assert_eq!(state.balance, 10000.0);
    assert_eq!(state.net_position, 1.0);
    assert_eq!(state.realized_pnl, 100.0);
    assert_eq!(state.unrealized_pnl, 50.0);
    assert_eq!(state.fees_paid, 10.0);
    assert_eq!(state.timestamp, timestamp);
}

#[test]
fn test_position_creation() {
    let open_time = Utc::now();
    let position = Position {
        symbol: "BTC/USDT".to_string(),
        side: PositionSide::Long,
        quantity: 1.0,
        entry_price: 50000.0,
        mark_price: Some(52000.0),
        realized_pnl: 0.0,
        unrealized_pnl: 2000.0,
        open_time,
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    assert_eq!(position.symbol, "BTC/USDT");
    assert_eq!(position.side, PositionSide::Long);
    assert_eq!(position.quantity, 1.0);
    assert_eq!(position.entry_price, 50000.0);
    assert_eq!(position.mark_price, Some(52000.0));
    assert_eq!(position.realized_pnl, 0.0);
    assert_eq!(position.unrealized_pnl, 2000.0);
    assert_eq!(position.open_time, open_time);
    assert!(position.close_time.is_none());
}

#[test]
fn test_position_side_equality() {
    assert_eq!(PositionSide::Long, PositionSide::Long);
    assert_eq!(PositionSide::Short, PositionSide::Short);
    assert_ne!(PositionSide::Long, PositionSide::Short);
}

#[test]
fn test_position_side_serialization() {
    let long_side = PositionSide::Long;
    let short_side = PositionSide::Short;
    
    let long_json = serde_json::to_string(&long_side).unwrap();
    let short_json = serde_json::to_string(&short_side).unwrap();
    
    assert!(long_json.contains("Long"));
    assert!(short_json.contains("Short"));
    
    // Test deserialization
    let deserialized_long: PositionSide = serde_json::from_str(&long_json).unwrap();
    let deserialized_short: PositionSide = serde_json::from_str(&short_json).unwrap();
    
    assert_eq!(deserialized_long, PositionSide::Long);
    assert_eq!(deserialized_short, PositionSide::Short);
}

#[test]
fn test_portfolio_state_serialization() {
    let timestamp = Utc::now();
    let state = PortfolioState {
        timestamp,
        balance: 10000.0,
        net_position: 1.0,
        positions: Vec::new(),
        realized_pnl: 100.0,
        unrealized_pnl: 50.0,
        fees_paid: 10.0,
        leverage: 1.0,
    };
    
    let json_str = serde_json::to_string(&state).unwrap();
    assert!(json_str.contains("balance"));
    assert!(json_str.contains("net_position"));
    assert!(json_str.contains("realized_pnl"));
    
    // Test deserialization
    let deserialized: PortfolioState = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.balance, state.balance);
    assert_eq!(deserialized.net_position, state.net_position);
    assert_eq!(deserialized.realized_pnl, state.realized_pnl);
}

#[test]
fn test_position_serialization() {
    let position = Position {
        symbol: "ETH/USDT".to_string(),
        side: PositionSide::Short,
        quantity: 5.0,
        entry_price: 3000.0,
        mark_price: Some(2900.0),
        realized_pnl: 0.0,
        unrealized_pnl: 500.0,
        open_time: Utc::now(),
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    let json_str = serde_json::to_string(&position).unwrap();
    assert!(json_str.contains("ETH/USDT"));
    assert!(json_str.contains("Short"));
    assert!(json_str.contains("quantity"));
    
    // Test deserialization
    let deserialized: Position = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.symbol, position.symbol);
    assert_eq!(deserialized.side, position.side);
    assert_eq!(deserialized.quantity, position.quantity);
}

#[test]
fn test_portfolio_with_positions() {
    let position1 = Position {
        symbol: "BTC/USDT".to_string(),
        side: PositionSide::Long,
        quantity: 1.0,
        entry_price: 50000.0,
        mark_price: Some(52000.0),
        realized_pnl: 0.0,
        unrealized_pnl: 2000.0,
        open_time: Utc::now(),
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    let position2 = Position {
        symbol: "ETH/USDT".to_string(),
        side: PositionSide::Short,
        quantity: 10.0,
        entry_price: 3000.0,
        mark_price: Some(2900.0),
        realized_pnl: 0.0,
        unrealized_pnl: 1000.0,
        open_time: Utc::now(),
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    let state = PortfolioState {
        timestamp: Utc::now(),
        balance: 10000.0,
        net_position: 1.0,
        positions: vec![position1, position2],
        realized_pnl: 0.0,
        unrealized_pnl: 3000.0,
        fees_paid: 20.0,
        leverage: 1.0,
    };
    
    assert_eq!(state.positions.len(), 2);
    assert_eq!(state.positions[0].symbol, "BTC/USDT");
    assert_eq!(state.positions[1].symbol, "ETH/USDT");
    assert_eq!(state.positions[0].side, PositionSide::Long);
    assert_eq!(state.positions[1].side, PositionSide::Short);
}

#[test]
fn test_closed_position() {
    let open_time = Utc::now();
    let close_time = Utc::now();
    
    let position = Position {
        symbol: "BTC/USDT".to_string(),
        side: PositionSide::Long,
        quantity: 1.0,
        entry_price: 50000.0,
        mark_price: Some(55000.0),
        realized_pnl: 5000.0,
        unrealized_pnl: 0.0,
        open_time,
        close_time: Some(close_time),
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    assert_eq!(position.realized_pnl, 5000.0);
    assert_eq!(position.unrealized_pnl, 0.0);
    assert!(position.close_time.is_some());
    assert_eq!(position.close_time.unwrap(), close_time);
}

#[test]
fn test_f64_operations() {
    let balance = 10000.0;
    let pnl = 500.0;
    
    // Test basic operations
    let new_balance = balance + pnl;
    assert_eq!(new_balance, 10500.0);
    
    // Test in position
    let position = Position {
        symbol: "TEST/USDT".to_string(),
        side: PositionSide::Long,
        quantity: 1.0,
        entry_price: 100.0,
        mark_price: Some(110.0),
        realized_pnl: 0.0,
        unrealized_pnl: 10.0,
        open_time: Utc::now(),
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    };
    
    assert_eq!(position.entry_price, 100.0);
    assert_eq!(position.mark_price, Some(110.0));
}