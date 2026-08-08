//! Additional orderbook tests covering Level operations, market order execution, and edge cases

use orderbook::{
    OrderBook, Level, BookSide, OrderInfo, SnapshotLevel,
    OrderBookConfig, OrderBookEvent,
};
use orderbook::types::OrderType;
use chrono::Utc;
use std::sync::Arc;

fn default_config() -> OrderBookConfig {
    OrderBookConfig::default()
}

fn new_order_event(id: &str, side: BookSide, price: f64, qty: f64) -> OrderBookEvent {
    OrderBookEvent::NewOrder {
        order_id: Arc::from(id),
        side,
        price,
        quantity: qty,
        timestamp: Utc::now(),
    }
}

fn make_order(id: &str, side: BookSide, price: f64, qty: f64) -> OrderInfo {
    OrderInfo {
        order_id: Arc::from(id),
        side,
        original_quantity: qty,
        remaining_quantity: qty,
        price,
        timestamp: Utc::now(),
        order_type: OrderType::Limit,
        liquidity_type: None,
    }
}

// ============================================================================
// LEVEL TESTS
// ============================================================================

#[test]
fn test_level_new_with_tracking() {
    let level = Level::new(100.0, BookSide::Bid, true);
    assert!((level.price - 100.0).abs() < f64::EPSILON);
    assert!(level.orders.is_some());
    assert!(level.is_empty());
    assert_eq!(level.order_count, 0);
    assert!((level.total_quantity - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_level_new_without_tracking() {
    let level = Level::new(100.0, BookSide::Ask, false);
    assert!(level.orders.is_none());
}

#[test]
fn test_level_add_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    let order = make_order("o1", BookSide::Bid, 100.0, 5.0);
    level.add_order(order).unwrap();
    
    assert_eq!(level.order_count, 1);
    assert!((level.total_quantity - 5.0).abs() < f64::EPSILON);
    assert!(!level.is_empty());
}

#[test]
fn test_level_add_multiple_orders() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    level.add_order(make_order("o2", BookSide::Bid, 100.0, 3.0)).unwrap();
    
    assert_eq!(level.order_count, 2);
    assert!((level.total_quantity - 8.0).abs() < f64::EPSILON);
}

#[test]
fn test_level_add_order_wrong_price() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    let order = make_order("o1", BookSide::Bid, 99.0, 5.0);
    assert!(level.add_order(order).is_err());
}

#[test]
fn test_level_add_order_wrong_side() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    let order = make_order("o1", BookSide::Ask, 100.0, 5.0);
    assert!(level.add_order(order).is_err());
}

#[test]
fn test_level_remove_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    let removed = level.remove_order("o1", Utc::now()).unwrap();
    assert!(removed.is_some());
    assert!(level.is_empty());
    assert_eq!(level.order_count, 0);
}

#[test]
fn test_level_remove_nonexistent_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    let removed = level.remove_order("o_nonexistent", Utc::now()).unwrap();
    assert!(removed.is_none());
    // Quantities should not change
    assert_eq!(level.order_count, 1);
}

#[test]
fn test_level_modify_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    level.modify_order("o1", 8.0, Utc::now()).unwrap();
    assert!((level.total_quantity - 8.0).abs() < f64::EPSILON);
}

#[test]
fn test_level_modify_nonexistent_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    assert!(level.modify_order("o_nonexistent", 3.0, Utc::now()).is_err());
}

#[test]
fn test_level_execute_trade_full_fill() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    let executed = level.execute_trade(5.0, Utc::now()).unwrap();
    assert_eq!(executed.len(), 1);
    assert!(level.is_empty());
}

#[test]
fn test_level_execute_trade_partial_fill() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 10.0)).unwrap();
    
    let executed = level.execute_trade(3.0, Utc::now()).unwrap();
    assert!(!level.is_empty());
    assert!((level.total_quantity - 7.0).abs() < f64::EPSILON);
}

#[test]
fn test_level_execute_trade_across_multiple_orders() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 3.0)).unwrap();
    level.add_order(make_order("o2", BookSide::Bid, 100.0, 4.0)).unwrap();
    level.add_order(make_order("o3", BookSide::Bid, 100.0, 5.0)).unwrap();
    
    // Execute 5 units — should fully fill o1(3) and partially fill o2(2 of 4)
    let executed = level.execute_trade(5.0, Utc::now()).unwrap();
    assert!(executed.len() >= 1);
    assert!((level.total_quantity - 7.0).abs() < f64::EPSILON);
}

#[test]
fn test_level_get_best_order() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    level.add_order(make_order("o2", BookSide::Bid, 100.0, 3.0)).unwrap();
    
    let best = level.get_best_order().unwrap();
    // FIFO — first order should be best
    assert_eq!(&*best.order_id, "o1");
}

#[test]
fn test_level_get_orders() {
    let mut level = Level::new(100.0, BookSide::Bid, true);
    level.add_order(make_order("o1", BookSide::Bid, 100.0, 5.0)).unwrap();
    level.add_order(make_order("o2", BookSide::Bid, 100.0, 3.0)).unwrap();
    
    let orders = level.get_orders();
    assert_eq!(orders.len(), 2);
}

// ============================================================================
// ORDERBOOK — ADD/CANCEL/MODIFY
// ============================================================================

#[tokio::test]
async fn test_orderbook_add_and_best_bid_ask() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 1.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 1.0)).unwrap();
    
    assert!((book.best_bid().unwrap() - 100.0).abs() < f64::EPSILON);
    assert!((book.best_ask().unwrap() - 101.0).abs() < f64::EPSILON);
    assert!((book.spread().unwrap() - 1.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_orderbook_empty_best_bid_ask() {
    let book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    assert!(book.best_bid().is_none());
    assert!(book.best_ask().is_none());
    assert!(book.spread().is_none());
}

#[tokio::test]
async fn test_orderbook_cancel_order() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 5.0)).unwrap();
    book.apply_event(OrderBookEvent::CancelOrder {
        order_id: Arc::from("b1"),
        timestamp: Utc::now(),
    }).unwrap();
    
    assert!(book.best_bid().is_none());
}

#[tokio::test]
async fn test_orderbook_modify_order() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 5.0)).unwrap();
    book.apply_event(OrderBookEvent::ModifyOrder {
        order_id: Arc::from("b1"),
        new_quantity: 10.0,
        timestamp: Utc::now(),
    }).unwrap();
    
    assert!((book.best_bid().unwrap() - 100.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_orderbook_is_crossed() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 1.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 1.0)).unwrap();
    assert!(!book.is_crossed());
}

#[tokio::test]
async fn test_orderbook_multiple_levels() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 1.0)).unwrap();
    book.apply_event(new_order_event("b2", BookSide::Bid, 99.0, 2.0)).unwrap();
    book.apply_event(new_order_event("b3", BookSide::Bid, 98.0, 3.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 1.0)).unwrap();
    book.apply_event(new_order_event("a2", BookSide::Ask, 102.0, 2.0)).unwrap();
    
    assert!((book.best_bid().unwrap() - 100.0).abs() < f64::EPSILON);
    assert!((book.best_ask().unwrap() - 101.0).abs() < f64::EPSILON);
}

// ============================================================================
// ORDERBOOK — SIMULATE/EXECUTE MARKET ORDERS
// ============================================================================

#[tokio::test]
async fn test_simulate_market_buy() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 5.0)).unwrap();
    book.apply_event(new_order_event("a2", BookSide::Ask, 102.0, 5.0)).unwrap();
    
    // BookSide::Bid = buying from asks
    let execution = book.simulate_market_order(BookSide::Bid, 3.0);
    assert!((execution.executed_quantity - 3.0).abs() < f64::EPSILON);
    assert!((execution.vwap - 101.0).abs() < 0.1);
}

#[tokio::test]
async fn test_simulate_market_buy_across_levels() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 2.0)).unwrap();
    book.apply_event(new_order_event("a2", BookSide::Ask, 102.0, 3.0)).unwrap();
    
    // BookSide::Bid = buying from asks
    let execution = book.simulate_market_order(BookSide::Bid, 5.0);
    assert!((execution.executed_quantity - 5.0).abs() < f64::EPSILON);
    // VWAP: (2*101 + 3*102) / 5 = 101.6
    assert!((execution.vwap - 101.6).abs() < 0.1);
}

#[tokio::test]
async fn test_simulate_market_order_empty_book() {
    let book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    let execution = book.simulate_market_order(BookSide::Bid, 5.0);
    assert!((execution.executed_quantity - 0.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_execute_market_order_removes_liquidity() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 5.0)).unwrap();
    
    // BookSide::Bid = buying from asks
    let execution = book.execute_market_order(BookSide::Bid, 3.0, Utc::now()).unwrap();
    assert!((execution.executed_quantity - 3.0).abs() < f64::EPSILON);
    
    // Should be 2.0 remaining at 101
    let sim = book.simulate_market_order(BookSide::Bid, 10.0);
    assert!((sim.executed_quantity - 2.0).abs() < f64::EPSILON);
}

// ============================================================================
// ORDERBOOK — SNAPSHOT
// ============================================================================

#[tokio::test]
async fn test_apply_snapshot() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    let bids = vec![
        SnapshotLevel { price: 100.0, quantity: 5.0, order_count: 2 },
        SnapshotLevel { price: 99.0, quantity: 3.0, order_count: 1 },
    ];
    let asks = vec![
        SnapshotLevel { price: 101.0, quantity: 4.0, order_count: 2 },
    ];
    
    book.apply_event(OrderBookEvent::Snapshot {
        bids,
        asks,
        timestamp: Utc::now(),
    }).unwrap();
    
    assert!((book.best_bid().unwrap() - 100.0).abs() < f64::EPSILON);
    assert!((book.best_ask().unwrap() - 101.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_apply_snapshot_replaces_existing() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(OrderBookEvent::Snapshot {
        bids: vec![SnapshotLevel { price: 100.0, quantity: 5.0, order_count: 1 }],
        asks: vec![SnapshotLevel { price: 101.0, quantity: 5.0, order_count: 1 }],
        timestamp: Utc::now(),
    }).unwrap();
    
    book.apply_event(OrderBookEvent::Snapshot {
        bids: vec![SnapshotLevel { price: 200.0, quantity: 10.0, order_count: 1 }],
        asks: vec![SnapshotLevel { price: 201.0, quantity: 10.0, order_count: 1 }],
        timestamp: Utc::now(),
    }).unwrap();
    
    assert!((book.best_bid().unwrap() - 200.0).abs() < f64::EPSILON);
    assert!((book.best_ask().unwrap() - 201.0).abs() < f64::EPSILON);
}

// ============================================================================
// ORDERBOOK — MARKET DEPTH
// ============================================================================

#[tokio::test]
async fn test_market_depth() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 5.0)).unwrap();
    book.apply_event(new_order_event("b2", BookSide::Bid, 99.0, 3.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 4.0)).unwrap();
    
    let depth = book.get_market_depth(Some(5));
    assert_eq!(depth.bids.len(), 2);
    assert_eq!(depth.asks.len(), 1);
    assert!(depth.spread.is_some());
    assert!((depth.spread.unwrap() - 1.0).abs() < f64::EPSILON);
}

// ============================================================================
// BOOKSIDE TYPE TESTS
// ============================================================================

#[test]
fn test_bookside_opposite() {
    assert_eq!(BookSide::Bid.opposite(), BookSide::Ask);
    assert_eq!(BookSide::Ask.opposite(), BookSide::Bid);
}

#[test]
fn test_bookside_from_str() {
    assert_eq!(BookSide::from_str("bid"), Some(BookSide::Bid));
    assert_eq!(BookSide::from_str("ask"), Some(BookSide::Ask));
    assert_eq!(BookSide::from_str("BID"), Some(BookSide::Bid));
    assert_eq!(BookSide::from_str("ASK"), Some(BookSide::Ask));
    assert_eq!(BookSide::from_str("buy"), Some(BookSide::Bid));
    assert_eq!(BookSide::from_str("sell"), Some(BookSide::Ask));
    assert!(BookSide::from_str("unknown").is_none());
}

// ============================================================================
// MATCHING ENGINE
// ============================================================================

#[tokio::test]
async fn test_match_all_orders_crossed_book() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    let now = Utc::now();
    
    // Create crossed book: bid > ask
    book.apply_event(new_order_event("b1", BookSide::Bid, 102.0, 2.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 100.0, 3.0)).unwrap();
    
    let fills = book.match_all_orders(now);
    assert!(!fills.is_empty(), "Crossed book should produce fills");
}

#[tokio::test]
async fn test_match_all_orders_no_cross() {
    let mut book = OrderBook::new("BTCUSD".to_string(), "test".to_string(), default_config());
    let now = Utc::now();
    
    book.apply_event(new_order_event("b1", BookSide::Bid, 100.0, 2.0)).unwrap();
    book.apply_event(new_order_event("a1", BookSide::Ask, 101.0, 3.0)).unwrap();
    
    let fills = book.match_all_orders(now);
    assert!(fills.is_empty(), "Non-crossed book should produce no fills");
}
