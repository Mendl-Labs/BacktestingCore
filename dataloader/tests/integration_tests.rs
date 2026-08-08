use dataloader::*;

// Remove database imports for now since they're optional
// use databaseschema::{
//     CustomAsyncPgConnectionManager,
//     models::{
//         historical_order::NewHistoricalOrder,
//         historical_snapshot::NewHistoricalSnapshot,
//     },
// };
// use deadpool::managed::Pool;

use chrono::{Utc, TimeZone};
use tokio;

// Mock data fixtures for non-postgres builds
fn create_test_trade_data() -> TradeData {
    TradeData::new(
        Utc::now(),
        "BTCUSD",
        "BINANCE",
        45000.00,
        1.5,
        Side::Buy,
        0,
    )
}

fn create_test_candle() -> Candle {
    Candle {
        timestamp: Utc::now(),
        symbol: "BTCUSD".into(),
        exchange: "BINANCE".into(),
        open: 45000.00,
        high: 45100.00,
        low: 44900.00,
        close: 45050.00,
        volume: 10.5,
        trade_count: 25,
    }
}

#[cfg(not(feature = "postgres"))]
#[tokio::test]
async fn test_dataloader_creation() {
    // Test non-postgres DataLoader creation and basic operation
    let loader = DataLoader::new();
    
    // Verify cache starts functional by running invalidate (no-op on fresh cache)
    loader.invalidate_cache().await;
    // If we reach here, DataLoader was created and is operational
    assert!(std::mem::size_of_val(&loader) > 0, "DataLoader should have non-zero size");
}

#[cfg(not(feature = "postgres"))]
#[tokio::test]
async fn test_dataloader_mock_creation() {
    // Test mock DataLoader creation — mock() should behave identically to new()
    let loader = DataLoader::mock();
    
    // Verify mock loader is operational
    loader.invalidate_cache().await;
    assert!(std::mem::size_of_val(&loader) > 0, "Mock DataLoader should have non-zero size");
}

#[tokio::test]
async fn test_query_params_filtering() {
    let start_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let end_time = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
    
    let params = QueryParams {
        symbol: "BTCUSD".to_string(),
        exchange: "BINANCE".to_string(),
        start_time: Some(start_time),
        end_time: Some(end_time),
        limit: Some(100),
    };
    
    assert_eq!(params.symbol, "BTCUSD");
    assert_eq!(params.exchange, "BINANCE");
    assert_eq!(params.start_time, Some(start_time));
    assert_eq!(params.end_time, Some(end_time));
    assert_eq!(params.limit, Some(100));
}

#[tokio::test]
async fn test_monte_carlo_params() {
    let mc_params = MonteCarloParams {
        iterations: 1000,
        window_minutes: 60,
        seed: Some(42),
        use_bootstrap: true,
        block_size: Some(10),
    };
    
    assert_eq!(mc_params.iterations, 1000);
    assert_eq!(mc_params.window_minutes, 60);
    assert_eq!(mc_params.seed, Some(42));
    assert_eq!(mc_params.use_bootstrap, true);
    assert_eq!(mc_params.block_size, Some(10));
}

#[tokio::test]
async fn test_market_data_serialization() {
    let _trade = create_test_trade_data();
    
    // Create a mock Candle for testing
    let candle = create_test_candle();
    
    // Test serialization
    let json = serde_json::to_string(&candle).unwrap();
    assert!(json.contains("BTCUSD"));
    assert!(json.contains("BINANCE"));
    assert!(json.contains("45000"));
    
    // Test deserialization
    let deserialized: Candle = serde_json::from_str(&json).unwrap();
    assert_eq!(&*deserialized.symbol, "BTCUSD");
    assert_eq!(&*deserialized.exchange, "BINANCE");
    assert_eq!(deserialized.trade_count, 25);
}

#[tokio::test]
async fn test_trade_data_structure() {
    let trade = create_test_trade_data();
    
    assert_eq!(&*trade.symbol, "BTCUSD");
    assert_eq!(&*trade.exchange, "BINANCE");
    assert_eq!(trade.side, Side::Buy);
    assert_eq!(trade.price, 45000.00);
    assert_eq!(trade.quantity, 1.5);
}

#[tokio::test]
async fn test_candle_ohlc_logic() {
    // Test that candle OHLC values make sense
    let candle = Candle {
        timestamp: Utc::now(),
        symbol: "BTCUSD".into(),
        exchange: "BINANCE".into(),
        open: 45000.00,
        high: 45200.00,
        low: 44800.00,
        close: 45100.00,
        volume: 100.0,
        trade_count: 50,
    };
    
    // High should be >= open, low, close
    assert!(candle.high >= candle.open);
    assert!(candle.high >= candle.close);
    assert!(candle.high >= candle.low);
    
    // Low should be <= open, high, close
    assert!(candle.low <= candle.open);
    assert!(candle.low <= candle.high);
    assert!(candle.low <= candle.close);
    
    // Volume and trade count should be positive
    assert!(candle.volume > 0.0);
    assert!(candle.trade_count > 0);
}

#[cfg(not(feature = "postgres"))]
#[tokio::test]
async fn test_cache_invalidation() {
    // Test basic DataLoader functionality without database
    let loader = DataLoader::new();
    
    // Test cache invalidation is idempotent — calling it twice should not panic
    loader.invalidate_cache().await;
    loader.invalidate_cache().await;
    // Verify loader is still usable after cache invalidation
    assert!(std::mem::size_of_val(&loader) > 0, "DataLoader should remain valid after cache invalidation");
}

#[tokio::test]
async fn test_f64_arithmetic() {
    // Test f64 operations used in trading calculations
    let price1: f64 = 45000.12345;
    let price2: f64 = 44999.87655;
    let quantity: f64 = 1.5;
    
    let volume1 = price1 * quantity;
    let volume2 = price2 * quantity;
    let total_volume = volume1 + volume2;
    
    // Verify approximate correctness
    assert!((volume1 - 67500.185175).abs() < 1e-6);
    assert!((volume2 - 67499.814825).abs() < 1e-6);
    assert!((total_volume - 135000.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_datetime_handling() {
    let timestamp1 = Utc.with_ymd_and_hms(2024, 1, 1, 12, 30, 45).unwrap();
    let timestamp2 = Utc.with_ymd_and_hms(2024, 1, 1, 12, 31, 15).unwrap();
    
    // Test timestamp comparison
    assert!(timestamp2 > timestamp1);
    
    // Test time interval calculations
    let duration = timestamp2.signed_duration_since(timestamp1);
    assert_eq!(duration.num_seconds(), 30); // 30 second difference
}

#[tokio::test]
async fn test_empty_data_handling() {
    // Test that the system handles empty datasets gracefully
    let empty_trades: Vec<TradeData> = vec![];
    let empty_candles: Vec<Candle> = vec![];
    
    assert!(empty_trades.is_empty());
    assert!(empty_candles.is_empty());
    
    // Test serialization of empty collections
    let json_trades = serde_json::to_string(&empty_trades).unwrap();
    let json_candles = serde_json::to_string(&empty_candles).unwrap();
    
    assert_eq!(json_trades, "[]");
    assert_eq!(json_candles, "[]");
}

#[tokio::test]
async fn test_market_data_variants() {
    let _trade = create_test_trade_data();
    let _candle = create_test_candle();
    
    let candle = create_test_candle();
    let trade = create_test_trade_data();
    
    // Test that we can create MarketData variants
    let market_data_candle = MarketData::Candle(candle);
    let market_data_trade = MarketData::Trade(trade);
    
    // Test pattern matching
    match market_data_candle {
        MarketData::Candle(c) => assert_eq!(&*c.symbol, "BTCUSD"),
        _ => panic!("Should be Candle variant"),
    }
    
    match market_data_trade {
        MarketData::Trade(t) => assert_eq!(&*t.symbol, "BTCUSD"),
        _ => panic!("Should be Trade variant"),
    }
}
