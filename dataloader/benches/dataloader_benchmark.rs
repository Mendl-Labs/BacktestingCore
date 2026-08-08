use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dataloader::{MarketData, Candle, TradeData};
use chrono::Utc;

// Helper function to create test candles
fn create_test_candle(symbol: &str, price: f64) -> MarketData {
    MarketData::Candle(Candle {
        symbol: symbol.into(),
        exchange: "TEST".into(),
        timestamp: Utc::now(),
        open: price,
        high: price * 1.05,
        low: price * 0.95,
        close: price * 1.02,
        volume: 1000.0,
        trade_count: 50,
    })
}

// Helper function to create test trades
fn create_test_trade(symbol: &str, price: f64) -> MarketData {
    MarketData::Trade(TradeData::new(
        Utc::now(),
        symbol.to_string(),
        "TEST",
        price,
        100.0,
        dataloader::Side::Buy,
        123,
    ))
}

// Benchmark DataLoader creation (only for non-postgres builds)
#[cfg(not(feature = "postgres"))]
fn bench_dataloader_creation(c: &mut Criterion) {
    c.bench_function("dataloader_creation", |b| {
        b.iter(|| {
            use dataloader::DataLoader;

            let loader = DataLoader::new();
            black_box(loader)
        })
    });
}

#[cfg(feature = "postgres")]
fn bench_dataloader_creation(_c: &mut Criterion) {
    // Skip for postgres builds - requires database connection
}

// Benchmark market data creation
fn bench_market_data_creation(c: &mut Criterion) {
    c.bench_function("candle_creation", |b| {
        b.iter(|| {
            let candle = create_test_candle("BTC/USDT", 50000.0);
            black_box(candle)
        })
    });
    
    c.bench_function("trade_creation", |b| {
        b.iter(|| {
            let trade = create_test_trade("BTC/USDT", 50000.0);
            black_box(trade)
        })
    });
}

// Benchmark market data processing
fn bench_market_data_processing(c: &mut Criterion) {
    let candles: Vec<MarketData> = (0..1000)
        .map(|i| create_test_candle("BTC/USDT", 50000.0 + i as f64))
        .collect();
    
    c.bench_function("process_candles", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for candle in &candles {
                match candle {
                    MarketData::Candle(candle) => {
                        sum += candle.close;
                    }
                    MarketData::Trade(trade) => {
                        sum += trade.price;
                    }
                    #[allow(unreachable_patterns)]
                    _ => {} // Handle Order/Snapshot variants when postgres is enabled
                }
            }
            black_box(sum)
        })
    });
}

criterion_group!(benches, bench_dataloader_creation, bench_market_data_creation, bench_market_data_processing);
criterion_main!(benches);
