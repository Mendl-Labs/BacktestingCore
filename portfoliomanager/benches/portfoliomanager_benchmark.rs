use criterion::{criterion_group, criterion_main, Criterion};
use portfoliomanager::{PortfolioState, Position, PositionSide};
use chrono::Utc;
use orderbook::{Fill, BookSide};

fn create_test_portfolio() -> PortfolioState {
    let mut portfolio = PortfolioState::default();
    portfolio.balance = 10000.0;
    portfolio.positions.push(Position {
        symbol: "BTC/USDT".to_string(),
        side: PositionSide::Long,
        quantity: 0.5,
        entry_price: 50000.0,
        mark_price: Some(52000.0),
        realized_pnl: 0.0,
        unrealized_pnl: 1000.0,
        open_time: Utc::now(),
        close_time: None,
        instrument: None,
        greeks: None,
        margin_posted: 0.0,
    });
    portfolio
}

fn create_test_fill() -> Fill {
    Fill {
        order_id: "test_order_123".into(),
        timestamp: Utc::now(),
        side: BookSide::Bid,
        price: 51000.0,
        base_price: Some(50900.0),
        quantity: 0.1,
        liquidity_type: orderbook::LiquidityType::Taker,
        fee_rate: 0.001,
        fee_amount: 5.1,
        slippage_bps: Some(20.0),
    }
}

fn bench_portfolio_creation(c: &mut Criterion) {
    c.bench_function("create_test_portfolio", |b| {
        b.iter(|| {
            create_test_portfolio()
        })
    });
}

fn bench_portfolio_fill_application(c: &mut Criterion) {
    c.bench_function("apply_portfolio_fill", |b| {
        b.iter(|| {
            let mut portfolio = create_test_portfolio();
            let fill = create_test_fill();
            let _ = portfolio.apply_fill(&fill);
            portfolio
        })
    });
}

fn bench_position_creation(c: &mut Criterion) {
    c.bench_function("create_position", |b| {
        b.iter(|| {
            Position {
                symbol: "ETH/USDT".to_string(),
                side: PositionSide::Short,
                quantity: 2.0,
                entry_price: 3000.0,
                mark_price: Some(2950.0),
                realized_pnl: 0.0,
                unrealized_pnl: 100.0,
                open_time: Utc::now(),
                close_time: None,
                instrument: None,
                greeks: None,
                margin_posted: 0.0,
            }
        })
    });
}

criterion_group!(benches, bench_portfolio_creation, bench_portfolio_fill_application, bench_position_creation);
criterion_main!(benches);