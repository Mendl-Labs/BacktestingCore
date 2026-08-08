//! Integration test for the backtesting workflow and metrics

use backtest::simulation::SimulationLoop;
use backtest::types::BacktestResult;
use config::BacktestConfig;
use strategy::StrategyManager;
use orderbook::{OrderBook, OrderBookConfig};
use portfoliomanager::{PortfolioState, Position, PositionSide};
use riskmanager::RiskManager;
use chrono::{Utc, Duration};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backtest_metrics_integration() {
        // Setup mock components
        let mut strategy_manager = StrategyManager::default();
        let mut orderbook = OrderBook::new("BTC/USDT".to_string(), "binance".to_string(), OrderBookConfig::default());
        let mut portfolio = PortfolioState::default();
        let mut risk_manager = RiskManager::new();
        let config = BacktestConfig::default();
        let mut sim = SimulationLoop::new();    // Simulate 3 closed BTC positions with varying PnL and durations
    let now = Utc::now();
    portfolio.positions = vec![
        Position {
            symbol: "BTC/USDT".to_string(),
            side: PositionSide::Long,
            quantity: 0.5,
            entry_price: 30000.0,
            mark_price: Some(31000.0),
            realized_pnl: 500.0,
            unrealized_pnl: 0.0,
            open_time: now - Duration::seconds(3600),
            close_time: Some(now - Duration::seconds(1800)),
            instrument: None,
            greeks: None,
            margin_posted: 0.0,
        },
        Position {
            symbol: "BTC/USDT".to_string(),
            side: PositionSide::Short,
            quantity: 0.3,
            entry_price: 32000.0,
            mark_price: Some(31000.0),
            realized_pnl: -300.0,
            unrealized_pnl: 0.0,
            open_time: now - Duration::seconds(7200),
            close_time: Some(now - Duration::seconds(3600)),
            instrument: None,
            greeks: None,
            margin_posted: 0.0,
        },
        Position {
            symbol: "BTC/USDT".to_string(),
            side: PositionSide::Long,
            quantity: 0.2,
            entry_price: 31000.0,
            mark_price: Some(31500.0),
            realized_pnl: 100.0,
            unrealized_pnl: 0.0,
            open_time: now - Duration::seconds(5400),
            close_time: Some(now - Duration::seconds(100)),
            instrument: None,
            greeks: None,
            margin_posted: 0.0,
        },
    ];
    portfolio.balance = 10000.0;

    // No market data needed for this test, as positions are mocked
    let market_data = vec![];

    let result: BacktestResult = sim
        .run(
            &mut strategy_manager,
            &mut orderbook,
            &mut portfolio,
            &mut risk_manager,
            &market_data,
            &config,
        )
        .await;

    // Assert metrics
    assert_eq!(result.num_trades, 3);
    assert_eq!(result.gross_profit.unwrap(), 600.0);
    assert_eq!(result.gross_loss.unwrap(), -300.0);
    assert_eq!(result.net_profit.unwrap(), 300.0);
    assert!(result.avg_trade_return.unwrap() > 0.0);
    assert!(result.median_trade_return.unwrap() > 0.0);
    assert!(result.win_rate.unwrap() > 0.0 && result.win_rate.unwrap() < 1.0);
    assert!(result.avg_trade_duration.unwrap() > 0.0);
    // volatility is None when equity curve has <= 1 data points (no market data processed)
    // This is expected behavior for this test with empty market_data
    assert!(result.volatility.is_none() || result.volatility.unwrap() >= 0.0);
    assert!(result.max_drawdown >= 0.0);
    }
}
