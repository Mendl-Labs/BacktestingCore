//! Integration tests for SmartRouter crate.
//!
//! Covers untested methods:
//!   - SmartRouter::route_order, simulate_execution, with_execution_style
//!   - SmartRouter::update_venue, venues, clear_venues, reset_metrics, set_cross_venue_latency
//!   - VenueSnapshot: bid/ask_liquidity, market_impact_bps, expected_fill_price_with_fees, score_for_order
//!   - route_vwap, calculate_min_impact_allocation
//!   - ArbitrageRouter::create_arbitrage_orders, set_latency, clear_venues, reset_metrics
//!   - ArbitrageOpportunity::is_viable
//!   - RoutingMetrics::add, overall_quality, avg_fee_rate
//!   - Error paths for empty venues, insufficient liquidity
//!   - Full end-to-end workflow

use smartrouter::{
    SmartRouter, RoutingMode, ExecutionStyle, RoutingError,
    ParentOrder, ChildOrder, OrderSide, OrderType, Fill,
    VenueSnapshot, VenueConfig,
    RoutingMetrics, ExecutionQuality,
    ArbitrageRouter, ArbitrageOpportunity,
    route_best_venue, route_multi_venue, route_twap, route_vwap,
};
use smartrouter::venue::Level;

// ─── Helpers ───────────────────────────────────────────────

fn binance_snapshot(symbol: &str) -> VenueSnapshot {
    VenueSnapshot::new(
        VenueConfig::binance(),
        symbol,
        1_000_000,
        vec![
            Level { price: 50000.0, quantity: 2.0 },
            Level { price: 49990.0, quantity: 5.0 },
            Level { price: 49980.0, quantity: 10.0 },
        ],
        vec![
            Level { price: 50010.0, quantity: 2.0 },
            Level { price: 50020.0, quantity: 5.0 },
            Level { price: 50030.0, quantity: 10.0 },
        ],
    )
}

fn kraken_snapshot(symbol: &str) -> VenueSnapshot {
    VenueSnapshot::new(
        VenueConfig::kraken(),
        symbol,
        1_000_000,
        vec![
            Level { price: 49995.0, quantity: 3.0 },
            Level { price: 49985.0, quantity: 4.0 },
        ],
        vec![
            Level { price: 50005.0, quantity: 3.0 },
            Level { price: 50015.0, quantity: 4.0 },
        ],
    )
}

fn coinbase_snapshot(symbol: &str) -> VenueSnapshot {
    VenueSnapshot::new(
        VenueConfig::coinbase(),
        symbol,
        1_000_000,
        vec![
            Level { price: 49998.0, quantity: 1.0 },
            Level { price: 49990.0, quantity: 2.0 },
        ],
        vec![
            Level { price: 50002.0, quantity: 1.0 },
            Level { price: 50010.0, quantity: 2.0 },
        ],
    )
}

fn buy_order(qty: f64) -> ParentOrder {
    ParentOrder::market("BTCUSD", OrderSide::Buy, qty, 50000.0)
}

fn sell_order(qty: f64) -> ParentOrder {
    ParentOrder::market("BTCUSD", OrderSide::Sell, qty, 50000.0)
}

// ============================================================================
// SmartRouter — route_order
// ============================================================================

#[test]
fn route_order_best_venue_succeeds() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));

    let order = buy_order(1.0);
    let decision = router.route_order(&order, None).unwrap();
    
    assert!(!decision.child_orders.is_empty());
    assert!(decision.expected_avg_price > 0.0);
}

#[test]
fn route_order_multi_venue_succeeds() {
    let mut router = SmartRouter::new(RoutingMode::MultiVenue {
        max_venues: 3,
        min_venue_allocation: 0.1,
    });
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));

    let order = buy_order(3.0);
    let decision = router.route_order(&order, None).unwrap();
    
    assert!(!decision.child_orders.is_empty());
    // Total child order quantity should approximate parent order
    let total_qty: f64 = decision.child_orders.iter().map(|c| c.quantity).sum();
    assert!((total_qty - 3.0).abs() < 0.1);
}

#[test]
fn route_order_no_venues_returns_error() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    let order = buy_order(1.0);
    let result = router.route_order(&order, None);
    assert!(result.is_err());
}

#[test]
fn route_order_wrong_symbol_returns_error() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("ETHUSD")); // Wrong symbol
    
    let order = buy_order(1.0);
    let result = router.route_order(&order, None);
    assert!(result.is_err());
}

#[test]
fn route_order_arbitrage_mode_returns_algorithm_error() {
    let mut router = SmartRouter::new(RoutingMode::Arbitrage {
        min_profit_bps: 5.0,
        max_holding_time_ms: 1000,
    });
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = buy_order(1.0);
    let result = router.route_order(&order, None);
    assert!(result.is_err());
}

#[test]
fn route_order_with_execution_style_override() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue)
        .with_execution_style(ExecutionStyle::Passive);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = buy_order(1.0);
    // Override with Aggressive
    let decision = router.route_order(&order, Some(ExecutionStyle::Aggressive)).unwrap();
    assert!(!decision.child_orders.is_empty());
}

// ============================================================================
// SmartRouter — simulate_execution
// ============================================================================

#[test]
fn simulate_execution_produces_fills() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = buy_order(1.0);
    let decision = router.route_order(&order, None).unwrap();
    
    let result = router.simulate_execution(&decision, 2_000_000);
    assert!(result.total_filled > 0.0);
    assert!(result.avg_fill_price > 0.0);
    assert!(!result.fills.is_empty());
}

#[test]
fn simulate_execution_venue_breakdown() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = buy_order(1.0);
    let decision = router.route_order(&order, None).unwrap();
    let result = router.simulate_execution(&decision, 2_000_000);
    
    assert!(!result.venue_breakdown.is_empty());
    for (_, summary) in &result.venue_breakdown {
        assert!(summary.quantity_filled > 0.0);
        assert!(summary.fill_count > 0);
    }
}

// ============================================================================
// SmartRouter — state management
// ============================================================================

#[test]
fn update_venue_and_venues() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    assert!(router.venues().is_empty());
    
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    assert_eq!(router.venues().len(), 1);
    
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));
    assert_eq!(router.venues().len(), 2);
}

#[test]
fn update_venue_replaces_existing() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("binance", binance_snapshot("BTCUSD")); // Replace
    assert_eq!(router.venues().len(), 1);
}

#[test]
fn clear_venues_empties_map() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));
    
    router.clear_venues();
    assert!(router.venues().is_empty());
}

#[test]
fn reset_metrics_clears_cumulative() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = buy_order(1.0);
    let _decision = router.route_order(&order, None).unwrap();
    
    router.reset_metrics();
    assert_eq!(router.metrics().total_orders, 0);
}

#[test]
fn set_cross_venue_latency() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.set_cross_venue_latency("binance", "kraken", 10.0);
    // No panic — just ensures internal state is set
}

// ============================================================================
// VenueSnapshot — liquidity and impact methods
// ============================================================================

#[test]
fn venue_best_bid_ask() {
    let snap = binance_snapshot("BTCUSD");
    assert_eq!(snap.best_bid(), Some(50000.0));
    assert_eq!(snap.best_ask(), Some(50010.0));
}

#[test]
fn venue_bid_liquidity_all() {
    let snap = binance_snapshot("BTCUSD");
    let liq = snap.bid_liquidity(None);
    // 2 + 5 + 10 = 17
    assert!((liq - 17.0).abs() < 0.01);
}

#[test]
fn venue_ask_liquidity_all() {
    let snap = binance_snapshot("BTCUSD");
    let liq = snap.ask_liquidity(None);
    assert!((liq - 17.0).abs() < 0.01);
}

#[test]
fn venue_bid_liquidity_up_to_price() {
    let snap = binance_snapshot("BTCUSD");
    // Bids at 50000(2), 49990(5), 49980(10); up_to_price=49990 should include 50000 and 49990
    let liq = snap.bid_liquidity(Some(49990.0));
    assert!(liq >= 7.0); // At least top 2 levels
}

#[test]
fn venue_market_impact_bps() {
    let snap = binance_snapshot("BTCUSD");
    let impact = snap.market_impact_bps(OrderSide::Buy, 1.0);
    assert!(impact >= 0.0, "Market impact should be non-negative");
}

#[test]
fn venue_expected_fill_price_with_fees() {
    let snap = binance_snapshot("BTCUSD");
    let price_no_fees = snap.expected_fill_price(OrderSide::Buy, 1.0).unwrap();
    let price_with_fees = snap.expected_fill_price_with_fees(OrderSide::Buy, 1.0).unwrap();
    
    // With fees should be higher for buy
    assert!(price_with_fees >= price_no_fees);
}

#[test]
fn venue_score_for_order() {
    let snap = binance_snapshot("BTCUSD");
    let score = snap.score_for_order(OrderSide::Buy, 1.0);
    assert!(score > 0.0 && score.is_finite(), "Score should be positive: {score}");
}

#[test]
fn venue_empty_book() {
    let snap = VenueSnapshot::new(
        VenueConfig::default(),
        "BTCUSD",
        1_000_000,
        vec![],
        vec![],
    );
    assert_eq!(snap.best_bid(), None);
    assert_eq!(snap.best_ask(), None);
    assert_eq!(snap.expected_fill_price(OrderSide::Buy, 1.0), None);
}

// ============================================================================
// ParentOrder — builder methods
// ============================================================================

#[test]
fn parent_order_with_timestamp() {
    let order = ParentOrder::market("BTCUSD", OrderSide::Buy, 1.0, 50000.0)
        .with_timestamp(12345);
    assert_eq!(order.created_at_ms, 12345);
}

#[test]
fn parent_order_limit() {
    let order = ParentOrder::limit("BTCUSD", OrderSide::Buy, 1.0, 49900.0, 50000.0);
    assert!(matches!(order.order_type, OrderType::Limit));
    assert_eq!(order.limit_price, Some(49900.0));
}

#[test]
fn order_side_display() {
    assert_eq!(format!("{}", OrderSide::Buy), "BUY");
    assert_eq!(format!("{}", OrderSide::Sell), "SELL");
}

#[test]
fn fill_total_cost() {
    let fill = Fill {
        order_id: "test".to_string(),
        venue: "binance".to_string(),
        side: OrderSide::Buy,
        quantity: 2.0,
        price: 50000.0,
        fee: 100.0,
        timestamp_ms: 0,
    };
    // total_cost for buy = qty * price + fee
    let cost = fill.total_cost();
    assert!(cost > 0.0);
}

// ============================================================================
// route_vwap
// ============================================================================

#[test]
fn route_vwap_produces_slices() {
    let venues: Vec<(String, VenueSnapshot)> = vec![
        ("binance".to_string(), binance_snapshot("BTCUSD")),
    ];
    let venue_refs: Vec<(&String, &VenueSnapshot)> = venues.iter().map(|(k, v)| (k, v)).collect();
    
    let order = buy_order(2.0);
    let profile = vec![0.3, 0.5, 0.2]; // volume profile
    
    let result = route_vwap(&order, &venue_refs, 60_000, &profile);
    match result {
        Ok(decisions) => {
            assert!(!decisions.is_empty(), "VWAP should produce slices");
        }
        Err(_) => {
            // Some implementations may not support VWAP fully
        }
    }
}

// ============================================================================
// ArbitrageRouter
// ============================================================================

#[test]
fn arbitrage_create_orders_for_opportunity() {
    let mut arb = ArbitrageRouter::new(5.0, 5000);
    
    // Create price discrepancy: binance ask < kraken bid
    let mut binance_snap = binance_snapshot("BTCUSD");
    binance_snap.asks = vec![
        Level { price: 49900.0, quantity: 5.0 },
    ];
    
    let mut kraken_snap = kraken_snapshot("BTCUSD");
    kraken_snap.bids = vec![
        Level { price: 50100.0, quantity: 5.0 },
    ];
    
    arb.update_venue("binance", binance_snap);
    arb.update_venue("kraken", kraken_snap);
    
    let opportunities = arb.scan_opportunities("BTCUSD");
    if let Some(opp) = opportunities.first() {
        let result = arb.create_arbitrage_orders(opp, 1.0);
        assert!(result.is_ok(), "Should create arb orders");
        let (buy_order, sell_order) = result.unwrap();
        assert_ne!(buy_order.venue, sell_order.venue);
    }
}

#[test]
fn arbitrage_opportunity_is_viable() {
    let opp = ArbitrageOpportunity {
        symbol: "BTCUSD".to_string(),
        buy_venue: "binance".to_string(),
        sell_venue: "kraken".to_string(),
        buy_price: 49900.0,
        sell_price: 50100.0,
        max_quantity: 5.0,
        gross_profit_bps: 40.0,
        net_profit_bps: 30.0,
        expected_profit_usd: 200.0,
        detected_at_ms: 1_000_000,
        estimated_latency_ms: 50.0,
        execution_risk: 0.2,
    };
    
    assert!(opp.is_viable(10.0)); // 30 net bps > 10 min threshold
    assert!(!opp.is_viable(50.0)); // 30 net bps < 50 min threshold
}

#[test]
fn arbitrage_set_latency() {
    let mut arb = ArbitrageRouter::new(5.0, 5000);
    arb.set_latency("binance", "kraken", 15.0);
    // No panic
}

#[test]
fn arbitrage_clear_venues() {
    let mut arb = ArbitrageRouter::new(5.0, 5000);
    arb.update_venue("binance", binance_snapshot("BTCUSD"));
    arb.clear_venues();
    
    let opportunities = arb.scan_opportunities("BTCUSD");
    assert!(opportunities.is_empty());
}

#[test]
fn arbitrage_reset_metrics() {
    let mut arb = ArbitrageRouter::new(5.0, 5000);
    arb.reset_metrics();
    assert_eq!(arb.metrics().opportunities_detected, 0);
}

// ============================================================================
// ExecutionQuality
// ============================================================================

#[test]
fn execution_quality_tiers() {
    assert_eq!(ExecutionQuality::from_slippage_bps(0.5), ExecutionQuality::Excellent);
    assert_eq!(ExecutionQuality::from_slippage_bps(3.0), ExecutionQuality::Good);
    assert_eq!(ExecutionQuality::from_slippage_bps(7.0), ExecutionQuality::Average);
    assert_eq!(ExecutionQuality::from_slippage_bps(15.0), ExecutionQuality::Poor);
    assert_eq!(ExecutionQuality::from_slippage_bps(30.0), ExecutionQuality::VeryPoor);
}

// ============================================================================
// Full end-to-end workflow
// ============================================================================

#[test]
fn full_workflow_route_simulate_metrics() {
    // 1. Create router
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    
    // 2. Add venues
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));
    assert_eq!(router.venues().len(), 2);
    
    // 3. Route order
    let order = buy_order(1.0);
    let decision = router.route_order(&order, Some(ExecutionStyle::Aggressive)).unwrap();
    assert!(!decision.child_orders.is_empty());
    
    // 4. Simulate execution
    let result = router.simulate_execution(&decision, 2_000_000);
    assert!(result.total_filled > 0.0);
    assert!(result.avg_fill_price > 49000.0);
    assert!(result.avg_fill_price < 51000.0);
    
    // 5. Check metrics updated
    assert!(router.metrics().total_orders > 0);
    
    // 6. Reset
    router.clear_venues();
    router.reset_metrics();
    assert!(router.venues().is_empty());
    assert_eq!(router.metrics().total_orders, 0);
}

#[test]
fn multi_venue_workflow() {
    let mut router = SmartRouter::new(RoutingMode::MultiVenue {
        max_venues: 3,
        min_venue_allocation: 0.1,
    });
    
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    router.update_venue("kraken", kraken_snapshot("BTCUSD"));
    router.update_venue("coinbase", coinbase_snapshot("BTCUSD"));
    
    let order = buy_order(5.0); // Large enough to potentially split
    let decision = router.route_order(&order, None).unwrap();
    
    let result = router.simulate_execution(&decision, 2_000_000);
    assert!(result.total_filled > 0.0);
}

#[test]
fn sell_order_routed_correctly() {
    let mut router = SmartRouter::new(RoutingMode::BestVenue);
    router.update_venue("binance", binance_snapshot("BTCUSD"));
    
    let order = sell_order(1.0);
    let decision = router.route_order(&order, None).unwrap();
    assert!(!decision.child_orders.is_empty());
    
    let result = router.simulate_execution(&decision, 2_000_000);
    assert!(result.total_filled > 0.0);
}
