//! Order Book Reconstruction Engine
//!
//! This module handles the reconstruction of order books from market data events,
//! converting raw order events into live order book state.
//!
//! NOTE: Historical market data is now loaded from Tardis API (see dataloader::tardis).
//! This reconstructor works with in-memory events rather than database queries.

use crate::order_book::OrderBook;
use crate::types::OrderBookEvent;
use crate::OrderBookConfig;
use anyhow::Result;
use chrono::{DateTime, Utc};
use dataloader::DataLoader;
use crate::logging_facade::ORDERBOOK_LOGGER;
use crate::{log_debug, log_info, log_warn};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Order book reconstructor that maintains order books from market data events
/// 
/// NOTE: This module no longer loads historical data from the database.
/// Historical market data should be loaded from Tardis API (see dataloader::tardis)
/// and converted to events before being fed to this reconstructor.
pub struct OrderBookReconstructor {
    /// Data loader for accessing candle data (historical orderbook data now comes from Tardis)
    #[allow(dead_code)]
    dataloader: Arc<DataLoader>,
    /// Currently active order books by symbol-exchange pair
    books: HashMap<String, OrderBook>,
    /// Configuration for order books
    config: OrderBookConfig,
    /// Event queue for processing
    event_queue: VecDeque<(DateTime<Utc>, OrderBookEvent)>,
    /// Last processed timestamp
    last_timestamp: Option<DateTime<Utc>>,
}

impl OrderBookReconstructor {
    /// Create a new order book reconstructor
    pub fn new(dataloader: Arc<DataLoader>, config: OrderBookConfig) -> Self {
        Self {
            dataloader,
            books: HashMap::new(),
            config,
            event_queue: VecDeque::new(),
            last_timestamp: None,
        }
    }

    /// Get or create an order book for a symbol-exchange pair
    pub fn get_or_create_book(&mut self, symbol: &str, exchange: &str) -> &mut OrderBook {
        let key = format!("{}:{}", symbol, exchange);
        self.books.entry(key).or_insert_with(|| {
            OrderBook::new(symbol.to_string(), exchange.to_string(), self.config.clone())
        })
    }

    /// Get a reference to an existing order book
    pub fn get_book(&self, symbol: &str, exchange: &str) -> Option<&OrderBook> {
        let key = format!("{}:{}", symbol, exchange);
        self.books.get(&key)
    }

    /// Queue an event for processing
    pub fn queue_event(&mut self, timestamp: DateTime<Utc>, event: OrderBookEvent) {
        self.event_queue.push_back((timestamp, event));
    }

    /// Queue multiple events for processing
    pub fn queue_events(&mut self, events: Vec<(DateTime<Utc>, OrderBookEvent)>) {
        for (timestamp, event) in events {
            self.event_queue.push_back((timestamp, event));
        }
        
        // Sort events by timestamp
        let mut all_events: Vec<_> = self.event_queue.drain(..).collect();
        all_events.sort_by_key(|(timestamp, _)| *timestamp);
        self.event_queue.extend(all_events);
    }

    /// Process all queued events up to a specific timestamp
    pub fn process_events_until(&mut self, until_timestamp: DateTime<Utc>) -> Result<usize> {
        let mut processed_count = 0;

        while let Some((timestamp, _)) = self.event_queue.front() {
            if *timestamp > until_timestamp {
                break;
            }

            if let Some((event_timestamp, event)) = self.event_queue.pop_front() {
                self.process_single_event(event)?;
                self.last_timestamp = Some(event_timestamp);
                processed_count += 1;
            }
        }

        log_debug!(ORDERBOOK_LOGGER, "Processed {} events until {}", processed_count, until_timestamp);
        Ok(processed_count)
    }

    /// Process all remaining events
    pub fn process_all_events(&mut self) -> Result<usize> {
        let mut processed_count = 0;

        while let Some((event_timestamp, event)) = self.event_queue.pop_front() {
            self.process_single_event(event)?;
            self.last_timestamp = Some(event_timestamp);
            processed_count += 1;
        }

        log_info!(ORDERBOOK_LOGGER, "Processed all {} remaining events", processed_count);
        Ok(processed_count)
    }

    /// Process a single order book event
    fn process_single_event(&mut self, event: OrderBookEvent) -> Result<()> {
        match &event {
            OrderBookEvent::NewOrder { .. }
            | OrderBookEvent::ModifyOrder { .. }
            | OrderBookEvent::CancelOrder { .. }
            | OrderBookEvent::Trade { .. } => {
                // Extract symbol and exchange from event (we'll need to enhance events with this info)
                log_warn!(ORDERBOOK_LOGGER, "Individual order events not yet supported - need symbol/exchange context");
            }
            OrderBookEvent::Snapshot { .. } => {
                // For snapshots, we'll need to enhance the event structure to include symbol/exchange
                log_warn!(ORDERBOOK_LOGGER, "Snapshot events need symbol/exchange context");
            }
        }

        Ok(())
    }

    /// Get the current state of all order books
    pub fn get_all_books(&self) -> &HashMap<String, OrderBook> {
        &self.books
    }

    /// Get the last processed timestamp
    pub fn get_last_timestamp(&self) -> Option<DateTime<Utc>> {
        self.last_timestamp
    }

    /// Get the number of pending events
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }

    /// Clear all data and reset
    pub fn reset(&mut self) {
        self.books.clear();
        self.event_queue.clear();
        self.last_timestamp = None;
    }

    /// Reconstruct order book for a specific symbol/exchange pair
    /// 
    /// NOTE: This method now expects events to be pre-queued via queue_events().
    /// Historical data should be loaded from Tardis API and converted to events first.
    pub async fn reconstruct_book(
        &mut self,
        symbol: &str,
        exchange: &str,
        _start_time: DateTime<Utc>,
        _end_time: DateTime<Utc>,
    ) -> Result<&OrderBook> {
        log_info!(ORDERBOOK_LOGGER, 
            "Starting order book reconstruction for {}:{}",
            symbol, exchange
        );

        // Process all queued events
        let processed_count = self.process_all_events()?;
        log_info!(ORDERBOOK_LOGGER, "Reconstruction complete. Processed {} events", processed_count);

        // Return the reconstructed book
        Ok(self.get_or_create_book(symbol, exchange))
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(feature = "postgres")]
    fn create_mock_dataloader() -> Arc<DataLoader> {
        // This is a mock for testing - the actual pool creation would be handled elsewhere
        use diesel_async::pooled_connection::AsyncDieselConnectionManager;
        use diesel_async::AsyncPgConnection;
        use deadpool::managed::Pool;
        
        let manager = AsyncDieselConnectionManager::<AsyncPgConnection>::new("postgresql://mock");
        let pool = Pool::builder(manager).max_size(1).build().unwrap();
        Arc::new(DataLoader::new(Arc::new(pool)))
    }

    #[cfg(not(feature = "postgres"))]
    fn create_mock_dataloader() -> Arc<DataLoader> {
        Arc::new(DataLoader::new())
    }

    #[test]
    fn test_reconstructor_creation() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let reconstructor = OrderBookReconstructor::new(dataloader, config);
        
        assert_eq!(reconstructor.books.len(), 0);
        assert_eq!(reconstructor.pending_events(), 0);
        assert!(reconstructor.get_last_timestamp().is_none());
    }

    #[test]
    fn test_book_management() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);
        
        // Create a book
        let _book = reconstructor.get_or_create_book("BTCUSD", "BINANCE");
        assert_eq!(reconstructor.books.len(), 1);
        
        // Get existing book
        let book = reconstructor.get_book("BTCUSD", "BINANCE");
        assert!(book.is_some());
        
        // Non-existent book
        let book = reconstructor.get_book("ETHUSD", "COINBASE");
        assert!(book.is_none());
    }

    #[test]
    fn test_reset_functionality() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);
        
        // Add some data
        let _book = reconstructor.get_or_create_book("BTCUSD", "BINANCE");
        assert_eq!(reconstructor.books.len(), 1);
        
        // Reset
        reconstructor.reset();
        assert_eq!(reconstructor.books.len(), 0);
        assert_eq!(reconstructor.pending_events(), 0);
        assert!(reconstructor.get_last_timestamp().is_none());
    }

    #[test]
    fn test_queue_single_event() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);

        let ts = Utc::now();
        let event = OrderBookEvent::Snapshot {
            bids: vec![],
            asks: vec![],
            timestamp: ts,
        };
        reconstructor.queue_event(ts, event);
        assert_eq!(reconstructor.pending_events(), 1);
    }

    #[test]
    fn test_queue_multiple_events_sorted() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);

        let ts1 = Utc::now();
        let ts2 = ts1 + chrono::Duration::seconds(10);
        let ts3 = ts1 + chrono::Duration::seconds(5); // Out of order

        let events = vec![
            (ts1, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts1 }),
            (ts3, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts3 }),
            (ts2, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts2 }),
        ];
        reconstructor.queue_events(events);
        assert_eq!(reconstructor.pending_events(), 3);
    }

    #[tokio::test]
    async fn test_process_all_events() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);

        let ts = Utc::now();
        reconstructor.queue_event(ts, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts });
        reconstructor.queue_event(ts + chrono::Duration::seconds(1), OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts + chrono::Duration::seconds(1) });

        let processed = reconstructor.process_all_events().unwrap();
        assert_eq!(processed, 2);
        assert_eq!(reconstructor.pending_events(), 0);
        assert!(reconstructor.get_last_timestamp().is_some());
    }

    #[tokio::test]
    async fn test_process_events_until() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);

        let ts1 = Utc::now();
        let ts2 = ts1 + chrono::Duration::seconds(5);
        let ts3 = ts1 + chrono::Duration::seconds(10);

        reconstructor.queue_event(ts1, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts1 });
        reconstructor.queue_event(ts2, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts2 });
        reconstructor.queue_event(ts3, OrderBookEvent::Snapshot { bids: vec![], asks: vec![], timestamp: ts3 });

        let processed = reconstructor.process_events_until(ts2).unwrap();
        assert_eq!(processed, 2);
        assert_eq!(reconstructor.pending_events(), 1);
    }

    #[test]
    fn test_get_all_books() {
        let dataloader = create_mock_dataloader();
        let config = OrderBookConfig::default();
        let mut reconstructor = OrderBookReconstructor::new(dataloader, config);

        let _ = reconstructor.get_or_create_book("BTC/USDT", "binance");
        let _ = reconstructor.get_or_create_book("ETH/USDT", "coinbase");

        let books = reconstructor.get_all_books();
        assert_eq!(books.len(), 2);
    }
}
