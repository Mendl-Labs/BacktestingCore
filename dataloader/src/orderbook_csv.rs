//! Vendor-agnostic CSV orderbook loader
//!
//! Loads L2 orderbook snapshots from gzipped CSV files in Tardis.dev format.
//! This format is the de-facto industry standard for L2 snapshots and is
//! supported by many data vendors.
//!
//! Data format (book_snapshot_25):
//! exchange,symbol,timestamp,local_timestamp,asks[0].price,asks[0].amount,...,bids[0].price,bids[0].amount,...

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use thiserror::Error;

/// Errors that can occur when loading orderbook CSV data
#[derive(Error, Debug)]
pub enum CsvOrderbookError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("File not found: {0}")]
    FileNotFound(String),
}

/// A single price level in the orderbook
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub amount: f64,
}

/// L2 orderbook snapshot (25 levels each side)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderbookSnapshot {
    /// Timestamp in microseconds since Unix epoch
    pub timestamp_us: i64,
    /// Local timestamp in nanoseconds
    pub local_timestamp_ns: i64,
    /// Ask levels (best ask at index 0)
    pub asks: [PriceLevel; 25],
    /// Bid levels (best bid at index 0)
    pub bids: [PriceLevel; 25],
}

impl OrderbookSnapshot {
    /// Get the best bid price
    #[inline]
    pub fn best_bid(&self) -> f64 {
        self.bids[0].price
    }

    /// Get the best ask price
    #[inline]
    pub fn best_ask(&self) -> f64 {
        self.asks[0].price
    }

    /// Get the mid price
    #[inline]
    pub fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }

    /// Get the spread
    #[inline]
    pub fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }

    /// Get the spread in basis points
    #[inline]
    pub fn spread_bps(&self) -> f64 {
        self.spread() / self.mid_price() * 10000.0
    }

    /// Get total bid liquidity up to N levels
    pub fn bid_liquidity(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.amount).sum()
    }

    /// Get total ask liquidity up to N levels
    pub fn ask_liquidity(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.amount).sum()
    }

    /// Get bid liquidity weighted by price distance from mid
    pub fn bid_depth_weighted(&self, levels: usize) -> f64 {
        let mid = self.mid_price();
        self.bids
            .iter()
            .take(levels)
            .map(|l| l.amount * l.price / mid)
            .sum()
    }

    /// Get ask depth weighted by price distance from mid
    pub fn ask_depth_weighted(&self, levels: usize) -> f64 {
        let mid = self.mid_price();
        self.asks
            .iter()
            .take(levels)
            .map(|l| l.amount * l.price / mid)
            .sum()
    }

    /// Calculate order book imbalance (-1 to 1, positive = more bids)
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_liq = self.bid_liquidity(levels);
        let ask_liq = self.ask_liquidity(levels);
        let total = bid_liq + ask_liq;
        if total == 0.0 {
            0.0
        } else {
            (bid_liq - ask_liq) / total
        }
    }

    /// Get timestamp as DateTime
    pub fn timestamp(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_micros(self.timestamp_us)
            .unwrap_or(DateTime::UNIX_EPOCH)
    }
}

/// Loader for L2 orderbook snapshots from CSV.gz files
pub struct CsvOrderbookLoader;

impl CsvOrderbookLoader {
    /// Load orderbook snapshots from a gzipped CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the .csv.gz file
    ///
    /// # Returns
    /// Vector of orderbook snapshots sorted by timestamp
    pub fn load<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<OrderbookSnapshot>, CsvOrderbookError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(CsvOrderbookError::FileNotFound(path.display().to_string()));
        }

        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::with_capacity(1024 * 1024, decoder);

        let mut snapshots = Vec::with_capacity(1_000_000);
        let mut lines = reader.lines();

        // Skip header line
        if let Some(header_result) = lines.next() {
            let header = header_result?;
            if !header.contains("timestamp") {
                // First line doesn't look like a header, try to parse it
                if let Ok(snapshot) = Self::parse_line(&header) {
                    snapshots.push(snapshot);
                }
            }
        }

        for (line_num, line_result) in lines.enumerate() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }

            match Self::parse_line(&line) {
                Ok(snapshot) => snapshots.push(snapshot),
                Err(msg) => {
                    if line_num < 10 {
                        crate::log_warn!(
                            crate::logging_facade::DATALOADER_LOGGER,
                            "Skipping line {}: {}",
                            line_num + 2,
                            msg
                        );
                    }
                }
            }
        }

        snapshots.sort_by_key(|s| s.timestamp_us);

        crate::log_info!(
            crate::logging_facade::DATALOADER_LOGGER,
            "Loaded {} orderbook snapshots from {}",
            snapshots.len(),
            path.display()
        );
        Ok(snapshots)
    }

    /// Load orderbook snapshots with time range filter
    pub fn load_range<P: AsRef<Path>>(
        path: P,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OrderbookSnapshot>, CsvOrderbookError> {
        let start_us = start.timestamp_micros();
        let end_us = end.timestamp_micros();

        let path = path.as_ref();
        if !path.exists() {
            return Err(CsvOrderbookError::FileNotFound(path.display().to_string()));
        }

        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::with_capacity(1024 * 1024, decoder);

        let mut snapshots = Vec::with_capacity(100_000);
        let mut lines = reader.lines();

        // Skip header line
        if let Some(header_result) = lines.next() {
            let header = header_result?;
            if !header.contains("timestamp") {
                if let Ok(snapshot) = Self::parse_line(&header) {
                    if snapshot.timestamp_us >= start_us && snapshot.timestamp_us < end_us {
                        snapshots.push(snapshot);
                    }
                }
            }
        }

        for line_result in lines {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }

            if let Ok(snapshot) = Self::parse_line(&line) {
                if snapshot.timestamp_us >= start_us && snapshot.timestamp_us < end_us {
                    snapshots.push(snapshot);
                } else if snapshot.timestamp_us >= end_us {
                    break;
                }
            }
        }

        snapshots.sort_by_key(|s| s.timestamp_us);
        Ok(snapshots)
    }

    /// Parse a single orderbook snapshot line
    /// Format: exchange,symbol,timestamp,local_timestamp,asks[0].price,asks[0].amount,bids[0].price,bids[0].amount,...(interleaved 25 levels)
    fn parse_line(line: &str) -> Result<OrderbookSnapshot, String> {
        let parts: Vec<&str> = line.split(',').collect();
        // 4 header fields + 25 levels * 4 (ask_price, ask_amount, bid_price, bid_amount) = 104
        if parts.len() < 104 {
            return Err(format!("Expected 104 fields, got {}", parts.len()));
        }

        let timestamp_us: i64 = parts[2]
            .parse()
            .map_err(|_| format!("Invalid timestamp: {}", parts[2]))?;
        let local_timestamp_ns: i64 = parts[3]
            .parse()
            .map_err(|_| format!("Invalid local_timestamp: {}", parts[3]))?;

        let mut asks = [PriceLevel::default(); 25];
        let mut bids = [PriceLevel::default(); 25];

        // Parse 25 interleaved levels starting at index 4
        // Each level: asks[i].price, asks[i].amount, bids[i].price, bids[i].amount
        for i in 0..25 {
            let base = 4 + i * 4;
            asks[i].price = parts[base].parse().unwrap_or(0.0);
            asks[i].amount = parts[base + 1].parse().unwrap_or(0.0);
            bids[i].price = parts[base + 2].parse().unwrap_or(0.0);
            bids[i].amount = parts[base + 3].parse().unwrap_or(0.0);
        }

        Ok(OrderbookSnapshot {
            timestamp_us,
            local_timestamp_ns,
            asks,
            bids,
        })
    }
}
