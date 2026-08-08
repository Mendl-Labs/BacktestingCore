use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Candle, MarketData};

/// Aggregation functions for market data processing
pub struct DataAggregator;

impl DataAggregator {
    /// Calculate OHLCV (Open, High, Low, Close, Volume) data from tick data
    pub fn calculate_ohlcv(
        ticks: &[(DateTime<Utc>, BigDecimal, BigDecimal)], // (timestamp, price, volume)
        interval_seconds: i64,
    ) -> Result<Vec<OHLCVData>> {
        if ticks.is_empty() {
            return Ok(vec![]);
        }

        let mut ohlcv_data = Vec::new();
        let mut current_interval_start = ticks[0].0;
        let mut open = ticks[0].1.clone();
        let mut high = ticks[0].1.clone();
        let mut low = ticks[0].1.clone();
        let mut close = ticks[0].1.clone();
        let mut volume = BigDecimal::from(0);

        for (timestamp, price, tick_volume) in ticks {
            // Check if we need to start a new interval
            let interval_duration = chrono::Duration::seconds(interval_seconds);
            if *timestamp >= current_interval_start + interval_duration {
                // Save current interval data
                ohlcv_data.push(OHLCVData {
                    timestamp: current_interval_start,
                    open: open.clone(),
                    high: high.clone(),
                    low: low.clone(),
                    close: close.clone(),
                    volume: volume.clone(),
                });

                // Start new interval
                current_interval_start = *timestamp;
                open = price.clone();
                high = price.clone();
                low = price.clone();
                close = price.clone();
                volume = tick_volume.clone();
            } else {
                // Update current interval data
                if price > &high {
                    high = price.clone();
                }
                if price < &low {
                    low = price.clone();
                }
                close = price.clone();
                volume += tick_volume;
            }
        }

        // Add the last interval
        if !ticks.is_empty() {
            ohlcv_data.push(OHLCVData {
                timestamp: current_interval_start,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        Ok(ohlcv_data)
    }

    /// Calculate moving average
    pub fn calculate_moving_average(
        prices: &[BigDecimal],
        window_size: usize,
    ) -> Result<Vec<BigDecimal>> {
        if prices.len() < window_size || window_size == 0 {
            return Ok(vec![]);
        }

        let mut moving_averages = Vec::new();
        
        for i in window_size.saturating_sub(1)..prices.len() {
            let start_idx = i.saturating_sub(window_size.saturating_sub(1));
            let window_sum: BigDecimal = prices[start_idx..=i]
                .iter()
                .sum();
            let window_size_bigint = (window_size as i64).into();
            let window_size_decimal = BigDecimal::new(window_size_bigint, 0);
            let avg = &window_sum / &window_size_decimal;
            moving_averages.push(avg);
        }

        Ok(moving_averages)
    }

    /// Calculate volume-weighted average price (VWAP)
    pub fn calculate_vwap(
        price_volume_data: &[(BigDecimal, BigDecimal)], // (price, volume)
    ) -> Result<BigDecimal> {
        if price_volume_data.is_empty() {
            return Ok(BigDecimal::from(0));
        }

        let mut total_pv = BigDecimal::from(0);
        let mut total_volume = BigDecimal::from(0);

        for (price, volume) in price_volume_data {
            total_pv += price * volume;
            total_volume += volume;
        }

        if total_volume == BigDecimal::from(0) {
            Ok(BigDecimal::from(0))
        } else {
            Ok(total_pv / total_volume)
        }
    }

    /// Aggregate trade data by symbol
    pub fn aggregate_by_symbol(
        trades: &[(String, BigDecimal, BigDecimal, DateTime<Utc>)], // (symbol, price, volume, timestamp)
    ) -> HashMap<String, Vec<(BigDecimal, BigDecimal, DateTime<Utc>)>> {
        let mut aggregated: HashMap<String, Vec<(BigDecimal, BigDecimal, DateTime<Utc>)>> = HashMap::new();

        for (symbol, price, volume, timestamp) in trades {
            aggregated
                .entry(symbol.clone())
                .or_insert_with(Vec::new)
                .push((price.clone(), volume.clone(), *timestamp));
        }

        aggregated
    }
}

/// OHLCV data structure
#[derive(Debug, Clone)]
pub struct OHLCVData {
    pub timestamp: DateTime<Utc>,
    pub open: BigDecimal,
    pub high: BigDecimal,
    pub low: BigDecimal,
    pub close: BigDecimal,
    pub volume: BigDecimal,
}

// ============================================================================
// MarketData-based Candle Aggregation
// ============================================================================

/// Aggregate raw trade events into OHLCV candles at the specified interval.
///
/// This is the primary aggregation function used by the backtest pipeline.
/// It operates on `MarketData::Trade` events, ignoring Candle and Generic variants.
///
/// # Arguments
/// * `trades` - Slice of market data events (only `Trade` variants are aggregated)
/// * `interval_minutes` - Candle interval in minutes (must be > 0)
///
/// # Returns
/// Sorted `Vec<MarketData::Candle>` on success, or an error message if
/// `interval_minutes` is not positive.
pub fn aggregate_to_candles(
    trades: &[MarketData],
    interval_minutes: i64,
) -> std::result::Result<Vec<MarketData>, String> {
    if interval_minutes <= 0 {
        return Err(format!("interval_minutes must be > 0, got {}", interval_minutes));
    }
    if trades.is_empty() {
        return Ok(vec![]);
    }

    let interval_secs = interval_minutes * 60;

    // Estimate number of candles for pre-allocation
    let first_ts = match &trades[0] {
        MarketData::Trade(t) => t.timestamp.timestamp(),
        MarketData::Candle(c) => c.timestamp.timestamp(),
        MarketData::Generic(g) => DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(DateTime::UNIX_EPOCH).timestamp(),
        MarketData::PoolSwap(s) => s.timestamp.timestamp(),
        MarketData::OptionCandle(c) => c.timestamp.timestamp(),
    };
    let last_ts = match trades.last().unwrap() {
        MarketData::Trade(t) => t.timestamp.timestamp(),
        MarketData::Candle(c) => c.timestamp.timestamp(),
        MarketData::Generic(g) => DateTime::from_timestamp_millis(g.timestamp_ms).unwrap_or(DateTime::UNIX_EPOCH).timestamp(),
        MarketData::PoolSwap(s) => s.timestamp.timestamp(),
        MarketData::OptionCandle(c) => c.timestamp.timestamp(),
    };
    let estimated_candles = ((last_ts - first_ts) / interval_secs + 1) as usize;

    // Tuple: (open, high, low, close, volume, trade_count, first_timestamp, symbol, exchange)
    let mut candles: HashMap<i64, (f64, f64, f64, f64, f64, i64, DateTime<Utc>, Arc<str>, Arc<str>)> = 
        HashMap::with_capacity(estimated_candles);

    for event in trades {
        if let MarketData::Trade(trade) = event {
            let bucket = (trade.timestamp.timestamp() / interval_secs) * interval_secs;
            
            let entry = candles.entry(bucket).or_insert_with(|| {
                (
                    trade.price,      // open
                    trade.price,      // high
                    trade.price,      // low
                    trade.price,      // close
                    0.0,              // volume
                    0,                // trade_count
                    trade.timestamp,  // first timestamp
                    trade.symbol.clone(),
                    trade.exchange.clone(),
                )
            });

            entry.1 = entry.1.max(trade.price); // high
            entry.2 = entry.2.min(trade.price); // low
            entry.3 = trade.price;              // close (last)
            entry.4 += trade.quantity;          // volume
            entry.5 += 1;                      // trade_count
        }
    }

    let mut result: Vec<MarketData> = candles
        .into_iter()
        .map(|(bucket, (open, high, low, close, volume, trade_count, _ts, symbol, exchange))| {
            let bucket_time = DateTime::from_timestamp(bucket, 0).unwrap_or(Utc::now());
            MarketData::Candle(Candle {
                timestamp: bucket_time,
                symbol,
                exchange,
                open,
                high,
                low,
                close,
                volume,
                trade_count,
            })
        })
        .collect();

    result.sort_by_key(|c| match c {
        MarketData::Candle(c) => c.timestamp,
        _ => Utc::now(),
    });

    // Fill-forward: insert synthetic candles for missing intervals so strategies
    // never see timestamp gaps. Synthetic candles use the previous close as OHLC
    // with zero volume/trade_count to mark them as generated.
    if result.len() >= 2 {
        let mut filled = Vec::with_capacity(estimated_candles);
        filled.push(result[0].clone());

        for i in 1..result.len() {
            let prev = match &filled[filled.len() - 1] {
                MarketData::Candle(c) => c.clone(),
                _ => continue,
            };
            let curr = match &result[i] {
                MarketData::Candle(c) => c.clone(),
                _ => continue,
            };

            let mut expected_ts = prev.timestamp.timestamp() + interval_secs;
            while expected_ts < curr.timestamp.timestamp() {
                let bucket_time = DateTime::from_timestamp(expected_ts, 0)
                    .unwrap_or(Utc::now());
                filled.push(MarketData::Candle(Candle {
                    timestamp: bucket_time,
                    symbol: prev.symbol.clone(),
                    exchange: prev.exchange.clone(),
                    open: prev.close,
                    high: prev.close,
                    low: prev.close,
                    close: prev.close,
                    volume: 0.0,
                    trade_count: 0,
                }));
                expected_ts += interval_secs;
            }
            filled.push(result[i].clone());
        }
        result = filled;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_calculate_moving_average() {
        let prices = vec![
            BigDecimal::from(100),
            BigDecimal::from(102),
            BigDecimal::from(104),
            BigDecimal::from(103),
            BigDecimal::from(105),
        ];

        let ma = DataAggregator::calculate_moving_average(&prices, 3).unwrap();
        assert_eq!(ma.len(), 3);
        
        // First MA should be (100 + 102 + 104) / 3 = 102
        assert_eq!(ma[0], BigDecimal::from(102));
    }

    #[test]
    fn test_calculate_vwap() {
        let price_volume_data = vec![
            (BigDecimal::from(100), BigDecimal::from(10)),  // 100 * 10 = 1000
            (BigDecimal::from(102), BigDecimal::from(20)),  // 102 * 20 = 2040
            (BigDecimal::from(104), BigDecimal::from(15)),  // 104 * 15 = 1560
        ];
        // Total PV = 4600, Total Volume = 45, VWAP = 4600/45 ≈ 102.22

        let vwap = DataAggregator::calculate_vwap(&price_volume_data).unwrap();
        // Using integer division, should be close to 102.22
        assert!(vwap > BigDecimal::from(102) && vwap < BigDecimal::from(103));
    }

    #[test]
    fn test_calculate_ohlcv() {
        let ticks = vec![
            (Utc.with_ymd_and_hms(2023, 1, 1, 10, 0, 0).unwrap(), BigDecimal::from(100), BigDecimal::from(10)),
            (Utc.with_ymd_and_hms(2023, 1, 1, 10, 0, 30).unwrap(), BigDecimal::from(102), BigDecimal::from(15)),
            (Utc.with_ymd_and_hms(2023, 1, 1, 10, 1, 0).unwrap(), BigDecimal::from(101), BigDecimal::from(12)),
            (Utc.with_ymd_and_hms(2023, 1, 1, 10, 1, 30).unwrap(), BigDecimal::from(103), BigDecimal::from(8)),
        ];

        let ohlcv = DataAggregator::calculate_ohlcv(&ticks, 60).unwrap(); // 60-second intervals
        assert_eq!(ohlcv.len(), 2);
        
        // First interval (10:00:00 - 10:01:00)
        assert_eq!(ohlcv[0].open, BigDecimal::from(100));
        assert_eq!(ohlcv[0].high, BigDecimal::from(102));
        assert_eq!(ohlcv[0].low, BigDecimal::from(100));
        assert_eq!(ohlcv[0].close, BigDecimal::from(102));
        assert_eq!(ohlcv[0].volume, BigDecimal::from(25)); // 10 + 15
    }

    #[test]
    fn test_aggregate_to_candles_basic() {
        use crate::{TradeData, tick::Side};
        use std::sync::Arc;

        let base = Utc.with_ymd_and_hms(2024, 1, 1, 10, 0, 0).unwrap();
        let mut trades: Vec<MarketData> = Vec::new();

        // 5 trades in minute 0 (10:00:00 - 10:00:59)
        for i in 0..5 {
            trades.push(MarketData::Trade(TradeData {
                timestamp: base + chrono::Duration::seconds(i * 10),
                symbol: Arc::from("BTC-USD"),
                exchange: Arc::from("massive"),
                price: 100.0 + i as f64,
                quantity: 1.0,
                side: Side::Buy,
                trade_id: i,
                value: (100.0 + i as f64) * 1.0,
            }));
        }

        // 3 trades in minute 1 (10:01:00 - 10:01:59)
        for i in 0..3 {
            trades.push(MarketData::Trade(TradeData {
                timestamp: base + chrono::Duration::seconds(60 + i * 10),
                symbol: Arc::from("BTC-USD"),
                exchange: Arc::from("massive"),
                price: 200.0 + i as f64,
                quantity: 2.0,
                side: Side::Sell,
                trade_id: 10 + i,
                value: (200.0 + i as f64) * 2.0,
            }));
        }

        let candles = aggregate_to_candles(&trades, 1).unwrap();
        assert_eq!(candles.len(), 2);

        // First candle: minute 0
        if let MarketData::Candle(c) = &candles[0] {
            assert_eq!(c.open, 100.0);
            assert_eq!(c.high, 104.0);
            assert_eq!(c.low, 100.0);
            assert_eq!(c.close, 104.0);
            assert_eq!(c.volume, 5.0);
            assert_eq!(c.trade_count, 5);
        } else {
            panic!("Expected Candle variant");
        }

        // Second candle: minute 1
        if let MarketData::Candle(c) = &candles[1] {
            assert_eq!(c.open, 200.0);
            assert_eq!(c.high, 202.0);
            assert_eq!(c.low, 200.0);
            assert_eq!(c.close, 202.0);
            assert_eq!(c.volume, 6.0);
            assert_eq!(c.trade_count, 3);
        } else {
            panic!("Expected Candle variant");
        }
    }

    #[test]
    fn test_aggregate_to_candles_invalid_interval() {
        assert!(aggregate_to_candles(&[], 0).is_err());
        assert!(aggregate_to_candles(&[], -1).is_err());
    }

    #[test]
    fn test_aggregate_to_candles_empty() {
        let result = aggregate_to_candles(&[], 1).unwrap();
        assert!(result.is_empty());
    }
}
