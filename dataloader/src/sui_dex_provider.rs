//! Sui DEX historical data provider via The Graph subgraphs.
//!
//! Fetches historical swap/trade events from Cetus (AMM) and DeepBook (CLOB)
//! subgraphs, converting them to [`MarketData::Trade`] or aggregated candles
//! for backtesting.

use async_trait::async_trait;
use chrono::DateTime;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::{CacheStats, DataRequest, CandleGranularity};
use crate::provider::{MarketDataProvider, ProviderError};
use crate::{Candle, MarketData, TradeData, tick};

const DEFAULT_CETUS_SUBGRAPH: &str = "https://api.thegraph.com/subgraphs/name/cetus-protocol/cetus-sui";
const DEFAULT_DEEPBOOK_SUBGRAPH: &str = "https://api.thegraph.com/subgraphs/name/deepbook-protocol/deepbook-sui";
const PAGE_SIZE: usize = 1000;

#[derive(Debug)]
pub struct SuiDexDataProvider {
    http_client: reqwest::Client,
    cetus_subgraph_url: String,
    deepbook_subgraph_url: String,
    cache: Arc<RwLock<HashMap<String, Vec<MarketData>>>>,
}

impl SuiDexDataProvider {
    pub fn new(cetus_url: impl Into<String>, deepbook_url: impl Into<String>) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            cetus_subgraph_url: cetus_url.into(),
            deepbook_subgraph_url: deepbook_url.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn from_env() -> Self {
        let cetus = std::env::var("CETUS_SUBGRAPH_URL")
            .unwrap_or_else(|_| DEFAULT_CETUS_SUBGRAPH.to_string());
        let deepbook = std::env::var("DEEPBOOK_SUBGRAPH_URL")
            .unwrap_or_else(|_| DEFAULT_DEEPBOOK_SUBGRAPH.to_string());
        Self::new(cetus, deepbook)
    }

    fn cache_key(request: &DataRequest) -> String {
        format!(
            "{}:{}:{}:{}",
            request.exchange.as_str(),
            request.symbol,
            request.from,
            request.to,
        )
    }

    fn is_cetus(exchange_str: &str) -> bool {
        let lower = exchange_str.to_lowercase();
        lower == "cetus" || lower == "cetus_amm" || lower == "cetusprotocol"
    }

    fn is_deepbook(exchange_str: &str) -> bool {
        let lower = exchange_str.to_lowercase();
        lower == "deepbook" || lower == "deepbookv2" || lower == "deep_book"
    }

    async fn fetch_cetus_swaps(&self, request: &DataRequest) -> Result<Vec<MarketData>, ProviderError> {
        let start_ts = request.from.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end_ts = request.to.and_hms_opt(23, 59, 59).unwrap().and_utc().timestamp();

        let mut all_trades = Vec::new();
        let mut last_timestamp = start_ts;

        loop {
            let query = serde_json::json!({
                "query": format!(
                    r#"{{
                        swaps(
                            where: {{ pool: "{}", timestamp_gte: "{}", timestamp_lte: "{}" }}
                            orderBy: timestamp
                            orderDirection: asc
                            first: {}
                        ) {{
                            id
                            timestamp
                            amountA
                            amountB
                            atob
                        }}
                    }}"#,
                    request.symbol, last_timestamp, end_ts, PAGE_SIZE
                )
            });

            let resp = self.http_client
                .post(&self.cetus_subgraph_url)
                .json(&query)
                .send()
                .await
                .map_err(|e| ProviderError::Fetch(format!("Cetus subgraph request failed: {}", e)))?;

            if !resp.status().is_success() {
                return Err(ProviderError::Fetch(format!(
                    "Cetus subgraph returned HTTP {}",
                    resp.status()
                )));
            }

            let body: GraphResponse<CetusSwapsData> = resp
                .json()
                .await
                .map_err(|e| ProviderError::Fetch(format!("Failed to parse Cetus response: {}", e)))?;

            let swaps = body.data.map(|d| d.swaps).unwrap_or_default();

            if swaps.is_empty() {
                break;
            }

            let page_count = swaps.len();

            for swap in swaps {
                let ts = swap.timestamp.parse::<i64>().unwrap_or(0);
                let timestamp = DateTime::from_timestamp(ts, 0).unwrap_or(DateTime::UNIX_EPOCH);

                let amount_a: f64 = swap.amount_a.parse().unwrap_or(0.0);
                let amount_b: f64 = swap.amount_b.parse().unwrap_or(0.0);

                if amount_a == 0.0 || amount_b == 0.0 {
                    continue;
                }

                let (price, quantity, side) = if swap.atob {
                    (amount_b / amount_a, amount_a, tick::Side::Sell)
                } else {
                    (amount_a / amount_b, amount_b, tick::Side::Buy)
                };

                all_trades.push(MarketData::Trade(TradeData::new(
                    timestamp,
                    request.symbol.clone(),
                    "cetus",
                    price,
                    quantity,
                    side,
                    ts,
                )));

                last_timestamp = ts;
            }

            if page_count < PAGE_SIZE {
                break;
            }
            last_timestamp += 1;
        }

        Ok(all_trades)
    }

    async fn fetch_deepbook_trades(&self, request: &DataRequest) -> Result<Vec<MarketData>, ProviderError> {
        let start_ts = request.from.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end_ts = request.to.and_hms_opt(23, 59, 59).unwrap().and_utc().timestamp();

        let mut all_trades = Vec::new();
        let mut last_timestamp = start_ts;

        loop {
            let query = serde_json::json!({
                "query": format!(
                    r#"{{
                        orderFills(
                            where: {{ pool: "{}", timestamp_gte: "{}", timestamp_lte: "{}" }}
                            orderBy: timestamp
                            orderDirection: asc
                            first: {}
                        ) {{
                            id
                            timestamp
                            price
                            quantity
                            isBid
                        }}
                    }}"#,
                    request.symbol, last_timestamp, end_ts, PAGE_SIZE
                )
            });

            let resp = self.http_client
                .post(&self.deepbook_subgraph_url)
                .json(&query)
                .send()
                .await
                .map_err(|e| ProviderError::Fetch(format!("DeepBook subgraph request failed: {}", e)))?;

            if !resp.status().is_success() {
                return Err(ProviderError::Fetch(format!(
                    "DeepBook subgraph returned HTTP {}",
                    resp.status()
                )));
            }

            let body: GraphResponse<DeepBookFillsData> = resp
                .json()
                .await
                .map_err(|e| ProviderError::Fetch(format!("Failed to parse DeepBook response: {}", e)))?;

            let fills = body.data.map(|d| d.order_fills).unwrap_or_default();

            if fills.is_empty() {
                break;
            }

            let page_count = fills.len();

            for fill in fills {
                let ts = fill.timestamp.parse::<i64>().unwrap_or(0);
                let timestamp = DateTime::from_timestamp(ts, 0).unwrap_or(DateTime::UNIX_EPOCH);

                let price: f64 = fill.price.parse().unwrap_or(0.0);
                let quantity: f64 = fill.quantity.parse().unwrap_or(0.0);

                if price == 0.0 || quantity == 0.0 {
                    continue;
                }

                let side = if fill.is_bid { tick::Side::Buy } else { tick::Side::Sell };

                all_trades.push(MarketData::Trade(TradeData::new(
                    timestamp,
                    request.symbol.clone(),
                    "deepbook",
                    price,
                    quantity,
                    side,
                    ts,
                )));

                last_timestamp = ts;
            }

            if page_count < PAGE_SIZE {
                break;
            }
            last_timestamp += 1;
        }

        Ok(all_trades)
    }

    fn aggregate_trades_to_candles(
        trades: Vec<MarketData>,
        granularity: CandleGranularity,
        symbol: &str,
        exchange: &str,
    ) -> Vec<MarketData> {
        if trades.is_empty() {
            return vec![];
        }

        let interval_secs = match granularity {
            CandleGranularity::Minute1 => 60,
            CandleGranularity::Minute5 => 300,
            CandleGranularity::Minute15 => 900,
            CandleGranularity::Hour1 => 3600,
            CandleGranularity::Hour4 => 14400,
            CandleGranularity::Day1 => 86400,
        };

        let mut buckets: std::collections::BTreeMap<i64, Vec<(f64, f64)>> = std::collections::BTreeMap::new();

        for trade in &trades {
            if let MarketData::Trade(t) = trade {
                let bucket_ts = (t.timestamp.timestamp() / interval_secs) * interval_secs;
                buckets.entry(bucket_ts).or_default().push((t.price, t.quantity));
            }
        }

        buckets
            .into_iter()
            .map(|(ts, price_qty)| {
                let open = price_qty.first().unwrap().0;
                let close = price_qty.last().unwrap().0;
                let high = price_qty.iter().map(|(p, _)| *p).fold(f64::NEG_INFINITY, f64::max);
                let low = price_qty.iter().map(|(p, _)| *p).fold(f64::INFINITY, f64::min);
                let volume: f64 = price_qty.iter().map(|(_, q)| *q).sum();
                let trade_count = price_qty.len() as i64;

                let timestamp = DateTime::from_timestamp(ts, 0).unwrap_or(DateTime::UNIX_EPOCH);

                MarketData::Candle(Candle {
                    timestamp,
                    symbol: Arc::from(symbol),
                    exchange: Arc::from(exchange),
                    open,
                    high,
                    low,
                    close,
                    volume,
                    trade_count,
                })
            })
            .collect()
    }
}

#[async_trait]
impl MarketDataProvider for SuiDexDataProvider {
    fn name(&self) -> &str {
        "Sui DEX (The Graph)"
    }

    async fn fetch(&self, request: &DataRequest) -> Result<Vec<MarketData>, ProviderError> {
        let cache_key = Self::cache_key(request);

        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let exchange_str = request.exchange.as_str();

        let trades = if Self::is_cetus(exchange_str) {
            self.fetch_cetus_swaps(request).await?
        } else if Self::is_deepbook(exchange_str) {
            self.fetch_deepbook_trades(request).await?
        } else {
            return Err(ProviderError::NotConfigured(format!(
                "Unknown Sui DEX exchange: '{}'. Use 'cetus' or 'deepbook'.",
                exchange_str
            )));
        };

        if trades.is_empty() {
            return Err(ProviderError::NoData {
                exchange: exchange_str.to_string(),
                symbol: request.symbol.clone(),
                from: request.from.to_string(),
                to: request.to.to_string(),
            });
        }

        let result = match request.granularity {
            Some(granularity) => {
                Self::aggregate_trades_to_candles(trades, granularity, &request.symbol, exchange_str)
            }
            None => trades,
        };

        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    async fn cache_stats(&self) -> Option<CacheStats> {
        let cache = self.cache.read().await;
        Some(CacheStats {
            entries: cache.len(),
            memory_bytes: 0,
            max_entries: 0,
            max_memory_bytes: 0,
        })
    }

    async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

// ============================================================================
// The Graph Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct GraphResponse<T> {
    data: Option<T>,
    #[allow(dead_code)]
    errors: Option<Vec<GraphError>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GraphError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct CetusSwapsData {
    swaps: Vec<CetusSwap>,
}

#[derive(Debug, Deserialize)]
struct CetusSwap {
    #[allow(dead_code)]
    id: String,
    timestamp: String,
    #[serde(rename = "amountA")]
    amount_a: String,
    #[serde(rename = "amountB")]
    amount_b: String,
    atob: bool,
}

#[derive(Debug, Deserialize)]
struct DeepBookFillsData {
    #[serde(rename = "orderFills")]
    order_fills: Vec<DeepBookFill>,
}

#[derive(Debug, Deserialize)]
struct DeepBookFill {
    #[allow(dead_code)]
    id: String,
    timestamp: String,
    price: String,
    quantity: String,
    #[serde(rename = "isBid")]
    is_bid: bool,
}

// ── PoolDataProvider implementation ─────────────────────────────────────────

use crate::pool_state::{PoolSwapEvent, SwapDirection};
use crate::pool_provider::{PoolDataProvider, PoolDataRequest};
use chrono::NaiveDate;

#[derive(Deserialize)]
struct PoolSwapGraphResponse {
    #[allow(dead_code)]
    data: Option<PoolSwapGraphData>,
}

#[derive(Deserialize)]
struct PoolSwapGraphData {
    swaps: Vec<CetusPoolSwapRaw>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct CetusPoolSwapRaw {
    #[serde(default)]
    timestamp: String,
    #[serde(default, rename = "amountA")]
    amount_a: String,
    #[serde(default, rename = "amountB")]
    amount_b: String,
    #[serde(default)]
    atob: bool,
    #[serde(default, rename = "feeAmount")]
    fee_amount: String,
    #[serde(default)]
    pool: Option<CetusPoolInfo>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct CetusPoolInfo {
    #[serde(default)]
    liquidity: String,
    #[serde(default, rename = "sqrtPrice")]
    sqrt_price: String,
    #[serde(default)]
    tick: String,
    #[serde(default, rename = "feeTier")]
    fee_tier: String,
}

#[async_trait]
impl PoolDataProvider for SuiDexDataProvider {
    async fn fetch_pool_swaps(&self, request: &PoolDataRequest) -> Result<Vec<PoolSwapEvent>, ProviderError> {
        let start_ts = request.start_date.and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        let end_ts = request.end_date.and_hms_opt(23, 59, 59)
            .unwrap()
            .and_utc()
            .timestamp();

        let subgraph_url = match request.exchange.as_str() {
            "deepbook" => &self.deepbook_subgraph_url,
            _ => &self.cetus_subgraph_url,
        };

        let mut all_swaps = Vec::new();
        let mut last_timestamp = start_ts;

        loop {
            let query = format!(
                r#"{{
                    swaps(
                        first: {PAGE_SIZE},
                        orderBy: timestamp,
                        orderDirection: asc,
                        where: {{
                            pool: "{}",
                            timestamp_gt: "{}",
                            timestamp_lte: "{}"
                        }}
                    ) {{
                        timestamp
                        amountA
                        amountB
                        atob
                        feeAmount
                        pool {{
                            liquidity
                            sqrtPrice
                            tick
                            feeTier
                        }}
                    }}
                }}"#,
                request.pool_id, last_timestamp, end_ts,
            );

            let body = serde_json::json!({ "query": query });
            let resp = self.http_client
                .post(subgraph_url)
                .json(&body)
                .send()
                .await
                .map_err(|e| ProviderError::Fetch(e.to_string()))?;

            let graph_resp: PoolSwapGraphResponse = resp.json().await
                .map_err(|e| ProviderError::Fetch(format!("JSON parse error: {}", e)))?;

            let swaps = match graph_resp.data {
                Some(d) => d.swaps,
                None => break,
            };

            if swaps.is_empty() {
                break;
            }

            for raw in &swaps {
                let ts: i64 = raw.timestamp.parse().unwrap_or(0);
                let timestamp = DateTime::from_timestamp(ts, 0)
                    .unwrap_or_default();

                let amount_a: f64 = raw.amount_a.parse().unwrap_or(0.0);
                let amount_b: f64 = raw.amount_b.parse().unwrap_or(0.0);
                let fee: f64 = raw.fee_amount.parse().unwrap_or(0.0);

                let (liquidity, sqrt_price, tick, fee_tier) = match &raw.pool {
                    Some(p) => (
                        p.liquidity.parse::<f64>().unwrap_or(0.0),
                        p.sqrt_price.parse::<f64>().unwrap_or(0.0),
                        p.tick.parse::<i32>().unwrap_or(0),
                        p.fee_tier.parse::<u32>().unwrap_or(30),
                    ),
                    None => (0.0, 0.0, 0, 30),
                };

                let (amount_in, amount_out, direction) = if raw.atob {
                    (amount_a, amount_b, SwapDirection::AToB)
                } else {
                    (amount_b, amount_a, SwapDirection::BToA)
                };

                all_swaps.push(PoolSwapEvent {
                    timestamp,
                    pool_id: Arc::from(request.pool_id.as_str()),
                    exchange: Arc::from(request.exchange.as_str()),
                    token_a: Arc::from(""),
                    token_b: Arc::from(""),
                    sqrt_price,
                    current_tick: tick,
                    total_liquidity: liquidity,
                    reserve_a: 0.0,
                    reserve_b: 0.0,
                    amount_in,
                    amount_out,
                    fee_amount: fee,
                    fee_tier_bps: fee_tier,
                    direction,
                    tvl_usd: None,
                });

                last_timestamp = ts;
            }

            if swaps.len() < PAGE_SIZE {
                break;
            }
        }

        Ok(all_swaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use crate::models::Exchange;

    #[test]
    fn test_is_cetus() {
        assert!(SuiDexDataProvider::is_cetus("cetus"));
        assert!(SuiDexDataProvider::is_cetus("Cetus"));
        assert!(SuiDexDataProvider::is_cetus("cetusprotocol"));
        assert!(!SuiDexDataProvider::is_cetus("deepbook"));
    }

    #[test]
    fn test_is_deepbook() {
        assert!(SuiDexDataProvider::is_deepbook("deepbook"));
        assert!(SuiDexDataProvider::is_deepbook("DeepBook"));
        assert!(SuiDexDataProvider::is_deepbook("deepbookv2"));
        assert!(!SuiDexDataProvider::is_deepbook("cetus"));
    }

    #[test]
    fn test_aggregate_trades_to_candles() {
        let trades = vec![
            MarketData::Trade(TradeData::new(
                DateTime::from_timestamp(1020, 0).unwrap(),
                "SUI/USDC", "cetus", 1.50, 100.0, tick::Side::Buy, 1,
            )),
            MarketData::Trade(TradeData::new(
                DateTime::from_timestamp(1030, 0).unwrap(),
                "SUI/USDC", "cetus", 1.55, 50.0, tick::Side::Sell, 2,
            )),
            MarketData::Trade(TradeData::new(
                DateTime::from_timestamp(1045, 0).unwrap(),
                "SUI/USDC", "cetus", 1.48, 75.0, tick::Side::Buy, 3,
            )),
        ];

        let candles = SuiDexDataProvider::aggregate_trades_to_candles(
            trades,
            CandleGranularity::Minute1,
            "SUI/USDC",
            "cetus",
        );

        assert_eq!(candles.len(), 1);
        if let MarketData::Candle(c) = &candles[0] {
            assert_eq!(c.open, 1.50);
            assert_eq!(c.close, 1.48);
            assert_eq!(c.high, 1.55);
            assert_eq!(c.low, 1.48);
            assert_eq!(c.volume, 225.0);
            assert_eq!(c.trade_count, 3);
        } else {
            panic!("Expected candle");
        }
    }

    #[test]
    fn test_cache_key() {
        let request = DataRequest::new(
            Exchange::Other("cetus".to_string()),
            "SUI/USDC",
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 31).unwrap(),
        );
        let key = SuiDexDataProvider::cache_key(&request);
        assert!(key.contains("cetus"));
        assert!(key.contains("SUI/USDC"));
    }
}
