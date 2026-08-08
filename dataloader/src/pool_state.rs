use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapDirection {
    AToB,
    BToA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolType {
    ConstantProduct,
    ConcentratedLiquidity,
    StableSwap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSwapEvent {
    pub timestamp: DateTime<Utc>,
    pub pool_id: Arc<str>,
    pub exchange: Arc<str>,
    pub token_a: Arc<str>,
    pub token_b: Arc<str>,
    pub sqrt_price: f64,
    pub current_tick: i32,
    pub total_liquidity: f64,
    pub reserve_a: f64,
    pub reserve_b: f64,
    pub amount_in: f64,
    pub amount_out: f64,
    pub fee_amount: f64,
    pub fee_tier_bps: u32,
    pub direction: SwapDirection,
    pub tvl_usd: Option<f64>,
}

impl PoolSwapEvent {
    pub fn price(&self) -> f64 {
        if self.amount_in > 0.0 {
            self.amount_out / self.amount_in
        } else {
            self.sqrt_price * self.sqrt_price
        }
    }
}
