use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LpPoolType {
    ConstantProduct,
    ConcentratedLiquidity,
    StableSwap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpPosition {
    pub id: String,
    pub pool_id: String,
    pub pool_type: LpPoolType,
    pub exchange: String,
    pub token_a: String,
    pub token_b: String,
    pub liquidity_units: f64,
    pub tick_lower: Option<i32>,
    pub tick_upper: Option<i32>,
    pub entry_sqrt_price: f64,
    pub entry_price: f64,
    pub entry_amount_a: f64,
    pub entry_amount_b: f64,
    pub current_amount_a: f64,
    pub current_amount_b: f64,
    pub fees_earned_a: f64,
    pub fees_earned_b: f64,
    pub fees_earned_usd: f64,
    pub impermanent_loss_usd: f64,
    pub gas_costs_usd: f64,
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub rebalance_count: u32,
    pub is_active: bool,
}

impl LpPosition {
    pub fn net_value_usd(&self, price: f64) -> f64 {
        let position_value = self.current_amount_a * price + self.current_amount_b;
        position_value + self.fees_earned_usd - self.gas_costs_usd
    }

    pub fn hodl_value_usd(&self, price: f64) -> f64 {
        self.entry_amount_a * price + self.entry_amount_b
    }

    pub fn net_pnl_usd(&self, price: f64) -> f64 {
        self.net_value_usd(price) - self.hodl_value_usd(self.entry_price)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LpPortfolioState {
    pub positions: Vec<LpPosition>,
    pub closed_positions: Vec<LpPosition>,
    pub total_fees_earned_usd: f64,
    pub total_il_usd: f64,
    pub total_gas_costs_usd: f64,
    pub balance_usd: f64,
}

impl LpPortfolioState {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance_usd: initial_balance,
            ..Default::default()
        }
    }

    pub fn active_positions(&self) -> impl Iterator<Item = &LpPosition> {
        self.positions.iter().filter(|p| p.is_active)
    }

    pub fn active_positions_mut(&mut self) -> impl Iterator<Item = &mut LpPosition> {
        self.positions.iter_mut().filter(|p| p.is_active)
    }

    pub fn total_value_usd(&self, price: f64) -> f64 {
        let positions_value: f64 = self.positions.iter()
            .filter(|p| p.is_active)
            .map(|p| p.net_value_usd(price))
            .sum();
        self.balance_usd + positions_value
    }
}
