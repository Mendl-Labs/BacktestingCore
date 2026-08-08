//! Declarative MM Pricing — Tier 0.5 executor
//!
//! Evaluates bid/ask pricing formulas in pure Rust using StateStore +
//! IndicatorBank values, avoiding Python on the hot path entirely.
//!
//! A Python strategy opts in by defining a `pricing()` method returning a dict
//! of expression strings:
//!
//! ```python
//! def pricing(self):
//!     return {
//!         "bid_price": "mid - spread / 2",
//!         "ask_price": "mid + spread / 2",
//!         "bid_size": "order_size",
//!         "ask_size": "order_size",
//!     }
//! ```

use std::collections::HashMap;
use crate::state_store::StateStore;
use crate::indicator_registry::IndicatorBank;

// ───────────────────────── Expression AST ──────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    UnaryNeg(Box<Expr>),
    Func(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

// ───────────────────────── Tokenizer ───────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => { i += 1; }
            '+' => { tokens.push(Token::Plus); i += 1; }
            '-' => { tokens.push(Token::Minus); i += 1; }
            '*' => { tokens.push(Token::Star); i += 1; }
            '/' => { tokens.push(Token::Slash); i += 1; }
            '(' => { tokens.push(Token::LParen); i += 1; }
            ')' => { tokens.push(Token::RParen); i += 1; }
            ',' => { tokens.push(Token::Comma); i += 1; }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str.parse()
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '.') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            c => return Err(format!("Unexpected character: '{}'", c)),
        }
    }
    Ok(tokens)
}

// ───────────────────────── Parser ──────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos)?.clone();
        self.pos += 1;
        Some(tok)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(format!("Expected {:?}, got {:?}", expected, tok)),
            None => Err(format!("Expected {:?}, got end of input", expected)),
        }
    }

    /// expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    let right = self.parse_term()?;
                    left = Expr::BinOp(Box::new(left), Op::Add, Box::new(right));
                }
                Some(Token::Minus) => {
                    self.advance();
                    let right = self.parse_term()?;
                    left = Expr::BinOp(Box::new(left), Op::Sub, Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = Expr::BinOp(Box::new(left), Op::Mul, Box::new(right));
                }
                Some(Token::Slash) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = Expr::BinOp(Box::new(left), Op::Div, Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// unary = '-' unary | atom
    fn parse_unary(&mut self) -> Result<Expr, String> {
        if let Some(Token::Minus) = self.peek() {
            self.advance();
            let inner = self.parse_unary()?;
            return Ok(Expr::UnaryNeg(Box::new(inner)));
        }
        self.parse_atom()
    }

    /// atom = NUMBER | IDENT '(' args ')' | IDENT | '(' expr ')'
    fn parse_atom(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Some(Token::Number(n)) => Ok(Expr::Literal(n)),
            Some(Token::Ident(name)) => {
                // Check for function call
                if let Some(Token::LParen) = self.peek() {
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    if self.peek() != Some(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while let Some(Token::Comma) = self.peek() {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Func(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Some(tok) => Err(format!("Unexpected token: {:?}", tok)),
            None => Err("Unexpected end of expression".into()),
        }
    }
}

/// Parse an expression string into an AST.
pub fn parse_expr(input: &str) -> Result<Expr, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err("Empty expression".into());
    }
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        return Err(format!("Unexpected trailing tokens starting at {:?}", parser.tokens[parser.pos]));
    }
    Ok(expr)
}

// ───────────────────────── Evaluator ───────────────────────────────

/// Variables available during pricing evaluation.
pub struct PricingVars<'a> {
    pub mid: f64,
    pub spread: f64,
    pub inventory: f64,
    pub position_value: f64,
    pub order_size: f64,
    pub volatility: f64,
    pub state: Option<&'a StateStore>,
    pub indicators: Option<&'a HashMap<String, f64>>,
}

/// Evaluate a compiled expression given the current pricing context.
pub fn eval_expr(expr: &Expr, vars: &PricingVars) -> f64 {
    match expr {
        Expr::Literal(n) => *n,
        Expr::Var(name) => resolve_var(name, vars),
        Expr::BinOp(lhs, op, rhs) => {
            let l = eval_expr(lhs, vars);
            let r = eval_expr(rhs, vars);
            match op {
                Op::Add => l + r,
                Op::Sub => l - r,
                Op::Mul => l * r,
                Op::Div => if r.abs() > f64::EPSILON { l / r } else { 0.0 },
            }
        }
        Expr::UnaryNeg(inner) => -eval_expr(inner, vars),
        Expr::Func(name, args) => {
            let evaluated: Vec<f64> = args.iter().map(|a| eval_expr(a, vars)).collect();
            match name.as_str() {
                "abs" if evaluated.len() == 1 => evaluated[0].abs(),
                "min" if evaluated.len() == 2 => evaluated[0].min(evaluated[1]),
                "max" if evaluated.len() == 2 => evaluated[0].max(evaluated[1]),
                "sqrt" if evaluated.len() == 1 => evaluated[0].sqrt(),
                "ln" if evaluated.len() == 1 => evaluated[0].ln(),
                _ => 0.0, // Unknown function — return 0
            }
        }
    }
}

fn resolve_var(name: &str, vars: &PricingVars) -> f64 {
    match name {
        "mid" | "mid_price" => vars.mid,
        "spread" => vars.spread,
        "inventory" | "inv" => vars.inventory,
        "position_value" => vars.position_value,
        "order_size" => vars.order_size,
        "volatility" | "vol" => vars.volatility,
        _ => {
            // Try state.xxx
            if let Some(state) = vars.state {
                if let Some(stripped) = name.strip_prefix("state.") {
                    if let Some(v) = state.floats.get(stripped) {
                        return *v;
                    }
                }
            }
            // Try ind.xxx or indicator name directly
            if let Some(indicators) = vars.indicators {
                if let Some(stripped) = name.strip_prefix("ind.") {
                    if let Some(v) = indicators.get(stripped) {
                        return *v;
                    }
                }
                if let Some(v) = indicators.get(name) {
                    return *v;
                }
            }
            0.0
        }
    }
}

// ───────────────────────── PricingFormula ───────────────────────────

/// Compiled pricing formula for an MM strategy.
#[derive(Debug, Clone)]
pub struct PricingFormula {
    pub bid_price: Expr,
    pub ask_price: Expr,
    pub bid_size: Expr,
    pub ask_size: Expr,
}

impl PricingFormula {
    /// Parse a pricing dict (string expressions) into compiled ASTs.
    pub fn compile(raw: &HashMap<String, String>) -> Result<Self, String> {
        let bid_price = raw.get("bid_price")
            .ok_or("Missing 'bid_price' in pricing()")?;
        let ask_price = raw.get("ask_price")
            .ok_or("Missing 'ask_price' in pricing()")?;
        let default_size = "order_size".to_string();
        let bid_size = raw.get("bid_size").unwrap_or(&default_size);
        let ask_size = raw.get("ask_size").unwrap_or(&default_size);

        Ok(Self {
            bid_price: parse_expr(bid_price).map_err(|e| format!("bid_price: {}", e))?,
            ask_price: parse_expr(ask_price).map_err(|e| format!("ask_price: {}", e))?,
            bid_size: parse_expr(bid_size).map_err(|e| format!("bid_size: {}", e))?,
            ask_size: parse_expr(ask_size).map_err(|e| format!("ask_size: {}", e))?,
        })
    }
}

// ───────────────────────── Extract pricing() from Python ───────────

/// Extract `pricing()` dict from Python strategy source.
/// Returns `None` if the strategy doesn't define `pricing()`.
#[cfg(feature = "python")]
pub fn extract_pricing(python_source: &str) -> Option<HashMap<String, String>> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyModule};

    Python::with_gil(|py| {
        crate::strategies::python_strategy::inject_sdk(py).ok()?;

        // Unique name per call -- see python_strategy.rs's
        // unique_module_name doc comment for why a fixed literal here leaked
        // a previous job's stale Strategy class into this compilation.
        let module =
            PyModule::from_code_bound(py, python_source, "strategy.py", &crate::strategies::python_strategy::unique_module_name()).ok()?;

        let strategy_class = module.getattr("Strategy").ok()?;
        let instance = strategy_class.call0().ok()?;

        if !instance.hasattr("pricing").unwrap_or(false) {
            return None;
        }

        let pricing_result = instance.call_method0("pricing").ok()?;
        let pricing_dict = pricing_result.downcast::<PyDict>().ok()?;

        let mut result = HashMap::new();
        for (key, value) in pricing_dict.iter() {
            let k: String = key.extract().ok()?;
            let v: String = value.extract().ok()?;
            result.insert(k, v);
        }

        if result.contains_key("bid_price") && result.contains_key("ask_price") {
            Some(result)
        } else {
            None
        }
    })
}

#[cfg(not(feature = "python"))]
pub fn extract_pricing(_python_source: &str) -> Option<HashMap<String, String>> {
    None
}

// ───────────────────────── MMPricingExecutor ────────────────────────

use async_trait::async_trait;
use crate::executor::{ExecutionTier, StrategyExecutor};
use crate::traits::{StrategyContext, StrategyResult};
use signal::{Signal, SignalStrength, SignalReason};
use dataloader::MarketData;

/// Tier 0.5 executor: evaluates declarative pricing formulas in pure Rust.
/// Uses StateStore + IndicatorBank for variable resolution.
pub struct MMPricingExecutor {
    pub state_store: StateStore,
    pub schema: crate::state_store::StateSchema,
    pub formula: PricingFormula,
    pub indicator_bank: Option<IndicatorBank>,
    pub price_windows: Vec<String>,
    pub volume_windows: Vec<String>,
    pub(crate) pre_enriched: bool,
}

impl MMPricingExecutor {
    fn extract_price_volume(data: &MarketData) -> (f64, f64) {
        match data {
            MarketData::Trade(t) => (t.price, t.quantity),
            MarketData::Candle(c) => (c.close, c.volume),
            MarketData::PoolSwap(s) => (s.price(), s.amount_in),
            MarketData::Generic(g) => (g.price, g.quantity),
            MarketData::OptionCandle(c) => (c.close, c.volume),
        }
    }
}

#[async_trait]
impl StrategyExecutor for MMPricingExecutor {
    fn tier(&self) -> ExecutionTier {
        ExecutionTier::Tier0_5
    }

    async fn on_tick(
        &mut self,
        data: &MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let (price, volume) = Self::extract_price_volume(data);

        if price <= 0.0 {
            return Ok(vec![]);
        }

        // Update indicator bank
        if let Some(ref mut bank) = self.indicator_bank {
            bank.update_all(price, volume);
        }

        // Update state store windows
        if !self.pre_enriched {
            for w in &self.price_windows {
                self.state_store.window_push(w, price);
            }
            for w in &self.volume_windows {
                self.state_store.window_push(w, volume);
            }
        }
        self.pre_enriched = false;

        // Resolve spread from the context or a simple estimate
        let spread = context.spread.unwrap_or(price * 0.001);

        let inventory = context.portfolio_state.net_position;
        let position_value = inventory.abs() * price;
        let order_size = context.portfolio_state.balance * 0.01; // 1% of balance default

        let indicator_vals = self.indicator_bank.as_ref().map(|b| b.values().clone());
        let vars = PricingVars {
            mid: price,
            spread,
            inventory,
            position_value,
            order_size,
            volatility: 0.0,
            state: Some(&self.state_store),
            indicators: indicator_vals.as_ref(),
        };

        let bid_price = eval_expr(&self.formula.bid_price, &vars);
        let ask_price = eval_expr(&self.formula.ask_price, &vars);
        let bid_size = eval_expr(&self.formula.bid_size, &vars);
        let ask_size = eval_expr(&self.formula.ask_size, &vars);

        if bid_price <= 0.0 || ask_price <= 0.0 || bid_price >= ask_price {
            return Ok(vec![]);
        }

        Ok(vec![Signal::two_sided_quote(
            "mm_pricing",
            bid_price,
            ask_price,
            bid_size.abs().max(f64::EPSILON),
            ask_size.abs().max(f64::EPSILON),
            SignalStrength::Strong,
            SignalReason::Custom("MM".into(), "Tier0.5 declarative pricing".into()),
        )])
    }

    fn pre_enrich(&mut self, data: &MarketData, custom_data: &mut HashMap<String, f64>) {
        let (price, volume) = Self::extract_price_volume(data);

        for w in &self.price_windows {
            self.state_store.window_push(w, price);
        }
        for w in &self.volume_windows {
            self.state_store.window_push(w, volume);
        }

        // Inject state values
        for (k, v) in &self.state_store.floats {
            custom_data.insert(format!("state.{}", k), *v);
        }

        // Inject indicator values
        if let Some(ref mut bank) = self.indicator_bank {
            bank.update_all(price, volume);
            for (k, v) in bank.values() {
                custom_data.insert(format!("ind.{}", k), *v);
            }
        }

        self.pre_enriched = true;
    }

    fn state_snapshot(&self) -> Option<&StateStore> {
        Some(&self.state_store)
    }

    fn tier_description(&self) -> String {
        "Tier 0.5 — Declarative MM Pricing (pure Rust formula evaluation)".into()
    }
}

// ───────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_expr() {
        let expr = parse_expr("mid - spread / 2").unwrap();
        let vars = PricingVars {
            mid: 100.0,
            spread: 0.5,
            inventory: 0.0,
            position_value: 0.0,
            order_size: 10.0,
            volatility: 0.0,
            state: None,
            indicators: None,
        };
        let result = eval_expr(&expr, &vars);
        assert!((result - 99.75).abs() < 1e-10);
    }

    #[test]
    fn test_parse_complex_expr() {
        let expr = parse_expr("mid - max(spread, volatility * 100) / 2").unwrap();
        let vars = PricingVars {
            mid: 100.0,
            spread: 0.5,
            inventory: 0.0,
            position_value: 0.0,
            order_size: 10.0,
            volatility: 0.01,
            state: None,
            indicators: None,
        };
        let result = eval_expr(&expr, &vars);
        // max(0.5, 0.01 * 100) = max(0.5, 1.0) = 1.0 → 100 - 1.0/2 = 99.5
        assert!((result - 99.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_unary_neg() {
        let expr = parse_expr("-inventory * mid").unwrap();
        let vars = PricingVars {
            mid: 50.0,
            spread: 0.0,
            inventory: 3.0,
            position_value: 0.0,
            order_size: 0.0,
            volatility: 0.0,
            state: None,
            indicators: None,
        };
        let result = eval_expr(&expr, &vars);
        assert!((result - (-150.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pricing_formula_compile() {
        let mut raw = HashMap::new();
        raw.insert("bid_price".into(), "mid - spread / 2".into());
        raw.insert("ask_price".into(), "mid + spread / 2".into());
        let formula = PricingFormula::compile(&raw).unwrap();
        let vars = PricingVars {
            mid: 100.0,
            spread: 1.0,
            inventory: 0.0,
            position_value: 0.0,
            order_size: 10.0,
            volatility: 0.0,
            state: None,
            indicators: None,
        };
        assert!((eval_expr(&formula.bid_price, &vars) - 99.5).abs() < 1e-10);
        assert!((eval_expr(&formula.ask_price, &vars) - 100.5).abs() < 1e-10);
    }

    #[test]
    fn test_indicator_var_resolution() {
        let expr = parse_expr("ind.sma_20 + 1.0").unwrap();
        let mut ind_vals = HashMap::new();
        ind_vals.insert("sma_20".into(), 50.0);
        let vars = PricingVars {
            mid: 0.0,
            spread: 0.0,
            inventory: 0.0,
            position_value: 0.0,
            order_size: 0.0,
            volatility: 0.0,
            state: None,
            indicators: Some(&ind_vals),
        };
        assert!((eval_expr(&expr, &vars) - 51.0).abs() < 1e-10);
    }
}
