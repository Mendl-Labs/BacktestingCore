//! Feature expression mini-language compiler.
//!
//! Compiles user-declared feature expressions (from Python `features()`) into
//! Rust closures that can be evaluated per-tick without touching Python.
//!
//! ## Supported expressions
//!
//! | Expression               | Compiled form                                        |
//! |--------------------------|------------------------------------------------------|
//! | `mean(window[-N:])`      | `state.float_windows["window"].tail_mean(N)`         |
//! | `std(window[-N:])`       | `state.float_windows["window"].tail_std(N)`          |
//! | `max(window[-N:])`       | `state.float_windows["window"].tail_max(N)`          |
//! | `min(window[-N:])`       | `state.float_windows["window"].tail_min(N)`          |
//! | `window[-1]`             | `state.float_windows["window"].back()`               |
//! | `window[-N]`             | `state.float_windows["window"].nth_from_back(N-1)`   |
//! | `len(window)`            | `state.float_windows["window"].len() as f64`         |
//! | `encode_enum(field)`     | `state.enums["field"].0 as f64`                      |
//! | `context.orderbook.*`    | Direct field access on `StrategyContext`              |
//! | `context.portfolio.*`    | Direct field access on portfolio state                |
//! | `context.custom_data.*`  | `context.custom_data["key"]`                          |
//! | `A + B`, `A - B`, etc.   | Arithmetic on sub-expressions                        |
//!
//! Expressions that fail compilation return `Err(CompileError)`, which triggers
//! a graceful downgrade from Tier 0 to Tier 1 (Rust state + Python logic).

use std::sync::Arc;

use crate::state_store::StateStore;
use crate::traits::StrategyContext;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A compiled feature expression: a closure from (state, context) → f64.
///
/// Wrapped in `Arc` so it can be cloned into the feature vector evaluation pipeline.
pub type CompiledExpr = Arc<dyn Fn(&StateStore, &StrategyContext) -> f64 + Send + Sync>;

/// Error from compiling a feature expression.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub expr: String,
    pub reason: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot compile '{}': {}", self.expr, self.reason)
    }
}

impl std::error::Error for CompileError {}

/// A successfully compiled feature with its name.
#[derive(Clone)]
pub struct CompiledFeature {
    pub name: String,
    pub expr_source: String,
    pub compiled: CompiledExpr,
}

impl std::fmt::Debug for CompiledFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFeature")
            .field("name", &self.name)
            .field("expr_source", &self.expr_source)
            .finish()
    }
}

/// Result of compiling a full features() list.
#[derive(Debug)]
pub struct CompilationResult {
    /// Successfully compiled features (in order).
    pub compiled: Vec<CompiledFeature>,
    /// Expressions that failed to compile (with reasons).
    pub errors: Vec<CompileError>,
    /// Whether ALL features compiled successfully (Tier 0 eligible).
    pub all_compiled: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compile a list of `(name, expr)` pairs into executable closures.
///
/// Returns a [`CompilationResult`] with both successes and failures.
/// If any expression fails, `all_compiled` is false → strategy falls back to Tier 1.
pub fn compile_features(
    features: &[(String, String)],
) -> CompilationResult {
    let mut compiled = Vec::with_capacity(features.len());
    let mut errors = Vec::new();

    for (name, expr_str) in features {
        match compile_expr(expr_str.trim()) {
            Ok(closure) => {
                compiled.push(CompiledFeature {
                    name: name.clone(),
                    expr_source: expr_str.clone(),
                    compiled: closure,
                });
            }
            Err(reason) => {
                crate::log_warn!(crate::logging_facade::STRATEGY_LOGGER, "[FEATURE_COMPILER] Failed to compile '{}' = '{}': {}", name, expr_str, reason);
                errors.push(CompileError {
                    expr: expr_str.clone(),
                    reason,
                });
            }
        }
    }

    let all_compiled = errors.is_empty();
    CompilationResult {
        compiled,
        errors,
        all_compiled,
    }
}

/// Evaluate all compiled features into a f64 vector (for ONNX input).
pub fn evaluate_features(
    features: &[CompiledFeature],
    state: &StateStore,
    context: &StrategyContext,
) -> Vec<f64> {
    features.iter().map(|f| (f.compiled)(state, context)).collect()
}

// ---------------------------------------------------------------------------
// Expression compiler — recursive descent parser
// ---------------------------------------------------------------------------

/// Compile a single expression string into a closure.
fn compile_expr(expr: &str) -> Result<CompiledExpr, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("Empty expression".into());
    }

    // Try to parse as an additive expression (handles +, -, *, /)
    let tokens = tokenize(expr)?;
    let (compiled, rest) = parse_additive(&tokens)?;
    if !rest.is_empty() {
        return Err(format!("Unexpected tokens after expression: {:?}", rest));
    }
    Ok(compiled)
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),     // variable name or function name
    Number(f64),       // numeric literal
    LParen,            // (
    RParen,            // )
    LBracket,          // [
    RBracket,          // ]
    Colon,             // :
    Dot,               // .
    Plus,              // +
    Minus,             // -
    Star,              // *
    Slash,             // /
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            ':' => {
                tokens.push(Token::Colon);
                i += 1;
            }
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                // Could be negative number or minus operator
                // If preceded by a number/ident/rparen/rbracket, it's minus
                // Otherwise it's a negative sign for a number
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(Token::LParen)
                            | Some(Token::LBracket)
                            | Some(Token::Plus)
                            | Some(Token::Minus)
                            | Some(Token::Star)
                            | Some(Token::Slash)
                            | Some(Token::Colon)
                    );

                if is_unary && i + 1 < chars.len() && (chars[i + 1].is_ascii_digit()) {
                    // Negative number
                    let start = i;
                    i += 1;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let num: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Invalid number: {}", num_str))?;
                    tokens.push(Token::Number(num));
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphanumeric() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            other => {
                return Err(format!("Unexpected character: '{}'", other));
            }
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Recursive descent parser
// ---------------------------------------------------------------------------

type ParseResult<'a> = Result<(CompiledExpr, &'a [Token]), String>;

/// additive = multiplicative (('+' | '-') multiplicative)*
fn parse_additive<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    let (mut left, mut rest) = parse_multiplicative(tokens)?;

    while !rest.is_empty() {
        match rest.first() {
            Some(Token::Plus) => {
                let (right, r) = parse_multiplicative(&rest[1..])?;
                let l = left;
                left = Arc::new(move |s: &StateStore, c: &StrategyContext| (l)(s, c) + (right)(s, c));
                rest = r;
            }
            Some(Token::Minus) => {
                let (right, r) = parse_multiplicative(&rest[1..])?;
                let l = left;
                left = Arc::new(move |s: &StateStore, c: &StrategyContext| (l)(s, c) - (right)(s, c));
                rest = r;
            }
            _ => break,
        }
    }

    Ok((left, rest))
}

/// multiplicative = unary (('*' | '/') unary)*
fn parse_multiplicative<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    let (mut left, mut rest) = parse_unary(tokens)?;

    while !rest.is_empty() {
        match rest.first() {
            Some(Token::Star) => {
                let (right, r) = parse_unary(&rest[1..])?;
                let l = left;
                left = Arc::new(move |s: &StateStore, c: &StrategyContext| (l)(s, c) * (right)(s, c));
                rest = r;
            }
            Some(Token::Slash) => {
                let (right, r) = parse_unary(&rest[1..])?;
                let l = left;
                left = Arc::new(move |s: &StateStore, c: &StrategyContext| {
                    let denom = (right)(s, c);
                    if denom.abs() < 1e-15 {
                        0.0 // prevent division by zero
                    } else {
                        (l)(s, c) / denom
                    }
                });
                rest = r;
            }
            _ => break,
        }
    }

    Ok((left, rest))
}

/// unary = '-' unary | primary
fn parse_unary<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    if let Some(Token::Minus) = tokens.first() {
        let (inner, rest) = parse_unary(&tokens[1..])?;
        let expr: CompiledExpr = Arc::new(move |s: &StateStore, c: &StrategyContext| -(inner)(s, c));
        return Ok((expr, rest));
    }
    parse_primary(tokens)
}

/// primary = number | function_call | context_access | window_index | '(' additive ')'
fn parse_primary<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    match tokens.first() {
        Some(Token::Number(n)) => {
            let val = *n;
            Ok((Arc::new(move |_, _| val), &tokens[1..]))
        }
        Some(Token::LParen) => {
            let (inner, rest) = parse_additive(&tokens[1..])?;
            match rest.first() {
                Some(Token::RParen) => Ok((inner, &rest[1..])),
                _ => Err("Expected closing ')'".into()),
            }
        }
        Some(Token::Ident(name)) => {
            let name = name.clone();

            // Check for function call: name(...)
            if tokens.len() > 1 && tokens[1] == Token::LParen {
                return parse_function_call(&name, &tokens[2..]);
            }

            // Check for dot-access: context.xxx.yyy
            if name == "context" && tokens.len() > 1 && tokens[1] == Token::Dot {
                return parse_context_access(&tokens[2..]);
            }

            // Check for window index: window[-N] or window[-N:]
            if tokens.len() > 1 && tokens[1] == Token::LBracket {
                return parse_window_index(&name, &tokens[2..]);
            }

            // Bare identifier — treat as state float variable
            let var_name = name.clone();
            Ok((
                Arc::new(move |s: &StateStore, _| {
                    s.get_float(&var_name).unwrap_or(0.0)
                }),
                &tokens[1..],
            ))
        }
        _ => Err(format!(
            "Unexpected token: {:?}",
            tokens.first().unwrap_or(&Token::RParen)
        )),
    }
}

/// Parse a function call: mean(...), std(...), max(...), min(...), len(...), encode_enum(...)
fn parse_function_call<'a>(func_name: &str, tokens: &'a [Token]) -> ParseResult<'a> {
    // tokens start after the opening '('
    match func_name {
        "mean" | "std" | "max" | "min" | "sum" => {
            // Expected: window_name[-N:])  or  window_name)
            let (window_name, tail_n, rest) = parse_window_arg(tokens)?;
            let wn = window_name.clone();
            let func = func_name.to_string();

            let expr: CompiledExpr = Arc::new(move |s: &StateStore, _| {
                match func.as_str() {
                    "mean" => {
                        if let Some(n) = tail_n {
                            s.window_tail_mean(&wn, n).unwrap_or(0.0)
                        } else {
                            s.window_mean(&wn).unwrap_or(0.0)
                        }
                    }
                    "std" => {
                        if let Some(n) = tail_n {
                            s.window_tail_std(&wn, n).unwrap_or(0.0)
                        } else {
                            s.window_std(&wn).unwrap_or(0.0)
                        }
                    }
                    "max" => {
                        if let Some(n) = tail_n {
                            s.window_tail_max(&wn, n).unwrap_or(0.0)
                        } else {
                            s.window_tail_max(&wn, usize::MAX).unwrap_or(0.0)
                        }
                    }
                    "min" => {
                        if let Some(n) = tail_n {
                            s.window_tail_min(&wn, n).unwrap_or(0.0)
                        } else {
                            s.window_tail_min(&wn, usize::MAX).unwrap_or(0.0)
                        }
                    }
                    "sum" => {
                        if let Some(w) = s.float_windows.get(&wn) {
                            if let Some(n) = tail_n {
                                w.tail_sum(n)
                            } else {
                                w.iter().sum()
                            }
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                }
            });
            Ok((expr, rest))
        }
        "len" => {
            // len(window_name)
            let (window_name, rest) = parse_simple_ident_arg(tokens)?;
            let wn = window_name.clone();
            let expr: CompiledExpr = Arc::new(move |s: &StateStore, _| {
                s.window_len(&wn).unwrap_or(0) as f64
            });
            Ok((expr, rest))
        }
        "encode_enum" => {
            // encode_enum(field_name)
            let (field_name, rest) = parse_simple_ident_arg(tokens)?;
            let fn_name = field_name.clone();
            let expr: CompiledExpr = Arc::new(move |s: &StateStore, _| {
                s.get_enum_index(&fn_name).unwrap_or(0) as f64
            });
            Ok((expr, rest))
        }
        "abs" => {
            // abs(expr)
            let (inner, rest) = parse_additive(tokens)?;
            match rest.first() {
                Some(Token::RParen) => {
                    let expr: CompiledExpr =
                        Arc::new(move |s: &StateStore, c: &StrategyContext| (inner)(s, c).abs());
                    Ok((expr, &rest[1..]))
                }
                _ => Err("Expected ')' after abs argument".into()),
            }
        }
        _ => Err(format!("Unknown function: {}", func_name)),
    }
}

/// Parse a window argument like `window_name[-N:]` or just `window_name`.
/// Returns (window_name, optional_tail_count, remaining_tokens).
fn parse_window_arg<'a>(
    tokens: &'a [Token],
) -> Result<(String, Option<usize>, &'a [Token]), String> {
    // Expect: Ident [ - Number : ] )
    //     or: Ident )
    let window_name = match tokens.first() {
        Some(Token::Ident(name)) => name.clone(),
        _ => return Err("Expected window name in function argument".into()),
    };

    let rest = &tokens[1..];

    // Check for [-N:]
    if rest.len() >= 4
        && rest[0] == Token::LBracket
        && matches!(&rest[1], Token::Number(_))
        && rest[2] == Token::Colon
        && rest[3] == Token::RBracket
    {
        let n = match &rest[1] {
            Token::Number(n) => n.abs() as usize,
            _ => unreachable!(),
        };
        // Expect ) after ]
        let after = &rest[4..];
        match after.first() {
            Some(Token::RParen) => Ok((window_name, Some(n), &after[1..])),
            _ => Err("Expected ')' after window slice".into()),
        }
    } else if rest.first() == Some(&Token::RParen) {
        // Just window_name)
        Ok((window_name, None, &rest[1..]))
    } else {
        Err("Expected ')' or '[-N:]' after window name".into())
    }
}

/// Parse a simple identifier argument: `ident_name )`.
fn parse_simple_ident_arg<'a>(tokens: &'a [Token]) -> Result<(String, &'a [Token]), String> {
    let name = match tokens.first() {
        Some(Token::Ident(name)) => name.clone(),
        _ => return Err("Expected identifier argument".into()),
    };
    match tokens.get(1) {
        Some(Token::RParen) => Ok((name, &tokens[2..])),
        _ => Err("Expected ')' after identifier argument".into()),
    }
}

/// Parse context.xxx.yyy access (tokens start after "context" + ".").
fn parse_context_access<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    let first = match tokens.first() {
        Some(Token::Ident(name)) => name.clone(),
        _ => return Err("Expected field name after 'context.'".into()),
    };

    match first.as_str() {
        "orderbook" => {
            // context.orderbook.field
            if tokens.len() < 3 || tokens[1] != Token::Dot {
                return Err("Expected 'context.orderbook.<field>'".into());
            }
            let field = match &tokens[2] {
                Token::Ident(f) => f.clone(),
                _ => return Err("Expected field name after 'context.orderbook.'".into()),
            };
            let expr: CompiledExpr = match field.as_str() {
                "spread" => Arc::new(|_, c: &StrategyContext| {
                    c.orderbook_snapshot
                        .as_ref()
                        .map(|ob| ob.spread)
                        .unwrap_or(0.0)
                }),
                "mid_price" => Arc::new(|_, c: &StrategyContext| {
                    c.orderbook_snapshot
                        .as_ref()
                        .map(|ob| ob.mid_price)
                        .unwrap_or(0.0)
                }),
                "imbalance" => Arc::new(|_, c: &StrategyContext| {
                    c.orderbook_snapshot
                        .as_ref()
                        .map(|ob| ob.imbalance)
                        .unwrap_or(0.0)
                }),
                "bid_depth" => Arc::new(|_, c: &StrategyContext| {
                    c.orderbook_snapshot
                        .as_ref()
                        .map(|ob| ob.bid_depth)
                        .unwrap_or(0.0)
                }),
                "ask_depth" => Arc::new(|_, c: &StrategyContext| {
                    c.orderbook_snapshot
                        .as_ref()
                        .map(|ob| ob.ask_depth)
                        .unwrap_or(0.0)
                }),
                _ => return Err(format!("Unknown orderbook field: '{}'", field)),
            };
            Ok((expr, &tokens[3..]))
        }
        "portfolio" => {
            // context.portfolio.field
            if tokens.len() < 3 || tokens[1] != Token::Dot {
                return Err("Expected 'context.portfolio.<field>'".into());
            }
            let field = match &tokens[2] {
                Token::Ident(f) => f.clone(),
                _ => return Err("Expected field name after 'context.portfolio.'".into()),
            };
            let expr: CompiledExpr = match field.as_str() {
                "unrealized_pnl" => Arc::new(|_, c: &StrategyContext| {
                    c.portfolio_state.unrealized_pnl
                }),
                "realized_pnl" => Arc::new(|_, c: &StrategyContext| {
                    c.portfolio_state.realized_pnl
                }),
                "balance" => Arc::new(|_, c: &StrategyContext| {
                    c.portfolio_state.balance
                }),
                "net_position" => {
                    Arc::new(|_, c: &StrategyContext| c.portfolio_state.net_position)
                }
                "fees_paid" => Arc::new(|_, c: &StrategyContext| {
                    c.portfolio_state.fees_paid
                }),
                _ => return Err(format!("Unknown portfolio field: '{}'", field)),
            };
            Ok((expr, &tokens[3..]))
        }
        "custom_data" => {
            // context.custom_data.key_name
            if tokens.len() < 3 || tokens[1] != Token::Dot {
                return Err("Expected 'context.custom_data.<key>'".into());
            }
            let key = match &tokens[2] {
                Token::Ident(k) => k.clone(),
                _ => return Err("Expected key name after 'context.custom_data.'".into()),
            };
            let k = key.clone();
            let expr: CompiledExpr =
                Arc::new(move |_, c: &StrategyContext| c.custom_data.get(&k).copied().unwrap_or(0.0));
            Ok((expr, &tokens[3..]))
        }
        "tick_number" => {
            let expr: CompiledExpr = Arc::new(|_, c: &StrategyContext| c.tick_number as f64);
            Ok((expr, &tokens[1..]))
        }
        "elapsed_seconds" => {
            let expr: CompiledExpr = Arc::new(|_, c: &StrategyContext| c.elapsed_seconds);
            Ok((expr, &tokens[1..]))
        }
        _ => Err(format!("Unknown context field: '{}'", first)),
    }
}

/// Parse window[-N] index access (tokens start after "[").
fn parse_window_index<'a>(window_name: &str, tokens: &'a [Token]) -> ParseResult<'a> {
    // Expect: -N ] or -N : ]
    // window[-N] → nth_from_back(N-1)
    // window[-N:] → not supported as primary (only in function args)
    let n = match tokens.first() {
        Some(Token::Number(n)) => n.abs() as usize,
        _ => return Err(format!("Expected number in window index for '{}'", window_name)),
    };

    let rest = &tokens[1..];
    match rest.first() {
        Some(Token::RBracket) => {
            let wn = window_name.to_string();
            let idx = if n > 0 { n - 1 } else { 0 };
            let expr: CompiledExpr = Arc::new(move |s: &StateStore, _| {
                s.window_nth_from_back(&wn, idx).unwrap_or(0.0)
            });
            Ok((expr, &rest[1..]))
        }
        _ => Err(format!(
            "Expected ']' after window index for '{}'",
            window_name
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_store::{StateSchema, StateStore};
    use crate::traits::{StrategyContext, OrderbookSnapshot};
    use std::collections::HashMap;
    use chrono::Utc;

    fn test_schema() -> StateSchema {
        let mut raw = HashMap::new();

        let mut w = HashMap::new();
        w.insert("type".into(), serde_json::json!("window[float]"));
        w.insert("capacity".into(), serde_json::json!(100));
        raw.insert("recent_prices".into(), w);

        let mut w2 = HashMap::new();
        w2.insert("type".into(), serde_json::json!("window[float]"));
        w2.insert("capacity".into(), serde_json::json!(100));
        raw.insert("recent_volumes".into(), w2);

        let mut e = HashMap::new();
        e.insert("type".into(), serde_json::json!("enum"));
        e.insert("values".into(), serde_json::json!(["flat", "long", "short"]));
        raw.insert("position_side".into(), e);

        let mut f = HashMap::new();
        f.insert("type".into(), serde_json::json!("float"));
        raw.insert("threshold".into(), f);

        StateSchema::from_raw_dict(&raw).unwrap()
    }

    fn test_context() -> StrategyContext {
        StrategyContext {
            timestamp: Utc::now(),
            portfolio_state: Default::default(),
            price_history: vec![],
            volume_history: vec![],
            spread: None,
            market_session: crate::traits::MarketSession {
                is_open: true,
                session_type: crate::traits::SessionType::Regular,
                next_open: None,
                next_close: None,
            },
            orderbook_snapshot: Some(OrderbookSnapshot {
                bids: vec![],
                asks: vec![],
                spread: 0.5,
                mid_price: 100.0,
                bid_depth: 1000.0,
                ask_depth: 900.0,
                imbalance: 0.1,
            }),
            custom_data: {
                let mut m = HashMap::new();
                m.insert("sentiment".into(), 0.75);
                m
            },
            assets: HashMap::new(),
            tick_number: 42,
            elapsed_seconds: 3600.0,
            iv_surface: None,
            futures_curve: None,
            funding_rates: None,
            portfolio_greeks: None,
            pool_state: None,
        }
    }

    fn populated_store(schema: &StateSchema) -> StateStore {
        let mut store = StateStore::from_schema(schema);
        for i in 1..=20 {
            store.window_push("recent_prices", i as f64);
        }
        for i in 1..=20 {
            store.window_push("recent_volumes", (i * 100) as f64);
        }
        store.set_enum("position_side", "long");
        store.set_float("threshold", 0.002);
        store
    }

    #[test]
    fn test_compile_mean() {
        let result = compile_features(&[
            ("sma_20".into(), "mean(recent_prices[-20:])".into()),
        ]);
        assert!(result.all_compiled, "errors: {:?}", result.errors);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        // mean(1..20) = 10.5
        assert!((values[0] - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_compile_arithmetic() {
        let result = compile_features(&[
            ("vol_ratio".into(), "mean(recent_volumes[-5:]) / mean(recent_volumes[-20:])".into()),
        ]);
        assert!(result.all_compiled, "errors: {:?}", result.errors);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        // volumes: 100,200,...,2000
        // last 5: 1600,1700,1800,1900,2000 → mean = 1800
        // last 20: 100..2000 → mean = 1050
        // ratio = 1800/1050 ≈ 1.714
        assert!((values[0] - 1800.0 / 1050.0).abs() < 0.01);
    }

    #[test]
    fn test_compile_context_access() {
        let result = compile_features(&[
            ("spread".into(), "context.orderbook.spread".into()),
            ("imbalance".into(), "context.orderbook.imbalance".into()),
            ("position".into(), "encode_enum(position_side)".into()),
            ("tick".into(), "context.tick_number".into()),
            ("sentiment".into(), "context.custom_data.sentiment".into()),
        ]);
        assert!(result.all_compiled, "errors: {:?}", result.errors);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        assert!((values[0] - 0.5).abs() < 1e-10); // spread
        assert!((values[1] - 0.1).abs() < 1e-10); // imbalance
        assert!((values[2] - 1.0).abs() < 1e-10); // encode_enum("long") = 1
        assert!((values[3] - 42.0).abs() < 1e-10); // tick_number
        assert!((values[4] - 0.75).abs() < 1e-10); // custom_data.sentiment
    }

    #[test]
    fn test_compile_window_index() {
        let result = compile_features(&[
            ("last_price".into(), "recent_prices[-1]".into()),
            ("prev_price".into(), "recent_prices[-2]".into()),
        ]);
        assert!(result.all_compiled, "errors: {:?}", result.errors);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        assert!((values[0] - 20.0).abs() < 1e-10); // last = 20
        assert!((values[1] - 19.0).abs() < 1e-10); // second-to-last = 19
    }

    #[test]
    fn test_compile_number_literal() {
        let result = compile_features(&[
            ("const".into(), "3.14".into()),
            ("expr".into(), "2.0 * threshold + 1.0".into()),
        ]);
        assert!(result.all_compiled, "errors: {:?}", result.errors);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        assert!((values[0] - 3.14).abs() < 1e-10);
        assert!((values[1] - (2.0 * 0.002 + 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compile_failure() {
        let result = compile_features(&[
            ("bad".into(), "unknown_function(x)".into()),
        ]);
        assert!(!result.all_compiled);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_division_by_zero() {
        let result = compile_features(&[
            ("safe".into(), "1.0 / 0.0".into()),
        ]);
        assert!(result.all_compiled);

        let schema = test_schema();
        let store = populated_store(&schema);
        let ctx = test_context();

        let values = evaluate_features(&result.compiled, &store, &ctx);
        assert_eq!(values[0], 0.0); // protected
    }
}
