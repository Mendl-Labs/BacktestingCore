//! Typed errors. Every public function of the crate returns `Result<_, EvalError>` for bad input and never panics
//! (property: no input, however degenerate, reaches an `unwrap`, an index out of range or a division that can panic).

use std::fmt;

/// Why a statistic could not be computed. A refusal, never a silent default: a statistic that is "0.5 because nothing
/// could be computed" is exactly the failure mode of the old DSR helper (memory: DSR trial-count bugs).
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// Fewer observations than the statistic needs.
    TooShort { what: &'static str, need: usize, got: usize },
    /// Two inputs that must have equal length do not.
    LengthMismatch { what: &'static str, left: usize, right: usize },
    /// A NaN or an infinity at `index` of the named input.
    NonFinite { what: &'static str, index: usize },
    /// The named series has (numerically) zero variance, so a Sharpe ratio or a regression is undefined.
    ZeroVariance { what: &'static str },
    /// A parameter is outside its documented domain.
    InvalidParameter { name: &'static str, reason: String },
    /// A linear system is singular (collinear benchmarks, constant regressor).
    Singular { what: &'static str },
    /// Too many bootstrap replicates were degenerate to report anything.
    DegenerateBootstrap { valid: usize, requested: usize },
    /// A sealed holdout window overlaps a range that was used for tuning or training.
    HoldoutOverlap { holdout_start: usize, holdout_end: usize, other_start: usize, other_end: usize },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::TooShort { what, need, got } => write!(f, "{what}: need at least {need} observations, got {got}"),
            EvalError::LengthMismatch { what, left, right } => {
                write!(f, "{what}: length mismatch ({left} vs {right})")
            }
            EvalError::NonFinite { what, index } => write!(f, "{what}: non-finite value at index {index}"),
            EvalError::ZeroVariance { what } => write!(f, "{what}: zero variance"),
            EvalError::InvalidParameter { name, reason } => write!(f, "invalid parameter `{name}`: {reason}"),
            EvalError::Singular { what } => write!(f, "{what}: singular system"),
            EvalError::DegenerateBootstrap { valid, requested } => {
                write!(f, "only {valid} of {requested} bootstrap replicates were usable")
            }
            EvalError::HoldoutOverlap { holdout_start, holdout_end, other_start, other_end } => write!(
                f,
                "sealed holdout [{holdout_start}, {holdout_end}) overlaps the range [{other_start}, {other_end}) used for tuning"
            ),
        }
    }
}

impl std::error::Error for EvalError {}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, EvalError>;

pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> EvalError {
    EvalError::InvalidParameter { name, reason: reason.into() }
}
