//! Trading strategy implementations
//!
//! This module contains concrete implementations of trading strategies that implement
//! the Strategy trait. All strategies are user-defined Python classes.

// Python strategy bridge (requires pyo3 feature)
#[cfg(feature = "python")]
pub mod python_strategy;

#[cfg(feature = "python")]
pub use python_strategy::PythonStrategy;
#[cfg(feature = "python")]
pub use python_strategy::inject_sdk;
#[cfg(feature = "python")]
pub use python_strategy::unique_module_name;

/// Registry of all available strategy types for dynamic creation
pub fn get_available_strategies() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut strategies = vec![];
    #[cfg(feature = "python")]
    strategies.push("Custom");
    strategies
}

/// Create a strategy by name with default parameters
pub fn create_strategy(strategy_name: &str) -> Option<Box<dyn crate::Strategy>> {
    match strategy_name {
        #[cfg(feature = "python")]
        "Custom" => Some(Box::new(PythonStrategy::new(python_strategy::PYTHON_STRATEGY_TEMPLATE.to_string()))),
        _ => None,
    }
}

/// Create a Python strategy from user-provided source code.
/// Returns None if the python feature is not enabled.
#[cfg(feature = "python")]
pub fn create_python_strategy(source_code: String) -> Box<dyn crate::Strategy> {
    Box::new(PythonStrategy::new(source_code))
}

/// Validate Python strategy source code without executing it.
/// Returns Ok(()) if valid, or Err(message) if invalid.
#[cfg(feature = "python")]
pub fn validate_python_strategy(source_code: &str) -> Result<(), String> {
    PythonStrategy::validate_source(source_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_strategies() {
        let strategies = get_available_strategies();
        // Only "Custom" when python feature is enabled
        #[cfg(feature = "python")]
        assert!(strategies.contains(&"Custom"));
    }

    #[test]
    fn test_create_strategy() {
        let invalid_strategy = create_strategy("NonExistentStrategy");
        assert!(invalid_strategy.is_none());
    }
}
