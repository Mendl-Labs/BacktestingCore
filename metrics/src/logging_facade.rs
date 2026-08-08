//! Metrics Module Logging Integration
//!
//! UltraLogger integration for the metrics module

use ultra_logger::UltraLogger;
use std::sync::Arc;
use once_cell::sync::Lazy;

/// Metrics-specific logger instance
pub static METRICS_LOGGER: Lazy<Arc<UltraLogger>> = 
    Lazy::new(|| {
        Arc::new(UltraLogger::new("metrics".to_string()))
    });

/// Performance-optimized logging macros for metrics module
/// These use sync versions to avoid async overhead in reporting contexts

#[macro_export]
macro_rules! log_info {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        let logger = $logger.clone();
        let msg = format!($fmt $(, $arg)*);
        logger.info_sync(msg);
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        let logger = $logger.clone();
        let msg = format!($fmt $(, $arg)*);
        logger.warn_sync(msg);
    }};
}

#[macro_export]
macro_rules! log_error {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        let logger = $logger.clone();
        let msg = format!($fmt $(, $arg)*);
        logger.error_sync(msg);
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        let logger = $logger.clone();
        let msg = format!($fmt $(, $arg)*);
        logger.debug_sync(msg);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_metrics_logger_initialization() {
        let logger: &Arc<UltraLogger> = &METRICS_LOGGER;
        // Just verify it initializes without panic
        assert!(Arc::strong_count(logger) >= 1);
    }

    #[test]
    fn test_metrics_logger_is_same_instance() {
        let a = &*METRICS_LOGGER;
        let b = &*METRICS_LOGGER;
        assert!(std::ptr::eq(a.as_ref(), b.as_ref()));
    }

    #[test]
    fn test_log_info_macro() {
        let logger = METRICS_LOGGER.clone();
        log_info!(logger, "test info message");
    }

    #[test]
    fn test_log_warn_macro() {
        let logger = METRICS_LOGGER.clone();
        log_warn!(logger, "test warn message: {}", 42);
    }

    #[test]
    fn test_log_error_macro() {
        let logger = METRICS_LOGGER.clone();
        log_error!(logger, "test error: {} {}", "foo", "bar");
    }

    #[test]
    fn test_log_debug_macro() {
        let logger = METRICS_LOGGER.clone();
        log_debug!(logger, "debug val={}", 3.14);
    }
}
