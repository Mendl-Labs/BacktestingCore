//! Dataloader Module Logging Integration
//!
//! UltraLogger integration for the dataloader module with Elasticsearch support

use ultra_logger::{UltraLogger, LoggerConfig, TransportConfig, ConnectionConfig};
use std::sync::Arc;
use std::collections::HashMap;
use once_cell::sync::Lazy;

fn create_elasticsearch_config() -> LoggerConfig {
    let use_elasticsearch = std::env::var("USE_ELASTICSEARCH_LOGGING")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(true);

    if use_elasticsearch {
        let endpoint = std::env::var("ELASTICSEARCH_ENDPOINT")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_ENDPOINT"))
            .unwrap_or_default();
        let username = std::env::var("ELASTICSEARCH_USERNAME")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_USERNAME"))
            .unwrap_or_else(|_| "elastic".to_string());
        let password = std::env::var("ELASTICSEARCH_PASSWORD")
            .or_else(|_| std::env::var("ELASTIC_CLOUD_PASSWORD"))
            .unwrap_or_default();

        if endpoint.is_empty() {
            return LoggerConfig::default();
        }

        let mut options = HashMap::new();
        options.insert("index".to_string(), "backtestingengine-dataloader-logs".to_string());

        LoggerConfig {
            level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            transport: TransportConfig {
                transport_type: "elasticsearch".to_string(),
                connection: ConnectionConfig {
                    host: endpoint,
                    port: 443,
                    username: Some(username),
                    password: Some(password),
                    options,
                },
            },
        }
    } else {
        LoggerConfig::default()
    }
}

pub static DATALOADER_LOGGER: Lazy<Arc<UltraLogger>> =
    Lazy::new(|| {
        Arc::new(UltraLogger::with_config(
            "BacktestingEngine-dataloader".to_string(),
            create_elasticsearch_config()
        ))
    });

#[macro_export]
macro_rules! log_info {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = format!($fmt $(, $arg)*);
            handle.spawn(async move {
                let _ = logger.info(msg).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = format!($fmt $(, $arg)*);
            handle.spawn(async move {
                let _ = logger.warn(msg).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_error {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = format!($fmt $(, $arg)*);
            handle.spawn(async move {
                let _ = logger.error(msg).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($logger:expr, $fmt:expr $(, $arg:expr)*) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = format!($fmt $(, $arg)*);
            handle.spawn(async move {
                let _ = logger.debug(msg).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_info_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Info, msg, metadata).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_warn_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Warn, msg, metadata).await;
            });
        }
    }};
}

#[macro_export]
macro_rules! log_error_structured {
    ($logger:expr, $msg:expr, $($key:expr => $value:expr),* $(,)?) => {{
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let logger = $logger.clone();
            let msg = $msg.to_string();
            let mut metadata = std::collections::HashMap::new();
            $( metadata.insert($key.to_string(), $value.to_string()); )*
            handle.spawn(async move {
                let _ = logger.log_with_metadata(ultra_logger::LogLevel::Error, msg, metadata).await;
            });
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env vars are process-global; serialise tests that modify them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_create_elasticsearch_config_no_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("USE_ELASTICSEARCH_LOGGING");
        std::env::remove_var("ELASTICSEARCH_ENDPOINT");
        std::env::remove_var("ELASTIC_CLOUD_ENDPOINT");
        let config = create_elasticsearch_config();
        assert_eq!(config.transport.transport_type, "stdout");
    }

    #[test]
    fn test_create_elasticsearch_config_disabled() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("USE_ELASTICSEARCH_LOGGING", "false");
        let config = create_elasticsearch_config();
        assert_eq!(config.transport.transport_type, "stdout");
        std::env::remove_var("USE_ELASTICSEARCH_LOGGING");
    }

    #[test]
    fn test_create_elasticsearch_config_with_endpoint() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("USE_ELASTICSEARCH_LOGGING", "true");
        std::env::set_var("ELASTICSEARCH_ENDPOINT", "https://es.example.com");
        std::env::set_var("ELASTICSEARCH_USERNAME", "testuser");
        std::env::set_var("ELASTICSEARCH_PASSWORD", "testpass");
        let config = create_elasticsearch_config();
        assert_eq!(config.transport.transport_type, "elasticsearch");
        assert_eq!(config.transport.connection.host, "https://es.example.com");
        assert_eq!(config.transport.connection.options.get("index").unwrap(), "backtestingengine-dataloader-logs");
        std::env::remove_var("USE_ELASTICSEARCH_LOGGING");
        std::env::remove_var("ELASTICSEARCH_ENDPOINT");
        std::env::remove_var("ELASTICSEARCH_USERNAME");
        std::env::remove_var("ELASTICSEARCH_PASSWORD");
    }

    #[test]
    fn test_create_elasticsearch_config_fallback_env_vars() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("USE_ELASTICSEARCH_LOGGING", "true");
        std::env::remove_var("ELASTICSEARCH_ENDPOINT");
        std::env::set_var("ELASTIC_CLOUD_ENDPOINT", "https://cloud.es.io");
        std::env::remove_var("ELASTICSEARCH_USERNAME");
        std::env::set_var("ELASTIC_CLOUD_USERNAME", "clouduser");
        std::env::remove_var("ELASTICSEARCH_PASSWORD");
        std::env::set_var("ELASTIC_CLOUD_PASSWORD", "cloudpass");
        let config = create_elasticsearch_config();
        assert_eq!(config.transport.connection.host, "https://cloud.es.io");
        assert_eq!(config.transport.connection.username, Some("clouduser".to_string()));
        assert_eq!(config.transport.connection.password, Some("cloudpass".to_string()));
        std::env::remove_var("USE_ELASTICSEARCH_LOGGING");
        std::env::remove_var("ELASTIC_CLOUD_ENDPOINT");
        std::env::remove_var("ELASTIC_CLOUD_USERNAME");
        std::env::remove_var("ELASTIC_CLOUD_PASSWORD");
    }

    #[test]
    fn test_dataloader_logger_initialization() {
        let logger: &Arc<UltraLogger> = &DATALOADER_LOGGER;
        assert!(Arc::strong_count(logger) >= 1);
    }
}
