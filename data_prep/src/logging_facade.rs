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
        options.insert("index".to_string(), "backtestingengine-dataprep-logs".to_string());

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

pub static DATA_PREP_LOGGER: Lazy<Arc<UltraLogger>> =
    Lazy::new(|| {
        Arc::new(UltraLogger::with_config(
            "BacktestingEngine-dataprep".to_string(),
            create_elasticsearch_config()
        ))
    });

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

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
    fn test_create_elasticsearch_config_enabled_no_endpoint() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("USE_ELASTICSEARCH_LOGGING", "true");
        std::env::remove_var("ELASTICSEARCH_ENDPOINT");
        std::env::remove_var("ELASTIC_CLOUD_ENDPOINT");
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
        assert_eq!(config.transport.connection.username, Some("testuser".to_string()));
        assert_eq!(config.transport.connection.password, Some("testpass".to_string()));
        assert_eq!(config.transport.connection.port, 443);
        assert_eq!(config.transport.connection.options.get("index").unwrap(), "backtestingengine-dataprep-logs");
        std::env::remove_var("USE_ELASTICSEARCH_LOGGING");
        std::env::remove_var("ELASTICSEARCH_ENDPOINT");
        std::env::remove_var("ELASTICSEARCH_USERNAME");
        std::env::remove_var("ELASTICSEARCH_PASSWORD");
    }

    #[test]
    fn test_data_prep_logger_initialization() {
        let logger: &Arc<UltraLogger> = &DATA_PREP_LOGGER;
        assert!(Arc::strong_count(logger) >= 1);
    }
}
