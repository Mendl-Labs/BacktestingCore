//! Database models for backtest report persistence
//!
//! This module provides database models for storing and retrieving
//! generated backtest reports and their metadata.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use diesel::{Insertable, Queryable, Selectable, ExpressionMethods, QueryDsl, OptionalExtension, SelectableHelper};
use std::collections::HashMap;
use bigdecimal::BigDecimal;
use serde_json::Value;
use diesel_async::RunQueryDsl;

// Import database schema
use databaseschema::schema::{backtest_reports, backtest_report_access_log, backtest_results};

/// Database model for backtest reports table
#[derive(Queryable, Insertable, Selectable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = backtest_reports)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct BacktestReportRecord {
    pub id: Uuid,
    pub backtest_result_id: Uuid,
    pub report_id: String,
    pub report_name: String,
    pub strategy_name: String,
    pub symbol: String,
    pub timeframe: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub initial_capital: BigDecimal,
    pub generated_at: DateTime<Utc>,
    pub generated_by: Option<String>,
    pub generation_source: String,
    pub backtest_duration_seconds: Option<BigDecimal>,
    pub data_points: Option<i32>,
    pub include_trades: bool,
    pub include_charts: bool,
    pub export_formats: Vec<Option<String>>,
    pub custom_css: Option<String>,
    pub template_version: Option<String>,
    pub file_paths: Value, // JSONB in database
    pub file_sizes: Option<Value>, // JSONB in database
    pub storage_location: String,
    pub performance_summary: Value, // JSONB in database
    pub risk_summary: Value, // JSONB in database
    pub trade_summary: Value, // JSONB in database
    pub status: String,
    pub error_message: Option<String>,
    pub tags: Option<Vec<Option<String>>>,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub accessed_at: Option<DateTime<Utc>>,
    pub access_count: i32,
}

/// New report record for insertion
#[derive(Insertable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = backtest_reports)]
pub struct NewBacktestReportRecord {
    pub backtest_result_id: Uuid,
    pub report_id: String,
    pub report_name: String,
    pub strategy_name: String,
    pub symbol: String,
    pub timeframe: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub initial_capital: BigDecimal,
    pub generated_at: DateTime<Utc>,
    pub generated_by: Option<String>,
    pub generation_source: String,
    pub backtest_duration_seconds: Option<BigDecimal>,
    pub data_points: Option<i32>,
    pub include_trades: bool,
    pub include_charts: bool,
    pub export_formats: Vec<Option<String>>,
    pub custom_css: Option<String>,
    pub template_version: Option<String>,
    pub file_paths: Value,
    pub file_sizes: Option<Value>,
    pub storage_location: String,
    pub performance_summary: Value,
    pub risk_summary: Value,
    pub trade_summary: Value,
    pub status: String,
    pub error_message: Option<String>,
    pub tags: Option<Vec<Option<String>>>,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Database model for report access log
#[derive(Queryable, Insertable, Selectable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = backtest_report_access_log)]
pub struct BacktestReportAccessLog {
    pub id: Uuid,
    pub report_id: Uuid,
    pub accessed_by: Option<String>,
    pub access_method: String,
    pub format_requested: Option<String>,
    pub user_agent: Option<String>,
    pub ip_address: Option<String>, // Store as string for now
    pub response_time_ms: Option<i32>,
    pub success: bool,
    pub error_message: Option<String>,
    pub accessed_at: DateTime<Utc>,
}

/// New access log entry for insertion
#[derive(Insertable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = backtest_report_access_log)]
pub struct NewBacktestReportAccessLog {
    pub report_id: Uuid,
    pub accessed_by: Option<String>,
    pub access_method: String,
    pub format_requested: Option<String>,
    pub user_agent: Option<String>,
    pub ip_address: Option<String>, // Store as string for now
    pub response_time_ms: Option<i32>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Report summary view model
#[derive(Queryable, Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReportSummary {
    pub id: Uuid,
    pub report_id: String,
    pub report_name: String,
    pub strategy_name: String,
    pub symbol: String,
    pub timeframe: String,
    pub generated_at: DateTime<Utc>,
    pub generated_by: Option<String>,
    pub export_formats: Vec<Option<String>>,
    pub status: String,
    pub access_count: i32,
    pub accessed_at: Option<DateTime<Utc>>,
    pub net_profit: Option<BigDecimal>,
    pub sharpe_ratio: Option<BigDecimal>,
    pub max_drawdown: Option<BigDecimal>,
    pub total_trades: Option<i32>,
    pub backtest_total_return: BigDecimal,
    pub backtest_created_at: DateTime<Utc>,
}

impl BacktestReportSummary {
    /// Build a summary from a full report record and joined backtest result data
    fn from_record(
        record: BacktestReportRecord,
        backtest_total_return: BigDecimal,
        backtest_created_at: DateTime<Utc>,
    ) -> Self {
        use bigdecimal::FromPrimitive;
        let perf = &record.performance_summary;

        let extract_bd = |key: &str| -> Option<BigDecimal> {
            perf.get(key)
                .and_then(|v| v.as_f64())
                .and_then(BigDecimal::from_f64)
        };

        Self {
            id: record.id,
            report_id: record.report_id,
            report_name: record.report_name,
            strategy_name: record.strategy_name,
            symbol: record.symbol,
            timeframe: record.timeframe,
            generated_at: record.generated_at,
            generated_by: record.generated_by,
            export_formats: record.export_formats,
            status: record.status,
            access_count: record.access_count,
            accessed_at: record.accessed_at,
            net_profit: extract_bd("net_profit"),
            sharpe_ratio: extract_bd("sharpe_ratio"),
            max_drawdown: extract_bd("max_drawdown"),
            total_trades: perf.get("total_trades").and_then(|v| v.as_i64()).map(|v| v as i32),
            backtest_total_return,
            backtest_created_at,
        }
    }
}

/// Helper to convert pool errors to diesel errors
fn pool_err(e: impl std::fmt::Display) -> diesel::result::Error {
    diesel::result::Error::DatabaseError(
        diesel::result::DatabaseErrorKind::UnableToSendCommand,
        Box::new(e.to_string()),
    )
}

/// Conversion helpers and utility functions
impl NewBacktestReportRecord {
    /// Create a new report record from BacktestReport and metadata
    pub fn from_backtest_report(
        report: &crate::reporting::BacktestReport,
        backtest_result_id: Uuid,
        file_paths: HashMap<String, String>,
        generated_by: Option<String>,
        generation_source: String,
    ) -> Self {
        let file_paths_json = serde_json::to_value(&file_paths)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        
        let performance_summary = serde_json::json!({
            "net_profit": report.performance.net_profit,
            "sharpe_ratio": report.performance.sharpe_ratio,
            "profit_factor": report.performance.profit_factor,
            "max_drawdown": report.performance.max_drawdown,
            "win_rate": report.performance.win_rate,
            "total_trades": report.performance.number_of_trades,
            // Enhanced metrics - transaction costs
            "total_commission": report.performance.total_commission,
            "total_slippage": report.performance.total_slippage,
            "total_market_impact": report.performance.total_market_impact,
            "transaction_costs_pct": report.performance.transaction_costs_percentage,
        });
        
        let risk_summary = serde_json::json!({
            "volatility": report.performance.volatility,
            "value_at_risk_95": report.analysis.risk_metrics.value_at_risk_95,
            "sortino_ratio": report.analysis.risk_metrics.sortino_ratio,
            "calmar_ratio": report.analysis.risk_metrics.calmar_ratio,
            // Enhanced metrics - data quality
            "data_quality_score": report.analysis.data_quality.overall_score,
            "data_completeness": report.analysis.data_quality.data_completeness_score,
            "gaps_detected": report.analysis.data_quality.gaps_detected,
            "max_gap_minutes": report.analysis.data_quality.max_gap_seconds / 60,
        });
        
        let trade_summary = serde_json::json!({
            "total_trades": report.analysis.trade_distribution.profitable_trades + report.analysis.trade_distribution.losing_trades,
            "profitable_trades": report.analysis.trade_distribution.profitable_trades,
            "losing_trades": report.analysis.trade_distribution.losing_trades,
            "best_trade": report.analysis.trade_distribution.best_trade,
            "worst_trade": report.analysis.trade_distribution.worst_trade,
            // Enhanced metrics - market impact
            "trades_with_impact": report.analysis.market_impact.trades_with_impact,
            "avg_impact_bps": report.analysis.market_impact.avg_impact_bps,
            "liquidity_warnings": report.analysis.market_impact.liquidity_warnings,
        });

        Self {
            backtest_result_id,
            report_id: report.metadata.report_id.clone(),
            report_name: format!("{} - {} Report", report.metadata.strategy_name, report.metadata.symbol),
            strategy_name: report.metadata.strategy_name.clone(),
            symbol: report.metadata.symbol.clone(),
            timeframe: report.metadata.timeframe.clone(),
            start_date: report.metadata.start_date,
            end_date: report.metadata.end_date,
            initial_capital: BigDecimal::from(report.metadata.initial_capital as i64),
            generated_at: report.metadata.generated_at,
            generated_by,
            generation_source,
            backtest_duration_seconds: report.metadata.backtest_duration_seconds
                .map(|d| BigDecimal::from(d as i64)),
            data_points: Some(report.metadata.data_points as i32),
            include_trades: report.trades.is_some(),
            include_charts: true, // Assume charts are included by default
            export_formats: vec![], // Will be filled based on generated files
            custom_css: None,
            template_version: Some("1.0".to_string()),
            file_paths: file_paths_json,
            file_sizes: None, // Can be calculated from files
            storage_location: "local".to_string(),
            performance_summary,
            risk_summary,
            trade_summary,
            status: "generated".to_string(),
            error_message: None,
            tags: None,
            notes: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Repository for database operations on reports
pub struct BacktestReportRepository {
    pool: std::sync::Arc<diesel_async::pooled_connection::deadpool::Pool<diesel_async::AsyncPgConnection>>,
}

impl BacktestReportRepository {
    /// Create a new repository with connection pool
    pub fn new(pool: std::sync::Arc<diesel_async::pooled_connection::deadpool::Pool<diesel_async::AsyncPgConnection>>) -> Self {
        Self { pool }
    }

    /// Save a new report record to the database
    pub async fn save_report(&self, report: NewBacktestReportRecord) -> Result<(), diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        diesel::insert_into(backtest_reports::table)
            .values(&report)
            .execute(&mut conn)
            .await?;

        Ok(())
    }

    /// Find a report by report_id string
    pub async fn find_by_report_id(&self, report_id: &str) -> Result<Option<BacktestReportRecord>, diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        backtest_reports::table
            .filter(backtest_reports::report_id.eq(report_id))
            .select(BacktestReportRecord::as_select())
            .first(&mut conn)
            .await
            .optional()
    }

    /// List reports for a specific backtest result
    pub async fn list_by_backtest_result(&self, backtest_result_id: Uuid) -> Result<Vec<BacktestReportSummary>, diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        let records: Vec<BacktestReportRecord> = backtest_reports::table
            .filter(backtest_reports::backtest_result_id.eq(backtest_result_id))
            .select(BacktestReportRecord::as_select())
            .order(backtest_reports::generated_at.desc())
            .load(&mut conn)
            .await?;

        // Fetch backtest result data for total_return and created_at
        let (total_return, backtest_created): (BigDecimal, DateTime<Utc>) = backtest_results::table
            .filter(backtest_results::id.eq(backtest_result_id))
            .select((backtest_results::total_return, backtest_results::created_at))
            .first(&mut conn)
            .await
            .unwrap_or_else(|_| (BigDecimal::from(0), Utc::now()));

        Ok(records
            .into_iter()
            .map(|r| BacktestReportSummary::from_record(r, total_return.clone(), backtest_created))
            .collect())
    }

    /// Insert a report access log entry
    pub async fn log_access(&self, access_log: NewBacktestReportAccessLog) -> Result<(), diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        diesel::insert_into(backtest_report_access_log::table)
            .values(&access_log)
            .execute(&mut conn)
            .await?;

        Ok(())
    }

    /// Mark report as accessed (increment counter, update timestamp)
    pub async fn mark_accessed(&self, report_id: Uuid) -> Result<(), diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        diesel::update(backtest_reports::table.filter(backtest_reports::id.eq(report_id)))
            .set((
                backtest_reports::access_count.eq(backtest_reports::access_count + 1),
                backtest_reports::accessed_at.eq(Some(Utc::now())),
                backtest_reports::updated_at.eq(Utc::now()),
            ))
            .execute(&mut conn)
            .await?;

        Ok(())
    }

    /// Search reports by various criteria using dynamic query building
    pub async fn search_reports(
        &self,
        strategy_name: Option<&str>,
        symbol: Option<&str>,
        generated_by: Option<&str>,
        from_date: Option<DateTime<Utc>>,
        to_date: Option<DateTime<Utc>>,
        _tags: Option<&[String]>,
    ) -> Result<Vec<BacktestReportSummary>, diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;

        // Build dynamic query with optional filters
        let mut query = backtest_reports::table.into_boxed();

        if let Some(name) = strategy_name {
            query = query.filter(backtest_reports::strategy_name.eq(name));
        }
        if let Some(sym) = symbol {
            query = query.filter(backtest_reports::symbol.eq(sym));
        }
        if let Some(by) = generated_by {
            query = query.filter(backtest_reports::generated_by.eq(by));
        }
        if let Some(from) = from_date {
            query = query.filter(backtest_reports::generated_at.ge(from));
        }
        if let Some(to) = to_date {
            query = query.filter(backtest_reports::generated_at.le(to));
        }
        // Note: tags filtering via array containment deferred

        // Load only IDs via typed boxed query, then load full records with Selectable
        let ids: Vec<Uuid> = query
            .select(backtest_reports::id)
            .order(backtest_reports::generated_at.desc())
            .load(&mut conn)
            .await?;

        if ids.is_empty() {
            return Ok(vec![]);
        }

        let records: Vec<BacktestReportRecord> = backtest_reports::table
            .filter(backtest_reports::id.eq_any(&ids))
            .select(BacktestReportRecord::as_select())
            .order(backtest_reports::generated_at.desc())
            .load(&mut conn)
            .await?;

        if records.is_empty() {
            return Ok(vec![]);
        }

        // Batch-load backtest_results for all unique backtest_result_ids
        let result_ids: Vec<Uuid> = records
            .iter()
            .map(|r| r.backtest_result_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let results: Vec<(Uuid, BigDecimal, DateTime<Utc>)> = backtest_results::table
            .filter(backtest_results::id.eq_any(&result_ids))
            .select((backtest_results::id, backtest_results::total_return, backtest_results::created_at))
            .load(&mut conn)
            .await?;

        let result_map: HashMap<Uuid, (BigDecimal, DateTime<Utc>)> = results
            .into_iter()
            .map(|(id, tr, ca)| (id, (tr, ca)))
            .collect();

        Ok(records
            .into_iter()
            .map(|r| {
                let (total_return, created_at) = result_map
                    .get(&r.backtest_result_id)
                    .cloned()
                    .unwrap_or_else(|| (BigDecimal::from(0), Utc::now()));
                BacktestReportSummary::from_record(r, total_return, created_at)
            })
            .collect())
    }

    /// Delete reports older than the specified number of days
    pub async fn cleanup_old_reports(&self, days_old: i32) -> Result<i32, diesel::result::Error> {
        let mut conn = self.pool.get().await.map_err(pool_err)?;
        let cutoff = Utc::now() - chrono::Duration::days(days_old as i64);

        let deleted = diesel::delete(
            backtest_reports::table.filter(backtest_reports::created_at.lt(cutoff)),
        )
        .execute(&mut conn)
        .await?;

        Ok(deleted as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_new_report_access_log_creation() {
        let log = NewBacktestReportAccessLog {
            report_id: uuid::Uuid::new_v4(),
            accessed_by: Some("test_user".into()),
            access_method: "api".into(),
            format_requested: Some("json".into()),
            user_agent: None,
            ip_address: None,
            response_time_ms: Some(42),
            success: true,
            error_message: None,
        };
        assert_eq!(log.access_method, "api");
        assert_eq!(log.response_time_ms, Some(42));
        assert!(log.success);
    }

    #[test]
    fn test_access_log_json_round_trip() {
        let log = NewBacktestReportAccessLog {
            report_id: uuid::Uuid::new_v4(),
            accessed_by: None,
            access_method: "web".into(),
            format_requested: None,
            user_agent: Some("test-agent".into()),
            ip_address: Some("127.0.0.1".into()),
            response_time_ms: Some(100),
            success: false,
            error_message: Some("timeout".into()),
        };
        let json = serde_json::to_string(&log).expect("serialize");
        let deserialized: NewBacktestReportAccessLog =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.access_method, "web");
        assert_eq!(deserialized.success, false);
        assert_eq!(deserialized.error_message, Some("timeout".into()));
    }
}
