//! Generic Data Loader for user-uploaded CSV (and optionally Parquet) files.
//!
//! Supports:
//! - Configurable column mapping (timestamp, price, quantity, side, OHLCV, custom fields)
//! - Auto-detection of delimiter, timestamp format, and header presence
//! - Extra columns passed through as `HashMap<String, f64>` custom fields
//! - Optional Parquet support (behind `parquet` feature flag)
//!
//! # Example
//! ```no_run
//! use dataloader::generic::{GenericLoader, ColumnMapping};
//!
//! let mapping = ColumnMapping::auto_detect();
//! let loader = GenericLoader::new(mapping);
//! let ticks = loader.load_csv("data/my_trades.csv").unwrap();
//! ```

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::tick::{Side, Tick};

// ── Error types ──────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GenericLoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Missing required column '{0}'. Available columns: {1}")]
    MissingColumn(String, String),

    #[error("Failed to parse timestamp '{0}': {1}")]
    TimestampParse(String, String),

    #[error("Failed to parse number in column '{column}', row {row}: {value}")]
    NumberParse {
        column: String,
        row: usize,
        value: String,
    },

    #[error("Empty file: no data rows found")]
    EmptyFile,

    #[error("Auto-detect failed: {0}")]
    AutoDetectFailed(String),

    #[cfg(feature = "parquet")]
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[cfg(feature = "parquet")]
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}

// ── Column mapping ───────────────────────────────────────────────────────────

/// Describes how columns in the user's file map to market data fields.
///
/// Only `timestamp` and `price` are required. Everything else is optional.
/// Columns not listed here are loaded as custom `f64` fields (if parseable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    /// Column name or index for the timestamp (REQUIRED)
    pub timestamp: ColumnRef,
    /// Column name or index for price (REQUIRED)
    pub price: ColumnRef,
    /// Column name or index for quantity/volume per tick
    pub quantity: Option<ColumnRef>,
    /// Column name or index for trade side (buy/sell)
    pub side: Option<ColumnRef>,
    /// OHLCV column mappings (for candle-style data)
    pub open: Option<ColumnRef>,
    pub high: Option<ColumnRef>,
    pub low: Option<ColumnRef>,
    pub close: Option<ColumnRef>,
    pub volume: Option<ColumnRef>,
    /// Timestamp format hint. If `None`, auto-detect is attempted.
    pub timestamp_format: Option<TimestampFormat>,
    /// If true, pass all unmapped numeric columns as custom fields.
    #[serde(default = "default_true")]
    pub include_extra_columns: bool,
    /// CSV delimiter override. If `None`, auto-detect is attempted.
    pub delimiter: Option<u8>,
}

fn default_true() -> bool {
    true
}

/// Reference to a CSV column — either by name (if headers present) or by 0-based index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColumnRef {
    Name(String),
    Index(usize),
}

impl ColumnRef {
    /// Resolve to a column index given the header row.
    fn resolve(&self, headers: &[String]) -> Option<usize> {
        match self {
            ColumnRef::Name(name) => {
                let lower = name.to_lowercase();
                headers.iter().position(|h| h.to_lowercase() == lower)
            }
            ColumnRef::Index(idx) => {
                if *idx < headers.len() {
                    Some(*idx)
                } else {
                    None
                }
            }
        }
    }

    #[allow(dead_code)]
    fn display_name(&self) -> String {
        match self {
            ColumnRef::Name(n) => n.clone(),
            ColumnRef::Index(i) => format!("column_{}", i),
        }
    }
}

/// Supported timestamp formats for auto-detection and explicit configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimestampFormat {
    /// Unix epoch in seconds (f64, supports fractional)
    EpochSeconds,
    /// Unix epoch in milliseconds (i64)
    EpochMilliseconds,
    /// Unix epoch in microseconds (i64)
    EpochMicroseconds,
    /// Unix epoch in nanoseconds (i64)
    EpochNanoseconds,
    /// ISO 8601 / RFC 3339 string (e.g. "2024-01-15T10:30:00Z")
    Iso8601,
    /// Custom strftime format string
    Custom(String),
}

// ── Generic tick output ──────────────────────────────────────────────────────

/// A single tick of market data loaded from a generic user file.
///
/// Contains the core fields (timestamp, price, quantity, side) plus
/// any extra numeric columns the user's file contained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTick {
    /// Timestamp as milliseconds since Unix epoch
    pub timestamp_ms: i64,
    /// Price
    pub price: f64,
    /// Quantity (defaults to 1.0 if not present in data)
    pub quantity: f64,
    /// Trade side
    pub side: Side,
    /// OHLCV fields (populated if the data is candle-style)
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    /// Extra columns from the user's file that were parseable as f64
    #[serde(default)]
    pub custom_fields: HashMap<String, f64>,
}

impl Tick for GenericTick {
    fn timestamp(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp_ms).unwrap_or(DateTime::UNIX_EPOCH)
    }

    #[inline]
    fn timestamp_ms(&self) -> i64 {
        self.timestamp_ms
    }

    #[inline]
    fn price(&self) -> f64 {
        self.price
    }

    #[inline]
    fn quantity(&self) -> f64 {
        self.quantity
    }

    #[inline]
    fn side(&self) -> Side {
        self.side
    }
}

#[allow(deprecated)]
impl From<&GenericTick> for crate::SimulationTick {
    fn from(tick: &GenericTick) -> Self {
        Self {
            timestamp_ms: tick.timestamp_ms,
            price: tick.price,
            quantity: tick.quantity,
            side: if tick.side.is_sell() { 1 } else { 0 },
        }
    }
}

// ── Auto-detect helpers ──────────────────────────────────────────────────────

impl ColumnMapping {
    /// Create a mapping that auto-detects everything from the file headers.
    ///
    /// Looks for common column names: "timestamp", "time", "date", "ts",
    /// "price", "close", "last", "qty", "quantity", "size", "amount",
    /// "side", "direction", "type", "open", "high", "low", "volume", "vol".
    pub fn auto_detect() -> Self {
        Self {
            timestamp: ColumnRef::Name("timestamp".into()),
            price: ColumnRef::Name("price".into()),
            quantity: None,
            side: None,
            open: None,
            high: None,
            low: None,
            close: None,
            volume: None,
            timestamp_format: None,
            include_extra_columns: true,
            delimiter: None,
        }
    }

    /// Resolve all column references against actual headers.
    /// Returns a `ResolvedMapping` with concrete indices.
    fn resolve(
        &self,
        headers: &[String],
    ) -> Result<ResolvedMapping, GenericLoaderError> {
        let available = headers.join(", ");

        // For auto-detect, try common names for timestamp
        let ts_idx = self.timestamp.resolve(headers).or_else(|| {
            let candidates = ["timestamp", "time", "date", "ts", "datetime", "epoch", "unix_timestamp"];
            candidates.iter().find_map(|c| {
                headers.iter().position(|h| h.to_lowercase() == *c)
            })
        });
        let ts_idx = ts_idx.ok_or_else(|| {
            GenericLoaderError::MissingColumn("timestamp".into(), available.clone())
        })?;

        // For auto-detect, try common names for price
        let price_idx = self.price.resolve(headers).or_else(|| {
            let candidates = ["price", "close", "last", "last_price", "mark_price"];
            candidates.iter().find_map(|c| {
                headers.iter().position(|h| h.to_lowercase() == *c)
            })
        });
        let price_idx = price_idx.ok_or_else(|| {
            GenericLoaderError::MissingColumn("price".into(), available.clone())
        })?;

        // Optional columns with auto-detect fallbacks
        let qty_idx = self.quantity.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            let candidates = ["quantity", "qty", "size", "amount", "vol", "volume"];
            candidates.iter().find_map(|c| {
                headers.iter().position(|h| h.to_lowercase() == *c)
            })
        });

        let side_idx = self.side.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            let candidates = ["side", "direction", "type", "taker_side", "aggressor"];
            candidates.iter().find_map(|c| {
                headers.iter().position(|h| h.to_lowercase() == *c)
            })
        });

        let open_idx = self.open.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            headers.iter().position(|h| h.to_lowercase() == "open")
        });
        let high_idx = self.high.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            headers.iter().position(|h| h.to_lowercase() == "high")
        });
        let low_idx = self.low.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            headers.iter().position(|h| h.to_lowercase() == "low")
        });
        let close_idx = self.close.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            headers.iter().position(|h| h.to_lowercase() == "close")
        });
        let volume_idx = self.volume.as_ref().and_then(|r| r.resolve(headers)).or_else(|| {
            let candidates = ["volume", "vol", "base_volume"];
            candidates.iter().find_map(|c| {
                headers.iter().position(|h| h.to_lowercase() == *c)
            })
        });

        // Collect mapped indices to know which columns are "extra"
        let mut mapped = vec![ts_idx, price_idx];
        for idx in [qty_idx, side_idx, open_idx, high_idx, low_idx, close_idx, volume_idx].iter().flatten() {
            mapped.push(*idx);
        }

        let extra_columns: Vec<(usize, String)> = if self.include_extra_columns {
            headers
                .iter()
                .enumerate()
                .filter(|(i, _)| !mapped.contains(i))
                .map(|(i, name)| (i, name.clone()))
                .collect()
        } else {
            Vec::new()
        };

        Ok(ResolvedMapping {
            timestamp_idx: ts_idx,
            price_idx,
            quantity_idx: qty_idx,
            side_idx,
            open_idx,
            high_idx,
            low_idx,
            close_idx,
            volume_idx,
            extra_columns,
        })
    }
}

/// Column mapping resolved to concrete indices.
struct ResolvedMapping {
    timestamp_idx: usize,
    price_idx: usize,
    quantity_idx: Option<usize>,
    side_idx: Option<usize>,
    open_idx: Option<usize>,
    high_idx: Option<usize>,
    low_idx: Option<usize>,
    close_idx: Option<usize>,
    volume_idx: Option<usize>,
    extra_columns: Vec<(usize, String)>,
}

// ── Timestamp parsing ────────────────────────────────────────────────────────

/// Try to detect the timestamp format from a sample value.
fn detect_timestamp_format(sample: &str) -> TimestampFormat {
    let trimmed = sample.trim();

    // Try parsing as integer
    if let Ok(v) = trimmed.parse::<i64>() {
        // Heuristic based on magnitude (order of magnitude for recent timestamps)
        if v > 1_000_000_000_000_000_000 {
            return TimestampFormat::EpochNanoseconds;
        } else if v > 1_000_000_000_000_000 {
            return TimestampFormat::EpochMicroseconds;
        } else if v > 1_000_000_000_000 {
            return TimestampFormat::EpochMilliseconds;
        } else if v > 1_000_000_000 {
            // Could be seconds or milliseconds for very old data;
            // 1e9 = year 2001 in seconds, but only ~12 days in ms.
            // Assume seconds for values in this range.
            return TimestampFormat::EpochSeconds;
        } else {
            return TimestampFormat::EpochSeconds;
        }
    }

    // Try as float (epoch seconds with fractional)
    if let Ok(_v) = trimmed.parse::<f64>() {
        return TimestampFormat::EpochSeconds;
    }

    // Try ISO8601 / RFC3339
    if trimmed.contains('T') || trimmed.contains('-') {
        return TimestampFormat::Iso8601;
    }

    // Default to epoch milliseconds
    TimestampFormat::EpochMilliseconds
}

/// Parse a timestamp string into epoch milliseconds.
fn parse_timestamp(raw: &str, format: &TimestampFormat) -> Result<i64, GenericLoaderError> {
    let trimmed = raw.trim();

    match format {
        TimestampFormat::EpochSeconds => {
            let secs: f64 = trimmed.parse().map_err(|_| {
                GenericLoaderError::TimestampParse(trimmed.into(), "expected epoch seconds".into())
            })?;
            Ok((secs * 1000.0) as i64)
        }
        TimestampFormat::EpochMilliseconds => {
            let ms: i64 = trimmed.parse().map_err(|_| {
                GenericLoaderError::TimestampParse(trimmed.into(), "expected epoch milliseconds".into())
            })?;
            Ok(ms)
        }
        TimestampFormat::EpochMicroseconds => {
            let us: i64 = trimmed.parse().map_err(|_| {
                GenericLoaderError::TimestampParse(trimmed.into(), "expected epoch microseconds".into())
            })?;
            Ok(us / 1_000)
        }
        TimestampFormat::EpochNanoseconds => {
            let ns: i64 = trimmed.parse().map_err(|_| {
                GenericLoaderError::TimestampParse(trimmed.into(), "expected epoch nanoseconds".into())
            })?;
            Ok(ns / 1_000_000)
        }
        TimestampFormat::Iso8601 => {
            // Try RFC3339 first (strict)
            if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
                return Ok(dt.with_timezone(&Utc).timestamp_millis());
            }
            // Try common ISO8601 variants
            let formats = [
                "%Y-%m-%dT%H:%M:%S%.fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%.f%:z",
                "%Y-%m-%dT%H:%M:%S%:z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ];
            for fmt in &formats {
                if let Ok(dt) = NaiveDateTime::parse_from_str(trimmed, fmt) {
                    return Ok(dt.and_utc().timestamp_millis());
                }
            }
            // Last resort: try just date
            if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
                if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                    return Ok(dt.and_utc().timestamp_millis());
                }
            }
            Err(GenericLoaderError::TimestampParse(
                trimmed.into(),
                "no ISO8601 variant matched".into(),
            ))
        }
        TimestampFormat::Custom(fmt) => {
            let dt = NaiveDateTime::parse_from_str(trimmed, fmt).map_err(|e| {
                GenericLoaderError::TimestampParse(trimmed.into(), e.to_string())
            })?;
            Ok(dt.and_utc().timestamp_millis())
        }
    }
}

/// Parse a side string into a Side enum.
fn parse_side(raw: &str) -> Side {
    match raw.trim().to_lowercase().as_str() {
        "sell" | "s" | "ask" | "a" | "short" | "1" => Side::Sell,
        _ => Side::Buy,
    }
}

// ── Delimiter detection ──────────────────────────────────────────────────────

/// Detect the CSV delimiter from the first few lines.
fn detect_delimiter<R: Read>(reader: &mut BufReader<R>) -> u8
where
    R: Seek,
{
    let mut buf = String::new();
    let start = reader.stream_position().unwrap_or(0);

    // Read up to 5 lines
    for _ in 0..5 {
        if reader.read_line(&mut buf).unwrap_or(0) == 0 {
            break;
        }
    }

    // Reset
    let _ = reader.seek(SeekFrom::Start(start));

    // Count occurrences of common delimiters
    let candidates: [(u8, char); 4] = [
        (b',', ','),
        (b'\t', '\t'),
        (b'|', '|'),
        (b';', ';'),
    ];

    let mut best = b',';
    let mut best_count = 0;

    for (byte, ch) in &candidates {
        let count = buf.matches(*ch).count();
        if count > best_count {
            best_count = count;
            best = *byte;
        }
    }

    best
}

/// Detect whether the first row is a header (contains non-numeric values).
fn looks_like_header(row: &[String]) -> bool {
    // If most fields can't be parsed as numbers, it's probably a header
    let non_numeric = row.iter().filter(|s| {
        let t = s.trim();
        t.parse::<f64>().is_err() && t.parse::<i64>().is_err()
    }).count();
    non_numeric > row.len() / 2
}

// ── GenericLoader ────────────────────────────────────────────────────────────

/// Loads market data from user-provided CSV (or Parquet) files.
pub struct GenericLoader {
    mapping: ColumnMapping,
}

impl GenericLoader {
    /// Create a new loader with an explicit column mapping.
    pub fn new(mapping: ColumnMapping) -> Self {
        Self { mapping }
    }

    /// Create a loader with full auto-detect.
    pub fn auto() -> Self {
        Self {
            mapping: ColumnMapping::auto_detect(),
        }
    }

    /// Load from a file path. Dispatches to CSV or Parquet based on extension.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GenericTick>, GenericLoaderError> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("csv").to_lowercase();

        match ext.as_str() {
            #[cfg(feature = "parquet")]
            "parquet" | "pq" => self.load_parquet(path),
            _ => self.load_csv(path),
        }
    }

    /// Load ticks from a CSV file.
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GenericTick>, GenericLoaderError> {
        let file = std::fs::File::open(path.as_ref())?;
        self.load_csv_reader(file)
    }

    /// Load ticks from any `Read + Seek` source (e.g. in-memory bytes).
    pub fn load_csv_reader<R: Read + Seek>(&self, source: R) -> Result<Vec<GenericTick>, GenericLoaderError> {
        let mut buf_reader = BufReader::new(source);

        // Detect or use configured delimiter
        let delimiter = self.mapping.delimiter.unwrap_or_else(|| detect_delimiter(&mut buf_reader));
        let _ = buf_reader.seek(SeekFrom::Start(0));

        // Build CSV reader
        let mut csv_reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .flexible(true)
            .has_headers(false) // We'll handle headers manually
            .from_reader(buf_reader);

        // Read first row to check if it's a header
        let mut records = csv_reader.records();
        let first_record = records
            .next()
            .ok_or(GenericLoaderError::EmptyFile)?
            .map_err(GenericLoaderError::Csv)?;
        let first_row: Vec<String> = first_record.iter().map(|s| s.to_string()).collect();

        let (headers, first_data_row) = if looks_like_header(&first_row) {
            // First row is headers
            let next_record = records
                .next()
                .ok_or(GenericLoaderError::EmptyFile)?
                .map_err(GenericLoaderError::Csv)?;
            let data_row: Vec<String> = next_record.iter().map(|s| s.to_string()).collect();
            (first_row, Some(data_row))
        } else {
            // No headers — generate synthetic ones
            let synth: Vec<String> = (0..first_row.len()).map(|i| format!("column_{}", i)).collect();
            (synth, Some(first_row))
        };

        // Resolve column mapping
        let resolved = self.mapping.resolve(&headers)?;

        // Detect timestamp format from first data value
        let ts_format = self.mapping.timestamp_format.clone().unwrap_or_else(|| {
            if let Some(ref row) = first_data_row {
                if let Some(ts_val) = row.get(resolved.timestamp_idx) {
                    detect_timestamp_format(ts_val)
                } else {
                    TimestampFormat::EpochMilliseconds
                }
            } else {
                TimestampFormat::EpochMilliseconds
            }
        });

        let mut ticks = Vec::new();

        // Process first data row if we have it
        if let Some(row) = first_data_row {
            if let Ok(tick) = self.parse_row(&row, &resolved, &ts_format, 1) {
                ticks.push(tick);
            }
        }

        // Process remaining rows
        let mut row_num = 2usize;
        for result in records {
            let record = result.map_err(GenericLoaderError::Csv)?;
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            match self.parse_row(&row, &resolved, &ts_format, row_num) {
                Ok(tick) => ticks.push(tick),
                Err(_) => {
                    // Skip malformed rows silently (log in production)
                }
            }
            row_num += 1;
        }

        if ticks.is_empty() {
            return Err(GenericLoaderError::EmptyFile);
        }

        // Sort by timestamp
        ticks.sort_by_key(|t| t.timestamp_ms);

        Ok(ticks)
    }

    /// Parse a single CSV row into a GenericTick.
    fn parse_row(
        &self,
        row: &[String],
        mapping: &ResolvedMapping,
        ts_format: &TimestampFormat,
        row_num: usize,
    ) -> Result<GenericTick, GenericLoaderError> {
        let ts_raw = row.get(mapping.timestamp_idx).ok_or_else(|| {
            GenericLoaderError::NumberParse {
                column: "timestamp".into(),
                row: row_num,
                value: String::new(),
            }
        })?;
        let timestamp_ms = parse_timestamp(ts_raw, ts_format)?;

        let price_raw = row.get(mapping.price_idx).ok_or_else(|| {
            GenericLoaderError::NumberParse {
                column: "price".into(),
                row: row_num,
                value: String::new(),
            }
        })?;
        let price: f64 = price_raw.trim().parse().map_err(|_| {
            GenericLoaderError::NumberParse {
                column: "price".into(),
                row: row_num,
                value: price_raw.clone(),
            }
        })?;

        let quantity = mapping
            .quantity_idx
            .and_then(|i| row.get(i))
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(1.0);

        let side = mapping
            .side_idx
            .and_then(|i| row.get(i))
            .map(|v| parse_side(v))
            .unwrap_or(Side::Buy);

        let open = mapping.open_idx.and_then(|i| row.get(i)).and_then(|v| v.trim().parse().ok());
        let high = mapping.high_idx.and_then(|i| row.get(i)).and_then(|v| v.trim().parse().ok());
        let low = mapping.low_idx.and_then(|i| row.get(i)).and_then(|v| v.trim().parse().ok());
        let close = mapping.close_idx.and_then(|i| row.get(i)).and_then(|v| v.trim().parse().ok());
        let volume = mapping.volume_idx.and_then(|i| row.get(i)).and_then(|v| v.trim().parse().ok());

        // Collect extra columns
        let mut custom_fields = HashMap::new();
        for (idx, name) in &mapping.extra_columns {
            if let Some(val) = row.get(*idx) {
                if let Ok(f) = val.trim().parse::<f64>() {
                    custom_fields.insert(name.clone(), f);
                }
            }
        }

        Ok(GenericTick {
            timestamp_ms,
            price,
            quantity,
            side,
            open,
            high,
            low,
            close,
            volume,
            custom_fields,
        })
    }

    /// Load ticks from a Parquet file (requires `parquet` feature).
    #[cfg(feature = "parquet")]
    pub fn load_parquet<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GenericTick>, GenericLoaderError> {
        use arrow::array::{Array, AsArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::DataType;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = std::fs::File::open(path.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut ticks = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;
            let schema = batch.schema();
            let headers: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
            let resolved = self.mapping.resolve(&headers)?;

            let num_rows = batch.num_rows();

            for row_idx in 0..num_rows {
                // Parse timestamp
                let ts_col = batch.column(resolved.timestamp_idx);
                let timestamp_ms = match ts_col.data_type() {
                    DataType::Int64 => {
                        let arr = ts_col.as_any().downcast_ref::<Int64Array>().unwrap();
                        let val = arr.value(row_idx);
                        // Detect epoch format from magnitude
                        if val > 1_700_000_000_000_000 { val / 1_000 }
                        else if val > 1_700_000_000_000 { val }
                        else { val * 1000 }
                    }
                    DataType::Float64 => {
                        let arr = ts_col.as_any().downcast_ref::<Float64Array>().unwrap();
                        (arr.value(row_idx) * 1000.0) as i64
                    }
                    DataType::Utf8 => {
                        let arr = ts_col.as_any().downcast_ref::<StringArray>().unwrap();
                        let raw = arr.value(row_idx);
                        let fmt = self.mapping.timestamp_format.clone()
                            .unwrap_or_else(|| detect_timestamp_format(raw));
                        parse_timestamp(raw, &fmt)?
                    }
                    _ => continue,
                };

                // Parse price
                let price_col = batch.column(resolved.price_idx);
                let price = match price_col.data_type() {
                    DataType::Float64 => {
                        price_col.as_any().downcast_ref::<Float64Array>().unwrap().value(row_idx)
                    }
                    DataType::Int64 => {
                        price_col.as_any().downcast_ref::<Int64Array>().unwrap().value(row_idx) as f64
                    }
                    _ => continue,
                };

                // Parse quantity
                let quantity = resolved.quantity_idx.map(|i| {
                    let col = batch.column(i);
                    match col.data_type() {
                        DataType::Float64 => col.as_any().downcast_ref::<Float64Array>().unwrap().value(row_idx),
                        DataType::Int64 => col.as_any().downcast_ref::<Int64Array>().unwrap().value(row_idx) as f64,
                        _ => 1.0,
                    }
                }).unwrap_or(1.0);

                // Parse side
                let side = resolved.side_idx.map(|i| {
                    let col = batch.column(i);
                    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        parse_side(arr.value(row_idx))
                    } else {
                        Side::Buy
                    }
                }).unwrap_or(Side::Buy);

                let get_f64 = |idx: Option<usize>| -> Option<f64> {
                    idx.and_then(|i| {
                        let col = batch.column(i);
                        col.as_any().downcast_ref::<Float64Array>().map(|a| a.value(row_idx))
                    })
                };

                // Extra columns
                let mut custom_fields = HashMap::new();
                for (idx, name) in &resolved.extra_columns {
                    let col = batch.column(*idx);
                    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        if !arr.is_null(row_idx) {
                            custom_fields.insert(name.clone(), arr.value(row_idx));
                        }
                    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        if !arr.is_null(row_idx) {
                            custom_fields.insert(name.clone(), arr.value(row_idx) as f64);
                        }
                    }
                }

                ticks.push(GenericTick {
                    timestamp_ms,
                    price,
                    quantity,
                    side,
                    open: get_f64(resolved.open_idx),
                    high: get_f64(resolved.high_idx),
                    low: get_f64(resolved.low_idx),
                    close: get_f64(resolved.close_idx),
                    volume: get_f64(resolved.volume_idx),
                    custom_fields,
                });
            }
        }

        if ticks.is_empty() {
            return Err(GenericLoaderError::EmptyFile);
        }

        ticks.sort_by_key(|t| t.timestamp_ms);
        Ok(ticks)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_cursor(data: &str) -> Cursor<Vec<u8>> {
        Cursor::new(data.as_bytes().to_vec())
    }

    #[test]
    fn test_auto_detect_csv_with_headers() {
        let csv_data = "timestamp,price,quantity,side\n\
                        1700000000000,42100.5,0.03,buy\n\
                        1700000001000,42150.0,0.05,sell\n\
                        1700000002000,42120.0,0.02,buy\n";

        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 3);
        assert_eq!(ticks[0].timestamp_ms, 1700000000000);
        assert!((ticks[0].price - 42100.5).abs() < 0.01);
        assert!((ticks[0].quantity - 0.03).abs() < 0.001);
        assert!(ticks[0].side.is_buy());
        assert!(ticks[1].side.is_sell());
    }

    #[test]
    fn test_auto_detect_epoch_seconds() {
        let csv_data = "timestamp,price\n\
                        1700000000.5,42100.0\n\
                        1700000001.0,42150.0\n";

        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].timestamp_ms, 1700000000500);
    }

    #[test]
    fn test_iso8601_timestamps() {
        let csv_data = "time,price,qty\n\
                        2024-01-15T10:30:00Z,42100.0,0.1\n\
                        2024-01-15T10:30:01Z,42150.0,0.2\n";

        let mut mapping = ColumnMapping::auto_detect();
        mapping.timestamp = ColumnRef::Name("time".into());
        mapping.quantity = Some(ColumnRef::Name("qty".into()));

        let loader = GenericLoader::new(mapping);
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
        assert!((ticks[0].price - 42100.0).abs() < 0.01);
        assert!((ticks[0].quantity - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_ohlcv_data() {
        let csv_data = "timestamp,open,high,low,close,volume\n\
                        1700000000000,42000.0,42200.0,41900.0,42100.0,150.5\n\
                        1700000060000,42100.0,42300.0,42050.0,42250.0,200.0\n";

        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
        // "close" auto-detected as price
        assert!((ticks[0].price - 42100.0).abs() < 0.01);
        assert_eq!(ticks[0].open, Some(42000.0));
        assert_eq!(ticks[0].high, Some(42200.0));
        assert_eq!(ticks[0].low, Some(41900.0));
        assert_eq!(ticks[0].close, Some(42100.0));
        assert_eq!(ticks[0].volume, Some(150.5));
    }

    #[test]
    fn test_extra_custom_columns() {
        let csv_data = "timestamp,price,quantity,funding_rate,oi_change\n\
                        1700000000000,42100.0,0.03,0.0001,500.0\n\
                        1700000001000,42150.0,0.05,0.0002,300.0\n";

        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
        assert!((ticks[0].custom_fields["funding_rate"] - 0.0001).abs() < 1e-8);
        assert!((ticks[0].custom_fields["oi_change"] - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_tab_delimited() {
        let csv_data = "timestamp\tprice\tquantity\n\
                        1700000000000\t42100.0\t0.03\n\
                        1700000001000\t42150.0\t0.05\n";

        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
        assert!((ticks[0].price - 42100.0).abs() < 0.01);
    }

    #[test]
    fn test_no_headers_numeric_only() {
        // All numeric data → detected as no-header
        let csv_data = "1700000000000,42100.0,0.03\n\
                        1700000001000,42150.0,0.05\n";

        let mut mapping = ColumnMapping::auto_detect();
        mapping.timestamp = ColumnRef::Index(0);
        mapping.price = ColumnRef::Index(1);
        mapping.quantity = Some(ColumnRef::Index(2));

        let loader = GenericLoader::new(mapping);
        let ticks = loader.load_csv_reader(make_cursor(csv_data)).unwrap();

        assert_eq!(ticks.len(), 2);
    }

    #[test]
    #[allow(deprecated)]
    fn test_simulation_tick_conversion() {
        let tick = GenericTick {
            timestamp_ms: 1700000000000,
            price: 42100.0,
            quantity: 0.03,
            side: Side::Sell,
            open: None,
            high: None,
            low: None,
            close: None,
            volume: None,
            custom_fields: HashMap::new(),
        };

        let sim_tick: crate::SimulationTick = (&tick).into();
        assert_eq!(sim_tick.timestamp_ms, 1700000000000);
        assert!((sim_tick.price - 42100.0).abs() < 0.01);
        assert!(sim_tick.is_sell());
    }

    #[test]
    fn test_timestamp_format_detection() {
        assert_eq!(detect_timestamp_format("1700000000"), TimestampFormat::EpochSeconds);
        assert_eq!(detect_timestamp_format("1700000000000"), TimestampFormat::EpochMilliseconds);
        assert_eq!(detect_timestamp_format("1700000000000000"), TimestampFormat::EpochMicroseconds);
        assert_eq!(detect_timestamp_format("1700000000000000000"), TimestampFormat::EpochNanoseconds);
        assert_eq!(detect_timestamp_format("2024-01-15T10:30:00Z"), TimestampFormat::Iso8601);
        assert_eq!(detect_timestamp_format("1700000000.5"), TimestampFormat::EpochSeconds);
    }

    #[test]
    fn test_empty_file_error() {
        let csv_data = "";
        let loader = GenericLoader::auto();
        assert!(loader.load_csv_reader(make_cursor(csv_data)).is_err());
    }
}
