//! Supplementary Data Loader
//!
//! Loads additional time-series data files (CSV / Parquet) and time-aligns them
//! with the primary market ticks using forward-fill interpolation.
//!
//! Use cases:
//! - Funding rate history
//! - Open interest snapshots
//! - On-chain metrics
//! - Sentiment scores
//! - Any custom indicator the user wants to feed into their Python strategy
//!
//! The output is a `BTreeMap<i64, HashMap<String, f64>>` where keys are
//! epoch milliseconds and values are field→value maps. This integrates
//! directly with the Python data interface.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek};
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::generic::{
    ColumnMapping, GenericLoader, GenericLoaderError, GenericTick,
};

// ── Error types ──────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SupplementaryError {
    #[error("Generic loader error: {0}")]
    Loader(#[from] GenericLoaderError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No numeric columns found in supplementary file")]
    NoNumericColumns,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a single supplementary data file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplementaryFileConfig {
    /// Path to the data file
    pub path: String,
    /// Optional column mapping. If None, auto-detect is used.
    /// The "price" column from auto-detect will be ignored;
    /// all non-timestamp columns are loaded as named fields.
    pub mapping: Option<ColumnMapping>,
    /// Optional prefix for column names to avoid collisions
    /// e.g. prefix = "funding_" → "rate" becomes "funding_rate"
    pub prefix: Option<String>,
}

// ── Aligned supplementary data ───────────────────────────────────────────────

/// Time-aligned supplementary data.
///
/// Contains a sparse BTreeMap of timestamp→fields. Use `get_at()` for
/// forward-fill lookup at any timestamp.
#[derive(Debug, Clone)]
pub struct SupplementaryData {
    /// Sorted map: epoch_ms → field_name → value
    data: BTreeMap<i64, HashMap<String, f64>>,
    /// All field names across all timestamps (for Python interface discovery)
    field_names: Vec<String>,
}

impl SupplementaryData {
    /// Create empty supplementary data.
    pub fn empty() -> Self {
        Self {
            data: BTreeMap::new(),
            field_names: Vec::new(),
        }
    }

    /// Check if there's any data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of data points.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get all field names.
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Get the underlying BTreeMap (for serialization / Python interface).
    pub fn as_map(&self) -> &BTreeMap<i64, HashMap<String, f64>> {
        &self.data
    }

    /// Consume and return the BTreeMap.
    pub fn into_map(self) -> BTreeMap<i64, HashMap<String, f64>> {
        self.data
    }

    /// Look up values at a given timestamp using forward-fill.
    ///
    /// Returns the most recent data point at or before `timestamp_ms`.
    /// Returns `None` if `timestamp_ms` is before all data points.
    pub fn get_at(&self, timestamp_ms: i64) -> Option<&HashMap<String, f64>> {
        // BTreeMap::range(..=ts) gives all entries ≤ ts; take the last one
        self.data.range(..=timestamp_ms).next_back().map(|(_, v)| v)
    }

    /// Get a specific field at a given timestamp using forward-fill.
    pub fn get_field_at(&self, timestamp_ms: i64, field: &str) -> Option<f64> {
        self.get_at(timestamp_ms).and_then(|m| m.get(field).copied())
    }

    /// Produce a time-aligned vector of field values matching the given tick timestamps.
    ///
    /// For each tick timestamp, looks up the most recent supplementary value
    /// (forward-fill). Returns `None` for ticks before first supplementary data.
    pub fn align_to_ticks(&self, tick_timestamps_ms: &[i64], field: &str) -> Vec<Option<f64>> {
        tick_timestamps_ms
            .iter()
            .map(|&ts| self.get_field_at(ts, field))
            .collect()
    }

    /// Merge another SupplementaryData into this one.
    /// On timestamp collision, fields from `other` overwrite fields in `self`.
    pub fn merge(&mut self, other: SupplementaryData) {
        for (ts, fields) in other.data {
            self.data
                .entry(ts)
                .or_insert_with(HashMap::new)
                .extend(fields);
        }
        // Update field names
        for name in other.field_names {
            if !self.field_names.contains(&name) {
                self.field_names.push(name);
            }
        }
    }
}

// ── Loader ───────────────────────────────────────────────────────────────────

/// Loads and merges multiple supplementary data files.
pub struct SupplementaryLoader;

impl SupplementaryLoader {
    /// Load a single supplementary file and extract all numeric fields.
    pub fn load_file(config: &SupplementaryFileConfig) -> Result<SupplementaryData, SupplementaryError> {
        let mapping = config.mapping.clone().unwrap_or_else(ColumnMapping::auto_detect);
        let loader = GenericLoader::new(mapping);

        let ticks = loader.load(Path::new(&config.path))?;

        Self::ticks_to_supplementary(&ticks, config.prefix.as_deref())
    }

    /// Load from a Read+Seek source (for testing / in-memory data).
    pub fn load_csv_reader<R: Read + Seek>(
        reader: R,
        prefix: Option<&str>,
    ) -> Result<SupplementaryData, SupplementaryError> {
        let loader = GenericLoader::auto();
        let ticks = loader.load_csv_reader(reader)?;
        Self::ticks_to_supplementary(&ticks, prefix)
    }

    /// Load and merge multiple supplementary files.
    pub fn load_multiple(configs: &[SupplementaryFileConfig]) -> Result<SupplementaryData, SupplementaryError> {
        let mut merged = SupplementaryData::empty();

        for config in configs {
            let data = Self::load_file(config)?;
            merged.merge(data);
        }

        Ok(merged)
    }

    /// Convert loaded GenericTicks into SupplementaryData.
    ///
    /// The "price" field from the generic loader becomes a supplementary field too
    /// (renamed to "value" or kept as-is). All custom_fields are included directly.
    fn ticks_to_supplementary(
        ticks: &[GenericTick],
        prefix: Option<&str>,
    ) -> Result<SupplementaryData, SupplementaryError> {
        let mut data = BTreeMap::new();
        let mut field_names_set = HashMap::<String, ()>::new();

        let pfx = prefix.unwrap_or("");

        for tick in ticks {
            let mut fields = HashMap::new();

            // Include the primary "price" as a field (useful for supplementary data
            // where price IS the metric, e.g. funding_rate as "price" column)
            let price_key = format!("{}value", pfx);
            fields.insert(price_key.clone(), tick.price);
            field_names_set.insert(price_key, ());

            if tick.quantity != 1.0 {
                let qty_key = format!("{}quantity", pfx);
                fields.insert(qty_key.clone(), tick.quantity);
                field_names_set.insert(qty_key, ());
            }

            // OHLCV
            if let Some(v) = tick.open {
                let k = format!("{}open", pfx);
                fields.insert(k.clone(), v);
                field_names_set.insert(k, ());
            }
            if let Some(v) = tick.high {
                let k = format!("{}high", pfx);
                fields.insert(k.clone(), v);
                field_names_set.insert(k, ());
            }
            if let Some(v) = tick.low {
                let k = format!("{}low", pfx);
                fields.insert(k.clone(), v);
                field_names_set.insert(k, ());
            }
            if let Some(v) = tick.close {
                let k = format!("{}close", pfx);
                fields.insert(k.clone(), v);
                field_names_set.insert(k, ());
            }
            if let Some(v) = tick.volume {
                let k = format!("{}volume", pfx);
                fields.insert(k.clone(), v);
                field_names_set.insert(k, ());
            }

            // Custom fields
            for (name, value) in &tick.custom_fields {
                let key = format!("{}{}", pfx, name);
                fields.insert(key.clone(), *value);
                field_names_set.insert(key, ());
            }

            data.insert(tick.timestamp_ms, fields);
        }

        if field_names_set.is_empty() {
            return Err(SupplementaryError::NoNumericColumns);
        }

        let mut field_names: Vec<String> = field_names_set.into_keys().collect();
        field_names.sort();

        Ok(SupplementaryData {
            data,
            field_names,
        })
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
    fn test_load_supplementary_csv() {
        let csv = "timestamp,price,funding_rate,oi\n\
                   1700000000000,42100.0,0.0001,50000.0\n\
                   1700000060000,42150.0,0.0002,51000.0\n\
                   1700000120000,42200.0,0.00015,52000.0\n";

        let data = SupplementaryLoader::load_csv_reader(make_cursor(csv), None).unwrap();

        assert_eq!(data.len(), 3);
        assert!(data.field_names().contains(&"funding_rate".to_string()));
        assert!(data.field_names().contains(&"oi".to_string()));
        assert!(data.field_names().contains(&"value".to_string())); // price → value
    }

    #[test]
    fn test_forward_fill() {
        let csv = "timestamp,price\n\
                   1700000000000,42100.0\n\
                   1700000060000,42200.0\n";

        let data = SupplementaryLoader::load_csv_reader(make_cursor(csv), None).unwrap();

        // Before first point
        assert!(data.get_at(1699999999999).is_none());

        // Exactly at first point
        let at_first = data.get_at(1700000000000).unwrap();
        assert!((at_first["value"] - 42100.0).abs() < 0.01);

        // Between points → forward-fill from first
        let between = data.get_at(1700000030000).unwrap();
        assert!((between["value"] - 42100.0).abs() < 0.01);

        // At second point
        let at_second = data.get_at(1700000060000).unwrap();
        assert!((at_second["value"] - 42200.0).abs() < 0.01);

        // After last point → forward-fill from second
        let after = data.get_at(1700000120000).unwrap();
        assert!((after["value"] - 42200.0).abs() < 0.01);
    }

    #[test]
    fn test_align_to_ticks() {
        let csv = "timestamp,price,funding_rate\n\
                   1700000000000,42100.0,0.0001\n\
                   1700000060000,42200.0,0.0002\n";

        let data = SupplementaryLoader::load_csv_reader(make_cursor(csv), None).unwrap();

        let tick_timestamps = vec![
            1699999900000, // before all data
            1700000000000, // at first
            1700000030000, // between
            1700000060000, // at second
            1700000090000, // after second
        ];

        let aligned = data.align_to_ticks(&tick_timestamps, "funding_rate");
        assert_eq!(aligned, vec![
            None,
            Some(0.0001),
            Some(0.0001),  // forward-fill
            Some(0.0002),
            Some(0.0002),  // forward-fill
        ]);
    }

    #[test]
    fn test_prefix() {
        let csv = "timestamp,price,rate\n\
                   1700000000000,42100.0,0.0001\n";

        let data = SupplementaryLoader::load_csv_reader(make_cursor(csv), Some("funding_")).unwrap();

        assert!(data.field_names().contains(&"funding_value".to_string()));
        assert!(data.field_names().contains(&"funding_rate".to_string()));
    }

    #[test]
    fn test_merge() {
        let csv1 = "timestamp,price,rate\n\
                    1700000000000,42100.0,0.0001\n";
        let csv2 = "timestamp,price,oi\n\
                    1700000000000,42100.0,50000.0\n\
                    1700000060000,42200.0,51000.0\n";

        let mut data1 = SupplementaryLoader::load_csv_reader(make_cursor(csv1), None).unwrap();
        let data2 = SupplementaryLoader::load_csv_reader(make_cursor(csv2), None).unwrap();

        data1.merge(data2);

        assert_eq!(data1.len(), 2); // Two distinct timestamps
        let at_first = data1.get_at(1700000000000).unwrap();
        assert!(at_first.contains_key("rate"));
        assert!(at_first.contains_key("oi"));
    }
}
