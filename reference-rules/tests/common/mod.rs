#![allow(dead_code)]
//! Shared test helpers: CSV loading, synthetic series, a seeded RNG (no external rand crate).

use std::collections::BTreeMap;
use std::path::PathBuf;

use chrono::{Datelike, Duration, NaiveDate};
use reference_rules::{Panel, PriceSeries};
use sha2::{Digest, Sha256};

pub fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

/// A file that ships WITH this crate (`tests/data`): synthetic data only. This repository is public and the
/// vendor-derived real-history goldens (ladder candles, shadow signals, the FX golden derived from them) are never
/// committed here; see `real_dir` for how the real-data tests find them.
pub fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

pub fn read_data(name: &str) -> String {
    std::fs::read_to_string(data_path(name)).unwrap()
}

pub fn sha256_of(name: &str) -> String {
    hex::encode(Sha256::digest(std::fs::read(data_path(name)).unwrap()))
}

/// The directory holding the REAL (vendor-derived) replication-ladder files, from the environment variable
/// `REFRULES_LADDER_DIR` (the local `replication_ladder` folder: `ladder_candles.csv`,
/// `shadow_S1_monthend_signals.csv`, `shadow_S3_daily_signals.csv`).
pub fn real_dir() -> Option<PathBuf> {
    std::env::var_os("REFRULES_LADDER_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Guard for the env-gated real-data tests: `let Some(_) = real_or_skip("name") else { return };`. Prints
/// `SKIPPED` (so a CI log shows the test did not run) and passes when the variable is unset.
pub fn real_or_skip(test: &str) -> Option<PathBuf> {
    let dir = real_dir();
    if dir.is_none() {
        println!("SKIPPED {test}: REFRULES_LADDER_DIR is not set (real-data golden; vendor data is not in this repository)");
    }
    dir
}

/// A file of the real ladder directory. Only call after `real_or_skip` returned `Some`.
pub fn real_path(name: &str) -> PathBuf {
    real_dir()
        .expect("REFRULES_LADDER_DIR must be set for a real-data test")
        .join(name)
}

pub fn read_real(name: &str) -> String {
    std::fs::read_to_string(real_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

pub fn sha256_of_real(name: &str) -> String {
    hex::encode(Sha256::digest(
        std::fs::read(real_path(name)).unwrap_or_else(|e| panic!("{name}: {e}")),
    ))
}

/// The FX golden derived from the real ladder (`golden_fx_tsmom.csv`, written by `tests/data/gen_fx_golden.py`):
/// `REFRULES_FX_GOLDEN_DIR` if set, else the real ladder directory.
pub fn real_fx_golden_path(name: &str) -> PathBuf {
    match std::env::var_os("REFRULES_FX_GOLDEN_DIR").filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir).join(name),
        None => real_path(name),
    }
}

/// symbol -> (date, close) rows in file order, from the REAL ladder (`REFRULES_LADDER_DIR`).
pub fn load_ladder() -> BTreeMap<String, Vec<(NaiveDate, f64)>> {
    parse_ladder(&read_real("ladder_candles.csv"))
}

/// The same long format from the small SYNTHETIC ladder that ships with this crate
/// (`tests/data/synthetic_ladder_candles.csv`, byte-identical to weightsim's fixture).
pub fn load_synthetic_ladder() -> BTreeMap<String, Vec<(NaiveDate, f64)>> {
    parse_ladder(&read_data("synthetic_ladder_candles.csv"))
}

fn parse_ladder(text: &str) -> BTreeMap<String, Vec<(NaiveDate, f64)>> {
    let mut out: BTreeMap<String, Vec<(NaiveDate, f64)>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            assert_eq!(line.trim(), "symbol,date_utc,close");
            continue;
        }
        let f: Vec<&str> = line.trim().split(',').collect();
        out.entry(f[0].to_string()).or_default().push((
            NaiveDate::parse_from_str(f[1], "%Y-%m-%d").unwrap(),
            f[2].parse().unwrap(),
        ));
    }
    out
}

pub fn panel_of(ladder: &BTreeMap<String, Vec<(NaiveDate, f64)>>, symbols: &[&str]) -> Panel {
    Panel::new(
        symbols
            .iter()
            .map(|s| {
                let rows = &ladder[*s];
                PriceSeries::new(
                    *s,
                    rows.iter().map(|r| r.0).collect(),
                    rows.iter().map(|r| r.1).collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

/// Keep only the dates present in EVERY series (what pandas `dropna` on the pivoted frame does).
pub fn inner_join(panel: &Panel) -> Panel {
    let mut common: Option<Vec<NaiveDate>> = None;
    for s in panel.iter() {
        common = Some(match common {
            None => s.dates().to_vec(),
            Some(c) => c
                .into_iter()
                .filter(|x| s.position_of(*x).is_some())
                .collect(),
        });
    }
    let common = common.unwrap();
    Panel::new(
        panel
            .iter()
            .map(|s| {
                let closes = common
                    .iter()
                    .map(|x| s.closes()[s.position_of(*x).unwrap()])
                    .collect();
                PriceSeries::new(s.symbol(), common.clone(), closes).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

/// The REAL shadow signal file `name` from `REFRULES_LADDER_DIR`: header `date,SYM1,SYM2..`, rows of 0/1.
pub fn load_shadow(name: &str) -> (Vec<String>, Vec<(NaiveDate, Vec<u8>)>) {
    parse_signals(&read_real(name))
}

/// The SYNTHETIC key signal file `name` from `tests/data` (same format as the shadow files).
pub fn load_key(name: &str) -> (Vec<String>, Vec<(NaiveDate, Vec<u8>)>) {
    parse_signals(&read_data(name))
}

fn parse_signals(text: &str) -> (Vec<String>, Vec<(NaiveDate, Vec<u8>)>) {
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .trim()
        .split(',')
        .skip(1)
        .map(str::to_string)
        .collect();
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.trim().split(',').collect();
            (
                NaiveDate::parse_from_str(f[0], "%Y-%m-%d").unwrap(),
                f[1..].iter().map(|x| x.parse().unwrap()).collect(),
            )
        })
        .collect();
    (header, rows)
}

/// SplitMix64: tiny, seedable, deterministic.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Weekday bars from `start` for `n_days` calendar days, with each weekday skipped with probability `holiday_p`
/// (never two consecutive weekdays skipped, so the default ETF gap tolerance holds).
pub fn weekday_dates(
    start: NaiveDate,
    n_days: i64,
    holiday_p: f64,
    rng: &mut Rng,
) -> Vec<NaiveDate> {
    let mut out = Vec::new();
    let mut skipped_last = false;
    for i in 0..n_days {
        let day = start + Duration::days(i);
        if day.weekday().number_from_monday() > 5 {
            continue;
        }
        if !skipped_last && rng.unit() < holiday_p {
            skipped_last = true;
            continue;
        }
        skipped_last = false;
        out.push(day);
    }
    out
}

pub fn calendar_dates(start: NaiveDate, n_days: i64) -> Vec<NaiveDate> {
    (0..n_days).map(|i| start + Duration::days(i)).collect()
}

/// Positive random-walk closes.
pub fn random_walk(n: usize, start: f64, vol: f64, rng: &mut Rng) -> Vec<f64> {
    let mut px = start;
    (0..n)
        .map(|_| {
            px *= 1.0 + vol * (rng.unit() - 0.5) * 2.0;
            px = px.max(0.01);
            // Round to cents so values look like real closes (and ties become possible).
            (px * 100.0).round() / 100.0
        })
        .collect()
}

/// One weekday bar per day from the first weekday of `start_year`/`start_month` through the last weekday of
/// the month `n_months - 1` later; close = `filler` except on each month's last weekday, which takes
/// `month_end_closes[k]`.
pub fn etf_series(
    symbol: &str,
    start_year: i32,
    start_month: u32,
    month_end_closes: &[f64],
    filler: f64,
) -> PriceSeries {
    let mut dates = Vec::new();
    let mut closes = Vec::new();
    let (mut y, mut m) = (start_year, start_month);
    for &month_end_close in month_end_closes {
        let first = d(y, m, 1);
        let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
        let next_first = d(ny, nm, 1);
        let days = (next_first - first).num_days();
        let weekdays: Vec<NaiveDate> = (0..days)
            .map(|i| first + Duration::days(i))
            .filter(|x| x.weekday().number_from_monday() <= 5)
            .collect();
        let last = weekdays.len() - 1;
        for (j, day) in weekdays.into_iter().enumerate() {
            dates.push(day);
            closes.push(if j == last { month_end_close } else { filler });
        }
        y = ny;
        m = nm;
    }
    PriceSeries::new(symbol, dates, closes).unwrap()
}

/// The last weekday of a month.
pub fn last_weekday(y: i32, m: u32) -> NaiveDate {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    let mut day = d(ny, nm, 1) - Duration::days(1);
    while day.weekday().number_from_monday() > 5 {
        day -= Duration::days(1);
    }
    day
}

/// Consecutive calendar-day series ending on `end`.
pub fn crypto_series(symbol: &str, end: NaiveDate, closes: &[f64]) -> PriceSeries {
    let n = closes.len() as i64;
    PriceSeries::new(
        symbol,
        calendar_dates(end - Duration::days(n - 1), n),
        closes.to_vec(),
    )
    .unwrap()
}
