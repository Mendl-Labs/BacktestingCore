//! Loading and verifying the answer-key fixtures (pre-registration Amendment 11, Engine
//! `program/tests/fixtures/replication_ladder/`).
//!
//! Trust chain, checked before anything is parsed: the pinned SHA-256 of `MANIFEST.json` (recorded in `ANCHOR.json`
//! and in the amendment) -> every file the manifest lists (bytes and SHA-256) -> the pinned SHA-256 of
//! `ladder_candles.csv` (the value SignalEngine's golden also pins). A missing file, a changed byte, an unlisted
//! required file or a wrong pin is an error; nothing is repaired or skipped.
//!
//! This module reads files only through a caller-supplied reader (`&dyn Fn(&str) -> Result<Vec<u8>, String>`), so the
//! Engine can later feed it `include_bytes!` data (design 5.2) instead of a directory. [`Fixtures::from_dir`] is the
//! directory reader.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use reference_rules::{CRYPTO_SYMBOLS, ETF_SYMBOLS};
use weightsim::{sha256_hex, Date, Panel, PriceSource};

use super::json::{self, Json};

/// sha256 of `MANIFEST.json` as recorded in `ANCHOR.json` and Amendment 11.
pub const REAL_MANIFEST_SHA256: &str = "d51444c33fc2630e12315e936f79580e13dca834ca0fdbc40a242ef496bd6d90";
/// sha256 of `ladder_candles.csv` (40,962 rows); the value pinned in SignalEngine's `reference-rules` golden.
pub const REAL_CANDLES_SHA256: &str = "d5cea5f25a4d0d4ecdf4f3c48ba0bd8d50f72fe208a7a6b5ace1399bb644a365";

pub const F_MANIFEST: &str = "MANIFEST.json";
pub const F_CANDLES: &str = "ladder_candles.csv";
pub const F_S1_PERBAR: &str = "key/S1_etf_trend_faber_perbar.csv";
pub const F_S3_PERBAR: &str = "key/S3_crypto_trend_100d_perbar.csv";
pub const F_KEY_METRICS: &str = "key/key_metrics.json";
pub const F_MUTANTS: &str = "key/mutants.json";
pub const F_SHADOW_S1: &str = "shadow_saved/shadow_S1_daily_returns.csv";
pub const F_SHADOW_S3: &str = "shadow_saved/shadow_S3_daily_returns.csv";

/// Files the ladder cannot run without; each must be listed in the manifest.
pub const REQUIRED_FILES: [&str; 7] =
    [F_CANDLES, F_S1_PERBAR, F_S3_PERBAR, F_KEY_METRICS, F_MUTANTS, F_SHADOW_S1, F_SHADOW_S3];

/// The two hashes a fixture set must match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pins<'a> {
    pub manifest_sha256: &'a str,
    pub candles_sha256: &'a str,
}

impl Pins<'static> {
    /// The pins of the committed real fixture set.
    pub const REAL: Pins<'static> = Pins { manifest_sha256: REAL_MANIFEST_SHA256, candles_sha256: REAL_CANDLES_SHA256 };
}

/// Infrastructure failures (bad or missing fixtures, a simulation error). A FAILED CHECK is not an error: it is
/// recorded in the report and turned into an `Err` only by `self_test`.
#[derive(Clone, Debug, PartialEq)]
pub enum LadderError {
    Io(String),
    Pin(String),
    Parse(String),
    Inconsistent(String),
    Sim(String),
}

impl fmt::Display for LadderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LadderError::Io(m) => write!(f, "fixture i/o: {m}"),
            LadderError::Pin(m) => write!(f, "fixture pin: {m}"),
            LadderError::Parse(m) => write!(f, "fixture parse: {m}"),
            LadderError::Inconsistent(m) => write!(f, "fixture inconsistent: {m}"),
            LadderError::Sim(m) => write!(f, "simulation: {m}"),
        }
    }
}

impl std::error::Error for LadderError {}

/// One row of a sleeve's per-bar key. Row dated `t` is the return over `(t-1, t]`.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyBar {
    pub date: Date,
    pub ret_gross: f64,
    pub ret_net: f64,
    pub equity_gross: f64,
    pub equity_net: f64,
    /// Cost paid at the close of `t` in the net run, as a fraction of the pre-cost equity at that rebalance
    /// (`cost = 0.001 x turnover` exactly).
    pub cost: f64,
    /// Traded notional at the close of `t` in the net run, as a fraction of the pre-cost equity at that rebalance.
    pub turnover: f64,
    pub decision: bool,
    /// Standing target in force during bar `t` (set by the decision at the close of the previous bar).
    pub w_target: Vec<f64>,
    /// Start-of-bar weights of the GROSS run (fractions of the equity at the close of the previous bar).
    pub w_held: Vec<f64>,
}

/// The answer key of one sleeve.
#[derive(Clone, Debug, PartialEq)]
pub struct SleeveKey {
    /// `S1` or `S3`.
    pub code: &'static str,
    pub rule_id: &'static str,
    pub symbols: Vec<String>,
    pub bars: Vec<KeyBar>,
    /// The key's trade counter (`signal_flips`), from `key_metrics.json`.
    pub flips: u64,
}

/// Metrics as recorded in `key_metrics.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedMetrics {
    pub obs: f64,
    pub cagr: f64,
    pub vol: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
}

/// What `mutants.json` (Amendment 11 section 2) says about one mutant.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedMutant {
    pub name: String,
    /// Subset of `tier1_bands`, `tier2_identity`, `tier3_weights`.
    pub caught_by: Vec<String>,
    pub escapes_tier1: bool,
    pub common_days: usize,
    pub corr: f64,
    pub d_sharpe: f64,
    pub d_cagr_pp: f64,
    pub mutant_sharpe: f64,
    pub max_abs_return_diff: f64,
    pub max_abs_w_target_diff: Option<f64>,
    pub max_abs_w_held_diff: Option<f64>,
    /// `(agreement, disagreeing cells, cells)`; `None` for a mutant that has no weights.
    pub tier3: Option<(f64, usize, usize)>,
}

/// Everything the ladder needs, verified.
#[derive(Clone, Debug)]
pub struct Fixtures {
    pub manifest_sha256: String,
    pub candles_sha256: String,
    /// `(path, sha256)` of every manifest entry that was verified, sorted by path.
    pub verified_files: Vec<(String, String)>,
    pub etf_panel: Panel,
    pub crypto_panel: Panel,
    pub s1: SleeveKey,
    pub s3: SleeveKey,
    pub recorded_s1: [RecordedMetrics; 2],
    pub recorded_s3: [RecordedMetrics; 2],
    /// The shadow's own saved daily returns (the original key), gross.
    pub shadow_s1: Vec<(Date, f64)>,
    pub shadow_s3: Vec<(Date, f64)>,
    pub expected_mutants: Vec<ExpectedMutant>,
}

type Reader<'a> = &'a dyn Fn(&str) -> Result<Vec<u8>, String>;

fn parse_err(what: &str, e: impl fmt::Display) -> LadderError {
    LadderError::Parse(format!("{what}: {e}"))
}

fn text(bytes: &[u8], what: &str) -> Result<String, LadderError> {
    String::from_utf8(bytes.to_vec()).map_err(|_| LadderError::Parse(format!("{what} is not valid UTF-8")))
}

fn num(j: &Json, path: &[&str], what: &str) -> Result<f64, LadderError> {
    j.path(path)
        .and_then(Json::as_f64)
        .ok_or_else(|| LadderError::Parse(format!("{what}: missing number at {}", path.join("."))))
}

impl Fixtures {
    /// Load from a directory laid out like the Engine's `program/tests/fixtures/replication_ladder/`, with the real pins.
    pub fn from_dir(dir: &Path) -> Result<Fixtures, LadderError> {
        Fixtures::from_dir_with_pins(dir, Pins::REAL)
    }

    /// Load from a directory with explicit pins (tests use synthetic fixture sets).
    pub fn from_dir_with_pins(dir: &Path, pins: Pins<'_>) -> Result<Fixtures, LadderError> {
        let dir = dir.to_path_buf();
        let reader = move |name: &str| -> Result<Vec<u8>, String> {
            std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()))
        };
        Fixtures::load(&reader, pins)
    }

    /// Load from an in-memory table of `(relative path, bytes)`, for instance `include_bytes!` data, verifying the
    /// same trust chain as [`Fixtures::load`] (which it wraps): the bytes are the only input, nothing touches the file
    /// system. A name that occurs twice in the table is an [`LadderError::Io`] error (a lookup must be unambiguous);
    /// a file the manifest does not list is ignored, exactly as with a directory.
    pub fn from_files(files: &[(&str, &[u8])], pins: Pins<'_>) -> Result<Fixtures, LadderError> {
        for (i, (name, _)) in files.iter().enumerate() {
            if files[..i].iter().any(|(n, _)| n == name) {
                return Err(LadderError::Io(format!("{name}: listed twice in the in-memory file table")));
            }
        }
        let reader = |name: &str| -> Result<Vec<u8>, String> {
            files
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, bytes)| bytes.to_vec())
                .ok_or_else(|| format!("{name}: not in the in-memory file table"))
        };
        Fixtures::load(&reader, pins)
    }

    /// Load through `read` (relative path -> bytes), verifying the trust chain first.
    pub fn load(read: Reader<'_>, pins: Pins<'_>) -> Result<Fixtures, LadderError> {
        let get = |name: &str| read(name).map_err(LadderError::Io);

        // 1. manifest pin
        let manifest_bytes = get(F_MANIFEST)?;
        let manifest_sha = sha256_hex(&manifest_bytes);
        if !manifest_sha.eq_ignore_ascii_case(pins.manifest_sha256) {
            return Err(LadderError::Pin(format!(
                "{F_MANIFEST}: sha256 {manifest_sha}, pinned {}",
                pins.manifest_sha256
            )));
        }
        let manifest = json::parse(&text(&manifest_bytes, F_MANIFEST)?).map_err(|e| parse_err(F_MANIFEST, e))?;
        let files = manifest
            .get("files")
            .and_then(Json::as_object)
            .ok_or_else(|| LadderError::Parse(format!("{F_MANIFEST}: no `files` object")))?;

        // 2. every listed file, bytes and sha256
        let mut verified: BTreeMap<String, String> = BTreeMap::new();
        let mut contents: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (name, meta) in files {
            let want = meta
                .get("sha256")
                .and_then(Json::as_str)
                .ok_or_else(|| LadderError::Parse(format!("{F_MANIFEST}: {name} has no sha256")))?;
            let want_len = meta.get("bytes").and_then(Json::as_f64);
            let bytes = get(name)?;
            let got = sha256_hex(&bytes);
            if !got.eq_ignore_ascii_case(want) {
                return Err(LadderError::Pin(format!("{name}: sha256 {got}, manifest says {want}")));
            }
            if let Some(l) = want_len {
                if l != bytes.len() as f64 {
                    return Err(LadderError::Pin(format!("{name}: {} bytes, manifest says {l}", bytes.len())));
                }
            }
            verified.insert(name.clone(), got);
            contents.insert(name.clone(), bytes);
        }
        for req in REQUIRED_FILES {
            if !verified.contains_key(req) {
                return Err(LadderError::Pin(format!("{req} is required but not listed in {F_MANIFEST}")));
            }
        }
        let candles_sha = verified[F_CANDLES].clone();
        if !candles_sha.eq_ignore_ascii_case(pins.candles_sha256) {
            return Err(LadderError::Pin(format!("{F_CANDLES}: sha256 {candles_sha}, pinned {}", pins.candles_sha256)));
        }

        // 3. parse (only verified bytes are parsed)
        let candles = &contents[F_CANDLES];
        let source = PriceSource::Fixture { csv: candles, expected_sha256: pins.candles_sha256 };
        let etf_panel = source.load(&ETF_SYMBOLS).map_err(|e| parse_err("ETF panel", e))?;
        let crypto_panel = source.load(&CRYPTO_SYMBOLS).map_err(|e| parse_err("crypto panel", e))?;

        let metrics_json =
            json::parse(&text(&contents[F_KEY_METRICS], F_KEY_METRICS)?).map_err(|e| parse_err(F_KEY_METRICS, e))?;
        let flips1 = num(&metrics_json, &["S1", "flips"], F_KEY_METRICS)? as u64;
        let flips3 = num(&metrics_json, &["S3", "flips"], F_KEY_METRICS)? as u64;
        let recorded = |sleeve: &str, basis: &str| -> Result<RecordedMetrics, LadderError> {
            Ok(RecordedMetrics {
                obs: num(&metrics_json, &[sleeve, basis, "obs"], F_KEY_METRICS)?,
                cagr: num(&metrics_json, &[sleeve, basis, "cagr"], F_KEY_METRICS)?,
                vol: num(&metrics_json, &[sleeve, basis, "vol"], F_KEY_METRICS)?,
                sharpe: num(&metrics_json, &[sleeve, basis, "sharpe"], F_KEY_METRICS)?,
                max_drawdown: num(&metrics_json, &[sleeve, basis, "max_drawdown"], F_KEY_METRICS)?,
            })
        };

        let s1 =
            parse_sleeve("S1", "etf_trend_faber", &ETF_SYMBOLS, &text(&contents[F_S1_PERBAR], F_S1_PERBAR)?, flips1)?;
        let s3 = parse_sleeve(
            "S3",
            "crypto_trend_100d",
            &CRYPTO_SYMBOLS,
            &text(&contents[F_S3_PERBAR], F_S3_PERBAR)?,
            flips3,
        )?;

        let mutants_json = json::parse(&text(&contents[F_MUTANTS], F_MUTANTS)?).map_err(|e| parse_err(F_MUTANTS, e))?;
        let expected_mutants = parse_expected_mutants(&mutants_json)?;

        Ok(Fixtures {
            manifest_sha256: manifest_sha,
            candles_sha256: candles_sha,
            verified_files: verified.into_iter().collect(),
            etf_panel,
            crypto_panel,
            s1,
            s3,
            recorded_s1: [recorded("S1", "gross")?, recorded("S1", "net")?],
            recorded_s3: [recorded("S3", "gross")?, recorded("S3", "net")?],
            shadow_s1: parse_shadow(&text(&contents[F_SHADOW_S1], F_SHADOW_S1)?, F_SHADOW_S1)?,
            shadow_s3: parse_shadow(&text(&contents[F_SHADOW_S3], F_SHADOW_S3)?, F_SHADOW_S3)?,
            expected_mutants,
        })
    }
}

fn parse_shadow(csv: &str, what: &str) -> Result<Vec<(Date, f64)>, LadderError> {
    let mut lines = csv.lines();
    let header = lines.next().unwrap_or("");
    if header.trim_end_matches('\r') != ",ret" && header.trim_end_matches('\r') != "date,ret" {
        return Err(LadderError::Parse(format!("{what}: unexpected header `{header}`")));
    }
    let mut out = Vec::new();
    for (n, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let (d, r) =
            line.split_once(',').ok_or_else(|| LadderError::Parse(format!("{what} line {}: no comma", n + 2)))?;
        let date = Date::parse(d).map_err(|e| parse_err(what, e))?;
        let ret: f64 = r.parse().map_err(|_| LadderError::Parse(format!("{what} line {}: bad return `{r}`", n + 2)))?;
        if let Some((prev, _)) = out.last() {
            if date <= *prev {
                return Err(LadderError::Parse(format!("{what}: dates not ascending at {date}")));
            }
        }
        out.push((date, ret));
    }
    if out.is_empty() {
        return Err(LadderError::Parse(format!("{what}: no rows")));
    }
    Ok(out)
}

fn parse_sleeve(
    code: &'static str,
    rule_id: &'static str,
    symbols: &[&str],
    csv: &str,
    flips: u64,
) -> Result<SleeveKey, LadderError> {
    let what = format!("{code} per-bar key");
    let mut lines = csv.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| LadderError::Parse(format!("{what}: empty")))?
        .trim_end_matches('\r')
        .split(',')
        .collect();
    let col = |name: &str| -> Result<usize, LadderError> {
        header.iter().position(|h| *h == name).ok_or_else(|| LadderError::Parse(format!("{what}: no column `{name}`")))
    };
    let (c_date, c_rg, c_rn, c_eg, c_en) =
        (col("date")?, col("ret_gross")?, col("ret_net")?, col("equity_gross")?, col("equity_net")?);
    let (c_cost, c_turn, c_dec, c_excl) = (col("cost")?, col("turnover")?, col("decision")?, col("excluded")?);
    let c_wt: Vec<usize> = symbols.iter().map(|s| col(&format!("w_target_{s}"))).collect::<Result<_, _>>()?;
    let c_wh: Vec<usize> = symbols.iter().map(|s| col(&format!("w_held_{s}"))).collect::<Result<_, _>>()?;

    let mut bars: Vec<KeyBar> = Vec::new();
    for (n, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != header.len() {
            return Err(LadderError::Parse(format!(
                "{what} line {}: {} fields, header has {}",
                n + 2,
                f.len(),
                header.len()
            )));
        }
        let x = |i: usize| -> Result<f64, LadderError> {
            f[i].parse::<f64>().map_err(|_| LadderError::Parse(format!("{what} line {}: bad number `{}`", n + 2, f[i])))
        };
        if f[c_excl] != "0" {
            return Err(LadderError::Inconsistent(format!(
                "{what} line {}: excluded bars are not expected for {code}",
                n + 2
            )));
        }
        let date = Date::parse(f[c_date]).map_err(|e| parse_err(&what, e))?;
        if let Some(prev) = bars.last() {
            if date <= prev.date {
                return Err(LadderError::Parse(format!("{what}: dates not ascending at {date}")));
            }
        }
        bars.push(KeyBar {
            date,
            ret_gross: x(c_rg)?,
            ret_net: x(c_rn)?,
            equity_gross: x(c_eg)?,
            equity_net: x(c_en)?,
            cost: x(c_cost)?,
            turnover: x(c_turn)?,
            decision: f[c_dec] == "1",
            w_target: c_wt.iter().map(|&i| x(i)).collect::<Result<_, _>>()?,
            w_held: c_wh.iter().map(|&i| x(i)).collect::<Result<_, _>>()?,
        });
    }
    if bars.len() < 2 {
        return Err(LadderError::Parse(format!("{what}: fewer than two bars")));
    }
    Ok(SleeveKey { code, rule_id, symbols: symbols.iter().map(|s| (*s).to_string()).collect(), bars, flips })
}

fn parse_expected_mutants(j: &Json) -> Result<Vec<ExpectedMutant>, LadderError> {
    let m = j
        .get("mutants")
        .and_then(Json::as_object)
        .ok_or_else(|| LadderError::Parse(format!("{F_MUTANTS}: no `mutants` object")))?;
    let mut out = Vec::new();
    for (name, v) in m {
        let n = |path: &[&str]| num(v, path, &format!("{F_MUTANTS} {name}"));
        let opt = |path: &[&str]| v.path(path).and_then(Json::as_f64);
        let caught_by: Vec<String> = v
            .get("caught_by")
            .and_then(Json::as_array)
            .ok_or_else(|| LadderError::Parse(format!("{F_MUTANTS} {name}: no caught_by")))?
            .iter()
            .filter_map(|c| c.as_str().map(str::to_string))
            .collect();
        let tier3 = match v.get("tier3") {
            Some(t) if !t.is_null() => Some((
                num(t, &["agreement"], name)?,
                num(t, &["disagreeing_cells"], name)? as usize,
                num(t, &["cells"], name)? as usize,
            )),
            _ => None,
        };
        out.push(ExpectedMutant {
            name: name.clone(),
            caught_by,
            escapes_tier1: v
                .get("escapes_tier1")
                .and_then(Json::as_bool)
                .ok_or_else(|| LadderError::Parse(format!("{F_MUTANTS} {name}: no escapes_tier1")))?,
            common_days: n(&["common_days"])? as usize,
            corr: n(&["corr"])?,
            d_sharpe: n(&["d_sharpe"])?,
            d_cagr_pp: n(&["d_cagr_pp"])?,
            mutant_sharpe: n(&["mutant_sharpe"])?,
            max_abs_return_diff: n(&["tier2", "max_abs_return_diff"])?,
            max_abs_w_target_diff: opt(&["tier2", "max_abs_w_target_diff"]),
            max_abs_w_held_diff: opt(&["tier2", "max_abs_w_held_diff"]),
            tier3,
        });
    }
    if out.is_empty() {
        return Err(LadderError::Parse(format!("{F_MUTANTS}: no mutants")));
    }
    Ok(out)
}
