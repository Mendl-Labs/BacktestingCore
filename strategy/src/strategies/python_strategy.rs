//! Python strategy bridge using PyO3
//!
//! This module provides `PythonStrategy`, a Rust struct that implements the `Strategy` trait
//! by delegating to a user-written Python class. The Python class is loaded from source code
//! at runtime via an embedded CPython interpreter.
//!
//! # Architecture
//!
//! ```text
//! Rust Simulation Loop
//!   └─ strategy_manager.process_market_data()
//!       └─ PythonStrategy::generate_signals()
//!           └─ PyO3 GIL → Python strategy.generate_signals(data, context)
//!               └─ Returns list[Signal] → converted to Vec<Signal>
//! ```
//!
//! # Security
//!
//! ⚠️ **This in-process layer is NOT the security boundary.** The import hook
//! and AST scan below are *defence-in-depth speed bumps*, not a sandbox. PyO3
//! runs user code inside this process with full access to the host once the
//! GIL is held; a determined attacker can bypass an in-interpreter import hook
//! (e.g. via `__builtins__`, C extensions, bytecode tricks, or reaching the
//! object graph). Do NOT treat passing these checks as "safe to run untrusted
//! code". The real isolation boundary is enforced *outside* this process:
//! - Kubernetes default-deny egress NetworkPolicy (metadata IP blocked)
//! - `seccompProfile: RuntimeDefault`, non-root, no privilege escalation
//! - Planned: one-shot, network-denied, syscall-restricted sandbox
//!   (gVisor / Firecracker) — see `SECURITY-TODO.md`.
//!
//! User Python code runs in a restricted environment (best-effort, in-process):
//! - Blocked imports: `os`, `subprocess`, `socket`, `ctypes`, `shutil`, `sys` (partial)
//! - Execution timeout per call (configurable, default 30s)
//! - No filesystem or network access (best-effort, enforced at the k8s layer)
//! - Runs on a dedicated thread (not tokio runtime) to avoid GIL contention

use crate::traits::{
    BasketLeg, BasketSpec, BasketWeightMode, DataRequirements, HedgeRatioMode, PairSpec, Strategy,
    StrategyCategory, StrategyContext, StrategyError, StrategyMetrics, StrategyResult, VenueSeries,
};
use crate::signal::{Signal, SignalMetadata, SignalReason, SignalStrength, SignalType, SpreadLeg};
use derivatives::{DerivativeMetadata, FundingRate, InstrumentKind};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use config::ParameterValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use riskmanager::RiskLimits;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use crate::{log_debug, log_warn};
use crate::logging_facade::STRATEGY_LOGGER;
use crate::log_error;

/// Monotonic counter feeding `unique_module_name` -- see that function's doc
/// comment for why every `PyModule::from_code_bound` call needs a name that
/// has never been used before in this process, not the same fixed literal.
static MODULE_NAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A module name that has never been used before in this process.
///
/// Real incident: both call sites below used the fixed literal
/// `"user_strategy"` as the module name passed to `PyModule::from_code_bound`,
/// which maps directly to CPython's `PyImport_ExecCodeModuleEx` --
/// `PyImport_AddModuleObject` documents that it returns the EXISTING module
/// object if one is already registered under that name in `sys.modules`,
/// rather than creating a fresh one. The embedded Python interpreter is
/// process-global and lives for this worker's entire lifetime (many
/// sequential jobs, per this file's own module doc comment), so a SECOND
/// job's compilation reused the FIRST job's `"user_strategy"` module object
/// and its (non-empty) `__dict__` -- exactly the well-known gotcha behind
/// `importlib.reload()`, where names the new source doesn't redefine are
/// left over from the previous run. A job whose class was named something
/// other than `Strategy` (a perfectly valid choice) would then have
/// `module.getattr("Strategy")` silently resolve to a PRIOR, unrelated job's
/// `Strategy` class instead of failing -- and if that stale class happened
/// to lack `pair_spec()`, the error correctly said "Strategy does not define
/// pair_spec()" while being completely wrong about which strategy that was.
/// A unique name per call guarantees `PyImport_AddModuleObject` always
/// creates a genuinely fresh, empty module -- no reliance on `sys.modules`
/// cleanup semantics at all.
pub fn unique_module_name() -> String {
    format!("user_strategy_{}", MODULE_NAME_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Pre-import allowed scientific libraries before the sandbox blocks `os` etc.
/// numpy/pandas internally import `os`, `subprocess`, etc. during init — we must
/// let that happen first, then lock down the sandbox.
const PRE_IMPORT_ALLOWED_LIBS: &str = r#"
import sys
for _lib in ["numpy", "pandas"]:
    try:
        __import__(_lib)
    except ImportError:
        pass  # not installed — skip
"#;

/// Sandbox import hook that blocks dangerous Python modules.
/// Guarded so it only installs once per interpreter (avoids recursive wrapping).
const SANDBOX_IMPORT_HOOK: &str = r#"
import builtins

if not getattr(builtins, '_sandbox_installed', False):
    _original_import = builtins.__import__

    _BLOCKED_MODULES = frozenset({
        'os', 'subprocess', 'socket', 'ctypes', 'shutil',
        'multiprocessing', 'threading', 'signal', 'resource',
        'pty', 'fcntl', 'termios', 'mmap', 'tempfile',
        'webbrowser', 'http', 'urllib', 'ftplib', 'smtplib',
        'xmlrpc', 'asyncio.subprocess',
    })

    def _restricted_import(name, *args, **kwargs):
        if name in _BLOCKED_MODULES or any(name.startswith(b + '.') for b in _BLOCKED_MODULES):
            raise ImportError(f"Module '{name}' is not allowed in sandboxed strategies")
        return _original_import(name, *args, **kwargs)

    builtins.__import__ = _restricted_import
    builtins._sandbox_installed = True
"#;

/// Default memory limit per Python strategy: 4 GB.
/// Applied via `resource.setrlimit(RLIMIT_AS)` on Linux before the sandbox
/// blocks the `resource` module. Silently skipped on Windows/macOS.
/// Note: RLIMIT_AS restricts the ENTIRE process's virtual address space, not
/// just Python. Each OS thread stack reserves ~8 MB of virtual memory.
/// The process needs headroom for tokio threads, market data, and the
/// validation pipeline. 512 MB was too low and caused EAGAIN on
/// pthread_create when spawning new threads.
const DEFAULT_MEMORY_LIMIT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Python SDK source — embedded from strategy/python_api/ at compile time.
/// These are injected into `sys.modules["trading_platform"]` before user code runs,
/// so `from trading_platform import BaseStrategy, Signal` works.
const PYTHON_SDK_TYPES: &str = include_str!("../../python_api/types.py");
const PYTHON_SDK_STRATEGY: &str = include_str!("../../python_api/strategy.py");
const PYTHON_SDK_INDICATORS: &str = include_str!("../../python_api/indicators.py");
const PYTHON_SDK_INIT: &str = include_str!("../../python_api/__init__.py");

/// Code that registers the `trading_platform` package in `sys.modules`.
/// Called once per interpreter init, after the sandbox but before user code.
fn sdk_injection_code() -> String {
    // We build a single Python script that:
    // 1. Creates a synthetic package module
    // 2. Executes each SDK sub-module into the package namespace
    // 3. Registers it in sys.modules
    //
    // The SDK source strings are embedded as raw literals, so we write them
    // into Python variables using PyO3's locals dict rather than string interpolation.
    r#"
import sys, types

# ── Build trading_platform package ────────────────────────────────────
_tp = types.ModuleType("trading_platform")
_tp.__package__ = "trading_platform"
_tp.__path__ = []  # make it a package

# Sub-modules (execute in dependency order)
for _name, _src in [
    ("types", _sdk_types_src),
    ("strategy", _sdk_strategy_src),
    ("indicators", _sdk_indicators_src),
]:
    _sub = types.ModuleType(f"trading_platform.{_name}")
    _sub.__package__ = "trading_platform"
    # The strategy module imports from .types — wire relative imports
    sys.modules[f"trading_platform.{_name}"] = _sub
    exec(compile(_src, f"trading_platform/{_name}.py", "exec"), _sub.__dict__)
    # Copy all public names into the package namespace
    for _k in dir(_sub):
        if not _k.startswith("_"):
            setattr(_tp, _k, getattr(_sub, _k))

# Execute __init__.py last (it does `from .types import ...` etc.)
# Register the package in sys.modules BEFORE executing __init__.py,
# otherwise `from .strategy import ...` triggers a fresh import of
# trading_platform, which re-executes __init__.py → infinite recursion.
sys.modules["trading_platform"] = _tp
exec(compile(_sdk_init_src, "trading_platform/__init__.py", "exec"), _tp.__dict__)

# Clean up temporary names from globals
del _name, _src, _sub, _k
"#.to_string()
}

/// SDK injection state, shared process-wide since `sys.modules` is a single
/// global table shared by every `Python::with_gil` caller regardless of
/// which OS thread invoked it.
const SDK_NOT_STARTED: u8 = 0;
const SDK_IN_PROGRESS: u8 = 1;
const SDK_DONE_OK: u8 = 2;
static SDK_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(SDK_NOT_STARTED);

/// Helper: inject the SDK into a Python interpreter.
/// Must be called after the sandbox is installed but before user code.
/// Public so other crates (`backtest`, etc.) can call it before loading
/// user Python modules that `from trading_platform import ...`.
///
/// This used to unconditionally re-exec the SDK source into `sys.modules`
/// on every call, i.e. once per backtest job via `init_python()`. `sys.modules`
/// is one process-wide table, and the GIL does not protect a multi-step
/// operation like this from being interleaved with another thread's
/// concurrent `inject_sdk` call: CPython yields the GIL mid-bytecode at
/// periodic check intervals, so one job's still-building
/// `trading_platform.types`/`trading_platform.strategy` submodules could be
/// observed (or clobbered by a fresh, empty module of the same name) by
/// another job running concurrently -- e.g. parallel GA workers or
/// portfolio-leg validation. That produced live, intermittent failures:
/// `ImportError: cannot import name 'DataRequirements' from
/// 'trading_platform.types' (unknown location)`,
/// `AttributeError: 'Strategy' object has no attribute 'initialize'`, and
/// `TypeError: Strategy.initialize() takes 1 positional argument but 2 were
/// given` (confirmed against real `backtest_jobs` rows from 2026-08-15).
///
/// Fixed by making injection idempotent and run-once: the first caller
/// performs the actual exec under an atomic CAS; concurrent callers detect
/// `SDK_IN_PROGRESS` and release the GIL (`py.allow_threads`) while they
/// wait, so the initializing thread can still be scheduled and finish --
/// spin-waiting here without releasing the GIL would deadlock the two
/// threads against each other.
///
/// A failed attempt resets the state back to `SDK_NOT_STARTED` rather than
/// latching a permanent failure -- confirmed live 2026-08-26 that an earlier
/// version which cached the failure (`SDK_DONE_ERR`, checked once via
/// `compare_exchange` and never retried) let a single transient first-attempt
/// failure (e.g. a numpy import hiccup during a concurrent cold start)
/// permanently poison every subsequent `validate_strategy`/
/// `submit_full_backtest` call for the rest of that worker process's life,
/// recoverable only by restarting the process. Injection is a one-time setup
/// step that should always eventually succeed; retrying on the next call is
/// the correct tradeoff over caching a possibly-spurious one-off failure
/// forever.
pub fn inject_sdk(py: Python<'_>) -> Result<(), StrategyError> {
    loop {
        match SDK_STATE.compare_exchange(
            SDK_NOT_STARTED,
            SDK_IN_PROGRESS,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                let locals = PyDict::new_bound(py);
                locals.set_item("_sdk_types_src", PYTHON_SDK_TYPES).unwrap();
                locals.set_item("_sdk_strategy_src", PYTHON_SDK_STRATEGY).unwrap();
                locals.set_item("_sdk_indicators_src", PYTHON_SDK_INDICATORS).unwrap();
                locals.set_item("_sdk_init_src", PYTHON_SDK_INIT).unwrap();

                return match py.run_bound(&sdk_injection_code(), None, Some(&locals)) {
                    Ok(_) => {
                        SDK_STATE.store(SDK_DONE_OK, Ordering::Release);
                        Ok(())
                    }
                    Err(e) => {
                        let msg = format!("SDK injection error: {}", e);
                        log_sdk_injection_failure_diagnostics(py, &msg);
                        SDK_STATE.store(SDK_NOT_STARTED, Ordering::Release);
                        Err(StrategyError::ConfigurationError(msg))
                    }
                };
            }
            Err(SDK_DONE_OK) => return Ok(()),
            Err(_) => {
                py.allow_threads(|| std::thread::sleep(std::time::Duration::from_micros(200)));
            }
        }
    }
}

/// Best-effort diagnostic dump on an `inject_sdk` failure -- gathers exactly
/// the state needed to distinguish the competing theories for why this
/// intermittently fails in production despite passing reliably both
/// single-threaded and under real concurrent load in isolated tests
/// (confirmed 2026-08-26, see `inject_sdk`'s own doc comment): is `numpy`
/// missing from `sys.path` entirely (a real environment problem), present
/// in `sys.path` but not yet in `sys.modules` (a transient import-timing
/// issue), or something else entirely (the error text alone doesn't say
/// which). Every lookup here is independently best-effort (`Result`/
/// `Option` swallowed to a placeholder string) so a failure while
/// diagnosing a failure can never mask or replace the real error being
/// reported -- this is purely additive log output.
fn log_sdk_injection_failure_diagnostics(py: Python<'_>, original_error: &str) {
    let sys_path = py
        .import_bound("sys")
        .and_then(|sys| sys.getattr("path"))
        .map(|p| format!("{:?}", p))
        .unwrap_or_else(|e| format!("<failed to read sys.path: {}>", e));
    let numpy_in_sys_modules = py
        .import_bound("sys")
        .and_then(|sys| sys.getattr("modules"))
        .and_then(|modules| modules.call_method1("__contains__", ("numpy",)))
        .and_then(|v| v.extract::<bool>())
        .map(|b| b.to_string())
        .unwrap_or_else(|e| format!("<failed to check sys.modules: {}>", e));
    let numpy_import_attempt = py
        .import_bound("numpy")
        .map(|m| m.getattr("__file__").map(|f| format!("{:?}", f)).unwrap_or_else(|_| "<no __file__>".to_string()))
        .unwrap_or_else(|e| format!("<direct `import numpy` also failed: {}>", e));
    log_error!(
        STRATEGY_LOGGER,
        "inject_sdk failure diagnostics -- thread={:?} pid={} original_error={} sys.path={} numpy_in_sys.modules={} direct_numpy_import={}",
        std::thread::current().id(),
        std::process::id(),
        original_error,
        sys_path,
        numpy_in_sys_modules,
        numpy_import_attempt
    );
}

/// Global kill switch for the optional `compute_features()` diagnostic hook.
/// Default off — this is extra per-backtest Python execution with no effect
/// on trading signals, so it stays opt-in until its cost is measured on real
/// traffic.
fn compute_features_enabled() -> bool {
    std::env::var("ENABLE_COMPUTE_FEATURES")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

/// Default Python strategy template (shown in the editor for new strategies).
pub const PYTHON_STRATEGY_TEMPLATE: &str = r#""""
Custom Trading Strategy

Implement your strategy by overriding compute_signals().
It is called ONCE with the full tick array — use numpy/pandas for speed.

Signal return values:
    1  = BUY   (open / increase long)
   -1  = SELL  (open / increase short)
    0  = HOLD  (no change, default)
    2  = CLOSE (close any open position)

=== Optional: per-trade dynamic position sizing ===
  Sizing defaults to a flat `position_size_pct` (alias `risk_per_trade`) fraction
  of equity, tunable via parameter_space(). To size entries dynamically instead
  (e.g. inverse to volatility), define:

  def compute_position_sizes(self, prices, volumes, timestamps) -> np.ndarray:
      # One fraction of equity (0.0, 1.0] per tick; only consulted for ticks
      # where compute_signals() returned BUY/SELL. An omitted method or an
      # invalid entry for a tick falls back to position_size_pct for that tick.
      ...

=== Optional: Rust-side indicator pre-computation ===
  Define indicators() to have SMA/EMA/RSI/ATR computed in Rust (fast):

  def indicators(self):
      return {
          "fast_sma": ("sma",  {"period": 10}),
          "slow_sma": ("sma",  {"period": 50}),
          "rsi_14":   ("rsi",  {"period": 14}),
          "macd":     ("macd", {"fast": 12, "slow": 26, "signal": 9}),
          "bb":       ("bollinger", {"period": 20, "std_dev": 2.0}),
          "atr_14":   ("atr",  {"period": 14}),
      }

  Access in compute_signals() via self._indicators dict (if supported).
  Supported types: sma, ema, rsi, macd, bollinger/bb, atr, stddev, vwap
"""

import numpy as np
import pandas as pd

from trading_platform import BaseStrategy


class Strategy(BaseStrategy):
    """
    Bollinger Band + RSI + Volume spike strategy.

    Entry:
      BUY  when price < lower BB  AND RSI < oversold  AND volume spike
      SELL when price > upper BB  AND RSI > overbought AND volume spike

    All computation is vectorized via pandas — no per-tick Python overhead.
    """

    def name(self) -> str:
        return "BBRsiStrategy"

    def parameter_space(self):
        return {
            "bb_period":         {"type": "int",   "min": 10,  "max": 50,  "default": 20},
            "bb_std":            {"type": "float", "min": 1.5, "max": 3.0, "default": 2.0},
            "rsi_period":        {"type": "int",   "min": 7,   "max": 21,  "default": 14},
            "rsi_oversold":      {"type": "float", "min": 20,  "max": 40,  "default": 30.0},
            "rsi_overbought":    {"type": "float", "min": 60,  "max": 80,  "default": 70.0},
            "volume_spike_mult": {"type": "float", "min": 1.2, "max": 3.0, "default": 1.5},
            "atr_period":        {"type": "int",   "min": 7,   "max": 21,  "default": 14},
        }

    def update_parameters(self, params):
        self.params = {
            "bb_period": 20, "bb_std": 2.0,
            "rsi_period": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0,
            "volume_spike_mult": 1.5, "atr_period": 14,
        }
        self.params.update(params)

    def _rsi(self, s, period):
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def compute_signals(self, prices, volumes, timestamps):
        p = self.params if hasattr(self, "params") else {}
        bb_period    = int(p.get("bb_period", 20))
        bb_std       = float(p.get("bb_std", 2.0))
        rsi_period   = int(p.get("rsi_period", 14))
        oversold     = float(p.get("rsi_oversold", 30.0))
        overbought   = float(p.get("rsi_overbought", 70.0))
        vol_mult     = float(p.get("volume_spike_mult", 1.5))
        atr_period   = int(p.get("atr_period", 14))

        s   = pd.Series(prices)
        vol = pd.Series(volumes)

        # Bollinger Bands
        sma   = s.rolling(bb_period).mean()
        std   = s.rolling(bb_period).std()
        upper = sma + bb_std * std
        lower = sma - bb_std * std

        # RSI
        rsi = self._rsi(s, rsi_period)

        # Volume spike filter
        avg_vol   = vol.rolling(atr_period).mean()
        vol_spike = vol > avg_vol * vol_mult

        sig = np.zeros(len(prices), dtype=np.int8)
        sig[(s < lower) & (rsi < oversold)   & vol_spike] = self.BUY   #  1
        sig[(s > upper) & (rsi > overbought) & vol_spike] = self.SELL  # -1
        return sig
"#;

/// A trading strategy implemented in Python, bridged to Rust via PyO3.
///
/// Holds the Python source code and lazily initializes a CPython interpreter
/// when `initialize()` is called. All Python calls are serialized through
/// a Mutex to handle the GIL safely.
pub struct PythonStrategy {
    /// Raw Python source code from the user
    source_code: String,
    /// The instantiated Python strategy object (behind a mutex for GIL safety)
    py_strategy: Arc<Mutex<Option<PyObject>>>,
    /// Strategy name (cached after initialization)
    cached_name: String,
    /// Strategy version (cached)
    cached_version: String,
    /// Strategy description (cached)
    cached_description: String,
    /// Strategy-declared max leverage cap from `get_risk_limits().max_leverage`
    /// (cached after initialization). `f64::INFINITY` -- no additional cap --
    /// if the strategy doesn't override `get_risk_limits()`, returns `None`
    /// for this field (the SDK default), or the call fails for any reason.
    /// Deliberately NOT `1.0`: nothing distinguishes "never overrode this" from
    /// "explicitly declared 1.0", so a numeric default would silently clamp
    /// every strategy that predates this feature down to 1x leverage.
    cached_max_leverage: f64,
    /// Whether the strategy has been initialized
    initialized: bool,
    /// Current parameters
    parameters: HashMap<String, ParameterValue>,
    /// Execution timeout in seconds
    timeout_secs: u64,
    /// Maximum memory (bytes) for Python execution (RLIMIT_AS on Linux)
    memory_limit_bytes: u64,
    /// Cached SDK MarketData class (avoids re-importing every tick)
    cached_md_class: Option<PyObject>,
    /// Cached SDK StrategyContext class (avoids re-importing every tick)
    cached_sc_class: Option<PyObject>,
    /// Cached numpy.frombuffer function for fast array transfer
    cached_np_frombuffer: Option<PyObject>,
    /// Cached numpy float64 dtype object
    cached_np_float64: Option<PyObject>,
    /// Cached numpy int64 dtype object (for timestamp arrays)
    cached_np_int64: Option<PyObject>,
    /// Whether the user's `compute_signals` accepts the optional OHLC arrays
    /// (opens/highs/lows). Detected once at initialization via signature
    /// inspection. When false the engine skips building/passing those arrays.
    compute_signals_accepts_ohlc: bool,
    /// Cross-exchange strategy support (2026-07-22): whether the user's
    /// Python class defines `compute_signals_multi_venue`. Detected once at
    /// initialization (a simple `hasattr` check — no signature introspection
    /// needed since this method has one fixed shape: `(venues: dict) -> list`).
    /// When false, the engine never builds the (more expensive, per-venue)
    /// data bundle at all and single-venue strategies are entirely unaffected.
    accepts_multi_venue: bool,
    /// Per-trade dynamic position sizing (2026-07-25): whether the user's
    /// Python class defines `compute_position_sizes`. Detected once at
    /// initialization via a simple `hasattr` check, mirroring
    /// `accepts_multi_venue` exactly. When false the engine never makes the
    /// extra Python call and every strategy without this method behaves
    /// exactly as before (falls back to the flat `position_size_pct`).
    accepts_position_sizes: bool,
    /// Whether the user's Python class defines `generate_signals`. Detected
    /// once at initialization via a simple `hasattr` check, mirroring
    /// `accepts_position_sizes` exactly. When true, the executor dispatches
    /// per-tick to this context-carrying method instead of the vectorized
    /// `compute_signals` bulk path, so strategies that need `context.iv_surface`
    /// (or `portfolio_greeks`, `orderbook_snapshot`, `custom_data`) can read it.
    defines_generate_signals: bool,
    /// Pairs-trading support: the user's `pair_spec()` return value, parsed
    /// once at initialization if the method exists. `None` for every
    /// ordinary (non-pairs) strategy -- mirrors `accepts_multi_venue`'s
    /// default-inert pattern, but unlike that simple presence check, this
    /// method's return value itself must be read (symbols/exchanges/hedge
    /// ratio), so it's called once here rather than deferred.
    pair_spec: Option<PairSpec>,
    /// Why `pair_spec` above is `None`, when the method exists but its
    /// return value didn't parse (missing/wrong-shaped keys, wrong type,
    /// or the call itself raised) -- distinct from the method simply not
    /// being defined at all. See the parsing site's doc comment for why
    /// this distinction matters (a real incident it would have caught).
    pair_spec_error: Option<String>,
    /// Basket statistical-arbitrage support: the user's `basket_spec()`
    /// return value, parsed once at initialization if the method exists.
    /// Mirrors `pair_spec`'s exact pattern -- see `BasketSpec`'s doc.
    basket_spec: Option<BasketSpec>,
    /// Mirrors `pair_spec_error`'s exact purpose, for `basket_spec`.
    basket_spec_error: Option<String>,
    /// Persistent buffers for context_to_py() — avoids 3 Vec allocations per tick.
    prices_buf: Vec<f64>,
    volumes_buf: Vec<f64>,
    timestamps_buf: Vec<i64>,
    /// Whether the persistent watchdog thread has been started in Python
    watchdog_initialized: bool,
}

impl std::fmt::Debug for PythonStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonStrategy")
            .field("name", &self.cached_name)
            .field("version", &self.cached_version)
            .field("initialized", &self.initialized)
            .field("source_len", &self.source_code.len())
            .finish()
    }
}

impl PythonStrategy {
    /// Create a new PythonStrategy from source code.
    ///
    /// The strategy is NOT initialized until `initialize()` is called.
    pub fn new(source_code: String) -> Self {
        Self {
            source_code,
            py_strategy: Arc::new(Mutex::new(None)),
            cached_name: "PythonStrategy".to_string(),
            cached_version: "1.0.0".to_string(),
            cached_description: "Custom Python strategy".to_string(),
            cached_max_leverage: f64::INFINITY,
            initialized: false,
            parameters: HashMap::new(),
            timeout_secs: 30,
            memory_limit_bytes: DEFAULT_MEMORY_LIMIT_BYTES,
            cached_md_class: None,
            cached_sc_class: None,
            cached_np_frombuffer: None,
            cached_np_float64: None,
            cached_np_int64: None,
            compute_signals_accepts_ohlc: false,
            accepts_multi_venue: false,
            accepts_position_sizes: false,
            defines_generate_signals: false,
            pair_spec: None,
            pair_spec_error: None,
            basket_spec: None,
            basket_spec_error: None,
            prices_buf: Vec::new(),
            volumes_buf: Vec::new(),
            timestamps_buf: Vec::new(),
            watchdog_initialized: false,
        }
    }

    /// Create with a custom timeout (in seconds).
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Create with a custom memory limit (in bytes).
    pub fn with_memory_limit(mut self, memory_limit_bytes: u64) -> Self {
        self.memory_limit_bytes = memory_limit_bytes;
        self
    }

    /// Static AST security scan — rejects dangerous Python patterns before execution.
    ///
    /// Walks the AST looking for:
    /// - Calls to `exec()`, `eval()`, `compile()`, `open()`, `breakpoint()`
    /// - Direct `__import__()` calls (bypasses our import hook)
    /// - Access to `__builtins__`, `__subclasses__`, `__bases__`, `__globals__`,
    ///   `__code__`, `__mro__`
    /// - `getattr`/`setattr`/`delattr` (can bypass sandbox via reflection)
    /// - `globals()`, `locals()`, `vars()` (introspection escape hatches)
    fn ast_security_scan(py: Python<'_>, source_code: &str) -> Result<(), String> {
        let scan_code = r#"
import ast

_BANNED_CALLS = frozenset({
    'exec', 'eval', 'compile', 'open', 'breakpoint',
    '__import__', 'getattr', 'setattr', 'delattr',
    'globals', 'locals', 'vars', 'dir',
    'memoryview', 'type',
})

_BANNED_ATTRS = frozenset({
    '__builtins__', '__subclasses__', '__bases__', '__mro__',
    '__globals__', '__code__', '__closure__', '__func__',
    '__self__', '__dict__', '__class__',
})

def scan(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Reject banned function calls: exec(...), eval(...), etc.
        if isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name and name in _BANNED_CALLS:
                return f"Forbidden call: {name}() is not allowed in strategies (line {node.lineno})"

        # Reject access to dangerous dunder attributes
        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f"Forbidden attribute access: .{node.attr} is not allowed (line {node.lineno})"

        # Reject string-based attribute access via subscript on __dict__
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == '__dict__':
                return f"Forbidden: __dict__ subscript access is not allowed (line {node.lineno})"

    return None  # all clear

result = scan(_source_code)
"#;

        let globals = PyDict::new_bound(py);
        globals.set_item("__builtins__", py.import_bound("builtins").map_err(|e| format!("AST scan setup error: {}", e))?)
            .map_err(|e| format!("AST scan setup error: {}", e))?;
        globals.set_item("_source_code", source_code).map_err(|e| format!("AST scan setup error: {}", e))?;

        // Use globals only (no separate locals) so top-level definitions
        // (_BANNED_CALLS, _BANNED_ATTRS, ast) are visible inside scan().
        py.run_bound(scan_code, Some(&globals), None)
            .map_err(|e| format!("AST scan internal error: {}", e))?;

        let result = globals.get_item("result")
            .map_err(|e| format!("AST scan result error: {}", e))?;

        if let Some(val) = result {
            if !val.is_none() {
                let msg: String = val.extract().map_err(|e| format!("AST scan extract error: {}", e))?;
                return Err(msg);
            }
        }

        Ok(())
    }

    /// Validate Python source code without running a full initialization.
    ///
    /// Checks that the code:
    /// 1. Static AST scan for dangerous patterns (exec, eval, __import__, etc.)
    /// 2. Parses without syntax errors
    /// 3. Defines a `Strategy` class
    /// 4. The class has a `compute_signals` method
    ///
    /// Returns Ok(()) on success, or an error message on failure.
    pub fn validate_source(source_code: &str) -> Result<(), String> {
        Python::with_gil(|py| {
            // ── Phase 0.3: Static AST pre-scan ──────────────────────────────
            // Reject dangerous patterns *before* executing any user code.
            // Done before SDK injection since this only needs the `ast` stdlib module.
            Self::ast_security_scan(py, source_code)?;

            // Inject SDK first (needs uuid → os internally), then sandbox
            inject_sdk(py).map_err(|e| format!("SDK injection error: {}", e))?;

            // Pre-import numpy/pandas so their internal os/subprocess deps are cached
            py.run_bound(PRE_IMPORT_ALLOWED_LIBS, None, None)
                .map_err(|e| format!("Pre-import error: {}", e))?;

            // Install sandbox import restrictions (after SDK deps are loaded)
            py.run_bound(SANDBOX_IMPORT_HOOK, None, None)
                .map_err(|e| format!("Sandbox setup error: {}", e))?;

            // Try to compile the module. Unique name per call -- see
            // unique_module_name's doc comment for why a fixed literal here
            // let a previous job's stale module (and its Strategy class)
            // leak into this compilation via a shared sys.modules entry.
            let module = PyModule::from_code_bound(py, source_code, "strategy.py", &unique_module_name())
                .map_err(|e| format!("Syntax error: {}", e))?;

            // Check for Strategy class
            let strategy_class = module
                .getattr("Strategy")
                .map_err(|_| "Missing 'Strategy' class. Your code must define a class named 'Strategy'.".to_string())?;

            // Instantiate and check for required methods
            let instance = strategy_class
                .call0()
                .map_err(|e| format!("Failed to instantiate Strategy: {}", e))?;

            // Check required methods exist
            if !instance.hasattr("name").unwrap_or(false) {
                return Err("Strategy class must implement 'name()' method".to_string());
            }
            if !instance.hasattr("compute_signals").unwrap_or(false) {
                return Err(
                    "Strategy class must implement 'compute_signals(self, prices, volumes, timestamps)' method (optionally add opens, highs, lows params for full OHLC)"
                        .to_string(),
                );
            }

            Ok(())
        })
    }

    /// Initialize the Python interpreter and load the user's strategy.
    fn init_python(&mut self) -> StrategyResult<PyObject> {
        Python::with_gil(|py| {
            // Inject the trading_platform SDK first (needs uuid → os internally)
            inject_sdk(py)?;

            // Pre-import numpy/pandas so their internal os/subprocess deps are cached
            py.run_bound(PRE_IMPORT_ALLOWED_LIBS, None, None).map_err(|e| {
                StrategyError::ConfigurationError(format!("Pre-import error: {}", e))
            })?;

            // NOTE: RLIMIT_AS memory sandboxing is DISABLED.
            // RLIMIT_AS restricts the entire process's virtual address space,
            // not just Python. This caused EAGAIN on pthread_create because
            // each OS thread stack reserves ~8 MB of virtual memory. The
            // container's cgroup memory limit (8 Gi) provides the actual
            // memory boundary. Python code is sandboxed via the import hook.

            // Install sandbox import restrictions (after SDK deps are loaded)
            py.run_bound(SANDBOX_IMPORT_HOOK, None, None).map_err(|e| {
                StrategyError::ConfigurationError(format!("Sandbox setup error: {}", e))
            })?;

            // Compile and load user module. Unique name per call -- see
            // unique_module_name's doc comment for why a fixed literal here
            // let a previous job's stale module (and its Strategy class)
            // leak into this compilation via a shared sys.modules entry.
            let module = PyModule::from_code_bound(
                py,
                &self.source_code,
                "strategy.py",
                &unique_module_name(),
            )
            .map_err(|e| {
                StrategyError::ConfigurationError(format!("Python compilation error: {}", e))
            })?;

            // Get Strategy class and instantiate
            let strategy_class = module.getattr("Strategy").map_err(|_| {
                StrategyError::ConfigurationError(
                    "No 'Strategy' class found in Python code".to_string(),
                )
            })?;

            let instance = strategy_class.call0().map_err(|e| {
                StrategyError::ConfigurationError(format!(
                    "Failed to instantiate Strategy: {}",
                    e
                ))
            })?;

            // Cache metadata
            if let Ok(name) = instance.call_method0("name") {
                if let Ok(s) = name.extract::<String>() {
                    self.cached_name = s;
                }
            }
            if let Ok(version) = instance.call_method0("version") {
                if let Ok(s) = version.extract::<String>() {
                    self.cached_version = s;
                }
            }
            if let Ok(desc) = instance.call_method0("description") {
                if let Ok(s) = desc.extract::<String>() {
                    self.cached_description = s;
                }
            }
            // Strategy-declared leverage cap (get_risk_limits().max_leverage) --
            // the SDK default is `None` (no cap), which fails `.extract::<f64>()`
            // and correctly leaves `cached_max_leverage` at its safe
            // no-additional-cap default. Any OTHER failure (missing method,
            // wrong shape, exception) leaves that same safe default in place.
            // A genuinely-declared value is floored at 1.0 -- no such thing as
            // sub-1x leverage in this platform's model, same convention as
            // `portfoliomanager::margin`'s own leverage floor.
            if let Ok(risk_limits) = instance.call_method0("get_risk_limits") {
                if let Ok(rl_dict) = risk_limits.call_method0("to_dict") {
                    if let Ok(dict) = rl_dict.downcast::<PyDict>() {
                        if let Some(val) = dict.get_item("max_leverage").ok().flatten() {
                            if let Ok(v) = val.extract::<f64>() {
                                self.cached_max_leverage = v.max(1.0);
                            }
                        }
                    }
                }
            }

            // Initialize state_schema() defaults as instance attributes so that
            // `self.in_position`, `self.position_side`, etc. are available in
            // generate_signals() from the very first tick.
            if let Ok(schema_dict) = instance.call_method0("state_schema") {
                if let Ok(dict) = schema_dict.downcast::<PyDict>() {
                    for (key, spec) in dict.iter() {
                        let key_str: String = match key.extract() {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        // Skip window types — they are managed by the Rust StateStore
                        // and exposed via context["prices"] / context["volumes"].
                        if let Ok(spec_dict) = spec.downcast::<PyDict>() {
                            let type_str: String = spec_dict
                                .get_item("type")
                                .ok()
                                .flatten()
                                .and_then(|v| v.extract().ok())
                                .unwrap_or_default();
                            if type_str.starts_with("window[") {
                                continue;
                            }
                            // Set default value as an instance attribute
                            if let Some(default_val) = spec_dict.get_item("default").ok().flatten() {
                                let _ = instance.setattr(key_str.as_str(), &default_val);
                            } else {
                                // No explicit default — use type-appropriate zero value
                                match type_str.as_str() {
                                    "bool" => { let _ = instance.setattr(key_str.as_str(), false); }
                                    "int" => { let _ = instance.setattr(key_str.as_str(), 0i64); }
                                    "float" => { let _ = instance.setattr(key_str.as_str(), 0.0f64); }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }

            Ok(instance.into())
        })
    }

    /// Convert Rust MarketData to a Python dict for the bridge.
    ///
    /// Provides a flat dict with fields: type, timestamp, price, quantity, side,
    /// plus variant-specific fields (symbol, exchange, OHLCV for candles, custom_fields for generic).
    fn market_data_to_py<'py>(
        py: Python<'py>,
        data: &dataloader::MarketData,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        match data {
            dataloader::MarketData::Trade(trade) => {
                dict.set_item("type", "trade")?;
                let ts_ms = trade.timestamp.timestamp_millis();
                dict.set_item("timestamp", ts_ms)?;
                dict.set_item("timestamp_ms", ts_ms)?;
                dict.set_item("symbol", &*trade.symbol)?;
                dict.set_item("exchange", &*trade.exchange)?;
                dict.set_item("price", trade.price)?;
                dict.set_item("quantity", trade.quantity)?;
                dict.set_item("side", trade.side.as_str())?;
                dict.set_item("value", trade.value)?;
                dict.set_item("trade_id", trade.trade_id)?;
            }
            dataloader::MarketData::Candle(candle) => {
                dict.set_item("type", "candle")?;
                let ts_ms = candle.timestamp.timestamp_millis();
                dict.set_item("timestamp", ts_ms)?;
                dict.set_item("timestamp_ms", ts_ms)?;
                dict.set_item("symbol", &*candle.symbol)?;
                dict.set_item("exchange", &*candle.exchange)?;
                dict.set_item("price", candle.close)?; // Alias: candle "price" = close
                dict.set_item("open", candle.open)?;
                dict.set_item("high", candle.high)?;
                dict.set_item("low", candle.low)?;
                dict.set_item("close", candle.close)?;
                dict.set_item("volume", candle.volume)?;
                dict.set_item("quantity", candle.volume)?; // Alias
                dict.set_item("trade_count", candle.trade_count)?;
                dict.set_item("side", "buy")?; // Neutral for candles
            }
            dataloader::MarketData::Generic(g) => {
                dict.set_item("type", "generic")?;
                dict.set_item("timestamp_ms", g.timestamp_ms)?;
                // Convert to RFC3339 for consistency
                if let Some(dt) = chrono::DateTime::from_timestamp_millis(g.timestamp_ms) {
                    dict.set_item("timestamp", dt.to_rfc3339())?;
                }
                dict.set_item("price", g.price)?;
                dict.set_item("quantity", g.quantity)?;
                dict.set_item("side", if g.side.is_sell() { "sell" } else { "buy" })?;
                // OHLCV fields (optional)
                if let Some(o) = g.open { dict.set_item("open", o)?; }
                if let Some(h) = g.high { dict.set_item("high", h)?; }
                if let Some(l) = g.low { dict.set_item("low", l)?; }
                if let Some(c) = g.close { dict.set_item("close", c)?; }
                if let Some(v) = g.volume { dict.set_item("volume", v)?; }
                // Custom fields as a sub-dict
                if !g.custom_fields.is_empty() {
                    let custom = PyDict::new_bound(py);
                    for (k, v) in &g.custom_fields {
                        custom.set_item(k, *v)?;
                    }
                    dict.set_item("custom_fields", custom)?;
                }
            }
            dataloader::MarketData::PoolSwap(s) => {
                dict.set_item("type", "pool_swap")?;
                let ts_ms = s.timestamp.timestamp_millis();
                dict.set_item("timestamp", ts_ms)?;
                dict.set_item("timestamp_ms", ts_ms)?;
                dict.set_item("exchange", &*s.exchange)?;
                dict.set_item("pool_id", &*s.pool_id)?;
                dict.set_item("token_a", &*s.token_a)?;
                dict.set_item("token_b", &*s.token_b)?;
                dict.set_item("price", s.price())?;
                dict.set_item("quantity", s.amount_in)?;
                dict.set_item("amount_in", s.amount_in)?;
                dict.set_item("amount_out", s.amount_out)?;
                dict.set_item("fee_amount", s.fee_amount)?;
                dict.set_item("sqrt_price", s.sqrt_price)?;
                dict.set_item("total_liquidity", s.total_liquidity)?;
                dict.set_item("side", "buy")?;
            }
            dataloader::MarketData::OptionCandle(c) => {
                dict.set_item("type", "option_candle")?;
                let ts_ms = c.timestamp.timestamp_millis();
                dict.set_item("timestamp", ts_ms)?;
                dict.set_item("timestamp_ms", ts_ms)?;
                dict.set_item("symbol", c.contract_ticker.as_str())?; // Alias, matches Candle's "symbol"
                dict.set_item("contract_ticker", c.contract_ticker.as_str())?;
                dict.set_item("underlying", c.underlying.as_str())?;
                dict.set_item("strike", c.strike)?;
                dict.set_item("expiration", c.expiration.to_string())?;
                dict.set_item("contract_type", match c.contract_type {
                    dataloader::OptionType::Call => "call",
                    dataloader::OptionType::Put => "put",
                })?;
                dict.set_item("price", c.close)?; // Alias: "price" = close, matches Candle
                dict.set_item("open", c.open)?;
                dict.set_item("high", c.high)?;
                dict.set_item("low", c.low)?;
                dict.set_item("close", c.close)?;
                dict.set_item("volume", c.volume)?;
                dict.set_item("quantity", c.volume)?; // Alias
                dict.set_item("side", "buy")?; // Neutral for candles, same as plain Candle
            }
        }

        Ok(dict)
    }

    /// Convert Rust StrategyContext to a rich Python dict.
    ///
    /// Produces a dict matching the Phase 1.4 spec:
    /// - `prices`: list of f64 (from price_history)
    /// - `volumes`: list of f64 (from volume_history)
    /// - `timestamps`: list of i64 (epoch ms from price_history)
    /// - `orderbook`: dict with bids, asks, spread, mid_price, bid_depth, ask_depth, imbalance
    /// - `portfolio`: dict with cash, positions, total_value, realized_pnl
    /// - `custom_data`: dict of supplementary user data (string → float)
    /// - `assets`: dict of per-asset snapshots (symbol → {price, position})
    /// - `market_session_open`: bool
    /// - `tick_number`: int
    /// - `elapsed_seconds`: float
    fn context_to_py<'py>(
        py: Python<'py>,
        context: &StrategyContext,
        np_frombuffer: Option<&PyObject>,
        np_float64: Option<&PyObject>,
        np_int64: Option<&PyObject>,
        prices_buf: &mut Vec<f64>,
        volumes_buf: &mut Vec<f64>,
        timestamps_buf: &mut Vec<i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("timestamp", context.timestamp.timestamp_millis())?;

        // Helper: create a numpy array from a f64 slice if numpy is available,
        // otherwise fall back to a Python list. numpy.frombuffer over a bytes
        // copy is ~100x faster than creating N individual Python float objects.
        let f64_to_py_array = |slice: &[f64], py: Python<'py>| -> PyResult<PyObject> {
            if let (Some(frombuf), Some(dtype)) = (np_frombuffer, np_float64) {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * std::mem::size_of::<f64>(),
                    )
                };
                let py_bytes = pyo3::types::PyBytes::new_bound(py, bytes);
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("dtype", dtype.bind(py))?;
                let arr = frombuf.bind(py).call((py_bytes,), Some(&kwargs))?;
                Ok(arr.call_method0("copy")?.unbind())
            } else {
                let py_list = PyList::new_bound(py, slice.iter());
                Ok(py_list.into_any().unbind())
            }
        };

        // --- Price history as numpy array (or list fallback) ---
        prices_buf.clear();
        prices_buf.extend(context.price_history.iter().map(|p| p.price));
        dict.set_item("prices", f64_to_py_array(prices_buf, py)?)?;

        // --- Volume history as numpy array (or list fallback) ---
        volumes_buf.clear();
        volumes_buf.extend(context.volume_history.iter().map(|v| v.volume));
        dict.set_item("volumes", f64_to_py_array(volumes_buf, py)?)?;

        // --- Timestamps as numpy int64 array (or list fallback) ---
        timestamps_buf.clear();
        timestamps_buf.extend(context.price_history.iter().map(|p| p.timestamp.timestamp_millis()));
        if let (Some(frombuf), Some(dtype)) = (np_frombuffer, np_int64) {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    timestamps_buf.as_ptr() as *const u8,
                    timestamps_buf.len() * std::mem::size_of::<i64>(),
                )
            };
            let py_bytes = pyo3::types::PyBytes::new_bound(py, bytes);
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", dtype.bind(py))?;
            let arr = frombuf.bind(py).call((py_bytes,), Some(&kwargs))?;
            dict.set_item("timestamps", arr.call_method0("copy")?)?;
        } else {
            dict.set_item("timestamps", timestamps_buf.as_slice())?;
        }

        // --- Orderbook snapshot ---
        if let Some(ref ob) = context.orderbook_snapshot {
            let ob_dict = PyDict::new_bound(py);
            // bids/asks as list of [price, size] pairs
            let bids_py: Vec<[f64; 2]> = ob.bids.clone();
            let asks_py: Vec<[f64; 2]> = ob.asks.clone();
            ob_dict.set_item("bids", bids_py)?;
            ob_dict.set_item("asks", asks_py)?;
            ob_dict.set_item("spread", ob.spread)?;
            ob_dict.set_item("mid_price", ob.mid_price)?;
            ob_dict.set_item("bid_depth", ob.bid_depth)?;
            ob_dict.set_item("ask_depth", ob.ask_depth)?;
            ob_dict.set_item("imbalance", ob.imbalance)?;
            dict.set_item("orderbook", ob_dict)?;
        } else {
            dict.set_item("orderbook", py.None())?;
        }

        // --- Portfolio state (structured dict, not raw JSON) ---
        {
            let port = &context.portfolio_state;
            let port_dict = PyDict::new_bound(py);
            port_dict.set_item("cash", port.balance)?;
            port_dict.set_item("total_value",
                port.balance
                    + port.unrealized_pnl)?;
            port_dict.set_item("realized_pnl", port.realized_pnl)?;
            port_dict.set_item("unrealized_pnl", port.unrealized_pnl)?;
            port_dict.set_item("fees_paid", port.fees_paid)?;
            port_dict.set_item("net_position", port.net_position)?;

            // Position count (PortfolioSnapshot doesn't carry individual positions)
            port_dict.set_item("position_count", port.position_count)?;
            // Legacy alias for dict-style strategies: portfolio["equity"]
            port_dict.set_item("equity", port.balance + port.unrealized_pnl)?;
            dict.set_item("portfolio", port_dict)?;
        }

        // --- Supplementary/custom data (string → float) ---
        if !context.custom_data.is_empty() {
            let custom_dict = PyDict::new_bound(py);
            for (k, v) in &context.custom_data {
                custom_dict.set_item(k, *v)?;
            }
            dict.set_item("custom_data", custom_dict)?;
        }

        // --- Multi-asset context ---
        if !context.assets.is_empty() {
            let assets_dict = PyDict::new_bound(py);
            for (symbol, snap) in &context.assets {
                let a = PyDict::new_bound(py);
                a.set_item("price", snap.price)?;
                a.set_item("position", snap.position)?;
                assets_dict.set_item(symbol, a)?;
            }
            dict.set_item("assets", assets_dict)?;
        }

        // --- Spread ---
        if let Some(spread) = context.spread {
            dict.set_item("spread", spread)?;
        }

        // --- Session + counters ---
        dict.set_item("market_session_open", context.market_session.is_open)?;
        dict.set_item("tick_number", context.tick_number)?;
        dict.set_item("elapsed_seconds", context.elapsed_seconds)?;

        // --- Derivatives context ---
        if let Some(ref iv) = context.iv_surface {
            let iv_dict = PyDict::new_bound(py);
            iv_dict.set_item("underlying", &iv.underlying)?;
            iv_dict.set_item("timestamp", iv.timestamp.to_rfc3339())?;
            let pts = PyList::empty_bound(py);
            for p in &iv.points {
                let pd = PyDict::new_bound(py);
                pd.set_item("strike", p.strike)?;
                pd.set_item("expiry", p.expiry.to_rfc3339())?;
                pd.set_item("iv", p.iv)?;
                pd.set_item("bid_iv", p.bid_iv)?;
                pd.set_item("ask_iv", p.ask_iv)?;
                pts.append(pd)?;
            }
            iv_dict.set_item("points", pts)?;
            dict.set_item("iv_surface", iv_dict)?;
        }

        if let Some(ref fc) = context.futures_curve {
            let fc_dict = PyDict::new_bound(py);
            fc_dict.set_item("underlying", &fc.underlying)?;
            fc_dict.set_item("spot_price", fc.spot_price)?;
            fc_dict.set_item("timestamp", fc.timestamp.to_rfc3339())?;
            let pts = PyList::empty_bound(py);
            for p in &fc.points {
                let pd = PyDict::new_bound(py);
                pd.set_item("expiry", p.expiry.to_rfc3339())?;
                pd.set_item("mark_price", p.mark_price)?;
                pd.set_item("basis_pct", p.basis_pct)?;
                pd.set_item("open_interest", p.open_interest)?;
                pts.append(pd)?;
            }
            fc_dict.set_item("points", pts)?;
            dict.set_item("futures_curve", fc_dict)?;
        }

        if let Some(ref rates) = context.funding_rates {
            let rates_dict = PyDict::new_bound(py);
            for (sym, fr) in rates {
                let fr_dict = PyDict::new_bound(py);
                fr_dict.set_item("symbol", &fr.symbol)?;
                fr_dict.set_item("rate", fr.rate)?;
                fr_dict.set_item("next_settlement", fr.next_settlement.to_rfc3339())?;
                if let Some(pred) = fr.predicted_rate {
                    fr_dict.set_item("predicted_rate", pred)?;
                }
                if let Some(ann) = fr.annualised_rate {
                    fr_dict.set_item("annualised_rate", ann)?;
                }
                rates_dict.set_item(sym, fr_dict)?;
            }
            dict.set_item("funding_rates", rates_dict)?;
        }

        if let Some(ref greeks) = context.portfolio_greeks {
            let g_dict = PyDict::new_bound(py);
            g_dict.set_item("delta", greeks.delta)?;
            g_dict.set_item("gamma", greeks.gamma)?;
            g_dict.set_item("theta", greeks.theta)?;
            g_dict.set_item("vega", greeks.vega)?;
            g_dict.set_item("rho", greeks.rho)?;
            dict.set_item("portfolio_greeks", g_dict)?;
        }

        Ok(dict)
    }

    /// Convert a Python signal dict back to a Rust Signal.
    ///
    /// Handles all signal types including:
    /// - `Signal.buy(size)`, `Signal.sell(size)`, `Signal.hold()`
    /// - `Signal.buy(size, limit=True)` for maker/passive fills
    /// - `Signal.quote(bid_price, ask_price, bid_size, ask_size)` → `TwoSidedQuote`
    /// - `Signal.cancel_quotes()` → `CancelQuotes`
    /// - `Signal.multi({asset: signal, ...})` → `CrossAsset`
    /// - Optional `metadata={"confidence": 0.95, ...}` on any signal
    fn py_signal_to_rust(py: Python<'_>, py_dict: &Bound<'_, PyDict>) -> StrategyResult<Signal> {
        let get_str = |key: &str| -> String {
            py_dict
                .get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
        };

        let get_f64 = |key: &str| -> Option<f64> {
            py_dict
                .get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
        };

        // Parse signal type — supports simple types + structured types
        let signal_type_str = get_str("signal_type");

        // Fast path: Hold signals are the most common (no trade) — skip expensive
        // UUID generation, metadata parsing, and field extraction.
        if signal_type_str == "Hold" {
            return Ok(Signal {
                id: Arc::from("hold"),
                timestamp: Utc::now(),
                signal_type: SignalType::Hold,
                strength: SignalStrength::Medium,
                price: None,
                quantity: None,
                reason: SignalReason::Custom("Python".to_string(), String::new()),
                metadata: HashMap::new(),
                numeric_metadata: HashMap::new(),
                is_limit: false,
            });
        }

        let signal_type = match signal_type_str.as_str() {
            "Buy" => SignalType::Buy,
            "Sell" => SignalType::Sell,
            "Hold" => SignalType::Hold, // unreachable due to fast path above
            "Close" => SignalType::Close,
            "ScaleIn" => SignalType::ScaleIn,
            "ScaleOut" => SignalType::ScaleOut,
            "StopLoss" => SignalType::StopLoss,
            "TakeProfit" => SignalType::TakeProfit,
            "CancelQuotes" => SignalType::CancelQuotes,
            "TwoSidedQuote" | "Quote" => {
                // Extract quote parameters
                let bid_price = get_f64("bid_price").unwrap_or(0.0);
                let ask_price = get_f64("ask_price").unwrap_or(0.0);
                let bid_size = get_f64("bid_size").unwrap_or(0.0);
                let ask_size = get_f64("ask_size").unwrap_or(0.0);
                SignalType::TwoSidedQuote {
                    bid_price,
                    ask_price,
                    bid_size,
                    ask_size,
                }
            }
            "CrossAsset" | "Multi" => {
                // Extract cross-asset parameters
                let target_symbol = get_str("target_symbol");
                let target_exchange = get_str("target_exchange");
                let action_str = get_str("action");
                let inner_action = match action_str.as_str() {
                    "Buy" => SignalType::Buy,
                    "Sell" => SignalType::Sell,
                    "Hold" => SignalType::Hold,
                    "Close" => SignalType::Close,
                    _ => SignalType::Custom(action_str),
                };
                SignalType::CrossAsset {
                    target_symbol,
                    target_exchange,
                    action: Box::new(inner_action),
                }
            }

            // ── Derivatives signal types ─────────────────────────────
            "BuyOption" => {
                let instrument = Self::py_dict_to_derivative_metadata(py, py_dict)?;
                let premium = get_f64("premium")
                    .or_else(|| get_f64("price"));
                SignalType::BuyOption { instrument, premium }
            }
            "SellOption" => {
                let instrument = Self::py_dict_to_derivative_metadata(py, py_dict)?;
                let premium = get_f64("premium")
                    .or_else(|| get_f64("price"));
                SignalType::SellOption { instrument, premium }
            }
            "BuyFuture" => {
                let instrument = Self::py_dict_to_derivative_metadata(py, py_dict)?;
                SignalType::BuyFuture { instrument }
            }
            "SellFuture" => {
                let instrument = Self::py_dict_to_derivative_metadata(py, py_dict)?;
                SignalType::SellFuture { instrument }
            }
            "OptionSpread" => {
                let legs = Self::py_dict_to_spread_legs(py, py_dict)?;
                SignalType::OptionSpread { legs }
            }
            "RollContract" => {
                let (from_inst, to_inst) = Self::py_dict_to_roll_instruments(py, py_dict)?;
                let from_symbol = from_inst.symbol.clone();
                let to_symbol = to_inst.symbol.clone();
                SignalType::RollContract {
                    from_symbol,
                    to_symbol,
                    from_instrument: from_inst,
                    to_instrument: to_inst,
                }
            }
            "ExerciseOption" => {
                let instrument = Self::py_dict_to_derivative_metadata(py, py_dict)?;
                SignalType::ExerciseOption { instrument }
            }
            "HedgeDelta" => {
                let target_delta = get_f64("target_delta").unwrap_or(0.0);
                let current_delta = get_f64("current_delta").unwrap_or(0.0);
                let hedge_instrument = get_str("hedge_instrument");
                SignalType::HedgeDelta {
                    target_delta,
                    current_delta,
                    hedge_instrument,
                }
            }

            other => SignalType::Custom(other.to_string()),
        };

        // Parse signal strength
        let strength = if let Some(custom_val) = get_f64("strength") {
            SignalStrength::Custom(custom_val)
        } else {
            match get_str("strength").as_str() {
                "Weak" => SignalStrength::Weak,
                "Medium" => SignalStrength::Medium,
                "Strong" => SignalStrength::Strong,
                _ => SignalStrength::Medium,
            }
        };

        // Parse price and quantity (size)
        let price = get_f64("price");
        // Accept both "quantity" and "size" keys
        let quantity = get_f64("quantity")
            .or_else(|| get_f64("size"));

        // Parse reason
        let reason_str = get_str("reason");
        let reason_detail = get_str("reason_detail");
        let reason = match reason_str.as_str() {
            "TechnicalAnalysis" => SignalReason::TechnicalAnalysis(reason_detail),
            "Momentum" => SignalReason::Momentum(reason_detail),
            "MeanReversion" => SignalReason::MeanReversion(reason_detail),
            "Arbitrage" => SignalReason::Arbitrage(reason_detail),
            "RiskManagement" => SignalReason::RiskManagement(reason_detail),
            "VolumeAnalysis" => SignalReason::VolumeAnalysis(reason_detail),
            "PriceAction" => SignalReason::PriceAction(reason_detail),
            _ => SignalReason::Custom("Python".to_string(), reason_detail),
        };

        // Parse metadata (mixed types go into metadata, pure f64 also into numeric_metadata)
        let mut metadata = HashMap::new();
        let mut numeric_metadata = HashMap::new();
        if let Ok(Some(meta_dict)) = py_dict.get_item("metadata") {
            if let Ok(meta) = meta_dict.downcast::<PyDict>() {
                for (key, value) in meta.iter() {
                    if let Ok(k) = key.extract::<String>() {
                        if let Ok(v) = value.extract::<f64>() {
                            metadata.insert(k.clone(), SignalMetadata::Number(v));
                            numeric_metadata.insert(k, v);
                        } else if let Ok(v) = value.extract::<bool>() {
                            metadata.insert(k, SignalMetadata::Boolean(v));
                        } else if let Ok(v) = value.extract::<String>() {
                            metadata.insert(k, SignalMetadata::Text(v));
                        }
                    }
                }
            }
        }

        let id = get_str("id");
        let id = if id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            id
        };

        // Parse limit flag — limit=True means maker/passive fill, default is taker
        let is_limit = py_dict
            .get_item("limit")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        Ok(Signal {
            id: Arc::from(id.as_str()),
            timestamp: Utc::now(),
            signal_type,
            strength,
            price,
            quantity,
            reason,
            metadata,
            numeric_metadata,
            is_limit,
        })
    }

    /// Convert one `compute_signals()`/`compute_signals_multi_venue()` output
    /// element to `i8`. The SDK documents an int8 array, but the most
    /// natural unannotated numpy code (`np.zeros(n)`, arithmetic results,
    /// `signals[mask] = 1.0`, etc.) produces **float64** by default --
    /// `.tolist()` then yields Python `float` objects, and PyO3's
    /// `i64: FromPyObject` requires an actual Python `int` (no implicit
    /// float truncation), so a bare `extract::<i64>()` FAILS for exactly
    /// that common shape.
    ///
    /// Previously that failure was silently swallowed by `.unwrap_or(0)`,
    /// turning every signal into a silent HOLD with no error surfaced --
    /// a strategy that unconditionally set `signals[5:] = 1.0` (float64)
    /// still produced a "completed" backtest with 0 trades. Try int first
    /// (the documented/native case), then f64 (the common case in
    /// practice), rounding rather than truncating so floating-point noise
    /// (e.g. `0.9999999999`) doesn't silently drop to 0.
    fn extract_signal_i8(item: &pyo3::Bound<'_, pyo3::PyAny>) -> i8 {
        if let Ok(v) = item.extract::<i64>() {
            return v.clamp(-128, 127) as i8;
        }
        if let Ok(v) = item.extract::<f64>() {
            return v.round().clamp(-128.0, 127.0) as i8;
        }
        0
    }

    /// Convert one `compute_position_sizes()` output element to `f64`.
    /// Unlike `extract_signal_i8`, returns `None` (not a clamped default) for
    /// anything non-numeric — the caller in `backtest::python_simulation`
    /// decides the per-tick fallback (flat `position_size_pct`), since a
    /// missing/invalid size at one tick must never fail the backtest, just
    /// lose the per-trade override for that one tick.
    fn extract_size_f64(item: &pyo3::Bound<'_, pyo3::PyAny>) -> Option<f64> {
        if let Ok(v) = item.extract::<f64>() {
            return Some(v);
        }
        if let Ok(v) = item.extract::<i64>() {
            return Some(v as f64);
        }
        None
    }

    /// Convert Rust parameters to Python dict.
    fn params_to_py<'py>(
        py: Python<'py>,
        params: &HashMap<String, ParameterValue>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (key, value) in params {
            match value {
                ParameterValue::Float(f) => dict.set_item(key, *f)?,
                ParameterValue::Int(i) => dict.set_item(key, *i)?,
                ParameterValue::String(s) => dict.set_item(key, s)?,
                ParameterValue::Bool(b) => dict.set_item(key, *b)?,
                ParameterValue::Array(_) => {} // Skip arrays for now
            }
        }
        Ok(dict)
    }

    // ── Derivatives helpers ──────────────────────────────────────────

    /// Extract `DerivativeMetadata` from the Python signal dict's `metadata.instrument`.
    fn py_dict_to_derivative_metadata(
        _py: Python<'_>,
        py_dict: &Bound<'_, PyDict>,
    ) -> StrategyResult<DerivativeMetadata> {
        // Look in metadata.instrument first, then top-level metadata
        let meta_obj = py_dict
            .get_item("metadata")
            .ok()
            .flatten()
            .and_then(|m| m.downcast::<PyDict>().ok().map(|d| d.clone()));

        let inst_dict = meta_obj
            .as_ref()
            .and_then(|m| m.get_item("instrument").ok().flatten())
            .and_then(|i| i.downcast::<PyDict>().ok().map(|d| d.clone()));

        let d = inst_dict.as_ref().unwrap_or_else(|| {
            meta_obj.as_ref().unwrap_or(py_dict)
        });

        let get = |key: &str| -> String {
            d.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
        };
        let get_f64 = |key: &str| -> f64 {
            d.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(0.0)
        };
        let get_opt_f64 = |key: &str| -> Option<f64> {
            d.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
        };

        let kind_str = get("instrument_kind");
        let instrument_kind = Self::str_to_instrument_kind(
            &kind_str,
            get_opt_f64("strike"),
            d.get_item("expiry")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok()),
        );

        Ok(DerivativeMetadata::new(
            get("symbol"),
            get("underlying"),
            instrument_kind,
            {
                let m = get_f64("contract_multiplier");
                if m == 0.0 { 1.0 } else { m }
            },
            get("settlement_currency"),
            get("exchange"),
        ))
    }

    /// Parse an instrument kind string + optional strike/expiry into `InstrumentKind`.
    fn str_to_instrument_kind(
        kind: &str,
        strike: Option<f64>,
        expiry: Option<String>,
    ) -> InstrumentKind {
        let parse_expiry = |s: &str| -> DateTime<Utc> {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|_| {
                    s.parse::<chrono::NaiveDateTime>()
                        .map(|ndt| ndt.and_utc())
                })
                .unwrap_or_else(|_| Utc::now())
        };

        match kind {
            "Spot" => InstrumentKind::Spot,
            "Perpetual" => InstrumentKind::Perpetual,
            "Future" => InstrumentKind::Future {
                expiry: expiry
                    .map(|e| parse_expiry(&e))
                    .unwrap_or_else(Utc::now),
            },
            "Call" => InstrumentKind::Call {
                strike: strike.unwrap_or(0.0),
                expiry: expiry
                    .map(|e| parse_expiry(&e))
                    .unwrap_or_else(Utc::now),
            },
            "Put" => InstrumentKind::Put {
                strike: strike.unwrap_or(0.0),
                expiry: expiry
                    .map(|e| parse_expiry(&e))
                    .unwrap_or_else(Utc::now),
            },
            _ => InstrumentKind::Spot,
        }
    }

    /// Extract spread legs from the Python signal dict's `metadata.legs`.
    fn py_dict_to_spread_legs(
        py: Python<'_>,
        py_dict: &Bound<'_, PyDict>,
    ) -> StrategyResult<Vec<SpreadLeg>> {
        let meta_obj = py_dict
            .get_item("metadata")
            .ok()
            .flatten()
            .and_then(|m| m.downcast::<PyDict>().ok().map(|d| d.clone()));

        let legs_list = meta_obj
            .as_ref()
            .and_then(|m| m.get_item("legs").ok().flatten())
            .and_then(|l| l.downcast::<pyo3::types::PyList>().ok().map(|l| l.clone()));

        let Some(legs) = legs_list else {
            return Ok(vec![]);
        };

        let mut result = Vec::new();
        for item in legs.iter() {
            let Ok(leg_dict) = item.downcast::<PyDict>() else {
                continue;
            };
            let inst = leg_dict
                .get_item("instrument")
                .ok()
                .flatten()
                .and_then(|i| i.downcast::<PyDict>().ok().map(|d| d.clone()));

            // Build a temp dict that looks like a signal dict with metadata.instrument
            let tmp = PyDict::new_bound(py);
            let tmp_meta = PyDict::new_bound(py);
            if let Some(ref i) = inst {
                tmp_meta
                    .set_item("instrument", i)
                    .map_err(|e| StrategyError::MarketDataError(format!("set instrument error: {e}")))?;
            }
            tmp.set_item("metadata", &tmp_meta)
                .map_err(|e| StrategyError::MarketDataError(format!("set metadata error: {e}")))?;

            let instrument = Self::py_dict_to_derivative_metadata(py, &tmp)?;

            let action_str: String = leg_dict
                .get_item("action")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "BuyOption".to_string());

            let ratio: i32 = leg_dict
                .get_item("ratio")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(1);

            let limit_price: Option<f64> = leg_dict
                .get_item("limit_price")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok());

            // Build the inner signal type for this leg
            let leg_signal = match action_str.as_str() {
                "BuyOption" => SignalType::BuyOption {
                    instrument: instrument.clone(),
                    premium: limit_price,
                },
                "SellOption" => SignalType::SellOption {
                    instrument: instrument.clone(),
                    premium: limit_price,
                },
                "BuyFuture" => SignalType::BuyFuture {
                    instrument: instrument.clone(),
                },
                "SellFuture" => SignalType::SellFuture {
                    instrument: instrument.clone(),
                },
                _ => SignalType::BuyOption {
                    instrument: instrument.clone(),
                    premium: limit_price,
                },
            };

            result.push(SpreadLeg {
                signal_type: leg_signal,
                ratio,
                limit_price,
            });
        }
        Ok(result)
    }

    /// Extract roll-contract from/to instrument pair from metadata.
    fn py_dict_to_roll_instruments(
        py: Python<'_>,
        py_dict: &Bound<'_, PyDict>,
    ) -> StrategyResult<(DerivativeMetadata, DerivativeMetadata)> {
        let meta_obj = py_dict
            .get_item("metadata")
            .ok()
            .flatten()
            .and_then(|m| m.downcast::<PyDict>().ok().map(|d| d.clone()));

        // from_instrument
        let from_tmp = PyDict::new_bound(py);
        let from_meta = PyDict::new_bound(py);
        if let Some(ref m) = meta_obj {
            if let Ok(Some(fi)) = m.get_item("from_instrument") {
                from_meta
                    .set_item("instrument", fi)
                    .map_err(|e| StrategyError::MarketDataError(format!("set from_instrument error: {e}")))?;
            }
        }
        from_tmp
            .set_item("metadata", &from_meta)
            .map_err(|e| StrategyError::MarketDataError(format!("set from metadata error: {e}")))?;
        let from_inst = Self::py_dict_to_derivative_metadata(py, &from_tmp)?;

        // to_instrument
        let to_tmp = PyDict::new_bound(py);
        let to_meta = PyDict::new_bound(py);
        if let Some(ref m) = meta_obj {
            if let Ok(Some(ti)) = m.get_item("to_instrument") {
                to_meta
                    .set_item("instrument", ti)
                    .map_err(|e| StrategyError::MarketDataError(format!("set to_instrument error: {e}")))?;
            }
        }
        to_tmp
            .set_item("metadata", &to_meta)
            .map_err(|e| StrategyError::MarketDataError(format!("set to metadata error: {e}")))?;
        let to_inst = Self::py_dict_to_derivative_metadata(py, &to_tmp)?;

        Ok((from_inst, to_inst))
    }

    /// Convert a Rust context dict (PyDict) into a Python StrategyContext object
    /// via StrategyContext._from_dict(). Used by lifecycle hooks (on_expiry etc.).
    fn dict_to_strategy_context<'py>(
        py: Python<'py>,
        ctx_dict: &Bound<'py, PyDict>,
    ) -> StrategyResult<Bound<'py, PyAny>> {
        let tp_mod = py.import_bound("trading_platform").map_err(|e| {
            StrategyError::CalculationError(format!("Cannot import trading_platform: {e}"))
        })?;
        let sc_class = tp_mod.getattr("StrategyContext").map_err(|e| {
            StrategyError::CalculationError(format!("Cannot get StrategyContext class: {e}"))
        })?;
        sc_class.call_method1("_from_dict", (ctx_dict,)).map_err(|e| {
            StrategyError::CalculationError(format!("StrategyContext._from_dict error: {e}"))
        })
    }
}

#[async_trait]
impl Strategy for PythonStrategy {
    fn name(&self) -> &str {
        &self.cached_name
    }

    fn version(&self) -> &str {
        &self.cached_version
    }

    fn description(&self) -> &str {
        &self.cached_description
    }

    fn declared_max_leverage(&self) -> f64 {
        self.cached_max_leverage
    }

    async fn initialize(
        &mut self,
        parameters: HashMap<String, ParameterValue>,
    ) -> StrategyResult<()> {
        self.parameters = parameters.clone();

        // Initialize Python interpreter and load strategy
        let py_obj = self.init_python()?;

        // Call Python initialize() with parameters.
        //
        // Strategy classes are expected to inherit BaseStrategy, whose
        // initialize(self, parameters: dict) accepts exactly this call
        // shape. Two malformed-but-real submissions show up in practice
        // (confirmed against live AI-generated strategy jobs --
        // backtest_jobs 1e0e7cc8/2efb9572/31eaa6ff/4b1c5a94, all predating
        // the SDK-injection-race fix and unaffected by it, since these are
        // deterministic authoring defects, not a race):
        //   1. The class doesn't inherit BaseStrategy at all, so it has no
        //      `initialize` attribute -- raises AttributeError. Treated as
        //      a no-op init rather than a hard failure: nothing in the
        //      actually-required interface (name(), compute_signals())
        //      depends on it.
        //   2. The class overrides `initialize(self)` with no parameters
        //      arg, raising "takes 1 positional argument but 2 were given"
        //      when called with the params dict. Retried with a bare
        //      zero-arg call -- a strategy that declined the parameters arg
        //      has nothing to do with the dict anyway.
        Python::with_gil(|py| {
            let strategy = py_obj.bind(py);

            if !strategy.hasattr("initialize").unwrap_or(false) {
                return Ok::<(), StrategyError>(());
            }

            let params_dict = Self::params_to_py(py, &parameters).map_err(|e| {
                StrategyError::ConfigurationError(format!("Parameter conversion error: {}", e))
            })?;

            if let Err(e) = strategy.call_method1("initialize", (params_dict,)) {
                let arity_mismatch = e.to_string().contains("positional argument");
                if arity_mismatch && strategy.call_method0("initialize").is_ok() {
                    return Ok(());
                }
                return Err(StrategyError::ConfigurationError(format!(
                    "Python initialize() error: {}",
                    e
                )));
            }

            Ok::<(), StrategyError>(())
        })?;

        // Cache the SDK wrapper classes so we don't re-import them every tick
        Python::with_gil(|py| {
            let tp = py.import_bound("trading_platform.types").map_err(|e| {
                StrategyError::ConfigurationError(format!("Failed to import trading_platform.types: {}", e))
            })?;
            self.cached_md_class = Some(tp.getattr("MarketData").map_err(|e| {
                StrategyError::ConfigurationError(format!("MarketData class not found: {}", e))
            })?.unbind());
            self.cached_sc_class = Some(tp.getattr("StrategyContext").map_err(|e| {
                StrategyError::ConfigurationError(format!("StrategyContext class not found: {}", e))
            })?.unbind());

            // Cache numpy.frombuffer + float64/int64 dtypes for fast array construction
            if let Ok(np) = py.import_bound("numpy") {
                self.cached_np_frombuffer = Some(np.getattr("frombuffer").map_err(|e| {
                    StrategyError::ConfigurationError(format!("numpy.frombuffer not found: {}", e))
                })?.unbind());
                self.cached_np_float64 = Some(np.getattr("float64").map_err(|e| {
                    StrategyError::ConfigurationError(format!("numpy.float64 not found: {}", e))
                })?.unbind());
                self.cached_np_int64 = Some(np.getattr("int64").map_err(|e| {
                    StrategyError::ConfigurationError(format!("numpy.int64 not found: {}", e))
                })?.unbind());
            }

            // Detect whether the user's compute_signals accepts the optional OHLC
            // arrays (opens/highs/lows). We read the function's __code__ directly
            // rather than importing `inspect` (keeps us off the sandbox import path).
            // OHLC is passed as keyword args, so we accept when: a param is named
            // opens/highs/lows, OR the function takes **kwargs.
            self.compute_signals_accepts_ohlc = {
                let strat = py_obj.bind(py);
                strat
                    .getattr("compute_signals")
                    .and_then(|m| m.getattr("__code__"))
                    .map(|code| {
                        let argcount: usize = code
                            .getattr("co_argcount")
                            .and_then(|v| v.extract())
                            .unwrap_or(0);
                        let kwonly: usize = code
                            .getattr("co_kwonlyargcount")
                            .and_then(|v| v.extract())
                            .unwrap_or(0);
                        let flags: u32 = code
                            .getattr("co_flags")
                            .and_then(|v| v.extract())
                            .unwrap_or(0);
                        // CO_VARKEYWORDS = 0x08
                        let has_varkw = (flags & 0x08) != 0;
                        let names: Vec<String> = code
                            .getattr("co_varnames")
                            .and_then(|v| v.extract())
                            .unwrap_or_default();
                        // Only the first (argcount + kwonly) entries are parameters;
                        // the rest are local variables and must be ignored.
                        let nparams = (argcount + kwonly).min(names.len());
                        let named_ohlc = names[..nparams]
                            .iter()
                            .any(|n| n == "opens" || n == "highs" || n == "lows");
                        named_ohlc || has_varkw
                    })
                    .unwrap_or(false)
            };

            // Cross-exchange strategy support (2026-07-22): does the user's
            // class define compute_signals_multi_venue at all? Simple
            // presence check -- this method has one fixed shape, unlike
            // compute_signals' optional OHLC kwargs above.
            self.accepts_multi_venue = {
                let strat = py_obj.bind(py);
                strat.hasattr("compute_signals_multi_venue").unwrap_or(false)
            };

            // Per-trade dynamic position sizing (2026-07-25): does the user's
            // class define compute_position_sizes at all? Simple presence
            // check, same reasoning as accepts_multi_venue above -- this
            // method has one fixed shape (prices, volumes, timestamps) ->
            // array of fractions, no optional kwargs to introspect.
            self.accepts_position_sizes = {
                let strat = py_obj.bind(py);
                strat.hasattr("compute_position_sizes").unwrap_or(false)
            };

            // Does the user's class define generate_signals? Simple presence
            // check, same reasoning as accepts_multi_venue/accepts_position_sizes
            // above -- when true, the executor dispatches per-tick to this
            // context-carrying method instead of the vectorized compute_signals
            // bulk path.
            self.defines_generate_signals = {
                let strat = py_obj.bind(py);
                strat.hasattr("generate_signals").unwrap_or(false)
            };

            // Pairs-trading support: does the user's class define pair_spec()?
            // If so, call it once (not per-tick) to extract the declaration --
            // this method's return value must actually be read, unlike the
            // simple presence checks above.
            //
            // Distinguishes "method not defined" from "method defined but its
            // return value doesn't match the required shape" (a real incident:
            // an AI-authored pair_spec() returned basket_spec()'s {legs: [...]}
            // shape instead of the required flat symbol_a/exchange_a/symbol_b/
            // exchange_b keys -- the old Option-only version silently produced
            // None either way, so worker.rs's error message claimed pair_spec()
            // "does not define" the method when it actually existed and ran,
            // just with the wrong keys). pair_spec_error surfaces which case it
            // was so the caller can report the real problem.
            let (pair_spec, pair_spec_error): (Option<PairSpec>, Option<String>) = {
                let strat = py_obj.bind(py);
                if strat.hasattr("pair_spec").unwrap_or(false) {
                    let parsed: Result<PairSpec, String> = (|| -> Result<PairSpec, String> {
                        let result = strat.call_method0("pair_spec")
                            .map_err(|e| format!("pair_spec() raised an exception: {}", e))?;
                        let dict = result.downcast::<pyo3::types::PyDict>()
                            .map_err(|_| "pair_spec() must return a dict".to_string())?;
                        let get_str = |key: &str| -> Option<String> {
                            dict.get_item(key).ok().flatten()?.extract().ok()
                        };
                        let missing_shape_hint = || {
                            if dict.get_item("legs").ok().flatten().is_some() {
                                " (this dict has a 'legs' key -- that's basket_spec()'s shape, \
                                  not pair_spec()'s; pair_spec() needs flat symbol_a/exchange_a/\
                                  symbol_b/exchange_b keys, one leg per key, not a legs list)"
                                    .to_string()
                            } else {
                                String::new()
                            }
                        };
                        let symbol_a = get_str("symbol_a").ok_or_else(|| format!(
                            "pair_spec() dict is missing required key 'symbol_a'{}", missing_shape_hint()
                        ))?;
                        let exchange_a = get_str("exchange_a").ok_or_else(|| format!(
                            "pair_spec() dict is missing required key 'exchange_a'{}", missing_shape_hint()
                        ))?;
                        let symbol_b = get_str("symbol_b").ok_or_else(|| format!(
                            "pair_spec() dict is missing required key 'symbol_b'{}", missing_shape_hint()
                        ))?;
                        let exchange_b = get_str("exchange_b").ok_or_else(|| format!(
                            "pair_spec() dict is missing required key 'exchange_b'{}", missing_shape_hint()
                        ))?;
                        let mode_str = get_str("hedge_ratio_mode").unwrap_or_else(|| "fixed".to_string());
                        let hedge_ratio_mode = if mode_str.eq_ignore_ascii_case("dynamic") {
                            HedgeRatioMode::Dynamic
                        } else {
                            let ratio: f64 = dict
                                .get_item("hedge_ratio")
                                .ok()
                                .flatten()
                                .and_then(|v| v.extract().ok())
                                .unwrap_or(1.0);
                            HedgeRatioMode::Fixed(ratio)
                        };
                        Ok(PairSpec { symbol_a, exchange_a, symbol_b, exchange_b, hedge_ratio_mode })
                    })();
                    match parsed {
                        Ok(spec) => (Some(spec), None),
                        Err(e) => (None, Some(e)),
                    }
                } else {
                    (None, None)
                }
            };
            self.pair_spec = pair_spec;
            self.pair_spec_error = pair_spec_error;

            // Basket statistical-arbitrage support: does the user's class
            // define basket_spec()? Mirrors pair_spec's exact pattern,
            // including capturing WHY parsing failed (see pair_spec_error's
            // doc comment above) rather than collapsing every failure into a
            // bare None indistinguishable from "method not defined".
            let (basket_spec, basket_spec_error): (Option<BasketSpec>, Option<String>) = {
                let strat = py_obj.bind(py);
                if strat.hasattr("basket_spec").unwrap_or(false) {
                    let parsed: Result<BasketSpec, String> = (|| -> Result<BasketSpec, String> {
                        let result = strat.call_method0("basket_spec")
                            .map_err(|e| format!("basket_spec() raised an exception: {}", e))?;
                        let dict = result.downcast::<pyo3::types::PyDict>()
                            .map_err(|_| "basket_spec() must return a dict".to_string())?;
                        let shape_hint = if dict.get_item("symbol_a").ok().flatten().is_some() {
                            " (this dict has a 'symbol_a' key -- that's pair_spec()'s shape, not \
                              basket_spec()'s; basket_spec() needs a 'legs' list of {symbol, \
                              exchange} dicts instead)"
                        } else {
                            ""
                        };
                        let legs_value = dict.get_item("legs").ok().flatten()
                            .ok_or_else(|| format!("basket_spec() dict is missing required key 'legs'{}", shape_hint))?;
                        let legs_list = legs_value.downcast::<pyo3::types::PyList>()
                            .map_err(|_| "basket_spec()'s 'legs' value must be a list".to_string())?;
                        let mut legs = Vec::with_capacity(legs_list.len());
                        for item in legs_list.iter() {
                            let leg_dict = item.downcast::<pyo3::types::PyDict>()
                                .map_err(|_| "each entry in basket_spec()'s 'legs' list must be a dict".to_string())?;
                            let symbol: String = leg_dict.get_item("symbol").ok().flatten()
                                .ok_or_else(|| "a leg in basket_spec()'s 'legs' list is missing 'symbol'".to_string())?
                                .extract().map_err(|_| "a leg's 'symbol' must be a string".to_string())?;
                            let exchange: String = leg_dict.get_item("exchange").ok().flatten()
                                .ok_or_else(|| "a leg in basket_spec()'s 'legs' list is missing 'exchange'".to_string())?
                                .extract().map_err(|_| "a leg's 'exchange' must be a string".to_string())?;
                            legs.push(BasketLeg { symbol, exchange });
                        }
                        if legs.len() < 3 {
                            return Err(format!(
                                "basket_spec() declared only {} leg(s) -- basket_spec is for 3+ legs; \
                                 use pair_spec() for exactly 2", legs.len()
                            ));
                        }
                        let mode_str: String = dict.get_item("weight_mode").ok().flatten()
                            .and_then(|v| v.extract().ok())
                            .unwrap_or_else(|| "fixed".to_string());
                        let weight_mode = if mode_str.eq_ignore_ascii_case("dynamic") {
                            BasketWeightMode::Dynamic
                        } else {
                            let weights_value = dict.get_item("weights").ok().flatten()
                                .ok_or_else(|| "basket_spec() dict is missing required key 'weights' (or set weight_mode: \"dynamic\")".to_string())?;
                            let weights_list = weights_value.downcast::<pyo3::types::PyList>()
                                .map_err(|_| "basket_spec()'s 'weights' value must be a list".to_string())?;
                            let mut weights = Vec::with_capacity(weights_list.len());
                            for w in weights_list.iter() {
                                weights.push(w.extract::<f64>().map_err(|_| "basket_spec()'s 'weights' entries must be numbers".to_string())?);
                            }
                            if weights.len() != legs.len() {
                                return Err(format!(
                                    "basket_spec() declared {} legs but {} weights -- these must match",
                                    legs.len(), weights.len()
                                ));
                            }
                            BasketWeightMode::Fixed(weights)
                        };
                        Ok(BasketSpec { legs, weight_mode })
                    })();
                    match parsed {
                        Ok(spec) => (Some(spec), None),
                        Err(e) => (None, Some(e)),
                    }
                } else {
                    (None, None)
                }
            };
            self.basket_spec = basket_spec;
            self.basket_spec_error = basket_spec_error;

            Ok::<(), StrategyError>(())
        })?;

        // Store the initialized strategy object
        let mut lock = self.py_strategy.lock().await;
        *lock = Some(py_obj);
        self.initialized = true;

        // Set up a single persistent watchdog thread in Python.
        // Instead of spawning a new thread per tick (which exhausts OS threads),
        // we spawn ONE watchdog that checks a shared deadline variable every second.
        // generate_signals() just updates _wd_deadline before each call.
        if !self.watchdog_initialized {
            Python::with_gil(|py| {
                let watchdog_init = r#"
import _thread, time as _wd_time
_wd_deadline = [0.0]
def _persistent_watchdog():
    while True:
        _wd_time.sleep(1.0)
        d = _wd_deadline[0]
        if d > 0.0 and _wd_time.time() > d:
            _wd_deadline[0] = 0.0
            _thread.interrupt_main()
_thread.start_new_thread(_persistent_watchdog, ())
"#;
                let _ = py.run_bound(watchdog_init, None, None);
            });
            self.watchdog_initialized = true;
        }

        Ok(())
    }

    async fn generate_signals(
        &mut self,
        market_data: &dataloader::MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        if !self.initialized {
            return Err(StrategyError::NotInitialized);
        }

        let lock = self.py_strategy.lock().await;
        let py_strategy = lock.as_ref().ok_or(StrategyError::NotInitialized)?;

        // Capture timeout duration before entering the GIL closure.
        let timeout = Duration::from_secs(self.timeout_secs);

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);

            // Arm the persistent watchdog by setting a deadline.
            // The single watchdog thread (started in initialize()) checks this
            // every second and interrupts if the deadline is exceeded.
            let arm_code = format!(
                "_wd_deadline[0] = __import__('time').time() + {}",
                timeout.as_secs_f64()
            );
            let _ = py.run_bound(&arm_code, None, None);

            let data_dict = Self::market_data_to_py(py, market_data).map_err(|e| {
                StrategyError::MarketDataError(format!("Data conversion error: {}", e))
            })?;

            let context_dict = Self::context_to_py(
                py, context,
                self.cached_np_frombuffer.as_ref(),
                self.cached_np_float64.as_ref(),
                self.cached_np_int64.as_ref(),
                &mut self.prices_buf,
                &mut self.volumes_buf,
                &mut self.timestamps_buf,
            ).map_err(|e| {
                StrategyError::MarketDataError(format!("Context conversion error: {}", e))
            })?;

            // Wrap raw dicts in typed SDK objects (MarketData / StrategyContext).
            let md_class = self.cached_md_class.as_ref().ok_or_else(|| {
                StrategyError::NotInitialized
            })?.bind(py);
            let sc_class = self.cached_sc_class.as_ref().ok_or_else(|| {
                StrategyError::NotInitialized
            })?.bind(py);

            let wrapped_data = md_class.call_method1("_from_dict", (&data_dict,)).map_err(|e| {
                StrategyError::MarketDataError(format!("MarketData wrap error: {}", e))
            })?;
            let wrapped_ctx = sc_class.call_method1("_from_dict", (&context_dict,)).map_err(|e| {
                StrategyError::MarketDataError(format!("StrategyContext wrap error: {}", e))
            })?;

            let result = strategy
                .call_method1("generate_signals", (wrapped_data, wrapped_ctx))
                .map_err(|e| {
                    // Disarm the watchdog on error path too
                    let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);
                    let err_str = format!("{}", e);
                    if err_str.contains("KeyboardInterrupt") {
                        return StrategyError::CalculationError(format!(
                            "Python strategy execution timed out after {}s",
                            timeout.as_secs()
                        ));
                    }
                    StrategyError::CalculationError(format!(
                        "Python generate_signals() error: {}",
                        e
                    ))
                })?;

            // Disarm the watchdog — execution completed within the deadline.
            let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);

            // Convert Python signal(s) to Vec<Signal>.
            // Accept both a single Signal and a list of Signals.
            let items: Vec<Bound<'_, PyAny>> = if let Ok(py_list) = result.downcast::<PyList>() {
                py_list.iter().collect()
            } else {
                // Single Signal object returned — wrap in a vec
                vec![result.clone()]
            };

            let mut signals = Vec::with_capacity(items.len());
            for item in &items {
                // The item should be a Signal object with a to_dict() method
                let dict_result = item.call_method0("to_dict");
                let dict = match dict_result {
                    Ok(d) => d,
                    Err(e) => {
                        log_debug!(STRATEGY_LOGGER, "to_dict() failed: {} | type: {:?}", e, item.get_type());
                        // Try treating it as a raw dict
                        item.clone()
                    }
                };

                if let Ok(d) = dict.downcast::<PyDict>() {
                    match Self::py_signal_to_rust(py, d) {
                        Ok(sig) => signals.push(sig),
                        Err(e) => {
                            log_debug!(STRATEGY_LOGGER, "py_signal_to_rust failed: {:?} | dict: {:?}", e, d);
                            return Err(e);
                        }
                    }
                } else {
                    log_debug!(STRATEGY_LOGGER, "downcast to PyDict failed | type: {:?}", dict.get_type());
                }
            }

            Ok(signals)
        })
    }

    /// Bulk-vectorized signal generation.
    ///
    /// Passes the entire tick array to Python as numpy arrays in a SINGLE FFI call,
    /// eliminating ~1M PyO3 crossings per chromosome evaluation.
    /// Returns `None` if the strategy doesn't implement `compute_signals`.
    async fn compute_all_signals(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> StrategyResult<Option<Vec<i8>>> {
        if !self.initialized {
            return Ok(None);
        }

        let lock = self.py_strategy.lock().await;
        let py_strategy = lock.as_ref().ok_or(StrategyError::NotInitialized)?;

        let timeout = Duration::from_secs(self.timeout_secs);

        // Copy slices so they can be moved into the GIL closure.
        let prices_vec = prices.to_vec();
        let volumes_vec = volumes.to_vec();
        let timestamps_vec = timestamps.to_vec();
        // OHLC is only forwarded when the strategy opted in (signature inspected
        // at init). When it didn't, skip the copies entirely to avoid churn on
        // the hot GA path.
        let pass_ohlc = self.compute_signals_accepts_ohlc;
        let (opens_vec, highs_vec, lows_vec) = if pass_ohlc {
            (opens.to_vec(), highs.to_vec(), lows.to_vec())
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        Python::with_gil(|py| {
            // Clone cached numpy refs inside the GIL — Py<T>::clone_ref(py) is the
            // correct way to clone a Py<T> in PyO3 (avoids requiring Clone on Py<T>).
            let np_frombuffer = self.cached_np_frombuffer.as_ref().map(|p| p.clone_ref(py));
            let np_float64   = self.cached_np_float64.as_ref().map(|p| p.clone_ref(py));
            let np_int64     = self.cached_np_int64.as_ref().map(|p| p.clone_ref(py));

            let strategy = py_strategy.bind(py);

            // Fast-path: skip if the method doesn't exist (native / built-in strategies)
            if !strategy.hasattr("compute_signals").unwrap_or(false) {
                return Ok(None);
            }

            // Arm the watchdog
            let arm_code = format!(
                "_wd_deadline[0] = __import__('time').time() + {}",
                timeout.as_secs_f64()
            );
            let _ = py.run_bound(&arm_code, None, None);

            // Build numpy arrays via frombuffer (zero-copy path) ─────────────
            // frombuffer needs a bytes object where each element is raw IEEE-754 bytes.
            // We build the bytes by reinterpreting the slice as u8 bytes.
            let build_f64_array = |data: &[f64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(float64)) = (np_frombuffer.as_ref(), np_float64.as_ref()) {
                    // SAFETY: f64 is Pod — reinterpret as u8 for frombuffer
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<f64>(),
                        )
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, float64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };
            let build_i64_array = |data: &[i64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(int64)) = (np_frombuffer.as_ref(), np_int64.as_ref()) {
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<i64>(),
                        )
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, int64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };

            let prices_arr = build_f64_array(&prices_vec).map_err(|e| {
                StrategyError::MarketDataError(format!("prices array build error: {}", e))
            })?;
            let volumes_arr = build_f64_array(&volumes_vec).map_err(|e| {
                StrategyError::MarketDataError(format!("volumes array build error: {}", e))
            })?;
            let timestamps_arr = build_i64_array(&timestamps_vec).map_err(|e| {
                StrategyError::MarketDataError(format!("timestamps array build error: {}", e))
            })?;

            // Single Python call for all N ticks ─────────────────────────────
            // When the strategy opted into OHLC, pass opens/highs/lows as keyword
            // arguments (works for both explicit params and **kwargs forms).
            let call_result = if pass_ohlc {
                let opens_arr = build_f64_array(&opens_vec).map_err(|e| {
                    StrategyError::MarketDataError(format!("opens array build error: {}", e))
                })?;
                let highs_arr = build_f64_array(&highs_vec).map_err(|e| {
                    StrategyError::MarketDataError(format!("highs array build error: {}", e))
                })?;
                let lows_arr = build_f64_array(&lows_vec).map_err(|e| {
                    StrategyError::MarketDataError(format!("lows array build error: {}", e))
                })?;
                let kwargs = pyo3::types::PyDict::new_bound(py);
                let _ = kwargs.set_item("opens", opens_arr);
                let _ = kwargs.set_item("highs", highs_arr);
                let _ = kwargs.set_item("lows", lows_arr);
                strategy.call_method(
                    "compute_signals",
                    (prices_arr, volumes_arr, timestamps_arr),
                    Some(&kwargs),
                )
            } else {
                strategy.call_method1(
                    "compute_signals",
                    (prices_arr, volumes_arr, timestamps_arr),
                )
            };

            let result = call_result
                .map_err(|e| {
                    let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);
                    let msg = format!("{}", e);
                    if msg.contains("KeyboardInterrupt") {
                        StrategyError::CalculationError(format!(
                            "compute_signals() timed out after {}s",
                            timeout.as_secs()
                        ))
                    } else {
                        StrategyError::CalculationError(format!("compute_signals() error: {}", msg))
                    }
                })?;

            let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);

            // Extract int8 ndarray ─────────────────────────────────────────────
            // Call tolist() to get a Python list of ints — avoids unsafe buffer casts.
            let py_list = result
                .call_method0("tolist")
                .or_else(|_| {
                    // result might already be a list (if user returned plain list)
                    Ok::<pyo3::Bound<'_, PyAny>, StrategyError>(result.clone())
                })
                .map_err(|e: StrategyError| StrategyError::CalculationError(
                    format!("compute_signals(): failed to convert output: {}", e)
                ))?;

            let items: Vec<pyo3::Bound<'_, PyAny>> =
                if let Ok(lst) = py_list.downcast::<pyo3::types::PyList>() {
                    lst.iter().collect()
                } else {
                    return Err(StrategyError::CalculationError(
                        "compute_signals() must return an ndarray or list of int8".to_string()
                    ));
                };

            let mut sigs = Vec::with_capacity(items.len());
            for item in &items {
                sigs.push(Self::extract_signal_i8(item));
            }

            log_debug!(
                STRATEGY_LOGGER,
                "compute_all_signals: {} ticks, 1 FFI call (vectorized path)",
                sigs.len()
            );

            Ok(Some(sigs))
        })
    }

    /// Bulk-vectorized per-trade position sizing (2026-07-25).
    ///
    /// Mirrors `compute_all_signals`'s single-FFI-call shape exactly, but for
    /// the optional `compute_position_sizes(prices, volumes, timestamps) ->
    /// np.ndarray` hook: one float (fraction of equity, `(0.0, 1.0]`) per
    /// tick. Unlike `compute_signals`, this must NEVER fail the backtest --
    /// sizing is an enhancement layered on top of the direction contract, not
    /// part of it. Any Python exception, or an output that isn't list-like,
    /// is caught here and turned into `Ok(None)` (logged as a warning) so the
    /// caller (`backtest::python_simulation`) falls back to the flat
    /// `position_size_pct` for the whole run. Per-tick invalid values (NaN,
    /// non-numeric) are mapped to `f64::NAN` rather than dropped, so the
    /// returned `Vec` still lines up index-for-index with the tick array --
    /// the caller's own `is_finite()` filter treats NaN as "fall back to
    /// position_size_pct for just this tick".
    async fn compute_all_position_sizes(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> StrategyResult<Option<Vec<f64>>> {
        if !self.initialized || !self.accepts_position_sizes {
            return Ok(None);
        }

        let lock = self.py_strategy.lock().await;
        let py_strategy = lock.as_ref().ok_or(StrategyError::NotInitialized)?;

        let timeout = Duration::from_secs(self.timeout_secs);

        let prices_vec = prices.to_vec();
        let volumes_vec = volumes.to_vec();
        let timestamps_vec = timestamps.to_vec();

        Python::with_gil(|py| {
            let np_frombuffer = self.cached_np_frombuffer.as_ref().map(|p| p.clone_ref(py));
            let np_float64   = self.cached_np_float64.as_ref().map(|p| p.clone_ref(py));
            let np_int64     = self.cached_np_int64.as_ref().map(|p| p.clone_ref(py));

            let strategy = py_strategy.bind(py);

            if !strategy.hasattr("compute_position_sizes").unwrap_or(false) {
                return Ok(None);
            }

            let arm_code = format!(
                "_wd_deadline[0] = __import__('time').time() + {}",
                timeout.as_secs_f64()
            );
            let _ = py.run_bound(&arm_code, None, None);

            let build_f64_array = |data: &[f64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(float64)) = (np_frombuffer.as_ref(), np_float64.as_ref()) {
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<f64>(),
                        )
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, float64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };
            let build_i64_array = |data: &[i64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(int64)) = (np_frombuffer.as_ref(), np_int64.as_ref()) {
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<i64>(),
                        )
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, int64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };

            let prices_arr = match build_f64_array(&prices_vec) {
                Ok(a) => a,
                Err(e) => {
                    log_warn!(STRATEGY_LOGGER, "compute_position_sizes(): prices array build error: {} — falling back to position_size_pct", e);
                    return Ok(None);
                }
            };
            let volumes_arr = match build_f64_array(&volumes_vec) {
                Ok(a) => a,
                Err(e) => {
                    log_warn!(STRATEGY_LOGGER, "compute_position_sizes(): volumes array build error: {} — falling back to position_size_pct", e);
                    return Ok(None);
                }
            };
            let timestamps_arr = match build_i64_array(&timestamps_vec) {
                Ok(a) => a,
                Err(e) => {
                    log_warn!(STRATEGY_LOGGER, "compute_position_sizes(): timestamps array build error: {} — falling back to position_size_pct", e);
                    return Ok(None);
                }
            };

            let call_result = strategy.call_method1(
                "compute_position_sizes",
                (prices_arr, volumes_arr, timestamps_arr),
            );

            let result = match call_result {
                Ok(r) => r,
                Err(e) => {
                    let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);
                    log_warn!(
                        STRATEGY_LOGGER,
                        "compute_position_sizes() failed: {} — falling back to position_size_pct for this run",
                        e
                    );
                    return Ok(None);
                }
            };

            let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);

            let py_list = match result.call_method0("tolist") {
                Ok(l) => l,
                Err(_) => result.clone(),
            };

            let items: Vec<pyo3::Bound<'_, PyAny>> =
                if let Ok(lst) = py_list.downcast::<pyo3::types::PyList>() {
                    lst.iter().collect()
                } else {
                    log_warn!(
                        STRATEGY_LOGGER,
                        "compute_position_sizes() must return an ndarray or list — falling back to position_size_pct for this run"
                    );
                    return Ok(None);
                };

            let sizes: Vec<f64> = items
                .iter()
                .map(|item| Self::extract_size_f64(item).unwrap_or(f64::NAN))
                .collect();

            log_debug!(
                STRATEGY_LOGGER,
                "compute_all_position_sizes: {} ticks, 1 FFI call (vectorized path)",
                sizes.len()
            );

            Ok(Some(sizes))
        })
    }

    fn accepts_position_sizes(&self) -> bool {
        self.accepts_position_sizes
    }

    fn defines_generate_signals(&self) -> bool {
        self.defines_generate_signals
    }

    fn compute_signals_accepts_ohlc(&self) -> bool {
        self.compute_signals_accepts_ohlc
    }

    fn accepts_multi_venue(&self) -> bool {
        self.accepts_multi_venue
    }

    fn declares_pair_trade(&self) -> Option<PairSpec> {
        self.pair_spec.clone()
    }

    fn pair_spec_error(&self) -> Option<String> {
        self.pair_spec_error.clone()
    }

    fn declares_basket_trade(&self) -> Option<BasketSpec> {
        self.basket_spec.clone()
    }

    fn basket_spec_error(&self) -> Option<String> {
        self.basket_spec_error.clone()
    }

    /// Cross-exchange strategy support (2026-07-22): bulk-vectorized signal
    /// generation from N declared venues at once. Builds a Python dict keyed
    /// by `"{exchange}:{symbol}"` (a single hashable string key -- pyo3
    /// dicts need a hashable Python key type, and a 2-tuple works too but a
    /// string is simpler on the Python side to read: `venues["kraken:BTC-USD"]`),
    /// each value itself a dict of `{"prices":..., "volumes":..., "timestamps":...,
    /// "opens":..., "highs":..., "lows":...}` numpy arrays -- built with the
    /// exact same `build_f64_array`/`build_i64_array` zero-copy helpers
    /// `compute_all_signals` above already uses, just once per venue instead
    /// of once total. Calls the user's `compute_signals_multi_venue(venues)`,
    /// same tolist()-based extraction as the single-venue path.
    async fn compute_all_signals_multi_venue(
        &mut self,
        venues: &std::collections::HashMap<(String, String), VenueSeries>,
    ) -> StrategyResult<Option<Vec<i8>>> {
        if !self.initialized || !self.accepts_multi_venue {
            return Ok(None);
        }

        let lock = self.py_strategy.lock().await;
        let py_strategy = lock.as_ref().ok_or(StrategyError::NotInitialized)?;
        let timeout = Duration::from_secs(self.timeout_secs);

        // Owned copies of each venue's arrays so they can move into the GIL closure.
        let venues_owned: Vec<(String, VenueSeries)> = venues
            .iter()
            .map(|((symbol, exchange), series)| (format!("{}:{}", exchange, symbol), series.clone()))
            .collect();

        Python::with_gil(|py| {
            let np_frombuffer = self.cached_np_frombuffer.as_ref().map(|p| p.clone_ref(py));
            let np_float64   = self.cached_np_float64.as_ref().map(|p| p.clone_ref(py));
            let np_int64     = self.cached_np_int64.as_ref().map(|p| p.clone_ref(py));

            let strategy = py_strategy.bind(py);
            if !strategy.hasattr("compute_signals_multi_venue").unwrap_or(false) {
                return Ok(None);
            }

            let arm_code = format!(
                "_wd_deadline[0] = __import__('time').time() + {}",
                timeout.as_secs_f64()
            );
            let _ = py.run_bound(&arm_code, None, None);

            let build_f64_array = |data: &[f64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(float64)) = (np_frombuffer.as_ref(), np_float64.as_ref()) {
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f64>())
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, float64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };
            let build_i64_array = |data: &[i64]| -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> {
                if let (Some(frombuffer), Some(int64)) = (np_frombuffer.as_ref(), np_int64.as_ref()) {
                    let raw: &[u8] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<i64>())
                    };
                    let bytes = pyo3::types::PyBytes::new_bound(py, raw);
                    let arr = frombuffer.bind(py).call1((bytes, int64.bind(py)))?;
                    arr.call_method0("copy")
                } else {
                    let py_list = pyo3::types::PyList::new_bound(py, data.iter().copied());
                    Ok(py_list.into_any())
                }
            };

            let venues_dict = pyo3::types::PyDict::new_bound(py);
            for (key, series) in &venues_owned {
                let venue_dict = pyo3::types::PyDict::new_bound(py);
                let prices_arr = build_f64_array(&series.prices).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue prices array build error ({}): {}", key, e))
                })?;
                let volumes_arr = build_f64_array(&series.volumes).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue volumes array build error ({}): {}", key, e))
                })?;
                let timestamps_arr = build_i64_array(&series.timestamps).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue timestamps array build error ({}): {}", key, e))
                })?;
                let opens_arr = build_f64_array(&series.opens).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue opens array build error ({}): {}", key, e))
                })?;
                let highs_arr = build_f64_array(&series.highs).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue highs array build error ({}): {}", key, e))
                })?;
                let lows_arr = build_f64_array(&series.lows).map_err(|e| {
                    StrategyError::MarketDataError(format!("multi-venue lows array build error ({}): {}", key, e))
                })?;
                let _ = venue_dict.set_item("prices", prices_arr);
                let _ = venue_dict.set_item("volumes", volumes_arr);
                let _ = venue_dict.set_item("timestamps", timestamps_arr);
                let _ = venue_dict.set_item("opens", opens_arr);
                let _ = venue_dict.set_item("highs", highs_arr);
                let _ = venue_dict.set_item("lows", lows_arr);
                let _ = venues_dict.set_item(key, venue_dict);
            }

            let call_result = strategy.call_method1("compute_signals_multi_venue", (venues_dict,));

            let result = call_result
                .map_err(|e| {
                    let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);
                    let msg = format!("{}", e);
                    if msg.contains("KeyboardInterrupt") {
                        StrategyError::CalculationError(format!(
                            "compute_signals_multi_venue() timed out after {}s",
                            timeout.as_secs()
                        ))
                    } else {
                        StrategyError::CalculationError(format!("compute_signals_multi_venue() error: {}", msg))
                    }
                })?;

            let _ = py.run_bound("_wd_deadline[0] = 0.0", None, None);

            let py_list = result
                .call_method0("tolist")
                .or_else(|_| Ok::<pyo3::Bound<'_, PyAny>, StrategyError>(result.clone()))
                .map_err(|e: StrategyError| StrategyError::CalculationError(
                    format!("compute_signals_multi_venue(): failed to convert output: {}", e)
                ))?;

            let items: Vec<pyo3::Bound<'_, PyAny>> =
                if let Ok(lst) = py_list.downcast::<pyo3::types::PyList>() {
                    lst.iter().collect()
                } else {
                    return Err(StrategyError::CalculationError(
                        "compute_signals_multi_venue() must return an ndarray or list of int8".to_string()
                    ));
                };

            let mut sigs = Vec::with_capacity(items.len());
            for item in &items {
                sigs.push(Self::extract_signal_i8(item));
            }

            log_debug!(
                STRATEGY_LOGGER,
                "compute_all_signals_multi_venue: {} venues, {} ticks, 1 FFI call",
                venues_owned.len(),
                sigs.len()
            );

            Ok(Some(sigs))
        })
    }

    async fn compute_all_features(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[i64],
    ) -> StrategyResult<Option<HashMap<String, Vec<f64>>>> {
        if !compute_features_enabled() {
            return Ok(None);
        }
        if !self.initialized {
            return Ok(None);
        }

        let lock = self.py_strategy.lock().await;
        let py_strategy = match lock.as_ref() {
            Some(p) => p,
            None => return Ok(None),
        };

        let prices_vec = prices.to_vec();
        let volumes_vec = volumes.to_vec();
        let timestamps_vec = timestamps.to_vec();
        let expected_len = prices_vec.len();

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);

            // Fast-path: skip entirely if the strategy didn't define this
            // optional hook — never error, this is diagnostic-only.
            if !strategy.hasattr("compute_features").unwrap_or(false) {
                return Ok(None);
            }

            let prices_arr = PyList::new_bound(py, prices_vec.iter().copied());
            let volumes_arr = PyList::new_bound(py, volumes_vec.iter().copied());
            let timestamps_arr = PyList::new_bound(py, timestamps_vec.iter().copied());

            let call_result = strategy.call_method1(
                "compute_features",
                (prices_arr, volumes_arr, timestamps_arr),
            );

            let result = match call_result {
                Ok(r) => r,
                Err(e) => {
                    return Err(StrategyError::CalculationError(format!(
                        "compute_features() error: {}", e
                    )));
                }
            };

            let dict = match result.downcast::<PyDict>() {
                Ok(d) => d,
                Err(_) => {
                    return Err(StrategyError::CalculationError(
                        "compute_features() must return a dict[str, array]".to_string(),
                    ));
                }
            };

            // Reserved names avoid ambiguity with the engine's own OHLCV/signal
            // columns downstream.
            const RESERVED_KEYS: &[&str] = &[
                "price", "prices", "volume", "volumes", "timestamp", "timestamps",
                "signal", "signals", "open", "high", "low", "close",
            ];

            let mut features: HashMap<String, Vec<f64>> = HashMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = match key.extract() {
                    Ok(k) => k,
                    Err(_) => continue,
                };
                if RESERVED_KEYS.contains(&key_str.to_lowercase().as_str()) {
                    continue;
                }

                let py_list = match value.call_method0("tolist") {
                    Ok(l) => l,
                    Err(_) => value.clone(),
                };
                let items: Vec<pyo3::Bound<'_, PyAny>> = match py_list.downcast::<PyList>() {
                    Ok(lst) => lst.iter().collect(),
                    Err(_) => continue, // not an array-like — skip this key, not the whole call
                };
                if items.len() != expected_len {
                    continue; // misaligned length — drop this feature, don't fail the call
                }

                let mut values = Vec::with_capacity(items.len());
                let mut all_numeric = true;
                for item in &items {
                    match item.extract::<f64>() {
                        Ok(v) => values.push(v),
                        Err(_) => {
                            all_numeric = false;
                            break;
                        }
                    }
                }
                if all_numeric {
                    features.insert(key_str, values);
                }
            }

            if features.is_empty() {
                Ok(None)
            } else {
                Ok(Some(features))
            }
        })
    }

    async fn update_state(
        &mut self,
        market_data: &dataloader::MarketData,
        context: &StrategyContext,
    ) -> StrategyResult<()> {
        if !self.initialized {
            return Ok(());
        }

        let lock = self.py_strategy.lock().await;
        if let Some(py_strategy) = lock.as_ref() {
            Python::with_gil(|py| {
                let strategy = py_strategy.bind(py);
                let data_dict = Self::market_data_to_py(py, market_data).map_err(|e| {
                    StrategyError::MarketDataError(format!("Data conversion error: {}", e))
                })?;
                let context_dict = Self::context_to_py(
                    py, context,
                    self.cached_np_frombuffer.as_ref(),
                    self.cached_np_float64.as_ref(),
                    self.cached_np_int64.as_ref(),
                    &mut self.prices_buf,
                    &mut self.volumes_buf,
                    &mut self.timestamps_buf,
                ).map_err(|e| {
                    StrategyError::MarketDataError(format!("Context conversion error: {}", e))
                })?;

                let _ = strategy.call_method1("update_state", (data_dict, context_dict));
                Ok(())
            })?;
        }
        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        self.parameters.clone()
    }

    async fn update_parameters(
        &mut self,
        parameters: HashMap<String, ParameterValue>,
    ) -> StrategyResult<()> {
        self.parameters = parameters.clone();

        let lock = self.py_strategy.lock().await;
        if let Some(py_strategy) = lock.as_ref() {
            Python::with_gil(|py| {
                let strategy = py_strategy.bind(py);
                let params_dict = Self::params_to_py(py, &parameters).map_err(|e| {
                    StrategyError::ConfigurationError(format!(
                        "Parameter conversion error: {}",
                        e
                    ))
                })?;

                strategy
                    .call_method1("update_parameters", (params_dict,))
                    .map_err(|e| {
                        StrategyError::ConfigurationError(format!(
                            "Python update_parameters() error: {}",
                            e
                        ))
                    })?;

                Ok::<(), StrategyError>(())
            })?;
        }
        Ok(())
    }

    fn get_risk_limits(&self) -> RiskLimits {
        // Default risk limits — Python strategy can override via get_risk_limits()
        // but we return safe defaults if the Python call fails
        RiskLimits::default()
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn required_data_window(&self) -> usize {
        // Call Python strategy's required_data_window() if available
        // Use try_lock() instead of blocking_lock() to avoid panicking
        // when called from within a tokio async runtime context.
        let Ok(lock) = self.py_strategy.try_lock() else {
            return 100;
        };
        if let Some(py_strategy) = lock.as_ref() {
            if let Some(window) = Python::with_gil(|py| -> Option<usize> {
                let strategy = py_strategy.bind(py);
                strategy.call_method0("required_data_window")
                    .ok()?
                    .extract::<usize>()
                    .ok()
            }) {
                return window;
            }
        }
        // Fallback to safe default if Python call fails
        100
    }

    fn data_requirements(&self) -> DataRequirements {
        // Call into Python data_requirements() if available, fall back to defaults
        let Ok(lock) = self.py_strategy.try_lock() else {
            return DataRequirements::default();
        };
        if let Some(py_strategy) = lock.as_ref() {
            let result = Python::with_gil(|py| -> Option<DataRequirements> {
                let strategy = py_strategy.bind(py);
                let dr = strategy.call_method0("data_requirements").ok()?;
                let dr_dict = dr.call_method0("to_dict").ok()?;
                let d = dr_dict.downcast::<PyDict>().ok()?;

                let get_bool = |key: &str, default: bool| -> bool {
                    d.get_item(key).ok().flatten()
                        .and_then(|v| v.extract::<bool>().ok())
                        .unwrap_or(default)
                };
                let get_i64 = |key: &str, default: i64| -> i64 {
                    d.get_item(key).ok().flatten()
                        .and_then(|v| v.extract::<i64>().ok())
                        .unwrap_or(default)
                };
                let get_str_opt = |key: &str| -> Option<String> {
                    d.get_item(key).ok().flatten()
                        .and_then(|v| v.extract::<String>().ok())
                };

                let category_str = get_str_opt("category").unwrap_or_default();
                let category = match category_str.as_str() {
                    "MarketMaking" => StrategyCategory::MarketMaking,
                    "Momentum" => StrategyCategory::Momentum,
                    "Arbitrage" => StrategyCategory::Arbitrage,
                    "StatisticalArbitrage" => StrategyCategory::StatisticalArbitrage,
                    "Options" => StrategyCategory::Options,
                    "Futures" => StrategyCategory::Futures,
                    "Volatility" => StrategyCategory::Volatility,
                    "Spread" => StrategyCategory::Spread,
                    _ => StrategyCategory::Custom,
                };

                Some(DataRequirements {
                    category,
                    needs_order_flow: get_bool("needs_order_flow", false),
                    needs_trades: get_bool("needs_trades", true),
                    needs_orderbook_snapshots: get_bool("needs_orderbook_snapshots", false),
                    needs_candles: get_bool("needs_candles", false),
                    lookback_window: get_i64("lookback_window", 100) as usize,
                    needs_depth: get_bool("needs_depth", false),
                    max_data_age_secs: d.get_item("max_data_age_secs").ok().flatten()
                        .and_then(|v| v.extract::<u64>().ok()),
                    needs_iv_surface: get_bool("needs_iv_surface", false),
                    needs_futures_curve: get_bool("needs_futures_curve", false),
                    needs_funding_rates: get_bool("needs_funding_rates", false),
                })
            });

            if let Some(dr) = result {
                return dr;
            }
        }

        // Fallback to defaults
        DataRequirements {
            category: StrategyCategory::Custom,
            needs_order_flow: false,
            needs_trades: true,
            needs_orderbook_snapshots: false,
            needs_candles: false,
            lookback_window: 100,
            needs_depth: false,
            max_data_age_secs: None,
            needs_iv_surface: false,
            needs_futures_curve: false,
            needs_funding_rates: false,
        }
    }

    fn get_metrics(&self) -> Option<StrategyMetrics> {
        None
    }

    async fn warmup(
        &mut self,
        historical_data: &[dataloader::MarketData],
        context: &StrategyContext,
    ) -> StrategyResult<()> {
        for data in historical_data {
            self.update_state(data, context).await?;
        }
        Ok(())
    }

    async fn reset(&mut self) -> StrategyResult<()> {
        let lock = self.py_strategy.lock().await;
        if let Some(py_strategy) = lock.as_ref() {
            Python::with_gil(|py| {
                let strategy = py_strategy.bind(py);
                let _ = strategy.call_method0("reset");
            });
        }
        Ok(())
    }

    fn validate_config(
        &self,
        _parameters: &HashMap<String, ParameterValue>,
    ) -> StrategyResult<()> {
        // Validation happens at source code level via validate_source()
        Ok(())
    }

    async fn on_expiry(
        &mut self,
        expired: &DerivativeMetadata,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let lock = self.py_strategy.lock().await;
        let Some(py_strategy) = lock.as_ref() else {
            return Ok(vec![]);
        };

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);
            let inst_dict = PyDict::new_bound(py);
            inst_dict
                .set_item("symbol", &expired.symbol)
                .map_err(|e| StrategyError::MarketDataError(format!("set symbol error: {e}")))?;
            inst_dict
                .set_item("underlying", &expired.underlying)
                .map_err(|e| StrategyError::MarketDataError(format!("set underlying error: {e}")))?;
            inst_dict
                .set_item("exchange", &expired.exchange)
                .map_err(|e| StrategyError::MarketDataError(format!("set exchange error: {e}")))?;
            let kind_str = match &expired.instrument_kind {
                InstrumentKind::Call { .. } => "Call",
                InstrumentKind::Put { .. } => "Put",
                InstrumentKind::Future { .. } => "Future",
                InstrumentKind::Perpetual => "Perpetual",
                InstrumentKind::Spot => "Spot",
            };
            inst_dict
                .set_item("instrument_kind", kind_str)
                .map_err(|e| StrategyError::MarketDataError(format!("set kind error: {e}")))?;
            if let Some(s) = expired.instrument_kind.strike() {
                inst_dict
                    .set_item("strike", s)
                    .map_err(|e| StrategyError::MarketDataError(format!("set strike error: {e}")))?;
            }
            if let Some(e) = expired.instrument_kind.expiry() {
                inst_dict
                    .set_item("expiry", e.to_rfc3339())
                    .map_err(|e2| StrategyError::MarketDataError(format!("set expiry error: {e2}")))?;
            }

            let ctx_dict = Self::context_to_py(
                py,
                context,
                self.cached_np_frombuffer.as_ref(),
                self.cached_np_float64.as_ref(),
                self.cached_np_int64.as_ref(),
                &mut self.prices_buf,
                &mut self.volumes_buf,
                &mut self.timestamps_buf,
            )
            .map_err(|e| StrategyError::MarketDataError(format!("context conversion error: {e}")))?;
            let ctx_obj = Self::dict_to_strategy_context(py, &ctx_dict)?;

            let py_signals = match strategy.call_method1("on_expiry", (inst_dict, ctx_obj)) {
                Ok(v) => v,
                Err(_) => return Ok(vec![]),
            };

            let list = py_signals
                .downcast::<PyList>()
                .map_err(|e| StrategyError::CalculationError(format!("on_expiry must return list: {e}")))?;
            let mut signals = Vec::new();
            for item in list.iter() {
                if let Ok(d) = item.call_method0("to_dict") {
                    if let Ok(sig_dict) = d.downcast::<PyDict>() {
                        if let Ok(sig) = Self::py_signal_to_rust(py, sig_dict) {
                            signals.push(sig);
                        }
                    }
                }
            }
            Ok(signals)
        })
    }

    async fn on_funding_rate_change(
        &mut self,
        symbol: &str,
        rate: &FundingRate,
        context: &StrategyContext,
    ) -> StrategyResult<Vec<Signal>> {
        let lock = self.py_strategy.lock().await;
        let Some(py_strategy) = lock.as_ref() else {
            return Ok(vec![]);
        };

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);
            let ctx_dict = Self::context_to_py(
                py,
                context,
                self.cached_np_frombuffer.as_ref(),
                self.cached_np_float64.as_ref(),
                self.cached_np_int64.as_ref(),
                &mut self.prices_buf,
                &mut self.volumes_buf,
                &mut self.timestamps_buf,
            )
            .map_err(|e| StrategyError::MarketDataError(format!("context conversion error: {e}")))?;
            let ctx_obj = Self::dict_to_strategy_context(py, &ctx_dict)?;

            // Python signature: on_funding_rate_change(symbol, old_rate, new_rate, context)
            // We don't track old_rate on the Rust side, so pass 0.0
            let result = strategy.call_method1(
                "on_funding_rate_change",
                (symbol, 0.0_f64, rate.rate, ctx_obj),
            );
            match result {
                Ok(py_signals) => {
                    let list = py_signals.downcast::<PyList>()
                        .map_err(|e| StrategyError::CalculationError(format!("on_funding_rate_change must return list: {e}")))?;
                    let mut signals = Vec::new();
                    for item in list.iter() {
                        let dict_result = item.call_method0("to_dict");
                        if let Ok(d) = dict_result {
                            if let Ok(sig_dict) = d.downcast::<PyDict>() {
                                if let Ok(sig) = Self::py_signal_to_rust(py, sig_dict) {
                                    signals.push(sig);
                                }
                            }
                        }
                    }
                    Ok(signals)
                }
                Err(_) => Ok(vec![]),
            }
        })
    }

    fn required_instruments(&self) -> Vec<InstrumentKind> {
        let Ok(lock) = self.py_strategy.try_lock() else {
            return vec![];
        };
        let Some(py_strategy) = lock.as_ref() else {
            return vec![];
        };

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);
            let Ok(items) = strategy.call_method0("required_instruments") else {
                return vec![];
            };
            let Ok(list) = items.downcast::<PyList>() else {
                return vec![];
            };

            let mut out = Vec::new();
            for item in list.iter() {
                let Some(d) = item
                    .call_method0("to_dict")
                    .ok()
                    .and_then(|v| v.downcast::<PyDict>().ok().map(|x| x.clone()))
                else {
                    continue;
                };

                let kind = d
                    .get_item("instrument_kind")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default();

                let strike = d
                    .get_item("strike")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<f64>().ok());

                let expiry = d
                    .get_item("expiry")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok());

                out.push(Self::str_to_instrument_kind(&kind, strike, expiry));
            }
            out
        })
    }

    fn target_net_delta(&self) -> Option<f64> {
        let Ok(lock) = self.py_strategy.try_lock() else {
            return None;
        };
        let py_strategy = lock.as_ref()?;

        Python::with_gil(|py| {
            let strategy = py_strategy.bind(py);
            let result = strategy.call_method0("target_net_delta").ok()?;
            result.extract::<Option<f64>>().ok().flatten()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sdk_injection_failure_diagnostics_never_panics() {
        // Smoke test: this only ever runs on inject_sdk's error path, so a
        // panic inside it would replace a recoverable failure with a hard
        // crash. Every lookup inside must degrade to a placeholder string
        // instead, even called here with a healthy interpreter where the
        // lookups should all genuinely succeed.
        Python::with_gil(|py| {
            log_sdk_injection_failure_diagnostics(py, "synthetic test error");
        });
    }

    const TEST_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "TestStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        sig = np.zeros(len(prices), dtype=np.int8)
        sig[prices > 100] = 1
        return sig
"#;

    #[test]
    fn test_validate_source_valid() {
        // Note: This test requires Python to be available
        // Skip in CI if Python is not installed
        let result = PythonStrategy::validate_source(TEST_STRATEGY);
        // Allow both success (Python available) and specific error (Python not available)
        match result {
            Ok(()) => {}
            Err(e) if e.contains("Python") || e.contains("interpreter") => {}
            Err(e) => panic!("Unexpected validation error: {}", e),
        }
    }

    #[test]
    fn test_validate_source_accepts_ohlc_signature() {
        // A compute_signals declaring the optional opens/highs/lows OHLC params
        // must validate exactly like the 3-arg form (backward compatible).
        const OHLC_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "OhlcStrategy"

    def compute_signals(self, prices, volumes, timestamps, opens=None, highs=None, lows=None):
        sig = np.zeros(len(prices), dtype=np.int8)
        if highs is not None:
            sig[highs > 100] = 1
        return sig
"#;
        let result = PythonStrategy::validate_source(OHLC_STRATEGY);
        match result {
            Ok(()) => {}
            Err(e) if e.contains("Python") || e.contains("interpreter") => {}
            Err(e) => panic!("OHLC signature should validate, got: {}", e),
        }
    }

    #[test]
    fn test_validate_source_no_class() {
        let bad_code = "x = 42\n";
        let result = PythonStrategy::validate_source(bad_code);
        match result {
            Err(e) => assert!(
                e.contains("Strategy") || e.contains("class") || e.contains("Python"),
                "Error should mention missing Strategy class: {}",
                e
            ),
            Ok(()) => {} // Python not available, skip
        }
    }

    #[test]
    fn test_python_strategy_debug() {
        let strategy = PythonStrategy::new("# test".to_string());
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("PythonStrategy"));
        assert!(debug.contains("source_len"));
    }

    // ── Builder pattern ─────────────────────────────────────────────

    #[test]
    fn test_python_strategy_new_defaults() {
        let strategy = PythonStrategy::new("class Strategy: pass".to_string());
        assert!(!strategy.initialized);
        assert_eq!(strategy.cached_name, "PythonStrategy");
        assert_eq!(strategy.cached_version, "1.0.0");
        assert_eq!(strategy.timeout_secs, 30);
    }

    #[test]
    fn test_python_strategy_with_timeout() {
        let strategy = PythonStrategy::new("# code".to_string())
            .with_timeout(60);
        assert_eq!(strategy.timeout_secs, 60);
    }

    #[test]
    fn test_python_strategy_with_memory_limit() {
        let strategy = PythonStrategy::new("# code".to_string())
            .with_memory_limit(1024 * 1024 * 512);
        assert_eq!(strategy.memory_limit_bytes, 1024 * 1024 * 512);
    }

    #[test]
    fn test_python_strategy_chained_builders() {
        let strategy = PythonStrategy::new("# chain".to_string())
            .with_timeout(120)
            .with_memory_limit(1_000_000);
        assert_eq!(strategy.timeout_secs, 120);
        assert_eq!(strategy.memory_limit_bytes, 1_000_000);
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn test_python_strategy_template_non_empty() {
        assert!(!PYTHON_STRATEGY_TEMPLATE.is_empty());
        assert!(PYTHON_STRATEGY_TEMPLATE.contains("BaseStrategy"));
    }

    #[test]
    fn test_sandbox_import_hook_non_empty() {
        assert!(!SANDBOX_IMPORT_HOOK.is_empty());
    }

    #[test]
    fn test_python_strategy_template_contains_required_methods() {
        assert!(PYTHON_STRATEGY_TEMPLATE.contains("compute_signals"));
        assert!(PYTHON_STRATEGY_TEMPLATE.contains("name"));
    }

    #[test]
    fn test_python_strategy_source_code_preserved() {
        let code = "class Strategy(BaseStrategy):\n    def name(self): return 'Test'\n";
        let strategy = PythonStrategy::new(code.to_string());
        assert_eq!(strategy.source_code, code);
    }

    #[test]
    fn test_python_strategy_new_not_initialized() {
        let strategy = PythonStrategy::new("# any code".to_string());
        assert!(!strategy.initialized);
        assert!(strategy.parameters.is_empty());
    }

    #[test]
    fn compute_features_enabled_defaults_to_true() {
        std::env::remove_var("ENABLE_COMPUTE_FEATURES");
        assert!(compute_features_enabled());
    }

    #[test]
    fn compute_features_enabled_respects_explicit_disable() {
        std::env::set_var("ENABLE_COMPUTE_FEATURES", "false");
        assert!(!compute_features_enabled());
        std::env::set_var("ENABLE_COMPUTE_FEATURES", "0");
        assert!(!compute_features_enabled());
        std::env::remove_var("ENABLE_COMPUTE_FEATURES");
    }

    #[test]
    fn compute_features_enabled_true_for_any_other_value() {
        std::env::set_var("ENABLE_COMPUTE_FEATURES", "true");
        assert!(compute_features_enabled());
        std::env::set_var("ENABLE_COMPUTE_FEATURES", "1");
        assert!(compute_features_enabled());
        std::env::remove_var("ENABLE_COMPUTE_FEATURES");
    }

    /// Regression test for job `8154ec1a`: a strategy that declares a
    /// `parameter_space()` key and reads it via direct dict indexing
    /// (`self.params['lookback_period']`, not `.get(key, default)`) must not
    /// KeyError when initialized with an empty `parameters` dict — which is
    /// exactly what Stage 0/Stage 1's quick backtest passes, since real
    /// values only arrive once GA runs. The SDK's `initialize()` must seed
    /// `self.params` from `parameter_space()`'s declared defaults first.
    #[cfg(feature = "python")]
    #[tokio::test]
    async fn initialize_seeds_declared_parameter_defaults_for_direct_indexing() {
        use crate::Strategy;

        const STRATEGY_WITH_DIRECT_INDEXING: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "DirectIndexStrategy"

    def parameter_space(self):
        return {
            "lookback_period": {"type": "int", "min": 5, "max": 100, "default": 20},
        }

    def compute_signals(self, prices, volumes, timestamps):
        # Direct indexing, NOT .get() -- relies on parameter_space()'s
        # declared default being pre-seeded rather than defending itself.
        lookback = self.params['lookback_period']
        sig = np.zeros(len(prices), dtype=np.int8)
        if len(prices) > lookback:
            sig[lookback:] = 1
        return sig
"#;

        let mut strategy = PythonStrategy::new(STRATEGY_WITH_DIRECT_INDEXING.to_string());
        // Empty parameters -- exactly what Stage 0/Stage 1 pass before GA
        // ever injects real values.
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(
            result.is_ok(),
            "initialize() with empty parameters should succeed by seeding \
             parameter_space() defaults, got: {:?}",
            result.err()
        );

        let signals = strategy
            .compute_all_signals(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                &[1.0; 10],
                &[0i64; 10],
                &[],
                &[],
                &[],
            )
            .await;
        assert!(
            signals.is_ok(),
            "compute_signals() should not KeyError on a declared-but-not-explicitly-passed \
             parameter, got: {:?}",
            signals.err()
        );
    }

    /// Regression test for production jobs `9e1ce0e1`/`5db4828c` (2026-07-24):
    /// a strategy whose `compute_signals()` returns a **float64** numpy array
    /// (the default dtype for `np.zeros(n)` when `dtype=np.int8` isn't
    /// explicitly passed -- nothing forces it, only the SDK docstring
    /// examples happen to include it) produced a "completed" backtest with
    /// 0 trades and no error at all, even for a deliberately unconditional
    /// always-long test strategy. Root cause: `.tolist()` on a float64 array
    /// yields Python `float` objects, and PyO3's `extract::<i64>()` requires
    /// an actual Python `int` -- it does not implicitly truncate a float, so
    /// the old code's bare `item.extract().unwrap_or(0)` silently wrote 0
    /// (HOLD) for every tick regardless of what the strategy intended.
    #[cfg(feature = "python")]
    #[tokio::test]
    async fn compute_all_signals_handles_float64_array_not_just_int8() {
        use crate::Strategy;

        const FLOAT_SIGNAL_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "AlwaysLongFloat"

    def compute_signals(self, prices, volumes, timestamps):
        # No dtype specified -- np.zeros() defaults to float64, not int8.
        # This is the exact shape a real production job hit.
        signals = np.zeros(len(prices))
        if len(prices) > 5:
            signals[5:] = 1.0
        return signals
"#;

        let mut strategy = PythonStrategy::new(FLOAT_SIGNAL_STRATEGY.to_string());
        strategy.initialize(std::collections::HashMap::new()).await.unwrap();

        let signals = strategy
            .compute_all_signals(
                &[1.0; 10],
                &[1.0; 10],
                &[0i64; 10],
                &[],
                &[],
                &[],
            )
            .await
            .expect("compute_all_signals should not error on a float64 signal array")
            .expect("compute_signals is implemented -- should return Some(signals)");

        assert_eq!(signals.len(), 10);
        assert_eq!(&signals[..5], &[0, 0, 0, 0, 0], "warmup bars should stay HOLD");
        assert_eq!(
            &signals[5..], &[1, 1, 1, 1, 1],
            "float64 1.0 entries must be read as BUY (1), not silently dropped to HOLD (0) -- \
             this is the exact bug that made a real production strategy trade 0 times",
        );
    }

    // ── compute_position_sizes() / accepts_position_sizes() ─────────

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn accepts_position_sizes_reflects_method_presence() {
        use crate::Strategy;

        const WITH_SIZING: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WithSizing"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        return np.full(len(prices), 0.1)
"#;
        const WITHOUT_SIZING: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WithoutSizing"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;

        let mut with_sizing = PythonStrategy::new(WITH_SIZING.to_string());
        with_sizing.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(with_sizing.accepts_position_sizes(), "should detect compute_position_sizes when defined");

        let mut without_sizing = PythonStrategy::new(WITHOUT_SIZING.to_string());
        without_sizing.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(!without_sizing.accepts_position_sizes(), "should stay false when compute_position_sizes is absent");
    }

    // ── generate_signals() / defines_generate_signals() ─────────

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn defines_generate_signals_reflects_method_presence() {
        use crate::Strategy;

        const WITH_GENERATE_SIGNALS: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WithGenerateSignals"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)

    def generate_signals(self, data, context):
        return []
"#;
        const WITHOUT_GENERATE_SIGNALS: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WithoutGenerateSignals"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;

        let mut with_generate = PythonStrategy::new(WITH_GENERATE_SIGNALS.to_string());
        with_generate.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(with_generate.defines_generate_signals(), "should detect generate_signals when defined");

        let mut without_generate = PythonStrategy::new(WITHOUT_GENERATE_SIGNALS.to_string());
        without_generate.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(!without_generate.defines_generate_signals(), "should stay false when generate_signals is absent");
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn compute_all_position_sizes_returns_per_tick_fractions() {
        use crate::Strategy;

        const SIZING_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "SizingStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.ones(len(prices), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        # Alternate a small and a large fraction across ticks.
        sizes = np.zeros(len(prices))
        sizes[::2] = 0.05
        sizes[1::2] = 0.5
        return sizes
"#;

        let mut strategy = PythonStrategy::new(SIZING_STRATEGY.to_string());
        strategy.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(strategy.accepts_position_sizes());

        let sizes = strategy
            .compute_all_position_sizes(&[1.0; 6], &[1.0; 6], &[0i64; 6])
            .await
            .expect("compute_all_position_sizes should not error")
            .expect("compute_position_sizes is implemented -- should return Some(sizes)");

        assert_eq!(sizes.len(), 6);
        assert_eq!(sizes[0], 0.05);
        assert_eq!(sizes[1], 0.5);
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn compute_all_position_sizes_falls_back_gracefully_on_error() {
        use crate::Strategy;

        const BROKEN_SIZING_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "BrokenSizingStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.ones(len(prices), dtype=np.int8)

    def compute_position_sizes(self, prices, volumes, timestamps):
        raise ValueError("deliberately broken")
"#;

        let mut strategy = PythonStrategy::new(BROKEN_SIZING_STRATEGY.to_string());
        strategy.initialize(std::collections::HashMap::new()).await.unwrap();
        assert!(strategy.accepts_position_sizes());

        let sizes = strategy
            .compute_all_position_sizes(&[1.0; 4], &[1.0; 4], &[0i64; 4])
            .await
            .expect("compute_all_position_sizes must never fail the backtest on a Python exception");
        assert!(sizes.is_none(), "an exception in compute_position_sizes must fall back to None, not propagate");
    }

    // ── declared_max_leverage() ─────────────────────────────────────

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn declared_max_leverage_reflects_strategy_override() {
        use crate::Strategy;

        const LEVERAGED_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy, RiskLimits

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "LeveragedStrategy"

    def get_risk_limits(self):
        return RiskLimits(max_leverage=3.0)

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(LEVERAGED_STRATEGY.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(
            (strategy.declared_max_leverage() - 3.0).abs() < 1e-9,
            "expected declared_max_leverage() == 3.0, got {}",
            strategy.declared_max_leverage()
        );
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn declared_max_leverage_defaults_to_no_cap_without_override() {
        use crate::Strategy;

        // Regression guard: the SDK's own RiskLimits.max_leverage defaults to
        // `None`, NOT a numeric 1.0 -- if this ever regressed to a numeric
        // default, every strategy that predates this feature (none of which
        // override get_risk_limits()) would suddenly get silently clamped to
        // 1x leverage regardless of what the job itself requested.
        const PLAIN_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "PlainStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(PLAIN_STRATEGY.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(
            strategy.declared_max_leverage().is_infinite(),
            "a strategy that never overrides get_risk_limits() must declare NO additional \
             cap (infinity), not a numeric default -- got {}",
            strategy.declared_max_leverage()
        );
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn declared_max_leverage_respects_an_explicit_one_x_cap() {
        use crate::Strategy;

        // Distinct from the "never overrode it" case above: an EXPLICIT
        // max_leverage=1.0 is a real, deliberate cap and must be honored,
        // not conflated with "unset".
        const EXPLICIT_ONE_X_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy, RiskLimits

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "ExplicitOneXStrategy"

    def get_risk_limits(self):
        return RiskLimits(max_leverage=1.0)

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(EXPLICIT_ONE_X_STRATEGY.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(
            (strategy.declared_max_leverage() - 1.0).abs() < 1e-9,
            "an explicit max_leverage=1.0 must be honored as a real cap, got {}",
            strategy.declared_max_leverage()
        );
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn declared_max_leverage_falls_back_to_no_cap_when_override_errors() {
        use crate::Strategy;

        const BROKEN_RISK_LIMITS_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "BrokenRiskLimitsStrategy"

    def get_risk_limits(self):
        raise RuntimeError("deliberately broken for the test")

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(BROKEN_RISK_LIMITS_STRATEGY.to_string());
        // initialize() must not fail just because get_risk_limits() raised --
        // the same defensive posture as every other cached-metadata call.
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed despite a broken get_risk_limits(), got: {:?}", result.err());
        assert!(
            strategy.declared_max_leverage().is_infinite(),
            "a broken get_risk_limits() override should fall back to the safe \
             no-additional-cap default, got {}",
            strategy.declared_max_leverage()
        );
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn pair_spec_error_names_the_real_problem_when_it_has_baskets_legs_shape_instead() {
        use crate::Strategy;

        // Live incident (quant council 2026-07-31): an AI-authored pair_spec()
        // returned basket_spec()'s {legs: [...]} shape instead of the required
        // flat symbol_a/exchange_a/symbol_b/exchange_b keys. Before this fix,
        // declares_pair_trade() returning None was indistinguishable from
        // pair_spec() not being defined at all -- the error message told the
        // user/AI to add a method that already existed. Assert pair_spec_error()
        // now names the actual problem and hints at the basket_spec() mixup.
        const WRONG_SHAPE_PAIR_SPEC: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WrongShapePairSpec"

    def pair_spec(self):
        return {
            "legs": [
                {"exchange": "kraken", "symbol": "ETH-USD", "weight": 1.0},
                {"exchange": "kraken", "symbol": "SOL-USD", "weight": -8.261},
            ],
            "hedge_ratio": 8.261,
        }

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(WRONG_SHAPE_PAIR_SPEC.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(
            strategy.declares_pair_trade().is_none(),
            "a pair_spec() missing symbol_a/exchange_a/symbol_b/exchange_b must not parse into a PairSpec"
        );
        let error = strategy.pair_spec_error().expect("pair_spec_error() must be Some when pair_spec() exists but doesn't parse");
        assert!(error.contains("symbol_a"), "error should name the missing key, got: {}", error);
        assert!(error.contains("basket_spec"), "error should hint at the basket_spec() shape mixup, got: {}", error);
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn basket_spec_error_names_the_real_problem_when_it_has_pairs_flat_shape_instead() {
        use crate::Strategy;

        const WRONG_SHAPE_BASKET_SPEC: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "WrongShapeBasketSpec"

    def basket_spec(self):
        return {
            "symbol_a": "ETH-USD",
            "exchange_a": "kraken",
            "symbol_b": "SOL-USD",
            "exchange_b": "kraken",
            "hedge_ratio": 8.261,
        }

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(WRONG_SHAPE_BASKET_SPEC.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(
            strategy.declares_basket_trade().is_none(),
            "a basket_spec() missing 'legs' must not parse into a BasketSpec"
        );
        let error = strategy.basket_spec_error().expect("basket_spec_error() must be Some when basket_spec() exists but doesn't parse");
        assert!(error.contains("legs"), "error should name the missing key, got: {}", error);
        assert!(error.contains("pair_spec"), "error should hint at the pair_spec() shape mixup, got: {}", error);
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn pair_spec_error_is_none_when_pair_spec_is_correctly_shaped() {
        use crate::Strategy;

        const CORRECT_PAIR_SPEC: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "CorrectPairSpec"

    def pair_spec(self):
        return {
            "symbol_a": "ETH-USD",
            "exchange_a": "kraken",
            "symbol_b": "SOL-USD",
            "exchange_b": "kraken",
            "hedge_ratio_mode": "fixed",
            "hedge_ratio": 8.261,
        }

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut strategy = PythonStrategy::new(CORRECT_PAIR_SPEC.to_string());
        let result = strategy.initialize(std::collections::HashMap::new()).await;
        assert!(result.is_ok(), "initialize() should succeed, got: {:?}", result.err());
        assert!(strategy.declares_pair_trade().is_some(), "a correctly-shaped pair_spec() must parse");
        assert!(strategy.pair_spec_error().is_none(), "no error should be recorded for a valid pair_spec()");
    }

    #[cfg(feature = "python")]
    #[tokio::test]
    async fn sequential_compilations_do_not_leak_a_stale_strategy_class_across_jobs() {
        // Real production incident: PyModule::from_code_bound used the SAME
        // fixed module name ("user_strategy") on every call. The embedded
        // Python interpreter is process-global and persists for this worker's
        // entire lifetime across many sequential jobs (see this file's module
        // doc comment), and CPython's PyImport_AddModuleObject (which
        // PyModule::from_code_bound maps to) returns the EXISTING module
        // object for a name already in sys.modules rather than a fresh one --
        // the same importlib.reload() gotcha where old names the new source
        // doesn't redefine are left over. A job whose code names its class
        // something OTHER than `Strategy` (a perfectly valid choice) would
        // then silently find a PRIOR job's leftover `Strategy` class instead
        // of correctly failing "No 'Strategy' class found".
        //
        // Compile a first strategy that DOES define `class Strategy`, then a
        // second whose source defines no such name at all. If module state
        // leaked, the second `initialize()` would succeed by picking up the
        // first job's stale class. It must instead fail honestly.
        const FIRST_JOB_STRATEGY: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class Strategy(BaseStrategy):
    def name(self) -> str:
        return "FirstJobStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut first = PythonStrategy::new(FIRST_JOB_STRATEGY.to_string());
        let first_result = first.initialize(std::collections::HashMap::new()).await;
        assert!(first_result.is_ok(), "first job's initialize() should succeed, got: {:?}", first_result.err());

        const SECOND_JOB_STRATEGY_NO_STRATEGY_CLASS: &str = r#"
import numpy as np
from trading_platform import BaseStrategy

class SomeDifferentlyNamedStrategy(BaseStrategy):
    def name(self) -> str:
        return "SecondJobStrategy"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let mut second = PythonStrategy::new(SECOND_JOB_STRATEGY_NO_STRATEGY_CLASS.to_string());
        let second_result = second.initialize(std::collections::HashMap::new()).await;
        assert!(
            second_result.is_err(),
            "a second job whose source defines no 'Strategy' class must fail, not silently \
             succeed by reusing the first job's leftover Strategy class"
        );
        let err = format!("{}", second_result.err().unwrap());
        assert!(
            err.contains("Strategy") && (err.contains("class") || err.contains("found")),
            "error should honestly report no Strategy class was found, got: {}",
            err
        );
    }
}
