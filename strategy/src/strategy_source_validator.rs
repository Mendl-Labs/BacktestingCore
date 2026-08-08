//! Pre-submission validation of user-supplied Python strategy source.
//!
//! The dominant real-world failure mode for AI-generated strategies is a
//! *truncated* generation: the model (or the SSE stream) stops mid-method, so
//! the code that lands in the editor is missing its `return signals` (or the
//! whole tail of the entrypoint). That code is often still syntactically valid
//! Python — a function that falls off the end simply returns `None` — so it
//! sails through queueing and only blows up minutes later at
//! "Stage 1: quick backtest failed: compute_signals() must return an ndarray
//! or list of int8".
//!
//! [`validate_strategy_source`] catches this at the API boundary so the user
//! gets an immediate, actionable error ("looks truncated — regenerate") rather
//! than a cryptic late failure that also burns a backtest run.

/// Modules the strategy sandbox refuses to import at runtime.
///
/// MUST stay in sync with `_BLOCKED_MODULES` in
/// `strategy/src/strategies/python_strategy.rs`. Catching a forbidden import
/// here — before the job is queued — turns a late, opaque
/// "Stage 1: quick backtest failed" into an immediate, actionable 400. The
/// dominant trigger is an AI-generated strategy that slips in `import os`
/// despite the prompt forbidding it.
pub const FORBIDDEN_MODULES: &[&str] = &[
    "os", "subprocess", "socket", "ctypes", "shutil",
    "multiprocessing", "threading", "signal", "resource",
    "pty", "fcntl", "termios", "mmap", "tempfile",
    "webbrowser", "http", "urllib", "ftplib", "smtplib",
    "xmlrpc", "asyncio.subprocess",
];

/// Returns the blocked base module when `imported` (a full dotted module name,
/// e.g. `os.path`) is forbidden — exact match or a submodule of a blocked
/// package. Mirrors the sandbox's `name == b or name.startswith(b + '.')`
/// exactly (so `os` blocks `os.path`, and `asyncio.subprocess` blocks only that
/// submodule, never bare `asyncio`).
fn forbidden_base(imported: &str) -> Option<&'static str> {
    FORBIDDEN_MODULES
        .iter()
        .copied()
        .find(|b| imported == *b || imported.starts_with(&format!("{b}.")))
}

/// Convenience predicate used by tests.
#[cfg(test)]
fn is_forbidden_module(imported: &str) -> bool {
    forbidden_base(imported).is_some()
}

/// Validate a Python strategy source string before it is queued.
///
/// Returns `Ok(())` if the source looks like a runnable strategy, or
/// `Err(message)` with a user-facing explanation when it appears truncated,
/// incomplete, syntactically invalid, or imports a sandbox-forbidden module.
///
/// A cheap textual check runs in every build; when compiled with the `python`
/// feature an AST-level check additionally verifies the entrypoint method
/// actually returns a value (the precise signature of a cut-off generation)
/// and catches forbidden imports precisely (aliases, `from X import …`).
pub fn validate_strategy_source(source: &str) -> Result<(), String> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Err("Strategy code is empty.".to_string());
    }

    // Must define one of the two supported entrypoints. A generation that was
    // cut off before reaching the method is caught here even without Python.
    let has_compute = source.contains("def compute_signals");
    let has_generate = source.contains("def generate_signals");
    if !has_compute && !has_generate {
        return Err(
            "Strategy is missing a `generate_signals` (or `compute_signals`) method. \
             The code may have been cut off before it finished generating — \
             try regenerating the strategy."
                .to_string(),
        );
    }

    // Cheap textual forbidden-import scan (runs in every build; the AST path
    // below supersedes it under the `python` feature but this keeps non-python
    // builds — and the fast path — honest).
    check_forbidden_imports_textual(source)?;

    #[cfg(feature = "python")]
    {
        ast_validate(source)?;
    }

    Ok(())
}

/// Line-oriented forbidden-import check. Only inspects lines that are actual
/// `import …` / `from … import …` statements, so a comment or string that
/// merely mentions `os` is never flagged.
fn check_forbidden_imports_textual(source: &str) -> Result<(), String> {
    for raw in source.lines() {
        let line = raw.trim();
        // Full dotted module name (keep the dots so `os.path` matches the `os`
        // base and `asyncio.subprocess` matches only itself). Strip `as`
        // aliases and trailing tokens by splitting on space/comma.
        let module: Option<&str> = if let Some(rest) = line.strip_prefix("import ") {
            rest.split([' ', ',']).next() // `import os.path as p` -> `os.path`
        } else if let Some(rest) = line.strip_prefix("from ") {
            rest.split(' ').next() // `from os.path import join` -> `os.path`
        } else {
            None
        };
        if let Some(m) = module {
            if let Some(base) = forbidden_base(m) {
                return Err(forbidden_import_message(base));
            }
        }
    }
    Ok(())
}

/// User-facing message for a forbidden import — names the module and points at
/// what IS allowed, so the iterate loop can regenerate correctly.
fn forbidden_import_message(module: &str) -> String {
    format!(
        "Strategy imports `{module}`, which is not allowed in the sandboxed \
         execution environment. Strategies may only import \
         `from trading_platform import BaseStrategy` (or MarketMakingStrategy); \
         numpy, pandas, and scipy are pre-imported. Remove the `{module}` import \
         and use pure numpy/pandas computation instead."
    )
}

/// AST-level validation via the embedded Python interpreter. Parses the source
/// (catching syntax errors from truncation) and verifies that every
/// `compute_signals` / `generate_signals` method returns a value — the exact
/// thing a cut-off generation is missing. The user code is parsed, never
/// executed, so this is safe and cheap.
/// The `check(src)` body, kept free of Rust `format!` interpolation so its
/// own Python `{}` placeholders survive verbatim. `FORBIDDEN` and
/// `_FORBIDDEN_MSG` are defined by the injected preamble in [`ast_validate`].
#[cfg(feature = "python")]
const CHECK_BODY: &str = r#"
def check(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        line = getattr(e, 'lineno', '?')
        return ("Strategy code has a Python syntax error on line {}: {}. "
                "The code may have been cut off before it finished generating "
                "— try regenerating the strategy.").format(line, e.msg)

    # Forbidden-import scan — precise (catches `import os as o`, `from os.path
    # import join`). Mirrors the sandbox's `name == b or name.startswith(b+'.')`.
    forbidden = FORBIDDEN
    def _blocked(mod):
        # `mod` is the full dotted name. Match the sandbox exactly: blocked when
        # it equals a listed module or is a submodule of one. Return the listed
        # base so the message names the package the user must drop.
        if mod is None:
            return None
        for b in forbidden:
            if mod == b or mod.startswith(b + '.'):
                return b
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                b = _blocked(alias.name)
                if b:
                    return _FORBIDDEN_MSG.format(mod=b)
        elif isinstance(node, ast.ImportFrom):
            b = _blocked(node.module)
            if b:
                return _FORBIDDEN_MSG.format(mod=b)

    targets = ("compute_signals", "generate_signals")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
            found.append(node)

    if not found:
        return ("Strategy is missing a generate_signals/compute_signals method. "
                "The code may have been cut off before it finished generating "
                "— try regenerating the strategy.")

    # The runtime loads the strategy via `module.getattr("Strategy")` -- it
    # looks up this EXACT class name, not "whichever class defines
    # compute_signals". A strategy whose entrypoint method lives on a
    # descriptively-named class (e.g. `class UsdCadMeanReversion(BaseStrategy)`)
    # passes every other check here, compiles cleanly, and then silently
    # fails at execution time with "No 'Strategy' class found in Python
    # code" -- or, worse, that failure can itself be masked by an earlier
    # gate (e.g. the cost-hurdle pre-check) that never reaches strategy
    # compilation at all, so the job just reports zero trades with no error.
    # Catching the naming mismatch here, before the job is ever queued, turns
    # both of those into one immediate, actionable message.
    top_level_class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if "Strategy" not in top_level_class_names:
        entrypoint_class = None
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                for item in n.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in targets:
                        entrypoint_class = n.name
                        break
                if entrypoint_class:
                    break
        if entrypoint_class:
            return ("Your strategy class is named `{}`, but the runtime looks up the class "
                    "by the exact name `Strategy` and will not find it under any other name. "
                    "Rename `class {}(BaseStrategy):` to `class Strategy(BaseStrategy):` "
                    "(keep everything else the same -- a descriptive name can still live in "
                    "the `name` attribute or a docstring).").format(entrypoint_class, entrypoint_class)

    for fn in found:
        has_value_return = False
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and n.value is not None:
                has_value_return = True
                break
        if not has_value_return:
            return ("Strategy method `{}()` never returns a value (no `return`). "
                    "The code looks truncated — try regenerating the strategy.").format(fn.name)

    return ""
"#;

#[cfg(feature = "python")]
fn ast_validate(source: &str) -> Result<(), String> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyModule};

    // Inject the forbidden-module list and message template from the Rust
    // const so this file is the single source of truth. `{msg_template:?}`
    // emits a properly-quoted, escaped Python string literal. The CHECK_BODY
    // below is NOT format-interpolated, so its own `{}`/`{{}}` stay literal.
    let forbidden_py = FORBIDDEN_MODULES
        .iter()
        .map(|m| format!("'{m}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let msg_template = forbidden_import_message("{mod}");
    let preamble = format!(
        "import ast\nFORBIDDEN = frozenset({{{forbidden_py}}})\n_FORBIDDEN_MSG = {msg_template:?}\n\n"
    );
    let helper = format!("{preamble}{CHECK_BODY}");

    Python::with_gil(|py| {
        let helper = helper.as_str();
        let module = PyModule::from_code_bound(py, helper, "strategy_validator.py", "strategy_validator")
            .map_err(|e| format!("internal strategy validator error: {e}"))?;
        let result = module
            .getattr("check")
            .map_err(|e| format!("internal strategy validator error: {e}"))?
            .call1((source,))
            .map_err(|e| format!("internal strategy validator error: {e}"))?;
        let msg: String = result
            .extract()
            .map_err(|e| format!("internal strategy validator error: {e}"))?;
        if msg.is_empty() {
            Ok(())
        } else {
            Err(msg)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
from trading_platform import BaseStrategy
import numpy as np

class Strategy(BaseStrategy):
    def compute_signals(self, prices, volumes, timestamps, opens=None, highs=None, lows=None):
        n = len(prices)
        signals = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                signals[i] = 1
        return signals
"#;

    // Mirrors the real job 6b3d1da9 failure: method body present but the
    // generation was cut off before `return signals` was written.
    const TRUNCATED_NO_RETURN: &str = r#"
from trading_platform import BaseStrategy
import numpy as np

class Strategy(BaseStrategy):
    def compute_signals(self, prices, volumes, timestamps, opens=None, highs=None, lows=None):
        n = len(prices)
        signals = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                signals[i] = 1
            # Trend break (price below EMA50)
"#;

    #[test]
    fn empty_source_rejected() {
        assert!(validate_strategy_source("   \n  ").is_err());
    }

    #[test]
    fn missing_entrypoint_rejected() {
        let src = "class Strategy:\n    def initialize(self):\n        pass\n";
        let err = validate_strategy_source(src).unwrap_err();
        assert!(err.contains("generate_signals") || err.contains("compute_signals"));
    }

    #[test]
    fn good_strategy_accepted() {
        assert!(validate_strategy_source(GOOD).is_ok());
    }

    // Mirrors two real live-production strategies (USD-CAD z-score mean
    // reversion, TSLA EMA trend) that both compiled, passed every other
    // check, and then either failed silently or were masked by an earlier
    // gate at execution time -- because the runtime loads the class by the
    // exact literal name "Strategy", not "whichever class defines
    // compute_signals".
    #[cfg(feature = "python")]
    #[test]
    fn descriptively_named_class_is_rejected_with_the_real_reason() {
        let src = r#"
from trading_platform import BaseStrategy
import numpy as np

class USDCADMeanReversionZScore(BaseStrategy):
    name = "USDCAD_ZScore_MeanReversion_4H"

    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let err = validate_strategy_source(src).unwrap_err();
        assert!(err.contains("USDCADMeanReversionZScore"), "message should name the actual class found: {err}");
        assert!(err.contains("Strategy"), "message should name the required class name: {err}");
        assert!(err.contains("rename") || err.contains("Rename"), "message should be actionable: {err}");
    }

    #[test]
    fn class_named_strategy_is_still_accepted() {
        let src = r#"
from trading_platform import BaseStrategy
import numpy as np

class Strategy(BaseStrategy):
    def compute_signals(self, prices, volumes, timestamps):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        assert!(validate_strategy_source(src).is_ok());
    }

    // Mirrors the real job c70381ba failure: an AI-generated strategy that
    // slipped in `import os`, blocked by the sandbox and caught here first.
    const IMPORTS_OS: &str = r#"
from trading_platform import BaseStrategy
import numpy as np
import os

class Strategy(BaseStrategy):
    def compute_signals(self, prices, volumes, timestamps, opens=None, highs=None, lows=None):
        return np.zeros(len(prices), dtype=np.int8)
"#;

    #[test]
    fn forbidden_import_os_rejected() {
        let err = validate_strategy_source(IMPORTS_OS).unwrap_err();
        assert!(err.contains("os"), "message should name the module: {err}");
        assert!(err.contains("not allowed"), "message should be actionable: {err}");
    }

    #[test]
    fn forbidden_from_import_rejected() {
        let src = "from subprocess import run\ndef compute_signals(self, p):\n    return p\n";
        assert!(validate_strategy_source(src).is_err());
    }

    #[test]
    fn forbidden_submodule_rejected() {
        // `os.path` is blocked because `os` is (prefix match).
        let src = "import os.path\ndef compute_signals(self, p):\n    return p\n";
        assert!(validate_strategy_source(src).is_err());
    }

    #[test]
    fn allowed_imports_accepted() {
        // numpy/pandas/scipy and the trading_platform base must not be flagged.
        assert!(is_forbidden_module("numpy") == false);
        assert!(is_forbidden_module("pandas") == false);
        assert!(is_forbidden_module("trading_platform") == false);
        assert!(is_forbidden_module("os"));
        assert!(is_forbidden_module("os.path"));
    }

    #[test]
    fn os_in_a_comment_is_not_flagged() {
        // The textual scan only inspects real import lines, never comments.
        let src = "# we do not import os here\nimport numpy as np\ndef compute_signals(self, p):\n    return p\n";
        assert!(validate_strategy_source(src).is_ok());
    }

    // The AST path (python feature) additionally catches aliased imports the
    // textual scan's split handles too, but this pins the precise behaviour.
    #[cfg(feature = "python")]
    #[test]
    fn aliased_forbidden_import_rejected_by_ast() {
        let src = r#"
from trading_platform import BaseStrategy
import numpy as np
import socket as sk

class Strategy(BaseStrategy):
    def compute_signals(self, prices):
        return np.zeros(len(prices), dtype=np.int8)
"#;
        let err = validate_strategy_source(src).unwrap_err();
        assert!(err.contains("socket"), "message should name the module: {err}");
    }

    // The AST-level "no return" detection only runs under the `python` feature.
    #[cfg(feature = "python")]
    #[test]
    fn truncated_missing_return_rejected() {
        let err = validate_strategy_source(TRUNCATED_NO_RETURN).unwrap_err();
        assert!(err.to_lowercase().contains("return") || err.to_lowercase().contains("truncated"));
    }

    #[cfg(feature = "python")]
    #[test]
    fn syntax_error_rejected() {
        let src = r#"
from trading_platform import BaseStrategy
class Strategy(BaseStrategy):
    def compute_signals(self, prices):
        x = (1 + 
        return x
"#;
        assert!(validate_strategy_source(src).is_err());
    }

    // Without the python feature the textual check still passes a
    // well-formed-looking strategy (no false positives).
    #[cfg(not(feature = "python"))]
    #[test]
    fn textual_only_accepts_entrypoint() {
        assert!(validate_strategy_source(TRUNCATED_NO_RETURN).is_ok());
    }
}
