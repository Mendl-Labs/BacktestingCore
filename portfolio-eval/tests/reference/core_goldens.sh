#!/bin/bash
# Produces the golden values in tests/core_equivalence.rs by running Core's OWN, UNMODIFIED source:
#   metrics/src/significance.rs                     deflated_sharpe_ratio(observed_sharpe, n_trials, n_days, sharpe_std, ppy)
#   metrics/src/performance.rs                       deflated_sharpe_ratio(num_trials, returns)
#   quant-diagnostics/src/multiple_testing.rs        benjamini_hochberg(p_values)
#   backtest/src/statistical_significance.rs         compute_deflated_sharpe_ratio(...), expected_max_sharpe_under_null(...)
#     (the THIRD, distinct DSR formula -- the one actually wired into the live Promising/Underperformed/Inconclusive
#     verdict via BacktestingEngine/program/src/worker.rs; see portfolio-eval/src/dsr.rs's module doc comment)
# The four files are copied verbatim into a throw-away crate (only a two-line logging shim is added, because
# significance.rs logs through Core's logging facade; statistical_significance.rs needs only `serde`, added as a
# real dependency of the throw-away crate) and a main() prints the numbers on the fixed inputs below.
#
#   bash tests/reference/core_goldens.sh              build and run in a temporary directory (cargo, no network, no dependencies)
#   bash tests/reference/core_goldens.sh --prepare D  only create the crate in directory D (then `cargo run` there)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE="$(cd "$HERE/../../.." && pwd)"
if [ "${1:-}" = "--prepare" ]; then
  TMP="$2"
  mkdir -p "$TMP"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
fi
mkdir -p "$TMP/src"
cp "$CORE/metrics/src/significance.rs" "$TMP/src/significance.rs"
cp "$CORE/metrics/src/performance.rs" "$TMP/src/performance.rs"
cp "$CORE/quant-diagnostics/src/multiple_testing.rs" "$TMP/src/multiple_testing.rs"
cp "$CORE/backtest/src/statistical_significance.rs" "$TMP/src/statistical_significance.rs"
cat > "$TMP/Cargo.toml" <<'EOF'
[package]
name = "core-goldens"
version = "0.0.0"
edition = "2021"
[dependencies]
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
rayon = "1.8"
[workspace]
EOF
cat > "$TMP/src/logging_facade.rs" <<'EOF'
pub const METRICS_LOGGER: &str = "metrics";
#[macro_export]
macro_rules! log_warn { ($($t:tt)*) => {{}}; }
EOF
cat > "$TMP/src/main.rs" <<'EOF'
#![allow(dead_code, unused_imports, unused_macros)]
mod logging_facade;
mod multiple_testing;
mod performance;
mod significance;
mod statistical_significance;

// the fixed return series shared with tests/core_equivalence.rs (deterministic, no RNG)
fn series(n: usize, mu: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let a = ((t * 37 + 11) % 101) as f64 / 101.0 - 0.5;
            let b = ((t * 53 + 7) % 89) as f64 / 89.0 - 0.5;
            mu + amp * (a + 0.6 * b * b * if t % 5 == 0 { 3.0 } else { 1.0 } - 0.1)
        })
        .collect()
}

fn main() {
    println!("# significance::deflated_sharpe_ratio(obs, n_trials, n_days, sharpe_std, ppy)");
    for (obs, k, n, sd, ppy) in [
        (1.5, 1usize, 500usize, 1.0, 365.0),
        (1.5, 20, 500, 1.0, 365.0),
        (2.0, 100, 1260, 0.5, 252.0),
        (0.8, 1000, 900, 0.7, 252.0),
        (-0.4, 10, 300, 1.0, 365.0),
        (3.0, 5000, 2500, 1.27, 252.0),
    ] {
        println!("sig ({obs}, {k}, {n}, {sd}, {ppy}) = {:?}", significance::deflated_sharpe_ratio(obs, k, n, sd, ppy));
    }
    println!("# performance::deflated_sharpe_ratio(num_trials, returns)");
    let r1 = series(500, 0.0004, 0.01);
    let r2 = series(1260, 0.0002, 0.012);
    let r3 = series(120, -0.0003, 0.02);
    for (name, r) in [("r1", &r1), ("r2", &r2), ("r3", &r3)] {
        for k in [1u32, 10, 200] {
            println!("perf ({name}, {k}) = {:?}", performance::deflated_sharpe_ratio(k, r));
        }
    }
    println!("# multiple_testing::benjamini_hochberg");
    println!("bh1 = {:?}", multiple_testing::benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005]));
    println!("bh2 = {:?}", multiple_testing::benjamini_hochberg(&[0.001, 0.2, 0.05, 0.3, 0.02, 0.02]));

    println!("# statistical_significance::expected_max_sharpe_under_null(num_strategies, lookback_periods)");
    // n_trials = 1 (edge: ln(1) = 0 -> early return 0.0), n_trials = 2 (smallest n with ln(n) > 0),
    // a spread of ordinary trial counts and lookback lengths, a very large n_trials, and lookback_periods = 1
    // (edge: (t-1)/t = 0 -> the finite-sample adjustment zeroes the result).
    let em_inputs: [(usize, usize); 10] = [
        (1, 500),
        (2, 500),
        (2, 2),
        (10, 252),
        (100, 1260),
        (1000, 900),
        (5000, 2500),
        (10_000_000, 1000),
        (100, 1),
        (100, 2),
    ];
    for (n, t) in em_inputs {
        println!(
            "em ({n}, {t}) = {:?}",
            statistical_significance::expected_max_sharpe_under_null(n, t)
        );
    }

    println!("# statistical_significance::compute_deflated_sharpe_ratio(sharpe, se, num_strategies_tested, expected_max_sharpe_under_null(k, lookback))");
    // Composed the way a caller composes the two real functions. Covers: n_trials = 1, a very large n_trials,
    // zero variance (se = 0.0, short-circuits to 0.0 regardless of the other inputs), num_strategies_tested = 0
    // (short-circuits to 0.0), sharpe = 0.0, a negative sharpe, and a spread of ordinary sharpe/se/trial-count/
    // lookback combinations.
    let dsr_inputs: [(f64, f64, usize, usize); 11] = [
        (1.5, 0.2, 1, 500),
        (1.5, 0.2, 20, 500),
        (2.0, 0.15, 100, 1260),
        (0.8, 0.25, 1000, 900),
        (-0.4, 0.3, 10, 300),
        (3.0, 0.05, 5000, 2500),
        (1.0, 0.2, 10_000_000, 1000),
        (1.0, 0.0, 5, 100),
        (1.0, 0.2, 0, 100),
        (0.0, 0.2, 50, 252),
        (5.0, 0.1, 2, 500),
    ];
    for (sharpe, se, k, lookback) in dsr_inputs {
        let em = statistical_significance::expected_max_sharpe_under_null(k, lookback);
        let dsr = statistical_significance::compute_deflated_sharpe_ratio(sharpe, se, k, em);
        println!("dsr ({sharpe}, {se}, {k}, {lookback}) = {dsr:?}");
    }
}
EOF
if [ "${1:-}" = "--prepare" ]; then
  echo "prepared $TMP"
  exit 0
fi
cd "$TMP"
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.90.0 run --quiet --offline 2>/dev/null
