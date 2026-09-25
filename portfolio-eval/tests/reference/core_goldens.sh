#!/bin/bash
# Produces the golden values in tests/core_equivalence.rs by running Core's OWN, UNMODIFIED source:
#   metrics/src/significance.rs      deflated_sharpe_ratio(observed_sharpe, n_trials, n_days, sharpe_std, ppy)
#   metrics/src/performance.rs       deflated_sharpe_ratio(num_trials, returns)
#   quant-diagnostics/src/multiple_testing.rs   benjamini_hochberg(p_values)
# The three files are copied verbatim into a throw-away crate (only a two-line logging shim is added, because
# significance.rs logs through Core's logging facade) and a main() prints the numbers on the fixed inputs below.
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
cat > "$TMP/Cargo.toml" <<'EOF'
[package]
name = "core-goldens"
version = "0.0.0"
edition = "2021"
[dependencies]
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
}
EOF
if [ "${1:-}" = "--prepare" ]; then
  echo "prepared $TMP"
  exit 0
fi
cd "$TMP"
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.90.0 run --quiet --offline 2>/dev/null
