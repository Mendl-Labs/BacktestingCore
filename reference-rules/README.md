# reference-rules

The pure decision rules of the documented sleeves: ETF trend (`decide_etf_trend`), crypto trend
(`decide_crypto_trend`) and FX time-series momentum (`decide_fx_tsmom`). No I/O, no network, no clock (the run
date is an argument). The interpretation choices are written at the top of `src/lib.rs`.

This crate is the ONE implementation of those rules. Both the backtester (through the weight-target simulator
`weightsim`, via a thin adapter) and the rebalancer (SignalEngine) consume it, so they agree by construction
(`product-mandate/BACKTESTER_TRUTH_DESIGN.md`, Section 2.3, stage T2). It moved here from SignalEngine
`crates/reference-rules` with the public API and the source unchanged.

Standalone on purpose (own `[workspace]` and `Cargo.lock`, excluded from Core's workspace, like `weightsim`), so a
hosted runner never has to resolve Core's private git dependencies. Three dependencies: `chrono` (no default
features), `sha2`, `hex`. Consume it by git tag or revision:

    reference-rules = { git = "https://github.com/Mendl-Labs/BacktestingCore", tag = "vX.Y.Z" }

## Tests

    cargo test --locked             # unit + integration; needs no vendor data

The synthetic golden (`tests/golden_synthetic.rs`) replays the rules over `tests/data/synthetic_ladder_candles.csv`
and requires agreement with the answer key produced by the pinned `shadow.py` on it (`key_S1_signals.csv`,
`key_S3_signals.csv`; byte-identical copies of `weightsim/tests/fixtures/`).

The REAL-history goldens (`tests/golden.rs`, `tests/golden_fx.rs`: ETF 555/555, crypto 3654/3654 in the
reference-semantics mode, FX weights to 1e-12) need vendor-derived data that is NOT in this public repository. They
are env-gated and print `SKIPPED` when the variable is unset:

    REFRULES_LADDER_DIR=<replication_ladder dir>           # ladder_candles.csv, shadow_S1_monthend_signals.csv, shadow_S3_daily_signals.csv
    REFRULES_FX_GOLDEN_DIR=<dir with golden_fx_tsmom.csv>  # optional, defaults to REFRULES_LADDER_DIR; written by tests/data/gen_fx_golden.py
    cargo test --locked -- --nocapture

## Mutation power

    python3 reference-rules/mutants/run_mutants.py     # REFRULES_TEST_CMD overrides the test command

Each hand-written mutant is one exact source edit (off-by-one lookback, SMA including or excluding today, wrong
month-end, sign flip, wrong sizing, wrong clip, wrong volatility window, ...); every one must be killed by at least
one always-on test.
