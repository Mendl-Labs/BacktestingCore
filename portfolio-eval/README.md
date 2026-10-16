# portfolio-eval

Pure, deterministic portfolio-level evaluation statistics (stage PF5 of `product-mandate/PORTFOLIO_FIRST_BACKTESTER_DESIGN.md`,
layer D). It works on SERIES OF PORTFOLIO RETURNS that a book simulator produced; it never simulates trading and never
reads data, the clock, the network, a database or the operating system's randomness.

* `folds`: walk-forward, purged K-fold and CPCV bar ranges with purge and embargo.
* `bootstrap`: stationary block bootstrap (Politis-Romano) on row indices; Politis-White automatic block length.
* `marginal`: the paired equal-volatility marginal-contribution test (Sharpe difference, bootstrap p-value, CI) and the
  minimum detectable effect. This replaces the zero-sum `marginal_contribution_gate`.
* `hac`: Newey-West standard errors and the spanning regression `x = alpha + beta' p + eps`.
* `dsr`: deflated / probabilistic Sharpe, MinTRL, Benjamini-Hochberg q-values, and `core_compat`, bit-for-bit mirrors of
  Core's own formulas (so equivalence is tested, not assumed).
* `ledger`: book-level trial ledger with robust dispersion, sealed-holdout bookkeeping, effective breadth.
* `pbo`: probability of backtest overfitting (CSCV).
* `power`: Monte Carlo of the marginal test's size and power; `POWER_TABLE.md` is its committed, pinned output.
* `detmath`, `rng`, `stats`, `sha256`: in-crate primitives (no libm on any pinned path, so results are bit-identical
  across platforms).

## Rules for this crate

* ZERO dependencies (`std` only), standalone (own `[workspace]`, excluded from Core's workspace), public: synthetic
  fixtures only, no vendor-derived data.
* It must never depend on `backtest`, `weightsim`, the Engine or SignalEngine. Consumers depend on it, not the reverse.
* Every public function returns `Result<_, EvalError>` for bad input; none panics.
* No wall clock, no OS randomness, no threads inside the statistics; the Monte Carlo is deterministic for any thread
  count.

## Tests

    cargo test                                   # unit, integration, hand-computed references, property tests, determinism
    cargo test --release --test power_table -- --ignored --nocapture     # regenerate POWER_TABLE.md and compare (about 40 s)
    cargo run --release --bin power_table -- --write                     # rewrite the generated block on purpose
    python3 tests/reference/gen_reference.py                             # independent recomputation of the embedded reference values
    bash tests/reference/core_goldens.sh                                 # golden values from Core's own unmodified source
    python3 mutants/run_mutants.py                                       # every hand-written mutant must be killed

`tests/reference/gen_reference.py` recomputes every hand-computed number (exact rational arithmetic, an independent
implementation of the RNG and the bootstrap in Python, brute-force PBO and folds). `tests/reference/core_goldens.sh`
copies three Core source files unmodified into a scratch crate and prints what Core computes.
