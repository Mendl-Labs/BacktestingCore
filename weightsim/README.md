# weightsim

Pure, deterministic weight-target sleeve simulator (Stage T1 of `product-mandate/BACKTESTER_TRUTH_DESIGN.md`).
Zero dependencies, no I/O. The semantics, the design choices C1-C14 and the public API map are in the crate docs
(`src/lib.rs`); run `cd weightsim && cargo doc --open`.

## Tests

    cargo test                      # unit + integration + doc tests, no vendor data needed
    WEIGHTSIM_LADDER_DIR=<dir> cargo test --test answer_key -- --nocapture

`WEIGHTSIM_LADDER_DIR` points at the `replication_ladder` directory (vendor-derived `ladder_candles.csv` and the recorded
`shadow_*` key files, all verified by sha256 before use). It is NOT copied into this repository; without the variable the
test prints `SKIPPED`.

## Answer-key fixtures (`tests/fixtures/`)

`gen_answer_key.py` runs the pinned `shadow.py` (sha256 `d7014041...5a6e`) on a small deterministic SYNTHETIC panel and
writes the key files; `MANIFEST.sha256` pins every fixture and is verified by a test. Regenerating them is a deliberate,
reviewed act. `--verify-real <dir>` proves the harness reproduces the recorded real key to 1e-16.

## Mutation power

    python3 weightsim/mutants/run_mutants.py     # WEIGHTSIM_TEST_CMD overrides the test command

Each hand-written mutant is one exact source edit; every one must be killed by at least one test.
