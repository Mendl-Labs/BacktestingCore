# weightsim

Pure, deterministic weight-target simulator: the single-sleeve simulator (Stage T1 of
`product-mandate/BACKTESTER_TRUTH_DESIGN.md`) and, since 0.2, the joint-account BOOK simulator (phase PF1 of
`product-mandate/PORTFOLIO_FIRST_BACKTESTER_DESIGN.md`: several sleeves in one account on the union clock, sleeve combine,
capital base, risk scale, trade filter, cash policy, both cadence modes, a frozen inverse-vol allocator, per-sleeve shadow
curves, attribution, an overlay hook). `simulate` is unchanged; `simulate_book` with one sleeve is bit-identical to it.
Zero dependencies, no I/O. The semantics, the design choices C1-C14 and the public API map are in the crate docs
(`src/lib.rs`); run `cd weightsim && cargo doc --open`.

## Tests

    cargo test                      # unit + integration + doc tests, no vendor data needed
    WEIGHTSIM_LADDER_DIR=<dir> cargo test --test answer_key -- --nocapture

`WEIGHTSIM_LADDER_DIR` points at the `replication_ladder` directory (vendor-derived `ladder_candles.csv` and the recorded
`shadow_*` key files, all verified by sha256 before use). It is NOT copied into this repository; without the variable the
test prints `SKIPPED`.

### Book tests (weightsim 0.2)

    cargo test --test book_identity --test book_key --test book_props --test book_causality   # always on, synthetic
    WEIGHTSIM_LADDER_DIR=<replication_ladder> WEIGHTSIM_BOOK_KEY_DIR=<replication_ladder_book> \
        cargo test --test book_key_real --test book_identity -- --nocapture                    # real data, env-gated

The real-data tests compare the simulator with the PF0 book key (`replication_ladder_book/`, pre-registered in Amendment 12,
private Engine repository): five per-bar files, the netting file, the T0 single-sleeve files and the finding-F1 numbers; and
check one-sleeve identity with `simulate` on the real candles. Without the variables they print `SKIPPED`.

## Answer-key fixtures (`tests/fixtures/`)

`gen_answer_key.py` runs the pinned `shadow.py` (sha256 `d7014041...5a6e`) on a small deterministic SYNTHETIC panel and
writes the key files; `MANIFEST.sha256` pins every fixture and is verified by a test. Regenerating them is a deliberate,
reviewed act. `--verify-real <dir>` proves the harness reproduces the recorded real key to 1e-16.

## Book fixtures (`tests/book_fixtures/`)

`gen_book_key.py` is an independent plain-Python account engine that MIRRORS the semantics of the PF0 book key on SYNTHETIC
prices (integer LCG) and writes per-bar files in the key's column layout; `--verify-real <key dir> <ladder dir>` runs the
same engine on the real pinned candles and proves it reproduces every cell of the real per-bar key files bit for bit, so the
committed synthetic outputs are the outputs of an engine proven equal to the key. `MANIFEST.sha256` pins every file and CI
regenerates the CSVs and compares them byte for byte. No vendor data and no real key file is copied into this repository.

## The PF2 boundary

`src/construct.rs` holds the minimal portfolio construction the book key pins (combine, capital base, risk scale, gross-cap
refusal, trade filter, cash policy, frozen inverse-vol shares) behind the `Construct` trait. PF2's shared crate replaces it;
`simulate_book_with` takes any `&dyn Construct`. The module docs list the exact signatures.

## Mutation power

    python3 weightsim/mutants/run_mutants.py     # WEIGHTSIM_TEST_CMD overrides the test command

Each hand-written mutant is one exact source edit; every one must be killed by at least one test.

    python3 weightsim/mutants/run_book_mutants.py           # 51 mutants of the book simulator (design 6.5 list + our own)
    python3 weightsim/mutants/run_book_mutants.py --check   # verify that every edit still matches exactly once

`WEIGHTSIM_BOOK_TEST_CMD` overrides the test command; every book mutant must be killed by the ALWAYS-ON tests.
