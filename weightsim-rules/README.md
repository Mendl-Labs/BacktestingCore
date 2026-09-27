# weightsim-rules

The library decision rules inside the weight-target simulator, and the replication ladder that certifies the pair
against the pinned answer key. Stage T3 (milestone A: ETF and crypto certified in CI) of
`product-mandate/BACKTESTER_TRUTH_DESIGN.md`.

* `adapters`: `EtfTrendRule` and `CryptoTrendRule` implement `weightsim::WeightRule` around
  `reference_rules::{decide_etf_trend, decide_crypto_trend}` (replay options, date conversion, `RuleError` to
  `RuleRefusal` mapping: `InsufficientHistory` is `Warmup`, gaps and stale data are `Data`, the rest is `Other`).
  `FlatUntil` starts a sleeve flat at the bar before its window, as the key's ledger does. The FX adapter is Stage T5.
* `ladder`: pure functions. `Fixtures::from_dir` verifies the pinned `MANIFEST.json` and every file it lists (sha256 and
  size) and parses the answer key; `run_ladder` runs both rules gross and net (`certification_flat_10bps_per_side`)
  through `weightsim` and applies pre-registration Amendment 11 exactly; `self_test` returns `Err` unless every check
  passes; `LadderReport` (plain structs, `Display`) carries every number and the series digests.
  * Tier I: return correlation >= 0.99, |dSharpe| <= 0.05, |dCAGR| <= 0.5 pp, trades (signal flips) within 5%.
  * Tier II: per-bar identity <= 1e-9 for returns, equity, cost, turnover, standing target and held weights.
  * Tier III: weight agreement >= 0.98 on cells with |dw| <= 1e-6.
  * Tier IV: eight named mutants (same-day peek, extra delay, SMA excludes today, half sizing, drifting sub-accounts,
    S1 SMA excludes current, S1 one bar late, S1 wrong rebalance mode); each must fail, in the tiers `mutants.json` names.
  * Layer C canaries, cost identity, simulator poisoning, rule truncation and determinism on the real fixture.
* `ladder::verify` (stage T4, slice C1 of `product-mandate/T4_FIXED_RULE_ENDPOINT_PLAN.md`): what a run store needs.
  * `Fixtures::from_files(&[(name, bytes)], pins)` loads the fixtures from embedded bytes (`include_bytes!`) with the
    same trust chain as `from_dir` (both wrap `Fixtures::load`); identical bytes give identical fixtures and digests.
  * `replicate(fx, rule_id)` runs a library rule under the ladder's base configuration and returns a `ReplicationRun`:
    the FULL-resolution gross and net series as plain `weightsim::SeriesColumns` (exactly the fields the series digest
    covers), their digests, a `RunSummary` per basis (Sharpe, CAGR, vol, max drawdown, flips; the key's definitions) and
    the refusals. `rule_facts(rule_id)` gives the rule's declared facts.
  * `verify(fx, rule_id, gross, net)` (or `verify_with` for a cost preset, the store's claims and options) re-derives a
    stored run from its columns alone: shape and finiteness, rule id, implementation version, symbols, cost preset,
    claimed digests, a re-run from the fixtures that must be bit-identical, then the counted window, the metrics and
    the flips recomputed FROM THE STORED COLUMNS, and Tiers I-III (gross and net) and Tier IV. Every tamper class is a
    typed `VerifyError` (`DigestMismatch` names the first differing cell, `Truncated`, `NonFinite`,
    `CostModelMismatch`, `ImplVersionMismatch`, `ClaimedMetricsMismatch`, ...); a genuine run that does not reproduce
    the key is `TierFailed` and carries the numbers.
  * `weightsim-rules` stays at 0.1.0: the crate version is part of `rule_impl_version`, a series-digest input, so a
    bump changes every pinned digest.

Dependencies: `weightsim` and `reference-rules` by path (neither depends on the other or on this crate) and `chrono`.
No serde, no I/O other than reading the fixture files, no network. Standalone (own `[workspace]` and `Cargo.lock`,
excluded from Core's workspace) so a hosted runner never has to resolve Core's private git dependencies. Consume by
git revision:

    weightsim-rules = { git = "https://github.com/Mendl-Labs/BacktestingCore", rev = "<sha>" }

## Tests

    cargo test --locked                       # always-on; no vendor data

* `tests/adapters_synthetic.rs`: the adapters reproduce the committed PYTHON key of `weightsim`'s synthetic fixture
  (`../weightsim/tests/fixtures/`), plus refusal mapping and causality.
* `tests/ladder_synthetic.rs`: the ladder logic end to end on an in-memory synthetic answer key (built from independent
  oracle rules) in the exact layout of the real fixture; every tier, tolerance and pin is broken on purpose, one at a
  time, and exactly the right check must fail.
* `tests/mutants_oracle.rs`: each of the eight mutants recomputed in closed form or with a plain ledger.
* `tests/negative_controls.rs`: a leaky, an end-of-array, a nondeterministic and a merely wrong rule are handed to the
  per-sleeve certification and must fail the matching checks.
* `tests/runner_units.rs`: alignment conventions, the entry bar, the key's trade counter, `FlatUntil`.
* `tests/replication_verify.rs`: replicate, package and re-verify on the synthetic fixtures: the round trip, bit-for-bit
  metrics, one test per tamper class (one ulp, a date, a fixture byte, the rule id, the implementation version, the cost
  preset, a dropped row, swapped gross/net columns, NaN, a wrong claimed digest or summary), determinism, and the bytes
  loader against the file loader.
* `tests/replication_verify_real.rs`: the same on the REAL pinned data (env-gated like `ladder_real.rs`).
* `tests/ladder_real.rs`: the REAL pinned data. Vendor-derived, so it is not in this public repository: env-gated.

      WEIGHTSIM_RULES_LADDER_DIR=<Engine>/program/tests/fixtures/replication_ladder cargo test --release --test ladder_real -- --nocapture

  Prints `SKIPPED` when the variable is unset. The Engine runs the same `ladder::self_test` on every CI run.

## Mutation power

    python3 weightsim-rules/mutants/run_mutants.py      # WEIGHTSIM_RULES_TEST_CMD overrides the test command

Each hand-written mutant is one exact source edit (tolerance loosened, Tier II skipped, a mutant not applied, a manifest
check skipped, net and gross swapped, a date conversion off by one, a refusal mapped wrongly, ...); every one must be
killed by at least one always-on test.
