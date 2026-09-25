# portfolio-construct

The shared portfolio-construction **specification**, on f64 (phase PF2 of `product-mandate/PORTFOLIO_FIRST_BACKTESTER_DESIGN.md`,
sections 3.2 and 5). Zero dependencies, no I/O, no clock, no price source: the same inputs give bit-identical outputs.

It answers one question in code: *given what each sleeve wants to hold, how much capital there is and what the risk overlay
allows, what should the account hold and what has to be traded to get there?* The backtester (`weightsim` v0.2, through
`simulate_book`, a later step) and the live rebalancer's parity tests (SignalEngine, PF3) can both call it.

It is **not** the live planner. The planner's exact-Decimal arithmetic, the pre-trade guard, the broker-reported buying
power, order tags and idempotency stay in SignalEngine. What this crate supports is the claim "same rules and same sizing
specification, equal to a stated tolerance", never "backtest equals live by construction" (design 1.4).

Minimum Rust: 1.82 (`Option::is_none_or`); CI uses 1.90.0.

## Public API

| Item | What it is |
|---|---|
| `construct(&ConstructInputs) -> Result<ConstructOutput, ConstructRefusal>` | the ten steps of the design's order of operations (module docs of `construct.rs`) |
| `Limits`, `LimitPolicy` | position / asset-class / gross / net caps, shorting, leverage; whole-book refusal (R1/R2) or planner-faithful |
| `MarginModel`, `NoMargin`, `OandaMargin` (R3), `AlpacaRegT` (R4), `BuyingPower` | margin used, ceiling, notional buying power with the binding factor recorded |
| `TradeFilter { min_abs, min_pct }` | the planner's two numbers (defaults 10 units, 2% of the target) |
| `QuantityRounder`, `LotRounder`, `LotRule`, `ExactUnits`, `SizeRefusal` | rounding by venue lot rules, always down, typed refusals |
| `Ladder`, `LadderState`, `Ladder::step(&mut state, equity, day_start)` | drawdown ladder and daily-loss limit in f64 |
| `AllocatorSpec { Fixed, Equal, InverseVol }`, `AllocatorState::review` | static shares, reviews at declared dates, past-only |
| `schedule::{due, Cadence, CivilDate, BookCadence, plan_flags}` | which sleeve is due, which sleeves a run plans |
| `approval_risk_scale(cap)`, `capital_base`, `round_toward_zero_dp`, `floor_dp`, `ceil_dp` | R2 constant scale, the capital base, the 8-decimal helpers |

Exact signatures are in the rustdoc (`cargo doc --open`); the entry point is

```rust
pub fn construct(i: &ConstructInputs<'_>) -> Result<ConstructOutput, ConstructRefusal>;
```

## Design choices (numbered; the crate docs in `src/lib.rs` carry the same list)

1. `ConstructInputs`/`ConstructOutput`/`ConstructRefusal` follow design 3.2. Three fields are added to `ConstructInputs`
   because the design leaves them to the planner: `funding` (cash or buying-power budget), `target_dp`, `unmanaged_gross`.
   Sleeve weights are sparse `(instrument index, weight)` pairs: "named with weight 0" (sell the held position) differs from
   "not named" (leave it alone), as in the planner.
2. **One Decimal quantum.** `target = (capital_base * raw) * risk_scale`, optionally rounded toward zero at 8 decimals
   (`target_dp = Some(8)`). The 4-ulp snap in `floor_dp` makes an exact decimal tie land on the same quantum as the planner;
   any residual disagreement is at most one quantum (1e-8 currency units). Research runs use `None` (the answer key is unrounded f64).
3. **Tie-preserving boundaries.** Every cap and threshold uses a 1e-12 relative tolerance in the direction that lets an exact
   decimal tie behave as the planner does: a delta of exactly 10 trades, gross exactly at the cap is permitted, 1e-9 over the cap refuses.
4. **Whole-book refusal, never clipping** (R1/R2). Any breached limit refuses the whole book; the only scaling is the constant
   risk scale fixed at approval (`approval_risk_scale`).
5. **Order independence.** Sleeves are combined in id order and cross-instrument sums use a sorted-order sum, so permuting
   sleeves or instruments does not change a bit of the targets or of gross/net/margin (property-tested bit-for-bit).
6. **Units.** Run `construct` in ACCOUNT CURRENCY (equity 100000, `min_abs` 10), not on an equity normalised to 1.0: the
   planner's 8-decimal fee ceiling and its 1e-8-per-buy budget slack are absolute currency amounts (`ceil_dp(.., 8)`,
   `FEE_SLACK_PER_BUY`); on a normalised equity they are 1e-8 relative and would break a 1e-9 comparison with the key.
7. **Ladder.** `step(&mut state, equity, day_start)`; the caller supplies day-start as in the design's signature. Halts are
   sticky and the state records why. Ties (equity exactly on a trigger or a release level) behave as in the Decimal overlay.
8. **Allocators** are static between reviews; `InverseVol` is the PF0 key's pre-registered definition (sample standard
   deviation, ddof 1, of each sleeve's own-calendar returns over the last `lookback_bars`; shares proportional to `1/sd`); a
   review without history or with a degenerate deviation leaves the shares unchanged. `floor` (default 0) is a lower bound on the deviation.
9. **Schedule** carries its own civil-date type (no `chrono`); `weightsim` converts its `Date` when it integrates.

## Mirrored from the planner, and the known-deviation ledger

Mirrored (SignalEngine `main` 4937a43, `crates/rebalancer-core`): `capital_base * sum(share*w) * risk_scale`; capital base
`min(equity, allocated)`; weight/share validation (long-only unit bounds, signed `max_abs_weight` in `(0, 3]`); trade filter with the
target (or the current value on a full exit) as the band reference; sizes rounded DOWN; a full exit sells exactly the held
quantity; a trade crossing zero is a close leg and an open leg that covers the close leg's rounding dust; reductions before
increases, ordered by (venue, symbol); cash budget `cash - ceil8(reserve_fraction * capital_base)`, fee `ceil8(notional * rate)`,
one common scaling factor for all increases, per-buy fee slack; sell proceeds credited (or not); buying power replaces cash on a
margin book; `GrossAboveCap` (whole plan) and `BuyingPowerRequired`; a held short in an instrument no signed sleeve manages is
never touched; `NoPrice` skip; rounding of targets toward zero.

Deviations (each is a test or a `LimitPolicy` switch):

1. **Limits.** The planner checks only the gross cap and only for plans with a signed sleeve; position, class and net limits and
   shorting are per-order guard denials (the offending order is dropped, the rest is placed). `construct` refuses the whole book on
   any breach (R1/R2, not yet in the planner). `LimitPolicy::PlannerFaithful` reproduces today's plan-level behaviour.
2. The guard's per-order checks (turnover, orders per day, price staleness, universe, venue leverage and position units, halted or
   expired mandate, currency mismatch) are not modelled here.
3. Lot rounding is a caller-supplied table (`LotRounder`), not the adapters' `prepare_order`; symbol canonicalisation (`EUR/USD`
   vs `EUR_USD`) is the caller's.
4. Quantities are f64: at a lot boundary the floor can differ from the Decimal quotient by one venue quantum (a value within 4 ulps
   under a boundary counts as on it).
5. A ladder factor of 0 (halted) is allowed and flattens the book; the planner refuses a zero risk scale.
6. Fees are `ceil8(notional * rate)` on f64.
7. Shorting is a whole-book refusal here; the planner's guard denies the short order and places the rest.

Differences between the PF0 key and the live planner (Amendment 12, D1-D8) and where they land in this crate: D1 unrounded f64
(`target_dp = None`); D2 fractional units (`rounding = None`); D3 `certification` cash = `Funding::Unconstrained`, `budget` cash =
`Funding::Cash` with reserve 0 and the fee equal to the cost rate; D4 gross cap refusal = hold-previous whole book (the caller keeps
the previous units on `Err`); D5 the absolute minimum trade is in currency (the caller supplies the account size); D6 fills and
prices are the simulator's; D7 cadence timing is `schedule` (wall-clock month-end vs data-calendar last bar); D8 ladder, margin,
financing, shorting availability and data gates are outside the key, and the ladder and margin models here have their own tests.

## Proposed call boundary for `simulate_book` (a later step; nothing here touches `weightsim`)

Per bar `t` of the union clock, after marking to market:

1. rules give the standing weights of every sleeve and the `due[s]` flags (`schedule::due` for the calendar cadences);
2. `plan_flags(book.cadence, &due, &tradable)` says which sleeves are planned; an instrument is `in_scope` when a planned sleeve
   names it (out-of-scope instruments still count in every limit);
3. build `ConstructInputs`: `equity` = pre-cost equity in currency (the key's `E_pre`), `allocated_capital` from the account spec,
   `sleeves` = `SleeveTargets` of every sleeve with a standing target and `share` from `AllocatorState::shares()`, `risk_scale` =
   `RiskScale { approval_constant, ladder: ladder.step(..).scale }`, `limits` from the mandate (`RefuseWholeBook`), `instruments` =
   `InstrumentFacts` with the carry-aware mark as `price` and the ledger's units as `held_units`, `margin` (`NoMargin`,
   `OandaMargin`, `AlpacaRegT`), `trade_filter`, `rounding` (`None` in research, a `LotRounder` in `live_faithful`), `funding`
   (`Unconstrained` for certification, `Cash` for `live_faithful`), `target_dp` (`None`, or `Some(8)` in `live_faithful`);
4. `Ok(out)`: execute `out.trades` at the mark (units change by the signed quantity); the simulator's `CostModel` charges the cost
   on `sum(|notional|)` (`est_fee` is the planner's fee and is only used in `live_faithful`); `Err(refusal)`: keep the previous
   units for the WHOLE book and record the refusal (the T1 `HoldPrevious` semantics);
5. allocator reviews (`AllocatorState::review`) at the declared review dates, from the per-sleeve shadow gross returns.

## Tests

    cargo test                                # 121 always-on tests (62 mutants), no vendor data, no environment gates
    python3 ../portfolio-construct/mutants/run_mutants.py   # from the Core root; every mutant must be killed

* `tests/golden_planner.rs`: golden vectors transcribed by hand from the planner's own tests (`planner.rs`, `signed.rs`,
  `oanda_rules.rs`); the header lists what is mirrored and what could not be (guard, tags, digest, adapters' `prepare_order`, the
  hash-only `long_only_golden.rs`).
* `tests/hand_computed.rs`: hand-computed numbers for every function (capital base, sizing, risk scale, R2, filter boundaries, 8-decimal
  rounding, lot rules, gross-cap boundary at 1e-9, margin R3/R4, budgets).
* `tests/ladder.rs`: golden vectors from `rebalancer-risk`'s overlay tests plus an exact-integer oracle over 10,000 generated paths.
* `tests/allocators.rs`: hand cases, poisoning (values beyond the visible history cannot change a bit) and a Python reference on
  synthetic series (`tests/fixtures/gen_invvol.py`).
* `tests/properties.rs`: seeded properties (limits, no cash creation, never oversell or overshoot, permutation invariance, linearity,
  monotonicity, idempotence). `tests/determinism.rs`: bit-identity and pinned digests. `tests/schedule.rs`: calendar cases and a
  30-year weekday calendar against an independent oracle.

Every fixture is synthetic; no vendor data and no account data are in this repository.
