# PF5 power table: what the marginal-contribution test can and cannot detect

Stage PF5 of `product-mandate/PORTFOLIO_FIRST_BACKTESTER_DESIGN.md` ("What to measure first (4)", doubt U7). The test
is `portfolio_eval::marginal::marginal_contribution`: a paired stationary block bootstrap of the difference of net
Sharpe ratios of book `P + X` and book `P` at equal volatility, one-sided, level 0.05, with the automatic
(Politis-White) block length. The tables below are a Monte Carlo of exactly that function.

The numbers between the `BEGIN GENERATED` and `END GENERATED` markers are produced by
`cargo run --release --bin power_table -- --write` (about 40 s on two cores; the result does not depend on the thread
count or on the platform, see `src/detmath.rs`) and are pinned three ways by `tests/power_table.rs`: an always-on check
of their SHA-256 against a constant in the test, an always-on check of the sanity properties of the parsed numbers
and of every number quoted in the prose below, and an `--ignored` test (run in CI in release mode) that regenerates the
whole block and compares it byte for byte. Editing the block by hand, or changing anything that alters the Monte Carlo,
fails those tests.

## Set-up (data-generating process)

Two unit-volatility sleeves, the existing book's sleeve `P` (annual Sharpe 0.5) and a candidate `X`, correlation
`rho`. The enlarged book is the static blend `B = (1 - w) P + w X` with `w = 0.5` (the robustness table also uses
`w = 0.2`). The TRUE incremental Sharpe is `effect = SR(B) - SR(P)` (annualised; Sharpe is scale free, so this is the
equal-volatility difference); `X`'s mean is solved from it in closed form (`PowerSpec::candidate_sharpe`). `effect = 0`
is the size column. Horizons are 2, 3.5, 5 and 10 years of daily bars (252 a year; 3.5 years is 882 bars, the crypto
row below uses 365 bars a year). Returns are iid Gaussian unless a scenario says otherwise. "MDE analytic" is the
Memmel (2003) iid-normal formula solved for the effect that a level-0.05 one-sided z-test detects with 80% power;
"MDE Monte Carlo" is the effect at which the simulated power curve crosses 80% (linear interpolation on the grid);
"mean bootstrap SE" is the average standard error of the Sharpe difference reported by the test at `effect = 0`.

<!-- BEGIN GENERATED (cargo run --release --bin power_table) -->
## Size and power by horizon and correlation

Experiment: 400 repetitions per cell, 199 bootstrap replicates, 252 bars/year, base Sharpe 0.5, candidate share 0.5, one-sided alpha 0.05, target power 0.8, seed 0x50463520504f5745.

Cells are the rejection rate in percent of the one-sided paired marginal test at each TRUE incremental annual Sharpe (columns); the `0 (size)` column is the false-positive rate. Monte Carlo standard error of a rate p is sqrt(p(1-p)/400), at most 2.5 percentage points.

### Correlation rho = 0 between the existing book's sleeve set and the candidate

| years | bars | 0 (size) | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | MDE analytic | MDE Monte Carlo | mean bootstrap SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 504 | 4.0 | 7.2 | 9.8 | 13.2 | 21.2 | 38.2 | 53.5 | 86.0 | 1.35 | 1.41 | 0.543 |
| 3.5 | 882 | 4.5 | 7.2 | 10.0 | 15.0 | 32.5 | 55.8 | 77.0 | 98.0 | 1.02 | 1.07 | 0.409 |
| 5 | 1260 | 5.2 | 6.8 | 12.0 | 21.2 | 42.5 | 68.5 | 87.5 | 99.5 | 0.85 | 0.90 | 0.344 |
| 10 | 2520 | 6.2 | 11.0 | 21.0 | 30.5 | 61.5 | 91.2 | 99.2 | 100.0 | 0.60 | 0.66 | 0.242 |

### Correlation rho = 0.3 between the existing book's sleeve set and the candidate

| years | bars | 0 (size) | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | MDE analytic | MDE Monte Carlo | mean bootstrap SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 504 | 5.2 | 8.5 | 10.8 | 16.5 | 28.2 | 50.0 | 69.5 | 96.8 | 1.10 | 1.19 | 0.441 |
| 3.5 | 882 | 5.8 | 10.8 | 18.0 | 24.2 | 43.5 | 70.5 | 86.8 | 99.8 | 0.83 | 0.90 | 0.334 |
| 5 | 1260 | 6.0 | 10.8 | 18.2 | 26.5 | 52.2 | 82.0 | 96.2 | 100.0 | 0.69 | 0.73 | 0.278 |
| 10 | 2520 | 3.5 | 8.5 | 22.5 | 39.2 | 78.5 | 98.8 | 100.0 | 100.0 | 0.49 | 0.52 | 0.197 |

### Correlation rho = 0.6 between the existing book's sleeve set and the candidate

| years | bars | 0 (size) | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | MDE analytic | MDE Monte Carlo | mean bootstrap SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 504 | 4.8 | 7.5 | 12.5 | 20.2 | 43.2 | 75.8 | 92.5 | 99.8 | 0.81 | 0.81 | 0.322 |
| 3.5 | 882 | 5.2 | 11.0 | 21.0 | 34.8 | 67.2 | 92.8 | 99.2 | 100.0 | 0.61 | 0.62 | 0.246 |
| 5 | 1260 | 6.2 | 15.8 | 28.2 | 46.5 | 77.8 | 97.8 | 100.0 | 100.0 | 0.51 | 0.53 | 0.206 |
| 10 | 2520 | 5.0 | 15.2 | 36.5 | 65.2 | 97.0 | 100.0 | 100.0 | 100.0 | 0.36 | 0.39 | 0.146 |

## Robustness scenarios

All rows: 3.5 years, rho = 0.3, 300 repetitions per cell, 199 bootstrap replicates, one-sided alpha 0.05.

| scenario | bars | 0 (size) | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | MDE analytic | MDE Monte Carlo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (Gaussian, 252 bars/yr, base SR 0.5, share 50%) | 882 | 4.7 | 7.0 | 10.7 | 19.7 | 39.3 | 69.0 | 87.0 | 99.0 | 0.83 | 0.90 |
| crypto calendar (365 bars/yr, 3.5y = 1278 bars) | 1278 | 3.7 | 7.0 | 13.0 | 21.0 | 42.0 | 70.7 | 90.0 | 98.7 | 0.83 | 0.87 |
| fat tails (Student-t, 4 df) | 882 | 7.0 | 11.3 | 18.7 | 27.3 | 52.7 | 73.7 | 88.3 | 99.7 | 0.83 | 0.86 |
| serial correlation (AR(1) 0.1) | 882 | 4.3 | 7.3 | 10.3 | 17.0 | 34.3 | 65.3 | 83.3 | 98.7 | 0.83 | 0.95 |
| bull-market base book (base SR 1.5) | 882 | 4.7 | 6.7 | 10.3 | 19.3 | 38.7 | 68.3 | 86.7 | 98.7 | 0.83 | 0.91 |
| small candidate (share 20%) | 882 | 4.7 | 16.0 | 48.7 | 78.0 | 99.3 | 100.0 | 100.0 | 100.0 | 0.29 | 0.32 |

Analytic z-multiplier (z_(1-alpha) + z_power) = 2.4865.

sha256 of the generated section above: `2b900978e702cd38bc6bade7d18df372f82adfed503669b39e8734bdedfe1831`
<!-- END GENERATED -->


## Reading the table honestly

**Size is right.** Under the null (candidate adds nothing) the test rejects 3.5% to 6.2% of the time across the twelve
horizon/correlation cells (mean about 5.1%; the Monte Carlo standard error of one cell is up to 2.5 percentage points),
so it is not liberal on Gaussian data. On fat tails (Student-t, 4 degrees of freedom) the rate was 7.0% with 300
repetitions (one standard error is 1.5 points): a mild inflation that is consistent with, but does not prove, a slightly
liberal test; treat a p-value just under 0.05 as borderline. The mean bootstrap SE equals the analytic SE
(`MDE analytic / 2.4865`) to within a few percent in every cell, and the Monte Carlo MDE is 0% to 10% ABOVE the
analytic MDE, so the reported MDE is not optimistic.

**Power is low.** With five years of daily data (`rho = 0.3`, half the book) the test detects an incremental Sharpe of
0.3 with 26.5% probability, 0.5 with 52%, and needs about 0.7 for 80%. With 3.5 years the MDE is about 0.83 to 0.90,
with 2 years about 1.1 to 1.2. Even ten years only reach about 0.5. The standard error of a Sharpe difference falls
like `1/sqrt(years)`: quadrupling the sample halves the MDE, and there is no shortcut.

**What ordinary diversification looks like.** Adding a second sleeve with the SAME standalone Sharpe 0.5 to a book
of Sharpe 0.5 at equal capital lifts the blend by only `0.5 / sqrt((1 + rho) / 2) - 0.5`: about 0.21 at `rho = 0`,
0.12 at `rho = 0.3`, 0.06 at `rho = 0.6`. At those true effects the 3.5-year test rejects only about 9% to 12% of
the time (interpolating the `0` (size), `0.1` and `0.2` columns), against 5% for a candidate that adds nothing. It cannot tell a useful diversifier from
noise, and, because the test is one-sided and unpowered, a failure to reject is NOT evidence of no contribution.
"Inconclusive" is the expected verdict for boring strategies on short samples. The platform must print the MDE next to
every marginal verdict and must not present a non-rejection as a rejection of the candidate (or a rejection of a
candidate tuned on the same window as a confirmation).

**What a 3.5-year crypto-bull window can and cannot support.** It can support only a candidate that lifts the blended
book's Sharpe by roughly 0.9 (half-book candidate) or, for a 20% candidate, by about 0.3, which for `rho = 0.3` means
a standalone Sharpe around 1.5 against a base of 0.5. It cannot confirm an ordinary diversifying addition, cannot rule
one out, and cannot certify that a Sharpe earned in a rising market persists. Three cautions specific to that setting:
(1) power depends on YEARS, not on bars: 3.5 years on the 365-day crypto calendar (1,278 bars) has the same power as
882 equity bars (the `crypto calendar` row), so 24/7 sampling does not buy statistical power; (2) an in-sample bull
Sharpe of 1.5 barely changes the test's precision under iid returns (the `base SR 1.5` row has the same MDE), so
strong past performance is not what makes the test powerful; the threat is regime dependence, which this Monte Carlo
does not model and which the bootstrap cannot see, because it resamples days inside the window it was given; (3)
high correlation between crypto sleeves makes the paired difference MORE precise (`rho = 0.6`: MDE 0.61 at 3.5 years)
and the diversification benefit SMALLER (about 0.06), so the detectable effect and the realistic effect move apart.

**What it does not cover.** Volatility clustering, regime changes, estimation error in the weights of the enlarged
book, selection of the candidate on the same data (the trial ledger and the sealed holdout in this crate exist for
that), and any correlation structure beyond two sleeves. AR(1) 0.1 serial correlation lowers power a little (the
Monte Carlo MDE is 0.95 against 0.83 analytic) because the automatic block length pays for dependence. Numbers here
are for one design point per row, not a guarantee for real books.
