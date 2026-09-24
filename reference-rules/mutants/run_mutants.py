#!/usr/bin/env python3
"""Hand-written mutation testing for `reference-rules` (BACKTESTER_TRUTH_DESIGN.md 6.1, stage T2).

Each mutant is ONE exact source edit (the old text must occur exactly once). For every mutant the script applies the
edit, runs the test command, records which tests FAILED (or that the build broke), and restores the file byte for byte.
A mutant that no test kills is reported as SURVIVED and the script exits non-zero.

Usage (from anywhere):
    REFRULES_TEST_CMD="cargo test --locked --no-fail-fast" python3 mutants/run_mutants.py [--only M03,M07]
`REFRULES_TEST_CMD` defaults to the command above and is run with the crate directory as its working directory.
The real-history goldens are env-gated (REFRULES_LADDER_DIR) and skip themselves when it is unset, so a table
produced without that variable is the one CI can reproduce (always-on tests only: unit, boundaries, refusals,
properties, synthetic goldens, the FX clip golden).
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (id, description, file relative to the crate, old text (exactly once), new text)
MUTANTS = [
    ("M01", "crypto: off-by-one lookback (99-day SMA instead of 100)", "src/crypto.rs",
     "pub const CRYPTO_SMA_DAYS: usize = 100;",
     "pub const CRYPTO_SMA_DAYS: usize = 99;"),
    ("M02", "crypto: SMA excludes today's close", "src/crypto.rs",
     "let window = &s.closes()[lo..=pos];",
     "let window = &s.closes()[lo..pos];"),
    ("M03", "etf: off-by-one lookback (9 month-end closes instead of 10)", "src/etf.rs",
     "pub const ETF_SMA_MONTH_ENDS: usize = 10;",
     "pub const ETF_SMA_MONTH_ENDS: usize = 9;"),
    ("M04", "etf: SMA excludes the current month-end close", "src/etf.rs",
     "compare_to_mean(close, &g.month_end_closes[..])",
     "compare_to_mean(close, &g.month_end_closes[..ETF_SMA_MONTH_ENDS - 1])"),
    ("M05", "wrong month-end: the FIRST bar of a month is taken as the month-end", "src/months.rs",
     "        .filter(|&i| i + 1 == dates.len() || !same_month(dates[i], dates[i + 1]))",
     "        .filter(|&i| i == 0 || !same_month(dates[i], dates[i - 1]))"),
    ("M06", "wrong month-end: same_month ignores the year", "src/months.rs",
     "a.year() == b.year() && a.month() == b.month()",
     "a.month() == b.month()"),
    ("M07", "sign flip: crypto goes long BELOW the average", "src/crypto.rs",
     "let signal = if cmp.ordering.is_gt() {",
     "let signal = if cmp.ordering.is_lt() {"),
    ("M08", "etf: a tie (close == average) counts as above (not strictly above)", "src/etf.rs",
     "let signal = if cmp.ordering.is_gt() {",
     "let signal = if cmp.ordering.is_ge() {"),
    ("M09", "etf: wrong sizing (25% per ETF instead of 20%)", "src/etf.rs",
     "pub const ETF_WEIGHT_PER_INSTRUMENT: f64 = 0.20;",
     "pub const ETF_WEIGHT_PER_INSTRUMENT: f64 = 0.25;"),
    ("M10", "crypto: wrong sizing (25% per coin instead of 50%)", "src/crypto.rs",
     "pub const CRYPTO_WEIGHT_PER_INSTRUMENT: f64 = 0.50;",
     "pub const CRYPTO_WEIGHT_PER_INSTRUMENT: f64 = 0.25;"),
    ("M11", "fx: sign flip of the momentum sign", "src/fx.rs",
     "sign[k] = sign_of(now / then - 1.0);",
     "sign[k] = -sign_of(now / then - 1.0);"),
    ("M12", "fx: off-by-one lookback (12 month-ends instead of 13)", "src/fx.rs",
     "pub const FX_MOMENTUM_MONTH_ENDS: usize = 13;",
     "pub const FX_MOMENTUM_MONTH_ENDS: usize = 12;"),
    ("M13", "fx: wrong clip (upper cap 6 instead of 3)", "src/fx.rs",
     "weight: scaled.clamp(-FX_WEIGHT_CAP, FX_WEIGHT_CAP),",
     "weight: scaled.clamp(-FX_WEIGHT_CAP, 2.0 * FX_WEIGHT_CAP),"),
    # (The tempting variant `>` -> `>=` is an EQUIVALENT mutant: it differs only when the scaled weight is exactly 3.0
    # in binary floating point, which no input reaches; it is deliberately not in this list.)
    ("M14", "fx: clipped flag threshold wrong (flag only when the weight exceeds twice the cap)", "src/fx.rs",
     "clipped: scaled.abs() > FX_WEIGHT_CAP,",
     "clipped: scaled.abs() > 2.0 * FX_WEIGHT_CAP,"),
    ("M15", "fx: wrong vol window (59 returns instead of 60)", "src/fx.rs",
     "pub const FX_VOL_WINDOW: usize = 60;",
     "pub const FX_VOL_WINDOW: usize = 59;"),
    ("M16", "fx: standard deviation with ddof 0 instead of 1", "src/fx.rs",
     "Some((ss / (n - 1.0)).sqrt())",
     "Some((ss / n).sqrt())"),
    ("M17", "fx: ppy uses 365 instead of 365.25 days per year", "src/fx.rs",
     "let years = days as f64 / 365.25;",
     "let years = days as f64 / 365.0;"),
    ("M18", "fx: wrong sleeve volatility target (11% instead of 10%)", "src/fx.rs",
     "pub const FX_SLEEVE_VOL_TARGET: f64 = 0.10;",
     "pub const FX_SLEEVE_VOL_TARGET: f64 = 0.11;"),
    ("M19", "gap policy: missing-days count off by one (adjacent days count as one missing)", "src/months.rs",
     "((b - a).num_days() - 1).max(0) as u32",
     "(b - a).num_days().max(0) as u32"),
    ("M20", "crypto: minimum history off by one (101 bars needed)", "src/crypto.rs",
     "if pos + 1 < CRYPTO_SMA_DAYS {",
     "if pos + 1 <= CRYPTO_SMA_DAYS {"),
]


def run(cmd):
    p = subprocess.run(cmd, shell=True, cwd=CRATE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def failing_tests(out):
    names = re.findall(r"^test (.+?) \.\.\. FAILED$", out, flags=re.M)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def main():
    cmd = os.environ.get("REFRULES_TEST_CMD", "cargo test --locked --no-fail-fast")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    code, out = run(cmd)
    if code != 0:
        print("BASELINE FAILED; refusing to mutate a red tree\n" + out[-3000:])
        sys.exit(2)
    print("baseline: green\n")
    survived = []
    print("| id | mutant | outcome | failing tests |")
    print("|---|---|---|---|")
    for mid, desc, rel, old, new in MUTANTS:
        if only and mid not in only:
            continue
        path = os.path.join(CRATE, rel)
        with open(path, "r", newline="") as f:
            original = f.read()
        n = original.count(old)
        if n != 1:
            print(f"| {mid} | {desc} | BAD MUTANT (old text occurs {n}x) | |")
            survived.append(mid)
            continue
        try:
            with open(path, "w", newline="") as f:
                f.write(original.replace(old, new))
            code, out = run(cmd)
        finally:
            with open(path, "w", newline="") as f:
                f.write(original)
        fails = failing_tests(out)
        compile_err = bool(re.search(r"^error(\[E\d+\])?:", out, flags=re.M)) and "could not compile" in out
        if fails:
            outcome = "KILLED"
        elif compile_err:
            outcome = "NOT COMPILABLE (invalid mutant)"
            survived.append(mid)
        elif code != 0:
            outcome = "KILLED (build/test aborted)"
        else:
            outcome = "SURVIVED"
            survived.append(mid)
        print(f"| {mid} | {desc} | {outcome} | {', '.join(fails[:8])}{' ...(+%d)' % (len(fails) - 8) if len(fails) > 8 else ''} |", flush=True)
    print()
    print("SURVIVORS / INVALID:", survived if survived else "none")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
