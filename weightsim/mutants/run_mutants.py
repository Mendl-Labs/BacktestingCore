#!/usr/bin/env python3
"""Hand-written mutation testing for `weightsim` (design 6.1 "Mutation testing", Tier IV).

Each mutant is ONE exact source edit (the old text must occur exactly once). For every mutant the script applies the
edit, runs the test command, records which tests FAILED (or that the build broke), and restores the file byte for byte.
A mutant that no test kills is reported as SURVIVED and the script exits non-zero.

Usage (from the crate directory or anywhere):
    WEIGHTSIM_TEST_CMD="cargo test --manifest-path weightsim/Cargo.toml --no-fail-fast" python3 mutants/run_mutants.py [--only M03,M07]
`WEIGHTSIM_TEST_CMD` defaults to the command above and is run with the workspace root as its working directory.
The output is a markdown table. The vendor-data test is env-gated (WEIGHTSIM_LADDER_DIR) and skips itself when unset,
so a table produced without that variable is the one CI can reproduce.
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CRATE)

# (id, description, file relative to the crate, old text (exactly once), new text)
MUTANTS = [
    ("M01", "one-bar-late decision (fill one bar after the decision)", "src/sim.rs",
     "pending.push_back((t + cfg.execution_delay_bars, scaled));",
     "pending.push_back((t + cfg.execution_delay_bars + 1, scaled));"),
    ("M02", "wrong rebalance mode (OnDecision behaves like EveryBar)", "src/sim.rs",
     "RebalancePolicy::OnDecision => newly_effective,",
     "RebalancePolicy::OnDecision => true,"),
    ("M03", "cost on notional held instead of turnover", "src/sim.rs",
     "traded += (nu - units[i]).abs() * p;",
     "traded += nu.abs() * p;"),
    ("M04", "ppy = 365 instead of n / years", "src/metrics.rs",
     "let ppy = n as f64 / years;",
     "let ppy: f64 = 365.0;"),
    ("M05", "look-ahead in HistoryView (view includes bar t+1)", "src/panel.rs",
     "HistoryView { panel, len: t + 1 }",
     "HistoryView { panel, len: (t + 2).min(panel.n_bars()) }"),
    ("M06", "sign flip of every target weight", "src/sim.rs",
     "let scaled: Vec<f64> = w.iter().map(|v| v * cfg.risk_scale).collect();",
     "let scaled: Vec<f64> = w.iter().map(|v| -v * cfg.risk_scale).collect();"),
    ("M07", "drop the refusal hold (a refusal flattens the book)", "src/sim.rs",
     "                    refused = true;\n                    res.refusals.push(Refusal {",
     "                    refused = true;\n                    pending.push_back((t + cfg.execution_delay_bars, vec![0.0; k]));\n                    res.refusals.push(Refusal {"),
    ("M08", "fill at the wrong bar (previous close)", "src/sim.rs",
     "                let p = panel.closes(i)[t];\n                let nu = w[i] * equity_pre / p;",
     "                let p = panel.closes(i)[t.saturating_sub(1)];\n                let nu = w[i] * equity_pre / p;"),
    ("M09", "std with ddof 0 instead of 1", "src/metrics.rs",
     "(ss / (x.len() - 1) as f64).sqrt()",
     "(ss / x.len() as f64).sqrt()"),
    ("M10", "cost recorded but not deducted from equity", "src/sim.rs",
     "let equity = equity_pre - cost;",
     "let equity = equity_pre;"),
    ("M11", "max drawdown measured from an initial 1.0 point", "src/metrics.rs",
     "let mut peak = f64::NEG_INFINITY;",
     "let mut peak = 1.0;"),
    ("M12", "month-end schedule fires on the FIRST bar of a month", "src/rule.rs",
     "t + 1 == dates.len() || !dates[t].same_month(dates[t + 1])",
     "t == 0 || !dates[t].same_month(dates[t - 1])"),
    ("M13", "financing sign error on long value", "src/costs.rs",
     "(cash_bps * cash - long_bps * long_value - short_bps * short_value)",
     "(cash_bps * cash + long_bps * long_value - short_bps * short_value)"),
    ("M14", "counted window includes the fill bar's own return", "src/sim.rs",
     "let i0 = (fe + 1).max(si);",
     "let i0 = fe.max(si);"),
    ("M15", "look-ahead in mark-to-market (uses the next bar's close)", "src/sim.rs",
     "                invested += units[i] * panel.closes(i)[t];\n            }\n            equity_pre = cash + invested;",
     "                invested += units[i] * panel.closes(i)[(t + 1).min(n - 1)];\n            }\n            equity_pre = cash + invested;"),
    ("M16", "EveryBar policy drifts (only trades when a new target arrives)", "src/sim.rs",
     "RebalancePolicy::EveryBar => true,",
     "RebalancePolicy::EveryBar => newly_effective,"),
    ("M17", "Abort policy ignored (refusals never abort)", "src/sim.rs",
     "if cfg.on_refusal == OnRefusal::Abort && !tolerated_warmup {",
     "if false && cfg.on_refusal == OnRefusal::Abort && !tolerated_warmup {"),
    ("M18", "nearest-rank percentile uses floor instead of ceil", "src/metrics.rs",
     "let rank = (p * v.len() as f64).ceil() as usize;",
     "let rank = (p * v.len() as f64).floor() as usize;"),
    ("M19", "series digest ignores held weights", "src/sim.rs",
     "for flat in [&r.target_weights, &r.held_weights, &r.units] {",
     "for flat in [&r.target_weights, &r.units] {"),
    ("M20", "same_month ignores the year", "src/date.rs",
     "self.year == other.year && self.month == other.month",
     "self.month == other.month"),
    ("M21", "cost rate off by 10x (bps / 1000)", "src/costs.rs",
     "self.rate_bps() / 10_000.0",
     "self.rate_bps() / 1_000.0"),
    ("M22", "risk_scale ignored", "src/sim.rs",
     "let scaled: Vec<f64> = w.iter().map(|v| v * cfg.risk_scale).collect();",
     "let scaled: Vec<f64> = w.iter().map(|v| v * 1.0).collect();"),
    ("M23", "flip counter includes the first decision as a flip", "src/sim.rs",
     "            if let Some(p) = prev {\n                for i in 0..k {\n                    if p[i] != signs[i] {\n                        res.signal_flips[i] += 1;\n                    }\n                }\n            }",
     "            let zero = vec![0i8; k];\n            {\n                let p = prev.unwrap_or(&zero);\n                for i in 0..k {\n                    if p[i] != signs[i] {\n                        res.signal_flips[i] += 1;\n                    }\n                }\n            }"),
    ("M24", "fixture pin not checked (hash mismatch ignored)", "src/panel.rs",
     "if !actual.eq_ignore_ascii_case(expected_sha256) {",
     "if false && !actual.eq_ignore_ascii_case(expected_sha256) {"),
    # Mutants of the hand-written TEST rules: show that the answer-key identity has teeth against rule-adapter errors
    # (the class of error the pre-registered bands cannot see for a monthly rule).
    ("M25", "S3 test rule sizes 25% per coin instead of 50% (the 'omitted sizing' class)", "tests/common/mod.rs",
     "w.push(if c[c.len() - 1] > sma { 0.5 } else { 0.0 });",
     "w.push(if c[c.len() - 1] > sma { 0.25 } else { 0.0 });"),
    ("M26", "S1 test rule averages only the last 9 month-end closes (still divides by 10)", "tests/common/mod.rs",
     "let last10 = &me[me.len() - 10..];",
     "let last10 = &me[me.len() - 9..];"),
]


def run(cmd):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def failing_tests(out):
    names = re.findall(r"^test (\S+) \.\.\. FAILED$", out, flags=re.M)
    # doc-tests print "test src/panel.rs - ... (line N) ... FAILED"
    names += re.findall(r"^test (.+?) \.\.\. FAILED$", out, flags=re.M)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def main():
    cmd = os.environ.get("WEIGHTSIM_TEST_CMD", "cargo test --manifest-path weightsim/Cargo.toml --no-fail-fast")
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
