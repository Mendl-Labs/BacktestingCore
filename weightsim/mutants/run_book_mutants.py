#!/usr/bin/env python3
"""Hand-written mutation testing for the BOOK simulator of `weightsim` v0.2 (design 6.5, PF1 mutant list, plus our own).

Each mutant is one or more EXACT source edits (every `old` text must occur exactly once in its file). For every mutant the
script applies the edits, runs the always-on book tests, records which tests FAILED (or that the build broke), and restores
every file byte for byte. A mutant that no test kills is reported as SURVIVED and the script exits non-zero.

Usage (from anywhere):
    python3 weightsim/mutants/run_book_mutants.py                 # run every mutant
    python3 weightsim/mutants/run_book_mutants.py --only B03,B10  # a subset
    python3 weightsim/mutants/run_book_mutants.py --check         # only verify that every `old` text matches exactly once
`WEIGHTSIM_BOOK_TEST_CMD` overrides the test command (default below), run with the repository root as working directory.
The real-data tests are env-gated (WEIGHTSIM_LADDER_DIR, WEIGHTSIM_BOOK_KEY_DIR) and skip themselves when unset, so the
table produced without those variables is the one CI can reproduce: every mutant is killed by the ALWAYS-ON tests.

Where the design's list (PORTFOLIO_FIRST_BACKTESTER_DESIGN.md 6.5, PF1 row) and this file meet:
  allocator reads same-bar sleeve return   -> B01 (stale window), B02 (net shadow): the allocator here is fed by ONLINE shadow
                                              accounts, so a read of the future is structurally impossible; its causality is
                                              the poisoning tests, and a look-ahead HARNESS mutant is B33
  share and risk-scale in the wrong order  -> B03
  trade filter on current, not target      -> B04
  carry adds return twice / drops it       -> B05 / B06
  per-sleeve policy swapped                -> B07
  shadow charged the joint account costs   -> B08
  ladder recovery hysteresis removed       -> B09 (the reference ladder of the overlay tests; the shared ladder is PF2's)
  refusal clips instead of refusing        -> B10
  signed sum replaced by absolute sum      -> B11
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CRATE)
DEFAULT_CMD = (
    "cargo test --manifest-path weightsim/Cargo.toml --locked --no-fail-fast "
    "--lib --test book_identity --test book_key --test book_props --test book_causality"
)

SIM = "src/book_sim.rs"
CON = "src/construct.rs"
PAN = "src/book_panel.rs"
RES = "src/book_result.rs"
HAR = "src/book_harness.rs"
STA = "src/stateful.rs"
PRP = "tests/book_props.rs"

WEIGHT_LINE = "let weight: Vec<Option<f64>> = raw.iter().map(|r| r.map(|v| v * p.risk_scale)).collect();"
CB_LINE = "let cb = capital_base(i.equity, p.allocated_capital);"
TERM_LINE = "let term = s.share * s.weights[k];"
WINDOWS_LINE = "let windows: Vec<&[f64]> = sl.iter().map(|r| r.sh_gross.as_slice()).collect();"
ALL_ON_ANY = "(0..s_n).map(|s| run && (has[s] || cfg.trade_on_closed_market)).collect()"

# (id, description, [(file relative to the crate, old text (exactly once), new text), ...])
MUTANTS = [
    ("B01", "allocator window is stale by one bar (excludes the review bar's own shadow return)", [(SIM, WINDOWS_LINE,
        "let windows: Vec<&[f64]> =\n                    sl.iter().map(|r| &r.sh_gross[..r.sh_gross.len().saturating_sub(1)]).collect();")]),
    ("B02", "allocator reads the NET shadow curves instead of the gross ones", [(SIM, WINDOWS_LINE,
        "let windows: Vec<&[f64]> = sl.iter().map(|r| r.sh_cost.as_slice()).collect();")]),
    ("B03", "risk scale applied inside the capital cap: min(equity*scale, allocated) instead of scale after the cap", [
        (CON, WEIGHT_LINE, "let weight: Vec<Option<f64>> = raw.clone();"),
        (CON, CB_LINE, "let cb = capital_base(i.equity * p.risk_scale, p.allocated_capital);")]),
    ("B04", "trade filter band is a fraction of the CURRENT value instead of the target", [(CON,
        "    if target == 0.0 {\n        current.abs()\n    } else {\n        target.abs()\n    }",
        "    if current == 0.0 {\n        target.abs()\n    } else {\n        current.abs()\n    }")]),
    ("B05", "carry adds a spurious return: a carried instrument drifts up 0.1% per closed bar", [(SIM,
        "                marked[j] = true;\n            }\n        }\n",
        "                marked[j] = true;\n            } else if marked[j] {\n                mark[j] *= 1.001;\n            }\n        }\n")]),
    ("B06", "carry drops the gap return: the first real bar after a carry keeps the stale mark", [(SIM,
        'mark[j] = panel.close(j)[u].expect("an open sleeve has every close");',
        'mark[j] = if u > 0 && marked[j] && panel.close(j)[u - 1].is_none() {\n                    mark[j]\n                } else {\n                    panel.close(j)[u].expect("an open sleeve has every close")\n                };')]),
    ("B07", "per-sleeve rebalance policy swapped (EveryBar sleeves become OnDecision and back)", [(SIM,
        "(sl[s].policy == RebalancePolicy::EveryBar || newly[s])", "(sl[s].policy != RebalancePolicy::EveryBar || newly[s])")]),
    ("B08", "shadow (sleeve-alone) curve is also charged the JOINT account's cost", [(SIM,
        "let rc = r.sh_c.finish(cost_c);", "let rc = r.sh_c.finish(cost_c + cost * r.sh_c.equity_pre / equity_pre);")]),
    ("B09", "ladder recovery hysteresis removed (reference ladder of the overlay tests recovers at the shrink level)", [(PRP,
        "} else if self.shrunk && dd >= -self.cfg.recover_at {", "} else if self.shrunk && dd >= -self.cfg.shrink_at {")]),
    ("B10", "gross-cap breach CLIPS every target to the cap instead of refusing the whole book", [
        (CON, WEIGHT_LINE, "let mut weight: Vec<Option<f64>> = raw.iter().map(|r| r.map(|v| v * p.risk_scale)).collect();"),
        (CON, "return Err(ConstructRefusal::GrossAboveCap { gross, cap });",
         "{\n                    let f = cap / gross;\n                    for w in weight.iter_mut().flatten() {\n                        *w *= f;\n                    }\n                }")]),
    ("B11", "signed sum across sleeves replaced by an absolute sum (opposite sleeves stop netting)", [(CON, TERM_LINE,
        "let term = (s.share * s.weights[k]).abs();")]),
    ("B12", "cadence modes swapped (PerSleeve plans every open sleeve, AllSleevesOnAnyDue plans only the due ones)", [
        (SIM, "BookCadence::PerSleeve => due.clone(),", "BookCadence::PerSleeve => (0..s_n).map(|s| run && has[s]).collect(),"),
        (SIM, ALL_ON_ANY, "due.clone()")]),
    ("B13", "no minimum-trade filter", [(CON, "if let Some(f) = p.trade_filter {", "if let Some(f) = p.trade_filter.filter(|_| false) {")]),
    ("B14", "the 10-unit absolute minimum read in the wrong units (x100000)", [(CON, "if ad < f.min_abs {", "if ad < f.min_abs * 100000.0 {")]),
    ("B15", "the approval risk scale is not applied", [(CON, "r.map(|v| v * p.risk_scale)", "r.map(|v| v * 1.0)")]),
    ("B16", "capital base = equity (the mandate's allocated capital is ignored)", [(CON, CB_LINE, "let cb = capital_base(i.equity, None);")]),
    ("B17", "buys execute at the full target regardless of the cash left (no common scale factor)", [(CON,
        "let factor = if needed <= avail {", "let factor = if true {")]),
    ("B18", "the whole-book gross cap is not checked", [(CON, "if let Some(cap) = p.max_gross {", "if let Some(cap) = p.max_gross.filter(|_| false) {")]),
    ("B19", "sleeve shares ignored, equal split used", [(CON, TERM_LINE, "let term = s.weights[k] / i.sleeves.len() as f64;")]),
    ("B20", "sleeves are re-targeted on bars where their market is closed (stale close)", [(SIM, ALL_ON_ANY, "(0..s_n).map(|_| run).collect()")]),
    ("B21", "sleeve calendar is the union of its instruments' bars instead of the inner join", [(PAN,
        "if universe.iter().all(|&i| self.close[i][u].is_some()) {", "if universe.iter().any(|&i| self.close[i][u].is_some()) {")]),
    ("B22", "a missing bar on an open market is classified as a declared closure (gaps become silent carries)", [(PAN,
        "} else if self.sessions[i].is_declared_closed(self.times[u].date()) {", "} else if true {")]),
    ("B23", "Sunday is not a closure of an exchange session", [(PAN, "date.weekday() >= 5 ||", "date.weekday() >= 6 ||")]),
    ("B24", "the book is sized on the PREVIOUS bar's equity", [(SIM, "                    equity: equity_pre,\n                    cash,", "                    equity: e_prev,\n                    cash,")]),
    ("B25", "the joint account is charged half the cost of its trades", [(SIM,
        "cost = traded * rate;\n                        cash = equity_pre - mv2 - cost;", "cost = traded * rate * 0.5;\n                        cash = equity_pre - mv2 - cost;")]),
    ("B26", "a refused bar liquidates the book instead of holding the previous units", [(SIM,
        "                        book_refused = true;\n", "                        book_refused = true;\n                        units = vec![0.0; n];\n")]),
    ("B27", "the counted window includes the initial-build bar", [(SIM, "let i0 = (fe + 1).max(si);", "let i0 = fe.max(si);")]),
    ("B28", "account start is exclusive (the start bar itself is skipped)", [(SIM, ".position(|x| *x >= t)", ".position(|x| *x > t)")]),
    ("B29", "sleeve contributions are computed on half the weights", [(SIM, "let piece = held_prev[j] * r;", "let piece = held_prev[j] * r * 0.5;")]),
    ("B30", "execution delay keyed to the account bar instead of the decision sleeve's own bars", [(SIM, "if front.0 == t {", "if front.0 == k {")]),
    ("B31", "the gross shadow (the allocator's input) is charged the cost preset", [(SIM,
        "sh_g: Shadow::new(k, 0.0, Financing::None),", "sh_g: Shadow::new(k, rate, Financing::None),")]),
    ("B32", "series digest ignores the share column", [(RES,
        "for col in [&r.share, &r.contrib, &r.shadow_ret_gross, &r.shadow_ret_cost] {", "for col in [&r.contrib, &r.shadow_ret_gross, &r.shadow_ret_cost] {")]),
    ("B33", "poisoning harness stops comparing one bar early (a one-bar look-ahead would pass)", [(HAR,
        "if a.clock_index[k] > through_clock_bar || b.clock_index[k] > through_clock_bar {",
        "if a.clock_index[k] >= through_clock_bar || b.clock_index[k] >= through_clock_bar {")]),
    ("B34", "truncation harness compares one bar less than it should", [(HAR, "let mut through = cut_bar;", "let mut through = cut_bar.saturating_sub(1);")]),
    ("B35", "a refused stateful step keeps the state it touched (no rollback)", [(STA, "self.state = backup;", "let _ = backup;")]),
    ("B36", "the overlay is shown the previous bar's equity", [(SIM,
        "equity: equity_pre, initial_equity: e0 });", "equity: e_prev, initial_equity: e0 });")]),
    ("B37", "an overlay halt does not flatten the book", [(SIM,
        'weights: if halted { &zero_w[s] } else { sl[s].standing.as_ref().expect("filtered") },',
        'weights: sl[s].standing.as_ref().expect("filtered"),')]),
    ("B38", "legacy sum of sub-accounts ignores the sub-account capital", [(SIM,
        "c += sub_cap[s] * sl[s].sh_c.cash;\n                    fin += sub_cap[s] * sh_fin_c[s];", "c += sl[s].sh_c.cash;\n                    fin += sub_cap[s] * sh_fin_c[s];")]),
    ("B39", "a shared instrument's contribution goes entirely to its first owner", [(SIM, "if tot > 0.0 {", "if false {")]),
    ("B40", "combine subtracts the later sleeves' weights", [(CON, "Some(a) => a + term,", "Some(a) => a - term,")]),
    ("B41", "capital base = max(equity, allocated) instead of the min", [(CON, "if a < equity {", "if a > equity {")]),
    ("B42", "budget sizing uses the cash before the sells' proceeds", [(CON, "let avail = cash_run;", "let avail = i.cash;")]),
    ("B43", "gross cap refuses at the cap (>=) instead of above it", [(CON, "if gross > cap * (1.0 + GROSS_CAP_REL_TOL) {", "if gross >= cap {")]),
    ("B44", "gross-cap tolerance loosened from 1e-12 to 1e-6", [(CON, "pub const GROSS_CAP_REL_TOL: f64 = 1e-12;", "pub const GROSS_CAP_REL_TOL: f64 = 1e-6;")]),
    ("B45", "inverse-volatility shares proportional to the volatility instead of its inverse", [(CON,
        "let invs: Vec<f64> = sds.iter().map(|sd| 1.0 / sd).collect();", "let invs: Vec<f64> = sds.iter().map(|sd| *sd).collect();")]),
    ("B46", "no allocator review on the last bar of the clock", [(SIM,
        "u + 1 == times.len() || !times[u + 1].date().same_month(times[u].date())", "u + 1 < times.len() && !times[u + 1].date().same_month(times[u].date())")]),
    ("B47", "the shadow's first (build) bar is published as a return row", [(SIM, "if !r.sh_g.first {", "if true {")]),
    ("B48", "construction ignores which instruments the driver planned (re-targets every named instrument)", [(CON, "if !i.planned[j] {", "if false {")]),
    ("B49", "financing accrues one extra calendar day per bar", [(SIM,
        "times[u - 1].date().days_until(date), cash, long_v, short_v", "times[u - 1].date().days_until(date) + 1, cash, long_v, short_v")]),
    ("B50", "a rule is asked one bar later than its declared minimum history", [(SIM, "&& t + 1 >= r.min_hist {", "&& t >= r.min_hist {")]),
    ("B51", "held weights are divided by the pre-cost equity", [(SIM, "let hw = units[j] * mark[j] / equity;", "let hw = units[j] * mark[j] / equity_pre;")]),
]


def run(cmd):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def failing_tests(out):
    names = re.findall(r"^test (\S+) \.\.\. FAILED$", out, flags=re.M)
    names += re.findall(r"^test (.+?) \.\.\. FAILED$", out, flags=re.M)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def read(path):
    with open(path, "r", newline="") as f:
        return f.read()


def write(path, text):
    with open(path, "w", newline="") as f:
        f.write(text)


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--check" in sys.argv:
        bad = 0
        for mid, desc, edits in MUTANTS:
            for rel, old, new in edits:
                n = read(os.path.join(CRATE, rel)).count(old)
                if n != 1:
                    print(f"{mid}: `old` text occurs {n}x in {rel}: {old[:70]!r}")
                    bad += 1
        print(f"{len(MUTANTS)} mutants, {bad} bad edits")
        sys.exit(1 if bad else 0)
    cmd = os.environ.get("WEIGHTSIM_BOOK_TEST_CMD", DEFAULT_CMD)
    code, out = run(cmd)
    if code != 0:
        print("BASELINE FAILED; refusing to mutate a red tree\n" + out[-3000:])
        sys.exit(2)
    print(f"baseline: green ({len(MUTANTS)} mutants)\n")
    survived = []
    print("| id | mutant | outcome | failing tests |")
    print("|---|---|---|---|")
    for mid, desc, edits in MUTANTS:
        if only and mid not in only:
            continue
        originals = {}
        ok = True
        for rel, old, new in edits:
            path = os.path.join(CRATE, rel)
            text = originals.get(path) or read(path)
            originals.setdefault(path, text)
        for rel, old, new in edits:
            if read(os.path.join(CRATE, rel)).count(old) != 1:
                ok = False
        if not ok:
            print(f"| {mid} | {desc} | BAD MUTANT (an `old` text does not occur exactly once) | |")
            survived.append(mid)
            continue
        try:
            for path, text in originals.items():
                cur = text
                for rel, old, new in edits:
                    if os.path.join(CRATE, rel) == path:
                        cur = cur.replace(old, new)
                write(path, cur)
            code, out = run(cmd)
        finally:
            for path, text in originals.items():
                write(path, text)
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
        extra = f" ...(+{len(fails) - 6})" if len(fails) > 6 else ""
        print(f"| {mid} | {desc} | {outcome} | {', '.join(fails[:6])}{extra} |", flush=True)
    print()
    print("SURVIVORS / INVALID:", survived if survived else "none")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
