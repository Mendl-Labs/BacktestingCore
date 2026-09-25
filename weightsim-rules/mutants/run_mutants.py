#!/usr/bin/env python3
"""Hand-written mutation testing for `weightsim-rules` (BACKTESTER_TRUTH_DESIGN.md 6.1, stage T3).

Mutants of the rule ADAPTERS and of the LADDER LOGIC: tolerances loosened, Tier II skipped, a mutant not applied,
the manifest check skipped, net and gross swapped, a date conversion off by one, the refusal mapping wrong, and so on.
Each mutant is ONE exact source edit (the old text must occur exactly once). For every mutant the script applies the
edit, runs the test command, records which tests FAILED (or that the build broke), and restores the file byte for byte.
A mutant that no test kills is reported as SURVIVED and the script exits non-zero.

Usage (from anywhere):
    WEIGHTSIM_RULES_TEST_CMD="cargo test --locked --no-fail-fast" python3 mutants/run_mutants.py [--only M03,M07] [--check]
`WEIGHTSIM_RULES_TEST_CMD` defaults to the command above and is run with the crate directory as its working directory.
`--check` only verifies that every mutant's old text occurs exactly once (no test run).
The real-data tests are env-gated (WEIGHTSIM_RULES_LADDER_DIR) and skip themselves when it is unset, so a table
produced without that variable is the one CI can reproduce (always-on tests only). With the variable set, the suite
additionally runs the real pinned data, which kills mutants of the mutant implementations that the synthetic data
cannot distinguish.
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

A = "src/adapters.rs"
C = "src/ladder/checks.rs"
F = "src/ladder/fixtures.rs"
M = "src/ladder/mod.rs"
R = "src/ladder/runner.rs"
U = "src/ladder/mutants.rs"
J = "src/ladder/json.rs"

# (id, description, file relative to the crate, old text (exactly once), new text)
MUTANTS = [
    # ------------------------------------------------------------------------------------------------ adapters
    ("M01", "date conversion off by one day (to_naive)", A,
     '        .expect("a weightsim::Date is always a valid calendar date")',
     '        .and_then(|n| n.succ_opt())\n        .expect("a weightsim::Date is always a valid calendar date")'),
    ("M02", "date conversion clamps the day of month (from_naive)", A,
     "Date::new(n.year(), n.month() as u8, n.day() as u8)",
     "Date::new(n.year(), n.month() as u8, n.day().min(28) as u8)"),
    ("M03", "error mapping: InsufficientHistory is a Data refusal instead of Warmup", A,
     "RuleError::InsufficientHistory { .. } => RefusalKind::Warmup,",
     "RuleError::InsufficientHistory { .. } => RefusalKind::Data,"),
    ("M04", "error mapping: DataGap is Other instead of Data", A,
     "RuleError::DataGap { .. } | RuleError::StaleData { .. } => RefusalKind::Data,",
     "RuleError::StaleData { .. } => RefusalKind::Data,"),
    ("M05", "error mapping: StaleData is Other instead of Data", A,
     "RuleError::DataGap { .. } | RuleError::StaleData { .. } => RefusalKind::Data,",
     "RuleError::DataGap { .. } => RefusalKind::Data,"),
    ("M06", "error mapping: every other error is Data instead of Other", A,
     "        _ => RefusalKind::Other,\n    };\n    RuleRefusal::new(kind, code, message)",
     "        _ => RefusalKind::Data,\n    };\n    RuleRefusal::new(kind, code, message)"),
    ("M07", "error mapping: wrong stable code for DataGap", A,
     'RuleError::DataGap { .. } => "data_gap",',
     'RuleError::DataGap { .. } => "gap",'),
    ("M08", "ETF adapter uses the live month-end mode (NextMonthBar) instead of Explicit", A,
     "&Options::etf_replay(MonthEndMode::Explicit)",
     "&Options::etf_replay(MonthEndMode::NextMonthBar)"),
    ("M09", "crypto adapter keeps the strict live gap policy instead of Unchecked", A,
     "Options { gap_policy: GapPolicy::Unchecked, ..Options::crypto_replay() }",
     "Options::crypto_replay()"),
    ("M10", "ETF adapter decides daily instead of at month-ends", A,
     "        DecisionSchedule::LastBarOfMonth\n    }",
     "        DecisionSchedule::Daily\n    }"),
    ("M11", "ETF adapter restores weights every bar instead of drifting", A,
     "        RebalancePolicy::OnDecision\n    }",
     "        RebalancePolicy::EveryBar\n    }"),
    ("M12", "crypto adapter drifts instead of restoring weights every bar", A,
     "        RebalancePolicy::EveryBar\n    }",
     "        RebalancePolicy::OnDecision\n    }"),
    ("M13", "crypto adapter asks the rule one bar early (99 bars of history)", A,
     "        CRYPTO_SMA_DAYS\n    }",
     "        CRYPTO_SMA_DAYS - 1\n    }"),
    ("M14", "adapter hands the rule its price columns in reverse order", A,
     "(0..h.n_assets()).map(|i| h.closes(i)).collect()",
     "(0..h.n_assets()).rev().map(|i| h.closes(i)).collect()"),
    ("M15", "FlatUntil trades one bar early (< becomes <=)", A,
     "if h.date() < self.first_trade {",
     "if h.date() <= self.first_trade {"),
    ("M16", "FlatUntil never holds cash", A,
     "if h.date() < self.first_trade {",
     "if false && h.date() < self.first_trade {"),
    # ------------------------------------------------------------------------------------------- tier checks
    ("M20", "Tier II tolerance loosened from 1e-9 to 1e-6", C,
     "pub const TIER2_TOL: f64 = 1e-9;", "pub const TIER2_TOL: f64 = 1e-6;"),
    ("M21", "Tier I correlation band loosened to 0.90", C,
     "pub const TIER1_CORR_MIN: f64 = 0.99;", "pub const TIER1_CORR_MIN: f64 = 0.90;"),
    ("M22", "Tier I Sharpe band loosened to 0.5", C,
     "pub const TIER1_D_SHARPE_MAX: f64 = 0.05;", "pub const TIER1_D_SHARPE_MAX: f64 = 0.5;"),
    ("M23", "Tier I CAGR band loosened to 5 pp", C,
     "pub const TIER1_D_CAGR_PP_MAX: f64 = 0.5;", "pub const TIER1_D_CAGR_PP_MAX: f64 = 5.0;"),
    ("M24", "Tier I trades band loosened to 50%", C,
     "pub const TIER1_TRADES_REL_MAX: f64 = 0.05;", "pub const TIER1_TRADES_REL_MAX: f64 = 0.5;"),
    ("M25", "Tier III cell tolerance loosened to 1e-3", C,
     "pub const TIER3_CELL_TOL: f64 = 1e-6;", "pub const TIER3_CELL_TOL: f64 = 1e-3;"),
    ("M26", "Tier III agreement floor lowered to 0.5", C,
     "pub const TIER3_MIN_AGREEMENT: f64 = 0.98;", "pub const TIER3_MIN_AGREEMENT: f64 = 0.5;"),
    ("M27", "dCAGR not converted to percentage points", C,
     "let d_cagr_pp = 100.0 * (rm.cagr - km.cagr);", "let d_cagr_pp = 1.0 * (rm.cagr - km.cagr);"),
    ("M28", "Tier II skipped (always passes)", C,
     "let mut tier2_pass = within(max_abs_ret_diff, TIER2_TOL);", "let mut tier2_pass = true;"),
    ("M29", "Tier II ignores held weights", C,
     "max_abs_traded_diff, max_abs_w_target_diff, max_abs_w_held_diff]\n",
     "max_abs_traded_diff, max_abs_w_target_diff]\n"),
    ("M30", "Tier II boundary exclusive (< instead of <=)", C,
     "    d <= tol\n", "    d < tol\n"),
    ("M31", "Tier III agreement boundary exclusive", C,
     "pass: agreement >= TIER3_MIN_AGREEMENT", "pass: agreement > TIER3_MIN_AGREEMENT"),
    ("M32", "trades band boundary exclusive", C,
     "(rel, rel <= TIER1_TRADES_REL_MAX)", "(rel, rel < TIER1_TRADES_REL_MAX)"),
    ("M33", "failed_tiers never reports Tier II", C,
     "        if !self.tier2_pass {", "        if false && !self.tier2_pass {"),
    ("M34", "a run may cover fewer bars than the key (covers_key_exactly loosened)", C,
     "self.common_days == self.key_days && self.common_days == self.run_days",
     "self.common_days == self.run_days"),
    ("M35", "common-day join skips the wrong side", C,
     "        } else if a[i] < b[j] {", "        } else if a[i] > b[j] {"),
    # --------------------------------------------------------------------------------------- fixtures / manifest
    ("M40", "manifest pin not checked", F,
     "if !manifest_sha.eq_ignore_ascii_case(pins.manifest_sha256) {",
     "if false && !manifest_sha.eq_ignore_ascii_case(pins.manifest_sha256) {"),
    ("M41", "listed files' sha256 not checked", F,
     "if !got.eq_ignore_ascii_case(want) {", "if false && !got.eq_ignore_ascii_case(want) {"),
    ("M42", "listed files' size not checked", F,
     "if l != bytes.len() as f64 {", "if false && l != bytes.len() as f64 {"),
    ("M43", "required-file rule skipped", F,
     "if !verified.contains_key(req) {", "if false && !verified.contains_key(req) {"),
    ("M44", "candles pin not checked (manifest-only trust)", F,
     "if !candles_sha.eq_ignore_ascii_case(pins.candles_sha256) {",
     "if false && !candles_sha.eq_ignore_ascii_case(pins.candles_sha256) {"),
    ("M45", "S1 key trade counter read from the S3 entry", F,
     '&metrics_json, &["S1", "flips"]', '&metrics_json, &["S3", "flips"]'),
    ("M46", "excluded key bars silently accepted", F,
     'if f[c_excl] != "0" {', 'if false && f[c_excl] != "0" {'),
    # ------------------------------------------------------------------------------------------------ ladder
    ("M50", "net compared against the GROSS key (net/gross swapped)", M,
     "let cmp = compare(&key_rows(key, basis), rows)?;",
     "let cmp = compare(&key_rows(key, Basis::Gross), rows)?;"),
    ("M51", "window check skipped", M,
     "b.window_matches_key && c.covers_key_exactly(),", "true,"),
    ("M52", "shadow (original key) check skipped", M,
     "vs_shadow <= checks::TIER2_TOL,", "true,"),
    ("M53", "key self-consistency tolerance loosened to 1", M,
     "pub const KEY_SELF_CONSISTENCY_TOL: f64 = 1e-8;", "pub const KEY_SELF_CONSISTENCY_TOL: f64 = 1.0;"),
    ("M54", "cost identity gap not checked", M,
     "ci.max_bar_cost_error <= 1e-15 && total_rel <= 1e-12 && ci.relative_gap <= COST_IDENTITY_MAX_REL_GAP,",
     "ci.max_bar_cost_error <= 1e-15 && total_rel <= 1e-12,"),
    ("M55", "mutants.json caught_by not compared", M,
     "    if got != want {", "    if false && got != want {"),
    ("M56", "mutants.json escapes_tier1 not compared", M,
     "    if cmp.bands_pass != e.escapes_tier1 {", "    if false && cmp.bands_pass != e.escapes_tier1 {"),
    ("M57", "canaries always pass", M,
     "(m.cmp.run_sharpe - centre).abs() <= tol,", "true,"),
    ("M58", "causality: poisoning always clean", M,
     "        poison_clean,\n", "        true,\n"),
    ("M59", "causality: rule truncation always clean", M,
     "        bad.is_empty(),\n", "        true,\n"),
    ("M60", "determinism check always clean", M,
     "        det.is_empty(),\n", "        true,\n"),
    ("M61", "an uncaught mutant does not fail Tier IV", M,
     "        !caught_by.is_empty(),\n", "        true,\n"),
    ("M62", "Tier I verdict ignores the trades band", M,
     "tier1_pass: cmp.bands_pass && trades_ok,", "tier1_pass: cmp.bands_pass,"),
    # ------------------------------------------------------------------------------------------------ runner
    ("M70", "key rows use the weights of the SAME bar instead of the bar before", R,
     "sim.row(&sim.target_weights, i - 1).to_vec()", "sim.row(&sim.target_weights, i).to_vec()"),
    ("M71", "key rows use held weights of the wrong bar", R,
     "sim.row(&sim.held_weights, i - 1).to_vec()", "sim.row(&sim.held_weights, i).to_vec()"),
    ("M72", "window opens on the first decision bar instead of the bar after", R,
     ".max(first_decision + 1);", ".max(first_decision);"),
    ("M73", "cost column not normalised by the pre-cost equity", R,
     "sim.cost[i] / pre(i)", "sim.cost[i]"),
    ("M74", "entry bar is the key's first bar itself", R,
     ".find(|&d| d < first)", ".find(|&d| d <= first)"),
    ("M75", "flip baseline is the first decision instead of the last one before the window", R,
     "sim.dates[t] < first_counted", "sim.dates[t] < sim.dates[bars[0]]"),
    # ------------------------------------------------------------------------------------------ mutant rules
    ("M80", "same-day-peek mutant does not peek", U,
     "(h.len() + 1).min(self.full.n_bars())", "h.len().min(self.full.n_bars())"),
    ("M81", "one-bar-late mutant is not late (delay 0)", U,
     "&EtfTrendRule, &fx.s1, 1)", "&EtfTrendRule, &fx.s1, 0)"),
    ("M82", "extra-delay mutant is not delayed", U,
     "&CryptoTrendRule, &fx.s3, 1)", "&CryptoTrendRule, &fx.s3, 0)"),
    ("M83", "half-sizing mutant sizes at 100%", U,
     'name: "mutant_s3_half_sizing", scale: 0.5,', 'name: "mutant_s3_half_sizing", scale: 1.0,'),
    ("M84", "S3 SMA-excludes-today mutant includes today after all", U,
     "for &v in &c[n - 101..n - 1] {", "for &v in &c[n - 100..n] {"),
    ("M85", "S1 SMA-excludes-current mutant includes the current month-end", U,
     "let prev = &me[me.len() - 11..me.len() - 1];", "let prev = &me[me.len() - 10..me.len()];"),
    ("M86", "drifting sub-accounts use the same-bar signal", U,
     "base.row(&base.target_weights, t - 1)[a] / 0.5", "base.row(&base.target_weights, t)[a] / 0.5"),
    # ------------------------------------------------------------------------------------------------- json
    ("M90", "json: trailing garbage accepted", J,
     "    if p.i != p.b.len() {", "    if false && p.i != p.b.len() {"),
    ("M91", "json: duplicate keys accepted", J,
     "            if kv.iter().any(|(e, _)| *e == k) {", "            if false && kv.iter().any(|(e, _)| *e == k) {"),
    ("M92", "json: nesting depth unbounded", J,
     "        if self.depth > 64 {", "        if self.depth > 6400 {"),
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
    cmd = os.environ.get("WEIGHTSIM_RULES_TEST_CMD", "cargo test --locked --no-fail-fast")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--check" in sys.argv:
        bad = []
        for mid, desc, rel, old, new in MUTANTS:
            with open(os.path.join(CRATE, rel), "r", newline="") as f:
                n = f.read().count(old)
            if n != 1 or old == new:
                bad.append((mid, n))
        print("mutants:", len(MUTANTS), "bad:", bad if bad else "none")
        sys.exit(1 if bad else 0)
    code, out = run(cmd)
    if code != 0:
        print("BASELINE FAILED; refusing to mutate a red tree\n" + out[-3000:])
        sys.exit(2)
    print("baseline: green\n")
    survived = []
    killed = 0
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
            killed += 1
        elif compile_err:
            outcome = "NOT COMPILABLE (invalid mutant)"
            survived.append(mid)
        elif code != 0:
            outcome = "KILLED (build/test aborted)"
            killed += 1
        else:
            outcome = "SURVIVED"
            survived.append(mid)
        print(f"| {mid} | {desc} | {outcome} | {', '.join(fails[:6])}{' ...(+%d)' % (len(fails) - 6) if len(fails) > 6 else ''} |", flush=True)
    print()
    print(f"killed {killed} of {killed + len(survived)}")
    print("SURVIVORS / INVALID:", survived if survived else "none")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
