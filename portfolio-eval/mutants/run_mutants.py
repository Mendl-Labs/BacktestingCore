#!/usr/bin/env python3
"""Hand-written mutation testing for `portfolio-eval` (same pattern as weightsim/mutants/run_mutants.py).

Each mutant is ONE exact source edit (the old text must occur exactly once in its file). For every mutant the script
applies the edit, runs the test command, records which tests FAILED (or that the build broke), and restores the file
byte for byte. A mutant that no test kills is reported as SURVIVED and the script exits non-zero. Only the always-on
tests count: `#[ignore]`d tests (the full power-table regeneration) are not run.

Usage (from anywhere):
    python3 portfolio-eval/mutants/run_mutants.py [--only M03,M07] [--list]
    PORTFOLIO_EVAL_TEST_CMD="cargo test --manifest-path portfolio-eval/Cargo.toml --locked" python3 ...

`PORTFOLIO_EVAL_TEST_CMD` defaults to the command above (fail-fast: the first failing test binary ends the run, which
is enough to kill a mutant) and is run with the repository root as its working directory. The output is a markdown
table of mutant, outcome and the names of the failing tests.
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CRATE)

# (id, description, file relative to the crate, old text (exactly once), new text)
MUTANTS = [
    # ---- folds: purge / embargo
    ("M01", "walk-forward purge off by one (train ends one bar too late)", "src/folds.rs",
     "let train_end = test_start - purge;",
     "let train_end = test_start - purge + 1;"),
    ("M02", "K-fold/CPCV purge off by one (one purge bar too few before a test range)", "src/folds.rs",
     "r.start.saturating_sub(purge)..",
     "r.start.saturating_sub(purge.saturating_sub(1)).."),
    ("M03", "K-fold/CPCV embargo off by one (one embargo bar too few after a test range)", "src/folds.rs",
     "r.end.saturating_add(embargo).min(n_bars)",
     "r.end.saturating_add(embargo.saturating_sub(1)).min(n_bars)"),
    ("M04", "embargo ignored entirely", "src/folds.rs",
     "r.end.saturating_add(embargo).min(n_bars)",
     "r.end.min(n_bars)"),
    ("M05", "purge/embargo not capped at half a test window", "src/folds.rs",
     "raw.min(test_len / 2)",
     "raw"),
    ("M06", "purge/embargo lower bound of 5 bars dropped", "src/folds.rs",
     "MIN_PURGE_BARS.max(rebalance_interval_bars)",
     "0usize.max(rebalance_interval_bars)"),
    ("M07", "CPCV adjacent test groups not merged into one block", "src/folds.rs",
     "Some(last) if last.end == r.start => last.end = r.end,",
     "Some(last) if last.end == r.start && false => last.end = r.end,"),
    ("M08", "walk-forward test windows do not end at the last bar", "src/folds.rs",
     "let first_test = n_bars - total_test;",
     "let first_test = n_bars - total_test - 1;"),
    # ---- bootstrap
    ("M09", "bootstrap block length wrong (p = 1/(b+1))", "src/bootstrap.rs",
     "Some(((1.0 / mean_block) * 18_446_744_073_709_551_616.0) as u64)",
     "Some(((1.0 / (mean_block + 1.0)) * 18_446_744_073_709_551_616.0) as u64)"),
    ("M10", "stationary bootstrap does not wrap circularly", "src/bootstrap.rs",
     "if nx == self.n {\n                                0\n                            } else {",
     "if nx == self.n {\n                                self.n - 1\n                            } else {"),
    ("M11", "percentile interval uses the wrong tail mass", "src/bootstrap.rs",
     "let tail = (1.0 - level) / 2.0;",
     "let tail = 1.0 - level;"),
    ("M12", "marginal test ignores the requested fixed block length", "src/marginal.rs",
     "BlockLength::Fixed(b) => b,",
     "BlockLength::Fixed(_) => 1.0,"),
    ("M13", "automatic block length looks at the base book only", "src/marginal.rs",
     "auto_block_length(&[base, combined, &diff])?",
     "auto_block_length(&[base])?"),
    # ---- marginal test
    ("M14", "missing equal-volatility scaling in the point estimate (raw mean difference)", "src/marginal.rs",
     "let delta_hat = (m1 / s1 - m0 / s0) * root_ppy;",
     "let delta_hat = (m1 - m0) * root_ppy;"),
    ("M15", "missing equal-volatility scaling inside the bootstrap replicates", "src/marginal.rs",
     "reps.push((mean1 / rs1 - mean0 / rs0) * root_ppy);",
     "reps.push((mean1 - mean0) * root_ppy);"),
    ("M16", "one-sided p-value computed as the two-sided one", "src/marginal.rs",
     "let p_one_sided = (1.0 + ge as f64) / (nv as f64 + 1.0);",
     "let p_one_sided = (1.0 + ge_abs as f64) / (nv as f64 + 1.0);"),
    ("M17", "p-value without the +1 correction (can be exactly 0)", "src/marginal.rs",
     "let p_one_sided = (1.0 + ge as f64) / (nv as f64 + 1.0);",
     "let p_one_sided = ge as f64 / (nv as f64 + 1.0);"),
    ("M18", "bootstrap distribution not centred at the estimate (null not imposed)", "src/marginal.rs",
     "let centred = v - delta_hat;",
     "let centred = *v;"),
    ("M19", "ties not counted in the bootstrap p-value (>= becomes >)", "src/marginal.rs",
     "if centred >= delta_hat {",
     "if centred > delta_hat {"),
    ("M20", "bootstrap replicate variance uses ddof 0", "src/marginal.rs",
     "let v0 = (b0 - a0 * a0 / nf) / (nf - 1.0);\n                let v1 = (b1 - a1 * a1 / nf) / (nf - 1.0);",
     "let v0 = (b0 - a0 * a0 / nf) / nf;\n                let v1 = (b1 - a1 * a1 / nf) / nf;"),
    ("M21", "MDE uses alpha/2 instead of alpha (two-sided z)", "src/marginal.rs",
     "Ok((detmath::norm_isf(alpha) + detmath::norm_ppf(power)) * se)",
     "Ok((detmath::norm_isf(alpha / 2.0) + detmath::norm_ppf(power)) * se)"),
    ("M22", "ex-ante volatilities swapped between the books", "src/marginal.rs",
     "Ok((base_vol, combined_vol))",
     "Ok((combined_vol, base_vol))"),
    ("M23", "minimum sample size for the marginal test lowered", "src/marginal.rs",
     "pub const MIN_MARGINAL_OBS: usize = 30;",
     "pub const MIN_MARGINAL_OBS: usize = 10;"),
    # ---- HAC / spanning
    ("M24", "Newey-West kernel weight wrong (1 - l/L instead of 1 - l/(L+1))", "src/hac.rs",
     "let w = 1.0 - k as f64 / (l as f64 + 1.0);",
     "let w = 1.0 - k as f64 / l as f64;"),
    ("M25", "HAC drops the last lag (off by one)", "src/hac.rs",
     "for k in 1..=l {",
     "for k in 1..l {"),
    ("M26", "automatic HAC lag rule constant wrong (5 instead of 4)", "src/hac.rs",
     "(4.0 * detmath::pow(n as f64 / 100.0, 2.0 / 9.0)).floor() as usize",
     "(5.0 * detmath::pow(n as f64 / 100.0, 2.0 / 9.0)).floor() as usize"),
    ("M27", "spanning one-sided p-value uses the wrong tail", "src/hac.rs",
     "p_one_sided: detmath::norm_sf(t_alpha),",
     "p_one_sided: detmath::norm_cdf(t_alpha),"),
    ("M28", "spanning residual variance ddof (divides by n, not n - k - 1)", "src/hac.rs",
     "let resid_var = sse / (n - p) as f64;",
     "let resid_var = sse / n as f64;"),
    # ---- statistics
    ("M29", "sample variance ddof 0 instead of 1", "src/stats.rs",
     "ss / (x.len() - 1) as f64",
     "ss / x.len() as f64"),
    ("M30", "MAD scale constant dropped (raw MAD, not normal-consistent)", "src/stats.rs",
     "Ok(1.4826 * median(&dev)?)",
     "Ok(1.0 * median(&dev)?)"),
    # ---- DSR / BH
    ("M31", "expected-maximum weights of the two quantile terms swapped", "src/dsr.rs",
     "(1.0 - EULER_GAMMA) * detmath::norm_isf(1.0 / n) + EULER_GAMMA * detmath::norm_isf(1.0 / (n * std::f64::consts::E))",
     "EULER_GAMMA * detmath::norm_isf(1.0 / n) + (1.0 - EULER_GAMMA) * detmath::norm_isf(1.0 / (n * std::f64::consts::E))"),
    ("M32", "DSR variance floor removed", "src/dsr.rs",
     "let v = if floor_at_normal { raw.max(baseline) } else { raw };",
     "let v = raw;"),
    # (the mirrored mutant norm_ppf -> norm_isf only flips the sign of z, which is squared: an EQUIVALENT mutant, so it
    # is not used)
    ("M33", "MinTRL uses the wrong confidence level in the normal quantile", "src/dsr.rs",
     "let z = detmath::norm_ppf(confidence) / (sharpe - benchmark);",
     "let z = detmath::norm_ppf(confidence - 0.05) / (sharpe - benchmark);"),
    ("M34", "BH q-values divide by the 0-based rank", "src/dsr.rs",
     "(indexed[rank].1 * m as f64 / (rank as f64 + 1.0)).min(1.0)",
     "(indexed[rank].1 * m as f64 / (rank as f64 + 1e-9)).min(1.0)"),
    # ---- trial ledger / holdout
    ("M35", "trial count not incremented for a new configuration", "src/ledger.rs",
     "self.trials.insert(id.to_string(), value);",
     "let _ = &value;"),
    ("M36", "universe variants ignored in K_effective", "src/ledger.rs",
     ".saturating_mul(self.universe_variants)",
     ".saturating_mul(1)"),
    ("M37", "lineage prior trials ignored in K_effective", "src/ledger.rs",
     ".saturating_add(self.lineage_prior)",
     ".saturating_add(0)"),
    ("M38", "a repeated configuration id overwrites the first recorded Sharpe", "src/ledger.rs",
     "if existing.is_none() && value.is_some() {",
     "if value.is_some() {"),
    ("M39", "holdout reuse not flagged as post hoc", "src/ledger.rs",
     "Some(t) => HoldoutLook { first_look: false, post_hoc: true, looks: self.looks, spent_at_ms: t },",
     "Some(t) => HoldoutLook { first_look: false, post_hoc: false, looks: self.looks, spent_at_ms: t },"),
    ("M40", "holdout overlap check treats touching ranges as overlapping", "src/ledger.rs",
     "if r.start < r.end && r.start < self.range.end && self.range.start < r.end {",
     "if r.start < r.end && r.start <= self.range.end && self.range.start <= r.end {"),
    # ---- PBO
    ("M41", "PBO counts lambda < 0 instead of lambda <= 0", "src/pbo.rs",
     "logits.iter().filter(|l| **l <= 0.0).count() as f64 / ns",
     "logits.iter().filter(|l| **l < 0.0).count() as f64 / ns"),
    ("M42", "PBO in-sample ties resolved to the last configuration", "src/pbo.rs",
     "if is_p[i] > is_p[best] {",
     "if is_p[i] >= is_p[best] {"),
    ("M43", "PBO drops the most recent remainder rows instead of the oldest", "src/pbo.rs",
     "let start = t - block_len * n_blocks;",
     "let start = 0;"),
    # ---- primitives and Monte Carlo
    ("M44", "erfc reflection sign error for negative arguments", "src/detmath.rs",
     "return 2.0 - erfc(-x);",
     "return erfc(-x);"),
    ("M45", "xoshiro256** state rotation constant wrong", "src/rng.rs",
     "self.s[3] = self.s[3].rotate_left(45);",
     "self.s[3] = self.s[3].rotate_left(44);"),
    ("M46", "power table: candidate/base correlation structure wrong", "src/power.rs",
     "let cross = (1.0 - rho * rho).sqrt();",
     "let cross = (1.0 - rho).sqrt();"),
    ("M47", "power table: candidate Sharpe solved with the wrong share term", "src/power.rs",
     "- (1.0 - w) * self.base_sharpe) / w",
     "- w * self.base_sharpe) / w"),
]


def run(cmd):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    cmd = os.environ.get("PORTFOLIO_EVAL_TEST_CMD", "cargo test --manifest-path portfolio-eval/Cargo.toml --locked")
    if "--list" in sys.argv:
        for mid, desc, rel, _, _ in MUTANTS:
            print(f"{mid}  {rel:18} {desc}")
        print(f"{len(MUTANTS)} mutants")
        return
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
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
        shown = ", ".join(fails[:6]) + (" ...(+%d)" % (len(fails) - 6) if len(fails) > 6 else "")
        print(f"| {mid} | {desc} | {outcome} | {shown} |", flush=True)
    print()
    print(f"killed {killed} of {killed + len(survived)}")
    print("SURVIVORS / INVALID:", survived if survived else "none")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
