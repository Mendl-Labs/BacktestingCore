#!/usr/bin/env python3
"""Hand-written mutation testing for `portfolio-construct` (design 6.5 PF2 row, Tier IV).

Each mutant is ONE exact source edit (the old text must occur exactly once). For every mutant the script applies the edit,
runs the test command, records which tests FAILED (or that the build broke), and restores the file byte for byte. A mutant
that no test kills is reported as SURVIVED and the script exits non-zero.

Usage (from anywhere):
    PORTFOLIO_CONSTRUCT_TEST_CMD="cargo test --manifest-path portfolio-construct/Cargo.toml --locked --no-fail-fast" \\
        python3 portfolio-construct/mutants/run_mutants.py [--only M03,M07] [--validate]
`PORTFOLIO_CONSTRUCT_TEST_CMD` defaults to the command above and is run with the Core repository root as its working
directory. `--validate` only checks that every old text occurs exactly once (no build). The output is a markdown table.
All tests are always-on (no vendor data, no environment gates), so a table produced locally is the one CI reproduces.
The pinned-digest test (`digests_match_the_pinned_goldens`) is SKIPPED here on purpose: it would "kill" any mutant that changes a
single bit and so hide mutants that no semantic test notices.
"""
import os
import re
import subprocess
import sys

CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CRATE)

# (id, description, file relative to the crate, old text (exactly once), new text)
MUTANTS = [
    # ---- the design's 6.5 portfolio-construction list -------------------------------------------------------------------
    ("M01", "wrong capital base: max(equity, allocated) instead of min", "src/construct.rs",
     "Some(a) => equity.min(a),", "Some(a) => equity.max(a),"),
    ("M02", "wrong capital base: the allocation cap is ignored", "src/construct.rs",
     "Some(a) => equity.min(a),", "Some(_) => equity,"),
    ("M03", "risk scale ignored", "src/construct.rs",
     "let rs = i.risk_scale.product();", "let rs = 1.0;"),
    ("M04", "ladder factor dropped from the risk scale", "src/construct.rs",
     "self.approval_constant * self.ladder", "self.approval_constant"),
    ("M05", "trade filter percentage band measured on the CURRENT holding instead of the target", "src/filter.rs",
     "let reference = if target == 0.0 { current.abs() } else { target.abs() };", "let reference = current.abs();"),
    ("M06", "netting by absolute instead of signed sum", "src/construct.rs",
     "Some(acc) => acc + term,", "Some(acc) => acc.abs() + term.abs(),"),
    ("M07", "target rounding AWAY from zero for shorts (ceil of the magnitude)", "src/num.rs",
     "-floor_dp(-x, dp)", "-ceil_dp(-x, dp)"),
    ("M08", "margin rate floor dropped: Alpaca uses the asset requirement as is", "src/margin.rs",
     "Some(asset.max(crate::ALPACA_MIN_OPENING_MARGIN_RATE))", "Some(asset)"),
    ("M09", "gross cap CLIPPED instead of refused", "src/construct.rs",
     "        return Err(ConstructRefusal::GrossAboveCap { gross, cap: gross_cap });",
     "        let k = gross_cap / gross;\n        for l in lines.iter_mut() {\n            l.target_notional *= k;\n        }"),
    ("M10", "gross cap not enforced at all", "src/construct.rs",
     "if (refuse_all || signed_plan) && exceeds(gross, gross_cap) {",
     "if false && (refuse_all || signed_plan) && exceeds(gross, gross_cap) {"),
    ("M11", "share ignored in the combine (weights summed as if share 1)", "src/construct.rs",
     "let term = s.share * w;", "let term = w;"),
    # ---- boundaries and tie handling --------------------------------------------------------------------------------------
    ("M12", "cap boundary: a value AT the cap refuses (>= instead of > with tolerance)", "src/num.rs",
     "value > cap + cap.abs() * EDGE_TOL", "value >= cap"),
    ("M13", "cap tolerance dropped: float noise above an exact-decimal tie refuses", "src/num.rs",
     "value > cap + cap.abs() * EDGE_TOL", "value > cap"),
    ("M14", "trade filter: a delta exactly at min_abs is dropped (<= instead of <)", "src/filter.rs",
     "if below(abs_delta, self.min_abs) {", "if abs_delta <= self.min_abs {"),
    ("M15", "trade filter: absolute minimum ignored", "src/filter.rs",
     "if below(abs_delta, self.min_abs) {", "if false && below(abs_delta, self.min_abs) {"),
    ("M16", "trade filter: percentage minimum ignored", "src/filter.rs",
     "if below(abs_delta, pct_min) {", "if false && below(abs_delta, pct_min) {"),
    ("M17", "rounding: 8-decimal snap removed (a value 4 ulps under a quantum floors to the quantum below)", "src/num.rs",
     "(y + y * SNAP_ULPS * f64::EPSILON).floor() / s", "y.floor() / s"),
    ("M18", "rounding: floor replaced by round-to-nearest", "src/num.rs",
     "(y + y * SNAP_ULPS * f64::EPSILON).floor() / s", "y.round() / s"),
    ("M19", "lot rounding: minimum order value ignored", "src/rounding.rs",
     "if q * price < rule.min_notional {", "if false && q * price < rule.min_notional {"),
    ("M20", "reserve rounded DOWN instead of up", "src/construct.rs",
     "let reserve = ceil_dp(reserve_fraction * cb, 8);", "let reserve = crate::num::floor_dp(reserve_fraction * cb, 8);"),
    # ---- limits, shorting, margin ------------------------------------------------------------------------------------------
    ("M21", "shorting check dropped", "src/construct.rs",
     "        if !lim.shorting {", "        if false && !lim.shorting {"),
    ("M22", "net cap measured on the signed sum, not its magnitude", "src/construct.rs",
     "if exceeds(net.abs(), net_cap) {", "if exceeds(net, net_cap) {"),
    ("M23", "position cap measured on the signed target, not its magnitude", "src/construct.rs",
     "if exceeds(l.target_notional.abs(), pos_cap) {", "if exceeds(l.target_notional, pos_cap) {"),
    ("M24", "asset-class cap ignores the capital base", "src/construct.rs",
     "let cap = frac * cb;", "let cap = frac * 1.0;"),
    ("M25", "OANDA margin nets long against short (offsets) instead of summing per position", "src/margin.rs",
     "terms.push(self.rate(inst)? * notional.abs());", "terms.push(self.rate(inst)? * notional);"),
    ("M26", "R3 default ceiling 50% of NAV raised to 100%", "src/lib.rs",
     "pub const OANDA_DEFAULT_MARGIN_CEILING: f64 = 0.50;", "pub const OANDA_DEFAULT_MARGIN_CEILING: f64 = 1.0;"),
    ("M27", "non-marginable Alpaca instrument charged the 50% floor instead of 100%", "src/margin.rs",
     "return Some(1.0);", "return Some(0.5);"),
    ("M28", "margin ceiling refusal removed", "src/construct.rs",
     "if exceeds(margin_used, ceiling) {", "if false && exceeds(margin_used, ceiling) {"),
    ("M29", "OANDA ceiling ignores the configured fraction", "src/margin.rs",
     "Some(self.ceiling_fraction_of_nav * equity)", "Some(equity)"),
    ("M30", "a margin plan on a cash budget is not refused (BuyingPowerRequired removed)", "src/construct.rs",
     "if signed_plan && needs_margin && matches!(i.funding, Funding::Cash { .. }) {",
     "if false && signed_plan && needs_margin && matches!(i.funding, Funding::Cash { .. }) {"),
    ("M31", "unmanaged positions dropped from the projected gross", "src/construct.rs",
     "let projected_gross = gross + i.unmanaged_gross;", "let projected_gross = gross;"),
    # ---- trades, legs, funding --------------------------------------------------------------------------------------------
    ("M32", "a trade that crosses zero is one order instead of two legs", "src/construct.rs",
     "let crossing = (current > 0.0 && target < 0.0) || (current < 0.0 && target > 0.0);", "let crossing = false;"),
    ("M33", "reductions classified as increases and vice versa (sells no longer first)", "src/construct.rs",
     "        for c in pending {\n            if c.reducing {", "        for c in pending {\n            if !c.reducing {"),
    ("M34", "open leg ignores the dust the close leg could not sell", "src/construct.rs",
     "let wished = if leg == LegKind::Open { wished + residual } else { wished };", "let wished = wished;"),
    ("M35", "a full exit is sized as |delta| / price (float dust) instead of the held quantity", "src/construct.rs",
     "            } else if target == 0.0 {\n                held_abs\n", "            } else if target == 0.0 {\n                abs_delta / price\n"),
    ("M36", "cash reserve ignored", "src/construct.rs",
     "let reserve = ceil_dp(reserve_fraction * cb, 8);", "let reserve = 0.0;"),
    ("M37", "sell proceeds credited when they should not be (flag inverted)", "src/construct.rs",
     "if credit_sell_proceeds {", "if !credit_sell_proceeds {"),
    ("M38", "increases are not scaled to fit the budget", "src/construct.rs",
     "let factor = if usable > 0.0 { usable / total } else { 0.0 };", "let factor = 1.0;"),
    ("M39", "fees left out of the budget total", "src/construct.rs",
     "                        let nt = q * c.price;\n                        nt + fee_for(nt, fee_rate)\n",
     "                        let nt = q * c.price;\n                        nt\n"),
    ("M40", "per-buy fee-rounding slack dropped from the budget", "src/construct.rs",
     "let usable = available - crate::FEE_SLACK_PER_BUY * increasers.len() as f64;", "let usable = available;"),
    ("M41", "an out-of-scope instrument is traded anyway", "src/construct.rs",
     "        if !inst.in_scope {", "        if false && !inst.in_scope {"),
    ("M42", "a zero price is accepted as a usable price", "src/construct.rs",
     "Some(p) if p.is_finite() && p > 0.0 => p,", "Some(p) if p.is_finite() => p,"),
    ("M43", "a held short in a long-only instrument is no longer skipped", "src/construct.rs",
     "if inst.held_units < 0.0 && !signed_inst[j] {", "if false && inst.held_units < 0.0 && !signed_inst[j] {"),
    ("M44", "sleeves combined in the caller's order instead of id order", "src/construct.rs",
     "    order.sort_by(|a, b| i.sleeves[*a].id.trim().cmp(i.sleeves[*b].id.trim()));", ""),
    ("M45", "cross-instrument sums added in the caller's order", "src/num.rs",
     "    v.sort_by(|a, b| a.total_cmp(b));\n", ""),
    # ---- ladder ---------------------------------------------------------------------------------------------------------------
    ("M46", "ladder: a rung is exclusive at the boundary (> instead of >=)", "src/ladder.rs",
     "loss >= threshold - threshold.abs() * EDGE_TOL", "loss > threshold + threshold.abs() * EDGE_TOL"),
    ("M47", "ladder: recovery hysteresis removed (release at the trigger itself)", "src/ladder.rs",
     "let threshold = (at * recovery) * reference;", "let threshold = at * reference;"),
    ("M48", "ladder: a halt is not sticky", "src/ladder.rs",
     "        if st.is_halted() {", "        if false && st.is_halted() {"),
    ("M49", "ladder: the high-water mark never ratchets up", "src/ladder.rs",
     "Some(h) if h >= equity => h,", "Some(h) => h,"),
    ("M50", "ladder: daily loss measured from the high-water mark", "src/ladder.rs",
     "daily_halt = breached(day_start, equity, self.daily_loss_limit);", "daily_halt = breached(hwm, equity, self.daily_loss_limit);"),
    ("M51", "ladder: daily loss ignored", "src/ladder.rs",
     "daily_halt = breached(day_start, equity, self.daily_loss_limit);", "daily_halt = false;"),
    ("M52", "ladder: recovery releases every rung at once instead of one at a time", "src/ladder.rs",
     "held = c.checked_sub(1);", "held = None;"),
    ("M53", "ladder: the drawdown reason is preferred over the daily-loss reason", "src/ladder.rs",
     "let why = if daily_halt { LadderHalt::DailyLoss } else { LadderHalt::DrawdownLadder };",
     "let why = if drawdown_halt { LadderHalt::DrawdownLadder } else { LadderHalt::DailyLoss };"),
    # ---- allocators and schedule ----------------------------------------------------------------------------------------------
    ("M54", "InverseVol reads one return beyond the visible history (look-ahead)", "src/alloc.rs",
     "let window = &rets[vis - lookback..vis];", "let window = &rets[vis + 1 - lookback..(vis + 1).min(rets.len())];"),
    ("M55", "InverseVol std with ddof 0", "src/alloc.rs",
     "(v / (n - 1) as f64).sqrt()", "(v / n as f64).sqrt()"),
    ("M56", "InverseVol proportional to volatility instead of inverse", "src/alloc.rs",
     "inv.push(1.0 / sd);", "inv.push(sd);"),
    ("M57", "InverseVol deviation floor ignored", "src/alloc.rs",
     "if floor > 0.0 && sd < floor {", "if false && floor > 0.0 && sd < floor {"),
    ("M58", "calendar: century rule dropped from the leap-year test", "src/schedule.rs",
     "(y % 4 == 0 && y % 100 != 0) || y % 400 == 0", "y % 4 == 0"),
    ("M59", "calendar: same_month ignores the year", "src/schedule.rs",
     "self.year == other.year && self.month == other.month", "self.month == other.month"),
    ("M60", "cadence: AllSleevesOnAnyDue plans sleeves whose market is closed", "src/schedule.rs",
     "tradable.iter().map(|t| any && *t).collect()", "tradable.iter().map(|_| any).collect()"),
    ("M61", "cadence: PerSleeve behaves like AllSleevesOnAnyDue", "src/schedule.rs",
     "BookCadence::PerSleeve => due.iter().zip(tradable).map(|(d, t)| *d && *t).collect(),",
     "BookCadence::PerSleeve => due.iter().zip(tradable).map(|(_, t)| any && *t).collect(),"),
    ("M62", "R2 approval headroom 0.9 raised to 1.0", "src/lib.rs",
     "pub const R2_HEADROOM: f64 = 0.9;", "pub const R2_HEADROOM: f64 = 1.0;"),
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
    cmd = os.environ.get(
        "PORTFOLIO_CONSTRUCT_TEST_CMD",
        "cargo test --manifest-path portfolio-construct/Cargo.toml --locked --no-fail-fast -- --skip digests_match_the_pinned_goldens",
    )
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--validate" in sys.argv:
        bad = []
        for mid, desc, rel, old, new in MUTANTS:
            with open(os.path.join(CRATE, rel), "r", newline="") as f:
                n = f.read().count(old)
            if n != 1:
                bad.append((mid, n))
        ids = [m[0] for m in MUTANTS]
        assert len(ids) == len(set(ids)), "duplicate ids"
        print("mutants:", len(MUTANTS), "bad:", bad if bad else "none")
        sys.exit(1 if bad else 0)
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
        print(f"| {mid} | {desc} | {outcome} | {', '.join(fails[:6])}{' ...(+%d)' % (len(fails) - 6) if len(fails) > 6 else ''} |", flush=True)
    print()
    print("SURVIVORS / INVALID:", survived if survived else "none")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
