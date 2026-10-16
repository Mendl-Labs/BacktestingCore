#!/usr/bin/env python3
"""Generate the SYNTHETIC book fixtures of `weightsim` v0.2 (design phase PF1) with an independent plain-Python account
engine that MIRRORS the semantics of the PF0 book key (`book_key.py` + `AMENDMENT_12.md`, private Engine repo).

WHY A MIRROR AND NOT THE KEY ITSELF. The real key reads vendor-derived candles (`ladder_candles.csv`) and writes vendor-derived
per-bar files; neither may enter this PUBLIC repository. `book_key.py` is also not runnable without the private T0 helpers and
pandas >= 3. This script re-implements the same account semantics (union clock with carry, combine, capital base, risk scale,
whole-book gross-cap refusal, trade filter, both cash policies, both cadence modes, frozen inverse-vol allocator, per-sleeve
shadow curves, contributions) in ~250 lines of stdlib Python, on SYNTHETIC prices generated here by an integer LCG, and writes
per-bar files in the SAME column layout as the key. Nothing here is vendor data.

THE MIRROR IS ITSELF CHECKED AGAINST THE REAL KEY. `--verify-real <replication_ladder_book dir> <replication_ladder dir>` runs
this engine on the real pinned candles and the real T0 decisions and compares every column of every per-bar key file, bit for
bit where the arithmetic is identical (`max|diff|` is printed; it is 0.0 on every file in the run recorded in the PR). So the
committed synthetic outputs are the outputs of an engine proven equal to the key on real data.

Modes
  python gen_book_key.py                      # write the synthetic fixtures next to this script (run ONCE; committed)
  python gen_book_key.py --verify-real KEYDIR LADDERDIR
                                              # read-only proof that this engine reproduces the real key files
Files written: synthetic_book_candles.csv, synthetic_netting_candles.csv, book_<config>_perbar.csv (+ single sleeve and netting),
synthetic_book_decisions.csv (the Python rules' decisions, which the Rust test rules must reproduce), synthetic_book_f1.csv (the
finding-F1 cadence numbers), synthetic_book_meta.csv (window and parameters), MANIFEST.sha256.
"""
import argparse
import bisect
import csv
import datetime
import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

CAPITAL0 = 100000.0  # currency capital used only to convert the planner's absolute minimum trade / allocated capital
FILTER = {"min_abs": 10.0, "min_pct": 0.02}
COST_RATE = 10.0 / 10000.0  # certification_flat_10bps_per_side
POLICY = {"etf": "on_decision", "cry": "every_bar"}


def seqsum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def std1(xs):
    n = len(xs)
    m = seqsum(xs) / n
    v = 0.0
    for x in xs:
        v += (x - m) * (x - m)
    return math.sqrt(v / (n - 1))


def fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


# ------------------------------------------------------------------------------------------------ account engine (mirror)
def month_end_review(clock, k):
    return k + 1 == len(clock) or clock[k + 1][:7] != clock[k][:7]


def run_engine(sleeves, prices, clock, cfg, rate, shadow_g=None):
    """One account run over `clock` (ISO dates, first = start bar). One dict per bar k >= 1 (the start bar is the initial build).
    sleeves: dict(id, symbols, cal_set, decisions{date: weights}, policy). prices: {symbol: {date: close}} (real bars only)."""
    S = len(sleeves)
    sids = [s["id"] for s in sleeves]
    syms, idx = [], {}
    for s in sleeves:
        for sym in s["symbols"]:
            if sym not in idx:
                idx[sym] = len(syms)
                syms.append(sym)
    n = len(syms)
    pos = [{sym: p for p, sym in enumerate(s["symbols"])} for s in sleeves]
    owners = [[si for si in range(S) if syms[j] in pos[si]] for j in range(n)]
    alloc = None if cfg["allocated_capital"] is None else cfg["allocated_capital"] / CAPITAL0
    rs = cfg["risk_scale"]
    max_gross = cfg["max_gross"]
    flt = cfg["trade_filter"]
    min_abs = None if flt is None else flt["min_abs"] / CAPITAL0
    min_pct = None if flt is None else flt["min_pct"]
    budget = cfg["cash_policy"] == "budget"
    cadence = cfg["cadence"]
    inv = cfg["allocator"]
    shares = {sid: cfg["shares"][sid] for sid in sids}
    stand = [None] * S
    units = [0.0] * n
    cash = 1.0
    E_prev = 1.0
    mark = [None] * n
    out = []

    def combined_w(stand_, shares_):
        w = [0.0] * n
        for j in range(n):
            terms = [shares_[sids[si]] * stand_[si][pos[si][syms[j]]] for si in owners[j] if stand_[si] is not None]
            if terms:
                s = 0.0
                for t in terms:
                    s += t
                w[j] = s * rs
        return w

    for k, d in enumerate(clock):
        has = [d in s["cal_set"] for s in sleeves]
        prev_mark = list(mark)
        for j in range(n):
            close = prices[syms[j]][d] if any(has[si] for si in owners[j]) else None
            mark[j] = prev_mark[j] if close is None else close
            assert mark[j] is not None, (d, syms[j])
        if k == 0:
            assert all(has), f"start bar {d} is not a bar of every sleeve"
            E_pre = 1.0
            w_open = [0.0] * n
        else:
            mv = 0.0
            for j in range(n):
                mv += units[j] * mark[j]
            E_pre = cash + mv
            w_open = [units[j] * prev_mark[j] / E_prev for j in range(n)]
        w_target_force = combined_w(stand, shares)
        shares_in_force = dict(shares)
        decided = [False] * S
        for si in range(S):
            dec = sleeves[si]["decisions"].get(d)
            if dec is not None:
                assert has[si]
                stand[si] = list(dec)
                decided[si] = True
        if inv is not None and month_end_review(clock, k):
            cut = clock[k]
            sds = []
            for sid in sids:
                dts, rts = shadow_g[sid]
                m = bisect.bisect_right(dts, cut)
                win = rts[max(0, m - inv["lookback"]):m]
                if len(win) < inv["lookback"]:
                    sds = None
                    break
                sd = std1(win)
                if not sd > 0.0:
                    sds = None
                    break
                sds.append(sd)
            if sds is not None:
                invs = [1.0 / sd for sd in sds]
                tot = seqsum(invs)
                shares = {sid: inv["total"] * (invs[i] / tot) for i, sid in enumerate(sids)}
        due = [has[si] and (sleeves[si]["policy"] == "every_bar" or decided[si]) for si in range(S)]
        run = any(due)
        if cadence == "only_due":
            planned = list(due)
        elif cadence == "all_on_any_due":
            planned = [run and (has[si] or cfg["trade_on_carry"]) for si in range(S)]
        else:
            raise ValueError(cadence)
        cash_start = cash
        units_new = list(units)
        traded = 0.0
        tr_by_inst = [0.0] * n
        refused = False
        reb = False
        cost = 0.0
        cash_id_err = 0.0
        if any(planned):
            plan_j = sorted({idx[sym] for si in range(S) if planned[si] for sym in sleeves[si]["symbols"]})
            raw = {}
            for j in range(n):
                terms = [shares[sids[si]] * stand[si][pos[si][syms[j]]] for si in owners[j] if stand[si] is not None]
                if terms:
                    s = 0.0
                    for t in terms:
                        s += t
                    raw[j] = s
            named = sorted(raw)
            cb = E_pre if alloc is None else (alloc if alloc < E_pre else E_pre)
            notional = {j: cb * raw[j] * rs for j in named}
            if max_gross is not None:
                tg = 0.0
                for j in named:
                    tg += abs(notional[j])
                if tg > max_gross * cb:
                    refused = True
            if not refused:
                reb = True
                wanted = {}
                for j in plan_j:
                    if j not in notional:
                        continue
                    price = mark[j]
                    if flt is not None:
                        cur_not = units[j] * price
                        delta = notional[j] - cur_not
                        if delta == 0.0:
                            continue
                        ad = abs(delta)
                        if ad < min_abs:
                            continue
                        ref = abs(cur_not) if notional[j] == 0.0 else abs(notional[j])
                        if ad < min_pct * ref:
                            continue
                    wanted[j] = notional[j] / price
                if not budget:
                    for j in plan_j:
                        if j in wanted:
                            units_new[j] = wanted[j]
                            tr = abs(units_new[j] - units[j]) * mark[j]
                            tr_by_inst[j] = tr
                            traded += tr
                else:
                    cash_run = cash
                    sells, buys = [], []
                    for j in sorted(wanted):
                        assert wanted[j] >= 0.0 and units[j] >= 0.0
                        if wanted[j] < units[j]:
                            sells.append((j, units[j] - wanted[j]))
                        elif wanted[j] > units[j]:
                            buys.append((j, wanted[j] - units[j]))
                    for j, q in sells:
                        ns = q * mark[j]
                        cash_run = cash_run + ns - rate * ns
                        traded += ns
                        tr_by_inst[j] = ns
                        units_new[j] = wanted[j]
                    needed = 0.0
                    for j, q in buys:
                        nb = q * mark[j]
                        needed += nb + rate * nb
                    avail = cash_run
                    factor = 1.0 if needed <= avail else (avail / needed if avail > 0.0 else 0.0)
                    for j, q in buys:
                        qq = q * factor
                        nb = qq * mark[j]
                        cash_run = cash_run - (nb + rate * nb)
                        traded += nb
                        tr_by_inst[j] = nb
                        units_new[j] = wanted[j] if factor == 1.0 else units[j] + qq
        if reb:
            mv2 = 0.0
            for j in range(n):
                mv2 += units_new[j] * mark[j]
            cost = rate * traded
            dmv = 0.0
            for j in range(n):
                dmv += (units_new[j] - units[j]) * mark[j]
            cash = E_pre - mv2 - cost
            cash_id_err = (cash - cash_start) - (-dmv - cost)
            units = units_new
        E = E_pre - cost
        if k > 0:
            r = [mark[j] / prev_mark[j] - 1.0 for j in range(n)]
            contrib = [0.0] * S
            for j in range(n):
                contrib[owners[j][0]] += w_open[j] * r[j]
            out.append(dict(
                k=k, date=d, ret=E / E_prev - 1, equity=E, E_pre=E_pre, E_prev=E_prev, traded=traded, cost=cost, w_open=w_open,
                w_target=w_target_force, decided=decided, planned=planned, run=run, refused=refused, shares=shares_in_force,
                cash_open=cash_start / E_prev, cash_end=cash, contrib=contrib, r=r, has=has,
                tr_sleeve=[seqsum(tr_by_inst[j] for j in range(n) if owners[j][0] == si) / E_pre for si in range(S)],
                cash_id_err=cash_id_err))
        E_prev = E
    return out


BASE = dict(sleeves=["etf", "cry"], shares={"etf": 0.6, "cry": 0.4}, allocator=None, cadence="only_due", trade_filter=None,
            cash_policy="certification", risk_scale=1.0, allocated_capital=None, max_gross=None, trade_on_carry=False, policy_override={})


def cfg_of(**kw):
    c = json.loads(json.dumps(BASE))
    c.update(kw)
    return c


def book_window(sleeves):
    start = max(min(s["decisions"]) for s in sleeves)
    end = min(s["cal"][-1] for s in sleeves)
    union = sorted(set().union(*[s["cal_set"] for s in sleeves]))
    clock = [d for d in union if start <= d <= end]
    assert clock[0] == start
    return start, end, clock


SHADOW_CFG = dict(allocator=None, cadence="only_due", trade_filter=None, cash_policy="certification", risk_scale=1.0,
                  allocated_capital=None, max_gross=None, trade_on_carry=False, policy_override={})


def simulate(sleeves_all, prices, cfg):
    sleeves = []
    for sid in cfg["sleeves"]:
        s = dict(next(x for x in sleeves_all if x["id"] == sid))
        s["policy"] = cfg["policy_override"].get(sid, s["policy"])
        sleeves.append(s)
    start, end, clock = book_window(sleeves)
    shadow = {}
    for s in sleeves:
        own = [d for d in s["cal"] if start <= d <= end]
        scfg = dict(SHADOW_CFG, sleeves=[s["id"]], shares={s["id"]: 1.0})
        g = run_engine([s], prices, own, scfg, 0.0)
        nn = run_engine([s], prices, own, scfg, COST_RATE)
        shadow[s["id"]] = dict(dates=[r["date"] for r in g], g=[r["ret"] for r in g], n=[r["ret"] for r in nn])
    shadow_g = {sid: (v["dates"], v["g"]) for sid, v in shadow.items()}
    G = run_engine(sleeves, prices, clock, cfg, 0.0, shadow_g)
    N = run_engine(sleeves, prices, clock, cfg, COST_RATE, shadow_g)
    return dict(cfg=cfg, sleeves=sleeves, start=start, end=end, clock=clock, G=G, N=N, shadow=shadow)


def table(res):
    sleeves = res["sleeves"]
    sids = [s["id"] for s in sleeves]
    syms = []
    for s in sleeves:
        for sym in s["symbols"]:
            if sym not in syms:
                syms.append(sym)
    G, N = res["G"], res["N"]
    disjoint = len(syms) == sum(len(s["symbols"]) for s in sleeves)
    header = (["date", "ret_gross", "ret_net", "equity_gross", "equity_net", "cost", "turnover", "cost_prev_eq", "gross_exposure",
               "net_exposure", "cash_frac", "run", "refused"]
              + [f"decision_{i}" for i in sids] + [f"planned_{i}" for i in sids] + [f"share_{i}" for i in sids]
              + [f"traded_gross_{i}" for i in sids]
              + [f"w_target_{s}" for s in syms] + [f"w_held_{s}" for s in syms]
              + ([f"contrib_{i}" for i in sids] if disjoint else [])
              + [f"shadow_ret_gross_{i}" for i in sids] + [f"shadow_ret_net_{i}" for i in sids])
    sh_pub = {}
    for sid in sids:
        gmap = dict(zip(res["shadow"][sid]["dates"], res["shadow"][sid]["g"]))
        nmap = dict(zip(res["shadow"][sid]["dates"], res["shadow"][sid]["n"]))
        sh_pub[sid] = ([gmap.get(g["date"], 0.0) for g in G], [nmap.get(g["date"], 0.0) for g in G])
    rows = []
    for i, (g, nn) in enumerate(zip(G, N)):
        wh, wt = g["w_open"], g["w_target"]
        rows.append([g["date"], g["ret"], nn["ret"], g["equity"], nn["equity"], nn["cost"] / nn["E_pre"], nn["traded"] / nn["E_pre"],
                     nn["cost"] / nn["E_prev"], seqsum(abs(x) for x in wh), seqsum(wh), g["cash_open"], 1 if g["run"] else 0,
                     1 if g["refused"] else 0]
                    + [1 if x else 0 for x in g["decided"]] + [1 if x else 0 for x in g["planned"]] + [g["shares"][sid] for sid in sids]
                    + list(g["tr_sleeve"]) + list(wt) + list(wh)
                    + (list(g["contrib"]) if disjoint else [])
                    + [sh_pub[sid][0][i] for sid in sids] + [sh_pub[sid][1][i] for sid in sids])
    cols = {h: [r[c] for r in rows] for c, h in enumerate(header)}
    return header, rows, cols


def write_csv(path, header, rows):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(c) for c in r) + "\n")


# ------------------------------------------------------------------------------------------------ metrics and F1
def _d(s):
    return datetime.date.fromisoformat(s)


def metrics(dates, rets):
    n = len(rets)
    years = (_d(dates[-1]) - _d(dates[0])).days / 365.25
    ppy = n / years
    m = seqsum(rets) / n
    sd = std1(rets)
    cum = 1.0
    for r in rets:
        cum *= 1.0 + r
    return dict(n=n, years=years, ppy=ppy, sharpe=m / sd * math.sqrt(ppy), cagr=cum ** (1.0 / years) - 1.0, final_equity=cum)


def pearson(a, b):
    n = len(a)
    ma, mb = seqsum(a) / n, seqsum(b) / n
    saa = sbb = sab = 0.0
    for x, y in zip(a, b):
        saa += (x - ma) * (x - ma)
        sbb += (y - mb) * (y - mb)
        sab += (x - ma) * (y - mb)
    return sab / math.sqrt(saa * sbb)


def f1_stats(colsX, colsY, etf_syms):
    n = len(colsX["date"])
    l1 = [seqsum(abs(colsY[f"w_held_{s}"][k] - colsX[f"w_held_{s}"][k]) for s in etf_syms) for k in range(n)]
    dg = [colsY["ret_gross"][k] - colsX["ret_gross"][k] for k in range(n)]
    mX, mY = metrics(colsX["date"], colsX["ret_net"]), metrics(colsY["date"], colsY["ret_net"])
    tX = {k for k in range(n) if colsX["traded_gross_etf"][k] > 0.0}
    tY = {k for k in range(n) if colsY["traded_gross_etf"][k] > 0.0}
    yrs = mX["years"]
    return [
        ("bars", n),
        ("etf_planned_bars_X", sum(colsX["planned_etf"])),
        ("etf_planned_bars_Y", sum(colsY["planned_etf"])),
        ("etf_decision_bars", sum(colsX["decision_etf"])),
        ("etf_trade_bars_gross_X", len(tX)),
        ("etf_trade_bars_gross_Y", len(tY)),
        ("etf_weight_gap_L1_max", max(l1)),
        ("etf_weight_gap_L1_mean", seqsum(l1) / n),
        ("etf_weight_gap_bars_gt_1e-9", sum(1 for v in l1 if v > 1e-9)),
        ("etf_weight_gap_bars_gt_1e-2", sum(1 for v in l1 if v > 1e-2)),
        ("ret_gross_diff_max_abs", max(abs(x) for x in dg)),
        ("ret_gross_diff_mean_abs", seqsum(abs(x) for x in dg) / n),
        ("ret_gross_diff_bars_gt_1e-9", sum(1 for x in dg if abs(x) > 1e-9)),
        ("ret_gross_corr", pearson(colsX["ret_gross"], colsY["ret_gross"])),
        ("ret_net_corr", pearson(colsX["ret_net"], colsY["ret_net"])),
        ("net_d_sharpe_Y_minus_X", mY["sharpe"] - mX["sharpe"]),
        ("net_d_cagr_pp_Y_minus_X", 100.0 * (mY["cagr"] - mX["cagr"])),
        ("turnover_per_year_X", seqsum(colsX["turnover"]) / yrs),
        ("turnover_per_year_Y", seqsum(colsY["turnover"]) / yrs),
        ("cost_bps_per_year_X", 10000.0 * seqsum(colsX["cost"]) / yrs),
        ("cost_bps_per_year_Y", 10000.0 * seqsum(colsY["cost"]) / yrs),
    ]


# ------------------------------------------------------------------------------------------------ synthetic data and rules
ETFS = ["E1", "E2", "E3"]
CRYS = ["C1", "C2"]
ETF_HOLIDAYS = {"2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04"}
CRY_LAST = "2019-06-28"  # the book ends where the crypto data ends; ETF data runs on for two more weeks (truncation matters)
ETF_LAST = "2019-07-12"
FIRST = "2019-01-02"
SEED = 4  # LCG seed of the synthetic prices (chosen so every code path is exercised, see the asserts in write_fixtures)


def days(a, b):
    d = _d(a)
    while d <= _d(b):
        yield d
        d += datetime.timedelta(days=1)


def synthetic_prices():
    state = SEED
    prices = {}
    cals = {}
    etf_cal = [d.isoformat() for d in days(FIRST, ETF_LAST) if d.weekday() < 5 and d.isoformat() not in ETF_HOLIDAYS]
    cry_cal = [d.isoformat() for d in days(FIRST, CRY_LAST)]
    cals["etf"], cals["cry"] = etf_cal, cry_cal
    for kind, syms, cal, vol, per in (("etf", ETFS, etf_cal, 0.012, 23), ("cry", CRYS, cry_cal, 0.045, 11)):
        for i, sym in enumerate(syms):
            p = 100.0 + 25.0 * i
            series = {}
            for t, d in enumerate(cal):
                state = (1103515245 * state + 12345) % 2147483648
                u = state / 2147483648.0
                regime = 0.002 if (t // (per + 7 * i)) % 2 == 0 else -0.002
                p = p * (1.0 + regime + (u - 0.5) * vol)
                series[d] = round(p, 4)
            prices[sym] = series
    return prices, cals


def etf_decisions(cal, prices):
    me = [i for i in range(len(cal)) if i + 1 == len(cal) or cal[i][:7] != cal[i + 1][:7]]
    out = {}
    for p, i in enumerate(me):
        if p + 1 < 3:
            continue
        last3 = me[p - 2:p + 1]
        w = []
        for sym in ETFS:
            c = prices[sym]
            s = 0.0
            for j in last3:
                s += c[cal[j]]
            w.append(0.3 if c[cal[i]] > s / 3.0 else 0.0)
        out[cal[i]] = w
    return out


def cry_decisions(cal, prices):
    out = {}
    for i in range(9, len(cal)):
        w = []
        for sym in CRYS:
            c = prices[sym]
            s = 0.0
            for j in range(i - 9, i + 1):
                s += c[cal[j]]
            w.append(0.5 if c[cal[i]] > s / 10.0 else 0.0)
        out[cal[i]] = w
    return out


def synthetic_sleeves():
    prices, cals = synthetic_prices()
    etf = dict(id="etf", symbols=ETFS, cal=cals["etf"], cal_set=set(cals["etf"]), decisions=etf_decisions(cals["etf"], prices), policy="on_decision")
    cry = dict(id="cry", symbols=CRYS, cal=cals["cry"], cal_set=set(cals["cry"]), decisions=cry_decisions(cals["cry"], prices), policy="every_bar")
    return prices, cals, [etf, cry]


NET_DAYS = 90


def netting_fixture():
    """Two sleeves sharing X with opposite signs (signed sum must NET). The LCG and the decision formulas are those of the PF0 key's
    `synthetic_fixture` (integer LCG, no libm), so this file is byte-comparable with the key's `synthetic_netting_perbar.csv`."""
    ds = [(datetime.date(2020, 1, 1) + datetime.timedelta(days=i)).isoformat() for i in range(NET_DAYS)]
    state = 12345
    prices = {}
    for sym, p0 in (("X", 100.0), ("Y", 50.0), ("Z", 20.0)):
        p = p0
        series = {}
        for d in ds:
            state = (1103515245 * state + 12345) % 2147483648
            r = (state / 2147483648.0 - 0.5) * 0.06
            p = p * (1.0 + r)
            series[d] = p
        prices[sym] = series
    dec_a, dec_b = {}, {}
    for i, d in enumerate(ds):
        dec_a[d] = [0.6, 0.4] if (i // 7) % 2 == 0 else [0.3, 0.7]
        dec_b[d] = [-0.8, 0.2] if (i // 5) % 3 != 1 else [0.5, -0.3]
    sl = [dict(id="a", symbols=["X", "Y"], cal=ds, cal_set=set(ds), decisions=dec_a, policy="every_bar"),
          dict(id="b", symbols=["X", "Z"], cal=ds, cal_set=set(ds), decisions=dec_b, policy="every_bar")]
    cfg = cfg_of(sleeves=["a", "b"], shares={"a": 0.5, "b": 0.5})
    return prices, ds, sl, cfg


# The synthetic configurations (same names and roles as the key's; parameters chosen for the 3-month synthetic window).
ALLOC_CURRENCY = 98500.0  # binds while equity > 0.985 (normalised)
INVVOL = {"kind": "inverse_vol", "lookback": 15, "review": "calendar_month_end", "total": 1.0}
CONFIGS = {
    "book_cert_60_40": cfg_of(),
    "book_live_60_40": cfg_of(cadence="all_on_any_due", trade_filter=FILTER, cash_policy="budget"),
    "book_due_filter_60_40": cfg_of(cadence="only_due", trade_filter=FILTER, cash_policy="budget"),
    "book_scaled_50_30": cfg_of(shares={"etf": 0.5, "cry": 0.3}, cadence="all_on_any_due", trade_filter=FILTER, cash_policy="budget",
                                risk_scale=0.8, allocated_capital=ALLOC_CURRENCY),
    "book_grosscap_60_40": cfg_of(max_gross=0.7),
    "book_invvol": cfg_of(shares={"etf": 0.5, "cry": 0.5}, allocator=INVVOL),
    "single_etf_everybar_filter": cfg_of(sleeves=["etf"], shares={"etf": 1.0}, trade_filter=FILTER, policy_override={"etf": "every_bar"}),
}
PERBAR = list(CONFIGS)


def write_fixtures(outdir):
    prices, cals, sleeves = synthetic_sleeves()
    # synthetic candles in the long format of the real fixture
    with open(os.path.join(outdir, "synthetic_book_candles.csv"), "w", newline="\n") as f:
        f.write("symbol,date_utc,close\n")
        for sym in ETFS + CRYS:
            for d in sorted(prices[sym]):
                f.write(f"{sym},{d},{prices[sym][d]!r}\n")
    with open(os.path.join(outdir, "synthetic_book_decisions.csv"), "w", newline="\n") as f:
        f.write("sleeve,date," + "w0,w1,w2\n")
        for s in sleeves:
            for d in sorted(s["decisions"]):
                f.write(f"{s['id']},{d}," + ",".join(repr(x) for x in s["decisions"][d]) + "\n")
    results, tables = {}, {}
    for name, cfg in CONFIGS.items():
        res = simulate(sleeves, prices, cfg)
        results[name] = res
        tables[name] = table(res)
        write_csv(os.path.join(outdir, f"{name}_perbar.csv"), tables[name][0], tables[name][1])
    # ---- self-checks so that the committed fixtures exercise what the tests claim they exercise
    cert = tables["book_cert_60_40"][2]
    live = tables["book_live_60_40"][2]
    assert 3 <= sum(cert["decision_etf"]) <= 6, sum(cert["decision_etf"])
    assert sum(cert["planned_etf"]) == sum(cert["decision_etf"])          # only_due: the ETF is planned on its decision bars only
    assert sum(live["planned_etf"]) > 4 * sum(cert["planned_etf"])       # all_on_any_due: planned whenever the ETF is open
    carry = sum(1 for g in results["book_cert_60_40"]["G"] if not g["has"][0])
    assert carry > 20, carry                                             # weekends and holidays are carry rows for the ETF sleeve
    gc = tables["book_grosscap_60_40"][2]
    assert 5 < sum(gc["refused"]) < len(gc["date"]) - 5, sum(gc["refused"])
    sc = results["book_scaled_50_30"]
    binding = sum(1 for g in sc["G"] if g["E_pre"] > ALLOC_CURRENCY / CAPITAL0)
    assert 5 < binding < len(sc["G"]) - 5, binding                       # the allocated-capital cap binds on some bars and not on others
    iv = tables["book_invvol"][2]
    assert len({round(x, 9) for x in iv["share_etf"]}) >= 3, "the inverse-vol allocator must change the shares"
    assert abs(iv["share_etf"][0] - 0.5) < 1e-15, "shares start at the initial 0.5 until a review has enough data"
    # ---- netting fixture
    nprices, nds, nsl, ncfg = netting_fixture()
    with open(os.path.join(outdir, "synthetic_netting_candles.csv"), "w", newline="\n") as f:
        f.write("symbol,date_utc,close\n")
        for sym in ("X", "Y", "Z"):
            for d in nds:
                f.write(f"{sym},{d},{nprices[sym][d]!r}\n")
    nres = simulate(nsl, nprices, ncfg)
    ntab = table(nres)
    write_csv(os.path.join(outdir, "synthetic_netting_perbar.csv"), ntab[0], ntab[1])
    # ---- F1 numbers on the synthetic fixture
    f1 = f1_stats(tables["book_due_filter_60_40"][2], tables["book_live_60_40"][2], ETFS)
    with open(os.path.join(outdir, "synthetic_book_f1.csv"), "w", newline="\n") as f:
        f.write("name,value\n")
        for k, v in f1:
            f.write(f"{k},{fmt(v)}\n")
    meta = [("start_bar", results["book_cert_60_40"]["start"]), ("end", results["book_cert_60_40"]["end"]), ("capital0", CAPITAL0),
            ("min_abs", FILTER["min_abs"]), ("min_pct", FILTER["min_pct"]), ("allocated_capital_currency", ALLOC_CURRENCY),
            ("risk_scale_scaled", 0.8), ("max_gross", 0.7), ("invvol_lookback", INVVOL["lookback"]),
            ("etf_first_decision", min(sleeves[0]["decisions"])), ("cry_first_decision", min(sleeves[1]["decisions"]))]
    with open(os.path.join(outdir, "synthetic_book_meta.csv"), "w", newline="\n") as f:
        f.write("name,value\n")
        for k, v in meta:
            f.write(f"{k},{fmt(v) if not isinstance(v, str) else v}\n")
    # ---- manifest
    names = sorted(n for n in os.listdir(outdir) if n.endswith(".csv") or n == "gen_book_key.py")
    with open(os.path.join(outdir, "MANIFEST.sha256"), "w", newline="\n") as f:
        for n in names:
            f.write(f"{hashlib.sha256(open(os.path.join(outdir, n), 'rb').read()).hexdigest()}  {n}\n")
    print("wrote", len(names), "files; ETF decision bars in window:", sum(cert["decision_etf"]), "; grosscap refusals:", sum(gc["refused"]),
          "; allocated cap binding bars:", binding, "; window", results["book_cert_60_40"]["start"], "..", results["book_cert_60_40"]["end"])


# ------------------------------------------------------------------------------------------------ verify against the REAL key
def verify_real(keydir, ladder):
    """Run this engine on the real pinned candles + real T0 decisions and compare every cell of every real per-bar key file."""
    prices = {}
    with open(os.path.join(ladder, "ladder_candles.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            prices.setdefault(r["symbol"], {})[r["date_utc"]] = float(r["close"])
    ETF = ["SPY", "EFA", "IEF", "DBC", "VNQ"]
    CRY = ["BTC", "ETH"]

    def joint(symbols):
        common = set(prices[symbols[0]])
        for s in symbols[1:]:
            common &= set(prices[s])
        return sorted(common)

    def read_decisions(name, symbols):
        dec = {}
        with open(os.path.join(ladder, "key", name), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            head = next(rd)
            assert head[3:] == [f"w_{s}" for s in symbols], head
            for row in rd:
                assert row[1] == "ok"
                dec[row[0]] = [float(x) for x in row[3:]]
        return dec

    cal1, cal3 = joint(ETF), joint(CRY)
    sleeves = [dict(id="etf", symbols=ETF, cal=cal1, cal_set=set(cal1), decisions=read_decisions("S1_etf_trend_faber_decisions.csv", ETF), policy="on_decision"),
               dict(id="crypto", symbols=CRY, cal=cal3, cal_set=set(cal3), decisions=read_decisions("S3_crypto_trend_100d_decisions.csv", CRY), policy="every_bar")]
    base = cfg_of(sleeves=["etf", "crypto"], shares={"etf": 0.6, "crypto": 0.4})
    real_cfgs = {
        "book_cert_60_40": base,
        "book_live_60_40": dict(base, cadence="all_on_any_due", trade_filter=FILTER, cash_policy="budget"),
        "book_scaled_50_30": dict(base, shares={"etf": 0.5, "crypto": 0.3}, cadence="all_on_any_due", trade_filter=FILTER, cash_policy="budget",
                                  risk_scale=0.8, allocated_capital=1.2 * CAPITAL0),
        "book_grosscap_60_40": dict(base, max_gross=0.85),
        "book_invvol": dict(base, shares={"etf": 0.5, "crypto": 0.5}, allocator={"kind": "inverse_vol", "lookback": 60, "review": "calendar_month_end", "total": 1.0}),
    }
    worst_all = 0.0
    for name, cfg in real_cfgs.items():
        res = simulate(sleeves, prices, cfg)
        header, rows, cols = table(res)
        with open(os.path.join(keydir, "key", f"{name}_perbar.csv"), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            head = next(rd)
            ref = list(rd)
        assert head == header, (name, head, header)
        assert len(ref) == len(rows), (name, len(ref), len(rows))
        worst = 0.0
        nne = 0
        for r_ref, r_mine in zip(ref, rows):
            assert r_ref[0] == r_mine[0]
            for a, b in zip(r_ref[1:], r_mine[1:]):
                d = abs(float(a) - float(b))
                worst = max(worst, d)
                if float(a) != float(b):
                    nne += 1
        print(f"verify-real {name:22s} rows {len(rows):5d} cells {len(rows) * (len(header) - 1):7d} not-bit-identical {nne:5d} max|diff| {worst:.3e}")
        worst_all = max(worst_all, worst)
    nprices, nds, nsl, ncfg = netting_fixture()
    nres = simulate(nsl, nprices, ncfg)
    nh, nrows, _ = table(nres)
    with open(os.path.join(keydir, "key", "synthetic_netting_perbar.csv"), newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        ref = list(rd)
    assert head == nh and len(ref) == len(nrows)
    worst = 0.0
    nne = 0
    for r_ref, r_mine in zip(ref, nrows):
        for a, b in zip(r_ref[1:], r_mine[1:]):
            worst = max(worst, abs(float(a) - float(b)))
            if float(a) != float(b):
                nne += 1
    print(f"verify-real synthetic_netting       rows {len(nrows):5d} not-bit-identical {nne:5d} max|diff| {worst:.3e}")
    worst_all = max(worst_all, worst)
    print("VERIFY-REAL", "PASS" if worst_all <= 1e-12 else "FAIL", f"(worst max|diff| {worst_all:.3e})")
    return 0 if worst_all <= 1e-12 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-real", nargs=2, metavar=("KEYDIR", "LADDERDIR"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    if a.verify_real:
        sys.exit(verify_real(*a.verify_real))
    write_fixtures(a.out)


if __name__ == "__main__":
    main()
