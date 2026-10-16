#!/usr/bin/env python3
"""Generate the weightsim ANSWER-KEY fixtures by running the pinned `shadow.py` logic (stage1-record replication ladder).

The key is Python only. This script does NOT re-implement S1/S3: it loads the pinned `shadow.py` (sha256 checked),
patches ONLY its data-directory constant `D` (a path on the original author's machine), and calls its own
`sleeve1()` / `sleeve3()` on a small deterministic SYNTHETIC price panel. Nothing here is vendor data.

Modes
  python gen_answer_key.py --shadow <path/to/shadow.py>
      Write the synthetic panel and the key files next to this script (run ONCE; the outputs are committed and pinned
      in MANIFEST.sha256; the Rust tests never regenerate them).
  python gen_answer_key.py --shadow <shadow.py> --verify-real <replication_ladder_dir>
      Proof that this harness runs the same logic as the recorded key: run `sleeve1()`/`sleeve3()` on the REAL pinned
      `ladder_candles.csv` in that directory (read-only, nothing is written) and check that they reproduce the recorded
      `shadow_S1_daily_returns.csv` / `shadow_S3_daily_returns.csv` and `shadow_summary.json`.

Files written (all small, all derived from synthetic prices):
  synthetic_ladder_candles.csv   symbol,date_utc,close   (same long format as the real fixture; SPY EFA IEF DBC VNQ BTC ETH)
  key_S1_returns.csv             date,ret,equity   (shadow sleeve1 returns; equity = cumprod(1+ret))
  key_S1_signals.csv             date,SPY,EFA,IEF,DBC,VNQ   (month-end signals, 0/1)
  key_S3_returns.csv / key_S3_signals.csv   same for sleeve3 (window 2016-01-01..2020-12-31 as in shadow.py)
  key_metrics.csv                sleeve,metric,value   (unrounded metrics recomputed with shadow's formulas, plus
                                 shadow's own rounded `metrics()` output, plus its `asset_trades` flip counter)
  MANIFEST.sha256                sha256 of every fixture file (and of this script), `sha256sum` format
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SHADOW_SHA256 = "d7014041512137950f404d14a5124bd5a9911a52c7b79d2c01d1b41e5e275a6e"
HERE = os.path.dirname(os.path.abspath(__file__))
ETF = ["SPY", "EFA", "IEF", "DBC", "VNQ"]
CRY = ["BTC", "ETH"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_shadow(shadow_path, data_dir):
    """Exec the pinned shadow.py with only its `D` constant redirected to `data_dir`."""
    got = sha256_file(shadow_path)
    if got != SHADOW_SHA256:
        sys.exit(f"shadow.py sha256 {got} != pinned {SHADOW_SHA256}; refusing to use a different key")
    src = open(shadow_path, "r", encoding="utf-8").read()
    src2, n = re.subn(r'(?m)^D = r".*"\s*$', 'D = r"' + data_dir.replace("\\", "/").rstrip("/") + '/"', src)
    if n != 1:
        sys.exit("could not patch the D constant of shadow.py (expected exactly one line)")
    ns = {"__name__": "shadow_key"}
    exec(compile(src2, shadow_path, "exec"), ns)
    return ns


def synthetic_long_csv(path):
    """Deterministic synthetic prices in the real long format. Regime-switching drift so SMA crossovers (flips) happen."""
    rng = np.random.default_rng(20260924)
    rows = []

    # ETFs: weekdays 2017-01-02 .. 2019-06-28; SPY is missing 2018-07-04 (joint-calendar test: shadow uses dropna()).
    days = pd.bdate_range("2017-01-02", "2019-06-28")
    t = np.arange(len(days))
    for j, sym in enumerate(ETF):
        drift = 0.0012 * np.sin(2 * np.pi * t / (170 + 23 * j) + j)
        r = drift + rng.normal(0.0, 0.009, len(days))
        px = np.round(100.0 * (1 + j * 0.3) * np.exp(np.cumsum(r)), 2)
        for d, p in zip(days, px):
            if sym == "SPY" and d == pd.Timestamp("2018-07-04"):
                continue
            rows.append((sym, d.strftime("%Y-%m-%d"), p))

    # Crypto: calendar days. BTC starts 2015-07-25, ETH 2015-08-01 and misses 2015-09-04 and 2016-03-15 (joint calendar).
    days = pd.date_range("2015-07-25", "2016-09-30", freq="D")
    t = np.arange(len(days))
    common = 0.006 * np.sin(2 * np.pi * t / 190 + 0.5)  # shared market factor: both coins sometimes below SMA100 (flat days)
    for j, sym in enumerate(CRY):
        drift = common + 0.002 * np.sin(2 * np.pi * t / (120 + 40 * j) + 1.7 * j)
        r = drift + rng.normal(0.0, 0.035, len(days))
        px = np.round(300.0 * (1 + 0.1 * j) * np.exp(np.cumsum(r)), 2)
        for d, p in zip(days, px):
            ds = d.strftime("%Y-%m-%d")
            if sym == "ETH" and (d < pd.Timestamp("2015-08-01") or ds in ("2015-09-04", "2016-03-15")):
                continue
            rows.append((sym, ds, p))

    with open(path, "w", newline="\n") as f:
        f.write("symbol,date_utc,close\n")
        for s, d, p in rows:
            f.write(f"{s},{d},{float(p)!r}\n")  # float(): numpy>=2 reprs as np.float64(...)


def unrounded_metrics(r):
    """shadow.metrics() formulas, without the rounding (so Rust can be compared at full precision)."""
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25
    ppy = len(r) / years
    cum = (1 + r).cumprod()
    return dict(
        obs=len(r),
        years=years,
        ppy=ppy,
        cagr=cum.iloc[-1] ** (1 / years) - 1,
        vol=r.std() * np.sqrt(ppy),
        sharpe=r.mean() / r.std() * np.sqrt(ppy),
        max_drawdown=(cum / cum.cummax() - 1).min(),
        final_equity=cum.iloc[-1],
    )


def run_key(ns):
    r1, sig1, m1 = ns["sleeve1"]()
    r3, sig3, m3 = ns["sleeve3"]()
    return (r1, sig1, m1), (r3, sig3, m3)


def write_key(ns, outdir):
    (r1, sig1, m1), (r3, sig3, m3) = run_key(ns)
    metrics = ns["metrics"]
    rows = []
    for name, r, m in (("S1", r1, m1), ("S3", r3, m3)):
        um = unrounded_metrics(r)
        rm = metrics(r, name)
        for k, v in um.items():
            rows.append((name, k, repr(float(v))))
        for k in ("cagr", "vol", "sharpe", "max_drawdown", "ppy"):
            rows.append((name, k + "_shadow_rounded", repr(float(rm[k]))))
        rows.append((name, "flips", str(int(m["asset_trades"]))))
        cum = (1 + r).cumprod()
        with open(os.path.join(outdir, f"key_{name}_returns.csv"), "w", newline="\n") as f:
            f.write("date,ret,equity\n")
            for d, x, e in zip(r.index, r.values, cum.values):
                f.write(f"{d.strftime('%Y-%m-%d')},{float(x)!r},{float(e)!r}\n")
    for name, sig in (("S1", sig1), ("S3", sig3)):
        cols = ETF if name == "S1" else CRY
        with open(os.path.join(outdir, f"key_{name}_signals.csv"), "w", newline="\n") as f:
            f.write("date," + ",".join(cols) + "\n")
            for d, row in sig.iterrows():
                f.write(d.strftime("%Y-%m-%d") + "," + ",".join(str(int(row[c])) for c in cols) + "\n")
    with open(os.path.join(outdir, "key_metrics.csv"), "w", newline="\n") as f:
        f.write("sleeve,metric,value\n")
        for s, k, v in rows:
            f.write(f"{s},{k},{v}\n")
    return m1["asset_trades"], m3["asset_trades"], len(r1), len(r3)


def verify_real(shadow_path, real_dir):
    ns = load_shadow(shadow_path, real_dir)
    (r1, sig1, m1), (r3, sig3, m3) = run_key(ns)
    ok = True
    for name, r in (("S1", r1), ("S3", r3)):
        rec = pd.read_csv(os.path.join(real_dir, f"shadow_{name}_daily_returns.csv"), index_col=0)
        rec.index = pd.to_datetime(rec.index)
        rec = rec["ret"]
        same_idx = list(rec.index) == list(r.index)
        diff = float(np.max(np.abs(rec.values - r.values))) if same_idx else float("nan")
        print(f"{name}: recorded rows {len(rec)}, regenerated rows {len(r)}, same index {same_idx}, max|diff| {diff:.3e}")
        ok &= same_idx and diff <= 1e-12
    summ = json.load(open(os.path.join(real_dir, "shadow_summary.json")))
    for name, m in (("S1", m1), ("S3", m3)):
        ok &= int(summ[name]["asset_trades"]) == int(m["asset_trades"])
        print(f"{name}: flips regenerated {m['asset_trades']} vs recorded {summ[name]['asset_trades']}")
    for name, r in (("S1", r1), ("S3", r3)):
        rm = ns["metrics"](r, name)
        for k in ("obs", "ppy", "cagr", "vol", "sharpe", "max_drawdown"):
            if rm[k] != summ[name][k]:
                ok = False
                print(f"  MISMATCH {name}.{k}: {rm[k]} vs recorded {summ[name][k]}")
    print("VERIFY-REAL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", required=True)
    ap.add_argument("--verify-real", metavar="DIR")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    if a.verify_real:
        verify_real(a.shadow, a.verify_real)
    synth = os.path.join(a.out, "synthetic_ladder_candles.csv")
    synthetic_long_csv(synth)
    with tempfile.TemporaryDirectory() as tmp:
        # shadow.py reads `<D>/ladder_candles.csv`; give it the synthetic panel under that name.
        with open(synth, "rb") as src, open(os.path.join(tmp, "ladder_candles.csv"), "wb") as dst:
            dst.write(src.read())
        ns = load_shadow(a.shadow, tmp)
        f1, f3, n1, n3 = write_key(ns, a.out)
    print(f"synthetic key written: S1 {n1} returns / {f1} flips, S3 {n3} returns / {f3} flips")
    names = sorted(
        n for n in os.listdir(a.out)
        if n.endswith(".csv") or n == "gen_answer_key.py"
    )
    with open(os.path.join(a.out, "MANIFEST.sha256"), "w", newline="\n") as f:
        for n in names:
            f.write(f"{sha256_file(os.path.join(a.out, n))}  {n}\n")
    print("MANIFEST.sha256 written for:", ", ".join(names))


if __name__ == "__main__":
    main()
