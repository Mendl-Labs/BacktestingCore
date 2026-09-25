#!/usr/bin/env python3
"""Generates `invvol_synthetic.txt`: SYNTHETIC sleeve return series (integer-LCG, no vendor data) and the InverseVol shares
the PF0 book key's definition gives for a list of reviews.

The share definition is the one pre-registered in Amendment 12 section 4 and implemented in `book_key.py`
(`run_engine`, the `inv` branch): for each sleeve the last `lookback` returns of its OWN calendar visible at the review,
`std1` = sample standard deviation with ddof 1 from SEQUENTIAL sums, `inv_i = 1 / sd_i`, `share_i = total * (inv_i / tot)`
with `tot` the sequential sum of the `inv`; a review with fewer than `lookback` visible returns for any sleeve, or a
non-positive deviation, leaves the shares unchanged. This script re-states that arithmetic in plain Python floats (+ - * /
and sqrt, sequential sums) so that the Rust implementation can be compared to it to 1e-12. It reads nothing and calls
nothing; running it twice gives byte-identical output.

File format (LF, UTF-8):
  SLEEVE <i> <n> <comma separated returns (repr)>
  REVIEW <visible_0>,<visible_1>,... UPDATED <share_0>,<share_1>,...
  REVIEW <visible_0>,<visible_1>,... HELD
"""
import math
import sys
from pathlib import Path

LOOKBACK = 60
TOTAL = 1.0
SEED = 20260924


def lcg_stream(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 11) / float(1 << 53)


def series(seed, n, scale):
    u = lcg_stream(seed)
    out = []
    for _ in range(n):
        # sum of four uniforms, centred: a bell-ish shape, deterministic
        s = next(u) + next(u) + next(u) + next(u) - 2.0
        out.append(s * scale)
    return out


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


def review(rets, visible, shares):
    sds = []
    for r, vis in zip(rets, visible):
        if vis < LOOKBACK:
            return None
        sd = std1(r[vis - LOOKBACK:vis])
        if not sd > 0.0:
            return None
        sds.append(sd)
    invs = [1.0 / sd for sd in sds]
    tot = seqsum(invs)
    return [TOTAL * (v / tot) for v in invs]


def main():
    rets = [series(SEED + 1, 300, 0.005), series(SEED + 2, 300, 0.012), series(SEED + 3, 250, 0.03)]
    reviews = [(59, 60, 60), (60, 60, 60), (100, 100, 100), (140, 140, 130), (200, 200, 180), (299, 299, 249), (300, 300, 250)]
    lines = []
    for i, r in enumerate(rets):
        lines.append("SLEEVE %d %d %s" % (i, len(r), ",".join(repr(x) for x in r)))
    shares = [TOTAL / len(rets)] * len(rets)
    for vis in reviews:
        new = review(rets, vis, shares)
        vs = ",".join(str(v) for v in vis)
        if new is None:
            lines.append("REVIEW %s HELD" % vs)
        else:
            shares = new
            lines.append("REVIEW %s UPDATED %s" % (vs, ",".join(repr(x) for x in new)))
    out = Path(__file__).resolve().parent / "invvol_synthetic.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out, "lookback", LOOKBACK)


if __name__ == "__main__":
    sys.exit(main())
