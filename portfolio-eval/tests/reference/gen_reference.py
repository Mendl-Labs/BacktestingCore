#!/usr/bin/env python3
"""Independent reference values for portfolio-eval's hand-computed tests.

Nothing here imports the Rust crate. Everything is recomputed from the definitions with `fractions.Fraction` (exact
rational arithmetic) wherever the quantity is rational, with the standard library's `statistics.NormalDist` for
normal quantiles/cdf, and with integer arithmetic for the random number generators and the stationary bootstrap. The
Rust tests embed the numbers printed by this script; rerun it to audit them:

    python3 tests/reference/gen_reference.py

Output is `name = value` lines (floats printed with repr, i.e. round-trip exact).
"""
from fractions import Fraction as F
from statistics import NormalDist
import hashlib
import itertools
import math

M64 = (1 << 64) - 1
ND = NormalDist()


def show(name, v):
    print(f"{name} = {v!r}")


# ----------------------------------------------------------------------------------------------- RNG
def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


class Xoshiro:
    def __init__(self, s):
        self.s = list(s)

    @staticmethod
    def seeded(seed):
        sm = seed & M64
        s = []
        for _ in range(4):
            sm, v = splitmix64(sm)
            s.append(v)
        return Xoshiro(s)

    def next_u64(self):
        s = self.s
        result = (rotl((s[1] * 5) & M64, 7) * 9) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return result

    def below(self, n):
        if n <= 1:
            return 0
        m = self.next_u64() * n
        lo = m & M64
        if lo < n:
            threshold = ((-n) & M64) % n
            while lo < threshold:
                m = self.next_u64() * n
                lo = m & M64
        return m >> 64

    def unit(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal_pair(self):
        u1 = ((self.next_u64() >> 11) + 1) * (1.0 / 9007199254740992.0)
        u2 = self.unit()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)


def stationary_indices(n, mean_block, rng):
    thr = None if mean_block <= 1.0 else int((1.0 / mean_block) * 18446744073709551616.0)
    out = []
    prev = 0
    for t in range(n):
        if t == 0:
            idx = rng.below(n)
        elif thr is None:
            idx = rng.below(n)
        elif rng.next_u64() < thr:
            idx = rng.below(n)
        else:
            idx = 0 if prev + 1 == n else prev + 1
        out.append(idx)
        prev = idx
    return out


def digest(idx):
    return hashlib.sha256(b"".join(i.to_bytes(8, "little") for i in idx)).hexdigest()


print("# --- rng")
sm = 0
outs = []
for _ in range(3):
    sm, v = splitmix64(sm)
    outs.append(v)
show("splitmix64(seed 0) x3", [hex(v) for v in outs])
x = Xoshiro([1, 2, 3, 4])
show("xoshiro256** state [1,2,3,4] x4", [x.next_u64() for _ in range(4)])
x = Xoshiro.seeded(42)
show("xoshiro256** seed 42 x3", [x.next_u64() for _ in range(3)])
x = Xoshiro.seeded(42)
show("below(10) x8 seed 42", [x.below(10) for _ in range(8)])
x = Xoshiro.seeded(9)
show("unit() x3 seed 9", [x.unit() for _ in range(3)])
x = Xoshiro.seeded(5)
z = []
for _ in range(2):
    a, b = x.normal_pair()
    z += [a, b]
show("normal() x4 seed 5 (libm; compare to 1e-12)", z)

print("# --- stationary bootstrap")
idx = stationary_indices(20, 4.0, Xoshiro.seeded(7))
show("indices n=20 b=4 seed 7", idx)
show("digest n=20 b=4 seed 7", digest(idx))
idx = stationary_indices(10, 1.0, Xoshiro.seeded(1))
show("indices n=10 b=1 seed 1", idx)
idx = stationary_indices(1000, 10.0, Xoshiro.seeded(123))
show("digest n=1000 b=10 seed 123", digest(idx))
idx = stationary_indices(500, 3.0, Xoshiro.seeded(2026))
show("digest n=500 b=3 seed 2026", digest(idx))

# ----------------------------------------------------------------------------------------------- basic stats
X = [F(1, 100), F(-2, 100), F(3, 100), F(0), F(15, 1000), F(-5, 1000), F(2, 100), F(-1, 100)]
n = len(X)
mean = sum(X) / n
var = sum((v - mean) ** 2 for v in X) / (n - 1)
m2 = sum((v - mean) ** 2 for v in X) / n
m3 = sum((v - mean) ** 3 for v in X) / n
m4 = sum((v - mean) ** 4 for v in X) / n
print("# --- stats on [0.01,-0.02,0.03,0,0.015,-0.005,0.02,-0.01]")
show("mean", float(mean))
show("variance ddof1", float(var))
show("sharpe per period", float(mean) / math.sqrt(float(var)))
show("skew", float(m3) / math.sqrt(float(m2)) ** 3)
show("kurt", float(m4 / (m2 * m2)))
show("autocov lag1 (1/n)", float(sum((X[t] - mean) * (X[t - 1] - mean) for t in range(1, n)) / n))
Y = [F(v, 1000) for v in (5, -3, 8, 1, -2, 6, 0, 4)]
my = sum(Y) / n
cov = sum((a - mean) * (b - my) for a, b in zip(X, Y))
show("corr(X,Y)", float(cov) / math.sqrt(float(sum((a - mean) ** 2 for a in X)) * float(sum((b - my) ** 2 for b in Y))))

# ----------------------------------------------------------------------------------------------- HAC
def solve_inverse(A):
    n = len(A)
    M = [[F(v) for v in row] + [F(1 if i == j else 0) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def spanning(y, bench, L):
    n = len(y)
    p = len(bench) + 1
    Z = [[F(1)] + [b[t] for b in bench] for t in range(n)]
    A = [[sum(Z[t][i] * Z[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    c = [sum(Z[t][i] * y[t] for t in range(n)) for i in range(p)]
    inv = solve_inverse(A)
    theta = [sum(inv[i][j] * c[j] for j in range(p)) for i in range(p)]
    e = [y[t] - sum(theta[j] * Z[t][j] for j in range(p)) for t in range(n)]
    s = [[Z[t][j] * e[t] for j in range(p)] for t in range(n)]
    om = [[sum(s[t][i] * s[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    for k in range(1, L + 1):
        w = 1 - F(k, L + 1)
        for t in range(k, n):
            for i in range(p):
                for j in range(p):
                    om[i][j] += w * (s[t][i] * s[t - k][j] + s[t - k][i] * s[t][j])
    tmp = [[sum(inv[i][k] * om[k][j] for k in range(p)) for j in range(p)] for i in range(p)]
    V = [[sum(tmp[i][k] * inv[k][j] for k in range(p)) for j in range(p)] for i in range(p)]
    sse = sum(v * v for v in e)
    return theta, V, sse, n - p


B1 = [F(v, 1000) for v in (10, -4, 7, 2, -9, 5, 1, -3, 8, -6)]
Y1 = [F(v, 1000) for v in (13, -5, 11, 4, -8, 9, 0, -2, 12, -9)]
print("# --- spanning regression, n=10, y on [1, b], b=[10,-4,7,2,-9,5,1,-3,8,-6]e-3, y=[13,-5,11,4,-8,9,0,-2,12,-9]e-3")
for L in (0, 2):
    theta, V, sse, dof = spanning(Y1, [B1], L)
    se_a = math.sqrt(float(V[0][0]))
    show(f"L={L} alpha", float(theta[0]))
    show(f"L={L} beta", float(theta[1]))
    show(f"L={L} se_alpha", se_a)
    show(f"L={L} se_beta", math.sqrt(float(V[1][1])))
    show(f"L={L} t_alpha", float(theta[0]) / se_a)
    show(f"L={L} p_two_sided", 2 * (1 - ND.cdf(abs(float(theta[0]) / se_a))))
    show(f"L={L} resid_var", float(sse / dof))
Z1 = [F(v, 1000) for v in (4, 9, -7, 3, 0, -5, 6, 2, -1, 8)]
theta, V, sse, dof = spanning(Y1, [B1, Z1], 1)
print("# two benchmarks, L=1")
show("2b alpha", float(theta[0]))
show("2b betas", [float(theta[1]), float(theta[2])])
show("2b se_alpha", math.sqrt(float(V[0][0])))
show("2b se_betas", [math.sqrt(float(V[1][1])), math.sqrt(float(V[2][2]))])


def lrv_mean(x, L):
    n = len(x)
    mu = sum(x) / n
    g = lambda l: sum((x[t] - mu) * (x[t - l] - mu) for t in range(l, n)) / n
    return g(0) + 2 * sum((1 - F(l, L + 1)) * g(l) for l in range(1, L + 1))


print("# --- long-run variance of the mean of X (Bartlett)")
for L in (0, 1, 3):
    show(f"lrv L={L}", float(lrv_mean(X, L)))
show("NW lag n=100", math.floor(4 * (100 / 100) ** (2 / 9)))
show("NW lag n=250", math.floor(4 * (250 / 100) ** (2 / 9)))
show("NW lag n=1260", math.floor(4 * (1260 / 100) ** (2 / 9)))
show("NW lag n=2520", math.floor(4 * (2520 / 100) ** (2 / 9)))

# ----------------------------------------------------------------------------------------------- DSR & co
print("# --- Deflated Sharpe (Bailey & Lopez de Prado 2014), NormalDist quantiles")
g = 0.5772156649015329


def emax(N):
    if N <= 1:
        return 0.0
    return (1 - g) * ND.inv_cdf(1 - 1 / N) + g * ND.inv_cdf(1 - 1 / (N * math.e))


for N in (1, 2, 10, 50, 100, 1000):
    show(f"expected_max_normal({N})", emax(N))


def dsr(sr, T, skew, kurt, N, sd, floor=True):
    se2 = (1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (T - 1)
    if floor:
        se2 = max(se2, 1 / (T - 1))
    sr0 = sd * emax(N)
    return ND.cdf((sr - sr0) / math.sqrt(se2)), sr0, math.sqrt(se2), (sr - sr0) / math.sqrt(se2)


show("dsr(sr=0.1,T=1250,skew=-0.5,kurt=5,N=50,sd=0.03)", dsr(0.1, 1250, -0.5, 5.0, 50, 0.03))
show("dsr(sr=0.05,T=500,skew=0.8,kurt=4,N=20,sd=0.02) floored", dsr(0.05, 500, 0.8, 4.0, 20, 0.02))
show("dsr same, unfloored", dsr(0.05, 500, 0.8, 4.0, 20, 0.02, floor=False))
show("dsr N=1", dsr(0.1, 1250, -0.5, 5.0, 1, 0.03))


def psr(sr, bench, T, skew, kurt, floor=True):
    se2 = (1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (T - 1)
    if floor:
        se2 = max(se2, 1 / (T - 1))
    return ND.cdf((sr - bench) / math.sqrt(se2))


show("psr(0.1, 0, 1250, -0.5, 5)", psr(0.1, 0.0, 1250, -0.5, 5.0))


def mintrl(sr, bench, skew, kurt, c):
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr * sr) * (ND.inv_cdf(c) / (sr - bench)) ** 2


show("mintrl(0.1, 0, -0.5, 5, 0.95)", mintrl(0.1, 0.0, -0.5, 5.0, 0.95))
show("mintrl(0.05, 0.02, 0.0, 3, 0.975)", mintrl(0.05, 0.02, 0.0, 3.0, 0.975))
show("z quantiles 0.95/0.975/0.8/0.99/0.999", [ND.inv_cdf(v) for v in (0.95, 0.975, 0.8, 0.99, 0.999)])

print("# --- BH")
def bh(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    q = [0.0] * m
    mn = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        mn = min(mn, min(1.0, p[i] * m / (rank + 1)))
        q[i] = mn
    return q


show("bh([0.01,0.04,0.03,0.005])", bh([0.01, 0.04, 0.03, 0.005]))
show("bh([0.001,0.2,0.05,0.3,0.02,0.02])", bh([0.001, 0.2, 0.05, 0.3, 0.02, 0.02]))

# ----------------------------------------------------------------------------------------------- PBO
print("# --- PBO CSCV brute force")


def pbo(cfgs, S, metric="sharpe"):
    T = len(cfgs[0])
    L = T // S
    start = T - L * S
    blocks = [[c[start + b * L: start + (b + 1) * L] for b in range(S)] for c in cfgs]

    def perf(vals):
        m = sum(vals) / len(vals)
        if metric == "mean":
            return m
        v = sum((a - m) ** 2 for a in vals) / (len(vals) - 1)
        return m / math.sqrt(v) if v > 1e-24 * (1 + m * m) else 0.0

    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        rest = [b for b in range(S) if b not in combo]
        isp = [perf([v for b in combo for v in blocks[i][b]]) for i in range(len(cfgs))]
        oop = [perf([v for b in rest for v in blocks[i][b]]) for i in range(len(cfgs))]
        best = 0
        for i in range(1, len(cfgs)):
            if isp[i] > isp[best]:
                best = i
        t = oop[best]
        below = sum(1 for v in oop if v < t)
        eq = sum(1 for i, v in enumerate(oop) if i != best and v == t)
        rank = below + 1 + eq / 2
        om = rank / (len(cfgs) + 1)
        logits.append(math.log(om / (1 - om)))
    return sum(1 for l in logits if l <= 0) / len(logits), logits


A = [0.02, 0.03, 0.01, 0.02, -0.02, -0.01, -0.03, -0.02, 0.02, 0.03, 0.01, 0.02, -0.02, -0.01, -0.03, -0.02]
Bc = [-a + 0.001 * ((i * 7) % 5 - 2) for i, a in enumerate(A)]
C = [0.001 * (((i * 5) % 7) - 3) for i in range(16)]
D = [0.004 * (1 if i % 2 == 0 else -1) + 0.0005 * (i % 3) for i in range(16)]
p, lg = pbo([A, Bc, C], 4)
show("pbo 3 cfgs S=4 (A,B anti-correlated in time, C noise)", p)
show("logits", lg)
p, lg = pbo([A, Bc, C, D], 8)
show("pbo 4 cfgs S=8", p)
show("logits S=8 first6", lg[:6])
dom = [0.01 + 0.002 * ((i * 3) % 4) for i in range(16)]
worse = [d - 0.02 + 0.001 * (i % 2) for i, d in enumerate(dom)]
p, lg = pbo([dom, worse], 4)
show("pbo dominant vs worse S=4", p)
show("A", A)
show("Bc", Bc)
show("C", C)
show("D", D)

# ----------------------------------------------------------------------------------------------- folds
print("# --- folds (brute force)")


def wf(n, k, tl, mt, purge, roll=None):
    first = n - k * tl
    out = []
    for f in range(k):
        ts = first + f * tl
        te = ts + tl
        tr_end = ts - purge
        tr_start = 0 if roll is None else max(0, tr_end - roll)
        out.append(((tr_start, tr_end), (ts, te)))
    return out


show("wf n=100 k=4 tl=10 min=20 purge=5 exp", wf(100, 4, 10, 20, 5))
show("wf n=100 k=4 tl=10 min=20 purge=5 roll=30", wf(100, 4, 10, 20, 5, 30))


def kfold(n, k, purge, emb):
    base, rem = divmod(n, k)
    start = 0
    out = []
    for g in range(k):
        ln = base + (1 if g < rem else 0)
        a, b = start, start + ln
        start = b
        forb = set(range(max(0, a - purge), min(n, b + emb)))
        train = [i for i in range(n) if i not in forb]
        # compress to ranges
        rs = []
        for i in train:
            if rs and rs[-1][1] == i:
                rs[-1][1] = i + 1
            else:
                rs.append([i, i + 1])
        out.append(((a, b), [tuple(r) for r in rs]))
    return out


show("kfold n=23 k=4 purge=2 emb=3", kfold(23, 4, 2, 3))

# ----------------------------------------------------------------------------------------------- Politis-White
print("# --- Politis-White block length (float reference)")


def pw(x):
    n = len(x)
    mu = sum(x) / n
    R = lambda k: sum((x[t] - mu) * (x[t - k] - mu) for t in range(k, n)) / n
    log10n = math.log(n) / math.log(10)
    kn = max(5, math.ceil(math.sqrt(log10n)))
    m_max = min(math.ceil(math.sqrt(n)) + kn, n - 1)
    r = [R(k) for k in range(m_max + 1)]
    crit = 2 * math.sqrt(log10n / n)
    mhat = max(m_max - kn, 0)
    if m_max >= kn:
        for m in range(0, m_max - kn + 1):
            if all(abs(r[m + j] / r[0]) < crit for j in range(1, kn + 1)):
                mhat = m
                break
    M = min(2 * max(mhat, 1), m_max)
    g = r[0]
    G = 0.0
    for k in range(1, M + 1):
        s = k / M
        lam = 1.0 if s <= 0.5 else 2 * (1 - s)
        g += 2 * lam * r[k]
        G += 2 * lam * k * r[k]
    bmax = math.ceil(min(3 * math.sqrt(n), n / 3))
    if not g > 0 or G == 0:
        return 1.0
    b = (n * G * G / (g * g)) ** (1 / 3)
    return min(max(b, 1.0), max(bmax, 1.0))


def ar1_series(n, phi, seed):
    # deterministic, uses the same xoshiro + libm Box-Muller (used only to make a fixture; the Rust test rebuilds it)
    r = Xoshiro.seeded(seed)
    vals = []
    x = 0.0
    spare = None
    for t in range(n + 50):
        if spare is not None:
            e = spare
            spare = None
        else:
            a, b = r.normal_pair()
            e, spare = a, b
        x = phi * x + e
        if t >= 50:
            vals.append(x)
    return vals


for phi in (0.0, 0.5, 0.8):
    s = ar1_series(400, phi, 11)
    show(f"pw AR({phi}) n=400 seed 11 (libm; compare to 1e-6)", pw(s))
alt = [(-1) ** t * 0.01 + 0.001 * ((t * 7) % 5) for t in range(60)]
show("pw deterministic alternating n=60", pw(alt))
trend = [0.001 * t + 0.01 * (((t * 13) % 7) - 3) for t in range(80)]
show("pw deterministic trend n=80", pw(trend))

# ----------------------------------------------------------------------------------------------- marginal test
print("# --- paired marginal test, independent implementation (same RNG and index scheme, plain float arithmetic)")


def q7(sorted_v, q):
    n = len(sorted_v)
    h = min(max(q, 0.0), 1.0) * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return sorted_v[lo] + (h - lo) * (sorted_v[hi] - sorted_v[lo])


def marginal_series(n):
    base = [(((t * 37 + 11) % 23) - 11) / 1000.0 for t in range(n)]
    comb = [0.8 * base[t] + (((t * 29 + 5) % 19) - 9) / 1500.0 + 0.0004 for t in range(n)]
    return base, comb


def sd1(x):
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def marginal(base, comb, ppy, B, block, seed, alpha=0.05, power=0.8, ci=0.95, exante=None):
    n = len(base)
    m0, m1 = sum(base) / n, sum(comb) / n
    s0, s1 = sd1(base), sd1(comb)
    root = math.sqrt(ppy)
    if exante:
        f0, f1 = exante
    else:
        f0, f1 = s0, s1
    delta = (m1 / f1 - m0 / f0) * root
    rng = Xoshiro.seeded(seed)
    reps = []
    for _ in range(B):
        idx = stationary_indices(n, block, rng)
        b = [base[i] for i in idx]
        c = [comb[i] for i in idx]
        if exante:
            r0, r1 = f0, f1
        else:
            r0, r1 = sd1(b), sd1(c)
        reps.append((sum(c) / n / r1 - sum(b) / n / r0) * root)
    ge = sum(1 for v in reps if v - delta >= delta)
    ge_abs = sum(1 for v in reps if abs(v - delta) >= abs(delta))
    p1 = (1 + ge) / (B + 1)
    p2 = min((1 + ge_abs) / (B + 1), 1.0)
    mu = sum(reps) / B
    se = math.sqrt(sum((v - mu) ** 2 for v in reps) / (B - 1))
    srt = sorted(reps)
    tail = (1 - ci) / 2
    mde = (ND.inv_cdf(1 - alpha) + ND.inv_cdf(power)) * se
    return dict(delta=delta, se=se, p1=p1, p2=p2, lo=q7(srt, tail), hi=q7(srt, 1 - tail), mde=mde,
                sr0=m0 / s0 * root, sr1=m1 / s1 * root)


b, c = marginal_series(60)
for name, kw in [("n=60 B=199 block=3 seed=99", dict(B=199, block=3.0, seed=99)),
                 ("n=60 B=99 block=1 seed=5", dict(B=99, block=1.0, seed=5)),
                 ("n=60 B=99 block=6.5 seed=77 exante", dict(B=99, block=6.5, seed=77, exante=(sd1(b) * 1.1, sd1(c) * 0.9)))]:
    r = marginal(b, c, 252.0, **kw)
    print(f"# {name}")
    for k, v in r.items():
        show(k, v)
# a series with a clearly positive contribution and one with none (for p-value ordering tests)
b2 = [(((t * 41 + 3) % 27) - 13) / 900.0 for t in range(80)]
c2 = [b2[t] + 0.0006 + (((t * 13 + 1) % 17) - 8) / 4000.0 for t in range(80)]
r = marginal(b2, c2, 252.0, B=199, block=2.0, seed=3)
print("# n=80 planted-shift series B=199 block=2 seed=3")
for k, v in r.items():
    show(k, v)
