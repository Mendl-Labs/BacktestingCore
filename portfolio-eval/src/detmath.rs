//! Deterministic elementary functions built only from the IEEE-754 basic operations (`+ - * /`, `sqrt`, `round`, bit
//! reinterpretation).
//!
//! Why not `f64::exp` / `ln` / `sin` / `cos`? Those call the platform libm, whose last-bit results differ between
//! x86_64 glibc, aarch64 glibc, musl and macOS. Every pinned digest in this crate (bootstrap draws, the power table)
//! must be bit-identical on the CI runner and on the aarch64 developer box, so nothing on a pinned path may call libm.
//! `+ - * / sqrt` are correctly rounded by IEEE-754 and Rust never fuses multiply-adds, so the functions below give the
//! same bits everywhere.
//!
//! Accuracy (verified in `tests/detmath.rs` against published constants and the platform libm): `exp`, `ln` about
//! 1e-15 relative; `erfc` about 1e-15 relative for x in [0, 27]; `norm_isf`/`norm_ppf` about 1e-15. That is far
//! better than the Abramowitz-Stegun 7.1.26 approximation (1.5e-7 absolute) that Core's `metrics` uses; the crate's
//! `core_compat` functions reproduce Core's approximation explicitly where equivalence is being tested.

// The fdlibm constants below are kept with their full published decimal expansion for provenance.
#![allow(clippy::excessive_precision)]

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1; // fdlibm split: LN2_HI has 32 significant bits so k * LN2_HI is exact
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const INV_LN2: f64 = std::f64::consts::LOG2_E;
const SQRT2: f64 = std::f64::consts::SQRT_2;
const TWO_PI: f64 = std::f64::consts::TAU;
const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const SQRT_2PI: f64 = 2.506_628_274_631_000_7;

fn pow2(k: i32) -> f64 {
    // valid for -1022 <= k <= 1023 (normal range)
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// `e^x`. Range reduction `x = k ln2 + r`, `|r| <= 0.35`, then an 18-term Taylor series.
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x <= -745.2 {
        return 0.0;
    }
    let kf = (x * INV_LN2).round();
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..=18 {
        term = term * r / n as f64;
        sum += term;
    }
    let k = kf as i32;
    let k1 = k / 2;
    let k2 = k - k1;
    sum * pow2(k1) * pow2(k2)
}

/// Natural logarithm. `NaN` for negative or NaN input, `-inf` at 0, `+inf` at `+inf`.
pub fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    // Subnormals: scale into the normal range first.
    let (x, bias) = if x < f64::MIN_POSITIVE { (x * 18_014_398_509_481_984.0, -54.0) } else { (x, 0.0) };
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    for k in 0..16 {
        sum += term / (2 * k + 1) as f64;
        term *= s2;
    }
    let ef = e as f64 + bias;
    ef * LN2_HI + (ef * LN2_LO + 2.0 * sum)
}

/// `x^y` for `x > 0` as `exp(y ln x)`; `NaN` otherwise.
pub fn pow(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    exp(y * ln(x))
}

/// `(sin(2 pi u), cos(2 pi u))` for `u` in `[0, 1)`; quadrant reduction is exact because `u - q/4` is exact.
pub fn sin_cos_2pi(u: f64) -> (f64, f64) {
    let q = (u * 4.0).floor();
    let r = (u - q * 0.25) * TWO_PI; // in [0, pi/2)
    let r2 = r * r;
    let mut s_term = r;
    let mut s = r;
    let mut c_term = 1.0;
    let mut c = 1.0;
    for k in 1..=12 {
        let kf = k as f64;
        s_term = s_term * (-r2) / ((2.0 * kf) * (2.0 * kf + 1.0));
        s += s_term;
        c_term = c_term * (-r2) / ((2.0 * kf - 1.0) * (2.0 * kf));
        c += c_term;
    }
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

fn erf_series(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) e^{-x^2} sum_n 2^n x^{2n+1} / (2n+1)!!   (all terms positive: no cancellation)
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0u32;
    loop {
        n += 1;
        term = term * 2.0 * x2 / (2 * n + 1) as f64;
        sum += term;
        if term < 1e-18 * sum || n > 200 {
            break;
        }
    }
    TWO_OVER_SQRT_PI * exp(-x2) * sum
}

fn erfc_cf(x: f64) -> f64 {
    // erfc(x) = e^{-x^2}/sqrt(pi) / (x + (1/2)/(x + (2/2)/(x + (3/2)/(x + ...)))), evaluated bottom-up at fixed depth
    let mut t = x;
    for k in (1..=200).rev() {
        t = x + (k as f64 / 2.0) / t;
    }
    exp(-x * x) * INV_SQRT_PI / t
}

/// Complementary error function.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        1.0 - erf_series(x)
    } else if x > 27.3 {
        0.0
    } else {
        erfc_cf(x)
    }
}

/// Standard normal cumulative distribution function.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT2)
}

/// Standard normal survival function `1 - Phi(x)`, accurate in the far upper tail.
pub fn norm_sf(x: f64) -> f64 {
    0.5 * erfc(x / SQRT2)
}

fn norm_ppf_lower(p: f64) -> f64 {
    // p in (0, 0.5]: Acklam's rational approximation, then two Halley steps on the accurate cdf.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] =
        [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];
    let p_low = 0.02425;
    let mut x = if p < p_low {
        let q = (-2.0 * ln(p)).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    for _ in 0..2 {
        let e = norm_cdf(x) - p;
        let u = e * SQRT_2PI * exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
    }
    x
}

/// Standard normal quantile function `Phi^{-1}(p)`; `NaN` outside `[0, 1]`, `-inf`/`+inf` at 0/1.
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p < 0.5 {
        norm_ppf_lower(p)
    } else {
        -norm_ppf_lower(1.0 - p)
    }
}

/// Inverse survival function: the `z` with `1 - Phi(z) = q`. Accurate for tiny `q` (used for `Phi^{-1}(1 - 1/N)`).
pub fn norm_isf(q: f64) -> f64 {
    if q.is_nan() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return f64::NEG_INFINITY;
    }
    if q == 0.5 {
        return 0.0;
    }
    if q < 0.5 {
        -norm_ppf_lower(q)
    } else {
        norm_ppf_lower(1.0 - q)
    }
}
