//! The PRNG against published/independent vectors and the stationary bootstrap against an independent implementation
//! (`tests/reference/gen_reference.py`: same integer scheme written separately in Python), with pinned digests.

use portfolio_eval::bootstrap::*;
use portfolio_eval::error::EvalError;
use portfolio_eval::rng::{splitmix64, Rng};

#[test]
fn splitmix64_matches_the_published_vectors_for_seed_zero() {
    let mut s = 0u64;
    assert_eq!(splitmix64(&mut s), 0xe220a8397b1dcdaf);
    assert_eq!(splitmix64(&mut s), 0x6e789e6aa1b965f4);
    assert_eq!(splitmix64(&mut s), 0x06c45d188009454f);
}

#[test]
fn xoshiro256ss_matches_the_reference_vector() {
    // State {1, 2, 3, 4}: the reference implementation's first outputs.
    let mut r = Rng::from_state([1, 2, 3, 4]);
    assert_eq!(r.next_u64(), 11520);
    assert_eq!(r.next_u64(), 0);
    assert_eq!(r.next_u64(), 1509978240);
    assert_eq!(r.next_u64(), 1215971899390074240);
}

#[test]
fn seeded_stream_matches_the_independent_python_implementation() {
    let mut r = Rng::seed_from_u64(42);
    assert_eq!(r.next_u64(), 1546998764402558742);
    assert_eq!(r.next_u64(), 6990951692964543102);
    assert_eq!(r.next_u64(), 12544586762248559009);
}

#[test]
fn bounded_integers_match_the_independent_implementation() {
    let mut r = Rng::seed_from_u64(42);
    let got: Vec<u64> = (0..8).map(|_| r.below(10)).collect();
    assert_eq!(got, vec![0, 3, 6, 9, 9, 7, 7, 8]);
}

#[test]
fn unit_floats_match_and_stay_in_range() {
    let mut r = Rng::seed_from_u64(9);
    assert_eq!(r.unit(), 0.0025834396857136177);
    assert_eq!(r.unit(), 0.25148937241585745);
    assert_eq!(r.unit(), 0.13246225011289547);
    let mut r = Rng::seed_from_u64(1);
    for _ in 0..10_000 {
        let u = r.unit();
        assert!((0.0..1.0).contains(&u));
    }
}

#[test]
fn normals_match_libm_box_muller_within_1e_12_and_have_the_right_moments() {
    let mut r = Rng::seed_from_u64(5);
    let want = [-1.2635341310943198, -0.943513318909646, 0.40369911386871266, -0.8366528667135981];
    for w in want {
        let z = r.normal();
        assert!((z - w).abs() < 1e-12, "{z} vs {w}");
    }
    let mut r = Rng::seed_from_u64(77);
    let n = 200_000;
    let z: Vec<f64> = (0..n).map(|_| r.normal()).collect();
    let mean = z.iter().sum::<f64>() / n as f64;
    let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    let kurt = z.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n as f64 / (var * var);
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.01, "var {var}");
    assert!((kurt - 3.0).abs() < 0.06, "kurtosis {kurt}");
}

#[test]
fn below_is_uniform_without_modulo_bias() {
    let mut r = Rng::seed_from_u64(31337);
    let k = 7u64;
    let n = 140_000;
    let mut counts = [0usize; 7];
    for _ in 0..n {
        counts[r.below(k) as usize] += 1;
    }
    let expected = n as f64 / k as f64;
    let chi2: f64 = counts.iter().map(|c| (*c as f64 - expected).powi(2) / expected).sum();
    // 6 degrees of freedom: 99.9th percentile is 22.5
    assert!(chi2 < 22.5, "chi2 {chi2} counts {counts:?}");
    assert_eq!(r.below(0), 0);
    assert_eq!(r.below(1), 0);
}

#[test]
fn streams_are_independent_and_reproducible() {
    let mut a = Rng::from_stream(1, 0);
    let mut b = Rng::from_stream(1, 1);
    let mut a2 = Rng::from_stream(1, 0);
    let (x, y, z) = (a.next_u64(), b.next_u64(), a2.next_u64());
    assert_ne!(x, y);
    assert_eq!(x, z);
    // the all-zero state is repaired instead of producing a stuck generator
    let mut r = Rng::from_state([0; 4]);
    assert_ne!(r.next_u64(), 0);
}

#[test]
fn stationary_indices_match_the_independent_implementation() {
    let idx = bootstrap_indices(20, 4.0, 7).unwrap();
    assert_eq!(idx, vec![14, 15, 16, 17, 18, 19, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 2, 13, 14, 15]);
    assert_eq!(indices_digest(&idx), "b91c86cc34ebc713b8c753e135574581e844ce8c9ef83f39d64f8cc4107f5521");
    let idx = bootstrap_indices(10, 1.0, 1).unwrap();
    assert_eq!(idx, vec![7, 5, 5, 3, 6, 1, 0, 3, 8, 5]);
    assert_eq!(
        indices_digest(&bootstrap_indices(1000, 10.0, 123).unwrap()),
        "27963599b79e50e408439263825d023e6def3fdb978be62420d65e8a6a241a2f"
    );
    assert_eq!(
        indices_digest(&bootstrap_indices(500, 3.0, 2026).unwrap()),
        "37253934cb49468b2495c270564393c4fdebaa38defef100c89f36cc5b520309"
    );
}

#[test]
fn a_continuing_block_wraps_circularly() {
    // block length so long that (almost) every step continues: consecutive indices differ by +1 mod n
    let n = 50;
    let idx = bootstrap_indices(n, n as f64, 3).unwrap();
    let mut continues = 0;
    for w in idx.windows(2) {
        if w[1] == (w[0] + 1) % n {
            continues += 1;
        }
    }
    assert!(continues >= 40, "continues {continues}");
    assert!(idx.iter().all(|i| *i < n));
}

#[test]
fn mean_block_length_is_the_requested_one() {
    // The number of block starts among n draws is about n / b, so the realised mean block is about b (geometric law).
    for (b, seed) in [(2.0, 1u64), (5.0, 2), (20.0, 3)] {
        let n = 200_000;
        let bs = StationaryBootstrap::new(n, b).unwrap();
        let mut rng = portfolio_eval::rng::Rng::seed_from_u64(seed);
        let mut out = vec![0usize; n];
        bs.fill(&mut rng, &mut out);
        // count breaks of the "previous + 1" pattern (a random restart can also happen to continue, prob 1/n: negligible)
        let starts = 1 + out.windows(2).filter(|w| w[1] != (w[0] + 1) % n).count();
        let realised = n as f64 / starts as f64;
        assert!((realised - b).abs() / b < 0.05, "b={b} realised {realised}");
        assert_eq!(bs.mean_block(), b);
    }
}

#[test]
fn block_length_one_is_the_iid_bootstrap() {
    let n = 100_000;
    let idx = bootstrap_indices(n, 1.0, 11).unwrap();
    let continues = idx.windows(2).filter(|w| w[1] == (w[0] + 1) % n).count();
    // iid draws continue with probability 1/n per step: expected 1, certainly below 20
    assert!(continues < 20, "{continues}");
}

#[test]
fn bootstrap_statistic_runs_replicates_in_order_and_is_deterministic() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let mean_of = |idx: &[usize]| idx.iter().map(|i| x[*i]).sum::<f64>() / idx.len() as f64;
    let a = bootstrap_statistic(30, 3.0, 50, 9, mean_of).unwrap();
    let b = bootstrap_statistic(30, 3.0, 50, 9, mean_of).unwrap();
    assert_eq!(a.len(), 50);
    assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let c = bootstrap_statistic(30, 3.0, 50, 10, mean_of).unwrap();
    assert_ne!(a, c);
    // the bootstrap mean is centred on the sample mean (14.5)
    let big = bootstrap_statistic(30, 3.0, 4000, 1, mean_of).unwrap();
    let m = big.iter().sum::<f64>() / big.len() as f64;
    assert!((m - 14.5).abs() < 0.2, "{m}");
}

#[test]
fn bootstrap_rejects_bad_parameters_with_typed_errors() {
    assert!(matches!(StationaryBootstrap::new(1, 2.0), Err(EvalError::TooShort { .. })));
    assert!(matches!(StationaryBootstrap::new(10, 0.5), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(StationaryBootstrap::new(10, 11.0), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(StationaryBootstrap::new(10, f64::NAN), Err(EvalError::InvalidParameter { .. })));
    assert!(matches!(StationaryBootstrap::new(10, f64::INFINITY), Err(EvalError::InvalidParameter { .. })));
    assert!(percentile_ci(&[1.0, 2.0, 3.0], 1.0).is_err());
    assert!(percentile_ci(&[1.0], 0.9).is_err());
    assert!(percentile_ci(&[f64::NAN, f64::NAN, 1.0], 0.9).is_err());
}

#[test]
fn percentile_ci_uses_type_7_quantiles() {
    let v: Vec<f64> = (1..=101).map(|i| i as f64).collect();
    let (lo, hi) = percentile_ci(&v, 0.9).unwrap();
    assert!((lo - 6.0).abs() < 1e-12 && (hi - 96.0).abs() < 1e-12, "{lo} {hi}");
}

fn ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    // Same construction as gen_reference.py::ar1_series (Box-Muller pairs, 50 burn-in draws).
    let mut r = Rng::seed_from_u64(seed);
    let mut x = 0.0;
    (0..n + 50)
        .map(|_| {
            x = phi * x + r.normal();
            x
        })
        .skip(50)
        .collect()
}

#[test]
fn politis_white_matches_the_independent_implementation_on_deterministic_series() {
    let alt: Vec<f64> =
        (0..60).map(|t| (if t % 2 == 0 { 1.0 } else { -1.0 }) * 0.01 + 0.001 * ((t * 7) % 5) as f64).collect();
    let b = politis_white_block_length(&alt).unwrap();
    assert!((b - 9.34054669356776).abs() < 1e-9, "{b}");
    let trend: Vec<f64> = (0..80).map(|t| 0.001 * t as f64 + 0.01 * (((t * 13) % 7) as f64 - 3.0)).collect();
    let b = politis_white_block_length(&trend).unwrap();
    assert!((b - 12.21521987886635).abs() < 1e-9, "{b}");
}

#[test]
fn politis_white_matches_the_python_reference_on_ar1_data() {
    // libm Box-Muller in Python vs detmath in Rust: same to about 1e-12 in the data, so 1e-6 in the block length
    for (phi, want) in [(0.0, 1.3058908127706719), (0.5, 6.674041764081687), (0.8, 14.197324114043079)] {
        let b = politis_white_block_length(&ar1(400, phi, 11)).unwrap();
        assert!((b - want).abs() < 1e-6, "phi {phi}: {b} vs {want}");
    }
}

#[test]
fn politis_white_grows_with_persistence_and_is_scale_and_shift_invariant() {
    let mut prev = 0.0;
    for phi in [0.0, 0.3, 0.6, 0.85] {
        let mut mean_b = 0.0;
        for seed in 0..12u64 {
            mean_b += politis_white_block_length(&ar1(600, phi, 100 + seed)).unwrap() / 12.0;
        }
        assert!(mean_b > prev, "phi {phi}: {mean_b} not above {prev}");
        prev = mean_b;
    }
    let s = ar1(500, 0.5, 5);
    let t: Vec<f64> = s.iter().map(|v| 3.0 * v + 7.0).collect();
    let (a, b) = (politis_white_block_length(&s).unwrap(), politis_white_block_length(&t).unwrap());
    assert!((a - b).abs() < 1e-9, "{a} vs {b}");
}

#[test]
fn politis_white_is_clamped_to_the_documented_range() {
    let iid = ar1(400, 0.0, 3);
    let b = politis_white_block_length(&iid).unwrap();
    assert!(b >= 1.0);
    let trending: Vec<f64> = (0..300).map(|t| t as f64).collect();
    let b = politis_white_block_length(&trending).unwrap();
    let bmax = (3.0 * 300f64.sqrt()).min(100.0).ceil();
    assert!(b <= bmax + 1e-9 && b >= 1.0, "{b} vs {bmax}");
}

#[test]
fn politis_white_errors_are_typed() {
    assert!(matches!(politis_white_block_length(&[0.1; 5]), Err(EvalError::TooShort { .. })));
    assert!(matches!(politis_white_block_length(&[0.1; 30]), Err(EvalError::ZeroVariance { .. })));
    let mut bad = ar1(30, 0.1, 1);
    bad[3] = f64::NAN;
    assert!(matches!(politis_white_block_length(&bad), Err(EvalError::NonFinite { index: 3, .. })));
    assert!(auto_block_length(&[]).is_err());
}

#[test]
fn auto_block_length_is_the_max_over_series() {
    let a = ar1(400, 0.0, 21);
    let b = ar1(400, 0.8, 22);
    let (la, lb) = (politis_white_block_length(&a).unwrap(), politis_white_block_length(&b).unwrap());
    let both = auto_block_length(&[&a, &b]).unwrap();
    assert_eq!(both, la.max(lb));
    assert!(lb > la);
}
