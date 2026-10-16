//! Seeded pseudo-random numbers implemented in this crate. Nothing is ever read from the operating system, the clock
//! or the environment: the same seed gives the same stream on every machine and every run (design: deterministic).
//!
//! * `SplitMix64` (Steele, Lea, Flood 2014) expands a 64-bit seed into the xoshiro state.
//! * `xoshiro256**` (Blackman, Vigna 2018) is the generator.
//! * Bounded integers use Lemire's multiply-and-reject method (exactly uniform, integer arithmetic only).
//! * Normals use Box-Muller on the deterministic `ln` / `sin_cos_2pi` of [`crate::detmath`].

use crate::detmath;

/// One step of SplitMix64.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// xoshiro256** generator with a cached Box-Muller spare.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    /// Expand `seed` with SplitMix64. Any seed (including 0) is valid.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Rng { s, spare: None }
    }

    /// An independent stream derived from `(seed, stream)`; used so that parallel workers and Monte-Carlo cells never
    /// share draws and the result does not depend on how work is split across threads.
    pub fn from_stream(seed: u64, stream: u64) -> Self {
        let mut sm = seed;
        let a = splitmix64(&mut sm);
        let mut sm2 = stream ^ a;
        let mixed = splitmix64(&mut sm2) ^ a.rotate_left(17);
        Rng::seed_from_u64(mixed)
    }

    /// Construct from a raw state (test vectors). The all-zero state is invalid and is replaced by a fixed one.
    pub fn from_state(s: [u64; 4]) -> Self {
        if s == [0; 4] {
            return Rng::seed_from_u64(0);
        }
        Rng { s, spare: None }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `[0, n)`; `n` must be positive (returns 0 for `n == 0` rather than panicking).
    pub fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut lo = m as u64;
        if lo < n {
            let threshold = n.wrapping_neg() % n;
            while lo < threshold {
                m = (self.next_u64() as u128) * (n as u128);
                lo = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Standard normal by Box-Muller (two values per pair of uniforms, the second is cached).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // u1 in (0, 1] so that ln(u1) is finite
        let u1 = ((self.next_u64() >> 11) + 1) as f64 * (1.0 / 9_007_199_254_740_992.0);
        let u2 = self.unit();
        let radius = (-2.0 * detmath::ln(u1)).sqrt();
        let (s, c) = detmath::sin_cos_2pi(u2);
        self.spare = Some(radius * s);
        radius * c
    }
}
