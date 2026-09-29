// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! One small deterministic generator (`SplitMix64`) shared by the shard
//! builder and the training loaders, so an example drawn at shard time
//! and one drawn on the fly follow the same distribution from the same
//! seed, and nothing depends on a third-party RNG's version.

/// `SplitMix64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Derive an independent stream (per epoch, per worker, per example).
    #[must_use]
    pub fn fork(&mut self, salt: u64) -> Self {
        let s = self.next_u64() ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Self(s)
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 24 random bits (exact in f32).
    pub fn next_f32(&mut self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        v
    }

    /// Uniform in `[0, 1)` with 53 random bits (exact in f64).
    pub fn next_f64(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v
    }

    /// Uniform integer in `[0, n)`; 0 when `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // Modulo bias is < 2^-40 for n < 2^24; fine here.
        usize::try_from(self.next_u64() % n as u64).unwrap_or(0)
    }

    /// Uniform integer in `[lo, hi]` (inclusive; `hi < lo` returns `lo`).
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        lo + self.below(hi - lo + 1)
    }

    /// Bernoulli with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        #[allow(clippy::cast_possible_truncation)]
        let v = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        v
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_in_range_and_reasonably_uniform() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
            let r = a.range(3, 9);
            assert!((3..=9).contains(&r));
            let f = b.next_f64();
            assert!((0.0..1.0).contains(&f));
            let _ = b.range(3, 9);
            let _ = a.next_f64();
        }
        let mut r = Rng::new(1);
        let mean: f32 = (0..10_000).map(|_| r.next_f32()).sum::<f32>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        let mut counts = [0usize; 5];
        for _ in 0..5_000 {
            counts[r.below(5)] += 1;
        }
        assert!(counts.iter().all(|&c| c > 800), "{counts:?}");
        let mut v: Vec<u32> = (0..20).collect();
        Rng::new(1).shuffle(&mut v);
        let mut w: Vec<u32> = (0..20).collect();
        Rng::new(1).shuffle(&mut w);
        assert_eq!(v, w);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_ne!(v, sorted);
        assert_eq!(Rng::new(3).range(5, 2), 5);
        assert_eq!(Rng::new(3).below(0), 0);
        assert_ne!(Rng::new(3).fork(1), Rng::new(3).fork(2));
    }
}
