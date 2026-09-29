// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Min/max-bucket downsampling for metric series (spec §8: ≤ 4 K points per
//! series). Each bucket of consecutive points contributes its minimum and
//! maximum value, in step order, so spikes survive and the chart's envelope
//! is exact; the first and last points are always kept so the series spans
//! its true range and the current value is visible.

use serde::Serialize;

/// Columnar series as the API returns it.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Series {
    /// Step per point.
    pub step: Vec<u64>,
    /// Unix milliseconds per point.
    pub t: Vec<u64>,
    /// Value per point.
    pub v: Vec<f64>,
}

impl Series {
    /// Number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.step.len()
    }

    /// No points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.step.is_empty()
    }

    /// Append a point.
    pub fn push(&mut self, step: u64, t: u64, v: f64) {
        self.step.push(step);
        self.t.push(t);
        self.v.push(v);
    }
}

/// The default cap.
pub const MAX_POINTS: usize = 4_000;

/// Reduce `s` to at most `max_points` points (`max_points >= 4`; smaller
/// values are raised to 4). Series already within the cap are returned
/// unchanged.
#[must_use]
pub fn min_max(s: &Series, max_points: usize) -> Series {
    let n = s.len();
    let cap = max_points.max(4);
    if n <= cap {
        return s.clone();
    }
    // Two points per bucket, two slots reserved for the first and last points.
    let buckets = ((cap - 2) / 2).max(1);
    let width = n.div_ceil(buckets);
    let mut out = Series::default();
    out.push(s.step[0], s.t[0], s.v[0]);
    for chunk in (0..n).step_by(width) {
        let end = (chunk + width).min(n);
        let (mut lo, mut hi) = (chunk, chunk);
        for i in chunk..end {
            if s.v[i] < s.v[lo] {
                lo = i;
            }
            if s.v[i] > s.v[hi] {
                hi = i;
            }
        }
        let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        if a != 0 {
            out.push(s.step[a], s.t[a], s.v[a]);
        }
        if b != a && b != 0 {
            out.push(s.step[b], s.t[b], s.v[b]);
        }
    }
    let last = n - 1;
    if out.step.last().copied() != Some(s.step[last]) {
        out.push(s.step[last], s.t[last], s.v[last]);
    }
    debug_assert!(out.len() <= cap);
    out
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Series {
        let mut s = Series::default();
        for i in 0..n {
            let f = i as f64;
            s.push(
                i as u64,
                1000 + i as u64,
                (f * 0.37).sin() * 10.0 + f * 0.001,
            );
        }
        s
    }

    #[test]
    fn short_series_pass_through() {
        let s = ramp(100);
        assert_eq!(min_max(&s, 4000), s);
        assert_eq!(min_max(&s, 100), s);
    }

    #[test]
    fn long_series_are_capped_and_keep_extremes_and_last_point() {
        let mut s = ramp(100_000);
        s.v[54_321] = 999.0;
        s.v[12_345] = -999.0;
        let d = min_max(&s, 4000);
        assert!(d.len() <= 4000, "{}", d.len());
        assert!(d.len() >= 3900, "{}", d.len());
        assert!(d.v.contains(&999.0));
        assert!(d.v.contains(&-999.0));
        assert_eq!(d.step.last(), Some(&99_999));
        assert_eq!(d.step[0], 0);
        assert!(d.step.windows(2).all(|w| w[0] < w[1]), "monotonic steps");
        assert_eq!(d.step.len(), d.t.len());
        assert_eq!(d.step.len(), d.v.len());
    }

    #[test]
    fn tiny_caps_are_raised_to_four() {
        let s = ramp(50);
        let d = min_max(&s, 1);
        assert!(d.len() <= 4 && !d.is_empty(), "{}", d.len());
        assert_eq!(d.step[0], 0);
        assert_eq!(d.step.last(), Some(&49));
    }
}
