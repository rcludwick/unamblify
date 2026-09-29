// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! **Do the sibilants survive?** An "s", "sh" or "f" is broadband noise
//! that lives above 4 kHz — the band a narrowband codec never carries and
//! a post-filter has to invent. A regression model that is unsure whether
//! a frame holds a loud hiss or nothing predicts the middle, and the
//! middle of "loud or absent" is "quiet": the owner heard the "s" go dull
//! before any column showed it (experiment log #36).
//!
//! [`sibilant_excess`] finds the sibilant frames in the *reference* — the
//! loudest [`TOP_SHARE`] of its 10 ms frames by 4–8 kHz energy, above a
//! floor — and at those frames compares the other signal's level in that
//! band with the reference's, and its high-to-mid balance (4–8 kHz over
//! 1–4 kHz) with the reference's. The level says whether the "s" is
//! there; the balance says whether it is an "s" or a dulled hum of the
//! right loudness. Both are signal minus reference in dB, so 0 is "as
//! bright as the recording". 16 kHz only: the band does not exist below
//! that.

use crate::spectral::stft;

/// Analysis rate the bands are defined at.
pub const RATE: u32 = 16_000;
/// FFT size (32 ms) and hop (10 ms) at [`RATE`].
const N_FFT: usize = 512;
const HOP: usize = 160;
/// Share of the reference's frames, loudest in 4–8 kHz first, taken as
/// sibilants.
pub const TOP_SHARE: f32 = 0.08;
/// A reference frame below this much high-band energy (dB, sum of
/// squared magnitudes) is never a sibilant, however loud relative to
/// the rest of a quiet clip.
const FLOOR_DB: f32 = -50.0;

/// How one signal's sibilants compare with a reference's, averaged over
/// the reference's sibilant frames. Signal minus reference, so 0 is "as
/// bright as the clean speech".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SibilantStats {
    /// Sibilant frames found in the reference.
    pub frames: usize,
    /// 4–8 kHz level difference, dB. Negative: the "s" is quieter.
    pub level_db: f32,
    /// High-to-mid balance difference — (4–8 kHz) − (1–4 kHz), signal
    /// minus reference — dB. Negative: duller than the recording even
    /// where it is as loud.
    pub balance_db: f32,
}

/// Per frame: (1–4 kHz energy, 4–8 kHz energy) in dB.
fn band_energies(x: &[f32]) -> Vec<(f32, f32)> {
    let hz_per_bin = RATE as f32 / N_FFT as f32;
    let bin = |hz: f32| (hz / hz_per_bin).round() as usize;
    let (lo0, lo1, hi0, hi1) = (bin(1000.0), bin(4000.0), bin(4000.0), bin(8000.0));
    stft(x, N_FFT, HOP)
        .iter()
        .map(|mag| {
            let e = |a: usize, b: usize| {
                let s: f32 = mag[a..b.min(mag.len())].iter().map(|m| m * m).sum();
                10.0 * (s + 1e-12).log10()
            };
            (e(lo0, lo1), e(hi0, hi1))
        })
        .collect()
}

/// Compare `a`'s sibilants with `b`'s at the sibilant frames of `b`.
/// `None` unless `rate` is [`RATE`] and `b` holds at least one sibilant
/// frame above the floor.
#[must_use]
pub fn sibilant_excess(a: &[f32], b: &[f32], rate: u32) -> Option<SibilantStats> {
    if rate != RATE {
        return None;
    }
    let n = a.len().min(b.len());
    let (ea, eb) = (band_energies(&a[..n]), band_energies(&b[..n]));
    let frames = ea.len().min(eb.len());
    let mut live: Vec<usize> = (0..frames).filter(|&t| eb[t].1 > FLOOR_DB).collect();
    if live.is_empty() {
        return None;
    }
    live.sort_by(|&p, &q| {
        eb[q]
            .1
            .partial_cmp(&eb[p].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let keep = ((live.len() as f32 * TOP_SHARE).ceil() as usize).max(1);
    let sel = &live[..keep.min(live.len())];
    let mean = |f: &dyn Fn(usize) -> f32| {
        #[allow(clippy::cast_precision_loss)]
        let n = sel.len() as f32;
        sel.iter().map(|&t| f(t)).sum::<f32>() / n
    };
    Some(SibilantStats {
        frames: sel.len(),
        level_db: mean(&|t| ea[t].1 - eb[t].1),
        balance_db: mean(&|t| (ea[t].1 - ea[t].0) - (eb[t].1 - eb[t].0)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White noise, band-limited by a crude FIR, at 16 kHz.
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let v = (s >> 16) as f32 / 65_536.0 - 0.5;
                v
            })
            .collect()
    }

    /// A vowel-like tone with an "s" (a noise burst) in the middle; the
    /// same with the burst's high band halved is 6 dB down in level and
    /// in balance, and a copy is 0.
    #[test]
    fn a_quieter_s_measures_as_quieter() {
        let n = 16_000;
        let mut clean: Vec<f32> = (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 16_000.0;
                0.2 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            })
            .collect();
        let burst = noise(3200, 3);
        // A first-difference makes the noise bright: energy tilts to 4–8 kHz.
        let bright: Vec<f32> = burst.windows(2).map(|w| 0.5 * (w[1] - w[0])).collect();
        for (i, v) in bright.iter().enumerate() {
            clean[6400 + i] += v;
        }
        // "quiet": the whole burst at half amplitude — 6 dB down, same colour.
        // "muffled": the high band halved and a 2.5 kHz component added, so
        // the mid band is louder than the recording's: duller by construction.
        let tone = |i: usize| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 16_000.0;
            0.2 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
        };
        let mid = |i: usize| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 16_000.0;
            0.15 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin()
        };
        let (mut quiet, mut muffled) = (clean.clone(), clean.clone());
        for i in 0..bright.len() {
            quiet[6400 + i] = tone(6400 + i) + 0.5 * bright[i];
            muffled[6400 + i] = tone(6400 + i) + 0.5 * bright[i] + mid(6400 + i);
        }
        let same = sibilant_excess(&clean, &clean, RATE).unwrap();
        assert!(same.frames > 0);
        assert!(same.level_db.abs() < 1e-3 && same.balance_db.abs() < 1e-3);
        let q = sibilant_excess(&quiet, &clean, RATE).unwrap();
        assert!((q.level_db + 6.0).abs() < 1.0, "half the burst: {q:?}");
        assert!(q.balance_db.abs() < 1.0, "same colour: {q:?}");
        let m = sibilant_excess(&muffled, &clean, RATE).unwrap();
        assert!(m.balance_db < -3.0, "duller: {m:?}");
        assert!(sibilant_excess(&clean, &clean, 8_000).is_none());
    }
}
