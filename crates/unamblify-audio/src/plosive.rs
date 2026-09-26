// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! **Do the stop consonants survive?** A plosive (p, t, k, b, d, g) is a
//! closure — 30–100 ms of near silence — then a burst a few milliseconds
//! long. A frame-based vocoder codes that burst as one 20 ms noise frame:
//! measured on this project's eval clips, D-STAR turns a 3 ms rise into
//! 14 ms and a 10 ms burst into 30 ms, 13 dB down at its peak. A
//! post-filter can make that worse in a second way, by leaking energy
//! into the closure. Both are invisible to the whole-clip spectral
//! metrics, which average over windows longer than the burst.
//!
//! [`plosive_excess`] finds bursts in the *reference* — a rise of
//! [`RISE_DB`] or more within [`RISE_MS`] in the 2–3.8 kHz envelope at
//! 1 ms resolution, after a [`CLOSURE_MS`] closure — and at each one
//! compares the other signal's burst peak, closure depth and rise time
//! with the reference's. The band is inside what every mode transmits,
//! so a bandwidth-extension head that adds a high band cannot score
//! here; only sharpening the burst the codec smeared can.

use crate::metrics::{mean, power_spectrum};

/// Bottom of the analysis band, Hz.
pub const BAND_LO_HZ: f32 = 2000.0;
/// Top of the analysis band, Hz — inside a narrowband vocoder's 3.7 kHz.
pub const BAND_HI_HZ: f32 = 3800.0;
/// A burst is a rise of at least this, dB …
pub const RISE_DB: f32 = 15.0;
/// … within this many milliseconds …
pub const RISE_MS: usize = 5;
/// … after a closure this long in which nothing came within 8 dB of it.
pub const CLOSURE_MS: usize = 40;
/// Bursts closer together than this are one burst.
const REFRACTORY_MS: usize = 60;
/// Envelope window, ms (Hann).
const WINDOW_MS: u32 = 4;
/// Rise time is measured from this far below the peak.
const RISE_FROM_DB: f32 = 15.0;
/// Frames the envelope needs after a burst (the refractory check).
const LOOKAHEAD_MS: usize = 50;

/// How one signal's plosives compare with a reference's, averaged over
/// the reference's bursts. Every field is *signal minus reference*, so
/// 0 is "as sharp as the clean speech".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlosiveStats {
    /// Bursts found in the reference.
    pub bursts: usize,
    /// Burst peak level difference, dB. Negative: the burst is quieter.
    pub burst_db: f32,
    /// Closure depth difference (burst peak above the quietest millisecond
    /// of the preceding closure), dB. Negative: the silence before the
    /// burst has been filled in.
    pub closure_db: f32,
    /// Rise time difference, ms (from 15 dB below the peak to the peak).
    /// Positive: the onset is smeared.
    pub rise_ms: f32,
}

/// Compare `a`'s plosives with `b`'s at the bursts found in `b`. `None`
/// when `b` holds no burst, or either signal is too short to frame.
#[must_use]
pub fn plosive_excess(a: &[f32], b: &[f32], rate: u32) -> Option<PlosiveStats> {
    let n = a.len().min(b.len());
    let (ea, eb) = (envelope_db(&a[..n], rate), envelope_db(&b[..n], rate));
    let bursts = find_bursts(&eb);
    let mut stats = Vec::with_capacity(bursts.len());
    for &t in &bursts {
        if let (Some(sa), Some(sb)) = (burst_stats(&ea, t), burst_stats(&eb, t)) {
            stats.push((sa.0 - sb.0, sa.1 - sb.1, sa.2 - sb.2));
        }
    }
    (!stats.is_empty()).then(|| PlosiveStats {
        bursts: stats.len(),
        burst_db: mean(&stats.iter().map(|s| s.0).collect::<Vec<_>>()),
        closure_db: mean(&stats.iter().map(|s| s.1).collect::<Vec<_>>()),
        rise_ms: mean(&stats.iter().map(|s| s.2).collect::<Vec<_>>()),
    })
}

/// The 2–3.8 kHz energy of `x` per millisecond, dB, from a 4 ms Hann
/// window hopped 1 ms. Entry `t` covers samples starting at `t` ms.
#[must_use]
pub fn envelope_db(x: &[f32], rate: u32) -> Vec<f32> {
    let win = (rate * WINDOW_MS / 1000) as usize;
    let hop = (rate / 1000) as usize;
    if win == 0 || hop == 0 || x.len() < win {
        return Vec::new();
    }
    let window: Vec<f32> = (0..win)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win as f32).cos())
        .collect();
    let bin_hz = rate as f32 / win as f32;
    let lo = (BAND_LO_HZ / bin_hz).ceil() as usize;
    let hi = ((BAND_HI_HZ.min(rate as f32 / 2.0 - bin_hz)) / bin_hz).floor() as usize;
    (0..=(x.len() - win) / hop)
        .map(|f| {
            let spec = power_spectrum(&x[f * hop..f * hop + win], &window);
            let e: f32 = spec.iter().take(hi + 1).skip(lo).sum();
            10.0 * (e + 1e-10).log10()
        })
        .collect()
}

/// Millisecond indices of the bursts in an envelope.
#[must_use]
pub fn find_bursts(env: &[f32]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut t = CLOSURE_MS;
    while t + LOOKAHEAD_MS + RISE_MS < env.len() {
        let closure = &env[t - CLOSURE_MS..t];
        let pre_min = closure.iter().copied().fold(f32::INFINITY, f32::min);
        let pre_max = closure.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (rise_idx, post) = env[t..t + RISE_MS]
            .iter()
            .copied()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
        let ahead_max = env[t..t + LOOKAHEAD_MS]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if post - pre_min >= RISE_DB && pre_max < post - 8.0 && ahead_max <= post {
            out.push(t + rise_idx);
            t += REFRACTORY_MS;
        } else {
            t += 1;
        }
    }
    out
}

/// `(peak dB, closure depth dB, rise ms)` of the burst at `t`.
fn burst_stats(env: &[f32], t: usize) -> Option<(f32, f32, f32)> {
    if t < CLOSURE_MS || t + 3 > env.len() {
        return None;
    }
    let peak = env[t - 2..t + 3]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let closure = env[t - CLOSURE_MS..t - 3]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    // Milliseconds from the last sample below (peak − 15 dB) to the peak,
    // looking back 20 ms.
    let back = 20usize;
    let seg = &env[t - back..=t];
    let last_below = seg.iter().rposition(|&v| v < peak - RISE_FROM_DB);
    let rise = last_below.map_or(back, |i| back - i) as f32;
    Some((peak, peak - closure, rise))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn noise(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 16) as f32 / 65_536.0 - 0.5
    }

    /// 200 ms near-silence, a burst `burst_ms` long that fades in over
    /// `fade_ms`, then 150 ms of a low vowel-like tone. `floor` is the
    /// noise in the closure.
    fn stop(burst_ms: usize, fade_ms: usize, level: f32, floor: f32) -> Vec<f32> {
        let ms = RATE as usize / 1000;
        let mut seed = 7u32;
        let mut x = Vec::new();
        for _ in 0..200 * ms {
            x.push(noise(&mut seed) * floor);
        }
        let burst = burst_ms * ms;
        for i in 0..burst {
            let ramp = if fade_ms == 0 {
                1.0
            } else {
                (i as f32 / (fade_ms * ms) as f32).min(1.0)
            };
            x.push(noise(&mut seed) * level * ramp);
        }
        for i in 0..150 * ms {
            let t = i as f32 / RATE as f32;
            let mut v = 0.0;
            for k in 1..=6 {
                v += (2.0 * std::f32::consts::PI * 180.0 * k as f32 * t).sin() * 0.3 / k as f32;
            }
            x.push(v + noise(&mut seed) * floor);
        }
        x
    }

    #[test]
    fn a_sharp_burst_is_found_and_matches_itself() {
        let b = stop(8, 0, 0.5, 1e-4);
        let s = plosive_excess(&b, &b, RATE).expect("one burst");
        assert_eq!(s.bursts, 1);
        assert!(s.burst_db.abs() < 1e-6 && s.closure_db.abs() < 1e-6 && s.rise_ms.abs() < 1e-6);
        let env = envelope_db(&b, RATE);
        let t = find_bursts(&env)[0];
        assert!((198..=206).contains(&t), "burst at {t} ms, expected ~200");
        let (_, depth, rise) = burst_stats(&env, t).unwrap();
        assert!(depth > 40.0, "closure depth {depth} dB");
        assert!(
            rise <= 8.0,
            "a step onset rises within a couple of windows: {rise} ms"
        );
    }

    /// A codec-shaped burst — a third of the level, spread over 30 ms
    /// with a 20 ms fade-in — reads quieter, slower and, with noise in
    /// the closure, shallower.
    #[test]
    fn a_smeared_burst_with_a_filled_closure_reads_negative_and_slow() {
        let b = stop(8, 0, 0.5, 1e-4);
        let a = stop(30, 20, 0.17, 0.02);
        let s = plosive_excess(&a, &b, RATE).expect("the reference has a burst");
        assert_eq!(s.bursts, 1);
        assert!(s.burst_db < -6.0, "burst {:+.1} dB", s.burst_db);
        assert!(s.closure_db < -10.0, "closure {:+.1} dB", s.closure_db);
        assert!(s.rise_ms > 5.0, "rise {:+.1} ms", s.rise_ms);
        // And the bursts are the reference's: a signal with no burst of
        // its own is still measured, at the reference's instants.
        let flat = vec![1e-4; b.len()];
        let f = plosive_excess(&flat, &b, RATE).unwrap();
        assert!(f.burst_db < -40.0);
        // While a reference with no burst has nothing to compare.
        assert!(plosive_excess(&b, &flat, RATE).is_none());
    }

    #[test]
    fn a_vowel_onset_is_not_a_burst() {
        // No closure before it: the tone fades in over 60 ms from silence.
        let ms = RATE as usize / 1000;
        let mut seed = 3u32;
        let x: Vec<f32> = (0..400 * ms)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                let ramp = ((i as f32 - 100.0 * ms as f32) / (60.0 * ms as f32)).clamp(0.0, 1.0);
                let mut v = 0.0;
                for k in 1..=20 {
                    v += (2.0 * std::f32::consts::PI * 150.0 * k as f32 * t).sin() * 0.3 / k as f32;
                }
                v * ramp + noise(&mut seed) * 1e-4
            })
            .collect();
        assert!(find_bursts(&envelope_db(&x, RATE)).is_empty());
    }
}
