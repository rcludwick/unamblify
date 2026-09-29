// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Loudness and silence: the prepare rules of the spec (trim below
//! −45 dBFS keeping 200 ms, normalise active-speech RMS to −26 dBFS with
//! the gain clamped to ±20 dB).

use crate::{from_db, to_db};

/// Analysis window for level decisions, ms.
const WINDOW_MS: u32 = 20;
/// Frames quieter than this do not count as speech for [`active_rms_dbfs`].
pub const ACTIVE_THRESHOLD_DBFS: f32 = -50.0;
/// Default trim threshold.
pub const TRIM_THRESHOLD_DBFS: f32 = -45.0;
/// Default margin kept on both sides of the speech.
pub const TRIM_MARGIN_S: f32 = 0.2;
/// Default loudness target.
pub const TARGET_DBFS: f32 = -26.0;
/// Default gain clamp.
pub const MAX_GAIN_DB: f32 = 20.0;

fn window(rate: u32) -> usize {
    ((rate * WINDOW_MS) / 1_000).max(1) as usize
}

/// RMS of every sample, dBFS (`20·log10(rms)`).
#[must_use]
pub fn rms_dbfs(x: &[f32]) -> f32 {
    if x.is_empty() {
        return -200.0;
    }
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    to_db(ms.sqrt())
}

/// RMS over the 20 ms frames whose own RMS exceeds
/// [`ACTIVE_THRESHOLD_DBFS`] — the "active speech" level. Returns `None`
/// when no frame is active.
#[must_use]
pub fn active_rms_dbfs(x: &[f32], rate: u32) -> Option<f32> {
    let w = window(rate);
    let thr = from_db(ACTIVE_THRESHOLD_DBFS);
    let thr2 = thr * thr;
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for frame in x.chunks(w) {
        let e: f32 = frame.iter().map(|v| v * v).sum();
        if e / frame.len() as f32 > thr2 {
            sum += f64::from(e);
            n += frame.len();
        }
    }
    (n > 0).then(|| to_db((sum / n as f64).sqrt() as f32))
}

/// Scale `x` so its active-speech RMS is `target_dbfs`, with the gain
/// clamped to `±max_gain_db`. Returns the scaled signal and the gain that
/// was applied, dB. Silence (no active frame) is returned unchanged with
/// gain 0.
#[must_use]
pub fn normalize_rms(x: &[f32], rate: u32, target_dbfs: f32, max_gain_db: f32) -> (Vec<f32>, f32) {
    let Some(level) = active_rms_dbfs(x, rate) else {
        return (x.to_vec(), 0.0);
    };
    let gain_db = (target_dbfs - level).clamp(-max_gain_db.abs(), max_gain_db.abs());
    let g = from_db(gain_db);
    (x.iter().map(|v| v * g).collect(), gain_db)
}

/// Result of [`trim_silence`].
#[derive(Debug, Clone, PartialEq)]
pub struct Trimmed {
    /// The kept samples.
    pub samples: Vec<f32>,
    /// Samples removed from the front.
    pub lead: usize,
    /// Samples removed from the back.
    pub tail: usize,
}

impl Trimmed {
    /// Leading trim, seconds.
    #[must_use]
    pub fn lead_s(&self, rate: u32) -> f64 {
        self.lead as f64 / f64::from(rate)
    }

    /// Trailing trim, seconds.
    #[must_use]
    pub fn tail_s(&self, rate: u32) -> f64 {
        self.tail as f64 / f64::from(rate)
    }
}

/// Remove leading and trailing stretches whose 20 ms frames are all below
/// `threshold_dbfs`, keeping `margin_s` of the quiet material on each side
/// where it exists. A signal with no frame above the threshold is returned
/// empty (lead = its whole length).
#[must_use]
pub fn trim_silence(x: &[f32], rate: u32, threshold_dbfs: f32, margin_s: f32) -> Trimmed {
    let w = window(rate);
    let thr = from_db(threshold_dbfs);
    let thr2 = thr * thr;
    let loud = |frame: &[f32]| frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32 > thr2;
    let frames: Vec<bool> = x.chunks(w).map(loud).collect();
    let Some(first) = frames.iter().position(|&l| l) else {
        return Trimmed {
            samples: Vec::new(),
            lead: x.len(),
            tail: 0,
        };
    };
    let last = frames.iter().rposition(|&l| l).unwrap_or(first);
    let margin = (margin_s.max(0.0) * rate as f32).round() as usize;
    let start = (first * w).saturating_sub(margin);
    let end = ((last + 1) * w + margin).min(x.len());
    Trimmed {
        samples: x[start..end].to_vec(),
        lead: start,
        tail: x.len() - end,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::testutil::{noise, sine};

    #[test]
    fn rms_of_a_full_scale_sine_is_minus_3_db() {
        let x = sine(440.0, 16_000, 16_000, 1.0);
        assert!((rms_dbfs(&x) + 3.01).abs() < 0.05, "{}", rms_dbfs(&x));
        assert_eq!(rms_dbfs(&[]), -200.0);
    }

    #[test]
    fn normalise_hits_the_target_within_a_tenth_of_a_db() {
        // Speech-like: 1 s of tone at -21 dBFS surrounded by near-silence,
        // which must not drag the active level down.
        let mut x = vec![0.0f32; 8_000];
        x.extend(sine(
            300.0,
            16_000,
            16_000,
            from_db(-21.0) * std::f32::consts::SQRT_2,
        ));
        x.extend(noise(8_000, 1e-4, 7));
        let (y, gain) = normalize_rms(&x, 16_000, TARGET_DBFS, MAX_GAIN_DB);
        assert!((gain + 5.0).abs() < 0.1, "gain {gain}");
        let level = active_rms_dbfs(&y, 16_000).unwrap();
        assert!((level - TARGET_DBFS).abs() < 0.1, "level {level}");
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn gain_is_clamped_and_silence_is_left_alone() {
        // RMS −49 dBFS: active (above −50), needs +23 dB, gets +20.
        let quiet = sine(
            300.0,
            16_000,
            16_000,
            from_db(-49.0) * std::f32::consts::SQRT_2,
        );
        let (_, gain) = normalize_rms(&quiet, 16_000, TARGET_DBFS, MAX_GAIN_DB);
        assert!((gain - 20.0).abs() < 1e-6, "{gain}");
        // Too hot: −6 dBFS wants −20 dB and is clamped there too.
        let hot = sine(
            300.0,
            16_000,
            16_000,
            from_db(-3.0) * std::f32::consts::SQRT_2,
        );
        let (_, gain) = normalize_rms(&hot, 16_000, TARGET_DBFS, MAX_GAIN_DB);
        assert!((gain + 20.0).abs() < 1e-6, "{gain}");
        let (y, gain) = normalize_rms(&[0.0; 1600], 16_000, TARGET_DBFS, MAX_GAIN_DB);
        assert_eq!(gain, 0.0);
        assert_eq!(y, vec![0.0; 1600]);
        assert_eq!(active_rms_dbfs(&[0.0; 1600], 16_000), None);
    }

    #[test]
    fn trim_keeps_the_margin() {
        let rate = 16_000;
        let mut x = vec![0.0f32; 16_000]; // 1.0 s silence
        x.extend(sine(300.0, rate, 8_000, 0.3)); // 0.5 s speech
        x.extend(vec![0.0f32; 4_800]); // 0.3 s silence
        let t = trim_silence(&x, rate, TRIM_THRESHOLD_DBFS, TRIM_MARGIN_S);
        // Speech starts at 16000; keep 0.2 s (3200) before it.
        assert_eq!(t.lead, 16_000 - 3_200);
        // Speech ends at 24000; keep 3200 after → end 27200; tail = 28800-27200.
        assert_eq!(t.tail, 1_600);
        assert_eq!(t.samples.len(), x.len() - t.lead - t.tail);
        assert!((t.lead_s(rate) - 0.8).abs() < 1e-9);
        assert!((t.tail_s(rate) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn trim_handles_no_silence_and_all_silence() {
        let x = sine(300.0, 16_000, 3_200, 0.3);
        let t = trim_silence(&x, 16_000, TRIM_THRESHOLD_DBFS, TRIM_MARGIN_S);
        assert_eq!((t.lead, t.tail), (0, 0));
        assert_eq!(t.samples, x);
        let t = trim_silence(&[0.0; 3_200], 16_000, TRIM_THRESHOLD_DBFS, TRIM_MARGIN_S);
        assert!(t.samples.is_empty());
        assert_eq!(t.lead, 3_200);
    }
}
