// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The dashboard's spectrogram JSON (`spec.json`): 80-bin log-mel at
//! 16 kHz, 1024 / 256, clamped to a fixed dB range and rounded to a
//! tenth, so every panel the UI draws — a checkpoint's clean / degraded /
//! out, a sample's clean / chip output / model output — shares one axis
//! and one colour scale. One writer for all of them lives here; the
//! trainer's eval, the `infer` verb and the web server's sample specs
//! all go through [`Spec`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::spectral::log_mel;

/// Sample rate every spectrogram is computed at, Hz.
pub const SPEC_RATE: u32 = 16_000;
/// FFT size.
pub const SPEC_N_FFT: usize = 1024;
/// Hop, samples (16 ms).
pub const SPEC_HOP: usize = 256;
/// Mel bins.
pub const SPEC_MELS: usize = 80;
/// Floor of the shared dB range.
pub const DB_MIN: f32 = -100.0;
/// Ceiling of the shared dB range.
pub const DB_MAX: f32 = 0.0;

/// `spec.json`: the axes, then one `[frames][n_mels]` matrix per named
/// signal (`clean`, `degraded`, `out` for a checkpoint clip; whatever
/// the caller adds for a sample).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    /// Sample rate the matrices were computed at, Hz.
    pub rate: u32,
    /// FFT size.
    pub n_fft: usize,
    /// Hop, samples.
    pub hop: usize,
    /// Mel bins per frame.
    pub n_mels: usize,
    /// Colour-scale floor, dB.
    pub db_min: f32,
    /// Colour-scale ceiling, dB.
    pub db_max: f32,
    /// Frames in every matrix (the longest, when they differ).
    pub frames: usize,
    /// Frames of real speech before an appended tail, when the caller
    /// knows (the trainer's eval clips carry a garbage tail).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_frames: Option<usize>,
    /// The matrices, by name.
    #[serde(flatten)]
    pub mats: BTreeMap<String, Vec<Vec<f32>>>,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            rate: SPEC_RATE,
            n_fft: SPEC_N_FFT,
            hop: SPEC_HOP,
            n_mels: SPEC_MELS,
            db_min: DB_MIN,
            db_max: DB_MAX,
            frames: 0,
            speech_frames: None,
            mats: BTreeMap::new(),
        }
    }
}

impl Spec {
    /// An empty spec on the shared axes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the spectrogram of `x16` (16 kHz samples) as `name`.
    #[must_use]
    pub fn with(mut self, name: &str, x16: &[f32]) -> Self {
        self.add(name, x16);
        self
    }

    /// Add the spectrogram of `x16` (16 kHz samples) as `name`.
    pub fn add(&mut self, name: &str, x16: &[f32]) {
        let rows = spec_rows(x16);
        self.frames = self.frames.max(rows.len());
        self.mats.insert(name.to_owned(), rows);
    }

    /// Record how many frames are speech (the rest a tail), from a
    /// 16 kHz sample count.
    #[must_use]
    pub fn with_speech_samples(mut self, n16: usize) -> Self {
        self.speech_frames = Some(n16 / SPEC_HOP);
        self
    }
}

/// `[frames][n_mels]` log-mel of `x16` on the shared axes, each value
/// clamped to `[DB_MIN, DB_MAX]` and rounded to 0.1 dB (a tenth is below
/// what the colour map resolves and keeps the JSON small).
#[must_use]
pub fn spec_rows(x16: &[f32]) -> Vec<Vec<f32>> {
    log_mel(x16, SPEC_RATE, SPEC_N_FFT, SPEC_HOP, SPEC_MELS)
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|v| (v.clamp(DB_MIN, DB_MAX) * 10.0).round() / 10.0)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::sine;

    #[test]
    fn spec_has_the_shared_axes_and_round_trips() {
        let x = sine(1_000.0, 16_000, 16_000, 0.5);
        let spec = Spec::new().with("clean", &x).with("out", &x[..8_000]);
        assert_eq!(spec.frames, 1 + 16_000 / SPEC_HOP);
        assert_eq!(spec.mats["out"].len(), 1 + 8_000 / SPEC_HOP);
        assert_eq!(spec.mats["clean"][0].len(), SPEC_MELS);
        for v in spec.mats.values().flatten().flatten() {
            assert!((DB_MIN..=DB_MAX).contains(v));
            assert!(((v * 10.0).round() - v * 10.0).abs() < 1e-3, "{v}");
        }
        let json = serde_json::to_string(&spec).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["n_mels"], 80);
        assert_eq!(v["hop"], 256);
        assert_eq!(v["rate"], 16_000);
        assert_eq!(v["db_min"], -100.0);
        assert!(v.get("speech_frames").is_none());
        assert_eq!(v["clean"].as_array().unwrap().len(), spec.frames);
        let back: Spec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
        let tagged = Spec::new().with_speech_samples(2_560);
        assert_eq!(tagged.speech_frames, Some(10));
    }
}
