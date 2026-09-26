// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Synthetic utterances for smoke tests and CI: a pitch-gliding harmonic
//! series under a syllable-rate envelope plus breath noise stands in for
//! speech; the "vocoder" is decimation to 8 kHz followed by coarse
//! quantisation, a 3.4 kHz one-pole lowpass and 20 ms gain steps, so the
//! degradation is deterministic, harmonic-preserving and clearly audible.
//! Nothing here is used for a real run.

use tch::Device;
use unamblify_audio::resample;

use super::{Batch, CropCfg, Loader, RxCfg, RxStage, Utterance, crop_example};
use crate::rng::Rng;

/// Build one synthetic utterance of `dur_s` seconds from `seed`.
pub fn utterance(seed: u64, dur_s: f32) -> anyhow::Result<Utterance> {
    let mut r = Rng::new(seed ^ 0xA5A5_5A5A);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n16 = ((dur_s * 16_000.0).round() as usize).div_ceil(320) * 320;
    let f0_base = 90.0 + r.next_f32() * 130.0;
    let syl_rate = 3.0 + r.next_f32() * 3.0;
    let mut phase = 0.0f32;
    let mut clean16 = Vec::with_capacity(n16);
    for i in 0..n16 {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / 16_000.0;
        let f0 = f0_base * (1.0 + 0.15 * (2.0 * std::f32::consts::PI * 0.7 * t).sin());
        phase += 2.0 * std::f32::consts::PI * f0 / 16_000.0;
        let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * syl_rate * t).sin()).powi(2);
        let mut v = 0.0f32;
        for h in 1..=12u32 {
            #[allow(clippy::cast_precision_loss)]
            let hf = h as f32;
            v += (phase * hf).sin() / (hf * hf).sqrt();
        }
        let breath = r.normal() * 0.02;
        clean16.push(0.12 * env * v + breath * env);
    }
    let clean8 = resample(&clean16, 16_000, 8_000)?;
    let mut y = 0.0f32;
    let mut gain = 1.0f32;
    let deg8: Vec<f32> = clean8
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if i % 160 == 0 {
                gain = 0.8 + 0.4 * r.next_f32();
            }
            let q = (s * 31.0).round() / 31.0;
            y = 0.35 * y + 0.65 * q * gain;
            y
        })
        .collect();
    Ok(Utterance {
        clean16,
        deg8,
        speaker: u32::try_from(seed % 4).unwrap_or(0),
        mode: 0,
    })
}

/// Draws crops from a fixed pool of synthetic utterances.
#[derive(Debug)]
pub struct SyntheticLoader {
    utts: Vec<Utterance>,
    cfg: CropCfg,
    rng: Rng,
    rx: Option<RxStage>,
    drawn: u64,
}

impl SyntheticLoader {
    /// `n_utts` utterances of `dur_s` seconds, seeded.
    pub fn new(n_utts: usize, dur_s: f32, cfg: CropCfg, seed: u64) -> anyhow::Result<Self> {
        let utts = (0..n_utts)
            .map(|i| utterance(seed.wrapping_mul(1000).wrapping_add(i as u64), dur_s))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            utts,
            cfg,
            rng: Rng::new(seed),
            rx: None,
            drawn: 0,
        })
    }

    /// Add receive-side noise on `deg8`.
    #[must_use]
    pub fn with_rx(mut self, cfg: RxCfg, seed: u64) -> Self {
        self.rx = RxStage::new(cfg, seed);
        self
    }
}

impl Loader for SyntheticLoader {
    fn next_batch(&mut self, batch: usize, device: Device) -> anyhow::Result<Batch> {
        let examples: Vec<_> = (0..batch)
            .map(|_| {
                let u = &self.utts[self.rng.below(self.utts.len())];
                let mut ex = crop_example(u, &self.cfg, &mut self.rng);
                if let Some(rx) = &self.rx {
                    rx.apply(&mut ex, self.drawn);
                }
                self.drawn += 1;
                ex
            })
            .collect();
        Batch::from_examples(&examples, device)
    }

    fn len(&self) -> usize {
        self.utts.len()
    }

    fn describe(&self) -> String {
        format!(
            "synthetic: {} utterances, crop {} s",
            self.utts.len(),
            self.cfg.crop_s
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_utterances_are_aligned_and_degraded() {
        let u = utterance(3, 1.0).unwrap();
        assert_eq!(u.clean16.len(), 16_000);
        assert_eq!(u.deg8.len(), 8_000);
        let clean8 = resample(&u.clean16, 16_000, 8_000).unwrap();
        // Aligned: the degraded signal correlates best at lag 0.
        assert_eq!(unamblify_audio::xcorr_lag(&clean8, &u.deg8, 50), 0);
        // Degraded: not identical, but the same order of loudness.
        let s = unamblify_audio::si_sdr(&u.deg8, &clean8);
        assert!(s > 3.0 && s < 25.0, "{s}");
        let mut l = SyntheticLoader::new(
            2,
            1.0,
            CropCfg {
                crop_s: 0.5,
                ..CropCfg::default()
            },
            1,
        )
        .unwrap();
        let b = l.next_batch(3, Device::Cpu).unwrap();
        assert_eq!(b.deg8.size(), [3, 1, 4000]);
        assert_eq!(l.len(), 2);
    }
}
