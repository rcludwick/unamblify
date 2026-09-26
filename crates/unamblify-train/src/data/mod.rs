// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Training data (spec §4): two loaders with identical batch output —
//! [`pipeline::PipelineLoader`] (manifest join + WAV reads + lag + crops on
//! the fly) and [`shards::ShardLoader`] (mmap of pre-packed examples) —
//! plus [`synthetic::SyntheticLoader`] for smoke tests. The crop, onset
//! and gain-jitter logic lives here; the garbage tail is the core crate's
//! `unamblify::tail`, shared with the data crate's shard builder so a
//! shard set and the on-the-fly loader draw tails from one distribution
//! (`tests/shard_roundtrip.rs` pins the two).
//!
//! A batch is `clean16 [B,1,N16]`, `deg8 [B,1,N16/2]`, `mask [B,1,N16]`
//! (1 where the target is real speech, 0 in the garbage tail), `onset [B]`
//! bool, `tail [B]` bool, `speaker [B]` i64 (`unamblify::speaker_id`),
//! `mode [B]` i64 — the example's vocoder mode as an index into the run's
//! `[data] modes` (0 in a single-mode run), what `[model] mode_embed`
//! conditions on.
//!
//! Receive-side noise (`[augment]`, `unamblify_audio::rx`) is an
//! [`RxStage`] every loader runs over `deg8` after the crop — never over
//! the target — seeded from the loader's seed and the example's index,
//! so a shard set and the pipeline draw the same noise for the same seed
//! and index.

pub mod pipeline;
pub mod shards;
pub mod synthetic;

use tch::{Device, Kind, Tensor};
use unamblify::tail::fill_tail;
pub use unamblify::tail::{garbage_tail, rms};
use unamblify::{FRAME_SAMPLES, VocoderMode};
pub use unamblify_audio::rx::RxCfg;

use crate::rng::Rng;

/// Receive-side noise on `deg8`, applied by every loader after the crop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RxStage {
    cfg: RxCfg,
    seed: u64,
}

impl RxStage {
    /// Salt that separates the rx stream from the loader's own draws.
    const SALT: u64 = 0x5258_4E4F_4953_4500;

    /// A stage for `cfg`, drawing from `seed`; `None` when nothing is on.
    #[must_use]
    pub fn new(cfg: RxCfg, seed: u64) -> Option<Self> {
        cfg.enabled().then_some(Self {
            cfg,
            seed: seed ^ Self::SALT,
        })
    }

    /// The generator for example `index`.
    #[must_use]
    pub fn rng_for(&self, index: u64) -> Rng {
        Rng::new(self.seed).fork(index)
    }

    /// Draw and apply to `ex.deg8` for example `index`; the target is
    /// untouched. Returns whether anything was added.
    pub fn apply(&self, ex: &mut Example, index: u64) -> bool {
        let mut rng = self.rng_for(index);
        unamblify_audio::rx::apply_drawn(&mut ex.deg8, &self.cfg, &mut rng).is_some()
    }

    /// The config.
    #[must_use]
    pub const fn cfg(&self) -> &RxCfg {
        &self.cfg
    }
}

/// One example in host memory, exactly what a shard stores.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// 16 kHz clean target.
    pub clean16: Vec<f32>,
    /// 8 kHz degraded input, lag-aligned to `clean16`.
    pub deg8: Vec<f32>,
    /// Sample index into `clean16` where the garbage tail begins
    /// (`clean16.len()` when there is none).
    pub mask_boundary: usize,
    /// The crop starts at the utterance's first sample (a key-up).
    pub onset: bool,
    /// The crop ends in a garbage tail with a silence target.
    pub tail: bool,
    /// `unamblify::speaker_id` of the speaker.
    pub speaker: u32,
    /// The vocoder mode as an index into the run's `[data] modes` (a
    /// shard set's `modes`), 0 in a single-mode run.
    pub mode: u8,
    /// Per feature frame (`deg8.len() / HOP` of them), 1 where the audio
    /// is the decoder's concealment of a lost channel frame. Empty means
    /// none was lost: a base capture, or a set built without the mask.
    pub erasure: Vec<u8>,
}

/// A batch on the training device.
#[derive(Debug)]
pub struct Batch {
    /// `[B, 1, N16]`.
    pub clean16: Tensor,
    /// `[B, 1, N16 / 2]`.
    pub deg8: Tensor,
    /// `[B, 1, N16]` float 0/1.
    pub mask: Tensor,
    /// `[B]` bool.
    pub onset: Tensor,
    /// `[B]` bool.
    pub tail: Tensor,
    /// `[B]` int64.
    pub speaker: Tensor,
    /// `[B]` int64: mode index per example.
    pub mode: Tensor,
    /// `[B, T]` float 0/1 at the feature-frame rate (`T = N8 / HOP`): the
    /// erasure mask `[model] erasure_in` reads. All zeros when nothing in
    /// the batch lost a frame.
    pub erasure: Tensor,
}

impl Batch {
    /// Collate host examples (all the same length) onto `device`.
    pub fn from_examples(examples: &[Example], device: Device) -> anyhow::Result<Self> {
        anyhow::ensure!(!examples.is_empty(), "empty batch");
        let n16 = examples[0].clean16.len();
        let n8 = examples[0].deg8.len();
        anyhow::ensure!(n16 == 2 * n8, "clean16 ({n16}) must be twice deg8 ({n8})");
        let b = i64::try_from(examples.len())?;
        let (n16_i, n8_i) = (i64::try_from(n16)?, i64::try_from(n8)?);
        let mut clean = Vec::with_capacity(examples.len() * n16);
        let mut deg = Vec::with_capacity(examples.len() * n8);
        let mut mask = Vec::with_capacity(examples.len() * n16);
        let mut onset = Vec::with_capacity(examples.len());
        let mut tail = Vec::with_capacity(examples.len());
        let mut speaker = Vec::with_capacity(examples.len());
        let mut mode = Vec::with_capacity(examples.len());
        let t = n8 / usize::try_from(crate::model::HOP).unwrap_or(80);
        let mut erasure = Vec::with_capacity(examples.len() * t);
        for ex in examples {
            anyhow::ensure!(
                ex.clean16.len() == n16 && ex.deg8.len() == n8,
                "ragged batch: {} / {} vs {n16} / {n8}",
                ex.clean16.len(),
                ex.deg8.len()
            );
            clean.extend_from_slice(&ex.clean16);
            deg.extend_from_slice(&ex.deg8);
            let boundary = ex.mask_boundary.min(n16);
            mask.extend(std::iter::repeat_n(1.0f32, boundary));
            mask.extend(std::iter::repeat_n(0.0f32, n16 - boundary));
            onset.push(ex.onset);
            tail.push(ex.tail);
            speaker.push(i64::from(ex.speaker));
            mode.push(i64::from(ex.mode));
            anyhow::ensure!(
                ex.erasure.is_empty() || ex.erasure.len() == t,
                "erasure mask is {} frames, the batch has {t}",
                ex.erasure.len()
            );
            if ex.erasure.is_empty() {
                erasure.extend(std::iter::repeat_n(0.0f32, t));
            } else {
                erasure.extend(ex.erasure.iter().map(|&v| f32::from(v.min(1))));
            }
        }
        Ok(Self {
            clean16: Tensor::from_slice(&clean)
                .view([b, 1, n16_i])
                .to_device(device),
            deg8: Tensor::from_slice(&deg)
                .view([b, 1, n8_i])
                .to_device(device),
            mask: Tensor::from_slice(&mask)
                .view([b, 1, n16_i])
                .to_device(device),
            onset: Tensor::from_slice(&onset).to_device(device),
            tail: Tensor::from_slice(&tail).to_device(device),
            speaker: Tensor::from_slice(&speaker).to_device(device),
            mode: Tensor::from_slice(&mode).to_device(device),
            erasure: Tensor::from_slice(&erasure)
                .view([b, i64::try_from(t)?])
                .to_device(device),
        })
    }

    /// `B`.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(self.speaker.size1().unwrap_or(0)).unwrap_or(0)
    }

    /// Whether the batch is empty (never, after `from_examples`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Samples per example at 16 kHz.
    #[must_use]
    pub fn n16(&self) -> i64 {
        self.clean16.size3().map_or(0, |s| s.2)
    }
}

/// Unused-kind guard so `Kind` stays imported for the tensor builders.
#[allow(dead_code)]
const KIND: Kind = Kind::Float;

/// Something that yields batches forever (epochs wrap and reshuffle).
pub trait Loader {
    /// The next `batch` examples on `device`.
    fn next_batch(&mut self, batch: usize, device: Device) -> anyhow::Result<Batch>;
    /// Number of source items (utterances or packed examples).
    fn len(&self) -> usize;
    /// Whether there is nothing to draw from.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// One line for the log.
    fn describe(&self) -> String;
}

/// How crops are drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropCfg {
    /// The vocoder mode: crops are whole frames of its length
    /// (`frame_samples()`), so a shard set and this loader draw the same
    /// windows.
    pub mode: VocoderMode,
    /// Crop length, seconds (whole frames of `mode`).
    pub crop_s: f32,
    /// Fraction of crops that start at the utterance start.
    pub onset_share: f32,
    /// Fraction of crops that end in a garbage tail.
    pub tail_share: f32,
    /// Gain jitter half-range, dB (0 = off; the shard path stores
    /// un-jittered examples, so parity tests use 0).
    pub gain_jitter_db: f32,
}

impl Default for CropCfg {
    fn default() -> Self {
        Self {
            mode: VocoderMode::Dstar,
            crop_s: 2.0,
            onset_share: 0.34,
            tail_share: 0.15,
            gain_jitter_db: 6.0,
        }
    }
}

impl CropCfg {
    /// Crop length in 8 kHz samples (whole frames of the mode).
    #[must_use]
    pub fn n8(&self) -> usize {
        // Whole-frame crops; truncation intended.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let millis = (self.crop_s.max(0.0) * 1000.0).round() as usize;
        (millis / self.mode.frame_ms() as usize) * self.mode.frame_samples()
    }

    /// Samples per channel frame of the mode.
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.mode.frame_samples()
    }
}

/// A whole utterance, lag-aligned, before cropping.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// 16 kHz clean.
    pub clean16: Vec<f32>,
    /// 8 kHz degraded, already shifted by the capture lag.
    pub deg8: Vec<f32>,
    /// `unamblify::speaker_id`.
    pub speaker: u32,
    /// Mode index of the capture (see [`Example::mode`]).
    pub mode: u8,
}

/// Shift the decoded signal back into alignment with its input: a lag of
/// `lag` samples (positive = decoded output is late) drops the first
/// `lag` samples; a negative lag prepends zeros.
#[must_use]
pub fn apply_lag(deg8: &[f32], lag: i32) -> Vec<f32> {
    if lag >= 0 {
        let l = usize::try_from(lag).unwrap_or(0).min(deg8.len());
        let mut v = deg8[l..].to_vec();
        v.extend(std::iter::repeat_n(0.0, l));
        v
    } else {
        let l = usize::try_from(-lag).unwrap_or(0);
        let mut v = vec![0.0; l];
        v.extend_from_slice(deg8);
        v.truncate(deg8.len());
        v
    }
}

/// Minimum garbage tail on a tail example, 8 kHz samples (200 ms).
pub const MIN_TAIL8: usize = 10 * FRAME_SAMPLES;

/// Draw one example from `utt` under `cfg`. Deterministic given `rng`.
/// `onset` is set whenever the crop starts at sample 0 — drawn as an
/// onset, or forced there because the utterance is no longer than a
/// crop — which is also the shard builder's rule (`FLAG_ONSET` iff the
/// crop's first frame is 0), so the onset weighting means the same thing
/// from both sources.
#[must_use]
pub fn crop_example(utt: &Utterance, cfg: &CropCfg, rng: &mut Rng) -> Example {
    let n8 = cfg.n8();
    let fs = cfg.frame_samples();
    let n16 = 2 * n8;
    let len8 = utt.deg8.len().min(utt.clean16.len() / 2);
    let drew_onset = rng.chance(cfg.onset_share);
    let tail = rng.chance(cfg.tail_share);
    let start8 = if drew_onset || len8 <= n8 {
        0
    } else {
        // Frame-aligned start.
        let slack = (len8 - n8) / fs;
        rng.below(slack + 1) * fs
    };
    let onset = start8 == 0;
    let avail8 = len8.saturating_sub(start8).min(n8);
    let mut deg8: Vec<f32> = utt.deg8[start8..start8 + avail8].to_vec();
    let mut clean16: Vec<f32> = utt.clean16[2 * start8..2 * (start8 + avail8)].to_vec();
    deg8.resize(n8, 0.0);
    clean16.resize(n16, 0.0);

    let mut mask_boundary = n16;
    if tail {
        // Speech ends at least MIN_TAIL8 before the crop ends; if the
        // utterance already ended, cut at a random frame in [n8/2, n8 − MIN_TAIL8].
        let latest = n8.saturating_sub(MIN_TAIL8);
        let speech_end8 = if avail8 <= latest {
            avail8
        } else {
            let lo = (n8 / 2) / fs;
            let hi = latest / fs;
            (lo + rng.below(hi.saturating_sub(lo) + 1)) * fs
        };
        mask_boundary = fill_tail(&mut deg8, &mut clean16, speech_end8, rng);
    }

    if cfg.gain_jitter_db > 0.0 {
        let db = (rng.next_f32() * 2.0 - 1.0) * cfg.gain_jitter_db;
        let g = 10f32.powf(db / 20.0);
        for v in &mut deg8 {
            *v *= g;
        }
        for v in &mut clean16 {
            *v *= g;
        }
    }

    Example {
        clean16,
        deg8,
        mask_boundary,
        onset,
        tail,
        speaker: utt.speaker,
        mode: utt.mode,
        // The pipeline source crops live from the base captures, which
        // lost nothing; the erasure mask comes with a shard set.
        erasure: Vec::new(),
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    fn utt(len8: usize, seed: u64) -> Utterance {
        let mut r = Rng::new(seed);
        let deg8: Vec<f32> = (0..len8).map(|_| r.next_f32() - 0.5).collect();
        let clean16: Vec<f32> = (0..2 * len8).map(|_| r.next_f32() - 0.5).collect();
        Utterance {
            clean16,
            deg8,
            speaker: 7,
            mode: 2,
        }
    }

    #[test]
    fn lag_shifts_and_pads() {
        assert_eq!(
            apply_lag(&[1.0, 2.0, 3.0, 4.0], 1),
            vec![2.0, 3.0, 4.0, 0.0]
        );
        assert_eq!(
            apply_lag(&[1.0, 2.0, 3.0, 4.0], -1),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(apply_lag(&[1.0, 2.0], 5), vec![0.0, 0.0]);
        assert_eq!(apply_lag(&[1.0, 2.0], 0), vec![1.0, 2.0]);
    }

    #[test]
    fn crops_have_the_right_shape_and_flags() {
        let cfg = CropCfg {
            mode: VocoderMode::Dstar,
            crop_s: 0.5,
            onset_share: 1.0,
            tail_share: 1.0,
            gain_jitter_db: 0.0,
        };
        assert_eq!(cfg.n8(), 4000);
        // Codec 2 1600 crops in 40 ms frames: 0.5 s rounds down to 12.
        assert_eq!(
            CropCfg {
                mode: VocoderMode::Codec2_1600,
                ..cfg
            }
            .n8(),
            12 * 320
        );
        assert_eq!(
            CropCfg {
                mode: VocoderMode::Codec2_3200,
                ..cfg
            }
            .n8(),
            4000
        );
        let u = utt(12_000, 1);
        let mut rng = Rng::new(3);
        let ex = crop_example(&u, &cfg, &mut rng);
        assert_eq!(ex.deg8.len(), 4000);
        assert_eq!(ex.clean16.len(), 8000);
        assert!(ex.onset && ex.tail);
        assert_eq!(&ex.deg8[..100], &u.deg8[..100], "onset starts at 0");
        assert!(ex.mask_boundary <= 8000 - 2 * MIN_TAIL8);
        assert!(ex.mask_boundary >= 4000);
        assert!(ex.clean16[ex.mask_boundary..].iter().all(|&v| v == 0.0));
        assert_eq!(
            &ex.clean16[..ex.mask_boundary],
            &u.clean16[..ex.mask_boundary]
        );

        // Short utterance: zero-padded, tail begins at the utterance end,
        // and the crop is an onset because it starts at sample 0 whether
        // or not one was drawn.
        let short = utt(2_000, 2);
        let ex = crop_example(&short, &cfg, &mut rng);
        assert_eq!(ex.mask_boundary, 4000);
        assert!(
            ex.deg8[2000..].iter().any(|&v| v != 0.0),
            "garbage, not silence"
        );
        let ex = crop_example(
            &short,
            &CropCfg {
                onset_share: 0.0,
                ..cfg
            },
            &mut rng,
        );
        assert!(ex.onset, "a forced start at 0 is an onset");

        // No onset / tail: random frame-aligned start, full mask.
        let cfg2 = CropCfg {
            onset_share: 0.0,
            tail_share: 0.0,
            ..cfg
        };
        let ex = crop_example(&u, &cfg2, &mut rng);
        assert!(!ex.onset && !ex.tail);
        assert_eq!(ex.mask_boundary, 8000);
        let start = u
            .deg8
            .windows(4000)
            .position(|w| w == ex.deg8.as_slice())
            .unwrap();
        assert_eq!(start % FRAME_SAMPLES, 0);
        assert_eq!(&ex.clean16[..], &u.clean16[2 * start..2 * start + 8000]);
    }

    #[test]
    fn same_rng_same_example_and_jitter_scales_both() {
        let cfg = CropCfg {
            crop_s: 0.5,
            ..CropCfg::default()
        };
        let u = utt(9_000, 5);
        let a = crop_example(&u, &cfg, &mut Rng::new(9));
        let b = crop_example(&u, &cfg, &mut Rng::new(9));
        assert_eq!(a, b);
        let no_jitter = CropCfg {
            gain_jitter_db: 0.0,
            ..cfg
        };
        let c = crop_example(&u, &no_jitter, &mut Rng::new(9));
        assert_eq!(c.onset, a.onset);
        assert_eq!(c.mask_boundary, a.mask_boundary);
        let i = (0..c.deg8.len()).find(|&i| c.deg8[i].abs() > 1e-3).unwrap();
        let g = a.deg8[i] / c.deg8[i];
        assert!((0.5..2.0).contains(&g), "{g}");
        let j = (0..c.mask_boundary)
            .find(|&i| c.clean16[i].abs() > 1e-3)
            .unwrap();
        assert!((a.clean16[j] / c.clean16[j] - g).abs() < 1e-4);
    }

    #[test]
    fn codec2_1600_crops_start_on_320_sample_frames() {
        let cfg = CropCfg {
            mode: VocoderMode::Codec2_1600,
            crop_s: 0.4,
            onset_share: 0.0,
            tail_share: 0.0,
            gain_jitter_db: 0.0,
        };
        let u = utt(12_000, 4);
        let mut rng = Rng::new(11);
        let mut starts = std::collections::BTreeSet::new();
        for _ in 0..40 {
            let ex = crop_example(&u, &cfg, &mut rng);
            assert_eq!(ex.deg8.len(), 3_200);
            let start = u
                .deg8
                .windows(3_200)
                .position(|w| w == ex.deg8.as_slice())
                .unwrap();
            assert_eq!(start % 320, 0, "{start}");
            starts.insert(start);
        }
        assert!(starts.len() > 1);
    }

    #[test]
    fn rx_stage_is_seeded_per_example_and_never_touches_the_target() {
        let cfg = RxCfg {
            share: 1.0,
            hum: true,
            broadband: true,
            whine: true,
            colouring: true,
            squelch: true,
        };
        let stage = RxStage::new(cfg, 7).expect("enabled");
        assert!(RxStage::new(RxCfg { share: 0.0, ..cfg }, 7).is_none());
        let u = utt(6_000, 3);
        let crop = CropCfg {
            crop_s: 0.5,
            gain_jitter_db: 0.0,
            ..CropCfg::default()
        };
        let base = crop_example(&u, &crop, &mut Rng::new(1));
        let mut a = base.clone();
        let mut b = base.clone();
        let mut c = base.clone();
        assert!(stage.apply(&mut a, 5));
        assert!(stage.apply(&mut b, 5));
        assert!(stage.apply(&mut c, 6));
        assert_eq!(a, b, "same seed and index, same noise");
        assert_ne!(a.deg8, c.deg8, "another index, other noise");
        assert_ne!(a.deg8, base.deg8);
        assert_eq!(a.clean16, base.clean16, "the target is never touched");
        assert_eq!(a.mask_boundary, base.mask_boundary);
        let other = RxStage::new(cfg, 8).unwrap();
        let mut d = base.clone();
        other.apply(&mut d, 5);
        assert_ne!(a.deg8, d.deg8, "another seed, other noise");
    }

    #[test]
    fn batch_collates_mask_and_flags() {
        let cfg = CropCfg {
            mode: VocoderMode::Dstar,
            crop_s: 0.2,
            onset_share: 0.0,
            tail_share: 1.0,
            gain_jitter_db: 0.0,
        };
        let u = utt(4_000, 8);
        let mut rng = Rng::new(2);
        let exs: Vec<Example> = (0..3).map(|_| crop_example(&u, &cfg, &mut rng)).collect();
        let b = Batch::from_examples(&exs, Device::Cpu).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b.clean16.size(), [3, 1, 3200]);
        assert_eq!(b.deg8.size(), [3, 1, 1600]);
        assert_eq!(b.mask.size(), [3, 1, 3200]);
        assert_eq!(b.onset.kind(), Kind::Bool);
        assert_eq!(b.speaker.kind(), Kind::Int64);
        assert_eq!(b.speaker.int64_value(&[0]), 7);
        assert_eq!(b.mode.kind(), Kind::Int64);
        assert_eq!(b.mode.size(), [3]);
        assert_eq!(b.mode.int64_value(&[2]), 2, "the mode index rides along");
        let m0 = b.mask.get(0).get(0);
        let ones = m0.sum(Kind::Float).int64_value(&[]);
        assert_eq!(ones, i64::try_from(exs[0].mask_boundary).unwrap());
        assert_eq!(b.tail.int64_value(&[1]), 1);
    }
}
