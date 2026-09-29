// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Audio plumbing shared by the data stages, the trainer's eval and the
//! dashboard: WAV / FLAC readers to f32 mono, a WAV writer, an anti-aliased
//! resampler, loudness normalisation, silence trimming, STFT / mel, and the
//! metrics of `docs/design/training.md` (LSD, mel-L1, SI-SDR, segmental
//! SNR), plus the augmentations of `docs/design/data-pipeline.md` stage 3
//! (noise mixing and the overdriven-mic chain in [`chain`], receive-side
//! noise in [`rx`]). Pure Rust, no tch; everything here is a plain
//! function on slices.
//!
//! Conventions: samples are `f32` in `[-1, 1]`, mono; rates are Hz; levels
//! are dBFS where 0 dBFS is a full-scale sine's RMS of `1/√2`... no — to
//! match the prepare rules of the spec, **dBFS here means `20·log10(rms)`
//! of the sample values**, so a full-scale square wave is 0 dBFS and a
//! full-scale sine is −3 dBFS.

// DSP code converts between sample counts and floats constantly; those
// casts are intended and the values involved are far below 2^24.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod chain;
pub mod io;
pub mod level;
pub mod metrics;
pub mod plosive;
pub mod resample;
pub mod rx;
pub mod sibilant;
pub mod spec;
pub mod spectral;

pub use io::{
    Audio, WavInfo, read, read_flac, read_mp3, read_wav, read_wav_range, wav_info, write_flac_s16,
    write_wav_s16,
};
pub use level::{Trimmed, active_rms_dbfs, normalize_rms, rms_dbfs, trim_silence};
pub use metrics::{
    HnrExcess, cpp_excess, hnr_db, hnr_excess, hnr_excess_stats, lsd, mel_l1, seg_snr, si_sdr,
    xcorr_lag,
};
pub use plosive::{PlosiveStats, plosive_excess};
pub use resample::resample;
pub use sibilant::{SibilantStats, sibilant_excess};
pub use spec::{Spec, spec_rows};
pub use spectral::{log_mel, mel_filterbank, stft};

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// File system.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// WAV read or write.
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    /// FLAC demux or decode.
    #[error("flac: {0}")]
    Flac(#[from] symphonia::core::errors::Error),
    /// Resampler construction.
    #[error("resampler: {0}")]
    ResamplerConstruction(#[from] rubato::ResamplerConstructionError),
    /// Resampler processing.
    #[error("resample: {0}")]
    Resample(#[from] rubato::ResampleError),
    /// FLAC encode (verify or write).
    #[error("flac encode: {0}")]
    FlacEncode(String),
    /// A file whose format this crate does not read.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// `Result` with this crate's error.
pub type Result<T> = std::result::Result<T, AudioError>;

/// `20·log10(x)`, floored at −200 dB so zero never becomes −inf.
#[must_use]
pub fn to_db(x: f32) -> f32 {
    20.0 * x.max(1e-10).log10()
}

/// Inverse of [`to_db`].
#[must_use]
pub fn from_db(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[cfg(test)]
pub(crate) mod testutil {
    /// A sine of `freq` Hz at `rate` for `n` samples, peak `amp`.
    pub fn sine(freq: f32, rate: u32, n: usize, amp: f32) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / rate as f32;
        (0..n).map(|i| amp * (w * i as f32).sin()).collect()
    }

    /// Deterministic white-ish noise in `[-amp, amp]` (xorshift).
    pub fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f32 / (1u64 << 53) as f32;
                amp * (2.0 * u - 1.0)
            })
            .collect()
    }
}
