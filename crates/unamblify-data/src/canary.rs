// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The canary clip and its checks (spec §3): a fixed reference clip
//! encoded on day one, its channel frames' sha256 stored in
//! `captured/<mode>/canary.json` together with the firmware strings and
//! the measured decode lag; every `--canary-every` utterances and on
//! every re-init the clip is encoded again and must match byte for byte.
//!
//! The AMBE+2 encoder carries state across frames (pitch tracking, the
//! warm-up rubbish after init), so a byte-for-byte comparison is only
//! meaningful when every encode of the clip starts from the same state:
//! reset → the [`WARM_UP_FRAMES`]-frame preamble → the clip. The record
//! stores the preamble length it was made with, and a harness whose
//! preamble differs refuses the record instead of failing the run. The
//! software codec follows the same discipline through the [`Vocoder`]
//! trait; there the check catches a `codec2` crate upgrade whose encoder
//! output changed.
//!
//! The clip is synthesised once into `canary/1khz-and-speech.8k.wav`
//! under the data root — 0.5 s of 1 kHz tone, then 1.5 s of a harmonic,
//! syllabically-modulated "voice" — and read back from there ever after,
//! so its bytes never depend on the synthesis code again.

use std::path::Path;

use unamblify::{CanaryRecord, VOCODER_SAMPLE_RATE, VocoderMode};
use unamblify_audio::{read_wav, write_wav_s16, xcorr_lag};

use crate::chip::{ChipInfo, WARM_UP_FRAMES};
use crate::util::{hex, now_rfc3339, read_json, sha256_hex, write_json_atomic};
use crate::vocoder::{Vocoder, pad_frames};
use crate::{DataError, DataRoot, Result};

/// Length of the clip, seconds.
pub const CLIP_SECONDS: f32 = 2.0;

/// Lag search range for the cross-correlation, samples at 8 kHz.
pub const LAG_SEARCH: usize = 400;

/// The synthesised clip: 0.5 s of 1 kHz at −20 dBFS, then 1.5 s of a
/// harmonic voice-like signal (fundamental gliding 120 → 180 Hz, eight
/// harmonics with a 1/k roll-off, a 4 Hz syllabic envelope).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn synth_clip() -> Vec<f32> {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let n = (CLIP_SECONDS * VOCODER_SAMPLE_RATE as f32) as usize;
    let rate = VOCODER_SAMPLE_RATE as f32;
    let tone_n = n / 4;
    let mut out = Vec::with_capacity(n);
    let tone_amp = 0.1 * std::f32::consts::SQRT_2; // −20 dBFS rms
    for i in 0..tone_n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / rate;
        out.push(tone_amp * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin());
    }
    let voice_n = n - tone_n;
    let mut phase = 0.0f32;
    for i in 0..voice_n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / rate;
        #[allow(clippy::cast_precision_loss)]
        let f0 = 120.0 + 60.0 * (i as f32 / voice_n as f32);
        phase += 2.0 * std::f32::consts::PI * f0 / rate;
        let env = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * 4.0 * t).cos());
        let mut s = 0.0f32;
        for k in 1..=8u32 {
            #[allow(clippy::cast_precision_loss)]
            let kf = k as f32;
            s += (kf * phase).sin() / kf;
        }
        out.push(0.12 * env * s);
    }
    out
}

/// Read the canary clip, synthesising and writing it first if absent.
pub fn ensure_clip(root: &DataRoot) -> Result<Vec<f32>> {
    let path = root.canary_clip();
    if path.exists() {
        let a = read_wav(&path)?;
        if a.rate != VOCODER_SAMPLE_RATE {
            return Err(DataError::Invalid(format!(
                "{}: canary clip is {} Hz, expected {}",
                path.display(),
                a.rate,
                VOCODER_SAMPLE_RATE
            )));
        }
        return Ok(a.samples);
    }
    let clip = synth_clip();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    write_wav_s16(&path, &clip, VOCODER_SAMPLE_RATE)?;
    // Read back so the in-memory clip is the s16-quantised one on disk.
    Ok(read_wav(&path)?.samples)
}

/// One encode of the canary: the frames and their digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanaryEncode {
    /// Concatenated channel frames.
    pub frames: Vec<u8>,
    /// SHA-256 of `frames`.
    pub sha256: String,
    /// Hex of the first 16 bytes.
    pub first_16: String,
}

/// Encode the clip through `voc`.
pub fn encode_clip(voc: &mut dyn Vocoder, clip: &[f32]) -> Result<CanaryEncode> {
    let frames = voc.encode(&pad_frames(clip, voc.mode()))?;
    let sha256 = sha256_hex(&frames);
    let first_16 = hex(&frames[..frames.len().min(16)]);
    Ok(CanaryEncode {
        frames,
        sha256,
        first_16,
    })
}

/// Day one: encode, decode, measure the lag, and build the record.
pub fn record(voc: &mut dyn Vocoder, pcm8k: &[f32]) -> Result<(CanaryRecord, CanaryEncode)> {
    let mode = voc.mode();
    let enc = encode_clip(voc, pcm8k)?;
    let pcm = voc.decode(&enc.frames)?;
    let decoded: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32_767.0).collect();
    let n = pcm8k.len().min(decoded.len());
    let lag_samples = xcorr_lag(&pcm8k[..n], &decoded[..n], LAG_SEARCH);
    let ChipInfo { prodid, version } = voc.info();
    let rec = CanaryRecord {
        mode,
        clip: DataRoot::CANARY_CLIP.to_owned(),
        frames_sha256: enc.sha256.clone(),
        frames_first_16: enc.first_16.clone(),
        lag_samples,
        prodid,
        version,
        recorded_at: now_rfc3339(),
        warm_up_frames: WARM_UP_FRAMES,
    };
    Ok((rec, enc))
}

/// Load `canary.json` if present.
pub fn load(path: &Path) -> Result<Option<CanaryRecord>> {
    if !path.exists() {
        return Ok(None);
    }
    read_json(path).map(Some)
}

/// Write `canary.json` — only ever when absent; a mismatch never
/// overwrites it.
pub fn store(path: &Path, rec: &CanaryRecord) -> Result<()> {
    if path.exists() {
        return Err(DataError::Invalid(format!(
            "{}: refusing to overwrite an existing canary.json",
            path.display()
        )));
    }
    write_json_atomic(path, rec)
}

/// Compare a fresh encode to the record. The encode must have been made
/// from the same state as the record (reset + preamble); a record made
/// with a different preamble is refused outright.
pub fn compare(rec: &CanaryRecord, enc: &CanaryEncode, port: &str) -> Result<()> {
    if rec.warm_up_frames != WARM_UP_FRAMES {
        return Err(DataError::Invalid(format!(
            "canary.json was recorded after a {}-frame warm-up; this harness warms up with {} frames, \
             so its re-encodes cannot be compared. Keep the harness version that made the record \
             or start a new capture directory.",
            rec.warm_up_frames, WARM_UP_FRAMES
        )));
    }
    if rec.frames_sha256 == enc.sha256 {
        Ok(())
    } else {
        Err(DataError::CanaryMismatch {
            port: port.to_owned(),
            mode: rec.mode,
            expected: rec.frames_sha256.clone(),
            got: enc.sha256.clone(),
        })
    }
}

/// Frames the clip occupies in `mode`.
#[must_use]
pub fn clip_frames(clip: &[f32], mode: VocoderMode) -> usize {
    clip.len().div_ceil(mode.frame_samples())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Faults, SimTransport};
    use crate::vocoder::ChipVocoder;

    #[test]
    fn clip_is_two_seconds_and_persisted_once() {
        let clip = synth_clip();
        assert_eq!(clip.len(), 16_000);
        assert_eq!(clip_frames(&clip, VocoderMode::Dstar), 100);
        assert_eq!(clip_frames(&clip, VocoderMode::Codec2_1600), 50);
        let peak = clip.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 1.0 && peak > 0.1, "{peak}");
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let a = ensure_clip(&root).unwrap();
        assert!(root.canary_clip().exists());
        let b = ensure_clip(&root).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 16_000);
    }

    #[test]
    fn record_measures_the_sim_lag_and_compare_catches_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let clip = ensure_clip(&root).unwrap();
        let mut dev = ChipVocoder::init(SimTransport::new(), VocoderMode::Dstar).unwrap();
        let (rec, enc) = record(&mut dev, &clip).unwrap();
        assert_eq!(rec.lag_samples, 42);
        assert_eq!(rec.prodid, "AMBE3000F");
        assert_eq!(rec.frames_first_16.len(), 32);
        assert_eq!(enc.frames.len(), 900);
        let path = root.canary_json(VocoderMode::Dstar);
        store(&path, &rec).unwrap();
        assert!(store(&path, &rec).is_err(), "must not overwrite");
        assert_eq!(load(&path).unwrap().as_ref(), Some(&rec));

        let again = encode_clip(&mut dev, &clip).unwrap();
        compare(&rec, &again, "/dev/sim").unwrap();

        let sim = SimTransport::new().with_faults(Faults {
            corrupt_frames_from: Some(0),
            ..Faults::default()
        });
        let mut bad = ChipVocoder::init(sim, VocoderMode::Dstar).unwrap();
        let corrupt = encode_clip(&mut bad, &clip).unwrap();
        let err = compare(&rec, &corrupt, "/dev/sim").unwrap_err();
        assert!(matches!(err, DataError::CanaryMismatch { .. }));
        assert!(err.to_string().contains("CANARY MISMATCH"));
        assert_eq!(load(&path).unwrap().as_ref(), Some(&rec));
        assert_eq!(rec.warm_up_frames, WARM_UP_FRAMES);
        let other = CanaryRecord {
            warm_up_frames: WARM_UP_FRAMES + 1,
            ..rec.clone()
        };
        let err = compare(&other, &again, "/dev/sim").unwrap_err();
        assert!(err.to_string().contains("warm-up"), "{err}");
    }
}
