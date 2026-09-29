// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `unamblify augment` (`docs/design/data-pipeline.md`, stage 3, the
//! decode-only stage): the pure parts. The stage itself is the capture
//! harness run with [`crate::capture::Stage::Augment`] — same lock,
//! canary, control, status, writer and pipelined decode — over the stored
//! channel frames of `captured/<mode>/` instead of the prepared 8 kHz
//! audio, writing `captured/<mode>+<kind>/`. Here: the spec, the seeded
//! share decision, the mute codeword of a mode (a constant for the AMBE
//! modes, digital silence encoded once with the crate for Codec 2), and
//! the mutation of one utterance's frames.

use unamblify::aug::{AugKind, CaptureAug, Fec, Subst, key_seed};
use unamblify::channel::{
    YsfVd, ambe_mute_frame, apply_ber, apply_drops, apply_ysf_ber, plan_drops,
};
use unamblify::{Rng, VocoderMode};

use crate::vocoder::open_software;
use crate::{DataError, Result};

/// What `--kind`, `--rate`, `--burst`, `--subst`, `--share` and `--seed`
/// asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentSpec {
    /// Drops or bit errors.
    pub kind: AugKind,
    /// Fraction of frames lost (`drops`) or per-bit flip probability
    /// (`ber`).
    pub rate: f32,
    /// Burst length range, frames (`drops`).
    pub burst: (u32, u32),
    /// Lost-frame substitution (`drops`).
    pub subst: Subst,
    /// The FEC to model around the voice bits (`ber`).
    pub fec: Fec,
    /// Share of the base capture's utterances to process.
    pub share: f32,
    /// Seed of the per-key share decision and mutation stream.
    pub seed: u64,
}

impl AugmentSpec {
    /// The design's defaults for a kind: drops at 2 % in bursts of 1–3
    /// muted; bit errors at 1e-3.
    #[must_use]
    pub fn new(kind: AugKind) -> Self {
        Self {
            kind,
            rate: match kind {
                AugKind::Drops => 0.02,
                AugKind::Ber => 1e-3,
                // Not a channel impairment: the sibling is the same
                // utterances re-encoded, so there is no rate to set.
                AugKind::Perens => 0.0,
            },
            burst: (1, 3),
            subst: Subst::Mute,
            fec: Fec::None,
            share: 0.3,
            seed: 1,
        }
    }

    /// Refuse what cannot run: a rate outside `[0, 1]`, an inverted
    /// burst range, and `erase` on an AMBE mode (where it would be the
    /// chip's own concealment, which this harness cannot ask for; the
    /// software modes conceal in the parameter domain instead).
    pub fn validate(&self, mode: VocoderMode) -> Result<()> {
        if !(0.0..=1.0).contains(&self.rate) || !self.rate.is_finite() {
            return Err(DataError::Invalid(format!(
                "--rate {} must be within 0..=1",
                self.rate
            )));
        }
        if !(0.0..=1.0).contains(&self.share) || !self.share.is_finite() {
            return Err(DataError::Invalid(format!(
                "--share {} must be within 0..=1",
                self.share
            )));
        }
        if self.burst.0 == 0 || self.burst.1 < self.burst.0 {
            return Err(DataError::Invalid(format!(
                "--burst {}..{} must be 1 <= lo <= hi",
                self.burst.0, self.burst.1
            )));
        }
        if self.kind == AugKind::Drops && self.subst == Subst::Erase && !mode.is_software() {
            return Err(DataError::Invalid(
                "--subst erase (mark the frame bad and let the chip conceal) is chip-only for \
                 the AMBE modes and not implemented; use mute or repeat"
                    .to_owned(),
            ));
        }
        if self.fec != Fec::None && self.kind != AugKind::Ber {
            return Err(DataError::Invalid(format!(
                "--fec {} applies to --kind ber",
                self.fec
            )));
        }
        Ok(())
    }

    /// Whether `key` is in the seeded share. Stable across runs and
    /// independent of visit order.
    #[must_use]
    pub fn selects(&self, key: &str) -> bool {
        Rng::new(key_seed(key, self.seed))
            .fork(0x5A)
            .chance(self.share)
    }

    /// The per-utterance mutation stream.
    #[must_use]
    pub fn stream(&self, key: &str) -> Rng {
        Rng::new(key_seed(key, self.seed)).fork(0xA5)
    }
}

/// Parse `--burst lo..hi` (or a single `n`).
pub fn parse_burst(s: &str) -> std::result::Result<(u32, u32), String> {
    let (lo, hi) = match s.split_once("..") {
        Some((a, b)) => (a, b),
        None => (s, s),
    };
    let lo: u32 = lo.trim().parse().map_err(|e| format!("burst {s:?}: {e}"))?;
    let hi: u32 = hi.trim().parse().map_err(|e| format!("burst {s:?}: {e}"))?;
    if lo == 0 || hi < lo {
        return Err(format!("burst {s:?}: need 1 <= lo <= hi"));
    }
    Ok((lo, hi))
}

/// The mute codeword of `mode`: the AMBE modes' constants, or — since an
/// all-zero Codec 2 frame is not silence — `frame_samples()` samples of
/// digital silence encoded once with the crate.
pub fn mute_frame(mode: VocoderMode) -> Result<Vec<u8>> {
    if let Some(m) = ambe_mute_frame(mode) {
        return Ok(m.to_vec());
    }
    let mut voc = open_software(mode)?;
    let silence = vec![0i16; mode.frame_samples()];
    let frame = voc.encode(&silence)?;
    if frame.len() != mode.frame_bytes() {
        return Err(DataError::Invalid(format!(
            "{mode}: encoding one silent frame gave {} bytes, not {}",
            frame.len(),
            mode.frame_bytes()
        )));
    }
    Ok(frame)
}

/// Mutate one utterance's frames per the spec, seeded by its key: the
/// mutated bytes (same length) and the row's `aug` record.
pub fn mutate(
    mode: VocoderMode,
    spec: &AugmentSpec,
    key: &str,
    ambe: &[u8],
    mute: &[u8],
) -> Result<(Vec<u8>, CaptureAug)> {
    let fb = mode.frame_bytes();
    if !ambe.len().is_multiple_of(fb) {
        return Err(DataError::Invalid(format!(
            "{key}: {} bytes is not a whole number of {fb}-byte {mode} frames",
            ambe.len()
        )));
    }
    let frames = ambe.len() / fb;
    let mut out = ambe.to_vec();
    let mut rng = spec.stream(key);
    let positions = match spec.kind {
        AugKind::Drops => {
            let positions = plan_drops(&mut rng, frames, spec.rate, spec.burst);
            apply_drops(&mut out, mode, &positions, spec.subst, mute)
                .map_err(DataError::Invalid)?;
            positions
        }
        // A re-encode, not a mutation of existing frames: there is
        // nothing here to corrupt. `Stage::Recode` produces this set by
        // encoding the prepared audio again with the other codec, so
        // reaching the decode-only path means a caller built the wrong
        // stage.
        AugKind::Perens => {
            return Err(DataError::Invalid(format!(
                "{key}: {} is a re-encode, not a channel mutation; it is produced by the \
                 recode stage, not by augment",
                AugKind::Perens
            )));
        }
        AugKind::Ber => match spec.fec {
            // Flipping the voice bits themselves, which no receiver sees.
            Fec::None => apply_ber(&mut out, fb, mode.channel_bits(), spec.rate, &mut rng)
                .map_err(DataError::Invalid)?,
            Fec::YsfVd1 => apply_ysf_ber(&mut out, mode, YsfVd::Vd1, spec.rate, &mut rng)
                .map_err(DataError::Invalid)?,
            Fec::YsfVd2 => apply_ysf_ber(&mut out, mode, YsfVd::Vd2, spec.rate, &mut rng)
                .map_err(DataError::Invalid)?,
        },
    };
    Ok((
        out,
        CaptureAug {
            kind: spec.kind,
            rate_ppm: CaptureAug::ppm(spec.rate),
            burst: (spec.kind == AugKind::Drops).then_some(spec.burst),
            subst: (spec.kind == AugKind::Drops).then_some(spec.subst),
            fec: (spec.kind == AugKind::Ber).then_some(spec.fec),
            seed: spec.seed,
            positions,
        },
    ))
}

#[cfg(test)]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use unamblify::channel::NULL_AMBE_FRAME;

    /// The decode-only path must refuse a re-encode kind outright: it
    /// mutates existing channel frames, and there is nothing to mutate.
    #[test]
    fn the_recode_kind_is_refused_by_the_mutation_path() {
        let spec = AugmentSpec::new(AugKind::Perens);
        let frames = vec![0u8; VocoderMode::Dstar.frame_bytes() * 4];
        let err = mutate(VocoderMode::Dstar, &spec, "k", &frames, &[0u8; 9])
            .expect_err("a re-encode must not go through the mutation path");
        assert!(
            err.to_string().contains("recode"),
            "the error should name the right stage: {err}"
        );
    }

    #[test]
    fn spec_validates_and_selects_a_stable_share() {
        let mut s = AugmentSpec::new(AugKind::Drops);
        assert!((s.rate - 0.02).abs() < 1e-9);
        s.validate(VocoderMode::Dstar).unwrap();
        s.subst = Subst::Erase;
        let err = s.validate(VocoderMode::Dstar).unwrap_err().to_string();
        assert!(err.contains("chip-only"), "{err}");
        s.subst = Subst::Mute;
        s.burst = (3, 1);
        assert!(s.validate(VocoderMode::Dstar).is_err());
        s.burst = (1, 3);
        s.rate = 1.5;
        assert!(s.validate(VocoderMode::Dstar).is_err());
        let b = AugmentSpec::new(AugKind::Ber);
        assert!((b.rate - 1e-3).abs() < 1e-12);
        assert_eq!(parse_burst("1..3").unwrap(), (1, 3));
        assert_eq!(parse_burst("2").unwrap(), (2, 2));
        assert!(parse_burst("0..2").is_err());
        assert!(parse_burst("3..1").is_err());
        assert!(parse_burst("x").is_err());
        let s = AugmentSpec {
            share: 0.3,
            ..AugmentSpec::new(AugKind::Drops)
        };
        let n = (0..5_000).filter(|i| s.selects(&format!("c/u{i}"))).count();
        assert!((1_350..1_650).contains(&n), "{n}");
        assert_eq!(s.selects("c/u7"), s.selects("c/u7"));
        let other = AugmentSpec {
            seed: 2,
            ..s.clone()
        };
        let picked = |spec: &AugmentSpec| -> Vec<usize> {
            (0..256)
                .filter(|i| spec.selects(&format!("c/u{i}")))
                .collect()
        };
        assert_ne!(picked(&s), picked(&other), "another seed, another share");
    }

    #[test]
    fn mute_frames_per_family() {
        assert_eq!(mute_frame(VocoderMode::Dstar).unwrap(), NULL_AMBE_FRAME);
        assert_eq!(mute_frame(VocoderMode::YsfDmr).unwrap().len(), 7);
        #[cfg(feature = "codec2")]
        for mode in [VocoderMode::Codec2_3200, VocoderMode::Codec2_1600] {
            let m = mute_frame(mode).unwrap();
            assert_eq!(m.len(), 8);
            assert_ne!(m, [0u8; 8], "{mode}: all-zero is not silence");
            assert_eq!(mute_frame(mode).unwrap(), m, "stable");
            // It decodes to (near) silence.
            let mut voc = open_software(mode).unwrap();
            let pcm = voc.decode(&m).unwrap();
            let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap();
            assert!(peak < 200, "{mode}: mute frame decodes with peak {peak}");
        }
    }

    #[test]
    fn mutation_is_seeded_by_key_and_keeps_the_frame_count() {
        let mode = VocoderMode::Dstar;
        let ambe: Vec<u8> = (0..9 * 400).map(|i| (i % 251) as u8).collect();
        let spec = AugmentSpec {
            share: 1.0,
            ..AugmentSpec::new(AugKind::Drops)
        };
        let (a, rec) = mutate(mode, &spec, "vctk/p225_001_mic2", &ambe, &NULL_AMBE_FRAME).unwrap();
        assert_eq!(a.len(), ambe.len());
        assert_eq!(rec.kind, AugKind::Drops);
        assert_eq!(rec.burst, Some((1, 3)));
        assert_eq!(rec.subst, Some(Subst::Mute));
        assert!(!rec.positions.is_empty());
        for f in 0..400u32 {
            let frame = &a[f as usize * 9..(f as usize + 1) * 9];
            if rec.positions.contains(&f) {
                assert_eq!(frame, &NULL_AMBE_FRAME[..]);
            } else {
                assert_eq!(frame, &ambe[f as usize * 9..(f as usize + 1) * 9]);
            }
        }
        let (b, rec2) = mutate(mode, &spec, "vctk/p225_001_mic2", &ambe, &NULL_AMBE_FRAME).unwrap();
        assert_eq!((a.clone(), rec.clone()), (b, rec2));
        let (c, _) = mutate(mode, &spec, "vctk/p225_002_mic2", &ambe, &NULL_AMBE_FRAME).unwrap();
        assert_ne!(a, c, "another key, another pattern");
        let ber = AugmentSpec {
            rate: 0.01,
            ..AugmentSpec::new(AugKind::Ber)
        };
        let (d, rec) = mutate(mode, &ber, "k/x", &ambe, &NULL_AMBE_FRAME).unwrap();
        assert_eq!(rec.burst, None);
        assert_eq!(rec.subst, None);
        assert_eq!(rec.rate_ppm, 10_000);
        let flips: u32 = d.iter().zip(&ambe).map(|(x, y)| (x ^ y).count_ones()).sum();
        assert!(
            (220..=360).contains(&flips),
            "{flips} flips over 28800 bits"
        );
        assert!(mutate(mode, &spec, "k", &ambe[..10], &NULL_AMBE_FRAME).is_err());
    }

    /// End to end against the real encoder: this is what pins the frame
    /// bit map. A wrong offset or a missed Gray step would put nonsense in
    /// the concealed frame and the decoded gap would not sit at the level
    /// of the speech either side of it.
    #[cfg(feature = "codec2")]
    #[test]
    fn codec2_erasure_fill_puts_speech_where_mute_puts_silence() {
        use unamblify::channel::fill_erasures;

        let mode = VocoderMode::Codec2_3200;
        let (fs, fb, n) = (mode.frame_samples(), mode.frame_bytes(), 30);
        // A steady voiced tone: stable pitch and energy, so the frames
        // either side of the gap are well-defined anchors.
        let pcm: Vec<i16> = (0..n * fs)
            .map(|i| {
                let t = i as f32 / 8_000.0;
                (8_000.0 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()) as i16
            })
            .collect();
        let mut enc = open_software(mode).unwrap();
        let mut bytes = Vec::new();
        for f in 0..n {
            bytes.extend_from_slice(&enc.encode(&pcm[f * fs..(f + 1) * fs]).unwrap());
        }
        let gap = [12u32, 13, 14];
        let mute = mute_frame(mode).unwrap();

        let mut filled = bytes.clone();
        fill_erasures(&mut filled, mode, &gap, &mute).unwrap();
        let mut muted = bytes.clone();
        apply_drops(&mut muted, mode, &gap, Subst::Mute, &mute).unwrap();
        assert_ne!(filled, muted, "concealment is not a mute");
        assert_eq!(filled.len(), bytes.len(), "the frame count never changes");
        // Only the gap is rewritten.
        for f in 0..n {
            if !gap.contains(&(f as u32)) {
                assert_eq!(
                    &filled[f * fb..(f + 1) * fb],
                    &bytes[f * fb..(f + 1) * fb],
                    "frame {f} outside the gap"
                );
            }
        }

        let decode_all = |b: &[u8]| {
            let mut d = open_software(mode).unwrap();
            let mut out = Vec::new();
            for f in 0..n {
                out.extend_from_slice(&d.decode(&b[f * fb..(f + 1) * fb]).unwrap());
            }
            out
        };
        let rms = |x: &[i16]| {
            (x.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / x.len() as f64).sqrt()
        };
        let span = (12 * fs)..(15 * fs);
        let orig = rms(&decode_all(&bytes)[span.clone()]);
        let fill = rms(&decode_all(&filled)[span.clone()]);
        let mute_rms = rms(&decode_all(&muted)[span]);
        assert!(mute_rms < orig * 0.2, "mute {mute_rms} vs orig {orig}");
        assert!(
            fill > orig * 0.5 && fill < orig * 2.0,
            "fill {fill} vs orig {orig}"
        );
    }
}
