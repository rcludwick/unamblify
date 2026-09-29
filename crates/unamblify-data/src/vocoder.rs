// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! One vocoder over a whole utterance, whatever runs it: the AMBE-3000
//! chip through the pipelined runner ([`ChipVocoder`]) or Codec 2 in
//! software ([`Codec2Vocoder`], behind the `codec2` feature). The capture
//! harness only ever sees a [`Vocoder`]: encode the padded 8 kHz PCM to
//! channel frames, decode them back, report what the codec is, and
//! `reset` + `warm_up` before a canary encode.
//!
//! The chip carries encoder state across frames, so its canary is only
//! comparable from a known state (reset → warm-up → clip). The software
//! codec starts every `encode` and `decode` call from a fresh codec
//! instance, so an utterance's bytes are a pure function of its input and
//! the crate version; `reset` is then a no-op and the warm-up just costs
//! a few frames. A `codec2` crate upgrade that changes the encoder's
//! output therefore shows up as a canary mismatch, exactly like a chip
//! whose state drifted.
//!
//! Only the *encoder* is bit-reproducible. The crate's decoder, like the C
//! reference, draws the phases of unvoiced harmonics from a process-global
//! generator (`codec2_rand`, a static LCG shared by every thread), so two
//! decodes of the same frames differ in their noise-like components. That
//! is what a real M17 receiver does too, and the training target is the
//! clean signal, so it is accepted: `.ambe` files are reproducible,
//! `.wav` files are one draw. `verify` checks a `.wav` against the hash
//! its own capture recorded, never against a fresh decode.

use std::sync::Arc;

use ambe_thumbdv::Transport;
use unamblify::aug::AugKind;
use unamblify::{VocoderFamily, VocoderMode};

use crate::chip::{Chip, ChipInfo, WARM_UP_FRAMES};
use crate::pipeline::{decode, encode, round_trip};
use crate::{DataError, Result};

/// The `codec2` crate version this crate is pinned to (`Cargo.toml`
/// carries `=` this version; a test checks the two agree). It is the
/// `version` string of every Codec 2 capture row and canary record.
pub const CODEC2_CRATE_VERSION: &str = "0.3.1";

/// The PRODID a software Codec 2 capture records.
pub const CODEC2_PRODID: &str = "codec2";

/// Upstream revision of the vendored D-STAR vocoder, recorded in every
/// row it captures. Pinned to `vendor/ham-digital-modes/VENDORED.md`; a
/// test checks the two agree, so a re-vendor cannot silently relabel the
/// captures an older revision produced.
pub const PERENS_REV: &str = "e403fcf39a00ab926984838fd23ab0ecc89d2462";

/// `prodid` of the vendored software D-STAR vocoder.
pub const PERENS_PRODID: &str = "ham_digital_modes";

/// A codec over a whole utterance.
pub trait Vocoder {
    /// The mode this instance is configured for.
    fn mode(&self) -> VocoderMode;

    /// What the codec says about itself (PRODID / VERSTRING for the chip,
    /// `codec2` / the crate version for software).
    fn info(&self) -> ChipInfo;

    /// Return to the post-init state: the chip's full reset + re-init;
    /// nothing for a software codec that starts fresh on every call.
    fn reset(&mut self) -> Result<()>;

    /// Encode `pcm8` (8 kHz s16, already padded to whole frames of
    /// `mode().frame_samples()`) to concatenated channel frames of
    /// `mode().frame_bytes()` each.
    fn encode(&mut self, pcm8: &[i16]) -> Result<Vec<u8>>;

    /// Decode concatenated channel frames back to 8 kHz s16 PCM,
    /// `mode().frame_samples()` per frame.
    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>>;

    /// One utterance both ways in one call: the channel frames and the
    /// samples decoded from them. The default is the two passes back to
    /// back, which is what a software codec wants (it is not waiting on a
    /// link); [`ChipVocoder`] overrides it with the interleaved runner,
    /// which is 1.73× faster because the UART is full duplex.
    ///
    /// `pcm8` must already be padded to whole frames, as for
    /// [`encode`](Vocoder::encode).
    fn round_trip(&mut self, pcm8: &[i16]) -> Result<(Vec<u8>, Vec<i16>)> {
        let frames = self.encode(pcm8)?;
        let pcm = self.decode(&frames)?;
        Ok((frames, pcm))
    }

    /// Whether [`round_trip`](Vocoder::round_trip) runs the two
    /// directions at the same time, so their times cannot be told apart.
    /// A capture row from such a vocoder carries `roundtrip_ms` and
    /// leaves `encode_ms` / `decode_ms` at 0.
    fn interleaves(&self) -> bool {
        false
    }

    /// Push [`WARM_UP_FRAMES`] frames of `pcm8k` through the encoder and
    /// discard them (the chip's first frames after init are rubbish).
    fn warm_up(&mut self, pcm8k: &[f32]) -> Result<()> {
        let n = WARM_UP_FRAMES * self.mode().frame_samples();
        let mut clip = pad_frames(&pcm8k[..n.min(pcm8k.len())], self.mode());
        clip.resize(n, 0);
        self.encode(&clip).map(drop)
    }
}

/// Whole frames needed for `samples` 8 kHz samples in `mode`
/// (zero-padded up).
#[must_use]
pub fn frames_for(samples: usize, mode: VocoderMode) -> usize {
    samples.div_ceil(mode.frame_samples())
}

/// `pcm8k` as s16, zero-padded to whole frames of `mode`.
#[must_use]
pub fn pad_frames(pcm8k: &[f32], mode: VocoderMode) -> Vec<i16> {
    let n = frames_for(pcm8k.len(), mode) * mode.frame_samples();
    let mut out = Vec::with_capacity(n);
    out.extend(pcm8k.iter().map(|&s| unamblify_audio::io::to_s16(s)));
    out.resize(n, 0);
    out
}

/// A vocoder opener: worker name (a port path, `sim:0`, `codec2:3`) →
/// an initialised vocoder for the mode. The seam that keeps the serial
/// port out of tests.
pub type Opener = Arc<dyn Fn(&str, VocoderMode) -> Result<Box<dyn Vocoder>> + Send + Sync>;

/// A transport opener: port path → transport.
pub type TransportOpener = Arc<dyn Fn(&str) -> Result<Box<dyn Transport>> + Send + Sync>;

/// `Box<dyn Transport>` as a `Transport`, so `Chip<T>` can hold one.
pub struct DynTransport(pub Box<dyn Transport>);

impl Transport for DynTransport {
    fn send(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.0.send(bytes)
    }

    fn recv_some(
        &mut self,
        buf: &mut [u8],
        timeout: std::time::Duration,
    ) -> std::io::Result<usize> {
        self.0.recv_some(buf, timeout)
    }
}

/// An [`Opener`] over a transport opener: open the port, run the chip's
/// init sequence for the mode. `sequential` is `capture --sequential`:
/// the escape hatch that goes back to the two separate passes.
#[must_use]
pub fn chip_opener(open: TransportOpener, sequential: bool) -> Opener {
    Arc::new(move |port: &str, mode: VocoderMode| {
        let transport = DynTransport(open(port)?);
        Ok(Box::new(ChipVocoder::init_with(transport, mode, sequential)?) as Box<dyn Vocoder>)
    })
}

/// An [`Opener`] for the software codecs: no port, one fresh
/// [`Codec2Vocoder`] per worker.
#[must_use]
pub fn software_opener() -> Opener {
    Arc::new(|_: &str, mode: VocoderMode| open_software(mode))
}

/// A software vocoder for `mode`, or an error naming why there is none
/// (a chip mode, or a build without the `codec2` feature).
pub fn open_software(mode: VocoderMode) -> Result<Box<dyn Vocoder>> {
    match mode.family() {
        // D-STAR is a chip mode, and stays one for `capture`: this
        // refusal is what stops an AMBE set being captured without the
        // ThumbDV. The vendored software vocoder is reached only through
        // [`open_recode`], which the recode stage calls deliberately, so
        // a plain capture can never silently produce software frames.
        VocoderFamily::Ambe => Err(DataError::Invalid(format!(
            "{mode} is a chip mode; it has no software vocoder"
        ))),
        VocoderFamily::Codec2 => {
            #[cfg(feature = "codec2")]
            {
                Ok(Box::new(Codec2Vocoder::new(mode)?))
            }
            #[cfg(not(feature = "codec2"))]
            {
                Err(DataError::Invalid(format!(
                    "{mode}: this binary was built without the `codec2` feature of unamblify-data"
                )))
            }
        }
    }
}

/// A second, independent implementation of `mode`'s codec, for the
/// recode stage: the sibling set it writes is the same utterances through
/// another encoder, not the chip's frames impaired.
///
/// Deliberately separate from [`open_software`], which refuses every AMBE
/// mode so a plain `capture` cannot produce software frames by accident.
pub fn open_recode(mode: VocoderMode, kind: AugKind) -> Result<Box<dyn Vocoder>> {
    match kind {
        AugKind::Perens => {
            #[cfg(feature = "perens")]
            {
                Ok(Box::new(PerensVocoder::new(mode)?))
            }
            #[cfg(not(feature = "perens"))]
            {
                Err(DataError::Invalid(format!(
                    "{mode}: this binary was built without the `perens` feature of unamblify-data"
                )))
            }
        }
        other => Err(DataError::Invalid(format!(
            "{other} is a channel impairment, not a second codec; it is produced by augment"
        ))),
    }
}

/// An [`Opener`] for the recode stage: no port, one fresh vocoder per
/// worker, exactly as the software modes do.
#[must_use]
pub fn recode_opener(kind: AugKind) -> Opener {
    Arc::new(move |_: &str, mode: VocoderMode| open_recode(mode, kind))
}

/// The AMBE-3000 as a [`Vocoder`]: the pipelined encode and decode passes
/// over an initialised [`Chip`], and — unless it was built `sequential` —
/// the interleaved runner for a whole utterance at once.
pub struct ChipVocoder<T: Transport> {
    chip: Chip<T>,
    sequential: bool,
}

impl<T: Transport> ChipVocoder<T> {
    /// Run the chip's init sequence on `transport` for `mode`,
    /// interleaving whole-utterance round trips.
    pub fn init(transport: T, mode: VocoderMode) -> Result<Self> {
        Self::init_with(transport, mode, false)
    }

    /// As [`init`](Self::init); `sequential` forces
    /// [`round_trip`](Vocoder::round_trip) back to the encode pass
    /// followed by the decode pass (`capture --sequential`).
    pub fn init_with(transport: T, mode: VocoderMode, sequential: bool) -> Result<Self> {
        Ok(Self {
            chip: Chip::init(transport, mode)?,
            sequential,
        })
    }

    /// Whether this vocoder was built with the two passes separate.
    #[must_use]
    pub fn is_sequential(&self) -> bool {
        self.sequential
    }

    /// One utterance with both directions in the chip at once
    /// ([`crate::pipeline::round_trip`]), whatever `sequential` says.
    pub fn round_trip_interleaved(&mut self, pcm8: &[i16]) -> Result<(Vec<u8>, Vec<i16>)> {
        let mode = self.chip.mode();
        Ok(round_trip(self.chip.transport_mut(), mode, pcm8)?)
    }

    /// The chip underneath.
    #[must_use]
    pub fn chip(&self) -> &Chip<T> {
        &self.chip
    }

    /// The chip underneath, mutably.
    pub fn chip_mut(&mut self) -> &mut Chip<T> {
        &mut self.chip
    }

    /// Give the transport back.
    pub fn into_transport(self) -> T {
        self.chip.into_transport()
    }
}

impl<T: Transport> Vocoder for ChipVocoder<T> {
    fn mode(&self) -> VocoderMode {
        self.chip.mode()
    }

    fn info(&self) -> ChipInfo {
        self.chip.info().clone()
    }

    /// Reset + the full init sequence, three attempts 200 ms apart.
    fn reset(&mut self) -> Result<()> {
        let mut last = None;
        for attempt in 1..=3 {
            match self.chip.reinit() {
                Ok(()) => return Ok(()),
                Err(e) => {
                    log::warn!("re-init attempt {attempt} failed: {e}");
                    last = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
        Err(last.map_or_else(
            || DataError::Invalid("re-init failed".to_owned()),
            DataError::from,
        ))
    }

    fn encode(&mut self, pcm8: &[i16]) -> Result<Vec<u8>> {
        let mode = self.chip.mode();
        Ok(encode(self.chip.transport_mut(), mode, pcm8)?)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        let mode = self.chip.mode();
        Ok(decode(self.chip.transport_mut(), mode, bytes)?)
    }

    fn round_trip(&mut self, pcm8: &[i16]) -> Result<(Vec<u8>, Vec<i16>)> {
        if self.sequential {
            let frames = self.encode(pcm8)?;
            let pcm = self.decode(&frames)?;
            return Ok((frames, pcm));
        }
        self.round_trip_interleaved(pcm8)
    }

    fn interleaves(&self) -> bool {
        !self.sequential
    }
}

/// Codec 2 in software, through the pure-Rust `codec2` crate. Every
/// `encode` / `decode` starts from a fresh codec instance (see the module
/// doc), so two workers, two runs or two machines produce the same bytes
/// for the same input and crate version.
#[cfg(feature = "codec2")]
pub struct Codec2Vocoder {
    mode: VocoderMode,
    c2mode: codec2::Codec2Mode,
}

#[cfg(feature = "codec2")]
impl Codec2Vocoder {
    /// A software vocoder for one of the Codec 2 modes.
    pub fn new(mode: VocoderMode) -> Result<Self> {
        let c2mode = match mode {
            VocoderMode::Codec2_3200 => codec2::Codec2Mode::MODE_3200,
            VocoderMode::Codec2_1600 => codec2::Codec2Mode::MODE_1600,
            other => {
                return Err(DataError::Invalid(format!("{other} is not a Codec 2 mode")));
            }
        };
        // The crate's own word for the mode must match ours.
        let probe = codec2::Codec2::new(c2mode);
        if probe.samples_per_frame() != mode.frame_samples()
            || probe.bits_per_frame() != usize::from(mode.channel_bits())
        {
            return Err(DataError::Invalid(format!(
                "codec2 {CODEC2_CRATE_VERSION} says {mode} is {} samples / {} bits per frame; \
                 this harness expects {} / {}",
                probe.samples_per_frame(),
                probe.bits_per_frame(),
                mode.frame_samples(),
                mode.channel_bits()
            )));
        }
        Ok(Self { mode, c2mode })
    }
}

#[cfg(feature = "codec2")]
impl Vocoder for Codec2Vocoder {
    fn mode(&self) -> VocoderMode {
        self.mode
    }

    fn info(&self) -> ChipInfo {
        ChipInfo {
            prodid: CODEC2_PRODID.to_owned(),
            version: CODEC2_CRATE_VERSION.to_owned(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    fn encode(&mut self, pcm8: &[i16]) -> Result<Vec<u8>> {
        let fs = self.mode.frame_samples();
        let fb = self.mode.frame_bytes();
        if !pcm8.len().is_multiple_of(fs) {
            return Err(DataError::Invalid(format!(
                "{}: {} samples is not a whole number of {fs}-sample frames",
                self.mode,
                pcm8.len()
            )));
        }
        let mut c2 = codec2::Codec2::new(self.c2mode);
        let mut out = Vec::with_capacity(pcm8.len() / fs * fb);
        let mut bits = vec![0u8; fb];
        for frame in pcm8.chunks_exact(fs) {
            bits.fill(0);
            c2.encode(&mut bits, frame);
            out.extend_from_slice(&bits);
        }
        Ok(out)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        let fs = self.mode.frame_samples();
        let fb = self.mode.frame_bytes();
        if !bytes.len().is_multiple_of(fb) {
            return Err(DataError::Invalid(format!(
                "{}: {} bytes is not a whole number of {fb}-byte frames",
                self.mode,
                bytes.len()
            )));
        }
        let mut c2 = codec2::Codec2::new(self.c2mode);
        let mut out = Vec::with_capacity(bytes.len() / fb * fs);
        let mut pcm = vec![0i16; fs];
        for frame in bytes.chunks_exact(fb) {
            c2.decode(&mut pcm, frame);
            out.extend_from_slice(&pcm);
        }
        Ok(out)
    }
}

/// D-STAR in software, through the vendored `ham_digital_modes` vocoder.
///
/// The AMBE-3000 is the real D-STAR encoder every radio carries; this is a
/// second, independent implementation of the same 72-bit frame, kept as a
/// sibling capture set (`captured/dstar+perens/`) so the model sees the
/// mode degraded two ways. It scored 2.18 MOS against the chip's 2.56 on
/// 1,298 dev utterances: close enough to be useful, different enough to be
/// worth training on.
///
/// Like [`Codec2Vocoder`], every call starts from a fresh encoder and
/// decoder, so two workers, two runs or two machines produce the same
/// bytes for the same input and vendored revision.
#[cfg(feature = "perens")]
pub struct PerensVocoder;

#[cfg(feature = "perens")]
impl PerensVocoder {
    /// A software D-STAR vocoder. Refuses every other mode: this codec
    /// implements D-STAR's own AMBE variant and nothing else (upstream's
    /// AMBE+2 is not vendored — its patents have not expired).
    pub fn new(mode: VocoderMode) -> Result<Self> {
        if mode != VocoderMode::Dstar {
            return Err(DataError::Invalid(format!(
                "{mode}: the vendored software vocoder implements D-STAR only"
            )));
        }
        Ok(Self)
    }
}

#[cfg(feature = "perens")]
impl Vocoder for PerensVocoder {
    fn mode(&self) -> VocoderMode {
        VocoderMode::Dstar
    }

    fn info(&self) -> ChipInfo {
        ChipInfo {
            prodid: PERENS_PRODID.to_owned(),
            version: PERENS_REV.to_owned(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    fn encode(&mut self, pcm8: &[i16]) -> Result<Vec<u8>> {
        use ham_digital_modes_dstar::ambe::float::dstar::encoder::Encoder;
        use ham_digital_modes_dstar::ambe::float::dstar::interleave::frame_to_wire_bytes;

        let fs = VocoderMode::Dstar.frame_samples();
        if !pcm8.len().is_multiple_of(fs) {
            return Err(DataError::Invalid(format!(
                "dstar: {} samples is not a whole number of {fs}-sample frames",
                pcm8.len()
            )));
        }
        let want = pcm8.len() / fs;
        // The codec works in i16-magnitude f64, not [-1, 1].
        let input: Vec<f64> = pcm8.iter().map(|&s| f64::from(s)).collect();
        let mut enc = Encoder::new();
        enc.push_samples(&input);
        let mut frames: Vec<u128> = Vec::with_capacity(want + 1);
        while let Some(f) = enc.next_frame() {
            frames.push(f);
        }
        frames.extend(enc.finish());
        // Lookahead plus the flush can emit one frame past the input;
        // upstream's own chip comparison truncates the same way.
        frames.truncate(want);
        if frames.len() != want {
            return Err(DataError::Invalid(format!(
                "dstar: the software vocoder produced {} frames for {want} frames of audio",
                frames.len()
            )));
        }
        let mut out = Vec::with_capacity(want * VocoderMode::Dstar.frame_bytes());
        for f in &frames {
            out.extend_from_slice(&frame_to_wire_bytes(*f));
        }
        Ok(out)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        use ham_digital_modes_dstar::ambe::float::dstar::interleave::wire_bytes_to_frame;
        use ham_digital_modes_dstar::ambe::float::dstar::synthesis::DStarSynthesisDecoder;

        let fs = VocoderMode::Dstar.frame_samples();
        let fb = VocoderMode::Dstar.frame_bytes();
        if !bytes.len().is_multiple_of(fb) {
            return Err(DataError::Invalid(format!(
                "dstar: {} bytes is not a whole number of {fb}-byte frames",
                bytes.len()
            )));
        }
        let mut dec = DStarSynthesisDecoder::new();
        let mut out = Vec::with_capacity(bytes.len() / fb * fs);
        for frame in bytes.chunks_exact(fb) {
            let mut wire = [0u8; 9];
            wire.copy_from_slice(frame);
            let pcm = dec
                .decode_frame(wire_bytes_to_frame(&wire))
                .unwrap_or([0.0; 160]);
            out.extend(pcm.iter().map(|&s| {
                let v = s.round();
                if v >= f64::from(i16::MAX) {
                    i16::MAX
                } else if v <= f64::from(i16::MIN) {
                    i16::MIN
                } else {
                    // Clamped to i16's range immediately above, so this
                    // cannot truncate.
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        v as i16
                    }
                }
            }));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimTransport;
    use crate::testutil::sine;

    /// The vendored revision must match what `VENDORED.md` claims, or a
    /// re-vendor silently relabels captures an older codec produced.
    #[cfg(feature = "perens")]
    #[test]
    fn the_pinned_perens_revision_matches_the_vendor_note() {
        let note = include_str!("../../../vendor/ham-digital-modes/VENDORED.md");
        assert!(
            note.contains(PERENS_REV),
            "VENDORED.md does not mention {PERENS_REV}; re-pin one or the other"
        );
    }

    /// `[-1, 1]` to s16 without a truncating float cast.
    #[cfg(feature = "perens")]
    fn to_s16(s: f32) -> i16 {
        let v = (s * f32::from(i16::MAX)).round();
        if v >= f32::from(i16::MAX) {
            i16::MAX
        } else if v <= f32::from(i16::MIN) {
            i16::MIN
        } else {
            // Clamped above.
            #[allow(clippy::cast_possible_truncation)]
            {
                v as i16
            }
        }
    }

    #[cfg(feature = "perens")]
    #[test]
    fn perens_round_trips_whole_frames() {
        let mode = VocoderMode::Dstar;
        let fs = mode.frame_samples();
        let frames = 12;
        let pcm = sine(220.0, 8_000, fs * frames, 0.3);
        let pcm16: Vec<i16> = pcm.iter().copied().map(to_s16).collect();
        let mut v = PerensVocoder::new(mode).unwrap();
        let bytes = v.encode(&pcm16).unwrap();
        assert_eq!(
            bytes.len(),
            frames * mode.frame_bytes(),
            "one frame in, one frame out"
        );
        let back = v.decode(&bytes).unwrap();
        assert_eq!(
            back.len(),
            frames * fs,
            "decode returns whole frames of audio"
        );
        assert!(back.iter().any(|&s| s != 0), "decoded to silence");
    }

    #[cfg(feature = "perens")]
    #[test]
    fn perens_refuses_a_partial_frame_and_other_modes() {
        let mut v = PerensVocoder::new(VocoderMode::Dstar).unwrap();
        assert!(
            v.encode(&vec![0i16; 161]).is_err(),
            "161 samples is not whole frames"
        );
        assert!(
            v.decode(&[0u8; 10]).is_err(),
            "10 bytes is not whole frames"
        );
        for m in [VocoderMode::YsfDmr, VocoderMode::Codec2_3200] {
            assert!(PerensVocoder::new(m).is_err(), "{m} is not D-STAR");
        }
    }

    /// Two runs, same bytes: the capture set must be reproducible.
    #[cfg(feature = "perens")]
    #[test]
    fn perens_is_deterministic_across_instances() {
        let pcm: Vec<i16> = sine(180.0, 8_000, 160 * 8, 0.4)
            .iter()
            .copied()
            .map(to_s16)
            .collect();
        let a = PerensVocoder::new(VocoderMode::Dstar)
            .unwrap()
            .encode(&pcm)
            .unwrap();
        let b = PerensVocoder::new(VocoderMode::Dstar)
            .unwrap()
            .encode(&pcm)
            .unwrap();
        assert_eq!(a, b, "same input, same vendored revision, same bytes");
    }

    #[test]
    fn padding_rounds_up_to_whole_frames_of_the_mode() {
        assert_eq!(frames_for(0, VocoderMode::Dstar), 0);
        assert_eq!(frames_for(1, VocoderMode::Dstar), 1);
        assert_eq!(frames_for(160, VocoderMode::Dstar), 1);
        assert_eq!(frames_for(161, VocoderMode::Dstar), 2);
        assert_eq!(frames_for(161, VocoderMode::Codec2_3200), 2);
        assert_eq!(frames_for(320, VocoderMode::Codec2_1600), 1);
        assert_eq!(frames_for(321, VocoderMode::Codec2_1600), 2);
        let p = pad_frames(&[0.5; 161], VocoderMode::Codec2_1600);
        assert_eq!(p.len(), 320);
        assert_eq!(p[0], 16_384);
        assert_eq!(p[160], 16_384);
        assert!(p[161..].iter().all(|&v| v == 0));
        assert_eq!(pad_frames(&[0.5; 321], VocoderMode::Codec2_1600).len(), 640);
    }

    #[test]
    fn chip_vocoder_wraps_the_sim() {
        let mut v = ChipVocoder::init(SimTransport::new(), VocoderMode::YsfDmr).unwrap();
        assert_eq!(v.mode(), VocoderMode::YsfDmr);
        assert_eq!(v.info().prodid, "AMBE3000F");
        let pcm = pad_frames(&sine(300.0, 8_000, 1_600, 0.3), VocoderMode::YsfDmr);
        let frames = v.encode(&pcm).unwrap();
        assert_eq!(frames.len(), 10 * 7);
        let out = v.decode(&frames).unwrap();
        assert_eq!(out.len(), 1_600);
        v.reset().unwrap();
        assert_eq!(v.chip().inits(), 2);
        v.warm_up(&sine(300.0, 8_000, 8_000, 0.3)).unwrap();
        let sim = v.into_transport();
        assert_eq!(sim.encodes, 10 + 20);
    }

    /// The chip vocoder interleaves by default and says so, and its
    /// round trip is the same bytes the two separate passes give.
    /// `--sequential` turns the interleaving off without changing either
    /// result.
    #[test]
    fn chip_round_trip_interleaves_unless_it_is_told_not_to() {
        let mode = VocoderMode::YsfDmr;
        let clip = sine(300.0, 8_000, 3_200, 0.3);
        let pcm = pad_frames(&clip, mode);

        let mut split = ChipVocoder::init(SimTransport::new(), mode).unwrap();
        let want_frames = split.encode(&pcm).unwrap();
        let want_pcm = split.decode(&want_frames).unwrap();

        let mut fast = ChipVocoder::init(
            SimTransport::new().with_max_in_flight(crate::pipeline::INTERLEAVE_DEPTH),
            mode,
        )
        .unwrap();
        assert!(fast.interleaves());
        assert!(!fast.is_sequential());
        let (frames, out) = fast.round_trip(&pcm).unwrap();
        assert_eq!(frames, want_frames);
        assert_eq!(out, want_pcm);
        let sim = fast.into_transport();
        assert_eq!(
            sim.peak_encodes_in_flight,
            crate::pipeline::INTERLEAVE_DEPTH
        );
        assert_eq!(
            sim.peak_decodes_in_flight,
            crate::pipeline::INTERLEAVE_DEPTH
        );

        let mut slow = ChipVocoder::init_with(SimTransport::new(), mode, true).unwrap();
        assert!(!slow.interleaves());
        assert!(slow.is_sequential());
        assert_eq!(slow.round_trip(&pcm).unwrap(), (want_frames, want_pcm));
        // Encode and decode alone are still there: `augment` decodes
        // without encoding and the canary encodes without decoding.
        let mut v = ChipVocoder::init(SimTransport::new(), mode).unwrap();
        let only_frames = v.encode(&pcm).unwrap();
        assert_eq!(only_frames.len(), 20 * mode.frame_bytes());
        assert_eq!(v.decode(&only_frames).unwrap().len(), pcm.len());
    }

    /// A software vocoder keeps the default: two passes, and it says so,
    /// so its rows keep `encode_ms` / `decode_ms`.
    #[cfg(feature = "codec2")]
    #[test]
    fn the_software_vocoder_does_not_interleave() {
        let mode = VocoderMode::Codec2_3200;
        let mut v = Codec2Vocoder::new(mode).unwrap();
        assert!(!v.interleaves());
        let pcm = pad_frames(&sine(220.0, 8_000, 1_600, 0.3), mode);
        let (frames, out) = v.round_trip(&pcm).unwrap();
        assert_eq!(frames.len(), 10 * mode.frame_bytes());
        assert_eq!(out.len(), pcm.len());
    }

    #[test]
    fn open_software_refuses_a_chip_mode() {
        let err = open_software(VocoderMode::Dstar).err().expect("refused");
        assert!(err.to_string().contains("chip mode"), "{err}");
    }

    #[cfg(feature = "codec2")]
    mod codec2 {
        use super::*;
        use unamblify_audio::xcorr_lag;

        #[test]
        fn crate_pin_matches_the_recorded_version() {
            let manifest =
                std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
                    .unwrap();
            let want = format!("codec2 = {{ version = \"={CODEC2_CRATE_VERSION}\"");
            assert!(
                manifest.contains(&want),
                "Cargo.toml must pin `{want}`; the version string goes into every capture row"
            );
        }

        #[test]
        fn both_modes_round_trip_with_the_right_frame_words() {
            for (mode, fs, fb) in [
                (VocoderMode::Codec2_3200, 160, 8),
                (VocoderMode::Codec2_1600, 320, 8),
            ] {
                let mut v = Codec2Vocoder::new(mode).unwrap();
                assert_eq!(v.mode(), mode);
                assert_eq!(v.info().prodid, "codec2");
                assert_eq!(v.info().version, CODEC2_CRATE_VERSION);
                let clip = sine(220.0, 8_000, 8_000 + 7, 0.3);
                let pcm = pad_frames(&clip, mode);
                assert_eq!(pcm.len() % fs, 0);
                let frames = v.encode(&pcm).unwrap();
                assert_eq!(frames.len(), pcm.len() / fs * fb, "{mode}");
                assert!(frames.iter().any(|&b| b != 0), "{mode}: silent frames");
                let out = v.decode(&frames).unwrap();
                assert_eq!(out.len(), pcm.len(), "{mode}");
                let out_f: Vec<f32> = out.iter().map(|&s| f32::from(s) / 32_767.0).collect();
                let peak = out_f.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                assert!(peak > 0.05, "{mode}: decoded peak {peak}");
                // The vocoder's own delay is small and the tone comes back
                // recognisably: the correlation lag lands inside a frame.
                let lag = xcorr_lag(&clip, &out_f[..clip.len()], 400);
                assert!(lag.unsigned_abs() as usize <= fs, "{mode}: lag {lag}");
                // Ragged input is refused, not silently padded.
                assert!(v.encode(&pcm[..=fs]).is_err());
                assert!(v.decode(&frames[..=fb]).is_err());
            }
        }

        #[test]
        fn encode_is_deterministic_and_stateless_across_calls() {
            let mut v = Codec2Vocoder::new(VocoderMode::Codec2_3200).unwrap();
            let a = pad_frames(&sine(180.0, 8_000, 3_200, 0.2), VocoderMode::Codec2_3200);
            let b = pad_frames(&sine(410.0, 8_000, 1_600, 0.2), VocoderMode::Codec2_3200);
            let fresh = v.encode(&a).unwrap();
            let _ = v.encode(&b).unwrap();
            let again = v.encode(&a).unwrap();
            assert_eq!(fresh, again, "history must not leak between utterances");
            let mut w = Codec2Vocoder::new(VocoderMode::Codec2_3200).unwrap();
            assert_eq!(w.encode(&a).unwrap(), fresh, "nor between instances");
            assert_ne!(
                Codec2Vocoder::new(VocoderMode::Codec2_1600)
                    .unwrap()
                    .encode(&a)
                    .unwrap(),
                fresh
            );
        }

        #[test]
        fn a_chip_mode_is_refused() {
            assert!(Codec2Vocoder::new(VocoderMode::Dstar).is_err());
            assert!(open_software(VocoderMode::Codec2_1600).is_ok());
        }
    }
}
