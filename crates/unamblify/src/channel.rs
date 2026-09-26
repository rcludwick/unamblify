// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Channel errors and dropped frames on stored channel frames
//! (`docs/design/data-pipeline.md`, stage 3, "decode-only"): the pure
//! byte mutations `unamblify augment` applies before re-running only the
//! decode pass. Time is never shifted — a lost frame is replaced, not
//! removed, as on the air — so every position maps straight onto the
//! decoded audio.
//!
//! The mute codewords of the AMBE modes are constants here; Codec 2 has
//! no fixed mute word (an all-zero frame is *not* silence), so the data
//! crate encodes 160 / 320 samples of digital silence once with the
//! crate and passes those bytes in.

use crate::Rng;
use crate::aug::Subst;
use crate::{VocoderFamily, VocoderMode};

/// The D-STAR AMBE null / silence frame (72 bits): what MMDVM-style hosts
/// and astar send the vocoder for a lost frame.
///
/// It is also the DMR null frame the future `augment --kind ber` stage
/// will use when it synthesises DMR framing (49 voice bits plus 23 bits
/// of Golay FEC) from a `ysf-dmr` capture: DMR is not a capture mode —
/// its voice bits are YSF DN's, and the FEC only matters under bit
/// errors — so the constant stays here for that path alone.
pub const NULL_AMBE_FRAME: [u8; 9] = [0x9E, 0x8D, 0x32, 0x88, 0x26, 0x1A, 0x3F, 0x61, 0xE8];

/// The YSF DN / DMR (AMBE+2 2450, 49 bits) mute frame, in chip wire order.
pub const YSF_DMR_MUTE_FRAME: [u8; 7] = [0xF0, 0x00, 0x31, 0x00, 0x00, 0x00, 0x00];

/// The mute codeword of an AMBE mode, `None` for the software family.
#[must_use]
pub const fn ambe_mute_frame(mode: VocoderMode) -> Option<&'static [u8]> {
    match mode {
        VocoderMode::Dstar => Some(&NULL_AMBE_FRAME),
        VocoderMode::YsfDmr => Some(&YSF_DMR_MUTE_FRAME),
        VocoderMode::Codec2_3200 | VocoderMode::Codec2_1600 => None,
    }
}

/// Whether a mode's mute frame is a constant of this module.
#[must_use]
pub const fn has_constant_mute(mode: VocoderMode) -> bool {
    matches!(mode.family(), VocoderFamily::Ambe)
}

/// Draw the lost frames of an utterance: bursts of `burst.0..=burst.1`
/// consecutive frames, started so that about `rate` of all frames are
/// lost (the burst-start probability is `rate` over the mean burst
/// length). Bursts never overlap. Positions ascend.
#[must_use]
pub fn plan_drops(rng: &mut Rng, frames: usize, rate: f32, burst: (u32, u32)) -> Vec<u32> {
    let (lo, hi) = (
        burst.0.max(1) as usize,
        burst.1.max(burst.0.max(1)) as usize,
    );
    #[allow(clippy::cast_precision_loss)]
    let mean = (lo + hi) as f32 / 2.0;
    let p_start = (rate.clamp(0.0, 1.0) / mean).min(1.0);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < frames {
        if rng.chance(p_start) {
            let len = rng.range(lo, hi);
            for f in i..(i + len).min(frames) {
                out.push(u32::try_from(f).unwrap_or(u32::MAX));
            }
            i += len;
        } else {
            i += 1;
        }
    }
    out
}

/// Substitute the frames at `positions` in place: `mute` copies the
/// codeword in, `repeat` holds the last good frame (the mute word when the
/// first frame is lost), and `erase` hands the frame to the receiver's own
/// concealment — [`fill_erasures`] in the parameter domain for the Codec 2
/// modes, and for the AMBE modes the chip's, which this harness cannot ask
/// for and so refuses. Frames are `mode.frame_bytes()` each; `mute` must be
/// one frame long. The frame count never changes.
pub fn apply_drops(
    bytes: &mut [u8],
    mode: VocoderMode,
    positions: &[u32],
    subst: Subst,
    mute: &[u8],
) -> Result<(), String> {
    let frame_bytes = mode.frame_bytes();
    if frame_bytes == 0 || !bytes.len().is_multiple_of(frame_bytes) {
        return Err(format!(
            "{} bytes is not a whole number of {frame_bytes}-byte frames",
            bytes.len()
        ));
    }
    if mute.len() != frame_bytes {
        return Err(format!(
            "mute frame is {} bytes, frames are {frame_bytes}",
            mute.len()
        ));
    }
    let frames = bytes.len() / frame_bytes;
    match subst {
        // The receiver's own concealment. The AMBE modes would need the
        // chip's (a driver capability this harness does not have, and
        // `vendor/ambe-thumbdv` is vendored verbatim); Codec 2 has none of
        // its own, so `fill_erasures` supplies it in the parameter domain.
        Subst::Erase => {
            if !mode.is_software() {
                return Err(
                    "erase (mark the frame bad and let the chip conceal) is chip-only for the \
                     AMBE modes and not implemented; use mute or repeat"
                        .to_owned(),
                );
            }
            return fill_erasures(bytes, mode, positions, mute);
        }
        Subst::Mute | Subst::Repeat => {}
    }
    for &p in positions {
        let p = p as usize;
        if p >= frames {
            return Err(format!("position {p} past the last frame ({frames})"));
        }
        let (before, at) = bytes.split_at_mut(p * frame_bytes);
        let frame = &mut at[..frame_bytes];
        match subst {
            Subst::Mute => frame.copy_from_slice(mute),
            Subst::Repeat if p == 0 => frame.copy_from_slice(mute),
            // The previous frame as it now stands: a held frame keeps
            // holding through a burst.
            Subst::Repeat => frame.copy_from_slice(&before[(p - 1) * frame_bytes..]),
            Subst::Erase => unreachable!("refused above"),
        }
    }
    Ok(())
}

// ── Codec 2 parameter-domain concealment ──────────────────────────────

/// One scalar field of a Codec 2 frame: bit offset from the frame's first
/// bit, and width in bits.
#[derive(Clone, Copy)]
struct BitField {
    offset: usize,
    bits: usize,
}

/// Where the parameters sit in a Codec 2 frame.
///
/// Both modes pack 64 bits into 8 bytes, MSB first, every field Gray
/// coded. 3200 codes one (pitch, energy) point per 20 ms frame; 1600
/// codes two per 40 ms frame (sub-frames 1 and 3). The LSP block is held
/// rather than interpolated: in 3200 it is differentially coded, and both
/// modes' ladders are the codec2 crate's LGPL codebooks, which this
/// AGPL crate does not restate. Holding the envelope across one lost
/// frame is barely audible — the decoder already interpolates its own
/// LSPs from the previous frame — while a pitch or energy step is the
/// click.
struct Codec2Layout {
    /// `(Wo, energy)` per coded point, in time order.
    points: &'static [(BitField, BitField)],
    /// The whole LSP block, held across a gap.
    lsp: BitField,
}

/// Mode 3200: voicing 0–1, Wo 2–8, energy 9–13, LSP deltas 14–63.
const C2_3200: Codec2Layout = Codec2Layout {
    points: &[(
        BitField { offset: 2, bits: 7 },
        BitField { offset: 9, bits: 5 },
    )],
    lsp: BitField {
        offset: 14,
        bits: 50,
    },
};

/// Mode 1600: two (voicing, voicing, Wo, energy) blocks, then LSPs 28–63.
const C2_1600: Codec2Layout = Codec2Layout {
    points: &[
        (
            BitField { offset: 2, bits: 7 },
            BitField { offset: 9, bits: 5 },
        ),
        (
            BitField {
                offset: 16,
                bits: 7,
            },
            BitField {
                offset: 23,
                bits: 5,
            },
        ),
    ],
    lsp: BitField {
        offset: 28,
        bits: 36,
    },
};

/// The frame layout of a Codec 2 mode, `None` for the AMBE modes.
const fn codec2_layout(mode: VocoderMode) -> Option<&'static Codec2Layout> {
    match mode {
        VocoderMode::Codec2_3200 => Some(&C2_3200),
        VocoderMode::Codec2_1600 => Some(&C2_1600),
        VocoderMode::Dstar | VocoderMode::YsfDmr => None,
    }
}

/// Read `n` bits MSB first at `bit_off`.
fn read_bits(frame: &[u8], bit_off: usize, n: usize) -> u32 {
    let mut v = 0;
    for i in 0..n {
        let b = bit_off + i;
        v = (v << 1) | u32::from((frame[b / 8] >> (7 - (b % 8))) & 1);
    }
    v
}

/// Write `n` bits MSB first at `bit_off`.
fn write_bits(frame: &mut [u8], bit_off: usize, n: usize, v: u32) {
    for i in 0..n {
        let b = bit_off + i;
        let mask = 1u8 << (7 - (b % 8));
        if (v >> (n - 1 - i)) & 1 == 1 {
            frame[b / 8] |= mask;
        } else {
            frame[b / 8] &= !mask;
        }
    }
}

/// Binary → Gray, as the codec2 crate packs every field.
const fn to_gray(v: u32) -> u32 {
    (v >> 1) ^ v
}

/// Gray → binary. The fold covers the widest field here (7 bits).
const fn from_gray(g: u32) -> u32 {
    let mut b = g;
    b ^= b >> 1;
    b ^= b >> 2;
    b ^= b >> 4;
    b
}

/// The value of a Gray-coded field.
fn get_field(frame: &[u8], f: BitField) -> u32 {
    from_gray(read_bits(frame, f.offset, f.bits))
}

/// Set a Gray-coded field.
fn set_field(frame: &mut [u8], f: BitField, v: u32) {
    write_bits(frame, f.offset, f.bits, to_gray(v));
}

/// Copy a field's raw bits between frames, whatever its width (the LSP
/// block is wider than a `u32`, and it is held, never interpreted).
fn copy_field(dst: &mut [u8], src: &[u8], f: BitField) {
    for i in 0..f.bits {
        let b = f.offset + i;
        let mask = 1u8 << (7 - (b % 8));
        if src[b / 8] & mask == 0 {
            dst[b / 8] &= !mask;
        } else {
            dst[b / 8] |= mask;
        }
    }
}

/// Round a linear interpolation between two quantiser indices, clamped to
/// the field.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn lerp_index(a: u32, b: u32, alpha: f64, max: u32) -> u32 {
    let v = f64::from(a).mul_add(1.0 - alpha, f64::from(b) * alpha);
    (v.round().max(0.0) as u32).min(max)
}

/// Energy indices shed per frame when a gap runs to the end of the
/// utterance and there is nothing to interpolate toward: the quantiser is
/// 50 dB over 32 steps, so two steps is about 3 dB per frame — a decay to
/// silence rather than a sustained tone.
const ENERGY_DECAY_PER_FRAME: u32 = 2;

/// Conceal the erased frames of a Codec 2 stream in the parameter domain.
///
/// Pitch and energy are interpolated across the gap between the good
/// frames either side of it. Both quantisers are uniform — Wo is linear in
/// radians and energy linear in dB — so interpolating the *index*
/// interpolates the value, and no codebook table is needed. Voicing and
/// the spectral envelope are held from the nearest good frame. A gap with
/// no good frame before it holds backward from the one after; a gap that
/// runs to the end of the utterance decays the energy toward silence. If
/// every frame is erased there is nothing to conceal from and all of them
/// are muted.
///
/// This is what `--subst erase` means for the software modes: Codec 2 has
/// no concealment of its own, and feeding the decoder a plausible frame
/// keeps its `prev_lsps` / `prev_e` interpolation state on a smooth
/// trajectory, which muting or repeating does not.
pub fn fill_erasures(
    bytes: &mut [u8],
    mode: VocoderMode,
    positions: &[u32],
    mute: &[u8],
) -> Result<(), String> {
    let Some(layout) = codec2_layout(mode) else {
        return Err(format!("{mode} has no parameter-domain concealment"));
    };
    let fb = mode.frame_bytes();
    if fb == 0 || !bytes.len().is_multiple_of(fb) {
        return Err(format!(
            "{} bytes is not a whole number of {fb}-byte frames",
            bytes.len()
        ));
    }
    if mute.len() != fb {
        return Err(format!(
            "mute frame is {} bytes, frames are {fb}",
            mute.len()
        ));
    }
    let frames = bytes.len() / fb;
    let mut lost = vec![false; frames];
    for &p in positions {
        let p = p as usize;
        if p >= frames {
            return Err(format!("position {p} past the last frame ({frames})"));
        }
        lost[p] = true;
    }
    if frames == 0 {
        return Ok(());
    }
    if lost.iter().all(|&l| l) {
        for f in 0..frames {
            bytes[f * fb..(f + 1) * fb].copy_from_slice(mute);
        }
        return Ok(());
    }

    let n_points = layout.points.len();
    // Coded points are spaced evenly and the last one lands on the frame
    // boundary, so point `s` of frame `f` sits at `f + (s + 1) / n_points`
    // in frame units.
    #[allow(clippy::cast_precision_loss)]
    let point_time = |f: usize, s: usize| f as f64 + (s + 1) as f64 / n_points as f64;

    let mut i = 0;
    while i < frames {
        if !lost[i] {
            i += 1;
            continue;
        }
        let (start, mut end) = (i, i);
        while end + 1 < frames && lost[end + 1] {
            end += 1;
        }
        let prev = start.checked_sub(1);
        let next = (end + 1 < frames).then_some(end + 1);

        // Anchors: the last coded point before the gap and the first one
        // after it, read before anything in the gap is rewritten.
        let before = prev.map(|p| {
            let fr = &bytes[p * fb..(p + 1) * fb];
            let (wo, e) = layout.points[n_points - 1];
            (
                get_field(fr, wo),
                get_field(fr, e),
                point_time(p, n_points - 1),
            )
        });
        let after = next.map(|n| {
            let fr = &bytes[n * fb..(n + 1) * fb];
            let (wo, e) = layout.points[0];
            (get_field(fr, wo), get_field(fr, e), point_time(n, 0))
        });
        // Voicing and the envelope are held from the nearer good frame.
        let held_from = prev.or(next).ok_or("no good frame to conceal from")?;
        let held = bytes[held_from * fb..(held_from + 1) * fb].to_vec();

        for f in start..=end {
            let mut out = held.clone();
            for (s, &(wo_f, e_f)) in layout.points.iter().enumerate() {
                let t = point_time(f, s);
                let (wo, e) = match (before, after) {
                    (Some((wa, ea, ta)), Some((wb, eb, tb))) => {
                        let alpha = if (tb - ta).abs() < f64::EPSILON {
                            0.0
                        } else {
                            ((t - ta) / (tb - ta)).clamp(0.0, 1.0)
                        };
                        (
                            lerp_index(wa, wb, alpha, mask_of(wo_f)),
                            lerp_index(ea, eb, alpha, mask_of(e_f)),
                        )
                    }
                    // Runs to the end of the utterance: hold the pitch and
                    // let the energy fall away.
                    (Some((wa, ea, _)), None) => {
                        let steps = u32::try_from(f - start + 1).unwrap_or(u32::MAX);
                        (wa, ea.saturating_sub(ENERGY_DECAY_PER_FRAME * steps))
                    }
                    // Starts the utterance: hold backward from the first
                    // good frame.
                    (None, Some((wb, eb, _))) => (wb, eb),
                    (None, None) => unreachable!("not every frame is lost"),
                };
                set_field(&mut out, wo_f, wo);
                set_field(&mut out, e_f, e);
            }
            copy_field(&mut out, &held, layout.lsp);
            bytes[f * fb..(f + 1) * fb].copy_from_slice(&out);
        }
        i = end + 1;
    }
    Ok(())
}

/// The largest value a field can hold.
const fn mask_of(f: BitField) -> u32 {
    (1u32 << f.bits) - 1
}

/// Flip each of the `channel_bits` valid bits of every frame with
/// probability `rate` (the padding bits of a 49-bit frame are left
/// alone). Bits are numbered MSB-first within the frame, as the chip
/// packs its channel packet. Returns the frames with at least one flip,
/// ascending.
pub fn apply_ber(
    bytes: &mut [u8],
    frame_bytes: usize,
    channel_bits: u16,
    rate: f32,
    rng: &mut Rng,
) -> Result<Vec<u32>, String> {
    if frame_bytes == 0 || !bytes.len().is_multiple_of(frame_bytes) {
        return Err(format!(
            "{} bytes is not a whole number of {frame_bytes}-byte frames",
            bytes.len()
        ));
    }
    let bits = usize::from(channel_bits).min(frame_bytes * 8);
    let mut touched = Vec::new();
    for (f, frame) in bytes.chunks_exact_mut(frame_bytes).enumerate() {
        let mut hit = false;
        for b in 0..bits {
            if rng.chance(rate) {
                frame[b / 8] ^= 0x80 >> (b % 8);
                hit = true;
            }
        }
        if hit {
            touched.push(u32::try_from(f).unwrap_or(u32::MAX));
        }
    }
    Ok(touched)
}

// ── YSF DN bit errors ─────────────────────────────────────────────────

/// Which YSF DN voice/data mode's FEC to model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YsfVd {
    /// V/D mode 1: Golay (24, 12) over twelve bits, Golay (23, 12) over
    /// twelve more, twenty-five bits bare.
    Vd1,
    /// V/D mode 2: the first twenty-seven bits sent three times and voted,
    /// twenty-two bits bare.
    Vd2,
}

fn read_bit(buf: &[u8], i: usize) -> bool {
    buf[i / 8] >> (7 - i % 8) & 1 == 1
}

fn write_bit(buf: &mut [u8], i: usize, v: bool) {
    let mask = 0x80u8 >> (i % 8);
    if v {
        buf[i / 8] |= mask;
    } else {
        buf[i / 8] &= !mask;
    }
}

/// Flip `bits` transmitted bits, each with probability `rate`.
fn hit(word: u32, bits: usize, rate: f32, rng: &mut Rng) -> u32 {
    let mut out = word;
    for i in 0..bits {
        if rng.chance(rate) {
            out ^= 1 << i;
        }
    }
    out
}

fn ysf_vd1_frame(frame: &mut [u8], rate: f32, rng: &mut Rng) {
    let (mut a, mut b, mut c) = (0u16, 0u16, 0u32);
    for i in 0..12 {
        a = (a << 1) | u16::from(read_bit(frame, i));
        b = (b << 1) | u16::from(read_bit(frame, 12 + i));
    }
    for i in 0..25 {
        c = (c << 1) | u32::from(read_bit(frame, 24 + i));
    }

    // Transmitted form: two Golay words, the second whitened by a sequence
    // keyed on the first, and twenty-five bits with no cover at all.
    let a24 = hit(crate::fec::golay24_encode(a), 24, rate, rng);
    let b23 = hit(
        (crate::fec::golay24_encode(b) >> 1) ^ (crate::fec::whiten(a) >> 1),
        23,
        rate,
        rng,
    );
    let c25 = hit(c, 25, rate, rng);

    // The receiver's turn. The first word keys the second, so losing it
    // loses the whole frame, the unprotected bits included.
    let Some(u0) = crate::fec::golay24_decode(a24) else {
        frame.copy_from_slice(&YSF_DMR_MUTE_FRAME);
        return;
    };
    let u1 = crate::fec::golay23_decode(b23 ^ (crate::fec::whiten(u0) >> 1));
    for i in 0..12 {
        write_bit(frame, i, (u0 >> (11 - i)) & 1 == 1);
        write_bit(frame, 12 + i, (u1 >> (11 - i)) & 1 == 1);
    }
    for i in 0..25 {
        write_bit(frame, 24 + i, (c25 >> (24 - i)) & 1 == 1);
    }
}

fn ysf_vd2_frame(frame: &mut [u8], rate: f32, rng: &mut Rng) {
    // Twenty-seven bits three times over and voted, then twenty-two bare.
    for k in 0..27 {
        let bit = read_bit(frame, k);
        let v0 = bit ^ rng.chance(rate);
        let v1 = bit ^ rng.chance(rate);
        let v2 = bit ^ rng.chance(rate);
        write_bit(frame, k, crate::fec::majority3(v0, v1, v2));
    }
    for k in 0..22 {
        let bit = read_bit(frame, 27 + k);
        write_bit(frame, 27 + k, bit ^ rng.chance(rate));
    }
}

/// Flip bits the way the air does to YSF DN frames, then decode as a
/// receiver would, leaving the residual errors in the 49 voice bits.
///
/// [`apply_ber`] flips voice bits directly, which no receiver ever sees:
/// on the air the protected bits are protected. Here each frame is wrapped
/// in the mode's FEC, every transmitted bit is flipped with probability
/// `rate`, and the result is decoded. What survives is what the vocoder
/// would actually have been handed.
///
/// A mode 1 frame whose first word cannot be repaired becomes
/// [`YSF_DMR_MUTE_FRAME`], the codeword a receiver substitutes. Returns the
/// frames whose voice bits ended up different.
pub fn apply_ysf_ber(
    bytes: &mut [u8],
    mode: VocoderMode,
    vd: YsfVd,
    rate: f32,
    rng: &mut Rng,
) -> Result<Vec<u32>, String> {
    if mode != VocoderMode::YsfDmr {
        return Err(format!(
            "{mode}: YSF bit errors model YSF DN framing and apply to ysf-dmr only"
        ));
    }
    let fb = mode.frame_bytes();
    if fb == 0 || !bytes.len().is_multiple_of(fb) {
        return Err(format!(
            "{} bytes is not a whole number of {fb}-byte frames",
            bytes.len()
        ));
    }
    let rate = rate.clamp(0.0, 1.0);
    let mut touched = Vec::new();
    for (f, frame) in bytes.chunks_exact_mut(fb).enumerate() {
        let before = frame.to_vec();
        match vd {
            YsfVd::Vd1 => ysf_vd1_frame(frame, rate, rng),
            YsfVd::Vd2 => ysf_vd2_frame(frame, rate, rng),
        }
        if frame != before.as_slice() {
            touched.push(u32::try_from(f).unwrap_or(u32::MAX));
        }
    }
    Ok(touched)
}

/// The erasure mask of a crop: which of its frames hold concealed audio.
///
/// One rule for the shard builder, the trainer's eval and a runtime: all
/// three must agree on where a lost channel frame's audio lands.
///
/// `erased` indexes channel frames of the whole utterance. The decoded
/// audio of channel frame `k` is `decoded[k·fs .. (k+1)·fs)`, and `deg8` is
/// that signal advanced by the codec's `lag` (`deg8[i] = decoded[i + lag]`),
/// so in the crop's own timeline — which starts at utterance sample
/// `f0·fs` — frame `k`'s audio occupies
/// `[k·fs − lag − f0·fs, (k+1)·fs − lag − f0·fs)`. With D-STAR's lag of 326
/// samples that is two frames earlier than the index alone would say. A
/// crop frame is marked when any part of it overlaps a lost frame's audio:
/// the model should distrust a frame that is even partly concealment. The
/// runtime can build the same mask, since the lag is fixed per mode.
#[must_use]
pub fn erasure_mask(
    erased: &[u32],
    f0: usize,
    speech_frames: usize,
    fs: usize,
    lag: i32,
    len: usize,
) -> Vec<u8> {
    let mut mask = vec![0u8; len];
    if len == 0 || erased.is_empty() {
        return mask;
    }
    // Signed sample arithmetic: a frame's audio can start before the crop.
    let to_i = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    let fs_i = to_i(fs);
    let origin = to_i(f0 * fs) + i64::from(lag);
    let end = to_i(speech_frames.min(len) * fs);
    for &k in erased {
        let a = i64::from(k) * fs_i - origin;
        let b = a + fs_i;
        if b <= 0 || a >= end {
            continue;
        }
        let first = usize::try_from(a.max(0) / fs_i).unwrap_or(usize::MAX);
        let last = usize::try_from((b.min(end) - 1) / fs_i).unwrap_or(0);
        for m in mask.iter_mut().take(last + 1).skip(first) {
            *m = 1;
        }
    }
    mask
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::verbose_bit_mask
)]
mod tests {
    use super::*;

    /// The mask marks where the concealed *audio* lands, not the index of
    /// the lost channel frame: `deg8` is advanced by the codec lag, so with
    /// D-STAR's 326 samples a lost frame damages the two frames before it.
    #[test]
    fn the_erasure_mask_follows_the_lag_aligned_audio() {
        let fs = 160;
        let ones = |m: &[u8]| -> Vec<usize> {
            m.iter()
                .enumerate()
                .filter(|(_, v)| **v == 1)
                .map(|(i, _)| i)
                .collect()
        };
        // No lag: frame k is frame k.
        assert_eq!(ones(&erasure_mask(&[10], 0, 100, fs, 0, 100)), vec![10]);
        // A whole-frame lag moves it by exactly that many frames.
        assert_eq!(ones(&erasure_mask(&[10], 0, 100, fs, 320, 100)), vec![8]);
        // D-STAR's 326: samples [1274, 1434) straddle frames 7 and 8.
        assert_eq!(ones(&erasure_mask(&[10], 0, 100, fs, 326, 100)), vec![7, 8]);
        // A crop that starts at frame 5 sees it five frames earlier.
        assert_eq!(ones(&erasure_mask(&[10], 5, 100, fs, 326, 100)), vec![2, 3]);
        // A burst is contiguous; frames outside the crop do not leak in.
        assert_eq!(
            ones(&erasure_mask(&[0, 20, 21, 22, 500], 0, 100, fs, 320, 100)),
            vec![18, 19, 20]
        );
        // Padding frames past the speech are never marked.
        assert_eq!(
            ones(&erasure_mask(&[52], 0, 50, fs, 0, 100)),
            Vec::<usize>::new()
        );
        // A set without the mask gets an empty one.
        assert!(erasure_mask(&[10], 0, 100, fs, 0, 0).is_empty());
    }

    #[test]
    fn mute_words_are_pinned() {
        assert_eq!(
            ambe_mute_frame(VocoderMode::Dstar),
            Some(&NULL_AMBE_FRAME[..])
        );
        assert_eq!(
            ambe_mute_frame(VocoderMode::YsfDmr),
            Some(&YSF_DMR_MUTE_FRAME[..])
        );
        assert_eq!(ambe_mute_frame(VocoderMode::Codec2_3200), None);
        assert!(has_constant_mute(VocoderMode::YsfDmr));
        assert!(!has_constant_mute(VocoderMode::Codec2_1600));
        assert_eq!(NULL_AMBE_FRAME.len(), VocoderMode::Dstar.frame_bytes());
        assert_eq!(YSF_DMR_MUTE_FRAME.len(), VocoderMode::YsfDmr.frame_bytes());
        // 49 bits: the low seven bits of the last byte are padding.
        assert_eq!(YSF_DMR_MUTE_FRAME[6] & 0x7F, 0);
    }

    #[test]
    fn drops_hit_the_rate_in_bursts_and_are_deterministic() {
        let frames = 20_000;
        let a = plan_drops(&mut Rng::new(5), frames, 0.02, (1, 3));
        let b = plan_drops(&mut Rng::new(5), frames, 0.02, (1, 3));
        assert_eq!(a, b);
        let share = a.len() as f32 / frames as f32;
        assert!((share - 0.02).abs() < 0.004, "{share}");
        assert!(a.windows(2).all(|w| w[0] < w[1]), "ascending, no repeats");
        assert!(a.iter().all(|&p| (p as usize) < frames));
        // Burst lengths are 1..=3.
        let mut runs = Vec::new();
        let mut run = 1;
        for w in a.windows(2) {
            if w[1] == w[0] + 1 {
                run += 1;
            } else {
                runs.push(run);
                run = 1;
            }
        }
        runs.push(run);
        // Two bursts can land back to back and read as one longer run;
        // that is rare at 2 %, and every run is otherwise 1..=3.
        let long = runs.iter().filter(|&&r| r > 3).count();
        assert!(
            long * 20 < runs.len(),
            "{long} merged runs of {}",
            runs.len()
        );
        assert!(runs.contains(&1) && runs.contains(&2) && runs.contains(&3));
        assert!(plan_drops(&mut Rng::new(1), 100, 0.0, (1, 3)).is_empty());
        assert_eq!(
            plan_drops(&mut Rng::new(1), 10, 1.0, (1, 1)).len(),
            10,
            "rate 1 with single-frame bursts loses everything"
        );
    }

    #[test]
    fn drops_substitute_the_right_bytes_and_keep_the_count() {
        let fb = 9;
        let frames = 8;
        let orig: Vec<u8> = (0..frames * fb).map(|i| (i / fb) as u8 + 1).collect();
        let mut muted = orig.clone();
        apply_drops(
            &mut muted,
            VocoderMode::Dstar,
            &[0, 3, 4],
            Subst::Mute,
            &NULL_AMBE_FRAME,
        )
        .unwrap();
        assert_eq!(muted.len(), orig.len());
        for f in 0..frames {
            let frame = &muted[f * fb..(f + 1) * fb];
            if [0, 3, 4].contains(&f) {
                assert_eq!(frame, &NULL_AMBE_FRAME[..], "frame {f}");
            } else {
                assert_eq!(frame, &orig[f * fb..(f + 1) * fb], "frame {f}");
            }
        }
        let mut held = orig.clone();
        apply_drops(
            &mut held,
            VocoderMode::Dstar,
            &[0, 3, 4],
            Subst::Repeat,
            &NULL_AMBE_FRAME,
        )
        .unwrap();
        assert_eq!(&held[..fb], &NULL_AMBE_FRAME[..], "nothing before frame 0");
        assert_eq!(held[3 * fb], 3, "frame 3 holds frame 2");
        assert_eq!(held[4 * fb], 3, "frame 4 keeps holding frame 2");
        assert_eq!(held[5 * fb], 6, "frame 5 is untouched");
        let err = apply_drops(
            &mut held,
            VocoderMode::Dstar,
            &[1],
            Subst::Erase,
            &NULL_AMBE_FRAME,
        )
        .unwrap_err();
        assert!(err.contains("chip-only"), "{err}");
        assert!(
            apply_drops(
                &mut held,
                VocoderMode::Dstar,
                &[8],
                Subst::Mute,
                &NULL_AMBE_FRAME
            )
            .is_err()
        );
        assert!(apply_drops(&mut held, VocoderMode::Dstar, &[1], Subst::Mute, &[0; 8]).is_err());
        assert!(
            apply_drops(
                &mut held[..10],
                VocoderMode::Dstar,
                &[],
                Subst::Mute,
                &NULL_AMBE_FRAME
            )
            .is_err()
        );
    }

    #[test]
    fn gray_coding_round_trips_and_bits_are_msb_first() {
        // Every field here is at most seven bits.
        for v in 0..128u32 {
            assert_eq!(from_gray(to_gray(v)), v, "{v}");
        }
        let mut f = [0u8; 8];
        write_bits(&mut f, 2, 7, 0b101_0101);
        assert_eq!(read_bits(&f, 2, 7), 0b101_0101);
        // Bit 0 of the stream is the MSB of byte 0, as the codec2 crate
        // packs it.
        let mut g = [0u8; 2];
        write_bits(&mut g, 0, 1, 1);
        assert_eq!(g[0], 0x80);
        // Writing a zero clears the bit again.
        write_bits(&mut g, 0, 1, 0);
        assert_eq!(g[0], 0x00);
    }

    #[test]
    fn codec2_erasure_fill_interpolates_pitch_and_energy() {
        let mode = VocoderMode::Codec2_3200;
        let fb = mode.frame_bytes();
        let (wo, e) = C2_3200.points[0];
        let mut bytes = vec![0u8; 5 * fb];
        set_field(&mut bytes[0..fb], wo, 10);
        set_field(&mut bytes[0..fb], e, 4);
        write_bits(&mut bytes[0..fb], C2_3200.lsp.offset, 32, 0xDEAD_BEEF);
        set_field(&mut bytes[4 * fb..5 * fb], wo, 50);
        set_field(&mut bytes[4 * fb..5 * fb], e, 24);

        fill_erasures(&mut bytes, mode, &[1, 2, 3], &vec![0u8; fb]).unwrap();

        let field = |f: usize, b| get_field(&bytes[f * fb..(f + 1) * fb], b);
        let wos: Vec<u32> = (0..5).map(|f| field(f, wo)).collect();
        let es: Vec<u32> = (0..5).map(|f| field(f, e)).collect();
        // The good frames are untouched and the gap is a straight ramp
        // between them in both quantisers.
        assert_eq!(wos, vec![10, 20, 30, 40, 50], "pitch");
        assert_eq!(es, vec![4, 9, 14, 19, 24], "energy");
        // The envelope is held from the last good frame before the gap.
        assert_eq!(
            read_bits(&bytes[2 * fb..3 * fb], C2_3200.lsp.offset, 32),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn a_codec2_gap_at_the_end_decays_the_energy_and_holds_the_pitch() {
        let mode = VocoderMode::Codec2_3200;
        let fb = mode.frame_bytes();
        let (wo, e) = C2_3200.points[0];
        let mut bytes = vec![0u8; 4 * fb];
        set_field(&mut bytes[0..fb], wo, 40);
        set_field(&mut bytes[0..fb], e, 20);

        fill_erasures(&mut bytes, mode, &[1, 2, 3], &vec![0u8; fb]).unwrap();

        let field = |f: usize, b| get_field(&bytes[f * fb..(f + 1) * fb], b);
        let es: Vec<u32> = (0..4).map(|f| field(f, e)).collect();
        assert_eq!(es, vec![20, 18, 16, 14], "energy falls toward silence");
        assert!(
            (0..4).all(|f| field(f, wo) == 40),
            "pitch is held, not swept"
        );
    }

    #[test]
    fn a_fully_erased_codec2_utterance_falls_back_to_mute() {
        let mode = VocoderMode::Codec2_3200;
        let fb = mode.frame_bytes();
        let mut bytes = vec![0xAA; 3 * fb];
        let mute = vec![7u8; fb];
        fill_erasures(&mut bytes, mode, &[0, 1, 2], &mute).unwrap();
        assert!(bytes.chunks(fb).all(|f| f == mute));
    }

    #[test]
    fn erasure_fill_is_codec2_only_and_checks_its_arguments() {
        let fb = VocoderMode::Codec2_1600.frame_bytes();
        let mut ok = vec![0u8; 2 * fb];
        // Both Codec 2 modes have a layout; the AMBE modes do not.
        assert!(fill_erasures(&mut ok, VocoderMode::Codec2_1600, &[0], &vec![0u8; fb]).is_ok());
        let err = fill_erasures(&mut ok, VocoderMode::Dstar, &[0], &NULL_AMBE_FRAME).unwrap_err();
        assert!(err.contains("no parameter-domain concealment"), "{err}");
        // A position past the end, and a mute frame of the wrong length.
        assert!(fill_erasures(&mut ok, VocoderMode::Codec2_1600, &[9], &vec![0u8; fb]).is_err());
        assert!(fill_erasures(&mut ok, VocoderMode::Codec2_1600, &[0], &[0; 3]).is_err());
    }

    /// A 7-byte YSF frame with the padding bits clear, so a whole-slice
    /// comparison is a comparison of the 49 voice bits.
    fn ysf_frames(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 7);
        for i in 0..n {
            let b = i as u8;
            v.extend_from_slice(&[0xAB ^ b, 0xCD, 0xEF ^ b, 0x12, 0x34 ^ b, 0x56, 0x80]);
        }
        v
    }

    fn differing_bits(a: &[u8], b: &[u8], from: usize, to: usize) -> usize {
        (from..to)
            .filter(|&i| read_bit(a, i) != read_bit(b, i))
            .count()
    }

    #[test]
    fn ysf_ber_at_zero_rate_changes_nothing() {
        for vd in [YsfVd::Vd1, YsfVd::Vd2] {
            let orig = ysf_frames(20);
            let mut x = orig.clone();
            let touched =
                apply_ysf_ber(&mut x, VocoderMode::YsfDmr, vd, 0.0, &mut Rng::new(1)).unwrap();
            assert_eq!(x, orig, "{vd:?}");
            assert!(touched.is_empty(), "{vd:?}");
        }
    }

    #[test]
    fn ysf_ber_is_deterministic_and_ysf_only() {
        let mut a = ysf_frames(50);
        let mut b = a.clone();
        let ta = apply_ysf_ber(
            &mut a,
            VocoderMode::YsfDmr,
            YsfVd::Vd1,
            0.05,
            &mut Rng::new(7),
        );
        let tb = apply_ysf_ber(
            &mut b,
            VocoderMode::YsfDmr,
            YsfVd::Vd1,
            0.05,
            &mut Rng::new(7),
        );
        assert_eq!((a, ta.unwrap()), (b, tb.unwrap()), "same seed, same result");

        let mut d = vec![0u8; 9 * 3];
        let err = apply_ysf_ber(
            &mut d,
            VocoderMode::Dstar,
            YsfVd::Vd1,
            0.01,
            &mut Rng::new(1),
        )
        .unwrap_err();
        assert!(err.contains("ysf-dmr only"), "{err}");
    }

    /// Mode 1 protects twenty-four of the forty-nine bits and leaves
    /// twenty-five bare, so at a rate the Golay words shrug off, the
    /// damage lands almost entirely on the unprotected tail.
    #[test]
    fn ysf_vd1_damage_lands_on_the_unprotected_bits() {
        let orig = ysf_frames(2_000);
        let mut x = orig.clone();
        apply_ysf_ber(
            &mut x,
            VocoderMode::YsfDmr,
            YsfVd::Vd1,
            0.02,
            &mut Rng::new(3),
        )
        .unwrap();
        let mut protected = 0usize;
        let mut bare = 0usize;
        for f in 0..2_000 {
            let (o, n) = (&orig[f * 7..f * 7 + 7], &x[f * 7..f * 7 + 7]);
            // Skip muted frames: they are the other failure mode.
            if n == YSF_DMR_MUTE_FRAME {
                continue;
            }
            protected += differing_bits(o, n, 0, 24);
            bare += differing_bits(o, n, 24, 49);
        }
        assert!(
            bare > protected * 4,
            "bare {bare} should dwarf protected {protected}"
        );
    }

    /// The first word carries the whitening key, so a word too damaged to
    /// repair takes the entire frame with it.
    #[test]
    fn ysf_vd1_mutes_when_the_first_word_is_beyond_repair() {
        let mut x = ysf_frames(500);
        apply_ysf_ber(
            &mut x,
            VocoderMode::YsfDmr,
            YsfVd::Vd1,
            0.15,
            &mut Rng::new(11),
        )
        .unwrap();
        let mutes = x
            .as_chunks::<7>()
            .0
            .iter()
            .filter(|f| **f == YSF_DMR_MUTE_FRAME)
            .count();
        assert!(mutes > 0, "no frame was lost at a 15 % bit error rate");
    }

    /// Mode 2 votes three copies of its first twenty-seven bits, so those
    /// survive a rate that corrupts the twenty-two bare ones.
    #[test]
    fn ysf_vd2_voting_protects_the_first_twenty_seven_bits() {
        let orig = ysf_frames(2_000);
        let mut x = orig.clone();
        apply_ysf_ber(
            &mut x,
            VocoderMode::YsfDmr,
            YsfVd::Vd2,
            0.02,
            &mut Rng::new(5),
        )
        .unwrap();
        let mut voted = 0usize;
        let mut bare = 0usize;
        for f in 0..2_000 {
            let (o, n) = (&orig[f * 7..f * 7 + 7], &x[f * 7..f * 7 + 7]);
            voted += differing_bits(o, n, 0, 27);
            bare += differing_bits(o, n, 27, 49);
        }
        assert!(bare > voted * 4, "bare {bare} should dwarf voted {voted}");
    }

    #[test]
    fn ber_flips_about_the_rate_only_in_valid_bits() {
        let fb = 7;
        let frames = 4_000;
        let orig = vec![0u8; frames * fb];
        let mut x = orig.clone();
        let touched = apply_ber(&mut x, fb, 49, 0.01, &mut Rng::new(3)).unwrap();
        let flipped: u32 = x.iter().map(|b| b.count_ones()).sum();
        let rate = flipped as f32 / (frames * 49) as f32;
        assert!((rate - 0.01).abs() < 0.0015, "{rate}");
        // Padding bits (the low seven of the seventh byte) never flip.
        assert!(x.chunks(fb).all(|f| f[6] & 0x7F == 0));
        assert!(!touched.is_empty() && touched.windows(2).all(|w| w[0] < w[1]));
        for (f, frame) in x.chunks(fb).enumerate() {
            let hit = frame.iter().any(|&b| b != 0);
            assert_eq!(hit, touched.contains(&(f as u32)), "frame {f}");
        }
        let mut y = orig.clone();
        let again = apply_ber(&mut y, fb, 49, 0.01, &mut Rng::new(3)).unwrap();
        assert_eq!((x, touched), (y, again), "deterministic");
        let mut z = orig;
        assert!(
            apply_ber(&mut z, fb, 49, 0.0, &mut Rng::new(1))
                .unwrap()
                .is_empty()
        );
        assert!(apply_ber(&mut z[..5], fb, 49, 0.5, &mut Rng::new(1)).is_err());
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
mod ysf_stats {
    use super::*;

    // The on-air placement of mode 1's three words, needed only to study
    // bursts. Production code does not use it: with independent bit flips
    // the position of a bit cannot change its error probability.
    fn a_pos(i: usize) -> usize {
        if i < 18 { 4 * i } else { 4 * (i - 18) + 1 }
    }
    fn b_pos(i: usize) -> usize {
        if i < 12 { 4 * i + 25 } else { 4 * (i - 12) + 2 }
    }
    fn c_pos(i: usize) -> usize {
        if i < 7 { 4 * i + 46 } else { 4 * (i - 7) + 3 }
    }

    /// Put one frame on the air, corrupt `len` contiguous bits starting at
    /// `start`, and decode it back. Returns the 49 voice bits, or `None`
    /// when the frame was lost.
    fn burst_once(voice: [u8; 7], start: usize, len: usize) -> Option<[u8; 7]> {
        let (mut a, mut b, mut c) = (0u16, 0u16, 0u32);
        for i in 0..12 {
            a = (a << 1) | u16::from(read_bit(&voice, i));
            b = (b << 1) | u16::from(read_bit(&voice, 12 + i));
        }
        for i in 0..25 {
            c = (c << 1) | u32::from(read_bit(&voice, 24 + i));
        }
        let a24 = crate::fec::golay24_encode(a);
        let b23 = (crate::fec::golay24_encode(b) >> 1) ^ (crate::fec::whiten(a) >> 1);

        // Lay the three words into the 72-bit block, flip the burst, read
        // them back out.
        let mut block = [false; 72];
        for i in 0..24 {
            block[a_pos(i)] = (a24 >> (23 - i)) & 1 == 1;
        }
        for i in 0..23 {
            block[b_pos(i)] = (b23 >> (22 - i)) & 1 == 1;
        }
        for i in 0..25 {
            block[c_pos(i)] = (c >> (24 - i)) & 1 == 1;
        }
        for bit in &mut block[start..(start + len).min(72)] {
            *bit = !*bit;
        }

        let mut a24r = 0u32;
        let mut b23r = 0u32;
        let mut c25r = 0u32;
        for i in 0..24 {
            a24r = (a24r << 1) | u32::from(block[a_pos(i)]);
        }
        for i in 0..23 {
            b23r = (b23r << 1) | u32::from(block[b_pos(i)]);
        }
        for i in 0..25 {
            c25r = (c25r << 1) | u32::from(block[c_pos(i)]);
        }

        let u0 = crate::fec::golay24_decode(a24r)?;
        let u1 = crate::fec::golay23_decode(b23r ^ (crate::fec::whiten(u0) >> 1));
        let mut out = [0u8; 7];
        for i in 0..12 {
            write_bit(&mut out, i, (u0 >> (11 - i)) & 1 == 1);
            write_bit(&mut out, 12 + i, (u1 >> (11 - i)) & 1 == 1);
        }
        for i in 0..25 {
            write_bit(&mut out, 24 + i, (c25r >> (24 - i)) & 1 == 1);
        }
        Some(out)
    }

    /// A burst of four contiguous bits is survivable; six is not. The three
    /// words are interleaved `aabc` at stride four, so the Golay (24, 12)
    /// word collects two bits of every four and runs out of correction at
    /// six. Independent errors at a realistic rate almost never do this,
    /// which is why bursts are a separate failure mode and not a detail.
    #[test]
    fn a_six_bit_burst_loses_a_mode_one_frame() {
        let voice = [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x80];
        let mut lost4 = 0;
        let mut lost6 = 0;
        for start in 0..=(72 - 6) {
            if burst_once(voice, start, 4).is_none() {
                lost4 += 1;
            }
            if burst_once(voice, start, 6).is_none() {
                lost6 += 1;
            }
        }
        assert_eq!(lost4, 0, "a four-bit burst should always be corrected");
        assert!(lost6 > 0, "a six-bit burst should lose frames somewhere");
    }

    /// Measurement: how often each burst length costs the frame.
    #[test]
    #[ignore = "measurement, run with --ignored"]
    fn ysf_burst_survival_by_length() {
        let voice = [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x80];
        println!(" burst bits   ms at 3600 bit/s   frames lost of 72 starts");
        for len in [2usize, 3, 4, 5, 6, 8, 12, 16, 24] {
            let starts = 72 - len + 1;
            let lost = (0..starts)
                .filter(|&s| burst_once(voice, s, len).is_none())
                .count();
            println!(
                " {len:>10}   {:>16.2}   {lost:>6} of {starts} ({:>5.1}%)",
                len as f64 * 1000.0 / 3600.0,
                100.0 * lost as f64 / starts as f64
            );
        }
    }

    /// Not an assertion, a measurement: what each rate actually does, so a
    /// default can be chosen from numbers instead of taste.
    #[test]
    #[ignore = "measurement, run with --ignored"]
    fn ysf_error_statistics_by_rate() {
        let n = 4_000usize;
        println!("rate   vd1: frames lost  voice bits wrong/frame | vd2: bits wrong/frame");
        for &rate in &[0.001f32, 0.005, 0.01, 0.02, 0.05, 0.10] {
            let orig: Vec<u8> = (0..n)
                .flat_map(|i| {
                    let b = i as u8;
                    [0xAB ^ b, 0xCD, 0xEF ^ b, 0x12, 0x34 ^ b, 0x56, 0x80]
                })
                .collect();
            let mut a = orig.clone();
            apply_ysf_ber(
                &mut a,
                VocoderMode::YsfDmr,
                YsfVd::Vd1,
                rate,
                &mut Rng::new(1),
            )
            .unwrap();
            let mut lost = 0usize;
            let mut wrong = 0usize;
            for f in 0..n {
                let (o, x) = (&orig[f * 7..f * 7 + 7], &a[f * 7..f * 7 + 7]);
                if x == YSF_DMR_MUTE_FRAME {
                    lost += 1;
                } else {
                    wrong += (0..49)
                        .filter(|&i| read_bit(o, i) != read_bit(x, i))
                        .count();
                }
            }
            let mut b = orig.clone();
            apply_ysf_ber(
                &mut b,
                VocoderMode::YsfDmr,
                YsfVd::Vd2,
                rate,
                &mut Rng::new(1),
            )
            .unwrap();
            let wrong2: usize = (0..n)
                .map(|f| {
                    let (o, x) = (&orig[f * 7..f * 7 + 7], &b[f * 7..f * 7 + 7]);
                    (0..49)
                        .filter(|&i| read_bit(o, i) != read_bit(x, i))
                        .count()
                })
                .sum();
            let kept = (n - lost).max(1);
            println!(
                "{rate:<6} {:>6} ({:>5.2}%)      {:>6.3}            |        {:>6.3}",
                lost,
                100.0 * lost as f64 / n as f64,
                wrong as f64 / kept as f64,
                wrong2 as f64 / n as f64
            );
        }
    }
}
