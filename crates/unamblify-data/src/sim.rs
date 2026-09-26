// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! A deterministic AMBE-3000 stand-in behind `ambe_thumbdv::Transport`,
//! for `--dry-run` and tests. It speaks the control protocol (reset →
//! Ready, PRODID, VERSTRING, RATEP → status, …), answers `speech_in` with
//! a channel frame whose bit count follows the RATEP it was given, and
//! decodes a frame it has seen back to the PCM that produced it, delayed
//! by [`SimTransport::lag`] samples so the canary's lag measurement has
//! something to find. Unknown frames decode to silence.
//!
//! Faults can be injected ([`Faults`]) to drive the failure paths: a
//! dropped response (on either direction: timeout → re-init → retry), a
//! wrong bit count (rate lost) and corrupted frames (canary mismatch).
//!
//! The interleaved runner ([`crate::pipeline::round_trip`]) has requests
//! of both kinds in the chip at once, so the sim models the two
//! properties that runner depends on: **responses are FIFO within a kind
//! and arbitrarily interleaved between kinds** ([`ReplyBias`] reorders
//! the two streams against each other without ever reordering one of
//! them), and requests are **bounded per kind** — the sim counts what it
//! owes on each side, records the peaks
//! ([`SimTransport::peak_encodes_in_flight`],
//! [`SimTransport::peak_decodes_in_flight`]) and, with
//! [`SimTransport::max_in_flight`] set, panics the moment a runner
//! exceeds the depth the real chip tolerates.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::Duration;

use ambe_thumbdv::Transport;
use ambe_thumbdv::packet::{Deframer, RawPacket};
use sha2::{Digest, Sha256};
use unamblify::FRAME_SAMPLES;

use crate::chip::dvsi_packet;

/// Fault injection knobs. Each `*_at` fires once, at that encode index
/// (counted from the sim's creation), then clears itself.
#[derive(Debug, Clone, Default)]
pub struct Faults {
    /// Swallow the response to encode request number `n`.
    pub drop_encode_response_at: Option<u64>,
    /// Swallow the response to decode request number `n`.
    pub drop_decode_response_at: Option<u64>,
    /// Answer encode request number `n` with the *other* bit count.
    pub wrong_bits_at: Option<u64>,
    /// From encode number `n` on, flip a bit in every frame produced.
    pub corrupt_frames_from: Option<u64>,
    /// Answer the next reset with silence (init fails), then clear.
    pub ignore_next_reset: bool,
    /// Make the encoder stateful like the real chip: every frame also
    /// hashes the frame encoded before it (reset clears that memory), so
    /// the same PCM yields different bytes after different history. What
    /// the canary discipline has to survive.
    pub history: bool,
}

/// Which direction a reply belongs to. Control replies are stop-and-wait
/// and never queue behind voice traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Control,
    Encode,
    Decode,
}

/// How the sim interleaves the two directions' replies against each
/// other. Within a direction replies are always FIFO — that is the one
/// property the chip guarantees and the runner relies on; *between*
/// directions the order is the chip's business, and a runner that pairs
/// responses by arrival order rather than by packet type must fail under
/// at least one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplyBias {
    /// Answer in the order the requests arrived (what a lightly loaded
    /// chip does).
    #[default]
    Submission,
    /// Flush every ready `Channel` reply before any ready `Speech` one.
    EncodeFirst,
    /// Flush every ready `Speech` reply before any ready `Channel` one.
    DecodeFirst,
}

/// The simulated chip.
pub struct SimTransport {
    deframer: Deframer,
    inbox: VecDeque<u8>,
    /// Voice replies produced but not yet handed to the reader, in
    /// submission order; [`ReplyBias`] decides how they reach `inbox`.
    staged: VecDeque<(Side, Vec<u8>)>,
    /// Bytes of each reply in `inbox` still to be delivered, in order, so
    /// a side's debt clears exactly when its reply's last byte leaves.
    pending: VecDeque<(Side, usize)>,
    enc_in_flight: usize,
    dec_in_flight: usize,
    rate_bits: u8,
    codebook: HashMap<Vec<u8>, [i16; FRAME_SAMPLES]>,
    delay: VecDeque<i16>,
    /// Digest of the previous frame's PCM since the last reset.
    prev: Option<[u8; 32]>,
    /// Decode delay, samples (what the canary's lag measurement finds).
    pub lag: usize,
    /// PRODID reply.
    pub prodid: String,
    /// VERSTRING reply.
    pub version: String,
    /// Resets seen.
    pub resets: u64,
    /// Speech-in requests seen.
    pub encodes: u64,
    /// Channel-in requests seen.
    pub decodes: u64,
    /// Most encode requests the reader ever had outstanding at once.
    pub peak_encodes_in_flight: usize,
    /// Most decode requests the reader ever had outstanding at once.
    pub peak_decodes_in_flight: usize,
    /// How the two directions' replies are ordered against each other.
    pub reply_bias: ReplyBias,
    /// Test guard: panic if either direction ever exceeds this many
    /// outstanding requests. `None` (the default, and what `--dry-run`
    /// uses) never panics.
    pub max_in_flight: Option<usize>,
    /// Fault injection.
    pub faults: Faults,
}

impl Default for SimTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl SimTransport {
    /// A fresh sim: lag 42, `AMBE3000F`, the project stick's version string.
    #[must_use]
    pub fn new() -> Self {
        Self {
            deframer: Deframer::new(),
            inbox: VecDeque::new(),
            staged: VecDeque::new(),
            pending: VecDeque::new(),
            enc_in_flight: 0,
            dec_in_flight: 0,
            rate_bits: 72,
            codebook: HashMap::new(),
            delay: VecDeque::new(),
            prev: None,
            lag: 42,
            prodid: "AMBE3000F".to_owned(),
            version: "V121.E100.XXXX.C110.G514.R014.A0030608.C0020208".to_owned(),
            resets: 0,
            encodes: 0,
            decodes: 0,
            peak_encodes_in_flight: 0,
            peak_decodes_in_flight: 0,
            reply_bias: ReplyBias::Submission,
            max_in_flight: None,
            faults: Faults::default(),
        }
    }

    /// With fault injection.
    #[must_use]
    pub fn with_faults(mut self, faults: Faults) -> Self {
        self.faults = faults;
        self
    }

    /// With a reply order between the two directions.
    #[must_use]
    pub fn with_reply_bias(mut self, bias: ReplyBias) -> Self {
        self.reply_bias = bias;
        self
    }

    /// With the in-flight guard armed: the sim panics if the reader ever
    /// holds more than `depth` requests of either kind.
    #[must_use]
    pub fn with_max_in_flight(mut self, depth: usize) -> Self {
        self.max_in_flight = Some(depth);
        self
    }

    /// A control reply: straight into the reader's stream, ahead of
    /// nothing (control transactions are stop-and-wait).
    fn reply(&mut self, bytes: &[u8]) {
        self.pending.push_back((Side::Control, bytes.len()));
        self.inbox.extend(bytes);
    }

    /// A voice reply: staged, so [`ReplyBias`] can order it against the
    /// other direction's before it reaches the reader.
    fn stage(&mut self, side: Side, bytes: Vec<u8>) {
        self.staged.push_back((side, bytes));
    }

    /// Move every staged voice reply into the reader's stream, in the
    /// bias's order. FIFO within a direction is preserved in every case.
    fn flush_staged(&mut self) {
        if self.staged.is_empty() {
            return;
        }
        let first = match self.reply_bias {
            ReplyBias::Submission => None,
            ReplyBias::EncodeFirst => Some(Side::Encode),
            ReplyBias::DecodeFirst => Some(Side::Decode),
        };
        let staged: Vec<(Side, Vec<u8>)> = self.staged.drain(..).collect();
        if let Some(first) = first {
            for (side, bytes) in staged.iter().filter(|(s, _)| *s == first) {
                self.pending.push_back((*side, bytes.len()));
                self.inbox.extend(bytes);
            }
            for (side, bytes) in staged.iter().filter(|(s, _)| *s != first) {
                self.pending.push_back((*side, bytes.len()));
                self.inbox.extend(bytes);
            }
        } else {
            for (side, bytes) in &staged {
                self.pending.push_back((*side, bytes.len()));
                self.inbox.extend(bytes);
            }
        }
    }

    /// Account `n` bytes leaving `inbox`: a direction's debt clears when
    /// the last byte of its reply is delivered.
    fn deliver(&mut self, mut n: usize) {
        while n > 0 {
            let Some((side, left)) = self.pending.front_mut() else {
                return;
            };
            let side = *side;
            if *left > n {
                *left -= n;
                return;
            }
            n -= *left;
            self.pending.pop_front();
            match side {
                Side::Encode => self.enc_in_flight = self.enc_in_flight.saturating_sub(1),
                Side::Decode => self.dec_in_flight = self.dec_in_flight.saturating_sub(1),
                Side::Control => {}
            }
        }
    }

    /// One more request outstanding on `side`; record the peak and, with
    /// the guard armed, refuse to model a depth the real chip cannot take
    /// (4 + 4 overflows its input queue and a response never comes back).
    fn took(&mut self, side: Side) {
        let (n, what) = match side {
            Side::Encode => {
                self.enc_in_flight += 1;
                self.peak_encodes_in_flight = self.peak_encodes_in_flight.max(self.enc_in_flight);
                (self.enc_in_flight, "encode")
            }
            Side::Decode => {
                self.dec_in_flight += 1;
                self.peak_decodes_in_flight = self.peak_decodes_in_flight.max(self.dec_in_flight);
                (self.dec_in_flight, "decode")
            }
            Side::Control => return,
        };
        assert!(
            self.max_in_flight.is_none_or(|max| n <= max),
            "sim: {n} {what} requests in flight, more than the {} this chip tolerates",
            self.max_in_flight.unwrap_or(0)
        );
    }

    fn status(&mut self, field: u8) {
        self.reply(&dvsi_packet(0, &[field, 0x00]));
    }

    fn handle(&mut self, pkt: &RawPacket) {
        match pkt.ptype {
            0 => self.handle_control(&pkt.payload),
            1 => self.handle_channel(&pkt.payload),
            2 => self.handle_speech(&pkt.payload),
            _ => {}
        }
    }

    fn handle_control(&mut self, payload: &[u8]) {
        let Some(&field) = payload.first() else {
            return;
        };
        match field {
            0x33 => {
                self.resets += 1;
                self.delay.clear();
                self.prev = None;
                self.deframer = Deframer::new();
                // The chip throws away everything it was working on, so a
                // re-init after a failed utterance never has the old
                // utterance's replies arriving behind the Ready.
                self.inbox.clear();
                self.staged.clear();
                self.pending.clear();
                self.enc_in_flight = 0;
                self.dec_in_flight = 0;
                if std::mem::take(&mut self.faults.ignore_next_reset) {
                    return;
                }
                self.reply(&dvsi_packet(0, &[0x39]));
            }
            0x30 => {
                let mut f = vec![0x30];
                f.extend_from_slice(self.prodid.as_bytes());
                f.push(0);
                self.reply(&dvsi_packet(0, &f));
            }
            0x31 => {
                let mut f = vec![0x31];
                f.extend_from_slice(self.version.as_bytes());
                f.push(0);
                self.reply(&dvsi_packet(0, &f));
            }
            0x0A => {
                // RATEP: the DN word (no FEC, `70 31` checksum) is 49 bits;
                // the D-STAR word (and the unused DMR one) is 72.
                self.rate_bits = if payload.len() >= 13 && payload[5] == 0x00 && payload[11] == 0x70
                {
                    49
                } else {
                    72
                };
                self.status(0x0A);
            }
            // INIT, ECMODE, DCMODE, GAIN and anything else: status 0.
            _ => self.status(field),
        }
    }

    fn frame_bytes(&self) -> usize {
        usize::from(self.rate_bits).div_ceil(8)
    }

    fn handle_speech(&mut self, payload: &[u8]) {
        let n = self.encodes;
        self.encodes += 1;
        if payload.len() < 2 + FRAME_SAMPLES * 2 || payload[0] != 0x00 || payload[1] != 0xA0 {
            return;
        }
        self.took(Side::Encode);
        let mut pcm = [0i16; FRAME_SAMPLES];
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = i16::from_be_bytes([payload[2 + 2 * i], payload[3 + 2 * i]]);
        }
        let mut frame = self.encode_frame(&pcm);
        if self.faults.history {
            let mut h = Sha256::new();
            for s in &pcm {
                h.update(s.to_le_bytes());
            }
            self.prev = Some(h.finalize().into());
        }
        if self
            .faults
            .corrupt_frames_from
            .is_some_and(|from| n >= from)
        {
            frame[0] ^= 0x01;
        }
        self.codebook.insert(frame.clone(), pcm);
        if self.faults.drop_encode_response_at == Some(n) {
            self.faults.drop_encode_response_at = None;
            return;
        }
        let mut bits = self.rate_bits;
        if self.faults.wrong_bits_at == Some(n) {
            self.faults.wrong_bits_at = None;
            bits = if bits == 72 { 49 } else { 72 };
            frame.resize(usize::from(bits).div_ceil(8), 0);
        }
        let mut f = vec![0x01, bits];
        f.extend_from_slice(&frame);
        let pkt = dvsi_packet(1, &f);
        self.stage(Side::Encode, pkt);
    }

    fn encode_frame(&self, pcm: &[i16; FRAME_SAMPLES]) -> Vec<u8> {
        let mut h = Sha256::new();
        h.update([self.rate_bits]);
        if let Some(prev) = &self.prev {
            h.update(prev);
        }
        for s in pcm {
            h.update(s.to_le_bytes());
        }
        let digest = h.finalize();
        let n = self.frame_bytes();
        let mut frame = digest[..n].to_vec();
        let spare = self.rate_bits % 8;
        if spare != 0 {
            frame[n - 1] &= (1u8 << spare) - 1;
        }
        frame
    }

    fn handle_channel(&mut self, payload: &[u8]) {
        let n = self.decodes;
        self.decodes += 1;
        if payload.len() < 2 || payload[0] != 0x01 {
            return;
        }
        self.took(Side::Decode);
        let bytes = usize::from(payload[1]).div_ceil(8);
        let frame = payload.get(2..2 + bytes).unwrap_or(&[]).to_vec();
        let pcm = self
            .codebook
            .get(&frame)
            .copied()
            .unwrap_or([0i16; FRAME_SAMPLES]);
        if self.delay.is_empty() && self.lag > 0 {
            self.delay.extend(std::iter::repeat_n(0i16, self.lag));
        }
        self.delay.extend(pcm);
        let mut f = vec![0x00, 0xA0];
        for _ in 0..FRAME_SAMPLES {
            let s = self.delay.pop_front().unwrap_or(0);
            f.extend_from_slice(&s.to_be_bytes());
        }
        if self.faults.drop_decode_response_at == Some(n) {
            self.faults.drop_decode_response_at = None;
            return;
        }
        let pkt = dvsi_packet(2, &f);
        self.stage(Side::Decode, pkt);
    }
}

impl Transport for SimTransport {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.deframer.push(bytes);
        while let Some(pkt) = self.deframer.next_packet() {
            self.handle(&pkt);
        }
        Ok(())
    }

    fn recv_some(&mut self, buf: &mut [u8], timeout: Duration) -> io::Result<usize> {
        if self.inbox.is_empty() {
            self.flush_staged();
        }
        if self.inbox.is_empty() {
            std::thread::sleep(timeout.min(Duration::from_millis(1)));
            return Ok(0);
        }
        let n = buf.len().min(self.inbox.len());
        for b in &mut buf[..n] {
            *b = self.inbox.pop_front().unwrap_or(0);
        }
        self.deliver(n);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::Chip;
    use crate::pipeline::{decode, encode};
    use crate::testutil::sine;
    use crate::vocoder::pad_frames;
    use unamblify::VocoderMode;

    /// `encode` over f32 PCM, padded to the mode's frames.
    fn enc(chip: &mut Chip<SimTransport>, mode: VocoderMode, pcm: &[f32]) -> Vec<u8> {
        encode(chip.transport_mut(), mode, &pad_frames(pcm, mode)).unwrap()
    }

    #[test]
    fn sim_inits_encodes_and_decodes_with_its_lag() {
        for mode in VocoderMode::ALL.into_iter().filter(|m| !m.is_software()) {
            let mut chip = Chip::init(SimTransport::new(), mode).unwrap();
            assert_eq!(chip.info().prodid, "AMBE3000F");
            let pcm = sine(300.0, 8_000, 1_600, 0.3);
            let frames = enc(&mut chip, mode, &pcm);
            assert_eq!(frames.len(), 10 * mode.frame_bytes());
            let out = decode(chip.transport_mut(), mode, &frames).unwrap();
            assert_eq!(out.len(), 1_600);
            let out_f: Vec<f32> = out.iter().map(|&s| f32::from(s) / 32_767.0).collect();
            assert_eq!(unamblify_audio::xcorr_lag(&pcm, &out_f, 400), 42);
            let sim = chip.into_transport();
            assert_eq!(sim.resets, 1);
            assert_eq!(sim.encodes, 10);
            assert_eq!(sim.decodes, 10);
        }
    }

    #[test]
    fn sim_is_deterministic_across_instances() {
        let pcm = sine(440.0, 8_000, 800, 0.2);
        let mut a = Chip::init(SimTransport::new(), VocoderMode::Dstar).unwrap();
        let mut b = Chip::init(SimTransport::new(), VocoderMode::Dstar).unwrap();
        let fa = enc(&mut a, VocoderMode::Dstar, &pcm);
        let fb = enc(&mut b, VocoderMode::Dstar, &pcm);
        assert_eq!(fa, fb);
        let mut c = Chip::init(SimTransport::new(), VocoderMode::YsfDmr).unwrap();
        let fc = enc(&mut c, VocoderMode::YsfDmr, &pcm);
        assert_ne!(fa[..7], fc[..7]);
        // 49 bits: the top seven bits of the seventh byte are zero.
        assert!(fc.chunks(7).all(|f| f[6] & 0xFE == 0));
    }

    #[test]
    fn history_makes_the_same_pcm_encode_differently_after_different_context() {
        let pcm = sine(440.0, 8_000, 320, 0.2);
        let other = sine(200.0, 8_000, 160, 0.2);
        let faults = Faults {
            history: true,
            ..Faults::default()
        };
        let mut a = Chip::init(
            SimTransport::new().with_faults(faults.clone()),
            VocoderMode::Dstar,
        )
        .unwrap();
        let fresh = enc(&mut a, VocoderMode::Dstar, &pcm);
        let _ = enc(&mut a, VocoderMode::Dstar, &other);
        let after = enc(&mut a, VocoderMode::Dstar, &pcm);
        assert_ne!(fresh[..9], after[..9], "first frame sees the history");
        assert_eq!(fresh[9..], after[9..], "later frames only see the clip");
        // A reset clears the memory: the same preamble gives the same bytes.
        a.reinit().unwrap();
        let again = enc(&mut a, VocoderMode::Dstar, &pcm);
        assert_eq!(fresh, again);
        let mut b = Chip::init(SimTransport::new(), VocoderMode::Dstar).unwrap();
        let _ = enc(&mut b, VocoderMode::Dstar, &other);
        let stateless = enc(&mut b, VocoderMode::Dstar, &pcm);
        let mut c = Chip::init(SimTransport::new(), VocoderMode::Dstar).unwrap();
        assert_eq!(stateless, enc(&mut c, VocoderMode::Dstar, &pcm));
    }
}
