// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The runners that drive the chip (spec §3).
//!
//! [`run`] is the one-direction pipelined runner: up to [`MAX_IN_FLIGHT`]
//! requests outstanding, `recv_some(1024, 5 ms)` polls, every deframed
//! packet delivered and matched FIFO to the oldest outstanding request,
//! the channel bit count checked against the mode, and an age-out of
//! [`REPLY_TIMEOUT`] since the oldest outstanding request that is an
//! error — never a substitution. Frames out must equal frames in. It is
//! what `encode` and `decode` use, and the software Codec 2 path never
//! comes near either.
//!
//! [`round_trip`] is the interleaved runner: one utterance's encode and
//! decode at the same time, [`INTERLEAVE_DEPTH`] of each in flight, every
//! response routed by packet type (`Channel` answers an encode, `Speech`
//! answers a decode) and matched FIFO *within its kind* — astar's
//! `AmbeStream` worker does exactly this with its two `PipelineSide`s.
//! The UART is full duplex, so the two directions overlap; the two
//! sides age out and count independently, and every failure is still the
//! whole utterance's failure.
//!
//! Measured on the project stick at 460 800 baud (300 pairs per point,
//! 2026-09-11): one direction pipelined four deep costs 7.12 ms
//! (encode) and 7.21 ms (decode) per frame — 7.07 ms of that is the
//! 326-byte speech packet's wire time, and 460 800 is the chip's maximum
//! baud, so a single direction cannot go faster. The two passes back to
//! back are 14.28 ms per audio frame; interleaved 3 + 3 they are
//! **8.27 ms, 1.73× faster**.

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use ambe_thumbdv::packet::{parse_response, speech_in};
use ambe_thumbdv::{Deframer, Response, Transport};
use unamblify::VocoderMode;

use crate::chip::channel_in_mode;

/// Requests written ahead of the oldest unanswered one, in one direction
/// ([`run`]).
pub const MAX_IN_FLIGHT: usize = 4;

/// Requests of *each* kind the interleaved runner ([`round_trip`]) keeps
/// outstanding, so at most `2 * INTERLEAVE_DEPTH` packets are ever in the
/// chip at once.
///
/// Measured on the project stick, 300 pairs per point: 3 encodes + 3
/// decodes in flight cost **8.27 ms per audio frame** against 14.28 ms
/// for the two passes back to back. 2 + 2 is 8.59 ms and every
/// asymmetric depth is worse (3/2 10.85, 2/3 10.96, 4/2 11.31, 2/4
/// 12.71). **4 + 4 stalls the chip**: its input queue overflows, a
/// response never arrives and the utterance ages out — which is why the
/// total in flight must never exceed 3 + 3, and why this is one constant
/// rather than a tunable.
pub const INTERLEAVE_DEPTH: usize = 3;

/// Poll length of each `recv_some`.
pub const READ_POLL: Duration = Duration::from_millis(5);

/// Age of the oldest outstanding request at which the run is an error.
pub const REPLY_TIMEOUT: Duration = Duration::from_millis(100);

/// Runner errors. Every one of them makes the utterance a redo.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Transport I/O.
    #[error("io: {0}")]
    Io(#[from] io::Error),
    /// The oldest outstanding request aged past [`REPLY_TIMEOUT`].
    #[error(
        "timeout: request {index} unanswered for {age_ms} ms ({outstanding} outstanding, \
         {delivered} delivered)"
    )]
    Timeout {
        /// Index of the request that aged out.
        index: usize,
        /// Its age, ms.
        age_ms: u64,
        /// Requests outstanding at the time.
        outstanding: usize,
        /// Responses delivered so far.
        delivered: usize,
    },
    /// A channel response with the wrong bit count: the chip lost its rate.
    #[error("rate lost: expected {expected} channel bits, the chip sent {got}")]
    RateLost {
        /// The mode's bits.
        expected: u16,
        /// The chip's.
        got: u8,
    },
    /// A response of the wrong kind, or one with nothing outstanding.
    #[error("unexpected packet: {0}")]
    Unexpected(String),
    /// A packet that did not parse.
    #[error("parse: {0}")]
    Parse(String),
    /// Responses delivered ≠ requests sent.
    #[error("count mismatch: sent {sent}, received {received}")]
    CountMismatch {
        /// Requests.
        sent: usize,
        /// Responses.
        received: usize,
    },
}

/// What kind of response every request in a run expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// `Response::Channel` with exactly this many bits.
    Channel {
        /// `mode.channel_bits()`.
        bits: u16,
    },
    /// `Response::Speech`.
    Speech,
}

/// Run `requests` through `transport` pipelined and return one response
/// per request, in order.
pub fn run<T: Transport + ?Sized>(
    transport: &mut T,
    requests: &[Vec<u8>],
    expect: Expect,
) -> Result<Vec<Response>, PipelineError> {
    let mut outstanding: VecDeque<(usize, Instant)> = VecDeque::with_capacity(MAX_IN_FLIGHT);
    let mut out = Vec::with_capacity(requests.len());
    let mut deframer = Deframer::new();
    let mut next = 0usize;
    let mut buf = [0u8; 1024];

    while next < requests.len() || !outstanding.is_empty() {
        while outstanding.len() < MAX_IN_FLIGHT && next < requests.len() {
            transport.send(&requests[next])?;
            outstanding.push_back((next, Instant::now()));
            next += 1;
        }
        let n = transport.recv_some(&mut buf, READ_POLL)?;
        if n > 0 {
            deframer.push(&buf[..n]);
            while let Some(pkt) = deframer.next_packet() {
                let resp = parse_response(&pkt).map_err(|e| PipelineError::Parse(e.to_string()))?;
                match (expect, &resp) {
                    (Expect::Channel { bits }, Response::Channel { bits: got, .. }) => {
                        if u16::from(*got) != bits {
                            return Err(PipelineError::RateLost {
                                expected: bits,
                                got: *got,
                            });
                        }
                    }
                    (Expect::Speech, Response::Speech(_)) => {}
                    (_, other) => {
                        return Err(PipelineError::Unexpected(format!(
                            "expected {expect:?}, got {}",
                            describe(other)
                        )));
                    }
                }
                if outstanding.pop_front().is_none() {
                    return Err(PipelineError::Unexpected(
                        "a response arrived with nothing outstanding".to_owned(),
                    ));
                }
                out.push(resp);
            }
        }
        age_out(&outstanding, out.len())?;
    }
    if out.len() != requests.len() {
        return Err(PipelineError::CountMismatch {
            sent: requests.len(),
            received: out.len(),
        });
    }
    Ok(out)
}

/// The oldest outstanding request of one side, as an error if it has
/// waited longer than [`REPLY_TIMEOUT`]. Each side of the interleaved
/// runner is aged on its own clock: one direction's silence must not be
/// charged to the other, which may be answering perfectly.
fn age_out(side: &VecDeque<(usize, Instant)>, delivered: usize) -> Result<(), PipelineError> {
    if let Some(&(index, sent_at)) = side.front() {
        let age = sent_at.elapsed();
        if age > REPLY_TIMEOUT {
            return Err(PipelineError::Timeout {
                index,
                age_ms: u64::try_from(age.as_millis()).unwrap_or(u64::MAX),
                outstanding: side.len(),
                delivered,
            });
        }
    }
    Ok(())
}

fn describe(r: &Response) -> String {
    match r {
        Response::Ready => "Ready".to_owned(),
        Response::Status { field, status } => format!("Status(0x{field:02X}=0x{status:02X})"),
        Response::ProdId(s) => format!("ProdId({s})"),
        Response::Version(s) => format!("Version({s})"),
        Response::Config(c) => format!("Config({c:?})"),
        Response::Channel { bits, .. } => format!("Channel({bits} bits)"),
        Response::Speech(_) => "Speech".to_owned(),
    }
}

/// One `speech_in` per frame of `pcm8k`, which must already be padded to
/// whole frames of `mode.frame_samples()` (`vocoder::pad_frames`).
pub fn speech_requests(pcm8k: &[i16], mode: VocoderMode) -> Result<Vec<Vec<u8>>, PipelineError> {
    let fs = mode.frame_samples();
    if !pcm8k.len().is_multiple_of(fs) {
        return Err(PipelineError::Unexpected(format!(
            "{} samples is not a whole number of {fs}-sample frames",
            pcm8k.len()
        )));
    }
    Ok(pcm8k
        .chunks_exact(fs)
        .map(|chunk| {
            let mut pcm = [0i16; unamblify::FRAME_SAMPLES];
            pcm.copy_from_slice(chunk);
            speech_in(&pcm)
        })
        .collect())
}

/// Encode pass: `pcm8k` (padded to whole frames) → channel frames, as the
/// chip emitted them, `mode.frame_bytes()` per frame, concatenated.
pub fn encode<T: Transport + ?Sized>(
    transport: &mut T,
    mode: VocoderMode,
    pcm8k: &[i16],
) -> Result<Vec<u8>, PipelineError> {
    let requests = speech_requests(pcm8k, mode)?;
    let responses = run(
        transport,
        &requests,
        Expect::Channel {
            bits: mode.channel_bits(),
        },
    )?;
    let bytes = mode.frame_bytes();
    let mut out = Vec::with_capacity(responses.len() * bytes);
    for r in &responses {
        if let Response::Channel { data, .. } = r {
            out.extend_from_slice(&data[..bytes]);
        }
    }
    Ok(out)
}

/// Decode pass: concatenated channel frames → 8 kHz PCM,
/// `mode.frame_samples()` per frame. `frames.len()` must be a multiple of
/// `mode.frame_bytes()`.
pub fn decode<T: Transport + ?Sized>(
    transport: &mut T,
    mode: VocoderMode,
    frames: &[u8],
) -> Result<Vec<i16>, PipelineError> {
    let bytes = mode.frame_bytes();
    if !frames.len().is_multiple_of(bytes) {
        return Err(PipelineError::CountMismatch {
            sent: frames.len() / bytes,
            received: 0,
        });
    }
    let requests: Vec<Vec<u8>> = frames
        .chunks_exact(bytes)
        .map(|f| channel_in_mode(mode, f))
        .collect();
    let responses = run(transport, &requests, Expect::Speech)?;
    let mut out = Vec::with_capacity(responses.len() * mode.frame_samples());
    for r in &responses {
        if let Response::Speech(pcm) = r {
            out.extend_from_slice(pcm);
        }
    }
    Ok(out)
}

/// One utterance through the chip with **both directions in flight at
/// once**: `speech_in` requests are written keeping at most
/// [`INTERLEAVE_DEPTH`] encodes outstanding, and every `Channel` reply is
/// recorded and handed straight back as a `channel_in` request, keeping
/// at most [`INTERLEAVE_DEPTH`] decodes outstanding. Returns the channel
/// frames (`mode.frame_bytes()` each, concatenated, in order) and the
/// decoded 8 kHz PCM (`mode.frame_samples()` per frame, in order) — the
/// same two values, byte for byte, that [`encode`] followed by
/// [`decode`] produces, for 1.73× the throughput.
///
/// Responses are routed by packet type, not by arrival order: a `Channel`
/// answers the oldest outstanding encode, a `Speech` the oldest
/// outstanding decode. Each side has its own FIFO, its own
/// [`REPLY_TIMEOUT`] age-out and its own count, so a chip that answers
/// one direction and not the other still fails the utterance rather than
/// mis-pairing anything. Every error here is the caller's cue to discard
/// the partial output, reset, re-init and redo the utterance: nothing is
/// ever spliced.
///
/// `pcm8k` must already be padded to whole frames of
/// `mode.frame_samples()` (`vocoder::pad_frames`).
pub fn round_trip<T: Transport + ?Sized>(
    transport: &mut T,
    mode: VocoderMode,
    pcm8k: &[i16],
) -> Result<(Vec<u8>, Vec<i16>), PipelineError> {
    let requests = speech_requests(pcm8k, mode)?;
    let n = requests.len();
    let want_bits = mode.channel_bits();
    let fb = mode.frame_bytes();
    let fs = mode.frame_samples();

    // One FIFO per direction; `queued` holds channel frames the chip has
    // already produced but the decode side has no slot for yet, so the
    // total in the chip never exceeds 2 * INTERLEAVE_DEPTH.
    let mut enc: VecDeque<(usize, Instant)> = VecDeque::with_capacity(INTERLEAVE_DEPTH);
    let mut dec: VecDeque<(usize, Instant)> = VecDeque::with_capacity(INTERLEAVE_DEPTH);
    let mut queued: VecDeque<Vec<u8>> = VecDeque::new();
    let mut frames: Vec<u8> = Vec::with_capacity(n * fb);
    let mut pcm: Vec<i16> = Vec::with_capacity(n * fs);
    let (mut next_enc, mut next_dec) = (0usize, 0usize);
    let mut deframer = Deframer::new();
    let mut buf = [0u8; 1024];

    while next_enc < n || !enc.is_empty() || !queued.is_empty() || !dec.is_empty() {
        while enc.len() < INTERLEAVE_DEPTH && next_enc < n {
            transport.send(&requests[next_enc])?;
            enc.push_back((next_enc, Instant::now()));
            next_enc += 1;
        }
        while dec.len() < INTERLEAVE_DEPTH {
            let Some(frame) = queued.pop_front() else {
                break;
            };
            transport.send(&channel_in_mode(mode, &frame))?;
            dec.push_back((next_dec, Instant::now()));
            next_dec += 1;
        }
        let got = transport.recv_some(&mut buf, READ_POLL)?;
        if got > 0 {
            deframer.push(&buf[..got]);
            while let Some(pkt) = deframer.next_packet() {
                let resp = parse_response(&pkt).map_err(|e| PipelineError::Parse(e.to_string()))?;
                match resp {
                    Response::Channel { bits, data } => {
                        if u16::from(bits) != want_bits {
                            return Err(PipelineError::RateLost {
                                expected: want_bits,
                                got: bits,
                            });
                        }
                        if enc.pop_front().is_none() {
                            return Err(PipelineError::Unexpected(
                                "a Channel response arrived with no encode outstanding".to_owned(),
                            ));
                        }
                        frames.extend_from_slice(&data[..fb]);
                        queued.push_back(data[..fb].to_vec());
                    }
                    Response::Speech(samples) => {
                        if dec.pop_front().is_none() {
                            return Err(PipelineError::Unexpected(
                                "a Speech response arrived with no decode outstanding".to_owned(),
                            ));
                        }
                        pcm.extend_from_slice(&samples);
                    }
                    other => {
                        return Err(PipelineError::Unexpected(format!(
                            "expected Channel or Speech, got {}",
                            describe(&other)
                        )));
                    }
                }
            }
        }
        age_out(&enc, frames.len() / fb)?;
        age_out(&dec, pcm.len() / fs)?;
    }
    if frames.len() / fb != n {
        return Err(PipelineError::CountMismatch {
            sent: n,
            received: frames.len() / fb,
        });
    }
    if pcm.len() / fs != n {
        return Err(PipelineError::CountMismatch {
            sent: n,
            received: pcm.len() / fs,
        });
    }
    Ok((frames, pcm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::channel_in_bits;
    use crate::testutil::hex;
    use crate::vocoder::pad_frames;
    use ambe_thumbdv::MockTransport;

    fn speech_pkt(fill: i16) -> Vec<u8> {
        let mut p = hex("61 01 42 02 00 A0");
        for _ in 0..160 {
            p.extend_from_slice(&fill.to_be_bytes());
        }
        p
    }

    fn channel_pkt(bits: u8, data: &[u8]) -> Vec<u8> {
        channel_in_bits(bits, data)
    }

    /// Three frames of encode then decode on the vendored mock. The
    /// pipeline writes all three ahead before reading, so the mock's
    /// per-request response queue is attached to the third request.
    #[test]
    fn three_frame_encode_then_decode_pins_the_wire_bytes() {
        let pcm: Vec<f32> = (0..480)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        let pcm = pad_frames(&pcm, VocoderMode::YsfDmr);
        let reqs = speech_requests(&pcm, VocoderMode::YsfDmr).unwrap();
        assert_eq!(reqs.len(), 3);
        assert_eq!(&reqs[0][..6], &hex("61 01 42 02 00 A0")[..]);
        assert_eq!(&reqs[0][6..8], &(8192i16).to_be_bytes());
        assert_eq!(&reqs[0][8..10], &(-8192i16).to_be_bytes());

        let mut m = MockTransport::new();
        m.expect(reqs[0].clone(), vec![]);
        m.expect(reqs[1].clone(), vec![]);
        m.expect(
            reqs[2].clone(),
            vec![
                channel_pkt(0x31, &[1; 7]),
                channel_pkt(0x31, &[2; 7]),
                channel_pkt(0x31, &[3; 7]),
            ],
        );
        let frames = encode(&mut m, VocoderMode::YsfDmr, &pcm).unwrap();
        assert_eq!(frames, [[1u8; 7], [2; 7], [3; 7]].concat());
        assert!(m.done());

        let mut m = MockTransport::new();
        m.expect(channel_pkt(0x31, &[1; 7]), vec![]);
        m.expect(channel_pkt(0x31, &[2; 7]), vec![]);
        // Split one response across two chunks to exercise the deframer.
        let third = speech_pkt(3);
        m.expect(
            channel_pkt(0x31, &[3; 7]),
            vec![
                speech_pkt(1),
                speech_pkt(2),
                third[..100].to_vec(),
                third[100..].to_vec(),
            ],
        );
        let pcm = decode(&mut m, VocoderMode::YsfDmr, &frames).unwrap();
        assert_eq!(pcm.len(), 480);
        assert_eq!(pcm[0], 1);
        assert_eq!(pcm[160], 2);
        assert_eq!(pcm[479], 3);
        assert!(m.done());
    }

    #[test]
    fn a_wrong_bit_count_is_rate_lost() {
        let pcm = vec![0i16; 160];
        let reqs = speech_requests(&pcm, VocoderMode::Dstar).unwrap();
        let mut m = MockTransport::new();
        m.expect(reqs[0].clone(), vec![channel_pkt(0x31, &[0; 7])]);
        let err = encode(&mut m, VocoderMode::Dstar, &pcm).unwrap_err();
        assert!(
            matches!(
                err,
                PipelineError::RateLost {
                    expected: 72,
                    got: 0x31
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn silence_ages_out_after_100_ms() {
        let pcm = vec![0i16; 320];
        let reqs = speech_requests(&pcm, VocoderMode::Dstar).unwrap();
        let mut m = MockTransport::new();
        m.expect(reqs[0].clone(), vec![]);
        m.expect(reqs[1].clone(), vec![]);
        let t = Instant::now();
        let err = encode(&mut m, VocoderMode::Dstar, &pcm).unwrap_err();
        assert!(
            matches!(
                err,
                PipelineError::Timeout {
                    index: 0,
                    outstanding: 2,
                    delivered: 0,
                    ..
                }
            ),
            "{err}"
        );
        assert!(t.elapsed() >= Duration::from_millis(100));
        assert!(t.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_speech_reply_to_an_encode_is_unexpected() {
        let pcm = vec![0i16; 160];
        let reqs = speech_requests(&pcm, VocoderMode::Dstar).unwrap();
        let mut m = MockTransport::new();
        m.expect(reqs[0].clone(), vec![speech_pkt(0)]);
        assert!(matches!(
            encode(&mut m, VocoderMode::Dstar, &pcm),
            Err(PipelineError::Unexpected(_))
        ));
        let mut m = MockTransport::new();
        m.expect(channel_pkt(0x48, &[0; 9]), vec![hex("61 00 02 00 0A 00")]);
        assert!(matches!(
            decode(&mut m, VocoderMode::Dstar, &[0; 9]),
            Err(PipelineError::Unexpected(_))
        ));
        assert!(matches!(
            decode(&mut MockTransport::new(), VocoderMode::Dstar, &[0; 8]),
            Err(PipelineError::CountMismatch { .. })
        ));
    }

    // ── The interleaved runner ────────────────────────────────────────
    //
    // Every one of these runs against `sim::SimTransport`. No test in this
    // crate opens a serial port.

    use crate::chip::Chip;
    use crate::sim::{Faults, ReplyBias, SimTransport};
    use crate::testutil::sine;

    const BIASES: [ReplyBias; 3] = [
        ReplyBias::Submission,
        ReplyBias::EncodeFirst,
        ReplyBias::DecodeFirst,
    ];

    fn sim(mode: VocoderMode, faults: Faults) -> Chip<SimTransport> {
        Chip::init(SimTransport::new().with_faults(faults), mode).unwrap()
    }

    /// The whole point: the interleaved runner is a faster way to get the
    /// *same two byte strings*. Both directions of one utterance are
    /// compared against the two separate passes over an identically
    /// initialised sim, for every chip mode and every way the sim can
    /// order the two reply streams against each other.
    #[test]
    fn interleaved_gives_the_same_frames_and_samples_as_the_two_passes() {
        for mode in VocoderMode::ALL.into_iter().filter(|m| !m.is_software()) {
            let pcm = pad_frames(&sine(300.0, 8_000, 4_000, 0.3), mode);
            let n = pcm.len() / mode.frame_samples();
            let mut seq = sim(mode, Faults::default());
            let want_frames = encode(seq.transport_mut(), mode, &pcm).unwrap();
            let want_pcm = decode(seq.transport_mut(), mode, &want_frames).unwrap();
            assert_eq!(want_frames.len(), n * mode.frame_bytes());
            assert_eq!(want_pcm.len(), n * mode.frame_samples());

            for bias in BIASES {
                let mut chip = Chip::init(
                    SimTransport::new()
                        .with_reply_bias(bias)
                        .with_max_in_flight(INTERLEAVE_DEPTH),
                    mode,
                )
                .unwrap();
                let (frames, out) = round_trip(chip.transport_mut(), mode, &pcm).unwrap();
                assert_eq!(frames, want_frames, "{mode} {bias:?}: channel frames");
                assert_eq!(out, want_pcm, "{mode} {bias:?}: decoded samples");
                // Frames in == frames out, both directions.
                assert_eq!(frames.len(), n * mode.frame_bytes());
                assert_eq!(out.len(), n * mode.frame_samples());
                let s = chip.into_transport();
                assert_eq!(s.encodes, n as u64, "{mode} {bias:?}");
                assert_eq!(s.decodes, n as u64, "{mode} {bias:?}");
                // The link was actually kept full in both directions, and
                // never deeper than the chip tolerates (4 + 4 stalls it).
                assert_eq!(
                    s.peak_encodes_in_flight, INTERLEAVE_DEPTH,
                    "{mode} {bias:?}"
                );
                assert_eq!(
                    s.peak_decodes_in_flight, INTERLEAVE_DEPTH,
                    "{mode} {bias:?}"
                );
            }
        }
    }

    /// A single frame still works — the decode side only ever gets work
    /// once the encode side has answered.
    #[test]
    fn one_frame_round_trips() {
        let mode = VocoderMode::Dstar;
        let pcm = pad_frames(&sine(440.0, 8_000, 160, 0.3), mode);
        let mut chip = sim(mode, Faults::default());
        let (frames, out) = round_trip(chip.transport_mut(), mode, &pcm).unwrap();
        assert_eq!(frames.len(), mode.frame_bytes());
        assert_eq!(out.len(), mode.frame_samples());
        assert!(round_trip(chip.transport_mut(), mode, &[0i16; 161]).is_err());
        assert!(
            round_trip(chip.transport_mut(), mode, &[])
                .unwrap()
                .0
                .is_empty()
        );
    }

    /// A lost encode reply leaves one encode outstanding that nothing
    /// will ever answer: the encode side ages out on its own clock and
    /// the whole utterance is an error, never a splice.
    #[test]
    fn a_lost_encode_reply_ages_out_the_encode_side() {
        let mode = VocoderMode::Dstar;
        let pcm = pad_frames(&sine(300.0, 8_000, 1_600, 0.3), mode);
        let mut chip = sim(
            mode,
            Faults {
                drop_encode_response_at: Some(1),
                ..Faults::default()
            },
        );
        let t = Instant::now();
        let err = round_trip(chip.transport_mut(), mode, &pcm).unwrap_err();
        assert!(matches!(err, PipelineError::Timeout { .. }), "{err}");
        assert!(t.elapsed() >= REPLY_TIMEOUT);
        assert!(t.elapsed() < Duration::from_secs(2));
    }

    /// The same for the decode side: the encode direction finishes
    /// perfectly and the run still fails.
    #[test]
    fn a_lost_decode_reply_ages_out_the_decode_side() {
        let mode = VocoderMode::YsfDmr;
        let pcm = pad_frames(&sine(300.0, 8_000, 1_600, 0.3), mode);
        let mut chip = sim(
            mode,
            Faults {
                drop_decode_response_at: Some(0),
                ..Faults::default()
            },
        );
        let t = Instant::now();
        let err = round_trip(chip.transport_mut(), mode, &pcm).unwrap_err();
        assert!(matches!(err, PipelineError::Timeout { .. }), "{err}");
        assert!(t.elapsed() >= REPLY_TIMEOUT);
        assert!(t.elapsed() < Duration::from_secs(2));
        // Every encode was answered; it is the decode side that is owed.
        let s = chip.into_transport();
        assert_eq!(s.encodes, 10);
    }

    /// A chip that lost its rate word answers with the wrong bit count:
    /// still a hard error, never a stored frame.
    #[test]
    fn a_wrong_bit_count_aborts_an_interleaved_run() {
        let mode = VocoderMode::Dstar;
        let pcm = pad_frames(&sine(300.0, 8_000, 1_600, 0.3), mode);
        let mut chip = sim(
            mode,
            Faults {
                wrong_bits_at: Some(2),
                ..Faults::default()
            },
        );
        let err = round_trip(chip.transport_mut(), mode, &pcm).unwrap_err();
        assert!(
            matches!(
                err,
                PipelineError::RateLost {
                    expected: 72,
                    got: 49
                }
            ),
            "{err}"
        );
    }

    /// A response of a kind nothing is waiting for is unexpected, not a
    /// mis-pairing: with no decode outstanding a `Speech` packet cannot
    /// belong to anything.
    #[test]
    fn a_speech_packet_with_no_decode_outstanding_is_unexpected() {
        let mut m = MockTransport::new();
        let pcm = vec![0i16; 160];
        let reqs = speech_requests(&pcm, VocoderMode::Dstar).unwrap();
        m.expect(reqs[0].clone(), vec![speech_pkt(0)]);
        let err = round_trip(&mut m, VocoderMode::Dstar, &pcm).unwrap_err();
        assert!(matches!(err, PipelineError::Unexpected(_)), "{err}");
        // And a control packet answers neither side.
        let mut m = MockTransport::new();
        m.expect(reqs[0].clone(), vec![hex("61 00 02 00 0A 00")]);
        let err = round_trip(&mut m, VocoderMode::Dstar, &pcm).unwrap_err();
        assert!(matches!(err, PipelineError::Unexpected(_)), "{err}");
    }

    #[test]
    fn ragged_input_is_refused() {
        assert!(speech_requests(&[0; 161], VocoderMode::Dstar).is_err());
        assert_eq!(
            speech_requests(&[0; 320], VocoderMode::Dstar)
                .unwrap()
                .len(),
            2
        );
    }
}
