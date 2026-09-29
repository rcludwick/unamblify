# AMBE primer

!!! info "Provenance"
    Written 2026-09-09. Summarised from
    [how-ambe-works](https://github.com/rcludwick/how-ambe-works) (chapters
    07, 09, 10, 11, 12 and `docs/assets/data/probe-rate.md`), the
    AMBE-3000R Users Manual v2.5, and the rate words pinned in astar's
    `astar-codec` and `vendor/ambe-thumbdv` crates. Refer to the long-form site
    for the underlying mathematics. This document includes only the information relevant to unamblify.

## The bit budget

Telephone-quality PCM (8 kHz, 16-bit) is 128 kbit/s. A D-STAR voice channel
allocates 2400 bit/s for voice and 1200 bit/s for forward error
correction. This results in 72 bits per 20 ms frame, with 48 bits describing
the speech. DMR, System Fusion DN, and NXDN use the newer AMBE+2 half-rate
at 2450 bit/s (49 voice bits per 20 ms), either with FEC (DMR: 1150 bit/s) or
without it (YSF DN, NXDN).

This represents a compression factor of approximately 50 compared to PCM. Because waveform coding is inefficient at this rate, AMBE encodes a parametric model of the speech instead.

## What one frame carries

For each 20 ms of speech, the Multi-Band Excitation model transmits:

| Parameter | What it is | Budget (patent example) |
|---|---|---|
| Fundamental frequency ω₀ | pitch, 50–400 Hz | 7 bits |
| Voiced / unvoiced flags | one bit per ~500 Hz band, 8 bands to 4 kHz (AMBE+2: 5 bits) | 8 bits |
| Spectral magnitudes | envelope sampled once per harmonic, L = ⌊3700 / F₀⌋ of them | 57 bits (AMBE+2: 37) |

The number of harmonics (L) is not transmitted directly. It is derived from the pitch.
A 70 Hz voice requires approximately 52 magnitudes and a 400 Hz voice requires 9, both fitting within the
same budget. The magnitudes are coded as a gain combined with a compact transform of
the log-envelope (PRBA and higher-order coefficients). Fine spectral detail uses simpler quantisers and receives less FEC protection.

## Omitted information

1. **Phase:** Phase information is not transmitted. The decoder
   synthesises it using either coherent phase with jitter proportional to
   the unvoiced fraction or a minimum-phase estimate derived from the
   magnitude envelope.
2. **Inter-harmonic data:** The envelope is sampled only at
   multiples of ω₀ and linearly interpolated between these points.
3. **Frequencies above 3.7 kHz:** The coded bandwidth is $0.925 \cdot \pi$ at
   8 kHz.
4. **Fine voicing detail:** Voicing decisions are limited to one bit per band per 20 ms. Breathy,
   mixed, and transitional sounds are categorised as strictly voiced or unvoiced.
5. **Sub-frame time resolution:** Parameters are averaged over 20 ms. This reduces the clarity of consonant attacks.

## Audio artifacts

There are two primary causes for the characteristic sound of AMBE, measured on the AMBE-3000 in how-ambe-works:

- **Buzz:** A band designated as "voiced" is synthesised as a bank of
  phase-locked sinusoids. If the original signal contained partial noise, the
  synthesised result sounds like a buzz. The multi-band flags mitigate this compared to
  single-band vocoders but do not eliminate it entirely.
- **Synthetic phase and texture:** Pitch accuracy is maintained within ~1%, the
  loudness envelope correlates above 0.96, and band levels match within ~2 dB
  of the original. However, signal texture such as noise shape, inter-harmonic
  detail, and consonant onsets are replaced by the decoder's
  reconstruction rules.

A learned post-filter can address this profile. The information missing from the
decoder is primarily statistical. It represents the typical texture and phase of a real voice given a specific pitch, voicing pattern, and envelope. A neural network trained on paired clean and degraded speech can approximate this missing information.

## Decoder synthesis overview

The output signal *s*(n) is the sum of *s*ᵥₒᵢced(n) and *s*ᵤₙᵥₒᵢced(n). Voiced bands are constructed from a sum of
oscillators (one per harmonic) with amplitude and frequency interpolated
across frame boundaries using a five-case transition table. Unvoiced bands
are generated using white noise shaped in the STFT domain and combined via overlap-add. Error handling follows a progression of
correct, smooth, repeat last frame, and mute. A post-filter operates on the output of
this process, whereas a bitstream-conditioned model would operate on the inputs.

## The AMBE-3000 hardware

The DVstick or ThumbDV consists of a DVSI AMBE-3000R behind an FTDI FT230X USB-UART
bridge (VID `0403`, PID `6015`) operating at 460 800 baud 8N1 (older units use 230 400).
It communicates using DVSI's packet protocol in packet mode. Each 20 ms speech packet
(160 samples of 8 kHz s16 PCM, 320 bytes) corresponds to one channel packet.
It operates as a single-channel device with one frame in flight at a
time unless pipelining is used (see [hardware throughput](hardware-throughput.md)).

The rate is configured with a `RATEP` control packet. The configuration words used by astar
are specified below:

| On-air mode | Voice + FEC | Channel bits | `RATEP` words (e u v w x y) | `VocoderMode` |
|---|---|---|---|---|
| D-STAR | 2400 + 1200 | 72 (9 bytes) | `0130 0763 4000 0000 0000 0048` | `dstar` |
| YSF DN, NXDN (and DMR's voice) | 2450 + 0 | 49 (7 bytes) | `0431 0754 0000 0000 0000 7031` | `ysf-dmr` |
| DMR | 2450 + 1150 | 72 (9 bytes) | `0431 0754 2400 0000 0000 6F48` | not captured: same voice bits as YSF/DMR, only the FEC differs. Used only for bit-error framing |

This design influences the capture harness in two ways:

- **YSF DN and DMR use the same 49 AMBE+2 voice bits.** DMR packages these in 23 bits of Golay FEC to create a 72-bit DMR frame. In a loopback environment without channel errors, the decoded audio is identical regardless of the configuration word. Therefore, a single capture covers both modes. The harness records D-STAR and `ysf-dmr` (displayed as "YSF/DMR") as the two supported AMBE modes. DMR's FEC is relevant only when bit errors are present. The `augment --kind ber` stage handles bit errors by synthesising DMR framing from the 49 bits. The DMR rate word and null frame are retained in the codebase as constants (`chip::ratep_dmr`, `channel::NULL_AMBE_FRAME`) for this purpose, but DMR is not maintained as a separate capture mode. Legacy designations (`ysf-dn`, `ysf`, and `dmr`) are parsed as `ysf-dmr`. The `unamblify migrate-modes` command converts data captured under older names.
- **System Fusion VW (full-rate, 7200 bit/s)** represents a third vocoder configuration. A corresponding rate word must be identified and verified against the manual's rate tables before VW capture can be supported in astar.

Performance measurements on this hardware (documented in how-ambe-works, `probe-rate.md`) indicate an encoding time of approximately 16.0 ms per frame, decoding time of approximately 24.5 ms per frame, and a round-trip time of approximately 40.5 ms per frame in stop-and-wait mode. When using four packets in flight (astar's `AmbeStream`), decode time decreases to approximately 7.45 ms per frame while encode time remains at approximately 15.9 ms. Pipelining masks the serial link overhead but does not reduce the encoder's computation time. Additional details are available in [hardware throughput](hardware-throughput.md).

## Codec 2 (M17)

M17 does not use AMBE. Its voice payload is Codec 2, an open and patent-free LGPL-2.1 vocoder developed by David Rowe. It belongs to the same sinusoidal and LPC family as MBE but is fully documented and re-implementable. unamblify captures it as a second vocoder family, allowing the same harness to train an M17 post-filter.
There are two M17 stream types corresponding to two Codec 2 modes:

| M17 stream type | Codec 2 mode | Frame | Samples @ 8 kHz | Bits (bytes) | `VocoderMode` |
|---|---|---|---|---|---|
| voice | 3200 | 20 ms | 160 | 64 (8) | `codec2-3200` |
| voice + data | 1600 | 40 ms | 320 | 64 (8) | `codec2-1600` |

Mode 1600 reduces the voice rate by half to allow 1600 bit/s of data to share the
3200 bit/s payload. In this mode, 40 ms channel frames are used instead of 20 ms frames. As a result, all frame counts in the pipeline are calculated using `VocoderMode::frame_samples()` rather than a fixed AMBE constant.

Codec 2 processing does not rely on a dedicated hardware chip. The capture harness executes the pure-Rust [`codec2` crate](https://crates.io/crates/codec2) concurrently across the corpus. The crate is a port of the C reference implementation, dual-licensed under `LGPL-2.1-only AND MIT`, and pinned at version 0.3.1. This version is recorded in the `version` field for each capture row. 

The Codec 2 encoder is deterministic. However, the decoder draws unvoiced-harmonic phases from a process-global generator similar to the C reference. Consequently, the decoded PCM output is not strictly bit-reproducible from the frames. Since the primary training target is the clean signal, this variation is acceptable. Measured throughput and capture guidelines are detailed in [data pipeline](../design/data-pipeline.md#software-capture), and licensing information is available in [license](../about/license.md). In astar, the same crate is utilized via the `codec2-static` feature in `astar-codec` (supporting mode 3200 only, corresponding to current M17 voice-only stream decoding).

Sources: The [M17 specification](https://spec.m17project.org/) defines the stream payload as Codec 2 3200 for "voice" and 1600 for "voice + data". Additional details can be found in the [codec2](https://github.com/drowe67/codec2) repository (mode table and license) and the `codec2` crate version 0.3.1 (`bits_per_frame` and `samples_per_frame`).

## Patents

The foundational MBE and AMBE patents expired in 2017. Two AMBE+2 patents remain active:
US 8,359,197 (half-rate encode and decode), which expires on 2028-05-20, and
US 8,036,886 (encoder analysis), which expires on 2029-10-02. 

The unamblify v1 post-filter processes decoded PCM audio provided by a licensed DVSI chip. It does not implement the encoding or decoding of AMBE and therefore does not employ the claimed methods. Developing a model that processes the AMBE+2 bitstream directly (a neural decoder) involves different considerations and is not currently implemented. Further details are discussed in [design](../design/index.md).
