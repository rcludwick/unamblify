# Hardware throughput budget

!!! info "Provenance"
    Written 2026-09-09. Pipelined timings corrected 2026-09-10. Interleaving
    measured and the day tables recomputed 2026-09-11. Stop-and-wait
    timings come from how-ambe-works (`docs/assets/data/probe-rate.md`,
    measured on a ThumbDV `AMBE3000F`, firmware `V121.E100…`). Pipelined
    timings come from astar's guard tests in `crates/astar-codec/src/ambe.rs`
    (four packets in flight, same stick). Corpus hours come from the
    corpora's own documentation.

The AMBE-3000 is a single-channel chip that processes one 20 ms frame per
packet exchange. This is the bottleneck for dataset creation rather than
disk or network speed. Every hour of training audio must be processed
through the chip for each vocoder mode during encoding and decoding.

## Per-frame cost

Measured on the project's ThumbDV (AMBE3000F, 460 800 baud) on
2026-09-11, 200–300 frames per point:

| Regime | ms per audio frame | Note |
|---|---|---|
| Stop-and-wait, one direction at a time | 40 | encode 16.0 + decode 24.0 |
| Pipelined 4 deep, encode pass then decode pass | 14.3 | encode 7.12 + decode 7.21. Now `capture --sequential` |
| Interleaved, 3 encodes and 3 decodes in flight | 8.27 | 1.73× faster because the UART is full duplex. Used by the harness. |

On real utterances the harness's interleaved runner records 9.9 ms per
audio frame (`roundtrip_ms / frames`, VCTK utterances of 250–370 frames)
rather than the probe's 8.27. An utterance also includes warm-up frames and a
per-utterance turnaround that the probe's steady stream does not. Measured
against itself, the same capture went from 68.9 to 97 frames/s and from
1 893 to ~2 500 utterances/hour. This is 1.41× end to end with no
age-outs.

Depth is not a free parameter. 2 + 2 costs 8.59 ms. Every asymmetric split
is worse (3 + 2 is 10.85 ms, 2 + 3 is 10.96, 4 + 2 is 11.31, 2 + 4 is
12.71). A 4 + 4 split stalls the chip because its input queue overflows and a
response is never returned.

7.07 ms of every direction is the 326-byte speech packet crossing the wire
at 460 800 baud. A single direction cannot go below ~7.1 ms and
depth beyond 4 provides no benefit. 460 800 is the AMBE-3000's maximum UART
rate (manual Table 19: 28800 / 57600 / 115200 / 230400 / 460800, and our
stick's `CFG2 = 0xEC` selects 460 800) so no baud change is available. On
one stick the only optimization left is interleaving the two directions. Beyond
that, further scaling requires more sticks. The harness takes a `--port` per stick
and assigns work by speaker.

## What the seed costs

| Corpus (tier 0) | Hours (approx.) |
|---|---|
| VoiceBank-DEMAND clean (28 spk train + test) | 11 |
| LibriTTS-R dev_clean + test_clean | 18 |
| Total | ≈ 29 |

| Modes captured | Stop-and-wait | Two passes | Interleaved |
|---|---|---|---|
| D-STAR only | 2.4 days | 0.9 days | 0.5 days |
| D-STAR + YSF/DMR | 4.8 days | 1.7 days | 1.0 days |

The whole pipeline (capture, sharding, initial training run, and first
metrics) can be exercised end-to-end within a week on one stick before
committing to the core set.

## What the core corpus costs

| Corpus (tiers 0 + 1) | Hours (approx.) |
|---|---|
| Seed (above) | 29 |
| LibriTTS-R train_clean_100 + train_clean_360 | 245 |
| VCTK 0.92 | 44 |
| LJSpeech | 24 |
| Total | ≈ 342 |

| Modes captured | Stop-and-wait | Two passes | Interleaved |
|---|---|---|---|
| D-STAR only | 29 days | 10 days | 6 days |
| D-STAR + YSF/DMR | 57 days | 20 days | 12 days |

DMR is not a third capture. It carries YSF DN's voice bits so the
`ysf-dmr` capture serves both ([primer](ambe-primer.md#the-ambe-3000-hardware)).

Adding tier 2 (Hi-Fi TTS 292 h, LibriSpeech train-clean-100 100 h) roughly
doubles all of these figures.

## Consequences for the harness

1. Interleave the two directions. Pipelining one direction only
   reaches the wire's floor. The remaining performance gain is from the
   UART being full duplex. `capture` on an AMBE mode now runs
   `pipeline::round_trip` which keeps three `speech_in` requests and
   three `channel_in` requests in the chip at once. Each `Channel` reply is
   recorded and immediately resubmitted as a decode. Responses are routed by
   packet type and matched FIFO within their kind (the same shape as astar's
   `AmbeStream` worker). The output is byte-identical to the two passes, and
   `capture --sequential` goes back to them for a stick that misbehaves.
   Because the directions overlap there is only one wall time to record. The
   capture manifest carries `roundtrip_ms` and leaves `encode_ms` / `decode_ms`
   at 0 instead of inventing a split. `unamblify stats` prints whichever are
   non-zero. The one-direction runner (`pipeline::run`, four in flight)
   stays for the canary, the warm-up, `--sequential`, and the decode-only
   `augment` stage.
2. Order the work by value. Capture the tier-0 seed first:
   VoiceBank-DEMAND (the enhancement community's standard benchmark) then
   LibriTTS-R dev/test so evaluation exists early. Then VCTK for speaker
   diversity, then LJSpeech, then LibriTTS-R train. Training starts on the
   seed while the rest captures.
3. Make it resume-safe. The run spans weeks. USB sticks get unplugged and
   the chip occasionally needs a reset. Every utterance is a unit of work
   with a done-marker, and the harness restarts from the manifest.
4. Keep the channel frames. At 450 bytes per second they require minimal
   storage. They let decode be re-run (for error-injection experiments)
   without re-encoding, which halves the time required for those experiments.
5. More sticks scale linearly. A second ThumbDV halves every number in
   the table. The harness takes a list of `--port`s and assigns work by
   speaker, with one worker thread per stick.
6. A software proxy is not a substitute. The open-source D-STAR codec
   in `rcludwick/ambe` runs faster than real time but is not bit-exact
   with DVSI. It could pre-train a model or augment the data. The network
   must learn the chip's decoder, so the paired data must come from the chip.

## Storage

8 kHz s16 PCM is 57.6 MB per hour, 16 kHz is 115 MB per hour, and channel
frames are 1.6 MB per hour. The whole tier-1 set is under 100 GB. This
includes a 16 kHz clean reference, an 8 kHz chip input, two modes of
degraded output, and channel frames. The 8 TB drive is not a constraint.
