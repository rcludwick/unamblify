# Real-time inference

!!! info "Status"
    2026-09-09: planned. As of 2026-09-25 two pieces are built. One is the
    gate, `unamblify bench` (ms per 20 ms frame on one CPU thread). The
    other is the [pipeline runtime](pipeline-runtime.md), which runs the
    restorer + waveform synthesiser in Rust, batch and frame by frame
    (`unamblify restore`). The astar integration is not started. This
    page was written for the filter model (candidate 1). The pipeline's
    own latency (about 400 ms) and budget are on the pipeline runtime
    page.

## Integration points

Inside [astar](https://github.com/rcludwick/astar)'s receive path, processing occurs after the
ThumbDV decode and before the mixer: `[u8; 9]` channel frame → chip →
`[i16; 160]` at 8 kHz → unamblify → 160 or 320 samples at 8 or 16 kHz →
astar's resampler → device. astar runs RNNoise (`nnnoiseless`) on
the transmit side; unamblify provides the receive-side counterpart using
the same structure: a `Send` object with `push_frame` and `pull_frame` methods.

## Profiles

The post-filter provides three configuration profiles varying by width. The runtime selects the largest model the host device can support.

| Profile | Runs on | Output | Size | Compute per 20 ms frame | Notes |
|---|---|---|---|---|---|
| full | Standard CPUs: Apple Silicon, EPYC, Xeon, desktop AMD and Intel | 16 kHz, bandwidth-extended | 2–8 M params, f32 or f16 | ≤ 5 ms on one core | Primary quality target meeting both design goals. |
| lite | Mobile and embedded ARM devices: Raspberry Pi 4/5, smartphones, SBCs | 8 kHz, natural narrowband | ≤ 1 M params, int8 weights | ≤ 5 ms on one Cortex-A72-class core | Meets Goal 1. Weights fit within L2 cache to remain compute-bound. (Reference: Opus BBWENet usage in [prior art](../research/prior-art.md)). |
| super lite | Microcontrollers: ESP32-S3 / P4, STM32H7, RP2350, other Cortex-M4F/M7-class parts with DSP or SIMD instructions, small NPUs | 8 kHz, narrowband de-buzz | ≤ 100 K params, int8 weights and activations, ≤ ~2 M MACs per frame | ≤ 10 ms of a 240 MHz Cortex-M7 | Meets Goal 1 with reduced complexity. Uses band-domain gains and a light excitation post-filter. Requires ≤ 128 KB RAM and no float support. |

The super lite profile operates in a `no_std` environment without an allocator or FPU. It uses a hand-written int8 kernel (or CMSIS-NN / Ethos-U) and targets hotspots and handheld modems. A DV hotspot can host it on the same board as the AMBE chip. This profile will be evaluated last and removed if it does not improve upon bypass mode.

Only one profile may be necessary. If the lite model performs equally to the full model in listening tests, the full model will be removed. If the full model runs efficiently on a Pi 5, the lite model will be removed. Both profiles are maintained until testing is complete. Training an additional width requires minimal overhead.

Design constraints for the lite profile:

- No attention mechanisms or large activations. It uses small causal convolutions and a GRU. Operations are NEON-friendly and layer working sets fit within L2 cache.
- Defaults to int8. Post-training quantisation is applied first. Quantisation-aware fine-tuning is used only if quality degrades.
- Trained using distillation. The full model acts as a teacher for feature and output distillation alongside standard losses.
- Unified export path. All profiles export as ONNX graphs from the same trainer. The super lite profile additionally exports a flat int8 weight blob and a `no_std` Rust or C table for the microcontroller kernel.

## Latency variants

Every profile is trained with two lookahead configurations:

| Variant | Lookahead | Added latency | Use |
|---|---|---|---|
| ll5 | 5 AMBE frames | 100 ms | Conversational default: suitable for quick-turnaround QSOs or repeater/hotspots where latency accumulates. |
| ll20 | 20 AMBE frames | 400 ms | Quality focus: the model processes a full syllable ahead, improving decisions on consonant onsets, pitch glides, and voicing transitions. Suitable for nets, listening, and recording. |

DV links introduce codec, network and framing delay. Since traffic is push-to-talk, both variants are usable on the air. The ll20 variant approaches the limit where additional lookahead provides minimal benefit. The variant is selected via configuration (`latency = "ll5" | "ll20"`) and the runtime buffers the corresponding number of decoded frames before outputting. A zero-lookahead variant is trained as an experimental control but is not shipped.

## Runtime selection

At stream initialization, the runtime benchmarks each available profile using synthetic input frames. It selects the largest profile with a worst-case frame time under 25% of the frame period (5 ms of 20 ms). This headroom accounts for audio thread resource sharing within astar and potential thermal throttling on embedded devices. Benchmark results are cached and invalidated if the CPU model string changes. Configuration keys allow manual overrides for the profile (`auto`, `full`, `lite`, `off`) and latency variant (`ll5`, `ll20`). If a profile misses its deadline for three consecutive frames, the system falls back to a lower profile or bypass mode.

## Budget

| Item | Target |
|---|---|
| Added algorithmic latency | 100 ms (ll5) or 400 ms (ll20), fixed per variant |
| Compute per 20 ms frame | ≤ 5 ms on one core of the target machine, either profile |
| Model size | full ≤ 8 M params; lite ≤ 1 M params, int8; super lite ≤ 100 K params, int8, ≤ 128 KB RAM |
| Dependencies | pure Rust; no libtorch, no Python at runtime |
| Allocation | none on the audio thread after warm-up |

## Runtime options

- `ort` (ONNX Runtime): Fast CPU kernels but requires a C++ dependency and increases binary size.
- `tract`: Pure Rust ONNX inference without C++ dependencies. Produces a smaller binary but may execute slower.
- Hand-written kernels: For candidate 1, the network consists of small GEMMs and a GRU. Implementing this directly or using `burn`'s ndarray backend provides explicit allocation control. This approach is suitable for the lite profile using int8 GEMM operations with NEON intrinsics.

The runtime will be selected after the final model is chosen. ONNX export is used during training to maintain compatibility with all options.

## Streaming discipline

- The model is trained with its variant's exact lookahead (5 or 20 frames) to match inference context. There is no offline normalisation or access to future frames beyond the lookahead buffer.
- Internal state (GRU hidden state, filter memories, overlap-add buffers) is stored in the post-filter object and resets at stream start. After reset, the lookahead buffer fills before the first output. Training follows this exact reset-then-buffer approach.
- The end of stream is handled explicitly and flushes at real time. The framing layer (D-STAR end frame, YSF terminator, DMR end-of-voice, or timeout) signals the transmission end. The post-filter pads the lookahead buffer with silence and drains it at one frame per 20 ms. It does not process stale state or flush faster than real time to prevent spurious outputs at key-down (see [training](training.md#the-edges-matter-most-key-up-and-key-down)).
- Invalid frames are gated. Frames marked by the framing layer as lost, unlocked, or decoded from noise bypass enhancement. The runtime passes them through or mutes them while freezing model state. The model is also trained to target silence on these frames.
- Hardware mute or repeat frames pass through unmodified with model state frozen to prevent hallucinations during dropouts.
- A bypass switch and dry/wet mix configuration are provided for real-time comparison.
