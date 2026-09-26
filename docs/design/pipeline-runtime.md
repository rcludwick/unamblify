# The pipeline runtime

!!! info "Status"
    2026-09-25: The restorer and waveform synthesiser from
    [training](training.md#candidate-3-concretely) run in Rust. They process
    both batch and frame-by-frame on weights exported from the spike, and
    reproduce the spike's numbers against PyTorch references. Feature extraction
    remains in the batch path, so the frame-by-frame [`Stream`] takes feature
    frames. This is the final missing piece for a receiver. Training remains in
    Python (`scripts/spike/`). This document covers inference.

## What runs

```text
x8 (8 kHz codec output)
  ─ resample ×3 (torchaudio's windowed-sinc kernel, from the file)
  ─ STFT 1024 / 256, periodic Hann, centred with reflection
  ─ low band: log |X| over 171 bins (0–4 kHz), floored at 1e-5
  ─ mel: 100 HTK bands (torchaudio's filterbank, from the file), log floored at 1e-7 (Vocos's)
  ─► restorer: [low ‖ mel ‖ embed(mode)] → conv k5 → 6 ConvNeXt blocks → GRU → LayerNorm → Linear → + mel
  ─► synthesiser (Vocos): conv k7 → LayerNorm → 8 ConvNeXt blocks → LayerNorm → Linear 1026
        → magnitude = exp(·) ≤ 100, phase → complex spectrum → torch.istft(center=True)
  ─► x24 (24 kHz), 256 × (T − 1) samples for T frames
```

The code is located in `crates/unamblify-train/src/pipeline/`, including `features`,
`restorer`, `synth`, `stream`, and `weights`. It requires the `train`
feature because it relies on tch-rs for libtorch. Binaries built without
default features do not include it.

## The weights file

The released weights are
[0.0.1-beta](https://github.com/rcludwick/unamblify/releases/tag/v0.0.1-beta)
(`AGPL-3.0-only`): `pipeline.safetensors`, `pipeline.json` and
`references.safetensors`, exported as below from restorer-v3 at step
30 000 and the G4 synthesiser.


`scripts/spike/export_weights.py <restorer.pt> <synth.pt> <out-dir>`
writes `pipeline.safetensors` and `pipeline.json` side by side. It also
writes PyTorch references for two eval clips (`references.safetensors`:
`<mode>.{x8,low,degmel,mel,wav24}`).

| In the safetensors | What |
|---|---|
| `restorer.*` | the spike's `Restorer` state dict: `emb`, `conv_in`, `blocks.N.{dw,norm,pw1,pw2,gamma}`, `gru.*_l0`, `norm`, `out` |
| `synth.backbone.*`, `synth.head.*` | Vocos's backbone and ISTFT head as fine-tuned (`embed`, `norm`, `convnext.N.{dwconv,norm,pwconv1,pwconv2,gamma}`, `final_layer_norm`, `head.out`, `head.istft.window`) |
| `features.mel_fbank` | torchaudio's `[513, 100]` HTK filterbank |
| `features.resample_kernel` | torchaudio's `[3, 1, 15]` sinc kernel for 8 → 24 kHz |
| `features.window`, `features.mel_window` | the periodic Hann window of the low band's STFT, and the one Vocos's checkpoint carries for the mel's. They are one ulp apart. The ulp moves the stop-band floor the restorer sees, so both are kept. |

The manifest specifies the mode order (the embedding index for each mode),
each convolution's `kernel / left / right` padding, LayerNorm epsilons,
the magnitude clip limit, and the constants defining the features. The runtime
rejects manifests with mismatched constants. The exporter rejects a restorer
trained with channel-bits input (`BITS=1`) or a causal converted synthesiser,
as the runtime supports neither.

## Numerical agreement

Running `cargo test -p unamblify-train --lib reference_matches_pytorch -- --ignored`
with `UNAMBLIFY_PIPELINE_EXPORT` set to the export directory tests each stage
on the reference input and the full pipeline on the 8 kHz clip. Performance
was measured on 2026-09-25 on the D-STAR and Codec 2 3200 eval clip
(`libritts_r/dev-clean/1272_128104_000005_000009`), using restorer-v3 at step 30 000
and the GTA-fine-tuned synthesiser. PyTorch references were generated with
torch 2.13.0 and 2.14.0. The runtime used libtorch 2.13.0:

| Stage, on the reference input | D-STAR | Codec 2 3200 | Asserted |
|---|---|---|---|
| low band and mel, linear magnitude / peak | 0 (bit-exact) | 0 | < 1e-6 |
| restorer, max \|Δ\| log-mel | 5.7e-6 | 3.8e-6 | < 1e-4 |
| synthesiser, max \|Δ\| / peak | 8.6e-6 | 4.9e-6 | < 1e-3 |
| end to end, predicted mel, linear / peak | 1.0e-6 | 6.8e-7 | < 1e-4 |
| end to end, audio, max \|Δ\| / peak | 3.2e-4 | 2.4e-5 | < 1e-2 |
| end to end, audio, rms / peak | 8.6e-6 | 1.1e-6 | |

Feature inputs must be bit-exact. The mel top bands and the low band last bins
sit at the resampler's stop-band floor, where magnitude is determined by float
rounding. The restorer processes them in log units. A single ulp difference in
the resampling kernel shifted a floor band by 0.5 nats, the predicted mel by
3e-3 of its peak, and the audio by 0.68 of its peak. Note that torchaudio builds
the kernel in float64 and casts it, which produces different results than building
it natively in float32. A single ulp difference in the window causes similar
deviations. Due to this sensitivity, the kernel and both windows are read from
the file rather than being recomputed. This constraint applies to any streaming
feature extractor. Its STFT must identically match libtorch's output, otherwise
the restorer requires retraining.

## Frame by frame

`Stream::new(&pipeline, mode)` initializes a stream. Calling `push(low_frame, mel_frame)`
returns the next 256 audio samples after the lookahead limit is reached. Calling
`flush()` at the end returns any remaining samples. The stream output matches the
batch path output up to floating-point noise, which is verified by the
`the_stream_matches_the_batch_path` test. Lookahead layers retain their last
`k` input frames and apply a delay based on right padding. The GRU maintains
its internal state. The overlap-add process stores one window of samples and
squared windows, emitting a block once the final covering window is added.
During flush, layers receive the zero-padding frames used by the batch path.

| Stage | Frames | ms |
|---|---|---|
| centred STFT (not yet streamed) | 2 | 21.3 |
| restorer | 8 | 85.3 |
| synthesiser | 27 | 288 |
| iSTFT | 2 | 21.3 |
| mel frame in → audio out | 37 | 394.7 |
| audio in → audio out, once the features stream | 39 (+ 7 samples at 8 kHz) | 416.9 |

This latency satisfies the 500 ms budget established in
[training](training.md#candidate-3-concretely). Lower latency requires
retraining the model with fewer look-ahead layers.

## Benchmarking

```sh
unamblify bench --kind pipeline [--threads 1] [--seconds 8] [--passes 5] [--weights pipeline.safetensors]
```

This benchmark outputs processing time in milliseconds per 20 ms of audio for
the features, restorer, and synthesiser on the batch path. It supports running
on random weights using exported shapes or an actual weights file. Single-thread
performance is the relevant metric for handheld devices. Setting `--threads 0`
allows torch to utilize all available cores. A single clip can be processed
using `unamblify restore --weights W --in <8 kHz wav> --mode M --out <wav>`.

Measurements on an M4 Pro (2026-09-25, release build, exported weights, 8 s clips)
per 20 ms of audio:

| Threads | Features | Restorer | Synthesiser | Total | Of the budget |
|---|---|---|---|---|---|
| 1 | 0.01 ms | 0.08 ms | 0.09 ms | 0.18 ms | 0.9 % |
| all (10) | 0.01 ms | 0.25 ms | 0.40 ms | 0.65 ms | 3.3 % |

A single thread outperforms ten threads because each batch contains one clip,
making parallelization overhead higher than the performance benefit. These
metrics do not reflect handheld performance. Apple's matrix unit accelerates
these GEMMs, whereas a Cortex-A55 lacks equivalent hardware. The 20 million
parameters require ~1.3 GFLOP per second of audio in the restorer and ~2.5 GFLOP
in the synthesiser (calculated as 2 × parameters × 93.75 frames). Accurate hardware
planning requires benchmarking directly on the FRDM-IMX93 board.

## What is still Python

Training, the exporter, and the streaming path features remain in Python. The
batch `Features` implementation is in Rust. The Rust runtime is inference-only,
lacking `VarStore`, gradients, and config keys. A restorer trained with a Rust
trainer requires the same export process.
