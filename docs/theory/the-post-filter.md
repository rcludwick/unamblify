# The post-filter

!!! info "Provenance"
    Written 2026-09-12 against `crates/unamblify-train/src/model.rs`
    (candidate 1) and the measured results of 2026-09-11/12. The
    configuration surface is in [Training](../design/training.md). The
    latency arithmetic is in [Real time](../design/realtime.md).

!!! note "Since this page was written"
    This page describes candidate 1, the adaptive filter (2.1 M params). It
    topped out near 2.1 predicted MOS. The best system is now the restorer
    + waveform synthesiser pipeline (candidate 3 in
    [Training](../design/training.md)), which does resynthesise. It scores
    3.57 predicted MOS on a faithful input (codec 2.22 on the same clips,
    clean 4.14) and 4.6 of 5 in a blind listening test. Its Rust runtime is
    `unamblify restore`.

## The central choice: filter, don't resynthesise

There are two primary approaches to improving low-rate codec audio quality.

Resynthesis involves discarding the decoder output, inferring the speech, and generating fresh audio with a neural vocoder. While this can produce high-quality audio, generative models can produce incorrect but confident outputs. For example, FreeDV's RADE can initially sound like a fast-talking foreign speaker before settling on the correct syllables. Generating audio also requires significant compute and context, which increases latency.

Filtering keeps the decoder output and applies a time-varying filter to emphasize desired frequencies and suppress noise. The output remains a filtered version of the actual received signal. Consequently, failure modes sound like a lack of improvement rather than artificial speech. Filtering requires less compute and provides stable results with configurable lookahead delay.

Candidate 1 uses the filtering approach. This decision builds on Xiph's [OSCE](../research/prior-art.md) work (LACE and NoLACE for Opus). This prior work demonstrated that a small network steering a filter provides most of the potential quality improvement with less computational overhead than generative models.

## The shape of the network

```text
deg8 [B,1,N] ─ frame conv (k=160, hop 80, causal) ────► [B,F,T]   T = N/80
            ─ context conv, pad (PAST_CTX, 2L) ───────► [B,F,T]
            ─ ⊕ mode embedding [16] at every frame ───► [B,T,F+16]
            ─ 2-layer GRU ───────────────────────────► g [B,T,F]
  g ─ fir_head   ─► 16 FIR taps per frame          ─┐
  g ─ pitch_head ─► pitch-lag mix + gain           ─┴► y8 = FIR(deg8) + gain·comb(deg8)
  y8 ─ fixed half-band upsampler ──────────────────► up16
       up16 + BWE(up16, g) ────────────────────────► out16 [B,1,2N]
```

The model operates in three stages: feature extraction, filter parameter generation, and filter application with bandwidth extension.

### Stage 1: features at frame rate, not sample rate

The initial convolution uses a 160-sample (20 ms) window with an 80-sample (10 ms) hop. This converts the 8 kHz sample rate into a 100 Hz feature frame rate. Subsequent processing operates at this lower 100 Hz rate.

This rate reduction is the primary factor enabling real-time operation. Processing per sample would require significantly more compute. The characteristics being corrected (pitch, envelope, voicing) change at the frame rate, which matches the codec transmission rate.

### Stage 2: a GRU that remembers

A context convolution expands each frame's context to `PAST_CTX = 8` past frames and `2L` future frames. A two-layer GRU then processes the sequence. Because the GRU maintains state, the filter parameters for a given frame depend on the entire transmission history rather than a fixed window.

This historical context is necessary because artefacts are contextual. Determining whether high-frequency noise is a fricative or background breath depends on adjacent sounds. Distinguishing between genuine intonation changes and quantization artifacts requires observing the surrounding pitch track.

The GRU width `F` varies by [profile](#profiles): 256 for full, 96 for lite, and 32 for super-lite.

### Stage 3: two filters and an extender

The GRU output produces filter coefficients at a rate of one set per 10 ms frame:

- A 16-tap FIR filter shapes the short-term spectrum. This provides formant and envelope correction and addresses quantization artifacts.
- A comb filter over 24 candidate pitch lags (20 to 160 samples, or 400 Hz down to 50 Hz) with an associated gain parameter. The comb filter restores harmonic structure at the pitch period. Higher gain sharpens harmonics, while lower gain preserves between-harmonic noise, balancing breathiness and buzz.

Per-frame quantities (taps, comb mix, gains, and BWE conditioning) are interpolated across the frame's samples rather than held constant. During the 80 samples of frame `t`, values ramp from the parameters of frame `t − 1` to those of frame `t`. This process remains causal and removes the 10 ms parameter steps present in earlier versions. The runtime implements the same interpolation.

When `[model] noise_head` is enabled, a third path adds shaped white noise with a per-frame gain relative to the local level. This path introduces energy into harmonic troughs emptied by the codec. If `[model] noise_mod` is active, this noise receives sub-frame time structure. It is multiplied by the envelope of the periodic path (a 2 ms causal RMS normalized to the frame level) at a learned depth, aligning breath noise with the glottal cycle. It also applies a millisecond-level gain (ten values per frame) to support short plosive bursts that occur faster than the 10 ms feature hop.

These paths are summed to produce the corrected 8 kHz signal. A fixed 31-tap half-band filter upsamples the signal to 16 kHz. A fixed filter is used here because interpolation does not require learned parameters. Finally, the BWE head adds a residual signal to the upsampled output, conditioned on the GRU features, to synthesize 4–8 kHz content. The super-lite profile omits the BWE head and outputs the upsampled signal directly.

### Mode embedding

When `[model] mode_embed = true`, the network includes a learned 16-element embedding vector for each vocoder mode. This vector is concatenated to the features of every frame, indicating whether the input is D-STAR, YSF/DMR, Codec 2 3200, or Codec 2 1600.

This embedding allows a single model to support multiple modes with minimal parameter overhead. The receiving radio provides the active mode during demodulation. [Experiments](what-the-experiments-showed.md) indicate that a single multi-mode model performs comparably to or better than separate models for each mode.

## Causality and latency

Processing after the context convolution is strictly causal in terms of frames. Lookahead is localized to the context convolution's right padding of `2L` feature frames, where `L` represents `[model] lookahead` in AMBE frames.

The total delay is configurable and equals `20·L` ms plus 0.94 ms of resampler group delay:

| Variant | Lookahead | Added delay |
|---|---|---|
| ll5 | 5 AMBE frames | ~101 ms |
| ll20 | 20 AMBE frames | ~401 ms |

Increased lookahead allows the filter to incorporate future context, which primarily improves performance at onsets. In testing, the two variants showed similar performance (11.02 vs 11.04), indicating that 100 ms of lookahead is generally sufficient for this task.

## State reset in training

Training examples initialize the GRU with a zero state and use zero padding on both convolutions. This matches the runtime conditions at the beginning of a transmission. Consequently, the initial frames of each example represent the conditions with minimal context. To account for this, the loss function [applies additional weight to onsets](how-it-learns.md#the-edges-matter-most).

## Profiles { #profiles }

The same architecture comes at three widths, each with a parameter budget
the trainer enforces:

| Profile | `F` | BWE | Parameters | Target |
|---|---|---|---|---|
| full | 256 | 32 ch | 2 117 130 | desktop and Apple Silicon cores |
| lite | 96 | 16 ch | 315 226 | mobile ARM, int8 |
| super-lite | 32 | none | ~38 700 | microcontrollers |

The `[model] width` parameter scales these values to test capacity increases. [Results indicate](what-the-experiments-showed.md#capacity) that increasing capacity beyond the full profile yields diminishing returns.
