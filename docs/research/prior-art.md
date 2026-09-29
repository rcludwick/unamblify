# Prior art

!!! info "Provenance"
    Literature survey 2026-09-09, from web search and the cited papers'
    abstracts or PDFs. Items marked **[unverified]** could not be confirmed
    against a primary source in that pass and should be re-checked before a
    decision rests on them.

## The short version

Nobody has published a neural post-filter for MBE / IMBE / AMBE, P25, DMR,
or amateur DV. Three lineages come closest. Xiph's OSCE work on Opus/SILK
(LACE, NoLACE, BBWENet, FARGAN) is one. David Rowe's Codec 2 / FreeDV neural
work is another, and it shares our *domain*: a sinusoidal vocoder on ham
radio. The third is Fraunhofer's coded-speech enhancement and
bandwidth-extension GANs. All three reach the same result: **a small
GAN-trained model, conditioned on the decoded signal and running causally,
recovers most of what a low-rate codec loses**. A bandwidth-extension head
on top is well trodden.

That literature assumes the codec's bitstream is unavailable and works from
decoded PCM only. We have the bitstream: the chip hands us the 49/72-bit
channel frames alongside the PCM. So a *bitstream-conditioned* model is an
option for us that it was not for them. See [design](../design/index.md)
for why v1 still starts from PCM.

## 1. Enhancement of low-bitrate codecs and parametric vocoders

**Xiph / Amazon / Meta (Valin, Büthe, Mustafa, Skoglund): OSCE, the
most relevant lineage.** The umbrella name is **OSCE, Opus Speech Coding
Enhancement**. It is the decoder-side neural enhancement that shipped in
Opus 1.5 (LACE, NoLACE), and the IETF mlcodec working group is
standardising it as
[draft-ietf-mlcodec-opus-speech-coding-enhancement](https://www.ietf.org/archive/id/draft-ietf-mlcodec-opus-speech-coding-enhancement-01.html).
Two details from the IETF 121/124 updates
([slides](https://datatracker.ietf.org/meeting/124/materials/slides-124-mlcodec-speech-enhancement-00))
matter to us directly:

- **BBWENet** is a blind bandwidth-extension net (wideband → 20 kHz), now
  part of OSCE. Quantised to 60 % for a 50 % size cut, it costs **15.9 % of
  a Cortex-A53 @ 1.4 GHz, 5.0 % of a Cortex-A72 @ 1.5 GHz, 1.2 % of a
  Cortex-A76 @ 2.4 GHz** on a single core. That is the yardstick for our
  *lite* profile on Pi-class hardware.
- Their evaluation rule, *"BWE should not be worse than doing nothing"*,
  found failure cases that objective metrics did not flag as audible. Our
  bandwidth-extension head must be tested against the same trap.

- *Improving Opus Low Bit Rate Quality with Neural Speech Synthesis*
  (Skoglund & Valin, Interspeech 2020, [arXiv 1905.04628](https://arxiv.org/abs/1905.04628)).
  It resynthesises 6 kb/s Opus from decoded parameters with LPCNet and
  clearly beats the standard decoder, at 25 ms delay and ~3 GFLOPS.
- *A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet*
  (Valin & Skoglund 2019, [arXiv 1903.12087](https://arxiv.org/abs/1903.12087)).
  Code at [xiph/LPCNet](https://github.com/xiph/LPCNet).
- *LACE* (Büthe, Valin, Mustafa, WASPAA 2023, [arXiv 2307.06610](https://arxiv.org/abs/2307.06610)).
  A DNN predicts per-frame coefficients for classical long-term (pitch) and
  short-term post-filters, and the signal path stays linear. 300 K params,
  ~100 MFLOPS, zero added delay.
- *NoLACE* (Büthe et al., ICASSP 2024, [arXiv 2309.14521](https://arxiv.org/abs/2309.14521)).
  It adds non-linear adaptive temporal shaping. Enlarged LACE = 280 MFLOPS /
  900 K params. NoLACE ≈ 620 MFLOPS / 1.8 M params. Training used 165 h of
  16 kHz clean speech through Opus at randomised bitrate, complexity and
  loss. A regression pre-training (10·L_phase + 2·L_env + L_spec) comes
  first, then adversarial + feature-matching fine-tuning. It achieves 92 %
  of the LPCNet resynthesis MOS gain at one fifth the complexity and zero
  delay. Both ship in Opus ≥ 1.5 ([release notes](https://opus-codec.org/demo/opus-1.5/)).
- *FARGAN* (Valin, Mustafa, Büthe, SPL 2024, [arXiv 2405.21069](https://arxiv.org/abs/2405.21069)).
  A framewise autoregressive GAN vocoder with pitch prediction, ~0.6 GFLOPS,
  used for Opus deep PLC and DRED. It runs at under 1 % of a laptop core.
  Precursor: *Framewise WaveGAN* ([arXiv 2212.04532](https://arxiv.org/pdf/2212.04532)).

**David Rowe / FreeDV.** The same domain, a different vocoder.

- [Codec 2](https://github.com/drowe67/codec2). FreeDV 2020 = Codec 2
  features + LPCNet synthesis at 8 kHz ([rowetel.com](https://www.rowetel.com/?cat=3)).
- *RADE: A Neural Codec for Transmitting Speech over HF Radio Channels*
  (Rowe & Valin, WASPAA 2025, [arXiv 2505.06671](https://arxiv.org/abs/2505.06671)).
  FARGAN is the vocoder. Portable C is at [freedv/rade_c](https://github.com/freedv/rade_c)
  and training at [drowe67/radae](https://github.com/drowe67/radae). See also
  *RADE for Land Mobile Radio* ([arXiv 2509.17286](https://arxiv.org/pdf/2509.17286)).
  These replace the vocoder rather than post-filter it. Still, FARGAN is a
  proven CPU-real-time synthesis backbone with public training recipes.
  On-air listening to RADE (2026-09) found that at the end of a
  transmission it sounds, for a moment, like a foreign speaker talking
  quickly. This is speech-shaped babble produced once the input stops being
  speech: a carrier drop, noise, or an unflushed buffer drained too fast.
  The [training design](../design/training.md#the-edges-matter-most-key-up-and-key-down)
  and the runtime's explicit end-of-stream handling are built to prevent
  that failure mode.

**Fraunhofer IIS.** Coded-speech enhancement and bandwidth extension.

- *PostGAN* (Korse et al., ICASSP 2022, [arXiv 2201.13093](https://arxiv.org/abs/2201.13093)).
  A sub-band U-Net GAN post-processor for LC3 at 16 kb/s, ~+20 MUSHRA.
- Mask-based post-filters, zero-delay and light
  ([arXiv 2010.05571](https://arxiv.org/abs/2010.05571),
  [arXiv 2201.12039](https://arxiv.org/pdf/2201.12039)).
- *UBGAN: Enhancing Coded Speech with Blind and Guided Bandwidth Extension*
  (Gupta et al. 2025, [arXiv 2505.16404](https://arxiv.org/abs/2505.16404)).
  A lightweight sub-band GAN that enhances coded wideband speech and also
  extends it to super-wideband, blind or guided by side info. Directly on
  topic for goal 2. The abstract does not state causality **[unverified]**.
- *Parallel Enhancement and Bandwidth Extension of Coded Speech* (Applied
  Sciences 16(3):1439, 2026, [MDPI](https://www.mdpi.com/2076-3417/16/3/1439)).
  Not retrievable in this pass **[unverified]**.

**Others.** *Convolutional Neural Networks to Enhance Coded Speech* (Zhao,
Fingscheidt et al., TASLP 2019, [arXiv 1806.09411](https://arxiv.org/pdf/1806.09411))
gains +0.25–0.82 PESQ across codecs. See also *AMRConvNet*
([arXiv 2008.10233](https://arxiv.org/pdf/2008.10233)).
*Parameter Enhancement for MELP Speech Codec* ([arXiv 1906.08407](https://arxiv.org/pdf/1906.08407))
enhances 2.4 kb/s MELP *parameters* with a DNN. It is the closest thing to
a bitstream-conditioned approach for a vocoder in AMBE's rate class.
[AMBETools](https://github.com/g4klx/AMBETools) drives AMBE-3000 hardware
from the command line. It is GPL, so it is useful as a behavioural
reference for the capture harness but not as code.

## 2. Bandwidth extension and small real-time restoration models

| Model | Year | Notes |
|---|---|---|
| SEANet BWE ([arXiv 2010.10677](https://arxiv.org/abs/2010.10677)) | 2021 | 8→16 kHz, **streaming** variant, 16 ms latency, 1.5 ms per 16 ms frame on one mobile core; feature + adversarial losses |
| HiFi++ ([arXiv 2203.13086](https://arxiv.org/abs/2203.13086)) | 2022 | HiFi-GAN-based unified BWE / enhancement |
| NVSR ([arXiv 2203.14941](https://arxiv.org/pdf/2203.14941)) | 2022 | mel-BWE + vocoder; 99 M params; benchmark [ssr_eval](https://github.com/haoheliu/ssr_eval) |
| VoiceFixer ([arXiv 2109.13731](https://arxiv.org/abs/2109.13731)) | 2021 | general restoration, offline |
| NU-Wave 2 ([arXiv 2206.08545](https://arxiv.org/abs/2206.08545)), AudioSR ([arXiv 2309.07314](https://arxiv.org/pdf/2309.07314)), AERO ([github](https://github.com/slp-rl/aero)) | 2022–23 | diffusion / spectral SR, offline |
| AP-BWE ([arXiv 2401.06387](https://arxiv.org/abs/2401.06387), [code](https://github.com/yxlu-0102/AP-BWE)) | 2024 | parallel amplitude + phase CNN GAN, 18× real time on CPU, non-causal |
| Vocos-based BWE ([arXiv 2603.07285](https://arxiv.org/abs/2603.07285)) | 2026 | 8–48 kHz, RTF 0.005 on an 8-core CPU, non-causal |
| MS-Wavehax ([arXiv 2506.03554](https://arxiv.org/abs/2506.03554)) | 2025 | **causal streaming** vocoder, 2.4 % of HiFi-GAN V1 size, one-frame lookahead |
| HiFi-Stream ([arXiv 2503.17141](https://arxiv.org/pdf/2503.17141)) | 2025 | streaming GAN speech enhancement |

Real-time enhancement backbones, for scale:

- **DeepFilterNet 2/3**: 2.3 M params, RTF 0.04 on a notebook i5, Rust
  runtime with `tract` inference ([repo](https://github.com/Rikorose/DeepFilterNet)).
- **GTCRN** (ICASSP 2024): 24–48 K params, ~35 MMAC/s
  ([repo](https://github.com/Xiaobin-Rong/gtcrn)).
- **RNNoise**: ~60 K params, ~40 MFLOPS.
- **Demucs denoiser**: causal, 33 M params
  ([repo](https://github.com/facebookresearch/denoiser)).
- **FRCRN**: 10 M params, 12 GMAC/s. Too heavy.

Mask-only models cannot *add* content. AMBE's damage is structural, so they
are not enough on their own.

## 3. Architecture candidates under our constraints

The constraints are 8 kHz input and 100 or 400 ms lookahead (see the
design's latency variants). The model must run in CPU real time inside
astar, against structured (not additive) degradation, trained with GAN +
multi-resolution STFT / mel losses. Diffusion is out. Multi-step sampling
defeats the latency budget, and the one-step distillations are large and
non-causal.

1. **NoLACE-style adaptive post-filter, with a small generative residual.**
   Per-frame long-term and short-term filter coefficients are predicted from
   decoded-signal features (cepstrum, pitch, voicing), plus a
   temporal-shaping branch. ~0.6 GFLOPS, zero added delay, phase-preserving.
   It is proven at 6 kb/s SILK, a similar regime, and small enough for
   hand-written Rust CPU kernels. It cannot extend bandwidth by itself.
   **Recommended first.**
2. **Causal HiFi-GAN / Vocos-family generator on a 20 ms hop** (MS-Wavehax or
   streaming SEANet shape). It is conditioned on the degraded waveform and
   extracted features, and trained with HiFi-GAN discriminators + multi-res
   STFT + mel. It handles 8→16 kHz natively. Best quality ceiling, at 5–20×
   the cost of candidate 1.
3. **Two-stage: feature enhancer + FARGAN-style resynthesis.** A causal
   enhancer maps AMBE-output features to clean-speech features, and FARGAN
   synthesises 16 kHz. This is essentially what RADE and Opus PLC do. It is
   excellent at restoring harmonic structure. The risks are speaker-identity
   drift and autoregressive synthesis that is awkward to express in
   burn/candle.

!!! note "Since this survey"
    Candidate 1 was built first. That adaptive filter (2.1 M params) topped
    out near 2.1 predicted MOS. A restorer + waveform synthesiser pipeline in
    the spirit of candidate 3 now leads: predicted MOS 3.57 on a faithful
    input (codec 2.22 on the same clips, clean 4.14), and 4.6 of 5 in a
    blind listening test. It has a Rust runtime (`unamblify restore`). See
    [training](../design/training.md).

## 4. Rust ML tooling (state as of 2026-09)

- **Burn** ([github](https://github.com/tracel-ai/burn)): 0.21 (May 2026).
  Backends: CUDA, ROCm (`burn-rocm` on `cubecl-hip`, Linux, ROCm 6.2–7.x),
  Metal, Vulkan, WebGPU, CubeCL CPU. The LibTorch backend is deprecated.
  ROCm was called experimental in 0.15/0.16. No later maturity statement or
  RDNA3 (gfx1100) test claim was found **[unverified]**. The Vulkan/wgpu
  backend runs on the 7900 XTX without ROCm at all.
- **Candle**: upstream is CPU/CUDA/Metal only. A ROCm PR
  ([#3424](https://github.com/huggingface/candle/pull/3424)) was open as of
  July 2026, tested on RX 7700 XT / 9070 XT and self-described as
  incomplete. Not a training platform yet.
- **tch-rs** ([github](https://github.com/LaurentMazare/tch-rs)): 0.24
  (March 2026) tracks libtorch 2.x. `LIBTORCH_USE_PYTORCH=1` points it at a
  ROCm PyTorch wheel. PyTorch-ROCm officially supports gfx1100, so this is
  the **lowest-risk route to full GPU training on the 7900 XTX today**.
  The exact libtorch version pinned by tch 0.24 is **[unverified]**.
- **ort** ([github](https://github.com/pykeio/ort)): 2.0 RC. ONNX Runtime
  **dropped its ROCm execution provider in 1.23** (AMD points at MIGraphX).
  Fine for CPU inference. `tract` is the pure-Rust alternative DeepFilterNet
  uses.

The practical path is to train with tch-rs on ROCm libtorch (Rust
throughout, as the project wants), export ONNX, and run inference in Rust
via ort/tract or hand-written burn/candle CPU kernels. Burn-on-ROCm or
Vulkan is worth an early spike. If it trains a small model stably on the
7900 XTX, it removes the libtorch dependency entirely.

## 5. Evaluation

- **Intrusive (needs an aligned reference):** PESQ (P.862.2 wideband,
  [python-pesq](https://github.com/ludlows/PESQ)), STOI/ESTOI
  ([pystoi](https://github.com/mpariente/pystoi)), ViSQOL v3
  ([google/visqol](https://github.com/google/visqol)), and LSD (standard for
  BWE). All of them under-rate generative / vocoded outputs. **WARP-Q**
  ([arXiv 2102.10449](https://arxiv.org/abs/2102.10449),
  [code](https://github.com/wjassim/WARP-Q)) was built for exactly that
  case.
- **Reference-rate rule:** the reference is the *clean source recording*,
  never the AMBE output. Score the 8 kHz path with PESQ-NB / STOI against
  the source decimated to 8 kHz. Score the 16 kHz path with PESQ-WB /
  ViSQOL / LSD against the full-band source. Never score a wideband output
  against a narrowband reference, because the added band is penalised.
- **Non-intrusive (for real off-air recordings with no reference):**
  DNSMOS P.835 ([arXiv 2110.01763](https://arxiv.org/abs/2110.01763)) and
  DNSMOS Pro, NISQA ([github](https://github.com/gabrielmittag/NISQA))
  whose *coloration* dimension maps directly onto AMBE's timbre, and
  UTMOS ([github](https://github.com/sarulab-speech/UTMOS22)). These expect
  16 kHz input, so 8 kHz outputs must be upsampled. Always report the raw
  AMBE baseline through the identical treatment.
- The final arbiter is a small MUSHRA / ACR listening test with operators.
  Objective metrics disagree most in exactly this generative regime.
