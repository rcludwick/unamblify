# Training

!!! info "Status"
    2026-09-25: the harness is built. `crates/unamblify-train` (model,
    losses, loaders, trainer, checkpoints, eval) and `crates/unamblify-web`
    (dashboard) sit behind `unamblify train` / `unamblify serve`. The best
    system so far is candidate 3, the restorer + waveform synthesiser
    pipeline ([below](#candidate-3-concretely)). It scores 3.57 predicted
    MOS on a faithful input, against 2.22 for the codec on the same clips
    and 4.14 for clean speech, and a blind listening test put it at 4.6 of
    5. Candidate 1, the adaptive filter (2.1 M parameters), topped out near
    2.1. A Rust runtime for the pipeline exists (`unamblify restore`). The
    candidates are compared in
    [prior art](../research/prior-art.md#3-architecture-candidates-under-our-constraints).

## Training targets

Training must run in four places. Each has a job, and the same Rust code
and the same experiment config must produce the same model on all of them.

| Target | Hardware | Role | GPU stack |
|---|---|---|---|
| Apple Silicon (local Mac) | M-series, unified memory | The inner development loop: smoke tests, data-loader work, seed-set runs, CPU inference profiling. Also where the DVstick is. Train on MPS: the real full model runs at 5.4 steps/s on the M4 Pro's GPU versus 0.16 steps/s on its CPU (34×, measured 2026-09-11). The CPU was faster only for the low frame-rate test, not the 16 kHz upsampling head. `configs/*.toml` default to `device = "mps"` on macOS. | Metal: libtorch MPS via `tch-rs`, `burn` via wgpu/Metal |
| AMD (home Linux box) | Radeon RX 7900 XTX, 24 GB, gfx1100 | The workhorse for long runs on the core set. | ROCm: libtorch ROCm wheel via `tch-rs`, `burn-rocm` (HIP) or Vulkan |
| CUDA (any NVIDIA box) | e.g. RTX 4090 / A100 / H100 | Reference target: the most-tested kernels, and what nearly every cloud offers. | CUDA: libtorch CUDA via `tch-rs`, `burn-cuda` |
| Cloud | rented GPUs | Bursts: sweeps, the biggest models, or when the home box is busy capturing. | The CUDA path in a container. AMD MI300X where offered |

Cloud is not a fifth stack. Every popular GPU-rental service (RunPod,
Lambda, Vast.ai, Modal, AWS/GCP/Azure) hands out NVIDIA cards, so "cloud"
means "the CUDA build, inside a container, with the dataset shards
uploaded to object storage". A few (RunPod, Vultr, Azure ND MI300X v5)
also offer AMD MI300X, which reuses the ROCm build. The cloud recipe is a
`Dockerfile` plus a script that pulls shards from a bucket and pushes
checkpoints back. Which provider to use is a price decision made per run.

### Economics (checked 2026-09-09)

Our models are 1–8 M parameters on 8 kHz audio. A datacentre GPU would sit
idle, so a rented consumer card on a marketplace is the economical choice.

| Option | Price | Fit |
|---|---|---|
| Vast.ai RTX 3090 (24 GB) | from $0.10/h | Enough for candidates 1 and 2, best $/run |
| Vast.ai / RunPod RTX 4090 | $0.27–0.34/h on demand, ~$0.15/h interruptible | 2× a 3090, the sweet spot for sweeps |
| A100 80 GB | $0.50/h unverified, $1.19/h RunPod | Only if a batch needs more than 24 GB |

A full run takes tens of GPU-hours, costing a few dollars on a 4090. A
ten-config sweep is under $50. Cost factors:

- Interruptible needs checkpointing. The cheap rate assumes the box can
  vanish. The harness checkpoints every few minutes to a bucket and resumes
  from it, which the four-target design needs anyway.
- Storage and egress, not compute. Persistent volumes run ~$0.07/GB/month.
  Only shards go up (GBs for the seed, tens of GB for the core set), and
  only checkpoints come back. Raw corpora remain on local storage.
- Unverified hosts are the cheapest on Vast.ai. They are fine for a
  rerunnable sweep and less fine for a headline run. Verified hosts and
  RunPod's rate card cost 30–50 % more.
- Free tiers for smoke tests. Colab / Kaggle T4 hours are enough to
  prove the CUDA build before paying anything.
- The 7900 XTX is the cheapest GPU we have. Cloud earns its place for
  parallel sweeps, or while the home box is busy.

Plan: Vast.ai first (RTX 3090/4090, interruptible, bucket-backed
checkpoints), with RunPod as the managed fallback. Sources:
[getdeploying RTX 4090 pricing](https://getdeploying.com/gpus/nvidia-rtx-4090),
[Spheron RunPod vs Vast.ai 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/),
[Northflank cheapest GPU providers 2026](https://northflank.com/blog/cheapest-cloud-gpu-providers).

## Stack

**Decided 2026-09-09 by three spikes (tch-rs, burn, candle) plus an
adversarial critique: tch-rs 0.26 on libtorch from the PyTorch 2.13.0
wheel**, confined to `crates/unamblify-train`. The backend is chosen at
run time by `[train] device = cpu | mps | cuda:N | rocm:N` (or
`--device`). libtorch ships official CUDA, ROCm (gfx1100) and macOS/MPS
builds, so one Rust training loop covers all four targets.

| Question | Decision | Evidence |
|---|---|---|
| Training stack | tch-rs on libtorch. burn remains a potential alternative for later evaluation. candle is not supported due to lack of ROCm and poor Metal performance. | At the frame-rate conv+GRU shape libtorch CPU trains at 6.3 ms/step vs burn's best 14.2 ms. libtorch has autograd STFT, fused GRU on CUDA/ROCm and fake-quant for QAT, and burn 0.21 has none. |
| Device on the Mac | `cpu` is the default for recurrent models on Apple Silicon. MPS is there for correctness checks and conv-heavy sweeps. | M4 Pro CPU 58 ms/step vs MPS 340 ms at sample rate, 6.3 vs 35 ms at frame rate. |
| Recurrence | Frame rate only (hop 80 samples at 8 kHz). No sample-rate RNN candidates. | Every GPU path is launch-bound at sample rate. |
| STFT losses | `reflection_pad1d(n_fft/2)` + non-centred `Tensor::stft`, magnitude as `sqrt(re²+im²+ε)`. | tch's `stft_center` binding is broken in 2.13, and libtorch's complex `abs` has a NaN gradient at exact zero (silent frames). Unit-tested against a DFT-as-conv reference to 1e-4. |
| Checkpoints | `VarStore::save` → `model.safetensors` + `optim.safetensors` + `meta.json`. | Cross-device round trip verified to 5e-7. |
| Optimiser | A Rust Adam over the VarStore (bias-corrected, global-norm clip 5.0). | tch's `nn::Adam` exposes no state, so exact resume was impossible with it. Grad norms of 3000+ at step 1 made the clip necessary. |
| Export | ONNX is a test fixture (Python mirror + onnxruntime parity, later milestone), not a product path. | No ONNX exporter from Rust in any stack. The product path is safetensors → hand-written Rust inference. |
| Determinism check | Fixed-init and fixed-batch files, loss curves compared within tolerance, never equality. | Seeds differ per device in every stack. |

Every target still runs the same cross-platform check: a fixed seed, a
fixed 200-step run on the seed shards, and the loss curve and final
metrics must agree within a tolerance across CPU, MPS, ROCm and CUDA.
Numerical drift between backends is a real failure mode (different conv
algorithms, different reduction orders). The check is what makes "the
model trained fine on the Mac but diverged on the AMD box" a debuggable
statement rather than a mystery.

### Environment

The train crate is the only crate that links libtorch. `just train-env`
(`scripts/train-env.sh [--gpu cpu|cuda|rocm]`) creates `.venv-torch` with
uv and installs exactly `torch==2.13.0`. It comes from PyPI on macOS
(CPU + MPS) and from `https://download.pytorch.org/whl/{cpu,cu128,rocm6.4}`
on Linux. The script also writes the gitignored `.cargo/config.toml`:

```toml
[env]
LIBTORCH = "<repo>/.venv-torch/lib/python3.12/site-packages/torch"

[target.aarch64-apple-darwin]            # or x86_64-unknown-linux-gnu
rustflags = ["-C", "link-arg=-Wl,-rpath,<that torch dir>/lib"]
```

`LIBTORCH` is used rather than `LIBTORCH_USE_PYTORCH=1` so that no python
needs to be on `PATH` at build time. The rpath lets test binaries load
libtorch without `DYLD_/LD_LIBRARY_PATH`. Do not export `RUSTFLAGS` in
that shell: it replaces the per-target rustflags and drops the rpath. For
builds that do not go through cargo's config, the script prints the
equivalent `LIBTORCH_USE_PYTORCH=1` / `PATH` / `RUSTFLAGS` exports.
`.cargo/config.toml.example` documents both. Without a venv the workspace
does not build. `cargo build --no-default-features -p unamblify-cli`
gives the libtorch-free binary (prepare / capture / shard / verify /
serve / runs / stats).

Data reaches every target the same way: sharded, `mmap`-able files
produced by [stage 4](data-pipeline.md#stage-4-shard),
copied (or synced to a bucket) as a unit. No target reads the raw corpora.

## Inputs and targets

| Tensor | Rate | Source |
|---|---|---|
| degraded PCM | 8 kHz | `captured/<mode>/…wav` |
| channel frames (optional side input, v2) | 50 frames/s | `captured/<mode>/…ambe` |
| target PCM | 16 kHz (decimated to 8 kHz for the narrowband loss) | `prepared/…16k.wav` |

Examples are fixed-length crops (2 s by default, `[data] crop_s`), with
speaker-disjoint splits fixed by the manifest. Two data sources yield
identical batch tensors: `clean16 [B,1,N]`, `deg8 [B,1,N/2]`,
`mask [B,1,N]` (1 where the target is real speech, 0 in garbage-tail
padding), `onset [B]`, `tail [B]`, `speaker [B]` and `mode [B]` (the
example's vocoder mode as an index into the run's `[data] modes`, 0 in a
single-mode run).

- `[data] source = "pipeline"` joins `prepared/manifest.jsonl` with
  `captured/<mode>/manifest.jsonl` (and each sibling set named in
  `[data] kinds`), opens the WAVs per example, applies the recorded lag,
  crops, and augments on the fly (gain jitter ±6 dB, garbage tails).
  This approach generates fresh data but is slower per step. It is suitable for the seed set.
  With several modes it joins each mode's capture with its own lag and
  draws round-robin over the modes (one example of each in turn). Every
  mode then contributes equally to a batch whatever its utterance count.
- `[data] source = "shards"` memory-maps `shards/<name>/NNNN.bin` and
  shuffles by index. It is deterministic and fast, and it is what long
  runs use. The set's own `kinds` decide which captures it holds.
  `[data] modes` must be a subset of the set's `modes`, in any order. The
  loader keeps only those examples and renumbers their `mode` to the
  config's order. Anything else is an error naming the set's modes.

One run, several vocoders. `[data] mode = "dstar"` is a single-mode
run. `[data] modes = ["dstar", "codec2-3200"]` trains one model on both.
The [design index](index.md#decisions-already-made) leaves *one model
per vocoder or one shared model* to experiment, and the experiment is a
config. Crops must be whole frames of every mode. 2 s is, but 0.5 s is
not for Codec 2 1600's 40 ms frames and is refused. The model is told the
mode only with `[model] mode_embed = true`. That adds a learned 16-wide
embedding per mode (`unamblify::MODE_EMBED_DIM`), concatenated to the
context features at every frame before the GRU (its input widens from `F`
to `F + 16`), so the shared network can specialise per vocoder. When it
is off (the default) the `mode` tensor is ignored and the model is blind
to which codec it is hearing. The blind and the conditioned arms are one
config key apart, and `configs/mixed-dstar-codec2-full-ll5.toml` is the
conditioned one. The checkpoint's `meta.json` records `modes` and
`mode_embed`, and a resume must match both.

Both sources resolve an augmented twin's target through its `parent` (the
twin's own clean row is the noisy input). Both also run the `[augment]`
receive-side noise stage over `deg8` after the crop, seeded per example.
See [data pipeline, stage 3](data-pipeline.md#stage-3-augment).

The garbage tails and the lag alignment have one implementation for both
sources (the core crate's `tail` module and `apply_lag` rule), so the two
draw from the same distribution as well as sharing a shape. A cross-crate
test packs the pipeline's examples into shards and checks that the shard
loader returns them bit-for-bit. It also checks that every example
`unamblify shard` packs is a crop the pipeline loader would draw from the
same aligned utterance (onset iff it starts at sample 0, garbage input
over a silent target on a tail).

## Three profiles from one trainer

Inference ships as full (2–8 M params, 16 kHz out, first-class CPUs),
lite (≤ 1 M params, int8, 8 kHz out, mobile ARM) and super lite
(≤ 100 K params, int8, microcontrollers). See
[real-time](realtime.md#profiles). The trainer treats profile as a
width/depth setting (and, for super lite, an architecture variant). It
trains full first, then trains each smaller profile with the next larger
one as a distillation teacher alongside the ordinary losses. Super lite
also needs quantisation-aware training from the start, since int8
activations at that size cannot be a post-hoc step. Every experiment
table reports all profiles that exist. Whether to keep or drop any of
them is decided from those numbers, not in advance.

## Lookahead is a config axis

Each profile is trained as ll5 (5 frames, 100 ms) and ll20 (20
frames, 400 ms). See [real-time](realtime.md#latency-variants). The
architecture is the same. The lookahead sets how many future frames the
encoder may attend to or convolve over, and the training crops are
arranged so the model never sees more than that. A 0-frame control is
trained once per architecture to quantify what lookahead buys. Every
experiment table is therefore profile × variant.

## The edges matter most: key-up and key-down

Field experience with RADE (FreeDV's neural codec) is that the end of a
transmission sounds like a foreign speaker talking quickly for a moment.
It is speech-shaped babble, not what the talker said. This is the
signature of a generative speech model whose input has stopped being
speech. After the carrier drops, the decoder is fed noise, repeated
frames or the tail of an unflushed buffer, and a model trained to produce
plausible phonemes produces plausible phonemes from garbage. The
"quickly" is a buffer being flushed faster than real time. The start of a
transmission has the mirror problem: recurrent state warming up in the
listener's ear.

A push-to-talk QSO is *all* edges, since every over has a key-up and a
key-down. Both are therefore first-class requirements that the trainer
enforces:

Key-down (the RADE failure):

- Garbage-in examples mapped to silence. Every batch includes
  tails where the input turns into what a real receiver produces after
  the carrier drops: AMBE mute/repeat frames, frames decoded from noise
  or from random channel bits, and abrupt truncation. The target is
  silence (or a fast fade of the last real frame). The model learns that
  non-speech in means non-speech out, and it is penalised for inventing
  phonemes.
- Explicit end-of-stream in training. Utterances end with an
  end-of-stream marker exactly as the runtime delivers it. The lookahead
  buffer is flushed with silence padding at the frame rate and is never
  accelerated. The model's tail behaviour under that flush is part of the
  loss.
- A *last-second* metric column (PESQ/STOI/WARP-Q over the final 1 s
  before key-down, plus an explicit babble detector that measures the
  speech-likeness of the output where the target is silence) beside the
  whole-utterance column.

Key-up:

- Onset crops are a fixed share of every batch. At least a third
  begin at a true utterance start or a synthetic key-up (silence →
  speech) with model state reset, not at a random offset.
- Onset-weighted loss on the first 50 frames after a reset.
- Lookahead is a head start. ll5 / ll20 buffer 100 / 400 ms before
  the first output, and training resets state under exactly that regime.
- A *first-second* metric column as above.

The demo page plays the last and the first second in isolation so
both effects can be heard as well as measured.

This system handles edge cases differently than RADE. A post-filter starts
from a decoded signal, and it sits inside a framing layer (astar's
D-STAR / YSF / DMR stacks) that knows *exactly* when a transmission ends
and which frames are invalid. A generative model will still hallucinate
from garbage unless it is taught not to, and the rules above are how it
is taught.

## Model candidates (in order of trial)

1. Adaptive post-filter (LACE/NoLACE shape). Small causal conv/GRU
   feature net → per-frame long-term + short-term filter coefficients +
   temporal-shaping gains. ~1 M params. Narrowband only.
2. Causal generator (MS-Wavehax / streaming SEANet shape) with an
   8→16 kHz head. 2–8 M params.
3. Feature enhancer + FARGAN-style resynthesis. Only if 1 and 2 fall
   short on harmonic structure. (Candidate 1 did, and candidate 3 went
   ahead after experiment #33: see [below](#candidate-3-concretely).)

### Review, 2026-09-19 { #review-2026-09-19 }

After two runs on the 200 k set (experiment log #25, #26) the model is at
2.12 MOS against a clean 4.10 and a degraded 1.79. That is about a
seventh of the gap, and the second run bought its gain by doubling the
steps and tripling `hnr_w`. More of either is not the road to 4.0. The
log had already said why (#16, #20): a per-frame FIR and comb can only
reshape what is there, every spectral loss rewards the easy part, and
every variant ever trained lands inside one narrow band.

What each path can and cannot do, as built:

| Path | Can | Cannot |
|---|---|---|
| FIR, 16 taps per frame | reshape the spectral envelope | put energy where there is none, invert clipping |
| comb at the pitch lag | raise or lower the harmonics | make a periodic signal less periodic |
| noise head | fill the troughs between harmonics, on average | match them frame by frame (#21) |
| 8 → 16 kHz head | restore the upper band | — |
| *(nothing)* | — | restore a plosive (#22), conceal a lost frame (zeros in, zeros out), shape the signal inside a pitch period |

Candidate 1 implementation is incomplete relative to its design. The list
above specifies the LACE/NoLACE shape *with temporal-shaping gains*. The
signal path has none (only the noise head gets per-millisecond gains), so
what exists is the LACE half. The losses also follow only the NoLACE
recipe's first phase, regression. Its second phase is adversarial. The
discriminators for it exist and are shape-tested, but `[train] gan =
true` only logs a warning and carries on. Those two gaps are the two
halves of #16. The next steps are therefore the rest of the plan rather
than new ideas:

1. Wire the adversarial phase (the objective half). Fine-tune from the
   best regression checkpoint with `--init-from`. It is free at
   inference. The discriminators are far larger than the 2.1 M generator,
   so training on MPS slows, and GAN stability under tch-rs on MPS is
   untested.
2. Temporal shaping on the signal path (the capability half). It is
   the smallest nonlinear element and the right tool for plosives, for
   the buzz inside a pitch period, and for what a linear filter cannot
   undo of a clipped waveform. It must pass `unamblify bench`.
3. The erasure mask as a per-frame input. *Built 2026-09-19* as
   `[model] erasure_in`. The shard builder writes the mask for a set that
   draws from a drops sibling. The loader spreads it over the feature
   frames (`Batch.erasure`, `[B, T]`), and the net takes it as one more
   channel into the context conv, not the GRU. That gives it the
   audio's own window, lookahead included, so a gap is seen coming a
   lookahead early instead of only once it has arrived. Eval hands the
   mask to the net on its drops clips, so `eval/lsd_drops` is scored the
   way the model was trained. An empty mask indicates no frame loss. It
   changes the tensor shapes, so a net built without it cannot be the
   `init_from` source. The data for the frame-loss phase is still to do.
   The Codec 2 drops siblings on disk are all `mute` substitution. The
   set to train on uses the decoder's concealment (`--subst erase`), and
   the AMBE modes need a drops pass through the chips. The mask is a
   prerequisite for learning concealment: the framing layer knows exactly
   which frames failed, and the model should be told. The input stays the
   decoder's own concealment output and never raw gaps. That is what a
   receiver hears, and it gives the multiplicative paths energy to work
   with.
4. Candidate 3 stays in reserve until 1 and 2 are measured.

How the data work under way maps onto this: underdriven speech is mostly
a gain problem, which this architecture is good at. Overdriven speech is
partly tractable now and better with step 2. The software-D-STAR sibling
is robustness data. Frame loss needs step 3 and the generative path. The
robotic quality itself needs steps 1 and 2.

One variable at a time. The new corpus (twins, `dstar+perens`) is
trained first on the unchanged architecture as the baseline. Steps 1 and
2 are then A/B'd separately on that fixed corpus. They are judged on
`scripts/mos.py`, with `eval/hnr_abs` and `eval/plosive_*` as the proxies
and `eval/lsd` as neither.

### Candidate 3, concretely (2026-09-20) { #candidate-3-concretely }

The probes of experiment #33 took the review's diagnosis one step
further and changed the order. Wiring the adversarial phase and temporal
shaping into candidate 1 would polish a signal path whose low-band
contribution measures as zero. With a perfect high band the model's
output scores 2.72 and the raw codec 2.76. The defect is in the spectral
magnitudes. The codec's phase is not the problem (natural phase on the
codec's mel: 1.70 against 1.79), and a clean mel through even an
off-the-shelf vocoder is worth 3.75. That is the feature-enhancer-plus-
resynthesis shape. It is two networks with one interface:

| Stage | In → out | Trained on | Carries |
|---|---|---|---|
| Restorer | codec output, 8 kHz (low-band log-STFT + its mel, mode embedding, erasure mask) → clean wideband log-mel, 10 ms hop. Causal conv + GRU at frame rate, the lookahead spent here. | codec pairs: every mode, `dstar+perens`, then the twins, then a drops set with its mask | formant sharpening, de-periodising, bandwidth extension, drive level, concealment. Everything that is a *mapping between features*, which is where a series of training sets plugs in |
| Vocoder | wideband log-mel → 16 kHz waveform. Frame-rate backbone, iSTFT head, a few frames of lookahead. | clean studio speech only, with no chip and no codec, so all of LibriTTS-R, VCTK and LJSpeech rather than what three sticks got through. Common Voice is left out of *this* stage because its recordings score 3.15 themselves and a generator learns its targets' room and microphone | natural excitation and phase. Regression, then the adversarial phase with the discriminators already written |

Then a joint adversarial fine-tune on codec pairs, so the vocoder meets
predicted mels and not only true ones. Everything runs at the frame rate
(about 1 GMAC/s for a 7 M-parameter full profile, against a measured 200 ×
real-time margin). It streams, and it needs only a GRU, convolutions and
an inverse STFT in the Rust runtime.

This interface enables modular development. The vocoder's ceiling is
measurable alone (copy synthesis from true mels). So is the restorer's
(its mel through a reference vocoder). Each new degradation, such as a
badly driven microphone or a lost frame, is more pairs for the restorer,
with nothing to relearn about making a voice.

Order, each step gated on predicted MOS and cheap before dear:

1. *Spike (throwaway, Python, a pretrained vocoder standing in):* a causal
   restorer with 85 ms of lookahead, L1 on log-mel, judged through the
   reference vocoder. Go if it clears the filter model's 2.12 by a margin.
   Done, 2026-09-21–23: go. It scored 2.26 in 85 minutes and 2.68 on
   studio targets (`studio-v1`). A step that was not in this list came
   next: the synthesiser, fine-tuned on the restorer's own blurred output
   with waveform discriminators, reached 3.05 (#34, #35) and **3.57 once
   the eval input was made faithful (#38)**, with Codec 2 3200 at 3.56
   beside the AMBE modes' 3.67. A blind listening test put it at 4.6 of 5
   against the codec's 1.6–1.9 (#36). The scripts are in
   `scripts/spike/`. The weights are kept with the shared data copy
   ([reproducing](reproducing.md)). The spike
   changed two things in the plan. The latency budget is now 500 ms, at
   which the pretrained synthesiser streams unmodified (~400 ms) and step
   3 below needs no *causal* retraining. The vocoder step also became a
   fine-tune of published weights (Vocos, MIT) rather than training from
   nothing.
2. The adversarial phase in the Rust trainer (second optimiser, feature
   matching, checkpointing of both nets). The vocoder and the fine-tune
   need it either way.
3. The vocoder on a clean-only shard set (`shard --corpora`). Gate: copy
   synthesis ≥ 3.6 causal.
4. The restorer in the Rust harness on the pairs. Gate: restorer + vocoder
   ≥ 2.8, per mode, none below its raw decode.
5. Joint fine-tune, then the twins and drops sets. Add an intelligibility
   guard (ASR word error rate of the output against the clean recording's
   own transcript) beside MOS, because a generator can buy naturalness
   with the wrong word and a ham would rather have the right one.

Open defects in the spike were measured with `unamblify measure` on a
faithful input (#37, #38). Sibilants are about 4.5 dB quiet on every mode
(a softer "s"). The 17 dB on Codec 2 reported before came from the
low-passed eval input, not the model. Plosive bursts are 5–8 dB quiet,
with the smear halved against the codec. Neither a band-energy term, a
mean-seeking power term nor the codec's channel bits as a second input
moved the sibilants.

Candidate 1 stays as the `lite` / `super-lite` fallback and as the
baseline every step is measured against. The restorer and synthesiser
run in Rust from an exported weights file, batch and frame by frame:
[the pipeline runtime](pipeline-runtime.md).

Latency (2026-09-21). The budget for this candidate is half a second,
not the filter's 100 ms. It was measured by truncating the future. The
restorer (85 ms) and the synthesiser together use about 300 ms audibly
and 380 ms to the numerical floor. The pretrained synthesiser therefore
streams as it is, frame by frame with cached convolution state, and needs
no causal retraining. Shorter latency is a later trade, bought by
retraining with fewer look-ahead layers.

## Losses

As built (`crates/unamblify-train/src/losses.rs`), regression only in this
milestone, following the NoLACE recipe's first phase:

- Multi-resolution STFT magnitude (n_fft 512/1024/2048 at 16 kHz, hop
  n/4, linear L1 normalised by the target's mean magnitude + log L1) on
  `out16` vs `clean16`, plus mel-L1 (80 bins, 1024/256).
- An SI-SDR term on the 8 kHz decimated path over `mask == 1`, weighted
  0.05. The target is time-aligned, so a waveform term is meaningful
  (unlike TTS).
- The edges rule: per-sample weight `1 + onset_w` (2.0) on the first 50
  frames of onset examples, and `tail_w` (3.0) where `mask == 0` on tail
  examples, whose target there is silence. A *babble penalty* is added
  too: the mean log-energy of `out16` above a −60 dBFS floor over the
  tail, weighted 0.05 so it does not swamp the spectral terms.
- HiFi-GAN discriminators (multi-period 2/3/5/7/11 + multi-resolution
  512/1024/2048) with feature matching exist and are shape-tested behind
  `[train] gan = true`, but the adversarial step is not wired into the
  loop yet.

`metrics.jsonl` carries `loss/total`, `loss/stft`, `loss/mel`,
`loss/sisdr`, `loss/onset`, `loss/tail` (region diagnostics) and
`loss/probe` (the loss on one fixed batch drawn at the start, the number
the smoke test asserts goes down).

## Evaluation

Run every checkpoint through the same script:

| Metric | Reference | Notes |
|---|---|---|
| PESQ-NB, STOI/ESTOI | source decimated to 8 kHz | narrowband path |
| PESQ-WB, ViSQOL, LSD | source at 16 kHz | extension path |
| WARP-Q | either | robust to generative outputs |
| DNSMOS P.835, NISQA | none | also run on real off-air recordings |
| raw AMBE baseline | same treatment | every table shows the baseline |
| per-mode columns | same clips, per mode | a run over several `[data] modes` loads the clip list once per mode (`<clip>@<mode>`) and reports `eval/lsd@dstar`, `eval/lsd@codec2-3200` (likewise `mel`, `sisdr`, `lsd_first1s`, `lsd_last1s`) beside the plain columns, which stay the mean over every clip. A single-mode run has no `@` columns |
| first-second and last-second columns | same metrics over the first / last 1 s, plus a babble detector on the tail | reported beside whole-utterance for every row. The eval appends 1 s of synthetic key-down garbage (noise, a stuck frame, or bursts, seeded per clip) to every degraded eval clip with silence as the target. The rendered `degraded.wav` therefore ends with audio the chip never produced, marked in the dashboard by the dashed line and recorded as `speech_frames` in `spec.json`. |
| `rx` and `drops` columns | LSD, same clips | `eval/lsd_rx`: the input with the fixed receive-side recipe (100 Hz hum at −40 dBFS plus white noise at 25 dB SNR). `eval/lsd_drops`: the input from the `drops` sibling capture (`unamblify augment --kind drops`, a seeded fixed loss pattern per utterance), reported only for the clips that sibling holds. See [data pipeline, stage 3](data-pipeline.md#stage-3-augment). |

There is also a fixed listening set: VoiceBank-DEMAND test speakers, the
LJSpeech clips used in how-ambe-works, and the ham evaluation set.

## Runs, configs, metrics, checkpoints

A run config is a TOML file under `configs/` parsed into
`unamblify::RunConfig`. Every key has a default except `name`
(`configs/default.toml` lists them all). **A key the binary does not
know is an error, not a silent no-op.** A misspelling, or a config
written by a newer build than the one reading it (a dashboard left
running across an upgrade), must never train something other than what
the file says:

```toml
name = "seed-dstar-full-ll5"
[model]   profile = "full"      # full | lite | super-lite
          lookahead = 5         # AMBE 20 ms frames: 5 = ll5 (100 ms), 20 = ll20 (400 ms)
          mode_embed = false    # true: learned per-mode embedding, the model is told the mode
[data]    source = "shards"     # pipeline | shards
          shards = "seed-dstar" # shard set name (shards)
          mode = "dstar"        # the one captured mode (pipeline; also the eval clips'), or
          # modes = ["dstar", "codec2-3200"]   # several, in the model's mode-index order
          crop_s = 2.0
          kinds = ["base"]      # capture sets of the mode for the pipeline source: base, drops, ber
          # mode_weights = [0.3, 0.15, 0.275, 0.275]   # share of the batches per mode (below)
          noise_head = false    # true: the aperiodic excitation path (below)
          noise_mod = false     # with noise_head: envelope modulation + 1 ms gains (below)
          erasure_in = false    # true: the lost-frame mask is a model input; needs a shard
                                # set with a drops kind, and cannot init_from a net without it
[train]   steps = 20000  batch = 16  lr = 2e-4  seed = 1  device = "cpu"
          onset_w = 2.0  tail_w = 3.0  gan = false  amp = false
          periodicity_w = 0.0   # cepstral periodicity term, a diagnostic (below)
          hnr_w = 0.0           # harmonic-to-trough term, the one that works
          transient_w = 0.0     # 1 ms high-band envelope term: stop consonants (below)
          # init_from = "<run-id>"   # start from that run's weights (below)
[ckpt]    every_steps = 500  keep = 5
[eval]    every_steps = 500  clips = "configs/eval-clips.txt"  max_items = 64
[augment] rx_share = 0.3  hum = true  broadband = true  whine = true  colouring = true
```

`unamblify train --config C [--run-dir D] [--resume D/checkpoints/step-N]
[--device X] [--steps N] [--shards NAME] [--init-from SPEC]` writes
`runs/<run_id>/`
(`run_id` = `YYYYMMDD-HHMMSS-<name>` under `$UNAMBLIFY_DATA`):

| File | Content |
|---|---|
| `config.toml` | the resolved config |
| `status.json` | `{status, step, total_steps, started, updated, pid, device, host, best:{metric,value,step}}`, rewritten every 5 s and at checkpoints |
| `metrics.jsonl` | `{"step":n,"t":unix_ms,"k":"loss/total","v":0.123}` per line: the loss keys above, `lr`, `grad_norm`, `sys/steps_per_s`, `sys/samples_per_s`, `sys/cpu`, `sys/mem_gb` (`sys/gpu_util`, `sys/gpu_mem_gb` with the `cuda` / `rocm` features), `eval/lsd`, `eval/mel`, `eval/sisdr`, `eval/lsd_first1s`, `eval/lsd_last1s`, `eval/babble`, `eval/hnr`, `eval/hnr_abs`, `eval/periodicity`, `eval/lsd_rx`, `eval/lsd_drops` when a `drops` sibling holds any eval clip, `eval/plosive_burst`, `eval/plosive_closure`, `eval/plosive_rise` when any eval clip holds a burst, and with more than one mode in the run `eval/<column>@<mode>` for the whole-clip columns. Flushed every 250 ms and at every checkpoint |
| `log.jsonl` | `{seq,t,level,msg}`. The trainer's own lines plus, under the dashboard, the supervisor's (spawned / exited) and whatever else the process printed, copied from `child.log` when it exits |
| `child.log` | under the dashboard only: the child's raw stdout/stderr (the trainer's echo is off there, `UNAMBLIFY_LOG_ECHO=0`) |
| `checkpoints/step-NNNNNN/` | `model.safetensors`, `optim.safetensors`, `meta.json` (step, profile, lookahead in AMBE frames, params, device, seed, `modes`, `mode_embed`, best, and `init_from` when the run started from another's weights). Pruned to `[ckpt] keep`, with the best `eval/lsd` protected |
| `checkpoints/step-NNNNNN/audio/` | per eval clip and mode `<clip>@<mode>.{clean,degraded,out}.wav` (all 16 kHz, sample-aligned) and `<clip>@<mode>.spec.json` (three log-mel spectrograms, 80 bins, shared dB range) for the dashboard's compare panel, whose clip picker shows the mode |
| `samples/step-NNNNNN/` | what `unamblify infer` rendered through that checkpoint on demand from the dashboard's Samples page: `<clip>.out.wav` (16 kHz) and `<clip>.out.spec.json`, plus `infer.log`. Never pruned with the checkpoint |

SIGINT/SIGTERM write a checkpoint (unless the current step's is already
on disk) and exit 0. Any other error lands in `log.jsonl`, and
`status.json` says `failed`. Resume restores model, optimiser and step.
Resuming into the run's own directory first drops any `metrics.jsonl`
rows past the checkpoint, so no step is logged twice. The batches after a
resume come from a stream derived from `(seed, step)` rather than a
replay of the first ones. The probe batch stays the seed stream's first
batch, so `loss/probe` is comparable across the resume. Eval clips are a
fixed list of dev-split prepared keys (`configs/eval-clips.txt`),
rendered whole with a 1 s synthetic garbage tail at every eval step. They
give LSD / mel-L1 / SI-SDR over the speech, LSD over the first and last
second, and the babble energy over the tail.

### Naturalness: the noise head and the periodicity term { #naturalness }

`[model] noise_head = true` adds an aperiodic excitation path. From
the same GRU features it computes `TAPS` shaping coefficients and a gain,
applies them to white noise and adds the result to the narrowband output.
The gain is a fraction of the local signal RMS at frame rate, so the head
learns a *ratio* rather than an absolute level. It starts near zero, so
an untrained noise head leaves the model sounding as it did before.

It exists because the rest of the network can only filter, and filtering
a periodic signal leaves it periodic. Measured on the eval clips, a
D-STAR decode is 3.6 dB more periodic than the speech it was made from
on the frames that matter. The filter-only model still leaves 2.2 dB of
that excess. The theory page
[What the codec destroys](../theory/what-the-codec-destroys.md)
has the table.

`[train] periodicity_w` weights the periodicity term: the absolute
difference between the cepstral peak prominence of the output and of the
target, averaged over speech frames. Every other term compares
magnitudes bin by bin, which a stochastic signal cannot win, so they all
push toward smooth, over-periodic output. This one compares a statistic
and can therefore be satisfied by adding *correctly distributed* noise.
`loss/periodicity` is logged signed (positive = too buzzy) and enters the
total as its absolute value.

`[train] hnr_w` weights the harmonic-to-trough term, and this is
the one to use. It measures how far the output's harmonics stand above
the valleys halfway between them, compared with the target's, in dB over
0–4 kHz on voiced speech frames. The pitch and the bin masks come from
the target and are detached. To lower the term the model must put energy
*into the named bins*, which only the noise head can do.

!!! warning "Implementation details to note"
    Scale. The cepstral term's error is ~0.006 on a batch against a
    total loss of ~3.6, so `periodicity_w = 0.5` contributed **0.08 % of
    the loss** and changed nothing (run `20260912-181010`). Check
    `loss/<term>` against `loss/total` early rather than assuming a
    weight does something. The hnr error is in dB (single digits), so
    weights of order 0.1 are right.

    Gameability. Raised to a weight that did bite, the cepstral term
    was *gamed*: prominence can be lowered by adding ripple at
    neighbouring quefrencies without filling a single trough. In run
    `20260912-183551` the prominence error fell to 0.01 while the
    harmonic-to-trough ratio got worse (24.6 → 26.0 dB) and LSD
    degraded by two points. `periodicity_w` is therefore kept at 0 and
    the column is a diagnostic only.

`loss/hnr` is reported signed (positive = buzzier than the target),
but the total uses `loss/hnr_l1`, the mean of the *per-frame* absolute
error. A model that is too buzzy on half its frames and too breathy on
the other half is not correct, and a signed mean would score it as if it
were. This has happened: the first noise-head run brought the eval mean
to zero while `loss/hnr_l1` went 4.4 → 4.0 dB. Since 2026-09-13 the ratio
is a geometric mean over bins (mean log power over the harmonic windows
against the trough windows), so every harmonic counts once. The
arithmetic version was the loudest harmonics' ratio, and the model
satisfied it by over-filling 0–1 kHz alone. The eval column
`eval/hnr_abs` is the metric-side twin of `hnr_l1`.

`[train] transient_w` weights the transient term: the L1, in dB,
between the output's and the target's 2–3.8 kHz energy envelope at
1 ms resolution, plus the L1 of its slope, over speech frames. The
spectral terms' finest window is 32 ms, so a stop consonant's burst (a
few milliseconds) is a fraction of one frame to them. A model trained on
them alone left the codec's smeared bursts exactly as it found them and
filled the closure before each by 3 dB (`eval/plosive_*`). Unweighted the
term reads ~10 dB, so 0.03 is about a tenth of the total. It needs
`[model] noise_mod` to have anything that can act inside a frame.

`[model] noise_mod` gives the noise path time structure. The noise is
multiplied by the envelope of the periodic path (a 2 ms causal RMS,
normalised to the frame's level) to a learned per-frame depth. Breath
noise rides the glottal cycle, and unmodulated noise on a comb reads as
hiss over buzz. The noise also carries a learned gain per millisecond,
ten per frame, the only quantity in the model that changes inside the
10 ms hop. That is eleven more outputs on the noise head, so it changes
the tensor shapes and is recorded in `meta.json`. Independently of it,
every per-frame parameter is now interpolated across its frame rather
than held.

`[data] mode_weights` sets each mode's share of the batches (relative
weights, one per `[data] modes` entry). Without it the shard loader
draws uniformly, so a mode's share is its share of the set (84 % Codec
2 in `mixed-large`, one AMBE step in six). The pipeline loader draws
round-robin. `[0.3, 0.15, 0.275, 0.275]` gives the two AMBE modes half
the batches without discarding any example the way `shard --balance`
does. YSF/DMR gets the smaller share because its capture was then a
third the size of D-STAR's and would otherwise be seen ten times over in
a run.

The hnr term is not perfectly immune either, since a spectral tilt moves
the harmonic peaks as well as the troughs. On speech-like signals,
though, it answers to inter-harmonic noise about six times more strongly
than to tilt, and tilt is independently expensive under the spectral
terms.

These components are co-dependent. The noise head without the term has no reason
to open its gain, and the term without the head asks for something the
model cannot produce. Both change the tensor shapes or the objective, so
`noise_head` is recorded in `meta.json` and must match to resume or
`init_from`.

### Generalist, then specialists (`[train] init_from`) { #init-from }

A resume continues one run: same config, same optimiser moments,
same step counter, and it refuses a checkpoint whose modes differ.
`[train] init_from` (or `--init-from`) is the other thing: the start of
a new run on an existing run's weights.

| | resume | init_from |
|---|---|---|
| Run directory | the same one | a new one |
| Step counter | continues | starts at 0 |
| Optimiser moments | restored | fresh |
| Metrics / best | continue | start empty |
| Modes | must be identical | this run's must be a subset of the checkpoint's |

`SPEC` is a run id (`20260912-000009-generalist-full-ll5`), which takes
that run's best checkpoint. A pruned best is an error, not a silent
fall back to the newest. `SPEC` can also be `<run-id>:<step>` for a
particular checkpoint, or a path to a `checkpoints/step-N` directory. The
model shape still has to match: same profile, same lookahead, same
`mode_embed`.

This is how the specialists are made. The generalist trains over
every mode with `mode_embed = true`, so it owns one embedding row per
mode. A D-STAR specialist lists `modes = ["dstar"]` and inherits the
generalist's *D-STAR row* (not row 0, not a fresh random row) along with
every other weight. It then fine-tunes on D-STAR data alone, usually at a
lower `lr`, since it starts from a trained net rather than noise. The
run that results is ordinary in every other way. It checkpoints,
evaluates and resumes like any other, and its `meta.json` names the
checkpoint it grew from, so a shipped model can be traced back to its
parent.

A resume always wins over `init_from`. The checkpoint being resumed
already contains whatever the run started from, so re-seeding would
throw away the run's own training.

`unamblify smoke` is the CI gate: lite ll5, 20 CPU steps, batch 4, on six
synthetic utterances. It fails unless `loss/probe` fell.

`unamblify infer --run-dir D --step N --key K [--mode M] [--out P]`
renders one prepared utterance through one checkpoint. It loads
`config.toml` and `checkpoints/step-N/model.safetensors`, then reads
`prepared/<key>.8k.wav` and `captured/<mode>/<key>.wav` for `--mode`
(one of the run's `[data] modes`, the first by default) with the canary
lag applied exactly as the loaders do. It runs the model over the whole
utterance at once on the CPU with no gradient, told that mode (the same
forward pass eval rendering uses). The output is a 16 kHz s16 WAV, twice
the 8 kHz input's length with no garbage tail, plus its `spec.json`
beside it. The default path is `samples/step-N/<clip>.out.wav` (one file
per clip and step, whichever mode rendered last). Every `spec.json` in
the project (checkpoint clips, infer outputs, the Samples page's own) is
written by `unamblify_audio::Spec`: `{rate:16000, n_fft:1024, hop:256,
n_mels:80, db_min:-100, db_max:0, frames, <name>:[frames][n_mels]…}`,
so any two can be drawn on one axis and differenced.

## Dashboard (`unamblify serve [--runs-dir D] [--bind 127.0.0.1:8787] [--token T]`) { #dashboard }

`crates/unamblify-web` is axum 0.8 + tokio + tower-http with SSE, and a
vanilla HTML/JS UI with uPlot 1.6 embedded through rust-embed (no build
step), styled on the docs site's dark palette. The trainer runs as a
child process of the server. The supervisor spawns
`current_exe() train --config … --run-dir …` (so `serve` must be the
`unamblify` binary), watches exit, and finalises `status.json`. Children
never get a pipe. A job is meant to outlive the dashboard and be adopted
after a restart, and a pipe whose only reader has gone turns the child's
next print into `EPIPE`. Their stdout/stderr go to `child.log` in the job
directory. The trainer writes `log.jsonl` itself with its echo off, and
whatever else it printed is copied in when it exits. A capture harness's
`child.log` is followed into its `log.jsonl` as it runs. Every `seq` the
server allocates is read from the file's tail, so `log.jsonl` has one
writer at a time and one sequence. On restart the server adopts runs
whose `status.json` says running with a live pid and marks dead ones
`stopped` (a `queued` run stays queued until it is started). Capture is
supervised the same way (`current_exe() capture --mode M …`). A start is
refused while `status.json` names a live harness the server did not
spawn, before `control.json` is touched.

Pages:

- Runs.
- New run: config picker and overrides.
- Capture: per mode, progress, frames/s, ETA, canary state, current
  utterance, log tail, and Start / Pause / Resume / Stop with in-page
  confirms. The decode-only siblings `captured/<mode>+<kind>/` are listed
  after the modes, read-only, with their status and log.
- Samples (below).
- Run: a uPlot loss panel with every `loss/*` toggleable, LR, sys,
  and eval with first/last-second series. A multi-mode run's
  `eval/<column>@<mode>` series are grouped under their mode in the
  toggles and drawn dashed against the whole-set mean, with `eval/lsd`
  and each mode's on by default. The page also has the log tail, config,
  checkpoints and the compare panel: three canvas spectrograms
  (clean / degraded / out) from `spec.json` on one colormap and dB range,
  an out − degraded / out − clean difference toggle, a waveform overlay,
  `<audio>` for each plus first-second and last-second slices, and a clip
  picker naming each clip's mode.
- Compare: series across runs.
- Host.

Samples is the navigator over everything the chip has been through.
It is `prepared/manifest.jsonl` inner-joined with every capture set's
manifest present: the base modes and the siblings, which appear as
further modes (`dstar+drops`). The join is indexed in memory and rebuilt
when a manifest's size or mtime changes (checked at most once per 2 s, so
a running capture shows up as it goes). On the left are filters (corpus,
split, speaker, mode, key substring), a Random button, and a paged list
(50 per page) of key / speaker / duration / captured-mode chips. ↑ ↓ move
the selection, and space plays or pauses the focused row. On the right
is the selected utterance: key, speaker, split, duration, and per-mode
frame counts (a sibling's with its mutation: `drops 12 frames (mute)`).
An augmented twin also shows its parent (a link) and what was done to it
(the noise clip and SNR, the chain's stages). Below that come rows of
clean (16 kHz, labelled input (twin) for a twin, whose target is
the parent's row), chip output per capture set, and model. The
model row has a run and checkpoint picker with a Render button that
spawns `unamblify infer` through the supervisor (one at a time, while the
page polls until the file exists), then shows the output as a row of its
own. Every row has `<audio controls>`, a min/max waveform strip decoded
through WebAudio, and a spectrogram canvas drawn by the same renderer as
the compare panel (`parseSpec` / `paintSpec`, the viridis LUT) on the
shared dB range. A difference toggle draws chip − clean, model − clean or
model − chip with the compare panel's diverging LUT. Clicking a
spectrogram plays from that instant. The A/B play head toggle makes
every row share one position: starting a row seeks it to the head and
pauses the others, so switching between clean, chip and model compares
the same moment. Sample spectrograms are computed by the server with the
shared `Spec` writer (8 kHz signals upsampled to 16 kHz so all rows share
axes) and cached under `<data-root>/cache/spec/`. Apart from
`cache/host-metrics.jsonl` (see `GET /api/drives` below), that cache is
the only thing the dashboard writes under the data root.

| Route | What |
|---|---|
| `GET /api/health`, `GET /api/sys` | version, host, dirs, devices. CPU / memory / GPU snapshot |
| `GET /api/drives` | every drive with its model, hottest sensor and a note when the enclosure passes no SMART through, plus 24 h of readings for the Host page chart. Each reading has `temps` and, per drive, `io: {r, w}`, the read and write rate in MiB/s over the interval that ended there (cumulative byte counters from `ioreg` on macOS and `/proc/diskstats` on Linux, and a re-plugged drive has no rate for that interval), so heat can be read against the load that caused it. The server samples every 30 s, so a request never runs `smartctl` itself, and `serve` refuses to start when smartmontools is missing. Both histories are appended to `<data-root>/cache/host-metrics.jsonl` and reloaded at startup, so a day of readings survives a restart |
| `GET /api/host` | 24 h of whole-host CPU use, one-minute load average and GPU use, sampled on the same 30 s timer as the drives. GPU comes from the accelerator's `PerformanceStatistics` in the IO registry (no privileges needed). The trainer's own `sys/gpu_*` sampler knows only NVML and ROCm and reports nothing on Apple Silicon |
| `GET /api/configs`, `GET /api/configs/{name}`, `POST /api/configs/validate` | the `configs/` directory |
| `GET|POST /api/runs`, `GET|DELETE /api/runs/{id}`, `POST /api/runs/{id}/stop|resume` | list / create (queue or start) / inspect / stop (SIGTERM → checkpoint) / resume from the latest checkpoint, or start a queued (or never-checkpointed) run from scratch |
| `GET /api/runs/{id}/metrics?keys=&max=&after_step=` | columnar series, min/max-downsampled to ≤ 4000 points |
| `GET /api/runs/{id}/logs?tail=|after=`, `…/config`, `…/checkpoints` | log tail, resolved TOML, checkpoint list with clips |
| `GET /api/runs/{id}/checkpoints/{step}/audio/{file}.wav`, `…/spec/{clip}` | rendered audio and spectrogram JSON |
| `GET /api/runs/{id}/events`, `GET /api/events` | SSE: `metric|status|log|sys|checkpoint` per run, `runs|capture` globally. No replay: the UI refetches what it missed (`metrics?after_step=`, `logs?after=`) on every reconnect |
| `GET /api/capture`, `POST /api/capture/{mode}/{start|pause|resume|stop}` | every mode's `status.json`, one row each with `name`, `mode`, `label` (the mode's name for a person, `VocoderMode::label`: `D-STAR`, `YSF/DMR`, …), `kind`, `readonly`, `alive` and `stopping` (alive with `stop` in `control.json` or a final status, which the page shows without controls). Siblings come after the modes. The POST routes write control.json and start supervised capture, base modes only. `{mode}` also accepts a retired spelling (`ysf-dn`) |
| `GET /api/samples?corpus=&split=&speaker=&mode=&q=&page=&per_page=`, `…/facets`, `…/random?…` | the prepared ⋈ captured join. `mode` is one capture set, or a comma list of sets a sample must all be in (`dstar,dstar+perens` lists what both hold, and the Samples page's second set dropdown sends it). The list returns one page `{total, page, per_page, items:[{key, corpus, speaker, split, duration_s, parent?, modes:[{mode, label, frames, frame_ms}]}]}` where `mode` is a capture set name (`dstar`, `dstar+drops`) and `label` its name for a person (`D-STAR`, `YSF/DMR +drops`). The facets give distinct corpora / splits / speakers / modes with counts (the mode facets carry `label` too). Random gives one row under the same filters |
| `GET /api/samples/{key}`, `…/{key}/audio/{clean8|clean16|degraded-<set>}`, `…/{key}/spec/{which}` | the joined row plus its `UtteranceRow` (with `parent` / `aug` for a twin) and each `CaptureRow` keyed by set name (with `aug` for a sibling). WAV passthrough. The `spec.json` of that signal (computed on the shared axes, cached). Keys contain `/` (and `+` for twins), so these are one wildcard route parsed and validated server-side: `..`, empty components and keys not in the index are refused before the disk is touched |
| `GET|POST /api/samples/{key}/model?run=&step=`, `…/model/{audio|spec}` | `{ready:true, audio, spec}` when `<run>/samples/step-N/<clip>.out.wav` exists, else `{ready:false, running, error?}`. POST `{run, step}` spawns `infer` (202, then poll GET), answering with `infer unavailable` when the server binary lacks the `train` feature |

Security: `serve` binds `127.0.0.1` unless `--token` is given. With a
token, `/api/*` requires `Authorization: Bearer` (constant-time compare)
and the default bind widens to `0.0.0.0`. A non-loopback bind without a
token is refused.

## Real-time budget is a training constraint

Every candidate is profiled on CPU in Rust *before* a long training run:
full on an Apple M-series core and an x86 desktop core, lite on a
Raspberry Pi 4 (Cortex-A72) with int8 weights, and super lite by MAC
count against a 240 MHz Cortex-M7 budget (and on real silicon once the
`no_std` kernel exists). A model that cannot process a 20 ms frame within
its profile's budget is not a candidate for that profile.

`unamblify bench [--profile full] [--width W] [--lookahead N] [--modes M]
[--seconds S] [--passes P] [--threads T]` is that gate. It builds the
model and runs random audio through `Net::forward` on the CPU with the
threads pinned (one by default: the budget is one core, and the radio
needs the rest). It reports parameters, the median milliseconds per 20 ms
output frame, and what fraction of the 20 ms that is. It exits non-zero
if the frame does not fit. It trains nothing and reads no data.
`unamblify bench --kind pipeline` does the same for the restorer +
synthesiser ([the pipeline runtime](pipeline-runtime.md)).

This value represents a lower bound. The whole clip goes through one
`forward`, as in training and `infer`. A deployed post-filter runs frame
by frame carrying the GRU state, which cannot amortise per-call overhead
the same way. Use the number to compare candidates and to rule out the
hopeless, not as a promise about astar's receive path.

Measured on one M-series core, ll5 with the 4-mode embedding
(2026-09-12):

| Model | Parameters | ms / 20 ms frame | Budget |
|---|---|---|---|
| full ×1 | 2 117 130 | 0.10 | 0.5 % |
| full ×1.5 | 4 712 410 | 0.14 | 0.7 % |
| full ×1.75 | 6 394 242 | 0.15 | 0.8 % |

**The full profile is nowhere near its real-time limit on a desktop-class
core**: 206× real time as built, 130× at three times the size. Capacity
is therefore not rationed by the clock here, and `[model] width` exists
to ask whether more of it actually helps. (The lite and super-lite
profiles answer to much slower silicon and are a different question.)

### `[model] width` { #width }

`width` multiplies the profile's channel widths (`model::scaled_widths`,
rounded to a multiple of 8, minimum 8). Parameters grow with roughly its
square, and the profile's parameter budget still applies, so `full ×2` is
refused at 8.4 M against the 8 M budget. The width is recorded in each
checkpoint's `meta.json` and must match to resume or to `init_from`,
since the tensors are different shapes otherwise.
