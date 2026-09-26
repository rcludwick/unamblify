# Loss terms and training curves

What the optimiser is actually minimising, written to `metrics.jsonl`
every step. These are **not** the evaluation columns: they are computed
on 2-second training crops rather than whole held-out clips, and some
use different units from their eval counterparts.

## The total

```
total = stft + mel + sisdr_term + babble_w·babble
        + periodicity_w·|periodicity|        (0 by default)
        + hnr_w·hnr_l1                       (0 by default)
        + transient_w·transient              (0 by default)
```

`loss/total` is what is back-propagated. Watch it for divergence. For
actual progress, watch `loss/probe` (the same fixed batch every time),
since `loss/total` moves with batch composition.

| Key | What it is |
|---|---|
| `loss/stft` | multi-resolution STFT magnitude error, windows 512/1024/2048, linear + log magnitude L1 per frame |
| `loss/mel` | 80 mel bands on the 1024/256 STFT, **natural-log** energies (so ~2.3× smaller than `eval/mel`, which is in dB. Do not compare them) |
| `loss/sisdr` | `−0.05 × SI-SDR(dB)` on the 8 kHz decimated path, over the speech region only |
| `loss/onset` | diagnostic: the unweighted spectral terms over the onset region |
| `loss/tail` | diagnostic: the same over the garbage tail, plus the weighted babble penalty |
| `loss/periodicity` | signed cepstral prominence error ([a diagnostic, never a target](periodicity.md)) |
| `loss/hnr` | signed harmonic-to-trough error in dB. Positive means buzzier than the target. A geometric-mean ratio (mean log power over the harmonic bins against mean log power over the trough bins), so every harmonic counts once. See below |
| `loss/hnr_l1` | mean of the **per-frame** absolute dB error. This is what the total actually uses |
| `loss/transient` | the stop-consonant term, dB: the L1 of the 2–3.8 kHz envelope at 1 ms resolution against the target's, plus the L1 of its slope, over speech frames. See below |
| `lr` | learning rate |
| `grad_norm` | gradient norm before clipping. A spike here precedes a divergence |
| `sys/steps_per_s`, `sys/samples_per_s`, `sys/cpu`, `sys/mem_gb` | throughput and machine load (`sys/gpu_*` with the CUDA/ROCm features) |

## The edges rule

Two regions carry extra weight, because they are where a post-filter
does the most damage:

- **Onsets.** The first 50 frames (500 ms) of onset examples count
  double (`onset_w`). Every example starts from a zero GRU state, which
  is exactly what the runtime sees at key-up.
- **Tails.** Where the target is silence the weight is `tail_w` (3.0).
  A **babble penalty** is added on top: the mean output energy over the
  tail in dB above a −60 dBFS floor, weighted 0.05. A tail at
  −20 dBFS costs about 2, comparable to the spectral terms.

## Signed versus absolute

`loss/hnr` is reported signed so you can see *which way* the model is
wrong. The total uses `loss/hnr_l1`, the per-frame absolute error.
The distinction matters. A model that is too buzzy on half its frames
and too breathy on the other half has a signed mean near zero and is
not correct. Using the signed mean as the objective under-reported
the error by about three times.

## Every harmonic counts once

The harmonic-to-trough term divided the mean of the harmonic-bin powers
by the mean of the trough-bin powers until 2026-09-13. The first few
harmonics are tens of dB louder than the rest, so that ratio was the
0–1 kHz band's. The model that trained under it fixed that band alone
(it over-filled it to −1.1 dB), while 2–4 kHz stayed where the filter
had left it. The term now takes the mean *log* power over each set of
bins and subtracts. The result is a geometric-mean ratio in which a
quiet harmonic at 3.5 kHz weighs the same as the loud one at 200 Hz. The
[metric](hnr.md) averages per-harmonic dB ratios for the same reason.

## The transient term

A stop consonant's burst is a few milliseconds long, while the finest
window in the spectral terms is 32 ms. To them a burst is a fraction of
one frame. The L1 against its magnitude is minimised by something smooth
at the average level, and that is what the codec already delivers. A
model trained on the spectral terms alone left the codec's smeared
bursts exactly as it found them. It also filled the closure before each
burst by 3 dB ([`eval/plosive_*`](plosive.md)).

`transient_w` weights a term that looks at the **2–3.8 kHz energy
envelope at 1 ms resolution**, in dB. It charges the L1 of the envelope
and of its slope against the target's, over speech frames. The level
term sees a filled closure and a smeared tail. In the log domain a
−40 dB closure filled to −28 dB costs as much as a burst 12 dB short.
The slope term sees a slow rise. The band is inside every mode's, so the
bandwidth-extension head cannot pay it off. Unweighted it reads about
10 dB, so `transient_w = 0.03` puts it near a tenth of the total. It
also needs somewhere to act. With `[model] noise_mod` the noise path
carries a gain per millisecond, the only thing in the model that can
change inside a 10 ms frame.

## Choosing a weight

Both naturalness terms are off by default. Turning one on takes more
than picking a number, and two runs were lost to getting it wrong:

1. **Check the scale before trusting a weight.** The cepstral error is
   ~0.006 per batch against a total of ~3.6, so `periodicity_w = 0.5`
   contributed **0.08 %** of the loss and did nothing at all. Read
   `loss/<term>` against `loss/total` in the first few hundred steps.
2. **Check the term is not gameable.** Raised to a weight that did bite,
   that same term was optimised sideways. See
   [`eval/periodicity`](periodicity.md).

The harmonic-to-trough error is in dB (single digits), so `hnr_w = 0.1`
puts it at about a tenth of the total, which is enough to steer without
overruling the spectral terms. Steering the *mean* is not the same as
steering every frame, though. Under that weight `loss/hnr_l1` fell only
from 4.4 to 4.0 dB over 40 000 steps while the eval mean crossed zero.
Watch `hnr_l1`, not `hnr`.

## The adversarial path

HiFi-GAN-style discriminators (multi-period and multi-resolution) are
implemented behind `[train] gan = true` and are **off**. Only their
shapes are unit-tested. The adversarial loss is not wired into the
training loop. GAN training is the usual way to gain perceptual
sharpness, and it tends to bring hallucination with it. Whether it helps
is a question for a listening test rather than a metric.
