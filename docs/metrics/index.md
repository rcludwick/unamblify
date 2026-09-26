# Metrics

Every number the trainer writes, one page each: what it measures, how it
is computed, what a good value looks like, **and how big a change has to
be before it means anything**. That last part matters. Two of these
columns are noisy enough that a single reading is worthless, and
mistaking noise for progress has cost this project real training runs.

For the practical side (which columns to trust, the traps, and what
none of them capture), start with
[Reading the numbers](../theory/reading-the-numbers.md).

## The evaluation columns

Written every `[eval] every_steps` (500 by default) over a fixed list of
held-out clips. Each is also reported per vocoder mode as
`<key>@<mode>` when a run covers several.

| Column | Measures | Good | Noise floor |
|---|---|---|---|
| [`eval/lsd`](lsd.md) | spectral distance from clean, dB | lower. ~10.9 is the best so far | **0.12** |
| [`eval/mel`](mel-l1.md) | the same on 80 perceptual bands, dB | lower, ~5.6 | **0.12** |
| [`eval/sisdr`](si-sdr.md) | waveform distortion, dB | higher. ~−20 is normal | **0.84** |
| [`eval/hnr`](hnr.md) | **how buzzy** on average, dB above clean | **0**. D-STAR decode +3.4, filter-only +1.5, noise head −0.9 | **0.48** (old scale) |
| [`eval/hnr_abs`](hnr.md) | the per-frame absolute excess behind it, dB | **0**. Codec 4.1, noise head 2.6, floor ~0.75 | — |
| [`eval/periodicity`](periodicity.md) | cepstral buzz proxy (*diagnostic only*) | 0 | 0.00 (no dynamic range) |
| [`eval/plosive_burst`, `_closure`, `_rise`](plosive.md) | **stop consonants**: burst level, closure depth, rise time vs clean | **0**. Both models ≈ −13 dB, −15 dB, +11 ms | — |
| [`eval/babble`](babble.md) | speaking into silence, dBFS | lower, ~−79 | **7.47** |
| [`eval/lsd_rx`, `_drops`, `_first1s`, `_last1s`](conditions.md) | LSD under noise, frame loss, and at the edges | lower | 0.67 / - / 0.17 / 0.14 |
| [sibilants](sibilant.md) (`unamblify measure`, standalone) | 4–8 kHz level and high-to-mid balance at the clean's sibilant frames, dB | 0 = as bright as the recording. The restorer pipeline is about −4.5 on every mode (2026-09-25, faithful input) | — |
| [predicted MOS](mos.md) (offline, `scripts/mos.py`) | a learned naturalness judge, 1–5 | higher. Restorer pipeline **3.57** against codec 2.22 and clean 4.14 (2026-09-25, faithful input). Filter model 2.12 against clean 4.10 and codec 1.79 (2026-09-19). The earlier clips gave 4.05 / 1.78 / 2.08 | — |

The noise floor is the standard deviation across evaluations of a
*converged* run on identical inputs. It is the amount a column moves
when nothing is changing. A difference smaller than that is not a
result.

## The training curves

| Key | Page |
|---|---|
| `loss/total`, `loss/stft`, `loss/mel`, `loss/sisdr`, `loss/onset`, `loss/tail`, `loss/hnr`, `loss/hnr_l1`, `loss/transient`, `loss/periodicity`, `lr`, `grad_norm`, `sys/*` | [Loss terms and training curves](loss-terms.md) |

## Where they live

The pure-DSP measures are `crates/unamblify-audio/src/metrics.rs` and
`plosive.rs` (no libtorch, unit-tested against synthetic signals). The
evaluation harness that renders clips and aggregates them is
`crates/unamblify-train/src/eval.rs`. The training losses are in
`crates/unamblify-train/src/losses.rs`. Every number here can be
recomputed from the WAVs each checkpoint writes.
