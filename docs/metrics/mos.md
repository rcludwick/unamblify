# Predicted MOS: `scripts/mos.py`

**A learned naturalness judge, run offline on a checkpoint's rendered
clips. Scores run 1–5. Clean studio speech scores about 4 and a raw AMBE
decode about 2. It is not a training metric, and it is not gameable by
construction, because it was trained on human ratings rather than on a
statistic we hope tracks them.**

## Why it exists

Every column the trainer writes is a hand-built proxy. Two of them have
read "solved" while the ear disagreed. One was the cepstral
[periodicity](periodicity.md) term, which was gamed. The other was the
[harmonic-to-trough](hnr.md) excess, whose mean reached zero while the
per-frame error stayed at five times its floor. A proxy that correlates
stops correlating once optimised. The defence is a yardstick built from
the thing we actually want.

## What it is

[UTMOS22](https://arxiv.org/abs/2204.02152) (Saeki et al. 2022), the
strong learner from the VoiceMOS Challenge 2022, via the
[SpeechMOS](https://github.com/tarepan/SpeechMOS) package (MIT). It
predicts the mean opinion score a panel of listeners would give a
16 kHz clip for naturalness. It is itself a model with its own biases.
It was trained on synthesised speech, mostly English. Treat it as a
second opinion with a known provenance, not as ground truth.

## How to run it

```
uv venv --python 3.12 .venv-mos
uv pip install --python .venv-mos/bin/python torch torchaudio numpy
.venv-mos/bin/python scripts/mos.py $UNAMBLIFY_DATA/runs/<run>/checkpoints/step-N/audio
```

It needs a separate venv. The predictor needs `torchaudio`, and the
training venv pins `torch==2.13.0` for libtorch, which no `torchaudio`
release matches. The first call fetches the weights into torch's hub
cache. `--json` gives per-clip scores, and `--kinds out` scores only the
model output.

## How to read it

The naturalness run's best checkpoint (`20260912-193149`, step 35 500),
219 clips:

| Kind | all | D-STAR | YSF/DMR | Codec 2 3200 | Codec 2 1600 |
|---|---|---|---|---|---|
| clean | 4.05 | 4.05 | 4.05 | 4.05 | 4.05 |
| codec decode | 1.78 | 1.88 | 2.16 | 1.76 | 1.34 |
| model output | **2.08** | 2.20 | 2.28 | 2.22 | 1.66 |

The model lifts a D-STAR decode by about a third of a point on a
five-point scale, and two full points remain to the recording.
`eval/lsd` fell from ~19 to ~11 over the same models and `eval/hnr`
reached parity, and neither said anything like this. It is also the
first number in the project that ranks the modes the way a listener
does: Codec 2 1600 worst by a distance, YSF/DMR least bad.

### 2026-09-19: the 200 k set, 80 000 steps

`mixed-50k-v4-ysf-heavy-ll5-long` (50 000 utterances per capture set,
`hnr_w` 0.3, mode weights 40 / 30 / 15 / 15), its best checkpoint by this
measure, step 76 000. The eval clips changed with the set, so read these
columns against their own `codec decode` row, not against the table above:

| Kind | all | D-STAR | YSF/DMR | Codec 2 3200 | Codec 2 1600 |
|---|---|---|---|---|---|
| clean | 4.10 | | | | |
| codec decode | 1.79 | 2.11 | 1.98 | 1.73 | 1.33 |
| model output | **2.12** | 2.39 | 2.49 | 2.12 | 1.46 |

That is a lift of +0.33, where the 40 000-step run on the same set
managed +0.12. The lift is largest where the batches were weighted
(YSF/DMR, +0.51). The final checkpoint scored 2.09 and the best by
`eval/hnr_abs` scored 2.07. The best by predicted MOS was neither, which
is why runs now keep every checkpoint (`[ckpt] keep = 40`) and several
are scored. It is still only about a seventh of the way to the
recording. The [2026-09-19 review](../design/training.md#review-2026-09-19)
is about the other six.

This filter (candidate 1) was the ceiling for that architecture at about
2.1. The restorer + waveform synthesiser pipeline (candidate 3,
[training](../design/training.md)) scored **3.57** on a faithful input,
against 2.22 for the codec and 4.14 for clean speech on the same clips
(2026-09-25).

### The judge against a listener (2026-09-22/23)

In a blind listening test (experiment log #36), one listener rated 128
clips: 16 held-out speakers, two chip modes, four systems, level-matched,
one mode per pass. Against those ratings the judge **ranks systems the
way the listener does** (Spearman 0.77 / 0.82 over all clips). It
**hardly ranks clips within a system at all** (0.04–0.09 within the
codec or the filter). It also **compresses the scale**. It scored the
raw codec 0.7–0.9 too high, the clean recording 0.7 too low and the
restorer 1.1–1.3 too low. Every gain this page reports is therefore
smaller than the one a listener hears. It also put the filter model
*below* the raw codec (2.37 vs 2.63 on D-STAR), where the listener put
it above on 22 of 32 sentences. Use it to compare architectures. Use
ears (`scripts/spike/listening/`) to choose between close variants or to
rank a post-filter against the codec it filters.

## Noise floor

Not yet measured across checkpoints. The predictor is deterministic for
a given clip, so the only variance is the model's. Measure it before
reading a difference under ~0.1.
