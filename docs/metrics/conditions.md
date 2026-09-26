# Condition columns: the same measure, split by circumstance

`eval/lsd` averages over everything. These columns re-measure it under
one specific difficulty each, and that is where the differences that
matter show up.

## `eval/lsd_first1s` and `eval/lsd_last1s`

LSD restricted to the **first and last second** of each clip: key-up and
key-down.

A post-filter is most likely to fail at these two points. At the start
the GRU has no history and the codec is still settling, yet the model
has to behave sensibly from its very first frame. That is why training
weights onset examples double and always starts them from a zero state.
At the end the input turns to garbage.

Measured on a converged run: `first1s` **median 10.85, σ 0.17** and
`last1s` **median 11.11, σ 0.14**. Both are close to the whole-clip
figure, so this model is not paying an edge penalty.

Watch for them diverging from `eval/lsd`. A model that averages well but
mangles the edges sounds bad on every over.

## `eval/lsd_rx`: the noisy receiver

The same clips with a **fixed receive-side recipe** applied to the
*input* only: 100 Hz hum at −40 dBFS plus white noise at 25 dB SNR
(`unamblify_audio::rx::RxRecipe::fixed`). The target stays clean. The
question is whether the model can still do its job on a signal that
came off a real radio.

**This is the largest gap in the whole evaluation.** A converged model
scores a median of **15.29** against 11.06 on the clean column. That is
over 4 dB worse, twenty times the difference between the largest and
smallest model this project has trained. Anything that closes it matters
more than anything that shaves LSD.

It is also the second-noisiest column: **range 14.16–17.89, σ 0.67**.
Nothing in the training objective ties the clean and noisy behaviour
together, so the model's noise robustness wanders freely during
training. A single reading is meaningless. One checkpoint had a
near-best `eval/lsd` of 10.99 and the run's *worst* `lsd_rx` of 17.89.

## `eval/lsd_drops`: lost frames

The same measure over clips whose input came from a `drops` sibling
capture: a seeded, fixed pattern of 2–3 consecutive channel frames
muted or repeated, as a real radio link loses them.

The column only appears when a `drops` sibling actually holds one of the
eval clips, so it is absent from runs whose modes have no such capture.

## `@<mode>`: per vocoder

With several modes in one run, **every whole-clip column is also
reported per mode**: `eval/lsd@dstar`, `eval/hnr@codec2-1600`, and so
on.

This is what made the generalist comparison possible. One run can be
set against models trained on each mode alone, mode by mode. It also
reveals differences the mean hides. Codec 2 1600 is consistently the
hardest mode, and the excess buzz a raw decode leaves is more than twice
as large for D-STAR as for Codec 2 3200.

!!! warning "Per-mode columns are only comparable when the clip set is"
    As a capture grows, more eval clips resolve for that mode. This
    project's count went from 19 to 66 mid-project. Columns for modes
    whose clip set did not change stay comparable. The **mean does
    not**, because it averages over a different mixture. The trainer
    logs the clip count at startup, so check it matches before comparing
    two runs.
