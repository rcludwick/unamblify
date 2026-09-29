# `eval/lsd`: log-spectral distance

**The headline number. Lower is better. Measured noise floor: 0.12 dB.**

## What it measures

How far the model's spectrum sits from the clean reference's, in
decibels, averaged over the whole clip.

## How it is computed

`unamblify_audio::lsd`:

1. Both signals are cut to their common length and transformed with a
   **32 ms window** (512-point FFT at 16 kHz) and a **hop of a quarter
   window** (128 samples).
2. Every bin of every frame becomes a power in dB, floored at −100 dB so
   silence stays finite.
3. Per frame, take the **RMS** of the per-bin dB differences.
4. Average over frames.

RMS rather than mean absolute error, so one badly wrong bin costs more
than several slightly wrong ones.

## How to read it

| Value | What it is |
|---|---|
| ~19 | a raw AMBE decode against clean, 0–4 kHz |
| 10.8–11.4 | every model this project has trained, 0–8 kHz |
| 10.85 | the best measured (6.4 M parameters, large data) |
| 0 | identical signals |

**A value around 11 is not as bad as it sounds.** The average runs over
*every* time-frequency cell. That includes the quiet ones between
harmonics and the entire 4–8 kHz band the codec never transmitted, where
the model invents plausible content rather than reproducing anything. A 20 dB
error in a cell 60 dB below the speech is inaudible and counts exactly
as much as a 20 dB error in a formant peak.

So it ranks models well and describes quality badly.

## How big a difference is real

Across 41 evaluations of a converged run, on identical inputs:
**median 11.06, range 10.89–11.42, standard deviation 0.12.**

- Under **0.12**: noise. Two models that close are tied.
- **0.25 or more**: probably real. Differences this size showed up
  consistently across modes when they mattered.

This is the tightest column in the suite, which is why it is the default
for selecting the best checkpoint.

## What it cannot see

- **Phase.** It is a magnitude measure. The codec destroyed phase, and
  this metric does not notice.
- **Whether the error is audible.** See above.
- **Buzziness.** A perfectly periodic output and a naturally breathy one
  can score identically. That is what [`eval/hnr`](hnr.md) exists for.
- **Which artefact.** For that, the condition columns
  ([`_rx`, `_drops`, `_first1s`, `_last1s`](conditions.md)) split the
  same measure by circumstance.

Every configuration this project has tried lands between 10.7 and 11.4.
That spread is only slightly wider than the measurement noise, which
suggests LSD has stopped discriminating between the models that remain.
