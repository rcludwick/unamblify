# `eval/hnr` and `eval/hnr_abs`: harmonic-to-trough excess

**The robotic-quality pair. `eval/hnr` is the mean excess over clean,
dB. 0 is the goal, positive is buzzy, negative is breathy.
`eval/hnr_abs` is the mean *per-frame* absolute excess. It has to be
near 0 too, or the mean is a coincidence. Noise floor of the mean: 0.48
dB on the old scale (see below).**

## What it measures

How far the output's harmonics stand above the valleys *between* them,
compared with how far the clean target's do.

Those valleys are the point. In real voiced speech they are not empty.
They hold breath noise, cycle-to-cycle jitter and the aperiodic part of
the glottal source. That content is a large part of what makes a voice
sound like a person. A vocoder that models voiced bands as exactly
periodic empties them, and [no amount of
filtering](../theory/what-the-codec-destroys.md) puts them back.

## How it is computed

`unamblify_audio::hnr_excess_stats(output, clean)`:

1. Frame at **64 ms with a 16 ms hop** (1024-point FFT at 16 kHz).
2. Find the **clean**'s pitch by autocorrelation of the frame. Keep the
   frame only if the normalised peak is ≥ 0.45 and `f0` lands in
   60–320 Hz. Unvoiced frames have no harmonic comb to measure.
3. For each harmonic `k·f0` below **4 kHz** (the band a narrowband
   vocoder actually transmits), take the **peak** power within ±0.18·f0
   and the **median** power in the same window around `(k + ½)·f0`.
   Their ratio in dB is the harmonic's value.
4. Require at least 6 harmonics, then the frame's value is the **mean
   of the per-harmonic dB ratios**.
5. Measure both signals on the clean's frames with the clean's pitch.
   The per-frame difference is the excess. `eval/hnr` is its mean over
   frames, `eval/hnr_abs` the mean of its absolute value.

The measure uses peak against median rather than mean against mean.
Otherwise the skirts of a harmonic leak into the window and wash the
contrast out.

**Every harmonic counts once.** Until 2026-09-13 the frame value was
`10·log10(mean of the peaks / mean of the troughs)`. The first few
harmonics are tens of dB louder than the rest and dominated it so
completely that the number was really the 0–1 kHz band's. A model
satisfied it by over-filling that band alone (−1.1 dB at 0–1 kHz, +0.1
at 3–4 kHz) and read as natural. Averaging per-harmonic dB ratios weights a quiet
harmonic at 3.5 kHz the same as the loud one at 200 Hz. The training
loss made the same change (a geometric-mean ratio, see [loss
terms](loss-terms.md)).

## How to read it

Measured on this project's D-STAR eval clips, per-harmonic scale
(2026-09-13, the noise-head run's best checkpoint):

| Signal | absolute | `eval/hnr` (mean excess) | `eval/hnr_abs` |
|---|---|---|---|
| Clean speech | **9.7 dB** | 0 by definition | 0 |
| Raw D-STAR decode | 13.1 dB | **+3.4** | 4.1 |
| Filter-only model | 11.3 dB | +1.5 | 3.0 |
| With the noise head and `hnr_w` | 8.9 dB | −0.9 | **2.6** |

Absolute values are lower than on the old scale (clean read 18.7 there)
because the quiet high harmonics, whose ratio is small, now count.

The two columns say different things. The codec is buzzy on average
*and* frame by frame. The noise-head model has crossed parity on
average and is now slightly breathy. Its per-frame error is still
**2.6 dB against a floor of about 0.75** (two realisations of the same
noise measure that far apart). Per frame, its excess runs from −5.3 dB
(p10) to +2.8 dB (p90). It adds a roughly constant share of noise
instead of following the target's breathiness. It overshoots the most
periodic vowels and undershoots the breathy frames. Regressing its
per-frame ratio on the clean's gives a slope of 0.82, where 1.0 would be
tracking. That is what `hnr_abs` is for. **A mean near zero with a large
absolute error is a coincidence of signs, and it does not sound
natural.**

Negative would mean too breathy. That is also wrong, and the
[loss](loss-terms.md) penalises it symmetrically.

## Both signals are measured on the *same* frames

`hnr_excess(a, b)` takes the voiced frames from **`b`, the reference**,
and measures both signals on those frames using the reference's pitch.

This matters more than it sounds. If each signal chose its own voiced
frames, a buzzier signal would pass the voicing test on *more* of them.
That includes frames where the reference is only weakly periodic and its
own harmonic-to-trough ratio is low. The comparison would then flatter
whichever signal is already the more periodic. Measured both ways on
the old scale, a D-STAR decode read **+10.2 dB** with a per-signal gate
and **+3.63 dB** gated on the clean reference. The first number was
inflated by frame selection. The second is the comparison you meant to
make.

The training loss has always gated on the target, so this also makes the
metric and the objective agree.

!!! warning "Two scale changes"
    `eval/hnr` values from runs before 2026-09-12 used a per-signal
    voicing gate (~3× inflated). Values from runs before 2026-09-13 used
    the mean-of-peaks ratio. Neither is comparable with later runs.
    `eval/hnr_abs` and the plosive columns exist only from 2026-09-13.

## How big a difference is real

39 evaluations of the naturalness run past step 8 000, old scale:
**median 3.66, range 2.47–4.94, standard deviation 0.48.** The spread
is what matters. The per-harmonic scale has not been measured across
checkpoints yet, so assume the same until it has.

That is four times noisier than LSD. **A single reading is not a
result.** This column has already misled the project twice inside the
same run, once optimistically and once pessimistically. Use the
median of several evaluations. Part of the spread is the noise head
itself: each evaluation draws fresh noise, so two renders of one
checkpoint differ.

## What it cannot see

- **Unvoiced speech.** Fricatives and silence are skipped entirely. The
  measure only applies where there is a harmonic comb.
- **Whether the noise is the right *kind*.** Breath noise in real
  speech rides the glottal cycle. Unmodulated noise on a comb fills the
  same troughs and sounds like hiss over buzz. `[model] noise_mod` exists
  because of this. Only listening (or the [MOS predictor](mos.md)) can
  tell the two apart.
- **It is not perfectly immune to spectral shaping.** A tilt moves the
  harmonic peaks as well as the troughs. On speech-like signals it
  answers to inter-harmonic noise about **six times** more strongly than
  to tilt. Tilt is also expensive under the spectral terms. Still, "six
  times" is not "not at all".

For the differentiable version used in training, and why the cepstral
alternative was abandoned, see [loss terms](loss-terms.md) and
[`eval/periodicity`](periodicity.md).
