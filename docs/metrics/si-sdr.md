# `eval/sisdr`: scale-invariant signal-to-distortion ratio

**Higher is better. Normally around −20 dB here, and that is fine.
Measured noise floor: 0.84 dB.**

## What it measures

How much of the output is the target waveform and how much is
everything else. It is the only measure in the suite that compares
*waveforms* rather than spectra.

## How it is computed

`unamblify_audio::si_sdr`, the standard formulation:

1. Remove the mean from both signals.
2. Project the estimate onto the reference to find the best scale
   `α = ⟨e, r⟩ / ⟨r, r⟩`. This is what makes it *scale-invariant*. A
   model that gets everything right but half as loud is not penalised.
3. `target = α·r`, `noise = e − α·r`.
4. `10·log₁₀(‖target‖² / ‖noise‖²)`.

## How to read it

**Do not read the absolute value as quality.** Around −20 dB looks
catastrophic by the standards of speech enhancement, where +10 dB is
ordinary. But those systems start from a noisy copy of the *same
waveform*. Here the vocoder discarded phase and re-synthesised the
speech, so the output waveform cannot resemble the target waveform even
when the two sound identical. A negative SI-SDR is the expected
consequence of the task, not a defect.

| Value | What it is |
|---|---|
| −19.9 | median of a converged model here |
| −22.4 to −18.7 | its full range |
| large negative jump | a real problem: misalignment, a scale blow-up or instability |

It is useful for **time alignment and gross errors**. If the model
drifts, inverts or blows up, this column moves first and
furthest.

## How big a difference is real

41 evaluations of a converged run: **median −19.93, range −22.43 to
−18.67, standard deviation 0.84.**

That is seven times noisier than LSD, so a difference under ~1.7 dB
between two models means nothing. Use it as an alarm, not a ranking.

## What it cannot see

Anything perceptual. Two outputs that sound equally good can differ by
several dB here purely through phase, and two that differ audibly can
score the same. It is in the loss (weighted 0.05) to keep the output
anchored in time, not to make it sound better.
