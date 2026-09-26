# `eval/periodicity`: cepstral buzz proxy

**A diagnostic and a cautionary tale. Never optimise it directly.**

## What it measures

Cepstral peak prominence of the output minus that of the clean target.
Prominence is how sharply the pitch peak stands out of the cepstrum.
It rises when a spectrum is a cleaner harmonic comb, so in principle it
tracks the same buzziness that [`eval/hnr`](hnr.md) measures, more
cheaply.

## How it is computed

`unamblify_audio::cpp_excess`, per frame with enough energy:

1. Log power spectrum of a **64 ms** frame.
2. Real cepstrum, restricted to the quefrency band for 60–320 Hz
   (50–266 samples at 16 kHz).
3. **Peak in that band minus the band's mean.** This is the prominence.
4. Average the difference between output and target over frames.

## Why it is only a diagnostic

Two independent failures, both measured here:

**It has almost no dynamic range on real speech.** Over 39 evaluations
of a converged run its median was **0.01 with a standard deviation of
0.00**, while `eval/hnr` over the same evaluations sat at 3.66 with real
movement. A model whose output is audibly buzzy and one that is not
score the same to two decimal places.

**It can be satisfied without changing anything that matters.** Cepstral
prominence falls if you add ripple at *neighbouring* quefrencies, which
requires putting no energy whatsoever between the harmonics. Weighted at
50 in the loss, the model found exactly that shortcut. The prominence
error fell to 0.01 while the harmonic-to-trough ratio got **worse**
(24.6 → 26.0 dB) and LSD degraded by two points. The run was abandoned
and the weight set back to 0.

This is Goodhart's law. A proxy that correlates with the target under
normal conditions stops correlating once you start optimising it.

## What it is still good for

It is cheap, and **its disagreement with `eval/hnr` is informative**.
That disagreement is what caught the gaming. Keeping a proxy alongside
the real measure costs nothing and gives a second opinion.

`[train] periodicity_w` remains in the config, defaults to 0, and should
stay there.
