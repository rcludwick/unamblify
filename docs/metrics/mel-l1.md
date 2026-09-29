# `eval/mel`: mel-band L1

**Lower is better. Measured noise floor: 0.12 dB.**

## What it measures

The same idea as [`eval/lsd`](lsd.md), but on 80 perceptually spaced
bands instead of raw FFT bins. The error is weighted roughly the way
hearing weights it.

## How it is computed

`unamblify_audio::mel_l1`:

1. A **64 ms window** (1024-point FFT at 16 kHz), hop a quarter of that
   (256 samples).
2. Power through an **80-band mel filterbank**, each band in dB
   (`10·log₁₀`), floored.
3. **Mean absolute difference** per band per frame, averaged.

There are two deliberate differences from LSD. Mel bands are wide at
high frequencies and narrow at low ones, matching the cochlea. And the
error is L1 rather than RMS, so single bad bins dominate it less.

!!! note "The loss uses a different log"
    The `loss/mel` term in training uses **natural-log** band energies,
    not dB. The two numbers therefore differ by a factor of ~2.3 and are
    not comparable. Do not read `loss/mel` against `eval/mel`.

## How to read it

| Value | What it is |
|---|---|
| ~5.6 | typical for a converged model here |
| 5.40–5.92 | the full range across a converged run |
| 0 | identical signals |

## How big a difference is real

41 evaluations of a converged run: **median 5.60, range 5.40–5.92,
standard deviation 0.12.**

In practice this column **moves less and discriminates less** than LSD.
Across every model this project has trained (315 K to 6.4 M parameters,
four vocoder modes, two lookaheads), mel-L1 stayed inside a band barely
wider than its own noise. It confirms LSD rather than adding to it.

## What it cannot see

It misses everything LSD misses ([phase, audibility, buzziness](lsd.md)),
and it is coarser by construction. The 80 bands smooth over exactly the
fine spectral structure whose absence makes a vocoder sound synthetic. A
model that fills the troughs between harmonics and one that does not can
score the same mel-L1, because both put the same total energy in each
band.

That blind spot is the reason [`eval/hnr`](hnr.md) had to be added.
