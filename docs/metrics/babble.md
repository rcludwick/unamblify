# `eval/babble`: speaking into silence

**Lower is better, in dBFS. Around −79 is the model correctly saying
nothing. Measured noise floor: 7.47 dB, by far the noisiest column.**

## What it measures

How loudly the model talks when its input has stopped making sense.

When a transmission ends, the vocoder keeps decoding whatever the
receiver hands it, and what comes out is babble that sounds like a fast
foreign speaker. A post-filter can make this *worse* by confidently
"restoring" nonsense. The right answer is silence.

## How it is computed

Every eval clip gets **1 second of synthetic garbage appended** (the
same generator the training tail examples use, seeded per clip). The
column is the mean frame energy of the model's output over that region:

1. 10 ms frames (160 samples at 16 kHz).
2. `10·log₁₀(mean power + 1e−10)` per frame.
3. Mean over the tail frames.

The training side pairs it with a target of **silence** over the tail,
weighted `tail_w` (3.0), plus an explicit babble penalty. See
[loss terms](loss-terms.md).

## How to read it

| Value | What it is |
|---|---|
| −100 | the floor: digital silence |
| ~−79 | typical for a converged model, which is correctly quiet |
| −60 and up | audible babble: the model is inventing speech from noise |

## How big a difference is real

41 evaluations of a converged run: **median −78.93, range −95.48 to
−62.75, standard deviation 7.47.**

That range is enormous, and it is inherent to the measure. The quantity
is the energy of something that should not exist, so it swings by tens
of dB depending on whether any frame of the garbage happened to trigger
the model. **Never compare two models on a single reading.** Use
the median over many evaluations, and treat a sustained rise above −65
as the signal worth acting on.

## Why it has its own column

This artefact is the one listeners complain about most in neural speech
systems on the air. FreeDV's RADE is known for a second of fast
gibberish at the end of a transmission. Averaging it into LSD would hide
it completely, since the tail is one second out of many and the target
there is silence.

The clip audio each checkpoint writes includes the tail. The dashboard
marks the boundary with a dashed line, so what the column measures can
also be heard directly.
