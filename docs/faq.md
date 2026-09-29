# FAQ

This document records questions and answers from the project development.
Numbers quoted were measured locally; refer to the [experiment log](theory/experiment-log.md)
for methodology.

The answers under "The model" describe the adaptive filter (candidate 1).
The current best system is the restorer + waveform synthesiser pipeline:
3.57 on [predicted MOS](metrics/mos.md) against the filter's ceiling near
2.1 (see [Training](design/training.md#candidate-3-concretely)).

## The model

### Would a bigger model help?

No. The full profile sits at the knee of the curve. Measured
on identical data, 315 K parameters scores 11.15, 2.1 M scores 10.89 and
6.4 M scores 10.85. Tripling the size is worth 0.04, which is inside the
measurement noise. Shrinking it sevenfold costs 0.26.

It is not a real-time limit either. The 6.4 M model uses 0.8 % of the
20 ms frame budget on one M-series core. The capacity is affordable but provides no benefit. What moved the number was more data. 13× the
training set was worth 0.20, five times the gain, and it costs nothing at
inference. See [Capacity](theory/what-the-experiments-showed.md#capacity).

### Would a second model, cleaning up after the first, do better?

Usually no. A cascade adds more capacity
arranged in series, and it is worse-conditioned than one wider network.
Stage 2 sees only stage 1's output, so whatever stage 1 destroyed is gone
and whatever it invented now looks like evidence. Since 3× the parameters
in *one* net bought 0.04, two nets are unlikely to do better.

The deeper point is what the residual damage turned out to be. It was a
capability gap: the architecture could only filter, and filtering
cannot add noise between harmonics. It was also an objective gap:
magnitude-matching losses prefer smooth output. A second identical stage
inherits both and fails the same way.

A second model earns its place only when it can do something the first
cannot. Examples are an *adversarially* trained refiner (a different
objective), a stage with much longer context (different information), or
a cleanup stage for the noisy-receive condition. For the adversarial case
the best argument is risk isolation more than quality. GAN training
is unstable, and keeping it in an optional second stage means the safe
model is always there to ship.

### How big is the model in memory?

The full profile is 2.1 M parameters, an 8.1 MB checkpoint at fp32.
Lite is 315 K (~1.2 MB) and super-lite ~39 K. Quantised to int8 those
shrink roughly fourfold, which is what the lite and super-lite
[profiles](design/realtime.md) assume. Optimiser state doubles the
on-disk size during training (16 MB) but never ships.

### Can models be combined, or must the whole dataset be trained at once?

Both work. Which to use depends on what you are changing.

- Training a fresh model on a bigger mixture is the reliable path.
  It produced every improvement so far.
- `[train] init_from` starts a new run from another run\'s weights,
  at step 0 with a fresh optimiser and its own metrics. It is how a
  specialist starts from a generalist. It re-indexes the mode embedding
  so a single-mode model inherits the parent's row for *its* mode.
- Resuming continues one run with identical config.

Averaging or stacking separately trained models does not combine their skills.

### Should we ship one model per mode, or one model for all of them?

One. A single 2.1 M generalist with a mode embedding matched four
separate single-mode models within 0.13 LSD while seeing a fifth as much
data per mode. Once trained on the larger set it beat all four outright.
Fine-tuning a D-STAR specialist from it gained *exactly nothing*: 11.06
against the generalist's own 11.06.

The four codecs damage speech in the same four ways, so undoing them is
one skill. The [mode embedding](theory/the-post-filter.md#mode-embedding)
supplies the per-codec part for the price of four rows of sixteen
numbers, and the receiving radio always knows which mode it is
demodulating.

### Why does it still sound robotic after the high frequencies are fixed?

The decoder\'s voiced bands are too periodic, which a filter cannot fix.
On average, real speech maintains about
10 dB between its harmonics and the valleys between them. Those valleys
hold breath, jitter and glottal noise. A D-STAR decode empties them by a
further 3.4 dB.

Filtering a periodic signal leaves it periodic, so the filter-only model
closed only half the gap. The fix is an aperiodic excitation path plus a
loss that rewards matching the target's periodicity instead of maximising
smoothness. Full story: [What the codec
destroys](theory/what-the-codec-destroys.md).

### The metric said the model was as natural as the recording. Why didn't it sound it?

The metric reported a mean, and a mean of zero does not guarantee a match. Per frame the model was too breathy on the steady vowels and still
buzzy on the breathy frames. The errors had opposite signs and cancelled
in the average. Per band it had over-filled 0–1 kHz and left 2–4 kHz
alone, because the loudest harmonics dominate the ratio it was scored on.

There are two lessons. Report the per-frame absolute error next to any
mean (`eval/hnr_abs` now does). And weight the things you average so that
the loud ones cannot carry it. A learned judge
([predicted MOS](metrics/mos.md)) put the same model at 2.2 where the
recording scores 4.05. The best filter model since, on a larger set,
reaches 2.12 over all four modes against a clean 4.10. The full account
is in [What the experiments showed](theory/what-the-experiments-showed.md).

### Does the model lose plosives?

The codec already loses them. Until 2026-09-13 the model restored nothing
of them, and it filled the silence before each burst by 3 dB. A stop
consonant's burst is a few milliseconds long. The codec smears it to
30 ms at a third of the level. A model that holds every gain for 10 ms,
with a finest loss window of 32 ms, cannot even see the difference.
There is now a [metric for it](metrics/plosive.md), a
[loss term](metrics/loss-terms.md) on the 1 ms high-band envelope, and a
noise path with a gain per millisecond. Whether that is enough is the
next run's question.

## Training

### Is one run of 20 000 steps enough?

One run is enough to converge. Additional runs are useful for testing variations.
Every run here flattens well before its final step. The last 10 000
steps of a 40 000-step run typically move the mean by under 0.1, and all
six surviving checkpoints of a run sit within 0.22 of each other.

So extra runs are worth it when they vary something (data, size,
objective, lookahead). They are not worth it to squeeze the same
configuration. Also, the "best" checkpoint is chosen by `eval/lsd` alone
and can be poor on the noisy column, so check both before shipping.

### What is the most economical way to train in the cloud?

A rented consumer card on a marketplace, rather than a datacentre GPU. These
models are 1–8 M parameters on 8 kHz audio, so an A100 would sit idle.
Vast.ai RTX 3090s start around $0.10/h and 4090s run $0.27–0.34/h on
demand. A full run costs a few dollars, and a ten-config sweep is under
$50.

The bill is decided less by compute than by checkpointing and storage. The cheap rate is interruptible, so the box can vanish, and
over a long project persistent volumes cost more than the GPU hours.
Only shards go up and only checkpoints come back. Raw corpora never leave
the 8 TB drive. Details and sources: [Economics](design/training.md).

### Is 100 ms of lookahead enough, or should it be 400 ms?

100 ms is enough. The ll5 and ll20 models scored 11.02 and 11.04,
which is indistinguishable, so four times the delay bought nothing
measurable. Both are kept as config variants because the choice is about
the conversation and not the model.

### There was static at the end of a degraded sample. Was that on purpose?

Yes. Every eval clip has one second of deliberate garbage appended.
After a transmission ends, a vocoder fed noise emits babble that sounds
like a fast foreign speaker. The model is trained to output *silence*
there, and the `eval/babble` column measures how well it does. The
dashboard marks the boundary with a dashed line so it is not mistaken for
a defect.

## Data

### Why train on LibriTTS-R's dev and test readers?

Holding them out entirely put 58 % of the captured AMBE hours
into splits no run trains on, to serve an evaluation that uses 17 clips
per mode. Those subsets were tier 0, captured through the chip first over
several days. The split rule sent every one of their readers to `dev` or
`test`, so D-STAR had 12 hours in train against 21 held out.

The rule now hashes those readers the way it hashes VCTK speakers. It
keeps about 8 % held out, plus the readers the eval clips come from.
Nothing leaks, because the split is by speaker and a reader is in one
split only. Growing the training set has the best measured return in the
project, and this grew the AMBE part of it for free.

### Can the chip capture a random sample instead of one corpus at a time?

Yes, and it does by default. The first captures ran corpus by corpus, so
after a week the D-STAR set was 30 hours of the same 128 speakers from
two datasets. `capture --order random` gives every prepared utterance a
fixed position from a hash of its key. At any moment the captured set is
a uniform random sample of everything prepared. A corpus prepared later
drops its utterances into the remaining order without restarting
anything. `--order balanced` round-robins over corpora instead, for when
a small corpus matters more than its hours. Details are in the
[data pipeline](design/data-pipeline.md#stage-2-capture).

### Is it worth capturing separate corpora for YSF DN and DMR?

No. They carry the same 49 voice bits. The difference is framing and
FEC, which sit outside the vocoder payload. One capture (`ysf-dmr`)
serves both. That is why DMR was retired as a capture mode and its rate
word kept only as a constant for future framing work. This saved weeks of
chip time.

### Are there more datasets with a variety of voices?

Yes. The manifest spans tiers 0–6: seed, core, extended, noise, voice
diversity, other languages, plus Common Voice by hand. Every corpus must
be usable commercially (CC BY, CC0, Apache, public domain). A CC BY-NC
or research-only corpus is never added, not even for evaluation.

A licence is not necessarily permission to re-host.
Common Voice comes from the Mozilla Data Collective, whose terms forbid
public re-hosting. Those corpora are flagged `redistributable=false`. They
must never be committed, served off loopback or published anywhere. See
[Data sources](research/data-sources.md).

## Hardware

### Can the DVstick's baud rate be increased?

No. 460 800 is the AMBE-3000\'s maximum UART rate (manual Table 19,
and the stick's `CFG2` already selects it). The real finding was that
baud was never the ceiling. 7.07 ms of each 20 ms frame is simply the
326-byte speech packet crossing the wire.

The UART is full duplex, so encode and
decode can be in flight at once. Keeping 3 of each outstanding took the
harness from 68.9 to 97 frames/s, a 1.41× speed-up end to end. Depth is
not a free knob: 4 + 4 stalls the chip. See
[Hardware throughput](research/hardware-throughput.md).

### What bitrates does M17 support?

Codec 2 at 3200 and 1600 bit/s. The 3200 mode is 64 bits per 20 ms
frame. The 1600 mode stretches to 64 bits per 40 ms frame, which is why
its `frame_samples()` is 320 rather than 160. Both are captured in
software with the `codec2` crate. No chip is needed, and it runs
thousands of times faster than real time.

## Metrics

### What am I looking at in the training dashboard?

Loss curves (what is being optimised), eval columns every 500 steps (how
good the model is on held-out speech), and a compare panel with clean,
degraded and model spectrograms side by side.

The number to watch is `eval/lsd`. Lower is better, and ~11 is normal
because the average includes the 4–8 kHz band the codec never sent.
[Reading the numbers](theory/reading-the-numbers.md) explains what each
column means and how large a difference must be to represent a real improvement.
The short version: 0.05 on `eval/lsd` is noise and 0.25 is probably
signal.

### Was the Xiph project called OSCE?

Yes. OSCE (Opus Speech Coding Enhancement) shipped in Opus 1.5. It
covers LACE and NoLACE, the post-filters this project's architecture
follows, with BBWENet for bandwidth extension and FARGAN as the neural
vocoder. The step from LACE to NoLACE is directly relevant here. NoLACE
beat LACE at low rates by adding paths that *generate* signal instead of
only filtering. [Prior art](research/prior-art.md) has the references.
