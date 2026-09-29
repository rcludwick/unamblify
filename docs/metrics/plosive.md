# `eval/plosive_*`: do the stop consonants survive?

**Three columns, all *output minus clean* at the bursts found in the
clean, so 0 is "as sharp as the recording". `plosive_burst` in dB
(negative: the burst is quieter), `plosive_closure` in dB (negative: the
silence before the burst has been filled in), `plosive_rise` in ms
(positive: the onset is smeared). Noise floor: not yet measured.**

## What it measures

A plosive (p, t, k, b, d, g) is a closure of 30–100 ms of near
silence, then a burst a few milliseconds long, then the vowel. It is the
most time-critical event in speech, and every whole-clip spectral metric
is blind to it. A 5 ms burst is a fraction of one 32 ms LSD window, so a
burst smeared to 30 ms at a third of the level moves `eval/lsd` by less
than its noise floor.

## How it is computed

`unamblify_audio::plosive_excess(out, clean)`:

1. The **2–3.8 kHz energy envelope** of each signal at **1 ms**
   resolution (a 4 ms Hann window hopped 1 ms), in dB. The band is
   inside what every mode transmits, so a bandwidth-extension head that
   adds 4–8 kHz cannot score here. Only sharpening the burst the codec
   smeared can.
2. **Bursts are found in the clean.** A burst is a rise of ≥ 15 dB
   within 5 ms, after 40 ms in which nothing came within 8 dB of it, with
   nothing louder in the 50 ms after it. Bursts closer than 60 ms are
   one burst. A vowel fading in from silence does not qualify.
3. At each burst, for both signals: the **peak** (the loudest of the
   ±2 ms around it), the **closure depth** (peak above the quietest
   millisecond of the preceding 40 ms), and the **rise time** (from
   15 dB below the peak to the peak, looking back 20 ms).
4. Output minus clean, averaged over the clip's bursts. The run-level
   column is the burst-weighted mean over clips, so a clip with ten
   stops counts ten times a clip with one. The column is absent when no
   clip holds a burst.

## Standalone

The same function runs over any directory of rendered clips, outside the
trainer. `unamblify measure --dir D [--kinds degraded,out] [--json]`
scores every kind against that stem's `clean`, per mode, burst-weighted.
It prints the [sibilant](sibilant.md) columns beside these. That is how
a pipeline that does not live in the Rust trainer (the spike scripts) is
held to the same yardstick.

## How to read it

Measured on the D-STAR eval clips, 73 bursts (2026-09-12):

| Signal | burst peak | closure depth | rise time | burst length |
|---|---|---|---|---|
| Clean | 0 | 43.5 dB | 2.9 ms | 10.1 ms |
| Raw D-STAR decode | −12.8 dB | 31.7 dB | 13.7 ms | 30.5 ms |
| Filter-only model | −13.5 dB | 28.8 dB | 13.8 ms | 28.8 ms |
| Noise-head model | −13.3 dB | 28.0 dB | 14.0 ms | 29.4 ms |

So `eval/plosive_burst ≈ −13`, `eval/plosive_closure ≈ −15`,
`eval/plosive_rise ≈ +11` for both models. The codec smears the burst
and the models pass it through unchanged, then fill the closure by a
further 3 dB. YSF/DMR reads the same to within a dB.

Nothing in the objective or the architecture before 2026-09-13 was built
for a 5 ms event. Every per-frame gain was held for 10 ms and the finest
loss window was 32 ms. That is why these columns exist. The
[transient term](loss-terms.md) and the noise path's 1 ms gains
(`[model] noise_mod`) are the attempt to move them, and a regression
here is now visible.

## What it cannot see

- **Voiced stops with a short closure** (b, d, g before a vowel) often
  fail the 40 ms closure test and are skipped.
- **Whether the burst is the right *kind*.** A click at the right
  instant and level scores as well as the real burst.
- **Where the detector is wrong.** It is a rule on an envelope, not a
  phonetic transcription. A sharp fricative onset after a pause counts
  too. The same instants are used for both signals, so a wrong instant
  biases the comparison little. Still, the burst count is not a phoneme
  count.
