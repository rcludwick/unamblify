# How it learns

!!! info "Provenance"
    Written 2026-09-12 against `crates/unamblify-train/src/losses.rs`
    and the capture and shard stages of
    [Data pipeline](../design/data-pipeline.md). Configuration keys are
    documented in [Training](../design/training.md).

## Training data pairs

The model is trained on pairs of audio files: a clean recording and the same recording processed through a real vocoder. The clean recording is the target and the damaged one is the input. The network maps the input to the target.

Simulating AMBE in software is an alternative approach. However, unamblify plays every utterance through an AMBE-3000 chip on a ThumbDV and records the output. This approach requires more time as discussed in [throughput engineering](../research/hardware-throughput.md). It ensures the model learns the artifacts of the hardware vocoder including its quantizer and firmware behavior. A software reimplementation would train the model to reverse a codec that is not used in practice.

Codec 2 is open source. The M17 modes are captured in software with the `codec2` crate which is faster than real time. The Codec 2 corpora contain 222 590 utterances while the AMBE corpora are smaller due to the hardware requirement.

## Fixed-size crops

Training uses fixed 2-second examples rather than whole utterances. The [shard stage](../design/data-pipeline.md) segments the data and packs it into flat binary files. Each example contains the clean 16 kHz audio, the degraded 8 kHz audio, the channel frames and associated flags.

Fixed-size examples facilitate batching. A 2-second duration provides enough context for the GRU while allowing a batch of 16 to cover multiple speakers and sounds.

The crops are sampled non-uniformly. One third are taken at onsets (the start of speech after silence) and one sixth are tail examples. This weighting ensures sufficient representation of these specific cases.

## Loss function

The total loss is a sum of the following terms:

| Term | What it compares | Purpose |
|---|---|---|
| Multi-resolution STFT | magnitude spectra at three window sizes (512/1024/2048) | Three resolutions capture both plosives and steady vowels. |
| Mel L1 | 80 mel-scaled band energies | Weights the error by frequency resolution. |
| SI-SDR | the 8 kHz waveform scale-invariantly | Aligns the output in time and evaluates the waveform directly. |
| Babble penalty | energy where the target is silent | Penalizes output during target silence. |

The loss function does not compare phase directly or require exact sample matching.

### Edge weighting { #the-edges-matter-most }

Two regions receive extra weight during training.

Onsets. The first 50 frames (500 ms) of an onset example receive double weight. This represents the key-up moment where the GRU has no history and the codec is settling. This weighting encourages stable behavior at the start of a transmission.

Tails. When a transmission ends the vocoder may emit noise or babble. Tail examples include this behavior with a target of silence. The loss applies triple weight to these frames and adds the babble penalty. This trains the model to output silence when the input signal ends.

Babble is a common artifact in neural speech systems. The tail weighting addresses this and the evaluation reports a separate `babble` metric.

## Data augmentation

The training input includes several types of signal degradation:

- Dropped frames. The `drops` sibling captures decode the same audio with 2–3 consecutive channel frames randomly muted or repeated to simulate lost frames.
- Receive-side noise. 30% of examples have mains hum, broadband noise, alternator whine or spectral tilt mixed into the input only.
- Noisy and overdriven twins. Some utterances are processed with background noise or clipping added to the input while the target remains clean.

In all cases the input is degraded while the target remains clean. This trains the model to reconstruct the signal rather than reproduce the input.

## Training configuration

Training uses the Adam optimizer with a learning rate of 2e-4 and a batch size of 16 for 20 000–40 000 steps on Apple Silicon via Metal. Checkpoints and evaluations on held-out development clips occur every 500 steps. The system retains the best five checkpoints and writes metrics to `metrics.jsonl`.

Speakers are split into train, dev, and test sets by a hash of the speaker ID. This ensures voices used in training do not appear in evaluation to measure generalization rather than memorization.

## Adversarial training

HiFi-GAN-style discriminators are implemented via the `[train] gan = true` configuration but are currently disabled. GAN training increases perceptual sharpness but reduces stability and can introduce artifacts. This conflicts with the goals of the [filtering architecture](the-post-filter.md). The implementation remains available for future testing.
