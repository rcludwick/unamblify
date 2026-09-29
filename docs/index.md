# ![](images/logo.svg){ .title-logo } unamblify

A real-time neural post-filter that reduces vocoder artifacts in AMBE
digital-voice audio.

D-STAR, DMR, System Fusion and NXDN carry speech through DVSI's AMBE
family of vocoders at 2400–2450 bit/s. AMBE keeps a voice intelligible at
this rate by transmitting a parametric model: pitch, voiced/unvoiced flags
and a spectral envelope sampled at the harmonics. The decoder synthesizes
the phase, fine spectral texture, consonant attacks and frequencies above
3.7 kHz. This process introduces the buzzy distortion characteristic of
digital voice.

unamblify trains a neural network on pairs of clean speech and the
corresponding AMBE-3000 decode. The network runs in real time on the
receive side to output a signal closer to clean SSB. It can also restore
high-frequency content to produce a wider-bandwidth signal.

## Hear it

Eight held-out voices, four male and four female. Each plays the original
recording, then the same recording passed through an AMBE-3000 chip in
D-STAR mode (AMBE, 2400 bit/s), then the unamblify output. None of these
speakers were in the training set.

<div class="samples" markdown="0">
  <div class="sample">
    <h3>LibriTTS-R &middot; speaker 1272</h3>
    <p class="who">audiobook read, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-1.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-1.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-1.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>LibriTTS-R &middot; speaker 5338</h3>
    <p class="who">audiobook read, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-2.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-2.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-2.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>VoiceBank &middot; speaker p228</h3>
    <p class="who">studio prompt, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-3.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-3.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-3.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>LibriTTS-R &middot; speaker 260</h3>
    <p class="who">audiobook read, test split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-4.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-4.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-4.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>LibriTTS-R &middot; speaker 1462</h3>
    <p class="who">audiobook read, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-5.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-5.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-5.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>VCTK &middot; speaker p278</h3>
    <p class="who">studio prompt, English, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-6.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-6.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-6.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>VCTK &middot; speaker p341</h3>
    <p class="who">studio prompt, American, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-7.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-7.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-7.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
  <div class="sample">
    <h3>VCTK &middot; speaker p298</h3>
    <p class="who">studio prompt, Irish, dev split</p>
    <div class="row">
      <span class="tag clean">Original</span>
      <audio controls preload="none"><source src="assets/audio/sample-8.clean.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag ambe">D-STAR</span>
      <audio controls preload="none"><source src="assets/audio/sample-8.degraded.wav" type="audio/wav"></audio>
    </div>
    <div class="row">
      <span class="tag out">Restored</span>
      <audio controls preload="none"><source src="assets/audio/sample-8.restored.wav" type="audio/wav"></audio>
    </div>
  </div>
</div>

<p class="samples-note" markdown="0">
  <b>Restored</b> is the restorer + waveform synthesiser pipeline
  (<a href="design/pipeline-runtime/">the Rust runtime</a>,
  <code>unamblify restore</code>, weights <code>pipeline-v3-g4</code>).
  It takes the chip's 8&nbsp;kHz decode and returns 24&nbsp;kHz speech, so
  everything above 4&nbsp;kHz is rebuilt by the model. The clips are
  level-matched. Across the eight, predicted MOS is 2.3 to 3.1 for the
  D-STAR decode, 3.3 to 3.9 restored and 3.6 to 4.4 for the original
  recordings
  (<a href="metrics/mos/">how that is measured</a>).
</p>

<p class="samples-note" markdown="0">
  Speech: <a href="https://www.openslr.org/141/">LibriTTS-R</a> (Koizumi et al.,
  Interspeech 2023, derived from LibriTTS, LibriSpeech and LibriVox) and
  <a href="https://datashare.ed.ac.uk/handle/10283/2791">VoiceBank-DEMAND</a>
  (Valentini-Botinhao et al., University of Edinburgh, 2017) and
  <a href="https://datashare.ed.ac.uk/handle/10283/3443">VCTK 0.92</a>
  (Yamagishi, Veaux and MacDonald, University of Edinburgh, 2019), all
  <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Only
  corpora that may be re-shared appear here. See
  <a href="research/data-sources/">data sources</a>.
</p>

## Goals

1. Natural narrowband first. Remove buzz and phasiness and restore the
   texture of unvoiced sounds and transients at the vocoder's 8 kHz rate.
   The reference is a clean SSB voice.
2. Bandwidth extension second. Output at 16 kHz with plausible
   4–8 kHz content. The reference is an AM broadcast voice.
3. Real time on a CPU in Rust. 20 ms frames, streaming, low added
   latency and small enough to run inside [astar](https://github.com/rcludwick/astar)
   alongside its audio pipeline.
4. Reproducible. Every step is scripted and documented. This includes
   which corpora were used, how they were fetched, how they were processed
   through the hardware and how the model was trained.

## Status

As of 2026-09-25 the data pipeline runs on real data. 2.2 M utterances are
prepared. Approximately 468 000 D-STAR and 363 000 YSF/DMR utterances have
been processed through the AMBE-3000, and both Codec 2 modes are captured
in software.

The current system uses a restorer feeding a waveform synthesizer. It scores
3.57 on [predicted MOS](metrics/mos.md) compared to the codec's 2.22 and
the recording's 4.14 (experiment #38 corrected an earlier 3.05 measured
on a low-passed input). In a blind listening test it scored 4.6 of 5
compared to the codec's 1.6–1.9 and the recording's 4.9–5.0 on both AMBE
modes. It streams with approximately 400 ms of latency. Training is done
in Python and inference runs in Rust (`unamblify restore`, see
[the pipeline runtime](design/pipeline-runtime.md)).

The older 2.1 M-parameter adaptive filter is also built in Rust and runs
in real time. It reaches approximately 2.1 on predicted MOS, and band-swap
probes indicated its low band was ineffective (experiment log #33). The
[design page](design/training.md#candidate-3-concretely) outlines current
and future work. The [experiment log](theory/experiment-log.md) records
measurements, and [Reproducing](design/reproducing.md) explains how to
reproduce the results using the shared data.

## Where things are

| Section | What it holds |
|---|---|
| [Theory](theory/index.md) | What the codec destroys and why. How the post-filter works and why it filters rather than resynthesizes. How it is trained, how to read the metrics and what the experiments showed. |
| [Research](research/index.md) | How AMBE works and why it sounds the way it does. The voice corpora and their licenses, hardware throughput budgets and prior art. |
| [Design](design/index.md) | The data-creation harness, the training harness and the real-time inference path. |
| [Notes](notes/index.md) | Findings from working sessions. Local-only for now. |

## Related projects

- [astar](https://github.com/rcludwick/astar): the ham-radio client this will
  eventually live in. It owns the ThumbDV driver and the D-STAR/YSF/DMR framing.
- [how-ambe-works](https://github.com/rcludwick/how-ambe-works): a long-form
  explanation of the vocoder, with measurements from the same AMBE-3000 hardware.
