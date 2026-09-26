# Spike scripts: the restorer + waveform-synthesiser pipeline

Throwaway Python behind experiment log #33–#38 (2026-09-20 to 25). These
are the measurements that retired the adaptive-filter model and the
pipeline that replaced it. They are not the Rust harness or the runtime
(that is `unamblify restore`). They are kept so the numbers in the log can
be reproduced and the measurements re-run on the next model. Run
everything with the `.venv-mos` environment
(`uv venv --python 3.12 .venv-mos && uv pip install --python .venv-mos/bin/python torch torchaudio numpy vocos soundfile scipy`).
`UNAMBLIFY_DATA` is the data root. The pretrained synthesiser
(`charactr/vocos-mel-24khz`, MIT) and the MOS judge (`tarepan/SpeechMOS`,
UTMOS22) download themselves on first use.

| Script | What it does |
|---|---|
| `restorer.py train <shard-dir> <out-dir> [steps]` | the restorer: low-band log-STFT + the codec's own mel + mode embedding → clean wideband log-mel, 6.8 M params, causal with 85 ms of lookahead. L1 on log-mel; env `INIT=` to start from a checkpoint, `LR=`, `INDEX=` (a `.npy` of example indices to restrict the draw), `BANDW=` (band-energy term, under-prediction costs double), `POWW=` (compressed-power term, did not help, #36) |
| `restorer.py render <ckpt> <audio-dir> <out-dir>` | every `*.degraded.wav` through the restorer and the *pretrained* synthesiser, in the layout `scripts/mos.py` scores |
| `gta_vocoder.py <restorer.pt> <pretrained\|synth.pt> <out-dir> <deadline-epoch> [index.npy]` | fine-tune the synthesiser on the restorer's predicted mels with Vocos' waveform discriminators (the text-to-speech "ground-truth-aligned" recipe): the step that undid regression's blur, 2.61 → 3.05. `SHARDS=` picks the set; continuing from a `synth.pt` resumes its discriminators |
| `combined.py <restorer.pt> <synth.pt> <audio-dir> <out-dir>` | the whole pipeline over a directory of `*.degraded.wav` |
| `causal_vocoder.py` | the streaming question: `render causal-untrained` (1.27), `train` (a causal fine-tune on clean studio speech, not needed at a 500 ms budget, #35), and the `load()` / `make_causal()` the other scripts use |
| `lookahead.py <restorer.pt> <synth.pt> <audio-dir>` | the pipeline's true lookahead, by truncating the future and watching the past (#35: ~300 ms audible, 380 ms exact) |
| `bandswap.py <audio-dir> <out-dir>` | the decomposition of #33: low band from one signal, high band from another, both through one synthesiser, four cells to score |
| `consonants.py <audio-dir> "label=<dir>:<kind>"...` | plosive burst / closure and the sibilant level and balance against the clean recording, per mode |
| `hf_diag.py <restorer.pt> <audio-dir>` | where the restorer under-predicts the high band: by mode and by how loud the clean band is |
| `listening/build_test.py <out-dir>` | the blind listening test of #36: 16 held-out speakers × 2 chip modes × 4 systems, level-matched, UTMOS-scored; `RESTORER=`, `SYNTH=`, the speaker list in `listening/listening-set-2026-09-22.json` |
| `listening/server.py` | a loopback-only server for the test: `index.html` (blind, calibrated, one mode per pass; ratings to `ratings.jsonl`) and `review.html` (unblinded, side by side, corrections saved as such) |

The weights behind the numbers (`restorer.pt`, `synthesiser.pt`) and the
listening-test ratings are kept with the shared data copy. See
[reproducing](../../docs/design/reproducing.md) for how to get access.
