# Reproducing from the shared data

!!! info "Provenance"
    Written 2026-09-20 and updated 2026-09-25 against the data root and
    the Google Drive copy. Counts change during capture runs. The manifests
    serve as the authority.

You can rebuild this data from public corpora and a ThumbDV. The chip captures take weeks of hardware time, so they are kept off the machine and shared with collaborators. This page is for researchers with access to that copy who want to repeat a result or build on it.

!!! warning "The shared data is not public"
    About 90% of the captured audio is derived from Common Voice, which is distributed only through the Mozilla Data Collective under terms that forbid re-hosting ([data sources](../research/data-sources.md)). If you have access, keep the data private and do not distribute it. Do not publish clips publicly or attempt to identify speakers. The opaque `client_id` is the only speaker handle.

## What is where

The shared folder is a Google Drive folder named `unamblify-corpus`, read with [rclone](https://rclone.org/drive/). Configure a remote using `rclone config`. Use your own OAuth client ID since the shared rclone ID is being retired in 2026. Point the scripts to the remote using `RCLONE_REMOTE=<your-remote>:unamblify-corpus`.

```
unamblify-corpus/
  captured/<set>/              one directory per chip or re-encode capture set
    <set>-0001.tar             ~5 GiB of utterances: <key>.flac + <key>.ambe
    <set>-0001.sha256          SHA-256 of every file inside the tar
    <set>-0001.tar.sha256      SHA-256 of the tar itself
    <set>-0001.keys            the utterance keys it holds
    manifest.jsonl             the capture manifest (one CaptureRow per utterance)
  shards/<name>/               a finished shard set: index.json + NNNN.bin
  models/spike-2026-09-23/     restorer.pt + synthesiser.pt behind experiment #35–#36, with a README
  models/pipeline-v3-g4/       pipeline.safetensors + pipeline.json for the Rust runtime, with references
  listening-test/…-owner/      the blind test's clips, predicted scores and the owner's ratings
```

| Set | What it is | Needs a chip to remake |
|---|---|---|
| `dstar` | the prepared audio through the AMBE-3000 at the D-STAR rate | yes |
| `ysf-dmr` | the same at the YSF DN / DMR rate (one capture serves both) | yes |
| `dstar+perens` | the same audio through the vendored software D-STAR vocoder | no: `unamblify recode` |

The Codec 2 captures are omitted. They are deterministic software outputs of 2.2 million utterances each. Running `unamblify capture --mode codec2-3200 --jobs N` recreates them faster than downloading.

## Path 1: train from a shard set

Shard sets are self-contained. Each example contains the codec output, the clean target, the channel frames, and associated flags. The `index.json` file records modes, lags, draws, and the SHA-256 hash of each source manifest ([shard format](data-pipeline.md#stage-4-shard)). Training does not read anything else from the data root.

```sh
export UNAMBLIFY_DATA=/path/with/space          # ~75 GB per set
mkdir -p "$UNAMBLIFY_DATA/shards"
rclone copy <remote>:unamblify-corpus/shards/mixed-50k-v6 \
            "$UNAMBLIFY_DATA/shards/mixed-50k-v6" --transfers 4 --progress

just train-env                                   # once per checkout: libtorch
just train configs/mixed-50k-v6-perens-ll5.toml  # --device cpu | mps | cuda:0 | rocm:0
just serve                                       # watch it: http://127.0.0.1:8787
```

The run config specifies the shard set under `[data] shards`. The config and shard set define the complete experiment. Seeds are fixed. Two runs on the same device match within the [noise floor](../metrics/mos.md). Runs on different devices will not match bit-for-bit. Score results using `scripts/mos.py` across several checkpoints. The best checkpoint by predicted MOS is rarely the last epoch or the one with the best trainer metrics.

The Drive contains two sets. `studio-v1` is used for current training. It uses the five capture sets but restricts `--corpora` to studio recordings (LibriTTS-R, VCTK, LJSpeech, VoiceBank). This provides up to 60,000 utterances per set for a total of 398,804 training examples. The model requires high-quality target audio, and Common Voice recordings are insufficient (see experiment log #34), so this set excludes Common Voice audio.

`mixed-50k-v6` includes 50,000 utterances from five capture sets (`ysf-dmr`, `dstar`, `dstar+perens`, `codec2-3200`, `codec2-1600`). We reserve 12% of each draw for overdriven (`+h`) and underdriven (`+u`) twins. Noisy (`+n`) twins are excluded because they are not prepared yet. Avoid using `mixed-50k-v5`. Its config (`configs/mixed-50k-v5-perens-ll5.toml`) is retained for historical purposes, but the `dstar+perens` examples are cut at the wrong lag (experiment log #31) and the trainer will reject it.

## Path 2: restore the chip captures and build your own sets

```sh
export UNAMBLIFY_DATA=/path/with/space
export RCLONE_REMOTE=<remote>:unamblify-corpus
scripts/backup-chunks.sh restore --mode dstar        # ~16 GB
scripts/backup-chunks.sh restore --mode ysf-dmr      # ~8 GB
scripts/backup-chunks.sh restore --mode dstar+perens # ~4 GB
```

The `restore` command downloads chunks sequentially and verifies each tar against its hash. It unpacks the tar into `captured/<set>/`, verifies the files against the sidecar, and deletes the tar. This process requires scratch space equal to one chunk. It can be interrupted and resumed. It only fetches `manifest.jsonl` if missing, preventing overwrites of active capture manifests.

A capture represents half of a training pair. The clean target is stored in `prepared/`. This directory is not shared as it contains 260 GB of reproducible data. Download the corpora ([data sources](../research/data-sources.md), including Common Voice from the Mozilla Data Collective) and run `just prepare`. Utterance keys use corpus paths, allowing prepared rows to join restored captures by key. The shard builder logs unmatched rows as `unjoined`. Twins use fixed seeds. Running `just prepare --twins-only --ham-chain-share … --underdrive-share …` with identical shares and seeds will reproduce the original `+h` and `+u` rows.

Then run `just shard …` as described in the [data pipeline](data-pipeline.md) and train as in Path 1.

## Hearing the result without training anything

You can use the Rust runtime with exported weights from the current pipeline (restorer studio-v3 at step 30,000, and synthesiser G4 with Vocos fine-tuned). This requires only the `train` feature's libtorch:

```sh
rclone copy <remote>:unamblify-corpus/models/pipeline-v3-g4 ./models/pipeline-v3-g4
unamblify restore --weights models/pipeline-v3-g4/pipeline.safetensors \
                  --in <8 kHz wav> --mode dstar --out out.wav
```

The [pipeline runtime](pipeline-runtime.md) page describes the weights file. The PyTorch checkpoints for experiments #35–#36 use the Python spike:

```sh
rclone copy <remote>:unamblify-corpus/models/spike-2026-09-23 ./models
.venv-mos/bin/python scripts/spike/combined.py models/restorer.pt models/synthesiser.pt <dir of *.degraded.wav> out/
```

The `scripts/spike/README.md` file documents the environment and scripts. The blind listening test is located at `scripts/spike/listening/`. Ratings for experiment #36 are stored on Google Drive with the clips. See [getting the data](../research/getting-the-data.md) for information on the corpora.

## Path 3: from nothing

Run `just fetch-data --tier 1`, `just prepare`, and `just capture --mode dstar` using a ThumbDV. The capture process takes weeks of chip time ([hardware throughput](../research/hardware-throughput.md)). It is resumable and verifies the chip with a canary clip at startup. The three USB sticks used in this project produce identical hashes for the canary clip from the same chip state. A different AMBE-3000 with the same firmware should produce interchangeable captures. The `canary.json` file is included with each restored set, and capture will abort if the chip output mismatches.

## Checking what you were given

The `restore` process verifies the tars and files against the original checksums. You can also run the following checks:

```sh
unamblify verify --mode dstar     # the capture set itself: files present, frame counts
unamblify stats                   # what the data root holds
```

The `pack`, `push`, and `verify` subcommands of `backup-chunks.sh` are for maintainers and require the local ledger. Restored copies lack this ledger.
