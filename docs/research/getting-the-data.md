# Getting the data

!!! info "Provenance"
    Written 2026-09-23 from `data/sources.tsv`, the fetch and import
    scripts, and the manifests on the drive that day. Counts move while
    capture runs, and the manifests are the authority.

Everything the models were trained on can be obtained again. Three kinds
of thing are involved:

- **speech corpora**: public downloads, one of them by hand
- **the vocoders** that degrade them: one chip and two pieces of software
- **derived sets** that took time to make, so they are also kept on the
  shared Drive

[Data sources](data-sources.md) surveys what could be used and under what
licence. This page answers the shorter question of how to get what *was*
used.

## The rule every source passes

Every source must be usable commercially: CC BY, CC0, Apache or public
domain. CC BY-SA is allowed only in a separate weight line, which has not
been trained. CC BY-NC and "research only" sources are never used, not even
for evaluation. The one source with terms beyond its licence is Common
Voice. Its Mozilla Data Collective terms forbid public re-hosting. It may be
trained on, backed up privately and shared with a named collaborator. It is
never published or de-anonymised
([the policy](data-sources.md#redistribution-is-separate-from-licence)).

## 1. Speech corpora

### Scripted: the studio corpora

`data/sources.tsv` is the manifest, with one row per archive giving its
tier, URL, size and SHA-256. `scripts/fetch-datasets.sh` walks it with
`curl`. It is resumable, verifies every archive and extracts into
`raw/<corpus>/`. It is plain bash on purpose, so no build is needed.

```sh
export UNAMBLIFY_DATA=/path/with/space
scripts/fetch-datasets.sh --tier 0      # ~5.5 GB: LibriTTS-R dev/test, VoiceBank-DEMAND
scripts/fetch-datasets.sh --tier 1      # ~51 GB more: LibriTTS-R train_clean_100/360, VCTK 0.92, LJSpeech 1.1
scripts/fetch-datasets.sh --only vctk   # one corpus
```

| Tier | Corpus | Licence | Used for |
|---|---|---|---|
| 0 | [LibriTTS-R](https://www.openslr.org/141/) `dev_clean`, `test_clean` | CC BY 4.0 | the held-out speakers: every eval clip and the listening test |
| 0 | [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) clean sets | CC BY 4.0 | training targets |
| 1 | LibriTTS-R `train_clean_100`, `train_clean_360` | CC BY 4.0 | the bulk of the studio targets (152 799 utterances prepared) |
| 1 | [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) | CC BY 4.0 | targets and accent diversity (44 304) |
| 1 | [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) | public domain | targets (13 100) |
| 2 | [Hi-Fi TTS](https://www.openslr.org/109/) | CC BY 4.0 | studio targets, opt-in (`prepare --corpus hifi_tts`): 290 h from 8 of its 10 speakers (the two that score 3.5 on the judge are excluded), all `train` |
| 2 | LibriSpeech | CC BY 4.0 | fetched, not prepared. It is 16 kHz and un-restored, so not a target |
| 4 | LibriTTS-R `train_other_500` | CC BY 4.0 | fetched 2026-09-25 (47 GB): 310 h, 1 160 speakers, restored like `train_clean`. The same `libritts_r` walker reads it once extracted |
| 3 | [DEMAND](https://zenodo.org/records/1227121), [MUSAN](https://www.openslr.org/17/) | CC BY 4.0 | noise for the `+n` twins. DEMAND is fetched, but no noisy twin has been prepared yet |

The four studio corpora (LibriTTS-R, VCTK, LJSpeech and VoiceBank) are
`prepare`'s default (`DEFAULT_CORPORA`). Together with the opt-in Hi-Fi
TTS, they are the only corpora whose recordings are used as *targets* for
anything generative. On the naturalness judge a Common Voice recording
scores 3.15 where these score 4.1, and a model that regresses onto its
targets learns their rooms and microphones (experiment log #34). Every
shard set for the restorer is built with
`--corpora libritts_r,vctk,ljspeech,voicebank_demand`.

### By hand: Common Voice

[Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) (CC0) is
gated behind a web form. Since October 2025 it is distributed only through
the [Mozilla Data Collective](https://mozilladatacollective.com), so the
download cannot be scripted. Download the variants in a browser, then
import them:

```sh
scripts/import-common-voice.sh ~/Downloads/*commonvoice-v24_*.tar.gz
```

The importer accepts the per-variant archives
(`commonvoice-v24_<lang>-<REGION>.tar.gz`) and the classic per-locale
`cv-corpus-*` tarballs. It moves each into `archives/`, records its SHA-256
and extracts it into `raw/common_voice/<variant>/`. Clips are MP3.
`prepare` decodes them and keys every row `common_voice/<variant>/<stem>`,
with the opaque `client_id` as the speaker.

Only the **English** variants were used: `en` (1 881 238 utterances
prepared), `en-AU` (55 249) and `cv26-southern-american-english` (1 174).
They make up nine tenths of every chip capture set and supply input variety
(accents, rooms, microphones). Their recordings are never targets for the
restorer. The other languages on the drive (`de`, `es`, `fr`, `ja`, `ko`,
`zh-CN-beijing`) were downloaded but not prepared. Common Voice is opt-in
for `prepare`: `unamblify prepare --corpus common_voice`.

## 2. Preparing

```sh
just prepare                                        # the four studio corpora → prepared/ (16 kHz + 8 kHz, manifest)
unamblify prepare --corpus common_voice             # add the English Common Voice variants
just prepare --twins-only --ham-chain-share S --underdrive-share S   # overdriven (+h) and underdriven (+u) twins
```

`prepare` resamples, trims and loudness-normalises the audio. It writes
`prepared/manifest.jsonl` with the split rule applied (the held-out readers
are pinned). The twins are deterministic from their seed and share. Each
twin's manifest row records the parameters that made it, so the same flags
reproduce the same rows. The drive holds 21 342 overdriven and 21 229
underdriven twins, and no noisy ones yet. [Data
pipeline](../design/data-pipeline.md) covers every stage in detail.

## 3. The vocoders

| Vocoder | Modes | How to get it |
|---|---|---|
| **AMBE-3000R** on a ThumbDV / DVstick 30 | `dstar`, `ysf-dmr` | hardware, about US$100 a stick, and three were used here. `unamblify capture --mode dstar --port …` runs for weeks and is resumable. Every start checks the chip against a canary clip ([hardware throughput](hardware-throughput.md)) |
| **Codec 2** ([`codec2` crate](https://crates.io/crates/codec2), LGPL-2.1 AND MIT) | `codec2-3200`, `codec2-1600` | built into the binary (`codec2` feature, on by default). `unamblify capture --mode codec2-3200 --jobs N` is software only and does the whole corpus in hours |
| **Bruce Perens' software D-STAR** (`ham_digital_modes`, LGPL-3.0-or-later) | `dstar+perens` | vendored at `vendor/ham-digital-modes` (rev `e403fcf`), D-STAR subset only, because AMBE+2's patents are live. `unamblify recode --mode dstar --kind perens` |

The chip captures took weeks and cannot be scripted from a public
download, so they are also kept on the shared Drive
([reproducing](../design/reproducing.md), path 2). As of 2026-09-23 the
counts were 467 629 D-STAR and 362 562 YSF/DMR utterances (with the studio
corpora still being pushed through), 102 661 `dstar+perens`, and both
Codec 2 modes over the full 2.2 M.

## 4. Third-party models

| Model | Licence | Used as | How to get it |
|---|---|---|---|
| [Vocos mel-24k](https://huggingface.co/charactr/vocos-mel-24khz) | MIT | the waveform synthesiser the restorer pipeline starts from (fine-tuned on this project's pairs) | `pip install vocos`, which downloads the weights on first use |
| [UTMOS22 strong](https://github.com/tarepan/SpeechMOS) | MIT | the predicted-MOS judge (`scripts/mos.py`) | `torch.hub`, on first run |

## 5. Derived sets on the shared Drive

These are not public (see the handling rules on the
[reproducing](../design/reproducing.md) page). They hold everything a
collaborator needs to work without a chip:

| Path | What |
|---|---|
| `captured/<set>/` | the chip and recode captures as checksummed tar chunks (`scripts/backup-chunks.sh restore`) |
| `shards/studio-v1` | 398 804 training examples, studio targets only, all five capture sets. The restorer trains on this |
| `shards/mixed-50k-v6` | the 200 k mixed set, the last filter-model baseline |
| `models/spike-2026-09-23/` | the restorer and fine-tuned synthesiser weights behind experiment #35–#36 |
| `models/pipeline-v3-g4/` | the studio-v3 restorer and the G4 synthesiser exported for the Rust runtime (`pipeline.safetensors`, `unamblify restore --weights`) |
| `listening-test/2026-09-22-owner/` | the blind test's clips, predicted scores and the owner's ratings |

## Disk

Fetched archives and raw corpora take about 160 GB (the English Common
Voice `en` archive alone is 97 GB). `prepared/` takes 260 GB, the chip
captures 16 + 8 GB and growing, and a shard set 62–77 GB. Plan for a
terabyte.
