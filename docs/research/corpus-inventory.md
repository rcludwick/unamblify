# Corpus inventory

!!! info "Provenance"
    Counted from the manifests on the data drive on 2026-09-13
    (`prepared/manifest.jsonl`, `captured/<mode>/manifest.jsonl`,
    `archives/`). Re-run the counts before quoting them. The captures
    grow, and the drive is the only copy.

!!! warning "2026-09-20: Updates since the count"
    Three items below are no longer accurate.

    Common Voice is in the captures. The English sets were prepared on
    2026-09-15. The capture order is a uniform sample of everything
    prepared so Common Voice is now about nine tenths of both AMBE sets
    (338 172 of 377 138 D-STAR utterances and 168 543 of 187 838 YSF/DMR). The
    rest (LibriTTS-R, VCTK, LJSpeech, and VoiceBank-DEMAND) is about 39 000
    and 19 000 utterances.

    The drive is no longer the only copy. The chip captures and
    `dstar+perens` are on a private Google Drive folder as checksummed tar
    chunks ([data pipeline](../design/data-pipeline.md)). These are shared
    with named collaborators only ([reproducing](../design/reproducing.md)).

    The captures as they stand cannot be the public release. This
    changes the section on a published corpus. The chunks mix corpora so a
    public bundle must be built separately from an explicit corpus list
    that leaves Common Voice out (the nine tenths above). The
    `redistributable` flag that section asks for still does not exist.

This page lists what is on the 8 TB drive as opposed to what
[data sources](data-sources.md) lists as wanted. There are three layers:
raw archives fetched, clean utterances prepared, and pairs captured
through a vocoder.

## Fetched

| Corpus | Tier | Licence | Re-shareable | On the drive | Prepared? |
|---|---|---|---|---|---|
| LibriTTS-R `dev_clean`, `test_clean`, `train_clean_100`, `train_clean_360` | 0–1 | CC BY 4.0 | yes | 39.7 GB | yes |
| VoiceBank-DEMAND clean train + test, noisy test | 0, 3 | CC BY 4.0 | yes | 2.8 GB | yes (clean sets) |
| VCTK 0.92 | 1 | CC BY 4.0 | yes | 11.8 GB | yes |
| LJSpeech 1.1 | 1 | public domain | yes | 2.7 GB | yes |
| Hi-Fi TTS | 2 | CC BY 4.0 | yes | 41.4 GB | no |
| LibriSpeech `train-clean-100`, `dev-clean`, `test-clean` | 2 | CC BY 4.0 | yes | 7.1 GB | no |
| DEMAND (18 environments) | 3 | CC BY 4.0 | yes | 2.1 GB | noise set, not a target |
| Common Voice 26 `en`, `es`, `de`, `fr`, `ja`, `ko`, `zh-CN` (Beijing), v24 `en-AU`, Southern US English, Korean noisy | 6 | CC0 | **no** (Mozilla Data Collective terms) | 232 GB | **no** |

Not fetched: MUSAN (tier 3), and everything in tiers 4 and 5.

## Prepared: the clean 16 kHz side

Four corpora. Each is CC BY 4.0 or public domain, and each can be
re-shared with attribution:

| Corpus | Utterances | Hours | Speakers |
|---|---|---|---|
| LibriTTS-R | 152 799 | 259.9 | 1 225 |
| VCTK | 44 304 | 32.0 | 110 |
| LJSpeech | 13 100 | 23.9 | 1 |
| VoiceBank-DEMAND | 12 387 | 9.4 | 30 |
| **Total** | **222 590** | **325.2** | **1 366** |

No noisy or overdriven twins have been prepared (`--noise-share` and
`--ham-chain-share` have not been run). DEMAND is on the drive but not
used anywhere. Common Voice has been imported into `raw/` but never
prepared. No Common Voice clip is in any manifest, shard set,
checkpoint, or rendered clip.

## Captured: the pairs

| Mode | Utterances | Hours | Corpora | Speakers |
|---|---|---|---|---|
| D-STAR (AMBE, chip) | 30 112 | 32.8 | LibriTTS-R dev/test 17.4 h, VoiceBank 9.4 h, VCTK 6.1 h | 128 |
| YSF/DMR (AMBE+2, chip) | 38 121 | 39.0 | LibriTTS-R dev/test 17.4 h, VCTK 12.2 h, VoiceBank 9.4 h | 148 |
| Codec 2 3200 (software) | 222 590 | 325.2 | everything prepared | 1 366 |
| Codec 2 1600 (software) | 222 590 | 325.2 | everything prepared | 1 366 |

Both Codec 2 modes also have `drops` siblings (decode-only, with a seeded
frame-loss pattern) of 66 779 utterances each. The YSF/DMR capture is
stopped at this count. D-STAR's chip time so far went to tier 0 (the
LibriTTS-R dev/test subsets and VoiceBank) and a fifth of VCTK.

Every captured utterance records its `key` and `mode`, the channel frames
(`.ambe`), the decoded 8 kHz WAV, both SHA-256s, the chip's product id and
firmware, and the encode/decode timings. The capture directory's
`canary.json` records the lag the pairs are aligned by.

## Publishing a corpus

The part worth publishing is the AMBE pairs. Each is a clean recording,
the bitstream a real AMBE-3000 produced from it, and what the chip decoded
it back to. Anyone can reproduce the Codec 2 pairs from the clean audio
with the `codec2` crate so those do not need to be shipped.

### Licence
Everything captured up to the 2026-09-13 count comes from
CC BY 4.0 or public-domain sources. Those pairs can be published as CC BY
4.0 with the [attribution list](data-sources.md#attribution-obligations)
for LibriTTS-R, VCTK, and VoiceBank-DEMAND. LJSpeech is not in the AMBE
captures yet. When it is, it needs no attribution but gets it anyway. A
CC BY source obliges the derived corpus to carry attribution and the
licence notice. It does not stop the derived corpus from being published.

### Exclusions
Common Voice cannot go in regardless of its CC0 licence. The
Mozilla Data Collective terms forbid hosting the dataset or any part of it
anywhere but the platform. A decoded copy of a clip is still a copy of
the clip. Since 2026-09-15 it is in the captures (see the warning above)
so the rule for a published corpus is to build the shard set and the
published bundle from an explicit list of corpora rather than from "everything
prepared" or "everything captured". The same applies to anything else
fetched through a click-through agreement.

### Publishing mechanism
[Data sources](data-sources.md#redistribution-is-separate-from-licence)
describes a per-row `redistributable` flag checked by every publishing
path. As of 2026-09-13 no manifest row carries such a field and no code
checks one. The rule is enforced by the corpus list above and by review.
The flag should exist before any bundle leaves the drive. It would be
derived from the corpus name in the core crate, written by `prepare`, and
recorded in the shard index. Whatever builds the bundle would refuse any
corpus not on the list.

### Size
A D-STAR pair is about 48 kB per second of speech as 16-bit WAV
(clean 16 kHz + decoded 8 kHz + 450 bytes/s of frames). The 32.8 captured
hours are about 5.7 GB per AMBE mode and roughly half that as FLAC. That fits
a Zenodo record or a Hugging Face dataset. It is too big for the
repository which is why the demo page only carries a few clips.

### Attribution
The bundle must carry the following attribution verbatim from the data-sources
page: LibriTTS-R (Koizumi et al. 2023, derived from LibriTTS, LibriSpeech
and LibriVox), VCTK (Yamagishi, Veaux, MacDonald 2019, v0.92) and
VoiceBank-DEMAND (Valentini-Botinhao et al. 2017). It must also note that
the channel frames were produced by a DVSI AMBE-3000R, that the AMBE
codec itself is DVSI's, patented and licensed, and no part of it is
included.
