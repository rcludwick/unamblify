# Voice data sources

!!! info "Provenance"
    Surveyed 2026-09-09. Sizes are the archive byte counts reported by the
    hosts on that date. Hours are from each corpus's own documentation.
    Every URL below was checked on that date and is the one
    `scripts/fetch-datasets.sh` uses, via `data/sources.tsv`.

## What the data has to be

The training pairs are *(clean speech, the same speech after a real
AMBE-3000 encode/decode)*. The degraded side is manufactured by the capture
harness, so the corpus only has to supply the clean side. That side must be:

- **Clean and close-miked.** Any noise or reverberation in the "clean"
  reference becomes something the network is taught to *add*. Only studio
  or studio-like recordings qualify. Noisy corpora are useful for
  evaluation, not as targets.
- **Above 8 kHz sample rate.** For goal 1 (natural narrowband) 16 kHz is
  enough. For goal 2 (bandwidth extension) the reference needs real content
  above 4 kHz, so 22.05 kHz and up is preferred.
- **Speaker-diverse.** The model must generalise to any operator, so many
  speakers matter more than many hours of one speaker.
- **Redistributable-friendly.** The weights inherit the corpora's terms. The
  core set is CC BY 4.0 or public domain. Anything with a non-commercial
  clause is excluded, from training and from evaluation alike.
- **Ham-shaped, eventually.** Real DV audio is mostly male, often older,
  through a hand mic or headset, in a room or a vehicle. No public corpus
  looks like that. See *The ham-radio gap* below.

## Tier 0: the seed (≈ 5.5 GB, ≈ 28 h)

The seed downloads in minutes and goes through one DVstick in about a day
per vocoder mode. That let the capture and training harnesses be built and
exercised on real data while the core set was still arriving.

| Corpus | Licence | Rate | Speakers | Hours | Archive | Why |
|---|---|---|---|---|---|---|
| [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) `clean_trainset_28spk_wav`, `clean_testset_wav`, `noisy_testset_wav` | CC BY 4.0 | 48 kHz | 28 + 2 | 11 | 2.8 GB | The speech-enhancement community's benchmark split (Valentini-Botinhao et al. 2016). Using its exact test speakers makes our numbers comparable to published PESQ/STOI tables. The noisy test set is for *evaluating* robustness only. |
| [LibriTTS-R](https://www.openslr.org/141/) `dev_clean`, `test_clean` | CC BY 4.0 | 24 kHz | 80 | 18 | 2.6 GB | The held-out speakers of the core corpus, so the dev/test split exists from day one and is never contaminated by later training data. |

## Tier 1: the core set (≈ 52 GB, ≈ 314 h)

| Corpus | Licence | Rate | Speakers | Hours | Archive | Why |
|---|---|---|---|---|---|---|
| [LibriTTS-R](https://www.openslr.org/141/) `train_clean_100`, `train_clean_360` | CC BY 4.0 | 24 kHz | ~1 100 | 245 | 37 GB | The bulk of the hours. LibriVox audiobooks restored with a speech-restoration model, so the "clean" side is genuinely clean at 24 kHz. Same speaker/utterance IDs as LibriSpeech. |
| [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) | CC BY 4.0 | 48 kHz | 110 | 44 | 11.7 GB | Accent diversity (UK, Irish, US, Canadian, Indian, …). Studio-recorded, and the standard multi-speaker corpus for vocoder and enhancement work. |
| [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) | Public domain | 22.05 kHz | 1 | 24 | 2.7 GB | One very consistent speaker with tens of hours. Good for debugging training before the multi-speaker set is captured. It is also the voice used in how-ambe-works, so its measurements transfer. |

## Tier 2: more hours, fewer speakers (≈ 48 GB, ≈ 392 h)

| Corpus | Licence | Rate | Speakers | Hours | Archive | Why |
|---|---|---|---|---|---|---|
| [Hi-Fi TTS](https://www.openslr.org/109/) | CC BY 4.0 | 44.1 kHz | 10 | 292 | 41 GB | Very high-bandwidth audiobook speech, and the best reference for the bandwidth-extension goal. It has only ten speakers, so it is a supplement, not a base. **Prepared 2026-09-25** (`prepare --corpus hifi_tts`) as studio-quality targets. On the naturalness judge it averages 3.9 against LibriTTS-R's 4.1 and Common Voice's 3.15 (clean tier 3.94, other 3.86). Speakers 11614 and 6671 score 3.5 and are left out (`HIFI_TTS_EXCLUDED_SPEAKERS`). Keys are `hifi_tts/<speaker>_<quality>/<book>/<stem>`, all `train`. |
| [LibriSpeech](https://www.openslr.org/12/) `train-clean-100`, `dev-clean`, `test-clean` | CC BY 4.0 | 16 kHz | 251 | 100 | 7 GB | The un-restored original of LibriTTS-R, kept for comparison. If the restoration in LibriTTS-R ever proves to have artefacts the network learns, this is the fallback. |

## Tier 3: noise, for mixing (≈ 16 GB)

These are not targets. They are mixed into the *input* side of training
pairs so the model learns to undo a noisy microphone as well as the
vocoder. The target stays the clean recording. The
[data pipeline](../design/data-pipeline.md#stage-3-augment) describes how.

| Corpus | Licence | Rate | Content | Archive | Why |
|---|---|---|---|---|---|
| [DEMAND](https://zenodo.org/records/1227121) (Thiemann, Ito, Vincent 2013) | CC BY 4.0 | 16 kHz (SCAFE only at 48 kHz) | 18 environments × 16 channels × 5 min: kitchen, living room, washing, field, park, river, hallway, meeting, office, cafeteria, restaurant, station, café, square, traffic, bus, car, metro | 2.1 GB | The standard environmental-noise set, and the one VoiceBank-DEMAND was built from, so results stay comparable. Car, bus and traffic are the closest thing to mobile ham operation. |
| [MUSAN](https://www.openslr.org/17/) (Snyder, Chen, Povey 2015) | CC BY 4.0 | 16 kHz | 6 h of noise (technical and ambient), 42 h of music, 60 h of speech | 11 GB | Broader noise types than DEMAND, plus music and background speech for robustness: a QSO with a radio playing in the room, or a second voice behind the talker. Only the noise and music portions are used for mixing. |
| [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) `noisy_trainset_28spk_wav` | CC BY 4.0 | 48 kHz | the clean train set pre-mixed at 0/5/10/15 dB with ten DEMAND noises | 2.8 GB | Evaluation only, alongside the noisy test set already fetched. It is the published benchmark condition, so numbers can be compared with the enhancement literature. Training mixes its own. |

Not used: WHAM! (CC BY-NC), FSD50K (mixed per-clip licences), AudioSet
(YouTube-derived, no clean licence path).

### Ham-specific noise is synthesised

No public corpus has the noises a DV operator's microphone actually
picks up. They are cheap to generate, so they are part of the augmentation
recipe instead of a download:

- vehicle alternator whine and road noise (from DEMAND TCAR/TBUS plus a
  synthetic tonal component)
- fan and PSU hum at 50/60 Hz with harmonics
- band-limited hiss
- mic handling thumps and clicks
- the radio's own mic chain (frequency response, AGC, limiter, mild
  clipping)

Each is a small deterministic generator seeded per utterance.

## Tier 4: voice diversity (≈ 55 GB)

The seed and core sets are almost entirely UK and US audiobook readers.
DV operators are different. They are older and more often male, they have
every accent, many are not native English speakers, and many talk on a
hand mic in a car. These sets widen the voice distribution within the same
licence family.

| Corpus | Licence | Rate | Speakers | Hours | Archive | Why |
|---|---|---|---|---|---|---|
| [LibriTTS-R](https://www.openslr.org/141/) `train_other_500` | CC BY 4.0 | 24 kHz | 1 160 | ~310 | 47 GB | The single biggest speaker boost available. These are the "other" readers LibriSpeech classed as harder (accents, older voices, rougher recordings), restored by the same model as the clean subsets. Same pipeline, same keys. |
| [UK and Ireland English dialects](https://www.openslr.org/83/) (Google, SLR83) | **CC BY-SA 4.0** | 48 kHz | 120 | 31 | 7.2 GB | Six dialects (Irish, Midlands, Northern, Scottish, Southern, Welsh), each with male and female speakers, at studio quality. The best accent coverage in one download. It is share-alike, so weights trained on it inherit CC BY-SA. See the licence note below. |
| [Device Recorded VCTK](https://datashare.ed.ac.uk/handle/10283/3038) (DR-VCTK) | CC BY 4.0 | 48 kHz | 30 | ~11 | 1.8 GB | VCTK speech *re-recorded through consumer devices in real rooms*, paired with the originals. This is the ham-mic domain exactly: a noisy, coloured, reverberant input whose target is the clean studio take. Used input-side, with the VCTK original as the target. |
| [CMU ARCTIC](http://festvox.org/cmu_arctic/) | free for any use | 16 kHz | 7 | ~7 | 0.6 GB | Small, but adds Scottish, Indian and Canadian English voices reading a phonetically balanced script. Useful as fixed demo and evaluation voices. |

## Tier 5: other languages and regions (≈ 130 GB)

DV is used worldwide, and a post-filter trained only on English will
generalise imperfectly to other phoneme inventories. The requested
coverage is French, German, Japanese, Korean and Mandarin, plus Mexican
and other Latin-American Spanish. Everything here is commercially usable.

| Corpus | Licence | Rate | Content | Archive | Why |
|---|---|---|---|---|---|
| [Multilingual LibriSpeech](https://www.openslr.org/94/) (MLS, opus builds) | CC BY 4.0 | 16 kHz (Opus) | German 2 000 h, French 1 080 h, Spanish 920 h, Italian 250 h, Portuguese 160 h, Polish 100 h. Thousands of speakers | 71 GB | The bulk of the non-English hours. `prepare` samples a per-language hour budget rather than using everything. |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | CC BY 4.0 | 16 kHz | ~10 h read speech per locale, many speakers each: `en_us`, `es_419` (Latin-American Spanish), `fr_fr`, `de_de`, `ja_jp`, `ko_kr`, `cmn_hans_cn` | 10.7 GB | One consistent, ungated, commercially usable source that covers every requested language including Japanese, which otherwise has no commercially licensed studio corpus (JSUT and JVS are research-only). |
| [Zeroth-Korean](https://www.openslr.org/40/) | CC BY 4.0 | 16 kHz | 51 h, 115 speakers | 10.3 GB | Korean at scale. |
| [AISHELL-3](https://www.openslr.org/93/) | Apache 2.0 | 44.1 kHz | 85 h, 218 speakers, Mandarin | 19 GB | Mandarin at scale, high sample rate. |
| [Latin-American Spanish](https://www.openslr.org/71/) (Google, SLR61/71–75) | **CC BY-SA 4.0** | 48 kHz | Argentinian, Chilean, Colombian, Peruvian, Puerto Rican, Venezuelan. Male and female sets | 8 GB | Regional Spanish at studio quality. There is no Mexican set on OpenSLR, so Mexican Spanish comes from Common Voice below. Share-alike, with the same handling as the UK/Ireland set. |

## Tier 6: Common Voice, fetched by hand (CC0)

[Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) is the
only source that covers several requested demographics at once. Each clip
carries a self-reported **accent, age and sex**. It covers Southern US
English, Canadian English, Mexican Spanish (the `es` corpus carries a
region field), Japanese (356 h), Mandarin (`zh-CN`, 239 h), French
(1 085 h) and German (1 382 h). Korean is tiny (2 h). It is CC0, so it is
commercially usable without attribution.

There are two caveats. First, the clips come from contributors' own
microphones, so it is an **input-side / lower-weight** set. It is fine for
teaching the model what accents and rooms sound like, but it is not the
reference for the bandwidth-extension head. Second, the download is gated
behind a web form (e-mail + terms), so it cannot be scripted end to end.
Mozilla's current site offers **per-variant archives**
(`commonvoice-v24_<lang>-<REGION>.tar.gz`, e.g. `en-AU`, `en-US`, `es-MX`,
`ja-JP`, `zh-CN`). Each archive has a CSV carrying `client_id, path (mp3),
sentence, age, gender, accents, locale, duration_ms`. The older
full-locale `cv-corpus-*` tarballs still exist. Download the variants you
want in a browser, then import the files (either format) from wherever
they landed:

```sh
scripts/import-common-voice.sh ~/Downloads/*commonvoice-v24_*.tar.gz
```

The importer refuses a file that is still being written and checks that
the archive is complete. It then moves it to `archives/`, records its
SHA-256 and extracts it into `raw/common_voice/<variant>/`. Clips are MP3,
so `prepare` decodes them with symphonia's `mp3` feature. Every Common
Voice row is treated as input-side. The plan is to filter on the accent /
gender / age columns to draw a balanced sample (with Southern US,
Canadian, Mexican and female speakers up-weighted) instead of ingesting
everything.

Variants wanted, in priority order: `en-US` (Southern US accents are
tagged in `accents`), `en-CA`, `es-MX`, `ja-JP`, `zh-CN`, `fr-FR`,
`fr-CA`, `de-DE`. `en-AU` was the first imported (55 673 clips).

### On female voices

The studio corpora are already close to balanced. LibriTTS-R clean is
49 % female by reader, VCTK 47 % and VoiceBank 50 %, and LJSpeech and
Hi-Fi TTS skew female. `prepare` records gender per row, so the sharder can
enforce a 50/50 draw. The `uk_ireland_dialects` and `latam_spanish` sets
ship separate female and male archives, and both are fetched.

### Licence note on share-alike

Everything in tiers 0–3 and 5 is CC BY 4.0 or public domain, so released
weights carry only attribution obligations. The UK and Ireland dialect
set is CC BY-SA 4.0. Whether model weights are an "adaptation" of
training data is legally unsettled. The conservative reading is that
weights trained on it should be released under CC BY-SA 4.0 as well. The
decision for now is to **train two weight lines** and compare them: the
default line without SLR83 (CC BY only), and a `+dialects` line with it.
If the dialect set measurably helps on accented evaluation speech, the
licence question is worth settling. If not, the set is dropped.

## Not usable: non-commercial or unclear licences

Excluded because the project must be usable commercially: JSUT and JVS
(Japanese, research and personal use only), EARS, DAPS, Expresso, Emilia,
L2-ARCTIC, WHAM! (all CC BY-NC), VoxPopuli (CC BY-NC), Speech Accent
Archive (CC BY-NC-SA), TIMIT / CSJ / KsponSpeech (LDC or institutional
terms), YouTube-derived sets (AudioSet, YODAS) whose underlying rights
are unclear.

## Considered and not (yet) used

| Corpus | Why not |
|---|---|
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) (CC0) | Enormous and diverse, but recorded on whatever microphone the contributor had, so too noisy to be a clean target. The download is gated behind an e-mail form, so it is not scriptable. Since 2026-09-15 its English variants are used as an *input-side* source (tier 6 above), never as targets. |
| [DAPS](https://ccrma.stanford.edu/~gautham/Site/daps.html), [EARS](https://sp-uhh.github.io/ears_dataset/), [Expresso](https://speechbot.github.io/expresso/), [Emilia](https://emilia-dataset.github.io/) | Non-commercial (CC BY-NC) clauses. Excluded from training and evaluation so the weights stay cleanly CC BY-compatible. |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) clean speech | Mostly LibriVox-derived (already covered) and hundreds of GB. Revisit if more hours are needed. |
| [People's Speech](https://mlcommons.org/datasets/peoples-speech/), [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) | Huge, but mixed quality and (GigaSpeech) agreement-gated. Not clean enough to be a target. |
| TIMIT, VoxCeleb, Blizzard | LDC / research-only licences. |

## The ham-radio gap

Nothing public sounds like a real QSO, with its hand mic or headset boom,
its vehicle, and the compression and clipping from the radio's mic AGC.
Two things follow:

1. **Domain augmentation on the input side.** Before the clean audio goes
   into the chip, apply the kinds of degradation a radio's front end
   applies (mic frequency response, AGC/limiting, mild room, a little
   noise). Keep the *clean* signal as the target. This teaches the model to
   undo the whole chain, not just the vocoder. The recipe lives in the
   [data pipeline](../design/data-pipeline.md) design.
2. **A ham evaluation set.** Recordings of the author's own voice through
   real radios and the DVstick, plus (with permission) other operators, held
   out purely for listening tests and metrics. Publicly recorded DV traffic
   (reflector archives, BrandMeister recordings) is AMBE-only, with no
   clean reference. It can only be used for *unpaired* listening checks,
   never as training pairs.

## Redistribution is separate from licence

!!! note "Project policy (clarified 2026-09-15): private storage is fine, public redistribution is not"
    Keeping any corpus in private cloud storage is acceptable, whether as
    a backup or as storage attached to a GPU box for training. So is
    sharing a copy with a specific named collaborator. Training pipelines
    routinely need the data in the cloud next to the GPUs. A private copy
    under the project's control is not the public re-hosting that the
    restrictions below target. The hard lines the project holds are
    *public* ones:

    - never make a restricted corpus publicly downloadable
    - never commit its clips to git or ship them in demo clips
    - never serve it off loopback to the open internet
    - never re-identify a speaker

    The Mozilla terms quoted below are the source material. This note is
    how the project applies them.

A permissive licence on a corpus does not mean this project may
*publicly* re-host or re-share its audio. Since October 2025 Common Voice
is distributed **only** through the
[Mozilla Data Collective](https://mozilladatacollective.com) (MDC). Its
platform terms sit on top of the CC0 data licence and were accepted at
download (checked 2026-09-10,
[terms](https://mozilladatacollective.com/terms),
[FAQ on other platforms](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/)):

- **No re-hosting anywhere.** Users may not be "hosting, storing or making
  the Dataset available on any platform … other than the Platform".
  Mozilla's stated reason is consent revocation. Contributors can withdraw
  their clips, and public mirrors would keep serving them. So there is no
  *public* mirror or download, no clip in the repository or on the demo
  page, and no serving off-loopback to the open internet. A private backup
  or a copy beside a GPU for training is within bounds (see the policy
  note above).
- **No re-identification.** Users may not "attempt to determine, trace,
  match the identity of, or re-identify the individual contributor". The
  harness uses only the opaque `client_id` hash as a speaker key and
  never links it to any other corpus, metadata source, or person. The
  Samples page shows the hash, nothing else.
- **Delete on termination.** All copies must be deleted if the MDC
  account is terminated or suspended. Keep the import list so that is
  mechanical.
- **Machine-generated outputs must be disclosed as such** and must not
  contain personal data. All audio the model emits is labelled as model
  output wherever it is presented. None of it on any public page is
  derived from a Common Voice clip.
- **Commercial use of the *data*** is governed by the dataset's own
  licence (CC0 for Common Voice), which permits it. The platform terms
  restrict commercial use of the *platform*. Trained weights are fine.
  This is the one point where it is worth reading the terms in full
  instead of trusting a summary.
- **Consent revocation** creates no stated obligation to refresh. Still,
  re-downloading a later release before a public weights release keeps
  withdrawn clips out of the training set. It is the considerate thing to
  do and costs little.

Several other hosts attach click-through terms too. So each corpus is
assigned a **redistributable** flag. Every publishing path is meant to
check the flag, not the licence:

!!! warning "Rule, not yet code"
    As of 2026-09-13 the flag exists in this table only. No manifest row
    carries a `redistributable` field and nothing checks one. Since
    2026-09-15 Common Voice is prepared and captured (see the
    [corpus inventory](corpus-inventory.md)), so the rule now holds only
    through explicit corpus lists and review. No software would stop a
    mistake. Implement the flag before any bundle, dashboard or demo
    leaves loopback.

| May be **publicly** re-shared by this project | Corpus |
|---|---|
| **No** | Common Voice (Mozilla terms), anything fetched through a click-through agreement |
| Yes, with attribution | LibriTTS-R, LibriSpeech, VCTK, VoiceBank-DEMAND, DR-VCTK, LJSpeech (public domain), Hi-Fi TTS, CMU ARCTIC, MLS, FLEURS, Zeroth-Korean, AISHELL-3, DEMAND, MUSAN, UK/Ireland dialects and Latin-American Spanish (share-alike) |

What the flag governs:

- **The demo page** commits audio to the repository and publishes it on
  GitHub Pages. Only clips from redistributable corpora, or clips
  synthesised with Piper, may appear there. A Common Voice clip never
  does.
- **Any dashboard bound to a non-loopback address** serves clip audio
  through the Samples page. The server is to refuse clips from
  non-redistributable corpora unless it is bound to loopback or the
  operator passes `--serve-restricted`. It is to log each such request.
- **Released weights** are derived works of everything they were trained
  on but contain no audio. Training on a non-redistributable corpus is
  allowed, and the corpus is credited in the weights' data card either
  way.
- **Shards and prepared audio** may be kept in private cloud storage (a
  backup, or beside a GPU for training). The `redistributable` flag gates
  only a *public* release of a shard set. The shard index records which
  corpora it contains, so that check stays mechanical.
- **The raw corpora** are never *publicly* mirrored by this project, but a
  private backup is fine. To reproduce the dataset, someone else runs the
  fetch scripts instead of receiving a copy.

## Attribution obligations

CC BY 4.0 requires attribution and a licence notice in anything derived.
Released weights and any demo audio must credit:

- LibriTTS-R: Koizumi et al., *LibriTTS-R: A Restored Multi-Speaker
  Text-to-Speech Corpus*, Interspeech 2023. Derived from LibriTTS
  (Zen et al. 2019), LibriSpeech (Panayotov et al. 2015) and LibriVox.
- VCTK: Yamagishi, Veaux, MacDonald, *CSTR VCTK Corpus*, University of
  Edinburgh, 2019 (version 0.92).
- VoiceBank-DEMAND: Valentini-Botinhao et al., *Noisy speech database for
  training speech enhancement algorithms and TTS models*, University of
  Edinburgh, 2017.
- Hi-Fi TTS: Bakhturina et al., *Hi-Fi Multi-Speaker English TTS Dataset*,
  Interspeech 2021.
- LibriSpeech: Panayotov, Chen, Povey, Khudanpur, ICASSP 2015.
- UK and Ireland dialects: Demirsahin, Kjartansson, Gutkin, Rivera,
  *Open-source Multi-speaker Corpora of the English Accents in the British
  Isles*, LREC 2020 (CC BY-SA 4.0).
- DR-VCTK: Sarfjoo, Yamagishi, *Device Recorded VCTK*, University of
  Edinburgh, 2018.
- CMU ARCTIC: Kominek, Black, *The CMU Arctic speech databases*, 2004.
- MLS: Pratap, Xu, Sriram, Synnaeve, Collobert, *MLS: A Large-Scale
  Multilingual Dataset for Speech Research*, Interspeech 2020.
- FLEURS: Conneau et al., *FLEURS: Few-shot Learning Evaluation of
  Universal Representations of Speech*, SLT 2022.
- Zeroth-Korean: Zeroth Project / Atlas Guide, 2017.
- AISHELL-3: Shi, Bu, Xu, Zhang, Li, *AISHELL-3: A Multi-speaker Mandarin
  TTS Corpus*, Interspeech 2021 (Apache 2.0).
- Latin-American Spanish: Guevara-Rukoz et al., *Crowdsourcing
  Latin American Spanish for Low-Resource Text-to-Speech*, LREC 2020
  (CC BY-SA 4.0).
- Common Voice: Ardila et al., *Common Voice: A Massively-Multilingual
  Speech Corpus*, LREC 2020 (CC0, attribution not required but given anyway).
- DEMAND: Thiemann, Ito, Vincent, *The Diverse Environments Multi-channel
  Acoustic Noise Database*, ICA 2013.
- MUSAN: Snyder, Chen, Povey, *MUSAN: A Music, Speech, and Noise Corpus*,
  arXiv 1510.08484, 2015.

LJSpeech is public domain and needs none, but is credited anyway.

## Where it goes and how to fetch it

```
/Volumes/data/training_data/unamblify/     # $UNAMBLIFY_DATA
├── archives/     downloaded archives as-is, plus <name>.ok markers
├── raw/<corpus>/ extracted, untouched
├── checksums/    sha256 of every archive as first observed
├── logs/         one log per fetch or capture run
├── prepared/     (later) resampled, normalised clean audio + manifest
└── captured/     (later) per-mode chip output + channel frames
```

```sh
just fetch-data --tier 0        # the seed, ~5.5 GB, minutes
just fetch-data --tier 1        # adds the core set, resumable, ~57 GB total
just fetch-data --tier 2        # adds Hi-Fi TTS and LibriSpeech
just fetch-data --tier 3        # adds the noise sets (DEMAND, MUSAN, VoiceBank noisy)
just fetch-data --tier 4        # adds voice diversity (LibriTTS-R other, UK/IE dialects, DR-VCTK, ARCTIC)
just fetch-data --tier 5        # adds other languages and regions (MLS, FLEURS, Zeroth, AISHELL-3, LatAm Spanish)
scripts/import-common-voice.sh ~/Downloads/*commonvoice-v24_*.tar.gz   # tier 6, after a browser download
just fetch-data --only vctk     # one corpus
scripts/fetch-datasets.sh -h    # all options
```

The manifest is `data/sources.tsv`, with one archive per line giving its
tier, URL, size and checksum. `--tier N` fetches every tier up to *N*.
OpenSLR publishes MD5 sums, and those are pinned. The Edinburgh DataShare
and LJSpeech hosts publish none. For those, the script records the SHA-256
it observes into `checksums/` on first fetch and verifies against it after
that. The observed sums should be copied back into the manifest once a
second machine has confirmed them.

!!! note "Edinburgh DataShare quirk"
    A `HEAD` request against a DataShare bitstream URL reports a 4 kB body.
    The real file is only revealed by a `GET`. The manifest sizes came from
    ranged `GET` requests, and the script compares the on-disk size against
    the manifest rather than trusting `HEAD`.
