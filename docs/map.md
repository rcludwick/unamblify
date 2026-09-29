# Docs map

One line per page: what it holds and when to open it. **Read this first**
when looking for anything. `ci/check-docs-map.sh` fails the build if a
page is missing from here, so the map is complete by construction.

## Top level

| Page | Summary |
|---|---|
| [Home](index.md) | What unamblify is, before/after clips through the restorer + synthesiser pipeline (redistributable corpora only), the four goals, current status (the restorer + synthesiser pipeline is the best system), the section table and related projects. |
| [Listen](demo/index.md) | The demo page built by `unamblify demo`: clean → AMBE → each model, with spectrograms and metrics. A placeholder for now. |
| [FAQ](faq.md) | Short answers from the measurements: model size and capacity, cascades, one model or one per mode, why it still sounds robotic, plosives, how many runs, cloud cost, lookahead, the babble tail, data splits and capture order, YSF vs DMR, datasets, DVstick baud, M17 bitrates, the dashboard, OSCE. |
| [Docs map](map.md) | This page. |

## Theory: why it works

| Page | Summary |
|---|---|
| [Theory index](theory/index.md) | What the theory section covers, and the whole idea in one paragraph. |
| [What the codec destroys](theory/what-the-codec-destroys.md) | The four losses in the AMBE bit budget and the artefact each causes. The measured excess periodicity and why a mean near zero was not a match. The smeared stop consonants and why the task has a floor. Open it to understand *what the model is fixing*. |
| [The post-filter](theory/the-post-filter.md) | How the candidate-1 filter works and why it filters instead of resynthesising (the OSCE/LACE lineage): each block, the mode embedding, where the latency lives, the three profiles. |
| [How it learns](theory/how-it-learns.md) | Why the pairs come from a real chip, the 2 s onset-weighted crops, each loss term, the edges rule and babble penalty, the deliberate input damage, the training loop, and why the GAN is off. |
| [Reading the numbers](theory/reading-the-numbers.md) | What every `eval/*` column means, why LSD ~11 and SI-SDR -19 are not alarming, what `eval/hnr` means, each column's noise floor (0.18 for `lsd`, 0.71 for `lsd_rx`), and the comparability traps. Open it before believing any table. |
| [Experiment log](theory/experiment-log.md) | Every experiment in one table: tooling, the vocoder chain, model, data and capacity runs, the naturalness attempts, the MOS yardstick, the restorer + synthesiser spike (#33–#38: 3.57 predicted MOS on a faithful input, 4.6 of 5 in the blind listening test), and what is still open. |
| [What the experiments showed](theory/what-the-experiments-showed.md) | The filter-model results with their caveats, through 2026-09-19: one generalist beats four specialists, 2.1 M is the capacity knee, data beat parameters, the naturalness fix moved the mean and not the frames, no plosive restored, MOS near the bottom of the scale, and where the headroom is. |

## Metrics: one page per number

| Page | Summary |
|---|---|
| [Metrics index](metrics/index.md) | Every column in one table with its units, good value, noise floor and where it is implemented. Start here. |
| [`eval/lsd`](metrics/lsd.md) | Log-spectral distance: why ~11 is normal (the invented 4-8 kHz band counts), the 0.12 noise floor, and what it cannot see. |
| [`eval/mel`](metrics/mel-l1.md) | Mel-band L1 on 80 bands, and why it discriminates less than LSD. The loss version uses natural logs, so the two are not comparable. |
| [`eval/sisdr`](metrics/si-sdr.md) | Scale-invariant SDR. Why it sits near -20 dB by design (the codec discarded phase), so read it as an alarm and not a ranking. |
| [`eval/hnr`, `eval/hnr_abs`](metrics/hnr.md) | **The robotic-quality pair.** Harmonic peaks against inter-harmonic troughs in dB above clean, as a mean and per frame. Measured values, why a mean of zero was not a match, the scale changes and the blind spots. |
| [`eval/plosive_*`](metrics/plosive.md) | **Do the stop consonants survive?** Burst level, closure depth and rise time at 1 ms resolution. The codec smears a burst to 30 ms at −13 dB and both filter models pass that through. |
| [Sibilants](metrics/sibilant.md) | `sibilant_level_db` and `sibilant_balance_db`: the 4–8 kHz level and high-to-mid balance at the clean recording's sibilant frames, why a regression model makes them quiet, and `unamblify measure`. |
| [Predicted MOS](metrics/mos.md) | The learned naturalness judge (UTMOS22 via `scripts/mos.py`): the yardstick every run is judged by, the filter model's scores against clean and codec, and why a correlated proxy stops correlating once optimised. |
| [`eval/periodicity`](metrics/periodicity.md) | The cepstral buzz proxy, kept as a diagnostic only because it was gamed when weighted. The Goodhart's-law case study. |
| [`eval/babble`](metrics/babble.md) | Speaking into silence after key-down: how it is measured over the injected garbage tail, and why its 7.47 dB noise floor makes single readings useless. |
| [Condition columns](metrics/conditions.md) | `lsd_first1s` / `lsd_last1s` (the edges), `lsd_rx` (the noisy receiver), `lsd_drops`, and `@<mode>` per vocoder, including when per-mode columns stop being comparable. |
| [Loss terms](metrics/loss-terms.md) | Every `loss/*` key, the edges rule, signed versus absolute error, how to choose a weight, and the dormant GAN path. |

## Research: facts with sources

| Page | Summary |
|---|---|
| [Research index](research/index.md) | The research pages and the question each answers. |
| [AMBE primer](research/ambe-primer.md) | How the vocoder works: bit budget, per-frame parameters, what is discarded, the AMBE-3000 and its `RATEP` words, why YSF DN and DMR are one capture mode, measured timings, Codec 2 for M17, patent status. Open it for any codec or chip fact. |
| [Data sources](research/data-sources.md) | The voice corpora by tier (0–6), all commercially usable, with licence, rate, speakers, hours and the redistributable flag. Excluded corpora, attribution obligations, disk layout and fetch commands. Open it before touching training data. |
| [Getting the data](research/getting-the-data.md) | How to get every source the models trained on: the scripted studio corpora, Common Voice by hand, `prepare` and the twins, the three vocoders, the two third-party models (Vocos, UTMOS), the derived sets and the disk they take. |
| [Corpus inventory](research/corpus-inventory.md) | What is on the drive (counted 2026-09-13, corrected 2026-09-20): archives and their licences, the four prepared corpora (325 h), captured hours per mode, and what publishing the AMBE pairs would involve. |
| [Hardware throughput](research/hardware-throughput.md) | ThumbDV per-frame cost: stop-and-wait, one direction pipelined, and the two directions interleaved 3 + 3 (1.73× the two passes, and why 4 + 4 stalls). Capture time per tier and storage sizes. Open it to plan capture runs. |
| [Prior art](research/prior-art.md) | Literature survey: OSCE, Codec 2 / RADE, coded-speech GANs, bandwidth extension, small real-time models, three architecture candidates, Rust ML tooling and evaluation metrics. Open it before choosing a model or a metric. |

## Design: decisions and plans

| Page | Summary |
|---|---|
| [Design index](design/index.md) | System diagram, the decisions made (PCM first, AMBE on the chip and Codec 2 in software, one shared model or one per vocoder, profiles, latency variants, training targets, Rust throughout, speaker-disjoint splits) and open questions. Open it first for any design question. |
| [Data pipeline](design/data-pipeline.md) | The built data harness: data-root layout, `prepare` and the split rule, chip and software `capture`, `verify`, the augmentation layers (twins, `augment`, `recode`), the shard format and its options, and the two backup scripts. Open it to run or change any data stage. |
| [Training](design/training.md) | **Candidate 3, concretely:** the restorer + synthesiser design, its results and its order of work. Also the training targets, cloud economics, the libtorch stack and environment, inputs, profiles, losses, eval, run configs and directory, the dashboard and its API, the real-time bench, and the candidate-1 review. |
| [Real-time](design/realtime.md) | Where inference runs in astar, the three profiles and their budgets, the two latency variants, runtime options, and the streaming rules (end-of-stream flush, invalid-frame gating). |
| [Reproducing from the shared data](design/reproducing.md) | For a collaborator given the private Drive copy: what it holds, the three ways in (shard set, restored captures, from nothing), what each needs, and the rules for Common Voice-derived audio. |
| [The pipeline runtime](design/pipeline-runtime.md) | The restorer + synthesiser in Rust: the weights file, the exact features, the batch and streaming paths, the latency table (39 frames audio to audio), `unamblify bench --kind pipeline`, `unamblify restore`, and what is still Python. |
| [Demo page](design/demo.md) | How the Listen page is built: clip sources, the `unamblify demo` pipeline, rendering, layout and styling. |

## Notes: hard-fought findings

| Page | Summary |
|---|---|
| [Notes index](notes/index.md) | What the notes are for and how to write one. Individual notes are gitignored and appear only when served locally. |

## About

| Page | Summary |
|---|---|
| [What's in the repo](about/whats-in-the-repo.md) | Map of the repository workspace: Rust crates, vendor drivers, scripts, configs, datasets, docs, and the ZRDC paper. |
| [License](about/license.md) | AGPL-3.0-only for the code and the 0.0.1-beta weights, the LGPL-2.1 AND MIT `codec2` crate, the Vocos MIT notice, corpus licences, and what the weights inherit. |
