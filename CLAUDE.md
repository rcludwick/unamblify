# unamblify

A real-time neural post-filter, in Rust, that removes the robotic voice
from AMBE / AMBE+2 digital-voice audio (D-STAR, DMR, System Fusion, NXDN)
and from Codec 2 (M17). Trained on pairs of clean speech and the same
speech after a real AMBE-3000 (ThumbDV / DVstick) encode → decode, or
after the `codec2` crate in software for the M17 modes. Will eventually
live inside `~/dev/astar` as its receive-side post-filter. **AGPL-3.0-only** (may
change; the `LICENSE` at a commit governs that commit).

Docs are a Zensical site (`just docs`, http://localhost:8000).

**Start at `docs/map.md`.** It is the index of every doc page with a
one-line summary of what each holds, so you can find a fact (a rate word,
a corpus licence, a latency budget) without reading the whole site. Open
the map, then the one page it points to. When you add, remove, rename or
materially change a page, update its row in `docs/map.md` in the same
commit; `ci/check-docs-map.sh` (run by `ci/build-docs.sh` and therefore by
`just ci`) fails if a page is missing from the map.

## Layout

```
crates/unamblify/        core: frame constants, VocoderMode (dstar, ysf-dmr,
                         codec2-3200, codec2-1600; family(), frame_samples(),
                         frame_bytes(), is_software(), label(); ALIASES ysf-dn /
                         ysf / dmr parse as ysf-dmr), and the serde types
                         every crate shares (manifest rows, RunConfig, run
                         status/metrics, ShardIndex + ExampleLayout), the split
                         rule, utt_key builders, speaker_id, aug (AugRecord /
                         CaptureAug / ChainParams), channel (drops, ber, mute
                         codewords). No I/O.
crates/unamblify-audio/  WAV/FLAC readers, s16 writer, rubato resampler,
                         loudness/trim, STFT/mel, LSD/mel-L1/SI-SDR/seg-SNR,
                         hnr (harmonic-to-trough, reference-gated), plosive
                         (burst/closure/rise), xcorr lag, chain (noise mixing +
                         overdriven-mic chain), rx (receive-side noise). Pure
                         Rust, no tch.
crates/unamblify-data/   prepare (raw → prepared/, plus twins.rs for the noisy /
                         overdriven twins), capture (encode → decode
                         through the vocoder::Vocoder trait: ThumbDV with the two
                         directions interleaved (pipeline::round_trip, 3 encodes +
                         3 decodes in flight, 1.73x; --sequential is the old two
                         passes) for the AMBE modes, the codec2 crate on --jobs threads
                         for the Codec 2 modes; canary, pause/resume/stop via
                         control.json, status.json; Stage::Augment = the decode-only
                         augment stage, pure parts in augment.rs), shard, verify. Generic over
                         ambe_thumbdv::Transport; sim::SimTransport is the
                         --dry-run / test chip. Feature `codec2` (default on)
                         pulls in the LGPL-2.1 AND MIT codec2 crate.
crates/unamblify-train/  tch-rs model (candidate 1), losses, pipeline + shard
                         loaders, trainer loop, checkpoints, eval + audio
                         rendering, smoke. THE ONLY CRATE THAT LINKS LIBTORCH.
crates/unamblify-web/    the dashboard: axum server, run/capture supervisor
                         (re-execs current_exe()), file poller, SSE, embedded UI
                         (ui/: index.html, app.js, style.css, uPlot). No tch.
crates/unamblify-cli/    the `unamblify` binary. Feature `train` (default on)
                         pulls in unamblify-train; --no-default-features gives
                         prepare/capture/shard/verify/serve/runs/stats only.
vendor/ambe-thumbdv/     the ThumbDV driver, vendored verbatim from astar
                         (MIT OR Apache-2.0 island; never edited here).
configs/                 run configs: default.toml (every key), smoke.toml,
                         seed-dstar-{full,lite}-ll{5,20}.toml, seed-ysf-dmr-full-ll5.toml,
                         seed-codec2-{3200,1600}-full-ll5.toml, generalist-*.toml,
                         natural-large-full-ll5.toml (noise head + hnr_w),
                         natural-mod-large-full-ll5.toml (+ noise_mod, transient_w,
                         mode_weights, the re-split set), eval-clips.txt.
scripts/                 fetch-datasets.sh, train-env.sh (uv venv + torch 2.13.0
                         + the gitignored .cargo/config.toml), mos.py (UTMOS22
                         predicted MOS over a checkpoint's clips; own venv),
                         backup-chunks.sh (pack | push | verify | status | restore:
                         the chip captures as checksummed tar chunks on a private
                         Drive folder), backup-corpus.sh (rsync mirror),
                         spike/ (the restorer + waveform-synthesiser pipeline of
                         experiments #33–#36 as throwaway Python, its measurements,
                         and listening/ — the blind listening test; README there).
docs/                    the Zensical site, published as-is. map.md is the index;
                         theory/ metrics/ research/ design/ demo/ notes/ about/
                         (design/reproducing.md: how a collaborator uses the Drive copy)
docs/notes/              "memories" — hard-fought findings. GITIGNORED except index.md.
docs/superpowers/        specs/plans working material. Gitignored, not published.
data/sources.tsv         the corpus manifest the fetch script reads.
ci/build-docs.sh         the one docs build entry point (local and CI).
```

## Environment and running each stage

The train crate links libtorch, so the workspace does not build until
`just train-env` has run once per checkout (creates `.venv-torch` with
`torch==2.13.0` and writes the gitignored `.cargo/config.toml` with
`LIBTORCH` + an rpath). Never export `RUSTFLAGS` in that shell: it replaces
the config's per-target rustflags and drops the rpath. Worktrees each need
their own venv (the script roots at its own checkout), or pass
`--venv /path/to/existing/.venv-torch`. Use a per-worktree
`CARGO_TARGET_DIR`.

```
just train-env                              # once per checkout
just prepare [--corpus vctk] [--force]      # raw/ → prepared/ (default: the 4 studio corpora; --corpus hifi_tts and
                                            # --corpus common_voice are opt-in; Hi-Fi TTS drops its two 3.5-MOS speakers)
just prepare --noise-share 0.25 --noise-sets demand,musan --ham-chain-share 0.25  # + twins <key>+n../+h..
just prepare --twins-only --underdrive-share 0.02   # (--twins-only: skip the raw walk + 4M stats)
just prepare --underdrive-share 0.02         # + underdriven twins <key>+u.. (15-30 dB down, NOT renormalised:
                                            # the low level is the augmentation). capture/recode --only-twins
                                            # points a chip at the twins instead of the whole base corpus
unamblify prepare --resplit                 # re-apply the split rule to the manifest, no audio; reshard after
just capture --mode dstar [--port P] [--dry-run] [--limit N] [--sequential] [--order random|balanced|design]
                                            # random (default): every key at a hashed position, so the captured
                                            # set is always a uniform sample of everything prepared
just capture --mode dstar --warmup-mix [--cold-share 0.34] [--warmup-seed 1]
                                            # AMBE only: per utterance (hashed from its key) capture cold (chip reset
                                            # first, so the first frames carry the keyup pitch-lock transient) or warm
                                            # (encoder locked onto the voice first); row.warm_state records which
just capture --mode codec2-3200 [--jobs N] [--limit N]   # software, no port
just capture status|pause|resume|stop --mode dstar
unamblify augment --mode codec2-3200 --kind drops|ber [--rate R] [--burst 1..3] [--subst mute|repeat|erase] [--share S] [--limit N]
                                            # decode-only sibling captured/<mode>+<kind>/ (chip modes: --port / --dry-run)
unamblify recode --mode dstar --kind perens [--jobs N] [--limit N]
                                            # sibling captured/dstar+perens/: the prepared audio
                                            # re-encoded by the vendored software D-STAR vocoder
                                            # (vendor/ham-digital-modes). Software only, no port.
just shard --mode dstar --name seed-dstar [--kinds base,drops]   # → shards/seed-dstar/
                                            # --kinds also takes a mode-scoped sibling,
                                            # dstar+perens, for a kind that exists for
                                            # one mode only
just shard --modes dstar,codec2-3200 --name mixed-dstar-codec2 [--no-balance] [--split dev] [--max-utterances N]  # one set, several modes
                                            # --twin-share S (with --max-utterances): reserve that share of
                                            # each capture set's draw for twins; uniform otherwise starves
                                            # them in the 2 M-utterance Codec 2 sets
                                            # --min-drop-rate R: of a drops sibling, draw only rows augmented at
                                            # a loss rate >= R (one sibling can hold a light and a heavy pass)
                                            # --corpora libritts_r,vctk,ljspeech,voicebank_demand: studio-quality
                                            # targets only (Common Voice targets score 3.15; experiment #34)
just train configs/seed-dstar-full-ll5.toml [--steps N] [--device mps] [--shards NAME]
unamblify train --config C --init-from <run-id>[:step]   # a new run on another run's
                                            # weights (step 0, fresh optimiser): how a
                                            # specialist starts from the generalist
just serve [--token T]                      # http://127.0.0.1:8787
unamblify infer --run-dir D --step N --key K [--mode M] [--out P]   # one utterance through a checkpoint
unamblify runs list|show ID|stop ID|resume ID [--json]
unamblify stats [--json] ; unamblify verify --mode dstar [--kind drops]
unamblify measure --dir D [--kinds degraded,out] [--json]   # plosive + sibilant columns over rendered
                                            # clips (<stem>@<mode>.<kind>.wav vs .clean.wav), standalone
unamblify migrate-modes [--dry-run]         # captured/ysf-dn/ → ysf-dmr/ after a mode rename
unamblify bench [--profile full] [--width W] [--threads 1]   # the real-time gate:
                                            # ms per 20 ms frame on CPU, before training
unamblify bench --kind pipeline [--threads 1] [--weights W]  # the restorer + synthesiser (crates/unamblify-train/src/
                                            # pipeline/): features / restorer / synth ms per 20 ms; random weights
                                            # of the exported shapes unless --weights (scripts/spike/export_weights.py)
unamblify restore --weights W --in <8 kHz wav> --mode M --out <24 kHz wav>   # one clip through the pipeline
just smoke                                  # the CI gate: 20 synthetic steps
```

Modes: `dstar | ysf-dmr` (AMBE, through the chip; `ysf-dmr` is one
capture for YSF DN and DMR, which carry the same 49 voice bits — DMR is
not a capture mode, its rate word and null frame stay as constants for
the future `ber` framing) and
`codec2-3200 | codec2-1600` (Codec 2 for M17, in software: no `--port`,
`--jobs N` threads, 20 ms / 40 ms frames — use `mode.frame_samples()`,
never a bare 160). Channel frames of every mode live in `.ambe` files. A prepared row with
`parent` is an augmented twin: its own clean row is the noisy input and
the parent's file is the target (`UtteranceRow::target_key`); a capture
set is `captured/<mode>[+<kind>]/` (`DataRoot::capture_dir`, `CaptureDir`)
and `[data] kinds` / `shard --kinds` pick which sets a run draws from
(a sibling is either a decode-only `augment` or a `recode`, the same audio
through a second implementation of the codec — `dstar+perens` is the only
one, 2.18 MOS against the chip's 2.56);
`[augment] rx_share` adds receive-side noise in the loaders. A run names
its modes (`[data] mode` for one, `[data] modes = [...]` for a shared
model over several; a shard set built with `--modes` holds them all,
each example tagged with its mode index, balanced per split) and
`[model] mode_embed = true` tells the model which one it hears; eval
reports `eval/lsd@<mode>` per mode beside the mean.
`[model] noise_head = true` adds the aperiodic excitation path and
`[train] hnr_w` the loss that rewards using it (the naturalness
pair: a filter alone cannot fill the troughs between harmonics, which is
the residual robotic quality — `eval/hnr` tracks the mean excess in dB,
0 = as periodic as clean speech, and `eval/hnr_abs` the per-frame
absolute excess, which is the one that has to fall: the first noise-head
model reached a mean of ~0 with 2.6 dB per frame, a coincidence of
signs, not naturalness. `[model] noise_mod` modulates the noise by the
periodic path's envelope and gives it a gain per 1 ms; `[train]
transient_w` is the stop-consonant term and `eval/plosive_*` its
columns; `[data] mode_weights` sets each mode's share of the batches;
`[model] erasure_in` feeds the per-frame lost-frame mask a drops shard
set carries into the context conv (so a gap is seen a lookahead early);
`periodicity_w` is a gameable proxy, kept as a diagnostic at 0).
`scripts/mos.py` (its own `.venv-mos`) scores rendered clips with a
learned MOS predictor — clean 4.14, codec 1.79, the filter model 2.12
(its ceiling: experiment #33), the restorer + synthesiser spike **3.57**
(2026-09-25, on a faithful input — codec 2.22 on the same clips; `scripts/spike/`; the owner's blind listening test puts it at
4.6 of 5, #36) — and is the yardstick for comparing architectures; it
compresses the scale and cannot rank clips within one system, so close
variants are judged by ear (`scripts/spike/listening/`). Every trainer
column is a proxy. Score several
checkpoints: the best by MOS has been neither the last nor the best by
any proxy, so runs keep them all (`[ckpt] keep = 40` at every 2 000).
Every verb takes `--data-root D` (else `$UNAMBLIFY_DATA`). `serve` spawns
`unamblify train` / `unamblify capture` / `unamblify infer` as child
processes of itself, so it must be the `unamblify` binary, not
`unamblify-web`. The dashboard writes nothing under the data root except
`cache/spec/` (sample spectrograms) and `cache/host-metrics.jsonl` (24 h
of drive temperature, drive I/O rate, CPU and GPU samples, so the graphs survive a restart). Runs live under
`<data-root>/runs/<YYYYMMDD-HHMMSS-name>/`; `docs/design/training.md`
documents the run directory, metrics keys and the HTTP API,
`docs/design/data-pipeline.md` the manifests, capture rules and shard
format.

## Training data — lives on the 8 TB drive, never in git

`UNAMBLIFY_DATA` = `/Volumes/data/training_data/unamblify` (default in the
script). Layout: `archives/ raw/ checksums/ logs/`, later `prepared/
captured/`. `just fetch-data --tier 0` is the ~5.5 GB seed, `--tier 1` adds the
core set; both are resumable; check `logs/` there
before re-running anything. Corpus licences and attribution obligations are
in `docs/research/data-sources.md` — **every corpus must be usable commercially** (CC BY, CC0, Apache,
public domain; CC BY-SA only in the separate `+sa` weight line) — never
add a CC BY-NC or research-only corpus, not even for evaluation.
**Private backup and training storage are fine; public redistribution is
not.** Common Voice comes from the Mozilla Data Collective. Keeping it in
private cloud storage is allowed — a backup, or storage attached to a GPU
box for training, the way any training data lives off this machine — and
it may be shared with a specific named collaborator. What stays forbidden
is *public* redistribution and de-anonymisation: never make Common Voice
(or any click-through corpus) publicly downloadable, never commit its
clips to git or ship them in demo clips, never serve it off loopback to
the open internet, and never attempt to re-identify a speaker (use only
the opaque `client_id`, and never link it to anything). `redistributable
= false` in the manifest marks a corpus that must stay out of the
*public* publishing paths (the docs site, demo clips, a public link); it
does not block a private backup or a collaborator share. Details:
docs/research/data-sources.md.

## Hardware

Three ThumbDV/DVstick 30s (AMBE-3000R, FTDI FT230X, VID 0403 PID 6015,
460800 baud); `--port` repeats, one worker per stick, and one capture
mode holds a stick at a time. The driver is `vendor/ambe-thumbdv`, vendored verbatim from
astar's copy (upstream `rcludwick/ambe`, MIT/Apache-2.0, never edited here —
see its `VENDORED.md`); its `serialport` dependency needs libudev on Linux
(`apt install libudev-dev pkg-config`), nothing extra on macOS. Rate words
for D-STAR / YSF DN / DMR are
in astar's `crates/astar-codec/src/{ambe,ysf,dmr}.rs` and reproduced in
`docs/research/ambe-primer.md` (the DMR word is kept but never captured
with). Capture runs for weeks — everything that
touches the chip must be resumable and must verify with a canary clip
(`docs/design/data-pipeline.md`).

Never point a serial opener at a port that is not the FTDI VID/PID match:
opening a USB radio interface's tty asserts RTS and keys a transmitter.
`unamblify_data::chip::candidate_ports_from` is astar's VID/PID
intersection rule; do not loosen it. No test opens a serial port;
`--dry-run` and the tests use `sim::SimTransport`.

## Conventions

- **Worktrees.** Implementation work happens in `.worktrees/<slug>` on
  branch `work/<slug>` (`git worktree add .worktrees/<slug> -b work/<slug>`).
  `main` in the main checkout is for merges and doc commits only.
- **Green before merge.** `just ci` = `cargo fmt --check`, `clippy -D
  warnings` (pedantic), `cargo test --locked`, a `--strict` docs build, and
  `cargo build --no-default-features -p unamblify-cli` (the libtorch-free
  binary). CI also runs `unamblify smoke`.
- **Conventional commits** with scopes: `feat(data):`, `fix(capture):`,
  `docs(research):`, `chore(ci):`. No AI attribution anywhere — not in
  commits, PRs, comments, or docs. Never write "generated by".
- **Docs-first.** A research finding lands in `docs/research/`; a design
  change lands in `docs/design/` alongside or before the code. Research
  pages carry a dated provenance admonition and cite sources.
- **Memories.** When something costs real effort to discover (a chip
  quirk, a training divergence, a metric that lied), write
  `docs/notes/YYYY-MM-DD-<slug>.md`: takeaway in bold, evidence, what to do.
- **Rust.** Stable toolchain, edition 2024, SPDX header on every source
  file, no `unwrap` outside tests, no allocation on the real-time path.
- **Datasets scripts are bash + curl + manifests**, on purpose: anyone
  should be able to recreate the corpus without building the project.
- macOS shell quirks: no `timeout`; zsh chokes on bare `==`; the Bash tool's
  cwd resets between calls — use absolute paths.

## Repositories, CI and Pages

Two GitHub repositories. **`rcludwick/unamblify-private`** (remote
`origin`) is where development happens, with the full history.
**`rcludwick/unamblify`** (remote `public`) is public and gets one commit
per release: `scripts/release-public.sh "<message>"` pushes the tree of
private `main` on top of the last public commit, and nothing of the
private history. Everything in the tree is therefore public at the next
release, so the Common Voice and no-AI-attribution rules above apply to
every commit, not just to releases.

`ci.yml` runs on GitHub-hosted runners in both repositories (apt
`libudev-dev` for serialport, uv + `scripts/train-env.sh --gpu cpu` for
libtorch, then fmt / clippy / test / smoke / no-default-features build,
on the latest stable Rust, so a new clippy lint can fail CI before it
fails locally: `rustup update stable`). `docs-pages.yml` builds the site on
the IONOS runner box (`runs-on: [self-hosted, ionos]`, registered per
repository on `rcludwick/unamblify`) and deploys it to GitHub Pages at
https://rcludwick.github.io/unamblify/. It runs on push to main and
manual dispatch only, never on a pull request, so a fork's code never
reaches the self-hosted box. It is skipped in the private repository,
where Pages is unavailable on the free plan.
