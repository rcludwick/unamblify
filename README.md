# unamblify

A real-time neural post-filter, written in Rust, that takes the robotic
voice out of AMBE digital-voice audio (D-STAR, DMR, System Fusion, NXDN)
and out of Codec 2 (M17).

AMBE keeps speech intelligible at 2400 bit/s by sending a tiny parametric
model of the voice: pitch, a few voiced/unvoiced flags and a spectral
envelope at the harmonics. The decoder invents everything else. unamblify
trains a network on pairs of clean speech and the same speech after a real
AMBE-3000 chip has encoded and decoded it. That network then runs on the
receive side. The aim is a voice closer to a clean SSB signal, and if the
model can learn it, closer to AM broadcast.

**Status (2026-09-25).** The data chain runs on real data. Three ThumbDVs
have pushed about 468 000 D-STAR and 363 000 YSF/DMR utterances through
the AMBE-3000. Both Codec 2 modes are captured in software over the full
2.2 M-utterance corpus. A software D-STAR vocoder (`dstar+perens`) gives a
second implementation of one bitstream.

The best system is a **restorer** (codec output → clean wideband
spectrum) feeding a **waveform synthesiser** (a fine-tuned Vocos). On
predicted MOS it scores **3.57** against the codec's 2.22 and the
recording's 4.14, measured 2026-09-25 on a faithful input. (The 3.05
reported before was measured on a low-passed input, experiment #38.) In a
blind listening test it scored **4.6 of 5** against the codec's 1.6–1.9
and the recording's 4.9–5.0, on both AMBE modes and with no gender gap.
It streams at about 400 ms of latency. Training is still Python
(`scripts/spike/`). Inference runs in Rust (`unamblify restore`, see
[the pipeline runtime](docs/design/pipeline-runtime.md)). The weights are
open: [0.0.1-beta](https://github.com/rcludwick/unamblify/releases/tag/v0.0.1-beta),
under the AGPL-3.0 like the code.

The older design, a 2.1 M-parameter adaptive filter (candidate 1), is
still built in Rust and runs in real time at 100 ms. It tops out near 2.1
on predicted MOS. Band-swap probes showed why: its low band was worth
nothing (experiment log #33). The
[design page](docs/design/training.md#candidate-3-concretely) has the
order of work and the [experiment log](docs/theory/experiment-log.md)
has every measurement.

## Docs

The documentation site (research, data sources, design) is a
[Zensical](https://zensical.org) project:

```sh
uvx zensical serve      # or: just docs   →  http://localhost:8000
```

## Reproducing it

There are three ways in, from least to most work. All three are written up
in [Reproducing from the shared data](docs/design/reproducing.md).

1. **Train from a shard set.** A shard set is self-contained. It holds
   codec output, clean targets and an index. With the repository, one
   shard set and its run config, a training run repeats exactly. No chip,
   no corpus download and no `prepare`.
2. **Restore the chip captures.** The AMBE captures are the part that
   cost weeks of hardware time. They are kept as tar chunks with per-file
   checksums. `scripts/backup-chunks.sh restore` puts them back under
   `captured/`, and from there you can build shard sets of your own.
3. **From nothing.** Fetch the corpora, run `prepare`, and capture with
   your own ThumbDV:

    ```sh
    UNAMBLIFY_DATA=/path/with/space just fetch-data --tier 1
    ```

    See `docs/research/data-sources.md` for what is fetched, why, and
    under which licences.

The shared data is not public. It holds audio derived from Common Voice,
whose terms forbid re-hosting it. If you were given access, train on it,
keep it private, and do not pass it on.

## Quick start

You need stable Rust, [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems). The trainer links libtorch from the
`torch==2.13.0` wheel, so set that up once per checkout:

```sh
just train-env                     # .venv-torch + the gitignored .cargo/config.toml
just smoke                         # 20 CPU steps on synthetic data; the CI gate
```

Then, with a ThumbDV on the USB bus and the corpora fetched:

```sh
just prepare                                  # raw/ → prepared/ (16 kHz + 8 kHz WAVs, manifest)
just capture --mode dstar                     # prepared 8 kHz → chip → captured/dstar/ (weeks)
just capture --mode codec2-3200               # Codec 2 runs in software: no chip, no port
just capture status --mode dstar              # or pause | resume | stop
just shard --mode dstar --name seed-dstar     # fixed-length training examples
just train configs/seed-dstar-full-ll5.toml   # runs/<id>/ with metrics, checkpoints, eval audio
just serve                                    # dashboard on http://127.0.0.1:8787
```

Every verb takes `--data-root`. The default is `$UNAMBLIFY_DATA`, else
`/Volumes/data/training_data/unamblify`. `unamblify --help` lists the
rest. `just ci` runs fmt, clippy, tests, a strict docs build and the
libtorch-free `--no-default-features` build. `docs/design/` documents the
formats, the run directory and the dashboard API.

## License

AGPL-3.0-only. See [LICENSE](LICENSE). Three components keep their own
licences: the vendored ThumbDV driver (MIT OR Apache-2.0), the vendored
software D-STAR vocoder (LGPL-3.0-or-later) and the `codec2` crate
(LGPL-2.1 AND MIT). [docs/about/license.md](docs/about/license.md) has
the details.
