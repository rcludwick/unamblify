# What's in the repo

A map of the repository layout, describing each crate, tool, directory, and configuration file.

## Crates (`crates/`)

The core functionality is split across multiple focused Rust crates:

* **`unamblify`** (`crates/unamblify`): Core shared domain definitions and types. Contains vocoder modes (`dstar`, `ysf-dmr`, `codec2-3200`, `codec2-1600`), frame constants, serialization schemas (manifests, shard layouts, run configurations, metrics), audio augmentation parameters, and channel degradation models (packet drops, bit error simulation). Does not perform direct I/O.
* **`unamblify-audio`** (`crates/unamblify-audio`): Pure-Rust audio processing and evaluation engine. Implements WAV and FLAC reading, sample rate conversion via `rubato`, STFT and mel-filterbanks, objective quality metrics (log-spectral distance, mel-L1, SI-SDR, harmonic-to-trough ratio, plosive burst metrics, sibilance measures), cross-correlation lag alignment, and acoustic mixing. Does not depend on LibTorch.
* **`unamblify-data`** (`crates/unamblify-data`): Data processing pipeline. Handles audio normalization and downsampling (`prepare`), hardware and software vocoder execution (`capture`), training shard construction (`shard`), dataset integrity checking (`verify`), decode-only augmentation such as lost frames (`augment`), and re-encoding through the software D-STAR vocoder (`recode`). Connects to AMBE hardware via the ThumbDV transport and drives Codec 2 in software.
* **`unamblify-train`** (`crates/unamblify-train`): Model architecture and training harness. Implements the neural network architectures, the restorer + waveform synthesiser runtime (`unamblify restore`), custom loss functions, data shard loaders, checkpoint serialization, and audio evaluation rendering. This is the only crate that links against LibTorch (`tch-rs`).
* **`unamblify-web`** (`crates/unamblify-web`): Local web dashboard. Built on `axum`, it serves a real-time monitoring interface with live training metrics, loss graphs, audio playback comparisons, and process supervisor capabilities for data capture runs.
* **`unamblify-cli`** (`crates/unamblify-cli`): The primary binary entry point (`unamblify`). Exposes subcommands for data management, model training, evaluation, benchmark passes, and launching the web dashboard.

## Vendored drivers (`vendor/`)

* **`vendor/ambe-thumbdv`**: Serial transport driver for ThumbDV and DVstick AMBE hardware interfaces, vendored from the `astar` project.
* **`vendor/ham-digital-modes`**: Vendored floating-point AMBE vocoder from Bruce Perens' `hams_open` suite, providing an independent software reference for D-STAR encoding.

## Scripts and tooling (`scripts/`)

* **`fetch-datasets.sh`**: Downloads third-party speech corpora across tiers into the local dataset root.
* **`train-env.sh`**: Configures the Python virtual environment and exports the required LibTorch runtime paths for `cargo`.
* **`mos.py`**: Offline evaluator calculating predicted Mean Opinion Score (MOS) using the UTMOS22 model on generated audio samples.
* **`backup-chunks.sh`**: Utilities to archive, split, checksum, and restore large hardware capture collections.
* **`backup-corpus.sh`**: Synchronization script for mirroring raw audio corpora to backup drives.
* **`spike/`**: Experimental prototypes and validation scripts, including the restorer plus Vocos synthesiser proof of concept and blind listening test fixtures.

## Run configurations (`configs/`)

TOML configuration files specifying parameters for training runs:

* **`default.toml`**: Reference configuration documenting all model hyperparameters, loss function weights, dataset composition ratios, and evaluation intervals.
* **`smoke.toml`**: Minimal 20-step configuration executed on synthetic audio for CI validation.
* **`seed-*.toml`**: Baseline single-vocoder training configurations.
* **`generalist-*.toml`**: Multi-vocoder joint training configurations.
* **`natural-*.toml`**: Advanced configurations evaluating custom losses, noise heads, and transient preservation.

## Documentation (`docs/`)

Published as a static site using Zensical:

* **`theory/`**: Principles of vocoder compression, distortion mechanisms, and neural post-filter theory.
* **`metrics/`**: Definitions and evaluation methods for all objective metrics and loss terms.
* **`research/`**: Vocoder technical specifications, hardware throughput benchmarks, corpus licensing, and prior art surveys.
* **`design/`**: Architecture specifications, data pipeline implementation details, and latency profiles.
* **`about/`**: Licensing and project repository overview.

## Paper (`paper/zrdc/`)

The project write-up for Zero Retries Digital Communications (ZRDC):

* **`unamblify-zrdc.pdf`**: The built paper.
* **`unamblify-zrdc.tex`** and **`bibliography.bib`**: Its LaTeX source. Build it with `make` (Tectonic) or `make ENGINE=pdflatex`.

## Build and CI (`ci/` and root)

* **`justfile`**: Task runner recipes for building, testing, dataset management, hardware capture, and running docs dev servers.
* **`ci/build-docs.sh`**: Builds the documentation site and validates navigation consistency.
* **`ci/check-docs-map.sh`**: Enforces that all markdown pages are catalogued in `docs/map.md`.
* **`data/sources.tsv`**: Tab-separated registry of speech corpora with source URLs, licensing constraints, and audio specs.
