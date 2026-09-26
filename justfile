# unamblify — task runner. `just` lists recipes.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# ── Rust ────────────────────────────────────────────────────────────────────

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

# `detect_without_hardware_errors_not_hangs` is the vendored driver's own
# test and it calls `detect()`, which opens and probes any ThumbDV that is
# plugged in. This project's rule is that no test opens a serial port: with
# the sticks attached and idle it reached real hardware and sat in the FTDI
# close() for 21 minutes before failing its own 5 s bound. Skipped by name
# rather than by editing vendored code.
test:
    cargo test --workspace --locked -- --skip detect_without_hardware_errors_not_hangs

# The libtorch-free binary (prepare / capture / shard / verify / serve /
# runs / stats) must keep building for hosts without a torch wheel.
build-lite:
    cargo build --locked --no-default-features -p unamblify-cli

# Everything CI runs. Must be green before a merge. Needs `just train-env`
# once per checkout (the train crate links libtorch).
ci: fmt-check clippy test docs-build build-lite
    @echo "✓ ci: fmt + clippy + test + docs + no-default-features build passed"

# ── Training environment ────────────────────────────────────────────────────

# Create .venv-torch (uv, torch==2.13.0) and write the gitignored
# .cargo/config.toml that points cargo at it. Linux: `--gpu cpu|cuda|rocm`.
train-env *ARGS:
    scripts/train-env.sh {{ARGS}}

# ── The harness: data → capture → shards → train → dashboard ────────────────
# Every stage reads $UNAMBLIFY_DATA (default /Volumes/data/training_data/
# unamblify); pass `--data-root D` to override. `unamblify --help` for the
# rest of each verb's flags.

# Walk raw/ into prepared/ (16 kHz + 8 kHz WAVs and the manifest).
prepare *ARGS:
    cargo run --release -p unamblify-cli -- prepare {{ARGS}}

# Capture: `just capture --mode dstar [--port P] [--dry-run]` (ThumbDV) or
# `just capture --mode codec2-3200 [--jobs N]` (Codec 2 in software, no port).
# `just capture pause|resume|stop|status --mode dstar` controls a run.
capture *ARGS:
    cargo run --release -p unamblify-cli -- capture {{ARGS}}

# Pack training examples: `just shard --mode dstar --name seed-dstar`, or
# several modes in one set: `just shard --modes dstar,codec2-3200 --name mixed`.
shard *ARGS:
    cargo run --release -p unamblify-cli -- shard {{ARGS}}

# Train a run: `just train configs/seed-dstar-full-ll5.toml [--steps N]`.
train CONFIG *ARGS:
    cargo run --release -p unamblify-cli -- train --config {{CONFIG}} {{ARGS}}

# The dashboard on http://127.0.0.1:8787 (`--token T` to expose it).
serve *ARGS:
    cargo run --release -p unamblify-cli -- serve {{ARGS}}

# 20-step CPU train on synthetic data; fails unless the loss went down.
# What CI runs after the unit tests.
smoke:
    cargo run --locked -p unamblify-cli -- smoke

# Manifest summaries for prepared/ and every captured mode.
stats *ARGS:
    cargo run --release -p unamblify-cli -- stats {{ARGS}}

# ── Docs (zensical) ─────────────────────────────────────────────────────────

# Serve the docs site with live reload at http://localhost:8000.
docs:
    uvx zensical serve

# Build the site (docs -> docs/.site) exactly as CI does; --strict fails on
# broken links and nav entries that point nowhere.
docs-build:
    ./ci/build-docs.sh

# ── Training data ───────────────────────────────────────────────────────────

# Download + verify + extract the voice corpora in data/sources.tsv into
# $UNAMBLIFY_DATA (default /Volumes/data/training_data/unamblify).
# `--tier 0` = the ~5.5 GB seed, `--tier 1` adds the core set; see
# docs/research/data-sources.md.
fetch-data *ARGS:
    scripts/fetch-datasets.sh {{ARGS}}
