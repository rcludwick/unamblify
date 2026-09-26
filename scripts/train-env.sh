#!/usr/bin/env bash
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
#
# Create the libtorch environment the train crate links against and write
# the workspace .cargo/config.toml that points cargo at it.
#
#   scripts/train-env.sh [--gpu cpu|cuda|rocm] [--python 3.12] [--venv DIR] [--no-config]
#
# tch-rs 0.26 pins libtorch 2.13.0, so the venv installs exactly
# `torch==2.13.0`: from PyPI on macOS (CPU + MPS), from the PyTorch index
# on Linux chosen by --gpu (cpu -> whl/cpu, cuda -> whl/cu128, rocm ->
# whl/rocm6.4). The generated .cargo/config.toml (gitignored; see
# .cargo/config.toml.example) sets LIBTORCH to the wheel's torch/ directory
# and adds an rpath link arg for the host target so binaries and test
# executables find libtorch at run time without DYLD_/LD_LIBRARY_PATH.
#
# Re-running is idempotent: `uv pip install` is a no-op when the pin is
# already satisfied and the config is rewritten in place.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="cpu"
python="3.12"
venv="$root/.venv-torch"
write_config=1
torch_version="2.13.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --gpu=*) gpu="${1#--gpu=}"; shift ;;
    --python) python="$2"; shift 2 ;;
    --python=*) python="${1#--python=}"; shift ;;
    --venv) venv="$2"; shift 2 ;;
    --venv=*) venv="${1#--venv=}"; shift ;;
    --no-config) write_config=0; shift ;;
    -h|--help) sed -n '5,22p' "$0"; exit 0 ;;
    *) echo "train-env.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$gpu" in
  cpu|cuda|rocm) ;;
  *) echo "train-env.sh: --gpu must be cpu, cuda or rocm (got '$gpu')" >&2; exit 2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "train-env.sh: uv is required (brew install uv / pipx install uv)" >&2
  exit 1
fi

os="$(uname -s)"
index_args=()
case "$os" in
  Darwin)
    if [ "$gpu" != "cpu" ]; then
      echo "train-env.sh: --gpu $gpu is Linux-only; macOS wheels carry CPU + MPS" >&2
      exit 2
    fi
    ;;
  Linux)
    case "$gpu" in
      cpu)  index_args=(--index-url https://download.pytorch.org/whl/cpu) ;;
      cuda) index_args=(--index-url https://download.pytorch.org/whl/cu128) ;;
      rocm) index_args=(--index-url https://download.pytorch.org/whl/rocm6.4) ;;
    esac
    ;;
  *)
    echo "train-env.sh: unsupported OS $os" >&2
    exit 1
    ;;
esac

[ -x "$venv/bin/python" ] || uv venv --python "$python" "$venv"
uv pip install --python "$venv/bin/python" "${index_args[@]}" "torch==$torch_version"

torch_dir="$("$venv/bin/python" - <<'PY'
import pathlib, torch
print(pathlib.Path(torch.__file__).resolve().parent)
PY
)"
torch_lib="$torch_dir/lib"
[ -d "$torch_lib" ] || { echo "train-env.sh: $torch_lib missing" >&2; exit 1; }

host="$(rustc -vV | sed -n 's/^host: //p')"

if [ "$write_config" = 1 ]; then
  cfg="$root/.cargo/config.toml"
  mkdir -p "$root/.cargo"
  cat > "$cfg" <<EOT
# Written by scripts/train-env.sh — do not edit, re-run the script.
# Machine-specific (absolute paths); gitignored. See config.toml.example.

[env]
# torch-sys resolves include/ and lib/ under this directory; no python
# needs to be on PATH at build time.
LIBTORCH = "$torch_dir"

[target.$host]
rustflags = ["-C", "link-arg=-Wl,-rpath,$torch_lib"]
EOT
  echo "wrote $cfg"
fi

cat <<EOT
# libtorch $torch_version ($os, $gpu) in $venv
# For builds that do not go through cargo's config (CI shells, IDEs):
export LIBTORCH_USE_PYTORCH=1
export PATH="$venv/bin:\$PATH"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$torch_lib"
# or, equivalently, without python on PATH:
export LIBTORCH="$torch_dir"
EOT
