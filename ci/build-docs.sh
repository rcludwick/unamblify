#!/usr/bin/env bash
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
# Build the Zensical documentation site, exactly as CI does.
#
# Config is ./zensical.toml (TOML — not mkdocs.yml); docs_dir = docs,
# site_dir = docs/.site (gitignored). `--strict` turns a broken link or a nav
# entry pointing at a missing page into a failure instead of a half-built site.
#
# Uses `uvx zensical` when uv is installed, else a throwaway virtualenv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The docs map must list every page before we bother building.
"$ROOT/ci/check-docs-map.sh"

SITE_DIR="$(python3 - zensical.toml <<'PY'
import sys
try:
    import tomllib
    with open(sys.argv[1], "rb") as fh:
        print(tomllib.load(fh).get("project", {}).get("site_dir", "site"))
except Exception:
    print("site")
PY
)"

if command -v uvx >/dev/null 2>&1; then
  uvx zensical build --clean --strict
else
  VENV="${DOCS_VENV:-${TMPDIR:-/tmp}/unamblify-docs-venv}"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  python3 -m pip install --quiet --upgrade pip zensical
  zensical build --clean --strict
fi

[ -f "$ROOT/$SITE_DIR/index.html" ] || { echo "FAIL: no $SITE_DIR/index.html" >&2; exit 1; }
echo "docs: built $(find "$ROOT/$SITE_DIR" -type f | wc -l | tr -d ' ') files into $SITE_DIR"
