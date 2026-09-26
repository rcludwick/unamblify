#!/usr/bin/env bash
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
# Every Markdown page under docs/ must be linked from docs/map.md, so the map
# stays a complete index agents and people can trust. Gitignored notes
# (docs/notes/*.md except index.md) and superpowers material are exempt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAP="$ROOT/docs/map.md"
missing=0
while IFS= read -r f; do
  rel="${f#"$ROOT"/docs/}"
  case "$rel" in
    map.md) continue ;;
    notes/*) [[ "$rel" == notes/index.md ]] || continue ;;
    superpowers/*|.site/*) continue ;;
  esac
  if ! grep -Fq "($rel)" "$MAP"; then
    echo "docs/map.md is missing an entry for docs/$rel" >&2
    missing=1
  fi
done < <(find "$ROOT/docs" -name '*.md' -not -path '*/.site/*' | sort)
if (( missing )); then
  echo "Add a row with a one-line summary to docs/map.md (see CLAUDE.md)." >&2
  exit 1
fi
echo "docs map: complete"
