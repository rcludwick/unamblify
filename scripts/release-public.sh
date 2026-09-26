#!/usr/bin/env bash
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
#
# Publish a release to the public repository as one commit.
#
# Development happens in rcludwick/unamblify-private with its full history.
# The public rcludwick/unamblify gets one commit per release: the tree of a
# private ref (default main) on top of the last public commit. Nothing of the
# private history is pushed.
#
#   scripts/release-public.sh "release 0.2: <what changed>"          # main
#   scripts/release-public.sh --ref v0.2 "release 0.2: <what changed>"
#   scripts/release-public.sh --dry-run "..."                         # show, don't push
#
# Env: PUBLIC_REMOTE (default `public`, git@github.com:rcludwick/unamblify.git).
set -euo pipefail

remote="${PUBLIC_REMOTE:-public}"
ref=main
dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) ref="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
msg="${1:?usage: release-public.sh [--ref R] [--dry-run] \"<message>\"}"

git fetch -q "$remote" main
parent="$(git rev-parse "$remote/main")"
tree="$(git rev-parse "$ref^{tree}")"
if [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
  echo "public main already holds $ref's tree; nothing to release"
  exit 0
fi

echo "files changed since the last release:"
git diff --stat "$parent" "$tree" | tail -1
if [ "$dry" = 1 ]; then
  echo "dry run: not pushing"
  exit 0
fi
commit="$(git commit-tree "$tree" -p "$parent" -m "$msg")"
git push "$remote" "$commit:refs/heads/main"
echo "released $commit to $remote/main"
