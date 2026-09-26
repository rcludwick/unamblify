#!/usr/bin/env bash
# backup-corpus.sh — mirror the unamblify corpus to a destination and
# validate the copy with SHA-256.
#
# The destination is any directory: a second disk, or a mounted Google
# Drive folder (the desktop app streams it to the cloud). The copy is
# rsync, so it is incremental and resumable — re-run it and only changed
# or missing files move. Validation is separate from the copy so it can
# run later, and the SHA256SUMS.txt it writes travels with the data, so
# whoever receives the folder checks their own download with a plain
#   shasum -a 256 -c SHA256SUMS.txt
#
# Subcommands:
#   copy      DEST   rsync SRC -> DEST (incremental, resumable)
#   manifest  DEST   write SHA256SUMS.txt (hash of every file) into DEST
#   verify    DEST   check DEST against SHA256SUMS.txt (the real validation)
#   diff      DEST   rsync dry-run: list anything that still differs
#   all       DEST   copy, then manifest, then verify
#
# Flags:
#   --include-archives       include archives/ (the downloaded tarballs);
#                            excluded by default — redundant with raw/ and
#                            re-fetchable from the original hosts
#   --exclude-common-voice   leave out raw/ Common Voice, for a copy meant
#                            for a wider audience than a private backup or
#                            a named collaborator (see data-sources.md)
#   --checksum               make `diff` compare bytes, not size+mtime
#   --dry-run                for `copy`: show what would move, move nothing
#   -h | --help
#
# Env:
#   UNAMBLIFY_DATA   source root (default /Volumes/data/training_data/unamblify)
#   RSYNC            rsync binary (default rsync; `brew install rsync` gives
#                    a newer one with better progress and resume than the
#                    macOS built-in)
#
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

SRC="${UNAMBLIFY_DATA:-/Volumes/data/training_data/unamblify}"
RSYNC="${RSYNC:-rsync}"
MANIFEST="SHA256SUMS.txt"

exclude_cv=0
exclude_archives=1
checksum=0
dry=0

die() { printf 'backup-corpus: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^#\{0,1\} \{0,1\}//'
  exit "${1:-0}"
}

# archives/ is the downloaded tarballs — redundant with the extracted raw/
# and re-fetchable from the original hosts, so it is left out by default.
ARCHIVE_GLOBS="archives"

# Common Voice lives under raw/; --exclude-common-voice keeps it out of a
# copy meant for a wider audience. One glob per known layout; extend if a
# future import lands elsewhere.
CV_GLOBS="raw/common_voice* raw/cv-corpus* raw/commonvoice*"

# Populate EX[] with rsync --exclude args (anchored at the transfer root)
# and PRUNE[] with a find expression, from whichever globs the flags select.
EX=()
PRUNE=()
build_excludes() {
  local globs=""
  (( exclude_archives )) && globs="$globs $ARCHIVE_GLOBS"
  (( exclude_cv )) && globs="$globs $CV_GLOBS"
  [ -n "$globs" ] || return 0
  local g first=1
  PRUNE=('(')
  for g in $globs; do
    EX+=(--exclude "/$g")
    (( first )) || PRUNE+=(-o)
    PRUNE+=(-path "./$g")
    first=0
  done
  PRUNE+=(')' -prune -o)
}

cmd_copy() {
  local dest=$1
  [ -d "$SRC" ] || die "source $SRC does not exist"
  mkdir -p "$dest"
  local args=(-rlt --partial --human-readable --stats)
  (( dry )) && args+=(--dry-run --itemize-changes)
  echo "==> copy  $SRC/  ->  $dest/"
  "$RSYNC" "${args[@]}" ${EX[@]+"${EX[@]}"} "$SRC/" "$dest/"
}

cmd_manifest() {
  local dest=$1
  [ -d "$dest" ] || die "destination $dest does not exist (run copy first)"
  local out="$dest/$MANIFEST"
  echo "==> manifest  hashing $SRC  ->  $out"
  echo "    (reads every file once; on a large corpus this takes a while)"
  ( cd "$SRC" && find . ${PRUNE[@]+"${PRUNE[@]}"} -type f ! -name "$MANIFEST" -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 shasum -a 256 ) > "$out.tmp"
  mv "$out.tmp" "$out"
  echo "    $(wc -l < "$out" | tr -d ' ') files hashed"
}

cmd_verify() {
  local dest=$1
  local man="$dest/$MANIFEST"
  [ -f "$man" ] || die "no $MANIFEST in $dest (run manifest first)"
  echo "==> verify  checking $dest against $MANIFEST"
  if ( cd "$dest" && shasum -a 256 -c "$MANIFEST" ); then
    echo "    OK: every file matches"
  else
    die "MISMATCH: some files failed the checksum (see the FAILED lines above)"
  fi
}

cmd_diff() {
  local dest=$1
  [ -d "$dest" ] || die "destination $dest does not exist"
  local args=(-rltn --itemize-changes)
  (( checksum )) && args+=(--checksum)
  echo "==> diff  $SRC/  vs  $dest/  ($( ((checksum)) && echo byte-for-byte || echo size+mtime ))"
  local out
  out="$("$RSYNC" "${args[@]}" ${EX[@]+"${EX[@]}"} "$SRC/" "$dest/")"
  if [ -z "$out" ]; then
    echo "    OK: destination is identical to the source"
  else
    printf '%s\n' "$out"
    die "destination differs from the source (lines above; '>' = needs copying)"
  fi
}

main() {
  local sub="" dest=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage 0 ;;
      --exclude-common-voice) exclude_cv=1 ;;
      --include-archives) exclude_archives=0 ;;
      --checksum) checksum=1 ;;
      --dry-run) dry=1 ;;
      -*) die "unknown flag $1 (see --help)" ;;
      *)
        if [ -z "$sub" ]; then sub=$1
        elif [ -z "$dest" ]; then dest=$1
        else die "unexpected argument $1"; fi ;;
    esac
    shift
  done
  [ -n "$sub" ] || usage 1
  [ -n "$dest" ] || die "$sub needs a destination directory"
  command -v "$RSYNC" >/dev/null || die "rsync not found (set RSYNC=)"
  command -v shasum >/dev/null || die "shasum not found"

  build_excludes
  case "$sub" in
    copy) cmd_copy "$dest" ;;
    manifest) cmd_manifest "$dest" ;;
    verify) cmd_verify "$dest" ;;
    diff) cmd_diff "$dest" ;;
    all) cmd_copy "$dest"; cmd_manifest "$dest"; cmd_verify "$dest" ;;
    *) die "unknown subcommand $sub (see --help)" ;;
  esac
}

main "$@"
