#!/usr/bin/env bash
# backup-chunks.sh — append-only chunked backup of the chip captures.
#
# Capture only ever appends: a written utterance is never modified again.
# So the backup never compares the two sides. Each run packs the
# utterances that are not in the local ledger yet into one immutable
# numbered tar, hashes every file in it, and uploads that single object.
# Nothing already uploaded is re-read, re-hashed or re-sent — which is
# what makes this cheap against Google Drive, where creating many small
# files is the slow part and the remote will not checksum for you.
#
# Each chunk carries a sidecar listing the SHA-256 of every file inside
# it plus the hash of the tar itself, so integrity is checkable at two
# levels: against the MD5 Drive computed at upload (no download), or by
# pulling a chunk back and verifying every file (`verify --deep`).
# Concatenating the sidecars gives a per-file manifest of the whole
# backup without storing it twice.
#
# Chunks are plain tar, not tar.gz: FLAC and the .ambe bitstreams are
# already compressed (measured: gzip -1 saves 3 %), so compressing only
# costs time and widens the blast radius of a corrupt byte.
#
# Subcommands:
#   pack    [--mode M] [--chunk-size BYTES] [--max-chunks N]
#   push    [--mode M]            rclone copy new chunks + the manifest
#                                 and canary.json
#   verify  [--mode M]            cheap: local hash vs Drive's stored MD5
#   verify  --deep CHUNK          pull one chunk back, check every file
#   status  [--mode M]
#   restore --mode M              the other direction: pull every chunk of
#                                 a capture set, check it at both levels,
#                                 and unpack it under captured/M/
#
# Env:
#   UNAMBLIFY_DATA  data root (default /Volumes/data/training_data/unamblify)
#   RCLONE_REMOTE   rclone target (default gdrive:unamblify-corpus)
#   CHUNK_BYTES     chunk size target (default 5 GiB)
#   QUIET_SECS      skip files touched this recently, so an utterance
#                   still being written is never packed (default 60)
#
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

SRC="${UNAMBLIFY_DATA:-/Volumes/data/training_data/unamblify}"
REMOTE="${RCLONE_REMOTE:-gdrive:unamblify-corpus}"
CHUNK_BYTES="${CHUNK_BYTES:-5368709120}"
QUIET_SECS="${QUIET_SECS:-60}"
BACKUP="$SRC/backup"
MODES="dstar ysf-dmr"

die() { printf 'backup-chunks: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

# BSD stat on macOS, GNU stat elsewhere.
fsize() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
fmtime() { stat -f%m "$1" 2>/dev/null || stat -c%Y "$1"; }

# Keys of a capture manifest, one per line.
manifest_keys() {
  python3 -c '
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        try: print(json.loads(line)["key"])
        except Exception: pass
' "$1"
}

cmd_pack() {
  local mode=$1 chunk_size=$2 max_chunks=$3
  local dir="$SRC/captured/$mode"
  [ -d "$dir" ] || die "no capture set at $dir"
  local mdir="$BACKUP/$mode" cdir="$BACKUP/$mode/chunks"
  mkdir -p "$cdir"
  local ledger="$mdir/packed.keys"
  [ -f "$ledger" ] || : > "$ledger"

  local tmp="$TMP/pack-$mode"; rm -rf "$tmp"; mkdir -p "$tmp"
  manifest_keys "$dir/manifest.jsonl" | LC_ALL=C sort -u > "$tmp/all"
  LC_ALL=C sort -u "$ledger" > "$tmp/done"
  LC_ALL=C comm -23 "$tmp/all" "$tmp/done" > "$tmp/todo"
  local todo; todo=$(wc -l < "$tmp/todo" | tr -d ' ')
  say "$mode: $todo utterance(s) not yet packed"
  [ "$todo" -gt 0 ] || return 0

  local cutoff; cutoff=$(( $(date +%s) - QUIET_SECS ))
  local made=0
  while :; do
    [ "$max_chunks" -eq 0 ] || [ "$made" -lt "$max_chunks" ] || break
    local last; last=$(ls "$cdir" 2>/dev/null \
      | sed -n "s/^$mode-\([0-9][0-9]*\)\.tar$/\1/p" | sort -n | tail -1)
    local n; n=$(printf '%04d' $(( 10#${last:-0} + 1 )))   # 10# or 0008 parses as octal
    local base="$mode-$n" list="$tmp/list" keys="$tmp/keys"
    : > "$list"; : > "$keys"
    local acc=0 packed=0
    while IFS= read -r key; do
      [ -n "$key" ] || continue
      local any=0 f try_bytes=0
      for ext in .ambe .flac .wav; do
        f="$dir/$key$ext"
        [ -f "$f" ] || continue
        # Still being written? leave it for the next run.
        [ "$(fmtime "$f")" -lt "$cutoff" ] || { any=0; break; }
        printf '%s\n' "$key$ext" >> "$list.try"
        try_bytes=$(( try_bytes + $(fsize "$f") ))
        any=1
      done
      if [ "$any" -eq 1 ] && [ -s "$list.try" ]; then
        cat "$list.try" >> "$list"; printf '%s\n' "$key" >> "$keys"
        packed=$((packed+1)); acc=$(( acc + try_bytes ))
      fi
      rm -f "$list.try"
      [ "$acc" -lt "$chunk_size" ] || break
    done < "$tmp/todo"

    [ "$packed" -gt 0 ] || { say "$mode: nothing ready to pack (all too recent)"; break; }

    say "$mode: packing $base — $packed utterance(s), $(awk -v b=$acc 'BEGIN{printf "%.2f GiB", b/1073741824}')"
    tar -C "$dir" -cf "$cdir/$base.tar" -T "$list"
    ( cd "$dir" && tr '\n' '\0' < "$list" | xargs -0 -n 200 shasum -a 256 ) > "$cdir/$base.sha256"
    ( cd "$cdir" && shasum -a 256 "$base.tar" ) > "$cdir/$base.tar.sha256"
    cp "$keys" "$cdir/$base.keys"
    cat "$keys" >> "$ledger"
    LC_ALL=C sort -u "$ledger" -o "$ledger"
    made=$((made+1))

    # Remaining work for the next chunk.
    LC_ALL=C sort -u "$ledger" > "$tmp/done"
    LC_ALL=C comm -23 "$tmp/all" "$tmp/done" > "$tmp/todo2" && mv "$tmp/todo2" "$tmp/todo"
    [ -s "$tmp/todo" ] || break
  done
  say "$mode: $made chunk(s) written to $cdir"
}

cmd_push() {
  local mode=$1
  command -v rclone >/dev/null || die "rclone not found"
  rclone listremotes 2>/dev/null | grep -q "^${REMOTE%%:*}:" \
    || die "rclone remote '${REMOTE%%:*}' is not configured; run: rclone config"
  local cdir="$BACKUP/$mode/chunks"
  [ -d "$cdir" ] || die "nothing packed for $mode yet"
  say "$mode: uploading new chunks to $REMOTE/captured/$mode/"
  rclone copy "$cdir" "$REMOTE/captured/$mode/" --transfers 4 --stats 30s --stats-one-line
  # The live manifest grows while capture runs, and rclone refuses to
  # upload a file whose size changes mid-transfer. Send a snapshot.
  local snap="$TMP/manifest-$mode.jsonl"
  cp "$SRC/captured/$mode/manifest.jsonl" "$snap"
  rclone copyto "$snap" "$REMOTE/captured/$mode/manifest.jsonl" --stats 30s --stats-one-line
  # The canary record, so a restored set can check the chip it meets next
  # against the one that made it. A software set has none.
  if [ -f "$SRC/captured/$mode/canary.json" ]; then
    rclone copyto "$SRC/captured/$mode/canary.json" "$REMOTE/captured/$mode/canary.json"
  fi
}

cmd_verify() {
  local mode=$1
  command -v rclone >/dev/null || die "rclone not found"
  say "$mode: comparing local hashes against Drive's stored checksums (no download)"
  rclone check "$BACKUP/$mode/chunks" "$REMOTE/captured/$mode/" --one-way
  say "$mode: OK"
}

cmd_verify_deep() {
  local chunk=$1
  local mode="${chunk%-*}"   # ysf-dmr-0001 -> ysf-dmr, dstar-0001 -> dstar
  local cdir="$BACKUP/$mode/chunks"
  [ -f "$cdir/$chunk.sha256" ] || die "no sidecar for $chunk"
  local tmp="$TMP/deep"; rm -rf "$tmp"; mkdir -p "$tmp"
  say "deep verify $chunk: pulling from $REMOTE"
  rclone copy "$REMOTE/captured/$mode/$chunk.tar" "$tmp/" --stats 30s --stats-one-line
  say "checking the tar's own hash"
  ( cd "$tmp" && shasum -a 256 -c "$cdir/$chunk.tar.sha256" ) >/dev/null \
    || die "deep verify $chunk: the tar itself does not match"
  say "untarring and checking every file"
  mkdir -p "$tmp/x"; tar -C "$tmp/x" -xf "$tmp/$chunk.tar"
  ( cd "$tmp/x" && shasum -a 256 -c "$cdir/$chunk.sha256" ) >/dev/null \
    && say "deep verify $chunk: every file matches" \
    || die "deep verify $chunk: MISMATCH"
}

# The other direction, for someone handed access to the backup: pull each
# chunk, check the tar against its own hash, unpack it into the capture
# set, check every file against the sidecar, and drop the tar. One chunk
# at a time, so it needs a chunk's worth of scratch space and no more. A
# marker per chunk makes it resumable. The manifest is fetched only when
# there is none: a live capture's manifest is never overwritten.
cmd_restore() {
  local mode=$1
  command -v rclone >/dev/null || die "rclone not found"
  local dir="$SRC/captured/$mode" marks="$SRC/captured/$mode/.restored"
  local tmp="$TMP/restore"
  mkdir -p "$dir" "$marks" "$tmp"
  local chunks
  chunks=$(rclone lsf "$REMOTE/captured/$mode/" --include '*.tar' | LC_ALL=C sort) \
    || die "cannot list $REMOTE/captured/$mode/"
  [ -n "$chunks" ] || die "no chunks at $REMOTE/captured/$mode/"
  local c base n=0
  for c in $chunks; do
    base="${c%.tar}"
    if [ -f "$marks/$base" ]; then say "$base: already restored"; continue; fi
    say "$base: pulling"
    rclone copy "$REMOTE/captured/$mode/" "$tmp/" \
      --include "$base.tar" --include "$base.tar.sha256" --include "$base.sha256" \
      --stats 30s --stats-one-line
    ( cd "$tmp" && shasum -a 256 -c "$base.tar.sha256" ) >/dev/null \
      || die "$base: the tar does not match its hash; nothing unpacked"
    tar -C "$dir" -xf "$tmp/$base.tar"
    ( cd "$dir" && shasum -a 256 -c "$tmp/$base.sha256" ) >/dev/null \
      || die "$base: a file does not match the sidecar after unpacking"
    : > "$marks/$base"
    rm -f "$tmp/$base.tar" "$tmp/$base.tar.sha256" "$tmp/$base.sha256"
    n=$((n+1))
  done
  if [ -f "$dir/manifest.jsonl" ]; then
    say "$mode: keeping the manifest already at $dir/manifest.jsonl"
  else
    rclone copyto "$REMOTE/captured/$mode/manifest.jsonl" "$dir/manifest.jsonl"
  fi
  if [ ! -f "$dir/canary.json" ] \
    && [ -n "$(rclone lsf "$REMOTE/captured/$mode/" --include canary.json)" ]; then
    rclone copyto "$REMOTE/captured/$mode/canary.json" "$dir/canary.json"
  fi
  say "$mode: $n chunk(s) restored into $dir"
}

cmd_status() {
  local mode=$1
  local cdir="$BACKUP/$mode/chunks" dir="$SRC/captured/$mode"
  local packed=0 chunks=0
  [ -f "$BACKUP/$mode/packed.keys" ] && packed=$(wc -l < "$BACKUP/$mode/packed.keys" | tr -d ' ')
  [ -d "$cdir" ] && chunks=$(ls "$cdir" 2>/dev/null | grep -c '\.tar$' || true)
  local total=0
  [ -f "$dir/manifest.jsonl" ] && total=$(wc -l < "$dir/manifest.jsonl" | tr -d ' ')
  printf '%-9s captured %-8s packed %-8s chunks %-4s\n' "$mode" "$total" "$packed" "$chunks"
}

main() {
  local sub="" mode="" chunk_size="$CHUNK_BYTES" max_chunks=0 deep=""
  [ $# -gt 0 ] || usage 1
  sub=$1; shift
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage 0 ;;
      --mode) mode=$2; shift ;;
      --chunk-size) chunk_size=$2; shift ;;
      --max-chunks) max_chunks=$2; shift ;;
      --deep) deep=$2; shift ;;
      *) die "unexpected argument $1 (see --help)" ;;
    esac
    shift
  done
  [ -d "$SRC" ] || die "data root $SRC does not exist"
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  local list="$MODES"; [ -z "$mode" ] || list="$mode"
  case "$sub" in
    pack)   for m in $list; do cmd_pack "$m" "$chunk_size" "$max_chunks"; done ;;
    push)   for m in $list; do cmd_push "$m"; done ;;
    verify) if [ -n "$deep" ]; then cmd_verify_deep "$deep"; else for m in $list; do cmd_verify "$m"; done; fi ;;
    status) for m in $list; do cmd_status "$m"; done ;;
    restore) [ -n "$mode" ] || die "restore needs --mode"; cmd_restore "$mode" ;;
    -h|--help) usage 0 ;;
    *) die "unknown subcommand $sub (see --help)" ;;
  esac
}

main "$@"
