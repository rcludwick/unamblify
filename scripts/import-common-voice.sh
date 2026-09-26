#!/usr/bin/env bash
# Import a Common Voice archive downloaded by hand (Mozilla gates the
# download behind a web form) into the data root.
#
#   scripts/import-common-voice.sh ~/Downloads/<file>.tar.gz [more files...]
#
# Accepts every shape the Mozilla Data Collective ships:
#   commonvoice-v24_<lang>-<REGION>.tar.gz   (per-variant, e.g. en-AU, es-MX)
#   cv-corpus-<ver>-<date>-<locale>.tar.gz   (classic per-locale corpus)
#   <anything>.tar.gz                        (community sets, named freely)
# with or without the leading download timestamp the browser adds.
# The archive is MOVED into $UNAMBLIFY_DATA/archives/ (rename on the same
# volume, else copy+delete), its sha256 recorded under checksums/, and it is
# extracted into raw/common_voice/<variant>/. Idempotent: an archive with an
# .ok marker is skipped.
set -euo pipefail
DATA="${UNAMBLIFY_DATA:-/Volumes/data/training_data/unamblify}"
mkdir -p "$DATA"/{archives,raw/common_voice,checksums,logs}
LOG="$DATA/logs/import-common-voice-$(date +%Y%m%d-%H%M%S).log"
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
for src in "$@"; do
  [[ -f "$src" ]] || { log "no such file: $src"; exit 1; }
  base="$(basename "$src")"
  # Strip Mozilla's leading download timestamp: 1768991319509-commonvoice-v24_en-AU.tar.gz
  name="${base#[0-9]*-}"
  case "$name" in
    commonvoice-v*_*.tar.gz) variant="${name%.tar.gz}"; variant="${variant#commonvoice-v*_}" ;;
    cv-corpus-*.tar.gz)      variant="${name%.tar.gz}"; variant="${variant##*-}" ;;
    *.tar.gz)
      # Mozilla Data Collective community sets are named freely
      # (cv26-southern-american-english, zh-CN-beijing, ...): keep the name.
      variant="${name%.tar.gz}"; variant="$(printf '%s' "$variant" | tr -c 'A-Za-z0-9._-' '_')" ;;
    *) log "not a .tar.gz: $base"; exit 1 ;;
  esac
  dest="$DATA/archives/$name"
  if [[ -f "$dest.ok" ]]; then log "already imported: $name"; continue; fi
  if lsof -t "$src" >/dev/null 2>&1; then log "$base is still open by another process (download in progress?); skipping"; continue; fi
  log "verifying $base is a complete archive"
  tar -tzf "$src" >/dev/null 2>>"$LOG" || { log "archive is truncated or corrupt: $src"; exit 1; }
  log "moving $base -> $dest"
  mv "$src" "$dest" 2>/dev/null || { cp "$src" "$dest" && rm -f "$src"; }
  shasum -a 256 "$dest" | awk '{print $1}' > "$DATA/checksums/$name.sha256"
  log "sha256 $(cat "$DATA/checksums/$name.sha256")  $name"
  out="$DATA/raw/common_voice/$variant"
  mkdir -p "$out"
  log "extracting -> $out"
  tar -xzf "$dest" -C "$out"
  touch "$dest.ok" "$out/.extracted-$name"
  log "done: $variant ($(find "$out" -name '*.mp3' | wc -l | tr -d ' ') clips)"
done
