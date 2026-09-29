#!/usr/bin/env bash
# Download and verify the voice corpora listed in data/sources.tsv.
#
# Usage:
#   scripts/fetch-datasets.sh [--tier N] [--only NAME] [--no-extract] [--dry-run]
#
# Environment:
#   UNAMBLIFY_DATA   destination root (default: /Volumes/data/training_data/unamblify)
#
# Layout under $UNAMBLIFY_DATA:
#   archives/    downloaded archives, resumable (curl -C -)
#   raw/<name>/  extracted corpora, untouched
#   checksums/   sha256 of every archive as observed on first fetch
#   logs/        one log per run
#
# Idempotent: re-running verifies checksums and skips completed work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/data/sources.tsv"
DATA="${UNAMBLIFY_DATA:-/Volumes/data/training_data/unamblify}"
MAX_TIER=99
ONLY=""
EXTRACT=1
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) MAX_TIER="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --no-extract) EXTRACT=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for tool in curl tar unzip shasum; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
if command -v md5sum >/dev/null; then MD5="md5sum"; else MD5="md5 -r"; fi

mkdir -p "$DATA"/{archives,raw,checksums,logs}
LOG="$DATA/logs/fetch-$(date +%Y%m%d-%H%M%S).log"
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
md5_of()    { $MD5 "$1" | awk '{print $1}'; }

verify() { # file expected(kind:hex or -) name
  local f="$1" want="$2" rec="$DATA/checksums/$(basename "$1").sha256"
  case "$want" in
    md5:*)
      local got; got="$(md5_of "$f")"
      [[ "$got" == "${want#md5:}" ]] || { log "MD5 MISMATCH $f got=$got want=${want#md5:}"; return 1; }
      ;;
    sha256:*)
      local got; got="$(sha256_of "$f")"
      [[ "$got" == "${want#sha256:}" ]] || { log "SHA256 MISMATCH $f got=$got want=${want#sha256:}"; return 1; }
      ;;
    -)
      local got; got="$(sha256_of "$f")"
      if [[ -f "$rec" ]]; then
        [[ "$got" == "$(cat "$rec")" ]] || { log "SHA256 DRIFT $f got=$got recorded=$(cat "$rec")"; return 1; }
      else
        echo "$got" > "$rec"; log "recorded sha256 $got  $(basename "$f")  (no upstream checksum; pin this in data/sources.tsv)"
      fi
      ;;
  esac
  # Always keep an observed sha256 on disk for cross-machine comparison.
  [[ -f "$rec" ]] || sha256_of "$f" > "$rec"
  return 0
}

extract() { # archive dest
  local a="$1" d="$2" stamp="$2/.extracted-$(basename "$1")"
  [[ -f "$stamp" ]] && { log "already extracted $(basename "$a")"; return 0; }
  mkdir -p "$d"
  log "extracting $(basename "$a") -> $d"
  case "$a" in
    *.tar.gz|*.tgz) tar -xzf "$a" -C "$d" ;;
    *.tar.bz2)      tar -xjf "$a" -C "$d" ;;
    *.zip)          unzip -q -o "$a" -d "$d" ;;
    *) log "don't know how to extract $a"; return 1 ;;
  esac
  touch "$stamp"
}

log "manifest=$MANIFEST data=$DATA max_tier=$MAX_TIER only=${ONLY:-*}"
fail=0
while IFS=$'\t' read -r name tier url size sum dir; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  (( tier <= MAX_TIER )) || continue
  [[ -z "$ONLY" || "$ONLY" == "$name" ]] || continue
  file="$DATA/archives/$(basename "$url")"
  if (( DRY )); then log "would fetch $url -> $file ($size bytes)"; continue; fi
  have=0; [[ -f "$file" ]] && have=$(stat -f %z "$file" 2>/dev/null || stat -c %s "$file")
  if [[ -f "$file.ok" ]] && { (( size == 0 )) || (( have == size )); }; then
    log "ok $(basename "$file")"
  else
    log "fetching $url ($have/$size bytes present)"
    curl -L --fail --retry 10 --retry-delay 15 --retry-all-errors -C - -o "$file" "$url" 2>>"$LOG" || { log "download failed: $url"; fail=1; continue; }
    if verify "$file" "$sum"; then touch "$file.ok"; log "verified $(basename "$file")"; else fail=1; continue; fi
  fi
  (( EXTRACT )) && { extract "$file" "$DATA/raw/$dir" || fail=1; }
done < "$MANIFEST"

if (( fail )); then log "FINISHED WITH ERRORS (see $LOG)"; exit 1; fi
log "done"
