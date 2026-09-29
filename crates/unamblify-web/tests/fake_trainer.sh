#!/bin/bash
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
#
# A stand-in for `unamblify train` / `unamblify capture` so the supervisor
# can be tested without libtorch or a ThumbDV. It writes the run-directory
# files of spec §7 (status.json, metrics.jsonl, a checkpoint on SIGTERM)
# and the capture files of spec §3 (status.json, control.json polling).
#
#   fake_trainer.sh train --config C --run-dir D [--resume R]
#   fake_trainer.sh capture --mode M
#   fake_trainer.sh infer --run-dir D --step N --key K --out P [--data-root R]
#
# `infer` copies the sample's 8 kHz WAV to --out and writes a spec beside
# it, after FAKE_INFER_MS (default 0) milliseconds; when the run directory
# holds a `no-infer` file it fails the way clap does for a verb the binary
# was built without (a file, not an env var: tests run in parallel).
#
# The run is as long as its config says (`steps = N`, 1000 when it says
# nothing). From the config, not the environment: tests run in parallel
# threads of one process, where an environment variable set by one leaks
# into a child spawned by another — a run asked for 12 steps came back with
# the 3 a neighbouring test had set. FAKE_STEPS still overrides, for use by
# hand. FAKE_STEP_MS (default 50) is the step period. Like the real trainer it writes its own log.jsonl rows and
# echoes each to stderr unless UNAMBLIFY_LOG_ECHO=0.
set -u

verb=${1:-}
shift || true
steps=${FAKE_STEPS:-1000}
step_ms=${FAKE_STEP_MS:-50}
now_ms() { python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || date +%s000; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop=0
trap 'stop=1' TERM INT

case "$verb" in
train)
  dir=""; cfg=""; resume=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --config) cfg=$2; shift 2 ;;
      --run-dir) dir=$2; shift 2 ;;
      --resume) resume=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  [ -n "$dir" ] || { echo "no --run-dir" >&2; exit 2; }
  [ -f "$cfg" ] || { echo "config missing: $cfg" >&2; exit 2; }
  if [ -z "${FAKE_STEPS:-}" ]; then
    cfg_steps=$(sed -n 's/^steps *= *\([0-9][0-9_]*\).*/\1/p' "$cfg" | head -1 | tr -d _)
    [ -n "$cfg_steps" ] && steps=$cfg_steps
  fi
  if grep -q 'FAKE_FAIL' "$cfg"; then echo "fake trainer: failing on purpose" >&2; exit 3; fi
  step=0
  if [ -n "$resume" ]; then
    step=$(basename "$resume" | sed 's/^step-0*//'); step=${step:-0}
    echo "fake trainer: resuming from $resume at step $step"
  fi
  write_status() {
    # $1 state, $2 step, $3 pid-or-empty
    pidfield=""
    [ -n "${3:-}" ] && pidfield=",\"pid\":$3"
    printf '{"status":"%s","step":%d,"total_steps":%d,"started":"%s","updated":"%s"%s,"device":"cpu","host":"fake"}\n' \
      "$1" "$2" "$steps" "$started" "$(stamp)" "$pidfield" > "$dir/status.json.tmp"
    mv "$dir/status.json.tmp" "$dir/status.json"
  }
  # The trainer's own log rows: seq continues from the file's line count.
  logrow() {
    n=$(( $(wc -l < "$dir/log.jsonl" 2>/dev/null || echo 0) + 1 ))
    printf '{"seq":%d,"t":%s,"level":"%s","msg":"%s"}\n' "$n" "$(now_ms)" "$1" "$2" >> "$dir/log.jsonl"
    if [ "${UNAMBLIFY_LOG_ECHO:-1}" != "0" ]; then echo "[$1] $2" >&2; fi
  }
  started=$(stamp)
  write_status running "$step" $$
  logrow info "fake trainer: started pid $$ steps=$steps"
  echo "fake trainer: stdout line"
  printf '{"seq":900001,"t":%s,"level":"debug","msg":"a jsonl line from the trainer"}\n' "$(now_ms)"
  while [ "$stop" -eq 0 ] && [ "$step" -lt "$steps" ]; do
    step=$((step + 1))
    t=$(now_ms)
    printf '{"step":%d,"t":%s,"k":"loss/total","v":%s}\n' "$step" "$t" "1.$((10000 - step))" >> "$dir/metrics.jsonl"
    printf '{"step":%d,"t":%s,"k":"lr","v":0.0002}\n' "$step" "$t" >> "$dir/metrics.jsonl"
    printf '{"step":%d,"t":%s,"k":"sys/cpu","v":%d}\n' "$step" "$t" "$((step % 100))" >> "$dir/metrics.jsonl"
    write_status running "$step" $$
    sleep "$(printf '%d.%03d' $((step_ms / 1000)) $((step_ms % 1000)))"
  done
  if [ "$stop" -eq 1 ]; then
    ck=$(printf '%s/checkpoints/step-%06d' "$dir" "$step")
    mkdir -p "$ck/audio"
    : > "$ck/model.safetensors"
    : > "$ck/optim.safetensors"
    printf '{"step":%d}\n' "$step" > "$ck/meta.json"
    logrow info "fake trainer: SIGTERM at step $step, checkpoint written"
    write_status stopped "$step" ""
    exit 0
  fi
  logrow info "fake trainer: finished"
  echo "fake trainer: finished"
  write_status finished "$step" ""
  exit 0
  ;;
capture)
  mode=""
  while [ $# -gt 0 ]; do
    case "$1" in --mode) mode=$2; shift 2 ;; *) shift ;; esac
  done
  root=${UNAMBLIFY_DATA:?UNAMBLIFY_DATA must be set for the fake capture}
  dir="$root/captured/$mode"
  mkdir -p "$dir"
  done_n=0
  write() {
    printf '{"state":"%s","mode":"%s","port":"/dev/fake","done":%d,"failed":0,"total":100,"frames_s":42.0,"utt_per_hour":300,"eta_s":10,"current_key":"vctk/p225_%03d","started":"%s","updated":"%s","canary_ok":true,"prodid":"FAKE3000","version":"V0","pid":%d}\n' \
      "$1" "$mode" "$done_n" "$done_n" "$started" "$(stamp)" $$ > "$dir/status.json.tmp"
    mv "$dir/status.json.tmp" "$dir/status.json"
  }
  started=$(stamp)
  write running
  echo "fake capture: started $mode pid $$"
  while [ "$stop" -eq 0 ]; do
    ctl=$(cat "$dir/control.json" 2>/dev/null || echo '{"state":"run"}')
    case "$ctl" in
      *'"stop"'*) echo "fake capture: control stop"; write stopped; exit 0 ;;
      *'"pause"'*) write paused ;;
      *) done_n=$((done_n + 1)); write running ;;
    esac
    sleep 0.05
  done
  echo "fake capture: SIGTERM"
  write stopped
  exit 0
  ;;
infer)
  dir=""; step=""; key=""; out=""; root=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --run-dir) dir=$2; shift 2 ;;
      --step) step=$2; shift 2 ;;
      --key) key=$2; shift 2 ;;
      --out) out=$2; shift 2 ;;
      --data-root) root=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  if [ -f "$dir/no-infer" ]; then
    echo "error: unrecognized subcommand 'infer'" >&2
    exit 2
  fi
  [ -n "$out" ] && [ -n "$key" ] && [ -n "$step" ] || { echo "infer: bad args" >&2; exit 2; }
  [ -d "$dir/checkpoints/step-$(printf '%06d' "$step")" ] || { echo "infer: no checkpoint at step $step" >&2; exit 1; }
  src="$root/prepared/$key.8k.wav"
  [ -f "$src" ] || { echo "infer: $src missing" >&2; exit 1; }
  if [ "${FAKE_INFER_MS:-0}" -gt 0 ]; then sleep "$(printf '%d.%03d' $((FAKE_INFER_MS / 1000)) $((FAKE_INFER_MS % 1000)))"; fi
  mkdir -p "$(dirname "$out")"
  cp "$src" "$out"
  printf '{"rate":16000,"n_fft":1024,"hop":256,"n_mels":2,"db_min":-100,"db_max":0,"frames":2,"out":[[-1,-2],[-3,-4]]}\n' > "${out%.wav}.spec.json"
  echo "fake infer: $key step $step -> $out"
  exit 0
  ;;
*)
  echo "fake trainer: unknown verb '$verb'" >&2
  exit 2
  ;;
esac
