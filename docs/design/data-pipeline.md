# Data pipeline

!!! info "Status"
    2026-09-10: built. `prepare`, `capture`, `shard` and `verify` live in
    `crates/unamblify-data` behind the `unamblify` binary. The formats
    below are what the code writes. 2026-09-11: the Codec 2 (M17) modes
    are captured in software through the same harness
    ([software capture](#software-capture)), and D-STAR capture from the
    real stick is under way. Later the same day the augmentation of
    [stage 3](#stage-3-augment) was built: prepare-time twins, the
    decode-only `augment` stage, and receive-side noise in the loaders.
    2026-09-25: three sticks capture both AMBE modes, and `recode` and the
    per-set lags have been added since.

Everything lives under the data root, `$UNAMBLIFY_DATA` (default
`/Volumes/data/training_data/unamblify`). `--data-root` overrides it on
every verb. Every stage is idempotent and resumable. It reads a manifest,
skips work that has a done-marker, and can be stopped and restarted at
any time.

```
raw/<corpus>/...                        untouched corpora (fetch)
prepared/manifest.jsonl                 one UtteranceRow per utterance
prepared/rejected.jsonl                 utterances prepare refused, and why
prepared/<corpus>/<utt_key>.16k.flac    s16 mono 16 kHz, normalised, trimmed, lossless FLAC
prepared/<corpus>/<utt_key>.8k.flac     the same, decimated to 8 kHz
prepared/<corpus>/<utt_key>+n0173.16k.flac an augmented twin (stage 3): the noisy /
                                        overdriven INPUT; its row names its parent
captured/<mode>/manifest.jsonl          one CaptureRow per finished utterance
captured/<mode>/failed.jsonl            utterances that failed three attempts
captured/<mode>/<corpus>/<utt_key>.ambe raw channel frames, no header (every mode: the
                                        extension means "channel frames", the mode says the codec)
captured/<mode>/<corpus>/<utt_key>.flac decoded 8 kHz s16, lossless FLAC
captured/<mode>/canary.json             day-one canary frames + codec id/version + lag
captured/<mode>/control.json            {"state":"run"|"pause"|"stop"}
captured/<mode>/status.json             live progress, rewritten every 5 s
captured/<mode>/lock                    flock held by the running harness (pid inside)
captured/<mode>/child.log               stdout/stderr of a dashboard-spawned harness
captured/<mode>+<kind>/...              a sibling capture set: either a decode-only
                                        augment (stage 3, `augment` - same keys as the
                                        base capture, rows carrying `aug`) or a recode
                                        (stage 3b, `recode` - the prepared audio encoded
                                        again by another implementation of the codec)
canary/1khz-and-speech.8k.wav           the canary clip (synthesised once)
shards/<name>/index.json + NNNN.bin     fixed-length training examples
runs/<run_id>/                          training runs (see training.md)
logs/                                   fetch logs
```

`<utt_key>` is corpus-relative and filesystem-safe:
`libritts_r/<subset>/<reader>_<chapter>_<para>_<sent>`,
`voicebank_demand/<set>/<spk>_<utt>`, `vctk/<spk>_<utt>_mic<N>`,
`ljspeech/LJ<book>-<utt>`. The path on disk is `prepared/` + key +
suffix. An augmented twin's key is its parent's key plus `+n<NNNN>`
and/or `+h<NNNN>`, and `unamblify::key::parent_of` strips them. `+` is
part of the key alphabet.

## Stage 0: fetch

Done: [data sources](../research/data-sources.md). Archives in `archives/`,
extracted corpora in `raw/<corpus>/`.

## Stage 1: prepare (`unamblify prepare [--corpus C]... [--force] [--jobs N] [--noise-share S --noise-sets demand,musan] [--ham-chain-share S] [--seed N]`) { #stage-1-prepare }

Walk `raw/`, emit one row per utterance into `prepared/manifest.jsonl`, and
write two aligned files per utterance:

| File | Content | Purpose |
|---|---|---|
| `prepared/<corpus>/<utt>.16k.flac` | 16 kHz s16 mono, loudness-normalised, silence-trimmed | the target (and, decimated, the narrowband target) |
| `prepared/<corpus>/<utt>.8k.flac` | the same signal anti-alias decimated to 8 kHz | the chip input |

### UtteranceRow

```json
{"key":"vctk/p225_001_mic2","corpus":"vctk","speaker":"p225","gender":"F",
 "split":"train","duration_s":3.12,"src_rate":48000,
 "src_path":"raw/vctk/wav48_silence_trimmed/p225/p225_001_mic2.flac",
 "licence":"CC-BY-4.0","rms_dbfs_in":-21.3,"gain_db":-4.7,
 "trim_lead_s":0.11,"trim_tail_s":0.20,"sha256_16k":"…","sha256_8k":"…",
 "prepared_at":"2026-09-10T03:00:00Z"}
```

An augmented twin ([stage 3](#stage-3-augment)) is a row like any other
with two more fields. `parent` is the key it was made from, whose clean
files are the twin's training target (`UtteranceRow::target_key`). `aug`
is the core crate's `AugRecord`:

```json
{"key":"vctk/p225_001_mic2+n0173+h0042","parent":"vctk/p225_001_mic2", …,
 "gain_db":-1.9,
 "aug":{"seed":1234,
        "noise":{"noise_set":"demand","noise_clip":"raw/demand/TCAR/ch03.wav",
                 "noise_offset_s":112.4,"snr_db":7.3},
        "chain":{"shelf_db":6.2,"mic":{"tilt_db":-2.1,"resonance_hz":1420.0,
                 "resonance_db":5.5,"resonance_q":3.1},
                 "pops":{"share":0.35,"freq_hz":88.0,"level_dbfs":-6.0},
                 "clip":{"kind":"hard","drive_db":14.0},
                 "agc":{"attack_ms":12.0,"release_ms":300.0}},
        "post_gain_db":-1.9}}
```

The twin's `corpus`, `speaker`, `split`, `duration_s`, `src_*` and
trims are the parent's. `gain_db` is the gain applied after the
augmentation (see stage 3), and `rms_dbfs_in` is the augmented input's
own active level before that gain.

**Split rule** (`unamblify::split`). The split is deterministic and
speaker-disjoint. VCTK and VoiceBank-DEMAND share one speaker namespace.

- `test`: VoiceBank's p232 and p257 in every corpus that has them, plus
  the LibriTTS-R `test-*` readers whose `sha1(reader)` first byte is
  below `0x14` (260, 5639).
- `dev`: the LibriTTS-R `dev-*` readers below the same limit (422, 6345,
  7850, 7976), plus the six readers the eval clips come from
  (`LIBRITTS_EVAL_READERS`, whatever their hash), plus the
  VCTK-namespace speakers whose `sha1(speaker)` first byte is below
  `0x14` (≈ 8 %: p228 p239 p244 p266 p278 p282 p284 p295 p298 p300 p323
  p334 p341).
- `train`: everything else. LJSpeech is `train` only.

The split is recorded per row so nothing downstream recomputes it.

Until 2026-09-13 every LibriTTS-R `dev-*` and `test-*` reader was held
out. Those subsets were tier 0, and holding them out left 58 % of the
captured AMBE hours in splits no run trains on. `unamblify prepare
--resplit` re-applies the rule to `prepared/manifest.jsonl` in place and
reports what moved (8 228 utterances to `train` on that date). No audio
is touched, and the captured manifests carry no split of their own.
Rebuild any shard set afterwards.

Rules, as implemented:

- Decode WAV/FLAC to f32 and resample to 16 kHz with rubato
  `SincFixedIn` (anti-aliased), never with linear interpolation.
- Trim leading and trailing silence below −45 dBFS, keeping a 200 ms
  margin. The vocoder's onset behaviour is part of what the model must
  learn.
- Normalise active-speech RMS to −26 dBFS, with the gain clamped to
  ±20 dB. Reject the utterance if the result still clips.
- Drop utterances shorter than 1.0 s. Rejections go to
  `prepared/rejected.jsonl` (key, reason) so a re-run does not decode
  them again. `--force` clears them.
- VCTK: prefer `mic2` and fall back to `mic1`, one row per utterance.
  LibriTTS-R: any file whose stem is not `<reader>_<chapter>_<para>_<sent>`
  (the doc files) is skipped.
- Idempotent: keys already in the manifest whose files exist with the
  expected sizes are skipped. A damaged file is redone and the key never
  appears twice. Work runs in parallel over utterances (rayon, `--jobs`)
  with one writer thread and an fsync every 1000 rows.

## Stage 2: capture (`unamblify capture --mode <dstar|ysf-dmr|codec2-3200|codec2-1600> [--port P]... [--jobs N] [--order random|balanced|design] [--order-seed 1] [--corpus C] [--split S] [--limit N] [--canary-every 200] [--dry-run] [--warmup-mix] [--cold-share 0.34] [--warmup-seed 1]`) { #stage-2-capture }

`<mode>` is a `VocoderMode`, and its `family()` decides who does the
encoding. The AMBE modes go through the ThumbDV as below. They are
`dstar` and `ysf-dmr`, which is one capture for YSF DN and DMR because
both carry the same 49 voice bits (see the
[design index](index.md#decisions-already-made)). The Codec 2 modes
(`codec2-3200` and `codec2-1600`, what M17 carries) run in software.
[Software capture](#software-capture) lists exactly what differs, and
everything else on this page applies to both. Capture sees only the
`vocoder::Vocoder` trait (`encode` a whole padded utterance, `decode` it
back, `info`, `reset`, `warm_up`). `ChipVocoder` wraps the pipelined
runner and `Codec2Vocoder` wraps the crate.

For every prepared utterance and requested AMBE mode, one worker thread per
stick does the following:

1. **Port selection.** The FTDI `0403:6015` / "ThumbDV" scan is
   intersected with any `--port` values, exactly as in astar's
   `thumbdv_candidate_ports_from`. A `--port` outside the scan is refused
   with an error naming the rule, because opening a radio interface's tty
   asserts RTS and keys a transmitter. A raw `O_NONBLOCK|O_NOCTTY` open
   first classifies the port as *busy* (naming the holder via `lsof`) or
   *absent*.
2. **Init per mode** (steps 1–9 of the ThumbDV reference): drain 50 ms,
   then `reset` → Ready, a check that `prodid` starts with `AMBE3000`,
   `verstring`, the mode's `RATEP` word, `init_encdec`, `ecmode_off`,
   `dcmode_off` and `gain_zero`. Each is a 300 ms transact with a status
   check. The D-STAR word comes from the vendored driver. The YSF DN word
   for `ysf-dmr` is re-implemented and pinned to the reference bytes by
   tests, as is the DMR word that only the future `ber` framing uses.
3. **Warm-up.** 20 canary frames go through encode and are discarded.
   The chip's first ~10 frames after init are rubbish (AMBE+2 pitch
   lock).

   With `--warmup-mix` (AMBE modes only), each utterance is instead
   captured under a warm-up state chosen from a hash of its key. A
   `cold` share (`--cold-share`, default 0.34) gets a chip reset right
   before the kept encode. Their first frames then carry the post-keyup
   pitch-lock transient that happens every time someone presses the
   button. The rest are `warm`, with the encoder locked onto the voice
   first. The row records which in `warm_state`. The mix costs a little
   chip time and gives training coverage of both states. `--warmup-seed`
   makes it reproducible.
4. **Canary.** On day one the canary clip (0.5 s of 1 kHz + 1.5 s of
   synthetic voice at 8 kHz) is encoded and its frames hashed. It is then
   decoded, and the lag between decoded output and input is measured by
   cross-correlation (±400 samples). All of this goes to `canary.json`.
   Every `--canary-every` utterances, and after every re-init, the clip
   is re-encoded and the hash compared. A mismatch stops the run loudly
   and leaves `canary.json` alone. The encoder carries state across
   frames (pitch tracking, the post-init rubbish), so every re-encode
   starts from the same state as the record: `reset` → the 20-frame
   warm-up → the clip. The periodic check therefore re-inits the chip
   first, and `canary.json` records `warm_up_frames`. A harness with a
   different preamble refuses the record rather than failing a healthy
   run.
5. **Both directions at once**, interleaved
   (`pipeline::round_trip`). The 8 kHz input is zero-padded to whole
   frames of `mode.frame_samples()`. `speech_in` packets are written with
   at most `INTERLEAVE_DEPTH` = 3 encodes outstanding. Every
   `Channel` reply is stored exactly as the chip emitted it
   (`mode.frame_bytes()` = `bits.div_ceil(8)` bytes per frame, DN frames
   in chip wire order). It is handed straight back as a
   `channel_in_bits(bits, data)` request, with at most 3 decodes
   outstanding. Nothing waits for the encode pass to finish because the
   UART is full duplex. Overlapping the two directions costs 8.27 ms
   per audio frame against 14.28 ms for the two passes back to back
   (1.73×) ([hardware throughput](../research/hardware-throughput.md)).
6. Responses are routed by packet type, never by arrival order.
   `Channel` answers the oldest outstanding encode and `Speech` the
   oldest outstanding decode, FIFO within each kind. astar's `AmbeStream`
   worker does the same with its two pipeline sides. Each direction has
   its own FIFO, its own 100 ms age-out and its own count.
   `Channel{bits == mode.channel_bits()}` is still enforced, frames out
   must still equal frames in, in both directions, and any failure is
   the whole utterance's failure.

    Depth 3 + 3 is a hard ceiling, not a tunable. 2 + 2 costs
    8.59 ms. Every asymmetric split is worse (3 + 2 is 10.85 ms, 2 + 3
    is 10.96, 4 + 2 is 11.31, 2 + 4 is 12.71), and 4 + 4 stalls the
    chip. Its input queue overflows, a response never arrives and the
    utterance ages out. The total in flight therefore never exceeds six
    packets.

    `capture --sequential` is the escape hatch for a stick that
    misbehaves with both directions in flight. It runs the old encode
    pass followed by the old decode pass, each four deep
    (`pipeline::run`, `MAX_IN_FLIGHT`), and produces byte-identical
    output in 1.73× the wall time. The single-direction runner is also
    what the canary (encode only), the warm-up and the decode-only
    `augment` stage use. The software Codec 2 path never comes near
    either runner.
7. A `CaptureRow` is appended (fsync per row) once both files are on disk.

**Failure policy.** Any of these discards the partial output: a timeout
(100 ms since the oldest outstanding packet), the rate being lost, a
parse error, an unexpected packet or a count mismatch. The harness then
runs `reset`, a full re-init, the warm-up and the canary re-check, and
redoes the utterance. After three attempts the utterance goes to
`failed.jsonl` and the run continues. That file is an audit log, not a
skip list: the next run retries it. Never splice.

**Resume and order.** Keys already in `manifest.jsonl` with both files on
disk are skipped. A row whose files have gone is dropped from the
manifest before its utterance is captured again, so a key never appears
twice. `--order` sequences what remains:

- `random` (the default since 2026-09-13): every utterance sits at a
  position given by `sha256(seed ‖ key)`, whatever its corpus. At any
  moment the captured set is a uniform random sample of everything
  prepared, so a model trained on a partial capture leans on no one
  dataset. A key's position never changes, so a corpus prepared later
  drops its utterances into the remaining order. Nothing restarts and
  nothing already captured is wasted. `--order-seed` picks the
  permutation.
- `balanced`: round-robin over corpora, each in its own hashed order.
  This gives equal utterance counts per corpus until a small one runs
  out, for when the small corpora matter more than their hours.
- `design`: corpus by corpus, voicebank_demand → libritts_r dev/test →
  vctk → ljspeech → libritts_r train, or the `--corpus` order given. The
  first captures were run this way. The chip spent its first weeks
  inside two corpora, which is why the D-STAR set is 30 hours of the
  same 128 speakers.

`--corpus` filters under every order and ranks only under `design`.
Work is then assigned by speaker (all of a speaker's utterances on one
stick), and each stick keeps the order. A running harness holds
`captured/<mode>/lock`, an `flock` that the kernel releases if the
harness dies. A second harness on the same mode is refused, as it is when
`status.json` names a live pid. A torn final line in any manifest (the
one thing an interrupted append leaves) is dropped on read and truncated
before the next append. A kill mid-write therefore never blocks a stage.

**Control.** Between utterances (never mid-utterance) the worker reads
`control.json`. `pause` finishes the utterance in flight, keeps the port
open and polls every 500 ms. `run` continues. `stop` exits 0 after the
utterance. SIGINT is `stop`. Opening the chip is retried up to four
times, 500 ms apart, before a worker gives up. The AMBE-3000
occasionally misses its reset for a second or two, and a single timeout
at open should not end a capture that runs for weeks (a mid-run reset is
already retried the same way). A stop takes as long as the utterance in
flight (up to ~10 s for a long LibriTTS-R clip). The dashboard shows
*stopping* from the click until the process is gone, with no controls.

Two teardown costs are kept off the path that writes the final status
and releases the lock. The worker leaks its vocoder, so the FTDI serial
`close()` happens at process exit instead. That call blocks: macOS
drains TX on a blocking fd, and a degraded stick stalls the USB close
for tens of seconds. The plan is also left for the OS to reclaim rather
than freeing ~180 k possibly paged-out rows. Both costs were holding a
stopped harness, and so the dashboard's *running* state, for 40–60 s.

A `stop` or `pause` left in the file by an earlier run is reset to `run`
(and logged) when a harness starts, so the SIGINT flag is the only stop
source at start. `status.json` is rewritten at least every 5 s with
`{state, mode, pid, ports, done, failed, total,
frames_s, utt_per_hour, eta_s, current_key, current_keys, started,
updated, canary_ok, prodid, version, error}`. The dashboard keys
liveness, adoption and its double-start guard on `pid`. `paused` means
every worker that still has work is holding on the file (an idle stick
does not keep the state at `running`). `unamblify capture
pause|resume|stop --mode M` writes the control file, `unamblify capture
status --mode M [--json]` prints the status, and the dashboard's Capture
page does both.

`--dry-run` runs everything except the serial port against a
deterministic simulated chip (`sim::SimTransport`, lag 42 samples, with
fault injection for the tests). The sim models what the interleaved
runner depends on. Replies are FIFO within a direction, but the two
directions' streams are ordered against each other arbitrarily
(`ReplyBias`). The sim also counts what each direction owes, so a test
can arm `with_max_in_flight(3)` and the sim panics the moment a runner
asks the chip to hold more than the real one tolerates. No test in the
workspace opens a serial port. `--stats` prints the captured manifest
summary.

### Software capture (Codec 2, M17) { #software-capture }

`unamblify capture --mode codec2-3200 [--jobs N]` (or `codec2-1600`) runs
the same harness with the `codec2` crate in place of the chip. The crate
is version 0.3.1, licensed `LGPL-2.1-only AND MIT`, and sits behind the
default-on `codec2` cargo feature of `unamblify-data`. What differs:

- **No port.** `--port` is refused ("it is a software codec") and nothing
  opens a serial device. `--dry-run` changes nothing (the real encoder
  runs anyway) and says so.
- **Workers are threads.** `--jobs N` runs N encoders in parallel. The
  default is the number of physical cores. The flag is refused for a chip
  mode, whose workers are its ports. `status.json` lists the threads as
  `ports = ["codec2:0", …,
  "codec2:<N-1>"]` and each `CaptureRow.port` names the thread that made
  it, so nothing downstream changes shape.
- **No speaker affinity.** The codec keeps no state across utterances, so
  work is spread greedily by duration rather than by speaker.
- **Fresh state per utterance.** Every `encode` and `decode` starts from
  a new codec instance. An utterance's `.ambe` is therefore a pure
  function of its input and the crate version: the same bytes from any
  worker, any run, any machine. The decoder is not bit-reproducible.
  Like the C reference, it draws unvoiced-harmonic phases from a
  process-global generator (`codec2_rand`, shared by every thread), so
  two decodes of the same frames differ in their noise-like components.
  A real M17 receiver does the same and the training target is the clean
  signal, so this is accepted. `verify` checks the decoded file against
  the hash its own capture recorded, never against a fresh decode. The
  decode is stored as lossless FLAC (`<key>.flac`). That is about half
  the WAV size on disk and to sync, and byte-identical on read. Every
  reader resolves the FLAC if it is present and the WAV otherwise, so a
  set captured before the switch still loads. `sha256_wav` is the hash of
  whichever file was written.
- **Canary, same discipline.** Reset → warm-up → encode the clip →
  sha256, compared to `canary.json` at start, every `--canary-every`
  utterances and after every failure. `reset` is a no-op and the warm-up
  costs a few frames. What the check catches here is a `codec2` crate
  upgrade whose encoder output changed. The crate is pinned with `=`, and
  `prodid = "codec2"`, `version = "0.3.1"` record the pin. The lag is
  measured the same way and lands inside one frame.
- **Frames.** `codec2-3200` is 160 samples / 8 bytes per 20 ms frame,
  like the AMBE modes. `codec2-1600` is 320 samples / 8 bytes per 40 ms
  frame. Its `frames` counts and `frames_s` are therefore half those of
  the same audio in another mode, and `stats` converts hours with
  `frame_ms()`.
- **Throughput** (2026-09-11, Apple M4 Max, 14 physical cores, default
  `--jobs`, a training run sharing the machine). `codec2-3200` processed
  200 VoiceBank utterances (496 s of audio, 24 815 frames) in about one
  second of wall time. `status.json` reported 24 659 frames/s and
  715 475 utt/h, and the closing log line 49 839 frames/s. Per thread the
  encoder costs ≈ 22 µs and the decoder ≈ 10 µs per 20 ms frame (≈ 600×
  real time per core). `codec2-1600` processed 50 utterances (169 s,
  4 219 frames of 40 ms) in under a second: 25 679 frames/s in the log,
  ≈ 41 µs encode and ≈ 22 µs decode per frame per thread. Runs this short
  are dominated by start-up (the `status.json` rates include the final
  write a second later), so take the per-frame costs as the planning
  numbers. The whole 222 590-utterance prepared corpus is minutes of
  Codec 2 capture, against ≈ 1 660 utt/h (weeks) for one ThumbDV.

Pause / resume / stop through `control.json`, the lock file, resume by
manifest, `failed.jsonl` and `status.json` are unchanged.

### CaptureRow

```json
{"key":"vctk/p225_001_mic2","mode":"dstar","frames":156,
 "port":"/dev/cu.usbserial-XXXXXXXX","prodid":"AMBE3000F",
 "version":"V121.E100.XXXX.C110.G514.R014.A0030608.C0020208",
 "encode_ms":0,"decode_ms":0,"roundtrip_ms":1290,
 "sha256_ambe":"…","sha256_wav":"…","captured_at":"…","attempts":1}
```

The stick reports `PRODID AMBE3000F` / `VERSTRING V121.E100…` (probed
2026-09-10). The harness records whatever the chip says and does not
hard-code these. A software row carries `"port":"codec2:3"`,
`"prodid":"codec2"`, `"version":"0.3.1"` (the crate pin).

**Timing fields.** An interleaved chip capture has one wall time, the
round trip, because the two directions overlap. `roundtrip_ms` carries
it and `encode_ms` / `decode_ms` stay at 0. There is no honest way to
split an interleaved round trip, and none is invented. Some rows really
did have separate passes: a software capture, `capture --sequential`,
and the decode-only `augment` stage (`encode_ms` 0, `decode_ms` set).
Those carry `encode_ms` / `decode_ms` and omit `roundtrip_ms`
entirely (`skip_serializing_if`), so rows written before interleaving
existed parse unchanged. `unamblify stats` sums all three over the
manifest and prints whichever are non-zero as `… ms/frame`, so a mixed
manifest reads correctly. `verify` never looks at the timing fields.

### canary.json

```json
{"mode":"dstar","clip":"canary/1khz-and-speech.8k.wav","frames_sha256":"…",
 "frames_first_16":"9e8d…","lag_samples":42,"warm_up_frames":20,"prodid":"…","version":"…",
 "recorded_at":"…"}
```

### Timing

Per 20 ms audio frame on the project stick at 460 800 baud (measured
2026-09-11, 300 pairs per point):

| Regime | ms per audio frame |
|---|---|
| Stop-and-wait, one direction at a time | 40 |
| Two passes, each pipelined four deep (`--sequential`) | 14.28 |
| Interleaved 3 + 3 (the default) | 8.27 |

7.07 ms of any single direction is the 326-byte speech packet's wire
time, and 460 800 is the chip's maximum baud. One direction therefore
cannot go below ≈ 7.1 ms, and depth beyond 4 buys nothing there.
Interleaving is the only lever left on one stick. At 8.27 ms per 20 ms
frame a stick captures a mode at about 2.4× real time (see
[hardware throughput](../research/hardware-throughput.md)). The 100 ms
per-direction age-out leaves an order of magnitude of headroom over the
worst observed round trip. It is the constant to revisit if a stick ever
ages out under load (`pipeline::REPLY_TIMEOUT`, `INTERLEAVE_DEPTH`,
`MAX_IN_FLIGHT`).

## Stage 2b: alignment

The decoded output for frame *n* is the chip's answer to input frame *n*,
so alignment is frame-exact by construction. Inside the frame, the
vocoder's analysis window still introduces a fixed lag. `capture`
measures it once per mode from the canary and records `lag_samples` in
`canary.json`. Prepared targets are not shifted. The shard builder and
the pipeline loader apply the recorded lag when they pair `clean16` with
`deg8`.

## Stage 2c: verify (`unamblify verify --mode M [--sample N]`)

Recomputes `sha256_ambe` / `sha256_wav` for every captured row (or a
seeded sample of them). It checks that `.ambe` is `frames × frame_bytes()`
long, that the `.wav` is 8 kHz with `frames × frame_samples()` samples,
and that `prodid` starts with what the mode's family expects
(`AMBE3000`, or `codec2`). It exits non-zero naming every bad key.

### Renamed modes (`unamblify migrate-modes [--dry-run]`) { #migrate-modes }

A mode's identifier is its directory name under `captured/` and the
`mode` field of every row, status, canary and shard index. Renaming one
therefore leaves old data invisible. When `ysf-dn` became `ysf-dmr` on
2026-09-11 (DMR was folded into it), the code started looking for
`captured/ysf-dmr/`. The retired spellings (`VocoderMode::ALIASES`:
`ysf-dn`, `ysf`, `dmr`) still *parse*, so a manifest row, a run config
or a dashboard URL carrying one loads as `ysf-dmr`. A directory under a
retired name is not a capture set until it is renamed.

`unamblify migrate-modes` does the renaming, idempotently. It renames
`captured/<old>/` and every `captured/<old>+<kind>/` sibling. It rewrites
the `mode` field row by row in `manifest.jsonl` and `failed.jsonl` (a
torn last line is left alone) and in `status.json` and `canary.json`. It
also rewrites `shards/*/index.json` (`mode`, `modes`, and every map keyed
by mode). It prints each rename and each file it rewrote. It refuses a
set whose `lock` a running capture holds (stop the capture first), and it
refuses to rename onto a directory that already exists (merge by hand).
It leaves `runs/*/config.toml` alone as a run's own record. `--dry-run`
prints what would change. A dashboard started before the rename keeps
looking under the old name until it is restarted.

## Stage 3: augment { #stage-3-augment }

Three layers were built on 2026-09-11, at three points of the pipeline:
twins at prepare time (before the codec), a decode-only stage on the
stored channel frames (inside the codec), and receive-side noise in the
training loaders (after the codec). The design reasoning stays with
each. What follows is what the code does.

### Noisy and overdriven-mic twins (prepare time)

Noise is mixed on the input side, before the chip, so the target stays
the clean recording and the model learns to undo the whole transmit
chain (a noisy mic and the vocoder). Mixing after the chip would teach
something different and cheaper: additive noise on decoded audio. That is
not what a receiver hears. AMBE turns background noise into its own
peculiar artefacts, and only the chip can produce those. Noise is
therefore a prepare-time decision that costs chip time, so it is
applied to a fraction of utterances rather than all of them
(`crates/unamblify-data/
src/twins.rs`, the DSP in `unamblify_audio::chain`):

- `unamblify prepare --noise-share 0.25 --noise-sets demand,musan`
  emits one extra noisy twin row, with key `<key>+n<NNNN>`, for a
  deterministic share of utterances. The twin is the parent's trimmed,
  normalised clean 16 kHz signal with a noise clip mixed in. The SNR is
  drawn uniformly from 0–20 dB, measured as the parent's
  active-speech RMS over the RMS of the noise segment. The clip is drawn
  by seed from the requested tier-3 sets, and the set is chosen
  uniformly. Within DEMAND the draw picks an environment's channel file
  (`raw/demand/<ENV>/chNN.wav`), with TCAR, TBUS, STRAFFIC and SPSQUARE
  weighted 3×. Within MUSAN it picks any clip of the `noise` and `music`
  trees, never `speech`, which would confuse the target. The clip is read
  from a random offset (wrapping when the clip is shorter) and resampled
  to 16 kHz. A requested set that is not under `raw/` is an error, not a
  silent half library. The twin's `.16k.wav` is the *noisy* input and its
  `.8k.wav` the decimated noisy chip input. Its row records `parent` and
  `aug.noise` (`noise_set`, `noise_clip`, `noise_offset_s`, `snr_db`).
  Its training target is the parent's clean 16 kHz file. The shard
  builder, the loaders, `infer` and the eval resolve it through `parent`,
  and refuse loudly when the parent row or file is missing.
- `--ham-chain-share 0.25` applies the synthetic transmit chain, either
  alone (`<key>+h<NNNN>`) or on a noisy twin (`<key>+n<NNNN>+h<NNNN>`).
  On a noisy twin the noise is mixed first, since it enters the mic
  before the amp saturates. The chain is recorded as `aug.chain`
  (`ChainParams`). This is where overdriven audio lives, because it
  is what the encoder sees. A clipped waveform confuses AMBE's pitch
  tracker and voicing decisions, so the vocoder's rendering of overdrive
  can only be had by pushing it through the chip. "Talking too close to
  the mic" is drawn as any combination of the stages below (each with
  probability ½, at least one always), applied in this order:

  | Effect | Draw |
  |---|---|
  | proximity effect | low shelf +4 to +10 dB below 200 Hz (a close directional mic's bass boom) |
  | mic response | a ±6 dB tilt (a low shelf cut at 250 Hz and a high shelf boost at 2.5 kHz of half the tilt each) and one resonance, 400–3500 Hz, +3 to +9 dB, Q 2–5, modelling the hand-mic capsule |
  | plosive pops | a decaying 60–120 Hz thump at −10 to −3 dBFS peak on 20–50 % of detected onsets (10 ms frames rising 8 dB over the previous one, ≥ 100 ms apart) |
  | clipping | hard (`clamp`) or soft (`tanh`), modelling the mic amp saturating. The signal is driven so its peak sits 6–24 dB above the ceiling. This is relative to the signal's own peak, so 6 dB always clips |
  | AGC and limiter | gain riding toward a −9 dBFS envelope with 5–50 ms attack, 100–500 ms release (pumping on syllables), then a hard limiter at −1 dBFS |

  The active level is normalised back to −26 dBFS *after* the chain so
  the chip sees its usual input level. The distortion is the
  augmentation, and the loudness is not. The gain applied is
  `aug.post_gain_db` and the row's `gain_db`. It includes any peak guard,
  since every twin is held under 0.99.
- Every decision and draw comes from `key_seed(parent, --seed)`, so a
  re-run emits the same twins with the same bytes. The pass is
  idempotent: a twin whose row and files are present is skipped.
  `--force` redoes the twins, a parent redone in this run has its twins
  redone, and a twin whose parent row has gone is dropped from the
  manifest. Twins take their parent's split, so no speaker leaks. Twins
  are captured exactly like any other utterance, and nothing in `capture`
  knows about them.

The design's `overdrive` eval column (a fixed 12 dB hard clip plus +6 dB
shelf, `ChainParams::fixed_overdrive`) is not reported yet. It needs the
recipe pushed through the codec, which means a captured twin per eval
clip.

### Underdriven twins (prepare time) { #underdriven-twins }

`prepare --underdrive-share S` emits `<key>+u<NNNN>`: the clean signal
15–30 dB below the working level (`twins::UNDER_GAIN_DB`). Half the time
it also gets a 3–8 dB low-shelf *cut* below 200 Hz, the proximity bass a
distant talker loses. With the corpus at −26 dBFS, that feeds the
vocoder at roughly −41 to −56 dBFS, where its gain quantiser goes coarse
and pitch tracking starts to fail.

This is the one twin that is not brought back to the prepare target.
After the overdriven chain the level is renormalised, because there the
distortion is the augmentation and the loudness is not. Here the low
level is the augmentation. The damage happens inside the codec, so
restoring the level first would hand it a healthy signal and the twin
would train on nothing. The attenuation is recorded in
`aug.under.gain_db` and `post_gain_db` stays 0. Nothing downstream puts
the level back either. Capture feeds the prepared 8 kHz audio to the
vocoder as it is, and neither the shard builder nor the loaders
normalise.

This has two consequences. A `+u` twin can sit wholly below the
−50 dBFS active-speech threshold, so `rms_dbfs_in` falls back to plain
RMS for it rather than recording "no speech". And an utterance never
gets both a `+h` and a `+u` twin, since a mic cannot be too hot and too
quiet at once. Noise plus underdrive (`+n…+u…`) is allowed: a quiet
talker in a noisy room. The underdrive decision and its tag are drawn
from a forked stream, so turning the share on moves no noise or chain
decision and renames no existing twin.

`prepare --twins-only` runs just the twin pass over the manifest as it
stands. The main pass walks every raw corpus and stats both files of
every prepared row before it reaches the twins. On the 2.16 M-utterance
corpus that is over four million metadata reads. They change nothing
once the corpus is prepared, and they would be paid again each time a
share is adjusted. `--twins-only` needs a share, and says so rather than
doing nothing.

`capture`, `augment` and `recode` take `--only-twins` to plan just the
rows with a `parent`. The default random order puts every key at a
hashed position. Without the flag, freshly prepared twins surface only as
fast as the base corpus around them is worked through, which takes weeks
on a chip.

### Channel errors and dropped frames (`unamblify augment --mode M --kind drops|ber [--rate R] [--burst lo..hi] [--subst mute|repeat|erase] [--fec none|ysf-vd1|ysf-vd2] [--share 0.3] [--seed 1] [--jobs N] [--limit N] [--port P] [--dry-run]`) { #augment }

A *decode-only* stage, because the stored channel frames make it cheap.
It mutates the `.ambe` frames and re-runs only the decode pass
(7.45 ms/frame on the chip, instant for Codec 2). It writes the result as
a sibling capture `captured/<mode>+<kind>/` with the same keys, the same
clean target, and the mutation positions in the manifest row. It is the
capture harness run in its second `Stage` (`crates/unamblify-data/src/
capture.rs`, the pure parts in `augment.rs` and the core crate's
`channel`). It shares the lock, `control.json`, `status.json` (with a
`kind` field), writer, failure policy, canary check and pipelined decode,
over a `CaptureDir` (base or sibling) that spells every per-set path.
The work list is a seeded `--share` of the base capture's utterances. It
is stable across runs, `--limit` caps one run, and done rows are
skipped. The chip modes need the stick (or `--dry-run` for the sim), and
the Codec 2 modes run on `--jobs` threads. The sibling gets a copy of
the base's `canary.json` (same decoder, same lag, and the base must have
one). The canary check still guards the chip's state between
utterances.

- **Dropped frames** (`--kind drops --rate 0.02 --burst 1..3`): bursts of
  one to three consecutive frames (20–60 ms, a syllable-sized hole),
  started so that about `--rate` of all frames are lost, seeded per
  utterance. What a receiver substitutes for a lost frame decides the
  artefact, so the substitution is a parameter. `mute`, the default,
  sends the mode's mute / null codeword, which is what MMDVM-style hosts
  and astar send the vocoder. `repeat` holds the last good frame, as some
  radios do, and uses the mute word when the first frame is lost. `erase`
  marks the frame bad and lets the receiver's own concealment act.

  On an AMBE mode, `erase` would be the AMBE-3000's own concealment. That
  needs a driver capability this harness does not have
  (`vendor/ambe-thumbdv` is vendored verbatim), so `erase` stays refused
  there. Codec 2 has no concealment of its own, so the harness supplies
  it in the parameter domain (`channel::fill_erasures`). Across a
  gap, the pitch (Wo) and energy indices are interpolated between the
  good frames either side. Both quantisers are uniform, Wo linear in
  radians and energy linear in dB, so interpolating the index
  interpolates the value and no codebook table is needed. The voicing
  bits and the LSP block are held from the nearest good frame. A gap at
  the start holds backward from the first good frame. A gap that runs to
  the end decays the energy toward silence (about 3 dB per frame) rather
  than sustaining a tone. An utterance with no good frame at all falls
  back to `mute`. Holding the envelope matters less than it sounds,
  because the decoder already interpolates its own LSPs from the
  previous frame, and a pitch or energy step is what produces the click.
  Feeding the decoder a plausible frame also keeps its `prev_lsps` /
  `prev_e` state on a smooth trajectory, which muting and repeating do
  not. The frame bit map is pinned by an end-to-end test against the
  real encoder rather than transcribed from the crate's LGPL codebooks.
  It is MSB first with every field Gray coded. 3200 codes one
  pitch/energy point per 20 ms frame, and 1600 codes two per 40 ms frame.

  The codewords are D-STAR's
  `NULL_AMBE_FRAME` `9E 8D 32 88 26 1A 3F 61 E8` and the YSF/DMR
  (AMBE+2 2450) mute frame `YSF_DMR_MUTE_FRAME` `F0 00 31 00 00 00 00`.
  Codec 2 has no fixed mute word, and an all-zero frame is *not*
  silence, so one frame of digital silence is encoded once with the
  crate. Time is never shifted: a lost frame is replaced, not removed, as
  on the air. This is the case lookahead is for. The ll20 model can see
  across a 60 ms gap and bridge it, which no conventional concealment
  can, so the eval reports `eval/lsd_drops` separately over the dev clips
  the `drops` sibling holds.
- **Bit errors** (`--kind ber --rate 1e-3 [--fec none|ysf-vd1|ysf-vd2]`):
  flip channel bits with the given probability. `--fec none` flips each
  valid voice bit directly, never a 49-bit frame's padding. No receiver
  sees that, because on the air the protected bits are protected, so the
  YSF settings model the FEC instead: each frame is wrapped in that V/D
  mode's code, every transmitted bit is flipped with the same
  probability, and the frame is decoded as a receiver would. What reaches
  the vocoder is the residual error. The row records which setting was
  used in `aug.fec`.

  The two YSF DN modes protect different bits, and neither resembles
  DMR. Mode 1 covers twelve voice bits with Golay (24, 12) and twelve
  with Golay (23, 12) and leaves twenty-five bare. Mode 2 sends its
  first twenty-seven bits three times and votes, leaving twenty-two
  bare. Mode 1 carries the interesting failure: the second word is
  whitened by a sequence keyed on the *decoded* first word, so
  mis-correcting the first destroys the second, and a first word more
  than three bits from any codeword cannot be repaired at all. Since it
  holds the key, the whole frame goes and becomes
  `YSF_DMR_MUTE_FRAME`, which is what a receiver substitutes. Bit errors
  therefore produce dropped-frame artefacts as a consequence, not as a
  separate augmentation.

  Measured over 4000 frames, the two modes are nearly identical while
  errors are independent: at a 2 % bit error rate both leave about half
  a wrong voice bit per frame, which is the unprotected bits and almost
  nothing else, and mode 1 loses 0.12 % of frames. They only diverge
  once the rate is high enough to lose frames often. Bursts are a
  different matter, and the reason the on-air placement is not modelled
  is in [the open questions](index.md#open-questions): six contiguous
  bits exhaust mode 1's Golay word and lose the frame, but where a bit
  sits cannot change its error probability while flips are independent.

  DMR is still not captured. Its FEC is the chip's own (2450 + 1150,
  `chip::ratep_dmr`), so modelling it would mean wrapping the 49 bits
  in DMR framing and letting the chip correct what it can.

The row is `CaptureRow` plus `"aug":{"kind":"drops","rate_ppm":20000,
"burst":[1,3],"subst":"mute","seed":1,"positions":[87,88,89]}`
(`positions` are the lost frames, or the frames with at least one flip),
and `encode_ms` is 0. Measured 2026-09-11 on the real `codec2-3200`
capture (Apple M4 Max, 14 threads): `--share 0.02 --limit 100` processed
100 utterances (14 040 frames, 281 s of audio, 261 frames dropped =
1.86 %) in 7.0 s of wall time. The decode was 144 ms of that in total.
The rest is reading the two 222 590-row manifests, which a no-op rerun
spends too (5.8 s). `verify --mode M --kind K` checks a sibling. `stats`
lists every capture set present (a sibling's `remaining` is against the
prepared manifest, as for a base capture). The dashboard's Capture page
lists siblings read-only after the modes, and its Samples page treats
them as further modes of an utterance.

## Stage 3b: recode (`unamblify recode --mode dstar --kind perens [--jobs N] [--corpus C] [--split S] [--limit N]`) { #stage-3b-recode }

A sibling capture that is not an impairment of the base set. It is
the same prepared audio, encoded and decoded again by a second,
independent implementation of the mode's codec. `augment` mutates
channel frames the chip already produced, but `recode` is a full round
trip. It therefore reads `prepared/` and is not confined to the keys the
base capture holds.

Today the only one is `dstar+perens`: the D-STAR AMBE vocoder from Bruce
Perens' `hams_open`, vendored as `vendor/ham-digital-modes`
(LGPL-3.0-or-later, reduced to the D-STAR path, see its `VENDORED.md`).
It runs in software on `--jobs` threads and never opens a port.
`vocoder::open_software` still refuses every AMBE mode, so a plain
`capture --mode dstar` cannot produce software frames by accident. The
recode stage reaches the vocoder through `vocoder::open_recode` on
purpose.

Measured against the chip on 1,298 dev utterances (UTMOS22, clean 4.10),
the AMBE-3000 scores 2.56 and this codec 2.18. It is a different
and somewhat worse D-STAR degradation than any radio produces. That is
why it is kept as a sibling rather than a replacement. `shard --kinds
base,perens` pools both under the one `dstar` mode, so the model must
repair D-STAR from either encoder. Examples carry a mode index, not a kind
index, so the model is not told which codec it is hearing.

`AugKind::Perens` is refused by `augment::mutate`: there are no frames to
impair, and reaching that path means a caller built the wrong stage.

### Receive-side noise (the loaders) { #rx-noise }

Receive-side noise is injected *after* the vocoder, on the decoded 8 kHz
audio, by the training loaders. It is added on the fly, seeded per
example, with the target unchanged, so it costs no capture and stacks
with everything else. It models what the receive path adds to
already-decoded speech: mains and power-supply hum through a hotspot's
audio cable, a radio's audio-stage hiss, alternator whine in a vehicle,
a cheap audio stage's colouring, and a squelch tail. The squelch
tail is a decaying noise burst at the end of the clip, the crash a
receiver makes when the carrier drops at the end of an over (40–250 ms,
−24 to −6 dBFS, exponential decay). The config section is `[augment]`
(`rx_share`, default 0.3, and the per-kind toggles `hum`, `broadband`,
`whine`, `colouring`, `squelch`). The generators are
`unamblify_audio::rx`. One `RxStage`
(`crates/unamblify-train/src/data/mod.rs`) runs over `deg8` after the
crop in the pipeline, shard and synthetic loaders. It is seeded from the
loader's seed and the example's index (the shard loader's global example
index, the pipeline's draw counter), so a shard set and the pipeline draw
the same noise for the same seed and index. Each example in the share
draws any combination (each kind with probability ½, at least one):

| Kind | Draw |
|---|---|
| hum | fundamental from {50, 60, 100, 120} Hz, 3–8 harmonics with 1/n amplitude, −30 to −50 dBFS RMS, ±0.5 Hz slow (0.2 Hz) wobble |
| broadband | white or pink noise at 15–35 dB SNR against the active speech of the crop (none on a silent crop) |
| whine | a tone sweeping linearly 200–600 Hz over the crop with 2–4 harmonics, −35 to −55 dBFS |
| colouring | a ±6 dB tilt (shelves at 250 Hz and 2 kHz) or a single −12 dB notch (Q 4, 300–3000 Hz), and with probability 0.2 soft clipping at −6 to −1 dBFS |

The eval adds `eval/lsd_rx`: the same dev clips with one fixed recipe
(100 Hz hum at −40 dBFS plus white noise at 25 dB SNR,
`RxRecipe::fixed`) on the input, so the gain is measured. Receive-side
noise never replaces the pre-chip twins. The two teach different things.

The loaders treat sibling captures as additional examples of the same
utterance: `[data] kinds = ["base", "drops"]` for the pipeline source,
and the sharder's `--kinds` for a shard set (recorded in `index.json`).

## Stage 4: shard (`unamblify shard --mode M | --modes A,B --name N [--crop-s 2.0] [--onset-share 0.34] [--tail-share 0.15] [--seed 1] [--kinds base,drops,dstar+perens] [--split S] [--no-balance] [--max-utterances N] [--twin-share S] [--min-drop-rate R] [--corpora A,B]`) { #stage-4-shard }

`--twin-share S` (with `--max-utterances`) reserves that fraction of each
capture set's draw for augmented twins, the rows with a `parent`. The
plain draw is uniform over the capture, so it takes twins in proportion.
That starves them exactly where the capture is largest. 42 k twins are a
fifth of the YSF set and a fiftieth of a 2.2 M-utterance Codec 2 one,
and a 50 k draw from the latter would hold about a thousand. Twins and
base are drawn separately, and whichever side runs short hands its
places to the other. The set is still `max` utterances whenever the
capture holds that many. Without the flag the draw is the uniform one,
unchanged, so every set built before the flag existed comes out the
same. `index.json` records the share.

A `--kinds` word may name the one mode it applies to (`dstar+perens`
rather than `perens`) for a sibling that exists for a single mode.
Without the scope the kind is demanded of every mode in the set, and a
missing manifest is an error, so a typo still fails loudly. The scoped
spelling is what `index.json` records.

The builder joins `captured/<mode>/manifest.jsonl` (and, with `--kinds`,
each decode-only sibling's) with `prepared/manifest.jsonl` in mode, kind,
then key order. It packs fixed-length examples into
`shards/<name>/NNNN.bin` (1024 examples per file) as a pure function of
the manifests and the seed. A twin's clean target is its parent's file,
and a twin whose parent is missing is an error. A sibling's examples are
further examples of the same utterances. `index.json` is the core
crate's `ShardIndex`:

```json
{"name":"mixed-dstar-codec2","mode":"dstar","modes":["dstar","codec2-3200"],
 "crop_s":2.0,"seed":1,
 "lag_samples":326,"lags":{"dstar":326,"codec2-3200":140},
 "counts":{"train":2000,"dev":160,"test":80},
 "counts_by_mode":{"train":{"dstar":1000,"codec2-3200":1000},"dev":{…},"test":{…}},
 "available_by_mode":{"train":{"dstar":1000,"codec2-3200":7400},"dev":{…},"test":{…}},
 "balance":true,
 "onset_share":0.34,"tail_share":0.15,
 "kinds":["base"],
 "source_sha256s":{"prepared":"…","captured":"…",
                   "by_set":{"dstar":"…","codec2-3200":"…"}},
 "example_layout":{"clean16_samples":32000,"deg8_samples":16000,
                   "frames":100,"frame_bytes":9,"flags_bytes":9},
 "mode_layouts":{"dstar":{…,"frames":100,"frame_bytes":9},
                 "codec2-3200":{…,"frames":100,"frame_bytes":8}}}
```

`kinds` and `siblings` are absent from sets built before the field
existed (`base` only). `modes`, `lags`, `counts_by_mode`,
`available_by_mode`, `balance`, `by_set` and `mode_layouts` are absent
from sets built before several modes could share one set. Those read
back as `modes = [mode]`, one lag, one layout, every example at mode
index 0, so nothing built earlier needs rebuilding.

**Several modes in one set.** `--modes dstar,codec2-3200` (`--mode M` is
the single-mode alias) joins each mode's capture in turn. `modes` lists
them in index order, and every example carries its mode's index in the
flags byte (bits 2–7, above onset and tail). Those bits are 0 in a set
written before they existed, which is why such sets still load. Each
utterance is cropped in its own mode's frame word with its own canary lag
(`lags`, while `lag_samples` stays the first mode's for older readers).
`mode_layouts` gives each mode's own `frames` / `frame_bytes`.
`example_layout` is the *storage* layout every example shares. Every
mode must agree on its audio fields, so a crop that is not whole frames
of every mode (0.5 s with Codec 2 1600's 40 ms frames) is refused. Its
channel-frame field is sized for the largest mode (`max frames ×
max frame_bytes` over the modes). An example of a smaller mode fills its
own `frames × frame_bytes` bytes of it, and the rest is zero. The
trainer never reads the channel bytes (they ride along for a future
bitstream-conditioned model), so the padding costs disk, not
correctness.

**Balance** is on by default (`--no-balance` keeps everything). Per
split, every mode is capped at the smallest mode's example count and the
surplus is dropped by a seeded draw. A mode with 200 000 utterances then
contributes no more examples than one with 30 000. `available_by_mode`
records what each mode had before the cap and `counts_by_mode` what was
kept. `--split dev` (repeatable) packs only those splits.
`--max-utterances N` keeps a seeded draw of at most `N` utterances per
capture set (recorded as `max_utterances`) for a bounded trial set.

`--min-drop-rate R` draws a `drops` row only if `augment` generated it
at a frame-loss rate of `R` or above (the row's own `aug.rate_ppm`,
recorded as `min_drop_rate`). A capture row carries its own parameters
and resume skips a key already done, so one `<mode>+drops` sibling can
hold several passes. The first Codec 2 pass was 2 % with `mute`. In it
28 % of utterances lose no frame at all, which is too sparse to train
frame restoration on. A heavier pass under another `--seed` lands beside
it, and this flag draws that pass alone. It applies before
`--max-utterances`, so the bounded draw is over eligible rows. Base rows
and every other kind are untouched.

`--corpora A,B` draws only the named corpora (the prepared row's
`corpus`, recorded as `corpora`) from every capture set. It exists
because of what a model regresses onto. Common Voice is nine tenths of
every capture set, and its recordings score 3.15 on the judge that gives
a studio corpus 4.1. A restorer fine-tuned on studio targets alone
gained 0.35 MOS in forty minutes (experiment log #34). A set for
anything generative is built with
`--corpora libritts_r,vctk,ljspeech,voicebank_demand`.

**One lag per capture set, not per mode.** The decoded side lags its
input by the `lag_samples` its capture set's `canary.json` recorded, and
the builder undoes exactly that, per *set*. A decode-only sibling
(`+drops`, `+ber`) shares its base capture's decoder and so its lag. A
recode sibling is another implementation of the codec with another delay
(`dstar` 326 samples, `dstar+perens` 216). `index.json` records both
`lags` (per mode, the base captures) and `set_lags` (per capture set
drawn). Until 2026-09-20 the builder used the mode's lag for every
sibling, which cut every `dstar+perens` example 110 samples off its
target. A model trained on that set scored *below the raw decode* on
D-STAR (experiment log #31) while every spectral metric looked normal.
The trainer now refuses a set that draws a recode sibling and records no
`set_lags`. The pipeline loader follows the same rule.

**The erasure mask.** A set that draws from a `drops` sibling (`--kinds
base,drops`, or a mode-scoped `codec2-3200+drops`) carries a per-frame
mask after the flag block (`erasure_bytes = frames`, `"erasure": true`
in the index). The mask is 1 where the frame's audio is the decoder's
concealment of a lost channel frame, and 0 where the frame arrived. It
comes from the sibling row's recorded `aug.positions`. A base capture's
mask is all zeros. A `ber` sibling's positions are deliberately not
erasures, because a receiver is told that a frame was lost, never that
one of its bits was wrong. The mask is in the example's own frame word
and lag-aligned like `deg8`. The decoded audio of channel frame `k`
reaches `deg8` a codec lag earlier. With D-STAR's 326 samples, a lost
frame therefore damages the two frames *before* its own index, and a
crop frame is marked when any part of it overlaps a lost frame's audio.
The runtime can build the same mask, since the lag is fixed per mode. A
set with no drops sibling stores no mask at all. The field is appended
last, so that set's examples are byte-identical to those of a set built
before the field existed. The input stays the decoder's own concealment
output, never a raw gap. That is what a receiver hears, and it gives the
model's multiplicative paths energy to work with. The mask tells the
model which frames to distrust.

Balancing at build time throws examples away. The trainer's
`[data] mode_weights` instead sets each mode's share of the batches at
draw time and keeps every example. (`mixed-large` is 84 % Codec 2 by
example, so without either an AMBE mode gets one gradient step in six.)

`lag_samples` is the capture lag the builder undid, from `canary.json`.
A build without one is refused, and the training loader refuses a set
that does not record it. `frames` and `deg8_samples` are whole frames of
the mode's own length (`frame_samples()`: a 2 s crop is 100 × 160 in an
AMBE mode or Codec 2 3200, 50 × 320 in Codec 2 1600), and `frame_bytes`
is its `frame_bytes()`. Each example is laid out little-endian and
unpadded:

- `clean16` as f32.
- `deg8` as f32, aligned to it exactly as the training loader's
  `apply_lag` does (`deg8[k] = decoded[k + lag]`). A crop at frame `f`
  reads the decoded signal from `fs·f + lag` and the clean one from
  `2·fs·f`, with `fs` the frame length (onsets from sample 0).
- The channel frames as raw bytes.
- Nine flag bytes: one byte of flags (bit 0 onset, bit 1 tail, bits 2–7
  the mode index), the `mask` boundary as a u32 sample index into
  `clean16` (samples at or after it are garbage-tail padding), and
  `unamblify::speaker_id(speaker)` (the first four bytes of
  `sha1(speaker)`) as a u32.

Examples are ordered by split across the files (train, then dev, then
test, per `counts`). A `files.json` sidecar records which file holds
which split. Per utterance, `floor(frames / crop_frames)` examples are
drawn. `onset_share` of them start at the utterance's first frame.
`tail_share` of them end early and are followed by 0.5–1.5 s of receiver
garbage in `deg8`, over digital silence in `clean16` (mask 0 there, zero
channel frames). The garbage is white noise, the last frame repeated,
gated bursts or rumble, from the core crate's `tail` module (the same
generator the training loader uses, seeded per example). The rest start
at a random frame boundary. A rebuild under the same name removes the
earlier `NNNN.bin` files first, and the loader opens exactly the files
`files.json` lists.

The training loader reads these with `mmap` and shuffles by index.
Opened with the run's `[data] modes`, it keeps only those modes'
examples and renumbers their index to the config's order. The
`pipeline` data source in [training](training.md) reads the WAVs
directly and yields the same tensors. Its crops are frame-aligned with
the mode's `frame_samples()` too, and it draws round-robin over several
modes as a balanced set does. A cross-crate test pins the two for an
AMBE mode, for Codec 2 1600 and for a set holding both. A set written
from the pipeline's examples reads back bit-identically, mode index
included, and every example `unamblify shard` packs is a crop the
pipeline loader would draw from the same aligned utterance of the same
mode.

## Backing up the corpus (`scripts/backup-corpus.sh`)

The drive is the only copy. `scripts/backup-corpus.sh` mirrors the data
root to any destination directory with rsync, so the copy is incremental
and resumable. The destination can be a second disk, or a Google Drive
folder mounted by the desktop app (in stream mode, so the copy uploads
without a full local mirror). Validation is by SHA-256 and separate from
the copy:

```
backup-corpus.sh copy      DEST   # rsync, incremental and resumable
backup-corpus.sh manifest  DEST   # write SHA256SUMS.txt into DEST
backup-corpus.sh verify    DEST   # check DEST against SHA256SUMS.txt
backup-corpus.sh diff      DEST   # rsync dry-run: what still differs
backup-corpus.sh all       DEST   # copy, then manifest, then verify
```

`SHA256SUMS.txt` travels with the data, so whoever receives the folder
can validate their own download with `shasum -a 256 -c SHA256SUMS.txt`,
with no access to the source. `--exclude-common-voice` leaves out the
Common Voice import under `raw/`. A *private* destination is allowed: a
second disk, a personal cloud folder, storage attached to a GPU box, or
a named collaborator. The Mozilla Data Collective terms forbid *public*
redistribution, so the flag is for a copy with a wider audience than
that (see [data sources](../research/data-sources.md)). `RSYNC=` points
at a newer rsync than the macOS built-in, and `UNAMBLIFY_DATA` overrides
the source.

## Backing up the chip captures in chunks (`scripts/backup-chunks.sh`)

Mirroring file for file is the wrong shape for Google Drive. The capture
sets are hundreds of thousands of small files (125 k in `captured/dstar`
alone at 63 k utterances), and Drive creates only a few files per
second. The per-file overhead dominates and the upload never catches up
with the capture. rsync also has to walk both sides, and the remote will
not checksum for you, so `verify` would mean downloading the whole copy
back.

Capture is append-only: a written utterance is never modified again.
(Measured: of 125 306 files in `captured/dstar`, exactly the 6 430
belonging to the last hour's 3 215 utterances had changed.) The backup
therefore never compares the two sides at all. Each run packs the
utterances that are not in the local ledger yet into one immutable
numbered tar, hashes every file in it, and uploads that single object
with rclone. Nothing already uploaded is re-read, re-hashed or re-sent.

```
backup-chunks.sh pack   [--mode M] [--chunk-size BYTES] [--max-chunks N]
backup-chunks.sh push   [--mode M]     # rclone copy new chunks + manifest + canary.json
backup-chunks.sh verify [--mode M]     # local hash vs Drive's stored MD5
backup-chunks.sh verify --deep CHUNK   # pull one back, check every file
backup-chunks.sh status [--mode M]
backup-chunks.sh restore --mode M      # the other direction: pull, check, unpack
```

`restore` is for whoever is handed the backup. One chunk at a time, it
pulls the tar, checks it against `.tar.sha256`, unpacks it under
`captured/<mode>/`, checks every file against the `.sha256` sidecar and
drops the tar. A marker per chunk (`captured/<mode>/.restored/`) makes it
resumable, and `manifest.jsonl` / `canary.json` are fetched only where
none exists, so a live capture's are never overwritten.
[Reproducing from the shared data](reproducing.md) is the walk-through.

Chunks are plain tar, not tar.gz. FLAC and the `.ambe` bitstreams are
already compressed (measured: `gzip -1` saves 3 %), so compressing only
costs time and widens the blast radius of a corrupt byte. Each chunk
carries a `.sha256` sidecar listing every file inside it, a `.tar.sha256`
holding the hash of the tar itself, and a `.keys` file naming the
utterances it holds. The two hashes live in separate files because
their paths are relative to different roots (the file lines to the
capture set, the tar to the chunk directory), so each is valid
`shasum -c` input on its own.
Integrity can therefore be checked at two levels: against the MD5 Drive
computed at upload (`verify`, no download and no work on Drive's side),
or by pulling a chunk back and checking every file against its sidecar
(`verify --deep`). Concatenating the `.sha256` sidecars gives a per-file checksum
manifest of the whole backup without storing it twice, and those
per-file hashes also catch bit-rot on the source drive, which the
chunk-level hash alone would not.

Files touched within `QUIET_SECS` (default 60) are held back, so an
utterance still being written is never packed while capture runs. Such
files land in the next chunk. `$UNAMBLIFY_DATA/backup/<mode>/packed.keys` is
the ledger of what is already in a chunk, and `chunks/` holds the
objects. `RCLONE_REMOTE` (default `gdrive:unamblify-corpus`) points at
the destination, `CHUNK_BYTES` (default 5 GiB) sets the target size.
rclone needs a Google login once (`rclone config`). rclone's shared
client ID is being retired during 2026, so a long-lived backup needs its
own OAuth client ID.
