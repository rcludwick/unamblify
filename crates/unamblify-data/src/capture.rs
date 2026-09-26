// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `capture` (spec §3): every prepared utterance's 8 kHz side through a
//! vocoder and back, with the canary,
//! fail-hard-and-redo, resume by manifest, pause / resume / stop control,
//! and `status.json`. For the AMBE modes that is one worker per ThumbDV
//! with the encode and decode directions interleaved on the full-duplex
//! UART (1.73× the two passes back to back; `--sequential` goes back to
//! them, and its rows carry `encode_ms` / `decode_ms` instead of
//! `roundtrip_ms`), work assigned by speaker; for the software modes
//! (Codec 2, M17) it is `--jobs` threads over the `codec2` crate, no port,
//! work balanced by duration. Both go through [`crate::vocoder::Vocoder`].
//!
//! Failure policy: any timeout, rate-lost, parse error, unexpected packet
//! or count mismatch discards the partial output, resets and fully
//! re-inits the codec (warm-up and canary check included), and re-runs the
//! utterance; after [`CaptureOptions::max_attempts`] it goes to
//! `failed.jsonl` and the run continues. A canary mismatch stops the run.
//!
//! The same harness runs the decode-only augment stage
//! ([`Stage::Augment`], `unamblify augment`): the work list is the base
//! capture's utterances (a seeded share of them), each utterance's stored
//! channel frames are mutated ([`crate::augment`]) and only the decode
//! pass runs, and everything lands in the sibling directory
//! `captured/<mode>+<kind>/` with its own manifest, lock, control and
//! status files and a copy of the base capture's `canary.json` (the same
//! decoder, the same lag; the canary check still guards the chip's state).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ambe_thumbdv::Transport;
use serde::{Deserialize, Serialize};
use unamblify::split::{CORPUS_LIBRITTS_R, CORPUS_LJSPEECH, CORPUS_VCTK, CORPUS_VOICEBANK_DEMAND};
use unamblify::{CanaryRecord, CaptureRow, Split, UtteranceRow, VocoderMode};
use unamblify_audio::write_flac_s16;

use crate::augment::{self, AugmentSpec};
use crate::canary;
use crate::chip::{ChipInfo, open_serial, select_ports};
use crate::control::{
    ControlState, Controller, Decision, RunState, Status, read_control, write_control, write_status,
};
use crate::sim::SimTransport;
use crate::util::{
    FileLock, JsonlWriter, now_rfc3339, read_file, read_jsonl, sha256_hex, write_file, write_jsonl,
};
pub use crate::vocoder::{DynTransport, Opener, TransportOpener};
use crate::vocoder::{
    Vocoder, chip_opener, frames_for, pad_frames, recode_opener, software_opener,
};
use crate::{CaptureDir, DataError, DataRoot, Result};
use unamblify::aug::{AugKind, CaptureAug};

/// What the harness does with each utterance.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Stage {
    /// Encode the prepared 8 kHz audio and decode it back
    /// (`captured/<mode>/`).
    #[default]
    Capture,
    /// Mutate the base capture's channel frames and decode only
    /// (`captured/<mode>+<kind>/`).
    Augment(AugmentSpec),
    /// Encode the prepared audio again with a second, independent
    /// implementation of the mode's codec, and decode it back
    /// (`captured/<mode>+<kind>/`). Unlike [`Stage::Augment`] this is a
    /// full round trip, not a mutation of frames the chip already
    /// produced, so it reads the prepared audio rather than the base
    /// capture.
    Recode(AugKind),
}

/// Options of one capture run.
// Independent command-line switches, not a state machine; the same
// allowance the CLI's own argument struct carries.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct CaptureOptions {
    /// The vocoder mode.
    pub mode: VocoderMode,
    /// `--port` values; empty = every scanned ThumbDV (or one sim). Must
    /// be empty for a software mode.
    pub ports: Vec<String>,
    /// `--jobs`: encoder threads for a software mode (default: the
    /// physical cores). Must be `None` for a chip mode, whose workers are
    /// its ports.
    pub jobs: Option<usize>,
    /// `--corpus`: only these corpora (and, for [`CaptureOrder::Design`],
    /// in this order); empty = every corpus.
    pub corpora: Vec<String>,
    /// `--order`: how the pending utterances are sequenced.
    pub order: CaptureOrder,
    /// `--order-seed`: the seed behind `random` and `balanced`.
    pub order_seed: u64,
    /// Only this split.
    pub split: Option<Split>,
    /// `--only-twins`: capture just the augmented twins (rows with a
    /// `parent`). The random order puts every key at a hashed position,
    /// so freshly prepared twins would otherwise surface only as fast as
    /// the base corpus around them is worked through — weeks, on a chip.
    pub only_twins: bool,
    /// At most this many utterances (after resume skipping).
    pub limit: Option<usize>,
    /// Re-check the canary every this many utterances per worker (0 = only
    /// at start and after re-inits).
    pub canary_every: usize,
    /// Use the simulated chip instead of a serial port.
    pub dry_run: bool,
    /// `--sequential`: run the chip's encode pass and decode pass one
    /// after the other instead of interleaving them. The escape hatch for
    /// a stick that misbehaves with both directions in flight; it costs
    /// 1.73× the wall time. No effect on a software mode, whose passes
    /// are separate anyway.
    pub sequential: bool,
    /// Attempts per utterance before it goes to `failed.jsonl`.
    pub max_attempts: u32,
    /// How often `status.json` is rewritten (≤ 5 s).
    pub status_interval: Duration,
    /// Capture, or the decode-only augment stage.
    pub stage: Stage,
    /// Mix the AMBE encoder's warm-up state across utterances: pick `cold`
    /// or `warm` per utterance from a hash of its key, so training sees
    /// both the post-keyup transient and the locked-encoder case. No
    /// effect on a software mode (its encoder is stateless) or on the
    /// decode-only augment stage.
    pub warmup_mix: bool,
    /// Fraction of utterances captured `cold` (chip reset first) when
    /// `warmup_mix` is on; the rest are `warm`. Clamped to `0.0..=1.0`.
    pub cold_share: f64,
    /// Seed behind the per-utterance `cold`/`warm` choice, so the mix is
    /// reproducible and independent of capture order.
    pub warmup_seed: u64,
}

impl CaptureOptions {
    /// The sibling kind being written, `None` for a base capture.
    #[must_use]
    pub fn kind(&self) -> Option<AugKind> {
        match &self.stage {
            Stage::Capture => None,
            Stage::Augment(spec) => Some(spec.kind),
            Stage::Recode(kind) => Some(*kind),
        }
    }

    /// The directory this run writes.
    #[must_use]
    pub fn out_dir(&self, root: &DataRoot) -> CaptureDir {
        root.capture_dir(self.mode, self.kind())
    }

    /// The base capture of the mode (what the augment stage reads).
    #[must_use]
    pub fn base_dir(&self, root: &DataRoot) -> CaptureDir {
        root.capture_dir(self.mode, None)
    }

    /// Defaults for `mode`: every port, every corpus, canary every 200,
    /// three attempts, status every second.
    #[must_use]
    pub fn new(mode: VocoderMode) -> Self {
        Self {
            mode,
            ports: Vec::new(),
            jobs: None,
            corpora: Vec::new(),
            order: CaptureOrder::Random,
            order_seed: 1,
            split: None,
            only_twins: false,
            limit: None,
            canary_every: 200,
            dry_run: false,
            sequential: false,
            max_attempts: 3,
            status_interval: Duration::from_secs(1),
            stage: Stage::Capture,
            warmup_mix: false,
            cold_share: 0.34,
            warmup_seed: 1,
        }
    }
}

/// The encoder-state condition an utterance is captured under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WarmState {
    /// Chip reset immediately before the encode: the first frames carry
    /// the post-init pitch-lock transient (every real keyup).
    Cold,
    /// The encoder locked onto this voice before the kept encode.
    Warm,
}

impl WarmState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }
}

/// The warm-up condition for an utterance, or `None` when mixing is off,
/// the mode is software (stateless encoder), or the stage is decode-only.
fn warm_state_of(opts: &CaptureOptions, key: &str) -> Option<WarmState> {
    if !opts.warmup_mix || opts.mode.is_software() || !matches!(opts.stage, Stage::Capture) {
        return None;
    }
    let share = opts.cold_share.clamp(0.0, 1.0);
    // `share` is in [0, 1], so `share * 2^32` is a non-negative value in
    // [0, 2^32] that fits a u64; the low 32 bits of the hash are a uniform
    // draw in [0, 2^32). `draw < cutoff` is then true with probability
    // `share`. The cast cannot truncate or lose a sign given those bounds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let cutoff = (share * 4_294_967_296.0) as u64;
    let draw = order_hash(opts.warmup_seed, key) & 0xFFFF_FFFF;
    if draw < cutoff {
        Some(WarmState::Cold)
    } else {
        Some(WarmState::Warm)
    }
}

/// One line of `captured/<mode>/failed.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedRow {
    /// Utterance key.
    pub key: String,
    /// Mode.
    pub mode: VocoderMode,
    /// Attempts made.
    pub attempts: u32,
    /// The last error.
    pub error: String,
    /// Port (or software worker name).
    pub port: String,
    /// RFC 3339.
    pub failed_at: String,
}

/// What a run did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureSummary {
    /// Utterances captured this run.
    pub done: u64,
    /// Utterances that went to `failed.jsonl` this run.
    pub failed: u64,
    /// Utterances skipped because the manifest already had them.
    pub skipped: u64,
    /// Utterances planned for this run.
    pub planned: u64,
    /// How the run ended.
    pub state: RunState,
}

/// How a capture sequences what is still to do. The order only decides
/// *which* utterances the chip reaches first; every run resumes from the
/// manifest, so it can be changed between runs without losing anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureOrder {
    /// The design's corpus order (`voicebank_demand` → `libritts_r`
    /// dev/test → `vctk` → `ljspeech` → `libritts_r` train), or the
    /// `--corpus` order given. How the first captures were run: the chip
    /// spent its first weeks entirely inside two corpora.
    Design,
    /// **Every utterance at a position drawn from a hash of the seed and
    /// its key**, whatever its corpus. At any moment the captured set is a
    /// uniform random sample of everything prepared, so a model trained
    /// on a partial capture leans on no one dataset; and because a key's
    /// position never changes, a corpus prepared later simply drops its
    /// utterances into the remaining order — nothing restarts, nothing
    /// already captured is wasted. The default.
    #[default]
    Random,
    /// Round-robin over corpora, each corpus in its own hashed order:
    /// equal utterance counts per corpus until a small one runs out.
    /// For when the small corpora matter more than their hours.
    Balanced,
}

impl CaptureOrder {
    /// Every variant, for the CLI's help text.
    pub const ALL: [Self; 3] = [Self::Design, Self::Random, Self::Balanced];

    /// The CLI spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Design => "design",
            Self::Random => "random",
            Self::Balanced => "balanced",
        }
    }
}

impl std::fmt::Display for CaptureOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for CaptureOrder {
    type Err = unamblify::ParseEnumError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|o| o.as_str() == s)
            .ok_or_else(|| unamblify::ParseEnumError {
                what: "capture order",
                input: s.to_owned(),
            })
    }
}

/// A key's fixed position in the random order: the first eight bytes of
/// `sha256(seed ‖ key)`. Stable across runs and machines, independent of
/// what else is in the manifest.
#[must_use]
pub fn order_hash(seed: u64, key: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(key.as_bytes());
    let d = h.finalize();
    u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

/// Sort `rows` (already filtered) into the order the run will work in.
fn sequence(rows: &mut [UtteranceRow], opts: &CaptureOptions) {
    match opts.order {
        CaptureOrder::Design => {
            let rank = |r: &UtteranceRow| {
                if opts.corpora.is_empty() {
                    design_rank(r)
                } else {
                    opts.corpora
                        .iter()
                        .position(|c| c == &r.corpus)
                        .unwrap_or(usize::MAX)
                }
            };
            rows.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.key.cmp(&b.key)));
        }
        CaptureOrder::Random => {
            let seed = opts.order_seed;
            rows.sort_by_cached_key(|r| (order_hash(seed, &r.key), r.key.clone()));
        }
        CaptureOrder::Balanced => {
            // Position of each key within its corpus's hashed order, then
            // round-robin over corpora by that position.
            let seed = opts.order_seed;
            let mut per_corpus: BTreeMap<&str, Vec<(u64, &str)>> = BTreeMap::new();
            for r in rows.iter() {
                per_corpus
                    .entry(r.corpus.as_str())
                    .or_default()
                    .push((order_hash(seed, &r.key), r.key.as_str()));
            }
            let mut position: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for keys in per_corpus.values_mut() {
                keys.sort_unstable();
                for (i, (_, k)) in keys.iter().enumerate() {
                    position.insert((*k).to_owned(), i);
                }
            }
            rows.sort_by_cached_key(|r| {
                (
                    position.get(&r.key).copied().unwrap_or(usize::MAX),
                    r.corpus.clone(),
                    r.key.clone(),
                )
            });
        }
    }
}

/// The work list of a run.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Utterances to capture, in order.
    pub rows: Vec<UtteranceRow>,
    /// Already captured (manifest row + both files present).
    pub skipped: u64,
    /// Manifest rows whose files had gone missing: dropped from the
    /// manifest so the re-capture does not leave the key in it twice.
    pub stale: u64,
}

/// The lock file a running harness holds under `captured/<mode>/`.
pub const LOCK_FILE: &str = "lock";

/// Position of a corpus in the design's capture order: `voicebank_demand`
/// → `libritts_r` dev/test → `vctk` → `ljspeech` → `libritts_r` train → the
/// rest.
#[must_use]
pub fn design_rank(row: &UtteranceRow) -> usize {
    match row.corpus.as_str() {
        CORPUS_VOICEBANK_DEMAND => 0,
        CORPUS_LIBRITTS_R if row.split != Split::Train => 1,
        CORPUS_VCTK => 2,
        CORPUS_LJSPEECH => 3,
        CORPUS_LIBRITTS_R => 4,
        _ => 5,
    }
}

/// Build the ordered, resume-filtered work list. A manifest row whose
/// `.ambe` or `.wav` is gone is re-captured, and the manifest is rewritten
/// without it first so the key never appears twice (`stats` and `verify`
/// read every row).
pub fn plan(root: &DataRoot, opts: &CaptureOptions) -> Result<Plan> {
    let prepared: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest())?;
    let out = opts.out_dir(root);
    let captured: Vec<CaptureRow> = read_jsonl(&out.manifest())?;
    let files_present = |r: &CaptureRow| out.ambe(&r.key).exists() && out.decoded(&r.key).exists();
    let stale = captured.iter().filter(|r| !files_present(r)).count() as u64;
    if stale > 0 {
        log::warn!(
            "{}: {stale} row(s) whose files are missing; dropping them from the manifest and re-capturing",
            out.manifest().display()
        );
        write_jsonl(
            &out.manifest(),
            captured.iter().filter(|r| files_present(r)),
        )?;
    }
    let done: std::collections::HashSet<&str> = captured
        .iter()
        .filter(|r| files_present(r))
        .map(|r| r.key.as_str())
        .collect();
    // The augment stage works from the base capture: only its utterances
    // (with both files present), and only the seeded share of them.
    let source: Option<std::collections::HashSet<String>> = match &opts.stage {
        Stage::Capture => None,
        // A recode encodes the prepared audio again, so — unlike an
        // augment, which can only impair frames the chip already made —
        // it is not confined to the base capture's keys. Every prepared
        // utterance is fair game, exactly as for a capture. Kept as its
        // own arm, not merged with `Capture`: the reason it is `None`
        // differs, and that is what a reader needs.
        #[allow(clippy::match_same_arms)]
        Stage::Recode(_) => None,
        Stage::Augment(spec) => {
            let base = opts.base_dir(root);
            let rows: Vec<CaptureRow> = read_jsonl(&base.manifest())?;
            Some(
                rows.into_iter()
                    .filter(|r| base.ambe(&r.key).exists() && base.decoded(&r.key).exists())
                    .filter(|r| spec.selects(&r.key))
                    .map(|r| r.key)
                    .collect(),
            )
        }
    };

    let mut rows: Vec<UtteranceRow> = prepared
        .into_iter()
        .filter(|r| source.as_ref().is_none_or(|s| s.contains(&r.key)))
        .filter(|r| opts.corpora.is_empty() || opts.corpora.iter().any(|c| c == &r.corpus))
        .filter(|r| opts.split.is_none_or(|s| s == r.split))
        .filter(|r| !opts.only_twins || r.parent.is_some())
        .collect();
    sequence(&mut rows, opts);
    let before = rows.len();
    rows.retain(|r| !done.contains(r.key.as_str()));
    let skipped = (before - rows.len()) as u64;
    if let Some(n) = opts.limit {
        rows.truncate(n);
    }
    Ok(Plan {
        rows,
        skipped,
        stale,
    })
}

/// Take the set's `lock`, refusing when another harness holds it or when
/// `status.json` names a live harness (one from before the lock existed).
fn take_lock(out: &CaptureDir) -> Result<FileLock> {
    let dir = out.dir();
    std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    let path = dir.join(LOCK_FILE);
    let name = out.name();
    let busy = |pid: Option<u32>| {
        DataError::Invalid(format!(
            "capture {name} is already running{}; a second harness on one set would interleave \
             manifest.jsonl (stop it first, or use another mode)",
            pid.map_or(String::new(), |p| format!(" (pid {p})"))
        ))
    };
    let lock = FileLock::try_lock(&path)?.ok_or_else(|| busy(FileLock::holder(&path)))?;
    if let Some(pid) = crate::control::live_capture_pid(&out.status_json())
        && pid != std::process::id()
    {
        return Err(busy(Some(pid)));
    }
    Ok(lock)
}

/// A stale `stop` (or `pause`) in `control.json` from an earlier run must
/// not end (or hold) the run that is starting: reset it to `run`, saying
/// what it was. The SIGINT flag is the only stop source at start.
fn reset_control(path: &std::path::Path) -> Result<()> {
    match read_control(path) {
        Ok(ControlState::Run) => Ok(()),
        Ok(prev) => {
            log::warn!(
                "{}: was {prev:?} from an earlier run; reset to run",
                path.display()
            );
            write_control(path, ControlState::Run)
        }
        Err(e) => {
            log::warn!("{}: unreadable ({e}); reset to run", path.display());
            write_control(path, ControlState::Run)
        }
    }
}

/// Assign utterances to `workers` sticks by speaker: every utterance of a
/// speaker goes to one worker, speakers are spread greedily by total
/// duration (largest first), and each worker keeps the plan's order.
/// Returns indices into `rows`.
#[must_use]
pub fn assign_by_speaker(rows: &[UtteranceRow], workers: usize) -> Vec<Vec<usize>> {
    let workers = workers.max(1);
    let mut load: BTreeMap<(String, String), f64> = BTreeMap::new();
    for r in rows {
        *load
            .entry((r.corpus.clone(), r.speaker.clone()))
            .or_insert(0.0) += r.duration_s.max(0.02);
    }
    let mut speakers: Vec<((String, String), f64)> = load.into_iter().collect();
    speakers.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut worker_of: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut totals = vec![0.0f64; workers];
    for (spk, dur) in speakers {
        let w = (0..workers)
            .min_by(|&a, &b| totals[a].total_cmp(&totals[b]).then(a.cmp(&b)))
            .unwrap_or(0);
        totals[w] += dur;
        worker_of.insert(spk, w);
    }
    let mut out = vec![Vec::new(); workers];
    for (i, r) in rows.iter().enumerate() {
        let w = worker_of
            .get(&(r.corpus.clone(), r.speaker.clone()))
            .copied()
            .unwrap_or(0);
        out[w].push(i);
    }
    out
}

/// Spread utterances over `workers` threads with no speaker affinity:
/// greedily by total duration (largest first), each worker keeping the
/// plan's order. For the software codecs, whose state never crosses an
/// utterance. Returns indices into `rows`.
#[must_use]
pub fn assign_balanced(rows: &[UtteranceRow], workers: usize) -> Vec<Vec<usize>> {
    let workers = workers.max(1);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        rows[b]
            .duration_s
            .total_cmp(&rows[a].duration_s)
            .then(a.cmp(&b))
    });
    let mut totals = vec![0.0f64; workers];
    let mut out = vec![Vec::new(); workers];
    for i in order {
        let w = (0..workers)
            .min_by(|&a, &b| totals[a].total_cmp(&totals[b]).then(a.cmp(&b)))
            .unwrap_or(0);
        totals[w] += rows[i].duration_s.max(0.02);
        out[w].push(i);
    }
    for q in &mut out {
        q.sort_unstable();
    }
    out
}

/// The worker names of a software run: `<family>:0` … `<family>:<n-1>`.
#[must_use]
pub fn software_workers(mode: VocoderMode, jobs: usize) -> Vec<String> {
    (0..jobs.max(1))
        .map(|i| format!("{}:{i}", mode.family()))
        .collect()
}

/// The worker names and vocoder opener a run uses: for a chip mode the
/// real ports (VID/PID intersection, busy-probed) or, with `dry_run`, one
/// simulated chip per requested port (`sim:0` when none is requested);
/// for a software mode `--jobs` threads (default: the physical cores) and
/// no port at all. `--port` on a software mode and `--jobs` on a chip mode
/// are refused rather than ignored.
pub fn workers_for(opts: &CaptureOptions) -> Result<(Vec<String>, Opener)> {
    // A recode runs a second software implementation of the mode's codec,
    // so it takes the software branch even for an AMBE mode: threads, no
    // port, and no ThumbDV. A plain capture of the same mode still opens
    // the chip (see `vocoder::open_software`'s own refusal).
    if let Stage::Recode(kind) = &opts.stage {
        if !opts.ports.is_empty() {
            return Err(DataError::Invalid(format!(
                "--port is not used for a {kind} recode: it runs in software \
                 (use --jobs N for the number of encoder threads)"
            )));
        }
        let jobs = opts.jobs.unwrap_or_else(num_cpus::get_physical).max(1);
        return Ok((software_workers(opts.mode, jobs), recode_opener(*kind)));
    }
    if opts.mode.is_software() {
        if !opts.ports.is_empty() {
            return Err(DataError::Invalid(format!(
                "--port is not used for {}: it is a software codec and opens no serial port \
                 (use --jobs N for the number of encoder threads)",
                opts.mode
            )));
        }
        if opts.dry_run {
            log::info!(
                "{}: --dry-run changes nothing for a software codec; the real encoder runs and no \
                 port is opened",
                opts.mode
            );
        }
        let jobs = opts.jobs.unwrap_or_else(num_cpus::get_physical).max(1);
        return Ok((software_workers(opts.mode, jobs), software_opener()));
    }
    if opts.jobs.is_some() {
        return Err(DataError::Invalid(format!(
            "--jobs applies to the software modes only; {} runs one worker per ThumbDV (--port)",
            opts.mode
        )));
    }
    if opts.dry_run {
        let ports = if opts.ports.is_empty() {
            vec!["sim:0".to_owned()]
        } else {
            opts.ports.clone()
        };
        let open: TransportOpener =
            Arc::new(|_: &str| Ok(Box::new(SimTransport::new()) as Box<dyn Transport>));
        return Ok((ports, chip_opener(open, opts.sequential)));
    }
    let open: TransportOpener =
        Arc::new(|p: &str| Ok(Box::new(open_serial(p)?) as Box<dyn Transport>));
    Ok((
        select_ports(&opts.ports)?,
        chip_opener(open, opts.sequential),
    ))
}

/// Run a capture with the workers [`workers_for`] chooses. Installs the
/// SIGINT → stop handler.
pub fn run(root: &DataRoot, opts: &CaptureOptions) -> Result<CaptureSummary> {
    let (ports, opener) = workers_for(opts)?;
    let controller = Controller::new(opts.out_dir(root).control_json());
    controller.install_sigint();
    run_with(root, opts, &ports, &opener, &controller)
}

struct Progress {
    done: u64,
    failed: u64,
    frames: u64,
    current: Vec<Option<String>>,
    paused_workers: usize,
    /// Workers that still have a queue and have not returned; `paused`
    /// means every one of them is holding on `control.json`.
    active_workers: usize,
    canary_ok: Option<bool>,
    info: Option<ChipInfo>,
    error: Option<String>,
    stopped: bool,
    /// When every active worker went on hold (the run is paused), so the
    /// hold does not count toward the rate and ETA.
    paused_since: Option<Instant>,
    /// Total time the run has spent paused, excluding a pause in progress.
    paused_total: Duration,
}

impl Progress {
    /// The run is paused when every worker that still has work is holding.
    fn all_paused(&self) -> bool {
        self.paused_workers > 0 && self.paused_workers >= self.active_workers
    }

    /// Time spent paused so far, including a pause still in progress.
    fn paused_for(&self) -> Duration {
        self.paused_total + self.paused_since.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Fold a pause in progress into the total (call on any resume).
    fn end_pause(&mut self) {
        if let Some(t) = self.paused_since.take() {
            self.paused_total += t.elapsed();
        }
    }
}

/// What the augment stage needs beyond a capture.
struct AugCtx {
    /// Where the channel frames come from.
    base: CaptureDir,
    spec: AugmentSpec,
    /// The mode's mute codeword.
    mute: Vec<u8>,
}

struct Shared {
    root: DataRoot,
    /// The set being written.
    out: CaptureDir,
    aug: Option<AugCtx>,
    opts: CaptureOptions,
    controller: Controller,
    abort: AtomicBool,
    progress: Mutex<Progress>,
    canary: Mutex<Option<CanaryRecord>>,
    clip: Vec<f32>,
    started: Instant,
    /// Utterances planned for this run (not counting `prior_done`).
    total: u64,
    /// Utterances already in the manifest when this run started. Reported
    /// in `status.json`'s `done` and `total` so a restarted run shows the
    /// whole job's progress; rates use only this run's work.
    prior_done: u64,
}

impl Shared {
    /// Wall-clock time this run has been *working*: elapsed minus paused.
    /// Rates and ETAs divide by this so a pause does not drag them down.
    fn active_elapsed(&self, p: &Progress) -> f64 {
        self.started
            .elapsed()
            .saturating_sub(p.paused_for())
            .as_secs_f64()
            .max(1e-3)
    }

    fn progress(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

enum WriterMsg {
    Row(CaptureRow),
    Failed(FailedRow),
}

/// The run with the worker names (ports, or `codec2:<n>`), opener and
/// controller injected.
pub fn run_with(
    root: &DataRoot,
    opts: &CaptureOptions,
    ports: &[String],
    opener: &Opener,
    controller: &Controller,
) -> Result<CaptureSummary> {
    if ports.is_empty() {
        return Err(DataError::PortAbsent);
    }
    let out = opts.out_dir(root);
    let aug = augment_ctx(root, opts)?;
    let _lock = take_lock(&out)?;
    reset_control(&out.control_json())?;
    let plan = plan(root, opts)?;
    let clip = canary::ensure_clip(root)?;
    let existing_canary = match (&aug, canary::load(&out.canary_json())?) {
        (Some(ctx), None) => {
            // The sibling decodes with the same codec and lag; its canary
            // is the base capture's, copied so every reader of the set
            // finds it where a capture set keeps it.
            let rec = canary::load(&ctx.base.canary_json())?.ok_or_else(|| {
                DataError::Invalid(format!(
                    "{}: no canary.json, so the base capture's codec state and lag are unknown; \
                     capture {} first",
                    ctx.base.canary_json().display(),
                    opts.mode
                ))
            })?;
            canary::store(&out.canary_json(), &rec)?;
            Some(rec)
        }
        (_, rec) => rec,
    };
    let total = plan.rows.len() as u64;
    log::info!(
        "capture {}: {} utterances planned, {} already done, {} worker(s): {:?}",
        out.name(),
        total,
        plan.skipped,
        ports.len(),
        ports
    );
    let assignment = if opts.mode.is_software() {
        assign_balanced(&plan.rows, ports.len())
    } else {
        assign_by_speaker(&plan.rows, ports.len())
    };
    let active_workers = assignment.iter().filter(|q| !q.is_empty()).count();

    let shared = Arc::new(Shared {
        root: root.clone(),
        out,
        aug,
        opts: opts.clone(),
        controller: controller.clone(),
        abort: AtomicBool::new(false),
        progress: Mutex::new(Progress {
            done: 0,
            failed: 0,
            frames: 0,
            current: vec![None; ports.len()],
            paused_workers: 0,
            active_workers,
            canary_ok: None,
            info: None,
            error: None,
            stopped: false,
            paused_since: None,
            paused_total: Duration::ZERO,
        }),
        canary: Mutex::new(existing_canary),
        clip,
        started: Instant::now(),
        total,
        prior_done: plan.skipped,
    });

    let started_at = now_rfc3339();
    let worker_errors = drive_workers(&shared, ports, opener, &plan.rows, &assignment, &started_at);

    let (done, failed, stopped) = {
        let p = shared.progress();
        (p.done, p.failed, p.stopped)
    };
    if let Some(err) = worker_errors.iter().find_map(|r| r.as_ref().err()) {
        write_status_file(&shared, ports, &started_at, Some(RunState::Error));
        return Err(DataError::Invalid(err.to_string()));
    }
    let state = if stopped || done + failed < total {
        RunState::Stopped
    } else {
        RunState::Done
    };
    write_status_file(&shared, ports, &started_at, Some(state));
    let summary = CaptureSummary {
        done,
        failed,
        skipped: plan.skipped,
        planned: total,
        state,
    };
    // Leave the plan for the OS too. The main stop tail is the serial
    // `close()` (see `worker`); this only avoids a second, smaller delay:
    // freeing ~180 k rows that have paged out beside a training run would
    // lag the lock release. The process exits next.
    std::mem::forget(plan);
    Ok(summary)
}

/// What the augment stage needs, validated; `None` for a capture.
fn augment_ctx(root: &DataRoot, opts: &CaptureOptions) -> Result<Option<AugCtx>> {
    let Stage::Augment(spec) = &opts.stage else {
        return Ok(None);
    };
    spec.validate(opts.mode)?;
    let base = opts.base_dir(root);
    if !base.manifest().is_file() {
        return Err(DataError::Invalid(format!(
            "{}: no base capture to augment; capture {} first",
            base.manifest().display(),
            opts.mode
        )));
    }
    Ok(Some(AugCtx {
        base,
        spec: spec.clone(),
        mute: augment::mute_frame(opts.mode)?,
    }))
}

/// Spawn the writer and one worker per port, write `status.json` on a
/// timer until every worker is done, and collect the workers' results.
fn drive_workers(
    shared: &Arc<Shared>,
    ports: &[String],
    opener: &Opener,
    rows: &[UtteranceRow],
    assignment: &[Vec<usize>],
    started_at: &str,
) -> Vec<Result<()>> {
    let (tx, rx) = channel::<WriterMsg>();
    std::thread::scope(|scope| {
        let writer = {
            let shared = Arc::clone(shared);
            scope.spawn(move || writer_thread(&shared, &rx))
        };
        let mut handles = Vec::new();
        for (idx, port) in ports.iter().enumerate() {
            // Workers borrow the plan's rows. A per-worker clone of every
            // remaining row doubled the resident set (~200 MB for 180 k
            // rows) and, after hours beside a training run, freeing it on
            // stop faulted all of it back in from swap: 40-60 s between
            // "stop requested" and the final status.
            let queue: &[usize] = &assignment[idx];
            let shared = Arc::clone(shared);
            let tx = tx.clone();
            let opener = Arc::clone(opener);
            let port = port.clone();
            handles.push(scope.spawn(move || {
                let had_work = !queue.is_empty();
                let r = worker(&shared, idx, &port, &opener, rows, queue, &tx);
                if had_work {
                    let mut p = shared.progress();
                    p.active_workers = p.active_workers.saturating_sub(1);
                }
                if let Err(e) = &r {
                    log::error!("worker on {port}: {e}");
                    shared.abort.store(true, Ordering::SeqCst);
                    shared.progress().error = Some(e.to_string());
                }
                r.map_err(|e| DataError::Worker {
                    port: port.clone(),
                    source: Box::new(e),
                })
            }));
        }
        drop(tx);

        // Status writer: this thread, until every worker is done.
        loop {
            let all_done = handles
                .iter()
                .all(std::thread::ScopedJoinHandle::is_finished);
            write_status_file(shared, ports, started_at, None);
            if all_done {
                break;
            }
            std::thread::sleep(shared.opts.status_interval.min(Duration::from_secs(5)));
        }
        let results: Vec<Result<()>> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(DataError::Invalid("worker panicked".to_owned())))
            })
            .collect();
        let _ = writer.join();
        results
    })
}

fn write_status_file(
    shared: &Shared,
    ports: &[String],
    started: &str,
    final_state: Option<RunState>,
) {
    let p = shared.progress();
    let elapsed = shared.active_elapsed(&p);
    #[allow(clippy::cast_precision_loss)]
    let frames_s = p.frames as f64 / elapsed;
    #[allow(clippy::cast_precision_loss)]
    let utt_per_hour = (p.done + p.failed) as f64 / elapsed * 3_600.0;
    let remaining = shared.total.saturating_sub(p.done + p.failed);
    #[allow(clippy::cast_precision_loss)]
    let eta_s = (utt_per_hour > 0.0).then(|| remaining as f64 / utt_per_hour * 3_600.0);
    let paused_s = p.paused_for().as_secs_f64();
    let state = final_state.unwrap_or(if p.error.is_some() {
        RunState::Error
    } else if p.all_paused() {
        RunState::Paused
    } else {
        RunState::Running
    });
    let status = Status {
        state,
        mode: shared.opts.mode,
        kind: shared.opts.kind().map(|k| k.to_string()),
        pid: Some(std::process::id()),
        ports: ports.to_vec(),
        done: shared.prior_done + p.done,
        failed: p.failed,
        total: shared.prior_done + shared.total,
        frames_s,
        utt_per_hour,
        eta_s,
        paused_s,
        current_key: p.current.first().cloned().flatten(),
        current_keys: p.current.clone(),
        started: started.to_owned(),
        updated: now_rfc3339(),
        canary_ok: p.canary_ok,
        prodid: p.info.as_ref().map(|i| i.prodid.clone()),
        version: p.info.as_ref().map(|i| i.version.clone()),
        error: p.error.clone(),
    };
    if let Err(e) = write_status(&shared.out.status_json(), &status) {
        log::warn!("status.json: {e}");
    }
}

fn writer_thread(shared: &Shared, rx: &Receiver<WriterMsg>) {
    let mut manifest = match JsonlWriter::open(&shared.out.manifest()) {
        Ok(w) => w,
        Err(e) => {
            log::error!("manifest: {e}");
            shared.abort.store(true, Ordering::SeqCst);
            shared.progress().error = Some(e.to_string());
            return;
        }
    };
    let mut failed: Option<JsonlWriter> = None;
    while let Ok(msg) = rx.recv() {
        let r = match msg {
            WriterMsg::Row(row) => {
                let r = manifest.append(&row).and_then(|()| manifest.sync());
                let mut p = shared.progress();
                p.done += 1;
                p.frames += u64::from(row.frames);
                let n = p.done + p.failed;
                drop(p);
                if n.is_multiple_of(50) {
                    log_progress(shared, n);
                }
                r
            }
            WriterMsg::Failed(row) => {
                let w = match failed.as_mut() {
                    Some(w) => Ok(w),
                    None => JsonlWriter::open(&shared.out.failed()).map(|w| failed.insert(w)),
                };
                let r = w.and_then(|w| w.append(&row).and_then(|()| w.sync()));
                shared.progress().failed += 1;
                r
            }
        };
        if let Err(e) = r {
            log::error!("manifest write: {e}");
            shared.abort.store(true, Ordering::SeqCst);
            shared.progress().error = Some(e.to_string());
        }
    }
}

fn log_progress(shared: &Shared, n: u64) {
    let p = shared.progress();
    let elapsed = shared.active_elapsed(&p);
    #[allow(clippy::cast_precision_loss)]
    let frames_s = p.frames as f64 / elapsed;
    #[allow(clippy::cast_precision_loss)]
    let uph = n as f64 / elapsed * 3_600.0;
    let remaining = shared.total.saturating_sub(n);
    #[allow(clippy::cast_precision_loss)]
    let eta = remaining as f64 / (uph / 3_600.0).max(1e-9);
    log::info!(
        "{}: {}/{} done ({} failed), {frames_s:.1} frames/s, {uph:.0} utt/h, ETA {:.0} min",
        shared.out.name(),
        shared.prior_done + n,
        shared.prior_done + shared.total,
        p.failed,
        eta / 60.0
    );
}

/// Bring a vocoder up: (re)init, warm-up, canary check. Used at start,
/// after every failure, and for the periodic canary: the chip's encoder
/// carries state across frames, so a byte-for-byte canary is only
/// comparable when every encode of the clip follows the same reset +
/// warm-up preamble as the day-one record
/// (`docs/design/data-pipeline.md`, stage 2 step 4). The software codec
/// starts fresh on every call, so for it this is just the check.
fn bring_up(shared: &Shared, voc: &mut dyn Vocoder, port: &str, first: bool) -> Result<()> {
    if !first {
        voc.reset()
            .map_err(|e| DataError::Invalid(format!("{port}: {e}")))?;
    }
    voc.warm_up(&shared.clip)?;
    canary_check(shared, voc, port)
}

/// Encode the canary and compare it to `canary.json`, creating the record
/// on day one (encode + decode + lag). A mismatch is fatal and never
/// touches the file.
fn canary_check(shared: &Shared, voc: &mut dyn Vocoder, port: &str) -> Result<()> {
    let mut guard = shared
        .canary
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = if let Some(rec) = guard.as_ref() {
        let enc = canary::encode_clip(voc, &shared.clip)?;
        canary::compare(rec, &enc, port)
    } else {
        let (rec, _) = canary::record(voc, &shared.clip)?;
        canary::store(&shared.out.canary_json(), &rec)?;
        log::info!(
            "{port}: canary recorded: sha256 {} lag {} samples, {} {}",
            rec.frames_sha256,
            rec.lag_samples,
            rec.prodid,
            rec.version
        );
        *guard = Some(rec);
        Ok(())
    };
    drop(guard);
    let mut p = shared.progress();
    match &result {
        Ok(()) => {
            if p.canary_ok != Some(false) {
                p.canary_ok = Some(true);
            }
        }
        Err(_) => p.canary_ok = Some(false),
    }
    result
}

/// How many times a worker tries to open and initialise the chip before
/// giving up, and how long it waits between tries. The AMBE-3000 answers
/// its reset within 300 ms in health but occasionally misses one for a
/// second or two (USB hiccup, a busy stick just re-enumerated); a single
/// timeout at open should not kill a capture that runs for weeks, the same
/// way a mid-run reset is already retried.
const OPEN_ATTEMPTS: usize = 4;
const OPEN_RETRY_DELAY: Duration = Duration::from_millis(500);

/// Open and initialise the vocoder, retrying a transient failure (a chip
/// that misses its reset) up to [`OPEN_ATTEMPTS`] times. A stop requested
/// during the wait ends it early.
fn open_with_retry(
    shared: &Shared,
    opener: &Opener,
    port: &str,
    mode: VocoderMode,
) -> Result<Box<dyn Vocoder>> {
    let mut last: Option<DataError> = None;
    for attempt in 1..=OPEN_ATTEMPTS {
        match opener(port, mode) {
            Ok(voc) => return Ok(voc),
            Err(e) => {
                log::warn!("{port}: open/init attempt {attempt}/{OPEN_ATTEMPTS} failed: {e}");
                last = Some(e);
            }
        }
        if attempt < OPEN_ATTEMPTS {
            if shared.controller.stop_requested() || shared.abort.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(OPEN_RETRY_DELAY);
        }
    }
    Err(last.unwrap_or_else(|| DataError::Invalid(format!("{port}: open failed"))))
}

fn worker(
    shared: &Shared,
    idx: usize,
    port: &str,
    opener: &Opener,
    rows: &[UtteranceRow],
    queue: &[usize],
    tx: &Sender<WriterMsg>,
) -> Result<()> {
    if queue.is_empty() {
        log::info!("{port}: nothing assigned");
        return Ok(());
    }
    let mut voc = open_with_retry(shared, opener, port, shared.opts.mode)?;
    let info = voc.info();
    log::info!(
        "{port}: {} {} initialised for {}",
        info.prodid,
        info.version,
        shared.opts.mode
    );
    if idx == 0 {
        shared.progress().info = Some(info);
    }
    let result = capture_queue(shared, idx, port, voc.as_mut(), rows, queue, tx);

    // Hand the vocoder to the OS instead of dropping it here. Closing the
    // FTDI's serial fd blocks: on a blocking fd macOS drains TX on close,
    // and when the stick is in a degraded state the USB close stalls for
    // tens of seconds (measured 39-62 s after long runs; a stack sample of
    // the tail is 100 % in `close()`, RSS flat). That close sat between
    // "stop requested" and this worker returning, which gates the final
    // status write and the lock release, so the dashboard showed the
    // capture "running" long after it had stopped. The process exits right
    // after the run; the kernel closes the fd at exit.
    std::mem::forget(voc);
    result
}

/// The capture loop, once the chip is open and brought up. Split out so
/// [`worker`] can leak the vocoder (skip the slow serial `close()`) the
/// moment this returns; see the call site.
#[allow(clippy::too_many_arguments)]
fn capture_queue(
    shared: &Shared,
    idx: usize,
    port: &str,
    voc: &mut dyn Vocoder,
    rows: &[UtteranceRow],
    queue: &[usize],
    tx: &Sender<WriterMsg>,
) -> Result<()> {
    bring_up(shared, voc, port, true)?;

    let mut since_canary = 0usize;
    for &i in queue {
        let row = &rows[i];
        if shared.abort.load(Ordering::SeqCst) {
            return Ok(());
        }
        let decision = shared.controller.checkpoint(|paused| {
            let mut p = shared.progress();
            if paused {
                p.paused_workers += 1;
                p.current[idx] = None;
                if p.all_paused() && p.paused_since.is_none() {
                    p.paused_since = Some(Instant::now());
                }
                log::info!("{port}: paused (control.json), port held open");
            } else {
                p.paused_workers = p.paused_workers.saturating_sub(1);
                p.end_pause();
                log::info!("{port}: resumed");
            }
        });
        if decision == Decision::Stop {
            shared.progress().stopped = true;
            log::info!("{port}: stop requested, exiting after the last utterance");
            return Ok(());
        }
        if shared.opts.canary_every > 0 && since_canary >= shared.opts.canary_every {
            log::info!("{port}: periodic canary: reset, warm-up, re-encode");
            bring_up(shared, voc, port, false)?;
            since_canary = 0;
        }
        shared.progress().current[idx] = Some(row.key.clone());
        match capture_utterance(shared, voc, port, row) {
            Ok(cap) => {
                let _ = tx.send(WriterMsg::Row(cap));
            }
            Err(UttError::Failed(f)) => {
                log::error!(
                    "{port}: {} FAILED after {} attempts: {}",
                    f.key,
                    f.attempts,
                    f.error
                );
                let _ = tx.send(WriterMsg::Failed(f));
            }
            Err(UttError::Fatal(e)) => return Err(e),
        }
        shared.progress().current[idx] = None;
        since_canary += 1;
    }
    Ok(())
}

enum UttError {
    Failed(FailedRow),
    Fatal(DataError),
}

/// What one utterance's attempts work from.
enum Input {
    /// The prepared 8 kHz audio (capture).
    Pcm(Vec<f32>),
    /// Mutated channel frames and their record (augment).
    Frames(Vec<u8>, CaptureAug),
}

/// One utterance's passes, before anything is written.
struct Captured {
    frames: u32,
    ambe: Vec<u8>,
    pcm: Vec<i16>,
    encode_ms: u64,
    decode_ms: u64,
    roundtrip_ms: Option<u64>,
    /// The augment stage's mutation record.
    aug: Option<CaptureAug>,
}

fn elapsed_ms(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// `.ambe` must be exactly `frames` whole frames of the mode.
fn check_frames(mode: VocoderMode, frames: u32, ambe: &[u8]) -> Result<()> {
    if ambe.len() != frames as usize * mode.frame_bytes() {
        return Err(DataError::Invalid(format!(
            "encode produced {} bytes for {frames} frames",
            ambe.len()
        )));
    }
    Ok(())
}

/// The decoded PCM must be exactly `frames` whole frames of the mode.
fn check_samples(mode: VocoderMode, frames: u32, pcm: &[i16]) -> Result<()> {
    if pcm.len() != frames as usize * mode.frame_samples() {
        return Err(DataError::Invalid(format!(
            "decode produced {} samples for {frames} frames",
            pcm.len()
        )));
    }
    Ok(())
}

/// The decode-only pass of the augment stage: `ambe` are the mutated
/// frames.
fn decode_once(
    voc: &mut dyn Vocoder,
    mode: VocoderMode,
    ambe: Vec<u8>,
    aug: CaptureAug,
) -> Result<Captured> {
    let frames = u32::try_from(ambe.len() / mode.frame_bytes())
        .map_err(|_| DataError::Invalid("utterance too long".to_owned()))?;
    let t = Instant::now();
    let pcm = voc.decode(&ambe)?;
    let decode_ms = elapsed_ms(t);
    check_samples(mode, frames, &pcm)?;
    Ok(Captured {
        frames,
        ambe,
        pcm,
        encode_ms: 0,
        decode_ms,
        roundtrip_ms: None,
        aug: Some(aug),
    })
}

fn capture_once(voc: &mut dyn Vocoder, mode: VocoderMode, pcm8k: &[f32]) -> Result<Captured> {
    let frames = u32::try_from(frames_for(pcm8k.len(), mode))
        .map_err(|_| DataError::Invalid("utterance too long".to_owned()))?;
    let padded = pad_frames(pcm8k, mode);
    // Interleaved, the two directions overlap and only the round trip has
    // a wall time; sequential, each pass is timed on its own and the
    // encode's frame count is checked before the decode spends the link
    // on it. Either way both counts are checked before anything is kept.
    let (ambe, pcm, encode_ms, decode_ms, roundtrip_ms) = if voc.interleaves() {
        let t = Instant::now();
        let (ambe, pcm) = voc.round_trip(&padded)?;
        (ambe, pcm, 0, 0, Some(elapsed_ms(t)))
    } else {
        let t = Instant::now();
        let ambe = voc.encode(&padded)?;
        let encode_ms = elapsed_ms(t);
        check_frames(mode, frames, &ambe)?;
        let t = Instant::now();
        let pcm = voc.decode(&ambe)?;
        (ambe, pcm, encode_ms, elapsed_ms(t), None)
    };
    check_frames(mode, frames, &ambe)?;
    check_samples(mode, frames, &pcm)?;
    Ok(Captured {
        frames,
        ambe,
        pcm,
        encode_ms,
        decode_ms,
        roundtrip_ms,
        aug: None,
    })
}

fn capture_utterance(
    shared: &Shared,
    voc: &mut dyn Vocoder,
    port: &str,
    row: &UtteranceRow,
) -> std::result::Result<CaptureRow, UttError> {
    let mode = shared.opts.mode;
    // What each attempt works from: the prepared audio (capture), or the
    // base capture's frames, mutated once (augment).
    let input = match &shared.aug {
        None => {
            let path8k = shared.root.prepared_8k_decoded(&row.key);
            let audio = unamblify_audio::read(&path8k).map_err(|e| UttError::Fatal(e.into()))?;
            if audio.rate != unamblify::VOCODER_SAMPLE_RATE {
                return Err(UttError::Fatal(DataError::Invalid(format!(
                    "{}: {} Hz, expected 8000",
                    path8k.display(),
                    audio.rate
                ))));
            }
            Input::Pcm(audio.samples)
        }
        Some(ctx) => {
            let ambe = read_file(&ctx.base.ambe(&row.key)).map_err(UttError::Fatal)?;
            let (mutated, aug) = augment::mutate(mode, &ctx.spec, &row.key, &ambe, &ctx.mute)
                .map_err(UttError::Fatal)?;
            Input::Frames(mutated, aug)
        }
    };
    // The encoder-state condition for this utterance (chip capture only).
    let warm = warm_state_of(&shared.opts, &row.key);
    let max = shared.opts.max_attempts.max(1);
    let mut last_err = String::new();
    for attempt in 1..=max {
        let result = match &input {
            // Set the encoder's warm-up state before the kept encode: a
            // reset for `cold` (first frames carry the keyup transient),
            // or lock it onto this voice for `warm`. A failure here falls
            // through to the same reset-and-retry as an encode failure.
            Input::Pcm(pcm) => match warm {
                Some(WarmState::Cold) => voc.reset(),
                Some(WarmState::Warm) => voc.warm_up(pcm),
                None => Ok(()),
            }
            .and_then(|()| capture_once(voc, mode, pcm)),
            Input::Frames(ambe, aug) => decode_once(voc, mode, ambe.clone(), aug.clone()),
        };
        match result {
            Ok(cap) => {
                let ambe_path = shared.out.ambe(&row.key);
                let audio_path = shared.out.flac(&row.key);
                write_file(&ambe_path, &cap.ambe).map_err(UttError::Fatal)?;
                let pcm_f: Vec<f32> = cap.pcm.iter().map(|&s| f32::from(s) / 32_767.0).collect();
                if let Some(parent) = audio_path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| UttError::Fatal(DataError::io(parent, e)))?;
                }
                // Lossless FLAC: half the WAV size on disk and to sync, and
                // the decode is byte-identical. Encoding is in-memory and
                // ~1 ms against a chip round trip of the utterance's whole
                // length, so it does not gate the AMBE pipeline.
                write_flac_s16(&audio_path, &pcm_f, unamblify::VOCODER_SAMPLE_RATE)
                    .map_err(|e| UttError::Fatal(e.into()))?;
                let sha256_wav = crate::util::sha256_file(&audio_path).map_err(UttError::Fatal)?;
                let info = voc.info();
                return Ok(CaptureRow {
                    key: row.key.clone(),
                    mode,
                    frames: cap.frames,
                    port: port.to_owned(),
                    prodid: info.prodid,
                    version: info.version,
                    encode_ms: cap.encode_ms,
                    decode_ms: cap.decode_ms,
                    roundtrip_ms: cap.roundtrip_ms,
                    sha256_ambe: sha256_hex(&cap.ambe),
                    sha256_wav,
                    captured_at: now_rfc3339(),
                    attempts: attempt,
                    warm_state: warm.map(|w| w.as_str().to_owned()),
                    aug: cap.aug,
                });
            }
            Err(e) => {
                last_err = e.to_string();
                log::warn!(
                    "{port}: {} attempt {attempt}/{max} failed: {e}; resetting and re-initialising",
                    row.key
                );
                bring_up(shared, voc, port, false).map_err(UttError::Fatal)?;
            }
        }
    }
    Err(UttError::Failed(FailedRow {
        key: row.key.clone(),
        mode,
        attempts: max,
        error: last_err,
        port: port.to_owned(),
        failed_at: now_rfc3339(),
    }))
}

/// Summary of a mode's capture manifest (`--stats`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestStats {
    /// Mode.
    pub mode: VocoderMode,
    /// The sibling kind, absent for a base capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<AugKind>,
    /// Rows in `captured/<mode>/manifest.jsonl`.
    pub captured: u64,
    /// Rows in `failed.jsonl`.
    pub failed: u64,
    /// Rows in `prepared/manifest.jsonl`.
    pub prepared: u64,
    /// Prepared but not captured.
    pub remaining: u64,
    /// Frames captured.
    pub frames: u64,
    /// Hours of audio captured.
    pub hours: f64,
    /// Captured rows per corpus.
    pub by_corpus: BTreeMap<String, u64>,
    /// Captured rows per split (joined with the prepared manifest).
    pub by_split: BTreeMap<Split, u64>,
    /// Mean encode ms per frame. Zero for a set captured with the two
    /// directions interleaved, which has no separable encode time — read
    /// `roundtrip_ms_per_frame` instead.
    pub encode_ms_per_frame: f64,
    /// Mean decode ms per frame. Zero for a set captured with the two
    /// directions interleaved.
    pub decode_ms_per_frame: f64,
    /// Mean ms per frame of the interleaved round trip. Zero for a set
    /// whose rows timed the two passes separately (software captures,
    /// `--sequential`, the decode-only `augment` stage), and a per-frame
    /// mean over *every* row for a mixed manifest, so the three numbers
    /// always add up to what the set really cost.
    #[serde(default)]
    pub roundtrip_ms_per_frame: f64,
    /// Rows that needed more than one attempt.
    pub retried: u64,
}

/// Compute [`ManifestStats`] of a base capture.
pub fn stats(root: &DataRoot, mode: VocoderMode) -> Result<ManifestStats> {
    stats_in(root, &root.capture_dir(mode, None))
}

/// Compute [`ManifestStats`] of any capture set (base or sibling).
pub fn stats_in(root: &DataRoot, dir: &CaptureDir) -> Result<ManifestStats> {
    let mode = dir.mode();
    let prepared: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest())?;
    let captured: Vec<CaptureRow> = read_jsonl(&dir.manifest())?;
    let failed: Vec<FailedRow> = read_jsonl(&dir.failed())?;
    let split_of: std::collections::HashMap<&str, Split> =
        prepared.iter().map(|r| (r.key.as_str(), r.split)).collect();
    let mut by_corpus = BTreeMap::new();
    let mut by_split = BTreeMap::new();
    let (mut frames, mut enc, mut dec, mut retried) = (0u64, 0u64, 0u64, 0u64);
    let mut round = 0u64;
    for r in &captured {
        let corpus = unamblify::key::corpus_of(&r.key).unwrap_or("?").to_owned();
        *by_corpus.entry(corpus).or_insert(0) += 1;
        if let Some(s) = split_of.get(r.key.as_str()) {
            *by_split.entry(*s).or_insert(0) += 1;
        }
        frames += u64::from(r.frames);
        enc += r.encode_ms;
        dec += r.decode_ms;
        round += r.roundtrip_ms.unwrap_or(0);
        if r.attempts > 1 {
            retried += 1;
        }
    }
    let captured_keys: std::collections::HashSet<&str> =
        captured.iter().map(|r| r.key.as_str()).collect();
    let remaining = prepared
        .iter()
        .filter(|r| !captured_keys.contains(r.key.as_str()))
        .count() as u64;
    #[allow(clippy::cast_precision_loss)]
    let per_frame = |ms: u64| {
        if frames == 0 {
            0.0
        } else {
            ms as f64 / frames as f64
        }
    };
    #[allow(clippy::cast_precision_loss)]
    let hours = frames as f64 * f64::from(mode.frame_ms()) / 1_000.0 / 3_600.0;
    Ok(ManifestStats {
        mode,
        kind: dir.kind(),
        captured: captured.len() as u64,
        failed: failed.len() as u64,
        prepared: prepared.len() as u64,
        remaining,
        frames,
        hours,
        by_corpus,
        by_split,
        encode_ms_per_frame: per_frame(enc),
        decode_ms_per_frame: per_frame(dec),
        roundtrip_ms_per_frame: per_frame(round),
        retried,
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::control::{ControlState, write_control};
    use crate::sim::Faults;
    use crate::testutil::sine;
    use crate::util::sha256_file;
    use std::sync::atomic::AtomicU64;
    use unamblify_audio::write_wav_s16;

    /// A prepared root with `n` short utterances over two speakers.
    fn fixture_root(n: usize) -> (tempfile::TempDir, DataRoot) {
        fixture_root_with(n, 2)
    }

    /// A prepared root with `n` short utterances over `speakers` speakers.
    fn fixture_root_with(n: usize, speakers: usize) -> (tempfile::TempDir, DataRoot) {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let mut w = JsonlWriter::open(&root.prepared_manifest()).unwrap();
        for i in 0..n {
            let spk = if i % speakers.max(1) == 0 {
                "p225"
            } else {
                "p226"
            };
            let key = format!("vctk/{spk}_{i:03}_mic2");
            let samples = 8_000 + 160 * i; // 1 s + i frames
            let pcm = sine(200.0 + 50.0 * i as f32, 8_000, samples, 0.3);
            let p8 = root.prepared_8k(&key);
            std::fs::create_dir_all(p8.parent().unwrap()).unwrap();
            write_wav_s16(&p8, &pcm, 8_000).unwrap();
            let p16 = root.prepared_16k(&key);
            write_wav_s16(&p16, &vec![0.0f32; samples * 2], 16_000).unwrap();
            w.append(&UtteranceRow {
                key,
                corpus: "vctk".to_owned(),
                speaker: spk.to_owned(),
                gender: None,
                split: Split::Train,
                duration_s: samples as f64 / 8_000.0,
                src_rate: 48_000,
                src_path: "raw/x".to_owned(),
                licence: "CC-BY-4.0".to_owned(),
                rms_dbfs_in: -20.0,
                gain_db: -6.0,
                trim_lead_s: 0.0,
                trim_tail_s: 0.0,
                sha256_16k: String::new(),
                sha256_8k: String::new(),
                prepared_at: now_rfc3339(),
                parent: None,
                aug: None,
            })
            .unwrap();
        }
        w.sync().unwrap();
        (dir, root)
    }

    fn sim_opener(faults: Faults) -> Opener {
        sim_opener_with(faults, false)
    }

    fn sim_opener_with(faults: Faults, sequential: bool) -> Opener {
        let n = AtomicU64::new(0);
        chip_opener(
            Arc::new(move |_: &str| {
                // Only the first opened transport carries the faults.
                let first = n.fetch_add(1, Ordering::SeqCst) == 0;
                let f = if first {
                    faults.clone()
                } else {
                    Faults::default()
                };
                Ok(Box::new(SimTransport::new().with_faults(f)) as Box<dyn Transport>)
            }),
            sequential,
        )
    }

    fn opts(mode: VocoderMode) -> CaptureOptions {
        let mut o = CaptureOptions::new(mode);
        o.dry_run = true;
        o.canary_every = 2;
        o.status_interval = Duration::from_millis(50);
        o
    }

    #[test]
    fn dry_run_captures_writes_manifest_canary_and_status_then_resumes() {
        let (_dir, root) = fixture_root(5);
        let o = opts(VocoderMode::YsfDmr);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 5);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.state, RunState::Done);

        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        assert_eq!(rows.len(), 5);
        for r in &rows {
            assert_eq!(r.mode, VocoderMode::YsfDmr);
            assert_eq!(r.attempts, 1);
            assert_eq!(r.port, "sim:0");
            assert_eq!(r.prodid, "AMBE3000F");
            let ambe = root.captured_ambe(o.mode, &r.key);
            let wav = root.captured_decoded(o.mode, &r.key);
            assert_eq!(
                std::fs::metadata(&ambe).unwrap().len(),
                u64::from(r.frames) * 7
            );
            assert_eq!(sha256_file(&ambe).unwrap(), r.sha256_ambe);
            assert_eq!(sha256_file(&wav).unwrap(), r.sha256_wav);
            let a = unamblify_audio::read(&wav).unwrap();
            assert_eq!(a.rate, 8_000);
            assert_eq!(a.samples.len(), r.frames as usize * 160);
        }
        // 1 s + i frames → 50 + i frames.
        assert_eq!(rows.iter().map(|r| r.frames).sum::<u32>(), 50 * 5 + 10);

        let rec = canary::load(&root.canary_json(o.mode)).unwrap().unwrap();
        assert_eq!(rec.lag_samples, 42);
        assert_eq!(rec.mode, VocoderMode::YsfDmr);
        let status = crate::control::read_status(&root.status_json(o.mode)).unwrap();
        assert_eq!(status.state, RunState::Done);
        assert_eq!(status.done, 5);
        assert_eq!(status.total, 5);
        assert_eq!(status.canary_ok, Some(true));
        assert_eq!(status.prodid.as_deref(), Some("AMBE3000F"));
        assert_eq!(
            status.pid,
            Some(std::process::id()),
            "the dashboard keys liveness on status.json's pid"
        );
        assert_eq!(
            rec.warm_up_frames,
            crate::chip::WARM_UP_FRAMES,
            "the preamble is recorded"
        );

        // Resume: nothing left to do.
        let again = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(again.done, 0);
        assert_eq!(again.skipped, 5);
        assert_eq!(again.planned, 0);
        assert_eq!(
            read_jsonl::<CaptureRow>(&root.captured_manifest(o.mode))
                .unwrap()
                .len(),
            5
        );
        let s = stats(&root, o.mode).unwrap();
        assert_eq!(s.captured, 5);
        assert_eq!(s.remaining, 0);
        assert_eq!(s.by_corpus["vctk"], 5);
        assert_eq!(s.by_split[&Split::Train], 5);

        // A row whose files vanished is re-captured, and the manifest
        // still names every key exactly once with the new hash.
        let victim = rows[1].key.clone();
        std::fs::remove_file(root.captured_decoded(o.mode, &victim)).unwrap();
        let third = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(third.done, 1);
        assert_eq!(third.skipped, 4);
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows.iter().filter(|r| r.key == victim).count(), 1);
        let row = rows.iter().find(|r| r.key == victim).unwrap();
        assert_eq!(
            sha256_file(&root.captured_decoded(o.mode, &victim)).unwrap(),
            row.sha256_wav
        );
        assert_eq!(stats(&root, o.mode).unwrap().captured, 5);
    }

    #[test]
    fn a_stale_stop_in_control_json_does_not_end_the_next_run() {
        let (_dir, root) = fixture_root(3);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        // Friday: `unamblify capture stop`. Monday: `unamblify capture`.
        write_control(&root.control_json(o.mode), ControlState::Stop).unwrap();
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, summary.planned);
        assert_eq!(summary.done, 3);
        assert_eq!(summary.state, RunState::Done);
        assert_eq!(
            crate::control::read_control(&root.control_json(o.mode)).unwrap(),
            ControlState::Run
        );
        // A stale pause is reset the same way (else the run starts held).
        write_control(&root.control_json(o.mode), ControlState::Pause).unwrap();
        std::fs::remove_file(root.captured_manifest(o.mode)).unwrap();
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 3);
    }

    #[test]
    fn periodic_canary_checks_start_from_a_reset_like_the_record() {
        // A stateful encoder (the real chip's pitch tracker): the same clip
        // hashes differently after an utterance than after reset + warm-up.
        // The record is made from reset + warm-up, so every re-check must
        // be too — a bare re-encode mid-run would stop a healthy run.
        let (_dir, root) = fixture_root(4);
        let mut o = opts(VocoderMode::Dstar);
        o.canary_every = 1;
        let faults = Faults {
            history: true,
            ..Faults::default()
        };
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(&root, &o, &ports, &sim_opener(faults), &controller).unwrap();
        assert_eq!(summary.done, 4);
        assert_eq!(summary.state, RunState::Done);
        let status = crate::control::read_status(&root.status_json(o.mode)).unwrap();
        assert_eq!(status.canary_ok, Some(true));
    }

    #[test]
    fn a_second_harness_on_the_same_mode_is_refused() {
        let (_dir, root) = fixture_root(2);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let lock_path = root.captured(o.mode).join(LOCK_FILE);
        let held = FileLock::try_lock(&lock_path).unwrap().unwrap();
        let err = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap_err();
        assert!(err.to_string().contains("already running"), "{err}");
        assert!(
            !root.captured_manifest(o.mode).exists(),
            "nothing captured while another harness holds the mode"
        );
        drop(held);
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 2);
    }

    /// A chip capture is interleaved: the row records the round trip and
    /// leaves the two pass times at 0 rather than inventing a split, and
    /// `stats` reports the round trip instead of two zeros. The very same
    /// run with `--sequential` produces byte-identical files and the old
    /// split timing.
    #[test]
    fn a_chip_capture_records_the_round_trip_and_sequential_records_the_split() {
        let (_dir, root) = fixture_root(2);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 2);
        let fast: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        assert_eq!(fast.len(), 2);
        for r in &fast {
            assert_eq!(r.encode_ms, 0, "{}: interleaved rows do not split", r.key);
            assert_eq!(r.decode_ms, 0, "{}: interleaved rows do not split", r.key);
            assert!(r.roundtrip_ms.is_some(), "{}: no round-trip time", r.key);
        }
        // Zeroed pass times must not confuse `stats`, and `verify` never
        // reads the timing fields at all.
        let st = stats(&root, o.mode).unwrap();
        assert_eq!(st.captured, 2);
        assert!((st.encode_ms_per_frame - 0.0).abs() < f64::EPSILON);
        assert!((st.decode_ms_per_frame - 0.0).abs() < f64::EPSILON);
        assert!(st.roundtrip_ms_per_frame >= 0.0);
        let v = crate::verify::run(
            &root,
            &crate::verify::VerifyOptions {
                mode: o.mode,
                kind: None,
                sample: 0,
                seed: 1,
            },
        )
        .unwrap();
        assert!(v.problems.is_empty(), "{:?}", v.problems);

        // The same corpus through the sequential path: same bytes, the
        // old timing shape.
        let (_dir2, root2) = fixture_root(2);
        let mut seq = opts(VocoderMode::Dstar);
        seq.sequential = true;
        let controller2 = Controller::new(root2.control_json(seq.mode));
        run_with(
            &root2,
            &seq,
            &ports,
            &sim_opener_with(Faults::default(), true),
            &controller2,
        )
        .unwrap();
        let slow: Vec<CaptureRow> = read_jsonl(&root2.captured_manifest(seq.mode)).unwrap();
        assert_eq!(slow.len(), 2);
        for r in &slow {
            assert_eq!(r.roundtrip_ms, None, "{}: sequential rows do not", r.key);
        }
        let by_key = |rows: &[CaptureRow]| {
            rows.iter()
                .map(|r| (r.key.clone(), (r.frames, r.sha256_ambe.clone())))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(
            by_key(&fast),
            by_key(&slow),
            "interleaving must not change a single captured byte"
        );
    }

    /// A lost *decode* reply is as fatal as a lost encode one: the
    /// utterance is discarded, the chip re-inited and the work redone.
    #[test]
    fn a_lost_decode_reply_re_inits_and_retries_the_utterance() {
        let (_dir, root) = fixture_root(2);
        let o = opts(VocoderMode::Dstar);
        // The canary's own decode is the first 100 decodes; drop a reply
        // inside the first utterance.
        let faults = Faults {
            drop_decode_response_at: Some(100 + 2),
            ..Faults::default()
        };
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(&root, &o, &ports, &sim_opener(faults), &controller).unwrap();
        assert_eq!(summary.done, 2);
        assert_eq!(summary.failed, 0);
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        let attempts: Vec<u32> = rows.iter().map(|r| r.attempts).collect();
        assert!(attempts.contains(&2), "{attempts:?}");
        assert!(!root.captured_failed(o.mode).exists());
    }

    #[test]
    fn a_timeout_re_inits_and_retries_the_utterance() {
        let (_dir, root) = fixture_root(2);
        let o = opts(VocoderMode::Dstar);
        // Warm-up 20 + canary 100 + canary decode... the first utterance's
        // encode starts after 20 + 100 encodes; drop its 3rd frame's reply.
        let faults = Faults {
            drop_encode_response_at: Some(20 + 100 + 2),
            ..Faults::default()
        };
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(&root, &o, &ports, &sim_opener(faults), &controller).unwrap();
        assert_eq!(summary.done, 2);
        assert_eq!(summary.failed, 0);
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        assert_eq!(rows.len(), 2);
        let attempts: Vec<u32> = rows.iter().map(|r| r.attempts).collect();
        assert!(attempts.contains(&2), "{attempts:?}");
        assert!(!root.captured_failed(o.mode).exists());
    }

    #[test]
    fn rate_lost_three_times_marks_the_utterance_failed_and_continues() {
        let (_dir, root) = fixture_root(2);
        let mut o = opts(VocoderMode::Dstar);
        o.max_attempts = 1;
        let faults = Faults {
            wrong_bits_at: Some(20 + 100 + 1),
            ..Faults::default()
        };
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(&root, &o, &ports, &sim_opener(faults), &controller).unwrap();
        assert_eq!(summary.done, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.state, RunState::Done);
        let failed: Vec<FailedRow> = read_jsonl(&root.captured_failed(o.mode)).unwrap();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].error.contains("rate lost"), "{}", failed[0].error);
        assert_eq!(failed[0].attempts, 1);
        // The failed one is retried on the next run (not in the manifest).
        let p = plan(&root, &o).unwrap();
        assert_eq!(p.rows.len(), 1);
        assert_eq!(p.rows[0].key, failed[0].key);
    }

    #[test]
    fn a_canary_mismatch_aborts_and_leaves_canary_json_alone() {
        let (_dir, root) = fixture_root(6);
        let o = opts(VocoderMode::Dstar);
        // Corrupt every frame after the first utterance has been encoded:
        // the periodic canary (every 2 utterances) then mismatches.
        let faults = Faults {
            corrupt_frames_from: Some(20 + 100 + 100 + 60),
            ..Faults::default()
        };
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let before_rows = 0;
        let err = run_with(&root, &o, &ports, &sim_opener(faults), &controller).unwrap_err();
        assert!(err.to_string().contains("CANARY MISMATCH"), "{err}");
        let rec = canary::load(&root.canary_json(o.mode)).unwrap().unwrap();
        assert_eq!(rec.mode, VocoderMode::Dstar);
        let status = crate::control::read_status(&root.status_json(o.mode)).unwrap();
        assert_eq!(status.state, RunState::Error);
        assert_eq!(status.canary_ok, Some(false));
        assert!(status.error.unwrap().contains("CANARY"));
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        assert!(rows.len() > before_rows && rows.len() < 6, "{}", rows.len());
    }

    #[test]
    fn status_counts_prior_captures_in_done_and_total() {
        let (_dir, root) = fixture_root(4);
        let ports = vec!["sim:0".to_owned()];
        let o = CaptureOptions {
            limit: Some(2),
            ..opts(VocoderMode::Dstar)
        };
        let controller = Controller::new(root.control_json(o.mode));
        let first = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(first.done, 2);
        let o = opts(VocoderMode::Dstar);
        let controller = Controller::new(root.control_json(o.mode));
        let again = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(again.done, 2, "this run's own work");
        let status = crate::control::read_status(&root.status_json(VocoderMode::Dstar)).unwrap();
        assert_eq!(status.done, 4, "prior + this run");
        assert_eq!(status.total, 4, "prior + planned");
    }

    #[test]
    fn paused_time_is_excluded_from_rates() {
        let mut p = Progress {
            done: 10,
            failed: 0,
            frames: 1000,
            current: vec![None],
            paused_workers: 0,
            active_workers: 1,
            canary_ok: None,
            info: None,
            error: None,
            stopped: false,
            paused_since: None,
            paused_total: Duration::ZERO,
        };
        // A pause that ended: two seconds folded into the total.
        p.paused_since = Instant::now().checked_sub(Duration::from_secs(2));
        p.end_pause();
        assert!(p.paused_since.is_none());
        assert!(p.paused_for() >= Duration::from_secs(2));
        // A pause in progress counts too.
        p.paused_workers = 1;
        assert!(p.all_paused());
        p.paused_since = Instant::now().checked_sub(Duration::from_secs(3));
        assert!(p.paused_for() >= Duration::from_secs(5));
        // Only when every active worker holds is the run paused.
        p.active_workers = 2;
        assert!(!p.all_paused());
    }

    /// An opener that fails its first `fail_first` calls, then behaves.
    fn flaky_opener(fail_first: u64) -> Opener {
        let n = Arc::new(AtomicU64::new(0));
        chip_opener(
            Arc::new(move |_: &str| {
                if n.fetch_add(1, Ordering::SeqCst) < fail_first {
                    return Err(DataError::Invalid("simulated reset timeout".to_owned()));
                }
                Ok(Box::new(SimTransport::new()) as Box<dyn Transport>)
            }),
            false,
        )
    }

    #[test]
    fn a_transient_open_failure_is_retried_not_fatal() {
        let (_dir, root) = fixture_root(1);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        // Two misses in a row, then the chip answers: the run must recover
        // and finish rather than die on the first timeout.
        let summary = run_with(&root, &o, &ports, &flaky_opener(2), &controller).unwrap();
        assert_eq!(summary.state, RunState::Done);
        assert_eq!(summary.done, 1);
    }

    #[test]
    fn warm_state_is_seeded_off_by_default_and_off_for_software() {
        let mut o = opts(VocoderMode::Dstar);
        assert_eq!(warm_state_of(&o, "vctk/p1_001"), None, "off by default");
        o.warmup_mix = true;
        // Deterministic in the key.
        assert_eq!(
            warm_state_of(&o, "vctk/p1_001"),
            warm_state_of(&o, "vctk/p1_001")
        );
        // A software mode is stateless: never tagged, even with mixing on.
        let mut sw = opts(VocoderMode::Codec2_3200);
        sw.warmup_mix = true;
        assert_eq!(warm_state_of(&sw, "vctk/p1_001"), None);
        // The share extremes are exact.
        o.cold_share = 1.0;
        assert_eq!(warm_state_of(&o, "vctk/p1_001"), Some(WarmState::Cold));
        o.cold_share = 0.0;
        assert_eq!(warm_state_of(&o, "vctk/p1_001"), Some(WarmState::Warm));
        // And roughly `cold_share` of keys come out cold.
        o.cold_share = 0.34;
        let cold = (0..2000)
            .filter(|i| warm_state_of(&o, &format!("vctk/p{i}_001")) == Some(WarmState::Cold))
            .count();
        assert!(
            (580..=780).contains(&cold),
            "cold={cold} of 2000, want ~680"
        );
    }

    #[test]
    fn warmup_mix_tags_every_chip_row_cold_or_warm() {
        let (_dir, root) = fixture_root(8);
        let mut o = opts(VocoderMode::Dstar);
        o.warmup_mix = true;
        o.cold_share = 0.5;
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        let rows: Vec<CaptureRow> = read_jsonl(&root.capture_dir(o.mode, None).manifest()).unwrap();
        assert!(!rows.is_empty());
        assert!(
            rows.iter()
                .all(|r| matches!(r.warm_state.as_deref(), Some("cold" | "warm"))),
            "every row tagged"
        );
        assert!(rows.iter().any(|r| r.warm_state.as_deref() == Some("cold")));
        assert!(rows.iter().any(|r| r.warm_state.as_deref() == Some("warm")));
    }

    #[test]
    fn stop_and_pause_are_honoured_between_utterances() {
        let (_dir, root) = fixture_root(4);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        controller.request_stop();
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 0);
        assert_eq!(summary.state, RunState::Stopped);
        let status = crate::control::read_status(&root.status_json(o.mode)).unwrap();
        assert_eq!(status.state, RunState::Stopped);

        // Pause via the file once the run is under way (a pause left from
        // before the start is reset, see `reset_control`), resume from
        // another thread, then finish.
        let controller = Controller::new(root.control_json(o.mode));
        write_control(&root.control_json(o.mode), ControlState::Pause).unwrap();
        let ctl = root.control_json(o.mode);
        let stat = root.status_json(o.mode);
        let resumer = std::thread::spawn(move || {
            let t = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(1));
                if matches!(crate::control::read_control(&ctl), Ok(ControlState::Run)) {
                    break;
                }
                assert!(t.elapsed() < Duration::from_secs(10), "never reset");
            }
            write_control(&ctl, ControlState::Pause).unwrap();
            let t = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(50));
                if let Ok(s) = crate::control::read_status(&stat)
                    && s.state == RunState::Paused
                {
                    break;
                }
                assert!(t.elapsed() < Duration::from_secs(10), "never paused");
            }
            write_control(&ctl, ControlState::Run).unwrap();
        });
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        resumer.join().unwrap();
        assert_eq!(summary.done, 4);
        assert_eq!(summary.state, RunState::Done);
    }

    #[test]
    fn status_reports_paused_when_every_active_worker_is_paused() {
        // Two ports, one speaker: the second worker has nothing to do and
        // returns at once; the first pauses. The run *is* paused.
        let (_dir, root) = fixture_root_with(4, 1);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned(), "sim:1".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let ctl = root.control_json(o.mode);
        let stat = root.status_json(o.mode);
        // The harness resets a stale pause to `run` at start; the moment
        // that reset lands, pause again so the first control checkpoint
        // (after init + canary, before any utterance) sees it.
        write_control(&ctl, ControlState::Pause).unwrap();
        let pauser = std::thread::spawn(move || {
            let t = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(1));
                if matches!(crate::control::read_control(&ctl), Ok(ControlState::Run)) {
                    break;
                }
                assert!(t.elapsed() < Duration::from_secs(10), "never reset");
            }
            write_control(&ctl, ControlState::Pause).unwrap();
            let t = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(20));
                if let Ok(s) = crate::control::read_status(&stat)
                    && s.state == RunState::Paused
                {
                    break;
                }
                assert!(
                    t.elapsed() < Duration::from_secs(10),
                    "never reported paused"
                );
            }
            write_control(&ctl, ControlState::Run).unwrap();
        });
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        pauser.join().unwrap();
        assert_eq!(summary.done, 4);
        assert_eq!(summary.state, RunState::Done);
    }

    #[test]
    fn two_ports_split_the_work_by_speaker() {
        let (_dir, root) = fixture_root(6);
        let o = opts(VocoderMode::YsfDmr);
        let ports = vec!["sim:0".to_owned(), "sim:1".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        let summary = run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        assert_eq!(summary.done, 6);
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        let mut by_port: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
        for r in &rows {
            let spk = r.key.split('/').nth(1).unwrap()[..4].to_owned();
            by_port.entry(r.port.clone()).or_default().insert(spk);
        }
        assert_eq!(by_port.len(), 2);
        for spks in by_port.values() {
            assert_eq!(spks.len(), 1, "{by_port:?}");
        }
    }

    /// The random order gives every key a fixed position: adding a corpus
    /// later interleaves its utterances into the remaining order without
    /// moving anything else, so the captured set stays a uniform sample of
    /// everything prepared. `balanced` alternates corpora; `design` is the
    /// old corpus-by-corpus sequence.
    #[test]
    fn capture_orders_are_stable_random_balanced_and_design() {
        let mk = |corpus: &str, key: &str| UtteranceRow {
            key: key.to_owned(),
            corpus: corpus.to_owned(),
            speaker: "s".to_owned(),
            gender: None,
            split: Split::Train,
            duration_s: 1.0,
            src_rate: 16_000,
            src_path: String::new(),
            licence: String::new(),
            rms_dbfs_in: 0.0,
            gain_db: 0.0,
            trim_lead_s: 0.0,
            trim_tail_s: 0.0,
            sha256_16k: String::new(),
            sha256_8k: String::new(),
            prepared_at: String::new(),
            parent: None,
            aug: None,
        };
        let big: Vec<UtteranceRow> = (0..40)
            .map(|i| {
                mk(
                    "libritts_r",
                    &format!("libritts_r/train-clean-100/{i}_1_1_1"),
                )
            })
            .collect();
        let small: Vec<UtteranceRow> = (0..4)
            .map(|i| mk("vctk", &format!("vctk/p22{i}_001_mic2")))
            .collect();
        let opts = |order: CaptureOrder| CaptureOptions {
            order,
            order_seed: 7,
            ..CaptureOptions::new(VocoderMode::Dstar)
        };

        // Random: a permutation, not the manifest order, and stable under
        // insertion of a new corpus.
        let mut before = big.clone();
        sequence(&mut before, &opts(CaptureOrder::Random));
        let keys_before: Vec<&str> = before.iter().map(|r| r.key.as_str()).collect();
        assert_ne!(
            keys_before,
            big.iter().map(|r| r.key.as_str()).collect::<Vec<_>>(),
            "shuffled"
        );
        let mut after: Vec<UtteranceRow> = big.iter().chain(&small).cloned().collect();
        sequence(&mut after, &opts(CaptureOrder::Random));
        let big_only: Vec<&str> = after
            .iter()
            .filter(|r| r.corpus == "libritts_r")
            .map(|r| r.key.as_str())
            .collect();
        assert_eq!(
            big_only, keys_before,
            "the old keys keep their relative order"
        );
        let first_vctk = after.iter().position(|r| r.corpus == "vctk").unwrap();
        assert!(
            first_vctk < 40,
            "the new corpus is interleaved, not appended"
        );
        // Another seed is another permutation.
        let mut other = big.clone();
        sequence(
            &mut other,
            &CaptureOptions {
                order_seed: 8,
                ..opts(CaptureOrder::Random)
            },
        );
        assert_ne!(
            other.iter().map(|r| r.key.as_str()).collect::<Vec<_>>(),
            keys_before
        );

        // Balanced: the two corpora alternate until the small one is out.
        let mut bal: Vec<UtteranceRow> = big.iter().chain(&small).cloned().collect();
        sequence(&mut bal, &opts(CaptureOrder::Balanced));
        let corpora: Vec<&str> = bal.iter().take(8).map(|r| r.corpus.as_str()).collect();
        assert_eq!(
            corpora,
            [
                "libritts_r",
                "vctk",
                "libritts_r",
                "vctk",
                "libritts_r",
                "vctk",
                "libritts_r",
                "vctk"
            ]
        );
        assert!(bal[8..].iter().all(|r| r.corpus == "libritts_r"));

        // Design: corpus rank, then key.
        let mut des: Vec<UtteranceRow> = big.iter().chain(&small).cloned().collect();
        sequence(&mut des, &opts(CaptureOrder::Design));
        assert!(des[..4].iter().all(|r| r.corpus == "vctk"));
        assert!(des[4..].windows(2).all(|w| w[0].key < w[1].key));
        assert_eq!(
            "balanced".parse::<CaptureOrder>().unwrap(),
            CaptureOrder::Balanced
        );
        assert!("shuffled".parse::<CaptureOrder>().is_err());
        assert_eq!(order_hash(1, "a"), order_hash(1, "a"));
        assert_ne!(order_hash(1, "a"), order_hash(2, "a"));
    }

    /// `--only-twins` plans the rows with a parent and nothing else, so a
    /// chip can be pointed at freshly prepared twins without first working
    /// through the base corpus around them.
    #[test]
    fn only_twins_plans_the_rows_with_a_parent() {
        let (_dir, root) = fixture_root(4);
        // Give two of the four rows a parent.
        let mut rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        let parent_key = rows[0].key.clone();
        for r in rows.iter_mut().skip(2) {
            r.parent = Some(parent_key.clone());
        }
        crate::util::write_jsonl(&root.prepared_manifest(), &rows).unwrap();

        let mut opts = CaptureOptions::new(VocoderMode::Dstar);
        assert_eq!(plan(&root, &opts).unwrap().rows.len(), 4);
        opts.only_twins = true;
        let p = plan(&root, &opts).unwrap();
        assert_eq!(p.rows.len(), 2);
        assert!(p.rows.iter().all(|r| r.parent.is_some()));
    }

    #[test]
    fn plan_orders_by_design_and_assignment_is_by_speaker() {
        let mk = |corpus: &str, spk: &str, split: Split, key: &str, dur: f64| UtteranceRow {
            key: key.to_owned(),
            corpus: corpus.to_owned(),
            speaker: spk.to_owned(),
            gender: None,
            split,
            duration_s: dur,
            src_rate: 16_000,
            src_path: String::new(),
            licence: String::new(),
            rms_dbfs_in: 0.0,
            gain_db: 0.0,
            trim_lead_s: 0.0,
            trim_tail_s: 0.0,
            sha256_16k: String::new(),
            sha256_8k: String::new(),
            prepared_at: String::new(),
            parent: None,
            aug: None,
        };
        let rows = vec![
            mk(
                "libritts_r",
                "84",
                Split::Train,
                "libritts_r/train-clean-100/84_1_1_1",
                3.0,
            ),
            mk("ljspeech", "LJ", Split::Train, "ljspeech/LJ001-0001", 5.0),
            mk("vctk", "p225", Split::Train, "vctk/p225_001_mic2", 2.0),
            mk(
                "libritts_r",
                "1089",
                Split::Test,
                "libritts_r/test-clean/1089_1_1_1",
                4.0,
            ),
            mk(
                "voicebank_demand",
                "p232",
                Split::Test,
                "voicebank_demand/test/p232_001",
                1.0,
            ),
            mk("vctk", "p225", Split::Train, "vctk/p225_002_mic2", 2.0),
        ];
        let ranks: Vec<usize> = rows.iter().map(design_rank).collect();
        assert_eq!(ranks, vec![4, 3, 2, 1, 0, 2]);
        let a = assign_by_speaker(&rows, 2);
        assert_eq!(a[0].len() + a[1].len(), 6);
        // p225's two utterances land together.
        let w = a.iter().position(|q| q.contains(&2)).unwrap();
        assert!(a[w].contains(&5));
        // Balanced-ish: the 5 s LJ speaker and the 4 s p225 are apart.
        let w_lj = a.iter().position(|q| q.contains(&1)).unwrap();
        assert_ne!(w, w_lj);
        assert_eq!(assign_by_speaker(&rows, 0).len(), 1);

        // The software assignment ignores speakers: every utterance is
        // placed on its own, largest first, and the totals stay close.
        let b = assign_balanced(&rows, 3);
        assert_eq!(b.iter().map(Vec::len).sum::<usize>(), 6);
        let total = |q: &Vec<usize>| q.iter().map(|&i| rows[i].duration_s).sum::<f64>();
        let (lo, hi) = b.iter().fold((f64::MAX, 0.0f64), |(lo, hi), q| {
            (lo.min(total(q)), hi.max(total(q)))
        });
        assert!(hi - lo <= 5.0, "{b:?}");
        for q in &b {
            assert!(q.windows(2).all(|w| w[0] < w[1]), "plan order kept: {q:?}");
        }
        assert_eq!(assign_balanced(&rows, 0).len(), 1);
        assert_eq!(
            software_workers(VocoderMode::Codec2_3200, 3),
            ["codec2:0", "codec2:1", "codec2:2"]
        );
    }

    #[test]
    fn workers_for_refuses_the_wrong_knob_for_the_family() {
        let mut o = opts(VocoderMode::Codec2_3200);
        o.ports = vec!["/dev/cu.usbserial-X".to_owned()];
        let err = workers_for(&o).err().expect("refused").to_string();
        assert!(err.contains("--port is not used"), "{err}");
        let mut o = opts(VocoderMode::Dstar);
        o.jobs = Some(4);
        let err = workers_for(&o).err().expect("refused").to_string();
        assert!(err.contains("--jobs applies"), "{err}");
        // A dry chip run gets one sim; a software run gets its threads.
        let (w, _) = workers_for(&opts(VocoderMode::YsfDmr)).unwrap();
        assert_eq!(w, ["sim:0"]);
        let mut o = opts(VocoderMode::Codec2_1600);
        o.jobs = Some(2);
        #[cfg(feature = "codec2")]
        {
            let (w, open) = workers_for(&o).unwrap();
            assert_eq!(w, ["codec2:0", "codec2:1"]);
            let v = open("codec2:1", VocoderMode::Codec2_1600).unwrap();
            assert_eq!(v.info().prodid, "codec2");
        }
        o.jobs = None;
        let (w, _) = workers_for(&o).unwrap();
        assert_eq!(w.len(), num_cpus::get_physical().max(1));
    }

    #[cfg(feature = "codec2")]
    #[test]
    fn software_capture_runs_without_a_port_and_verifies() {
        for (mode, fs) in [
            (VocoderMode::Codec2_3200, 160usize),
            (VocoderMode::Codec2_1600, 320usize),
        ] {
            let (_dir, root) = fixture_root_with(6, 3);
            let mut o = opts(mode);
            o.jobs = Some(3);
            let (workers, opener) = workers_for(&o).unwrap();
            let controller = Controller::new(root.control_json(o.mode));
            let summary = run_with(&root, &o, &workers, &opener, &controller).unwrap();
            assert_eq!(summary.done, 6, "{mode}");
            assert_eq!(summary.failed, 0);
            assert_eq!(summary.state, RunState::Done);
            let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
            assert_eq!(rows.len(), 6);
            let mut ports: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for r in &rows {
                assert_eq!(r.mode, mode);
                assert_eq!(r.prodid, "codec2");
                assert_eq!(r.version, crate::vocoder::CODEC2_CRATE_VERSION);
                assert!(r.port.starts_with("codec2:"), "{}", r.port);
                ports.insert(&r.port);
                let ambe = root.captured_ambe(mode, &r.key);
                assert_eq!(
                    std::fs::metadata(&ambe).unwrap().len(),
                    u64::from(r.frames) * 8
                );
                let a = unamblify_audio::read(root.captured_decoded(mode, &r.key)).unwrap();
                assert_eq!(a.samples.len(), r.frames as usize * fs, "{mode}");
                // 1 s + i AMBE frames of input → whole frames of the mode.
                let samples = 8_000 + 160 * r.key[10..13].parse::<usize>().unwrap();
                assert_eq!(r.frames as usize, samples.div_ceil(fs), "{mode} {}", r.key);
            }
            assert!(
                ports.len() > 1,
                "work was spread over the workers: {ports:?}"
            );
            let status = crate::control::read_status(&root.status_json(mode)).unwrap();
            assert_eq!(status.ports, ["codec2:0", "codec2:1", "codec2:2"]);
            assert_eq!(status.prodid.as_deref(), Some("codec2"));
            assert_eq!(status.canary_ok, Some(true));
            let rec = canary::load(&root.canary_json(mode)).unwrap().unwrap();
            assert_eq!(rec.prodid, "codec2");
            assert_eq!(rec.mode, mode);
            assert!(
                rec.lag_samples.unsigned_abs() as usize <= fs,
                "{}",
                rec.lag_samples
            );
            let v = crate::verify::run(
                &root,
                &crate::verify::VerifyOptions {
                    mode,
                    kind: None,
                    sample: 0,
                    seed: 1,
                },
            )
            .unwrap();
            assert!(v.problems.is_empty(), "{:?}", v.problems);
            assert_eq!(v.ok, 6);
            // The encoder is deterministic: a re-capture of a vanished
            // file gives the same channel frames, whichever worker takes
            // it. The decoded audio is one draw of the decoder's random
            // unvoiced phases (see `vocoder`), so its hash is not compared.
            let victim = rows[0].clone();
            std::fs::remove_file(root.captured_decoded(mode, &victim.key)).unwrap();
            let again = run_with(&root, &o, &workers, &opener, &controller).unwrap();
            assert_eq!(again.done, 1);
            let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
            let redo = rows.iter().find(|r| r.key == victim.key).unwrap();
            assert_eq!(redo.sha256_ambe, victim.sha256_ambe);
            assert_eq!(redo.frames, victim.frames);
            let s = stats(&root, mode).unwrap();
            assert_eq!(s.captured, 6);
            let frames: u64 = rows.iter().map(|r| u64::from(r.frames)).sum();
            assert!((s.hours * 3_600.0 - frames as f64 * fs as f64 / 8_000.0).abs() < 1e-6);
        }
    }

    /// The sim decodes unknown frames (the mute word) to silence and
    /// delays by 42 samples, so a dropped frame reads as a zero window in
    /// the sibling's audio.
    #[test]
    fn augment_drops_reuses_the_harness_on_the_sim_and_is_idempotent() {
        use crate::augment::AugmentSpec;
        use unamblify::aug::{AugKind, Subst};
        use unamblify::channel::NULL_AMBE_FRAME;

        let (_dir, root) = fixture_root(5);
        let o = opts(VocoderMode::Dstar);
        let ports = vec!["sim:0".to_owned()];
        let controller = Controller::new(root.control_json(o.mode));
        // No base capture yet: refused.
        let mut a = opts(VocoderMode::Dstar);
        a.stage = Stage::Augment(AugmentSpec {
            rate: 0.1,
            share: 1.0,
            ..AugmentSpec::new(AugKind::Drops)
        });
        let sib = a.out_dir(&root);
        assert_eq!(sib.name(), "dstar+drops");
        let err = run_with(
            &root,
            &a,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no base capture"), "{err}");

        run_with(
            &root,
            &o,
            &ports,
            &sim_opener(Faults::default()),
            &controller,
        )
        .unwrap();
        // The sibling's own controller and lock; the sim is deterministic,
        // so the same frames decode the same on a fresh sim.
        let sib_controller = Controller::new(sib.control_json());
        let s = run_with(
            &root,
            &a,
            &ports,
            &sim_opener(Faults::default()),
            &sib_controller,
        )
        .unwrap();
        assert_eq!(s.done, 5, "{s:?}");
        assert_eq!(s.state, RunState::Done);
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(o.mode)).unwrap();
        let rows: Vec<CaptureRow> = read_jsonl(&sib.manifest()).unwrap();
        assert_eq!(rows.len(), 5);
        assert!(sib.canary_json().is_file(), "the base canary is copied");
        assert_eq!(
            canary::load(&sib.canary_json())
                .unwrap()
                .unwrap()
                .lag_samples,
            42
        );
        let status = crate::control::read_status(&sib.status_json()).unwrap();
        assert_eq!(status.kind.as_deref(), Some("drops"));
        assert_eq!(status.mode, VocoderMode::Dstar);
        assert_eq!(status.canary_ok, Some(true));
        let mut dropped_total = 0usize;
        for r in &rows {
            let b = base.iter().find(|b| b.key == r.key).unwrap();
            assert_eq!(r.frames, b.frames, "time is never shifted");
            assert_eq!(r.encode_ms, 0);
            let aug = r.aug.as_ref().expect("augment record");
            assert_eq!(aug.kind, AugKind::Drops);
            assert_eq!(aug.subst, Some(Subst::Mute));
            assert_eq!(aug.burst, Some((1, 3)));
            assert!(aug.positions.windows(2).all(|w| w[0] < w[1]));
            dropped_total += aug.positions.len();
            let orig = std::fs::read(root.captured_ambe(o.mode, &r.key)).unwrap();
            let mutated = std::fs::read(sib.ambe(&r.key)).unwrap();
            assert_eq!(orig.len(), mutated.len());
            assert_eq!(sha256_hex(&mutated), r.sha256_ambe);
            let wav = unamblify_audio::read(sib.decoded(&r.key)).unwrap();
            assert_eq!(wav.samples.len(), r.frames as usize * 160);
            for f in 0..r.frames {
                let fr = &mutated[f as usize * 9..(f as usize + 1) * 9];
                if aug.positions.contains(&f) {
                    assert_eq!(fr, &NULL_AMBE_FRAME[..]);
                    // Decoded as silence, 42 samples late.
                    let at = f as usize * 160 + 42;
                    let win = &wav.samples[at..(at + 100).min(wav.samples.len())];
                    assert!(win.iter().all(|&v| v == 0.0), "frame {f} not muted");
                } else {
                    assert_eq!(fr, &orig[f as usize * 9..(f as usize + 1) * 9]);
                }
            }
        }
        assert!(dropped_total > 0);
        // Idempotent: nothing left to do; the base is untouched.
        let again = run_with(
            &root,
            &a,
            &ports,
            &sim_opener(Faults::default()),
            &sib_controller,
        )
        .unwrap();
        assert_eq!(again.done, 0);
        assert_eq!(again.skipped, 5);
        assert_eq!(
            read_jsonl::<CaptureRow>(&root.captured_manifest(o.mode))
                .unwrap()
                .len(),
            5
        );
        assert_eq!(stats_in(&root, &sib).unwrap().kind, Some(AugKind::Drops));
        assert_eq!(stats_in(&root, &sib).unwrap().captured, 5);
        // The share is seeded: a third of the base at share 0.4, the same
        // keys every time; --limit caps it.
        let mut half = a.clone();
        half.stage = Stage::Augment(AugmentSpec {
            share: 0.4,
            seed: 7,
            ..AugmentSpec::new(AugKind::Ber)
        });
        half.limit = Some(1);
        let p1 = plan(&root, &half).unwrap();
        assert_eq!(p1.rows.len(), 1);
        half.limit = None;
        let p2 = plan(&root, &half).unwrap();
        assert!(
            !p2.rows.is_empty() && p2.rows.len() < 5,
            "{}",
            p2.rows.len()
        );
        assert_eq!(
            plan(&root, &half)
                .unwrap()
                .rows
                .iter()
                .map(|r| r.key.clone())
                .collect::<Vec<_>>(),
            p2.rows.iter().map(|r| r.key.clone()).collect::<Vec<_>>()
        );
        // The chip-only substitution is refused before anything is touched.
        let mut erase = a.clone();
        erase.stage = Stage::Augment(AugmentSpec {
            subst: Subst::Erase,
            ..AugmentSpec::new(AugKind::Drops)
        });
        let err = run_with(
            &root,
            &erase,
            &ports,
            &sim_opener(Faults::default()),
            &sib_controller,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("chip-only"), "{err}");
    }

    #[cfg(feature = "codec2")]
    #[test]
    fn augment_drops_and_ber_run_through_the_codec2_vocoder() {
        use crate::augment::{AugmentSpec, mute_frame};
        use unamblify::aug::{AugKind, Subst};

        let mode = VocoderMode::Codec2_3200;
        let (_dir, root) = fixture_root_with(6, 3);
        let mut o = opts(mode);
        o.jobs = Some(2);
        let (workers, opener) = workers_for(&o).unwrap();
        let controller = Controller::new(root.control_json(mode));
        run_with(&root, &o, &workers, &opener, &controller).unwrap();
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let mute = mute_frame(mode).unwrap();

        for (kind, subst) in [
            (AugKind::Drops, Subst::Mute),
            (AugKind::Drops, Subst::Repeat),
            (AugKind::Ber, Subst::Mute),
        ] {
            let mut a = opts(mode);
            a.jobs = Some(2);
            a.stage = Stage::Augment(AugmentSpec {
                rate: if kind == AugKind::Drops { 0.1 } else { 0.01 },
                subst,
                share: 1.0,
                seed: 3,
                ..AugmentSpec::new(kind)
            });
            let sib = a.out_dir(&root);
            // A rerun under the same kind directory redoes nothing, so
            // start each variant from a clean sibling.
            let _ = std::fs::remove_dir_all(sib.dir());
            let (workers, opener) = workers_for(&a).unwrap();
            let s = run_with(
                &root,
                &a,
                &workers,
                &opener,
                &Controller::new(sib.control_json()),
            )
            .unwrap();
            assert_eq!(s.done, 6, "{kind} {subst}: {s:?}");
            let rows: Vec<CaptureRow> = read_jsonl(&sib.manifest()).unwrap();
            assert_eq!(rows.len(), 6);
            let v = crate::verify::run(
                &root,
                &crate::verify::VerifyOptions {
                    mode,
                    kind: Some(kind),
                    sample: 0,
                    seed: 1,
                },
            )
            .unwrap();
            assert!(v.problems.is_empty(), "{kind}: {:?}", v.problems);
            for r in &rows {
                let b = base.iter().find(|b| b.key == r.key).unwrap();
                assert_eq!(r.frames, b.frames);
                assert_eq!(r.prodid, "codec2");
                let aug = r.aug.as_ref().unwrap();
                assert_eq!(aug.kind, kind);
                let orig = std::fs::read(root.captured_ambe(mode, &r.key)).unwrap();
                let mutated = std::fs::read(sib.ambe(&r.key)).unwrap();
                assert_eq!(orig.len(), mutated.len());
                let wav = unamblify_audio::read(sib.decoded(&r.key)).unwrap();
                assert_eq!(wav.samples.len(), r.frames as usize * 160);
                for f in 0..r.frames as usize {
                    let fr = &mutated[f * 8..(f + 1) * 8];
                    let was = &orig[f * 8..(f + 1) * 8];
                    if aug.positions.contains(&(f as u32)) {
                        match (kind, subst) {
                            (AugKind::Drops, Subst::Mute) => assert_eq!(fr, &mute[..]),
                            (AugKind::Drops, _) => assert_eq!(
                                fr,
                                if f == 0 {
                                    &mute[..]
                                } else {
                                    &mutated[(f - 1) * 8..f * 8]
                                }
                            ),
                            (AugKind::Ber, _) => assert_ne!(fr, was),
                            // This test drives the decode-only kinds; a
                            // re-encode never reaches the mutation path
                            // (see augment::mutate's own refusal).
                            (AugKind::Perens, _) => {
                                unreachable!("{kind} is produced by the recode stage")
                            }
                        }
                    } else {
                        assert_eq!(fr, was, "{kind}: frame {f} untouched");
                    }
                }
                if kind == AugKind::Drops && subst == Subst::Mute && !aug.positions.is_empty() {
                    // A muted frame decodes quieter than the speech around it.
                    let lag = canary::load(&sib.canary_json())
                        .unwrap()
                        .unwrap()
                        .lag_samples
                        .max(0) as usize;
                    let rms =
                        |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
                    let speech = rms(&wav.samples);
                    let muted: Vec<f32> = aug
                        .positions
                        .iter()
                        .flat_map(|&f| {
                            let at = f as usize * 160 + lag + 40;
                            wav.samples[at.min(wav.samples.len())..(at + 80).min(wav.samples.len())]
                                .to_vec()
                        })
                        .collect();
                    assert!(
                        rms(&muted) < speech * 0.5,
                        "muted {} vs speech {speech}",
                        rms(&muted)
                    );
                }
            }
        }
    }
}
