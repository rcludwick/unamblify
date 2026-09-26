// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The `unamblify` binary (spec §9): one front end over the data, train and
//! web crates. `serve` re-execs this same binary for `train` and `capture`,
//! so the dashboard's supervisor needs nothing but the path it was started
//! from.
//!
//! `train`, `infer` and `smoke` exist only with the `train` cargo feature
//! (default on); `--no-default-features` gives a libtorch-free build with
//! every other verb.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use unamblify::aug::{AugKind, Fec, Subst, parse_kind_word};
use unamblify::{NoiseSet, RunState, Split, VocoderMode};
use unamblify_data::DataRoot;
use unamblify_data::augment::{AugmentSpec, parse_burst};
use unamblify_data::capture::{self, CaptureOptions, CaptureOrder, Stage};
use unamblify_data::control::{ControlState, read_status, write_control};
use unamblify_data::prepare::{self, PrepareOptions};
use unamblify_data::shard::{self, KindSel, ShardOptions};
use unamblify_data::twins::TwinOptions;
use unamblify_data::verify::{self, VerifyOptions};
use unamblify_web::runs;

#[derive(Parser)]
#[command(
    name = "unamblify",
    version,
    about = "Digital-voice post-filter harness: corpus preparation, AMBE (ThumbDV) and Codec 2 (M17, software) capture, sharding, training and the dashboard",
    propagate_version = true
)]
struct Cli {
    #[arg(
        long,
        global = true,
        help = "Data root; overrides $UNAMBLIFY_DATA (default /Volumes/data/training_data/unamblify)"
    )]
    data_root: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Walk raw/ into prepared/ (16 kHz + 8 kHz WAVs and the manifest),
    /// then emit augmented twins for the shares asked for.
    Prepare {
        /// Corpus to walk, in order (repeatable; default: all four).
        #[arg(long = "corpus")]
        corpora: Vec<String>,
        /// Re-do utterances already in the manifest (and their twins).
        #[arg(long)]
        force: bool,
        /// Worker threads (default: all cores).
        #[arg(long)]
        jobs: Option<usize>,
        /// Share of utterances that get a noisy twin `<key>+n<NNNN>`: the
        /// clean signal with a tier-3 noise clip mixed at 0–20 dB SNR.
        #[arg(long, default_value_t = 0.0)]
        noise_share: f32,
        /// Noise sets to draw from, comma-separated (default: demand,musan;
        /// each must be fetched under raw/).
        #[arg(long, value_delimiter = ',')]
        noise_sets: Vec<NoiseSet>,
        /// Share of utterances that get an overdriven-mic twin
        /// `<key>+h<NNNN>` (clipping, proximity, pops, AGC, mic response).
        #[arg(long, default_value_t = 0.0)]
        ham_chain_share: f32,
        /// Share of utterances that get an underdriven twin
        /// `<key>+u<NNNN>`: 15–30 dB below the working level and, half
        /// the time, the proximity bass a distant talker loses. Never on
        /// an utterance that got an overdriven twin.
        #[arg(long, default_value_t = 0.0)]
        underdrive_share: f32,
        /// Seed of the per-key twin decisions and parameter draws.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Only the twin pass, over the manifest as it stands: skips the
        /// walk of raw/ and the check of every prepared file, which
        /// change nothing once the corpus is prepared. Needs a share.
        #[arg(long)]
        twins_only: bool,
        /// Only re-apply the split rule to `prepared/manifest.jsonl` (no
        /// audio is touched); rebuild any shard set afterwards.
        #[arg(long)]
        resplit: bool,
    },
    /// Run an encode → decode capture (ThumbDV for the AMBE modes, the
    /// codec2 crate in software for codec2-3200 / codec2-1600), or control
    /// a running one.
    Capture(CaptureArgs),
    /// Decode-only augmentation: mutate a base capture's channel frames
    /// (dropped frames or bit errors), re-run only the decode pass, and
    /// write the sibling capture captured/<mode>+<kind>/.
    Augment(AugmentArgs),
    /// Re-encode the prepared audio with a second, independent
    /// implementation of the mode's codec and write the sibling capture
    /// captured/<mode>+<kind>/. Runs in software on --jobs threads; no
    /// ThumbDV, and the base capture is left alone.
    Recode(RecodeArgs),
    /// Consonant metrics over a directory of rendered clips: what the
    /// trainer's `eval/plosive_*` columns measure, plus the sibilant level
    /// and balance, standalone. Clips are `<stem>@<mode>.<kind>.wav` (or
    /// .flac) as eval and the spike scripts write them; every kind is
    /// scored against that stem's `clean`.
    Measure {
        /// Directory of clips.
        #[arg(long)]
        dir: PathBuf,
        /// Kinds to score against clean (comma-separated).
        #[arg(long, default_value = "degraded,out", value_delimiter = ',')]
        kinds: Vec<String>,
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Recompute hashes and frame counts of a capture.
    Verify {
        #[arg(long)]
        mode: VocoderMode,
        /// A decode-only sibling (drops | ber) instead of the base capture.
        #[arg(long)]
        kind: Option<AugKind>,
        /// Rows to check (0 = all).
        #[arg(long, default_value_t = 0)]
        sample: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Pack fixed-length training examples into shards/<name>/, from one
    /// mode (--mode) or several at once (--modes a,b: one set holding
    /// every mode, each example tagged with its mode, drawn equally per
    /// split unless --no-balance).
    Shard {
        /// One mode (the single-mode alias of --modes).
        #[arg(long, conflicts_with = "modes", required_unless_present = "modes")]
        mode: Option<VocoderMode>,
        /// Modes to pack together, comma-separated, in index order.
        #[arg(long, value_delimiter = ',')]
        modes: Vec<VocoderMode>,
        /// Shard set name (shards/<name>/).
        #[arg(long)]
        name: String,
        #[arg(long, default_value_t = 2.0)]
        crop_s: f32,
        #[arg(long, default_value_t = 0.34)]
        onset_share: f32,
        #[arg(long, default_value_t = 0.15)]
        tail_share: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Examples per NNNN.bin.
        #[arg(long, default_value_t = 1024)]
        examples_per_file: usize,
        /// Only these splits (repeatable; default: all).
        #[arg(long = "split")]
        splits: Vec<Split>,
        /// Capture sets to draw from, comma-separated: base, a sibling
        /// (drops, ber, perens), or a sibling scoped to one mode
        /// (dstar+perens) for a kind that exists for that mode only.
        /// Default: base.
        #[arg(long, value_delimiter = ',', default_value = "base")]
        kinds: Vec<String>,
        /// With several modes, draw the same number of examples from each
        /// per split (the smallest mode sets the cap; default on).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        balance: bool,
        /// Keep every example of every mode (no cap).
        #[arg(long, conflicts_with = "balance")]
        no_balance: bool,
        /// At most this many utterances per capture set (a seeded draw):
        /// a bounded trial set rather than the whole capture.
        #[arg(long)]
        max_utterances: Option<usize>,
        /// With --max-utterances: reserve this fraction of each capture
        /// set's draw for augmented twins (noisy, overdriven,
        /// underdriven), instead of taking them in proportion to the
        /// capture — where they are a sliver of a 2 M-utterance set.
        #[arg(long, requires = "max_utterances")]
        twin_share: Option<f32>,
        /// Draw a drops row only if `augment` generated it at this
        /// frame-loss rate or above. One drops sibling can hold a light
        /// pass and a heavy one; this picks the heavy one for a set that
        /// trains frame restoration. Other kinds are untouched.
        #[arg(long)]
        min_drop_rate: Option<f32>,
        /// Draw only these corpora (comma-separated or repeated), e.g.
        /// `libritts_r,vctk,ljspeech,voicebank_demand` for a set whose
        /// targets are studio recordings. Default: every corpus.
        #[arg(long = "corpora", value_delimiter = ',')]
        corpora: Vec<String>,
    },
    /// Train a run from a config (needs the `train` feature).
    #[cfg(feature = "train")]
    Train {
        /// Run config TOML.
        #[arg(long)]
        config: PathBuf,
        /// Run directory (default: <data-root>/runs/<YYYYMMDD-HHMMSS-name>).
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Checkpoint directory to resume from.
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Override [train] device (cpu | mps | cuda:N | rocm:N).
        #[arg(long)]
        device: Option<String>,
        /// Override [train] steps.
        #[arg(long)]
        steps: Option<u64>,
        /// Override [data] shards (another shard set name).
        #[arg(long)]
        shards: Option<String>,
        /// Start from another run's weights (`[train] init_from`): a run
        /// id, <run-id>:<step>, or a checkpoints/step-N path. Step 0,
        /// fresh optimiser, only the weights are inherited.
        #[arg(long)]
        init_from: Option<String>,
    },
    /// Render one prepared utterance through one checkpoint (needs the
    /// `train` feature). What the dashboard's Samples page spawns.
    #[cfg(feature = "train")]
    Infer {
        /// Run directory (holds config.toml and checkpoints/).
        #[arg(long)]
        run_dir: PathBuf,
        /// Checkpoint step (checkpoints/step-NNNNNN).
        #[arg(long)]
        step: u64,
        /// Utterance key (`vctk/p225_001_mic2`), captured for the mode.
        #[arg(long)]
        key: String,
        /// Which of the run's [data] modes to render (default: the first).
        #[arg(long)]
        mode: Option<VocoderMode>,
        /// Output WAV (default: <run-dir>/samples/step-NNNNNN/<key>.out.wav).
        /// Its spec.json is written beside it.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// The training dashboard (spawns `unamblify train` / `capture` itself).
    Serve {
        /// Runs directory (default: <data-root>/runs).
        #[arg(long)]
        runs_dir: Option<PathBuf>,
        /// Where run configs live (default: ./configs).
        #[arg(long)]
        configs_dir: Option<PathBuf>,
        /// Bind address (default 127.0.0.1:8787; 0.0.0.0:8787 with --token).
        #[arg(long)]
        bind: Option<std::net::SocketAddr>,
        /// Bearer token for /api/*; required for a non-loopback bind.
        #[arg(long)]
        token: Option<String>,
        /// Seconds between SIGTERM and SIGKILL when stopping a run.
        #[arg(long, default_value_t = 30)]
        grace_s: u64,
    },
    /// List, inspect, stop or resume training runs.
    Runs {
        /// Runs directory (default: <data-root>/runs).
        #[arg(long, global = true)]
        runs_dir: Option<PathBuf>,
        #[command(subcommand)]
        cmd: RunsCmd,
    },
    /// Manifest summaries: prepared/ and every captured/<mode>/.
    Stats {
        /// One mode only.
        #[arg(long)]
        mode: Option<VocoderMode>,
        /// Print as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Rename what an older layout left under a retired mode name
    /// (captured/ysf-dn/ → captured/ysf-dmr/, its siblings, the `mode`
    /// fields of its manifests / status / canary, and shards/*/index.json).
    /// Idempotent; refuses a set a running capture holds.
    MigrateModes {
        /// Print what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Time one model on the CPU against the 20 ms real-time budget —
    /// the gate a candidate passes before a long training run, not
    /// after. Trains nothing and touches no data.
    #[cfg(feature = "train")]
    Bench {
        /// What to time: filter (candidate 1, built from --profile) or
        /// pipeline (the restorer + waveform synthesiser, random weights
        /// of the exported shapes, or --weights).
        #[arg(long, default_value = "filter")]
        kind: String,
        /// Pipeline weights file (.safetensors with its .json beside it);
        /// without it the pipeline bench uses random weights, which cost
        /// the same.
        #[arg(long)]
        weights: Option<PathBuf>,
        /// Size profile: full | lite | super-lite.
        #[arg(long, default_value = "full")]
        profile: String,
        /// [model] width multiplier on the profile's channel widths.
        #[arg(long, default_value_t = 1.0)]
        width: f32,
        /// Lookahead in AMBE 20 ms frames.
        #[arg(long, default_value_t = 5)]
        lookahead: u32,
        /// Size the mode embedding for this many modes (0 = blind).
        #[arg(long, default_value_t = 4)]
        modes: usize,
        /// Seconds of audio per timed pass.
        #[arg(long, default_value_t = 8.0)]
        seconds: f32,
        /// Timed passes; the median is reported.
        #[arg(long, default_value_t = 5)]
        passes: usize,
        /// CPU threads. One core is the budget that matters.
        #[arg(long, default_value_t = 1)]
        threads: i32,
        /// Include the aperiodic excitation path (`[model] noise_head`).
        #[arg(long)]
        noise_head: bool,
        /// … with its modulation and 1 ms gains (`[model] noise_mod`).
        #[arg(long)]
        noise_mod: bool,
    },
    /// One clip through the restorer + waveform synthesiser: 8 kHz codec
    /// output in, 24 kHz speech out (needs the `train` feature).
    #[cfg(feature = "train")]
    Restore {
        /// Pipeline weights (.safetensors, its .json manifest beside it).
        #[arg(long)]
        weights: PathBuf,
        /// Input WAV or FLAC at 8 kHz: the decoded codec output.
        #[arg(long = "in")]
        input: PathBuf,
        /// Which codec it came from: dstar | ysf-dmr | codec2-3200 | codec2-1600.
        #[arg(long)]
        mode: VocoderMode,
        /// Output WAV, 24 kHz.
        #[arg(long)]
        out: PathBuf,
    },
    /// 20-step CPU train on synthetic data; fails unless the loss went down.
    #[cfg(feature = "train")]
    Smoke {
        /// Keep the run here instead of a temp dir.
        #[arg(long)]
        run_dir: Option<PathBuf>,
    },
}

/// `capture --mode M …` runs a capture; `capture pause|resume|stop|status
/// --mode M` talks to a running one through control.json / status.json.
#[derive(Args)]
#[command(
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true,
    after_help = "Modes: dstar | ysf-dmr (one capture serves YSF DN and DMR: same voice bits) go through the ThumbDV (one worker per --port, \
        every scanned stick by default). codec2-3200 (M17 voice, 20 ms frames) and \
        codec2-1600 (M17 voice + data, 40 ms frames) run in software: no --port, --jobs N \
        encoder threads (default: the physical cores), the same canary, control.json, \
        status.json and manifest as a chip capture. Every mode's channel frames land in \
        captured/<mode>/<key>.ambe (the extension means \"channel frames\"; the mode says \
        which codec)."
)]
// Independent CLI flags, not a state machine.
#[allow(clippy::struct_excessive_bools)]
struct CaptureArgs {
    #[command(subcommand)]
    control: Option<CaptureCmd>,
    /// dstar | ysf-dmr | codec2-3200 | codec2-1600
    #[arg(long, required = true)]
    mode: Option<VocoderMode>,
    /// Serial port (repeatable). Must be in the ThumbDV scan (FTDI 0403:6015).
    /// Chip modes only; refused for codec2-*.
    #[arg(long = "port")]
    ports: Vec<String>,
    /// Encoder threads for the software modes (default: physical cores).
    /// Refused for the chip modes, whose workers are their ports.
    #[arg(long)]
    jobs: Option<usize>,
    /// Only these corpora (repeatable; with --order design, in this order).
    #[arg(long = "corpus")]
    corpora: Vec<String>,
    /// Sequence of the pending utterances: random (every key at a fixed
    /// hashed position, so the captured set is always a uniform sample of
    /// everything prepared and a corpus added later just interleaves),
    /// balanced (round-robin over corpora), or design (corpus by corpus).
    #[arg(long, default_value = "random")]
    order: CaptureOrder,
    /// Seed behind --order random / balanced.
    #[arg(long, default_value_t = 1)]
    order_seed: u64,
    /// Only this split.
    #[arg(long)]
    split: Option<Split>,
    /// Only the augmented twins (rows with a parent): noisy, overdriven
    /// and underdriven. Without it fresh twins surface only at their
    /// hashed positions among the whole base corpus.
    #[arg(long)]
    only_twins: bool,
    /// At most this many utterances this run.
    #[arg(long)]
    limit: Option<usize>,
    /// Re-check the canary every N utterances per stick.
    #[arg(long, default_value_t = 200)]
    canary_every: usize,
    /// Use the simulated chip instead of a serial port (chip modes; a
    /// software mode runs its real encoder either way).
    #[arg(long)]
    dry_run: bool,
    /// Run the chip's encode pass and decode pass one after the other
    /// instead of interleaving them (1.73x slower; the escape hatch for a
    /// stick that misbehaves with both directions in flight). Rows then
    /// time the two passes separately instead of the round trip. No
    /// effect on a software mode.
    #[arg(long)]
    sequential: bool,
    /// Mix the AMBE encoder's warm-up state across utterances: each is
    /// captured `cold` (chip reset first, so the first frames carry the
    /// keyup pitch-lock transient) or `warm` (encoder locked onto the
    /// voice first), chosen from a hash of its key. No effect on a
    /// software mode. AMBE chip modes only.
    #[arg(long)]
    warmup_mix: bool,
    /// Fraction captured `cold` when --warmup-mix is on (the rest `warm`).
    #[arg(long, default_value_t = 0.34)]
    cold_share: f64,
    /// Seed behind the per-utterance cold/warm choice.
    #[arg(long, default_value_t = 1)]
    warmup_seed: u64,
    /// Print the captured manifest summary instead of capturing.
    #[arg(long)]
    stats: bool,
}

/// `augment --mode M --kind K …`: the capture harness in its decode-only
/// stage. Pause / stop through `captured/<mode>+<kind>/control.json`
/// (SIGINT stops after the utterance in flight); the lock, canary check
/// and status.json live in that directory too.
#[derive(Args)]
#[command(
    after_help = "drops: bursts of --burst consecutive frames (default 1..3) at --rate of all \
        frames (default 0.02), each replaced by --subst mute (the mode's mute / null codeword; \
        Codec 2's is digital silence encoded once with the crate), repeat (hold the last good \
        frame), or erase (hand the frame to the receiver's own concealment: for the Codec 2 \
        modes a parameter-domain fill that interpolates pitch and energy across the gap and \
        holds the envelope; chip-only and refused for the AMBE modes). ber: flip channel bits \
        with probability --rate (default 1e-3). With --fec none that flips the voice bits \
        themselves, which no receiver ever sees; with --fec ysf-vd1 or ysf-vd2 (ysf-dmr only) \
        each frame is wrapped in that YSF DN mode's FEC, the transmitted bits are flipped, \
        and the frame is decoded as a receiver would, so only the residual error reaches the \
        vocoder. A mode 1 frame whose first Golay word is beyond repair is lost entirely. The \
        seeded --share of the base capture's utterances is \
        processed (stable across runs; --limit caps one run). AMBE modes need the ThumbDV \
        (--port / --dry-run); the Codec 2 modes run in software on --jobs threads."
)]
struct AugmentArgs {
    /// dstar | ysf-dmr | codec2-3200 | codec2-1600
    #[arg(long)]
    mode: VocoderMode,
    /// drops | ber
    #[arg(long)]
    kind: AugKind,
    /// Fraction of frames lost (drops) or per-bit flip probability (ber).
    /// Default: 0.02 for drops, 1e-3 for ber.
    #[arg(long)]
    rate: Option<f32>,
    /// Burst length range in frames, `lo..hi` or `n` (drops).
    #[arg(long, default_value = "1..3", value_parser = parse_burst)]
    burst: (u32, u32),
    /// mute | repeat | erase (drops).
    #[arg(long, default_value = "mute")]
    subst: Subst,
    /// none | ysf-vd1 | ysf-vd2 (ber): which FEC to model around the
    /// voice bits. none flips them directly, which no receiver sees.
    #[arg(long, default_value = "none")]
    fec: Fec,
    /// Share of the base capture's utterances to process (seeded by key).
    #[arg(long, default_value_t = 0.3)]
    share: f32,
    /// Seed of the share decision and the per-utterance mutation stream.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Serial port (repeatable). Must be in the ThumbDV scan. Chip modes only.
    #[arg(long = "port")]
    ports: Vec<String>,
    /// Decoder threads for the software modes (default: physical cores).
    #[arg(long)]
    jobs: Option<usize>,
    /// Corpus order (repeatable; default: the design's order).
    #[arg(long = "corpus")]
    corpora: Vec<String>,
    /// Only this split.
    #[arg(long)]
    split: Option<Split>,
    /// Only the augmented twins (rows with a parent): noisy, overdriven
    /// and underdriven. Without it fresh twins surface only at their
    /// hashed positions among the whole base corpus.
    #[arg(long)]
    only_twins: bool,
    /// At most this many utterances this run.
    #[arg(long)]
    limit: Option<usize>,
    /// Re-check the canary every N utterances per worker.
    #[arg(long, default_value_t = 200)]
    canary_every: usize,
    /// Use the simulated chip instead of a serial port (chip modes).
    #[arg(long)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum CaptureCmd {
    /// Finish the utterance in flight, then hold the port and wait.
    Pause {
        #[arg(long)]
        mode: VocoderMode,
    },
    /// Continue a paused capture.
    Resume {
        #[arg(long)]
        mode: VocoderMode,
    },
    /// Finish the utterance in flight, then exit.
    Stop {
        #[arg(long)]
        mode: VocoderMode,
    },
    /// Print captured/<mode>/status.json.
    Status {
        #[arg(long)]
        mode: VocoderMode,
        /// Print as JSON (the default is a one-line summary).
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum RunsCmd {
    /// List runs, newest first.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Status, config and checkpoints of one run.
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// SIGTERM the trainer (it checkpoints and exits 0) and wait for it.
    Stop {
        id: String,
        /// Seconds to wait for the run to leave `running`.
        #[arg(long, default_value_t = 60)]
        wait_s: u64,
    },
    /// Continue a run from its latest checkpoint, in this process.
    Resume {
        id: String,
        /// Override [train] device.
        #[arg(long)]
        device: Option<String>,
        /// Override [train] steps.
        #[arg(long)]
        steps: Option<u64>,
    },
}

// One arm per verb; the dispatcher grows with the verbs and does nothing
// else.
#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    let root = cli.data_root.map_or_else(DataRoot::from_env, DataRoot::new);
    match cli.cmd {
        Cmd::Prepare {
            corpora,
            force,
            jobs,
            noise_share,
            noise_sets,
            ham_chain_share,
            underdrive_share,
            seed,
            twins_only,
            resplit,
        } => {
            if resplit {
                resplit_cmd(&root)?;
            } else {
                prepare_cmd(
                    &root,
                    &PrepareOptions {
                        corpora,
                        force,
                        jobs,
                        twins: TwinOptions {
                            noise_share,
                            noise_sets,
                            chain_share: ham_chain_share,
                            under_share: underdrive_share,
                            seed,
                        },
                        twins_only,
                    },
                )?;
            }
        }
        Cmd::Capture(args) => capture_cmd(&root, args)?,
        Cmd::Augment(args) => augment_cmd(&root, args)?,
        Cmd::Recode(args) => recode_cmd(&root, args)?,
        Cmd::Measure { dir, kinds, json } => measure_cmd(&dir, &kinds, json)?,
        Cmd::Verify {
            mode,
            kind,
            sample,
            seed,
            json,
        } => verify_cmd(
            &root,
            &VerifyOptions {
                mode,
                kind,
                sample,
                seed,
            },
            json,
        )?,
        Cmd::Shard {
            mode,
            modes,
            name,
            crop_s,
            onset_share,
            tail_share,
            seed,
            examples_per_file,
            splits,
            kinds,
            balance,
            no_balance,
            max_utterances,
            twin_share,
            min_drop_rate,
            corpora,
        } => {
            let modes = match mode {
                Some(m) => vec![m],
                None => modes,
            };
            let mut o = ShardOptions::for_modes(name, modes);
            o.balance = balance && !no_balance;
            o.max_utterances = max_utterances;
            o.twin_share = twin_share;
            o.min_drop_rate = min_drop_rate;
            o.corpora = corpora;
            o.crop_s = crop_s;
            o.onset_share = onset_share;
            o.tail_share = tail_share;
            o.seed = seed;
            o.examples_per_file = examples_per_file;
            o.splits = splits;
            o.kinds = kinds
                .iter()
                .map(|w| parse_kind_sel(w))
                .collect::<Result<_, _>>()
                .context("--kinds")?;
            shard_cmd(&root, &o)?;
        }
        #[cfg(feature = "train")]
        Cmd::Train {
            config,
            run_dir,
            resume,
            device,
            steps,
            shards,
            init_from,
        } => {
            let cfg = read_run_config(&config)?;
            let overrides = unamblify_train::Overrides {
                device,
                steps,
                shards,
                init_from,
            };
            let out = with_data_root(&root, || {
                unamblify_train::run(cfg, run_dir.as_deref(), resume.as_deref(), &overrides)
            })?;
            print_outcome(&out);
            if out.state == RunState::Failed {
                bail!("run failed; see {}/log.jsonl", out.run_dir.display());
            }
        }
        #[cfg(feature = "train")]
        Cmd::Infer {
            run_dir,
            step,
            key,
            mode,
            out,
        } => infer_cmd(&root, &run_dir, step, &key, mode, out.as_deref())?,
        Cmd::Serve {
            runs_dir,
            configs_dir,
            bind,
            token,
            grace_s,
        } => {
            let opts = unamblify_web::ServeOpts {
                runs_dir,
                data_root: Some(root.path().to_path_buf()),
                configs_dir,
                bind,
                token,
                // The supervisor re-execs current_exe(): this binary.
                exe: None,
                grace: Duration::from_secs(grace_s),
                ..unamblify_web::ServeOpts::default()
            };
            serve_cmd(opts)?;
        }
        Cmd::Runs { runs_dir, cmd } => {
            let runs_dir = runs_dir.unwrap_or_else(|| root.path().join("runs"));
            runs_cmd(&root, &runs_dir, cmd)?;
        }
        Cmd::Stats { mode, json } => stats_cmd(&root, mode, json)?,
        Cmd::MigrateModes { dry_run } => migrate_cmd(&root, dry_run)?,
        #[cfg(feature = "train")]
        Cmd::Restore {
            weights,
            input,
            mode,
            out,
        } => {
            let audio =
                unamblify_audio::read(&input).with_context(|| input.display().to_string())?;
            anyhow::ensure!(
                audio.rate == 8000,
                "{}: {} Hz; the pipeline takes the 8 kHz codec output as captured",
                input.display(),
                audio.rate
            );
            let p = unamblify_train::pipeline::Pipeline::load_cpu(&weights)
                .with_context(|| weights.display().to_string())?;
            let m = p.mode_index(mode.as_str())?;
            let samples = p.run_slice(&audio.samples, m)?;
            unamblify_audio::write_wav_s16(&out, &samples, 24_000)?;
            println!(
                "{}: {} samples at 24 kHz from {} at 8 kHz ({mode})",
                out.display(),
                samples.len(),
                audio.samples.len()
            );
        }
        #[cfg(feature = "train")]
        Cmd::Bench {
            kind,
            weights,
            profile,
            width,
            lookahead,
            modes,
            seconds,
            passes,
            threads,
            noise_head,
            noise_mod,
        } => {
            if kind == "pipeline" {
                let r = unamblify_train::bench::run_pipeline(
                    seconds,
                    passes,
                    threads,
                    weights.as_deref(),
                )
                .context("bench")?;
                println!(
                    "restorer {} + synthesiser {} parameters{}",
                    r.params_restorer,
                    r.params_synth,
                    weights
                        .as_ref()
                        .map_or(" (random weights)".to_owned(), |w| format!(
                            " from {}",
                            w.display()
                        ))
                );
                println!(
                    "per 20 ms of audio on {} thread(s): features {:.2} ms, restorer {:.2} ms, synthesiser {:.2} ms, total {:.2} ms",
                    r.threads, r.features_ms, r.restorer_ms, r.synth_ms, r.total_ms
                );
                println!(
                    "{:.1} % of the real-time budget — {}",
                    r.budget_used() * 100.0,
                    if r.budget_used() < 1.0 {
                        format!("{:.1}x faster than real time", 1.0 / r.budget_used())
                    } else {
                        "TOO SLOW for real time on this core".to_owned()
                    }
                );
                return Ok(());
            }
            anyhow::ensure!(
                kind == "filter",
                "--kind must be filter or pipeline, got {kind}"
            );
            let opts = unamblify_train::bench::BenchOptions {
                noise_head,
                noise_mod,
                profile: profile.parse().map_err(|e| anyhow::anyhow!("{e}"))?,
                width,
                lookahead,
                embed_modes: (modes > 0).then_some(modes),
                seconds,
                passes,
                threads,
            };
            let r = unamblify_train::bench::run(&opts).context("bench")?;
            println!(
                "{} x{} ll{}{}: {} parameters",
                opts.profile,
                opts.width,
                opts.lookahead,
                opts.embed_modes
                    .map_or(String::new(), |m| format!(" with {m} modes")),
                r.params
            );
            println!(
                "{:.2} ms per 20 ms frame on {} thread(s) ({:.2}-{:.2} over {} passes)",
                r.ms_per_frame, r.threads, r.ms_per_frame_min, r.ms_per_frame_max, opts.passes
            );
            println!(
                "{:.1} % of the real-time budget — {}",
                r.budget_used * 100.0,
                if r.real_time() {
                    format!("{:.1}x faster than real time", 1.0 / r.budget_used)
                } else {
                    "TOO SLOW for this profile".to_owned()
                }
            );
            println!("note: whole-clip forward, so this is a lower bound on the streaming cost");
            if !r.real_time() {
                bail!(
                    "{} x{} does not make its frame in 20 ms",
                    opts.profile,
                    opts.width
                );
            }
        }
        #[cfg(feature = "train")]
        Cmd::Smoke { run_dir } => {
            let out = unamblify_train::smoke(run_dir.as_deref()).context("smoke")?;
            print_outcome(&out);
            println!("smoke: ok");
        }
    }
    Ok(())
}

fn serve_cmd(opts: unamblify_web::ServeOpts) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?
        .block_on(unamblify_web::serve(opts))
}

fn prepare_cmd(root: &DataRoot, opts: &PrepareOptions) -> anyhow::Result<()> {
    let s = prepare::run(root, opts).context("prepare")?;
    println!(
        "prepare: {} sources, {} written, {} skipped, {} rejected",
        s.sources, s.written, s.skipped, s.rejected
    );
    for (c, n) in &s.by_corpus {
        println!("  {c}: {n}");
    }
    if opts.twins.enabled() {
        println!(
            "twins: {} planned, {} written, {} skipped, {} failed",
            s.twins_planned, s.twins_written, s.twins_skipped, s.twins_errors
        );
    }
    Ok(())
}

fn resplit_cmd(root: &DataRoot) -> anyhow::Result<()> {
    let s = prepare::resplit(root).context("resplit")?;
    println!("resplit: {} rows, {} moved", s.rows, s.moved);
    for (m, n) in &s.moves {
        println!("  {m}: {n}");
    }
    let now: Vec<String> = s.by_split.iter().map(|(k, v)| format!("{k} {v}")).collect();
    println!("  now: {}", now.join(", "));
    if s.moved > 0 {
        println!("  rebuild any shard set before training on it");
    }
    Ok(())
}

fn verify_cmd(root: &DataRoot, opts: &VerifyOptions, json: bool) -> anyhow::Result<()> {
    let mode = unamblify::aug::capture_dir_name(opts.mode, opts.kind);
    let r = verify::run(root, opts).context("verify")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        println!(
            "verify {mode}: {} rows, {} checked, {} ok, {} problems",
            r.rows,
            r.checked,
            r.ok,
            r.problems.len()
        );
        for p in &r.problems {
            println!("  {}: {}", p.key, p.what);
        }
    }
    if !r.problems.is_empty() {
        bail!("{} problems", r.problems.len());
    }
    Ok(())
}

fn migrate_cmd(root: &DataRoot, dry_run: bool) -> anyhow::Result<()> {
    let r = unamblify_data::migrate::run(root, dry_run).context("migrate-modes")?;
    let verb = if dry_run { "would " } else { "" };
    for (from, to) in &r.renamed {
        println!("{verb}rename {} -> {}", from.display(), to.display());
    }
    for (path, n) in &r.rewritten {
        println!("{verb}rewrite {} ({n} mode fields)", path.display());
    }
    if r.is_empty() {
        println!(
            "migrate-modes: nothing to do under {}",
            root.path().display()
        );
    } else {
        println!(
            "migrate-modes{}: {} directories renamed, {} files rewritten",
            if dry_run { " (dry run)" } else { "" },
            r.renamed.len(),
            r.rewritten.len()
        );
    }
    Ok(())
}

fn shard_cmd(root: &DataRoot, opts: &ShardOptions) -> anyhow::Result<()> {
    let s = shard::build(root, opts).context("shard")?;
    println!(
        "shard {}: {} files, {} utterances, {} too short, {} unjoined; examples {:?}",
        root.shards(&opts.name).display(),
        s.files,
        s.utterances,
        s.too_short,
        s.unjoined,
        s.counts
    );
    if opts.modes.len() > 1 {
        for (split, per) in &s.counts_by_mode {
            let cols: Vec<String> = per
                .iter()
                .map(|(m, n)| {
                    let avail = s.available_by_mode[split][m];
                    if avail == *n {
                        format!("{m} {n}")
                    } else {
                        format!("{m} {n} of {avail}")
                    }
                })
                .collect();
            println!("  {split}: {}", cols.join(", "));
        }
    }
    Ok(())
}

fn capture_cmd(root: &DataRoot, args: CaptureArgs) -> anyhow::Result<()> {
    if let Some(control) = args.control {
        return match control {
            CaptureCmd::Pause { mode } => set_control(root, mode, ControlState::Pause),
            CaptureCmd::Resume { mode } => set_control(root, mode, ControlState::Run),
            CaptureCmd::Stop { mode } => set_control(root, mode, ControlState::Stop),
            CaptureCmd::Status { mode, json } => {
                let path = root.status_json(mode);
                let s = read_status(&path).with_context(|| path.display().to_string())?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&s)?);
                } else {
                    println!(
                        "capture {}: {:?} {}/{} done, {} failed, {:.1} frames/s, {:.0} utt/h{}{}{}",
                        s.mode,
                        s.state,
                        s.done,
                        s.total,
                        s.failed,
                        s.frames_s,
                        s.utt_per_hour,
                        s.eta_s
                            .map_or(String::new(), |e| format!(", eta {:.0} min", e / 60.0)),
                        s.current_key
                            .as_deref()
                            .map_or(String::new(), |k| format!(", at {k}")),
                        s.error
                            .as_deref()
                            .map_or(String::new(), |e| format!(", error: {e}")),
                    );
                }
                Ok(())
            }
        };
    }
    let mode = args
        .mode
        .context("--mode is required (dstar | ysf-dmr | codec2-3200 | codec2-1600)")?;
    if args.stats {
        let s = capture::stats(root, mode).context("capture stats")?;
        println!("{}", serde_json::to_string_pretty(&s)?);
        return Ok(());
    }
    let mut o = CaptureOptions::new(mode);
    o.ports = args.ports;
    o.jobs = args.jobs;
    o.corpora = args.corpora;
    o.order = args.order;
    o.order_seed = args.order_seed;
    o.split = args.split;
    o.limit = args.limit;
    o.only_twins = args.only_twins;
    o.canary_every = args.canary_every;
    o.dry_run = args.dry_run;
    o.sequential = args.sequential;
    o.warmup_mix = args.warmup_mix;
    o.cold_share = args.cold_share;
    o.warmup_seed = args.warmup_seed;
    let s = capture::run(root, &o).context("capture")?;
    println!(
        "capture {mode}: {:?}; {} done, {} failed, {} skipped of {} planned",
        s.state, s.done, s.failed, s.skipped, s.planned
    );
    Ok(())
}

#[derive(Args)]
struct RecodeArgs {
    /// dstar (the only mode with a second implementation vendored)
    #[arg(long)]
    mode: VocoderMode,
    /// perens — the vendored software D-STAR vocoder.
    #[arg(long)]
    kind: AugKind,
    /// Encoder threads (default: physical cores). This stage never opens
    /// a port, so there is nothing else to parallelise over.
    #[arg(long)]
    jobs: Option<usize>,
    /// Corpus order (repeatable; default: the design's order).
    #[arg(long = "corpus")]
    corpora: Vec<String>,
    /// Only this split.
    #[arg(long)]
    split: Option<Split>,
    /// Only the augmented twins (rows with a parent): noisy, overdriven
    /// and underdriven. Without it fresh twins surface only at their
    /// hashed positions among the whole base corpus.
    #[arg(long)]
    only_twins: bool,
    /// At most this many utterances this run.
    #[arg(long)]
    limit: Option<usize>,
    /// Re-check the canary every N utterances per worker.
    #[arg(long, default_value_t = 200)]
    canary_every: usize,
}

/// Encode the prepared audio again with another implementation of the
/// mode's codec. Unlike `augment`, this does not read the base capture:
/// it is a full round trip, so every prepared utterance is eligible.
fn recode_cmd(root: &DataRoot, args: RecodeArgs) -> anyhow::Result<()> {
    let mut o = CaptureOptions::new(args.mode);
    o.stage = Stage::Recode(args.kind);
    o.jobs = args.jobs;
    o.corpora = args.corpora;
    o.split = args.split;
    o.limit = args.limit;
    o.only_twins = args.only_twins;
    o.canary_every = args.canary_every;
    let name = o.out_dir(root).name();
    let t = Instant::now();
    let s = capture::run(root, &o).context("recode")?;
    println!(
        "recode {name}: {:?}; {} done, {} failed, {} skipped of {} planned in {:.1} s",
        s.state,
        s.done,
        s.failed,
        s.skipped,
        s.planned,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One `--kinds` word: `base`, a sibling (`drops`), or a sibling scoped
/// to one mode (`dstar+perens`) for a kind that exists for that mode
/// alone. The scoped spelling is the capture directory's own name, so
/// what is typed matches what is on disk.
fn parse_kind_sel(word: &str) -> anyhow::Result<KindSel> {
    let Some((m, k)) = word.split_once('+') else {
        return Ok(KindSel::every(parse_kind_word(word)?));
    };
    let mode: VocoderMode = m
        .parse()
        .with_context(|| format!("{word}: {m} is not a mode"))?;
    let kind: AugKind = k
        .parse()
        .with_context(|| format!("{word}: {k} is not a capture kind"))?;
    Ok(KindSel::scoped(mode, kind))
}

fn augment_cmd(root: &DataRoot, args: AugmentArgs) -> anyhow::Result<()> {
    let mut spec = AugmentSpec::new(args.kind);
    if let Some(r) = args.rate {
        spec.rate = r;
    }
    spec.burst = args.burst;
    spec.subst = args.subst;
    spec.fec = args.fec;
    spec.share = args.share;
    spec.seed = args.seed;
    let mut o = CaptureOptions::new(args.mode);
    o.stage = Stage::Augment(spec);
    o.ports = args.ports;
    o.jobs = args.jobs;
    o.corpora = args.corpora;
    o.split = args.split;
    o.limit = args.limit;
    o.only_twins = args.only_twins;
    o.canary_every = args.canary_every;
    o.dry_run = args.dry_run;
    let name = o.out_dir(root).name();
    let t = Instant::now();
    let s = capture::run(root, &o).context("augment")?;
    println!(
        "augment {name}: {:?}; {} done, {} failed, {} skipped of {} planned in {:.1} s",
        s.state,
        s.done,
        s.failed,
        s.skipped,
        s.planned,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn set_control(root: &DataRoot, mode: VocoderMode, state: ControlState) -> anyhow::Result<()> {
    let path = root.control_json(mode);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    }
    write_control(&path, state).with_context(|| path.display().to_string())?;
    println!("wrote {} = {state:?}", path.display());
    if state == ControlState::Stop {
        println!(
            "the harness finishes the utterance in flight, then exits (capture status --mode {mode})"
        );
    }
    Ok(())
}

fn runs_cmd(root: &DataRoot, runs_dir: &Path, cmd: RunsCmd) -> anyhow::Result<()> {
    match cmd {
        RunsCmd::List { json } => runs_list(runs_dir, json),
        RunsCmd::Show { id, json } => runs_show(runs_dir, &id, json),
        RunsCmd::Stop { id, wait_s } => runs_stop(runs_dir, &id, wait_s),
        RunsCmd::Resume { id, device, steps } => {
            let shards = None;
            let dir = runs::run_dir(runs_dir, &id).context("run dir")?;
            let ckpt = runs::latest_checkpoint(&dir)
                .with_context(|| format!("run {id} has no checkpoint to resume from"))?;
            if let Some(st) = runs::read_status(&dir)
                && st.status == RunState::Running
                && st.pid.is_some_and(unamblify_web::supervisor::pid_alive)
            {
                bail!("run {id} is still running (pid {:?})", st.pid);
            }
            resume_run(root, &dir, &ckpt, device, steps, shards)
        }
    }
}

fn runs_list(runs_dir: &Path, json: bool) -> anyhow::Result<()> {
    let ids = runs::list_ids(runs_dir).context("list runs")?;
    let rows: Vec<runs::RunSummary> = ids
        .iter()
        .map(|id| runs::summarize(runs_dir, id, false))
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("no runs under {}", runs_dir.display());
        return Ok(());
    }
    println!(
        "{:<36} {:<9} {:>7}/{:<7} {:>9} {:>5}  device",
        "run", "status", "step", "total", "loss", "ckpts"
    );
    for r in &rows {
        let (status, step, total, device) = r.status.as_ref().map_or_else(
            || ("-".to_owned(), 0, 0, "-".to_owned()),
            |s| {
                (
                    s.status.to_string(),
                    s.step,
                    s.total_steps,
                    s.device.clone(),
                )
            },
        );
        println!(
            "{:<36} {:<9} {:>7}/{:<7} {:>9} {:>5}  {}",
            r.id,
            status,
            step,
            total,
            r.last_loss.map_or("-".to_owned(), |l| format!("{l:.4}")),
            r.checkpoints,
            device
        );
    }
    Ok(())
}

fn runs_show(runs_dir: &Path, id: &str, json: bool) -> anyhow::Result<()> {
    let dir = runs::run_dir(runs_dir, id).context("run dir")?;
    anyhow::ensure!(dir.is_dir(), "no run {id} under {}", runs_dir.display());
    let summary = runs::summarize(runs_dir, id, false);
    let checkpoints = runs::list_checkpoints(&dir);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "run": summary,
                "checkpoints": checkpoints,
            }))?
        );
        return Ok(());
    }
    println!("{}  ({})", summary.id, dir.display());
    if let Some(s) = &summary.status {
        println!(
            "  status {}  step {}/{}  device {}  host {}  started {}  updated {}",
            s.status, s.step, s.total_steps, s.device, s.host, s.started, s.updated
        );
        if let Some(b) = &s.best {
            println!("  best {} = {:.4} at step {}", b.metric, b.value, b.step);
        }
    }
    if let Some(l) = summary.last_loss {
        println!("  last loss/total {l:.4}");
    }
    if let Ok((text, _)) = runs::read_config(&dir) {
        println!("  config.toml:");
        for line in text.lines() {
            println!("    {line}");
        }
    }
    println!("  checkpoints: {}", checkpoints.len());
    for c in &checkpoints {
        println!(
            "    {}  model={} optim={} clips={}",
            c.dir,
            c.model,
            c.optim,
            c.clips.len()
        );
    }
    Ok(())
}

fn runs_stop(runs_dir: &Path, id: &str, wait_s: u64) -> anyhow::Result<()> {
    let dir = runs::run_dir(runs_dir, id).context("run dir")?;
    let st =
        runs::read_status(&dir).with_context(|| format!("{}: no status.json", dir.display()))?;
    if st.status != RunState::Running {
        println!("run {id} is {}, nothing to stop", st.status);
        return Ok(());
    }
    let pid = st.pid.context("status.json has no pid")?;
    if !unamblify_web::supervisor::pid_alive(pid) {
        // The trainer died without anyone finalising status.json (a
        // crash, a power loss); the pid may now belong to something else,
        // so never signal it — record what happened instead.
        let mut st = st;
        st.status = RunState::Stopped;
        st.pid = None;
        st.updated = unamblify_web::clock::rfc3339_now();
        runs::write_status(&dir, &st).context("status.json")?;
        println!("run {id}: pid {pid} is not alive; marked stopped");
        return Ok(());
    }
    let raw = i32::try_from(pid).context("pid")?;
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(raw), nix::sys::signal::SIGTERM)
        .with_context(|| format!("SIGTERM pid {pid}"))?;
    println!("sent SIGTERM to pid {pid}; waiting for a checkpoint");
    let deadline = Instant::now() + Duration::from_secs(wait_s);
    loop {
        let now = runs::read_status(&dir).map(|s| s.status);
        if let Some(state) = now
            && state != RunState::Running
        {
            println!("run {id} is now {state}");
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("run {id} still running after {wait_s} s");
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

#[cfg(feature = "train")]
fn resume_run(
    root: &DataRoot,
    dir: &Path,
    ckpt: &Path,
    device: Option<String>,
    steps: Option<u64>,
    shards: Option<String>,
) -> anyhow::Result<()> {
    let (_, cfg) = runs::read_config(dir).context("config.toml")?;
    let overrides = unamblify_train::Overrides {
        device,
        steps,
        shards,
        init_from: None,
    };
    println!("resuming {} from {}", dir.display(), ckpt.display());
    let out = with_data_root(root, || {
        unamblify_train::run(cfg, Some(dir), Some(ckpt), &overrides)
    })?;
    print_outcome(&out);
    if out.state == RunState::Failed {
        bail!("run failed; see {}/log.jsonl", out.run_dir.display());
    }
    Ok(())
}

#[cfg(not(feature = "train"))]
fn resume_run(
    _root: &DataRoot,
    dir: &Path,
    ckpt: &Path,
    _device: Option<String>,
    _steps: Option<u64>,
    _shards: Option<String>,
) -> anyhow::Result<()> {
    bail!(
        "this binary was built without the `train` feature; run `unamblify train --config {}/config.toml --run-dir {} --resume {}` with a full build",
        dir.display(),
        dir.display(),
        ckpt.display()
    )
}

/// The timing part of a `stats` line. An interleaved capture has no
/// separable encode and decode time, so it reports the round trip
/// instead of two zeros; a mixed manifest reports whatever is non-zero.
fn timing_line(c: &unamblify_data::capture::ManifestStats) -> String {
    let mut parts = Vec::new();
    if c.encode_ms_per_frame > 0.0 || c.roundtrip_ms_per_frame == 0.0 {
        parts.push(format!("encode {:.2} ms/frame", c.encode_ms_per_frame));
    }
    if c.decode_ms_per_frame > 0.0 || c.roundtrip_ms_per_frame == 0.0 {
        parts.push(format!("decode {:.2} ms/frame", c.decode_ms_per_frame));
    }
    if c.roundtrip_ms_per_frame > 0.0 {
        parts.push(format!(
            "round trip {:.2} ms/frame",
            c.roundtrip_ms_per_frame
        ));
    }
    parts.join(", ")
}

fn stats_cmd(root: &DataRoot, mode: Option<VocoderMode>, json: bool) -> anyhow::Result<()> {
    let prepared = prepare::stats(root).context("prepared stats")?;
    // Every capture set present (base and decode-only siblings), or the
    // one mode asked for and its siblings.
    let dirs: Vec<unamblify_data::CaptureDir> = root
        .capture_dirs_present()
        .into_iter()
        .filter(|d| mode.is_none_or(|m| d.mode() == m))
        .filter(|d| d.manifest().is_file())
        .collect();
    let mut captured = Vec::new();
    for d in dirs {
        captured.push(
            capture::stats_in(root, &d).with_context(|| format!("captured stats {}", d.name()))?,
        );
    }
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "data_root": root.path(),
                "prepared": prepared,
                "captured": captured,
            }))?
        );
        return Ok(());
    }
    println!("data root {}", root.path().display());
    println!(
        "prepared: {} rows ({} twins), {:.1} h, {} speakers, {} rejected",
        prepared.rows, prepared.twins, prepared.hours, prepared.speakers, prepared.rejected
    );
    for (c, n) in &prepared.by_corpus {
        println!("  {c}: {n}");
    }
    for (s, n) in &prepared.by_split {
        println!("  {s}: {n}");
    }
    if captured.is_empty() {
        println!("captured: nothing yet");
    }
    for c in &captured {
        println!(
            "captured {}: {} rows ({} failed), {} remaining, {:.1} h, {} frames, {}",
            unamblify::aug::capture_dir_name(c.mode, c.kind),
            c.captured,
            c.failed,
            c.remaining,
            c.hours,
            c.frames,
            timing_line(c)
        );
        for (k, n) in &c.by_corpus {
            println!("  {k}: {n}");
        }
        for (s, n) in &c.by_split {
            println!("  {s}: {n}");
        }
    }
    Ok(())
}

#[cfg(feature = "train")]
fn infer_cmd(
    root: &DataRoot,
    run_dir: &Path,
    step: u64,
    key: &str,
    mode: Option<VocoderMode>,
    out: Option<&Path>,
) -> anyhow::Result<()> {
    let r = unamblify_train::infer::run_one_in(run_dir, step, key, mode, root.path(), out)
        .context("infer")?;
    println!(
        "infer {key} through {} step {step} ({}): {} samples at 16 kHz -> {}",
        run_dir.display(),
        r.mode,
        r.samples,
        r.out_wav.display()
    );
    Ok(())
}

#[cfg(feature = "train")]
fn read_run_config(path: &Path) -> anyhow::Result<unamblify::RunConfig> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    toml::from_str(&text).with_context(|| format!("{}: not a run config", path.display()))
}

/// The train crate reads `$UNAMBLIFY_DATA` itself; make `--data-root`
/// reach it without threading a path through its API.
#[cfg(feature = "train")]
fn with_data_root<T>(root: &DataRoot, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    // SAFETY: called on the main thread before any worker threads exist
    // (the trainer spawns none until it runs), so no concurrent reader.
    unsafe { std::env::set_var("UNAMBLIFY_DATA", root.path()) };
    f()
}

#[cfg(feature = "train")]
fn print_outcome(out: &unamblify_train::Outcome) {
    println!(
        "run {}: {} at step {}",
        out.run_dir.display(),
        out.state,
        out.step
    );
    if let Some((s0, p0)) = out.probe.first()
        && let Some((s1, p1)) = out.probe.last()
        && s0 != s1
    {
        println!("  loss/probe {p0:.4} (step {s0}) -> {p1:.4} (step {s1})");
    }
    if let Some(e) = &out.last_eval {
        println!("  last eval: {e:?}");
    }
}

/// One (mode, kind) cell of `unamblify measure`: burst-weighted plosive
/// means and frame-weighted sibilant means over every clip that had them.
#[derive(Default, serde::Serialize)]
struct MeasureCell {
    clips: usize,
    bursts: usize,
    plosive_burst_db: f32,
    plosive_closure_db: f32,
    plosive_rise_ms: f32,
    sibilant_frames: usize,
    sibilant_level_db: f32,
    sibilant_balance_db: f32,
    /// Weighted sums while accumulating: burst × (burst, closure, rise),
    /// frames × (level, balance). Turned into the means by `finish`.
    #[serde(skip)]
    sums: [f64; 5],
}

impl MeasureCell {
    fn add(&mut self, other: &[f32], clean: &[f32], rate: u32) {
        use unamblify_audio::{plosive_excess, sibilant_excess};
        self.clips += 1;
        if let Some(pl) = plosive_excess(other, clean, rate) {
            self.bursts += pl.bursts;
            #[allow(clippy::cast_precision_loss)]
            let w = pl.bursts as f64;
            self.sums[0] += f64::from(pl.burst_db) * w;
            self.sums[1] += f64::from(pl.closure_db) * w;
            self.sums[2] += f64::from(pl.rise_ms) * w;
        }
        if let Some(si) = sibilant_excess(other, clean, rate) {
            self.sibilant_frames += si.frames;
            #[allow(clippy::cast_precision_loss)]
            let w = si.frames as f64;
            self.sums[3] += f64::from(si.level_db) * w;
            self.sums[4] += f64::from(si.balance_db) * w;
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn finish(&mut self) {
        if self.bursts > 0 {
            let b = self.bursts as f64;
            self.plosive_burst_db = (self.sums[0] / b) as f32;
            self.plosive_closure_db = (self.sums[1] / b) as f32;
            self.plosive_rise_ms = (self.sums[2] / b) as f32;
        }
        if self.sibilant_frames > 0 {
            let f = self.sibilant_frames as f64;
            self.sibilant_level_db = (self.sums[3] / f) as f32;
            self.sibilant_balance_db = (self.sums[4] / f) as f32;
        }
    }
}

/// The `*.clean.{wav,flac}` files under `dir`, sorted, as (path, stem,
/// extension).
fn clean_clips(dir: &Path) -> anyhow::Result<Vec<(PathBuf, String, String)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| dir.display().to_string())? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext = ext.to_ascii_lowercase();
        if ext != "wav" && ext != "flac" {
            continue;
        }
        let Some(stem) = name.strip_suffix(&format!(".clean.{ext}")) else {
            continue;
        };
        out.push((path.clone(), stem.to_owned(), ext));
    }
    out.sort();
    anyhow::ensure!(!out.is_empty(), "no *.clean.wav under {}", dir.display());
    Ok(out)
}

/// `unamblify measure`: plosive and sibilant excess of every kind
/// against `clean`, per mode, over a directory of rendered clips.
fn measure_cmd(dir: &Path, kinds: &[String], json: bool) -> anyhow::Result<()> {
    let mut cells: std::collections::BTreeMap<(String, String), MeasureCell> =
        std::collections::BTreeMap::new();
    for (clean_path, stem, ext) in clean_clips(dir)? {
        let mode = stem
            .rsplit('@')
            .next()
            .filter(|_| stem.contains('@'))
            .unwrap_or("-")
            .to_owned();
        let clean = unamblify_audio::read(&clean_path)?;
        for kind in kinds {
            let p = dir.join(format!("{stem}.{kind}.{ext}"));
            if !p.is_file() {
                continue;
            }
            let other = unamblify_audio::read(&p)?;
            if other.rate != clean.rate {
                log::warn!(
                    "{}: {} Hz against clean's {} Hz, skipped",
                    p.display(),
                    other.rate,
                    clean.rate
                );
                continue;
            }
            cells.entry((mode.clone(), kind.clone())).or_default().add(
                &other.samples,
                &clean.samples,
                clean.rate,
            );
        }
    }
    for cell in cells.values_mut() {
        cell.finish();
    }
    if json {
        let rows: Vec<serde_json::Value> = cells
            .iter()
            .map(|((mode, kind), c)| {
                let mut v = serde_json::to_value(c).unwrap_or_default();
                v["mode"] = serde_json::Value::String(mode.clone());
                v["kind"] = serde_json::Value::String(kind.clone());
                v
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    println!(
        "{:<12} {:<9} {:>5} {:>6} {:>9} {:>10} {:>8} {:>7} {:>9} {:>10}",
        "mode",
        "kind",
        "clips",
        "bursts",
        "burst dB",
        "closure dB",
        "rise ms",
        "frames",
        "s lvl dB",
        "s bal dB"
    );
    for ((mode, kind), c) in &cells {
        println!(
            "{mode:<12} {kind:<9} {:>5} {:>6} {:>9.1} {:>10.1} {:>8.1} {:>7} {:>9.1} {:>10.1}",
            c.clips,
            c.bursts,
            c.plosive_burst_db,
            c.plosive_closure_db,
            c.plosive_rise_ms,
            c.sibilant_frames,
            c.sibilant_level_db,
            c.sibilant_balance_db
        );
    }
    println!(
        "(every column is kind minus clean at the clean's own bursts / sibilant frames; 0 = as the recording)"
    );
    Ok(())
}
