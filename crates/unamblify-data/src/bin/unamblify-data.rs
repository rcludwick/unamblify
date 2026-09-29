// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! A thin local-testing front end for the data stages. The real CLI is
//! `unamblify` in `crates/unamblify-cli`; this one exists so each stage
//! can be run on its own while that crate lands.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use unamblify::{Split, VocoderMode};
use unamblify_data::DataRoot;
use unamblify_data::capture::{self, CaptureOptions};
use unamblify_data::control::{ControlState, read_status, write_control};
use unamblify_data::prepare::{self, PrepareOptions};
use unamblify_data::shard::{self, ShardOptions};
use unamblify_data::verify::{self, VerifyOptions};

#[derive(Parser)]
#[command(
    name = "unamblify-data",
    about = "unamblify data stages (local testing front end)"
)]
struct Cli {
    /// Data root (default `$UNAMBLIFY_DATA`, else `/Volumes/data/training_data/unamblify`).
    #[arg(long, global = true)]
    data_root: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Walk raw/ into prepared/.
    Prepare {
        /// Corpus (repeatable, in order).
        #[arg(long = "corpus")]
        corpora: Vec<String>,
        /// Re-do everything found.
        #[arg(long)]
        force: bool,
        /// Worker threads.
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Run a capture.
    Capture {
        #[arg(long)]
        mode: VocoderMode,
        /// Serial port (repeatable). Must be in the ThumbDV scan. Chip
        /// modes only.
        #[arg(long = "port")]
        ports: Vec<String>,
        /// Encoder threads (software modes only; default: physical cores).
        #[arg(long)]
        jobs: Option<usize>,
        #[arg(long = "corpus")]
        corpora: Vec<String>,
        #[arg(long)]
        split: Option<Split>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 200)]
        canary_every: usize,
        /// Use the simulated chip.
        #[arg(long)]
        dry_run: bool,
        /// Print the manifest summary instead of capturing.
        #[arg(long)]
        stats: bool,
    },
    /// Write control.json for a running capture.
    Control {
        #[arg(long)]
        mode: VocoderMode,
        /// run | pause | stop
        #[arg(value_enum)]
        state: State,
    },
    /// Print status.json.
    Status {
        #[arg(long)]
        mode: VocoderMode,
    },
    /// Recheck a capture's hashes and frame counts.
    Verify {
        #[arg(long)]
        mode: VocoderMode,
        #[arg(long, default_value_t = 0)]
        sample: usize,
    },
    /// Pack fixed-length examples.
    Shard {
        #[arg(long)]
        mode: VocoderMode,
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
    },
    /// Print the prepared manifest summary.
    Stats,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum State {
    Run,
    Pause,
    Stop,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    let root = cli.data_root.map_or_else(DataRoot::from_env, DataRoot::new);
    match cli.cmd {
        Cmd::Prepare {
            corpora,
            force,
            jobs,
        } => {
            let s = prepare::run(
                &root,
                &PrepareOptions {
                    corpora,
                    force,
                    jobs,
                    twins: unamblify_data::twins::TwinOptions::default(),
                    twins_only: false,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        Cmd::Capture {
            mode,
            ports,
            jobs,
            corpora,
            split,
            limit,
            canary_every,
            dry_run,
            stats,
        } => {
            if stats {
                let s = capture::stats(&root, mode)?;
                println!("{}", serde_json::to_string_pretty(&s)?);
                return Ok(());
            }
            let mut o = CaptureOptions::new(mode);
            o.ports = ports;
            o.jobs = jobs;
            o.corpora = corpora;
            o.split = split;
            o.limit = limit;
            o.canary_every = canary_every;
            o.dry_run = dry_run;
            let s = capture::run(&root, &o).context("capture")?;
            println!("{s:?}");
        }
        Cmd::Control { mode, state } => {
            let state = match state {
                State::Run => ControlState::Run,
                State::Pause => ControlState::Pause,
                State::Stop => ControlState::Stop,
            };
            write_control(&root.control_json(mode), state)?;
        }
        Cmd::Status { mode } => {
            let s = read_status(&root.status_json(mode))?;
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        Cmd::Verify { mode, sample } => {
            let r = verify::run(
                &root,
                &VerifyOptions {
                    kind: None,
                    mode,
                    sample,
                    seed: 1,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if !r.problems.is_empty() {
                anyhow::bail!("{} problems", r.problems.len());
            }
        }
        Cmd::Shard {
            mode,
            name,
            crop_s,
            onset_share,
            tail_share,
            seed,
        } => {
            let mut o = ShardOptions::new(name, mode);
            o.crop_s = crop_s;
            o.onset_share = onset_share;
            o.tail_share = tail_share;
            o.seed = seed;
            let s = shard::build(&root, &o)?;
            println!("{s:?}");
        }
        Cmd::Stats => {
            let s = prepare::stats(&root)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
    }
    Ok(())
}
