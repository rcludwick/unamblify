// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The training harness (spec §4–§7): the tch-rs model, losses, dataset
//! loaders, trainer loop, checkpointing, evaluation, metrics emitter and
//! checkpoint audio rendering. The only crate in the workspace that links
//! libtorch — build it after `just train-env` (see
//! `scripts/train-env.sh` and `.cargo/config.toml.example`).
//!
//! Entry points for the CLI: [`run`] (a config to completion, resumable),
//! [`smoke`] (20 CPU steps on synthetic data; fails unless the loss went
//! down) and [`infer::run_one`] (one utterance through one checkpoint,
//! for the dashboard's Samples page).

pub mod bench;
pub mod checkpoint;
pub mod data;
pub mod device;
pub mod eval;
pub mod infer;
pub mod losses;
pub mod model;
pub mod optim;
pub mod pipeline;
pub mod rng;
pub mod runio;
pub mod stft;
pub mod sys;
pub mod time;
pub mod trainer;

pub use trainer::{DataSel, Outcome, Overrides, data_root, run, run_with, smoke, smoke_config};
