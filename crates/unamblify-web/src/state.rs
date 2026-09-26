// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Shared server state handed to every handler.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sysinfo::System;

use crate::error::Result;
use crate::events::Hub;
use crate::runs::MetricKeysTrack;
use crate::samples;
use crate::supervisor::Supervisor;

/// Everything the handlers need.
#[derive(Debug)]
pub struct AppState {
    /// `runs/`.
    pub runs_dir: PathBuf,
    /// `$UNAMBLIFY_DATA`.
    pub data_root: PathBuf,
    /// Where `configs/*.toml` live.
    pub configs_dir: PathBuf,
    /// Event bus.
    pub hub: Arc<Hub>,
    /// Process supervisor.
    pub sup: Arc<Supervisor>,
    /// `sysinfo` handle (kept so CPU usage is measured between calls).
    pub sys: Mutex<System>,
    /// When the server started, Unix seconds.
    pub started_s: u64,
    /// Per-run metric key sets, advanced incrementally.
    pub metric_keys: Mutex<HashMap<String, MetricKeysTrack>>,
    /// The Samples navigator's manifest join.
    pub samples: Mutex<samples::Cache>,
    /// Rolling drive temperatures for the Host page's chart.
    pub drives: Mutex<crate::drives::History>,
    /// Rolling CPU and GPU use for the Host page's chart.
    pub host: Mutex<crate::host::History>,
}

impl AppState {
    /// `captured/` under the data root.
    #[must_use]
    pub fn captured_dir(&self) -> PathBuf {
        self.data_root.join("captured")
    }

    /// The metric keys of run `id` at `dir`, reading only what
    /// `metrics.jsonl` gained since the last call.
    pub fn metric_keys(&self, id: &str, dir: &Path) -> Result<Vec<String>> {
        let mut track = self
            .metric_keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id)
            .unwrap_or_default();
        let keys = track.update(dir);
        self.metric_keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.to_owned(), track);
        keys
    }

    /// The samples index, rebuilt when a manifest changed. Blocking (a
    /// first build parses the whole prepared manifest): call it from
    /// `spawn_blocking`.
    pub fn samples_index(&self) -> Result<Arc<samples::Index>> {
        self.samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .index(&self.data_root)
    }
}
