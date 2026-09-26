// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The runs-dir poller. Every `interval` (250 ms by default, which is also
//! the SSE coalescing window of spec §8) it stats each run's `status.json`,
//! reads the new bytes of `metrics.jsonl` and `log.jsonl`, notices new
//! checkpoint directories, and publishes one coalesced event per kind per
//! run on the [`Hub`]. It also watches every capture set's `status.json`
//! for the list stream's `capture` event.
//!
//! Polling rather than `notify`: the files are appended by another process
//! at ≥ 250 ms cadence, a stat per file per tick is negligible, and it
//! behaves identically on macOS, Linux and network mounts.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde_json::{Value, json};
use unamblify::{LogRow, MetricRow, VocoderMode};

use crate::events::{Event, Hub, Kind};
use crate::runs;

/// Default tick.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stamp {
    len: u64,
    mtime: Option<SystemTime>,
}

fn stamp(p: &Path) -> Stamp {
    fs::metadata(p)
        .map(|m| Stamp {
            len: m.len(),
            mtime: m.modified().ok(),
        })
        .unwrap_or_default()
}

#[derive(Debug, Default)]
struct RunTrack {
    status: Stamp,
    live: bool,
    metrics_off: u64,
    log_off: u64,
    ckpts: BTreeSet<u64>,
    seen_ckpts: bool,
}

/// Poller state; drive it with [`Watcher::run`] or tick it by hand in
/// tests with [`Watcher::tick`].
#[derive(Debug)]
pub struct Watcher {
    runs_dir: PathBuf,
    captured_dir: PathBuf,
    hub: Arc<Hub>,
    runs: HashMap<String, RunTrack>,
    capture: HashMap<String, (Stamp, u64)>,
    known_ids: BTreeSet<String>,
}

impl Watcher {
    /// A poller over `runs_dir` and `captured_dir` publishing on `hub`.
    #[must_use]
    pub fn new(runs_dir: PathBuf, captured_dir: PathBuf, hub: Arc<Hub>) -> Self {
        Self {
            runs_dir,
            captured_dir,
            hub,
            runs: HashMap::new(),
            capture: HashMap::new(),
            known_ids: BTreeSet::new(),
        }
    }

    /// Loop forever at `interval`. Each tick runs on the blocking pool so
    /// the file reads never stall the executor, whatever its flavour.
    pub async fn run(mut self, interval: Duration) {
        let mut t = tokio::time::interval(interval);
        t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            t.tick().await;
            self = match tokio::task::spawn_blocking(move || {
                self.tick();
                self
            })
            .await
            {
                Ok(w) => w,
                Err(_) => return,
            };
        }
    }

    /// One pass. Returns the number of events published.
    pub fn tick(&mut self) -> usize {
        let mut published = 0;
        let listeners = self.hub.listeners();
        let ids = runs::list_ids(&self.runs_dir).unwrap_or_default();
        let id_set: BTreeSet<String> = ids.iter().cloned().collect();
        let mut list_changed = id_set != self.known_ids;
        self.runs.retain(|k, _| id_set.contains(k));
        for id in &ids {
            let dir = self.runs_dir.join(id);
            let first = !self.runs.contains_key(id);
            let tr = self.runs.entry(id.clone()).or_default();
            let (n, changed) = poll_run(&self.hub, &dir, id, tr, listeners, first);
            published += n;
            list_changed |= changed;
        }
        if list_changed {
            self.known_ids = id_set;
            if listeners > 0 {
                self.hub
                    .publish(Event::global(Kind::Runs, json!({ "ids": ids })));
                published += 1;
            }
        }
        // Capture: every base mode, plus every sibling set present.
        let mut names: Vec<String> = VocoderMode::ALL
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect();
        names.extend(crate::capture::capture_set_names(&self.captured_dir));
        names.dedup();
        for mode in names {
            let dir = self.captured_dir.join(&mode);
            let st = stamp(&dir.join("status.json"));
            let log_len = stamp(&dir.join("log.jsonl")).len;
            let first = !self.capture.contains_key(&mode);
            let entry = self
                .capture
                .entry(mode.clone())
                .or_insert_with(|| (st.clone(), log_len));
            if !first && (entry.0 != st || entry.1 != log_len) {
                *entry = (st, log_len);
                if listeners > 0 {
                    let status: Option<Value> = fs::read(dir.join("status.json"))
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok());
                    self.hub.publish(Event::global(
                        Kind::Capture,
                        json!({ "mode": mode, "event": "status", "status": status }),
                    ));
                    published += 1;
                }
            }
        }
        published
    }
}

/// Poll one run. Returns (events published, whether the list view changed).
fn poll_run(
    hub: &Hub,
    dir: &Path,
    id: &str,
    tr: &mut RunTrack,
    listeners: usize,
    first: bool,
) -> (usize, bool) {
    let mut published = 0;
    let mut list_changed = false;
    let st = stamp(&dir.join("status.json"));
    let status_changed = st != tr.status;
    if status_changed {
        tr.status = st;
        let status = runs::read_status(dir);
        tr.live = status.as_ref().is_some_and(|s| s.status.is_live());
        if !first {
            list_changed = true;
            if listeners > 0
                && let Some(s) = status
            {
                hub.publish(Event::for_run(
                    id,
                    Kind::Status,
                    serde_json::to_value(s).unwrap_or(Value::Null),
                ));
                published += 1;
            }
        }
    }
    if first {
        // Start at the end: history comes from the REST endpoints.
        tr.metrics_off = stamp(&dir.join("metrics.jsonl")).len;
        tr.log_off = stamp(&dir.join("log.jsonl")).len;
        tr.ckpts = runs::list_checkpoints(dir)
            .into_iter()
            .map(|c| c.step)
            .collect();
        tr.seen_ckpts = true;
        return (published, list_changed);
    }
    if !(tr.live || status_changed) {
        return (published, list_changed);
    }
    // Metrics.
    if let Ok((lines, off)) = runs::read_new_lines(&dir.join("metrics.jsonl"), tr.metrics_off) {
        tr.metrics_off = off;
        if listeners > 0 && !lines.is_empty() {
            let rows: Vec<MetricRow> = lines
                .iter()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            let (sys, metric): (Vec<&MetricRow>, Vec<&MetricRow>) =
                rows.iter().partition(|r| r.k.starts_with("sys/"));
            if !metric.is_empty() {
                hub.publish(Event::for_run(id, Kind::Metric, json!(metric)));
                published += 1;
            }
            if !sys.is_empty() {
                hub.publish(Event::for_run(id, Kind::Sys, json!(sys)));
                published += 1;
            }
        }
    }
    // Logs.
    if let Ok((lines, off)) = runs::read_new_lines(&dir.join("log.jsonl"), tr.log_off) {
        tr.log_off = off;
        if listeners > 0 && !lines.is_empty() {
            let rows: Vec<LogRow> = lines
                .iter()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            if !rows.is_empty() {
                hub.publish(Event::for_run(id, Kind::Log, json!(rows)));
                published += 1;
            }
        }
    }
    // Checkpoints.
    for c in runs::list_checkpoints(dir) {
        if tr.ckpts.insert(c.step) && tr.seen_ckpts && listeners > 0 {
            hub.publish(Event::for_run(
                id,
                Kind::Checkpoint,
                serde_json::to_value(&c).unwrap_or(Value::Null),
            ));
            published += 1;
        }
    }
    (published, list_changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn tick_publishes_coalesced_metric_log_status_and_checkpoint_events() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        let cap = tmp.path().join("captured");
        let dir = runs_dir.join("20260910-050000-a");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.toml"), "name = \"a\"\n").unwrap();
        let mut st = runs::new_status(
            &unamblify::RunConfig::default(),
            unamblify::RunState::Running,
            Some(1),
        );
        runs::write_status(&dir, &st).unwrap();
        fs::write(
            dir.join("metrics.jsonl"),
            "{\"step\":1,\"t\":1,\"k\":\"loss/total\",\"v\":1.0}\n",
        )
        .unwrap();

        let hub = Arc::new(Hub::default());
        let mut rx = hub.subscribe();
        let mut w = Watcher::new(runs_dir.clone(), cap, Arc::clone(&hub));
        // First tick: learn the run, publish nothing but the list.
        let n = w.tick();
        assert_eq!(n, 1);
        assert_eq!(rx.try_recv().unwrap().kind, Kind::Runs);

        // Append two metric rows (one sys), a log row, and a checkpoint.
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(dir.join("metrics.jsonl"))
            .unwrap();
        writeln!(f, "{{\"step\":2,\"t\":2,\"k\":\"loss/total\",\"v\":0.9}}").unwrap();
        writeln!(f, "{{\"step\":2,\"t\":2,\"k\":\"sys/cpu\",\"v\":50}}").unwrap();
        write!(f, "{{\"step\":3,\"t\":3,\"k\":\"loss/total\"").unwrap(); // partial
        runs::append_log(&dir, &runs::log_row(1, unamblify::LogLevel::Info, "hi")).unwrap();
        fs::create_dir_all(dir.join("checkpoints/step-000002")).unwrap();
        st.step = 2;
        st.updated = "later".to_owned();
        runs::write_status(&dir, &st).unwrap();

        let n = w.tick();
        let mut kinds = Vec::new();
        for _ in 0..n {
            kinds.push(rx.try_recv().unwrap());
        }
        let names: Vec<Kind> = kinds.iter().map(|e| e.kind).collect();
        assert!(names.contains(&Kind::Status), "{names:?}");
        assert!(names.contains(&Kind::Metric), "{names:?}");
        assert!(names.contains(&Kind::Sys), "{names:?}");
        assert!(names.contains(&Kind::Log), "{names:?}");
        assert!(names.contains(&Kind::Checkpoint), "{names:?}");
        let metric = kinds.iter().find(|e| e.kind == Kind::Metric).unwrap();
        assert_eq!(
            metric.data.as_array().unwrap().len(),
            1,
            "partial line not emitted"
        );
        assert_eq!(metric.run.as_deref(), Some("20260910-050000-a"));

        // Completing the partial line emits it on the next tick.
        writeln!(f, ",\"v\":0.8}}").unwrap();
        let n = w.tick();
        assert_eq!(n, 1);
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.kind, Kind::Metric);
        assert_eq!(ev.data[0]["step"], 3);
    }
}
