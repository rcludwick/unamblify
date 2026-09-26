// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Host samples on disk, so a day of history survives a restart.
//!
//! The chart exists to show a climb over hours. An in-memory ring loses
//! that every time the server bounces, which is exactly when a drive is
//! most likely to have just misbehaved. One JSONL row per sample is
//! enough: appending is cheap, and a corrupt tail line is skipped
//! rather than losing the file.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How much history is kept, in seconds.
pub const KEEP_S: u64 = 24 * 60 * 60;

/// One moment of host state: temperatures plus CPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// Unix seconds.
    pub t: u64,
    /// Whole-host CPU use, percent.
    pub cpu: f32,
    /// One-minute load average.
    pub load1: f64,
    /// GPU use, percent, when the accelerator reports it.
    #[serde(default)]
    pub gpu: Option<f32>,
    /// GPU memory in use, GiB.
    #[serde(default)]
    pub gpu_mem_gb: Option<f64>,
    /// Hottest sensor per device node.
    pub temps: BTreeMap<String, i32>,
    /// Read / write rate per device node, MiB/s. Absent in rows written
    /// before 2026-09-21, which load with none.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub io: BTreeMap<String, crate::diskio::Rate>,
}

/// Where the history file lives under the data root.
#[must_use]
pub fn path(data_root: &Path) -> PathBuf {
    data_root.join("cache").join("host-metrics.jsonl")
}

/// Append one row. Failures are ignored: losing a sample must never
/// take the server down.
pub fn append(path: &Path, row: &Row) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
        && let Ok(line) = serde_json::to_string(row)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Every row newer than `cutoff`, oldest first. A line that does not
/// parse is skipped.
#[must_use]
pub fn load(path: &Path, cutoff: u64) -> Vec<Row> {
    let Ok(f) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str::<Row>(&l).ok())
        .filter(|r| r.t >= cutoff)
        .collect()
}

/// Rewrite the file with just `rows`, so it cannot grow without bound
/// across restarts.
pub fn compact(path: &Path, rows: &[Row]) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("jsonl.tmp");
    let Ok(mut f) = File::create(&tmp) else {
        return;
    };
    for r in rows {
        if let Ok(line) = serde_json::to_string(r) {
            let _ = writeln!(f, "{line}");
        }
    }
    drop(f);
    let _ = std::fs::rename(&tmp, path);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(t: u64, cpu: f32) -> Row {
        Row {
            t,
            cpu,
            load1: 1.0,
            gpu: Some(50.0),
            gpu_mem_gb: Some(2.0),
            temps: BTreeMap::from([("/dev/disk8".to_owned(), 59)]),
            io: BTreeMap::from([(
                "/dev/disk8".to_owned(),
                crate::diskio::Rate { r: 120.5, w: 3.0 },
            )]),
        }
    }

    #[test]
    fn a_round_trip_keeps_what_is_new_enough() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(dir.path());
        append(&p, &row(100, 10.0));
        append(&p, &row(200, 20.0));
        let kept = load(&p, 150);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].t, 200);
        assert_eq!(kept[0].temps["/dev/disk8"], 59);
    }

    #[test]
    fn a_corrupt_line_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(dir.path());
        append(&p, &row(100, 10.0));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{ this is not json\n")
            .unwrap();
        append(&p, &row(300, 30.0));
        let kept = load(&p, 0);
        assert_eq!(kept.len(), 2, "the good rows survive a torn tail");
    }

    #[test]
    fn compaction_drops_the_old_rows() {
        let dir = tempfile::tempdir().unwrap();
        let p = path(dir.path());
        for t in [10, 20, 30] {
            append(&p, &row(t, 1.0));
        }
        let keep = load(&p, 20);
        compact(&p, &keep);
        assert_eq!(load(&p, 0).len(), 2);
    }
}
