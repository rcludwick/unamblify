// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The run directory's live files (spec §7): `metrics.jsonl` (flushed
//! every 250 ms and at every checkpoint), `log.jsonl` (flushed per line),
//! `status.json` (written atomically).

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use unamblify::{LogLevel, LogRow, MetricRow, RunStatus};

use crate::time::unix_ms;

/// Metrics buffer age at which a flush is forced.
pub const FLUSH_EVERY: Duration = Duration::from_millis(250);

/// Set to `0` to stop the trainer echoing its log lines to stderr (the
/// dashboard's supervisor sets it: the trainer already writes
/// `log.jsonl` itself, and an echo would land there twice).
pub const ECHO_ENV: &str = unamblify::run::LOG_ECHO_ENV;

/// Writer for one run directory.
#[derive(Debug)]
pub struct RunWriter {
    dir: PathBuf,
    metrics: BufWriter<File>,
    log: BufWriter<File>,
    last_flush: Instant,
    seq: u64,
    /// Echo log lines to stderr (default on unless [`ECHO_ENV`] is `0`).
    /// A failed echo — the reader of stderr went away — is ignored: a
    /// closed pipe must never take the trainer down.
    pub echo: bool,
}

fn count_lines(path: &Path) -> u64 {
    File::open(path).map_or(0, |f| {
        u64::try_from(BufReader::new(f).lines().count()).unwrap_or(0)
    })
}

impl RunWriter {
    /// Open (append) the files under `dir`, creating it.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        let append = |name: &str| -> anyhow::Result<File> {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(name))
                .with_context(|| dir.join(name).display().to_string())
        };
        let seq = count_lines(&dir.join("log.jsonl"));
        Ok(Self {
            dir: dir.to_path_buf(),
            metrics: BufWriter::new(append("metrics.jsonl")?),
            log: BufWriter::new(append("log.jsonl")?),
            last_flush: Instant::now(),
            seq,
            echo: std::env::var_os(ECHO_ENV).is_none_or(|v| v != "0"),
        })
    }

    /// The run directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Append one metric sample.
    pub fn metric(&mut self, step: u64, key: &str, value: f64) -> anyhow::Result<()> {
        if !value.is_finite() {
            return Ok(());
        }
        let row = MetricRow {
            step,
            t: unix_ms(),
            k: key.to_owned(),
            v: value,
        };
        serde_json::to_writer(&mut self.metrics, &row)?;
        self.metrics.write_all(b"\n")?;
        self.maybe_flush()
    }

    /// Append several samples for the same step.
    pub fn metrics(&mut self, step: u64, rows: &[(&str, f64)]) -> anyhow::Result<()> {
        for (k, v) in rows {
            self.metric(step, k, *v)?;
        }
        Ok(())
    }

    /// Append a log line (flushed immediately).
    pub fn log(&mut self, level: LogLevel, msg: impl Into<String>) -> anyhow::Result<()> {
        let msg = msg.into();
        self.seq += 1;
        let row = LogRow {
            seq: self.seq,
            t: unix_ms(),
            level,
            msg,
        };
        if self.echo {
            let _ = writeln!(io::stderr().lock(), "[{}] {}", row.level, row.msg);
        }
        serde_json::to_writer(&mut self.log, &row)?;
        self.log.write_all(b"\n")?;
        self.log.flush()?;
        Ok(())
    }

    /// Flush metrics if the last flush is older than [`FLUSH_EVERY`].
    pub fn maybe_flush(&mut self) -> anyhow::Result<()> {
        if self.last_flush.elapsed() >= FLUSH_EVERY {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush everything now.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        self.metrics.flush()?;
        self.log.flush()?;
        self.last_flush = Instant::now();
        Ok(())
    }

    /// Write `status.json` atomically.
    pub fn status(&self, status: &RunStatus) -> anyhow::Result<()> {
        write_status(&self.dir, status)
    }
}

/// Write `status.json` atomically into `dir`.
pub fn write_status(dir: &Path, status: &RunStatus) -> anyhow::Result<()> {
    let tmp = dir.join(".status.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(status)?)?;
    std::fs::rename(&tmp, dir.join("status.json"))?;
    Ok(())
}

/// Drop every `metrics.jsonl` row past `step` (temp file + rename), so a
/// run resumed into its own directory from an older checkpoint does not
/// log the steps between that checkpoint and where it died twice.
/// Returns the number of rows removed.
pub fn truncate_metrics_after(dir: &Path, step: u64) -> anyhow::Result<usize> {
    let p = dir.join("metrics.jsonl");
    let f = match File::open(&p) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e).with_context(|| p.display().to_string()),
    };
    let mut kept = Vec::new();
    let mut removed = 0usize;
    for line in BufReader::new(f).lines() {
        let line = line?;
        match serde_json::from_str::<MetricRow>(&line) {
            Ok(row) if row.step > step => removed += 1,
            Ok(_) => kept.push(line),
            // A torn last line is dropped with the rest.
            Err(_) => removed += 1,
        }
    }
    if removed == 0 {
        return Ok(0);
    }
    let tmp = dir.join(".metrics.jsonl.tmp");
    let mut text = kept.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &p)?;
    Ok(removed)
}

/// Read `metrics.jsonl` back (tests and the eval summary).
pub fn read_metrics(dir: &Path) -> anyhow::Result<Vec<MetricRow>> {
    let p = dir.join("metrics.jsonl");
    let f = File::open(&p).with_context(|| p.display().to_string())?;
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            rows.push(serde_json::from_str(&line)?);
        }
    }
    Ok(rows)
}

impl Drop for RunWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unamblify::RunState;

    #[test]
    fn files_are_written_and_seq_continues_on_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("run");
        {
            let mut w = RunWriter::open(&dir).unwrap();
            w.echo = false;
            w.metric(1, "loss/total", 0.5).unwrap();
            w.metric(1, "loss/nan", f64::NAN).unwrap();
            w.log(LogLevel::Info, "hello").unwrap();
            w.status(&RunStatus {
                status: RunState::Running,
                step: 1,
                total_steps: 2,
                started: "s".to_owned(),
                updated: "u".to_owned(),
                pid: Some(1),
                device: "cpu".to_owned(),
                host: "h".to_owned(),
                best: None,
            })
            .unwrap();
            w.flush().unwrap();
        }
        let rows = read_metrics(&dir).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].k, "loss/total");
        let status: RunStatus =
            serde_json::from_str(&std::fs::read_to_string(dir.join("status.json")).unwrap())
                .unwrap();
        assert_eq!(status.status, RunState::Running);
        let mut w = RunWriter::open(&dir).unwrap();
        w.echo = false;
        w.log(LogLevel::Warn, "again").unwrap();
        drop(w);
        let log = std::fs::read_to_string(dir.join("log.jsonl")).unwrap();
        let rows: Vec<LogRow> = log
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].seq, 2);
        assert_eq!(rows[1].level, LogLevel::Warn);
    }

    #[test]
    fn metrics_past_a_step_are_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("run");
        {
            let mut w = RunWriter::open(&dir).unwrap();
            w.echo = false;
            for s in 1..=6 {
                w.metric(s, "loss/total", 0.5).unwrap();
            }
        }
        assert_eq!(truncate_metrics_after(&dir, 4).unwrap(), 2);
        assert_eq!(truncate_metrics_after(&dir, 4).unwrap(), 0);
        let rows = read_metrics(&dir).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.step).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            truncate_metrics_after(&tmp.path().join("none"), 1).unwrap(),
            0
        );
    }
}
