// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Reading and writing `runs/<run_id>/` (spec §7) with the core crate's
//! types: `status.json` ([`RunStatus`]), `config.toml` ([`RunConfig`]),
//! `metrics.jsonl` ([`MetricRow`]), `log.jsonl` ([`LogRow`]) and the
//! `checkpoints/step-NNNNNN/` tree. Everything is plain blocking std I/O;
//! handlers call it through `spawn_blocking` where the files can be large.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use unamblify::{LogLevel, LogRow, MetricRow, RunConfig, RunState, RunStatus};

use crate::clock;
use crate::downsample::Series;
use crate::error::{Result, WebError};

/// A run id, clip name or checkpoint name is a single path component of
/// safe characters (`@` for a clip's `<clip>@<mode>`). Anything else is
/// refused before it touches the disk.
pub fn validate_component(what: &str, s: &str) -> Result<()> {
    let ok = !s.is_empty()
        && s.len() <= 200
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'+' | b'@'));
    if ok {
        Ok(())
    } else {
        Err(WebError::BadRequest(format!("invalid {what}: {s:?}")))
    }
}

/// Run name as it may appear in an id: the same alphabet minus `.`.
pub fn validate_name(name: &str) -> Result<()> {
    validate_component("run name", name)?;
    if name.contains('.') {
        return Err(WebError::BadRequest(format!(
            "invalid run name: {name:?} (no dots)"
        )));
    }
    Ok(())
}

/// The run directory for an id, after validation.
pub fn run_dir(runs_dir: &Path, id: &str) -> Result<PathBuf> {
    validate_component("run id", id)?;
    let dir = runs_dir.join(id);
    if dir.is_dir() {
        Ok(dir)
    } else {
        Err(WebError::NotFound(format!("run {id}")))
    }
}

/// Write `data` to `path` atomically (temp file + rename).
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// `status.json`, if present and parseable.
#[must_use]
pub fn read_status(dir: &Path) -> Option<RunStatus> {
    let text = fs::read(dir.join("status.json")).ok()?;
    serde_json::from_slice(&text).ok()
}

/// Write `status.json`.
pub fn write_status(dir: &Path, status: &RunStatus) -> Result<()> {
    write_atomic(
        &dir.join("status.json"),
        &serde_json::to_vec_pretty(status)?,
    )
}

/// `config.toml` text and parsed form.
pub fn read_config(dir: &Path) -> Result<(String, RunConfig)> {
    let text = fs::read_to_string(dir.join("config.toml"))?;
    let cfg = RunConfig::from_toml(&text)?;
    Ok((text, cfg))
}

/// The run name: the config's `name`, else the id minus its timestamp.
#[must_use]
pub fn name_of(id: &str, cfg: Option<&RunConfig>) -> String {
    if let Some(c) = cfg {
        return c.name.clone();
    }
    // `YYYYMMDD-HHMMSS-<name>`
    let mut parts = id.splitn(3, '-');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(_), Some(_), Some(rest)) => rest.to_owned(),
        _ => id.to_owned(),
    }
}

/// One row of `GET /api/runs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Run id (directory name).
    pub id: String,
    /// Run name.
    pub name: String,
    /// `status.json`, when it exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<RunStatus>,
    /// Model profile / lookahead / device / mode, for the list view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<RunConfig>,
    /// Whether this server's supervisor holds the trainer process.
    pub supervised: bool,
    /// Last loss/total, when metrics exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f64>,
    /// Number of checkpoints on disk.
    pub checkpoints: usize,
}

/// Enumerate run directories, newest id first.
pub fn list_ids(runs_dir: &Path) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    let rd = match fs::read_dir(runs_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ids),
        Err(e) => return Err(e.into()),
    };
    for entry in rd {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if validate_component("run id", &name).is_ok() && !name.starts_with('.') {
            ids.push(name);
        }
    }
    ids.sort_unstable_by(|a, b| b.cmp(a));
    Ok(ids)
}

/// Build the list row for one run.
#[must_use]
pub fn summarize(runs_dir: &Path, id: &str, supervised: bool) -> RunSummary {
    let dir = runs_dir.join(id);
    let cfg = read_config(&dir).ok().map(|(_, c)| c);
    let status = read_status(&dir);
    let last_loss = last_metric(&dir, "loss/total");
    RunSummary {
        id: id.to_owned(),
        name: name_of(id, cfg.as_ref()),
        status,
        config: cfg,
        supervised,
        last_loss,
        checkpoints: list_checkpoints(&dir).len(),
    }
}

/// Last value of one key in `metrics.jsonl`, scanning backwards over the
/// file's tail (at most the last 64 KiB).
#[must_use]
pub fn last_metric(dir: &Path, key: &str) -> Option<f64> {
    let tail = read_tail_bytes(&dir.join("metrics.jsonl"), 64 * 1024).ok()?;
    let text = String::from_utf8_lossy(&tail);
    text.lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<MetricRow>(l).ok())
        .find(|r| r.k == key)
        .map(|r| r.v)
}

/// The last `max` bytes of a file, starting at a line boundary.
pub fn read_tail_bytes(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut f = fs::File::open(path)?;
    let len = f.metadata()?.len();
    let start = len.saturating_sub(max);
    f.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if start > 0
        && let Some(nl) = buf.iter().position(|&b| b == b'\n')
    {
        buf.drain(..=nl);
    }
    Ok(buf)
}

/// Read whole lines from `path` starting at byte `offset`. Returns the
/// lines and the new offset (the end of the last complete line), so a
/// half-written trailing line is picked up on the next call.
pub fn read_new_lines(path: &Path, offset: u64) -> Result<(Vec<String>, u64)> {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), offset)),
        Err(e) => return Err(e.into()),
    };
    let len = f.metadata()?.len();
    // Truncated (rewritten) file: start over.
    let offset = if offset > len { 0 } else { offset };
    if len == offset {
        return Ok((Vec::new(), offset));
    }
    f.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(f);
    let mut lines = Vec::new();
    let mut consumed = offset;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        if !buf.ends_with('\n') {
            break;
        }
        consumed += n as u64;
        let line = buf.trim_end();
        if !line.is_empty() {
            lines.push(line.to_owned());
        }
    }
    Ok((lines, consumed))
}

/// Per-key series read from `metrics.jsonl`. `keys` empty = all keys;
/// `after_step` skips rows at or below it.
pub fn read_metrics(
    dir: &Path,
    keys: &[String],
    after_step: Option<u64>,
) -> Result<BTreeMap<String, Series>> {
    let mut out: BTreeMap<String, Series> = BTreeMap::new();
    let f = match fs::File::open(dir.join("metrics.jsonl")) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    for line in BufReader::new(f).lines() {
        let line = line?;
        let Ok(row) = serde_json::from_str::<MetricRow>(&line) else {
            continue;
        };
        if after_step.is_some_and(|a| row.step <= a) {
            continue;
        }
        if !keys.is_empty() && !keys.contains(&row.k) {
            continue;
        }
        out.entry(row.k).or_default().push(row.step, row.t, row.v);
    }
    Ok(out)
}

/// All metric keys present (a full parse; handlers use
/// [`MetricKeysTrack`] instead, which only reads what was appended).
pub fn metric_keys(dir: &Path) -> Result<Vec<String>> {
    Ok(read_metrics(dir, &[], None)?.into_keys().collect())
}

/// The metric keys of one run, kept up to date by parsing only the bytes
/// appended to `metrics.jsonl` since the last look — `GET /api/runs/{id}`
/// is hit on every status event of every live run, and the file grows to
/// tens of MB over a long run.
#[derive(Debug, Default, Clone)]
pub struct MetricKeysTrack {
    offset: u64,
    keys: std::collections::BTreeSet<String>,
}

impl MetricKeysTrack {
    /// Read whatever is new and return every key seen so far, sorted.
    pub fn update(&mut self, dir: &Path) -> Result<Vec<String>> {
        let (lines, off) = read_new_lines(&dir.join("metrics.jsonl"), self.offset)?;
        if off < self.offset {
            // Rewritten (a resume truncated it): start over.
            self.keys.clear();
        }
        self.offset = off;
        for l in &lines {
            if let Ok(row) = serde_json::from_str::<MetricRow>(l) {
                self.keys.insert(row.k);
            }
        }
        Ok(self.keys.iter().cloned().collect())
    }
}

/// Log rows: the last `tail` rows, or every row with `seq > after`.
pub fn read_logs(dir: &Path, tail: Option<usize>, after: Option<u64>) -> Result<Vec<LogRow>> {
    let path = dir.join("log.jsonl");
    if let (Some(n), None) = (tail, after) {
        return read_log_tail(&path, n);
    }
    let f = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let after = after.unwrap_or(0);
    Ok(BufReader::new(f)
        .lines()
        .map_while(std::result::Result::ok)
        .filter_map(|l| serde_json::from_str::<LogRow>(&l).ok())
        .filter(|r| r.seq > after)
        .collect())
}

/// The last `n` rows, reading a growing tail window until enough lines
/// (or the whole file) came back.
fn read_log_tail(path: &Path, n: usize) -> Result<Vec<LogRow>> {
    // Rough: 256 bytes per row.
    let mut want = (n as u64).saturating_mul(256).max(4096);
    loop {
        let bytes = match read_tail_bytes(path, want) {
            Ok(b) => b,
            Err(WebError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Vec::new());
            }
            Err(e) => return Err(e),
        };
        let text = String::from_utf8_lossy(&bytes);
        let rows: Vec<LogRow> = text
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let file_len = fs::metadata(path)?.len();
        if rows.len() >= n || want >= file_len {
            let skip = rows.len().saturating_sub(n);
            return Ok(rows.into_iter().skip(skip).collect());
        }
        want = want.saturating_mul(4);
    }
}

/// Highest `seq` in `log.jsonl` (0 when absent).
#[must_use]
pub fn last_log_seq(dir: &Path) -> u64 {
    read_tail_bytes(&dir.join("log.jsonl"), 16 * 1024)
        .ok()
        .and_then(|b| {
            String::from_utf8_lossy(&b)
                .lines()
                .rev()
                .find_map(|l| serde_json::from_str::<LogRow>(l).ok())
                .map(|r| r.seq)
        })
        .unwrap_or(0)
}

/// Append one row to `log.jsonl`.
pub fn append_log(dir: &Path, row: &LogRow) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("log.jsonl"))?;
    let mut line = serde_json::to_vec(row)?;
    line.push(b'\n');
    f.write_all(&line)?;
    Ok(())
}

/// A log row stamped now.
#[must_use]
pub fn log_row(seq: u64, level: LogLevel, msg: impl Into<String>) -> LogRow {
    LogRow {
        seq,
        t: clock::now_ms(),
        level,
        msg: msg.into(),
    }
}

/// One rendered eval clip inside a checkpoint's `audio/` directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clip {
    /// Clip name (`<clip>` in `<clip>.clean.wav`).
    pub name: String,
    /// Which of `clean`, `degraded`, `out` exist.
    pub variants: Vec<String>,
    /// Whether `<clip>.spec.json` exists.
    pub spec: bool,
}

/// One `checkpoints/step-NNNNNN/`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Step number parsed from the directory name.
    pub step: u64,
    /// Directory name (`step-000500`).
    pub dir: String,
    /// `model.safetensors` present.
    pub model: bool,
    /// `optim.safetensors` present.
    pub optim: bool,
    /// `meta.json` contents, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Rendered clips.
    pub clips: Vec<Clip>,
    /// Modified time of the directory, Unix ms.
    pub mtime_ms: u64,
}

/// Every checkpoint, ascending by step.
#[must_use]
pub fn list_checkpoints(dir: &Path) -> Vec<Checkpoint> {
    let Ok(rd) = fs::read_dir(dir.join("checkpoints")) else {
        return Vec::new();
    };
    let mut out: Vec<Checkpoint> = rd
        .filter_map(std::result::Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let step = name.strip_prefix("step-")?.parse::<u64>().ok()?;
            let p = e.path();
            if !p.is_dir() {
                return None;
            }
            let mtime_ms = e
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
            let meta = fs::read(p.join("meta.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            Some(Checkpoint {
                step,
                dir: name,
                model: p.join("model.safetensors").is_file(),
                optim: p.join("optim.safetensors").is_file(),
                meta,
                clips: list_clips(&p.join("audio")),
                mtime_ms,
            })
        })
        .collect();
    out.sort_by_key(|c| c.step);
    out
}

/// The latest checkpoint directory, for `--resume`.
#[must_use]
pub fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    list_checkpoints(dir)
        .into_iter()
        .rfind(|c| c.model)
        .map(|c| dir.join("checkpoints").join(c.dir))
}

/// Clips in an `audio/` directory: `<clip>.{clean,degraded,out}.wav` and
/// `<clip>.spec.json`.
#[must_use]
pub fn list_clips(audio_dir: &Path) -> Vec<Clip> {
    let Ok(rd) = fs::read_dir(audio_dir) else {
        return Vec::new();
    };
    let mut map: BTreeMap<String, Clip> = BTreeMap::new();
    for e in rd.filter_map(std::result::Result::ok) {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_suffix(".spec.json") {
            map.entry(stem.to_owned())
                .or_insert_with(|| Clip {
                    name: stem.to_owned(),
                    variants: Vec::new(),
                    spec: false,
                })
                .spec = true;
        } else if let Some(stem) = name.strip_suffix(".wav") {
            let Some((clip, variant)) = stem.rsplit_once('.') else {
                continue;
            };
            if !matches!(variant, "clean" | "degraded" | "out") {
                continue;
            }
            map.entry(clip.to_owned())
                .or_insert_with(|| Clip {
                    name: clip.to_owned(),
                    variants: Vec::new(),
                    spec: false,
                })
                .variants
                .push(variant.to_owned());
        }
    }
    let mut out: Vec<Clip> = map.into_values().collect();
    for c in &mut out {
        c.variants.sort_by_key(|v| match v.as_str() {
            "clean" => 0,
            "degraded" => 1,
            _ => 2,
        });
    }
    out
}

/// A fresh `status.json` for a just-created run.
#[must_use]
pub fn new_status(cfg: &RunConfig, state: RunState, pid: Option<u32>) -> RunStatus {
    let now = clock::rfc3339_now();
    RunStatus {
        status: state,
        step: 0,
        total_steps: cfg.train.steps,
        started: now.clone(),
        updated: now,
        pid,
        device: cfg.train.device.clone(),
        host: crate::sys::hostname(),
        best: None,
    }
}

/// Create `runs/<id>/` for `cfg`, choosing a unique id, and write the
/// resolved `config.toml` plus a `queued` `status.json`.
pub fn create_run(runs_dir: &Path, cfg: &RunConfig) -> Result<(String, PathBuf)> {
    validate_name(&cfg.name)?;
    fs::create_dir_all(runs_dir)?;
    let base = clock::run_id(clock::now_s(), &cfg.name);
    let mut id = base.clone();
    let mut n = 1;
    while runs_dir.join(&id).exists() {
        n += 1;
        id = format!("{base}-{n}");
    }
    let dir = runs_dir.join(&id);
    fs::create_dir(&dir)?;
    fs::write(dir.join("config.toml"), cfg.to_toml()?)?;
    write_status(&dir, &new_status(cfg, RunState::Queued, None))?;
    Ok((id, dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_are_validated() {
        assert!(validate_component("id", "20260910-050000-smoke").is_ok());
        assert!(validate_component("clip", "vctk_p228_001_mic2@codec2-3200").is_ok());
        assert!(validate_component("id", "../x").is_err());
        assert!(validate_component("id", "a/b").is_err());
        assert!(validate_component("id", "").is_err());
        assert!(validate_component("id", "..").is_err());
        assert!(validate_name("with.dot").is_err());
        assert!(validate_name("ok-name_1").is_ok());
    }

    #[test]
    fn name_falls_back_to_the_id_tail() {
        assert_eq!(name_of("20260910-050000-seed-dstar", None), "seed-dstar");
        assert_eq!(name_of("weird", None), "weird");
    }

    #[test]
    fn new_lines_are_read_incrementally() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("m.jsonl");
        fs::write(&p, "a\nb\npartial").unwrap();
        let (lines, off) = read_new_lines(&p, 0).unwrap();
        assert_eq!(lines, vec!["a", "b"]);
        assert_eq!(off, 4);
        fs::write(&p, "a\nb\npartial-done\nc\n").unwrap();
        let (lines, off2) = read_new_lines(&p, off).unwrap();
        assert_eq!(lines, vec!["partial-done", "c"]);
        assert_eq!(off2, 19);
        let (lines, _) = read_new_lines(&p, off2).unwrap();
        assert!(lines.is_empty());
        // Truncation restarts from zero.
        fs::write(&p, "z\n").unwrap();
        let (lines, _) = read_new_lines(&p, off2).unwrap();
        assert_eq!(lines, vec!["z"]);
    }

    #[test]
    fn clips_and_checkpoints_are_enumerated() {
        let tmp = tempfile::tempdir().unwrap();
        let ck = tmp.path().join("checkpoints/step-000500/audio");
        fs::create_dir_all(&ck).unwrap();
        for f in [
            "p225_001@dstar.clean.wav",
            "p225_001@dstar.degraded.wav",
            "p225_001@dstar.out.wav",
            "p225_001@dstar.spec.json",
            "other.out.wav",
            "junk.txt",
        ] {
            fs::write(ck.join(f), b"x").unwrap();
        }
        fs::write(
            tmp.path().join("checkpoints/step-000500/model.safetensors"),
            b"x",
        )
        .unwrap();
        fs::create_dir_all(tmp.path().join("checkpoints/not-a-step")).unwrap();
        let cks = list_checkpoints(tmp.path());
        assert_eq!(cks.len(), 1);
        assert_eq!(cks[0].step, 500);
        assert!(cks[0].model && !cks[0].optim);
        assert_eq!(cks[0].clips.len(), 2);
        assert_eq!(cks[0].clips[0].name, "other");
        assert_eq!(cks[0].clips[1].name, "p225_001@dstar");
        assert_eq!(cks[0].clips[1].variants, vec!["clean", "degraded", "out"]);
        assert!(cks[0].clips[1].spec);
        assert!(
            latest_checkpoint(tmp.path())
                .unwrap()
                .ends_with("step-000500")
        );
    }

    #[test]
    fn metric_keys_track_reads_only_the_new_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("metrics.jsonl");
        fs::write(&p, "{\"step\":1,\"t\":1,\"k\":\"loss/total\",\"v\":1.0}\n").unwrap();
        let mut t = MetricKeysTrack::default();
        assert_eq!(t.update(tmp.path()).unwrap(), vec!["loss/total"]);
        let len = fs::metadata(&p).unwrap().len();
        assert_eq!(t.offset, len);
        let mut f = fs::OpenOptions::new().append(true).open(&p).unwrap();
        writeln!(f, "{{\"step\":2,\"t\":2,\"k\":\"lr\",\"v\":0.1}}").unwrap();
        assert_eq!(t.update(tmp.path()).unwrap(), vec!["loss/total", "lr"]);
        assert_eq!(metric_keys(tmp.path()).unwrap(), vec!["loss/total", "lr"]);
        // Truncated by a resume: keys are rebuilt from what is left.
        fs::write(&p, "{\"step\":1,\"t\":1,\"k\":\"eval/lsd\",\"v\":1.0}\n").unwrap();
        assert_eq!(t.update(tmp.path()).unwrap(), vec!["eval/lsd"]);
        assert!(
            MetricKeysTrack::default()
                .update(&tmp.path().join("none"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn logs_tail_and_after() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 1..=50u64 {
            append_log(tmp.path(), &log_row(i, LogLevel::Info, format!("m{i}"))).unwrap();
        }
        let t = read_logs(tmp.path(), Some(5), None).unwrap();
        assert_eq!(t.len(), 5);
        assert_eq!(t[0].seq, 46);
        let a = read_logs(tmp.path(), None, Some(48)).unwrap();
        assert_eq!(a.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![49, 50]);
        assert_eq!(last_log_seq(tmp.path()), 50);
        assert!(
            read_logs(&tmp.path().join("nope"), Some(5), None)
                .unwrap()
                .is_empty()
        );
    }
}
