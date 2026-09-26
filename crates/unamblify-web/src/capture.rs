// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Capture control from the dashboard (spec §3): read every mode's
//! `captured/<mode>/status.json`, write `control.json`, and start the
//! harness under the supervisor. The status file is the harness's own
//! (`{state, mode, port(s), done, failed, total, frames_s, utt_per_hour,
//! eta_s, current_key, started, updated, canary_ok, prodid, version}`); it
//! is passed through as JSON rather than re-typed here so the harness may
//! add fields. A software mode (Codec 2) is flagged so the page shows
//! worker threads instead of a port and offers `--jobs`. The decode-only
//! siblings (`captured/<mode>+<kind>/`, written by `unamblify augment`)
//! are listed read-only after the base modes: their status and log are
//! shown, nothing is started or controlled from here.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use serde_json::{Value, json};
use unamblify::aug::{capture_dir_name, parse_capture_dir_name};
use unamblify::{LogRow, VocoderMode};

use crate::error::{Result, WebError};
use crate::runs;
use crate::supervisor::{Supervisor, capture_key};

/// `captured/<mode>/`.
#[must_use]
pub fn mode_dir(captured_dir: &Path, mode: VocoderMode) -> PathBuf {
    captured_dir.join(mode.as_str())
}

/// The capture set directories present under `captured/` (names that
/// parse as a mode or `<mode>+<kind>`), sorted by name.
#[must_use]
pub fn capture_set_names(captured_dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(captured_dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| parse_capture_dir_name(n).is_ok())
        .collect();
    names.sort();
    names
}

/// One capture set's row in `GET /api/capture`.
#[derive(Debug, Clone, Serialize)]
// Independent flags of a view row, not a state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct ModeView {
    /// The set's name: the mode, or `<mode>+<kind>` for a sibling.
    pub name: String,
    /// The mode.
    pub mode: VocoderMode,
    /// The mode's name for a person (`VocoderMode::label`: `D-STAR`,
    /// `YSF/DMR`, …); `mode` stays the identifier.
    pub label: String,
    /// The sibling kind (`drops`, `ber`); absent for a base capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// A decode-only sibling: shown, never started or controlled here.
    pub readonly: bool,
    /// The codec family (`ambe` | `codec2`).
    pub family: String,
    /// Whether the codec runs in software: no port, `--jobs` threads.
    pub software: bool,
    /// Milliseconds per channel frame (20, or 40 for Codec 2 1600).
    pub frame_ms: u32,
    /// `status.json` contents, or `null`.
    pub status: Option<Value>,
    /// `control.json` state (`run` | `pause` | `stop`), or `null`.
    pub control: Option<String>,
    /// Whether this server's supervisor holds the harness process.
    pub supervised: bool,
    /// The supervised pid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Whether the status's pid (or the supervised pid) is alive.
    pub alive: bool,
    /// Alive but on its way out: `control.json` says `stop` (the harness
    /// finishes the utterance in flight first), or `status.json` is
    /// already final and the process has not exited yet. Shown as
    /// *stopping*, with no controls: a second stop changes nothing and a
    /// start is refused until the lock is released.
    pub stopping: bool,
    /// Rows in `manifest.jsonl` (done count from disk, independent of the
    /// harness's own `done`).
    pub manifest_rows: u64,
    /// Last rows of `log.jsonl`.
    pub log_tail: Vec<LogRow>,
}

/// A control verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Continue.
    Run,
    /// Finish the utterance in flight, then wait.
    Pause,
    /// Finish the utterance in flight, then exit 0.
    Stop,
}

impl Control {
    /// The `state` string in `control.json`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Pause => "pause",
            Self::Stop => "stop",
        }
    }

    /// Parse a URL verb (`pause` | `resume` | `stop`).
    pub fn from_verb(verb: &str) -> Result<Self> {
        match verb {
            "pause" => Ok(Self::Pause),
            "resume" => Ok(Self::Run),
            "stop" => Ok(Self::Stop),
            other => Err(WebError::BadRequest(format!(
                "unknown capture verb {other:?} (pause | resume | stop | start)"
            ))),
        }
    }
}

/// Parse a mode path segment.
pub fn parse_mode(s: &str) -> Result<VocoderMode> {
    s.parse::<VocoderMode>()
        .map_err(|e| WebError::BadRequest(e.to_string()))
}

/// Write `control.json`.
pub fn write_control(captured_dir: &Path, mode: VocoderMode, c: Control) -> Result<()> {
    let dir = mode_dir(captured_dir, mode);
    std::fs::create_dir_all(&dir)?;
    runs::write_atomic(
        &dir.join("control.json"),
        &serde_json::to_vec(&json!({ "state": c.as_str() }))?,
    )
}

fn read_json(p: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

fn count_lines(p: &Path) -> u64 {
    std::fs::read(p).map_or(0, |b| {
        b.split(|&c| c == b'\n').count().saturating_sub(1) as u64
    })
}

/// The view of one mode's base capture.
#[must_use]
pub fn view(captured_dir: &Path, sup: &Supervisor, mode: VocoderMode, log_tail: usize) -> ModeView {
    view_set(captured_dir, sup, mode, None, log_tail)
}

/// The view of one capture set.
#[must_use]
pub fn view_set(
    captured_dir: &Path,
    sup: &Supervisor,
    mode: VocoderMode,
    kind: Option<unamblify::AugKind>,
    log_tail: usize,
) -> ModeView {
    let name = capture_dir_name(mode, kind);
    let dir = captured_dir.join(&name);
    let status = read_json(&dir.join("status.json"));
    let control = read_json(&dir.join("control.json"))
        .and_then(|v| v.get("state").and_then(Value::as_str).map(str::to_owned));
    let key = capture_key(mode);
    let proc_ = sup.get(&key);
    let supervised = proc_
        .as_ref()
        .is_some_and(|p| crate::supervisor::pid_alive(p.pid));
    let status_pid = status
        .as_ref()
        .and_then(|s| s.get("pid"))
        .and_then(Value::as_u64)
        .and_then(|p| u32::try_from(p).ok());
    let alive = supervised || status_pid.is_some_and(crate::supervisor::pid_alive);
    let finished = status
        .as_ref()
        .and_then(|s| s.get("state"))
        .and_then(Value::as_str)
        .is_some_and(|s| matches!(s, "stopped" | "done" | "error"));
    let stopping = alive && (control.as_deref() == Some("stop") || finished);
    ModeView {
        name,
        mode,
        label: mode.label().to_owned(),
        kind: kind.map(|k| k.to_string()),
        readonly: kind.is_some(),
        family: mode.family().to_string(),
        software: mode.is_software(),
        frame_ms: mode.frame_ms(),
        status,
        control,
        supervised,
        pid: proc_.map(|p| p.pid).or(status_pid),
        alive,
        stopping,
        manifest_rows: count_lines(&dir.join("manifest.jsonl")),
        log_tail: runs::read_logs(&dir, Some(log_tail), None).unwrap_or_default(),
    }
}

/// Every mode's view, then every decode-only sibling present, read-only.
#[must_use]
pub fn view_all(captured_dir: &Path, sup: &Supervisor, log_tail: usize) -> Vec<ModeView> {
    let mut out: Vec<ModeView> = VocoderMode::ALL
        .into_iter()
        .map(|m| view(captured_dir, sup, m, log_tail))
        .collect();
    for name in capture_set_names(captured_dir) {
        if let Ok((mode, Some(kind))) = parse_capture_dir_name(&name) {
            out.push(view_set(captured_dir, sup, mode, Some(kind), log_tail));
        }
    }
    out
}

// ── Voice sets: which corpora each capture set holds ───────────────────

/// One corpus's share of a capture set.
#[derive(Clone, Serialize)]
pub struct CorpusCount {
    /// Corpus id (the key's leading segment): `libritts_r`, `vctk`, ….
    pub corpus: String,
    /// Utterances of this corpus captured in the set.
    pub utterances: u64,
    /// Their audio duration in hours (frames x the mode's frame length).
    pub hours: f64,
}

/// The voice-set breakdown of one capture set.
#[derive(Clone, Serialize)]
pub struct VoiceSets {
    /// Set directory name (`dstar`, `codec2-3200+drops`, …).
    pub name: String,
    /// The mode.
    pub mode: VocoderMode,
    /// Human label (`D-STAR`, …).
    pub label: String,
    /// Sibling kind (`drops`), absent for a base capture.
    pub kind: Option<String>,
    /// A decode-only sibling.
    pub readonly: bool,
    /// Total utterances captured in the set.
    pub utterances: u64,
    /// Total hours captured in the set.
    pub hours: f64,
    /// Per-corpus counts, largest first.
    pub corpora: Vec<CorpusCount>,
}

/// Only the two manifest fields the voice-set counts need.
#[derive(serde::Deserialize)]
struct RowKf {
    key: String,
    #[serde(default)]
    frames: u32,
}

/// A cached aggregate of one manifest, valid while its mtime and length
/// are unchanged. Entries are `(corpus, utterances, frames)`.
struct VsCacheEntry {
    mtime: std::time::SystemTime,
    len: u64,
    by_corpus: Vec<(String, u64, u64)>,
}

fn vs_cache() -> &'static Mutex<HashMap<PathBuf, VsCacheEntry>> {
    static C: OnceLock<Mutex<HashMap<PathBuf, VsCacheEntry>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Aggregate a capture manifest by corpus, `(corpus, utterances, frames)`
/// per entry. Cached on the file's mtime and length, so a manifest that
/// has not changed since the last call is not re-read — a big capture set
/// (222 k rows) is parsed once, not on every request.
fn aggregate_manifest(path: &Path) -> Vec<(String, u64, u64)> {
    let meta = std::fs::metadata(path).ok();
    let mtime = meta.as_ref().and_then(|m| m.modified().ok());
    let len = meta.as_ref().map_or(0, std::fs::Metadata::len);
    if let Some(mtime) = mtime
        && let Ok(cache) = vs_cache().lock()
        && let Some(e) = cache.get(path)
        && e.mtime == mtime
        && e.len == len
    {
        return e.by_corpus.clone();
    }
    let mut map: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    if let Ok(file) = std::fs::File::open(path) {
        use std::io::BufRead;
        for line in std::io::BufReader::new(file)
            .lines()
            .map_while(std::result::Result::ok)
        {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(row) = serde_json::from_str::<RowKf>(line) {
                let corpus = unamblify::key::corpus_of(&row.key)
                    .unwrap_or("?")
                    .to_owned();
                let e = map.entry(corpus).or_insert((0, 0));
                e.0 += 1;
                e.1 += u64::from(row.frames);
            }
        }
    }
    let out: Vec<(String, u64, u64)> = map.into_iter().map(|(c, (u, f))| (c, u, f)).collect();
    if let Some(mtime) = mtime
        && let Ok(mut cache) = vs_cache().lock()
    {
        cache.insert(
            path.to_path_buf(),
            VsCacheEntry {
                mtime,
                len,
                by_corpus: out.clone(),
            },
        );
    }
    out
}

/// The voice-set breakdown of one capture set.
#[must_use]
fn voiceset_of(
    captured_dir: &Path,
    mode: VocoderMode,
    kind: Option<unamblify::AugKind>,
) -> VoiceSets {
    let name = capture_dir_name(mode, kind);
    let manifest = captured_dir.join(&name).join("manifest.jsonl");
    let agg = aggregate_manifest(&manifest);
    #[allow(clippy::cast_precision_loss)]
    let hours_of = |frames: u64| frames as f64 * f64::from(mode.frame_ms()) / 1_000.0 / 3_600.0;
    let mut corpora: Vec<CorpusCount> = agg
        .iter()
        .map(|(c, u, f)| CorpusCount {
            corpus: c.clone(),
            utterances: *u,
            hours: hours_of(*f),
        })
        .collect();
    corpora.sort_by(|a, b| {
        b.utterances
            .cmp(&a.utterances)
            .then(a.corpus.cmp(&b.corpus))
    });
    let utterances = agg.iter().map(|(_, u, _)| *u).sum();
    let frames: u64 = agg.iter().map(|(_, _, f)| *f).sum();
    VoiceSets {
        name,
        mode,
        label: mode.label().to_owned(),
        kind: kind.map(|k| k.to_string()),
        readonly: kind.is_some(),
        utterances,
        hours: hours_of(frames),
        corpora,
    }
}

/// Every capture set's voice-set breakdown: the base modes, then any
/// decode-only siblings present on disk.
#[must_use]
pub fn voicesets(captured_dir: &Path) -> Vec<VoiceSets> {
    let mut out: Vec<VoiceSets> = VocoderMode::ALL
        .into_iter()
        .map(|m| voiceset_of(captured_dir, m, None))
        .collect();
    for name in capture_set_names(captured_dir) {
        if let Ok((mode, Some(kind))) = parse_capture_dir_name(&name) {
            out.push(voiceset_of(captured_dir, mode, Some(kind)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbs_parse() {
        assert_eq!(Control::from_verb("pause").unwrap(), Control::Pause);
        assert_eq!(Control::from_verb("resume").unwrap(), Control::Run);
        assert_eq!(Control::from_verb("stop").unwrap(), Control::Stop);
        assert!(Control::from_verb("start").is_err());
        assert_eq!(parse_mode("ysf-dmr").unwrap(), VocoderMode::YsfDmr);
        // The retired spelling still routes (a bookmarked URL).
        assert_eq!(parse_mode("ysf-dn").unwrap(), VocoderMode::YsfDmr);
        assert_eq!(parse_mode("codec2-1600").unwrap(), VocoderMode::Codec2_1600);
        assert!(parse_mode("fm").is_err());
    }

    #[test]
    fn every_mode_is_listed_with_its_family() {
        let tmp = tempfile::tempdir().unwrap();
        let sup = Supervisor::new(
            std::path::PathBuf::from("/bin/true"),
            std::time::Duration::from_secs(1),
            std::sync::Arc::new(crate::events::Hub::new(8)),
        );
        let views = view_all(tmp.path(), &sup, 5);
        assert_eq!(views.len(), VocoderMode::ALL.len());
        let c2 = views
            .iter()
            .find(|v| v.mode == VocoderMode::Codec2_1600)
            .unwrap();
        assert!(c2.software);
        assert_eq!(c2.family, "codec2");
        assert_eq!(c2.frame_ms, 40);
        let d = views.iter().find(|v| v.mode == VocoderMode::Dstar).unwrap();
        assert!(!d.software);
        assert_eq!(d.label, "D-STAR");
        assert_eq!(
            (views[1].name.as_str(), views[1].label.as_str()),
            ("ysf-dmr", "YSF/DMR")
        );
        assert!(
            views.iter().all(|v| v.name != "dmr"),
            "dmr is no capture mode"
        );
        assert_eq!((d.family.as_str(), d.frame_ms), ("ambe", 20));
        assert!(views.iter().all(|v| !v.readonly && v.kind.is_none()));
        // Siblings present on disk are listed after the modes, read-only.
        for d in ["dstar+drops", "codec2-3200+ber", "junk", "dstar+x"] {
            std::fs::create_dir_all(tmp.path().join(d)).unwrap();
        }
        assert_eq!(
            capture_set_names(tmp.path()),
            vec!["codec2-3200+ber", "dstar+drops"]
        );
        let views = view_all(tmp.path(), &sup, 5);
        assert_eq!(views.len(), VocoderMode::ALL.len() + 2);
        let sib = views.iter().find(|v| v.name == "dstar+drops").unwrap();
        assert!(sib.readonly);
        assert_eq!(sib.kind.as_deref(), Some("drops"));
        assert_eq!(sib.mode, VocoderMode::Dstar);
        assert_eq!(views[0].name, "dstar");
    }

    #[test]
    fn a_live_harness_is_stopping_once_stop_is_requested_or_its_status_is_final() {
        let tmp = tempfile::tempdir().unwrap();
        let sup = Supervisor::new(
            std::path::PathBuf::from("/bin/true"),
            std::time::Duration::from_secs(1),
            std::sync::Arc::new(crate::events::Hub::new(8)),
        );
        let dir = tmp.path().join("dstar");
        std::fs::create_dir_all(&dir).unwrap();
        // This process stands in for the harness: its pid is alive.
        let pid = std::process::id();
        let status = |state: &str| {
            std::fs::write(
                dir.join("status.json"),
                format!(r#"{{"state":"{state}","pid":{pid}}}"#),
            )
            .unwrap();
        };
        status("running");
        let v = view(tmp.path(), &sup, VocoderMode::Dstar, 0);
        assert!(v.alive && !v.stopping, "running, no control");

        write_control(tmp.path(), VocoderMode::Dstar, Control::Stop).unwrap();
        let v = view(tmp.path(), &sup, VocoderMode::Dstar, 0);
        assert!(v.alive && v.stopping, "stop requested, utterance in flight");

        // The final status is written before the process exits.
        write_control(tmp.path(), VocoderMode::Dstar, Control::Run).unwrap();
        status("stopped");
        let v = view(tmp.path(), &sup, VocoderMode::Dstar, 0);
        assert!(v.alive && v.stopping, "final status, process still exiting");

        // A dead pid is idle, whatever the files say.
        std::fs::write(
            dir.join("status.json"),
            r#"{"state":"running","pid":4000000000}"#,
        )
        .unwrap();
        write_control(tmp.path(), VocoderMode::Dstar, Control::Stop).unwrap();
        let v = view(tmp.path(), &sup, VocoderMode::Dstar, 0);
        assert!(!v.alive && !v.stopping, "dead harness");
    }

    #[test]
    fn control_file_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        write_control(tmp.path(), VocoderMode::YsfDmr, Control::Pause).unwrap();
        let v: Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("ysf-dmr/control.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(v["state"], "pause");
    }
}
