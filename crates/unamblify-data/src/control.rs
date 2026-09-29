// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Pause / resume / stop control of a capture run (spec §3).
//!
//! `captured/<mode>/control.json` holds `{"state":"run"|"pause"|"stop"}`
//! and is polled between utterances — never mid-utterance. On `pause` the
//! worker finishes the utterance in flight, keeps the port open, and polls
//! every [`PAUSE_POLL`]; on `run` it continues; on `stop` it exits after
//! the utterance. SIGINT is `stop`. `captured/<mode>/status.json` is
//! rewritten at least every [`STATUS_INTERVAL`] while running.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use unamblify::VocoderMode;

use crate::Result;
use crate::util::{read_json, write_json_atomic};

/// Poll interval while paused.
pub const PAUSE_POLL: Duration = Duration::from_millis(500);

/// Maximum interval between `status.json` rewrites while running.
pub const STATUS_INTERVAL: Duration = Duration::from_secs(5);

/// What the operator asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ControlState {
    /// Keep going (also the state when the file is absent).
    #[default]
    Run,
    /// Finish the current utterance, then wait.
    Pause,
    /// Finish the current utterance, then exit 0.
    Stop,
}

/// `control.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ControlFile {
    /// The requested state.
    pub state: ControlState,
}

/// Read `control.json`; absent → `Run`. An unparsable file (for example
/// one being written) is reported as an error so the caller can keep its
/// last state.
pub fn read_control(path: &Path) -> Result<ControlState> {
    if !path.exists() {
        return Ok(ControlState::Run);
    }
    read_json::<ControlFile>(path).map(|c| c.state)
}

/// Write `control.json` atomically.
pub fn write_control(path: &Path, state: ControlState) -> Result<()> {
    write_json_atomic(path, &ControlFile { state })
}

/// Run state as reported in `status.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    /// Capturing.
    Running,
    /// Waiting on `control.json`, port open.
    Paused,
    /// Exited on `stop` / SIGINT with work remaining.
    Stopped,
    /// Every planned utterance is done or failed.
    Done,
    /// Exited on an error (canary mismatch, port loss, …).
    Error,
}

/// `status.json` (spec §3), rewritten at least every 5 s while running.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// Run state.
    pub state: RunState,
    /// Mode being captured.
    pub mode: VocoderMode,
    /// For a decode-only sibling (`unamblify augment`): its kind
    /// (`drops`, `ber`); absent for a base capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The harness process that wrote this file. The dashboard keys
    /// liveness, adoption and its double-start guard on it; absent only
    /// in files written before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Ports in use, one per worker.
    pub ports: Vec<String>,
    /// Utterances finished this run + already in the manifest.
    pub done: u64,
    /// Utterances in `failed.jsonl` from this run.
    pub failed: u64,
    /// Utterances planned (done + remaining).
    pub total: u64,
    /// Frames per second over this run.
    pub frames_s: f64,
    /// Utterances per hour over this run.
    pub utt_per_hour: f64,
    /// Estimated seconds to finish, when a rate is known. Rates and the ETA
    /// exclude time spent paused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_s: Option<f64>,
    /// Seconds this run has spent paused (not counted in the rates).
    #[serde(default)]
    pub paused_s: f64,
    /// The utterance in flight on the first worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_key: Option<String>,
    /// The utterance in flight on every worker, by port order.
    #[serde(default)]
    pub current_keys: Vec<Option<String>>,
    /// RFC 3339 start of this run.
    pub started: String,
    /// RFC 3339 time of this write.
    pub updated: String,
    /// `true` after every canary check passed; `false` after a mismatch;
    /// absent before the first check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canary_ok: Option<bool>,
    /// PRODID of the first worker's chip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prodid: Option<String>,
    /// VERSTRING of the first worker's chip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// The last error, when `state` is `error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Write `status.json` atomically.
pub fn write_status(path: &Path, status: &Status) -> Result<()> {
    write_json_atomic(path, status)
}

/// Read `status.json`.
pub fn read_status(path: &Path) -> Result<Status> {
    read_json(path)
}

/// The pid of a harness that `status.json` says is running or paused and
/// that is still alive, if any. What both the CLI and the dashboard check
/// before starting a second harness on the same mode.
#[must_use]
pub fn live_capture_pid(status_path: &Path) -> Option<u32> {
    let st = read_status(status_path).ok()?;
    if !matches!(st.state, RunState::Running | RunState::Paused) {
        return None;
    }
    st.pid.filter(|&p| crate::util::pid_alive(p))
}

/// The control-file poller plus the SIGINT flag, shared by every worker.
#[derive(Debug, Clone)]
pub struct Controller {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    last: Arc<std::sync::Mutex<ControlState>>,
}

/// What a worker does after checking control between utterances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Take the next utterance.
    Continue,
    /// Exit after the utterance that just finished.
    Stop,
}

impl Controller {
    /// A controller for `control.json` at `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            stop: Arc::new(AtomicBool::new(false)),
            last: Arc::new(std::sync::Mutex::new(ControlState::Run)),
        }
    }

    /// The flag SIGINT (or anything else) sets to request a stop.
    #[must_use]
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Request a stop programmatically.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Whether a stop has been requested (flag or file).
    #[must_use]
    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::SeqCst) || self.state() == ControlState::Stop
    }

    /// The current control state: the SIGINT flag wins, then the file,
    /// then the last readable state when the file is mid-write.
    #[must_use]
    pub fn state(&self) -> ControlState {
        if self.stop.load(Ordering::SeqCst) {
            return ControlState::Stop;
        }
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match read_control(&self.path) {
            Ok(s) => {
                *last = s;
                s
            }
            Err(e) => {
                log::warn!(
                    "{}: unreadable, keeping {:?}: {e}",
                    self.path.display(),
                    *last
                );
                *last
            }
        }
    }

    /// Between utterances: return at once on `run`, wait (calling
    /// `on_pause(true)` once when pausing and `on_pause(false)` when
    /// resuming, polling every [`PAUSE_POLL`]) on `pause`, and return
    /// [`Decision::Stop`] on `stop`.
    pub fn checkpoint(&self, mut on_pause: impl FnMut(bool)) -> Decision {
        let mut paused = false;
        loop {
            match self.state() {
                ControlState::Run => {
                    if paused {
                        on_pause(false);
                    }
                    return Decision::Continue;
                }
                ControlState::Stop => {
                    if paused {
                        on_pause(false);
                    }
                    return Decision::Stop;
                }
                ControlState::Pause => {
                    if !paused {
                        paused = true;
                        on_pause(true);
                    }
                    std::thread::sleep(PAUSE_POLL);
                }
            }
        }
    }

    /// Install a SIGINT handler that sets the stop flag. A second call in
    /// the same process is a no-op (the handler is process-global).
    pub fn install_sigint(&self) {
        let flag = self.stop_flag();
        if let Err(e) = ctrlc::try_set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        }) {
            log::debug!("SIGINT handler not installed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_file_round_trips_and_defaults_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("control.json");
        assert_eq!(read_control(&p).unwrap(), ControlState::Run);
        write_control(&p, ControlState::Pause).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p)
                .unwrap()
                .replace(char::is_whitespace, ""),
            r#"{"state":"pause"}"#
        );
        assert_eq!(read_control(&p).unwrap(), ControlState::Pause);
        write_control(&p, ControlState::Stop).unwrap();
        assert_eq!(read_control(&p).unwrap(), ControlState::Stop);
    }

    #[test]
    fn checkpoint_pauses_then_resumes_or_stops() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("control.json");
        let c = Controller::new(&p);
        assert_eq!(c.checkpoint(|_| {}), Decision::Continue);

        write_control(&p, ControlState::Pause).unwrap();
        let p2 = p.clone();
        let resumer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(700));
            write_control(&p2, ControlState::Run).unwrap();
        });
        let mut events = Vec::new();
        let t = std::time::Instant::now();
        assert_eq!(c.checkpoint(|p| events.push(p)), Decision::Continue);
        resumer.join().unwrap();
        assert_eq!(events, vec![true, false]);
        assert!(t.elapsed() >= Duration::from_millis(500));

        c.request_stop();
        assert_eq!(c.checkpoint(|_| {}), Decision::Stop);
        assert!(c.stop_requested());
        let c2 = Controller::new(&p);
        write_control(&p, ControlState::Stop).unwrap();
        assert_eq!(c2.checkpoint(|_| {}), Decision::Stop);
        // A half-written file keeps the last readable state.
        std::fs::write(&p, "{\"state\":\"pa").unwrap();
        assert_eq!(c2.state(), ControlState::Stop);
    }

    #[test]
    fn status_round_trips() {
        let s = Status {
            state: RunState::Paused,
            mode: VocoderMode::Dstar,
            kind: None,
            pid: Some(std::process::id()),
            ports: vec!["/dev/sim".to_owned()],
            done: 3,
            failed: 1,
            total: 10,
            frames_s: 41.5,
            utt_per_hour: 900.0,
            eta_s: Some(28.0),
            paused_s: 0.0,
            current_key: Some("vctk/p225_001_mic2".to_owned()),
            current_keys: vec![Some("vctk/p225_001_mic2".to_owned())],
            started: "2026-09-10T03:00:00Z".to_owned(),
            updated: "2026-09-10T03:00:05Z".to_owned(),
            canary_ok: Some(true),
            prodid: Some("AMBE3000F".to_owned()),
            version: Some("V121".to_owned()),
            error: None,
        };
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("status.json");
        write_status(&p, &s).unwrap();
        assert_eq!(read_status(&p).unwrap(), s);
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"state\": \"paused\""));
        assert!(text.contains(&format!("\"pid\": {}", std::process::id())));
        assert!(!text.contains("error"));
    }
}
