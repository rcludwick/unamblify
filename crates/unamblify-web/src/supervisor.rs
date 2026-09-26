// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The run supervisor (spec §8). It re-execs this binary as
//! `<exe> train --config … --run-dir …` (or `<exe> capture --mode M`) via
//! `tokio::process`, waits for exit and finalises `status.json`. On
//! server start it adopts jobs whose `status.json` names a live pid and
//! marks dead ones `stopped` (a `queued` run is left as it is). Stop is
//! SIGTERM, a grace period, then SIGKILL.
//!
//! **Children never get a pipe.** A supervised job is meant to outlive the
//! dashboard (restart, Ctrl-C) and be adopted later; a pipe whose only
//! reader is this server would turn its next write into `EPIPE`, and
//! Rust's `eprintln!` panics on that. So stdout and stderr go to
//! `<dir>/child.log` (append). A trainer writes `log.jsonl` itself and is
//! told not to echo (`unamblify::run::LOG_ECHO_ENV`); whatever else it
//! printed (a panic, a libtorch warning) is copied into `log.jsonl` when
//! it exits. A capture harness has no `log.jsonl` of its own, so its
//! `child.log` is tailed into one while it runs. Either way `log.jsonl`
//! has one writer at a time, and every `seq` this server allocates is
//! read from the file's tail, so it never collides with the trainer's.
//!
//! `infer` is the one short-lived child: `<exe> infer --run-dir … --step
//! … --key … --out …` for the Samples page, run one at a time under a
//! lock, awaited to exit with its output captured (it never outlives the
//! server, so a pipe is safe), never tracked as a job.
//!
//! The trainer is never linked: this crate only knows the run-directory
//! formats of the core crate and the command line of spec §7.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use serde_json::{Value, json};
use tokio::process::Command;
use unamblify::run::LOG_ECHO_ENV;
use unamblify::{LogLevel, LogRow, RunState, VocoderMode};

/// Serialises this server's appends to one job's `log.jsonl`: the `seq`
/// is read from the file's tail and the row written under the same lock,
/// so `seq` order equals file order and never collides with rows the
/// trainer wrote itself.
type LogLock = Arc<Mutex<()>>;

/// Where a supervised child's stdout and stderr go, under the job's dir.
pub const CHILD_LOG: &str = "child.log";

/// How often a capture harness's `child.log` is copied into `log.jsonl`.
const TAIL_EVERY: Duration = Duration::from_millis(250);

/// At most this many trailing `child.log` lines are copied into
/// `log.jsonl` when a trainer exits.
const EXIT_TAIL_LINES: usize = 200;

use crate::error::{Result, WebError};
use crate::events::{Event, Hub, Kind};
use crate::{clock, runs};

/// What a supervised process is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobKind {
    /// A training run; the key is the run id.
    Train,
    /// A capture harness; the key is `capture:<mode>`.
    Capture(VocoderMode),
}

/// One tracked process.
#[derive(Debug, Clone)]
pub struct Proc {
    /// OS pid.
    pub pid: u32,
    /// Train or capture.
    pub kind: JobKind,
    /// Directory whose `status.json` / `log.jsonl` the job owns.
    pub dir: PathBuf,
    /// `true` when this server did not spawn it (found on restart).
    pub adopted: bool,
    /// Set when a stop was requested, so a clean exit is `stopped`,
    /// not `finished`.
    pub stop_requested: bool,
}

/// Job key for a capture mode.
#[must_use]
pub fn capture_key(mode: VocoderMode) -> String {
    format!("capture:{mode}")
}

/// Whether `pid` is alive (signal 0; `EPERM` counts as alive).
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    let Ok(raw) = i32::try_from(pid) else {
        return false;
    };
    matches!(
        kill(Pid::from_raw(raw), None),
        Ok(()) | Err(nix::errno::Errno::EPERM)
    )
}

fn send(pid: u32, sig: Signal) {
    if let Ok(raw) = i32::try_from(pid) {
        let _ = kill(Pid::from_raw(raw), sig);
    }
}

/// What an `infer` child is doing, by `run|step|key`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum InferState {
    /// Queued behind the lock or running.
    Running,
    /// Exited without producing the file.
    Failed {
        /// Why.
        error: String,
    },
}

/// Key of an infer job.
#[must_use]
pub fn infer_key(run: &str, step: u64, key: &str) -> String {
    format!("{run}|{step}|{key}")
}

/// The supervisor. Cheap to clone via `Arc`.
#[derive(Debug)]
pub struct Supervisor {
    exe: PathBuf,
    grace: Duration,
    hub: Arc<Hub>,
    procs: Mutex<HashMap<String, Proc>>,
    /// At most one `infer` child at a time.
    infer_lock: tokio::sync::Mutex<()>,
    infers: Mutex<HashMap<String, InferState>>,
}

impl Supervisor {
    /// A supervisor that re-execs `exe` and gives stopped jobs `grace`
    /// before SIGKILL.
    #[must_use]
    pub fn new(exe: PathBuf, grace: Duration, hub: Arc<Hub>) -> Self {
        Self {
            exe,
            grace,
            hub,
            procs: Mutex::new(HashMap::new()),
            infer_lock: tokio::sync::Mutex::new(()),
            infers: Mutex::new(HashMap::new()),
        }
    }

    /// The state of an infer job, if one was started and has not
    /// produced its file.
    #[must_use]
    pub fn infer_state(&self, run: &str, step: u64, key: &str) -> Option<InferState> {
        self.infers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&infer_key(run, step, key))
            .cloned()
    }

    /// Start `infer --run-dir <dir> --step <step> --key <key> --out <out>
    /// --data-root <root>` in the background unless the same job is
    /// already running. Returns `true` when a job was started. Poll
    /// [`Self::infer_state`] (or the output file) for the result.
    pub fn start_infer(
        self: &Arc<Self>,
        run: &str,
        dir: &Path,
        step: u64,
        key: &str,
        out: &Path,
        data_root: &Path,
    ) -> bool {
        let job = infer_key(run, step, key);
        {
            let mut map = self
                .infers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if map.get(&job) == Some(&InferState::Running) {
                return false;
            }
            map.insert(job.clone(), InferState::Running);
        }
        let me = Arc::clone(self);
        let (dir, key, out, root) = (
            dir.to_path_buf(),
            key.to_owned(),
            out.to_path_buf(),
            data_root.to_path_buf(),
        );
        tokio::spawn(async move {
            let result = me.run_infer(&dir, step, &key, &out, &root).await;
            let mut map = me
                .infers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match result {
                Ok(()) => {
                    map.remove(&job);
                }
                Err(e) => {
                    map.insert(
                        job,
                        InferState::Failed {
                            error: e.to_string(),
                        },
                    );
                }
            }
        });
        true
    }

    async fn run_infer(
        &self,
        dir: &Path,
        step: u64,
        key: &str,
        out: &Path,
        data_root: &Path,
    ) -> Result<()> {
        let _one_at_a_time = self.infer_lock.lock().await;
        if out.is_file() {
            return Ok(());
        }
        let mut cmd = Command::new(&self.exe);
        cmd.arg("infer")
            .arg("--run-dir")
            .arg(dir)
            .arg("--step")
            .arg(step.to_string())
            .arg("--key")
            .arg(key)
            .arg("--out")
            .arg(out)
            .arg("--data-root")
            .arg(data_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let describe = describe(&cmd);
        let output = cmd
            .output()
            .await
            .map_err(|e| WebError::Supervisor(format!("spawn {}: {e}", self.exe.display())))?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        // A record beside the outputs, whatever happened.
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(parent.join("infer.log"))
            {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{} {describe}
  exit {:?}
{}{}",
                    clock::rfc3339_now(),
                    output.status.code(),
                    indent(&stdout),
                    indent(&stderr)
                );
            }
        }
        if output.status.success() && out.is_file() {
            return Ok(());
        }
        // clap's message for a verb this binary was built without.
        if stderr.contains("unrecognized subcommand") || stderr.contains("unknown verb") {
            return Err(WebError::Conflict(
                "infer unavailable: the server binary was built without the `train` feature"
                    .to_owned(),
            ));
        }
        let last = stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("no output")
            .trim()
            .to_owned();
        Err(WebError::Supervisor(format!(
            "infer exited with {:?}: {last}",
            output.status.code()
        )))
    }

    /// The executable that gets re-exec'd.
    #[must_use]
    pub fn exe(&self) -> &Path {
        &self.exe
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Proc>> {
        // A poisoned map only means a panicking task held it; the data
        // is plain values and still consistent.
        self.procs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The tracked process for `key`, if any.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Proc> {
        self.lock().get(key).cloned()
    }

    /// Whether `key` is tracked and alive.
    #[must_use]
    pub fn is_running(&self, key: &str) -> bool {
        self.get(key).is_some_and(|p| pid_alive(p.pid))
    }

    /// Every tracked key.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// Start `train --config <dir>/config.toml --run-dir <dir>
    /// [--resume <ckpt>]` for the run at `dir`. Must be called from within
    /// a tokio runtime (the exit watcher and log pipes are tasks).
    pub fn start_train(
        self: &Arc<Self>,
        id: &str,
        dir: &Path,
        resume: Option<&Path>,
    ) -> Result<u32> {
        if self.is_running(id) {
            return Err(WebError::Conflict(format!("run {id} is already running")));
        }
        let mut cmd = Command::new(&self.exe);
        cmd.arg("train")
            .arg("--config")
            .arg(dir.join("config.toml"))
            .arg("--run-dir")
            .arg(dir);
        if let Some(r) = resume {
            cmd.arg("--resume").arg(r);
        }
        let pid = self.spawn(id, JobKind::Train, dir.to_path_buf(), cmd)?;
        // Mark running now; the trainer rewrites status.json itself once up.
        let (_, cfg) = runs::read_config(dir)?;
        let mut st = runs::read_status(dir)
            .unwrap_or_else(|| runs::new_status(&cfg, RunState::Running, None));
        st.status = RunState::Running;
        st.pid = Some(pid);
        st.updated = clock::rfc3339_now();
        if resume.is_none() {
            st.started = st.updated.clone();
        }
        runs::write_status(dir, &st)?;
        self.hub
            .publish(Event::for_run(id, Kind::Status, serde_json::to_value(&st)?));
        self.hub
            .publish(Event::global(Kind::Runs, json!({ "changed": id })));
        Ok(pid)
    }

    /// Start `capture --mode <mode> [extra…]` with `UNAMBLIFY_DATA` set to
    /// `data_root`, logging into `<data_root>/captured/<mode>/log.jsonl`.
    /// Refused while any harness is alive on the mode — one this server
    /// holds, or one `status.json` names (started from the shell, or
    /// supervised before a restart) — and `control.json` is not touched
    /// until that check passes, so a start never un-pauses someone else's
    /// run.
    pub fn start_capture(
        self: &Arc<Self>,
        mode: VocoderMode,
        data_root: &Path,
        extra_args: &[String],
    ) -> Result<u32> {
        let key = capture_key(mode);
        let capture_dir = data_root.join("captured").join(mode.as_str());
        let capture_dir = capture_dir.as_path();
        if self.is_running(&key) {
            return Err(WebError::Conflict(format!(
                "capture {mode} is already running"
            )));
        }
        if let Some(pid) = live_capture_pid(&capture_dir.join("status.json")) {
            return Err(WebError::Conflict(format!(
                "capture {mode} is already running (pid {pid}, not started by this server); stop \
                 it first"
            )));
        }
        std::fs::create_dir_all(capture_dir)?;
        // A stale `stop` in control.json would end the new run at once.
        runs::write_atomic(
            &capture_dir.join("control.json"),
            &serde_json::to_vec(&json!({ "state": "run" }))?,
        )?;
        let mut cmd = Command::new(&self.exe);
        cmd.env("UNAMBLIFY_DATA", data_root)
            .arg("capture")
            .arg("--mode")
            .arg(mode.as_str());
        for a in extra_args {
            cmd.arg(a);
        }
        let pid = self.spawn(&key, JobKind::Capture(mode), capture_dir.to_path_buf(), cmd)?;
        self.hub.publish(Event::global(
            Kind::Capture,
            json!({ "mode": mode, "event": "started", "pid": pid }),
        ));
        Ok(pid)
    }

    fn spawn(
        self: &Arc<Self>,
        key: &str,
        kind: JobKind,
        dir: PathBuf,
        mut cmd: Command,
    ) -> Result<u32> {
        std::fs::create_dir_all(&dir)?;
        let child_log = dir.join(CHILD_LOG);
        let out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&child_log)?;
        let err = out.try_clone()?;
        let offset = out.metadata().map_or(0, |m| m.len());
        cmd.stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .kill_on_drop(false);
        if kind == JobKind::Train {
            cmd.env(LOG_ECHO_ENV, "0");
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| WebError::Supervisor(format!("spawn {}: {e}", self.exe.display())))?;
        let pid = child
            .id()
            .ok_or_else(|| WebError::Supervisor("child exited before it had a pid".to_owned()))?;
        let lock: LogLock = Arc::new(Mutex::new(()));
        append(
            &dir,
            &lock,
            LogLevel::Info,
            format!("supervisor: spawned pid {pid}: {}", describe(&cmd)),
        );
        let offset = Arc::new(Mutex::new(offset));
        let tailer = matches!(kind, JobKind::Capture(_)).then(|| {
            tokio::spawn(tail_child_log(
                dir.clone(),
                Arc::clone(&lock),
                Arc::clone(&offset),
            ))
        });
        self.lock().insert(
            key.to_owned(),
            Proc {
                pid,
                kind: kind.clone(),
                dir: dir.clone(),
                adopted: false,
                stop_requested: false,
            },
        );
        let me = Arc::clone(self);
        let key = key.to_owned();
        tokio::spawn(async move {
            let code = match child.wait().await {
                Ok(s) => s.code(),
                Err(_) => None,
            };
            if let Some(t) = tailer {
                t.abort();
            }
            me.finalize(&key, pid, &dir, &kind, &lock, &offset, code)
                .await;
        });
        Ok(pid)
    }

    /// Adopt jobs found in `runs_dir` (and the capture dirs under
    /// `captured_dir`) whose status names a live pid; mark dead ones
    /// `stopped`.
    pub fn adopt(self: &Arc<Self>, runs_dir: &Path, captured_dir: &Path) {
        for id in runs::list_ids(runs_dir).unwrap_or_default() {
            let dir = runs_dir.join(&id);
            let Some(mut st) = runs::read_status(&dir) else {
                continue;
            };
            // A queued run has never had a process; it stays queued until
            // someone starts it.
            if !st.status.is_live() || st.status == RunState::Queued {
                continue;
            }
            if let Some(pid) = st.pid.filter(|&p| pid_alive(p)) {
                self.lock().insert(
                    id.clone(),
                    Proc {
                        pid,
                        kind: JobKind::Train,
                        dir: dir.clone(),
                        adopted: true,
                        stop_requested: false,
                    },
                );
                self.watch_adopted(id, dir, JobKind::Train, pid);
            } else {
                st.status = RunState::Stopped;
                st.pid = None;
                st.updated = clock::rfc3339_now();
                let _ = runs::write_status(&dir, &st);
                let seq = runs::last_log_seq(&dir) + 1;
                let _ = runs::append_log(
                    &dir,
                    &runs::log_row(
                        seq,
                        LogLevel::Warn,
                        "supervisor: trainer not alive on server start; marked stopped",
                    ),
                );
            }
        }
        for mode in VocoderMode::ALL {
            let dir = captured_dir.join(mode.as_str());
            let Some(st) = read_json(&dir.join("status.json")) else {
                continue;
            };
            if let Some(pid) = live_capture_pid_in(&st) {
                self.lock().insert(
                    capture_key(mode),
                    Proc {
                        pid,
                        kind: JobKind::Capture(mode),
                        dir: dir.clone(),
                        adopted: true,
                        stop_requested: false,
                    },
                );
                self.watch_adopted(capture_key(mode), dir, JobKind::Capture(mode), pid);
            }
        }
    }

    fn watch_adopted(self: &Arc<Self>, key: String, dir: PathBuf, kind: JobKind, pid: u32) {
        let me = Arc::clone(self);
        let lock: LogLock = Arc::new(Mutex::new(()));
        // Only what the child prints from now on is ours to copy.
        let offset = Arc::new(Mutex::new(
            std::fs::metadata(dir.join(CHILD_LOG)).map_or(0, |m| m.len()),
        ));
        let tailer = matches!(kind, JobKind::Capture(_)).then(|| {
            tokio::spawn(tail_child_log(
                dir.clone(),
                Arc::clone(&lock),
                Arc::clone(&offset),
            ))
        });
        tokio::spawn(async move {
            while pid_alive(pid) {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            if let Some(t) = tailer {
                t.abort();
            }
            me.finalize(&key, pid, &dir, &kind, &lock, &offset, None)
                .await;
        });
    }

    /// Finalise a job after its process is gone: rewrite `status.json` if
    /// the process left it live, log the exit, publish, then forget the
    /// key (last, so `stop` returns only once everything is on disk).
    #[allow(clippy::too_many_arguments)]
    async fn finalize(
        &self,
        key: &str,
        pid: u32,
        dir: &Path,
        kind: &JobKind,
        lock: &LogLock,
        offset: &Mutex<u64>,
        code: Option<i32>,
    ) {
        let stop_requested = self.get(key).is_some_and(|p| p.stop_requested);
        // Give the trainer's own final status write a moment to land.
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Whatever the child printed and nobody copied yet: everything for
        // a capture harness (its tailer is stopped), the last lines for a
        // trainer (its own rows are already in log.jsonl; this is for a
        // panic or a library's stderr).
        let fallback = if matches!(code, Some(0) | None) {
            LogLevel::Info
        } else {
            LogLevel::Warn
        };
        let keep = match kind {
            JobKind::Train => Some(EXIT_TAIL_LINES),
            JobKind::Capture(_) => None,
        };
        drain_child_log(dir, lock, offset, fallback, keep);
        match kind {
            JobKind::Train => {
                if let Some(mut st) = runs::read_status(dir) {
                    if st.status.is_live() {
                        st.status = match code {
                            Some(0) if !stop_requested && st.step >= st.total_steps => {
                                RunState::Finished
                            }
                            Some(0) | None => RunState::Stopped,
                            Some(_) => RunState::Failed,
                        };
                        st.pid = None;
                        st.updated = clock::rfc3339_now();
                        let _ = runs::write_status(dir, &st);
                    }
                    self.hub.publish(Event::for_run(
                        key,
                        Kind::Status,
                        serde_json::to_value(&st).unwrap_or(Value::Null),
                    ));
                }
                self.hub
                    .publish(Event::global(Kind::Runs, json!({ "changed": key })));
            }
            JobKind::Capture(mode) => {
                let p = dir.join("status.json");
                if let Some(mut st) = read_json(&p) {
                    let live = matches!(
                        st.get("state").and_then(Value::as_str),
                        Some("running" | "paused")
                    );
                    if live {
                        if let Some(obj) = st.as_object_mut() {
                            obj.insert(
                                "state".to_owned(),
                                Value::String(
                                    if code == Some(0) || code.is_none() {
                                        "stopped"
                                    } else {
                                        "failed"
                                    }
                                    .to_owned(),
                                ),
                            );
                            obj.insert("updated".to_owned(), Value::String(clock::rfc3339_now()));
                            obj.remove("pid");
                        }
                        if let Ok(bytes) = serde_json::to_vec_pretty(&st) {
                            let _ = runs::write_atomic(&p, &bytes);
                        }
                    }
                }
                self.hub.publish(Event::global(
                    Kind::Capture,
                    json!({ "mode": mode, "event": "exited", "code": code }),
                ));
            }
        }
        let level = if matches!(code, Some(0) | None) {
            LogLevel::Info
        } else {
            LogLevel::Error
        };
        let how = match code {
            Some(c) => format!("exit code {c}"),
            None => "exited (signal or adopted)".to_owned(),
        };
        append(
            dir,
            lock,
            level,
            format!("supervisor: process exited: {how}"),
        );
        let mut map = self.lock();
        if map.get(key).is_some_and(|p| p.pid == pid) {
            map.remove(key);
        }
    }

    /// Stop `key`: SIGTERM, wait up to the grace period, then SIGKILL.
    /// Returns when the process is gone.
    pub async fn stop(&self, key: &str) -> Result<()> {
        let pid = {
            let mut map = self.lock();
            let Some(p) = map.get_mut(key) else {
                return Err(WebError::Conflict(format!("{key} is not running")));
            };
            p.stop_requested = true;
            p.pid
        };
        if !pid_alive(pid) {
            return Ok(());
        }
        send(pid, Signal::SIGTERM);
        let deadline = tokio::time::Instant::now() + self.grace;
        while pid_alive(pid) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if pid_alive(pid) {
            send(pid, Signal::SIGKILL);
        }
        // Wait for the exit watcher (or, for adopted jobs, the liveness
        // poll) to reap and finalise; bounded so a wedged child never
        // hangs the request.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while self.get(key).is_some() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        Ok(())
    }

    /// Stop every tracked job (server shutdown does not do this: jobs are
    /// meant to outlive the dashboard and be adopted). Provided for tests
    /// and an explicit "stop all".
    pub async fn stop_all(&self) {
        for k in self.keys() {
            let _ = self.stop(&k).await;
        }
    }
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The pid a capture `status.json` names, when its state is `running` or
/// `paused` and that process is alive.
#[must_use]
pub fn live_capture_pid(status_path: &Path) -> Option<u32> {
    live_capture_pid_in(&read_json(status_path)?)
}

fn live_capture_pid_in(st: &Value) -> Option<u32> {
    let live = matches!(
        st.get("state").and_then(Value::as_str),
        Some("running" | "paused")
    );
    if !live {
        return None;
    }
    st.get("pid")
        .and_then(Value::as_u64)
        .and_then(|p| u32::try_from(p).ok())
        .filter(|&p| pid_alive(p))
}

fn indent(text: &str) -> String {
    let mut out = String::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        out.push_str("  ");
        out.push_str(l);
        out.push('\n');
    }
    out
}

fn describe(cmd: &Command) -> String {
    let c = cmd.as_std();
    let mut s = c.get_program().to_string_lossy().into_owned();
    for a in c.get_args() {
        s.push(' ');
        s.push_str(&a.to_string_lossy());
    }
    s
}

fn append(dir: &Path, lock: &LogLock, level: LogLevel, msg: String) {
    append_rows(dir, lock, std::iter::once(msg), |n, m| {
        runs::log_row(n, level, m)
    });
}

/// Append rows under the lock, each with the next `seq` after the file's
/// last one.
fn append_rows(
    dir: &Path,
    lock: &LogLock,
    lines: impl IntoIterator<Item = String>,
    make: impl Fn(u64, String) -> LogRow,
) {
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut n = runs::last_log_seq(dir);
    for line in lines {
        n += 1;
        let _ = runs::append_log(dir, &make(n, line));
    }
}

/// A row for one line the child printed: a line that is already `LogRow`
/// JSON keeps its level and message; an `env_logger` / `[level]` line
/// keeps its severity; anything else lands at `fallback`.
fn row_for_line(n: u64, line: String, fallback: LogLevel) -> LogRow {
    if let Ok(mut r) = serde_json::from_str::<LogRow>(&line) {
        r.seq = n;
        return r;
    }
    let level = if line.contains(" ERROR ")
        || line.starts_with("[error]")
        || line.contains("panicked at")
    {
        LogLevel::Error
    } else if line.contains(" WARN ") || line.starts_with("[warn]") {
        LogLevel::Warn
    } else {
        fallback
    };
    runs::log_row(n, level, line)
}

/// Copy the lines appended to `child.log` since `offset` into
/// `log.jsonl` (the last `keep_last` of them when given) and advance the
/// offset.
fn drain_child_log(
    dir: &Path,
    lock: &LogLock,
    offset: &Mutex<u64>,
    fallback: LogLevel,
    keep_last: Option<usize>,
) {
    let mut off = offset
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Ok((lines, new_off)) = runs::read_new_lines(&dir.join(CHILD_LOG), *off) else {
        return;
    };
    *off = new_off;
    let skip = keep_last.map_or(0, |k| lines.len().saturating_sub(k));
    append_rows(dir, lock, lines.into_iter().skip(skip), |n, l| {
        row_for_line(n, l, fallback)
    });
}

/// Follow a capture harness's `child.log` into its `log.jsonl` while it
/// runs; aborted when the process exits, after which `finalize` drains
/// the rest.
async fn tail_child_log(dir: PathBuf, lock: LogLock, offset: Arc<Mutex<u64>>) {
    loop {
        tokio::time::sleep(TAIL_EVERY).await;
        let (dir2, lock2, offset2) = (dir.clone(), Arc::clone(&lock), Arc::clone(&offset));
        let _ = tokio::task::spawn_blocking(move || {
            drain_child_log(&dir2, &lock2, &offset2, LogLevel::Info, None);
        })
        .await;
    }
}
