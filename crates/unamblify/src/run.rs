// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! What a training run writes into `runs/<run_id>/` (spec §7) and the
//! dashboard reads back: `status.json`, one [`MetricRow`] per line of
//! `metrics.jsonl`, one [`LogRow`] per line of `log.jsonl`.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::ParseEnumError;

/// Environment variable the dashboard's supervisor sets to `0` on a
/// trainer it spawns: the trainer then does not echo its `log.jsonl`
/// lines to stderr (they would otherwise be copied into `log.jsonl` a
/// second time from the child's captured output).
pub const LOG_ECHO_ENV: &str = "UNAMBLIFY_LOG_ECHO";

/// `checkpoints/step-NNNNNN` under a run directory.
#[must_use]
pub fn checkpoint_dir(run_dir: &Path, step: u64) -> PathBuf {
    run_dir.join("checkpoints").join(format!("step-{step:06}"))
}

/// `samples/step-NNNNNN` under a run directory: where `unamblify infer`
/// writes a whole utterance rendered through that checkpoint, and where
/// the dashboard's Samples page looks for it.
#[must_use]
pub fn samples_dir(run_dir: &Path, step: u64) -> PathBuf {
    run_dir.join("samples").join(format!("step-{step:06}"))
}

/// `<samples_dir>/<clip_name(key)>.out.wav`: the `infer` output for
/// `key` at `step`; its spectrogram sits beside it as `.out.spec.json`.
#[must_use]
pub fn sample_out_path(run_dir: &Path, step: u64, key: &str) -> PathBuf {
    samples_dir(run_dir, step).join(format!("{}.out.wav", crate::key::clip_name(key)))
}

/// The `spec.json` written beside an `infer` output WAV.
#[must_use]
pub fn spec_path_for(out_wav: &Path) -> PathBuf {
    out_wav.with_extension("spec.json")
}

/// One metric sample: `{"step":n,"t":unix_ms,"k":"loss/total","v":0.123}`.
///
/// Keys are `loss/total`, `loss/stft`, `loss/mel`, `loss/sisdr`,
/// `loss/onset`, `loss/tail`, `lr`, `sys/steps_per_s`, `sys/samples_per_s`,
/// `sys/cpu`, `sys/gpu_util`, `sys/gpu_mem_gb`, `eval/lsd`, `eval/mel`,
/// `eval/sisdr`, `eval/lsd_first1s`, `eval/lsd_last1s`, `eval/babble`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricRow {
    /// Optimiser step the value belongs to.
    pub step: u64,
    /// Wall-clock time, Unix milliseconds.
    pub t: u64,
    /// Metric key.
    pub k: String,
    /// Value.
    pub v: f64,
}

/// Severity of a [`LogRow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Chatter.
    Debug,
    /// Normal progress.
    Info,
    /// Something recoverable.
    Warn,
    /// The run is going to stop.
    Error,
}

impl LogLevel {
    /// Every level, least severe first.
    pub const ALL: [Self; 4] = [Self::Debug, Self::Info, Self::Warn, Self::Error];

    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LogLevel {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|l| l.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "log level",
                input: s.to_owned(),
            })
    }
}

/// One line of `log.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogRow {
    /// Monotonic sequence number within the run (SSE resume cursor).
    pub seq: u64,
    /// Wall-clock time, Unix milliseconds.
    pub t: u64,
    /// Severity.
    pub level: LogLevel,
    /// Message.
    pub msg: String,
}

/// Lifecycle state in `status.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    /// Created, trainer not started yet.
    Queued,
    /// Trainer process alive.
    Running,
    /// Stopped by request (checkpoint written) or found dead on adoption.
    Stopped,
    /// Reached `total_steps`.
    Finished,
    /// Trainer exited non-zero; see `log.jsonl`.
    Failed,
}

impl RunState {
    /// Every state.
    pub const ALL: [Self; 5] = [
        Self::Queued,
        Self::Running,
        Self::Stopped,
        Self::Finished,
        Self::Failed,
    ];

    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Stopped => "stopped",
            Self::Finished => "finished",
            Self::Failed => "failed",
        }
    }

    /// Whether the run may still change.
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

impl fmt::Display for RunState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RunState {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|v| v.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "run state",
                input: s.to_owned(),
            })
    }
}

/// The best eval metric seen so far.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Best {
    /// Metric key (an `eval/*` key).
    pub metric: String,
    /// Its value at `step`.
    pub value: f64,
    /// Step of the checkpoint that achieved it.
    pub step: u64,
}

/// `status.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunStatus {
    /// Lifecycle state.
    pub status: RunState,
    /// Last completed step.
    pub step: u64,
    /// Steps the run was asked for.
    pub total_steps: u64,
    /// Start time (RFC 3339 UTC).
    pub started: String,
    /// Last update time (RFC 3339 UTC).
    pub updated: String,
    /// Trainer pid while running; absent afterwards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Device string the trainer resolved to.
    pub device: String,
    /// Hostname the trainer ran on.
    pub host: String,
    /// Best eval metric so far; absent before the first eval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<Best>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_directory_paths_are_fixed() {
        let run = Path::new("/runs/20260910-050000-x");
        assert_eq!(
            checkpoint_dir(run, 500),
            Path::new("/runs/20260910-050000-x/checkpoints/step-000500")
        );
        assert_eq!(
            samples_dir(run, 20),
            Path::new("/runs/20260910-050000-x/samples/step-000020")
        );
        let out = sample_out_path(run, 20, "vctk/p225_001_mic2");
        assert_eq!(
            out,
            Path::new("/runs/20260910-050000-x/samples/step-000020/vctk_p225_001_mic2.out.wav")
        );
        assert_eq!(
            spec_path_for(&out),
            Path::new(
                "/runs/20260910-050000-x/samples/step-000020/vctk_p225_001_mic2.out.spec.json"
            )
        );
    }

    #[test]
    fn metric_row_matches_the_spec_line() {
        let line = r#"{"step":12,"t":1757480000123,"k":"loss/total","v":0.123}"#;
        let row: MetricRow = serde_json::from_str(line).unwrap();
        assert_eq!(row.k, "loss/total");
        assert_eq!(serde_json::to_string(&row).unwrap(), line);
    }

    #[test]
    fn log_row_round_trips() {
        let row = LogRow {
            seq: 3,
            t: 1,
            level: LogLevel::Warn,
            msg: "checkpoint skipped".to_owned(),
        };
        let text = serde_json::to_string(&row).unwrap();
        assert!(text.contains("\"level\":\"warn\""));
        assert_eq!(serde_json::from_str::<LogRow>(&text).unwrap(), row);
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
    }

    #[test]
    fn status_round_trips_with_and_without_optionals() {
        let running = RunStatus {
            status: RunState::Running,
            step: 500,
            total_steps: 20_000,
            started: "2026-09-10T05:00:00Z".to_owned(),
            updated: "2026-09-10T05:10:00Z".to_owned(),
            pid: Some(4242),
            device: "cpu".to_owned(),
            host: "mac".to_owned(),
            best: Some(Best {
                metric: "eval/lsd".to_owned(),
                value: 1.25,
                step: 500,
            }),
        };
        let text = serde_json::to_string(&running).unwrap();
        assert!(text.contains("\"status\":\"running\""));
        assert_eq!(serde_json::from_str::<RunStatus>(&text).unwrap(), running);

        let queued = RunStatus {
            status: RunState::Queued,
            pid: None,
            best: None,
            ..running
        };
        let text = serde_json::to_string(&queued).unwrap();
        assert!(!text.contains("pid") && !text.contains("best"));
        assert_eq!(serde_json::from_str::<RunStatus>(&text).unwrap(), queued);
        assert!(RunState::Queued.is_live() && !RunState::Failed.is_live());
    }
}
