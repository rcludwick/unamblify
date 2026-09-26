// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The data stages of the unamblify harness (spec §2–§4): `prepare` walks
//! the raw corpora into normalised 16 kHz / 8 kHz pairs (and, with
//! `--noise-share` / `--ham-chain-share`, emits augmented twins through
//! [`twins`]), `capture` runs the
//! 8 kHz side through a vocoder's encode → decode — a ThumbDV (AMBE-3000)
//! for the AMBE modes, the `codec2` crate in software for the Codec 2
//! (M17) modes — `shard` packs fixed-length training examples, and
//! `verify` re-checks a capture; `migrate` renames what an older layout
//! left under a retired mode name.
//!
//! Every stage is a library function over a [`DataRoot`]; the thin
//! `unamblify-data` binary in this crate exists for local testing and the
//! real CLI lives in `unamblify-cli`. Nothing here opens a serial port in a
//! test: the chip side is generic over `ambe_thumbdv::Transport`, and
//! [`sim::SimTransport`] stands in for the dongle in `--dry-run` and tests.
//! Capture itself sees only the [`vocoder::Vocoder`] trait.

pub mod augment;
pub mod canary;
pub mod capture;
pub mod chip;
pub mod control;
pub mod migrate;
pub mod pipeline;
pub mod prepare;
pub mod shard;
pub mod sim;
#[cfg(test)]
pub(crate) mod testutil;
pub mod twins;
pub mod util;
pub mod verify;
pub mod vocoder;

use std::io;
use std::path::{Path, PathBuf};

use unamblify::VocoderMode;
use unamblify::aug::{AugKind, capture_dir_name, parse_capture_dir_name};

/// Default data root when `UNAMBLIFY_DATA` is unset.
pub const DEFAULT_DATA_ROOT: &str = "/Volumes/data/training_data/unamblify";

/// Environment variable naming the data root.
pub const DATA_ROOT_ENV: &str = "UNAMBLIFY_DATA";

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// File system, with the path that failed.
    #[error("{path}: {source}")]
    Io {
        /// The path being read or written.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// Audio decode / encode / resample.
    #[error("audio: {0}")]
    Audio(#[from] unamblify_audio::AudioError),
    /// JSON (de)serialisation.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// A manifest line that does not parse.
    #[error("{path}:{line}: {source}")]
    Manifest {
        /// The manifest file.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
    /// The chip's control protocol (init, transact).
    #[error("chip: {0}")]
    Chip(#[from] chip::ChipError),
    /// The pipelined encode / decode runner.
    #[error("pipeline: {0}")]
    Pipeline(#[from] pipeline::PipelineError),
    /// The canary re-encode did not match `canary.json`.
    #[error(
        "CANARY MISMATCH on {port} ({mode}): canary.json has frames sha256 {expected}, the codec \
         now produces {got}. The chip is in a different state or the link is dropping bytes (or, \
         for a software codec, the crate's encoder changed); the run is stopped and canary.json \
         is untouched."
    )]
    CanaryMismatch {
        /// The port whose chip disagreed.
        port: String,
        /// The mode.
        mode: VocoderMode,
        /// `canary.json`'s sha256.
        expected: String,
        /// What the chip produced now.
        got: String,
    },
    /// A `--port` that the FTDI VID/PID scan did not find.
    #[error("{0}")]
    PortRefused(String),
    /// A candidate port that something else holds open.
    #[error("ThumbDV at {port} is busy — another process has it open{holder}")]
    PortBusy {
        /// The port.
        port: String,
        /// `" (lsof: …)"` when `lsof` named the holder, else empty.
        holder: String,
    },
    /// No candidate port at all.
    #[error("no ThumbDV detected — plug in the dongle and try again")]
    PortAbsent,
    /// A capture worker hit an error that stops the run.
    #[error("worker on {port}: {source}")]
    Worker {
        /// The worker's port.
        port: String,
        /// What went wrong.
        #[source]
        source: Box<DataError>,
    },
    /// Anything else with a message.
    #[error("{0}")]
    Invalid(String),
}

impl DataError {
    /// Wrap an `io::Error` with the path it concerns.
    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// `Result` with this crate's error.
pub type Result<T> = std::result::Result<T, DataError>;

/// `$UNAMBLIFY_DATA` and the paths under it (spec §2). Every stage takes
/// one of these; nothing else in the crate spells a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataRoot(PathBuf);

impl DataRoot {
    /// A root at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// `$UNAMBLIFY_DATA`, else [`DEFAULT_DATA_ROOT`].
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(
            std::env::var_os(DATA_ROOT_ENV)
                .filter(|v| !v.is_empty())
                .map_or_else(|| PathBuf::from(DEFAULT_DATA_ROOT), PathBuf::from),
        )
    }

    /// The root directory.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// `raw/`.
    #[must_use]
    pub fn raw(&self) -> PathBuf {
        self.0.join("raw")
    }

    /// `prepared/`.
    #[must_use]
    pub fn prepared(&self) -> PathBuf {
        self.0.join("prepared")
    }

    /// `prepared/manifest.jsonl`.
    #[must_use]
    pub fn prepared_manifest(&self) -> PathBuf {
        self.prepared().join("manifest.jsonl")
    }

    /// `prepared/rejected.jsonl` — utterances `prepare` refused, so a re-run
    /// does not decode them again.
    #[must_use]
    pub fn prepared_rejected(&self) -> PathBuf {
        self.prepared().join("rejected.jsonl")
    }

    /// `prepared/<key>.16k.wav`.
    #[must_use]
    pub fn prepared_16k(&self, key: &str) -> PathBuf {
        self.prepared().join(format!("{key}.16k.wav"))
    }

    /// `prepared/<key>.8k.wav`.
    #[must_use]
    pub fn prepared_8k(&self, key: &str) -> PathBuf {
        self.prepared().join(format!("{key}.8k.wav"))
    }

    /// `prepared/<key>.16k.flac` — the lossless FLAC prepare writes now.
    #[must_use]
    pub fn prepared_16k_flac(&self, key: &str) -> PathBuf {
        self.prepared().join(format!("{key}.16k.flac"))
    }

    /// `prepared/<key>.8k.flac`.
    #[must_use]
    pub fn prepared_8k_flac(&self, key: &str) -> PathBuf {
        self.prepared().join(format!("{key}.8k.flac"))
    }

    /// The prepared 16 kHz file that exists for `key`: FLAC if present,
    /// else WAV. Both may be present during the move to FLAC.
    #[must_use]
    pub fn prepared_16k_decoded(&self, key: &str) -> PathBuf {
        let flac = self.prepared_16k_flac(key);
        if flac.is_file() {
            flac
        } else {
            self.prepared_16k(key)
        }
    }

    /// The prepared 8 kHz file that exists for `key`: FLAC if present,
    /// else WAV.
    #[must_use]
    pub fn prepared_8k_decoded(&self, key: &str) -> PathBuf {
        let flac = self.prepared_8k_flac(key);
        if flac.is_file() {
            flac
        } else {
            self.prepared_8k(key)
        }
    }

    /// `captured/<mode>/`.
    #[must_use]
    pub fn captured(&self, mode: VocoderMode) -> PathBuf {
        self.0.join("captured").join(mode.as_str())
    }

    /// One capture set's directory: the base `captured/<mode>/` (`kind`
    /// `None`) or a decode-only sibling `captured/<mode>+<kind>/`.
    #[must_use]
    pub fn capture_dir(&self, mode: VocoderMode, kind: Option<AugKind>) -> CaptureDir {
        CaptureDir {
            dir: self.0.join("captured").join(capture_dir_name(mode, kind)),
            mode,
            kind,
        }
    }

    /// Every capture set directory present under `captured/`, base and
    /// siblings, in name order; directories whose name is not a capture
    /// set name are ignored.
    #[must_use]
    pub fn capture_dirs_present(&self) -> Vec<CaptureDir> {
        let Ok(rd) = std::fs::read_dir(self.0.join("captured")) else {
            return Vec::new();
        };
        let mut out: Vec<CaptureDir> = rd
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                let name = e.file_name();
                let (mode, kind) = parse_capture_dir_name(name.to_str()?).ok()?;
                Some(self.capture_dir(mode, kind))
            })
            .collect();
        out.sort_by(|a, b| a.dir.cmp(&b.dir));
        out
    }

    /// `captured/<mode>/manifest.jsonl`.
    #[must_use]
    pub fn captured_manifest(&self, mode: VocoderMode) -> PathBuf {
        self.captured(mode).join("manifest.jsonl")
    }

    /// `captured/<mode>/failed.jsonl`.
    #[must_use]
    pub fn captured_failed(&self, mode: VocoderMode) -> PathBuf {
        self.captured(mode).join("failed.jsonl")
    }

    /// `captured/<mode>/canary.json`.
    #[must_use]
    pub fn canary_json(&self, mode: VocoderMode) -> PathBuf {
        self.captured(mode).join("canary.json")
    }

    /// `captured/<mode>/control.json`.
    #[must_use]
    pub fn control_json(&self, mode: VocoderMode) -> PathBuf {
        self.captured(mode).join("control.json")
    }

    /// `captured/<mode>/status.json`.
    #[must_use]
    pub fn status_json(&self, mode: VocoderMode) -> PathBuf {
        self.captured(mode).join("status.json")
    }

    /// `captured/<mode>/<key>.ambe` — the raw channel frames of any mode.
    /// The extension means "channel frames"; the mode says which codec
    /// (Codec 2 frames live in `.ambe` files too).
    #[must_use]
    pub fn captured_ambe(&self, mode: VocoderMode, key: &str) -> PathBuf {
        self.captured(mode).join(format!("{key}.ambe"))
    }

    /// `captured/<mode>/<key>.wav`.
    #[must_use]
    pub fn captured_wav(&self, mode: VocoderMode, key: &str) -> PathBuf {
        self.captured(mode).join(format!("{key}.wav"))
    }

    /// The decoded-audio file for `key` in a base capture: the FLAC if
    /// present, else the WAV.
    #[must_use]
    pub fn captured_decoded(&self, mode: VocoderMode, key: &str) -> PathBuf {
        let flac = self.captured(mode).join(format!("{key}.flac"));
        if flac.is_file() {
            flac
        } else {
            self.captured_wav(mode, key)
        }
    }

    /// `canary/1khz-and-speech.8k.wav`, relative to the root.
    pub const CANARY_CLIP: &'static str = "canary/1khz-and-speech.8k.wav";

    /// The canary clip's path on disk.
    #[must_use]
    pub fn canary_clip(&self) -> PathBuf {
        self.0.join(Self::CANARY_CLIP)
    }

    /// `shards/<name>/`.
    #[must_use]
    pub fn shards(&self, name: &str) -> PathBuf {
        self.0.join("shards").join(name)
    }

    /// `logs/`.
    #[must_use]
    pub fn logs(&self) -> PathBuf {
        self.0.join("logs")
    }
}

/// One capture set under `captured/`: the base capture of a mode or a
/// decode-only sibling (`docs/design/data-pipeline.md`, stage 3). Every
/// per-set file is spelled here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureDir {
    dir: PathBuf,
    mode: VocoderMode,
    kind: Option<AugKind>,
}

impl CaptureDir {
    /// The directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The set's name (`dstar`, `dstar+drops`).
    #[must_use]
    pub fn name(&self) -> String {
        capture_dir_name(self.mode, self.kind)
    }

    /// The mode.
    #[must_use]
    pub const fn mode(&self) -> VocoderMode {
        self.mode
    }

    /// The sibling kind, `None` for the base capture.
    #[must_use]
    pub const fn kind(&self) -> Option<AugKind> {
        self.kind
    }

    /// `manifest.jsonl`.
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        self.dir.join("manifest.jsonl")
    }

    /// `failed.jsonl`.
    #[must_use]
    pub fn failed(&self) -> PathBuf {
        self.dir.join("failed.jsonl")
    }

    /// `canary.json`.
    #[must_use]
    pub fn canary_json(&self) -> PathBuf {
        self.dir.join("canary.json")
    }

    /// `control.json`.
    #[must_use]
    pub fn control_json(&self) -> PathBuf {
        self.dir.join("control.json")
    }

    /// `status.json`.
    #[must_use]
    pub fn status_json(&self) -> PathBuf {
        self.dir.join("status.json")
    }

    /// `<key>.ambe` — the channel frames.
    #[must_use]
    pub fn ambe(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.ambe"))
    }

    /// `<key>.wav` — the decoded 8 kHz audio, WAV.
    #[must_use]
    pub fn wav(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.wav"))
    }

    /// `<key>.flac` — the decoded 8 kHz audio, lossless FLAC (what a
    /// capture writes now; smaller on disk and to sync).
    #[must_use]
    pub fn flac(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.flac"))
    }

    /// The decoded-audio file that exists for `key`: the FLAC if present,
    /// else the WAV. Both may be present during the move to FLAC; the WAV
    /// path is returned when neither is, so a caller reports the miss.
    #[must_use]
    pub fn decoded(&self, key: &str) -> PathBuf {
        let flac = self.flac(key);
        if flac.is_file() { flac } else { self.wav(key) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_dirs_name_base_and_siblings() {
        let r = DataRoot::new("/d");
        let base = r.capture_dir(VocoderMode::Dstar, None);
        assert_eq!(base.dir(), Path::new("/d/captured/dstar"));
        assert_eq!(base.name(), "dstar");
        assert_eq!(base.manifest(), r.captured_manifest(VocoderMode::Dstar));
        assert_eq!(base.ambe("a/b"), r.captured_ambe(VocoderMode::Dstar, "a/b"));
        assert_eq!(base.wav("a/b"), r.captured_wav(VocoderMode::Dstar, "a/b"));
        assert_eq!(base.canary_json(), r.canary_json(VocoderMode::Dstar));
        let sib = r.capture_dir(VocoderMode::Codec2_3200, Some(AugKind::Drops));
        assert_eq!(sib.dir(), Path::new("/d/captured/codec2-3200+drops"));
        assert_eq!(sib.name(), "codec2-3200+drops");
        assert_eq!(sib.kind(), Some(AugKind::Drops));
        assert_eq!(sib.mode(), VocoderMode::Codec2_3200);
        assert_eq!(
            sib.status_json(),
            PathBuf::from("/d/captured/codec2-3200+drops/status.json")
        );
        let tmp = tempfile::tempdir().unwrap();
        let r = DataRoot::new(tmp.path());
        for d in ["dstar", "dstar+drops", "codec2-3200", "junk", "dstar+x"] {
            std::fs::create_dir_all(r.path().join("captured").join(d)).unwrap();
        }
        std::fs::write(r.path().join("captured/ysf-dmr"), b"a file").unwrap();
        let names: Vec<String> = r
            .capture_dirs_present()
            .iter()
            .map(CaptureDir::name)
            .collect();
        assert_eq!(names, vec!["codec2-3200", "dstar", "dstar+drops"]);
        assert!(
            DataRoot::new("/nonexistent")
                .capture_dirs_present()
                .is_empty()
        );
    }

    #[test]
    fn paths_follow_the_spec_layout() {
        let r = DataRoot::new("/d");
        assert_eq!(
            r.prepared_16k("vctk/p225_001_mic2"),
            PathBuf::from("/d/prepared/vctk/p225_001_mic2.16k.wav")
        );
        assert_eq!(
            r.captured_ambe(VocoderMode::YsfDmr, "ljspeech/LJ001-0001"),
            PathBuf::from("/d/captured/ysf-dmr/ljspeech/LJ001-0001.ambe")
        );
        assert_eq!(
            r.canary_json(VocoderMode::YsfDmr),
            PathBuf::from("/d/captured/ysf-dmr/canary.json")
        );
        assert_eq!(
            r.shards("seed-dstar"),
            PathBuf::from("/d/shards/seed-dstar")
        );
        assert_eq!(
            r.canary_clip(),
            PathBuf::from("/d/canary/1khz-and-speech.8k.wav")
        );
    }
}
