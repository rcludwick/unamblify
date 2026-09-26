// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Small shared helpers: SHA-256 hex, RFC 3339 timestamps, JSONL reading
//! and appending, atomic JSON writes.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use crate::{DataError, Result};

/// Lowercase hex SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Lowercase hex SHA-256 of a file's bytes.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).map_err(|e| DataError::io(path, e))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| DataError::io(path, e))?;
    Ok(hex(&h.finalize()))
}

/// Lowercase hex of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// The current time as an RFC 3339 UTC string with second precision
/// (`2026-09-10T03:00:00Z`).
#[must_use]
pub fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339_from_unix(secs)
}

/// Seconds since the epoch, for elapsed-time arithmetic in status files.
#[must_use]
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// RFC 3339 UTC from Unix seconds (Howard Hinnant's civil-from-days).
#[must_use]
pub fn rfc3339_from_unix(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        (rem % 3_600) / 60,
        rem % 60
    )
}

/// Read every line of a JSONL file as `T`. A missing file is an empty list.
///
/// A final line with no trailing newline that does not parse is the only
/// damage an interrupted append can leave (a kill or power loss
/// mid-write); it is dropped with a warning so every stage stays
/// resumable, and the next [`JsonlWriter::open`] truncates it away before
/// appending. A malformed line that *is* newline-terminated is still a
/// hard error: that is corruption, not a torn write.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(DataError::io(path, e)),
    };
    let torn = !text.is_empty() && !text.ends_with('\n');
    let mut out = Vec::new();
    let mut lines = text.split('\n').enumerate().peekable();
    while let Some((i, line)) = lines.next() {
        let last = lines.peek().is_none();
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(line) {
            Ok(row) => out.push(row),
            Err(source) if last && torn => {
                log::warn!(
                    "{}:{}: dropping torn final line ({} bytes, no newline): {source}",
                    path.display(),
                    i + 1,
                    line.len()
                );
            }
            Err(source) => {
                return Err(DataError::Manifest {
                    path: path.to_path_buf(),
                    line: i + 1,
                    source,
                });
            }
        }
    }
    Ok(out)
}

/// Write `rows` as a whole JSONL file atomically (temp file + rename).
/// Used to drop rows from a manifest (a redo, a row whose files are gone).
pub fn write_jsonl<'a, T: Serialize + 'a>(
    path: &Path,
    rows: impl IntoIterator<Item = &'a T>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    let mut text = String::new();
    for r in rows {
        text.push_str(&serde_json::to_string(r)?);
        text.push('\n');
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    fs::write(&tmp, text).map_err(|e| DataError::io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| DataError::io(path, e))
}

/// An append-only JSONL writer: one line per row, flushed on every row,
/// `fsync` on demand (and on drop).
pub struct JsonlWriter {
    file: File,
    path: std::path::PathBuf,
    rows_since_sync: usize,
}

impl JsonlWriter {
    /// Open (creating directories and the file as needed) for append. A
    /// torn final line (no trailing newline, see [`read_jsonl`]) is
    /// truncated first so the next row does not glue onto it.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
        }
        truncate_torn_tail(path)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| DataError::io(path, e))?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
            rows_since_sync: 0,
        })
    }

    /// Append one row and flush it to the OS.
    pub fn append<T: Serialize>(&mut self, row: &T) -> Result<()> {
        let mut line = serde_json::to_string(row)?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| DataError::io(&self.path, e))?;
        self.rows_since_sync += 1;
        Ok(())
    }

    /// Rows appended since the last [`sync`](Self::sync).
    #[must_use]
    pub fn rows_since_sync(&self) -> usize {
        self.rows_since_sync
    }

    /// `fsync` the file.
    pub fn sync(&mut self) -> Result<()> {
        self.file
            .sync_data()
            .map_err(|e| DataError::io(&self.path, e))?;
        self.rows_since_sync = 0;
        Ok(())
    }
}

/// Cut a JSONL file back to its last newline when its final line is
/// incomplete.
fn truncate_torn_tail(path: &Path) -> Result<()> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(DataError::io(path, e)),
    };
    if bytes.is_empty() || bytes.ends_with(b"\n") {
        return Ok(());
    }
    let keep = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    log::warn!(
        "{}: truncating {} bytes of a torn final line before appending",
        path.display(),
        bytes.len() - keep
    );
    let f = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| DataError::io(path, e))?;
    f.set_len(keep as u64).map_err(|e| DataError::io(path, e))
}

/// Whether `pid` names a live process (signal 0; `EPERM` counts as alive).
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    let Ok(raw) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: kill(2) with signal 0 only checks for the process; no memory
    // is touched.
    let r = unsafe { libc::kill(raw, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// An advisory exclusive lock on a file, held while the value lives. Two
/// harnesses on one `captured/<mode>/` must not both append to
/// `manifest.jsonl`; `flock(2)` is released by the kernel when the holder
/// dies, so a crash never leaves a stale lock behind.
#[derive(Debug)]
pub struct FileLock {
    _file: File,
    path: std::path::PathBuf,
}

impl FileLock {
    /// Take the lock at `path`, writing this process id into the file;
    /// `Ok(None)` when another process holds it.
    pub fn try_lock(path: &Path) -> Result<Option<Self>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|e| DataError::io(path, e))?;
        // SAFETY: flock(2) on an fd this File owns.
        let r = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if r != 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Ok(None);
            }
            return Err(DataError::io(path, e));
        }
        file.set_len(0).map_err(|e| DataError::io(path, e))?;
        file.write_all(std::process::id().to_string().as_bytes())
            .map_err(|e| DataError::io(path, e))?;
        let _ = file.sync_data();
        Ok(Some(Self {
            _file: file,
            path: path.to_path_buf(),
        }))
    }

    /// The pid written into a lock file, if it holds one.
    #[must_use]
    pub fn holder(path: &Path) -> Option<u32> {
        fs::read_to_string(path).ok()?.trim().parse().ok()
    }

    /// The lock file's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for JsonlWriter {
    fn drop(&mut self) {
        let _ = self.file.sync_data();
    }
}

/// Write `value` as pretty JSON to `path` atomically (temp file + rename).
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&tmp, text).map_err(|e| DataError::io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| DataError::io(path, e))?;
    Ok(())
}

/// Read a JSON file as `T`.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(|e| DataError::io(path, e))?;
    Ok(serde_json::from_str(&text)?)
}

/// Write `bytes` to `path`, creating parent directories.
pub fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    fs::write(path, bytes).map_err(|e| DataError::io(path, e))
}

/// Read a whole file.
pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| DataError::io(path, e))
}

/// The shared deterministic PRNG (`unamblify::rng::Rng`), so a shard set
/// is a pure function of manifests + seed and draws the same garbage tails
/// as the training loader.
pub use unamblify::Rng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_empty_is_the_known_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn rfc3339_matches_known_instants() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_789_009_200), "2026-09-10T03:00:00Z");
        assert!(now_rfc3339().ends_with('Z'));
    }

    #[test]
    fn jsonl_round_trips_and_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a/b.jsonl");
        {
            let mut w = JsonlWriter::open(&p).unwrap();
            w.append(&serde_json::json!({"k": 1})).unwrap();
            w.append(&serde_json::json!({"k": 2})).unwrap();
            assert_eq!(w.rows_since_sync(), 2);
            w.sync().unwrap();
            assert_eq!(w.rows_since_sync(), 0);
        }
        fs::write(&p, format!("{}\n\n", fs::read_to_string(&p).unwrap())).unwrap();
        let rows: Vec<serde_json::Value> = read_jsonl(&p).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["k"], 2);
        let none: Vec<serde_json::Value> = read_jsonl(&dir.path().join("missing")).unwrap();
        assert!(none.is_empty());
        fs::write(&p, "{not json\n").unwrap();
        assert!(matches!(
            read_jsonl::<serde_json::Value>(&p),
            Err(DataError::Manifest { line: 1, .. })
        ));
    }

    #[test]
    fn a_torn_final_line_is_dropped_and_truncated_but_corruption_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.jsonl");
        fs::write(&p, "{\"k\":1}\n{\"k\":2}\n{\"k\":3}").unwrap();
        let rows: Vec<serde_json::Value> = read_jsonl(&p).unwrap();
        assert_eq!(
            rows.len(),
            3,
            "a complete last line without newline still counts"
        );
        fs::write(&p, "{\"k\":1}\n{\"k\":2}\n{\"k\":").unwrap();
        let rows: Vec<serde_json::Value> = read_jsonl(&p).unwrap();
        assert_eq!(rows.len(), 2, "torn tail dropped");
        // Appending truncates the torn tail first so rows never glue.
        {
            let mut w = JsonlWriter::open(&p).unwrap();
            w.append(&serde_json::json!({"k": 9})).unwrap();
        }
        let rows: Vec<serde_json::Value> = read_jsonl(&p).unwrap();
        assert_eq!(
            rows.iter()
                .map(|r| r["k"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 9]
        );
        // Newline-terminated garbage in the middle is corruption.
        fs::write(&p, "{\"k\":1}\n{\"k\":\n{\"k\":3}\n").unwrap();
        assert!(matches!(
            read_jsonl::<serde_json::Value>(&p),
            Err(DataError::Manifest { line: 2, .. })
        ));
        write_jsonl(&p, &[serde_json::json!({"k": 5})]).unwrap();
        let rows: Vec<serde_json::Value> = read_jsonl(&p).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn file_lock_is_exclusive_and_records_the_holder() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a/lock");
        let held = FileLock::try_lock(&p).unwrap().expect("first lock");
        assert_eq!(held.path(), p);
        assert_eq!(FileLock::holder(&p), Some(std::process::id()));
        // flock locks belong to the open file description, so a second
        // open — in this process or another — contends.
        assert!(
            FileLock::try_lock(&p).unwrap().is_none(),
            "second holder refused"
        );
        drop(held);
        assert!(
            FileLock::try_lock(&p).unwrap().is_some(),
            "released on drop"
        );
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(u32::MAX - 1));
    }
}
