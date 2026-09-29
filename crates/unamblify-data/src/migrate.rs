// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `unamblify migrate-modes`: bring a data root written before a mode was
//! renamed up to the current spellings (`docs/design/data-pipeline.md`,
//! "Renamed modes"). Every retired name in [`VocoderMode::ALIASES`] still
//! *parses*, so nothing breaks before this runs; but the current code
//! looks for `captured/<mode.as_str()>/` and shard indexes name their
//! modes, so a set captured as `ysf-dn` is invisible until it is
//! `ysf-dmr` on disk.
//!
//! What it does, idempotently:
//!
//! 1. renames `captured/<old>/` and every sibling `captured/<old>+<kind>/`
//!    to the current name;
//! 2. rewrites the `mode` field of every row of `manifest.jsonl` and
//!    `failed.jsonl`, and of `status.json` and `canary.json`, in every
//!    capture set (a torn last line is left as it was);
//! 3. rewrites `shards/*/index.json`: the `mode` value, the `modes` list,
//!    and every map keyed by mode (`lags`, `counts_by_mode`, …).
//!
//! It refuses to touch a capture set whose `lock` another process holds
//! or whose `status.json` names a live harness, and refuses to rename
//! onto a directory that already exists (two captures of one mode have
//! to be merged by hand). Run configs (`runs/*/config.toml`) are not
//! rewritten: they load through the aliases and are a run's own record.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use unamblify::VocoderMode;

use crate::capture::LOCK_FILE;
use crate::util::FileLock;
use crate::{DataError, DataRoot, Result};

/// What a run did (or, with `dry_run`, would do).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MigrateReport {
    /// Nothing was written.
    pub dry_run: bool,
    /// Directories renamed, `(from, to)`.
    pub renamed: Vec<(PathBuf, PathBuf)>,
    /// Files rewritten, with how many `mode` occurrences changed in each.
    pub rewritten: Vec<(PathBuf, usize)>,
}

impl MigrateReport {
    /// Whether there was anything to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.renamed.is_empty() && self.rewritten.is_empty()
    }
}

/// A capture set directory name's mode segment and the rest (`+kind`).
fn split_set_name(name: &str) -> (&str, &str) {
    match name.split_once('+') {
        None => (name, ""),
        Some((m, _)) => (m, &name[m.len()..]),
    }
}

/// The current spelling of a mode name that parses (through an alias or
/// not); `None` when it is not a mode at all.
fn current(name: &str) -> Option<&'static str> {
    name.parse::<VocoderMode>().ok().map(VocoderMode::as_str)
}

/// The current spelling of a mode string, if it is a retired one.
fn retired(name: &str) -> Option<&'static str> {
    current(name).filter(|c| *c != name)
}

/// The current name of a capture set directory, if it is under a retired
/// one.
fn retired_set_name(name: &str) -> Option<String> {
    let (m, rest) = split_set_name(name);
    retired(m).map(|c| format!("{c}{rest}"))
}

/// Refuse when a harness is running on `dir`: the set's `lock` is held or
/// `status.json` names a live process.
fn ensure_idle(dir: &Path) -> Result<()> {
    let busy = |pid: Option<u32>| {
        DataError::Invalid(format!(
            "{} is in use by a running capture{}; stop it first (unamblify capture stop \
             --mode M) and run migrate-modes again",
            dir.display(),
            pid.map_or(String::new(), |p| format!(" (pid {p})"))
        ))
    };
    let lock = dir.join(LOCK_FILE);
    if lock.exists() {
        // Taken and dropped at once: a probe, not a hold.
        let held = FileLock::try_lock(&lock)?;
        if held.is_none() {
            return Err(busy(FileLock::holder(&lock)));
        }
    }
    if let Some(pid) = crate::control::live_capture_pid(&dir.join("status.json"))
        && pid != std::process::id()
    {
        return Err(busy(Some(pid)));
    }
    Ok(())
}

/// Rewrite retired mode spellings inside a JSON value: the `mode` field,
/// every string of a `modes` list, and object keys that are (or start
/// with, before a `+kind`) a retired mode. Returns how many changed.
fn rewrite_value(v: &mut Value) -> usize {
    let mut n = 0;
    match v {
        Value::Object(map) => {
            let renames: Vec<(String, String)> = map
                .keys()
                .filter_map(|k| retired_set_name(k).map(|c| (k.clone(), c)))
                .collect();
            for (old, new) in renames {
                if let Some(val) = map.remove(&old) {
                    map.insert(new, val);
                    n += 1;
                }
            }
            for (k, val) in map.iter_mut() {
                match (k.as_str(), &mut *val) {
                    ("mode", Value::String(s)) => {
                        if let Some(c) = retired(s) {
                            *s = c.to_owned();
                            n += 1;
                        }
                    }
                    ("modes", Value::Array(items)) => {
                        for it in items {
                            if let Value::String(s) = it
                                && let Some(c) = retired(s)
                            {
                                *s = c.to_owned();
                                n += 1;
                            }
                        }
                    }
                    _ => n += rewrite_value(val),
                }
            }
        }
        Value::Array(items) => {
            for it in items {
                n += rewrite_value(it);
            }
        }
        _ => {}
    }
    n
}

/// Rewrite a `.jsonl` file row by row; a line that does not parse (a torn
/// last line) is kept verbatim. Returns the rewritten text and how many
/// rows changed.
fn rewrite_jsonl(text: &str) -> Result<(String, usize)> {
    let mut out = String::with_capacity(text.len());
    let mut n = 0;
    for line in text.split_inclusive('\n') {
        let body = line.trim_end_matches('\n');
        match serde_json::from_str::<Value>(body) {
            Ok(mut v) if body.trim().starts_with('{') => {
                let changed = rewrite_value(&mut v);
                if changed > 0 {
                    n += changed;
                    out.push_str(&serde_json::to_string(&v)?);
                    if line.ends_with('\n') {
                        out.push('\n');
                    }
                } else {
                    out.push_str(line);
                }
            }
            _ => out.push_str(line),
        }
    }
    Ok((out, n))
}

/// Rewrite one file in place (atomically) if it holds retired spellings.
/// `.jsonl` files go row by row, anything else as one JSON document.
fn rewrite_file(path: &Path, dry_run: bool, report: &mut MigrateReport) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let text = fs::read_to_string(path).map_err(|e| DataError::io(path, e))?;
    let (new_text, n) = if path.extension().is_some_and(|e| e == "jsonl") {
        rewrite_jsonl(&text)?
    } else {
        let Ok(mut v) = serde_json::from_str::<Value>(&text) else {
            return Ok(());
        };
        let n = rewrite_value(&mut v);
        (serde_json::to_string_pretty(&v)?, n)
    };
    if n == 0 {
        return Ok(());
    }
    if !dry_run {
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        fs::write(&tmp, new_text).map_err(|e| DataError::io(&tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| DataError::io(path, e))?;
    }
    report.rewritten.push((path.to_path_buf(), n));
    Ok(())
}

/// The files of a capture set that carry a `mode` field.
const SET_FILES: [&str; 4] = [
    "manifest.jsonl",
    "failed.jsonl",
    "status.json",
    "canary.json",
];

/// Migrate `root`. With `dry_run` nothing is written and the report says
/// what would be.
pub fn run(root: &DataRoot, dry_run: bool) -> Result<MigrateReport> {
    let mut report = MigrateReport {
        dry_run,
        ..MigrateReport::default()
    };
    let captured = root.path().join("captured");
    let mut sets: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    if let Ok(rd) = fs::read_dir(&captured) {
        for e in rd.filter_map(std::result::Result::ok) {
            let path = e.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let (m, _) = split_set_name(name);
            if current(m).is_none() {
                continue;
            }
            let target = retired_set_name(name).map(|c| captured.join(c));
            sets.push((path, target));
        }
    }
    sets.sort();
    // Plan first, so a busy set stops the run before anything moved.
    for (dir, target) in &sets {
        ensure_idle(dir)?;
        if let Some(t) = target
            && t.exists()
        {
            return Err(DataError::Invalid(format!(
                "cannot rename {} to {}: it already exists; merge the two captures by hand, \
                 then run migrate-modes again",
                dir.display(),
                t.display()
            )));
        }
    }
    for (dir, target) in sets {
        // A dry run leaves the directory where it is and reports its
        // files under the old name.
        let dir = match target {
            Some(t) if !dry_run => {
                fs::rename(&dir, &t).map_err(|e| DataError::io(&dir, e))?;
                report.renamed.push((dir, t.clone()));
                t
            }
            Some(t) => {
                report.renamed.push((dir.clone(), t));
                dir
            }
            None => dir,
        };
        for f in SET_FILES {
            rewrite_file(&dir.join(f), dry_run, &mut report)?;
        }
    }
    if let Ok(rd) = fs::read_dir(root.path().join("shards")) {
        let mut indexes: Vec<PathBuf> = rd
            .filter_map(std::result::Result::ok)
            .map(|e| e.path().join("index.json"))
            .filter(|p| p.is_file())
            .collect();
        indexes.sort();
        for p in indexes {
            rewrite_file(&p, dry_run, &mut report)?;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(p: &Path, s: &str) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, s).unwrap();
    }

    fn old_root() -> (tempfile::TempDir, DataRoot) {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let cap = dir.path().join("captured");
        write(
            &cap.join("ysf-dn/manifest.jsonl"),
            "{\"key\":\"vctk/a\",\"mode\":\"ysf-dn\",\"frames\":3}\n\
             {\"key\":\"vctk/b\",\"mode\":\"ysf-dn\",\"frames\":4}\n\
             {\"key\":\"vctk/c\",\"mo",
        );
        write(
            &cap.join("ysf-dn/failed.jsonl"),
            "{\"key\":\"vctk/z\",\"mode\":\"ysf-dn\",\"attempts\":3}\n",
        );
        write(
            &cap.join("ysf-dn/status.json"),
            "{\"state\":\"done\",\"mode\":\"ysf-dn\",\"done\":2}",
        );
        write(
            &cap.join("ysf-dn/canary.json"),
            "{\"mode\":\"ysf-dn\",\"lag_samples\":5}",
        );
        write(&cap.join("ysf-dn/control.json"), "{\"state\":\"run\"}");
        write(&cap.join("ysf-dn/vctk/a.ambe"), "frames");
        write(
            &cap.join("ysf-dn+drops/manifest.jsonl"),
            "{\"key\":\"vctk/a\",\"mode\":\"ysf-dn\",\"aug\":{\"kind\":\"drops\"}}\n",
        );
        write(
            &cap.join("dstar/manifest.jsonl"),
            "{\"key\":\"vctk/a\",\"mode\":\"dstar\",\"frames\":3}\n",
        );
        write(&cap.join("notes"), "not a set");
        write(
            &dir.path().join("shards/mixed/index.json"),
            &json!({
                "name": "mixed",
                "mode": "ysf-dn",
                "modes": ["ysf-dn", "dstar"],
                "lags": {"ysf-dn": 5, "dstar": 326},
                "counts_by_mode": {"train": {"ysf-dn": 10, "dstar": 10}},
                "kinds": {"by_set": {"ysf-dn+drops": "x", "dstar": "y"}},
                "splits": {"train": "…"}
            })
            .to_string(),
        );
        write(
            &dir.path().join("shards/seed-dstar/index.json"),
            &json!({"name": "seed-dstar", "mode": "dstar", "modes": ["dstar"]}).to_string(),
        );
        (dir, root)
    }

    #[test]
    fn dry_run_reports_and_writes_nothing() {
        let (dir, root) = old_root();
        let r = run(&root, true).unwrap();
        assert!(r.dry_run);
        let cap = dir.path().join("captured");
        assert_eq!(
            r.renamed,
            vec![
                (cap.join("ysf-dn"), cap.join("ysf-dmr")),
                (cap.join("ysf-dn+drops"), cap.join("ysf-dmr+drops")),
            ]
        );
        let files: Vec<PathBuf> = r.rewritten.iter().map(|(p, _)| p.clone()).collect();
        assert_eq!(
            files,
            vec![
                cap.join("ysf-dn/manifest.jsonl"),
                cap.join("ysf-dn/failed.jsonl"),
                cap.join("ysf-dn/status.json"),
                cap.join("ysf-dn/canary.json"),
                cap.join("ysf-dn+drops/manifest.jsonl"),
                dir.path().join("shards/mixed/index.json"),
            ]
        );
        assert!(cap.join("ysf-dn").is_dir());
        assert!(!cap.join("ysf-dmr").exists());
        let st = fs::read_to_string(cap.join("ysf-dn/status.json")).unwrap();
        assert!(st.contains("ysf-dn"), "{st}");
    }

    #[test]
    fn migrates_a_root_once_and_is_then_a_no_op() {
        let (dir, root) = old_root();
        let r = run(&root, false).unwrap();
        assert!(!r.dry_run);
        assert_eq!(r.renamed.len(), 2);
        assert_eq!(r.rewritten.len(), 6, "{:?}", r.rewritten);
        let cap = dir.path().join("captured");
        assert!(!cap.join("ysf-dn").exists());
        assert!(cap.join("ysf-dmr/vctk/a.ambe").is_file());
        assert!(cap.join("ysf-dmr+drops").is_dir());
        // Rows rewritten one per line, the torn tail untouched.
        let m = fs::read_to_string(cap.join("ysf-dmr/manifest.jsonl")).unwrap();
        assert_eq!(
            m,
            "{\"frames\":3,\"key\":\"vctk/a\",\"mode\":\"ysf-dmr\"}\n\
             {\"frames\":4,\"key\":\"vctk/b\",\"mode\":\"ysf-dmr\"}\n\
             {\"key\":\"vctk/c\",\"mo"
        );
        assert_eq!(r.rewritten[0], (cap.join("ysf-dmr/manifest.jsonl"), 2));
        let st: Value =
            serde_json::from_str(&fs::read_to_string(cap.join("ysf-dmr/status.json")).unwrap())
                .unwrap();
        assert_eq!(st["mode"], "ysf-dmr");
        assert_eq!(st["done"], 2);
        let can: Value =
            serde_json::from_str(&fs::read_to_string(cap.join("ysf-dmr/canary.json")).unwrap())
                .unwrap();
        assert_eq!(can["mode"], "ysf-dmr");
        let f = fs::read_to_string(cap.join("ysf-dmr/failed.jsonl")).unwrap();
        assert!(f.contains("\"mode\":\"ysf-dmr\""), "{f}");
        // The sibling's rows too.
        let s = fs::read_to_string(cap.join("ysf-dmr+drops/manifest.jsonl")).unwrap();
        assert!(s.contains("\"mode\":\"ysf-dmr\""), "{s}");
        assert!(s.contains("\"kind\":\"drops\""), "{s}");
        // Untouched: dstar, the non-set file, control.json.
        assert_eq!(
            fs::read_to_string(cap.join("dstar/manifest.jsonl")).unwrap(),
            "{\"key\":\"vctk/a\",\"mode\":\"dstar\",\"frames\":3}\n"
        );
        assert!(cap.join("notes").is_file());
        assert_eq!(
            fs::read_to_string(cap.join("ysf-dmr/control.json")).unwrap(),
            "{\"state\":\"run\"}"
        );
        // The shard index: value, list, and every map keyed by mode.
        let idx: Value = serde_json::from_str(
            &fs::read_to_string(dir.path().join("shards/mixed/index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(idx["name"], "mixed");
        assert_eq!(idx["mode"], "ysf-dmr");
        assert_eq!(idx["modes"], json!(["ysf-dmr", "dstar"]));
        assert_eq!(idx["lags"], json!({"ysf-dmr": 5, "dstar": 326}));
        assert_eq!(idx["counts_by_mode"]["train"]["ysf-dmr"], 10);
        assert_eq!(idx["kinds"]["by_set"]["ysf-dmr+drops"], "x");
        assert_eq!(idx["splits"]["train"], "…");
        assert!(idx.get("ysf-dn").is_none());
        let seed = fs::read_to_string(dir.path().join("shards/seed-dstar/index.json")).unwrap();
        assert!(!seed.contains('\n'), "an untouched file keeps its bytes");
        // Every migrated set is a capture set the data root now sees.
        let names: Vec<String> = root
            .capture_dirs_present()
            .iter()
            .map(crate::CaptureDir::name)
            .collect();
        assert_eq!(names, vec!["dstar", "ysf-dmr", "ysf-dmr+drops"]);
        // Second run: nothing.
        let again = run(&root, false).unwrap();
        assert!(again.is_empty(), "{again:?}");
    }

    #[test]
    fn refuses_a_set_in_use_or_a_rename_onto_an_existing_dir() {
        let (dir, root) = old_root();
        let cap = dir.path().join("captured");
        // flock(2) is per open file description, so a lock held on another
        // descriptor in this process blocks exactly as another harness's.
        let held = FileLock::try_lock(&cap.join("ysf-dn").join(LOCK_FILE))
            .unwrap()
            .unwrap();
        let err = run(&root, false).unwrap_err().to_string();
        assert!(err.contains("in use by a running capture"), "{err}");
        assert!(cap.join("ysf-dn").is_dir(), "nothing moved");
        drop(held);

        fs::create_dir_all(cap.join("ysf-dmr")).unwrap();
        let err = run(&root, false).unwrap_err().to_string();
        assert!(err.contains("already exists"), "{err}");
        assert!(cap.join("ysf-dn").is_dir(), "nothing moved");
    }

    #[test]
    fn a_live_status_pid_counts_as_in_use() {
        let (dir, root) = old_root();
        let cap = dir.path().join("captured");
        // A status.json from before the lock file existed, naming a live
        // harness: a child that sleeps stands in for it.
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        write(
            &cap.join("ysf-dn/status.json"),
            &json!({
                "state": "running", "mode": "ysf-dn", "pid": child.id(), "ports": ["sim:0"],
                "done": 2, "failed": 0, "total": 9, "frames_s": 1.0, "utt_per_hour": 1.0,
                "started": "2026-09-11T00:00:00Z", "updated": "2026-09-11T00:00:00Z"
            })
            .to_string(),
        );
        let err = run(&root, false).unwrap_err().to_string();
        assert!(err.contains("in use"), "{err}");
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[test]
    fn value_rewrite_is_targeted() {
        let mut v = json!({
            "mode": "dmr", "modes": ["ysf", "codec2-3200"], "name": "ysf-dn",
            "nested": {"ysf-dn": {"mode": "ysf-dn"}, "dstar+drops": 1},
            "list": [{"mode": "ysf-dn"}, "ysf-dn"]
        });
        let n = rewrite_value(&mut v);
        assert_eq!(
            v,
            json!({
                "mode": "ysf-dmr", "modes": ["ysf-dmr", "codec2-3200"], "name": "ysf-dn",
                "nested": {"ysf-dmr": {"mode": "ysf-dmr"}, "dstar+drops": 1},
                "list": [{"mode": "ysf-dmr"}, "ysf-dn"]
            })
        );
        assert_eq!(n, 5);
        assert_eq!(rewrite_value(&mut v), 0);
        assert_eq!(split_set_name("ysf-dn+drops"), ("ysf-dn", "+drops"));
        assert_eq!(split_set_name("dstar"), ("dstar", ""));
        assert_eq!(
            retired_set_name("ysf-dn+drops").as_deref(),
            Some("ysf-dmr+drops")
        );
        assert_eq!(retired_set_name("ysf-dmr+drops"), None);
        assert_eq!(retired_set_name("junk"), None);
    }
}
