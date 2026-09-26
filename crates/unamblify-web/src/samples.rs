// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The Samples navigator's index: `prepared/manifest.jsonl` inner-joined
//! with every capture set's `captured/<name>/manifest.jsonl` present —
//! the base modes and the decode-only siblings (`dstar+drops`) — held in memory
//! and rebuilt when a manifest's size or mtime changes (checked at most
//! once per [`RECHECK`]). The prepared manifest is a few hundred thousand
//! rows and the captured ones grow a row every few seconds while a
//! harness runs, so each file is parsed only when *it* changed and the
//! join is redone from the parsed rows.
//!
//! Only what the list needs is kept per row (key, corpus, speaker, split,
//! duration, per-mode frame counts) plus each row's byte offset in its
//! manifest, so `GET /api/samples/{key}` can hand back the full
//! [`UtteranceRow`] / [`CaptureRow`] by seeking rather than by holding
//! them all.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use unamblify::aug::parse_capture_dir_name;
use unamblify::{CaptureRow, Rng, Split, UtteranceRow, VocoderMode};
use unamblify_audio::Spec;

use crate::capture::capture_set_names;
use crate::error::{Result, WebError};

/// Manifests are stat'ed at most this often.
pub const RECHECK: Duration = Duration::from_secs(2);
/// Default and maximum page sizes.
pub const DEFAULT_PER_PAGE: usize = 50;
/// Largest page a client may ask for.
pub const MAX_PER_PAGE: usize = 500;

/// Frames captured in one capture set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeFrames {
    /// The capture set's name: a mode (`dstar`) or a decode-only sibling
    /// (`dstar+drops`).
    pub mode: String,
    /// The set's name for a person (`D-STAR`, `YSF/DMR +drops`); `mode`
    /// stays the identifier.
    #[serde(default)]
    pub label: String,
    /// Whole frames the codec encoded.
    pub frames: u32,
    /// Milliseconds per frame (20, or 40 for Codec 2 1600), so a client
    /// can turn `frames` into seconds.
    pub frame_ms: u32,
}

/// Where a row sits in its manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Loc {
    offset: u64,
    len: u32,
}

/// One joined row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SampleItem {
    /// Utterance key.
    pub key: String,
    /// Corpus id.
    pub corpus: String,
    /// Corpus speaker id.
    pub speaker: String,
    /// Split.
    pub split: Split,
    /// Length, seconds.
    pub duration_s: f64,
    /// The capture sets it is in, base modes in [`VocoderMode::ALL`]
    /// order, each followed by its siblings.
    pub modes: Vec<ModeFrames>,
    /// For an augmented twin: the utterance it was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip)]
    prepared_loc: Loc,
    #[serde(skip)]
    captured_loc: Vec<(String, Loc)>,
}

/// One facet value and how many rows carry it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Facet {
    /// The value.
    pub name: String,
    /// The value's name for a person; only the mode facets carry one
    /// (`VocoderMode::label`, plus ` +<kind>` for a sibling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Rows.
    pub count: usize,
}

/// A capture set's name for a person: the mode's `label()`, with
/// ` +<kind>` for a decode-only sibling; a name that is not a set is
/// returned as it is.
#[must_use]
pub fn set_label(name: &str) -> String {
    parse_capture_dir_name(name).map_or_else(
        |_| name.to_owned(),
        |(m, k)| match k {
            None => m.label().to_owned(),
            Some(k) => format!("{} +{k}", m.label()),
        },
    )
}

/// A speaker facet, with its corpus so a client can narrow the list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeakerFacet {
    /// Speaker id.
    pub name: String,
    /// Corpus.
    pub corpus: String,
    /// Rows.
    pub count: usize,
}

/// `GET /api/samples/facets`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Facets {
    /// Corpora.
    pub corpora: Vec<Facet>,
    /// Splits.
    pub splits: Vec<Facet>,
    /// Speakers.
    pub speakers: Vec<SpeakerFacet>,
    /// Capture sets with a manifest and at least one joined row.
    pub modes: Vec<Facet>,
    /// Joined rows in total.
    pub total: usize,
}

/// Size + mtime of a file, or `None` when absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    len: u64,
    mtime: Option<SystemTime>,
}

fn stamp(p: &Path) -> Option<Stamp> {
    let m = fs::metadata(p).ok()?;
    Some(Stamp {
        len: m.len(),
        mtime: m.modified().ok(),
    })
}

/// What the list keeps of a prepared row.
#[derive(Debug, Clone, Deserialize)]
struct PreparedLite {
    key: String,
    corpus: String,
    speaker: String,
    split: Split,
    duration_s: f64,
    #[serde(default)]
    parent: Option<String>,
}

/// What the list keeps of a captured row.
#[derive(Debug, Clone, Deserialize)]
struct CapturedLite {
    key: String,
    frames: u32,
}

/// Walk a JSONL file, yielding each line's byte offset and parsed row. A
/// torn final line (an interrupted append) is dropped; any other line
/// that does not parse is skipped and counted.
fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<(Vec<(Loc, T)>, usize)> {
    let bytes = fs::read(path)?;
    let mut out = Vec::new();
    let mut skipped = 0usize;
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = bytes[offset..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| offset + i);
        let line_end = end.unwrap_or(bytes.len());
        let line = &bytes[offset..line_end];
        if !line.iter().all(u8::is_ascii_whitespace) {
            match serde_json::from_slice::<T>(line) {
                Ok(row) => out.push((
                    Loc {
                        offset: offset as u64,
                        len: u32::try_from(line.len()).unwrap_or(u32::MAX),
                    },
                    row,
                )),
                // A torn final line is expected; anything else is counted.
                Err(_) if end.is_none() => {}
                Err(_) => skipped += 1,
            }
        }
        offset = line_end + 1;
    }
    Ok((out, skipped))
}

/// Read one row back from its manifest by location, checking the key.
fn read_row_at<T: serde::de::DeserializeOwned>(path: &Path, loc: Loc, key: &str) -> Result<T> {
    let mut f = fs::File::open(path)?;
    f.seek(SeekFrom::Start(loc.offset))?;
    let mut buf = vec![0u8; loc.len as usize];
    f.read_exact(&mut buf)?;
    let v: Value = serde_json::from_slice(&buf)?;
    if v.get("key").and_then(Value::as_str) != Some(key) {
        return Err(WebError::Conflict(format!(
            "{}: manifest changed under the index; retry",
            path.display()
        )));
    }
    Ok(serde_json::from_value(v)?)
}

/// Parsed manifests plus the join.
#[derive(Debug, Default)]
pub struct Index {
    /// Joined rows in key order.
    pub items: Vec<SampleItem>,
    /// Row index by key.
    pub by_key: HashMap<String, usize>,
    /// The facets of `items`.
    pub facets: Facets,
    /// Lines that did not parse, per manifest.
    pub skipped: BTreeMap<String, usize>,
    prepared: Vec<(Loc, PreparedLite)>,
    prepared_stamp: Option<Stamp>,
    captured: BTreeMap<String, Vec<(Loc, CapturedLite)>>,
    captured_stamps: BTreeMap<String, Option<Stamp>>,
}

/// `prepared/manifest.jsonl`.
#[must_use]
pub fn prepared_manifest(root: &Path) -> PathBuf {
    root.join("prepared").join("manifest.jsonl")
}

/// `captured/<name>/manifest.jsonl` of a capture set.
#[must_use]
pub fn captured_manifest(root: &Path, name: &str) -> PathBuf {
    root.join("captured").join(name).join("manifest.jsonl")
}

/// Sort key of a capture set name: base modes in [`VocoderMode::ALL`]
/// order, each followed by its siblings by kind.
fn set_order(name: &str) -> (usize, String) {
    parse_capture_dir_name(name).map_or((usize::MAX, name.to_owned()), |(m, k)| {
        (
            VocoderMode::ALL
                .iter()
                .position(|x| *x == m)
                .unwrap_or(usize::MAX),
            k.map_or(String::new(), |k| k.to_string()),
        )
    })
}

/// Per capture set, for the join: name, frame ms, key → (frames, loc).
type SetMap<'a> = (&'a String, u32, HashMap<&'a str, (u32, Loc)>);

impl Index {
    /// The capture set names to consider: every one present under
    /// `captured/` plus every one parsed before (so a removed set is
    /// dropped), in [`set_order`].
    fn set_names(&self, root: &Path) -> Vec<String> {
        let mut names: Vec<String> = capture_set_names(&root.join("captured"));
        names.extend(self.captured_stamps.keys().cloned());
        names.sort_by_key(|n| set_order(n));
        names.dedup();
        names
    }

    /// Whether any manifest's size or mtime differs from what was parsed.
    fn stale(&self, root: &Path) -> bool {
        if stamp(&prepared_manifest(root)) != self.prepared_stamp {
            return true;
        }
        self.set_names(root).into_iter().any(|n| {
            stamp(&captured_manifest(root, &n)) != self.captured_stamps.get(&n).copied().flatten()
        })
    }

    /// Re-parse what changed and redo the join.
    fn refresh(&mut self, root: &Path) -> Result<()> {
        let pp = prepared_manifest(root);
        let ps = stamp(&pp);
        if ps != self.prepared_stamp || self.prepared_stamp.is_none() {
            if ps.is_some() {
                let (rows, skipped) = read_jsonl::<PreparedLite>(&pp)?;
                self.prepared = rows;
                self.skipped.insert("prepared".to_owned(), skipped);
            } else {
                self.prepared.clear();
            }
            self.prepared_stamp = ps;
        }
        for name in self.set_names(root) {
            let cp = captured_manifest(root, &name);
            let cs = stamp(&cp);
            if cs != self.captured_stamps.get(&name).copied().flatten()
                || !self.captured_stamps.contains_key(&name)
            {
                if cs.is_some() {
                    let (rows, skipped) = read_jsonl::<CapturedLite>(&cp)?;
                    self.captured.insert(name.clone(), rows);
                    self.skipped.insert(name.clone(), skipped);
                } else {
                    self.captured.remove(&name);
                    self.captured_stamps.remove(&name);
                    continue;
                }
                self.captured_stamps.insert(name, cs);
            }
        }
        self.join();
        Ok(())
    }

    fn join(&mut self) {
        // Per mode: key -> (frames, loc); the last row for a key wins,
        // as the capture harness's own resume rule reads it.
        let mut names: Vec<&String> = self.captured.keys().collect();
        names.sort_by_key(|n| set_order(n));
        let sets: Vec<SetMap<'_>> = names
            .into_iter()
            .map(|name| {
                let rows = &self.captured[name];
                let mut m = HashMap::with_capacity(rows.len());
                for (loc, r) in rows {
                    m.insert(r.key.as_str(), (r.frames, *loc));
                }
                let frame_ms = parse_capture_dir_name(name).map_or(20, |(mode, _)| mode.frame_ms());
                (name, frame_ms, m)
            })
            .collect();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut items: Vec<SampleItem> = Vec::new();
        for (loc, p) in &self.prepared {
            if !seen.insert(p.key.as_str()) {
                continue;
            }
            let mut modes = Vec::new();
            let mut captured_loc = Vec::new();
            for (name, frame_ms, m) in &sets {
                if let Some((frames, cloc)) = m.get(p.key.as_str()) {
                    modes.push(ModeFrames {
                        mode: (*name).clone(),
                        label: set_label(name),
                        frames: *frames,
                        frame_ms: *frame_ms,
                    });
                    captured_loc.push(((*name).clone(), *cloc));
                }
            }
            if modes.is_empty() {
                continue;
            }
            items.push(SampleItem {
                key: p.key.clone(),
                corpus: p.corpus.clone(),
                speaker: p.speaker.clone(),
                split: p.split,
                duration_s: p.duration_s,
                modes,
                parent: p.parent.clone(),
                prepared_loc: *loc,
                captured_loc,
            });
        }
        items.sort_by(|a, b| a.key.cmp(&b.key));
        self.by_key = items
            .iter()
            .enumerate()
            .map(|(i, it)| (it.key.clone(), i))
            .collect();
        self.facets = facets_of(&items);
        self.items = items;
    }

    /// The item for `key`, if joined.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&SampleItem> {
        self.by_key.get(key).map(|&i| &self.items[i])
    }

    /// The full manifest rows behind an item.
    pub fn rows(
        &self,
        root: &Path,
        item: &SampleItem,
    ) -> Result<(UtteranceRow, BTreeMap<String, CaptureRow>)> {
        let prepared: UtteranceRow =
            read_row_at(&prepared_manifest(root), item.prepared_loc, &item.key)?;
        let mut captured = BTreeMap::new();
        for (name, loc) in &item.captured_loc {
            let row: CaptureRow = read_row_at(&captured_manifest(root, name), *loc, &item.key)?;
            captured.insert(name.clone(), row);
        }
        Ok((prepared, captured))
    }
}

fn facets_of(items: &[SampleItem]) -> Facets {
    let mut corpora: BTreeMap<&str, usize> = BTreeMap::new();
    let mut splits: BTreeMap<Split, usize> = BTreeMap::new();
    let mut speakers: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut modes: BTreeMap<&str, usize> = BTreeMap::new();
    for it in items {
        *corpora.entry(&it.corpus).or_default() += 1;
        *splits.entry(it.split).or_default() += 1;
        *speakers.entry((&it.corpus, &it.speaker)).or_default() += 1;
        for m in &it.modes {
            *modes.entry(m.mode.as_str()).or_default() += 1;
        }
    }
    let mut mode_names: Vec<&str> = modes.keys().copied().collect();
    mode_names.sort_by_key(|n| set_order(n));
    Facets {
        corpora: corpora
            .into_iter()
            .map(|(name, count)| Facet {
                name: name.to_owned(),
                label: None,
                count,
            })
            .collect(),
        splits: Split::ALL
            .into_iter()
            .filter_map(|s| {
                splits.get(&s).map(|&count| Facet {
                    name: s.to_string(),
                    label: None,
                    count,
                })
            })
            .collect(),
        speakers: speakers
            .into_iter()
            .map(|((corpus, name), count)| SpeakerFacet {
                name: name.to_owned(),
                corpus: corpus.to_owned(),
                count,
            })
            .collect(),
        modes: mode_names
            .into_iter()
            .map(|n| Facet {
                name: n.to_owned(),
                label: Some(set_label(n)),
                count: modes[n],
            })
            .collect(),
        total: items.len(),
    }
}

/// The filters of `GET /api/samples`, `…/random`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Filter {
    /// Corpus id.
    #[serde(default)]
    pub corpus: Option<String>,
    /// Split.
    #[serde(default)]
    pub split: Option<String>,
    /// Speaker id.
    #[serde(default)]
    pub speaker: Option<String>,
    /// Keep keys captured in this capture set (`dstar`, `dstar+drops`), or
    /// in every one of a comma-separated list (`dstar,dstar+perens`): the
    /// utterances two sets share, which is what a side-by-side needs.
    #[serde(default)]
    pub mode: Option<String>,
    /// Substring of the key.
    #[serde(default)]
    pub q: Option<String>,
    /// 1-based page.
    #[serde(default)]
    pub page: Option<usize>,
    /// Rows per page.
    #[serde(default)]
    pub per_page: Option<usize>,
}

impl Filter {
    fn parsed(&self) -> Result<(Option<Split>, Vec<String>)> {
        let split = match self.split.as_deref().filter(|s| !s.is_empty()) {
            Some(s) => Some(
                s.parse::<Split>()
                    .map_err(|e| WebError::BadRequest(e.to_string()))?,
            ),
            None => None,
        };
        let mut sets = Vec::new();
        for m in self.mode.as_deref().unwrap_or_default().split(',') {
            let m = m.trim();
            if m.is_empty() || sets.iter().any(|s| s == m) {
                continue;
            }
            parse_capture_dir_name(m).map_err(|e| WebError::BadRequest(e.to_string()))?;
            sets.push(m.to_owned());
        }
        Ok((split, sets))
    }

    /// Indices into `index.items` that pass the filters, in key order.
    pub fn apply(&self, index: &Index) -> Result<Vec<usize>> {
        let (split, sets) = self.parsed()?;
        let corpus = self.corpus.as_deref().filter(|s| !s.is_empty());
        let speaker = self.speaker.as_deref().filter(|s| !s.is_empty());
        let q = self.q.as_deref().map(str::trim).filter(|s| !s.is_empty());
        Ok(index
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| {
                corpus.is_none_or(|c| it.corpus == c)
                    && speaker.is_none_or(|s| it.speaker == s)
                    && split.is_none_or(|s| it.split == s)
                    && sets.iter().all(|m| it.modes.iter().any(|x| &x.mode == m))
                    && q.is_none_or(|q| it.key.contains(q))
            })
            .map(|(i, _)| i)
            .collect())
    }

    /// `(page, per_page)` after defaults and clamps.
    #[must_use]
    pub fn paging(&self) -> (usize, usize) {
        (
            self.page.unwrap_or(1).max(1),
            self.per_page
                .unwrap_or(DEFAULT_PER_PAGE)
                .clamp(1, MAX_PER_PAGE),
        )
    }
}

/// One page of the list.
#[derive(Debug, Clone, Serialize)]
pub struct Page {
    /// Rows passing the filters.
    pub total: usize,
    /// 1-based page.
    pub page: usize,
    /// Rows per page.
    pub per_page: usize,
    /// This page's rows.
    pub items: Vec<SampleItem>,
}

/// `GET /api/samples`.
pub fn list(index: &Index, f: &Filter) -> Result<Page> {
    let hits = f.apply(index)?;
    let (page, per_page) = f.paging();
    let items = hits
        .iter()
        .skip((page - 1) * per_page)
        .take(per_page)
        .map(|&i| index.items[i].clone())
        .collect();
    Ok(Page {
        total: hits.len(),
        page,
        per_page,
        items,
    })
}

/// `GET /api/samples/random`: one row passing the filters.
pub fn random(index: &Index, f: &Filter, seed: u64) -> Result<Option<SampleItem>> {
    let hits = f.apply(index)?;
    if hits.is_empty() {
        return Ok(None);
    }
    let i = Rng::new(seed).below(hits.len());
    Ok(Some(index.items[hits[i]].clone()))
}

/// The in-memory index with its recheck clock.
#[derive(Debug, Default)]
pub struct Cache {
    index: Option<Arc<Index>>,
    checked: Option<Instant>,
}

impl Cache {
    /// The current index, rebuilt when a manifest changed (stat'ed at
    /// most once per [`RECHECK`]). Blocking: call from `spawn_blocking`.
    pub fn index(&mut self, root: &Path) -> Result<Arc<Index>> {
        let fresh = self.checked.is_some_and(|t| t.elapsed() < RECHECK);
        if let Some(ix) = &self.index
            && (fresh || !ix.stale(root))
        {
            self.checked = Some(Instant::now());
            return Ok(Arc::clone(ix));
        }
        // Rebuild from the previous parse where the files did not change.
        let mut ix = match self.index.take() {
            Some(old) => Arc::try_unwrap(old).unwrap_or_else(|shared| Index {
                prepared: shared.prepared.clone(),
                prepared_stamp: shared.prepared_stamp,
                captured: shared.captured.clone(),
                captured_stamps: shared.captured_stamps.clone(),
                ..Index::default()
            }),
            None => Index::default(),
        };
        ix.refresh(root)?;
        let ix = Arc::new(ix);
        self.index = Some(Arc::clone(&ix));
        self.checked = Some(Instant::now());
        Ok(ix)
    }
}

/// Which signal of a sample is asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Which {
    /// `prepared/<key>.8k.wav`.
    Clean8,
    /// `prepared/<key>.16k.wav`.
    Clean16,
    /// `captured/<name>/<key>.wav` of a capture set (`dstar`,
    /// `dstar+drops`).
    Degraded(String),
}

impl Which {
    /// Parse `clean8 | clean16 | degraded-<set>`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "clean8" => Ok(Self::Clean8),
            "clean16" => Ok(Self::Clean16),
            _ => match s.strip_prefix("degraded-") {
                Some(m) => parse_capture_dir_name(m)
                    .map(|_| Self::Degraded(m.to_owned()))
                    .map_err(|e| WebError::BadRequest(e.to_string())),
                None => Err(WebError::BadRequest(format!(
                    "unknown signal {s:?} (clean8 | clean16 | degraded-<set>)"
                ))),
            },
        }
    }

    /// The URL / cache spelling.
    #[must_use]
    pub fn as_string(&self) -> String {
        match self {
            Self::Clean8 => "clean8".to_owned(),
            Self::Clean16 => "clean16".to_owned(),
            Self::Degraded(m) => format!("degraded-{m}"),
        }
    }

    /// The WAV on disk for `key`.
    #[must_use]
    pub fn path(&self, root: &Path, key: &str) -> PathBuf {
        match self {
            Self::Clean8 => root.join("prepared").join(format!("{key}.8k.wav")),
            Self::Clean16 => root.join("prepared").join(format!("{key}.16k.wav")),
            Self::Degraded(m) => root.join("captured").join(m).join(format!("{key}.wav")),
        }
    }

    /// Directory and dotted basename (no audio extension) of `key`.
    fn stem(&self, root: &Path, key: &str) -> (PathBuf, String) {
        match self {
            Self::Clean8 => (root.join("prepared"), format!("{key}.8k")),
            Self::Clean16 => (root.join("prepared"), format!("{key}.16k")),
            Self::Degraded(m) => (root.join("captured").join(m), key.to_owned()),
        }
    }

    /// The audio file that actually exists for `key`: the FLAC if it is
    /// present, else the WAV (whose path is returned whether or not it
    /// exists, so the caller reports the miss). The corpus is moving to
    /// FLAC; both may be present during the transition.
    #[must_use]
    pub fn resolve(&self, root: &Path, key: &str) -> PathBuf {
        let (dir, base) = self.stem(root, key);
        let flac = dir.join(format!("{base}.flac"));
        if flac.is_file() {
            flac
        } else {
            dir.join(format!("{base}.wav"))
        }
    }
}

/// The audio MIME for a resolved clip path (`audio/flac` or `audio/wav`).
#[must_use]
pub fn audio_mime(path: &Path) -> &'static str {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("flac"))
    {
        "audio/flac"
    } else {
        "audio/wav"
    }
}

/// `<root>/cache/spec/<sha1(key|which)>.json`.
#[must_use]
pub fn spec_cache_path(root: &Path, key: &str, which: &Which) -> PathBuf {
    use std::fmt::Write;

    use sha1::{Digest, Sha1};
    let mut h = Sha1::new();
    h.update(key.as_bytes());
    h.update(b"|");
    h.update(which.as_string().as_bytes());
    let hex = h.finalize().iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    });
    root.join("cache").join("spec").join(format!("{hex}.json"))
}

/// The `spec.json` bytes for one signal of a sample: from the cache when
/// it is newer than the WAV, else computed (8 kHz upsampled to 16 kHz so
/// every panel shares the axes) and cached. Returns `(bytes, cache_hit)`.
pub fn spec_json(root: &Path, key: &str, which: &Which) -> Result<(Vec<u8>, bool)> {
    let wav = which.resolve(root, key);
    let src = fs::metadata(&wav).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            WebError::NotFound(wav.display().to_string())
        } else {
            WebError::Io(e)
        }
    })?;
    let cache = spec_cache_path(root, key, which);
    if let Ok(cm) = fs::metadata(&cache)
        && cm.modified().ok() >= src.modified().ok()
        && let Ok(bytes) = fs::read(&cache)
    {
        return Ok((bytes, true));
    }
    let audio = unamblify_audio::read(&wav)
        .map_err(|e| WebError::Supervisor(format!("{}: {e}", wav.display())))?;
    let x16 = match audio.rate {
        16_000 => audio.samples,
        8_000 => unamblify_audio::resample(&audio.samples, 8_000, 16_000)
            .map_err(|e| WebError::Supervisor(format!("resample {}: {e}", wav.display())))?,
        r => {
            return Err(WebError::Supervisor(format!(
                "{}: {r} Hz, want 8000 or 16000",
                wav.display()
            )));
        }
    };
    let spec = Spec::new().with(&which.as_string(), &x16);
    let bytes = serde_json::to_vec(&spec)?;
    if let Some(dir) = cache.parent() {
        fs::create_dir_all(dir)?;
    }
    crate::runs::write_atomic(&cache, &bytes)?;
    Ok((bytes, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_flac_then_wav_and_keeps_the_dotted_stem() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let which = Which::Degraded("dstar".to_owned());
        let key = "vctk/p225_001";
        // Neither on disk: the WAV path (for the caller's not-found), WAV mime.
        let p = which.resolve(root, key);
        assert!(
            p.to_string_lossy()
                .ends_with("captured/dstar/vctk/p225_001.wav")
        );
        assert_eq!(audio_mime(&p), "audio/wav");
        // WAV present: WAV.
        std::fs::create_dir_all(root.join("captured/dstar/vctk")).unwrap();
        std::fs::write(root.join("captured/dstar/vctk/p225_001.wav"), b"x").unwrap();
        assert!(which.resolve(root, key).to_string_lossy().ends_with(".wav"));
        // FLAC present: FLAC wins, and the mime follows.
        std::fs::write(root.join("captured/dstar/vctk/p225_001.flac"), b"x").unwrap();
        let p = which.resolve(root, key);
        assert!(p.to_string_lossy().ends_with(".flac"));
        assert_eq!(audio_mime(&p), "audio/flac");
        // The clean stems keep their `.16k` / `.8k` segment.
        std::fs::create_dir_all(root.join("prepared/vctk")).unwrap();
        std::fs::write(root.join("prepared/vctk/p225_001.16k.flac"), b"x").unwrap();
        let c = Which::Clean16.resolve(root, "vctk/p225_001");
        assert!(
            c.to_string_lossy()
                .ends_with("prepared/vctk/p225_001.16k.flac"),
            "{c:?}"
        );
    }

    #[test]
    fn which_parses_and_names_paths() {
        assert_eq!(Which::parse("clean8").unwrap(), Which::Clean8);
        assert_eq!(
            Which::parse("degraded-ysf-dmr").unwrap(),
            Which::Degraded("ysf-dmr".to_owned())
        );
        assert_eq!(
            Which::parse("degraded-dstar+drops").unwrap(),
            Which::Degraded("dstar+drops".to_owned())
        );
        assert!(Which::parse("degraded-fm").is_err());
        assert!(
            Which::parse("degraded-ysf-dn").is_err(),
            "a retired name is not a capture set"
        );
        assert!(Which::parse("degraded-dstar+x").is_err());
        assert!(Which::parse("out").is_err());
        let root = Path::new("/d");
        assert_eq!(
            Which::Clean16.path(root, "vctk/p225_001_mic2"),
            Path::new("/d/prepared/vctk/p225_001_mic2.16k.wav")
        );
        assert_eq!(
            Which::Degraded("ysf-dmr".to_owned()).path(root, "vctk/p225_001_mic2"),
            Path::new("/d/captured/ysf-dmr/vctk/p225_001_mic2.wav")
        );
        assert_eq!(
            Which::Degraded("ysf-dmr+ber".to_owned()).path(root, "vctk/p225_001_mic2"),
            Path::new("/d/captured/ysf-dmr+ber/vctk/p225_001_mic2.wav")
        );
        assert_eq!(set_order("dstar"), (0, String::new()));
        assert_eq!(set_order("dstar+drops"), (0, "drops".to_owned()));
        assert!(set_order("dstar+drops") < set_order("ysf-dmr"));
        assert_eq!(set_label("dstar"), "D-STAR");
        assert_eq!(set_label("ysf-dmr+drops"), "YSF/DMR +drops");
        assert_eq!(set_label("codec2-1600"), "Codec 2 1600 (M17)");
        assert_eq!(set_label("junk"), "junk");
        let a = spec_cache_path(root, "vctk/p225_001_mic2", &Which::Clean16);
        let b = spec_cache_path(root, "vctk/p225_001_mic2", &Which::Clean8);
        assert_ne!(a, b);
        assert!(a.starts_with("/d/cache/spec"));
        assert_eq!(a.file_name().unwrap().len(), 40 + 5);
    }

    #[test]
    fn jsonl_reader_drops_a_torn_tail_and_counts_bad_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("m.jsonl");
        fs::write(
            &p,
            "{\"key\":\"a/1\",\"frames\":3}\n\nnot json\n{\"key\":\"a/2\",\"frames\":4}\n{\"key\":\"a/3\",\"fra",
        )
        .unwrap();
        let (rows, skipped) = read_jsonl::<CapturedLite>(&p).unwrap();
        assert_eq!(skipped, 1);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, Loc { offset: 0, len: 24 });
        assert_eq!(rows[1].1.key, "a/2");
        let back: CapturedLite = read_row_at(&p, rows[1].0, "a/2").unwrap();
        assert_eq!(back.frames, 4);
        assert!(read_row_at::<CapturedLite>(&p, rows[1].0, "a/9").is_err());
    }
}
