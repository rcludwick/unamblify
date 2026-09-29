// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `prepare` (spec §2): walk the raw corpora, and for every utterance
//! decode → resample to 16 kHz → trim silence → normalise → write
//! `prepared/<key>.16k.wav` and the 8 kHz decimation, appending one
//! [`UtteranceRow`] to `prepared/manifest.jsonl`. Parallel over
//! utterances (rayon), one writer thread with an `fsync` every 1000 rows,
//! idempotent (keys already in the manifest with consistent files are
//! skipped; `force` re-does them), and rejections (`< 1 s`, silence,
//! clipping after gain, unreadable) go to `prepared/rejected.jsonl` so a
//! re-run does not decode them again. A *processing* failure (a full or
//! unplugged drive, an unwritable directory, a resampler error) is not a
//! rejection: it is logged and counted, nothing is written, and the next
//! run retries the utterance.
//!
//! Corpus layouts under `raw/` (from the corpus survey):
//!
//! | corpus | files |
//! |---|---|
//! | LibriTTS-R | `libritts_r/LibriTTS_R/<subset>/<reader>/<chapter>/*.wav` (only stems of the form `<reader>_<chapter>_<para>_<sent>`; the doc files in `train-clean-100` do not parse and are skipped) |
//! | VoiceBank-DEMAND | `voicebank_demand/{clean_trainset_28spk_wav,clean_testset_wav}/pNNN_NNN.wav` (`noisy_*` ignored) |
//! | VCTK | `vctk/wav48_silence_trimmed/<spk>/<spk>_<utt>_mic{1,2}.flac`, `mic2` preferred, one row per utterance |
//! | LJSpeech | `ljspeech/LJSpeech-1.1/wavs/LJ*.wav`, speaker `LJ` |
//! | Common Voice (English) | `common_voice/<variant>/…/{clips,audio_files}/*.mp3` with a `client_id`/`path` table (`validated.tsv`, else `train`/`dev`/`test.tsv`, or the `en-AU` CSV export); opt-in via `--corpus common_voice`, speaker is the opaque `client_id` |

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use unamblify::split::{
    CORPUS_HIFI_TTS, CORPUS_LIBRITTS_R, CORPUS_LJSPEECH, CORPUS_VCTK, CORPUS_VOICEBANK_DEMAND,
    assign,
};
use unamblify::{UtteranceRow, VOCODER_SAMPLE_RATE, WIDEBAND_SAMPLE_RATE, key};
use unamblify_audio::level::{MAX_GAIN_DB, TARGET_DBFS, TRIM_MARGIN_S, TRIM_THRESHOLD_DBFS};
use unamblify_audio::{
    active_rms_dbfs, normalize_rms, read, resample, trim_silence, write_flac_s16,
};

use crate::twins::{self, TwinOptions, TwinSummary};
use crate::util::{
    JsonlWriter, now_rfc3339, read_jsonl, sha256_file, write_json_atomic, write_jsonl,
};
use crate::{DataError, DataRoot, Result};

/// Corpora in the order `prepare` walks them when none is given. Common
/// Voice is deliberately absent: it carries Mozilla Data Collective terms
/// (no re-hosting, no re-identifying speakers) and is opt-in only, via an
/// explicit `--corpus common_voice`.
pub const DEFAULT_CORPORA: [&str; 4] = [
    CORPUS_VOICEBANK_DEMAND,
    CORPUS_LIBRITTS_R,
    CORPUS_VCTK,
    CORPUS_LJSPEECH,
];

/// Common Voice corpus id. Not in `unamblify::split` because the split
/// rule needs no special case for it — its speakers (`client_id`) fall
/// through the general rule like any unknown corpus — so it lives here,
/// where the walker is the only user.
pub const CORPUS_COMMON_VOICE: &str = "common_voice";

/// Utterances shorter than this after trimming are dropped, seconds.
pub const MIN_DURATION_S: f64 = 1.0;

/// Rows between `fsync`s of the manifest.
pub const SYNC_EVERY: usize = 1000;

/// Options of one prepare run.
#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    /// Corpora to walk, in order; empty = [`DEFAULT_CORPORA`].
    pub corpora: Vec<String>,
    /// Re-do utterances already in the manifest or rejected.
    pub force: bool,
    /// Rayon threads; `None` = rayon's default.
    pub jobs: Option<usize>,
    /// The augmented-twin pass that follows the main one
    /// (`--noise-share`, `--noise-sets`, `--ham-chain-share`,
    /// `--underdrive-share`, `--seed`).
    pub twins: TwinOptions,
    /// `--twins-only`: run just the twin pass over the manifest as it
    /// stands. The main pass walks every raw corpus and stats both files
    /// of every prepared row before it reaches the twins — millions of
    /// metadata reads that change nothing once the corpus is prepared,
    /// paid again each time a share is adjusted.
    pub twins_only: bool,
}

/// One utterance found by the walker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// Utterance key.
    pub key: String,
    /// Corpus id.
    pub corpus: String,
    /// Speaker id.
    pub speaker: String,
    /// Gender, when the corpus metadata says.
    pub gender: Option<String>,
    /// Subset for the split rule (LibriTTS-R subset, VoiceBank set).
    pub subset: Option<String>,
    /// Source path relative to the data root.
    pub src_rel: String,
    /// Licence id.
    pub licence: String,
}

/// One line of `prepared/rejected.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectRow {
    /// Utterance key.
    pub key: String,
    /// Why.
    pub reason: String,
    /// Source path relative to the data root.
    pub src_path: String,
    /// RFC 3339.
    pub rejected_at: String,
}

/// What preparing one utterance produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Written; the row.
    Row(Box<UtteranceRow>),
    /// Refused; the reason.
    Rejected(RejectRow),
    /// Neither: processing failed (I/O, resampler). Retried next run.
    Failed {
        /// Utterance key.
        key: String,
        /// The error.
        error: String,
    },
}

/// Counts of one run.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct PrepareSummary {
    /// Utterances the walkers found.
    pub sources: u64,
    /// Rows written this run.
    pub written: u64,
    /// Already in the manifest (or rejected) and left alone.
    pub skipped: u64,
    /// Rejected this run.
    pub rejected: u64,
    /// Processing failures this run (not rejected; retried next run).
    pub errors: u64,
    /// Written rows per corpus.
    pub by_corpus: BTreeMap<String, u64>,
    /// Twins planned by the shares (0 when no twin was asked for).
    pub twins_planned: u64,
    /// Twins written this run.
    pub twins_written: u64,
    /// Twins already on disk and left alone.
    pub twins_skipped: u64,
    /// Twin processing failures this run (retried next run).
    pub twins_errors: u64,
}

/// Licence id per corpus.
#[must_use]
pub fn licence_of(corpus: &str) -> &'static str {
    match corpus {
        CORPUS_LJSPEECH => "Public-Domain",
        CORPUS_COMMON_VOICE => {
            "CC0 (Mozilla Data Collective terms — not for public redistribution)"
        }
        _ => "CC-BY-4.0",
    }
}

fn sorted_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| DataError::io(dir, e))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .collect();
    v.sort();
    Ok(v)
}

fn file_name(p: &Path) -> &str {
    p.file_name().and_then(|s| s.to_str()).unwrap_or("")
}

fn stem(p: &Path) -> &str {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("")
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

fn rel(root: &DataRoot, p: &Path) -> String {
    p.strip_prefix(root.path())
        .unwrap_or(p)
        .to_string_lossy()
        .into_owned()
}

/// `raw/vctk/speaker-info.txt` → speaker → gender (`F` / `M`). Also used
/// for VoiceBank, whose speakers are VCTK's.
fn vctk_genders(root: &DataRoot) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let p = root.raw().join("vctk").join("speaker-info.txt");
    if let Ok(text) = fs::read_to_string(&p) {
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() >= 3 && f[0].starts_with('p') {
                m.insert(f[0].to_owned(), f[2].to_owned());
            }
        }
    }
    m
}

/// `raw/libritts_r/LibriTTS_R/speakers.tsv` → reader → gender.
fn libritts_genders(base: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Ok(text) = fs::read_to_string(base.join("speakers.tsv")) {
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 2 {
                m.insert(f[0].trim().to_owned(), f[1].trim().to_owned());
            }
        }
    }
    m
}

/// Walk one corpus. A missing corpus directory is an empty list with a
/// warning, not an error, so a partial fetch prepares what is there.
pub fn walk(root: &DataRoot, corpus: &str) -> Result<Vec<Source>> {
    let out = match corpus {
        CORPUS_LIBRITTS_R => walk_libritts_r(root),
        CORPUS_VOICEBANK_DEMAND => walk_voicebank(root),
        CORPUS_VCTK => walk_vctk(root),
        CORPUS_LJSPEECH => walk_ljspeech(root),
        CORPUS_HIFI_TTS => walk_hifi_tts(root),
        CORPUS_COMMON_VOICE => walk_common_voice(root),
        other => Err(DataError::Invalid(format!("unknown corpus {other:?}"))),
    }?;
    Ok(out)
}

fn missing(dir: &Path, corpus: &str) -> bool {
    if dir.is_dir() {
        false
    } else {
        log::warn!("{corpus}: {} not found, skipping", dir.display());
        true
    }
}

/// LibriTTS-R.
pub fn walk_libritts_r(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root.raw().join("libritts_r").join("LibriTTS_R");
    if missing(&base, CORPUS_LIBRITTS_R) {
        return Ok(Vec::new());
    }
    let genders = libritts_genders(&base);
    let mut out = Vec::new();
    let mut skipped = 0usize;
    for subset in sorted_entries(&base)?.into_iter().filter(|p| p.is_dir()) {
        let subset_name = file_name(&subset).to_owned();
        for reader in sorted_entries(&subset)?.into_iter().filter(|p| p.is_dir()) {
            let reader_name = file_name(&reader).to_owned();
            for chapter in sorted_entries(&reader)?.into_iter().filter(|p| p.is_dir()) {
                for f in sorted_entries(&chapter)? {
                    if !f.is_file() || !has_ext(&f, "wav") {
                        continue;
                    }
                    let Some(k) = key::libritts_r_from_stem(&subset_name, stem(&f)) else {
                        skipped += 1;
                        continue;
                    };
                    out.push(Source {
                        key: k,
                        corpus: CORPUS_LIBRITTS_R.to_owned(),
                        speaker: reader_name.clone(),
                        gender: genders.get(&reader_name).cloned(),
                        subset: Some(subset_name.clone()),
                        src_rel: rel(root, &f),
                        licence: licence_of(CORPUS_LIBRITTS_R).to_owned(),
                    });
                }
            }
        }
    }
    if skipped > 0 {
        log::info!("libritts_r: skipped {skipped} .wav files whose names are not utterances");
    }
    Ok(out)
}

/// Hi-Fi TTS speakers left out: on the naturalness judge that scores
/// LibriTTS-R at 4.1 and Common Voice at 3.15, the corpus averages 3.9
/// (clean tier 3.94, other 3.86) but these two score 3.5, and a target
/// teaches the model its own recording (experiment log #34).
pub const HIFI_TTS_EXCLUDED_SPEAKERS: [&str; 2] = ["11614", "6671"];

/// Hi-Fi TTS: `hi_fi_tts_v0/audio/<speaker>_<quality>/<book>/<stem>.flac`,
/// 44.1 kHz. The quality tier is the subset; the JSON manifests beside
/// `audio/` are not read (the audio and the layout say everything the
/// pipeline needs). No gender metadata ships with it.
pub fn walk_hifi_tts(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root
        .raw()
        .join("hifi_tts")
        .join("hi_fi_tts_v0")
        .join("audio");
    if missing(&base, CORPUS_HIFI_TTS) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut excluded = 0usize;
    for spk_dir in sorted_entries(&base)?.into_iter().filter(|p| p.is_dir()) {
        let Some((speaker, quality)) = file_name(&spk_dir).split_once('_') else {
            continue;
        };
        if HIFI_TTS_EXCLUDED_SPEAKERS.contains(&speaker) {
            excluded += 1;
            continue;
        }
        let (speaker, quality) = (speaker.to_owned(), quality.to_owned());
        for book in sorted_entries(&spk_dir)?.into_iter().filter(|p| p.is_dir()) {
            let book_name = file_name(&book).to_owned();
            for f in sorted_entries(&book)? {
                if !f.is_file() || !has_ext(&f, "flac") {
                    continue;
                }
                out.push(Source {
                    key: key::hifi_tts(&speaker, &quality, &book_name, stem(&f)),
                    corpus: CORPUS_HIFI_TTS.to_owned(),
                    speaker: speaker.clone(),
                    gender: None,
                    subset: Some(quality.clone()),
                    src_rel: rel(root, &f),
                    licence: licence_of(CORPUS_HIFI_TTS).to_owned(),
                });
            }
        }
    }
    if excluded > 0 {
        log::info!("hifi_tts: {excluded} speaker set(s) excluded by HIFI_TTS_EXCLUDED_SPEAKERS");
    }
    Ok(out)
}

/// VoiceBank-DEMAND clean sets.
pub fn walk_voicebank(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root.raw().join("voicebank_demand");
    if missing(&base, CORPUS_VOICEBANK_DEMAND) {
        return Ok(Vec::new());
    }
    let genders = vctk_genders(root);
    let mut out = Vec::new();
    for (dir, set) in [
        ("clean_trainset_28spk_wav", "train"),
        ("clean_testset_wav", "test"),
    ] {
        let d = base.join(dir);
        if !d.is_dir() {
            log::warn!("voicebank_demand: {} not found", d.display());
            continue;
        }
        for f in sorted_entries(&d)? {
            if !f.is_file() || !has_ext(&f, "wav") {
                continue;
            }
            let Some((spk, utt)) = stem(&f).split_once('_') else {
                continue;
            };
            if !spk.starts_with('p') || utt.is_empty() {
                continue;
            }
            out.push(Source {
                key: key::voicebank_demand(set, spk, utt),
                corpus: CORPUS_VOICEBANK_DEMAND.to_owned(),
                speaker: spk.to_owned(),
                gender: genders.get(spk).cloned(),
                subset: Some(set.to_owned()),
                src_rel: rel(root, &f),
                licence: licence_of(CORPUS_VOICEBANK_DEMAND).to_owned(),
            });
        }
    }
    Ok(out)
}

/// VCTK, `mic2` preferred, one row per utterance.
pub fn walk_vctk(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root.raw().join("vctk").join("wav48_silence_trimmed");
    if missing(&base, CORPUS_VCTK) {
        return Ok(Vec::new());
    }
    let genders = vctk_genders(root);
    let mut out = Vec::new();
    for spk_dir in sorted_entries(&base)?.into_iter().filter(|p| p.is_dir()) {
        let spk = file_name(&spk_dir).to_owned();
        // (utt) → (mic, path); a later mic2 replaces mic1.
        let mut best: BTreeMap<String, (u8, PathBuf)> = BTreeMap::new();
        for f in sorted_entries(&spk_dir)? {
            if !f.is_file() || !has_ext(&f, "flac") {
                continue;
            }
            let s = stem(&f);
            let Some((rest, mic)) = s.rsplit_once("_mic") else {
                continue;
            };
            let Ok(mic) = mic.parse::<u8>() else {
                continue;
            };
            let Some((s2, utt)) = rest.split_once('_') else {
                continue;
            };
            if s2 != spk {
                continue;
            }
            let entry = best.entry(utt.to_owned()).or_insert((mic, f.clone()));
            if mic == 2 && entry.0 != 2 {
                *entry = (mic, f.clone());
            }
        }
        for (utt, (mic, path)) in best {
            out.push(Source {
                key: key::vctk(&spk, &utt, mic),
                corpus: CORPUS_VCTK.to_owned(),
                speaker: spk.clone(),
                gender: genders.get(&spk).cloned(),
                subset: None,
                src_rel: rel(root, &path),
                licence: licence_of(CORPUS_VCTK).to_owned(),
            });
        }
    }
    Ok(out)
}

/// LJSpeech.
pub fn walk_ljspeech(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root
        .raw()
        .join("ljspeech")
        .join("LJSpeech-1.1")
        .join("wavs");
    if missing(&base, CORPUS_LJSPEECH) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for f in sorted_entries(&base)? {
        if !f.is_file() || !has_ext(&f, "wav") {
            continue;
        }
        let s = stem(&f);
        let Some((book, utt)) = s.strip_prefix("LJ").and_then(|r| r.split_once('-')) else {
            continue;
        };
        let (Ok(book), Ok(utt)) = (book.parse::<u32>(), utt.parse::<u32>()) else {
            continue;
        };
        out.push(Source {
            key: key::ljspeech(book, utt),
            corpus: CORPUS_LJSPEECH.to_owned(),
            speaker: "LJ".to_owned(),
            gender: Some("F".to_owned()),
            subset: None,
            src_rel: rel(root, &f),
            licence: licence_of(CORPUS_LJSPEECH).to_owned(),
        });
    }
    Ok(out)
}

/// One English Common Voice variant, keyed by its directory under
/// `raw/common_voice/`. The `en-AU` export differs from a standard release
/// — comma-separated CSV metadata, clips under `audio_files/` — so the
/// clips subdirectory and delimiter are per-variant. Non-English
/// directories on disk (`de`, `ja`, …) are simply not listed here.
struct CvVariant {
    /// Directory under `raw/common_voice/` and the key's `<variant>`.
    dir: &'static str,
    /// Subdirectory holding the `.mp3` clips.
    clips: &'static str,
    /// Metadata column delimiter (`\t` for `.tsv`, `,` for the CSV export).
    delim: char,
}

/// The English variants ingested, English only, in a stable order.
const COMMON_VOICE_VARIANTS: [CvVariant; 3] = [
    CvVariant {
        dir: "en",
        clips: "clips",
        delim: '\t',
    },
    CvVariant {
        dir: "en-AU",
        clips: "audio_files",
        delim: ',',
    },
    CvVariant {
        dir: "cv26-southern-american-english",
        clips: "clips",
        delim: '\t',
    },
];

/// The nearest directory at or under `base` (searching at most `max_depth`
/// levels down) that has a subdirectory named `child`; `base` itself
/// qualifies. Resolves a variant's dataset directory without hard-coding
/// the release-version folder (`cv-corpus-26.0-…`), which changes.
fn find_dir_with_child(base: &Path, child: &str, max_depth: usize) -> Option<PathBuf> {
    if base.join(child).is_dir() {
        return Some(base.to_path_buf());
    }
    if max_depth == 0 {
        return None;
    }
    let mut subs: Vec<PathBuf> = fs::read_dir(base)
        .ok()?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subs.sort();
    subs.iter()
        .find_map(|s| find_dir_with_child(s, child, max_depth - 1))
}

/// The metadata table(s) to read in a Common Voice dataset directory: the
/// validated set when present, else the union of the `train`/`dev`/`test`
/// splits; for the CSV export, the main table (never a `-split` sibling,
/// which repeats the same rows).
fn cv_metadata_files(dir: &Path, delim: char) -> Vec<PathBuf> {
    if delim == '\t' {
        let validated = dir.join("validated.tsv");
        if validated.is_file() {
            return vec![validated];
        }
        return ["train.tsv", "dev.tsv", "test.tsv"]
            .iter()
            .map(|n| dir.join(n))
            .filter(|p| p.is_file())
            .collect();
    }
    let mut csvs: Vec<PathBuf> = sorted_entries(dir)
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.is_file() && has_ext(p, "csv"))
        .collect();
    csvs.sort();
    if let Some(main) = csvs.iter().find(|p| !file_name(p).contains("split")) {
        return vec![main.clone()];
    }
    csvs.into_iter().take(1).collect()
}

/// The `client_id` and `path` column indices from a header row.
fn cv_column_indices(header: &str, delim: char) -> Option<(usize, usize)> {
    let cols: Vec<&str> = header.trim_end_matches(['\r', '\n']).split(delim).collect();
    let find = |name: &str| cols.iter().position(|c| c.trim_matches('"') == name);
    Some((find("client_id")?, find("path")?))
}

/// Stream one metadata table, recording mp3-stem → `client_id` only for
/// stems in `present`, so memory stays bounded by the clips actually on
/// disk even when the table lists millions of rows. The first `client_id`
/// seen for a stem wins. **Only `client_id` and `path` are read; the
/// transcript and every other column are ignored** (Mozilla Data
/// Collective terms — the `client_id` is used verbatim and linked to
/// nothing).
fn read_cv_table(
    path: &Path,
    delim: char,
    present: &HashSet<String>,
    speaker_of: &mut HashMap<String, String>,
) -> Result<()> {
    let file = fs::File::open(path).map_err(|e| DataError::io(path, e))?;
    let mut lines = std::io::BufReader::new(file).lines();
    let Some(header) = lines.next() else {
        return Ok(());
    };
    let header = header.map_err(|e| DataError::io(path, e))?;
    let Some((ci, pi)) = cv_column_indices(&header, delim) else {
        log::warn!(
            "common_voice: {} has no client_id/path columns",
            path.display()
        );
        return Ok(());
    };
    for line in lines {
        let line = line.map_err(|e| DataError::io(path, e))?;
        let fields: Vec<&str> = line.split(delim).collect();
        let (Some(cid), Some(rel_path)) = (fields.get(ci), fields.get(pi)) else {
            continue;
        };
        let Some(s) = Path::new(rel_path.trim_matches('"'))
            .file_stem()
            .and_then(|s| s.to_str())
        else {
            continue;
        };
        if present.contains(s) {
            speaker_of
                .entry(s.to_owned())
                .or_insert_with(|| cid.trim_matches('"').to_owned());
        }
    }
    Ok(())
}

/// Common Voice English (`en`, `en-AU`, `cv26-southern-american-english`).
/// Like the other corpora, the walk is driven by the clips on disk: a
/// table row without its audio yields no source. The speaker is the opaque
/// `client_id` and nothing else is derived from it; the transcript is never
/// read. Key: `common_voice/<variant>/<mp3-stem>`.
pub fn walk_common_voice(root: &DataRoot) -> Result<Vec<Source>> {
    let base = root.raw().join("common_voice");
    if missing(&base, CORPUS_COMMON_VOICE) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for v in &COMMON_VOICE_VARIANTS {
        let variant_root = base.join(v.dir);
        if !variant_root.is_dir() {
            log::warn!("common_voice: variant {} not found, skipping", v.dir);
            continue;
        }
        let Some(dataset) = find_dir_with_child(&variant_root, v.clips, 3) else {
            log::warn!("common_voice/{}: no {}/ directory found", v.dir, v.clips);
            continue;
        };
        let clips = dataset.join(v.clips);
        let entries = sorted_entries(&clips)?;
        let present: HashSet<String> = entries
            .iter()
            .filter(|p| p.is_file() && has_ext(p, "mp3"))
            .map(|p| stem(p).to_owned())
            .collect();
        if present.is_empty() {
            log::warn!("common_voice/{}: no .mp3 clips under {}", v.dir, v.clips);
            continue;
        }
        let mut speaker_of: HashMap<String, String> = HashMap::with_capacity(present.len());
        for meta in cv_metadata_files(&dataset, v.delim) {
            read_cv_table(&meta, v.delim, &present, &mut speaker_of)?;
        }
        let mut without_speaker = 0usize;
        for f in entries {
            if !f.is_file() || !has_ext(&f, "mp3") {
                continue;
            }
            let s = stem(&f);
            let Some(client_id) = speaker_of.get(s) else {
                without_speaker += 1;
                continue;
            };
            out.push(Source {
                key: format!("{CORPUS_COMMON_VOICE}/{}/{s}", v.dir),
                corpus: CORPUS_COMMON_VOICE.to_owned(),
                speaker: client_id.clone(),
                gender: None,
                subset: None,
                src_rel: rel(root, &f),
                licence: licence_of(CORPUS_COMMON_VOICE).to_owned(),
            });
        }
        if without_speaker > 0 {
            log::info!(
                "common_voice/{}: skipped {without_speaker} clips absent from the metadata table",
                v.dir
            );
        }
    }
    Ok(out)
}

/// Bytes of an s16 mono WAV with `n` samples as hound writes it.
fn wav_bytes(n: u64) -> u64 {
    44 + 2 * n
}

/// Whether the manifest row's files exist with the sizes its duration
/// implies.
#[must_use]
pub fn files_consistent(root: &DataRoot, row: &UtteranceRow) -> bool {
    let n16 = (row.duration_s * f64::from(WIDEBAND_SAMPLE_RATE)).round();
    let n8 = (row.duration_s * f64::from(VOCODER_SAMPLE_RATE)).round();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (n16, n8) = (n16.max(0.0) as u64, n8.max(0.0) as u64);
    let ok = |wav: PathBuf, flac: PathBuf, n: u64| {
        flac.is_file() || fs::metadata(wav).is_ok_and(|m| m.len() == wav_bytes(n))
    };
    ok(
        root.prepared_16k(&row.key),
        root.prepared_16k_flac(&row.key),
        n16,
    ) && ok(
        root.prepared_8k(&row.key),
        root.prepared_8k_flac(&row.key),
        n8,
    )
}

/// Prepare one utterance: decode, resample, trim, normalise, write both
/// rates, hash. Returns the row or a rejection; an `Err` is an I/O
/// failure writing the outputs.
pub fn prepare_one(root: &DataRoot, src: &Source) -> Result<Outcome> {
    let reject = |reason: String| {
        Ok(Outcome::Rejected(RejectRow {
            key: src.key.clone(),
            reason,
            src_path: src.src_rel.clone(),
            rejected_at: now_rfc3339(),
        }))
    };
    let src_path = root.path().join(&src.src_rel);
    let audio = match read(&src_path) {
        Ok(a) => a,
        Err(e) => return reject(format!("unreadable: {e}")),
    };
    if audio.samples.is_empty() {
        return reject("empty".to_owned());
    }
    let x16 = resample(&audio.samples, audio.rate, WIDEBAND_SAMPLE_RATE)?;
    let trimmed = trim_silence(
        &x16,
        WIDEBAND_SAMPLE_RATE,
        TRIM_THRESHOLD_DBFS,
        TRIM_MARGIN_S,
    );
    if trimmed.samples.is_empty() {
        return reject("silence: no frame above the trim threshold".to_owned());
    }
    let Some(rms_in) = active_rms_dbfs(&trimmed.samples, WIDEBAND_SAMPLE_RATE) else {
        return reject("silence: no active frame".to_owned());
    };
    let (y16, gain_db) = normalize_rms(
        &trimmed.samples,
        WIDEBAND_SAMPLE_RATE,
        TARGET_DBFS,
        MAX_GAIN_DB,
    );
    let peak = y16.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1.0 {
        return reject(format!(
            "clipping after {gain_db:+.1} dB gain (peak {peak:.3})"
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let duration_s = y16.len() as f64 / f64::from(WIDEBAND_SAMPLE_RATE);
    if duration_s < MIN_DURATION_S {
        return reject(format!("too short: {duration_s:.2} s after trimming"));
    }
    let y8 = resample(&y16, WIDEBAND_SAMPLE_RATE, VOCODER_SAMPLE_RATE)?;

    let p16 = root.prepared_16k_flac(&src.key);
    let p8 = root.prepared_8k_flac(&src.key);
    if let Some(parent) = p16.parent() {
        fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    write_flac_s16(&p16, &y16, WIDEBAND_SAMPLE_RATE)?;
    write_flac_s16(&p8, &y8, VOCODER_SAMPLE_RATE)?;
    let row = UtteranceRow {
        key: src.key.clone(),
        corpus: src.corpus.clone(),
        speaker: src.speaker.clone(),
        gender: src.gender.clone(),
        split: assign(&src.corpus, &src.speaker, src.subset.as_deref()),
        duration_s,
        src_rate: audio.rate,
        src_path: src.src_rel.clone(),
        licence: src.licence.clone(),
        rms_dbfs_in: f64::from(rms_in),
        gain_db: f64::from(gain_db),
        trim_lead_s: trimmed.lead_s(WIDEBAND_SAMPLE_RATE),
        trim_tail_s: trimmed.tail_s(WIDEBAND_SAMPLE_RATE),
        sha256_16k: sha256_file(&p16)?,
        sha256_8k: sha256_file(&p8)?,
        prepared_at: now_rfc3339(),
        parent: None,
        aug: None,
    };
    Ok(Outcome::Row(Box::new(row)))
}

/// What `--resplit` did.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ResplitSummary {
    /// Manifest rows examined.
    pub rows: u64,
    /// Rows whose split changed.
    pub moved: u64,
    /// `"<from>->to"` → rows, for every move that happened.
    pub moves: BTreeMap<String, u64>,
    /// Rows per split afterwards.
    pub by_split: BTreeMap<String, u64>,
}

/// Re-apply [`assign`] to every row of `prepared/manifest.jsonl` in
/// place — no audio is touched — after a change to the split rule. The
/// captured manifests carry no split of their own; the shard stage and
/// the pipeline loader take it from here, so rebuild any shard set
/// afterwards. A twin follows its own row, whose corpus, speaker and
/// subset are its parent's.
pub fn resplit(root: &DataRoot) -> Result<ResplitSummary> {
    let path = root.prepared_manifest();
    let mut rows: Vec<UtteranceRow> = read_jsonl(&path)?;
    let mut summary = ResplitSummary::default();
    for r in &mut rows {
        let new = assign(&r.corpus, &r.speaker, subset_of_key(&r.key));
        summary.rows += 1;
        if new != r.split {
            *summary
                .moves
                .entry(format!("{}->{new}", r.split))
                .or_default() += 1;
            summary.moved += 1;
            r.split = new;
        }
        *summary.by_split.entry(new.to_string()).or_default() += 1;
    }
    if summary.moved > 0 {
        write_jsonl(&path, &rows)?;
    }
    Ok(summary)
}

/// The corpus subset a key encodes — `libritts_r/dev-clean/…` →
/// `dev-clean`, `voicebank_demand/train/…` → `train` — when it has one
/// (`vctk/…` and `ljspeech/…` do not).
fn subset_of_key(key: &str) -> Option<&str> {
    let mut parts = key.split('/');
    let _corpus = parts.next()?;
    let second = parts.next()?;
    parts.next().map(|_| second)
}

/// `--twins-only`: the twin pass alone, over the manifest as it stands.
fn run_twins_only(root: &DataRoot, opts: &PrepareOptions) -> Result<PrepareSummary> {
    if !opts.twins.enabled() {
        return Err(DataError::Invalid(
            "--twins-only needs a twin share: --noise-share, --ham-chain-share or \
             --underdrive-share"
                .to_owned(),
        ));
    }
    // No parent was redone, so no existing twin is stale.
    let mut summary = PrepareSummary::default();
    twin_pass(root, opts, &HashSet::new(), &mut summary)?;
    Ok(summary)
}

/// The whole stage.
pub fn run(root: &DataRoot, opts: &PrepareOptions) -> Result<PrepareSummary> {
    if opts.twins_only {
        return run_twins_only(root, opts);
    }
    let corpora: Vec<String> = if opts.corpora.is_empty() {
        DEFAULT_CORPORA.iter().map(|&s| s.to_owned()).collect()
    } else {
        opts.corpora.clone()
    };
    let mut sources = Vec::new();
    for c in &corpora {
        let found = walk(root, c)?;
        log::info!("{c}: {} utterances found", found.len());
        sources.extend(found);
    }
    let mut summary = PrepareSummary {
        sources: sources.len() as u64,
        ..PrepareSummary::default()
    };

    let manifest_path = root.prepared_manifest();
    let rejected_path = root.prepared_rejected();
    let existing: Vec<UtteranceRow> = read_jsonl(&manifest_path)?;
    let rejected: Vec<RejectRow> = read_jsonl(&rejected_path)?;

    let todo: Vec<Source> = if opts.force {
        sources
    } else {
        let by_key: HashMap<&str, &UtteranceRow> =
            existing.iter().map(|r| (r.key.as_str(), r)).collect();
        // Rows an earlier build wrote for processing errors are not
        // rejections; they are redone (and dropped from the file below).
        let rejected_keys: HashSet<&str> = rejected
            .iter()
            .filter(|r| !r.reason.starts_with("error:"))
            .map(|r| r.key.as_str())
            .collect();
        sources
            .into_iter()
            .filter(|s| {
                let done = by_key
                    .get(s.key.as_str())
                    .is_some_and(|r| files_consistent(root, r))
                    || rejected_keys.contains(s.key.as_str());
                if done {
                    summary.skipped += 1;
                }
                !done
            })
            .collect()
    };
    // Whatever is about to be redone (forced, or a row whose files went
    // missing) leaves the manifests first, so a key never appears twice.
    let redo: HashSet<&str> = todo.iter().map(|s| s.key.as_str()).collect();
    if existing.iter().any(|r| redo.contains(r.key.as_str())) {
        rewrite_without(&manifest_path, &existing, &redo)?;
    }
    if rejected.iter().any(|r| redo.contains(r.key.as_str())) {
        rewrite_without(&rejected_path, &rejected, &redo)?;
    }
    log::info!("prepare: {} to do, {} skipped", todo.len(), summary.skipped);
    if todo.is_empty() {
        twin_pass(root, opts, &HashSet::new(), &mut summary)?;
        return Ok(summary);
    }
    let attempted: HashSet<String> = todo.iter().map(|s| s.key.clone()).collect();

    let (tx, rx) = channel::<Outcome>();
    let writer = spawn_writer(rx, manifest_path.clone(), rejected_path.clone());

    let work = |src: &Source, tx: &Sender<Outcome>| {
        let outcome = match prepare_one(root, src) {
            Ok(o) => o,
            Err(e) => {
                log::error!("{}: {e} (will be retried next run)", src.key);
                Outcome::Failed {
                    key: src.key.clone(),
                    error: e.to_string(),
                }
            }
        };
        let _ = tx.send(outcome);
    };
    match opts.jobs {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| DataError::Invalid(format!("rayon: {e}")))?;
            pool.install(|| todo.par_iter().for_each_with(tx, |tx, s| work(s, tx)));
        }
        None => todo.par_iter().for_each_with(tx, |tx, s| work(s, tx)),
    }
    let (written, rejected, errors, by_corpus) = writer
        .join()
        .unwrap_or_else(|_| Err(DataError::Invalid("manifest writer panicked".to_owned())))?;
    summary.written = written;
    summary.rejected = rejected;
    summary.errors = errors;
    summary.by_corpus = by_corpus;
    log::info!(
        "prepare: {} written, {} rejected, {} skipped, {} failed (retried next run)",
        summary.written,
        summary.rejected,
        summary.skipped,
        summary.errors
    );
    twin_pass(root, opts, &attempted, &mut summary)?;
    Ok(summary)
}

/// The twin pass after the main one, when a share asks for it. A parent
/// attempted this run has its twin redone (its files may have changed).
fn twin_pass(
    root: &DataRoot,
    opts: &PrepareOptions,
    attempted: &HashSet<String>,
    summary: &mut PrepareSummary,
) -> Result<()> {
    if !opts.twins.enabled() {
        return Ok(());
    }
    let TwinSummary {
        planned,
        written,
        skipped,
        errors,
        ..
    } = twins::run(root, &opts.twins, attempted, opts.force, opts.jobs)?;
    summary.twins_planned = planned;
    summary.twins_written = written;
    summary.twins_skipped = skipped;
    summary.twins_errors = errors;
    Ok(())
}

/// Counts the writer thread returns: rows written, rows rejected,
/// processing failures, rows per corpus.
pub(crate) type WriterCounts = (u64, u64, u64, BTreeMap<String, u64>);

/// The single manifest writer: appends every outcome, `fsync`s the
/// manifest every [`SYNC_EVERY`] rows and at the end, rejections at once.
pub(crate) fn spawn_writer(
    rx: std::sync::mpsc::Receiver<Outcome>,
    manifest_path: PathBuf,
    rejected_path: PathBuf,
) -> std::thread::JoinHandle<Result<WriterCounts>> {
    std::thread::spawn(move || {
        let mut manifest = JsonlWriter::open(&manifest_path)?;
        let mut rejects: Option<JsonlWriter> = None;
        let (mut written, mut rejected, mut errors) = (0u64, 0u64, 0u64);
        let mut by_corpus = BTreeMap::new();
        while let Ok(outcome) = rx.recv() {
            match outcome {
                Outcome::Row(row) => {
                    manifest.append(&*row)?;
                    written += 1;
                    *by_corpus.entry(row.corpus).or_insert(0) += 1;
                    if manifest.rows_since_sync() >= SYNC_EVERY {
                        manifest.sync()?;
                    }
                    if written.is_multiple_of(500) {
                        log::info!("prepare: {written} written, {rejected} rejected");
                    }
                }
                Outcome::Rejected(r) => {
                    log::warn!("reject {}: {}", r.key, r.reason);
                    let w = match rejects.as_mut() {
                        Some(w) => w,
                        None => rejects.insert(JsonlWriter::open(&rejected_path)?),
                    };
                    w.append(&r)?;
                    w.sync()?;
                    rejected += 1;
                }
                Outcome::Failed { key, error } => {
                    log::warn!("failed {key}: {error}");
                    errors += 1;
                }
            }
        }
        manifest.sync()?;
        Ok((written, rejected, errors, by_corpus))
    })
}

fn rewrite_without<T: Serialize + HasKey>(
    path: &Path,
    rows: &[T],
    drop: &HashSet<&str>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    write_jsonl(path, rows.iter().filter(|r| !drop.contains(r.key())))
}

/// Rows that carry an utterance key.
trait HasKey {
    fn key(&self) -> &str;
}

impl HasKey for UtteranceRow {
    fn key(&self) -> &str {
        &self.key
    }
}

impl HasKey for RejectRow {
    fn key(&self) -> &str {
        &self.key
    }
}

/// Summary of `prepared/manifest.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PreparedStats {
    /// Rows.
    pub rows: u64,
    /// Hours.
    pub hours: f64,
    /// Rows per corpus.
    pub by_corpus: BTreeMap<String, u64>,
    /// Rows per split.
    pub by_split: BTreeMap<unamblify::Split, u64>,
    /// Distinct speakers.
    pub speakers: u64,
    /// Rejected rows.
    pub rejected: u64,
    /// Rows that are augmented twins (counted in `rows` and `hours` too).
    pub twins: u64,
}

/// Compute [`PreparedStats`].
pub fn stats(root: &DataRoot) -> Result<PreparedStats> {
    let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest())?;
    let rejected: Vec<RejectRow> = read_jsonl(&root.prepared_rejected())?;
    let mut by_corpus = BTreeMap::new();
    let mut by_split = BTreeMap::new();
    let mut speakers = HashSet::new();
    let mut secs = 0.0;
    let mut twins = 0u64;
    for r in &rows {
        *by_corpus.entry(r.corpus.clone()).or_insert(0) += 1;
        *by_split.entry(r.split).or_insert(0) += 1;
        speakers.insert((r.corpus.as_str(), r.speaker.as_str()));
        secs += r.duration_s;
        if r.parent.is_some() {
            twins += 1;
        }
    }
    Ok(PreparedStats {
        rows: rows.len() as u64,
        hours: secs / 3_600.0,
        by_corpus,
        by_split,
        speakers: speakers.len() as u64,
        rejected: rejected.len() as u64,
        twins,
    })
}

/// Write a JSON summary next to the manifest (`prepared/summary.json`).
pub fn write_summary(root: &DataRoot) -> Result<PreparedStats> {
    let s = stats(root)?;
    write_json_atomic(&root.prepared().join("summary.json"), &s)?;
    Ok(s)
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::testutil::sine;
    use unamblify::Split;
    use unamblify_audio::write_wav_s16;

    /// A raw tree with every corpus layout, made of generated WAVs (and
    /// the checked-in FLAC fixture for VCTK).
    /// `--resplit` rewrites only the split column, from the corpus,
    /// speaker and the subset the key encodes; nothing else moves.
    /// Hi-Fi TTS: the corpus's own `audio/<speaker>_<quality>/<book>/`
    /// layout gives the key, the speaker and the subset; the two speakers
    /// the judge scored at 3.5 are left out; every row is `train`.
    #[test]
    fn hifi_tts_walks_its_layout_and_leaves_the_excluded_speakers_out() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let audio = root
            .raw()
            .join("hifi_tts")
            .join("hi_fi_tts_v0")
            .join("audio");
        let tone: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        for (set, book, stem) in [
            ("92_clean", "12345", "somebook_01_author_0007"),
            ("6097_other", "777", "another_02_reader_0001"),
            ("11614_other", "12352", "prideofjennico_01_castle_0028"),
        ] {
            let d = audio.join(set).join(book);
            fs::create_dir_all(&d).unwrap();
            write_flac_s16(d.join(format!("{stem}.flac")), &tone, 44_100).unwrap();
            fs::write(d.join("notes.txt"), "not audio").unwrap();
        }
        let out = walk_hifi_tts(&root).unwrap();
        let keys: Vec<&str> = out.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "hifi_tts/6097_other/777/another_02_reader_0001",
                "hifi_tts/92_clean/12345/somebook_01_author_0007",
            ],
            "sorted, flac only, 11614 excluded"
        );
        let first = &out[1];
        assert_eq!(
            (first.speaker.as_str(), first.subset.as_deref()),
            ("92", Some("clean"))
        );
        assert_eq!(first.corpus, CORPUS_HIFI_TTS);
        assert_eq!(first.licence, "CC-BY-4.0");
        assert!(first.gender.is_none());
        assert!(
            first
                .src_rel
                .ends_with("92_clean/12345/somebook_01_author_0007.flac")
        );
        for s in &out {
            assert_eq!(
                assign(&s.corpus, &s.speaker, s.subset.as_deref()),
                Split::Train
            );
        }
        assert!(
            walk_hifi_tts(&DataRoot::new(dir.path().join("nowhere")))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn resplit_reassigns_rows_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let row = |key: &str, speaker: &str, split: Split| UtteranceRow {
            key: key.to_owned(),
            corpus: key.split('/').next().unwrap().to_owned(),
            speaker: speaker.to_owned(),
            gender: None,
            split,
            duration_s: 1.5,
            src_rate: 24_000,
            src_path: String::new(),
            licence: "CC-BY-4.0".to_owned(),
            rms_dbfs_in: -20.0,
            gain_db: 0.0,
            trim_lead_s: 0.0,
            trim_tail_s: 0.0,
            sha256_16k: String::new(),
            sha256_8k: String::new(),
            prepared_at: String::new(),
            parent: None,
            aug: None,
        };
        let rows = vec![
            // A dev-clean reader above the hash limit: was dev, now trains.
            row(
                "libritts_r/dev-clean/1188_133604_000001_000000",
                "1188",
                Split::Dev,
            ),
            // One below it stays dev; an eval reader stays dev.
            row(
                "libritts_r/dev-clean/422_122949_000001_000000",
                "422",
                Split::Dev,
            ),
            row(
                "libritts_r/dev-clean/84_121123_000007_000001",
                "84",
                Split::Dev,
            ),
            // VCTK is untouched by the change.
            row("vctk/p225_001_mic2", "p225", Split::Train),
            row("voicebank_demand/train/p228_001", "p228", Split::Dev),
        ];
        write_jsonl(&root.prepared_manifest(), &rows).unwrap();
        let s = resplit(&root).unwrap();
        assert_eq!((s.rows, s.moved), (5, 1));
        assert_eq!(s.moves.get("dev->train"), Some(&1));
        assert_eq!((s.by_split["train"], s.by_split["dev"]), (2, 3));
        let back: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(back[0].split, Split::Train);
        assert!(back[1..].iter().zip(&rows[1..]).all(|(a, b)| a == b));
        // A second pass moves nothing and leaves the file alone.
        assert_eq!(resplit(&root).unwrap().moved, 0);
        assert_eq!(subset_of_key("vctk/p225_001_mic2"), None);
        assert_eq!(
            subset_of_key("libritts_r/dev-clean/84_121123_000007_000001+n0001"),
            Some("dev-clean")
        );
    }

    fn fixture(dir: &Path) -> DataRoot {
        let root = DataRoot::new(dir);
        let raw = root.raw();
        let speech = |dur_s: f32, rate: u32, f: f32| {
            let n = (dur_s * rate as f32) as usize;
            let mut x = vec![0.0f32; rate as usize / 2]; // 0.5 s lead silence
            x.extend(sine(f, rate, n, 0.2));
            x.extend(vec![0.0f32; rate as usize / 4]);
            x
        };
        let w = |p: PathBuf, x: &[f32], rate: u32| {
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            write_wav_s16(&p, x, rate).unwrap();
        };
        // LibriTTS-R: two subsets, a doc file that must be skipped.
        let lt = raw.join("libritts_r/LibriTTS_R");
        w(
            lt.join("dev-clean/84/121123/84_121123_000007_000001.wav"),
            &speech(1.5, 24_000, 220.0),
            24_000,
        );
        w(
            lt.join("train-clean-100/19/198/19_198_000000_000000.wav"),
            &speech(2.0, 24_000, 330.0),
            24_000,
        );
        w(
            lt.join("train-clean-100/19/198/19_198.wav"),
            &speech(2.0, 24_000, 330.0),
            24_000,
        );
        fs::write(
            lt.join("speakers.tsv"),
            "READER\tGENDER\tSUBSET\tNAME\n84\tF\tdev-clean\tx\n19\tM\ttrain-clean-100\ty\n",
        )
        .unwrap();
        // VoiceBank: train + test, noisy ignored; one too-short file.
        let vb = raw.join("voicebank_demand");
        w(
            vb.join("clean_trainset_28spk_wav/p226_001.wav"),
            &speech(1.2, 48_000, 200.0),
            48_000,
        );
        w(
            vb.join("clean_testset_wav/p232_001.wav"),
            &speech(1.1, 48_000, 250.0),
            48_000,
        );
        w(
            vb.join("clean_testset_wav/p257_002.wav"),
            &speech(0.3, 48_000, 250.0),
            48_000,
        );
        w(
            vb.join("noisy_testset_wav/p232_001.wav"),
            &speech(1.1, 48_000, 250.0),
            48_000,
        );
        // VCTK: the checked-in FLAC (0.1 s — too short) as mic1 and mic2
        // for one utterance, and mic1 only for another.
        let flac = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../unamblify-audio/tests/fixtures/sine440-stereo-24bit-16k.flac");
        let vc = raw.join("vctk/wav48_silence_trimmed/p225");
        fs::create_dir_all(&vc).unwrap();
        for name in [
            "p225_001_mic1.flac",
            "p225_001_mic2.flac",
            "p225_002_mic1.flac",
        ] {
            fs::copy(&flac, vc.join(name)).unwrap();
        }
        fs::write(
            raw.join("vctk/speaker-info.txt"),
            "ID  AGE  GENDER  ACCENTS  REGION\np225  23  F  English  Southern  England\n",
        )
        .unwrap();
        // LJSpeech.
        w(
            raw.join("ljspeech/LJSpeech-1.1/wavs/LJ001-0001.wav"),
            &speech(1.6, 22_050, 180.0),
            22_050,
        );
        root
    }

    #[test]
    fn walkers_find_the_survey_layouts() {
        let dir = tempfile::tempdir().unwrap();
        let root = fixture(dir.path());
        let lt = walk_libritts_r(&root).unwrap();
        let keys: Vec<&str> = lt.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "libritts_r/dev-clean/84_121123_000007_000001",
                "libritts_r/train-clean-100/19_198_000000_000000"
            ]
        );
        assert_eq!(lt[0].gender.as_deref(), Some("F"));
        assert_eq!(lt[0].subset.as_deref(), Some("dev-clean"));
        assert_eq!(lt[0].speaker, "84");
        assert_eq!(
            lt[0].src_rel,
            "raw/libritts_r/LibriTTS_R/dev-clean/84/121123/84_121123_000007_000001.wav"
        );

        let vb = walk_voicebank(&root).unwrap();
        let keys: Vec<&str> = vb.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "voicebank_demand/train/p226_001",
                "voicebank_demand/test/p232_001",
                "voicebank_demand/test/p257_002"
            ]
        );
        assert_eq!(vb[0].subset.as_deref(), Some("train"));

        let vc = walk_vctk(&root).unwrap();
        let keys: Vec<&str> = vc.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, vec!["vctk/p225_001_mic2", "vctk/p225_002_mic1"]);
        assert_eq!(vc[0].gender.as_deref(), Some("F"));
        assert!(vc[0].src_rel.ends_with("p225_001_mic2.flac"));

        let lj = walk_ljspeech(&root).unwrap();
        assert_eq!(lj.len(), 1);
        assert_eq!(lj[0].key, "ljspeech/LJ001-0001");
        assert_eq!(lj[0].speaker, "LJ");
        assert_eq!(lj[0].licence, "Public-Domain");

        assert!(walk(&root, "nope").is_err());
        let empty = DataRoot::new(dir.path().join("empty"));
        assert!(walk(&empty, CORPUS_VCTK).unwrap().is_empty());
    }

    #[test]
    fn common_voice_english_variants_are_walked() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let cv = root.raw().join("common_voice");
        let touch = |p: PathBuf| {
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, b"").unwrap();
        };
        let write = |p: PathBuf, s: &str| {
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, s).unwrap();
        };

        // en: standard release, validated.tsv preferred. One tsv row has no
        // clip on disk (skipped); one clip has no tsv row (skipped).
        let en = cv.join("en/cv-corpus-26.0-2026-06-12/en");
        touch(en.join("clips/common_voice_en_1001.mp3"));
        touch(en.join("clips/common_voice_en_1002.mp3"));
        touch(en.join("clips/common_voice_en_1003.mp3")); // not in the table
        write(
            en.join("validated.tsv"),
            "client_id\tpath\tsentence\n\
             EN1\tcommon_voice_en_1001.mp3\tA sentence, with a comma.\n\
             EN2\tcommon_voice_en_1002.mp3\tAnother.\n\
             EN9\tcommon_voice_en_9999.mp3\tNo clip on disk.\n",
        );
        // A train.tsv that must be ignored because validated.tsv exists.
        write(
            en.join("train.tsv"),
            "client_id\tpath\tsentence\nENX\tcommon_voice_en_1003.mp3\tx\n",
        );

        // en-AU: comma-separated CSV export, clips under audio_files/, a
        // leading unnamed index column. The `-split` sibling repeats rows
        // and must not double-count.
        let au = cv.join("en-AU/commonvoice-v24_en-AU");
        touch(au.join("audio_files/common_voice_en_2001.mp3"));
        write(
            au.join("commonvoice-v24_en-AU.csv"),
            ",client_id,path,sentence,gender\n\
             182,AU1,common_voice_en_2001.mp3,\"Two courts, four ovals.\",male\n",
        );
        write(
            au.join("commonvoice-v24_en-AU-split.csv"),
            ",client_id,path,sentence,gender\n\
             182,AU1,common_voice_en_2001.mp3,\"Two courts, four ovals.\",male\n",
        );

        // Southern: flat, union of train/dev/test.tsv.
        let so = cv.join("cv26-southern-american-english");
        touch(so.join("clips/common_voice_en_3001.mp3"));
        touch(so.join("clips/common_voice_en_3002.mp3"));
        write(
            so.join("train.tsv"),
            "client_id\tpath\tsentence\nS1\tcommon_voice_en_3001.mp3\tx\n",
        );
        write(
            so.join("dev.tsv"),
            "client_id\tpath\tsentence\nS2\tcommon_voice_en_3002.mp3\tx\n",
        );
        write(so.join("test.tsv"), "client_id\tpath\tsentence\n");

        // A non-English variant that must be ignored entirely.
        let de = cv.join("de/cv-corpus-26.0-2026-06-12/de");
        touch(de.join("clips/common_voice_de_5001.mp3"));
        write(
            de.join("validated.tsv"),
            "client_id\tpath\tsentence\nDE1\tcommon_voice_de_5001.mp3\tx\n",
        );

        let mut got = walk_common_voice(&root).unwrap();
        got.sort_by(|a, b| a.key.cmp(&b.key));
        let by_key: HashMap<&str, &Source> = got.iter().map(|s| (s.key.as_str(), s)).collect();

        let keys: Vec<&str> = got.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "common_voice/cv26-southern-american-english/common_voice_en_3001",
                "common_voice/cv26-southern-american-english/common_voice_en_3002",
                "common_voice/en-AU/common_voice_en_2001",
                "common_voice/en/common_voice_en_1001",
                "common_voice/en/common_voice_en_1002",
            ],
            "English clips with a table row only; no German, no phantom rows"
        );
        let en1 = by_key["common_voice/en/common_voice_en_1001"];
        assert_eq!(en1.speaker, "EN1");
        assert_eq!(en1.corpus, CORPUS_COMMON_VOICE);
        assert_eq!(en1.gender, None);
        assert_eq!(en1.subset, None);
        assert!(en1.licence.starts_with("CC0 (Mozilla Data Collective"));
        assert!(en1.src_rel.ends_with("clips/common_voice_en_1001.mp3"));
        assert_eq!(
            by_key["common_voice/en-AU/common_voice_en_2001"].speaker,
            "AU1"
        );
        assert!(
            by_key["common_voice/en-AU/common_voice_en_2001"]
                .src_rel
                .ends_with("audio_files/common_voice_en_2001.mp3")
        );
        assert_eq!(
            by_key["common_voice/cv26-southern-american-english/common_voice_en_3002"].speaker,
            "S2"
        );

        // Routes through `walk` and is opt-in (never a default corpus).
        assert_eq!(walk(&root, CORPUS_COMMON_VOICE).unwrap().len(), got.len());
        assert!(!DEFAULT_CORPORA.contains(&CORPUS_COMMON_VOICE));

        // A missing corpus directory is an empty list, not an error.
        let empty = DataRoot::new(dir.path().join("empty"));
        assert!(walk_common_voice(&empty).unwrap().is_empty());
    }

    #[test]
    fn prepare_writes_rows_rejects_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let root = fixture(dir.path());
        let opts = PrepareOptions {
            jobs: Some(2),
            ..PrepareOptions::default()
        };
        let s = run(&root, &opts).unwrap();
        assert_eq!(s.sources, 8);
        // Rejected: p257_002 (0.3 s), both VCTK FLACs (0.1 s).
        assert_eq!(s.rejected, 3, "{s:?}");
        assert_eq!(s.written, 5, "{s:?}");
        assert_eq!(s.skipped, 0);
        assert_eq!(s.by_corpus["libritts_r"], 2);

        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(rows.len(), 5);
        let by_key: HashMap<&str, &UtteranceRow> =
            rows.iter().map(|r| (r.key.as_str(), r)).collect();
        let lj = by_key["ljspeech/LJ001-0001"];
        assert_eq!(lj.split, Split::Train);
        assert_eq!(lj.src_rate, 22_050);
        assert_eq!(lj.licence, "Public-Domain");
        assert!(
            lj.duration_s >= 1.6 && lj.duration_s < 2.2,
            "{}",
            lj.duration_s
        );
        assert!((lj.trim_lead_s - 0.3).abs() < 0.05, "{}", lj.trim_lead_s);
        assert!((lj.trim_tail_s - 0.05).abs() < 0.05, "{}", lj.trim_tail_s);
        assert_eq!(by_key["voicebank_demand/test/p232_001"].split, Split::Test);
        assert_eq!(
            by_key["libritts_r/dev-clean/84_121123_000007_000001"].split,
            Split::Dev
        );
        assert_eq!(
            by_key["voicebank_demand/train/p226_001"].split,
            Split::Train
        );
        assert_eq!(
            by_key["libritts_r/dev-clean/84_121123_000007_000001"]
                .gender
                .as_deref(),
            Some("F")
        );

        let a16 = unamblify_audio::read(root.prepared_16k_decoded(&lj.key)).unwrap();
        let a8 = unamblify_audio::read(root.prepared_8k_decoded(&lj.key)).unwrap();
        assert_eq!(a16.rate, 16_000);
        assert_eq!(a8.rate, 8_000);
        assert_eq!(a16.samples.len(), 2 * a8.samples.len());
        assert!((a16.duration_s() - lj.duration_s).abs() < 1e-9);
        let level = active_rms_dbfs(&a16.samples, 16_000).unwrap();
        assert!((level - TARGET_DBFS).abs() < 0.5, "{level}");
        assert_eq!(
            sha256_file(&root.prepared_16k_decoded(&lj.key)).unwrap(),
            lj.sha256_16k
        );
        assert!(files_consistent(&root, lj));

        let rej: Vec<RejectRow> = read_jsonl(&root.prepared_rejected()).unwrap();
        assert_eq!(rej.len(), 3);
        assert!(
            rej.iter().all(|r| r.reason.starts_with("too short")),
            "{rej:?}"
        );

        // Idempotent: nothing to do.
        let s2 = run(&root, &opts).unwrap();
        assert_eq!(s2.written, 0);
        assert_eq!(s2.rejected, 0);
        assert_eq!(s2.skipped, 8);
        assert_eq!(
            read_jsonl::<UtteranceRow>(&root.prepared_manifest())
                .unwrap()
                .len(),
            5
        );

        // A missing decoded file is redone; --force redoes all without duplicates.
        fs::remove_file(root.prepared_8k_flac(&lj.key)).unwrap();
        let s3 = run(&root, &opts).unwrap();
        assert_eq!(s3.written, 1);
        assert_eq!(
            read_jsonl::<UtteranceRow>(&root.prepared_manifest())
                .unwrap()
                .len(),
            5,
            "a redo replaces its row"
        );
        let s4 = run(
            &root,
            &PrepareOptions {
                force: true,
                corpora: vec![CORPUS_LJSPEECH.to_owned()],
                jobs: Some(1),
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        assert_eq!(s4.written, 1);
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(rows.iter().filter(|r| r.key == lj.key).count(), 1);
        assert_eq!(rows.len(), 5);

        let st = stats(&root).unwrap();
        assert_eq!(st.rows, 5);
        assert_eq!(st.rejected, 3);
        assert_eq!(st.speakers, 5);
        assert_eq!(st.by_split[&Split::Test], 1);
    }

    /// `--twins-only` emits twins from the manifest as it stands, without
    /// walking raw/: the corpus can be gone and the twins still come out.
    #[test]
    fn twins_only_skips_the_corpus_walk() {
        let dir = tempfile::tempdir().unwrap();
        let root = fixture(dir.path());
        let base = PrepareOptions {
            jobs: Some(2),
            ..PrepareOptions::default()
        };
        assert_eq!(run(&root, &base).unwrap().written, 5);
        // Take the raw corpus away: a walk would now find nothing to do,
        // and a main pass would report zero sources.
        fs::remove_dir_all(root.raw()).unwrap();

        let opts = PrepareOptions {
            jobs: Some(2),
            twins_only: true,
            twins: TwinOptions {
                under_share: 1.0,
                ..TwinOptions::default()
            },
            ..PrepareOptions::default()
        };
        let s = run(&root, &opts).unwrap();
        assert_eq!(s.sources, 0, "no corpus was walked");
        assert_eq!(s.twins_written, 5, "{s:?}");
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(rows.iter().filter(|r| r.parent.is_some()).count(), 5);

        // Without a share there is nothing to do, and that is an error
        // rather than a silent no-op.
        let none = PrepareOptions {
            twins_only: true,
            ..PrepareOptions::default()
        };
        assert!(run(&root, &none).is_err());
    }

    #[test]
    fn a_processing_error_is_retried_next_run_not_rejected_forever() {
        let dir = tempfile::tempdir().unwrap();
        let root = fixture(dir.path());
        let opts = PrepareOptions {
            jobs: Some(2),
            ..PrepareOptions::default()
        };
        // A file where `prepared/ljspeech/` must go: every LJ write fails.
        fs::create_dir_all(root.prepared()).unwrap();
        let blocker = root.prepared().join("ljspeech");
        fs::write(&blocker, b"in the way").unwrap();
        let s = run(&root, &opts).unwrap();
        assert_eq!(s.errors, 1, "{s:?}");
        assert_eq!(s.written, 4, "{s:?}");
        assert_eq!(s.rejected, 3, "{s:?}");
        let rej: Vec<RejectRow> = read_jsonl(&root.prepared_rejected()).unwrap();
        assert!(
            rej.iter().all(|r| !r.key.starts_with("ljspeech/")),
            "an I/O failure is not a rejection: {rej:?}"
        );
        // Drive back: the utterance is retried without --force.
        fs::remove_file(&blocker).unwrap();
        let s2 = run(&root, &opts).unwrap();
        assert_eq!(s2.written, 1, "{s2:?}");
        assert_eq!(s2.errors, 0);
        assert_eq!(s2.skipped, 7);
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert!(rows.iter().any(|r| r.key == "ljspeech/LJ001-0001"));
        assert_eq!(rows.len(), 5);
        // A legacy `error:` rejection from an earlier build is redone too.
        let mut rej: Vec<RejectRow> = read_jsonl(&root.prepared_rejected()).unwrap();
        rej.push(RejectRow {
            key: "ljspeech/LJ001-0001".to_owned(),
            reason: "error: disk full".to_owned(),
            src_path: String::new(),
            rejected_at: now_rfc3339(),
        });
        write_jsonl(&root.prepared_rejected(), &rej).unwrap();
        fs::remove_file(root.prepared_8k_flac("ljspeech/LJ001-0001")).unwrap();
        let s3 = run(&root, &opts).unwrap();
        assert_eq!(s3.written, 1, "{s3:?}");
        let rej: Vec<RejectRow> = read_jsonl(&root.prepared_rejected()).unwrap();
        assert_eq!(rej.len(), 3);
        assert!(rej.iter().all(|r| !r.reason.starts_with("error:")));
    }
}
