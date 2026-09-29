// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Prepare-time augmentation (`docs/design/data-pipeline.md`, stage 3):
//! for a deterministic share of prepared utterances, one extra **twin**
//! row whose input is the parent's clean 16 kHz signal with a noise clip
//! mixed in (`<key>+n<NNNN>`), pushed through the synthetic overdriven-mic
//! chain (`<key>+h<NNNN>`), or both (`<key>+n<NNNN>+h<NNNN>`). The twin's
//! `.16k.flac` is the *noisy* input and its `.8k.flac` the decimated chip
//! input; its manifest row names its `parent`, whose clean files are the
//! training target, and records every parameter drawn in `aug`.
//!
//! Every decision and draw comes from `key_seed(parent, seed)`, so a
//! re-run emits the same twins with the same bytes, a twin is emitted
//! only for a parent row that exists, and twins take their parent's split.
//! Noise clips come from the tier-3 sets under `raw/`: DEMAND
//! (`raw/demand/<ENV>/chNN.wav`, one channel file drawn at random, the
//! car / bus / traffic / square environments weighted 3×) and MUSAN's
//! `noise` and `music` trees (never `speech`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};

use rayon::prelude::*;
use unamblify::aug::{
    AgcAug, AugRecord, ChainParams, ClipAug, ClipKind, MicAug, NoiseAug, NoiseSet, PopsAug,
    UnderAug, key_seed,
};
use unamblify::{Rng, UtteranceRow, VOCODER_SAMPLE_RATE, WIDEBAND_SAMPLE_RATE, key};
use unamblify_audio::chain::{Biquad, PROXIMITY_HZ, apply_chain, mix_at_snr};
use unamblify_audio::level::{MAX_GAIN_DB, TARGET_DBFS};
use unamblify_audio::{
    active_rms_dbfs, normalize_rms, read_wav_range, resample, wav_info, write_flac_s16,
};

use crate::prepare::{Outcome, files_consistent, spawn_writer};
use crate::util::{now_rfc3339, read_jsonl, sha256_file, write_jsonl};
use crate::{DataError, DataRoot, Result};

/// The SNR range of a noisy twin, dB, drawn uniformly.
pub const SNR_RANGE_DB: (f32, f32) = (0.0, 20.0);
/// DEMAND environments weighted toward mobile operation.
pub const DEMAND_FAVOURED: [&str; 4] = ["TCAR", "TBUS", "STRAFFIC", "SPSQUARE"];
/// Weight of a favoured DEMAND environment against the rest.
pub const DEMAND_FAVOUR_WEIGHT: u32 = 3;
/// Peak the twin input is held under (its s16 write must not clip).
pub const PEAK_CEILING: f32 = 0.99;

/// The twin knobs of a prepare run.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinOptions {
    /// Share of utterances that get a noisy twin (0 = none).
    pub noise_share: f32,
    /// Which noise sets to draw from (all of them when empty).
    pub noise_sets: Vec<NoiseSet>,
    /// Share of utterances that get an overdriven-mic twin (0 = none).
    pub chain_share: f32,
    /// Share of utterances that get an underdriven twin (0 = none). An
    /// utterance already given an overdriven twin is skipped: a mic
    /// cannot be both too hot and too quiet.
    pub under_share: f32,
    /// The run seed every per-key stream derives from.
    pub seed: u64,
}

impl Default for TwinOptions {
    fn default() -> Self {
        Self {
            noise_share: 0.0,
            noise_sets: Vec::new(),
            chain_share: 0.0,
            under_share: 0.0,
            seed: 1,
        }
    }
}

impl TwinOptions {
    /// Whether any twin is asked for.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.noise_share > 0.0 || self.chain_share > 0.0 || self.under_share > 0.0
    }
}

/// One noise clip on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseClip {
    /// Which set.
    pub set: NoiseSet,
    /// Path relative to the data root.
    pub rel: String,
    /// Draw weight within its set.
    pub weight: u32,
    /// Samples per channel.
    pub samples: usize,
    /// Sample rate, Hz.
    pub rate: u32,
}

/// The clips of every requested set, each set's clips together with
/// their cumulative weights so a draw is one binary search.
#[derive(Debug, Clone, Default)]
pub struct NoiseLibrary {
    sets: Vec<(NoiseSet, Vec<NoiseClip>, Vec<u32>)>,
}

impl NoiseLibrary {
    /// Walk `raw/<set>/` for every requested set (all sets when `sets` is
    /// empty). A requested set whose directory is missing or holds no
    /// usable clip is an error: a twin run must not silently draw from
    /// half the material.
    pub fn scan(root: &DataRoot, sets: &[NoiseSet]) -> Result<Self> {
        let sets: Vec<NoiseSet> = if sets.is_empty() {
            NoiseSet::ALL.to_vec()
        } else {
            sets.to_vec()
        };
        let mut out = Self::default();
        for set in sets {
            let dir = root.raw().join(set.as_str());
            if !dir.is_dir() {
                return Err(DataError::Invalid(format!(
                    "noise set {set}: {} is not a directory (fetch tier 3 first, or drop it from --noise-sets)",
                    dir.display()
                )));
            }
            let mut clips = Vec::new();
            walk_wavs(&dir, &mut |p| {
                let rel = p
                    .strip_prefix(root.path())
                    .unwrap_or(p)
                    .to_string_lossy()
                    .into_owned();
                let comps: Vec<&str> = rel.split('/').collect();
                // `raw/<set>/<first>/...`
                let first = comps.get(2).copied().unwrap_or("");
                let weight = match set {
                    NoiseSet::Demand => {
                        if DEMAND_FAVOURED.contains(&first) {
                            DEMAND_FAVOUR_WEIGHT
                        } else {
                            1
                        }
                    }
                    NoiseSet::Musan => {
                        let kinds: Vec<&str> = comps[2..comps.len().saturating_sub(1)].to_vec();
                        u32::from(kinds.contains(&"noise") || kinds.contains(&"music"))
                    }
                };
                if weight == 0 {
                    return Ok(());
                }
                let info = match wav_info(p) {
                    Ok(i) => i,
                    Err(e) => {
                        log::warn!("{rel}: unreadable noise clip, skipped: {e}");
                        return Ok(());
                    }
                };
                if info.samples == 0 {
                    return Ok(());
                }
                clips.push(NoiseClip {
                    set,
                    rel,
                    weight,
                    samples: info.samples,
                    rate: info.rate,
                });
                Ok(())
            })?;
            clips.sort_by(|a, b| a.rel.cmp(&b.rel));
            if clips.is_empty() {
                return Err(DataError::Invalid(format!(
                    "noise set {set}: no usable .wav under {}",
                    dir.display()
                )));
            }
            let mut acc = 0u32;
            let cumulative = clips
                .iter()
                .map(|c| {
                    acc += c.weight;
                    acc
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let hours = clips
                .iter()
                .map(|c| c.samples as f64 / f64::from(c.rate))
                .sum::<f64>()
                / 3_600.0;
            log::info!("noise set {set}: {} clips, {hours:.1} h", clips.len());
            out.sets.push((set, clips, cumulative));
        }
        Ok(out)
    }

    /// Clips in every set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sets.iter().map(|(_, c, _)| c.len()).sum()
    }

    /// No clips at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Draw one clip: the set uniformly, then a clip by weight.
    #[must_use]
    pub fn draw(&self, rng: &mut Rng) -> Option<&NoiseClip> {
        let (_, clips, cumulative) = self.sets.get(rng.below(self.sets.len()))?;
        let total = *cumulative.last()?;
        let r = u32::try_from(rng.below(total as usize)).unwrap_or(0);
        let i = cumulative.partition_point(|&c| c <= r);
        clips.get(i.min(clips.len() - 1))
    }
}

fn walk_wavs(dir: &Path, f: &mut dyn FnMut(&Path) -> Result<()>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| DataError::io(dir, e))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            walk_wavs(&p, f)?;
        } else if p
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        {
            f(&p)?;
        }
    }
    Ok(())
}

/// What the per-key draw decided for one parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwinPlan {
    /// The twin's key.
    pub key: String,
    /// Mix noise.
    pub noise: bool,
    /// Apply the chain.
    pub chain: bool,
    /// Underdrive: attenuate, and do not bring the level back.
    pub under: bool,
    /// The per-key seed the parameter draws continue from.
    pub seed: u64,
}

/// Attenuation range of an underdriven twin, dB below the prepare
/// target. The corpus sits at −26 dBFS, so this feeds the vocoder at
/// roughly −41 to −56 dBFS: quiet enough that its gain quantiser goes
/// coarse and pitch tracking starts to fail, not so quiet that there is
/// nothing left to restore.
pub const UNDER_GAIN_DB: (f32, f32) = (-30.0, -15.0);

/// Draw an underdriven transmit.
pub fn draw_under(rng: &mut Rng) -> UnderAug {
    let uniform = |rng: &mut Rng, lo: f32, hi: f32| lo + rng.next_f32() * (hi - lo);
    UnderAug {
        gain_db: uniform(rng, UNDER_GAIN_DB.0, UNDER_GAIN_DB.1),
        shelf_db: rng.chance(0.5).then(|| uniform(rng, -8.0, -3.0)),
    }
}

/// The share decision for `parent`: whether it gets a twin and which
/// kind. Pure: the same key, seed and shares always give the same plan.
#[must_use]
pub fn plan_for(parent: &str, opts: &TwinOptions) -> Option<TwinPlan> {
    let seed = key_seed(parent, opts.seed);
    let mut rng = Rng::new(seed);
    let noise = rng.chance(opts.noise_share);
    let chain = rng.chance(opts.chain_share);
    // The underdrive decision and its tag come from a fork, so turning
    // the share on moves no noise or chain decision and renames no
    // existing twin.
    let mut urng = Rng::new(seed).fork(0x0D1);
    let under = !chain && urng.chance(opts.under_share);
    if !noise && !chain && !under {
        return None;
    }
    let mut k = parent.to_owned();
    if noise {
        k = key::twin(&k, &format!("n{:04}", rng.below(10_000)));
    }
    if chain {
        k = key::twin(&k, &format!("h{:04}", rng.below(10_000)));
    }
    if under {
        k = key::twin(&k, &format!("u{:04}", urng.below(10_000)));
    }
    Some(TwinPlan {
        key: k,
        noise,
        chain,
        under,
        seed,
    })
}

/// Draw the overdriven-mic chain: each stage with probability one half,
/// at least one stage always.
#[must_use]
pub fn draw_chain(rng: &mut Rng) -> ChainParams {
    let uniform = |rng: &mut Rng, lo: f32, hi: f32| lo + rng.next_f32() * (hi - lo);
    loop {
        let p = ChainParams {
            shelf_db: rng.chance(0.5).then(|| uniform(rng, 4.0, 10.0)),
            mic: rng.chance(0.5).then(|| MicAug {
                tilt_db: uniform(rng, -6.0, 6.0),
                resonance_hz: 400.0 * (3_500.0f32 / 400.0).powf(rng.next_f32()),
                resonance_db: uniform(rng, 3.0, 9.0),
                resonance_q: uniform(rng, 2.0, 5.0),
            }),
            pops: rng.chance(0.5).then(|| PopsAug {
                share: uniform(rng, 0.2, 0.5),
                freq_hz: uniform(rng, 60.0, 120.0),
                level_dbfs: uniform(rng, -10.0, -3.0),
            }),
            clip: rng.chance(0.5).then(|| ClipAug {
                kind: if rng.chance(0.5) {
                    ClipKind::Hard
                } else {
                    ClipKind::Soft
                },
                drive_db: uniform(rng, 6.0, 24.0),
            }),
            agc: rng.chance(0.5).then(|| AgcAug {
                attack_ms: uniform(rng, 5.0, 50.0),
                release_ms: uniform(rng, 100.0, 500.0),
            }),
        };
        if !p.is_empty() {
            return p;
        }
    }
}

/// Read `n16` samples of a noise clip at 16 kHz from a random offset,
/// wrapping when the clip is shorter. Returns the samples and the offset
/// in seconds.
fn noise_segment(
    root: &DataRoot,
    clip: &NoiseClip,
    n16: usize,
    rng: &mut Rng,
) -> Result<(Vec<f32>, f64)> {
    let path = root.path().join(&clip.rel);
    let offset = rng.below(clip.samples);
    let need = usize::try_from(
        (n16 as u64 * u64::from(clip.rate)).div_ceil(u64::from(WIDEBAND_SAMPLE_RATE)),
    )
    .unwrap_or(usize::MAX)
    .saturating_add(2);
    let mut src = Vec::with_capacity(need);
    let mut at = offset;
    while src.len() < need {
        let take = need - src.len();
        let a = read_wav_range(&path, at, take)?;
        if a.samples.is_empty() {
            at = 0;
            if clip.samples == 0 {
                break;
            }
            continue;
        }
        at += a.samples.len();
        src.extend(a.samples);
        if at >= clip.samples {
            at = 0;
        }
    }
    let mut y = resample(&src, clip.rate, WIDEBAND_SAMPLE_RATE)?;
    y.resize(n16, 0.0);
    #[allow(clippy::cast_precision_loss)]
    let offset_s = offset as f64 / f64::from(clip.rate);
    Ok((y, offset_s))
}

/// Make one twin: read the parent's clean 16 kHz file, mix and / or
/// chain, level-guard, write both rates, hash. `Err` is an I/O or
/// resampler failure (retried next run).
pub fn make_twin(
    root: &DataRoot,
    parent: &UtteranceRow,
    plan: &TwinPlan,
    library: &NoiseLibrary,
) -> Result<UtteranceRow> {
    let clean = unamblify_audio::read(root.prepared_16k_decoded(&parent.key))?;
    if clean.rate != WIDEBAND_SAMPLE_RATE {
        return Err(DataError::Invalid(format!(
            "{}: {} Hz, expected {}",
            parent.key, clean.rate, WIDEBAND_SAMPLE_RATE
        )));
    }
    // The plan's rng drew the decision and tags; the draws continue on a
    // forked stream so a change to the tag format does not move them.
    let mut rng = Rng::new(plan.seed).fork(0xA06);
    let mut x = clean.samples;
    let mut noise = None;
    if plan.noise {
        let clip = library.draw(&mut rng).ok_or_else(|| {
            DataError::Invalid("no noise clips available for a noisy twin".to_owned())
        })?;
        let (seg, offset_s) = noise_segment(root, clip, x.len(), &mut rng)?;
        let snr_db = SNR_RANGE_DB.0 + rng.next_f32() * (SNR_RANGE_DB.1 - SNR_RANGE_DB.0);
        let (mix, _) = mix_at_snr(&x, &seg, WIDEBAND_SAMPLE_RATE, snr_db);
        x = mix;
        noise = Some(NoiseAug {
            noise_set: clip.set,
            noise_clip: clip.rel.clone(),
            noise_offset_s: offset_s,
            snr_db,
        });
    }
    let mut chain = None;
    let mut post_gain_db = 0.0f32;
    if plan.chain {
        let p = draw_chain(&mut rng);
        x = apply_chain(&x, WIDEBAND_SAMPLE_RATE, &p, &mut rng);
        // The distortion, not the loudness, is the augmentation.
        let (y, g) = normalize_rms(&x, WIDEBAND_SAMPLE_RATE, TARGET_DBFS, MAX_GAIN_DB);
        x = y;
        post_gain_db += g;
        chain = Some(p);
    }
    let mut under = None;
    if plan.under {
        let u = draw_under(&mut rng);
        if let Some(db) = u.shelf_db {
            Biquad::low_shelf(WIDEBAND_SAMPLE_RATE, PROXIMITY_HZ, db).run(&mut x);
        }
        let g = 10f32.powf(u.gain_db / 20.0);
        for v in &mut x {
            *v *= g;
        }
        // Deliberately NOT renormalised, unlike the chain above: here the
        // low level is the augmentation. The damage an underdriven mic
        // does happens inside the vocoder, and bringing the level back
        // first would hand the codec a healthy signal and train on a
        // null. The attenuation is recorded in `aug.under.gain_db`.
        under = Some(u);
    }
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > PEAK_CEILING {
        let g = PEAK_CEILING / peak;
        for v in &mut x {
            *v *= g;
        }
        post_gain_db += 20.0 * g.log10();
    }
    // An underdriven twin can sit wholly below the active-speech threshold,
    // where the gated measure finds nothing; its plain RMS is still a level.
    let rms_in = active_rms_dbfs(&x, WIDEBAND_SAMPLE_RATE)
        .unwrap_or_else(|| unamblify_audio::level::rms_dbfs(&x));
    let x8 = resample(&x, WIDEBAND_SAMPLE_RATE, VOCODER_SAMPLE_RATE)?;
    let p16 = root.prepared_16k_flac(&plan.key);
    let p8 = root.prepared_8k_flac(&plan.key);
    if let Some(dir) = p16.parent() {
        std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    }
    write_flac_s16(&p16, &x, WIDEBAND_SAMPLE_RATE)?;
    write_flac_s16(&p8, &x8, VOCODER_SAMPLE_RATE)?;
    Ok(UtteranceRow {
        key: plan.key.clone(),
        corpus: parent.corpus.clone(),
        speaker: parent.speaker.clone(),
        gender: parent.gender.clone(),
        split: parent.split,
        duration_s: parent.duration_s,
        src_rate: parent.src_rate,
        src_path: parent.src_path.clone(),
        licence: parent.licence.clone(),
        rms_dbfs_in: f64::from(rms_in),
        gain_db: f64::from(post_gain_db),
        trim_lead_s: parent.trim_lead_s,
        trim_tail_s: parent.trim_tail_s,
        sha256_16k: sha256_file(&p16)?,
        sha256_8k: sha256_file(&p8)?,
        prepared_at: now_rfc3339(),
        parent: Some(parent.key.clone()),
        aug: Some(AugRecord {
            seed: plan.seed,
            noise,
            chain,
            under,
            post_gain_db,
        }),
    })
}

/// Counts of a twin pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TwinSummary {
    /// Twins the shares asked for.
    pub planned: u64,
    /// Written this run.
    pub written: u64,
    /// Already on disk and left alone.
    pub skipped: u64,
    /// Failed this run (retried next run).
    pub errors: u64,
    /// Twin rows dropped because their parent row is gone.
    pub orphaned: u64,
}

/// The work list of a twin pass: what to render and which manifest rows
/// to drop first.
struct TwinWork<'a> {
    todo: Vec<(&'a UtteranceRow, TwinPlan)>,
    drop: HashSet<&'a str>,
    summary: TwinSummary,
}

/// Decide a twin for every parent row, skip the ones already on disk
/// (unless `force`, or their parent was redone this run), and mark for
/// dropping the rows to be redone and the twins whose parent has gone.
fn plan_work<'a>(
    root: &DataRoot,
    opts: &TwinOptions,
    rows: &'a [UtteranceRow],
    redone_parents: &HashSet<String>,
    force: bool,
) -> TwinWork<'a> {
    let parents: HashMap<&str, &UtteranceRow> = rows
        .iter()
        .filter(|r| r.parent.is_none())
        .map(|r| (r.key.as_str(), r))
        .collect();
    let twins: HashMap<&str, &UtteranceRow> = rows
        .iter()
        .filter(|r| r.parent.is_some())
        .map(|r| (r.key.as_str(), r))
        .collect();
    let mut work = TwinWork {
        todo: Vec::new(),
        drop: HashSet::new(),
        summary: TwinSummary::default(),
    };
    for (k, t) in &twins {
        if t.parent.as_deref().is_none_or(|p| !parents.contains_key(p)) {
            work.drop.insert(k);
            work.summary.orphaned += 1;
        }
    }
    let mut parent_keys: Vec<&&str> = parents.keys().collect();
    parent_keys.sort();
    for pk in parent_keys {
        let Some(plan) = plan_for(pk, opts) else {
            continue;
        };
        work.summary.planned += 1;
        let existing = twins.get(plan.key.as_str()).copied();
        let done = existing.is_some_and(|t| files_consistent(root, t));
        if done && !force && !redone_parents.contains(*pk) {
            work.summary.skipped += 1;
            continue;
        }
        if let Some(t) = existing {
            work.drop.insert(t.key.as_str());
        }
        work.todo.push((parents[*pk], plan));
    }
    work
}

/// The twin pass: read the manifest as it stands after the main pass,
/// plan ([`plan_work`]), rewrite the manifest without the rows to be
/// redone or orphaned, and render the rest in parallel through the
/// prepare writer.
#[allow(clippy::implicit_hasher)]
pub fn run(
    root: &DataRoot,
    opts: &TwinOptions,
    redone_parents: &HashSet<String>,
    force: bool,
    jobs: Option<usize>,
) -> Result<TwinSummary> {
    let library = if opts.noise_share > 0.0 {
        NoiseLibrary::scan(root, &opts.noise_sets)?
    } else {
        NoiseLibrary::default()
    };
    let manifest_path = root.prepared_manifest();
    let rows: Vec<UtteranceRow> = read_jsonl(&manifest_path)?;
    let TwinWork {
        todo,
        drop,
        mut summary,
    } = plan_work(root, opts, &rows, redone_parents, force);
    if !drop.is_empty() {
        log::info!(
            "twins: dropping {} row(s) to be redone or orphaned",
            drop.len()
        );
        write_jsonl(
            &manifest_path,
            rows.iter().filter(|r| !drop.contains(r.key.as_str())),
        )?;
    }
    log::info!(
        "twins: {} planned, {} to do, {} skipped, {} orphaned",
        summary.planned,
        todo.len(),
        summary.skipped,
        summary.orphaned
    );
    if todo.is_empty() {
        return Ok(summary);
    }
    let (tx, rx) = channel::<Outcome>();
    let writer = spawn_writer(rx, manifest_path, root.prepared_rejected());
    let work = |(parent, plan): &(&UtteranceRow, TwinPlan), tx: &Sender<Outcome>| {
        let outcome = match make_twin(root, parent, plan, &library) {
            Ok(row) => Outcome::Row(Box::new(row)),
            Err(e) => {
                log::error!("{}: {e} (will be retried next run)", plan.key);
                Outcome::Failed {
                    key: plan.key.clone(),
                    error: e.to_string(),
                }
            }
        };
        let _ = tx.send(outcome);
    };
    match jobs {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| DataError::Invalid(format!("rayon: {e}")))?;
            pool.install(|| todo.par_iter().for_each_with(tx, |tx, t| work(t, tx)));
        }
        None => todo.par_iter().for_each_with(tx, |tx, t| work(t, tx)),
    }
    let (written, _, errors, _) = writer
        .join()
        .unwrap_or_else(|_| Err(DataError::Invalid("manifest writer panicked".to_owned())))?;
    summary.written = written;
    summary.errors = errors;
    log::info!(
        "twins: {} written, {} failed (retried next run)",
        summary.written,
        summary.errors
    );
    Ok(summary)
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::testutil::sine;
    use unamblify::Split;
    use unamblify_audio::chain::measured_snr_db;
    use unamblify_audio::write_wav_s16;

    /// A prepared root with `n` parents (1.5 s tones, prepare-normalised)
    /// and a tiny DEMAND tree: two environments, one favoured, two
    /// channels each, plus a hidden zip marker that must be ignored.
    fn fixture(n: usize) -> (tempfile::TempDir, DataRoot) {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let mut w = crate::util::JsonlWriter::open(&root.prepared_manifest()).unwrap();
        for i in 0..n {
            let key = format!("vctk/p2{i:02}_001_mic2");
            let mut x = vec![0.0f32; 4_000];
            x.extend(sine(180.0 + 30.0 * i as f32, 16_000, 20_000, 0.3));
            let (x, _) = normalize_rms(&x, 16_000, TARGET_DBFS, MAX_GAIN_DB);
            let x8 = resample(&x, 16_000, 8_000).unwrap();
            let p16 = root.prepared_16k(&key);
            std::fs::create_dir_all(p16.parent().unwrap()).unwrap();
            write_wav_s16(&p16, &x, 16_000).unwrap();
            write_wav_s16(root.prepared_8k(&key), &x8, 8_000).unwrap();
            w.append(&UtteranceRow {
                key: key.clone(),
                corpus: "vctk".to_owned(),
                speaker: format!("p2{i:02}"),
                gender: None,
                split: if i % 2 == 0 { Split::Dev } else { Split::Train },
                duration_s: x.len() as f64 / 16_000.0,
                src_rate: 48_000,
                src_path: "raw/x".to_owned(),
                licence: "CC-BY-4.0".to_owned(),
                rms_dbfs_in: -20.0,
                gain_db: -6.0,
                trim_lead_s: 0.0,
                trim_tail_s: 0.0,
                sha256_16k: sha256_file(&p16).unwrap(),
                sha256_8k: sha256_file(&root.prepared_8k(&key)).unwrap(),
                prepared_at: now_rfc3339(),
                parent: None,
                aug: None,
            })
            .unwrap();
        }
        drop(w);
        let demand = root.raw().join("demand");
        for (env, f) in [("TCAR", 90.0f32), ("DKITCHEN", 1_300.0)] {
            for ch in 1..=2 {
                let p = demand.join(env).join(format!("ch{ch:02}.wav"));
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                // 3 s of a tone plus noise-like content at 16 kHz.
                let mut x = sine(f * ch as f32, 16_000, 48_000, 0.1);
                let mut s = 7u64 + ch as u64;
                for v in &mut x {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *v += ((s >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 0.1;
                }
                write_wav_s16(&p, &x, 16_000).unwrap();
            }
        }
        std::fs::write(demand.join(".extracted-TCAR_16k.zip"), b"").unwrap();
        (dir, root)
    }

    #[test]
    fn library_scans_weights_and_draws_deterministically() {
        let (_dir, root) = fixture(1);
        let lib = NoiseLibrary::scan(&root, &[NoiseSet::Demand]).unwrap();
        assert_eq!(lib.len(), 4);
        assert!(!lib.is_empty());
        let (_, clips, cum) = &lib.sets[0];
        assert_eq!(clips[0].rel, "raw/demand/DKITCHEN/ch01.wav");
        assert_eq!(clips[0].weight, 1);
        assert_eq!(clips[2].rel, "raw/demand/TCAR/ch01.wav");
        assert_eq!(clips[2].weight, DEMAND_FAVOUR_WEIGHT);
        assert_eq!(cum, &vec![1, 2, 5, 8]);
        assert_eq!(clips[0].samples, 48_000);
        // Favoured environments come up about three times as often.
        let mut rng = Rng::new(9);
        let mut tcar = 0;
        for _ in 0..4_000 {
            if lib.draw(&mut rng).unwrap().rel.contains("TCAR") {
                tcar += 1;
            }
        }
        assert!((2_700..3_300).contains(&tcar), "{tcar}");
        let a = lib.draw(&mut Rng::new(4)).unwrap().rel.clone();
        let b = lib.draw(&mut Rng::new(4)).unwrap().rel.clone();
        assert_eq!(a, b);
        // MUSAN is requested but absent: loud failure, not a half library.
        let err = NoiseLibrary::scan(&root, &[]).unwrap_err().to_string();
        assert!(err.contains("musan"), "{err}");
        // A MUSAN tree keeps noise and music, never speech.
        for (k, name) in [
            ("noise", "free-sound/n1.wav"),
            ("music", "fma/m1.wav"),
            ("speech", "librivox/s1.wav"),
        ] {
            let p = root.raw().join("musan").join(k).join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            write_wav_s16(&p, &sine(200.0, 16_000, 1_600, 0.1), 16_000).unwrap();
        }
        let lib = NoiseLibrary::scan(&root, &[]).unwrap();
        assert_eq!(lib.len(), 6);
        assert!(lib.sets[1].1.iter().all(|c| !c.rel.contains("speech")));
    }

    /// Turning the underdrive share on must not move a single noise or
    /// chain decision, nor rename a twin that already exists.
    #[test]
    fn the_underdrive_share_leaves_every_other_plan_alone() {
        let before = TwinOptions {
            noise_share: 0.3,
            chain_share: 0.3,
            ..TwinOptions::default()
        };
        let after = TwinOptions {
            under_share: 0.5,
            ..before.clone()
        };
        let mut gained = 0;
        for i in 0..2_000 {
            let k = format!("c/u{i}");
            let (a, b) = (plan_for(&k, &before), plan_for(&k, &after));
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!((a.noise, a.chain), (b.noise, b.chain), "{k}");
                    assert!(b.key.starts_with(&a.key), "{} renamed to {}", a.key, b.key);
                    assert!(!(b.chain && b.under), "{k}: too hot and too quiet at once");
                }
                (None, Some(b)) => {
                    assert!(b.under && !b.noise && !b.chain, "{k}");
                    assert!(b.key.contains("+u"), "{}", b.key);
                    gained += 1;
                }
                (None, None) => {}
                (Some(a), None) => panic!("{}: the share removed a twin", a.key),
            }
        }
        assert!(gained > 300, "{gained}");
        for s in 0..64 {
            let u = draw_under(&mut Rng::new(s));
            assert!(
                (UNDER_GAIN_DB.0..=UNDER_GAIN_DB.1).contains(&u.gain_db),
                "{u:?}"
            );
            assert!(
                u.shelf_db.is_none_or(|d| (-8.0..=-3.0).contains(&d)),
                "{u:?}"
            );
        }
    }

    #[test]
    fn plans_are_seeded_by_key_and_shares_hold() {
        let all = TwinOptions {
            noise_share: 1.0,
            chain_share: 1.0,
            ..TwinOptions::default()
        };
        let p = plan_for("vctk/p225_001_mic2", &all).unwrap();
        assert!(p.noise && p.chain);
        assert!(p.key.starts_with("vctk/p225_001_mic2+n"), "{}", p.key);
        assert!(p.key.contains("+h"), "{}", p.key);
        assert_eq!(key::parent_of(&p.key), Some("vctk/p225_001_mic2"));
        assert_eq!(plan_for("vctk/p225_001_mic2", &all), Some(p.clone()));
        let other = TwinOptions {
            seed: 2,
            ..all.clone()
        };
        assert_ne!(plan_for("vctk/p225_001_mic2", &other).unwrap().key, p.key);
        assert_eq!(plan_for("x/y", &TwinOptions::default()), None);
        let quarter = TwinOptions {
            noise_share: 0.25,
            chain_share: 0.0,
            ..TwinOptions::default()
        };
        let n = (0..4_000)
            .filter(|i| plan_for(&format!("c/u{i}"), &quarter).is_some())
            .count();
        assert!((850..1_150).contains(&n), "{n}");
        let only_chain = TwinOptions {
            noise_share: 0.0,
            chain_share: 1.0,
            ..TwinOptions::default()
        };
        let p = plan_for("c/u", &only_chain).unwrap();
        assert!(!p.noise && p.chain && !p.key.contains("+n"));
        // The chain draw always has at least one stage.
        for s in 0..64 {
            assert!(!draw_chain(&mut Rng::new(s)).is_empty());
        }
    }

    /// The whole point of a `+u` twin: it reaches the codec quiet. If the
    /// level were brought back, as it is after the overdriven chain, the
    /// vocoder would be handed a healthy signal and the twin would train
    /// on nothing.
    #[test]
    fn an_underdriven_twin_stays_quiet_and_says_by_how_much() {
        let (_dir, root) = fixture(4);
        let opts = TwinOptions {
            under_share: 1.0,
            ..TwinOptions::default()
        };
        let s = run(&root, &opts, &HashSet::new(), false, Some(2)).unwrap();
        assert_eq!((s.written, s.errors), (4, 0), "{s:?}");
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        let parents: HashMap<&str, &UtteranceRow> = rows
            .iter()
            .filter(|r| r.parent.is_none())
            .map(|r| (r.key.as_str(), r))
            .collect();
        let twins: Vec<&UtteranceRow> = rows.iter().filter(|r| r.parent.is_some()).collect();
        assert_eq!(twins.len(), 4);
        for t in twins {
            assert!(t.key.contains("+u"), "{}", t.key);
            let aug = t.aug.as_ref().unwrap();
            assert!(aug.chain.is_none() && aug.noise.is_none());
            let u = aug.under.expect("the attenuation is recorded");
            assert_eq!(aug.post_gain_db, 0.0, "no gain is given back");
            let parent = parents[t.parent.as_deref().unwrap()];
            // Plain RMS: the gated "active speech" level finds no speech
            // at all in a twin this quiet, which is rather the point.
            let rms = |k: &str| {
                let a = unamblify_audio::read(root.prepared_8k_decoded(k)).unwrap();
                unamblify_audio::level::rms_dbfs(&a.samples)
            };
            let drop_db = rms(&t.key) - rms(&parent.key);
            // The shelf cut takes a little more off a bass-heavy fixture.
            assert!(
                (u.gain_db - 9.0..=u.gain_db + 1.0).contains(&drop_db),
                "{}: fell {drop_db:.1} dB, drew {:.1} dB",
                t.key,
                u.gain_db
            );
            assert_eq!(t.target_key(), parent.key, "the target is the clean parent");
        }
    }

    #[test]
    fn twins_are_written_recorded_idempotent_and_redone_on_force() {
        let (_dir, root) = fixture(4);
        let opts = TwinOptions {
            noise_share: 1.0,
            noise_sets: vec![NoiseSet::Demand],
            chain_share: 0.5,
            under_share: 0.0,
            seed: 1,
        };
        let s = run(&root, &opts, &HashSet::new(), false, Some(2)).unwrap();
        assert_eq!(s.planned, 4);
        assert_eq!(s.written, 4, "{s:?}");
        assert_eq!(s.errors, 0);
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(rows.len(), 8);
        let twins: Vec<&UtteranceRow> = rows.iter().filter(|r| r.parent.is_some()).collect();
        assert_eq!(twins.len(), 4);
        let parents: HashMap<&str, &UtteranceRow> = rows
            .iter()
            .filter(|r| r.parent.is_none())
            .map(|r| (r.key.as_str(), r))
            .collect();
        let mut chained = 0;
        for t in &twins {
            let parent = parents[t.parent.as_deref().unwrap()];
            assert_eq!(t.split, parent.split, "twins take the parent's split");
            assert_eq!(t.speaker, parent.speaker);
            assert_eq!(t.duration_s, parent.duration_s);
            assert_eq!(t.target_key(), parent.key);
            assert!(files_consistent(&root, t));
            let aug = t.aug.as_ref().unwrap();
            let noise = aug.noise.as_ref().expect("every twin here is noisy");
            assert_eq!(noise.noise_set, NoiseSet::Demand);
            assert!(noise.noise_clip.starts_with("raw/demand/"));
            assert!((0.0..=20.0).contains(&noise.snr_db));
            assert!(noise.noise_offset_s >= 0.0 && noise.noise_offset_s < 3.0);
            let twin16 = unamblify_audio::read(root.prepared_16k_decoded(&t.key)).unwrap();
            let clean16 = unamblify_audio::read(root.prepared_16k_decoded(&parent.key)).unwrap();
            assert_eq!(twin16.samples.len(), clean16.samples.len());
            assert_ne!(twin16.samples, clean16.samples);
            let twin8 = unamblify_audio::read(root.prepared_8k_decoded(&t.key)).unwrap();
            assert_eq!(twin8.rate, 8_000);
            assert_eq!(twin8.samples.len(), clean16.samples.len() / 2);
            if let Some(chain) = &aug.chain {
                chained += 1;
                assert!(!chain.is_empty());
                assert!(t.key.contains("+h"));
                let level = active_rms_dbfs(&twin16.samples, 16_000).unwrap();
                assert!(
                    (level - TARGET_DBFS).abs() < 0.6,
                    "level after chain {level}"
                );
            } else {
                assert!(!t.key.contains("+h"));
                // Noise only: twin − parent is the scaled noise, at the
                // recorded SNR against the clean active speech.
                let diff: Vec<f32> = twin16
                    .samples
                    .iter()
                    .zip(&clean16.samples)
                    .map(|(a, b)| a - b)
                    .collect();
                let snr = measured_snr_db(&clean16.samples, &diff, 16_000).unwrap();
                assert!(
                    (snr - noise.snr_db).abs() < 0.5,
                    "snr {snr} vs recorded {}",
                    noise.snr_db
                );
                assert_eq!(aug.post_gain_db, 0.0);
            }
        }
        assert!(chained > 0 && chained < 4, "{chained} chained of 4");

        // Idempotent.
        let s2 = run(&root, &opts, &HashSet::new(), false, Some(2)).unwrap();
        assert_eq!(s2.written, 0);
        assert_eq!(s2.skipped, 4);
        let again: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(again.len(), 8);
        // A redone parent redoes its twin; the bytes come out the same.
        let victim = twins[0].parent.clone().unwrap();
        let before = std::fs::read(root.prepared_16k_decoded(&twins[0].key)).unwrap();
        let redone: HashSet<String> = [victim].into_iter().collect();
        let s3 = run(&root, &opts, &redone, false, Some(1)).unwrap();
        assert_eq!(s3.written, 1);
        assert_eq!(
            read_jsonl::<UtteranceRow>(&root.prepared_manifest())
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            std::fs::read(root.prepared_16k_decoded(&twins[0].key)).unwrap(),
            before,
            "a twin is a pure function of its parent, key and seed"
        );
        // --force redoes them all, still without duplicates.
        let s4 = run(&root, &opts, &HashSet::new(), true, Some(2)).unwrap();
        assert_eq!(s4.written, 4);
        let rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        assert_eq!(rows.len(), 8);
        let mut keys: Vec<&str> = rows.iter().map(|r| r.key.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 8);
        // An orphaned twin (its parent row gone) is dropped.
        let parent_gone: Vec<&UtteranceRow> = rows
            .iter()
            .filter(|r| r.key != twins[1].parent.clone().unwrap())
            .collect();
        write_jsonl(&root.prepared_manifest(), parent_gone).unwrap();
        let s5 = run(&root, &opts, &HashSet::new(), false, Some(1)).unwrap();
        assert_eq!(s5.orphaned, 1);
        assert_eq!(s5.planned, 3);
        assert_eq!(
            read_jsonl::<UtteranceRow>(&root.prepared_manifest())
                .unwrap()
                .len(),
            6
        );
    }
}
