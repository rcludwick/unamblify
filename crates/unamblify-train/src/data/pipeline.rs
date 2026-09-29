// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `[data] source = "pipeline"`: join `prepared/manifest.jsonl` with
//! `captured/<mode>/manifest.jsonl` on key, filter by split, and read the
//! WAV pair per example, applying the capture lag from
//! `captured/<mode>/canary.json` before cropping. No transformation step,
//! always fresh data, slower per step. An augmented twin (a prepared row
//! with `parent`) is paired with its *parent's* clean file: the twin's
//! own 16 kHz file is the noisy input the codec saw, not a target. With
//! `[data] kinds`, the decode-only siblings (`captured/<mode>+<kind>/`)
//! are joined too, each as further examples of the same utterances.
//!
//! With several modes (`[data] modes`), each mode's manifests are joined
//! in turn, every utterance is cropped in its own mode's frame word with
//! its own lag, and the loader draws round-robin over the modes — one
//! example of each in turn, each mode with its own shuffled cursor — so
//! every mode contributes equally to a batch regardless of how many
//! utterances it has, the same balance a shard set is built with. The
//! example's `mode` is its index in the list.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Context;
use tch::Device;
use unamblify::aug::{AugKind, capture_dir_name};
use unamblify::{
    CanaryRecord, CaptureRow, ExampleLayout, Split, UtteranceRow, VocoderMode, speaker_id,
};

use super::{Batch, CropCfg, Example, Loader, RxCfg, RxStage, Utterance, apply_lag, crop_example};
use crate::rng::Rng;

/// One joined manifest row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// Utterance key.
    pub key: String,
    /// Whose prepared 16 kHz file is the clean target: the parent's for an
    /// augmented twin, `key` otherwise.
    pub target_key: String,
    /// Corpus speaker id.
    pub speaker: String,
    /// Split the prepared row recorded.
    pub split: Split,
    /// Which capture set the degraded side comes from (`None` = base).
    pub kind: Option<AugKind>,
    /// The mode the degraded side was captured in.
    pub mode: VocoderMode,
    /// Its index in the run's mode list (what the batch's `mode` carries).
    pub mode_idx: u8,
    /// Channel frames lost and concealed (a `drops` sibling's recorded
    /// positions, in order); empty for every other capture set.
    pub erased: Vec<u32>,
}

impl Item {
    /// A plain (non-twin) item of `mode`'s base capture, mode index 0.
    #[must_use]
    pub fn plain(mode: VocoderMode, key: &str, speaker: &str, split: Split) -> Self {
        Self {
            key: key.to_owned(),
            target_key: key.to_owned(),
            speaker: speaker.to_owned(),
            split,
            kind: None,
            mode,
            mode_idx: 0,
            erased: Vec::new(),
        }
    }
}

/// Read a JSONL manifest into rows.
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line).with_context(|| format!("{}:{}", path.display(), i + 1))?,
        );
    }
    Ok(rows)
}

/// The capture lag for `mode`, from `captured/<mode>/canary.json`.
pub fn read_lag(root: &Path, mode: VocoderMode) -> anyhow::Result<i32> {
    let path = root
        .join("captured")
        .join(mode.as_str())
        .join("canary.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("{}: the capture lag is unknown without it", path.display()))?;
    let rec: CanaryRecord =
        serde_json::from_str(&text).with_context(|| path.display().to_string())?;
    Ok(rec.lag_samples)
}

/// The lag of one capture set: its own `canary.json` where it has one,
/// else its mode's. A recode sibling (`dstar+perens`) is another
/// implementation of the codec with another delay — 216 samples where the
/// chip's D-STAR decode lags by 326 — so it cannot borrow its mode's.
pub fn read_set_lag(root: &Path, mode: VocoderMode, kind: Option<AugKind>) -> anyhow::Result<i32> {
    let own = root
        .join("captured")
        .join(capture_dir_name(mode, kind))
        .join("canary.json");
    if kind.is_some() && own.is_file() {
        let text = std::fs::read_to_string(&own).with_context(|| own.display().to_string())?;
        let rec: CanaryRecord =
            serde_json::from_str(&text).with_context(|| own.display().to_string())?;
        return Ok(rec.lag_samples);
    }
    read_lag(root, mode)
}

/// The capture lag of every mode of `modes`.
pub fn read_lags(root: &Path, modes: &[VocoderMode]) -> anyhow::Result<BTreeMap<VocoderMode, i32>> {
    modes.iter().map(|&m| Ok((m, read_lag(root, m)?))).collect()
}

/// `captured/<mode>[+<kind>]/manifest.jsonl`.
#[must_use]
pub fn captured_manifest(root: &Path, mode: VocoderMode, kind: Option<AugKind>) -> PathBuf {
    root.join("captured")
        .join(capture_dir_name(mode, kind))
        .join("manifest.jsonl")
}

/// Inner join of the prepared manifest with the base capture's, in key
/// order.
pub fn join_manifests(root: &Path, mode: VocoderMode) -> anyhow::Result<Vec<Item>> {
    join_manifests_modes(root, &[mode], &[None])
}

/// Inner join of the prepared manifest with each capture set of `kinds`
/// (`None` = base, `Some` = a decode-only sibling), in kind then key
/// order; a sibling whose manifest does not exist is an error. A
/// captured twin whose parent row is missing from the prepared manifest
/// is an error: its target would be nothing.
pub fn join_manifests_kinds(
    root: &Path,
    mode: VocoderMode,
    kinds: &[Option<AugKind>],
) -> anyhow::Result<Vec<Item>> {
    join_manifests_modes(root, &[mode], kinds)
}

/// [`join_manifests_kinds`] over several modes, in mode then kind then
/// key order; each item's `mode_idx` is its mode's position in `modes`.
pub fn join_manifests_modes(
    root: &Path,
    modes: &[VocoderMode],
    kinds: &[Option<AugKind>],
) -> anyhow::Result<Vec<Item>> {
    let prepared: Vec<UtteranceRow> = read_jsonl(&root.join("prepared").join("manifest.jsonl"))?;
    let prepared_keys: HashSet<&str> = prepared.iter().map(|u| u.key.as_str()).collect();
    let mut items: Vec<Item> = Vec::new();
    for (mi, &mode) in modes.iter().enumerate() {
        let mode_idx = u8::try_from(mi).context("more modes than a mode index holds")?;
        for &kind in kinds {
            let captured: Vec<CaptureRow> = read_jsonl(&captured_manifest(root, mode, kind))?;
            let have: std::collections::HashMap<&str, &CaptureRow> =
                captured.iter().map(|c| (c.key.as_str(), c)).collect();
            let mut found: Vec<Item> = Vec::new();
            for u in &prepared {
                let Some(cap) = have.get(u.key.as_str()) else {
                    continue;
                };
                if let Some(parent) = &u.parent {
                    anyhow::ensure!(
                        prepared_keys.contains(parent.as_str()),
                        "{}: twin of {parent}, which is not in prepared/manifest.jsonl (re-run prepare)",
                        u.key
                    );
                }
                found.push(Item {
                    key: u.key.clone(),
                    target_key: u.target_key().to_owned(),
                    speaker: u.speaker.clone(),
                    split: u.split,
                    kind,
                    mode,
                    mode_idx,
                    // Erasures are lost frames only: a `ber` sibling's
                    // positions are bit errors no receiver is told about.
                    erased: cap
                        .aug
                        .as_ref()
                        .filter(|a| a.kind == AugKind::Drops)
                        .map(|a| {
                            let mut p = a.positions.clone();
                            p.sort_unstable();
                            p
                        })
                        .unwrap_or_default(),
                });
            }
            found.sort_by(|a, b| a.key.cmp(&b.key));
            found.dedup_by(|a, b| a.key == b.key);
            items.extend(found);
        }
    }
    Ok(items)
}

/// The prepared 16 kHz clean file for `key`: the FLAC if present, else
/// the WAV (`prepared/<key>.16k.flac` | `.16k.wav`).
#[must_use]
pub fn clean_path(root: &Path, key: &str) -> PathBuf {
    resolve_audio(&root.join("prepared").join(key), "16k")
}

/// `<stem>.<seg>.flac` if it exists, else `<stem>.<seg>.wav` (an empty
/// `seg` gives `<stem>.flac` | `<stem>.wav`). The move to FLAC is
/// gradual, so both may be present; the WAV path is returned when neither
/// is, so the caller reports the miss.
#[must_use]
fn resolve_audio(stem: &Path, seg: &str) -> PathBuf {
    let dot = |ext: &str| {
        if seg.is_empty() {
            stem.with_extension(ext)
        } else {
            stem.with_extension(format!("{seg}.{ext}"))
        }
    };
    let flac = dot("flac");
    if flac.is_file() { flac } else { dot("wav") }
}

/// `captured/<mode>/<key>.wav`.
#[must_use]
pub fn degraded_path(root: &Path, mode: VocoderMode, key: &str) -> PathBuf {
    degraded_path_in(root, mode, None, key)
}

/// `captured/<mode>[+<kind>]/<key>.wav`.
#[must_use]
pub fn degraded_path_in(
    root: &Path,
    mode: VocoderMode,
    kind: Option<AugKind>,
    key: &str,
) -> PathBuf {
    let stem = root
        .join("captured")
        .join(capture_dir_name(mode, kind))
        .join(key);
    resolve_audio(&stem, "")
}

/// Read one utterance pair and align it: the clean target from
/// `item.target_key` (a twin's parent), the degraded side from
/// `item.key` in `item.mode`'s capture set, `lag` samples undone.
pub fn load_utterance(root: &Path, item: &Item, lag: i32) -> anyhow::Result<Utterance> {
    let cp = clean_path(root, &item.target_key);
    let dp = degraded_path_in(root, item.mode, item.kind, &item.key);
    let clean = unamblify_audio::read(&cp).with_context(|| cp.display().to_string())?;
    let deg = unamblify_audio::read(&dp).with_context(|| dp.display().to_string())?;
    anyhow::ensure!(
        clean.rate == 16_000,
        "{}: {} Hz, want 16000",
        cp.display(),
        clean.rate
    );
    anyhow::ensure!(
        deg.rate == 8_000,
        "{}: {} Hz, want 8000",
        dp.display(),
        deg.rate
    );
    Ok(Utterance {
        clean16: clean.samples,
        deg8: apply_lag(&deg.samples, lag),
        speaker: speaker_id(&item.speaker),
        mode: item.mode_idx,
    })
}

/// `, rx N %` for a describe line.
pub(crate) fn rx_note(rx: Option<&RxStage>) -> String {
    rx.map_or(String::new(), |r| {
        format!(", rx noise on {:.0} %", r.cfg().share * 100.0)
    })
}

/// One mode's shuffled cursor over its items.
#[derive(Debug)]
struct ModeOrder {
    /// Indices into the loader's `items` of this mode.
    items: Vec<usize>,
    order: Vec<usize>,
    cursor: usize,
    epoch: u64,
}

/// The pipeline loader.
#[derive(Debug)]
pub struct PipelineLoader {
    root: PathBuf,
    modes: Vec<VocoderMode>,
    kinds: Vec<Option<AugKind>>,
    items: Vec<Item>,
    lags: BTreeMap<VocoderMode, i32>,
    /// Per capture set, by directory name: what `lags` cannot say for a
    /// recode sibling.
    set_lags: BTreeMap<String, i32>,
    cfg: CropCfg,
    orders: Vec<ModeOrder>,
    /// Which mode the next draw comes from (round-robin).
    next_mode: usize,
    /// `[data] mode_weights` as cumulative shares: draw modes by share
    /// instead of round-robin.
    weights: Option<Vec<f32>>,
    rng: Rng,
    rx: Option<RxStage>,
    /// Examples drawn so far: the rx stage's per-example index.
    drawn: u64,
}

impl PipelineLoader {
    /// Open under `root` for `mode`'s base capture, keeping the rows of
    /// `split`.
    pub fn open(
        root: &Path,
        mode: VocoderMode,
        split: Split,
        cfg: CropCfg,
        seed: u64,
    ) -> anyhow::Result<Self> {
        Self::open_modes(root, &[mode], &[None], split, cfg, seed)
    }

    /// Open under `root` for the capture sets `kinds` of `mode`, keeping
    /// the rows of `split`.
    pub fn open_kinds(
        root: &Path,
        mode: VocoderMode,
        kinds: &[Option<AugKind>],
        split: Split,
        cfg: CropCfg,
        seed: u64,
    ) -> anyhow::Result<Self> {
        Self::open_modes(root, &[mode], kinds, split, cfg, seed)
    }

    /// Open under `root` for the capture sets `kinds` of every mode of
    /// `modes`, keeping the rows of `split`. Every mode must have at
    /// least one such utterance, and `cfg.crop_s` must be whole frames of
    /// every mode (so the batches are one length); `cfg.mode` is ignored
    /// in favour of each item's own.
    pub fn open_modes(
        root: &Path,
        modes: &[VocoderMode],
        kinds: &[Option<AugKind>],
        split: Split,
        cfg: CropCfg,
        seed: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !modes.is_empty(),
            "[data] modes must name at least one vocoder mode"
        );
        anyhow::ensure!(
            !kinds.is_empty(),
            "[data] kinds must name at least one capture set"
        );
        ExampleLayout::for_modes(modes, cfg.crop_s).context("[data] crop_s")?;
        let items: Vec<Item> = join_manifests_modes(root, modes, kinds)?
            .into_iter()
            .filter(|i| i.split == split)
            .collect();
        let mut orders = Vec::with_capacity(modes.len());
        for (mi, &mode) in modes.iter().enumerate() {
            let mine: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, i)| usize::from(i.mode_idx) == mi)
                .map(|(k, _)| k)
                .collect();
            anyhow::ensure!(
                !mine.is_empty(),
                "no {split} utterances captured for {mode} under {}",
                root.display()
            );
            orders.push(ModeOrder {
                items: mine,
                order: Vec::new(),
                cursor: 0,
                epoch: 0,
            });
        }
        let lags = read_lags(root, modes)?;
        let mut set_lags = BTreeMap::new();
        for item in &items {
            if let std::collections::btree_map::Entry::Vacant(slot) =
                set_lags.entry(capture_dir_name(item.mode, item.kind))
            {
                slot.insert(read_set_lag(root, item.mode, item.kind)?);
            }
        }
        let mut loader = Self {
            root: root.to_path_buf(),
            modes: modes.to_vec(),
            kinds: kinds.to_vec(),
            items,
            lags,
            set_lags,
            cfg,
            orders,
            next_mode: 0,
            weights: None,
            rng: Rng::new(seed),
            rx: None,
            drawn: 0,
        };
        for mi in 0..loader.orders.len() {
            loader.reshuffle(mi);
        }
        Ok(loader)
    }

    /// Add receive-side noise on `deg8`, seeded from the loader's seed and
    /// the draw index.
    #[must_use]
    pub fn with_rx(mut self, cfg: RxCfg, seed: u64) -> Self {
        self.rx = RxStage::new(cfg, seed);
        self
    }

    /// Draw the modes by `weights` (`[data] mode_weights`, one per mode,
    /// relative) instead of round-robin.
    pub fn with_mode_weights(mut self, weights: &[f32]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weights.len() == self.modes.len(),
            "[data] mode_weights has {} entries for {} modes",
            weights.len(),
            self.modes.len()
        );
        anyhow::ensure!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "[data] mode_weights must be finite and >= 0"
        );
        let sum: f32 = weights.iter().sum();
        anyhow::ensure!(sum > 0.0, "[data] mode_weights are all zero");
        let mut acc = 0.0;
        self.weights = Some(
            weights
                .iter()
                .map(|w| {
                    acc += w / sum;
                    acc
                })
                .collect(),
        );
        Ok(self)
    }

    /// The joined rows in use.
    #[must_use]
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// The modes in index order.
    #[must_use]
    pub fn modes(&self) -> &[VocoderMode] {
        &self.modes
    }

    /// Lag applied to every degraded signal of the first mode.
    #[must_use]
    pub fn lag(&self) -> i32 {
        self.lag_for(self.modes[0])
    }

    /// Lag applied to `mode`'s degraded signals.
    #[must_use]
    pub fn lag_for(&self, mode: VocoderMode) -> i32 {
        self.lags.get(&mode).copied().unwrap_or(0)
    }

    /// Reshuffle mode `mi`'s items for its next epoch. The generator is
    /// forked per (epoch, mode); with one mode that is `fork(epoch)`, the
    /// stream a single-mode loader has always drawn.
    fn reshuffle(&mut self, mi: usize) {
        let n_modes = self.orders.len() as u64;
        let o = &mut self.orders[mi];
        o.order.clone_from(&o.items);
        let mut r = self.rng.fork(o.epoch * n_modes + mi as u64);
        r.shuffle(&mut o.order);
        o.cursor = 0;
        o.epoch += 1;
    }

    /// The next `n` examples in host memory.
    pub fn next_examples(&mut self, n: usize) -> anyhow::Result<Vec<Example>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mi = if let Some(cdf) = &self.weights {
                let u = self.rng.next_f32();
                cdf.partition_point(|&c| c <= u).min(cdf.len() - 1)
            } else {
                self.next_mode
            };
            self.next_mode = (mi + 1) % self.orders.len();
            if self.orders[mi].cursor >= self.orders[mi].order.len() {
                self.reshuffle(mi);
            }
            let o = &mut self.orders[mi];
            let item = &self.items[o.order[o.cursor]];
            o.cursor += 1;
            let lag = self
                .set_lags
                .get(&capture_dir_name(item.mode, item.kind))
                .copied()
                .unwrap_or_else(|| self.lags.get(&item.mode).copied().unwrap_or(0));
            let utt = load_utterance(&self.root, item, lag)?;
            let cfg = CropCfg {
                mode: item.mode,
                ..self.cfg
            };
            let mut ex = crop_example(&utt, &cfg, &mut self.rng);
            if let Some(rx) = &self.rx {
                rx.apply(&mut ex, self.drawn);
            }
            self.drawn += 1;
            out.push(ex);
        }
        Ok(out)
    }
}

impl Loader for PipelineLoader {
    fn next_batch(&mut self, batch: usize, device: Device) -> anyhow::Result<Batch> {
        let examples = self.next_examples(batch)?;
        Batch::from_examples(&examples, device)
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn describe(&self) -> String {
        let kinds: Vec<String> = self
            .kinds
            .iter()
            .map(|k| k.map_or_else(|| "base".to_owned(), |k| k.to_string()))
            .collect();
        let per_mode: Vec<String> = self
            .modes
            .iter()
            .zip(&self.orders)
            .map(|(m, o)| format!("{m} {} (lag {})", o.items.len(), self.lag_for(*m)))
            .collect();
        format!(
            "pipeline: {} utterances for {} (kinds {}) under {}, crop {} s{}{}",
            self.items.len(),
            per_mode.join(", "),
            kinds.join(","),
            self.root.display(),
            self.cfg.crop_s,
            super::shards::weights_note(self.weights.as_ref(), &self.modes),
            rx_note(self.rx.as_ref())
        )
    }
}
