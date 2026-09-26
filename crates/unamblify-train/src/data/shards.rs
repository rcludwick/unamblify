// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `[data] source = "shards"`: memory-map `shards/<name>/NNNN.bin` and read
//! fixed-length examples laid out per `unamblify::ExampleLayout`
//! (little-endian f32 `clean16`, f32 `deg8`, raw channel frames, then the
//! flag block). Examples across files are ordered by split (see
//! `unamblify::ShardIndex`), so a split is a contiguous index range.
//! [`encode_example`] / [`decode_example`] are the layout's codec; the data
//! crate's `shard` stage and this loader must agree on them. The files
//! opened are exactly those `files.json` lists (never a stale `*.bin`
//! from an earlier build), each checked against its example count, and a
//! set whose `index.json` does not record the lag it applied is refused.
//!
//! A set's examples carry their mode in the flags byte (an index into the
//! set's `modes`). Opened with a mode list ([`ShardLoader::open_modes`],
//! the run's `[data] modes`), the loader keeps only the split's examples
//! of those modes — the list must be a subset of the set's, in any order
//! — and yields each example's mode as its index in *that* list, so the
//! model's mode embedding counts the run's modes, not the set's. A set
//! written before modes existed has one mode and every example at
//! index 0.

use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Context;
use memmap2::Mmap;
use tch::Device;
use unamblify::{ExampleLayout, FLAGS_BYTES, ShardFile, ShardIndex, Split, VocoderMode};

use super::pipeline::rx_note;
use super::{Batch, Example, Loader, RxCfg, RxStage};
use crate::rng::Rng;

/// Serialise one example (with its channel frames, which the trainer does
/// not read but the layout carries) to exactly `layout.example_bytes()`.
pub fn encode_example(
    layout: &ExampleLayout,
    ex: &Example,
    frames: &[u8],
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        ex.clean16.len() == layout.clean16_samples && ex.deg8.len() == layout.deg8_samples,
        "example {} / {} does not match layout {} / {}",
        ex.clean16.len(),
        ex.deg8.len(),
        layout.clean16_samples,
        layout.deg8_samples
    );
    anyhow::ensure!(
        frames.len() == layout.frames * layout.frame_bytes,
        "frames: {} bytes, layout wants {}",
        frames.len(),
        layout.frames * layout.frame_bytes
    );
    anyhow::ensure!(
        layout.flags_bytes == FLAGS_BYTES,
        "flags_bytes {} != {FLAGS_BYTES}",
        layout.flags_bytes
    );
    let mut out = Vec::with_capacity(layout.example_bytes());
    for v in &ex.clean16 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &ex.deg8 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(frames);
    anyhow::ensure!(
        usize::from(ex.mode) < ExampleLayout::MAX_MODES,
        "mode index {} does not fit the flags byte",
        ex.mode
    );
    let mut flags = ExampleLayout::with_mode(0, ex.mode);
    if ex.onset {
        flags |= ExampleLayout::FLAG_ONSET;
    }
    if ex.tail {
        flags |= ExampleLayout::FLAG_TAIL;
    }
    out.push(flags);
    let boundary = u32::try_from(ex.mask_boundary.min(layout.clean16_samples))?;
    out.extend_from_slice(&boundary.to_le_bytes());
    out.extend_from_slice(&ex.speaker.to_le_bytes());
    debug_assert_eq!(out.len(), layout.example_bytes());
    Ok(out)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Parse one example from `layout.example_bytes()` bytes.
pub fn decode_example(layout: &ExampleLayout, bytes: &[u8]) -> anyhow::Result<Example> {
    anyhow::ensure!(
        bytes.len() == layout.example_bytes(),
        "example is {} bytes, layout says {}",
        bytes.len(),
        layout.example_bytes()
    );
    anyhow::ensure!(
        layout.flags_bytes == FLAGS_BYTES,
        "flags_bytes {} != {FLAGS_BYTES}",
        layout.flags_bytes
    );
    let clean16 = f32s(&bytes[..layout.deg8_offset()]);
    let deg8 = f32s(&bytes[layout.deg8_offset()..layout.frames_offset()]);
    let fo = layout.flags_offset();
    let flags = bytes[fo];
    let mask_boundary = usize::try_from(u32_at(bytes, fo + 1))?.min(layout.clean16_samples);
    let speaker = u32_at(bytes, fo + 5);
    Ok(Example {
        clean16,
        deg8,
        mask_boundary,
        onset: flags & ExampleLayout::FLAG_ONSET != 0,
        tail: flags & ExampleLayout::FLAG_TAIL != 0,
        speaker,
        mode: ExampleLayout::mode_index(flags),
        // As stored: one byte per channel frame of the example's own mode.
        // [`ShardLoader::raw_example`] knows the mode and spreads it over
        // the feature frames; on its own this is not yet that.
        erasure: if layout.erasure_bytes == 0 {
            Vec::new()
        } else {
            let eo = layout.erasure_offset();
            bytes[eo..eo + layout.erasure_bytes].to_vec()
        },
    })
}

/// Spread a stored erasure mask — one byte per channel frame of `mode`,
/// lag-aligned like `deg8` — over the model's feature frames. Feature
/// frame `t` is centred on sample `HOP·t`, which lies in channel frame
/// `HOP·t / frame_samples`: two feature frames per 20 ms frame, four per
/// Codec 2 1600 frame. Empty when nothing was lost, so a base capture's
/// examples stay cheap.
#[must_use]
pub fn spread_erasure(stored: &[u8], mode: VocoderMode, n8: usize) -> Vec<u8> {
    if !stored.iter().any(|&v| v != 0) {
        return Vec::new();
    }
    let hop = usize::try_from(crate::model::HOP).unwrap_or(80);
    let fs = mode.frame_samples();
    (0..n8 / hop)
        .map(|t| stored.get(t * hop / fs).copied().unwrap_or(0).min(1))
        .collect()
}

/// Split ranges over the concatenated example sequence.
#[must_use]
pub fn split_range(index: &ShardIndex, split: Split) -> Range<usize> {
    let mut start = 0usize;
    for s in Split::ALL {
        let n = usize::try_from(index.counts.get(&s).copied().unwrap_or(0)).unwrap_or(0);
        if s == split {
            return start..start + n;
        }
        start += n;
    }
    start..start
}

/// The modes a loader yields and the stored-index → yielded-index map:
/// the set's own when `req` is `None`, else `req` (each must be in the
/// set, none twice).
fn mode_remap(
    index: &ShardIndex,
    req: Option<&[VocoderMode]>,
) -> anyhow::Result<(Vec<VocoderMode>, Vec<Option<u8>>)> {
    let Some(req) = req else {
        let remap = (0..index.modes.len())
            .map(|i| u8::try_from(i).ok())
            .collect();
        return Ok((index.modes.clone(), remap));
    };
    anyhow::ensure!(!req.is_empty(), "[data] modes must name at least one mode");
    let mut remap = vec![None; index.modes.len()];
    for (ri, &m) in req.iter().enumerate() {
        let Some(si) = index.mode_index_of(m) else {
            anyhow::bail!(
                "shard set {} holds modes {} but the config asks for {m}; [data] modes must be a \
                 subset of the set's (rebuild with `unamblify shard --modes …`)",
                index.name,
                index
                    .modes
                    .iter()
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        };
        anyhow::ensure!(
            remap[usize::from(si)].is_none(),
            "[data] modes lists {m} twice"
        );
        remap[usize::from(si)] = Some(u8::try_from(ri)?);
    }
    Ok((req.to_vec(), remap))
}

/// Map every file `files.json` lists, checking each against its example
/// count. Returns the maps, the cumulative example count before each
/// file, and the total.
fn map_files(dir: &Path, layout: &ExampleLayout) -> anyhow::Result<(Vec<Mmap>, Vec<usize>, usize)> {
    let files_path = dir.join("files.json");
    let files: Vec<ShardFile> = serde_json::from_str(
        &std::fs::read_to_string(&files_path).with_context(|| files_path.display().to_string())?,
    )
    .with_context(|| files_path.display().to_string())?;
    anyhow::ensure!(
        !files.is_empty(),
        "no files listed in {}",
        files_path.display()
    );
    let mut maps = Vec::with_capacity(files.len());
    let mut starts = Vec::with_capacity(files.len());
    let mut total = 0usize;
    for sf in &files {
        let f = dir.join(&sf.file);
        let file = std::fs::File::open(&f).with_context(|| f.display().to_string())?;
        // SAFETY: the shard files are written once by the data stage
        // and never modified while a run reads them; the map is
        // read-only.
        let map = unsafe { Mmap::map(&file) }.with_context(|| f.display().to_string())?;
        let n = usize::try_from(sf.examples)?;
        anyhow::ensure!(
            map.len() == n * layout.example_bytes(),
            "{}: {} bytes, but files.json says {n} examples of {} bytes",
            f.display(),
            map.len(),
            layout.example_bytes()
        );
        starts.push(total);
        total += n;
        maps.push(map);
    }
    Ok((maps, starts, total))
}

/// The shard loader.
#[derive(Debug)]
pub struct ShardLoader {
    dir: PathBuf,
    index: ShardIndex,
    layout: ExampleLayout,
    maps: Vec<Mmap>,
    /// Cumulative example count before each file.
    starts: Vec<usize>,
    total: usize,
    /// The split's global index range.
    range: Range<usize>,
    /// The split's examples this loader draws from: the whole range, or
    /// those of the requested modes.
    indices: Vec<usize>,
    /// The modes yielded, in the order the examples' `mode` indexes.
    modes: Vec<VocoderMode>,
    /// Stored mode index → yielded mode index (`None` = filtered out).
    remap: Vec<Option<u8>>,
    order: Vec<usize>,
    cursor: usize,
    rng: Rng,
    epoch: u64,
    rx: Option<RxStage>,
    /// `[data] mode_weights`: draw modes by share instead of uniformly.
    weighted: Option<Weighted>,
}

/// Per-mode draw shares, each mode with its own shuffled cursor.
#[derive(Debug)]
struct Weighted {
    /// Cumulative shares over the loader's modes, ending at 1.
    cdf: Vec<f32>,
    /// The split's examples of each mode.
    groups: Vec<Vec<usize>>,
    orders: Vec<Vec<usize>>,
    cursors: Vec<usize>,
    epochs: Vec<u64>,
}

impl Weighted {
    /// Normalised cumulative shares of `weights`, or why they are unusable.
    fn cdf(weights: &[f32], n_modes: usize) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            weights.len() == n_modes,
            "[data] mode_weights has {} entries for {n_modes} modes",
            weights.len()
        );
        anyhow::ensure!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "[data] mode_weights must be finite and >= 0"
        );
        let sum: f32 = weights.iter().sum();
        anyhow::ensure!(sum > 0.0, "[data] mode_weights are all zero");
        let mut acc = 0.0;
        Ok(weights
            .iter()
            .map(|w| {
                acc += w / sum;
                acc
            })
            .collect())
    }

    /// The mode a uniform draw `u` in `[0, 1)` lands in.
    fn pick(&self, u: f32) -> usize {
        self.cdf
            .partition_point(|&c| c <= u)
            .min(self.cdf.len() - 1)
    }

    fn reshuffle(&mut self, rng: &mut Rng, m: usize) {
        self.orders[m].clone_from(&self.groups[m]);
        let n = self.groups.len() as u64;
        let mut r = rng.fork(0x5747 + self.epochs[m] * n + m as u64);
        r.shuffle(&mut self.orders[m]);
        self.cursors[m] = 0;
        self.epochs[m] += 1;
    }
}

impl ShardLoader {
    /// Draw the modes by `weights` (`[data] mode_weights`, one per mode
    /// of this loader, relative) instead of uniformly over the split's
    /// examples: each draw picks a mode by share, then the next example
    /// of that mode's own shuffled order. A mode with a positive weight
    /// and no examples is an error.
    pub fn with_mode_weights(mut self, weights: &[f32]) -> anyhow::Result<Self> {
        let n = self.modes.len();
        let cdf = Weighted::cdf(weights, n)?;
        let mut groups = vec![Vec::new(); n];
        for &i in &self.indices {
            if let Some(m) = self
                .remap
                .get(usize::from(ExampleLayout::mode_index(self.raw_flags(i))))
                .copied()
                .flatten()
            {
                groups[usize::from(m)].push(i);
            }
        }
        for (m, (w, g)) in weights.iter().zip(&groups).enumerate() {
            anyhow::ensure!(
                *w == 0.0 || !g.is_empty(),
                "[data] mode_weights gives {} a share but the split has no examples of it",
                self.modes[m]
            );
        }
        let mut w = Weighted {
            cdf,
            orders: vec![Vec::new(); n],
            cursors: vec![0; n],
            epochs: vec![0; n],
            groups,
        };
        for m in 0..n {
            w.reshuffle(&mut self.rng, m);
        }
        self.weighted = Some(w);
        Ok(self)
    }

    /// Open `shards/<name>/` (`dir`) for `split`, every mode of the set,
    /// mode indices as stored.
    pub fn open(dir: &Path, split: Split, seed: u64) -> anyhow::Result<Self> {
        Self::open_modes(dir, split, seed, None)
    }

    /// Open `shards/<name>/` (`dir`) for `split`. With `modes`, keep only
    /// the split's examples of those modes and yield each example's mode
    /// as its index in `modes`; every one of them must be in the set (a
    /// subset in any order is fine), else an error naming the set's modes.
    pub fn open_modes(
        dir: &Path,
        split: Split,
        seed: u64,
        modes: Option<&[VocoderMode]>,
    ) -> anyhow::Result<Self> {
        let index_path = dir.join("index.json");
        let index: ShardIndex = serde_json::from_str(
            &std::fs::read_to_string(&index_path)
                .with_context(|| index_path.display().to_string())?,
        )
        .with_context(|| index_path.display().to_string())?;
        let layout = index.example_layout.clone();
        anyhow::ensure!(
            index.lag_samples.is_some(),
            "{}: index.json does not record lag_samples, so the set's alignment is unknown; rebuild it \
             with `unamblify shard` (a set built without canary.json is misaligned)",
            index_path.display()
        );
        // A recode sibling has its own codec delay. A set that drew one
        // before the builder recorded a lag per capture set cut it at its
        // mode's lag: `dstar+perens` at 326 where the truth is 216. A model
        // trained on that came out worse than the raw decode on D-STAR.
        anyhow::ensure!(
            !(index.set_lags.is_empty() && index.kinds.iter().any(|k| k.contains("perens"))),
            "{}: the set draws a recode sibling ({}) but records no per-set lags, so that sibling \
             was cut at its mode's lag and is misaligned by the difference; rebuild it with \
             `unamblify shard`",
            index_path.display(),
            index.kinds.join(",")
        );
        let (modes, remap) = mode_remap(&index, modes)?;
        let (maps, starts, total) = map_files(dir, &layout)?;
        let counted: u64 = index.counts.values().sum();
        anyhow::ensure!(
            counted == total as u64,
            "index.json counts {counted} examples but files.json lists {total}"
        );
        let range = split_range(&index, split);
        anyhow::ensure!(
            !range.is_empty(),
            "shard set {} has no {split} examples",
            index.name
        );
        let mut loader = Self {
            dir: dir.to_path_buf(),
            index,
            layout,
            maps,
            starts,
            total,
            range: range.clone(),
            indices: Vec::new(),
            modes,
            remap,
            order: Vec::new(),
            cursor: 0,
            rng: Rng::new(seed),
            epoch: 0,
            rx: None,
            weighted: None,
        };
        // Every stored mode wanted: the whole range. Otherwise read each
        // example's flags byte once and keep the wanted modes.
        loader.indices = if loader.remap.iter().all(Option::is_some) {
            range.collect()
        } else {
            range
                .filter(|&i| {
                    let flags = loader.raw_flags(i);
                    loader
                        .remap
                        .get(usize::from(ExampleLayout::mode_index(flags)))
                        .is_some_and(Option::is_some)
                })
                .collect()
        };
        anyhow::ensure!(
            !loader.indices.is_empty(),
            "shard set {} has no {split} examples of modes {}",
            loader.index.name,
            loader
                .modes
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        loader.reshuffle();
        Ok(loader)
    }

    /// The flags byte of stored example `i`.
    fn raw_flags(&self, i: usize) -> u8 {
        let file = self.starts.partition_point(|&s| s <= i) - 1;
        let local = i - self.starts[file];
        let eb = self.layout.example_bytes();
        self.maps[file][local * eb + self.layout.flags_offset()]
    }

    /// Add receive-side noise on `deg8`, seeded from the loader's seed and
    /// the example's global index, so the same example always gets the
    /// same noise for a given seed.
    #[must_use]
    pub fn with_rx(mut self, cfg: RxCfg, seed: u64) -> Self {
        self.rx = RxStage::new(cfg, seed);
        self
    }

    /// The parsed `index.json`.
    #[must_use]
    pub const fn index(&self) -> &ShardIndex {
        &self.index
    }

    /// Examples in every split.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.total
    }

    /// The modes yielded, in the order the examples' `mode` indexes them.
    #[must_use]
    pub fn modes(&self) -> &[VocoderMode] {
        &self.modes
    }

    /// Examples of the split (every mode, as stored).
    #[must_use]
    pub fn split_len(&self) -> usize {
        self.range.len()
    }

    /// The capture lag the set undid when it paired `deg8` with `clean16`.
    #[must_use]
    pub fn lag(&self) -> i32 {
        self.index.lag_samples.unwrap_or(0)
    }

    fn reshuffle(&mut self) {
        self.order.clone_from(&self.indices);
        let mut r = self.rng.fork(self.epoch);
        r.shuffle(&mut self.order);
        self.cursor = 0;
        self.epoch += 1;
    }

    /// Example `i` of the whole set (global index), as stored except that
    /// its `mode` is remapped to this loader's mode list; an example of a
    /// mode the loader was not opened for is an error.
    pub fn raw_example(&self, i: usize) -> anyhow::Result<Example> {
        anyhow::ensure!(i < self.total, "example {i} of {}", self.total);
        let file = self.starts.partition_point(|&s| s <= i) - 1;
        let local = i - self.starts[file];
        let eb = self.layout.example_bytes();
        let bytes = &self.maps[file][local * eb..(local + 1) * eb];
        let mut ex = decode_example(&self.layout, bytes)?;
        ex.mode = self
            .remap
            .get(usize::from(ex.mode))
            .copied()
            .flatten()
            .with_context(|| {
                format!(
                    "example {i} is mode index {} of {}, not one this loader was opened for",
                    ex.mode, self.index.name
                )
            })?;
        let mode = self.modes[usize::from(ex.mode)];
        ex.erasure = spread_erasure(&ex.erasure, mode, ex.deg8.len());
        Ok(ex)
    }

    /// Example `i` of the whole set (global index), with the rx stage
    /// applied when one is on.
    pub fn example(&self, i: usize) -> anyhow::Result<Example> {
        let mut ex = self.raw_example(i)?;
        if let Some(rx) = &self.rx {
            rx.apply(&mut ex, i as u64);
        }
        Ok(ex)
    }

    /// The next `n` examples in host memory.
    pub fn next_examples(&mut self, n: usize) -> anyhow::Result<Vec<Example>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let i = if let Some(w) = self.weighted.as_mut() {
                let m = w.pick(self.rng.next_f32());
                if w.cursors[m] >= w.orders[m].len() {
                    w.reshuffle(&mut self.rng, m);
                }
                let i = w.orders[m][w.cursors[m]];
                w.cursors[m] += 1;
                i
            } else {
                if self.cursor >= self.order.len() {
                    self.reshuffle();
                }
                let i = self.order[self.cursor];
                self.cursor += 1;
                i
            };
            out.push(self.example(i)?);
        }
        Ok(out)
    }
}

impl Loader for ShardLoader {
    fn next_batch(&mut self, batch: usize, device: Device) -> anyhow::Result<Batch> {
        let examples = self.next_examples(batch)?;
        Batch::from_examples(&examples, device)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn describe(&self) -> String {
        let modes: Vec<String> = self
            .modes
            .iter()
            .map(|m| format!("{m} (lag {})", self.index.lag_for(*m).unwrap_or(0)))
            .collect();
        format!(
            "shards: {} (modes {}, kinds {}), {} of {} examples under {}{}{}{}",
            self.index.name,
            modes.join(", "),
            self.index.kinds.join(","),
            self.indices.len(),
            self.total,
            self.dir.display(),
            if self.index.balance { ", balanced" } else { "" },
            weights_note(self.weighted.as_ref().map(|w| &w.cdf), &self.modes),
            rx_note(self.rx.as_ref())
        )
    }
}

/// `, mode shares dstar 30 %, ysf-dmr 15 %, …` for a `describe` line.
pub(crate) fn weights_note(cdf: Option<&Vec<f32>>, modes: &[VocoderMode]) -> String {
    let Some(cdf) = cdf else {
        return String::new();
    };
    let mut prev = 0.0;
    let parts: Vec<String> = cdf
        .iter()
        .zip(modes)
        .map(|(&c, m)| {
            let share = c - prev;
            prev = c;
            format!("{m} {:.0} %", share * 100.0)
        })
        .collect();
    format!(", mode shares {}", parts.join(", "))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_precision_loss
)]
pub(crate) mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use unamblify::{CaptureRow, SourceSha256s, UtteranceRow, VocoderMode};

    use super::*;
    use crate::data::CropCfg;
    use crate::data::pipeline::{PipelineLoader, load_utterance};
    use crate::data::synthetic;

    /// Write a shard set from host examples, `per_file` examples per bin,
    /// counts recorded for `split` only.
    pub(crate) fn write_shard_set(
        dir: &Path,
        mode: VocoderMode,
        crop_s: f32,
        split: Split,
        examples: &[Example],
        per_file: usize,
    ) -> anyhow::Result<ShardIndex> {
        write_shard_set_modes(dir, &[mode], crop_s, split, examples, per_file)
    }

    /// [`write_shard_set`] over several modes: the storage layout is
    /// theirs, each example's `mode` indexes `modes`, and the per-mode
    /// counts are tallied from the examples.
    pub(crate) fn write_shard_set_modes(
        dir: &Path,
        modes: &[VocoderMode],
        crop_s: f32,
        split: Split,
        examples: &[Example],
        per_file: usize,
    ) -> anyhow::Result<ShardIndex> {
        std::fs::create_dir_all(dir)?;
        let layout = ExampleLayout::for_modes(modes, crop_s)?;
        let frames = vec![0u8; layout.frames * layout.frame_bytes];
        let mut files = Vec::new();
        for (fi, chunk) in examples.chunks(per_file).enumerate() {
            let mut bytes = Vec::new();
            for ex in chunk {
                bytes.extend(encode_example(&layout, ex, &frames)?);
            }
            let name = format!("{fi:04}.bin");
            std::fs::write(dir.join(&name), bytes)?;
            files.push(ShardFile {
                file: name,
                split,
                examples: u64::try_from(chunk.len())?,
            });
        }
        std::fs::write(
            dir.join("files.json"),
            serde_json::to_string_pretty(&files)?,
        )?;
        let mut counts = BTreeMap::new();
        counts.insert(split, u64::try_from(examples.len())?);
        let mut per_mode: BTreeMap<VocoderMode, u64> = modes.iter().map(|&m| (m, 0)).collect();
        for ex in examples {
            *per_mode.entry(modes[usize::from(ex.mode)]).or_insert(0) += 1;
        }
        let counts_by_mode: BTreeMap<Split, BTreeMap<VocoderMode, u64>> =
            [(split, per_mode)].into();
        let index = ShardIndex {
            name: dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            mode: modes[0],
            modes: modes.to_vec(),
            erasure: false,
            crop_s,
            seed: 1,
            lag_samples: Some(5),
            lags: modes.iter().map(|&m| (m, 5)).collect(),
            set_lags: std::collections::BTreeMap::new(),
            counts,
            counts_by_mode: counts_by_mode.clone(),
            available_by_mode: counts_by_mode,
            balance: false,
            max_utterances: None,
            twin_share: None,
            min_drop_rate: None,
            corpora: Vec::new(),
            onset_share: 0.34,
            tail_share: 0.15,
            kinds: vec!["base".to_owned()],
            source_sha256s: SourceSha256s {
                prepared: "fixture".to_owned(),
                captured: "fixture".to_owned(),
                siblings: std::collections::BTreeMap::new(),
                by_set: std::collections::BTreeMap::new(),
            },
            example_layout: layout,
            mode_layouts: modes
                .iter()
                .map(|&m| (m, ExampleLayout::for_crop(m, crop_s)))
                .collect(),
        };
        std::fs::write(
            dir.join("index.json"),
            serde_json::to_string_pretty(&index)?,
        )?;
        Ok(index)
    }

    /// A tiny data root: `n` synthetic utterances as prepared 16 kHz WAVs
    /// and captured 8 kHz WAVs delayed by `lag` samples, with manifests
    /// and a canary record. Returns the keys.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn write_fixture_root(
        root: &Path,
        mode: VocoderMode,
        n: usize,
        lag: i32,
        split: Split,
    ) -> anyhow::Result<Vec<String>> {
        let prepared = root.join("prepared").join("fx");
        let captured = root.join("captured").join(mode.as_str()).join("fx");
        std::fs::create_dir_all(&prepared)?;
        std::fs::create_dir_all(&captured)?;
        let mut utt_rows = Vec::new();
        let mut cap_rows = Vec::new();
        let mut keys = Vec::new();
        for i in 0..n {
            let key = format!("fx/u{i:03}");
            let u = synthetic::utterance(100 + i as u64, 1.0 + 0.5 * i as f32)?;
            unamblify_audio::write_wav_s16(
                root.join("prepared").join(format!("{key}.16k.wav")),
                &u.clean16,
                16_000,
            )?;
            let mut delayed = vec![0.0f32; usize::try_from(lag.max(0))?];
            delayed.extend_from_slice(&u.deg8);
            unamblify_audio::write_wav_s16(
                root.join("captured")
                    .join(mode.as_str())
                    .join(format!("{key}.wav")),
                &delayed,
                8_000,
            )?;
            utt_rows.push(UtteranceRow {
                key: key.clone(),
                corpus: "fx".to_owned(),
                speaker: format!("s{}", i % 2),
                gender: None,
                split,
                duration_s: f64::from(1.0 + 0.5 * i as f32),
                src_rate: 16_000,
                src_path: format!("raw/{key}.wav"),
                licence: "test".to_owned(),
                rms_dbfs_in: -26.0,
                gain_db: 0.0,
                trim_lead_s: 0.0,
                trim_tail_s: 0.0,
                sha256_16k: "x".to_owned(),
                sha256_8k: "y".to_owned(),
                prepared_at: "2026-09-10T00:00:00Z".to_owned(),
                parent: None,
                aug: None,
            });
            cap_rows.push(CaptureRow {
                key: key.clone(),
                mode,
                frames: u32::try_from(delayed.len().div_ceil(mode.frame_samples()))?,
                port: "/dev/null".to_owned(),
                prodid: "AMBE3000F".to_owned(),
                version: "V0".to_owned(),
                encode_ms: 1,
                decode_ms: 1,
                roundtrip_ms: None,
                sha256_ambe: "a".to_owned(),
                sha256_wav: "w".to_owned(),
                captured_at: "2026-09-10T00:00:00Z".to_owned(),
                attempts: 1,
                warm_state: None,
                aug: None,
            });
            keys.push(key);
        }
        let jsonl = |rows: Vec<String>| rows.join("\n") + "\n";
        std::fs::write(
            root.join("prepared").join("manifest.jsonl"),
            jsonl(
                utt_rows
                    .iter()
                    .map(|r| serde_json::to_string(r).unwrap())
                    .collect(),
            ),
        )?;
        std::fs::write(
            root.join("captured")
                .join(mode.as_str())
                .join("manifest.jsonl"),
            jsonl(
                cap_rows
                    .iter()
                    .map(|r| serde_json::to_string(r).unwrap())
                    .collect(),
            ),
        )?;
        let canary = unamblify::CanaryRecord {
            mode,
            clip: "canary/x.wav".to_owned(),
            frames_sha256: "c".to_owned(),
            frames_first_16: "00".to_owned(),
            lag_samples: lag,
            prodid: "AMBE3000F".to_owned(),
            version: "V0".to_owned(),
            recorded_at: "2026-09-10T00:00:00Z".to_owned(),
            warm_up_frames: 20,
        };
        std::fs::write(
            root.join("captured")
                .join(mode.as_str())
                .join("canary.json"),
            serde_json::to_string(&canary)?,
        )?;
        Ok(keys)
    }

    #[test]
    fn a_twin_is_paired_with_its_parents_clean_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        let mode = VocoderMode::Codec2_3200;
        let keys = write_fixture_root(&root, mode, 2, 3, Split::Train).unwrap();
        // A twin of the first utterance: its own (silent) input files and
        // a captured side, its parent's clean file as the target.
        let parent = &keys[0];
        let twin = format!("{parent}+h0042");
        let p16 =
            unamblify_audio::read_wav(root.join("prepared").join(format!("{parent}.16k.wav")))
                .unwrap()
                .samples;
        let n16 = p16.len();
        unamblify_audio::write_wav_s16(
            root.join("prepared").join(format!("{twin}.16k.wav")),
            &vec![0.0f32; n16],
            16_000,
        )
        .unwrap();
        let deg: Vec<f32> = (0..n16 / 2)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        unamblify_audio::write_wav_s16(
            root.join("captured")
                .join(mode.as_str())
                .join(format!("{twin}.wav")),
            &deg,
            8_000,
        )
        .unwrap();
        let append = |name: &str, line: String| {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(root.join(name))
                .unwrap();
            writeln!(f, "{line}").unwrap();
        };
        let rows: Vec<UtteranceRow> =
            crate::data::pipeline::read_jsonl(&root.join("prepared/manifest.jsonl")).unwrap();
        let twin_row = UtteranceRow {
            key: twin.clone(),
            parent: Some(parent.clone()),
            ..rows[0].clone()
        };
        append(
            "prepared/manifest.jsonl",
            serde_json::to_string(&twin_row).unwrap(),
        );
        let caps: Vec<CaptureRow> = crate::data::pipeline::read_jsonl(
            &root
                .join("captured")
                .join(mode.as_str())
                .join("manifest.jsonl"),
        )
        .unwrap();
        let cap = CaptureRow {
            key: twin.clone(),
            ..caps[0].clone()
        };
        append(
            &format!("captured/{}/manifest.jsonl", mode.as_str()),
            serde_json::to_string(&cap).unwrap(),
        );
        let items = crate::data::pipeline::join_manifests(&root, mode).unwrap();
        assert_eq!(items.len(), 3);
        let it = items.iter().find(|i| i.key == twin).unwrap();
        assert_eq!(it.target_key, *parent);
        assert_eq!(items[0].target_key, items[0].key);
        let utt = crate::data::pipeline::load_utterance(&root, it, 3).unwrap();
        assert_eq!(utt.clean16, p16, "the target is the parent's clean file");
        assert!((utt.deg8[0] - 0.25).abs() < 1e-3 || (utt.deg8[0] + 0.25).abs() < 1e-3);
        // A twin whose parent row is gone is refused.
        let kept: Vec<String> = rows
            .iter()
            .filter(|r| &r.key != parent)
            .chain(std::iter::once(&twin_row))
            .map(|r| serde_json::to_string(r).unwrap())
            .collect();
        std::fs::write(root.join("prepared/manifest.jsonl"), kept.join("\n") + "\n").unwrap();
        let err = crate::data::pipeline::join_manifests(&root, mode)
            .unwrap_err()
            .to_string();
        assert!(err.contains("twin of"), "{err}");
    }

    /// The stored mask is per channel frame of the example's mode; the
    /// model reads feature frames. Two per 20 ms frame, four per Codec 2
    /// 1600 frame, and nothing at all when nothing was lost.
    #[test]
    fn the_erasure_mask_spreads_over_feature_frames_per_mode() {
        let ones = |m: &[u8]| -> Vec<usize> {
            m.iter()
                .enumerate()
                .filter(|(_, v)| **v == 1)
                .map(|(i, _)| i)
                .collect()
        };
        // 1 s of D-STAR: 50 channel frames, 100 feature frames.
        let mut stored = vec![0u8; 50];
        stored[7] = 1;
        let m = spread_erasure(&stored, VocoderMode::Dstar, 8_000);
        assert_eq!((m.len(), ones(&m)), (100, vec![14, 15]));
        // The same second of Codec 2 1600: 25 channel frames of 40 ms.
        let mut stored = vec![0u8; 50]; // storage is sized for the largest mode
        stored[7] = 1;
        let m = spread_erasure(&stored, VocoderMode::Codec2_1600, 8_000);
        assert_eq!((m.len(), ones(&m)), (100, vec![28, 29, 30, 31]));
        // Nothing lost: no mask at all, so a base example costs nothing.
        assert!(spread_erasure(&[0u8; 50], VocoderMode::Dstar, 8_000).is_empty());
        assert!(spread_erasure(&[], VocoderMode::Dstar, 8_000).is_empty());
    }

    #[test]
    fn example_codec_round_trips() {
        let layout = ExampleLayout::for_crop(VocoderMode::YsfDmr, 0.1);
        let ex = Example {
            clean16: (0..layout.clean16_samples)
                .map(|i| i as f32 * 0.001)
                .collect(),
            deg8: (0..layout.deg8_samples)
                .map(|i| -(i as f32) * 0.002)
                .collect(),
            mask_boundary: 700,
            onset: true,
            tail: true,
            speaker: 0xDEAD_BEEF,
            mode: 5,
            erasure: Vec::new(),
        };
        let frames = vec![7u8; layout.frames * layout.frame_bytes];
        let bytes = encode_example(&layout, &ex, &frames).unwrap();
        assert_eq!(bytes.len(), layout.example_bytes());
        assert_eq!(decode_example(&layout, &bytes).unwrap(), ex);
        // The mode rides in the flags byte above onset and tail.
        let flags = bytes[layout.flags_offset()];
        assert_eq!(flags, ExampleLayout::with_mode(0b11, 5));
        let blind = Example {
            mode: 64,
            ..ex.clone()
        };
        assert!(encode_example(&layout, &blind, &frames).is_err());
        assert!(encode_example(&layout, &ex, &frames[1..]).is_err());
        assert!(decode_example(&layout, &bytes[1..]).is_err());
    }

    #[test]
    fn pipeline_and_shards_yield_identical_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        let mode = VocoderMode::Dstar;
        let lag = 5;
        write_fixture_root(&root, mode, 3, lag, Split::Train).unwrap();
        let cfg = CropCfg {
            mode,
            crop_s: 0.5,
            onset_share: 0.5,
            tail_share: 0.5,
            gain_jitter_db: 0.0,
        };
        let mut pipe = PipelineLoader::open(&root, mode, Split::Train, cfg, 42).unwrap();
        assert_eq!(pipe.len(), 3);
        assert_eq!(pipe.lag(), lag);
        assert!(PipelineLoader::open(&root, mode, Split::Dev, cfg, 42).is_err());
        let examples = pipe.next_examples(7).unwrap();
        assert!(examples.iter().any(|e| e.onset));
        assert!(examples.iter().any(|e| e.tail));
        // Lag was undone: the degraded crop of an onset example starts
        // with the utterance's own first samples, not the zero delay.
        let onset = examples.iter().find(|e| e.onset).unwrap();
        assert!(onset.deg8[..160].iter().any(|&v| v != 0.0));

        let dir = tmp.path().join("shards").join("fx-dstar");
        write_shard_set(&dir, mode, 0.5, Split::Train, &examples, 3).unwrap();
        assert!(dir.join("0002.bin").exists());
        let mut shards = ShardLoader::open(&dir, Split::Train, 42).unwrap();
        assert_eq!(shards.len(), 7);
        assert_eq!(shards.total(), 7);
        for (i, ex) in examples.iter().enumerate() {
            assert_eq!(&shards.example(i).unwrap(), ex, "example {i}");
        }
        assert!(ShardLoader::open(&dir, Split::Dev, 1).is_err());
        // A stale file from an earlier build is ignored: only files.json
        // counts. A set without the lag is refused.
        std::fs::write(dir.join("0009.bin"), b"stale").unwrap();
        assert_eq!(
            ShardLoader::open(&dir, Split::Train, 42).unwrap().total(),
            7
        );
        let idx_path = dir.join("index.json");
        let text = std::fs::read_to_string(&idx_path).unwrap();
        std::fs::write(&idx_path, text.replace("\"lag_samples\": 5,", "")).unwrap();
        let err = ShardLoader::open(&dir, Split::Train, 42).unwrap_err();
        assert!(err.to_string().contains("lag_samples"), "{err}");
        std::fs::write(&idx_path, text).unwrap();

        // Batches: one epoch of the shard loader covers every example
        // once, each equal to the pipeline's tensors for that example.
        let batch = shards.next_batch(7, Device::Cpu).unwrap();
        let reference = Batch::from_examples(&examples, Device::Cpu).unwrap();
        let mut seen = [false; 7];
        for b in 0..7i64 {
            let deg = batch.deg8.get(b);
            let found = (0..7i64)
                .find(|&r| (&deg - reference.deg8.get(r)).abs().max().double_value(&[]) == 0.0)
                .expect("shard example matches a pipeline example");
            let fi = usize::try_from(found).unwrap();
            assert!(!seen[fi]);
            seen[fi] = true;
            let same =
                |a: &tch::Tensor, c: &tch::Tensor| (a - c).abs().max().double_value(&[]) == 0.0;
            assert!(same(&batch.clean16.get(b), &reference.clean16.get(found)));
            assert!(same(&batch.mask.get(b), &reference.mask.get(found)));
            assert_eq!(
                batch.onset.int64_value(&[b]),
                reference.onset.int64_value(&[found])
            );
            assert_eq!(
                batch.tail.int64_value(&[b]),
                reference.tail.int64_value(&[found])
            );
            assert_eq!(
                batch.speaker.int64_value(&[b]),
                reference.speaker.int64_value(&[found])
            );
        }
        assert!(seen.iter().all(|&s| s));
        assert!(shards.describe().contains("fx-dstar"));
        assert!(pipe.describe().contains("pipeline"));
    }

    /// With the rx stage on, both loaders add the noise `RxStage` draws
    /// for (seed, index) on top of the same raw example: the shard loader
    /// keyed by the example's global index, the pipeline by its draw
    /// index. A raw pipeline draw written to a shard set reads back with
    /// exactly the noise the stage gives that index.
    #[test]
    fn rx_noise_is_the_same_stage_in_both_loaders() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        let mode = VocoderMode::Dstar;
        write_fixture_root(&root, mode, 3, 5, Split::Train).unwrap();
        let cfg = CropCfg {
            mode,
            crop_s: 0.5,
            onset_share: 0.5,
            tail_share: 0.5,
            gain_jitter_db: 0.0,
        };
        let rx = RxCfg {
            share: 1.0,
            hum: true,
            broadband: true,
            whine: true,
            colouring: true,
            squelch: true,
        };
        // Raw draws, and the same draws with rx on: the k-th noisy draw is
        // the k-th raw draw plus the stage's noise for index k.
        let raw = PipelineLoader::open(&root, mode, Split::Train, cfg, 42)
            .unwrap()
            .next_examples(5)
            .unwrap();
        let noisy = PipelineLoader::open(&root, mode, Split::Train, cfg, 42)
            .unwrap()
            .with_rx(rx, 42)
            .next_examples(5)
            .unwrap();
        let stage = RxStage::new(rx, 42).unwrap();
        for (k, (r, n)) in raw.iter().zip(&noisy).enumerate() {
            let mut expect = r.clone();
            stage.apply(&mut expect, k as u64);
            assert_eq!(n, &expect, "pipeline draw {k}");
            assert_eq!(n.clean16, r.clean16, "target untouched");
            assert_ne!(n.deg8, r.deg8, "input noised");
        }
        let dir = tmp.path().join("shards").join("rx");
        write_shard_set(&dir, mode, 0.5, Split::Train, &raw, 2).unwrap();
        let shards = ShardLoader::open(&dir, Split::Train, 42)
            .unwrap()
            .with_rx(rx, 42);
        for (i, r) in raw.iter().enumerate() {
            assert_eq!(&shards.raw_example(i).unwrap(), r);
            let mut expect = r.clone();
            stage.apply(&mut expect, i as u64);
            assert_eq!(shards.example(i).unwrap(), expect, "shard example {i}");
            assert_eq!(
                shards.example(i).unwrap(),
                noisy[i],
                "same stage as the pipeline"
            );
        }
        assert!(shards.describe().contains("rx noise on 100 %"));
        // Off: the loaders hand out the raw examples.
        let plain = ShardLoader::open(&dir, Split::Train, 42).unwrap();
        assert_eq!(plain.example(2).unwrap(), raw[2]);
    }

    /// A set written before `modes` existed (no `modes`, `lags`,
    /// `mode_layouts` or per-mode counts in its index, no mode bits in
    /// its flags) opens as a single-mode set at index 0 — with no mode
    /// list, and with the run's `[data] modes` naming that one mode.
    #[test]
    fn a_single_mode_set_without_modes_in_its_index_still_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        let mode = VocoderMode::YsfDmr;
        write_fixture_root(&root, mode, 2, 5, Split::Train).unwrap();
        let cfg = CropCfg {
            mode,
            crop_s: 0.5,
            gain_jitter_db: 0.0,
            ..CropCfg::default()
        };
        let examples = PipelineLoader::open(&root, mode, Split::Train, cfg, 1)
            .unwrap()
            .next_examples(4)
            .unwrap();
        let dir = tmp.path().join("shards").join("v1");
        write_shard_set(&dir, mode, 0.5, Split::Train, &examples, 4).unwrap();
        // Strip everything a v1 index did not have.
        let idx_path = dir.join("index.json");
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&idx_path).unwrap()).unwrap();
        for k in [
            "modes",
            "lags",
            "mode_layouts",
            "counts_by_mode",
            "available_by_mode",
            "balance",
        ] {
            assert!(v.as_object_mut().unwrap().remove(k).is_some(), "{k}");
        }
        std::fs::write(&idx_path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        let loader = ShardLoader::open(&dir, Split::Train, 1).unwrap();
        assert_eq!(loader.modes(), &[mode]);
        assert_eq!(loader.index().modes, vec![mode]);
        assert_eq!(loader.index().lag_for(mode), Some(5));
        assert_eq!(
            loader.index().mode_layouts[&mode],
            loader.index().example_layout
        );
        assert_eq!(loader.len(), 4);
        for (i, ex) in examples.iter().enumerate() {
            let got = loader.raw_example(i).unwrap();
            assert_eq!(got.mode, 0);
            assert_eq!(&got, ex);
        }
        let named = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[mode])).unwrap();
        assert_eq!(named.len(), 4);
        assert_eq!(named.raw_example(2).unwrap().mode, 0);
        let err = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[VocoderMode::Dstar]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("holds modes ysf-dmr"), "{err}");
        assert!(
            named.describe().contains("modes ysf-dmr (lag 5)"),
            "{}",
            named.describe()
        );
    }

    /// Opened with a subset of a mixed set's modes the loader keeps only
    /// those examples and renumbers their `mode` to the requested order.
    #[test]
    fn open_modes_filters_a_mixed_set_and_remaps_the_index() {
        use VocoderMode::{Codec2_3200, Dstar, YsfDmr};
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        write_fixture_root(&root, Dstar, 3, 5, Split::Train).unwrap();
        let cfg = CropCfg {
            mode: Dstar,
            crop_s: 0.5,
            gain_jitter_db: 0.0,
            ..CropCfg::default()
        };
        let mut examples = PipelineLoader::open(&root, Dstar, Split::Train, cfg, 3)
            .unwrap()
            .next_examples(9)
            .unwrap();
        // Pretend every third example came from Codec 2 3200 (the same
        // frame word, so one storage layout): stored mode index 1.
        for (i, ex) in examples.iter_mut().enumerate() {
            ex.mode = u8::from(i % 3 == 2);
        }
        let dir = tmp.path().join("shards").join("mixed");
        let index =
            write_shard_set_modes(&dir, &[Dstar, Codec2_3200], 0.5, Split::Train, &examples, 4)
                .unwrap();
        assert_eq!(index.counts_by_mode[&Split::Train][&Codec2_3200], 3);
        // Everything, as stored.
        let all = ShardLoader::open(&dir, Split::Train, 1).unwrap();
        assert_eq!(all.len(), 9);
        // Weighted: a zero share never draws a mode, a share draws only
        // its own examples, and equal shares split the draws.
        let only_c2 = ShardLoader::open(&dir, Split::Train, 1)
            .unwrap()
            .with_mode_weights(&[0.0, 1.0])
            .unwrap()
            .next_examples(12)
            .unwrap();
        assert!(only_c2.iter().all(|e| e.mode == 1), "share 0 for dstar");
        let mut even = ShardLoader::open(&dir, Split::Train, 1)
            .unwrap()
            .with_mode_weights(&[1.0, 1.0])
            .unwrap();
        assert!(
            even.describe()
                .contains("mode shares dstar 50 %, codec2-3200 50 %")
        );
        let drawn = even.next_examples(60).unwrap();
        let c2 = drawn.iter().filter(|e| e.mode == 1).count();
        assert!(
            (18..=42).contains(&c2),
            "half the draws, roughly: {c2} of 60"
        );
        assert!(
            ShardLoader::open(&dir, Split::Train, 1)
                .unwrap()
                .with_mode_weights(&[1.0])
                .is_err()
        );
        assert!(
            ShardLoader::open(&dir, Split::Train, 1)
                .unwrap()
                .with_mode_weights(&[0.0, 0.0])
                .is_err()
        );
        assert_eq!(all.modes(), &[Dstar, Codec2_3200]);
        assert_eq!(all.raw_example(2).unwrap().mode, 1);
        // The Codec 2 subset: three examples, each now mode 0.
        let c2 = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[Codec2_3200])).unwrap();
        assert_eq!(c2.len(), 3);
        assert_eq!(c2.split_len(), 9);
        assert_eq!(c2.modes(), &[Codec2_3200]);
        let got = c2.raw_example(2).unwrap();
        assert_eq!(got.mode, 0);
        assert_eq!(got.deg8, examples[2].deg8);
        assert!(
            c2.raw_example(0).is_err(),
            "a filtered-out example is refused"
        );
        let mut c2 = c2;
        let drawn = c2.next_examples(6).unwrap();
        assert!(drawn.iter().all(|e| e.mode == 0));
        let seen: std::collections::BTreeSet<Vec<u32>> = drawn
            .iter()
            .map(|e| e.deg8.iter().map(|v| v.to_bits()).collect())
            .collect();
        assert_eq!(seen.len(), 3, "two epochs over the three Codec 2 examples");
        let batch = c2.next_batch(3, Device::Cpu).unwrap();
        assert_eq!(batch.mode.size(), [3]);
        assert_eq!(batch.mode.int64_value(&[0]), 0);
        // Reversed order swaps the indices.
        let rev =
            ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[Codec2_3200, Dstar])).unwrap();
        assert_eq!(rev.len(), 9);
        assert_eq!(rev.raw_example(0).unwrap().mode, 1);
        assert_eq!(rev.raw_example(2).unwrap().mode, 0);
        // A mode the set lacks is an error; a duplicate is an error.
        let err = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[Dstar, YsfDmr]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("asks for ysf-dmr"), "{err}");
        assert!(ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[Dstar, Dstar])).is_err());
    }

    /// Two modes of different frame words through both loaders: the
    /// pipeline joins each mode's capture with its own lag and draws them
    /// round-robin; packed into one set they read back identically,
    /// mode index included, and a one-mode subset of the set is exactly
    /// that mode's draws.
    #[test]
    fn pipeline_and_shards_agree_over_two_modes() {
        use VocoderMode::{Codec2_1600, Dstar};
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        write_fixture_root(&root, Dstar, 3, 5, Split::Train).unwrap();
        write_fixture_root(&root, Codec2_1600, 3, 3, Split::Train).unwrap();
        let cfg = CropCfg {
            mode: Dstar,
            crop_s: 1.0,
            onset_share: 0.5,
            tail_share: 0.5,
            gain_jitter_db: 0.0,
        };
        let modes = [Dstar, Codec2_1600];
        let mut pipe =
            PipelineLoader::open_modes(&root, &modes, &[None], Split::Train, cfg, 42).unwrap();
        assert_eq!(pipe.len(), 6);
        assert_eq!(pipe.modes(), &modes);
        assert_eq!((pipe.lag_for(Dstar), pipe.lag_for(Codec2_1600)), (5, 3));
        assert!(
            pipe.describe().contains("codec2-1600 3 (lag 3)"),
            "{}",
            pipe.describe()
        );
        // Half a second is not whole 40 ms frames: refused for a mixed loader.
        let half = CropCfg { crop_s: 0.5, ..cfg };
        assert!(PipelineLoader::open_modes(&root, &modes, &[None], Split::Train, half, 1).is_err());
        let examples = pipe.next_examples(8).unwrap();
        let idx: Vec<u8> = examples.iter().map(|e| e.mode).collect();
        assert_eq!(
            idx,
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            "round-robin over the modes"
        );
        // Weighted: a zero share never draws that mode.
        let mut only_dstar =
            PipelineLoader::open_modes(&root, &modes, &[None], Split::Train, cfg, 42)
                .unwrap()
                .with_mode_weights(&[1.0, 0.0])
                .unwrap();
        assert!(only_dstar.describe().contains("mode shares dstar 100 %"));
        assert!(
            only_dstar
                .next_examples(8)
                .unwrap()
                .iter()
                .all(|e| e.mode == 0)
        );
        assert!(examples.iter().all(|e| e.deg8.len() == 8_000));
        // A Codec 2 1600 crop starts on a 320-sample boundary of its own
        // aligned utterance; a D-STAR one on a 160-sample boundary.
        let items = pipe.items().to_vec();
        for ex in &examples {
            let mode = modes[usize::from(ex.mode)];
            let speech8 = ex.mask_boundary / 2;
            let head = &ex.deg8[..speech8];
            let found = items.iter().filter(|i| i.mode == mode).any(|it| {
                let utt = load_utterance(&root, it, pipe.lag_for(mode)).unwrap();
                (0..=utt.deg8.len().saturating_sub(speech8))
                    .step_by(mode.frame_samples())
                    .any(|s| utt.deg8[s..s + speech8] == *head)
            });
            assert!(
                found,
                "{mode} crop is a frame-aligned window of its utterance"
            );
        }

        let dir = tmp.path().join("shards").join("two");
        write_shard_set_modes(&dir, &modes, 1.0, Split::Train, &examples, 3).unwrap();
        let mut both = ShardLoader::open_modes(&dir, Split::Train, 42, Some(&modes)).unwrap();
        assert_eq!(both.len(), 8);
        for (i, ex) in examples.iter().enumerate() {
            assert_eq!(&both.raw_example(i).unwrap(), ex, "example {i}");
        }
        let batch = both.next_batch(8, Device::Cpu).unwrap();
        assert_eq!(batch.mode.sum(tch::Kind::Int64).int64_value(&[]), 4);
        // The Codec 2 1600 subset is the odd draws, renumbered to mode 0.
        let c2 = ShardLoader::open_modes(&dir, Split::Train, 42, Some(&[Codec2_1600])).unwrap();
        assert_eq!(c2.len(), 4);
        for i in (1..8).step_by(2) {
            let got = c2.raw_example(i).unwrap();
            assert_eq!(got.mode, 0);
            assert_eq!(got.deg8, examples[i].deg8);
        }
    }
}
