// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `shard` (spec §4): pack fixed-length training examples from the
//! prepared and captured manifests into `shards/<name>/NNNN.bin`, one
//! [`ExampleLayout`] record after another, little-endian, with
//! `index.json` (the core crate's [`ShardIndex`]) describing the set and a
//! `files.json` sidecar listing each file's split and example count.
//!
//! Example kinds, drawn per example with the set's seeded PRNG:
//!
//! * **random** — a crop starting on a frame boundary anywhere in the
//!   utterance;
//! * **onset** (`onset_share`) — a crop starting at frame 0, where the
//!   vocoder's key-down behaviour lives; flag [`ExampleLayout::FLAG_ONSET`];
//! * **tail** (`tail_share`) — the utterance's last frames followed by
//!   0.5–1.5 s of receiver garbage in `deg8` (`unamblify::tail`: white
//!   noise, the last frame repeated, gated bursts or rumble, drawn with
//!   the set's PRNG — the same generator the on-the-fly loader uses) and
//!   digital silence in `clean16`, the mask boundary at the start of the
//!   garbage and zero bytes as its frames; flag
//!   [`ExampleLayout::FLAG_TAIL`]. Chip-decoded mute frames as a further
//!   garbage kind are a capture-time addition that comes later.
//!
//! Alignment: the decoded 8 kHz output lags its input by
//! `canary.json`'s `lag_samples`, which is undone exactly as the training
//! loader's `apply_lag` does: aligned `deg8[k] = decoded[k + lag]` (zero
//! past the end), paired with `clean16[2k]`, so a crop at frame `f`
//! takes `deg8` from `fs·f + lag` and `clean16` from `2·fs·f`, where `fs`
//! is the mode's `frame_samples()` (160 for the AMBE modes and Codec 2
//! 3200, 320 for Codec 2 1600). A set is refused without `canary.json` and
//! records the lag it applied in `index.json`.
//!
//! The example plan (utterance, kind, start) is built first, shuffled per
//! split, and only then rendered, so the files mix speakers and the whole
//! thing is a pure function of the manifests and the seed.
//!
//! A set may span several modes (`--modes dstar,codec2-3200`): each mode's
//! captured manifest (and siblings) is joined in turn, each utterance is
//! cropped in its own mode's frame word with its own canary lag, and every
//! example carries its mode's index in the flags byte. The storage layout
//! is [`ExampleLayout::for_modes`] (the audio fields must agree — a crop
//! that is not whole frames of every mode is refused — and the
//! channel-frame field is the largest mode's, zero-padded for the rest).
//! With `balance` (default on) every mode contributes the same number of
//! examples per split: each is capped at the smallest mode's count and
//! the surplus dropped by a seeded draw, recorded in `index.json` as
//! `available_by_mode` (before) and `counts_by_mode` (after).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub use unamblify::ShardFile;
use unamblify::aug::{AugKind, CaptureAug};
use unamblify::tail::fill_tail;
use unamblify::{
    CaptureRow, ExampleLayout, FLAGS_BYTES, ShardIndex, SourceSha256s, Split, UtteranceRow,
    VocoderMode, speaker_id,
};
use unamblify_audio::read;

use crate::canary;
use crate::util::{Rng, read_file, read_json, read_jsonl, sha256_file, write_json_atomic};
use crate::{CaptureDir, DataError, DataRoot, Result};

/// One entry of `shard --kinds`: a capture kind, optionally scoped to
/// the single mode it applies to.
///
/// `base` and `drops` apply to every mode in the set; `dstar+perens`
/// applies to D-STAR alone. Scoping exists because a sibling need not
/// exist for every mode — the software D-STAR recode has no `ysf-dmr`
/// counterpart — and an unscoped kind whose manifest is missing is still
/// an error, so a typo fails loudly instead of quietly building a
/// smaller set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindSel {
    /// The mode this applies to, or every mode in the set.
    pub mode: Option<VocoderMode>,
    /// The capture kind: `None` is the base capture.
    pub kind: Option<AugKind>,
}

impl KindSel {
    /// An unscoped kind: every mode in the set.
    #[must_use]
    pub const fn every(kind: Option<AugKind>) -> Self {
        Self { mode: None, kind }
    }

    /// A kind scoped to one mode.
    #[must_use]
    pub const fn scoped(mode: VocoderMode, kind: AugKind) -> Self {
        Self {
            mode: Some(mode),
            kind: Some(kind),
        }
    }

    /// Whether this entry contributes a capture set for `mode`.
    #[must_use]
    pub fn applies_to(self, mode: VocoderMode) -> bool {
        self.mode.is_none_or(|m| m == mode)
    }

    /// The word `index.json` records: `base`, `drops`, `dstar+perens`.
    #[must_use]
    pub fn word(self) -> String {
        match (self.mode, self.kind) {
            (None, None) => "base".to_owned(),
            (None, Some(k)) => k.to_string(),
            (Some(m), None) => format!("{}+base", m.as_str()),
            (Some(m), Some(k)) => format!("{}+{}", m.as_str(), k.as_str()),
        }
    }
}

/// Options of one shard build.
#[derive(Debug, Clone)]
pub struct ShardOptions {
    /// `shards/<name>/`.
    pub name: String,
    /// Modes of the degraded input, in the order the set indexes them
    /// (`index.json`'s `modes`; an example's mode bits index this list).
    pub modes: Vec<VocoderMode>,
    /// Crop length, seconds (rounded down to whole frames).
    pub crop_s: f32,
    /// Fraction of onset examples.
    pub onset_share: f32,
    /// Fraction of tail examples.
    pub tail_share: f32,
    /// PRNG seed.
    pub seed: u64,
    /// Examples per `NNNN.bin`.
    pub examples_per_file: usize,
    /// Only these splits (empty = all).
    pub splits: Vec<Split>,
    /// The capture sets to draw from. Default: base only.
    ///
    /// A kind may name the single mode it applies to, so a sibling that
    /// exists for one mode only (`dstar+perens`) can be drawn without
    /// demanding the same sibling of every other mode in the set.
    pub kinds: Vec<KindSel>,
    /// With more than one mode, draw equally from every mode per split:
    /// each mode is capped at the smallest mode's example count and the
    /// surplus dropped by a seeded draw. Default on; no-op with one mode.
    pub balance: bool,
    /// Keep at most this many joined utterances per capture set (a seeded
    /// draw, then key order), for a bounded trial set. Default: all.
    pub max_utterances: Option<usize>,
    /// With `max_utterances`: reserve this fraction of each capture set's
    /// draw for augmented twins. A uniform draw takes twins in proportion
    /// to the capture, which starves them where the capture is huge: 42 k
    /// twins are a fifth of the YSF set and a fiftieth of a Codec 2 one.
    /// `None` keeps the uniform draw, and every set built before this
    /// option existed.
    pub twin_share: Option<f32>,
    /// Draw a `drops` row only if it was generated at this frame-loss rate
    /// or above (`CaptureAug::rate_ppm`). Rows record their own
    /// parameters, so one sibling can hold a light pass and a heavy one;
    /// this picks the heavy one. Every other kind is untouched. `None`
    /// draws every row.
    pub min_drop_rate: Option<f32>,
    /// Draw only utterances of these corpora (the prepared row's `corpus`;
    /// a twin carries its parent's). Empty draws them all. What a model
    /// *regresses onto* has to be good: Common Voice recordings score 3.15
    /// on the judge a studio corpus scores 4.1 on, and a restorer
    /// fine-tuned on studio targets alone gained 0.35 MOS in forty minutes
    /// (experiment log #34).
    pub corpora: Vec<String>,
}

impl ShardOptions {
    /// The spec's example: 2 s crops, 34 % onset, 15 % tail, seed 1, 1024
    /// examples per file, the base capture only, one mode.
    #[must_use]
    pub fn new(name: impl Into<String>, mode: VocoderMode) -> Self {
        Self::for_modes(name, vec![mode])
    }

    /// [`ShardOptions::new`] over several modes.
    #[must_use]
    pub fn for_modes(name: impl Into<String>, modes: Vec<VocoderMode>) -> Self {
        Self {
            name: name.into(),
            modes,
            crop_s: 2.0,
            onset_share: 0.34,
            tail_share: 0.15,
            seed: 1,
            examples_per_file: 1024,
            splits: Vec::new(),
            kinds: vec![KindSel::every(None)],
            balance: true,
            max_utterances: None,
            twin_share: None,
            min_drop_rate: None,
            corpora: Vec::new(),
        }
    }

    /// The first mode: what `index.json`'s `mode` names.
    #[must_use]
    pub fn primary_mode(&self) -> VocoderMode {
        self.modes.first().copied().unwrap_or(VocoderMode::Dstar)
    }

    /// The kinds as the words `index.json` records (`base`, `drops`,
    /// `dstar+perens`, …).
    #[must_use]
    pub fn kind_words(&self) -> Vec<String> {
        self.kinds.iter().map(|k| k.word()).collect()
    }
}

/// Tail padding range, frames (0.5–1.5 s).
pub const TAIL_PAD_FRAMES: (usize, usize) = (25, 75);

/// What a build produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSummary {
    /// Examples per split.
    pub counts: BTreeMap<Split, u64>,
    /// Examples per split and mode, after balancing.
    pub counts_by_mode: ModeCounts,
    /// Examples per split and mode before balancing.
    pub available_by_mode: ModeCounts,
    /// Files written.
    pub files: u64,
    /// Utterances used (an utterance captured in two modes counts twice).
    pub utterances: u64,
    /// Utterances shorter than one crop, skipped.
    pub too_short: u64,
    /// Captured rows with no prepared row (or missing files), skipped.
    pub unjoined: u64,
}

/// One decoded example.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// 16 kHz clean target.
    pub clean16: Vec<f32>,
    /// 8 kHz degraded input.
    pub deg8: Vec<f32>,
    /// Channel frames: the storage layout's field, of which the example's
    /// own mode fills the first `frames × frame_bytes` bytes.
    pub frames: Vec<u8>,
    /// Flag bits (onset, tail, and the mode index above them).
    pub flags: u8,
    /// Samples of `clean16` at or after this index are padding (mask 0).
    pub mask_boundary: u32,
    /// `unamblify::speaker_id` of the source speaker.
    pub speaker_id: u32,
    /// Per-frame erasure mask, `layout.erasure_bytes` long (empty in a set
    /// built without it): 1 where the frame's audio is the decoder's
    /// concealment of a lost channel frame, 0 where the frame arrived.
    /// In the example's own frame word and lag-aligned like `deg8`, so
    /// entry `j` speaks for `deg8[j * frame_samples ..]`.
    pub erasure: Vec<u8>,
}

impl Example {
    /// Onset flag set.
    #[must_use]
    pub fn is_onset(&self) -> bool {
        self.flags & ExampleLayout::FLAG_ONSET != 0
    }

    /// Tail flag set.
    #[must_use]
    pub fn is_tail(&self) -> bool {
        self.flags & ExampleLayout::FLAG_TAIL != 0
    }

    /// The example's mode as an index into the set's `modes`.
    #[must_use]
    pub fn mode_index(&self) -> u8 {
        ExampleLayout::mode_index(self.flags)
    }

    /// Serialise per the layout.
    pub fn encode(&self, layout: &ExampleLayout) -> Result<Vec<u8>> {
        if self.clean16.len() != layout.clean16_samples
            || self.deg8.len() != layout.deg8_samples
            || self.frames.len() != layout.frames * layout.frame_bytes
            || self.erasure.len() != layout.erasure_bytes
        {
            return Err(DataError::Invalid(format!(
                "example does not match the layout: clean16 {} deg8 {} frames {} bytes",
                self.clean16.len(),
                self.deg8.len(),
                self.frames.len()
            )));
        }
        let mut out = Vec::with_capacity(layout.example_bytes());
        for v in &self.clean16 {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.deg8 {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.frames);
        out.push(self.flags);
        out.extend_from_slice(&self.mask_boundary.to_le_bytes());
        out.extend_from_slice(&self.speaker_id.to_le_bytes());
        out.extend_from_slice(&self.erasure);
        Ok(out)
    }

    /// Parse per the layout.
    pub fn decode(bytes: &[u8], layout: &ExampleLayout) -> Result<Self> {
        if bytes.len() != layout.example_bytes() {
            return Err(DataError::Invalid(format!(
                "example is {} bytes, layout says {}",
                bytes.len(),
                layout.example_bytes()
            )));
        }
        if layout.flags_bytes != FLAGS_BYTES {
            return Err(DataError::Invalid(format!(
                "layout flags_bytes {} != {FLAGS_BYTES}",
                layout.flags_bytes
            )));
        }
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let mut at = 0;
        let clean16 = f32s(&bytes[at..at + layout.clean16_samples * 4]);
        at += layout.clean16_samples * 4;
        let deg8 = f32s(&bytes[at..at + layout.deg8_samples * 4]);
        at += layout.deg8_samples * 4;
        let frames = bytes[at..at + layout.frames * layout.frame_bytes].to_vec();
        at += layout.frames * layout.frame_bytes;
        let flags = bytes[at];
        let mask_boundary =
            u32::from_le_bytes([bytes[at + 1], bytes[at + 2], bytes[at + 3], bytes[at + 4]]);
        let speaker_id =
            u32::from_le_bytes([bytes[at + 5], bytes[at + 6], bytes[at + 7], bytes[at + 8]]);
        let erasure = bytes[at + FLAGS_BYTES..at + FLAGS_BYTES + layout.erasure_bytes].to_vec();
        Ok(Self {
            clean16,
            deg8,
            frames,
            flags,
            mask_boundary,
            speaker_id,
            erasure,
        })
    }
}

/// Kind of a planned example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Random,
    Onset,
    Tail,
}

/// One planned example: which utterance, what kind, where.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Planned {
    utt: usize,
    kind: Kind,
    /// First frame of the crop in the degraded / frames domain (frames of
    /// the utterance's own mode).
    f0: usize,
    /// Tail examples: frames of garbage appended.
    pad_frames: usize,
    /// Seed of the example's own generator (the garbage tail).
    seed: u64,
}

/// One joined utterance.
#[derive(Debug, Clone)]
struct Item {
    key: String,
    /// Whose prepared 16 kHz file is the clean target: the parent's for
    /// an augmented twin, the key's own otherwise.
    target_key: String,
    /// Which mode the degraded side was captured in.
    mode: VocoderMode,
    /// Its index in the set's `modes`.
    mode_idx: u8,
    /// Which capture set the degraded side comes from.
    kind: Option<AugKind>,
    split: Split,
    frames: usize,
    speaker_id: u32,
    /// Channel frames lost on the way and concealed by the decoder, in
    /// order: a `drops` sibling's recorded positions, empty otherwise. A
    /// `ber` sibling's positions are not erasures — a receiver is told a
    /// frame was lost, never that one of its bits was wrong.
    erased: Vec<u32>,
    /// Samples the decoded side lags its input by, in *this* capture set.
    /// Not the mode's: a recode sibling is another implementation of the
    /// codec with another delay (the chip's D-STAR decode lags by 326
    /// samples, the software vocoder's by 216), and cutting it at the
    /// mode's lag trains on pairs 14 ms apart.
    lag: i32,
}

/// Per-mode facts a build needs: the mode's own layout and its lag.
#[derive(Debug, Clone)]
struct ModeInfo {
    layout: ExampleLayout,
    lag: i32,
}

/// Examples per split and mode.
pub type ModeCounts = BTreeMap<Split, BTreeMap<VocoderMode, u64>>;

/// What the join of every capture set produced.
struct Joined {
    items: Vec<Item>,
    unjoined: u64,
    /// The first mode's sibling manifests by kind.
    siblings: BTreeMap<String, String>,
    /// Every capture set's manifest by directory name.
    by_set: BTreeMap<String, String>,
    /// The lag undone per capture set, by directory name.
    set_lags: BTreeMap<String, i32>,
}

/// Check the option's modes and work out the storage layout.
/// Whether the set draws from a `drops` sibling, and so carries the
/// per-frame erasure mask. A set without one stores no mask at all: the
/// field is appended last, so its examples stay byte-identical to those of
/// a set built before the mask existed.
fn has_erasures(opts: &ShardOptions) -> bool {
    opts.kinds.iter().any(|k| k.kind == Some(AugKind::Drops))
}

fn storage_layout(opts: &ShardOptions) -> Result<ExampleLayout> {
    if opts.modes.is_empty() {
        return Err(DataError::Invalid(
            "at least one mode is needed (--mode M or --modes A,B)".to_owned(),
        ));
    }
    if opts.modes.len() > ExampleLayout::MAX_MODES {
        return Err(DataError::Invalid(format!(
            "{} modes; a set indexes at most {}",
            opts.modes.len(),
            ExampleLayout::MAX_MODES
        )));
    }
    for (i, m) in opts.modes.iter().enumerate() {
        if opts.modes[..i].contains(m) {
            return Err(DataError::Invalid(format!("mode {m} listed twice")));
        }
    }
    let layout = ExampleLayout::for_modes(&opts.modes, opts.crop_s)
        .map_err(|e| DataError::Invalid(e.to_string()))?
        .with_erasure(has_erasures(opts));
    if layout.frames == 0 {
        return Err(DataError::Invalid(
            "crop_s must be at least one frame".to_owned(),
        ));
    }
    if opts.kinds.is_empty() {
        return Err(DataError::Invalid(
            "at least one capture kind is needed (base, drops, ber, or a \
             mode-scoped kind such as dstar+perens)"
                .to_owned(),
        ));
    }
    Ok(layout)
}

/// Each mode's own layout and canary lag; a mode without a canary is
/// refused, since its alignment would be unknown.
fn mode_infos(root: &DataRoot, opts: &ShardOptions) -> Result<BTreeMap<VocoderMode, ModeInfo>> {
    let mut infos: BTreeMap<VocoderMode, ModeInfo> = BTreeMap::new();
    for &mode in &opts.modes {
        let lag = canary::load(&root.canary_json(mode))?
            .ok_or_else(|| {
                DataError::Invalid(format!(
                    "{}: no canary.json, so the capture lag is unknown; a set built without it would be \
                     misaligned (run a capture for {mode} first)",
                    root.canary_json(mode).display()
                ))
            })?
            .lag_samples;
        infos.insert(
            mode,
            ModeInfo {
                layout: ExampleLayout::for_crop(mode, opts.crop_s),
                lag,
            },
        );
    }
    Ok(infos)
}

/// Join every capture set of every mode with the prepared manifest.
fn join_all(root: &DataRoot, opts: &ShardOptions, prepared: &[UtteranceRow]) -> Result<Joined> {
    let mut items = Vec::new();
    let mut unjoined = 0u64;
    let mut siblings = BTreeMap::new();
    let mut by_set = BTreeMap::new();
    let mut set_lags = BTreeMap::new();
    for (mi, &mode) in opts.modes.iter().enumerate() {
        let mode_idx = u8::try_from(mi).unwrap_or(u8::MAX);
        for (ki, &sel) in opts.kinds.iter().enumerate() {
            // A kind scoped to another mode contributes nothing here.
            if !sel.applies_to(mode) {
                continue;
            }
            let kind = sel.kind;
            let dir = root.capture_dir(mode, kind);
            if !dir.manifest().is_file() {
                return Err(DataError::Invalid(format!(
                    "{}: no manifest for capture set {} (run `unamblify {}` first)",
                    dir.manifest().display(),
                    dir.name(),
                    if kind.is_some() { "augment" } else { "capture" }
                )));
            }
            let captured: Vec<CaptureRow> = read_jsonl(&dir.manifest())?;
            let lag = set_lag(root, &dir)?;
            set_lags.insert(dir.name(), lag);
            let (found, missing) = join(root, opts, &dir, mode_idx, lag, prepared, &captured)?;
            let found = limit_utterances(found, opts, u64::from(mode_idx) * 8 + ki as u64);
            log::info!(
                "shard {}: {} utterances from {} ({missing} unjoined)",
                opts.name,
                found.len(),
                dir.name()
            );
            items.extend(found);
            unjoined += missing;
            let sha = sha256_file(&dir.manifest())?;
            if let Some(k) = kind {
                // Keyed by the kind's word, scoped or not: a sibling that
                // exists for one mode only is recorded under that mode's
                // spelling, so `index.json` names exactly what was drawn.
                let key = if sel.mode.is_some() {
                    sel.word()
                } else {
                    k.to_string()
                };
                siblings.entry(key).or_insert_with(|| sha.clone());
            }
            by_set.insert(dir.name(), sha);
        }
    }
    Ok(Joined {
        items,
        unjoined,
        siblings,
        by_set,
        set_lags,
    })
}

/// The lag of one capture set: its own `canary.json` where it has one — a
/// recode sibling measures its own codec's delay, and a decode-only
/// sibling's record agrees with the base capture's — else the base
/// capture's. Neither is an error: the alignment would be a guess.
fn set_lag(root: &DataRoot, dir: &CaptureDir) -> Result<i32> {
    for path in [dir.canary_json(), root.canary_json(dir.mode())] {
        if let Some(rec) = canary::load(&path)? {
            return Ok(rec.lag_samples);
        }
    }
    Err(DataError::Invalid(format!(
        "{}: no canary.json here or in the base capture, so the lag of {} is unknown",
        dir.canary_json().display(),
        dir.name()
    )))
}

/// `opts.max_utterances`: a seeded draw of that many items (salted per
/// capture set), back in key order; everything when unset or smaller.
fn limit_utterances(mut items: Vec<Item>, opts: &ShardOptions, salt: u64) -> Vec<Item> {
    let Some(max) = opts.max_utterances else {
        return items;
    };
    if items.len() <= max {
        return items;
    }
    let mut r = Rng::new(opts.seed ^ 0x4D41_5855_5454_5300).fork(salt);
    let Some(share) = opts.twin_share else {
        // The uniform draw, unchanged: sets built before `twin_share`
        // existed must come out the same.
        r.shuffle(&mut items);
        items.truncate(max);
        items.sort_by(|a, b| a.key.cmp(&b.key));
        return items;
    };
    // Stratified: twins and base are drawn separately. Whichever side
    // runs short hands its unused places to the other, so the set is
    // still `max` utterances whenever the capture holds that many.
    let (mut twins, mut base): (Vec<Item>, Vec<Item>) =
        items.into_iter().partition(|i| i.key != i.target_key);
    r.shuffle(&mut twins);
    r.shuffle(&mut base);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let want_twins = ((max as f64) * f64::from(share.clamp(0.0, 1.0))).round() as usize;
    let n_twins = want_twins
        .min(twins.len())
        .max(max.saturating_sub(base.len()));
    let n_twins = n_twins.min(twins.len());
    let n_base = (max - n_twins).min(base.len());
    twins.truncate(n_twins);
    base.truncate(n_base);
    twins.append(&mut base);
    twins.sort_by(|a, b| a.key.cmp(&b.key));
    twins
}

/// Build the set. Deterministic given the manifests and `opts.seed`.
pub fn build(root: &DataRoot, opts: &ShardOptions) -> Result<ShardSummary> {
    let layout = storage_layout(opts)?;
    let prepared_path = root.prepared_manifest();
    let prepared: Vec<UtteranceRow> = read_jsonl(&prepared_path)?;
    let infos = mode_infos(root, opts)?;
    let Joined {
        items,
        unjoined,
        siblings,
        by_set,
        set_lags,
    } = join_all(root, opts, &prepared)?;
    let (mut plans, used, too_short) = plan_examples(opts, &infos, &items);
    let mut rng = Rng::new(opts.seed ^ 0x4241_4C41_4E43_4500);
    let (available_by_mode, counts_by_mode) = balance_plans(opts, &mut plans, &items, &mut rng);
    let (files, counts) = render_files(root, opts, &layout, &infos, &items, &plans)?;
    let dir = root.shards(&opts.name);
    let captured_path = root.captured_manifest(opts.primary_mode());
    let index = ShardIndex {
        name: opts.name.clone(),
        mode: opts.primary_mode(),
        modes: opts.modes.clone(),
        erasure: has_erasures(opts),
        crop_s: opts.crop_s,
        seed: opts.seed,
        lag_samples: infos.get(&opts.primary_mode()).map(|i| i.lag),
        lags: infos.iter().map(|(&m, i)| (m, i.lag)).collect(),
        set_lags,
        counts: counts.clone(),
        counts_by_mode: counts_by_mode.clone(),
        available_by_mode: available_by_mode.clone(),
        balance: opts.balance && opts.modes.len() > 1,
        max_utterances: opts.max_utterances.map(|n| n as u64),
        twin_share: opts.max_utterances.and(opts.twin_share),
        min_drop_rate: opts.min_drop_rate,
        corpora: opts.corpora.clone(),
        onset_share: opts.onset_share,
        tail_share: opts.tail_share,
        kinds: opts.kind_words(),
        source_sha256s: SourceSha256s {
            prepared: sha256_file(&prepared_path)?,
            captured: if captured_path.is_file() {
                sha256_file(&captured_path)?
            } else {
                String::new()
            },
            siblings,
            by_set,
        },
        example_layout: layout,
        mode_layouts: infos.iter().map(|(&m, i)| (m, i.layout.clone())).collect(),
    };
    write_json_atomic(&dir.join("index.json"), &index)?;
    write_json_atomic(&dir.join("files.json"), &files)?;
    log::info!(
        "shard {}: {} examples in {} files from {} utterances ({} too short, {} unjoined)",
        opts.name,
        counts.values().sum::<u64>(),
        files.len(),
        used,
        too_short,
        unjoined
    );
    Ok(ShardSummary {
        counts,
        counts_by_mode,
        available_by_mode,
        files: files.len() as u64,
        utterances: used,
        too_short,
        unjoined,
    })
}

/// Join the captured rows with the prepared manifest, in key order,
/// keeping only utterances whose three files exist. Returns the items and
/// the count of captured rows left out. An augmented twin's clean target
/// is its parent's file; a twin whose parent row or file is missing is an
/// error, not a skip — the set would silently train against nothing.
fn join(
    root: &DataRoot,
    opts: &ShardOptions,
    dir: &CaptureDir,
    mode_idx: u8,
    lag: i32,
    prepared: &[UtteranceRow],
    captured: &[CaptureRow],
) -> Result<(Vec<Item>, u64)> {
    let by_key: std::collections::HashMap<&str, &UtteranceRow> =
        prepared.iter().map(|r| (r.key.as_str(), r)).collect();
    let mut items: Vec<Item> = Vec::new();
    let mut unjoined = 0u64;
    let mut seen = std::collections::HashSet::new();
    let mut captured_sorted: Vec<&CaptureRow> = captured.iter().collect();
    captured_sorted.sort_by(|a, b| a.key.cmp(&b.key));
    for c in captured_sorted {
        if !seen.insert(c.key.as_str()) {
            continue;
        }
        let Some(row) = by_key.get(c.key.as_str()) else {
            unjoined += 1;
            continue;
        };
        if !opts.splits.is_empty() && !opts.splits.contains(&row.split) {
            continue;
        }
        if !opts.corpora.is_empty() && !opts.corpora.contains(&row.corpus) {
            continue;
        }
        if below_min_drop_rate(c, opts.min_drop_rate) {
            continue;
        }
        let target_key = row.target_key();
        if let Some(parent) = &row.parent {
            let target = root.prepared_16k_decoded(parent);
            if !by_key.contains_key(parent.as_str()) || !target.exists() {
                return Err(DataError::Invalid(format!(
                    "{}: twin of {parent}, whose prepared row or {} is missing; the twin's clean                      target is the parent's file (re-run prepare)",
                    c.key,
                    target.display()
                )));
            }
        } else if !root.prepared_16k_decoded(&c.key).exists() {
            unjoined += 1;
            continue;
        }
        if !dir.decoded(&c.key).exists() || !dir.ambe(&c.key).exists() {
            unjoined += 1;
            continue;
        }
        items.push(Item {
            key: c.key.clone(),
            target_key: target_key.to_owned(),
            mode: dir.mode(),
            mode_idx,
            kind: dir.kind(),
            split: row.split,
            frames: c.frames as usize,
            speaker_id: speaker_id(&row.speaker),
            erased: c
                .aug
                .as_ref()
                .filter(|a| a.kind == AugKind::Drops)
                .map(|a| {
                    let mut p = a.positions.clone();
                    p.sort_unstable();
                    p
                })
                .unwrap_or_default(),
            lag,
        });
    }

    Ok((items, unjoined))
}

/// `opts.min_drop_rate`: whether a `drops` row was generated below the
/// floor. Rows of any other kind, and base rows, never are.
fn below_min_drop_rate(row: &CaptureRow, min: Option<f32>) -> bool {
    let Some(min) = min else {
        return false;
    };
    row.aug
        .as_ref()
        .is_some_and(|a| a.kind == AugKind::Drops && a.rate_ppm < CaptureAug::ppm(min))
}

/// Draw the examples per split with the seeded PRNG, in each utterance's
/// own frame word. Returns the plans (unshuffled), utterances used,
/// utterances too short.
fn plan_examples(
    opts: &ShardOptions,
    infos: &BTreeMap<VocoderMode, ModeInfo>,
    items: &[Item],
) -> (BTreeMap<Split, Vec<Planned>>, u64, u64) {
    let mut rng = Rng::new(opts.seed);
    let mut plans: BTreeMap<Split, Vec<Planned>> = BTreeMap::new();
    let mut too_short = 0u64;
    let mut used = 0u64;
    for (i, it) in items.iter().enumerate() {
        let Some(info) = infos.get(&it.mode) else {
            continue;
        };
        let layout = &info.layout;
        if it.frames < layout.frames {
            too_short += 1;
            continue;
        }
        used += 1;
        let n = (it.frames / layout.frames).max(1);
        for _ in 0..n {
            let r = rng.next_f64();
            let kind = if r < f64::from(opts.onset_share) {
                Kind::Onset
            } else if r < f64::from(opts.onset_share + opts.tail_share) {
                Kind::Tail
            } else {
                Kind::Random
            };
            let seed = rng.next_u64();
            let planned = match kind {
                Kind::Onset => Planned {
                    utt: i,
                    kind,
                    f0: 0,
                    pad_frames: 0,
                    seed,
                },
                Kind::Random => Planned {
                    utt: i,
                    kind,
                    f0: rng.range(0, it.frames - layout.frames),
                    pad_frames: 0,
                    seed,
                },
                Kind::Tail => {
                    let pad = rng
                        .range(TAIL_PAD_FRAMES.0, TAIL_PAD_FRAMES.1)
                        .min(layout.frames - 1);
                    let speech = layout.frames - pad;
                    Planned {
                        utt: i,
                        kind,
                        f0: it.frames - speech,
                        pad_frames: pad,
                        seed,
                    }
                }
            };
            plans.entry(it.split).or_default().push(planned);
        }
    }
    for v in plans.values_mut() {
        rng.shuffle(v);
    }

    (plans, used, too_short)
}

/// Per-mode example counts of each split's plans.
fn count_by_mode(
    opts: &ShardOptions,
    plans: &BTreeMap<Split, Vec<Planned>>,
    items: &[Item],
) -> ModeCounts {
    let mut out = BTreeMap::new();
    for (&split, list) in plans {
        let mut per: BTreeMap<VocoderMode, u64> = BTreeMap::new();
        for p in list {
            *per.entry(items[p.utt].mode).or_insert(0) += 1;
        }
        // Every mode of the set appears, with 0 when it has nothing here.
        for &m in &opts.modes {
            per.entry(m).or_insert(0);
        }
        out.insert(split, per);
    }
    out
}

/// With `balance` on and more than one mode, cap every mode at the
/// smallest non-empty mode's count per split, dropping the surplus by a
/// seeded draw, and reshuffle the split. Returns (available, kept) per
/// split and mode. One mode, or `balance` off, changes nothing.
fn balance_plans(
    opts: &ShardOptions,
    plans: &mut BTreeMap<Split, Vec<Planned>>,
    items: &[Item],
    rng: &mut Rng,
) -> (ModeCounts, ModeCounts) {
    let available = count_by_mode(opts, plans, items);
    if !opts.balance || opts.modes.len() < 2 {
        return (available.clone(), available);
    }
    for (split, list) in plans.iter_mut() {
        let per = &available[split];
        let Some(cap) = per.values().filter(|&&n| n > 0).min().copied() else {
            continue;
        };
        for (&m, &n) in per {
            if n == 0 {
                log::warn!("shard {}: {split} has no {m} examples", opts.name);
            }
        }
        let mut by_mode: BTreeMap<VocoderMode, Vec<Planned>> = BTreeMap::new();
        for p in list.drain(..) {
            by_mode.entry(items[p.utt].mode).or_default().push(p);
        }
        for (m, v) in &mut by_mode {
            if v.len() as u64 > cap {
                rng.shuffle(v);
                v.truncate(usize::try_from(cap).unwrap_or(usize::MAX));
                log::info!(
                    "shard {}: {split} {m} capped at {cap} of {} examples",
                    opts.name,
                    per[m]
                );
            }
        }
        for v in by_mode.into_values() {
            list.extend(v);
        }
        rng.shuffle(list);
    }
    let kept = count_by_mode(opts, plans, items);
    (available, kept)
}

/// Remove every `*.bin` under `dir` so a rebuild under the same name
/// leaves no stale file for a loader to trip over.
fn clear_bins(dir: &Path) -> Result<()> {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(DataError::io(dir, e)),
    };
    let mut removed = 0usize;
    for entry in rd {
        let p = entry.map_err(|e| DataError::io(dir, e))?.path();
        if p.extension().is_some_and(|e| e == "bin") {
            std::fs::remove_file(&p).map_err(|e| DataError::io(&p, e))?;
            removed += 1;
        }
    }
    if removed > 0 {
        log::info!(
            "{}: removed {removed} shard file(s) from an earlier build",
            dir.display()
        );
    }
    Ok(())
}

/// Write the planned examples to `NNNN.bin` files.
fn render_files(
    root: &DataRoot,
    opts: &ShardOptions,
    layout: &ExampleLayout,
    infos: &BTreeMap<VocoderMode, ModeInfo>,
    items: &[Item],
    plans: &BTreeMap<Split, Vec<Planned>>,
) -> Result<(Vec<ShardFile>, BTreeMap<Split, u64>)> {
    let dir = root.shards(&opts.name);
    std::fs::create_dir_all(&dir).map_err(|e| DataError::io(&dir, e))?;
    clear_bins(&dir)?;
    let mut files: Vec<ShardFile> = Vec::new();
    let mut counts: BTreeMap<Split, u64> = BTreeMap::new();
    let mut cache: Cache = Cache::new(root, 8);
    let per_file = opts.examples_per_file.max(1);
    for (split, list) in plans {
        for chunk in list.chunks(per_file) {
            let name = format!("{:04}.bin", files.len());
            let path = dir.join(&name);
            let mut w = BufWriter::new(File::create(&path).map_err(|e| DataError::io(&path, e))?);
            for p in chunk {
                let it = &items[p.utt];
                let info = infos.get(&it.mode).ok_or_else(|| {
                    DataError::Invalid(format!("{}: mode {} is not in the set", it.key, it.mode))
                })?;
                let ex = render(&mut cache, it, p, layout, &info.layout, it.lag)?;
                w.write_all(&ex.encode(layout)?)
                    .map_err(|e| DataError::io(&path, e))?;
            }
            w.flush().map_err(|e| DataError::io(&path, e))?;
            files.push(ShardFile {
                file: name,
                split: *split,
                examples: chunk.len() as u64,
            });
            *counts.entry(*split).or_insert(0) += chunk.len() as u64;
        }
    }
    Ok((files, counts))
}

/// Key of a decoded utterance in the cache: mode, capture set, key.
type CacheKey = (VocoderMode, Option<AugKind>, String);

/// A small LRU of decoded utterances, since the shuffled plan revisits
/// each one a few times.
struct Cache {
    root: DataRoot,
    cap: usize,
    order: Vec<CacheKey>,
    map: std::collections::HashMap<CacheKey, std::rc::Rc<Loaded>>,
}

struct Loaded {
    clean16: Vec<f32>,
    deg8: Vec<f32>,
    ambe: Vec<u8>,
}

impl Cache {
    fn new(root: &DataRoot, cap: usize) -> Self {
        Self {
            root: root.clone(),
            cap,
            order: Vec::new(),
            map: std::collections::HashMap::new(),
        }
    }

    fn get(&mut self, it: &Item) -> Result<std::rc::Rc<Loaded>> {
        let id = (it.mode, it.kind, it.key.clone());
        if let Some(l) = self.map.get(&id) {
            return Ok(std::rc::Rc::clone(l));
        }
        let key = it.key.as_str();
        let dir = self.root.capture_dir(it.mode, it.kind);
        let clean = read(self.root.prepared_16k_decoded(&it.target_key))?;
        let deg = read(dir.decoded(key))?;
        let ambe = read_file(&dir.ambe(key))?;
        if clean.rate != 16_000 || deg.rate != 8_000 {
            return Err(DataError::Invalid(format!(
                "{key}: rates {} / {} are not 16000 / 8000",
                clean.rate, deg.rate
            )));
        }
        let l = std::rc::Rc::new(Loaded {
            clean16: clean.samples,
            deg8: deg.samples,
            ambe,
        });
        if self.order.len() >= self.cap
            && let Some(old) = self.order.first().cloned()
        {
            self.order.remove(0);
            self.map.remove(&old);
        }
        self.order.push(id.clone());
        self.map.insert(id, std::rc::Rc::clone(&l));
        Ok(l)
    }
}

/// Copy `src[start..start + n]` into a zero-filled vector, treating
/// indices outside `src` as zero.
fn window(src: &[f32], start: i64, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        let j = start + i64::try_from(i).unwrap_or(i64::MAX);
        if j >= 0
            && let Some(&v) = src.get(usize::try_from(j).unwrap_or(usize::MAX))
        {
            *o = v;
        }
    }
    out
}

/// Render one planned example: the crop in the utterance's own frame
/// word (`mode_layout`), stored in the set's `layout` (the channel-frame
/// field zero-padded to the storage size, the mode index in the flags).
fn render(
    cache: &mut Cache,
    it: &Item,
    p: &Planned,
    layout: &ExampleLayout,
    mode_layout: &ExampleLayout,
    lag: i32,
) -> Result<Example> {
    let l = cache.get(it)?;
    let bytes = it.mode.frame_bytes();
    let fs = it.mode.frame_samples();
    let speech_frames = mode_layout.frames - p.pad_frames;
    let d0 = p.f0 * fs;
    let deg_n = speech_frames * fs;
    // The loader's `apply_lag`: aligned deg8[k] = decoded[k + lag].
    let deg_start = i64::try_from(d0).unwrap_or(0) + i64::from(lag);
    let clean_start = 2 * i64::try_from(d0).unwrap_or(0);
    let clean_n = 2 * deg_n;

    let mut deg8 = window(&l.deg8, deg_start, deg_n);
    deg8.resize(layout.deg8_samples, 0.0);
    let mut clean16 = window(&l.clean16, clean_start, clean_n);
    clean16.resize(layout.clean16_samples, 0.0);
    let f_start = p.f0 * bytes;
    let f_end = (f_start + speech_frames * bytes).min(l.ambe.len());
    let mut frames = l.ambe.get(f_start..f_end).unwrap_or(&[]).to_vec();
    frames.resize(layout.frames * layout.frame_bytes, 0);
    let erasure = unamblify::channel::erasure_mask(
        &it.erased,
        p.f0,
        speech_frames,
        fs,
        lag,
        layout.erasure_bytes,
    );

    let mut flags = ExampleLayout::with_mode(0, it.mode_idx);
    if p.f0 == 0 {
        flags |= ExampleLayout::FLAG_ONSET;
    }
    let mask_boundary = if p.kind == Kind::Tail {
        flags |= ExampleLayout::FLAG_TAIL;
        let b = fill_tail(&mut deg8, &mut clean16, deg_n, &mut Rng::new(p.seed));
        u32::try_from(b).unwrap_or(u32::MAX)
    } else {
        u32::try_from(layout.clean16_samples).unwrap_or(u32::MAX)
    };
    Ok(Example {
        clean16,
        deg8,
        frames,
        flags,
        mask_boundary,
        speaker_id: it.speaker_id,
        erasure,
    })
}

/// A shard set opened for reading.
#[derive(Debug)]
pub struct ShardSet {
    /// The directory.
    pub dir: PathBuf,
    /// `index.json`.
    pub index: ShardIndex,
    /// `files.json`.
    pub files: Vec<ShardFile>,
}

impl ShardSet {
    /// Open `shards/<name>/`.
    pub fn open(dir: &Path) -> Result<Self> {
        let index: ShardIndex = read_json(&dir.join("index.json"))?;
        let files: Vec<ShardFile> = read_json(&dir.join("files.json"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            files,
        })
    }

    /// Total examples in `split` (or all).
    #[must_use]
    pub fn count(&self, split: Option<Split>) -> u64 {
        self.files
            .iter()
            .filter(|f| split.is_none_or(|s| s == f.split))
            .map(|f| f.examples)
            .sum()
    }

    /// The mode of a decoded example, from its mode bits.
    #[must_use]
    pub fn mode_of(&self, ex: &Example) -> Option<VocoderMode> {
        self.index.mode_at(ex.mode_index())
    }

    /// Read example `i` of file `file`.
    pub fn read(&self, file: usize, i: u64) -> Result<Example> {
        let f = self
            .files
            .get(file)
            .ok_or_else(|| DataError::Invalid(format!("no shard file {file}")))?;
        if i >= f.examples {
            return Err(DataError::Invalid(format!(
                "{}: example {i} of {}",
                f.file, f.examples
            )));
        }
        let path = self.dir.join(&f.file);
        let layout = &self.index.example_layout;
        let mut fh = File::open(&path).map_err(|e| DataError::io(&path, e))?;
        let n = layout.example_bytes();
        fh.seek(SeekFrom::Start(i * n as u64))
            .map_err(|e| DataError::io(&path, e))?;
        let mut buf = vec![0u8; n];
        fh.read_exact(&mut buf)
            .map_err(|e| DataError::io(&path, e))?;
        Example::decode(&buf, layout)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::testutil::sine;
    use crate::util::{JsonlWriter, now_rfc3339, write_jsonl};
    use unamblify::CanaryRecord;
    use unamblify::aug::AugKind;
    use unamblify_audio::{read_wav, write_flac_s16, write_wav_s16};

    /// Prepared + captured fixture: `n` utterances of 1.0 + 0.5·i s with
    /// the degraded side a delayed copy of the clean one (lag 42 at 8 kHz)
    /// and frames whose bytes encode their index.
    fn fixture(n: usize, mode: VocoderMode) -> (tempfile::TempDir, DataRoot) {
        fixture_with(n, &[mode], mode.frame_samples())
    }

    /// [`fixture`] captured in every mode of `modes`: utterance `i` is
    /// `(50 + 25 i) × unit` samples at 8 kHz (`unit` a multiple of every
    /// mode's frame word), so each mode has its own frame count of it.
    fn fixture_with(n: usize, modes: &[VocoderMode], unit: usize) -> (tempfile::TempDir, DataRoot) {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let mut pw = JsonlWriter::open(&root.prepared_manifest()).unwrap();
        let mut cws: Vec<JsonlWriter> = modes
            .iter()
            .map(|&m| JsonlWriter::open(&root.captured_manifest(m)).unwrap())
            .collect();
        for i in 0..n {
            let key = format!("vctk/p2{i:02}_001_mic2");
            let n8 = (50 + 25 * i) * unit;
            let clean8 = sine(150.0 + 40.0 * i as f32, 8_000, n8, 0.3);
            // 16 kHz "clean": sample-doubled 8 kHz so clean16[2k] == clean8[k].
            let mut clean16 = Vec::with_capacity(2 * n8);
            for &v in &clean8 {
                clean16.push(v);
                clean16.push(v);
            }
            let mut deg8 = vec![0.0f32; 42];
            deg8.extend_from_slice(&clean8[..n8 - 42]);
            let p16 = root.prepared_16k(&key);
            std::fs::create_dir_all(p16.parent().unwrap()).unwrap();
            write_wav_s16(&p16, &clean16, 16_000).unwrap();
            write_wav_s16(root.prepared_8k(&key), &clean8, 8_000).unwrap();
            pw.append(&UtteranceRow {
                key: key.clone(),
                corpus: "vctk".to_owned(),
                speaker: format!("p2{i:02}"),
                gender: None,
                split: if i % 3 == 0 { Split::Dev } else { Split::Train },
                duration_s: n8 as f64 / 8_000.0,
                src_rate: 48_000,
                src_path: String::new(),
                licence: "CC-BY-4.0".to_owned(),
                rms_dbfs_in: -20.0,
                gain_db: 0.0,
                trim_lead_s: 0.0,
                trim_tail_s: 0.0,
                sha256_16k: String::new(),
                sha256_8k: String::new(),
                prepared_at: now_rfc3339(),
                parent: None,
                aug: None,
            })
            .unwrap();
            for (&mode, cw) in modes.iter().zip(&mut cws) {
                let frames = n8 / mode.frame_samples();
                let cw_path = root.captured_wav(mode, &key);
                std::fs::create_dir_all(cw_path.parent().unwrap()).unwrap();
                write_wav_s16(&cw_path, &deg8, 8_000).unwrap();
                let bytes = mode.frame_bytes();
                let ambe: Vec<u8> = (0..frames)
                    .flat_map(|f| std::iter::repeat_n(f as u8, bytes))
                    .collect();
                std::fs::write(root.captured_ambe(mode, &key), &ambe).unwrap();
                cw.append(&CaptureRow {
                    key: key.clone(),
                    mode,
                    frames: frames as u32,
                    port: "sim:0".to_owned(),
                    prodid: "AMBE3000F".to_owned(),
                    version: "V".to_owned(),
                    encode_ms: 0,
                    decode_ms: 0,
                    roundtrip_ms: None,
                    sha256_ambe: String::new(),
                    sha256_wav: String::new(),
                    captured_at: now_rfc3339(),
                    attempts: 1,
                    warm_state: None,
                    aug: None,
                })
                .unwrap();
            }
        }
        for &mode in modes {
            write_json_atomic(
                &root.canary_json(mode),
                &CanaryRecord {
                    mode,
                    clip: DataRoot::CANARY_CLIP.to_owned(),
                    frames_sha256: String::new(),
                    frames_first_16: String::new(),
                    lag_samples: 42,
                    prodid: "AMBE3000F".to_owned(),
                    version: "V".to_owned(),
                    recorded_at: now_rfc3339(),
                    warm_up_frames: 20,
                },
            )
            .unwrap();
        }
        (dir, root)
    }

    #[test]
    fn example_encode_decode_round_trip() {
        let layout = ExampleLayout::for_crop(VocoderMode::YsfDmr, 0.1);
        assert_eq!(layout.frames, 5);
        let ex = Example {
            clean16: (0..layout.clean16_samples)
                .map(|i| i as f32 * 0.001)
                .collect(),
            deg8: (0..layout.deg8_samples)
                .map(|i| -(i as f32) * 0.002)
                .collect(),
            frames: (0..35u8).collect(),
            flags: ExampleLayout::FLAG_TAIL,
            mask_boundary: 1234,
            speaker_id: 0xdead_beef,
            erasure: Vec::new(),
        };
        let bytes = ex.encode(&layout).unwrap();
        assert_eq!(bytes.len(), layout.example_bytes());
        assert_eq!(Example::decode(&bytes, &layout).unwrap(), ex);
        assert!(Example::decode(&bytes[1..], &layout).is_err());
        let bad = Example {
            frames: vec![0; 3],
            ..ex
        };
        assert!(bad.encode(&layout).is_err());
    }

    #[test]
    fn build_is_deterministic_aligned_and_round_trips() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(6, mode);
        let mut opts = ShardOptions::new("t", mode);
        opts.crop_s = 1.0;
        opts.examples_per_file = 4;
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.utterances, 6);
        assert_eq!(s.too_short, 0);
        assert_eq!(s.unjoined, 0);
        // Utterance i has 50 + 25i frames → floor((50+25i)/50) examples:
        // 1,1,2,2,3,3 = 12.
        assert_eq!(s.counts.values().sum::<u64>(), 12);
        assert_eq!(s.counts[&Split::Dev], 1 + 2); // i = 0, 3
        assert_eq!(s.counts[&Split::Train], 9);

        let set = ShardSet::open(&root.shards("t")).unwrap();
        assert_eq!(set.index.mode, mode);
        assert_eq!(set.index.modes, vec![mode]);
        assert_eq!(set.index.lags, [(mode, 42)].into());
        assert_eq!(set.index.mode_layouts[&mode], set.index.example_layout);
        assert!(!set.index.balance, "one mode: nothing to balance");
        assert_eq!(set.index.counts_by_mode[&Split::Dev][&mode], 3);
        assert_eq!(set.index.available_by_mode, set.index.counts_by_mode);
        assert_eq!(
            set.index.source_sha256s.by_set["dstar"],
            set.index.source_sha256s.captured
        );
        assert_eq!(
            set.index.lag_samples,
            Some(42),
            "the lag applied is recorded"
        );
        assert_eq!(set.index.example_layout.frames, 50);
        assert_eq!(set.index.counts, s.counts);
        assert_eq!(set.count(None), 12);
        assert_eq!(set.count(Some(Split::Dev)), 3);
        assert_eq!(set.files.len(), 1 + 3); // dev: 3 in one file; train: 4+4+1
        assert!(set.files.iter().all(|f| f.examples <= 4));
        assert_eq!(
            set.index.source_sha256s.prepared,
            sha256_file(&root.prepared_manifest()).unwrap()
        );

        let mut onsets = 0;
        let mut tails = 0;
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                assert_eq!(ex.clean16.len(), 16_000);
                assert_eq!(ex.deg8.len(), 8_000);
                assert_eq!(ex.frames.len(), 450);
                assert_eq!(ex.mode_index(), 0);
                assert_eq!(set.mode_of(&ex), Some(mode));
                if ex.is_onset() {
                    onsets += 1;
                    assert_eq!(ex.frames[0], 0);
                }
                if ex.is_tail() {
                    tails += 1;
                    let b = ex.mask_boundary as usize;
                    // pad ∈ [25, 49] frames of 50 → 1..=25 speech frames.
                    assert!((320..=8_000).contains(&b), "{b}");
                    assert!(ex.clean16[b..].iter().all(|&v| v == 0.0));
                    assert!(
                        ex.deg8[b / 2..].iter().any(|&v| v != 0.0),
                        "the tail's input is receiver garbage, not silence"
                    );
                    assert!(ex.frames[b / 2 / 160 * 9..].iter().all(|&v| v == 0));
                } else {
                    assert_eq!(ex.mask_boundary, 16_000);
                }
                // Alignment: the lag is undone, so deg8[k] equals
                // clean16[2k] within s16 quantisation over the speech —
                // from k = 0 for onsets too. Only the utterance's last
                // `lag` samples (never decoded) may differ.
                let speech = ex.mask_boundary as usize / 2;
                let mismatches = (0..speech)
                    .filter(|&k| (ex.deg8[k] - ex.clean16[2 * k]).abs() >= 2.0 / 32_767.0)
                    .count();
                assert!(mismatches <= 42, "{mismatches} misaligned samples");
                if ex.is_onset() {
                    assert!(
                        ex.deg8[..160].iter().any(|&v| v != 0.0),
                        "no zero delay kept"
                    );
                }
                // The frame bytes name their source frame index; the crop's
                // first frame is f0, so byte[9·j] == f0 + j over the speech.
                let f0 = ex.frames[0] as usize;
                for j in 0..speech / 160 {
                    assert_eq!(ex.frames[9 * j] as usize, f0 + j, "j={j}");
                }
            }
        }
        assert!(onsets > 0 && tails > 0, "{onsets} onsets, {tails} tails");

        // Same seed → identical bytes; another seed → different plan.
        let first: Vec<Vec<u8>> = set
            .files
            .iter()
            .map(|f| std::fs::read(root.shards("t").join(&f.file)).unwrap())
            .collect();
        build(&root, &opts).unwrap();
        let again: Vec<Vec<u8>> = set
            .files
            .iter()
            .map(|f| std::fs::read(root.shards("t").join(&f.file)).unwrap())
            .collect();
        assert_eq!(first, again);
        let mut other = opts.clone();
        other.seed = 2;
        other.name = "u".to_owned();
        build(&root, &other).unwrap();
        let set2 = ShardSet::open(&root.shards("u")).unwrap();
        let b1 = std::fs::read(root.shards("t").join("0000.bin")).unwrap();
        let b2 = std::fs::read(root.shards("u").join("0000.bin")).unwrap();
        assert_eq!(set2.count(None), 12);
        assert_ne!(b1, b2);

        // Split filter and too-short accounting.
        let mut dev_only = opts.clone();
        dev_only.name = "d".to_owned();
        dev_only.splits = vec![Split::Dev];
        dev_only.crop_s = 1.2;
        let s = build(&root, &dev_only).unwrap();
        assert_eq!(s.too_short, 1); // i = 0 has 50 frames < 60
        assert_eq!(s.counts.get(&Split::Train), None);
    }

    #[test]
    fn codec2_1600_crops_are_whole_320_sample_frames() {
        let mode = VocoderMode::Codec2_1600;
        let (_dir, root) = fixture(4, mode);
        let mut opts = ShardOptions::new("c2", mode);
        opts.crop_s = 1.0;
        let s = build(&root, &opts).unwrap();
        // Utterance i has 50 + 25i frames of 40 ms; a 1 s crop is 25.
        assert_eq!(s.counts.values().sum::<u64>(), 2 + 3 + 4 + 5);
        let set = ShardSet::open(&root.shards("c2")).unwrap();
        let l = &set.index.example_layout;
        assert_eq!((l.frames, l.frame_bytes, l.deg8_samples), (25, 8, 8_000));
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                assert_eq!(ex.deg8.len(), 8_000);
                assert_eq!(ex.frames.len(), 25 * 8);
                let speech = ex.mask_boundary as usize / 2;
                let mismatches = (0..speech)
                    .filter(|&k| (ex.deg8[k] - ex.clean16[2 * k]).abs() >= 2.0 / 32_767.0)
                    .count();
                assert!(mismatches <= 42, "{mismatches} misaligned samples");
                // The crop starts on a 320-sample frame boundary: the
                // frame bytes name their frame index in steps of 8 bytes.
                let f0 = ex.frames[0] as usize;
                for j in 0..speech / 320 {
                    assert_eq!(ex.frames[8 * j] as usize, f0 + j, "j={j}");
                }
                if ex.is_tail() {
                    assert!(ex.frames[speech / 320 * 8..].iter().all(|&v| v == 0));
                }
            }
        }
    }

    /// Add a twin of utterance 0: its own (different) input files and a
    /// captured pair, its clean target the parent's file.
    fn add_twin(root: &DataRoot, mode: VocoderMode) -> (String, String) {
        let parent = "vctk/p200_001_mic2".to_owned();
        let key = format!("{parent}+n0007");
        let mut rows: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        let p = rows.iter().find(|r| r.key == parent).unwrap().clone();
        let n8 = 50 * mode.frame_samples();
        // The twin's input is a different signal; its 8 kHz side is what
        // the codec saw, delayed by the same lag.
        let noisy8 = sine(700.0, 8_000, n8, 0.2);
        let mut noisy16 = Vec::with_capacity(2 * n8);
        for &v in &noisy8 {
            noisy16.push(v);
            noisy16.push(v);
        }
        write_wav_s16(root.prepared_16k(&key), &noisy16, 16_000).unwrap();
        write_wav_s16(root.prepared_8k(&key), &noisy8, 8_000).unwrap();
        let mut deg8 = vec![0.0f32; 42];
        deg8.extend_from_slice(&noisy8[..n8 - 42]);
        write_wav_s16(root.captured_wav(mode, &key), &deg8, 8_000).unwrap();
        std::fs::write(
            root.captured_ambe(mode, &key),
            vec![9u8; 50 * mode.frame_bytes()],
        )
        .unwrap();
        rows.push(UtteranceRow {
            key: key.clone(),
            parent: Some(parent.clone()),
            ..p
        });
        write_jsonl(&root.prepared_manifest(), &rows).unwrap();
        let mut cap: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        cap.push(CaptureRow {
            key: key.clone(),
            frames: 50,
            ..cap[0].clone()
        });
        write_jsonl(&root.captured_manifest(mode), &cap).unwrap();
        (parent, key)
    }

    #[test]
    fn a_twins_target_is_its_parents_clean_file() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        let (parent, key) = add_twin(&root, mode);
        let mut opts = ShardOptions::new("tw", mode);
        opts.crop_s = 1.0;
        opts.onset_share = 1.0;
        opts.tail_share = 0.0;
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.utterances, 4);
        assert_eq!(s.unjoined, 0);
        let set = ShardSet::open(&root.shards("tw")).unwrap();
        let parent16 = read_wav(root.prepared_16k(&parent)).unwrap().samples;
        let twin16 = read_wav(root.prepared_16k(&key)).unwrap().samples;
        let mut found = false;
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                if ex.frames[0] == 9 {
                    found = true;
                    assert_eq!(&ex.clean16[..], &parent16[..8_000 * 2], "target = parent");
                    assert_ne!(&ex.clean16[..], &twin16[..8_000 * 2]);
                    // The input is the twin's own capture.
                    assert!((ex.deg8[100] - twin16[200]).abs() < 2.0 / 32_767.0);
                }
            }
        }
        assert!(found, "the twin was packed");
        // A twin whose parent is gone is an error, never a silent skip.
        std::fs::remove_file(root.prepared_16k(&parent)).unwrap();
        let err = build(&root, &opts).unwrap_err().to_string();
        assert!(err.contains("twin of"), "{err}");
    }

    /// The corpus moved to FLAC (prepare and capture both write it), and
    /// the manifests name no extension. A set built from a FLAC corpus
    /// must join exactly as a WAV one does. Before the path helpers here
    /// resolved FLAC, every such utterance counted as `unjoined` and was
    /// dropped silently: no error, just a set missing all of the newer
    /// captures and all of Common Voice.
    #[test]
    fn a_flac_corpus_joins_and_packs() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let cdir = root.capture_dir(mode, None);
        for r in &rows {
            let k = r.key.as_str();
            for (wav, flac) in [
                (root.prepared_16k(k), root.prepared_16k_flac(k)),
                (root.prepared_8k(k), root.prepared_8k_flac(k)),
                (cdir.wav(k), cdir.flac(k)),
            ] {
                let a = read_wav(&wav).unwrap();
                write_flac_s16(&flac, &a.samples, a.rate).unwrap();
                std::fs::remove_file(&wav).unwrap();
            }
        }
        let mut opts = ShardOptions::new("flac", mode);
        opts.crop_s = 1.0;
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.unjoined, 0, "a FLAC corpus must join like a WAV one");
        assert!(s.utterances > 0 && s.files > 0, "{s:?}");
    }

    #[test]
    fn kinds_add_a_siblings_examples_and_are_recorded() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        // A `drops` sibling of utterance 1 only: same frames count, its
        // frames' bytes all 0xEE, its audio silent.
        let sib = root.capture_dir(mode, Some(AugKind::Drops));
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let r = base[1].clone();
        std::fs::create_dir_all(sib.ambe(&r.key).parent().unwrap()).unwrap();
        std::fs::write(sib.ambe(&r.key), vec![0xEE; r.frames as usize * 9]).unwrap();
        write_wav_s16(
            sib.wav(&r.key),
            &vec![0.0f32; r.frames as usize * 160],
            8_000,
        )
        .unwrap();
        write_jsonl(&sib.manifest(), std::slice::from_ref(&r)).unwrap();
        let mut opts = ShardOptions::new("k", mode);
        opts.crop_s = 1.0;
        opts.kinds = vec![KindSel::every(None), KindSel::every(Some(AugKind::Drops))];
        let s = build(&root, &opts).unwrap();
        // Base: 1 + 1 + 2 examples; the sibling adds utterance 1's one.
        assert_eq!(s.utterances, 4);
        assert_eq!(s.counts.values().sum::<u64>(), 5);
        let set = ShardSet::open(&root.shards("k")).unwrap();
        assert_eq!(set.index.kinds, vec!["base", "drops"]);
        assert_eq!(
            set.index.source_sha256s.siblings["drops"],
            sha256_file(&sib.manifest()).unwrap()
        );
        let mut from_sibling = 0;
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                if ex.frames[0] == 0xEE {
                    from_sibling += 1;
                    let speech = ex.mask_boundary as usize / 2;
                    assert!(ex.deg8[..speech].iter().all(|&v| v == 0.0));
                    assert!(
                        ex.clean16.iter().any(|&v| v != 0.0),
                        "target is the clean file"
                    );
                }
            }
        }
        assert_eq!(from_sibling, 1);
        // A kind with no manifest is refused; no kinds at all is refused.
        opts.kinds = vec![KindSel::every(Some(AugKind::Ber))];
        let err = build(&root, &opts).unwrap_err().to_string();
        assert!(err.contains("dstar+ber"), "{err}");
        opts.kinds = Vec::new();
        assert!(build(&root, &opts).is_err());
    }

    /// A kind that exists for one mode only can be drawn without
    /// demanding the same sibling of every other mode in the set. This is
    /// the software D-STAR recode: `dstar+perens` has no `ysf-dmr`
    /// counterpart and never will.
    #[test]
    fn a_mode_scoped_kind_is_drawn_for_that_mode_alone() {
        use VocoderMode::{Codec2_1600, Dstar};
        let (_dir, root) = fixture_with(3, &[Dstar, Codec2_1600], 320);
        // A sibling of D-STAR only.
        let sib = root.capture_dir(Dstar, Some(AugKind::Perens));
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(Dstar)).unwrap();
        let r = base[0].clone();
        std::fs::create_dir_all(sib.ambe(&r.key).parent().unwrap()).unwrap();
        std::fs::write(sib.ambe(&r.key), vec![0xAB; r.frames as usize * 9]).unwrap();
        write_wav_s16(
            sib.wav(&r.key),
            &vec![0.1f32; r.frames as usize * 160],
            8_000,
        )
        .unwrap();
        write_jsonl(&sib.manifest(), std::slice::from_ref(&r)).unwrap();

        // Codec 2 listed FIRST, so the sibling's mode is not index 0:
        // its provenance must still be recorded.
        let mut opts = ShardOptions::for_modes("scoped", vec![Codec2_1600, Dstar]);
        opts.crop_s = 1.0;
        opts.balance = false;
        opts.kinds = vec![
            KindSel::every(None),
            KindSel::scoped(Dstar, AugKind::Perens),
        ];
        build(&root, &opts).expect("a kind missing for the other mode must not fail the build");

        let set = ShardSet::open(&root.shards("scoped")).unwrap();
        assert_eq!(
            set.index.kinds,
            vec!["base", "dstar+perens"],
            "index.json records the scoped spelling"
        );
        assert!(
            set.index
                .source_sha256s
                .siblings
                .contains_key("dstar+perens"),
            "a scoped sibling on a non-first mode keeps its manifest hash: {:?}",
            set.index.source_sha256s.siblings
        );
    }

    /// A recode sibling is another implementation of the codec, with
    /// another delay. Its examples must be cut at its *own* canary lag, not
    /// its mode's: the first `dstar+perens` set was cut at the chip's 326
    /// where the software vocoder lags by 216, which trained a model on
    /// D-STAR pairs 14 ms apart and left it worse than the raw decode.
    #[test]
    fn a_recode_sibling_is_aligned_by_its_own_lag() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode); // the base capture lags by 42
        let sib = root.capture_dir(mode, Some(AugKind::Perens));
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let sib_lag = 17usize;
        for r in &base {
            let clean8 = read_wav(root.prepared_8k(&r.key)).unwrap().samples;
            let mut late = vec![0.0f32; sib_lag];
            late.extend_from_slice(&clean8[..clean8.len() - sib_lag]);
            std::fs::create_dir_all(sib.ambe(&r.key).parent().unwrap()).unwrap();
            std::fs::write(sib.ambe(&r.key), vec![0xAB; r.frames as usize * 9]).unwrap();
            write_wav_s16(sib.wav(&r.key), &late, 8_000).unwrap();
        }
        write_jsonl(&sib.manifest(), &base).unwrap();
        let mut rec: CanaryRecord =
            serde_json::from_slice(&std::fs::read(root.canary_json(mode)).unwrap()).unwrap();
        rec.lag_samples = i32::try_from(sib_lag).unwrap();
        write_json_atomic(&sib.canary_json(), &rec).unwrap();

        let mut opts = ShardOptions::new("lags", mode);
        opts.crop_s = 1.0;
        opts.kinds = vec![KindSel::every(None), KindSel::scoped(mode, AugKind::Perens)];
        build(&root, &opts).unwrap();
        let set = ShardSet::open(&root.shards("lags")).unwrap();
        assert_eq!(
            set.index.set_lags,
            [("dstar".to_owned(), 42), ("dstar+perens".to_owned(), 17)].into()
        );
        let (mut from_base, mut from_sibling) = (0, 0);
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                let sibling = ex.frames[0] == 0xAB;
                // Speech only: a tail crop's degraded side is garbage past it.
                // And not the last lag's worth: undoing the lag runs the
                // decoded side out before the clean one, which pads zeros.
                let end = if ex.is_tail() {
                    ex.mask_boundary as usize / 2
                } else {
                    ex.deg8.len()
                };
                let end = end.saturating_sub(64);
                let worst = (0..end)
                    .map(|k| (ex.deg8[k] - ex.clean16[2 * k]).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst < 2e-3,
                    "{} example is off its target by up to {worst}",
                    if sibling { "sibling" } else { "base" }
                );
                if sibling {
                    from_sibling += 1;
                } else {
                    from_base += 1;
                }
            }
        }
        assert!(
            from_base > 0 && from_sibling > 0,
            "{from_base} / {from_sibling}"
        );
    }

    /// An *unscoped* kind whose manifest is missing is still an error, so
    /// a typo fails loudly instead of quietly building a smaller set.
    #[test]
    fn an_unscoped_missing_kind_still_fails_loudly() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(2, mode);
        let mut opts = ShardOptions::new("typo", mode);
        opts.crop_s = 1.0;
        opts.kinds = vec![KindSel::every(Some(AugKind::Drops))];
        let err = build(&root, &opts).unwrap_err().to_string();
        assert!(err.contains("dstar+drops"), "{err}");
    }

    /// Two modes with different frame words in one set: D-STAR (160-sample
    /// frames, 9 bytes) and Codec 2 1600 (320-sample frames, 8 bytes).
    /// Every example carries its mode, is cropped in that mode's frame
    /// word and lag, and stores its frames at the front of the storage
    /// layout's (largest-mode) field.
    #[test]
    fn two_modes_of_different_frame_sizes_round_trip() {
        use VocoderMode::{Codec2_1600, Dstar};
        let (_dir, root) = fixture_with(6, &[Dstar, Codec2_1600], 320);
        let mut opts = ShardOptions::for_modes("mix", vec![Dstar, Codec2_1600]);
        opts.crop_s = 1.0;
        opts.balance = false;
        opts.examples_per_file = 5;
        let s = build(&root, &opts).unwrap();
        // Utterance i is (50 + 25 i) × 320 samples: 100 + 50 i D-STAR frames
        // (2 + i one-second crops) and 50 + 25 i Codec 2 1600 frames
        // (the same 2 + i crops). 27 per mode.
        assert_eq!(s.utterances, 12);
        assert_eq!(s.counts.values().sum::<u64>(), 54);
        assert_eq!(s.counts_by_mode[&Split::Train][&Dstar], 27 - (2 + 5));
        assert_eq!(s.counts_by_mode[&Split::Train][&Codec2_1600], 27 - (2 + 5));
        assert_eq!(s.available_by_mode, s.counts_by_mode);

        let set = ShardSet::open(&root.shards("mix")).unwrap();
        let idx = &set.index;
        assert_eq!(idx.mode, Dstar);
        assert_eq!(idx.modes, vec![Dstar, Codec2_1600]);
        assert!(!idx.balance);
        assert_eq!(idx.lags, [(Dstar, 42), (Codec2_1600, 42)].into());
        let l = &idx.example_layout;
        assert_eq!((l.frames, l.frame_bytes, l.deg8_samples), (50, 9, 8_000));
        assert_eq!(idx.mode_layouts[&Dstar].frames, 50);
        assert_eq!(idx.mode_layouts[&Codec2_1600].frames, 25);
        assert_eq!(idx.mode_layouts[&Codec2_1600].frame_bytes, 8);
        assert_eq!(
            idx.source_sha256s.by_set["codec2-1600"],
            sha256_file(&root.captured_manifest(Codec2_1600)).unwrap()
        );
        let mut per_mode = [0usize; 2];
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                assert_eq!(ex.deg8.len(), 8_000);
                assert_eq!(
                    ex.frames.len(),
                    450,
                    "the storage field is the largest mode's"
                );
                let mode = set.mode_of(&ex).expect("a mode of the set");
                per_mode[usize::from(ex.mode_index())] += 1;
                let (fs, bytes) = (mode.frame_samples(), mode.frame_bytes());
                let speech = ex.mask_boundary as usize / 2;
                let mismatches = (0..speech)
                    .filter(|&k| (ex.deg8[k] - ex.clean16[2 * k]).abs() >= 2.0 / 32_767.0)
                    .count();
                assert!(mismatches <= 42, "{mode}: {mismatches} misaligned samples");
                // Frame bytes name their frame index (mod 256) in the mode's
                // own word, crop start on the mode's own frame boundary.
                let f0 = ex.frames[0] as usize;
                for j in 0..speech / fs {
                    assert_eq!(
                        ex.frames[bytes * j] as usize,
                        (f0 + j) % 256,
                        "{mode} j={j}"
                    );
                }
                // Past the mode's own frames the field is zero padding.
                let own = idx.layout_for(mode).frames * bytes;
                assert!(ex.frames[own..].iter().all(|&b| b == 0), "{mode}: padding");
                assert_eq!(ex.is_onset(), f0 == 0);
            }
        }
        assert_eq!(per_mode, [27, 27]);
        // Deterministic: the same build again is byte-identical.
        let first = std::fs::read(root.shards("mix").join("0000.bin")).unwrap();
        build(&root, &opts).unwrap();
        assert_eq!(
            std::fs::read(root.shards("mix").join("0000.bin")).unwrap(),
            first
        );
        // A crop that is whole frames of one mode but not the other is refused.
        let mut half = opts.clone();
        half.crop_s = 0.5;
        let err = build(&root, &half).unwrap_err().to_string();
        assert!(err.contains("whole frames"), "{err}");
        // A mode listed twice is refused.
        let mut dup = opts.clone();
        dup.modes = vec![Dstar, Dstar];
        assert!(build(&root, &dup).is_err());
    }

    /// With `balance`, each split draws the same number of examples from
    /// every mode: the mode with fewer utterances sets the cap and the
    /// other's surplus is dropped by a seeded draw, both counts recorded.
    #[test]
    fn balance_draws_equally_from_every_mode_per_split() {
        use VocoderMode::{Codec2_3200, Dstar};
        let (_dir, root) = fixture_with(6, &[Dstar, Codec2_3200], 160);
        // Codec 2 has only utterances 0, 1, 2 captured (dev: 0; train: 1, 2).
        let rows: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(Codec2_3200)).unwrap();
        write_jsonl(&root.captured_manifest(Codec2_3200), &rows[..3]).unwrap();
        let mut opts = ShardOptions::for_modes("bal", vec![Dstar, Codec2_3200]);
        opts.crop_s = 1.0;
        // Utterance i: 50 + 25 i frames, 1 + i / 2 crops: 1,1,2,2,3,3.
        // D-STAR train (1, 2, 4, 5) = 1 + 2 + 3 + 3 = 9, dev (0, 3) = 1 + 2 = 3.
        // Codec 2 train (1, 2) = 1 + 2 = 3, dev (0) = 1.
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.available_by_mode[&Split::Train][&Dstar], 9);
        assert_eq!(s.available_by_mode[&Split::Train][&Codec2_3200], 3);
        assert_eq!(s.available_by_mode[&Split::Dev][&Dstar], 3);
        assert_eq!(s.available_by_mode[&Split::Dev][&Codec2_3200], 1);
        assert_eq!(s.counts_by_mode[&Split::Train][&Dstar], 3);
        assert_eq!(s.counts_by_mode[&Split::Train][&Codec2_3200], 3);
        assert_eq!(s.counts_by_mode[&Split::Dev][&Dstar], 1);
        assert_eq!(s.counts_by_mode[&Split::Dev][&Codec2_3200], 1);
        assert_eq!(s.counts[&Split::Train], 6);
        assert_eq!(s.counts[&Split::Dev], 2);
        let set = ShardSet::open(&root.shards("bal")).unwrap();
        assert!(set.index.balance);
        assert_eq!(set.index.counts_by_mode, s.counts_by_mode);
        assert_eq!(set.index.available_by_mode, s.available_by_mode);
        assert_eq!(set.count(None), 8);
        let mut seen = [0u64; 2];
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                seen[usize::from(set.read(fi, i).unwrap().mode_index())] += 1;
            }
        }
        assert_eq!(seen, [4, 4]);
        // The draw is seeded: the same build is byte-identical; another
        // seed keeps the counts but picks other examples.
        let first = std::fs::read(root.shards("bal").join("0000.bin")).unwrap();
        build(&root, &opts).unwrap();
        assert_eq!(
            std::fs::read(root.shards("bal").join("0000.bin")).unwrap(),
            first
        );
        // Off: everything is kept and the index says so.
        opts.balance = false;
        opts.name = "nobal".to_owned();
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.counts[&Split::Train], 12);
        assert_eq!(s.counts_by_mode, s.available_by_mode);
        assert!(!ShardSet::open(&root.shards("nobal")).unwrap().index.balance);
    }

    /// `corpora` draws only the named corpora, from every capture set, and
    /// says so in the index; unset, nothing is filtered.
    #[test]
    fn corpora_restricts_the_draw_and_is_recorded() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        let mut prepared: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        let noisy = prepared[1].key.clone();
        prepared[1].corpus = "common_voice".to_owned();
        write_jsonl(&root.prepared_manifest(), &prepared).unwrap();

        let mut opts = ShardOptions::new("studio", mode);
        let all = join_all(&root, &opts, &prepared).unwrap().items;
        opts.corpora = vec!["vctk".to_owned(), "ljspeech".to_owned()];
        let studio = join_all(&root, &opts, &prepared).unwrap().items;
        assert_eq!(all.len(), 3);
        assert_eq!(studio.len(), 2);
        assert!(studio.iter().all(|i| i.key != noisy));

        opts.crop_s = 1.0;
        build(&root, &opts).unwrap();
        let idx = ShardSet::open(&root.shards("studio")).unwrap().index;
        assert_eq!(idx.corpora, vec!["vctk", "ljspeech"]);
    }

    /// One drops sibling, two passes: `min_drop_rate` draws the heavy rows
    /// only, leaves the base capture alone, and is recorded in the index.
    #[test]
    fn min_drop_rate_keeps_the_heavy_drops_rows_and_every_base_row() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        let sib = root.capture_dir(mode, Some(AugKind::Drops));
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let mut rows = Vec::new();
        for (r, ppm) in base.iter().take(2).zip([20_000u32, 100_000]) {
            let mut r = r.clone();
            std::fs::create_dir_all(sib.ambe(&r.key).parent().unwrap()).unwrap();
            std::fs::write(sib.ambe(&r.key), vec![0xEE; r.frames as usize * 9]).unwrap();
            write_wav_s16(
                sib.wav(&r.key),
                &vec![0.0f32; r.frames as usize * 160],
                8_000,
            )
            .unwrap();
            r.aug = Some(CaptureAug {
                kind: AugKind::Drops,
                rate_ppm: ppm,
                burst: Some((1, 3)),
                subst: Some(unamblify::Subst::Mute),
                fec: None,
                seed: 1,
                positions: vec![1],
            });
            rows.push(r);
        }
        write_jsonl(&sib.manifest(), &rows).unwrap();
        assert!(below_min_drop_rate(&rows[0], Some(0.05)));
        assert!(!below_min_drop_rate(&rows[1], Some(0.05)));
        assert!(!below_min_drop_rate(&rows[0], None));
        assert!(
            !below_min_drop_rate(&base[0], Some(0.05)),
            "a base row has no rate"
        );

        let prepared: Vec<UtteranceRow> = read_jsonl(&root.prepared_manifest()).unwrap();
        let mut opts = ShardOptions::new("heavy", mode);
        opts.kinds = vec![KindSel::every(None), KindSel::every(Some(AugKind::Drops))];
        let all = join_all(&root, &opts, &prepared).unwrap().items;
        opts.min_drop_rate = Some(0.05);
        let heavy = join_all(&root, &opts, &prepared).unwrap().items;
        let drops = |v: &[Item]| v.iter().filter(|i| i.kind.is_some()).count();
        assert_eq!((drops(&all), drops(&heavy)), (2, 1));
        assert_eq!(all.len() - heavy.len(), 1, "only a drops row may go");
        assert_eq!(
            heavy.iter().find(|i| i.kind.is_some()).map(|i| &i.key),
            Some(&rows[1].key)
        );

        opts.crop_s = 1.0;
        build(&root, &opts).unwrap();
        let idx = ShardSet::open(&root.shards("heavy")).unwrap().index;
        assert_eq!(idx.min_drop_rate, Some(0.05));
    }

    /// A set that draws from a drops sibling carries the mask, marks the
    /// sibling's lost frames and nothing in the base capture; a set that
    /// does not carries none, so its examples are the size they always were.
    #[test]
    fn a_drops_set_carries_the_mask_and_a_base_set_does_not() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        let sib = root.capture_dir(mode, Some(AugKind::Drops));
        let base: Vec<CaptureRow> = read_jsonl(&root.captured_manifest(mode)).unwrap();
        let mut r = base[1].clone();
        std::fs::create_dir_all(sib.ambe(&r.key).parent().unwrap()).unwrap();
        std::fs::write(sib.ambe(&r.key), vec![0xEE; r.frames as usize * 9]).unwrap();
        write_wav_s16(
            sib.wav(&r.key),
            &vec![0.0f32; r.frames as usize * 160],
            8_000,
        )
        .unwrap();
        r.aug = Some(unamblify::aug::CaptureAug {
            kind: AugKind::Drops,
            rate_ppm: 20_000,
            burst: Some((1, 3)),
            subst: Some(unamblify::Subst::Mute),
            fec: None,
            seed: 1,
            // Every frame: the seeded planner gives this utterance one
            // crop, and it can be a tail crop with a single speech frame,
            // so only losing them all guarantees the crop holds one. The
            // exact geometry is the unit test above.
            positions: (0..r.frames).collect(),
        });
        write_jsonl(&sib.manifest(), std::slice::from_ref(&r)).unwrap();

        let mut plain = ShardOptions::new("plain", mode);
        plain.crop_s = 1.0;
        build(&root, &plain).unwrap();
        let plain_set = ShardSet::open(&root.shards("plain")).unwrap();
        assert!(!plain_set.index.erasure);
        assert_eq!(plain_set.index.example_layout.erasure_bytes, 0);

        let mut opts = ShardOptions::new("drops", mode);
        opts.crop_s = 1.0;
        opts.kinds = vec![KindSel::every(None), KindSel::every(Some(AugKind::Drops))];
        build(&root, &opts).unwrap();
        let set = ShardSet::open(&root.shards("drops")).unwrap();
        assert!(set.index.erasure);
        let layout = set.index.example_layout.clone();
        assert_eq!(layout.erasure_bytes, layout.frames);
        assert_eq!(
            layout.example_bytes(),
            plain_set.index.example_layout.example_bytes() + layout.frames
        );
        let (mut marked_sibling, mut marked_base) = (0usize, 0usize);
        for (fi, f) in set.files.iter().enumerate() {
            for i in 0..f.examples {
                let ex = set.read(fi, i).unwrap();
                assert_eq!(ex.erasure.len(), layout.frames);
                let n: usize = ex.erasure.iter().map(|&v| usize::from(v)).sum();
                if ex.frames[0] == 0xEE {
                    marked_sibling += n;
                } else {
                    marked_base += n;
                }
            }
        }
        assert!(marked_sibling > 0, "the sibling's lost frames are marked");
        assert_eq!(marked_base, 0, "the base capture lost nothing");
    }

    /// 42 k twins are a fifth of the YSF capture and a fiftieth of a
    /// Codec 2 one; a uniform draw takes them in that proportion.
    /// `twin_share` draws them separately.
    #[test]
    fn twin_share_stratifies_the_draw_and_leaves_the_uniform_one_alone() {
        let item = |i: usize, twin: bool| {
            let parent = format!("c/u{i:05}");
            Item {
                key: if twin {
                    format!("{parent}+u0001")
                } else {
                    parent.clone()
                },
                target_key: parent,
                mode: VocoderMode::Codec2_3200,
                mode_idx: 0,
                kind: None,
                split: Split::Train,
                frames: 100,
                speaker_id: 0,
                erased: Vec::new(),
                lag: 0,
            }
        };
        // A Codec 2-shaped set: 2 % twins.
        let pool = || -> Vec<Item> {
            (0..5_000)
                .map(|i| item(i, false))
                .chain((0..100).map(|i| item(i, true)))
                .collect()
        };
        let twins_in = |v: &[Item]| v.iter().filter(|i| i.key != i.target_key).count();
        let mut opts = ShardOptions::new("t", VocoderMode::Codec2_3200);
        opts.max_utterances = Some(500);

        // Uniform: about 2 % of the draw, and the same items as before
        // the option existed (the path is untouched when it is unset).
        let uniform = limit_utterances(pool(), &opts, 7);
        assert_eq!(uniform.len(), 500);
        assert!(twins_in(&uniform) < 30, "{}", twins_in(&uniform));
        let again = limit_utterances(pool(), &opts, 7);
        assert!(
            uniform.iter().zip(&again).all(|(a, b)| a.key == b.key),
            "seeded"
        );

        // Stratified: a tenth of the draw is twins.
        opts.twin_share = Some(0.10);
        let strat = limit_utterances(pool(), &opts, 7);
        assert_eq!((strat.len(), twins_in(&strat)), (500, 50));
        assert!(strat.windows(2).all(|w| w[0].key <= w[1].key), "key order");

        // More twins asked for than exist: all of them, base fills the rest.
        opts.twin_share = Some(0.50);
        let short = limit_utterances(pool(), &opts, 7);
        assert_eq!((short.len(), twins_in(&short)), (500, 100));

        // Base runs short instead: twins take the unused places.
        let few_base: Vec<Item> = (0..20)
            .map(|i| item(i, false))
            .chain((0..600).map(|i| item(i, true)))
            .collect();
        opts.twin_share = Some(0.10);
        let topped = limit_utterances(few_base, &opts, 7);
        assert_eq!((topped.len(), twins_in(&topped)), (500, 480));

        // A capture smaller than the cap is kept whole either way.
        let small: Vec<Item> = (0..40).map(|i| item(i, i % 4 == 0)).collect();
        assert_eq!(limit_utterances(small, &opts, 7).len(), 40);
    }

    /// `max_utterances` keeps a seeded subset of each capture set, in
    /// key order, and the index says so.
    #[test]
    fn max_utterances_bounds_a_trial_set() {
        use VocoderMode::{Codec2_3200, Dstar};
        let (_dir, root) = fixture_with(6, &[Dstar, Codec2_3200], 160);
        let mut opts = ShardOptions::for_modes("cap", vec![Dstar, Codec2_3200]);
        opts.crop_s = 1.0;
        opts.max_utterances = Some(2);
        let s = build(&root, &opts).unwrap();
        assert_eq!(s.utterances, 4, "two per mode");
        let set = ShardSet::open(&root.shards("cap")).unwrap();
        assert_eq!(set.index.max_utterances, Some(2));
        let again = build(&root, &opts).unwrap();
        assert_eq!(again.counts, s.counts, "seeded");
        opts.max_utterances = Some(100);
        assert_eq!(build(&root, &opts).unwrap().utterances, 12);
        assert_eq!(
            ShardSet::open(&root.shards("cap"))
                .unwrap()
                .index
                .max_utterances,
            Some(100)
        );
    }

    #[test]
    fn build_refuses_a_missing_canary() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(3, mode);
        std::fs::remove_file(root.canary_json(mode)).unwrap();
        let err = build(&root, &ShardOptions::new("nc", mode)).unwrap_err();
        assert!(err.to_string().contains("canary.json"), "{err}");
        assert!(!root.shards("nc").join("index.json").exists());
    }

    #[test]
    fn a_rebuild_leaves_no_stale_shard_files() {
        let mode = VocoderMode::Dstar;
        let (_dir, root) = fixture(6, mode);
        let mut opts = ShardOptions::new("s", mode);
        opts.crop_s = 0.5;
        opts.examples_per_file = 1;
        let first = build(&root, &opts).unwrap();
        opts.crop_s = 1.0;
        let second = build(&root, &opts).unwrap();
        assert!(
            second.files < first.files,
            "{} vs {}",
            second.files,
            first.files
        );
        let bins = std::fs::read_dir(root.shards("s"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
            .count();
        assert_eq!(bins as u64, second.files);
        let set = ShardSet::open(&root.shards("s")).unwrap();
        assert_eq!(set.files.len() as u64, second.files);
        let eb = set.index.example_layout.example_bytes() as u64;
        for f in &set.files {
            let len = std::fs::metadata(root.shards("s").join(&f.file))
                .unwrap()
                .len();
            assert_eq!(len, f.examples * eb, "{}", f.file);
        }
    }
}
