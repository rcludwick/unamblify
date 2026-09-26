// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `shards/<name>/index.json` (spec §4): what a pre-packed shard set was
//! built from and how each fixed-length example is laid out in `NNNN.bin`.
//!
//! Example order across the files is by split: taking the files in
//! `NNNN` order and concatenating their examples, the first
//! `counts[train]` are the train split, the next `counts[dev]` the dev
//! split, and the last `counts[test]` the test split. Within a split the
//! order is the deterministic packing order (seeded); a loader shuffles by
//! index.
//!
//! A set may hold more than one vocoder mode (`unamblify shard --modes
//! dstar,codec2-3200`): `modes` lists them, every example carries the
//! index of its mode in the flags byte, and `example_layout` is the
//! *storage* layout every example shares (the channel-frame field sized
//! for the largest mode) while `mode_layouts` gives each mode's own frame
//! word inside it. A set written before any of that existed has one mode
//! and no mode bits; it reads back as `modes = [mode]` with every example
//! at index 0.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::{Split, VOCODER_SAMPLE_RATE, VocoderMode, WIDEBAND_SAMPLE_RATE};

/// SHA-256 of the manifests a shard set was derived from, so a stale set is
/// detectable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSha256s {
    /// `prepared/manifest.jsonl`.
    pub prepared: String,
    /// `captured/<mode>/manifest.jsonl`.
    pub captured: String,
    /// `captured/<mode>+<kind>/manifest.jsonl` per sibling kind the set
    /// includes (`kinds` minus `base`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub siblings: BTreeMap<String, String>,
    /// Every capture set the build read, keyed by its directory name
    /// (`dstar`, `codec2-3200`, `codec2-3200+drops`), so a multi-mode set
    /// records each mode's manifest too. `captured` and `siblings` are the
    /// first mode's, for readers that predate `modes`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_set: BTreeMap<String, String>,
}

/// Byte layout of one example in a shard file. Every field is stored in
/// this order, little-endian, with no padding: `clean16` as f32, `deg8` as
/// f32, `frames` as raw bytes, then `flags`.
///
/// In a multi-mode set this is the storage layout every example shares:
/// `frames × frame_bytes` is the largest mode's channel-frame field
/// (`max frames × max frame_bytes` over the set's modes) and an example
/// of a smaller mode fills its own `frames × frame_bytes` bytes of it,
/// the rest zero — see [`ExampleLayout::for_modes`]. The audio fields are
/// the same length in every mode by construction (a crop is whole frames
/// of each mode; a crop that is not is refused).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleLayout {
    /// f32 samples of the 16 kHz clean target.
    pub clean16_samples: usize,
    /// f32 samples of the 8 kHz degraded input
    /// (`frames × VocoderMode::frame_samples`).
    pub deg8_samples: usize,
    /// Channel frames per example.
    pub frames: usize,
    /// Bytes per channel frame (`VocoderMode::frame_bytes`).
    pub frame_bytes: usize,
    /// Trailing flag bytes, in order: one byte of flags (bit 0 = onset
    /// example, bit 1 = tail example, bits 2–7 = the example's mode as an
    /// index into the set's `modes`, 0 in a single-mode set); the `mask`
    /// boundary as a little-endian u32 sample index into `clean16`
    /// (samples at or after it are garbage-tail padding, `clean16_samples`
    /// when there is none); then the [`speaker_id`] as a little-endian
    /// u32. Nine bytes.
    pub flags_bytes: usize,
    /// Erasure-mask bytes, after the flag block: one byte per channel
    /// frame, 1 where that frame was lost on the channel and concealed,
    /// 0 where it arrived. `0` in a set built without the mask — the
    /// field is appended last, so every earlier offset is unchanged and
    /// a set written before it existed still reads (`serde` default).
    #[serde(default)]
    pub erasure_bytes: usize,
}

/// A crop that is not a whole number of frames of every mode in a set:
/// the modes would store audio fields of different lengths.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutMismatch {
    /// The crop asked for, seconds.
    pub crop_s: f32,
    /// The first mode and its `deg8` samples for the crop.
    pub first: (VocoderMode, usize),
    /// The mode that disagrees and its `deg8` samples.
    pub other: (VocoderMode, usize),
}

impl fmt::Display for LayoutMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a {} s crop is {} samples of {} but {} of {}: use a crop that is whole frames of every mode (a multiple of {} ms)",
            self.crop_s,
            self.first.1,
            self.first.0,
            self.other.1,
            self.other.0,
            self.first.0.frame_ms().max(self.other.0.frame_ms())
        )
    }
}

impl std::error::Error for LayoutMismatch {}

/// Size of the trailing flag block: flags u8 + mask boundary u32 +
/// speaker id u32.
pub const FLAGS_BYTES: usize = 9;

/// Stable 32-bit speaker identifier stored in shards and yielded as the
/// batch's `speaker` tensor by every loader: the first four bytes of
/// `sha1(speaker)`, big-endian. Needs no vocabulary, so the pipeline and
/// shard loaders agree without sharing state.
#[must_use]
pub fn speaker_id(speaker: &str) -> u32 {
    let digest = Sha1::digest(speaker.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
}

impl ExampleLayout {
    /// Flag bit: the example starts at a key-down (onset).
    pub const FLAG_ONSET: u8 = 0b01;
    /// Flag bit: the example ends in a garbage tail (mask == 0 region).
    pub const FLAG_TAIL: u8 = 0b10;
    /// The flags byte's bits above the onset and tail bits carry the
    /// example's mode index (`flags >> FLAG_MODE_SHIFT`).
    pub const FLAG_MODE_SHIFT: u8 = 2;
    /// Mask of the mode-index bits of the flags byte.
    pub const FLAG_MODE_MASK: u8 = 0b1111_1100;
    /// Modes a set can hold: the six mode bits of the flags byte.
    pub const MAX_MODES: usize = 64;

    /// The mode index stored in a flags byte (0 for a set written before
    /// the bits existed).
    #[must_use]
    pub const fn mode_index(flags: u8) -> u8 {
        (flags & Self::FLAG_MODE_MASK) >> Self::FLAG_MODE_SHIFT
    }

    /// `flags` with its mode bits set to `index` (which must be below
    /// [`ExampleLayout::MAX_MODES`]; higher bits are dropped).
    #[must_use]
    pub const fn with_mode(flags: u8, index: u8) -> u8 {
        (flags & !Self::FLAG_MODE_MASK) | (index << Self::FLAG_MODE_SHIFT)
    }

    /// The storage layout of a set holding `modes` at `crop_s`: the audio
    /// fields of the first mode (every mode must agree, or the crop is not
    /// whole frames of each and this is an error) and a channel-frame
    /// field of `max frames × max frame_bytes` over the modes, so the
    /// largest mode's frames fit and a smaller mode's are zero-padded. With
    /// one mode this is [`ExampleLayout::for_crop`].
    pub fn for_modes(modes: &[VocoderMode], crop_s: f32) -> Result<Self, LayoutMismatch> {
        let Some(&first) = modes.first() else {
            return Ok(Self::for_crop(VocoderMode::Dstar, 0.0));
        };
        let mut out = Self::for_crop(first, crop_s);
        for &m in &modes[1..] {
            let l = Self::for_crop(m, crop_s);
            if l.deg8_samples != out.deg8_samples {
                return Err(LayoutMismatch {
                    crop_s,
                    first: (first, out.deg8_samples),
                    other: (m, l.deg8_samples),
                });
            }
            out.frames = out.frames.max(l.frames);
            out.frame_bytes = out.frame_bytes.max(l.frame_bytes);
        }
        Ok(out)
    }

    /// The layout for a crop of `crop_s` seconds in `mode`: whole frames of
    /// the mode's own length, so a 2 s crop is 100 frames of 160 samples in
    /// an AMBE mode and 50 frames of 320 in Codec 2 1600.
    #[must_use]
    pub fn for_crop(mode: VocoderMode, crop_s: f32) -> Self {
        // Whole-frame crops only; truncation is intended.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let millis = (crop_s.max(0.0) * 1000.0).round() as usize;
        let frames = millis / mode.frame_ms() as usize;
        let deg8_samples = frames * mode.frame_samples();
        let clean16_samples = deg8_samples * (WIDEBAND_SAMPLE_RATE / VOCODER_SAMPLE_RATE) as usize;
        Self {
            clean16_samples,
            deg8_samples,
            frames,
            frame_bytes: mode.frame_bytes(),
            flags_bytes: FLAGS_BYTES,
            erasure_bytes: 0,
        }
    }

    /// The same layout with (or without) the per-frame erasure mask: one
    /// byte per channel frame. Apply it after [`ExampleLayout::for_modes`],
    /// which settles `frames` at the largest mode's count.
    #[must_use]
    pub const fn with_erasure(mut self, on: bool) -> Self {
        self.erasure_bytes = if on { self.frames } else { 0 };
        self
    }

    /// Byte offset of `deg8` within an example.
    #[must_use]
    pub const fn deg8_offset(&self) -> usize {
        self.clean16_samples * 4
    }

    /// Byte offset of the channel frames within an example.
    #[must_use]
    pub const fn frames_offset(&self) -> usize {
        self.deg8_offset() + self.deg8_samples * 4
    }

    /// Byte offset of the flag bytes within an example.
    #[must_use]
    pub const fn flags_offset(&self) -> usize {
        self.frames_offset() + self.frames * self.frame_bytes
    }

    /// Byte offset of the erasure mask within an example. Only meaningful
    /// when `erasure_bytes` is non-zero.
    #[must_use]
    pub const fn erasure_offset(&self) -> usize {
        self.flags_offset() + self.flags_bytes
    }

    /// Total bytes per example.
    #[must_use]
    pub const fn example_bytes(&self) -> usize {
        (self.clean16_samples + self.deg8_samples) * 4
            + self.frames * self.frame_bytes
            + self.flags_bytes
            + self.erasure_bytes
    }
}

/// `kinds` of a set built before the field existed.
#[must_use]
pub fn default_kinds() -> Vec<String> {
    vec!["base".to_owned()]
}

/// One entry of `shards/<name>/files.json`: which file holds which split
/// and how many examples it carries. A loader opens exactly these files,
/// so a stale `NNNN.bin` left over from an earlier build is never read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardFile {
    /// File name (`0000.bin`).
    pub file: String,
    /// Split of every example in it.
    pub split: Split,
    /// Examples in it.
    pub examples: u64,
}

/// `index.json` of a shard set.
///
/// Deserialises sets written before `modes` existed: `modes` becomes
/// `[mode]`, `mode_layouts` `{mode: example_layout}`, `lags`
/// `{mode: lag_samples}`, and the per-mode counts the split counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "ShardIndexDe")]
pub struct ShardIndex {
    /// Shard set name (`shards/<name>/`).
    pub name: String,
    /// Vocoder mode of the degraded input: the first of `modes`.
    pub mode: VocoderMode,
    /// Every mode in the set, in index order: an example's mode bits
    /// (`ExampleLayout::mode_index`) index this list.
    pub modes: Vec<VocoderMode>,
    /// Crop length, seconds.
    pub crop_s: f32,
    /// Seed the deterministic packing used.
    pub seed: u64,
    /// The capture lag (`canary.json`'s `lag_samples`) that was undone when
    /// `deg8` was paired with `clean16`, for `mode`. Absent in sets built
    /// before the field existed; a loader refuses those, since their
    /// alignment is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_samples: Option<i32>,
    /// The lag undone per mode (every mode of `modes`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lags: BTreeMap<VocoderMode, i32>,
    /// The lag undone per capture set, by directory name (`dstar`,
    /// `dstar+perens`). A recode sibling is another implementation of the
    /// codec with its own delay, so it cannot share its mode's entry in
    /// `lags`. Absent in a set built before 2026-09-20, every sibling of
    /// which was cut at its mode's lag — wrong for `dstar+perens` (216
    /// against 326): such a set must be rebuilt, not trained on.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub set_lags: BTreeMap<String, i32>,
    /// Examples per split.
    pub counts: BTreeMap<Split, u64>,
    /// Examples per split and mode — the draw after balancing.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counts_by_mode: BTreeMap<Split, BTreeMap<VocoderMode, u64>>,
    /// Examples each mode could have contributed per split before
    /// balancing capped it; equal to `counts_by_mode` when nothing was
    /// dropped.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub available_by_mode: BTreeMap<Split, BTreeMap<VocoderMode, u64>>,
    /// Whether the build drew equally from every mode per split (each
    /// mode capped at the smallest mode's count, the surplus dropped by a
    /// seeded draw).
    #[serde(default)]
    pub balance: bool,
    /// The build kept at most this many utterances per capture set (a
    /// seeded draw) — a trial set, not the whole capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_utterances: Option<u64>,
    /// With `max_utterances`: the fraction of each capture set's draw
    /// reserved for augmented twins (rows with a `parent`), so a set whose
    /// twins are a sliver of a huge capture still trains on them. Absent
    /// in a set drawn uniformly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twin_share: Option<f32>,
    /// The lowest frame-loss rate a `drops` row was generated at to be
    /// drawn. A drops sibling may hold rows of several `augment` passes;
    /// this keeps a set built to train frame restoration from being
    /// mostly the light 2 % pass. Absent when every row was eligible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_drop_rate: Option<f32>,
    /// The corpora the draw was restricted to; empty when it took them
    /// all. A set whose targets are studio recordings only names them here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corpora: Vec<String>,
    /// Fraction of examples that are onset crops.
    pub onset_share: f32,
    /// Fraction of examples that carry a garbage tail.
    pub tail_share: f32,
    /// The capture sets the examples were drawn from: `base` and any
    /// decode-only sibling kinds (`drops`, `ber`). Sets built before the
    /// field existed are `base` only.
    #[serde(default = "default_kinds")]
    pub kinds: Vec<String>,
    /// Whether every example carries the per-frame erasure mask
    /// (`ExampleLayout::erasure_bytes`). False in a set built before the
    /// mask existed, whose examples have no mask bytes at all.
    #[serde(default)]
    pub erasure: bool,
    /// What the set was built from.
    pub source_sha256s: SourceSha256s,
    /// How to read one example: the storage layout every example shares.
    pub example_layout: ExampleLayout,
    /// Each mode's own layout inside the storage layout: the same audio
    /// fields, its own `frames` and `frame_bytes` (the first
    /// `frames × frame_bytes` bytes of the channel-frame field).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mode_layouts: BTreeMap<VocoderMode, ExampleLayout>,
}

impl ShardIndex {
    /// Index of `mode` in `modes` — what an example of that mode carries
    /// in its flags byte.
    #[must_use]
    pub fn mode_index_of(&self, mode: VocoderMode) -> Option<u8> {
        self.modes
            .iter()
            .position(|&m| m == mode)
            .and_then(|i| u8::try_from(i).ok())
    }

    /// The mode at `index`, if the set has one.
    #[must_use]
    pub fn mode_at(&self, index: u8) -> Option<VocoderMode> {
        self.modes.get(usize::from(index)).copied()
    }

    /// `mode`'s own layout (its frame word inside the storage layout);
    /// the storage layout when the set does not record one.
    #[must_use]
    pub fn layout_for(&self, mode: VocoderMode) -> &ExampleLayout {
        self.mode_layouts.get(&mode).unwrap_or(&self.example_layout)
    }

    /// The lag undone for `mode`.
    #[must_use]
    pub fn lag_for(&self, mode: VocoderMode) -> Option<i32> {
        self.lags.get(&mode).copied().or(self.lag_samples)
    }
}

/// `ShardIndex` as written to disk by any version: the fields that came
/// later default here and are filled in from the older ones.
#[derive(Deserialize)]
struct ShardIndexDe {
    name: String,
    mode: VocoderMode,
    #[serde(default)]
    modes: Vec<VocoderMode>,
    crop_s: f32,
    seed: u64,
    #[serde(default)]
    lag_samples: Option<i32>,
    #[serde(default)]
    lags: BTreeMap<VocoderMode, i32>,
    #[serde(default)]
    set_lags: BTreeMap<String, i32>,
    counts: BTreeMap<Split, u64>,
    #[serde(default)]
    counts_by_mode: BTreeMap<Split, BTreeMap<VocoderMode, u64>>,
    #[serde(default)]
    available_by_mode: BTreeMap<Split, BTreeMap<VocoderMode, u64>>,
    #[serde(default)]
    balance: bool,
    #[serde(default)]
    max_utterances: Option<u64>,
    #[serde(default)]
    twin_share: Option<f32>,
    #[serde(default)]
    min_drop_rate: Option<f32>,
    #[serde(default)]
    corpora: Vec<String>,
    onset_share: f32,
    tail_share: f32,
    #[serde(default = "default_kinds")]
    kinds: Vec<String>,
    #[serde(default)]
    erasure: bool,
    source_sha256s: SourceSha256s,
    example_layout: ExampleLayout,
    #[serde(default)]
    mode_layouts: BTreeMap<VocoderMode, ExampleLayout>,
}

impl From<ShardIndexDe> for ShardIndex {
    fn from(d: ShardIndexDe) -> Self {
        let modes = if d.modes.is_empty() {
            vec![d.mode]
        } else {
            d.modes
        };
        let mut lags = d.lags;
        if lags.is_empty()
            && let Some(l) = d.lag_samples
        {
            lags.insert(d.mode, l);
        }
        let mut counts_by_mode = d.counts_by_mode;
        if counts_by_mode.is_empty() {
            for (&s, &n) in &d.counts {
                counts_by_mode.insert(s, [(d.mode, n)].into());
            }
        }
        let mut available_by_mode = d.available_by_mode;
        if available_by_mode.is_empty() {
            available_by_mode.clone_from(&counts_by_mode);
        }
        let mut mode_layouts = d.mode_layouts;
        if mode_layouts.is_empty() {
            mode_layouts.insert(d.mode, d.example_layout.clone());
        }
        Self {
            name: d.name,
            mode: d.mode,
            modes,
            crop_s: d.crop_s,
            seed: d.seed,
            lag_samples: d.lag_samples,
            lags,
            set_lags: d.set_lags,
            counts: d.counts,
            counts_by_mode,
            available_by_mode,
            balance: d.balance,
            max_utterances: d.max_utterances,
            twin_share: d.twin_share,
            min_drop_rate: d.min_drop_rate,
            corpora: d.corpora,
            onset_share: d.onset_share,
            tail_share: d.tail_share,
            kinds: d.kinds,
            erasure: d.erasure,
            source_sha256s: d.source_sha256s,
            example_layout: d.example_layout,
            mode_layouts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_for_a_two_second_dstar_crop() {
        let l = ExampleLayout::for_crop(VocoderMode::Dstar, 2.0);
        assert_eq!(l.frames, 100);
        assert_eq!(l.deg8_samples, 16_000);
        assert_eq!(l.clean16_samples, 32_000);
        assert_eq!(l.frame_bytes, 9);
        assert_eq!(l.example_bytes(), 48_000 * 4 + 900 + 9);
        assert_eq!(l.deg8_offset(), 128_000);
        assert_eq!(l.frames_offset(), 192_000);
        assert_eq!(l.flags_offset(), 192_900);
        assert_eq!(
            ExampleLayout::for_crop(VocoderMode::YsfDmr, 2.0).frame_bytes,
            7
        );
    }

    #[test]
    fn layout_for_the_codec2_modes() {
        let l = ExampleLayout::for_crop(VocoderMode::Codec2_3200, 2.0);
        assert_eq!(
            (l.frames, l.deg8_samples, l.clean16_samples),
            (100, 16_000, 32_000)
        );
        assert_eq!(l.frame_bytes, 8);
        assert_eq!(l.example_bytes(), 48_000 * 4 + 800 + 9);
        // 40 ms frames: half as many, twice as long, same audio.
        let l = ExampleLayout::for_crop(VocoderMode::Codec2_1600, 2.0);
        assert_eq!(
            (l.frames, l.deg8_samples, l.clean16_samples),
            (50, 16_000, 32_000)
        );
        assert_eq!(l.frame_bytes, 8);
        assert_eq!(l.example_bytes(), 48_000 * 4 + 400 + 9);
        // A crop that is not a whole number of 40 ms frames rounds down.
        let l = ExampleLayout::for_crop(VocoderMode::Codec2_1600, 0.5);
        assert_eq!(l.frames, 12);
        assert_eq!(l.deg8_samples, 12 * 320);
    }

    #[test]
    fn speaker_id_is_pinned_and_matches_the_split_hash() {
        // First byte agrees with the split rule's `speaker_hash_byte`.
        assert_eq!(speaker_id("p228") >> 24, 0x00);
        assert_eq!(speaker_id("p225") >> 24, 0xf3);
        assert_ne!(speaker_id("p225"), speaker_id("p226"));
        assert_eq!(speaker_id("p225"), speaker_id("p225"));
    }

    #[test]
    fn index_round_trips_with_split_keys() {
        let layout = ExampleLayout::for_crop(VocoderMode::Dstar, 2.0);
        let idx = ShardIndex {
            name: "seed-dstar".to_owned(),
            mode: VocoderMode::Dstar,
            modes: vec![VocoderMode::Dstar],
            erasure: false,
            crop_s: 2.0,
            seed: 1,
            lag_samples: Some(42),
            lags: [(VocoderMode::Dstar, 42)].into(),
            set_lags: BTreeMap::new(),
            counts: [(Split::Train, 1000), (Split::Dev, 80), (Split::Test, 40)].into(),
            counts_by_mode: [
                (Split::Train, [(VocoderMode::Dstar, 1000)].into()),
                (Split::Dev, [(VocoderMode::Dstar, 80)].into()),
                (Split::Test, [(VocoderMode::Dstar, 40)].into()),
            ]
            .into(),
            available_by_mode: [
                (Split::Train, [(VocoderMode::Dstar, 1000)].into()),
                (Split::Dev, [(VocoderMode::Dstar, 80)].into()),
                (Split::Test, [(VocoderMode::Dstar, 40)].into()),
            ]
            .into(),
            balance: true,
            max_utterances: None,
            twin_share: None,
            min_drop_rate: None,
            corpora: Vec::new(),
            onset_share: 0.34,
            tail_share: 0.15,
            kinds: vec!["base".to_owned(), "drops".to_owned()],
            source_sha256s: SourceSha256s {
                prepared: "p".to_owned(),
                captured: "c".to_owned(),
                siblings: [("drops".to_owned(), "d".to_owned())].into(),
                by_set: [
                    ("dstar".to_owned(), "c".to_owned()),
                    ("dstar+drops".to_owned(), "d".to_owned()),
                ]
                .into(),
            },
            example_layout: layout.clone(),
            mode_layouts: [(VocoderMode::Dstar, layout)].into(),
        };
        let text = serde_json::to_string_pretty(&idx).unwrap();
        assert!(text.contains("\"train\": 1000"));
        assert!(text.contains("\"mode\": \"dstar\""));
        assert!(text.contains("\"modes\": [\n    \"dstar\"\n  ]"));
        assert!(text.contains("\"lag_samples\": 42"));
        assert!(text.contains("\"drops\": \"d\""));
        assert!(text.contains("\"dstar+drops\": \"d\""));
        assert_eq!(serde_json::from_str::<ShardIndex>(&text).unwrap(), idx);
        // A set built before the lag was recorded reads back with none,
        // and one built before kinds were recorded is base only.
        let old = text
            .replace("\"lag_samples\": 42,", "")
            .replace("\"kinds\": [\n    \"base\",\n    \"drops\"\n  ],", "");
        let parsed = serde_json::from_str::<ShardIndex>(&old).unwrap();
        assert_eq!(parsed.lag_samples, None);
        assert_eq!(parsed.kinds, vec!["base".to_owned()]);
        let base_only = ShardIndex {
            source_sha256s: SourceSha256s {
                siblings: BTreeMap::new(),
                by_set: BTreeMap::new(),
                ..idx.source_sha256s.clone()
            },
            ..idx
        };
        assert!(
            !serde_json::to_string(&base_only)
                .unwrap()
                .contains("siblings")
        );
        let f: ShardFile =
            serde_json::from_str(r#"{"file":"0000.bin","split":"dev","examples":3}"#).unwrap();
        assert_eq!(f.split, Split::Dev);
    }

    /// The exact `index.json` shape written before `modes` existed.
    const V1_INDEX: &str = r#"{"name":"seed-dstar","mode":"dstar","crop_s":2.0,"seed":1,
        "lag_samples":42,"counts":{"train":1000,"dev":80},"onset_share":0.34,"tail_share":0.15,
        "source_sha256s":{"prepared":"p","captured":"c"},
        "example_layout":{"clean16_samples":32000,"deg8_samples":16000,"frames":100,
        "frame_bytes":9,"flags_bytes":9}}"#;

    #[test]
    fn a_single_mode_index_without_modes_reads_as_one_mode() {
        let idx: ShardIndex = serde_json::from_str(V1_INDEX).unwrap();
        assert_eq!(idx.modes, vec![VocoderMode::Dstar]);
        assert_eq!(idx.mode_index_of(VocoderMode::Dstar), Some(0));
        assert_eq!(idx.mode_index_of(VocoderMode::YsfDmr), None);
        assert_eq!(idx.mode_at(0), Some(VocoderMode::Dstar));
        assert_eq!(idx.mode_at(1), None);
        assert_eq!(idx.lag_for(VocoderMode::Dstar), Some(42));
        assert_eq!(idx.lags, [(VocoderMode::Dstar, 42)].into());
        assert_eq!(idx.layout_for(VocoderMode::Dstar), &idx.example_layout);
        assert_eq!(idx.mode_layouts.len(), 1);
        assert_eq!(idx.counts_by_mode[&Split::Train][&VocoderMode::Dstar], 1000);
        assert_eq!(idx.available_by_mode[&Split::Dev][&VocoderMode::Dstar], 80);
        assert!(!idx.balance);
        assert_eq!(idx.kinds, vec!["base"]);
        // Round trip keeps the filled-in fields.
        let again: ShardIndex =
            serde_json::from_str(&serde_json::to_string(&idx).unwrap()).unwrap();
        assert_eq!(again, idx);
    }

    #[test]
    fn mode_bits_share_the_flags_byte_with_onset_and_tail() {
        let f = ExampleLayout::with_mode(ExampleLayout::FLAG_ONSET, 3);
        assert_eq!(ExampleLayout::mode_index(f), 3);
        assert_ne!(f & ExampleLayout::FLAG_ONSET, 0);
        assert_eq!(f & ExampleLayout::FLAG_TAIL, 0);
        let g = ExampleLayout::with_mode(f, 0);
        assert_eq!(g, ExampleLayout::FLAG_ONSET);
        // A pre-modes flags byte has no mode bits: index 0.
        assert_eq!(ExampleLayout::mode_index(0b11), 0);
        assert_eq!(
            ExampleLayout::mode_index(ExampleLayout::with_mode(0, 63)),
            63
        );
        assert_eq!(ExampleLayout::MAX_MODES, 64);
    }

    #[test]
    fn a_multi_mode_layout_is_the_largest_modes_frame_field() {
        use VocoderMode::{Codec2_1600, Codec2_3200, Dstar, YsfDmr};
        let l = ExampleLayout::for_modes(&[Dstar, Codec2_3200], 2.0).unwrap();
        assert_eq!(l, ExampleLayout::for_crop(Dstar, 2.0));
        let l = ExampleLayout::for_modes(&[YsfDmr, Codec2_3200], 2.0).unwrap();
        assert_eq!((l.frames, l.frame_bytes), (100, 8));
        // Codec 2 1600 has half the frames; the field stays the max of each.
        let l = ExampleLayout::for_modes(&[Codec2_1600, Dstar], 2.0).unwrap();
        assert_eq!((l.frames, l.frame_bytes, l.deg8_samples), (100, 9, 16_000));
        assert_eq!(l.example_bytes(), 48_000 * 4 + 900 + 9);
        // Half a second is 25 frames of 20 ms but 12 of 40 ms: refused.
        let err = ExampleLayout::for_modes(&[Dstar, Codec2_1600], 0.5).unwrap_err();
        assert_eq!(err.first, (Dstar, 4000));
        assert_eq!(err.other, (Codec2_1600, 3840));
        assert!(err.to_string().contains("multiple of 40 ms"), "{err}");
        assert_eq!(
            ExampleLayout::for_modes(&[Dstar], 0.5).unwrap(),
            ExampleLayout::for_crop(Dstar, 0.5)
        );
    }
}
