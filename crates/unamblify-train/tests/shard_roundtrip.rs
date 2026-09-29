// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Cross-crate contract: a shard set written by `unamblify_data::shard`
//! is read back bit-for-bit by `unamblify_train::data::shards::ShardLoader`
//! (spec §4: both crates follow `unamblify::ExampleLayout`), and every
//! example in it is one the pipeline loader could have drawn from the
//! same utterance: lag-aligned the same way, onset iff it starts at
//! sample 0, and a tail whose input is receiver garbage over a silent
//! target. Run for an AMBE mode (160-sample frames) and for Codec 2 1600
//! (320-sample frames), so the frame word threads through both crates,
//! and for a set holding both at once, where every example carries its
//! mode and the loaders index the run's mode list.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::Path;

use tch::Device;
use unamblify::{CanaryRecord, CaptureRow, ExampleLayout, Split, UtteranceRow, VocoderMode};
use unamblify_data::DataRoot;
use unamblify_data::shard::{ShardOptions, ShardSet, build};
use unamblify_train::data::Loader;
use unamblify_train::data::pipeline::{Item, load_utterance};
use unamblify_train::data::shards::ShardLoader;

const LAG: usize = 42;

fn sine(freq: f32, rate: u32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
        .collect()
}

/// `n` utterances: clean16 is the sample-doubled 8 kHz sine, the captured
/// side is the same sine delayed by `LAG`, frames encode their index.
fn fixture(root: &DataRoot, mode: VocoderMode, n: usize) {
    fixture_modes(root, &[mode], n, mode.frame_samples());
}

/// [`fixture`] captured in every mode of `modes`: utterance `i` is
/// `(60 + 30 i) × unit` samples, so each mode has its own frame count.
fn fixture_modes(root: &DataRoot, modes: &[VocoderMode], n: usize, unit: usize) {
    let mut prepared = String::new();
    let mut captured: Vec<String> = vec![String::new(); modes.len()];
    for i in 0..n {
        let key = format!("vctk/p2{i:02}_001_mic2");
        let n8 = (60 + 30 * i) * unit;
        let clean8 = sine(120.0 + 35.0 * i as f32, 8_000, n8, 0.25);
        let mut clean16 = Vec::with_capacity(2 * n8);
        for &v in &clean8 {
            clean16.push(v);
            clean16.push(v);
        }
        let mut deg8 = vec![0.0f32; LAG];
        deg8.extend_from_slice(&clean8[..n8 - LAG]);
        let p16 = root.prepared_16k(&key);
        std::fs::create_dir_all(p16.parent().unwrap()).unwrap();
        unamblify_audio::write_wav_s16(&p16, &clean16, 16_000).unwrap();
        unamblify_audio::write_wav_s16(root.prepared_8k(&key), &clean8, 8_000).unwrap();
        let row = UtteranceRow {
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
            prepared_at: "2026-09-10T00:00:00Z".to_owned(),
            parent: None,
            aug: None,
        };
        prepared.push_str(&serde_json::to_string(&row).unwrap());
        prepared.push('\n');
        for (&mode, cap_text) in modes.iter().zip(&mut captured) {
            let frames = n8 / mode.frame_samples();
            let cw = root.captured_wav(mode, &key);
            std::fs::create_dir_all(cw.parent().unwrap()).unwrap();
            unamblify_audio::write_wav_s16(&cw, &deg8, 8_000).unwrap();
            let ambe: Vec<u8> = (0..frames)
                .flat_map(|f| std::iter::repeat_n(f as u8, mode.frame_bytes()))
                .collect();
            std::fs::write(root.captured_ambe(mode, &key), &ambe).unwrap();
            let cap = CaptureRow {
                key: key.clone(),
                mode,
                frames: frames as u32,
                port: "sim:0".to_owned(),
                prodid: if mode.is_software() {
                    "codec2"
                } else {
                    "AMBE3000F"
                }
                .to_owned(),
                version: "V".to_owned(),
                encode_ms: 0,
                decode_ms: 0,
                roundtrip_ms: None,
                sha256_ambe: String::new(),
                sha256_wav: String::new(),
                captured_at: "2026-09-10T00:00:00Z".to_owned(),
                attempts: 1,
                warm_state: None,
                aug: None,
            };
            cap_text.push_str(&serde_json::to_string(&cap).unwrap());
            cap_text.push('\n');
        }
    }
    std::fs::write(root.prepared_manifest(), prepared).unwrap();
    for (&mode, cap_text) in modes.iter().zip(&captured) {
        std::fs::write(root.captured_manifest(mode), cap_text).unwrap();
        let canary = CanaryRecord {
            mode,
            clip: "canary/1khz-and-speech.8k.wav".to_owned(),
            frames_sha256: String::new(),
            frames_first_16: String::new(),
            lag_samples: i32::try_from(LAG).unwrap(),
            prodid: "AMBE3000F".to_owned(),
            version: "V".to_owned(),
            recorded_at: "2026-09-10T00:00:00Z".to_owned(),
            warm_up_frames: 20,
        };
        std::fs::write(
            root.canary_json(mode),
            serde_json::to_string_pretty(&canary).unwrap(),
        )
        .unwrap();
    }
}

fn assert_same_examples(dir: &Path, index_total: usize, split: Split) {
    let set = ShardSet::open(dir).unwrap();
    let loader = ShardLoader::open(dir, split, 7).unwrap();
    assert_eq!(loader.total(), index_total);
    let mut global = 0usize;
    for (fi, file) in set.files.iter().enumerate() {
        for i in 0..file.examples {
            let a = set.read(fi, i).unwrap();
            let b = loader.example(global).unwrap();
            assert_eq!(a.clean16, b.clean16, "clean16 of example {global}");
            assert_eq!(a.deg8, b.deg8, "deg8 of example {global}");
            assert_eq!(a.mask_boundary as usize, b.mask_boundary);
            assert_eq!(a.is_onset(), b.onset);
            assert_eq!(a.is_tail(), b.tail);
            assert_eq!(a.speaker_id, b.speaker, "speaker of example {global}");
            assert_eq!(a.mode_index(), b.mode, "mode of example {global}");
            global += 1;
        }
    }
    assert_eq!(global, index_total);
}

#[test]
fn data_crate_shards_are_read_by_the_train_loader() {
    shards_are_read_by_the_train_loader(VocoderMode::Dstar);
}

#[test]
fn codec2_1600_shards_are_read_by_the_train_loader() {
    shards_are_read_by_the_train_loader(VocoderMode::Codec2_1600);
}

fn shards_are_read_by_the_train_loader(mode: VocoderMode) {
    let tmp = tempfile::tempdir().unwrap();
    let root = DataRoot::new(tmp.path());
    fixture(&root, mode, 5);
    let layout = ExampleLayout::for_crop(mode, 0.5);
    let n8 = i64::try_from(layout.deg8_samples).unwrap();
    assert_eq!(
        layout.frames,
        if mode == VocoderMode::Codec2_1600 {
            12
        } else {
            25
        }
    );

    let mut opts = ShardOptions::new("rt", mode);
    opts.crop_s = 0.5;
    opts.examples_per_file = 3;
    let summary = build(&root, &opts).unwrap();
    let total: u64 = summary.counts.values().sum();
    assert!(total > 3, "expected several files, got {total} examples");
    let dir = root.shards("rt");

    assert_same_examples(&dir, total as usize, Split::Train);

    // The loader's batches carry the same content and the speaker ids.
    let mut loader = ShardLoader::open(&dir, Split::Train, 1).unwrap();
    let n_train = loader.len();
    assert_eq!(
        n_train as u64,
        summary.counts.get(&Split::Train).copied().unwrap_or(0)
    );
    let batch = loader.next_batch(2, Device::Cpu).unwrap();
    assert_eq!(batch.clean16.size(), [2, 1, 2 * n8]);
    assert_eq!(batch.deg8.size(), [2, 1, n8]);
    assert_eq!(batch.mask.size(), [2, 1, 2 * n8]);
    let speakers = Vec::<i64>::try_from(&batch.speaker).unwrap();
    for s in speakers {
        let s = u32::try_from(s).unwrap();
        assert!(
            (0..5).any(|i| unamblify::speaker_id(&format!("p2{i:02}")) == s),
            "speaker id {s:#x} is not one of the fixture speakers"
        );
    }

    // Alignment through the lag survives the packing: for a non-tail
    // example the doubled clean16 matches deg8 shifted by the lag.
    let dev = ShardLoader::open(&dir, Split::Dev, 1).unwrap();
    assert_eq!(dev.lag(), i32::try_from(LAG).unwrap());
    let ex = dev
        .example(unamblify_train::data::shards::split_range(dev.index(), Split::Dev).start)
        .unwrap();
    let speech = ex.mask_boundary / 2;
    let mut mismatches = 0usize;
    for k in 0..speech.min(ex.deg8.len()) {
        if (ex.clean16[2 * k] - ex.deg8[k]).abs() > 2.0 / 32_768.0 {
            mismatches += 1;
        }
    }
    assert!(mismatches <= LAG, "{mismatches} misaligned samples");
}

/// Every example the data crate packs is a crop the pipeline loader would
/// produce from the same aligned utterance: the speech part is a
/// frame-aligned window of `load_utterance`'s output (lag undone the same
/// way), `onset` holds iff that window starts at sample 0, and a tail
/// example carries garbage input past the mask boundary over a zero
/// target, never digital silence.
#[test]
fn packed_examples_are_pipeline_crops() {
    packed_examples_are_crops_of(VocoderMode::Dstar);
}

#[test]
fn packed_codec2_1600_examples_are_pipeline_crops() {
    packed_examples_are_crops_of(VocoderMode::Codec2_1600);
}

fn packed_examples_are_crops_of(mode: VocoderMode) {
    let tmp = tempfile::tempdir().unwrap();
    let root = DataRoot::new(tmp.path());
    fixture(&root, mode, 5);
    let lag = i32::try_from(LAG).unwrap();
    let mut opts = ShardOptions::new("pc", mode);
    opts.crop_s = 0.5;
    opts.onset_share = 0.3;
    opts.tail_share = 0.4;
    build(&root, &opts).unwrap();
    let dir = root.shards("pc");
    let set = ShardSet::open(&dir).unwrap();
    // Speaker id → key (one speaker per fixture utterance).
    let by_speaker: std::collections::HashMap<u32, (String, String)> = (0..5)
        .map(|i| {
            let spk = format!("p2{i:02}");
            (
                unamblify::speaker_id(&spk),
                (format!("vctk/{spk}_001_mic2"), spk),
            )
        })
        .collect();
    let (mut onsets, mut tails, mut randoms) = (0, 0, 0);
    for (fi, f) in set.files.iter().enumerate() {
        for i in 0..f.examples {
            let ex = set.read(fi, i).unwrap();
            let (key, spk) = &by_speaker[&ex.speaker_id];
            let item = Item::plain(mode, key, spk, Split::Train);
            let utt = load_utterance(root.path(), &item, lag).unwrap();
            let n8 = ex.deg8.len();
            let speech8 = ex.mask_boundary as usize / 2;
            let head = &ex.deg8[..speech8];
            let start = (0..=utt.deg8.len().saturating_sub(speech8))
                .step_by(mode.frame_samples())
                .find(|&s| utt.deg8[s..s + speech8] == *head)
                .unwrap_or_else(|| {
                    panic!("example {fi}/{i}: not a window of the aligned utterance")
                });
            assert_eq!(
                &ex.clean16[..2 * speech8],
                &utt.clean16[2 * start..2 * start + 2 * speech8],
                "clean target is the same window"
            );
            assert_eq!(ex.is_onset(), start == 0, "onset iff the crop starts at 0");
            if ex.is_tail() {
                tails += 1;
                assert!(speech8 < n8);
                assert!(ex.clean16[2 * speech8..].iter().all(|&v| v == 0.0));
                assert!(
                    ex.deg8[speech8..].iter().any(|&v| v != 0.0),
                    "tail input is garbage"
                );
            } else {
                assert_eq!(ex.mask_boundary as usize, 2 * n8);
                if ex.is_onset() {
                    onsets += 1;
                } else {
                    randoms += 1;
                }
            }
        }
    }
    assert!(
        onsets > 0 && tails > 0 && randoms > 0,
        "{onsets} {tails} {randoms}"
    );
}

/// A set of D-STAR and Codec 2 1600 at once: the data crate packs both
/// (each in its own frame word, balanced per split), the train loader
/// reads every example back with its mode, a one-mode subset of the set
/// is that mode's examples renumbered, and each packed example is a
/// frame-aligned crop of its own mode's aligned utterance.
#[test]
fn a_two_mode_set_round_trips_and_filters_by_mode() {
    use VocoderMode::{Codec2_1600, Dstar};
    let tmp = tempfile::tempdir().unwrap();
    let root = DataRoot::new(tmp.path());
    let modes = [Dstar, Codec2_1600];
    fixture_modes(&root, &modes, 5, 320);
    let mut opts = ShardOptions::for_modes("mixed", modes.to_vec());
    opts.crop_s = 1.0;
    opts.examples_per_file = 4;
    let summary = build(&root, &opts).unwrap();
    let total: u64 = summary.counts.values().sum();
    let dir = root.shards("mixed");
    let set = ShardSet::open(&dir).unwrap();
    assert_eq!(set.index.modes, modes);
    assert!(set.index.balance);
    for split in [Split::Train, Split::Dev] {
        let per = &set.index.counts_by_mode[&split];
        assert_eq!(per[&Dstar], per[&Codec2_1600], "{split} balanced");
    }
    assert_same_examples(&dir, total as usize, Split::Train);

    // Every stored example is a crop of its own mode's utterance.
    let by_speaker: std::collections::HashMap<u32, (String, String)> = (0..5)
        .map(|i| {
            let spk = format!("p2{i:02}");
            (
                unamblify::speaker_id(&spk),
                (format!("vctk/{spk}_001_mic2"), spk),
            )
        })
        .collect();
    let lag = i32::try_from(LAG).unwrap();
    let mut seen = [0usize; 2];
    for (fi, f) in set.files.iter().enumerate() {
        for i in 0..f.examples {
            let ex = set.read(fi, i).unwrap();
            let mode = set.mode_of(&ex).unwrap();
            seen[usize::from(ex.mode_index())] += 1;
            let (key, spk) = &by_speaker[&ex.speaker_id];
            let utt = load_utterance(root.path(), &Item::plain(mode, key, spk, Split::Train), lag)
                .unwrap();
            let speech8 = ex.mask_boundary as usize / 2;
            let head = &ex.deg8[..speech8];
            let start = (0..=utt.deg8.len().saturating_sub(speech8))
                .step_by(mode.frame_samples())
                .find(|&s| utt.deg8[s..s + speech8] == *head)
                .unwrap_or_else(|| panic!("{mode} example {fi}/{i}: not a window"));
            assert_eq!(ex.is_onset(), start == 0);
            assert_eq!(ex.frames.len(), 50 * 9, "the largest mode's field");
        }
    }
    assert_eq!(seen[0], seen[1]);

    // The train loader over the run's mode list, and over a subset.
    let mut both = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&modes)).unwrap();
    assert_eq!(both.modes(), &modes);
    let batch = both.next_batch(4, Device::Cpu).unwrap();
    assert_eq!(batch.mode.size(), [4]);
    assert_eq!(batch.deg8.size(), [4, 1, 8_000]);
    let c2 = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[Codec2_1600])).unwrap();
    assert_eq!(
        c2.len(),
        seen[1] - set.index.counts_by_mode[&Split::Dev][&Codec2_1600] as usize
    );
    let train = unamblify_train::data::shards::split_range(c2.index(), Split::Train);
    let mut found = 0;
    for i in train {
        if let Ok(ex) = c2.raw_example(i) {
            assert_eq!(ex.mode, 0, "renumbered to the subset's index");
            found += 1;
        }
    }
    assert_eq!(found, c2.len());
    let err = ShardLoader::open_modes(&dir, Split::Train, 1, Some(&[VocoderMode::YsfDmr]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("holds modes dstar,codec2-1600"), "{err}");
}
