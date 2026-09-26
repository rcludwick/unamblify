// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `verify` (spec §3): recompute the sha256s of a sample of captured
//! utterances and check their frame counts against the files, in the
//! mode's own frame words (`frame_bytes`, `frame_samples`), and that the
//! row names the codec its family expects.

use serde::Serialize;
use unamblify::aug::AugKind;
use unamblify::{CaptureRow, VOCODER_SAMPLE_RATE, VocoderFamily, VocoderMode};
use unamblify_audio::read as read_audio;

use crate::util::{Rng, read_jsonl, sha256_file};
use crate::{CaptureDir, DataRoot, Result};

/// Options.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Mode.
    pub mode: VocoderMode,
    /// A decode-only sibling (`captured/<mode>+<kind>/`) instead of the
    /// base capture.
    pub kind: Option<AugKind>,
    /// Rows to check (0 = all).
    pub sample: usize,
    /// Seed of the sample.
    pub seed: u64,
}

/// One problem found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Problem {
    /// Utterance key.
    pub key: String,
    /// What is wrong.
    pub what: String,
}

/// The report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct VerifyReport {
    /// Rows in the manifest.
    pub rows: u64,
    /// Rows checked.
    pub checked: u64,
    /// Rows that passed.
    pub ok: u64,
    /// Problems.
    pub problems: Vec<Problem>,
}

/// Check one row of the base capture.
#[must_use]
pub fn check_row(root: &DataRoot, row: &CaptureRow) -> Vec<String> {
    check_row_in(&root.capture_dir(row.mode, None), row)
}

/// Check one row of any capture set.
#[must_use]
pub fn check_row_in(dir: &CaptureDir, row: &CaptureRow) -> Vec<String> {
    let mut problems = Vec::new();
    let mode = row.mode;
    let ambe = dir.ambe(&row.key);
    let wav = dir.decoded(&row.key);
    match std::fs::metadata(&ambe) {
        Ok(m) => {
            let want = u64::from(row.frames) * mode.frame_bytes() as u64;
            if m.len() != want {
                problems.push(format!(
                    ".ambe is {} bytes, {} frames need {want}",
                    m.len(),
                    row.frames
                ));
            }
            match sha256_file(&ambe) {
                Ok(h) if h != row.sha256_ambe => problems.push("sha256_ambe mismatch".to_owned()),
                Ok(_) => {}
                Err(e) => problems.push(format!(".ambe unreadable: {e}")),
            }
        }
        Err(e) => problems.push(format!(".ambe missing: {e}")),
    }
    match read_audio(&wav) {
        Ok(a) => {
            if a.rate != VOCODER_SAMPLE_RATE {
                problems.push(format!(".wav is {} Hz", a.rate));
            }
            let want = row.frames as usize * mode.frame_samples();
            if a.samples.len() != want {
                problems.push(format!(
                    ".wav has {} samples, {} frames need {want}",
                    a.samples.len(),
                    row.frames
                ));
            }
            match sha256_file(&wav) {
                Ok(h) if h != row.sha256_wav => problems.push("sha256_wav mismatch".to_owned()),
                Ok(_) => {}
                Err(e) => problems.push(format!(".wav unreadable: {e}")),
            }
        }
        Err(e) => problems.push(format!(".wav unreadable: {e}")),
    }
    if !row.prodid.starts_with(expected_prodid(mode)) {
        problems.push(format!(
            "prodid {:?} (expected {}…)",
            row.prodid,
            expected_prodid(mode)
        ));
    }
    problems
}

/// What a row's `prodid` must start with for its mode's family.
#[must_use]
pub const fn expected_prodid(mode: VocoderMode) -> &'static str {
    match mode.family() {
        VocoderFamily::Ambe => "AMBE3000",
        VocoderFamily::Codec2 => crate::vocoder::CODEC2_PRODID,
    }
}

/// Run the check.
pub fn run(root: &DataRoot, opts: &VerifyOptions) -> Result<VerifyReport> {
    let dir = root.capture_dir(opts.mode, opts.kind);
    let rows: Vec<CaptureRow> = read_jsonl(&dir.manifest())?;
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    if opts.sample > 0 && opts.sample < rows.len() {
        Rng::new(opts.seed).shuffle(&mut idx);
        idx.truncate(opts.sample);
        idx.sort_unstable();
    }
    let mut report = VerifyReport {
        rows: rows.len() as u64,
        ..VerifyReport::default()
    };
    for i in idx {
        let row = &rows[i];
        report.checked += 1;
        let problems = check_row_in(&dir, row);
        if problems.is_empty() {
            report.ok += 1;
        } else {
            for what in problems {
                log::error!("{}: {what}", row.key);
                report.problems.push(Problem {
                    key: row.key.clone(),
                    what,
                });
            }
        }
    }
    log::info!(
        "verify {}: {} of {} rows checked, {} ok, {} problems",
        dir.name(),
        report.checked,
        report.rows,
        report.ok,
        report.problems.len()
    );
    Ok(report)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::util::{JsonlWriter, now_rfc3339, sha256_hex};
    use unamblify_audio::write_wav_s16;

    #[test]
    fn verify_passes_good_rows_and_names_bad_ones() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let mode = VocoderMode::Dstar;
        let mut w = JsonlWriter::open(&root.captured_manifest(mode)).unwrap();
        for i in 0..3u32 {
            let key = format!("ljspeech/LJ001-000{i}");
            let ambe = vec![i as u8; 9 * 5];
            let a = root.captured_ambe(mode, &key);
            std::fs::create_dir_all(a.parent().unwrap()).unwrap();
            std::fs::write(&a, &ambe).unwrap();
            let wav = root.captured_wav(mode, &key);
            write_wav_s16(&wav, &vec![0.1f32; 800], 8_000).unwrap();
            w.append(&CaptureRow {
                key,
                mode,
                frames: 5,
                port: "p".to_owned(),
                prodid: "AMBE3000F".to_owned(),
                version: "V".to_owned(),
                encode_ms: 1,
                decode_ms: 1,
                roundtrip_ms: None,
                sha256_ambe: sha256_hex(&ambe),
                sha256_wav: sha256_file(&wav).unwrap(),
                captured_at: now_rfc3339(),
                attempts: 1,
                warm_state: None,
                aug: None,
            })
            .unwrap();
        }
        let opts = VerifyOptions {
            mode,
            kind: None,
            sample: 0,
            seed: 1,
        };
        let r = run(&root, &opts).unwrap();
        assert_eq!((r.rows, r.checked, r.ok), (3, 3, 3));
        assert!(r.problems.is_empty());

        std::fs::write(root.captured_ambe(mode, "ljspeech/LJ001-0001"), [0u8; 44]).unwrap();
        std::fs::remove_file(root.captured_wav(mode, "ljspeech/LJ001-0002")).unwrap();
        let r = run(&root, &opts).unwrap();
        assert_eq!(r.ok, 1);
        assert_eq!(r.problems.len(), 3, "{:?}", r.problems);
        assert!(r.problems[0].what.contains("44 bytes"));
        assert!(r.problems[1].what.contains("sha256_ambe"));
        assert!(r.problems[2].what.contains(".wav unreadable"));
        let r = run(&root, &VerifyOptions { sample: 2, ..opts }).unwrap();
        assert_eq!(r.checked, 2);
    }

    #[test]
    fn codec2_1600_rows_are_checked_in_320_sample_frames() {
        let dir = tempfile::tempdir().unwrap();
        let root = DataRoot::new(dir.path());
        let mode = VocoderMode::Codec2_1600;
        let mut w = JsonlWriter::open(&root.captured_manifest(mode)).unwrap();
        let key = "vctk/p225_001_mic2";
        let ambe = vec![7u8; 8 * 5];
        let a = root.captured_ambe(mode, key);
        std::fs::create_dir_all(a.parent().unwrap()).unwrap();
        std::fs::write(&a, &ambe).unwrap();
        let wav = root.captured_wav(mode, key);
        write_wav_s16(&wav, &vec![0.1f32; 5 * 320], 8_000).unwrap();
        let mut row = CaptureRow {
            key: key.to_owned(),
            mode,
            frames: 5,
            port: "codec2:0".to_owned(),
            prodid: "codec2".to_owned(),
            version: "0.3.1".to_owned(),
            encode_ms: 1,
            decode_ms: 1,
            roundtrip_ms: None,
            sha256_ambe: sha256_hex(&ambe),
            sha256_wav: sha256_file(&wav).unwrap(),
            captured_at: now_rfc3339(),
            attempts: 1,
            warm_state: None,
            aug: None,
        };
        w.append(&row).unwrap();
        w.sync().unwrap();
        let opts = VerifyOptions {
            mode,
            kind: None,
            sample: 0,
            seed: 1,
        };
        let r = run(&root, &opts).unwrap();
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        // A chip prodid on a software row is a problem, and vice versa.
        row.prodid = "AMBE3000F".to_owned();
        let p = check_row(&root, &row);
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("prodid"));
        assert_eq!(expected_prodid(VocoderMode::YsfDmr), "AMBE3000");
    }
}
