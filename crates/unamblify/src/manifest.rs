// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Manifest rows written by the data stages (spec §2): one JSON object per
//! line in `prepared/manifest.jsonl` and `captured/<mode>/manifest.jsonl`,
//! plus the day-one canary record. Timestamps are RFC 3339 UTC strings;
//! hashes are lowercase hex SHA-256 of the file bytes.

use serde::{Deserialize, Serialize};

use crate::aug::{AugRecord, CaptureAug};
use crate::{Split, VocoderMode};

/// One prepared utterance (`prepared/manifest.jsonl`). An augmented twin
/// (`docs/design/data-pipeline.md`, stage 3) is a row like any other whose
/// `key` carries a `+` suffix, whose `parent` names the utterance it was
/// made from — the parent's clean files are the twin's training target —
/// and whose `aug` records what was done.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtteranceRow {
    /// Utterance key (see [`crate::key`]); `prepared/<key>.16k.wav` exists.
    pub key: String,
    /// Corpus id, the key's leading directory.
    pub corpus: String,
    /// Corpus speaker id (`p225`, `84`, `LJ`).
    pub speaker: String,
    /// `"F"` / `"M"` when the corpus says; absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    /// Assigned once by [`crate::split::assign`].
    pub split: Split,
    /// Length after trimming, seconds.
    pub duration_s: f64,
    /// Sample rate of the source file, Hz.
    pub src_rate: u32,
    /// Source path relative to the data root.
    pub src_path: String,
    /// SPDX-ish licence id of the corpus (`CC-BY-4.0`).
    pub licence: String,
    /// Active-speech RMS of the source before normalisation, dBFS.
    pub rms_dbfs_in: f64,
    /// Gain applied to reach the target loudness, dB (clamped to ±20).
    pub gain_db: f64,
    /// Leading silence removed, seconds.
    pub trim_lead_s: f64,
    /// Trailing silence removed, seconds.
    pub trim_tail_s: f64,
    /// SHA-256 of `<key>.16k.wav`.
    pub sha256_16k: String,
    /// SHA-256 of `<key>.8k.wav`.
    pub sha256_8k: String,
    /// When the row was written (RFC 3339 UTC).
    pub prepared_at: String,
    /// For a twin: the key of the utterance it was made from, whose clean
    /// files are this row's training target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// For a twin: what was done to the parent to make this input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aug: Option<AugRecord>,
}

impl UtteranceRow {
    /// The key whose prepared files are this row's clean target: the
    /// parent for a twin, the row's own key otherwise.
    #[must_use]
    pub fn target_key(&self) -> &str {
        self.parent.as_deref().unwrap_or(&self.key)
    }
}

/// One captured utterance (`captured/<mode>/manifest.jsonl`, or a
/// decode-only sibling `captured/<mode>+<kind>/manifest.jsonl`, whose rows
/// carry `aug`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureRow {
    /// Utterance key.
    pub key: String,
    /// Vocoder mode the chip was configured for.
    pub mode: VocoderMode,
    /// Whole 20 ms frames fed to the encoder (input zero-padded to fit).
    pub frames: u32,
    /// Serial port the stick was on.
    pub port: String,
    /// The chip's PRODID reply, e.g. `AMBE3000F`.
    pub prodid: String,
    /// The chip's VERSTRING reply.
    pub version: String,
    /// Wall time of the encode pass, ms. Zero when the two directions ran
    /// interleaved — see `roundtrip_ms`.
    pub encode_ms: u64,
    /// Wall time of the decode pass, ms. Zero when the two directions ran
    /// interleaved — see `roundtrip_ms`.
    pub decode_ms: u64,
    /// Wall time of the whole utterance when the chip ran both directions
    /// at once (`capture` on an AMBE mode, the default). The two passes
    /// overlap, so there is no honest way to split this into an encode
    /// time and a decode time: `encode_ms` and `decode_ms` are left at 0
    /// rather than invented. Absent on a row whose passes were separate
    /// (a software capture, `capture --sequential`, the decode-only
    /// `augment` stage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roundtrip_ms: Option<u64>,
    /// SHA-256 of `<key>.ambe`.
    pub sha256_ambe: String,
    /// SHA-256 of `<key>.wav` (decoded 8 kHz s16).
    pub sha256_wav: String,
    /// When the row was written (RFC 3339 UTC).
    pub captured_at: String,
    /// How many attempts it took (1 = first try).
    pub attempts: u32,
    /// The encoder-state condition this utterance was captured under, when
    /// warm-up mixing was on: `cold` (chip reset immediately before, so
    /// the first frames carry the post-init pitch-lock transient) or
    /// `warm` (the encoder locked onto this voice first). Absent when
    /// mixing was off or for a software mode, whose encoder is stateless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_state: Option<String>,
    /// For a sibling capture: the channel mutation this row's frames went
    /// through before the decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aug: Option<CaptureAug>,
}

/// The day-one canary (`captured/<mode>/canary.json`): what the chip produced
/// for the reference clip, so every later re-encode can be compared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanaryRecord {
    /// Vocoder mode the record belongs to.
    pub mode: VocoderMode,
    /// Path of the canary clip relative to the data root.
    pub clip: String,
    /// SHA-256 of the concatenated channel frames.
    pub frames_sha256: String,
    /// Hex of the first 16 bytes of channel data, for a human at a glance.
    pub frames_first_16: String,
    /// Lag of the decoded canary relative to its 8 kHz input, samples
    /// (positive = decoded output is late).
    pub lag_samples: i32,
    /// The chip's PRODID reply.
    pub prodid: String,
    /// The chip's VERSTRING reply.
    pub version: String,
    /// When the record was written (RFC 3339 UTC).
    pub recorded_at: String,
    /// Frames of the clip pushed through the encoder (and discarded)
    /// between a chip reset and the canary encode whose hash this is.
    /// Every re-check starts from the same reset + preamble, and a harness
    /// with a different preamble must refuse the record rather than fail
    /// the run on a mismatch it caused itself.
    #[serde(default = "default_warm_up_frames")]
    pub warm_up_frames: usize,
}

/// The preamble length of records written before the field existed.
#[must_use]
pub const fn default_warm_up_frames() -> usize {
    20
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTT: &str = r#"{"key":"vctk/p225_001_mic2","corpus":"vctk","speaker":"p225","gender":"F",
 "split":"train","duration_s":3.12,"src_rate":48000,"src_path":"raw/vctk/wav48_silence_trimmed/p225/p225_001_mic2.flac",
 "licence":"CC-BY-4.0","rms_dbfs_in":-21.3,"gain_db":-4.7,"trim_lead_s":0.11,"trim_tail_s":0.20,
 "sha256_16k":"a","sha256_8k":"b","prepared_at":"2026-09-10T03:00:00Z"}"#;

    const CAP: &str = r#"{"key":"vctk/p225_001_mic2","mode":"dstar","frames":156,"port":"/dev/cu.usbserial-DK0EOQVS",
 "prodid":"AMBE3000F","version":"V121.E100.XXXX.C110.G514.R014.A0030608.C0020208",
 "encode_ms":2480,"decode_ms":1160,"sha256_ambe":"c","sha256_wav":"d","captured_at":"2026-09-10T04:00:00Z","attempts":1}"#;

    const CANARY: &str = r#"{"mode":"dstar","clip":"canary/1khz-and-speech.8k.wav","frames_sha256":"e","frames_first_16":"9e8d",
 "lag_samples":42,"prodid":"AMBE3000F","version":"V121","recorded_at":"2026-09-10T04:00:00Z"}"#;

    #[test]
    fn utterance_row_round_trips_the_spec_example() {
        let row: UtteranceRow = serde_json::from_str(UTT).unwrap();
        assert_eq!(row.split, Split::Train);
        assert_eq!(row.gender.as_deref(), Some("F"));
        assert_eq!(row.src_rate, 48_000);
        let back: UtteranceRow =
            serde_json::from_str(&serde_json::to_string(&row).unwrap()).unwrap();
        assert_eq!(back, row);
        let no_gender: UtteranceRow =
            serde_json::from_str(&UTT.replace(r#""gender":"F","#, "")).unwrap();
        assert_eq!(no_gender.gender, None);
        assert!(
            !serde_json::to_string(&no_gender)
                .unwrap()
                .contains("gender")
        );
        assert_eq!(row.parent, None);
        assert_eq!(row.aug, None);
        assert_eq!(row.target_key(), "vctk/p225_001_mic2");
        let text = serde_json::to_string(&row).unwrap();
        assert!(!text.contains("parent") && !text.contains("\"aug\""));
        let twin = UtteranceRow {
            key: "vctk/p225_001_mic2+n0173".to_owned(),
            parent: Some("vctk/p225_001_mic2".to_owned()),
            aug: Some(crate::aug::AugRecord {
                seed: 1,
                noise: None,
                chain: Some(crate::aug::ChainParams::fixed_overdrive()),
                under: None,
                post_gain_db: -2.5,
            }),
            ..row.clone()
        };
        assert_eq!(twin.target_key(), "vctk/p225_001_mic2");
        let back: UtteranceRow =
            serde_json::from_str(&serde_json::to_string(&twin).unwrap()).unwrap();
        assert_eq!(back, twin);
    }

    #[test]
    fn capture_row_round_trips_the_spec_example() {
        let row: CaptureRow = serde_json::from_str(CAP).unwrap();
        assert_eq!(row.mode, VocoderMode::Dstar);
        assert_eq!(row.frames, 156);
        assert_eq!(row.aug, None);
        let back: CaptureRow = serde_json::from_str(&serde_json::to_string(&row).unwrap()).unwrap();
        assert_eq!(back, row);
        assert!(!serde_json::to_string(&row).unwrap().contains("aug"));
        // A row written before interleaving existed has no round trip,
        // and a row that never split its passes writes only the one
        // number — an interleaved capture must not invent an encode /
        // decode split.
        assert_eq!(row.roundtrip_ms, None);
        assert!(
            !serde_json::to_string(&row)
                .unwrap()
                .contains("roundtrip_ms")
        );
        let inter: CaptureRow = serde_json::from_str(&CAP.replace(
            r#""encode_ms":2480,"decode_ms":1160"#,
            r#""encode_ms":0,"decode_ms":0,"roundtrip_ms":2070"#,
        ))
        .unwrap();
        assert_eq!(inter.encode_ms, 0);
        assert_eq!(inter.decode_ms, 0);
        assert_eq!(inter.roundtrip_ms, Some(2_070));
        let back: CaptureRow =
            serde_json::from_str(&serde_json::to_string(&inter).unwrap()).unwrap();
        assert_eq!(back, inter);
    }

    #[test]
    fn canary_round_trips_the_spec_example() {
        let rec: CanaryRecord = serde_json::from_str(CANARY).unwrap();
        assert_eq!(rec.lag_samples, 42);
        assert_eq!(
            rec.warm_up_frames, 20,
            "records without the field are the 20-frame preamble"
        );
        let back: CanaryRecord =
            serde_json::from_str(&serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(back, rec);
    }
}
