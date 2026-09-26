// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The deterministic, speaker-disjoint train / dev / test assignment.
//!
//! The rule (spec §2) is applied once, at prepare time, and recorded per
//! [`UtteranceRow`](crate::UtteranceRow) so nothing downstream recomputes it.
//! It is here, not in the data crate, so the trainer and the dashboard can
//! reason about it without depending on the pipeline.
//!
//! * `test` = VoiceBank's two held-out speakers, `p232` and `p257`, in every
//!   corpus that has them (VCTK and VoiceBank-DEMAND share the speaker
//!   namespace) + the LibriTTS-R `test-*` readers whose `sha1(reader)`
//!   first byte is below `0x14` (≈ 8 %).
//! * `dev` = the LibriTTS-R `dev-*` readers below the same hash limit,
//!   plus the readers the eval clips are drawn from ([`LIBRITTS_EVAL_READERS`]),
//!   plus VCTK-namespace speakers whose `sha1(speaker)` first byte is
//!   below `0x14`.
//! * everything else is `train`; LJSpeech is always `train`.
//!
//! LibriTTS-R's own `dev-*` / `test-*` subsets are *not* held out whole.
//! They were tier 0 — captured through the chip first — and holding
//! every reader of them out put 58 % of the captured AMBE hours in
//! splits no run trains on, to serve an evaluation that uses a few dozen
//! clips. Readers are disjoint across subsets, so a hash over the reader
//! keeps the split speaker-disjoint exactly as the VCTK rule does.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::ParseEnumError;

/// Which partition an utterance belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Used for gradient steps.
    Train,
    /// Used for checkpoint selection and the eval clips.
    Dev,
    /// Held out; reported once, at the end.
    Test,
}

impl Split {
    /// Every split, in the order counts are reported.
    pub const ALL: [Self; 3] = [Self::Train, Self::Dev, Self::Test];

    /// Stable lowercase identifier used in manifests and the CLI.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Dev => "dev",
            Self::Test => "test",
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Split {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|v| v.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "split",
                input: s.to_owned(),
            })
    }
}

/// Corpus identifier of the LibriTTS-R corpus (the leading key directory).
pub const CORPUS_LIBRITTS_R: &str = "libritts_r";
/// Corpus identifier of VoiceBank-DEMAND.
pub const CORPUS_VOICEBANK_DEMAND: &str = "voicebank_demand";
/// Corpus identifier of VCTK.
pub const CORPUS_VCTK: &str = "vctk";
/// Corpus identifier of LJSpeech.
pub const CORPUS_LJSPEECH: &str = "ljspeech";
/// Corpus identifier of Hi-Fi TTS. Ten speakers, all `train`: with so few,
/// hashing one into `dev` would hold out a tenth of the corpus for nothing
/// the LibriTTS-R readers do not already give the eval.
pub const CORPUS_HIFI_TTS: &str = "hifi_tts";

/// VoiceBank-DEMAND's two held-out test speakers; `test` wherever they appear.
pub const TEST_SPEAKERS: [&str; 2] = ["p232", "p257"];

/// A VCTK-namespace speaker is `dev` when the first byte of `sha1(speaker)`
/// is below this value (`0x14 / 0x100` ≈ 7.8 %); a LibriTTS-R `dev-*` /
/// `test-*` reader stays in its subset's split under the same rule.
pub const DEV_HASH_LIMIT: u8 = 0x14;
/// LibriTTS-R readers the eval clips (`configs/eval-clips.txt`) come
/// from: always `dev`, whatever their hash, so the clips every run is
/// scored on stay held out.
pub const LIBRITTS_EVAL_READERS: [&str; 6] = ["1272", "5338", "84", "2803", "1462", "777"];

/// Assign the split for one utterance.
///
/// `corpus` is the leading directory of the utterance key (see
/// [`crate::key`]), `speaker` the corpus's speaker id (`p225`, `84`, `LJ`),
/// and `subset` the corpus subset when the corpus has one (LibriTTS-R
/// `train-clean-100` / `dev-clean` / `test-clean`; VoiceBank `train` /
/// `test`), else `None`.
///
/// The function is total: an unknown corpus falls through to the general
/// rules and ends up in `train` unless its speaker is a held-out one.
#[must_use]
pub fn assign(corpus: &str, speaker: &str, subset: Option<&str>) -> Split {
    if TEST_SPEAKERS.contains(&speaker) {
        return Split::Test;
    }
    match corpus {
        CORPUS_LIBRITTS_R => match subset {
            Some(_) if LIBRITTS_EVAL_READERS.contains(&speaker) => Split::Dev,
            Some(s) if s.starts_with("test") && speaker_hash_byte(speaker) < DEV_HASH_LIMIT => {
                Split::Test
            }
            Some(s) if s.starts_with("dev") && speaker_hash_byte(speaker) < DEV_HASH_LIMIT => {
                Split::Dev
            }
            _ => Split::Train,
        },
        CORPUS_VCTK | CORPUS_VOICEBANK_DEMAND => {
            if speaker_hash_byte(speaker) < DEV_HASH_LIMIT {
                Split::Dev
            } else {
                Split::Train
            }
        }
        // LJSpeech (one speaker) and anything unknown.
        _ => Split::Train,
    }
}

/// First byte of `sha1(speaker)`, the quantity the `dev` rule thresholds.
#[must_use]
pub fn speaker_hash_byte(speaker: &str) -> u8 {
    let digest = Sha1::digest(speaker.as_bytes());
    digest[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_out_speakers_are_test_in_every_corpus() {
        for corpus in [CORPUS_VCTK, CORPUS_VOICEBANK_DEMAND, "something_else"] {
            assert_eq!(assign(corpus, "p232", None), Split::Test, "{corpus}");
            assert_eq!(
                assign(corpus, "p257", Some("train")),
                Split::Test,
                "{corpus}"
            );
        }
    }

    /// LibriTTS-R `dev-*` / `test-*` readers stay in their subset's split
    /// only below the hash limit; the rest train. The `train-*` subsets
    /// always train. Hash bytes pinned with `hashlib.sha1`.
    #[test]
    fn libritts_r_dev_and_test_readers_are_hashed() {
        // 5639 (test-clean) hashes to 0x04 and 422 (dev-clean) to 0x02,
        // below the limit; 1089 to 0x78 and 1188 to 0xde, above it.
        assert_eq!(speaker_hash_byte("5639"), 0x04);
        assert_eq!(speaker_hash_byte("422"), 0x02);
        assert_eq!(speaker_hash_byte("1089"), 0x78);
        assert_eq!(speaker_hash_byte("1188"), 0xde);
        assert_eq!(
            assign(CORPUS_LIBRITTS_R, "5639", Some("test-clean")),
            Split::Test
        );
        assert_eq!(
            assign(CORPUS_LIBRITTS_R, "1089", Some("test-clean")),
            Split::Train
        );
        assert_eq!(
            assign(CORPUS_LIBRITTS_R, "422", Some("dev-clean")),
            Split::Dev
        );
        assert_eq!(
            assign(CORPUS_LIBRITTS_R, "1188", Some("dev-clean")),
            Split::Train
        );
        for subset in ["train-clean-100", "train-clean-360"] {
            assert_eq!(assign(CORPUS_LIBRITTS_R, "422", Some(subset)), Split::Train);
        }
        assert_eq!(assign(CORPUS_LIBRITTS_R, "422", None), Split::Train);
    }

    /// The readers the eval clips come from are `dev` whatever their hash
    /// — 84 hashes to 0xbe, well above the limit.
    #[test]
    fn eval_readers_stay_dev() {
        assert_eq!(speaker_hash_byte("84"), 0xbe);
        for r in LIBRITTS_EVAL_READERS {
            assert_eq!(
                assign(CORPUS_LIBRITTS_R, r, Some("dev-clean")),
                Split::Dev,
                "{r}"
            );
        }
        // Even from a test subset: dev, so the clip can be an eval clip.
        assert_eq!(
            assign(CORPUS_LIBRITTS_R, "84", Some("test-clean")),
            Split::Dev
        );
    }

    #[test]
    fn ljspeech_is_train_only() {
        assert_eq!(assign(CORPUS_LJSPEECH, "LJ", None), Split::Train);
        assert_eq!(assign(CORPUS_LJSPEECH, "LJ", Some("dev")), Split::Train);
    }

    /// Fixed expectations computed with `hashlib.sha1` — the first byte of
    /// the digest is written beside each speaker. Any change here changes
    /// every manifest, so the values are pinned.
    #[test]
    fn vctk_namespace_hash_rule_is_pinned() {
        let expected = [
            ("p225", 0xf3, Split::Train),
            ("p226", 0x35, Split::Train),
            ("p228", 0x00, Split::Dev),
            ("p239", 0x13, Split::Dev),
            ("p244", 0x09, Split::Dev),
            ("p255", 0x1c, Split::Train),
            ("p256", 0x24, Split::Train),
        ];
        for (speaker, byte, split) in expected {
            assert_eq!(speaker_hash_byte(speaker), byte, "{speaker}");
            assert_eq!(assign(CORPUS_VCTK, speaker, None), split, "{speaker}");
            // Shared namespace: VoiceBank's copy of the speaker lands in the
            // same split, whatever its subset says.
            assert_eq!(
                assign(CORPUS_VOICEBANK_DEMAND, speaker, Some("train")),
                split,
                "{speaker}"
            );
        }
    }

    #[test]
    fn split_serde_is_lowercase() {
        assert_eq!(serde_json::to_string(&Split::Dev).unwrap(), "\"dev\"");
        assert_eq!(
            serde_json::from_str::<Split>("\"test\"").unwrap(),
            Split::Test
        );
        assert_eq!("train".parse::<Split>().unwrap(), Split::Train);
        assert!("validation".parse::<Split>().is_err());
    }
}
