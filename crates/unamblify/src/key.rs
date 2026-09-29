// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Utterance keys: the corpus-relative, filesystem-safe identifier that
//! names an utterance everywhere (`prepared/<key>.16k.wav`,
//! `captured/<mode>/<key>.ambe`, manifests, shards, eval-clip lists).
//!
//! Shapes (spec §2):
//!
//! | corpus | key |
//! |---|---|
//! | LibriTTS-R | `libritts_r/<subset>/<reader>_<chapter>_<para>_<sent>` |
//! | VoiceBank-DEMAND | `voicebank_demand/<set>/<spk>_<utt>` |
//! | VCTK | `vctk/<spk>_<utt>_mic<N>` |
//! | LJSpeech | `ljspeech/LJ<book>-<utt>` |
//!
//! The leading directory is the corpus id, so `corpus_of(key)` never needs
//! a lookup table.
//!
//! An augmented **twin** of an utterance (a noisy or overdriven copy made
//! by `prepare --noise-share` / `--ham-chain-share`) carries its parent's
//! key plus one or more `+<tag>` suffixes: `vctk/p225_001_mic2+n0173`,
//! `…+h0042`, `…+n0173+h0042`. [`parent_of`] strips them; the parent's
//! clean files are the twin's training target.

use std::fmt;

/// A key contains a character that is not allowed on disk or in a
/// manifest, or has no corpus directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyError {
    /// The rejected key.
    pub key: String,
    /// Why it was rejected.
    pub reason: &'static str,
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid utterance key {:?}: {}", self.key, self.reason)
    }
}

impl std::error::Error for KeyError {}

/// `libritts_r/<subset>/<reader>_<chapter>_<para>_<sent>`.
///
/// The four ids are passed as the zero-padded strings LibriTTS-R uses in
/// its filenames (`84_121123_000007_000001`), not as integers, so the key
/// stays byte-identical to the source stem.
#[must_use]
pub fn libritts_r(subset: &str, reader: &str, chapter: &str, para: &str, sent: &str) -> String {
    format!("libritts_r/{subset}/{reader}_{chapter}_{para}_{sent}")
}

/// `libritts_r/<subset>/<stem>` from a source file stem that is already
/// `<reader>_<chapter>_<para>_<sent>`. Returns `None` when the stem does
/// not have exactly four underscore-separated, non-empty parts.
#[must_use]
pub fn libritts_r_from_stem(subset: &str, stem: &str) -> Option<String> {
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() != 4 || parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    Some(libritts_r(subset, parts[0], parts[1], parts[2], parts[3]))
}

/// `voicebank_demand/<set>/<spk>_<utt>`, e.g. `voicebank_demand/train/p226_001`.
#[must_use]
pub fn voicebank_demand(set: &str, speaker: &str, utt: &str) -> String {
    format!("voicebank_demand/{set}/{speaker}_{utt}")
}

/// `vctk/<spk>_<utt>_mic<N>`, e.g. `vctk/p225_001_mic2`.
#[must_use]
pub fn vctk(speaker: &str, utt: &str, mic: u8) -> String {
    format!("vctk/{speaker}_{utt}_mic{mic}")
}

/// `ljspeech/LJ<book>-<utt>`, zero-padded as LJSpeech names its files
/// (`LJ001-0001`).
#[must_use]
pub fn ljspeech(book: u32, utt: u32) -> String {
    format!("ljspeech/LJ{book:03}-{utt:04}")
}

/// `hifi_tts/<speaker>_<quality>/<book>/<stem>`, e.g.
/// `hifi_tts/92_clean/12345/somebook_01_author_0007` — the corpus's own
/// `audio/` layout, so a key finds its file and its speaker.
#[must_use]
pub fn hifi_tts(speaker: &str, quality: &str, book: &str, stem: &str) -> String {
    format!("hifi_tts/{speaker}_{quality}/{book}/{stem}")
}

/// The corpus id: everything before the first `/`. `None` when the key has
/// no directory component.
#[must_use]
pub fn corpus_of(key: &str) -> Option<&str> {
    let (corpus, rest) = key.split_once('/')?;
    (!corpus.is_empty() && !rest.is_empty()).then_some(corpus)
}

/// The last path component of the key (the utterance stem without suffix).
#[must_use]
pub fn stem_of(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// The parent of an augmented twin (`vctk/p225_001_mic2+n0173` →
/// `vctk/p225_001_mic2`); `None` when the key carries no `+` suffix, i.e.
/// is a plain prepared utterance.
#[must_use]
pub fn parent_of(key: &str) -> Option<&str> {
    let stem_start = key.rfind('/').map_or(0, |i| i + 1);
    let plus = key[stem_start..].find('+')? + stem_start;
    (plus > stem_start).then(|| &key[..plus])
}

/// The base utterance a key refers to: its parent when it is a twin, else
/// itself.
#[must_use]
pub fn base_of(key: &str) -> &str {
    parent_of(key).unwrap_or(key)
}

/// A twin key: `<parent>+<tag>` (`tag` such as `n0173`, `h0042`).
#[must_use]
pub fn twin(parent: &str, tag: &str) -> String {
    format!("{parent}+{tag}")
}

/// A key as a single filesystem-safe file stem (`vctk_p228_003_mic2`):
/// every character outside `[A-Za-z0-9.-]` becomes `_`. Names the
/// trainer's rendered eval clips and the `infer` verb's outputs.
#[must_use]
pub fn clip_name(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Check that a key is filesystem-safe: ASCII letters, digits, `_`, `-`,
/// `.`, `+` (twin suffixes) and `/` only, no empty or `.`/`..` components,
/// and a corpus directory in front.
pub fn validate(key: &str) -> Result<(), KeyError> {
    let err = |reason| KeyError {
        key: key.to_owned(),
        reason,
    };
    if corpus_of(key).is_none() {
        return Err(err("missing corpus directory"));
    }
    if !key
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b'+'))
    {
        return Err(err("contains a character outside [A-Za-z0-9_./+-]"));
    }
    if key
        .split('/')
        .any(|c| c.is_empty() || c == "." || c == "..")
    {
        return Err(err("empty, `.` or `..` path component"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_match_the_spec_examples() {
        assert_eq!(vctk("p225", "001", 2), "vctk/p225_001_mic2");
        assert_eq!(
            hifi_tts("92", "clean", "12345", "somebook_01_author_0007"),
            "hifi_tts/92_clean/12345/somebook_01_author_0007"
        );
        validate("hifi_tts/92_clean/12345/somebook_01_author_0007").unwrap();
        assert_eq!(
            libritts_r("dev-clean", "84", "121123", "000007", "000001"),
            "libritts_r/dev-clean/84_121123_000007_000001"
        );
        assert_eq!(
            libritts_r_from_stem("test-clean", "1089_134686_000001_000001").as_deref(),
            Some("libritts_r/test-clean/1089_134686_000001_000001")
        );
        assert_eq!(libritts_r_from_stem("test-clean", "1089_134686"), None);
        assert_eq!(
            voicebank_demand("test", "p232", "001"),
            "voicebank_demand/test/p232_001"
        );
        assert_eq!(ljspeech(1, 1), "ljspeech/LJ001-0001");
        assert_eq!(ljspeech(50, 277), "ljspeech/LJ050-0277");
    }

    #[test]
    fn corpus_and_stem_are_read_back() {
        assert_eq!(corpus_of("vctk/p225_001_mic2"), Some("vctk"));
        assert_eq!(
            corpus_of("libritts_r/dev-clean/84_1_2_3"),
            Some("libritts_r")
        );
        assert_eq!(corpus_of("noslash"), None);
        assert_eq!(corpus_of("/x"), None);
        assert_eq!(stem_of("libritts_r/dev-clean/84_1_2_3"), "84_1_2_3");
        assert_eq!(clip_name("vctk/p228_003_mic2"), "vctk_p228_003_mic2");
        assert_eq!(clip_name("ljspeech/LJ001-0001"), "ljspeech_LJ001-0001");
    }

    #[test]
    fn validation_rejects_unsafe_keys() {
        assert!(validate("vctk/p225_001_mic2").is_ok());
        assert!(validate("ljspeech/LJ001-0001").is_ok());
        assert!(validate("p225_001").is_err());
        assert!(validate("vctk/../etc").is_err());
        assert!(validate("vctk//x").is_err());
        assert!(validate("vctk/p225 001").is_err());
        assert!(validate("vctk/p225\u{e9}").is_err());
        assert!(validate("vctk/p225_001_mic2+n0173+h0042").is_ok());
    }

    #[test]
    fn twin_keys_name_their_parent() {
        assert_eq!(parent_of("vctk/p225_001_mic2"), None);
        assert_eq!(
            parent_of("vctk/p225_001_mic2+n0173"),
            Some("vctk/p225_001_mic2")
        );
        assert_eq!(
            parent_of("libritts_r/dev-clean/84_1_2_3+n0173+h0042"),
            Some("libritts_r/dev-clean/84_1_2_3")
        );
        assert_eq!(parent_of("vctk/+n0173"), None, "no empty parent");
        assert_eq!(base_of("vctk/p225_001_mic2+h0042"), "vctk/p225_001_mic2");
        assert_eq!(base_of("vctk/p225_001_mic2"), "vctk/p225_001_mic2");
        assert_eq!(
            twin("vctk/p225_001_mic2", "n0173"),
            "vctk/p225_001_mic2+n0173"
        );
        assert_eq!(
            clip_name("vctk/p225_001_mic2+n0173"),
            "vctk_p225_001_mic2_n0173"
        );
    }
}
