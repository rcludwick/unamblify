// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Shared constants and types for the unamblify pipeline.
//!
//! The AMBE family works in 20 ms frames of 8 kHz, 16-bit, mono PCM, and so
//! does Codec 2 3200 (M17 voice); Codec 2 1600 (M17 voice + data) uses 40 ms
//! frames at the same rate. Every stage — corpus preparation, the capture
//! harness, training, and the real-time post-filter — speaks in these
//! units, so they live here; [`VocoderMode::frame_samples`] is the per-mode
//! word and [`FRAME_SAMPLES`] the AMBE constant.
//!
//! This crate also holds the serde types every other crate exchanges on disk
//! (manifest rows, run configs, metrics, shard indexes) and the two pure
//! functions that must agree everywhere: the speaker-disjoint [`split`] rule
//! and the utterance [`key`] builders. It does no I/O beyond serde.

pub mod aug;
pub mod channel;
pub mod config;
pub mod fec;
pub mod key;
pub mod manifest;
pub mod rng;
pub mod run;
pub mod shard;
pub mod split;
pub mod tail;

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub use aug::{AugKind, AugRecord, CaptureAug, ChainParams, NoiseSet, Subst};
pub use config::{
    AugmentCfg, CkptCfg, DataCfg, DataSource, EvalCfg, ModelCfg, Profile, RunConfig, TrainCfg,
};
pub use config::{ConfigError, MODE_EMBED_DIM};
pub use manifest::{CanaryRecord, CaptureRow, UtteranceRow};
pub use rng::Rng;
pub use run::{Best, LogLevel, LogRow, MetricRow, RunState, RunStatus};
pub use shard::{
    ExampleLayout, FLAGS_BYTES, LayoutMismatch, ShardFile, ShardIndex, SourceSha256s, speaker_id,
};
pub use split::Split;

/// Sample rate of the vocoder's speech interface, in Hz.
pub const VOCODER_SAMPLE_RATE: u32 = 8_000;

/// Sample rate of the clean (wideband) training target, in Hz.
pub const WIDEBAND_SAMPLE_RATE: u32 = 16_000;

/// Duration of one AMBE vocoder frame, in milliseconds (also the model's
/// lookahead unit). Per mode: [`VocoderMode::frame_ms`].
pub const FRAME_MS: u32 = 20;

/// PCM samples per AMBE vocoder frame (160 at 8 kHz / 20 ms). Per mode:
/// [`VocoderMode::frame_samples`].
pub const FRAME_SAMPLES: usize = (VOCODER_SAMPLE_RATE * FRAME_MS / 1_000) as usize;

/// The two codec families the harness captures: AMBE through a DVSI
/// chip, Codec 2 in software.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VocoderFamily {
    /// AMBE / AMBE+2, encoded and decoded by an AMBE-3000 (ThumbDV).
    Ambe,
    /// Codec 2 (M17), encoded and decoded by the pure-Rust `codec2` crate.
    Codec2,
}

impl VocoderFamily {
    /// Stable lowercase identifier (`"ambe"`, `"codec2"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ambe => "ambe",
            Self::Codec2 => "codec2",
        }
    }
}

impl fmt::Display for VocoderFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The vocoder configurations the capture harness records. The name is
/// the on-air mode; the payload is what the codec is told to produce. The
/// AMBE modes go through the AMBE-3000 chip; the Codec 2 modes (what M17
/// carries) run in software.
///
/// Serialises as its [`VocoderMode::as_str`] identifier (`"dstar"`,
/// `"ysf-dmr"`, `"codec2-3200"`, `"codec2-1600"`), which is also the
/// directory name under `captured/`. Parsing (serde and [`FromStr`]) also
/// accepts the retired spellings in [`VocoderMode::ALIASES`] — `"ysf-dn"`,
/// `"ysf"`, `"dmr"` — so manifests, status files and shard indexes written
/// before the rename still load; `unamblify migrate-modes` rewrites them.
///
/// There is no DMR mode: DMR carries the identical 49 AMBE+2 voice bits as
/// YSF DN, wrapped in 23 bits of Golay FEC, so with no channel errors the
/// chip's decoded audio is the same and one capture serves both. DMR's
/// FEC only matters under bit errors, which `augment --kind ber` will
/// synthesise from the 49 bits; its rate word and null frame stay in the
/// code as constants for that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VocoderMode {
    /// D-STAR: AMBE 2400 bps voice + 1200 bps FEC, 72-bit (9-byte) frames.
    Dstar,
    /// System Fusion DN, DMR (and NXDN): AMBE+2 2450 bps voice, 49-bit
    /// (7-byte) frames — captured without FEC, as YSF DN sends them; DMR's
    /// Golay wrapper adds nothing the chip's decode can hear.
    #[serde(rename = "ysf-dmr", alias = "ysf-dn", alias = "ysf", alias = "dmr")]
    YsfDmr,
    /// M17 stream type "voice": Codec 2 mode 3200, 64 bits (8 bytes) per
    /// 20 ms frame of 160 samples.
    #[serde(rename = "codec2-3200")]
    Codec2_3200,
    /// M17 stream type "voice + data": Codec 2 mode 1600, 64 bits (8 bytes)
    /// per 40 ms frame of 320 samples (the other 1600 bit/s is data).
    #[serde(rename = "codec2-1600")]
    Codec2_1600,
}

impl VocoderMode {
    /// Every mode, in capture order.
    pub const ALL: [Self; 4] = [
        Self::Dstar,
        Self::YsfDmr,
        Self::Codec2_3200,
        Self::Codec2_1600,
    ];

    /// Retired spellings still accepted on input, never written:
    /// `ysf-dn` (the mode's name before DMR was folded into it), `ysf`,
    /// and `dmr` (once a mode of its own).
    pub const ALIASES: [(&'static str, Self); 3] = [
        ("ysf-dn", Self::YsfDmr),
        ("ysf", Self::YsfDmr),
        ("dmr", Self::YsfDmr),
    ];

    /// Which codec family produces this mode.
    #[must_use]
    pub const fn family(self) -> VocoderFamily {
        match self {
            Self::Dstar | Self::YsfDmr => VocoderFamily::Ambe,
            Self::Codec2_3200 | Self::Codec2_1600 => VocoderFamily::Codec2,
        }
    }

    /// Whether the codec runs in software (no chip, no serial port).
    #[must_use]
    pub const fn is_software(self) -> bool {
        matches!(self.family(), VocoderFamily::Codec2)
    }

    /// PCM samples per channel frame at 8 kHz: 160 for every AMBE mode and
    /// Codec 2 3200, 320 for Codec 2 1600.
    #[must_use]
    pub const fn frame_samples(self) -> usize {
        match self {
            Self::Dstar | Self::YsfDmr | Self::Codec2_3200 => FRAME_SAMPLES,
            Self::Codec2_1600 => 2 * FRAME_SAMPLES,
        }
    }

    /// Duration of one channel frame, milliseconds.
    #[must_use]
    pub const fn frame_ms(self) -> u32 {
        match self {
            Self::Dstar | Self::YsfDmr | Self::Codec2_3200 => FRAME_MS,
            Self::Codec2_1600 => 2 * FRAME_MS,
        }
    }

    /// Bits of channel data the codec emits per frame in this mode (for
    /// the chip, the `bits` field of its channel packet).
    #[must_use]
    pub const fn channel_bits(self) -> u16 {
        match self {
            Self::Dstar => 72,
            Self::YsfDmr => 49,
            Self::Codec2_3200 | Self::Codec2_1600 => 64,
        }
    }

    /// Bytes of channel data per frame in this mode
    /// (`channel_bits().div_ceil(8)`): 9, 7, 8, 8.
    #[must_use]
    pub const fn frame_bytes(self) -> usize {
        match self {
            Self::Dstar => 9,
            Self::YsfDmr => 7,
            Self::Codec2_3200 | Self::Codec2_1600 => 8,
        }
    }

    /// Stable lowercase identifier used in dataset paths and manifests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dstar => "dstar",
            Self::YsfDmr => "ysf-dmr",
            Self::Codec2_3200 => "codec2-3200",
            Self::Codec2_1600 => "codec2-1600",
        }
    }

    /// The name shown to a person (`"D-STAR"`, `"YSF/DMR"`, ...); the
    /// identifier stays [`VocoderMode::as_str`].
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dstar => "D-STAR",
            Self::YsfDmr => "YSF/DMR",
            Self::Codec2_3200 => "Codec 2 3200 (M17)",
            Self::Codec2_1600 => "Codec 2 1600 (M17)",
        }
    }

    /// Parse an identifier exactly as [`VocoderMode::as_str`] spells it;
    /// an alias is refused. Directory names under `captured/` are matched
    /// this way, so a set left under a retired name is not mistaken for
    /// the current one until `migrate-modes` has renamed it.
    pub fn from_canonical(s: &str) -> Result<Self, ParseEnumError> {
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "vocoder mode",
                input: s.to_owned(),
            })
    }
}

impl fmt::Display for VocoderMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a string is not a known identifier of one of this
/// crate's enums.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseEnumError {
    /// What was being parsed (`"vocoder mode"`, `"profile"`, ...).
    pub what: &'static str,
    /// The offending input.
    pub input: String,
}

impl fmt::Display for ParseEnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {}: {:?}", self.what, self.input)
    }
}

impl std::error::Error for ParseEnumError {}

impl FromStr for VocoderMode {
    type Err = ParseEnumError;

    /// The identifier, or one of the [`VocoderMode::ALIASES`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_canonical(s).or_else(|e| {
            Self::ALIASES
                .into_iter()
                .find_map(|(a, m)| (a == s).then_some(m))
                .ok_or(e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_160_samples() {
        assert_eq!(FRAME_SAMPLES, 160);
    }

    #[test]
    fn mode_identifiers_are_stable() {
        assert_eq!(VocoderMode::Dstar.as_str(), "dstar");
        assert_eq!(VocoderMode::YsfDmr.as_str(), "ysf-dmr");
        assert_eq!(VocoderMode::Codec2_3200.as_str(), "codec2-3200");
        assert_eq!(VocoderMode::Codec2_1600.as_str(), "codec2-1600");
        assert_eq!(VocoderMode::YsfDmr.frame_bytes(), 7);
        assert_eq!(VocoderMode::YsfDmr.channel_bits(), 49);
        assert_eq!(VocoderMode::Dstar.channel_bits(), 72);
        assert_eq!(VocoderMode::ALL.len(), 4);
        for m in VocoderMode::ALL {
            assert!(!m.label().is_empty());
            assert_ne!(m.label(), m.as_str(), "{m}");
        }
        assert_eq!(VocoderMode::YsfDmr.label(), "YSF/DMR");
        assert_eq!(VocoderMode::Dstar.label(), "D-STAR");
        assert_eq!(VocoderMode::Codec2_3200.label(), "Codec 2 3200 (M17)");
        assert_eq!(VocoderMode::Codec2_1600.label(), "Codec 2 1600 (M17)");
    }

    #[test]
    fn retired_mode_names_parse_but_are_never_written() {
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Row {
            mode: VocoderMode,
            lags: std::collections::BTreeMap<VocoderMode, i32>,
        }
        // FromStr: the identifier and every alias; nothing else.
        for m in VocoderMode::ALL {
            assert_eq!(m.as_str().parse::<VocoderMode>().unwrap(), m);
            assert_eq!(VocoderMode::from_canonical(m.as_str()).unwrap(), m);
        }
        for (alias, m) in VocoderMode::ALIASES {
            assert_eq!(alias.parse::<VocoderMode>().unwrap(), m, "{alias}");
            assert!(VocoderMode::from_canonical(alias).is_err(), "{alias}");
        }
        assert!("fm".parse::<VocoderMode>().is_err());
        assert!("YSF-DMR".parse::<VocoderMode>().is_err());
        // serde: a row captured under the old name still loads, a map key
        // too, and everything serialises as the current identifier.
        for old in ["ysf-dn", "ysf", "dmr", "ysf-dmr"] {
            let json = format!(r#"{{"mode":"{old}","lags":{{"{old}":5}}}}"#);
            let row: Row = serde_json::from_str(&json).unwrap();
            assert_eq!(row.mode, VocoderMode::YsfDmr, "{old}");
            assert_eq!(row.lags[&VocoderMode::YsfDmr], 5, "{old}");
            assert_eq!(
                serde_json::to_string(&row).unwrap(),
                r#"{"mode":"ysf-dmr","lags":{"ysf-dmr":5}}"#
            );
        }
        assert!(serde_json::from_str::<Row>(r#"{"mode":"fm","lags":{}}"#).is_err());
        assert_eq!(
            serde_json::to_string(&VocoderMode::ALL).unwrap(),
            r#"["dstar","ysf-dmr","codec2-3200","codec2-1600"]"#
        );
    }

    #[test]
    fn frame_words_per_mode() {
        use VocoderMode::{Codec2_1600, Codec2_3200, Dstar, YsfDmr};
        let rows = [
            (Dstar, VocoderFamily::Ambe, 160, 20, 9, 72),
            (YsfDmr, VocoderFamily::Ambe, 160, 20, 7, 49),
            (Codec2_3200, VocoderFamily::Codec2, 160, 20, 8, 64),
            (Codec2_1600, VocoderFamily::Codec2, 320, 40, 8, 64),
        ];
        for (m, fam, samples, ms, bytes, bits) in rows {
            assert_eq!(m.family(), fam, "{m}");
            assert_eq!(m.is_software(), fam == VocoderFamily::Codec2, "{m}");
            assert_eq!(m.frame_samples(), samples, "{m}");
            assert_eq!(m.frame_ms(), ms, "{m}");
            assert_eq!(m.frame_bytes(), bytes, "{m}");
            assert_eq!(m.channel_bits(), bits, "{m}");
            assert_eq!(usize::from(m.channel_bits()).div_ceil(8), m.frame_bytes());
            assert_eq!(
                m.frame_samples(),
                (VOCODER_SAMPLE_RATE * m.frame_ms() / 1_000) as usize
            );
        }
        assert_eq!(VocoderFamily::Codec2.to_string(), "codec2");
        assert_eq!(
            serde_json::to_string(&VocoderFamily::Ambe).unwrap(),
            "\"ambe\""
        );
    }

    #[test]
    fn mode_serde_and_from_str_agree() {
        for m in VocoderMode::ALL {
            let json = serde_json::to_string(&m).unwrap();
            assert_eq!(json, format!("\"{}\"", m.as_str()));
            assert_eq!(serde_json::from_str::<VocoderMode>(&json).unwrap(), m);
            assert_eq!(m.as_str().parse::<VocoderMode>().unwrap(), m);
            assert_eq!(m.to_string(), m.as_str());
        }
        assert!("d-star".parse::<VocoderMode>().is_err());
    }
}
