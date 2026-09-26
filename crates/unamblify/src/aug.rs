// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! What an augmentation did, as recorded in the manifests
//! (`docs/design/data-pipeline.md`, stage 3): the noisy-twin and
//! overdriven-mic parameters of a prepared twin row ([`AugRecord`]), and
//! the channel mutation of a decode-only sibling capture ([`CaptureAug`]).
//! The parameters are drawn by the data crate and rendered by the audio
//! crate; this module is the serde shape both agree on, plus the one
//! seeding rule every stage shares ([`key_seed`]).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::ParseEnumError;

/// A per-key stream seed: the first eight bytes of `sha1(key)` xor `seed`,
/// so a share decision or a parameter draw for one utterance is stable
/// across runs, independent of the order utterances are visited in, and
/// changes when the run seed does.
#[must_use]
pub fn key_seed(key: &str, seed: u64) -> u64 {
    let d = Sha1::digest(key.as_bytes());
    u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]) ^ seed
}

/// Which noise set a clip came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NoiseSet {
    /// DEMAND: `raw/demand/<ENV>/chNN.wav`, 16 kHz, one channel per file.
    Demand,
    /// MUSAN noise and music: `raw/musan/{noise,music}/**/*.wav` (never
    /// `speech`, which would confuse the target).
    Musan,
}

impl NoiseSet {
    /// Every set, in the order `--noise-sets` lists them by default.
    pub const ALL: [Self; 2] = [Self::Demand, Self::Musan];

    /// Stable identifier (`demand`, `musan`); also the directory under `raw/`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Demand => "demand",
            Self::Musan => "musan",
        }
    }
}

impl fmt::Display for NoiseSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NoiseSet {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|n| n.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "noise set",
                input: s.to_owned(),
            })
    }
}

/// The noise mixed into a noisy twin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseAug {
    /// Which set.
    pub noise_set: NoiseSet,
    /// The clip, relative to the data root (`raw/demand/TCAR/ch03.wav`).
    pub noise_clip: String,
    /// Where in the clip the segment starts, seconds (wrapping when the
    /// clip is shorter than the utterance).
    pub noise_offset_s: f64,
    /// Signal-to-noise ratio, dB: active-speech RMS of the clean parent
    /// over the RMS of the noise segment as mixed.
    pub snr_db: f32,
}

/// How the mic amp saturates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClipKind {
    /// `clamp(g·x, −1, 1)`.
    Hard,
    /// `tanh(g·x)`.
    Soft,
}

/// Clipping: `drive_db` of gain into the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipAug {
    /// Hard or soft.
    pub kind: ClipKind,
    /// Gain into the ceiling, dB (6–24).
    pub drive_db: f32,
}

/// An underdriven transmit: the talker too far from the mic or the gain
/// too low, so the vocoder is fed well below its working level. Unlike
/// the overdriven chain, the low level *is* the degradation — the damage
/// happens inside the codec, where a coarse gain quantiser and a starved
/// pitch tracker meet a quiet signal — so a `+u` twin is deliberately not
/// brought back to the prepare target before it is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnderAug {
    /// Attenuation below the prepare target, dB (−30 to −15).
    pub gain_db: f32,
    /// Distance: the proximity bass that is *missing*, as a low shelf cut
    /// below 200 Hz, dB (−8 to −3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shelf_db: Option<f32>,
}

/// Plosive pops on detected onsets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PopsAug {
    /// Fraction of detected onsets that get a thump (0.2–0.5).
    pub share: f32,
    /// Thump frequency, Hz (60–120).
    pub freq_hz: f32,
    /// Thump peak level, dBFS (−10 to −3).
    pub level_dbfs: f32,
}

/// Gain riding into a hard limiter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AgcAug {
    /// Attack, ms (5–50).
    pub attack_ms: f32,
    /// Release, ms (100–500).
    pub release_ms: f32,
}

/// The hand-mic capsule: a tilt and one resonance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicAug {
    /// Spectral tilt across the band, dB (±6; positive = brighter).
    pub tilt_db: f32,
    /// Resonance centre, Hz.
    pub resonance_hz: f32,
    /// Resonance gain, dB.
    pub resonance_db: f32,
    /// Resonance Q.
    pub resonance_q: f32,
}

/// The synthetic transmit chain of an overdriven-mic twin, in the order
/// it is applied: proximity shelf → mic response → pops → clipping → AGC
/// and limiter. Any combination; every field is optional and at least one
/// is drawn.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChainParams {
    /// Proximity effect: low shelf gain below 200 Hz, dB (+4 to +10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shelf_db: Option<f32>,
    /// Mic response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mic: Option<MicAug>,
    /// Plosive pops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pops: Option<PopsAug>,
    /// Clipping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<ClipAug>,
    /// AGC then a hard limiter at −1 dBFS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agc: Option<AgcAug>,
}

impl ChainParams {
    /// Whether any stage is on.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.shelf_db.is_none()
            && self.mic.is_none()
            && self.pops.is_none()
            && self.clip.is_none()
            && self.agc.is_none()
    }

    /// The eval's fixed overdrive recipe: 12 dB hard clip plus a +6 dB
    /// proximity shelf.
    #[must_use]
    pub const fn fixed_overdrive() -> Self {
        Self {
            shelf_db: Some(6.0),
            mic: None,
            pops: None,
            clip: Some(ClipAug {
                kind: ClipKind::Hard,
                drive_db: 12.0,
            }),
            agc: None,
        }
    }
}

/// The `aug` field of a twin's [`crate::UtteranceRow`]: what was done to
/// the parent's clean signal to make this row's input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AugRecord {
    /// The per-key seed every draw came from ([`key_seed`]).
    pub seed: u64,
    /// The noise mixed in (a `+n` twin).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<NoiseAug>,
    /// The transmit chain applied (a `+h` twin).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<ChainParams>,
    /// The underdriven transmit applied (a `+u` twin). Never together
    /// with `chain`: a mic cannot be both too hot and too quiet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub under: Option<UnderAug>,
    /// Gain applied after everything to bring the active level back to
    /// the prepare target (0 when nothing was needed), dB.
    #[serde(default)]
    pub post_gain_db: f32,
}

/// What `unamblify augment` does to the stored channel frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AugKind {
    /// Bursts of lost frames, substituted.
    Drops,
    /// Random channel bit errors.
    Ber,
    /// The same utterances re-encoded by a second, independent
    /// implementation of the mode's codec rather than the chip. D-STAR
    /// only: the vendored software vocoder implements no other mode.
    Perens,
}

impl AugKind {
    /// Every kind.
    pub const ALL: [Self; 3] = [Self::Drops, Self::Ber, Self::Perens];

    /// Stable identifier; also the `+<kind>` suffix of the sibling
    /// capture directory.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Drops => "drops",
            Self::Ber => "ber",
            Self::Perens => "perens",
        }
    }
}

impl fmt::Display for AugKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AugKind {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "augment kind",
                input: s.to_owned(),
            })
    }
}

/// What a receiver substitutes for a lost frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Subst {
    /// The mode's mute / null codeword (what MMDVM-style hosts and astar
    /// send the vocoder). The default.
    Mute,
    /// Hold the last good frame (what some radios do).
    Repeat,
    /// Mark the frame bad and let the chip's own concealment act.
    /// Chip-only; not implemented by this harness.
    Erase,
}

/// Which forward error correction to model around the voice bits.
///
/// `ber` without this flips voice bits directly, which no receiver sees:
/// on the air the protected bits are protected. The YSF settings wrap each
/// frame in a V/D mode's FEC, flip transmitted bits, and decode as a
/// receiver would, so what reaches the vocoder is the residual error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fec {
    /// Flip the voice bits themselves. No FEC modelled.
    None,
    /// YSF DN V/D mode 1: Golay (24, 12) over twelve bits, Golay (23, 12)
    /// over twelve more, twenty-five bits bare.
    YsfVd1,
    /// YSF DN V/D mode 2: the first twenty-seven bits sent three times and
    /// voted, twenty-two bits bare.
    YsfVd2,
}

impl Fec {
    /// Every setting.
    pub const ALL: [Self; 3] = [Self::None, Self::YsfVd1, Self::YsfVd2];

    /// Stable identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::YsfVd1 => "ysf-vd1",
            Self::YsfVd2 => "ysf-vd2",
        }
    }
}

impl fmt::Display for Fec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Fec {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "fec",
                input: s.to_owned(),
            })
    }
}

impl Subst {
    /// Every substitution.
    pub const ALL: [Self; 3] = [Self::Mute, Self::Repeat, Self::Erase];

    /// Stable identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mute => "mute",
            Self::Repeat => "repeat",
            Self::Erase => "erase",
        }
    }
}

impl fmt::Display for Subst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Subst {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "substitution",
                input: s.to_owned(),
            })
    }
}

/// The `aug` field of a sibling capture's [`crate::CaptureRow`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureAug {
    /// Drops or bit errors.
    pub kind: AugKind,
    /// The requested rate: the fraction of frames lost (`drops`) or the
    /// per-bit flip probability (`ber`), stored as parts per million so
    /// the row stays `Eq`.
    pub rate_ppm: u32,
    /// Burst length range, frames (`drops` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub burst: Option<(u32, u32)>,
    /// The substitution (`drops` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subst: Option<Subst>,
    /// The FEC modelled around the voice bits (`ber` only). Absent on a
    /// sibling made before the setting existed, which flipped voice bits
    /// directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fec: Option<Fec>,
    /// The run seed the per-utterance stream derived from.
    pub seed: u64,
    /// Frame indices mutated: every lost frame, or every frame with at
    /// least one flipped bit. Time is never shifted, so a position maps
    /// straight onto the decoded audio.
    pub positions: Vec<u32>,
}

impl CaptureAug {
    /// The rate as a fraction.
    #[must_use]
    pub fn rate(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let r = self.rate_ppm as f32 / 1e6;
        r
    }

    /// A rate as parts per million.
    #[must_use]
    pub fn ppm(rate: f32) -> u32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let p = (rate.clamp(0.0, 1.0) * 1e6).round() as u32;
        p
    }
}

/// The directory name of a capture set under `captured/`: the mode for
/// the base capture, `<mode>+<kind>` for a decode-only sibling.
#[must_use]
pub fn capture_dir_name(mode: crate::VocoderMode, kind: Option<AugKind>) -> String {
    match kind {
        None => mode.as_str().to_owned(),
        Some(k) => format!("{}+{}", mode.as_str(), k.as_str()),
    }
}

/// Parse a capture set name (`dstar`, `dstar+drops`) into its mode and
/// kind. The mode must be spelled as [`crate::VocoderMode::as_str`] does:
/// a directory still under a retired name (`ysf-dn`) is not a capture set
/// of the current mode until `unamblify migrate-modes` has renamed it.
pub fn parse_capture_dir_name(
    name: &str,
) -> Result<(crate::VocoderMode, Option<AugKind>), ParseEnumError> {
    match name.split_once('+') {
        None => Ok((crate::VocoderMode::from_canonical(name)?, None)),
        Some((m, k)) => Ok((crate::VocoderMode::from_canonical(m)?, Some(k.parse()?))),
    }
}

/// The `kind` word of a `[data] kinds` list: `base` for the capture
/// itself, else an [`AugKind`].
pub fn parse_kind_word(word: &str) -> Result<Option<AugKind>, ParseEnumError> {
    if word == "base" {
        Ok(None)
    } else {
        word.parse().map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VocoderMode;

    /// Rows written before underdriven twins existed have no `under`
    /// field; they must still read, and a `+u` row must round-trip.
    #[test]
    fn an_aug_record_without_under_still_reads_and_one_with_it_round_trips() {
        let old = r#"{"seed":7,"post_gain_db":-1.5}"#;
        let rec: AugRecord = serde_json::from_str(old).unwrap();
        assert!(rec.under.is_none() && rec.chain.is_none());
        let u = AugRecord {
            seed: 9,
            noise: None,
            chain: None,
            under: Some(UnderAug {
                gain_db: -22.5,
                shelf_db: Some(-4.0),
            }),
            post_gain_db: 0.0,
        };
        let back: AugRecord = serde_json::from_str(&serde_json::to_string(&u).unwrap()).unwrap();
        assert_eq!(back, u);
    }

    #[test]
    fn the_perens_sibling_names_and_parses_back() {
        let name = capture_dir_name(VocoderMode::Dstar, Some(AugKind::Perens));
        assert_eq!(name, "dstar+perens");
        assert_eq!(
            parse_capture_dir_name(&name).unwrap(),
            (VocoderMode::Dstar, Some(AugKind::Perens))
        );
        assert_eq!(parse_kind_word("perens").unwrap(), Some(AugKind::Perens));
        assert_eq!("perens".parse::<AugKind>().unwrap(), AugKind::Perens);
    }

    #[test]
    fn every_kind_round_trips_its_word() {
        for k in AugKind::ALL {
            assert_eq!(k.as_str().parse::<AugKind>().unwrap(), k, "{k}");
        }
    }

    #[test]
    fn key_seed_is_stable_and_key_specific() {
        assert_eq!(
            key_seed("vctk/p225_001_mic2", 1),
            key_seed("vctk/p225_001_mic2", 1)
        );
        assert_ne!(
            key_seed("vctk/p225_001_mic2", 1),
            key_seed("vctk/p225_001_mic2", 2)
        );
        assert_ne!(
            key_seed("vctk/p225_001_mic2", 1),
            key_seed("vctk/p225_002_mic2", 1)
        );
    }

    #[test]
    fn records_round_trip_and_optional_fields_are_omitted() {
        let rec = AugRecord {
            seed: 7,
            noise: Some(NoiseAug {
                noise_set: NoiseSet::Demand,
                noise_clip: "raw/demand/TCAR/ch03.wav".to_owned(),
                noise_offset_s: 12.5,
                snr_db: 7.25,
            }),
            chain: None,
            under: None,
            post_gain_db: 0.0,
        };
        let text = serde_json::to_string(&rec).unwrap();
        assert!(text.contains("\"noise_set\":\"demand\""));
        assert!(!text.contains("chain"));
        assert_eq!(serde_json::from_str::<AugRecord>(&text).unwrap(), rec);
        let chain = ChainParams::fixed_overdrive();
        assert!(!chain.is_empty());
        assert!(ChainParams::default().is_empty());
        let text = serde_json::to_string(&chain).unwrap();
        assert_eq!(
            text,
            r#"{"shelf_db":6.0,"clip":{"kind":"hard","drive_db":12.0}}"#
        );
        let cap = CaptureAug {
            kind: AugKind::Drops,
            rate_ppm: CaptureAug::ppm(0.02),
            burst: Some((1, 3)),
            subst: Some(Subst::Mute),
            fec: None,
            seed: 1,
            positions: vec![4, 5, 60],
        };
        assert_eq!(cap.rate_ppm, 20_000);
        assert!((cap.rate() - 0.02).abs() < 1e-6);
        let text = serde_json::to_string(&cap).unwrap();
        assert!(text.contains("\"kind\":\"drops\""));
        // A drops row carries no FEC, so the field stays out of the JSON.
        assert!(!text.contains("fec"), "{text}");
        assert_eq!(serde_json::from_str::<CaptureAug>(&text).unwrap(), cap);

        // A bit-error row records which FEC was modelled, so a sibling can
        // be told apart from one made before the setting existed.
        let ber = CaptureAug {
            kind: AugKind::Ber,
            rate_ppm: CaptureAug::ppm(0.01),
            burst: None,
            subst: None,
            fec: Some(Fec::YsfVd1),
            seed: 1,
            positions: vec![7],
        };
        let text = serde_json::to_string(&ber).unwrap();
        assert!(text.contains("\"fec\":\"ysf-vd1\""), "{text}");
        assert_eq!(serde_json::from_str::<CaptureAug>(&text).unwrap(), ber);
        // A row written before the field existed still reads.
        let old = r#"{"kind":"ber","rate_ppm":1000,"seed":1,"positions":[7]}"#;
        assert_eq!(serde_json::from_str::<CaptureAug>(old).unwrap().fec, None);
    }

    #[test]
    fn names_parse_both_ways() {
        assert_eq!("drops".parse::<AugKind>().unwrap(), AugKind::Drops);
        assert!("dropz".parse::<AugKind>().is_err());
        assert_eq!("repeat".parse::<Subst>().unwrap(), Subst::Repeat);
        assert_eq!("musan".parse::<NoiseSet>().unwrap(), NoiseSet::Musan);
        assert_eq!(NoiseSet::Demand.to_string(), "demand");
        assert_eq!(capture_dir_name(VocoderMode::Dstar, None), "dstar");
        assert_eq!(
            capture_dir_name(VocoderMode::Codec2_3200, Some(AugKind::Ber)),
            "codec2-3200+ber"
        );
        assert_eq!(
            parse_capture_dir_name("dstar+drops").unwrap(),
            (VocoderMode::Dstar, Some(AugKind::Drops))
        );
        assert_eq!(
            parse_capture_dir_name("ysf-dmr").unwrap(),
            (VocoderMode::YsfDmr, None)
        );
        // A retired spelling parses as a mode but is not a capture set name.
        assert_eq!(
            "ysf-dn".parse::<VocoderMode>().unwrap(),
            VocoderMode::YsfDmr
        );
        assert!(parse_capture_dir_name("ysf-dn").is_err());
        assert!(parse_capture_dir_name("dmr+ber").is_err());
        assert!(parse_capture_dir_name("dstar+x").is_err());
        assert_eq!(parse_kind_word("base").unwrap(), None);
        assert_eq!(parse_kind_word("drops").unwrap(), Some(AugKind::Drops));
        assert!(parse_kind_word("x").is_err());
    }
}
