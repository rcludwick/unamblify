// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The run config (`configs/*.toml`, spec §7). Every field except `name`
//! has a default equal to the spec's example, so a config can be as short
//! as `name = "x"`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{ParseEnumError, VocoderMode};

/// Model size profile (`docs/design/realtime.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// Desktop / full-quality: F = 256, GRU 256, ≤ 8 M parameters.
    Full,
    /// Embedded: F = 96, GRU 96, ≤ 1 M parameters.
    Lite,
    /// Microcontroller-class, no BWE head: F = 32, GRU 32, ≤ 100 K parameters.
    SuperLite,
}

impl Profile {
    /// Every profile, largest first.
    pub const ALL: [Self; 3] = [Self::Full, Self::Lite, Self::SuperLite];

    /// Stable identifier used in configs and run names.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Lite => "lite",
            Self::SuperLite => "super-lite",
        }
    }

    /// Maximum parameter count the trainer asserts at start.
    #[must_use]
    pub const fn param_budget(self) -> usize {
        match self {
            Self::Full => 8_000_000,
            Self::Lite => 1_000_000,
            Self::SuperLite => 100_000,
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Profile {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|p| p.as_str() == s)
            .ok_or_else(|| ParseEnumError {
                what: "profile",
                input: s.to_owned(),
            })
    }
}

/// Where training examples come from (spec §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataSource {
    /// Join the prepared and captured manifests and read WAVs per example.
    Pipeline,
    /// Read pre-packed `shards/<name>/NNNN.bin`.
    Shards,
}

/// Width of the learned per-mode embedding (`[model] mode_embed`).
pub const MODE_EMBED_DIM: usize = 16;

/// A `[data]` section that names its modes inconsistently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// `[model]`.
// Independent `[model]` switches, each its own TOML key; not a state
// machine. The same allowance `AugmentCfg` carries below.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelCfg {
    /// Size profile.
    pub profile: Profile,
    /// Multiplier on the profile's channel widths (1.0 = the profile as
    /// specified; widths round to a multiple of 8). Parameters grow
    /// roughly with its square and the profile's parameter budget still
    /// applies, so this asks whether more capacity helps before paying
    /// for it in real time — check `unamblify bench` before training.
    pub width: f32,
    /// Lookahead in AMBE 20 ms frames: 5 = ll5 (100 ms), 20 = ll20
    /// (400 ms), the variants of `docs/design/realtime.md`. The model runs
    /// at a 10 ms feature hop and sees `2 × lookahead` feature frames of
    /// future.
    pub lookahead: u32,
    /// Add the aperiodic excitation path: a per-frame shaped-noise
    /// generator whose gain is a fraction of the local signal level.
    /// Without it the network can only *filter* the decoder's output,
    /// and filtering cannot fill the troughs between harmonics that a
    /// vocoder empties — the residual "robotic" quality. Changes the
    /// tensor shapes, so it must match to resume or `init_from`.
    pub noise_head: bool,
    /// Give the noise path time structure inside a frame: the noise is
    /// modulated by the envelope of the periodic path (breath noise in
    /// real speech rides the glottal cycle; unmodulated noise on a comb
    /// reads as hiss over buzz), with a learned depth per frame, and
    /// carries a learned gain per **1 ms** sub-frame so it can form a
    /// plosive burst shorter than the 10 ms feature hop. Needs
    /// `noise_head`; adds eleven outputs to its head, so it changes the
    /// tensor shapes and must match to resume or `init_from`.
    pub noise_mod: bool,
    /// Condition the model on the vocoder mode: a learned
    /// [`MODE_EMBED_DIM`]-wide embedding per mode of `[data] modes`,
    /// concatenated to the feature stream at every frame, so one model
    /// trained on several modes can specialise per vocoder when told which
    /// one it is hearing. Off (the default) the model is blind to the mode.
    pub mode_embed: bool,
    /// Tell the model which frames were lost: a per-frame 0/1 erasure
    /// mask, one more channel into the context conv. A receiver's framing
    /// layer knows exactly which channel frames failed; the audio it hands
    /// on for them is the decoder's concealment, and the model should be
    /// told to distrust it. Entering at the context conv gives the mask the
    /// audio's own window — the lookahead included — so a gap is seen
    /// coming. With no mask supplied the model sees zeros: nothing was
    /// lost. It changes the tensor shapes, so it must match to resume or
    /// `init_from`.
    #[serde(default)]
    pub erasure_in: bool,
}

impl ModelCfg {
    /// Algorithmic latency the lookahead adds, in milliseconds.
    #[must_use]
    pub const fn lookahead_ms(&self) -> u32 {
        self.lookahead * crate::FRAME_MS
    }
}

impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            profile: Profile::Full,
            width: 1.0,
            lookahead: 5,
            noise_head: false,
            noise_mod: false,
            mode_embed: false,
            erasure_in: false,
        }
    }
}

/// `[data]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DataCfg {
    /// Pipeline or shards.
    pub source: DataSource,
    /// Shard set name under `shards/` when `source = "shards"`. No default:
    /// a shard set has to be named.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shards: Option<String>,
    /// Vocoder mode of the degraded input (a single-mode run). Either
    /// this or `modes`; neither means `dstar`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<VocoderMode>,
    /// The modes a run trains on, in the order the model's mode index
    /// counts them: the pipeline source joins each mode's captured
    /// manifest, the shards source needs every one of them in the set
    /// (a subset of the set's `modes` is fine; the loader filters), and
    /// the eval clips are loaded for each. One entry is a single-mode run.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub modes: Vec<VocoderMode>,
    /// Share of the batches each mode of `modes` gets, in the same order
    /// — relative weights, any positive scale. Empty (the default) draws
    /// a shard set's examples uniformly, so a mode's share is its share
    /// of the set (84 % Codec 2 in `mixed-large`), and the pipeline
    /// source draws round-robin. `[0.3, 0.15, 0.275, 0.275]` over
    /// `[dstar, ysf-dmr, codec2-3200, codec2-1600]` gives the two AMBE
    /// modes half the batches without discarding any Codec 2 example
    /// the way `shard --balance` does. One entry per mode, finite, ≥ 0,
    /// not all zero.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mode_weights: Vec<f32>,
    /// Example crop length, seconds.
    pub crop_s: f32,
    /// Which capture sets of `mode` the pipeline source draws from:
    /// `base` (`captured/<mode>/`) and any decode-only siblings
    /// (`drops`, `ber` → `captured/<mode>+<kind>/`). A shard set records
    /// its own kinds in `index.json`; this list is what `shard --kinds`
    /// and the pipeline loader use.
    pub kinds: Vec<String>,
}

impl DataCfg {
    /// The modes of the run, in index order: `modes` when given (`mode`,
    /// if also given, must be one of them), else `[mode]`, else `[dstar]`.
    /// Duplicates, and more modes than a shard's mode bits can index, are
    /// errors.
    pub fn modes(&self) -> Result<Vec<VocoderMode>, ConfigError> {
        let modes = if self.modes.is_empty() {
            vec![self.mode.unwrap_or(VocoderMode::Dstar)]
        } else {
            if let Some(m) = self.mode
                && !self.modes.contains(&m)
            {
                return Err(ConfigError(format!(
                    "[data] mode = \"{m}\" is not one of [data] modes {:?}; set one or the other",
                    self.modes.iter().map(|m| m.as_str()).collect::<Vec<_>>()
                )));
            }
            self.modes.clone()
        };
        for (i, m) in modes.iter().enumerate() {
            if modes[..i].contains(m) {
                return Err(ConfigError(format!("[data] modes lists {m} twice")));
            }
        }
        if modes.len() > crate::ExampleLayout::MAX_MODES {
            return Err(ConfigError(format!(
                "[data] modes lists {} modes; at most {} fit a shard's mode index",
                modes.len(),
                crate::ExampleLayout::MAX_MODES
            )));
        }
        Ok(modes)
    }

    /// The first mode of [`DataCfg::modes`]: what `unamblify infer`
    /// renders by default.
    pub fn primary_mode(&self) -> Result<VocoderMode, ConfigError> {
        Ok(self.modes()?[0])
    }

    /// The `kinds` list parsed: `None` for `base`, else the sibling kind.
    pub fn kinds(&self) -> Result<Vec<Option<crate::AugKind>>, ParseEnumError> {
        self.kinds
            .iter()
            .map(|w| crate::aug::parse_kind_word(w))
            .collect()
    }
}

impl Default for DataCfg {
    fn default() -> Self {
        Self {
            source: DataSource::Shards,
            shards: None,
            mode: None,
            modes: Vec::new(),
            mode_weights: Vec::new(),
            crop_s: 2.0,
            kinds: vec!["base".to_owned()],
        }
    }
}

/// `[augment]`: receive-side noise the loaders add to `deg8` on the fly
/// (`docs/design/data-pipeline.md`, stage 3), never to the target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
// Per-kind on/off switches of a config section, not a state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct AugmentCfg {
    /// Fraction of examples that get receive-side noise (0 = off).
    pub rx_share: f32,
    /// Mains / PSU hum.
    pub hum: bool,
    /// White or pink broadband noise.
    pub broadband: bool,
    /// Alternator whine.
    pub whine: bool,
    /// A cheap audio stage's colouring (tilt or notch, sometimes soft
    /// clipping).
    pub colouring: bool,
    /// A squelch tail: a decaying noise burst at the end of the clip, the
    /// crash a receiver makes when the carrier drops on an over.
    pub squelch: bool,
}

impl Default for AugmentCfg {
    fn default() -> Self {
        Self {
            rx_share: 0.3,
            hum: true,
            broadband: true,
            whine: true,
            colouring: true,
            squelch: true,
        }
    }
}

/// `[train]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainCfg {
    /// Optimiser steps to run.
    pub steps: u64,
    /// Examples per step.
    pub batch: u32,
    /// Peak learning rate.
    pub lr: f64,
    /// RNG seed for init, shuffling and crops.
    pub seed: u64,
    /// Device string: `cpu` | `mps` | `cuda:0` | `rocm:0`.
    pub device: String,
    /// Extra loss weight on the first 50 frames of onset examples.
    pub onset_w: f32,
    /// Loss weight on the silence target of garbage-tail examples.
    pub tail_w: f32,
    /// Enable the adversarial losses.
    pub gan: bool,
    /// Mixed precision.
    pub amp: bool,
    /// Weight of the periodicity term: the L1 between the cepstral peak
    /// prominence of the output and of the target, which rewards being
    /// *as periodic as the clean speech* rather than as periodic as
    /// possible. 0 = off. Pointless without `[model] noise_head`, which
    /// is what gives the model a way to comply.
    pub periodicity_w: f32,
    /// Weight of the harmonic-to-trough term: the |dB| between how far
    /// the output's harmonics stand above the valleys between them and
    /// how far the target's do. This is the one that names the bins the
    /// model has to fill, so it cannot be satisfied sideways the way
    /// `periodicity_w` can. Needs `[model] noise_head`. 0 = off;
    /// the error is in dB (single digits), so weights of order 0.05
    /// make it a few per cent of the total.
    pub hnr_w: f32,
    /// Weight of the transient term: the L1, in dB, between the output's
    /// and the target's 2–3.8 kHz energy envelope at 1 ms resolution,
    /// plus the L1 of its slope. The spectral terms' finest window is
    /// 32 ms, so a stop consonant's burst — a few milliseconds — is a
    /// fraction of one frame to them and they let the codec's smeared
    /// version stand; this term sees the burst, its rise, and the
    /// closure before it. 0 = off. The unweighted term is ~10 dB, so
    /// 0.03 puts it at about a tenth of the total.
    pub transient_w: f32,
    /// Start from another run's weights instead of a fresh init: a run
    /// id (`20260912-000009-generalist-full-ll5`, optionally
    /// `<id>:<step>`; without a step, that run's best checkpoint) or a
    /// path to a `checkpoints/step-N` directory. The new run is its own
    /// run — step 0, its own optimiser, its own metrics — and only the
    /// weights are inherited. A specialist that lists fewer modes than
    /// the checkpoint takes its mode-embedding rows from the parent's,
    /// one per mode it keeps. Ignored when resuming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from: Option<String>,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            steps: 20_000,
            batch: 16,
            lr: 2e-4,
            seed: 1,
            device: "cpu".to_owned(),
            onset_w: 2.0,
            tail_w: 3.0,
            gan: false,
            amp: false,
            periodicity_w: 0.0,
            hnr_w: 0.0,
            transient_w: 0.0,
            init_from: None,
        }
    }
}

/// `[ckpt]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CkptCfg {
    /// Write a checkpoint every this many steps.
    pub every_steps: u64,
    /// Keep the newest N checkpoints (the best is always kept).
    pub keep: u32,
}

impl Default for CkptCfg {
    fn default() -> Self {
        Self {
            every_steps: 500,
            keep: 5,
        }
    }
}

/// `[eval]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvalCfg {
    /// Evaluate every this many steps.
    pub every_steps: u64,
    /// Path of the eval-clip list (one prepared key per line, dev split).
    pub clips: String,
    /// Cap on the number of clips rendered per eval.
    pub max_items: u32,
}

impl Default for EvalCfg {
    fn default() -> Self {
        Self {
            every_steps: 500,
            clips: "configs/eval-clips.txt".to_owned(),
            max_items: 64,
        }
    }
}

/// A whole run config. `name` is required; every section defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
// A key this binary does not know is an error, never a silent no-op: a
// misspelling, or a config written by a newer build, must not train
// something other than what it says. See `from_toml`.
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Run name; the run id is `YYYYMMDD-HHMMSS-<name>`.
    pub name: String,
    /// `[model]`.
    #[serde(default)]
    pub model: ModelCfg,
    /// `[data]`.
    #[serde(default)]
    pub data: DataCfg,
    /// `[train]`.
    #[serde(default)]
    pub train: TrainCfg,
    /// `[ckpt]`.
    #[serde(default)]
    pub ckpt: CkptCfg,
    /// `[eval]`.
    #[serde(default)]
    pub eval: EvalCfg,
    /// `[augment]`.
    #[serde(default)]
    pub augment: AugmentCfg,
}

impl RunConfig {
    /// Parse a TOML document.
    /// Parse, rejecting any key this build does not know. A silently
    /// ignored key is the worst outcome: a typo, or a config written by
    /// a newer build (a dashboard left running across an upgrade), would
    /// otherwise train something other than what the file says.
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Render as TOML (the resolved `config.toml` written into a run dir).
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// A key this build does not know is an error, wherever it sits —
    /// the run's own shape depends on keys like `[train] init_from`, and
    /// a dashboard left running across an upgrade used to drop them
    /// silently and train the wrong thing.
    #[test]
    fn an_unknown_key_is_an_error_not_a_silent_no_op() {
        for (text, want) in [
            ("name = \"t\"\nsteps = 10\n", "steps"),
            (
                "name = \"t\"\n[train]\nintit_from = \"run\"\n",
                "intit_from",
            ),
            ("name = \"t\"\n[model]\nprofle = \"full\"\n", "profle"),
            ("name = \"t\"\n[data]\nshard = \"s\"\n", "shard"),
            ("name = \"t\"\n[eval]\nclip = \"c\"\n", "clip"),
            ("name = \"t\"\n[ckpt]\nkeeep = 2\n", "keeep"),
            ("name = \"t\"\n[augment]\nrxshare = 0.5\n", "rxshare"),
        ] {
            let e = RunConfig::from_toml(text)
                .expect_err(&format!("{text:?} parsed"))
                .to_string();
            assert!(e.contains(want), "{text:?}: {e}");
        }
        // The keys it does know still parse, including the newest.
        let ok =
            RunConfig::from_toml("name = \"t\"\n[train]\ninit_from = \"20260912-000009-gen\"\n")
                .unwrap();
        assert_eq!(ok.train.init_from.as_deref(), Some("20260912-000009-gen"));
    }

    /// The spec's §7 example, written out as real TOML: the spec condenses
    /// each section onto the `[table]` line, which TOML does not allow.
    const SPEC_EXAMPLE: &str = r#"
name = "seed-dstar-full-ll5"
[model]
profile = "full"      # full | lite | super-lite
lookahead = 5         # AMBE 20 ms frames
[data]
source = "shards"     # pipeline | shards
shards = "seed-dstar" # or mode = "dstar" for pipeline
mode = "dstar"
crop_s = 2.0
[train]
steps = 20000
batch = 16
lr = 2e-4
seed = 1
device = "cpu"   # cpu | mps | cuda:0 | rocm:0
onset_w = 2.0
tail_w = 3.0
gan = false
amp = false
[ckpt]
every_steps = 500
keep = 5
[eval]
every_steps = 500
clips = "configs/eval-clips.txt"
max_items = 64
[augment]
rx_share = 0.3
hum = true
broadband = true
whine = true
colouring = true
"#;

    #[test]
    fn spec_example_equals_the_defaults() {
        let cfg = RunConfig::from_toml(SPEC_EXAMPLE).unwrap();
        let mut expected = RunConfig {
            name: "seed-dstar-full-ll5".to_owned(),
            ..RunConfig::default()
        };
        expected.data.shards = Some("seed-dstar".to_owned());
        expected.data.mode = Some(VocoderMode::Dstar);
        assert_eq!(cfg, expected);
        assert_eq!(cfg.data.modes().unwrap(), vec![VocoderMode::Dstar]);
        assert_eq!(
            RunConfig::default().data.modes().unwrap(),
            vec![VocoderMode::Dstar],
            "no mode at all is dstar"
        );
        assert!(!cfg.model.mode_embed);
        assert_eq!(cfg.model.profile, Profile::Full);
        assert_eq!(cfg.data.shards.as_deref(), Some("seed-dstar"));
        assert_eq!(cfg.train.device, "cpu");
        // ll5 is 100 ms and ll20 400 ms, as realtime.md promises.
        assert_eq!(cfg.model.lookahead_ms(), 100);
        let ll20 = ModelCfg {
            lookahead: 20,
            ..ModelCfg::default()
        };
        assert_eq!(ll20.lookahead_ms(), 400);
    }

    #[test]
    fn minimal_config_fills_every_default() {
        let cfg = RunConfig::from_toml("name = \"x\"\n").unwrap();
        assert_eq!(cfg.train.steps, 20_000);
        assert_eq!(cfg.ckpt.keep, 5);
        assert_eq!(cfg.eval.max_items, 64);
        assert_eq!(cfg.data.crop_s, 2.0);
        assert_eq!(cfg.data.kinds, vec!["base".to_owned()]);
        assert_eq!(cfg.data.kinds().unwrap(), vec![None]);
        assert_eq!(cfg.augment.rx_share, 0.3);
        assert!(cfg.augment.hum && cfg.augment.colouring);
        assert!(
            RunConfig::from_toml("[model]\nprofile = \"lite\"\n").is_err(),
            "name is required"
        );
        let cfg = RunConfig::from_toml(
            "name = \"k\"\n[data]\nkinds = [\"base\", \"drops\"]\n[augment]\nrx_share = 0.0\nwhine = false\n",
        )
        .unwrap();
        assert_eq!(
            cfg.data.kinds().unwrap(),
            vec![None, Some(crate::AugKind::Drops)]
        );
        assert_eq!(cfg.augment.rx_share, 0.0);
        assert!(!cfg.augment.whine && cfg.augment.hum);
        assert!(
            RunConfig::from_toml("name = \"k\"\n[data]\nkinds = [\"x\"]\n")
                .unwrap()
                .data
                .kinds()
                .is_err()
        );
    }

    #[test]
    fn partial_sections_override_only_what_they_name() {
        let cfg = RunConfig::from_toml(
            "name = \"lite\"\n[model]\nprofile = \"super-lite\"\n[data]\nsource = \"pipeline\"\nmode = \"ysf-dmr\"\n[train]\ndevice = \"cuda:0\"\n",
        )
        .unwrap();
        assert_eq!(cfg.model.profile, Profile::SuperLite);
        assert_eq!(cfg.model.lookahead, 5);
        assert_eq!(cfg.data.source, DataSource::Pipeline);
        assert_eq!(cfg.data.mode, Some(VocoderMode::YsfDmr));
        assert_eq!(cfg.data.primary_mode().unwrap(), VocoderMode::YsfDmr);
        assert_eq!(cfg.train.device, "cuda:0");
        assert_eq!(cfg.train.lr, 2e-4);
    }

    #[test]
    fn modes_and_the_mode_embedding_parse() {
        use VocoderMode::{Codec2_3200, Dstar, YsfDmr};
        let cfg = RunConfig::from_toml(
            "name = \"m\"\n[model]\nmode_embed = true\n[data]\nmodes = [\"dstar\", \"codec2-3200\"]\n",
        )
        .unwrap();
        assert!(cfg.model.mode_embed);
        assert_eq!(cfg.data.mode, None);
        assert_eq!(cfg.data.modes, vec![Dstar, Codec2_3200]);
        assert_eq!(cfg.data.modes().unwrap(), vec![Dstar, Codec2_3200]);
        assert_eq!(cfg.data.primary_mode().unwrap(), Dstar);
        let text = cfg.to_toml().unwrap();
        assert!(
            text.contains("modes = [") && text.contains("\"codec2-3200\""),
            "{text}"
        );
        assert!(!text.contains("mode = "), "{text}");
        assert!(text.contains("mode_embed = true"), "{text}");
        assert_eq!(RunConfig::from_toml(&text).unwrap(), cfg);
        // `mode` beside `modes` is fine when it is one of them, an error
        // otherwise; duplicates are an error.
        let both = RunConfig::from_toml(
            "name = \"b\"\n[data]\nmode = \"codec2-3200\"\nmodes = [\"dstar\", \"codec2-3200\"]\n",
        )
        .unwrap();
        assert_eq!(both.data.modes().unwrap(), vec![Dstar, Codec2_3200]);
        let bad = RunConfig::from_toml(
            "name = \"b\"\n[data]\nmode = \"ysf-dmr\"\nmodes = [\"dstar\", \"codec2-3200\"]\n",
        )
        .unwrap();
        let err = bad.data.modes().unwrap_err().to_string();
        assert!(err.contains("ysf-dmr"), "{err}");
        let dup = DataCfg {
            modes: vec![YsfDmr, YsfDmr],
            ..DataCfg::default()
        };
        assert!(dup.modes().unwrap_err().to_string().contains("twice"));
        assert_eq!(MODE_EMBED_DIM, 16);
    }

    #[test]
    fn toml_round_trip_is_lossless() {
        let mut cfg = RunConfig {
            name: "rt".to_owned(),
            ..RunConfig::default()
        };
        cfg.data.shards = None;
        cfg.data.source = DataSource::Pipeline;
        cfg.train.gan = true;
        let text = cfg.to_toml().unwrap();
        assert!(!text.contains("shards ="), "{text}");
        let back = RunConfig::from_toml(&text).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn profile_budgets_and_names() {
        assert_eq!(Profile::Full.param_budget(), 8_000_000);
        assert_eq!(Profile::Lite.param_budget(), 1_000_000);
        assert_eq!(Profile::SuperLite.param_budget(), 100_000);
        assert_eq!(Profile::SuperLite.as_str(), "super-lite");
        assert_eq!("super-lite".parse::<Profile>().unwrap(), Profile::SuperLite);
        assert_eq!(
            serde_json::to_string(&Profile::SuperLite).unwrap(),
            "\"super-lite\""
        );
        assert!("superlite".parse::<Profile>().is_err());
    }
}
