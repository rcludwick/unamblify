// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The exported weights file and its manifest.
//!
//! `scripts/spike/export_weights.py` writes one safetensors file holding
//! every tensor the runtime needs — the restorer, the synthesiser, the
//! mel filterbank, the resampling kernel and the analysis window — beside
//! a JSON manifest naming the mode order, the convolution pads and the
//! constants the features are defined by. The two are read together; a
//! manifest whose constants disagree with this runtime's is refused,
//! since a model evaluated on features it was not trained on is worse
//! than no model (experiment log #38).

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use tch::{Device, Tensor};

use super::{HOP, INPUT_RATE, LOW_BINS, N_FFT, N_MELS, SAMPLE_RATE};

/// One convolution's kernel and how it is padded: `left` frames of the
/// past and `right` of the future.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub struct Pad {
    /// Kernel size in frames.
    pub kernel: i64,
    /// Zero frames prepended.
    pub left: i64,
    /// Zero frames appended: the layer's lookahead.
    pub right: i64,
}

/// The restorer's architecture as exported.
#[derive(Debug, Clone, Deserialize)]
pub struct RestorerSpec {
    /// Channel width.
    pub dim: i64,
    /// Width of the pointwise expansion.
    pub inter: i64,
    /// Mode embedding width.
    pub embed_dim: i64,
    /// The input convolution.
    pub conv_in: Pad,
    /// The blocks, in order.
    pub blocks: Vec<Pad>,
    /// `LayerNorm` epsilon.
    pub ln_eps: f64,
    /// Frames of lookahead the whole restorer needs.
    pub lookahead_frames: i64,
}

/// The synthesiser's architecture as exported.
#[derive(Debug, Clone, Deserialize)]
pub struct SynthSpec {
    /// Channel width.
    pub dim: i64,
    /// Width of the pointwise expansion.
    pub inter: i64,
    /// Number of blocks.
    pub layers: i64,
    /// The input convolution.
    pub embed: Pad,
    /// Every block's depthwise convolution.
    pub block: Pad,
    /// `LayerNorm` epsilon.
    pub ln_eps: f64,
    /// Ceiling on the predicted magnitude.
    pub mag_clip: f64,
    /// Frames of lookahead the whole synthesiser needs.
    pub lookahead_frames: i64,
}

/// The resampler's kernel geometry, as `torchaudio.functional.resample`
/// builds it.
#[derive(Debug, Clone, Deserialize)]
pub struct ResampleSpec {
    /// Input rate over the gcd.
    pub orig_freq: i64,
    /// Output rate over the gcd.
    pub new_freq: i64,
    /// Half-width of the kernel in input samples.
    pub width: i64,
}

/// `pipeline.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// Manifest format.
    pub format: u32,
    /// Mode order: the index the restorer's embedding is fed.
    pub modes: Vec<String>,
    /// Working sample rate.
    pub sample_rate: i64,
    /// Rate of the codec output fed in.
    pub input_rate: i64,
    /// STFT size.
    pub n_fft: i64,
    /// STFT hop.
    pub hop: i64,
    /// Mel bands.
    pub n_mels: i64,
    /// Bins of the low-band log spectrum.
    pub low_bins: i64,
    /// Floor under the low band before its log.
    pub log_clamp: f64,
    /// Floor under the mel before its log.
    pub mel_clamp: f64,
    /// The resampler.
    pub resample: ResampleSpec,
    /// The restorer.
    pub restorer: RestorerSpec,
    /// The synthesiser.
    pub synth: SynthSpec,
}

/// The tensors and the manifest, loaded.
#[derive(Debug)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    /// The manifest.
    pub manifest: Manifest,
}

impl Weights {
    /// Load `<stem>.safetensors` and `<stem>.json`, or a `.safetensors`
    /// path whose `.json` sits beside it.
    pub fn load(path: &Path, device: Device) -> anyhow::Result<Self> {
        let path = if path.extension().is_some_and(|e| e == "json") {
            path.with_extension("safetensors")
        } else {
            path.to_path_buf()
        };
        let manifest_path = path.with_extension("json");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)
                .with_context(|| manifest_path.display().to_string())?,
        )
        .with_context(|| manifest_path.display().to_string())?;
        anyhow::ensure!(
            manifest.format == 1,
            "{}: manifest format {} is not 1",
            manifest_path.display(),
            manifest.format
        );
        for (what, got, want) in [
            ("sample_rate", manifest.sample_rate, SAMPLE_RATE),
            ("input_rate", manifest.input_rate, INPUT_RATE),
            ("n_fft", manifest.n_fft, N_FFT),
            ("hop", manifest.hop, HOP),
            ("n_mels", manifest.n_mels, N_MELS),
            ("low_bins", manifest.low_bins, LOW_BINS),
        ] {
            anyhow::ensure!(
                got == want,
                "{}: {what} = {got}, this runtime is built for {want}",
                manifest_path.display()
            );
        }
        for (what, got, want) in [
            ("log_clamp", manifest.log_clamp, super::LOG_CLAMP),
            ("mel_clamp", manifest.mel_clamp, super::MEL_CLAMP),
        ] {
            anyhow::ensure!(
                (got - want).abs() <= want * 1e-6,
                "{}: {what} = {got}, this runtime is built for {want}",
                manifest_path.display()
            );
        }
        let tensors = Tensor::read_safetensors(&path)
            .with_context(|| path.display().to_string())?
            .into_iter()
            .map(|(k, t)| (k, t.to_device(device)))
            .collect();
        Ok(Self { tensors, manifest })
    }

    /// Build from tensors already in memory (tests, the bench).
    #[must_use]
    pub fn from_parts(tensors: HashMap<String, Tensor>, manifest: Manifest) -> Self {
        Self { tensors, manifest }
    }

    /// The tensor named `name`.
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor> {
        self.tensors
            .get(name)
            .map(Tensor::shallow_clone)
            .with_context(|| format!("weights file has no tensor {name}"))
    }

    /// Every tensor name, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tensors.keys().cloned().collect();
        v.sort();
        v
    }
}
