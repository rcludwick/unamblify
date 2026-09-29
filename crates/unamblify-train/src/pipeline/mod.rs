// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The restorer + waveform-synthesiser pipeline, for inference.
//!
//! The model of `docs/design/training.md` "Candidate 3, concretely",
//! trained as Python spikes (`scripts/spike/`) and exported by
//! `scripts/spike/export_weights.py` into one weights file this module
//! runs: codec output at 8 kHz in, 24 kHz speech out.
//!
//! ```text
//! x8 ─ features (resample, STFT, low band + mel) ─► restorer ─► clean mel
//!                                                  ─► synth ─► x24
//! ```
//!
//! [`Pipeline::run`] is the batch path, one utterance at a time;
//! [`stream::Stream`] the frame-by-frame path with the same numbers.
//! Everything here is inference on weights read from the file: no
//! `VarStore`, no gradients, and the exact layer definitions the spike
//! trained, so the reference clips the exporter writes can be reproduced
//! (the `reference_matches_pytorch` test).

pub mod features;
pub mod restorer;
pub mod stream;
pub mod synth;
pub mod weights;

use std::path::Path;

use tch::{Device, Kind, Tensor};

pub use features::Features;
pub use restorer::Restorer;
pub use stream::Stream;
pub use synth::Synth;
pub use weights::{Manifest, Weights};

/// The working sample rate.
pub const SAMPLE_RATE: i64 = 24_000;
/// The rate of the codec output fed in.
pub const INPUT_RATE: i64 = 8_000;
/// STFT size at [`SAMPLE_RATE`].
pub const N_FFT: i64 = 1024;
/// STFT hop: one feature frame, 10.67 ms.
pub const HOP: i64 = 256;
/// Mel bands.
pub const N_MELS: i64 = 100;
/// Bins of the low-band log spectrum fed to the restorer (0–4 kHz).
pub const LOW_BINS: i64 = 171;
/// Floor under the low-band magnitude before its log (the spike's).
pub const LOG_CLAMP: f64 = 1e-5;
/// Floor under the mel before its log (Vocos's `safe_log`).
pub const MEL_CLAMP: f64 = 1e-7;

/// The loaded pipeline.
#[derive(Debug)]
pub struct Pipeline {
    /// The feature extractor.
    pub features: Features,
    /// The restorer.
    pub restorer: Restorer,
    /// The synthesiser.
    pub synth: Synth,
    /// Mode order: the index each name is fed as.
    pub modes: Vec<String>,
}

impl Pipeline {
    /// Load from `<stem>.safetensors` + `<stem>.json`.
    pub fn load(path: &Path, device: Device) -> anyhow::Result<Self> {
        Self::from_weights(&Weights::load(path, device)?)
    }

    /// Build from weights already loaded.
    pub fn from_weights(w: &Weights) -> anyhow::Result<Self> {
        Ok(Self {
            features: Features::from_weights(w)?,
            restorer: Restorer::from_weights(w)?,
            synth: Synth::from_weights(w)?,
            modes: w.manifest.modes.clone(),
        })
    }

    /// The index of a mode name.
    pub fn mode_index(&self, mode: &str) -> anyhow::Result<i64> {
        self.modes
            .iter()
            .position(|m| m == mode)
            .map(|i| i64::try_from(i).unwrap_or(0))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "mode {mode} is not in the weights file (it holds {})",
                    self.modes.join(", ")
                )
            })
    }

    /// Frames of the future the audio for a frame depends on, restorer
    /// and synthesiser together; the iSTFT adds two more before a sample
    /// is final, the centred STFT two in front (see [`stream`]).
    #[must_use]
    pub fn lookahead_frames(&self) -> i64 {
        self.restorer.lookahead_frames() + self.synth.lookahead_frames()
    }

    /// The restorer's output for `x8 [N]`: the predicted clean log-mel
    /// `[N_MELS, T]`.
    pub fn restore_mel(&self, x8: &Tensor, mode: i64) -> Tensor {
        let (low, mel) = self.features.compute(x8);
        self.restorer
            .forward(&low.unsqueeze(0), &mel.unsqueeze(0), mode)
            .squeeze_dim(0)
    }

    /// [`load`](Self::load) on the CPU, for callers that do not link tch.
    pub fn load_cpu(path: &Path) -> anyhow::Result<Self> {
        Self::load(path, Device::Cpu)
    }

    /// [`run`](Self::run) over plain samples: 8 kHz in, 24 kHz out.
    pub fn run_slice(&self, x8: &[f32], mode: i64) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            i64::try_from(x8.len()).unwrap_or(i64::MAX) >= HOP,
            "the input is shorter than one frame"
        );
        let x = Tensor::from_slice(x8).to_device(self.features.device());
        let (_, wav) = self.run(&x, mode);
        Ok(Vec::<f32>::try_from(
            wav.contiguous().to_device(Device::Cpu),
        )?)
    }

    /// Codec output `x8 [N]` at 8 kHz → `(mel [N_MELS, T], x24 [HOP × (T − 1)])`.
    pub fn run(&self, x8: &Tensor, mode: i64) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let mel = self.restore_mel(x8, mode);
            let wav = self.synth.forward(&mel.unsqueeze(0)).squeeze_dim(0);
            (mel, wav)
        })
    }
}

/// Random weights of the shapes a manifest describes: for the bench and
/// the tests, where only the shapes and the arithmetic matter.
#[must_use]
pub fn synthetic_weights(manifest: &Manifest, device: Device) -> Weights {
    use std::collections::HashMap;
    let r = &manifest.restorer;
    let s = &manifest.synth;
    let opts = (Kind::Float, device);
    let n = |shape: &[i64]| Tensor::randn(shape, opts) * 0.05;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let block =
        |t: &mut HashMap<String, Tensor>, p: &str, names: [&str; 4], c: i64, i: i64, k: i64| {
            let [dw, norm, pw1, pw2] = names;
            t.insert(format!("{p}.{dw}.weight"), n(&[c, 1, k]));
            t.insert(format!("{p}.{dw}.bias"), n(&[c]));
            t.insert(format!("{p}.{norm}.weight"), Tensor::ones([c], opts));
            t.insert(format!("{p}.{norm}.bias"), Tensor::zeros([c], opts));
            t.insert(format!("{p}.{pw1}.weight"), n(&[i, c]));
            t.insert(format!("{p}.{pw1}.bias"), n(&[i]));
            t.insert(format!("{p}.{pw2}.weight"), n(&[c, i]));
            t.insert(format!("{p}.{pw2}.bias"), n(&[c]));
            t.insert(format!("{p}.gamma"), Tensor::full([c], 0.1, opts));
        };
    let modes = i64::try_from(manifest.modes.len()).unwrap_or(1).max(1);
    t.insert("restorer.emb.weight".into(), n(&[modes, r.embed_dim]));
    t.insert(
        "restorer.conv_in.weight".into(),
        n(&[r.dim, LOW_BINS + N_MELS + r.embed_dim, r.conv_in.kernel]),
    );
    t.insert("restorer.conv_in.bias".into(), n(&[r.dim]));
    for (i, b) in r.blocks.iter().enumerate() {
        block(
            &mut t,
            &format!("restorer.blocks.{i}"),
            ["dw", "norm", "pw1", "pw2"],
            r.dim,
            r.inter,
            b.kernel,
        );
    }
    t.insert("restorer.gru.weight_ih_l0".into(), n(&[3 * r.dim, r.dim]));
    t.insert("restorer.gru.weight_hh_l0".into(), n(&[3 * r.dim, r.dim]));
    t.insert("restorer.gru.bias_ih_l0".into(), n(&[3 * r.dim]));
    t.insert("restorer.gru.bias_hh_l0".into(), n(&[3 * r.dim]));
    t.insert("restorer.norm.weight".into(), Tensor::ones([r.dim], opts));
    t.insert("restorer.norm.bias".into(), Tensor::zeros([r.dim], opts));
    t.insert("restorer.out.weight".into(), n(&[N_MELS, r.dim]));
    t.insert("restorer.out.bias".into(), n(&[N_MELS]));
    t.insert(
        "synth.backbone.embed.weight".into(),
        n(&[s.dim, N_MELS, s.embed.kernel]),
    );
    t.insert("synth.backbone.embed.bias".into(), n(&[s.dim]));
    t.insert(
        "synth.backbone.norm.weight".into(),
        Tensor::ones([s.dim], opts),
    );
    t.insert(
        "synth.backbone.norm.bias".into(),
        Tensor::zeros([s.dim], opts),
    );
    for i in 0..s.layers {
        block(
            &mut t,
            &format!("synth.backbone.convnext.{i}"),
            ["dwconv", "norm", "pwconv1", "pwconv2"],
            s.dim,
            s.inter,
            s.block.kernel,
        );
    }
    t.insert(
        "synth.backbone.final_layer_norm.weight".into(),
        Tensor::ones([s.dim], opts),
    );
    t.insert(
        "synth.backbone.final_layer_norm.bias".into(),
        Tensor::zeros([s.dim], opts),
    );
    t.insert("synth.head.out.weight".into(), n(&[N_FFT + 2, s.dim]));
    t.insert("synth.head.out.bias".into(), n(&[N_FFT + 2]));
    t.insert(
        "synth.head.istft.window".into(),
        Tensor::hann_window(N_FFT, opts),
    );
    t.insert("features.window".into(), Tensor::hann_window(N_FFT, opts));
    t.insert(
        "features.mel_window".into(),
        Tensor::hann_window(N_FFT, opts),
    );
    // A crude triangular filterbank and the torchaudio-shaped kernel: the
    // real ones come from the file; these only have to have the shapes.
    t.insert(
        "features.mel_fbank".into(),
        Tensor::rand([N_FFT / 2 + 1, N_MELS], opts) / 100.0,
    );
    let rs = &manifest.resample;
    t.insert(
        "features.resample_kernel".into(),
        n(&[rs.new_freq, 1, 2 * rs.width + rs.orig_freq]),
    );
    Weights::from_parts(t, manifest.clone())
}

/// The manifest the exporter writes for the spike's architecture, for
/// building [`synthetic_weights`] without a file.
#[must_use]
pub fn spike_manifest() -> Manifest {
    let block = |right: i64| weights::Pad {
        kernel: 7,
        left: 6 - right,
        right,
    };
    Manifest {
        format: 1,
        modes: ["ysf-dmr", "dstar", "codec2-3200", "codec2-1600"]
            .map(String::from)
            .to_vec(),
        sample_rate: SAMPLE_RATE,
        input_rate: INPUT_RATE,
        n_fft: N_FFT,
        hop: HOP,
        n_mels: N_MELS,
        low_bins: LOW_BINS,
        log_clamp: LOG_CLAMP,
        mel_clamp: MEL_CLAMP,
        resample: weights::ResampleSpec {
            orig_freq: 1,
            new_freq: 3,
            width: 7,
        },
        restorer: weights::RestorerSpec {
            dim: 384,
            inter: 1152,
            embed_dim: 16,
            conv_in: weights::Pad {
                kernel: 5,
                left: 2,
                right: 2,
            },
            blocks: vec![block(3), block(3), block(0), block(0), block(0), block(0)],
            ln_eps: 1e-5,
            lookahead_frames: 8,
        },
        synth: weights::SynthSpec {
            dim: 512,
            inter: 1536,
            layers: 8,
            embed: block(3),
            block: block(3),
            ln_eps: 1e-6,
            mag_clip: 100.0,
            lookahead_frames: 27,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of codec output through random weights of the exported
    /// shapes: the frames and the audio come out the sizes the spike's
    /// `PyTorch` produces (`1 + 3N / 256` frames, `256 (T − 1)` samples).
    #[test]
    fn shapes_follow_the_spike() {
        let w = synthetic_weights(&spike_manifest(), Device::Cpu);
        let p = Pipeline::from_weights(&w).unwrap();
        let x8 = Tensor::rand([8_000], (Kind::Float, Device::Cpu)) - 0.5;
        let (mel, wav) = p.run(&x8, p.mode_index("dstar").unwrap());
        let t = 1 + 3 * 8_000 / HOP;
        assert_eq!(mel.size(), [N_MELS, t]);
        assert_eq!(wav.size(), [HOP * (t - 1)]);
        assert_eq!(p.lookahead_frames(), 8 + 27);
        assert!(p.mode_index("fm").is_err());
    }

    /// Against the exporter's reference clips (`UNAMBLIFY_PIPELINE_EXPORT`
    /// names the directory): the features, the restorer's mel and the
    /// audio, each within a tolerance that the resampler's float order
    /// sets. Ignored unless the directory is given.
    #[test]
    #[ignore = "needs an export directory in UNAMBLIFY_PIPELINE_EXPORT"]
    fn reference_matches_pytorch() {
        let Ok(dir) = std::env::var("UNAMBLIFY_PIPELINE_EXPORT") else {
            eprintln!("UNAMBLIFY_PIPELINE_EXPORT unset; skipping");
            return;
        };
        let dir = Path::new(&dir);
        let p = Pipeline::load(&dir.join("pipeline.safetensors"), Device::Cpu).unwrap();
        let refs: std::collections::HashMap<String, Tensor> =
            Tensor::read_safetensors(dir.join("references.safetensors"))
                .unwrap()
                .into_iter()
                .collect();
        let max = |a: &Tensor, b: &Tensor| f64::try_from((a - b).abs().max()).unwrap();
        let rms = |a: &Tensor, b: &Tensor| {
            f64::try_from((a - b).square().mean(Kind::Float).sqrt()).unwrap()
        };
        for mode in ["dstar", "codec2-3200"] {
            let load = |what: &str| refs[&format!("{mode}.{what}")].shallow_clone();
            let (x8, r_low, r_degmel, r_mel, r_wav) = (
                load("x8"),
                load("low"),
                load("degmel"),
                load("mel"),
                load("wav24"),
            );
            let (low, degmel) = p.features.compute(&x8);
            // Log features: the worst bin is always one at the floor of the resampler's stop band; the linear
            // magnitude, relative to the clip's loudest bin, is the number that says whether the spectrum is right.
            let e_low = max(&low, &r_low);
            let e_degmel = max(&degmel, &r_degmel);
            let lin = |a: &Tensor, b: &Tensor| {
                max(&a.exp(), &b.exp()) / f64::try_from(b.exp().max()).unwrap()
            };
            let (l_low, l_degmel) = (lin(&low, &r_low), lin(&degmel, &r_degmel));
            let mode_ix = p.mode_index(mode).unwrap();
            let mel_on_ref = p
                .restorer
                .forward(&r_low.unsqueeze(0), &r_degmel.unsqueeze(0), mode_ix)
                .squeeze_dim(0);
            let e_mel_on_ref = max(&mel_on_ref, &r_mel);
            let wav_on_ref = p.synth.forward(&r_mel.unsqueeze(0)).squeeze_dim(0);
            let peak = f64::try_from(r_wav.abs().max()).unwrap();
            let (e_wav_on_ref, r_wav_on_ref) = (
                max(&wav_on_ref, &r_wav) / peak,
                rms(&wav_on_ref, &r_wav) / peak,
            );
            let (mel, wav) = p.run(&x8, mode_ix);
            let (e_mel, l_mel) = (max(&mel, &r_mel), lin(&mel, &r_mel));
            let (e_wav, r_e_wav) = (max(&wav, &r_wav) / peak, rms(&wav, &r_wav) / peak);
            eprintln!(
                "{mode}: features max|Δlog| low {e_low:.1e} mel {e_degmel:.1e}, linear/peak low {l_low:.1e} mel {l_degmel:.1e}\n\
                 {mode}: restorer on the reference features max|Δ| {e_mel_on_ref:.1e}\n\
                 {mode}: synth on the reference mel max|Δ|/peak {e_wav_on_ref:.1e} rms/peak {r_wav_on_ref:.1e}\n\
                 {mode}: end to end mel max|Δlog| {e_mel:.1e} linear/peak {l_mel:.1e}; wav max|Δ|/peak {e_wav:.1e} rms/peak {r_e_wav:.1e}"
            );
            assert!(l_low < 1e-6, "{mode}: low band off by {l_low} of the peak");
            assert!(
                l_degmel < 1e-6,
                "{mode}: input mel off by {l_degmel} of the peak"
            );
            assert!(
                e_mel_on_ref < 1e-4,
                "{mode}: the restorer is off by {e_mel_on_ref} on the reference features"
            );
            assert!(
                e_wav_on_ref < 1e-3,
                "{mode}: the synthesiser is off by {e_wav_on_ref} of the peak on the reference mel"
            );
            assert!(
                l_mel < 1e-4,
                "{mode}: the predicted mel is off by {l_mel} of the peak"
            );
            assert!(
                e_wav < 1e-2,
                "{mode}: the audio is off by {e_wav} of the peak"
            );
        }
    }
}
