// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The waveform synthesiser: a clean log-mel in, 24 kHz audio out.
//!
//! Vocos (`charactr/vocos-mel-24khz`, MIT) as fine-tuned by the spike:
//!
//! ```text
//! mel 100 ─ conv k7 (pad 3 | 3) ─► 512 ─ LayerNorm
//!   ─ 8 × block: dwconv k7 (pad 3 | 3), LayerNorm, Linear 1536, GELU,
//!                Linear 512, × γ, + residual
//!   ─ LayerNorm ─ Linear 1026 ─► magnitude = exp(·) clipped at 100, phase
//!   ─ complex spectrum ─ iSTFT (1024 / 256, periodic Hann, centred)
//! ```
//!
//! The iSTFT is `torch.istft(center=True)`: overlap-add of the windowed
//! inverse FFTs, divided by the overlap-added squared window, the first
//! and last 512 samples dropped — `256 × (T − 1)` samples for `T` frames.

use tch::{Kind, Tensor};

use super::restorer::Block;
use super::weights::{Pad, Weights};
use super::{HOP, N_FFT};

/// The synthesiser's weights.
#[derive(Debug)]
pub struct Synth {
    /// Input conv `[C, n_mels, k]` and bias.
    pub embed_w: Tensor,
    /// Its bias.
    pub embed_b: Tensor,
    /// The input conv's padding.
    pub embed_pad: Pad,
    /// `LayerNorm` after the input conv.
    pub norm_w: Tensor,
    /// Its bias.
    pub norm_b: Tensor,
    /// The blocks.
    pub blocks: Vec<Block>,
    /// The final `LayerNorm`.
    pub final_w: Tensor,
    /// Its bias.
    pub final_b: Tensor,
    /// Head linear `[n_fft + 2, C]` and bias.
    pub out_w: Tensor,
    /// Its bias.
    pub out_b: Tensor,
    /// The synthesis window `[n_fft]`.
    pub window: Tensor,
    /// `LayerNorm` epsilon.
    pub eps: f64,
    /// Ceiling on the magnitude.
    pub mag_clip: f64,
}

impl Synth {
    /// From the weights file.
    pub fn from_weights(w: &Weights) -> anyhow::Result<Self> {
        let spec = &w.manifest.synth;
        let blocks = (0..spec.layers)
            .map(|i| {
                Block::load(
                    w,
                    &format!("synth.backbone.convnext.{i}"),
                    ["dwconv", "norm", "pwconv1", "pwconv2"],
                    spec.ln_eps,
                    spec.block,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let s = Self {
            embed_w: w.get("synth.backbone.embed.weight")?,
            embed_b: w.get("synth.backbone.embed.bias")?,
            embed_pad: spec.embed,
            norm_w: w.get("synth.backbone.norm.weight")?,
            norm_b: w.get("synth.backbone.norm.bias")?,
            blocks,
            final_w: w.get("synth.backbone.final_layer_norm.weight")?,
            final_b: w.get("synth.backbone.final_layer_norm.bias")?,
            out_w: w.get("synth.head.out.weight")?,
            out_b: w.get("synth.head.out.bias")?,
            window: w.get("synth.head.istft.window")?,
            eps: spec.ln_eps,
            mag_clip: spec.mag_clip,
        };
        anyhow::ensure!(
            s.embed_w.size()[0] == spec.dim && s.out_w.size()[0] == N_FFT + 2,
            "synthesiser weights do not match the manifest: embed {:?}, out {:?}",
            s.embed_w.size(),
            s.out_w.size()
        );
        Ok(s)
    }

    /// Channel width.
    #[must_use]
    pub fn dim(&self) -> i64 {
        self.embed_w.size()[0]
    }

    /// Frames of the future the spectrum for frame `t` depends on.
    #[must_use]
    pub fn lookahead_frames(&self) -> i64 {
        self.embed_pad.right + self.blocks.iter().map(|b| b.pad.right).sum::<i64>()
    }

    /// The backbone on `mel [B, n_mels, T]`: `[B, T, C]`.
    pub fn backbone(&self, mel: &Tensor) -> Tensor {
        let c = self.dim();
        let x = mel
            .constant_pad_nd([self.embed_pad.left, self.embed_pad.right])
            .conv1d(&self.embed_w, Some(&self.embed_b), 1, 0, 1, 1)
            .transpose(1, 2)
            .layer_norm([c], Some(&self.norm_w), Some(&self.norm_b), self.eps, true)
            .transpose(1, 2);
        let mut x = x;
        for b in &self.blocks {
            x = b.forward(&x);
        }
        x.transpose(1, 2).layer_norm(
            [c],
            Some(&self.final_w),
            Some(&self.final_b),
            self.eps,
            true,
        )
    }

    /// The head on `[B, T, C]`: the complex spectrum `[B, n_fft / 2 + 1, T]`.
    pub fn spectrum(&self, x: &Tensor) -> Tensor {
        let y = x.linear(&self.out_w, Some(&self.out_b)).transpose(1, 2); // [B, n_fft+2, T]
        let bins = N_FFT / 2 + 1;
        let mag = y.narrow(1, 0, bins).exp().clamp_max(self.mag_clip);
        let p = y.narrow(1, bins, bins);
        Tensor::complex(&(&mag * p.cos()), &(&mag * p.sin()))
    }

    /// `torch.istft(center=True)` of a complex spectrum `[B, bins, T]`:
    /// `[B, HOP × (T − 1)]`.
    pub fn istft(&self, spec: &Tensor) -> Tensor {
        spec.istft(
            N_FFT,
            HOP,
            N_FFT,
            Some(&self.window),
            true,
            false,
            true,
            None,
            false,
        )
    }

    /// Batch forward: `mel [B, n_mels, T]` → audio `[B, HOP × (T − 1)]`.
    pub fn forward(&self, mel: &Tensor) -> Tensor {
        let x = self.backbone(mel);
        self.istft(&self.spectrum(&x)).to_kind(Kind::Float)
    }
}
