// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The restorer: the codec's spectrum in, the clean wideband log-mel out.
//!
//! ```text
//! [low 171 ‖ mel 100 ‖ embed(mode) 16] ─ conv k5 (pad 2 | 2) ─► 384
//!   ─ 6 × block: dwconv k7 (pad 6−r | r), LayerNorm, Linear 1152, GELU,
//!                Linear 384, × γ, + residual        r = 3, 3, 0, 0, 0, 0
//!   ─ GRU 384 ─► h
//!   ─ LayerNorm(h + x) ─ Linear 100 ─► + mel(codec)
//! ```
//!
//! Every layer is a plain tensor op on weights read from the file, so
//! the batch forward here and the frame-by-frame [`super::stream`] share
//! one definition of each layer.

use tch::{Kind, Tensor};

use super::weights::{Pad, Weights};

/// A depthwise-conv `ConvNeXt` block's weights.
#[derive(Debug)]
pub struct Block {
    /// Depthwise kernel `[C, 1, k]`.
    pub dw_w: Tensor,
    /// Its bias `[C]`.
    pub dw_b: Tensor,
    /// `LayerNorm` weight and bias `[C]`.
    pub ln_w: Tensor,
    /// `LayerNorm` bias.
    pub ln_b: Tensor,
    /// First pointwise `[I, C]`.
    pub pw1_w: Tensor,
    /// Its bias `[I]`.
    pub pw1_b: Tensor,
    /// Second pointwise `[C, I]`.
    pub pw2_w: Tensor,
    /// Its bias `[C]`.
    pub pw2_b: Tensor,
    /// Layer scale `[C]`.
    pub gamma: Tensor,
    /// `LayerNorm` epsilon.
    pub eps: f64,
    /// The depthwise conv's padding.
    pub pad: Pad,
}

impl Block {
    pub(super) fn load(
        w: &Weights,
        p: &str,
        names: [&str; 4],
        eps: f64,
        pad: Pad,
    ) -> anyhow::Result<Self> {
        let [dw, norm, pw1, pw2] = names;
        Ok(Self {
            dw_w: w.get(&format!("{p}.{dw}.weight"))?,
            dw_b: w.get(&format!("{p}.{dw}.bias"))?,
            ln_w: w.get(&format!("{p}.{norm}.weight"))?,
            ln_b: w.get(&format!("{p}.{norm}.bias"))?,
            pw1_w: w.get(&format!("{p}.{pw1}.weight"))?,
            pw1_b: w.get(&format!("{p}.{pw1}.bias"))?,
            pw2_w: w.get(&format!("{p}.{pw2}.weight"))?,
            pw2_b: w.get(&format!("{p}.{pw2}.bias"))?,
            gamma: w.get(&format!("{p}.gamma"))?,
            eps,
            pad,
        })
    }

    /// Channels.
    #[must_use]
    pub fn dim(&self) -> i64 {
        self.gamma.size1().unwrap_or(0)
    }

    /// The pointwise half on `[B, T, C]`: `LayerNorm` → pw1 → GELU → pw2 → γ.
    pub fn pointwise(&self, y: &Tensor) -> Tensor {
        let c = self.dim();
        let y = y.layer_norm([c], Some(&self.ln_w), Some(&self.ln_b), self.eps, true);
        let y = y.linear(&self.pw1_w, Some(&self.pw1_b)).gelu("none");
        y.linear(&self.pw2_w, Some(&self.pw2_b)) * &self.gamma
    }

    /// Batch forward on `x [B, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let c = self.dim();
        let y = x
            .constant_pad_nd([self.pad.left, self.pad.right])
            .conv1d(&self.dw_w, Some(&self.dw_b), 1, 0, 1, c)
            .transpose(1, 2);
        x + self.pointwise(&y).transpose(1, 2)
    }
}

/// The restorer's weights.
#[derive(Debug)]
pub struct Restorer {
    /// Mode embedding `[modes, E]`.
    pub emb: Tensor,
    /// Input conv `[C, 171 + 100 + E, k]` and bias.
    pub conv_in_w: Tensor,
    /// Its bias.
    pub conv_in_b: Tensor,
    /// The input conv's padding.
    pub conv_in_pad: Pad,
    /// The blocks.
    pub blocks: Vec<Block>,
    /// GRU input weights `[3C, C]`.
    pub gru_w_ih: Tensor,
    /// GRU hidden weights `[3C, C]`.
    pub gru_w_hh: Tensor,
    /// GRU input bias `[3C]`.
    pub gru_b_ih: Tensor,
    /// GRU hidden bias `[3C]`.
    pub gru_b_hh: Tensor,
    /// Final `LayerNorm` weight and bias.
    pub ln_w: Tensor,
    /// Final `LayerNorm` bias.
    pub ln_b: Tensor,
    /// Output linear `[n_mels, C]` and bias.
    pub out_w: Tensor,
    /// Its bias.
    pub out_b: Tensor,
    /// `LayerNorm` epsilon.
    pub eps: f64,
}

impl Restorer {
    /// From the weights file.
    pub fn from_weights(w: &Weights) -> anyhow::Result<Self> {
        let spec = &w.manifest.restorer;
        let blocks = spec
            .blocks
            .iter()
            .enumerate()
            .map(|(i, &pad)| {
                Block::load(
                    w,
                    &format!("restorer.blocks.{i}"),
                    ["dw", "norm", "pw1", "pw2"],
                    spec.ln_eps,
                    pad,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let r = Self {
            emb: w.get("restorer.emb.weight")?,
            conv_in_w: w.get("restorer.conv_in.weight")?,
            conv_in_b: w.get("restorer.conv_in.bias")?,
            conv_in_pad: spec.conv_in,
            blocks,
            gru_w_ih: w.get("restorer.gru.weight_ih_l0")?,
            gru_w_hh: w.get("restorer.gru.weight_hh_l0")?,
            gru_b_ih: w.get("restorer.gru.bias_ih_l0")?,
            gru_b_hh: w.get("restorer.gru.bias_hh_l0")?,
            ln_w: w.get("restorer.norm.weight")?,
            ln_b: w.get("restorer.norm.bias")?,
            out_w: w.get("restorer.out.weight")?,
            out_b: w.get("restorer.out.bias")?,
            eps: spec.ln_eps,
        };
        anyhow::ensure!(
            r.conv_in_w.size()[0] == spec.dim && r.emb.size()[1] == spec.embed_dim,
            "restorer weights do not match the manifest: conv_in {:?}, emb {:?}",
            r.conv_in_w.size(),
            r.emb.size()
        );
        Ok(r)
    }

    /// Channel width.
    #[must_use]
    pub fn dim(&self) -> i64 {
        self.conv_in_w.size()[0]
    }

    /// Modes the embedding covers.
    #[must_use]
    pub fn n_modes(&self) -> i64 {
        self.emb.size()[0]
    }

    /// Frames of the future the output for frame `t` depends on.
    #[must_use]
    pub fn lookahead_frames(&self) -> i64 {
        self.conv_in_pad.right + self.blocks.iter().map(|b| b.pad.right).sum::<i64>()
    }

    /// The input conv's input `[B, 171 + 100 + E, T]` for `low [B, 171, T]`,
    /// `mel [B, 100, T]` and a mode index.
    pub fn stack_input(&self, low: &Tensor, mel: &Tensor, mode: i64) -> Tensor {
        let (b, _, t) = low.size3().unwrap_or((1, 0, 0));
        let e = self
            .emb
            .narrow(0, mode, 1)
            .transpose(0, 1)
            .unsqueeze(0)
            .expand([b, -1, t], false);
        Tensor::cat(&[low, mel, &e], 1)
    }

    /// The trunk on a stacked input `[B, I, T]`: everything before the GRU.
    pub fn trunk(&self, input: &Tensor) -> Tensor {
        let mut x = input
            .constant_pad_nd([self.conv_in_pad.left, self.conv_in_pad.right])
            .conv1d(&self.conv_in_w, Some(&self.conv_in_b), 1, 0, 1, 1);
        for b in &self.blocks {
            x = b.forward(&x);
        }
        x
    }

    /// The GRU over `x [B, T, C]` from `h0 [1, B, C]`: `(output, hn)`.
    pub fn gru(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        x.gru(
            h0,
            &[
                &self.gru_w_ih,
                &self.gru_w_hh,
                &self.gru_b_ih,
                &self.gru_b_hh,
            ],
            true,
            1,
            0.0,
            false,
            false,
            true,
        )
    }

    /// The head on `h + x` `[B, T, C]`: `[B, T, n_mels]` of correction.
    pub fn head(&self, hx: &Tensor) -> Tensor {
        let c = self.dim();
        hx.layer_norm([c], Some(&self.ln_w), Some(&self.ln_b), self.eps, true)
            .linear(&self.out_w, Some(&self.out_b))
    }

    /// Batch forward: `low [B, 171, T]`, `mel [B, 100, T]`, one mode for
    /// the batch → the predicted clean log-mel `[B, 100, T]`.
    pub fn forward(&self, low: &Tensor, mel: &Tensor, mode: i64) -> Tensor {
        let x = self
            .trunk(&self.stack_input(low, mel, mode))
            .transpose(1, 2); // [B,T,C]
        let b = x.size3().map_or(1, |s| s.0);
        let h0 = Tensor::zeros([1, b, self.dim()], (Kind::Float, x.device()));
        let (h, _) = self.gru(&x, &h0);
        mel + self.head(&(h + &x)).transpose(1, 2)
    }
}
