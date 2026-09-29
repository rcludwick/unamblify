// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Candidate 1 (spec §5): a frame-rate conv + GRU feature net driving a
//! LACE-shaped adaptive filter on the degraded 8 kHz signal, plus a
//! residual 8 → 16 kHz upsampler head.
//!
//! ```text
//! deg8 [B,1,N] ─ frame conv (k=160, hop 80, causal) ─► [B,F,T]   T = N/80
//!             ─ context conv over frames, pad (PAST_CTX, 2L) ─► [B,F,T]
//!             ─ (mode_embed) ⊕ embed(mode) [B,16] at every frame ─► [B,T,F+16]
//!             ─ 2-layer GRU ─► g [B,T,F]
//!   g ─ fir_head ─► 16 FIR taps / frame    ─┐
//!   g ─ pitch_head ─► pitch-lag mix + gain ─┴► y8 = FIR(deg8) + gain·comb(deg8)
//!   y8 ─ fixed half-band upsampler ─► up16
//!   (full, lite) up16 + BWE(up16, g) ─► out16 [B,1,2N]
//!   (super-lite) out16 = up16
//! ```
//!
//! **Lookahead and delay.** `[model] lookahead = L` is in AMBE 20 ms
//! frames, the unit of `docs/design/realtime.md` (ll5 = 100 ms, ll20 =
//! 400 ms); at the 10 ms feature hop that is `2L` feature frames. The
//! context conv is padded `(PAST_CTX, 2L)` over the frame axis, so the
//! feature for frame `t` sees frames `t − PAST_CTX ..= t + 2L`; every
//! later stage (GRU, heads, causal BWE convs) is strictly causal in
//! frames. Output frame `t` therefore depends on input samples
//! `< 80·(t + 2L + 1)` and nothing later, up to the fixed half-band
//! resampler: its 31-tap linear-phase kernel is centred, so `out16[i]`
//! also sees `y8` up to 7 samples (0.9 ms) past `i / 2`. The runtime
//! realises this by emitting frame `t` once AMBE frame `t + L` has
//! arrived plus the resampler's 15-sample (16 kHz) group delay — a fixed
//! delay of `L` AMBE frames (`20·L` ms) + 0.94 ms, identical for ll5 and
//! ll20 training and inference.
//!
//! **State reset.** Every training example starts the GRU from a zero
//! state (`RNN::seq`), the frame conv's left padding is zero, and the
//! context conv's right padding is zero: an example's first frames are
//! exactly what the runtime produces right after a key-up, which is what
//! the onset weighting in `losses` is for.
//!
//! **Mode conditioning.** With `[model] mode_embed = true` the net owns a
//! learned `[n_modes, MODE_EMBED_DIM]` table (one row per entry of
//! `[data] modes`); the row of each example's `mode` is concatenated to
//! the context features at every frame, so the GRU input is `F + 16`
//! wide and one model trained on several vocoders can specialise per
//! mode when told which one it is hearing. Without it the `mode` tensor
//! is ignored and the model is blind to the vocoder.
//!
//! Profile widths: full F = 256 (BWE 32 channels), lite F = 96 (BWE 16),
//! super-lite F = 32 and no BWE head. `param_count` is asserted against
//! `Profile::param_budget` by the trainer and by the tests.

use tch::nn::{self, Module, RNN};
use tch::{Device, Kind, Tensor};
use unamblify::{MODE_EMBED_DIM, Profile};

/// Feature hop at 8 kHz, samples (10 ms).
pub const HOP: i64 = 80;
/// Frame conv kernel, samples (20 ms, one frame of context to the left).
pub const FRAME_KERNEL: i64 = 2 * HOP;
/// Past frames the context conv sees.
pub const PAST_CTX: i64 = 8;
/// Short-term adaptive FIR taps.
pub const TAPS: i64 = 16;
/// Number of candidate pitch lags for the long-term term.
pub const N_LAGS: i64 = 24;
/// Shortest pitch lag, samples at 8 kHz (400 Hz).
pub const MIN_LAG: i64 = 20;
/// Longest pitch lag, samples at 8 kHz (50 Hz).
pub const MAX_LAG: i64 = 160;
/// Taps of the fixed half-band resampling filter (odd, linear phase).
pub const HALFBAND_TAPS: i64 = 31;
/// Sub-frames per feature frame for the noise path's fine gain
/// (`[model] noise_mod`): 10 per 10 ms, one per millisecond.
pub const FINE: i64 = 10;
/// Samples per fine sub-frame at 8 kHz.
pub const FINE_HOP: i64 = HOP / FINE;
/// Window, samples at 8 kHz, of the periodic path's envelope that
/// modulates the noise (2 ms: shorter than any pitch period).
pub const MOD_WINDOW: i64 = 16;
/// BWE conv kernel (causal, 16 kHz samples).
const BWE_KERNEL: i64 = 9;

/// Channel widths for a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    /// Feature / GRU width `F`.
    pub f: i64,
    /// BWE head channels; `None` = no BWE head (super-lite).
    pub bwe: Option<i64>,
}

/// Widths per profile (spec §5), at `[model] width = 1.0`.
#[must_use]
pub const fn widths(profile: Profile) -> Widths {
    match profile {
        Profile::Full => Widths {
            f: 256,
            bwe: Some(32),
        },
        Profile::Lite => Widths {
            f: 96,
            bwe: Some(16),
        },
        Profile::SuperLite => Widths { f: 32, bwe: None },
    }
}

/// The profile's widths scaled by `[model] width`, each rounded to a
/// multiple of 8 (kinder to SIMD and to the GRU's gate packing) and
/// never below 8. Parameters grow roughly with the square of this, so
/// the profile's budget — checked in [`build`] — is what stops it.
#[must_use]
pub fn scaled_widths(profile: Profile, width: f32) -> Widths {
    let w = widths(profile);
    if (width - 1.0).abs() < f32::EPSILON {
        return w;
    }
    Widths {
        f: round8(w.f, width),
        bwe: w.bwe.map(|b| round8(b, width)),
    }
}

/// `n * mult`, rounded to the nearest multiple of 8, at least 8.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn round8(n: i64, mult: f32) -> i64 {
    let scaled = (n as f32 * mult).max(1.0);
    let rounded = ((scaled / 8.0).round() as i64) * 8;
    rounded.max(8)
}

/// The candidate pitch lags, geometrically spaced in `[MIN_LAG, MAX_LAG]`.
#[must_use]
pub fn pitch_lags() -> Vec<i64> {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    (0..N_LAGS)
        .map(|i| {
            let t = i as f64 / (N_LAGS - 1) as f64;
            let lag = (MIN_LAG as f64) * ((MAX_LAG as f64) / (MIN_LAG as f64)).powf(t);
            lag.round() as i64
        })
        .collect()
}

/// Half-band windowed-sinc lowpass, `HALFBAND_TAPS` long: `h[c] = 0.5`,
/// `h[c ± 2k] = 0`, so zero-stuffing + `2h` reproduces the input samples
/// exactly and decimation is a clean anti-aliased pick of every other
/// sample. Symmetric Hann window.
#[must_use]
pub fn halfband() -> Vec<f32> {
    let n = HALFBAND_TAPS;
    let c = n / 2;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    (0..n)
        .map(|i| {
            let x = (i - c) as f64 / 2.0;
            let sinc = if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            (0.5 * sinc * w) as f32
        })
        .collect()
}

/// Repeat each frame's vector `hop` times along the time axis:
/// `[B, T, C]` → `[B, T · hop, C]`.
fn frames_to_samples(x: &Tensor, hop: i64) -> Tensor {
    let (b, t, c) = x.size3().unwrap_or((0, 0, 0));
    x.unsqueeze(2)
        .expand([b, t, hop, c], false)
        .reshape([b, t * hop, c])
}

/// Frame parameters to sample rate by **linear interpolation**: over
/// frame `t`'s `hop` samples the value ramps from frame `t − 1`'s to
/// frame `t`'s (frame 0 ramps from its own value), reaching `x[t]` at
/// the last sample. Causal — nothing from frame `t + 1` is used — and
/// free of the 10 ms steps a held parameter puts into every filter tap,
/// comb mix and gain. The runtime must interpolate the same way.
fn frames_to_samples_interp(x: &Tensor, hop: i64) -> Tensor {
    let (batch, frames, ch) = x.size3().unwrap_or((0, 0, 0));
    let prev = Tensor::cat(&[x.narrow(1, 0, 1), x.narrow(1, 0, (frames - 1).max(0))], 1); // [B,T,C]
    #[allow(clippy::cast_precision_loss)]
    let weight = (Tensor::arange(hop, (Kind::Float, x.device())) + 1.0) / hop as f64; // [hop]
    let weight = weight.view([1, 1, hop, 1]);
    let ramp: Tensor = prev.unsqueeze(2) * (1.0 - &weight) + x.unsqueeze(2) * &weight; // [B,T,hop,C]
    ramp.reshape([batch, frames * hop, ch])
}

/// The bandwidth-extension head.
#[derive(Debug)]
struct Bwe {
    cond: nn::Linear,
    conv_in: nn::Conv1D,
    conv_mid: nn::Conv1D,
    conv_out: nn::Conv1D,
}

impl Bwe {
    fn new(p: &nn::Path, f: i64, c: i64) -> Self {
        let causal = nn::ConvConfig {
            padding: 0,
            ..Default::default()
        };
        let small = nn::ConvConfig {
            padding: 0,
            ws_init: nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            ..Default::default()
        };
        Self {
            cond: nn::linear(p / "cond", f, c, nn::LinearConfig::default()),
            conv_in: nn::conv1d(p / "conv_in", 1, c, BWE_KERNEL, causal),
            conv_mid: nn::conv1d(p / "conv_mid", c, c, BWE_KERNEL, causal),
            conv_out: nn::conv1d(p / "conv_out", c, 1, BWE_KERNEL, small),
        }
    }

    /// Residual to add to `up16 [B,1,2N]` given features `g [B,T,F]`.
    fn forward(&self, up16: &Tensor, g: &Tensor) -> Tensor {
        let cond = frames_to_samples_interp(&self.cond.forward(g), 2 * HOP).transpose(1, 2); // [B,C,2N]
        let causal = |x: &Tensor| x.constant_pad_nd([BWE_KERNEL - 1, 0]);
        let a = self.conv_in.forward(&causal(up16)).leaky_relu();
        let a = a * cond.sigmoid();
        let a = self.conv_mid.forward(&causal(&a)).leaky_relu();
        self.conv_out.forward(&causal(&a))
    }
}

/// Everything in `[model]` that changes the network's tensor shapes, so
/// a checkpoint is only interchangeable with a net built from the same
/// values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetOpts {
    /// Size profile.
    pub profile: Profile,
    /// `[model] width`: multiplier on the profile's channel widths.
    pub width: f32,
    /// `[model] lookahead`, in AMBE 20 ms frames.
    pub lookahead: u32,
    /// `Some(n)` builds the mode embedding over `n` modes.
    pub embed_modes: Option<usize>,
    /// `[model] noise_head`: add the aperiodic excitation path.
    pub noise_head: bool,
    /// `[model] noise_mod`: modulate that path by the periodic path's
    /// envelope and give it a gain per millisecond.
    pub noise_mod: bool,
    /// `[model] erasure_in`: a per-frame lost-frame mask joins the
    /// features beside the mode embedding.
    pub erasure_in: bool,
}

impl Default for NetOpts {
    fn default() -> Self {
        Self {
            profile: Profile::Full,
            width: 1.0,
            lookahead: 5,
            embed_modes: None,
            noise_head: false,
            noise_mod: false,
            erasure_in: false,
        }
    }
}

/// The context conv over feature frames: `PAST_CTX` frames back and the
/// whole lookahead forward.
///
/// With `erasure_in` the lost-frame mask is one more channel *into* it, so
/// the mask gets the same window as the audio. A receiver that buffers
/// 100 ms of frames knows which of them were lost, and a model that can see
/// a gap coming can prepare for it; entering at the causal GRU instead, the
/// mask would only ever say "this frame is gone".
fn context_conv(
    p: &nn::Path,
    f: i64,
    erasure_in: bool,
    lookahead: i64,
    cfg: nn::ConvConfig,
) -> nn::Conv1D {
    nn::conv1d(
        p / "ctx",
        f + i64::from(erasure_in),
        f,
        PAST_CTX + 1 + lookahead,
        cfg,
    )
}

/// The post-filter network.
#[derive(Debug)]
pub struct Net {
    profile: Profile,
    /// Lookahead in feature frames (`2 ×` the config's AMBE frames).
    lookahead: i64,
    widths: Widths,
    frame: nn::Conv1D,
    ctx: nn::Conv1D,
    gru: nn::GRU,
    fir_head: nn::Linear,
    pitch_head: nn::Linear,
    /// `[model] noise_head`: per-frame shaping taps and a gain for an
    /// aperiodic excitation added to the narrowband output. Without it
    /// the net can only *filter* its input, and a filtered periodic
    /// signal stays periodic — see `docs/theory/what-the-codec-destroys.md`.
    noise_head: Option<nn::Linear>,
    /// `[model] noise_mod`: the noise head also emits a modulation depth
    /// and [`FINE`] sub-frame gains.
    noise_mod: bool,
    bwe: Option<Bwe>,
    /// `[model] mode_embed`: one row per mode, concatenated to the
    /// features before the GRU.
    mode_embed: Option<nn::Embedding>,
    /// `[model] erasure_in`: the context conv takes one more channel,
    /// the per-frame lost-frame mask, over the same window as the audio.
    erasure_in: bool,
    n_modes: usize,
    lags: Vec<i64>,
    delta: Tensor,
    up_kernel: Tensor,
    dec_kernel: Tensor,
}

impl Net {
    /// Feature frames per AMBE frame (20 ms / 10 ms).
    pub const FEATURE_FRAMES_PER_AMBE_FRAME: i64 = 2;

    /// Build under `p` (normally `vs.root()`) for `profile` with
    /// `lookahead` AMBE 20 ms frames; `embed_modes = Some(n)` adds the
    /// mode embedding over `n` modes. No noise head.
    #[must_use]
    pub fn new(p: &nn::Path, profile: Profile, lookahead: u32, embed_modes: Option<usize>) -> Self {
        Self::with_width(p, profile, 1.0, lookahead, embed_modes)
    }

    /// As [`new`](Self::new) with `[model] width`: the profile's widths
    /// scaled by that factor ([`scaled_widths`]). No noise head.
    #[must_use]
    pub fn with_width(
        p: &nn::Path,
        profile: Profile,
        width: f32,
        lookahead: u32,
        embed_modes: Option<usize>,
    ) -> Self {
        Self::build(
            p,
            &NetOpts {
                profile,
                width,
                lookahead,
                embed_modes,
                noise_head: false,
                noise_mod: false,
                erasure_in: false,
            },
        )
    }

    /// The general constructor: every `[model]` key that changes the
    /// tensor shapes.
    #[must_use]
    pub fn build(p: &nn::Path, opts: &NetOpts) -> Self {
        let NetOpts {
            profile,
            width,
            lookahead,
            embed_modes,
            noise_head: want_noise,
            noise_mod,
            erasure_in,
        } = *opts;
        let noise_mod = noise_mod && want_noise;
        let widths = scaled_widths(profile, width);
        let f = widths.f;
        let embed_dim = i64::try_from(MODE_EMBED_DIM).unwrap_or(16);
        let n_modes = embed_modes.unwrap_or(0);
        let mode_embed = embed_modes.map(|n| {
            nn::embedding(
                p / "mode_embed",
                i64::try_from(n).unwrap_or(1).max(1),
                embed_dim,
                nn::EmbeddingConfig::default(),
            )
        });
        let gru_in = if mode_embed.is_some() {
            f + embed_dim
        } else {
            f
        };
        let lookahead = i64::from(lookahead) * Self::FEATURE_FRAMES_PER_AMBE_FRAME;
        let no_pad = nn::ConvConfig {
            padding: 0,
            ..Default::default()
        };
        let frame = nn::conv1d(
            p / "frame",
            1,
            f,
            FRAME_KERNEL,
            nn::ConvConfig {
                stride: HOP,
                ..no_pad
            },
        );
        let ctx = context_conv(p, f, erasure_in, lookahead, no_pad);
        let gru = nn::gru(
            p / "gru",
            gru_in,
            f,
            nn::RNNConfig {
                num_layers: 2,
                batch_first: true,
                ..Default::default()
            },
        );
        let fir_head = nn::linear(p / "fir_head", f, TAPS, nn::LinearConfig::default());
        let pitch_head = nn::linear(p / "pitch_head", f, 1 + N_LAGS, nn::LinearConfig::default());
        // TAPS shaping coefficients + one gain, mirroring fir_head; with
        // noise_mod also FINE sub-frame gains and a modulation depth.
        let noise_outputs = TAPS + 1 + if noise_mod { FINE + 1 } else { 0 };
        let noise_head = want_noise.then(|| {
            nn::linear(
                p / "noise_head",
                f,
                noise_outputs,
                nn::LinearConfig::default(),
            )
        });
        let bwe = widths.bwe.map(|c| Bwe::new(&(p / "bwe"), f, c));
        let device = p.device();
        let mut delta = vec![0f32; usize::try_from(TAPS).unwrap_or(0)];
        if let Some(last) = delta.last_mut() {
            *last = 1.0;
        }
        let h = halfband();
        let dec_kernel = Tensor::from_slice(&h)
            .view([1, 1, HALFBAND_TAPS])
            .to_device(device);
        let up_kernel = &dec_kernel * 2.0;
        Self {
            profile,
            lookahead,
            widths,
            frame,
            ctx,
            gru,
            fir_head,
            pitch_head,
            noise_head,
            noise_mod,
            bwe,
            mode_embed,
            n_modes,
            erasure_in,
            lags: pitch_lags(),
            delta: Tensor::from_slice(&delta).to_device(device),
            up_kernel,
            dec_kernel,
        }
    }

    /// The profile this net was built for.
    #[must_use]
    pub const fn profile(&self) -> Profile {
        self.profile
    }

    /// Lookahead in 10 ms feature frames (`2 ×` the config's AMBE frames).
    #[must_use]
    pub const fn lookahead(&self) -> i64 {
        self.lookahead
    }

    /// Lookahead in 8 kHz input samples: how far past frame `t` the
    /// output for `t` may look.
    #[must_use]
    pub const fn lookahead_samples(&self) -> i64 {
        self.lookahead * HOP
    }

    /// Lookahead in milliseconds.
    #[must_use]
    pub const fn lookahead_ms(&self) -> i64 {
        self.lookahead_samples() / 8
    }

    /// Widths in use.
    #[must_use]
    pub const fn widths(&self) -> Widths {
        self.widths
    }

    /// Whether the aperiodic excitation path is built.
    #[must_use]
    pub const fn has_noise_head(&self) -> bool {
        self.noise_head.is_some()
    }

    /// Whether that path carries modulation and 1 ms gains.
    #[must_use]
    pub const fn has_noise_mod(&self) -> bool {
        self.noise_mod
    }

    /// Modes the embedding covers; `None` without `[model] mode_embed`.
    #[must_use]
    pub fn embed_modes(&self) -> Option<usize> {
        self.mode_embed.as_ref().map(|_| self.n_modes)
    }

    /// Per-frame features `g [B, T, F]` for `deg8 [B, 1, N]` (`N` a
    /// multiple of [`HOP`]) and `mode [B]` (int64 mode indices, used only
    /// with the embedding).
    pub fn features(&self, deg8: &Tensor, mode: &Tensor) -> Tensor {
        self.features_with(deg8, mode, None)
    }

    /// Whether the net takes the per-frame erasure mask.
    #[must_use]
    pub const fn takes_erasure(&self) -> bool {
        self.erasure_in
    }

    /// [`features`](Self::features) with the erasure mask: `erasure [B, T]`
    /// float 0/1 at the feature-frame rate (`T = N / HOP`), 1 where the
    /// frame's audio is the decoder's concealment of a lost channel frame.
    /// `None` is "nothing was lost" — zeros — which is what clean audio
    /// and a link that reported no loss both mean. Ignored by a net built
    /// without `[model] erasure_in`.
    pub fn features_with(&self, deg8: &Tensor, mode: &Tensor, erasure: Option<&Tensor>) -> Tensor {
        // Causal framing: frame t covers samples [80t − 80, 80t + 80).
        let padded = deg8.constant_pad_nd([HOP, 0]);
        let feats = self.frame.forward(&padded).leaky_relu(); // [B,F,T]
        let feats = if self.erasure_in {
            let (batch, _, frames) = feats.size3().unwrap_or((0, 0, 0));
            let e = match erasure {
                Some(e) => e
                    .to_device(feats.device())
                    .to_kind(feats.kind())
                    .reshape([batch, 1, frames]),
                None => Tensor::zeros([batch, 1, frames], (feats.kind(), feats.device())),
            };
            Tensor::cat(&[feats, e], 1) // [B,F+1,T]
        } else {
            feats
        };
        // Zero padding on the right reads as "nothing lost" past the end.
        let feats = feats.constant_pad_nd([PAST_CTX, self.lookahead]);
        let feats = self.ctx.forward(&feats).leaky_relu().transpose(1, 2); // [B,T,F]
        let feats = match &self.mode_embed {
            Some(emb) => {
                let (batch, frames, _) = feats.size3().unwrap_or((0, 0, 0));
                let embed = emb
                    .forward(&mode.to_device(feats.device()).to_kind(Kind::Int64))
                    .unsqueeze(1); // [B,1,E]
                let embed = embed.expand([batch, frames, embed.size3().map_or(0, |s| s.2)], false);
                Tensor::cat(&[feats, embed], 2) // [B,T,F+E]
            }
            None => feats,
        };
        let (g, _) = self.gru.seq(&feats); // [B,T,F]
        g
    }

    /// The adaptive-filter (narrowband) output `y8 [B, 1, N]`.
    pub fn narrowband(&self, deg8: &Tensor, g: &Tensor) -> Tensor {
        let n = deg8.size3().map_or(0, |s| s.2);
        // Short-term FIR: taps[..., j] multiplies x[n − (TAPS − 1 − j)],
        // so the identity filter is a delta at the last tap.
        let taps = &self.delta + self.fir_head.forward(g) * 0.1; // [B,T,TAPS]
        let taps_up = frames_to_samples_interp(&taps, HOP); // [B,N,TAPS]
        let unf = deg8
            .constant_pad_nd([TAPS - 1, 0])
            .unfold(2, TAPS, 1)
            .squeeze_dim(1); // [B,N,TAPS]
        let fir = (unf * taps_up).sum_dim_intlist(-1, true, Kind::Float); // [B,N,1]

        // Long-term: a gain on a soft mixture of delayed copies.
        let lt = self.pitch_head.forward(g); // [B,T,1+N_LAGS]
        let gain = lt.narrow(2, 0, 1).tanh() * 0.5;
        let mix = lt.narrow(2, 1, N_LAGS).softmax(-1, Kind::Float);
        let shifted: Vec<Tensor> = self
            .lags
            .iter()
            .map(|&l| deg8.constant_pad_nd([l, 0]).narrow(2, 0, n))
            .collect();
        let stack = Tensor::cat(&shifted, 1); // [B,N_LAGS,N]
        let mix_up = frames_to_samples_interp(&mix, HOP).transpose(1, 2); // [B,N_LAGS,N]
        let comb = (stack * mix_up).sum_dim_intlist(1, true, Kind::Float); // [B,1,N]
        let gain_up = frames_to_samples_interp(&gain, HOP).transpose(1, 2); // [B,1,N]
        let periodic = fir.transpose(1, 2) + gain_up * comb;
        match &self.noise_head {
            None => periodic,
            Some(head) => {
                let noise = Self::aperiodic(deg8, &periodic, g, head, self.noise_mod);
                periodic + noise
            }
        }
    }

    /// The aperiodic excitation: white noise, shaped by [`TAPS`]
    /// per-frame coefficients and scaled by a per-frame gain *relative
    /// to the local signal level*, so the head learns a ratio rather
    /// than an absolute amplitude.
    ///
    /// This is the only narrowband path that can put energy where the
    /// input has none. A codec that codes voiced bands as exactly
    /// periodic leaves the troughs between harmonics empty; filtering
    /// cannot fill them, because a filtered periodic signal is still
    /// periodic. Measured on the project's own eval clips, D-STAR comes
    /// out of the decoder several dB more periodic than the clean speech
    /// it was made from, and that excess is what listeners call
    /// "robotic".
    ///
    /// With `noise_mod`, two things give the noise time structure inside
    /// a frame. It is multiplied by the **envelope of the periodic
    /// path** (a [`MOD_WINDOW`] causal RMS, normalised to the frame's
    /// level), to a per-frame depth the head chooses: breath noise in
    /// real speech rides the glottal cycle, and unmodulated noise on a
    /// comb reads as hiss over buzz rather than as a voice. And it
    /// carries a gain per [`FINE`] sub-frame — one per millisecond — so
    /// it can form a plosive burst, which is shorter than the feature
    /// hop; the slow-varying frame gain alone cannot. The envelope is
    /// detached, so the noise path cannot steer the filter through it.
    fn aperiodic(
        deg8: &Tensor,
        periodic: &Tensor,
        g: &Tensor,
        head: &nn::Linear,
        noise_mod: bool,
    ) -> Tensor {
        let n = deg8.size3().map_or(0, |s| s.2);
        let out = head.forward(g); // [B,T,TAPS+1(+FINE+1)]
        // Gain in [0, 1) of the local RMS; starts near zero so an
        // untrained net is the old filter-only model.
        let gain = (out.narrow(2, 0, 1) - 2.0).sigmoid(); // [B,T,1]
        let taps = out.narrow(2, 1, TAPS) * 0.1; // [B,T,TAPS]

        // Local RMS at frame rate: the same 20 ms window and 10 ms hop
        // the frame conv uses, so level tracks the features.
        let energy =
            deg8.pow_tensor_scalar(2.0)
                .avg_pool1d([FRAME_KERNEL], [HOP], [HOP / 2], false, true);
        let frames = g.size3().map_or(0, |s| s.1);
        let have = energy.size3().map_or(0, |s| s.2);
        let energy = if have >= frames {
            energy.narrow(2, 0, frames)
        } else {
            energy.constant_pad_nd([0, frames - have])
        };
        let rms = (energy + 1e-10).sqrt().transpose(1, 2); // [B,T,1]

        let noise = Tensor::randn_like(deg8);
        let taps_up = frames_to_samples_interp(&taps, HOP); // [B,N,TAPS]
        let unf = noise
            .constant_pad_nd([TAPS - 1, 0])
            .unfold(2, TAPS, 1)
            .squeeze_dim(1); // [B,N,TAPS]
        let shaped = (unf * taps_up).sum_dim_intlist(-1, true, Kind::Float); // [B,N,1]
        let rms_up = frames_to_samples_interp(&rms, HOP); // [B,N,1]
        let level = frames_to_samples_interp(&gain, HOP) * &rms_up; // [B,N,1]
        let shaped = if noise_mod {
            let (b, t, _) = g.size3().unwrap_or((0, 0, 0));
            // Fine gain per millisecond, (0, 2), 1 at init; held per sub-frame.
            let fine = out.narrow(2, TAPS + 1, FINE).sigmoid() * 2.0; // [B,T,FINE]
            let fine = frames_to_samples(&fine.reshape([b, t * FINE, 1]), FINE_HOP); // [B,N,1]
            // Modulation depth per frame, (0, 1), ½ at init.
            let depth = frames_to_samples_interp(&out.narrow(2, TAPS + 1 + FINE, 1).sigmoid(), HOP);
            // Envelope of the periodic path, normalised to the frame level.
            let env = periodic
                .detach()
                .pow_tensor_scalar(2.0)
                .constant_pad_nd([MOD_WINDOW - 1, 0])
                .avg_pool1d([MOD_WINDOW], [1], [0], false, true)
                .clamp_min(1e-12)
                .sqrt()
                .transpose(1, 2); // [B,N,1]
            let env = env / (&rms_up + 1e-6);
            let modulation = 1.0 + depth * (env - 1.0);
            shaped * modulation * fine
        } else {
            shaped
        };
        (shaped * level).transpose(1, 2).narrow(2, 0, n)
    }

    /// Fixed 8 → 16 kHz half-band interpolation: `out[2n] = in[n]`.
    pub fn upsample(&self, x8: &Tensor) -> Tensor {
        x8.conv_transpose1d(
            &self.up_kernel,
            None::<Tensor>,
            2,
            HALFBAND_TAPS / 2,
            1,
            1,
            1,
        )
    }

    /// Fixed 16 → 8 kHz anti-aliased decimation, aligned so that
    /// `decimate(upsample(x)) ≈ x`. Used for the narrowband loss path.
    pub fn decimate(&self, x16: &Tensor) -> Tensor {
        x16.conv1d(&self.dec_kernel, None::<Tensor>, 2, HALFBAND_TAPS / 2, 1, 1)
    }

    /// Full forward: `deg8 [B, 1, N]` (+ `mode [B]`) → `out16 [B, 1, 2N]`.
    pub fn forward(&self, deg8: &Tensor, mode: &Tensor) -> Tensor {
        self.forward_with(deg8, mode, None)
    }

    /// [`forward`](Self::forward) with the per-frame erasure mask; see
    /// [`features_with`](Self::features_with).
    pub fn forward_with(&self, deg8: &Tensor, mode: &Tensor, erasure: Option<&Tensor>) -> Tensor {
        let g = self.features_with(deg8, mode, erasure);
        let y8 = self.narrowband(deg8, &g);
        let up16 = self.upsample(&y8);
        match &self.bwe {
            Some(bwe) => &up16 + bwe.forward(&up16, &g),
            None => up16,
        }
    }
}

/// Trainable parameter count of a var store.
#[must_use]
pub fn param_count(vs: &nn::VarStore) -> usize {
    vs.trainable_variables()
        .iter()
        .map(tch::Tensor::numel)
        .sum()
}

/// Build a net and check its parameter budget.
/// [`NetOpts`] for a run config's `[model]` section.
#[must_use]
pub fn net_opts(cfg: &unamblify::ModelCfg, embed_modes: Option<usize>) -> NetOpts {
    NetOpts {
        profile: cfg.profile,
        width: cfg.width,
        lookahead: cfg.lookahead,
        embed_modes,
        noise_head: cfg.noise_head,
        noise_mod: cfg.noise_mod,
        erasure_in: cfg.erasure_in,
    }
}

pub fn build(vs: &nn::VarStore, opts: &NetOpts) -> anyhow::Result<(Net, usize)> {
    let (profile, width, lookahead, embed_modes) =
        (opts.profile, opts.width, opts.lookahead, opts.embed_modes);
    anyhow::ensure!(
        width.is_finite() && width > 0.0,
        "[model] width must be a positive number, got {width}"
    );
    let net = Net::build(&vs.root(), opts);
    let n = param_count(vs);
    anyhow::ensure!(
        n <= profile.param_budget(),
        "{profile} ll{lookahead}{}{}: {n} parameters exceed the budget of {}",
        if (width - 1.0).abs() < f32::EPSILON {
            String::new()
        } else {
            format!(" at width {width}")
        },
        embed_modes.map_or(String::new(), |m| format!(" with {m}-mode embedding")),
        profile.param_budget()
    );
    Ok((net, n))
}

/// A `[B]` int64 tensor of zeros: the mode of every example in a
/// single-mode run, and what a blind model is handed.
pub fn mode_zeros(batch: i64, device: Device) -> Tensor {
    Tensor::zeros([batch], (Kind::Int64, device))
}

/// Unused-warning guard for the device type in signatures.
#[allow(dead_code)]
const fn _device(_: Device) {}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    /// **The reason the noise head exists.** A filter cannot put energy
    /// between the harmonics of a perfectly periodic signal: whatever it
    /// does, the output stays periodic. The noise path can. Feed both a
    /// synthetic buzz — a pulse train, which is what a vocoder's voiced
    /// excitation amounts to — and measure the energy in the troughs
    /// halfway between harmonics.
    #[test]
    // A hand-rolled DFT for the assertion; the casts are all small and
    // exact, and keeping them explicit reads better than the wrappers.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::manual_range_contains
    )]
    fn only_the_noise_head_can_fill_the_troughs_between_harmonics() {
        const PERIOD: usize = 64; // 125 Hz at 8 kHz
        let n = 8_000; // whole feature hops (and whole pitch periods)
        let buzz: Vec<f32> = (0..n)
            .map(|i| if i % PERIOD == 0 { 1.0 } else { 0.0 })
            .collect();
        let x = Tensor::from_slice(&buzz).view([1, 1, n as i64]);

        // Trough energy relative to harmonic energy, in dB.
        let trough_db = |y: &Tensor| -> f64 {
            let v: Vec<f32> = Vec::<f32>::try_from(y.reshape([-1])).unwrap();
            let win = 1024usize;
            let spec: Vec<f64> = (0..win / 2)
                .map(|k| {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for (i, s) in v.iter().take(win).enumerate() {
                        #[allow(clippy::cast_precision_loss)]
                        let ph =
                            -2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / (win as f64);
                        let w = 0.5
                            - 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / (win as f64)).cos();
                        re += f64::from(*s) * w * ph.cos();
                        im += f64::from(*s) * w * ph.sin();
                    }
                    re * re + im * im
                })
                .collect();
            // Harmonics of 125 Hz land every 16 bins at 8 kHz / 1024.
            let (mut peak, mut trough) = (0.0f64, 0.0f64);
            let (mut np, mut nt) = (0.0f64, 0.0f64);
            for (k, p) in spec.iter().enumerate() {
                if k < 16 || k > 400 {
                    continue;
                }
                if k % 16 == 0 {
                    peak += p;
                    np += 1.0;
                } else if k % 16 == 8 {
                    trough += p;
                    nt += 1.0;
                }
            }
            10.0 * ((peak / np.max(1.0) + 1e-30) / (trough / nt.max(1.0) + 1e-30)).log10()
        };

        let before = trough_db(&x);
        assert!(
            before > 40.0,
            "a pulse train should be near-perfectly periodic, got {before}"
        );

        // Filter-only: whatever the heads do, the troughs stay empty.
        let vs = nn::VarStore::new(Device::Cpu);
        let plain = Net::build(
            &vs.root(),
            &NetOpts {
                profile: Profile::SuperLite,
                ..NetOpts::default()
            },
        );
        assert!(!plain.has_noise_head());
        let g = plain.features(&x, &mode_zeros(1, Device::Cpu));
        let filtered = plain.narrowband(&x, &g);
        let after_filter = trough_db(&filtered);
        assert!(
            after_filter > 30.0,
            "filtering left the signal aperiodic ({after_filter} dB), which should be impossible"
        );

        // With the noise head, and its gain opened up, the troughs fill.
        let vs2 = nn::VarStore::new(Device::Cpu);
        let noisy = Net::build(
            &vs2.root(),
            &NetOpts {
                profile: Profile::SuperLite,
                noise_head: true,
                ..NetOpts::default()
            },
        );
        assert!(noisy.has_noise_head());
        tch::no_grad(|| {
            for (name, mut v) in vs2.variables() {
                if name == "noise_head.bias" {
                    let _ = v.fill_(0.5);
                } else if name == "noise_head.weight" {
                    let _ = v.zero_();
                }
            }
        });
        let g2 = noisy.features(&x, &mode_zeros(1, Device::Cpu));
        let after_noise = trough_db(&noisy.narrowband(&x, &g2));
        assert!(
            after_noise < after_filter - 10.0,
            "the noise head did not fill the troughs: {after_filter:.1} dB -> {after_noise:.1} dB"
        );
    }

    /// An untrained noise head is nearly silent, so adding the path does
    /// not change what an existing model sounds like until it learns to
    /// use it.
    #[test]
    fn the_noise_head_starts_quiet() {
        let x = Tensor::rand([1, 1, 1600], (Kind::Float, Device::Cpu)) - 0.5;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::build(
            &vs.root(),
            &NetOpts {
                profile: Profile::SuperLite,
                noise_head: true,
                ..NetOpts::default()
            },
        );
        let g = net.features(&x, &mode_zeros(1, Device::Cpu));
        let noise = Net::aperiodic(&x, &x, &g, net.noise_head.as_ref().unwrap(), false);
        let rms = |t: &Tensor| {
            f64::try_from(t.pow_tensor_scalar(2.0).mean(Kind::Float))
                .unwrap()
                .sqrt()
        };
        assert!(
            rms(&noise) < 0.1 * rms(&x),
            "untrained noise path is loud: {} vs signal {}",
            rms(&noise),
            rms(&x)
        );
    }

    /// With `noise_mod` the noise follows the periodic path's envelope:
    /// drive the depth to one on a pulse train and the noise sits in the
    /// 2 ms after each pulse, with the rest of the cycle quiet. Held
    /// parameters would not care where in the cycle they were; this is
    /// the difference between breath and hiss.
    #[test]
    fn modulated_noise_rides_the_pulses() {
        let n = 1600i64;
        let v: Vec<f32> = (0..n)
            .map(|i| if i % 80 == 0 { 0.8 } else { 0.0 })
            .collect();
        let x = Tensor::from_slice(&v).view([1, 1, n]);
        let vs = nn::VarStore::new(Device::Cpu);
        let mut net = Net::build(
            &vs.root(),
            &NetOpts {
                profile: Profile::SuperLite,
                noise_head: true,
                noise_mod: true,
                ..NetOpts::default()
            },
        );
        let outputs = usize::try_from(TAPS + 1 + FINE + 1).unwrap();
        {
            let head = net.noise_head.as_mut().unwrap();
            assert_eq!(
                head.ws.size()[0],
                TAPS + 1 + FINE + 1,
                "eleven more outputs"
            );
            tch::no_grad(|| {
                let _ = head.ws.zero_();
                let mut b = vec![0f32; outputs];
                b[0] = 6.0; // gain: sigmoid(4) of the local RMS
                b[usize::try_from(TAPS).unwrap()] = 10.0; // last shaping tap = 1.0
                b[usize::try_from(TAPS + 1 + FINE).unwrap()] = 12.0; // depth → 1
                head.bs.as_mut().unwrap().copy_(&Tensor::from_slice(&b));
            });
        }
        let g = net.features(&x, &mode_zeros(1, Device::Cpu));
        let noise = Net::aperiodic(&x, &x, &g, net.noise_head.as_ref().unwrap(), true);
        let e: Vec<f32> = Vec::try_from(noise.pow_tensor_scalar(2.0).view([-1])).unwrap();
        let (mut on, mut off) = (0.0f32, 0.0f32);
        for k in 2..20usize {
            let s = k * 80;
            on += e[s..s + 16].iter().sum::<f32>();
            off += e[s + 16..s + 80].iter().sum::<f32>();
        }
        assert!(on > 0.0, "the noise path is on");
        assert!(
            off < 0.05 * on,
            "noise between pulses {off} should be far below noise on them {on}"
        );
        // Without the depth, the same head spreads noise over the cycle.
        tch::no_grad(|| {
            let _ = net
                .noise_head
                .as_mut()
                .unwrap()
                .bs
                .as_mut()
                .unwrap()
                .narrow(0, TAPS + 1 + FINE, 1)
                .fill_(-12.0);
        });
        let flat = Net::aperiodic(&x, &x, &g, net.noise_head.as_ref().unwrap(), true);
        let e: Vec<f32> = Vec::try_from(flat.pow_tensor_scalar(2.0).view([-1])).unwrap();
        let (mut on, mut off) = (0.0f32, 0.0f32);
        for k in 2..20usize {
            let s = k * 80;
            on += e[s..s + 16].iter().sum::<f32>();
            off += e[s + 16..s + 80].iter().sum::<f32>();
        }
        assert!(
            off > on,
            "at depth 0 the longer part of the cycle holds more noise"
        );
    }

    /// Interpolated frame parameters reach each frame's value at the end
    /// of its hop and never look past it: a step from 0 to 1 at frame 3
    /// ramps across frame 3's samples only.
    #[test]
    fn frame_parameters_interpolate_causally() {
        let v: Vec<f32> = (0..6).map(|t| if t >= 3 { 1.0 } else { 0.0 }).collect();
        let x = Tensor::from_slice(&v).view([1, 6, 1]);
        let y: Vec<f32> = Vec::try_from(frames_to_samples_interp(&x, 4).view([-1])).unwrap();
        assert_eq!(&y[..12], &[0.0; 12], "nothing before frame 3");
        assert_eq!(&y[12..16], &[0.25, 0.5, 0.75, 1.0], "ramps across frame 3");
        assert_eq!(&y[16..], &[1.0; 8], "and holds");
    }

    fn input(b: i64, n: i64, seed: u64) -> Tensor {
        let mut r = crate::rng::Rng::new(seed);
        let v: Vec<f32> = (0..b * n).map(|_| r.next_f32() - 0.5).collect();
        Tensor::from_slice(&v).view([b, 1, n])
    }

    #[test]
    fn shapes_and_budgets_for_every_profile_and_lookahead() {
        for profile in Profile::ALL {
            for lookahead in [5u32, 20] {
                for embed in [None, Some(2)] {
                    let vs = nn::VarStore::new(Device::Cpu);
                    let (net, n) = build(
                        &vs,
                        &NetOpts {
                            profile,
                            lookahead,
                            embed_modes: embed,
                            ..NetOpts::default()
                        },
                    )
                    .unwrap();
                    assert!(
                        n <= profile.param_budget(),
                        "{profile} ll{lookahead} {embed:?}: {n}"
                    );
                    assert!(n > 1000, "{profile}: {n}");
                    let x = input(2, 1600, 1);
                    let m = mode_zeros(2, Device::Cpu);
                    let g = net.features(&x, &m);
                    assert_eq!(g.size(), [2, 20, widths(profile).f]);
                    let y = net.forward(&x, &m);
                    assert_eq!(y.size(), [2, 1, 3200]);
                    assert_eq!(y.isfinite().all().int64_value(&[]), 1);
                    assert_eq!(net.decimate(&y).size(), [2, 1, 1600]);
                    assert_eq!(net.embed_modes(), embed);
                    eprintln!("{profile} ll{lookahead} {embed:?}: {n} params");
                }
            }
        }
    }

    /// `[model] erasure_in`: the mask is one more channel into the context
    /// conv, conditions the output, means "nothing was lost" when it is
    /// not given — and is seen coming, a lookahead early.
    #[test]
    fn the_erasure_mask_conditions_the_output_and_is_seen_coming() {
        let n = 8_000; // 1 s: 100 feature frames
        let x = input(2, n, 4);
        let mode = mode_zeros(2, Device::Cpu);
        let frames = n / HOP;
        let build = |erasure_in: bool| {
            let vs = nn::VarStore::new(Device::Cpu);
            let net = Net::build(
                &vs.root(),
                &NetOpts {
                    profile: Profile::Lite,
                    erasure_in,
                    ..NetOpts::default()
                },
            );
            (vs, net)
        };
        let (vs, on) = build(true);
        let (vs_off, off) = build(false);
        assert!(on.takes_erasure() && !off.takes_erasure());
        // Exactly one more input channel into the context conv.
        let width = |v: &nn::VarStore| v.variables()["ctx.weight"].size()[1];
        assert_eq!(width(&vs), width(&vs_off) + 1);
        assert_eq!(
            vs.variables()["gru.weight_ih_l0"].size(),
            vs_off.variables()["gru.weight_ih_l0"].size(),
            "the GRU is untouched"
        );

        let none = on.forward(&x, &mode);
        let zeros = Tensor::zeros([2, frames], (Kind::Float, Device::Cpu));
        let same = on.forward_with(&x, &mode, Some(&zeros));
        assert!(
            (&none - &same).abs().max().double_value(&[]) < 1e-9,
            "no mask means nothing was lost"
        );

        // Lose feature frames 60..65 of the first row only. ll5 is 10
        // feature frames of lookahead, so the model first sees the loss at
        // frame 50: 8 kHz sample 4000, 16 kHz sample 8000.
        let lost = Tensor::zeros([2, frames], (Kind::Float, Device::Cpu));
        let _ = lost.get(0).narrow(0, 60, 5).fill_(1.0);
        let y = on.forward_with(&x, &mode, Some(&lost));
        let moved = |a: i64, len: i64| {
            (y.get(0) - none.get(0))
                .narrow(1, a, len)
                .abs()
                .max()
                .double_value(&[])
        };
        assert!(
            moved(0, 7_000) < 1e-9,
            "nothing before the lookahead window moves"
        );
        assert!(
            moved(8_200, 1_000) > 1e-7,
            "frames 51..57 already differ: the gap at 60 is seen coming"
        );
        assert!(
            (y.get(1) - none.get(1)).abs().max().double_value(&[]) < 1e-9,
            "only in the row that lost one"
        );

        // A net built without the flag ignores a mask it is handed.
        let a = off.forward(&x, &mode);
        let b = off.forward_with(&x, &mode, Some(&lost));
        assert!((a - b).abs().max().double_value(&[]) < 1e-9);
    }

    #[test]
    fn the_mode_embedding_conditions_the_output_and_is_absent_when_off() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::Lite, 5, Some(3));
        assert!(vs.variables().contains_key("mode_embed.weight"));
        assert_eq!(
            vs.variables()["mode_embed.weight"].size(),
            [3, i64::try_from(MODE_EMBED_DIM).unwrap()]
        );
        let x = input(2, 1600, 4);
        let y0 = net.forward(&x, &mode_zeros(2, Device::Cpu));
        let y2 = net.forward(&x, &Tensor::from_slice(&[2i64, 2]));
        let diff = (&y0 - &y2).abs().max().double_value(&[]);
        assert!(diff > 1e-6, "another mode, another output: {diff}");
        // The batch may mix modes: each row is what its own index gives.
        let mixed = net.forward(&x, &Tensor::from_slice(&[0i64, 2]));
        assert!((mixed.get(0) - y0.get(0)).abs().max().double_value(&[]) < 1e-6);
        assert!((mixed.get(1) - y2.get(1)).abs().max().double_value(&[]) < 1e-6);
        // Off: no table, the mode tensor is ignored.
        let vs = nn::VarStore::new(Device::Cpu);
        let blind = Net::new(&vs.root(), Profile::Lite, 5, None);
        assert!(!vs.variables().contains_key("mode_embed.weight"));
        assert_eq!(blind.embed_modes(), None);
        let a = blind.forward(&x, &mode_zeros(2, Device::Cpu));
        let b = blind.forward(&x, &Tensor::from_slice(&[1i64, 0]));
        assert!((a - b).abs().max().double_value(&[]) < 1e-9);
    }

    #[test]
    fn super_lite_has_no_bwe_head_and_full_does() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, None);
        assert!(net.bwe.is_none());
        assert!(!vs.variables().keys().any(|k| k.starts_with("bwe")));
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::Full, 5, None);
        assert!(net.bwe.is_some());
        assert!(vs.variables().contains_key("bwe.conv_out.weight"));
        assert!(vs.variables().contains_key("gru.weight_ih_l1"));
    }

    #[test]
    fn resampling_pair_is_aligned_and_near_identity() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, None);
        // A 500 Hz tone at 8 kHz — well inside the half-band passband.
        let n = 1600usize;
        #[allow(clippy::cast_precision_loss)]
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / 8000.0).sin())
            .collect();
        let x = Tensor::from_slice(&tone).view([1, 1, 1600]);
        let up = net.upsample(&x);
        assert_eq!(up.size(), [1, 1, 3200]);
        // Even samples reproduce the input exactly.
        let even = up.slice(2, 0, 3200, 2);
        assert!((&even - &x).abs().max().double_value(&[]) < 1e-6);
        let back = net.decimate(&up);
        assert_eq!(back.size(), [1, 1, 1600]);
        let mid = back.narrow(2, 100, 1400);
        let want = x.narrow(2, 100, 1400);
        let err = (&mid - &want).abs().max().double_value(&[]);
        assert!(err < 0.02, "{err}");
        let h = halfband();
        assert!((h[15] - 0.5).abs() < 1e-7);
        assert!(h[13].abs() < 1e-7 && h[17].abs() < 1e-7);
    }

    #[test]
    fn ll5_and_ll20_are_100_and_400_ms_of_lookahead() {
        assert_eq!(
            usize::try_from(Net::FEATURE_FRAMES_PER_AMBE_FRAME * HOP).unwrap(),
            unamblify::FRAME_SAMPLES
        );
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::Lite, 5, None);
        assert_eq!(net.lookahead(), 10, "5 AMBE frames = 10 feature frames");
        assert_eq!(net.lookahead_samples(), 800);
        assert_eq!(net.lookahead_ms(), 100);
        let net = Net::new(&vs.root(), Profile::Lite, 20, None);
        assert_eq!(net.lookahead_ms(), 400);
        let cfg = unamblify::ModelCfg {
            lookahead: 5,
            ..unamblify::ModelCfg::default()
        };
        assert_eq!(i64::from(cfg.lookahead_ms()), net.lookahead_ms() / 4);
    }

    #[test]
    fn output_depends_on_exactly_l_frames_of_future() {
        for (profile, lookahead) in [(Profile::Lite, 5i64), (Profile::SuperLite, 20)] {
            let vs = nn::VarStore::new(Device::Cpu);
            let net = Net::new(
                &vs.root(),
                profile,
                u32::try_from(lookahead).unwrap(),
                Some(2),
            );
            let a = input(1, 8000, 3);
            let m = 70i64; // inputs differ from sample 80·m onward
            let b = a.copy();
            let _ = b.narrow(2, HOP * m, 8000 - HOP * m).fill_(0.25);
            let one = Tensor::from_slice(&[1i64]);
            let ya = net.forward(&a, &one);
            let yb = net.forward(&b, &one);
            // Frames < m − 2L are untouched, less the half-band kernel's
            // 15-sample (16 kHz) reach across the boundary. 2L feature
            // frames = L AMBE frames = 20·L ms.
            assert_eq!(net.lookahead(), 2 * lookahead);
            let same_until = 2 * HOP * (m - 2 * lookahead) - HALFBAND_TAPS / 2;
            let head = (ya.narrow(2, 0, same_until) - yb.narrow(2, 0, same_until))
                .abs()
                .max()
                .double_value(&[]);
            assert!(head < 1e-6, "{profile}: leaked future, diff {head}");
            let after = (ya.narrow(2, same_until, 2 * HOP) - yb.narrow(2, same_until, 2 * HOP))
                .abs()
                .max()
                .double_value(&[]);
            assert!(after > 1e-6, "{profile}: lookahead not used");
        }
    }
}
