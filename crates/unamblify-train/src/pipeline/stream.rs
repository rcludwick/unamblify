// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The pipeline one frame at a time, with the batch path's numbers.
//!
//! Every layer that looks ahead is a convolution over frames, and a
//! convolution fed frame by frame needs only its last `k` inputs and a
//! delay of `right` frames: output frame `t` exists once input `t + right`
//! has arrived, and the zeros the batch path pads in front are what an
//! empty buffer holds. The GRU carries its state. So the stream keeps one
//! small buffer per layer, and its output for frame `t` is the batch
//! output for frame `t`, to floating-point noise, once the lookahead has
//! arrived. At the end of a stream, [`Stream::flush`] feeds each layer
//! the `right` zero frames the batch path pads it with, in layer order,
//! so the last frames come out identical too and the stream's length is
//! the batch path's `HOP × (T − 1)`.
//!
//! **Latency**, in feature frames of [`HOP`] samples (10.67 ms):
//!
//! | stage | frames | why |
//! |---|---|---|
//! | centred STFT | 2 | frame `t` is centred on sample `t · HOP` and reaches `HOP / 2` past it (not part of this stream; see below) |
//! | restorer | 8 | conv 2 + two blocks × 3 |
//! | synthesiser | 27 | conv 3 + eight blocks × 3 |
//! | iSTFT | 2 | a sample is final once the last window covering it is added; with the centre trim that is two frames after its own |
//!
//! 37 frames from a mel frame in to its audio out (394.7 ms), 39 from
//! audio in to audio out (416 ms) once the feature extraction is
//! streamed too, plus the resampler's 7-sample (0.9 ms) kernel. The
//! feature extraction is still the batch path — [`Stream`] takes feature
//! frames — which is the one piece left before a receiver can run this.

use std::collections::VecDeque;

use tch::{Kind, Tensor};

use super::restorer::Block;
use super::{HOP, N_FFT, Pipeline};

/// A convolution fed one frame at a time: the last `k` inputs and the
/// delay its right padding implies.
#[derive(Debug)]
struct FrameConv {
    buf: Tensor,
    right: i64,
    seen: i64,
}

impl FrameConv {
    fn new(c_in: i64, k: i64, right: i64, like: &Tensor) -> Self {
        Self {
            buf: Tensor::zeros([1, c_in, k], (Kind::Float, like.device())),
            right,
            seen: 0,
        }
    }

    fn c_in(&self) -> i64 {
        self.buf.size()[1]
    }

    /// Push `x [C]`; the window over which output frame `seen − 1 − right`
    /// is computed, if that frame exists yet.
    fn push(&mut self, x: &Tensor) -> Option<Tensor> {
        let k = self.buf.size()[2];
        let c = self.c_in();
        self.buf = Tensor::cat(&[self.buf.narrow(2, 1, k - 1), x.view([1, c, 1])], 2);
        self.seen += 1;
        (self.seen - 1 - self.right >= 0).then(|| self.buf.shallow_clone())
    }

    /// A zero frame of this layer's input: the batch path's right padding.
    fn zero(&self) -> Tensor {
        Tensor::zeros([self.c_in()], (Kind::Float, self.buf.device()))
    }
}

/// A block fed one frame at a time.
#[derive(Debug)]
struct FrameBlock<'a> {
    block: &'a Block,
    conv: FrameConv,
}

impl<'a> FrameBlock<'a> {
    fn new(block: &'a Block, like: &Tensor) -> Self {
        Self {
            block,
            conv: FrameConv::new(block.dim(), block.pad.kernel, block.pad.right, like),
        }
    }

    /// Push `x [C]`; the block's output frame `[C]`, once it exists.
    fn push(&mut self, x: &Tensor) -> Option<Tensor> {
        let b = self.block;
        let win = self.conv.push(x)?;
        let c = b.dim();
        let y = win
            .conv1d(&b.dw_w, Some(&b.dw_b), 1, 0, 1, c)
            .view([1, 1, c]); // [B=1, T=1, C]
        let y = b.pointwise(&y).view([c]);
        // The residual is the input frame this output belongs to.
        let x_t = win.narrow(2, b.pad.left, 1).view([c]);
        Some(x_t + y)
    }
}

/// The restorer, one frame at a time.
#[derive(Debug)]
struct RestorerStream<'a> {
    p: &'a Pipeline,
    embed: Tensor,
    conv_in: FrameConv,
    blocks: Vec<FrameBlock<'a>>,
    h: Tensor,
    mels: VecDeque<Tensor>,
}

impl<'a> RestorerStream<'a> {
    fn new(p: &'a Pipeline, mode: i64, like: &Tensor) -> Self {
        let r = &p.restorer;
        let c_in = r.conv_in_w.size()[1];
        Self {
            p,
            embed: r.emb.narrow(0, mode, 1).view([-1]),
            conv_in: FrameConv::new(c_in, r.conv_in_pad.kernel, r.conv_in_pad.right, like),
            blocks: r.blocks.iter().map(|b| FrameBlock::new(b, like)).collect(),
            h: Tensor::zeros([1, r.dim()], (Kind::Float, like.device())),
            mels: VecDeque::new(),
        }
    }

    /// The input conv on a window it has just completed.
    fn conv_in(&self, win: &Tensor) -> Tensor {
        let r = &self.p.restorer;
        win.conv1d(&r.conv_in_w, Some(&r.conv_in_b), 1, 0, 1, 1)
            .view([-1])
    }

    /// Frame `x` through blocks `from..`, the GRU and the head: the
    /// predicted mel frame, once the blocks' lookahead allows one.
    fn run_from(&mut self, mut x: Tensor, from: usize) -> Option<Tensor> {
        for b in &mut self.blocks[from..] {
            x = b.push(&x)?;
        }
        let r = &self.p.restorer;
        let x2 = x.view([1, -1]);
        self.h = x2.gru_cell(
            &self.h,
            &r.gru_w_ih,
            &r.gru_w_hh,
            Some(&r.gru_b_ih),
            Some(&r.gru_b_hh),
        );
        let corr = r.head(&(&self.h + &x2).view([1, 1, -1])).view([-1]);
        let mel_t = self.mels.pop_front()?;
        Some(mel_t + corr)
    }

    /// Push one frame of `low [171]` and `mel [100]`; the predicted clean
    /// mel frame `[100]`, once the lookahead has arrived.
    fn push(&mut self, low: &Tensor, mel: &Tensor) -> Option<Tensor> {
        self.mels.push_back(mel.shallow_clone());
        let input = Tensor::cat(&[low, mel, &self.embed], 0);
        let win = self.conv_in.push(&input)?;
        let x = self.conv_in(&win);
        self.run_from(x, 0)
    }

    /// The batch path's right zero padding, layer by layer: every frame
    /// still owed.
    fn flush(&mut self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for _ in 0..self.conv_in.right {
            let zero = self.conv_in.zero();
            let x = match self.conv_in.push(&zero) {
                Some(win) => self.conv_in(&win),
                None => continue,
            };
            out.extend(self.run_from(x, 0));
        }
        for i in 0..self.blocks.len() {
            for _ in 0..self.blocks[i].conv.right {
                let zero = self.blocks[i].conv.zero();
                if let Some(y) = self.blocks[i].push(&zero) {
                    out.extend(self.run_from(y, i + 1));
                }
            }
        }
        out
    }
}

/// The synthesiser, one frame at a time, down to audio.
#[derive(Debug)]
struct SynthStream<'a> {
    p: &'a Pipeline,
    embed: FrameConv,
    blocks: Vec<FrameBlock<'a>>,
    /// Overlap-add of windowed frames, and of squared windows, for the
    /// [`N_FFT`] samples from the newest frame's start.
    ola: Tensor,
    env: Tensor,
    frames_added: i64,
}

impl<'a> SynthStream<'a> {
    fn new(p: &'a Pipeline, like: &Tensor) -> Self {
        let s = &p.synth;
        let opts = (Kind::Float, like.device());
        Self {
            p,
            embed: FrameConv::new(
                s.embed_w.size()[1],
                s.embed_pad.kernel,
                s.embed_pad.right,
                like,
            ),
            blocks: s.blocks.iter().map(|b| FrameBlock::new(b, like)).collect(),
            ola: Tensor::zeros([N_FFT], opts),
            env: Tensor::zeros([N_FFT], opts),
            frames_added: 0,
        }
    }

    /// The input conv and its `LayerNorm` on a completed window.
    fn embed(&self, win: &Tensor) -> Tensor {
        let s = &self.p.synth;
        let c = s.dim();
        win.conv1d(&s.embed_w, Some(&s.embed_b), 1, 0, 1, 1)
            .view([1, 1, c])
            .layer_norm([c], Some(&s.norm_w), Some(&s.norm_b), s.eps, true)
            .view([c])
    }

    /// Frame `x` through blocks `from..`, the head and the overlap-add:
    /// the next [`HOP`] samples, if a block became final.
    fn run_from(&mut self, mut x: Tensor, from: usize) -> Option<Tensor> {
        for b in &mut self.blocks[from..] {
            x = b.push(&x)?;
        }
        let s = &self.p.synth;
        let c = s.dim();
        let x = x
            .view([1, 1, c])
            .layer_norm([c], Some(&s.final_w), Some(&s.final_b), s.eps, true);
        let spec = s.spectrum(&x).view([-1]); // [bins] complex
        self.add_frame(&spec);
        self.emit()
    }

    /// Push a mel frame `[100]`; the next [`HOP`] samples of audio, if a
    /// block became final.
    fn push(&mut self, mel: &Tensor) -> Option<Tensor> {
        let win = self.embed.push(mel)?;
        let x = self.embed(&win);
        self.run_from(x, 0)
    }

    /// Slide the overlap-add buffers one hop.
    fn slide(&mut self) {
        let zeros = Tensor::zeros([HOP], (Kind::Float, self.ola.device()));
        self.ola = Tensor::cat(
            &[self.ola.narrow(0, HOP, N_FFT - HOP), zeros.shallow_clone()],
            0,
        );
        self.env = Tensor::cat(&[self.env.narrow(0, HOP, N_FFT - HOP), zeros], 0);
    }

    /// Overlap-add one complex frame at offset [`HOP`] × its index.
    fn add_frame(&mut self, spec: &Tensor) {
        let s = &self.p.synth;
        let frame = spec.fft_irfft(N_FFT, -1, "backward") * &s.window;
        if self.frames_added > 0 {
            self.slide();
        }
        self.ola += frame;
        self.env += &s.window * &s.window;
        self.frames_added += 1;
    }

    /// The block of padded samples `[HOP × (frames_added − 1), + HOP)` is
    /// final; the first two are what `torch.istft(center=True)` trims.
    fn emit(&mut self) -> Option<Tensor> {
        if self.frames_added - 1 < N_FFT / HOP / 2 {
            return None;
        }
        Some((&self.ola.narrow(0, 0, HOP) / self.env.narrow(0, 0, HOP)).to_kind(Kind::Float))
    }

    /// The batch path's right zero padding, layer by layer, then the one
    /// block the overlap-add still owes.
    fn flush(&mut self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for _ in 0..self.embed.right {
            let zero = self.embed.zero();
            let x = match self.embed.push(&zero) {
                Some(win) => self.embed(&win),
                None => continue,
            };
            out.extend(self.run_from(x, 0));
        }
        for i in 0..self.blocks.len() {
            for _ in 0..self.blocks[i].conv.right {
                let zero = self.blocks[i].conv.zero();
                if let Some(y) = self.blocks[i].push(&zero) {
                    out.extend(self.run_from(y, i + 1));
                }
            }
        }
        if self.frames_added >= 2 {
            self.slide();
            let env = self.env.narrow(0, 0, HOP);
            if f64::try_from(env.min()).unwrap_or(0.0) > 1e-11 {
                out.push((&self.ola.narrow(0, 0, HOP) / env).to_kind(Kind::Float));
            }
        }
        out
    }
}

/// The whole pipeline fed one feature frame at a time.
#[derive(Debug)]
pub struct Stream<'a> {
    restorer: RestorerStream<'a>,
    synth: SynthStream<'a>,
}

impl<'a> Stream<'a> {
    /// A stream through `p` for one mode.
    #[must_use]
    pub fn new(p: &'a Pipeline, mode: i64) -> Self {
        let like = &p.restorer.emb;
        Self {
            restorer: RestorerStream::new(p, mode, like),
            synth: SynthStream::new(p, like),
        }
    }

    /// Push one feature frame — `low [171]`, the codec's `mel [100]` — and
    /// get the next [`HOP`] samples of 24 kHz audio once the lookahead has
    /// arrived.
    pub fn push(&mut self, low: &Tensor, mel: &Tensor) -> Option<Tensor> {
        let clean = self.restorer.push(low, mel)?;
        self.synth.push(&clean)
    }

    /// Push a frame and get the restorer's output frame for the frame
    /// the lookahead reached, without synthesising.
    pub fn push_mel(&mut self, low: &Tensor, mel: &Tensor) -> Option<Tensor> {
        self.restorer.push(low, mel)
    }

    /// End of the stream: every block of audio still owed, so that the
    /// whole output equals the batch path's.
    #[must_use]
    pub fn flush(&mut self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for clean in self.restorer.flush() {
            out.extend(self.synth.push(&clean));
        }
        out.extend(self.synth.flush());
        out
    }

    /// Feature frames of delay between a frame going in and its audio
    /// coming out.
    #[must_use]
    pub fn latency_frames(p: &Pipeline) -> i64 {
        p.lookahead_frames() + N_FFT / HOP / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{LOW_BINS, N_MELS, spike_manifest, synthetic_weights};
    use tch::Device;

    /// Frame by frame equals the batch path — the mel, then the audio,
    /// to the last sample after a flush — and the first audio comes out
    /// exactly the lookahead late.
    #[test]
    fn the_stream_matches_the_batch_path() {
        let weights = synthetic_weights(&spike_manifest(), Device::Cpu);
        let pipe = Pipeline::from_weights(&weights).unwrap();
        let frames = 60;
        let low = Tensor::randn([LOW_BINS, frames], (Kind::Float, Device::Cpu));
        let mel = Tensor::randn([N_MELS, frames], (Kind::Float, Device::Cpu));
        let (mel_b, wav_b) = tch::no_grad(|| {
            let batch_mel = pipe
                .restorer
                .forward(&low.unsqueeze(0), &mel.unsqueeze(0), 1)
                .squeeze_dim(0);
            let y = pipe.synth.forward(&batch_mel.unsqueeze(0)).squeeze_dim(0);
            (batch_mel, y)
        });
        let frame = |x: &Tensor, i: i64| x.narrow(1, i, 1).view([-1]);

        let mut stream = Stream::new(&pipe, 1);
        let mut mels = Vec::new();
        tch::no_grad(|| {
            for i in 0..frames {
                mels.extend(stream.push_mel(&frame(&low, i), &frame(&mel, i)));
            }
            mels.extend(stream.restorer.flush());
        });
        let streamed_mel = Tensor::stack(&mels, 1);
        assert_eq!(
            streamed_mel.size(),
            mel_b.size(),
            "the restorer stream owes frames"
        );
        let d = f64::try_from((&streamed_mel - &mel_b).abs().max()).unwrap();
        assert!(d < 1e-4, "restorer stream vs batch: {d}");

        let mut stream = Stream::new(&pipe, 1);
        let mut blocks = Vec::new();
        let mut first_audio = None;
        tch::no_grad(|| {
            for i in 0..frames {
                if let Some(b) = stream.push(&frame(&low, i), &frame(&mel, i)) {
                    first_audio.get_or_insert(i);
                    blocks.push(b);
                }
            }
            blocks.extend(stream.flush());
        });
        let wav_s = Tensor::cat(&blocks, 0);
        assert_eq!(wav_s.size(), wav_b.size(), "the audio stream owes samples");
        let d = f64::try_from((&wav_s - &wav_b).abs().max()).unwrap();
        let scale = f64::try_from(wav_b.abs().max()).unwrap().max(1e-6);
        assert!(
            d < 1e-4 * scale.max(1.0),
            "audio stream vs batch: {d} (peak {scale})"
        );
        assert_eq!(first_audio, Some(Stream::latency_frames(&pipe)));
    }
}
