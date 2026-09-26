// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! The restorer's input features, defined exactly as the spike computed
//! them in `PyTorch`, so a model trained there behaves the same here.
//!
//! 1. **Resample** the 8 kHz codec output to 24 kHz with the kernel
//!    `torchaudio.functional.resample` builds (windowed sinc, Hann,
//!    `lowpass_filter_width` 6, rolloff 0.99), applied as torchaudio does:
//!    zero-pad `width` before and `width + 1` after, stride-1 convolution
//!    with the `[3, 1, K]` kernel, interleave the three phases, cut to
//!    `3 × N`. The kernel is read from the weights file rather than
//!    rebuilt, so the two sides cannot drift.
//! 2. **STFT**: 1024-point, hop 256, periodic Hann, centred with
//!    reflection padding of 512 on both sides, one-sided magnitude —
//!    `torch.stft(center=True, pad_mode="reflect")`. `1 + L / 256`
//!    frames for `L` samples.
//! 3. **Low band**: the natural log of the first [`LOW_BINS`] magnitudes
//!    (0–4 kHz), floored at [`LOG_CLAMP`]; the mel is floored at [`MEL_CLAMP`].
//! 4. **Mel**: the same magnitude through the 100-band HTK filterbank
//!    torchaudio's `MelSpectrogram(power=1)` uses, then the same floored
//!    log — Vocos's `MelSpectrogramFeatures`. The filterbank is read from
//!    the weights file.

use tch::{Device, Kind, Tensor};

use super::weights::Weights;
use super::{HOP, LOG_CLAMP, LOW_BINS, MEL_CLAMP, N_FFT};

/// The feature extractor.
#[derive(Debug)]
pub struct Features {
    kernel: Tensor,
    width: i64,
    orig_freq: i64,
    new_freq: i64,
    window: Tensor,
    mel_window: Tensor,
    fbank: Tensor,
}

impl Features {
    /// From the weights file.
    pub fn from_weights(w: &Weights) -> anyhow::Result<Self> {
        let kernel = w.get("features.resample_kernel")?;
        let r = &w.manifest.resample;
        anyhow::ensure!(
            kernel.size() == [r.new_freq, 1, 2 * r.width + r.orig_freq],
            "resample kernel is {:?}, the manifest says new_freq {} width {}",
            kernel.size(),
            r.new_freq,
            r.width
        );
        let window = w.get("features.window")?;
        anyhow::ensure!(window.size() == [N_FFT], "window is {:?}", window.size());
        let mel_window = w.get("features.mel_window")?;
        anyhow::ensure!(
            mel_window.size() == [N_FFT],
            "mel window is {:?}",
            mel_window.size()
        );
        let fbank = w.get("features.mel_fbank")?;
        anyhow::ensure!(
            fbank.size() == [N_FFT / 2 + 1, w.manifest.n_mels],
            "mel filterbank is {:?}",
            fbank.size()
        );
        Ok(Self {
            kernel,
            width: r.width,
            orig_freq: r.orig_freq,
            new_freq: r.new_freq,
            window,
            mel_window,
            fbank,
        })
    }

    /// The device the tensors live on.
    #[must_use]
    pub fn device(&self) -> Device {
        self.window.device()
    }

    /// Output samples per input sample.
    #[must_use]
    pub const fn ratio(&self) -> i64 {
        self.new_freq / self.orig_freq
    }

    /// `x8 [N]` at the input rate → `[N · ratio]` at the working rate.
    pub fn resample(&self, x8: &Tensor) -> Tensor {
        let n = x8.size1().unwrap_or(0);
        let padded = x8
            .view([1, 1, n])
            .constant_pad_nd([self.width, self.width + self.orig_freq]);
        let y = padded.conv1d(&self.kernel, None::<Tensor>, self.orig_freq, 0, 1, 1); // [1, new, L']
        let y = y.transpose(1, 2).reshape([-1]);
        let target = (self.new_freq * n + self.orig_freq - 1) / self.orig_freq;
        y.narrow(0, 0, target.min(y.size1().unwrap_or(0)))
    }

    /// One-sided STFT magnitude of `x24 [L]` under the low band's window:
    /// `[N_FFT / 2 + 1, 1 + L / HOP]`.
    pub fn magnitude(&self, x24: &Tensor) -> Tensor {
        Self::stft_magnitude(x24, &self.window)
    }

    /// The same under the mel's window: Vocos's checkpoint carries its own
    /// Hann window, one ulp from `torch.hann_window`, and the spike's low
    /// band uses the latter. One ulp in the window moves the stop-band
    /// floor the restorer sees, so each spectrum keeps its own window.
    pub fn mel_magnitude(&self, x24: &Tensor) -> Tensor {
        Self::stft_magnitude(x24, &self.mel_window)
    }

    fn stft_magnitude(x24: &Tensor, window: &Tensor) -> Tensor {
        let l = x24.size1().unwrap_or(0);
        let padded = x24
            .view([1, 1, l])
            .reflection_pad1d([N_FFT / 2, N_FFT / 2])
            .view([1, l + N_FFT]);
        let spec = padded.stft(N_FFT, HOP, N_FFT, Some(window), false, true, true, false); // [1, bins, T] complex
        spec.abs().squeeze_dim(0)
    }

    /// The low-band log spectrum `[LOW_BINS, T]` from a magnitude.
    pub fn low_band(mag: &Tensor) -> Tensor {
        mag.narrow(0, 0, LOW_BINS).clamp_min(LOG_CLAMP).log()
    }

    /// The log-mel `[n_mels, T]` from a magnitude.
    pub fn mel(&self, mag: &Tensor) -> Tensor {
        // torchaudio's order of operations (`[T, 513] @ [513, 100]`), so the
        // rounding at the floor of the stop band is the same as the spike's.
        mag.transpose(0, 1)
            .contiguous()
            .matmul(&self.fbank)
            .transpose(0, 1)
            .clamp_min(MEL_CLAMP)
            .log()
    }

    /// Everything the restorer takes, from `x8 [N]`: `(low, mel)`.
    pub fn compute(&self, x8: &Tensor) -> (Tensor, Tensor) {
        let x24 = self.resample(&x8.to_kind(Kind::Float).to_device(self.device()));
        (
            Self::low_band(&self.magnitude(&x24)),
            self.mel(&self.mel_magnitude(&x24)),
        )
    }
}
