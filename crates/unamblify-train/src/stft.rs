// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Differentiable STFT magnitudes and a mel projection on tch.
//!
//! Frames are centred exactly as `unamblify_audio::spectral::stft` centres
//! them: reflection padding by `n_fft / 2` on both sides, then libtorch's
//! *non-centred* `stft` (tch 0.26's `stft_center` binding passes an
//! `align_to_window` argument libtorch 2.13 rejects when `center = true`,
//! spec §0). Frame `i` is centred on sample `i · hop` and there are
//! `1 + len / hop` frames, so the audio crate's metrics and the trainer's
//! losses look at the same frames.

use tch::{Device, Kind, Tensor};

/// Squared magnitude floor inside the square root (magnitude ≥ 1e-10).
const MAG_EPS_SQ: f64 = 1e-20;

/// One STFT configuration with its window on the right device.
#[derive(Debug)]
pub struct Stft {
    /// FFT size.
    pub n_fft: i64,
    /// Hop, samples.
    pub hop: i64,
    window: Tensor,
}

impl Stft {
    /// Periodic Hann window of `n_fft`, hop `hop`.
    #[must_use]
    pub fn new(n_fft: i64, hop: i64, device: Device) -> Self {
        let window = Tensor::hann_window(n_fft, (Kind::Float, device));
        Self { n_fft, hop, window }
    }

    /// Frequency bins per frame.
    #[must_use]
    pub const fn bins(&self) -> i64 {
        self.n_fft / 2 + 1
    }

    /// Frames for a signal of `len` samples.
    #[must_use]
    pub const fn frames(&self, len: i64) -> i64 {
        1 + len / self.hop
    }

    /// Magnitude spectrogram `[B, bins, frames]` of `x` (`[B, N]` or
    /// `[B, 1, N]`).
    pub fn magnitude(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = if x.dim() == 3 {
            x.squeeze_dim(1)
        } else {
            x.shallow_clone()
        };
        let pad = self.n_fft / 2;
        let xp = x.f_reflection_pad1d([pad, pad])?;
        let spec = xp.f_stft(
            self.n_fft,
            self.hop,
            self.n_fft,
            Some(&self.window),
            false,
            true,
            true,
            false,
        )?;
        // Not `spec.abs()`: libtorch's complex-abs backward is NaN at an
        // exact zero, and silent frames (zero padding, gated garbage) are
        // exactly zero. A floor inside the square root keeps the gradient
        // finite and changes magnitudes by < 1e-10.
        let parts = spec.view_as_real();
        Ok((parts.square().sum_dim_intlist(-1, false, Kind::Float) + MAG_EPS_SQ).sqrt())
    }
}

/// The audio crate's HTK mel filterbank as a `[n_mels, bins]` tensor.
pub fn mel_filterbank(rate: u32, n_fft: i64, n_mels: i64, device: Device) -> Tensor {
    let fb = unamblify_audio::mel_filterbank(
        rate,
        usize::try_from(n_fft).unwrap_or(0),
        usize::try_from(n_mels).unwrap_or(0),
    );
    let flat: Vec<f32> = fb.into_iter().flatten().collect();
    Tensor::from_slice(&flat)
        .view([n_mels, n_fft / 2 + 1])
        .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut r = crate::rng::Rng::new(seed);
        (0..n).map(|_| r.next_f32() * 2.0 - 1.0).collect()
    }

    /// The spec's reference: the same reflection-padded signal convolved
    /// with cosine / sine kernels (a DFT written as a strided conv) —
    /// nothing from libtorch's FFT path.
    fn dft_as_conv(x: &Tensor, n_fft: i64, hop: i64) -> Tensor {
        let pad = n_fft / 2;
        let xp = x.reflection_pad1d([pad, pad]).unsqueeze(1); // [B,1,N+n_fft]
        let win = unamblify_audio::spectral::hann_periodic(usize::try_from(n_fft).unwrap());
        let bins = usize::try_from(n_fft / 2 + 1).unwrap();
        let n = usize::try_from(n_fft).unwrap();
        let mut cos = Vec::with_capacity(bins * n);
        let mut sin = Vec::with_capacity(bins * n);
        for k in 0..bins {
            for (i, w) in win.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let ph = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                #[allow(clippy::cast_possible_truncation)]
                {
                    cos.push(w * ph.cos() as f32);
                    sin.push(w * ph.sin() as f32);
                }
            }
        }
        let bins_i = i64::try_from(bins).unwrap();
        let kc = Tensor::from_slice(&cos).view([bins_i, 1, n_fft]);
        let ks = Tensor::from_slice(&sin).view([bins_i, 1, n_fft]);
        let re = xp.conv1d(&kc, None::<Tensor>, hop, 0, 1, 1);
        let im = xp.conv1d(&ks, None::<Tensor>, hop, 0, 1, 1);
        (re.square() + im.square()).sqrt()
    }

    #[test]
    fn matches_the_dft_conv_reference_to_1e_4() {
        for (n_fft, hop) in [(64, 16), (256, 64), (512, 128)] {
            let x = Tensor::from_slice(&noise(2048, 3)).view([1, 2048]);
            let x = Tensor::cat(&[&x, &(&x * 0.5)], 0); // B = 2
            let stft = Stft::new(n_fft, hop, Device::Cpu);
            let got = stft.magnitude(&x).unwrap();
            let want = dft_as_conv(&x, n_fft, hop);
            assert_eq!(got.size(), want.size(), "n_fft {n_fft}");
            assert_eq!(got.size(), [2, n_fft / 2 + 1, 1 + 2048 / hop]);
            // Magnitudes are O(n_fft); compare relative to the peak.
            let peak = want.max().double_value(&[]);
            let err = (&got - &want).abs().max().double_value(&[]) / peak;
            assert!(err < 1e-4, "n_fft {n_fft}: rel err {err:.3e}");
        }
    }

    #[test]
    fn agrees_with_the_audio_crate_frames() {
        let x = noise(1600, 9);
        let cpu = unamblify_audio::stft(&x, 256, 64);
        let t = Stft::new(256, 64, Device::Cpu)
            .magnitude(&Tensor::from_slice(&x).view([1, 1600]))
            .unwrap();
        assert_eq!(t.size(), [1, 129, i64::try_from(cpu.len()).unwrap()]);
        let flat: Vec<f32> = cpu.iter().flatten().copied().collect();
        let want = Tensor::from_slice(&flat)
            .view([1, i64::try_from(cpu.len()).unwrap(), 129])
            .transpose(1, 2);
        let err = (&t - &want).abs().max().double_value(&[]);
        assert!(err < 1e-3, "{err}");
        let fb = mel_filterbank(16_000, 256, 20, Device::Cpu);
        assert_eq!(fb.size(), [20, 129]);
    }

    #[test]
    fn magnitudes_have_finite_gradients_even_on_silence() {
        let mut v = noise(800, 1);
        for s in &mut v[300..] {
            *s = 0.0;
        }
        let x = Tensor::from_slice(&v)
            .view([1, 800])
            .set_requires_grad(true);
        let m = Stft::new(128, 32, Device::Cpu).magnitude(&x).unwrap();
        m.sum(Kind::Float).backward();
        let g = x.grad();
        assert_eq!(
            g.isfinite().all().int64_value(&[]),
            1,
            "NaN gradient on silent frames"
        );
        assert!(g.abs().sum(Kind::Float).double_value(&[]) > 0.0);
    }
}
