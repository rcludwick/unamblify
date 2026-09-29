// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! STFT magnitudes and log-mel spectrograms (rustfft), for the metrics and
//! for the dashboard's spectrogram data. Frames are centred: the signal is
//! reflection-padded by `n_fft / 2` on both sides, matching the trainer's
//! `reflection_pad1d + stft` (spec §0), so frame `i` is centred on sample
//! `i · hop` and there are `1 + len / hop` frames.

use std::f32::consts::PI;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Periodic Hann window of length `n`.
#[must_use]
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflection padding by `pad` on both sides (`abc` → `cb|abc|ba`).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    let reflect = |i: isize| -> f32 {
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return x[0];
        }
        let period = 2 * (n as isize - 1);
        let mut j = i.rem_euclid(period);
        if j >= n as isize {
            j = period - j;
        }
        x[j as usize]
    };
    for i in -(pad as isize)..(n + pad) as isize {
        out.push(reflect(i));
    }
    out
}

/// STFT magnitude spectrogram: `frames × (n_fft / 2 + 1)`, periodic Hann,
/// centred frames (see the module docs). `hop` must be > 0.
#[must_use]
pub fn stft(x: &[f32], n_fft: usize, hop: usize) -> Vec<Vec<f32>> {
    let hop = hop.max(1);
    let n_fft = n_fft.max(2);
    let padded = reflect_pad(x, n_fft / 2);
    let win = hann_periodic(n_fft);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);
    let n_frames = 1 + x.len() / hop;
    let n_bins = n_fft / 2 + 1;
    let mut buf = vec![Complex::new(0.0f32, 0.0); n_fft];
    let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * hop;
        for (i, b) in buf.iter_mut().enumerate() {
            let s = padded.get(start + i).copied().unwrap_or(0.0);
            *b = Complex::new(s * win[i], 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        out.push(buf[..n_bins].iter().map(|c| c.norm()).collect());
    }
    out
}

/// Hz → mel (HTK formula).
#[must_use]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// mel → Hz (HTK formula).
#[must_use]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular mel filterbank `n_mels × (n_fft / 2 + 1)` from 0 Hz to
/// `rate / 2`, HTK mel scale, peak 1 (no area normalisation).
#[must_use]
pub fn mel_filterbank(rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let f_max = rate as f32 / 2.0;
    let m_max = hz_to_mel(f_max);
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_max * i as f32 / (n_mels + 1) as f32) * n_fft as f32 / rate as f32)
        .collect();
    (0..n_mels)
        .map(|m| {
            let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            (0..n_bins)
                .map(|k| {
                    let k = k as f32;
                    let up = if mid > lo { (k - lo) / (mid - lo) } else { 0.0 };
                    let down = if hi > mid { (hi - k) / (hi - mid) } else { 0.0 };
                    up.min(down).max(0.0)
                })
                .collect()
        })
        .collect()
}

/// Floor for mel energies before the log, so silence is finite (−100 dB).
pub const MEL_FLOOR: f32 = 1e-10;

/// Log-mel spectrogram `frames × n_mels` in dB: `10·log10(mel · |X|²)`
/// floored at [`MEL_FLOOR`].
#[must_use]
pub fn log_mel(x: &[f32], rate: u32, n_fft: usize, hop: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let fb = mel_filterbank(rate, n_fft, n_mels);
    stft(x, n_fft, hop)
        .iter()
        .map(|mags| {
            fb.iter()
                .map(|filt| {
                    let e: f32 = filt.iter().zip(mags).map(|(w, m)| w * m * m).sum();
                    10.0 * e.max(MEL_FLOOR).log10()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::testutil::sine;

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        assert_eq!(reflect_pad(&[], 2), vec![0.0; 4]);
        assert_eq!(reflect_pad(&[5.0], 1), vec![5.0; 3]);
    }

    #[test]
    fn stft_of_a_sine_peaks_at_the_right_bin() {
        let (rate, n_fft, hop) = (16_000, 512, 128);
        let f = 1_000.0; // bin 32 exactly
        let x = sine(f, rate, 16_000, 0.5);
        let s = stft(&x, n_fft, hop);
        assert_eq!(s.len(), 1 + 16_000 / hop);
        assert_eq!(s[0].len(), n_fft / 2 + 1);
        let mid = &s[s.len() / 2];
        let peak = mid
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        assert_eq!(peak.0, 32);
        // Periodic Hann has coherent gain 0.5: peak ≈ 0.5 · 0.5 · N / 2.
        let expect = 0.5 * 0.5 * n_fft as f32 / 2.0;
        assert!(
            (peak.1 - expect).abs() / expect < 0.01,
            "{} vs {expect}",
            peak.1
        );
        // Two bins away the leakage is > 40 dB down.
        assert!(mid[36] / peak.1 < 0.01);
    }

    #[test]
    fn stft_matches_a_naive_dft_on_one_frame() {
        let x = crate::testutil::noise(64, 1.0, 3);
        let n_fft = 64;
        let s = stft(&x, n_fft, 64);
        // Frame 1 is centred on sample 64 → covers padded[64..128] = x[32..] + reflected.
        let padded = reflect_pad(&x, 32);
        let win = hann_periodic(n_fft);
        for (k, got) in s[1].iter().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for n in 0..n_fft {
                let v = padded[64 + n] * win[n];
                let ph = -2.0 * PI * (k * n) as f32 / n_fft as f32;
                re += v * ph.cos();
                im += v * ph.sin();
            }
            let want = (re * re + im * im).sqrt();
            assert!((got - want).abs() < 1e-3, "bin {k}: {got} vs {want}");
        }
    }

    #[test]
    fn filterbank_covers_the_band_and_log_mel_has_the_right_shape() {
        let fb = mel_filterbank(16_000, 1024, 80);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 513);
        for (m, filt) in fb.iter().enumerate() {
            let peak = filt.iter().copied().fold(0.0, f32::max);
            assert!(peak > 0.5, "mel {m} peak {peak}");
        }
        // Every bin except DC/Nyquist is covered by some filter.
        for k in 1..512 {
            assert!(fb.iter().any(|f| f[k] > 0.0), "bin {k} uncovered");
        }
        assert!((hz_to_mel(mel_to_hz(1234.0)) - 1234.0).abs() < 1e-2);

        let lm = log_mel(&sine(1_000.0, 16_000, 16_000, 0.5), 16_000, 1024, 256, 80);
        assert_eq!(lm.len(), 1 + 16_000 / 256);
        assert_eq!(lm[0].len(), 80);
        let silent = log_mel(&[0.0; 4096], 16_000, 1024, 256, 80);
        assert!(silent.iter().flatten().all(|&v| (v + 100.0).abs() < 1e-3));
    }
}
