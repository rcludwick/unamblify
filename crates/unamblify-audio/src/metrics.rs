// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The objective metrics of `docs/design/training.md` that need no C or
//! Python: log-spectral distance, mel-L1, SI-SDR, segmental SNR — and the
//! cross-correlation lag finder the capture harness uses for the canary.
//!
//! Pairs are compared over their common length; a mismatch in length is
//! not an error, the longer input is truncated.

use crate::spectral::{log_mel, stft};

/// Normalised autocorrelation peak a frame needs before [`hnr_db`]
/// treats it as voiced enough to have harmonics worth measuring.
const VOICED_PERIODICITY: f32 = 0.45;
/// Top of the band [`hnr_db`] measures: what a narrowband vocoder sends.
const HNR_MAX_HZ: f32 = 4000.0;
/// Half-width of the harmonic and trough windows, in units of `f0`.
const HNR_HALF_WIDTH: f32 = 0.18;

/// STFT size for a window of `ms` milliseconds at `rate`, rounded up to a
/// power of two (32 ms → 512 at 16 kHz, 256 at 8 kHz).
#[must_use]
pub fn n_fft_for(rate: u32, ms: u32) -> usize {
    ((rate * ms / 1_000) as usize).next_power_of_two().max(16)
}

/// Power floor inside the logs, so silence is finite.
const POWER_FLOOR: f32 = 1e-10;

/// Log-spectral distance in dB: per frame the RMS over bins of the
/// difference of `10·log10(|X|²)`, averaged over frames. 32 ms window,
/// 8 ms hop. 0 for identical signals; lower is better.
#[must_use]
pub fn lsd(a: &[f32], b: &[f32], rate: u32) -> f32 {
    let n = a.len().min(b.len());
    let n_fft = n_fft_for(rate, 32);
    let (sa, sb) = (
        stft(&a[..n], n_fft, n_fft / 4),
        stft(&b[..n], n_fft, n_fft / 4),
    );
    let per_frame: Vec<f32> = sa
        .iter()
        .zip(&sb)
        .map(|(fa, fb)| {
            let ms: f32 = fa
                .iter()
                .zip(fb)
                .map(|(x, y)| {
                    let d = 10.0
                        * ((x * x).max(POWER_FLOOR).log10() - (y * y).max(POWER_FLOOR).log10());
                    d * d
                })
                .sum::<f32>()
                / fa.len() as f32;
            ms.sqrt()
        })
        .collect();
    mean(&per_frame)
}

/// **How much more periodic `a` is than `b`.**
///
/// Cepstral peak prominence — the height of the pitch peak in the real
/// cepstrum above that quefrency band's mean — measures how sharply a
/// harmonic comb stands out of the spectrum. A perfectly periodic buzz
/// has a tall peak; real speech, with breath noise and jitter between
/// its harmonics, has a shorter one. This returns
/// `mean(cpp(a) − cpp(b))` over frames loud enough to be speech.
///
/// **Positive means `a` is buzzier than `b`.** A vocoder's output scores
/// well above its own input, and that excess is what listeners call
/// robotic; see `docs/theory/what-the-codec-destroys.md`.
#[must_use]
pub fn cpp_excess(a: &[f32], b: &[f32], rate: u32) -> f32 {
    let n = a.len().min(b.len());
    let n_fft = n_fft_for(rate, 64);
    let hop = n_fft / 4;
    let (sa, sb) = (stft(&a[..n], n_fft, hop), stft(&b[..n], n_fft, hop));
    // 60-320 Hz: the quefrency band a human pitch can land in.
    let lo = (rate as usize / 320).max(1);
    let hi = (rate as usize / 60).min(n_fft / 2);
    let mut diffs = Vec::with_capacity(sa.len());
    for (fa, fb) in sa.iter().zip(&sb) {
        // Silence has no meaningful periodicity.
        let energy: f32 = fb.iter().map(|x| x * x).sum::<f32>() / fb.len() as f32;
        if energy.sqrt() < 1e-3 || hi <= lo + 2 {
            continue;
        }
        if let (Some(ca), Some(cb)) = (cpp_frame(fa, lo, hi), cpp_frame(fb, lo, hi)) {
            diffs.push(ca - cb);
        }
    }
    mean(&diffs)
}

/// Cepstral peak prominence of one magnitude frame: the largest value in
/// the pitch quefrency band of the real cepstrum, over that band's mean.
fn cpp_frame(mag: &[f32], lo: usize, hi: usize) -> Option<f32> {
    cpp_frame_at(mag, lo, hi).map(|(p, _)| p)
}

/// As [`cpp_frame`], also returning the quefrency (in samples) the peak
/// sat at — the pitch period, which [`hnr_db`] needs to know where the
/// harmonics are.
fn cpp_frame_at(mag: &[f32], lo: usize, hi: usize) -> Option<(f32, usize)> {
    let log_mag: Vec<f32> = mag
        .iter()
        .map(|m| (m * m).max(POWER_FLOOR).log10())
        .collect();
    // Real cepstrum of a real, even log-spectrum: a DCT-like sum. The
    // spectrum is half of a symmetric sequence, so this is the real part
    // of its inverse DFT, which is all that the prominence needs.
    let bins = log_mag.len();
    let hi = hi.min(bins.saturating_sub(1));
    if hi <= lo + 2 {
        return None;
    }
    let mut band = Vec::with_capacity(hi - lo);
    for q in lo..hi {
        let mut acc = 0.0f32;
        for (k, v) in log_mag.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let phase = std::f32::consts::PI * (k as f32) * (q as f32) / (bins as f32);
            acc += v * phase.cos();
        }
        #[allow(clippy::cast_precision_loss)]
        band.push(acc / bins as f32);
    }
    let (idx, peak) = band
        .iter()
        .copied()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        );
    let base = mean(&band);
    (peak.is_finite() && base.is_finite()).then_some((peak - base, lo + idx))
}

/// **How far the harmonics stand above the valleys between them, in dB.**
///
/// For every clearly voiced frame this finds the pitch by
/// autocorrelation, then compares the *peak* power at each harmonic
/// (`k·f0`) with the *median* power halfway between them
/// (`(k + ½)·f0`), over 0–4 kHz — the band a narrowband vocoder
/// actually transmits. Peak against median, rather than mean against
/// mean: the skirts of a harmonic would otherwise fill the window and
/// wash the contrast out.
///
/// **Every harmonic counts once.** The ratio is taken per harmonic, in
/// dB, and those are averaged — not the mean of the peaks over the mean
/// of the troughs, which the first few (loudest) harmonics dominate so
/// completely that the number is really the 0–1 kHz band's. Measured
/// that way, a model that fixed the low band and left 2–4 kHz alone
/// read as natural; per harmonic, it does not.
///
/// Natural speech scores roughly **17 dB**: the valleys are not empty,
/// they hold breath, jitter and glottal noise. A vocoder that codes
/// voiced bands as exactly periodic empties them, and a D-STAR decode
/// measures several dB more. **Higher is buzzier**, and the excess over
/// the clean reference is the measurable part of "robotic".
///
/// Returns `None` when nothing in `x` is voiced enough to measure.
#[must_use]
pub fn hnr_db(x: &[f32], rate: u32) -> Option<f32> {
    hnr_on(x, rate, &voiced_frames(x, rate))
}

/// **How much buzzier `a` is than `b`, dB**, measured on the *same*
/// frames — the ones `b` says are voiced, using `b`'s pitch.
///
/// Gating on the reference is the whole point. If each signal chose its
/// own voiced frames, a buzzier signal would pass the voicing test on
/// more of them — including frames where the reference is only weakly
/// periodic — and the comparison would flatter whichever signal is
/// already the more periodic. The training loss gates on the target for
/// the same reason.
///
/// **Positive means `a` is the more periodic**, which for a model output
/// against clean speech is the residual robotic quality. This is the
/// mean of the per-frame differences; [`hnr_excess_stats`] also gives
/// the mean *absolute* per-frame difference, which a signal that is too
/// buzzy on half its frames and too breathy on the rest cannot hide from.
#[must_use]
pub fn hnr_excess(a: &[f32], b: &[f32], rate: u32) -> Option<f32> {
    hnr_excess_stats(a, b, rate).map(|s| s.mean)
}

/// The harmonic-to-trough comparison of two signals, frame by frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnrExcess {
    /// Mean of the per-frame differences, dB (positive = `a` buzzier).
    pub mean: f32,
    /// Mean of the per-frame *absolute* differences, dB. Equal to
    /// `|mean|` only when every frame errs the same way; a model that
    /// adds a constant share of noise instead of following the
    /// reference's breathiness has a mean near zero and a large `abs`.
    pub abs: f32,
    /// Voiced frames the two were compared on.
    pub frames: usize,
}

/// [`hnr_excess`] with the per-frame absolute error alongside the mean.
/// `None` when the reference has no voiced frame both signals cover.
#[must_use]
pub fn hnr_excess_stats(a: &[f32], b: &[f32], rate: u32) -> Option<HnrExcess> {
    let frames = voiced_frames(b, rate);
    let (fa, fb) = (hnr_frames(a, rate, &frames), hnr_frames(b, rate, &frames));
    let diffs: Vec<f32> = fa
        .iter()
        .zip(&fb)
        .filter_map(|(x, y)| Some(x.as_ref()? - y.as_ref()?))
        .collect();
    (!diffs.is_empty()).then(|| HnrExcess {
        mean: mean(&diffs),
        abs: mean(&diffs.iter().map(|d| d.abs()).collect::<Vec<_>>()),
        frames: diffs.len(),
    })
}

/// The voiced frames of `x`: `(start sample, f0)` for every analysis
/// frame periodic enough to have harmonics worth measuring.
fn voiced_frames(x: &[f32], rate: u32) -> Vec<(usize, f32)> {
    let n_fft = n_fft_for(rate, 64);
    let hop = n_fft / 4;
    let lo_lag = (rate as usize / 320).max(1);
    let hi_lag = (rate as usize / 60).min(n_fft - 1);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + n_fft <= x.len() {
        if let Some((f0, periodicity)) = f0_autocorr(&x[start..start + n_fft], rate, lo_lag, hi_lag)
            && periodicity >= VOICED_PERIODICITY
        {
            out.push((start, f0));
        }
        start += hop;
    }
    out
}

/// Mean harmonic-to-trough ratio of `x` over the given frames, dB.
/// Frames past the end of `x` are skipped, so two signals of slightly
/// different length can still be compared on one frame list.
fn hnr_on(x: &[f32], rate: u32, frames: &[(usize, f32)]) -> Option<f32> {
    let per_frame: Vec<f32> = hnr_frames(x, rate, frames).into_iter().flatten().collect();
    (!per_frame.is_empty()).then(|| mean(&per_frame))
}

/// The harmonic-to-trough ratio of `x` on each of `frames`, dB; `None`
/// for a frame past the end of `x` or with too few harmonics to measure.
/// One entry per frame, in order, so two signals measured on the same
/// list can be compared frame by frame.
fn hnr_frames(x: &[f32], rate: u32, frames: &[(usize, f32)]) -> Vec<Option<f32>> {
    let n_fft = n_fft_for(rate, 64);
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / n_fft as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect();
    frames
        .iter()
        .map(|&(start, f0)| {
            if start + n_fft > x.len() {
                return None;
            }
            let spec = power_spectrum(&x[start..start + n_fft], &window);
            frame_hnr(&spec, f0, rate, n_fft)
        })
        .collect()
}

/// Pitch and its normalised autocorrelation peak, or `None` for silence.
fn f0_autocorr(frame: &[f32], rate: u32, lo_lag: usize, hi_lag: usize) -> Option<(f32, f32)> {
    #[allow(clippy::cast_precision_loss)]
    let n = frame.len() as f32;
    let dc = frame.iter().sum::<f32>() / n;
    let x: Vec<f32> = frame.iter().map(|v| v - dc).collect();
    let energy: f32 = x.iter().map(|v| v * v).sum();
    if (energy / n).sqrt() < 1e-4 || hi_lag <= lo_lag {
        return None;
    }
    let mut best = (0usize, f32::NEG_INFINITY);
    for lag in lo_lag..hi_lag.min(x.len() - 1) {
        let mut acc = 0.0f32;
        for i in 0..x.len() - lag {
            acc += x[i] * x[i + lag];
        }
        if acc > best.1 {
            best = (lag, acc);
        }
    }
    if best.0 == 0 || !best.1.is_finite() {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let f0 = rate as f32 / best.0 as f32;
    (60.0..=320.0)
        .contains(&f0)
        .then_some((f0, best.1 / energy))
}

/// Windowed power spectrum, `n_fft / 2` bins.
pub(crate) fn power_spectrum(frame: &[f32], window: &[f32]) -> Vec<f32> {
    let n = frame.len();
    (0..n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, (s, w)) in frame.iter().zip(window).enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let ph = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                let v = s * w;
                re += v * ph.cos();
                im += v * ph.sin();
            }
            re * re + im * im
        })
        .collect()
}

/// Harmonic-peak against inter-harmonic-median power for one frame, dB:
/// the mean over harmonics of each harmonic's own ratio in dB, so every
/// harmonic below 4 kHz counts once whatever its level.
fn frame_hnr(power: &[f32], f0: f32, rate: u32, n_fft: usize) -> Option<f32> {
    #[allow(clippy::cast_precision_loss)]
    let bin_hz = rate as f32 / n_fft as f32;
    let bw = f0 * HNR_HALF_WIDTH;
    let mut ratios = Vec::new();
    let mut k = 1.0f32;
    while k * f0 < HNR_MAX_HZ {
        let fh = k * f0;
        let ft = (k + 0.5) * f0;
        let band = |centre: f32| -> Vec<f32> {
            let lo = ((centre - bw) / bin_hz).ceil().max(0.0);
            let hi = ((centre + bw) / bin_hz).floor();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let (lo, hi) = (
                lo as usize,
                (hi as usize).min(power.len().saturating_sub(1)),
            );
            if lo > hi {
                Vec::new()
            } else {
                power[lo..=hi].to_vec()
            }
        };
        let (hb, tb) = (band(fh), band(ft));
        if !hb.is_empty() && !tb.is_empty() && ft < HNR_MAX_HZ {
            let peak = hb.iter().copied().fold(0.0f32, f32::max);
            let ratio = (peak + POWER_FLOOR) / (median(tb) + POWER_FLOOR);
            if ratio.is_finite() {
                ratios.push(10.0 * ratio.log10());
            }
        }
        k += 1.0;
    }
    (ratios.len() >= 6).then(|| mean(&ratios))
}

/// Median of a small slice (sorts a copy).
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        f32::midpoint(v[n / 2 - 1], v[n / 2])
    }
}

/// Mean absolute difference of the 80-bin log-mel spectrograms, dB. 64 ms
/// window, 16 ms hop. 0 for identical signals; lower is better.
#[must_use]
pub fn mel_l1(a: &[f32], b: &[f32], rate: u32) -> f32 {
    let n = a.len().min(b.len());
    let n_fft = n_fft_for(rate, 64);
    let (ma, mb) = (
        log_mel(&a[..n], rate, n_fft, n_fft / 4, 80),
        log_mel(&b[..n], rate, n_fft, n_fft / 4, 80),
    );
    let diffs: Vec<f32> = ma
        .iter()
        .zip(&mb)
        .flat_map(|(fa, fb)| fa.iter().zip(fb).map(|(x, y)| (x - y).abs()))
        .collect();
    mean(&diffs)
}

/// Scale-invariant SDR in dB (Le Roux et al. 2019), both inputs
/// zero-meaned. Identical signals give a large positive value (capped by
/// the floor at ≈ 100 dB); higher is better.
#[must_use]
pub fn si_sdr(est: &[f32], reference: &[f32]) -> f32 {
    let n = est.len().min(reference.len());
    if n == 0 {
        return f32::NEG_INFINITY;
    }
    let (me, mr) = (mean(&est[..n]), mean(&reference[..n]));
    let e: Vec<f64> = est[..n].iter().map(|&v| f64::from(v - me)).collect();
    let r: Vec<f64> = reference[..n].iter().map(|&v| f64::from(v - mr)).collect();
    let rr: f64 = r.iter().map(|v| v * v).sum();
    if rr <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let er: f64 = e.iter().zip(&r).map(|(x, y)| x * y).sum();
    let alpha = er / rr;
    let target: f64 = alpha * alpha * rr;
    let noise: f64 = e.iter().zip(&r).map(|(x, y)| (x - alpha * y).powi(2)).sum();
    (10.0 * (target / noise.max(f64::from(POWER_FLOOR) * target.max(1e-30))).log10()) as f32
}

/// Segmental SNR in dB over 20 ms segments, each clamped to `[-10, 35]` dB
/// as is conventional, then averaged. Higher is better.
#[must_use]
pub fn seg_snr(est: &[f32], reference: &[f32], rate: u32) -> f32 {
    let n = est.len().min(reference.len());
    let seg = ((rate / 50) as usize).max(1);
    let per_seg: Vec<f32> = est[..n]
        .chunks_exact(seg)
        .zip(reference[..n].chunks_exact(seg))
        .map(|(e, r)| {
            let sig: f32 = r.iter().map(|v| v * v).sum();
            let err: f32 = e.iter().zip(r).map(|(x, y)| (x - y) * (x - y)).sum();
            (10.0 * (sig.max(POWER_FLOOR) / err.max(POWER_FLOOR)).log10()).clamp(-10.0, 35.0)
        })
        .collect();
    mean(&per_seg)
}

/// Lag of `b` relative to `a` in samples, searched over `[-max_lag,
/// max_lag]`: the `lag` maximising `Σ a[n] · b[n + lag]`. A positive value
/// means `b` is a delayed copy of `a` (`b[n] ≈ a[n − lag]`) — the shape of
/// a vocoder's output against its input. Ties resolve to the smallest
/// `|lag|`.
#[must_use]
pub fn xcorr_lag(a: &[f32], b: &[f32], max_lag: usize) -> i32 {
    let max_lag = max_lag as i64;
    let (na, nb) = (a.len() as i64, b.len() as i64);
    let mut best = (0i32, f64::NEG_INFINITY);
    // Visit lags by increasing |lag| so ties keep the smaller shift.
    let order = (0..=max_lag).flat_map(|l| if l == 0 { vec![0] } else { vec![l, -l] });
    for lag in order {
        let lo = (-lag).max(0);
        let hi = na.min(nb - lag);
        if hi <= lo {
            continue;
        }
        let mut acc = 0.0f64;
        for n in lo..hi {
            acc += f64::from(a[n as usize]) * f64::from(b[(n + lag) as usize]);
        }
        if acc > best.1 {
            best = (lag as i32, acc);
        }
    }
    best.0
}

pub(crate) fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64) as f32
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// `hnr_db` reads a bare pulse train as far buzzier than the same
    /// train with noise between its harmonics, and `hnr_excess` is the
    /// signed difference. These are the units the project quotes:
    /// clean speech ~19 dB, a D-STAR decode ~29 dB.
    #[test]
    fn hnr_db_separates_a_bare_comb_from_one_with_a_noise_floor() {
        let rate = 16_000;
        let n = 16_000;
        let buzz: Vec<f32> = (0..n)
            .map(|i| if i % 128 == 0 { 0.8 } else { 0.0 })
            .collect();
        let mut seed = 4u32;
        let breathy: Vec<f32> = buzz
            .iter()
            .map(|v| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let u = (seed >> 16) as f32 / 65_536.0 - 0.5;
                v + u * 0.03
            })
            .collect();
        let hard = hnr_db(&buzz, rate).expect("a pulse train is voiced");
        let soft = hnr_db(&breathy, rate).expect("still voiced");
        assert!(hard > soft + 5.0, "bare comb {hard} vs breathy {soft}");
        let excess = hnr_excess(&buzz, &breathy, rate).unwrap();
        assert!(excess > 0.0, "the bare comb is the buzzier one");
        assert!(
            hnr_excess(&breathy, &buzz, rate).unwrap() < 0.0,
            "and the comparison is signed"
        );
        // Silence has nothing to measure.
        assert!(hnr_db(&vec![0.0; n], rate).is_none());
        assert!(hnr_excess(&buzz, &vec![0.0; n], rate).is_none());
    }

    /// **The comparison is gated on the reference, not on each signal.**
    /// A buzzier signal passes a per-signal voicing test on more frames,
    /// including ones where the reference is only weakly periodic, which
    /// flatters whichever signal is already the more periodic. Here the
    /// reference is voiced only in its first half; buzzy content far
    /// outside that — beyond any frame the reference marks voiced — must
    /// not move the answer at all.
    #[test]
    fn hnr_excess_measures_both_signals_on_the_references_frames() {
        let rate = 16_000;
        let n = 32_000;
        let half = n / 2;
        let tail = 3 * n / 4; // a silent guard band between half and tail
        let mut seed = 21u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let v = (seed >> 16) as f32 / 65_536.0 - 0.5;
            v
        };
        let pulse = |i: usize| if i.is_multiple_of(128) { 0.8 } else { 0.0 };
        // Voiced, with a real noise floor, in the first half only.
        let reference: Vec<f32> = (0..n)
            .map(|i| {
                if i < half {
                    pulse(i) + noise() * 0.03
                } else {
                    0.0
                }
            })
            .collect();
        // Bare comb over the same region, and again in the last quarter.
        let with_tail: Vec<f32> = (0..n)
            .map(|i| if i < half || i >= tail { pulse(i) } else { 0.0 })
            .collect();
        let without_tail: Vec<f32> = (0..n)
            .map(|i| if i < half { pulse(i) } else { 0.0 })
            .collect();

        let a = hnr_excess(&with_tail, &reference, rate).expect("voiced frames exist");
        let b = hnr_excess(&without_tail, &reference, rate).unwrap();
        assert!(
            (a - b).abs() < 1e-3,
            "buzz outside the reference's voiced frames moved the answer: {a} vs {b}"
        );
        assert!(
            a > 0.0,
            "a bare comb is buzzier than one with a noise floor"
        );
    }

    /// A harmonic comb with `1/k` amplitudes plus band-limited noise:
    /// `n_sines` random-frequency, random-phase sines in `lo..hi` Hz.
    fn comb_with_noise(n: usize, lo: f32, hi: f32, n_sines: usize, seed: u32) -> Vec<f32> {
        let rate = 16_000.0f32;
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let v = (s >> 16) as f32 / 65_536.0;
            v
        };
        let sines: Vec<(f32, f32)> = (0..n_sines)
            .map(|_| (lo + rnd() * (hi - lo), rnd() * 2.0 * std::f32::consts::PI))
            .collect();
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / rate;
                let mut v = 0.0f32;
                for k in 1..=31 {
                    #[allow(clippy::cast_precision_loss)]
                    let kf = k as f32;
                    v += (2.0 * std::f32::consts::PI * 125.0 * kf * t).cos() * 0.3 / kf;
                }
                for &(f, ph) in &sines {
                    v += (2.0 * std::f32::consts::PI * f * t + ph).sin() * 0.004;
                }
                v
            })
            .collect()
    }

    /// **Every harmonic counts once.** Noise under the quiet top half of
    /// the band moves the excess about half as much as the same noise
    /// density over the whole band — whereas a mean-of-peaks over
    /// mean-of-troughs ratio is the loudest harmonics' ratio, and
    /// whatever happens above them barely registers.
    #[test]
    fn hnr_weights_every_harmonic_equally() {
        let rate = 16_000;
        let n = 32_000;
        let bare = comb_with_noise(n, 2000.0, 4000.0, 0, 1);
        let top = comb_with_noise(n, 2000.0, 4000.0, 200, 2);
        let full = comb_with_noise(n, 125.0, 4000.0, 400, 3);
        let over_top = hnr_excess(&bare, &top, rate).unwrap();
        let over_full = hnr_excess(&bare, &full, rate).unwrap();
        assert!(over_full > 5.0, "the comparison has range: {over_full}");
        let share = over_top / over_full;
        assert!(
            (0.35..=0.65).contains(&share),
            "noise over half the harmonics should count about half: {over_top} of {over_full} ({share:.2})"
        );
    }

    /// The mean of the per-frame differences can be zero while the
    /// per-frame error is large: too buzzy on one half of the frames and
    /// too breathy on the other. `abs` says so; `mean` does not.
    #[test]
    fn hnr_excess_stats_abs_sees_errors_that_cancel_in_the_mean() {
        let rate = 16_000;
        let n = 32_000;
        let half = n / 2;
        let mid = comb_with_noise(n, 125.0, 4000.0, 200, 4);
        let bare = comb_with_noise(n, 125.0, 4000.0, 0, 5);
        let noisy = comb_with_noise(n, 125.0, 4000.0, 800, 6);
        let mixed: Vec<f32> = (0..n)
            .map(|i| if i < half { bare[i] } else { noisy[i] })
            .collect();
        let same = hnr_excess_stats(&mid, &mid, rate).unwrap();
        assert!(same.mean.abs() < 1e-6 && same.abs < 1e-6 && same.frames > 10);
        let s = hnr_excess_stats(&mixed, &mid, rate).unwrap();
        assert!(
            s.abs > s.mean.abs() + 2.0,
            "abs {} should exceed |mean| {} by a margin",
            s.abs,
            s.mean
        );
        assert!(s.abs > 3.0, "the per-frame error is large: {}", s.abs);
    }

    /// `cpp_excess` is positive when the first signal is the more
    /// periodic of the two: a bare pulse train against the same train
    /// with noise between its harmonics.
    #[test]
    fn cpp_excess_is_positive_for_the_buzzier_signal() {
        let rate = 16_000;
        let period = 128; // 125 Hz
        let n = 16_000;
        let buzz: Vec<f32> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        // The same pulses, plus low-level noise — what breath does to a
        // real voice.
        let mut seed = 12_345u32;
        let breathy: Vec<f32> = buzz
            .iter()
            .map(|p| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let u = (seed >> 8) as f32 / f32::from(u16::MAX) / 256.0 - 0.5;
                p + u * 0.15
            })
            .collect();
        let excess = cpp_excess(&buzz, &breathy, rate);
        assert!(excess > 0.0, "buzz should be the more periodic: {excess}");
        assert!(
            cpp_excess(&breathy, &buzz, rate) < 0.0,
            "and the comparison should be antisymmetric"
        );
        // A signal against itself has no excess.
        assert!(cpp_excess(&buzz, &buzz, rate).abs() < 1e-4);
    }

    use crate::testutil::{noise, sine};

    #[test]
    fn identical_signals_score_perfectly() {
        let x = noise(16_000, 0.3, 11);
        assert!(si_sdr(&x, &x) > 80.0, "{}", si_sdr(&x, &x));
        assert!(lsd(&x, &x, 16_000).abs() < 1e-6);
        assert!(mel_l1(&x, &x, 16_000).abs() < 1e-6);
        assert!((seg_snr(&x, &x, 16_000) - 35.0).abs() < 1e-6);
    }

    #[test]
    fn si_sdr_is_scale_invariant_and_penalises_noise() {
        let x = sine(440.0, 16_000, 16_000, 0.5);
        let scaled: Vec<f32> = x.iter().map(|v| v * 0.1).collect();
        assert!(si_sdr(&scaled, &x) > 80.0);
        let noisy: Vec<f32> = x
            .iter()
            .zip(noise(16_000, 0.05, 5))
            .map(|(a, b)| a + b)
            .collect();
        let s = si_sdr(&noisy, &x);
        // sine rms 0.354, noise rms ≈ 0.05/√3 = 0.0289 → ≈ 21.8 dB.
        assert!((s - 21.8).abs() < 1.0, "{s}");
        assert_eq!(si_sdr(&[], &[]), f32::NEG_INFINITY);
        assert_eq!(si_sdr(&x, &[0.0; 100]), f32::NEG_INFINITY);
    }

    #[test]
    fn spectral_metrics_grow_with_distortion() {
        let x = noise(16_000, 0.3, 21);
        let a: Vec<f32> = x
            .iter()
            .zip(noise(16_000, 0.03, 22))
            .map(|(a, b)| a + b)
            .collect();
        let b: Vec<f32> = x
            .iter()
            .zip(noise(16_000, 0.3, 23))
            .map(|(a, b)| a + b)
            .collect();
        assert!(lsd(&a, &x, 16_000) < lsd(&b, &x, 16_000));
        assert!(mel_l1(&a, &x, 16_000) < mel_l1(&b, &x, 16_000));
        assert!(seg_snr(&a, &x, 16_000) > seg_snr(&b, &x, 16_000));
        assert_eq!(n_fft_for(16_000, 32), 512);
        assert_eq!(n_fft_for(8_000, 32), 256);
        assert_eq!(n_fft_for(16_000, 64), 1024);
    }

    #[test]
    fn lag_is_found_on_a_shifted_signal() {
        let x = noise(8_000, 0.5, 42);
        let d = 42usize;
        let mut delayed = vec![0.0f32; d];
        delayed.extend_from_slice(&x[..x.len() - d]);
        assert_eq!(xcorr_lag(&x, &delayed, 400), 42);
        assert_eq!(xcorr_lag(&delayed, &x, 400), -42);
        assert_eq!(xcorr_lag(&x, &x, 400), 0);
        // Out of search range: the true lag is not reachable, but nothing panics.
        let _ = xcorr_lag(&x, &delayed, 10);
        // A decoded-with-noise copy still locks.
        let noisy: Vec<f32> = delayed
            .iter()
            .zip(noise(8_000, 0.2, 9))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(xcorr_lag(&x, &noisy, 400), 42);
    }
}
