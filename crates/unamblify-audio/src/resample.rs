// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Anti-aliased resampling on rubato's windowed-sinc `SincFixedIn`.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::Result;

/// Input frames handed to rubato per call.
const CHUNK: usize = 1024;

/// Output length of resampling `n` samples from `from` to `to` Hz:
/// `n · to / from`, rounded to nearest.
#[must_use]
pub fn output_len(n: usize, from: u32, to: u32) -> usize {
    if from == 0 {
        return 0;
    }
    ((n as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Resample `x` from `from` Hz to `to` Hz with a 256-tap windowed sinc
/// (Blackman-Harris², cutoff 0.95 of the lower Nyquist). The output is
/// sample-exact in time with the input (`y[k]` is `x` at `k · from / to`)
/// and exactly [`output_len`] long.
///
/// Alignment: rubato's `SincFixedIn` (0.16) produces `y[k] = x((k + 1) ·
/// t − 1)` with `t = from / to` — it compensates the sinc's group delay
/// itself, so `output_delay()` must not be discarded, but a residual
/// `t − 1` input samples of lead remains. It is cancelled exactly by
/// prepending `from / g − 1` zeros to the input and dropping `to / g − 1`
/// output samples (`g = gcd(from, to)`). Measured, not documented; see
/// `docs/notes/2026-09-10-rubato-output-delay.md`.
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(x.to_vec());
    }
    let want = output_len(x.len(), from, to);
    if want == 0 {
        return Ok(Vec::new());
    }
    let g = gcd(u64::from(from), u64::from(to));
    let lead_in = (u64::from(from) / g - 1) as usize;
    let drop_out = (u64::from(to) / g - 1) as usize;

    let ratio = f64::from(to) / f64::from(from);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut rs = SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK, 1)?;
    let mut out: Vec<f32> = Vec::with_capacity(drop_out + want + 2 * CHUNK);

    let mut padded = Vec::with_capacity(lead_in + x.len());
    padded.resize(lead_in, 0.0);
    padded.extend_from_slice(x);
    let (chunks, rest) = padded.as_chunks::<CHUNK>();
    for chunk in chunks {
        let y = rs.process(&[chunk.as_slice()], None)?;
        out.extend_from_slice(&y[0]);
    }
    if !rest.is_empty() {
        let y = rs.process_partial(Some(&[rest]), None)?;
        out.extend_from_slice(&y[0]);
    }
    // Flush the filter's tail until the whole input has come out.
    while out.len() < drop_out + want {
        let y = rs.process_partial::<&[f32]>(None, None)?;
        out.extend_from_slice(&y[0]);
    }
    out.drain(..drop_out);
    out.truncate(want);
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::spectral::stft;
    use crate::testutil::sine;

    fn peak_bin(mags: &[Vec<f32>]) -> usize {
        let mid = &mags[mags.len() / 2];
        mid.iter()
            .enumerate()
            .fold((0, 0.0f32), |a, (i, &v)| if v > a.1 { (i, v) } else { a })
            .0
    }

    #[test]
    fn lengths_are_exact() {
        assert_eq!(output_len(48_000, 48_000, 16_000), 16_000);
        assert_eq!(output_len(16_000, 16_000, 8_000), 8_000);
        assert_eq!(output_len(44_100, 44_100, 16_000), 16_000);
        assert_eq!(output_len(1001, 44_100, 16_000), 363);
        let x = sine(300.0, 44_100, 1001, 0.5);
        assert_eq!(resample(&x, 44_100, 16_000).unwrap().len(), 363);
        assert_eq!(resample(&x, 44_100, 44_100).unwrap(), x);
        assert!(resample(&[], 48_000, 16_000).unwrap().is_empty());
    }

    #[test]
    fn a_sine_keeps_its_frequency_amplitude_and_phase() {
        let f = 1_000.0;
        let x = sine(f, 48_000, 48_000, 0.5);
        let y = resample(&x, 48_000, 16_000).unwrap();
        assert_eq!(y.len(), 16_000);
        let n_fft = 1024;
        let bin = peak_bin(&stft(&y, n_fft, 256));
        let expect = (f / 16_000.0 * n_fft as f32).round() as usize;
        assert_eq!(bin, expect);
        // Amplitude and alignment: compare against the ideal 16 kHz sine
        // away from the edges.
        let ideal = sine(f, 16_000, 16_000, 0.5);
        let err = y[1000..15_000]
            .iter()
            .zip(&ideal[1000..15_000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.01, "max err {err}");
    }

    #[test]
    fn output_is_time_aligned_with_the_input() {
        use crate::metrics::xcorr_lag;
        // White noise has energy up to Nyquist, which the anti-alias
        // transition band removes; one round trip band-limits it, and the
        // second must then land on the first to within a hundredth.
        let n8 = crate::testutil::noise(16_000, 0.3, 5);
        let base = resample(&resample(&n8, 8_000, 16_000).unwrap(), 16_000, 8_000).unwrap();
        assert_eq!(base.len(), 16_000);
        assert_eq!(xcorr_lag(&n8, &base, 400), 0);
        let up = resample(&base, 8_000, 16_000).unwrap();
        assert_eq!(up.len(), 32_000);
        let down = resample(&up, 16_000, 8_000).unwrap();
        assert_eq!(xcorr_lag(&base, &down, 400), 0);
        // The transition band (3.6–4 kHz, a tenth of white noise's energy)
        // is attenuated a little more on every pass, so compare energies
        // rather than demand sample equality; a misalignment of even half
        // a sample would push this past 0.5.
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (a, b) in base[500..15_500].iter().zip(&down[500..15_500]) {
            num += (a - b) * (a - b);
            den += a * a;
        }
        let rel = (num / den).sqrt();
        assert!(rel < 0.06, "round-trip relative rms error {rel}");
        let via48 = resample(&resample(&base, 8_000, 48_000).unwrap(), 48_000, 16_000).unwrap();
        assert_eq!(xcorr_lag(&up, &via48, 400), 0);
    }

    /// Fit a fractional delay to a resampled sine: the output must sit on
    /// the ideal sine to within 0.02 samples with unity gain, at every
    /// ratio the corpora need (48 k, 44.1 k, 22.05 k, 24 k → 16 k; 16 k ↔ 8 k).
    #[test]
    fn output_is_sub_sample_aligned_at_every_corpus_ratio() {
        let ratios = [
            (48_000u32, 16_000u32),
            (44_100, 16_000),
            (22_050, 16_000),
            (24_000, 16_000),
            (16_000, 8_000),
            (8_000, 16_000),
        ];
        for (from, to) in ratios {
            for f in [300.0f32, 3_000.0] {
                let x = sine(f, from, from as usize, 0.5);
                let y = resample(&x, from, to).unwrap();
                let w = 2.0 * std::f32::consts::PI * f / to as f32;
                let fit = |d: f32| {
                    (1000..y.len() - 1000)
                        .map(|i| (y[i] - 0.5 * (w * (i as f32 - d)).sin()).abs())
                        .fold(0.0, f32::max)
                };
                let mut best = (0.0f32, f32::MAX);
                let mut d = -1.0f32;
                while d <= 1.0 {
                    let err = fit(d);
                    if err < best.1 {
                        best = (d, err);
                    }
                    d += 0.01;
                }
                assert!(
                    best.0.abs() < 0.02,
                    "{from}->{to} {f} Hz: delay {} err {}",
                    best.0,
                    best.1
                );
                assert!(best.1 < 0.004, "{from}->{to} {f} Hz: err {}", best.1);
            }
        }
    }

    #[test]
    fn content_above_the_new_nyquist_is_removed() {
        // 6 kHz at 16 kHz is above 8 kHz-Nyquist (4 kHz) — must vanish, not alias.
        let x = sine(6_000.0, 16_000, 16_000, 0.5);
        let y = resample(&x, 16_000, 8_000).unwrap();
        assert_eq!(y.len(), 8_000);
        let rms = (y[1000..7000].iter().map(|v| v * v).sum::<f32>() / 6000.0).sqrt();
        assert!(rms < 0.005, "aliased energy rms {rms}");
    }
}
