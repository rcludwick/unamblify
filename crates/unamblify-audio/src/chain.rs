// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The prepare-time augmentations of `docs/design/data-pipeline.md`
//! (stage 3): mixing a noise clip into clean speech at a chosen SNR, and
//! the synthetic "talking too close to the mic" transmit chain —
//! proximity shelf, mic tilt and resonance, plosive pops, clipping, AGC
//! into a hard limiter — driven by the parameters the core crate's
//! [`ChainParams`] records in the manifest. Everything here is a plain
//! function on slices; the parameters are drawn elsewhere.

use std::f32::consts::PI;

use unamblify::Rng;
use unamblify::aug::{AgcAug, ChainParams, ClipAug, ClipKind, MicAug, PopsAug};

use crate::level::{ACTIVE_THRESHOLD_DBFS, active_rms_dbfs, rms_dbfs};
use crate::{from_db, to_db};

/// The limiter ceiling, dBFS.
pub const LIMITER_DBFS: f32 = -1.0;
/// Where the proximity shelf turns over, Hz.
pub const PROXIMITY_HZ: f32 = 200.0;
/// The tilt's low and high shelf corners, Hz.
pub const TILT_LO_HZ: f32 = 250.0;
/// See [`TILT_LO_HZ`].
pub const TILT_HI_HZ: f32 = 2_500.0;
/// The AGC's target envelope (peak-ish), linear.
pub const AGC_TARGET: f32 = 0.35;
/// The AGC's gain ceiling, dB.
pub const AGC_MAX_GAIN_DB: f32 = 30.0;
/// Pop decay time constant, seconds.
pub const POP_TAU_S: f32 = 0.025;
/// Pop length, seconds.
pub const POP_LEN_S: f32 = 0.08;
/// Minimum spacing between detected onsets, seconds.
pub const ONSET_REFRACTORY_S: f32 = 0.1;
/// An onset is a 10 ms frame this much louder than the one before it, dB.
pub const ONSET_RISE_DB: f32 = 8.0;

/// SNR as this crate measures it: the active-speech RMS of `speech` over
/// the RMS of `noise`, dB. `None` when the speech has no active frame or
/// the noise is silent.
#[must_use]
pub fn measured_snr_db(speech: &[f32], noise: &[f32], rate: u32) -> Option<f32> {
    let s = active_rms_dbfs(speech, rate)?;
    let n = rms_dbfs(noise);
    (n > -150.0).then_some(s - n)
}

/// Mix `noise` into `speech` (same length; a longer noise is truncated, a
/// shorter one is zero-extended) so that [`measured_snr_db`] of the pair is
/// `snr_db`. Returns the mix and the gain applied to the noise, dB. A
/// silent noise or silent speech is mixed at unity gain.
#[must_use]
pub fn mix_at_snr(speech: &[f32], noise: &[f32], rate: u32, snr_db: f32) -> (Vec<f32>, f32) {
    let gain_db = measured_snr_db(speech, noise, rate).map_or(0.0, |snr| snr - snr_db);
    let g = from_db(gain_db);
    let mix = speech
        .iter()
        .enumerate()
        .map(|(i, &s)| s + noise.get(i).copied().unwrap_or(0.0) * g)
        .collect();
    (mix, gain_db)
}

/// A second-order section (RBJ cookbook), transposed direct form II.
#[derive(Debug, Clone, PartialEq)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn normalised(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Low shelf: `gain_db` below `f0`, slope 1.
    #[must_use]
    pub fn low_shelf(rate: u32, f0: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / rate as f32;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / 2.0 * 2f32.sqrt();
        let sa = 2.0 * a.sqrt() * alpha;
        Self::normalised(
            a * ((a + 1.0) - (a - 1.0) * cs + sa),
            2.0 * a * ((a - 1.0) - (a + 1.0) * cs),
            a * ((a + 1.0) - (a - 1.0) * cs - sa),
            (a + 1.0) + (a - 1.0) * cs + sa,
            -2.0 * ((a - 1.0) + (a + 1.0) * cs),
            (a + 1.0) + (a - 1.0) * cs - sa,
        )
    }

    /// High shelf: `gain_db` above `f0`, slope 1.
    #[must_use]
    pub fn high_shelf(rate: u32, f0: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / rate as f32;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / 2.0 * 2f32.sqrt();
        let sa = 2.0 * a.sqrt() * alpha;
        Self::normalised(
            a * ((a + 1.0) + (a - 1.0) * cs + sa),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * cs),
            a * ((a + 1.0) + (a - 1.0) * cs - sa),
            (a + 1.0) - (a - 1.0) * cs + sa,
            2.0 * ((a - 1.0) - (a + 1.0) * cs),
            (a + 1.0) - (a - 1.0) * cs - sa,
        )
    }

    /// Peaking EQ: `gain_db` at `f0` with quality `q`.
    #[must_use]
    pub fn peaking(rate: u32, f0: f32, gain_db: f32, q: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / rate as f32;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * q.max(0.05));
        Self::normalised(
            1.0 + alpha * a,
            -2.0 * cs,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cs,
            1.0 - alpha / a,
        )
    }

    /// One sample.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Filter `xs` in place.
    pub fn run(&mut self, xs: &mut [f32]) {
        for v in xs {
            *v = self.process(*v);
        }
    }
}

/// Sample indices where speech starts: 10 ms frames that jump by
/// [`ONSET_RISE_DB`] over the frame before, above the active threshold,
/// at least [`ONSET_REFRACTORY_S`] apart.
#[must_use]
pub fn detect_onsets(x: &[f32], rate: u32) -> Vec<usize> {
    let w = ((rate / 100) as usize).max(1);
    let energies: Vec<f32> = x
        .chunks(w)
        .map(|f| to_db((f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt()))
        .collect();
    let refractory = (ONSET_REFRACTORY_S * rate as f32) as usize;
    let mut out = Vec::new();
    let mut last: Option<usize> = None;
    for i in 1..energies.len() {
        let at = i * w;
        if energies[i] > ACTIVE_THRESHOLD_DBFS
            && energies[i] - energies[i - 1] >= ONSET_RISE_DB
            && last.is_none_or(|l| at - l >= refractory)
        {
            out.push(at);
            last = Some(at);
        }
    }
    out
}

/// Add a decaying low-frequency thump at each of `at`.
pub fn add_pops(x: &mut [f32], rate: u32, at: &[usize], freq_hz: f32, level_dbfs: f32) {
    let amp = from_db(level_dbfs);
    let n = (POP_LEN_S * rate as f32) as usize;
    for &s in at {
        for i in 0..n {
            let Some(v) = x.get_mut(s + i) else { break };
            let t = i as f32 / rate as f32;
            *v += amp * (-t / POP_TAU_S).exp() * (2.0 * PI * freq_hz * t).sin();
        }
    }
}

/// Clip in place: the signal is driven so that its peak sits `drive_db`
/// above a ceiling of 1.0 (a mic amp saturating), hard or soft. The drive
/// is relative to the signal's own peak, so 6 dB always clips and 24 dB
/// always flattens, whatever the input level; the level is normalised
/// again after the chain.
pub fn clip(x: &mut [f32], c: ClipAug) {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let g = from_db(c.drive_db) / peak.max(1e-6);
    match c.kind {
        ClipKind::Hard => {
            for v in x {
                *v = (*v * g).clamp(-1.0, 1.0);
            }
        }
        ClipKind::Soft => {
            for v in x {
                *v = (*v * g).tanh();
            }
        }
    }
}

/// Gain riding toward [`AGC_TARGET`] with the given attack and release,
/// then a hard limiter at [`LIMITER_DBFS`]. In place.
pub fn agc_limiter(x: &mut [f32], rate: u32, a: AgcAug) {
    let coef = |ms: f32| (-1.0 / (ms.max(0.1) * 1e-3 * rate as f32)).exp();
    let (att, rel) = (coef(a.attack_ms), coef(a.release_ms));
    let max_gain = from_db(AGC_MAX_GAIN_DB);
    let ceiling = from_db(LIMITER_DBFS);
    let mut env = 0.0f32;
    for v in x {
        let mag = v.abs();
        let c = if mag > env { att } else { rel };
        env = c * env + (1.0 - c) * mag;
        let g = (AGC_TARGET / env.max(1e-4)).min(max_gain);
        *v = (*v * g).clamp(-ceiling, ceiling);
    }
}

/// The mic response: a tilt (a low shelf cut and a high shelf boost of
/// half the tilt each) and one resonance. In place.
pub fn mic_response(x: &mut [f32], rate: u32, m: MicAug) {
    Biquad::low_shelf(rate, TILT_LO_HZ, -m.tilt_db / 2.0).run(x);
    Biquad::high_shelf(rate, TILT_HI_HZ, m.tilt_db / 2.0).run(x);
    Biquad::peaking(rate, m.resonance_hz, m.resonance_db, m.resonance_q).run(x);
}

/// Run the chain over `x` at `rate` in the order the parameters document:
/// proximity shelf → mic response → pops (which onsets get one is drawn
/// from `rng` with the recorded share) → clipping → AGC + limiter. The
/// level is *not* normalised here; the caller does that after.
#[must_use]
pub fn apply_chain(x: &[f32], rate: u32, p: &ChainParams, rng: &mut Rng) -> Vec<f32> {
    let mut y = x.to_vec();
    if let Some(db) = p.shelf_db {
        Biquad::low_shelf(rate, PROXIMITY_HZ, db).run(&mut y);
    }
    if let Some(m) = p.mic {
        mic_response(&mut y, rate, m);
    }
    if let Some(PopsAug {
        share,
        freq_hz,
        level_dbfs,
    }) = p.pops
    {
        let chosen: Vec<usize> = detect_onsets(&y, rate)
            .into_iter()
            .filter(|_| rng.chance(share))
            .collect();
        add_pops(&mut y, rate, &chosen, freq_hz, level_dbfs);
    }
    if let Some(c) = p.clip {
        clip(&mut y, c);
    }
    if let Some(a) = p.agc {
        agc_limiter(&mut y, rate, a);
    }
    y
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::testutil::{noise, sine};

    /// Speech-like: half a second of silence, a tone, silence.
    fn speech(rate: u32) -> Vec<f32> {
        let mut x = vec![0.0f32; rate as usize / 2];
        x.extend(sine(220.0, rate, rate as usize, 0.1));
        x.extend(vec![0.0f32; rate as usize / 4]);
        x
    }

    #[test]
    fn mix_hits_the_requested_snr_within_a_tenth_of_a_db() {
        let rate = 16_000;
        let s = speech(rate);
        let n = noise(s.len(), 0.3, 4);
        for snr in [0.0f32, 7.5, 20.0] {
            let (mix, gain_db) = mix_at_snr(&s, &n, rate, snr);
            assert_eq!(mix.len(), s.len());
            let scaled: Vec<f32> = mix.iter().zip(&s).map(|(m, s)| m - s).collect();
            let got = measured_snr_db(&s, &scaled, rate).unwrap();
            assert!((got - snr).abs() < 0.1, "snr {snr}: got {got}");
            assert!(gain_db.is_finite());
        }
        // A shorter noise is zero-extended; silent noise mixes at unity.
        let (mix, _) = mix_at_snr(&s, &n[..100], rate, 10.0);
        assert_eq!(mix[5000], s[5000]);
        let (mix, g) = mix_at_snr(&s, &[0.0; 8], rate, 10.0);
        assert_eq!(g, 0.0);
        assert_eq!(mix, s);
        assert_eq!(measured_snr_db(&[0.0; 1600], &n, rate), None);
    }

    #[test]
    fn hard_clipping_clips_and_soft_clipping_rounds() {
        let x = sine(300.0, 16_000, 1600, 0.25); // −12 dBFS peak
        let mut hard = x.clone();
        clip(
            &mut hard,
            ClipAug {
                kind: ClipKind::Hard,
                drive_db: 12.0,
            },
        );
        let peak = hard.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert_eq!(peak, 1.0);
        let at_ceiling = hard.iter().filter(|v| v.abs() == 1.0).count();
        // 12 dB over the ceiling: |sin| > 1/3.98 → about 84 % of samples.
        assert!(
            at_ceiling > x.len() * 3 / 4,
            "{at_ceiling} of {} at the ceiling",
            x.len()
        );
        // The drive is relative to the peak: a quiet input clips as much.
        let mut quiet: Vec<f32> = x.iter().map(|v| v * 0.01).collect();
        clip(
            &mut quiet,
            ClipAug {
                kind: ClipKind::Hard,
                drive_db: 12.0,
            },
        );
        assert_eq!(quiet.iter().filter(|v| v.abs() == 1.0).count(), at_ceiling);
        let mut soft = x;
        clip(
            &mut soft,
            ClipAug {
                kind: ClipKind::Soft,
                drive_db: 18.0,
            },
        );
        let peak = soft.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 1.0 && peak > 0.9, "{peak}");
        assert!(soft.iter().all(|v| v.abs() < 1.0));
    }

    /// Steady-state gain of a filter at `f`, dB (measured on the middle of
    /// a long sine).
    fn gain_db_at(mut filt: Biquad, rate: u32, f: f32) -> f32 {
        let mut x = sine(f, rate, rate as usize, 0.1);
        filt.run(&mut x);
        rms_dbfs(&x[rate as usize / 2..]) - rms_dbfs(&sine(f, rate, rate as usize / 2, 0.1))
    }

    #[test]
    fn shelves_and_peaks_land_where_they_say() {
        let rate = 16_000;
        let lo = gain_db_at(Biquad::low_shelf(rate, 200.0, 8.0), rate, 40.0);
        assert!((lo - 8.0).abs() < 0.3, "low shelf at 40 Hz: {lo}");
        let hi = gain_db_at(Biquad::low_shelf(rate, 200.0, 8.0), rate, 3_000.0);
        assert!(hi.abs() < 0.3, "low shelf at 3 kHz: {hi}");
        let hs = gain_db_at(Biquad::high_shelf(rate, 2_500.0, -6.0), rate, 6_000.0);
        assert!((hs + 6.0).abs() < 0.5, "high shelf at 6 kHz: {hs}");
        let hs_lo = gain_db_at(Biquad::high_shelf(rate, 2_500.0, -6.0), rate, 100.0);
        assert!(hs_lo.abs() < 0.3, "high shelf at 100 Hz: {hs_lo}");
        let pk = gain_db_at(Biquad::peaking(rate, 1_000.0, 6.0, 3.0), rate, 1_000.0);
        assert!((pk - 6.0).abs() < 0.3, "peak at f0: {pk}");
        let off = gain_db_at(Biquad::peaking(rate, 1_000.0, 6.0, 3.0), rate, 250.0);
        assert!(off.abs() < 0.5, "peak two octaves down: {off}");
        // A tilt brightens: more gain at 4 kHz than at 100 Hz.
        let mut bright = sine(4_000.0, rate, rate as usize, 0.1);
        let mut dull = sine(100.0, rate, rate as usize, 0.1);
        let m = MicAug {
            tilt_db: 6.0,
            resonance_hz: 1_500.0,
            resonance_db: 0.0,
            resonance_q: 3.0,
        };
        mic_response(&mut bright, rate, m);
        mic_response(&mut dull, rate, m);
        let d = rms_dbfs(&bright[8_000..]) - rms_dbfs(&dull[8_000..]);
        assert!((d - 6.0).abs() < 0.8, "tilt {d}");
    }

    #[test]
    fn onsets_are_found_and_pops_add_a_thump() {
        let rate = 16_000;
        let mut x = vec![0.0f32; 4_000];
        x.extend(sine(300.0, rate, 4_000, 0.2));
        x.extend(vec![0.0f32; 4_000]);
        x.extend(sine(300.0, rate, 4_000, 0.2));
        let on = detect_onsets(&x, rate);
        assert_eq!(on, vec![4_000, 12_000], "{on:?}");
        assert!(detect_onsets(&sine(300.0, rate, 8_000, 0.2), rate).is_empty());
        let before = x.clone();
        add_pops(&mut x, rate, &on, 80.0, -6.0);
        let diff: Vec<f32> = x.iter().zip(&before).map(|(a, b)| a - b).collect();
        let peak = diff.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - from_db(-6.0)).abs() < 0.1, "{peak}");
        assert!(
            diff[..4_000].iter().all(|&v| v == 0.0),
            "nothing before the onset"
        );
        // 80 Hz: the STFT peak of the added thump is in the lowest bins.
        let mags = crate::spectral::stft(&diff[4_000..5_024], 1024, 1024);
        let peak_bin = mags[1]
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |a, (i, &v)| if v > a.1 { (i, v) } else { a })
            .0;
        assert!((3..=8).contains(&peak_bin), "bin {peak_bin}");
    }

    #[test]
    fn agc_evens_out_levels_and_the_limiter_holds_the_ceiling() {
        let rate = 16_000;
        let mut x = sine(300.0, rate, 16_000, 0.02); // quiet
        x.extend(sine(300.0, rate, 16_000, 0.5)); // loud
        agc_limiter(
            &mut x,
            rate,
            AgcAug {
                attack_ms: 10.0,
                release_ms: 200.0,
            },
        );
        let ceiling = from_db(LIMITER_DBFS);
        assert!(x.iter().all(|v| v.abs() <= ceiling + 1e-6));
        let quiet = rms_dbfs(&x[8_000..16_000]);
        let loud = rms_dbfs(&x[24_000..]);
        assert!((quiet - loud).abs() < 6.0, "quiet {quiet} loud {loud}");
        assert!(quiet > -20.0, "the quiet part was brought up: {quiet}");
    }

    #[test]
    fn chain_is_deterministic_and_identity_when_empty() {
        let rate = 16_000;
        let s = speech(rate);
        let p = ChainParams {
            shelf_db: Some(6.0),
            mic: Some(MicAug {
                tilt_db: -3.0,
                resonance_hz: 1_200.0,
                resonance_db: 5.0,
                resonance_q: 3.0,
            }),
            pops: Some(PopsAug {
                share: 1.0,
                freq_hz: 90.0,
                level_dbfs: -6.0,
            }),
            clip: Some(ClipAug {
                kind: ClipKind::Soft,
                drive_db: 12.0,
            }),
            agc: Some(AgcAug {
                attack_ms: 20.0,
                release_ms: 300.0,
            }),
        };
        let a = apply_chain(&s, rate, &p, &mut Rng::new(3));
        let b = apply_chain(&s, rate, &p, &mut Rng::new(3));
        assert_eq!(a, b);
        assert_eq!(a.len(), s.len());
        assert!(a.iter().all(|v| v.is_finite() && v.abs() < 1.0));
        assert_ne!(a, s);
        assert_eq!(
            apply_chain(&s, rate, &ChainParams::default(), &mut Rng::new(3)),
            s
        );
        let fixed = apply_chain(&s, rate, &ChainParams::fixed_overdrive(), &mut Rng::new(1));
        assert!(
            fixed.iter().any(|v| v.abs() == 1.0),
            "12 dB into the ceiling clips"
        );
    }
}
