// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Receive-side noise (`docs/design/data-pipeline.md`, stage 3): what the
//! receive path adds to already-decoded 8 kHz speech — mains and PSU hum
//! through a hotspot's audio cable, a radio's audio-stage hiss, alternator
//! whine, a cheap audio stage's colouring. Applied by the training loaders
//! to `deg8` on the fly, never to the target, seeded per example, so it
//! costs no capture and stacks with everything else. One implementation
//! for the pipeline and shard loaders, drawn from the shared `Rng`.

use std::f32::consts::PI;

use unamblify::Rng;
use unamblify::config::AugmentCfg;

use crate::chain::Biquad;
use crate::level::active_rms_dbfs;
use crate::{from_db, rms_dbfs};

/// The sample rate this module works at.
pub const RATE: u32 = 8_000;
/// Mains fundamentals and their doubles, Hz.
pub const HUM_FUNDAMENTALS: [f32; 4] = [50.0, 60.0, 100.0, 120.0];
/// Hum wobble amplitude, Hz, and its slow rate, Hz.
pub const HUM_WOBBLE_HZ: f32 = 0.5;
/// See [`HUM_WOBBLE_HZ`].
pub const HUM_WOBBLE_RATE_HZ: f32 = 0.2;
/// Corners of the colouring tilt, Hz.
pub const TILT_LO_HZ: f32 = 250.0;
/// See [`TILT_LO_HZ`].
pub const TILT_HI_HZ: f32 = 2_000.0;
/// Notch depth, dB, and Q.
pub const NOTCH_DB: f32 = -12.0;
/// See [`NOTCH_DB`].
pub const NOTCH_Q: f32 = 4.0;
/// Probability that colouring also soft-clips.
pub const SOFT_CLIP_P: f32 = 0.2;

/// Which kinds are on and how often; `From<&AugmentCfg>`.
#[derive(Debug, Clone, Copy, PartialEq)]
// Per-kind switches mirroring the config section, not a state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct RxCfg {
    /// Fraction of examples that get any receive-side noise.
    pub share: f32,
    /// Mains / PSU hum.
    pub hum: bool,
    /// White or pink noise.
    pub broadband: bool,
    /// Alternator whine.
    pub whine: bool,
    /// Tilt or notch, sometimes soft clipping.
    pub colouring: bool,
    /// A squelch tail at the end of the clip.
    pub squelch: bool,
}

impl From<&AugmentCfg> for RxCfg {
    fn from(c: &AugmentCfg) -> Self {
        Self {
            share: c.rx_share,
            hum: c.hum,
            broadband: c.broadband,
            whine: c.whine,
            colouring: c.colouring,
            squelch: c.squelch,
        }
    }
}

impl RxCfg {
    /// Whether anything can happen.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.share > 0.0
            && (self.hum || self.broadband || self.whine || self.colouring || self.squelch)
    }
}

/// Hum: `harmonics` at `1/n` amplitude on a fundamental that wobbles
/// slowly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hum {
    /// Fundamental, Hz.
    pub f0: f32,
    /// Harmonics (3–8).
    pub harmonics: u32,
    /// RMS level of the whole hum, dBFS (−50 to −30).
    pub level_dbfs: f32,
    /// Phase of the slow wobble, radians.
    pub wobble_phase: f32,
}

/// Broadband noise at an SNR against active speech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Broadband {
    /// Pink (1/f) rather than white.
    pub pink: bool,
    /// SNR, dB (15–35).
    pub snr_db: f32,
}

/// A tone sweeping across the crop with harmonics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Whine {
    /// Frequency at the start, Hz (200–600).
    pub f_start: f32,
    /// Frequency at the end, Hz (200–600).
    pub f_end: f32,
    /// Harmonics (2–4).
    pub harmonics: u32,
    /// RMS level, dBFS (−55 to −35).
    pub level_dbfs: f32,
}

/// A cheap audio stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Colouring {
    /// A tilt across the band, dB (±6), or…
    pub tilt_db: Option<f32>,
    /// …a single notch at this frequency, Hz.
    pub notch_hz: Option<f32>,
    /// Soft clipping at this level, dBFS (−6 to −1).
    pub soft_clip_dbfs: Option<f32>,
}

/// A squelch tail: a burst of noise at the very end of the clip that
/// decays away, the crash a receiver makes when the carrier drops at the
/// end of an over. Sharp onset, exponential decay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquelchTail {
    /// Length of the burst, ms (40–250).
    pub len_ms: f32,
    /// RMS level of the burst, dBFS (−24 to −6): loud, it is the open
    /// squelch before it closes.
    pub level_dbfs: f32,
    /// Pink rather than white.
    pub pink: bool,
    /// Exponential decay time constant, ms (20–120): how fast it fades.
    pub decay_ms: f32,
}

/// One example's draw: any combination of the four kinds.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RxRecipe {
    /// Hum.
    pub hum: Option<Hum>,
    /// Broadband noise.
    pub broadband: Option<Broadband>,
    /// Whine.
    pub whine: Option<Whine>,
    /// Colouring.
    pub colouring: Option<Colouring>,
    /// Squelch tail.
    pub squelch: Option<SquelchTail>,
}

impl RxRecipe {
    /// Nothing drawn.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hum.is_none()
            && self.broadband.is_none()
            && self.whine.is_none()
            && self.colouring.is_none()
            && self.squelch.is_none()
    }

    /// The eval's fixed recipe: 100 Hz hum at −40 dBFS plus white noise
    /// at 25 dB SNR.
    #[must_use]
    pub const fn fixed() -> Self {
        Self {
            hum: Some(Hum {
                f0: 100.0,
                harmonics: 5,
                level_dbfs: -40.0,
                wobble_phase: 0.0,
            }),
            broadband: Some(Broadband {
                pink: false,
                snr_db: 25.0,
            }),
            whine: None,
            colouring: None,
            squelch: None,
        }
    }
}

fn uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    lo + rng.next_f32() * (hi - lo)
}

/// Draw a recipe: `None` when the example is not in the share; else each
/// enabled kind with probability one half, at least one always.
#[must_use]
pub fn draw(cfg: &RxCfg, rng: &mut Rng) -> Option<RxRecipe> {
    if !cfg.enabled() || !rng.chance(cfg.share) {
        return None;
    }
    for _ in 0..16 {
        let r = RxRecipe {
            hum: (cfg.hum && rng.chance(0.5)).then(|| Hum {
                f0: HUM_FUNDAMENTALS[rng.below(HUM_FUNDAMENTALS.len())],
                harmonics: u32::try_from(rng.range(3, 8)).unwrap_or(3),
                level_dbfs: uniform(rng, -50.0, -30.0),
                wobble_phase: uniform(rng, 0.0, 2.0 * PI),
            }),
            broadband: (cfg.broadband && rng.chance(0.5)).then(|| Broadband {
                pink: rng.chance(0.5),
                snr_db: uniform(rng, 15.0, 35.0),
            }),
            whine: (cfg.whine && rng.chance(0.5)).then(|| Whine {
                f_start: uniform(rng, 200.0, 600.0),
                f_end: uniform(rng, 200.0, 600.0),
                harmonics: u32::try_from(rng.range(2, 4)).unwrap_or(2),
                level_dbfs: uniform(rng, -55.0, -35.0),
            }),
            colouring: (cfg.colouring && rng.chance(0.5)).then(|| {
                let tilt = rng.chance(0.5);
                Colouring {
                    tilt_db: tilt.then(|| uniform(rng, -6.0, 6.0)),
                    notch_hz: (!tilt).then(|| 300.0 * 10f32.powf(rng.next_f32())),
                    soft_clip_dbfs: rng.chance(SOFT_CLIP_P).then(|| uniform(rng, -6.0, -1.0)),
                }
            }),
            squelch: (cfg.squelch && rng.chance(0.5)).then(|| SquelchTail {
                len_ms: uniform(rng, 40.0, 250.0),
                level_dbfs: uniform(rng, -24.0, -6.0),
                pink: rng.chance(0.5),
                decay_ms: uniform(rng, 20.0, 120.0),
            }),
        };
        if !r.is_empty() {
            return Some(r);
        }
    }
    None
}

/// `len` samples of hum at [`RATE`], RMS at `level_dbfs`.
#[must_use]
pub fn hum(h: &Hum, len: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let f =
                h.f0 + HUM_WOBBLE_HZ * (2.0 * PI * HUM_WOBBLE_RATE_HZ * t + h.wobble_phase).sin();
            phase += 2.0 * PI * f / RATE as f32;
            (1..=h.harmonics.max(1))
                .map(|n| (n as f32 * phase).sin() / n as f32)
                .sum::<f32>()
        })
        .collect();
    scale_to_dbfs(&mut out, h.level_dbfs);
    out
}

/// `len` samples of white or pink noise, unit-ish RMS, from `rng`.
#[must_use]
pub fn broadband(pink: bool, len: usize, rng: &mut Rng) -> Vec<f32> {
    if !pink {
        return (0..len).map(|_| rng.normal()).collect();
    }
    // Paul Kellet's three-pole pink filter (the "economy" version).
    let (mut b0, mut b1, mut b2) = (0.0f32, 0.0f32, 0.0f32);
    (0..len)
        .map(|_| {
            let w = rng.normal();
            b0 = 0.997_65 * b0 + w * 0.099_046_0;
            b1 = 0.963_00 * b1 + w * 0.296_514;
            b2 = 0.570_00 * b2 + w * 1.052_652;
            (b0 + b1 + b2 + w * 0.184_8) * 0.3
        })
        .collect()
}

/// `len` samples of whine sweeping `f_start` → `f_end`, RMS at the level.
#[must_use]
pub fn whine(w: &Whine, len: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out: Vec<f32> = (0..len)
        .map(|i| {
            let x = if len > 1 {
                i as f32 / (len - 1) as f32
            } else {
                0.0
            };
            let f = w.f_start + (w.f_end - w.f_start) * x;
            phase += 2.0 * PI * f / RATE as f32;
            (1..=w.harmonics.max(1))
                .map(|n| (n as f32 * phase).sin() / n as f32)
                .sum::<f32>()
        })
        .collect();
    scale_to_dbfs(&mut out, w.level_dbfs);
    out
}

/// Colour `x` in place: the tilt or notch, then the soft clip.
pub fn colour(x: &mut [f32], c: &Colouring) {
    if let Some(t) = c.tilt_db {
        Biquad::low_shelf(RATE, TILT_LO_HZ, -t / 2.0).run(x);
        Biquad::high_shelf(RATE, TILT_HI_HZ, t / 2.0).run(x);
    }
    if let Some(f) = c.notch_hz {
        Biquad::peaking(RATE, f.clamp(50.0, 3_800.0), NOTCH_DB, NOTCH_Q).run(x);
    }
    if let Some(db) = c.soft_clip_dbfs {
        let l = from_db(db);
        for v in x {
            *v = l * (*v / l).tanh();
        }
    }
}

/// A squelch tail placed at the end of a `total_len`-sample buffer: the
/// last `len_ms` are broadband noise with a sharp onset and exponential
/// decay, RMS at `level_dbfs`; everything before is zero.
#[must_use]
pub fn squelch_tail(s: &SquelchTail, total_len: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = vec![0.0f32; total_len];
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let burst = ((s.len_ms / 1_000.0) * RATE as f32).round() as usize;
    let burst = burst.min(total_len);
    if burst == 0 {
        return out;
    }
    let mut noise = broadband(s.pink, burst, rng);
    let tau = (s.decay_ms.max(1.0) / 1_000.0) * RATE as f32;
    for (i, v) in noise.iter_mut().enumerate() {
        *v *= (-(i as f32) / tau).exp();
    }
    scale_to_dbfs(&mut noise, s.level_dbfs);
    for (o, x) in out[total_len - burst..].iter_mut().zip(noise) {
        *o = x;
    }
    out
}

fn scale_to_dbfs(x: &mut [f32], level_dbfs: f32) {
    let g = from_db(level_dbfs - rms_dbfs(x));
    for v in x {
        *v *= g;
    }
}

/// Apply `recipe` to `deg8` (8 kHz) in place: hum and whine at their
/// absolute levels, broadband at its SNR against the active speech of
/// `deg8`, then the colouring over everything. Broadband noise is drawn
/// from `rng`.
pub fn apply(deg8: &mut [f32], recipe: &RxRecipe, rng: &mut Rng) {
    let n = deg8.len();
    if let Some(h) = &recipe.hum {
        for (v, s) in deg8.iter_mut().zip(hum(h, n)) {
            *v += s;
        }
    }
    if let Some(w) = &recipe.whine {
        for (v, s) in deg8.iter_mut().zip(whine(w, n)) {
            *v += s;
        }
    }
    if let Some(b) = &recipe.broadband {
        let noise = broadband(b.pink, n, rng);
        // Against the speech, silent input gets none.
        if let Some(s) = active_rms_dbfs(deg8, RATE) {
            let g = from_db(s - b.snr_db - rms_dbfs(&noise));
            for (v, x) in deg8.iter_mut().zip(noise) {
                *v += x * g;
            }
        }
    }
    if let Some(c) = &recipe.colouring {
        colour(deg8, c);
    }
    if let Some(s) = &recipe.squelch {
        for (v, x) in deg8.iter_mut().zip(squelch_tail(s, n, rng)) {
            *v += x;
        }
    }
}

/// Draw and apply in one go; returns the recipe used, `None` when the
/// example was left alone.
pub fn apply_drawn(deg8: &mut [f32], cfg: &RxCfg, rng: &mut Rng) -> Option<RxRecipe> {
    let recipe = draw(cfg, rng)?;
    apply(deg8, &recipe, rng);
    Some(recipe)
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::spectral::stft;
    use crate::testutil::sine;

    fn all() -> RxCfg {
        RxCfg {
            share: 1.0,
            hum: true,
            broadband: true,
            whine: true,
            colouring: true,
            squelch: true,
        }
    }

    fn peak_bins(x: &[f32], n_fft: usize) -> Vec<usize> {
        let s = stft(x, n_fft, n_fft);
        let mid = &s[s.len() / 2];
        let max = mid.iter().copied().fold(0.0f32, f32::max);
        (0..mid.len())
            .filter(|&k| mid[k] > max * 0.05)
            .filter(|&k| k > 0 && mid[k] > mid[k - 1] && mid[k] >= mid[k + 1])
            .collect()
    }

    #[test]
    fn hum_has_its_harmonics_at_the_right_bins_and_level() {
        let h = Hum {
            f0: 100.0,
            harmonics: 4,
            level_dbfs: -40.0,
            wobble_phase: 0.0,
        };
        let x = hum(&h, 16_000);
        assert!((rms_dbfs(&x) + 40.0).abs() < 0.05, "{}", rms_dbfs(&x));
        // 8192-point STFT at 8 kHz: 0.977 Hz per bin; the ±0.5 Hz wobble
        // moves the nth harmonic by up to ±0.5·n Hz.
        let bins = peak_bins(&x, 8_192);
        for n in 1..=4usize {
            let w = (100.0 * n as f32 * 8_192.0 / 8_000.0).round() as i64;
            assert!(
                bins.iter().any(|b| (*b as i64 - w).abs() <= n as i64 + 1),
                "harmonic {n} (bin {w}) missing from {bins:?}"
            );
        }
        assert!(bins.len() <= 8, "{bins:?}");
        let w = Whine {
            f_start: 300.0,
            f_end: 300.0,
            harmonics: 2,
            level_dbfs: -45.0,
        };
        let y = whine(&w, 8_000);
        assert!((rms_dbfs(&y) + 45.0).abs() < 0.05);
        let bins = peak_bins(&y, 8_000);
        let near = |want: i64| bins.iter().any(|b| (*b as i64 - want).abs() <= 2);
        assert!(near(300) && near(600) && !near(450), "{bins:?}");
    }

    #[test]
    fn broadband_hits_its_snr_and_pink_is_darker() {
        let mut speech = vec![0.0f32; 4_000];
        speech.extend(sine(220.0, 8_000, 16_000, 0.1));
        let r = RxRecipe {
            broadband: Some(Broadband {
                pink: false,
                snr_db: 25.0,
            }),
            ..RxRecipe::default()
        };
        let mut x = speech.clone();
        apply(&mut x, &r, &mut Rng::new(4));
        let noise: Vec<f32> = x.iter().zip(&speech).map(|(a, b)| a - b).collect();
        let snr = active_rms_dbfs(&speech, 8_000).unwrap() - rms_dbfs(&noise);
        assert!((snr - 25.0).abs() < 0.5, "snr {snr}");
        // Silence gets no broadband noise (nothing to measure against).
        let mut silent = vec![0.0f32; 8_000];
        apply(&mut silent, &r, &mut Rng::new(4));
        assert!(silent.iter().all(|&v| v == 0.0));
        let white = broadband(false, 32_000, &mut Rng::new(1));
        let pink = broadband(true, 32_000, &mut Rng::new(1));
        let hi = |x: &[f32]| {
            let s = stft(x, 512, 512);
            let mid = &s[s.len() / 2];
            mid[200..256].iter().sum::<f32>() / mid[8..64].iter().sum::<f32>()
        };
        assert!(
            hi(&pink) < hi(&white) * 0.5,
            "pink {} white {}",
            hi(&pink),
            hi(&white)
        );
    }

    #[test]
    fn colouring_tilts_notches_and_soft_clips() {
        let mut lo = sine(100.0, 8_000, 8_000, 0.1);
        let mut hi = sine(3_000.0, 8_000, 8_000, 0.1);
        let c = Colouring {
            tilt_db: Some(6.0),
            notch_hz: None,
            soft_clip_dbfs: None,
        };
        colour(&mut lo, &c);
        colour(&mut hi, &c);
        let d = rms_dbfs(&hi[4_000..]) - rms_dbfs(&lo[4_000..]);
        assert!((d - 6.0).abs() < 1.0, "tilt {d}");
        let mut at = sine(1_000.0, 8_000, 8_000, 0.1);
        let mut off = sine(250.0, 8_000, 8_000, 0.1);
        let n = Colouring {
            tilt_db: None,
            notch_hz: Some(1_000.0),
            soft_clip_dbfs: None,
        };
        colour(&mut at, &n);
        colour(&mut off, &n);
        assert!(
            rms_dbfs(&at[4_000..]) < -20.0 - 10.0,
            "notched {}",
            rms_dbfs(&at[4_000..])
        );
        assert!((rms_dbfs(&off[4_000..]) + 23.0).abs() < 1.0);
        let mut hot = sine(300.0, 8_000, 8_000, 0.9);
        colour(
            &mut hot,
            &Colouring {
                tilt_db: None,
                notch_hz: None,
                soft_clip_dbfs: Some(-6.0),
            },
        );
        let peak = hot.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            peak <= from_db(-6.0) + 1e-6 && peak > from_db(-6.0) * 0.9,
            "{peak}"
        );
    }

    #[test]
    fn draws_are_seeded_respect_the_share_and_the_toggles() {
        let cfg = all();
        let mut a = Rng::new(9);
        let mut b = Rng::new(9);
        for _ in 0..20 {
            assert_eq!(draw(&cfg, &mut a), draw(&cfg, &mut b));
        }
        let none = RxCfg {
            share: 0.0,
            ..all()
        };
        assert_eq!(draw(&none, &mut Rng::new(1)), None);
        assert!(!none.enabled());
        let off = RxCfg {
            hum: false,
            broadband: false,
            whine: false,
            colouring: false,
            squelch: false,
            share: 1.0,
        };
        assert!(!off.enabled());
        let hum_only = RxCfg {
            broadband: false,
            whine: false,
            colouring: false,
            squelch: false,
            ..all()
        };
        let mut rng = Rng::new(2);
        for _ in 0..50 {
            let r = draw(&hum_only, &mut rng).unwrap();
            assert!(r.hum.is_some() && r.broadband.is_none() && r.whine.is_none());
        }
        let third = RxCfg {
            share: 0.3,
            ..all()
        };
        let mut rng = Rng::new(3);
        let n = (0..2_000)
            .filter(|_| draw(&third, &mut rng).is_some())
            .count();
        assert!((500..700).contains(&n), "{n}");
        // The whole thing: seeded, in range, finite; leaves the input alone
        // when nothing is drawn.
        let mut speech = vec![0.0f32; 2_000];
        speech.extend(sine(180.0, 8_000, 6_000, 0.1));
        let mut x = speech.clone();
        let mut y = speech.clone();
        let ra = apply_drawn(&mut x, &cfg, &mut Rng::new(11));
        let rb = apply_drawn(&mut y, &cfg, &mut Rng::new(11));
        assert_eq!(ra, rb);
        assert!(ra.is_some());
        assert_eq!(x, y);
        assert_ne!(x, speech);
        assert!(x.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let mut z = speech.clone();
        assert_eq!(apply_drawn(&mut z, &none, &mut Rng::new(11)), None);
        assert_eq!(z, speech);
        let cfg_from: RxCfg = (&AugmentCfg::default()).into();
        assert_eq!(cfg_from.share, 0.3);
        assert!(cfg_from.hum && cfg_from.colouring);
    }

    #[test]
    fn the_fixed_recipe_is_a_hum_and_white_noise() {
        let f = RxRecipe::fixed();
        assert!(!f.is_empty());
        let mut silent = vec![0.0f32; 8_000];
        apply(&mut silent, &f, &mut Rng::new(1));
        // Only the hum lands on silence: −40 dBFS at 100 Hz harmonics.
        assert!(
            (rms_dbfs(&silent) + 40.0).abs() < 0.1,
            "{}",
            rms_dbfs(&silent)
        );
        let bins = peak_bins(&silent, 8_000);
        let near = |want: i64, tol: i64| bins.iter().any(|b| (*b as i64 - want).abs() <= tol);
        assert!(near(100, 1) && near(200, 3) && near(300, 4), "{bins:?}");
        assert!(!near(150, 10), "nothing between the harmonics: {bins:?}");
        let mut speech = sine(220.0, 8_000, 8_000, 0.1);
        let before = speech.clone();
        apply(&mut speech, &f, &mut Rng::new(1));
        let added: Vec<f32> = speech.iter().zip(&before).map(|(a, b)| a - b).collect();
        let snr = rms_dbfs(&before) - rms_dbfs(&added);
        // Hum at −40 dBFS plus white 25 dB below the −23 dBFS sine
        // (−48 dBFS): together −39.4 dBFS, so ≈ 16.4 dB against the sine.
        assert!((15.5..17.5).contains(&snr), "{snr}");
    }

    #[test]
    fn squelch_tail_is_a_decaying_burst_at_the_end() {
        let s = SquelchTail {
            len_ms: 100.0,
            level_dbfs: -12.0,
            pink: false,
            decay_ms: 40.0,
        };
        let n = 8_000;
        let out = squelch_tail(&s, n, &mut Rng::new(1));
        let burst = 800; // 100 ms at 8 kHz
        assert!(
            out[..n - burst].iter().all(|&x| x == 0.0),
            "silent before the burst"
        );
        assert!(
            out[n - burst..].iter().any(|&x| x.abs() > 1e-4),
            "has energy"
        );
        let q = burst / 4;
        let e0: f32 = out[n - burst..n - burst + q].iter().map(|x| x * x).sum();
        let e1: f32 = out[n - q..].iter().map(|x| x * x).sum();
        assert!(e0 > e1, "decays: {e0} vs {e1}");
    }
}
