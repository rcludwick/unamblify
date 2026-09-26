// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Losses and the edges rule (spec §6).
//!
//! * Multi-resolution STFT on `out16` vs `clean16`: `n_fft` 512 / 1024 /
//!   2048 at 16 kHz, hop `n_fft / 4`, per frame the mean over bins of the
//!   linear-magnitude L1 (normalised by the batch's mean target
//!   magnitude) plus the log-magnitude L1.
//! * Mel L1: 80 HTK mel bins on the 1024 / 256 STFT, natural-log energies.
//! * SI-SDR on the 8 kHz decimated path (`decimate(out16)` vs
//!   `decimate(clean16)`, both through the model's fixed half-band filter
//!   so they stay time-aligned), over the `mask == 1` region only,
//!   entering the total as `−sisdr_w · SI-SDR(dB)`.
//! * Edges: every per-frame spectral term is averaged with a per-sample
//!   weight `w = 1 + onset_w` on the first `onset_frames` (50 × 10 ms)
//!   of onset examples and `w = tail_w` where `mask == 0` on tail
//!   examples (the target there is silence, so the ordinary terms *are*
//!   the silence loss), `w = 1` elsewhere. A *babble penalty* — the mean
//!   log-energy of `out16` over the `mask == 0` frames, in dB above a
//!   −60 dBFS floor — enters as `babble_w · babble` (default 0.05, so a
//!   tail at −20 dBFS costs 2, comparable to the spectral terms).
//!
//! `total = stft + mel + sisdr_term + babble_w · babble`. The `onset` and
//! `tail` terms that go to `metrics.jsonl` are region diagnostics: the
//! unweighted mean of `stft + mel` over the onset region / over the
//! garbage-tail region (the latter plus the weighted babble penalty).
//! Their weights are already inside `stft` and `mel`, so they are not
//! added again.
//!
//! The HiFi-GAN-style discriminators (multi-period + multi-resolution)
//! and the LSGAN + feature-matching plumbing live here too, behind
//! `[train] gan = true`; this milestone unit-tests their shapes only.

use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

use crate::stft::{Stft, mel_filterbank};

/// Sample rate of the wideband target.
const RATE16: u32 = 16_000;
/// Top of the band the harmonic-to-trough term measures: what a
/// narrowband vocoder transmits.
const HNR_MAX_HZ: f64 = 4000.0;
/// Half-width of the harmonic and trough windows, in units of `f0`.
const HNR_HALF_WIDTH: f64 = 0.18;
/// Normalised autocorrelation peak a frame needs to count as voiced.
/// Lower than the 0.45 `unamblify_audio::hnr_db` uses, because this one
/// is computed from the *windowed* frame's power spectrum, whose
/// autocorrelation carries the window's own taper and reads lower for
/// the same speech.
const VOICED_PERIODICITY: f64 = 0.3;
/// Mel bins.
const N_MELS: i64 = 80;
/// Floor inside the logs.
const EPS: f64 = 1e-5;
/// Samples per 10 ms feature frame at 16 kHz.
const FRAME16: i64 = 160;

/// Loss weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossCfg {
    /// Extra weight on the onset region of onset examples (`[train] onset_w`).
    pub onset_w: f32,
    /// Weight on the silence target of tail examples (`[train] tail_w`).
    pub tail_w: f32,
    /// Weight of `−SI-SDR(dB)` in the total.
    pub sisdr_w: f32,
    /// Weight of the babble penalty (dB above −60 dBFS) in the total.
    pub babble_w: f32,
    /// Onset region length, 10 ms feature frames.
    pub onset_frames: i64,
    /// Weight of the periodicity term (`[train] periodicity_w`).
    pub periodicity_w: f32,
    /// Weight of the harmonic-to-trough term (`[train] hnr_w`).
    pub hnr_w: f32,
    /// Weight of the transient term (`[train] transient_w`).
    pub transient_w: f32,
}

impl Default for LossCfg {
    fn default() -> Self {
        Self {
            onset_w: 2.0,
            tail_w: 3.0,
            sisdr_w: 0.05,
            babble_w: 0.05,
            onset_frames: 50,
            periodicity_w: 0.0,
            hnr_w: 0.0,
            transient_w: 0.0,
        }
    }
}

/// What the loss needs from the model and the batch. Shapes: `out16`,
/// `clean16`, `mask16` are `[B, 1, N16]`; `out8`, `clean8` are
/// `[B, 1, N16 / 2]`; `onset`, `tail` are `[B]` bool.
#[derive(Debug)]
pub struct LossInputs<'a> {
    /// Model output, 16 kHz.
    pub out16: &'a Tensor,
    /// Target, 16 kHz.
    pub clean16: &'a Tensor,
    /// 1 where the target is real speech, 0 in the garbage tail.
    pub mask16: &'a Tensor,
    /// Model output decimated to 8 kHz.
    pub out8: &'a Tensor,
    /// Target decimated to 8 kHz.
    pub clean8: &'a Tensor,
    /// Onset example flags.
    pub onset: &'a Tensor,
    /// Tail example flags.
    pub tail: &'a Tensor,
}

/// The loss terms as scalar tensors (`total` carries the graph).
#[derive(Debug)]
pub struct LossTerms {
    /// What is back-propagated.
    pub total: Tensor,
    /// Weighted multi-resolution STFT term.
    pub stft: Tensor,
    /// Weighted mel term.
    pub mel: Tensor,
    /// `−sisdr_w · SI-SDR` (dB) on the 8 kHz path.
    pub sisdr: Tensor,
    /// Onset-region diagnostic.
    pub onset: Tensor,
    /// Tail-region diagnostic (+ babble).
    pub tail: Tensor,
    /// Babble penalty, dB above the floor (enters the total × `babble_w`).
    pub babble: Tensor,
    /// Signed periodicity error: mean cepstral peak prominence of the
    /// output minus that of the target. **Positive means the output is
    /// more periodic than natural speech** — buzzy. Enters the total as
    /// `periodicity_w × |this|`.
    ///
    /// Diagnostic by default: it is *gameable*, and was gamed — see
    /// [`Losses::hnr`].
    pub periodicity: Tensor,
    /// Signed harmonic-to-trough error, dB: how much further the
    /// output's harmonics stand above the valleys between them than the
    /// target's do. **Positive is buzzy.** Enters the total as
    /// `hnr_w × `[`hnr_l1`](Self::hnr_l1).
    pub hnr: Tensor,
    /// Mean of the *per-frame* absolute dB error — what the total
    /// actually uses, so frames that are too buzzy and frames that are
    /// too breathy both cost rather than cancelling.
    pub hnr_l1: Tensor,
    /// The transient term, dB: how far the 2–3.8 kHz envelope at 1 ms
    /// resolution, and its slope, differ from the target's. Enters the
    /// total as `transient_w ×` this. See [`Losses::transient`].
    pub transient: Tensor,
}

/// The loss terms as numbers, for the metrics writer.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
pub struct LossValues {
    /// `loss/total`.
    pub total: f64,
    /// `loss/stft`.
    pub stft: f64,
    /// `loss/mel`.
    pub mel: f64,
    /// `loss/sisdr`.
    pub sisdr: f64,
    /// `loss/onset`.
    pub onset: f64,
    /// `loss/tail`.
    pub tail: f64,
    /// `loss/periodicity`: signed, positive = output more periodic than
    /// the target (buzzy).
    pub periodicity: f64,
    /// `loss/hnr`: signed dB, positive = harmonics stand further above
    /// the troughs than the target's do (buzzy).
    pub hnr: f64,
    /// `loss/hnr_l1`: mean per-frame |dB error|, what the total uses.
    pub hnr_l1: f64,
    /// `loss/transient`: the unweighted transient term, dB.
    pub transient: f64,
}

impl LossTerms {
    /// Read every term back (synchronises the device).
    #[must_use]
    pub fn values(&self) -> LossValues {
        LossValues {
            total: self.total.double_value(&[]),
            stft: self.stft.double_value(&[]),
            mel: self.mel.double_value(&[]),
            sisdr: self.sisdr.double_value(&[]),
            onset: self.onset.double_value(&[]),
            tail: self.tail.double_value(&[]),
            periodicity: self.periodicity.double_value(&[]),
            hnr: self.hnr.double_value(&[]),
            hnr_l1: self.hnr_l1.double_value(&[]),
            transient: self.transient.double_value(&[]),
        }
    }
}

/// The loss function with its STFT windows and mel filterbank resident on
/// the training device.
#[derive(Debug)]
pub struct Losses {
    cfg: LossCfg,
    stfts: Vec<Stft>,
    mel_stft: Stft,
    mel_fb: Tensor,
    /// 4 ms window, 1 ms hop: the transient term's envelope.
    fine_stft: Stft,
    device: Device,
}

/// The transient term's window and hop at 16 kHz (4 ms / 1 ms).
const FINE_FFT: i64 = 64;
const FINE_HOP: i64 = 16;
/// The transient term's band, Hz — inside what every mode transmits.
const TRANSIENT_LO_HZ: f64 = 2000.0;
const TRANSIENT_HI_HZ: f64 = 3800.0;

/// Weighted mean of a per-frame loss `l [B, F]` under weights `w [B, F]`.
fn weighted_mean(l: &Tensor, w: &Tensor) -> Tensor {
    (l * w).sum(Kind::Float) / w.sum(Kind::Float).clamp_min(1e-6)
}

/// Mean of `l` over a 0/1 region `r` (0 when the region is empty).
fn region_mean(l: &Tensor, r: &Tensor) -> Tensor {
    (l * r).sum(Kind::Float) / r.sum(Kind::Float).clamp_min(1.0)
}

impl Losses {
    /// Build for `device`.
    #[must_use]
    pub fn new(cfg: LossCfg, device: Device) -> Self {
        let stfts = [512i64, 1024, 2048]
            .into_iter()
            .map(|n| Stft::new(n, n / 4, device))
            .collect();
        Self {
            cfg,
            stfts,
            mel_stft: Stft::new(1024, 256, device),
            mel_fb: mel_filterbank(RATE16, 1024, N_MELS, device),
            fine_stft: Stft::new(FINE_FFT, FINE_HOP, device),
            device,
        }
    }

    /// Weights and configuration.
    #[must_use]
    pub const fn cfg(&self) -> LossCfg {
        self.cfg
    }

    /// Per-sample weights `[B, N16]` and the onset / tail region masks.
    fn sample_weights(&self, x: &LossInputs<'_>) -> (Tensor, Tensor, Tensor) {
        let (b, _, n) = x.clean16.size3().unwrap_or((0, 0, 0));
        let pos = Tensor::arange(n, (Kind::Float, self.device));
        let onset_region = pos
            .lt(self.cfg.onset_frames * FRAME16)
            .to_kind(Kind::Float)
            .unsqueeze(0)
            .expand([b, n], false);
        let onset_col = x.onset.to_kind(Kind::Float).unsqueeze(1);
        let tail_col = x.tail.to_kind(Kind::Float).unsqueeze(1);
        let onset_region = onset_region * onset_col; // [B,N]
        let tail_region = (1.0 - x.mask16.squeeze_dim(1)) * tail_col; // [B,N]
        let w = 1.0
            + &onset_region * f64::from(self.cfg.onset_w)
            + &tail_region * f64::from(self.cfg.tail_w - 1.0);
        (w, onset_region, tail_region)
    }

    /// Pick per-sample values at frame centres: `[B, N]` → `[B, frames]`.
    fn at_frames(v: &Tensor, stft: &Stft, n: i64) -> Tensor {
        let frames = stft.frames(n);
        let idx = (Tensor::arange(frames, (Kind::Int64, v.device())) * stft.hop).clamp_max(n - 1);
        v.index_select(1, &idx)
    }

    /// Per-frame spectral loss `[B, frames]` for one resolution.
    fn frame_loss(stft: &Stft, out: &Tensor, target: &Tensor) -> anyhow::Result<Tensor> {
        let mo = stft.magnitude(out)?;
        let mc = stft.magnitude(target)?;
        let scale = mc.mean(Kind::Float).detach().clamp_min(1e-3);
        let lin = (&mo - &mc).abs().mean_dim(1, false, Kind::Float) / scale;
        let log = ((mo + EPS).log() - (mc + EPS).log())
            .abs()
            .mean_dim(1, false, Kind::Float);
        Ok(lin + log)
    }

    /// Per-frame mel loss `[B, frames]`.
    fn mel_frame_loss(&self, out: &Tensor, target: &Tensor) -> anyhow::Result<Tensor> {
        let mo = (self.mel_fb.matmul(&self.mel_stft.magnitude(out)?) + EPS).log();
        let mc = (self.mel_fb.matmul(&self.mel_stft.magnitude(target)?) + EPS).log();
        Ok((mo - mc).abs().mean_dim(1, false, Kind::Float))
    }

    /// Mean SI-SDR in dB over the batch, computed on the masked 8 kHz
    /// path; examples with no target energy are skipped.
    fn si_sdr(inp: &LossInputs<'_>) -> Tensor {
        let mask8 = inp.mask16.slice(2, 0, None, 2); // [B,1,N8]
        let cnt = mask8.sum_dim_intlist(-1, true, Kind::Float).clamp_min(1.0);
        let center = |s: &Tensor| {
            let sm = s * &mask8;
            (&sm - sm.sum_dim_intlist(-1, true, Kind::Float) / &cnt) * &mask8
        };
        let est = center(inp.out8);
        let refr = center(inp.clean8);
        let ref_energy = (&refr * &refr).sum_dim_intlist(-1, false, Kind::Float); // [B,1]
        let cross = (&est * &refr).sum_dim_intlist(-1, false, Kind::Float);
        let alpha = (&cross / ref_energy.clamp_min(EPS)).unsqueeze(-1);
        let target = &alpha * &refr;
        let noise = &est - &target;
        let target_energy = (&target * &target).sum_dim_intlist(-1, false, Kind::Float);
        let noise_energy = (&noise * &noise).sum_dim_intlist(-1, false, Kind::Float);
        let sisdr = (target_energy / noise_energy.clamp_min(EPS) + EPS).log10() * 10.0; // [B,1]
        let valid = ref_energy.gt(1e-6).to_kind(Kind::Float);
        (sisdr * &valid).sum(Kind::Float) / valid.sum(Kind::Float).clamp_min(1.0)
    }

    /// Compute every term.
    pub fn compute(&self, inp: &LossInputs<'_>) -> anyhow::Result<LossTerms> {
        let (batch, _, n16) = inp.clean16.size3()?;
        let (weights, onset_region, tail_region) = self.sample_weights(inp);

        let mut stft_total = Tensor::zeros([], (Kind::Float, self.device));
        let mut onset_acc = Tensor::zeros([], (Kind::Float, self.device));
        let mut tail_acc = Tensor::zeros([], (Kind::Float, self.device));
        let mut region_terms = 0.0;
        for stft in &self.stfts {
            let per_frame = Self::frame_loss(stft, inp.out16, inp.clean16)?; // [B,F]
            let wf = Self::at_frames(&weights, stft, n16);
            stft_total += weighted_mean(&per_frame, &wf);
            onset_acc += region_mean(&per_frame, &Self::at_frames(&onset_region, stft, n16));
            tail_acc += region_mean(&per_frame, &Self::at_frames(&tail_region, stft, n16));
            region_terms += 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let n_res = self.stfts.len() as f64;
        let stft_term = stft_total / n_res;

        let mel_frames = self.mel_frame_loss(inp.out16, inp.clean16)?;
        let wf = Self::at_frames(&weights, &self.mel_stft, n16);
        let mel_term = weighted_mean(&mel_frames, &wf);
        onset_acc += region_mean(
            &mel_frames,
            &Self::at_frames(&onset_region, &self.mel_stft, n16),
        );
        tail_acc += region_mean(
            &mel_frames,
            &Self::at_frames(&tail_region, &self.mel_stft, n16),
        );
        region_terms += 1.0;

        let sisdr_term = Self::si_sdr(inp) * f64::from(-self.cfg.sisdr_w);

        // Babble: per 10 ms frame energy of out16 over the tail region.
        let energy = (inp.out16 * inp.out16)
            .avg_pool1d(FRAME16, FRAME16, 0, false, true)
            .squeeze_dim(1); // [B,T]
        let tail_frames = tail_region
            .view([batch, 1, n16])
            .avg_pool1d(FRAME16, FRAME16, 0, false, true)
            .squeeze_dim(1)
            .gt(0.5)
            .to_kind(Kind::Float);
        let babble_db = ((energy + 1e-6).log10() * 10.0 + 60.0).clamp_min(0.0);
        let babble = region_mean(&babble_db, &tail_frames);

        let weighted_babble = &babble * f64::from(self.cfg.babble_w);
        let periodicity = self.periodicity(inp.out16, inp.clean16, &tail_region, n16, batch)?;
        let (hnr, hnr_l1) = self.hnr(inp.out16, inp.clean16, &tail_region, n16)?;
        let transient = self.transient(inp.out16, inp.clean16, &tail_region, n16)?;
        let onset = onset_acc / region_terms;
        let tail = tail_acc / region_terms + &weighted_babble;
        let mut total = &stft_term + &mel_term + &sisdr_term + &weighted_babble;
        if self.cfg.periodicity_w > 0.0 {
            total += periodicity.abs() * f64::from(self.cfg.periodicity_w);
        }
        if self.cfg.hnr_w > 0.0 {
            total += &hnr_l1 * f64::from(self.cfg.hnr_w);
        }
        if self.cfg.transient_w > 0.0 {
            total += &transient * f64::from(self.cfg.transient_w);
        }
        Ok(LossTerms {
            total,
            stft: stft_term,
            mel: mel_term,
            sisdr: sisdr_term,
            onset,
            tail,
            babble,
            periodicity,
            hnr,
            hnr_l1,
            transient,
        })
    }

    /// **Are the stop consonants still there?**
    ///
    /// A plosive burst is a few milliseconds long. The finest window in
    /// the spectral terms is 32 ms, so a burst is a fraction of one
    /// frame there, and the L1 against its magnitude is minimised by
    /// something smooth at the average level — which is what the codec
    /// already delivers: on the eval clips, D-STAR turns a 3 ms rise into
    /// 14 ms and a 10 ms burst into 30 ms, 13 dB down, and a model
    /// trained on the spectral terms alone leaves that exactly as it
    /// found it, then fills the closure before the burst by 3 dB.
    ///
    /// This term looks at the 2–3.8 kHz energy envelope — inside every
    /// mode's band, so the bandwidth-extension head cannot pay it off —
    /// at **1 ms** resolution, in dB, and charges the L1 of the envelope
    /// *and* of its slope against the target's: the level term sees a
    /// filled closure and a smeared tail, the slope term a slow rise.
    /// Log domain, so a −40 dB closure filled to −28 dB costs as much as
    /// a burst 12 dB short. Speech frames only (the tail has the babble
    /// penalty). Returns the unweighted sum, dB.
    fn transient(
        &self,
        out16: &Tensor,
        clean16: &Tensor,
        tail_region: &Tensor,
        n16: i64,
    ) -> anyhow::Result<Tensor> {
        let stft = &self.fine_stft;
        let bin_hz = f64::from(RATE16) / f64::from(i32::try_from(FINE_FFT)?);
        #[allow(clippy::cast_possible_truncation)]
        let (lo, hi) = (
            (TRANSIENT_LO_HZ / bin_hz).ceil() as i64,
            (TRANSIENT_HI_HZ / bin_hz).floor() as i64,
        );
        let band_db = |x: &Tensor| -> anyhow::Result<Tensor> {
            let p = stft.magnitude(x)?.pow_tensor_scalar(2.0); // [B,bins,F]
            let e = p
                .narrow(1, lo, hi - lo + 1)
                .sum_dim_intlist(1, false, Kind::Float); // [B,F]
            Ok((e + 1e-8).log10() * 10.0)
        };
        let (eo, ec) = (band_db(out16)?, band_db(clean16)?.detach());
        let speech = Self::at_frames(&(1.0 - tail_region), stft, n16);
        let frames = eo.size2()?.1.min(ec.size2()?.1).min(speech.size2()?.1);
        let (eo, ec, speech) = (
            eo.narrow(1, 0, frames),
            ec.narrow(1, 0, frames),
            speech.narrow(1, 0, frames),
        );
        let level = region_mean(&(&eo - &ec).abs(), &speech);
        let slope = |e: &Tensor| e.narrow(1, 1, frames - 1) - e.narrow(1, 0, frames - 1);
        let slope_err = (slope(&eo) - slope(&ec)).abs();
        let slope_term = region_mean(&slope_err, &speech.narrow(1, 1, frames - 1));
        Ok(level + slope_term)
    }

    /// **How periodic the output is, against how periodic it should be.**
    ///
    /// Every other term compares magnitudes bin by bin, and a stochastic
    /// signal cannot be matched that way: the safest way to minimise an
    /// L1 against a particular realisation of breath noise is to emit
    /// nothing and stay smooth. That pressure is precisely what leaves a
    /// vocoded voice sounding robotic, because the thing the codec
    /// destroyed *is* the noise between the harmonics.
    ///
    /// So this term compares a *statistic* instead. The cepstral peak
    /// prominence — the height of the pitch peak in the cepstrum over
    /// the local trend — measures how strongly a harmonic comb stands
    /// out of the spectrum. Perfectly periodic speech has a tall peak;
    /// breathy, jittery, human speech has a shorter one. Matching the
    /// target's prominence rewards being *as periodic as the clean
    /// speech was*, in either direction, and says nothing about which
    /// particular noise samples to use.
    ///
    /// Returns `mean(cpp(out) − cpp(target))` over the speech region,
    /// signed so the sign tells you which way the model is wrong.
    fn periodicity(
        &self,
        out16: &Tensor,
        clean16: &Tensor,
        tail_region: &Tensor,
        n16: i64,
        batch: i64,
    ) -> anyhow::Result<Tensor> {
        let _ = batch;
        // Speech frames only: the garbage tail's target is silence, and
        // the prominence of silence is meaningless.
        let keep = Self::at_frames(&(1.0 - tail_region), &self.mel_stft, n16); // [B,F]
        let a = self.cpp(out16)?;
        let b = self.cpp(clean16)?;
        let frames = a.size2()?.1.min(keep.size2()?.1);
        let (a, b, keep) = (
            a.narrow(1, 0, frames),
            b.narrow(1, 0, frames),
            keep.narrow(1, 0, frames),
        );
        Ok(region_mean(&(a - b), &keep))
    }

    /// **How far the harmonics stand above the valleys between them,
    /// against how far they should.**
    ///
    /// The measure a listener's "still sounds robotic" turns into: a
    /// vocoder codes voiced bands as exactly periodic, so the troughs
    /// halfway between harmonics — where breath, jitter and glottal
    /// noise live in real speech — come back empty. Measured on this
    /// project's own eval clips, clean speech sits at 18.7 dB and a
    /// D-STAR decode at 28.9 dB.
    ///
    /// Unlike [`Losses::periodicity`] this cannot be satisfied
    /// sideways. Cepstral peak prominence can be lowered by adding
    /// ripple at *neighbouring* quefrencies, which moves the statistic
    /// without filling a single trough — and at a weight high enough to
    /// matter, that is exactly what the model did (run
    /// `20260912-183551`: the prominence error fell to 0.01 while the
    /// harmonic-to-trough ratio got *worse*, 24.6 → 26.0 dB). This term
    /// names the bins: to lower it the model must put energy between
    /// the harmonics, which only the noise head can do.
    ///
    /// The pitch, the bin masks and the target's ratio all come from
    /// `clean16` and are detached; only the output's ratio carries a
    /// gradient. Returns the signed dB error over voiced speech frames,
    /// positive meaning buzzier than the target.
    /// Returns `(signed mean, mean of |per-frame error|)`. The signed
    /// mean is the diagnostic — it says *which way* the model is wrong —
    /// but the total uses the per-frame absolute error, because a run
    /// that is buzzy on half its frames and breathy on the other half
    /// is not correct, and a signed mean would call it so.
    fn hnr(
        &self,
        out16: &Tensor,
        clean16: &Tensor,
        tail_region: &Tensor,
        n16: i64,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let stft = &self.mel_stft;
        let po = stft
            .magnitude(out16)?
            .pow_tensor_scalar(2.0)
            .transpose(1, 2); // [B,F,bins]
        let pc = stft
            .magnitude(clean16)?
            .pow_tensor_scalar(2.0)
            .transpose(1, 2)
            .detach();
        let (_, frames, bins) = po.size3()?;

        // Pitch from the target, by autocorrelation (Wiener-Khinchin:
        // the inverse transform of the power spectrum).
        let ac = pc.fft_irfft(None, -1, "backward"); // [B,F,lags]
        let lags = ac.size3()?.2;
        let lo = (i64::from(RATE16) / 320).min(lags - 2);
        let hi = (i64::from(RATE16) / 60).min(lags - 1);
        let band = ac.narrow(2, lo, (hi - lo).max(1));
        let (peak, idx) = band.max_dim(2, false);
        let zero = ac.narrow(2, 0, 1).squeeze_dim(2).abs().clamp_min(1e-12);
        let periodicity = &peak / &zero;
        let lag = (idx + lo).to_kind(Kind::Float).clamp_min(1.0);
        let f0 = f64::from(RATE16) / lag; // [B,F]

        // Harmonic and trough windows, in units of f0.
        let hz_per_bin = f64::from(RATE16) / (2.0 * f64::from(i32::try_from(bins - 1)?));
        let freqs = Tensor::arange(bins, (Kind::Float, po.device())) * hz_per_bin; // [bins]
        let ratio = freqs.view([1, 1, bins]) / f0.unsqueeze(2).clamp_min(1.0);
        let frac = &ratio - ratio.round();
        let in_band = freqs
            .view([1, 1, bins])
            .lt_tensor(&Tensor::from(HNR_MAX_HZ))
            .logical_and(&freqs.view([1, 1, bins]).gt_tensor(&(f0.unsqueeze(2) * 1.5)));
        let harmonic = frac
            .abs()
            .lt(HNR_HALF_WIDTH)
            .logical_and(&in_band)
            .to_kind(Kind::Float);
        let trough = (frac.abs() - 0.5)
            .abs()
            .lt(HNR_HALF_WIDTH)
            .logical_and(&in_band)
            .to_kind(Kind::Float);
        let (hn, tn) = (
            harmonic
                .sum_dim_intlist(-1, false, Kind::Float)
                .clamp_min(1.0),
            trough
                .sum_dim_intlist(-1, false, Kind::Float)
                .clamp_min(1.0),
        );
        // Mean *log* power over the harmonic bins against mean log power
        // over the trough bins: a geometric-mean ratio, so every
        // harmonic counts once whatever its level. The arithmetic
        // version — mean(peaks) / mean(troughs) — is the loudest
        // harmonics' ratio, and a model satisfied it by fixing the
        // 0–1 kHz band alone (docs/notes, 2026-09-12).
        let ratio_db = |p: &Tensor| {
            let lp = (p + 1e-12).log10();
            let h = (&lp * &harmonic).sum_dim_intlist(-1, false, Kind::Float) / &hn;
            let t = (&lp * &trough).sum_dim_intlist(-1, false, Kind::Float) / &tn;
            (h - t) * 10.0
        };
        let diff = ratio_db(&po) - ratio_db(&pc).detach(); // [B,F]

        // Voiced speech frames only: an unvoiced frame has no comb, and
        // the tail's target is silence.
        let speech = Self::at_frames(&(1.0 - tail_region), stft, n16);
        let frames_kept = speech.size2()?.1.min(frames);
        let voiced = periodicity
            .ge(VOICED_PERIODICITY)
            .to_kind(Kind::Float)
            .narrow(1, 0, frames_kept)
            * speech.narrow(1, 0, frames_kept);
        let diff = diff.narrow(1, 0, frames_kept);
        Ok((
            region_mean(&diff, &voiced),
            region_mean(&diff.abs(), &voiced),
        ))
    }

    /// Per-frame cepstral peak prominence `[B, F]`: the largest value in
    /// the pitch quefrency band of the real cepstrum, measured against
    /// that band's own mean so a change in overall spectral tilt does
    /// not move it.
    fn cpp(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mag = self.mel_stft.magnitude(x)?; // [B,bins,frames]
        let log_mag = (mag + EPS).log().transpose(1, 2); // [B,frames,bins]
        let cep = log_mag.fft_irfft(None, -1, "backward");
        let q = cep.size3()?.2;
        // 60-320 Hz at 16 kHz: quefrency 50 to 266 samples.
        let lo = (i64::from(RATE16) / 320).min(q - 2);
        let hi = (i64::from(RATE16) / 60).min(q - 1);
        let band = cep.narrow(2, lo, (hi - lo).max(1));
        let peak = band.max_dim(2, false).0;
        let base = band.mean_dim(-1, false, Kind::Float);
        Ok(peak - base)
    }
}

// ───────────────────────── discriminators (gan = true) ─────────────────────

/// One discriminator's verdict: logits and the feature maps behind them.
#[derive(Debug)]
pub struct DiscOut {
    /// Logits, any shape.
    pub logits: Tensor,
    /// Intermediate activations for feature matching.
    pub fmaps: Vec<Tensor>,
}

fn conv2(p: &nn::Path, i: i64, o: i64, k: [i64; 2], stride: [i64; 2], pad: [i64; 2]) -> nn::Conv2D {
    nn::conv(
        p,
        i,
        o,
        k,
        nn::ConvConfigND::<[i64; 2]> {
            stride,
            padding: pad,
            ..Default::default()
        },
    )
}

/// Period discriminator: folds the waveform into `[B, 1, T / p, p]` and
/// runs a (5, 1)-kernel conv stack down the time axis.
#[derive(Debug)]
struct PeriodDisc {
    period: i64,
    convs: Vec<nn::Conv2D>,
    post: nn::Conv2D,
}

impl PeriodDisc {
    fn new(p: &nn::Path, period: i64, c: i64) -> Self {
        let widths = [1, c, 2 * c, 4 * c, 4 * c];
        let convs = widths
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                conv2(
                    &(p / format!("conv{i}")),
                    w[0],
                    w[1],
                    [5, 1],
                    [3, 1],
                    [2, 0],
                )
            })
            .collect::<Vec<_>>();
        Self {
            period,
            convs,
            post: conv2(&(p / "post"), 4 * c, 1, [3, 1], [1, 1], [1, 0]),
        }
    }

    fn forward(&self, x: &Tensor) -> DiscOut {
        let (b, ch, t) = x.size3().unwrap_or((0, 1, 0));
        let pad = (self.period - t % self.period) % self.period;
        let x = x
            .constant_pad_nd([0, pad])
            .view([b, ch, (t + pad) / self.period, self.period]);
        let mut fmaps = Vec::with_capacity(self.convs.len());
        let mut h = x;
        for c in &self.convs {
            h = c.forward(&h).leaky_relu();
            fmaps.push(h.shallow_clone());
        }
        DiscOut {
            logits: self.post.forward(&h),
            fmaps,
        }
    }
}

/// Resolution discriminator: a conv stack over one STFT magnitude
/// `[B, 1, bins, frames]`.
#[derive(Debug)]
struct ResDisc {
    stft: Stft,
    convs: Vec<nn::Conv2D>,
    post: nn::Conv2D,
}

impl ResDisc {
    fn new(p: &nn::Path, n_fft: i64, c: i64, device: Device) -> Self {
        let widths = [1, c, c, 2 * c, 2 * c];
        let convs = widths
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let stride = if i == 0 { [1, 1] } else { [2, 1] };
                conv2(
                    &(p / format!("conv{i}")),
                    w[0],
                    w[1],
                    [3, 3],
                    stride,
                    [1, 1],
                )
            })
            .collect::<Vec<_>>();
        Self {
            stft: Stft::new(n_fft, n_fft / 4, device),
            convs,
            post: conv2(&(p / "post"), 2 * c, 1, [3, 3], [1, 1], [1, 1]),
        }
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<DiscOut> {
        let mag = self.stft.magnitude(x)?.unsqueeze(1); // [B,1,bins,frames]
        let mut h = (mag + EPS).log();
        let mut fmaps = Vec::with_capacity(self.convs.len());
        for c in &self.convs {
            h = c.forward(&h).leaky_relu();
            fmaps.push(h.shallow_clone());
        }
        Ok(DiscOut {
            logits: self.post.forward(&h),
            fmaps,
        })
    }
}

/// Multi-period + multi-resolution discriminators.
#[derive(Debug)]
pub struct Discriminators {
    periods: Vec<PeriodDisc>,
    resolutions: Vec<ResDisc>,
}

/// MPD periods.
pub const PERIODS: [i64; 5] = [2, 3, 5, 7, 11];
/// MRD FFT sizes (hop `n_fft / 4`).
pub const RESOLUTIONS: [i64; 3] = [512, 1024, 2048];

impl Discriminators {
    /// Build under `p` with base width `channels` (32 for a real run, 4–8
    /// for tests).
    #[must_use]
    pub fn new(p: &nn::Path, channels: i64) -> Self {
        let device = p.device();
        let periods = PERIODS
            .iter()
            .map(|&per| PeriodDisc::new(&(p / format!("mpd_p{per}")), per, channels))
            .collect();
        let resolutions = RESOLUTIONS
            .iter()
            .map(|&n| ResDisc::new(&(p / format!("mrd_n{n}")), n, channels, device))
            .collect();
        Self {
            periods,
            resolutions,
        }
    }

    /// Number of sub-discriminators.
    #[must_use]
    pub fn len(&self) -> usize {
        self.periods.len() + self.resolutions.len()
    }

    /// Whether there are no sub-discriminators (never, but clippy asks).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run every sub-discriminator on `x [B, 1, N16]`.
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Vec<DiscOut>> {
        let mut outs = Vec::with_capacity(self.len());
        for d in &self.periods {
            outs.push(d.forward(x));
        }
        for d in &self.resolutions {
            outs.push(d.forward(x)?);
        }
        Ok(outs)
    }
}

/// LSGAN discriminator loss: `mean((1 − D(real))²) + mean(D(fake)²)`,
/// summed over sub-discriminators.
pub fn disc_loss(real: &[DiscOut], fake: &[DiscOut]) -> Tensor {
    real.iter()
        .zip(fake)
        .fold(Tensor::from(0f32), |acc, (r, f)| {
            acc + (&r.logits - 1.0).square().mean(Kind::Float) + f.logits.square().mean(Kind::Float)
        })
}

/// LSGAN generator loss: `mean((1 − D(fake))²)` summed over
/// sub-discriminators.
pub fn gen_adv_loss(fake: &[DiscOut]) -> Tensor {
    fake.iter().fold(Tensor::from(0f32), |acc, f| {
        acc + (&f.logits - 1.0).square().mean(Kind::Float)
    })
}

/// Feature matching: mean L1 between real (detached) and fake feature
/// maps, averaged over layers, summed over sub-discriminators.
pub fn feature_matching(real: &[DiscOut], fake: &[DiscOut]) -> Tensor {
    real.iter()
        .zip(fake)
        .fold(Tensor::from(0f32), |acc, (r, f)| {
            let per = r
                .fmaps
                .iter()
                .zip(&f.fmaps)
                .fold(Tensor::from(0f32), |a, (rm, fm)| {
                    a + (rm.detach() - fm).abs().mean(Kind::Float)
                });
            #[allow(clippy::cast_precision_loss)]
            let n = r.fmaps.len().max(1) as f64;
            acc + per / n
        })
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    const N16: i64 = 6400; // 0.4 s

    fn tone16(freq: f32, amp: f32, n: i64) -> Tensor {
        #[allow(clippy::cast_precision_loss)]
        let v: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / 16_000.0).sin())
            .collect();
        Tensor::from_slice(&v).view([1, 1, n])
    }

    struct Fix {
        out16: Tensor,
        clean16: Tensor,
        mask16: Tensor,
        out8: Tensor,
        clean8: Tensor,
        onset: Tensor,
        tail: Tensor,
    }

    impl Fix {
        fn new(out16: Tensor, clean16: Tensor, mask16: Tensor, onset: bool, tail: bool) -> Self {
            let out8 = out16.slice(2, 0, None, 2);
            let clean8 = clean16.slice(2, 0, None, 2);
            Self {
                out16,
                clean16,
                mask16,
                out8,
                clean8,
                onset: Tensor::from_slice(&[onset]),
                tail: Tensor::from_slice(&[tail]),
            }
        }
        fn inputs(&self) -> LossInputs<'_> {
            LossInputs {
                out16: &self.out16,
                clean16: &self.clean16,
                mask16: &self.mask16,
                out8: &self.out8,
                clean8: &self.clean8,
                onset: &self.onset,
                tail: &self.tail,
            }
        }
    }

    /// A stop consonant: 150 ms of near silence, an 8 ms burst, then a
    /// tone. A codec-shaped output — the burst a third as loud, spread
    /// over 30 ms with a slow fade-in, and noise in the closure — pays
    /// the transient term; the clean target itself pays nothing; and the
    /// weight puts the term into the total.
    #[test]
    fn the_transient_term_charges_a_smeared_burst_and_a_filled_closure() {
        let ms = 16i64;
        let stop = |burst_ms: i64, fade_ms: i64, level: f32, floor: f32, seed: u32| -> Tensor {
            let mut s = seed;
            let mut noise = move || {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let v = (s >> 16) as f32 / 65_536.0 - 0.5;
                v
            };
            let mut v = Vec::new();
            for _ in 0..150 * ms {
                v.push(noise() * floor);
            }
            for i in 0..burst_ms * ms {
                #[allow(clippy::cast_precision_loss)]
                let ramp = if fade_ms == 0 {
                    1.0
                } else {
                    (i as f32 / (fade_ms * ms) as f32).min(1.0)
                };
                v.push(noise() * level * ramp);
            }
            while v.len() < usize::try_from(N16).unwrap() {
                #[allow(clippy::cast_precision_loss)]
                let t = v.len() as f32 / 16_000.0;
                v.push((2.0 * std::f32::consts::PI * 180.0 * t).sin() * 0.3 + noise() * floor);
            }
            Tensor::from_slice(&v).view([1, 1, N16])
        };
        let clean = stop(8, 0, 0.5, 1e-4, 1);
        let smeared = stop(30, 20, 0.17, 0.02, 2);
        let mask = Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu));
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let same = Fix::new(
            clean.shallow_clone(),
            clean.shallow_clone(),
            mask.shallow_clone(),
            false,
            false,
        );
        let bad = Fix::new(smeared, clean.shallow_clone(), mask, false, false);
        let t_same = losses.compute(&same.inputs()).unwrap().values().transient;
        let t_bad = losses.compute(&bad.inputs()).unwrap().values().transient;
        assert!(t_same.abs() < 1e-4, "identical signals: {t_same}");
        assert!(
            t_bad > 5.0,
            "a smeared burst over a filled closure costs dB: {t_bad}"
        );
        let weighted = Losses::new(
            LossCfg {
                transient_w: 0.1,
                ..LossCfg::default()
            },
            Device::Cpu,
        );
        let (a, b) = (
            losses.compute(&bad.inputs()).unwrap().values(),
            weighted.compute(&bad.inputs()).unwrap().values(),
        );
        assert!(
            (b.total - a.total - 0.1 * a.transient).abs() < 1e-3,
            "the weight adds the term: {} vs {} + 0.1 × {}",
            b.total,
            a.total,
            a.transient
        );
    }

    /// A pulse train is perfectly periodic; the same train with noise
    /// between its harmonics is not. The periodicity term is **signed**,
    /// so it says *which way* the model is wrong, and it is ~0 when the
    /// output is exactly as periodic as its target.
    #[test]
    fn the_periodicity_term_is_signed_and_zero_when_they_match() {
        let pulses = |n: i64, noise: f32| {
            let mut seed = 7u32;
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    #[allow(clippy::cast_precision_loss)]
                    let u = (seed >> 16) as f32 / 65_536.0 - 0.5;
                    (if i % 128 == 0 { 1.0 } else { 0.0 }) + u * noise
                })
                .collect();
            Tensor::from_slice(&v).view([1, 1, n])
        };
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let ones = || Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu));

        // Output buzzier than the target → positive.
        let f = Fix::new(pulses(N16, 0.0), pulses(N16, 0.3), ones(), false, false);
        let buzzy = losses.compute(&f.inputs()).unwrap().values().periodicity;
        assert!(buzzy > 0.0, "buzzier output should read positive: {buzzy}");

        // The other way round → negative.
        let f = Fix::new(pulses(N16, 0.3), pulses(N16, 0.0), ones(), false, false);
        let breathy = losses.compute(&f.inputs()).unwrap().values().periodicity;
        assert!(
            breathy < 0.0,
            "breathier output should read negative: {breathy}"
        );

        // Same signal → nothing to fix.
        let x = pulses(N16, 0.2);
        let f = Fix::new(x.copy(), x, ones(), false, false);
        let same = losses.compute(&f.inputs()).unwrap().values().periodicity;
        assert!(same.abs() < 1e-5, "identical signals: {same}");
    }

    /// The term only enters the total when `periodicity_w > 0`, and it
    /// enters as its absolute value — too breathy is as wrong as too
    /// buzzy.
    #[test]
    fn the_periodicity_weight_adds_the_absolute_error_to_the_total() {
        let buzz: Vec<f32> = (0..N16)
            .map(|i| if i % 128 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut seed = 99u32;
        let breathy: Vec<f32> = buzz
            .iter()
            .map(|p| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let u = (seed >> 16) as f32 / 65_536.0 - 0.5;
                p + u * 0.3
            })
            .collect();
        let t = |v: &[f32]| Tensor::from_slice(v).view([1, 1, N16]);
        let f = Fix::new(
            t(&buzz),
            t(&breathy),
            Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu)),
            false,
            false,
        );
        let off = Losses::new(LossCfg::default(), Device::Cpu)
            .compute(&f.inputs())
            .unwrap()
            .values();
        let on = Losses::new(
            LossCfg {
                periodicity_w: 1.0,
                ..LossCfg::default()
            },
            Device::Cpu,
        )
        .compute(&f.inputs())
        .unwrap()
        .values();
        assert!(
            (off.periodicity - on.periodicity).abs() < 1e-6,
            "the diagnostic is the same"
        );
        assert!(
            (on.total - (off.total + off.periodicity.abs())).abs() < 1e-5,
            "weighted total: {} vs {} + |{}|",
            on.total,
            off.total,
            off.periodicity
        );
        assert!(on.total > off.total, "the term should cost something");
    }

    /// A buzz (empty troughs) against the same buzz with the troughs
    /// filled: the hnr term reads **positive** when the output is the
    /// more periodic, negative the other way, and ~0 when they match.
    /// This is the loss that the residual robotic quality is measured
    /// by, so its sign convention matters.
    #[test]
    fn the_hnr_term_is_signed_by_how_much_buzzier_the_output_is() {
        // 125 Hz pulse train at 16 kHz: harmonics every 125 Hz, nothing
        // between them.
        let buzz = |noise: f32, seed0: u32| {
            let mut seed = seed0;
            let v: Vec<f32> = (0..N16)
                .map(|i| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    #[allow(clippy::cast_precision_loss)]
                    let u = (seed >> 16) as f32 / 65_536.0 - 0.5;
                    (if i % 128 == 0 { 0.8 } else { 0.0 }) + u * noise
                })
                .collect();
            Tensor::from_slice(&v).view([1, 1, N16])
        };
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let ones = || Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu));

        let f = Fix::new(buzz(0.0, 1), buzz(0.05, 2), ones(), false, false);
        let buzzier = losses.compute(&f.inputs()).unwrap().values().hnr;
        assert!(
            buzzier > 1.0,
            "empty troughs should read clearly positive: {buzzier}"
        );

        let f = Fix::new(buzz(0.05, 2), buzz(0.0, 1), ones(), false, false);
        let breathier = losses.compute(&f.inputs()).unwrap().values().hnr;
        assert!(
            breathier < -1.0,
            "filled troughs should read negative: {breathier}"
        );

        let x = buzz(0.02, 3);
        let f = Fix::new(x.copy(), x, ones(), false, false);
        let same = losses.compute(&f.inputs()).unwrap().values().hnr;
        assert!(same.abs() < 1e-4, "identical signals: {same}");
    }

    /// The term enters the total as its absolute value, weighted, and
    /// only when `hnr_w > 0`.
    #[test]
    fn the_hnr_weight_adds_the_absolute_error_to_the_total() {
        let mut seed = 5u32;
        let clean: Vec<f32> = (0..N16)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let u = (seed >> 16) as f32 / 65_536.0 - 0.5;
                (if i % 128 == 0 { 0.8 } else { 0.0 }) + u * 0.05
            })
            .collect();
        let buzz: Vec<f32> = (0..N16)
            .map(|i| if i % 128 == 0 { 0.8 } else { 0.0 })
            .collect();
        let t = |v: &[f32]| Tensor::from_slice(v).view([1, 1, N16]);
        let f = Fix::new(
            t(&buzz),
            t(&clean),
            Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu)),
            false,
            false,
        );
        let off = Losses::new(LossCfg::default(), Device::Cpu)
            .compute(&f.inputs())
            .unwrap()
            .values();
        let on = Losses::new(
            LossCfg {
                hnr_w: 0.5,
                ..LossCfg::default()
            },
            Device::Cpu,
        )
        .compute(&f.inputs())
        .unwrap()
        .values();
        assert!(
            (off.hnr - on.hnr).abs() < 1e-6,
            "the diagnostic is unweighted"
        );
        assert!(
            (on.total - (off.total + 0.5 * off.hnr_l1)).abs() < 1e-4,
            "{} vs {} + 0.5*{}",
            on.total,
            off.total,
            off.hnr_l1
        );
    }

    /// **The property the fix rests on**, and its limit. On a
    /// speech-like signal — a harmonic comb over a real noise floor —
    /// the term is dominated by what sits *between* the harmonics:
    /// adding inter-harmonic noise moves it about 6 dB, while a
    /// spectral tilt of the kind the filter heads can apply moves it
    /// about 1. It is not perfectly immune to shaping (a tilt changes
    /// the harmonic peaks too), but shaping is independently expensive
    /// under the spectral terms, and noise is the cheap way to satisfy
    /// it — which is the incentive the noise head needs.
    #[test]
    fn the_hnr_term_answers_to_noise_far_more_than_to_spectral_shaping() {
        let mut seed = 3u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let v = (seed >> 16) as f32 / 65_536.0 - 0.5;
            v
        };
        let base: Vec<f32> = (0..N16)
            .map(|i| (if i % 128 == 0 { 0.8 } else { 0.0 }) + rnd() * 0.02)
            .collect();
        let mut tilted = vec![0.0f32; base.len()];
        for i in 1..base.len() {
            tilted[i] = 0.8 * base[i] + 0.2 * base[i - 1];
        }
        let mut seed2 = 9u32;
        let noisier: Vec<f32> = base
            .iter()
            .map(|v| {
                seed2 = seed2.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let u = (seed2 >> 16) as f32 / 65_536.0 - 0.5;
                v + u * 0.04
            })
            .collect();

        let t = |v: &[f32]| Tensor::from_slice(v).view([1, 1, N16]);
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let ones = || Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu));
        let hnr = |out: &Vec<f32>| {
            losses
                .compute(&Fix::new(t(out), t(&base), ones(), false, false).inputs())
                .unwrap()
                .values()
                .hnr
        };
        let (tilt, noise) = (hnr(&tilted), hnr(&noisier));
        assert!(hnr(&base).abs() < 1e-4, "identical signals must read 0");
        assert!(
            noise < -3.0,
            "added noise should read much less buzzy: {noise}"
        );
        assert!(
            noise.abs() > 3.0 * tilt.abs(),
            "noise {noise} should dominate tilt {tilt}"
        );
    }

    #[test]
    fn identical_signals_give_zero_spectral_loss_and_high_sisdr() {
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let x = tone16(440.0, 0.3, N16);
        let f = Fix::new(
            x.copy(),
            x,
            Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu)),
            false,
            false,
        );
        let v = losses.compute(&f.inputs()).unwrap().values();
        assert!(v.stft.abs() < 1e-5, "{v:?}");
        assert!(v.mel.abs() < 1e-5, "{v:?}");
        // sisdr term = −0.05 · SI-SDR; identical → SI-SDR ≈ 50 dB (eps floor).
        assert!(v.sisdr < -2.0, "{v:?}");
        assert!(v.onset.abs() < f64::EPSILON);
        assert!(v.tail.abs() < f64::EPSILON);
        assert!((v.total - (v.stft + v.mel + v.sisdr)).abs() < 1e-6);
    }

    #[test]
    fn scaled_output_is_penalised_by_the_spectral_terms_only() {
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let x = tone16(440.0, 0.3, N16);
        let f = Fix::new(
            &x * 0.5,
            x,
            Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu)),
            false,
            false,
        );
        let v = losses.compute(&f.inputs()).unwrap().values();
        // log term: |ln 0.5| = 0.693 in the bins that carry energy; the
        // linear term is 0.5 · mean|X| / mean|X| = 0.5.
        assert!(v.stft > 0.3 && v.stft < 1.5, "{v:?}");
        assert!(v.mel > 0.1, "{v:?}");
        // SI-SDR is scale invariant.
        assert!(v.sisdr < -2.0, "{v:?}");
    }

    #[test]
    fn noise_has_lower_sisdr_than_a_scaled_copy() {
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let x = tone16(300.0, 0.3, N16);
        let mut r = crate::rng::Rng::new(5);
        let noise: Vec<f32> = (0..N16).map(|_| 0.1 * r.normal()).collect();
        let corrupted = &x + Tensor::from_slice(&noise).view([1, 1, N16]);
        let ones = Tensor::ones([1, 1, N16], (Kind::Float, Device::Cpu));
        let a = Fix::new(&x * 0.5, x.copy(), ones.copy(), false, false);
        let b = Fix::new(corrupted, x, ones, false, false);
        let va = losses.compute(&a.inputs()).unwrap().values();
        let vb = losses.compute(&b.inputs()).unwrap().values();
        assert!(vb.sisdr > va.sisdr, "{va:?} {vb:?}");
        // −0.05 · SI-SDR with SI-SDR ≈ 10·log10(0.045/0.01) ≈ 6.5 dB.
        assert!(vb.sisdr > -0.5 && vb.sisdr < -0.1, "{vb:?}");
    }

    #[test]
    fn tail_region_uses_the_silence_target_and_babble_penalty() {
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let half = N16 / 2;
        let speech = tone16(440.0, 0.3, N16);
        // Target: speech then silence; mask 1 then 0.
        let clean = Tensor::cat(
            &[
                speech.narrow(2, 0, half),
                Tensor::zeros([1, 1, half], (Kind::Float, Device::Cpu)),
            ],
            2,
        );
        let mask = Tensor::cat(
            &[
                Tensor::ones([1, 1, half], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 1, half], (Kind::Float, Device::Cpu)),
            ],
            2,
        );
        // Output A: perfect. Output B: babbles in the tail at −20 dBFS.
        let quiet = Fix::new(clean.copy(), clean.copy(), mask.copy(), false, true);
        let babbling = Fix::new(
            Tensor::cat(
                &[
                    speech.narrow(2, 0, half),
                    tone16(700.0, 0.1, N16).narrow(2, 0, half),
                ],
                2,
            ),
            clean,
            mask,
            false,
            true,
        );
        let vq = losses.compute(&quiet.inputs()).unwrap().values();
        let vb = losses.compute(&babbling.inputs()).unwrap().values();
        assert!(vq.tail.abs() < 1e-4, "{vq:?}");
        assert!(vb.tail > 1.0, "{vb:?}");
        assert!(vb.total > vq.total + 1.0, "{vq:?} {vb:?}");
        // Babble ≈ 10·log10(0.005) + 60 = 37 dB above the floor.
        let terms = losses.compute(&babbling.inputs()).unwrap();
        let babble = terms.babble.double_value(&[]);
        assert!((babble - 37.0).abs() < 3.0, "{babble}");
        // With tail_w = 1 (no extra weight) the stft term drops.
        let flat = Losses::new(
            LossCfg {
                tail_w: 1.0,
                ..LossCfg::default()
            },
            Device::Cpu,
        );
        let vf = flat.compute(&babbling.inputs()).unwrap().values();
        assert!(vf.stft < vb.stft, "{vf:?} {vb:?}");
    }

    #[test]
    fn onset_weight_raises_the_loss_on_early_errors_only() {
        let base = Losses::new(LossCfg::default(), Device::Cpu);
        // 1 s so that the 0.5 s onset region is a proper subset.
        let x = tone16(440.0, 0.3, 16_000);
        let ones = Tensor::ones([1, 1, 16_000], (Kind::Float, Device::Cpu));
        // Error confined to the first 0.2 s (inside the 0.5 s onset region).
        let bad = x.copy();
        let _ = bad
            .narrow(2, 0, 3200)
            .f_mul_(&Tensor::from(0.3f32))
            .unwrap();
        let onset = Fix::new(bad.copy(), x.copy(), ones.copy(), true, false);
        let plain = Fix::new(bad, x, ones, false, false);
        let vo = base.compute(&onset.inputs()).unwrap().values();
        let vp = base.compute(&plain.inputs()).unwrap().values();
        assert!(vo.stft > vp.stft, "{vo:?} {vp:?}");
        assert!(vo.onset > 0.0 && vp.onset == 0.0, "{vo:?} {vp:?}");
    }

    #[test]
    fn losses_have_gradients_and_batch_2_works() {
        let losses = Losses::new(LossCfg::default(), Device::Cpu);
        let x = Tensor::cat(&[tone16(440.0, 0.3, N16), tone16(220.0, 0.2, N16)], 0);
        let out = (&x * 0.7).set_requires_grad(true);
        let out8 = out.slice(2, 0, None, 2);
        let clean8 = x.slice(2, 0, None, 2);
        let mask = Tensor::ones([2, 1, N16], (Kind::Float, Device::Cpu));
        let onset = Tensor::from_slice(&[true, false]);
        let tail = Tensor::from_slice(&[false, true]);
        let terms = losses
            .compute(&LossInputs {
                out16: &out,
                clean16: &x,
                mask16: &mask,
                out8: &out8,
                clean8: &clean8,
                onset: &onset,
                tail: &tail,
            })
            .unwrap();
        terms.total.backward();
        assert!(out.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(terms.values().total.is_finite());
    }

    #[test]
    fn discriminator_shapes_and_gan_loss_plumbing() {
        let vs = nn::VarStore::new(Device::Cpu);
        let d = Discriminators::new(&vs.root(), 4);
        assert_eq!(d.len(), 8);
        let real = tone16(440.0, 0.3, N16);
        let fake = tone16(450.0, 0.3, N16).set_requires_grad(true);
        let ro = d.forward(&real).unwrap();
        let fo = d.forward(&fake).unwrap();
        assert_eq!(ro.len(), 8);
        for (i, o) in ro.iter().enumerate() {
            assert_eq!(o.fmaps.len(), 4, "disc {i}");
            assert_eq!(o.logits.size()[0], 1, "disc {i}");
            assert_eq!(o.logits.size()[1], 1, "disc {i}");
            if i < PERIODS.len() {
                assert_eq!(o.logits.size()[3], PERIODS[i], "period {i}");
            } else {
                let n = RESOLUTIONS[i - PERIODS.len()];
                // bins halved three times by the strided convs.
                assert_eq!(o.logits.size()[2], (n / 2 + 1 + 7) / 8, "res {n}");
                assert_eq!(o.logits.size()[3], 1 + N16 / (n / 4), "res {n}");
            }
        }
        let dl = disc_loss(&ro, &fo);
        let gl = gen_adv_loss(&fo) + feature_matching(&ro, &fo);
        assert_eq!(dl.size(), Vec::<i64>::new());
        assert!(dl.double_value(&[]) > 0.0);
        gl.backward();
        assert!(fake.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
    }
}
