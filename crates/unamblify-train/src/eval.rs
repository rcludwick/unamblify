// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Evaluation on fixed dev clips (spec §7): LSD, mel-L1 and SI-SDR over
//! the whole utterance, LSD over its first and last second, and the
//! babble energy over a one-second garbage tail appended to every clip
//! (the same generator as the training tail examples, seeded per clip),
//! so the key-down behaviour is measured on real speech. Two more LSD
//! columns measure the augmentations: `eval/lsd_rx`, the same clips with
//! the fixed receive-side recipe (100 Hz hum at −40 dBFS plus white noise
//! at 25 dB SNR, `unamblify_audio::rx::RxRecipe::fixed`) on the input;
//! and, when a `drops` sibling capture (`unamblify augment --kind drops`,
//! a seeded, fixed pattern of lost frames per utterance) holds a clip,
//! `eval/lsd_drops`.
//!
//! The clip list is loaded once per mode of the run (`[data] modes`):
//! clip `<clip>` captured in mode `m` is the eval clip `<clip>@<m>`, and
//! with more than one mode in the run every whole-clip column is also
//! reported per mode as `eval/<column>@<mode>` (`eval/lsd@dstar`, …) while
//! the plain columns stay the mean over every clip.
//!
//! Every eval step that coincides with a checkpoint also renders
//! `checkpoints/step-N/audio/<clip>@<mode>.{clean,degraded,out}.wav` — all
//! three at 16 kHz and sample-aligned, the degraded input upsampled for
//! the purpose — plus `<clip>@<mode>.spec.json`: three 80-bin log-mel
//! spectrograms on the shared axes of [`unamblify_audio::Spec`].

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use tch::{Device, Tensor};
use unamblify::aug::AugKind;
use unamblify::{Split, VocoderMode};
use unamblify_audio::rx::RxRecipe;
use unamblify_audio::{
    Spec, cpp_excess, hnr_excess_stats, lsd, mel_l1, plosive_excess, resample, si_sdr,
    write_wav_s16,
};

use crate::data::pipeline::{join_manifests_modes, load_utterance, read_lag};
use crate::data::{Utterance, garbage_tail, synthetic};
use crate::model::{HOP, Net};
use crate::rng::Rng;

/// Garbage tail appended to every eval clip, seconds.
pub const TAIL_S: f32 = 1.0;
/// One second at 16 kHz.
const SEC16: usize = 16_000;

/// The file stem of a key; see [`unamblify::key::clip_name`].
pub use unamblify::key::clip_name;

/// `<clip>@<mode>`: the name of a clip rendered for one mode.
#[must_use]
pub fn clip_mode_name(key: &str, mode: VocoderMode) -> String {
    format!("{}@{mode}", clip_name(key))
}

/// One dev clip in one mode.
#[derive(Debug, Clone)]
pub struct EvalClip {
    /// Filesystem-safe name (`vctk_p228_003_mic2@dstar`).
    pub name: String,
    /// The mode the degraded side was captured in.
    pub mode: VocoderMode,
    /// Its index in the run's modes (what the model is told).
    pub mode_idx: u8,
    /// The aligned pair.
    pub utt: Utterance,
    /// The same utterance's decoded audio from the `drops` sibling
    /// capture, lag-aligned, when that sibling holds it.
    pub drops_deg8: Option<Vec<f32>>,
    /// That variant's erasure mask at the feature-frame rate, lag-aligned
    /// like its audio; what a net built with `[model] erasure_in` is
    /// handed for it. Empty when no frame was lost.
    pub drops_erasure: Vec<u8>,
}

/// The whole-clip columns of one mode.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
pub struct ModeMetrics {
    /// `eval/lsd@<mode>`, dB.
    pub lsd: f64,
    /// `eval/mel@<mode>`, dB.
    pub mel: f64,
    /// `eval/sisdr@<mode>`, dB.
    pub sisdr: f64,
    /// `eval/lsd_first1s@<mode>`.
    pub lsd_first1s: f64,
    /// `eval/lsd_last1s@<mode>`.
    pub lsd_last1s: f64,
    /// `eval/periodicity@<mode>`: how much buzzier than clean this
    /// mode's output is. The modes differ a lot — AMBE leaves far more
    /// excess periodicity than Codec 2 — so this is worth splitting.
    pub periodicity: f64,
    /// `eval/hnr@<mode>`, dB.
    pub hnr: f64,
    /// `eval/hnr_abs@<mode>`, dB.
    pub hnr_abs: f64,
    /// `eval/plosive_burst@<mode>`, dB; absent without a burst.
    pub plosive_burst: Option<f64>,
    /// `eval/plosive_closure@<mode>`, dB.
    pub plosive_closure: Option<f64>,
    /// `eval/plosive_rise@<mode>`, ms.
    pub plosive_rise: Option<f64>,
    /// Bursts behind the plosive columns.
    pub bursts: usize,
    /// Clips of the mode averaged.
    pub clips: usize,
}

impl ModeMetrics {
    fn add(&mut self, m: &EvalMetrics) {
        self.lsd += m.lsd;
        self.mel += m.mel;
        self.sisdr += m.sisdr;
        self.lsd_first1s += m.lsd_first1s;
        self.lsd_last1s += m.lsd_last1s;
        self.periodicity += m.periodicity;
        self.hnr += m.hnr;
        self.hnr_abs += m.hnr_abs;
        self.clips += 1;
        add_plosives(
            &mut self.plosive_burst,
            &mut self.plosive_closure,
            &mut self.plosive_rise,
            &mut self.bursts,
            m,
        );
    }

    fn finish(&mut self) {
        if self.clips > 0 {
            #[allow(clippy::cast_precision_loss)]
            let n = self.clips as f64;
            self.lsd /= n;
            self.mel /= n;
            self.sisdr /= n;
            self.lsd_first1s /= n;
            self.lsd_last1s /= n;
            self.periodicity /= n;
            self.hnr /= n;
            self.hnr_abs /= n;
        }
        finish_plosives(
            &mut self.plosive_burst,
            &mut self.plosive_closure,
            &mut self.plosive_rise,
            self.bursts,
        );
    }
}

/// Accumulate one clip's plosive columns, weighted by its burst count,
/// so a clip with ten stops counts ten times a clip with one.
fn add_plosives(
    burst: &mut Option<f64>,
    closure: &mut Option<f64>,
    rise: &mut Option<f64>,
    bursts: &mut usize,
    m: &EvalMetrics,
) {
    if m.bursts == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let w = m.bursts as f64;
    for (acc, v) in [
        (burst, m.plosive_burst),
        (closure, m.plosive_closure),
        (rise, m.plosive_rise),
    ] {
        if let Some(v) = v {
            *acc = Some(acc.unwrap_or(0.0) + v * w);
        }
    }
    *bursts += m.bursts;
}

/// Turn the burst-weighted sums back into means.
fn finish_plosives(
    burst: &mut Option<f64>,
    closure: &mut Option<f64>,
    rise: &mut Option<f64>,
    bursts: usize,
) {
    if bursts == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = bursts as f64;
    for acc in [burst, closure, rise] {
        *acc = acc.map(|v| v / n);
    }
}

/// Averages over the clips.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct EvalMetrics {
    /// `eval/lsd`, dB.
    pub lsd: f64,
    /// `eval/mel`, dB.
    pub mel: f64,
    /// `eval/sisdr`, dB.
    pub sisdr: f64,
    /// `eval/lsd_first1s`.
    pub lsd_first1s: f64,
    /// `eval/lsd_last1s`.
    pub lsd_last1s: f64,
    /// `eval/babble`: mean output frame energy over the garbage tail, dBFS.
    pub babble: f64,
    /// `eval/hnr`: how much further the output's harmonics stand above
    /// the valleys between them than the clean target's do, in dB.
    /// **This is the robotic-quality number**: 0 is as natural as the
    /// reference; a raw D-STAR decode measures several dB above it.
    pub hnr: f64,
    /// `eval/hnr_abs`: the mean *absolute* per-frame error behind
    /// `eval/hnr`, dB. A model that adds a constant share of noise rather
    /// than following the target's breathiness has `hnr` near 0 and a
    /// large `hnr_abs`; only both near 0 is natural.
    pub hnr_abs: f64,
    /// `eval/plosive_burst`: the output's stop-consonant burst peaks
    /// against the clean's, dB, at the bursts found in the clean.
    /// Negative is a quieter burst. Absent when no clip holds a burst.
    pub plosive_burst: Option<f64>,
    /// `eval/plosive_closure`: burst peak over the quietest millisecond
    /// of the closure before it, output minus clean, dB. Negative means
    /// the silence before the burst has been filled in.
    pub plosive_closure: Option<f64>,
    /// `eval/plosive_rise`: burst rise time, output minus clean, ms.
    /// Positive is a smeared onset.
    pub plosive_rise: Option<f64>,
    /// Bursts behind the plosive columns.
    pub bursts: usize,
    /// `eval/periodicity`: cepstral peak prominence of the output minus
    /// that of the clean target. **Positive = buzzier than natural
    /// speech**, which is what a vocoder's leftover robotic quality
    /// measures as. 0 would be exactly as periodic as the reference.
    pub periodicity: f64,
    /// `eval/lsd_rx`: LSD with the fixed receive-side recipe on the input.
    pub lsd_rx: f64,
    /// `eval/lsd_drops`: LSD over the clips whose input came from the
    /// `drops` sibling capture; absent when no clip has one.
    pub lsd_drops: Option<f64>,
    /// Clips averaged.
    pub clips: usize,
    /// Clips that had a `drops` variant.
    pub drops_clips: usize,
    /// The whole-clip columns per mode (`eval/<column>@<mode>`), present
    /// only when the clips span more than one mode.
    pub by_mode: BTreeMap<VocoderMode, ModeMetrics>,
}

impl EvalMetrics {
    /// The metric rows to log.
    #[must_use]
    pub fn rows(&self) -> Vec<(String, f64)> {
        let mut rows = vec![
            ("eval/lsd".to_owned(), self.lsd),
            ("eval/mel".to_owned(), self.mel),
            ("eval/sisdr".to_owned(), self.sisdr),
            ("eval/lsd_first1s".to_owned(), self.lsd_first1s),
            ("eval/lsd_last1s".to_owned(), self.lsd_last1s),
            ("eval/babble".to_owned(), self.babble),
            ("eval/hnr".to_owned(), self.hnr),
            ("eval/hnr_abs".to_owned(), self.hnr_abs),
            ("eval/periodicity".to_owned(), self.periodicity),
            ("eval/lsd_rx".to_owned(), self.lsd_rx),
        ];
        if let Some(d) = self.lsd_drops {
            rows.push(("eval/lsd_drops".to_owned(), d));
        }
        Self::push_plosives(
            &mut rows,
            "",
            self.plosive_burst,
            self.plosive_closure,
            self.plosive_rise,
        );
        for (mode, m) in &self.by_mode {
            rows.push((format!("eval/lsd@{mode}"), m.lsd));
            rows.push((format!("eval/mel@{mode}"), m.mel));
            rows.push((format!("eval/sisdr@{mode}"), m.sisdr));
            rows.push((format!("eval/lsd_first1s@{mode}"), m.lsd_first1s));
            rows.push((format!("eval/lsd_last1s@{mode}"), m.lsd_last1s));
            rows.push((format!("eval/hnr@{mode}"), m.hnr));
            rows.push((format!("eval/hnr_abs@{mode}"), m.hnr_abs));
            rows.push((format!("eval/periodicity@{mode}"), m.periodicity));
            Self::push_plosives(
                &mut rows,
                &format!("@{mode}"),
                m.plosive_burst,
                m.plosive_closure,
                m.plosive_rise,
            );
        }
        rows
    }

    /// The three plosive rows, when there were bursts to measure.
    fn push_plosives(
        rows: &mut Vec<(String, f64)>,
        suffix: &str,
        burst: Option<f64>,
        closure: Option<f64>,
        rise: Option<f64>,
    ) {
        for (k, v) in [
            ("eval/plosive_burst", burst),
            ("eval/plosive_closure", closure),
            ("eval/plosive_rise", rise),
        ] {
            if let Some(v) = v {
                rows.push((format!("{k}{suffix}"), v));
            }
        }
    }

    /// `, dstar 1.23, codec2-3200 1.45` — the per-mode LSDs for a log line.
    #[must_use]
    pub fn per_mode_note(&self) -> String {
        use std::fmt::Write as _;
        self.by_mode.iter().fold(String::new(), |mut out, (m, v)| {
            let _ = write!(out, ", {m} {:.3}", v.lsd);
            out
        })
    }
}

/// Read the eval-clip list (one prepared key per line, `#` comments) and
/// load up to `max_items` of them for each mode of `modes`, clip-major
/// (`<clip>@<mode>` for each mode of a key before the next key). Keys must
/// be in the dev split; a key that is not yet in both manifests for a
/// mode (the list is written before the corpus is captured) is skipped
/// for that mode with a warning on stderr, and it is an error only when
/// no listed key is available in any mode.
pub fn load_clips(
    root: &Path,
    modes: &[VocoderMode],
    list: &Path,
    max_items: usize,
) -> anyhow::Result<Vec<EvalClip>> {
    anyhow::ensure!(!modes.is_empty(), "no modes to load eval clips for");
    let text = std::fs::read_to_string(list).with_context(|| list.display().to_string())?;
    let keys: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let items = join_manifests_modes(root, modes, &[None])?;
    let mut lags = BTreeMap::new();
    let mut drops_items = Vec::new();
    for &mode in modes {
        lags.insert(mode, read_lag(root, mode)?);
        // The drops sibling, when it exists: the same clips with a fixed,
        // seeded pattern of lost frames, for the `eval/lsd_drops` column.
        if crate::data::pipeline::captured_manifest(root, mode, Some(AugKind::Drops)).is_file() {
            drops_items.extend(join_manifests_modes(
                root,
                &[mode],
                &[Some(AugKind::Drops)],
            )?);
        }
    }
    let mut clips = Vec::with_capacity(keys.len().min(max_items) * modes.len());
    let mut skipped = 0usize;
    let mut loaded: BTreeMap<VocoderMode, usize> = BTreeMap::new();
    for key in &keys {
        for (mi, &mode) in modes.iter().enumerate() {
            if loaded.get(&mode).copied().unwrap_or(0) >= max_items {
                continue;
            }
            let Some(item) = items.iter().find(|i| i.key == *key && i.mode == mode) else {
                eprintln!("[warn] eval clip {key}: not in both manifests for {mode}, skipped");
                skipped += 1;
                continue;
            };
            anyhow::ensure!(
                item.split == Split::Dev,
                "eval clip {key} is in the {} split, not dev",
                item.split
            );
            let lag = lags[&mode];
            let mut utt = load_utterance(root, item, lag)?;
            let mode_idx = u8::try_from(mi)?;
            utt.mode = mode_idx;
            let (drops_deg8, drops_erasure) =
                match drops_items.iter().find(|i| i.key == *key && i.mode == mode) {
                    Some(d) => {
                        let deg8 = load_utterance(root, d, lag)?.deg8;
                        let fs = mode.frame_samples();
                        let frames = deg8.len().div_ceil(fs);
                        let stored =
                            unamblify::channel::erasure_mask(&d.erased, 0, frames, fs, lag, frames);
                        let mask = crate::data::shards::spread_erasure(&stored, mode, deg8.len());
                        (Some(deg8), mask)
                    }
                    None => (None, Vec::new()),
                };
            clips.push(EvalClip {
                name: clip_mode_name(key, mode),
                mode,
                mode_idx,
                utt,
                drops_deg8,
                drops_erasure,
            });
            *loaded.entry(mode).or_insert(0) += 1;
        }
    }
    anyhow::ensure!(
        !clips.is_empty(),
        "none of the {} eval clips in {} is captured for {} ({skipped} skipped)",
        keys.len(),
        list.display(),
        modes
            .iter()
            .map(|m| m.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(clips)
}

/// `n` synthetic dev clips for smoke runs (one mode, `dstar`, index 0).
pub fn synthetic_clips(n: usize, dur_s: f32) -> anyhow::Result<Vec<EvalClip>> {
    (0..n)
        .map(|i| {
            Ok(EvalClip {
                name: format!("synthetic_{i:02}"),
                mode: VocoderMode::Dstar,
                mode_idx: 0,
                utt: synthetic::utterance(900 + i as u64, dur_s)?,
                drops_deg8: None,
                drops_erasure: Vec::new(),
            })
        })
        .collect()
}

/// `n` synthetic clips per mode of `modes`, named `synthetic_NN@<mode>`,
/// each mode's index its position in `modes`.
pub fn synthetic_clips_modes(
    n: usize,
    dur_s: f32,
    modes: &[VocoderMode],
) -> anyhow::Result<Vec<EvalClip>> {
    let mut out = Vec::with_capacity(n * modes.len());
    for i in 0..n {
        for (mi, &mode) in modes.iter().enumerate() {
            let mode_idx = u8::try_from(mi)?;
            let mut utt = synthetic::utterance(900 + i as u64, dur_s)?;
            utt.mode = mode_idx;
            out.push(EvalClip {
                name: format!("synthetic_{i:02}@{mode}"),
                mode,
                mode_idx,
                utt,
                drops_deg8: None,
                drops_erasure: Vec::new(),
            });
        }
    }
    Ok(out)
}

/// A rendered clip.
#[derive(Debug)]
pub struct Rendered {
    /// Clean target with the zero tail, 16 kHz.
    pub clean16: Vec<f32>,
    /// Degraded input with the garbage tail, 8 kHz.
    pub deg8: Vec<f32>,
    /// Model output, 16 kHz.
    pub out16: Vec<f32>,
    /// Samples of real speech at 16 kHz before the tail.
    pub speech_len16: usize,
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    ms.sqrt()
}

/// Run the model over one whole 8 kHz signal at once, no gradient: the
/// input is zero-padded to a whole number of model hops and the 16 kHz
/// output (twice the padded length) comes back on the host. `mode_idx` is
/// what a mode-conditioned model is told. This is the one forward pass
/// eval rendering and `unamblify infer` share.
pub fn forward_whole(
    net: &Net,
    deg8: &[f32],
    mode_idx: u8,
    device: Device,
) -> anyhow::Result<Vec<f32>> {
    forward_whole_with(net, deg8, mode_idx, &[], device)
}

/// [`forward_whole`] with the clip's erasure mask at the feature-frame
/// rate (empty: nothing was lost). Shorter than the clip, it is padded
/// with zeros — a garbage tail appended after the speech lost no frames.
pub fn forward_whole_with(
    net: &Net,
    deg8: &[f32],
    mode_idx: u8,
    erasure: &[u8],
    device: Device,
) -> anyhow::Result<Vec<f32>> {
    let hop = usize::try_from(HOP)?;
    let len8 = deg8.len().div_ceil(hop) * hop;
    let mut padded = deg8.to_vec();
    padded.resize(len8, 0.0);
    let x = Tensor::from_slice(&padded)
        .view([1, 1, i64::try_from(padded.len())?])
        .to_device(device);
    let m = Tensor::from_slice(&[i64::from(mode_idx)]).to_device(device);
    let mask = (net.takes_erasure() && erasure.iter().any(|&v| v != 0)).then(|| {
        let mut m: Vec<f32> = erasure.iter().map(|&v| f32::from(v.min(1))).collect();
        m.resize(len8 / hop, 0.0);
        Tensor::from_slice(&m).view([1, -1]).to_device(device)
    });
    let out = tch::no_grad(|| net.forward_with(&x, &m, mask.as_ref()))
        .to_device(Device::Cpu)
        .view([-1]);
    Ok(Vec::<f32>::try_from(&out)?)
}

/// Run the model over a whole clip plus its garbage tail.
pub fn render(net: &Net, clip: &EvalClip, device: Device, seed: u64) -> anyhow::Result<Rendered> {
    render_input(
        net,
        &clip.utt.clean16,
        &clip.utt.deg8,
        clip.mode_idx,
        device,
        seed,
    )
}

/// [`render`] over a given degraded input (a clip's base capture, or its
/// `drops` sibling) against the clip's clean target, as mode `mode_idx`.
pub fn render_input(
    net: &Net,
    clean: &[f32],
    deg: &[f32],
    mode_idx: u8,
    device: Device,
    seed: u64,
) -> anyhow::Result<Rendered> {
    render_input_with(net, clean, deg, mode_idx, &[], device, seed)
}

/// [`render_input`] with the input's erasure mask (see
/// [`forward_whole_with`]).
pub fn render_input_with(
    net: &Net,
    clean: &[f32],
    deg: &[f32],
    mode_idx: u8,
    erasure: &[u8],
    device: Device,
    seed: u64,
) -> anyhow::Result<Rendered> {
    let hop = usize::try_from(HOP)?;
    let len8 = deg.len().min(clean.len() / 2);
    let len8 = len8.div_ceil(hop) * hop;
    let mut deg8 = deg.to_vec();
    deg8.resize(len8, 0.0);
    let mut clean16 = clean.to_vec();
    clean16.resize(2 * len8, 0.0);
    let speech_len16 = 2 * len8;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let tail8 = ((TAIL_S * 8000.0) as usize).div_ceil(hop) * hop;
    let mut rng = Rng::new(seed ^ 0x5EED);
    let last = deg8[len8.saturating_sub(160)..].to_vec();
    let garbage = garbage_tail(&mut rng, tail8, rms(&deg8), &last);
    deg8.extend_from_slice(&garbage);
    clean16.resize(clean16.len() + 2 * tail8, 0.0);

    let out16 = forward_whole_with(net, &deg8, mode_idx, erasure, device)?;
    Ok(Rendered {
        clean16,
        deg8,
        out16,
        speech_len16,
    })
}

/// The clip's degraded input with the fixed receive-side recipe added
/// (seeded by `seed` for the white noise).
#[must_use]
pub fn rx_input(deg8: &[f32], seed: u64) -> Vec<f32> {
    let mut x = deg8.to_vec();
    unamblify_audio::rx::apply(&mut x, &RxRecipe::fixed(), &mut Rng::new(seed ^ 0x5258));
    x
}

/// Mean frame energy in dBFS over `x` (10 ms frames).
fn energy_db(x: &[f32]) -> f64 {
    let frames: Vec<f64> = x
        .chunks(160)
        .map(|f| {
            #[allow(clippy::cast_precision_loss)]
            let e = f.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / f.len() as f64;
            10.0 * (e + 1e-10).log10()
        })
        .collect();
    if frames.is_empty() {
        return -100.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = frames.len() as f64;
    frames.iter().sum::<f64>() / n
}

/// Metrics of one rendered clip.
#[must_use]
pub fn measure(r: &Rendered) -> EvalMetrics {
    let n = r.speech_len16.min(r.out16.len()).min(r.clean16.len());
    let (out, clean) = (&r.out16[..n], &r.clean16[..n]);
    let first = n.min(SEC16);
    let last_start = n.saturating_sub(SEC16);
    let tail = &r.out16[n..];
    let s = si_sdr(out, clean);
    let hnr = hnr_excess_stats(out, clean, 16_000);
    let plosive = plosive_excess(out, clean, 16_000);
    EvalMetrics {
        lsd: f64::from(lsd(out, clean, 16_000)),
        mel: f64::from(mel_l1(out, clean, 16_000)),
        sisdr: if s.is_finite() { f64::from(s) } else { 0.0 },
        lsd_first1s: f64::from(lsd(&out[..first], &clean[..first], 16_000)),
        lsd_last1s: f64::from(lsd(&out[last_start..], &clean[last_start..], 16_000)),
        babble: energy_db(tail),
        periodicity: f64::from(cpp_excess(out, clean, 16_000)),
        hnr: hnr.map_or(0.0, |h| f64::from(h.mean)),
        hnr_abs: hnr.map_or(0.0, |h| f64::from(h.abs)),
        plosive_burst: plosive.map(|p| f64::from(p.burst_db)),
        plosive_closure: plosive.map(|p| f64::from(p.closure_db)),
        plosive_rise: plosive.map(|p| f64::from(p.rise_ms)),
        bursts: plosive.map_or(0, |p| p.bursts),
        lsd_rx: 0.0,
        lsd_drops: None,
        clips: 1,
        drops_clips: 0,
        by_mode: BTreeMap::new(),
    }
}

/// Write the three WAVs and `spec.json` for one clip.
pub fn write_audio(dir: &Path, name: &str, r: &Rendered) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let deg16 = resample(&r.deg8, 8_000, 16_000)?;
    write_wav_s16(dir.join(format!("{name}.clean.wav")), &r.clean16, 16_000)?;
    write_wav_s16(dir.join(format!("{name}.degraded.wav")), &deg16, 16_000)?;
    write_wav_s16(dir.join(format!("{name}.out.wav")), &r.out16, 16_000)?;
    let spec = Spec::new()
        .with("clean", &r.clean16)
        .with("degraded", &deg16)
        .with("out", &r.out16)
        .with_speech_samples(r.speech_len16);
    std::fs::write(
        dir.join(format!("{name}.spec.json")),
        serde_json::to_string(&spec)?,
    )?;
    Ok(())
}

/// Evaluate every clip; render audio under `audio_dir` when given. The
/// plain columns average every clip; with clips of more than one mode
/// the whole-clip columns are also averaged per mode into `by_mode`.
pub fn evaluate(
    net: &Net,
    clips: &[EvalClip],
    device: Device,
    audio_dir: Option<&Path>,
) -> anyhow::Result<EvalMetrics> {
    let mut acc = EvalMetrics::default();
    let mut by_mode: BTreeMap<VocoderMode, ModeMetrics> = BTreeMap::new();
    for (i, clip) in clips.iter().enumerate() {
        let r = render(net, clip, device, i as u64)?;
        let m = measure(&r);
        acc.lsd += m.lsd;
        acc.mel += m.mel;
        acc.sisdr += m.sisdr;
        acc.lsd_first1s += m.lsd_first1s;
        acc.lsd_last1s += m.lsd_last1s;
        acc.babble += m.babble;
        acc.periodicity += m.periodicity;
        acc.hnr += m.hnr;
        acc.hnr_abs += m.hnr_abs;
        acc.clips += 1;
        add_plosives(
            &mut acc.plosive_burst,
            &mut acc.plosive_closure,
            &mut acc.plosive_rise,
            &mut acc.bursts,
            &m,
        );
        by_mode.entry(clip.mode).or_default().add(&m);
        if let Some(dir) = audio_dir {
            write_audio(dir, &clip.name, &r)?;
        }
        let rx = render_input(
            net,
            &clip.utt.clean16,
            &rx_input(&clip.utt.deg8, i as u64),
            clip.mode_idx,
            device,
            i as u64,
        )?;
        acc.lsd_rx += measure(&rx).lsd;
        if let Some(deg) = &clip.drops_deg8 {
            let rd = render_input_with(
                net,
                &clip.utt.clean16,
                deg,
                clip.mode_idx,
                &clip.drops_erasure,
                device,
                i as u64,
            )?;
            let md = measure(&rd);
            acc.lsd_drops = Some(acc.lsd_drops.unwrap_or(0.0) + md.lsd);
            acc.drops_clips += 1;
        }
    }
    if acc.clips > 0 {
        #[allow(clippy::cast_precision_loss)]
        let n = acc.clips as f64;
        acc.lsd /= n;
        acc.mel /= n;
        acc.sisdr /= n;
        acc.lsd_first1s /= n;
        acc.lsd_last1s /= n;
        acc.babble /= n;
        acc.periodicity /= n;
        acc.hnr /= n;
        acc.hnr_abs /= n;
        acc.lsd_rx /= n;
    }
    finish_plosives(
        &mut acc.plosive_burst,
        &mut acc.plosive_closure,
        &mut acc.plosive_rise,
        acc.bursts,
    );
    if acc.drops_clips > 0 {
        #[allow(clippy::cast_precision_loss)]
        let n = acc.drops_clips as f64;
        acc.lsd_drops = acc.lsd_drops.map(|d| d / n);
    }
    if by_mode.len() > 1 {
        for m in by_mode.values_mut() {
            m.finish();
        }
        acc.by_mode = by_mode;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use unamblify::Profile;

    #[test]
    fn evaluation_renders_aligned_files_and_finite_metrics() {
        let tmp = tempfile::tempdir().unwrap();
        let vs = VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, None);
        let mut clips = synthetic_clips(2, 1.2).unwrap();
        let m = evaluate(&net, &clips, Device::Cpu, Some(tmp.path())).unwrap();
        assert_eq!(m.clips, 2);
        assert!(m.lsd.is_finite() && m.lsd > 0.0, "{m:?}");
        assert!(m.lsd_first1s.is_finite() && m.lsd_last1s.is_finite());
        assert!(m.babble > -100.0 && m.babble < 0.0, "{m:?}");
        assert_eq!(m.lsd_drops, None, "no drops sibling, no column");
        // Ten whole-clip columns, plus the three plosive ones when the
        // synthetic clips happened to hold a burst.
        let plosive_rows = if m.bursts > 0 { 3 } else { 0 };
        assert_eq!(m.rows().len(), 10 + plosive_rows);
        assert!(
            m.hnr_abs >= m.hnr.abs() - 1e-6,
            "abs bounds the mean: {m:?}"
        );
        assert!(m.by_mode.is_empty(), "one mode: no per-mode columns");
        assert!(m.lsd_rx.is_finite() && m.lsd_rx > 0.0, "{m:?}");
        assert!((m.lsd_rx - m.lsd).abs() > 1e-6, "the rx input differs");
        assert!(m.rows().iter().any(|(k, _)| k == "eval/lsd_rx"));
        let noised = rx_input(&clips[0].utt.deg8, 0);
        assert_eq!(noised.len(), clips[0].utt.deg8.len());
        assert_ne!(noised, clips[0].utt.deg8);
        assert_eq!(noised, rx_input(&clips[0].utt.deg8, 0), "fixed and seeded");
        // A clip with a drops variant (here: every fifth frame muted)
        // adds the column, averaged over the clips that have one.
        let mut dropped = clips[0].utt.deg8.clone();
        for (i, v) in dropped.iter_mut().enumerate() {
            if (i / 160) % 5 == 0 {
                *v = 0.0;
            }
        }
        clips[0].drops_deg8 = Some(dropped);
        let m2 = evaluate(&net, &clips, Device::Cpu, None).unwrap();
        assert_eq!(m2.drops_clips, 1);
        let d = m2.lsd_drops.expect("drops column");
        assert!(d.is_finite() && d > 0.0);
        assert!(
            m2.rows()
                .iter()
                .any(|(k, v)| k == "eval/lsd_drops" && (*v - d).abs() < 1e-12),
            "{:?}",
            m2.rows()
        );
        assert!(
            (m2.lsd - m.lsd).abs() < 1e-9,
            "the base column is unchanged"
        );
        for name in ["synthetic_00", "synthetic_01"] {
            let clean =
                unamblify_audio::read_wav(tmp.path().join(format!("{name}.clean.wav"))).unwrap();
            let deg =
                unamblify_audio::read_wav(tmp.path().join(format!("{name}.degraded.wav"))).unwrap();
            let out =
                unamblify_audio::read_wav(tmp.path().join(format!("{name}.out.wav"))).unwrap();
            assert_eq!((clean.rate, deg.rate, out.rate), (16_000, 16_000, 16_000));
            assert_eq!(clean.samples.len(), out.samples.len());
            assert_eq!(clean.samples.len(), deg.samples.len());
            // 1.2 s → 1.2 s speech + 1 s tail = 2.2 s.
            assert_eq!(clean.samples.len(), 35_200);
            let spec: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(tmp.path().join(format!("{name}.spec.json"))).unwrap(),
            )
            .unwrap();
            let frames = spec["frames"].as_u64().unwrap();
            assert_eq!(frames, 1 + 35_200 / 256);
            for k in ["clean", "degraded", "out"] {
                let rows = spec[k].as_array().unwrap();
                assert_eq!(rows.len() as u64, frames, "{k}");
                assert_eq!(rows[0].as_array().unwrap().len(), 80, "{k}");
            }
            // The tail of the clean target is silence.
            let tail = &clean.samples[19_200..];
            assert!(tail.iter().all(|&v| v == 0.0));
        }
        assert_eq!(clip_name("vctk/p228_003_mic2"), "vctk_p228_003_mic2");
        assert_eq!(
            clip_mode_name("vctk/p228_003_mic2", VocoderMode::Codec2_3200),
            "vctk_p228_003_mic2@codec2-3200"
        );
    }

    /// Two modes of clips: the plain columns are the mean over every
    /// clip, each whole-clip column also appears per mode, the rendered
    /// files carry the mode, and a mode-conditioned model is told the
    /// clip's mode (so the same audio scores differently per mode).
    #[test]
    fn two_modes_add_per_mode_columns_and_mode_named_files() {
        use VocoderMode::{Codec2_3200, Dstar};
        let tmp = tempfile::tempdir().unwrap();
        let vs = VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, Some(2));
        let clips = synthetic_clips_modes(2, 1.0, &[Dstar, Codec2_3200]).unwrap();
        assert_eq!(clips.len(), 4);
        assert_eq!(clips[1].name, "synthetic_00@codec2-3200");
        assert_eq!((clips[1].mode, clips[1].mode_idx), (Codec2_3200, 1));
        let m = evaluate(&net, &clips, Device::Cpu, Some(tmp.path())).unwrap();
        assert_eq!(m.clips, 4);
        assert_eq!(m.by_mode.len(), 2);
        let (d, c) = (m.by_mode[&Dstar], m.by_mode[&Codec2_3200]);
        assert_eq!((d.clips, c.clips), (2, 2));
        assert!(
            (m.lsd - f64::midpoint(d.lsd, c.lsd)).abs() < 1e-9,
            "the mean over every clip"
        );
        assert!((m.mel - f64::midpoint(d.mel, c.mel)).abs() < 1e-9);
        assert!(
            (d.lsd - c.lsd).abs() > 1e-9,
            "the embedding makes the modes score differently: {d:?} {c:?}"
        );
        let rows = m.rows();
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        for k in [
            "eval/lsd",
            "eval/lsd@dstar",
            "eval/lsd@codec2-3200",
            "eval/mel@dstar",
            "eval/sisdr@codec2-3200",
            "eval/lsd_first1s@dstar",
            "eval/lsd_last1s@codec2-3200",
        ] {
            assert!(keys.contains(&k), "missing {k} in {keys:?}");
        }
        let plosive_rows = |bursts: usize| if bursts > 0 { 3 } else { 0 };
        let per_mode: usize = m.by_mode.values().map(|v| 8 + plosive_rows(v.bursts)).sum();
        assert_eq!(rows.len(), 10 + plosive_rows(m.bursts) + per_mode);
        let lsd_c2 = rows
            .iter()
            .find(|(k, _)| k == "eval/lsd@codec2-3200")
            .unwrap()
            .1;
        assert!((lsd_c2 - c.lsd).abs() < 1e-12);
        assert!(m.per_mode_note().contains(", codec2-3200 "));
        for name in [
            "synthetic_00@dstar",
            "synthetic_00@codec2-3200",
            "synthetic_01@codec2-3200",
        ] {
            assert!(
                tmp.path().join(format!("{name}.out.wav")).is_file(),
                "{name}"
            );
            assert!(
                tmp.path().join(format!("{name}.spec.json")).is_file(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_perfect_model_scores_perfectly_on_the_speech_part() {
        let clip = &synthetic_clips(1, 1.0).unwrap()[0];
        let clean = clip.utt.clean16.clone();
        let r = Rendered {
            clean16: clean.clone(),
            deg8: vec![0.0; 8_000],
            out16: clean,
            speech_len16: 16_000,
        };
        let m = measure(&r);
        assert!(m.lsd.abs() < 1e-5 && m.mel.abs() < 1e-5, "{m:?}");
        assert!(m.sisdr > 50.0, "{m:?}");
        assert!((m.babble + 100.0).abs() < 1e-6, "{m:?}");
    }
}
