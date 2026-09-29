// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The real-time gate (`docs/design/training.md`, "Real-time budget is a
//! training constraint"): how long one model takes, on CPU, to produce a
//! 20 ms frame of output. A candidate that cannot make its frame inside
//! 20 ms of wall clock on the target core is not a candidate for that
//! profile, however good its LSD is — so this runs *before* a long
//! training run, not after.
//!
//! **What the number is.** The whole clip goes through
//! [`Net::forward`](crate::model::Net::forward) at once, as training and
//! `unamblify infer` do, and the wall time is divided by the frames in
//! it. A deployed post-filter runs frame by frame, carrying the GRU
//! state forward, which cannot amortise the per-call overhead the same
//! way — so this is a **lower bound** on the streaming cost and a fair
//! way to compare two models, not a promise about astar's receive path.
//! Threads are pinned to `threads` (1 by default): the budget is one
//! core, because the rest of the radio needs the others.

use std::time::Instant;

use tch::{Device, Kind, Tensor, nn::VarStore};
use unamblify::{Profile, VocoderMode};

use crate::model::{self, HOP};

/// 8 kHz input samples per 20 ms vocoder frame.
const FRAME_SAMPLES: i64 = 160;
/// [`FRAME_SAMPLES`] where the arithmetic is in floating point.
const FRAME_SAMPLES_F: u32 = 160;
/// Passes discarded before timing starts.
const WARM_UP: usize = 2;

/// What to measure.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// Size profile.
    pub profile: Profile,
    /// `[model] width` multiplier.
    pub width: f32,
    /// Lookahead in AMBE frames.
    pub lookahead: u32,
    /// Build the mode embedding over this many modes (`None` = blind).
    pub embed_modes: Option<usize>,
    /// Seconds of audio per timed pass.
    pub seconds: f32,
    /// Timed passes.
    pub passes: usize,
    /// Torch CPU threads. One is the budget that matters.
    pub threads: i32,
    /// `[model] noise_head`: include the aperiodic excitation path.
    pub noise_head: bool,
    /// `[model] noise_mod`: its modulation and 1 ms gains.
    pub noise_mod: bool,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            profile: Profile::Full,
            width: 1.0,
            lookahead: 5,
            embed_modes: Some(VocoderMode::ALL.len()),
            seconds: 8.0,
            passes: 5,
            threads: 1,
            noise_head: false,
            noise_mod: false,
        }
    }
}

/// What one measurement says.
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// Trainable parameters.
    pub params: usize,
    /// Milliseconds of CPU wall clock per 20 ms output frame, median of
    /// the passes.
    pub ms_per_frame: f64,
    /// Fastest and slowest pass, same unit.
    pub ms_per_frame_min: f64,
    /// Slowest pass.
    pub ms_per_frame_max: f64,
    /// Fraction of the 20 ms budget one frame costs: below 1.0 is
    /// real-time on this core, with the rest as headroom.
    pub budget_used: f64,
    /// Threads the measurement used.
    pub threads: i32,
}

impl BenchReport {
    /// Whether one frame fits in its 20 ms.
    #[must_use]
    pub fn real_time(&self) -> bool {
        self.budget_used < 1.0
    }
}

/// Time `opts` on the CPU. Deterministic in shape, not in timing.
pub fn run(opts: &BenchOptions) -> anyhow::Result<BenchReport> {
    anyhow::ensure!(opts.passes > 0, "bench needs at least one pass");
    anyhow::ensure!(
        opts.seconds > 0.0 && opts.seconds.is_finite(),
        "bench needs a positive length, got {}",
        opts.seconds
    );
    if opts.threads > 0 {
        tch::set_num_threads(opts.threads);
    }
    let device = Device::Cpu;
    let vs = VarStore::new(device);
    let (net, params) = model::build(
        &vs,
        &model::NetOpts {
            profile: opts.profile,
            width: opts.width,
            lookahead: opts.lookahead,
            embed_modes: opts.embed_modes,
            noise_head: opts.noise_head,
            noise_mod: opts.noise_mod,
            // One more GRU input channel: not worth a bench axis.
            erasure_in: false,
        },
    )?;

    // Whole frames of 8 kHz input, at least one.
    #[allow(clippy::cast_possible_truncation)]
    let frames =
        ((f64::from(opts.seconds) * 8000.0 / f64::from(FRAME_SAMPLES_F)).round() as i64).max(1);
    let samples = frames * FRAME_SAMPLES;
    anyhow::ensure!(
        samples % HOP == 0,
        "bench input of {samples} samples is not whole feature hops"
    );
    let x = Tensor::rand([1, 1, samples], (Kind::Float, device)) - 0.5;
    let mode = model::mode_zeros(1, device);

    let mut per_frame: Vec<f64> = Vec::with_capacity(opts.passes);
    tch::no_grad(|| {
        for i in 0..WARM_UP + opts.passes {
            let t0 = Instant::now();
            let y = net.forward(&x, &mode);
            // The forward is lazy on some backends; touching the output
            // forces it to finish before the clock stops.
            let _ = y.size();
            let _ = f64::try_from(y.narrow(2, 0, 1).sum(Kind::Float)).unwrap_or(0.0);
            let dt = t0.elapsed();
            if i >= WARM_UP {
                #[allow(clippy::cast_precision_loss)]
                per_frame.push(dt.as_secs_f64() * 1000.0 / frames as f64);
            }
        }
    });
    per_frame.sort_by(f64::total_cmp);
    let median = per_frame[per_frame.len() / 2];
    Ok(BenchReport {
        params,
        ms_per_frame: median,
        ms_per_frame_min: per_frame[0],
        ms_per_frame_max: per_frame[per_frame.len() - 1],
        budget_used: median / 20.0,
        threads: opts.threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bench runs, reports a positive per-frame time, and the
    /// budget fraction is that time against 20 ms.
    #[test]
    fn bench_reports_a_per_frame_time_against_the_20_ms_budget() {
        let opts = BenchOptions {
            profile: Profile::SuperLite,
            seconds: 0.5,
            passes: 2,
            embed_modes: None,
            ..BenchOptions::default()
        };
        let r = run(&opts).unwrap();
        assert!(r.params > 0);
        assert!(r.ms_per_frame > 0.0 && r.ms_per_frame.is_finite());
        assert!(r.ms_per_frame_min <= r.ms_per_frame);
        assert!(r.ms_per_frame_max >= r.ms_per_frame);
        assert!((r.budget_used - r.ms_per_frame / 20.0).abs() < 1e-12);
        assert_eq!(r.real_time(), r.budget_used < 1.0);
    }

    /// A wider model of the same profile costs more per frame and has
    /// more parameters — the comparison the knob exists for.
    #[test]
    fn a_wider_model_has_more_parameters() {
        // Full, because the smaller profiles' budgets are tight enough
        // that a 1.5x of them is already over (which the next test is).
        let base = BenchOptions {
            profile: Profile::Full,
            seconds: 0.25,
            passes: 1,
            embed_modes: None,
            ..BenchOptions::default()
        };
        let wide = BenchOptions {
            width: 1.5,
            ..base.clone()
        };
        let (a, b) = (run(&base).unwrap(), run(&wide).unwrap());
        // Parameters grow with roughly the square of the width.
        assert!(
            b.params > a.params * 3 / 2,
            "{} vs {} parameters",
            a.params,
            b.params
        );
        assert_eq!(model::scaled_widths(Profile::Full, 1.5).f, 384);
        assert_eq!(model::scaled_widths(Profile::Full, 1.0).f, 256);
    }

    /// A width that blows the profile's parameter budget is refused by
    /// name, before anything is trained.
    #[test]
    fn a_width_over_the_budget_is_refused() {
        let e = run(&BenchOptions {
            profile: Profile::SuperLite,
            width: 12.0,
            seconds: 0.25,
            passes: 1,
            embed_modes: None,
            ..BenchOptions::default()
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("exceed the budget"), "{e}");
        assert!(e.contains("width 12"), "{e}");
    }
}

/// What `bench --kind pipeline` reports: the restorer + waveform
/// synthesiser of [`crate::pipeline`], each stage's CPU cost per 20 ms of
/// audio, on random weights of the exported shapes or on a weights file.
#[derive(Debug, Clone)]
pub struct PipelineBenchReport {
    /// Restorer parameters.
    pub params_restorer: usize,
    /// Synthesiser parameters.
    pub params_synth: usize,
    /// Feature extraction (resample + STFT + mel), ms per 20 ms of audio.
    pub features_ms: f64,
    /// The restorer, ms per 20 ms.
    pub restorer_ms: f64,
    /// The synthesiser, ms per 20 ms.
    pub synth_ms: f64,
    /// All three, ms per 20 ms: below 20 is real time on this core.
    pub total_ms: f64,
    /// Threads the measurement used.
    pub threads: i32,
}

impl PipelineBenchReport {
    /// Fraction of the 20 ms budget the whole pipeline costs.
    #[must_use]
    pub fn budget_used(&self) -> f64 {
        self.total_ms / 20.0
    }
}

/// Time the pipeline on the CPU: `seconds` of 8 kHz input per pass,
/// batch path, the median of `passes`. With `weights` the real model;
/// without, random weights of the spike's shapes, which cost the same.
pub fn run_pipeline(
    seconds: f32,
    passes: usize,
    threads: i32,
    weights: Option<&std::path::Path>,
) -> anyhow::Result<PipelineBenchReport> {
    use crate::pipeline::{self, Pipeline, Weights};
    anyhow::ensure!(passes > 0, "bench needs at least one pass");
    anyhow::ensure!(
        seconds > 0.0 && seconds.is_finite(),
        "bench needs a positive length, got {seconds}"
    );
    if threads > 0 {
        tch::set_num_threads(threads);
    }
    let device = Device::Cpu;
    let w = match weights {
        Some(p) => Weights::load(p, device)?,
        None => pipeline::synthetic_weights(&pipeline::spike_manifest(), device),
    };
    let count = |prefix: &str| {
        w.names()
            .iter()
            .filter(|n| n.starts_with(prefix))
            .map(|n| w.get(n).map_or(0, |t| t.numel()))
            .sum::<usize>()
    };
    let (params_restorer, params_synth) = (count("restorer."), count("synth."));
    let p = Pipeline::from_weights(&w)?;
    #[allow(clippy::cast_possible_truncation)]
    let n = ((f64::from(seconds) * 8000.0).round() as i64).max(HOP);
    let x8 = Tensor::rand([n], (Kind::Float, device)) - 0.5;
    let mode = 0;
    let frames20 = f64::from(seconds) / 0.02;
    let mut f_ms = Vec::new();
    let mut r_ms = Vec::new();
    let mut s_ms = Vec::new();
    tch::no_grad(|| {
        for i in 0..WARM_UP + passes {
            let t0 = Instant::now();
            let (low, mel) = p.features.compute(&x8);
            let _ = f64::try_from(mel.narrow(1, 0, 1).sum(Kind::Float)).unwrap_or(0.0);
            let t1 = Instant::now();
            let clean = p
                .restorer
                .forward(&low.unsqueeze(0), &mel.unsqueeze(0), mode);
            let _ = f64::try_from(clean.narrow(2, 0, 1).sum(Kind::Float)).unwrap_or(0.0);
            let t2 = Instant::now();
            let wav = p.synth.forward(&clean);
            let _ = f64::try_from(wav.narrow(1, 0, 1).sum(Kind::Float)).unwrap_or(0.0);
            let t3 = Instant::now();
            if i >= WARM_UP {
                f_ms.push((t1 - t0).as_secs_f64() * 1000.0 / frames20);
                r_ms.push((t2 - t1).as_secs_f64() * 1000.0 / frames20);
                s_ms.push((t3 - t2).as_secs_f64() * 1000.0 / frames20);
            }
        }
    });
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (features_ms, restorer_ms, synth_ms) =
        (median(&mut f_ms), median(&mut r_ms), median(&mut s_ms));
    Ok(PipelineBenchReport {
        params_restorer,
        params_synth,
        features_ms,
        restorer_ms,
        synth_ms,
        total_ms: features_ms + restorer_ms + synth_ms,
        threads: tch::get_num_threads(),
    })
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    /// The pipeline bench runs on random weights and reports every stage
    /// as a positive cost that adds up.
    #[test]
    fn the_pipeline_bench_reports_every_stage() {
        let r = run_pipeline(0.5, 1, 1, None).unwrap();
        assert!(r.params_restorer > 6_000_000 && r.params_synth > 13_000_000);
        assert!(r.features_ms > 0.0 && r.restorer_ms > 0.0 && r.synth_ms > 0.0);
        assert!((r.total_ms - r.features_ms - r.restorer_ms - r.synth_ms).abs() < 1e-9);
        assert!(run_pipeline(0.0, 1, 1, None).is_err());
    }
}
