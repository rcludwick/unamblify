// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The step loop (spec §7): builds model, optimiser, loader and eval
//! clips from a `RunConfig`, writes `config.toml` / `status.json` /
//! `metrics.jsonl` / `log.jsonl` into the run directory, checkpoints every
//! `[ckpt] every_steps` (keeping `keep`, the best always), evaluates every
//! `[eval] every_steps`, resumes from a checkpoint directory, and turns
//! SIGINT / SIGTERM into "checkpoint, then exit 0".

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use tch::nn::VarStore;
use unamblify::{Best, DataSource, LogLevel, RunConfig, RunState, RunStatus, Split, VocoderMode};

use crate::checkpoint::{self, Meta};
use crate::data::pipeline::PipelineLoader;
use crate::data::shards::ShardLoader;
use crate::data::synthetic::SyntheticLoader;
use crate::data::{Batch, CropCfg, Loader, RxCfg};
use crate::device::{self, DeviceSpec};
use crate::eval::{self, EvalClip, EvalMetrics};
use crate::losses::{LossCfg, LossInputs, Losses};
use crate::model::{self, Net};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::runio::RunWriter;
use crate::sys::{self, SysSampler};
use crate::time::{rfc3339_now, run_id, unix_ms};

/// Command-line overrides of the config.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overrides {
    /// `--device`.
    pub device: Option<String>,
    /// `--steps`.
    pub steps: Option<u64>,
    /// `--shards`: another shard set name for `[data] shards`.
    pub shards: Option<String>,
    /// `--init-from`: another run's weights to start from
    /// (`[train] init_from`).
    pub init_from: Option<String>,
}

/// Where the examples come from.
#[derive(Debug, Clone, PartialEq)]
pub enum DataSel {
    /// `[data]` of the config, under the data root.
    Config,
    /// Synthetic utterances (smoke).
    Synthetic {
        /// Utterances in the pool.
        n_utts: usize,
        /// Utterance length, seconds.
        dur_s: f32,
    },
}

/// What a finished (or stopped) run reports back.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The run directory.
    pub run_dir: PathBuf,
    /// Last completed step.
    pub step: u64,
    /// Final state.
    pub state: RunState,
    /// `loss/total` per step run in this invocation.
    pub losses: Vec<f64>,
    /// Last eval, if one ran.
    pub last_eval: Option<EvalMetrics>,
    /// `loss/probe` — the loss on one fixed batch drawn at start — at the
    /// start step and at every eval step, in order.
    pub probe: Vec<(u64, f64)>,
}

/// `$UNAMBLIFY_DATA` or the default drive path.
#[must_use]
pub fn data_root() -> PathBuf {
    std::env::var_os("UNAMBLIFY_DATA").map_or_else(
        || PathBuf::from("/Volumes/data/training_data/unamblify"),
        PathBuf::from,
    )
}

/// Status writes while running are at most this far apart.
const STATUS_EVERY: Duration = Duration::from_secs(5);
/// `sys/*` and throughput are sampled every this many steps.
const SYS_EVERY: u64 = 10;
/// Log line cadence.
const LOG_EVERY: u64 = 50;

/// Run a config to completion. `run_dir` defaults to
/// `<data root>/runs/<run_id>`; `resume` names a `checkpoints/step-N`
/// directory.
pub fn run(
    cfg: RunConfig,
    run_dir: Option<&Path>,
    resume: Option<&Path>,
    overrides: &Overrides,
) -> anyhow::Result<Outcome> {
    run_with(cfg, run_dir, resume, overrides, &DataSel::Config)
}

/// [`run`] with an explicit data source.
pub fn run_with(
    mut cfg: RunConfig,
    run_dir: Option<&Path>,
    resume: Option<&Path>,
    overrides: &Overrides,
    data: &DataSel,
) -> anyhow::Result<Outcome> {
    if let Some(d) = &overrides.device {
        cfg.train.device.clone_from(d);
    }
    if let Some(s) = overrides.steps {
        cfg.train.steps = s;
    }
    if let Some(s) = &overrides.shards {
        cfg.data.shards = Some(s.clone());
    }
    if let Some(s) = &overrides.init_from {
        cfg.train.init_from = Some(s.clone());
    }
    let run_dir = run_dir.map_or_else(
        || data_root().join("runs").join(run_id(&cfg.name, unix_ms())),
        Path::to_path_buf,
    );
    std::fs::create_dir_all(&run_dir).with_context(|| run_dir.display().to_string())?;
    std::fs::write(run_dir.join("config.toml"), cfg.to_toml()?)?;
    if let Some(ckpt) = resume {
        // Resuming into a directory that already logged past the
        // checkpoint (the trainer died after it): drop those rows so no
        // step appears twice.
        let step = checkpoint::read_meta(ckpt)?.step;
        let removed = crate::runio::truncate_metrics_after(&run_dir, step)?;
        if removed > 0 {
            let mut w = RunWriter::open(&run_dir)?;
            w.log(
                LogLevel::Warn,
                format!("dropped {removed} metrics rows past step {step} before resuming"),
            )?;
        }
    }
    let mut writer = RunWriter::open(&run_dir)?;
    let started = rfc3339_now();
    let host = sys::hostname();
    let mut status = RunStatus {
        status: RunState::Running,
        step: 0,
        total_steps: cfg.train.steps,
        started,
        updated: rfc3339_now(),
        pid: Some(std::process::id()),
        device: cfg.train.device.clone(),
        host,
        best: None,
    };
    writer.status(&status)?;
    let result = train(&cfg, &run_dir, resume, data, &mut writer, &mut status);
    match &result {
        Ok(outcome) => {
            status.status = outcome.state;
            status.step = outcome.step;
        }
        Err(e) => {
            let _ = writer.log(LogLevel::Error, format!("{e:#}"));
            status.status = RunState::Failed;
        }
    }
    status.pid = None;
    status.updated = rfc3339_now();
    writer.status(&status)?;
    writer.flush()?;
    result
}

fn crop_cfg(cfg: &RunConfig) -> anyhow::Result<CropCfg> {
    Ok(CropCfg {
        mode: cfg.data.primary_mode()?,
        crop_s: cfg.data.crop_s,
        ..CropCfg::default()
    })
}

fn rx_cfg(cfg: &RunConfig) -> RxCfg {
    RxCfg::from(&cfg.augment)
}

/// Open the training loader. A fresh run draws from `[train] seed`; a run
/// resumed at `start_step` draws from a stream derived from `(seed,
/// start_step)`, so it does not replay the batches the checkpoint already
/// trained on. (The exact continuation of the original stream would need
/// the loader's cursor and generator persisted per checkpoint; the derived
/// stream gives new data, which is what matters for the epoch count.)
fn open_loader(
    cfg: &RunConfig,
    data: &DataSel,
    start_step: u64,
) -> anyhow::Result<Box<dyn Loader>> {
    let seed = if start_step == 0 {
        cfg.train.seed
    } else {
        Rng::new(cfg.train.seed).fork(start_step).next_u64()
    };
    let rx = rx_cfg(cfg);
    let modes = cfg.data.modes()?;
    let weights = cfg.data.mode_weights.as_slice();
    anyhow::ensure!(
        weights.is_empty() || weights.len() == modes.len(),
        "[data] mode_weights has {} entries for {} modes",
        weights.len(),
        modes.len()
    );
    Ok(match data {
        DataSel::Synthetic { n_utts, dur_s } => {
            Box::new(SyntheticLoader::new(*n_utts, *dur_s, crop_cfg(cfg)?, seed)?.with_rx(rx, seed))
        }
        DataSel::Config => match cfg.data.source {
            DataSource::Pipeline => {
                let loader = PipelineLoader::open_modes(
                    &data_root(),
                    &modes,
                    &cfg.data.kinds()?,
                    Split::Train,
                    crop_cfg(cfg)?,
                    seed,
                )?;
                let loader = if weights.is_empty() {
                    loader
                } else {
                    loader.with_mode_weights(weights)?
                };
                Box::new(loader.with_rx(rx, seed))
            }
            DataSource::Shards => {
                let name = cfg
                    .data
                    .shards
                    .as_deref()
                    .context("[data] source = \"shards\" needs [data] shards = \"<name>\"")?;
                let loader = ShardLoader::open_modes(
                    &data_root().join("shards").join(name),
                    Split::Train,
                    seed,
                    Some(&modes),
                )?;
                let loader = if weights.is_empty() {
                    loader
                } else {
                    loader.with_mode_weights(weights)?
                };
                Box::new(loader.with_rx(rx, seed))
            }
        },
    })
}

fn open_clips(
    cfg: &RunConfig,
    data: &DataSel,
    writer: &mut RunWriter,
) -> anyhow::Result<Vec<EvalClip>> {
    let max = usize::try_from(cfg.eval.max_items).unwrap_or(usize::MAX);
    match data {
        DataSel::Synthetic { dur_s, .. } => eval::synthetic_clips(max.min(4), *dur_s),
        DataSel::Config => {
            let modes = cfg.data.modes()?;
            match eval::load_clips(&data_root(), &modes, Path::new(&cfg.eval.clips), max) {
                Ok(c) => Ok(c),
                Err(e) => {
                    writer.log(LogLevel::Warn, format!("eval disabled: {e:#}"))?;
                    Ok(Vec::new())
                }
            }
        }
    }
}

fn install_signal_flag() -> anyhow::Result<Arc<AtomicBool>> {
    let flag = Arc::new(AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(sig, Arc::clone(&flag))
            .with_context(|| format!("register signal {sig}"))?;
    }
    Ok(flag)
}

/// Everything the loop needs.
struct Trainer<'a> {
    cfg: &'a RunConfig,
    run_dir: &'a Path,
    writer: &'a mut RunWriter,
    status: &'a mut RunStatus,
    device: DeviceSpec,
    vs: VarStore,
    net: Net,
    adam: Adam,
    losses: Losses,
    loader: Box<dyn Loader>,
    clips: Vec<EvalClip>,
    sys: SysSampler,
    params: usize,
    /// `[data] modes`, in the model's index order.
    modes: Vec<VocoderMode>,
    best: Option<Best>,
    probe: Batch,
    /// The checkpoint directory `[train] init_from` resolved to, for the
    /// provenance line in every checkpoint this run writes.
    init_from: Option<String>,
    /// The step whose checkpoint is already on disk, so a stop right after
    /// a checkpoint step does not write (and replace) it again.
    last_ckpt_step: Option<u64>,
}

impl Trainer<'_> {
    fn ckpt_root(&self) -> PathBuf {
        self.run_dir.join("checkpoints")
    }

    fn meta(&self, step: u64) -> Meta {
        Meta {
            step,
            name: self.cfg.name.clone(),
            profile: self.cfg.model.profile,
            width: self.cfg.model.width,
            lookahead: self.cfg.model.lookahead,
            params: self.params,
            saved_at: rfc3339_now(),
            device: self.cfg.train.device.clone(),
            seed: self.cfg.train.seed,
            modes: self.modes.clone(),
            mode_embed: self.cfg.model.mode_embed,
            noise_head: self.cfg.model.noise_head,
            noise_mod: self.cfg.model.noise_mod,
            best: self.best.clone(),
            init_from: self.init_from.clone(),
        }
    }

    fn write_status(&mut self, step: u64, state: RunState) -> anyhow::Result<()> {
        self.status.status = state;
        self.status.step = step;
        self.status.updated = rfc3339_now();
        self.status.best.clone_from(&self.best);
        self.writer.status(self.status)
    }

    fn checkpoint(&mut self, step: u64) -> anyhow::Result<PathBuf> {
        let root = self.ckpt_root();
        let meta = self.meta(step);
        let dir = checkpoint::save(&root, &self.vs, &self.adam, &meta)?;
        self.last_ckpt_step = Some(step);
        let removed = checkpoint::prune(
            &root,
            self.cfg.ckpt.keep,
            self.best.as_ref().map(|b| b.step),
        )?;
        self.writer.log(
            LogLevel::Info,
            format!(
                "checkpoint {} written{}",
                dir.display(),
                if removed.is_empty() {
                    String::new()
                } else {
                    format!(", pruned {}", removed.len())
                }
            ),
        )?;
        self.writer.flush()?;
        Ok(dir)
    }

    /// SIGINT / SIGTERM: checkpoint (unless this step's checkpoint is
    /// already on disk — a stop that lands right after a checkpoint step
    /// must not replace it) and mark the run stopped.
    fn stop(&mut self, step: u64) -> anyhow::Result<()> {
        if self.last_ckpt_step == Some(step) {
            self.writer.log(
                LogLevel::Warn,
                format!("signal received at step {step}: checkpoint already written, stopping"),
            )?;
        } else {
            self.writer.log(
                LogLevel::Warn,
                format!("signal received at step {step}: checkpointing and stopping"),
            )?;
            self.checkpoint(step)?;
        }
        self.write_status(step, RunState::Stopped)
    }

    fn eval(&mut self, step: u64, audio_dir: Option<&Path>) -> anyhow::Result<Option<EvalMetrics>> {
        if self.clips.is_empty() {
            return Ok(None);
        }
        let t0 = Instant::now();
        let m = eval::evaluate(&self.net, &self.clips, self.device.device, audio_dir)?;
        let rows = m.rows();
        let rows: Vec<(&str, f64)> = rows.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        self.writer.metrics(step, &rows)?;
        let better = self.best.as_ref().is_none_or(|b| m.lsd < b.value);
        if better {
            self.best = Some(Best {
                metric: "eval/lsd".to_owned(),
                value: m.lsd,
                step,
            });
        }
        self.writer.log(
            LogLevel::Info,
            format!(
                "eval step {step}: lsd {:.3} (first1s {:.3}, last1s {:.3}, rx {:.3}{}{}) mel {:.3} sisdr {:.2} babble {:.1} dB hnr {:+.2}/{:.2} dB{} over {} clips in {:.1} s{}",
                m.lsd,
                m.lsd_first1s,
                m.lsd_last1s,
                m.lsd_rx,
                m.lsd_drops
                    .map_or(String::new(), |d| format!(", drops {d:.3}")),
                m.per_mode_note(),
                m.mel,
                m.sisdr,
                m.babble,
                m.hnr,
                m.hnr_abs,
                m.plosive_burst
                    .map_or(String::new(), |b| format!(" plosive {b:+.1} dB")),
                m.clips,
                t0.elapsed().as_secs_f64(),
                if better { " (best)" } else { "" }
            ),
        )?;
        Ok(Some(m))
    }

    fn loss_on(&self, batch: &Batch) -> anyhow::Result<crate::losses::LossTerms> {
        let out16 = self
            .net
            .forward_with(&batch.deg8, &batch.mode, Some(&batch.erasure));
        let out8 = self.net.decimate(&out16);
        let clean8 = self.net.decimate(&batch.clean16);
        self.losses.compute(&LossInputs {
            out16: &out16,
            clean16: &batch.clean16,
            mask16: &batch.mask,
            out8: &out8,
            clean8: &clean8,
            onset: &batch.onset,
            tail: &batch.tail,
        })
    }

    /// Loss on the fixed probe batch (no gradient), logged as `loss/probe`.
    fn probe(&mut self, step: u64) -> anyhow::Result<f64> {
        let v = tch::no_grad(|| self.loss_on(&self.probe))?.values().total;
        self.writer.metric(step, "loss/probe", v)?;
        Ok(v)
    }

    fn train_step(&mut self, step: u64) -> anyhow::Result<f64> {
        let batch = self
            .loader
            .next_batch(usize::try_from(self.cfg.train.batch)?, self.device.device)?;
        let terms = self.loss_on(&batch)?;
        self.adam.zero_grad();
        terms.total.backward();
        let grad_norm = self.adam.step();
        let v = terms.values();
        anyhow::ensure!(v.total.is_finite(), "loss is not finite at step {step}");
        self.writer.metrics(
            step,
            &[
                ("loss/total", v.total),
                ("loss/stft", v.stft),
                ("loss/mel", v.mel),
                ("loss/sisdr", v.sisdr),
                ("loss/onset", v.onset),
                ("loss/tail", v.tail),
                ("loss/periodicity", v.periodicity),
                ("loss/hnr", v.hnr),
                ("loss/hnr_l1", v.hnr_l1),
                ("loss/transient", v.transient),
                ("lr", self.adam.lr),
                ("grad_norm", grad_norm),
            ],
        )?;
        Ok(v.total)
    }
}

/// Every `total_steps` cadence check: multiples of `every`, or the final
/// step.
const fn due(step: u64, every: u64, total: u64) -> bool {
    (every > 0 && step.is_multiple_of(every)) || step == total
}

/// A checkpoint can only continue a config of the same shape: profile,
/// lookahead, mode embedding and (when it recorded them) modes.
fn resume_compatible(meta: &Meta, cfg: &RunConfig, modes: &[VocoderMode]) -> anyhow::Result<()> {
    anyhow::ensure!(
        meta.profile == cfg.model.profile
            && meta.lookahead == cfg.model.lookahead
            && (meta.width - cfg.model.width).abs() < f32::EPSILON,
        "checkpoint is {} x{} ll{}, config is {} x{} ll{}",
        meta.profile,
        meta.width,
        meta.lookahead,
        cfg.model.profile,
        cfg.model.width,
        cfg.model.lookahead
    );
    anyhow::ensure!(
        meta.mode_embed == cfg.model.mode_embed,
        "checkpoint was written {} the mode embedding, config says mode_embed = {}",
        if meta.mode_embed { "with" } else { "without" },
        cfg.model.mode_embed
    );
    anyhow::ensure!(
        meta.noise_head == cfg.model.noise_head,
        "checkpoint was written {} the noise head, config says noise_head = {}",
        if meta.noise_head { "with" } else { "without" },
        cfg.model.noise_head
    );
    anyhow::ensure!(
        meta.noise_mod == cfg.model.noise_mod,
        "checkpoint was written {} noise modulation, config says noise_mod = {}",
        if meta.noise_mod { "with" } else { "without" },
        cfg.model.noise_mod
    );
    anyhow::ensure!(
        meta.modes.is_empty() || meta.modes == modes,
        "checkpoint trained on modes {:?}, config lists {:?}",
        meta.modes.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
        modes.iter().map(|m| m.as_str()).collect::<Vec<_>>()
    );
    Ok(())
}

/// A checkpoint can *initialise* a config of the same model shape whose
/// modes it covers: profile, lookahead and the mode embedding must
/// match, and every mode this run trains on must be one the checkpoint
/// knows (the rows are re-indexed in [`checkpoint::load_init`]). Unlike
/// a resume the mode lists need not be equal — that is the whole point
/// of a specialist.
fn init_compatible(meta: &Meta, cfg: &RunConfig, modes: &[VocoderMode]) -> anyhow::Result<()> {
    anyhow::ensure!(
        meta.profile == cfg.model.profile
            && meta.lookahead == cfg.model.lookahead
            && (meta.width - cfg.model.width).abs() < f32::EPSILON,
        "checkpoint is {} x{} ll{}, config is {} x{} ll{}",
        meta.profile,
        meta.width,
        meta.lookahead,
        cfg.model.profile,
        cfg.model.width,
        cfg.model.lookahead
    );
    anyhow::ensure!(
        meta.mode_embed == cfg.model.mode_embed,
        "checkpoint was written {} the mode embedding, config says mode_embed = {}",
        if meta.mode_embed { "with" } else { "without" },
        cfg.model.mode_embed
    );
    if meta.modes.is_empty() {
        anyhow::ensure!(
            modes.len() == 1,
            "checkpoint does not record its modes (a single-mode run), so it cannot initialise a run over {:?}",
            modes.iter().map(|m| m.as_str()).collect::<Vec<_>>()
        );
    } else {
        for m in modes {
            anyhow::ensure!(
                meta.modes.contains(m),
                "checkpoint trained on modes {:?}, which do not include {}",
                meta.modes.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
                m.as_str()
            );
        }
    }
    Ok(())
}

/// Build the model for `cfg` and log what was built — shape, parameter
/// count against the profile's budget, modes, and where the run lives.
fn build_and_log(
    cfg: &RunConfig,
    run_dir: &Path,
    modes: &[VocoderMode],
    vs: &VarStore,
    writer: &mut RunWriter,
) -> anyhow::Result<(Net, usize)> {
    let embed = cfg.model.mode_embed.then_some(modes.len());
    let (net, params) = model::build(vs, &model::net_opts(&cfg.model, embed))?;
    let width = if (cfg.model.width - 1.0).abs() < f32::EPSILON {
        String::new()
    } else {
        format!(" x{}", cfg.model.width)
    };
    let names = modes
        .iter()
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(",");
    writer.log(
        LogLevel::Info,
        format!(
            "{}{width} ll{} on {}: {params} parameters (budget {}), modes {names}{}, run dir {}",
            cfg.model.profile,
            cfg.model.lookahead,
            cfg.train.device,
            cfg.model.profile.param_budget(),
            if cfg.model.mode_embed {
                " with the mode embedding"
            } else {
                ""
            },
            run_dir.display()
        ),
    )?;
    Ok((net, params))
}

/// Load another run's weights into `vs` as this run's initialisation,
/// and log where they came from. Returns the checkpoint directory, for
/// the provenance line of every checkpoint this run writes.
fn initialise_from(
    spec: &str,
    cfg: &RunConfig,
    modes: &[VocoderMode],
    vs: &mut VarStore,
    writer: &mut RunWriter,
) -> anyhow::Result<String> {
    let dir = checkpoint::resolve_init(spec, &data_root().join("runs"))?;
    init_compatible(&checkpoint::read_meta(&dir)?, cfg, modes)?;
    let meta = checkpoint::load_init(&dir, vs, modes)?;
    let parent_modes = if meta.modes.is_empty() {
        "unrecorded".to_owned()
    } else {
        meta.modes
            .iter()
            .map(|m| m.as_str())
            .collect::<Vec<_>>()
            .join(",")
    };
    writer.log(
        LogLevel::Info,
        format!(
            "initialised from {} (run {}, step {}, modes {parent_modes}); this run starts at step 0 with a fresh optimiser",
            dir.display(),
            meta.name,
            meta.step,
        ),
    )?;
    Ok(dir.display().to_string())
}

/// Build everything the loop needs; returns the trainer and the step to
/// start from.
fn setup<'a>(
    cfg: &'a RunConfig,
    run_dir: &'a Path,
    resume: Option<&Path>,
    data: &DataSel,
    writer: &'a mut RunWriter,
    status: &'a mut RunStatus,
) -> anyhow::Result<(Trainer<'a>, u64)> {
    let device = device::resolve(&cfg.train.device)?;
    tch::manual_seed(i64::try_from(cfg.train.seed).unwrap_or(1));
    let modes = cfg.data.modes()?;
    let mut vs = VarStore::new(device.device);
    let (net, params) = build_and_log(cfg, run_dir, &modes, &vs, writer)?;
    let mut adam = Adam::new(&vs, cfg.train.lr);
    let mut best = None;
    let mut start_step = 0;
    // `init_from` seeds a *new* run's weights; a resume continues this
    // run and always wins, its checkpoint already carrying whatever the
    // run started from.
    let init_from = match (resume, cfg.train.init_from.as_deref()) {
        (None, Some(spec)) => {
            let dir = initialise_from(spec, cfg, &modes, &mut vs, writer)?;
            adam = Adam::new(&vs, cfg.train.lr);
            Some(dir)
        }
        _ => None,
    };
    if let Some(dir) = resume {
        resume_compatible(&checkpoint::read_meta(dir)?, cfg, &modes)?;
        let meta = checkpoint::load(dir, &mut vs, Some(&mut adam))?;
        start_step = meta.step;
        best = meta.best;
        writer.log(
            LogLevel::Info,
            format!("resumed from {} at step {start_step}", dir.display()),
        )?;
    }
    let losses = Losses::new(
        LossCfg {
            onset_w: cfg.train.onset_w,
            tail_w: cfg.train.tail_w,
            periodicity_w: cfg.train.periodicity_w,
            hnr_w: cfg.train.hnr_w,
            transient_w: cfg.train.transient_w,
            ..LossCfg::default()
        },
        device.device,
    );
    if cfg.train.gan {
        writer.log(
            LogLevel::Warn,
            "[train] gan = true: the discriminators exist but adversarial training is not wired into this milestone's loop; continuing with the regression losses",
        )?;
    }
    let mut loader = open_loader(cfg, data, start_step)?;
    writer.log(LogLevel::Info, loader.describe())?;
    let batch = usize::try_from(cfg.train.batch)?;
    // The probe batch is the seed stream's first batch in every
    // invocation, so `loss/probe` stays comparable across a resume.
    let probe = if start_step == 0 {
        loader.next_batch(batch, device.device)?
    } else {
        open_loader(cfg, data, 0)?.next_batch(batch, device.device)?
    };
    let clips = open_clips(cfg, data, writer)?;
    writer.log(LogLevel::Info, format!("{} eval clips", clips.len()))?;
    let index = match device.device {
        tch::Device::Cuda(n) => n,
        _ => 0,
    };
    let sys = SysSampler::new(device.kind, index);

    let trainer = Trainer {
        cfg,
        run_dir,
        writer,
        status,
        device,
        vs,
        net,
        adam,
        losses,
        loader,
        clips,
        sys,
        params,
        modes,
        best,
        probe,
        init_from,
        last_ckpt_step: (start_step > 0).then_some(start_step),
    };
    Ok((trainer, start_step))
}

fn train(
    cfg: &RunConfig,
    run_dir: &Path,
    resume: Option<&Path>,
    data: &DataSel,
    writer: &mut RunWriter,
    status: &mut RunStatus,
) -> anyhow::Result<Outcome> {
    let stop_flag = install_signal_flag()?;
    let (mut t, start_step) = setup(cfg, run_dir, resume, data, writer, status)?;
    let total = cfg.train.steps;
    let mut step = start_step;
    let mut losses_seen: Vec<f64> = Vec::new();
    let mut last_eval = None;
    let mut probe_seen = vec![(step, t.probe(step)?)];
    let mut window = Instant::now();
    let mut window_steps = 0u64;
    let mut last_status = Instant::now();
    t.write_status(step, RunState::Running)?;

    while step < total {
        if stop_flag.load(Ordering::Relaxed) {
            t.stop(step)?;
            return Ok(Outcome {
                run_dir: run_dir.to_path_buf(),
                step,
                state: RunState::Stopped,
                losses: losses_seen,
                last_eval,
                probe: probe_seen,
            });
        }
        step += 1;
        let loss = t.train_step(step)?;
        losses_seen.push(loss);
        window_steps += 1;

        if due(step, SYS_EVERY, total) {
            let el = window.elapsed().as_secs_f64().max(1e-9);
            #[allow(clippy::cast_precision_loss)]
            let sps = window_steps as f64 / el;
            let sample = t.sys.sample();
            let mut rows = vec![
                ("sys/steps_per_s", sps),
                ("sys/samples_per_s", sps * f64::from(cfg.train.batch)),
                ("sys/cpu", sample.cpu),
                ("sys/mem_gb", sample.mem_gb),
            ];
            if let Some(g) = sample.gpu_util {
                rows.push(("sys/gpu_util", g));
            }
            if let Some(g) = sample.gpu_mem_gb {
                rows.push(("sys/gpu_mem_gb", g));
            }
            t.writer.metrics(step, &rows)?;
            window = Instant::now();
            window_steps = 0;
        }
        if step == start_step + 1 || due(step, LOG_EVERY, total) {
            t.writer.log(
                LogLevel::Info,
                format!("step {step}/{total} loss {loss:.4}"),
            )?;
        }

        let do_eval = due(step, cfg.eval.every_steps, total);
        let do_ckpt = due(step, cfg.ckpt.every_steps, total);
        if do_eval {
            probe_seen.push((step, t.probe(step)?));
            let staging = do_ckpt.then(|| run_dir.join(".eval-tmp"));
            if let Some(s) = &staging {
                let _ = std::fs::remove_dir_all(s);
            }
            if let Some(m) = t.eval(step, staging.as_deref())? {
                last_eval = Some(m);
            }
            if do_ckpt {
                let dir = t.checkpoint(step)?;
                if let Some(s) = staging
                    && s.exists()
                {
                    std::fs::rename(&s, dir.join("audio"))?;
                }
            }
        } else if do_ckpt {
            t.checkpoint(step)?;
        }
        if do_ckpt || last_status.elapsed() >= STATUS_EVERY {
            t.write_status(step, RunState::Running)?;
            last_status = Instant::now();
        }
    }
    t.writer
        .log(LogLevel::Info, format!("finished {total} steps"))?;
    t.write_status(step, RunState::Finished)?;
    Ok(Outcome {
        run_dir: run_dir.to_path_buf(),
        step,
        state: RunState::Finished,
        losses: losses_seen,
        last_eval,
        probe: probe_seen,
    })
}

/// The config `smoke()` runs: lite ll5, 20 steps of batch 4 on CPU over
/// half-second crops, checkpoint + eval every 10 steps.
#[must_use]
pub fn smoke_config() -> RunConfig {
    let mut cfg = RunConfig {
        name: "smoke".to_owned(),
        ..RunConfig::default()
    };
    cfg.model.profile = unamblify::Profile::Lite;
    cfg.model.lookahead = 5;
    cfg.data.crop_s = 0.5;
    cfg.train.steps = 20;
    cfg.train.batch = 4;
    cfg.train.lr = 1e-3;
    "cpu".clone_into(&mut cfg.train.device);
    cfg.ckpt.every_steps = 10;
    cfg.ckpt.keep = 2;
    cfg.eval.every_steps = 10;
    cfg.eval.max_items = 2;
    cfg
}

/// 20-step CPU train on synthetic data into `run_dir` (a temporary
/// directory when `None`); fails unless the loss decreased.
pub fn smoke(run_dir: Option<&Path>) -> anyhow::Result<Outcome> {
    let tmp = tempfile_dir()?;
    let dir = run_dir.map_or_else(|| tmp.clone(), Path::to_path_buf);
    let outcome = run_with(
        smoke_config(),
        Some(&dir),
        None,
        &Overrides::default(),
        &DataSel::Synthetic {
            n_utts: 6,
            dur_s: 1.5,
        },
    )?;
    let n = outcome.losses.len();
    anyhow::ensure!(n >= 10, "smoke ran only {n} steps");
    // The loss on one fixed batch before and after: per-step losses vary
    // with batch composition (a garbage-tail example costs a few units
    // more), so they are not compared directly.
    let (first, last) = match (outcome.probe.first(), outcome.probe.last()) {
        (Some(f), Some(l)) if outcome.probe.len() >= 2 => (f.1, l.1),
        _ => anyhow::bail!("smoke: no probe losses recorded"),
    };
    anyhow::ensure!(
        last < first,
        "smoke: loss on the fixed batch did not decrease ({first:.4} → {last:.4})"
    );
    if run_dir.is_none() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    Ok(outcome)
}

fn tempfile_dir() -> anyhow::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "unamblify-smoke-{}-{}",
        std::process::id(),
        unix_ms()
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::runio::read_metrics;

    /// `[train] init_from` starts a *new* run from another run's
    /// weights: step 0, its own metrics, a fresh optimiser — but the
    /// weights are the parent's, so the loss on the fixed probe batch
    /// picks up where the parent left off instead of at a fresh init's.
    #[test]
    fn init_from_starts_a_new_run_on_another_runs_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let parent_dir = tmp.path().join("parent");
        let parent = smoke(Some(&parent_dir)).unwrap();
        let parent_probe = parent.probe.last().unwrap().1;
        let ckpt = parent_dir.join("checkpoints/step-000020");

        let mut cfg = smoke_config();
        "child".clone_into(&mut cfg.name);
        cfg.train.steps = 10;
        cfg.train.init_from = Some(ckpt.display().to_string());
        let child_dir = tmp.path().join("child");
        let child = run_with(
            cfg,
            Some(&child_dir),
            None,
            &Overrides::default(),
            &DataSel::Synthetic {
                n_utts: 6,
                dur_s: 1.0,
            },
        )
        .unwrap();
        assert_eq!(child.state, RunState::Finished);
        assert_eq!(child.step, 10, "an init is not a resume: it starts at 0");
        assert_eq!(child.losses.len(), 10);

        // The inherited weights, not a fresh init: the child's very
        // first probe loss is the parent's last, not the ~9 a fresh
        // super-lite/lite net starts at.
        let child_probe0 = child.probe.first().unwrap().1;
        assert!(
            (child_probe0 - parent_probe).abs() < 0.25 * parent_probe.abs().max(1.0),
            "child starts at {child_probe0:.4}, parent finished at {parent_probe:.4}"
        );
        let fresh_probe0 = smoke(Some(&tmp.path().join("fresh")))
            .unwrap()
            .probe
            .first()
            .unwrap()
            .1;
        assert!(
            child_probe0 < fresh_probe0,
            "child {child_probe0:.4} is no better than a fresh init {fresh_probe0:.4}"
        );

        // Provenance, and a fresh optimiser (the parent's moments are
        // not carried over, so this is a new run in every other sense).
        let meta = checkpoint::read_meta(&child_dir.join("checkpoints/step-000010")).unwrap();
        assert_eq!(meta.step, 10);
        assert_eq!(meta.name, "child");
        assert_eq!(
            meta.init_from.as_deref(),
            Some(ckpt.display().to_string().as_str())
        );
        let log = std::fs::read_to_string(child_dir.join("log.jsonl")).unwrap();
        assert!(log.contains("initialised from"), "{log}");

        // A resume of the child ignores init_from and continues it.
        let written = std::fs::read_to_string(child_dir.join("config.toml")).unwrap();
        let resumed = run_with(
            RunConfig::from_toml(&written).unwrap(),
            Some(&child_dir),
            Some(&child_dir.join("checkpoints/step-000010")),
            &Overrides {
                steps: Some(15),
                ..Overrides::default()
            },
            &DataSel::Synthetic {
                n_utts: 6,
                dur_s: 1.0,
            },
        )
        .unwrap();
        assert_eq!(resumed.step, 15);
        assert_eq!(
            resumed.losses.len(),
            5,
            "a resume continues, it does not restart"
        );
    }

    #[test]
    fn smoke_trains_checkpoints_evaluates_and_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("run");
        let outcome = smoke(Some(&dir)).unwrap();
        assert_eq!(outcome.state, RunState::Finished);
        assert_eq!(outcome.step, 20);
        assert_eq!(outcome.losses.len(), 20);
        assert!(outcome.last_eval.is_some());
        assert_eq!(
            outcome.probe.iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![0, 10, 20]
        );
        assert!(
            outcome.probe[2].1 < outcome.probe[0].1,
            "{:?}",
            outcome.probe
        );

        // Files per spec §7.
        assert!(dir.join("config.toml").exists());
        let status: RunStatus =
            serde_json::from_str(&std::fs::read_to_string(dir.join("status.json")).unwrap())
                .unwrap();
        assert_eq!(status.status, RunState::Finished);
        assert_eq!(status.step, 20);
        assert!(status.pid.is_none());
        assert_eq!(status.best.as_ref().unwrap().metric, "eval/lsd");
        let rows = read_metrics(&dir).unwrap();
        let keys: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.k.as_str()).collect();
        for k in [
            "loss/total",
            "loss/stft",
            "loss/mel",
            "loss/sisdr",
            "loss/onset",
            "loss/tail",
            "lr",
            "loss/probe",
            "sys/steps_per_s",
            "sys/samples_per_s",
            "sys/cpu",
            "eval/lsd",
            "eval/mel",
            "eval/sisdr",
            "eval/lsd_first1s",
            "eval/lsd_last1s",
            "eval/babble",
            "eval/hnr",
            "eval/hnr_abs",
            "eval/lsd_rx",
        ] {
            assert!(keys.contains(k), "missing metric {k}");
        }
        assert!(
            !keys.contains("eval/lsd_drops"),
            "no drops sibling for synthetic clips"
        );
        assert_eq!(rows.iter().filter(|r| r.k == "loss/total").count(), 20);
        let log = std::fs::read_to_string(dir.join("log.jsonl")).unwrap();
        assert!(log.lines().count() >= 4);
        let ck = dir.join("checkpoints");
        let list = checkpoint::list(&ck).unwrap();
        assert_eq!(
            list.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![10, 20]
        );
        let audio = ck.join("step-000020").join("audio");
        assert!(audio.join("synthetic_00.out.wav").exists());
        assert!(audio.join("synthetic_00.spec.json").exists());
        assert!(audio.join("synthetic_01.clean.wav").exists());
        assert!(!dir.join(".eval-tmp").exists());

        // Resume from step 10 for five more steps into a fresh run dir.
        let dir2 = tmp.path().join("resumed");
        let out2 = run_with(
            smoke_config(),
            Some(&dir2),
            Some(&ck.join("step-000010")),
            &Overrides {
                device: Some("cpu".to_owned()),
                steps: Some(15),
                shards: None,
                init_from: None,
            },
            &DataSel::Synthetic {
                n_utts: 6,
                dur_s: 1.5,
            },
        )
        .unwrap();
        assert_eq!(out2.step, 15);
        assert_eq!(out2.losses.len(), 5);
        let rows2 = read_metrics(&dir2).unwrap();
        assert_eq!(
            rows2
                .iter()
                .filter(|r| r.k == "loss/total")
                .map(|r| r.step)
                .min(),
            Some(11)
        );
        let meta = checkpoint::read_meta(&dir2.join("checkpoints").join("step-000015")).unwrap();
        assert_eq!(meta.step, 15);
        assert!(meta.best.is_some());
        assert_eq!(meta.modes, vec![unamblify::VocoderMode::Dstar]);
        assert!(!meta.mode_embed);
        // The probe is the same batch in both invocations, and the model
        // at step 10 is restored exactly, so loss/probe at the resume
        // step equals the original run's value there.
        assert_eq!(out2.probe[0].0, 10);
        assert!(
            (out2.probe[0].1 - outcome.probe[1].1).abs() < 1e-4,
            "{} vs {}",
            out2.probe[0].1,
            outcome.probe[1].1
        );

        // Resume into the *same* directory (what `runs resume` and the
        // dashboard do) from step 10: the rows steps 11..20 logged before
        // the "crash" are dropped, so every step appears exactly once.
        let _ = std::fs::remove_dir_all(ck.join("step-000020"));
        let out3 = run_with(
            smoke_config(),
            Some(&dir),
            Some(&ck.join("step-000010")),
            &Overrides {
                device: Some("cpu".to_owned()),
                steps: Some(15),
                shards: None,
                init_from: None,
            },
            &DataSel::Synthetic {
                n_utts: 6,
                dur_s: 1.5,
            },
        )
        .unwrap();
        assert_eq!(out3.step, 15);
        let rows3 = read_metrics(&dir).unwrap();
        let mut steps: Vec<u64> = rows3
            .iter()
            .filter(|r| r.k == "loss/total")
            .map(|r| r.step)
            .collect();
        let uniq: std::collections::BTreeSet<u64> = steps.iter().copied().collect();
        assert_eq!(steps.len(), uniq.len(), "no step logged twice: {steps:?}");
        steps.sort_unstable();
        assert_eq!(steps, (1..=15).collect::<Vec<_>>());
        assert!(
            std::fs::read_to_string(dir.join("log.jsonl"))
                .unwrap()
                .contains("dropped")
        );

        // A mismatched profile is refused and recorded as failed.
        let mut bad = smoke_config();
        bad.model.profile = unamblify::Profile::SuperLite;
        let dir3 = tmp.path().join("bad");
        let err = run_with(
            bad,
            Some(&dir3),
            Some(&ck.join("step-000010")),
            &Overrides::default(),
            &DataSel::Synthetic {
                n_utts: 2,
                dur_s: 1.0,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("checkpoint is lite"), "{err}");
        let status: RunStatus =
            serde_json::from_str(&std::fs::read_to_string(dir3.join("status.json")).unwrap())
                .unwrap();
        assert_eq!(status.status, RunState::Failed);
        assert!(
            std::fs::read_to_string(dir3.join("log.jsonl"))
                .unwrap()
                .contains("\"error\"")
        );
    }

    /// The mode embedding trains on the synthetic single-mode stream (every
    /// example mode 0), lands in the checkpoint's meta, and a resume must
    /// agree with it.
    #[test]
    fn the_mode_embedding_trains_and_is_pinned_by_the_checkpoint() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = smoke_config();
        cfg.model.mode_embed = true;
        cfg.data.modes = vec![
            unamblify::VocoderMode::Dstar,
            unamblify::VocoderMode::Codec2_3200,
        ];
        cfg.train.steps = 10;
        cfg.ckpt.every_steps = 10;
        cfg.eval.every_steps = 10;
        let dir = tmp.path().join("run");
        let data = DataSel::Synthetic {
            n_utts: 4,
            dur_s: 1.0,
        };
        let out = run_with(cfg.clone(), Some(&dir), None, &Overrides::default(), &data).unwrap();
        assert_eq!(out.state, RunState::Finished);
        let ck = dir.join("checkpoints").join("step-000010");
        let meta = checkpoint::read_meta(&ck).unwrap();
        assert!(meta.mode_embed);
        assert_eq!(meta.modes.len(), 2);
        let log = std::fs::read_to_string(dir.join("log.jsonl")).unwrap();
        assert!(
            log.contains("modes dstar,codec2-3200 with the mode embedding"),
            "{log}"
        );
        // Resume with the embedding off: refused.
        let mut blind = cfg.clone();
        blind.model.mode_embed = false;
        let err = run_with(
            blind,
            Some(&tmp.path().join("blind")),
            Some(&ck),
            &Overrides::default(),
            &data,
        )
        .unwrap_err();
        assert!(err.to_string().contains("mode embedding"), "{err}");
        // Resume with other modes: refused.
        let mut other = cfg.clone();
        other.data.modes = vec![
            unamblify::VocoderMode::YsfDmr,
            unamblify::VocoderMode::Dstar,
        ];
        let err = run_with(
            other,
            Some(&tmp.path().join("other")),
            Some(&ck),
            &Overrides::default(),
            &data,
        )
        .unwrap_err();
        assert!(err.to_string().contains("trained on modes"), "{err}");
        // The same config resumes.
        let out2 = run_with(
            cfg,
            Some(&tmp.path().join("again")),
            Some(&ck),
            &Overrides {
                steps: Some(12),
                ..Overrides::default()
            },
            &data,
        )
        .unwrap();
        assert_eq!(out2.step, 12);
    }

    #[test]
    fn a_resumed_run_does_not_replay_the_first_batches() {
        let cfg = smoke_config();
        let data = DataSel::Synthetic {
            n_utts: 6,
            dur_s: 1.5,
        };
        let first = |start: u64| {
            let mut l = open_loader(&cfg, &data, start).unwrap();
            l.next_batch(4, tch::Device::Cpu).unwrap()
        };
        let same = |a: &tch::Tensor, b: &tch::Tensor| (a - b).abs().max().double_value(&[]) == 0.0;
        assert!(
            same(&first(0).deg8, &first(0).deg8),
            "fresh runs are reproducible"
        );
        assert!(
            !same(&first(0).deg8, &first(8_000).deg8),
            "resume draws new data"
        );
        assert!(!same(&first(8_000).deg8, &first(16_000).deg8));
    }

    #[test]
    fn unknown_device_fails_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let err = run_with(
            smoke_config(),
            Some(tmp.path()),
            None,
            &Overrides {
                device: Some("tpu:0".to_owned()),
                steps: None,
                shards: None,
                init_from: None,
            },
            &DataSel::Synthetic {
                n_utts: 2,
                dur_s: 1.0,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown device"), "{err}");
        assert!(due(10, 5, 100) && due(100, 7, 100) && !due(9, 5, 100) && !due(9, 0, 100));
    }
}
