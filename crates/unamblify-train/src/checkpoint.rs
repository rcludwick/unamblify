// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `checkpoints/step-NNNNNN/{model.safetensors, optim.safetensors,
//! meta.json}` (spec §7): written atomically (into a temporary sibling,
//! then renamed), pruned to the newest `keep` with the best step always
//! kept, and loaded for `--resume`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use tch::Tensor;
use tch::nn::VarStore;
use unamblify::{Best, Profile, VocoderMode};

use crate::optim::Adam;

/// `meta.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    /// Optimiser step the weights correspond to.
    pub step: u64,
    /// Run name.
    pub name: String,
    /// Model profile.
    pub profile: Profile,
    /// `[model] width`: the multiplier on the profile's widths these
    /// weights were built at. Absent in checkpoints written before the
    /// knob existed, which were all 1.0.
    #[serde(default = "one")]
    pub width: f32,
    /// Lookahead in AMBE 20 ms frames (`[model] lookahead`; ll5 = 100 ms).
    pub lookahead: u32,
    /// Trainable parameter count.
    pub params: usize,
    /// RFC 3339 UTC write time.
    pub saved_at: String,
    /// Device string the run used.
    pub device: String,
    /// Seed of the run.
    pub seed: u64,
    /// The modes the run trains on, in the order the model's mode index
    /// counts them (`[data] modes`). Absent in checkpoints written before
    /// multi-mode runs existed: one mode, unrecorded here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modes: Vec<VocoderMode>,
    /// Whether the weights include the mode embedding (`[model]
    /// mode_embed`).
    #[serde(default)]
    pub mode_embed: bool,
    /// Whether the weights include the aperiodic excitation path
    /// (`[model] noise_head`). Absent in checkpoints written before it
    /// existed, which had none.
    #[serde(default)]
    pub noise_head: bool,
    /// Whether the noise path carried modulation and 1 ms gains
    /// (`[model] noise_mod`). Absent in checkpoints written before it
    /// existed, which had neither.
    #[serde(default)]
    pub noise_mod: bool,
    /// Best eval so far, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<Best>,
    /// The checkpoint this run's weights started from (`[train]
    /// init_from`), as the directory it resolved to. `None` for a run
    /// that started from a fresh initialisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from: Option<String>,
}

/// Serde default for [`Meta::width`]: every checkpoint written before
/// the knob existed was the profile's own width.
const fn one() -> f32 {
    1.0
}

/// `checkpoints/step-NNNNNN` under `ckpt_root`.
#[must_use]
pub fn dir_for(ckpt_root: &Path, step: u64) -> PathBuf {
    ckpt_root.join(format!("step-{step:06}"))
}

/// Parse `step-NNNNNN` back to a step.
#[must_use]
pub fn step_of(dir: &Path) -> Option<u64> {
    dir.file_name()?
        .to_str()?
        .strip_prefix("step-")?
        .parse()
        .ok()
}

/// Every checkpoint under `ckpt_root`, ascending by step.
pub fn list(ckpt_root: &Path) -> anyhow::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    if !ckpt_root.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(ckpt_root)? {
        let path = entry?.path();
        if let Some(step) = step_of(&path)
            && path.join("meta.json").exists()
        {
            out.push((step, path));
        }
    }
    out.sort();
    Ok(out)
}

/// Write model + optimiser + meta into `dir_for(ckpt_root, meta.step)`.
/// A checkpoint already at that step is replaced, but its rendered
/// `audio/` (the eval clips the dashboard compares) is carried over.
pub fn save(ckpt_root: &Path, vs: &VarStore, adam: &Adam, meta: &Meta) -> anyhow::Result<PathBuf> {
    let final_dir = dir_for(ckpt_root, meta.step);
    let tmp = ckpt_root.join(format!(".step-{:06}.tmp", meta.step));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    vs.save(tmp.join("model.safetensors"))
        .context("VarStore::save")?;
    adam.save(&tmp.join("optim.safetensors"))?;
    std::fs::write(tmp.join("meta.json"), serde_json::to_string_pretty(meta)?)?;
    if final_dir.exists() {
        let audio = final_dir.join("audio");
        if audio.is_dir() {
            std::fs::rename(&audio, tmp.join("audio"))
                .with_context(|| format!("keep {}", audio.display()))?;
        }
        std::fs::remove_dir_all(&final_dir)?;
    }
    std::fs::rename(&tmp, &final_dir)
        .with_context(|| format!("rename {} → {}", tmp.display(), final_dir.display()))?;
    Ok(final_dir)
}

/// Read `meta.json` of a checkpoint directory.
pub fn read_meta(dir: &Path) -> anyhow::Result<Meta> {
    let p = dir.join("meta.json");
    let text = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
    serde_json::from_str(&text).with_context(|| p.display().to_string())
}

/// Load weights (and, when given, optimiser state) from `dir`.
pub fn load(dir: &Path, vs: &mut VarStore, adam: Option<&mut Adam>) -> anyhow::Result<Meta> {
    let meta = read_meta(dir)?;
    vs.load(dir.join("model.safetensors"))
        .with_context(|| format!("{}: model.safetensors", dir.display()))?;
    if let Some(adam) = adam {
        adam.load(&dir.join("optim.safetensors"))?;
    }
    Ok(meta)
}

/// The `mode_embed` weight in the `VarStore`'s flat namespace: one row
/// per mode of the run that wrote it.
const MODE_EMBED_WEIGHT: &str = "mode_embed.weight";

/// Resolve `[train] init_from` to a checkpoint directory. `spec` is
/// either a path (anything containing a separator) to a
/// `checkpoints/step-N` directory, or a run id under `runs_root`
/// optionally suffixed `:<step>`. Without a step it picks that run's
/// best checkpoint, falling back to its newest.
pub fn resolve_init(spec: &str, runs_root: &Path) -> anyhow::Result<PathBuf> {
    if spec.contains(std::path::MAIN_SEPARATOR) {
        let dir = PathBuf::from(spec);
        ensure!(
            dir.join("meta.json").is_file(),
            "{}: not a checkpoint directory (no meta.json)",
            dir.display()
        );
        return Ok(dir);
    }
    let (id, want) = match spec.split_once(':') {
        Some((id, step)) => (
            id,
            Some(
                step.parse::<u64>()
                    .with_context(|| format!("init_from: {step:?} is not a step number"))?,
            ),
        ),
        None => (spec, None),
    };
    let root = runs_root.join(id).join("checkpoints");
    let all = list(&root)?;
    ensure!(!all.is_empty(), "{}: no checkpoints", root.display());
    if let Some(step) = want {
        let dir = dir_for(&root, step);
        ensure!(
            dir.join("meta.json").is_file(),
            "{}: no such checkpoint, the run has {}",
            dir.display(),
            all.iter()
                .map(|(s, _)| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Ok(dir);
    }
    let (newest_step, newest) = all
        .last()
        .ok_or_else(|| anyhow!("{}: no checkpoints", root.display()))?;
    if let Some(best) = read_meta(newest)?.best {
        let dir = dir_for(&root, best.step);
        if dir.join("meta.json").is_file() {
            return Ok(dir);
        }
        bail!(
            "{}: best checkpoint (step {}, {} = {:.4}) has been pruned; name a step explicitly",
            root.display(),
            best.step,
            best.metric,
            best.value
        );
    }
    let _ = newest_step;
    Ok(newest.clone())
}

/// Load `dir`'s weights into `vs` as the *initialisation* of a new run:
/// no optimiser state, and the mode embedding re-indexed so a run over
/// `want` modes inherits the parent's row for each of them. The parent's
/// rows for modes this run drops are left behind. `want` must be a
/// subset of the checkpoint's modes (a checkpoint that recorded none is
/// a single-mode run and is taken as-is).
pub fn load_init(dir: &Path, vs: &mut VarStore, want: &[VocoderMode]) -> anyhow::Result<Meta> {
    let meta = read_meta(dir)?;
    let path = dir.join("model.safetensors");
    if meta.modes.is_empty() || meta.modes == want {
        vs.load(&path)
            .with_context(|| format!("{}: model.safetensors", path.display()))?;
        return Ok(meta);
    }
    let rows: Vec<i64> = want
        .iter()
        .map(|m| {
            meta.modes
                .iter()
                .position(|c| c == m)
                .map(|i| i64::try_from(i).unwrap_or(0))
                .ok_or_else(|| {
                    anyhow!(
                        "checkpoint holds modes {:?}, which do not include {}",
                        meta.modes.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
                        m.as_str()
                    )
                })
        })
        .collect::<anyhow::Result<_>>()?;
    let mut src: HashMap<String, Tensor> = Tensor::read_safetensors(&path)
        .with_context(|| format!("{}: model.safetensors", path.display()))?
        .into_iter()
        .collect();
    let index = Tensor::from_slice(&rows);
    tch::no_grad(|| -> anyhow::Result<()> {
        for (name, mut var) in vs.variables() {
            let t = src
                .remove(&name)
                .ok_or_else(|| anyhow!("{}: no tensor named {name}", path.display()))?;
            let t = if name == MODE_EMBED_WEIGHT {
                t.index_select(0, &index)
            } else {
                t
            };
            ensure!(
                t.size() == var.size(),
                "{name}: checkpoint has {:?}, this run wants {:?}",
                t.size(),
                var.size()
            );
            var.f_copy_(&t.to_device(var.device()))?;
        }
        Ok(())
    })?;
    Ok(meta)
}

/// Delete all but the newest `keep` checkpoints, never deleting
/// `protect` (the best step). Returns the removed directories.
pub fn prune(ckpt_root: &Path, keep: u32, protect: Option<u64>) -> anyhow::Result<Vec<PathBuf>> {
    let all = list(ckpt_root)?;
    let keep = usize::try_from(keep).unwrap_or(usize::MAX).max(1);
    let mut removed = Vec::new();
    if all.len() <= keep {
        return Ok(removed);
    }
    let cutoff = all.len() - keep;
    for (step, dir) in all.into_iter().take(cutoff) {
        if Some(step) == protect {
            continue;
        }
        std::fs::remove_dir_all(&dir).with_context(|| dir.display().to_string())?;
        removed.push(dir);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Net;
    use tch::{Device, Kind, Tensor};

    fn meta(step: u64) -> Meta {
        Meta {
            step,
            name: "t".to_owned(),
            profile: Profile::SuperLite,
            width: 1.0,
            lookahead: 5,
            params: 0,
            saved_at: "2026-09-10T00:00:00Z".to_owned(),
            device: "cpu".to_owned(),
            seed: 1,
            modes: vec![VocoderMode::Dstar],
            mode_embed: true,
            noise_head: false,
            noise_mod: false,
            best: None,
            init_from: None,
        }
    }

    /// A four-mode generalist's meta, optionally with a best step.
    fn generalist_meta(step: u64, best: Option<u64>) -> Meta {
        Meta {
            modes: vec![
                VocoderMode::Dstar,
                VocoderMode::YsfDmr,
                VocoderMode::Codec2_3200,
                VocoderMode::Codec2_1600,
            ],
            best: best.map(|step| Best {
                metric: "eval/lsd".to_owned(),
                value: 11.0,
                step,
            }),
            ..meta(step)
        }
    }

    #[test]
    fn checkpoint_round_trip_reproduces_outputs_and_optimizer_state() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("checkpoints");
        let vs = VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, Some(1));
        let m0 = crate::model::mode_zeros(1, Device::Cpu);
        let mut adam = Adam::new(&vs, 1e-3);
        let x = Tensor::rand([1, 1, 800], (Kind::Float, Device::Cpu)) - 0.5;
        // One update so the moments are non-trivial.
        net.forward(&x, &m0).square().mean(Kind::Float).backward();
        adam.step();
        let y = net.forward(&x, &m0);
        let dir = save(&root, &vs, &adam, &meta(7)).unwrap();
        assert_eq!(dir, root.join("step-000007"));
        assert!(dir.join("model.safetensors").exists());
        assert!(dir.join("optim.safetensors").exists());
        // Saving the same step again keeps the rendered audio.
        std::fs::create_dir_all(dir.join("audio")).unwrap();
        std::fs::write(dir.join("audio/clip.out.wav"), b"wav").unwrap();
        let again = save(&root, &vs, &adam, &meta(7)).unwrap();
        assert_eq!(again, dir);
        assert_eq!(
            std::fs::read(dir.join("audio/clip.out.wav")).unwrap(),
            b"wav"
        );
        assert!(dir.join("model.safetensors").exists());

        let mut vs2 = VarStore::new(Device::Cpu);
        let net2 = Net::new(&vs2.root(), Profile::SuperLite, 5, Some(1));
        let mut adam2 = Adam::new(&vs2, 1e-3);
        let m = load(&dir, &mut vs2, Some(&mut adam2)).unwrap();
        assert_eq!(m.step, 7);
        assert_eq!(adam2.steps(), 1);
        let diff = (net2.forward(&x, &m0) - &y).abs().max().double_value(&[]);
        assert!(diff < 5e-7, "{diff}");
        // Same next update on both.
        adam.zero_grad();
        net.forward(&x, &m0).square().mean(Kind::Float).backward();
        adam.step();
        adam2.zero_grad();
        net2.forward(&x, &m0).square().mean(Kind::Float).backward();
        adam2.step();
        let diff = (net2.forward(&x, &m0) - net.forward(&x, &m0))
            .abs()
            .max()
            .double_value(&[]);
        assert!(diff < 1e-5, "{diff}");
    }

    #[test]
    fn prune_keeps_the_newest_and_the_best() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("checkpoints");
        let vs = VarStore::new(Device::Cpu);
        let _net = Net::new(&vs.root(), Profile::SuperLite, 5, None);
        let adam = Adam::new(&vs, 1e-3);
        for step in [10, 20, 30, 40, 50] {
            save(&root, &vs, &adam, &meta(step)).unwrap();
        }
        assert_eq!(list(&root).unwrap().len(), 5);
        let removed = prune(&root, 2, Some(20)).unwrap();
        assert_eq!(removed.len(), 2);
        let left: Vec<u64> = list(&root).unwrap().into_iter().map(|(s, _)| s).collect();
        assert_eq!(left, vec![20, 40, 50]);
        assert_eq!(step_of(&root.join("step-000040")), Some(40));
        assert_eq!(step_of(&root.join("junk")), None);
        assert_eq!(prune(&root, 0, None).unwrap().len(), 2);
    }

    /// The specialist's whole point: it starts life as the generalist,
    /// keeping the generalist's own embedding row for the one mode it
    /// still trains on — not row 0, and not a fresh random row.
    #[test]
    fn a_specialist_inherits_the_generalists_weights_and_its_own_mode_row() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("checkpoints");
        let modes = [
            VocoderMode::Dstar,
            VocoderMode::YsfDmr,
            VocoderMode::Codec2_3200,
            VocoderMode::Codec2_1600,
        ];
        let vs = VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root(), Profile::SuperLite, 5, Some(modes.len()));
        let adam = Adam::new(&vs, 1e-3);
        // Make every mode row distinguishable.
        tch::no_grad(|| {
            for (name, mut v) in vs.variables() {
                if name == MODE_EMBED_WEIGHT {
                    let rows = v.size()[0];
                    for r in 0..rows {
                        let _ = v
                            .get(r)
                            .fill_(f64::from(i32::try_from(r).unwrap_or(0)) + 1.0);
                    }
                } else {
                    let _ = v.uniform_(-0.2, 0.2);
                }
            }
        });
        let dir = save(&root, &vs, &adam, &generalist_meta(12_000, None)).unwrap();
        let want_row: Vec<f64> = vs
            .variables()
            .get(MODE_EMBED_WEIGHT)
            .unwrap()
            .get(2)
            .iter::<f64>()
            .unwrap()
            .collect();

        // A codec2-3200 specialist: one mode, the generalist's row 2.
        let mut vs2 = VarStore::new(Device::Cpu);
        let net2 = Net::new(&vs2.root(), Profile::SuperLite, 5, Some(1));
        let m = load_init(&dir, &mut vs2, &[VocoderMode::Codec2_3200]).unwrap();
        assert_eq!(m.step, 12_000);
        let got = vs2.variables();
        let row = got.get(MODE_EMBED_WEIGHT).unwrap();
        assert_eq!(row.size(), vec![1, i64::try_from(want_row.len()).unwrap()]);
        let got_row: Vec<f64> = row.get(0).iter::<f64>().unwrap().collect();
        assert_eq!(got_row, want_row, "the specialist took the wrong mode row");

        // Every other weight is the generalist's, so with that one mode
        // the two nets agree to floating-point noise.
        let x = Tensor::rand([1, 1, 800], (Kind::Float, Device::Cpu)) - 0.5;
        let m4 = Tensor::from_slice(&[2i64]).to_device(Device::Cpu);
        let m1 = crate::model::mode_zeros(1, Device::Cpu);
        let diff = (net2.forward(&x, &m1) - net.forward(&x, &m4))
            .abs()
            .max()
            .double_value(&[]);
        assert!(diff < 5e-6, "{diff}");
    }

    /// A mode the generalist never saw cannot be specialised, and the
    /// error says which modes it does hold.
    #[test]
    fn a_mode_the_checkpoint_never_saw_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("checkpoints");
        let vs = VarStore::new(Device::Cpu);
        let _net = Net::new(&vs.root(), Profile::SuperLite, 5, Some(2));
        let adam = Adam::new(&vs, 1e-3);
        let m = Meta {
            modes: vec![VocoderMode::Dstar, VocoderMode::YsfDmr],
            ..meta(500)
        };
        let dir = save(&root, &vs, &adam, &m).unwrap();
        let mut vs2 = VarStore::new(Device::Cpu);
        let _n2 = Net::new(&vs2.root(), Profile::SuperLite, 5, Some(1));
        let e = load_init(&dir, &mut vs2, &[VocoderMode::Codec2_1600])
            .unwrap_err()
            .to_string();
        assert!(e.contains("codec2-1600"), "{e}");
        assert!(e.contains("dstar"), "{e}");
    }

    /// `resolve_init`: a bare run id takes the best checkpoint, `:step`
    /// takes that one, a path is taken as-is, and a pruned best is an
    /// error rather than a silent fallback to the newest.
    #[test]
    fn resolve_init_finds_the_best_checkpoint_of_a_run() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path().join("runs");
        let root = runs.join("20260912-000009-gen").join("checkpoints");
        let vs = VarStore::new(Device::Cpu);
        let _net = Net::new(&vs.root(), Profile::SuperLite, 5, Some(4));
        let adam = Adam::new(&vs, 1e-3);
        for step in [500, 1000, 1500] {
            save(&root, &vs, &adam, &generalist_meta(step, Some(1000))).unwrap();
        }
        assert_eq!(
            resolve_init("20260912-000009-gen", &runs).unwrap(),
            root.join("step-001000")
        );
        assert_eq!(
            resolve_init("20260912-000009-gen:500", &runs).unwrap(),
            root.join("step-000500")
        );
        assert_eq!(
            resolve_init(&root.join("step-001500").display().to_string(), &runs).unwrap(),
            root.join("step-001500")
        );
        // No best recorded: the newest.
        let plain = runs.join("plain").join("checkpoints");
        save(&plain, &vs, &adam, &generalist_meta(20, None)).unwrap();
        save(&plain, &vs, &adam, &generalist_meta(40, None)).unwrap();
        assert_eq!(
            resolve_init("plain", &runs).unwrap(),
            plain.join("step-000040")
        );
        // A pruned best is named, not silently replaced.
        std::fs::remove_dir_all(root.join("step-001000")).unwrap();
        let e = resolve_init("20260912-000009-gen", &runs)
            .unwrap_err()
            .to_string();
        assert!(e.contains("pruned"), "{e}");
        assert!(resolve_init("20260912-000009-gen:999", &runs).is_err());
        assert!(resolve_init("no-such-run", &runs).is_err());
    }
}
