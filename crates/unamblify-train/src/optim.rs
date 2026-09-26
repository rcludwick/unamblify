// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Adam with state that can be written to `optim.safetensors` and read
//! back, so a resumed run continues exactly. tch's `nn::Optimizer` wraps a
//! libtorch optimiser whose moments are not reachable from Rust, hence
//! this ~100-line implementation over the var store's trainable tensors
//! (Kingma & Ba 2015, with bias correction; optional global-norm clipping).

use std::path::Path;

use anyhow::Context;
use tch::nn::VarStore;
use tch::{Kind, Tensor};

/// One tracked parameter and its moments.
#[derive(Debug)]
struct Param {
    name: String,
    var: Tensor,
    m: Tensor,
    v: Tensor,
}

/// The optimiser.
#[derive(Debug)]
pub struct Adam {
    params: Vec<Param>,
    steps: u64,
    /// Learning rate (mutable so a schedule can drive it).
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Denominator floor.
    pub eps: f64,
    /// Global gradient-norm clip; `None` = off.
    pub clip_norm: Option<f64>,
}

impl Adam {
    /// Track every trainable variable of `vs`, in name order.
    #[must_use]
    pub fn new(vs: &VarStore, lr: f64) -> Self {
        let mut vars: Vec<(String, Tensor)> = vs
            .variables()
            .into_iter()
            .filter(|(_, t)| t.requires_grad())
            .collect();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        let params = vars
            .into_iter()
            .map(|(name, var)| {
                let m = Tensor::zeros_like(&var);
                let v = Tensor::zeros_like(&var);
                Param { name, var, m, v }
            })
            .collect();
        Self {
            params,
            steps: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clip_norm: Some(5.0),
        }
    }

    /// Updates applied so far.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.steps
    }

    /// Tracked parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether nothing is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Zero every gradient.
    pub fn zero_grad(&self) {
        for p in &self.params {
            let mut g = p.var.grad();
            if g.defined() {
                let _ = g.zero_();
            }
        }
    }

    /// One update from the gradients currently on the parameters. Returns
    /// the global gradient norm before clipping.
    pub fn step(&mut self) -> f64 {
        self.steps += 1;
        #[allow(clippy::cast_precision_loss)]
        let t = self.steps as f64;
        let (b1, b2) = (self.beta1, self.beta2);
        let lr_t = self.lr * (1.0 - b2.powf(t)).sqrt() / (1.0 - b1.powf(t));
        tch::no_grad(|| {
            let grads: Vec<Option<Tensor>> = self
                .params
                .iter()
                .map(|p| {
                    let g = p.var.grad();
                    g.defined().then_some(g)
                })
                .collect();
            let sq: f64 = grads
                .iter()
                .flatten()
                .map(|g| g.square().sum(Kind::Float).double_value(&[]))
                .sum();
            let norm = sq.sqrt();
            let scale = match self.clip_norm {
                Some(c) if norm > c => Some(c / (norm + 1e-6)),
                _ => None,
            };
            for (p, g) in self.params.iter_mut().zip(grads) {
                let Some(g) = g else { continue };
                let g = scale.map_or(g.shallow_clone(), |s| &g * s);
                let m_new = &p.m * b1 + &g * (1.0 - b1);
                let v_new = &p.v * b2 + &g * &g * (1.0 - b2);
                p.m.copy_(&m_new);
                p.v.copy_(&v_new);
                let update = &p.m / (p.v.sqrt() + self.eps) * lr_t;
                let _ = p.var.f_sub_(&update);
            }
            norm
        })
    }

    /// Write the moments and step count as safetensors.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut named: Vec<(String, Tensor)> = Vec::with_capacity(2 * self.params.len() + 1);
        for p in &self.params {
            named.push((format!("{}.m", p.name), p.m.shallow_clone()));
            named.push((format!("{}.v", p.name), p.v.shallow_clone()));
        }
        #[allow(clippy::cast_possible_wrap)]
        named.push(("adam.steps".to_owned(), Tensor::from(self.steps as i64)));
        Tensor::write_safetensors(&named, path).with_context(|| path.display().to_string())?;
        Ok(())
    }

    /// Restore moments and step count written by [`Adam::save`] for the
    /// same parameter set.
    pub fn load(&mut self, path: &Path) -> anyhow::Result<()> {
        let named = Tensor::read_safetensors(path).with_context(|| path.display().to_string())?;
        let map: std::collections::HashMap<String, Tensor> = named.into_iter().collect();
        tch::no_grad(|| -> anyhow::Result<()> {
            for p in &mut self.params {
                for (suffix, slot) in [("m", &mut p.m), ("v", &mut p.v)] {
                    let key = format!("{}.{suffix}", p.name);
                    let src = map
                        .get(&key)
                        .with_context(|| format!("{}: missing {key}", path.display()))?;
                    anyhow::ensure!(
                        src.size() == slot.size(),
                        "{key}: shape {:?} != {:?}",
                        src.size(),
                        slot.size()
                    );
                    slot.copy_(&src.to_device(slot.device()));
                }
            }
            Ok(())
        })?;
        let steps = map
            .get("adam.steps")
            .with_context(|| format!("{}: missing adam.steps", path.display()))?
            .int64_value(&[]);
        self.steps = u64::try_from(steps)?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use tch::Device;
    use tch::nn::Module;

    #[test]
    fn adam_fits_a_line_and_round_trips_its_state() {
        let vs = VarStore::new(Device::Cpu);
        let lin = tch::nn::linear(vs.root() / "l", 1, 1, tch::nn::LinearConfig::default());
        let mut opt = Adam::new(&vs, 0.05);
        assert_eq!(opt.len(), 2);
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0]).view([4, 1]);
        let y = &x * 2.0 + 1.0;
        let mut first = 0.0;
        let mut last = 0.0;
        for i in 0..200 {
            let loss = lin.forward(&x).mse_loss(&y, tch::Reduction::Mean);
            opt.zero_grad();
            loss.backward();
            let norm = opt.step();
            assert!(norm.is_finite());
            let l = loss.double_value(&[]);
            if i == 0 {
                first = l;
            }
            last = l;
        }
        assert!(last < first * 0.05, "{first} → {last}");
        assert_eq!(opt.steps(), 200);

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("optim.safetensors");
        opt.save(&p).unwrap();
        let vs2 = VarStore::new(Device::Cpu);
        let _lin2 = tch::nn::linear(vs2.root() / "l", 1, 1, tch::nn::LinearConfig::default());
        let mut opt2 = Adam::new(&vs2, 0.05);
        opt2.load(&p).unwrap();
        assert_eq!(opt2.steps(), 200);
        for (a, b) in opt.params.iter().zip(&opt2.params) {
            assert_eq!(a.name, b.name);
            assert!((&a.m - &b.m).abs().max().double_value(&[]) < f64::EPSILON);
            assert!((&a.v - &b.v).abs().max().double_value(&[]) < f64::EPSILON);
        }
        // A mismatched parameter set is refused.
        let vs3 = VarStore::new(Device::Cpu);
        let _other = tch::nn::linear(vs3.root() / "z", 1, 1, tch::nn::LinearConfig::default());
        assert!(Adam::new(&vs3, 0.1).load(&p).is_err());
    }
}
