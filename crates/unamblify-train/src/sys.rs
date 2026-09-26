// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `sys/*` metrics: CPU and memory from `sysinfo`; GPU utilisation and
//! memory via NVML (`cuda` feature), amdgpu sysfs (`rocm` feature), or
//! not at all on macOS (MPS exposes no counters).

use sysinfo::System;

use crate::device::DeviceKind;

/// One sample.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SysSample {
    /// Whole-machine CPU utilisation, percent.
    pub cpu: f64,
    /// Used memory, GiB.
    pub mem_gb: f64,
    /// GPU utilisation, percent, when a sampler exists.
    pub gpu_util: Option<f64>,
    /// GPU memory used, GiB, when a sampler exists.
    pub gpu_mem_gb: Option<f64>,
}

/// The sampler.
pub struct SysSampler {
    sys: System,
    #[cfg(feature = "cuda")]
    nvml: Option<(nvml_wrapper::Nvml, u32)>,
    #[cfg(feature = "rocm")]
    amd: Option<amdgpu_sysfs::gpu_handle::GpuHandle>,
}

impl std::fmt::Debug for SysSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysSampler").finish_non_exhaustive()
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

impl SysSampler {
    /// Build for the device kind and index the run uses.
    #[must_use]
    pub fn new(kind: DeviceKind, index: usize) -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        let _ = (kind, index);
        Self {
            sys,
            #[cfg(feature = "cuda")]
            nvml: (kind == DeviceKind::Cuda)
                .then(|| nvml_wrapper::Nvml::init().ok())
                .flatten()
                .map(|n| (n, u32::try_from(index).unwrap_or(0))),
            #[cfg(feature = "rocm")]
            amd: (kind == DeviceKind::Rocm)
                .then(|| {
                    amdgpu_sysfs::gpu_handle::GpuHandle::new_from_path(
                        format!("/sys/class/drm/card{index}/device").into(),
                    )
                    .ok()
                })
                .flatten(),
        }
    }

    /// Take a sample.
    pub fn sample(&mut self) -> SysSample {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        #[allow(clippy::cast_precision_loss, unused_mut)]
        let mut s = SysSample {
            cpu: f64::from(self.sys.global_cpu_usage()),
            mem_gb: self.sys.used_memory() as f64 / GIB,
            gpu_util: None,
            gpu_mem_gb: None,
        };
        #[cfg(feature = "cuda")]
        if let Some((nvml, idx)) = &self.nvml
            && let Ok(dev) = nvml.device_by_index(*idx)
        {
            s.gpu_util = dev.utilization_rates().ok().map(|u| f64::from(u.gpu));
            #[allow(clippy::cast_precision_loss)]
            {
                s.gpu_mem_gb = dev.memory_info().ok().map(|m| m.used as f64 / GIB);
            }
        }
        #[cfg(feature = "rocm")]
        if let Some(gpu) = &self.amd {
            s.gpu_util = gpu.get_busy_percent().ok().map(f64::from);
            #[allow(clippy::cast_precision_loss)]
            {
                s.gpu_mem_gb = gpu.get_used_vram().ok().map(|b| b as f64 / GIB);
            }
        }
        s
    }
}

/// The host name for `status.json`.
#[must_use]
pub fn hostname() -> String {
    System::host_name().unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_sane_on_the_host() {
        let mut s = SysSampler::new(DeviceKind::Cpu, 0);
        let a = s.sample();
        assert!(a.mem_gb > 0.0);
        assert!((0.0..=100.0).contains(&a.cpu));
        assert!(!hostname().is_empty());
    }
}
