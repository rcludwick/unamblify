// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `/api/sys` and the device list for `/api/health`: host facts from
//! `sysinfo` plus a cheap probe of which training devices this host can
//! offer (`cpu` always; `mps` on Apple Silicon; `cuda` / `rocm` when the
//! driver nodes are present).

use std::path::Path;

use serde::Serialize;
use sysinfo::System;

/// One training device the host can offer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Device {
    /// Device string as the run config's `[train] device` expects.
    pub name: String,
    /// Why it was listed.
    pub note: String,
}

/// Devices this host can train on.
#[must_use]
pub fn devices() -> Vec<Device> {
    let mut out = vec![Device {
        name: "cpu".to_owned(),
        note: format!("{} logical cpus", num_cpus()),
    }];
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        out.push(Device {
            name: "mps".to_owned(),
            note: "Apple Silicon Metal (correctness and conv-heavy sweeps)".to_owned(),
        });
    }
    if Path::new("/dev/nvidia0").exists() || which("nvidia-smi") {
        out.push(Device {
            name: "cuda:0".to_owned(),
            note: "NVIDIA driver present".to_owned(),
        });
    }
    if Path::new("/dev/kfd").exists() {
        out.push(Device {
            name: "rocm:0".to_owned(),
            note: "/dev/kfd present".to_owned(),
        });
    }
    out
}

fn which(bin: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
}

/// Logical CPU count.
#[must_use]
pub fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// The host name the trainer would record.
#[must_use]
pub fn hostname() -> String {
    System::host_name().unwrap_or_else(|| "unknown".to_owned())
}

/// `/api/sys`.
#[derive(Debug, Clone, Serialize)]
pub struct SysSnapshot {
    /// Host name.
    pub host: String,
    /// OS name and version.
    pub os: String,
    /// Kernel version.
    pub kernel: String,
    /// CPU architecture.
    pub arch: String,
    /// Logical CPUs.
    pub cpus: usize,
    /// Physical cores, when known.
    pub physical_cores: Option<usize>,
    /// Global CPU usage since the previous snapshot, percent.
    pub cpu_percent: f32,
    /// Load averages (1, 5, 15 min).
    pub load: [f64; 3],
    /// Memory total, bytes.
    pub mem_total: u64,
    /// Memory used, bytes.
    pub mem_used: u64,
    /// Host uptime, seconds.
    pub uptime_s: u64,
    /// Devices this host can train on.
    pub devices: Vec<Device>,
    /// This server's pid.
    pub pid: u32,
    /// The executable the supervisor re-execs for `train` / `capture`.
    pub exe: String,
    /// Runs directory being served.
    pub runs_dir: String,
    /// Data root (capture status lives under it).
    pub data_root: String,
    /// Server uptime, seconds.
    pub server_uptime_s: u64,
}

/// Refresh `sys` and take a snapshot.
#[must_use]
pub fn snapshot(
    sys: &mut System,
    exe: &Path,
    runs_dir: &Path,
    data_root: &Path,
    server_started_s: u64,
) -> SysSnapshot {
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let load = System::load_average();
    SysSnapshot {
        host: hostname(),
        os: format!(
            "{} {}",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default()
        )
        .trim()
        .to_owned(),
        kernel: System::kernel_version().unwrap_or_default(),
        arch: System::cpu_arch(),
        cpus: num_cpus(),
        physical_cores: System::physical_core_count(),
        cpu_percent: sys.global_cpu_usage(),
        load: [load.one, load.five, load.fifteen],
        mem_total: sys.total_memory(),
        mem_used: sys.used_memory(),
        uptime_s: System::uptime(),
        devices: devices(),
        pid: std::process::id(),
        exe: exe.display().to_string(),
        runs_dir: runs_dir.display().to_string(),
        data_root: data_root.display().to_string(),
        server_uptime_s: crate::clock::now_s().saturating_sub(server_started_s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_offered_first() {
        let d = devices();
        assert_eq!(d[0].name, "cpu");
        assert!(num_cpus() >= 1);
    }
}
