// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Whole-host compute: CPU use, load average and GPU use.
//!
//! CPU temperature is not readable on Apple Silicon without a private
//! API or an extra tool, but sustained compute is what drives the heat,
//! so these curves answer the same question the temperature would.
//!
//! GPU use comes from the accelerator's `PerformanceStatistics` in the
//! IO registry, which `ioreg` prints without privileges. The trainer's
//! own `sys/gpu_*` sampler knows only `NVML` and `ROCm`, so it reports
//! nothing on this machine; this is the Host page's own reader.

use std::collections::VecDeque;
use std::process::Command;

use serde::Serialize;
use sysinfo::System;

/// How many samples are kept: 24 h at one sample per 30 s.
pub const HISTORY: usize = 2_880;

/// Bytes per GiB.
const GIB: f64 = 1_073_741_824.0;

/// One host compute reading.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Sample {
    /// Unix seconds.
    pub t: u64,
    /// Whole-host CPU use, percent of all cores.
    pub cpu: f32,
    /// One-minute load average.
    pub load1: f64,
    /// GPU use, percent, when the accelerator reports it.
    pub gpu: Option<f32>,
    /// GPU memory in use, GiB, when the accelerator reports it.
    pub gpu_mem_gb: Option<f64>,
}

/// A bounded ring of host readings.
#[derive(Debug, Default)]
pub struct History {
    samples: VecDeque<Sample>,
}

impl History {
    /// Add a sample, dropping the oldest past [`HISTORY`].
    pub fn push(&mut self, s: Sample) {
        if self.samples.len() >= HISTORY {
            self.samples.pop_front();
        }
        self.samples.push_back(s);
    }

    /// Every sample held, oldest first.
    #[must_use]
    pub fn samples(&self) -> Vec<Sample> {
        self.samples.iter().copied().collect()
    }

    /// How many samples are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether nothing has been sampled yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// What the accelerator reports right now.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Gpu {
    /// Percent busy.
    pub util: Option<f32>,
    /// Memory in use, GiB.
    pub mem_gb: Option<f64>,
}

/// The unsigned number printed after `"<key>"=`, if it is there.
///
/// Keys are matched with their quotes, so `"In use system memory"`
/// cannot match `"In use system memory (driver)"`.
fn num_after(out: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"=");
    let rest = &out[out.find(&pat)? + pat.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Read GPU use out of `ioreg`'s accelerator statistics.
#[must_use]
pub fn parse_gpu(out: &str) -> Gpu {
    Gpu {
        // A percentage, so it fits a u16 long before it reaches f32.
        util: num_after(out, "Device Utilization %")
            .and_then(|v| u16::try_from(v).ok())
            .map(f32::from),
        // Byte counts stay exact in f64 well past any real memory size.
        mem_gb: num_after(out, "In use system memory")
            .and_then(|v| u32::try_from(v / 1024).ok())
            .map(|kib| f64::from(kib) * 1024.0 / GIB),
    }
}

/// Ask the IO registry what the GPU is doing.
///
/// Anything unexpected reads as "no GPU": the chart simply omits the
/// series rather than the server failing over a meter.
#[must_use]
pub fn probe_gpu() -> Gpu {
    let Ok(out) = Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"])
        .output()
    else {
        return Gpu::default();
    };
    parse_gpu(&String::from_utf8_lossy(&out.stdout))
}

/// Read CPU use from a dedicated [`System`], plus the GPU.
///
/// `global_cpu_usage` is the delta since this instance last refreshed,
/// so the sampler keeps its own and never shares the one the `/api/sys`
/// snapshot uses: two readers of one instance would each steal the
/// other's interval.
pub fn probe(sys: &mut System, gpu: Gpu) -> Sample {
    sys.refresh_cpu_usage();
    let load = System::load_average();
    Sample {
        t: crate::clock::now_s(),
        cpu: sys.global_cpu_usage(),
        load1: load.one,
        gpu: gpu.util,
        gpu_mem_gb: gpu.mem_gb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real line, as `ioreg -c AGXAccelerator` prints it on an M4 Pro.
    const REAL: &str = r#"      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=4084400128,"Tiler Utilization %"=0,"recoveryCount"=0,"Renderer Utilization %"=4,"Device Utilization %"=89,"In use system memory"=2220294144}"#;

    #[test]
    fn the_real_output_gives_use_and_memory() {
        let g = parse_gpu(REAL);
        assert_eq!(g.util, Some(89.0));
        let mem = g.mem_gb.expect("memory");
        assert!((mem - 2.068).abs() < 0.01, "got {mem} GiB");
    }

    #[test]
    fn the_driver_key_does_not_shadow_the_real_one() {
        // "In use system memory (driver)"=0 comes first in the output.
        assert!(parse_gpu(REAL).mem_gb.expect("memory") > 1.0);
    }

    #[test]
    fn nothing_useful_reads_as_no_gpu() {
        assert_eq!(parse_gpu("no accelerator here"), Gpu::default());
    }

    #[test]
    fn the_ring_drops_the_oldest() {
        let mut h = History::default();
        for i in 0..(HISTORY + 10) {
            h.push(Sample {
                t: i as u64,
                cpu: 1.0,
                load1: 1.0,
                gpu: None,
                gpu_mem_gb: None,
            });
        }
        assert_eq!(h.len(), HISTORY);
        assert_eq!(h.samples()[0].t, 10, "the oldest fell off the front");
    }

    #[test]
    fn a_fresh_history_is_empty() {
        assert!(History::default().is_empty());
    }
}
