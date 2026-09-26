// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The `[train] device` string: `cpu` | `mps` | `cuda:N` | `rocm:N`.
//! `ROCm` builds of libtorch present GPUs through the CUDA device type, so
//! `rocm:N` maps to `tch::Device::Cuda(N)` too; the distinction only
//! matters for the `sys/gpu_*` metrics sampler.

use tch::Device;

/// A parsed device string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSpec {
    /// The libtorch device.
    pub device: Device,
    /// Which GPU stack the string named, for the metrics sampler.
    pub kind: DeviceKind,
}

/// What the device string named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Host CPU.
    Cpu,
    /// Apple Metal.
    Mps,
    /// NVIDIA.
    Cuda,
    /// AMD (libtorch's HIP build, addressed as CUDA).
    Rocm,
}

/// A device string that is not one of the four shapes, or names a device
/// this libtorch cannot see.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DeviceError {
    /// Not `cpu`, `mps`, `cuda:N` or `rocm:N`.
    #[error("unknown device {0:?}: expected cpu | mps | cuda:N | rocm:N")]
    Unknown(String),
    /// Parsed but unavailable at run time.
    #[error("device {0:?} is not available in this libtorch build / on this host")]
    Unavailable(String),
}

/// Parse without checking availability.
pub fn parse(s: &str) -> Result<DeviceSpec, DeviceError> {
    let s = s.trim();
    let index = |rest: &str| {
        rest.parse::<usize>()
            .map_err(|_| DeviceError::Unknown(s.to_owned()))
    };
    let (device, kind) = match s {
        "cpu" => (Device::Cpu, DeviceKind::Cpu),
        "mps" => (Device::Mps, DeviceKind::Mps),
        _ => {
            if let Some(n) = s.strip_prefix("cuda:") {
                (Device::Cuda(index(n)?), DeviceKind::Cuda)
            } else if let Some(n) = s.strip_prefix("rocm:") {
                (Device::Cuda(index(n)?), DeviceKind::Rocm)
            } else {
                return Err(DeviceError::Unknown(s.to_owned()));
            }
        }
    };
    Ok(DeviceSpec { device, kind })
}

/// Parse and check that libtorch can use the device.
pub fn resolve(s: &str) -> Result<DeviceSpec, DeviceError> {
    let spec = parse(s)?;
    let ok = match spec.device {
        Device::Cpu => true,
        Device::Mps => tch::utils::has_mps(),
        Device::Cuda(n) => {
            tch::Cuda::is_available()
                && i64::try_from(n).is_ok_and(|n| n < tch::Cuda::device_count())
        }
        Device::Vulkan => false,
    };
    if ok {
        Ok(spec)
    } else {
        Err(DeviceError::Unavailable(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_shapes_parse() {
        assert_eq!(parse("cpu").unwrap().device, Device::Cpu);
        assert_eq!(parse("mps").unwrap().kind, DeviceKind::Mps);
        assert_eq!(parse("cuda:1").unwrap().device, Device::Cuda(1));
        let r = parse("rocm:0").unwrap();
        assert_eq!((r.device, r.kind), (Device::Cuda(0), DeviceKind::Rocm));
        assert!(matches!(parse("gpu"), Err(DeviceError::Unknown(_))));
        assert!(matches!(parse("cuda:x"), Err(DeviceError::Unknown(_))));
        assert!(matches!(parse("cuda"), Err(DeviceError::Unknown(_))));
        assert!(resolve("cpu").is_ok());
        assert!(matches!(
            resolve("cuda:99"),
            Err(DeviceError::Unavailable(_))
        ));
    }
}
