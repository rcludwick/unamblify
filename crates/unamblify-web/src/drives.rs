// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Physical drive temperatures, for the Host page's chart.
//!
//! The corpus lives on an external `NVMe`, and that drive has dropped off
//! the bus under sustained load. Whether it is heating up is worth
//! watching, so the server samples every drive's temperature on a timer
//! and keeps a bounded history the UI can plot. Point-in-time numbers
//! would not show a climb; the history is the whole point.
//!
//! Temperatures come from `smartctl`, which is required: a chart that
//! silently shows nothing is worse than no chart, so the server refuses
//! to start without it.
//!
//! Not every enclosure passes SMART through. A USB bridge may report a
//! drive's identity and no health at all, so a drive with no readable
//! temperature is listed with a note rather than hidden.

use std::collections::{BTreeMap, VecDeque};
use std::process::Command;

use serde::Serialize;

/// How many samples the history keeps: a day at 30 s.
pub const HISTORY: usize = 2_880;

/// One drive as the Host page sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Drive {
    /// Device node, `/dev/disk8`.
    pub device: String,
    /// Model, from SMART or `diskutil`.
    pub model: String,
    /// Composite temperature in Celsius, absent when the bridge does not
    /// pass SMART through.
    pub composite_c: Option<i32>,
    /// Per-sensor temperatures, hottest first. An `NVMe` often reports a
    /// controller sensor well above the composite.
    pub sensors: Vec<i32>,
    /// Why there is no temperature, when there is none.
    pub note: Option<String>,
}

/// One moment: every drive that reported, by device node.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sample {
    /// Unix seconds.
    pub t: u64,
    /// Hottest reading per device, Celsius.
    pub temps: BTreeMap<String, i32>,
    /// Read and write rate per device over the interval that ended here,
    /// MiB/s ([`crate::diskio`]). Empty for the first sample after a start
    /// and for history written before the rates were kept.
    pub io: BTreeMap<String, crate::diskio::Rate>,
}

/// A bounded ring of samples.
#[derive(Debug, Default)]
pub struct History {
    samples: VecDeque<Sample>,
    last: Vec<Drive>,
}

impl History {
    /// Add a sample, dropping the oldest past [`HISTORY`].
    pub fn push(&mut self, s: Sample) {
        if self.samples.len() >= HISTORY {
            self.samples.pop_front();
        }
        self.samples.push_back(s);
    }

    /// Record the most recent probe, which is what the API serves.
    pub fn set_last(&mut self, d: Vec<Drive>) {
        self.last = d;
    }

    /// The most recent probe: every drive, with its note if unreadable.
    #[must_use]
    pub fn last(&self) -> Vec<Drive> {
        self.last.clone()
    }

    /// Every sample held, oldest first.
    #[must_use]
    pub fn samples(&self) -> Vec<Sample> {
        self.samples.iter().cloned().collect()
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

/// Refuse to run without `smartctl`.
///
/// The temperature chart is the reason this module exists, and a chart
/// that quietly plots nothing hides exactly the failure it was added to
/// catch.
pub fn require_smartctl() -> Result<(), String> {
    match Command::new("smartctl").arg("--version").output() {
        Ok(o) if o.status.success() => Ok(()),
        Ok(_) | Err(_) => Err(
            "smartctl not found: the Host page's drive temperatures need it \
             (brew install smartmontools)"
                .to_owned(),
        ),
    }
}

/// Physical disks, as `/dev/diskN`.
///
/// `smartctl --scan` returns `IOService` paths on macOS, which are useless
/// as labels and do not line up with anything else the harness prints.
/// `diskutil` gives the device nodes smartctl also accepts.
#[must_use]
pub fn devices() -> Vec<String> {
    let Ok(out) = Command::new("diskutil").arg("list").output() else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.contains("physical"))
        .filter_map(|l| l.split_whitespace().next())
        .filter(|d| d.starts_with("/dev/disk"))
        .map(ToOwned::to_owned)
        .collect()
}

/// The value after the last colon, trimmed.
fn field(line: &str) -> String {
    line.rsplit(':').next().unwrap_or("").trim().to_owned()
}

/// Read one drive. Tries `-d nvme` first, since a USB bridge often needs
/// it, then plain.
///
/// smartctl's exit status is a bitmask, not pass or fail: the internal
/// Apple SSD exits 4 while still reporting a temperature, so the output
/// is parsed regardless of status.
#[must_use]
pub fn probe_one(device: &str) -> Drive {
    let mut model = String::new();
    let mut composite = None;
    let mut sensors: Vec<i32> = Vec::new();

    for args in [vec!["-a", "-d", "nvme", device], vec!["-a", device]] {
        let Ok(out) = Command::new("smartctl").args(&args).output() else {
            continue;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let l = line.trim();
            if (l.starts_with("Model Number:") || l.starts_with("Device Model:"))
                && model.is_empty()
            {
                model = field(l);
            } else if l.starts_with("Temperature:") {
                composite = field(l)
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok());
            } else if l.starts_with("Temperature Sensor ")
                && let Some(v) = field(l)
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
            {
                sensors.push(v);
            }
        }
        if composite.is_some() || !model.is_empty() {
            break;
        }
    }

    sensors.sort_unstable_by(|a, b| b.cmp(a));
    let note = if composite.is_none() && sensors.is_empty() {
        Some("no temperature: this enclosure does not pass SMART health through".to_owned())
    } else {
        None
    };
    Drive {
        device: device.to_owned(),
        model: if model.is_empty() {
            "unknown".to_owned()
        } else {
            model
        },
        composite_c: composite,
        sensors,
        note,
    }
}

/// Every physical drive, probed.
#[must_use]
pub fn probe() -> Vec<Drive> {
    devices().iter().map(|d| probe_one(d)).collect()
}

/// The hottest reading a drive reported, for the history series.
#[must_use]
pub fn hottest(d: &Drive) -> Option<i32> {
    d.sensors.first().copied().or(d.composite_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_drops_the_oldest() {
        let mut h = History::default();
        assert!(h.is_empty());
        for t in 0..(HISTORY as u64 + 10) {
            h.push(Sample {
                t,
                temps: BTreeMap::new(),
                io: BTreeMap::new(),
            });
        }
        assert_eq!(h.len(), HISTORY);
        assert_eq!(h.samples().first().map(|s| s.t), Some(10));
    }

    #[test]
    fn a_drive_with_no_sensors_says_why() {
        let d = Drive {
            device: "/dev/disk6".to_owned(),
            model: "Western Digital SN580E 1TB".to_owned(),
            composite_c: None,
            sensors: vec![],
            note: Some("x".to_owned()),
        };
        assert_eq!(hottest(&d), None);
    }

    #[test]
    fn the_hottest_sensor_wins_over_the_composite() {
        let d = Drive {
            device: "/dev/disk8".to_owned(),
            model: "Samsung SSD 9100 PRO 8TB".to_owned(),
            composite_c: Some(52),
            sensors: vec![63, 52],
            note: None,
        };
        assert_eq!(hottest(&d), Some(63));
    }
}
