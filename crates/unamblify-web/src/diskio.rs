// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//! Per-drive I/O rate, for the Host page's chart beside the temperatures.
//!
//! A drive's temperature is an effect; what it is being asked to do is the
//! cause, and the two only explain each other side by side: a shard build
//! reading flat out, a backup packing, a capture trickling. The operating
//! system keeps cumulative bytes read and written per physical drive, so a
//! rate is the difference between two samples over the time between them.
//!
//! macOS: `ioreg -r -c IOBlockStorageDriver -l` — each driver's
//! `Statistics` dictionary, followed by the whole-disk `BSD Name` it
//! serves. Linux: `/proc/diskstats`, sectors of 512 bytes, whole disks
//! only. Anywhere else, and whenever a source cannot be read, there are
//! simply no rates: the temperature chart does not depend on this one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Cumulative bytes a drive has moved since boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Bytes read.
    pub read: u64,
    /// Bytes written.
    pub write: u64,
}

/// A drive's rate over one sampling interval, MiB per second.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    /// Read, MiB/s.
    pub r: f32,
    /// Written, MiB/s.
    pub w: f32,
}

/// The number after `key` in an ioreg dictionary line (`"key"=123`).
fn field(line: &str, key: &str) -> Option<u64> {
    let at = line.find(key)? + key.len();
    let rest = line[at..].trim_start_matches(['"', '=', ' ']);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// `ioreg -r -c IOBlockStorageDriver -l`: per driver block, the first
/// `Statistics` line is the driver's own and the first `BSD Name` after it
/// is the whole disk. Deeper `Statistics` (the media's, APFS's) are
/// skipped by taking only the first of each per block.
#[must_use]
pub fn parse_ioreg(text: &str) -> BTreeMap<String, Counters> {
    let mut out = BTreeMap::new();
    let mut stats: Option<Counters> = None;
    let mut named = true;
    for line in text.lines() {
        if line.contains("+-o IOBlockStorageDriver") {
            stats = None;
            named = false;
        } else if stats.is_none() && line.contains("\"Statistics\"") {
            if let (Some(read), Some(write)) = (
                field(line, "\"Bytes (Read)\""),
                field(line, "\"Bytes (Write)\""),
            ) {
                stats = Some(Counters { read, write });
            }
        } else if !named && line.contains("\"BSD Name\"") {
            if let (Some(c), Some(name)) = (stats, line.split('"').nth(3)) {
                out.insert(format!("/dev/{name}"), c);
            }
            named = true;
        }
    }
    out
}

/// `/proc/diskstats`: `major minor name reads _ sectors_read _ writes _
/// sectors_written …`. Whole disks only — a partition's name ends in a
/// digit after a disk's (`sda1`, `nvme0n1p2`), and counting both would
/// count the same bytes twice.
#[must_use]
pub fn parse_diskstats(text: &str) -> BTreeMap<String, Counters> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (Some(name), Some(rd), Some(wr)) = (f.get(2), f.get(5), f.get(9)) else {
            continue;
        };
        let whole = if name.starts_with("nvme") || name.starts_with("mmcblk") {
            !name.contains('p')
        } else {
            (name.starts_with("sd") || name.starts_with("vd"))
                && !name.ends_with(|c: char| c.is_ascii_digit())
        };
        if let (true, Ok(rd), Ok(wr)) = (whole, rd.parse::<u64>(), wr.parse::<u64>()) {
            out.insert(
                format!("/dev/{name}"),
                Counters {
                    read: rd * 512,
                    write: wr * 512,
                },
            );
        }
    }
    out
}

/// Every drive's counters now; empty where there is no source.
#[must_use]
pub fn probe() -> BTreeMap<String, Counters> {
    if cfg!(target_os = "macos") {
        std::process::Command::new("ioreg")
            .args(["-r", "-c", "IOBlockStorageDriver", "-l", "-w", "0"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| parse_ioreg(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or_default()
    } else {
        std::fs::read_to_string("/proc/diskstats")
            .map(|t| parse_diskstats(&t))
            .unwrap_or_default()
    }
}

/// Rates between two probes `dt_s` seconds apart. A drive missing from
/// either, or whose counter went backwards (re-plugged, so it restarted
/// from zero), has no rate for this interval rather than a nonsense one.
#[must_use]
pub fn rates(
    prev: &BTreeMap<String, Counters>,
    now: &BTreeMap<String, Counters>,
    dt_s: f64,
) -> BTreeMap<String, Rate> {
    const MIB: f64 = 1024.0 * 1024.0;
    let mut out = BTreeMap::new();
    if dt_s <= 0.0 {
        return out;
    }
    for (dev, n) in now {
        let Some(p) = prev.get(dev) else { continue };
        let (Some(dr), Some(dw)) = (n.read.checked_sub(p.read), n.write.checked_sub(p.write))
        else {
            continue;
        };
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let rate = Rate {
            r: (dr as f64 / MIB / dt_s) as f32,
            w: (dw as f64 / MIB / dt_s) as f32,
        };
        out.insert(dev.clone(), rate);
    }
    out
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const IOREG: &str = r#"+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x10002e67c, registered>
  |   "IOClass" = "IOBlockStorageDriver"
  |   "Statistics" = {"Operations (Write)"=44166758,"Bytes (Read)"=2376793851904,"Errors (Write)"=0,"Bytes (Write)"=637312151552,"Operations (Read)"=151795834}
    |   "BSD Name" = "disk8"
          |   "Statistics" = {"Operations (Read)"=1,"Bytes (Write)"=7,"Bytes (Read)"=9}
            |   "BSD Name" = "disk9"
+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x1000008e6, registered>
  |   "Statistics" = {"Bytes (Read)"=2174566457344,"Bytes (Write)"=2177403920384}
    |   "BSD Name" = "disk0"
"#;

    #[test]
    fn ioreg_gives_the_whole_disk_and_the_drivers_own_counters() {
        let c = parse_ioreg(IOREG);
        assert_eq!(
            c.len(),
            2,
            "the APFS container under disk8 is not a drive: {c:?}"
        );
        assert_eq!(
            c["/dev/disk8"],
            Counters {
                read: 2_376_793_851_904,
                write: 637_312_151_552
            }
        );
        assert_eq!(c["/dev/disk0"].write, 2_177_403_920_384);
    }

    #[test]
    fn diskstats_counts_whole_disks_once() {
        let text = "   8       0 sda 100 0 2048 0 50 0 4096 0 0 0 0\n   8       1 sda1 90 0 2000 0 40 0 4000 0 0 0 0\n 259       0 nvme0n1 5 0 1000 0 5 0 3000 0 0 0 0\n 259       1 nvme0n1p1 5 0 900 0 5 0 2900 0 0 0 0\n   7       0 loop0 1 0 8 0 0 0 0 0 0 0 0\n";
        let c = parse_diskstats(text);
        assert_eq!(c.keys().collect::<Vec<_>>(), ["/dev/nvme0n1", "/dev/sda"]);
        assert_eq!(
            c["/dev/sda"],
            Counters {
                read: 2048 * 512,
                write: 4096 * 512
            }
        );
    }

    #[test]
    fn a_rate_is_the_difference_over_the_interval_and_a_reset_is_no_rate() {
        let mib = 1024 * 1024;
        let at = |r: u64, w: u64| Counters { read: r, write: w };
        let prev = BTreeMap::from([
            ("/dev/a".to_owned(), at(0, 10 * mib)),
            ("/dev/b".to_owned(), at(900, 900)),
        ]);
        let now = BTreeMap::from([
            ("/dev/a".to_owned(), at(300 * mib, 40 * mib)),
            ("/dev/b".to_owned(), at(5, 5)), // re-plugged: counters restarted
            ("/dev/c".to_owned(), at(1, 1)), // new this interval
        ]);
        let r = rates(&prev, &now, 30.0);
        assert_eq!(r.len(), 1);
        assert_eq!(r["/dev/a"], Rate { r: 10.0, w: 1.0 });
        assert!(rates(&prev, &now, 0.0).is_empty());
    }
}
