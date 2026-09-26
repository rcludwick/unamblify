// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Wall-clock helpers: Unix milliseconds for `metrics.jsonl`, RFC 3339 UTC
//! for `status.json`, and the `YYYYMMDD-HHMMSS-<name>` run id. No
//! dependency: the civil-date conversion is Howard Hinnant's algorithm.

use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since the Unix epoch.
#[must_use]
pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Broken-down UTC time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utc {
    /// Year.
    pub year: i64,
    /// Month, 1–12.
    pub month: u32,
    /// Day of month, 1–31.
    pub day: u32,
    /// Hour, 0–23.
    pub hour: u32,
    /// Minute, 0–59.
    pub minute: u32,
    /// Second, 0–59.
    pub second: u32,
}

/// Convert Unix milliseconds to UTC civil time.
#[must_use]
pub fn utc(unix_ms: u64) -> Utc {
    let secs = i64::try_from(unix_ms / 1000).unwrap_or(i64::MAX);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Hinnant, "civil_from_days".
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    // Fields are bounded by the arithmetic above; the casts cannot truncate.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Utc {
        year,
        month: m as u32,
        day: d as u32,
        hour: (sod / 3600) as u32,
        minute: (sod % 3600 / 60) as u32,
        second: (sod % 60) as u32,
    }
}

/// `2026-09-10T05:00:00Z`.
#[must_use]
pub fn rfc3339(unix_ms: u64) -> String {
    let t = utc(unix_ms);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        t.year, t.month, t.day, t.hour, t.minute, t.second
    )
}

/// [`rfc3339`] of now.
#[must_use]
pub fn rfc3339_now() -> String {
    rfc3339(unix_ms())
}

/// `YYYYMMDD-HHMMSS-<name>` (spec §7).
#[must_use]
pub fn run_id(name: &str, unix_ms: u64) -> String {
    let t = utc(unix_ms);
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}-{name}",
        t.year, t.month, t.day, t.hour, t.minute, t.second
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_conversion_is_right_on_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        // 2026-09-10T05:00:00Z = 1789016400 s.
        assert_eq!(rfc3339(1_789_016_400_000), "2026-09-10T05:00:00Z");
        // Leap day.
        assert_eq!(rfc3339(1_709_164_800_000), "2024-02-29T00:00:00Z");
        assert_eq!(run_id("x", 1_789_016_400_000), "20260910-050000-x");
        assert!(unix_ms() > 1_700_000_000_000);
    }
}
