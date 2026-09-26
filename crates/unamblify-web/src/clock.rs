// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Wall-clock helpers: Unix milliseconds, RFC 3339 UTC strings and the
//! `YYYYMMDD-HHMMSS-<name>` run id of spec §7. Hand-rolled civil-date
//! arithmetic (Howard Hinnant's `civil_from_days`) so the crate needs no
//! calendar dependency.

use std::time::{SystemTime, UNIX_EPOCH};

/// Now, Unix milliseconds.
#[must_use]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Now, Unix seconds.
#[must_use]
pub fn now_s() -> u64 {
    now_ms() / 1000
}

/// Broken-down UTC time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Civil {
    /// Year.
    pub year: i64,
    /// Month, 1–12.
    pub month: u32,
    /// Day, 1–31.
    pub day: u32,
    /// Hour, 0–23.
    pub hour: u32,
    /// Minute, 0–59.
    pub minute: u32,
    /// Second, 0–59.
    pub second: u32,
}

/// Unix seconds → UTC civil time.
#[must_use]
pub fn civil(unix_s: u64) -> Civil {
    let secs = i64::try_from(unix_s).unwrap_or(i64::MAX);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Hinnant: days since 1970-01-01 → (y, m, d).
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
    // All of these are bounded small non-negative values.
    let to_u32 = |v: i64| u32::try_from(v).unwrap_or(0);
    Civil {
        year,
        month: to_u32(m),
        day: to_u32(d),
        hour: to_u32(sod / 3600),
        minute: to_u32(sod % 3600 / 60),
        second: to_u32(sod % 60),
    }
}

/// `2026-09-10T05:00:00Z` for a Unix-seconds instant.
#[must_use]
pub fn rfc3339(unix_s: u64) -> String {
    let c = civil(unix_s);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        c.year, c.month, c.day, c.hour, c.minute, c.second
    )
}

/// Now as RFC 3339 UTC.
#[must_use]
pub fn rfc3339_now() -> String {
    rfc3339(now_s())
}

/// `YYYYMMDD-HHMMSS-<name>` for a Unix-seconds instant.
#[must_use]
pub fn run_id(unix_s: u64, name: &str) -> String {
    let c = civil(unix_s);
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}-{name}",
        c.year, c.month, c.day, c.hour, c.minute, c.second
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        // 2026-09-10T05:00:00Z
        assert_eq!(rfc3339(1_789_016_400), "2026-09-10T05:00:00Z");
        assert_eq!(run_id(1_789_016_400, "smoke"), "20260910-050000-smoke");
        // Leap day.
        assert_eq!(rfc3339(1_709_164_800), "2024-02-29T00:00:00Z");
    }
}
