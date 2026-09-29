// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Test helpers shared across modules: hex literals, the scripted init on
//! the vendored `MockTransport`, and small fixture generators.

use ambe_thumbdv::MockTransport;
use unamblify::VocoderMode;

use crate::chip::{dvsi_packet, ratep};

/// `"61 00 01 00 33"` → bytes.
pub fn hex(s: &str) -> Vec<u8> {
    s.split_whitespace()
        .map(|b| u8::from_str_radix(b, 16).unwrap())
        .collect()
}

/// The full init script for one mode on the vendored `MockTransport`,
/// pinning every request's wire bytes in order.
pub fn scripted_init(mode: VocoderMode) -> MockTransport {
    let mut m = MockTransport::new();
    m.expect(hex("61 00 01 00 33"), vec![hex("61 00 01 00 39")]);
    let mut prodid = hex("61 00 0B 00 30");
    prodid.extend_from_slice(b"AMBE3000F\0");
    m.expect(hex("61 00 01 00 30"), vec![prodid]);
    let ver = b"V121.E100.XXXX.C110.G514.R014.A0030608.C0020208\0";
    let version = dvsi_packet(0, &[&[0x31][..], &ver[..]].concat());
    m.expect(hex("61 00 01 00 31"), vec![version]);
    m.expect(ratep(mode).unwrap(), vec![hex("61 00 02 00 0A 00")]);
    m.expect(hex("61 00 02 00 0B 03"), vec![hex("61 00 02 00 0B 00")]);
    m.expect(hex("61 00 03 00 05 00 00"), vec![hex("61 00 02 00 05 00")]);
    m.expect(hex("61 00 03 00 06 00 00"), vec![hex("61 00 02 00 06 00")]);
    m.expect(hex("61 00 03 00 4B 00 00"), vec![hex("61 00 02 00 4B 00")]);
    m
}

/// A sine of `freq` Hz at `rate` for `n` samples, peak `amp`.
pub fn sine(freq: f32, rate: u32, n: usize, amp: f32) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let w = 2.0 * std::f32::consts::PI * freq / rate as f32;
    #[allow(clippy::cast_precision_loss)]
    (0..n).map(|i| amp * (w * i as f32).sin()).collect()
}
