// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The FLAC path through symphonia, on a fixture made with the reference
//! encoder: `sox -n -r 16000 -b 24 -c 2 x.wav synth 0.1 sine 440 vol 0.5
//! remix 1 0 && flac --best x.wav` — 440 Hz at peak 0.5 on the left,
//! silence on the right, 0.1 s, 16 kHz, 24-bit.

use std::path::PathBuf;

use unamblify_audio::{read, read_flac, stft};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sine440-stereo-24bit-16k.flac")
}

#[test]
fn flac_decodes_downmixes_and_scales() {
    let a = read_flac(fixture()).unwrap();
    assert_eq!(a.rate, 16_000);
    assert_eq!(a.samples.len(), 1_600);
    // Stereo downmix halves the left-only sine: peak 0.25.
    let peak = a.samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!((peak - 0.25).abs() < 0.002, "peak {peak}");
    // Frequency: 440 Hz at 16 kHz with n_fft 1600 is bin 44.
    let s = stft(&a.samples, 1_600, 400);
    let mid = &s[s.len() / 2];
    let bin = mid
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0;
    assert_eq!(bin, 44);
    assert_eq!(read(fixture()).unwrap(), a);
}
