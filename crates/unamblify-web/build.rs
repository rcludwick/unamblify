// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The dashboard UI under `ui/` is embedded by `rust-embed` at macro
//! expansion time, which Cargo cannot see: without this, editing
//! `app.js` alone does not rebuild the crate and a freshly built server
//! serves the old script. Tell Cargo the directory is an input.
fn main() {
    println!("cargo:rerun-if-changed=ui");
    for entry in std::fs::read_dir("ui").into_iter().flatten().flatten() {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
}
