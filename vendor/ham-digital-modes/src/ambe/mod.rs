// SPDX-License-Identifier: LGPL-3.0-or-later
//! AMBE vocoder family, reduced to the floating-point D-STAR path.
//!
//! Upstream also carries a `fixed` (no-FPU) mirror and the P25/AMBE+2
//! modes; both are removed here. See `VENDORED.md`.

pub mod float;
pub mod general;
