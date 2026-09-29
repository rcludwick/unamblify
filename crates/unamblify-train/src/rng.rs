// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The deterministic generator every loader and augmentation uses. It is
//! the core crate's `SplitMix64` so the shard builder in the data crate
//! draws exactly the same garbage tails from the same seed.

pub use unamblify::Rng;
