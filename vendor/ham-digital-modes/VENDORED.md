# Vendored: `ham_digital_modes` (D-STAR AMBE subset)

The D-STAR AMBE vocoder from Bruce Perens' `hams_open`, **reduced to the
encode/decode path this repository uses**. It gives unamblify a software
D-STAR codec to set beside the AMBE-3000 chip captures.

| | |
|---|---|
| Upstream | <https://github.com/BrucePerens/hams_open> |
| Path upstream | `daemons/ham_digital_modes` |
| Revision | `e403fcf39a00ab926984838fd23ab0ecc89d2462` (2026-09-19) |
| Licence | LGPL-3.0-or-later (see `LICENSE`; every source file keeps its SPDX header) |
| Upstream crate name | `ham_digital_modes` (renamed here to `ham-digital-modes-dstar`) |

## Why it is vendored

The workspace is AGPL-3.0-only. This crate is LGPL-3.0-or-later and stays
that way: it lives under `vendor/` so the licence boundary is visible in the
tree, and its `Cargo.toml` hard-codes `license`, `version`, `edition` and
`repository` rather than inheriting them from the workspace — inheriting
would resolve `license` to `AGPL-3.0-only` and silently relicense someone
else's terms. Same reasoning as `vendor/ambe-thumbdv/VENDORED.md`.

It is not published to crates.io, so the alternative was a git dependency;
vendoring keeps the build offline and reproducible, and pins exactly which
revision of a young, fast-moving codec produced a given capture set.

## This copy is NOT verbatim

`vendor/ambe-thumbdv` is vendored verbatim and never edited. **This one is
not**, and the difference matters when reconciling against upstream.
Upstream is 115 files / 44,678 lines covering FT8, WSPR, PSK31, RTTY, its
own Codec 2, a fixed-point AMBE port and the P25/AMBE+2 vocoders. Only the
floating-point D-STAR path is kept: 45 files / 12,589 lines.

Removed:

| Removed | Why |
|---|---|
| `src/ambe/float/ambe_plus_2/` | AMBE+2's 2009 addendum names live patents (D-STAR's expired 2017). Upstream gates it off by default; it covers `ysf-dmr`, which this path cannot serve anyway. |
| `src/ambe/fixed/` | Fixed-point mirror for CPUs with no FPU. We run the float path. |
| `src/codec2_3200/`, `src/codec2_1600/`, `vendor/codec2-mod` | We have our own Codec 2 path (the `codec2` crate). |
| `src/ft8*`, `src/wspr*`, `src/psk31.rs`, `src/rtty.rs`, `vendor/ft8_lib` | Unrelated digital modes; `ft8_lib` is vendored C. |
| `src/dstar.rs` | D-STAR *protocol* framing (header/checksum), not the vocoder. |
| `src/timing_characterizer.rs` | Unused. |
| `examples/`, `tests/` | Upstream's chip-validation harnesses need a DVSI chip on the LAN. |

`src/lib.rs` and `src/ambe/mod.rs` are rewritten to declare only what is
kept; `src/ambe/float/mod.rs` has its `ambe_plus_2` declaration removed,
along with the comment above it describing that module's patent gating
(left in place it reads as though D-STAR were the encumbered mode).
Nothing else is edited: every kept file is byte-for-byte upstream.

## Linting

Unlike `vendor/ambe-thumbdv`, this crate's `Cargo.toml` carries a small
`[lints.clippy]` section. It has to: `just clippy` runs `-D warnings` over
the whole compilation and this crate is built as a dependency of
`unamblify-data`, so `--exclude` cannot spare it (that excludes a package
as a build target, not as a dependency). Upstream trips two default-level
`needless_range_loop` warnings.

The exemption lives in the manifest so every vendored **source** file stays
byte-for-byte upstream — editing the sources to satisfy our lint settings
would break re-derivation for no benefit. Only the lints upstream actually
trips are listed, so a re-vendor that trips a new one fails loudly rather
than being waved through.

## Re-deriving this copy

Clone upstream at the revision above, copy `daemons/ham_digital_modes/src`
and `LICENSE`, delete the paths in the table, rewrite the three `mod`
declarations, and confirm the result still round-trips bit-identically to
the unstripped crate (encode the same PCM through both and compare the
9-byte wire frames and decoded PCM). A strip that quietly drops a module
still compiles — the first attempt at this one left `ambe_plus_2`'s `cfg`
attribute dangling onto `pub mod dstar;`, which configured the entire
vocoder out of a crate that built without complaint.

## Provenance of the codec itself

Upstream reverse-derived D-STAR's frame structure and quantizer tables from
`mbelib` (ISC), confirmed the on-chip configuration against DVSI's USB-3000
manual (Table 30) and G4KLX's AMBETools, and validated the result live
against a real DVSI AMBE3003. Its own `docs/references/AMBE_CHIP_VALIDATION_FINDINGS.md`
is kept here for that trail.
