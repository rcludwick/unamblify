// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Local test binary for the dashboard. The product path is `unamblify
//! serve` in the CLI crate; this one takes the same flags without clap.
//!
//! `unamblify-web [--runs-dir D] [--data-root D] [--configs-dir D]
//! [--bind ADDR] [--token T] [--exe PATH]`
//!
//! Because the supervisor re-execs `current_exe()` and this binary has no
//! `train` / `capture` verbs, pass `--exe` pointing at the real `unamblify`
//! binary to start runs from here.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, bail};
use unamblify_web::ServeOpts;

fn main() -> anyhow::Result<()> {
    let mut opts = ServeOpts::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().with_context(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--runs-dir" => opts.runs_dir = Some(PathBuf::from(val("--runs-dir")?)),
            "--data-root" => opts.data_root = Some(PathBuf::from(val("--data-root")?)),
            "--configs-dir" => opts.configs_dir = Some(PathBuf::from(val("--configs-dir")?)),
            "--bind" => opts.bind = Some(val("--bind")?.parse().context("--bind")?),
            "--token" => opts.token = Some(val("--token")?),
            "--exe" => opts.exe = Some(PathBuf::from(val("--exe")?)),
            "--grace-s" => {
                opts.grace = Duration::from_secs(val("--grace-s")?.parse().context("--grace-s")?);
            }
            "-h" | "--help" => {
                println!(
                    "unamblify-web [--runs-dir D] [--data-root D] [--configs-dir D] [--bind ADDR] [--token T] [--exe PATH] [--grace-s N]"
                );
                return Ok(());
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?
        .block_on(unamblify_web::serve(opts))
}
