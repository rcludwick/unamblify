// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The training dashboard (spec §8): an axum server over a runs directory,
//! a supervisor that re-execs this binary as the trainer or the capture
//! harness, SSE streams, and an embedded single-page UI.
//!
//! The trainer is never linked. This crate reads the run-directory formats
//! of the core crate ([`unamblify::RunStatus`], [`unamblify::MetricRow`],
//! [`unamblify::LogRow`], [`unamblify::RunConfig`]) and shells out.
//!
//! Entry points: [`serve`] for the CLI, [`build`] + [`router`] for tests.

pub mod api;
pub mod capture;
pub mod clock;
pub mod diskio;
pub mod downsample;
pub mod drives;
pub mod error;
pub mod events;
pub mod host;
pub mod persist;
pub mod runs;
pub mod samples;
pub mod state;
pub mod supervisor;
pub mod sys;
pub mod ui;
pub mod watch;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use axum::routing::get;
use tower_http::validate_request::{ValidateRequest, ValidateRequestHeaderLayer};

pub use error::WebError;
pub use state::AppState;

/// Default bind without a token.
pub const DEFAULT_BIND: &str = "127.0.0.1:8787";
/// Default bind with a token.
pub const TOKEN_BIND: &str = "0.0.0.0:8787";
/// Default data root when `$UNAMBLIFY_DATA` is unset.
pub const DEFAULT_DATA_ROOT: &str = "/Volumes/data/training_data/unamblify";

/// Options for [`serve`].
#[derive(Debug, Clone)]
pub struct ServeOpts {
    /// `runs/`. Defaults to `<data_root>/runs`.
    pub runs_dir: Option<PathBuf>,
    /// `$UNAMBLIFY_DATA`; capture status lives under it.
    pub data_root: Option<PathBuf>,
    /// Where run configs live. Defaults to `configs/` in the cwd.
    pub configs_dir: Option<PathBuf>,
    /// Address to bind. Defaults to [`DEFAULT_BIND`], or [`TOKEN_BIND`]
    /// when `token` is set.
    pub bind: Option<SocketAddr>,
    /// Bearer token for `/api/*`. Also switches the default bind to all
    /// interfaces.
    pub token: Option<String>,
    /// Executable to re-exec for `train` / `capture`. Defaults to
    /// `std::env::current_exe()`.
    pub exe: Option<PathBuf>,
    /// Grace between SIGTERM and SIGKILL on stop.
    pub grace: Duration,
    /// Poller tick (also the SSE coalescing window).
    pub poll_interval: Duration,
}

impl Default for ServeOpts {
    fn default() -> Self {
        Self {
            runs_dir: None,
            data_root: None,
            configs_dir: None,
            bind: None,
            token: None,
            exe: None,
            grace: Duration::from_secs(30),
            poll_interval: watch::DEFAULT_INTERVAL,
        }
    }
}

impl ServeOpts {
    /// The data root after defaults.
    #[must_use]
    pub fn data_root(&self) -> PathBuf {
        self.data_root.clone().unwrap_or_else(|| {
            std::env::var_os("UNAMBLIFY_DATA")
                .map_or_else(|| PathBuf::from(DEFAULT_DATA_ROOT), PathBuf::from)
        })
    }

    /// The runs dir after defaults.
    #[must_use]
    pub fn runs_dir(&self) -> PathBuf {
        self.runs_dir
            .clone()
            .unwrap_or_else(|| self.data_root().join("runs"))
    }

    /// The bind address after defaults.
    pub fn bind_addr(&self) -> anyhow::Result<SocketAddr> {
        if let Some(b) = self.bind {
            return Ok(b);
        }
        let s = if self.token.is_some() {
            TOKEN_BIND
        } else {
            DEFAULT_BIND
        };
        s.parse().context("default bind")
    }
}

/// Build the shared state (adopting live runs) and start the poller.
pub fn build(opts: &ServeOpts) -> anyhow::Result<Arc<AppState>> {
    let runs_dir = opts.runs_dir();
    let data_root = opts.data_root();
    let configs_dir = opts
        .configs_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("configs"));
    let exe = match &opts.exe {
        Some(e) => e.clone(),
        None => std::env::current_exe().context("current_exe")?,
    };
    std::fs::create_dir_all(&runs_dir)
        .with_context(|| format!("create runs dir {}", runs_dir.display()))?;
    let hub = Arc::new(events::Hub::default());
    let sup = Arc::new(supervisor::Supervisor::new(
        exe,
        opts.grace,
        Arc::clone(&hub),
    ));
    let captured_dir = data_root.join("captured");
    sup.adopt(&runs_dir, &captured_dir);
    let watcher = watch::Watcher::new(runs_dir.clone(), captured_dir, Arc::clone(&hub));
    tokio::spawn(watcher.run(opts.poll_interval));
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    // The temperature chart is why the drive module exists; a chart that
    // quietly plots nothing would hide the failure it was added to catch.
    drives::require_smartctl().map_err(|e| anyhow::anyhow!(e))?;
    // A day of history is the point, so it is reloaded rather than
    // restarted from empty every time the server bounces.
    let hist_path = persist::path(&data_root);
    let cutoff = clock::now_s().saturating_sub(persist::KEEP_S);
    let rows = persist::load(&hist_path, cutoff);
    persist::compact(&hist_path, &rows);
    let mut drive_hist = drives::History::default();
    let mut host_hist = host::History::default();
    for r in &rows {
        drive_hist.push(drives::Sample {
            t: r.t,
            temps: r.temps.clone(),
            io: r.io.clone(),
        });
        host_hist.push(host::Sample {
            t: r.t,
            cpu: r.cpu,
            load1: r.load1,
            gpu: r.gpu,
            gpu_mem_gb: r.gpu_mem_gb,
        });
    }
    let state = Arc::new(AppState {
        runs_dir,
        data_root,
        configs_dir,
        hub,
        sup,
        sys: Mutex::new(sys),
        started_s: clock::now_s(),
        metric_keys: Mutex::new(std::collections::HashMap::new()),
        samples: Mutex::new(samples::Cache::default()),
        drives: Mutex::new(drive_hist),
        host: Mutex::new(host_hist),
    });
    tokio::spawn(sample_host(Arc::clone(&state), hist_path, HOST_SAMPLE));
    Ok(state)
}

/// How often host temperatures and CPU are sampled.
pub const HOST_SAMPLE: Duration = Duration::from_secs(30);

/// Sample drives and CPU on a timer, into the rings and onto disk.
///
/// `smartctl` shells out per drive, so the probe runs on the blocking
/// pool and never holds a lock across it.
async fn sample_host(state: Arc<AppState>, hist_path: std::path::PathBuf, every: Duration) {
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Its own handle: CPU use is a delta since this instance last read.
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    // I/O is a rate between two readings of cumulative counters.
    let mut io_prev: Option<(std::time::Instant, _)> = None;
    loop {
        tick.tick().await;
        let io_now = tokio::task::spawn_blocking(diskio::probe)
            .await
            .unwrap_or_default();
        let read_at = std::time::Instant::now();
        let io = io_prev
            .as_ref()
            .map_or_else(Default::default, |(at, prev)| {
                diskio::rates(prev, &io_now, read_at.duration_since(*at).as_secs_f64())
            });
        io_prev = Some((read_at, io_now));
        let Ok(found) = tokio::task::spawn_blocking(drives::probe).await else {
            continue;
        };
        let mut temps = std::collections::BTreeMap::new();
        for d in &found {
            if let Some(c) = drives::hottest(d) {
                temps.insert(d.device.clone(), c);
            }
        }
        // ioreg shells out, so it goes to the blocking pool too.
        let gpu = tokio::task::spawn_blocking(host::probe_gpu)
            .await
            .unwrap_or_default();
        let c = host::probe(&mut sys, gpu);
        if let Ok(mut h) = state.drives.lock() {
            h.push(drives::Sample {
                t: c.t,
                temps: temps.clone(),
                io: io.clone(),
            });
            h.set_last(found);
        }
        if let Ok(mut h) = state.host.lock() {
            h.push(c);
        }
        let row = persist::Row {
            t: c.t,
            cpu: c.cpu,
            load1: c.load1,
            gpu: c.gpu,
            gpu_mem_gb: c.gpu_mem_gb,
            temps,
            io,
        };
        let p = hist_path.clone();
        let _ = tokio::task::spawn_blocking(move || persist::append(&p, &row)).await;
    }
}

/// The full router: UI at `/`, API under `/api`, optionally behind a
/// bearer token.
pub fn router(state: Arc<AppState>, token: Option<&str>) -> Router {
    let mut api = api::router();
    if let Some(t) = token {
        api = api.layer(ValidateRequestHeaderLayer::custom(BearerToken::new(t)));
    }
    Router::new()
        .route("/", get(ui::index))
        .route("/ui/{*path}", get(ui::asset))
        .route("/favicon.ico", get(ui::favicon))
        .nest("/api", api)
        .with_state(state)
}

/// `Authorization: Bearer <token>` check for `/api/*`. tower-http's own
/// `bearer` constructor is deprecated as "too basic"; this is the same
/// check with a constant-time comparison.
#[derive(Debug, Clone)]
pub struct BearerToken {
    expected: Arc<[u8]>,
}

impl BearerToken {
    /// A validator for `token`.
    #[must_use]
    pub fn new(token: &str) -> Self {
        Self {
            expected: Arc::from(token.as_bytes()),
        }
    }

    fn matches(&self, header: &[u8]) -> bool {
        let Some(given) = header.strip_prefix(b"Bearer ") else {
            return false;
        };
        let given = given.trim_ascii();
        if given.len() != self.expected.len() {
            return false;
        }
        given
            .iter()
            .zip(self.expected.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

impl<B> ValidateRequest<B> for BearerToken {
    type ResponseBody = axum::body::Body;

    fn validate(
        &mut self,
        request: &mut axum::http::Request<B>,
    ) -> std::result::Result<(), axum::http::Response<Self::ResponseBody>> {
        let ok = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .is_some_and(|h| self.matches(h.as_bytes()));
        if ok {
            Ok(())
        } else {
            let mut res =
                axum::http::Response::new(axum::body::Body::from("{\"error\":\"unauthorized\"}"));
            *res.status_mut() = axum::http::StatusCode::UNAUTHORIZED;
            res.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer"),
            );
            res.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            Err(res)
        }
    }
}

/// Run the server until the process is told to stop (Ctrl-C). Supervised
/// jobs are left running so a restarted server can adopt them.
pub async fn serve(opts: ServeOpts) -> anyhow::Result<()> {
    let addr = opts.bind_addr()?;
    if opts.token.is_none() && !addr.ip().is_loopback() {
        anyhow::bail!(
            "refusing to bind {addr} without --token; non-loopback binds need bearer auth"
        );
    }
    let state = build(&opts)?;
    let app = router(Arc::clone(&state), opts.token.as_deref());
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    eprintln!(
        "unamblify-web: http://{addr}/  runs={}  data={}  auth={}",
        state.runs_dir.display(),
        state.data_root.display(),
        if opts.token.is_some() {
            "bearer"
        } else {
            "none (loopback)"
        }
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("serve")
}
