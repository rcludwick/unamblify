// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The HTTP API (spec §8, §3). Every route is JSON except the SSE streams
//! and the checkpoint audio passthrough.
//!
//! | Method | Path | What |
//! |---|---|---|
//! | GET | `/api/health` | `{ok, version, host, runs_dir, devices}` |
//! | GET | `/api/configs` | configs in the configs dir |
//! | GET | `/api/configs/{name}` | one config: TOML text + parsed |
//! | POST | `/api/configs/validate` | `{toml}` → `{ok, config}` or 400 |
//! | GET | `/api/runs` | run list |
//! | POST | `/api/runs` | `{toml \| config, overrides?}` → create + start |
//! | GET | `/api/runs/{id}` | status, config, checkpoints, metric keys |
//! | POST | `/api/runs/{id}/stop` | SIGTERM → grace → SIGKILL |
//! | POST | `/api/runs/{id}/resume` | re-exec with `--resume <latest ckpt>`, or start a queued / never-checkpointed run from scratch |
//! | DELETE | `/api/runs/{id}` | remove the directory (not while running) |
//! | GET | `/api/runs/{id}/metrics?keys=a,b&max=4000&after_step=n` | downsampled series |
//! | GET | `/api/runs/{id}/logs?tail=n \| after=seq` | log rows |
//! | GET | `/api/runs/{id}/config` | resolved `config.toml` text |
//! | GET | `/api/runs/{id}/checkpoints` | checkpoint list |
//! | GET | `/api/runs/{id}/checkpoints/{ckpt}/audio/{file}` | WAV passthrough |
//! | GET | `/api/runs/{id}/checkpoints/{ckpt}/spec/{clip}` | `<clip>.spec.json` |
//! | GET | `/api/runs/{id}/events` | SSE `metric\|status\|log\|sys\|checkpoint` |
//! | GET | `/api/events` | SSE `runs\|capture` |
//! | GET | `/api/sys` | host snapshot |
//! | GET | `/api/capture` | every mode's status |
//! | POST | `/api/capture/{mode}/{pause\|resume\|stop}` | write `control.json` |
//! | POST | `/api/capture/{mode}/start` | spawn the harness |
//! | GET | `/api/samples?corpus=&split=&speaker=&mode=&q=&page=&per_page=` | prepared ⋈ captured, one page; `mode` is one capture set or a comma list, all required (`dstar,dstar+perens`) |
//! | GET | `/api/samples/facets` | corpora, splits, speakers, modes with counts |
//! | GET | `/api/samples/random?…` | one row under the same filters |
//! | GET | `/api/samples/{key}` | the joined row plus its manifest rows |
//! | GET | `/api/samples/{key}/audio/{clean8\|clean16\|degraded-<mode>}` | WAV passthrough |
//! | GET | `/api/samples/{key}/spec/{which}` | `spec.json` of that signal, cached under `cache/spec/` |
//! | GET | `/api/samples/{key}/model?run=&step=` | `{ready, audio, spec}` or `{ready:false, …}` |
//! | POST | `/api/samples/{key}/model` `{run, step}` | spawn `infer`; 202, then poll GET |
//! | GET | `/api/samples/{key}/model/{audio\|spec}?run=&step=` | the rendered output |
//!
//! Keys contain `/`, so the sample routes are one wildcard parsed here.

// axum extractors are taken by value by design, and handlers that only do
// small synchronous file reads are still `async fn` because axum needs a
// future.
#![allow(clippy::needless_pass_by_value, clippy::unused_async)]

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use unamblify::{RunConfig, RunState};

use crate::capture::{self, Control};
use crate::downsample::{self, Series};
use crate::error::{Result, WebError};
use crate::events::{Event, Kind};
use crate::samples::{self, Filter, Which};
use crate::state::AppState;
use crate::{runs, sys};

type St = State<Arc<AppState>>;

/// All `/api` routes (mounted by [`crate::router`]).
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health))
        .route("/configs", get(configs_list))
        .route("/configs/validate", post(configs_validate))
        .route("/configs/{name}", get(configs_get))
        .route("/runs", get(runs_list).post(runs_create))
        .route("/runs/{id}", get(runs_get).delete(runs_delete))
        .route("/runs/{id}/stop", post(runs_stop))
        .route("/runs/{id}/resume", post(runs_resume))
        .route("/runs/{id}/metrics", get(runs_metrics))
        .route("/runs/{id}/logs", get(runs_logs))
        .route("/runs/{id}/config", get(runs_config))
        .route("/runs/{id}/checkpoints", get(runs_checkpoints))
        .route(
            "/runs/{id}/checkpoints/{ckpt}/audio/{file}",
            get(ckpt_audio),
        )
        .route("/runs/{id}/checkpoints/{ckpt}/spec/{clip}", get(ckpt_spec))
        .route("/runs/{id}/events", get(run_events))
        .route("/events", get(events))
        .route("/sys", get(sys_get))
        .route("/drives", get(drives_get))
        .route("/host", get(host_get))
        .route("/capture", get(capture_get))
        .route("/capture/{mode}/{verb}", post(capture_post))
        .route("/voicesets", get(voicesets_get))
        .route("/samples", get(samples_list))
        .route("/samples/facets", get(samples_facets))
        .route("/samples/random", get(samples_random))
        .route("/samples/{*rest}", get(samples_get).post(samples_post))
}

// ── health / configs ────────────────────────────────────────────────────

async fn health(State(s): St) -> Response {
    axum::Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "host": sys::hostname(),
        "runs_dir": s.runs_dir,
        "data_root": s.data_root,
        "configs_dir": s.configs_dir,
        "devices": sys::devices(),
        "supervised": s.sup.keys(),
    }))
    .into_response()
}

/// A config file in the configs dir.
#[derive(Debug, Serialize)]
struct ConfigEntry {
    name: String,
    path: PathBuf,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<RunConfig>,
}

fn config_path(s: &AppState, name: &str) -> Result<PathBuf> {
    runs::validate_component("config name", name)?;
    let p = s
        .configs_dir
        .join(format!("{}.toml", name.trim_end_matches(".toml")));
    if p.is_file() {
        Ok(p)
    } else {
        Err(WebError::NotFound(format!("config {name}")))
    }
}

async fn configs_list(State(s): St) -> Result<Response> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&s.configs_dir) {
        for e in rd.filter_map(std::result::Result::ok) {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("toml") {
                continue;
            }
            let name = p
                .file_stem()
                .map(|x| x.to_string_lossy().into_owned())
                .unwrap_or_default();
            let parsed = std::fs::read_to_string(&p)
                .map_err(|e| e.to_string())
                .and_then(|t| RunConfig::from_toml(&t).map_err(|e| e.to_string()));
            out.push(match parsed {
                Ok(c) => ConfigEntry {
                    name,
                    path: p,
                    ok: true,
                    error: None,
                    config: Some(c),
                },
                Err(err) => ConfigEntry {
                    name,
                    path: p,
                    ok: false,
                    error: Some(err),
                    config: None,
                },
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(axum::Json(out).into_response())
}

async fn configs_get(State(s): St, Path(name): Path<String>) -> Result<Response> {
    let p = config_path(&s, &name)?;
    let text = std::fs::read_to_string(&p)?;
    let cfg = RunConfig::from_toml(&text)?;
    Ok(axum::Json(json!({ "name": name, "path": p, "toml": text, "config": cfg })).into_response())
}

#[derive(Debug, Deserialize)]
struct ValidateBody {
    toml: String,
}

async fn configs_validate(axum::Json(b): axum::Json<ValidateBody>) -> Response {
    match RunConfig::from_toml(&b.toml) {
        Ok(cfg) => match runs::validate_name(&cfg.name) {
            Ok(()) => axum::Json(json!({ "ok": true, "config": cfg })).into_response(),
            Err(e) => (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({ "ok": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ── runs ────────────────────────────────────────────────────────────────

async fn runs_list(State(s): St) -> Result<Response> {
    let ids = runs::list_ids(&s.runs_dir)?;
    let rows: Vec<runs::RunSummary> = ids
        .iter()
        .map(|id| runs::summarize(&s.runs_dir, id, s.sup.is_running(id)))
        .collect();
    Ok(axum::Json(rows).into_response())
}

/// `POST /api/runs`.
#[derive(Debug, Default, Deserialize)]
struct CreateBody {
    /// TOML text of the config.
    #[serde(default)]
    toml: Option<String>,
    /// Or: the name of a config in the configs dir.
    #[serde(default)]
    config: Option<String>,
    /// Applied on top before the resolved config is written.
    #[serde(default)]
    overrides: Overrides,
    /// Create the directory but do not start the trainer.
    #[serde(default)]
    queue_only: bool,
}

#[derive(Debug, Default, Deserialize)]
struct Overrides {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    steps: Option<u64>,
    /// `[train] init_from`: another run's weights to start from, so the
    /// dashboard can launch a specialist off a generalist.
    #[serde(default)]
    init_from: Option<String>,
}

async fn runs_create(State(s): St, axum::Json(b): axum::Json<CreateBody>) -> Result<Response> {
    let text = match (&b.toml, &b.config) {
        (Some(t), _) => t.clone(),
        (None, Some(name)) => std::fs::read_to_string(config_path(&s, name)?)?,
        (None, None) => {
            return Err(WebError::BadRequest(
                "body needs `toml` or `config`".to_owned(),
            ));
        }
    };
    let mut cfg = RunConfig::from_toml(&text)?;
    if let Some(n) = &b.overrides.name {
        cfg.name.clone_from(n);
    }
    if let Some(d) = &b.overrides.device {
        cfg.train.device.clone_from(d);
    }
    if let Some(n) = b.overrides.steps {
        cfg.train.steps = n;
    }
    if let Some(i) = &b.overrides.init_from {
        cfg.train.init_from = Some(i.clone());
    }
    let (id, dir) = runs::create_run(&s.runs_dir, &cfg)?;
    if !b.queue_only {
        s.sup.start_train(&id, &dir, None)?;
    }
    let summary = runs::summarize(&s.runs_dir, &id, s.sup.is_running(&id));
    Ok((StatusCode::CREATED, axum::Json(summary)).into_response())
}

async fn runs_get(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    let summary = runs::summarize(&s.runs_dir, &id, s.sup.is_running(&id));
    let config_toml = std::fs::read_to_string(dir.join("config.toml")).ok();
    let (st, id2) = (Arc::clone(&s), id.clone());
    let keys = tokio::task::spawn_blocking(move || st.metric_keys(&id2, &dir))
        .await
        .map_err(|e| WebError::Supervisor(e.to_string()))??;
    Ok(axum::Json(json!({
        "id": id,
        "name": summary.name,
        "status": summary.status,
        "config": summary.config,
        "config_toml": config_toml,
        "supervised": summary.supervised,
        "last_loss": summary.last_loss,
        "checkpoints": runs::list_checkpoints(&s.runs_dir.join(&id)),
        "metric_keys": keys,
    }))
    .into_response())
}

async fn runs_delete(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    if s.sup.is_running(&id) {
        return Err(WebError::Conflict(format!(
            "run {id} is running; stop it first"
        )));
    }
    if let Some(st) = runs::read_status(&dir)
        && st.status == RunState::Running
        && let Some(pid) = st.pid.filter(|&p| crate::supervisor::pid_alive(p))
    {
        return Err(WebError::Conflict(format!(
            "run {id} has a live pid {pid}; stop it first"
        )));
    }
    std::fs::remove_dir_all(&dir)?;
    s.hub
        .publish(Event::global(Kind::Runs, json!({ "deleted": id })));
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn runs_stop(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    if !s.sup.is_running(&id) {
        // Not ours, but maybe alive (another server's child): refuse
        // rather than signal a pid we do not track.
        return Err(WebError::Conflict(format!(
            "run {id} is not supervised here"
        )));
    }
    s.sup.stop(&id).await?;
    let st = runs::read_status(&dir);
    Ok(axum::Json(json!({ "id": id, "status": st })).into_response())
}

/// Resume from the latest checkpoint; a run that has none — queued and
/// never started, or one that died before its first checkpoint — is
/// started from scratch instead.
async fn runs_resume(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    if s.sup.is_running(&id) {
        return Err(WebError::Conflict(format!("run {id} is already running")));
    }
    if let Some(st) = runs::read_status(&dir)
        && st.status == RunState::Running
        && let Some(pid) = st.pid.filter(|&p| crate::supervisor::pid_alive(p))
    {
        return Err(WebError::Conflict(format!(
            "run {id} has a live pid {pid} not started by this server"
        )));
    }
    let ckpt = runs::latest_checkpoint(&dir);
    let pid = s.sup.start_train(&id, &dir, ckpt.as_deref())?;
    Ok(axum::Json(json!({ "id": id, "pid": pid, "resume": ckpt })).into_response())
}

#[derive(Debug, Default, Deserialize)]
struct MetricsQuery {
    #[serde(default)]
    keys: Option<String>,
    #[serde(default)]
    max: Option<usize>,
    #[serde(default)]
    after_step: Option<u64>,
}

async fn runs_metrics(
    State(s): St,
    Path(id): Path<String>,
    Query(q): Query<MetricsQuery>,
) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    let keys: Vec<String> = q
        .keys
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_owned)
        .collect();
    let max = q
        .max
        .unwrap_or(downsample::MAX_POINTS)
        .clamp(4, downsample::MAX_POINTS);
    let after = q.after_step;
    let series: BTreeMap<String, Series> = tokio::task::spawn_blocking(move || {
        let raw = runs::read_metrics(&dir, &keys, after)?;
        Ok::<_, WebError>(
            raw.into_iter()
                .map(|(k, v)| (k, downsample::min_max(&v, max)))
                .collect(),
        )
    })
    .await
    .map_err(|e| WebError::Supervisor(e.to_string()))??;
    Ok(axum::Json(json!({ "id": id, "max": max, "series": series })).into_response())
}

#[derive(Debug, Default, Deserialize)]
struct LogsQuery {
    #[serde(default)]
    tail: Option<usize>,
    #[serde(default)]
    after: Option<u64>,
}

async fn runs_logs(
    State(s): St,
    Path(id): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    let (tail, after) = match (q.tail, q.after) {
        (None, None) => (Some(500), None),
        (t, a) => (t, a),
    };
    let rows = tokio::task::spawn_blocking(move || runs::read_logs(&dir, tail, after))
        .await
        .map_err(|e| WebError::Supervisor(e.to_string()))??;
    Ok(axum::Json(rows).into_response())
}

async fn runs_config(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    let text = std::fs::read_to_string(dir.join("config.toml"))?;
    Ok((
        [(header::CONTENT_TYPE, "application/toml; charset=utf-8")],
        text,
    )
        .into_response())
}

async fn runs_checkpoints(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    Ok(axum::Json(runs::list_checkpoints(&dir)).into_response())
}

fn ckpt_dir(s: &AppState, id: &str, ckpt: &str) -> Result<PathBuf> {
    let dir = runs::run_dir(&s.runs_dir, id)?;
    runs::validate_component("checkpoint", ckpt)?;
    let name = if ckpt.starts_with("step-") {
        ckpt.to_owned()
    } else {
        // Accept a bare step number too.
        let n: u64 = ckpt
            .parse()
            .map_err(|_| WebError::BadRequest(format!("bad checkpoint {ckpt:?}")))?;
        format!("step-{n:06}")
    };
    let p = dir.join("checkpoints").join(name);
    if p.is_dir() {
        Ok(p)
    } else {
        Err(WebError::NotFound(format!("checkpoint {ckpt} of run {id}")))
    }
}

async fn ckpt_audio(
    State(s): St,
    Path((id, ckpt, file)): Path<(String, String, String)>,
) -> Result<Response> {
    let dir = ckpt_dir(&s, &id, &ckpt)?;
    runs::validate_component("audio file", &file)?;
    if !file.to_ascii_lowercase().ends_with(".wav") {
        return Err(WebError::BadRequest("audio files are .wav".to_owned()));
    }
    stream_file(&dir.join("audio").join(&file), "audio/wav").await
}

async fn ckpt_spec(
    State(s): St,
    Path((id, ckpt, clip)): Path<(String, String, String)>,
) -> Result<Response> {
    let dir = ckpt_dir(&s, &id, &ckpt)?;
    let clip = clip.trim_end_matches(".spec.json");
    runs::validate_component("clip", clip)?;
    stream_file(
        &dir.join("audio").join(format!("{clip}.spec.json")),
        "application/json",
    )
    .await
}

async fn stream_file(path: &FsPath, mime: &str) -> Result<Response> {
    let f = tokio::fs::File::open(path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            WebError::NotFound(path.display().to_string())
        } else {
            WebError::Io(e)
        }
    })?;
    let len = f.metadata().await?.len();
    let body = Body::from_stream(tokio_util_compat::ReaderStream::new(f));
    Ok((
        [
            (header::CONTENT_TYPE, mime.to_owned()),
            (header::CONTENT_LENGTH, len.to_string()),
            (header::CACHE_CONTROL, "private, max-age=3600".to_owned()),
        ],
        body,
    )
        .into_response())
}

/// A tiny `ReaderStream` so the crate need not pull `tokio-util`.
mod tokio_util_compat {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use axum::body::Bytes;
    use futures_core::Stream;
    use tokio::io::{AsyncRead, ReadBuf};

    pub struct ReaderStream<R> {
        r: R,
        buf: Vec<u8>,
        done: bool,
    }

    impl<R> ReaderStream<R> {
        pub fn new(r: R) -> Self {
            Self {
                r,
                buf: vec![0; 64 * 1024],
                done: false,
            }
        }
    }

    impl<R: AsyncRead + Unpin> Stream for ReaderStream<R> {
        type Item = Result<Bytes, std::io::Error>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            if this.done {
                return Poll::Ready(None);
            }
            let mut rb = ReadBuf::new(&mut this.buf);
            match Pin::new(&mut this.r).poll_read(cx, &mut rb) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(e)) => {
                    this.done = true;
                    Poll::Ready(Some(Err(e)))
                }
                Poll::Ready(Ok(())) => {
                    let n = rb.filled().len();
                    if n == 0 {
                        this.done = true;
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Ok(Bytes::copy_from_slice(&this.buf[..n]))))
                    }
                }
            }
        }
    }
}

// ── SSE ─────────────────────────────────────────────────────────────────

fn sse_event(ev: &Event) -> SseEvent {
    let data = serde_json::to_string(&ev.data).unwrap_or_else(|_| "null".to_owned());
    SseEvent::default().event(ev.kind.as_str()).data(data)
}

fn sse_stream(
    rx: tokio::sync::broadcast::Receiver<Event>,
    first: Vec<SseEvent>,
    filter: impl Fn(&Event) -> bool + Send + 'static,
) -> Sse<impl Stream<Item = std::result::Result<SseEvent, Infallible>>> {
    let head = tokio_stream::iter(first.into_iter().map(Ok::<_, Infallible>));
    let live = BroadcastStream::new(rx).filter_map(move |r| match r {
        Ok(ev) if filter(&ev) => Some(Ok(sse_event(&ev))),
        Ok(_) => None,
        Err(_lagged) => Some(Ok(SseEvent::default()
            .event("lagged")
            .data("{\"note\":\"events dropped; refetch\"}"))),
    });
    Sse::new(head.chain(live)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

async fn run_events(State(s): St, Path(id): Path<String>) -> Result<Response> {
    let dir = runs::run_dir(&s.runs_dir, &id)?;
    let rx = s.hub.subscribe();
    let mut first = Vec::new();
    if let Some(st) = runs::read_status(&dir) {
        first.push(sse_event(&Event::for_run(
            &id,
            Kind::Status,
            serde_json::to_value(st)?,
        )));
    }
    let want = id.clone();
    Ok(sse_stream(rx, first, move |e| e.run.as_deref() == Some(want.as_str())).into_response())
}

async fn events(State(s): St) -> Result<Response> {
    let rx = s.hub.subscribe();
    let ids = runs::list_ids(&s.runs_dir)?;
    let first = vec![
        sse_event(&Event::global(Kind::Runs, json!({ "ids": ids }))),
        sse_event(&Event::global(
            Kind::Capture,
            json!({ "event": "snapshot", "modes": capture::view_all(&s.captured_dir(), &s.sup, 0) }),
        )),
    ];
    Ok(sse_stream(rx, first, |e| e.run.is_none()).into_response())
}

// ── sys / capture ───────────────────────────────────────────────────────

/// `GET /api/drives` — every physical drive with its current
/// temperature, plus the rolling history the Host page charts. A drive
/// whose enclosure does not pass SMART through is listed with a note and
/// no reading.
async fn drives_get(State(s): St) -> Response {
    let (now, history) = s
        .drives
        .lock()
        .map(|h| (h.last(), h.samples()))
        .unwrap_or_default();
    axum::Json(json!({ "drives": now, "history": history })).into_response()
}

/// `GET /api/host` — 24 h of whole-host CPU use, load average and GPU use.
async fn host_get(State(s): St) -> Response {
    let history = s.host.lock().map(|h| h.samples()).unwrap_or_default();
    axum::Json(json!({ "history": history })).into_response()
}

async fn sys_get(State(s): St) -> Response {
    let snap = {
        let mut sys = s
            .sys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sys::snapshot(
            &mut sys,
            s.sup.exe(),
            &s.runs_dir,
            &s.data_root,
            s.started_s,
        )
    };
    axum::Json(snap).into_response()
}

#[derive(Debug, Default, Deserialize)]
struct CaptureQuery {
    #[serde(default)]
    log_tail: Option<usize>,
}

async fn capture_get(State(s): St, Query(q): Query<CaptureQuery>) -> Response {
    let modes = capture::view_all(
        &s.captured_dir(),
        &s.sup,
        q.log_tail.unwrap_or(50).min(1000),
    );
    axum::Json(json!({ "captured_dir": s.captured_dir(), "modes": modes })).into_response()
}

/// `GET /api/voicesets` — for each capture set, the corpora it holds with
/// utterance and hour counts. Aggregated off the async threads; the
/// per-manifest cache keeps a repeat call cheap.
async fn voicesets_get(State(s): St) -> Response {
    let dir = s.captured_dir();
    let sets = tokio::task::spawn_blocking(move || capture::voicesets(&dir))
        .await
        .unwrap_or_default();
    axum::Json(json!({ "sets": sets })).into_response()
}

/// `POST /api/capture/{mode}/start` body (optional).
#[derive(Debug, Default, Deserialize)]
struct CaptureStartBody {
    /// Extra `capture` arguments (`--corpus`, `--limit`, `--port`, `--jobs`…).
    #[serde(default)]
    args: Vec<String>,
}

async fn capture_post(
    State(s): St,
    Path((mode, verb)): Path<(String, String)>,
    body: Option<axum::Json<CaptureStartBody>>,
) -> Result<Response> {
    let mode = capture::parse_mode(&mode)?;
    let captured = s.captured_dir();
    if verb == "start" {
        let args = body.map(|b| b.0.args).unwrap_or_default();
        for a in &args {
            if a.contains('\0') {
                return Err(WebError::BadRequest("bad argument".to_owned()));
            }
        }
        let pid = s.sup.start_capture(mode, &s.data_root, &args)?;
        return Ok((
            StatusCode::CREATED,
            axum::Json(json!({ "mode": mode, "pid": pid })),
        )
            .into_response());
    }
    let c = Control::from_verb(&verb)?;
    capture::write_control(&captured, mode, c)?;
    s.hub.publish(Event::global(
        Kind::Capture,
        json!({ "mode": mode, "event": "control", "control": c.as_str() }),
    ));
    Ok(axum::Json(json!({ "mode": mode, "control": c.as_str() })).into_response())
}

// ── samples ─────────────────────────────────────────────────────────────

async fn samples_index(s: &Arc<AppState>) -> Result<Arc<samples::Index>> {
    let st = Arc::clone(s);
    tokio::task::spawn_blocking(move || st.samples_index())
        .await
        .map_err(|e| WebError::Supervisor(e.to_string()))?
}

async fn samples_list(State(s): St, Query(f): Query<Filter>) -> Result<Response> {
    let ix = samples_index(&s).await?;
    Ok(axum::Json(samples::list(&ix, &f)?).into_response())
}

async fn samples_facets(State(s): St) -> Result<Response> {
    let ix = samples_index(&s).await?;
    Ok(axum::Json(&ix.facets).into_response())
}

async fn samples_random(State(s): St, Query(f): Query<Filter>) -> Result<Response> {
    let ix = samples_index(&s).await?;
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    match samples::random(&ix, &f, seed)? {
        Some(item) => Ok(axum::Json(item).into_response()),
        None => Err(WebError::NotFound(
            "no sample matches the filters".to_owned(),
        )),
    }
}

/// What follows the key in `/api/samples/{*rest}`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SampleSub {
    Get,
    Audio(String),
    Spec(String),
    Model,
    ModelAudio,
    ModelSpec,
}

/// Split the wildcard into a validated key and the sub-resource. A key
/// with `..`, an empty component or a character outside the key
/// alphabet is refused before anything touches the disk.
fn parse_rest(rest: &str) -> Result<(String, SampleSub)> {
    let rest = rest.trim_matches('/');
    let (key, sub) = if let Some((k, w)) = rest.rsplit_once("/audio/") {
        (k, SampleSub::Audio(w.to_owned()))
    } else if let Some((k, w)) = rest.rsplit_once("/spec/") {
        (k, SampleSub::Spec(w.to_owned()))
    } else if let Some(k) = rest.strip_suffix("/model/audio") {
        (k, SampleSub::ModelAudio)
    } else if let Some(k) = rest.strip_suffix("/model/spec") {
        (k, SampleSub::ModelSpec)
    } else if let Some(k) = rest.strip_suffix("/model") {
        (k, SampleSub::Model)
    } else {
        (rest, SampleSub::Get)
    };
    unamblify::key::validate(key).map_err(|e| WebError::BadRequest(e.to_string()))?;
    if let SampleSub::Audio(w) | SampleSub::Spec(w) = &sub {
        runs::validate_component("signal", w)?;
    }
    Ok((key.to_owned(), sub))
}

/// `?run=&step=` of the model routes.
#[derive(Debug, Default, Deserialize)]
struct ModelQuery {
    #[serde(default)]
    run: Option<String>,
    #[serde(default)]
    step: Option<u64>,
}

/// `POST …/model` body.
#[derive(Debug, Deserialize)]
struct ModelBody {
    run: String,
    step: u64,
}

fn model_target(s: &AppState, run: &str, step: u64, key: &str) -> Result<(PathBuf, PathBuf)> {
    let dir = runs::run_dir(&s.runs_dir, run)?;
    if !unamblify::run::checkpoint_dir(&dir, step)
        .join("model.safetensors")
        .is_file()
    {
        return Err(WebError::NotFound(format!(
            "checkpoint step {step} of run {run}"
        )));
    }
    Ok((
        dir.clone(),
        unamblify::run::sample_out_path(&dir, step, key),
    ))
}

fn model_view(s: &AppState, run: &str, step: u64, key: &str, out: &FsPath) -> serde_json::Value {
    let url_key: String = key
        .split('/')
        .map(urlencoding_component)
        .collect::<Vec<_>>()
        .join("/");
    let q = format!("?run={}&step={step}", urlencoding_component(run));
    if out.is_file() {
        return json!({
            "ready": true,
            "run": run,
            "step": step,
            "audio": format!("/api/samples/{url_key}/model/audio{q}"),
            "spec": format!("/api/samples/{url_key}/model/spec{q}"),
        });
    }
    match s.sup.infer_state(run, step, key) {
        Some(crate::supervisor::InferState::Running) => {
            json!({ "ready": false, "running": true, "run": run, "step": step })
        }
        Some(crate::supervisor::InferState::Failed { error }) => {
            json!({ "ready": false, "running": false, "error": error, "run": run, "step": step })
        }
        None => json!({ "ready": false, "running": false, "run": run, "step": step }),
    }
}

/// Percent-encode one path component (the key alphabet needs nothing,
/// but a run id or an unexpected byte must not break the URL).
fn urlencoding_component(c: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(c.len());
    for b in c.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

async fn samples_get(
    State(s): St,
    Path(rest): Path<String>,
    Query(mq): Query<ModelQuery>,
) -> Result<Response> {
    let (key, sub) = parse_rest(&rest)?;
    let ix = samples_index(&s).await?;
    let item = ix
        .get(&key)
        .ok_or_else(|| WebError::NotFound(format!("sample {key}")))?
        .clone();
    match sub {
        SampleSub::Get => {
            let (root, ix2, item2) = (s.data_root.clone(), Arc::clone(&ix), item.clone());
            let (prepared, captured) = tokio::task::spawn_blocking(move || ix2.rows(&root, &item2))
                .await
                .map_err(|e| WebError::Supervisor(e.to_string()))??;
            let mut v = serde_json::to_value(&item)?;
            if let Some(o) = v.as_object_mut() {
                o.insert("prepared".to_owned(), serde_json::to_value(prepared)?);
                o.insert("captured".to_owned(), serde_json::to_value(captured)?);
            }
            Ok(axum::Json(v).into_response())
        }
        SampleSub::Audio(w) => {
            let which = Which::parse(&w)?;
            if let Which::Degraded(m) = &which
                && !item.modes.iter().any(|x| &x.mode == m)
            {
                return Err(WebError::NotFound(format!("{key} is not captured for {m}")));
            }
            // Resolve the clip on disk (FLAC once the corpus is FLAC, WAV
            // otherwise) and serve it with the matching MIME; browsers play
            // both in <audio> and decode both for the waveform.
            let path = which.resolve(&s.data_root, &key);
            let mime = samples::audio_mime(&path);
            stream_file(&path, mime).await
        }
        SampleSub::Spec(w) => {
            let which = Which::parse(&w)?;
            if let Which::Degraded(m) = &which
                && !item.modes.iter().any(|x| &x.mode == m)
            {
                return Err(WebError::NotFound(format!("{key} is not captured for {m}")));
            }
            let root = s.data_root.clone();
            let (bytes, hit) =
                tokio::task::spawn_blocking(move || samples::spec_json(&root, &key, &which))
                    .await
                    .map_err(|e| WebError::Supervisor(e.to_string()))??;
            Ok((
                [
                    (header::CONTENT_TYPE, "application/json".to_owned()),
                    (header::CACHE_CONTROL, "private, max-age=3600".to_owned()),
                    (
                        header::HeaderName::from_static("x-cache"),
                        if hit { "hit" } else { "miss" }.to_owned(),
                    ),
                ],
                bytes,
            )
                .into_response())
        }
        SampleSub::Model | SampleSub::ModelAudio | SampleSub::ModelSpec => {
            let (Some(run), Some(step)) = (mq.run.as_deref(), mq.step) else {
                return Err(WebError::BadRequest(
                    "model routes need ?run=&step=".to_owned(),
                ));
            };
            let (_, out) = model_target(&s, run, step, &key)?;
            match sub {
                SampleSub::Model => {
                    Ok(axum::Json(model_view(&s, run, step, &key, &out)).into_response())
                }
                SampleSub::ModelAudio => stream_file(&out, "audio/wav").await,
                _ => stream_file(&unamblify::run::spec_path_for(&out), "application/json").await,
            }
        }
    }
}

async fn samples_post(
    State(s): St,
    Path(rest): Path<String>,
    axum::Json(b): axum::Json<ModelBody>,
) -> Result<Response> {
    let (key, sub) = parse_rest(&rest)?;
    if sub != SampleSub::Model {
        return Err(WebError::BadRequest("POST only on …/model".to_owned()));
    }
    let ix = samples_index(&s).await?;
    if ix.get(&key).is_none() {
        return Err(WebError::NotFound(format!("sample {key}")));
    }
    let (dir, out) = model_target(&s, &b.run, b.step, &key)?;
    if out.is_file() {
        return Ok(axum::Json(model_view(&s, &b.run, b.step, &key, &out)).into_response());
    }
    s.sup
        .start_infer(&b.run, &dir, b.step, &key, &out, &s.data_root);
    Ok((
        StatusCode::ACCEPTED,
        axum::Json(model_view(&s, &b.run, b.step, &key, &out)),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_paths_parse_and_reject_traversal() {
        assert_eq!(
            parse_rest("vctk/p225_001_mic2").unwrap(),
            ("vctk/p225_001_mic2".to_owned(), SampleSub::Get)
        );
        assert_eq!(
            parse_rest("libritts_r/dev-clean/84_1_2_3/audio/degraded-dstar").unwrap(),
            (
                "libritts_r/dev-clean/84_1_2_3".to_owned(),
                SampleSub::Audio("degraded-dstar".to_owned())
            )
        );
        assert_eq!(
            parse_rest("vctk/p225_001_mic2/spec/clean16").unwrap().1,
            SampleSub::Spec("clean16".to_owned())
        );
        assert_eq!(parse_rest("vctk/x/model").unwrap().1, SampleSub::Model);
        assert_eq!(
            parse_rest("vctk/x/model/audio").unwrap().1,
            SampleSub::ModelAudio
        );
        assert_eq!(
            parse_rest("vctk/x/model/spec").unwrap().1,
            SampleSub::ModelSpec
        );
        for bad in [
            "../etc/passwd",
            "vctk/../../etc/passwd/audio/clean16",
            "vctk/./x",
            "vctk//x",
            "noslash",
            "vctk/x/audio/../y",
            "vctk/p225 001",
        ] {
            assert!(parse_rest(bad).is_err(), "{bad}");
        }
        assert_eq!(urlencoding_component("a b/c"), "a%20b%2Fc");
    }
}
