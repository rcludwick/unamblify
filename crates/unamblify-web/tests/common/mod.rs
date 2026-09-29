// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Shared fixture: a temp data root with hand-written runs, a configs dir,
//! the fake trainer as the supervisor's executable, and request helpers
//! over the router (no sockets).

#![allow(dead_code, clippy::cast_precision_loss, clippy::format_push_string)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;
use unamblify_web::{AppState, ServeOpts};

pub struct Fixture {
    pub tmp: tempfile::TempDir,
    pub state: Arc<AppState>,
    pub app: Router,
    pub token: Option<String>,
}

pub fn fake_trainer() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fake_trainer.sh")
}

pub fn opts(tmp: &Path) -> ServeOpts {
    ServeOpts {
        runs_dir: Some(tmp.join("runs")),
        data_root: Some(tmp.to_path_buf()),
        configs_dir: Some(tmp.join("configs")),
        bind: None,
        token: None,
        exe: Some(fake_trainer()),
        grace: Duration::from_secs(5),
        poll_interval: Duration::from_millis(50),
    }
}

impl Fixture {
    pub fn new() -> Self {
        Self::with(|_| {})
    }

    pub fn with(f: impl FnOnce(&mut ServeOpts)) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("configs")).unwrap();
        let mut o = opts(tmp.path());
        f(&mut o);
        let state = unamblify_web::build(&o).unwrap();
        let app = unamblify_web::router(Arc::clone(&state), o.token.as_deref());
        Self {
            tmp,
            state,
            app,
            token: o.token,
        }
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.tmp.path().join("runs")
    }

    fn build(&self, method: &str, path: &str, body: Option<Value>) -> Request<Body> {
        let mut b = Request::builder().method(method).uri(path);
        if let Some(t) = &self.token {
            b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        match body {
            Some(v) => b
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        }
    }

    pub async fn raw(&self, method: &str, path: &str, body: Option<Value>) -> Response<Body> {
        self.app
            .clone()
            .oneshot(self.build(method, path, body))
            .await
            .unwrap()
    }

    /// Request and parse JSON. Panics on a non-JSON body.
    pub async fn json(&self, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let res = self.raw(method, path, body).await;
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!(
                "{method} {path}: {status} non-json body {e}: {}",
                String::from_utf8_lossy(&bytes)
            )
        });
        (status, v)
    }

    pub async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.json("GET", path, None).await
    }

    pub async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.json("POST", path, Some(body)).await
    }

    /// Read SSE frames from `path` until `pred` matches the accumulated
    /// text or `timeout` elapses. Returns the text.
    pub async fn sse(&self, path: &str, timeout: Duration, pred: impl Fn(&str) -> bool) -> String {
        let res = self.raw("GET", path, None).await;
        assert_eq!(res.status(), StatusCode::OK, "{path}");
        assert!(
            res.headers()
                .get(header::CONTENT_TYPE)
                .is_some_and(|v| v.to_str().unwrap().starts_with("text/event-stream")),
            "{path} is not an event stream"
        );
        let mut body = res.into_body();
        let mut text = String::new();
        let deadline = tokio::time::Instant::now() + timeout;
        while !pred(&text) {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                break;
            }
            match tokio::time::timeout(left, body.frame()).await {
                Ok(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        text.push_str(&String::from_utf8_lossy(data));
                    }
                }
                _ => break,
            }
        }
        text
    }
}

/// Write a whole hand-made run directory.
pub fn write_run(
    runs_dir: &Path,
    id: &str,
    status: &str,
    step: u64,
    total: u64,
    n_metrics: u64,
) -> PathBuf {
    let dir = runs_dir.join(id);
    std::fs::create_dir_all(dir.join("checkpoints")).unwrap();
    let name = id.splitn(3, '-').nth(2).unwrap_or(id);
    std::fs::write(
        dir.join("config.toml"),
        format!("name = \"{name}\"\n[model]\nprofile = \"lite\"\n[train]\nsteps = {total}\ndevice = \"cpu\"\n"),
    )
    .unwrap();
    std::fs::write(
        dir.join("status.json"),
        format!(
            "{{\"status\":\"{status}\",\"step\":{step},\"total_steps\":{total},\"started\":\"2026-09-10T05:00:00Z\",\"updated\":\"2026-09-10T05:10:00Z\",\"device\":\"cpu\",\"host\":\"mac\",\"best\":{{\"metric\":\"eval/lsd\",\"value\":1.25,\"step\":500}}}}"
        ),
    )
    .unwrap();
    let mut m = String::new();
    for i in 1..=n_metrics {
        let t = 1_757_480_000_000 + i * 100;
        m.push_str(&format!(
            "{{\"step\":{i},\"t\":{t},\"k\":\"loss/total\",\"v\":{}}}\n",
            2.0 / (i as f64).sqrt()
        ));
        m.push_str(&format!(
            "{{\"step\":{i},\"t\":{t},\"k\":\"loss/stft\",\"v\":{}}}\n",
            1.0 / (i as f64).sqrt()
        ));
        if i % 10 == 0 {
            m.push_str(&format!(
                "{{\"step\":{i},\"t\":{t},\"k\":\"sys/cpu\",\"v\":{}}}\n",
                i % 100
            ));
        }
        if i % 500 == 0 {
            m.push_str(&format!(
                "{{\"step\":{i},\"t\":{t},\"k\":\"eval/lsd\",\"v\":{}}}\n",
                1.5 - (i as f64) / 10_000.0
            ));
        }
    }
    std::fs::write(dir.join("metrics.jsonl"), m).unwrap();
    let mut l = String::new();
    for i in 1..=20u64 {
        l.push_str(&format!(
            "{{\"seq\":{i},\"t\":{},\"level\":\"info\",\"msg\":\"line {i}\"}}\n",
            1_757_480_000_000 + i
        ));
    }
    std::fs::write(dir.join("log.jsonl"), l).unwrap();
    dir
}

/// Add a checkpoint with one rendered clip to a run.
pub fn write_checkpoint(dir: &Path, step: u64, clip: &str) -> PathBuf {
    let ck = dir.join(format!("checkpoints/step-{step:06}"));
    let audio = ck.join("audio");
    std::fs::create_dir_all(&audio).unwrap();
    std::fs::write(ck.join("model.safetensors"), b"model").unwrap();
    std::fs::write(ck.join("optim.safetensors"), b"optim").unwrap();
    std::fs::write(
        ck.join("meta.json"),
        format!("{{\"step\":{step},\"lsd\":1.2}}"),
    )
    .unwrap();
    for v in ["clean", "degraded", "out"] {
        std::fs::write(audio.join(format!("{clip}.{v}.wav")), fake_wav(v)).unwrap();
    }
    std::fs::write(
        audio.join(format!("{clip}.spec.json")),
        r#"{"n_mels":4,"frames":3,"hop_s":0.01,"db_min":-80,"db_max":0,"clean":[[0,-10,-20],[-1,-11,-21],[-2,-12,-22],[-3,-13,-23]],"degraded":[[0,-10,-20],[-1,-11,-21],[-2,-12,-22],[-3,-13,-23]],"out":[[0,-10,-20],[-1,-11,-21],[-2,-12,-22],[-3,-13,-23]]}"#,
    )
    .unwrap();
    ck
}

/// A minimal (invalid-length but recognisable) RIFF header plus a marker.
pub fn fake_wav(marker: &str) -> Vec<u8> {
    let mut v = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
    v.extend_from_slice(marker.as_bytes());
    v
}

pub async fn wait_for(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    f()
}
