// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! API contract tests over hand-written run directories (spec §8, §3).

// Test code: fixture builders cast freely and long scenario functions read
// best as one flow.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::format_push_string
)]

mod common;

use std::time::Duration;

use axum::http::{StatusCode, header};
use http_body_util::BodyExt;
use serde_json::json;

use common::{Fixture, fake_wav, write_checkpoint, write_run};

#[tokio::test(flavor = "multi_thread")]
async fn health_sys_and_ui() {
    let f = Fixture::new();
    let (st, v) = f.get("/api/health").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ok"], true);
    assert_eq!(v["devices"][0]["name"], "cpu");
    assert!(v["runs_dir"].as_str().unwrap().ends_with("runs"));

    let (st, v) = f.get("/api/sys").await;
    assert_eq!(st, StatusCode::OK);
    assert!(v["cpus"].as_u64().unwrap() >= 1);
    assert!(v["mem_total"].as_u64().unwrap() > 0);
    assert!(v["exe"].as_str().unwrap().ends_with("fake_trainer.sh"));
    assert_eq!(v["load"].as_array().unwrap().len(), 3);

    let res = f.raw("GET", "/", None).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(
        res.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    let res = f.raw("GET", "/ui/app.js", None).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(
        res.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("javascript")
    );
    let res = f.raw("GET", "/ui/uplot.iife.min.js", None).await;
    assert_eq!(res.status(), StatusCode::OK);
    let res = f.raw("GET", "/ui/nope.js", None).await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread")]
async fn runs_list_get_metrics_logs_and_checkpoints() {
    let f = Fixture::new();
    let a = write_run(
        &f.runs_dir(),
        "20260910-050000-alpha",
        "finished",
        1000,
        1000,
        1000,
    );
    write_run(
        &f.runs_dir(),
        "20260910-060000-beta",
        "stopped",
        40,
        100,
        40,
    );
    write_checkpoint(&a, 500, "p225_001");
    write_checkpoint(&a, 1000, "p225_001");

    let (st, v) = f.get("/api/runs").await;
    assert_eq!(st, StatusCode::OK);
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], "20260910-060000-beta", "newest first");
    assert_eq!(rows[0]["name"], "beta");
    assert_eq!(rows[0]["status"]["status"], "stopped");
    assert_eq!(rows[0]["supervised"], false);
    assert_eq!(rows[1]["checkpoints"], 2);
    assert_eq!(rows[1]["config"]["model"]["profile"], "lite");
    assert!((rows[1]["last_loss"].as_f64().unwrap() - 2.0 / 1000f64.sqrt()).abs() < 1e-9);

    let (st, v) = f.get("/api/runs/20260910-050000-alpha").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["status"]["best"]["metric"], "eval/lsd");
    assert!(
        v["config_toml"]
            .as_str()
            .unwrap()
            .contains("profile = \"lite\"")
    );
    let keys: Vec<&str> = v["metric_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    assert_eq!(keys, vec!["eval/lsd", "loss/stft", "loss/total", "sys/cpu"]);
    let cks = v["checkpoints"].as_array().unwrap();
    assert_eq!(cks.len(), 2);
    assert_eq!(cks[0]["step"], 500);
    assert_eq!(cks[0]["dir"], "step-000500");
    assert_eq!(cks[0]["meta"]["lsd"], 1.2);
    assert_eq!(cks[0]["clips"][0]["name"], "p225_001");
    assert_eq!(
        cks[0]["clips"][0]["variants"],
        json!(["clean", "degraded", "out"])
    );
    assert_eq!(cks[0]["clips"][0]["spec"], true);

    let (st, _) = f.get("/api/runs/nope").await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, v) = f.get("/api/runs/bad%20id").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("invalid run id"));
    let (st, _) = f.get("/api/runs/..").await;
    assert_ne!(st, StatusCode::OK);

    // Metrics: full, downsampled, filtered, after_step.
    let (st, v) = f.get("/api/runs/20260910-050000-alpha/metrics").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(
        v["series"]["loss/total"]["step"].as_array().unwrap().len(),
        1000
    );
    assert_eq!(v["series"]["eval/lsd"]["v"].as_array().unwrap().len(), 2);
    let (_, v) = f
        .get("/api/runs/20260910-050000-alpha/metrics?keys=loss/total,lr&max=100")
        .await;
    let s = &v["series"];
    assert!(s.get("loss/stft").is_none());
    let n = s["loss/total"]["step"].as_array().unwrap().len();
    assert!((90..=100).contains(&n), "{n}");
    assert_eq!(s["loss/total"]["step"][0], 1);
    assert_eq!(s["loss/total"]["step"][n - 1], 1000);
    let (_, v) = f
        .get("/api/runs/20260910-050000-alpha/metrics?after_step=990")
        .await;
    assert_eq!(
        v["series"]["loss/total"]["step"].as_array().unwrap().len(),
        10
    );
    assert_eq!(v["series"]["loss/total"]["step"][0], 991);
    // max is clamped to the spec's 4 K.
    let (_, v) = f
        .get("/api/runs/20260910-050000-alpha/metrics?max=999999")
        .await;
    assert_eq!(v["max"], 4000);

    // Logs: default tail, explicit tail, after.
    let (st, v) = f.get("/api/runs/20260910-050000-alpha/logs").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v.as_array().unwrap().len(), 20);
    let (_, v) = f.get("/api/runs/20260910-050000-alpha/logs?tail=3").await;
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["seq"], 18);
    assert_eq!(rows[2]["msg"], "line 20");
    let (_, v) = f.get("/api/runs/20260910-050000-alpha/logs?after=17").await;
    assert_eq!(v.as_array().unwrap().len(), 3);

    // Config text and checkpoint list.
    let res = f
        .raw("GET", "/api/runs/20260910-050000-alpha/config", None)
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(
        res.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("toml")
    );
    let (st, v) = f.get("/api/runs/20260910-050000-alpha/checkpoints").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v.as_array().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn checkpoint_audio_and_spec_passthrough() {
    let f = Fixture::new();
    let a = write_run(
        &f.runs_dir(),
        "20260910-050000-alpha",
        "finished",
        10,
        10,
        10,
    );
    write_checkpoint(&a, 500, "p225_001");
    let base = "/api/runs/20260910-050000-alpha/checkpoints/step-000500";

    let res = f
        .raw("GET", &format!("{base}/audio/p225_001.out.wav"), None)
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[header::CONTENT_TYPE], "audio/wav");
    let len: usize = res.headers()[header::CONTENT_LENGTH]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(bytes.as_ref(), fake_wav("out").as_slice());
    assert_eq!(bytes.len(), len);

    // Bare step number works too.
    let res = f
        .raw(
            "GET",
            "/api/runs/20260910-050000-alpha/checkpoints/500/audio/p225_001.clean.wav",
            None,
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);

    let (st, v) = f.get(&format!("{base}/spec/p225_001")).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["n_mels"], 4);
    assert_eq!(v["out"].as_array().unwrap().len(), 4);
    let (st, _) = f.get(&format!("{base}/spec/p225_001.spec.json")).await;
    assert_eq!(st, StatusCode::OK);

    let (st, _) = f.get(&format!("{base}/spec/missing")).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f.get(&format!("{base}/audio/meta.json")).await;
    assert_eq!(
        st,
        StatusCode::BAD_REQUEST,
        "only wav is served from audio/"
    );
    let (st, _) = f
        .get("/api/runs/20260910-050000-alpha/checkpoints/step-000999/audio/x.wav")
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f.get(&format!("{base}/audio/..%2Fmodel.safetensors")).await;
    assert_ne!(st, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn configs_list_get_validate() {
    let f = Fixture::new();
    let cd = f.tmp.path().join("configs");
    std::fs::write(
        cd.join("smoke.toml"),
        "name = \"smoke\"\n[train]\nsteps = 20\n",
    )
    .unwrap();
    std::fs::write(cd.join("broken.toml"), "name = \n").unwrap();
    std::fs::write(cd.join("README.md"), "not a config").unwrap();

    let (st, v) = f.get("/api/configs").await;
    assert_eq!(st, StatusCode::OK);
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "broken");
    assert_eq!(rows[0]["ok"], false);
    assert!(rows[0]["error"].is_string());
    assert_eq!(rows[1]["name"], "smoke");
    assert_eq!(rows[1]["config"]["train"]["steps"], 20);

    let (st, v) = f.get("/api/configs/smoke").await;
    assert_eq!(st, StatusCode::OK);
    assert!(v["toml"].as_str().unwrap().starts_with("name = \"smoke\""));
    assert_eq!(v["config"]["model"]["profile"], "full");
    let (st, _) = f.get("/api/configs/missing").await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f.get("/api/configs/..%2Fsmoke").await;
    assert_ne!(st, StatusCode::OK);

    let (st, v) = f
        .post(
            "/api/configs/validate",
            json!({ "toml": "name = \"ok\"\n[model]\nprofile = \"lite\"\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ok"], true);
    assert_eq!(v["config"]["model"]["profile"], "lite");
    let (st, v) = f
        .post(
            "/api/configs/validate",
            json!({ "toml": "[model]\nprofile = \"lite\"\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert_eq!(v["ok"], false);
    assert!(v["error"].as_str().unwrap().contains("name"));
    let (st, v) = f
        .post(
            "/api/configs/validate",
            json!({ "toml": "name = \"has.dot\"\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("no dots"));
}

#[tokio::test(flavor = "multi_thread")]
async fn create_queue_only_and_delete() {
    let f = Fixture::new();
    let body = json!({
        "toml": "name = \"seed\"\n[train]\nsteps = 100\n",
        "overrides": { "device": "mps", "steps": 7, "name": "seed-mps" },
        "queue_only": true
    });
    let (st, v) = f.post("/api/runs", body).await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.ends_with("-seed-mps"), "{id}");
    assert_eq!(v["status"]["status"], "queued");
    assert_eq!(v["status"]["total_steps"], 7);
    assert_eq!(v["config"]["train"]["device"], "mps");
    let dir = f.runs_dir().join(&id);
    let toml = std::fs::read_to_string(dir.join("config.toml")).unwrap();
    assert!(toml.contains("device = \"mps\"") && toml.contains("steps = 7"));

    // Same second, same name: a distinct id.
    let (st, v2) = f
        .post(
            "/api/runs",
            json!({ "toml": "name = \"seed-mps\"\n", "queue_only": true }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED);
    assert_ne!(v2["id"], v["id"]);

    let (st, v) = f.post("/api/runs", json!({ "toml": "nope = \n" })).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("config"));
    let (st, _) = f.post("/api/runs", json!({})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f.post("/api/runs", json!({ "config": "missing" })).await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    let res = f.raw("DELETE", &format!("/api/runs/{id}"), None).await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    assert!(!dir.exists());
    let (st, _) = f.get(&format!("/api/runs/{id}")).await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    // Stop on a run that is not supervised here is refused; "resume" on a
    // queued run (no checkpoint) starts it from scratch.
    let other = v2["id"].as_str().unwrap();
    let (st, _) = f.post(&format!("/api/runs/{other}/stop"), json!({})).await;
    assert_eq!(st, StatusCode::CONFLICT);
    let (st, v) = f
        .post(&format!("/api/runs/{other}/resume"), json!({}))
        .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(v["resume"].is_null(), "{v}");
    let pid = v["pid"].as_u64().unwrap() as u32;
    assert!(unamblify_web::supervisor::pid_alive(pid));
    let (_, v) = f.get(&format!("/api/runs/{other}")).await;
    assert_eq!(v["status"]["status"], "running");
    assert_eq!(v["supervised"], true);
    let (st, _) = f
        .post(&format!("/api/runs/{other}/resume"), json!({}))
        .await;
    assert_eq!(st, StatusCode::CONFLICT, "already running");
    f.state.sup.stop(other).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn capture_status_and_control() {
    let f = Fixture::new();
    let dstar = f.tmp.path().join("captured/dstar");
    std::fs::create_dir_all(&dstar).unwrap();
    // The field set the harness writes (`unamblify_data::control::Status`).
    let status = |pid: u32| {
        format!(
            r#"{{"state":"running","mode":"dstar","pid":{pid},"ports":["/dev/cu.usbserial-X"],"done":120,"failed":1,"total":1000,"frames_s":42.5,"utt_per_hour":300.0,"eta_s":10560.0,"current_key":"vctk/p225_010_mic2","current_keys":["vctk/p225_010_mic2"],"started":"2026-09-10T05:00:00Z","updated":"2026-09-10T05:10:00Z","canary_ok":true,"prodid":"AMBE3000F","version":"V121"}}"#
        )
    };
    std::fs::write(dstar.join("status.json"), status(999_999)).unwrap();
    std::fs::write(dstar.join("manifest.jsonl"), "{}\n{}\n{}\n").unwrap();
    std::fs::write(
        dstar.join("log.jsonl"),
        "{\"seq\":1,\"t\":1,\"level\":\"info\",\"msg\":\"cap\"}\n",
    )
    .unwrap();

    let (st, v) = f.get("/api/capture").await;
    assert_eq!(st, StatusCode::OK);
    let modes = v["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 4, "every VocoderMode, chip and software");
    assert_eq!(modes[0]["mode"], "dstar");
    assert_eq!(modes[0]["label"], "D-STAR");
    assert_eq!(modes[0]["software"], false);
    assert_eq!(modes[0]["family"], "ambe");
    assert_eq!(modes[0]["frame_ms"], 20);
    assert_eq!(modes[0]["status"]["done"], 120);
    assert_eq!(modes[0]["status"]["canary_ok"], true);
    assert_eq!(modes[0]["manifest_rows"], 3);
    assert_eq!(modes[0]["supervised"], false);
    assert_eq!(modes[0]["alive"], false, "pid 999999 is not alive");
    assert_eq!(modes[0]["log_tail"][0]["msg"], "cap");
    assert_eq!(modes[1]["mode"], "ysf-dmr");
    assert_eq!(modes[1]["label"], "YSF/DMR");
    assert!(modes[1]["status"].is_null());
    assert_eq!(modes[2]["mode"], "codec2-3200");
    assert_eq!(modes[2]["software"], true);
    assert_eq!(modes[2]["family"], "codec2");
    assert_eq!(modes[2]["frame_ms"], 20);
    assert_eq!(modes[3]["mode"], "codec2-1600");
    assert_eq!(modes[3]["label"], "Codec 2 1600 (M17)");
    assert_eq!(modes[3]["frame_ms"], 40);
    assert!(modes[3]["status"].is_null());

    let (st, v) = f.post("/api/capture/dstar/pause", json!({})).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["control"], "pause");
    let c: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dstar.join("control.json")).unwrap()).unwrap();
    assert_eq!(c["state"], "pause");
    let (_, _) = f.post("/api/capture/ysf-dmr/resume", json!({})).await;
    let c: serde_json::Value = serde_json::from_slice(
        &std::fs::read(f.tmp.path().join("captured/ysf-dmr/control.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(c["state"], "run");
    // The retired spelling routes to the current directory.
    let (_, _) = f.post("/api/capture/ysf-dn/stop", json!({})).await;
    let c: serde_json::Value = serde_json::from_slice(
        &std::fs::read(f.tmp.path().join("captured/ysf-dmr/control.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(c["state"], "stop");
    assert!(!f.tmp.path().join("captured/ysf-dn").exists());
    assert!(!f.tmp.path().join("captured/dmr").exists());

    let (st, _) = f.post("/api/capture/dstar/bogus", json!({})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f.post("/api/capture/fm/pause", json!({})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    let (_, v) = f.get("/api/capture").await;
    assert_eq!(v["modes"][0]["control"], "pause");

    // A harness that is alive but not ours (started from the shell, or
    // supervised before a restart): reported alive, and Start is refused
    // without touching control.json — a start must never un-pause it.
    std::fs::write(dstar.join("status.json"), status(std::process::id())).unwrap();
    let (_, v) = f.get("/api/capture").await;
    assert_eq!(v["modes"][0]["alive"], true);
    assert_eq!(v["modes"][0]["supervised"], false);
    assert_eq!(v["modes"][0]["pid"], std::process::id());
    let (st, v) = f
        .post("/api/capture/dstar/start", json!({ "args": ["--dry-run"] }))
        .await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert!(v["error"].as_str().unwrap().contains("already running"));
    let c: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dstar.join("control.json")).unwrap()).unwrap();
    assert_eq!(c["state"], "pause", "control.json left alone");
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_streams_carry_initial_and_live_events() {
    let f = Fixture::new();
    let a = write_run(
        &f.runs_dir(),
        "20260910-050000-alpha",
        "running",
        10,
        100,
        10,
    );
    // Let the poller learn the run before we append.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // /api/events opens with the run list and a capture snapshot.
    let text = f
        .sse("/api/events", Duration::from_secs(2), |t| {
            t.contains("event: capture")
        })
        .await;
    assert!(text.contains("event: runs"), "{text}");
    assert!(text.contains("20260910-050000-alpha"), "{text}");
    assert!(text.contains("\"snapshot\""), "{text}");

    // Per-run stream: initial status, then metric/log/sys/checkpoint as
    // the files grow. Append in a task while we read.
    let dir = a.clone();
    let writer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        use std::io::Write;
        let mut m = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join("metrics.jsonl"))
            .unwrap();
        writeln!(m, "{{\"step\":11,\"t\":1,\"k\":\"loss/total\",\"v\":0.5}}").unwrap();
        writeln!(m, "{{\"step\":11,\"t\":1,\"k\":\"sys/cpu\",\"v\":33}}").unwrap();
        let mut l = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join("log.jsonl"))
            .unwrap();
        writeln!(
            l,
            "{{\"seq\":21,\"t\":1,\"level\":\"warn\",\"msg\":\"live line\"}}"
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("checkpoints/step-000011")).unwrap();
    });
    let text = f
        .sse(
            "/api/runs/20260910-050000-alpha/events",
            Duration::from_secs(3),
            |t| {
                t.contains("event: checkpoint")
                    && t.contains("event: log")
                    && t.contains("event: sys")
                    && t.contains("event: metric")
            },
        )
        .await;
    writer.await.unwrap();
    assert!(text.starts_with("event: status"), "{text}");
    assert!(text.contains("\"total_steps\":100"), "{text}");
    assert!(text.contains("event: metric"), "{text}");
    assert!(text.contains("\"loss/total\""), "{text}");
    assert!(text.contains("event: sys"), "{text}");
    assert!(text.contains("event: log"), "{text}");
    assert!(text.contains("live line"), "{text}");
    assert!(text.contains("event: checkpoint"), "{text}");
    assert!(text.contains("step-000011"), "{text}");
    // sys/* rows are not in the metric event.
    let metric_line = text
        .lines()
        .skip_while(|l| *l != "event: metric")
        .nth(1)
        .unwrap();
    assert!(!metric_line.contains("sys/cpu"), "{metric_line}");

    let (st, _) = f.get("/api/runs/nope/events").await;
    assert_eq!(st, StatusCode::NOT_FOUND);
}
