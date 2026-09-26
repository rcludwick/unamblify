// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Supervisor lifecycle with the fake trainer: start → running → stop →
//! stopped (with a checkpoint), resume, failure, adoption on restart, and
//! capture start / control / stop.

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

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use serde_json::{Value, json};
use unamblify::LogRow;

use common::{Fixture, fake_trainer, opts, wait_for};

fn status(dir: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join("status.json")).unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn start_running_stop_stopped_resume() {
    let f = Fixture::new();
    let (st, v) = f
        .post(
            "/api/runs",
            json!({ "toml": "name = \"life\"\n[train]\nsteps = 100000\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    assert_eq!(v["status"]["status"], "running");
    assert_eq!(v["supervised"], true);
    let pid = v["status"]["pid"].as_u64().unwrap() as u32;
    assert!(unamblify_web::supervisor::pid_alive(pid));
    let dir = f.runs_dir().join(&id);

    // The fake trainer writes steps; wait until it has done a few.
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["step"]
            .as_u64()
            .unwrap_or(0)
            >= 3)
        .await
    );
    let (_, v) = f.get("/api/runs").await;
    assert_eq!(v[0]["supervised"], true);
    assert_eq!(v[0]["status"]["status"], "running");

    // Delete while running is refused.
    let res = f.raw("DELETE", &format!("/api/runs/{id}"), None).await;
    assert_eq!(res.status(), StatusCode::CONFLICT);

    // Stop: SIGTERM → the fake writes a checkpoint and exits 0.
    let (st, v) = f.post(&format!("/api/runs/{id}/stop"), json!({})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["status"]["status"], "stopped");
    assert!(v["status"].get("pid").is_none(), "{v}");
    assert!(!unamblify_web::supervisor::pid_alive(pid));
    assert!(!f.state.sup.is_running(&id));
    let step = status(&dir)["step"].as_u64().unwrap();
    assert!(step >= 3);
    let (_, v) = f.get(&format!("/api/runs/{id}/checkpoints")).await;
    assert_eq!(v.as_array().unwrap().len(), 1);
    assert_eq!(v[0]["step"], step);
    assert_eq!(v[0]["model"], true);

    // log.jsonl: the supervisor's lines, the trainer's own rows, and what
    // the child printed (copied from child.log at exit) — each once, with
    // one monotonic seq sequence. The trainer's echo was off, so nothing
    // it logged itself appears a second time from stderr.
    let (_, logs) = f.get(&format!("/api/runs/{id}/logs")).await;
    let rows: Vec<LogRow> = serde_json::from_value(logs).unwrap();
    let msgs: Vec<&str> = rows.iter().map(|r| r.msg.as_str()).collect();
    assert!(msgs[0].starts_with("supervisor: spawned pid"), "{msgs:?}");
    assert_eq!(
        msgs.iter()
            .filter(|m| m.starts_with("fake trainer: started pid"))
            .count(),
        1,
        "{msgs:?}"
    );
    assert_eq!(
        msgs.iter()
            .filter(|m| m.contains("checkpoint written"))
            .count(),
        1,
        "{msgs:?}"
    );
    assert!(
        msgs.contains(&"fake trainer: stdout line"),
        "stdout copied at exit: {msgs:?}"
    );
    assert!(
        msgs.last().unwrap().contains("process exited: exit code 0"),
        "{msgs:?}"
    );
    assert!(
        !msgs.iter().any(|m| m.starts_with("[info]")),
        "no echoed duplicates: {msgs:?}"
    );
    // A JSON log line from the trainer keeps its level, gets a fresh seq.
    let dbg = rows
        .iter()
        .find(|r| r.msg == "a jsonl line from the trainer")
        .unwrap();
    assert_eq!(dbg.level, unamblify::LogLevel::Debug);
    assert!(dbg.seq < 900_000);
    assert!(
        rows.windows(2).all(|w| w[0].seq < w[1].seq),
        "seq strictly monotonic (one owner): {:?}",
        rows.iter().map(|r| r.seq).collect::<Vec<_>>()
    );
    let child_log = std::fs::read_to_string(dir.join("child.log")).unwrap();
    assert!(
        child_log.contains("fake trainer: stdout line"),
        "{child_log}"
    );

    // Resume: re-exec with --resume <latest ckpt>; the fake continues from
    // that step.
    let (st, v) = f.post(&format!("/api/runs/{id}/resume"), json!({})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(v["resume"].as_str().unwrap().contains("step-"));
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["step"]
            .as_u64()
            .unwrap_or(0)
            > step + 1)
        .await
    );
    assert_eq!(status(&dir)["status"], "running");
    let (st, _) = f.post(&format!("/api/runs/{id}/resume"), json!({})).await;
    assert_eq!(st, StatusCode::CONFLICT, "already running");
    let (st, v) = f.post(&format!("/api/runs/{id}/stop"), json!({})).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["status"]["status"], "stopped");
    let (_, logs) = f.get(&format!("/api/runs/{id}/logs?tail=200")).await;
    assert!(
        logs.as_array()
            .unwrap()
            .iter()
            .any(|r| r["msg"].as_str().unwrap().contains("resuming from")),
        "{logs}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn finished_and_failed_runs_get_the_right_status() {
    let f = Fixture::with(|o| o.grace = Duration::from_secs(2));
    // The fake trainer runs for as long as the config says: a tiny run.
    let (st, v) = f
        .post(
            "/api/runs",
            json!({ "toml": "name = \"short\"\n[train]\nsteps = 3\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    let dir = f.runs_dir().join(&id);
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["status"]
            == "finished")
        .await,
        "{}",
        status(&dir)
    );
    assert!(wait_for(Duration::from_secs(2), || !f.state.sup.is_running(&id)).await);
    assert!(status(&dir).get("pid").is_none());

    let (st, v) = f
        .post(
            "/api/runs",
            json!({ "toml": "name = \"bad\"\n[eval]\nclips = \"FAKE_FAIL\"\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    let dir = f.runs_dir().join(&id);
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["status"]
            == "failed")
        .await,
        "{}",
        status(&dir)
    );
    let (_, logs) = f.get(&format!("/api/runs/{id}/logs")).await;
    let text = logs.to_string();
    assert!(text.contains("failing on purpose"), "{text}");
    assert!(text.contains("exit code 3"), "{text}");
    assert!(text.contains("\"level\":\"error\""), "{text}");
    // Now that it is dead, delete works.
    let res = f.raw("DELETE", &format!("/api/runs/{id}"), None).await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_adopts_live_runs_and_marks_dead_ones_stopped() {
    let f = Fixture::new();
    let (st, v) = f
        .post(
            "/api/runs",
            json!({ "toml": "name = \"adopt\"\n[train]\nsteps = 100000\n" }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    let pid = v["status"]["pid"].as_u64().unwrap() as u32;
    let dir = f.runs_dir().join(&id);
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["step"]
            .as_u64()
            .unwrap_or(0)
            >= 2)
        .await
    );

    // A run whose recorded pid is long gone.
    let dead = common::write_run(&f.runs_dir(), "20260910-010000-dead", "running", 5, 10, 5);
    let mut st_dead = status(&dead);
    st_dead["pid"] = json!(9_999_999u32);
    std::fs::write(dead.join("status.json"), st_dead.to_string()).unwrap();
    // And one queued with no pid.
    let queued = common::write_run(&f.runs_dir(), "20260910-020000-queued", "queued", 0, 10, 0);

    // "Restart": a second server over the same directories. The first
    // one's supervisor still holds the child, which is exactly the
    // situation after a crash (the child is reparented, not killed).
    let o = opts(f.tmp.path());
    let state2 = unamblify_web::build(&o).unwrap();
    let app2 = unamblify_web::router(Arc::clone(&state2), None);
    let f2 = Fixture {
        tmp: tempfile::tempdir().unwrap(),
        state: state2,
        app: app2,
        token: None,
    };

    assert!(f2.state.sup.is_running(&id), "adopted by pid liveness");
    assert_eq!(f2.state.sup.get(&id).unwrap().pid, pid);
    assert!(f2.state.sup.get(&id).unwrap().adopted);
    assert_eq!(status(&dead)["status"], "stopped");
    assert!(status(&dead).get("pid").is_none());
    assert_eq!(
        status(&queued)["status"],
        "queued",
        "a queued run has no process to be dead; it waits for a start"
    );
    let (_, v) = f2.get("/api/runs").await;
    let row = v
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == id)
        .unwrap();
    assert_eq!(row["supervised"], true);

    // Stop through the adopting server: SIGTERM → the fake checkpoints
    // and exits; the adopted watcher finalises.
    let (st, v) = f2.post(&format!("/api/runs/{id}/stop"), json!({})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["status"]
            == "stopped")
        .await,
        "{}",
        status(&dir)
    );
    assert!(!unamblify_web::supervisor::pid_alive(pid));
    assert!(wait_for(Duration::from_secs(3), || !f2.state.sup.is_running(&id)).await);
    // The original server's exit watcher also finalised and forgot it.
    assert!(wait_for(Duration::from_secs(3), || !f.state.sup.is_running(&id)).await);
    let (_, logs) = f.get(&format!("/api/runs/{id}/logs?tail=50")).await;
    assert!(logs.to_string().contains("process exited"), "{logs}");
}

#[tokio::test(flavor = "multi_thread")]
async fn capture_start_control_and_stop() {
    let f = Fixture::new();
    // The fake capture finds its dir through UNAMBLIFY_DATA, which the
    // supervisor sets on the child.
    let (st, v) = f
        .post(
            "/api/capture/ysf-dmr/start",
            json!({ "args": ["--limit", "5"] }),
        )
        .await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let pid = v["pid"].as_u64().unwrap() as u32;
    let dir = f.tmp.path().join("captured/ysf-dmr");
    assert!(
        wait_for(Duration::from_secs(5), || dir.join("status.json").exists()
            && status(&dir)["done"].as_u64().unwrap_or(0) >= 2)
        .await
    );
    let (st, _) = f.post("/api/capture/ysf-dmr/start", json!({})).await;
    assert_eq!(st, StatusCode::CONFLICT, "already running");
    // The harness's stderr is followed into log.jsonl while it runs.
    assert!(
        wait_for(Duration::from_secs(3), || {
            unamblify_web::runs::read_logs(&dir, Some(50), None)
                .unwrap_or_default()
                .iter()
                .any(|r| r.msg.contains("fake capture: started"))
        })
        .await
    );

    let (_, v) = f.get("/api/capture").await;
    let m = &v["modes"][1];
    assert_eq!(m["mode"], "ysf-dmr");
    assert_eq!(m["supervised"], true);
    assert_eq!(m["alive"], true);
    assert_eq!(m["pid"], pid);
    assert_eq!(m["status"]["state"], "running");
    assert_eq!(m["control"], "run");
    assert!(
        m["log_tail"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["msg"].as_str().unwrap().contains("fake capture: started")),
        "{m}"
    );
    let spawned = m["log_tail"][0]["msg"].as_str().unwrap();
    assert!(
        spawned.contains("capture --mode ysf-dmr --limit 5"),
        "{spawned}"
    );

    // Pause via control.json; the harness reports paused.
    f.post("/api/capture/ysf-dmr/pause", json!({})).await;
    assert!(
        wait_for(Duration::from_secs(3), || status(&dir)["state"] == "paused").await,
        "{}",
        status(&dir)
    );
    let paused_done = status(&dir)["done"].as_u64().unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        status(&dir)["done"].as_u64().unwrap(),
        paused_done,
        "no progress while paused"
    );
    f.post("/api/capture/ysf-dmr/resume", json!({})).await;
    assert!(
        wait_for(Duration::from_secs(3), || status(&dir)["done"]
            .as_u64()
            .unwrap()
            > paused_done)
        .await
    );

    // Stop via control.json: the harness exits 0 on its own; the
    // supervisor notices and forgets it.
    f.post("/api/capture/ysf-dmr/stop", json!({})).await;
    assert!(
        wait_for(Duration::from_secs(5), || {
            !unamblify_web::supervisor::pid_alive(pid)
        })
        .await
    );
    assert!(
        wait_for(Duration::from_secs(3), || !f
            .state
            .sup
            .is_running("capture:ysf-dmr"))
        .await
    );
    assert_eq!(status(&dir)["state"], "stopped");
    let (_, v) = f.get("/api/capture").await;
    assert_eq!(v["modes"][1]["supervised"], false);
    assert_eq!(v["modes"][1]["alive"], false);

    // Start again after a stop: control.json is reset to run first.
    let (st, v) = f.post("/api/capture/ysf-dmr/start", json!({})).await;
    assert_eq!(st, StatusCode::CREATED, "{v}");
    let pid2 = v["pid"].as_u64().unwrap() as u32;
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["state"]
            == "running"
            && status(&dir)["done"].as_u64().unwrap_or(0) >= 2)
        .await
    );
    // Supervisor stop (SIGTERM path) for a capture job.
    f.state.sup.stop("capture:ysf-dmr").await.unwrap();
    assert!(!unamblify_web::supervisor::pid_alive(pid2));
    assert_eq!(status(&dir)["state"], "stopped");
}

/// The server going away must not take its children with it: they get
/// files, never pipes, so the next write after the server is gone does
/// not hit EPIPE (bash dies of SIGPIPE on that, Rust's eprintln! panics).
#[test]
fn children_survive_the_server_going_away() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = unamblify::RunConfig {
        name: "orphan".to_owned(),
        ..Default::default()
    };
    // Twelve steps, from the run's own config: the step count used to
    // travel in a process-wide environment variable, which a neighbouring
    // test setting its own raced with.
    cfg.train.steps = 12;
    let (id, dir) = unamblify_web::runs::create_run(tmp.path(), &cfg).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let pid = rt.block_on(async {
        let hub = Arc::new(unamblify_web::events::Hub::default());
        let sup = Arc::new(unamblify_web::supervisor::Supervisor::new(
            fake_trainer(),
            Duration::from_secs(5),
            hub,
        ));
        let pid = sup.start_train(&id, &dir, None).unwrap();
        // Let it print a couple of lines while the server still exists.
        tokio::time::sleep(Duration::from_millis(150)).await;
        pid
    });
    // "Server exit": every task, pipe reader and handle is dropped.
    rt.shutdown_background();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while unamblify_web::supervisor::pid_alive(pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !unamblify_web::supervisor::pid_alive(pid),
        "child never finished"
    );
    let st = status(&dir);
    assert_eq!(st["status"], "finished", "{st}");
    assert_eq!(st["step"], 12);
    let child_log = std::fs::read_to_string(dir.join("child.log")).unwrap();
    assert!(
        child_log.contains("fake trainer: finished"),
        "the child kept writing after the server was gone: {child_log}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn direct_supervisor_api_and_sigkill_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = Arc::new(unamblify_web::events::Hub::default());
    let sup = Arc::new(unamblify_web::supervisor::Supervisor::new(
        fake_trainer(),
        Duration::from_millis(300),
        Arc::clone(&hub),
    ));
    let cfg = unamblify::RunConfig {
        name: "direct".to_owned(),
        ..Default::default()
    };
    let (id, dir) = unamblify_web::runs::create_run(tmp.path(), &cfg).unwrap();
    let mut rx = hub.subscribe();
    let pid = sup.start_train(&id, &dir, None).unwrap();
    assert!(sup.is_running(&id));
    let ev = rx.recv().await.unwrap();
    assert_eq!(ev.kind, unamblify_web::events::Kind::Status);
    assert_eq!(ev.data["status"], "running");
    assert_eq!(ev.data["pid"], pid);
    assert!(
        wait_for(Duration::from_secs(5), || status(&dir)["step"]
            .as_u64()
            .unwrap_or(0)
            >= 1)
        .await
    );
    sup.stop(&id).await.unwrap();
    assert!(!sup.is_running(&id));
    assert_eq!(status(&dir)["status"], "stopped");
    assert!(
        sup.stop(&id).await.is_err(),
        "stopping a stopped run is a conflict"
    );
    assert!(sup.keys().is_empty());
}
