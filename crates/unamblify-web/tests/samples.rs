// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Contract tests for the Samples navigator over a fake data root: a
//! prepared manifest, one captured manifest, tiny real WAVs.

#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

mod common;

use std::path::Path;
use std::time::Duration;

use axum::http::{StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};

use common::{Fixture, wait_for, write_checkpoint, write_run};

/// `n` samples of a decaying sine at `rate`.
fn tone(n: usize, rate: u32, f: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.4 * (2.0 * std::f32::consts::PI * f * t).sin() * (-t).exp()
        })
        .collect()
}

fn prepared_row(key: &str, speaker: &str, split: &str, dur: f64) -> String {
    let corpus = key.split('/').next().unwrap();
    json!({
        "key": key, "corpus": corpus, "speaker": speaker, "gender": "F", "split": split,
        "duration_s": dur, "src_rate": 48000, "src_path": format!("raw/{key}.flac"),
        "licence": "CC-BY-4.0", "rms_dbfs_in": -21.0, "gain_db": -4.0, "trim_lead_s": 0.0,
        "trim_tail_s": 0.0, "sha256_16k": "a", "sha256_8k": "b", "prepared_at": "2026-09-10T03:00:00Z"
    })
    .to_string()
}

fn capture_row(key: &str, mode: &str, frames: u32) -> String {
    json!({
        "key": key, "mode": mode, "frames": frames, "port": "/dev/fake", "prodid": "AMBE3000F",
        "version": "V121", "encode_ms": 10, "decode_ms": 5, "sha256_ambe": "c", "sha256_wav": "d",
        "captured_at": "2026-09-10T04:00:00Z", "attempts": 1
    })
    .to_string()
}

/// Write a prepared + captured data root under `root`. Keys:
/// `vctk/p225_001_mic2` (dstar + ysf-dmr), `vctk/p226_002_mic2` (dstar),
/// `ljspeech/LJ001-0001` (dstar, dev), `ljspeech/LJ001-0002` (prepared only).
fn write_data_root(root: &Path) {
    let prepared = root.join("prepared");
    let rows = [
        ("vctk/p225_001_mic2", "p225", "train", 0.5),
        ("vctk/p226_002_mic2", "p226", "train", 0.25),
        ("ljspeech/LJ001-0001", "LJ", "dev", 0.75),
        ("ljspeech/LJ001-0002", "LJ", "dev", 0.75),
    ];
    let mut m = String::new();
    for (key, spk, split, dur) in rows {
        m.push_str(&prepared_row(key, spk, split, dur));
        m.push('\n');
        let dir = prepared.join(key);
        std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
        let n16 = (dur * 16_000.0) as usize;
        unamblify_audio::write_wav_s16(
            prepared.join(format!("{key}.16k.wav")),
            &tone(n16, 16_000, 440.0),
            16_000,
        )
        .unwrap();
        unamblify_audio::write_wav_s16(
            prepared.join(format!("{key}.8k.wav")),
            &tone(n16 / 2, 8_000, 440.0),
            8_000,
        )
        .unwrap();
    }
    // A torn last line, as an interrupted append leaves.
    m.push_str("{\"key\":\"vctk/p2");
    std::fs::write(prepared.join("manifest.jsonl"), m).unwrap();

    for (mode, keys) in [
        (
            "dstar",
            vec![
                "vctk/p225_001_mic2",
                "vctk/p226_002_mic2",
                "ljspeech/LJ001-0001",
            ],
        ),
        ("ysf-dmr", vec!["vctk/p225_001_mic2"]),
    ] {
        let dir = root.join("captured").join(mode);
        let mut c = String::new();
        for key in keys {
            let frames = 25;
            c.push_str(&capture_row(key, mode, frames));
            c.push('\n');
            let p = dir.join(format!("{key}.wav"));
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            unamblify_audio::write_wav_s16(&p, &tone(frames as usize * 160, 8_000, 220.0), 8_000)
                .unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("manifest.jsonl"), c).unwrap();
        std::fs::write(
            dir.join("canary.json"),
            json!({"mode": mode, "clip": "canary/x.wav", "frames_sha256": "e", "frames_first_16": "f",
                   "lag_samples": 3, "prodid": "AMBE3000F", "version": "V121", "recorded_at": "2026-09-10T04:00:00Z"})
                .to_string(),
        )
        .unwrap();
    }
}

fn keys_of(v: &Value) -> Vec<&str> {
    v["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["key"].as_str().unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn list_filter_facets_random_and_get() {
    let f = Fixture::new();
    write_data_root(f.tmp.path());

    let (st, v) = f.get("/api/samples").await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["total"], 3, "prepared-only rows are not joined: {v}");
    assert_eq!(v["page"], 1);
    assert_eq!(v["per_page"], 50);
    assert_eq!(
        keys_of(&v),
        vec![
            "ljspeech/LJ001-0001",
            "vctk/p225_001_mic2",
            "vctk/p226_002_mic2"
        ],
        "key order"
    );
    let p225 = &v["items"][1];
    assert_eq!(p225["speaker"], "p225");
    assert_eq!(p225["corpus"], "vctk");
    assert_eq!(p225["split"], "train");
    assert_eq!(p225["duration_s"], 0.5);
    assert_eq!(
        p225["modes"],
        json!([
            {"mode": "dstar", "label": "D-STAR", "frames": 25, "frame_ms": 20},
            {"mode": "ysf-dmr", "label": "YSF/DMR", "frames": 25, "frame_ms": 20}
        ])
    );
    assert!(p225.get("prepared_loc").is_none());

    // Filters.
    let (_, v) = f.get("/api/samples?corpus=vctk").await;
    assert_eq!(v["total"], 2);
    let (_, v) = f.get("/api/samples?split=dev").await;
    assert_eq!(keys_of(&v), vec!["ljspeech/LJ001-0001"]);
    let (_, v) = f.get("/api/samples?speaker=p226").await;
    assert_eq!(keys_of(&v), vec!["vctk/p226_002_mic2"]);
    let (_, v) = f.get("/api/samples?mode=ysf-dmr").await;
    assert_eq!(keys_of(&v), vec!["vctk/p225_001_mic2"]);
    let (_, v) = f.get("/api/samples?q=001").await;
    assert_eq!(v["total"], 2);
    let (_, v) = f
        .get("/api/samples?q=001&corpus=vctk&mode=dstar&split=train")
        .await;
    assert_eq!(keys_of(&v), vec!["vctk/p225_001_mic2"]);
    let (st, _) = f.get("/api/samples?split=nope").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f.get("/api/samples?mode=fm").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    // Several sets: only what all of them captured, for a side-by-side.
    let (_, one) = f.get("/api/samples?mode=dstar").await;
    let (_, v) = f.get("/api/samples?mode=dstar,ysf-dmr").await;
    assert_eq!(keys_of(&v), vec!["vctk/p225_001_mic2"]);
    assert!(one["total"].as_u64() > v["total"].as_u64());
    let (_, v) = f.get("/api/samples?mode=dstar,,dstar").await;
    assert_eq!(
        v["total"], one["total"],
        "a repeat or a gap is not a second set"
    );
    let (st, _) = f.get("/api/samples?mode=dstar,fm").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    // Paging.
    let (_, v) = f.get("/api/samples?per_page=2").await;
    assert_eq!(v["items"].as_array().unwrap().len(), 2);
    assert_eq!(v["total"], 3);
    let (_, v) = f.get("/api/samples?per_page=2&page=2").await;
    assert_eq!(keys_of(&v), vec!["vctk/p226_002_mic2"]);
    let (_, v) = f.get("/api/samples?per_page=2&page=3").await;
    assert!(v["items"].as_array().unwrap().is_empty());
    let (_, v) = f.get("/api/samples?per_page=99999").await;
    assert_eq!(v["per_page"], 500);

    // Facets.
    let (st, v) = f.get("/api/samples/facets").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["total"], 3);
    assert_eq!(
        v["corpora"],
        json!([{"name": "ljspeech", "count": 1}, {"name": "vctk", "count": 2}])
    );
    assert_eq!(
        v["splits"],
        json!([{"name": "train", "count": 2}, {"name": "dev", "count": 1}])
    );
    assert_eq!(
        v["modes"],
        json!([
            {"name": "dstar", "label": "D-STAR", "count": 3},
            {"name": "ysf-dmr", "label": "YSF/DMR", "count": 1}
        ])
    );
    let spk = v["speakers"].as_array().unwrap();
    assert_eq!(spk.len(), 3);
    assert_eq!(
        spk[0],
        json!({"name": "LJ", "corpus": "ljspeech", "count": 1})
    );

    // Random honours the filters.
    for _ in 0..5 {
        let (st, v) = f.get("/api/samples/random?mode=ysf-dmr").await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["key"], "vctk/p225_001_mic2");
    }
    let (st, v) = f.get("/api/samples/random").await;
    assert_eq!(st, StatusCode::OK);
    assert!(v["key"].as_str().unwrap().contains('/'));
    let (st, _) = f.get("/api/samples/random?speaker=nobody").await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    // One sample: the joined row plus its manifest rows.
    let (st, v) = f.get("/api/samples/vctk/p225_001_mic2").await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["key"], "vctk/p225_001_mic2");
    assert_eq!(v["prepared"]["src_rate"], 48000);
    assert_eq!(v["prepared"]["sha256_16k"], "a");
    assert_eq!(v["captured"]["dstar"]["prodid"], "AMBE3000F");
    assert_eq!(v["captured"]["ysf-dmr"]["frames"], 25);
    assert!(v["captured"].get("codec2-3200").is_none());
    let (st, _) = f.get("/api/samples/ljspeech/LJ001-0002").await;
    assert_eq!(st, StatusCode::NOT_FOUND, "prepared but not captured");
    let (st, _) = f.get("/api/samples/vctk/nope").await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, v) = f.get("/api/samples/noslash").await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn audio_spec_cache_and_traversal() {
    let f = Fixture::new();
    write_data_root(f.tmp.path());

    // Audio passthrough for each signal.
    for (which, rate, n) in [
        ("clean16", 16_000u32, 8_000usize),
        ("clean8", 8_000, 4_000),
        ("degraded-dstar", 8_000, 25 * 160),
        ("degraded-ysf-dmr", 8_000, 25 * 160),
    ] {
        let res = f
            .raw(
                "GET",
                &format!("/api/samples/vctk/p225_001_mic2/audio/{which}"),
                None,
            )
            .await;
        assert_eq!(res.status(), StatusCode::OK, "{which}");
        assert_eq!(res.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(
            res.headers()[header::CACHE_CONTROL],
            "private, max-age=3600"
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.starts_with(b"RIFF"), "{which}");
        let tmp = f.tmp.path().join(format!("got-{which}.wav"));
        std::fs::write(&tmp, &bytes).unwrap();
        let a = unamblify_audio::read_wav(&tmp).unwrap();
        assert_eq!((a.rate, a.samples.len()), (rate, n), "{which}");
    }
    // Not captured in that mode, unknown signal, unknown key.
    let (st, _) = f
        .get("/api/samples/vctk/p226_002_mic2/audio/degraded-ysf-dmr")
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f
        .get("/api/samples/vctk/p226_002_mic2/audio/degraded-fm")
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f.get("/api/samples/vctk/p226_002_mic2/audio/out").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f.get("/api/samples/vctk/nope/audio/clean16").await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    // Path traversal is refused before the disk is touched.
    std::fs::write(f.tmp.path().join("secret.wav"), b"RIFFsecret").unwrap();
    for bad in [
        "/api/samples/../secret.wav",
        "/api/samples/..%2Fsecret.wav",
        "/api/samples/vctk/../../secret.wav/audio/clean16",
        "/api/samples/vctk/..%2F..%2Fsecret.wav/audio/clean16",
        "/api/samples/vctk/p225_001_mic2/audio/..%2F..%2F..%2Fsecret",
        "/api/samples/vctk/p225_001_mic2/spec/../x",
        "/api/samples/vctk/./p225_001_mic2",
    ] {
        let res = f.raw("GET", bad, None).await;
        assert!(
            matches!(
                res.status(),
                StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND
            ),
            "{bad}: {}",
            res.status()
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        assert!(!body.starts_with(b"RIFF"), "{bad} served a file");
    }

    // Spectrograms: the trainer's shape, 8 kHz signals upsampled onto the
    // same axes, cached on the second request.
    let res = f
        .raw("GET", "/api/samples/vctk/p225_001_mic2/spec/clean16", None)
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["x-cache"], "miss");
    assert_eq!(res.headers()[header::CONTENT_TYPE], "application/json");
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["n_mels"], 80);
    assert_eq!(v["hop"], 256);
    assert_eq!(v["n_fft"], 1024);
    assert_eq!(v["rate"], 16000);
    assert_eq!(v["db_min"], -100.0);
    assert_eq!(v["db_max"], 0.0);
    assert_eq!(v["frames"], 1 + 8_000 / 256);
    let rows = v["clean16"].as_array().unwrap();
    assert_eq!(rows.len(), 1 + 8_000 / 256);
    assert_eq!(rows[0].as_array().unwrap().len(), 80);
    let cache = f.tmp.path().join("cache/spec");
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
    let res = f
        .raw("GET", "/api/samples/vctk/p225_001_mic2/spec/clean16", None)
        .await;
    assert_eq!(res.headers()["x-cache"], "hit");
    let again = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(again, bytes);
    // 8 kHz signals land on the same axes (frames of the 16 kHz length).
    let (st, v) = f.get("/api/samples/vctk/p225_001_mic2/spec/clean8").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["frames"], 1 + 8_000 / 256);
    assert_eq!(v["clean8"].as_array().unwrap().len(), 1 + 8_000 / 256);
    let (st, v) = f
        .get("/api/samples/vctk/p225_001_mic2/spec/degraded-dstar")
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["frames"], 1 + 2 * 25 * 160 / 256);
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 3);
    let (st, _) = f
        .get("/api/samples/vctk/p226_002_mic2/spec/degraded-ysf-dmr")
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    // Nothing else was written under the data root.
    let mut top: Vec<String> = std::fs::read_dir(f.tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with("got-"))
        .collect();
    top.sort();
    assert_eq!(
        top,
        vec![
            "cache",
            "captured",
            "configs",
            "prepared",
            "runs",
            "secret.wav"
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn index_refreshes_when_a_manifest_grows() {
    let f = Fixture::new();
    write_data_root(f.tmp.path());
    let (_, v) = f.get("/api/samples").await;
    assert_eq!(v["total"], 3);
    // Capture another utterance: append a row (and make the file's stamp
    // differ even on a coarse clock).
    let dir = f.tmp.path().join("captured/dstar");
    let mut c = std::fs::read_to_string(dir.join("manifest.jsonl")).unwrap();
    c.push_str(&capture_row("ljspeech/LJ001-0002", "dstar", 30));
    c.push('\n');
    std::fs::write(dir.join("manifest.jsonl"), c).unwrap();
    // The recheck is rate-limited to once per 2 s.
    let (_, v) = f.get("/api/samples").await;
    assert_eq!(v["total"], 3, "no stat within the recheck window");
    tokio::time::sleep(unamblify_web::samples::RECHECK + Duration::from_millis(100)).await;
    let (_, v) = f.get("/api/samples").await;
    assert_eq!(v["total"], 4);
    let (_, v) = f.get("/api/samples/facets").await;
    assert_eq!(
        v["modes"][0],
        json!({"name": "dstar", "label": "D-STAR", "count": 4})
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn model_render_spawns_infer_and_polls() {
    let f = Fixture::new();
    write_data_root(f.tmp.path());
    let run = write_run(
        &f.runs_dir(),
        "20260910-050000-alpha",
        "finished",
        20,
        20,
        20,
    );
    write_checkpoint(&run, 20, "p225_001");
    let key = "vctk/p225_001_mic2";

    let (st, v) = f
        .get(&format!(
            "/api/samples/{key}/model?run=20260910-050000-alpha&step=20"
        ))
        .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["ready"], false);
    assert_eq!(v["running"], false);
    let (st, _) = f.get(&format!("/api/samples/{key}/model")).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = f
        .get(&format!(
            "/api/samples/{key}/model?run=20260910-050000-alpha&step=15"
        ))
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND, "no such checkpoint");
    let (st, _) = f
        .get(&format!("/api/samples/{key}/model?run=nope&step=20"))
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f
        .get(&format!(
            "/api/samples/{key}/model/audio?run=20260910-050000-alpha&step=20"
        ))
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND, "not rendered yet");

    let (st, v) = f
        .post(
            &format!("/api/samples/{key}/model"),
            json!({"run": "20260910-050000-alpha", "step": 20}),
        )
        .await;
    assert_eq!(st, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["ready"], false);
    assert_eq!(v["running"], true);
    let out = run.join("samples/step-000020/vctk_p225_001_mic2.out.wav");
    assert!(
        wait_for(Duration::from_secs(10), || out.is_file()).await,
        "infer never wrote {}",
        out.display()
    );
    let ok = wait_for(Duration::from_secs(5), || {
        f.state
            .sup
            .infer_state("20260910-050000-alpha", 20, key)
            .is_none()
    })
    .await;
    assert!(ok, "job not cleared");
    let (st, v) = f
        .get(&format!(
            "/api/samples/{key}/model?run=20260910-050000-alpha&step=20"
        ))
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ready"], true);
    assert_eq!(
        v["audio"],
        "/api/samples/vctk/p225_001_mic2/model/audio?run=20260910-050000-alpha&step=20"
    );
    assert_eq!(
        v["spec"],
        "/api/samples/vctk/p225_001_mic2/model/spec?run=20260910-050000-alpha&step=20"
    );
    let res = f.raw("GET", v["audio"].as_str().unwrap(), None).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[header::CONTENT_TYPE], "audio/wav");
    let (st, sv) = f.get(v["spec"].as_str().unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(sv["out"].as_array().unwrap().len(), 2);
    assert!(run.join("samples/step-000020/infer.log").is_file());
    // A second POST on a rendered target answers 200 without spawning.
    let (st, v) = f
        .post(
            &format!("/api/samples/{key}/model"),
            json!({"run": "20260910-050000-alpha", "step": 20}),
        )
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ready"], true);
    // Unknown key, missing checkpoint.
    let (st, _) = f
        .post(
            "/api/samples/vctk/nope/model",
            json!({"run": "20260910-050000-alpha", "step": 20}),
        )
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = f
        .post(
            &format!("/api/samples/{key}/model"),
            json!({"run": "20260910-050000-alpha", "step": 7}),
        )
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread")]
async fn model_render_reports_a_binary_without_infer() {
    let f = Fixture::new();
    write_data_root(f.tmp.path());
    let run = write_run(
        &f.runs_dir(),
        "20260910-050000-alpha",
        "finished",
        20,
        20,
        20,
    );
    write_checkpoint(&run, 20, "p225_001");
    let key = "vctk/p226_002_mic2";
    std::fs::write(run.join("no-infer"), b"").unwrap();
    let (st, _) = f
        .post(
            &format!("/api/samples/{key}/model"),
            json!({"run": "20260910-050000-alpha", "step": 20}),
        )
        .await;
    assert_eq!(st, StatusCode::ACCEPTED);
    let ok = wait_for(Duration::from_secs(10), || {
        matches!(
            f.state.sup.infer_state("20260910-050000-alpha", 20, key),
            Some(unamblify_web::supervisor::InferState::Failed { .. })
        )
    })
    .await;
    assert!(ok, "job did not fail");
    let (st, v) = f
        .get(&format!(
            "/api/samples/{key}/model?run=20260910-050000-alpha&step=20"
        ))
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ready"], false);
    assert_eq!(v["running"], false);
    assert!(
        v["error"].as_str().unwrap().contains("infer unavailable"),
        "{v}"
    );
}
