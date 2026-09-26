// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `--token`: `/api/*` needs `Authorization: Bearer <token>`; the UI does
//! not; the default bind flips to all interfaces.

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

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use tower::ServiceExt;
use unamblify_web::ServeOpts;

use common::Fixture;

#[tokio::test(flavor = "multi_thread")]
async fn bearer_token_rejects_and_accepts() {
    let f = Fixture::with(|o| o.token = Some("s3cret-token".to_owned()));

    // Fixture requests carry the token.
    let (st, v) = f.get("/api/health").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["ok"], true);

    // No header, wrong scheme, wrong token, prefix of the token: 401.
    for auth in [
        None,
        Some("Basic s3cret-token"),
        Some("Bearer nope"),
        Some("Bearer s3cret-toke"),
        Some("Bearer s3cret-token-plus"),
    ] {
        let mut b = Request::builder().uri("/api/health");
        if let Some(a) = auth {
            b = b.header(header::AUTHORIZATION, a);
        }
        let res = f
            .app
            .clone()
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{auth:?}");
        assert_eq!(res.headers()[header::WWW_AUTHENTICATE], "Bearer");
    }
    // Non-API routes (the UI) are open.
    let res = f
        .app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = f
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ui/style.css")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // POSTs and SSE are covered by the same layer.
    let res = f
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/dstar/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let res = f
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[test]
fn bind_defaults_follow_the_token() {
    let o = ServeOpts::default();
    assert_eq!(o.bind_addr().unwrap().to_string(), "127.0.0.1:8787");
    let o = ServeOpts {
        token: Some("t".to_owned()),
        ..ServeOpts::default()
    };
    assert_eq!(o.bind_addr().unwrap().to_string(), "0.0.0.0:8787");
    let o = ServeOpts {
        bind: Some("127.0.0.1:9000".parse().unwrap()),
        token: Some("t".to_owned()),
        ..ServeOpts::default()
    };
    assert_eq!(o.bind_addr().unwrap().to_string(), "127.0.0.1:9000");
}

#[tokio::test(flavor = "multi_thread")]
async fn non_loopback_bind_without_token_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mut o = common::opts(tmp.path());
    o.bind = Some("0.0.0.0:0".parse().unwrap());
    let err = unamblify_web::serve(o).await.unwrap_err().to_string();
    assert!(err.contains("--token"), "{err}");
}
