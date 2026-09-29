// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The embedded UI (`ui/`): `index.html`, `app.js`, `style.css`, and the
//! vendored uPlot build. Served at `/` and `/ui/<file>`; no build step.

use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use rust_embed::Embed;

/// The `ui/` directory, compiled in.
#[derive(Embed)]
#[folder = "ui/"]
pub struct Assets;

/// Files the UI must ship with; the embed test checks each exists.
pub const REQUIRED: [&str; 5] = [
    "index.html",
    "app.js",
    "style.css",
    "uplot.iife.min.js",
    "uplot.min.css",
];

fn serve(path: &str) -> Response {
    match Assets::get(path) {
        Some(f) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [
                    (header::CONTENT_TYPE, mime.as_ref().to_owned()),
                    (header::CACHE_CONTROL, "no-cache".to_owned()),
                ],
                f.data.into_owned(),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

/// `GET /`.
pub async fn index() -> Response {
    serve("index.html")
}

/// `GET /favicon.ico`: nothing to show, but no 404 noise in the console.
pub async fn favicon() -> Response {
    StatusCode::NO_CONTENT.into_response()
}

/// `GET /ui/{*path}`.
pub async fn asset(Path(path): Path<String>) -> Response {
    serve(path.trim_start_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_assets_are_embedded_and_index_is_well_formed() {
        for name in REQUIRED {
            assert!(Assets::get(name).is_some(), "missing ui/{name}");
        }
        let html = String::from_utf8(Assets::get("index.html").unwrap().data.into_owned()).unwrap();
        assert!(html.starts_with("<!doctype html>") || html.starts_with("<!DOCTYPE html>"));
        for needle in [
            "<html",
            "</html>",
            "<head>",
            "</head>",
            "<body>",
            "</body>",
            "ui/style.css",
            "ui/uplot.min.css",
            "ui/uplot.iife.min.js",
            "ui/app.js",
            "viewport",
        ] {
            assert!(html.contains(needle), "index.html lacks {needle}");
        }
        // Balanced tags for the containers the router mounts into.
        for tag in ["div", "nav", "main", "header", "section", "template"] {
            let open = html.matches(&format!("<{tag}")).count();
            let close = html.matches(&format!("</{tag}>")).count();
            assert_eq!(open, close, "unbalanced <{tag}>");
        }
        let js =
            String::from_utf8(Assets::get("uplot.iife.min.js").unwrap().data.into_owned()).unwrap();
        assert!(js.contains("uPlot"));
        let css = String::from_utf8(Assets::get("style.css").unwrap().data.into_owned()).unwrap();
        assert!(css.contains("NOTICE") && css.contains("1.6.32"));
        assert!(css.contains("#0b0b0f") && css.contains("#0a84ff"));
    }
}
