// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The crate's error type. Handlers return it and it renders as
//! `{"error": "..."}` with the matching HTTP status.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Everything that can go wrong serving the dashboard.
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    /// A run, config, checkpoint or clip that does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// A malformed id, path or body.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// The operation contradicts the run's state (delete while running…).
    #[error("conflict: {0}")]
    Conflict(String),
    /// File system.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON on disk or in a body.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// A config that does not parse.
    #[error("config: {0}")]
    Toml(#[from] toml::de::Error),
    /// Rendering a config.
    #[error("config: {0}")]
    TomlSer(#[from] toml::ser::Error),
    /// The supervisor could not do what was asked.
    #[error("supervisor: {0}")]
    Supervisor(String),
}

impl WebError {
    /// HTTP status for the variant.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) | Self::Toml(_) => StatusCode::BAD_REQUEST,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Io(e) if e.kind() == std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            Self::Io(_) | Self::Json(_) | Self::TomlSer(_) | Self::Supervisor(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = self.status();
        (status, axum::Json(json!({ "error": self.to_string() }))).into_response()
    }
}

/// `Result` with this crate's error.
pub type Result<T> = std::result::Result<T, WebError>;
