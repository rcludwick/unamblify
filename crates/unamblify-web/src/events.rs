// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The in-process event bus behind both SSE endpoints. The runs-dir
//! poller ([`crate::watch`]) and the [`crate::supervisor`] publish;
//! `/api/runs/{id}/events` forwards the events of one run and `/api/events`
//! forwards the run-list and capture events.

use serde::Serialize;
use serde_json::Value;
use tokio::sync::broadcast;

/// Event kinds. Per-run streams carry `metric | status | log | sys |
/// checkpoint`; the list stream carries `runs | capture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// New `metrics.jsonl` rows (non-`sys/*` keys), coalesced.
    Metric,
    /// `status.json` changed.
    Status,
    /// New `log.jsonl` rows, coalesced.
    Log,
    /// New `sys/*` metric rows, coalesced.
    Sys,
    /// A new checkpoint directory appeared.
    Checkpoint,
    /// The run list changed (any run's status).
    Runs,
    /// A capture mode's `status.json` or supervision changed.
    Capture,
}

impl Kind {
    /// SSE `event:` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metric => "metric",
            Self::Status => "status",
            Self::Log => "log",
            Self::Sys => "sys",
            Self::Checkpoint => "checkpoint",
            Self::Runs => "runs",
            Self::Capture => "capture",
        }
    }
}

/// One bus message.
#[derive(Debug, Clone, Serialize)]
pub struct Event {
    /// Run id for per-run kinds; `None` for list-level kinds.
    pub run: Option<String>,
    /// Kind.
    pub kind: Kind,
    /// Payload (shape depends on `kind`).
    pub data: Value,
}

impl Event {
    /// A per-run event.
    #[must_use]
    pub fn for_run(run: &str, kind: Kind, data: Value) -> Self {
        Self {
            run: Some(run.to_owned()),
            kind,
            data,
        }
    }

    /// A list-level event.
    #[must_use]
    pub fn global(kind: Kind, data: Value) -> Self {
        Self {
            run: None,
            kind,
            data,
        }
    }
}

/// Broadcast hub.
#[derive(Debug)]
pub struct Hub {
    tx: broadcast::Sender<Event>,
}

impl Default for Hub {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl Hub {
    /// A hub whose slowest subscriber may lag `capacity` events before it
    /// is told it missed some.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish; silently dropped when nobody listens.
    pub fn publish(&self, ev: Event) {
        let _ = self.tx.send(ev);
    }

    /// Subscribe to everything.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    /// Live subscriber count.
    #[must_use]
    pub fn listeners(&self) -> usize {
        self.tx.receiver_count()
    }
}
