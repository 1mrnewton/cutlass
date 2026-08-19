//! Cutlass cloud client (see `docs/cloud-roadmap.md`).
//!
//! One crate owns all backend HTTP for the editor, shaped like
//! `cutlass-ai`: engine-free, blocking HTTP on worker threads, trait-based
//! so tests use scripted fakes. Responsibilities:
//!
//! - [`dto`]: the request/response types for every `cutlass-backend` route —
//!   **the contract source of truth**. The backend consumes this crate as a
//!   git dependency and its contract tests fail CI on drift. Everything is
//!   unknown-field tolerant (additive-only `/v1` evolution: old clients keep
//!   working).
//! - [`client`]: the anonymous half — asset catalogs and the update check
//!   need no account.
//! - [`download`] / [`cache`]: media files never transit the backend; the
//!   client downloads **directly from the CDN** into a quota-managed cache
//!   (LRU eviction, clear-cache action) with atomic tmp-then-rename writes,
//!   progress callbacks, and cancellation.
//!
//! Invariants carried from the rest of the app: network stays off the UI
//! thread (callers run this on workers); the editor never blocks on the
//! backend (catalog fetches are background work, failures degrade to the
//! Library placeholders).

pub mod cache;
pub mod client;
pub mod download;
pub mod dto;
pub mod error;

pub use client::CloudClient;
pub use error::CloudError;

/// Default production backend base URL. Overridable via the `[cloud]`
/// table in `~/.cutlass/config.toml` (points at staging or a self-hosted
/// instance).
pub const DEFAULT_BASE_URL: &str = "https://api.cutlass.sh";
