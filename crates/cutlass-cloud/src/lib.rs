//! Cutlass cloud client (see `docs/cloud-roadmap.md`).
//!
//! One crate owns all backend/provider HTTP for the editor, shaped like
//! `cutlass-ai`: engine-free, blocking HTTP on worker threads, trait-based
//! so tests use scripted fakes. Three responsibilities:
//!
//! - [`dto`]: the request/response types for every `cutlass-backend` route —
//!   **the contract source of truth**. The backend consumes this crate as a
//!   git dependency and its contract tests fail CI on drift. Everything is
//!   unknown-field tolerant (additive-only `/v1` evolution: old clients keep
//!   working).
//! - [`client`] / [`stock`]: the anonymous half — asset catalogs and stock
//!   search need no account. [`stock::StockProvider`] has two
//!   implementations: backend-routed (anonymous, server holds the provider
//!   keys) and direct Pexels/Pixabay (BYOK stock keys — then even search
//!   skips the backend).
//! - [`download`] / [`cache`]: media files never transit the backend; the
//!   client downloads **directly from provider CDNs** into a quota-managed
//!   cache (LRU eviction, clear-cache action) with atomic tmp-then-rename
//!   writes, progress callbacks, and cancellation.
//! - [`auth`] / [`token_store`]: the account half — device-authorization
//!   sign-in (RFC 8628 through the system browser against the website's
//!   better-auth), JWT refresh, balance, ledger; tokens live in the OS
//!   keychain, never in a file. Billing UI lives on the website.
//!
//! **The routing rule** (BYOK-first): a user-configured provider key routes
//! the call direct to the provider; else a signed-in session takes the
//! managed path through the backend; else only the anonymous surface
//! (stock, catalogs) is available.
//!
//! Invariants carried from the rest of the app: network stays off the UI
//! thread (callers run this on workers); the editor never blocks on the
//! backend (catalog fetches are background work, failures degrade to the
//! Library placeholders); BYOK keys never transit our servers.

pub mod auth;
pub mod cache;
pub mod client;
pub mod download;
pub mod dto;
pub mod error;
pub mod generate;
pub mod stock;
pub mod token_store;

pub use client::CloudClient;
pub use error::CloudError;
pub use stock::{DirectStockProvider, StockProvider};

/// Default production backend base URL. Overridable via the `[account]`
/// table in `~/.cutlass/config.toml` (points at staging or a self-hosted
/// instance).
pub const DEFAULT_BASE_URL: &str = "https://api.cutlass.sh";

/// Default production auth base URL — the website, where better-auth
/// lives (`/api/auth/*`, the `/device` approval page, `/account`
/// billing). Overridable via `[account] auth_base_url`.
pub const DEFAULT_AUTH_BASE_URL: &str = "https://cutlass.sh";
