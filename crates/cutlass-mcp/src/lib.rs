//! Headless Cutlass MCP server (stdio) for external agents.
//!
//! v1 exposes project lifecycle, validated wire-edit batches, composited
//! frame grabs, and timeline export so Cursor / Claude Code and similar hosts
//! can script Cutlass without the desktop UI. **Edits never go through a raw
//! engine door** — mutations lower into `cutlass-ai` validate →
//! `cutlass-engine` apply, the same path as the in-app agent. The engine lives
//! on a dedicated OS thread ([`host::EngineHost`]); async tool handlers
//! round-trip over a channel.

pub mod host;
pub mod server;

mod tools;
