//! Cutlass MCP server: project control for external agents over stdio.
//!
//! Exposes a Model Context Protocol (MCP) surface so Cursor, Claude Code,
//! and similar hosts can open/create `.cutlass` projects, inspect timelines,
//! and apply edits without embedding the editor UI.
//!
//! **Edits never go through a raw engine door.** Mutation tools will lower
//! into the existing validated wire-command pipeline (`cutlass-ai` validate
//! → `cutlass-engine` apply), the same path the in-app agent uses. This
//! crate owns transport, tool routing, and the host-facing contract; the
//! engine stays behind that gate.
//!
//! The engine lives on a dedicated OS thread ([`host::EngineHost`]) because
//! it is not safely shared across tokio workers. Async tool handlers
//! round-trip requests over a channel. This milestone covers project
//! lifecycle + media import; edit tools land next.

pub mod host;
pub mod server;

mod tools;
