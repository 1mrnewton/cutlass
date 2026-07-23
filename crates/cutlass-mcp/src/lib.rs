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
//! This milestone is the transport/server scaffold only — an rmcp stdio
//! server that starts, reports server info, and hosts a minimal tool
//! router. Project I/O and edit tools land in later milestones.

pub mod server;
