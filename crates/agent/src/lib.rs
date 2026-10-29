//! The harness for the open project.
//!
//! One ask becomes model rounds. Tool results go back to the model. Draft
//! writes are one session combo. Settings writes use the settings generation.
//! The desktop injects the player, media, speech, catalog, packages, export,
//! sandbox executor, and MCP transport. This crate does not open media, start
//! a download, or open a socket.

#![forbid(unsafe_code)]

mod approval;
mod r#loop;
mod model;

pub use approval::{ApprovalAction, ApprovalError, Approver, authorize, authorize_then};
pub use r#loop::{LoopError, ToolOutcome, ToolSink, Turn, cap_from_settings, run};
pub use model::{
    CloudModel, Completion, CompletionRequest, LocalModel, Message, Model, ModelError, ToolCall,
    ToolDefinition,
};
