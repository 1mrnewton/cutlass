//! The harness for the open project.
//!
//! One ask becomes model rounds. Tool results go back to the model. Draft
//! writes are one session combo. Settings writes use the settings generation.
//! The desktop injects the player, media, speech, catalog, packages, export,
//! sandbox executor, and MCP transport. This crate does not open media, start
//! a download, or open a socket.

#![forbid(unsafe_code)]

mod approval;
mod draft;
mod r#loop;
mod mcp;
mod model;
mod sandbox;
mod settings_tool;
mod skill;

pub use approval::{ApprovalAction, ApprovalError, Approver, authorize, authorize_then};
pub use draft::{
    ApplyInput, ApplyMode, ApplyResult, DraftError, DraftQuery, DraftTools, FullSnapshot,
    ProjectSnapshot, QueryView, RangeSnapshot, SegmentSnapshot, TrackSnapshot, request_name,
};
pub use r#loop::{LoopError, ToolOutcome, ToolSink, Turn, cap_from_settings, run};
pub use mcp::{McpClient, McpError, McpTransport};
pub use model::{
    CloudModel, Completion, CompletionRequest, LocalModel, Message, Model, ModelError, ToolCall,
    ToolDefinition,
};
pub use sandbox::{
    LocalSandbox, OnlineSandbox, Sandbox, SandboxError, SandboxExecutor, SandboxPolicy,
    SandboxRoute, route_for,
};
pub use settings_tool::{SettingsTool, SettingsToolError, SettingsView};
pub use skill::{Skill, SkillError, SkillLibrary, SkillPlan, parse_skill_md};
