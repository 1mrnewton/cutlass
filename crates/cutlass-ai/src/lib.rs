//! AI agent foundation for Cutlass (v1-roadmap M3, `ai-agent-roadmap.md`).
//!
//! Phase 1 of the agent: the machine-callable surface over the command
//! layer, with no network anywhere.
//!
//! - [`wire`]: the LLM-facing JSON vocabulary — every timeline edit the
//!   agent may request, as flat tagged objects with times in seconds and
//!   ids as plain integers, plus the generated tool schemas.
//! - [`validate`]: lowering wire commands to real `cutlass_commands`
//!   commands against a project snapshot, with model-readable rejections.
//!   The whitelist lives here (and in the wire types themselves): no
//!   project commands, no phantom generators.
//! - [`describe`]: the compact project summary + editor context pushed
//!   into every prompt.
//! - [`provider`] / [`providers`]: the `ChatProvider` seam (blocking, tool
//!   calling, streamed text) with the generic OpenAI-compatible HTTP
//!   implementation (Ollama / llama.cpp / LM Studio / cloud gateways) and
//!   the deterministic `ScriptedProvider` test double.
//! - [`config`]: API-key resolution (`api_key_env` indirection). The config
//!   *file* is owned by `cutlass-settings`; keys never live in project files.
//! - [`extend`]: prompt-level extensibility — user/project rules, skills
//!   behind the read-only `read_skill` tool, and slash-command templates.
//!   Rules and skills shape *how* the closed vocabulary is used; they can
//!   never add mutation surface.
//!
//! Invariant: **AI proposes, the engine disposes.** Nothing in this crate
//! mutates a project; output is validated commands for the caller to apply
//! through normal dispatch (one history group per prompt, rollback on
//! abort).
//!
//! # Growing the vocabulary (the checklist)
//!
//! The tool schema is closed and versioned; engine commands join the
//! agent's vocabulary deliberately, never by accident. Every new
//! `EditCommand` lands with **all four** of:
//!
//! 1. **Wire DTO + validation** — a serde DTO in [`wire`] shaped for LLM
//!    ergonomics (times as fractional seconds, ids as plain integers,
//!    enums as lowercase strings) and a lowering arm in [`validate`] with
//!    a model-readable rejection for every way the call can be wrong.
//! 2. **Schema snapshot update** — re-bless `tests/snapshots/tools.json`
//!    (`BLESS_TOOL_SCHEMA=1 cargo test -p cutlass-ai`) and bump
//!    [`TOOL_SCHEMA_VERSION`] when the surface changes shape, so the
//!    prompt-visible schema only ever changes as a reviewed diff.
//! 3. **Action-log line** — a [`agent::describe_action`] arm rendering
//!    the command in editor language ("split clip 7 at 12.40s"); it is
//!    the transcript entry, the undo tooltip, and the eval assertion
//!    format, all at once.
//! 4. **One eval case** — a scripted-provider test in
//!    `tests/agent_eval.rs` driving the new command through the full
//!    loop against a real engine, asserting the final timeline, the
//!    action log, and single-undo behavior.
//!
//! This is how M2's keyframe commands and M4's effect commands join:
//! "the vocabulary grows for free" is this checklist, enforced in
//! review. A command missing any step is not in the vocabulary.

pub mod agent;
pub mod config;
pub mod describe;
pub mod extend;
pub mod provider;
pub mod providers;
pub mod validate;
pub mod wire;

pub use agent::{
    ActionLogEntry, AgentConfig, AgentEvent, EngineBridge, PromptOutcome, PromptStatus, run_prompt,
};
pub use describe::{EditorContext, ProjectSummary, summarize};
pub use extend::{
    AgentDir, AgentExtensions, MAX_RULES_BYTES, Skill, SlashCommand, bundled_skills, compose_rules,
    expand_slash_command, load_agent_dir, merge_skills,
};
pub use provider::Message;
pub use validate::{Rejection, validate};
pub use wire::{TOOL_SCHEMA_VERSION, ToolSpec, WireCommand, tool_specs};
