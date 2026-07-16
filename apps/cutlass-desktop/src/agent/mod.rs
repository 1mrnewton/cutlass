//! AI agent worker: runs prompts against a sandbox engine, then replays
//! the validated plan on the live engine as one undoable history group.
//!
//! Why a sandbox? The agent loop holds a conversation across network
//! waits, and the engine's history groups don't nest — an open group on
//! the live engine would swallow any user edit made while the model
//! thinks. Instead the prompt edits a throwaway [`Engine`] seeded with a
//! snapshot of the live project: tool calls really apply (so the model
//! sees created clip/track ids and the world it changed), and nothing
//! touches the user's timeline until the plan replays atomically via
//! [`WorkerHandle::agent_apply_plan`]. Replay re-validates every step
//! against the live project and remaps ids the sandbox allocated, so a
//! mid-prompt user edit can only fail the plan loudly — never corrupt it.
//!
//! With the dry-run toggle on (the default), the plan is parked here and
//! the chat panel shows an Apply / Discard card instead of auto-applying.

mod approval;
mod run;
mod sandbox;
#[cfg(test)]
mod tests;
mod tool_host;
mod transcript;
mod types;

pub use tool_host::DesktopToolHost;
pub use types::{AgentCreated, AgentHandle, AgentPlanStep, AgentWorker};

// Re-export submodule items so `agent::tests` (`use super::*`) keeps
// the same names as the former single-file module.
#[allow(unused_imports)]
pub(crate) use approval::*;
#[allow(unused_imports)]
pub(crate) use run::*;
#[allow(unused_imports)]
pub(crate) use sandbox::*;
#[allow(unused_imports)]
pub(crate) use tool_host::*;
#[allow(unused_imports)]
pub(crate) use transcript::*;
#[allow(unused_imports)]
pub(crate) use types::*;
