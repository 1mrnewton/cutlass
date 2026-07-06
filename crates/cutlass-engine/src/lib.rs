//! Cutlass editing engine: structured commands applied to a project with
//! inverse-based undo/redo.
//!
//! UI gestures and the AI agent both emit [`Command`]s; [`Engine::apply`]
//! executes them against the timeline and records an inverse, so every edit is
//! undoable (and gestures group into one history entry). Preview frames and
//! file export run on the GPU `cutlass-render` pipeline and the native
//! `cutlass-encoder` — there is no CPU compositing in this crate.

mod action;
mod engine;
mod error;
mod import;

pub use action::ApplyOutcome;
pub use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
pub use cutlass_render::{FrameStats, SeekPolicy};
pub use engine::{Engine, EngineConfig};
pub use error::EngineError;
