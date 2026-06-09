//! Editing engine: project session, inverse-command undo/redo, frame cache.

mod action;
mod composite;
mod decoder_pool;
mod engine;
mod error;
mod frame;
mod import;
mod preview;

pub use action::ApplyOutcome;
pub use engine::{DEFAULT_CACHE_BUDGET_BYTES, Engine, EngineConfig};
pub use error::EngineError;
pub use frame::RgbaFrame;
pub use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};

use tracing::info;

pub fn init() {
    cutlass_cache::init();
    cutlass_decoder::init();
    cutlass_compositor::init();
    info!("cutlass-engine ready");
}
