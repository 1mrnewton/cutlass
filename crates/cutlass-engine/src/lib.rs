//! Editing engine: project session, inverse-command undo/redo, frame cache.

mod action;
mod composite;
mod decoder_pool;
mod engine;
mod error;
mod export;
mod export_audio;
mod frame;
mod import;
mod preview;

pub use action::ApplyOutcome;
pub use engine::{ColorConvertPath, DEFAULT_CACHE_BUDGET_BYTES, Engine, EngineConfig};
pub use error::EngineError;
pub use export::{
    ExportProgress, ExportSettings, export_config_for, export_config_with, export_project,
    export_project_with, export_timeline, export_timeline_with,
};
pub use export_audio::EXPORT_AUDIO_RATE;
pub use frame::RgbaFrame;
pub use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
pub use cutlass_encoder::{ExportConfig, ExportStats};

use tracing::info;

pub fn init() {
    cutlass_cache::init();
    cutlass_probe::init();
    cutlass_decoder::init();
    cutlass_compositor::init();
    cutlass_encoder::init();
    info!("cutlass-engine ready");
}
