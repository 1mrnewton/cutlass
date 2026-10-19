//! Editing engine: project session, inverse-command undo/redo, frame cache.

// The native-decode build (mobile, or `--features native-decode`) compiles out
// the FFmpeg decode path, leaving its helpers (keyframe-index seek, swscale
// downscale, YUV pack/unpack cache) unused. They remain for the desktop default
// build, so only quiet them on the native build.
#![cfg_attr(engine_native_decode, allow(dead_code, unused_imports))]

mod action;
mod composite;
mod decoder_pool;
mod engine;
mod error;
// Export + audio export ride the FFmpeg encoder/decoder, so they're desktop-only.
// The native-decode (mobile) build is preview/render-only for now.
#[cfg(not(engine_native_decode))]
mod export;
#[cfg(not(engine_native_decode))]
mod export_audio;
mod frame;
mod generator_raster;
mod import;
mod preview;

pub use action::ApplyOutcome;
pub use composite::{
    PREVIEW_MAX_HEIGHT, anchor_canvas_position, composite_canvas_config, composite_canvas_size,
    content_uv, cropped_layer_placement, generator_layer_placement, layer_placement,
    position_preserving_center, preview_canvas_size, preview_scaled_dims, reposition_anchor,
};
pub use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
#[cfg(not(engine_native_decode))]
pub use cutlass_encoder::{ExportConfig, ExportStats};
pub use engine::{ColorConvertPath, DEFAULT_CACHE_BUDGET_BYTES, Engine, EngineConfig};
pub use error::EngineError;
#[cfg(not(engine_native_decode))]
pub use export::{
    ExportProgress, ExportSettings, export_config_for, export_config_with, export_project,
    export_project_with, export_timeline, export_timeline_with,
};
#[cfg(not(engine_native_decode))]
pub use export_audio::EXPORT_AUDIO_RATE;
pub use frame::RgbaFrame;
pub use generator_raster::system_font_families;

use tracing::info;

pub fn init() {
    cutlass_cache::init();
    cutlass_probe::init();
    cutlass_decoder::init();
    cutlass_compositor::init();
    cutlass_encoder::init();
    info!("cutlass-engine ready");
}
