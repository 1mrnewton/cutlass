//! cutlass-render: turn a [`cutlass_models::Project`] timeline into composited
//! frames, and export those frames to a video file.
//!
//! Two layers sit here:
//!
//! - [`resolve`] is pure: it maps a project at a timeline instant to a [`Scene`]
//!   — the canvas plus an ordered stack of placed layers — without decoding or
//!   touching the GPU. All the geometry (canvas sizing, z-order, transforms,
//!   crop/mirror) lives here so it can be tested deterministically anywhere.
//! - [`Renderer`] realizes a [`Scene`]: it decodes each media frame with the
//!   platform codec, rasterizes text, and composites the stack on a `wgpu`
//!   device into an [`RgbaImage`]. `render_frame` ties the two together.
//!
//! The export path (`composite every frame 0..duration → encode`) builds on
//! `render_frame` and drives a [`cutlass_core::VideoEncoder`].

mod animation;
mod audio_dsp;
mod error;
mod export;
mod export_audio;
mod grade;
mod lut_pack;
mod motion_blur;
mod render;
mod resolve;
mod scene;
mod transactional_file;

pub use cutlass_compositor::FrameSink;
pub use cutlass_core::RgbaImage;
pub use error::RenderError;
pub use export::{
    ExportObserver, ExportSettings, PngSequenceEncoder, decode_png, encode_jpeg, encode_png,
    export, export_config, export_config_with, export_observed, export_to_file,
    export_to_file_observed,
};
pub use export_audio::{EXPORT_AUDIO_CHANNELS, EXPORT_AUDIO_RATE, ExportAudioMixer};
pub use lut_pack::{STARTER_LUT_SIZE, StarterLut, starter_lut_pack};
#[doc(hidden)]
pub use render::text_anim_bench;
pub use render::{FrameStats, GestureFrames, Renderer, SeekPolicy};
pub use resolve::{
    GESTURE_IDENTITY_TRANSFORM, GestureScenePartition, ResolveOverrides, canvas_size, resolve,
    resolve_gesture_partitions, resolve_with,
};
pub use scene::{LayerSource, Scene, SceneLayer, SizeSpec, TextAnimation};
