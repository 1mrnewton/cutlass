//! The GPU renderer: realize a [`Scene`] into a composited [`RgbaImage`].
//!
//! [`Renderer`] owns the expensive, reusable pieces — a `wgpu` device, the
//! compositor pipelines, a text rasterizer, and a per-media decoder cache — so
//! a single instance renders many frames (preview scrub, export) without
//! re-initializing the GPU or re-opening decoders.

mod api;
mod effects;
mod media_cache;
mod raster_fit;
mod realize;
mod text_anim;

/// Hot-path helpers exposed for Criterion benches — not a stable API.
#[doc(hidden)]
pub mod text_anim_bench {
    pub use super::text_anim::{ClusterDelta, cluster_deltas, place_clusters};
}

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;

use cutlass_compositor::{Compositor, GpuContext, RgbaImage};
use cutlass_core::VideoDecoder;
use cutlass_decoder::OutputMode;
use cutlass_models::MediaId;
use cutlass_shapes::PathRaster;
use cutlass_text::TextRenderer;

use media_cache::{CubeLutState, LottieState, StickerSequence};

/// A composited frame slower than this logs its stage breakdown at `info`
/// (default-visible): interactive preview budgets a few frames of latency,
/// and anything past this is worth attributing to decode vs GPU work.
pub(super) const SLOW_FRAME_LOG_MS: f64 = 150.0;

/// Per-stage timing of the most recent successful frame render.
///
/// Callers that adapt render *resolution* to cost (the preview quality
/// ladder) need the split, not the total: decode runs at the source's native
/// size no matter how small the output canvas is, so only
/// [`scaled_cost_ms`](Self::scaled_cost_ms) responds to rendering smaller.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Media decode time summed across layers (resolution-independent).
    pub decode_ms: f64,
    /// Text/shape/still realize time (raster caches, still decodes).
    pub raster_ms: f64,
    /// GPU composite + readback — scales with output pixels.
    pub composite_ms: f64,
}

impl FrameStats {
    /// The portion of the frame cost that shrinks with output resolution
    /// (composite + raster) — what a quality ladder can actually buy back.
    pub fn scaled_cost_ms(&self) -> f64 {
        self.raster_ms + self.composite_ms
    }

    /// Whole-frame cost (decode + raster + composite).
    pub fn total_ms(&self) -> f64 {
        self.decode_ms + self.raster_ms + self.composite_ms
    }
}

/// How media decoders are positioned when realizing a frame.
///
/// `Exact` is correctness (export, settled preview); `NearestSync` is the
/// scrub-latency escape hatch: on long-GOP sources an exact mid-GOP target
/// costs a keyframe-prefix walk of hundreds of decodes, where the nearest
/// sync frame costs one. Frames rendered under `NearestSync` may show
/// content up to a GOP *before* the requested time — callers own the
/// follow-up exact render and must never cache snapped output under an
/// exact key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeekPolicy {
    /// Decode the exact frame covering each layer's source time.
    #[default]
    Exact,
    /// Snap to the cheapest frame near the target: one decode from the
    /// sync point at/before it (or a short exact roll when the target is
    /// just ahead of the decoder's position).
    NearestSync,
}

/// Renders project frames on a headless (or shared) GPU.
pub struct Renderer {
    gpu: GpuContext,
    compositor: Compositor,
    text: TextRenderer,
    /// Pen-path rasterizer (memoized, like `text`). Parametric shapes never
    /// touch it — they realize as GPU SDF layers.
    paths: PathRaster,
    /// One open decoder per **on-screen use** of a media source, reused
    /// across frames. Decoders are stateful (seek + walk), so keeping them
    /// warm makes sequential export and nearby scrubbing cheap. Keyed by
    /// `(media, occurrence slot)` — the per-scene index of that media in the
    /// layer walk — because two simultaneously visible clips of the same
    /// file sit at *different* source times: sharing one decoder cursor
    /// would ping-pong it between them, paying a seek plus GOP-prefix
    /// re-decode per layer per frame. Slots are stable while the stack is
    /// (same walk order), so each clip keeps its warm decoder.
    decoders: HashMap<(MediaId, u32), Box<dyn VideoDecoder>>,
    /// Decode-once cache for still images: one straight-alpha RGBA bitmap per
    /// media source, reused for every frame the still is on screen. Bounded by
    /// the project's still count, with each entry capped at
    /// [`cutlass_decoder::image::MAX_DECODE_DIMENSION`] on the long side.
    stills: HashMap<MediaId, RgbaImage>,
    /// Decode-once cache for bundled stickers, keyed by catalog id: the whole
    /// frame sequence (a static sticker is one frame) plus per-frame delays.
    /// Bounded by the catalog (small, embedded assets); frame lookup on the
    /// hot path is O(frames) over the delay table, no decode.
    stickers: HashMap<String, StickerSequence>,
    /// File-backed Lottie animations, keyed by path: parsed composition +
    /// LRU of rasterized frames (capped-fps sampling, per-asset byte budget
    /// — never the pre-render-everything sticker strategy; see
    /// `docs/lottie-design.md`). Failed loads are remembered so a missing
    /// file logs once and draws nothing instead of re-probing every frame.
    lottie: HashMap<String, LottieState>,
    /// Monotonic per-scene stamp for the Lottie frame LRU: frames touched
    /// by the scene currently being composed are never evicted, so two
    /// clips of one asset can't alias mid-frame.
    lottie_stamp: u64,
    /// Parsed `.cube` LUTs, keyed by path. The GPU texture lives in the
    /// compositor's own cache (same key); this holds the CPU parse so a
    /// re-render never re-reads the file. Failed loads are remembered so a
    /// missing file logs once and grades nothing instead of re-probing
    /// every frame.
    luts: HashMap<String, CubeLutState>,
    /// Preferred decoder output mode. Apple and Windows start in
    /// [`OutputMode::Gpu`] so hardware-decoded surfaces (`CVPixelBuffer` /
    /// shared D3D11 NV12 textures) import into the compositor with no CPU
    /// copy; if a produced surface can't be imported (e.g. 10-bit/HDR, or a
    /// GPU without NV12 texture support), the renderer permanently falls back
    /// to [`OutputMode::Cpu`] and retries.
    decode_mode: OutputMode,
    /// Runtime-only substitute decode paths (preview proxies): when present
    /// (and [`use_proxies`](Self::use_proxies) holds), decoders for a media
    /// id open this file instead of the project's. Session state — never
    /// serialized, cleared when the session's media-id space changes
    /// (open/load/relink). Content must match the original frame-for-frame;
    /// only resolution/GOP may differ, so placement geometry (driven by the
    /// model's dimensions) stays valid and normalized UVs sample correctly.
    proxies: HashMap<MediaId, PathBuf>,
    /// Whether [`decode`](Self::decode) honors `proxies`. Preview leaves
    /// this on; full-quality paths sharing this renderer (the engine's
    /// export command) flip it off for the pass. Toggling drops the
    /// proxied media's open decoders so no cursor outlives its file.
    use_proxies: bool,
    /// When true, export-quality motion blur runs: clips with motion blur
    /// enabled and an animated transform are supersampled across the shutter.
    /// Interactive preview leaves this false (N× layer draws is too costly
    /// for scrub). Flipped on for the duration of [`crate::export_observed`].
    pub(crate) apply_motion_blur: bool,
    /// Stage timings of the last successful render (see [`FrameStats`]).
    last_stats: FrameStats,
}

/// Three fit-sized RGBA images for a zero-drift preview transform gesture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GestureFrames {
    pub below: RgbaImage,
    pub sprite: RgbaImage,
    pub above: Option<RgbaImage>,
}
