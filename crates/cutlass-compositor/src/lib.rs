//! WGPU compositor for Cutlass: turns decoded [`cutlass_core::VideoFrame`]s and
//! generated layers into a composited RGBA canvas for preview and export.
//!
//! The decoder hands frames across the [`cutlass_core`] seam as either CPU
//! planes (NV12 / I420 / packed RGB) or, on the fast path, a zero-copy GPU
//! surface. The compositor uploads those planes, converts YUV→RGB on the GPU
//! using the frame's own [`cutlass_core::ColorSpace`] (so BT.601/709/2020 and
//! limited/full range are honored rather than assumed), places each layer on the
//! canvas, and blends them bottom-to-top with src-over alpha. Generated content
//! (text, shapes, stickers) arrives pre-rasterized as a straight-alpha
//! [`RgbaImage`] and composites through the same placement model.
//!
//! [`GpuContext::new_headless_blocking`] is the entry point for tests and export;
//! the desktop UI will instead share its device via [`GpuContext::from_parts`].
//!
//! ## Scope today
//!
//! - Inputs: 8-bit **NV12** and **I420** CPU video frames, straight-alpha
//!   **RGBA** bitmap layers (text/shape/sticker generators, stills), solid
//!   fills, and zero-copy NV12 GPU surfaces — on Apple, `CVPixelBuffer`s
//!   mapped via `CVMetalTextureCache` (see [`metal_import`]); on Windows,
//!   shared D3D11 textures opened on the D3D12 device as `wgpu` NV12 plane
//!   views (see [`dx12_import`]).
//! - 10-bit/HDR (P010, PQ/HLG tone-mapping), packed-RGB *video frames*, and
//!   GPU-surface import on Android (`AHardwareBuffer`) are recognized but not
//!   yet handled — they surface as [`CompositorError::UnsupportedFormat`].

mod compositor;
#[cfg(target_os = "windows")]
mod dx12_import;
mod error;
mod gpu;
mod layer;
#[cfg(target_vendor = "apple")]
mod metal_import;

pub use compositor::{Compositor, FrameSink, ImageSink};
pub use cutlass_core::RgbaImage;
pub use cutlass_shapes::{SdfParams, SdfShape, Stroke};
pub use error::CompositorError;
pub use gpu::GpuContext;
pub use layer::{
    CompositeLayer, CompositorConfig, FULL_UV, LayerContent, LayerPlacement, SdfLayer,
};
