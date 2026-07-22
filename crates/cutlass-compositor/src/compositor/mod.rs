//! The composite pass: scene description in, RGBA canvas out.
//!
//! [`Compositor::render`] clears the canvas to the config background, then draws
//! each layer's placed quad bottom-to-top and reads the result back to an
//! [`RgbaImage`]. Three layer kinds share one placement model and target:
//!
//! - **video frames** — uploaded as planes and converted YUV→RGB on the GPU
//!   using the frame's own [`cutlass_core::ColorSpace`]; straight-alpha src-over.
//! - **RGBA bitmaps** — pre-rasterized generators (text/shapes/stickers) or
//!   stills; premultiplied on upload and composited premultiplied src-over so
//!   anti-aliased edges stay clean.
//! - **solid fills** — a fast no-texture path; straight-alpha src-over.

mod glyphs;
mod layers;
mod pipelines;
mod render;
mod upload;

use std::cell::RefCell;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use cutlass_core::RgbaImage;

use crate::effect_render::{OffscreenPool, PassRegistry};

use glyphs::CachedAtlas;

pub use glyphs::identity_instances;

/// Canvas pixel format. Plain (non-sRGB) `Unorm`: the YUV shader already emits
/// gamma-encoded R'G'B', so we store those bytes verbatim (display-ready SDR)
/// and read them straight back into PNG/export without an extra transfer.
pub(super) const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Caller-provided destination for a composited frame's pixels, so a render
/// can write its readback rows straight into storage the caller owns (a UI
/// framebuffer) instead of allocating an [`RgbaImage`] and copying again.
pub trait FrameSink {
    /// Called **once per successful render**, after the GPU work and buffer
    /// map succeeded, with the output dimensions. Must return exactly
    /// `width * height * 4` bytes; the compositor fills them with tightly
    /// packed row-major RGBA. A failed render never calls this.
    fn pixels(&mut self, width: u32, height: u32) -> &mut [u8];
}

/// The [`FrameSink`] behind the [`RgbaImage`]-returning render paths: it
/// allocates the image at the reported size and hands out its bytes.
#[derive(Default)]
pub struct ImageSink(Option<RgbaImage>);

impl ImageSink {
    /// The rendered image; `None` until a render succeeded into this sink.
    pub fn into_image(self) -> Option<RgbaImage> {
        self.0
    }
}

impl FrameSink for ImageSink {
    fn pixels(&mut self, width: u32, height: u32) -> &mut [u8] {
        let image = self.0.insert(RgbaImage::new(
            width,
            height,
            vec![0; (width as usize) * (height as usize) * 4],
        ));
        &mut image.pixels
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct YuvUniforms {
    linear: [f32; 4],
    trans_opacity: [f32; 4],
    uv_rect: [f32; 4],
    coeffs: [f32; 4],
    grade_adj0: [f32; 4],
    grade_adj1: [f32; 4],
    grade_adj2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SolidUniforms {
    color: [f32; 4],
    linear: [f32; 4],
    trans_opacity: [f32; 4],
    grade_adj0: [f32; 4],
    grade_adj1: [f32; 4],
    grade_adj2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RgbaUniforms {
    linear: [f32; 4],
    trans_opacity: [f32; 4],
    uv_rect: [f32; 4],
    grade_adj0: [f32; 4],
    grade_adj1: [f32; 4],
    grade_adj2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct FxEffectsUniform {
    /// mask_kind, mask_feather, mask_invert, mask_enabled.
    mask: [f32; 4],
    /// chroma r/g/b (normalized), chroma_enabled.
    chroma: [f32; 4],
    /// chroma_strength, chroma_shadow, pad, pad.
    chroma_params: [f32; 4],
    /// quad half-extents (x, y), mask rotation_rad (z), mask roundness (w).
    half: [f32; 4],
    /// mask center xy (fraction of layer size), mask size xy (fraction).
    mask_geo: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SdfUniforms {
    fill: [f32; 4],
    stroke_color: [f32; 4],
    linear: [f32; 4],
    /// Clip translation (x, y), layer opacity (z), stroke width px (w).
    trans_opacity: [f32; 4],
    /// Shape half-extents px (x, y), shape kind id (z), corner radius px (w).
    geo: [f32; 4],
    /// Star points (x), inner fraction (y); quad half-extents px (z, w).
    star: [f32; 4],
    grade_adj0: [f32; 4],
    grade_adj1: [f32; 4],
    grade_adj2: [f32; 4],
}

/// Which pipeline draws a built layer.
#[derive(Clone, Copy)]
pub(super) enum LayerPipeline {
    Yuv,
    YuvFx,
    Rgba,
    RgbaFx,
    Solid,
    Sdf,
    Glyphs,
}

/// Instanced draw payload (glyph atlas quads).
pub(super) struct InstanceDraw {
    pub buffer: wgpu::Buffer,
    pub count: u32,
}

/// GPU resources backing one layer for the duration of a render. The bind group
/// retains its textures/buffer, but we hold them too so the lifetimes are
/// unmistakable across the pass.
pub(super) struct LayerGpu {
    pipeline: LayerPipeline,
    bind_group: wgpu::BindGroup,
    _textures: Vec<wgpu::Texture>,
    _uniform: wgpu::Buffer,
    /// Platform keep-alive for zero-copy imported surfaces (e.g. the Apple
    /// `CVMetalTexture` wrappers a GPU frame's planes are mapped through). Held
    /// until the layer's GPU resources drop, i.e. after the pass is submitted
    /// and waited on. `None` for CPU-uploaded and generated layers.
    _keep_alive: Option<Box<dyn core::any::Any + Send>>,
    /// Instanced draw (glyphs). `None` for single-quad layers.
    instances: Option<InstanceDraw>,
}

/// Ceiling on cached LUT textures; crossing it clears the cache (a project
/// realistically uses a handful of LUTs, so this only guards runaway churn).
pub(super) const LUT_CACHE_MAX: usize = 32;

/// One uploaded `.cube` table: the 3D texture plus the sampling parameters
/// captured at upload time.
pub(super) struct LutTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
}

/// Canvas texture + readback buffer reused across renders while the size is
/// stable. Interactive preview (and export) renders the same size every
/// frame, so recreating these per render is pure allocation overhead on the
/// hot path; the pass clears the texture, so no state leaks between frames.
pub(super) struct TargetCache {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    /// Row pitch of `readback` (256-byte copy alignment).
    padded_bytes_per_row: u32,
}

/// A WGPU alpha-over compositor. Build once (pipelines are reused), render many.
pub struct Compositor {
    yuv_pipeline: wgpu::RenderPipeline,
    yuv_fx_pipeline: wgpu::RenderPipeline,
    rgba_pipeline: wgpu::RenderPipeline,
    rgba_fx_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    sdf_pipeline: wgpu::RenderPipeline,
    glyphs_pipeline: wgpu::RenderPipeline,
    yuv_layout: wgpu::BindGroupLayout,
    yuv_fx_layout: wgpu::BindGroupLayout,
    rgba_layout: wgpu::BindGroupLayout,
    rgba_fx_layout: wgpu::BindGroupLayout,
    solid_layout: wgpu::BindGroupLayout,
    sdf_layout: wgpu::BindGroupLayout,
    glyphs_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: Option<TargetCache>,
    pass_registry: PassRegistry,
    offscreen: Option<OffscreenPool>,
    /// Uploaded `.cube` 3D textures keyed by [`LayerLut::key`] (source path).
    /// LUTs are small (a 33³ RGBA8 table is ~140 KB), so the cache just
    /// resets past [`LUT_CACHE_MAX`] entries instead of tracking recency.
    lut_cache: HashMap<String, LutTexture>,
    /// Packed glyph atlases keyed by [`crate::layer::GlyphsLayer::atlas_key`].
    /// `RefCell` so [`Self::build_layer`] can stay `&self` while the render
    /// loop holds an immutable borrow of the canvas target.
    glyph_atlases: RefCell<HashMap<u64, CachedAtlas>>,
    /// Maps Apple `CVPixelBuffer` GPU surfaces into `wgpu` textures with no CPU
    /// copy. `None` when the device isn't a Metal device (e.g. a software
    /// adapter) — GPU frames then fall back to [`CompositorError::UnsupportedFormat`]
    /// and the caller can decode in CPU mode instead.
    #[cfg(target_vendor = "apple")]
    metal_import: Option<crate::metal_import::MetalSurfaceImporter>,
    /// Maps Windows shared D3D11 NV12 textures into `wgpu` textures with no
    /// CPU copy. `None` when the device isn't the D3D12 backend or lacks
    /// `TEXTURE_FORMAT_NV12` — GPU frames then fall back to
    /// [`CompositorError::UnsupportedFormat`] and the caller can decode in CPU
    /// mode instead.
    #[cfg(target_os = "windows")]
    dx12_import: Option<crate::dx12_import::Dx12SurfaceImporter>,
}
