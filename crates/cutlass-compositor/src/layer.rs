//! The scene description a composite pass renders: a canvas plus an ordered
//! stack of placed layers.

use cutlass_core::{RgbaImage, VideoFrame};
use cutlass_shapes::{SdfShape, Stroke};

/// Canvas dimensions and background for one composite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositorConfig {
    pub width: u32,
    pub height: u32,
    /// Opaque background the canvas clears to before layers composite over it.
    pub background: [u8; 4],
}

impl CompositorConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            background: [0, 0, 0, 255],
        }
    }

    pub const fn with_background(mut self, background: [u8; 4]) -> Self {
        self.background = background;
        self
    }
}

/// Content UV rect covering the whole visible picture (no crop, no mirror).
pub const FULL_UV: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Where a layer lands on the canvas, in canvas pixels.
///
/// The compositor draws a quad of `size` centered on `center`, rotated by
/// `rotation` (clockwise, +y down), with content alpha scaled by `opacity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerPlacement {
    /// Content center in canvas pixels (+x right, +y down).
    pub center: [f32; 2],
    /// Pre-rotation content extent (width, height) in canvas pixels.
    pub size: [f32; 2],
    /// Clockwise rotation about the center, in radians. A frame's *container*
    /// rotation is added on top of this by the compositor.
    pub rotation: f32,
    /// Layer opacity in `0.0..=1.0`; multiplies the content's alpha.
    pub opacity: f32,
}

impl LayerPlacement {
    /// Stretch content across the whole canvas (the no-transform default).
    pub fn full_canvas(config: &CompositorConfig) -> Self {
        Self {
            center: [config.width as f32 / 2.0, config.height as f32 / 2.0],
            size: [config.width as f32, config.height as f32],
            rotation: 0.0,
            opacity: 1.0,
        }
    }
}

/// A parametric vector shape drawn as a signed-distance field by the shape
/// pipeline: no texture, no rasterization — geometry parameters ride in the
/// layer's uniform block, so animated shapes cost a uniform update per frame
/// and stay crisp at any scale.
///
/// The shape is evaluated in quad-local pixels centered on the placement.
/// The placed quad must be at least as large as the shape plus its stroke
/// overhang plus the anti-alias ramp (`stroke.width / 2 + 2px` per side), or
/// the ink clips at the quad edge; callers (the renderer's resolver) pad the
/// placement size accordingly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfLayer {
    /// Half-extents + shape parameters, in canvas pixels.
    pub shape: SdfShape,
    /// Straight-alpha fill color; alpha 0 draws no fill (stroke-only).
    pub fill: [u8; 4],
    /// Optional centered outline (width in canvas pixels).
    pub stroke: Option<Stroke>,
}

/// The pixel source for a [`CompositeLayer`].
///
/// Frames are borrowed: the engine pulls a [`VideoFrame`] from the decoder for
/// the current tick and hands it to the compositor without copying the planes.
pub enum LayerContent<'a> {
    /// A decoded video frame (CPU planes; GPU-surface import is a follow-up).
    Frame(&'a VideoFrame),
    /// A pre-rasterized straight-alpha RGBA bitmap: text, pen-tool shape
    /// paths, stickers, or a decoded still. The compositor premultiplies it
    /// on upload so its anti-aliased edges blend cleanly.
    Rgba(&'a RgbaImage),
    /// A solid RGBA fill across the placed quad.
    Solid([u8; 4]),
    /// A parametric vector shape evaluated in the fragment shader.
    Sdf(SdfLayer),
}

/// One layer in bottom-to-top stacking order: content plus placement.
pub struct CompositeLayer<'a> {
    pub content: LayerContent<'a>,
    pub placement: LayerPlacement,
    /// Sampled UV rect `[u0, v0, u1, v1]` across the **visible picture**
    /// (`(0,0)`=top-left, `(1,1)`=bottom-right of the frame's visible region).
    /// A sub-rect crops; a reversed axis mirrors. Ignored by solid fills.
    pub uv: [f32; 4],
}

impl<'a> CompositeLayer<'a> {
    /// A video-frame layer with no crop/mirror.
    pub fn frame(frame: &'a VideoFrame, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Frame(frame),
            placement,
            uv: FULL_UV,
        }
    }

    /// An RGBA bitmap layer (text/shape/sticker/still) with no crop/mirror.
    pub fn rgba(image: &'a RgbaImage, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Rgba(image),
            placement,
            uv: FULL_UV,
        }
    }

    /// A solid-color layer.
    pub fn solid(rgba: [u8; 4], placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Solid(rgba),
            placement,
            uv: FULL_UV,
        }
    }

    /// A parametric shape layer (GPU SDF; UV does not apply).
    pub fn sdf(shape: SdfLayer, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Sdf(shape),
            placement,
            uv: FULL_UV,
        }
    }

    /// Replace the sampled UV rect (crop / mirror).
    pub fn with_uv(mut self, uv: [f32; 4]) -> Self {
        self.uv = uv;
        self
    }
}
