//! The intermediate scene description a render pass consumes.
//!
//! [`resolve`](crate::resolve) turns a [`Project`](cutlass_models::Project) at a
//! timeline instant into a [`Scene`]: the canvas plus an ordered, bottom-to-top
//! stack of placed layers. A `Scene` is a pure value — it names *what* to draw
//! (which media frame, which text, which fill) and *where*, but holds no decoded
//! pixels and touches no GPU. That split keeps the geometry (canvas sizing,
//! z-order, transforms, crop) deterministic and unit-testable without a device,
//! while [`Renderer`](crate::Renderer) does the decode + rasterize + composite.

use cutlass_models::MediaId;
use cutlass_text::TextStyle;

pub use cutlass_core::RationalTime;

/// A canvas plus the ordered layer stack to composite for one timeline instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Opaque background the canvas clears to before layers composite over it.
    pub background: [u8; 4],
    /// Layers in bottom-to-top stacking order (index 0 draws first).
    pub layers: Vec<SceneLayer>,
}

impl Scene {
    /// An empty canvas of `width`×`height` over `background` (no layers).
    pub fn empty(width: u32, height: u32, background: [u8; 4]) -> Self {
        Self {
            width,
            height,
            background,
            layers: Vec::new(),
        }
    }
}

/// One placed layer: a pixel source plus where it lands on the canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneLayer {
    /// What to draw.
    pub source: LayerSource,
    /// Center of the placed quad in canvas pixels (+x right, +y down).
    pub center: [f32; 2],
    /// On-canvas extent of the content.
    pub size: SizeSpec,
    /// Clockwise rotation about `center`, in radians.
    pub rotation: f32,
    /// Layer opacity in `0.0..=1.0`; multiplies the content's alpha.
    pub opacity: f32,
    /// Sampled UV rect `[u0, v0, u1, v1]` across the visible picture. A sub-rect
    /// crops; a reversed axis mirrors. Ignored by solid fills.
    pub uv: [f32; 4],
}

/// How a layer's on-canvas size is determined.
///
/// Most content has a size the resolver can compute up front (media aspect-fit,
/// shapes, solids). Text is the exception: its pixel extent isn't known until
/// it is shaped and rasterized, so the resolver defers it to the renderer as a
/// multiplier on the rasterized bitmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeSpec {
    /// A known on-canvas size in pixels (scale already folded in).
    Fixed([f32; 2]),
    /// Multiply the rasterized content's pixel size by this factor (text).
    BitmapScaled(f32),
}

/// The pixel source for a [`SceneLayer`].
#[derive(Debug, Clone, PartialEq)]
pub enum LayerSource {
    /// A decoded video frame: decode `media` at `source_time` and place it.
    Media {
        media: MediaId,
        source_time: RationalTime,
    },
    /// A rasterized text run.
    Text { content: String, style: TextStyle },
    /// A solid RGBA fill across the placed quad.
    Solid([u8; 4]),
}
