//! The scene description a composite pass renders: a canvas plus an ordered
//! stack of placed layers.

use cutlass_core::{RgbaImage, VideoFrame};
use cutlass_shapes::{SdfShape, Stroke};

use crate::grade::ColorGrade;
use crate::lut::CubeLut;
use crate::passes::PassInstance;

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

/// Mask shape kind ids shared with WGSL (`mask.wgsl`).
pub mod mask_kind {
    pub const LINEAR: u32 = 0;
    pub const MIRROR: u32 = 1;
    pub const CIRCLE: u32 = 2;
    pub const RECTANGLE: u32 = 3;
    pub const HEART: u32 = 4;
    pub const STAR: u32 = 5;
}

/// GPU-ready mask parameters (no `cutlass-models` dependency).
///
/// Geometry uses the same fraction-of-layer units as the model: `center` is an
/// offset from the layer center, `size` scales the mask (`[1,1]` = full layer),
/// `rotation_rad` is about the mask center, and `roundness` rounds rectangle
/// corners only (`0`…`1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerMask {
    pub kind: u32,
    pub feather: f32,
    pub invert: u32,
    /// Offset from layer center, fraction of layer size per axis.
    pub center: [f32; 2],
    /// Mask size as a fraction of layer size (`[1, 1]` = legacy full-quad).
    pub size: [f32; 2],
    pub rotation_rad: f32,
    /// Rectangle corner rounding, `0` … `1` (ignored for other kinds).
    pub roundness: f32,
}

impl LayerMask {
    /// Mask with identity geometry (centered, full size, no rotation/roundness).
    pub const fn new(kind: u32) -> Self {
        Self {
            kind,
            feather: 0.0,
            invert: 0,
            center: [0.0, 0.0],
            size: [1.0, 1.0],
            rotation_rad: 0.0,
            roundness: 0.0,
        }
    }
}

/// GPU-ready chroma-key parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerChromaKey {
    pub rgb: [f32; 3],
    pub strength: f32,
    pub shadow: f32,
}

/// Drop shadow drawn from the layer's alpha (offset + blur), in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerShadow {
    pub rgba: [u8; 4],
    pub offset: [f32; 2],
    pub blur: f32,
}

/// Soft glow bloom drawn from the layer's alpha, in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerGlow {
    pub rgba: [u8; 4],
    pub radius: f32,
    /// Multiplier on silhouette alpha before blur; typical range `0..=4`.
    pub intensity: f32,
}

/// Hard outline / stroke around the layer's alpha silhouette, in canvas pixels.
///
/// Drawn as silhouette → box blur (`width`) → harden threshold, composited
/// after the layer content (outside stroke). Width or alpha 0 skips the pass.
/// Blur+threshold approximates morphological dilation; the effective width is
/// capped by the style blur iteration limit (see `run_style_blur`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerOutline {
    pub rgba: [u8; 4],
    pub width: f32,
}

/// Solid rounded-rect plate behind the layer, in canvas pixels.
///
/// Sized as the content placement plus `2 * padding` per axis (same center and
/// rotation), with SDF AA pad on the quad. Alpha 0 skips the pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerBackground {
    pub rgba: [u8; 4],
    pub padding: f32,
    pub radius: f32,
}

/// Resolved layer styles in canvas pixels, rendered from the layer's alpha.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerStyles {
    pub shadow: Option<LayerShadow>,
    pub glow: Option<LayerGlow>,
    pub outline: Option<LayerOutline>,
    pub background: Option<LayerBackground>,
}

impl LayerStyles {
    /// True iff no style block is present.
    pub fn is_empty(&self) -> bool {
        self.shadow.is_none()
            && self.glow.is_none()
            && self.outline.is_none()
            && self.background.is_none()
    }
}

/// How a layer's pixels combine with the accumulated canvas below it.
/// `Normal` is plain source-over (the fast path); every other mode runs
/// the dst-sampling blend pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Darken,
    Multiply,
    ColorBurn,
    Lighten,
    Screen,
    ColorDodge,
    Add,
    Overlay,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

impl BlendMode {
    /// WGSL `mode` uniform id (`Normal` = 0, then 1..=12 in declaration order).
    pub(crate) fn shader_id(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Darken => 1,
            BlendMode::Multiply => 2,
            BlendMode::ColorBurn => 3,
            BlendMode::Lighten => 4,
            BlendMode::Screen => 5,
            BlendMode::ColorDodge => 6,
            BlendMode::Add => 7,
            BlendMode::Overlay => 8,
            BlendMode::SoftLight => 9,
            BlendMode::HardLight => 10,
            BlendMode::Difference => 11,
            BlendMode::Exclusion => 12,
        }
    }
}

/// Per-layer mask and chroma-key state for the fx pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerEffects {
    pub mask: Option<LayerMask>,
    pub chroma_key: Option<LayerChromaKey>,
}

impl LayerEffects {
    pub const IDENTITY: Self = Self {
        mask: None,
        chroma_key: None,
    };

    pub fn is_identity(&self) -> bool {
        self.mask.is_none() && self.chroma_key.is_none()
    }
}

/// A `.cube` 3D LUT applied to a layer (or the composited canvas) after its
/// color grade. `key` is a stable identity for the parsed table (the source
/// file path); the compositor caches the uploaded 3D texture under it, so the
/// same LUT costs one upload no matter how many layers or frames use it.
#[derive(Clone, Copy)]
pub struct LayerLut<'a> {
    /// Cache identity for `lut` (its source path).
    pub key: &'a str,
    /// The parsed table (uploaded once per `key`).
    pub lut: &'a CubeLut,
    /// Blend of the looked-up result over the input, `0` … `1`.
    pub intensity: f32,
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

/// One glyph placed from a [`GlyphsLayer`] atlas.
///
/// `glyph` indexes into [`GlyphsLayer::glyphs`]. Placement is absolute canvas
/// pixels (the render crate folds clip transform + per-character animation
/// into these fields before submit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphInstance {
    /// Index into [`GlyphsLayer::glyphs`].
    pub glyph: u32,
    /// Content center in canvas pixels (+x right, +y down).
    pub center: [f32; 2],
    /// Pre-rotation content extent in canvas pixels.
    pub size: [f32; 2],
    /// Clockwise rotation about the center, in radians.
    pub rotation: f32,
    /// Instance opacity in `0.0..=1.0`; multiplied by the layer opacity.
    pub opacity: f32,
}

/// Per-character text drawn as instanced quads from a shared glyph atlas.
///
/// `atlas_key` is a stable identity for the glyph set (e.g. a hash of text +
/// style). Matching keys reuse the uploaded atlas; only the instance buffer
/// is rewritten per frame.
#[derive(Debug, Clone, Copy)]
pub struct GlyphsLayer<'a> {
    /// Cache identity for the packed atlas.
    pub atlas_key: u64,
    /// Straight-alpha cluster bitmaps (color already folded in). Zero-area
    /// entries are whitespace placeholders and are not packed.
    pub glyphs: &'a [RgbaImage],
    /// Instance placements referencing `glyphs` by index.
    pub instances: &'a [GlyphInstance],
}

/// The pixel source for a [`CompositeLayer`].
///
/// Frames are borrowed: the engine pulls a [`VideoFrame`] from the decoder for
/// the current tick and hands it to the compositor without copying the planes.
pub enum LayerContent<'a> {
    /// A decoded video frame (CPU planes or imported GPU surfaces).
    Frame(&'a VideoFrame),
    /// A pre-rasterized straight-alpha RGBA bitmap: text, pen-tool shape
    /// paths, stickers, or a decoded still. The compositor premultiplies it
    /// on upload so its anti-aliased edges blend cleanly.
    Rgba(&'a RgbaImage),
    /// A solid RGBA fill across the placed quad.
    Solid([u8; 4]),
    /// A parametric vector shape evaluated in the fragment shader.
    Sdf(SdfLayer),
    /// Per-character text as instanced atlas quads.
    Glyphs(GlyphsLayer<'a>),
}

/// One layer in bottom-to-top stacking order: content plus placement.
pub struct CompositeLayer<'a> {
    pub content: LayerContent<'a>,
    pub placement: LayerPlacement,
    /// Sampled UV rect `[u0, v0, u1, v1]` across the **visible picture**
    /// (`(0,0)`=top-left, `(1,1)`=bottom-right of the frame's visible region).
    /// A sub-rect crops; a reversed axis mirrors. Ignored by solid fills.
    pub uv: [f32; 4],
    /// Optional GPU effect chain applied after the base content is realized.
    pub effects: &'a [PassInstance<'a>],
    /// Mask/chroma-key state; identity uses the fast path pipelines.
    pub fx: LayerEffects,
    /// Resolved color grade for this layer; `None` is the identity fast path.
    pub color_grade: Option<ColorGrade>,
    /// `.cube` LUT applied after the grade; `None` is the identity fast path.
    /// Skipped while the layer is a transition side (matching effect chains,
    /// which also pause during the blend).
    pub lut: Option<LayerLut<'a>>,
    /// How this layer composites over the accumulated canvas. `Normal` keeps
    /// the direct src-over fast path; other modes force the offscreen blend pass.
    pub blend_mode: BlendMode,
    /// Shadow/glow/outline/background styles. Order on canvas: background →
    /// shadow → glow → content → outline. Non-empty styles force the offscreen path.
    pub styles: LayerStyles,
    /// When non-empty, the compositor draws the layer content once per
    /// placement into a cleared offscreen at weight `1/N` (additive), then
    /// composites that average like a normal premultiplied layer. Texture
    /// content stays shared across passes (transform-only motion blur).
    /// Styles / blend modes apply to the accumulated result.
    pub blur_passes: Vec<LayerPlacement>,
}

/// A layer, a canvas-wide pass, or a dual-source transition submitted to the compositor.
pub enum CompositorLayer<'a> {
    /// A standard layer (optionally with an effect chain).
    Layer(&'a CompositeLayer<'a>),
    /// Apply an effect chain, grade, and LUT to the current composited canvas.
    CanvasPass {
        effects: &'a [PassInstance<'a>],
        grade: Option<ColorGrade>,
        lut: Option<LayerLut<'a>>,
    },
    /// Blend two independently placed layers by transition progress.
    Transition {
        outgoing: &'a CompositeLayer<'a>,
        incoming: &'a CompositeLayer<'a>,
        transition_id: &'a str,
        progress: f32,
    },
}

impl<'a> CompositorLayer<'a> {
    pub fn layer(layer: &'a CompositeLayer<'a>) -> Self {
        Self::Layer(layer)
    }
}

impl<'a> CompositeLayer<'a> {
    /// A video-frame layer with no crop/mirror.
    pub fn frame(frame: &'a VideoFrame, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Frame(frame),
            placement,
            uv: FULL_UV,
            effects: &[],
            fx: LayerEffects::IDENTITY,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: LayerStyles::default(),
            blur_passes: Vec::new(),
        }
    }

    /// An RGBA bitmap layer (text/shape/sticker/still) with no crop/mirror.
    pub fn rgba(image: &'a RgbaImage, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Rgba(image),
            placement,
            uv: FULL_UV,
            effects: &[],
            fx: LayerEffects::IDENTITY,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: LayerStyles::default(),
            blur_passes: Vec::new(),
        }
    }

    /// A solid-color layer.
    pub fn solid(rgba: [u8; 4], placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Solid(rgba),
            placement,
            uv: FULL_UV,
            effects: &[],
            fx: LayerEffects::IDENTITY,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: LayerStyles::default(),
            blur_passes: Vec::new(),
        }
    }

    /// A parametric shape layer (GPU SDF; UV does not apply).
    pub fn sdf(shape: SdfLayer, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Sdf(shape),
            placement,
            uv: FULL_UV,
            effects: &[],
            fx: LayerEffects::IDENTITY,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: LayerStyles::default(),
            blur_passes: Vec::new(),
        }
    }

    /// Per-character text as instanced atlas quads.
    ///
    /// `placement` supplies the layer opacity multiplier (and is the grade/fx
    /// anchor); each glyph's on-canvas transform lives on the instances.
    pub fn glyphs(glyphs: GlyphsLayer<'a>, placement: LayerPlacement) -> Self {
        Self {
            content: LayerContent::Glyphs(glyphs),
            placement,
            uv: FULL_UV,
            effects: &[],
            fx: LayerEffects::IDENTITY,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: LayerStyles::default(),
            blur_passes: Vec::new(),
        }
    }

    /// Attach an effect chain (sampled at resolve time).
    pub fn with_effects(mut self, effects: &'a [PassInstance<'a>]) -> Self {
        self.effects = effects;
        self
    }

    /// Replace the sampled UV rect (crop / mirror).
    pub fn with_uv(mut self, uv: [f32; 4]) -> Self {
        self.uv = uv;
        self
    }

    /// Attach mask/chroma-key fx (routes to the fx pipelines when non-identity).
    pub fn with_fx(mut self, fx: LayerEffects) -> Self {
        self.fx = fx;
        self
    }

    /// Replace the per-layer color grade.
    pub fn with_grade(mut self, grade: ColorGrade) -> Self {
        self.color_grade = (!grade.is_identity()).then_some(grade);
        self
    }

    /// Attach a resolved color grade (filter preset + manual adjustments).
    pub fn with_color_grade(mut self, grade: Option<ColorGrade>) -> Self {
        self.color_grade = grade.filter(|g| !g.is_identity());
        self
    }

    /// Attach a `.cube` LUT (applied after the grade). Zero intensity is the
    /// identity fast path and drops the pass entirely.
    pub fn with_lut(mut self, lut: Option<LayerLut<'a>>) -> Self {
        self.lut = lut.filter(|l| l.intensity > 0.0);
        self
    }

    /// Set the per-layer blend mode (`Normal` keeps the direct fast path).
    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    /// Attach resolved shadow/glow/outline/background styles (empty keeps the
    /// direct fast path when blend/effects/LUT also allow it).
    pub fn with_styles(mut self, styles: LayerStyles) -> Self {
        self.styles = styles;
        self
    }

    /// Attach motion-blur supersample placements (empty keeps the single-draw
    /// path). Forces the offscreen accumulate path when non-empty.
    pub fn with_blur_passes(mut self, blur_passes: Vec<LayerPlacement>) -> Self {
        self.blur_passes = blur_passes;
        self
    }
}
