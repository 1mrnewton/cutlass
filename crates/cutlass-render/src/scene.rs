//! The intermediate scene description a render pass consumes.
//!
//! [`resolve`](crate::resolve) turns a [`Project`](cutlass_models::Project) at a
//! timeline instant into a [`Scene`]: the canvas plus an ordered, bottom-to-top
//! stack of placed layers. A `Scene` is a pure value — it names *what* to draw
//! (which media frame, which text, which fill) and *where*, but holds no decoded
//! pixels and touches no GPU. That split keeps the geometry (canvas sizing,
//! z-order, transforms, crop) deterministic and unit-testable without a device,
//! while [`Renderer`](crate::Renderer) does the decode + rasterize + composite.

use cutlass_compositor::ColorGrade;
use cutlass_models::{AnimationSlot, BlendMode, ClipId, MaskKind, MediaId};
use cutlass_shapes::{BezierPath, SdfParams, Stroke};
use cutlass_text::{TextAlign, TextStyle, TextVerticalAlign};

/// Sampled per-character text animation attached to a [`LayerSource::Text`].
///
/// Pure resolve-time data: the preset id, which slot it came from, a
/// normalized progress / phase in `0…1`, and the clip's tunable knobs.
/// Realize expands this into per-cluster placement deltas — the Scene stays
/// GPU-free and unit-testable.
#[derive(Debug, Clone, PartialEq)]
pub struct TextAnimation {
    pub id: String,
    pub slot: AnimationSlot,
    /// Eased entrance/exit progress, or combo phase (both in `0…1`).
    pub t: f32,
    /// Magnitude multiplier (`1` = catalog feel).
    pub intensity: f32,
    /// Per-character stagger stretch (`1` = catalog feel).
    pub stagger: f32,
}

pub use cutlass_core::RationalTime;

/// One sampled GPU effect pass attached to a clip at resolve time.
///
/// `params` is the catalog-ordered flattened value list for the GPU uniform
/// upload: each scalar contributes 1 float, each vec2 contributes 2, and each
/// color contributes 4 floats in `0…1` (encoded 8-bit ÷ 255). Slot order
/// matches [`cutlass_models::EffectSpec::params`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPass {
    pub id: String,
    pub params: Vec<f32>,
}

/// A canvas plus the ordered layer stack to composite for one timeline instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Canvas clear color before layers composite. Alpha 0 is supported for
    /// gesture sprite/foreground passes that stack over an opaque backdrop.
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

    /// Uniformly scale the whole scene — canvas and every layer's geometry —
    /// by `factor`. Content keeps its composition exactly (same relative
    /// placement, crop, rotation); only the pixel density changes. This is
    /// how preview renders at fit-to-view size and export renders at a
    /// non-native resolution without touching the resolver.
    ///
    /// Degenerate factors (non-finite or ≤ 0) are ignored.
    pub fn scale(&mut self, factor: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        self.width = scaled_dim(self.width, factor);
        self.height = scaled_dim(self.height, factor);
        for layer in &mut self.layers {
            layer.center = [layer.center[0] * factor, layer.center[1] * factor];
            layer.size = match layer.size {
                SizeSpec::Fixed([w, h]) => SizeSpec::Fixed([w * factor, h * factor]),
                // Text / path bitmaps rasterize at their reference resolution
                // and ride the quad; scaling the multiplier scales the quad.
                SizeSpec::BitmapScaled([sx, sy]) => {
                    SizeSpec::BitmapScaled([sx * factor, sy * factor])
                }
            };
            match &mut layer.source {
                // SDF stroke width and AA pad are in canvas pixels.
                LayerSource::Shape { stroke, pad, .. } => {
                    *pad *= factor;
                    if let Some(stroke) = stroke {
                        stroke.width *= factor;
                    }
                }
                // Path strokes live in path-local pixels folded into the
                // raster, so scaling the raster factor scales them too.
                LayerSource::PathShape { raster_scale, .. } => *raster_scale *= factor,
                LayerSource::CanvasPass
                | LayerSource::Media { .. }
                | LayerSource::Still { .. }
                | LayerSource::Sticker { .. }
                | LayerSource::Lottie { .. }
                | LayerSource::Text { .. }
                | LayerSource::Solid(_)
                | LayerSource::Transition { .. } => {}
            }
        }
    }

    /// Scale the scene to fit within `max_width`×`max_height`, preserving
    /// aspect and never upscaling. The result has no letterbox: the canvas
    /// itself shrinks to the fitted box.
    pub fn fit_within(&mut self, max_width: u32, max_height: u32) {
        if self.width == 0 || self.height == 0 || max_width == 0 || max_height == 0 {
            return;
        }
        let factor = (max_width as f32 / self.width as f32)
            .min(max_height as f32 / self.height as f32)
            .min(1.0);
        if factor < 1.0 {
            self.scale(factor);
        }
    }

    /// Scale the scene to exactly `width`×`height`: uniform aspect-preserving
    /// scale (up or down), content centered, any aspect mismatch letterboxed
    /// with the scene background. This is the export path for a requested
    /// output resolution.
    pub fn fit_into(&mut self, width: u32, height: u32) {
        if self.width == 0 || self.height == 0 || width == 0 || height == 0 {
            return;
        }
        let (cw, ch) = (self.width as f32, self.height as f32);
        let factor = (width as f32 / cw).min(height as f32 / ch);
        self.scale(factor);
        let dx = (width as f32 - cw * factor) * 0.5;
        let dy = (height as f32 - ch * factor) * 0.5;
        for layer in &mut self.layers {
            layer.center = [layer.center[0] + dx, layer.center[1] + dy];
        }
        self.width = width;
        self.height = height;
    }
}

/// A scaled canvas dimension: rounded, never collapsing to zero.
fn scaled_dim(dim: u32, factor: f32) -> u32 {
    ((dim as f32 * factor).round() as u32).max(1)
}

/// One placed layer: a pixel source plus where it lands on the canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneLayer {
    /// Originating timeline clip, when this layer maps 1:1 to one clip.
    /// `None` for transition composites and other multi-source layers.
    pub clip: Option<ClipId>,
    /// What to draw.
    pub source: LayerSource,
    /// Canvas position of the content's anchor point (+x right, +y down) —
    /// the pivot `rotation` spins about, and what the clip transform's
    /// `position` places. Equals the placed quad's center for the default
    /// centered `anchor_point`.
    pub center: [f32; 2],
    /// Pivot within the content bounds, normalized to the placed size
    /// (`[0.5, 0.5]` = content center). The renderer derives the quad center
    /// from `center` once the final pixel size is known — deferred because
    /// text/path bitmaps only get a size after rasterization.
    pub anchor_point: [f32; 2],
    /// On-canvas extent of the content.
    pub size: SizeSpec,
    /// Clockwise rotation about the anchor (`center`), in radians.
    pub rotation: f32,
    /// Layer opacity in `0.0..=1.0`; multiplies the content's alpha.
    pub opacity: f32,
    /// Sampled UV rect `[u0, v0, u1, v1]` across the visible picture. A sub-rect
    /// crops; a reversed axis mirrors. Ignored by solid fills.
    pub uv: [f32; 4],
    /// GPU effect chain sampled at clip-local tick (empty when none).
    pub effects: Vec<ResolvedPass>,
    /// Shaped alpha mask (media clips only).
    pub mask: Option<SceneMask>,
    /// Green-screen keying (media clips only).
    pub chroma_key: Option<SceneChromaKey>,
    /// Resolved color grade (filter preset + manual adjustments); `None` when
    /// the clip's look is identity.
    pub color_grade: Option<ColorGrade>,
    /// `.cube` 3D LUT applied after the grade; `None` when the clip has none.
    /// File-backed: the renderer parses and uploads the table on first use
    /// and skips missing/unparseable files gracefully.
    pub lut: Option<SceneLut>,
    /// How this layer composites over the stack below (`Normal` = source-over).
    /// Canvas-wide passes and transition wrappers stay `Normal`.
    pub blend_mode: BlendMode,
    /// Layer-quad styles (shadow/glow/outline/background) sampled at the
    /// clip-local tick into canvas pixels. `None` when the clip has no style
    /// blocks — the compositor fast path. Canvas-wide passes and transition
    /// wrappers stay `None`.
    pub styles: Option<SceneStyles>,
}

/// Mask values sampled at a clip-local tick.
///
/// Geometry (`center`, `size`, `rotation_rad`, `roundness`) stays in the same
/// fraction-of-layer units as the model — the compositor consumes those
/// fractions directly; no canvas-pixel conversion happens here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneMask {
    pub kind: MaskKind,
    pub feather: f32,
    /// Mask center offset from the layer center, as a fraction of layer size.
    pub center: [f32; 2],
    /// Mask size as a fraction of layer size (`[1,1]` = cover the layer).
    pub size: [f32; 2],
    /// Mask rotation in radians (converted from model degrees at resolve).
    pub rotation_rad: f32,
    /// Rectangle corner rounding, `0` … `1`.
    pub roundness: f32,
    pub invert: bool,
}

/// Chroma-key values sampled at a clip-local tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneChromaKey {
    pub rgb: [u8; 3],
    pub strength: f32,
    pub shadow: f32,
}

/// A file-backed `.cube` LUT reference on a [`SceneLayer`].
#[derive(Debug, Clone, PartialEq)]
pub struct SceneLut {
    /// Absolute path to the `.cube` file.
    pub path: String,
    /// Blend of the looked-up result over the original, `0` … `1`.
    pub intensity: f32,
}

/// Drop shadow drawn from the layer's alpha (offset + blur), in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneShadow {
    pub rgba: [u8; 4],
    pub offset: [f32; 2],
    pub blur: f32,
}

/// Soft glow bloom drawn from the layer's alpha, in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneGlow {
    pub rgba: [u8; 4],
    pub radius: f32,
    pub intensity: f32,
}

/// Hard outline / stroke around the layer's alpha silhouette, in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneOutline {
    pub rgba: [u8; 4],
    pub width: f32,
}

/// Solid plate behind the layer (padded AABB of the alpha bounds), in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneBackground {
    pub rgba: [u8; 4],
    pub padding: f32,
    pub radius: f32,
}

/// Layer-quad styles sampled at a clip-local tick into canvas pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneStyles {
    pub shadow: Option<SceneShadow>,
    pub glow: Option<SceneGlow>,
    pub outline: Option<SceneOutline>,
    pub background: Option<SceneBackground>,
}

impl SceneStyles {
    /// True iff no style block is present.
    pub fn is_empty(&self) -> bool {
        self.shadow.is_none()
            && self.glow.is_none()
            && self.outline.is_none()
            && self.background.is_none()
    }
}

impl SceneLayer {
    /// The placed quad's center for a final pixel `size`: offset the anchor
    /// by the (rotated) anchor→center vector. Identity for center anchors.
    pub fn quad_center(&self, size: [f32; 2]) -> [f32; 2] {
        let to_center = [
            (0.5 - self.anchor_point[0]) * size[0],
            (0.5 - self.anchor_point[1]) * size[1],
        ];
        if to_center == [0.0, 0.0] {
            return self.center;
        }
        let (sin, cos) = self.rotation.sin_cos();
        [
            self.center[0] + to_center[0] * cos - to_center[1] * sin,
            self.center[1] + to_center[0] * sin + to_center[1] * cos,
        ]
    }

    /// Place an ink-tight text bitmap at its requested canvas alignment while
    /// retaining the clip transform's position and anchor offsets.
    pub fn text_quad_center(
        &self,
        style: &TextStyle,
        size: [f32; 2],
        canvas: [f32; 2],
    ) -> [f32; 2] {
        let mut center = self.quad_center(size);
        center[0] += match style.align {
            TextAlign::Left => (size[0] - canvas[0]) * 0.5,
            TextAlign::Center => 0.0,
            TextAlign::Right => (canvas[0] - size[0]) * 0.5,
        };
        center[1] += match style.vertical_align {
            TextVerticalAlign::Top => (size[1] - canvas[1]) * 0.5,
            TextVerticalAlign::Middle => 0.0,
            TextVerticalAlign::Bottom => (canvas[1] - size[1]) * 0.5,
        };
        center
    }
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
    /// Multiply the rasterized content's pixel size by these per-axis factors
    /// (text / path shapes). Uniform scales use equal components.
    BitmapScaled([f32; 2]),
}

/// The pixel source for a [`SceneLayer`].
#[derive(Debug, Clone, PartialEq)]
pub enum LayerSource {
    /// A decoded video frame: decode `media` at `source_time` and place it.
    Media {
        media: MediaId,
        source_time: RationalTime,
    },
    /// A still image: decode `media`'s single frame once (the renderer
    /// caches it) and place it for the clip's whole extent.
    Still { media: MediaId },
    /// A bundled sticker (static or animated): the renderer decodes the
    /// catalog asset once into a frame sequence and picks the frame at
    /// `local_time`, looping.
    Sticker {
        /// Catalog id (see [`cutlass_models::sticker_catalog`]).
        asset: String,
        /// Seconds since the clip's timeline start.
        local_time: f64,
    },
    /// A file-backed Lottie animation: the renderer parses `path` once and
    /// rasterizes the capped-fps frame at `local_time` on demand (LRU-cached,
    /// looping). A missing/unparseable file draws nothing.
    Lottie {
        /// Absolute path to the `.json` on disk.
        path: String,
        /// Seconds since the clip's timeline start.
        local_time: f64,
    },
    /// A text run. When `animation` is `Some`, realize draws per-character
    /// instanced glyphs; otherwise the whole run is rasterized as one bitmap.
    Text {
        content: String,
        style: TextStyle,
        animation: Option<TextAnimation>,
    },
    /// A solid RGBA fill across the placed quad.
    Solid([u8; 4]),
    /// Apply this layer's effect chain and color grade to the current canvas.
    ///
    /// Lane-level effect/filter/adjustment generator bars use this geometry-free
    /// marker to process everything already drawn below their track.
    CanvasPass,
    /// A parametric vector shape, every animatable parameter already sampled
    /// at this instant (canvas pixels). Realized as a GPU SDF layer: the
    /// layer's `size` is the *padded quad* (shape + stroke overhang + AA);
    /// `params` + the fill/stroke style travel to the fragment shader as
    /// uniforms.
    Shape {
        /// Size-free shape parameters (the shape's pixel box is derived from
        /// the layer's padded `size` minus `pad`).
        params: SdfParams,
        /// Straight-alpha fill; alpha 0 means stroke-only.
        fill: [u8; 4],
        /// Optional centered outline (width in canvas pixels).
        stroke: Option<Stroke>,
        /// Padding per side between the shape box and the placed quad
        /// (stroke overhang + AA margin), in canvas pixels.
        pad: f32,
    },
    /// A pen-tool bezier path (shape-local pixels), rasterized on the CPU at
    /// `raster_scale` and composited as a bitmap like text.
    PathShape {
        path: BezierPath,
        fill: [u8; 4],
        stroke: Option<Stroke>,
        /// Path-local px → canvas px factor folded into the raster
        /// (reference scale; the clip's animated transform scale rides the
        /// quad via [`SizeSpec::BitmapScaled`]).
        raster_scale: f32,
    },
    /// A track transition between two abutting clips, sampled at `progress`.
    Transition {
        outgoing: Box<SceneLayer>,
        incoming: Box<SceneLayer>,
        transition_id: String,
        /// `0.0` = fully outgoing, `1.0` = fully incoming.
        progress: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_core::Rational;

    fn media_layer(center: [f32; 2], size: [f32; 2]) -> SceneLayer {
        SceneLayer {
            clip: None,
            source: LayerSource::Media {
                media: MediaId::from_raw(1),
                source_time: RationalTime::new(0, Rational::FPS_30),
            },
            center,
            anchor_point: [0.5, 0.5],
            size: SizeSpec::Fixed(size),
            rotation: 0.5,
            opacity: 0.8,
            uv: [0.1, 0.2, 0.9, 0.8],
            effects: Vec::new(),
            mask: None,
            chroma_key: None,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: None,
        }
    }

    fn shape_layer() -> SceneLayer {
        SceneLayer {
            clip: None,
            source: LayerSource::Shape {
                params: SdfParams::Ellipse,
                fill: [255, 0, 0, 255],
                stroke: Some(Stroke {
                    rgba: [0, 0, 0, 255],
                    width: 8.0,
                }),
                pad: 6.0,
            },
            center: [100.0, 100.0],
            anchor_point: [0.5, 0.5],
            size: SizeSpec::Fixed([212.0, 112.0]),
            rotation: 0.0,
            opacity: 1.0,
            uv: [0.0, 0.0, 1.0, 1.0],
            effects: Vec::new(),
            mask: None,
            chroma_key: None,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: None,
        }
    }

    fn text_layer() -> SceneLayer {
        SceneLayer {
            clip: None,
            source: LayerSource::Text {
                content: "hi".into(),
                style: TextStyle::new(48.0),
                animation: None,
            },
            center: [50.0, 25.0],
            anchor_point: [0.5, 0.5],
            size: SizeSpec::BitmapScaled([2.0, 2.0]),
            rotation: 0.0,
            opacity: 1.0,
            uv: [0.0, 0.0, 1.0, 1.0],
            effects: Vec::new(),
            mask: None,
            chroma_key: None,
            color_grade: None,
            lut: None,
            blend_mode: BlendMode::Normal,
            styles: None,
        }
    }

    #[test]
    fn text_alignment_positions_tight_bounds_against_canvas_edges() {
        let layer = text_layer();
        let size = [200.0, 100.0];
        let canvas = [1920.0, 1080.0];

        assert_eq!(
            layer.text_quad_center(
                &TextStyle::new(48.0)
                    .with_align(TextAlign::Left)
                    .with_vertical_align(TextVerticalAlign::Top),
                size,
                canvas,
            ),
            [-810.0, -465.0]
        );
        assert_eq!(
            layer.text_quad_center(
                &TextStyle::new(48.0)
                    .with_align(TextAlign::Right)
                    .with_vertical_align(TextVerticalAlign::Bottom),
                size,
                canvas,
            ),
            [910.0, 515.0]
        );
    }

    #[test]
    fn scale_halves_canvas_and_layer_geometry() {
        let mut scene = Scene::empty(1920, 1080, [0, 0, 0, 255]);
        scene
            .layers
            .push(media_layer([960.0, 540.0], [1920.0, 1080.0]));
        scene.layers.push(text_layer());
        scene.layers.push(shape_layer());

        scene.scale(0.5);

        assert_eq!((scene.width, scene.height), (960, 540));
        let SizeSpec::Fixed(size) = scene.layers[0].size else {
            panic!("media layer keeps a fixed size");
        };
        assert_eq!(scene.layers[0].center, [480.0, 270.0]);
        assert_eq!(size, [960.0, 540.0]);
        // Rotation, opacity, and uv (content-relative) are untouched.
        assert_eq!(scene.layers[0].rotation, 0.5);
        assert_eq!(scene.layers[0].opacity, 0.8);
        assert_eq!(scene.layers[0].uv, [0.1, 0.2, 0.9, 0.8]);

        // Text scales through its bitmap multiplier.
        assert_eq!(scene.layers[1].size, SizeSpec::BitmapScaled([1.0, 1.0]));

        // SDF stroke width and pad are canvas-pixel quantities.
        let LayerSource::Shape { stroke, pad, .. } = &scene.layers[2].source else {
            panic!("shape layer");
        };
        assert_eq!(stroke.unwrap().width, 4.0);
        assert_eq!(*pad, 3.0);
    }

    #[test]
    fn scale_ignores_degenerate_factors() {
        let mut scene = Scene::empty(100, 50, [0, 0, 0, 255]);
        scene.scale(0.0);
        scene.scale(-1.0);
        scene.scale(f32::NAN);
        assert_eq!((scene.width, scene.height), (100, 50));
    }

    #[test]
    fn fit_within_never_upscales() {
        let mut scene = Scene::empty(640, 360, [0, 0, 0, 255]);
        scene.fit_within(4000, 4000);
        assert_eq!((scene.width, scene.height), (640, 360));
    }

    #[test]
    fn fit_within_shrinks_to_the_tighter_axis() {
        let mut scene = Scene::empty(1920, 1080, [0, 0, 0, 255]);
        scene.fit_within(400, 400);
        assert_eq!((scene.width, scene.height), (400, 225));
    }

    #[test]
    fn fit_within_survives_a_zero_canvas() {
        let mut scene = Scene::empty(0, 0, [0, 0, 0, 255]);
        scene.fit_within(100, 100);
        assert_eq!((scene.width, scene.height), (0, 0));
    }

    #[test]
    fn fit_into_letterboxes_an_aspect_mismatch() {
        // 16:9 content into a square: scaled to 400×225, centered vertically.
        let mut scene = Scene::empty(1920, 1080, [1, 2, 3, 255]);
        scene
            .layers
            .push(media_layer([960.0, 540.0], [1920.0, 1080.0]));

        scene.fit_into(400, 400);

        assert_eq!((scene.width, scene.height), (400, 400));
        let SizeSpec::Fixed(size) = scene.layers[0].size else {
            panic!("media layer keeps a fixed size");
        };
        // Content box is 400×225; its center sits at the canvas center.
        assert!((size[0] - 400.0).abs() < 1e-3);
        assert!((size[1] - 225.0).abs() < 1e-3);
        assert!((scene.layers[0].center[0] - 200.0).abs() < 1e-3);
        assert!((scene.layers[0].center[1] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn fit_into_upscales_for_export_overrides() {
        let mut scene = Scene::empty(960, 540, [0, 0, 0, 255]);
        scene
            .layers
            .push(media_layer([480.0, 270.0], [960.0, 540.0]));
        scene.fit_into(1920, 1080);
        assert_eq!((scene.width, scene.height), (1920, 1080));
        assert_eq!(scene.layers[0].center, [960.0, 540.0]);
    }
}
