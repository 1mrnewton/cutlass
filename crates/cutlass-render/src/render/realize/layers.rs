//! The owned realized-layer model: what a decoded/rasterized layer is, how a
//! single scene layer becomes one, and how it converts to a [`CompositeLayer`].

use std::collections::HashMap;

use cutlass_compositor::{
    ColorGrade, CompositeLayer, GlyphInstance, GlyphsLayer, LayerEffects, LayerPlacement,
    PassInstance, RgbaImage, SdfLayer,
};
use cutlass_core::VideoFrame;
use cutlass_models::{MediaId, Project};
use cutlass_shapes::ShapeStyle;

use crate::error::RenderError;
use crate::scene::{LayerSource, ResolvedPass, Scene, SceneLut, SizeSpec};

use super::super::effects::layer_effects;
use super::super::media_cache::{CubeLutState, LottieState, StickerSequence, layer_lut};
use super::super::{Renderer, SeekPolicy};
use super::text;

pub(super) fn composite_from_realized<'a>(
    r: &'a Realized,
    stills: &'a HashMap<MediaId, RgbaImage>,
    stickers: &'a HashMap<String, StickerSequence>,
    lottie: &'a HashMap<String, LottieState>,
    luts: &'a HashMap<String, CubeLutState>,
    effects: &'a [PassInstance<'a>],
) -> CompositeLayer<'a> {
    match r {
        Realized::Solid {
            rgba,
            placement,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::solid(*rgba, *placement)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Bitmap {
            image,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(image, *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Frame {
            frame,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::frame(frame, *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Still {
            media,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(&stills[media], *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Sticker {
            asset,
            frame_index,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(&stickers[asset].frames[*frame_index], *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Lottie {
            path,
            frame_index,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => {
            // Realize only emits `Realized::Lottie` after `ensure_lottie_frame`
            // cached this exact frame, and the LRU never evicts frames stamped
            // by the scene being composed.
            let LottieState::Loaded(player) = &lottie[path] else {
                unreachable!("realized lottie layer without a loaded player")
            };
            CompositeLayer::rgba(&player.frames[frame_index].0, *placement)
                .with_uv(*uv)
                .with_fx(*fx)
                .with_effects(effects)
                .with_color_grade(*color_grade)
                .with_lut(layer_lut(lut, luts))
        }
        Realized::Sdf {
            shape,
            placement,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::sdf(*shape, *placement)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Glyphs {
            glyphs,
            instances,
            atlas_key,
            background,
            placement,
            fx,
            color_grade,
            lut,
            ..
        } => {
            // Background is composited as a separate preceding layer in the
            // job walk; this arm only builds the glyph instances.
            let _ = background;
            CompositeLayer::glyphs(
                GlyphsLayer {
                    atlas_key: *atlas_key,
                    glyphs,
                    instances,
                },
                *placement,
            )
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts))
        }
        Realized::Transition { .. } | Realized::CanvasPass { .. } => {
            unreachable!("non-layer realized items handled separately")
        }
    }
}

/// An owned, decoded/rasterized layer kept alive while the compositor borrows it.
pub(super) enum Realized {
    Transition {
        outgoing: Box<Realized>,
        incoming: Box<Realized>,
        transition_id: String,
        progress: f32,
    },
    CanvasPass {
        effects: Vec<ResolvedPass>,
        grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Frame {
        frame: VideoFrame,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Still {
        media: MediaId,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Sticker {
        asset: String,
        frame_index: usize,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Lottie {
        path: String,
        frame_index: usize,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Bitmap {
        image: RgbaImage,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Solid {
        rgba: [u8; 4],
        placement: LayerPlacement,
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Sdf {
        shape: SdfLayer,
        placement: LayerPlacement,
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    /// Per-character text: cluster bitmaps + instanced placements.
    Glyphs {
        glyphs: Vec<RgbaImage>,
        instances: Vec<GlyphInstance>,
        atlas_key: u64,
        /// Optional whole-run background card drawn behind the glyphs.
        background: Option<(RgbaImage, LayerPlacement)>,
        /// Layer opacity multiplier (instance opacities are pre-multiplied).
        placement: LayerPlacement,
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
}

impl Realized {
    pub(super) fn effects(&self) -> Option<&[ResolvedPass]> {
        match self {
            Realized::Transition { .. } => None,
            Realized::CanvasPass { effects, .. } => Some(effects),
            Realized::Frame { effects, .. }
            | Realized::Still { effects, .. }
            | Realized::Sticker { effects, .. }
            | Realized::Lottie { effects, .. }
            | Realized::Bitmap { effects, .. }
            | Realized::Solid { effects, .. }
            | Realized::Sdf { effects, .. }
            | Realized::Glyphs { effects, .. } => Some(effects),
        }
    }
}

/// The on-canvas size for a non-text layer, falling back to the canvas if a
/// bitmap-scaled spec ever reaches here (it shouldn't for media/solid).
pub(super) fn fixed_size(size: SizeSpec, canvas: [f32; 2]) -> [f32; 2] {
    match size {
        SizeSpec::Fixed(s) => s,
        SizeSpec::BitmapScaled(_) => canvas,
    }
}

impl Renderer {
    pub(super) fn realize_subscene_layer(
        &mut self,
        project: &Project,
        scene: &Scene,
        layer: &crate::scene::SceneLayer,
        policy: SeekPolicy,
    ) -> Result<Box<Realized>, RenderError> {
        let place = |size: [f32; 2]| LayerPlacement {
            center: layer.quad_center(size),
            size,
            rotation: layer.rotation,
            opacity: layer.opacity,
        };
        let fx = layer_effects(layer);
        let color_grade = layer.color_grade;
        let realized = match &layer.source {
            LayerSource::Solid(rgba) => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Solid {
                    rgba: *rgba,
                    placement: place(size),
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Text {
                content,
                style,
                animation,
            } => {
                // Transition sides keep the bitmap path — per-character
                // animation on a transition edge is not a supported surface.
                let _ = animation;
                text::realize_text_bitmap(
                    &mut self.text,
                    layer,
                    content,
                    style,
                    [scene.width as f32, scene.height as f32],
                    layer.effects.clone(),
                    fx,
                    color_grade,
                )?
            }
            LayerSource::Media { media, source_time } => {
                let frame = self.decode(project, *media, 0, *source_time, policy)?;
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Frame {
                    frame,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Still { media } => {
                self.ensure_still(project, *media)?;
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Still {
                    media: *media,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Lottie { path, local_time } => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                match self.ensure_lottie_frame(path, *local_time) {
                    Some(frame_index) => Realized::Lottie {
                        path: path.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: None,
                    },
                    // Missing file inside a transition: a transparent side,
                    // matching the draw-nothing policy of the main path.
                    None => Realized::Solid {
                        rgba: [0, 0, 0, 0],
                        placement: place(size),
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: None,
                    },
                }
            }
            LayerSource::Sticker { asset, local_time } => {
                let spec = cutlass_models::sticker_spec(asset)
                    .ok_or_else(|| RenderError::unsupported("unknown sticker asset"))?;
                self.ensure_sticker(spec)?;
                let frame_index = self.stickers[spec.id].frame_at(*local_time);
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Sticker {
                    asset: asset.clone(),
                    frame_index,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Shape {
                params,
                fill,
                stroke,
                pad,
            } => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                let half = [
                    (size[0] * 0.5 - pad).max(0.0),
                    (size[1] * 0.5 - pad).max(0.0),
                ];
                Realized::Sdf {
                    shape: SdfLayer {
                        shape: params.with_half(half),
                        fill: *fill,
                        stroke: *stroke,
                    },
                    placement: place(size),
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::PathShape {
                path,
                fill,
                stroke,
                raster_scale,
            } => {
                let style = ShapeStyle {
                    fill: Some(*fill).filter(|c| c[3] > 0),
                    stroke: *stroke,
                };
                let image = self.paths.rasterize(path, &style, *raster_scale);
                if image.width == 0 || image.height == 0 {
                    return Err(RenderError::unsupported("degenerate path layer"));
                }
                let scale = match layer.size {
                    SizeSpec::BitmapScaled(s) => s,
                    SizeSpec::Fixed(_) => 1.0,
                };
                let size = [image.width as f32 * scale, image.height as f32 * scale];
                Realized::Bitmap {
                    image,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Transition { .. } => {
                return Err(RenderError::unsupported("nested transitions"));
            }
            LayerSource::CanvasPass => {
                return Err(RenderError::unsupported("nested canvas pass"));
            }
        };
        Ok(Box::new(realized))
    }
}
