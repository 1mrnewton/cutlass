use std::collections::HashMap;
use std::time::Instant;

use cutlass_compositor::{
    ColorGrade, CompositeLayer, CompositorConfig, CompositorLayer, FrameSink, GlyphInstance,
    GlyphsLayer, LayerEffects, LayerPlacement, PassInstance, RgbaImage, SdfLayer,
};
use cutlass_core::VideoFrame;
use cutlass_models::{MediaId, Project};
use cutlass_shapes::ShapeStyle;

use crate::error::RenderError;
use crate::scene::{LayerSource, ResolvedPass, Scene, SceneLut, SizeSpec};

use super::effects::{EffectChain, layer_effects, pack_effects};
use super::media_cache::{CubeLutState, LottieState, StickerSequence, layer_lut};
use super::text_anim::{atlas_key, cluster_deltas, extent_origin, place_clusters};
use super::{FrameStats, Renderer, SLOW_FRAME_LOG_MS, SeekPolicy};

fn composite_from_realized<'a>(
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
enum Realized {
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
    fn effects(&self) -> Option<&[ResolvedPass]> {
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
fn fixed_size(size: SizeSpec, canvas: [f32; 2]) -> [f32; 2] {
    match size {
        SizeSpec::Fixed(s) => s,
        SizeSpec::BitmapScaled(_) => canvas,
    }
}

impl Renderer {
    pub(super) fn render_scene_once(
        &mut self,
        project: &Project,
        scene: &Scene,
        sink: &mut dyn FrameSink,
        policy: SeekPolicy,
    ) -> Result<(), RenderError> {
        let realize_started = Instant::now();
        // New scene, new LRU stamp: frames touched below are eviction-exempt
        // until the next scene.
        self.lottie_stamp += 1;
        // Decode time accumulated across media layers — on weak machines this
        // is where whole-frame seconds hide, so the stage log splits it out.
        let mut decode_ms = 0.0f64;
        // First pass: decode/rasterize each layer into owned pixels and a final
        // placement. Held in `realized` so the borrowed `CompositeLayer`s built
        // below stay valid through the composite call.
        let mut realized: Vec<Realized> = Vec::with_capacity(scene.layers.len());
        let mut effect_store: Vec<EffectChain> = Vec::new();
        let mut occurrence: HashMap<MediaId, u32> = HashMap::new();
        for layer in &scene.layers {
            let fx = layer_effects(layer);
            let color_grade = layer.color_grade;
            // Load (or recall) the layer's .cube table; unreadable files
            // resolve to None and grade nothing.
            let scene_lut = self.resolve_scene_lut(&layer.lut);
            // The layer carries the anchor position; the quad center falls out
            // of the final pixel size (bitmap sizes only exist after raster).
            let place = |size: [f32; 2]| LayerPlacement {
                center: layer.quad_center(size),
                size,
                rotation: layer.rotation,
                opacity: layer.opacity,
            };
            match &layer.source {
                LayerSource::CanvasPass => {
                    realized.push(Realized::CanvasPass {
                        effects: layer.effects.clone(),
                        grade: color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Transition {
                    outgoing,
                    incoming,
                    transition_id,
                    progress,
                } => {
                    let out = self.realize_subscene_layer(project, scene, outgoing, policy)?;
                    let inc = self.realize_subscene_layer(project, scene, incoming, policy)?;
                    realized.push(Realized::Transition {
                        outgoing: out,
                        incoming: inc,
                        transition_id: transition_id.clone(),
                        progress: *progress,
                    });
                }
                LayerSource::Solid(rgba) => {
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Solid {
                        rgba: *rgba,
                        placement: place(size),
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Text {
                    content,
                    style,
                    animation,
                } => {
                    let scale = match layer.size {
                        SizeSpec::BitmapScaled(s) => s,
                        SizeSpec::Fixed(_) => 1.0,
                    };
                    if let Some(anim) = animation {
                        let shaped = self.text.shape(content, style);
                        if !shaped.has_ink() {
                            continue;
                        }
                        let painted = cutlass_text::paint_animated(&shaped, style);
                        // Deltas use the ink-tight shaped clusters (logical
                        // order / baselines); placement uses painted images.
                        let deltas = cluster_deltas(&shaped, anim);
                        let extent_size = [
                            painted.extent.0 as f32 * scale,
                            painted.extent.1 as f32 * scale,
                        ];
                        let aligned = layer.text_quad_center(
                            style,
                            extent_size,
                            [scene.width as f32, scene.height as f32],
                        );
                        let origin = extent_origin(aligned, painted.extent, scale);
                        let glyphs: Vec<RgbaImage> =
                            painted.clusters.iter().map(|c| c.image.clone()).collect();
                        // place_clusters reads offsets/baselines from ShapedText;
                        // rebuild a shaped view over the painted clusters.
                        let painted_shaped = cutlass_text::ShapedText {
                            extent: painted.extent,
                            clusters: painted.clusters.clone(),
                        };
                        let instances = place_clusters(
                            &painted_shaped,
                            &deltas,
                            origin,
                            scale,
                            layer.rotation,
                            layer.opacity,
                        );
                        if instances.is_empty() {
                            continue;
                        }
                        let background = painted.background.map(|bg| {
                            let size = [bg.width as f32 * scale, bg.height as f32 * scale];
                            let center = [
                                origin[0] + painted.background_offset[0] * scale + size[0] * 0.5,
                                origin[1] + painted.background_offset[1] * scale + size[1] * 0.5,
                            ];
                            (
                                bg,
                                LayerPlacement {
                                    center,
                                    size,
                                    rotation: layer.rotation,
                                    opacity: layer.opacity,
                                },
                            )
                        });
                        realized.push(Realized::Glyphs {
                            glyphs,
                            instances,
                            atlas_key: atlas_key(content, style),
                            background,
                            placement: LayerPlacement {
                                center: aligned,
                                size: extent_size,
                                rotation: 0.0,
                                opacity: 1.0,
                            },
                            effects: layer.effects.clone(),
                            fx,
                            color_grade,
                            lut: scene_lut,
                        });
                    } else {
                        let image = self.text.rasterize(content, style);
                        if image.width == 0 || image.height == 0 {
                            continue; // nothing rasterized (no fonts / empty run)
                        }
                        let size = [image.width as f32 * scale, image.height as f32 * scale];
                        let placement = LayerPlacement {
                            center: layer.text_quad_center(
                                style,
                                size,
                                [scene.width as f32, scene.height as f32],
                            ),
                            size,
                            rotation: layer.rotation,
                            opacity: layer.opacity,
                        };
                        realized.push(Realized::Bitmap {
                            image,
                            placement,
                            uv: layer.uv,
                            effects: layer.effects.clone(),
                            fx,
                            color_grade,
                            lut: scene_lut,
                        });
                    }
                }
                LayerSource::Media { media, source_time } => {
                    let slot = occurrence.entry(*media).or_insert(0);
                    let decode_started = Instant::now();
                    let frame = self.decode(project, *media, *slot, *source_time, policy)?;
                    decode_ms += decode_started.elapsed().as_secs_f64() * 1000.0;
                    *slot += 1;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Frame {
                        frame,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Still { media } => {
                    self.ensure_still(project, *media)?;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Still {
                        media: *media,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Lottie { path, local_time } => {
                    // A missing or unsupported file draws nothing (the media
                    // offline story — projects move machines), never an error.
                    let Some(frame_index) = self.ensure_lottie_frame(path, *local_time) else {
                        continue;
                    };
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Lottie {
                        path: path.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Sticker { asset, local_time } => {
                    // The resolver only emits catalog ids, but stay graceful:
                    // an unknown id draws nothing rather than failing a frame.
                    let Some(spec) = cutlass_models::sticker_spec(asset) else {
                        continue;
                    };
                    self.ensure_sticker(spec)?;
                    let frame_index = self.stickers[spec.id].frame_at(*local_time);
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Sticker {
                        asset: asset.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Shape {
                    params,
                    fill,
                    stroke,
                    pad,
                } => {
                    // The resolver sized the quad as shape + pad per side;
                    // recover the shape's own half-extents for the shader.
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    let half = [
                        (size[0] * 0.5 - pad).max(0.0),
                        (size[1] * 0.5 - pad).max(0.0),
                    ];
                    realized.push(Realized::Sdf {
                        shape: SdfLayer {
                            shape: params.with_half(half),
                            fill: *fill,
                            stroke: *stroke,
                        },
                        placement: place(size),
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
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
                        continue; // nothing inked (degenerate path or style)
                    }
                    let scale = match layer.size {
                        SizeSpec::BitmapScaled(s) => s,
                        SizeSpec::Fixed(_) => 1.0,
                    };
                    let size = [image.width as f32 * scale, image.height as f32 * scale];
                    realized.push(Realized::Bitmap {
                        image,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
            }
        }

        // Pack effect chains and build compositor layers with stable borrows.
        // Transition sides are nested; pack outgoing then incoming to match
        // the phase-1 consumption order below.
        for r in &realized {
            match r {
                Realized::Transition {
                    outgoing, incoming, ..
                } => {
                    for side in [&**outgoing, &**incoming] {
                        if let Some(effects) = side.effects().filter(|e| !e.is_empty()) {
                            effect_store.push(pack_effects(effects));
                        }
                    }
                }
                other => {
                    if let Some(effects) = other.effects().filter(|e| !e.is_empty()) {
                        effect_store.push(pack_effects(effects));
                    }
                }
            }
        }
        let instance_store: Vec<Vec<PassInstance<'_>>> =
            effect_store.iter().map(EffectChain::instances).collect();

        let mut effect_idx = 0usize;
        let mut layer_storage: Vec<CompositeLayer<'_>> = Vec::new();
        // Phase 1: build all composite layers (indices only for transitions).
        enum LayerJob<'a> {
            Plain {
                storage_idx: usize,
            },
            CanvasPass {
                effects: &'a [PassInstance<'a>],
                grade: Option<ColorGrade>,
                lut: &'a Option<SceneLut>,
            },
            Transition {
                out_idx: usize,
                in_idx: usize,
                transition_id: &'a str,
                progress: f32,
            },
        }
        let mut jobs: Vec<LayerJob<'_>> = Vec::new();

        for r in &realized {
            match r {
                Realized::CanvasPass {
                    effects,
                    grade,
                    lut,
                } => {
                    let effects = if effects.is_empty() {
                        &[]
                    } else {
                        let chain = &instance_store[effect_idx];
                        effect_idx += 1;
                        chain.as_slice()
                    };
                    jobs.push(LayerJob::CanvasPass {
                        effects,
                        grade: *grade,
                        lut,
                    });
                }
                Realized::Transition {
                    outgoing,
                    incoming,
                    transition_id,
                    progress,
                } => {
                    let out_effects = outgoing
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        outgoing.as_ref(),
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        out_effects,
                    ));
                    let out_idx = layer_storage.len() - 1;
                    let in_effects = incoming
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        incoming.as_ref(),
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        in_effects,
                    ));
                    let in_idx = layer_storage.len() - 1;
                    jobs.push(LayerJob::Transition {
                        out_idx,
                        in_idx,
                        transition_id: transition_id.as_str(),
                        progress: *progress,
                    });
                }
                Realized::Glyphs {
                    background: Some((bg_image, bg_placement)),
                    effects,
                    fx,
                    color_grade,
                    lut,
                    ..
                } => {
                    // Whole-run background card sits behind the glyphs.
                    let bg_effects = if effects.is_empty() {
                        &[]
                    } else {
                        let chain = &instance_store[effect_idx];
                        effect_idx += 1;
                        chain.as_slice()
                    };
                    layer_storage.push(
                        CompositeLayer::rgba(bg_image, *bg_placement)
                            .with_fx(*fx)
                            .with_effects(bg_effects)
                            .with_color_grade(*color_grade)
                            .with_lut(layer_lut(lut, &self.luts)),
                    );
                    jobs.push(LayerJob::Plain {
                        storage_idx: layer_storage.len() - 1,
                    });
                    // Glyphs share the same effect chain reference when present.
                    let glyph_effects = if effects.is_empty() { &[] } else { bg_effects };
                    layer_storage.push(composite_from_realized(
                        r,
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        glyph_effects,
                    ));
                    jobs.push(LayerJob::Plain {
                        storage_idx: layer_storage.len() - 1,
                    });
                }
                other => {
                    let effects = other
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        other,
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        effects,
                    ));
                    jobs.push(LayerJob::Plain {
                        storage_idx: layer_storage.len() - 1,
                    });
                }
            }
        }

        // Phase 2: borrow storage immutably for compositor dispatch.
        let compositor_layers: Vec<CompositorLayer<'_>> = jobs
            .iter()
            .map(|job| match job {
                LayerJob::Plain { storage_idx } => {
                    CompositorLayer::layer(&layer_storage[*storage_idx])
                }
                LayerJob::CanvasPass {
                    effects,
                    grade,
                    lut,
                } => CompositorLayer::CanvasPass {
                    effects,
                    grade: *grade,
                    lut: layer_lut(lut, &self.luts),
                },
                LayerJob::Transition {
                    out_idx,
                    in_idx,
                    transition_id,
                    progress,
                } => CompositorLayer::Transition {
                    outgoing: &layer_storage[*out_idx],
                    incoming: &layer_storage[*in_idx],
                    transition_id,
                    progress: *progress,
                },
            })
            .collect();

        let config =
            CompositorConfig::new(scene.width, scene.height).with_background(scene.background);
        let realize_ms = realize_started.elapsed().as_secs_f64() * 1000.0;
        let composite_started = Instant::now();
        self.compositor.render_compositor_layers_into(
            &self.gpu,
            &config,
            &compositor_layers,
            sink,
        )?;

        // Stage breakdown per frame: decode (media layers), raster (text/
        // shape/still realize minus decode), composite (GPU submit + mapped
        // readback). Slow frames surface at `info` so a default-filtered log
        // shows where the seconds go on decode- or GPU-bound machines.
        let composite_ms = composite_started.elapsed().as_secs_f64() * 1000.0;
        let raster_ms = (realize_ms - decode_ms).max(0.0);
        let total_ms = realize_ms + composite_ms;
        self.last_stats = FrameStats {
            decode_ms,
            raster_ms,
            composite_ms,
        };
        if total_ms > SLOW_FRAME_LOG_MS {
            tracing::info!(
                decode_ms = %format_args!("{decode_ms:.1}"),
                raster_ms = %format_args!("{raster_ms:.1}"),
                composite_ms = %format_args!("{composite_ms:.1}"),
                layers = compositor_layers.len(),
                width = scene.width,
                height = scene.height,
                "slow frame render: {total_ms:.0} ms"
            );
        } else {
            tracing::trace!(
                decode_ms = %format_args!("{decode_ms:.1}"),
                raster_ms = %format_args!("{raster_ms:.1}"),
                composite_ms = %format_args!("{composite_ms:.1}"),
                layers = compositor_layers.len(),
                "frame render: {total_ms:.1} ms"
            );
        }
        Ok(())
    }

    fn realize_subscene_layer(
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
                let image = self.text.rasterize(content, style);
                if image.width == 0 || image.height == 0 {
                    return Err(RenderError::unsupported("empty text layer"));
                }
                let scale = match layer.size {
                    SizeSpec::BitmapScaled(s) => s,
                    SizeSpec::Fixed(_) => 1.0,
                };
                let size = [image.width as f32 * scale, image.height as f32 * scale];
                let placement = LayerPlacement {
                    center: layer.text_quad_center(
                        style,
                        size,
                        [scene.width as f32, scene.height as f32],
                    ),
                    size,
                    rotation: layer.rotation,
                    opacity: layer.opacity,
                };
                Realized::Bitmap {
                    image,
                    placement,
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
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
