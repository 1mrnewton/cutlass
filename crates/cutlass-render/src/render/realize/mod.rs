use std::collections::HashMap;
use std::time::Instant;

mod layers;
mod text;

use layers::{Realized, composite_from_realized, fixed_size};

use cutlass_compositor::{
    ColorGrade, CompositeLayer, CompositorConfig, CompositorLayer, FrameSink, LayerPlacement,
    PassInstance, SdfLayer,
};
use cutlass_models::{MediaId, Project};
use cutlass_shapes::ShapeStyle;

use crate::error::RenderError;
use crate::scene::{LayerSource, Scene, SceneLut, SizeSpec};

use super::effects::{EffectChain, blend_mode, layer_effects, layer_styles, pack_effects};
use super::media_cache::layer_lut;
use super::{FrameStats, Renderer, SLOW_FRAME_LOG_MS, SeekPolicy};

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
            let mode = blend_mode(layer.blend_mode);
            let styles = layer_styles(layer.styles.as_ref());
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
                        blend_mode: mode,
                        styles,
                    });
                }
                LayerSource::Text {
                    content,
                    style,
                    animation,
                } => {
                    let Some(layer_realized) = text::realize_text_layer(
                        &mut self.text,
                        layer,
                        content,
                        style,
                        animation,
                        [scene.width as f32, scene.height as f32],
                        layer.effects.clone(),
                        fx,
                        color_grade,
                        scene_lut,
                        mode,
                        styles,
                    ) else {
                        continue;
                    };
                    realized.push(layer_realized);
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
                        blend_mode: mode,
                        styles,
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
                        blend_mode: mode,
                        styles,
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
                        blend_mode: mode,
                        styles,
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
                        blend_mode: mode,
                        styles,
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
                        blend_mode: mode,
                        styles,
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
                        SizeSpec::Fixed(_) => [1.0, 1.0],
                    };
                    let size = [
                        image.width as f32 * scale[0],
                        image.height as f32 * scale[1],
                    ];
                    realized.push(Realized::Bitmap {
                        image,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                        blend_mode: mode,
                        styles,
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
                    blend_mode,
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
                            .with_lut(layer_lut(lut, &self.luts))
                            .with_blend_mode(*blend_mode),
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
}
