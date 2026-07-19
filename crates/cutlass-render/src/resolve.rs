//! The pure timeline → [`Scene`] resolver.
//!
//! Given a [`Project`] and a timeline instant, [`resolve`] walks the visual
//! track stack bottom-to-top, finds the clip active on each lane, and turns it
//! into a placed [`SceneLayer`]: canvas geometry, transform, crop/mirror, and a
//! classified pixel source. It decodes nothing and touches no GPU, so the
//! geometry is deterministic and unit-testable on any platform.
//!
//! ## Coverage (v1)
//!
//! - **Media**: video sources and still images (both aspect-fit into the
//!   canvas, then scaled; stills place one cached frame for the clip's whole
//!   extent).
//! - **Generators**: text, solid fills, and every shape kind — parametric
//!   shapes resolve to sampled SDF layers (animated geometry/colors are
//!   sampled per instant here, evaluated on the GPU), pen paths to CPU-raster
//!   layers.
//! - **Lane passes**: effect, filter, and adjustment generator bars resolve to
//!   canvas-wide passes over everything below their track.
//! - **Deferred**: stickers are skipped (they produce no layer) rather than
//!   rendered wrong.

use cutlass_compositor::ColorGrade;
use cutlass_core::{RationalTime, resample};
use cutlass_models::{
    ClipId, ClipSource, ClipTransform, ColorAdjustments, EffectInstance, Filter, Generator,
    MediaKind, Param, Project, Shape, ShapePath, ShapeStroke, TextAlignH,
    TextStyle as ModelTextStyle,
};
use cutlass_shapes::{BezierPath, PathPoint, SDF_AA, SdfParams, Stroke};
use cutlass_text::{FontFamily, TextAlign, TextBackground, TextShadow, TextStroke, TextStyle};

use crate::animation::apply_look_animations;
use crate::grade::resolve_color_grade;
use crate::scene::{LayerSource, ResolvedPass, Scene, SceneLayer, SceneLut, SizeSpec};

/// Vertical reference height that a generator's reference-pixel sizes (text
/// `size`, shape `width`/`height`) are authored against. Matches the model's
/// `canvas_height / 1080` convention.
const REFERENCE_HEIGHT: f32 = 1080.0;

/// Fallback canvas size when `Auto` aspect can't find any video media.
const DEFAULT_CANVAS: (u32, u32) = (1920, 1080);

/// Live-preview substitutions resolved in place of committed clip state.
///
/// A drag/scale/rotate gesture overrides one clip's transform; a live
/// inspector edit (font-size slider, shape color) overrides one clip's
/// generator. Both are session-side only: the project, history, and export
/// never see them — release commits one real edit and clears the override.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResolveOverrides<'a> {
    pub transform: Option<(ClipId, ClipTransform)>,
    pub generator: Option<(ClipId, &'a Generator)>,
    pub look: Option<(ClipId, Option<&'a Filter>, &'a ColorAdjustments)>,
}

/// Identity transform used when rasterizing the gesture sprite: the clip's
/// pixels at scale 1, centered on the canvas, with no rotation.
pub const GESTURE_IDENTITY_TRANSFORM: ClipTransform = ClipTransform {
    position: [0.0, 0.0],
    anchor_point: [0.5, 0.5],
    scale: 1.0,
    rotation: 0.0,
    opacity: 1.0,
};

/// Three scene partitions for zero-drift preview transform gestures.
#[derive(Debug, Clone, PartialEq)]
pub struct GestureScenePartition {
    /// Layers below the dragged clip over the canvas background.
    pub below: Scene,
    /// The dragged clip alone at [`GESTURE_IDENTITY_TRANSFORM`].
    pub sprite: Scene,
    /// Layers above the dragged clip (may be empty).
    pub above: Scene,
}

/// Resolve the scene at `t`, force `clip_id`'s transform to identity, and
/// split the stack into below / sprite / above partitions. Returns `None` when
/// the clip isn't composited at `t`, sits inside a transition window, or
/// otherwise can't be sprite-partitioned (caller falls back to per-move
/// override rendering).
pub fn resolve_gesture_partitions(
    project: &Project,
    t: RationalTime,
    clip_id: ClipId,
) -> Result<Option<GestureScenePartition>, cutlass_models::ModelError> {
    let overrides = ResolveOverrides {
        transform: Some((clip_id, GESTURE_IDENTITY_TRANSFORM)),
        generator: None,
        look: None,
    };
    let scene = resolve_with(project, t, overrides)?;
    let index = scene
        .layers
        .iter()
        .position(|layer| layer.clip == Some(clip_id));
    let Some(index) = index else {
        return Ok(None);
    };
    if matches!(
        scene.layers[index].source,
        LayerSource::Transition { .. } | LayerSource::CanvasPass
    ) {
        return Ok(None);
    }
    if scene.layers[index + 1..]
        .iter()
        .any(|layer| matches!(layer.source, LayerSource::CanvasPass))
    {
        return Ok(None);
    }

    Ok(Some(GestureScenePartition {
        below: Scene {
            width: scene.width,
            height: scene.height,
            background: scene.background,
            layers: scene.layers[..index].to_vec(),
        },
        sprite: Scene {
            width: scene.width,
            height: scene.height,
            background: [0, 0, 0, 0],
            layers: vec![scene.layers[index].clone()],
        },
        above: Scene {
            width: scene.width,
            height: scene.height,
            background: [0, 0, 0, 0],
            layers: scene.layers[index + 1..].to_vec(),
        },
    }))
}

/// Resolve `project` at timeline instant `t` into a [`Scene`].
///
/// `t` is interpreted at the timeline frame rate (it is resampled to it first),
/// so callers may pass a tick at any rate.
pub fn resolve(project: &Project, t: RationalTime) -> Result<Scene, cutlass_models::ModelError> {
    resolve_with(project, t, ResolveOverrides::default())
}

/// [`resolve`] with live-preview [`ResolveOverrides`] applied.
pub fn resolve_with(
    project: &Project,
    t: RationalTime,
    overrides: ResolveOverrides<'_>,
) -> Result<Scene, cutlass_models::ModelError> {
    let timeline = project.timeline();
    let rate = timeline.frame_rate;
    let t = resample(t, rate);

    let (width, height) = canvas_size(project);
    let bg = timeline.canvas().background;
    let mut scene = Scene::empty(width, height, [bg[0], bg[1], bg[2], 255]);

    let cw = width as f32;
    let ch = height as f32;

    for track in timeline.tracks_ordered() {
        if !track.kind.is_visual() || !track.enabled {
            continue;
        }
        if let Some(layer) = resolve_track_at(project, track, t, cw, ch, overrides)? {
            scene.layers.push(layer);
        }
    }

    Ok(scene)
}

/// Resolve one visual track at timeline instant `t`.
fn resolve_track_at(
    project: &Project,
    track: &cutlass_models::Track,
    t: RationalTime,
    cw: f32,
    ch: f32,
    overrides: ResolveOverrides<'_>,
) -> Result<Option<SceneLayer>, cutlass_models::ModelError> {
    // Transition window takes precedence over single-clip resolve.
    for transition in track.transitions() {
        let left = track
            .clip(transition.left)
            .ok_or(cutlass_models::ModelError::UnknownClip(transition.left))?;
        let right = track
            .clip(transition.right)
            .ok_or(cutlass_models::ModelError::UnknownClip(transition.right))?;
        if left.timeline.end_tick() != right.timeline.start.value {
            continue;
        }
        let cut = left.timeline.end_tick();
        let half = transition.duration / 2;
        let window_start = cut - half;
        let window_end = window_start + transition.duration;
        if t.value >= window_start && t.value < window_end {
            let progress = (t.value - window_start) as f32 / transition.duration as f32;
            // Each side plays live wherever it has material and holds its
            // boundary frame past it: the outgoing clip runs until the cut
            // then freezes on its last frame, the incoming holds its first
            // frame until the cut then runs — CapCut's motion, and the tail
            // frame is only requested for the window's back half. Clamped
            // into each clip's extent (not just at the cut) because the
            // model doesn't bound the duration by the clips' lengths.
            let outgoing_t = RationalTime::new(
                t.value
                    .clamp(left.timeline.start.value, left.timeline.end_tick() - 1),
                t.rate,
            );
            let incoming_t = RationalTime::new(
                t.value
                    .clamp(right.timeline.start.value, right.timeline.end_tick() - 1),
                t.rate,
            );
            // A side that produces no layer (e.g. empty text) or a canvas-wide
            // pass (effect/filter/adjustment segments) can't be composited as
            // a transition frame — the renderer rejects nested canvas passes.
            // Skip the transition and resolve the track normally so the
            // preview keeps updating instead of erroring every frame.
            let outgoing = resolve_clip(project, left, outgoing_t, cw, ch, overrides)?;
            let incoming = resolve_clip(project, right, incoming_t, cw, ch, overrides)?;
            let (Some(outgoing), Some(incoming)) = (outgoing, incoming) else {
                break;
            };
            if matches!(outgoing.source, LayerSource::CanvasPass)
                || matches!(incoming.source, LayerSource::CanvasPass)
            {
                break;
            }
            let outgoing = Box::new(outgoing);
            let incoming = Box::new(incoming);
            return Ok(Some(SceneLayer {
                clip: None,
                source: LayerSource::Transition {
                    outgoing,
                    incoming,
                    transition_id: transition.transition_id.clone(),
                    progress,
                },
                center: [cw * 0.5, ch * 0.5],
                anchor_point: [0.5, 0.5],
                size: SizeSpec::Fixed([cw, ch]),
                rotation: 0.0,
                opacity: 1.0,
                uv: [0.0, 0.0, 1.0, 1.0],
                effects: Vec::new(),
                mask: None,
                chroma_key: None,
                color_grade: None,
                lut: None,
            }));
        }
    }

    let Some(clip) = track.clip_at(t)? else {
        return Ok(None);
    };
    resolve_clip(project, clip, t, cw, ch, overrides)
}

/// Canvas pixel size for `project`: fixed presets resolve to a 1080-baseline
/// box on the longer side; `Auto` follows the largest video media (falling back
/// to 1920×1080 when there is none).
pub fn canvas_size(project: &Project) -> (u32, u32) {
    match project.timeline().canvas().aspect.ratio() {
        Some((rw, rh)) => ratio_to_pixels(rw, rh),
        None => auto_canvas_size(project),
    }
}

/// Largest visible dimension box for a `w:h` ratio, even-rounded for encoders.
fn ratio_to_pixels(rw: u32, rh: u32) -> (u32, u32) {
    const BASE: f32 = REFERENCE_HEIGHT;
    let (rw, rh) = (rw as f32, rh as f32);
    let (w, h) = if rw >= rh {
        ((BASE * rw / rh).round(), BASE)
    } else {
        (BASE, (BASE * rh / rw).round())
    };
    (even(w as u32), even(h as u32))
}

/// The largest video media used anywhere on the timeline, or the default.
fn auto_canvas_size(project: &Project) -> (u32, u32) {
    let mut best: Option<(u32, u32)> = None;
    for track in project.timeline().tracks_ordered() {
        if !track.kind.is_visual() {
            continue;
        }
        for clip in track.clips() {
            let Some(id) = clip.media() else { continue };
            let Some(media) = project.media(id) else {
                continue;
            };
            if media.kind() != MediaKind::Video {
                continue;
            }
            let area = u64::from(media.width) * u64::from(media.height);
            if best.is_none_or(|(bw, bh)| area > u64::from(bw) * u64::from(bh)) {
                best = Some((media.width, media.height));
            }
        }
    }
    best.map_or(DEFAULT_CANVAS, |(w, h)| (even(w), even(h)))
}

/// Round `v` down to the nearest even value (≥ 2): H.264 needs even dimensions.
fn even(v: u32) -> u32 {
    (v & !1).max(2)
}

fn resolve_clip(
    project: &Project,
    clip: &cutlass_models::Clip,
    t: RationalTime,
    cw: f32,
    ch: f32,
    overrides: ResolveOverrides<'_>,
) -> Result<Option<SceneLayer>, cutlass_models::ModelError> {
    // Clip-relative tick at the timeline rate (both `t` and the clip start are
    // expressed at it), which is what animated transforms key against.
    let local_tick = clip.animation_tick(t.value);
    let local_tick_f = clip.animation_tick_f(t.value as f64);
    // A live gesture replaces the whole sampled transform for its clip.
    let xf = match overrides.transform {
        Some((id, xf)) if id == clip.id => xf,
        _ => {
            let base = clip.transform.sample(local_tick);
            apply_look_animations(clip, base, local_tick, local_tick_f, t.rate)
        }
    };

    // `position` is the anchor's offset from the canvas center, as a fraction
    // of canvas width/height. The layer carries the anchor position plus the
    // normalized `anchor_point`; the renderer derives the quad center once the
    // final pixel size is known (identity for the default center anchor).
    let center = [cw * (0.5 + xf.position[0]), ch * (0.5 + xf.position[1])];
    let anchor_point = xf.anchor_point;
    let rotation = xf.rotation.to_radians();
    let opacity = xf.opacity.clamp(0.0, 1.0);
    let uv = crop_flip_uv(clip);
    let effects = resolve_effects(clip, local_tick);
    let (filter, adjust) = match overrides.look {
        Some((id, filter, adjust)) if id == clip.id => (filter, adjust),
        _ => (clip.filter.as_ref(), &clip.adjust),
    };
    let color_grade = resolve_color_grade(filter, adjust);
    // File-backed `.cube` LUT (applied after the grade). Zero intensity is
    // identity — drop it here so downstream stages keep their fast paths.
    let lut = clip
        .lut
        .as_ref()
        .filter(|l| l.intensity > 0.0)
        .map(|l| SceneLut {
            path: l.path.clone(),
            intensity: l.intensity,
        });

    match &clip.content {
        ClipSource::Media { media, .. } => {
            let Some(src) = project.media(*media) else {
                return Ok(None);
            };
            // Both picture kinds aspect-fit into the canvas at their probed
            // size; audio-only media places nothing.
            let source = match src.kind() {
                MediaKind::Video => {
                    let Some(source_time) = clip.source_time_at(t)? else {
                        return Ok(None);
                    };
                    LayerSource::Media {
                        media: *media,
                        source_time,
                    }
                }
                // One frame for the clip's whole extent: no source time, and
                // retime/reverse are irrelevant by construction.
                MediaKind::Image => LayerSource::Still { media: *media },
                MediaKind::Audio => return Ok(None),
            };
            let fit = fit_scale(src.width as f32, src.height as f32, cw, ch);
            let size = SizeSpec::Fixed([
                src.width as f32 * fit * xf.scale,
                src.height as f32 * fit * xf.scale,
            ]);
            Ok(Some(SceneLayer {
                clip: Some(clip.id),
                source,
                center,
                anchor_point,
                size,
                rotation,
                opacity,
                uv,
                effects,
                mask: clip.mask,
                chroma_key: clip.chroma_key,
                color_grade,
                lut,
            }))
        }
        ClipSource::Generated(generator) => {
            // A live inspector edit replaces the clip's generator content.
            let generator = match overrides.generator {
                Some((id, live)) if id == clip.id => live,
                _ => generator,
            };
            Ok(resolve_generator(
                generator,
                center,
                anchor_point,
                rotation,
                opacity,
                uv,
                color_grade,
                lut,
                cw,
                ch,
                xf.scale,
                local_tick,
                local_tick as f64 * t.rate.seconds_per_unit(),
                effects,
            )
            .map(|mut layer| {
                layer.clip = Some(clip.id);
                layer
            }))
        }
    }
}

/// Sample `clip.effects` at clip-local `tick` into compositor-ready passes.
fn resolve_effects(clip: &cutlass_models::Clip, tick: i64) -> Vec<ResolvedPass> {
    let tick_f = tick as f64;
    clip.effects
        .iter()
        .filter_map(|fx| pack_effect(fx, tick_f).ok())
        .collect()
}

fn pack_effect(fx: &EffectInstance, tick: f64) -> Result<ResolvedPass, cutlass_models::ModelError> {
    let spec = fx.spec()?;
    let mut params = Vec::with_capacity(spec.params.len());
    for pspec in spec.params {
        let value = fx.sample_param(pspec.name, tick).unwrap_or(pspec.default);
        params.push(value);
    }
    Ok(ResolvedPass {
        id: fx.effect_id.clone(),
        params,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_generator(
    generator: &Generator,
    center: [f32; 2],
    anchor_point: [f32; 2],
    rotation: f32,
    opacity: f32,
    uv: [f32; 4],
    color_grade: Option<ColorGrade>,
    lut: Option<SceneLut>,
    cw: f32,
    ch: f32,
    scale: f32,
    tick: i64,
    local_seconds: f64,
    effects: Vec<ResolvedPass>,
) -> Option<SceneLayer> {
    let ref_scale = ch / REFERENCE_HEIGHT;
    let has_lut = lut.is_some();
    let mut layer = match generator {
        Generator::Text { content, style } => {
            let text = style.case.apply(content);
            if text.trim().is_empty() {
                return None;
            }
            Some(SceneLayer {
                clip: None,
                source: LayerSource::Text {
                    content: text,
                    style: map_text_style(style, cw, ch),
                },
                center,
                anchor_point,
                size: SizeSpec::BitmapScaled(scale),
                rotation,
                opacity,
                uv,
                effects,
                mask: None,
                chroma_key: None,
                color_grade,
                lut: None,
            })
        }
        Generator::SolidColor { rgba } => Some(SceneLayer {
            clip: None,
            source: LayerSource::Solid(*rgba),
            center,
            anchor_point,
            size: SizeSpec::Fixed([cw * scale, ch * scale]),
            rotation,
            opacity,
            uv,
            effects,
            mask: None,
            chroma_key: None,
            color_grade,
            lut: None,
        }),
        Generator::Shape {
            shape,
            rgba,
            width,
            height,
            corner_radius,
            stroke,
        } => resolve_shape(
            shape,
            rgba,
            width,
            height,
            corner_radius,
            stroke.as_ref(),
            tick,
            ref_scale * scale,
            center,
            anchor_point,
            rotation,
            opacity,
            uv,
            color_grade,
            scale,
            effects,
        ),
        Generator::Effect => canvas_pass(effects, None, has_lut, cw, ch),
        Generator::Filter | Generator::Adjustment => {
            canvas_pass(Vec::new(), color_grade, has_lut, cw, ch)
        }
        Generator::Lottie {
            path,
            width,
            height,
        } => {
            // Same placement convention as stickers: intrinsic pixels are
            // reference pixels. The stored size drives placement so this
            // stays pure — the renderer probes the file itself.
            let px = ref_scale * scale;
            Some(SceneLayer {
                clip: None,
                source: LayerSource::Lottie {
                    path: path.clone(),
                    local_time: local_seconds,
                },
                center,
                anchor_point,
                size: SizeSpec::Fixed([*width as f32 * px, *height as f32 * px]),
                rotation,
                opacity,
                uv,
                effects,
                mask: None,
                chroma_key: None,
                color_grade,
                lut: None,
            })
        }
        Generator::Sticker { asset } => {
            // Unknown/empty ids place nothing — the legacy payload-less
            // sticker behavior, never an error.
            let spec = cutlass_models::sticker_spec(asset)?;
            // Intrinsic pixels are *reference pixels* (1080p canvas), the
            // same convention as shapes: a 256 px sticker lands at a
            // CapCut-like overlay size and samples ~1:1 instead of being
            // blown up to canvas height like aspect-fit media.
            let px = ref_scale * scale;
            Some(SceneLayer {
                clip: None,
                source: LayerSource::Sticker {
                    asset: asset.clone(),
                    local_time: local_seconds,
                },
                center,
                anchor_point,
                size: SizeSpec::Fixed([spec.width as f32 * px, spec.height as f32 * px]),
                rotation,
                opacity,
                uv,
                effects,
                mask: None,
                chroma_key: None,
                color_grade,
                lut: None,
            })
        }
    }?;
    layer.lut = lut;
    Some(layer)
}

fn canvas_pass(
    effects: Vec<ResolvedPass>,
    color_grade: Option<ColorGrade>,
    has_lut: bool,
    cw: f32,
    ch: f32,
) -> Option<SceneLayer> {
    (!effects.is_empty() || color_grade.is_some() || has_lut).then_some(SceneLayer {
        clip: None,
        source: LayerSource::CanvasPass,
        center: [cw * 0.5, ch * 0.5],
        anchor_point: [0.5, 0.5],
        size: SizeSpec::Fixed([cw, ch]),
        rotation: 0.0,
        opacity: 1.0,
        uv: [0.0, 0.0, 1.0, 1.0],
        effects,
        mask: None,
        chroma_key: None,
        color_grade,
        lut: None,
    })
}

/// Resolve one shape generator at `tick` into a placed layer.
///
/// All `Param` curves are sampled here (the resolver is the "animation →
/// values" boundary), and every length is converted to canvas pixels with
/// `px_scale` (reference scale × the clip's animated transform scale), so
/// downstream stages see plain numbers. Parametric shapes become SDF layers
/// whose quad is padded for stroke overhang + anti-aliasing; pen paths become
/// CPU-raster layers that scale like text bitmaps.
#[allow(clippy::too_many_arguments)]
fn resolve_shape(
    shape: &Shape,
    rgba: &Param<[u8; 4]>,
    width: &Param<f32>,
    height: &Param<f32>,
    corner_radius: &Param<f32>,
    stroke: Option<&ShapeStroke>,
    tick: i64,
    px_scale: f32,
    center: [f32; 2],
    anchor_point: [f32; 2],
    rotation: f32,
    opacity: f32,
    uv: [f32; 4],
    color_grade: Option<ColorGrade>,
    transform_scale: f32,
    effects: Vec<ResolvedPass>,
) -> Option<SceneLayer> {
    let fill = rgba.sample(tick);
    let stroke_px = stroke.map(|s| Stroke {
        rgba: s.rgba.sample(tick),
        width: (s.width.sample(tick) * px_scale).max(0.0),
    });

    // Pen paths: rasterized on the CPU at the *reference* scale so the memo
    // stays warm under transform-scale animation (the quad magnifies the
    // bitmap, like text). `px_scale / transform_scale` recovers ref_scale.
    if let Shape::Path(path) = shape {
        let bezier = to_bezier(path);
        if !bezier.is_drawable() {
            return None;
        }
        let raster_scale = if transform_scale > 0.0 {
            px_scale / transform_scale
        } else {
            px_scale
        };
        return Some(SceneLayer {
            clip: None,
            source: LayerSource::PathShape {
                path: bezier,
                fill,
                // Raster-space stroke: `PathRaster` folds `raster_scale` into
                // the width itself, so hand it the unscaled model value.
                stroke: stroke.map(|s| Stroke {
                    rgba: s.rgba.sample(tick),
                    width: s.width.sample(tick).max(0.0),
                }),
                raster_scale,
            },
            center,
            anchor_point,
            size: SizeSpec::BitmapScaled(transform_scale),
            rotation,
            opacity,
            uv,
            effects,
            mask: None,
            chroma_key: None,
            color_grade,
            lut: None,
        });
    }

    let w = width.sample(tick) * px_scale;
    let h = height.sample(tick) * px_scale;
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let radius = (corner_radius.sample(tick) * px_scale).max(0.0);

    // Plain rectangles keep the no-texture solid fast path.
    if matches!(shape, Shape::Rectangle) && radius == 0.0 && stroke_px.is_none() {
        return Some(SceneLayer {
            clip: None,
            source: LayerSource::Solid(fill),
            center,
            anchor_point,
            size: SizeSpec::Fixed([w, h]),
            rotation,
            opacity,
            uv,
            effects,
            mask: None,
            chroma_key: None,
            color_grade,
            lut: None,
        });
    }

    let params = match shape {
        Shape::Rectangle => SdfParams::RoundedRect { radius },
        Shape::Ellipse => SdfParams::Ellipse,
        Shape::Polygon { sides } => SdfParams::polygon(*sides, radius),
        Shape::Star {
            points,
            inner_ratio,
        } => SdfParams::Star {
            points: *points,
            inner: inner_ratio.sample(tick).clamp(0.0, 1.0),
            round: radius,
        },
        Shape::Line => SdfParams::Line,
        Shape::Arrow => SdfParams::Arrow,
        Shape::Heart => SdfParams::Heart,
        Shape::Path(_) => unreachable!("handled above"),
    };

    // The quad must cover the stroke's outward half plus the AA ramp, or the
    // shader's ink clips at the quad edge (same margin as the CPU raster).
    let pad = stroke_px.map_or(0.0, |s| s.width * 0.5) + 2.0 * SDF_AA;
    Some(SceneLayer {
        clip: None,
        source: LayerSource::Shape {
            params,
            fill,
            stroke: stroke_px,
            pad,
        },
        center,
        anchor_point,
        size: SizeSpec::Fixed([w + 2.0 * pad, h + 2.0 * pad]),
        rotation,
        opacity,
        uv,
        effects,
        mask: None,
        chroma_key: None,
        color_grade,
        lut: None,
    })
}

/// Convert the model's serialized path into the shapes crate's bezier form.
fn to_bezier(path: &ShapePath) -> BezierPath {
    BezierPath {
        points: path
            .points
            .iter()
            .map(|p| PathPoint {
                anchor: p.anchor,
                handle_in: p.handle_in,
                handle_out: p.handle_out,
            })
            .collect(),
        closed: path.closed,
    }
}

/// Map a model [`ModelTextStyle`] onto a [`cutlass_text`] render style.
///
/// Font size, stroke width, and shadow distance are scaled from reference
/// (1080-tall) pixels to the canvas. Lossy for now: bold/italic/underline,
/// letter-spacing, wrap, and vertical alignment are not rendered yet.
///
/// The rasterized bitmap is kept tight to the glyphs (no canvas-width wrap):
/// the layer's placement centers it on the canvas, so passing the canvas width
/// as a wrap constraint here would only double-center the run. Multi-line wrap
/// within a fixed width is a follow-up.
fn map_text_style(style: &ModelTextStyle, _cw: f32, ch: f32) -> TextStyle {
    let scale = ch / REFERENCE_HEIGHT;
    let font_size = style.size * scale;
    let family = if style.font.is_empty() {
        FontFamily::SansSerif
    } else {
        FontFamily::Named(style.font.clone())
    };
    let align = match style.align_h {
        TextAlignH::Left => TextAlign::Left,
        TextAlignH::Center => TextAlign::Center,
        TextAlignH::Right => TextAlign::Right,
    };
    let mut mapped = TextStyle::new(font_size)
        .with_color(style.fill)
        .with_family(family)
        .with_align(align)
        .with_line_height(font_size * style.line_spacing);
    if let Some(stroke) = style.stroke {
        mapped = mapped.with_stroke(TextStroke {
            rgba: stroke.rgba,
            width: stroke.width * scale,
        });
    }
    if let Some(background) = style.background {
        mapped = mapped.with_background(TextBackground {
            rgba: background.rgba,
            radius: background.radius,
        });
    }
    if let Some(shadow) = style.shadow {
        mapped = mapped.with_shadow(TextShadow {
            rgba: shadow.rgba,
            // Fraction of font size — keep relative; rasterizer multiplies.
            blur: shadow.blur,
            distance: shadow.distance * scale,
        });
    }
    mapped
}

/// UV rect from a clip's crop, with axes reversed for mirror flags.
fn crop_flip_uv(clip: &cutlass_models::Clip) -> [f32; 4] {
    let c = clip.crop;
    let (mut u0, mut u1) = (c.x, c.x + c.w);
    let (mut v0, mut v1) = (c.y, c.y + c.h);
    if clip.flip_h {
        core::mem::swap(&mut u0, &mut u1);
    }
    if clip.flip_v {
        core::mem::swap(&mut v0, &mut v1);
    }
    [u0, v0, u1, v1]
}

/// Uniform "contain" scale fitting `nw`×`nh` content inside a `cw`×`ch` canvas.
fn fit_scale(nw: f32, nh: f32, cw: f32, ch: f32) -> f32 {
    if nw <= 0.0 || nh <= 0.0 {
        return 1.0;
    }
    (cw / nw).min(ch / nh)
}

#[cfg(test)]
mod tests;
