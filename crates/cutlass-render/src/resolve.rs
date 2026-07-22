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
//! - **Stickers & Lottie**: bundled stickers and Lottie compositions resolve
//!   to placed layers, sized by their intrinsic reference-pixel dimensions.

use cutlass_core::{RationalTime, resample};
use cutlass_models::{
    AnimationSlot, ClipId, ClipSource, ClipTransform, ColorAdjustments, Easing, EffectInstance,
    Filter, Generator, MediaKind, Project, look_animation_combo_period_ticks,
    look_animation_window_ticks,
};

use crate::animation::{apply_look_animations, is_per_character, scaled_ticks, text_knobs};
use crate::grade::resolve_color_grade_at;
use crate::scene::{
    LayerSource, ResolvedPass, Scene, SceneChromaKey, SceneLayer, SceneLut, SceneMask, SizeSpec,
    TextAnimation,
};

mod generator;
mod shape;

#[cfg(test)]
use generator::map_text_style;
pub(crate) use generator::resolve_generator;

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
                blend_mode: cutlass_models::BlendMode::Normal,
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
    let color_grade = resolve_color_grade_at(filter, adjust, local_tick);
    // File-backed `.cube` LUT (applied after the grade). Zero intensity is
    // identity — drop it here so downstream stages keep their fast paths.
    let lut = clip
        .lut
        .as_ref()
        .filter(|l| l.intensity.sample(local_tick) > 0.0)
        .map(|l| SceneLut {
            path: l.path.clone(),
            intensity: l.intensity.sample(local_tick),
        });
    let mask = clip.mask.as_ref().map(|mask| SceneMask {
        kind: mask.kind,
        feather: mask.feather.sample(local_tick),
        invert: mask.invert,
    });
    let chroma_key = clip.chroma_key.as_ref().map(|chroma| SceneChromaKey {
        rgb: chroma.rgb,
        strength: chroma.strength.sample(local_tick),
        shadow: chroma.shadow.sample(local_tick),
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
                mask,
                chroma_key,
                color_grade,
                lut,
                blend_mode: clip.blend_mode,
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
                // Canvas passes grade/effect the whole stack — blend modes
                // only apply to layer quads, so keep those Normal.
                if !matches!(layer.source, LayerSource::CanvasPass) {
                    layer.blend_mode = clip.blend_mode;
                }
                if let LayerSource::Text { animation, .. } = &mut layer.source {
                    *animation = sample_text_animation(clip, local_tick, local_tick_f, t.rate);
                }
                layer
            }))
        }
    }
}

/// Sample an active per-character look preset into resolve-time data.
fn sample_text_animation(
    clip: &cutlass_models::Clip,
    local_tick: i64,
    local_tick_f: f64,
    rate: cutlass_core::Rational,
) -> Option<TextAnimation> {
    let duration = clip.timeline.duration.value.max(1);
    let base_window = look_animation_window_ticks(duration, rate);

    if let Some(combo) = &clip.animation_combo {
        if is_per_character(&combo.id) {
            let period = scaled_ticks(look_animation_combo_period_ticks(rate), combo.speed);
            let phase = ((local_tick_f % period as f64) / period as f64).clamp(0.0, 1.0) as f32;
            let (intensity, stagger) = text_knobs(combo);
            return Some(TextAnimation {
                id: combo.id.clone(),
                slot: AnimationSlot::Combo,
                t: phase,
                intensity,
                stagger,
            });
        }
        return None;
    }

    if let Some(anim) = &clip.animation_in
        && is_per_character(&anim.id)
    {
        let window = scaled_ticks(base_window, anim.speed).min(duration);
        if local_tick < window {
            let raw = (local_tick_f / window as f64).clamp(0.0, 1.0) as f32;
            let eased = Easing::EaseOut.apply(raw);
            let (intensity, stagger) = text_knobs(anim);
            return Some(TextAnimation {
                id: anim.id.clone(),
                slot: AnimationSlot::In,
                t: eased,
                intensity,
                stagger,
            });
        }
    }

    if let Some(anim) = &clip.animation_out
        && is_per_character(&anim.id)
    {
        let window = scaled_ticks(base_window, anim.speed).min(duration);
        let out_start = duration - window;
        if local_tick >= out_start {
            let raw = ((local_tick_f - out_start as f64) / (window - 1).max(1) as f64)
                .clamp(0.0, 1.0) as f32;
            let eased = Easing::EaseIn.apply(raw);
            let (intensity, stagger) = text_knobs(anim);
            return Some(TextAnimation {
                id: anim.id.clone(),
                slot: AnimationSlot::Out,
                t: eased,
                intensity,
                stagger,
            });
        }
    }

    None
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
mod per_char_tests;
#[cfg(test)]
mod tests;
