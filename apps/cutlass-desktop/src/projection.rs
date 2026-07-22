//! Build the Slint view model (`crate::Project`) from the engine's authoritative
//! [`cutlass_models::Project`].
//!
//! The engine is the single source of truth; the Slint `EditorStore.project` is
//! a read-only projection of it. This runs on the UI thread (Slint model types
//! are `!Send`), fed a `Send` snapshot cloned off the engine thread.
//!
//! A few Slint fields are presentation-only and have no engine equivalent yet
//! (sequence name, drop-frame, per-lane clip color). Those are derived or
//! defaulted here; everything structural — tracks, clips, placement, fps,
//! canvas size — is read straight from the engine.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use cutlass_models::{
    Clip as EngineClip, ClipCapabilities as EngineCaps, ClipSource, Generator, Keyframe, Lerp,
    Marker as EngineMarker, MediaSource, Param, Project as EngineProject,
    Rational as EngineRational, RationalTime as EngineTime, TextAlignH, TextAlignV, TextCase,
    TextStyle as EngineTextStyle, TimeRange as EngineRange, Track as EngineTrack,
    TrackKind as EngineKind, rate_eq, resample,
};
use slint::{Color, ModelRc, VecModel};

use crate::params::easing_to_ui;
use crate::{
    Clip, ClipCapabilities, EffectParamView, EffectView, Media, ParamKeyframe, Project, Rational,
    RationalTime, Sequence, TextClipStyle, TimeRange, TimelineMarker, Track, TrackKind,
    TransitionView,
};

mod helpers;
mod keyframes;
mod look;

use helpers::*;
use keyframes::*;
use look::*;

/// Project the engine's project state into the Slint view model.
///
/// `generator_sizes` maps raw clip ids of generated clips to their
/// drawn-content size in canvas px (computed on the engine thread, where the
/// raster cache lives) — the preview's selection geometry needs it because
/// generators raster at full canvas size.
///
/// `missing_media` holds raw ids of pool entries whose backing file is gone
/// (computed worker-side — stat'ing here would block the UI thread on a dead
/// network mount). Flags the library tiles' missing badges and the relink
/// dialog rows.
pub fn project_to_slint(
    project: &EngineProject,
    generator_sizes: &HashMap<u64, (i32, i32)>,
    missing_media: &HashSet<u64>,
) -> Project {
    let timeline = project.timeline();
    let (width, height) = canvas_size(project);
    let canvas = timeline.canvas();

    // The engine stacks bottom→top (last track composites in front); the lane
    // list shows the stack top-first so the top lane is the front layer, like
    // CapCut/Premiere. UI row r ↔ engine order index (track_count - 1 - r).
    let mut tracks: Vec<Track> = timeline
        .tracks_ordered()
        .filter(|track| kind_visible(track.kind))
        .map(|track| track_to_slint(project, track, generator_sizes))
        .collect();
    tracks.reverse();

    let id = project.id.raw().to_string();

    let usage = media_usage_counts(project);
    let pool = media_pool(project, missing_media, &usage);
    // Audio-only subset for the library's Audio > Local section — projected
    // here because Slint's `for` can't filter a model. `Media` clones are
    // cheap (the thumbnail is a refcounted image handle).
    let audio_pool: Vec<Media> = pool.iter().filter(|m| m.is_audio).cloned().collect();

    Project {
        id: id.clone().into(),
        title: project.name.clone().into(),
        sequence: Sequence {
            id: id.into(),
            name: "Sequence 1".into(),
            fps: rational(timeline.frame_rate),
            drop_frame: false,
            width,
            height,
            tracks: model(tracks),
            markers: model(
                timeline
                    .markers()
                    .iter()
                    .map(marker_to_slint)
                    .collect::<Vec<_>>(),
            ),
            aspect_index: aspect_to_index(canvas.aspect),
            background: Color::from_rgb_u8(
                canvas.background[0],
                canvas.background[1],
                canvas.background[2],
            ),
        },
        media: model(pool),
        media_audio: model(audio_pool),
        agent_rules: project.metadata().agent_rules.clone().into(),
    }
}

/// Count, per pool entry, how many timeline clips reference it — in one pass
/// over every clip (O(clips), not O(media × clips)). Drives the library tile's
/// delete confirmation; only media-backed clips carry a `MediaId`.
fn media_usage_counts(project: &EngineProject) -> HashMap<u64, i32> {
    let mut counts: HashMap<u64, i32> = HashMap::new();
    for track in project.timeline().tracks_ordered() {
        for clip in track.clips() {
            if let Some(media) = clip.media() {
                *counts.entry(media.raw()).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// The media pool as Library bin entries, ordered by id (the engine's pool is a
/// hash map, so a stable sort keeps tile order from jumping between imports).
fn media_pool(
    project: &EngineProject,
    missing_media: &HashSet<u64>,
    usage: &HashMap<u64, i32>,
) -> Vec<Media> {
    let tl_rate = project.timeline().frame_rate;
    let mut sources: Vec<&MediaSource> = project.media_iter().collect();
    sources.sort_by_key(|media| media.id.raw());
    sources
        .into_iter()
        .map(|media| {
            media_to_slint(
                media,
                tl_rate,
                missing_media.contains(&media.id.raw()),
                usage.get(&media.id.raw()).copied().unwrap_or(0),
            )
        })
        .collect()
}

fn media_to_slint(
    media: &MediaSource,
    tl_rate: cutlass_models::Rational,
    is_missing: bool,
    usage_count: i32,
) -> Media {
    Media {
        id: media.id.raw().to_string().into(),
        name: media_name(media).into(),
        path: media.path().display().to_string().into(),
        is_missing,
        width: media.width as i32,
        height: media.height as i32,
        has_audio: media.has_audio,
        duration_ticks: clamp_i32(resample(media.duration, tl_rate).value),
        is_audio: media.is_audio_only(),
        is_image: media.is_image,
        duration_label: duration_label(media.duration).into(),
        usage_count,
        // Generated asynchronously after import; until it lands the tile shows
        // its placeholder card (see src/thumbnails.rs).
        thumbnail: crate::thumbnails::thumbnail_for(media.id.raw()).unwrap_or_default(),
    }
}

/// Source length as `MM:SS` (or `H:MM:SS` from one hour up), CapCut-style.
fn duration_label(duration: EngineTime) -> String {
    let (num, den) = (i64::from(duration.rate.num), i64::from(duration.rate.den));
    if num <= 0 || den <= 0 {
        return String::new();
    }
    let secs = (duration.value.max(0) * den + num / 2) / num;
    let (h, m, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// File stem of the source, falling back to the id when the path has none.
fn media_name(media: &MediaSource) -> String {
    media
        .path()
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Media {}", media.id.raw()))
}

fn track_to_slint(
    project: &EngineProject,
    track: &EngineTrack,
    generator_sizes: &HashMap<u64, (i32, i32)>,
) -> Track {
    let clips: Vec<Clip> = track
        .clips_ordered()
        .into_iter()
        .map(|clip| clip_to_slint(project, clip, track.kind, generator_sizes))
        .collect();

    Track {
        id: track.id.raw().to_string().into(),
        name: track.name.clone().into(),
        kind: track_kind(track.kind),
        color: kind_color(track.kind),
        clips: model(clips),
        enabled: track.enabled,
        muted: track.muted,
        locked: track.locked,
        duck_source: track.duck_source,
        pinned: track.pinned,
        is_main: track.main,
        transitions: project_transitions(track),
    }
}

fn clip_to_slint(
    project: &EngineProject,
    clip: &EngineClip,
    track_kind: EngineKind,
    generator_sizes: &HashMap<u64, (i32, i32)>,
) -> Clip {
    // The timeline UI positions a clip at `timeline-start` and derives its width
    // from `source-range.duration`, both in sequence ticks. The engine's
    // authoritative on-sequence placement is `clip.timeline`, so mirror it here
    // (1:1 playback; the true media in/out isn't needed until time-remap or a
    // live inspector requires it).
    let (name, text_content) = clip_labels(project, clip);
    let (generator_kind, fill_color) = clip_generator_visual(clip);
    let (head_room, tail_room) = trim_rooms(project, clip);
    let (media_id, source_in_s) = match &clip.content {
        ClipSource::Media { media, source } => (
            media.raw().to_string(),
            time_to_seconds(source.start) as f32,
        ),
        ClipSource::Generated(_) => (String::new(), 0.0),
    };
    // Whether the clip's source carries sound — drives the embedded waveform on
    // video clips (CapCut keeps a video's audio on the clip itself).
    let has_audio = match &clip.content {
        ClipSource::Media { media, .. } => project.media(*media).is_some_and(|m| m.has_audio),
        ClipSource::Generated(_) => false,
    };
    // Natural content size for preview placement: the media's native pixels
    // (aspect-fit into the canvas), or a generator's drawn-content bounds in
    // canvas px (fit 1:1). 0×0 ⇔ unknown — the selection geometry falls back
    // to a canvas-sized box. Media that vanished from the pool degrades the
    // same way.
    let (media_w, media_h) = match &clip.content {
        ClipSource::Media { media, .. } => project
            .media(*media)
            .map(|m| (m.width as i32, m.height as i32))
            .unwrap_or((0, 0)),
        ClipSource::Generated(_) => generator_sizes
            .get(&clip.id.raw())
            .copied()
            .unwrap_or((0, 0)),
    };
    let transform = clip.transform.sample(0);
    let clip_start = clip.timeline.start.value;
    let (shape_width, shape_height) = clip_shape_size(clip);
    let (filter_id, filter_label, filter_intensity) = clip_filter(clip);
    let (lut_id, lut_label, lut_path, lut_intensity) = clip_lut(clip);
    let blend_mode = clip.blend_mode.id().to_string();
    let blend_label = clip.blend_mode.label().to_string();
    let mask = clip_mask(clip, clip_start);
    let chroma = clip_chroma(clip, clip_start);
    let styles = clip_layer_styles(clip, clip_start);
    let animation_in = clip_animation(clip.animation_in.as_ref());
    let animation_out = clip_animation(clip.animation_out.as_ref());
    let animation_combo = clip_animation(clip.animation_combo.as_ref());
    let caps = clip_capabilities(project, clip, track_kind);

    Clip {
        id: clip.id.raw().to_string().into(),
        name: name.into(),
        timeline_start: rational_time(clip.timeline.start),
        source_range: time_range(clip.timeline),
        media_id: media_id.into(),
        source_in_s,
        duration_label: clip_duration_label(clip.timeline.duration).into(),
        speed: speed_factor(clip.speed),
        reversed: clip.reversed,
        speed_label: speed_label(clip).into(),
        preserve_pitch: clip.preserve_pitch,
        // The clip-start sample: the constant gain for a flat clip; the
        // envelope start for an animated one (the inspector samples the
        // published curve UI-side for playhead accuracy).
        volume: clip.volume.sample(0),
        pan: clip.pan.sample(0),
        fade_in_s: time_to_seconds(EngineTime::new(clip.fade_in, clip.timeline.start.rate)) as f32,
        fade_out_s: time_to_seconds(EngineTime::new(clip.fade_out, clip.timeline.start.rate))
            as f32,
        denoise: clip.denoise,
        has_audio,
        text_content: text_content.into(),
        text_style: clip_text_style(clip),
        generator_kind: generator_kind.into(),
        fill_color,
        shape_width,
        shape_height,
        head_room_ticks: head_room,
        tail_room_ticks: tail_room,
        link_id: clip
            .link
            .map(|link| link.raw().to_string())
            .unwrap_or_default()
            .into(),
        media_width: media_w,
        media_height: media_h,
        // The clip-start sample: exact for constant properties; animated
        // properties additionally publish their curve below, and consumers
        // that need playhead accuracy sample it UI-side (src/params.rs).
        transform_position_x: transform.position[0],
        transform_position_y: transform.position[1],
        transform_anchor_x: transform.anchor_point[0],
        transform_anchor_y: transform.anchor_point[1],
        transform_scale: transform.scale.x,
        transform_scale_y: transform.scale.y,
        transform_scale_linked: transform.scale.is_uniform(),
        transform_rotation: transform.rotation,
        transform_opacity: transform.opacity,
        // Clip-start sample (same convention as transform/volume above).
        // Playhead-accurate crop sampling is UI-side; no kf-crop list yet
        // (crop inspector is constant-commit / gizmo-only).
        crop_x: clip.crop.sample(0).x,
        crop_y: clip.crop.sample(0).y,
        crop_w: clip.crop.sample(0).w,
        crop_h: clip.crop.sample(0).h,
        flip_h: clip.flip_h,
        flip_v: clip.flip_v,
        blend_mode: blend_mode.into(),
        blend_label: blend_label.into(),
        motion_blur_enabled: clip.motion_blur.enabled,
        motion_blur_shutter: clip.motion_blur.shutter_deg,
        motion_blur_samples: clip.motion_blur.samples as i32,
        mask_kind: mask.kind.into(),
        mask_label: mask.label.into(),
        mask_invert: mask.invert,
        mask_feather: mask.feather,
        mask_center_x: mask.center[0],
        mask_center_y: mask.center[1],
        mask_size_w: mask.size[0],
        mask_size_h: mask.size[1],
        mask_rotation: mask.rotation,
        mask_roundness: mask.roundness,
        chroma_enabled: chroma.enabled,
        chroma_color: chroma.color,
        chroma_strength: chroma.strength,
        chroma_shadow: chroma.shadow,
        style_shadow_enabled: styles.shadow_enabled,
        style_shadow_color: styles.shadow_color,
        style_shadow_offset_x: styles.shadow_offset[0],
        style_shadow_offset_y: styles.shadow_offset[1],
        style_shadow_blur: styles.shadow_blur,
        style_glow_enabled: styles.glow_enabled,
        style_glow_color: styles.glow_color,
        style_glow_radius: styles.glow_radius,
        style_glow_intensity: styles.glow_intensity,
        style_outline_enabled: styles.outline_enabled,
        style_outline_color: styles.outline_color,
        style_outline_width: styles.outline_width,
        style_background_enabled: styles.background_enabled,
        style_background_color: styles.background_color,
        style_background_padding: styles.background_padding,
        style_background_radius: styles.background_radius,
        filter_id: filter_id.into(),
        filter_label: filter_label.into(),
        filter_intensity,
        adjust_brightness: clip.adjust.brightness.sample(0),
        adjust_contrast: clip.adjust.contrast.sample(0),
        adjust_saturation: clip.adjust.saturation.sample(0),
        adjust_exposure: clip.adjust.exposure.sample(0),
        adjust_temperature: clip.adjust.temperature.sample(0),
        adjust_tint: clip.adjust.tint.sample(0),
        adjust_hue: clip.adjust.hue.sample(0),
        adjust_highlights: clip.adjust.highlights.sample(0),
        adjust_shadows: clip.adjust.shadows.sample(0),
        adjust_sharpness: clip.adjust.sharpness.sample(0),
        adjust_vignette: clip.adjust.vignette.sample(0),
        lut_id: lut_id.into(),
        lut_label: lut_label.into(),
        lut_path: lut_path.into(),
        lut_intensity,
        animation_in_id: animation_in.id.into(),
        animation_in_label: animation_in.label.into(),
        animation_in_speed: animation_in.speed,
        animation_in_intensity: animation_in.intensity,
        animation_in_stagger: animation_in.stagger,
        animation_in_has_speed: animation_in.has_speed,
        animation_in_has_intensity: animation_in.has_intensity,
        animation_in_has_stagger: animation_in.has_stagger,
        animation_out_id: animation_out.id.into(),
        animation_out_label: animation_out.label.into(),
        animation_out_speed: animation_out.speed,
        animation_out_intensity: animation_out.intensity,
        animation_out_stagger: animation_out.stagger,
        animation_out_has_speed: animation_out.has_speed,
        animation_out_has_intensity: animation_out.has_intensity,
        animation_out_has_stagger: animation_out.has_stagger,
        animation_combo_id: animation_combo.id.into(),
        animation_combo_label: animation_combo.label.into(),
        animation_combo_speed: animation_combo.speed,
        animation_combo_intensity: animation_combo.intensity,
        animation_combo_stagger: animation_combo.stagger,
        animation_combo_has_speed: animation_combo.has_speed,
        animation_combo_has_intensity: animation_combo.has_intensity,
        animation_combo_has_stagger: animation_combo.has_stagger,
        kf_position: keyframes_to_slint(&clip.transform.position, clip_start, |v| (v[0], v[1])),
        kf_anchor: keyframes_to_slint(&clip.transform.anchor_point, clip_start, |v| (v[0], v[1])),
        kf_scale: keyframes_to_slint(&clip.transform.scale, clip_start, |v| (v.x, v.y)),
        kf_rotation: keyframes_to_slint(&clip.transform.rotation, clip_start, |v| (*v, 0.0)),
        kf_opacity: keyframes_to_slint(&clip.transform.opacity, clip_start, |v| (*v, 0.0)),
        kf_text_size: text_keyframes(clip, clip_start, |style| &style.size),
        kf_text_fill: text_color_keyframes(clip, clip_start, |style| &style.fill),
        kf_text_letter_spacing: text_keyframes(clip, clip_start, |style| &style.letter_spacing),
        kf_text_line_spacing: text_keyframes(clip, clip_start, |style| &style.line_spacing),
        kf_text_stroke_width: text_stroke_keyframes(clip, clip_start, |stroke| &stroke.width),
        kf_text_stroke_color: text_stroke_color_keyframes(clip, clip_start, |stroke| &stroke.rgba),
        kf_text_background_color: text_background_color_keyframes(clip, clip_start, |bg| &bg.rgba),
        kf_text_background_radius: text_background_keyframes(clip, clip_start, |bg| &bg.radius),
        kf_text_shadow_blur: text_shadow_keyframes(clip, clip_start, |shadow| &shadow.blur),
        kf_text_shadow_distance: text_shadow_keyframes(clip, clip_start, |shadow| &shadow.distance),
        kf_text_shadow_color: text_shadow_color_keyframes(clip, clip_start, |shadow| &shadow.rgba),
        kf_look_filter_intensity: clip.filter.as_ref().map_or_else(empty_keyframes, |filter| {
            keyframes_to_slint(&filter.intensity, clip_start, |v| (*v, 0.0))
        }),
        kf_look_lut_intensity: clip.lut.as_ref().map_or_else(empty_keyframes, |lut| {
            keyframes_to_slint(&lut.intensity, clip_start, |v| (*v, 0.0))
        }),
        kf_look_adjust_brightness: keyframes_to_slint(&clip.adjust.brightness, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_contrast: keyframes_to_slint(&clip.adjust.contrast, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_saturation: keyframes_to_slint(&clip.adjust.saturation, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_exposure: keyframes_to_slint(&clip.adjust.exposure, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_temperature: keyframes_to_slint(&clip.adjust.temperature, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_tint: keyframes_to_slint(&clip.adjust.tint, clip_start, |v| (*v, 0.0)),
        kf_look_adjust_hue: keyframes_to_slint(&clip.adjust.hue, clip_start, |v| (*v, 0.0)),
        kf_look_adjust_highlights: keyframes_to_slint(&clip.adjust.highlights, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_shadows: keyframes_to_slint(&clip.adjust.shadows, clip_start, |v| (*v, 0.0)),
        kf_look_adjust_sharpness: keyframes_to_slint(&clip.adjust.sharpness, clip_start, |v| {
            (*v, 0.0)
        }),
        kf_look_adjust_vignette: keyframes_to_slint(&clip.adjust.vignette, clip_start, |v| {
            (*v, 0.0)
        }),
        // Layer-style scalar / vec2 curves. Color style params intentionally
        // omit kf lists (color keyframes stay AI-only for now).
        kf_style_shadow_offset: styles.kf_shadow_offset,
        kf_style_shadow_blur: styles.kf_shadow_blur,
        kf_style_glow_radius: styles.kf_glow_radius,
        kf_style_glow_intensity: styles.kf_glow_intensity,
        kf_style_outline_width: styles.kf_outline_width,
        kf_style_background_padding: styles.kf_background_padding,
        kf_style_background_radius: styles.kf_background_radius,
        kf_look_mask_feather: mask.kf_feather,
        kf_look_mask_center: mask.kf_center,
        kf_look_mask_size: mask.kf_size,
        kf_look_mask_rotation: mask.kf_rotation,
        kf_look_mask_roundness: mask.kf_roundness,
        kf_look_chroma_strength: chroma.kf_strength,
        kf_look_chroma_shadow: chroma.kf_shadow,
        kf_speed_curve: speed_curve_to_slint(&clip.speed_curve),
        has_speed_curve: clip.has_speed_curve(),
        speed_curve_avg: clip.speed_curve_average() as f32,
        speed_curve_samples: speed_curve_samples(clip),
        // Volume automation (M8): the envelope as absolute-tick keyframes
        // (transform pattern), plus a normalized path string for the on-clip
        // automation line.
        kf_volume: keyframes_to_slint(&clip.volume, clip_start, |v| (*v, 0.0)),
        has_volume_envelope: clip.has_volume_envelope(),
        volume_path: volume_path(clip).into(),
        kf_pan: keyframes_to_slint(&clip.pan, clip_start, |v| (*v, 0.0)),
        has_pan_envelope: clip.has_pan_envelope(),
        effects: project_effects(clip),
        // Beat markers (M8 Phase 6) as absolute sequence ticks, already
        // filtered to the clip's current window by the model helper.
        beats: model(
            clip.beat_timeline_ticks()
                .into_iter()
                .map(clamp_i32)
                .collect(),
        ),
        caps,
    }
}

/// Project a clip's speed ramp keyframes (M2 speed curves) as the inspector's
/// draggable graph handles. Unlike transform keyframes, ramp ticks stay in
/// their NORMALIZED domain (`0..=SPEED_CURVE_SCALE`) — no `clip_start` offset
/// — because the curve is defined over the clip's span, not the sequence.
/// Empty ⇔ a flat constant-speed clip.
fn speed_curve_to_slint(curve: &Param<f32>) -> ModelRc<ParamKeyframe> {
    let rows: Vec<ParamKeyframe> = curve
        .keyframes()
        .iter()
        .map(|kf: &Keyframe<f32>| {
            let (easing, [bez_x1, bez_y1, bez_x2, bez_y2]) = easing_to_ui(kf.easing);
            ParamKeyframe {
                tick: clamp_i32(kf.tick),
                value_x: kf.value,
                value_y: 0.0,
                easing,
                bez_x1,
                bez_y1,
                bez_x2,
                bez_y2,
                has_tangents: false,
                out_tx: 0.0,
                out_ty: 0.0,
                in_tx: 0.0,
                in_ty: 0.0,
            }
        })
        .collect();
    model(rows)
}

/// Number of polyline samples the inspector velocity graph plots across a
/// ramp. Odd so the midpoint lands on a sample; cheap enough to recompute on
/// every projection republish (only clips that actually carry a ramp pay).
const SPEED_GRAPH_SAMPLES: usize = 49;

/// Dense, evenly-spaced multiplier samples of a clip's speed ramp across its
/// normalized span (engine `Param` math, so easing curvature shows). Empty
/// for a flat clip — the graph then just draws the 1.0× baseline.
fn speed_curve_samples(clip: &EngineClip) -> ModelRc<f32> {
    if !clip.has_speed_curve() {
        return model(Vec::new());
    }
    let last = (SPEED_GRAPH_SAMPLES - 1) as f64;
    let scale = cutlass_models::SPEED_CURVE_SCALE as f64;
    let rows: Vec<f32> = (0..SPEED_GRAPH_SAMPLES)
        .map(|i| {
            let tick = (i as f64 / last) * scale;
            clip.speed_curve.sample_at(tick)
        })
        .collect();
    model(rows)
}

/// The full gain a slider/automation line maps to (200%, the inspector's
/// volume max), so 100% sits mid-band and a boost still has headroom.
const VOLUME_GRAPH_MAX: f64 = 2.0;

/// A clip's volume envelope as an SVG path-commands string for the on-clip
/// automation line: dense, evenly-spaced samples (engine `Param` math, so
/// easing shows) in a normalized 1000×1000 viewbox — x runs clip start→end,
/// y is the gain top-down (0 at the top of the band, `VOLUME_GRAPH_MAX` at
/// the bottom), clamped into the band. Empty for a constant-gain clip — the
/// card then draws no line.
fn volume_path(clip: &EngineClip) -> String {
    if !clip.has_volume_envelope() {
        return String::new();
    }
    let span = (clip.timeline.duration.value - 1).max(0) as f64;
    let last = (SPEED_GRAPH_SAMPLES - 1) as f64;
    let mut path = String::with_capacity(SPEED_GRAPH_SAMPLES * 14);
    for i in 0..SPEED_GRAPH_SAMPLES {
        let tick = ((i as f64 / last) * span).round() as i64;
        let gain = f64::from(clip.volume.sample(tick));
        let x = (i as f64 / last) * 1000.0;
        let y = (1.0 - (gain / VOLUME_GRAPH_MAX).clamp(0.0, 1.0)) * 1000.0;
        if i == 0 {
            path.push_str(&format!("M {x:.1} {y:.1}"));
        } else {
            path.push_str(&format!(" L {x:.1} {y:.1}"));
        }
    }
    path
}

/// Project a clip's effect chain (M4) for the inspector Effects section, each
/// parameter sampled at the clip start with its catalog label, kind, and range.
fn project_effects(clip: &EngineClip) -> ModelRc<EffectView> {
    use cutlass_models::EffectParamKind;

    let rows: Vec<EffectView> = clip
        .effects
        .iter()
        .map(|fx| {
            let spec = cutlass_models::effect_spec(&fx.effect_id);
            let label = spec.map(|s| s.label).unwrap_or(fx.effect_id.as_str());
            let params: Vec<EffectParamView> = spec
                .map(|spec| {
                    spec.params
                        .iter()
                        .map(|p| {
                            let (kind, value, color, vec2) = match p.kind {
                                EffectParamKind::Scalar => (
                                    "scalar",
                                    fx.sample_param(p.name, 0.0).unwrap_or(p.default),
                                    [0, 0, 0, 0],
                                    [0.0, 0.0],
                                ),
                                EffectParamKind::Color => (
                                    "color",
                                    0.0,
                                    fx.sample_color_param(p.name, 0.0)
                                        .unwrap_or(p.default_color),
                                    [0.0, 0.0],
                                ),
                                EffectParamKind::Vec2 => (
                                    "vec2",
                                    0.0,
                                    [0, 0, 0, 0],
                                    fx.sample_vec2_param(p.name, 0.0).unwrap_or(p.default_vec2),
                                ),
                            };
                            EffectParamView {
                                name: p.name.into(),
                                label: p.label.into(),
                                kind: kind.into(),
                                value,
                                min: p.min,
                                max: p.max,
                                color: rgba_color(color),
                                vec2_x: vec2[0],
                                vec2_y: vec2[1],
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            EffectView {
                effect_id: fx.effect_id.clone().into(),
                label: label.into(),
                params: model(params),
            }
        })
        .collect();
    model(rows)
}

/// Project a track's transitions (M4) for the timeline junction pills, with
/// the absolute cut tick (the left clip's end) and the catalog label.
fn project_transitions(track: &EngineTrack) -> ModelRc<TransitionView> {
    let rows: Vec<TransitionView> = track
        .transitions()
        .iter()
        .filter_map(|t| {
            let cut = track.clip(t.left)?.timeline.end_tick();
            let label = cutlass_models::transition_spec(&t.transition_id)
                .map(|s| s.label)
                .unwrap_or(t.transition_id.as_str());
            Some(TransitionView {
                left_clip_id: t.left.raw().to_string().into(),
                transition_id: t.transition_id.clone().into(),
                label: label.into(),
                duration_ticks: clamp_i32(t.duration),
                cut_tick: clamp_i32(cut),
            })
        })
        .collect();
    model(rows)
}

#[cfg(test)]
mod tests;
