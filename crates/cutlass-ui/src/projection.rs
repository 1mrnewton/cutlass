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
    Clip as EngineClip, ClipSource, Generator, Keyframe, Lerp, Marker as EngineMarker,
    MediaSource, Param, Project as EngineProject,
    Rational as EngineRational, RationalTime as EngineTime, TextAlignH, TextAlignV, TextCase,
    TextStyle as EngineTextStyle, TimeRange as EngineRange, Track as EngineTrack,
    TrackKind as EngineKind, rate_eq, resample,
};
use slint::{Color, ModelRc, VecModel};

use crate::params::easing_to_ui;
use crate::{
    Clip, EffectParamView, EffectView, Media, ParamKeyframe, Project, Rational, RationalTime,
    Sequence, TextClipStyle, TimeRange, TimelineMarker, Track, TrackKind, TransitionView,
};


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

    let pool = media_pool(project, missing_media);
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
    }
}

/// The media pool as Library bin entries, ordered by id (the engine's pool is a
/// hash map, so a stable sort keeps tile order from jumping between imports).
fn media_pool(project: &EngineProject, missing_media: &HashSet<u64>) -> Vec<Media> {
    let tl_rate = project.timeline().frame_rate;
    let mut sources: Vec<&MediaSource> = project.media_iter().collect();
    sources.sort_by_key(|media| media.id.raw());
    sources
        .into_iter()
        .map(|media| media_to_slint(media, tl_rate, missing_media.contains(&media.id.raw())))
        .collect()
}

fn media_to_slint(media: &MediaSource, tl_rate: cutlass_models::Rational, is_missing: bool) -> Media {
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
        // Generated asynchronously after import; until then the tile shows
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
        .map(|clip| clip_to_slint(project, clip, generator_sizes))
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
        transitions: project_transitions(track),
    }
}

fn clip_to_slint(
    project: &EngineProject,
    clip: &EngineClip,
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
        ClipSource::Media { media, source } => {
            (media.raw().to_string(), time_to_seconds(source.start) as f32)
        }
        ClipSource::Generated(_) => (String::new(), 0.0),
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
        fade_in_s: time_to_seconds(EngineTime::new(clip.fade_in, clip.timeline.start.rate)) as f32,
        fade_out_s: time_to_seconds(EngineTime::new(clip.fade_out, clip.timeline.start.rate))
            as f32,
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
        transform_scale: transform.scale,
        transform_rotation: transform.rotation,
        transform_opacity: transform.opacity,
        crop_x: clip.crop.x,
        crop_y: clip.crop.y,
        crop_w: clip.crop.w,
        crop_h: clip.crop.h,
        flip_h: clip.flip_h,
        flip_v: clip.flip_v,
        kf_position: keyframes_to_slint(&clip.transform.position, clip_start, |v| (v[0], v[1])),
        kf_scale: keyframes_to_slint(&clip.transform.scale, clip_start, |v| (*v, 0.0)),
        kf_rotation: keyframes_to_slint(&clip.transform.rotation, clip_start, |v| (*v, 0.0)),
        kf_opacity: keyframes_to_slint(&clip.transform.opacity, clip_start, |v| (*v, 0.0)),
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
        effects: project_effects(clip),
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
/// parameter sampled at the clip start with its catalog label and range.
fn project_effects(clip: &EngineClip) -> ModelRc<EffectView> {
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
                        .map(|p| EffectParamView {
                            name: p.name.into(),
                            label: p.label.into(),
                            value: fx.sample_param(p.name, 0.0).unwrap_or(p.default),
                            min: p.min,
                            max: p.max,
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

/// Project one animatable property's keyframes for the UI: clip-relative
/// engine ticks become ABSOLUTE sequence ticks (start + offset), easing is
/// flattened to the Slint encoding, and `split` maps the value into the
/// `(value-x, value-y)` pair (scalars leave y at 0). Empty ⇔ constant.
fn keyframes_to_slint<T: Lerp>(
    param: &Param<T>,
    clip_start: i64,
    split: impl Fn(&T) -> (f32, f32),
) -> ModelRc<ParamKeyframe> {
    let rows: Vec<ParamKeyframe> = param
        .keyframes()
        .iter()
        .map(|kf: &Keyframe<T>| {
            let (value_x, value_y) = split(&kf.value);
            let (easing, [bez_x1, bez_y1, bez_x2, bez_y2]) = easing_to_ui(kf.easing);
            ParamKeyframe {
                tick: clamp_i32(clip_start + kf.tick),
                value_x,
                value_y,
                easing,
                bez_x1,
                bez_y1,
                bez_x2,
                bez_y2,
            }
        })
        .collect();
    model(rows)
}

/// The clip's speed as a display/scale float (1.0 for degenerate rationals,
/// which the model rejects anyway).
fn speed_factor(speed: EngineRational) -> f32 {
    if speed.num <= 0 || speed.den <= 0 {
        return 1.0;
    }
    speed.num as f32 / speed.den as f32
}

/// Retime badge for the timeline card: `2x` / `0.5x` (trailing zeros
/// trimmed), with ` R` appended when reversed — a reversed 1× clip shows
/// just `R`. A speed ramp (M2 curve) shows its *effective* average rate with
/// a `~` prefix (`~1.4x`) so it reads as varying, not constant. Empty ⇔
/// forward 1× with no ramp (no badge).
fn speed_label(clip: &EngineClip) -> String {
    if !clip.is_retimed() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    // A ramp's effective rate is the base speed times the curve's average;
    // the `~` marks that the instantaneous rate varies across the clip.
    let ramped = clip.has_speed_curve();
    let factor = speed_factor(clip.speed) * if ramped { clip.speed_curve_average() as f32 } else { 1.0 };
    if ramped || (factor - 1.0).abs() > f32::EPSILON {
        let mut s = format!("{factor:.2}");
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        parts.push(format!("{}{s}x", if ramped { "~" } else { "" }));
    }
    if clip.reversed {
        parts.push("R".into());
    }
    parts.join(" ")
}

/// `time` as seconds, exact rational division in floating point.
fn time_to_seconds(time: EngineTime) -> f64 {
    if time.rate.num <= 0 || time.rate.den <= 0 {
        return 0.0;
    }
    time.value as f64 * f64::from(time.rate.den) / f64::from(time.rate.num)
}

/// Clip badge: CapCut-style `3.4s` under a minute, `M:SS` (or `H:MM:SS`)
/// from there up.
fn clip_duration_label(duration: EngineTime) -> String {
    let secs = time_to_seconds(duration).max(0.0);
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let whole = secs.round() as i64;
        let (h, m, s) = (whole / 3600, (whole / 60) % 60, whole % 60);
        if h > 0 {
            format!("{h}:{m:02}:{s:02}")
        } else {
            format!("{m}:{s:02}")
        }
    }
}

/// Trim headroom for generated clips, which have no source bounds. Big enough
/// to never clamp, small enough that `clip end + room` can't overflow `i32`.
const UNBOUNDED_ROOM: i32 = i32::MAX / 4;

/// How far (sequence ticks) each clip edge can extend before running out of
/// source media: `(head, tail)`. Head room is the media before the in-point,
/// tail room the media after the out-point, both projected to the sequence
/// rate *conservatively* (see [`room_to_sequence_ticks`]) so the trim ghost
/// never offers an extension `Project::trim_clip` would reject.
fn trim_rooms(project: &EngineProject, clip: &EngineClip) -> (i32, i32) {
    let tl_rate = project.timeline().frame_rate;
    match &clip.content {
        ClipSource::Media { media, source } => {
            let Some(media) = project.media(*media) else {
                return (0, 0);
            };
            // Stills extend freely: the one frame repeats, and the pool
            // duration is a default placement length, not material bounds
            // (the engine relaxes `trim_clip` the same way).
            if media.is_image {
                return (UNBOUNDED_ROOM, UNBOUNDED_ROOM);
            }
            let head_media = source.start.value;
            let tail_media = media.duration.value - source.end_tick();
            (
                room_to_sequence_ticks(head_media, media.frame_rate, tl_rate),
                room_to_sequence_ticks(tail_media, media.frame_rate, tl_rate),
            )
        }
        ClipSource::Generated(_) => (UNBOUNDED_ROOM, UNBOUNDED_ROOM),
    }
}

/// Largest number of sequence ticks an edge may extend such that the engine's
/// media-rate resample of that delta stays within `room_media` ticks.
///
/// `Project::trim_clip` re-derives the source delta by resampling the
/// timeline delta (round-to-nearest), so a naive media→sequence conversion
/// can overshoot by a tick and get the commit rejected. Convert, then verify
/// by round-tripping and step down until it fits; when the rates differ, keep
/// one extra media tick in reserve for the duration-resample's own rounding.
fn room_to_sequence_ticks(
    room_media: i64,
    media_rate: EngineRational,
    tl_rate: EngineRational,
) -> i32 {
    let mut room = room_media.max(0);
    if !rate_eq(media_rate, tl_rate) {
        room = (room - 1).max(0);
    }
    let mut ticks = resample(EngineTime::new(room, media_rate), tl_rate).value;
    while ticks > 0 && resample(EngineTime::new(ticks, tl_rate), media_rate).value > room {
        ticks -= 1;
    }
    clamp_i32(ticks)
}

/// `(lane label, text-generator content)` for a clip.
fn clip_labels(project: &EngineProject, clip: &EngineClip) -> (String, String) {
    match &clip.content {
        ClipSource::Media { media, .. } => {
            let name = project
                .media(*media)
                .map(media_name)
                .unwrap_or_else(|| format!("Clip {}", clip.id.raw()));
            (name, String::new())
        }
        ClipSource::Generated(generator) => match generator {
            Generator::Text { content, .. } => ("Text".to_owned(), content.clone()),
            Generator::SolidColor { .. } => ("Solid".to_owned(), String::new()),
            Generator::Shape { .. } => ("Shape".to_owned(), String::new()),
            Generator::Sticker => ("Sticker".to_owned(), String::new()),
            Generator::Effect => ("Effect".to_owned(), String::new()),
            Generator::Filter => ("Filter".to_owned(), String::new()),
            Generator::Adjustment => ("Adjustment".to_owned(), String::new()),
        },
    }
}

fn clip_shape_size(clip: &EngineClip) -> (f32, f32) {
    match &clip.content {
        ClipSource::Generated(Generator::Shape { width, height, .. }) => (*width, *height),
        _ => (0.0, 0.0),
    }
}

/// `(generator-kind tag, fill color)` for the timeline card. The tag selects
/// the card's preview rendering (see `panels/timeline/clip.slint`); the color
/// is the solid/shape fill (transparent for everything else).
fn clip_generator_visual(clip: &EngineClip) -> (&'static str, Color) {
    let transparent = Color::from_argb_u8(0, 0, 0, 0);
    match &clip.content {
        ClipSource::Generated(Generator::Text { .. }) => ("text", transparent),
        ClipSource::Generated(Generator::SolidColor { rgba }) => ("solid", rgba_color(*rgba)),
        ClipSource::Generated(Generator::Shape { shape, rgba, .. }) => {
            let tag = match shape {
                cutlass_models::Shape::Rectangle => "rect",
                cutlass_models::Shape::Ellipse => "ellipse",
            };
            (tag, rgba_color(*rgba))
        }
        _ => ("", transparent),
    }
}

fn rgba_color(rgba: [u8; 4]) -> Color {
    Color::from_argb_u8(rgba[3], rgba[0], rgba[1], rgba[2])
}

/// Project a clip's text styling into the Slint `TextStyle`. Non-text clips
/// (and text clips written before styling existed) get the engine default
/// look, so the inspector always has a coherent style to edit.
fn clip_text_style(clip: &EngineClip) -> TextClipStyle {
    let style = match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style.clone(),
        _ => EngineTextStyle::default(),
    };
    text_style_to_ui(&style)
}

/// Convert an engine `TextStyle` to the Slint struct. Effect opacities are
/// pulled out of their rgba alpha into a dedicated 0..=1 control, and the
/// swatch colors are made opaque so the picker preview reads cleanly.
fn text_style_to_ui(style: &EngineTextStyle) -> TextClipStyle {
    let opaque = |rgba: [u8; 4]| Color::from_rgb_u8(rgba[0], rgba[1], rgba[2]);
    let alpha01 = |rgba: [u8; 4]| rgba[3] as f32 / 255.0;
    let stroke = style.stroke.unwrap_or_default();
    let background = style.background.unwrap_or_default();
    let shadow = style.shadow.unwrap_or_default();
    TextClipStyle {
        font: style.font.clone().into(),
        size: style.size,
        bold: style.bold,
        italic: style.italic,
        underline: style.underline,
        case: text_case_to_int(style.case),
        fill: Color::from_argb_u8(style.fill[3], style.fill[0], style.fill[1], style.fill[2]),
        letter_spacing: style.letter_spacing,
        line_spacing: style.line_spacing,
        align_h: align_h_to_int(style.align_h),
        align_v: align_v_to_int(style.align_v),
        wrap: style.wrap,
        stroke_enabled: style.stroke.is_some(),
        stroke_color: opaque(stroke.rgba),
        stroke_width: stroke.width,
        background_enabled: style.background.is_some(),
        background_color: opaque(background.rgba),
        background_opacity: alpha01(background.rgba),
        background_radius: background.radius,
        shadow_enabled: style.shadow.is_some(),
        shadow_color: opaque(shadow.rgba),
        shadow_opacity: alpha01(shadow.rgba),
        shadow_blur: shadow.blur,
        shadow_distance: shadow.distance,
    }
}

fn text_case_to_int(case: TextCase) -> i32 {
    match case {
        TextCase::Normal => 0,
        TextCase::Upper => 1,
        TextCase::Lower => 2,
        TextCase::Title => 3,
    }
}

fn align_h_to_int(align: TextAlignH) -> i32 {
    match align {
        TextAlignH::Left => 0,
        TextAlignH::Center => 1,
        TextAlignH::Right => 2,
    }
}

fn align_v_to_int(align: TextAlignV) -> i32 {
    match align {
        TextAlignV::Top => 0,
        TextAlignV::Middle => 1,
        TextAlignV::Bottom => 2,
    }
}

/// The engine's composite canvas size, as Slint lengths. Delegating to
/// `composite_canvas_size` (rather than mirroring it, as this module used
/// to) keeps preview hit-test geometry pixel-identical to the composited
/// frame by construction — including the M1 aspect presets. It also fixes a
/// drift the old mirror had: it let still images vote on the canvas size,
/// the engine never did.
fn canvas_size(project: &EngineProject) -> (f32, f32) {
    let (w, h) = cutlass_engine::composite_canvas_size(project);
    (w as f32, h as f32)
}

/// `CanvasAspect` as the preset index the canvas dialog's ratio list uses.
fn aspect_to_index(aspect: cutlass_models::CanvasAspect) -> i32 {
    cutlass_models::CanvasAspect::ALL
        .iter()
        .position(|a| *a == aspect)
        .map_or(0, |i| i as i32)
}

/// Lane kinds the UI surfaces today. Effect / filter lanes are still phantom
/// until their engines land (v1 roadmap M0 "hide phantom kinds", M5): the
/// model keeps them — they round-trip through save/load untouched and
/// composite nothing — but the projection skips them so users never see lanes
/// that do nothing. Adjustment lanes became real in M4 and are now shown.
fn kind_visible(kind: EngineKind) -> bool {
    !matches!(kind, EngineKind::Effect | EngineKind::Filter)
}

fn track_kind(kind: EngineKind) -> TrackKind {
    match kind {
        EngineKind::Video => TrackKind::Video,
        EngineKind::Audio => TrackKind::Audio,
        EngineKind::Text => TrackKind::Text,
        EngineKind::Sticker => TrackKind::Sticker,
        EngineKind::Effect => TrackKind::Effect,
        EngineKind::Filter => TrackKind::Filter,
        EngineKind::Adjustment => TrackKind::Adjustment,
    }
}

/// One color per lane kind (the engine has no per-track color). Matches the
/// palette the UI previously hardcoded in `editor-store.slint`.
fn kind_color(kind: EngineKind) -> Color {
    let (r, g, b) = match kind {
        EngineKind::Video => (0x4A, 0x6F, 0xA5),
        EngineKind::Audio => (0xC9, 0x98, 0x46),
        EngineKind::Text => (0x5E, 0x8B, 0x7E),
        EngineKind::Sticker => (0xBF, 0x6F, 0x4A),
        EngineKind::Effect => (0x7B, 0x68, 0xA6),
        EngineKind::Filter => (0x4A, 0x8C, 0x8C),
        EngineKind::Adjustment => (0x6C, 0x5B, 0x7B),
    };
    Color::from_rgb_u8(r, g, b)
}

fn marker_to_slint(marker: &EngineMarker) -> TimelineMarker {
    let [r, g, b, a] = marker.color.rgba();
    TimelineMarker {
        id: marker.id.raw().to_string().into(),
        tick: clamp_i32(marker.tick.value),
        name: marker.name.clone().into(),
        color: Color::from_argb_u8(a, r, g, b),
    }
}

fn rational(rate: cutlass_models::Rational) -> Rational {
    Rational {
        num: rate.num,
        den: rate.den,
    }
}

fn rational_time(time: EngineTime) -> RationalTime {
    RationalTime {
        // Slint's time model is `i32`; clamp the engine's `i64` ticks. Realistic
        // projects stay well inside `i32` (≈24 days at 1000 fps).
        value: clamp_i32(time.value),
        rate: rational(time.rate),
    }
}

fn time_range(range: EngineRange) -> TimeRange {
    TimeRange {
        start: rational_time(range.start),
        duration: rational_time(range.duration),
    }
}

/// The single choke point projecting engine `i64` ticks into Slint's `i32`
/// time model (keyframes roadmap Phase 4 — tick audit). Every tick that
/// crosses the boundary (`rational_time`, markers, keyframe + speed-ramp
/// rows) routes through here so an out-of-range value **saturates** instead
/// of wrapping — a clip parked past the bound clamps to the edge of the
/// addressable timeline rather than teleporting to a negative tick.
///
/// ## Timeline-length bound
///
/// `i32::MAX` ticks is the hard ceiling. In wall-clock time that is
/// `i32::MAX / fps` seconds: ≈ 20.7 hours at 30 fps, ≈ 8.3 hours at 72 fps,
/// ≈ 24.8 days at 1000 fps. Real projects stay orders of magnitude inside
/// it; the clamp only exists so a pathological/corrupt tick can never alias
/// to a bogus on-screen position. Promoting the Slint model to `i64` is the
/// long-term fix (tracked in `timeline-roadmap.md`).
fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn model<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::from(Rc::new(VecModel::from(items)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(value: i64, num: i32, den: i32) -> EngineTime {
        EngineTime::new(value, EngineRational { num, den })
    }

    #[test]
    fn duration_label_uses_seconds_under_a_minute() {
        assert_eq!(clip_duration_label(t(90, 30, 1)), "3.0s");
        assert_eq!(clip_duration_label(t(101, 30, 1)), "3.4s");
        assert_eq!(clip_duration_label(t(0, 30, 1)), "0.0s");
    }

    #[test]
    fn duration_label_switches_to_timecode_at_a_minute() {
        assert_eq!(clip_duration_label(t(1800, 30, 1)), "1:00");
        // 1h 0m 23s at 30fps.
        assert_eq!(clip_duration_label(t(30 * 3623, 30, 1)), "1:00:23");
    }

    #[test]
    fn duration_label_handles_ntsc_rates() {
        // Exactly 60 logical frames at 29.97: just under 60.06s.
        assert_eq!(clip_duration_label(t(1800, 30000, 1001)), "1:00");
    }

    #[test]
    fn time_to_seconds_is_rate_exact() {
        assert_eq!(time_to_seconds(t(48, 24, 1)), 2.0);
        assert_eq!(time_to_seconds(t(500, 1000, 1)), 0.5);
        assert_eq!(time_to_seconds(t(1, 0, 1)), 0.0, "degenerate rate is safe");
    }

    #[test]
    fn speed_label_formats_retimes() {
        use cutlass_models::{Clip as MClip, MediaId, TimeRange};
        let mut clip = MClip::from_media(
            MediaId::from_raw(1),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        );
        assert_eq!(speed_label(&clip), "", "1× forward has no badge");

        clip.speed = EngineRational::new(2, 1);
        assert_eq!(speed_label(&clip), "2x");
        clip.speed = EngineRational::new(1, 2);
        assert_eq!(speed_label(&clip), "0.5x");
        clip.speed = EngineRational::new(3, 4);
        assert_eq!(speed_label(&clip), "0.75x");

        clip.reversed = true;
        assert_eq!(speed_label(&clip), "0.75x R");
        clip.speed = EngineRational::new(1, 1);
        assert_eq!(speed_label(&clip), "R");
    }

    #[test]
    fn phantom_lanes_are_not_projected() {
        use slint::Model;

        let mut project = EngineProject::new("test", EngineRational::FPS_24);
        project.add_track(cutlass_models::TrackKind::Video, "V1");
        project.add_track(cutlass_models::TrackKind::Effect, "FX1");
        project.add_track(cutlass_models::TrackKind::Filter, "F1");
        project.add_track(cutlass_models::TrackKind::Adjustment, "ADJ1");
        project.add_track(cutlass_models::TrackKind::Sticker, "ST1");

        let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
        let tracks = &projected.sequence.tracks;
        // Top-first: sticker, then the now-real adjustment lane (M4), then
        // video; the effect / filter lanes stay model-only (M0 "hide phantom
        // kinds", their engines land in M5).
        assert_eq!(tracks.row_count(), 3);
        assert_eq!(tracks.row_data(0).unwrap().kind, TrackKind::Sticker);
        assert_eq!(tracks.row_data(1).unwrap().kind, TrackKind::Adjustment);
        assert_eq!(tracks.row_data(2).unwrap().kind, TrackKind::Video);
    }

    #[test]
    fn media_pool_flags_missing_entries() {
        use cutlass_models::MediaSource;
        use slint::Model;

        let mut project = EngineProject::new("test", EngineRational::FPS_24);
        let here = project.add_media(MediaSource::new("/tmp/a.mp4", 1920, 1080, EngineRational::FPS_24, 48, true));
        let gone = project.add_media(MediaSource::new("/tmp/b.mp4", 1920, 1080, EngineRational::FPS_24, 48, true));

        let missing: HashSet<u64> = [gone.raw()].into();
        let projected = project_to_slint(&project, &HashMap::new(), &missing);
        let media = &projected.media;
        assert_eq!(media.row_count(), 2);
        // The pool is sorted by raw id, so rows follow insertion here.
        let first = media.row_data(0).unwrap();
        let second = media.row_data(1).unwrap();
        assert_eq!(first.id.as_str(), here.raw().to_string());
        assert!(!first.is_missing);
        assert!(second.is_missing);
        assert_eq!(second.path.as_str(), "/tmp/b.mp4", "dialog shows where the file used to be");
    }

    #[test]
    fn keyframes_publish_absolute_ticks_and_easing() {
        use cutlass_models::{Easing, Keyframe, Param};
        use slint::Model;

        let constant: Param<f32> = Param::Constant(1.0);
        assert_eq!(keyframes_to_slint(&constant, 100, |v| (*v, 0.0)).row_count(), 0);

        let param = Param::Keyframed {
            keyframes: vec![
                Keyframe { tick: 0, value: 0.5f32, easing: Easing::EaseOut },
                Keyframe {
                    tick: 24,
                    value: 1.0,
                    easing: Easing::Bezier { points: [0.42, 0.0, 0.58, 1.0] },
                },
            ],
        };
        let rows = keyframes_to_slint(&param, 100, |v| (*v, 0.0));
        assert_eq!(rows.row_count(), 2);
        let first = rows.row_data(0).unwrap();
        assert_eq!((first.tick, first.value_x, first.easing), (100, 0.5, 2));
        let second = rows.row_data(1).unwrap();
        assert_eq!((second.tick, second.easing), (124, 4));
        assert_eq!(
            [second.bez_x1, second.bez_y1, second.bez_x2, second.bez_y2],
            [0.42, 0.0, 0.58, 1.0]
        );
    }

    // --- Phase 4 tick audit: i64 → i32 projection saturates, never wraps. ---

    #[test]
    fn clamp_i32_saturates_at_the_bounds() {
        assert_eq!(clamp_i32(0), 0);
        assert_eq!(clamp_i32(1_000), 1_000);
        // Above/below i32 range pin to the edge instead of wrapping (a naive
        // `as i32` would alias these to small / negative ticks).
        assert_eq!(clamp_i32(i64::from(i32::MAX) + 1), i32::MAX);
        assert_eq!(clamp_i32(i64::MAX), i32::MAX);
        assert_eq!(clamp_i32(i64::from(i32::MIN) - 1), i32::MIN);
        assert_eq!(clamp_i32(i64::MIN), i32::MIN);
    }

    #[test]
    fn rational_time_saturates_huge_ticks() {
        // A tick parked past the i32 ceiling clamps to the edge of the
        // addressable timeline rather than teleporting to a negative frame.
        let huge = rational_time(t(i64::from(i32::MAX) + 5_000, 30, 1));
        assert_eq!(huge.value, i32::MAX);
        assert_eq!((huge.rate.num, huge.rate.den), (30, 1));
        // In-range ticks pass through untouched.
        assert_eq!(rational_time(t(123, 30, 1)).value, 123);
    }

    #[test]
    fn speed_label_marks_ramps_with_a_tilde() {
        use cutlass_models::{Clip as MClip, MediaId, TimeRange, speed_preset};
        let mut clip = MClip::from_media(
            MediaId::from_raw(1),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        );
        clip.speed_curve = speed_preset("montage").unwrap();
        let label = speed_label(&clip);
        assert!(label.starts_with('~'), "ramp badge is tilde-prefixed: {label}");
        assert!(label.ends_with('x'), "ramp badge reports an effective rate: {label}");
    }

    #[test]
    fn speed_curve_projects_dense_samples_and_handles() {
        use cutlass_models::{MediaId, MediaSource, TimeRange, speed_preset};
        use slint::Model;

        let mut project = EngineProject::new("test", EngineRational::FPS_24);
        let media = project.add_media(MediaSource::new(
            "/tmp/a.mp4",
            1920,
            1080,
            EngineRational::FPS_24,
            480,
            true,
        ));
        let _ = media;
        let mut clip = cutlass_models::Clip::from_media(
            MediaId::from_raw(media.raw()),
            TimeRange::at_rate(0, 240, EngineRational::FPS_24),
            TimeRange::at_rate(0, 240, EngineRational::FPS_24),
        );
        // Flat clip: no ramp data projected.
        let flat = clip_to_slint(&project, &clip, &HashMap::new());
        assert!(!flat.has_speed_curve);
        assert_eq!(flat.kf_speed_curve.row_count(), 0);
        assert_eq!(flat.speed_curve_samples.row_count(), 0);

        // Montage ramp: handles mirror the curve's control points (normalized
        // ticks, no clip-start offset), and the dense sample strip fills in.
        clip.speed_curve = speed_preset("montage").unwrap();
        let ramped = clip_to_slint(&project, &clip, &HashMap::new());
        assert!(ramped.has_speed_curve);
        assert_eq!(ramped.kf_speed_curve.row_count(), 3);
        assert_eq!(ramped.kf_speed_curve.row_data(0).unwrap().tick, 0);
        assert_eq!(
            ramped.kf_speed_curve.row_data(2).unwrap().tick,
            cutlass_models::SPEED_CURVE_SCALE as i32
        );
        assert_eq!(ramped.speed_curve_samples.row_count(), SPEED_GRAPH_SAMPLES);
        assert!(ramped.speed_curve_avg > 0.0);
    }
}
