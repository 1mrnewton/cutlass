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

use std::collections::HashMap;
use std::rc::Rc;

use cutlass_models::{
    Clip as EngineClip, ClipSource, Generator, Keyframe, Lerp, MediaSource, Param,
    Project as EngineProject, Rational as EngineRational, RationalTime as EngineTime, TextAlignH,
    TextAlignV, TextCase, TextStyle as EngineTextStyle, TimeRange as EngineRange,
    Track as EngineTrack, TrackKind as EngineKind, rate_eq, resample,
};
use slint::{Color, ModelRc, VecModel};

use crate::params::easing_to_ui;
use crate::{
    Clip, Media, ParamKeyframe, Project, Rational, RationalTime, Sequence, TextClipStyle,
    TimeRange, Track, TrackKind,
};

/// Fallback canvas size when no video media has been imported yet. Mirrors the
/// engine's `composite_canvas_size` default so preview aspect ratio is stable.
const DEFAULT_CANVAS_W: f32 = 1920.0;
const DEFAULT_CANVAS_H: f32 = 1080.0;

/// Project the engine's project state into the Slint view model.
///
/// `generator_sizes` maps raw clip ids of generated clips to their
/// drawn-content size in canvas px (computed on the engine thread, where the
/// raster cache lives) — the preview's selection geometry needs it because
/// generators raster at full canvas size.
pub fn project_to_slint(
    project: &EngineProject,
    generator_sizes: &HashMap<u64, (i32, i32)>,
) -> Project {
    let timeline = project.timeline();
    let (width, height) = canvas_size(project);

    // The engine stacks bottom→top (last track composites in front); the lane
    // list shows the stack top-first so the top lane is the front layer, like
    // CapCut/Premiere. UI row r ↔ engine order index (track_count - 1 - r).
    let mut tracks: Vec<Track> = timeline
        .tracks_ordered()
        .map(|track| track_to_slint(project, track, generator_sizes))
        .collect();
    tracks.reverse();

    let id = project.id.raw().to_string();

    let pool = media_pool(project);
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
        },
        media: model(pool),
        media_audio: model(audio_pool),
    }
}

/// The media pool as Library bin entries, ordered by id (the engine's pool is a
/// hash map, so a stable sort keeps tile order from jumping between imports).
fn media_pool(project: &EngineProject) -> Vec<Media> {
    let tl_rate = project.timeline().frame_rate;
    let mut sources: Vec<&MediaSource> = project.media_iter().collect();
    sources.sort_by_key(|media| media.id.raw());
    sources
        .into_iter()
        .map(|media| media_to_slint(media, tl_rate))
        .collect()
}

fn media_to_slint(media: &MediaSource, tl_rate: cutlass_models::Rational) -> Media {
    Media {
        id: media.id.raw().to_string().into(),
        name: media_name(media).into(),
        width: media.width as i32,
        height: media.height as i32,
        has_audio: media.has_audio,
        duration_ticks: clamp_i32(resample(media.duration, tl_rate).value),
        is_audio: media.is_audio_only(),
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

    Clip {
        id: clip.id.raw().to_string().into(),
        name: name.into(),
        timeline_start: rational_time(clip.timeline.start),
        source_range: time_range(clip.timeline),
        media_id: media_id.into(),
        source_in_s,
        duration_label: clip_duration_label(clip.timeline.duration).into(),
        text_content: text_content.into(),
        text_style: clip_text_style(clip),
        generator_kind: generator_kind.into(),
        fill_color,
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
        kf_position: keyframes_to_slint(&clip.transform.position, clip_start, |v| (v[0], v[1])),
        kf_scale: keyframes_to_slint(&clip.transform.scale, clip_start, |v| (*v, 0.0)),
        kf_rotation: keyframes_to_slint(&clip.transform.rotation, clip_start, |v| (*v, 0.0)),
        kf_opacity: keyframes_to_slint(&clip.transform.opacity, clip_start, |v| (*v, 0.0)),
    }
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

/// `(generator-kind tag, fill color)` for the timeline card. The tag selects
/// the card's preview rendering (see `panels/timeline/clip.slint`); the color
/// is the solid/shape fill (transparent for everything else).
fn clip_generator_visual(clip: &EngineClip) -> (&'static str, Color) {
    let transparent = Color::from_argb_u8(0, 0, 0, 0);
    match &clip.content {
        ClipSource::Generated(Generator::Text { .. }) => ("text", transparent),
        ClipSource::Generated(Generator::SolidColor { rgba }) => ("solid", rgba_color(*rgba)),
        ClipSource::Generated(Generator::Shape { shape, rgba }) => {
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

/// Largest video-media resolution in the project, or the default canvas.
/// Mirrors `cutlass_engine`'s `composite_canvas_size`.
fn canvas_size(project: &EngineProject) -> (f32, f32) {
    let mut max_w = 0u32;
    let mut max_h = 0u32;

    for track in project.timeline().tracks_ordered() {
        if track.kind != EngineKind::Video {
            continue;
        }
        for clip in track.clips() {
            if let Some(media_id) = clip.media()
                && let Some(media) = project.media(media_id)
            {
                max_w = max_w.max(media.width);
                max_h = max_h.max(media.height);
            }
        }
    }

    if max_w == 0 || max_h == 0 {
        (DEFAULT_CANVAS_W, DEFAULT_CANVAS_H)
    } else {
        // Mirror the engine's even-rounding (H.264 requirement) so preview
        // hit-test geometry matches the composited frame exactly.
        (to_even(max_w) as f32, to_even(max_h) as f32)
    }
}

fn to_even(v: u32) -> u32 {
    if v.is_multiple_of(2) { v } else { v + 1 }
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
}
