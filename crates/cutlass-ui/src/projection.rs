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

use std::rc::Rc;

use cutlass_models::{
    Clip as EngineClip, ClipSource, Generator, MediaSource, Project as EngineProject,
    Rational as EngineRational, RationalTime as EngineTime, TimeRange as EngineRange,
    Track as EngineTrack, TrackKind as EngineKind, rate_eq, resample,
};
use slint::{Color, ModelRc, VecModel};

use crate::{Clip, Media, Project, Rational, RationalTime, Sequence, TimeRange, Track, TrackKind};

/// Fallback canvas size when no video media has been imported yet. Mirrors the
/// engine's `composite_canvas_size` default so preview aspect ratio is stable.
const DEFAULT_CANVAS_W: f32 = 1920.0;
const DEFAULT_CANVAS_H: f32 = 1080.0;

/// Project the engine's project state into the Slint view model.
pub fn project_to_slint(project: &EngineProject) -> Project {
    let timeline = project.timeline();
    let (width, height) = canvas_size(project);

    // The engine stacks bottom→top (last track composites in front); the lane
    // list shows the stack top-first so the top lane is the front layer, like
    // CapCut/Premiere. UI row r ↔ engine order index (track_count - 1 - r).
    let mut tracks: Vec<Track> = timeline
        .tracks_ordered()
        .map(|track| track_to_slint(project, track))
        .collect();
    tracks.reverse();

    let id = project.id.raw().to_string();

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
        media: model(media_pool(project)),
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

fn track_to_slint(project: &EngineProject, track: &EngineTrack) -> Track {
    let clips: Vec<Clip> = track
        .clips_ordered()
        .into_iter()
        .map(|clip| clip_to_slint(project, clip))
        .collect();

    Track {
        id: track.id.raw().to_string().into(),
        name: track.name.clone().into(),
        kind: track_kind(track.kind),
        color: kind_color(track.kind),
        clips: model(clips),
    }
}

fn clip_to_slint(project: &EngineProject, clip: &EngineClip) -> Clip {
    // The timeline UI positions a clip at `timeline-start` and derives its width
    // from `source-range.duration`, both in sequence ticks. The engine's
    // authoritative on-sequence placement is `clip.timeline`, so mirror it here
    // (1:1 playback; the true media in/out isn't needed until time-remap or a
    // live inspector requires it).
    let (name, text_content) = clip_labels(project, clip);
    let (head_room, tail_room) = trim_rooms(project, clip);
    let (media_id, source_in_s) = match &clip.content {
        ClipSource::Media { media, source } => {
            (media.raw().to_string(), time_to_seconds(source.start) as f32)
        }
        ClipSource::Generated(_) => (String::new(), 0.0),
    };

    Clip {
        id: clip.id.raw().to_string().into(),
        name: name.into(),
        timeline_start: rational_time(clip.timeline.start),
        source_range: time_range(clip.timeline),
        media_id: media_id.into(),
        source_in_s,
        duration_label: clip_duration_label(clip.timeline.duration).into(),
        text_content: text_content.into(),
        head_room_ticks: head_room,
        tail_room_ticks: tail_room,
    }
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
            Generator::Text { content } => ("Text".to_owned(), content.clone()),
            Generator::SolidColor { .. } => ("Solid".to_owned(), String::new()),
            Generator::Shape { .. } => ("Shape".to_owned(), String::new()),
            Generator::Sticker => ("Sticker".to_owned(), String::new()),
            Generator::Effect => ("Effect".to_owned(), String::new()),
            Generator::Filter => ("Filter".to_owned(), String::new()),
            Generator::Adjustment => ("Adjustment".to_owned(), String::new()),
        },
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
        (max_w as f32, max_h as f32)
    }
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
}
