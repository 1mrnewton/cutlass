//! One immutable snapshot at one time.
//!
//! `frame` returns the composited picture. It does not write the draft.
//! A video segment opens through the caller’s reader factory. [`OpenClips`]
//! keeps that file for one draft revision and seeks it at a later time.
//! A compound segment is the inlined child draft's picture, placed with the
//! compound clip. `audio_slice` is the engine mix at that time, and a
//! compound segment contributes the child's mix. A text segment is drawn
//! when [`Fonts`] supplies the face named by `font_path`. Word columns on
//! that segment are the caption. An effect package is drawn when the
//! snapshot supplies the package store. A missing package stays off the frame.
//! The project canvas color is drawn under every layer. An absent or
//! unreadable color leaves the clear frame. A segment canvas is drawn
//! behind that segment's picture. A transition package fades the two
//! pictures that meet at its cut. A missing package stays a hard cut.

#![forbid(unsafe_code)]

mod audio;
mod canvas;
mod effect;
mod frame;
mod freeze;
mod gpu;
mod heard;
mod lottie;
mod media;
mod nest;
mod sample;
mod scenes;
mod sources;
mod text;
mod track;
mod transition;

pub use audio::{AudioError, MixWindow, audio_slice};
pub use cutlass_compositor::{Device, Frame};
pub use cutlass_engine::{
    ClipPcm, ClockKind, MixResult, ReaderFactory, SceneError, scene_cuts, timeline_for_source,
};
pub use cutlass_gpu::{ToneCurve, YuvTone};
pub use cutlass_model::Micros;
pub use frame::{
    EvalError, Fonts, NoFonts, Snapshot, frame, frame_in, layers_in, wait_export_frame,
};
pub use freeze::{FreezeMiss, freeze_toggle};
pub use gpu::{Composed, FrameLayers, GpuBackdrop, GpuLayer, GpuSource, YuvSamples};
pub use heard::{ClipAsk, Heard};
pub use media::OpenClips;
pub use scenes::{blend_blocks, kept_cuts, scene_split_times};
pub use text::text_size;
pub use track::{TrackAsk, TrackError, TrackSpan, track_clip};

/// The source timestamp Freeze stores so the clip keeps showing `time`.
///
/// The stored value is the position before reverse. Playback flips it when
/// the clip is reversed, and a stored freeze replaces this map. `None` when
/// `time` is outside the clip or the clip has no source range.
#[must_use]
pub fn freeze_timestamp(
    draft: &cutlass_model::Draft,
    segment: &cutlass_model::Segment,
    time: Micros,
) -> Option<Micros> {
    let mut map = cutlass_engine::placement_of(draft, segment).ok()?;
    map.freeze = None;
    map.reverse = false;
    cutlass_engine::map_timeline(&map, time).ok()
}

/// Timeline time that shows `source` on `segment`.
///
/// Speed, a curve, and reverse follow the playback map. `None` when the
/// segment has no source range, or no frame in the clip is that source time.
#[must_use]
pub fn source_on_timeline(
    draft: &cutlass_model::Draft,
    segment: &cutlass_model::Segment,
    source: Micros,
) -> Option<Micros> {
    let map = cutlass_engine::placement_of(draft, segment).ok()?;
    timeline_for_source(&map, source)
}

/// The point of `segment`'s own picture under canvas pixel `x`, `y` at
/// `time`, on a `canvas` sized canvas: -1 to 1 across the picture, y up, as
/// masks and selections read it. `None` when the clip is not drawn at
/// `time`. A point off the picture lies outside -1 to 1.
#[must_use]
pub fn clip_point(
    draft: &cutlass_model::Draft,
    segment: &cutlass_model::Segment,
    time: Micros,
    canvas: (u32, u32),
    x: f64,
    y: f64,
) -> Option<(f64, f64)> {
    Some(media::clip_frame(draft, segment, time, canvas)?.apply(x, y))
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, TimeRange};

    use super::{clip_point, freeze_timestamp};

    #[test]
    fn a_freeze_timestamp_follows_speed_and_stays_before_reverse() {
        let draft = Draft::empty("D");
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        assert_eq!(
            freeze_timestamp(&draft, &segment, Micros(500_000)),
            Some(Micros(500_000))
        );
        assert_eq!(freeze_timestamp(&draft, &segment, Micros(2_000_000)), None);
        segment.speed = 2.0;
        segment.target.duration = Micros(1_000_000);
        assert_eq!(
            freeze_timestamp(&draft, &segment, Micros(400_000)),
            Some(Micros(800_000))
        );
        segment.reverse = true;
        segment.speed = 1.0;
        segment.target.duration = Micros(1_000);
        segment.source_range = Some(TimeRange::new(0, 1_000));
        assert_eq!(
            freeze_timestamp(&draft, &segment, Micros(250)),
            Some(Micros(250))
        );
    }

    #[test]
    fn a_canvas_point_maps_into_the_clip_and_follows_its_move() {
        let mut draft = Draft::empty("D");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        let mut video = cutlass_model::MaterialVideo::new("v");
        video.width = 1920;
        video.height = 1080;
        draft.materials.videos.push(video);
        let mut segment = Segment::new("a", "v", TimeRange::new(0, 1_000_000));
        let at = |segment: &Segment, x: f64, y: f64| {
            let (u, v) = clip_point(&draft, segment, Micros(0), (1920, 1080), x, y).unwrap();
            ((u * 1000.0).round() / 1000.0, (v * 1000.0).round() / 1000.0)
        };
        assert_eq!(at(&segment, 960.0, 540.0), (0.0, 0.0));
        assert_eq!(at(&segment, 0.0, 0.0), (-1.0, 1.0));
        assert_eq!(at(&segment, 1920.0, 1080.0), (1.0, -1.0));
        // Moved right by half the canvas, the clip's centre is under 1440.
        segment.clip.transform.x = 0.5;
        assert_eq!(at(&segment, 1440.0, 540.0), (0.0, 0.0));
        assert!(at(&segment, 100.0, 540.0).0 < -1.0);
    }
}
