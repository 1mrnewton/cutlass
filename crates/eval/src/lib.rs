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

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, TimeRange};

    use super::freeze_timestamp;

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
}
