//! Renders: a stretch of the timeline drawn ahead into a movie, so the
//! preview plays one file there instead of drawing every layer each frame.
//!
//! A render is named by the project as it was drawn: its key is a hash of
//! the whole draft, so any edit leaves it behind and an undo back to the
//! same draft finds it again. The movie has pictures only; the preview's
//! sound still comes from the timeline. A render is never used for export.

use std::path::Path;

use cutlass_codec::{CodecId, ContainerId};
use cutlass_model::{Draft, FrameRate, MaterialVideo, Micros, Segment, TimeRange, Track};

use crate::{ExportRequest, ProresProfile, VIDEO_SHORT_SIDES, video_bitrate};

/// The short side of a preview render.
pub const PREVIEW_SIDE: u32 = 480;

/// The short side a render of a `width` by `height` canvas is drawn at:
/// [`PREVIEW_SIDE`], or for `full` the first export step that holds the
/// canvas's own short side.
#[must_use]
pub fn render_side(width: i32, height: i32, full: bool) -> u32 {
    let own = u32::try_from(width.min(height)).unwrap_or(0);
    if !full {
        return PREVIEW_SIDE;
    }
    VIDEO_SHORT_SIDES
        .iter()
        .copied()
        .find(|step| *step >= own)
        .unwrap_or(VIDEO_SHORT_SIDES[VIDEO_SHORT_SIDES.len() - 1])
}

/// A hash of the whole draft, the same on every run and machine.
#[must_use]
pub fn render_key(draft: &Draft) -> u64 {
    let bytes = serde_json::to_vec(draft).unwrap_or_default();
    // FNV-1a.
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The file name a render of `range` at `short_side` takes.
#[must_use]
pub fn render_name(key: u64, range: &TimeRange, short_side: u32) -> String {
    format!(
        "{key:016x}-{}-{}-{short_side}.mp4",
        range.start.0, range.duration.0
    )
}

/// The key, span and short side a render file name holds.
#[must_use]
pub fn read_render_name(name: &str) -> Option<(u64, TimeRange, u32)> {
    let stem = name.strip_suffix(".mp4")?;
    let mut parts = stem.split('-');
    let key = u64::from_str_radix(parts.next()?, 16).ok()?;
    let start = parts.next()?.parse::<i64>().ok()?;
    let duration = parts.next()?.parse::<i64>().ok()?;
    let side = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || start < 0 || duration <= 0 {
        return None;
    }
    Some((key, TimeRange::new(start, duration), side))
}

/// What a render of `range` writes: pictures only, in H.264, at the
/// project's rate, with room enough in the bitrate that it looks like the
/// timeline it stands in for.
#[must_use]
pub fn render_request(draft: &Draft, range: TimeRange, short_side: u32) -> ExportRequest {
    ExportRequest {
        range,
        short_side,
        codec: CodecId::H264,
        bitrate: video_bitrate(short_side).saturating_mul(2),
        frame_rate: draft.frame_rate().unwrap_or(FrameRate::FPS_30),
        container: ContainerId::Isobmff,
        audio: None,
        sidecar: None,
        late_by: 0,
        constant_rate: false,
        prores: ProresProfile::Standard,
    }
}

/// A draft that plays the render at `path`, of `range`, where the
/// timeline had it, filling a canvas the size of `draft`'s.
#[must_use]
pub fn stand_in(draft: &Draft, path: &Path, range: &TimeRange, key: u64) -> Draft {
    let mut stand = Draft::empty(format!("render-{key:016x}-{}", range.start.0));
    stand.canvas_config = draft.canvas_config.clone();
    stand.fps = draft.fps;
    let mut movie = MaterialVideo::new("render");
    movie.kind = "video".to_owned();
    movie.path = path.to_string_lossy().into_owned();
    movie.width = draft.canvas_config.width;
    movie.height = draft.canvas_config.height;
    movie.duration = range.duration;
    stand.materials.videos.push(movie);
    let mut clip = Segment::new("render", "render", range.clone());
    clip.source_range = Some(TimeRange::new(0, range.duration.0));
    let mut track = Track::main("render");
    track.segments.push(clip);
    stand.tracks = vec![track];
    stand.duration = Micros(range.start.0 + range.duration.0);
    stand
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_holds_its_key_span_and_side() {
        let range = TimeRange::new(2_000_000, 1_500_000);
        let name = render_name(0xabc, &range, 540);
        assert_eq!(name, "0000000000000abc-2000000-1500000-540.mp4");
        assert_eq!(read_render_name(&name), Some((0xabc, range, 540)));
        assert_eq!(read_render_name("cut.mp4"), None);
        assert_eq!(read_render_name("0abc-0-0-540.mp4"), None);
    }

    #[test]
    fn the_key_follows_the_draft_and_comes_back_with_it() {
        let draft = Draft::empty("d");
        let mut edited = draft.clone();
        edited.canvas_config.width += 2;
        assert_eq!(render_key(&draft), render_key(&draft.clone()));
        assert_ne!(render_key(&draft), render_key(&edited));
    }

    #[test]
    fn a_stand_in_plays_the_file_over_the_span_on_the_canvas() {
        let mut draft = Draft::empty("d");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        let range = TimeRange::new(1_000_000, 2_000_000);
        let stand = stand_in(&draft, Path::new("/tmp/r.mp4"), &range, 7);
        assert_eq!(stand.canvas_config, draft.canvas_config);
        let clip = &stand.tracks[0].segments[0];
        assert_eq!(clip.target, range);
        assert_eq!(clip.source_range, Some(TimeRange::new(0, 2_000_000)));
        assert_eq!(stand.materials.videos[0].path, "/tmp/r.mp4");
        assert_eq!(render_side(1920, 1080, true), 1080);
        assert_eq!(render_side(1080, 1920, false), PREVIEW_SIDE);
        assert_eq!(render_side(64, 32, true), 480);
        assert_eq!(render_side(9000, 9000, true), 4320);
        let request = render_request(&draft, range.clone(), PREVIEW_SIDE);
        assert!(crate::validate(&request).is_ok());
        assert_eq!((request.codec, request.audio), (CodecId::H264, None));
        assert_eq!(request.range, range);
    }
}
