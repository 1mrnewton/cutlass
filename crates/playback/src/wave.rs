//! A clip's waveform over the columns a timeline shows, shared by every
//! shell so the desktop and the phones draw the same sound.

use std::ops::Range;
use std::path::{Path, PathBuf};

use cutlass_engine::Waveforms;
use cutlass_model::{Draft, MaterialRef, MetaType, Micros, SegmentId, TrackAttribute};

use crate::film::media_file;

/// Columns one wave carries at most.
const MAX_COLUMNS: usize = 16_384;

/// A clip's waveform over its visible columns.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipWave {
    /// The first column, from the clip's left edge.
    pub first: u32,
    /// One level, 0 to 1, per column.
    pub levels: Vec<f32>,
    /// The clip's gain at each column, its volume keyframes applied: 1 is
    /// the file as recorded.
    pub gains: Vec<f32>,
    /// The sound of a video clip, drawn in a band under its picture.
    pub under_picture: bool,
    /// The clip's track is muted, or silenced by another track's solo.
    pub muted: bool,
}

/// The waveform of clip `id`, `clip_width` columns wide at `pixel_size`
/// columns per microsecond: one level, 0 to 1, per column of `visible`,
/// columns from the clip's left edge, at the clip's volume.
///
/// A file not read yet is pushed onto `asks`, as is one still being read:
/// an audio clip has no waveform until its peaks arrive, a video clip an
/// empty one, so its band holds open. A video file with no sound, a photo,
/// a held frame, or an unknown id has none.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn clip_wave(
    draft: &Draft,
    project: Option<&Path>,
    waveforms: &Waveforms,
    id: &SegmentId,
    pixel_size: f64,
    clip_width: f64,
    visible: Range<f64>,
    asks: &mut Vec<PathBuf>,
) -> Option<ClipWave> {
    let (track, segment) = draft.tracks.iter().find_map(|track| {
        track
            .segments
            .iter()
            .find(|segment| &segment.id == id)
            .map(|segment| (track, segment))
    })?;
    let muted = track.attribute.contains(TrackAttribute::MUTE)
        || track.attribute.contains(TrackAttribute::MUTE_BY_SOLO);
    let under_picture = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Audio(_)) => false,
        // A held frame is silent: no wave, no volume line.
        Some(MaterialRef::Video(video)) if video.freeze.is_some() => return None,
        // So is a picture whose sound was taken off it.
        Some(MaterialRef::Video(video)) if video.has_sound_separated => return None,
        Some(MaterialRef::Video(video)) if video.kind != MetaType::Photo.as_str() => true,
        _ => return None,
    };
    let path = media_file(project, cutlass_engine::media_path(draft, segment))?;
    let map = cutlass_engine::placement_of(draft, segment).ok()?;
    let Some(peaks) = waveforms.get(&path) else {
        asks.push(path);
        return under_picture.then(|| ClipWave {
            first: 0,
            levels: Vec::new(),
            gains: Vec::new(),
            under_picture,
            muted,
        });
    };
    if !peaks.complete {
        asks.push(path);
    } else if peaks.levels.is_empty() {
        return None;
    }
    let usable = |value: f64| value.is_finite() && value > 0.0;
    if !usable(pixel_size) || !usable(clip_width) {
        return None;
    }
    let first = visible.start.max(0.0).floor();
    let last = visible.end.min(clip_width).ceil();
    let start = segment.target.start.0;
    let end = segment.target.end().0.saturating_sub(1).max(start);
    let at = |x: f64| {
        let time = Micros((start as f64 + x / pixel_size) as i64).min(Micros(end));
        cutlass_engine::map_timeline(&map, time).ok()
    };
    let mut levels = Vec::new();
    let mut gains = Vec::new();
    let mut x = first;
    while x < last && levels.len() < MAX_COLUMNS {
        let level = match (at(x), at(x + 1.0)) {
            (Some(from), Some(to)) => peaks.level(from, to),
            _ => 0.0,
        };
        let offset = Micros((x / pixel_size) as i64);
        let gain = cutlass_engine::clip_volume(segment, &draft.keyframe_graph_list, offset).max(0.0)
            as f32;
        levels.push((level * gain).min(1.0));
        gains.push(gain);
        x += 1.0;
    }
    Some(ClipWave {
        first: first as u32,
        levels,
        gains,
        under_picture,
        muted,
    })
}

#[cfg(test)]
mod tests {
    use cutlass_engine::{ReaderFactory, Waveforms};
    use cutlass_model::{FreezeInfo, MaterialVideo, Segment, SegmentId, TimeRange};

    use super::clip_wave;

    #[test]
    fn a_held_frame_has_no_wave_and_its_video_does() {
        let mut draft = crate::color_session(4_000_000, 4_000_000).draft().clone();
        draft.materials.videos.push(MaterialVideo {
            id: "movie".into(),
            path: "/movie.mp4".to_owned(),
            ..MaterialVideo::default()
        });
        draft.materials.videos.push(MaterialVideo {
            id: "held".into(),
            path: "/movie.mp4".to_owned(),
            freeze: Some(FreezeInfo::default()),
            ..MaterialVideo::default()
        });
        // A movie whose sound was extracted, on the same file.
        draft.materials.videos.push(MaterialVideo {
            id: "quiet".into(),
            path: "/movie.mp4".to_owned(),
            has_sound_separated: true,
            ..MaterialVideo::default()
        });
        for (id, material) in [("playing", "movie"), ("still", "held"), ("mute", "quiet")] {
            let mut segment = Segment::new(id, material, TimeRange::new(0, 4_000_000));
            segment.source_range = Some(TimeRange::new(0, 4_000_000));
            draft.tracks[0].segments.push(segment);
        }
        let waveforms = Waveforms::start(ReaderFactory::default(), || {});
        let mut asks = Vec::new();
        let mut wave = |id: &str| {
            clip_wave(
                &draft,
                None,
                &waveforms,
                &SegmentId::new(id),
                0.0001,
                400.0,
                0.0..400.0,
                &mut asks,
            )
        };
        // Its file is not read yet, so the playing clip holds its band open.
        assert!(wave("playing").is_some());
        assert!(wave("still").is_none());
        assert!(wave("mute").is_none(), "its sound is no longer on it");
    }
}
