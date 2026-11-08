//! The steadying job: measure how the camera moved through a video clip,
//! for `cutlass_motion::plan` to steady.
//!
//! [`stable_ask`] names what to measure: the clip's whole file, from its
//! start to its end, at the project's frame rate. That is also what the
//! measured path is kept under, so a shell looks for a kept path first and
//! runs [`measure_camera`] only when there is none. The file is read
//! forward once, frame by frame in the file's own time, so a trim, a split
//! or a speed change on the timeline reuses the same path. Nothing here
//! writes the draft or the disk.

use std::path::{Path, PathBuf};

use cutlass_codec::ContainerId;
use cutlass_engine::{ReaderFactory, decoded_picture, is_graphic, media_in_project, media_path};
use cutlass_model::{Draft, MaterialRef, Micros, SegmentId, TimeRange};
use cutlass_motion::{Camera, Gray, MotionError, Path as CameraPath};
use cutlass_picture::Picture;

use crate::track::{project_rate, upright};

/// The most frames one clip may measure: over four hours at 60 a second.
const MAX_FRAMES: usize = 1_000_000;

/// Why a clip's camera could not be measured.
#[derive(Clone, Debug, PartialEq)]
pub enum StableError {
    /// The clip is not a video clip on the timeline, or has no file.
    Clip { reason: String },
    /// The file or one of its frames could not be read.
    Frame { reason: String },
    /// The camera path refused a frame.
    Motion(MotionError),
    /// The callback asked to stop.
    Stopped,
}

impl std::fmt::Display for StableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clip { reason } | Self::Frame { reason } => f.write_str(reason),
            Self::Motion(error) => write!(f, "{error}"),
            Self::Stopped => f.write_str("steadying was stopped"),
        }
    }
}

impl std::error::Error for StableError {}

/// What to measure: a stretch of one file at a frame rate.
#[derive(Clone, Debug, PartialEq)]
pub struct StableAsk {
    /// The clip's file on disk.
    pub source: PathBuf,
    /// The whole file, in its own time.
    pub range: TimeRange,
    /// Frames measured per second: the project's.
    pub rate: f64,
}

/// What steadying clip `segment_id` measures. A relative file stays inside
/// `root`.
pub fn stable_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Result<StableAsk, StableError> {
    let clip = |reason: &str| StableError::Clip {
        reason: reason.into(),
    };
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or_else(|| clip("the clip is not on the timeline"))?;
    if is_graphic(draft, segment) {
        return Err(clip("only a video clip can be steadied"));
    }
    let stored = media_path(draft, segment)
        .filter(|stored| !stored.is_empty())
        .ok_or_else(|| clip("only a video clip can be steadied"))?;
    let path = Path::new(stored);
    let source = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match root {
            Some(root) => media_in_project(root, stored)
                .ok_or_else(|| clip("the clip's file is not in the project folder"))?,
            None => path.to_path_buf(),
        }
    };
    // The whole file: the material's length, or at least as far as the
    // clip reads.
    let used = segment
        .source_range
        .as_ref()
        .filter(|range| range.start.0 >= 0 && range.duration.0 > 0)
        .map_or(0, |range| range.start.0.saturating_add(range.duration.0));
    let length = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => video.duration.0.max(used),
        _ => used,
    };
    if length <= 0 {
        return Err(clip("the clip has no length to steady"));
    }
    let range = TimeRange::new(0, length);
    Ok(StableAsk {
        source,
        range,
        rate: project_rate(draft),
    })
}

/// Measure `ask`'s camera. `progress` hears the share of frames read so far
/// after each one, and stops the job by answering `false`.
pub fn measure_camera(
    readers: &ReaderFactory,
    ask: &StableAsk,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<CameraPath, StableError> {
    let times = source_times(&ask.range, ask.rate)?;
    let mut media = readers
        .open(&ask.source)
        .map_err(|error| StableError::Frame {
            reason: format!("the clip's file could not be opened: {error}"),
        })?;
    if media.info().family == ContainerId::Still {
        return Err(StableError::Clip {
            reason: "a still picture has no shake to steady".into(),
        });
    }
    let display = media.info().display;
    let mut frame_at = |time: Micros| -> Result<Gray, StableError> {
        let unreadable = |what: &str| StableError::Frame {
            reason: format!("the clip's frame at {} µs {what}", time.0),
        };
        let held = media
            .decode_at(time, &ask.range)
            .map_err(|_| unreadable("could not be decoded"))?;
        let Ok(Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
            return Err(unreadable("is not in memory"));
        };
        let gray = Gray::from_rgba(picture.width(), picture.height(), picture.rgba())
            .map_err(StableError::Motion)?;
        upright(gray, display).map_err(StableError::Motion)
    };
    measure_frames(&times, ask.rate, &mut frame_at, progress)
}

/// The source times measured through `range`, `rate` a second from its
/// start.
pub(crate) fn source_times(range: &TimeRange, rate: f64) -> Result<Vec<Micros>, StableError> {
    if range.start.0 < 0 || range.duration.0 <= 0 {
        return Err(StableError::Clip {
            reason: "the clip has no length to steady".into(),
        });
    }
    if !(rate.is_finite() && rate > 0.0) {
        return Err(StableError::Clip {
            reason: format!("{rate} frames a second is not a frame rate"),
        });
    }
    let step = 1_000_000.0 / rate;
    let frames = (range.duration.0 as f64 / step).ceil();
    if frames > MAX_FRAMES as f64 {
        return Err(StableError::Clip {
            reason: "the clip is too long to steady".into(),
        });
    }
    let end = range.start.0.saturating_add(range.duration.0);
    Ok((0..frames as usize)
        .map(|k| Micros(range.start.0 + (k as f64 * step).round() as i64))
        .filter(|time| time.0 < end)
        .collect())
}

/// Measure through `times`, reading each frame with `frame_at`.
pub(crate) fn measure_frames(
    times: &[Micros],
    rate: f64,
    frame_at: &mut dyn FnMut(Micros) -> Result<Gray, StableError>,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<CameraPath, StableError> {
    let Some((&first, rest)) = times.split_first() else {
        return Err(StableError::Clip {
            reason: "the clip has no frames to steady".into(),
        });
    };
    let total = times.len() as f32;
    let mut camera = Camera::start(&frame_at(first)?, rate).map_err(StableError::Motion)?;
    if !progress(1.0 / total) {
        return Err(StableError::Stopped);
    }
    for (read, &time) in rest.iter().enumerate() {
        let frame = frame_at(time)?;
        camera.step(&frame).map_err(StableError::Motion)?;
        if !progress((read + 2) as f32 / total) {
            return Err(StableError::Stopped);
        }
    }
    Ok(camera.finish())
}

#[cfg(test)]
mod tests;
