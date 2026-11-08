//! Enhance quality, Reduce image noise and Optical flow, as eval sees
//! them: what a clip's copy is made from, the clip's frames upright for
//! the models, and which copy draws a clip.
//!
//! [`enhance_ask`] names the copy: the stretch of the clip's file its
//! timeline shows, at a frame rate, and the three levels. A clip with more
//! than one on has one copy, denoised, then filled with frames made
//! between its own, then enhanced. The copy's rate is the project's, or
//! with Optical flow on its frame rate over the clip's slowest speed, so a
//! slowed clip still plays that many different frames a second. The job
//! reads that stretch through [`EnhanceFrames`] and writes the copy where
//! `cutlass_store::enhanced_file` says. Nothing here runs a model or
//! writes the draft. Once the copy is there, a clip that asks for the
//! copy's levels draws from it, in preview and export alike; until then it
//! draws its own file.

use std::path::{Path, PathBuf};

use cutlass_codec::ContainerId;
use cutlass_engine::{DisplayQuarter, Media, ReaderFactory, decoded_picture};
use cutlass_model::{Draft, MaterialRef, Micros, SegmentId, TimeRange};
use cutlass_picture::{CpuBytes, Picture};
use cutlass_store::{EnhanceKey, find_enhanced};

use crate::stable::{StableError, source_times, stable_ask};

/// The model enhanced copies are made with. A copy made by another is not
/// used.
pub const ENHANCE_MODEL: &str = "realesr-general-x4v3";

/// The model denoised copies are made with.
pub const DENOISE_MODEL: &str = "fastdvdnet-clipped";

/// The model that makes the frames between a clip's own for Optical flow.
pub const INTERPOLATE_MODEL: &str = "rife-v4.26";

/// The slowest speed Optical flow fills for, so a copy's rate stays
/// bounded.
const SLOWEST: f64 = 0.01;

/// How many places along a speed curve are read for its slowest speed.
const SPEED_SAMPLES: usize = 256;

/// A frame made this close to one of the clip's own, as a share of the
/// time between them, is that frame.
const SNAP: f64 = 0.02;

/// The most of a file's own frames read for one copy.
const MAX_FILE_FRAMES: usize = 1_000_000;

/// Why a clip could not be enhanced.
#[derive(Clone, Debug, PartialEq)]
pub enum EnhanceError {
    /// The clip is not a video clip on the timeline, or has no file.
    Clip { reason: String },
    /// The file or one of its frames could not be read.
    Frame { reason: String },
}

impl std::fmt::Display for EnhanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clip { reason } | Self::Frame { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for EnhanceError {}

/// What a copy is made from: a stretch of one file at a frame rate, and
/// the levels it is enhanced, denoised and interpolated at.
#[derive(Clone, Debug, PartialEq)]
pub struct EnhanceAsk {
    /// The clip's file on disk.
    pub source: PathBuf,
    /// The stretch the clip shows, in the file's own time.
    pub range: TimeRange,
    /// Pictures per second of the file's time in the copy: the project's,
    /// or with Optical flow on its frame rate over the clip's slowest
    /// speed, whichever is more.
    pub rate: f64,
    /// The Enhance quality level; 0 makes a copy that is only denoised.
    pub level: i64,
    /// The Reduce image noise level; 0 makes a copy that is not denoised.
    pub denoise: i64,
    /// The Optical flow frame rate; 0 makes a copy with only the clip's
    /// own frames.
    pub interpolate: i64,
}

impl EnhanceAsk {
    /// The name its copy is kept under.
    #[must_use]
    pub fn key(&self) -> EnhanceKey {
        // In the order the job runs them.
        let model = [
            (self.denoise > 0, DENOISE_MODEL),
            (self.interpolate > 0, INTERPOLATE_MODEL),
            (self.level > 0, ENHANCE_MODEL),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, model)| *model)
        .collect::<Vec<_>>()
        .join("+");
        EnhanceKey {
            source: self.source.clone(),
            level: self.level.max(0),
            denoise: self.denoise.max(0),
            interpolate: self.interpolate.max(0),
            model,
            rate: self.rate,
        }
    }
}

/// What enhancing clip `segment_id` at `level`, denoising it at `denoise`
/// and filling it to `interpolate` frames a second makes. A relative file
/// stays inside `root`.
pub fn enhance_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
    level: i64,
    denoise: i64,
    interpolate: i64,
) -> Result<EnhanceAsk, EnhanceError> {
    let clip = |reason: &str| EnhanceError::Clip {
        reason: reason.into(),
    };
    if level <= 0 && denoise <= 0 && interpolate <= 0 {
        return Err(clip(
            "Enhance quality, Reduce image noise and Optical flow are off for this clip",
        ));
    }
    let file = stable_ask(draft, root, segment_id).map_err(|error| match error {
        StableError::Clip { reason } => EnhanceError::Clip {
            reason: reason.replace("steadied", "enhanced"),
        },
        other => EnhanceError::Frame {
            reason: other.to_string(),
        },
    })?;
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or_else(|| clip("the clip is not on the timeline"))?;
    let range = segment
        .source_range
        .clone()
        .filter(|range| range.start.0 >= 0 && range.duration.0 > 0)
        .ok_or_else(|| clip("the clip shows none of its file"))?;
    let rate = if interpolate > 0 {
        let filled = interpolate as f64 / slowest_speed(draft, segment);
        filled.max(file.rate)
    } else {
        file.rate
    };
    Ok(EnhanceAsk {
        source: file.source,
        range,
        rate,
        level: level.max(0),
        denoise: denoise.max(0),
        interpolate: interpolate.max(0),
    })
}

/// The slowest `segment` plays its file anywhere along it, no slower than
/// [`SLOWEST`]. A held frame and a map that will not read play at 1.
fn slowest_speed(draft: &Draft, segment: &cutlass_model::Segment) -> f64 {
    let Ok(map) = cutlass_engine::placement_of(draft, segment) else {
        return 1.0;
    };
    let (start, length) = (map.target.start.0, map.target.duration.0.max(0));
    (0..=SPEED_SAMPLES)
        .filter_map(|k| {
            let at = start + (length as f64 * k as f64 / SPEED_SAMPLES as f64).round() as i64;
            cutlass_engine::speed_at(&map, Micros(at)).ok()
        })
        .filter(|speed| speed.is_finite() && *speed > 0.0)
        .reduce(f64::min)
        .map_or(1.0, |speed| speed.max(SLOWEST))
}

/// The copy that draws `segment_id` and where in the file it starts, when
/// the clip asks for Enhance quality or Reduce image noise and its copy is
/// in the project.
pub(crate) fn enhanced_copy(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Option<(PathBuf, Micros)> {
    let root = root?;
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let level = |algorithm: Option<&cutlass_model::Algorithm>| algorithm.map_or(0, |a| a.level);
    let ask = enhance_ask(
        draft,
        Some(root),
        segment_id,
        level(video.enhance()),
        level(video.denoise()),
        level(video.interpolate()),
    )
    .ok()?;
    let (path, held) = find_enhanced(root, &ask.key(), &ask.range)?;
    Some((path, held.start))
}

/// One frame of a copy: frame `from` of the clip, or with `at` above 0
/// the frame made `at` of the way from it to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    pub from: usize,
    pub at: f32,
}

/// The frames of an [`EnhanceAsk`], upright, one at a time: the clip's
/// own, and for each frame of the copy, which of them it is or lies
/// between.
pub struct EnhanceFrames {
    media: Media,
    times: Vec<Micros>,
    blends: Vec<Blend>,
    range: TimeRange,
    display: DisplayQuarter,
}

impl EnhanceFrames {
    /// Opens `ask`'s file. A still picture is refused. With Optical flow
    /// on, the clip's own frames are the file's, found by reading through
    /// the stretch once; otherwise they are the copy's.
    pub fn open(readers: &ReaderFactory, ask: &EnhanceAsk) -> Result<Self, EnhanceError> {
        let copy = source_times(&ask.range, ask.rate).map_err(|error| EnhanceError::Clip {
            reason: error.to_string().replace("steady", "enhance"),
        })?;
        let mut media = readers
            .open(&ask.source)
            .map_err(|error| EnhanceError::Frame {
                reason: format!("the clip's file could not be opened: {error}"),
            })?;
        if media.info().family == ContainerId::Still {
            return Err(EnhanceError::Clip {
                reason: "only a video clip can be enhanced".into(),
            });
        }
        let display = media.info().display;
        let (times, blends) = if ask.interpolate > 0 {
            let times = file_times(&mut media, &ask.range)?;
            let blends = blends(&times, &copy);
            (times, blends)
        } else {
            let blends = (0..copy.len())
                .map(|from| Blend { from, at: 0.0 })
                .collect();
            (copy, blends)
        };
        Ok(Self {
            media,
            times,
            blends,
            range: ask.range.clone(),
            display,
        })
    }

    /// How many of the clip's own frames there are to read.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Each frame of the copy, in order. Each one's `from` is at least the
    /// last one's.
    #[must_use]
    pub fn blends(&self) -> &[Blend] {
        &self.blends
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The clip's own frame `index`, upright, as RGBA.
    pub fn frame(&mut self, index: usize) -> Result<CpuBytes, EnhanceError> {
        let time = *self.times.get(index).ok_or_else(|| EnhanceError::Frame {
            reason: format!("the clip has no frame {index}"),
        })?;
        let unreadable = |what: &str| EnhanceError::Frame {
            reason: format!("the clip's frame at {} µs {what}", time.0),
        };
        let held = self
            .media
            .decode_at(time, &self.range)
            .map_err(|_| unreadable("could not be decoded"))?;
        let Ok(Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
            return Err(unreadable("is not in memory"));
        };
        upright_rgba(&picture, self.display).ok_or_else(|| unreadable("could not be turned"))
    }
}

/// When each of the file's own frames in `range` starts, and the one after
/// the stretch when there is one, for the frames made at its end.
fn file_times(media: &mut Media, range: &TimeRange) -> Result<Vec<Micros>, EnhanceError> {
    let end = range.start.0.saturating_add(range.duration.0);
    let mut times: Vec<Micros> = Vec::new();
    let mut at = range.start;
    for _ in 0..MAX_FILE_FRAMES {
        let held = media
            .decode_at(at, range)
            .map_err(|error| EnhanceError::Frame {
                reason: format!(
                    "the clip's frame at {} µs could not be decoded: {error}",
                    at.0
                ),
            })?;
        if times.last().is_some_and(|last| held.start.0 <= last.0) {
            break;
        }
        times.push(held.start);
        if held.start.0 >= end || held.end.0 <= at.0 {
            break;
        }
        at = held.end;
    }
    if times.is_empty() {
        return Err(EnhanceError::Frame {
            reason: "the clip has no frames".into(),
        });
    }
    Ok(times)
}

/// Where each time of `copy` lies among the file's frames at `times`: on
/// one, or a share of the way to the next. A time before the first frame
/// is the first, and one past the last is the last.
fn blends(times: &[Micros], copy: &[Micros]) -> Vec<Blend> {
    copy.iter()
        .map(|time| {
            let from = times
                .partition_point(|start| start.0 <= time.0)
                .saturating_sub(1);
            let Some(next) = times.get(from + 1) else {
                return Blend { from, at: 0.0 };
            };
            let start = times.get(from).map_or(time.0, |start| start.0);
            let share = ((time.0 - start) as f64 / (next.0 - start).max(1) as f64).clamp(0.0, 1.0);
            if share < SNAP {
                Blend { from, at: 0.0 }
            } else if share > 1.0 - SNAP {
                Blend {
                    from: from + 1,
                    at: 0.0,
                }
            } else {
                Blend {
                    from,
                    at: share as f32,
                }
            }
        })
        .collect()
}

/// `picture` turned by its file's display quarter, so it stands as shown.
pub(crate) fn upright_rgba(picture: &CpuBytes, display: DisplayQuarter) -> Option<CpuBytes> {
    if display == DisplayQuarter::Zero {
        return Some(picture.clone());
    }
    let (w, h) = (picture.width() as usize, picture.height() as usize);
    let (out_w, out_h) = match display {
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (h, w),
        DisplayQuarter::Zero | DisplayQuarter::Half => (w, h),
    };
    let src = picture.rgba();
    let mut out = vec![0_u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = match display {
                DisplayQuarter::Zero => (x, y),
                DisplayQuarter::Clockwise90 => (h - 1 - y, x),
                DisplayQuarter::Half => (w - 1 - x, h - 1 - y),
                DisplayQuarter::Clockwise270 => (y, w - 1 - x),
            };
            let from = (y * w + x) * 4;
            let to = (oy * out_w + ox) * 4;
            out.get_mut(to..to + 4)?
                .copy_from_slice(src.get(from..from + 4)?);
        }
    }
    CpuBytes::new(u32::try_from(out_w).ok()?, u32::try_from(out_h).ok()?, out).ok()
}

#[cfg(test)]
mod tests;
