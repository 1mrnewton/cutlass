//! Enhance quality, as eval sees it: what a clip's enhanced copy is made
//! from, the clip's frames upright for the model, and which copy draws a
//! clip.
//!
//! [`enhance_ask`] names the copy: the stretch of the clip's file its
//! timeline shows, at the project's frame rate. The job reads that stretch
//! through [`EnhanceFrames`] and writes the copy where
//! `cutlass_store::enhanced_file` says. Nothing here runs the model or
//! writes the draft. Once the copy is there, a clip that asks for Enhance
//! quality at the copy's level draws from it, in preview and export alike;
//! until then it draws its own file.

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

/// What an enhanced copy is made from: a stretch of one file at a frame
/// rate and a level.
#[derive(Clone, Debug, PartialEq)]
pub struct EnhanceAsk {
    /// The clip's file on disk.
    pub source: PathBuf,
    /// The stretch the clip shows, in the file's own time.
    pub range: TimeRange,
    /// Pictures per second: the project's.
    pub rate: f64,
    /// The Enhance quality level.
    pub level: i64,
}

impl EnhanceAsk {
    /// The name its copy is kept under.
    #[must_use]
    pub fn key(&self) -> EnhanceKey {
        EnhanceKey {
            source: self.source.clone(),
            level: self.level,
            model: ENHANCE_MODEL.to_owned(),
            rate: self.rate,
        }
    }
}

/// What enhancing clip `segment_id` at `level` makes. A relative file
/// stays inside `root`.
pub fn enhance_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
    level: i64,
) -> Result<EnhanceAsk, EnhanceError> {
    let clip = |reason: &str| EnhanceError::Clip {
        reason: reason.into(),
    };
    if level <= 0 {
        return Err(clip("Enhance quality is off for this clip"));
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
    Ok(EnhanceAsk {
        source: file.source,
        range,
        rate: file.rate,
        level,
    })
}

/// The copy that draws `segment_id` and where in the file it starts, when
/// the clip asks for Enhance quality and its copy is in the project.
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
    let level = video.enhance()?.level;
    let ask = enhance_ask(draft, Some(root), segment_id, level).ok()?;
    let (path, held) = find_enhanced(root, &ask.key(), &ask.range)?;
    Some((path, held.start))
}

/// The frames of an [`EnhanceAsk`], upright, one at a time.
pub struct EnhanceFrames {
    media: Media,
    times: Vec<Micros>,
    range: TimeRange,
    display: DisplayQuarter,
}

impl EnhanceFrames {
    /// Opens `ask`'s file. A still picture is refused.
    pub fn open(readers: &ReaderFactory, ask: &EnhanceAsk) -> Result<Self, EnhanceError> {
        let times = source_times(&ask.range, ask.rate).map_err(|error| EnhanceError::Clip {
            reason: error.to_string().replace("steady", "enhance"),
        })?;
        let media = readers
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
        Ok(Self {
            media,
            times,
            range: ask.range.clone(),
            display,
        })
    }

    /// How many frames the copy holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Frame `index` of the copy, upright, as RGBA.
    pub fn frame(&mut self, index: usize) -> Result<CpuBytes, EnhanceError> {
        let time = *self.times.get(index).ok_or_else(|| EnhanceError::Frame {
            reason: format!("the copy has no frame {index}"),
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
