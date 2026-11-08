//! The Enhance quality job: read a clip's stretch frame by frame, run the
//! model on each, size it for the level, and write the enhanced copy into
//! the project, where eval draws the clip from it.
//!
//! The job needs no session and writes no draft, so a shell runs it off
//! the main thread. A stopped or failed job leaves no copy behind.

use std::fmt;
use std::path::{Path, PathBuf};

use cutlass_eval::{EnhanceAsk, EnhanceError, EnhanceFrames, ReaderFactory};
use cutlass_export::{ExportError, PictureMovie, video_bitrate, write_pictures};
use cutlass_model::{Algorithm, FrameRate};
use cutlass_store::{StoreError, enhanced_file};
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Rgb, Rgba};

use crate::{Enhancer, Image, VisionError};

/// The short side of each level's copy.
const HD_SIDE: u32 = 1080;
const UHD_SIDE: u32 = 2160;

/// Why a clip was not enhanced.
#[derive(Debug, PartialEq)]
pub enum EnhanceJobError {
    Clip(EnhanceError),
    Vision(VisionError),
    Movie(ExportError),
    /// The project folder would not take the copy.
    Store(String),
    /// The callback asked to stop.
    Stopped,
}

impl fmt::Display for EnhanceJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clip(error) => write!(f, "{error}"),
            Self::Vision(error) => write!(f, "{error}"),
            Self::Movie(error) => write!(f, "{error}"),
            Self::Store(reason) => f.write_str(reason),
            Self::Stopped => f.write_str("enhancing was stopped"),
        }
    }
}

impl std::error::Error for EnhanceJobError {}

/// The size a `width` by `height` clip's copy is at `level`: the level's
/// short side, or the clip's own size when it is already that large. Even,
/// as H.264 codes it.
#[must_use]
pub fn enhanced_size(width: u32, height: u32, level: i64) -> (u32, u32) {
    let target = if level >= Algorithm::UHD {
        UHD_SIDE
    } else {
        HD_SIDE
    };
    let short = width.min(height).max(1);
    let (w, h) = if short >= target {
        (width, height)
    } else {
        let scale = f64::from(target) / f64::from(short);
        let grow = |side: u32| (f64::from(side) * scale).round() as u32;
        (grow(width), grow(height))
    };
    ((w & !1).max(2), (h & !1).max(2))
}

/// Enhance `ask` into the project at `project` and return the copy's path.
/// `progress` hears the share done after each frame, and stops the job by
/// answering `false`.
pub fn enhance_clip(
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    enhancer: &Enhancer,
    ask: &EnhanceAsk,
    project: &Path,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let mut frames = EnhanceFrames::open(readers, ask).map_err(EnhanceJobError::Clip)?;
    if frames.is_empty() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "the clip has no frames to enhance".into(),
        }));
    }
    let first = frames.frame(0).map_err(EnhanceJobError::Clip)?;
    let (width, height) = enhanced_size(first.width(), first.height(), ask.level);
    let rate = FrameRate::from_stored_fps(ask.rate).ok_or_else(|| {
        EnhanceJobError::Clip(EnhanceError::Clip {
            reason: format!("{} frames a second is not a frame rate", ask.rate),
        })
    })?;
    let movie = PictureMovie {
        width,
        height,
        frame_rate: rate,
        bitrate: video_bitrate(width.min(height)),
        frames: frames.len() as u64,
    };
    let file = enhanced_file(project, &ask.key(), &ask.range)
        .map_err(|error: StoreError| EnhanceJobError::Store(error.to_string()))?;
    let mut failed = None;
    let mut first = Some(first);
    let total = frames.len() as f32;
    let written = write_pictures(
        encoders,
        &movie,
        &file,
        &mut |index| {
            let picture = match first.take() {
                Some(picture) => Ok(picture),
                None => usize::try_from(index)
                    .map_err(|_| EnhanceError::Frame {
                        reason: format!("frame {index}"),
                    })
                    .and_then(|index| frames.frame(index)),
            };
            let enhanced = picture.map_err(EnhanceJobError::Clip).and_then(|picture| {
                enhance_one(
                    enhancer,
                    picture.width(),
                    picture.height(),
                    picture.rgba(),
                    (width, height),
                )
            });
            enhanced.map_err(|error| {
                let message = error.to_string();
                failed = Some(error);
                ExportError::Picture(message)
            })
        },
        &mut |done, _| progress(done as f32 / total),
    );
    match written {
        Ok(()) => Ok(file),
        Err(ExportError::Cancelled) => Err(EnhanceJobError::Stopped),
        Err(error) => Err(failed.unwrap_or(EnhanceJobError::Movie(error))),
    }
}

/// One frame through the model, sized to `out`, as RGBA.
fn enhance_one(
    enhancer: &Enhancer,
    width: u32,
    height: u32,
    rgba: &[u8],
    out: (u32, u32),
) -> Result<Vec<u8>, EnhanceJobError> {
    let model_error = |message: &str| {
        EnhanceJobError::Vision(VisionError::Model {
            message: message.into(),
        })
    };
    let source: ImageBuffer<Rgba<u8>, &[u8]> = ImageBuffer::from_raw(width, height, rgba)
        .ok_or_else(|| model_error("a frame does not fit its size"))?;
    // A frame already larger than the copy is brought down first, since
    // detail above the copy's size would be thrown away after the model.
    let (in_w, in_h) = if width.min(height) > out.0.min(out.1) {
        out
    } else {
        (width, height)
    };
    let rgb: Vec<u8> = if (in_w, in_h) == (width, height) {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect()
    } else {
        imageops::resize(&source, in_w, in_h, FilterType::Triangle)
            .as_raw()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect()
    };
    let enhanced = enhancer
        .enhance(&Image {
            width: in_w,
            height: in_h,
            rgb: &rgb,
        })
        .map_err(EnhanceJobError::Vision)?;
    let grown: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(enhanced.width, enhanced.height, enhanced.rgb)
            .ok_or_else(|| model_error("the model's frame does not fit its size"))?;
    let sized = if (enhanced.width, enhanced.height) == out {
        grown
    } else {
        imageops::resize(&grown, out.0, out.1, FilterType::Triangle)
    };
    Ok(sized
        .as_raw()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect())
}

#[cfg(test)]
mod tests;
