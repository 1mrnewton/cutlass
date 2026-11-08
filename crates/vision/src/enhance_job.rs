//! The Enhance quality and Reduce image noise job: read a clip's stretch
//! frame by frame, denoise it from the frames around it, enhance it and
//! size it for the level, and write the copy into the project, where eval
//! draws the clip from it. A clip with both on gets one copy, denoised and
//! then enhanced, since the enhance model sharpens noise it is handed.
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
use image::{ImageBuffer, Rgb};

use crate::{Denoiser, Enhancer, Image, VisionError};

/// The short side of each level's copy.
const HD_SIDE: u32 = 1080;
const UHD_SIDE: u32 = 2160;

/// The noise each Reduce image noise level takes out, in 8-bit steps,
/// inside the 5 to 55 the model learned.
const WEAK_NOISE: f32 = 15.0;
const STRONG_NOISE: f32 = 30.0;

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
/// short side, or the clip's own size when it is already that large or the
/// copy is only denoised. Even, as H.264 codes it.
#[must_use]
pub fn enhanced_size(width: u32, height: u32, level: i64) -> (u32, u32) {
    let target = if level <= 0 {
        0
    } else if level >= Algorithm::UHD {
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

/// The spread of noise each Reduce image noise level takes out, in 8-bit
/// steps, as the denoiser reads it.
#[must_use]
pub fn noise_for_level(level: i64) -> f32 {
    if level >= Algorithm::STRONG {
        STRONG_NOISE
    } else {
        WEAK_NOISE
    }
}

/// The models a copy is made with. Each is needed only when its level is
/// on.
#[derive(Clone, Copy, Default)]
pub struct CopyModels<'a> {
    pub enhancer: Option<&'a Enhancer>,
    pub denoiser: Option<&'a Denoiser>,
}

/// Make `ask`'s copy in the project at `project` and return its path: the
/// clip denoised when its noise level is on, then enhanced when its
/// Enhance quality level is. `progress` hears the share done after each
/// frame, and stops the job by answering `false`.
pub fn enhance_clip(
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    models: CopyModels<'_>,
    ask: &EnhanceAsk,
    project: &Path,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let missing = |what: &str| {
        EnhanceJobError::Vision(VisionError::Model {
            message: format!("the {what} model is not loaded"),
        })
    };
    let enhancer = match (ask.level > 0, models.enhancer) {
        (false, _) => None,
        (true, Some(enhancer)) => Some(enhancer),
        (true, None) => return Err(missing("Enhance quality")),
    };
    let denoiser = match (ask.denoise > 0, models.denoiser) {
        (false, _) => None,
        (true, Some(denoiser)) => Some(denoiser),
        (true, None) => return Err(missing("Reduce image noise")),
    };
    if enhancer.is_none() && denoiser.is_none() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "Enhance quality and Reduce image noise are off for this clip".into(),
        }));
    }
    let mut frames = EnhanceFrames::open(readers, ask).map_err(EnhanceJobError::Clip)?;
    if frames.is_empty() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "the clip has no frames to enhance".into(),
        }));
    }
    let first = frames.frame(0).map_err(EnhanceJobError::Clip)?;
    let source = (first.width(), first.height());
    let (width, height) = enhanced_size(source.0, source.1, ask.level);
    let rate = FrameRate::from_stored_fps(ask.rate).ok_or_else(|| {
        EnhanceJobError::Clip(EnhanceError::Clip {
            reason: format!("{} frames a second is not a frame rate", ask.rate),
        })
    })?;
    let count = frames.len();
    let mut stream = denoiser
        .map(|denoiser| denoiser.stream(count, source.0, source.1, noise_for_level(ask.denoise)))
        .transpose()
        .map_err(EnhanceJobError::Vision)?;
    let movie = PictureMovie {
        width,
        height,
        frame_rate: rate,
        bitrate: video_bitrate(width.min(height)),
        frames: count as u64,
    };
    let file = enhanced_file(project, &ask.key(), &ask.range)
        .map_err(|error: StoreError| EnhanceJobError::Store(error.to_string()))?;
    let mut first = Some(first);
    // Frame `k` of the clip as RGB at the first frame's size.
    let mut read = |k: usize| -> Result<Vec<u8>, EnhanceJobError> {
        let picture = match first.take() {
            Some(picture) if k == 0 => picture,
            kept => {
                first = kept;
                frames.frame(k).map_err(EnhanceJobError::Clip)?
            }
        };
        if (picture.width(), picture.height()) != source {
            return Err(EnhanceJobError::Clip(EnhanceError::Frame {
                reason: format!(
                    "frame {k} is {}x{}, not {}x{}",
                    picture.width(),
                    picture.height(),
                    source.0,
                    source.1
                ),
            }));
        }
        Ok(rgb_of(picture.rgba()))
    };
    let mut failed = None;
    let total = count as f32;
    let written = write_pictures(
        encoders,
        &movie,
        &file,
        &mut |index| {
            let made = usize::try_from(index)
                .map_err(|_| {
                    EnhanceJobError::Clip(EnhanceError::Frame {
                        reason: format!("frame {index}"),
                    })
                })
                .and_then(|index| match stream.as_mut() {
                    Some(stream) => {
                        let mut lost = None;
                        let next = stream.next(&mut |k| {
                            read(k).map_err(|error| {
                                let message = error.to_string();
                                lost = Some(error);
                                VisionError::Model { message }
                            })
                        });
                        match next {
                            Ok(Some(rgb)) => Ok(rgb),
                            Ok(None) => Err(EnhanceJobError::Clip(EnhanceError::Frame {
                                reason: format!("the clip ended before frame {index}"),
                            })),
                            Err(error) => Err(lost.unwrap_or(EnhanceJobError::Vision(error))),
                        }
                    }
                    None => read(index),
                })
                .and_then(|rgb| match enhancer {
                    Some(enhancer) => enhance_one(enhancer, source, &rgb, (width, height)),
                    None => Ok(cropped_rgba(source, &rgb, (width, height))),
                });
            made.map_err(|error| {
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

fn rgb_of(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

/// A `size` frame of RGB as RGBA `out`, its odd last row or column left
/// off: a denoised copy keeps its clip's size.
fn cropped_rgba(size: (u32, u32), rgb: &[u8], out: (u32, u32)) -> Vec<u8> {
    let row = size.0 as usize * 3;
    rgb.chunks(row)
        .take(out.1 as usize)
        .flat_map(|line| {
            line.as_chunks::<3>()
                .0
                .iter()
                .take(out.0 as usize)
                .flat_map(|p| [p[0], p[1], p[2], 255])
        })
        .collect()
}

/// One frame of RGB through the enhance model, sized to `out`, as RGBA.
fn enhance_one(
    enhancer: &Enhancer,
    size: (u32, u32),
    rgb: &[u8],
    out: (u32, u32),
) -> Result<Vec<u8>, EnhanceJobError> {
    let model_error = |message: &str| {
        EnhanceJobError::Vision(VisionError::Model {
            message: message.into(),
        })
    };
    let (width, height) = size;
    // A frame already larger than the copy is brought down first, since
    // detail above the copy's size would be thrown away after the model.
    let (in_w, in_h) = if width.min(height) > out.0.min(out.1) {
        out
    } else {
        (width, height)
    };
    let shrunk;
    let input: &[u8] = if (in_w, in_h) == (width, height) {
        rgb
    } else {
        let source: ImageBuffer<Rgb<u8>, &[u8]> = ImageBuffer::from_raw(width, height, rgb)
            .ok_or_else(|| model_error("a frame does not fit its size"))?;
        shrunk = imageops::resize(&source, in_w, in_h, FilterType::Triangle);
        shrunk.as_raw()
    };
    let enhanced = enhancer
        .enhance(&Image {
            width: in_w,
            height: in_h,
            rgb: input,
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
