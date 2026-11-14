//! The Enhance quality, Reduce image noise, Remove flicker, Optical flow
//! and Motion blur job: read a clip's stretch frame by frame, even out its
//! brightness, denoise it from the frames around it, make the frames between its own that the copy's rate asks
//! for, blur each frame by averaging frames made across its blur in linear
//! light, enhance it and size it for the level, and write the copy into
//! the project, where eval draws the clip from it. A clip with more than
//! one on gets one copy, in that order: the brightness is evened before
//! the denoiser reads it, the enhance model sharpens noise
//! it is handed, and frames are made from the clip's own before they are
//! grown, at a quarter or less of the pixels.
//!
//! The job needs no session and writes no draft, so a shell runs it off
//! the main thread. A stopped or failed job leaves no copy behind.

use std::fmt;
use std::path::{Path, PathBuf};

use cutlass_eval::{Blend, EnhanceAsk, EnhanceError, EnhanceFrames, ReaderFactory};
use cutlass_export::{ExportError, PictureMovie, video_bitrate, write_pictures};
use cutlass_model::{Algorithm, FrameRate};
use cutlass_store::{StoreError, enhanced_file};
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Rgb};

use crate::{Deflicker, Denoiser, Enhancer, Image, Interpolator, Prepared, VisionError};

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
    pub interpolator: Option<&'a Interpolator>,
}

/// Make `ask`'s copy in the project at `project` and return its path: the
/// clip's brightness evened out when Remove flicker is on, denoised when
/// its noise level is, filled with frames made
/// between its own when Optical flow is, then enhanced when its Enhance
/// quality level is. `progress` hears the share done after each
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
    let flows = ask.interpolate > 0 || ask.motion_blur.is_some();
    let interpolator = match (flows, models.interpolator) {
        (false, _) => None,
        (true, Some(interpolator)) => Some(interpolator),
        (true, None) if ask.interpolate > 0 => return Err(missing("Optical flow")),
        (true, None) => return Err(missing("Motion blur")),
    };
    if enhancer.is_none() && denoiser.is_none() && interpolator.is_none() && ask.deflicker <= 0 {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "Enhance quality, Reduce image noise, Remove flicker, Optical flow and Motion \
                     blur are off for this clip"
                .into(),
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
    let blends = frames.blends().to_vec();
    let samples = frames.samples().to_vec();
    let blurred = ask.motion_blur.is_some();
    let mut stream = denoiser
        .map(|denoiser| denoiser.stream(count, source.0, source.1, noise_for_level(ask.denoise)))
        .transpose()
        .map_err(EnhanceJobError::Vision)?;
    let movie = PictureMovie {
        width,
        height,
        frame_rate: rate,
        bitrate: video_bitrate(width.min(height)),
        frames: blends.len() as u64,
    };
    let file = enhanced_file(project, &ask.key(), &ask.range)
        .map_err(|error: StoreError| EnhanceJobError::Store(error.to_string()))?;
    let mut first = Some(first);
    // Frame `k` of the clip as RGB at the first frame's size.
    let mut decode = |k: usize| -> Result<Vec<u8>, EnhanceJobError> {
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
    // Its brightness evened out first, when Remove flicker is on.
    let mut deflicker = (ask.deflicker > 0).then(|| Deflicker::new(count));
    let mut read = |k: usize| -> Result<Vec<u8>, EnhanceJobError> {
        match deflicker.as_mut() {
            Some(deflicker) => deflicker.frame(k, &mut decode),
            None => decode(k),
        }
    };
    // The clip's own frames in order, denoised when that is on.
    let mut own = |index: usize| -> Result<Vec<u8>, EnhanceJobError> {
        match stream.as_mut() {
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
        }
    };
    let mut window = Window::default();
    let mut failed = None;
    let total = blends.len() as f32;
    let written = write_pictures(
        encoders,
        &movie,
        &file,
        &mut |index| {
            let made = usize::try_from(index)
                .ok()
                .and_then(|index| blends.get(index).copied())
                .ok_or_else(|| {
                    EnhanceJobError::Clip(EnhanceError::Frame {
                        reason: format!("the copy has no frame {index}"),
                    })
                })
                .and_then(|blend| {
                    let across = usize::try_from(index)
                        .ok()
                        .and_then(|index| samples.get(index))
                        .filter(|_| blurred);
                    let first = across
                        .and_then(|across| across.first())
                        .map_or(blend.from, |sample| sample.from.min(blend.from));
                    window.forget_before(first);
                    let sharp = window.frame(blend, count, source, interpolator, &mut own)?;
                    let Some(across) = across else {
                        return Ok(sharp);
                    };
                    let mut made = Vec::with_capacity(across.len());
                    for sample in across {
                        made.push(window.frame(*sample, count, source, interpolator, &mut own)?);
                    }
                    // A keyed Blend mixes each frame's own share back in.
                    #[allow(clippy::cast_possible_truncation)]
                    let mix =
                        usize::try_from(index).map_or(0.0, |index| ask.blend_at(index) as f32);
                    Ok(shutter(&made, &sharp, mix))
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

/// The clip's own frames a frame of the copy is made from: the last two
/// read, each prepared for the interpolator once it is needed.
#[derive(Default)]
struct Window {
    /// The next of the clip's own frames to read.
    next: usize,
    kept: Vec<Own>,
}

struct Own {
    index: usize,
    rgb: Vec<u8>,
    prepared: Option<Prepared>,
}

impl Window {
    /// Drop the frames before `index`, which no later frame of the copy
    /// reads.
    fn forget_before(&mut self, index: usize) {
        self.kept.retain(|kept| kept.index >= index);
    }

    /// Frame `blend` of the copy as RGB: one of the clip's `count` frames
    /// as read by `own`, or one made between two of them.
    fn frame(
        &mut self,
        blend: Blend,
        count: usize,
        size: (u32, u32),
        interpolator: Option<&Interpolator>,
        own: &mut dyn FnMut(usize) -> Result<Vec<u8>, EnhanceJobError>,
    ) -> Result<Vec<u8>, EnhanceJobError> {
        let made = blend.at > 0.0 && blend.from + 1 < count;
        let last = if made { blend.from + 1 } else { blend.from };
        while self.next <= last.min(count.saturating_sub(1)) {
            let rgb = own(self.next)?;
            self.kept.push(Own {
                index: self.next,
                rgb,
                prepared: None,
            });
            self.next += 1;
        }
        let interpolator = match interpolator {
            Some(interpolator) if made => interpolator,
            _ => {
                return self
                    .kept
                    .iter()
                    .find(|kept| kept.index == blend.from)
                    .map(|kept| kept.rgb.clone())
                    .ok_or_else(|| lost(blend.from));
            }
        };
        for kept in &mut self.kept {
            if kept.prepared.is_none() {
                let image = Image {
                    width: size.0,
                    height: size.1,
                    rgb: &kept.rgb,
                };
                kept.prepared = Some(
                    interpolator
                        .prepare(&image)
                        .map_err(EnhanceJobError::Vision)?,
                );
            }
        }
        let prepared = |index: usize| {
            self.kept
                .iter()
                .find(|kept| kept.index == index)
                .and_then(|kept| kept.prepared.as_ref())
                .ok_or_else(|| lost(index))
        };
        interpolator
            .between(prepared(blend.from)?, prepared(blend.from + 1)?, blend.at)
            .map_err(EnhanceJobError::Vision)
    }
}

/// The average of `frames` in linear light, with `mix` of `sharp` put
/// back. All are RGB of one size; with none, `sharp`.
fn shutter(frames: &[Vec<u8>], sharp: &[u8], mix: f32) -> Vec<u8> {
    if frames.is_empty() || frames.iter().any(|frame| frame.len() != sharp.len()) {
        return sharp.to_vec();
    }
    let linear: Vec<f32> = (0..=255_u8)
        .map(|byte| (f32::from(byte) / 255.0).powf(2.2))
        .collect();
    let share = 1.0 / frames.len() as f32;
    let mix = mix.clamp(0.0, 1.0);
    (0..sharp.len())
        .map(|i| {
            let mean: f32 = frames
                .iter()
                .map(|frame| linear[usize::from(frame[i])])
                .sum::<f32>()
                * share;
            let value = mean * (1.0 - mix) + linear[usize::from(sharp[i])] * mix;
            (value.max(0.0).powf(1.0 / 2.2) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn lost(index: usize) -> EnhanceJobError {
    EnhanceJobError::Clip(EnhanceError::Frame {
        reason: format!("the clip's frame {index} was read past"),
    })
}

pub(crate) fn rgb_of(rgba: &[u8]) -> Vec<u8> {
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
