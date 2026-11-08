//! The Relight job: read a clip's stretch frame by frame, guess how near
//! each pixel is, and write the guesses into the project as a grey video,
//! nearer lighter, where eval lights the clip from it.
//!
//! The model's guesses have no units and drift from frame to frame, so each
//! frame is spread from its own nearly-farthest to its nearly-nearest pixel,
//! and those two ease from frame to frame so the light does not pump. The
//! video is the size the model reads at, which is plenty for light and
//! small to keep.
//!
//! The job needs no session and writes no draft, so a shell runs it off the
//! main thread. A stopped or failed job leaves no depth video behind.

use std::path::{Path, PathBuf};

use cutlass_eval::{EnhanceError, EnhanceFrames, ReaderFactory, RelightAsk};
use cutlass_export::{ExportError, PictureMovie, video_bitrate, write_pictures};
use cutlass_model::FrameRate;
use cutlass_store::relight_file;

use crate::enhance_job::rgb_of;
use crate::{DepthMap, DepthModel, EnhanceJobError, Image};

/// The short side each frame is read at: about 65 ms a 1080p frame on
/// Metal, and enough for light.
pub const RELIGHT_SIDE: u32 = 364;

/// The share of pixels left out at each end of a frame's spread, so a stray
/// pixel does not squash the rest.
const TAIL: f32 = 0.02;

/// How much of each frame's own spread the eased one takes.
const EASE: f32 = 0.25;

/// The fewest steps between the farthest and nearest kept, so a flat frame
/// does not blow up its noise.
const MIN_SPREAD: f32 = 1e-3;

/// Make `ask`'s depth video in the project at `project` with `model` and
/// return its path. `progress` hears the share done after each frame, and
/// stops the job by answering `false`.
pub fn relight_clip(
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    model: &DepthModel,
    ask: &RelightAsk,
    project: &Path,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let mut frames = EnhanceFrames::open(readers, &ask.frames()).map_err(|error| {
        EnhanceJobError::Clip(match error {
            EnhanceError::Clip { reason } => EnhanceError::Clip {
                reason: reason.replace("enhanced", "relit"),
            },
            other => other,
        })
    })?;
    if frames.is_empty() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "the clip has no frames to relight".into(),
        }));
    }
    let rate = FrameRate::from_stored_fps(ask.rate).ok_or_else(|| {
        EnhanceJobError::Clip(EnhanceError::Clip {
            reason: format!("{} frames a second is not a frame rate", ask.rate),
        })
    })?;
    let count = frames.len();
    let mut depth = |k: usize| -> Result<DepthMap, EnhanceJobError> {
        let picture = frames.frame(k).map_err(EnhanceJobError::Clip)?;
        let rgb = rgb_of(picture.rgba());
        let image = Image {
            width: picture.width(),
            height: picture.height(),
            rgb: &rgb,
        };
        model
            .estimate_at(&image, RELIGHT_SIDE)
            .map_err(EnhanceJobError::Vision)
    };
    let first = depth(0)?;
    let (width, height) = ((first.width & !1).max(2), (first.height & !1).max(2));
    if first.width < 2 || first.height < 2 {
        return Err(EnhanceJobError::Clip(EnhanceError::Frame {
            reason: format!("a depth of {}x{} is too small", first.width, first.height),
        }));
    }
    let movie = PictureMovie {
        width,
        height,
        frame_rate: rate,
        bitrate: video_bitrate(width.min(height)),
        frames: count as u64,
    };
    let file = relight_file(project, &ask.key(), &ask.range)
        .map_err(|error| EnhanceJobError::Store(error.to_string()))?;
    let mut first = Some(first);
    let mut spread: Option<(f32, f32)> = None;
    let mut failed = None;
    let total = count as f32;
    let written = write_pictures(
        encoders,
        &movie,
        &file,
        &mut |index| {
            let made = usize::try_from(index)
                .map_err(|_| lost(index))
                .and_then(|k| match first.take() {
                    Some(map) if k == 0 => Ok(map),
                    kept => {
                        first = kept;
                        depth(k)
                    }
                })
                .and_then(|map| {
                    if (map.width & !1, map.height & !1) != (width, height) {
                        return Err(EnhanceJobError::Clip(EnhanceError::Frame {
                            reason: format!(
                                "frame {index}'s depth is {}x{}, not {width}x{height}",
                                map.width, map.height
                            ),
                        }));
                    }
                    let own = tails(&map.values);
                    let eased = match spread {
                        Some((lo, hi)) => (lo + (own.0 - lo) * EASE, hi + (own.1 - hi) * EASE),
                        None => own,
                    };
                    spread = Some(eased);
                    Ok(grey(&map, (width, height), eased))
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

fn lost(index: u64) -> EnhanceJobError {
    EnhanceJobError::Clip(EnhanceError::Frame {
        reason: format!("the clip has no frame {index}"),
    })
}

/// The values [`TAIL`] in from each end of `values`, ignoring any that are
/// not numbers.
pub(crate) fn tails(values: &[f32]) -> (f32, f32) {
    let mut kept: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if kept.is_empty() {
        return (0.0, 1.0);
    }
    let last = kept.len() - 1;
    let at = |share: f32| ((last as f32 * share).round() as usize).min(last);
    let (lo, hi) = (at(TAIL), at(1.0 - TAIL));
    let low = *kept.select_nth_unstable_by(lo, f32::total_cmp).1;
    let high = *kept.select_nth_unstable_by(hi, f32::total_cmp).1;
    (low, high.max(low + MIN_SPREAD))
}

/// `map` at `size`, its odd last row or column left off, spread from `lo`
/// as black to `hi` as white, as grey RGBA.
fn grey(map: &DepthMap, size: (u32, u32), (lo, hi): (f32, f32)) -> Vec<u8> {
    let (w, h) = (size.0 as usize, size.1 as usize);
    let stride = map.width as usize;
    let scale = 255.0 / (hi - lo).max(MIN_SPREAD);
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let v = map.values.get(y * stride + x).copied().unwrap_or(lo);
            let byte = if v.is_finite() {
                ((v - lo) * scale).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
            out.extend_from_slice(&[byte, byte, byte, 255]);
        }
    }
    out
}

#[cfg(test)]
mod tests;
