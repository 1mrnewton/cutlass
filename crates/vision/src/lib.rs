//! On-device image models for editing.
//!
//! Nothing here leaves the machine. The first model is click-to-select: the
//! user clicks a point in a frame and gets a mask of the object under it.
//! [`Selector`] is that boundary. It takes RGB pixels and clicks in source
//! pixels, and returns a [`Mask`] at the source size, so the engine behind it
//! can change without touching a caller.
//!
//! A selection runs in two steps. [`Selector::encode`] reads the frame once.
//! [`Selector::select`] turns clicks into a mask and runs on every click.
//!
//! The second is Enhance quality: [`Enhancer`] makes a frame four times
//! larger and repairs blur, noise and compression on the way.
//!
//! The third is Reduce image noise: [`Denoiser`] cleans each frame of a
//! clip from the frames around it, so the noise it takes out stays still.
//!
//! Remove flicker needs no model: [`Deflicker`] evens out each frame's
//! brightness with the frames around it.
//!
//! The fourth is Optical flow: [`Interpolator`] makes the frames between
//! two frames, for a clip at a higher frame rate or slowed down.
//!
//! The fifth is depth for Relight: [`DepthModel`] guesses how near each
//! pixel of a frame is, from which the light finds the way each surface
//! faces; [`relight_clip`] writes a clip's guesses as a grey video, and
//! [`find_subjects`] finds what stands out nearest for Auto reframe.
//!
//! The sixth is finding a face, a body or a hand for tracking to start from:
//! [`Detector`] runs YuNet or MediaPipe's person or palm detector.
//!
//! The seventh is Remove BG: a [`Matter`] takes the background off a frame
//! as a soft [`Matte`], and a [`Follower`] carries what the user picked
//! through a clip: [`Portrait`] and [`Matting`] are the first, and
//! [`FollowModel`] the second.
//!
//! One model hears instead of sees. Auto beats: [`BeatModel`] finds where
//! the beats of a piece of music fall, from its samples.

#![forbid(unsafe_code)]

mod beats;
mod deflicker;
mod denoise;
mod depth;
mod detect;
mod device;
mod enhance;
mod enhance_job;
mod follow;
mod frame;
mod generated;
mod gpu_check;
mod interpolate;
mod job;
mod mask;
mod matte;
mod matte_frame;
mod matte_job;
mod matting;
mod mobile_sam;
mod portrait;
mod reframe_job;
mod relight_job;

use std::fmt;
use std::path::PathBuf;

pub use beats::{
    BEATS_MAX_SAMPLES, BEATS_MIN_SAMPLES, BEATS_RATE, BEATS_WEIGHTS, BeatModel, Beats,
    FRAMES_A_SECOND, convert_beat_this,
};
pub use deflicker::{DEFLICKER_REACH, Deflicker, mean_brightness};
pub use denoise::{
    DENOISE_MAX_SIDE, DENOISE_WEIGHTS, DenoiseStream, Denoiser, MAX_NOISE, MIN_NOISE,
    convert_fastdvdnet,
};
pub use depth::{
    DEPTH_MAX_SIDE, DEPTH_SIDE, DEPTH_WEIGHTS, DepthMap, DepthModel, convert_depth_anything,
};
pub use detect::{BODY_WEIGHTS, Detector, FACE_WEIGHTS, Found, HAND_WEIGHTS, Part};
pub use device::{Compute, gpu_available, gpu_trusted, keep_kernels_in, trust_gpu};
pub use enhance::{ENHANCE_MAX_SIDE, ENHANCE_WEIGHTS, Enhanced, Enhancer, SCALE, convert_pytorch};
pub use enhance_job::{CopyModels, EnhanceJobError, enhance_clip, enhanced_size, noise_for_level};
pub use follow::{FOLLOW_MODEL, FOLLOW_SIDE, FOLLOW_WEIGHTS, FollowModel, convert_sam2};
pub use frame::MAX_SIDE;
pub use gpu_check::{GpuCheck, check_gpu};
pub use interpolate::{
    INTERPOLATE_MAX_SIDE, INTERPOLATE_WEIGHTS, Interpolator, Prepared, convert_rife,
};
pub use job::{JobError, SelectJob};
pub use mask::Mask;
pub use matte::{Follower, Matte, Matter, Prompt};
pub use matte_frame::MATTE_MAX_SIDE;
pub use matte_job::{MATTE_SIDE, follow_clip, matte_clip, matte_size};
pub use matting::{
    MATTING_MODEL, MATTING_MULTIPLE, MATTING_SIDE, MATTING_WEIGHTS, Matting, convert_birefnet,
};
pub use mobile_sam::{DECODER_WEIGHTS, ENCODER_WEIGHTS, MobileSam};
pub use portrait::{PORTRAIT_MODEL, PORTRAIT_SIDE, PORTRAIT_WEIGHTS, Portrait, convert_modnet};
pub use reframe_job::{REFRAME_SIDE, find_subjects, subject_of};
pub use relight_job::{RELIGHT_SIDE, relight_clip};

/// The most clicks one selection takes.
pub const MAX_CLICKS: usize = 32;

/// RGB pixels, three bytes each, row by row with no padding.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [u8],
}

/// A click in the frame's own pixels. `keep` adds the object under it, and
/// `false` takes it out of the selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Click {
    pub x: f32,
    pub y: f32,
    pub keep: bool,
}

/// Selects an object in a frame from clicks.
pub trait Selector {
    /// Reads `image`. Later clicks select in this frame until the next call.
    fn encode(&mut self, image: &Image<'_>) -> Result<(), VisionError>;

    /// The mask for `clicks` in the last encoded frame, at that frame's size.
    fn select(&self, clicks: &[Click]) -> Result<Mask, VisionError>;
}

/// Why a frame or a selection failed.
#[derive(Clone, Debug, PartialEq)]
pub enum VisionError {
    /// A side is zero or above [`MAX_SIDE`], or `rgb` is not `width * height * 3` bytes.
    Image {
        width: u32,
        height: u32,
        bytes: usize,
    },
    /// No clicks, more than [`MAX_CLICKS`], or one that is not a finite point in the frame.
    Clicks { reason: &'static str },
    /// `select` ran before any frame was encoded.
    NoFrame,
    /// A weights file is missing or does not fit the model.
    Weights { path: PathBuf, message: String },
    /// This build has no GPU path for the model.
    NoGpu,
    /// The kernel cache folder could not be made.
    Cache { path: PathBuf, message: String },
    /// The model returned something other than the shapes it was built for.
    Model { message: String },
    /// A sound shorter than [`BEATS_MIN_SAMPLES`], longer than
    /// [`BEATS_MAX_SAMPLES`], or with a sample that is not a number.
    Sound { samples: usize },
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image {
                width,
                height,
                bytes,
            } => write!(
                f,
                "a {width}x{height} image with {bytes} bytes of RGB is not one the model can read"
            ),
            Self::Clicks { reason } => write!(f, "clicks refused: {reason}"),
            Self::NoFrame => f.write_str("no frame has been encoded"),
            Self::Weights { path, message } => write!(f, "{}: {message}", path.display()),
            Self::NoGpu => f.write_str("this build has no GPU path for the vision models"),
            Self::Cache { path, message } => {
                write!(f, "kernel cache {}: {message}", path.display())
            }
            Self::Model { message } => write!(f, "model: {message}"),
            Self::Sound { samples } => write!(
                f,
                "a sound of {samples} samples is not one the model can hear"
            ),
        }
    }
}

impl std::error::Error for VisionError {}

/// `x`'s values read back from its device. A failure the device met while
/// computing them (out of memory, a lost device) is an error here, where
/// burn's own `into_data` panics.
pub(crate) fn read_back<const D: usize>(
    x: burn::tensor::Tensor<D>,
) -> Result<burn::tensor::TensorData, VisionError> {
    x.try_into_data().map_err(|error| VisionError::Model {
        message: format!("the device could not finish: {error}"),
    })
}

/// Refuses clicks the model cannot take, before they reach it.
pub(crate) fn check_clicks(clicks: &[Click], width: u32, height: u32) -> Result<(), VisionError> {
    if clicks.is_empty() {
        return Err(VisionError::Clicks { reason: "none" });
    }
    if clicks.len() > MAX_CLICKS {
        return Err(VisionError::Clicks {
            reason: "more than MAX_CLICKS",
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (width as f32, height as f32);
    for click in clicks {
        if !click.x.is_finite() || !click.y.is_finite() {
            return Err(VisionError::Clicks {
                reason: "not a finite point",
            });
        }
        if click.x < 0.0 || click.y < 0.0 || click.x > w || click.y > h {
            return Err(VisionError::Clicks {
                reason: "outside the frame",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn click(x: f32, y: f32) -> Click {
        Click { x, y, keep: true }
    }

    #[test]
    fn clicks_inside_the_frame_pass() {
        assert_eq!(
            check_clicks(&[click(0.0, 0.0), click(64.0, 32.0)], 64, 32),
            Ok(())
        );
    }

    #[test]
    fn no_clicks_are_refused() {
        assert_eq!(
            check_clicks(&[], 64, 32),
            Err(VisionError::Clicks { reason: "none" })
        );
    }

    #[test]
    fn too_many_clicks_are_refused() {
        let clicks = vec![click(1.0, 1.0); MAX_CLICKS + 1];
        assert!(check_clicks(&clicks[..MAX_CLICKS], 64, 32).is_ok());
        assert!(matches!(
            check_clicks(&clicks, 64, 32),
            Err(VisionError::Clicks { .. })
        ));
    }

    #[test]
    fn a_click_outside_or_not_finite_is_refused() {
        for bad in [
            click(-0.5, 1.0),
            click(1.0, 32.5),
            click(64.5, 1.0),
            click(f32::NAN, 1.0),
            click(1.0, f32::INFINITY),
        ] {
            assert!(
                matches!(
                    check_clicks(&[click(1.0, 1.0), bad], 64, 32),
                    Err(VisionError::Clicks { .. })
                ),
                "{bad:?}"
            );
        }
    }
}
