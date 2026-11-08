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
//! The fourth is Optical flow: [`Interpolator`] makes the frames between
//! two frames, for a clip at a higher frame rate or slowed down.
//!
//! The fifth is depth for Relight: [`DepthModel`] guesses how near each
//! pixel of a frame is, from which the light finds the way each surface
//! faces; [`relight_clip`] writes a clip's guesses as a grey video.

#![forbid(unsafe_code)]

mod denoise;
mod depth;
mod device;
mod enhance;
mod enhance_job;
mod frame;
mod generated;
mod interpolate;
mod job;
mod mask;
mod mobile_sam;
mod relight_job;

use std::fmt;
use std::path::PathBuf;

pub use denoise::{
    DENOISE_MAX_SIDE, DENOISE_WEIGHTS, DenoiseStream, Denoiser, MAX_NOISE, MIN_NOISE,
    convert_fastdvdnet,
};
pub use depth::{
    DEPTH_MAX_SIDE, DEPTH_SIDE, DEPTH_WEIGHTS, DepthMap, DepthModel, convert_depth_anything,
};
pub use device::{Compute, gpu_available, keep_kernels_in};
pub use enhance::{ENHANCE_MAX_SIDE, ENHANCE_WEIGHTS, Enhanced, Enhancer, SCALE, convert_pytorch};
pub use enhance_job::{CopyModels, EnhanceJobError, enhance_clip, enhanced_size, noise_for_level};
pub use frame::MAX_SIDE;
pub use interpolate::{
    INTERPOLATE_MAX_SIDE, INTERPOLATE_WEIGHTS, Interpolator, Prepared, convert_rife,
};
pub use job::{JobError, SelectJob};
pub use mask::Mask;
pub use mobile_sam::{DECODER_WEIGHTS, ENCODER_WEIGHTS, MobileSam};
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
        }
    }
}

impl std::error::Error for VisionError {}

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
