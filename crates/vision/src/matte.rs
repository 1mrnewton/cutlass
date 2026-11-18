//! Remove BG's models, as callers see them.
//!
//! A [`Matte`] says how much of each pixel of a frame is kept: 255 the
//! subject, 0 the background, and the soft values between on hair and
//! motion blur. Two kinds of model make one:
//!
//! - A [`Matter`] takes the background off one frame on its own, with no
//!   help: Auto removal.
//! - A [`Follower`] is told what to keep on some frames, by clicks or a
//!   painted mask, and carries that through the frames after it, keeping
//!   its own memory of the frames it has seen: Custom removal.
//!
//! Both answer at the frame's own size, so the engine behind them can
//! change without touching a caller.

use crate::{Click, Image, VisionError, check_clicks};

/// How much of each pixel of a frame is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matte {
    pub width: u32,
    pub height: u32,
    /// One byte per pixel, row by row: 255 kept, 0 removed.
    pub alpha: Vec<u8>,
}

impl Matte {
    /// A matte of `alpha`, when it holds exactly one byte per pixel.
    #[must_use]
    pub fn new(width: u32, height: u32, alpha: Vec<u8>) -> Option<Self> {
        (width > 0 && height > 0 && alpha.len() == width as usize * height as usize).then_some(
            Self {
                width,
                height,
                alpha,
            },
        )
    }

    /// The share of the frame kept, 0 to 1, soft pixels counted in part.
    #[must_use]
    pub fn coverage(&self) -> f64 {
        if self.alpha.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.alpha.iter().map(|a| u64::from(*a)).sum();
        #[allow(clippy::cast_precision_loss)]
        let share = sum as f64 / (self.alpha.len() as f64 * 255.0);
        share
    }
}

/// Takes the background off one frame.
pub trait Matter {
    /// The model and the size it reads at, kept in a matte video's name so
    /// a better model makes new ones.
    fn id(&self) -> &str;

    /// The matte of `image`, at its size.
    fn matte(&mut self, image: &Image<'_>) -> Result<Matte, VisionError>;
}

/// What the user said to keep on one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    /// Clicks in the frame's own pixels: `keep` adds what is under one,
    /// `false` takes it out.
    pub clicks: Vec<Click>,
    /// A painted matte at the frame's size: what the user brushed in.
    pub mask: Option<Matte>,
}

impl Prompt {
    /// Refuses a prompt the model cannot take for a `width` by `height`
    /// frame: nothing in it, bad clicks, or a mask of another size.
    pub fn check(&self, width: u32, height: u32) -> Result<(), VisionError> {
        match &self.mask {
            Some(mask) if mask.width != width || mask.height != height => {
                return Err(VisionError::Clicks {
                    reason: "a painted mask of another size than the frame",
                });
            }
            Some(_) if self.clicks.is_empty() => return Ok(()),
            None if self.clicks.is_empty() => {
                return Err(VisionError::Clicks { reason: "none" });
            }
            _ => {}
        }
        check_clicks(&self.clicks, width, height)
    }
}

/// Follows what the user picked from frame to frame.
pub trait Follower {
    /// The model and the size it reads at, as [`Matter::id`].
    fn id(&self) -> &str;

    /// Forgets every frame seen, to start a new run.
    fn reset(&mut self);

    /// The matte of the next frame of the run. The first frame of a run
    /// must carry a prompt; a later one may, to correct the run there.
    fn step(&mut self, image: &Image<'_>, prompt: Option<&Prompt>) -> Result<Matte, VisionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matte_holds_one_byte_a_pixel() {
        assert!(Matte::new(2, 2, vec![0; 4]).is_some());
        assert!(Matte::new(2, 2, vec![0; 3]).is_none());
        assert!(Matte::new(0, 2, Vec::new()).is_none());
    }

    #[test]
    fn coverage_counts_soft_pixels_in_part() {
        let matte = Matte::new(4, 1, vec![255, 0, 255, 0]).unwrap();
        assert!((matte.coverage() - 0.5).abs() < 1e-9);
        let soft = Matte::new(2, 1, vec![255, 51]).unwrap();
        assert!((soft.coverage() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn a_prompt_needs_clicks_or_a_mask_of_the_frame_size() {
        let click = Click {
            x: 1.0,
            y: 1.0,
            keep: true,
        };
        assert!(Prompt::default().check(4, 4).is_err());
        let clicks = Prompt {
            clicks: vec![click],
            mask: None,
        };
        assert!(clicks.check(4, 4).is_ok());
        let painted = Prompt {
            clicks: Vec::new(),
            mask: Matte::new(4, 4, vec![255; 16]),
        };
        assert!(painted.check(4, 4).is_ok());
        assert!(painted.check(8, 4).is_err());
        let outside = Prompt {
            clicks: vec![Click { x: 9.0, ..click }],
            mask: None,
        };
        assert!(outside.check(4, 4).is_err());
    }
}
