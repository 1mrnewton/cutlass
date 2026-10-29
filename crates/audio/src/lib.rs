//! Turns decoded PCM into samples the mix can add.
//!
//! Each clip is resampled to the mix rate the caller chooses. Speed goes
//! through [`stretch`]: keeping pitch uses overlap-add, and letting the
//! pitch follow the speed is a resample. Fades are per clip. This crate
//! does not open a file.

#![forbid(unsafe_code)]

mod buffer;
mod fade;
mod resample;
mod stretch;

pub use fade::{FadeCurve, fade};
pub use resample::resample;
pub use stretch::stretch;

use std::fmt;

/// A PCM buffer this crate will not process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The sample rate was zero.
    Rate,
    /// Playback speed was not finite and greater than zero.
    Speed,
    /// A fade duration was negative, or the curve number is not 0 through 4.
    Fade,
    /// The output frame count does not fit in a PCM buffer.
    TooLong,
    /// A sample was NaN or infinite.
    NonFinite,
    /// The interleaved length does not match the channel count.
    Layout,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Rate => "sample rate is zero",
            Self::Speed => "speed must be finite and positive",
            Self::Fade => "fade duration or curve is not usable",
            Self::TooLong => "audio buffer is too long",
            Self::NonFinite => "non-finite sample",
            Self::Layout => "pcm layout does not match its samples",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for Error {}
