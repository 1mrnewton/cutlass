//! Probe a file and choose a decoder.
//!
//! The operating-system codec is asked first. The Rust codec is next. A codec
//! crate never makes that choice itself.

#![deny(unsafe_code)]

mod choose;
mod error;
mod probe;
mod session;

pub use choose::{DecoderKind, choose, choose_video_decoder};
pub use error::Error;
pub use probe::{Content, FrameTiming, Probe, TrackInfo};
pub use session::VideoSession;
