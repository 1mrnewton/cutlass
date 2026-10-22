//! Probe a file and choose a decoder.
//!
//! The operating-system codec is asked first. The Rust codec is next. A codec
//! crate never makes that choice itself. `DeviceCodecs::query` asks Android
//! and Linux which codecs the device has. The chooser does not open those
//! decoders yet.

#![deny(unsafe_code)]

mod choose;
mod device;
mod error;
mod probe;
mod session;

pub use choose::{DecoderKind, choose, choose_video_decoder};
pub use device::DeviceCodecs;
pub use error::Error;
pub use probe::{Content, FrameTiming, Probe, TrackInfo};
pub use session::VideoSession;
