//! Media Foundation decoders and encoders, and the WASAPI preview queue.
//!
//! [`OutputQueue`] is a shared-mode WASAPI stream. Playback calls that safe
//! queue. This crate does not mix samples. Media Foundation openers are not
//! in this crate yet.

mod format;
mod slice;

#[cfg(not(target_os = "windows"))]
mod off;
#[cfg(target_os = "windows")]
mod wasapi;

#[cfg(not(target_os = "windows"))]
pub use off::OutputQueue;
#[cfg(target_os = "windows")]
pub use wasapi::OutputQueue;

pub use slice::QueueError;
