//! Plays planar PCM on the output device.
//!
//! The device hears interleaved `f32`. This crate does that copy at the
//! boundary. Empty on non-Apple targets. Android, Windows, and Linux each
//! get their own output path later.

#![cfg(target_vendor = "apple")]

mod error;
mod queue;

pub use error::Error;
pub use queue::Output;
