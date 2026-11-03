//! MediaCodec decoders and encoders, and the AAudio preview queue.
//!
//! [`OutputQueue`] is a shared-mode AAudio stream. Playback calls that safe
//! queue. This crate does not mix samples. MediaCodec openers are not in
//! this crate yet.

mod slice;

#[cfg(target_os = "android")]
mod aaudio;
#[cfg(not(target_os = "android"))]
mod off;

#[cfg(target_os = "android")]
pub use aaudio::OutputQueue;
#[cfg(not(target_os = "android"))]
pub use off::OutputQueue;

pub use slice::QueueError;
