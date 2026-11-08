//! The ALSA preview queue.
//!
//! [`OutputQueue`] plays 48 kHz stereo `f32` on the default ALSA device,
//! which on a desktop is usually PipeWire or PulseAudio behind ALSA's plugin.
//! `libasound.so.2` is loaded when the queue opens, so the build needs no
//! ALSA headers, and a system without it opens no device instead of failing
//! to start. Playback calls this safe queue. This crate does not mix samples.

mod slice;

#[cfg(target_os = "linux")]
mod alsa;
#[cfg(not(target_os = "linux"))]
mod off;

#[cfg(target_os = "linux")]
pub use alsa::OutputQueue;
#[cfg(not(target_os = "linux"))]
pub use off::OutputQueue;

pub use slice::QueueError;
