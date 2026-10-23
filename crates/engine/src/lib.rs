//! The session the shells share.
//!
//! The clock moves the playhead and lands it on a frame. Picture readers and
//! the mix are still in the desktop app. This crate does not open a file, a
//! decoder, or a device, and it does not draw.

#![deny(unsafe_code)]

mod clock;

pub use clock::{Motion, Run, advance, frame_index, frame_time, frame_wait, snap, toggle};
