//! The session the shells share.
//!
//! The clock moves the playhead and lands it on a frame. The mix sums the
//! audible clips. Picture readers are still in the desktop app. This crate
//! does not open a file, a decoder, or a device, and it does not draw.

#![deny(unsafe_code)]

mod clock;
mod sound;

pub use clock::{Motion, Run, advance, frame_index, frame_time, frame_wait, snap, toggle};
pub use sound::{Cue, Lead, MixClip, Piece, clock_us, cues, mix, pieces, ticks_to_samples};
