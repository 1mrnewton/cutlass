//! The session the shells share.
//!
//! The clock moves the playhead and lands it on a frame. The mix sums the
//! audible clips. On Apple the picture readers choose the frame for each
//! track. The shell uploads that frame and draws. This crate does not draw.

#![deny(unsafe_code)]

mod clock;
mod path;
#[cfg(target_vendor = "apple")]
mod picture;
mod sound;

pub use clock::{Motion, Run, advance, frame_index, frame_time, frame_wait, snap, toggle};
pub use path::file_path;
#[cfg(target_vendor = "apple")]
pub use picture::{Ask, Body, LayerFrame, Pictures, Proxy, ShowLayer, ShowNext, shows_frame};
pub use sound::{Cue, Lead, MixClip, Piece, clock_us, cues, mix, pieces, ticks_to_samples};
