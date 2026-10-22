//! The video frames kept ahead of a playhead.
//!
//! The caller supplies the playhead. When audio is playing, that time is the
//! audio clock. This crate does not open a decoder and it does not read a device.
//! A playhead still inside the open group of pictures keeps that decoder.
//! A later keyframe, or a move backward, starts again at the keyframe.

#![deny(unsafe_code)]

mod ahead;
mod queue;
mod seek;

pub use ahead::{Ahead, Decode, FillError};
pub use queue::{PushError, Queue};
pub use seek::{Reach, keyframe_before, reach};
