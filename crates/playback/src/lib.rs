//! The video frames kept ahead of a playhead.
//!
//! The caller supplies the playhead. When audio is playing, that time is the
//! audio clock. This crate does not decode and it does not read a device.

#![deny(unsafe_code)]

mod ahead;
mod queue;
mod seek;

pub use ahead::{Ahead, Decode, FillError};
pub use queue::{PushError, Queue};
pub use seek::keyframe_before;
