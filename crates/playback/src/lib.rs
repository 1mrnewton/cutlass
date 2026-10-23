//! The video frames kept ahead of a playhead, and the pictures already decoded.
//!
//! The caller supplies the playhead. When audio is playing, that time is the
//! audio clock. This crate does not open a decoder and it does not read a device.
//! A playhead still inside the open group of pictures keeps that decoder.
//! A later keyframe starts again at that keyframe. A move backward does too,
//! once a later frame is already queued or the decoder has finished.
//! [`FrameCache`] keeps pictures a reader has already decoded. [`FramePool`]
//! keeps pictures from more than one source.

#![deny(unsafe_code)]

mod ahead;
mod cache;
mod queue;
mod seek;

pub use ahead::{Ahead, Decode, FillError};
pub use cache::{FrameCache, FramePool, Remember};
pub use queue::{PushError, Queue};
pub use seek::{Reach, keyframe_before, reach};
