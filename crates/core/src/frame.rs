//! Decoded frames.

use crate::Timestamp;

/// A decoded video frame. `I` is the image storage, which depends on the
/// decoder: a GPU-shareable buffer from a native decoder, CPU planes from a
/// software one.
#[derive(Debug, Clone)]
pub struct VideoFrame<I> {
    pub pts: Timestamp,
    pub duration: Option<Timestamp>,
    pub image: I,
}
