//! Decoded frames.

use crate::Timestamp;
use crate::color::ColorTags;

/// A decoded video frame. `I` is the image storage, which depends on the
/// decoder: a GPU-shareable buffer from a native decoder, CPU planes from a
/// software one.
#[derive(Debug, Clone)]
pub struct VideoFrame<I> {
    pub pts: Timestamp,
    pub duration: Option<Timestamp>,
    /// Primaries, transfer, and matrix from the container. `None` when it
    /// gave no numeric tags. The decoder copies these. It does not read them
    /// from the bitstream.
    pub color: Option<ColorTags>,
    pub image: I,
}
