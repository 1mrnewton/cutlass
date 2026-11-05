//! VP8 key frames (RFC 6386) to 8-bit 4:2:0 planes.
//!
//! This is the lossy picture inside a WebP file: one key frame, decoded
//! exactly as the RFC's reconstruction defines, loop filter included. The
//! planes come back cropped to the coded size, with chroma rounded up to
//! whole samples. Inter frames are refused as unsupported: a WebP never
//! carries one, and this crate is not registered as a video decoder.
//!
//! Every table is transcribed from the RFC text, section by section, in
//! `tables`. A frame larger than the caller's pixel limit is refused
//! before anything is allocated. A partition that runs out before the
//! frame is done is [`CodecError::Truncated`].

#![forbid(unsafe_code)]

mod bool_decoder;
mod decode;
mod filter;
mod predict;
mod tables;
mod transform;

pub use cutlass_decode::CodecError;
pub use decode::{decode_frame, frame_size};

/// One decoded frame: tightly packed Y, U, and V planes.
///
/// `y` is `width` by `height`. `u` and `v` are each
/// [`Self::chroma_width`] by [`Self::chroma_height`]. Samples are the
/// coded values, BT.601 studio range as VP8 defines them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vp8Picture {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Vp8Picture {
    #[must_use]
    pub fn chroma_width(&self) -> usize {
        usize::try_from(self.width.div_ceil(2)).unwrap_or(0)
    }

    #[must_use]
    pub fn chroma_height(&self) -> usize {
        usize::try_from(self.height.div_ceil(2)).unwrap_or(0)
    }
}
