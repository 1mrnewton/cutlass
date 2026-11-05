//! VP8L, the WebP lossless bitstream (RFC 9649 section 3).

pub(crate) mod bits;
pub(crate) mod codes;
pub(crate) mod decode;
pub(crate) mod encode;
mod entropy;
pub(crate) mod huffman;
pub(crate) mod transform;
