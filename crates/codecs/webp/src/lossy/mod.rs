//! VP8 key frames (RFC 6386), the lossy bitstream of a WebP.

mod bool_decoder;
mod dsp;
mod frame;
mod header;
mod predict;
mod tables;
pub(crate) mod yuv;

pub(crate) use frame::decode;
