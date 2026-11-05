//! VP8 key frames (RFC 6386), the lossy bitstream of a WebP.

mod analysis;
mod bool_decoder;
mod bool_encoder;
mod dsp;
pub(crate) mod encode;
mod forward;
mod frame;
mod header;
mod predict;
pub(crate) mod source;
mod tables;
mod tokens;
pub(crate) mod yuv;

pub(crate) use frame::decode;
