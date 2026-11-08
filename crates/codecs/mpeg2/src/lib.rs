//! MPEG-2 video (ISO/IEC 13818-2, ITU-T H.262) and MPEG-1 video decoding,
//! and MPEG-2 Main profile encoding.

#![forbid(unsafe_code)]

mod bits;
mod dct;
mod decoder;
mod engine;
pub mod es;
mod frame;
mod headers;
mod mc;
mod slice;
mod tables;

pub use decoder::{DEFAULT_MAX_PIXELS, Mpeg2Decoder};
pub use engine::register_decoders;
