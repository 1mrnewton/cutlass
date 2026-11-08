//! MPEG-2 video (ISO/IEC 13818-2, ITU-T H.262) and MPEG-1 video decoding,
//! and MPEG-2 Main profile encoding.

#![forbid(unsafe_code)]

mod bits;
mod dct;
mod decoder;
mod encoder;
mod engine;
pub mod es;
mod frame;
mod headers;
mod mc;
mod motion;
mod picture;
mod rate;
mod slice;
mod tables;

pub use decoder::{DEFAULT_MAX_PIXELS, Mpeg2Decoder};
pub use encoder::{Mpeg2Encoder, main_profile_level, register_encoders};
pub use engine::register_decoders;
