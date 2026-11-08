//! MPEG-4 Part 2 (MPEG-4 Visual, ISO/IEC 14496-2) decoder and encoder.

#![forbid(unsafe_code)]

mod bits;
mod block;
mod dec;
mod enc;
mod engine;
mod frame;
mod header;
mod idct;
mod mc;
mod tables;
mod vlc;
mod vop;

#[cfg(test)]
mod prove;

pub use enc::{encode_advanced, encode_intra};
pub use engine::{
    ADVANCED_SIMPLE, EngineDecoder, EngineEncoder, register_engine_decoders,
    register_engine_encoders,
};
