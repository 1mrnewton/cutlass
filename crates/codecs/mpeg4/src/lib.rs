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

pub use enc::encode_intra;
pub use engine::{EngineDecoder, register_engine_decoders};
