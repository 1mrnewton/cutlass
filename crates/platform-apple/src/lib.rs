//! VideoToolbox, AudioToolbox, and ImageIO behind [`cutlass_decode::Decoder`]
//! and [`cutlass_encode::Encoder`].
//!
//! The desktop registers these openers first. An opener that this framework
//! cannot run returns [`cutlass_decode::CodecError::Unsupported`], and the
//! factory asks the next opener.

mod apple;
mod limits;
mod nal;
mod orient;
mod status;

pub use apple::{register_decoders, register_encoders};

#[cfg(target_vendor = "apple")]
pub use apple::{decode_file, encode_file};
