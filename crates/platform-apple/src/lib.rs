//! VideoToolbox, AudioToolbox, and ImageIO behind [`cutlass_decode::Decoder`]
//! and [`cutlass_encode::Encoder`].
//!
//! The desktop registers these openers first. An opener that this framework
//! cannot run returns [`cutlass_decode::CodecError::Unsupported`], and the
//! factory asks the next opener. Linux builds this crate as that decline:
//! the frameworks are linked only on Apple.

#![forbid(unsafe_code)]

mod limits;
mod orient;
mod status;
