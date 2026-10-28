//! One frame at one microsecond.
//!
//! Each [`PictureSource`] is sampled at that time. A miss draws nothing.
//! Ready CPU bytes are drawn in the order given, with [`Blend::Over`]. A
//! platform buffer is refused by the software device. This crate does not
//! depend on a source crate. A transition between two pictures at a cut is
//! not implemented.

#![forbid(unsafe_code)]

mod blend;
mod frame;
mod seek;

pub use cutlass_gpu::{Blend, Device, Extent};
pub use cutlass_picture::PictureSource;
pub use frame::{ComposeError, Frame, submit};
