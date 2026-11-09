//! Model code written by `tools/vision-models` from the pinned ONNX files.
//!
//! It is checked in as written; do not edit it by hand. It `unwrap`s and
//! `panic!`s on shapes it does not expect, which the workspace lints forbid
//! everywhere else. This module is the one exception: [`crate::mobile_sam`]
//! checks every input before a call reaches this code, and loads the weights
//! itself so a bad file is an error.

#[allow(
    clippy::all,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::get_unwrap,
    rust_2018_idioms,
    unused,
    unused_qualifications,
    trivial_casts,
    trivial_numeric_casts
)]
#[rustfmt::skip]
pub(crate) mod mobile_sam_image_encoder;

#[allow(
    clippy::all,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::get_unwrap,
    rust_2018_idioms,
    unused,
    unused_qualifications,
    trivial_casts,
    trivial_numeric_casts
)]
#[rustfmt::skip]
pub(crate) mod sam_mask_decoder_lowres;
