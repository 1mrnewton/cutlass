//! Model code written by `tools/vision-models` from the pinned ONNX files.
//!
//! It is checked in as written; do not edit it by hand. It `unwrap`s and
//! `panic!`s on shapes it does not expect, which the workspace lints forbid
//! everywhere else. This module is the one exception: [`crate::mobile_sam`]
//! and [`crate::detect`] check every input before a call reaches this code,
//! and load the weights themselves so a bad file is an error.

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
pub(crate) mod face_detection_yunet_2023mar;

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
pub(crate) mod person_detection_mediapipe_2023mar;

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
pub(crate) mod palm_detection_mediapipe_2023feb;
