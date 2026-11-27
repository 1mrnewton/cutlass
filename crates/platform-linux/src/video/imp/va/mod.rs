//! VA-API: video on Intel and AMD GPUs through libva.
//!
//! libva and its DRM backend are opened at run time, so a machine without
//! them builds and runs the same and every opener declines.

// Generated from the libva headers; it names more than is used.
#[allow(dead_code)]
mod abi;
mod api;
mod bits;
mod decode;
mod display;
mod encode;
#[cfg(test)]
mod encode_tests;
pub mod h264;
pub mod h264_enc;
mod header;
pub mod hevc;
pub mod hevc_enc;
mod surface;

pub use display::report;
