//! VA-API: video on Intel and AMD GPUs through libva.
//!
//! libva and its DRM backend are opened at run time, so a machine without
//! them builds and runs the same and every opener declines.

// Generated from the libva headers; it names more than is used.
#[allow(dead_code)]
mod abi;
mod api;
mod decode;
mod display;
pub mod h264;
pub mod hevc;
mod surface;

pub use display::report;
