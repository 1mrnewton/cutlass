//! Apple openers, or a decline when the frameworks are not linked.

#[cfg(not(target_vendor = "apple"))]
mod off;
#[cfg(not(target_vendor = "apple"))]
pub use off::{register_decoders, register_encoders};

#[cfg(target_vendor = "apple")]
#[allow(unsafe_code)]
mod imp;
#[cfg(target_vendor = "apple")]
pub use imp::{
    OutputQueue, QueueError, decode_file, encode_file, register_decoders, register_encoders,
    release_buffer, to_cpu, with_pixel_buffer,
};
