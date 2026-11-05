//! Apple openers, or a decline when the frameworks are not linked.

#[cfg(not(target_vendor = "apple"))]
mod off;
#[cfg(not(target_vendor = "apple"))]
pub use off::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
    register_surface_decoders_with,
};

#[cfg(target_vendor = "apple")]
#[allow(unsafe_code)]
mod imp;
#[cfg(target_vendor = "apple")]
pub use imp::{
    OutputQueue, QueueError, StreamSink, decode_file, encode_file, import_surface,
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
    register_surface_decoders_with, release_buffer, run_at_interface_priority, to_cpu,
    with_pixel_buffer,
};
