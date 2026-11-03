//! Media Foundation openers, or a decline when the platform is not Windows.

#[cfg(not(target_os = "windows"))]
mod off;
#[cfg(not(target_os = "windows"))]
pub use off::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
mod imp;
#[cfg(target_os = "windows")]
pub use imp::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};
