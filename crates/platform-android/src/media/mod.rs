//! MediaCodec openers, or a decline when the platform is not Android.

pub(crate) mod clock;
pub(crate) mod kind;

#[cfg(not(target_os = "android"))]
mod off;
#[cfg(not(target_os = "android"))]
pub use off::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

#[cfg(target_os = "android")]
#[allow(unsafe_code)]
mod imp;
#[cfg(target_os = "android")]
pub use imp::{
    VideoDecoder, VideoEncoder, open_decoder_on, open_encoder_on, register_decoders,
    register_decoders_with, register_encoders, register_encoders_with,
};
