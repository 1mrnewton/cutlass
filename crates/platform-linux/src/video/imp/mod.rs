//! The Linux hardware openers: NVIDIA first, then VA-API.

mod blob;
mod cuda;
mod nvdec;
mod nvenc;
// Generated from nvEncodeAPI.h; it names more of the header than is used.
#[allow(dead_code)]
mod nvenc_abi;
mod probe;
#[cfg(test)]
mod tests;
mod va;

pub use probe::probe;

use cutlass_decode::{CodecId, DecodeFactory};
use cutlass_encode::EncodeFactory;

use super::backend::Backends;

/// Register the hardware decoders.
pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false registers nothing, so the factory asks the next opener.
/// [`super::BACKEND_VAR`] narrows which backends register.
pub fn register_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if !video {
        return;
    }
    let backends = Backends::from_env();
    if backends.nvidia {
        for id in [CodecId::H264, CodecId::Hevc, CodecId::Av1] {
            factory.register(id, nvdec::open_decoder);
        }
    }
    if backends.vaapi {
        factory.register(CodecId::H264, va::h264::open_decoder);
        factory.register(CodecId::Hevc, va::hevc::open_decoder);
    }
}

/// Register the hardware encoders.
pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false registers nothing, so the factory asks the next opener.
/// [`super::BACKEND_VAR`] narrows which backends register.
pub fn register_encoders_with(factory: &mut EncodeFactory, video: bool) {
    if !video {
        return;
    }
    let backends = Backends::from_env();
    if backends.nvidia {
        for id in [CodecId::H264, CodecId::Hevc] {
            factory.register(id, nvenc::open_encoder);
        }
    }
    if backends.vaapi {
        factory.register(CodecId::H264, va::h264_enc::open_encoder);
        factory.register(CodecId::Hevc, va::hevc_enc::open_encoder);
    }
}
