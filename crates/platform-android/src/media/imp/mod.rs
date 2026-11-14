//! MediaCodec through the NDK.

mod decoder;
mod encoder;
mod ndk;

#[cfg(test)]
mod tests;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub use decoder::{VideoDecoder, open_decoder_on};
pub use encoder::{VideoEncoder, open_encoder_on};

/// The ids MediaCodec decodes here.
const VIDEO: [CodecId; 4] = [CodecId::H264, CodecId::Hevc, CodecId::Vp9, CodecId::Av1];

/// Register the MediaCodec decoders, the platform's pick first.
pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `hardware` false keeps to Android's software decoders.
pub fn register_decoders_with(factory: &mut DecodeFactory, hardware: bool) {
    let open = if hardware {
        decoder::open_decoder
    } else {
        open_software
    };
    for id in VIDEO {
        factory.register(id, open);
    }
}

fn open_software(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open_decoder_on(params, false)?))
}

/// The ids MediaCodec encodes here.
const ENCODED: [CodecId; 2] = [CodecId::H264, CodecId::Hevc];

/// Register the MediaCodec encoders, the platform's pick first.
pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `hardware` false keeps to Android's software encoders.
pub fn register_encoders_with(factory: &mut EncodeFactory, hardware: bool) {
    let open = if hardware {
        encoder::open_encoder
    } else {
        open_software_encoder
    };
    for id in ENCODED {
        factory.register(id, open);
    }
}

fn open_software_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(open_encoder_on(params, false)?))
}
