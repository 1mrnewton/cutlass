//! Media Foundation transforms.

mod com;
mod video;

#[cfg(test)]
mod tests;

use cutlass_decode::{CodecId, DecodeFactory};
use cutlass_encode::EncodeFactory;

pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false leaves the picture decoders out, so the factory asks the
/// next opener for those ids.
pub fn register_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, video::open_decoder);
        }
    }
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false leaves the picture encoders out, so the factory asks the
/// next opener for those ids.
pub fn register_encoders_with(factory: &mut EncodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, video::open_encoder);
        }
    }
}

const VIDEO: [CodecId; 1] = [CodecId::H264];
