//! Media Foundation transforms.

mod audio;
mod com;
mod image;
mod transform;
mod video;

#[cfg(test)]
mod audio_tests;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod tests;

use cutlass_decode::{CodecId, DecodeFactory};
use cutlass_encode::EncodeFactory;

pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false leaves the picture decoders out, so the factory asks the
/// next opener for those ids. Audio and still decoders are registered
/// either way.
pub fn register_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, video::open_decoder);
        }
    }
    for id in AUDIO {
        factory.register(id, audio::open_decoder);
    }
    for id in STILL {
        factory.register(id, image::open_decoder);
    }
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false leaves the picture encoders out, so the factory asks the
/// next opener for those ids. Audio and still encoders are registered
/// either way.
pub fn register_encoders_with(factory: &mut EncodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, video::open_encoder);
        }
    }
    for id in AUDIO {
        factory.register(id, audio::open_encoder);
    }
    for id in STILL {
        factory.register(id, image::open_encoder);
    }
}

const VIDEO: [CodecId; 2] = [CodecId::H264, CodecId::Hevc];

const AUDIO: [CodecId; 1] = [CodecId::Aac];

const STILL: [CodecId; 2] = [CodecId::Jpeg, CodecId::Png];
