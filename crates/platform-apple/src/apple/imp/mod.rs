//! VideoToolbox, AudioToolbox, and ImageIO.

mod audio;
mod cf;
mod image;
mod sys;
mod video;

use cutlass_decode::{CodecId, DecodeFactory};
use cutlass_encode::EncodeFactory;

pub use image::{decode_file, encode_file};
pub use video::{release_buffer, to_cpu, with_pixel_buffer};

pub fn register_decoders(factory: &mut DecodeFactory) {
    for id in VIDEO {
        factory.register(id, video::open_decoder);
    }
    for id in AUDIO {
        factory.register(id, audio::open_decoder);
    }
    for id in STILL {
        factory.register(id, image::open_decoder);
    }
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    for id in VIDEO {
        factory.register(id, video::open_encoder);
    }
    for id in AUDIO {
        factory.register(id, audio::open_encoder);
    }
    for id in STILL {
        factory.register(id, image::open_encoder);
    }
}

const VIDEO: [CodecId; 9] = [
    CodecId::H264,
    CodecId::Hevc,
    CodecId::Av1,
    CodecId::Vp8,
    CodecId::Vp9,
    CodecId::Mpeg4,
    CodecId::Mpeg2,
    CodecId::Mjpeg,
    CodecId::Prores,
];

const AUDIO: [CodecId; 8] = [
    CodecId::Aac,
    CodecId::Opus,
    CodecId::Mp3,
    CodecId::Flac,
    CodecId::Pcm,
    CodecId::Vorbis,
    CodecId::Alac,
    CodecId::Ac3,
];

const STILL: [CodecId; 4] = [CodecId::Jpeg, CodecId::Png, CodecId::Webp, CodecId::Gif];
