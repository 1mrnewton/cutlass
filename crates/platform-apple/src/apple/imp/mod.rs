//! VideoToolbox, AudioToolbox, and ImageIO.

mod audio;
mod cf;
mod image;
mod metal;
mod queue;
mod surface;
mod sys;
mod thread;
mod video;

use cutlass_decode::{CodecId, DecodeFactory};
use cutlass_encode::EncodeFactory;

pub use image::{decode_file, encode_file};
pub use metal::import_surface;
pub use queue::{OutputQueue, QueueError, StreamSink};
pub use thread::run_at_interface_priority;
pub use video::{release_buffer, to_cpu, with_pixel_buffer};

pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false leaves the picture decoders out, so the factory asks the
/// next opener for those ids. Audio and still decoders are registered either way.
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

/// As [`register_decoders_with`], but a 4:2:0 picture stays in its
/// IOSurface as a [`cutlass_decode::Surface`], for a renderer that imports
/// it with [`import_surface`].
pub fn register_surface_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, video::open_surface_decoder);
        }
    }
    register_decoders_with(factory, false);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false leaves the picture encoders out, so the factory asks the
/// next opener for those ids. Audio and still encoders are registered either way.
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
