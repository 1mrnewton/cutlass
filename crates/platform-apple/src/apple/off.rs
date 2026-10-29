//! Decline every opener where VideoToolbox, AudioToolbox, and ImageIO are
//! not linked. The factory then uses the Rust codec. The error is
//! [`CodecError::Unsupported`] so this opener does not hide that codec.

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

use crate::limits::{self, audio_ok, even_picture, packet_ok, picture_ok};
use crate::status::{self, open_status, unit_status};

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

/// Register the Apple decoders. Off Apple every opener declines.
pub fn register_decoders(factory: &mut DecodeFactory) {
    for id in VIDEO {
        factory.register(id, open_video_decoder);
    }
    for id in AUDIO {
        factory.register(id, open_audio_decoder);
    }
    for id in STILL {
        factory.register(id, open_still_decoder);
    }
}

/// Register the Apple encoders. Off Apple every opener declines.
pub fn register_encoders(factory: &mut EncodeFactory) {
    for id in VIDEO {
        factory.register(id, open_video_encoder);
    }
    for id in AUDIO {
        factory.register(id, open_audio_encoder);
    }
    for id in STILL {
        factory.register(id, open_still_encoder);
    }
}

fn decline(code: i32, what: &'static str) -> CodecError {
    if let Err(err) = open_status(code, what) {
        return err;
    }
    match unit_status(code, what) {
        Err(err) => err,
        Ok(()) => CodecError::Unsupported { what },
    }
}

fn video_limits(params: &CodecParams) -> Result<(), CodecError> {
    match even_picture(params.width, params.height) {
        Ok(()) => {}
        Err(CodecError::Limit { .. }) => {
            return Err(CodecError::Unsupported {
                what: "picture size",
            });
        }
        Err(err) => return Err(err),
    }
    if params.extra.len() > limits::MAX_PACKET {
        packet_ok(params.extra.len())?;
    }
    Ok(())
}

fn audio_limits(params: &CodecParams) -> Result<(), CodecError> {
    let Some(layout) = params.layout.as_ref() else {
        return Err(CodecError::Unsupported {
            what: "audio layout",
        });
    };
    audio_ok(params.sample_rate, layout.len())
}

fn still_limits(params: &CodecParams) -> Result<(), CodecError> {
    if params.width == 0 && params.height == 0 {
        return Ok(());
    }
    match picture_ok(params.width, params.height) {
        Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "picture edge",
        }),
        other => other,
    }
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    video_limits(params)?;
    Err(decline(status::NO_VIDEO_DECODER, "apple codec"))
}

fn open_video_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    video_limits(params)?;
    Err(decline(status::NO_VIDEO_ENCODER, "apple codec"))
}

fn open_audio_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    audio_limits(params)?;
    Err(decline(status::AUDIO_FORMAT, "apple codec"))
}

fn open_audio_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    audio_limits(params)?;
    Err(decline(status::AUDIO_FORMAT, "apple codec"))
}

fn open_still_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    still_limits(params)?;
    Err(decline(status::NO_VIDEO_DECODER, "imageio"))
}

fn open_still_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    still_limits(params)?;
    Err(decline(status::NO_VIDEO_DECODER, "imageio"))
}
