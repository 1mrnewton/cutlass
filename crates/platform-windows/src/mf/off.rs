//! Decline every opener where Media Foundation is not linked. The factory
//! then uses the Rust codec. The error is [`CodecError::Unsupported`] so
//! this opener does not hide that codec.

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

use crate::limits::even_picture;

const VIDEO: [CodecId; 2] = [CodecId::H264, CodecId::Hevc];

/// AAC, JPEG, and PNG, which are registered whatever the picture switch says.
const OTHER: [CodecId; 3] = [CodecId::Aac, CodecId::Jpeg, CodecId::Png];

/// Register the Media Foundation decoders. Off Windows every opener declines.
pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false leaves the picture decoders out, so the factory asks the
/// next opener for those ids. Audio and still decoders are registered
/// either way.
pub fn register_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, open_video_decoder);
        }
    }
    for id in OTHER {
        factory.register(id, open_other_decoder);
    }
}

/// Register the Media Foundation encoders. Off Windows every opener declines.
pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false leaves the picture encoders out, so the factory asks the
/// next opener for those ids. Audio and still encoders are registered
/// either way.
pub fn register_encoders_with(factory: &mut EncodeFactory, video: bool) {
    if video {
        for id in VIDEO {
            factory.register(id, open_video_encoder);
        }
    }
    for id in OTHER {
        factory.register(id, open_other_encoder);
    }
}

/// The same size check the Windows opener runs, then a decline.
fn video_limits(params: &CodecParams) -> Result<(), CodecError> {
    match even_picture(params.width, params.height) {
        Ok(()) | Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "media foundation",
        }),
        Err(err) => Err(err),
    }
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    video_limits(params)?;
    Err(CodecError::Unsupported {
        what: "media foundation",
    })
}

fn open_video_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    video_limits(params)?;
    Err(CodecError::Unsupported {
        what: "media foundation",
    })
}

fn open_other_decoder(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Err(CodecError::Unsupported { what: "wic" })
}

fn open_other_encoder(_params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Err(CodecError::Unsupported { what: "wic" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::PixelFormat;

    #[test]
    fn every_opener_declines_and_a_bad_size_is_still_an_error() {
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Rgba8);
        assert!(matches!(
            open_video_decoder(&params),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            open_video_encoder(&params),
            Err(CodecError::Unsupported { .. })
        ));
        let empty = CodecParams::with_format(CodecId::H264, 0, 48, PixelFormat::Rgba8);
        assert!(matches!(
            open_video_decoder(&empty),
            Err(CodecError::InvalidApi { .. })
        ));
    }
}
