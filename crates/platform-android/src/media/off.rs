//! Decline every opener where MediaCodec is not linked. The factory then
//! uses the Rust codec. The error is [`CodecError::Unsupported`] so this
//! opener does not hide that codec.

use cutlass_decode::{CodecError, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

use super::kind::Kind;

/// The ids MediaCodec decodes on a phone.
const VIDEO: [cutlass_decode::CodecId; 4] = [
    cutlass_decode::CodecId::H264,
    cutlass_decode::CodecId::Hevc,
    cutlass_decode::CodecId::Vp9,
    cutlass_decode::CodecId::Av1,
];

/// Register the MediaCodec decoders. Off Android every opener declines.
pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `hardware` false keeps to Android's software decoders. Off Android
/// every opener declines either way.
pub fn register_decoders_with(factory: &mut DecodeFactory, hardware: bool) {
    let _ = hardware;
    for id in VIDEO {
        factory.register(id, open_decoder);
    }
}

/// Register the MediaCodec encoders. Off Android every opener declines.
pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `hardware` false keeps to Android's software encoders. Off Android
/// every opener declines either way.
pub fn register_encoders_with(factory: &mut EncodeFactory, hardware: bool) {
    let _ = hardware;
    for id in [cutlass_decode::CodecId::H264, cutlass_decode::CodecId::Hevc] {
        factory.register(id, open_encoder);
    }
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Kind::of(params.codec)?;
    if params.width == 0 || params.height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    Err(CodecError::Unsupported { what: "mediacodec" })
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Kind::of(params.codec)?;
    if params.width == 0 || params.height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    Err(CodecError::Unsupported { what: "mediacodec" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{CodecId, PixelFormat};

    #[test]
    fn every_opener_declines_and_an_empty_picture_is_still_an_error() {
        let mut factory = DecodeFactory::new();
        register_decoders(&mut factory);
        for codec in VIDEO {
            let params = CodecParams::with_format(codec, 64, 48, PixelFormat::Yuv420p);
            assert!(matches!(
                factory.open(&params),
                Err(CodecError::Unsupported { .. })
            ));
        }
        let empty = CodecParams::with_format(CodecId::H264, 0, 48, PixelFormat::Yuv420p);
        assert!(matches!(
            open_decoder(&empty),
            Err(CodecError::InvalidApi { .. })
        ));
        assert!(matches!(
            open_encoder(&empty),
            Err(CodecError::InvalidApi { .. })
        ));
        let mut encoders = EncodeFactory::new();
        register_encoders(&mut encoders);
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Rgba8);
        assert!(matches!(
            encoders.open(&params),
            Err(CodecError::Unsupported { .. })
        ));
    }
}
