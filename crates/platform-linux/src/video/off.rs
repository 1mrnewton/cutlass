//! Decline every opener where VA-API, NVDEC, and NVENC are not loaded. The
//! factory then uses the Rust codec. The error is
//! [`CodecError::Unsupported`] so this opener does not hide that codec.

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

use super::limits::video_size;

const DECODE: [CodecId; 3] = [CodecId::H264, CodecId::Hevc, CodecId::Av1];
const ENCODE: [CodecId; 2] = [CodecId::H264, CodecId::Hevc];

/// Register the hardware decoders. Off Linux every opener declines.
pub fn register_decoders(factory: &mut DecodeFactory) {
    register_decoders_with(factory, true);
}

/// `video` false registers nothing, so the factory asks the next opener.
pub fn register_decoders_with(factory: &mut DecodeFactory, video: bool) {
    if video {
        for id in DECODE {
            factory.register(id, open_decoder);
        }
    }
}

/// Register the hardware encoders. Off Linux every opener declines.
pub fn register_encoders(factory: &mut EncodeFactory) {
    register_encoders_with(factory, true);
}

/// `video` false registers nothing, so the factory asks the next opener.
pub fn register_encoders_with(factory: &mut EncodeFactory, video: bool) {
    if video {
        for id in ENCODE {
            factory.register(id, open_encoder);
        }
    }
}

/// What hardware video this machine offers: off Linux, none.
pub fn probe() -> Vec<String> {
    vec![String::from(
        "Hardware video through VA-API, NVDEC and NVENC is Linux only.",
    )]
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    video_size(params.width, params.height)?;
    Err(CodecError::Unsupported {
        what: "linux hardware video",
    })
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    video_size(params.width, params.height)?;
    Err(CodecError::Unsupported {
        what: "linux hardware video",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::PixelFormat;

    #[test]
    fn every_opener_declines_and_an_empty_size_is_still_an_error() {
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Rgba8);
        assert!(matches!(
            open_decoder(&params),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            open_encoder(&params),
            Err(CodecError::Unsupported { .. })
        ));
        let empty = CodecParams::with_format(CodecId::H264, 0, 48, PixelFormat::Rgba8);
        assert!(matches!(
            open_decoder(&empty),
            Err(CodecError::InvalidApi { .. })
        ));
    }

    #[test]
    fn the_rust_decoder_still_opens_behind_the_decline() {
        let mut factory = DecodeFactory::new();
        register_decoders(&mut factory);
        cutlass_h264::register_decoders(&mut factory);
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        assert!(factory.open(&params).is_ok());
    }
}
