//! VP9 profile 0 decoder and encoder.
//!
//! The support record is `STATUS.md` next to this crate. Profiles 1, 2, and 3
//! are rejected. Playback and a WebM container are outside this crate.

#![forbid(unsafe_code)]

mod bits;
mod coef;
mod decode;
mod encode;
mod fwd;
mod geom;
mod header;
mod intra;
mod lpf;
mod prob;
mod recon;
mod tables;
mod txfm;

pub use decode::{Decoder, Vp9Decoder};
pub use encode::Vp9Encoder;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, CodecId, CodecParams, VideoDecodeFactory};
use cutlass_encode::VideoEncodeFactory;
use cutlass_encode::video::VideoEncoder;

/// Registers the profile-0 decoder on a video decode factory.
pub fn register_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Vp9, open_decoder);
}

/// Registers the profile-0 encoder on a video encode factory.
pub fn register_encoders(factory: &mut VideoEncodeFactory) {
    factory.register(CodecId::Vp9, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(Vp9Decoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(Vp9Encoder::open(params)?))
}

#[cfg(test)]
mod tests {
    use super::tables::COEF_PROBS;

    #[test]
    fn default_dc_coef_prob_matches_libvpx() {
        assert_eq!(COEF_PROBS[0][0][0][0][0], [195, 29, 183]);
    }
}
