//! Decoder trait for one PNG datastream.

use cutlass_decode::{
    AudioOutput, CodecError, CodecId, CodecParams, Decoder, Packet, PictureOutput,
};

use crate::read::decode_png;

pub struct PngDecoder;

impl PngDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Png {
            return Err(CodecError::Unsupported { what: "png" });
        }
        Ok(Self)
    }
}

impl Decoder for PngDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Png
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<AudioOutput>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "png produces a picture",
        })
    }

    fn drain(&mut self) -> Result<Vec<AudioOutput>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<PictureOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let image = decode_png(&packet.data)?;
        Ok(vec![PictureOutput {
            image,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }
}
