//! Decoder trait for WebP packets.

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};

use crate::limits::picture_ok;
use crate::picture::{self, Picture};
use crate::riff::{self, Still};

/// Software WebP decoder: lossless, lossy, alpha, and animation.
///
/// One packet is what the `still` container hands over: the file for a
/// still, or one `ANMF` frame of an animation (the first packet also holds
/// `VP8X` and `ANIM`). A still decodes to one picture. An animation frame
/// decodes to the whole canvas after that frame is composed.
pub struct WebpDecoder {
    _private: (),
}

impl WebpDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Webp {
            return Err(CodecError::Unsupported { what: "webp" });
        }
        Ok(Self { _private: () })
    }
}

impl Decoder for WebpDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Webp
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "webp produces a picture",
        })
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<StillOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let still = riff::parse(&packet.data)?;
        let image = decode_still(&still)?;
        Ok(vec![StillOutput {
            image,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }
}

fn decode_still(still: &Still<'_>) -> Result<StillImage, CodecError> {
    let decoded = picture::decode(still.bitstream)?;
    if let Some((width, height)) = still.canvas {
        picture_ok(width, height)?;
        if (width, height) != (decoded.width, decoded.height) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "webp canvas does not match the bitstream",
            });
        }
    }
    let mut image = still_image(&decoded);
    if let Some(icc) = still.icc {
        image.icc_profile = icc.to_vec();
    }
    Ok(image)
}

/// RGBA, or RGB when the picture carries no alpha, as 8-bit samples.
pub(crate) fn still_image(picture: &Picture) -> StillImage {
    let (color, samples) = if picture.alpha {
        (
            ColorType::Rgba,
            picture.rgba.iter().map(|&byte| u16::from(byte)).collect(),
        )
    } else {
        (
            ColorType::Rgb,
            picture
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .map(u16::from)
                .collect(),
        )
    };
    StillImage::from_samples(
        picture.width,
        picture.height,
        color,
        BitDepth::Eight,
        samples,
    )
}
