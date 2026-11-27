//! The still decoder: one packet is one image, turned upright.

use cutlass_decode::orient::{Orientation, upright};
use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};

use crate::image::{ColorModel, Tables, decode_image, read_header};

/// Size and coding of one JPEG frame.
///
/// `orientation` is the EXIF tag when APP1 carries one. `width` and
/// `height` are the frame header, as stored; the decoded picture is turned
/// upright.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegInfo {
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub progressive: bool,
    pub orientation: Option<u16>,
}

/// Read the frame size without decoding coefficients.
pub fn read_jpeg_info(bytes: &[u8]) -> Result<JpegInfo, CodecError> {
    let header = read_header(bytes)?;
    Ok(JpegInfo {
        width: header.width,
        height: header.height,
        components: header.components,
        progressive: header.progressive,
        orientation: header.orientation,
    })
}

/// Decode the picture and turn it upright by its EXIF orientation.
pub(crate) fn decode_jpeg(bytes: &[u8]) -> Result<StillImage, CodecError> {
    let image = decode_image(bytes, &mut Tables::new())?.image;
    let (color, channels) = match image.color {
        ColorModel::Gray => (ColorType::Gray, 1),
        ColorModel::Rgb | ColorModel::YCbCr => (ColorType::Rgb, 3),
    };
    let (width, height) = (image.width, image.height);
    let turn = image
        .orientation
        .map(|tag| Orientation::from_tag(u32::from(tag)))
        .filter(|turn| turn.tag() != 1);
    // A still holds two bytes a sample, so the planes are let go and the
    // picture is turned while it is one byte a sample: a 48-megapixel photo
    // is 290 MB once widened, and no more than that and its bytes are held
    // at once.
    let rgb = image.to_rgb()?;
    drop(image);
    let (width, height, rgb) = match turn {
        Some(turn) => upright(turn, width, height, channels, &rgb)?,
        None => (width, height, rgb),
    };
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(rgb.len())
        .map_err(|_| CodecError::Limit { what: "jpeg size" })?;
    samples.extend(rgb.iter().map(|&sample| u16::from(sample)));
    drop(rgb);
    StillImage::from_samples(width, height, color, BitDepth::Eight, samples)
        .checked()
        .map_err(|_| CodecError::Malformed {
            at_bit: 0,
            what: "picture",
        })
}

/// Software JPEG decoder. One push is one complete image.
pub struct JpegDecoder;

impl JpegDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Jpeg {
            return Err(CodecError::Unsupported { what: "jpeg" });
        }
        Ok(Self)
    }
}

impl Decoder for JpegDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "jpeg produces a picture",
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
        let image = decode_jpeg(&packet.data)?;
        Ok(vec![StillOutput {
            image,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }
}
