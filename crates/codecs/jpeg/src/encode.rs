//! The still encoder.
//!
//! Color is 4:2:0 and grey is 4:4:4. Quality 1..=100 scales the Annex K
//! quantizers the way the Independent JPEG Group does. `extra` empty is
//! quality 75. A single byte is that quality. Bitrate is not a JPEG mode.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, BitDepth, CodecError, CodecId, CodecParams, ColorType, Encoder, FrameInput, Packet,
    PictureInput, StillImage,
};

use crate::compress::{Coefficients, Planes, Sampling, WriteSettings};
use crate::limits::dimensions_ok;

pub struct JpegEncoder {
    quality: u8,
    cancel: Arc<AtomicBool>,
}

impl JpegEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Jpeg {
            return Err(CodecError::Unsupported { what: "jpeg" });
        }
        let quality = match params.extra.as_slice() {
            [] => 75,
            [quality] if (1..=100).contains(quality) => *quality,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "jpeg quality",
                });
            }
        };
        Ok(Self {
            quality,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl Encoder for JpegEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "jpeg encodes a picture",
        })
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "frame" })
    }

    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let data = encode_jpeg(&input.image, self.quality, &self.cancel)?;
        Ok(vec![Packet {
            stream: 0,
            data,
            pts: 0,
            duration: 0,
            timescale: 0,
            keyframe: true,
            side: Vec::new(),
        }])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

pub(crate) fn encode_jpeg(
    image: &StillImage,
    quality: u8,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, CodecError> {
    image.validate().map_err(|_| CodecError::InvalidApi {
        what: "jpeg picture",
    })?;
    if !dimensions_ok(image.width, image.height) {
        return Err(CodecError::Limit { what: "jpeg size" });
    }
    let color = match (image.color, image.depth) {
        (ColorType::Gray, BitDepth::Eight) => false,
        (ColorType::Rgb, BitDepth::Eight) => true,
        (ColorType::Rgba | ColorType::GrayAlpha, _) => {
            return Err(CodecError::Unsupported { what: "jpeg alpha" });
        }
        (ColorType::Indexed, _) => {
            return Err(CodecError::Unsupported {
                what: "jpeg indexed",
            });
        }
        (_, _) => {
            return Err(CodecError::Unsupported {
                what: "jpeg precision",
            });
        }
    };
    let bytes: Vec<u8> = image
        .samples
        .iter()
        .map(|sample| u8::try_from(*sample).unwrap_or(u8::MAX))
        .collect();
    let mut planes = Planes::default();
    if color {
        planes.from_rgb(&bytes, 3, image.width, image.height, Sampling::Yuv420)?;
    } else {
        planes.width = image.width;
        planes.height = image.height;
        planes.sampling = Some(Sampling::Gray);
        planes.y = bytes;
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(CodecError::Cancelled);
    }
    let mut coefficients = Coefficients::new();
    coefficients.transform(&planes.source()?)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(CodecError::Cancelled);
    }
    let mut file = Vec::new();
    coefficients.write(
        &WriteSettings {
            quality,
            restart_interval: 0,
            jfif: true,
        },
        &mut file,
    )?;
    Ok(file)
}
