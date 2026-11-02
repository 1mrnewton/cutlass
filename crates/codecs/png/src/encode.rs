//! Encoder trait for one PNG datastream.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, BitDepth, CodecError, CodecId, CodecParams, ColorType, Encoder, FrameInput, Packet,
    PictureInput, PixelFormat, StillImage,
};

use crate::write::encode_png;

pub struct PngEncoder {
    interlace: bool,
    cancel: Arc<AtomicBool>,
}

impl PngEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Png {
            return Err(CodecError::Unsupported { what: "png" });
        }
        let interlace = match params.extra.as_slice() {
            [] | [0] => false,
            [1] => true,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "png interlace",
                });
            }
        };
        Ok(Self {
            interlace,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl Encoder for PngEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Png
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "png encodes a picture",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform picture",
            });
        };
        if frame.pixel() != PixelFormat::Rgba8 {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let samples = frame.bytes().iter().copied().map(u16::from).collect();
        let image = StillImage::from_samples(
            frame.width(),
            frame.height(),
            ColorType::Rgba,
            BitDepth::Eight,
            samples,
        )
        .checked()
        .map_err(|_| CodecError::InvalidApi {
            what: "still image",
        })?;
        let data = encode_png(&image, self.interlace, &self.cancel)?;
        Ok(vec![Packet::video(
            data,
            frame.time().pts(),
            frame.time().duration(),
            frame.time().timescale(),
            true,
        )])
    }

    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let data = encode_png(&input.image, self.interlace, &self.cancel)?;
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
