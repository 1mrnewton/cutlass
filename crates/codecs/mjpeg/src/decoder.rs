//! The decoder behind both decode traits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, Transfer,
};

use crate::config::{Config, FieldOrder};
use crate::picture::{Picture, Unpacker};

/// Decodes Motion JPEG packets. Every frame is a key frame; only tables
/// carry from one packet to the next.
pub struct MjpegDecoder {
    unpacker: Unpacker,
    cancel: Arc<AtomicBool>,
    last_fields: Option<FieldOrder>,
}

impl MjpegDecoder {
    /// A decoder for `params`. `width` and `height` are the container's
    /// frame size, which tells a field pair from a frame; zero for unknown,
    /// in which case every packet is one frame. `extra` is described in
    /// [`crate::config`].
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mjpeg {
            return Err(CodecError::Unsupported {
                what: "codec is not motion jpeg",
            });
        }
        let config = Config::parse(&params.extra)?;
        Ok(Self {
            unpacker: Unpacker::new(config, params.width, params.height)?,
            cancel: Arc::new(AtomicBool::new(false)),
            last_fields: None,
        })
    }

    /// Decodes one packet.
    pub fn decode(&mut self, data: &[u8]) -> Result<Picture, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let picture = self.unpacker.unpack(data)?;
        self.last_fields = picture.fields;
        Ok(picture)
    }

    /// The stored field order of the last decoded frame, `None` when it was
    /// progressive. The woven frame does not carry it.
    #[must_use]
    pub fn last_field_order(&self) -> Option<FieldOrder> {
        self.last_fields
    }

    fn reset_state(&mut self) {
        self.unpacker.reset();
        self.cancel.store(false, Ordering::Relaxed);
        self.last_fields = None;
    }
}

impl VideoDecoder for MjpegDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        let frame = self.decode(&packet.data)?.into_frame()?;
        Ok(vec![VideoOutput {
            frame,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// The app's picture decoder: each frame as straight RGBA.
pub struct MjpegPictureDecoder {
    inner: MjpegDecoder,
}

impl MjpegPictureDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Ok(Self {
            inner: MjpegDecoder::open(params)?,
        })
    }

    /// The decoder underneath, for its field order.
    #[must_use]
    pub fn inner(&self) -> &MjpegDecoder {
        &self.inner
    }
}

impl Decoder for MjpegPictureDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let picture = self.inner.decode(&packet.data)?;
        let mut rgba = Vec::new();
        picture.to_rgba(&mut rgba)?;
        let info = PictureInfo::new(
            picture.width,
            picture.height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            Transfer::Srgb,
        );
        Ok(vec![Output::Picture(PictureOutput::cpu(
            info,
            rgba,
            packet.pts,
            packet.timescale,
        )?)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.inner.reset_state();
    }
}
