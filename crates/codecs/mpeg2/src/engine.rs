//! The decoder behind the engine's `Decoder` and `VideoDecoder` traits.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, VideoDecoder, VideoOutput,
};

use crate::decoder::{DEFAULT_MAX_PIXELS, Mpeg2Decoder};

/// Registers the MPEG-2 decoder for playback and for the full-chroma picture path.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mpeg2, open_registered);
    factory.register_video(CodecId::Mpeg2, open_video);
}

impl Mpeg2Decoder {
    /// Opens a decoder for `params`. `extra` may hold the sequence header.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mpeg2 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut decoder = Self::new(DEFAULT_MAX_PIXELS);
        decoder.configure(&params.extra)?;
        Ok(decoder)
    }
}

fn open_registered(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(AccessDecoder {
        inner: Mpeg2Decoder::open(params)?,
    }))
}

fn open_video(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(Mpeg2Decoder::open(params)?))
}

/// The engine's decoder. It returns one packed 4:2:0 picture per frame.
struct AccessDecoder {
    inner: Mpeg2Decoder,
}

impl Decoder for AccessDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg2
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        self.inner
            .push(packet)?
            .into_iter()
            .map(picture_output)
            .collect()
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        self.inner
            .drain()?
            .into_iter()
            .map(picture_output)
            .collect()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

fn picture_output(frame: VideoOutput) -> Result<Output, CodecError> {
    let picture = &frame.picture;
    let mut bytes = Vec::with_capacity(picture.y().len() + picture.u().len() + picture.v().len());
    bytes.extend_from_slice(picture.y());
    bytes.extend_from_slice(picture.u());
    bytes.extend_from_slice(picture.v());
    let mut info = PictureInfo::new(
        picture.width(),
        picture.height(),
        PixelFormat::Yuv420p,
        frame.primaries,
        frame.transfer,
    );
    info.matrix = frame.matrix;
    info.num_units_in_display_tick = frame.timing.num_units;
    info.time_scale = frame.timing.time_scale;
    Ok(Output::Picture(PictureOutput::cpu(
        info,
        bytes,
        frame.pts,
        frame.timescale,
    )?))
}

impl VideoDecoder for Mpeg2Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg2
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        Mpeg2Decoder::push(self, packet)
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        Mpeg2Decoder::drain(self)
    }

    fn reset(&mut self) {
        Mpeg2Decoder::reset(self);
    }
}
