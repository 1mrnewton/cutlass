//! Decoder trait and the factory that picks an implementation.
//!
//! A decoder consumes one complete access unit per [`Decoder::push`].
//! Incomplete byte streams are a container concern: the container returns
//! [`CodecError::NeedMoreData`] until a unit is whole, then hands that unit
//! over. A unit that claims to be whole and runs out of bits is
//! [`CodecError::Truncated`].
//!
//! `push` of a unit that fails leaves the decoder as it was before that
//! call. `drain` flushes overlap that a following silent unit would have
//! released. A second `drain` is empty. `reset` drops overlap and the
//! sample clock; the configuration stays. `push` after `drain` continues.
//! `push` of an empty packet is an invalid call, not a silent success.
//!
//! [`VideoDecoder`] is the picture path. It has the same push, drain, and
//! reset rules. One push is one access unit and may return zero pictures
//! when display order is still waiting on a later picture. `drain` releases
//! those pictures. A second `drain` is empty. `reset` drops references and
//! the reorder queue. Configuration from the open parameters stays.

#![forbid(unsafe_code)]

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    Channel, ChannelLayout, CodecError, CodecParams, ColorMatrix, ColorPrimaries, ColorRange,
    FrameError, Packet, Pcm, PixelFormat, SampleFormat, SideData, Transfer, Yuv420p,
};

/// One decoded audio buffer.
///
/// `skip_leading` and `skip_trailing` count samples inside `pcm` that are
/// delay or padding when this buffer is the place those samples landed.
/// Most frames leave both at zero and report delay on the packet side data
/// instead. `concealed` is set only when a caller opted into concealment
/// and this buffer is not a faithful decode. This build does not conceal:
/// a bad unit returns an error and `concealed` stays false.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioOutput {
    pub pcm: Pcm,
    pub pts: i64,
    pub timescale: u32,
    pub skip_leading: u32,
    pub skip_trailing: u32,
    pub concealed: bool,
}

pub trait Decoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<AudioOutput>, CodecError>;

    fn drain(&mut self) -> Result<Vec<AudioOutput>, CodecError>;

    fn reset(&mut self);
}

/// One decoded picture, in display order once it is released.
///
/// `picture` is the cropped coded-domain sample buffer. Color tags describe
/// that buffer. They are not a conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    pub picture: Yuv420p,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
}

pub trait VideoDecoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError>;

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError>;

    fn reset(&mut self);
}

pub struct DecodeFactory {
    openers: Vec<Opener>,
    video_openers: Vec<VideoOpener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
}

struct VideoOpener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
}

impl DecodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
            video_openers: Vec::new(),
        }
    }

    /// Later registrations do not hide an earlier one for the same id.
    /// The desktop registers a platform opener first, then the Rust opener.
    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    /// Later video registrations do not hide an earlier one for the same id.
    pub fn register_video(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
    ) {
        self.video_openers.push(VideoOpener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        for opener in &self.openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "codec is not registered",
        })
    }

    pub fn open_video(&self, params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        for opener in &self.video_openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "video codec is not registered",
        })
    }
}

impl Default for DecodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioOutput, DecodeFactory, Decoder, VideoDecoder, VideoOutput};
    use cutlass_codec::CodecId;
    use cutlass_codec::access::{
        ChannelLayout, CodecError, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, Packet,
        Pcm, SampleFormat, Transfer, Yuv420p,
    };

    struct Silent;

    impl Decoder for Silent {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<AudioOutput>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.0]).expect("pcm");
            assert_eq!(pcm.format, SampleFormat::F32);
            Ok(vec![AudioOutput {
                pcm,
                pts: packet.pts,
                timescale: packet.timescale,
                skip_leading: 0,
                skip_trailing: 0,
                concealed: false,
            }])
        }

        fn drain(&mut self) -> Result<Vec<AudioOutput>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_silent(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported {
                what: "silent test",
            });
        }
        Ok(Box::new(Silent))
    }

    #[test]
    fn factory_uses_the_first_registration() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Pcm, open_silent);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let mut decoder = factory.open(&params).expect("open");
        let packet = Packet::audio(vec![0], 0, 1, 48_000);
        let out = decoder.push(&packet).expect("push");
        assert_eq!(out.len(), 1);
        assert!(!out[0].concealed);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn unregistered_codec_is_unsupported() {
        let factory = DecodeFactory::new();
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::stereo());
        let err = match factory.open(&params) {
            Err(err) => err,
            Ok(_) => panic!("missing"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "codec is not registered",
            }
        );
    }

    struct Solid;

    impl VideoDecoder for Solid {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let picture = Yuv420p::solid(2, 2, 16, 128, 128).expect("picture");
            Ok(vec![VideoOutput {
                picture,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
                keyframe: packet.keyframe,
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
                matrix: ColorMatrix::Bt709,
            }])
        }

        fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_solid(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "solid test" });
        }
        Ok(Box::new(Solid))
    }

    #[test]
    fn video_factory_uses_the_first_registration() {
        let mut factory = DecodeFactory::new();
        factory.register_video(CodecId::H264, open_solid);
        let params = CodecParams::video(CodecId::H264, 2, 2);
        let mut decoder = factory.open_video(&params).expect("open");
        let packet = Packet::video(vec![1], 0, 1, 30, true);
        let out = decoder.push(&packet).expect("push");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].picture.y(), &[16, 16, 16, 16]);
        assert!(decoder.drain().expect("drain").is_empty());
        let audio = factory.open(&CodecParams::audio(
            CodecId::H264,
            48_000,
            ChannelLayout::mono(),
        ));
        assert!(matches!(
            audio,
            Err(CodecError::Unsupported {
                what: "codec is not registered",
            })
        ));
    }
}
