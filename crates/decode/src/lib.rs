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

#![forbid(unsafe_code)]

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    Channel, ChannelLayout, CodecError, CodecParams, Packet, Pcm, PixelFormat, Plane, SampleFormat,
    SideData, Yuv420,
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

/// One decoded picture. The planes are the samples a display would show,
/// after any output-only synthesis the codec applies. Reference frames stay
/// inside the decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    pub picture: Yuv420,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
}

/// A decoder returns audio or a picture. A codec that only does one of them
/// never produces the other.
#[derive(Clone, Debug, PartialEq)]
pub enum Decoded {
    Audio(AudioOutput),
    Video(VideoOutput),
}

impl Decoded {
    #[must_use]
    pub fn as_audio(&self) -> Option<&AudioOutput> {
        match self {
            Self::Audio(audio) => Some(audio),
            Self::Video(_) => None,
        }
    }

    #[must_use]
    pub fn as_video(&self) -> Option<&VideoOutput> {
        match self {
            Self::Video(video) => Some(video),
            Self::Audio(_) => None,
        }
    }
}

pub trait Decoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<Decoded>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Decoded>, CodecError>;

    fn reset(&mut self);
}

pub struct DecodeFactory {
    openers: Vec<Opener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
}

impl DecodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
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
}

impl Default for DecodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioOutput, DecodeFactory, Decoded, Decoder};
    use cutlass_codec::CodecId;
    use cutlass_codec::access::{
        ChannelLayout, CodecError, CodecParams, Packet, Pcm, SampleFormat,
    };

    struct Silent;

    impl Decoder for Silent {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<Decoded>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.0]).expect("pcm");
            assert_eq!(pcm.format, SampleFormat::F32);
            Ok(vec![Decoded::Audio(AudioOutput {
                pcm,
                pts: packet.pts,
                timescale: packet.timescale,
                skip_leading: 0,
                skip_trailing: 0,
                concealed: false,
            })])
        }

        fn drain(&mut self) -> Result<Vec<Decoded>, CodecError> {
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
        assert!(!out[0].as_audio().expect("audio").concealed);
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
}
