//! Encoder trait and the factory that picks an implementation.
//!
//! `push` accepts any number of PCM frames, including zero. It consumes
//! every finite, in-range sample it is given, buffering less than two
//! frames internally, and returns the access units those samples completed.
//! A sample that is not legal fails the call and adds nothing from that
//! push. `drain` pads and flushes. A second `drain` is empty. `reset`
//! drops the buffer, the overlap, and the sample clock.
//!
//! [`VideoEncoder`] takes pictures and returns access units. `push` may
//! buffer pictures for reorder and return nothing until a picture completes.
//! `drain` emits the buffered pictures. A second `drain` is empty. `reset`
//! drops that buffer and the picture clock. `extradata` is the decoder
//! configuration record for the units `push` emits.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    Channel, ChannelLayout, CodecError, CodecParams, FrameError, Packet, Pcm, SampleFormat,
    SideData, Yuv420p,
};

/// PCM presented to an encoder, contiguous with the samples already accepted.
///
/// The encoder clocks packets from the count of accepted samples. `pts` on
/// the input is not a second clock: a discontinuity is a [`Encoder::reset`].
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    pub pcm: Pcm,
}

pub trait Encoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError>;

    fn reset(&mut self);

    /// A handle another thread can raise. The encoder checks it between
    /// frames, never inside a lock this crate does not take.
    fn cancel_flag(&self) -> Arc<AtomicBool>;
}

/// One picture, contiguous with the pictures already accepted.
///
/// `pts` is the presentation time in `timescale` ticks. `duration` is one
/// picture. A zero timescale is refused by the encoder that cares about a
/// rate. `force_keyframe` asks for a random-access picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoInput {
    pub picture: Yuv420p,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub force_keyframe: bool,
}

pub trait VideoEncoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError>;

    fn reset(&mut self);

    fn cancel_flag(&self) -> Arc<AtomicBool>;

    /// Decoder configuration that matches the access units from [`Self::push`].
    fn extradata(&self) -> &[u8];
}

pub struct EncodeFactory {
    openers: Vec<Opener>,
    video_openers: Vec<VideoOpener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
}

struct VideoOpener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError>,
}

impl EncodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
            video_openers: Vec::new(),
        }
    }

    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    /// Later video registrations do not hide an earlier one for the same id.
    pub fn register_video(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError>,
    ) {
        self.video_openers.push(VideoOpener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        for opener in &self.openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "encoder is not registered",
        })
    }

    pub fn open_video(&self, params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
        for opener in &self.video_openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "video encoder is not registered",
        })
    }
}

impl Default for EncodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Raised by the owner of [`Encoder::cancel_flag`].
pub fn request_cancel(flag: &AtomicBool) {
    flag.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{AudioInput, EncodeFactory, Encoder, VideoEncoder, VideoInput};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cutlass_codec::CodecId;
    use cutlass_codec::access::{ChannelLayout, CodecError, CodecParams, Packet, Pcm, Yuv420p};

    struct Copy;

    impl Encoder for Copy {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            let Some(samples) = input.pcm.as_f32() else {
                return Err(CodecError::InvalidApi {
                    what: "copy encoder wants f32",
                });
            };
            if samples.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![Packet::audio(
                samples
                    .iter()
                    .flat_map(|sample| sample.to_bits().to_le_bytes())
                    .collect(),
                0,
                i64::from(input.pcm.samples),
                input.pcm.rate,
            )])
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    fn open_copy(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported { what: "copy" });
        }
        Ok(Box::new(Copy))
    }

    #[test]
    fn empty_push_emits_nothing() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, open_copy);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let mut encoder = factory.open(&params).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), Vec::new()).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("push");
        assert!(packets.is_empty());
        assert!(encoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn missing_encoder_is_unsupported() {
        let err = match EncodeFactory::new().open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        )) {
            Err(err) => err,
            Ok(_) => panic!("missing"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "encoder is not registered",
            }
        );
    }

    struct Stamp;

    impl VideoEncoder for Stamp {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }

        fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
            Ok(vec![Packet::video(
                vec![input.picture.y().first().copied().unwrap_or(0)],
                input.pts,
                input.duration,
                input.timescale,
                input.force_keyframe,
            )])
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }

        fn extradata(&self) -> &[u8] {
            &[]
        }
    }

    fn open_stamp(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "stamp" });
        }
        Ok(Box::new(Stamp))
    }

    #[test]
    fn a_video_push_returns_an_access_unit() {
        let mut factory = EncodeFactory::new();
        factory.register_video(CodecId::H264, open_stamp);
        let params = CodecParams::video(CodecId::H264, 2, 2);
        let mut encoder = factory.open_video(&params).expect("open");
        let picture = Yuv420p::solid(2, 2, 9, 128, 128).expect("picture");
        let packets = encoder
            .push(&VideoInput {
                picture,
                pts: 0,
                duration: 1,
                timescale: 30,
                force_keyframe: true,
            })
            .expect("push");
        assert_eq!(packets.len(), 1);
        assert!(packets[0].keyframe);
        assert!(encoder.extradata().is_empty());
        assert!(encoder.drain().expect("drain").is_empty());
    }
}
