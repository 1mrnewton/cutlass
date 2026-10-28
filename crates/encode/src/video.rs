//! Video encoder trait.
//!
//! `push` accepts one picture. It may buffer that picture and return no
//! packet yet. `drain` flushes a buffered picture. A second `drain` is
//! empty. `reset` drops the buffer and the sample clock. A picture that is
//! not legal fails the call and adds nothing from that push.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_codec::access::{CodecError, CodecParams, Packet, VideoFrame};

use crate::CodecId;

/// One picture presented to an encoder.
///
/// The encoder clocks packets from the pictures it accepts. `pts` on the
/// input is stored on the packet that picture completes. A discontinuity is
/// an [`VideoEncoder::reset`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoInput {
    pub frame: VideoFrame,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
}

pub trait VideoEncoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError>;

    fn reset(&mut self);

    /// A handle another thread can raise. The encoder checks it between
    /// pictures.
    fn cancel_flag(&self) -> Arc<AtomicBool>;
}

pub struct VideoEncodeFactory {
    openers: Vec<Opener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError>,
}

impl VideoEncodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
        }
    }

    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
        for opener in &self.openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "encoder is not registered",
        })
    }
}

impl Default for VideoEncodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Raised by the owner of [`VideoEncoder::cancel_flag`].
pub fn request_video_cancel(flag: &AtomicBool) {
    flag.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{VideoEncodeFactory, VideoEncoder, VideoInput};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cutlass_codec::CodecId;
    use cutlass_codec::access::{CodecError, CodecParams, Packet, PixelFormat, VideoFrame};

    struct Copy;

    impl VideoEncoder for Copy {
        fn codec(&self) -> CodecId {
            CodecId::Vp9
        }

        fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
            if input.frame.format != PixelFormat::Yuv420p {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
            Ok(vec![Packet::video(
                input.frame.y.samples.clone(),
                input.pts,
                input.duration,
                input.timescale,
                true,
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

    fn open_copy(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported { what: "copy" });
        }
        Ok(Box::new(Copy))
    }

    #[test]
    fn a_picture_becomes_one_packet() {
        let mut factory = VideoEncodeFactory::new();
        factory.register(CodecId::Vp9, open_copy);
        let params = CodecParams::video(CodecId::Vp9, 2, 2, PixelFormat::Yuv420p);
        let mut encoder = factory.open(&params).expect("open");
        let frame = VideoFrame::blank_yuv420p(2, 2).expect("frame");
        let packets = encoder
            .push(&VideoInput {
                frame,
                pts: 0,
                duration: 1,
                timescale: 30,
            })
            .expect("push");
        assert_eq!(packets.len(), 1);
        assert!(packets[0].keyframe);
        assert!(encoder.drain().expect("drain").is_empty());
    }
}
