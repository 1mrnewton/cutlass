//! Video decoder trait.
//!
//! One [`VideoDecoder::push`] consumes one access unit. A unit that does not
//! display a picture returns an empty vec. `drain` is empty for codecs that
//! do not reorder. A second `drain` is empty. `reset` drops reference
//! pictures and the sample clock; the configuration stays. A failed `push`
//! leaves the decoder as it was before that call.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cutlass_codec::access::{CodecError, CodecParams, Packet, VideoFrame};

use crate::CodecId;

/// One displayed picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    pub frame: VideoFrame,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
}

pub trait VideoDecoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError>;

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError>;

    fn reset(&mut self);

    /// A handle another thread can raise. The decoder checks it between
    /// frames.
    fn cancel_flag(&self) -> Arc<AtomicBool>;
}

pub struct VideoDecodeFactory {
    openers: Vec<Opener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
}

impl VideoDecodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
        }
    }

    /// Later registrations do not hide an earlier one for the same id.
    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
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

impl Default for VideoDecodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{VideoDecodeFactory, VideoDecoder, VideoOutput};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cutlass_codec::CodecId;
    use cutlass_codec::access::{CodecError, CodecParams, Packet, PixelFormat, VideoFrame};

    struct Blank;

    impl VideoDecoder for Blank {
        fn codec(&self) -> CodecId {
            CodecId::Vp9
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            Ok(vec![VideoOutput {
                frame: VideoFrame::blank_yuv420p(2, 2)?,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
            }])
        }

        fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    fn open_blank(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported { what: "blank test" });
        }
        Ok(Box::new(Blank))
    }

    #[test]
    fn factory_opens_the_first_video_decoder() {
        let mut factory = VideoDecodeFactory::new();
        factory.register(CodecId::Vp9, open_blank);
        let params = CodecParams::with_format(CodecId::Vp9, 2, 2, PixelFormat::Yuv420p);
        let mut decoder = factory.open(&params).expect("open");
        let packet = Packet::video(vec![1], 0, 1, 30, true);
        let out = decoder.push(&packet).expect("push");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frame.width, 2);
        assert!(decoder.drain().expect("drain").is_empty());
    }
}
