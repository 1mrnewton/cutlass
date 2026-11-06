//! The decoder behind both decode traits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureInfo, PictureOutput,
    PixelFormat,
};
use cutlass_pool::Pool;

use crate::convert::to_yuv420p;
use crate::decode::{DecodedFrame, decode_frame};
use crate::header::{AlphaBits, Chroma};
use crate::profile::Profile;

/// Where slice jobs run.
enum Workers {
    Shared,
    Own(Pool),
}

/// Decodes ProRes frames. Every frame is a key frame, so nothing is held
/// between packets.
pub struct ProresDecoder {
    profile: Option<Profile>,
    depth: Option<u8>,
    workers: Workers,
    cancel: Arc<AtomicBool>,
}

impl ProresDecoder {
    /// A decoder for `params`. The sample-entry code in `params.extra`, when
    /// there is one, sets the output depth: 12 bits for `ap4h` and `ap4x`,
    /// 10 for the 4:2:2 profiles. Without it a 4:4:4 or alpha frame is 12-bit.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Prores {
            return Err(CodecError::Unsupported {
                what: "codec is not ProRes",
            });
        }
        Ok(Self {
            profile: Profile::from_fourcc(&params.extra),
            depth: None,
            workers: Workers::Shared,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Runs slice jobs on `threads` workers of its own; 1 keeps them on the caller.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.workers = Workers::Own(Pool::new(threads));
        self
    }

    /// Forces the output depth to 10 or 12 bits.
    pub fn with_depth(mut self, depth: u8) -> Result<Self, CodecError> {
        if depth != 10 && depth != 12 {
            return Err(CodecError::InvalidApi {
                what: "output bit depth",
            });
        }
        self.depth = Some(depth);
        Ok(self)
    }

    /// The profile the sample-entry code named, if any.
    #[must_use]
    pub fn profile(&self) -> Option<Profile> {
        self.profile
    }

    /// Decodes one frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedFrame, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let depth = match (self.depth, self.profile) {
            (Some(depth), _) => depth,
            (None, Some(profile)) => profile.depth(),
            (None, None) => {
                let header = crate::header::parse_frame_header(data)?;
                if header.chroma == Chroma::Yuv444 || header.alpha != AlphaBits::None {
                    12
                } else {
                    10
                }
            }
        };
        let pool = match &self.workers {
            Workers::Shared => Pool::shared(),
            Workers::Own(pool) => pool,
        };
        decode_frame(data, depth, pool, &self.cancel)
    }
}

impl VideoDecoder for ProresDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Prores
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        let decoded = self.decode(&packet.data)?;
        Ok(vec![VideoOutput {
            frame: decoded.frame,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// The app's picture decoder: each frame as 8-bit 4:2:0. Alpha is dropped,
/// since the app's picture has none.
pub struct ProresPictureDecoder {
    inner: ProresDecoder,
}

impl ProresPictureDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Ok(Self {
            inner: ProresDecoder::open(params)?,
        })
    }
}

impl Decoder for ProresPictureDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Prores
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let decoded = self.inner.decode(&packet.data)?;
        let (width, height, bytes) = to_yuv420p(&decoded.frame)?;
        let info = PictureInfo::new(
            width,
            height,
            PixelFormat::Yuv420p,
            decoded.frame.primaries,
            decoded.frame.transfer,
        );
        Ok(vec![Output::Picture(PictureOutput::cpu(
            info,
            bytes,
            packet.pts,
            packet.timescale,
        )?)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}
