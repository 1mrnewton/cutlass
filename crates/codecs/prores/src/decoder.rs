//! The decoder behind both decode traits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureInfo, PictureOutput,
    PixelFormat, VideoFrame,
};
use cutlass_pool::Pool;

use crate::color;

use crate::decode::{DecodedFrame, Picture420, Reuse, decode_wide, decode_yuv420};
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
    reuse: Reuse,
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
            reuse: Reuse::default(),
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
        let depth = self.depth_for(data)?;
        decode_wide(data, depth, self.pool(), &self.cancel, &self.reuse)
    }

    /// Hands a frame's planes back, so a later frame of the same size is
    /// decoded into them and nothing is allocated for it.
    pub fn recycle(&mut self, frame: VideoFrame) {
        let VideoFrame { y, u, v, alpha, .. } = frame;
        let alpha = alpha.map(|alpha| alpha.plane);
        for plane in [Some(y), Some(u), Some(v), alpha].into_iter().flatten() {
            self.reuse.keep_plane(plane.samples.into_vec());
        }
    }

    /// Decodes one frame as 8-bit 4:2:0, without alpha.
    pub(crate) fn decode_yuv420(&mut self, data: &[u8]) -> Result<Picture420, CodecError> {
        let depth = self.depth_for(data)?;
        decode_yuv420(data, depth, self.pool(), &self.cancel, &self.reuse)
    }

    fn pool(&self) -> &Pool {
        match &self.workers {
            Workers::Shared => Pool::shared(),
            Workers::Own(pool) => pool,
        }
    }

    fn depth_for(&self, data: &[u8]) -> Result<u8, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        Ok(match (self.depth, self.profile) {
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
        })
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

/// The picture decoder over a wide decoder already set up, with its own
/// threads say.
impl From<ProresDecoder> for ProresPictureDecoder {
    fn from(inner: ProresDecoder) -> Self {
        Self { inner }
    }
}

impl Decoder for ProresPictureDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Prores
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let picture = self.inner.decode_yuv420(&packet.data)?;
        let info = PictureInfo::new(
            picture.width,
            picture.height,
            PixelFormat::Yuv420p,
            color::primaries(picture.header.primaries),
            color::transfer(picture.header.transfer),
        );
        Ok(vec![Output::Picture(PictureOutput::cpu(
            info,
            picture.bytes,
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
