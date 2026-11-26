//! The encoder: Proxy, LT, 422, HQ, 4444 and 4444 XQ, progressive.

mod frame;
mod rate;
mod slice;
mod source;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, PixelFormat, VideoFrame};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet};
use cutlass_pool::Pool;

pub use rate::{target_frame_bits, target_mbps};

use self::frame::{ColorCodes, Kept, Settings, encode_frame};
use self::source::Picture;
use crate::color;
use crate::header::{Chroma, MAX_EDGE, MAX_PIXELS};
use crate::profile::Profile;

enum Workers {
    Shared,
    Own(Pool),
}

/// Encodes frames to one ProRes profile. Every frame is coded alone.
///
/// The bit budget is Apple's target data rate for the profile and frame
/// size (ProRes white paper, 2022) over 29.97 frames, so a frame costs the
/// same at any frame rate. `CodecParams::bitrate` is not used. Alpha rides
/// the 4444 profiles only and does not count against the budget.
pub struct ProresEncoder {
    profile: Profile,
    fourcc: [u8; 4],
    width: u32,
    height: u32,
    color: ColorCodes,
    budget_scale: f64,
    workers: Workers,
    cancel: Arc<AtomicBool>,
    last_index: Option<u8>,
    kept: Kept,
}

impl ProresEncoder {
    /// An encoder for `params`. The profile is the sample-entry code in
    /// `params.extra`, `apcn` (422) when there is none.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Prores {
            return Err(CodecError::Unsupported {
                what: "codec is not ProRes",
            });
        }
        let profile = if params.extra.is_empty() {
            Profile::Standard
        } else {
            Profile::from_fourcc(&params.extra).ok_or(CodecError::Unsupported {
                what: "ProRes profile",
            })?
        };
        if params.width > MAX_EDGE
            || params.height > MAX_EDGE
            || u64::from(params.width) * u64::from(params.height) > MAX_PIXELS
        {
            return Err(CodecError::Limit {
                what: "frame dimensions",
            });
        }
        Ok(Self {
            profile,
            fourcc: profile.fourcc(),
            width: params.width,
            height: params.height,
            color: ColorCodes {
                primaries: color::primaries_code(params.primaries),
                transfer: color::transfer_code(params.transfer),
                matrix: color::matrix_code(params.matrix),
            },
            budget_scale: 1.0,
            workers: Workers::Shared,
            cancel: Arc::new(AtomicBool::new(false)),
            last_index: None,
            kept: Kept::default(),
        })
    }

    /// Runs slice jobs on `threads` workers of its own; 1 keeps them on the caller.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.workers = Workers::Own(Pool::new(threads));
        self
    }

    /// Scales the frame budget, 0.05 to 4; 1 is Apple's target.
    pub fn with_budget_scale(mut self, scale: f64) -> Result<Self, CodecError> {
        if !(0.05..=4.0).contains(&scale) {
            return Err(CodecError::InvalidApi {
                what: "budget scale",
            });
        }
        self.budget_scale = scale;
        Ok(self)
    }

    #[must_use]
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// The profile's sample-entry code. A movie muxer takes these four
    /// bytes as a ProRes track's extradata and writes them as the entry.
    #[must_use]
    pub fn extradata(&self) -> &[u8] {
        &self.fourcc
    }

    /// The quantization index most slices of the last frame used.
    #[must_use]
    pub fn last_quantizer(&self) -> Option<u8> {
        self.last_index
    }

    /// Encodes a planar frame: 8-, 10- or 12-bit 4:2:0, 4:2:2 or 4:4:4.
    /// The frame's color tags are written when it names them, else the
    /// parameters'. A 4444 profile keeps the frame's alpha. A plane that
    /// does not hold the samples its size and stride say is refused.
    pub fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<u8>, CodecError> {
        self.check_size(frame.width, frame.height)?;
        let picture = Picture::planar(frame, target_chroma(self.profile))?;
        let mut color = self.color;
        if frame.primaries != cutlass_decode::ColorPrimaries::Unspecified {
            color.primaries = color::primaries_code(frame.primaries);
        }
        if frame.transfer != cutlass_decode::Transfer::Unspecified {
            color.transfer = color::transfer_code(frame.transfer);
        }
        if frame.matrix != cutlass_decode::MatrixCoefficients::Unspecified {
            color.matrix = color::matrix_code(frame.matrix);
        }
        self.run(&picture, color)
    }

    /// Encodes 8-bit RGBA as BT.709 limited range. A 4444 profile keeps alpha.
    pub fn encode_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<Vec<u8>, CodecError> {
        self.check_size(width, height)?;
        let pixels = width as usize * height as usize;
        if rgba.len() != pixels * 4 {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        let picture = Picture::rgba(width, height, rgba, target_chroma(self.profile))?;
        let color = ColorCodes {
            primaries: 1,
            transfer: 1,
            matrix: 1,
        };
        self.run(&picture, color)
    }

    /// Encodes tight 8-bit 4:2:0: Y, then U, then V, with the parameters'
    /// color tags.
    fn encode_yuv420(
        &mut self,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<Vec<u8>, CodecError> {
        self.check_size(width, height)?;
        let picture = Picture::yuv420(width, height, bytes, target_chroma(self.profile))?;
        self.run(&picture, self.color)
    }

    fn check_size(&self, width: u32, height: u32) -> Result<(), CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "empty picture",
            });
        }
        if width > MAX_EDGE
            || height > MAX_EDGE
            || u64::from(width) * u64::from(height) > MAX_PIXELS
        {
            return Err(CodecError::Limit {
                what: "frame dimensions",
            });
        }
        if (self.width != 0 && self.width != width) || (self.height != 0 && self.height != height) {
            return Err(CodecError::InvalidApi {
                what: "frame size differs from the parameters",
            });
        }
        Ok(())
    }

    fn run(&mut self, picture: &Picture<'_>, color: ColorCodes) -> Result<Vec<u8>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let pool = match &self.workers {
            Workers::Shared => Pool::shared(),
            Workers::Own(pool) => pool,
        };
        let settings = Settings {
            profile: self.profile,
            color,
            budget_scale: self.budget_scale,
            hint: self.last_index,
        };
        let encoded = encode_frame(picture, settings, pool, &self.cancel, &self.kept)?;
        self.last_index = Some(encoded.index);
        Ok(encoded.bytes)
    }
}

fn target_chroma(profile: Profile) -> Chroma {
    if profile.is_444() {
        Chroma::Yuv444
    } else {
        Chroma::Yuv422
    }
}

impl VideoEncoder for ProresEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Prores
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let bytes = self.encode(&input.frame)?;
        Ok(vec![Packet::video(
            bytes,
            input.pts,
            input.duration,
            input.timescale,
            true,
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
        self.last_index = None;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl Encoder for ProresEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Prores
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "ProRes codes pictures",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frames",
            });
        };
        let bytes = match frame.pixel() {
            PixelFormat::Rgba8 => self.encode_rgba(frame.width(), frame.height(), frame.bytes())?,
            PixelFormat::Yuv420p => {
                self.encode_yuv420(frame.width(), frame.height(), frame.bytes())?
            }
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        let time = frame.time();
        Ok(vec![Packet::video(
            bytes,
            time.pts(),
            time.duration(),
            time.timescale(),
            true,
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        VideoEncoder::reset(self);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
