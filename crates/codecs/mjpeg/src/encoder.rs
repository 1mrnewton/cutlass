//! The encoder behind both encode traits.
//!
//! Every frame is one baseline JPEG image with its own DQT and DHT, so any
//! packet decodes alone. Fields are not written.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorRange, MatrixCoefficients, PixelFormat, VideoFrame,
};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet};
use cutlass_jpeg::{
    Coefficients, MAX_EDGE, MAX_PIXELS, PlaneRef, Planes, Sampling, Source, WriteSettings,
};

/// Quality when neither a quality nor a bitrate is named.
pub const DEFAULT_QUALITY: u8 = 75;

/// How the quantizer is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMode {
    /// One quality, 1..=100 on the Independent JPEG Group scale.
    Quality(u8),
    /// Bits per second. Each frame takes the highest quality whose image
    /// fits its share of the rate, plus what earlier frames left unused.
    Bitrate(u64),
}

/// Encodes frames to Motion JPEG (`jpeg` sample entry).
pub struct MjpegEncoder {
    width: u32,
    height: u32,
    sampling: Sampling,
    mode: RateMode,
    restart_interval: u16,
    planes: Planes,
    coefficients: Coefficients,
    /// Quality of the last frame, where the next search starts.
    last_quality: Option<u8>,
    /// Bits earlier frames left under their budget, negative when over.
    credit: i64,
    cancel: Arc<AtomicBool>,
}

impl MjpegEncoder {
    /// An encoder for `params`.
    ///
    /// `extra` empty, or one byte in 1..=100 for a fixed quality. Without a
    /// quality, a nonzero `bitrate` selects [`RateMode::Bitrate`], else the
    /// quality is [`DEFAULT_QUALITY`]. RGBA input is coded 4:2:0, grey
    /// `pixel` as one component; planar input keeps its own sampling.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mjpeg {
            return Err(CodecError::Unsupported {
                what: "codec is not motion jpeg",
            });
        }
        let mode = match params.extra.as_slice() {
            [] if params.bitrate > 0 => RateMode::Bitrate(u64::from(params.bitrate)),
            [] => RateMode::Quality(DEFAULT_QUALITY),
            [quality] if (1..=100).contains(quality) => RateMode::Quality(*quality),
            _ => {
                return Err(CodecError::Unsupported {
                    what: "motion jpeg quality",
                });
            }
        };
        let sampling = match params.pixel {
            PixelFormat::Gray8 => Sampling::Gray,
            PixelFormat::Yuv422p => Sampling::Yuv422,
            PixelFormat::Yuv444p => Sampling::Yuv444,
            PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Yuv420p => Sampling::Yuv420,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        if !size_ok(params.width, params.height, true) {
            return Err(CodecError::Limit {
                what: "frame dimensions",
            });
        }
        Ok(Self {
            width: params.width,
            height: params.height,
            sampling,
            mode,
            restart_interval: 0,
            planes: Planes::default(),
            coefficients: Coefficients::new(),
            last_quality: None,
            credit: 0,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Chroma sampling for RGBA input.
    #[must_use]
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_mode(mut self, mode: RateMode) -> Result<Self, CodecError> {
        match mode {
            RateMode::Quality(quality) if !(1..=100).contains(&quality) => {
                Err(CodecError::InvalidApi {
                    what: "motion jpeg quality",
                })
            }
            RateMode::Bitrate(0) => Err(CodecError::InvalidApi {
                what: "motion jpeg bitrate",
            }),
            _ => {
                self.mode = mode;
                Ok(self)
            }
        }
    }

    /// MCUs between restart markers; zero writes none.
    #[must_use]
    pub fn with_restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    #[must_use]
    pub fn mode(&self) -> RateMode {
        self.mode
    }

    /// The quality of the last frame written.
    #[must_use]
    pub fn last_quality(&self) -> Option<u8> {
        self.last_quality
    }

    /// The sample-entry type a movie muxer writes for this track.
    #[must_use]
    pub fn extradata(&self) -> &[u8] {
        b"jpeg"
    }

    /// Encodes straight RGBA; alpha is dropped. `duration` over `timescale`
    /// sizes the frame's share of a bitrate.
    pub fn encode_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
        duration: i64,
        timescale: u32,
    ) -> Result<Vec<u8>, CodecError> {
        self.check(width, height)?;
        let pixels = usize::try_from(u64::from(width) * u64::from(height)).map_err(|_| {
            CodecError::Limit {
                what: "frame dimensions",
            }
        })?;
        if rgba.len() != pixels * 4 {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        self.planes
            .from_rgb(rgba, 4, width, height, self.sampling)?;
        let source = self.planes.source()?;
        self.coefficients.transform(&source)?;
        self.write(duration, timescale)
    }

    /// Encodes full-range BT.601 planar samples: `Gray8`, `Yuv420p`,
    /// `Yuv422p`, or `Yuv444p`. Studio range and other matrices are refused,
    /// since JFIF has no slot for them.
    pub fn encode_frame(
        &mut self,
        frame: &VideoFrame,
        duration: i64,
        timescale: u32,
    ) -> Result<Vec<u8>, CodecError> {
        self.check(frame.width, frame.height)?;
        if matches!(frame.range, ColorRange::Limited | ColorRange::Studio) {
            return Err(CodecError::Unsupported {
                what: "studio range input",
            });
        }
        if !matches!(
            frame.matrix,
            MatrixCoefficients::Unspecified | MatrixCoefficients::Bt601
        ) {
            return Err(CodecError::Unsupported {
                what: "color matrix",
            });
        }
        let sampling = match frame.format {
            PixelFormat::Gray8 => Sampling::Gray,
            PixelFormat::Yuv420p => Sampling::Yuv420,
            PixelFormat::Yuv422p => Sampling::Yuv422,
            PixelFormat::Yuv444p => Sampling::Yuv444,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        let (cw, ch) = sampling.chroma_size(frame.width, frame.height);
        let color = sampling != Sampling::Gray;
        if frame.y.width != frame.width
            || frame.y.height != frame.height
            || (color
                && (frame.u.width, frame.u.height, frame.v.width, frame.v.height)
                    != (cw, ch, cw, ch))
        {
            return Err(CodecError::InvalidApi { what: "plane size" });
        }
        let source = Source {
            width: frame.width,
            height: frame.height,
            sampling,
            y: PlaneRef {
                samples: &frame.y.samples,
                stride: frame.y.stride,
            },
            cb: PlaneRef {
                samples: &frame.u.samples,
                stride: frame.u.stride,
            },
            cr: PlaneRef {
                samples: &frame.v.samples,
                stride: frame.v.stride,
            },
        };
        self.coefficients.transform(&source)?;
        self.write(duration, timescale)
    }

    fn check(&self, width: u32, height: u32) -> Result<(), CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "empty picture",
            });
        }
        if !size_ok(width, height, false) {
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

    /// Choose the quality and write the transformed frame.
    fn write(&mut self, duration: i64, timescale: u32) -> Result<Vec<u8>, CodecError> {
        let (quality, budget) = match self.mode {
            RateMode::Quality(quality) => (quality, None),
            RateMode::Bitrate(bitrate) => {
                let budget = frame_budget(bitrate, duration, timescale)?;
                let allowed = budget.saturating_add(self.credit).max(budget / 2);
                (self.search(allowed)?, Some(budget))
            }
        };
        let settings = self.settings(quality);
        let mut out = Vec::new();
        self.coefficients.write(&settings, &mut out)?;
        self.last_quality = Some(quality);
        if let Some(budget) = budget {
            let used = i64::try_from(out.len())
                .unwrap_or(i64::MAX)
                .saturating_mul(8);
            self.credit = self
                .credit
                .saturating_add(budget)
                .saturating_sub(used)
                .clamp(-budget, budget);
        }
        Ok(out)
    }

    fn settings(&self, quality: u8) -> WriteSettings {
        WriteSettings {
            quality,
            restart_interval: self.restart_interval,
            jfif: true,
        }
    }

    /// Estimated image bits at `quality`: headers plus the counted scan.
    fn bits_at(&self, quality: u8) -> Result<i64, CodecError> {
        let header = self.coefficients.header_len(&self.settings(quality))?;
        let scan = self.coefficients.scan_bits(quality)?;
        let total = u64::try_from(header)
            .unwrap_or(u64::MAX)
            .saturating_mul(8)
            .saturating_add(scan);
        Ok(i64::try_from(total).unwrap_or(i64::MAX))
    }

    /// The highest quality whose estimate fits `allowed` bits, or 1. Starts
    /// at the last frame's quality and widens its step until it brackets.
    fn search(&self, allowed: i64) -> Result<u8, CodecError> {
        let start = self.last_quality.unwrap_or(DEFAULT_QUALITY);
        let fits = |quality: u8| self.bits_at(quality).map(|bits| bits <= allowed);
        let (mut low, mut high);
        if fits(start)? {
            low = start;
            high = 101u8;
            let mut step = 1u8;
            loop {
                let probe = low.saturating_add(step);
                if probe > 100 {
                    break;
                }
                if fits(probe)? {
                    low = probe;
                    step = step.saturating_mul(2);
                } else {
                    high = probe;
                    break;
                }
            }
        } else {
            high = start;
            low = 0;
            let mut step = 1u8;
            loop {
                let probe = high.saturating_sub(step);
                if probe == 0 {
                    break;
                }
                if fits(probe)? {
                    low = probe;
                    break;
                }
                high = probe;
                step = step.saturating_mul(2);
            }
        }
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if fits(middle)? {
                low = middle;
            } else {
                high = middle;
            }
        }
        Ok(low.max(1))
    }

    fn packet(&self, bytes: Vec<u8>, pts: i64, duration: i64, timescale: u32) -> Packet {
        let _ = self;
        Packet::video(bytes, pts, duration, timescale, true)
    }

    fn reset_state(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
        self.last_quality = None;
        self.credit = 0;
    }
}

/// Bits a frame of `duration` / `timescale` seconds may use at `bitrate`.
fn frame_budget(bitrate: u64, duration: i64, timescale: u32) -> Result<i64, CodecError> {
    if duration <= 0 || timescale == 0 {
        return Err(CodecError::InvalidApi {
            what: "frame duration",
        });
    }
    let bits = u128::from(bitrate) * u128::from(duration.unsigned_abs()) / u128::from(timescale);
    Ok(i64::try_from(bits).unwrap_or(i64::MAX).max(1))
}

/// Zero edges are allowed when `unknown_ok`, for parameters that leave
/// the size to the first frame.
fn size_ok(width: u32, height: u32, unknown_ok: bool) -> bool {
    if width == 0 || height == 0 {
        return unknown_ok && width == 0 && height == 0;
    }
    width <= MAX_EDGE && height <= MAX_EDGE && u64::from(width) * u64::from(height) <= MAX_PIXELS
}

impl VideoEncoder for MjpegEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let bytes = self.encode_frame(&input.frame, input.duration, input.timescale)?;
        Ok(vec![self.packet(
            bytes,
            input.pts,
            input.duration,
            input.timescale,
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl Encoder for MjpegEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "motion jpeg codes pictures",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frames",
            });
        };
        if frame.pixel() != PixelFormat::Rgba8 {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let time = frame.time();
        let bytes = self.encode_rgba(
            frame.width(),
            frame.height(),
            frame.bytes(),
            time.duration(),
            time.timescale(),
        )?;
        Ok(vec![self.packet(
            bytes,
            time.pts(),
            time.duration(),
            time.timescale(),
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
