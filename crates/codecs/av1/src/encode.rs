//! Public AV1 encoder over the shared [`Encoder`] trait.
//!
//! The coding itself lives in [`crate::enc`]. This type maps the input pixel
//! format to the AV1 bit depth and subsampling, starts the encoder on the
//! first frame (when the frame rate for a bitrate target is known), and wraps
//! each temporal unit in a [`Packet`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, PixelFormat};
use cutlass_encode::{AudioInput, Encoder, FrameInput};

use crate::enc::encoder::{Av1Enc, EncConfig, EncodedUnit, InputFrame, RateControl};
use crate::yuv::VideoInput;

/// Rust AV1 encoder. `bitrate` of zero keeps the quality index.
pub struct Av1Encoder {
    width: u32,
    height: u32,
    pixel: PixelFormat,
    bitrate: u32,
    qindex: u8,
    speed: u8,
    gop: u32,
    enc: Option<Av1Enc>,
    cancel: Arc<AtomicBool>,
    /// Timestamps of queued pictures, by display index.
    times: std::collections::VecDeque<(i64, i64, u32)>,
}

/// Bit depth, horizontal and vertical subsampling, and monochrome flag of a pixel format.
fn format_of(pixel: PixelFormat) -> Option<(u32, bool, bool)> {
    Some(match pixel {
        PixelFormat::Yuv420p => (8, true, true),
        PixelFormat::Yuv422p => (8, true, false),
        PixelFormat::Yuv444p | PixelFormat::Gbrp => (8, false, false),
        PixelFormat::Yuv420p10 => (10, true, true),
        PixelFormat::Yuv422p10 => (10, true, false),
        PixelFormat::Yuv444p10 | PixelFormat::Gbrp10 => (10, false, false),
        PixelFormat::Yuv420p12 => (12, true, true),
        PixelFormat::Yuv422p12 => (12, true, false),
        PixelFormat::Yuv444p12 | PixelFormat::Gbrp12 => (12, false, false),
        _ => return None,
    })
}

impl Av1Encoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Av1 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if format_of(params.pixel).is_none() {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        if params.width == 0 || params.height == 0 || params.width > 16384 || params.height > 16384
        {
            return Err(CodecError::Limit { what: "frame size" });
        }
        Ok(Self {
            width: params.width,
            height: params.height,
            pixel: params.pixel,
            bitrate: params.bitrate,
            qindex: 100,
            speed: 6,
            gop: 8,
            enc: None,
            cancel: Arc::new(AtomicBool::new(false)),
            times: std::collections::VecDeque::new(),
        })
    }

    /// Quantizer index used when `bitrate` is zero. Zero is raised to one.
    pub fn set_quality(&mut self, qindex: u8) {
        self.qindex = qindex.clamp(1, 255);
        self.enc = None;
    }

    /// Mini-GOP length. 1 gives one packet per picture with no delay; longer
    /// GOPs reorder frames behind a hidden alt-ref (packets come out later,
    /// and `drain` returns the rest).
    pub fn set_gop(&mut self, gop: u32) {
        self.gop = gop.clamp(1, 32);
        self.enc = None;
    }

    /// Speed preset: 0 is slowest and best, 10 fastest.
    pub fn set_speed(&mut self, speed: u8) {
        self.speed = speed.min(10);
        self.enc = None;
    }

    fn encoder(&mut self, duration: i64, timescale: u32) -> Result<&mut Av1Enc, CodecError> {
        if self.enc.is_none() {
            let Some((bd, sx, sy)) = format_of(self.pixel) else {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            };
            let mut cfg = EncConfig::new(self.width as usize, self.height as usize);
            cfg.bit_depth = bd;
            cfg.sub_x = sx;
            cfg.sub_y = sy;
            cfg.speed = self.speed;
            cfg.gop = self.gop;
            if matches!(
                self.pixel,
                PixelFormat::Gbrp | PixelFormat::Gbrp10 | PixelFormat::Gbrp12
            ) {
                cfg.color_primaries = 1;
                cfg.transfer = 13;
                cfg.matrix = 0;
                cfg.full_range = true;
            }
            cfg.rc = if self.bitrate > 0 {
                let fps = if duration > 0 && timescale > 0 {
                    f64::from(timescale) / duration as f64
                } else {
                    30.0
                };
                RateControl::Bitrate {
                    bps: u64::from(self.bitrate),
                    fps,
                }
            } else {
                RateControl::Qindex(self.qindex)
            };
            self.enc = Some(Av1Enc::new(cfg)?);
        }
        self.enc
            .as_mut()
            .ok_or(CodecError::InvalidApi { what: "encoder" })
    }

    /// Encodes tightly packed planes in `self.pixel` layout.
    fn encode_packed(
        &mut self,
        bytes: &[u8],
        pts: i64,
        duration: i64,
        timescale: u32,
    ) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let Some((cw, ch)) = self.pixel.chroma_wh(self.width, self.height) else {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        };
        let (cw, ch) = (cw as usize, ch as usize);
        let wide = format_of(self.pixel).is_some_and(|f| f.0 > 8);
        let need = (w * h + 2 * cw * ch) * if wide { 2 } else { 1 };
        if bytes.len() != need {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        let enc = self.encoder(duration, timescale)?;
        let data = if wide {
            let s: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let (y, rest) = s.split_at(w * h);
            let (u, v) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [y, u, v],
                strides: [w, cw, cw],
            })?
        } else {
            let (y, rest) = bytes.split_at(w * h);
            let (u, v) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [y, u, v],
                strides: [w, cw, cw],
            })?
        };
        self.times.push_back((pts, duration, timescale));
        Ok(self.packets(data))
    }

    /// Wraps temporal units in packets with their pictures' timestamps.
    fn packets(&mut self, units: Vec<EncodedUnit>) -> Vec<Packet> {
        units
            .into_iter()
            .map(|u| {
                let (pts, duration, timescale) = self.times.pop_front().unwrap_or((0, 0, 1));
                let key = u.key || is_key(&u.data);
                Packet::video(u.data, pts, duration, timescale, key)
            })
            .collect()
    }

    /// Encode one 4:2:0 picture given as separate planes.
    pub fn push_picture(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.pixel != PixelFormat::Yuv420p
            || input.picture.y.width() != self.width
            || input.picture.y.height() != self.height
        {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        let mut bytes = Vec::with_capacity(input.picture.y.samples().len() * 3 / 2);
        bytes.extend_from_slice(input.picture.y.samples());
        bytes.extend_from_slice(input.picture.u.samples());
        bytes.extend_from_slice(input.picture.v.samples());
        self.encode_packed(&bytes, input.pts, input.duration, input.timescale)
    }
}

/// True when the temporal unit carries a sequence header (the encoder emits
/// one with every key frame).
fn is_key(data: &[u8]) -> bool {
    // The second OBU follows the 2-byte temporal delimiter.
    data.get(2).is_some_and(|h| (h >> 3) & 15 == 1)
}

impl Encoder for Av1Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "audio" })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        if frame.pixel() != self.pixel
            || frame.width() != self.width
            || frame.height() != self.height
        {
            return Err(CodecError::InvalidApi {
                what: "frame format",
            });
        }
        let time = frame.time();
        self.encode_packed(frame.bytes(), time.pts(), time.duration(), time.timescale())
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        let wide = format_of(self.pixel).is_some_and(|f| f.0 > 8);
        let units = match self.enc.as_mut() {
            None => return Ok(Vec::new()),
            Some(e) if wide => e.flush::<u16>()?,
            Some(e) => e.flush::<u8>()?,
        };
        Ok(self.packets(units))
    }

    fn reset(&mut self) {
        self.enc = None;
        self.times.clear();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
