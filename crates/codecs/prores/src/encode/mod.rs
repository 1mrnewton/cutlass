//! The encoder: Proxy, LT, 422, HQ, 4444 and 4444 XQ, progressive.

mod frame;
mod rate;
mod slice;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, PixelFormat, Plane, VideoFrame};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet};
use cutlass_pool::Pool;

pub use rate::{target_frame_bits, target_mbps};

use self::frame::{ColorCodes, encode_frame};
use self::slice::{AlphaPlane, Planes};
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
    /// parameters'. A 4444 profile keeps the frame's alpha.
    pub fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<u8>, CodecError> {
        self.check_size(frame.width, frame.height)?;
        let planes = planes_from_frame(frame, self.profile)?;
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
        self.run(&planes, color)
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
        let planes = planes_from_rgba(width, height, rgba, self.profile);
        let color = ColorCodes {
            primaries: 1,
            transfer: 1,
            matrix: 1,
        };
        self.run(&planes, color)
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

    fn run(&mut self, planes: &Planes, color: ColorCodes) -> Result<Vec<u8>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let pool = match &self.workers {
            Workers::Shared => Pool::shared(),
            Workers::Own(pool) => pool,
        };
        let encoded = encode_frame(
            planes,
            self.profile,
            color,
            self.budget_scale,
            self.last_index,
            pool,
            &self.cancel,
        )?;
        self.last_index = Some(encoded.index);
        Ok(encoded.bytes)
    }
}

/// Reads one sample of a tight or strided plane, 1 or 2 bytes little-endian.
fn sample(plane: &Plane, wide: bool, x: usize, y: usize) -> u32 {
    if wide {
        let at = y * plane.stride + x * 2;
        plane
            .samples
            .get(at..at + 2)
            .map_or(0, |pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
    } else {
        plane
            .samples
            .get(y * plane.stride + x)
            .map_or(0, |&byte| u32::from(byte))
    }
}

fn target_chroma(profile: Profile) -> Chroma {
    if profile.is_444() {
        Chroma::Yuv444
    } else {
        Chroma::Yuv422
    }
}

fn padded(width: u32, height: u32, chroma: Chroma) -> (usize, usize, usize) {
    let luma = width.div_ceil(16) as usize * 16;
    let rows = height.div_ceil(16) as usize * 16;
    let chroma_stride = match chroma {
        Chroma::Yuv422 => luma / 2,
        Chroma::Yuv444 => luma,
    };
    (luma, chroma_stride, rows)
}

fn planes_from_frame(frame: &VideoFrame, profile: Profile) -> Result<Planes, CodecError> {
    let (depth, sx, sy): (u32, usize, usize) = match frame.format {
        PixelFormat::Yuv420p => (8, 2, 2),
        PixelFormat::Yuv422p => (8, 2, 1),
        PixelFormat::Yuv444p => (8, 1, 1),
        PixelFormat::Yuv420p10 => (10, 2, 2),
        PixelFormat::Yuv420p12 => (12, 2, 2),
        PixelFormat::Yuv422p10 => (10, 2, 1),
        PixelFormat::Yuv422p12 => (12, 2, 1),
        PixelFormat::Yuv444p10 => (10, 1, 1),
        PixelFormat::Yuv444p12 => (12, 1, 1),
        _ => {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
    };
    let wide = depth > 8;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let chroma = target_chroma(profile);
    let (luma_stride, chroma_stride, rows) = padded(frame.width, frame.height, chroma);
    let mid = (1u32 << (depth - 1)) as f32;
    let unit = 2f32.powi(depth as i32 - 9);
    let norm = move |value: u32| (value as f32 - mid) / unit;
    let mut y = vec![0f32; luma_stride * rows];
    for r in 0..rows {
        let sr = r.min(height - 1);
        for c in 0..luma_stride {
            y[r * luma_stride + c] = norm(sample(&frame.y, wide, c.min(width - 1), sr));
        }
    }
    let source_w = frame.u.width as usize;
    let source_h = frame.u.height as usize;
    let mut u = vec![0f32; chroma_stride * rows];
    let mut v = vec![0f32; chroma_stride * rows];
    let target_w = match chroma {
        Chroma::Yuv422 => width.div_ceil(2),
        Chroma::Yuv444 => width,
    };
    for (plane, out) in [(&frame.u, &mut u), (&frame.v, &mut v)] {
        for r in 0..rows {
            let fr = r.min(height - 1);
            let sr = (fr / sy).min(source_h - 1);
            for c in 0..chroma_stride {
                let tc = c.min(target_w - 1);
                let value = match (chroma, sx) {
                    // 4:4:4 source into 4:2:2: average the pair.
                    (Chroma::Yuv422, 1) => {
                        let a = sample(plane, wide, (2 * tc).min(source_w - 1), sr);
                        let b = sample(plane, wide, (2 * tc + 1).min(source_w - 1), sr);
                        (norm(a) + norm(b)) * 0.5
                    }
                    (Chroma::Yuv422, _) => norm(sample(plane, wide, tc.min(source_w - 1), sr)),
                    (Chroma::Yuv444, _) => {
                        norm(sample(plane, wide, (tc / sx).min(source_w - 1), sr))
                    }
                };
                out[r * chroma_stride + c] = value;
            }
        }
    }
    let alpha = match (&frame.alpha, chroma) {
        (Some(alpha), Chroma::Yuv444) => {
            let bits = u32::from(alpha.bit_depth);
            let sixteen = bits > 8;
            let max_in = (1u32 << bits) - 1;
            let max_out = if sixteen { 65_535 } else { 255 };
            let mut samples = vec![0u16; luma_stride * height];
            for r in 0..height {
                for c in 0..luma_stride {
                    let raw = sample(&alpha.plane, sixteen, c.min(width - 1), r);
                    let value = if max_in == max_out {
                        raw
                    } else {
                        (raw * max_out + max_in / 2) / max_in
                    };
                    samples[r * luma_stride + c] = value as u16;
                }
            }
            Some(AlphaPlane {
                sixteen,
                stride: luma_stride,
                samples,
            })
        }
        _ => None,
    };
    Ok(Planes {
        width: frame.width,
        height: frame.height,
        chroma,
        luma_stride,
        chroma_stride,
        y,
        u,
        v,
        alpha,
    })
}

/// BT.709 limited-range Y'CbCr of one RGBA pixel, in the transform domain:
/// twice the 8-bit code less 128.
fn ycbcr(pixel: &[u8]) -> (f32, f32, f32) {
    let (r, g, b) = (
        f32::from(pixel[0]),
        f32::from(pixel[1]),
        f32::from(pixel[2]),
    );
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let y = 16.0 + 219.0 / 255.0 * luma;
    let cb = 128.0 + 224.0 / 255.0 * (b - luma) / 1.8556;
    let cr = 128.0 + 224.0 / 255.0 * (r - luma) / 1.5748;
    (2.0 * (y - 128.0), 2.0 * (cb - 128.0), 2.0 * (cr - 128.0))
}

fn planes_from_rgba(width: u32, height: u32, rgba: &[u8], profile: Profile) -> Planes {
    let chroma = target_chroma(profile);
    let (luma_stride, chroma_stride, rows) = padded(width, height, chroma);
    let w = width as usize;
    let h = height as usize;
    let pixel = |x: usize, y: usize| {
        let at = (y.min(h - 1) * w + x.min(w - 1)) * 4;
        &rgba[at..at + 4]
    };
    let mut y = vec![0f32; luma_stride * rows];
    let mut u = vec![0f32; chroma_stride * rows];
    let mut v = vec![0f32; chroma_stride * rows];
    for r in 0..rows {
        for c in 0..luma_stride {
            y[r * luma_stride + c] = ycbcr(pixel(c, r)).0;
        }
        for c in 0..chroma_stride {
            let (cb, cr) = match chroma {
                Chroma::Yuv444 => {
                    let (_, cb, cr) = ycbcr(pixel(c, r));
                    (cb, cr)
                }
                Chroma::Yuv422 => {
                    let (_, b0, r0) = ycbcr(pixel(2 * c, r));
                    let (_, b1, r1) = ycbcr(pixel(2 * c + 1, r));
                    ((b0 + b1) * 0.5, (r0 + r1) * 0.5)
                }
            };
            u[r * chroma_stride + c] = cb;
            v[r * chroma_stride + c] = cr;
        }
    }
    let alpha = (chroma == Chroma::Yuv444).then(|| {
        let mut samples = vec![0u16; luma_stride * h];
        for r in 0..h {
            for c in 0..luma_stride {
                samples[r * luma_stride + c] = u16::from(pixel(c, r)[3]);
            }
        }
        AlphaPlane {
            sixteen: false,
            stride: luma_stride,
            samples,
        }
    });
    Planes {
        width,
        height,
        chroma,
        luma_stride,
        chroma_stride,
        y,
        u,
        v,
        alpha,
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
                let (w, h) = (frame.width(), frame.height());
                let luma = w as usize * h as usize;
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                let chroma = cw as usize * ch as usize;
                let bytes = frame.bytes();
                if bytes.len() != luma + 2 * chroma {
                    return Err(CodecError::InvalidApi {
                        what: "picture byte length",
                    });
                }
                let picture = VideoFrame::yuv420p(
                    w,
                    h,
                    Plane::tight(w, h, bytes[..luma].to_vec())?,
                    Plane::tight(cw, ch, bytes[luma..luma + chroma].to_vec())?,
                    Plane::tight(cw, ch, bytes[luma + chroma..].to_vec())?,
                )?;
                self.encode(&picture)?
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
