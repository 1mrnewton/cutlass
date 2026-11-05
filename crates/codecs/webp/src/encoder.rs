//! The encoder trait for WebP stills and animations.
//!
//! A picture becomes one still file. Frames become one animation: the
//! encoder holds them until `drain`, because the first packet carries the
//! size of the whole file. Packet 0 is the file start through the first
//! `ANMF` chunk and each later packet is one `ANMF` chunk, the layout the
//! `still` container splits a file into and joins back.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, BitDepth, CodecError, CodecId, CodecParams, ColorType, Encoder, FrameInput, Packet,
    PictureInput, PixelFormat, StillImage, Transparency,
};

use crate::limits::{MAX_EDGE, MAX_FRAMES_PER_PACKET, picture_ok};
use crate::lossless::encode::{self as lossless, Config};
use crate::lossy::encode as lossy;
use crate::riff::{
    Bitstream, FLAG_ALPHA, FLAG_ANIMATION, file, push_bitstream, put_chunk, vp8x, write_still,
};

/// How to code pictures, carried in `CodecParams::extra` as
/// [`WebpOptions::to_extra`] writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WebpOptions {
    /// VP8L instead of VP8.
    pub lossless: bool,
    /// 0 to 100, as `cwebp -q`. Ignored when lossless.
    pub quality: u8,
    /// Lossy: 0 to 6, as `cwebp -m`. Lossless: 0 to 9, as `cwebp -z`.
    pub effort: u8,
    /// Keep the RGB under fully transparent pixels.
    pub exact: bool,
}

impl Default for WebpOptions {
    /// `cwebp`'s defaults: lossy, quality 75, method 4.
    fn default() -> Self {
        Self {
            lossless: false,
            quality: 75,
            effort: 4,
            exact: false,
        }
    }
}

impl WebpOptions {
    /// Lossless at `cwebp -z 6`, the effort `cwebp -lossless` uses.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            lossless: true,
            quality: 75,
            effort: 6,
            exact: false,
        }
    }

    /// Four bytes: lossless, quality, effort, exact.
    #[must_use]
    pub fn to_extra(self) -> Vec<u8> {
        vec![
            u8::from(self.lossless),
            self.quality,
            self.effort,
            u8::from(self.exact),
        ]
    }

    /// Empty bytes are the defaults.
    pub fn from_extra(extra: &[u8]) -> Result<Self, CodecError> {
        match *extra {
            [] => Ok(Self::default()),
            [lossless @ 0..=1, quality @ 0..=100, effort, exact @ 0..=1]
                if effort <= if lossless == 1 { 9 } else { 6 } =>
            {
                Ok(Self {
                    lossless: lossless == 1,
                    quality,
                    effort,
                    exact: exact == 1,
                })
            }
            _ => Err(CodecError::Unsupported {
                what: "webp encoder options",
            }),
        }
    }
}

/// One coded picture: its chunks' payloads.
struct Coded {
    vp8: Vec<u8>,
    alpha: Option<Vec<u8>>,
    lossless: bool,
}

impl Coded {
    fn bitstream(&self) -> Bitstream<'_> {
        if self.lossless {
            Bitstream::Lossless(&self.vp8)
        } else {
            Bitstream::Lossy {
                vp8: &self.vp8,
                alpha: self.alpha.as_deref(),
            }
        }
    }
}

fn code(options: WebpOptions, rgba: &[u8], width: u32, height: u32) -> Result<Coded, CodecError> {
    if options.lossless {
        let argb: Vec<u32> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| u32::from_be_bytes([p[3], p[0], p[1], p[2]]))
            .collect();
        let config = Config {
            effort: options.effort,
            exact: options.exact,
        };
        Ok(Coded {
            vp8: lossless::encode(&argb, width, height, config),
            alpha: None,
            lossless: true,
        })
    } else {
        let config = lossy::Config {
            quality: options.quality,
            method: options.effort,
            exact: options.exact,
        };
        let (vp8, alpha) = lossy::encode(rgba, width, height, config)?;
        Ok(Coded {
            vp8,
            alpha,
            lossless: false,
        })
    }
}

fn has_alpha(rgba: &[u8]) -> bool {
    rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255)
}

/// One held animation frame.
struct Held {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    coded: Coded,
    alpha: bool,
    duration_ms: u32,
    pts: i64,
    duration: i64,
    timescale: u32,
}

pub struct WebpEncoder {
    options: WebpOptions,
    /// Canvas size and the last frame's pixels.
    canvas: Option<(u32, u32, Vec<u8>)>,
    frames: Vec<Held>,
    cancel: Arc<AtomicBool>,
}

impl WebpEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Webp {
            return Err(CodecError::Unsupported { what: "webp" });
        }
        Ok(Self {
            options: WebpOptions::from_extra(&params.extra)?,
            canvas: None,
            frames: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    fn check_cancel(&self) -> Result<(), CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(CodecError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// The smallest rectangle, with an even corner, holding every pixel that
/// differs. A frame equal to the last is one pixel.
fn changed(previous: &[u8], next: &[u8], width: u32) -> (u32, u32, u32, u32) {
    let w = width as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (y, (a, b)) in previous
        .chunks_exact(w * 4)
        .zip(next.chunks_exact(w * 4))
        .enumerate()
    {
        if a == b {
            continue;
        }
        let (Some(first), Some(last)) = (
            a.as_chunks::<4>()
                .0
                .iter()
                .zip(b.as_chunks::<4>().0)
                .position(|(p, q)| p != q),
            a.as_chunks::<4>()
                .0
                .iter()
                .zip(b.as_chunks::<4>().0)
                .rposition(|(p, q)| p != q),
        ) else {
            continue;
        };
        x0 = x0.min(first);
        x1 = x1.max(last + 1);
        y0 = y0.min(y);
        y1 = y + 1;
    }
    if x0 == usize::MAX {
        return (0, 0, 1, 1);
    }
    let x0 = x0 & !1;
    let y0 = y0 & !1;
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
}

fn crop(rgba: &[u8], width: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let at = ((row * width + x) * 4) as usize;
        out.extend_from_slice(&rgba[at..at + (w * 4) as usize]);
    }
    out
}

fn u24(value: u32) -> [u8; 3] {
    let b = value.to_le_bytes();
    [b[0], b[1], b[2]]
}

/// One `ANMF` chunk.
fn anmf(frame: &Held) -> Vec<u8> {
    let mut payload = Vec::with_capacity(frame.coded.vp8.len() + 32);
    payload.extend_from_slice(&u24(frame.x / 2));
    payload.extend_from_slice(&u24(frame.y / 2));
    payload.extend_from_slice(&u24(frame.width - 1));
    payload.extend_from_slice(&u24(frame.height - 1));
    payload.extend_from_slice(&u24(frame.duration_ms));
    // Do not blend, do not dispose: the rectangle replaces the canvas.
    payload.push(0x02);
    let mut chunks = Vec::with_capacity(2);
    push_bitstream(&mut chunks, frame.coded.bitstream());
    for (fourcc, data) in chunks {
        put_chunk(&mut payload, fourcc, data);
    }
    let mut out = Vec::with_capacity(payload.len() + 8);
    put_chunk(&mut out, b"ANMF", &payload);
    out
}

/// Milliseconds of `duration` ticks, as the 24 bits of `ANMF` hold.
fn milliseconds(duration: i64, timescale: u32) -> u32 {
    let ms =
        (i128::from(duration) * 1000 + i128::from(timescale) / 2) / i128::from(timescale.max(1));
    u32::try_from(ms.clamp(0, (1 << 24) - 1)).unwrap_or(0)
}

/// Straight 8-bit RGBA of any still.
pub(crate) fn rgba_of(image: &StillImage) -> Result<Vec<u8>, CodecError> {
    let pixels = (image.width as usize) * (image.height as usize);
    let channels = image.color.channels();
    if image.samples.len() != pixels * channels {
        return Err(CodecError::InvalidApi {
            what: "still samples",
        });
    }
    let max = u32::from(image.depth.max_sample());
    let scale = |v: u16| -> u8 {
        if image.depth == BitDepth::Eight {
            v as u8
        } else {
            ((u32::from(v).min(max) * 255 + max / 2) / max) as u8
        }
    };
    let key = match &image.transparency {
        Transparency::ColorKey(key) if key.len() == channels => Some(key.as_slice()),
        _ => None,
    };
    let mut out = Vec::with_capacity(pixels * 4);
    for sample in image.samples.chunks_exact(channels) {
        let keyed = key.is_some_and(|key| key == sample);
        let pixel = match image.color {
            ColorType::Gray => {
                let g = scale(sample[0]);
                [g, g, g, 255]
            }
            ColorType::GrayAlpha => {
                let g = scale(sample[0]);
                [g, g, g, scale(sample[1])]
            }
            ColorType::Rgb => [scale(sample[0]), scale(sample[1]), scale(sample[2]), 255],
            ColorType::Rgba => [
                scale(sample[0]),
                scale(sample[1]),
                scale(sample[2]),
                scale(sample[3]),
            ],
            ColorType::Indexed => {
                let index = usize::from(sample[0]);
                let color = image.palette.get(index).ok_or(CodecError::InvalidApi {
                    what: "palette index",
                })?;
                let alpha = match &image.transparency {
                    Transparency::PaletteAlpha(alpha) => alpha.get(index).copied().unwrap_or(255),
                    _ => 255,
                };
                [color.red, color.green, color.blue, alpha]
            }
        };
        out.extend_from_slice(&if keyed {
            [pixel[0], pixel[1], pixel[2], 0]
        } else {
            pixel
        });
    }
    Ok(out)
}

fn size_ok(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return Err(CodecError::Unsupported { what: "webp size" });
    }
    picture_ok(width, height)
}

impl Encoder for WebpEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Webp
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "webp encodes pictures",
        })
    }

    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        self.check_cancel()?;
        let image = &input.image;
        size_ok(image.width, image.height)?;
        let rgba = rgba_of(image)?;
        let coded = code(self.options, &rgba, image.width, image.height)?;
        self.check_cancel()?;
        let icc = (!image.icc_profile.is_empty()).then_some(image.icc_profile.as_slice());
        let data = write_still(
            coded.bitstream(),
            image.width,
            image.height,
            has_alpha(&rgba),
            icc,
        );
        Ok(vec![Packet {
            stream: 0,
            data,
            pts: 0,
            duration: 0,
            timescale: 0,
            keyframe: true,
            side: Vec::new(),
        }])
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        self.check_cancel()?;
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform picture",
            });
        };
        if frame.pixel() != PixelFormat::Rgba8 {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let (width, height) = (frame.width(), frame.height());
        size_ok(width, height)?;
        if self.frames.len() >= MAX_FRAMES_PER_PACKET {
            return Err(CodecError::Limit {
                what: "webp frames",
            });
        }
        let bytes = frame.bytes();
        let (x, y, w, h) = match &self.canvas {
            None => (0, 0, width, height),
            Some((cw, ch, _)) if (*cw, *ch) != (width, height) => {
                return Err(CodecError::InvalidApi {
                    what: "webp frame size changed",
                });
            }
            Some((_, _, previous)) => changed(previous, bytes, width),
        };
        let rect = crop(bytes, width, x, y, w, h);
        let coded = code(self.options, &rect, w, h)?;
        let time = frame.time();
        self.frames.push(Held {
            x,
            y,
            width: w,
            height: h,
            alpha: has_alpha(&rect),
            coded,
            duration_ms: milliseconds(time.duration(), time.timescale()),
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
        });
        self.canvas = Some((width, height, bytes.to_vec()));
        Ok(Vec::new())
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        self.check_cancel()?;
        let Some((width, height, _)) = self.canvas.take() else {
            return Ok(Vec::new());
        };
        let frames = std::mem::take(&mut self.frames);
        let chunks: Vec<Vec<u8>> = frames.iter().map(anmf).collect();
        let mut flags = FLAG_ANIMATION;
        if frames.iter().any(|frame| frame.alpha) {
            flags |= FLAG_ALPHA;
        }
        let header = vp8x(flags, width, height);
        // Background (ignored by players that follow libwebp) and loop
        // count, which the muxer patches.
        let anim = [0u8, 0, 0, 0, 0, 0];
        let start = file(&[(b"VP8X", &header), (b"ANIM", &anim)]);
        let total = start.len() + chunks.iter().map(Vec::len).sum::<usize>();
        let riff_size = u32::try_from(total - 8).map_err(|_| CodecError::Limit {
            what: "webp file size",
        })?;
        let mut packets = Vec::with_capacity(frames.len());
        for (index, (frame, chunk)) in frames.iter().zip(chunks).enumerate() {
            let data = if index == 0 {
                let mut data = start.clone();
                data[4..8].copy_from_slice(&riff_size.to_le_bytes());
                data.extend_from_slice(&chunk);
                data
            } else {
                chunk
            };
            packets.push(Packet::video(
                data,
                frame.pts,
                frame.duration,
                frame.timescale,
                true,
            ));
        }
        Ok(packets)
    }

    fn reset(&mut self) {
        self.canvas = None;
        self.frames.clear();
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
