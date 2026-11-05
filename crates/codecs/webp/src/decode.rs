//! WebP pictures from the still container's packets.
//!
//! A still file is one packet: the whole file. An animated file is one
//! packet per `ANMF` frame; the first also carries the RIFF header,
//! `VP8X`, and `ANIM`, so the decoder keeps the canvas between packets and
//! paints each frame onto it. Every output is the whole canvas as straight
//! RGBA. Packets of an animation must arrive in order from the first; a
//! seek starts again at the first packet, the only keyframe.
//!
//! Compositing follows the container specification: the previous frame's
//! rectangle is cleared to transparent when it asked to be disposed, then
//! the new frame is drawn into its rectangle, alpha-blended over the
//! canvas unless it asked not to be. The `ANIM` background color is a hint
//! the specification lets a decoder ignore; like the reference decoder and
//! browsers, the canvas starts and is disposed to transparent black. The
//! blend uses the reference decoder's integer approximation of the
//! non-premultiplied formula, and its keyframe rule (a frame that needs
//! nothing from the canvas is drawn onto a cleared canvas without
//! blending), so frames match it bit for bit.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, Transfer,
};

use crate::picture::{
    Chunk, Rgba, chunks, decode_picture, malformed, picture_size, truncated, u24,
};

/// The largest canvas or picture this decoder paints, in pixels: a quarter
/// of the format's 16384 x 16384, about 268 MB of RGBA.
pub const MAX_PIXELS: usize = 16_384 * 16_384 / 4;

const VP8X_ANIMATION: u8 = 0x02;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Rect {
    fn covers(&self, width: usize, height: usize) -> bool {
        self.x == 0 && self.y == 0 && self.width == width && self.height == height
    }
}

/// What the last frame left for the next one.
#[derive(Clone, Copy)]
struct Previous {
    rect: Rect,
    dispose: bool,
    keyframe: bool,
}

struct Canvas {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    previous: Option<Previous>,
}

/// One `ANMF` frame, decoded and ready to paint.
struct Frame {
    rect: Rect,
    dispose: bool,
    blend: bool,
    picture: Rgba,
}

/// A WebP decoder that keeps an animation's canvas from packet to packet.
#[derive(Default)]
pub struct WebpDecoder {
    canvas: Option<Canvas>,
    max_pixels: usize,
}

impl WebpDecoder {
    /// Open a decoder. The canvas size comes from the first packet.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Webp {
            return Err(CodecError::Unsupported { what: "webp" });
        }
        Ok(Self::with_limit(MAX_PIXELS))
    }

    /// A decoder that refuses a canvas or picture above `max_pixels`.
    #[must_use]
    pub fn with_limit(max_pixels: usize) -> Self {
        Self {
            canvas: None,
            max_pixels: max_pixels.min(MAX_PIXELS),
        }
    }

    fn picture(
        width: usize,
        height: usize,
        rgba: Vec<u8>,
        packet: &Packet,
    ) -> Result<Output, CodecError> {
        let info = PictureInfo {
            width: u32::try_from(width).map_err(|_| limit("webp width"))?,
            height: u32::try_from(height).map_err(|_| limit("webp height"))?,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Srgb,
        };
        Ok(Output::Picture(PictureOutput::cpu(
            info,
            rgba,
            packet.pts,
            packet.timescale,
        )?))
    }

    fn check_size(&self, width: usize, height: usize) -> Result<(), CodecError> {
        match width.checked_mul(height) {
            Some(pixels) if pixels > 0 && pixels <= self.max_pixels => Ok(()),
            Some(0) => Err(malformed(0, "webp size")),
            _ => Err(limit("webp canvas size")),
        }
    }

    /// A packet that starts a file: a still picture, or the start of an
    /// animation followed by its first frames.
    fn start(&mut self, data: &[u8], packet: &Packet) -> Result<Vec<Output>, CodecError> {
        // An animation's first packet stops before the end the RIFF size
        // names; a still file may carry trailing bytes after it.
        let declared = usize::try_from(u32::from_le_bytes([data[4], data[5], data[6], data[7]]))
            .unwrap_or(usize::MAX);
        let end = declared.saturating_add(8).min(data.len());
        if end < 12 {
            return Err(malformed(4, "riff size"));
        }
        let body = &data[12..end];
        let list = chunks(body)?;
        let Some(first) = list.first() else {
            return Err(malformed(12, "webp picture missing"));
        };
        if &first.fourcc != b"VP8X" {
            let (width, height) = picture_size(&list)?;
            self.check_size(to_usize(width), to_usize(height))?;
            let picture = decode_picture(&list, self.max_pixels)?;
            self.canvas = None;
            return Ok(vec![Self::picture(
                picture.width,
                picture.height,
                picture.pixels,
                packet,
            )?]);
        }
        let header = first.payload;
        if header.len() < 10 {
            return Err(truncated(first.at, "vp8x"));
        }
        let width = to_usize(u24(header, 4)) + 1;
        let height = to_usize(u24(header, 7)) + 1;
        self.check_size(width, height)?;
        if header[0] & VP8X_ANIMATION == 0 {
            let picture = decode_picture(&list[1..], self.max_pixels)?;
            if (picture.width, picture.height) != (width, height) {
                return Err(malformed(first.at, "webp canvas and picture size"));
            }
            self.canvas = None;
            return Ok(vec![Self::picture(width, height, picture.pixels, packet)?]);
        }
        let frames = self.frames(&list[1..], width, height)?;
        let mut canvas = Canvas {
            width,
            height,
            rgba: vec![0; width * height * 4],
            previous: None,
        };
        let outputs = paint_all(&mut canvas, frames, packet)?;
        self.canvas = Some(canvas);
        Ok(outputs)
    }

    /// Decode every `ANMF` among `list` against a canvas of this size.
    fn frames(
        &self,
        list: &[Chunk<'_>],
        width: usize,
        height: usize,
    ) -> Result<Vec<Frame>, CodecError> {
        let mut frames = Vec::new();
        for chunk in list.iter().filter(|chunk| &chunk.fourcc == b"ANMF") {
            let data = chunk.payload;
            if data.len() < 16 {
                return Err(truncated(chunk.at, "anmf header"));
            }
            let rect = Rect {
                x: to_usize(u24(data, 0)) * 2,
                y: to_usize(u24(data, 3)) * 2,
                width: to_usize(u24(data, 6)) + 1,
                height: to_usize(u24(data, 9)) + 1,
            };
            if rect.x + rect.width > width || rect.y + rect.height > height {
                return Err(malformed(chunk.at, "anmf outside the canvas"));
            }
            let flags = data[15];
            let inner = chunks(&data[16..])?;
            let picture = decode_picture(&inner, self.max_pixels)?;
            if (picture.width, picture.height) != (rect.width, rect.height) {
                return Err(malformed(chunk.at, "anmf and picture size"));
            }
            frames.push(Frame {
                rect,
                dispose: flags & 1 == 1,
                blend: flags & 2 == 0,
                picture,
            });
        }
        Ok(frames)
    }
}

fn paint_all(
    canvas: &mut Canvas,
    frames: Vec<Frame>,
    packet: &Packet,
) -> Result<Vec<Output>, CodecError> {
    let mut outputs = Vec::with_capacity(frames.len());
    for frame in frames {
        paint(canvas, &frame);
        outputs.push(WebpDecoder::picture(
            canvas.width,
            canvas.height,
            canvas.rgba.clone(),
            packet,
        )?);
    }
    Ok(outputs)
}

/// Draw one frame: dispose the last, then copy or blend the new one.
fn paint(canvas: &mut Canvas, frame: &Frame) {
    let (width, height) = (canvas.width, canvas.height);
    let keyframe = match canvas.previous {
        None => true,
        Some(previous) => {
            ((!frame.picture.has_alpha || !frame.blend) && frame.rect.covers(width, height))
                || (previous.dispose && (previous.rect.covers(width, height) || previous.keyframe))
        }
    };
    if keyframe {
        canvas.rgba.fill(0);
    } else if let Some(previous) = canvas.previous
        && previous.dispose
    {
        clear(canvas, previous.rect);
    }
    let rect = frame.rect;
    for row in 0..rect.height {
        let source = &frame.picture.pixels[row * rect.width * 4..(row + 1) * rect.width * 4];
        let start = ((rect.y + row) * width + rect.x) * 4;
        let target = &mut canvas.rgba[start..start + rect.width * 4];
        if keyframe || !frame.blend {
            target.copy_from_slice(source);
            continue;
        }
        for (dst, src) in target
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(source.as_chunks::<4>().0)
        {
            *dst = blend(*src, *dst);
        }
    }
    canvas.previous = Some(Previous {
        rect,
        dispose: frame.dispose,
        keyframe,
    });
}

fn clear(canvas: &mut Canvas, rect: Rect) {
    for row in rect.y..rect.y + rect.height {
        let start = (row * canvas.width + rect.x) * 4;
        canvas.rgba[start..start + rect.width * 4].fill(0);
    }
}

/// `src` over `dst`, both straight RGBA.
///
/// The container specification's formula, `A = As + Ad * (1 - As)` and
/// each color `(Cs * As + Cd * Ad * (1 - As)) / A`, in the reference
/// decoder's fixed point: `Ad * (1 - As)` is `Ad * (256 - As) >> 8`, and
/// the division is a multiply by `2^24 / A`. An opaque source replaces
/// the pixel, and a clear one leaves it.
pub(crate) fn blend(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let src_alpha = u32::from(src[3]);
    if src_alpha == 255 {
        return src;
    }
    if src_alpha == 0 {
        return dst;
    }
    let dst_factor = (u32::from(dst[3]) * (256 - src_alpha)) >> 8;
    let alpha = src_alpha + dst_factor;
    let scale = (1_u32 << 24) / alpha;
    let channel = |s: u8, d: u8| {
        let mixed = u32::from(s) * src_alpha + u32::from(d) * dst_factor;
        u8::try_from((u64::from(mixed) * u64::from(scale)) >> 24).unwrap_or(u8::MAX)
    };
    [
        channel(src[0], dst[0]),
        channel(src[1], dst[1]),
        channel(src[2], dst[2]),
        u8::try_from(alpha).unwrap_or(u8::MAX),
    ]
}

impl Decoder for WebpDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Webp
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let data = &packet.data;
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return self.start(data, packet);
        }
        if data.starts_with(b"RIFF") {
            return Err(malformed(0, "riff form"));
        }
        let Some(canvas) = self.canvas.as_ref() else {
            return Err(CodecError::InvalidApi {
                what: "webp header",
            });
        };
        let list = chunks(data)?;
        let frames = self.frames(&list, canvas.width, canvas.height)?;
        // Decoding is done; painting cannot fail, so a bad packet above
        // left the canvas as it was.
        let Some(canvas) = self.canvas.as_mut() else {
            return Err(CodecError::InvalidApi {
                what: "webp header",
            });
        };
        paint_all(canvas, frames, packet)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.canvas = None;
    }
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn limit(what: &'static str) -> CodecError {
    CodecError::Limit { what }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rect: Rect, color: [u8; 4], dispose: bool, blend: bool) -> Frame {
        Frame {
            rect,
            dispose,
            blend,
            picture: Rgba {
                width: rect.width,
                height: rect.height,
                pixels: color.repeat(rect.width * rect.height),
                has_alpha: color[3] != 255,
            },
        }
    }

    fn canvas(width: usize, height: usize) -> Canvas {
        Canvas {
            width,
            height,
            rgba: vec![0; width * height * 4],
            previous: None,
        }
    }

    fn pixel(canvas: &Canvas, x: usize, y: usize) -> [u8; 4] {
        let at = (y * canvas.width + x) * 4;
        [
            canvas.rgba[at],
            canvas.rgba[at + 1],
            canvas.rgba[at + 2],
            canvas.rgba[at + 3],
        ]
    }

    const RED: [u8; 4] = [255, 0, 0, 255];
    const HALF_BLUE: [u8; 4] = [0, 0, 255, 128];

    #[test]
    fn blend_follows_the_formula_in_fixed_point() {
        assert_eq!(blend(RED, HALF_BLUE), RED);
        assert_eq!(blend([9, 9, 9, 0], HALF_BLUE), HALF_BLUE);
        // Half blue over opaque red: alpha 128 + 255 * 128 / 256 = 255.
        let mixed = blend(HALF_BLUE, RED);
        assert_eq!(mixed[3], 255);
        assert!(
            (126..=128).contains(&mixed[0]) && (126..=128).contains(&mixed[2]),
            "{mixed:?}"
        );
        // Over a clear pixel the color is kept, the alpha is the source's.
        let alone = blend(HALF_BLUE, [0, 0, 0, 0]);
        assert_eq!(alone, [0, 0, 255, 128]);
    }

    #[test]
    fn dispose_and_blend_paint_the_canvas() {
        let mut canvas = canvas(4, 2);
        let whole = Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 2,
        };
        let corner = Rect {
            x: 2,
            y: 0,
            width: 2,
            height: 1,
        };
        paint(&mut canvas, &frame(whole, RED, false, true));
        assert_eq!(pixel(&canvas, 3, 1), RED);
        // Blended half blue over red; disposed afterwards.
        paint(&mut canvas, &frame(corner, HALF_BLUE, true, true));
        let mixed = pixel(&canvas, 2, 0);
        assert_eq!(mixed[3], 255);
        assert!(mixed[0] > 100 && mixed[2] > 100, "{mixed:?}");
        assert_eq!(pixel(&canvas, 1, 0), RED);
        // The corner was cleared to transparent before this frame, and
        // no-blend copies the half blue as it is.
        let left = Rect {
            x: 0,
            y: 1,
            width: 1,
            height: 1,
        };
        paint(&mut canvas, &frame(left, HALF_BLUE, false, false));
        assert_eq!(pixel(&canvas, 2, 0), [0, 0, 0, 0]);
        assert_eq!(pixel(&canvas, 0, 1), HALF_BLUE);
        assert_eq!(pixel(&canvas, 3, 1), RED);
        // An opaque frame over the whole canvas is a keyframe.
        paint(&mut canvas, &frame(whole, [1, 2, 3, 255], false, true));
        assert!(
            canvas
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [1, 2, 3, 255])
        );
        assert!(canvas.previous.is_some_and(|p| p.keyframe));
    }
}
