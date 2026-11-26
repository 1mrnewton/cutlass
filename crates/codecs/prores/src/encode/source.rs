//! The picture being encoded, read where it lies: each slice job takes
//! its own samples from the caller's planes, so no whole-frame copy is made.

use cutlass_decode::{CodecError, PixelFormat, Plane, VideoFrame};

use crate::header::Chroma;

/// Samples in the widest slice row: 8 macroblocks of 16.
pub(crate) const MAX_COLUMNS: usize = 128;

/// One plane of 8-bit or 16-bit little-endian samples.
#[derive(Clone, Copy)]
pub(crate) struct PlaneView<'a> {
    samples: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
    wide: bool,
}

impl<'a> PlaneView<'a> {
    /// A view of `width` by `height` samples, refused when the bytes do
    /// not hold them.
    pub(crate) fn new(
        samples: &'a [u8],
        stride: usize,
        width: usize,
        height: usize,
        wide: bool,
    ) -> Result<Self, CodecError> {
        let short = CodecError::InvalidApi { what: "plane size" };
        let row = width.checked_mul(if wide { 2 } else { 1 }).ok_or(short)?;
        if width == 0 || height == 0 || stride < row {
            return Err(short);
        }
        let need = stride
            .checked_mul(height - 1)
            .and_then(|above| above.checked_add(row))
            .ok_or(short)?;
        if samples.len() < need {
            return Err(short);
        }
        Ok(Self {
            samples,
            stride,
            width,
            height,
            wide,
        })
    }

    fn of(plane: &'a Plane, wide: bool) -> Result<Self, CodecError> {
        Self::new(
            &plane.samples,
            plane.stride,
            plane.width as usize,
            plane.height as usize,
            wide,
        )
    }

    /// The bytes of row `y`, the last row for any `y` past it.
    #[inline(always)]
    fn row(&self, y: usize) -> &'a [u8] {
        let start = y.min(self.height - 1) * self.stride;
        let len = self.width * if self.wide { 2 } else { 1 };
        self.samples.get(start..start + len).unwrap_or_default()
    }

    /// Sample `x` of a row, the last sample for any `x` past it.
    #[inline(always)]
    fn at(&self, row: &[u8], x: usize) -> u32 {
        let x = x.min(self.width - 1);
        if self.wide {
            match row.get(2 * x..2 * x + 2) {
                Some(pair) => u32::from(u16::from_le_bytes([pair[0], pair[1]])),
                None => 0,
            }
        } else {
            row.get(x).map_or(0, |&byte| u32::from(byte))
        }
    }

    /// Fills `out` with samples `x0..` of row `y` through `map`, repeating
    /// the last sample past the plane's width or past `limit`.
    #[inline(always)]
    fn fill(&self, y: usize, x0: usize, limit: usize, out: &mut [f32], map: impl Fn(u32) -> f32) {
        let row = self.row(y);
        let width = self.width.min(limit);
        let inside = width.saturating_sub(x0).min(out.len());
        let (head, tail) = out.split_at_mut(inside);
        if self.wide {
            let bytes = row.get(2 * x0..2 * (x0 + inside)).unwrap_or_default();
            for (value, pair) in head.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *value = map(u32::from(u16::from_le_bytes(*pair)));
            }
        } else {
            let bytes = row.get(x0..x0 + inside).unwrap_or_default();
            for (value, &byte) in head.iter_mut().zip(bytes) {
                *value = map(u32::from(byte));
            }
        }
        if !tail.is_empty() {
            tail.fill(map(self.at(row, width.saturating_sub(1))));
        }
    }
}

/// Where the samples come from.
pub(crate) enum Source<'a> {
    /// Planar Y'CbCr at `depth` bits, chroma subsampled by `sx` and `sy`.
    Planar {
        y: PlaneView<'a>,
        u: PlaneView<'a>,
        v: PlaneView<'a>,
        depth: u32,
        sx: usize,
        sy: usize,
        alpha: Option<(PlaneView<'a>, u32)>,
    },
    /// 8-bit RGBA, coded as BT.709 limited range.
    Rgba { pixels: PlaneRgba<'a> },
}

/// Tight 8-bit RGBA rows.
#[derive(Clone, Copy)]
pub(crate) struct PlaneRgba<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> PlaneRgba<'a> {
    #[inline(always)]
    fn row(&self, y: usize) -> &'a [u8] {
        let start = y.min(self.height - 1) * self.width * 4;
        self.bytes
            .get(start..start + self.width * 4)
            .unwrap_or_default()
    }
}

/// A picture and the layout it is coded in.
pub(crate) struct Picture<'a> {
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
    /// Whether alpha is coded, and then whether at 16 bits.
    pub alpha: Option<bool>,
    source: Source<'a>,
}

impl<'a> Picture<'a> {
    /// A planar frame, coded with `chroma`. Alpha is kept for 4:4:4 only.
    pub(crate) fn planar(frame: &'a VideoFrame, chroma: Chroma) -> Result<Self, CodecError> {
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
        let y = PlaneView::of(&frame.y, wide)?;
        let u = PlaneView::of(&frame.u, wide)?;
        let v = PlaneView::of(&frame.v, wide)?;
        let short = CodecError::InvalidApi { what: "plane size" };
        let (chroma_w, chroma_h) = frame
            .format
            .chroma_wh(frame.width, frame.height)
            .ok_or(short)?;
        let sized = |plane: &PlaneView<'_>, width: u32, height: u32| {
            plane.width == width as usize && plane.height == height as usize
        };
        if !sized(&y, frame.width, frame.height)
            || !sized(&u, chroma_w, chroma_h)
            || !sized(&v, chroma_w, chroma_h)
        {
            return Err(short);
        }
        let alpha = match (&frame.alpha, chroma) {
            (Some(alpha), Chroma::Yuv444) => {
                let bits = u32::from(alpha.bit_depth);
                if bits == 0 || bits > 16 {
                    return Err(CodecError::InvalidApi {
                        what: "alpha bit depth",
                    });
                }
                let view = PlaneView::of(&alpha.plane, bits > 8)?;
                if !sized(&view, frame.width, frame.height) {
                    return Err(short);
                }
                Some((view, bits))
            }
            _ => None,
        };
        Ok(Self {
            width: frame.width,
            height: frame.height,
            chroma,
            alpha: alpha.map(|(_, bits)| bits > 8),
            source: Source::Planar {
                y,
                u,
                v,
                depth,
                sx,
                sy,
                alpha,
            },
        })
    }

    /// Tight 8-bit 4:2:0: Y, then U, then V, the chroma planes rounded up.
    pub(crate) fn yuv420(
        width: u32,
        height: u32,
        bytes: &'a [u8],
        chroma: Chroma,
    ) -> Result<Self, CodecError> {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let luma = w * h;
        if bytes.len() != luma + 2 * cw * ch {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        let (y, rest) = bytes.split_at(luma);
        let (u, v) = rest.split_at(cw * ch);
        Ok(Self {
            width,
            height,
            chroma,
            alpha: None,
            source: Source::Planar {
                y: PlaneView::new(y, w, w, h, false)?,
                u: PlaneView::new(u, cw, cw, ch, false)?,
                v: PlaneView::new(v, cw, cw, ch, false)?,
                depth: 8,
                sx: 2,
                sy: 2,
                alpha: None,
            },
        })
    }

    /// Tight 8-bit RGBA. A 4:4:4 layout keeps the alpha channel.
    pub(crate) fn rgba(
        width: u32,
        height: u32,
        bytes: &'a [u8],
        chroma: Chroma,
    ) -> Result<Self, CodecError> {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 || bytes.len() != w * h * 4 {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        Ok(Self {
            width,
            height,
            chroma,
            alpha: (chroma == Chroma::Yuv444).then_some(false),
            source: Source::Rgba {
                pixels: PlaneRgba {
                    bytes,
                    width: w,
                    height: h,
                },
            },
        })
    }

    /// Samples in a slice's chroma row for `columns` luma samples.
    #[inline(always)]
    pub(crate) fn chroma_columns(&self, columns: usize) -> usize {
        match self.chroma {
            Chroma::Yuv422 => columns / 2,
            Chroma::Yuv444 => columns,
        }
    }

    /// Reads the 16 rows of the slice at macroblock `(mb_x, row)` in the
    /// transform's domain: each sample less mid-grey, scaled to RDD 36's
    /// 9-bit range. Rows and columns past the picture repeat its last.
    /// `ys` takes `columns` samples a row, `us` and `vs` the chroma's.
    #[inline(always)]
    pub(crate) fn gather(
        &self,
        row: usize,
        mb_x: usize,
        columns: usize,
        ys: &mut [f32],
        us: &mut [f32],
        vs: &mut [f32],
    ) {
        let width = self.width as usize;
        let height = self.height as usize;
        let x0 = mb_x * 16;
        let chroma_columns = self.chroma_columns(columns);
        let lines = ys
            .chunks_exact_mut(columns)
            .zip(us.chunks_exact_mut(chroma_columns))
            .zip(vs.chunks_exact_mut(chroma_columns))
            .take(16)
            .enumerate();
        match &self.source {
            Source::Planar {
                y,
                u,
                v,
                depth,
                sx,
                sy,
                ..
            } => {
                let mid = (1u32 << (depth - 1)) as f32;
                let unit = 2f32.powi(*depth as i32 - 9);
                let norm = move |value: u32| (value as f32 - mid) / unit;
                let target_w = match self.chroma {
                    Chroma::Yuv422 => width.div_ceil(2),
                    Chroma::Yuv444 => width,
                };
                let c0 = self.chroma_columns(x0);
                for (r, ((ys, us), vs)) in lines {
                    let fr = (row * 16 + r).min(height - 1);
                    y.fill(fr, x0, width, ys, norm);
                    let sr = fr / sy;
                    for (plane, out) in [(u, us), (v, vs)] {
                        match (self.chroma, *sx) {
                            // 4:4:4 source into 4:2:2: average the pair.
                            (Chroma::Yuv422, 1) => {
                                let line = plane.row(sr);
                                for (c, value) in out.iter_mut().enumerate() {
                                    let tc = (c0 + c).min(target_w - 1);
                                    let a = norm(plane.at(line, 2 * tc));
                                    let b = norm(plane.at(line, 2 * tc + 1));
                                    *value = (a + b) * 0.5;
                                }
                            }
                            // Narrower chroma into 4:4:4: each sample repeats.
                            (Chroma::Yuv444, 2..) => {
                                let line = plane.row(sr);
                                for (c, value) in out.iter_mut().enumerate() {
                                    let tc = (c0 + c).min(target_w - 1);
                                    *value = norm(plane.at(line, tc / *sx));
                                }
                            }
                            _ => plane.fill(sr, c0, target_w, out, norm),
                        }
                    }
                }
            }
            Source::Rgba { pixels } => {
                let inside = pixels.width.saturating_sub(x0).min(columns);
                let mut luma = [0f32; MAX_COLUMNS];
                let mut cb = [0f32; MAX_COLUMNS];
                let mut cr = [0f32; MAX_COLUMNS];
                for (r, ((ys, us), vs)) in lines {
                    let line = pixels.row(row * 16 + r);
                    let line = line.get(4 * x0..4 * (x0 + inside)).unwrap_or_default();
                    for (((pixel, y), b), r) in line
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(&mut luma)
                        .zip(&mut cb)
                        .zip(&mut cr)
                    {
                        (*y, *b, *r) = ycbcr(pixel);
                    }
                    // Columns past the picture repeat its last pixel.
                    if inside > 0 {
                        let last = (luma[inside - 1], cb[inside - 1], cr[inside - 1]);
                        luma[inside..].fill(last.0);
                        cb[inside..].fill(last.1);
                        cr[inside..].fill(last.2);
                    }
                    ys.copy_from_slice(&luma[..columns]);
                    match self.chroma {
                        Chroma::Yuv444 => {
                            us.copy_from_slice(&cb[..columns]);
                            vs.copy_from_slice(&cr[..columns]);
                        }
                        Chroma::Yuv422 => {
                            for (out, pair) in us.iter_mut().zip(cb.as_chunks::<2>().0) {
                                *out = (pair[0] + pair[1]) * 0.5;
                            }
                            for (out, pair) in vs.iter_mut().zip(cr.as_chunks::<2>().0) {
                                *out = (pair[0] + pair[1]) * 0.5;
                            }
                        }
                    }
                }
            }
        }
    }

    /// The slice's alpha, raster order: `columns` values for each row of
    /// the macroblock row that is inside the picture, at 8 or 16 bits.
    pub(crate) fn alpha(&self, row: usize, mb_x: usize, columns: usize, out: &mut Vec<u16>) {
        out.clear();
        let Some(sixteen) = self.alpha else {
            return;
        };
        let width = self.width as usize;
        let height = self.height as usize;
        let visible = height.saturating_sub(row * 16).min(16);
        let x0 = mb_x * 16;
        for r in 0..visible {
            let fr = row * 16 + r;
            match &self.source {
                Source::Planar {
                    alpha: Some((plane, bits)),
                    ..
                } => {
                    let line = plane.row(fr);
                    let max_in = (1u32 << bits) - 1;
                    let max_out = if sixteen { 65_535 } else { 255 };
                    out.extend((0..columns).map(|c| {
                        let raw = plane.at(line, (x0 + c).min(width - 1));
                        if max_in == max_out {
                            raw as u16
                        } else {
                            ((raw * max_out + max_in / 2) / max_in) as u16
                        }
                    }));
                }
                Source::Planar { alpha: None, .. } => out.extend((0..columns).map(|_| 0)),
                Source::Rgba { pixels } => {
                    let line = pixels.row(fr);
                    out.extend((0..columns).map(|c| {
                        let at = (x0 + c).min(width - 1) * 4 + 3;
                        line.get(at).map_or(0, |&byte| u16::from(byte))
                    }));
                }
            }
        }
    }
}

/// BT.709 limited-range Y'CbCr of one RGBA pixel, in the transform domain:
/// twice the 8-bit code less 128.
#[inline(always)]
fn ycbcr(pixel: &[u8; 4]) -> (f32, f32, f32) {
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
