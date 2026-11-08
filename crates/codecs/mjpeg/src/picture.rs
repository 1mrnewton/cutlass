//! One packet to one picture.
//!
//! A packet is one JPEG image, or two when the track is interlaced: each
//! image is one field, and the frame is the two woven row by row.
//!
//! As in FFmpeg, the first packet after open or reset decides whether the
//! stream is interlaced, and a later size change does not make it so. That
//! packet is a field pair when its first image is field-shaped (the
//! container's width, or an unknown one, and a height `h` with
//! `4h < 3H` for the container height `H`) and a second image follows it in
//! the packet. A field-shaped image alone is a whole frame unless the track
//! says it is a field: a `fiel` atom naming two fields, or an APP0 `AVI1`
//! polarity of 1 or 2. That case is refused as one field per packet.
//!
//! Once a stream is interlaced, a packet whose first image has the first
//! field's size is a pair, and any other size is a frame. Once it is
//! progressive, every packet is a frame, so a stream may drop to half the
//! container height. A woven frame is `min(H, 2h)` rows. Bytes after the
//! last image are ignored.

use cutlass_decode::{
    CodecError, ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Plane, Transfer,
    VideoFrame,
};
use cutlass_jpeg::{ColorModel, Image, Tables, decode_image};

use crate::config::{Config, FieldOrder, Flavour};

/// Largest packet accepted.
pub const MAX_PACKET: usize = 64 << 20;

/// One decoded frame: a single image, or two fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    /// The stored order of the two fields, `None` for a progressive frame.
    pub fields: Option<FieldOrder>,
    /// The frame, or the top field.
    top: Image,
    bottom: Option<Image>,
}

/// Packet framing and the tables carried from packet to packet.
pub(crate) struct Unpacker {
    config: Config,
    tables: Tables,
    /// Container size; zero when unknown.
    width: u32,
    height: u32,
    /// Set by the first packet that decodes after open or reset.
    layout: Option<Layout>,
}

/// What the first decoded packet showed about the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Frames,
    /// Two fields per packet, each this size.
    Fields {
        width: u32,
        height: u32,
    },
}

impl Unpacker {
    pub(crate) fn new(config: Config, width: u32, height: u32) -> Result<Self, CodecError> {
        if config.flavour == Some(Flavour::FormatB) {
            return Err(format_b());
        }
        Ok(Self {
            tables: config.tables.clone(),
            config,
            width,
            height,
            layout: None,
        })
    }

    /// Back to the tables the track opened with, and to an undecided
    /// layout.
    pub(crate) fn reset(&mut self) {
        self.tables = self.config.tables.clone();
        self.layout = None;
    }

    /// Decode one packet. Tables the packet defines are kept only when the
    /// whole packet decodes.
    pub(crate) fn unpack(&mut self, data: &[u8]) -> Result<Picture, CodecError> {
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if data.len() > MAX_PACKET {
            return Err(CodecError::Limit {
                what: "motion jpeg packet",
            });
        }
        if data.len() >= 8 && data[..4] == [0; 4] && &data[4..8] == b"mjpg" {
            return Err(format_b());
        }
        let mut tables = self.tables.clone();
        let first = decode_image(data, &mut tables)?;
        let (w, h) = (first.image.width, first.image.height);
        let pair = match self.layout {
            Some(Layout::Frames) => false,
            Some(Layout::Fields { width, height }) => (w, h) == (width, height),
            None => self.field_shaped(&first.image),
        };
        let rest = data.get(first.len..).unwrap_or(&[]);
        let second = rest.windows(2).position(|pair| pair == [0xff, 0xd8]);
        let (picture, layout) = match (pair, second) {
            (true, Some(start)) => {
                let second = decode_image(&rest[start..], &mut tables)?.image;
                if !same_layout(&first.image, &second) {
                    return Err(CodecError::Malformed {
                        at_bit: bit_offset(first.len + start),
                        what: "second field",
                    });
                }
                let picture = self.weave_pair(first.image, second);
                (
                    picture,
                    Layout::Fields {
                        width: w,
                        height: h,
                    },
                )
            }
            (true, None) if self.layout.is_some() || self.signals_fields(&first.image) => {
                return Err(CodecError::Unsupported {
                    what: "one field per packet",
                });
            }
            _ => (
                Picture {
                    width: w,
                    height: h,
                    fields: None,
                    top: first.image,
                    bottom: None,
                },
                self.layout.unwrap_or(Layout::Frames),
            ),
        };
        self.tables = tables;
        self.layout = Some(layout);
        Ok(picture)
    }

    fn weave_pair(&self, first: Image, second: Image) -> Picture {
        let order = self
            .config
            .field_order
            .or(match first.avid_standard {
                Some(1) => Some(FieldOrder::BottomFirst),
                Some(2) => Some(FieldOrder::TopFirst),
                _ => None,
            })
            .unwrap_or(FieldOrder::TopFirst);
        let (top, bottom) = match order {
            FieldOrder::TopFirst => (first, second),
            FieldOrder::BottomFirst => (second, first),
        };
        Picture {
            width: top.width,
            height: self.height.min(top.height.saturating_mul(2)),
            fields: Some(order),
            top,
            bottom: Some(bottom),
        }
    }

    /// FFmpeg's test: an image under three quarters of the container
    /// height is a field. The width must match too, when it is known.
    fn field_shaped(&self, image: &Image) -> bool {
        self.height != 0
            && image.height > 0
            && u64::from(image.height) * 4 < u64::from(self.height) * 3
            && (self.width == 0 || self.width == image.width)
    }

    /// The track or the image names it a field: `fiel` with two fields,
    /// or an `AVI1` polarity of 1 or 2.
    fn signals_fields(&self, image: &Image) -> bool {
        self.config.fields == Some(2) || matches!(image.avi1_polarity, Some(1 | 2))
    }
}

fn same_layout(a: &Image, b: &Image) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.color == b.color
        && a.studio_range == b.studio_range
        && a.components.len() == b.components.len()
        && a.components
            .iter()
            .zip(&b.components)
            .all(|(x, y)| (x.h, x.v) == (y.h, y.v))
}

fn format_b() -> CodecError {
    CodecError::Unsupported {
        what: "motion jpeg format b",
    }
}

fn bit_offset(bytes: usize) -> u64 {
    u64::try_from(bytes).unwrap_or(u64::MAX).saturating_mul(8)
}

fn limit() -> CodecError {
    CodecError::Limit {
        what: "motion jpeg frame",
    }
}

impl Picture {
    /// The coded images: the frame, or the top field then the bottom.
    pub fn images(&self) -> impl Iterator<Item = &Image> {
        std::iter::once(&self.top).chain(self.bottom.as_ref())
    }

    /// Straight RGBA, alpha 255, `width * height * 4` bytes in `out`.
    ///
    /// JFIF YCbCr uses the T.871 full-range conversion and libjpeg's chroma
    /// filters, so a progressive frame equals `djpeg -dct int`. A frame
    /// marked `CS=ITU601` is converted as studio-range BT.601.
    pub fn to_rgba(&self, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let Some(bottom) = &self.bottom else {
            return image_rgba(&self.top, out);
        };
        let mut top_rgba = Vec::new();
        let mut bottom_rgba = Vec::new();
        image_rgba(&self.top, &mut top_rgba)?;
        image_rgba(bottom, &mut bottom_rgba)?;
        let row = usize::try_from(self.width).map_err(|_| limit())? * 4;
        let rows = usize::try_from(self.height).map_err(|_| limit())?;
        weave(&top_rgba, &bottom_rgba, row, rows, out)
    }

    /// Planar samples at their coded sampling.
    ///
    /// Grey is `Gray8`. YCbCr is `Yuv444p`, `Yuv422p`, `Yuv440p`, or
    /// `Yuv420p` when the chroma components share one of those ratios to
    /// luma, and `Yuv444p` after upsampling otherwise. RGB is `Gbrp`.
    /// Range is full unless the frame is marked `CS=ITU601`; the matrix is
    /// BT.601.
    pub fn into_frame(self) -> Result<VideoFrame, CodecError> {
        let studio = self.top.studio_range;
        let (format, top) = planes(self.top)?;
        let planes = match self.bottom {
            None => top,
            Some(bottom) => {
                let (_, bottom) = planes(bottom)?;
                let mut woven = Vec::with_capacity(top.len());
                for (index, (a, b)) in top.iter().zip(&bottom).enumerate() {
                    let (width, height) = plane_size(format, self.width, self.height, index);
                    let row = usize::try_from(width).map_err(|_| limit())?;
                    let rows = usize::try_from(height).map_err(|_| limit())?;
                    let mut out = Vec::new();
                    weave(&a.samples, &b.samples, row, rows, &mut out)?;
                    woven.push(Plane::tight(width, height, out)?);
                }
                woven
            }
        };
        let mut planes = planes.into_iter();
        let (Some(y), u, v) = (planes.next(), planes.next(), planes.next()) else {
            return Err(limit());
        };
        let mut frame = match (u, v) {
            (Some(u), Some(v)) => VideoFrame::planar(format, self.width, self.height, y, u, v)?,
            _ => VideoFrame {
                format,
                width: self.width,
                height: self.height,
                y,
                u: Plane::tight(0, 0, Vec::new())?,
                v: Plane::tight(0, 0, Vec::new())?,
                range: ColorRange::Full,
                primaries: ColorPrimaries::Unspecified,
                transfer: Transfer::Unspecified,
                matrix: MatrixCoefficients::Unspecified,
                render_width: self.width,
                render_height: self.height,
                alpha: None,
            },
        };
        frame.range = if studio && format != PixelFormat::Gbrp {
            ColorRange::Limited
        } else {
            ColorRange::Full
        };
        frame.matrix = match format {
            PixelFormat::Gbrp => MatrixCoefficients::Rgb,
            PixelFormat::Gray8 => MatrixCoefficients::Unspecified,
            _ => MatrixCoefficients::Bt601,
        };
        Ok(frame)
    }
}

/// Plane `index` of a `format` picture `width` × `height`.
fn plane_size(format: PixelFormat, width: u32, height: u32, index: usize) -> (u32, u32) {
    if index == 0 {
        (width, height)
    } else {
        format.chroma_wh(width, height).unwrap_or((width, height))
    }
}

fn image_rgba(image: &Image, out: &mut Vec<u8>) -> Result<(), CodecError> {
    if image.studio_range && image.color == ColorModel::YCbCr {
        studio_rgba(image, out)
    } else {
        image.to_rgba(out)
    }
}

/// Studio-range BT.601: luma 16..=235 and chroma 16..=240, in 16.16 fixed
/// point.
fn studio_rgba(image: &Image, out: &mut Vec<u8>) -> Result<(), CodecError> {
    let planes = image.full_planes()?;
    let [y, cb, cr] = planes.as_slice() else {
        return Err(limit());
    };
    out.clear();
    out.try_reserve_exact(y.len() * 4).map_err(|_| limit())?;
    for ((&y, &cb), &cr) in y.iter().zip(cb).zip(cr) {
        let luma = 76_309 * (i32::from(y) - 16) + 32_768;
        let (b, r) = (i32::from(cb) - 128, i32::from(cr) - 128);
        let clamp = |value: i32| u8::try_from((value >> 16).clamp(0, 255)).unwrap_or(0);
        out.extend_from_slice(&[
            clamp(luma + 104_597 * r),
            clamp(luma - 25_675 * b - 53_279 * r),
            clamp(luma + 132_201 * b),
            u8::MAX,
        ]);
    }
    Ok(())
}

/// Even rows from `top`, odd rows from `bottom`, `rows` rows of `row` bytes.
fn weave(
    top: &[u8],
    bottom: &[u8],
    row: usize,
    rows: usize,
    out: &mut Vec<u8>,
) -> Result<(), CodecError> {
    out.clear();
    out.try_reserve_exact(row * rows).map_err(|_| limit())?;
    for index in 0..rows {
        let field = if index % 2 == 0 { top } else { bottom };
        let at = (index / 2) * row;
        let Some(line) = field.get(at..at + row) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "field height",
            });
        };
        out.extend_from_slice(line);
    }
    Ok(())
}

/// The image's planes and the format they make.
fn planes(image: Image) -> Result<(PixelFormat, Vec<Plane>), CodecError> {
    let (width, height) = (image.width, image.height);
    let format = match image.color {
        ColorModel::Gray => PixelFormat::Gray8,
        ColorModel::Rgb => PixelFormat::Gbrp,
        ColorModel::YCbCr => subsampled_format(&image).unwrap_or(PixelFormat::Yuv444p),
    };
    let planes = match format {
        PixelFormat::Gray8 => {
            let Some(luma) = image.components.into_iter().next() else {
                return Err(limit());
            };
            vec![Plane::tight(luma.width, luma.height, luma.samples)?]
        }
        PixelFormat::Gbrp => {
            let full = image.full_planes()?;
            let [r, g, b] = <[Vec<u8>; 3]>::try_from(full).map_err(|_| limit())?;
            vec![
                Plane::tight(width, height, g)?,
                Plane::tight(width, height, b)?,
                Plane::tight(width, height, r)?,
            ]
        }
        _ if image.components.iter().enumerate().all(|(index, comp)| {
            (comp.width, comp.height) == plane_size(format, width, height, index)
        }) =>
        {
            image
                .components
                .into_iter()
                .map(|comp| Plane::tight(comp.width, comp.height, comp.samples))
                .collect::<Result<_, _>>()?
        }
        _ => image
            .full_planes()?
            .into_iter()
            .map(|samples| Plane::tight(width, height, samples))
            .collect::<Result<_, _>>()?,
    };
    Ok((format, planes))
}

/// The 8-bit planar format whose chroma ratio both chroma components have,
/// with luma at the full size.
fn subsampled_format(image: &Image) -> Option<PixelFormat> {
    let [luma, cb, cr] = image.components.as_slice() else {
        return None;
    };
    if (luma.h, luma.v) != (image.max_h, image.max_v) || (cb.h, cb.v) != (cr.h, cr.v) {
        return None;
    }
    let ratio = |max: u8, factor: u8| max.is_multiple_of(factor).then(|| max / factor);
    match (ratio(image.max_h, cb.h)?, ratio(image.max_v, cb.v)?) {
        (1, 1) => Some(PixelFormat::Yuv444p),
        (2, 1) => Some(PixelFormat::Yuv422p),
        (1, 2) => Some(PixelFormat::Yuv440p),
        (2, 2) => Some(PixelFormat::Yuv420p),
        _ => None,
    }
}
