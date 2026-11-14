//! Where the Y, U and V samples sit in a MediaCodec byte buffer.
//!
//! A decoder describes its output in one of two ways. Codec2 writes an
//! `image-data` blob, a `MediaImage2`: per plane an offset, the step between
//! two samples in a row, and the step between two rows. Older OMX decoders
//! only name a `color-format` with a `stride` and a `slice-height`, and the
//! layout is read the way FFmpeg reads it: planar 4:2:0 puts U then V after
//! `slice-height` rows of Y, each at half the stride; the semi-planar formats
//! put one plane of interleaved U and V there. Tiled and opaque formats are
//! refused.
//!
//! A crop rectangle moves the picture's top-left corner inside the buffer.
//! Ten-bit samples (`COLOR_FormatYUVP010`, or a `MediaImage2` with 16
//! allocated bits) sit in the top bits of a little-endian `u16`; they come
//! out as [`PixelFormat::Yuv420p10`], a little-endian `u16` per sample.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

use cutlass_decode::{CodecError, PixelFormat, picture_byte_len};

/// `COLOR_FormatYUV420Planar`.
const PLANAR: i32 = 0x13;
/// `COLOR_FormatYUV420SemiPlanar`.
const SEMI_PLANAR: i32 = 0x15;
/// `COLOR_FormatYUVP010`.
const P010: i32 = 0x36;
/// `COLOR_FormatYUV420Flexible`. Only `image-data` says where its planes are.
#[cfg(test)]
const FLEXIBLE: i32 = 0x7F42_0888;
/// `COLOR_QCOM_FormatYUV420SemiPlanar`.
const QCOM_SEMI_PLANAR: i32 = 0x7FA3_0C00;
/// `COLOR_QCOM_FormatYUV420SemiPlanar32m`.
const QCOM_SEMI_PLANAR_32M: i32 = 0x7FA3_0C04;
/// `COLOR_TI_FormatYUV420PackedSemiPlanar`.
const TI_PACKED_SEMI_PLANAR: i32 = 0x7F00_0100;

/// `MediaImage2::Type::MEDIA_IMAGE_TYPE_YUV`.
const MEDIA_IMAGE_YUV: u32 = 1;

/// Longest picture edge this crate hands to MediaCodec or reads back.
pub const MAX_EDGE: u32 = 8_192;

/// One plane: its first sample, the byte step to the next sample in a row,
/// and the byte step to the next row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub offset: usize,
    pub column: usize,
    pub row: usize,
}

/// A 4:2:0 picture inside a codec buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Planes {
    /// The picture, after the crop.
    pub width: u32,
    pub height: u32,
    /// 8, or 10 for samples in the top bits of a `u16`.
    pub depth: u8,
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
}

/// What an output format said about its buffers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Described {
    pub width: i32,
    pub height: i32,
    pub color_format: i32,
    pub stride: i32,
    pub slice_height: i32,
    /// `crop`: left, top, right and bottom, inclusive.
    pub crop: Option<[i32; 4]>,
    /// The `image-data` blob, when the codec wrote one.
    pub image: Option<Vec<u8>>,
}

impl Planes {
    /// The planes of one described buffer.
    pub fn of(described: &Described) -> Result<Self, CodecError> {
        let (left, top, width, height) = crop_of(described)?;
        let full = if let Some(image) = described.image.as_deref() {
            from_media_image(image)?
        } else {
            from_color_format(described)?
        };
        full.cropped(left, top, width, height)
    }

    /// The pixel format [`Self::read`] writes.
    pub fn pixel(&self) -> PixelFormat {
        if self.depth > 8 {
            PixelFormat::Yuv420p10
        } else {
            PixelFormat::Yuv420p
        }
    }

    /// Bytes the buffer must hold for the last sample of every plane.
    pub fn reach(&self) -> Result<usize, CodecError> {
        let sample = if self.depth > 8 { 2 } else { 1 };
        let (w, h) = (self.width as usize, self.height as usize);
        let mut reach = 0_usize;
        for (plane, columns, rows) in [
            (self.y, w, h),
            (self.u, w.div_ceil(2), h.div_ceil(2)),
            (self.v, w.div_ceil(2), h.div_ceil(2)),
        ] {
            let last = (rows - 1)
                .checked_mul(plane.row)
                .and_then(|at| at.checked_add((columns - 1).checked_mul(plane.column)?))
                .and_then(|at| at.checked_add(plane.offset))
                .and_then(|at| at.checked_add(sample))
                .ok_or(LIMIT)?;
            reach = reach.max(last);
        }
        Ok(reach)
    }

    /// The picture as tightly packed planar 4:2:0, eight or ten bits.
    pub fn read(&self, buffer: &[u8]) -> Result<Vec<u8>, CodecError> {
        if buffer.len() < self.reach()? {
            return Err(CodecError::Truncated {
                at_bit: (buffer.len() as u64).saturating_mul(8),
                what: "mediacodec buffer",
            });
        }
        let len = picture_byte_len(self.width, self.height, self.pixel())?;
        let mut out = Vec::with_capacity(len);
        let (w, h) = (self.width as usize, self.height as usize);
        for (plane, columns, rows) in [
            (self.y, w, h),
            (self.u, w / 2, h / 2),
            (self.v, w / 2, h / 2),
        ] {
            for line in 0..rows {
                let start = plane.offset + line * plane.row;
                if self.depth > 8 {
                    for column in 0..columns {
                        let at = start + column * plane.column;
                        let sample = u16::from_le_bytes([buffer[at], buffer[at + 1]]) >> 6;
                        out.extend_from_slice(&sample.to_le_bytes());
                    }
                } else if plane.column == 1 {
                    out.extend_from_slice(&buffer[start..start + columns]);
                } else {
                    out.extend(
                        buffer[start..]
                            .iter()
                            .step_by(plane.column)
                            .take(columns)
                            .copied(),
                    );
                }
            }
        }
        Ok(out)
    }

    /// The same planes moved to the crop's top-left corner.
    fn cropped(self, left: usize, top: usize, width: u32, height: u32) -> Result<Self, CodecError> {
        let shift = |plane: Plane, x: usize, y: usize| -> Result<Plane, CodecError> {
            let offset = y
                .checked_mul(plane.row)
                .and_then(|at| at.checked_add(x.checked_mul(plane.column)?))
                .and_then(|at| at.checked_add(plane.offset))
                .ok_or(LIMIT)?;
            Ok(Plane { offset, ..plane })
        };
        Ok(Self {
            width,
            height,
            depth: self.depth,
            y: shift(self.y, left, top)?,
            u: shift(self.u, left / 2, top / 2)?,
            v: shift(self.v, left / 2, top / 2)?,
        })
    }
}

const LIMIT: CodecError = CodecError::Limit {
    what: "picture bytes",
};

fn size(value: i32, what: &'static str) -> Result<u32, CodecError> {
    let value = u32::try_from(value).map_err(|_| CodecError::InvalidApi { what })?;
    if value == 0 {
        return Err(CodecError::InvalidApi { what });
    }
    if value > MAX_EDGE {
        return Err(CodecError::Limit { what });
    }
    Ok(value)
}

/// Left, top, width and height of the shown picture, even-sized.
fn crop_of(described: &Described) -> Result<(usize, usize, u32, u32), CodecError> {
    let full_width = size(described.width, "picture width")?;
    let full_height = size(described.height, "picture height")?;
    let Some([left, top, right, bottom]) = described.crop else {
        return Ok((0, 0, full_width & !1, full_height & !1));
    };
    let inside = left >= 0
        && top >= 0
        && right >= left
        && bottom >= top
        && i64::from(right) < i64::from(full_width)
        && i64::from(bottom) < i64::from(full_height);
    if !inside {
        return Err(CodecError::InvalidApi {
            what: "crop rectangle",
        });
    }
    let width = size(right - left + 1, "crop width")? & !1;
    let height = size(bottom - top + 1, "crop height")? & !1;
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "crop rectangle",
        });
    }
    // An odd corner would split a chroma sample.
    let (left, top) = (left as usize & !1, top as usize & !1);
    Ok((left, top, width, height))
}

/// The planes a `MediaImage2` names, before the crop.
fn from_media_image(image: &[u8]) -> Result<Planes, CodecError> {
    let word = |index: usize| -> Result<u32, CodecError> {
        let at = index * 4;
        image
            .get(at..at + 4)
            .map(|bytes| u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .ok_or(CodecError::Truncated {
                at_bit: (image.len() as u64).saturating_mul(8),
                what: "image-data",
            })
    };
    if word(0)? != MEDIA_IMAGE_YUV || word(1)? != 3 {
        return Err(CodecError::Unsupported {
            what: "image-data layout",
        });
    }
    let (depth, allocated) = (word(4)?, word(5)?);
    let depth = match (depth, allocated) {
        (8, 8) => 8,
        (10, 16) => 10,
        _ => {
            return Err(CodecError::Unsupported {
                what: "image-data depth",
            });
        }
    };
    let plane = |index: usize| -> Result<(Plane, u32, u32), CodecError> {
        let base = 6 + index * 5;
        let (offset, column, row) = (word(base)?, word(base + 1)?, word(base + 2)?);
        let (across, down) = (word(base + 3)?, word(base + 4)?);
        let plane = Plane {
            offset: offset as usize,
            column: column as usize,
            row: row as usize,
        };
        Ok((plane, across, down))
    };
    let (y, y_across, y_down) = plane(0)?;
    let (u, u_across, u_down) = plane(1)?;
    let (v, v_across, v_down) = plane(2)?;
    let sample = if depth > 8 { 2 } else { 1 };
    let shaped = (y_across, y_down, u_across, u_down, v_across, v_down) == (1, 1, 2, 2, 2, 2)
        && [y, u, v]
            .iter()
            .all(|plane| plane.column >= sample && plane.row > 0);
    if !shaped {
        return Err(CodecError::Unsupported {
            what: "image-data sampling",
        });
    }
    Ok(Planes {
        width: 0,
        height: 0,
        depth,
        y,
        u,
        v,
    })
}

/// The planes a color format, stride and slice height name, as FFmpeg
/// reads them, before the crop.
fn from_color_format(described: &Described) -> Result<Planes, CodecError> {
    let width = size(described.width, "picture width")? as usize;
    let height = size(described.height, "picture height")? as usize;
    let stride = match usize::try_from(described.stride) {
        Ok(stride) if stride >= width => stride,
        _ => width,
    };
    let rows = match usize::try_from(described.slice_height) {
        Ok(rows) if rows >= height => rows,
        _ => height,
    };
    let luma = stride.checked_mul(rows).ok_or(LIMIT)?;
    let y = Plane {
        offset: 0,
        column: 1,
        row: stride,
    };
    let planes = |depth, u, v| Planes {
        width: 0,
        height: 0,
        depth,
        y,
        u,
        v,
    };
    match described.color_format {
        PLANAR => {
            let half = stride.div_ceil(2);
            let u = Plane {
                offset: luma,
                column: 1,
                row: half,
            };
            let v_at = half
                .checked_mul(rows.div_ceil(2))
                .and_then(|size| size.checked_add(luma))
                .ok_or(LIMIT)?;
            let v = Plane {
                offset: v_at,
                column: 1,
                row: half,
            };
            Ok(planes(8, u, v))
        }
        SEMI_PLANAR | QCOM_SEMI_PLANAR | QCOM_SEMI_PLANAR_32M | TI_PACKED_SEMI_PLANAR => {
            let u = Plane {
                offset: luma,
                column: 2,
                row: stride,
            };
            Ok(planes(
                8,
                u,
                Plane {
                    offset: luma + 1,
                    ..u
                },
            ))
        }
        P010 => {
            // The stride is in bytes; each sample is two.
            let y = Plane {
                offset: 0,
                column: 2,
                row: stride.max(width * 2),
            };
            let luma = y.row.checked_mul(rows).ok_or(LIMIT)?;
            let u = Plane {
                offset: luma,
                column: 4,
                row: y.row,
            };
            Ok(Planes {
                width: 0,
                height: 0,
                depth: 10,
                y,
                u,
                v: Plane {
                    offset: luma + 2,
                    ..u
                },
            })
        }
        _ => Err(CodecError::Unsupported {
            what: "mediacodec color format",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn described(color_format: i32, width: i32, height: i32, stride: i32, rows: i32) -> Described {
        Described {
            width,
            height,
            color_format,
            stride,
            slice_height: rows,
            ..Described::default()
        }
    }

    /// A 4 by 2 picture whose samples count up: Y 0 to 7, U 8 and 9, V 10 and 11.
    const PACKED: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    #[test]
    fn planar_with_a_wide_stride_and_tall_slice_reads_packed() {
        let planes = Planes::of(&described(PLANAR, 4, 2, 8, 4)).expect("planes");
        let mut buffer = vec![0xEE_u8; 8 * 4 + 4 * 2 * 2];
        buffer[0..4].copy_from_slice(&[0, 1, 2, 3]);
        buffer[8..12].copy_from_slice(&[4, 5, 6, 7]);
        buffer[32..34].copy_from_slice(&[8, 9]);
        buffer[40..42].copy_from_slice(&[10, 11]);
        assert_eq!(planes.read(&buffer).expect("read"), PACKED);
        assert_eq!(planes.pixel(), PixelFormat::Yuv420p);
    }

    #[test]
    fn semi_planar_interleaves_u_and_v() {
        for format in [SEMI_PLANAR, QCOM_SEMI_PLANAR, TI_PACKED_SEMI_PLANAR] {
            let planes = Planes::of(&described(format, 4, 2, 4, 2)).expect("planes");
            let buffer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11];
            assert_eq!(planes.read(&buffer).expect("read"), PACKED);
        }
    }

    #[test]
    fn a_missing_stride_or_slice_height_is_the_picture() {
        let planes = Planes::of(&described(SEMI_PLANAR, 4, 2, 0, 0)).expect("planes");
        assert_eq!(planes.y.row, 4);
        assert_eq!(planes.u.offset, 8);
    }

    #[test]
    fn media_image_describes_nv21() {
        // Y, then V and U interleaved: U starts one byte after V.
        let words: [u32; 21] = [
            1, 3, 4, 2, 8, 8, 0, 1, 4, 1, 1, 9, 2, 4, 2, 2, 8, 2, 4, 2, 2,
        ];
        let image: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        let mut info = described(FLEXIBLE, 4, 2, 0, 0);
        info.image = Some(image);
        let planes = Planes::of(&info).expect("planes");
        let buffer = [0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9];
        assert_eq!(planes.read(&buffer).expect("read"), PACKED);
    }

    #[test]
    fn a_crop_moves_the_corner_and_keeps_even_edges() {
        let mut info = described(PLANAR, 8, 4, 8, 4);
        info.crop = Some([2, 2, 6, 3]);
        let planes = Planes::of(&info).expect("planes");
        assert_eq!((planes.width, planes.height), (4, 2));
        assert_eq!(planes.y.offset, 2 * 8 + 2);
        assert_eq!(planes.u.offset, 32 + 4 + 1);
        info.crop = Some([0, 0, 8, 3]);
        assert!(Planes::of(&info).is_err(), "right edge past the picture");
        info.crop = Some([3, 0, 2, 3]);
        assert!(Planes::of(&info).is_err(), "right before left");
    }

    #[test]
    fn ten_bit_samples_come_out_of_the_top_bits() {
        let planes = Planes::of(&described(P010, 2, 2, 4, 2)).expect("planes");
        assert_eq!(planes.pixel(), PixelFormat::Yuv420p10);
        let sample = |value: u16| (value << 6).to_le_bytes();
        let mut buffer = Vec::new();
        for value in [1_u16, 2, 3, 4, 500, 1023] {
            buffer.extend_from_slice(&sample(value));
        }
        let out = planes.read(&buffer).expect("read");
        let values: Vec<u16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        assert_eq!(values, [1, 2, 3, 4, 500, 1023]);
    }

    #[test]
    fn opaque_tiled_and_hostile_layouts_are_refused() {
        for format in [0x7F00_0789, 0x7FA3_0C03, FLEXIBLE] {
            assert!(matches!(
                Planes::of(&described(format, 4, 2, 4, 2)),
                Err(CodecError::Unsupported { .. })
            ));
        }
        assert!(Planes::of(&described(PLANAR, 0, 2, 0, 0)).is_err());
        assert!(matches!(
            Planes::of(&described(PLANAR, 9_000, 2, 0, 0)),
            Err(CodecError::Limit { .. })
        ));
        let mut info = described(FLEXIBLE, 4, 2, 0, 0);
        info.image = Some(vec![1, 0, 0, 0]);
        assert!(matches!(
            Planes::of(&info),
            Err(CodecError::Truncated { .. })
        ));
    }

    #[test]
    fn a_short_buffer_is_truncated() {
        let planes = Planes::of(&described(PLANAR, 4, 2, 4, 2)).expect("planes");
        assert!(matches!(
            planes.read(&[0; 11]),
            Err(CodecError::Truncated { .. })
        ));
        assert_eq!(planes.read(&PACKED).expect("exact"), PACKED);
    }
}
